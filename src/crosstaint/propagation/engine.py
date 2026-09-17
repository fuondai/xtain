from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from crosstaint.config import Config
from crosstaint.propagation.decay import DecayScheduler, MEVResistantDecay
from crosstaint.propagation.weight import EdgeWeightLearner
from crosstaint.types import (
    AddressTag,
    EdgeType,
    IREdge,
    IRNode,
    NodeType,
    PropagationResult,
    PropagationState,
    SuspectEntry,
    TaintOperator,
)


@dataclass
class _FrontierEntry:
    score: float
    node_id: str
    remaining_value: int
    hop_distance: int

    def __lt__(self, other: "_FrontierEntry") -> bool:
        return self.score > other.score


@dataclass
class _VisitedEdge:
    edge_id: str
    source: str
    target: str

    def __hash__(self) -> int:
        return hash((self.edge_id, self.source, self.target))


class PropagationEngine:
    KNOWN_MIXER_PATTERNS: frozenset[str] = frozenset({
        "tornado", "tornadocash", "mixer", "whirlpool", "cyclone",
        "gemini", "binance-mixer",
    })

    def __init__(
        self,
        rho: float = 0.81,
        threshold: float = 0.04,
        time_window_days: int = 30,
        seed: int = 42,
        skip_hop_recovery: bool = True,
        skip_gate_multiplier: float = 2.0,
        mev_decay: MEVResistantDecay | None = None,
        adaptive_fmax_cap: float = 0.30,
        intent_margin_bound: float = 0.08,
        cert_ood_g_max: int = 4,
        value_conservation_enabled: bool = True,
    ) -> None:
        if not (0.0 < rho <= 1.0):
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.rho = rho
        self.threshold = threshold
        self.time_window_days = time_window_days
        self.seed = seed
        self.skip_hop_recovery = skip_hop_recovery
        self.skip_gate = min(1.0, threshold * skip_gate_multiplier)
        self._mev_decay = mev_decay
        self.adaptive_fmax_cap = adaptive_fmax_cap
        self.intent_margin_bound = intent_margin_bound
        self.cert_ood_g_max = cert_ood_g_max
        self.value_conservation_enabled = value_conservation_enabled

        self.decay_scheduler = DecayScheduler(rho)
        self.edge_learner = EdgeWeightLearner()

    @classmethod
    def from_config(
        cls,
        rho: float = 0.81,
        threshold: float = 0.04,
        **kwargs: Any,
    ) -> "PropagationEngine":
        """Build an engine, enabling MEV-resistant decay from the propagation config."""
        from crosstaint.propagation.decay import calibrate_gamma

        cfg = Config.load().propagation if Config._instance is not None else {}
        mev_cfg = cfg.get("mev_resistant_decay", {}) if isinstance(cfg, dict) else {}
        mev_decay = None
        if mev_cfg.get("enabled"):
            if mev_cfg.get("gamma") is not None:
                gamma = float(mev_cfg["gamma"])
            else:
                gamma = calibrate_gamma(
                    alpha_fn=float(mev_cfg.get("alpha_fn", 0.01)),
                    sigma_f=float(mev_cfg.get("sigma_f", 0.018)),
                    z_alpha=float(mev_cfg.get("z_alpha", 2.33)),
                )
            mev_decay = MEVResistantDecay(
                rho0=float(mev_cfg.get("rho0", rho)),
                gamma=gamma,
                f0=float(mev_cfg.get("f0", 0.01)),
            )
        return cls(rho=rho, threshold=threshold, mev_decay=mev_decay, **kwargs)

    def _edge_rho(self, edge: IREdge, remaining_value: int) -> float:
        if self._mev_decay is None:
            return self.rho
        if remaining_value <= 0:
            return self._mev_decay.rho0
        observed_fmax = max(0.0, (remaining_value - edge.value) / remaining_value)
        operator_f0 = self._effective_fee_bound_pct(edge)
        return self._mev_decay.rho(observed_fmax, f0=operator_f0)

    def _effective_fee_bound_pct(self, edge: IREdge) -> float:
        fee_bound_pct = max(0.0, edge.fee_bound_pct)
        if edge.operator == TaintOperator.INTENT_FILL:
            # Intent fills are evaluated from the realized destination-chain
            # settlement when available; otherwise use the calibrated solver
            # margin envelope and cap it to avoid unbounded catch-all matches.
            return min(self.adaptive_fmax_cap, max(fee_bound_pct, self.intent_margin_bound))
        return fee_bound_pct

    def _is_mixer(self, node: IRNode) -> bool:
        address_lower = node.address.lower()
        return (
            node.node_type == NodeType.MIXER
            or any(p in address_lower for p in self.KNOWN_MIXER_PATTERNS)
            or AddressTag.MIXER_DEPOSIT in node.tags
        )

    def _init_frontier(
        self,
        origin: str,
        origin_chain: str,
        origin_value: int,
    ) -> PropagationState:
        entry = _FrontierEntry(
            score=1.0,
            node_id=origin,
            remaining_value=origin_value,
            hop_distance=0,
        )
        return PropagationState(
            frontier=[entry],
            scores={origin: 1.0},
            predecessors={},
            hop_distances={origin: 0},
            terminated=set(),
            visited_edges=set(),
            values={origin: origin_value},
            observations=[],
        )

    def _value_conserved(self, edge: IREdge, remaining_value: int) -> bool:
        if remaining_value <= 0 or edge.value <= 0:
            return False
        fee_bound_pct = self._effective_fee_bound_pct(edge)
        fee_bound = math.ceil(remaining_value * fee_bound_pct)
        lower_bound = max(0, remaining_value - fee_bound)
        return lower_bound <= edge.value <= remaining_value

    def _value_conserved_skip(self, edge: IREdge, remaining_value: int) -> bool:
        if remaining_value <= 0 or edge.value <= 0:
            return False
        fee_bound_pct = self._effective_fee_bound_pct(edge)
        composed_bound = math.ceil(remaining_value * min(1.0, 2.0 * fee_bound_pct))
        lower_bound = max(0, remaining_value - composed_bound)
        return lower_bound <= edge.value <= remaining_value

    def _matcher_confidence(self, edge: IREdge, matcher: Any | None) -> float:
        if edge.edge_type != EdgeType.CROSS_CHAIN_BRIDGE:
            return 1.0
        if matcher is None:
            return 1.0
        if hasattr(matcher, "score_edge"):
            score = matcher.score_edge(edge)
        elif hasattr(matcher, "match") and edge.source_event and edge.target_event:
            score = matcher.match(edge.source_event, edge.target_event)
        elif callable(matcher):
            score = matcher(edge)
        else:
            score = 1.0
        return float(min(1.0, max(0.0, score)))

    def _resolver_multiplier(
        self,
        source: str,
        target: str,
        graph: dict[str, IRNode],
        resolver: Any | None,
        matcher_confidence: float,
    ) -> float:
        target_node = graph.get(target)
        if target_node is None:
            return 1.0
        if target_node.degree_at_first_seen <= 2:
            return matcher_confidence
        if resolver is None or not hasattr(resolver, "resolve"):
            return 1.0
        output = resolver.resolve(source, target)
        similarity = getattr(output, "similarity", 1.0)
        return float(min(1.0, max(0.0, similarity)))

    def _step(
        self,
        state: PropagationState,
        graph: dict[str, IRNode],
        outgoing_by_source: dict[str, list[IREdge]],
        matcher: Any | None,
        resolver: Any | None,
    ) -> bool:
        if not state.frontier:
            return False

        entry: _FrontierEntry = heapq.heappop(state.frontier)
        score = entry.score
        node_id = entry.node_id
        remaining_value = entry.remaining_value
        hop_distance = entry.hop_distance

        if score < state.scores.get(node_id, 0.0):
            return bool(state.frontier)

        if score < self.threshold:
            return bool(state.frontier)

        if node_id in state.terminated:
            return bool(state.frontier)

        node = graph.get(node_id)
        if node is not None and self._is_mixer(node):
            state.terminated.add(node_id)
            return bool(state.frontier)

        has_cross_chain_exit = False
        matched_cross_chain = False
        for edge in outgoing_by_source.get(node_id, []):
            if edge.edge_type == EdgeType.CROSS_CHAIN_BRIDGE:
                has_cross_chain_exit = True
            edge_key = _VisitedEdge(edge.edge_id, edge.source_node, edge.target_node)
            if edge_key in state.visited_edges:
                continue
            state.visited_edges.add(edge_key)

            if self.value_conservation_enabled and not self._value_conserved(edge, remaining_value):
                continue

            edge_weight = self.edge_learner.get_weight(edge)
            matcher_confidence = self._matcher_confidence(edge, matcher)
            resolver_weight = self._resolver_multiplier(
                edge.source_node,
                edge.target_node,
                graph,
                resolver,
                matcher_confidence,
            )
            hop = hop_distance + 1
            edge_rho = self._edge_rho(edge, remaining_value)
            decayed_score = score * edge_rho * edge_weight * matcher_confidence * resolver_weight
            new_value = int(edge.value)

            if decayed_score >= self.threshold:
                target = edge.target_node
                state.observations.append((matcher_confidence, edge.block_number))
                if edge.edge_type == EdgeType.CROSS_CHAIN_BRIDGE:
                    matched_cross_chain = True
                if (
                    target not in state.scores
                    or decayed_score > state.scores[target]
                ):
                    state.scores[target] = decayed_score
                    state.hop_distances[target] = hop
                    state.values[target] = new_value
                    state.predecessors[target] = node_id
                    heapq.heappush(
                        state.frontier,
                        _FrontierEntry(
                            score=decayed_score,
                            node_id=target,
                            remaining_value=new_value,
                            hop_distance=hop,
                        ),
                    )

        if (
            self.skip_hop_recovery
            and has_cross_chain_exit
            and not matched_cross_chain
        ):
            self._skip_hop(
                state,
                graph,
                outgoing_by_source,
                matcher,
                resolver,
                node_id,
                score,
                remaining_value,
                hop_distance,
            )

        return bool(state.frontier)

    def _skip_hop(
        self,
        state: PropagationState,
        graph: dict[str, IRNode],
        outgoing_by_source: dict[str, list[IREdge]],
        matcher: Any | None,
        resolver: Any | None,
        node_id: str,
        score: float,
        remaining_value: int,
        hop_distance: int,
    ) -> None:
        # Bounded single-hop reconnection: when no direct cross-chain edge cleared
        # the gate, search two-hop continuations a -> x -> b' that satisfy the
        # composed value envelope and clear the stricter skip gate. The recovery
        # never chains skips, so it cannot manufacture long speculative trails.
        for first_edge in outgoing_by_source.get(node_id, []):
            intermediate = first_edge.target_node
            for second_edge in outgoing_by_source.get(intermediate, []):
                if second_edge.edge_type != EdgeType.CROSS_CHAIN_BRIDGE:
                    continue
                target = second_edge.target_node
                if target == node_id:
                    continue
                if self.value_conservation_enabled and not self._value_conserved_skip(second_edge, remaining_value):
                    continue
                edge_key = _VisitedEdge(
                    f"skip:{first_edge.edge_id}:{second_edge.edge_id}",
                    node_id,
                    target,
                )
                if edge_key in state.visited_edges:
                    continue
                state.visited_edges.add(edge_key)

                edge_weight = self.edge_learner.get_weight(second_edge)
                matcher_confidence = self._matcher_confidence(second_edge, matcher)
                resolver_weight = self._resolver_multiplier(
                    node_id,
                    target,
                    graph,
                    resolver,
                    matcher_confidence,
                )
                hop = hop_distance + 2
                edge_rho = self._edge_rho(second_edge, remaining_value)
                decayed_score = (
                    score
                    * (edge_rho ** 2)
                    * edge_weight
                    * matcher_confidence
                    * resolver_weight
                )
                new_value = int(second_edge.value)

                if decayed_score > self.skip_gate and (
                    target not in state.scores
                    or decayed_score > state.scores[target]
                ):
                    state.observations.append((matcher_confidence, second_edge.block_number))
                    state.scores[target] = decayed_score
                    state.hop_distances[target] = hop
                    state.values[target] = new_value
                    state.predecessors[target] = node_id
                    heapq.heappush(
                        state.frontier,
                        _FrontierEntry(
                            score=decayed_score,
                            node_id=target,
                            remaining_value=new_value,
                            hop_distance=hop,
                        ),
                    )

    def propagate(
        self,
        origin: str,
        origin_chain: str,
        origin_value: int,
        graph: dict[str, IRNode],
        edges: list[IREdge],
        case_id: str,
        matcher: Any | None = None,
        resolver: Any | None = None,
    ) -> PropagationResult:
        start_time = time.monotonic()

        state = self._init_frontier(origin, origin_chain, origin_value)
        path_count = 0
        outgoing_by_source: dict[str, list[IREdge]] = defaultdict(list)
        for edge in edges:
            outgoing_by_source[edge.source_node].append(edge)

        while self._step(state, graph, outgoing_by_source, matcher, resolver):
            path_count += 1

        runtime_ms = (time.monotonic() - start_time) * 1000

        suspect_entries: list[SuspectEntry] = []
        for node_id, score in state.scores.items():
            node = graph.get(node_id)
            if node is None:
                continue
            if node_id == origin:
                continue

            pred = state.predecessors.get(node_id, "")
            hop = state.hop_distances.get(node_id, 0)
            flags: set[str] = set()
            if node_id in state.terminated:
                flags.add("mixer_terminated")

            suspect_entries.append(
                SuspectEntry(
                    address=node.address,
                    chain=node.chain,
                    taint_score=score,
                    hop_distance=hop,
                    predecessors=(pred,) if pred else (),
                    is_cold_start=node.degree_at_first_seen <= 2,
                    flags=frozenset(flags),
                )
            )

        suspect_entries.sort(key=lambda e: e.taint_score, reverse=True)

        config_hash = Config.config_hash() if Config._instance is not None else ""

        cert = self._emit_certificate(state.observations)

        return PropagationResult(
            case_id=case_id,
            origin=origin,
            origin_chain=origin_chain,
            origin_value=origin_value,
            suspect_set=tuple(suspect_entries),
            runtime_ms=runtime_ms,
            config_hash=config_hash,
            seed=self.seed,
            path_count=path_count,
            certificate=cert["certificate"],
            cert_g_hat=cert["g_hat"],
            cert_beta_hat=cert["beta_hat"],
            cert_epsilon=cert["epsilon"],
            cert_ood=cert["ood"],
        )

    def _emit_certificate(self, observations: list) -> dict[str, Any]:
        """Per-case Azuma-Hoeffding certificate over the observed mismatch trail.

        ``hat_beta`` is the empirical off-trace mismatch rate (1 - matcher
        confidence) and ``hat_g`` is the number of distinct correlated batch
        groups (edges sharing a block collapse to one group). The certificate
        widens when batch correlation is high (large ``hat_g``) and an OOD flag
        is raised when it crosses the historical maximum used for calibration.
        """
        from crosstaint.propagation.certificate import azuma_radius, bounded_difference_C

        n = len(observations)
        if n == 0:
            return {"certificate": None, "g_hat": 0, "beta_hat": 0.0, "epsilon": 0.0, "ood": False}

        beta_hat = sum(max(0.0, 1.0 - conf) for conf, _ in observations) / n
        g_hat = len({block for _, block in observations})
        c_bound = bounded_difference_C(n)
        epsilon = azuma_radius(n, c_bound, alpha=0.05)
        certificate = min(1.0, g_hat * (beta_hat + epsilon)) if g_hat > 0 else 0.0
        ood = g_hat > self.cert_ood_g_max
        return {
            "certificate": certificate,
            "g_hat": g_hat,
            "beta_hat": beta_hat,
            "epsilon": epsilon,
            "ood": ood,
        }
