"""CrossTaint smoke tests: verifies core components load and basic operations succeed.

These tests validate executability only.
Run via: ``python -m crosstaint.smoke_test`` from the ``src/`` directory.
"""

from __future__ import annotations

import math
import sys
import uuid

import numpy as np


class pytest_approx:
    """Tolerance-based comparison helper usable with ``==`` without pytest."""

    def __init__(self, expected: float, rel: float = 1e-6, abs_: float = 1e-9) -> None:
        self.expected = float(expected)
        self.rel = rel
        self.abs = abs_

    def __eq__(self, other: object) -> bool:
        return math.isclose(float(other), self.expected, rel_tol=self.rel, abs_tol=self.abs)

    def __repr__(self) -> str:
        return f"approx({self.expected})"


def test_types() -> None:
    from crosstaint.types import (
        IRNode,
        IREdge,
        DecodedEvent,
        SyntheticHop,
        SyntheticTrajectory,
        PropagationResult,
        SuspectEntry,
        EvalMetrics,
        EdgeType,
        NodeType,
        TaintOperator,
    )

    node = IRNode(
        node_id="eth:0x1234",
        address="0x1234",
        chain="ethereum",
        node_type=NodeType.EOA,
        tags=frozenset(["known_exploit"]),
        first_seen=1700000000,
        degree_at_first_seen=2,
        in_value_total=1000000,
        out_value_total=500000,
        bridge_count=1,
        dex_swap_count=0,
    )
    assert node.node_id == "eth:0x1234"
    assert node.chain == "ethereum"
    assert node.tags == frozenset(["known_exploit"])

    edge = IREdge(
        edge_id="edge_001",
        source_node="eth:0xabcd",
        target_node="bsc:0x5678",
        edge_type=EdgeType.CROSS_CHAIN_BRIDGE,
        operator=TaintOperator.LOCK,
        bridge="wormhole",
        source_event="event_src",
        target_event="event_dst",
        value=500000,
        asset="USDC",
        timestamp=1700000100,
        block_number=19000000,
        fee_bound_pct=0.01,
    )
    assert edge.edge_type == EdgeType.CROSS_CHAIN_BRIDGE
    assert edge.value == 500000

    event = DecodedEvent(
        event_id="evt_001",
        chain="ethereum",
        block_number=19000000,
        timestamp=1700000100,
        tx_hash="0x" + "cd" * 32,
        event_name="Transfer",
        emitter_address="0xabcd",
        topics=(b"\x00" * 32, (b"\x00" * 11 + b"\xab" * 20 + b"\x00"),),
        params={
            "from": "0x0000000000000000000000000000000000000f01",
            "to": "0x0000000000000000000000000000000000000f02",
            "value": 1000000,
        },
        bridge="wormhole",
        selector="0xa9059cbb",
        value=1000000,
    )
    assert event.chain == "ethereum"
    assert event.value == 1000000

    hop = SyntheticHop(
        from_chain="ethereum",
        to_chain="bsc",
        bridge="wormhole",
        operator=TaintOperator.LOCK,
        value=1000,
        asset="USDC",
        inter_arrival_seconds=60.0,
    )
    trajectory = SyntheticTrajectory(
        trajectory_id=uuid.uuid4(),
        hops=(hop,),
        origin_address="0x0000000000000000000000000000000000000f01",
        addresses=(
            "0x0000000000000000000000000000000000000f01",
            "0x0000000000000000000000000000000000000f02",
        ),
        total_hops=1,
        total_value=1000,
        salt="smoke",
    )
    assert trajectory.total_hops == 1

    suspect = SuspectEntry(
        address="0x5678",
        chain="bsc",
        taint_score=0.85,
        hop_distance=2,
        predecessors=("0xabcd",),
        is_cold_start=False,
        flags=frozenset(["CrossTaint"]),
    )
    assert suspect.taint_score == 0.85

    result = PropagationResult(
        case_id=str(uuid.uuid4()),
        origin="0xabcd",
        origin_chain="ethereum",
        origin_value=1000000,
        suspect_set=(suspect,),
        runtime_ms=12.5,
        config_hash="test_hash",
        seed=42,
        path_count=1,
    )
    assert result.origin == "0xabcd"

    metrics = EvalMetrics(
        hop_recall=0.8,
        precision=0.9,
        false_positive_rate=0.05,
        true_positive=80,
        false_positive=5,
        false_negative=20,
        runtime_median_ms=10.0,
        runtime_p95_ms=25.0,
        num_cases=100,
    )
    assert metrics.hop_recall == 0.8


def test_config() -> None:
    from crosstaint.config import Config

    Config.reset()
    cfg = Config.load()
    assert cfg is not None
    bridges = cfg.bridge_list
    assert isinstance(bridges, list)
    chains = cfg.chain_list
    assert isinstance(chains, list)
    spec = cfg.bridge_spec("wormhole")
    assert isinstance(spec, dict)
    chain_spec = cfg.chain_spec("ethereum")
    assert isinstance(chain_spec, dict)
    Config.reset()


def test_synthetic_generator() -> None:
    from crosstaint.synth import (
        SyntheticBenignGenerator,
        BRIDGES,
        BRIDGE_CHAINS,
        BRIDGE_USAGE_WEIGHTS,
    )

    assert len(BRIDGES) > 0
    assert len(BRIDGE_CHAINS) > 0
    assert len(BRIDGE_USAGE_WEIGHTS) > 0

    generator = SyntheticBenignGenerator(seed=42)
    trajectories = generator.generate(num_trajectories=10)
    assert len(trajectories) == 10
    for traj in trajectories:
        assert traj.total_hops >= 1
        assert len(traj.addresses) >= 2
        assert all(isinstance(a, str) for a in traj.addresses)


def test_indexer() -> None:
    from crosstaint.indexer import SyntheticGraphBuilder
    from crosstaint.synth import SyntheticBenignGenerator

    generator = SyntheticBenignGenerator(seed=42)
    trajectories = generator.generate(num_trajectories=5)
    exploit_addresses = list(trajectories[0].addresses[:2]) if trajectories[0].addresses else []

    builder = SyntheticGraphBuilder(trajectories, exploit_addresses=exploit_addresses)
    nodes, edges = builder.build()
    assert len(nodes) > 0
    assert len(edges) > 0

    sample_trajectory = trajectories[0]
    hops = [(h.from_chain, h.to_chain, h.bridge) for h in sample_trajectory.hops[:2]]
    case_nodes, case_edges = builder.create_exploit_case(
        origin=sample_trajectory.origin_address,
        chain=sample_trajectory.hops[0].from_chain,
        value=1000000,
        hops=hops,
    )
    assert len(case_nodes) > 0


def test_propagation() -> None:
    from crosstaint.propagation import PropagationEngine
    from crosstaint.propagation.decay import DecayScheduler
    from crosstaint.propagation.weight import EdgeWeightLearner
    from crosstaint.types import IRNode, IREdge, EdgeType, NodeType, TaintOperator

    nodes = {
        "eth:0xaaaa": IRNode(
            node_id="eth:0xaaaa", address="0xaaaa", chain="ethereum",
            node_type=NodeType.EOA, tags=frozenset(),
            first_seen=1700000000, degree_at_first_seen=0,
            in_value_total=0, out_value_total=0, bridge_count=0, dex_swap_count=0,
        ),
        "bsc:0xbbbb": IRNode(
            node_id="bsc:0xbbbb", address="0xbbbb", chain="bsc",
            node_type=NodeType.EOA, tags=frozenset(),
            first_seen=1700000100, degree_at_first_seen=0,
            in_value_total=0, out_value_total=0, bridge_count=0, dex_swap_count=0,
        ),
    }

    edges = [
        IREdge(
            edge_id="edge_0",
            source_node="eth:0xaaaa",
            target_node="bsc:0xbbbb",
            edge_type=EdgeType.CROSS_CHAIN_BRIDGE,
            operator=TaintOperator.LOCK,
            bridge="wormhole",
            source_event="evt_0",
            target_event="evt_1",
            value=500000,
            asset="USDC",
            timestamp=1700000100,
            block_number=19000000,
            fee_bound_pct=0.01,
        ),
    ]

    scheduler = DecayScheduler(rho=0.81)
    assert scheduler.decay(1) == pytest_approx(0.81)
    assert scheduler.decay(2) == pytest_approx(0.81 ** 2)

    weight_learner = EdgeWeightLearner()
    w_bridge = weight_learner.get_weight_by_type(EdgeType.CROSS_CHAIN_BRIDGE)
    w_intra = weight_learner.get_weight_by_type(EdgeType.INTRA_CHAIN_TRANSFER)
    w_dex = weight_learner.get_weight_by_type(EdgeType.DEX_SWAP)
    assert 0.0 < w_bridge <= 1.0
    assert 0.0 < w_intra <= 1.0
    assert 0.0 < w_dex <= 1.0

    engine = PropagationEngine(rho=0.81, threshold=0.04, seed=42)
    result = engine.propagate(
        origin="eth:0xaaaa",
        origin_chain="ethereum",
        origin_value=500000,
        graph=nodes,
        edges=edges,
        case_id="test_case_001",
    )
    assert result.origin == "eth:0xaaaa"
    assert result.origin_chain == "ethereum"
    assert result.runtime_ms >= 0.0


def test_bridge_adapters() -> None:
    from crosstaint.ir.adapters import (
        WormholeAdapter,
        LayerZeroAdapter,
        HyperlaneAdapter,
        ALL_BRIDGE_ADAPTERS,
        AdapterRegistry,
    )
    from crosstaint.types import DecodedEvent

    adapters = [WormholeAdapter(), LayerZeroAdapter(), HyperlaneAdapter()]
    assert len(adapters) == 3

    for adapter in adapters:
        assert adapter.bridge_id() in ("wormhole", "layerzero", "hyperlane")
        src_events = adapter.source_event_names()
        assert isinstance(src_events, tuple)
        assert len(src_events) > 0
        dst_events = adapter.dest_event_names()
        assert isinstance(dst_events, tuple)
        assert len(dst_events) > 0
        mode = adapter.bridge_mode()
        assert isinstance(mode, str)

    wormhole = WormholeAdapter()
    assert wormhole.bridge_id() == "wormhole"
    assert wormhole.bridge_mode() == "lock_mint"

    assert len(ALL_BRIDGE_ADAPTERS) == 9

    sender_topic = b"\x00" * 12 + b"\xaa" * 20
    recipient_topic = b"\x00" * 12 + b"\xbb" * 20
    sample_lock_event = DecodedEvent(
        event_id="sample_lock",
        chain="ethereum",
        block_number=19000000,
        timestamp=1700000100,
        tx_hash="0x" + "cd" * 32,
        event_name="LogMessagePublished",
        emitter_address="0x" + "00" * 20,
        params={"amount": 500000, "sender": "0x" + "aa" * 20, "recipient": "0x" + "bb" * 20},
        topics=(b"\x00" * 32, sender_topic, recipient_topic),
        bridge="wormhole",
        selector="0x00000000",
    )
    lock_edge = wormhole.decode_lock(sample_lock_event)
    assert lock_edge is not None

    registry = AdapterRegistry()
    assert "wormhole" in registry.supported_bridges
    assert registry.get("wormhole") is not None


def test_ir_translator() -> None:
    from crosstaint.ir import BridgeIRTranslator
    from crosstaint.types import DecodedEvent

    translator = BridgeIRTranslator()

    src_event = DecodedEvent(
        event_id="src_001",
        chain="ethereum",
        block_number=19000000,
        timestamp=1700000000,
        tx_hash="0x" + "cd" * 32,
        event_name="Lock",
        emitter_address="0xaaaa",
        params={"token": "USDC", "amount": 1000000, "recipient": "0xbbbb"},
        topics=(),
        bridge="wormhole",
        selector="0x1b2c3d4e",
    )
    dst_event = DecodedEvent(
        event_id="dst_001",
        chain="bsc",
        block_number=18000000,
        timestamp=1700000100,
        tx_hash="0x" + "34" * 32,
        event_name="Mint",
        emitter_address="0xcccc",
        params={"token": "USDC", "amount": 1000000, "recipient": "0xdddd"},
        topics=(),
        bridge="wormhole",
        selector="0x5f6a7b8c",
    )

    edge = translator.translate(
        src_event, dst_event, "wormhole", "lock_mint", "eth:0xaaaa", "bsc:0xdddd"
    )
    assert edge is not None
    assert edge.edge_type == "CROSS_CHAIN_BRIDGE"
    assert edge.operator == "Lock"
    assert edge.bridge == "wormhole"
    assert edge.value == 1000000
    assert edge.asset == "USDC"


def test_graph_builder() -> None:
    from crosstaint.ir import IRGraphBuilder, WormholeAdapter
    from crosstaint.types import DecodedEvent

    adapter = WormholeAdapter()
    builder = IRGraphBuilder(bridge_adapters={"wormhole": adapter})

    sender_topic = b"\x00" * 12 + b"\xaa" * 20
    recipient_topic = b"\x00" * 12 + b"\xbb" * 20
    src_event = DecodedEvent(
        event_id="src_002",
        chain="ethereum",
        block_number=19000000,
        timestamp=1700000000,
        tx_hash="0x" + "cd" * 32,
        event_name="LogMessagePublished",
        emitter_address="0x" + "aa" * 20,
        params={"amount": 500000, "sender": "0x" + "aa" * 20, "recipient": "0x" + "bb" * 20},
        topics=(b"\x00" * 32, sender_topic, recipient_topic, b"\x00" * 32),
        bridge="wormhole",
        selector="0x1b2c3d4e",
    )
    dst_event = DecodedEvent(
        event_id="dst_002",
        chain="bsc",
        block_number=38000000,
        timestamp=1700000200,
        tx_hash="0x" + "ef" * 32,
        event_name="TransferRedeemed",
        emitter_address="0x" + "bb" * 20,
        params={"amount": 500000, "sender": "0x" + "aa" * 20, "recipient": "0x" + "bb" * 20},
        topics=(b"\x00" * 32, sender_topic, recipient_topic, b"\x00" * 32),
        bridge="wormhole",
        selector="0x5f6a7b8c",
    )
    nodes, edges = builder.build_from_events(
        raw_events=[src_event, dst_event],
        pairs=[(src_event, dst_event)],
    )
    assert any(n.endswith(":0x" + "aa" * 20) for n in nodes)
    assert any(n.endswith(":0x" + "bb" * 20) for n in nodes)


def test_eval() -> None:
    from crosstaint.eval import (
        compute_hop_recall,
        compute_precision,
        compute_false_positive_rate,
        compute_f1,
        compute_aggregate_metrics,
    )
    from crosstaint.types import EvalMetrics

    predicted = [["0xaaa", "0xbbb", "0xccc"]]
    ground_truth = [["0xaaa", "0xbbb", "0xddd"]]

    recall = compute_hop_recall(predicted, ground_truth)
    assert 0.0 <= recall <= 1.0
    assert recall == pytest_approx(2 / 3)

    precision = compute_precision(predicted, ground_truth)
    assert 0.0 <= precision <= 1.0
    assert precision == pytest_approx(2 / 3)

    f1 = compute_f1(recall, precision)
    assert 0.0 <= f1 <= 1.0

    fpr = compute_false_positive_rate(["0xaaa", "0xbbb"], ["0xaaa", "0xeee", "0xfff"])
    assert 0.0 <= fpr <= 1.0

    metrics = [
        EvalMetrics(
            hop_recall=0.8,
            precision=0.9,
            false_positive_rate=0.05,
            true_positive=80,
            false_positive=5,
            false_negative=20,
            runtime_median_ms=10.0,
            runtime_p95_ms=25.0,
            num_cases=100,
        ),
    ]
    agg = compute_aggregate_metrics(metrics)
    assert agg["hop_recall_mean"] == pytest_approx(0.8)
    assert agg["num_runs"] == 1


def test_experiment_run() -> None:
    from crosstaint.experiments.run import (
        ExperimentConfig,
        _generate_exploit_cases,
        _graph_repr,
    )

    cfg = ExperimentConfig(num_benign=10, num_exploit=2, num_runs=1, seed=42)
    assert cfg.num_benign == 10
    assert cfg.num_exploit == 2

    cases = _generate_exploit_cases(5, seed=42)
    assert len(cases) == 5
    for case in cases:
        assert case.chain == case.hops[0][0]
        assert len(case.hops) >= 1

    from crosstaint.synth import SyntheticBenignGenerator
    from crosstaint.indexer import SyntheticGraphBuilder

    generator = SyntheticBenignGenerator(seed=42)
    trajectories = generator.generate(num_trajectories=5)
    builder = SyntheticGraphBuilder(
        trajectories,
        exploit_addresses=[trajectories[0].origin_address] if trajectories else [],
    )
    nodes, _ = builder.build()
    nodes_snap, edges_snap = builder.snapshot()
    graph = _graph_repr(nodes_snap, list(edges_snap.values()))
    assert "nodes" in graph
    assert "edges" in graph


def test_corpus_loading() -> None:
    from crosstaint.experiments.corpus import BenchmarkCase, BenchmarkDataset

    case = BenchmarkCase(
        case_id="test_case",
        origin_address="0xaaaa",
        origin_chain="ethereum",
        origin_value=1000000,
        ground_truth_addresses=["0xbbbb", "0xcccc"],
        source_count=2,
    )
    assert case.origin_address == "0xaaaa"
    assert len(case.ground_truth_addresses) == 2
    assert case.origin_node_id == "ethereum:0xaaaa"


def test_adaptive_fmax() -> None:
    from crosstaint.propagation import AdaptiveFMax, AlarmEvent

    fmax = AdaptiveFMax(base_fee_bound=0.01, f_max_cap=0.30, window_seconds=3600.0)
    fmax.feed_slippage("bsc", 0.20, timestamp=1000.0)
    fmax.feed_slippage("bsc", 0.18, timestamp=1100.0)
    fmax.feed_slippage("bsc", 0.22, timestamp=1200.0)
    accepted, alarms = fmax.evaluate(
        edge_value=800_000,
        remaining_value=1_000_000,
        chain="bsc",
        timestamp=1200.0,
    )
    assert accepted is True
    assert isinstance(alarms, list)
    over_accepted, over_alarms = fmax.evaluate(
        edge_value=1_500_000,
        remaining_value=1_000_000,
        chain="bsc",
        timestamp=1200.0,
    )
    assert over_accepted is False
    assert len(over_alarms) == 1
    assert isinstance(over_alarms[0], AlarmEvent)
    accepted_zero, _ = fmax.evaluate(
        edge_value=995_000,
        remaining_value=1_000_000,
        chain="polygon",
        timestamp=1200.0,
    )
    assert accepted_zero is True


def test_g_estimator() -> None:
    from crosstaint.propagation import GEstimator, union_bound_static

    est = GEstimator(eps=1e-3, p99=4)
    case = est.begin_case()
    for conf in (0.9, 0.8, 0.7, 0.95, 0.85):
        case.observe_edge(conf)
    cert = case.finalize(case_id="smoke_case", timestamp=0.0)
    assert cert.hat_g == 5
    assert abs(cert.hat_beta - 0.16) < 1e-12
    assert 0.0 <= cert.hat_beta <= 1.0
    assert 0.0 <= cert.certificate <= 1.0
    assert cert.ood is True
    alarms = est.drain_alarms()
    assert len(alarms) == 1
    assert 0.0 <= union_bound_static(5, cert.hat_beta) <= 1.0
    case2 = est.begin_case()
    case2.observe_edge(0.9)
    cert2 = case2.finalize(case_id="short_case", timestamp=0.0)
    assert abs(cert2.hat_beta - 0.1) < 1e-12
    assert cert2.ood is False
    assert est.drain_alarms() == []


def test_adversarial_slippage_generator() -> None:
    from crosstaint.synth import AdversarialSlippageGenerator

    gen = AdversarialSlippageGenerator(slippage_rate=1.0, seed=42)
    trajs = gen.generate(num_trajectories=4)
    assert len(trajs) == 4
    assert all(1 <= h.value for t in trajs for h in t.hops)
    assert all(t.salt.startswith("adv-") for t in trajs)


def test_synthetic_vs_real_fpr() -> None:
    from crosstaint.eval import compute_synthetic_vs_real_fpr

    out = compute_synthetic_vs_real_fpr(
        predicted_set=["0xaaa", "0xbbb"],
        real_address_holdout=["0xaaa"],
        synthetic_benign=["0xbbb", "0xccc"],
    )
    assert out["fpr_real"] == 1.0
    assert out["fpr_synthetic"] == 0.5
    assert 0.0 <= out["fpr_combined"] <= 1.0
    out2 = compute_synthetic_vs_real_fpr(
        predicted_set=[],
        real_address_holdout=["0xaaa"],
        synthetic_benign=["0xbbb"],
    )
    assert out2["fpr_real"] == 0.0
    assert out2["fpr_synthetic"] == 0.0


def test_skip_hop_recovery_keeps_chain_bounded() -> None:
    from crosstaint.propagation import PropagationEngine
    from crosstaint.types import IRNode, IREdge, EdgeType, NodeType, TaintOperator

    nodes = {
        "eth:0xa": IRNode(
            node_id="eth:0xa", address="0xa", chain="ethereum",
            node_type=NodeType.EOA, tags=frozenset(),
            first_seen=0, degree_at_first_seen=0,
            in_value_total=0, out_value_total=0, bridge_count=0, dex_swap_count=0,
        ),
        "eth:0xb": IRNode(
            node_id="eth:0xb", address="0xb", chain="ethereum",
            node_type=NodeType.EOA, tags=frozenset(),
            first_seen=0, degree_at_first_seen=0,
            in_value_total=0, out_value_total=0, bridge_count=0, dex_swap_count=0,
        ),
        "bsc:0xc": IRNode(
            node_id="bsc:0xc", address="0xc", chain="bsc",
            node_type=NodeType.EOA, tags=frozenset(),
            first_seen=0, degree_at_first_seen=0,
            in_value_total=0, out_value_total=0, bridge_count=0, dex_swap_count=0,
        ),
    }
    edges = [
        IREdge(
            edge_id="intra", source_node="eth:0xa", target_node="eth:0xb",
            edge_type=EdgeType.INTRA_CHAIN_TRANSFER,
            operator=TaintOperator.LOCK, bridge="",
            source_event=None, target_event=None, value=1_000_000,
            asset="USDC", timestamp=0, block_number=0, fee_bound_pct=0.0,
        ),
        IREdge(
            edge_id="cross", source_node="eth:0xb", target_node="bsc:0xc",
            edge_type=EdgeType.CROSS_CHAIN_BRIDGE,
            operator=TaintOperator.MINT, bridge="wormhole",
            source_event=None, target_event=None, value=900_000,
            asset="USDC", timestamp=1, block_number=1, fee_bound_pct=0.10,
        ),
    ]
    engine = PropagationEngine(
        rho=0.81, threshold=0.04, seed=42,
        skip_hop_recovery=True, skip_gate_multiplier=2.0,
    )
    result = engine.propagate(
        origin="eth:0xa", origin_chain="ethereum", origin_value=1_000_000,
        graph=nodes, edges=edges, case_id="skip_test",
    )
    assert any(s.address == "0xc" for s in result.suspect_set)


def run_all() -> bool:
    tests = [
        ("types", test_types),
        ("config", test_config),
        ("synthetic_generator", test_synthetic_generator),
        ("indexer", test_indexer),
        ("propagation", test_propagation),
        ("bridge_adapters", test_bridge_adapters),
        ("ir_translator", test_ir_translator),
        ("graph_builder", test_graph_builder),
        ("eval", test_eval),
        ("experiment_run", test_experiment_run),
        ("corpus_loading", test_corpus_loading),
        ("adaptive_fmax", test_adaptive_fmax),
        ("g_estimator", test_g_estimator),
        ("adversarial_slippage", test_adversarial_slippage_generator),
        ("synthetic_vs_real_fpr", test_synthetic_vs_real_fpr),
        ("skip_hop_recovery", test_skip_hop_recovery_keeps_chain_bounded),
    ]

    passed = 0
    failed = 0
    failures: list[tuple[str, str]] = []

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            failures.append((name, str(exc)))
            failed += 1

    print(f"\n{passed}/{len(tests)} tests passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
