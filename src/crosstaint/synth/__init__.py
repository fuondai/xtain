from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Iterator
from uuid import UUID

import numpy as np

from crosstaint.types import SyntheticHop, SyntheticTrajectory, BridgeMode


BRIDGES = ["wormhole", "layerzero", "multichain", "stargate", "across", "hop", "cbridge", "synapse", "hyperlane"]
CHAINS = ["ethereum", "bsc", "polygon", "arbitrum", "avalanche"]

BRIDGE_CHAINS: dict[str, list[str]] = {
    "wormhole": ["ethereum", "bsc", "polygon", "arbitrum", "avalanche"],
    "layerzero": ["ethereum", "bsc", "polygon", "arbitrum", "avalanche"],
    "multichain": ["ethereum", "bsc", "polygon", "arbitrum", "avalanche"],
    "stargate": ["ethereum", "bsc", "polygon", "arbitrum", "avalanche"],
    "across": ["ethereum", "polygon", "arbitrum"],
    "hop": ["ethereum", "polygon", "arbitrum"],
    "cbridge": ["ethereum", "bsc", "polygon", "arbitrum", "avalanche"],
    "synapse": ["ethereum", "bsc", "polygon", "arbitrum", "avalanche"],
    "hyperlane": ["ethereum", "bsc", "polygon", "arbitrum", "avalanche"],
}

BRIDGE_MODES: dict[str, str] = {
    "wormhole": BridgeMode.LOCK_MINT,
    "layerzero": BridgeMode.LOCK_MINT,
    "multichain": BridgeMode.LOCK_MINT,
    "stargate": BridgeMode.POOL,
    "across": BridgeMode.INTENT,
    "hop": BridgeMode.POOL,
    "cbridge": BridgeMode.POOL,
    "synapse": BridgeMode.BURN_MINT,
    "hyperlane": BridgeMode.LOCK_MINT,
}

OPERATOR_PAIRS: dict[str, tuple[str, str]] = {
    BridgeMode.LOCK_MINT: ("Lock", "Mint"),
    BridgeMode.BURN_MINT: ("Burn", "Mint"),
    BridgeMode.POOL: ("Lock", "Release"),
    BridgeMode.INTENT: ("Lock", "IntentFill"),
}

HOP_COUNT_DIST: dict[int, float] = {
    1: 0.35, 2: 0.28, 3: 0.18, 4: 0.10,
    5: 0.05, 6: 0.02, 7: 0.01, 8: 0.005, 9: 0.003, 10: 0.002,
}

BRIDGE_USAGE_WEIGHTS: dict[str, float] = {
    "stargate": 0.22, "wormhole": 0.16, "layerzero": 0.15,
    "across": 0.12, "hop": 0.10, "cbridge": 0.09,
    "synapse": 0.08, "hyperlane": 0.05, "multichain": 0.03,
}

ASSETS = ["ETH", "USDC", "USDT", "DAI", "WBTC", "WETH"]
ASSET_WEIGHTS = [0.30, 0.28, 0.20, 0.10, 0.07, 0.05]


@dataclass(frozen=True, slots=True)
class _WeightedItem:
    value: Any
    weight: float


class SyntheticBenignGenerator:
    def __init__(self, salt: str | None = None, seed: int = 42) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        if salt is None:
            self._salt = self._derive_salt(seed)
        else:
            self._salt = salt

        self._bridge_items = [
            _WeightedItem(bridge, weight)
            for bridge, weight in BRIDGE_USAGE_WEIGHTS.items()
        ]
        self._hop_items = [
            _WeightedItem(hops, weight)
            for hops, weight in HOP_COUNT_DIST.items()
        ]
        self._asset_items = [
            _WeightedItem(asset, weight)
            for asset, weight in zip(ASSETS, ASSET_WEIGHTS)
        ]

    def _derive_salt(self, seed: int) -> str:
        seed_bytes = str(seed).encode("utf-8")
        return hashlib.sha256(seed_bytes).hexdigest()[:32]

    def _weighted_choice(self, items: list[_WeightedItem]) -> str:
        values = [item.value for item in items]
        weights = [item.weight for item in items]
        return self._rng.choices(values, weights=weights, k=1)[0]

    def _sample_bridge(self) -> str:
        return self._weighted_choice(self._bridge_items)

    def _sample_hop_count(self, hop_distribution: dict[int, float]) -> int:
        items = [
            _WeightedItem(hops, weight)
            for hops, weight in hop_distribution.items()
        ]
        return int(self._weighted_choice(items))

    def _sample_asset(self, value: int) -> str:
        if value < 1e16:
            return self._rng.choices(["ETH", "WETH", "USDC", "USDT"], weights=[0.35, 0.25, 0.25, 0.15], k=1)[0]
        elif value < 1e17:
            return self._rng.choices(["ETH", "WETH", "USDC", "USDT", "DAI"], weights=[0.30, 0.20, 0.25, 0.15, 0.10], k=1)[0]
        elif value < 1e18:
            return self._rng.choices(["ETH", "WETH", "USDC", "USDT", "DAI", "WBTC"], weights=[0.25, 0.20, 0.20, 0.15, 0.10, 0.10], k=1)[0]
        else:
            return self._rng.choices(["ETH", "WETH", "WBTC", "USDC"], weights=[0.40, 0.30, 0.20, 0.10], k=1)[0]

    def _sample_value(self) -> int:
        log_min = 15.0
        log_max = 18.0
        log_value = self._np_rng.uniform(log_min, log_max)
        value = 10 ** log_value
        return int(value)

    def _sample_inter_arrival(self, mu: float = 2.1, sigma: float = 1.4) -> float:
        return float(self._np_rng.lognormal(mu, sigma))

    def _fee_bound_for_bridge(self, bridge: str) -> float:
        mode = BRIDGE_MODES.get(bridge, BridgeMode.LOCK_MINT)
        if mode in (BridgeMode.LOCK_MINT, BridgeMode.BURN_MINT):
            return 0.01
        if mode == BridgeMode.POOL:
            return 0.05
        if mode == BridgeMode.INTENT:
            return 0.08
        return 0.01

    def _sample_fee_fraction(self, bridge: str) -> float:
        return float(self._np_rng.uniform(0.0, self._fee_bound_for_bridge(bridge)))

    def _sample_chains_for_bridge(self, bridge: str) -> tuple[str, str]:
        chains = BRIDGE_CHAINS.get(bridge, CHAINS)
        if len(chains) < 2:
            chains = CHAINS

        source = self._rng.choice(chains)
        dest_candidates = [c for c in chains if c != source]
        dest = self._rng.choice(dest_candidates) if dest_candidates else source

        return source, dest

    def _sample_bridge_for_chain(self, chain: str, allowed_bridges: list[str]) -> str:
        candidates = [
            bridge for bridge in allowed_bridges
            if chain in BRIDGE_CHAINS.get(bridge, CHAINS)
        ]
        if not candidates:
            candidates = allowed_bridges

        total = sum(BRIDGE_USAGE_WEIGHTS.get(b, 0.01) for b in candidates)
        weights = [
            BRIDGE_USAGE_WEIGHTS.get(b, 0.01) / total
            if total > 0 else 1.0 / len(candidates)
            for b in candidates
        ]
        return self._rng.choices(candidates, weights=weights, k=1)[0]

    def _sample_dest_for_bridge(self, bridge: str, source_chain: str) -> str:
        chains = BRIDGE_CHAINS.get(bridge, CHAINS)
        dest_candidates = [chain for chain in chains if chain != source_chain]
        if not dest_candidates:
            dest_candidates = [chain for chain in CHAINS if chain != source_chain]
        return self._rng.choice(dest_candidates)

    def _generate_salted_addresses(
        self, count: int, trajectory_id: str
    ) -> list[str]:
        addresses = []
        for i in range(count):
            data = f"{self._salt}:{trajectory_id}:{i}".encode("utf-8")
            hash_digest = hashlib.sha256(data).hexdigest()
            address = f"0x{hash_digest[:40]}"
            addresses.append(address)
        return addresses

    def _generate_hops(
        self,
        num_hops: int,
        bridge: str,
        trajectory_id: str,
        inter_arrival_mu: float,
        inter_arrival_sigma: float,
        allowed_bridges: list[str],
    ) -> tuple[list[SyntheticHop], list[str]]:
        hops = []
        addresses = self._generate_salted_addresses(num_hops + 1, trajectory_id)

        source_chain, dest_chain = self._sample_chains_for_bridge(bridge)
        current_chain = source_chain
        current_value = self._sample_value()
        for i in range(num_hops):
            hop_bridge = bridge if i == 0 else self._sample_bridge_for_chain(current_chain, allowed_bridges)
            source_chain = current_chain
            dest_chain = self._sample_dest_for_bridge(hop_bridge, source_chain)
            bridge_mode = BRIDGE_MODES.get(hop_bridge, BridgeMode.LOCK_MINT)
            source_op, dest_op = OPERATOR_PAIRS.get(bridge_mode, ("Lock", "Mint"))

            fee_fraction = self._sample_fee_fraction(hop_bridge)
            value = max(1, int(current_value * (1.0 - fee_fraction)))
            asset = self._sample_asset(value)
            inter_arrival = self._sample_inter_arrival(inter_arrival_mu, inter_arrival_sigma)

            operator = source_op if i == 0 else dest_op

            hop = SyntheticHop(
                from_chain=source_chain,
                to_chain=dest_chain,
                bridge=hop_bridge,
                operator=operator,
                value=value,
                asset=asset,
                inter_arrival_seconds=inter_arrival,
            )
            hops.append(hop)
            current_chain = dest_chain
            current_value = value

        return hops, addresses

    def generate(
        self,
        num_trajectories: int,
        hop_distribution: dict[int, float] | None = None,
        bridges: list[str] | None = None,
        inter_arrival_mu: float = 2.1,
        inter_arrival_sigma: float = 1.4,
    ) -> list[SyntheticTrajectory]:
        if bridges is None:
            allowed_bridges = list(BRIDGES)
            bridge_weights = {
                b: BRIDGE_USAGE_WEIGHTS.get(b, 0.01) for b in BRIDGES
            }
        else:
            allowed_bridges = list(bridges)
            total = sum(BRIDGE_USAGE_WEIGHTS.get(b, 0.01) for b in bridges)
            bridge_weights = {
                b: BRIDGE_USAGE_WEIGHTS.get(b, 0.01) / total if total > 0 else 0.0
                for b in bridges
            }

        hop_dist = hop_distribution if hop_distribution is not None else HOP_COUNT_DIST

        trajectories = []
        for i in range(num_trajectories):
            trajectory_id = hashlib.sha256(
                f"{self._salt}:{self._seed}:{i}".encode("utf-8")
            ).hexdigest()
            trajectory_uuid = UUID(trajectory_id[:32])

            bridge = self._rng.choices(
                list(bridge_weights.keys()),
                weights=list(bridge_weights.values()),
                k=1,
            )[0]

            hop_count = self._sample_hop_count(hop_dist)
            hops, addresses = self._generate_hops(
                hop_count,
                bridge,
                trajectory_id,
                inter_arrival_mu,
                inter_arrival_sigma,
                allowed_bridges,
            )

            total_value = sum(h.value for h in hops)

            trajectory = SyntheticTrajectory(
                trajectory_id=trajectory_uuid,
                hops=tuple(hops),
                origin_address=addresses[0] if addresses else "",
                addresses=tuple(addresses),
                total_hops=len(hops),
                total_value=total_value,
                salt=self._salt,
            )
            trajectories.append(trajectory)

        return trajectories

    def generate_stream(
        self,
        batch_size: int = 100,
        total: int | None = None,
        hop_distribution: dict[int, float] | None = None,
        bridges: list[str] | None = None,
        inter_arrival_mu: float = 2.1,
        inter_arrival_sigma: float = 1.4,
    ) -> Iterator[SyntheticTrajectory]:
        generated = 0
        while total is None or generated < total:
            batch = self.generate(
                num_trajectories=min(batch_size, (total or batch_size) - generated),
                hop_distribution=hop_distribution,
                bridges=bridges,
                inter_arrival_mu=inter_arrival_mu,
                inter_arrival_sigma=inter_arrival_sigma,
            )
            for trajectory in batch:
                yield trajectory
                generated += 1
                if total is not None and generated >= total:
                    break

    @staticmethod
    def save_parquet(
        trajectories: list[SyntheticTrajectory], path: str
    ) -> None:
        import pandas as pd

        rows = []
        for traj in trajectories:
            for hop_idx, hop in enumerate(traj.hops):
                rows.append({
                    "trajectory_id": str(traj.trajectory_id),
                    "hop_index": hop_idx,
                    "from_chain": hop.from_chain,
                    "to_chain": hop.to_chain,
                    "bridge": hop.bridge,
                    "operator": hop.operator,
                    "value": hop.value,
                    "asset": hop.asset,
                    "inter_arrival_seconds": hop.inter_arrival_seconds,
                    "origin_address": traj.origin_address,
                    "addresses": list(traj.addresses),
                    "total_hops": traj.total_hops,
                    "total_value": traj.total_value,
                    "salt": traj.salt,
                })

        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)

    @staticmethod
    def load_parquet(path: str):
        import pandas as pd

        return pd.read_parquet(path)

    @staticmethod
    def compute_wasserstein(
        gen_samples: list[float], ref_samples: list[float], axis_name: str = "value"
    ) -> float:
        del axis_name
        gen_arr = np.array(gen_samples, dtype=np.float64)
        ref_arr = np.array(ref_samples, dtype=np.float64)

        if len(gen_arr) == 0 or len(ref_arr) == 0:
            return float("inf")

        gen_sorted = np.sort(gen_arr)
        ref_sorted = np.sort(ref_arr)

        n_gen = len(gen_sorted)
        n_ref = len(ref_sorted)

        if n_gen == n_ref:
            distance = np.mean(np.abs(gen_sorted - ref_sorted))
        else:
            gen_quantiles = (np.arange(n_gen) + 0.5) / n_gen
            ref_quantiles = (np.arange(n_ref) + 0.5) / n_ref

            gen_quantile_values = np.quantile(gen_sorted, gen_quantiles)
            ref_quantile_values = np.quantile(ref_sorted, ref_quantiles)

            distance = float(np.mean(np.abs(gen_quantile_values - ref_quantile_values)))

        return distance


class AdversarialSlippageGenerator:
    """Generate exploit trajectories that deliberately exceed fee bounds.

    RQ-1 fix: a static value-conservation envelope is blind to a laundering
    actor that routes value through a high-slippage AMM on the destination
    chain to defeat the invariant. This generator emits a fraction ``slippage_rate``
    of trajectories whose per-hop loss exceeds the default fee bound of the
    chosen bridge mode, so the smoke test exercises the adaptive
    :class:`AdaptiveEnvelope` and a known failure mode is recorded honestly
    in the methodology.
    """

    def __init__(self, slippage_rate: float = 0.20, seed: int = 42) -> None:
        if not (0.0 <= slippage_rate <= 1.0):
            raise ValueError(f"slippage_rate must be in [0, 1], got {slippage_rate}")
        self.slippage_rate = slippage_rate
        self._seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._benign = SyntheticBenignGenerator(seed=seed)

    def generate(self, num_trajectories: int) -> list[SyntheticTrajectory]:
        trajectories: list[SyntheticTrajectory] = []
        for idx in range(num_trajectories):
            if self._rng.random() < self.slippage_rate:
                trajectories.append(self._slippage_trajectory(idx))
            else:
                trajectories.extend(self._benign.generate(num_trajectories=1))
        return trajectories

    def _slippage_trajectory(self, idx: int) -> SyntheticTrajectory:
        base = self._benign.generate(num_trajectories=1)[0]
        # Replace the per-hop value with a degraded fraction. The actor is
        # willing to lose 15-25% per hop to blind the static envelope.
        loss = float(self._np_rng.uniform(0.15, 0.25))
        degraded_hops = tuple(
            SyntheticHop(
                from_chain=h.from_chain,
                to_chain=h.to_chain,
                bridge=h.bridge,
                operator=h.operator,
                value=max(1, int(h.value * (1.0 - loss))),
                asset=h.asset,
                inter_arrival_seconds=h.inter_arrival_seconds,
            )
            for h in base.hops
        )
        return SyntheticTrajectory(
            trajectory_id=base.trajectory_id,
            hops=degraded_hops,
            origin_address=base.origin_address,
            addresses=base.addresses,
            total_hops=base.total_hops,
            total_value=sum(h.value for h in degraded_hops),
            salt=f"adv-{idx}-{base.salt}",
        )
