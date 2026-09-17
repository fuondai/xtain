"""Adaptive value-conservation bound ``f_max(t)`` for cross-chain propagation.

A static ``fee_bound_pct`` blinds the engine when a laundering actor
deliberately routes value through a high-slippage AMM on the destination chain
to defeat the invariant. This module tracks the empirical slippage observed
on each destination chain within a rolling time window and raises the upper
bound of the value-conservation envelope accordingly. A static envelope that
rejects a propagating edge emits an alarm event so analysts can branch to the
slippage-tolerant trace; an adaptive envelope that accepts the edge records
the observed slippage so the bound keeps tracking the worst case seen.

Public surface is intentionally small:

* :class:`SlippageWindow` — circular buffer of (chain, observed_slippage_pct).
* :class:`AdaptiveFMax` — bound calculator and alarm sink.
* :class:`AdaptiveEnvelope` — convenience wrapper used by ``PropagationEngine``.

No external dependencies beyond the standard library. Concurrency is not
required: the engine is single-threaded.
"""

from __future__ import annotations

import bisect
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(slots=True)
class _SlippageSample:
    chain: str
    slippage_pct: float
    timestamp: float


class SlippageWindow:
    """Per-chain rolling slippage statistics over a fixed time window."""

    def __init__(self, window_seconds: float = 3600.0) -> None:
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        self.window_seconds = window_seconds
        self._samples: dict[str, deque[_SlippageSample]] = defaultdict(deque)

    def record(self, chain: str, slippage_pct: float, timestamp: float) -> None:
        if slippage_pct < 0:
            raise ValueError(f"slippage_pct must be >= 0, got {slippage_pct}")
        self._samples[chain].append(_SlippageSample(chain, slippage_pct, timestamp))
        self._evict(chain, timestamp)

    def _evict(self, chain: str, now: float) -> None:
        bucket = self._samples[chain]
        cutoff = now - self.window_seconds
        while bucket and bucket[0].timestamp < cutoff:
            bucket.popleft()
        if not bucket:
            self._samples.pop(chain, None)

    def quantile(self, chain: str, q: float, now: float) -> float:
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"q must be in [0, 1], got {q}")
        self._evict(chain, now)
        bucket = self._samples.get(chain)
        if not bucket:
            return 0.0
        sorted_pcts = sorted(s.slippage_pct for s in bucket)
        idx = min(len(sorted_pcts) - 1, int(q * (len(sorted_pcts) - 1)))
        return float(sorted_pcts[idx])

    def count(self, chain: str, now: float) -> int:
        self._evict(chain, now)
        return len(self._samples.get(chain, ()))


@dataclass(slots=True)
class AlarmEvent:
    """Emitted when the static envelope rejects a propagating edge."""

    chain: str
    timestamp: float
    edge_value: int
    remaining_value: int
    static_bound: float
    observed_slippage_pct: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "chain": self.chain,
            "timestamp": self.timestamp,
            "edge_value": self.edge_value,
            "remaining_value": self.remaining_value,
            "static_bound": self.static_bound,
            "observed_slippage_pct": self.observed_slippage_pct,
            "reason": self.reason,
        }


@dataclass(slots=True)
class AdaptiveEnvelope:
    """Slippage-aware value-conservation envelope.

    ``f_max(t)`` = ``min(f_max_cap, base_fee_bound + worst_recent_slippage)``.

    * ``base_fee_bound`` mirrors the per-edge ``fee_bound_pct`` of the engine.
    * ``f_max_cap`` is a hard ceiling (default 30%) so a misbehaving data
      source cannot make the bound collapse to 1.0 (i.e. accept everything).
    * ``slippage_quantile`` is the rolling quantile used to inflate the bound
      (default p95 of the destination chain within the last hour).
    """

    base_fee_bound: float = 0.01
    f_max_cap: float = 0.30
    slippage_quantile: float = 0.95
    window_seconds: float = 3600.0
    window: SlippageWindow = field(init=False)

    def __post_init__(self) -> None:
        self.window = SlippageWindow(window_seconds=self.window_seconds)

    def feed(self, chain: str, slippage_pct: float, timestamp: float) -> None:
        self.window.record(chain, slippage_pct, timestamp)

    def f_max(self, chain: str, timestamp: float) -> float:
        observed = self.window.quantile(chain, self.slippage_quantile, timestamp)
        return min(self.f_max_cap, self.base_fee_bound + observed)

    def accepts(
        self,
        edge_value: int,
        remaining_value: int,
        chain: str,
        timestamp: float,
        alarms: list[AlarmEvent] | None = None,
    ) -> bool:
        if remaining_value <= 0 or edge_value <= 0:
            return False
        static = self.base_fee_bound
        lower_static = max(0, remaining_value - int(remaining_value * static))
        if lower_static <= edge_value <= remaining_value:
            return True
        # Static gate failed: try the slippage-inflated bound and record an
        # alarm so the analyst can branch to the slippage-tolerant trace.
        envelope = self.f_max(chain, timestamp)
        observed = self.window.quantile(chain, self.slippage_quantile, timestamp)
        lower_adaptive = max(0, int(remaining_value * (1.0 - envelope)))
        if alarms is not None and edge_value > remaining_value:
            alarms.append(
                AlarmEvent(
                    chain=chain,
                    timestamp=timestamp,
                    edge_value=edge_value,
                    remaining_value=remaining_value,
                    static_bound=static,
                    observed_slippage_pct=observed,
                    reason="edge_value_exceeds_remaining",
                )
            )
        return lower_adaptive <= edge_value <= remaining_value


class AdaptiveFMax:
    """Factory that hands out :class:`AdaptiveEnvelope` objects and aggregates alarms."""

    def __init__(
        self,
        base_fee_bound: float = 0.01,
        f_max_cap: float = 0.30,
        slippage_quantile: float = 0.95,
        window_seconds: float = 3600.0,
    ) -> None:
        self.envelope = AdaptiveEnvelope(
            base_fee_bound=base_fee_bound,
            f_max_cap=f_max_cap,
            slippage_quantile=slippage_quantile,
            window_seconds=window_seconds,
        )
        self._alarms: list[AlarmEvent] = []

    def feed_slippage(self, chain: str, slippage_pct: float, timestamp: float) -> None:
        self.envelope.feed(chain, slippage_pct, timestamp)

    def f_max(self, chain: str, timestamp: float) -> float:
        return self.envelope.f_max(chain, timestamp)

    def evaluate(
        self,
        edge_value: int,
        remaining_value: int,
        chain: str,
        timestamp: float,
    ) -> tuple[bool, list[AlarmEvent]]:
        emitted: list[AlarmEvent] = []
        accepted = self.envelope.accepts(
            edge_value=edge_value,
            remaining_value=remaining_value,
            chain=chain,
            timestamp=timestamp,
            alarms=emitted,
        )
        self._alarms.extend(emitted)
        return accepted, emitted

    def consume_alarms(self) -> list[AlarmEvent]:
        out = list(self._alarms)
        self._alarms.clear()
        return out

    def drain_alarms(self) -> Iterable[AlarmEvent]:
        while self._alarms:
            yield self._alarms.pop(0)


def find_insertion_index(samples: list[float], value: float) -> int:
    return bisect.bisect_left(samples, value)
