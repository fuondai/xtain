"""Online estimator for the per-case ``g`` term used in the union-bound certificate.

The static union bound ``1 - (1 - beta)^g`` becomes brittle when
``g >> 4`` (long laundering trails). The replacement is a per-case certificate
``min(1, hat_g * (hat_beta + eps))`` where:

* ``hat_g`` is the realised count of independent edges observed on the
  propagating trail (streaming, no allocation),
* ``hat_beta`` is the empirical off-trace mismatch rate over the same trail,
* ``eps`` is a non-negative slack defaulting to ``1e-3`` to keep the bound
  strict on edge cases where ``hat_beta`` collapses to zero.

The estimator also tracks the empirical p99 of ``hat_g`` from a recent
training run and emits an out-of-distribution alarm when a live case exceeds
it, so a CrossTaint analyst can flag an unusually long trail for manual
investigation.

No dependencies beyond the standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(slots=True)
class CertificateResult:
    hat_g: int
    hat_beta: float
    certificate: float
    ood: bool
    p99_threshold: int

    def as_dict(self) -> dict[str, object]:
        return {
            "hat_g": self.hat_g,
            "hat_beta": self.hat_beta,
            "certificate": self.certificate,
            "ood": self.ood,
            "p99_threshold": self.p99_threshold,
        }


@dataclass(slots=True)
class OODAlarm:
    case_id: str
    hat_g: int
    p99_threshold: int
    timestamp: float

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "hat_g": self.hat_g,
            "p99_threshold": self.p99_threshold,
            "timestamp": self.timestamp,
        }


class GEstimator:
    """Streaming ``hat_g`` + ``hat_beta`` with an OOD alarm hook."""

    def __init__(
        self,
        eps: float = 1e-3,
        training_g_quantile: float = 0.99,
        p99: int | None = None,
    ) -> None:
        if eps < 0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if not (0.0 < training_g_quantile <= 1.0):
            raise ValueError(f"training_g_quantile must be in (0, 1], got {training_g_quantile}")
        self.eps = eps
        self.training_g_quantile = training_g_quantile
        self._training_g_values: list[int] = []
        self._p99 = p99
        self._alarms: list[OODAlarm] = []

    def fit(self, training_g_values: Iterable[int]) -> None:
        values = sorted(int(v) for v in training_g_values)
        if not values:
            self._p99 = 4
            return
        idx = min(len(values) - 1, int(self.training_g_quantile * (len(values) - 1)))
        self._p99 = int(values[idx])

    @property
    def p99_threshold(self) -> int:
        if self._p99 is None:
            return 4
        return self._p99

    def begin_case(self) -> "_CaseAccumulator":
        return _CaseAccumulator(self)

    def emit_alarm(self, alarm: OODAlarm) -> None:
        self._alarms.append(alarm)

    def drain_alarms(self) -> list[OODAlarm]:
        out = list(self._alarms)
        self._alarms.clear()
        return out


@dataclass(slots=True)
class _CaseAccumulator:
    estimator: GEstimator
    beta_sum: float = 0.0
    beta_count: int = 0
    _groups: set = field(default_factory=set)
    _anon_counter: int = 0

    def observe_edge(
        self,
        matcher_confidence: float,
        batch_group: object | None = None,
    ) -> None:
        # Clamp to [0, 1] defensively: a misbehaving matcher must not push
        # the certificate above the legitimate bound of 1.0.
        conf = max(0.0, min(1.0, float(matcher_confidence)))
        self.beta_sum += 1.0 - conf
        self.beta_count += 1
        # hat_g counts distinct correlated batch groups, not raw edges: edges
        # sharing a batch_group key (a transactional-entropy window) collapse to
        # one group. Edges with no key are treated as independent (own group).
        if batch_group is None:
            key: object = ("_anon", self._anon_counter)
            self._anon_counter += 1
        else:
            key = ("_batch", batch_group)
        self._groups.add(key)

    @property
    def hat_g(self) -> int:
        return len(self._groups)

    @property
    def hat_beta(self) -> float:
        if self.beta_count == 0:
            return 0.0
        return self.beta_sum / self.beta_count

    def finalize(self, case_id: str = "", timestamp: float = 0.0) -> CertificateResult:
        raw = self.hat_g * (self.hat_beta + self.estimator.eps)
        certificate = min(1.0, raw) if raw > 0 else 0.0
        ood = self.hat_g > self.estimator.p99_threshold
        if ood:
            self.estimator.emit_alarm(
                OODAlarm(
                    case_id=case_id,
                    hat_g=self.hat_g,
                    p99_threshold=self.estimator.p99_threshold,
                    timestamp=timestamp,
                )
            )
        return CertificateResult(
            hat_g=self.hat_g,
            hat_beta=self.hat_beta,
            certificate=certificate,
            ood=ood,
            p99_threshold=self.estimator.p99_threshold,
        )


def union_bound_static(g: int, beta: float) -> float:
    """Closed-form correlated-error union bound 1 - (1 - beta)^g."""
    if g < 0:
        raise ValueError(f"g must be >= 0, got {g}")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if g == 0:
        return 0.0
    return 1.0 - math.pow(max(0.0, 1.0 - beta), g)
