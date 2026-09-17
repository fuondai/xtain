from __future__ import annotations

import bisect
import math
from typing import Callable, Optional, Sequence

from crosstaint.types import EdgeType


class DecayScheduler:
    def __init__(self, rho: float = 0.81) -> None:
        if not (0.0 < rho <= 1.0):
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        self.rho = rho

    def decay(self, hop_distance: int) -> float:
        if hop_distance < 0:
            raise ValueError(f"hop_distance must be non-negative, got {hop_distance}")
        if hop_distance == 0:
            return 1.0
        return self.rho ** hop_distance

    def decay_with_value(
        self,
        value: int,
        hop: int,
        initial_value: int,
    ) -> int:
        decayed = initial_value * (self.rho ** hop)
        return max(0, int(decayed))


def calibrate_gamma(alpha_fn: float, sigma_f: float, z_alpha: float) -> float:
    """Penalty sensitivity gamma = ln(1/alpha_fn) / (z_alpha * sigma_f).

    Derived so a natural fee deviation at the ``(1 - alpha_fn)`` tail of the
    calibration slippage distribution (``z_alpha * sigma_f``) is penalised by at
    most a factor ``alpha_fn``; deviations beyond that tail are penalised
    exponentially harder. With alpha_fn=0.01, sigma_f=0.018, and z_alpha=2.33,
    this closed-form initializer yields gamma ~= 109.8. Reported experiments use
    the explicit calibrated value in ``config/propagation.yaml``.
    """
    if not (0.0 < alpha_fn < 1.0):
        raise ValueError(f"alpha_fn must be in (0, 1), got {alpha_fn}")
    if sigma_f <= 0.0 or z_alpha <= 0.0:
        raise ValueError("sigma_f and z_alpha must be positive")
    return math.log(1.0 / alpha_fn) / (z_alpha * sigma_f)


def calibrate_gamma_from_samples(
    fee_deviations: Sequence[float], alpha_fn: float
) -> float:
    """Calibrate gamma from observed natural fee deviations above baseline.

    ``gamma = ln(1/alpha_fn) / q`` where ``q`` is the empirical
    ``(1 - alpha_fn)`` quantile of the deviations. Equivalent to
    :func:`calibrate_gamma` with ``z_alpha * sigma_f = q`` but makes no Gaussian
    assumption, which matters for the heavy-tailed mainnet fee distribution.
    """
    if not (0.0 < alpha_fn < 1.0):
        raise ValueError(f"alpha_fn must be in (0, 1), got {alpha_fn}")
    deviations = sorted(max(0.0, float(d)) for d in fee_deviations)
    if not deviations:
        raise ValueError("fee_deviations must be non-empty")
    idx = min(len(deviations) - 1, int((1.0 - alpha_fn) * (len(deviations) - 1)))
    q = deviations[idx]
    if q <= 0.0:
        raise ValueError("calibration tail quantile collapsed to zero")
    return math.log(1.0 / alpha_fn) / q


class MEVResistantDecay:
    """Adaptive decay rho(t) = rho0 * exp(-gamma * max(0, f_max(t) - f0)).

    Falls back to ``rho0`` whenever the observed fee-plus-slippage bound stays
    within the baseline ``f0`` (legitimate congestion is not penalised);
    deviations above ``f0`` (MEV sandwiches, flash-loan slippage manipulation)
    are penalised exponentially with calibrated sensitivity ``gamma``.
    """

    def __init__(self, rho0: float, gamma: float, f0: float) -> None:
        if not (0.0 < rho0 <= 1.0):
            raise ValueError(f"rho0 must be in (0, 1], got {rho0}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if f0 < 0.0:
            raise ValueError(f"f0 must be >= 0, got {f0}")
        self.rho0 = rho0
        self.gamma = gamma
        self.f0 = f0

    def rho(self, f_max_t: float, f0: float | None = None) -> float:
        baseline = self.f0 if f0 is None else max(self.f0, float(f0))
        excess = max(0.0, float(f_max_t) - baseline)
        return self.rho0 * math.exp(-self.gamma * excess)


class AutoTuner:
    def __init__(
        self,
        rho_range: tuple[float, float] = (0.70, 0.90),
        target_fp_rate: float = 0.05,
        max_steps: int = 8,
        eval_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        self.rho_range = rho_range
        self.target_fp_rate = target_fp_rate
        self.max_steps = max_steps
        self._eval_fn = eval_fn

        self._current_rho = (rho_range[0] + rho_range[1]) / 2.0
        self._best_rho = self._current_rho
        self._best_diff = float("inf")

    def suggest_rho(self) -> float:
        return self._current_rho

    def tune(
        self,
        eval_fn: Optional[Callable[[float], float]] = None,
    ) -> float:
        fn = eval_fn if eval_fn is not None else self._eval_fn
        if fn is None:
            raise ValueError("eval_fn must be provided either at init or at tune()")

        lo, hi = self.rho_range

        for _ in range(self.max_steps):
            mid = (lo + hi) / 2.0
            fpr = fn(mid)
            diff = abs(fpr - self.target_fp_rate)

            if diff < 0.01:
                self._current_rho = mid
                self._best_rho = mid
                self._best_diff = diff
                return mid

            if fpr > self.target_fp_rate:
                hi = mid
            else:
                lo = mid

            if diff < self._best_diff:
                self._best_diff = diff
                self._best_rho = mid

        self._current_rho = self._best_rho
        return self._best_rho
