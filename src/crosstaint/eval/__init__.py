from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb, log
from typing import Callable

import numpy as np
import scipy.stats

from crosstaint.types import (
    EvalMetrics,
    BootstrapCI,
    WilcoxonResult,
    BoundEstimate,
)


def compute_hop_recall(
    predicted_paths: list[list[str]],
    true_paths: list[list[str]],
) -> float:
    true_addresses = {
        address.lower()
        for path in true_paths
        for address in path
    }
    predicted_addresses = {
        address.lower()
        for path in predicted_paths
        for address in path
    }

    if not true_addresses:
        return 1.0 if not predicted_addresses else 0.0

    return len(true_addresses & predicted_addresses) / len(true_addresses)


def compute_precision(
    predicted_paths: list[list[str]],
    true_paths: list[list[str]],
) -> float:
    predicted_addresses = {
        address.lower()
        for path in predicted_paths
        for address in path
    }
    true_addresses = {
        address.lower()
        for path in true_paths
        for address in path
    }

    if not predicted_addresses:
        return 1.0 if not true_addresses else 0.0

    return len(predicted_addresses & true_addresses) / len(predicted_addresses)


def compute_false_positive_rate(
    predicted_set: list[str],
    benign_set: list[str],
) -> float:
    benign_lower = {addr.lower() for addr in benign_set}
    predicted_lower = {addr.lower() for addr in predicted_set}

    if not benign_lower:
        return 0.0

    if not predicted_lower:
        return 0.0

    false_positives = len(predicted_lower & benign_lower)
    return false_positives / len(benign_lower)


def compute_f1(recall: float, precision: float) -> float:
    if recall + precision == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_bootstrap_ci(
    metric_fn: Callable,
    cases: list,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    rng = np.random.default_rng(seed)

    if not cases:
        return BootstrapCI(
            metric_name="unknown",
            mean=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            n_resamples=n_resamples,
            alpha=alpha,
        )

    metric_name = getattr(metric_fn, "__name__", "metric")
    resampled_metrics: list[float] = []

    for _ in range(n_resamples):
        indices = rng.integers(0, len(cases), size=len(cases))
        resampled_cases = [cases[i] for i in indices]

        try:
            metric_value = metric_fn(resampled_cases)
            if metric_value is not None and not np.isnan(metric_value):
                resampled_metrics.append(float(metric_value))
        except Exception:
            continue

    if not resampled_metrics:
        return BootstrapCI(
            metric_name=metric_name,
            mean=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            n_resamples=n_resamples,
            alpha=alpha,
        )

    resampled_metrics = np.array(resampled_metrics)
    mean = float(np.mean(resampled_metrics))
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    ci_lower = float(np.percentile(resampled_metrics, lower_percentile))
    ci_upper = float(np.percentile(resampled_metrics, upper_percentile))

    return BootstrapCI(
        metric_name=metric_name,
        mean=mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_resamples=n_resamples,
        alpha=alpha,
    )


def compute_wilcoxon(
    metrics_a: list[float],
    metrics_b: list[float],
    alternative: str = "greater",
) -> WilcoxonResult:
    if len(metrics_a) != len(metrics_b):
        min_len = min(len(metrics_a), len(metrics_b))
        metrics_a = metrics_a[:min_len]
        metrics_b = metrics_b[:min_len]

    if len(metrics_a) == 0:
        return WilcoxonResult(
            method_a="A",
            method_b="B",
            statistic=0.0,
            p_value=1.0,
            n_pairs=0,
            significant=False,
        )

    diff = np.array(metrics_a) - np.array(metrics_b)
    nonzero_diffs = diff != 0
    if not np.any(nonzero_diffs):
        return WilcoxonResult(
            method_a="A",
            method_b="B",
            statistic=0.0,
            p_value=1.0,
            n_pairs=len(diff),
            significant=False,
        )

    try:
        statistic, p_value = scipy.stats.wilcoxon(
            diff,
            alternative=alternative,
        )
        significant = p_value < 0.05
    except Exception:
        statistic = 0.0
        p_value = 1.0
        significant = False

    return WilcoxonResult(
        method_a="MethodA",
        method_b="MethodB",
        statistic=float(statistic),
        p_value=float(p_value),
        n_pairs=len(metrics_a),
        significant=significant,
    )


def compute_bound_estimate(
    edges: list[tuple] | list[dict],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rho: float = 0.81,
    threshold: float = 0.04,
    max_hops: int = 11,
    seed: int = 42,
) -> BoundEstimate:
    observations = _normalise_bound_observations(edges)
    if not observations:
        return BoundEstimate(
            delta_hat=0.0,
            delta_ci_lower=0.0,
            delta_ci_upper=1.0,
            beta_hat=0.0,
            beta_ci_lower=0.0,
            beta_ci_upper=1.0,
            n_edges=0,
            n_resamples=n_resamples,
            predicted_offtrace_rate_independent=0.0,
            observed_offtrace_rate=0.0,
            n_batch_groups=None,
        )

    delta_hat, beta_hat, observed_rate = _fit_bounded_mismatch(observations)
    delta_samples: list[float] = []
    beta_samples: list[float] = []

    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        indices = rng.integers(0, len(observations), size=len(observations))
        sample = [observations[i] for i in indices]
        sample_delta, sample_beta, _ = _fit_bounded_mismatch(sample)
        delta_samples.append(sample_delta)
        beta_samples.append(sample_beta)

    delta_ci_lower = float(np.percentile(delta_samples, alpha / 2 * 100))
    delta_ci_upper = float(np.percentile(delta_samples, (1 - alpha / 2) * 100))
    beta_ci_lower = float(np.percentile(beta_samples, alpha / 2 * 100))
    beta_ci_upper = float(np.percentile(beta_samples, (1 - alpha / 2) * 100))

    j_star = _threshold_crossing_depth(rho, delta_hat, threshold, max_hops)
    independent_rate = _binomial_tail(max_hops, j_star, beta_hat)
    batch_groups = {
        obs["batch_group"]
        for obs in observations
        if obs.get("batch_group") is not None
    }

    return BoundEstimate(
        delta_hat=delta_hat,
        delta_ci_lower=delta_ci_lower,
        delta_ci_upper=delta_ci_upper,
        beta_hat=beta_hat,
        beta_ci_lower=beta_ci_lower,
        beta_ci_upper=beta_ci_upper,
        n_edges=len(observations),
        n_resamples=n_resamples,
        predicted_offtrace_rate_independent=independent_rate,
        observed_offtrace_rate=observed_rate,
        n_batch_groups=len(batch_groups) if batch_groups else None,
    )


def _normalise_bound_observations(edges: list[tuple] | list[dict]) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for item in edges:
        if isinstance(item, dict):
            if "score" not in item or "label" not in item:
                raise ValueError("bound observation dictionaries require score and label")
            observations.append({
                "score": float(item["score"]),
                "label": bool(item["label"]),
                "batch_group": item.get("batch_group"),
            })
            continue
        if len(item) < 2:
            raise ValueError("bound observation tuples require at least two scores")
        observations.append({"score": float(item[0]), "label": True, "batch_group": None})
        observations.append({"score": float(item[1]), "label": False, "batch_group": None})
    return observations


def _fit_bounded_mismatch(observations: list[dict[str, object]]) -> tuple[float, float, float]:
    positives = [float(obs["score"]) for obs in observations if bool(obs["label"])]
    negatives = [float(obs["score"]) for obs in observations if not bool(obs["label"])]
    if not positives or not negatives:
        return 0.0, 1.0, 0.0

    best_delta = 0.49
    best_beta = 1.0
    for delta in np.linspace(0.01, 0.49, 49):
        false_positive_rate = sum(score > delta for score in negatives) / len(negatives)
        false_negative_rate = sum(score < 1.0 - delta for score in positives) / len(positives)
        beta = max(false_positive_rate, false_negative_rate)
        if beta < best_beta or (beta == best_beta and delta < best_delta):
            best_delta = float(delta)
            best_beta = float(beta)

    observed_offtrace = sum(score > best_delta for score in negatives) / len(negatives)
    return best_delta, best_beta, float(observed_offtrace)


def _threshold_crossing_depth(rho: float, delta: float, threshold: float, max_hops: int) -> int:
    if not (0.0 < rho <= 1.0) or not (0.0 < delta < 1.0) or not (0.0 < threshold < 1.0):
        return 1
    if max_hops < 1:
        return 1
    # Minimum number of off-trace edges that must jointly mismatch for an
    # off-trace path of length ``max_hops`` to reach the acceptance threshold.
    # A path with m mismatched edges admits the score bound rho^k * delta^(k-m);
    # requiring rho^k * delta^(k-m) > threshold and solving for m gives the count
    # below. This couples the crossing count to the path length, unlike a bound
    # that depends on delta alone.
    l_eta = -log(threshold)
    l_rho = -log(rho) if rho < 1.0 else 0.0
    l_delta = -log(delta)
    if l_delta <= 0.0:
        return max_hops
    count = max_hops - (l_eta - max_hops * l_rho) / l_delta
    return int(min(max_hops, max(1, ceil(count))))


def _binomial_tail(k: int, j_star: int, beta: float) -> float:
    if j_star > k:
        return 0.0
    beta = float(min(1.0, max(0.0, beta)))
    return float(sum(
        comb(k, j) * (beta ** j) * ((1.0 - beta) ** (k - j))
        for j in range(j_star, k + 1)
    ))


def finite_sample_radius(
    n_samples: int,
    alpha: float = 0.05,
    c_bound: float = 1.0,
) -> float:
    # Azuma-Hoeffding finite-sample radius epsilon = C * sqrt(ln(1/alpha) / (2n)),
    # the additive uncertainty on the bounded-mismatch estimate under the Doob
    # martingale filtration. C is the bounded difference of a single batch event,
    # finite by the block gas limit (see propagation.certificate). With C = 1 this
    # reduces to the classical Hoeffding radius for the worst-case unit influence.
    from crosstaint.propagation.certificate import azuma_radius

    return azuma_radius(n_samples, c_bound=c_bound, alpha=alpha)


def compute_suspect_set_certificate(
    beta_hat: float,
    n_samples: int,
    g_hat: int = 1,
    alpha: float = 0.05,
    c_bound: float = 1.0,
) -> float:
    # Per-case adaptive suspect-set certificate min(1, g_hat * (beta_hat + epsilon)).
    # With g_hat = 1 this reduces to the per-group independence ceiling; it widens
    # linearly with the number of correlated batch groups g_hat estimated for the
    # case, matching the correlated-error union bound. The finite-sample radius
    # folds in the Azuma-Hoeffding estimation error of beta.
    beta_hat = float(min(1.0, max(0.0, beta_hat)))
    g_hat = max(1, int(g_hat))
    epsilon = finite_sample_radius(n_samples, alpha, c_bound)
    return float(min(1.0, g_hat * (beta_hat + epsilon)))


def compute_synthetic_vs_real_fpr(
    predicted_set: list[str],
    real_address_holdout: list[str],
    synthetic_benign: list[str],
) -> dict[str, float]:
    """Side-by-side FPR on the real-address holdout and the synthetic pool.

    A single FPR number mixes two different sources of false positives.
    Reporting them separately reveals whether a low headline FPR is hiding a
    high synthetic-pool FPR (over-fit to the synthetic distribution) or vice
    versa. Returns both numbers plus a combined figure.
    """
    real = {a.lower() for a in real_address_holdout}
    synth = {a.lower() for a in synthetic_benign}
    predicted = {a.lower() for a in predicted_set}
    if not real and not synth:
        return {
            "fpr_real": 0.0,
            "fpr_synthetic": 0.0,
            "fpr_combined": 0.0,
            "real_count": 0,
            "synthetic_count": 0,
        }

    real_overlap = len(predicted & real) / len(real) if real else 0.0
    synth_overlap = len(predicted & synth) / len(synth) if synth else 0.0
    combined_denom = len(real) + len(synth)
    combined = (len(predicted & real) + len(predicted & synth)) / combined_denom
    return {
        "fpr_real": float(real_overlap),
        "fpr_synthetic": float(synth_overlap),
        "fpr_combined": float(combined),
        "real_count": len(real),
        "synthetic_count": len(synth),
    }


def compute_aggregate_metrics(
    results: list[EvalMetrics],
) -> dict[str, float]:
    if not results:
        return {}

    metrics = [
        "hop_recall",
        "precision",
        "false_positive_rate",
        "true_positive",
        "false_positive",
        "false_negative",
        "runtime_median_ms",
        "runtime_p95_ms",
    ]

    aggregated: dict[str, float] = {}

    for metric in metrics:
        values = []
        for result in results:
            value = getattr(result, metric, None)
            if value is not None:
                values.append(float(value))

        if values:
            aggregated[f"{metric}_mean"] = float(np.mean(values))
            aggregated[f"{metric}_median"] = float(np.median(values))
            aggregated[f"{metric}_std"] = float(np.std(values))
            aggregated[f"{metric}_min"] = float(np.min(values))
            aggregated[f"{metric}_max"] = float(np.max(values))

    aggregated["num_runs"] = len(results)
    aggregated["total_cases"] = sum(r.num_cases for r in results)

    return aggregated
