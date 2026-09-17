"""Azuma-Hoeffding suspect-set certificate for correlated batch mismatch.

Transactions on a public chain are heavily correlated by block-space demand,
gas economics, and arbitrage, so the standard Hoeffding bound (which requires
independence) is invalid. The mismatch sequence is instead modelled as a Doob
martingale over the graph filtration, and the per-case certificate is obtained
from the Azuma-Hoeffding inequality.

The bound's validity hinges on the bounded difference of a single batch event
being finite. A flash-loan adversary could attempt to inflate it with an
enormous single-block batch, but the EVM block gas limit caps the number of
state-modifying operations per block, so the influence any one block can have
on the empirical mismatch rate is bounded by protocol.
"""

from __future__ import annotations

import math

# Ethereum L1 block gas limit (G_max) and the minimum gas an observable bridge
# contract interaction costs (g_min). Their ratio caps the number of bridge
# events per block, which makes the Azuma bounded difference finite.
ETH_BLOCK_GAS_LIMIT = 30_000_000
MIN_BRIDGE_EVENT_GAS = 50_000


def max_bridge_events_per_block(
    g_max: int = ETH_BLOCK_GAS_LIMIT,
    g_min: int = MIN_BRIDGE_EVENT_GAS,
) -> int:
    """B_max = floor(G_max / g_min); ~600 at current mainnet parameters.

    This is the gas-limit constraint that keeps the martingale bounded
    difference finite against flash-loan-backed batch inflation.
    """
    if g_min <= 0:
        raise ValueError(f"g_min must be > 0, got {g_min}")
    return int(g_max // g_min)


def bounded_difference_C(
    n_samples: int,
    g_max: int = ETH_BLOCK_GAS_LIMIT,
    g_min: int = MIN_BRIDGE_EVENT_GAS,
) -> float:
    """Bounded difference of the empirical mismatch rate per revealed block.

    The Doob martingale reveals one block's events at a time. A single block
    holds at most ``B_max`` events, so revealing it can move the running mean
    over ``n_samples`` total observations by at most ``B_max / n_samples`` --
    a quantity on the rate scale [0, 1], finite by the block gas limit.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    b_max = max_bridge_events_per_block(g_max, g_min)
    return min(1.0, b_max / float(n_samples))


def azuma_radius(n_samples: int, c_bound: float, alpha: float = 0.05) -> float:
    """Azuma-Hoeffding finite-sample radius epsilon.

    From Pr[true_beta > beta_hat + epsilon] <= exp(-2 N epsilon^2 / C^2),
    setting the tail to ``alpha`` gives epsilon = C * sqrt(ln(1/alpha) / (2N)),
    where ``C`` is the rate-scale bounded difference from
    :func:`bounded_difference_C`.
    """
    if c_bound < 0:
        raise ValueError(f"c_bound must be >= 0, got {c_bound}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_samples <= 0:
        return 1.0
    return float(c_bound * math.sqrt(math.log(1.0 / alpha) / (2.0 * n_samples)))


def suspect_set_certificate(g_hat: int, beta_hat: float, epsilon: float) -> float:
    """Per-case upper certificate min(1, g_hat * (beta_hat + epsilon))."""
    g_hat = max(0, int(g_hat))
    beta_hat = min(1.0, max(0.0, float(beta_hat)))
    epsilon = max(0.0, float(epsilon))
    if g_hat == 0:
        return 0.0
    return float(min(1.0, g_hat * (beta_hat + epsilon)))


def case_certificate(
    g_hat: int,
    beta_hat: float,
    n_samples: int,
    alpha: float = 0.05,
) -> float:
    """End-to-end per-case certificate from the gas-limit-bounded Azuma radius.

    Derives the rate-scale bounded difference from the block gas limit, the
    Azuma-Hoeffding radius over ``n_samples`` observations, and the union over
    the ``g_hat`` correlated batch groups estimated for the case.
    """
    if n_samples <= 0:
        return 1.0
    c_bound = bounded_difference_C(n_samples)
    epsilon = azuma_radius(n_samples, c_bound, alpha)
    return suspect_set_certificate(g_hat, beta_hat, epsilon)
