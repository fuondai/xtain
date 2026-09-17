"""Domain-adversarial neural network (DANN) for mainnet domain adaptation.

Section III.E-F: HGT resolvers trained purely on synthetic traces suffer
covariate shift because the marginal feature distributions of synthetic and
mainnet exploit traces differ (``P_s(X) != P_t(X)``). The feature extractor is
trained to be simultaneously discriminative for taint propagation and invariant
to the domain origin, via a minimax game against a domain discriminator
connected through a gradient-reversal layer.

The objective (Eq. 3) is

    L(theta_f, theta_y, theta_d) = sum_{x in D_s} L_y(G_y(G_f(x)), y)
                                 - lambda * sum_{x in D_s u D_t} L_d(G_d(G_f(x)), d)

optimised at the saddle point: minimise over (theta_f, theta_y), maximise over
theta_d. The gradient-reversal layer implements the sign flip so a single
backward pass realises both directions.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversalFn(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Reverse the gradient so the feature extractor maximises domain
        # confusion while the discriminator minimises domain loss.
        return grad_output.neg() * ctx.lambda_, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lambda_)


def dann_lambda(progress: float, gamma_s: float = 10.0) -> float:
    """Sigmoidal annealing schedule lambda_p = 2 / (1 + exp(-gamma_s * p)) - 1.

    Ramps the domain-confusion weight from 0 to 1 over training so the feature
    extractor first learns discriminative taint representations before the
    domain-confusion gradient is introduced, preventing premature collapse of
    the classification boundary under heavy class imbalance.
    """
    p = min(1.0, max(0.0, float(progress)))
    return 2.0 / (1.0 + math.exp(-gamma_s * p)) - 1.0


class DomainDiscriminator(nn.Module):
    """Classifies whether an embedding came from the synthetic or mainnet domain."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        reversed_features = gradient_reversal(features, lambda_)
        return self.net(reversed_features)


class DANNObjective(nn.Module):
    """Combined taint-classification and domain-confusion loss (Eq. 3).

    ``taint_logits`` and ``taint_labels`` cover the labelled synthetic source
    domain only; ``domain_logits`` and ``domain_labels`` cover both domains
    (source label 0, mainnet target label 1).
    """

    def __init__(self) -> None:
        super().__init__()
        self._taint_loss = nn.BCEWithLogitsLoss()
        self._domain_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        taint_logits: torch.Tensor,
        taint_labels: torch.Tensor,
        domain_logits: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The gradient-reversal layer already negates the domain gradient w.r.t.
        # the feature extractor, so the total loss is a straight sum: minimising
        # it minimises taint loss and (through the reversal) maximises domain
        # confusion in the extractor while the discriminator minimises its loss.
        l_y = self._taint_loss(taint_logits, taint_labels)
        l_d = self._domain_loss(domain_logits, domain_labels)
        return l_y + l_d, l_y, l_d
