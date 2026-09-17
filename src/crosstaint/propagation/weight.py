from __future__ import annotations

import torch
import torch.nn as nn

from crosstaint.types import EdgeType, IREdge


class EdgeWeightLearner:
    DEFAULT_WEIGHTS: dict[str, float] = {
        EdgeType.INTRA_CHAIN_TRANSFER: 1.00,
        EdgeType.CROSS_CHAIN_BRIDGE: 0.90,
        EdgeType.DEX_SWAP: 0.80,
    }

    def __init__(self, trainable: bool = False) -> None:
        self.trainable = trainable

        if trainable:
            self._mlp = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid(),
            )
            base = torch.tensor(
                [self.DEFAULT_WEIGHTS.get(EdgeType.INTRA_CHAIN_TRANSFER, 1.0)],
                dtype=torch.float32,
            )
            with torch.no_grad():
                self._mlp[2].bias.fill_(torch.log(base / (1 - base + 1e-8)).item())
        else:
            self._mlp = None

    def get_weight(self, edge: IREdge) -> float:
        return self.get_weight_by_type(edge.edge_type)

    def get_weight_by_type(self, edge_type: str) -> float:
        if self._mlp is not None and self.trainable:
            weight = self._mlp(torch.tensor([[self.DEFAULT_WEIGHTS.get(edge_type, 0.9)]]))
            return float(weight.item())
        return self.DEFAULT_WEIGHTS.get(edge_type, 0.9)
