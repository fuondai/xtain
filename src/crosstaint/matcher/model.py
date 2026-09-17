"""Bridge event matcher with transformer, bilinear, and MLP scoring variants."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class EventEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(x))


class BilinearScoringHead(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim, embed_dim, dtype=torch.float32) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sum(a.unsqueeze(1) * torch.t(self.W) @ b.unsqueeze(1).transpose(-1, -2), dim=-1) + self.bias


class BridgeEventMatcher(nn.Module):
    """Siamese transformer for matching source and destination bridge events."""

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        score_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.source_embed = EventEmbedding(embed_dim, embed_dim)
        self.dest_embed = EventEmbedding(embed_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.source_cls = nn.Parameter(torch.randn(embed_dim, dtype=torch.float32) * 0.02)
        self.dest_cls = nn.Parameter(torch.randn(embed_dim, dtype=torch.float32) * 0.02)
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim * 4, score_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden, 1),
        )

    def _encode(self, embed_module: nn.Module, x: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        cls = cls_token.view(1, 1, D).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoding(x)
        x = embed_module(x)
        encoded = self.encoder(x)
        return encoded[:, 0]

    def forward(
        self,
        source_features: torch.Tensor,
        dest_features: torch.Tensor,
    ) -> torch.Tensor:
        source_emb = self.source_embed(source_features)
        dest_emb = self.dest_embed(dest_features)
        source_repr = self._encode(self.source_embed, source_emb, self.source_cls)
        dest_repr = self._encode(self.dest_embed, dest_emb, self.dest_cls)
        combined = torch.cat(
            [
                source_repr,
                dest_repr,
                torch.abs(source_repr - dest_repr),
                source_repr * dest_repr,
            ],
            dim=-1,
        )
        return torch.sigmoid(self.fc_score(combined)).squeeze(-1)


class BilinearScoringMatcher(nn.Module):
    """Bilinear matcher variant: score = sigmoid(x^T W y + b)."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim, embed_dim, dtype=torch.float32) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(
        self,
        source_features: torch.Tensor,
        dest_features: torch.Tensor,
    ) -> torch.Tensor:
        source_flat = source_features.mean(dim=1) if source_features.dim() == 3 else source_features
        dest_flat = dest_features.mean(dim=1) if dest_features.dim() == 3 else dest_features
        score = torch.sum(source_flat.unsqueeze(1) * torch.t(self.W) @ dest_flat.unsqueeze(1).transpose(-1, -2), dim=-1)
        return torch.sigmoid(score + self.bias)


class SiameseMLPMatcher(nn.Module):
    """MLP Siamese matcher using cosine similarity for comparison."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        source_features: torch.Tensor,
        dest_features: torch.Tensor,
    ) -> torch.Tensor:
        source_flat = source_features.mean(dim=1) if source_features.dim() == 3 else source_features
        dest_flat = dest_features.mean(dim=1) if dest_features.dim() == 3 else dest_features
        source_enc = self.encoder(source_flat)
        dest_enc = self.encoder(dest_flat)
        return (self.cos_sim(source_enc, dest_enc).unsqueeze(-1) + 1.0) / 2.0


def load_matcher_checkpoint(
    path: str,
    device: str = "cpu",
) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=True)
    return state


def save_matcher_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    path: str,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )
