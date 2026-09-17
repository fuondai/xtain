"""Heterogeneous Graph Transformer (HGT) for cross-chain pseudonym resolution."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = query.shape
        q = self.q_linear(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if edge_type is not None:
            attn_weights = attn_weights + edge_type.unsqueeze(1)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_linear(output)


class HGTRelationUpdater(nn.Module):
    def __init__(self, embed_dim: int, num_relations: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_relations = num_relations
        self.edge_type_proj = nn.Linear(num_relations, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRUCell(input_size=embed_dim, hidden_size=embed_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_type_matrix: torch.Tensor,
    ) -> torch.Tensor:
        type_emb = self.edge_type_proj(edge_type_matrix)
        normed = self.norm(node_embeddings + type_emb)
        dropped = self.dropout(normed)
        return dropped


class HGTLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_relations: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.relation_updater = HGTRelationUpdater(embed_dim, num_relations, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_type_matrix: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.attention(node_embeddings, node_embeddings, node_embeddings, edge_type_matrix)
        updated = self.relation_updater(self.norm1(node_embeddings + attended), edge_type_matrix)
        ffn_out = self.ffn(self.norm2(updated))
        return updated + ffn_out


class HGTResolver(nn.Module):
    """Heterogeneous Graph Transformer for cross-chain pseudonym resolution."""

    def __init__(
        self,
        num_node_types: int = 8,
        num_relation_types: int = 16,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.node_type_embedding = nn.Embedding(num_node_types, embed_dim)
        self.rel_type_embedding = nn.Embedding(num_relation_types, embed_dim)
        self.layers = nn.ModuleList(
            [
                HGTLayer(embed_dim, num_heads, num_relation_types, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        # Persistent bilinear scoring head over the two node embeddings. Trained
        # jointly with the HGT layers; deterministic at inference (no per-call
        # random weights).
        self.score_head = nn.Bilinear(embed_dim, embed_dim, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = node_features
        x = x + self.node_type_embedding(node_type_ids)
        num_relations = self.rel_type_embedding.num_embeddings
        edge_type_emb = self.rel_type_embedding(
            edge_type_ids.clamp(0, num_relations - 1)
        )
        edge_type_matrix = edge_type_emb.sum(dim=1)

        for layer in self.layers:
            x = layer(x, edge_type_matrix)

        return self.dropout(x)

    def resolve_score(
        self,
        address_a: str,
        address_b: str,
        chain_a: str,
        chain_b: str,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
    ) -> float:
        self.eval()
        with torch.no_grad():
            a = features_a.reshape(1, -1).float()
            b = features_b.reshape(1, -1).float()
            score = torch.sigmoid(self.score_head(a, b))
        return float(score.item())


class HGTPretrainedEmbeddings(nn.Module):
    """Pretrained address embeddings for the HGT resolver."""

    def __init__(self, num_addresses: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_addresses, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, address_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(address_ids)


def build_hgt_input(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    chain_to_id: dict[str, int],
    edge_type_to_id: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_features_list: list[torch.Tensor] = []
    node_type_ids: list[int] = []

    for node in nodes:
        feat_dim = 64
        features = torch.zeros(feat_dim, dtype=torch.float32)
        features[0] = float(node.get("in_value_total", 0)) / 1e20
        features[1] = float(node.get("out_value_total", 0)) / 1e20
        features[2] = float(node.get("bridge_count", 0)) / 10.0
        features[3] = float(node.get("dex_swap_count", 0)) / 10.0
        chain_id = chain_to_id.get(node.get("chain", "unknown"), 0)
        node_type = str(node.get("node_type", "EOA"))
        node_type_id = _node_type_id(node_type)
        node_type_ids.append(node_type_id)
        node_features_list.append(features)

    node_features = torch.stack(node_features_list)
    node_type_tensor = torch.tensor(node_type_ids, dtype=torch.long)

    src_nodes = [edge["source"] for edge in edges]
    dst_nodes = [edge["target"] for edge in edges]
    edge_type_ids = [
        edge_type_to_id.get(str(edge.get("edge_type", "default")), 0)
        for edge in edges
    ]
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_type_tensor = torch.tensor(edge_type_ids, dtype=torch.long)

    return node_features, node_type_tensor, edge_index, edge_type_tensor


def _node_type_id(node_type: str) -> int:
    mapping = {"EOA": 0, "CONTRACT": 1, "EXCHANGE": 2, "BRIDGE": 3, "MIXER": 4, "CEX": 5, "UNKNOWN": 6}
    return mapping.get(node_type, 7)


def save_hgt_checkpoint(model: nn.Module, path: str, epoch: int, best_metric: float) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def load_hgt_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> tuple[int, float]:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    return state.get("epoch", 0), state.get("best_metric", 0.0)
