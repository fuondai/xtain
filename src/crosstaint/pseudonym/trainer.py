"""HGT pseudonym resolver trainer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from crosstaint.pseudonym.model import (
    HGTResolver,
    build_hgt_input,
    save_hgt_checkpoint,
    load_hgt_checkpoint,
)


logger = logging.getLogger(__name__)


class PseudonymPairDataset(Dataset):
    def __init__(
        self,
        node_features: list[torch.Tensor],
        node_type_ids: list[torch.Tensor],
        edge_indices: list[torch.Tensor],
        edge_type_ids: list[torch.Tensor],
        labels: list[int],
    ) -> None:
        self.node_features = node_features
        self.node_type_ids = node_type_ids
        self.edge_indices = edge_indices
        self.edge_type_ids = edge_type_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.node_features[idx],
            self.node_type_ids[idx],
            self.edge_indices[idx],
            self.edge_type_ids[idx],
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class HGTrainer:
    def __init__(
        self,
        model: HGTResolver,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        self._model = model.to(device)
        self._device = device
        self._optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self._bce_loss = nn.BCELoss()
        self._epoch = 0
        self._best_metric = 0.0

    def train_epoch(self, dataloader: DataLoader) -> float:
        self._model.train()
        total_loss = 0.0
        n_batches = 0

        for node_feat, node_type, edge_idx, edge_type, labels in dataloader:
            node_feat = node_feat.to(self._device)
            node_type = node_type.to(self._device)
            edge_idx = edge_idx.to(self._device)
            edge_type = edge_type.to(self._device)
            labels = labels.to(self._device)

            self._optimizer.zero_grad()
            embeddings = self._model(node_feat, node_type, edge_idx, edge_type)
            pooled = embeddings.mean(dim=1) if embeddings.dim() > 2 else embeddings
            score = torch.sigmoid(pooled[:, 0:1].mean())
            loss = self._bce_loss(score.unsqueeze(0), labels.unsqueeze(0))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        self._epoch += 1
        return total_loss / max(n_batches, 1)

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self._model.eval()
        all_scores: list[float] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for node_feat, node_type, edge_idx, edge_type, labels in dataloader:
                node_feat = node_feat.to(self._device)
                node_type = node_type.to(self._device)
                edge_idx = edge_idx.to(self._device)
                edge_type = edge_type.to(self._device)

                embeddings = self._model(node_feat, node_type, edge_idx, edge_type)
                pooled = embeddings.mean(dim=1) if embeddings.dim() > 2 else embeddings
                score = float(torch.sigmoid(pooled[0:1, 0:1].mean()).item())
                all_scores.append(score)
                all_labels.append(int(labels[0].item()))

        if not all_scores:
            return {"loss": 0.0, "accuracy": 0.0}

        predictions = [(1.0 if s >= 0.5 else 0.0) for s in all_scores]
        accuracy = sum(int(p == l) for p, l in zip(predictions, all_labels)) / len(all_labels)
        return {"accuracy": float(accuracy)}

    def checkpoint(self, path: str, metric: float | None = None) -> None:
        if metric is not None and metric > self._best_metric:
            self._best_metric = metric
        save_hgt_checkpoint(self._model, path, self._epoch, self._best_metric)
        logger.info("HGT checkpoint saved to %s (epoch=%s best=%.4f)", path, self._epoch, self._best_metric)

    def load(self, path: str) -> None:
        epoch, best_metric = load_hgt_checkpoint(self._model, path, device=self._device)
        self._epoch = epoch
        self._best_metric = best_metric
        logger.info("HGT checkpoint loaded from %s (epoch=%s best=%.4f)", path, self._epoch, self._best_metric)


def train_hgt_resolver(
    train_data: list[tuple[dict[str, Any], dict[str, Any], int]],
    output_dir: str = "./pseudonym_checkpoints",
    epochs: int = 20,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 42,
) -> HGTResolver:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HGTResolver(embed_dim=64, num_heads=4, num_layers=2)
    trainer = HGTrainer(model, device=device)

    chain_to_id = {"ethereum": 0, "bsc": 1, "polygon": 2, "arbitrum": 3, "avalanche": 4, "avalanche-c": 5}
    edge_type_to_id = {"INTRA_CHAIN_TRANSFER": 0, "CROSS_CHAIN_BRIDGE": 1, "DEX_SWAP": 2}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(output_path / "hgt_resolver_best.pt")

    for epoch in range(epochs):
        loss = trainer.train_epoch(DataLoader(train_data, batch_size=batch_size, shuffle=True))
        metrics = trainer.evaluate(DataLoader(train_data, batch_size=batch_size, shuffle=False))
        logger.info("HGT Epoch %s/%s loss=%.4f accuracy=%.4f", epoch + 1, epochs, loss, metrics.get("accuracy", 0.0))
        trainer.checkpoint(checkpoint_path, metric=metrics.get("accuracy", 0.0))

    return model
