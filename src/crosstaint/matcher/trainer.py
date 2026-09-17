"""Bridge event matcher trainer with hard negative mining, BCE and triplet loss, and checkpointing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from crosstaint.matcher.featurizer import EventFeaturizer
from crosstaint.matcher.model import (
    BridgeEventMatcher,
    BilinearScoringMatcher,
    SiameseMLPMatcher,
    load_matcher_checkpoint,
    save_matcher_checkpoint,
)
from crosstaint.types import DecodedEvent


logger = logging.getLogger(__name__)


class EventPairDataset(Dataset):
    def __init__(
        self,
        source_features: list[np.ndarray],
        dest_features: list[np.ndarray],
        labels: list[int],
    ) -> None:
        self.source_features = [torch.from_numpy(f).float() for f in source_features]
        self.dest_features = [torch.from_numpy(f).float() for f in dest_features]
        self.labels = [torch.tensor(l, dtype=torch.float32) for l in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.source_features[idx], self.dest_features[idx], self.labels[idx]


class HardNegativeMiner:
    def __init__(self, featurizer: EventFeaturizer, matcher: nn.Module, batch_size: int = 64) -> None:
        self._featurizer = featurizer
        self._matcher = matcher
        self._batch_size = batch_size

    def mine(
        self,
        positive_pairs: list[tuple[DecodedEvent, DecodedEvent]],
        candidate_events: list[DecodedEvent],
        top_k: int = 5,
    ) -> list[tuple[DecodedEvent, DecodedEvent]]:
        if not positive_pairs or not candidate_events:
            return []

        self._matcher.eval()
        candidate_features = [
            self._featurizer.featurize(event)[np.newaxis, :]
            for event in candidate_events
        ]

        hard_negatives: list[tuple[DecodedEvent, DecodedEvent]] = []
        for source_event, _ in positive_pairs[: min(len(positive_pairs), self._batch_size)]:
            source_feat = self._featurizer.featurize(source_event)[np.newaxis, :]
            scores: list[tuple[float, DecodedEvent]] = []
            for i, cand_feat in enumerate(candidate_features):
                score = self._score_pair(source_feat, cand_feat)
                scores.append((float(score), candidate_events[i]))

            scores.sort(key=lambda x: x[0], reverse=True)
            for score_val, cand_event in scores[:top_k]:
                if score_val < 0.95:
                    hard_negatives.append((source_event, cand_event))

        return hard_negatives

    def _score_pair(self, source: np.ndarray, dest: np.ndarray) -> float:
        with torch.no_grad():
            source_t = torch.from_numpy(source.astype(np.float32))
            dest_t = torch.from_numpy(dest.astype(np.float32))
            return float(self._matcher(source_t, dest_t).item())


class MatcherTrainer:
    def __init__(
        self,
        model: BridgeEventMatcher | BilinearScoringMatcher | SiameseMLPMatcher,
        featurizer: EventFeaturizer,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        self._model = model.to(device)
        self._featurizer = featurizer
        self._device = device
        self._optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self._bce_loss = nn.BCELoss()
        self._epoch = 0
        self._best_metric = 0.0

    def train_epoch(
        self,
        dataloader: DataLoader,
        triplet_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    ) -> float:
        self._model.train()
        total_loss = 0.0
        n_batches = 0

        for source_batch, dest_batch, labels_batch in dataloader:
            source_batch = source_batch.to(self._device)
            dest_batch = dest_batch.to(self._device)
            labels_batch = labels_batch.to(self._device)

            self._optimizer.zero_grad()
            scores = self._model(source_batch, dest_batch)
            loss = self._bce_loss(scores, labels_batch)

            if triplet_pairs is not None and len(triplet_pairs) > 0:
                triplet_loss = self._triplet_loss(triplet_pairs[: len(source_batch)])
                loss = loss + 0.1 * triplet_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        self._epoch += 1
        return total_loss / max(n_batches, 1)

    def _triplet_loss(
        self,
        triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        margin: float = 0.2,
    ) -> torch.Tensor:
        if not triplets:
            return torch.tensor(0.0, device=self._device)

        positive_scores: list[float] = []
        negative_scores: list[float] = []
        for anchor, positive, negative in triplets:
            anchor_t = torch.from_numpy(anchor[np.newaxis, :].astype(np.float32)).to(self._device)
            positive_t = torch.from_numpy(positive[np.newaxis, :].astype(np.float32)).to(self._device)
            negative_t = torch.from_numpy(negative[np.newaxis, :].astype(np.float32)).to(self._device)
            pos_score = float(self._model(anchor_t, positive_t).item())
            neg_score = float(self._model(anchor_t, negative_t).item())
            positive_scores.append(pos_score)
            negative_scores.append(neg_score)

        if not positive_scores or not negative_scores:
            return torch.tensor(0.0, device=self._device)

        max_pos = max(positive_scores)
        max_neg = max(negative_scores)
        loss_val = max(0.0, max_pos - max_neg + margin)
        return torch.tensor(loss_val, device=self._device, requires_grad=True)

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self._model.eval()
        all_scores: list[float] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for source_batch, dest_batch, labels_batch in dataloader:
                source_batch = source_batch.to(self._device)
                dest_batch = dest_batch.to(self._device)
                labels_batch = labels_batch.to(self._device)
                scores = self._model(source_batch, dest_batch)
                all_scores.extend(scores.cpu().tolist())
                all_labels.extend(labels_batch.int().cpu().tolist())

        if not all_scores:
            return {"loss": 0.0, "auc": 0.0, "accuracy": 0.0}

        all_scores_arr = np.array(all_scores)
        all_labels_arr = np.array(all_labels)
        predictions = (all_scores_arr >= 0.5).astype(int)
        accuracy = float(np.mean(predictions == all_labels_arr))

        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(all_labels_arr, all_scores_arr))
        except Exception:
            auc = 0.0

        return {"accuracy": accuracy, "auc": auc}

    def checkpoint(self, path: str, metric: float | None = None) -> None:
        if metric is not None and metric > self._best_metric:
            self._best_metric = metric
        save_matcher_checkpoint(
            self._model,
            self._optimizer,
            self._epoch,
            self._best_metric,
            path,
        )
        logger.info("Checkpoint saved to %s (epoch=%s best=%.4f)", path, self._epoch, self._best_metric)

    def load(self, path: str) -> None:
        state = load_matcher_checkpoint(path, device=self._device)
        self._model.load_state_dict(state["model_state"])
        self._optimizer.load_state_dict(state["optimizer_state"])
        self._epoch = state.get("epoch", 0)
        self._best_metric = state.get("best_metric", 0.0)
        logger.info("Checkpoint loaded from %s (epoch=%s best=%.4f)", path, self._epoch, self._best_metric)


def build_dataloader(
    source_features: list[np.ndarray],
    dest_features: list[np.ndarray],
    labels: list[int],
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    dataset = EventPairDataset(source_features, dest_features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_bridge_matcher(
    train_pairs: list[tuple[DecodedEvent, DecodedEvent, int]],
    val_pairs: list[tuple[DecodedEvent, DecodedEvent, int]] | None = None,
    output_dir: str = "./matcher_checkpoints",
    epochs: int = 20,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 42,
) -> BridgeEventMatcher:
    torch.manual_seed(seed)
    np.random.seed(seed)
    featurizer = EventFeaturizer(seed=seed)
    model = BridgeEventMatcher(embed_dim=64, num_heads=4, num_layers=2)
    trainer = MatcherTrainer(model, featurizer, device=device)

    train_source = [featurizer.featurize(src) for src, _, _ in train_pairs]
    train_dest = [featurizer.featurize(dst) for _, dst, _ in train_pairs]
    train_labels = [label for _, _, label in train_pairs]
    train_loader = build_dataloader(train_source, train_dest, train_labels, batch_size=batch_size)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(output_path / "matcher_best.pt")

    for epoch in range(epochs):
        loss = trainer.train_epoch(train_loader)
        metrics = trainer.evaluate(train_loader)
        logger.info(
            "Epoch %s/%s loss=%.4f accuracy=%.4f auc=%.4f",
            epoch + 1,
            epochs,
            loss,
            metrics.get("accuracy", 0.0),
            metrics.get("auc", 0.0),
        )
        trainer.checkpoint(checkpoint_path, metric=metrics.get("auc", 0.0))

    return model
