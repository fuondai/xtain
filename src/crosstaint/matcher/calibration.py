"""Platt calibration for bridge event matcher scores."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


logger = logging.getLogger(__name__)


class PlattCalibrator:
    """Platt scaling: fit logistic regression on calibrated outputs to produce well-calibrated probabilities."""

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
        self._fitted = False

    def fit(self, raw_scores: list[float], true_labels: list[int]) -> None:
        if len(raw_scores) < 2 or len(set(true_labels)) < 2:
            logger.warning("Insufficient data for Platt calibration (n=%s unique_labels=%s)", len(raw_scores), len(set(true_labels)))
            self._fitted = False
            return
        X = np.array(raw_scores, dtype=np.float64).reshape(-1, 1)
        y = np.array(true_labels, dtype=np.int32)
        try:
            self._model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            self._model.fit(X, y)
            self._fitted = True
        except Exception as e:
            logger.warning("Platt calibration failed: %s", e)
            self._fitted = False

    def calibrate(self, raw_scores: list[float]) -> list[float]:
        if not self._fitted or self._model is None:
            return raw_scores
        X = np.array(raw_scores, dtype=np.float64).reshape(-1, 1)
        try:
            calibrated = self._model.predict_proba(X)[:, 1].tolist()
            return calibrated
        except Exception:
            return raw_scores

    def is_fitted(self) -> bool:
        return self._fitted


class TemperatureScaling:
    """Temperature scaling: single-parameter calibration by dividing logits by a learned temperature."""

    def __init__(self) -> None:
        self._temperature = 1.0
        self._fitted = False

    def fit(self, logits: list[float], true_labels: list[int]) -> None:
        if len(logits) < 2:
            self._fitted = False
            return
        best_temp = 1.0
        best_nll = float("inf")
        for temp in np.linspace(0.1, 5.0, 100):
            scaled = [l / temp for l in logits]
            nll = _binary_cross_entropy(scaled, true_labels)
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        self._temperature = best_temp
        self._fitted = True

    def calibrate(self, logits: list[float]) -> list[float]:
        if not self._fitted:
            return logits
        return [1.0 / (1.0 + np.exp(-l / self._temperature)) for l in logits]


def _binary_cross_entropy(logits: list[float], labels: list[int]) -> float:
    eps = 1e-7
    total = 0.0
    for logit, label in zip(logits, labels):
        p = 1.0 / (1.0 + np.exp(-logit))
        p = min(max(p, eps), 1.0 - eps)
        total -= label * np.log(p) + (1 - label) * np.log(1 - p)
    return total / len(logits)
