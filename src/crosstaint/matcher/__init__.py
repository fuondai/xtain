"""Bridge event matcher: featurization, transformer model, training, and calibration."""

from __future__ import annotations

from .featurizer import EventFeaturizer
from .model import BridgeEventMatcher, BilinearScoringMatcher, SiameseMLPMatcher
from .trainer import MatcherTrainer
from .calibration import PlattCalibrator

__all__ = [
    "EventFeaturizer",
    "BridgeEventMatcher",
    "BilinearScoringMatcher",
    "SiameseMLPMatcher",
    "MatcherTrainer",
    "PlattCalibrator",
]
