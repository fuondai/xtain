"""Benchmark corpus loading and CrossTaint experiment execution."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .corpus import BenchmarkCase, BenchmarkDataset, load_benchmark_dataset

_RUN_EXPORTS = {
    "ExperimentConfig",
    "ExploitCase",
    "MethodResult",
    "run_experiment",
    "run_local_benchmark",
    "run_smoke_benchmark",
    "run_corpus_benchmark",
    "write_results",
    "write_per_case_csv",
    "print_results",
}

__all__ = [
    "BenchmarkCase",
    "BenchmarkDataset",
    "load_benchmark_dataset",
    "ExperimentConfig",
    "ExploitCase",
    "MethodResult",
    "run_experiment",
    "run_local_benchmark",
    "run_smoke_benchmark",
    "run_corpus_benchmark",
    "write_results",
    "write_per_case_csv",
    "print_results",
]


def __getattr__(name: str) -> Any:
    if name in _RUN_EXPORTS:
        module = import_module(".run", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
