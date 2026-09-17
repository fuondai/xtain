"""CrossTaint experiment runner for synthetic smoke and manifest-backed runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from crosstaint.config import Config
from crosstaint.eval import (
    compute_f1,
    compute_false_positive_rate,
    compute_hop_recall,
    compute_precision,
)
from crosstaint.experiments.corpus import BenchmarkCase, load_benchmark_dataset
from crosstaint.indexer import SyntheticGraphBuilder
from crosstaint.propagation import PropagationEngine
from crosstaint.synth import BRIDGE_CHAINS, BRIDGES, BRIDGE_USAGE_WEIGHTS, SyntheticBenignGenerator
from crosstaint.types import IREdge, IRNode, PropagationResult, SimilarityOutput


logger = logging.getLogger(__name__)

SUPPORTED_VARIANTS = frozenset({
    "full",
    "coldstart-only",
    "no-coldstart",
    "no-valuecons",
    "no-decay",
    "relaxed-threshold",
    "strict-threshold",
})


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    num_benign: int = 100
    num_exploit: int = 10
    num_runs: int = 5
    seed: int = 42
    rho: float = 0.81
    threshold: float = 0.04
    output_dir: str = "./results"
    benchmark_manifest: str | None = None
    benign_addresses: str | None = None
    require_two_source_ground_truth: bool = True
    variant: str = "full"
    held_out_bridge: str | None = None
    neg_ratio: int = 0


@dataclass(slots=True)
class ExploitCase:
    case_id: str
    origin_address: str
    chain: str
    value: int
    hops: list[tuple[str, str, str]]
    ground_truth_addresses: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MethodResult:
    name: str
    hop_recalls: list[float] = field(default_factory=list)
    precisions: list[float] = field(default_factory=list)
    fprs: list[float] = field(default_factory=list)
    f1s: list[float] = field(default_factory=list)
    runtimes_ms: list[float] = field(default_factory=list)

    def add_run(self, recall: float, precision: float, fpr: float, runtime_ms: float) -> None:
        self.hop_recalls.append(recall)
        self.precisions.append(precision)
        self.fprs.append(fpr)
        self.f1s.append(compute_f1(recall, precision))
        self.runtimes_ms.append(runtime_ms)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "hop_recall": _stats(self.hop_recalls),
            "precision": _stats(self.precisions),
            "fpr": _stats(self.fprs),
            "f1": _stats(self.f1s),
            "runtime_ms": _stats(self.runtimes_ms),
        }


class _ZeroResolver:
    def resolve(self, source: str, target: str) -> SimilarityOutput:
        return SimilarityOutput(source, target, similarity=0.0, is_cold_start=False)


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(values),
    }


def _node_id(chain: str, address: str) -> str:
    return f"{chain}:{address.lower()}"


def _address_from_digest(seed: int, label: str) -> str:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).hexdigest()
    return f"0x{digest[:40]}"


def _sample_bridge_for_chain(
    rng: np.random.Generator,
    chain: str,
    allowed_bridges: list[str],
) -> str:
    candidates = [
        bridge for bridge in allowed_bridges
        if chain in BRIDGE_CHAINS.get(bridge, [])
    ]
    if not candidates:
        candidates = list(allowed_bridges)
    weights = np.array([BRIDGE_USAGE_WEIGHTS.get(b, 0.01) for b in candidates], dtype=np.float64)
    weights = weights / weights.sum()
    return str(rng.choice(candidates, p=weights))


def _sample_destination_chain(rng: np.random.Generator, bridge: str, source_chain: str) -> str:
    candidates = [
        chain for chain in BRIDGE_CHAINS.get(bridge, [])
        if chain != source_chain
    ]
    if not candidates:
        candidates = [chain for chain in ("ethereum", "bsc", "polygon", "arbitrum", "avalanche") if chain != source_chain]
    return str(rng.choice(candidates))


def _generate_exploit_cases(
    n_cases: int,
    seed: int,
    allowed_bridges: list[str] | None = None,
) -> list[ExploitCase]:
    rng = np.random.default_rng(seed)
    bridges = list(allowed_bridges or BRIDGES)
    if not bridges:
        raise ValueError("at least one bridge is required to generate exploit cases")
    base_chains = ["ethereum", "bsc", "polygon", "arbitrum", "avalanche"]
    cases: list[ExploitCase] = []

    for case_idx in range(n_cases):
        current_chain = str(rng.choice(base_chains))
        origin = _address_from_digest(seed, f"exploit:{case_idx}:origin")
        n_hops = int(rng.integers(2, 6))
        value = int(10 ** rng.uniform(17.0, 20.0))
        hops: list[tuple[str, str, str]] = []

        for _ in range(n_hops):
            bridge = _sample_bridge_for_chain(rng, current_chain, bridges)
            dest_chain = _sample_destination_chain(rng, bridge, current_chain)
            hops.append((current_chain, dest_chain, bridge))
            current_chain = dest_chain

        case_id = str(uuid5(NAMESPACE_URL, f"crosstaint:{seed}:{case_idx}:{origin}"))
        cases.append(
            ExploitCase(
                case_id=case_id,
                origin_address=origin,
                chain=hops[0][0],
                value=value,
                hops=hops,
            )
        )

    return cases


def _case_ground_truth(nodes: list[IRNode]) -> list[str]:
    return [node.address for node in nodes[1:]]


def _graph_repr(nodes: dict[str, IRNode], edges: list[IREdge]) -> dict[str, object]:
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "address": node.address,
                "chain": node.chain,
                "node_type": node.node_type,
                "degree": node.bridge_count + node.dex_swap_count,
                "bridge_count": node.bridge_count,
                "dex_swap_count": node.dex_swap_count,
            }
            for node in nodes.values()
        ],
        "edges": [
            {
                "edge_id": edge.edge_id,
                "source": edge.source_node,
                "target": edge.target_node,
                "edge_type": edge.edge_type,
                "bridge": edge.bridge,
                "value": edge.value,
                "asset": edge.asset,
            }
            for edge in edges
        ],
    }


def _predicted_addresses(result: PropagationResult) -> list[str]:
    return [entry.address for entry in result.suspect_set]


def _validate_variant(variant: str) -> None:
    if variant not in SUPPORTED_VARIANTS:
        supported = ", ".join(sorted(SUPPORTED_VARIANTS))
        raise ValueError(f"unsupported variant {variant!r}; supported variants: {supported}")


def _engine_for_config(cfg: ExperimentConfig, seed: int) -> PropagationEngine:
    _validate_variant(cfg.variant)
    rho = 1.0 if cfg.variant == "no-decay" else cfg.rho
    threshold = cfg.threshold
    if cfg.variant == "relaxed-threshold":
        threshold = max(0.0, cfg.threshold * 0.5)
    elif cfg.variant == "strict-threshold":
        threshold = min(1.0, cfg.threshold * 2.0)
    return PropagationEngine.from_config(
        rho=rho,
        threshold=threshold,
        seed=seed,
        value_conservation_enabled=(cfg.variant != "no-valuecons"),
    )


def _resolver_for_variant(cfg: ExperimentConfig) -> object | None:
    if cfg.variant == "no-coldstart":
        return _ZeroResolver()
    return None


def _prepare_graph_for_variant(cfg: ExperimentConfig, nodes: dict[str, IRNode]) -> dict[str, IRNode]:
    if cfg.variant != "no-coldstart":
        return nodes
    updated: dict[str, IRNode] = {}
    for key, node in nodes.items():
        updated[key] = IRNode(
            node_id=node.node_id,
            address=node.address,
            chain=node.chain,
            node_type=node.node_type,
            tags=node.tags,
            first_seen=node.first_seen,
            degree_at_first_seen=max(3, node.degree_at_first_seen),
            in_value_total=node.in_value_total,
            out_value_total=node.out_value_total,
            bridge_count=node.bridge_count,
            dex_swap_count=node.dex_swap_count,
        )
    return updated


def run_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    Config.reset()
    Config.load()
    _validate_variant(cfg.variant)

    start_time = time.monotonic()
    result_acc = MethodResult("CrossTaint")
    per_case: list[dict[str, object]] = []
    allowed_exploit_bridges = [cfg.held_out_bridge] if cfg.held_out_bridge else None

    for run_idx in range(cfg.num_runs):
        seed = cfg.seed + run_idx
        logger.info("Synthetic run %s/%s with seed %s", run_idx + 1, cfg.num_runs, seed)

        generator = SyntheticBenignGenerator(seed=seed)
        benign_bridges = [b for b in BRIDGES if b != cfg.held_out_bridge] if cfg.held_out_bridge else None
        benign_trajs = generator.generate(num_trajectories=cfg.num_benign, bridges=benign_bridges)
        exploit_cases = _generate_exploit_cases(cfg.num_exploit, seed, allowed_exploit_bridges)

        builder = SyntheticGraphBuilder(
            benign_trajs,
            exploit_addresses=[case.origin_address for case in exploit_cases],
        )
        nodes, _ = builder.build()

        for case in exploit_cases:
            case_nodes, _case_edges = builder.create_exploit_case(
                origin=case.origin_address,
                chain=case.chain,
                value=case.value,
                hops=case.hops,
            )
            case.ground_truth_addresses = _case_ground_truth(case_nodes)

        nodes, edge_map = builder.snapshot()
        graph_nodes = _prepare_graph_for_variant(cfg, nodes)
        edges = list(edge_map.values())
        benign_addresses = [
            address for trajectory in benign_trajs for address in trajectory.addresses
        ]

        for case in exploit_cases:
            origin_node_id = _node_id(case.chain, case.origin_address)
            engine = _engine_for_config(cfg, seed)
            result = engine.propagate(
                origin=origin_node_id,
                origin_chain=case.chain,
                origin_value=case.value,
                graph=graph_nodes,
                edges=edges,
                case_id=case.case_id,
                resolver=_resolver_for_variant(cfg),
            )
            _record_result(
                method_name="CrossTaint",
                result=result,
                ground_truth=case.ground_truth_addresses,
                benign_addresses=benign_addresses,
                run_idx=run_idx,
                case_id=case.case_id,
                ground_truth_hops=len(case.ground_truth_addresses),
                result_acc=result_acc,
                per_case=per_case,
                bridge_sequence=[hop[2] for hop in case.hops],
            )

    summary = {"CrossTaint": result_acc.summary()}
    summary["_metadata"] = _metadata(cfg, time.monotonic() - start_time, "synthetic")
    summary["_per_case"] = per_case
    return summary


def run_corpus_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    if not cfg.benchmark_manifest:
        raise ValueError("benchmark_manifest is required for corpus benchmark mode")

    Config.reset()
    Config.load()
    _validate_variant(cfg.variant)
    dataset = load_benchmark_dataset(
        manifest_path=Path(cfg.benchmark_manifest),
        benign_addresses_path=Path(cfg.benign_addresses) if cfg.benign_addresses else None,
        require_two_source_ground_truth=cfg.require_two_source_ground_truth,
    )

    start_time = time.monotonic()
    result_acc = MethodResult("CrossTaint")
    per_case: list[dict[str, object]] = []
    graph_nodes = _prepare_graph_for_variant(cfg, dataset.nodes)

    for run_idx in range(cfg.num_runs):
        seed = cfg.seed + run_idx
        logger.info("Corpus run %s/%s with seed %s", run_idx + 1, cfg.num_runs, seed)
        for case in dataset.cases:
            engine = _engine_for_config(cfg, seed)
            result = engine.propagate(
                origin=case.origin_node_id,
                origin_chain=case.origin_chain,
                origin_value=case.origin_value,
                graph=graph_nodes,
                edges=dataset.edges,
                case_id=case.case_id,
                resolver=_resolver_for_variant(cfg),
            )
            _record_result(
                method_name="CrossTaint",
                result=result,
                ground_truth=case.ground_truth_addresses,
                benign_addresses=dataset.benign_addresses,
                run_idx=run_idx,
                case_id=case.case_id,
                ground_truth_hops=len(case.ground_truth_addresses),
                result_acc=result_acc,
                per_case=per_case,
                bridge_sequence=[],
            )

    summary = {"CrossTaint": result_acc.summary()}
    summary["_metadata"] = _metadata(cfg, time.monotonic() - start_time, "corpus")
    summary["_dataset"] = {
        "num_cases": len(dataset.cases),
        "num_nodes": len(dataset.nodes),
        "num_edges": len(dataset.edges),
        "num_benign_addresses": len(dataset.benign_addresses),
        **dataset.metadata,
    }
    summary["_per_case"] = per_case
    return summary


def _record_result(
    method_name: str,
    result: PropagationResult,
    ground_truth: list[str],
    benign_addresses: list[str],
    run_idx: int,
    case_id: str,
    ground_truth_hops: int,
    result_acc: MethodResult,
    per_case: list[dict[str, object]],
    bridge_sequence: list[str],
) -> None:
    predicted = _predicted_addresses(result)
    recall = compute_hop_recall([predicted], [ground_truth])
    precision = compute_precision([predicted], [ground_truth])
    fpr = compute_false_positive_rate(predicted, benign_addresses)
    result_acc.add_run(recall, precision, fpr, float(result.runtime_ms))
    per_case.append(
        {
            "run": run_idx,
            "case_id": case_id,
            "method": method_name,
            "hop_recall": recall,
            "precision": precision,
            "fpr": fpr,
            "f1": compute_f1(recall, precision),
            "runtime_ms": result.runtime_ms,
            "ground_truth_hops": ground_truth_hops,
            "predicted_count": len(predicted),
            "certificate": result.certificate,
            "cert_g_hat": result.cert_g_hat,
            "cert_beta_hat": result.cert_beta_hat,
            "cert_epsilon": result.cert_epsilon,
            "cert_ood": result.cert_ood,
            "bridge_sequence": bridge_sequence,
        }
    )


def _metadata(
    cfg: ExperimentConfig,
    elapsed_seconds: float,
    scope: str,
) -> dict[str, object]:
    return {
        "config": _public_config(cfg),
        "elapsed_seconds": elapsed_seconds,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
        "config_hash": Config.config_hash(),
        "scope": scope,
        "run_mode": _run_mode(scope),
        "framework_only": True,
    }


def _public_config(cfg: ExperimentConfig) -> dict[str, object]:
    data = asdict(cfg)
    for key in ("output_dir", "benchmark_manifest", "benign_addresses"):
        value = data.get(key)
        if value:
            data[key] = Path(str(value)).name
    return data


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {"numpy": np.__version__}
    try:
        import pandas as pd
        versions["pandas"] = pd.__version__
    except Exception:
        pass
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import torch
        versions["torch"] = torch.__version__
    except Exception:
        pass
    return versions


def _run_mode(scope: str) -> str:
    if scope == "corpus":
        return "corpus"
    return "synthetic"


def write_results(results: dict[str, object], output_dir: str, filename: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / filename
    with open(results_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results_file


def write_per_case_csv(results: dict[str, object], output_path: str | Path) -> Path:
    rows = results.get("_per_case", [])
    if not isinstance(rows, list):
        raise ValueError("results do not contain per-case rows")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = sorted({key for row in rows if isinstance(row, dict) for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def print_results(results: dict[str, object]) -> None:
    logger.info("Experiment results summary")
    for name, data in results.items():
        if name.startswith("_") or not isinstance(data, dict):
            continue
        logger.info("%s", name)
        for metric in ("hop_recall", "precision", "fpr", "f1", "runtime_ms"):
            metric_data = data.get(metric)
            if not isinstance(metric_data, dict) or metric_data.get("n", 0) == 0:
                continue
            logger.info(
                "  %-12s mean=%.4f std=%.4f n=%s",
                metric,
                metric_data.get("mean", 0.0),
                metric_data.get("std", 0.0),
                metric_data.get("n", 0),
            )


def run_smoke_benchmark(output_dir: str = "./results") -> Path:
    cfg = ExperimentConfig(
        num_benign=50,
        num_exploit=5,
        num_runs=2,
        seed=42,
        output_dir=output_dir,
    )
    logger.info("Running smoke benchmark")
    results = run_experiment(cfg)
    results_file = write_results(results, cfg.output_dir, "smoke_results.json")
    logger.info("Results saved to %s", results_file)
    print_results(results)
    return results_file


def run_local_benchmark(cfg: ExperimentConfig) -> Path:
    logger.info(
        "Running local benchmark: benign=%s exploit=%s runs=%s variant=%s",
        cfg.num_benign,
        cfg.num_exploit,
        cfg.num_runs,
        cfg.variant,
    )
    results = run_experiment(cfg)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = write_results(results, cfg.output_dir, f"results_{timestamp}.json")
    logger.info("Results saved to %s", results_file)
    print_results(results)
    return results_file


def run_corpus_benchmark(cfg: ExperimentConfig) -> Path:
    logger.info("Running corpus benchmark from %s", cfg.benchmark_manifest)
    results = run_corpus_experiment(cfg)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = write_results(results, cfg.output_dir, f"corpus_results_{timestamp}.json")
    logger.info("Results saved to %s", results_file)
    print_results(results)
    return results_file


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="CrossTaint experiment runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke benchmark")
    parser.add_argument("--local-benchmark", action="store_true", help="Run local synthetic benchmark")
    parser.add_argument("--full-benchmark", action="store_true", help="Run corpus benchmark from a manifest")
    parser.add_argument("--num-benign", type=int, default=100)
    parser.add_argument("--num-exploit", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=0.81)
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--variant", type=str, default="full", choices=sorted(SUPPORTED_VARIANTS))
    parser.add_argument("--held-out-bridge", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--benchmark-manifest", type=str, default=None)
    parser.add_argument("--benign-addresses", type=str, default=None)
    parser.add_argument(
        "--allow-single-source-ground-truth",
        action="store_true",
        help="Disable two-source validation for local debugging only",
    )

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_benchmark(args.output_dir)
        return

    cfg = ExperimentConfig(
        num_benign=args.num_benign,
        num_exploit=args.num_exploit,
        num_runs=args.num_runs,
        seed=args.seed,
        rho=args.rho,
        threshold=args.threshold,
        output_dir=args.output_dir,
        benchmark_manifest=args.benchmark_manifest,
        benign_addresses=args.benign_addresses,
        require_two_source_ground_truth=not args.allow_single_source_ground_truth,
        variant=args.variant,
        held_out_bridge=args.held_out_bridge,
    )
    if args.full_benchmark or args.benchmark_manifest:
        run_corpus_benchmark(cfg)
    elif args.local_benchmark:
        run_local_benchmark(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
