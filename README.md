# CrossTaint

CrossTaint is an open-source framework for probabilistically bounded multi-hop cross-chain taint tracking in bridge-exploit forensics. The implementation contains the bridge intermediate representation, value-conserving propagation engine, bridge-event matcher, heterogeneous pseudonym resolver, cold-start fallback, synthetic trajectory generator, and evaluation utilities described in the accompanying anonymous artifact.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Runtime configuration is stored in `config/`. RPC endpoints and API keys are read from environment variables referenced by the YAML files. Set `CROSSTAINT_CONFIG_DIR` to use an external configuration directory.

## Minimal Use

```python
from crosstaint.pipeline import InferenceStack

stack = InferenceStack.from_config(
    matcher_checkpoint=matcher_checkpoint,
    resolver_checkpoint=resolver_checkpoint,
    event_index=event_index,
    nodes=nodes,
    edges=edges,
)

result = stack.propagate(
    origin=origin_node_id,
    origin_chain=origin_chain,
    origin_value=origin_value,
    graph=nodes,
    edges=edges,
    case_id=case_id,
)
```

`result.suspect_set` contains ranked suspect addresses and `result.soundness_certificate` records the per-case bounded-mismatch certificate metadata.

## Benchmark Dataset & Reproduction

The repository includes the verified corpus of 55 real-world cross-chain bridge exploits spanning 2021-2026:
- Raw verified exploit incidents: `data/crosschain_bridge_exploits_55.json`
- Ground-truth evaluation manifest: `data/manifest_55_corpus.json`

To run the complete evaluation pipeline (smoke tests, manifest verification, 10-run comparative benchmarks, and vector figure generation):

```bash
bash run_pipeline.sh
```

Or run individual reproduction scripts:

```bash
python -m crosstaint.smoke_test
python scripts/run_paper_benchmarks.py
python scripts/reproduce_paper_full.py
python generate_analytic_figures.py
```

## License

Apache License 2.0. See `LICENSE`.
