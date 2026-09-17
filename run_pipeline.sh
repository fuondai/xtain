#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo "Step 1: Running smoke tests..."
PYTHONPATH=src "$PYTHON_EXEC" -m crosstaint.smoke_test

echo "Step 2: Rebuilding 55-incident manifest..."
"$PYTHON_EXEC" scripts/build_55_manifest.py

echo "Step 3: Running comprehensive 10-run benchmark on 55 incidents..."
"$PYTHON_EXEC" scripts/run_paper_benchmarks.py

echo "Step 4: Generating analytical vector figures..."
"$PYTHON_EXEC" generate_analytic_figures.py

echo "CrossTaint evaluation pipeline completed successfully."
