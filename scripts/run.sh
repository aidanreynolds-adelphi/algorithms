#!/bin/bash

# Run all three models (logreg, nn, xgboost) using the configured test_size.
# Results are written to report/. Must be run from repository root.

set -euo pipefail

source "$(dirname "$0")/common.sh"

echo "=== Running logreg, nn, xgboost ==="
"$(dirname "$0")/run_logreg.sh"
"$(dirname "$0")/run_nn.sh"
"$(dirname "$0")/run_xgboost.sh"

echo "=== Comparing results and writing model comparison report ==="
poetry run python -m algorithms.compare_results

echo "Done. Reports are in report/."
echo "Model comparison: report/model_comparison.txt"
