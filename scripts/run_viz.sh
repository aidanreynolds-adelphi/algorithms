#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

FIGURES_DIR="${REPO_ROOT}/figures"
mkdir -p "${FIGURES_DIR}"

poetry run python -m algorithms.obesity_viz

