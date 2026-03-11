#!/usr/bin/env bash
# Generate obesity MLP architecture diagram with PlotNeuralNet (clones repo if needed).
# Can be run from anywhere; switches to repo root before running.

set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "$0")" && pwd -P)"
REPO_ROOT="$(cd -P "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"

poetry run python -m algorithms.plot_nn_diagram
