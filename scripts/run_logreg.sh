#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

REPORT_DIR="${REPO_ROOT}/report"
mkdir -p "${REPORT_DIR}"

output_file="${REPORT_DIR}/logreg_report.txt"
rm -f "${output_file}"

poetry run python -m algorithms.obesity_logreg | tee "${output_file}"

