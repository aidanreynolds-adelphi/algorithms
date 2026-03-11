#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

REPORT_DIR="${REPO_ROOT}/report"
mkdir -p "${REPORT_DIR}"

log_file="${REPORT_DIR}/xgboost_gridsearch_log.txt"
rm -f "${log_file}"

echo "Running XGBoost GridSearchCV (output also in ${log_file})..."
poetry run python -m algorithms.obesity_xgboost_gridsearch | tee "${log_file}"

echo "Grid search report: ${REPORT_DIR}/xgboost_gridsearch_report.txt"
echo "CV results CSV:      ${REPORT_DIR}/xgboost_gridsearch_cv_results.csv"

