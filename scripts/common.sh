#!/bin/bash

# `common.sh` - shared bash helpers for scripts.
#
# Contract:
# - Any script that sources this file MUST be run from the repository root.
# - This file has no side-effects like `cd`; it only validates and exports `REPO_ROOT`.

_common_sh_fail() {
  echo "Error: $*" 1>&2
  # If sourced, return so the caller can handle it. If executed, exit.
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 1
  fi
  exit 1
}

# Resolve the absolute path to the directory containing this file.
COMMON_SH_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || _common_sh_fail "Failed to resolve common.sh directory"

if [[ "$(basename "$COMMON_SH_DIR")" != "scripts" ]]; then
  _common_sh_fail "common.sh must be located in the scripts/ directory (found: $COMMON_SH_DIR)"
fi

# Repo root is the parent of `scripts/`.
REPO_ROOT="$(cd -P "$COMMON_SH_DIR/.." && pwd -P)" || _common_sh_fail "Failed to resolve repository root"

# Enforce: the sourcing script must be run from repo root.
PWD_ABS="$(pwd -P)" || _common_sh_fail "Failed to resolve current working directory"
if [[ "$PWD_ABS" != "$REPO_ROOT" ]]; then
  _common_sh_fail "Script must be run from repository root.
  Expected: $REPO_ROOT
  Actual:   $PWD_ABS"
fi

export REPO_ROOT
