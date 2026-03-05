#!/usr/bin/env bash
#
# setup_mac.sh - Install system-level dependencies on macOS for this project.
#
# Installs only tools that cannot be managed by asdf or Poetry (e.g. Homebrew, Git).
# Python, Poetry, and project Python deps are expected via asdf + Poetry; see
# .tool-versions and pyproject.toml.
#
# Usage: ./scripts/setup_mac.sh
# Can be run from anywhere; does not require repository root.

set -euo pipefail

OS="$(uname -s)"
if [[ "$OS" != "Darwin" ]]; then
  echo "This script is for macOS only (detected: $OS)." 1>&2
  exit 1
fi

# --- Homebrew ---
if ! command -v brew &>/dev/null; then
  echo "Error: Homebrew is not installed. Please install it from https://brew.sh and re-run this script." 1>&2
  exit 1
fi

# --- Git (required for pre-commit and version control) ---
if ! command -v git &>/dev/null; then
  echo "Error: Git is not installed. Please install it with Homebrew (e.g., 'brew install git') and re-run this script." 1>&2
  exit 1
fi

# --- libomp (required by some numeric libraries, e.g. XGBoost on macOS) ---
if ! brew list --versions libomp &>/dev/null; then
  echo "Installing libomp via Homebrew..."
  brew install libomp
else
  echo "libomp already installed."
fi

# --- asdf (recommended for managing Python and Poetry versions) ---
if command -v asdf &>/dev/null; then
  echo "asdf is installed ($(asdf version))."
else
  echo "asdf is not installed. We recommend installing asdf to manage Python and Poetry versions. See: https://asdf-vm.com" 1>&2
fi

# --- Poetry (required for dependency management) ---
if command -v poetry &>/dev/null; then
  echo "Poetry is installed ($(poetry --version))."
else
  echo "Error: Poetry is not installed but is required for this project. Install it (e.g., via asdf) and re-run this script." 1>&2
  exit 1
fi

echo ""
echo "System dependencies are present."
