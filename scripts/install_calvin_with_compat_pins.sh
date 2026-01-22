#!/usr/bin/env bash
set -euo pipefail

# Install CALVIN with compatibility pins for common modern-Python packaging breakages.
#
# Motivation:
# - CALVIN may depend on `pyhash==0.9.3`, whose setup uses `use_2to3`.
# - `use_2to3` was removed in newer `setuptools`, causing:
#     "error in pyhash setup command: use_2to3 is invalid."
#
# This script pins setuptools to a version that still supports `use_2to3`,
# then runs CALVIN's installer.
#
# Usage (on the SSH machine):
#   conda activate vpp
#   export CALVIN_ROOT=~/calvin
#   bash scripts/install_calvin_with_compat_pins.sh

CALVIN_ROOT="${CALVIN_ROOT:-}"
if [[ -z "${CALVIN_ROOT}" ]]; then
  echo "Set CALVIN_ROOT first, e.g.: export CALVIN_ROOT=~/calvin" >&2
  exit 2
fi

if [[ ! -d "${CALVIN_ROOT}" ]]; then
  echo "CALVIN_ROOT does not exist: ${CALVIN_ROOT}" >&2
  echo "Clone calvin first (recommended):" >&2
  echo "  git clone --recurse-submodules https://github.com/mees/calvin.git \"${CALVIN_ROOT}\"" >&2
  exit 2
fi

echo "[calvin-install] python: $(python3 -V)"

echo "[calvin-install] Applying packaging compatibility pins..."
python3 -m pip install --upgrade "pip<25" "wheel" "setuptools<58" >/dev/null

echo "[calvin-install] Installing CALVIN via install.sh..."
cd "${CALVIN_ROOT}"
sh install.sh

echo "[calvin-install] Done."

