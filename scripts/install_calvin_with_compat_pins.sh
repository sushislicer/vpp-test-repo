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

# Optional (recommended in CN environments):
#   export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#   export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

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

# Ensure build-time dependencies pulled via legacy `setup_requires` can be resolved.
# Some mirrors (or HTTP URLs) are treated as untrusted by pip and will be ignored,
# causing failures like: "No matching distribution found for pytest-runner".
if [[ -z "${PIP_INDEX_URL:-}" ]]; then
  export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
fi
if [[ -z "${PIP_TRUSTED_HOST:-}" ]]; then
  export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
fi

if [[ -n "${PIP_INDEX_URL:-}" ]]; then
  echo "[calvin-install] Using PIP_INDEX_URL=${PIP_INDEX_URL}"
fi
if [[ -n "${PIP_TRUSTED_HOST:-}" ]]; then
  echo "[calvin-install] Using PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}"
fi

# pyhash (a CALVIN dependency) sometimes declares build-time deps via setup_requires.
# If those build deps fail to resolve, installs can error out while generating metadata.
# Preinstall the common culprits explicitly.
python3 -m pip install -U pytest-runner pytest-benchmark >/dev/null || true

echo "[calvin-install] Installing CALVIN via install.sh..."
cd "${CALVIN_ROOT}"
sh install.sh

echo "[calvin-install] Done."
