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

# Ensure build-time dependencies pulled via legacy `setup_requires` can be resolved.
# Some mirrors (or HTTP URLs) are treated as untrusted by pip and will be ignored,
# causing failures like: "No matching distribution found for pytest-runner".
if [[ -z "${PIP_INDEX_URL:-}" ]]; then
  export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
fi
if [[ -z "${PIP_TRUSTED_HOST:-}" ]]; then
  export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
fi

# In some containers/environments, a global pip config may force an HTTP mirror
# (e.g. mirrors.baidubce.com). That can be treated as untrusted and break builds.
# Force pip to ignore user/system config during this install flow.
export PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-/dev/null}"

pip_install() {
  # Usage: pip_install <args...>
  python3 -m pip install \
    --index-url "${PIP_INDEX_URL}" \
    --trusted-host "${PIP_TRUSTED_HOST}" \
    "$@"
}

pip_install --upgrade "pip<25" wheel "setuptools<58" >/dev/null

# PyTorch is a very large wheel on PyPI and can be painfully slow to download in some
# networks/containers. If you have conda available, preinstalling torch via conda is
# often faster and avoids pip trying to fetch ~GB wheels.
#
# Opt-in via:
#   export CALVIN_PREINSTALL_TORCH=1
#
# You can also choose a CPU-only install (smaller) via:
#   export CALVIN_TORCH_VARIANT=cpu
CALVIN_PREINSTALL_TORCH="${CALVIN_PREINSTALL_TORCH:-0}"
CALVIN_TORCH_VARIANT="${CALVIN_TORCH_VARIANT:-cuda}"
if [[ "${CALVIN_PREINSTALL_TORCH}" == "1" ]]; then
  if python3 -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
    echo "[calvin-install] torch already installed; skipping torch preinstall"
  else
    if command -v conda >/dev/null 2>&1; then
      echo "[calvin-install] Preinstalling torch via conda to avoid slow pip wheel downloads..."
      if [[ "${CALVIN_TORCH_VARIANT}" == "cpu" ]]; then
        # CPU-only (smaller), good enough to get CALVIN installed.
        conda install -y pytorch==1.13.1 torchvision torchaudio cpuonly -c pytorch || true
      else
        # CUDA variant (requires working NVIDIA stack inside the container).
        conda install -y pytorch==1.13.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia || true
      fi
    else
      echo "[calvin-install] conda not found; cannot preinstall torch via conda" >&2
    fi
  fi
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
pip_install -U pytest-runner pytest-benchmark

echo "[calvin-install] Installing CALVIN via install.sh..."
cd "${CALVIN_ROOT}"
sh install.sh

echo "[calvin-install] Done."
