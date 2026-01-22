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

# Large wheels (notably torch) can trigger transient network read timeouts in some
# container/VM networks. Make pip more tolerant by default.
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-600}"
export PIP_RETRIES="${PIP_RETRIES:-20}"

pip_install() {
  # Usage: pip_install <args...>
  python3 -m pip install \
    --index-url "${PIP_INDEX_URL}" \
    --trusted-host "${PIP_TRUSTED_HOST}" \
    --retries "${PIP_RETRIES}" \
    --timeout "${PIP_DEFAULT_TIMEOUT}" \
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
  TORCH_VER=""
  TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || true)
  echo "[calvin-install] torch (before): ${TORCH_VER:-<not installed>}"

  # If torch is missing OR not the exact pinned version CALVIN requests, pip will
  # attempt to download torch==1.13.1 during `install.sh`. Prefer fixing torch here.
  if [[ "${TORCH_VER}" != "1.13.1" ]]; then
    if command -v conda >/dev/null 2>&1; then
      echo "[calvin-install] Installing/downgrading torch to 1.13.1 via conda (avoids pip wheel download)..."
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

  TORCH_VER_AFTER=""
  TORCH_VER_AFTER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || true)
  echo "[calvin-install] torch (after): ${TORCH_VER_AFTER:-<not installed>}"
  if [[ "${TORCH_VER_AFTER}" != "1.13.1" ]]; then
    echo "[calvin-install] WARNING: torch is not exactly 1.13.1; CALVIN install may still trigger a pip download of torch==1.13.1" >&2
    echo "[calvin-install] If you are okay with a slower install, you can ignore this warning and let pip download it." >&2
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

# Some environments (notably ML containers that already ship torch) may not want to
# run CALVIN's upstream `install.sh`, because it can force-download pinned torch
# wheels and other heavy deps.
#
# Modes:
#   - install_sh (default): run upstream `install.sh` as-is.
#   - manual: mimic the common install.sh flow by installing subpackages editable.
#   - no_deps: like manual, but installs `calvin_models` with `--no-deps` (avoids
#              forcing torch==1.13.1 and problematic deps like pyhash).
CALVIN_INSTALL_MODE="${CALVIN_INSTALL_MODE:-install_sh}"

cd "${CALVIN_ROOT}"
case "${CALVIN_INSTALL_MODE}" in
  install_sh)
    sh install.sh
    ;;
  manual)
    echo "[calvin-install] CALVIN_INSTALL_MODE=manual (editable installs)"
    pip_install -e calvin_env/tacto
    pip_install -e calvin_env
    pip_install -e calvin_models
    ;;
  no_deps)
    echo "[calvin-install] CALVIN_INSTALL_MODE=no_deps (skip deps for calvin_models)"
    echo "[calvin-install] NOTE: this avoids forcing torch==1.13.1; if you later hit runtime import errors, switch to install_sh or a dedicated conda env." >&2
    pip_install -e calvin_env/tacto
    pip_install -e calvin_env
    pip_install -e calvin_models --no-deps
    ;;
  *)
    echo "Unknown CALVIN_INSTALL_MODE='${CALVIN_INSTALL_MODE}'. Use install_sh|manual|no_deps" >&2
    exit 2
    ;;
esac

echo "[calvin-install] Done."
