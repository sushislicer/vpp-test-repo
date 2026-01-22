#!/usr/bin/env bash
set -euo pipefail

# Configure pip to use an HTTPS mirror (recommended behind GFW).
#
# Usage:
#   bash scripts/configure_pip_mirror_cn.sh
#
# Optional:
#   PIP_MIRROR=tuna|aliyun|pypi bash scripts/configure_pip_mirror_cn.sh

PIP_MIRROR="${PIP_MIRROR:-tuna}"

case "${PIP_MIRROR}" in
  tuna)
    INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
    TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
    ;;
  aliyun)
    INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
    TRUSTED_HOST="mirrors.aliyun.com"
    ;;
  pypi)
    INDEX_URL="https://pypi.org/simple"
    TRUSTED_HOST="pypi.org"
    ;;
  *)
    echo "Unknown PIP_MIRROR='${PIP_MIRROR}'. Use tuna|aliyun|pypi" >&2
    exit 2
    ;;
esac

echo "Setting pip index-url to: ${INDEX_URL}" >&2
python3 -m pip config set global.index-url "${INDEX_URL}" >/dev/null
python3 -m pip config set global.trusted-host "${TRUSTED_HOST}" >/dev/null

echo "Done. Current pip config (relevant):" >&2
python3 -m pip config list | grep -E 'global\.(index-url|trusted-host)' || true

