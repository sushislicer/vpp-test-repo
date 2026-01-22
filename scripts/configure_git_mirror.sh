#!/usr/bin/env bash
set -euo pipefail

# Configure git to rewrite GitHub URLs to a mirror.
#
# Usage:
#   MIRROR_PREFIX='https://ghfast.top/https://github.com/' bash scripts/configure_git_mirror.sh
#
# Notes:
# - This is the safest way to speed up submodule clones without hard-coding mirror
#   URLs into `.gitmodules`.

MIRROR_PREFIX="${MIRROR_PREFIX:-}"
if [[ -z "${MIRROR_PREFIX}" ]]; then
  echo "Set MIRROR_PREFIX, e.g.:" >&2
  echo "  MIRROR_PREFIX='https://ghfast.top/https://github.com/' bash scripts/configure_git_mirror.sh" >&2
  exit 2
fi

echo "Configuring git URL rewrite:" >&2
echo "  https://github.com/  ->  ${MIRROR_PREFIX}" >&2

git config --global "url.${MIRROR_PREFIX}.insteadOf" "https://github.com/"

echo "Done. Test with:" >&2
echo "  git ls-remote https://github.com/roboterax/video-prediction-policy.git" >&2
