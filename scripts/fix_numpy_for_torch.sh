#!/usr/bin/env bash
set -euo pipefail

# Fix common NumPy 2.x incompatibilities for some PyTorch/Transformers installs.
#
# Symptom:
#   "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"
#
# Usage (after `conda activate vpp`):
#   bash scripts/fix_numpy_for_torch.sh

python3 - <<'PY'
import sys
print('python', sys.version)
try:
    import numpy as np
    print('numpy before', np.__version__)
except Exception as e:
    print('numpy import error (before):', e)
PY

python3 -m pip install --upgrade --force-reinstall "numpy<2" 

python3 - <<'PY'
import numpy as np
print('numpy after', np.__version__)
PY

echo "Done. If you still see NumPy ABI warnings, reinstall compiled deps in this env."

