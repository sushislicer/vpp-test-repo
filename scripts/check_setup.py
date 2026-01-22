#!/usr/bin/env python3
"""Preflight checks for VPP scripts.

This is a *lightweight* checker intended to catch common setup issues early
(missing Python packages, missing env vars, nonexistent paths).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ok(msg: str) -> None:
    print(f"✓ {msg}")


def _warn(msg: str) -> None:
    print(f"⚠ {msg}")


def _fail(msg: str) -> None:
    print(f"✗ {msg}")


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception as e:
        _fail(f"import {name}: {type(e).__name__}: {e}")
        return False


def _check_path(var: str) -> bool:
    val = os.environ.get(var)
    if not val:
        _warn(f"{var} not set")
        return False
    p = Path(val).expanduser()
    if not p.exists():
        _fail(f"{var} points to missing path: {p}")
        return False
    _ok(f"{var} exists: {p}")
    return True


def main() -> int:
    print("VPP preflight checks")
    print("=" * 80)

    # Python deps (most common failure is diffusers)
    deps = [
        "torch",
        "accelerate",
        "hydra",
        "omegaconf",
        "transformers",
        "huggingface_hub",
        "diffusers",
    ]

    ok = True
    for d in deps:
        ok = _try_import(d) and ok

    if ok:
        _ok("core Python imports look good")

    # Optional: print CUDA visibility
    try:
        import torch

        _ok(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} n_gpus={torch.cuda.device_count()}")
    except Exception:
        pass

    print("\nEnvironment variables (Task 1/2)")
    print("-" * 80)
    _check_path("VIDEO_DATASET_DIR")
    _check_path("CALVIN_ROOT_DATA_DIR")
    _check_path("SVD_BASE_MODEL")
    _check_path("CLIP_MODEL")

    print("\nDone.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
