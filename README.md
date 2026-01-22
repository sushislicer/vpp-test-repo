# VPP (Video Prediction Policy) — Repo Overview

This repo contains:

* Upstream VPP implementation in [`video-prediction-policy/`](video-prediction-policy/README.md)
* Convenience wrappers + launchers in [`scripts/`](scripts/README.md)

## What to read

* End-to-end setup + training: [`scripts/README.md`](scripts/README.md)
* (Optional) interactive setup for Calvin D→D benchmark: [`scripts/setup_vpp_d2d.sh`](scripts/setup_vpp_d2d.sh)

## Submodule: `video-prediction-policy/`

The upstream VPP code is tracked as a git submodule at [`video-prediction-policy/`](video-prediction-policy/README.md).

Clone with submodules:

```bash
git clone --recurse-submodules <YOUR_REPO_URL>
```

If you already cloned:

```bash
git submodule update --init video-prediction-policy
```

Note: this repo intentionally does **not** recurse into any *nested* submodules under
`video-prediction-policy/`.

## Smoke tests (no datasets/models required)

```bash
python3 -m compileall -q scripts video-prediction-policy
python3 scripts/test_d2d_setup.py
```
