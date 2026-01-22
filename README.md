# VPP (Video Prediction Policy) — Repo Overview

This repo contains:

* Upstream VPP implementation in [`video-prediction-policy/`](video-prediction-policy/README.md)
* Convenience wrappers + launchers in [`scripts/`](scripts/README.md)

## What to read

* End-to-end setup + training: [`scripts/README.md`](scripts/README.md)
* (Optional) interactive setup for Calvin D→D benchmark: [`scripts/setup_vpp_d2d.sh`](scripts/setup_vpp_d2d.sh)

## Smoke tests (no datasets/models required)

```bash
python3 -m compileall -q scripts video-prediction-policy
python3 scripts/test_d2d_setup.py
```

