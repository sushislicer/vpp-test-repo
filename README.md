# VPP (Video Prediction Policy) — Repo Overview

This repo contains:

* Upstream VPP implementation in [`video-prediction-policy/`](video-prediction-policy/README.md)
* Convenience wrappers + launchers in [`scripts/`](scripts/README.md)

## What to read

* End-to-end setup + training (including container instructions):
  [`scripts/README.md`](scripts/README.md)
* (Optional) interactive setup for Calvin D→D benchmark: [`scripts/setup_vpp_d2d.sh`](scripts/setup_vpp_d2d.sh)

## Running inside a PyTorch container (recommended on A800/A100/H100)

If you are using a container that already provides CUDA + PyTorch (e.g.
`registry.baidubce.com/inference/aibox-pytorch:v1.0-torch2.5.1-cu12.4`), follow
the **Option B** path in [`scripts/README.md`](scripts/README.md:118):

* Don’t create a new conda env.
* Install upstream VPP requirements **without** downgrading torch.
* Install CALVIN via the wrapper in `no_deps` mode to avoid `torch==1.13.1` wheel downloads.

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

### (China/GFW) Faster submodule download via a GitHub mirror

You can configure git to rewrite GitHub URLs to a mirror (recommended, because it
also applies to submodules):

```bash
# Option A: mirror expects URLs like: https://ghfast.top/https://github.com/<org>/<repo>.git
git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"

# Option B: mirror expects URLs like: https://ghfast.top/github.com/<org>/<repo>.git
# git config --global url."https://ghfast.top/github.com/".insteadOf "https://github.com/"
```

Then run submodule init normally:

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
