# VPP Calvin (smaller D/D split) — Experiment Runner

This folder is the **only** place you should need to touch to run experiments.

* Upstream VPP code lives in [`../video-prediction-policy/`](../video-prediction-policy/README.md) and is treated as read-only.
* This folder provides wrappers + launch scripts to run:
  1. **Task 1**: reproduce VPP training on a **smaller Calvin D/D dataset (~166GB)** (vs full Calvin ABCD→D ~500GB).
  2. **Task 2**: seed sensitivity test (Video A/B seeds, fixed Actor A).

## Repo layout

* Launchers (bash):
  * Task 1: [`run_task1_train_vpp_calvin_d2d.sh`](run_task1_train_vpp_calvin_d2d.sh)
  * Task 2: [`run_task2_video_seed_fixed_actor.sh`](run_task2_video_seed_fixed_actor.sh)
* Wrappers (python):
  * Actor training wrapper (avoids upstream `__main__` side-effects):
    [`train_actor_calvin.py`](train_actor_calvin.py)
* Presets:
  * [`config/cluster_presets.yaml`](config/cluster_presets.yaml)

## QUICK START (remote SSH, end-to-end)

This section is **fully command-driven** (copy/paste).

This README assumes your repo is cloned to:

* `~/workspace/vpp-test-repo`

Set:

```bash
export REPO_ROOT=~/workspace/vpp-test-repo
```

### 0) System prerequisites

You need:

* NVIDIA driver + CUDA runtime working (`nvidia-smi`)
* A working Python environment
  * Option A: a conda installation (Miniconda/Mambaforge)
  * Option B: an ML container that already includes Python + CUDA + PyTorch (e.g. aibox-pytorch)

### 1) Clone repo (with submodule)

```bash
mkdir -p ~/workspace
git clone --recurse-submodules <YOUR_REPO_URL> ~/workspace/vpp-test-repo
```

#### (China/GFW) Faster submodule download via a GitHub mirror

If `github.com` is slow/unreachable, configure git to rewrite GitHub URLs to your mirror.
This works for submodules too.

Example using `ghfast.top` (pick the prefix that matches your mirror):

```bash
# Option A: mirror expects URLs like: https://ghfast.top/https://github.com/<org>/<repo>.git
git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"

# Option B: mirror expects URLs like: https://ghfast.top/github.com/<org>/<repo>.git
# git config --global url."https://ghfast.top/github.com/".insteadOf "https://github.com/"
```

Then clone / update submodules as usual:

```bash
git clone --recurse-submodules <YOUR_REPO_URL> ~/workspace/vpp-test-repo
# or (if already cloned)
cd ~/workspace/vpp-test-repo && git submodule update --init video-prediction-policy
```

If you already cloned without submodules:

```bash
cd ~/workspace/vpp-test-repo
git submodule update --init --recursive
```

If your environment needs a mirror, run the git `insteadOf` config above first.

Verify:

```bash
ls -la ~/workspace/vpp-test-repo
ls -la ~/workspace/vpp-test-repo/video-prediction-policy
```

Note: upstream VPP is tracked as a git submodule at `video-prediction-policy/`.

### 2) Python environment + install Python deps

Two supported paths:

#### Option A (recommended for generic machines): conda env `vpp`

```bash
cd "${REPO_ROOT:-~/workspace/vpp-test-repo}"
conda create -n vpp python=3.10 -y
conda activate vpp

# (China/GFW) use an HTTPS pip mirror to avoid HTTP/untrusted-host issues
bash scripts/configure_pip_mirror_cn.sh

# Core deps for upstream VPP
pip install -r video-prediction-policy/requirements.txt
pip install accelerate "huggingface_hub[cli]"

# Avoid common NumPy 2.x ABI issues in some environments
pip install "numpy<2"

# (Optional but common)
pip install wandb
```

#### Option B (your setup): aibox-pytorch container (Torch 2.5.1 + CUDA 12.4)

If you are running inside:

* `registry.baidubce.com/inference/aibox-pytorch:v1.0-torch2.5.1-cu12.4`
* 4× A800 80GB GPUs

then **do not create a new conda env** unless you want to.
Use the container’s existing Python/torch and only install missing deps:

```bash
cd "${REPO_ROOT:-/root/workspace/vpp-test-repo}"

python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'n_gpus', torch.cuda.device_count())"

# (China/GFW) use an HTTPS pip mirror
bash scripts/configure_pip_mirror_cn.sh

# Container images sometimes ship a global pip config that enables strict settings
# (e.g. hash-checking) or uses an HTTP mirror. For a clean, reproducible install,
# ignore system/user pip config for this repo session.
export PIP_CONFIG_FILE=/dev/null

# Make pip more tolerant of slow/unstable networks for large wheels.
export PIP_DEFAULT_TIMEOUT=600
export PIP_RETRIES=20

# If you see "DO NOT MATCH THE HASHES" errors, it is usually a corrupted/partial
# download or a strict pip hash policy. Purge cache and retry with no cache.
python3 -m pip cache purge || true

# Install upstream VPP deps BUT DO NOT install/override torch.
# The upstream requirements pin `torch==2.0.1`, which would downgrade your container torch.
grep -v -E '^torch==' video-prediction-policy/requirements.txt > /tmp/vpp_requirements.no_torch.txt
pip install \
  --no-cache-dir \
  --timeout "${PIP_DEFAULT_TIMEOUT}" \
  --retries "${PIP_RETRIES}" \
  -r /tmp/vpp_requirements.no_torch.txt
pip install accelerate "huggingface_hub[cli]"

# Optional
pip install wandb
```

If your pip downloads still time out in this container, increase timeouts further:

```bash
export PIP_DEFAULT_TIMEOUT=3600
export PIP_RETRIES=50
```

### 3) Configure `accelerate`

Non-interactive default config (may import `transformers`):

```bash
accelerate config default
```

If `accelerate config` prints NumPy ABI errors/warnings on your machine, skip it and write a config
directly (recommended on SSH VMs):

```bash
# For a single node with 4× A800
bash scripts/configure_accelerate_ssh_4gpu_a800.sh
```

Interactive config (if you prefer):

```bash
accelerate config
```

Use these answers for a single node:

* compute environment: `LOCAL_MACHINE`
* distributed: `MULTI_GPU`
* processes: `4` (for 4 GPUs) or `8` (for 8 GPUs)
* mixed precision: `fp16` (or `bf16` on A800/A100/H100)

### 4) Install CALVIN + dataset

Actor training + env evaluation require CALVIN.

```bash
cd ~
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=~/calvin
cd "$CALVIN_ROOT"
sh install.sh
```

If CALVIN install fails with:

* `error in pyhash setup command: use_2to3 is invalid.`

it is usually due to a too-new `setuptools` in your env. Fix by pinning `setuptools<58` and rerun:

```bash
conda activate vpp
export CALVIN_ROOT=~/calvin
bash scripts/install_calvin_with_compat_pins.sh
```

If CALVIN install fails while building `pyhash` and mentions your pip index is HTTP or “not a trusted host”,
configure an HTTPS mirror first:

```bash
conda activate vpp
bash scripts/configure_pip_mirror_cn.sh
export CALVIN_ROOT=~/calvin
bash scripts/install_calvin_with_compat_pins.sh
```

##### aibox-pytorch container note (Torch 2.5.1 already installed)

In the `aibox-pytorch` container, CALVIN’s upstream `install.sh` may try to download a pinned
`torch==1.13.1` wheel from pip, which is large and can time out.

Use the repo’s installer wrapper with the **no-deps** mode to avoid re-installing torch:

```bash
export CALVIN_ROOT=/root/workspace/calvin   # adjust to your clone path
export CALVIN_INSTALL_MODE=no_deps
bash scripts/install_calvin_with_compat_pins.sh
```

This installs CALVIN components in editable mode but skips dependency resolution for `calvin_models`
(so it does not force `torch==1.13.1`).

If the install appears to **hang downloading PyTorch** (e.g. `torch-1.13.1-...whl` at only a few kB/s),
switch to a different mirror and retry. For example, Aliyun is often faster than TUNA on some networks:

```bash
conda activate vpp
PIP_MIRROR=aliyun bash scripts/configure_pip_mirror_cn.sh
export CALVIN_ROOT=~/calvin
bash scripts/install_calvin_with_compat_pins.sh
```

Alternative: preinstall torch via conda (often faster/more reliable than large wheels over pip), then rerun
the CALVIN installer; it should reuse the existing torch install:

```bash
conda activate vpp
conda install -y pytorch==1.13.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
export CALVIN_ROOT=~/calvin
bash scripts/install_calvin_with_compat_pins.sh
```

Set the dataset path env var (your smaller D/D split):

```bash
# Change this to wherever your smaller D/D split actually lives on the server.
export CALVIN_ROOT_DATA_DIR=/data/calvin/task_D_D
ls -la "$CALVIN_ROOT_DATA_DIR"
```

### 5) Download models

You need at minimum:

* CLIP text encoder (e.g. `openai/clip-vit-base-patch32`)
* SVD base model for video finetuning (the upstream config expects a local diffusers directory)

Example (CLIP):

```bash
mkdir -p ~/models
hf auth login
hf download openai/clip-vit-base-patch32 --local-dir ~/models/clip-vit-base-patch32 --local-dir-use-symlinks False
export CLIP_MODEL=~/models/clip-vit-base-patch32
```

SVD base model (`SVD_BASE_MODEL`): upstream training expects a **local diffusers directory**.

Example source (Hugging Face; may require accepting model terms):

```bash
mkdir -p ~/models
hf download stabilityai/stable-video-diffusion-img2vid --local-dir ~/models/stable-video-diffusion-img2vid --local-dir-use-symlinks False
export SVD_BASE_MODEL=~/models/stable-video-diffusion-img2vid
```

Then:

```bash
export SVD_BASE_MODEL=~/models/stable-video-diffusion-img2vid
```

### 6) Point paths + run experiments

Export these (or pass inline):

```bash
mkdir -p ~/exp/vpp
export OUTPUT_ROOT=~/exp/vpp
```

### 7) Obtain `VIDEO_DATASET_DIR` (latent-video dataset)

Fastest path is downloading CALVIN latents from the released HF dataset:

```bash
mkdir -p /data/vpp_svd_latent
hf download yjguo/vpp_svd_latent \
  --repo-type dataset \
  --local-dir /data/vpp_svd_latent \
  --local-dir-use-symlinks False \
  --include "calvin/**"
export VIDEO_DATASET_DIR=/data/vpp_svd_latent
ls -la "$VIDEO_DATASET_DIR/calvin" || true
```

### 8) Run Task 1

```bash
cd "${REPO_ROOT:-~/workspace/vpp-test-repo}"

# Example preset: 8× A800
PRESET=a800_8gpu \
VIDEO_DATASETS=calvin VIDEO_DATASET_PROB='[1.0]' \
bash scripts/run_task1_train_vpp_calvin_d2d.sh
```

### 9) Run Task 2

```bash
cd "${REPO_ROOT:-~/workspace/vpp-test-repo}"

PRESET=a800_8gpu \
VIDEO_DATASETS=calvin VIDEO_DATASET_PROB='[1.0]' \
VIDEO_A_SEED=42 VIDEO_B_SEED=123 ACTOR_A_SEED=456 \
bash scripts/run_task2_video_seed_fixed_actor.sh
```

## Configuration checklist (what you *must* set)

The launch scripts are thin wrappers; most “configuration” is just environment variables + choosing a preset.

### Required environment variables

| Variable | Meaning | Example |
|---|---|---|
| `VIDEO_DATASET_DIR` | Latent-video dataset root used by upstream SVD finetuning loader | `/data/vpp_svd_latent` |
| `CALVIN_ROOT_DATA_DIR` | Calvin dataset root for actor training + eval (your smaller D/D split) | `/data/calvin/task_D_D` |
| `SVD_BASE_MODEL` | Base SVD diffusers directory (local path) | `~/models/stable-video-diffusion-img2vid` |
| `CLIP_MODEL` | CLIP model directory | `~/models/clip-vit-base-patch32` |
| `OUTPUT_ROOT` | Where all outputs go | `~/exp/vpp` |

### Where to get each path (what the env vars point to)

#### `CALVIN_ROOT_DATA_DIR` (smaller D/D split)

After downloading CALVIN datasets following the upstream CALVIN instructions, point this variable at the dataset root for your **smaller D/D split**.

Typical pattern:

```bash
export CALVIN_ROOT_DATA_DIR=/data/calvin/task_D_D
ls -la "$CALVIN_ROOT_DATA_DIR"
```

#### `VIDEO_DATASET_DIR` (latent-video dataset root for SVD finetune)

This is the dataset root consumed by VPP’s video finetuning loader. It should contain dataset folders with `annotation/` and pre-encoded latent videos.

Common ways to obtain it:

1) Download the precomputed latents released by the VPP authors (the upstream VPP README links a Hugging Face dataset).
2) Generate latents yourself from raw videos using the upstream latent-preparation pipeline.

##### Option A (recommended): download the released latents from Hugging Face

The upstream VPP README references the dataset **`yjguo/vpp_svd_latent`**.

Example:

```bash
# If needed (first time on this machine):
hf auth login

mkdir -p /data/vpp_svd_latent
hf download yjguo/vpp_svd_latent \
  --repo-type dataset \
  --local-dir /data/vpp_svd_latent \
  --local-dir-use-symlinks False
export VIDEO_DATASET_DIR=/data/vpp_svd_latent
```

If you only need CALVIN-related files, you can try to download a subset (exact paths depend on the dataset layout):

```bash
hf download yjguo/vpp_svd_latent \
  --repo-type dataset \
  --local-dir /data/vpp_svd_latent \
  --include "calvin/**"
```

Do you need the *entire* dataset repo?

* Only if your video finetune config mixes multiple datasets.
* If you want CALVIN-only finetuning, you can run the launchers with:

```bash
export VIDEO_DATASETS=calvin
export VIDEO_DATASET_PROB='[1.0]'
```

and then downloading only the CALVIN subset is sufficient.

What “download only the CALVIN subset” means:

* The Hugging Face dataset repo `yjguo/vpp_svd_latent` contains multiple dataset folders (e.g. `sthv2/`, `bridge/`, `rt1/`, `calvin/`, …).
* The upstream video finetuning loader reads from `VIDEO_DATASET_DIR` and then picks which dataset folder(s) to use via `train_args.dataset`.
* If you only train on `calvin`, you only need the files under the `calvin/` directory inside the HF dataset, not the entire repo.

Concretely, you can download only the `calvin/` directory by using `--include "calvin/**"`:

```bash
mkdir -p /data/vpp_svd_latent
hf download yjguo/vpp_svd_latent \
  --repo-type dataset \
  --local-dir /data/vpp_svd_latent \
  --local-dir-use-symlinks False \
  --include "calvin/**"
export VIDEO_DATASET_DIR=/data/vpp_svd_latent
```

After this, you should see a `/data/vpp_svd_latent/calvin/` directory. The exact internal layout under `calvin/` is defined by what the dataset author uploaded and what upstream expects.

##### Option B: build your own latent-video dataset

This requires converting raw robot videos into the dataset structure expected by upstream
[`video_dataset/dataset_mix.py`](../video-prediction-policy/video_dataset/dataset_mix.py:41) (JSON annotations + videos + latent `.pt`).
Use the upstream latent-preparation scripts as a starting point.

Then:

```bash
export VIDEO_DATASET_DIR=/data/vpp_svd_latent
ls -la "$VIDEO_DATASET_DIR"
```

#### `CLIP_MODEL`

Download from Hugging Face:

```bash
mkdir -p ~/models
hf download openai/clip-vit-base-patch32 --local-dir ~/models/clip-vit-base-patch32 --local-dir-use-symlinks False
export CLIP_MODEL=~/models/clip-vit-base-patch32
```

#### `SVD_BASE_MODEL`

Download from Hugging Face (example):

```bash
mkdir -p ~/models
hf download stabilityai/stable-video-diffusion-img2vid --local-dir ~/models/stable-video-diffusion-img2vid --local-dir-use-symlinks False
export SVD_BASE_MODEL=~/models/stable-video-diffusion-img2vid
```

#### `OUTPUT_ROOT`

Any directory with sufficient disk quota:

```bash
export OUTPUT_ROOT=~/exp/vpp
mkdir -p "$OUTPUT_ROOT"
```

### Presets (GPU configs)

Presets are defined in [`config/cluster_presets.yaml`](config/cluster_presets.yaml) and mirrored in the launcher `case` statements.

* `PRESET=default`
* `PRESET=rtx5090_4gpu_reduced`
* `PRESET=a800_8gpu`

If you need to tune memory:

* video finetune batch size is controlled inside [`run_task1_train_vpp_calvin_d2d.sh`](run_task1_train_vpp_calvin_d2d.sh)
  via `SVD_TRAIN_BS`.
* actor batch size is passed as a Hydra override into [`train_actor_calvin.py`](train_actor_calvin.py)
  (example: `batch_size=12`).

### What config files are actually used (upstream)

You normally do **not** edit upstream configs, but it helps to know what’s being driven:

* Video finetune uses upstream config:
  * [`../video-prediction-policy/video_conf/train_calvin_svd.yaml`](../video-prediction-policy/video_conf/train_calvin_svd.yaml)
* Actor training uses upstream config:
  * [`../video-prediction-policy/policy_conf/VPP_Calvinabc_train.yaml`](../video-prediction-policy/policy_conf/VPP_Calvinabc_train.yaml)

Our wrapper [`train_actor_calvin.py`](train_actor_calvin.py) overrides the dataset/model/log paths in code, without modifying upstream files.

Run Task 1 (reproduce training):

```bash
PRESET=a800_8gpu bash scripts/run_task1_train_vpp_calvin_d2d.sh
```

Run Task 2 (seed sensitivity):

```bash
PRESET=a800_8gpu \
VIDEO_A_SEED=42 VIDEO_B_SEED=123 ACTOR_A_SEED=456 \
bash scripts/run_task2_video_seed_fixed_actor.sh
```

## Remote SSH setup (minimal)

Deprecated; use the QUICK START above.

## Sanity checks

```bash
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'n_gpus', torch.cuda.device_count())"
python3 -c "import accelerate; print('accelerate', accelerate.__version__)"
```

Optional preflight checker (catches missing `diffusers` / missing env vars early):

```bash
python3 scripts/check_setup.py
```

If you see `ModuleNotFoundError: diffusers`, ensure you ran:

```bash
pip install -r video-prediction-policy/requirements.txt
```


## Task 1 — Reproduce VPP training on the smaller D/D split

You typically run the two VPP stages:

1) Video model finetune: [`../video-prediction-policy/step1_train_svd.py`](../video-prediction-policy/step1_train_svd.py)
2) Actor/policy training: wrapper [`train_actor_calvin.py`](train_actor_calvin.py)

Launch script: [`run_task1_train_vpp_calvin_d2d.sh`](run_task1_train_vpp_calvin_d2d.sh)

Required environment variables:

* `VIDEO_DATASET_DIR`: latent-video dataset root used by VPP’s SVD training loader
* `CALVIN_ROOT_DATA_DIR`: Calvin dataset root for actor training (your smaller D/D split)
* `SVD_BASE_MODEL`: base SVD diffusers path
* `CLIP_MODEL`: CLIP model path
* `OUTPUT_ROOT`: where you want outputs

Example (8× A800):

```bash
PRESET=a800_8gpu \
VIDEO_DATASET_DIR=/data/vpp_svd_latent \
CALVIN_ROOT_DATA_DIR=/data/calvin/task_D_D \
SVD_BASE_MODEL=/models/stable-video-diffusion-img2vid \
CLIP_MODEL=/models/clip-vit-base-patch32 \
OUTPUT_ROOT=/exp/vpp_calvin_dd \
bash scripts/run_task1_train_vpp_calvin_d2d.sh
```

## Task 2 — Seed sensitivity: [Video A + Actor A] vs [Video B + Actor A]

Launch script: [`run_task2_video_seed_fixed_actor.sh`](run_task2_video_seed_fixed_actor.sh)

Example:

```bash
PRESET=rtx5090_4gpu_reduced \
VIDEO_DATASET_DIR=/data/vpp_svd_latent \
CALVIN_ROOT_DATA_DIR=/data/calvin/task_D_D \
SVD_BASE_MODEL=/models/stable-video-diffusion-img2vid \
CLIP_MODEL=/models/clip-vit-base-patch32 \
OUTPUT_ROOT=/exp/vpp_seed_test \
VIDEO_A_SEED=42 VIDEO_B_SEED=123 ACTOR_A_SEED=456 \
bash scripts/run_task2_video_seed_fixed_actor.sh
```

## Notes / limitations

* The video SVD training data loader in upstream VPP has some CALVIN-specific assumptions.
  If your “smaller D/D dataset” is organized differently from what upstream expects, you may need
  to provide a compatible latent-video dataset layout under `VIDEO_DATASET_DIR`.
