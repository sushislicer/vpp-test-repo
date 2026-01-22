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

## QUICK START (Container Environment)

This guide assumes you are running in the **aibox-pytorch container** (`registry.baidubce.com/inference/aibox-pytorch:v1.0-torch2.5.1-cu12.4`) on a machine with **4× A800 80GB GPUs**.

The repo should be cloned at: `~/workspace/vpp-test-repo`

### 1) Setup Environment & Dependencies

Run these commands inside the container to set up the environment without breaking the pre-installed PyTorch.

```bash
cd ~/workspace/vpp-test-repo

# 1. Configure pip for China/GFW (HTTPS mirror, resilient settings)
bash scripts/configure_pip_mirror_cn.sh
export PIP_CONFIG_FILE=/dev/null
export PIP_DEFAULT_TIMEOUT=600
export PIP_RETRIES=20

# 2. Install VPP dependencies (skipping torch to preserve container's version)
#    We filter out 'torch==' from requirements.txt to avoid downgrading.
grep -v -E '^torch==' video-prediction-policy/requirements.txt > /tmp/vpp_requirements.no_torch.txt
pip install --no-cache-dir -r /tmp/vpp_requirements.no_torch.txt
pip install accelerate "huggingface_hub[cli]" wandb

# 3. Verify setup
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'n_gpus', torch.cuda.device_count())"
```

### 2) Install CALVIN (Python 3.8 Env)

CALVIN requires Python 3.8. Since the container uses Python 3.11, we create a separate conda environment just for CALVIN installation, then expose the source code to the main environment.

```bash
# 1. Create Python 3.8 env for CALVIN
conda create -n calvin38 python=3.8 -y
conda activate calvin38

# 2. Clone CALVIN
cd ~
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=~/calvin

# 3. Install CALVIN (using no_deps mode to avoid heavy/conflicting deps)
#    This installs the package in editable mode without forcing torch==1.13.1
cd ~/workspace/vpp-test-repo
export CALVIN_INSTALL_MODE=no_deps
bash scripts/install_calvin_with_compat_pins.sh

# 4. Deactivate and return to base env
conda deactivate
```

### 3) Configure Paths & Download Models

Set up the required environment variables and download the necessary models.

```bash
# --- PATH CONFIGURATION ---
export REPO_ROOT=~/workspace/vpp-test-repo
export CALVIN_ROOT=~/calvin
# Expose CALVIN source to the main Python environment
export PYTHONPATH="${CALVIN_ROOT}:${PYTHONPATH:-}"

# Verify CALVIN import works in main env
python3 -c "import calvin_env; print('calvin_env import OK from', calvin_env.__file__)"

# --- DATASET PATHS (Update these to match your actual paths) ---
# 1. Calvin D/D Split (for Actor training)
export CALVIN_ROOT_DATA_DIR=/data/calvin/task_D_D

# 2. Latent Video Dataset (for SVD finetuning)
#    If you haven't downloaded it yet:
#    mkdir -p /data/vpp_svd_latent
#    hf download yjguo/vpp_svd_latent --repo-type dataset --local-dir /data/vpp_svd_latent --local-dir-use-symlinks False --include "calvin/**"
export VIDEO_DATASET_DIR=/data/vpp_svd_latent

# --- MODEL DOWNLOADS ---
mkdir -p ~/models

# 1. CLIP Model
hf download openai/clip-vit-base-patch32 --local-dir ~/models/clip-vit-base-patch32 --local-dir-use-symlinks False
export CLIP_MODEL=~/models/clip-vit-base-patch32

# 2. SVD Base Model
hf download stabilityai/stable-video-diffusion-img2vid --local-dir ~/models/stable-video-diffusion-img2vid --local-dir-use-symlinks False
export SVD_BASE_MODEL=~/models/stable-video-diffusion-img2vid

# --- OUTPUT DIRECTORY ---
export OUTPUT_ROOT=~/exp/vpp
mkdir -p "$OUTPUT_ROOT"
```

### 4) Run Experiments

Now you can run the tasks. Ensure you are in the repo root and the environment variables from Step 3 are set.

**Task 1: Reproduce VPP Training (D/D Split)**

```bash
cd ~/workspace/vpp-test-repo

# Using a800_8gpu preset (adjusts batch sizes/precision for A800)
PRESET=a800_8gpu \
VIDEO_DATASETS=calvin VIDEO_DATASET_PROB='[1.0]' \
bash scripts/run_task1_train_vpp_calvin_d2d.sh
```

**Task 2: Seed Sensitivity Test**

```bash
cd ~/workspace/vpp-test-repo

PRESET=a800_8gpu \
VIDEO_DATASETS=calvin VIDEO_DATASET_PROB='[1.0]' \
VIDEO_A_SEED=42 VIDEO_B_SEED=123 ACTOR_A_SEED=456 \
bash scripts/run_task2_video_seed_fixed_actor.sh
```

## Configuration Reference

| Variable | Meaning | Example |
|---|---|---|
| `VIDEO_DATASET_DIR` | Latent-video dataset root used by upstream SVD finetuning loader | `/data/vpp_svd_latent` |
| `CALVIN_ROOT_DATA_DIR` | Calvin dataset root for actor training + eval (your smaller D/D split) | `/data/calvin/task_D_D` |
| `SVD_BASE_MODEL` | Base SVD diffusers directory (local path) | `~/models/stable-video-diffusion-img2vid` |
| `CLIP_MODEL` | CLIP model directory | `~/models/clip-vit-base-patch32` |
| `OUTPUT_ROOT` | Where all outputs go | `~/exp/vpp` |

### Presets (GPU configs)

Presets are defined in [`config/cluster_presets.yaml`](config/cluster_presets.yaml).

* `PRESET=default` (8 GPUs, fp16)
* `PRESET=rtx5090_4gpu_reduced` (4 GPUs, fp16, smaller batch size)
* `PRESET=a800_8gpu` (8 GPUs, bf16)

## Troubleshooting

* **`ModuleNotFoundError: calvin_env`**: Ensure you exported `PYTHONPATH="${CALVIN_ROOT}:${PYTHONPATH:-}"`.
* **`ReadTimeoutError` during pip install**: The script sets `PIP_DEFAULT_TIMEOUT=600`. If that's not enough, export `PIP_DEFAULT_TIMEOUT=3600` and try again.
* **Hash mismatch errors**: Run `python3 -m pip cache purge` and try again.
