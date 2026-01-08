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

This section is designed to be copy/paste-friendly on a fresh server.

### 0) System prerequisites

You need:

* NVIDIA driver + CUDA runtime working (`nvidia-smi`)
* A conda installation (Miniconda/Mambaforge)

### 1) Clone repos

```bash
cd ~
git clone https://github.com/sushislicer/vpp-test-repo.git VPP
cd VPP
```

### 2) Create environment + install Python deps

```bash
conda create -n vpp python=3.10 -y
conda activate vpp

# Core deps for upstream VPP
pip install -r video-prediction-policy/requirements.txt
pip install accelerate huggingface-hub

# (Optional but common)
pip install wandb
```

### 3) Configure `accelerate`

Run once per machine/user:

```bash
accelerate config
```

Suggested answers (single node):

* compute environment: LOCAL_MACHINE
* distributed training: MULTI_GPU
* mixed precision: fp16 (or bf16 on A800/A100/H100)

### 4) Install CALVIN + dataset

Actor training + env evaluation require CALVIN.

```bash
cd ~
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=~/calvin
cd "$CALVIN_ROOT"
sh install.sh
```

Dataset:

* Follow CALVIN’s dataset instructions.
* For this project you want the **smaller D/D split (~166GB)**.

Expected env var used by our launchers:

* `CALVIN_ROOT_DATA_DIR=/path/to/calvin/task_D_D`

### 5) Download models

You need at minimum:

* CLIP text encoder (e.g. `openai/clip-vit-base-patch32`)
* SVD base model for video finetuning (the upstream config expects a local diffusers directory)

Example (CLIP):

```bash
mkdir -p ~/models
huggingface-cli download openai/clip-vit-base-patch32 --local-dir ~/models/clip-vit-base-patch32
```

For the SVD base model, set `SVD_BASE_MODEL` to a local diffusers checkout that matches your upstream training config.

### 6) Point paths + run experiments

Export these (or pass inline):

```bash
export VIDEO_DATASET_DIR=/data/vpp_svd_latent
export CALVIN_ROOT_DATA_DIR=/data/calvin/task_D_D
export SVD_BASE_MODEL=~/models/stable-video-diffusion-img2vid
export CLIP_MODEL=~/models/clip-vit-base-patch32
export OUTPUT_ROOT=~/exp/vpp
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

From a fresh clone under `~/`:

```bash
conda create -n vpp python=3.10 -y
conda activate vpp

pip install -r video-prediction-policy/requirements.txt
pip install accelerate
```

Install CALVIN if you need environment evaluation.

## Sanity checks

```bash
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'n_gpus', torch.cuda.device_count())"
python3 -c "import accelerate; print('accelerate', accelerate.__version__)"
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
