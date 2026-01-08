#!/usr/bin/env bash
set -euo pipefail

# Task 1: Reproduce VPP training on the Calvin D→D split.
#
# This script is intentionally simple and relies on environment variables for paths.
# It launches:
#   (1) SVD video model finetune (step1_train_svd.py)
#   (2) Actor/policy training (step2_train_action_calvin.py)
#
# Usage:
#   PRESET=rtx5090_4gpu_reduced STAGE=all \
#   VIDEO_DATASET_DIR=/data/vpp_svd_latent \
#   CALVIN_ROOT_DATA_DIR=/data/calvin/task_D_D \
#   SVD_BASE_MODEL=/models/stable-video-diffusion-img2vid \
#   CLIP_MODEL=/models/clip-vit-base-patch32 \
#   OUTPUT_ROOT=/exp/vpp_calvin_d2d \
#   bash scripts/run_task1_train_vpp_calvin_d2d.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VPP_DIR="${VPP_DIR:-${REPO_ROOT}/video-prediction-policy}"

PRESET="${PRESET:-default}"
STAGE="${STAGE:-all}"  # video|actor|all

# Required paths
: "${VIDEO_DATASET_DIR:?Set VIDEO_DATASET_DIR (latent-video dataset root for step1_train_svd)}"
: "${CALVIN_ROOT_DATA_DIR:?Set CALVIN_ROOT_DATA_DIR (Calvin task_D_D root for actor training)}"
: "${SVD_BASE_MODEL:?Set SVD_BASE_MODEL (diffusers SVD base model path)}"
: "${CLIP_MODEL:?Set CLIP_MODEL (clip-vit-base-patch32 path)}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT (experiment output root)}"

MAIN_PORT="${MAIN_PORT:-29506}"
VIDEO_SEED="${VIDEO_SEED:-42}"

case "${PRESET}" in
  default)
    NUM_GPUS=8
    CUDA_VISIBLE_DEVICES_VALUE="0,1,2,3,4,5,6,7"
    MIXED_PRECISION="fp16"
    SVD_TRAIN_BS=6
    SVD_GRAD_CKPT=true
    ACTOR_BATCH_SIZE=28
    ;;
  rtx5090_4gpu_reduced)
    NUM_GPUS=4
    CUDA_VISIBLE_DEVICES_VALUE="0,1,2,3"
    MIXED_PRECISION="fp16"
    SVD_TRAIN_BS=2
    SVD_GRAD_CKPT=true
    ACTOR_BATCH_SIZE=12
    ;;
  a800_8gpu)
    NUM_GPUS=8
    CUDA_VISIBLE_DEVICES_VALUE="0,1,2,3,4,5,6,7"
    MIXED_PRECISION="bf16"
    SVD_TRAIN_BS=6
    SVD_GRAD_CKPT=true
    ACTOR_BATCH_SIZE=28
    ;;
  *)
    echo "Unknown PRESET='${PRESET}'. Valid: default | rtx5090_4gpu_reduced | a800_8gpu" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"

mkdir -p "${OUTPUT_ROOT}"

VIDEO_OUT="${OUTPUT_ROOT}/video_seed_${VIDEO_SEED}"

if [[ "${STAGE}" == "video" || "${STAGE}" == "all" ]]; then
  echo "[Task1][video] PRESET=${PRESET} GPUs=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[Task1][video] Output: ${VIDEO_OUT}"

  accelerate launch \
    --num_processes="${NUM_GPUS}" \
    --main_process_port="${MAIN_PORT}" \
    "${VPP_DIR}/step1_train_svd.py" \
    --config "${VPP_DIR}/video_conf/train_calvin_svd.yaml" \
    pretrained_model_path="${SVD_BASE_MODEL}" \
    output_dir="${VIDEO_OUT}" \
    mixed_precision="${MIXED_PRECISION}" \
    train_batch_size="${SVD_TRAIN_BS}" \
    gradient_checkpointing="${SVD_GRAD_CKPT}" \
    seed="${VIDEO_SEED}" \
    train_args.dataset_dir="${VIDEO_DATASET_DIR}"
fi

if [[ "${STAGE}" == "actor" || "${STAGE}" == "all" ]]; then
  # The actor script saves checkpoints under a run directory; for reproducibility,
  # point log_dir at a dedicated folder.
  ACTOR_OUT="${OUTPUT_ROOT}/actor_from_video_seed_${VIDEO_SEED}"
  mkdir -p "${ACTOR_OUT}"

  echo "[Task1][actor] PRESET=${PRESET} GPUs=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[Task1][actor] Output: ${ACTOR_OUT}"

  accelerate launch \
    --num_processes="${NUM_GPUS}" \
    --main_process_port="$((MAIN_PORT + 1))" \
    "${REPO_ROOT}/scripts/train_actor_calvin.py" \
    --vpp_dir "${VPP_DIR}" \
    --root_data_dir "${CALVIN_ROOT_DATA_DIR}" \
    --video_model_path "${VIDEO_OUT}" \
    --text_encoder_path "${CLIP_MODEL}" \
    --log_dir "${ACTOR_OUT}" \
    --seed "${VIDEO_SEED}" \
    batch_size="${ACTOR_BATCH_SIZE}"
fi
