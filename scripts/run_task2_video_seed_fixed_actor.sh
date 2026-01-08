#!/usr/bin/env bash
set -euo pipefail

# Task 2: Seed sensitivity test with a FIXED actor.
#
# Experiment:
#   1) Train Video A (seed A)
#   2) Train Video B (seed B)
#   3) Train Actor A using Video A
#   4) Evaluate [Video A + Actor A] vs [Video B + Actor A]
#
# Usage:
#   PRESET=a800_8gpu \
#   VIDEO_DATASET_DIR=/data/vpp_svd_latent \
#   CALVIN_ROOT_DATA_DIR=/data/calvin/task_D_D \
#   SVD_BASE_MODEL=/models/stable-video-diffusion-img2vid \
#   CLIP_MODEL=/models/clip-vit-base-patch32 \
#   OUTPUT_ROOT=/exp/vpp_seed_test \
#   VIDEO_A_SEED=42 VIDEO_B_SEED=123 ACTOR_A_SEED=456 \
#   bash scripts/run_task2_video_seed_fixed_actor.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VPP_DIR="${VPP_DIR:-${REPO_ROOT}/video-prediction-policy}"

PRESET="${PRESET:-default}"

: "${VIDEO_DATASET_DIR:?Set VIDEO_DATASET_DIR (latent-video dataset root for step1_train_svd)}"
: "${CALVIN_ROOT_DATA_DIR:?Set CALVIN_ROOT_DATA_DIR (Calvin task_D_D root for actor training)}"
: "${SVD_BASE_MODEL:?Set SVD_BASE_MODEL (diffusers SVD base model path)}"
: "${CLIP_MODEL:?Set CLIP_MODEL (clip-vit-base-patch32 path)}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT (experiment output root)}"

MAIN_PORT="${MAIN_PORT:-29516}"

VIDEO_A_SEED="${VIDEO_A_SEED:-42}"
VIDEO_B_SEED="${VIDEO_B_SEED:-123}"
ACTOR_A_SEED="${ACTOR_A_SEED:-456}"

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

VIDEO_A_OUT="${OUTPUT_ROOT}/video_A_seed_${VIDEO_A_SEED}"
VIDEO_B_OUT="${OUTPUT_ROOT}/video_B_seed_${VIDEO_B_SEED}"
ACTOR_A_OUT="${OUTPUT_ROOT}/actor_A_from_video_A_seed_${VIDEO_A_SEED}"

echo "[Task2] PRESET=${PRESET} GPUs=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "[Task2][1/4] Train Video A (seed=${VIDEO_A_SEED})"
accelerate launch \
  --num_processes="${NUM_GPUS}" \
  --main_process_port="${MAIN_PORT}" \
  "${VPP_DIR}/step1_train_svd.py" \
  --config "${VPP_DIR}/video_conf/train_calvin_svd.yaml" \
  pretrained_model_path="${SVD_BASE_MODEL}" \
  output_dir="${VIDEO_A_OUT}" \
  mixed_precision="${MIXED_PRECISION}" \
  train_batch_size="${SVD_TRAIN_BS}" \
  gradient_checkpointing="${SVD_GRAD_CKPT}" \
  seed="${VIDEO_A_SEED}" \
  train_args.dataset_dir="${VIDEO_DATASET_DIR}"

echo "[Task2][2/4] Train Video B (seed=${VIDEO_B_SEED})"
accelerate launch \
  --num_processes="${NUM_GPUS}" \
  --main_process_port="$((MAIN_PORT + 1))" \
  "${VPP_DIR}/step1_train_svd.py" \
  --config "${VPP_DIR}/video_conf/train_calvin_svd.yaml" \
  pretrained_model_path="${SVD_BASE_MODEL}" \
  output_dir="${VIDEO_B_OUT}" \
  mixed_precision="${MIXED_PRECISION}" \
  train_batch_size="${SVD_TRAIN_BS}" \
  gradient_checkpointing="${SVD_GRAD_CKPT}" \
  seed="${VIDEO_B_SEED}" \
  train_args.dataset_dir="${VIDEO_DATASET_DIR}"

echo "[Task2][3/4] Train Actor A with Video A (actor_seed=${ACTOR_A_SEED})"
mkdir -p "${ACTOR_A_OUT}"
accelerate launch \
  --num_processes="${NUM_GPUS}" \
  --main_process_port="$((MAIN_PORT + 2))" \
  "${REPO_ROOT}/scripts/train_actor_calvin.py" \
  --vpp_dir "${VPP_DIR}" \
  --root_data_dir "${CALVIN_ROOT_DATA_DIR}" \
  --video_model_path "${VIDEO_A_OUT}" \
  --text_encoder_path "${CLIP_MODEL}" \
  --log_dir "${ACTOR_A_OUT}" \
  --seed "${ACTOR_A_SEED}" \
  batch_size="${ACTOR_BATCH_SIZE}"

echo "[Task2][4/4] Evaluate [Video A + Actor A] vs [Video B + Actor A]"
EVAL_OUT="${OUTPUT_ROOT}/eval_fixed_actor_A"
mkdir -p "${EVAL_OUT}"

python3 "${VPP_DIR}/policy_evaluation/calvin_evaluate.py" \
  --video_model_path "${VIDEO_A_OUT}" \
  --action_model_folder "${ACTOR_A_OUT}" \
  --clip_model_path "${CLIP_MODEL}" \
  --calvin_abc_dir "${CALVIN_ROOT_DATA_DIR}"

python3 "${VPP_DIR}/policy_evaluation/calvin_evaluate.py" \
  --video_model_path "${VIDEO_B_OUT}" \
  --action_model_folder "${ACTOR_A_OUT}" \
  --clip_model_path "${CLIP_MODEL}" \
  --calvin_abc_dir "${CALVIN_ROOT_DATA_DIR}"

echo "[Task2] Done. Inspect eval logs under the actor folder and VPP evaluation output directory."
