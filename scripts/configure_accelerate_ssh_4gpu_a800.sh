#!/usr/bin/env bash
set -euo pipefail

# Write a non-interactive HuggingFace Accelerate config suitable for:
# - single node
# - 4× NVIDIA A800 GPUs
# - running over SSH
#
# This avoids `accelerate config` interactive prompts and avoids importing
# heavy deps during configuration.
#
# Usage:
#   bash scripts/configure_accelerate_ssh_4gpu_a800.sh
#
# Optional overrides:
#   NUM_GPUS=4 MIXED_PRECISION=bf16 MAIN_PROCESS_PORT=29506 \
#     bash scripts/configure_accelerate_ssh_4gpu_a800.sh

NUM_GPUS="${NUM_GPUS:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29506}"

ACCEL_DIR="${HOME}/.cache/huggingface/accelerate"
ACCEL_CFG="${ACCEL_DIR}/default_config.yaml"

mkdir -p "${ACCEL_DIR}"

cat >"${ACCEL_CFG}" <<YAML
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_process_ip: null
main_process_port: ${MAIN_PROCESS_PORT}
mixed_precision: ${MIXED_PRECISION}
num_machines: 1
num_processes: ${NUM_GPUS}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML

echo "Wrote accelerate config: ${ACCEL_CFG}"
echo "Contents:"
sed -n '1,200p' "${ACCEL_CFG}"

