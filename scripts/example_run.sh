#!/bin/bash
# Example script to run Calvin D -> D benchmark with VPP on 4 GPUs

# Set your paths here
VIDEO_MODEL_PATH="/path/to/svd-robot-calvin"
ACTION_MODEL_FOLDER="/path/to/dp-calvin"
CLIP_MODEL_PATH="/path/to/clip-vit-base-patch32"
CALVIN_D2D_DIR="/path/to/calvin/task_D_D"

# Optional: Set custom log directory
LOG_DIR="./logs/calvin_d2d"

# Optional: Set wandb entity
# export WANDB_ENTITY="your-wandb-username"

# Run the benchmark on 4 GPUs
accelerate launch --num_processes=4 \
    --main_process_port=29506 \
    run_calvin_d2d_benchmark.py \
    --config config/calvin_d2d_config.yaml \
    --video_model_path "$VIDEO_MODEL_PATH" \
    --action_model_folder "$ACTION_MODEL_FOLDER" \
    --clip_model_path "$CLIP_MODEL_PATH" \
    --calvin_d2d_dir "$CALVIN_D2D_DIR" \
    --log_dir "$LOG_DIR" \
    --num_videos 10 \
    --num_sequences 1000

# For testing with fewer sequences, modify the command:
# accelerate launch --num_processes=4 \
#     --main_process_port=29506 \
#     run_calvin_d2d_benchmark.py \
#     --config config/calvin_d2d_config.yaml \
#     --video_model_path "$VIDEO_MODEL_PATH" \
#     --action_model_folder "$ACTION_MODEL_FOLDER" \
#     --clip_model_path "$CLIP_MODEL_PATH" \
#     --calvin_d2d_dir "$CALVIN_D2D_DIR" \
#     --log_dir "$LOG_DIR" \
#     --num_videos 5 \
#     --num_sequences 100

# For single GPU testing:
# python run_calvin_d2d_benchmark.py \
#     --config config/calvin_d2d_config.yaml \
#     --video_model_path "$VIDEO_MODEL_PATH" \
#     --action_model_folder "$ACTION_MODEL_FOLDER" \
#     --clip_model_path "$CLIP_MODEL_PATH" \
#     --calvin_d2d_dir "$CALVIN_D2D_DIR" \
#     --log_dir "$LOG_DIR" \
#     --num_videos 5 \
#     --num_sequences 100

# For debug mode:
# python run_calvin_d2d_benchmark.py \
#     --config config/calvin_d2d_config.yaml \
#     --video_model_path "$VIDEO_MODEL_PATH" \
#     --action_model_folder "$ACTION_MODEL_FOLDER" \
#     --clip_model_path "$CLIP_MODEL_PATH" \
#     --calvin_d2d_dir "$CALVIN_D2D_DIR" \
#     --debug
