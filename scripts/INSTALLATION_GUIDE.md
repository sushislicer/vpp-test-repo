# Installation Guide for Calvin D -> D Benchmark

This guide provides step-by-step instructions for setting up the environment to run VPP on Calvin D -> D benchmark.

## Overview

Before running the benchmark scripts, you need to:
1. Install VPP dependencies and create conda environment
2. Install Calvin environment (optional but recommended)
3. Download pre-trained models from HuggingFace
4. Download Calvin dataset

## Step 1: Create Conda Environment

```bash
# Navigate to VPP directory
cd /home/yangc/Lab/VPP/video-prediction-policy

# Create conda environment with Python 3.10
conda create -n vpp python==3.10
conda activate vpp
```

## Step 2: Install Calvin Environment (Optional but Recommended)

**Note**: You can skip Calvin installation if you only want to run evaluation on pre-downloaded datasets. However, installing Calvin is recommended for full functionality.

```bash
# Navigate to VPP directory
cd /home/yangc/Lab/VPP

# Clone Calvin repository with submodules
git clone --recurse-submodules https://github.com/mees/calvin.git

# Set CALVIN_ROOT environment variable
export CALVIN_ROOT=$(pwd)/calvin

# Navigate to Calvin directory
cd $CALVIN_ROOT

# Install Calvin
sh install.sh

# Note: You may encounter some render issues during installation.
# Refer to the Calvin repository documentation for solutions.
```

**Important**: After installation, keep the `CALVIN_ROOT` environment variable set for future use:
```bash
# Add to your ~/.bashrc or ~/.zshrc for persistence
echo 'export CALVIN_ROOT=/home/yangc/Lab/VPP/calvin' >> ~/.bashrc
source ~/.bashrc
```

## Step 3: Install VPP Dependencies

```bash
# Navigate to VPP directory
cd /home/yangc/Lab/VPP/video-prediction-policy

# Install VPP requirements
pip install -r requirements.txt

# Note: Calvin requires torch==1.13, but it also works with torch>2.0
# You can ignore the warning if you're using a newer torch version
```

## Step 4: Install Accelerate for Distributed Training

```bash
# Install accelerate for multi-GPU support
pip install accelerate

# Optional: Configure accelerate (first-time setup)
accelerate config
```

## Step 5: Download Pre-trained Models

You need to download three pre-trained models from HuggingFace:

### 5.1 Install HuggingFace CLI (if not already installed)

```bash
pip install huggingface-hub
```

### 5.2 Download Models

#### Option A: Using HuggingFace CLI (Recommended)

```bash
# Create a directory for models
mkdir -p /path/to/models
cd /path/to/models

# Download CLIP text encoder (~600MB)
huggingface-cli download openai/clip-vit-base-patch32 \
    --local-dir clip-vit-base-patch32

# Download SVD video model finetuned on Calvin (~8GB)
huggingface-cli download yjguo/svd-robot-calvin-ft \
    --local-dir svd-robot-calvin

# Download action model trained on Calvin ABC (~1GB)
huggingface-cli download yjguo/dp-calvin \
    --local-dir dp-calvin
```

#### Option B: Manual Download

Visit these HuggingFace links and download manually:
- [CLIP ViT Base Patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- [SVD Robot Calvin](https://huggingface.co/yjguo/svd-robot-calvin-ft/tree/main)
- [DP Calvin](https://huggingface.co/yjguo/dp-calvin/tree/main)

### 5.3 Verify Model Downloads

```bash
# Check that models are downloaded
ls -lh /path/to/models/

# Expected output:
# clip-vit-base-patch32/  (~600MB)
# svd-robot-calvin/        (~8GB)
# dp-calvin/                  (~1GB)
```

## Step 6: Download Calvin Dataset

The Calvin D -> D dataset is approximately 500GB and contains multiple domains with different visual conditions.

### 6.1 Follow Official Calvin Instructions

Follow the instructions in the [official Calvin repository](https://github.com/mees/calvin) to download the dataset.

### 6.2 Expected Dataset Structure

The dataset should be downloaded to a location like `/path/to/calvin/task_D_D/` with the following structure:

```
/path/to/calvin/task_D_D/
├── domain_A/
│   ├── episodes.json
│   ├── episode_000/
│   │   ├── rgb.mp4
│   │   └── state.npy
│   ├── episode_001/
│   │   ├── rgb.mp4
│   │   └── state.npy
│   └── ...
├── domain_B/
│   ├── episodes.json
│   └── ...
├── domain_C/
│   ├── episodes.json
│   └── ...
└── domain_D/
    ├── episodes.json
    └── ...
```

### 6.3 Dataset Size and Storage

- **Total size**: ~500GB
- **Per domain**: ~125GB
- **Recommended storage**: SSD or NVMe for faster loading

### 6.4 Alternative: Pre-processed D -> D Dataset

If a pre-processed D -> D dataset is available, download it and ensure it follows the structure above.

## Step 7: Set Environment Variables

Set these environment variables for convenience:

```bash
# Set CUDA devices (adjust based on your available GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set Calvin root (if you installed Calvin)
export CALVIN_ROOT=/home/yangc/Lab/VPP/calvin

# Set wandb entity for logging (optional)
export WANDB_ENTITY=your-wandb-username
```

For persistence, add these to your `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export CUDA_VISIBLE_DEVICES=0,1,2,3' >> ~/.bashrc
echo 'export CALVIN_ROOT=/home/yangc/Lab/VPP/calvin' >> ~/.bashrc
echo 'export WANDB_ENTITY=your-wandb-username' >> ~/.bashrc
source ~/.bashrc
```

## Step 8: Verify Installation

Run the test script to verify your setup:

```bash
# Navigate to scripts directory
cd /home/yangc/Lab/VPP/scripts

# Basic test (checks imports and config)
python test_d2d_setup.py

# Full test with model and dataset paths
python test_d2d_setup.py \
    --video_model_path /path/to/models/svd-robot-calvin \
    --action_model_folder /path/to/models/dp-calvin \
    --clip_model_path /path/to/models/clip-vit-base-patch32 \
    --calvin_d2d_dir /path/to/calvin/task_D_D
```

Expected output:
```
===============================================================================
CALVIN D -> D BENCHMARK SETUP TEST
===============================================================================
Testing imports...
✓ Basic imports successful
✓ D -> D utils imports successful

Testing configuration file: config/calvin_d2d_config.yaml
✓ Config file loaded successfully
✓ All required fields present
✓ Found 16 domain transfer pairs

Testing D -> D utility functions...
✓ get_d2d_domains: 16 pairs
✓ get_domain_info: Domain A
✓ count_d2d_success: [0.2, 0.4, 0.6, 0.8, 1.0]
✓ calculate_transfer_metrics: 4 metric categories

===============================================================================
✓ ALL BASIC TESTS PASSED!
===============================================================================
```

## Step 9: Quick Setup Script

Create a setup script to automate the process:

```bash
#!/bin/bash
# setup_vpp_d2d.sh

# Set your paths
VPP_DIR="/home/yangc/Lab/VPP/video-prediction-policy"
MODELS_DIR="/path/to/models"
DATASET_DIR="/path/to/calvin/task_D_D"

# Activate conda environment
conda activate vpp

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CALVIN_ROOT="/home/yangc/Lab/VPP/calvin"

# Verify setup
cd /home/yangc/Lab/VPP/scripts
python test_d2d_setup.py \
    --video_model_path "$MODELS_DIR/svd-robot-calvin" \
    --action_model_folder "$MODELS_DIR/dp-calvin" \
    --clip_model_path "$MODELS_DIR/clip-vit-base-patch32" \
    --calvin_d2d_dir "$DATASET_DIR"

echo "Setup complete! Ready to run benchmark."
```

Save this as `setup_vpp_d2d.sh` and run:
```bash
chmod +x setup_vpp_d2d.sh
./setup_vpp_d2d.sh
```

## Summary of Required Components

| Component | Size | Source | Required |
|------------|------|---------|-----------|
| Conda environment (vpp) | - | Yes |
| VPP dependencies | ~2GB | Yes |
| Calvin environment | ~5GB | Optional but recommended |
| CLIP model | ~600MB | Yes |
| SVD video model | ~8GB | Yes |
| Action model | ~1GB | Yes |
| Calvin D -> D dataset | ~500GB | Yes |
| Accelerate | ~50MB | Yes |

**Total storage required**: ~516GB (without Calvin) or ~521GB (with Calvin)

## Troubleshooting

### Issue: Conda environment not found

```bash
# Check if conda is installed
conda --version

# If not installed, install conda or miniconda
# Follow instructions at: https://docs.conda.io/en/latest/miniconda.html
```

### Issue: CUDA out of memory during model loading

```bash
# Check available GPU memory
nvidia-smi

# Reduce batch size in config if needed
# Or use fewer GPUs
```

### Issue: Dataset not found

```bash
# Verify dataset structure
ls -la /path/to/calvin/task_D_D/

# Ensure domain directories exist
ls /path/to/calvin/task_D_D/domain_A/
ls /path/to/calvin/task_D_D/domain_B/
ls /path/to/calvin/task_D_D/domain_C/
ls /path/to/calvin/task_D_D/domain_D/
```

### Issue: Import errors

```bash
# Ensure you're in the correct conda environment
conda activate vpp

# Verify dependencies are installed
pip list | grep -E "torch|hydra|omegaconf|accelerate"

# Reinstall if needed
pip install -r /home/yangc/Lab/VPP/video-prediction-policy/requirements.txt
```

## Next Steps

After completing the installation:

1. Verify your setup with the test script
2. Review the configuration file: `config/calvin_d2d_config.yaml`
3. Run the benchmark using the example script or custom commands

See [`README.md`](README.md) for usage instructions.
