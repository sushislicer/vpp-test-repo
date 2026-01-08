# Calvin D -> D Benchmark - Complete Setup Summary

This document provides a complete overview of the Calvin D -> D benchmark implementation and how to get started.

## What Was Created

### 1. Main Benchmark Script
**File**: [`run_calvin_d2d_benchmark.py`](run_calvin_d2d_benchmark.py)
- Distributed evaluation across 4 GPUs using accelerate
- Support for 16 domain transfer pairs (A, B, C, D domains)
- Automatic result aggregation across processes
- Wandb logging integration
- Video recording for visualization
- Comprehensive transfer metrics (transfer gap, domain similarity)

### 2. Configuration File
**File**: [`config/calvin_d2d_config.yaml`](config/calvin_d2d_config.yaml)
- Complete 4-card GPU setup configuration
- Domain definitions (A: standard, B: dim lighting, C: bright lighting, D: alternate camera)
- All 16 domain pairs (4 same-domain + 12 cross-domain)
- Performance optimization settings
- Video recording and logging options

### 3. Utility Functions
**File**: [`utils/calvin_d2d_utils.py`](utils/calvin_d2d_utils.py)
- Domain management and dataset loading
- D -> D sequence evaluation with domain transfer
- Transfer metrics calculation
- Results formatting and saving (JSON + text)
- Result comparison utilities

### 4. Test Script
**File**: [`test_d2d_setup.py`](test_d2d_setup.py)
- Verifies imports and dependencies
- Validates configuration file
- Tests utility functions
- Optional model/dataset path verification

### 5. Installation Guide
**File**: [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md)
- Detailed step-by-step installation instructions
- Based on VPP README requirements
- Covers conda setup, Calvin installation, model downloads, dataset setup
- Troubleshooting guide
- Setup automation

### 6. Automated Setup Script
**File**: [`setup_vpp_d2d.sh`](setup_vpp_d2d.sh)
- Automates entire installation process
- Interactive prompts for optional components
- Color-coded output for clarity
- Verification tests

### 7. Documentation
**Files**: 
- [`README.md`](README.md) - Complete usage guide
- [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) - Technical implementation details
- [`example_run.sh`](example_run.sh) - Example bash script for running benchmark

## Quick Start Guide

### Step 1: Install and Setup

**Option A: Automated Setup (Recommended)**

```bash
cd /home/yangc/Lab/VPP/scripts
./setup_vpp_d2d.sh
```

This script will:
1. Create conda environment
2. Install Calvin (optional)
3. Install VPP dependencies
4. Install accelerate
5. Download pre-trained models
6. Set environment variables
7. Verify installation

**Option B: Manual Setup**

Follow the detailed instructions in [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md).

### Step 2: Download Calvin Dataset

After installation, download the Calvin D -> D dataset (~500GB):

1. Follow instructions at: https://github.com/mees/calvin
2. Download to: `/path/to/calvin/task_D_D/`
3. Verify structure matches expected format

Expected structure:
```
/path/to/calvin/task_D_D/
├── domain_A/
│   ├── episodes.json
│   └── episode_000/
│       ├── rgb.mp4
│       └── state.npy
├── domain_B/
├── domain_C/
└── domain_D/
```

### Step 3: Verify Setup

```bash
cd /home/yangc/Lab/VPP/scripts

# Basic test
python test_d2d_setup.py

# Full test with paths
python test_d2d_setup.py \
    --video_model_path /path/to/models/svd-robot-calvin \
    --action_model_folder /path/to/models/dp-calvin \
    --clip_model_path /path/to/models/clip-vit-base-patch32 \
    --calvin_d2d_dir /path/to/calvin/task_D_D
```

### Step 4: Run Benchmark

**On 4 GPUs:**

```bash
accelerate launch --num_processes=4 run_calvin_d2d_benchmark.py \
    --config config/calvin_d2d_config.yaml \
    --video_model_path /path/to/models/svd-robot-calvin \
    --action_model_folder /path/to/models/dp-calvin \
    --clip_model_path /path/to/models/clip-vit-base-patch32 \
    --calvin_d2d_dir /path/to/calvin/task_D_D
```

**On single GPU (for testing):**

```bash
python run_calvin_d2d_benchmark.py \
    --config config/calvin_d2d_config.yaml \
    --video_model_path /path/to/models/svd-robot-calvin \
    --action_model_folder /path/to/models/dp-calvin \
    --clip_model_path /path/to/models/clip-vit-base-patch32 \
    --calvin_d2d_dir /path/to/calvin/task_D_D \
    --num_sequences 100 \
    --num_videos 5
```

## File Structure

```
scripts/
├── README.md                          # Complete usage guide
├── INSTALLATION_GUIDE.md              # Detailed installation instructions
├── SETUP_SUMMARY.md                  # This file
├── IMPLEMENTATION_SUMMARY.md           # Technical implementation details
├── run_calvin_d2d_benchmark.py        # Main benchmark script
├── test_d2d_setup.py                 # Setup verification script
├── setup_vpp_d2d.sh                  # Automated setup script (executable)
├── example_run.sh                     # Example run script (executable)
├── config/
│   └── calvin_d2d_config.yaml        # 4-card GPU configuration
└── utils/
    ├── __init__.py                    # Utils package
    └── calvin_d2d_utils.py           # D -> D utility functions
```

## Key Features

### 1. Multi-GPU Support
- Uses accelerate for distributed training
- Automatic sequence distribution across 4 GPUs
- Efficient result aggregation

### 2. Comprehensive Evaluation
- Evaluates 16 domain pairs (4 same-domain + 12 cross-domain)
- Calculates transfer learning metrics
- Provides detailed per-domain and overall statistics

### 3. Flexible Configuration
- Easy to customize domain pairs
- Adjustable evaluation parameters
- Support for different sampling strategies

### 4. Robust Logging
- Wandb integration for experiment tracking
- JSON and text output formats
- Video recording for visualization

### 5. Automated Setup
- One-command installation script
- Interactive prompts for optional components
- Automatic verification

## Installation Requirements

Based on VPP README, you need to install:

### Required Components

| Component | Size | Source | Required |
|-----------|------|---------|-----------|
| Conda environment (vpp) | - | Yes |
| VPP dependencies | ~2GB | Yes |
| CLIP model | ~600MB | Yes |
| SVD video model | ~8GB | Yes |
| Action model | ~1GB | Yes |
| Calvin D -> D dataset | ~500GB | Yes |
| Accelerate | ~50MB | Yes |

**Total storage required**: ~512GB

### Installation Steps

1. **Create conda environment** with Python 3.10
2. **Install Calvin** (optional but recommended)
3. **Install VPP dependencies** from requirements.txt
4. **Install accelerate** for multi-GPU support
5. **Download pre-trained models** from HuggingFace
6. **Download Calvin dataset** (~500GB)

See [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md) for detailed instructions.

## Output and Results

### Generated Files

1. **Log Directory**: `./logs/calvin_d2d/YYYY-MM-DD_HH-MM-SS/`
   - `d2d_results.json`: Detailed results in JSON format
   - `d2d_summary.txt`: Human-readable summary
   - `wandb/`: Wandb logs (if enabled)

2. **Videos**: `./logs/calvin_d2d/YYYY-MM-DD_HH-MM-SS/videos/`
   - Videos for each domain pair (if `num_videos > 0`)

3. **Console Output**: Real-time progress and results

### Metrics Calculated

1. **Average Sequence Length**: Average number of successfully completed tasks (0-5)
2. **Chain Success Rates**: Success rate for i tasks in a row (i=1..5)
3. **Transfer Metrics**:
   - Same-domain performance (baseline)
   - Cross-domain performance
   - Transfer gap (performance drop)
   - Domain similarity (0-1)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use fewer GPUs

2. **Slow Evaluation**
   - Reduce num_sequences
   - Disable video recording
   - Reduce ep_len

3. **Dataset Not Found**
   - Check dataset structure
   - Verify path is correct
   - Ensure domain directories exist

4. **Import Errors**
   - Install dependencies
   - Check PYTHONPATH
   - Verify video-prediction-policy path

For detailed troubleshooting, see [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md).

## Documentation

- **Usage Guide**: [`README.md`](README.md)
- **Installation Guide**: [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md)
- **Implementation Details**: [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md)

## Next Steps

1. Run automated setup: `./setup_vpp_d2d.sh`
2. Download Calvin dataset (~500GB)
3. Verify setup: `python test_d2d_setup.py`
4. Run benchmark: `accelerate launch --num_processes=4 run_calvin_d2d_benchmark.py ...`
5. Analyze results in `./logs/calvin_d2d/`

## Support

For questions or issues:
- See [`README.md`](README.md) for usage instructions
- See [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md) for installation help
- Refer to VPP GitHub repository for VPP-specific issues
- Refer to Calvin GitHub repository for Calvin-specific issues

## Citation

If you use this benchmark code, please cite the VPP paper:

```bibtex
@article{hu2024video,
  title={Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations},
  author={Hu, Yucheng and Guo, Yanjiang and Wang, Pengchao and Chen, Xiaoyu and Wang, Yen-Jen and Zhang, Jianke and Sreenath, Koushil and Lu, Chaochao and Chen, Jianyu},
  journal={arXiv preprint arXiv:2412.14803},
  year={2024}
}
```
