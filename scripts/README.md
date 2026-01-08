# Calvin D -> D Benchmark Scripts

This directory contains scripts for running the Video Prediction Policy (VPP) on the Calvin D -> D (Domain to Domain) benchmark.

## Overview

The Calvin D -> D benchmark evaluates a robot policy's ability to generalize across different visual domains, including:
- **Domain A**: Standard lighting and camera angle
- **Domain B**: Dim lighting conditions
- **Domain C**: Bright lighting conditions
- **Domain D**: Alternate camera angle

The benchmark tests both same-domain performance (baseline) and cross-domain transfer performance.

## Directory Structure

```
scripts/
├── README.md                          # This file
├── INSTALLATION_GUIDE.md              # Detailed installation instructions
├── run_calvin_d2d_benchmark.py        # Main benchmark script
├── test_d2d_setup.py                 # Setup verification script
├── example_run.sh                     # Example bash script
├── config/
│   └── calvin_d2d_config.yaml        # Configuration file for 4-card GPU setup
└── utils/
    ├── __init__.py                    # Utils package initialization
    └── calvin_d2d_utils.py           # D -> D specific utility functions
```

## Prerequisites

**IMPORTANT**: Before running the benchmark, you must set up the environment. **Please follow the detailed installation guide in [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md)**.

### Quick Summary

You need to:
1. **Install VPP environment** (conda + dependencies)
2. **Install Calvin environment** (optional but recommended)
3. **Download pre-trained models** from HuggingFace (~10GB total)
4. **Download Calvin D -> D dataset** (~500GB)

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

### Quick Installation Steps

For detailed step-by-step instructions, see [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md).

```bash
# 1. Create conda environment
cd /home/yangc/Lab/VPP/video-prediction-policy
conda create -n vpp python==3.10
conda activate vpp

# 2. Install Calvin (optional)
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT
sh install.sh

# 3. Install VPP dependencies
cd /home/yangc/Lab/VPP/video-prediction-policy
pip install -r requirements.txt

# 4. Install accelerate
pip install accelerate

# 5. Download models (see INSTALLATION_GUIDE.md for details)
pip install huggingface-hub
huggingface-cli download openai/clip-vit-base-patch32 --local-dir /path/to/models/clip-vit-base-patch32
huggingface-cli download yjguo/svd-robot-calvin-ft --local-dir /path/to/models/svd-robot-calvin
huggingface-cli download yjguo/dp-calvin --local-dir /path/to/models/dp-calvin

# 6. Download Calvin dataset (see INSTALLATION_GUIDE.md for details)
# Follow instructions in https://github.com/mees/calvin

# 7. Verify installation
cd /home/yangc/Lab/VPP/scripts
python test_d2d_setup.py
```

### Verification

After installation, verify your setup:

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

For complete installation instructions, troubleshooting, and setup automation, see [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md).

## GPU Compatibility

The benchmark is designed to run on 4 GPUs. The configuration includes GPU-specific batch size settings:

| GPU Type | VRAM | Recommended Batch Size |
|-----------|------|----------------------|
| RTX 5090 | 24GB | 16-20 |
| RTX 4090 | 24GB | 20-24 |
| A100 40GB | 40GB | 32-40 |
| A100 80GB | 80GB | 64-80 |

**Default configuration**: `batch_size: 38` (optimized for A100 40GB)

**For RTX 5090 (24GB VRAM)**: Edit [`config/calvin_d2d_config.yaml`](config/calvin_d2d_config.yaml) and set:
```yaml
batch_size: 18  # Or 16-20 depending on available memory
```

**For A100 GPUs**: Default configuration should work well.

## Quick Start

### 1. Basic Usage (4 GPUs)

Run benchmark on 4 GPU cards:

```bash
accelerate launch --num_processes=4 run_calvin_d2d_benchmark.py \
    --config config/calvin_d2d_config.yaml \
    --video_model_path /path/to/svd-robot-calvin \
    --action_model_folder /path/to/models/dp-calvin \
    --clip_model_path /path/to/models/clip-vit-base-patch32 \
    --calvin_d2d_dir /path/to/calvin/task_D_D
```

### 2. Custom Configuration

Modify `config/calvin_d2d_config.yaml` to customize:
- Domain pairs to evaluate
- Number of sequences
- Video recording settings
- Logging options

### 3. Single GPU Usage

For testing on a single GPU:

```bash
python run_calvin_d2d_benchmark.py \
    --config config/calvin_d2d_config.yaml \
    --video_model_path /path/to/svd-robot-calvin \
    --action_model_folder /path/to/models/dp-calvin \
    --clip_model_path /path/to/models/clip-vit-base-patch32 \
    --calvin_d2d_dir /path/to/calvin/task_D_D \
    --num_sequences 100 \
    --num_videos 5
```

## Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--config` | str | Yes | - | Path to configuration file |
| `--video_model_path` | str | Yes | - | Path to video model (svd-robot-calvin) |
| `--action_model_folder` | str | Yes | - | Path to action model folder (dp-calvin) |
| `--clip_model_path` | str | Yes | - | Path to CLIP model |
| `--calvin_d2d_dir` | str | Yes | - | Path to Calvin D -> D dataset |
| `--log_dir` | str | No | None | Directory to save logs |
| `--num_videos` | int | No | 10 | Number of videos to record |
| `--num_sequences` | int | No | 1000 | Number of evaluation sequences |
| `--debug` | flag | No | False | Run in debug mode |
| `--no_wandb` | flag | No | False | Disable wandb logging |

## Configuration File

The `config/calvin_d2d_config.yaml` file contains:

### Model Configuration
- Video prediction model parameters
- Action model parameters
- Sampling parameters (num_sampling_steps, sampler_type, etc.)

### Domain Configuration
- List of domain pairs to evaluate
- Domain-specific settings (lighting, camera angle, etc.)

### Evaluation Configuration
- Episode length
- Number of sequences
- Video recording settings

### Distributed Training Configuration
- Number of processes (4 for 4 GPUs)
- GPU IDs
- Mixed precision settings

## Output

The benchmark generates:

1. **Log Directory**: `./logs/calvin_d2d/YYYY-MM-DD_HH-MM-SS/`
   - `d2d_results.json`: Detailed results in JSON format
   - `d2d_summary.txt`: Human-readable summary
   - `wandb/`: Wandb logs (if enabled)

2. **Videos**: `./logs/calvin_d2d/YYYY-MM-DD_HH-MM-SS/videos/`
   - Videos for each domain pair (if `num_videos > 0`)

3. **Console Output**: Real-time progress and results

## Results Interpretation

### Metrics

1. **Average Sequence Length**: Average number of successfully completed tasks (0-5)
   - Higher is better (max 5.0)

2. **Chain Success Rates**: Success rate for completing i tasks in a row
   - `1 task`: Success rate for completing at least 1 task
   - `2 tasks`: Success rate for completing at least 2 tasks in a row
   - ...
   - `5 tasks`: Success rate for completing all 5 tasks in a row

3. **Transfer Metrics**:
   - **Same-domain performance**: Baseline performance on each domain
   - **Cross-domain performance**: Performance when transferring between domains
   - **Transfer gap**: Performance drop when transferring (baseline - cross-domain)
   - **Domain similarity**: Similarity score based on transfer performance (0-1)

### Example Output

```
A -> A:
  Average sequence length: 4.50
  Success rates:
    1 tasks in a row: 95.0%
    2 tasks in a row: 90.0%
    3 tasks in a row: 85.0%
    4 tasks in a row: 80.0%
    5 tasks in a row: 75.0%

A -> B:
  Average sequence length: 3.80
  Success rates:
    1 tasks in a row: 90.0%
    2 tasks in a row: 80.0%
    3 tasks in a row: 70.0%
    4 tasks in a row: 60.0%
    5 tasks in a row: 50.0%

TRANSFER METRICS
Same-domain performance (baseline):
  Domain A: 4.50
  Domain B: 4.30

Transfer gap (performance drop):
  A -> B: 0.70
  B -> A: 0.50
```

## Advanced Usage

### Custom Domain Pairs

Edit `config/calvin_d2d_config.yaml` to specify custom domain pairs:

```yaml
domains:
  - ["A", "B"]  # Only evaluate A -> B
  - ["B", "A"]  # And B -> A
```

### Wandb Integration

Set your wandb entity in config:

```yaml
wandb:
  project: "vpp-calvin-d2d"
  entity: "your-wandb-username"
  mode: "online"
```

### Debug Mode

Run in debug mode for detailed output and visualization:

```bash
python run_calvin_d2d_benchmark.py \
    --config config/calvin_d2d_config.yaml \
    --video_model_path /path/to/svd-robot-calvin \
    --action_model_folder /path/to/models/dp-calvin \
    --clip_model_path /path/to/models/clip-vit-base-patch32 \
    --calvin_d2d_dir /path/to/calvin/task_D_D \
    --debug
```

### Performance Optimization

For faster evaluation, adjust these parameters in config:

```yaml
num_sequences: 100  # Reduce number of sequences
num_videos: 0  # Disable video recording
ep_len: 180  # Reduce episode length
```

## GPU Compatibility

The benchmark is designed to run on 4 GPUs. The configuration includes GPU-specific batch size settings:

| GPU Type | VRAM | Recommended Batch Size |
|-----------|------|----------------------|
| RTX 5090 | 24GB | 16-20 |
| RTX 4090 | 24GB | 20-24 |
| A100 40GB | 40GB | 32-40 |
| A100 80GB | 80GB | 64-80 |

**Default configuration**: `batch_size: 38` (optimized for A100 40GB)

**For RTX 5090 (24GB VRAM)**: Edit [`config/calvin_d2d_config.yaml`](config/calvin_d2d_config.yaml) and set:
```yaml
batch_size: 18  # Or 16-20 depending on available memory
```

**For A100 GPUs**: Default configuration should work well.

For detailed GPU compatibility information and troubleshooting, see [`GPU_COMPATIBILITY.md`](GPU_COMPATIBILITY.md).

## Troubleshooting

### Out of Memory

If you encounter CUDA out of memory errors:
1. Reduce `batch_size` in config
2. Reduce `num_sampling_steps`
3. Use gradient checkpointing: `gradient_checkpointing: true`
4. Use fewer GPUs

### Slow Evaluation

To speed up evaluation:
1. Reduce `num_sequences`
2. Disable video recording: `num_videos: 0`
3. Reduce `ep_len`
4. Use more GPUs if available

### Dataset Not Found

Ensure that Calvin D -> D dataset is properly structured:
```
/path/to/calvin/task_D_D/
├── domain_A/
│   ├── episodes.json
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

### Installation Issues

For installation problems, see [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md) for:
- Detailed step-by-step instructions
- Troubleshooting common issues
- Setup automation scripts

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

## License

This code follows the same license as the VPP repository.

## Contact

For questions or issues, please open an issue on the VPP GitHub repository.
