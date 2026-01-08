# Calvin D -> D Benchmark Implementation Summary

## Overview

This document summarizes the implementation of the Calvin D -> D (Domain to Domain) benchmark scripts for running Video Prediction Policy (VPP) on 4 GPU cards.

## Created Files

### 1. Main Benchmark Script
**File**: [`run_calvin_d2d_benchmark.py`](run_calvin_d2d_benchmark.py)

**Purpose**: Main entry point for running VPP on Calvin D -> D benchmark

**Key Features**:
- Distributed evaluation across 4 GPUs using accelerate
- Support for multiple domain transfer pairs (A->A, A->B, B->A, etc.)
- Automatic result aggregation across processes
- Wandb logging integration
- Video recording for visualization
- Comprehensive metrics calculation (transfer gap, domain similarity)

**Usage**:
```bash
accelerate launch --num_processes=4 run_calvin_d2d_benchmark.py \
    --config config/calvin_d2d_config.yaml \
    --video_model_path /path/to/svd-robot-calvin \
    --action_model_folder /path/to/dp-calvin \
    --clip_model_path /path/to/clip-vit-base-patch32 \
    --calvin_d2d_dir /path/to/calvin/task_D_D
```

### 2. Configuration File
**File**: [`config/calvin_d2d_config.yaml`](config/calvin_d2d_config.yaml)

**Purpose**: Configuration for 4-card GPU setup and D -> D evaluation parameters

**Key Sections**:
- **Model Configuration**: Video prediction and action model parameters
- **Domain Configuration**: List of domain pairs to evaluate (A, B, C, D)
- **Evaluation Configuration**: Episode length, number of sequences, video recording
- **Distributed Training**: 4 GPU setup with accelerate
- **D -> D Settings**: Domain-specific configurations and transfer metrics
- **Performance Optimization**: Memory and speed optimizations

**Default Domain Pairs**:
- Same-domain: A->A, B->B, C->C, D->D (baseline)
- Cross-domain: All 12 cross-domain pairs (A->B, A->C, A->D, B->A, etc.)

### 3. Utility Functions
**File**: [`utils/calvin_d2d_utils.py`](utils/calvin_d2d_utils.py)

**Purpose**: D -> D specific utility functions

**Key Functions**:
- `get_d2d_domains()`: Parse domain configuration
- `get_domain_info()`: Get domain information
- `load_d2d_dataset()`: Load dataset for specific domain
- `evaluate_d2d_sequence()`: Evaluate sequence with domain transfer
- `rollout_d2d()`: Rollout single subtask with domain transfer
- `count_d2d_success()`: Calculate success rates
- `calculate_transfer_metrics()`: Calculate transfer learning metrics
- `print_d2d_results()`: Print formatted results
- `save_d2d_results()`: Save results to JSON and text files
- `load_d2d_results()`: Load saved results
- `compare_d2d_results()`: Compare two result sets

**Domain Definitions**:
- **Domain A**: Standard lighting and camera angle
- **Domain B**: Dim lighting conditions
- **Domain C**: Bright lighting conditions
- **Domain D**: Alternate camera angle

### 4. Test Script
**File**: [`test_d2d_setup.py`](test_d2d_setup.py)

**Purpose**: Verify setup before running benchmark

**Tests**:
- Import dependencies
- Configuration file validity
- D -> D utility functions
- Model paths (optional)
- Dataset structure (optional)

**Usage**:
```bash
# Basic test
python test_d2d_setup.py

# Full test with paths
python test_d2d_setup.py \
    --video_model_path /path/to/svd-robot-calvin \
    --action_model_folder /path/to/dp-calvin \
    --clip_model_path /path/to/clip-vit-base-patch32 \
    --calvin_d2d_dir /path/to/calvin/task_D_D
```

### 5. Example Run Script
**File**: [`example_run.sh`](example_run.sh)

**Purpose**: Example bash script for running benchmark

**Features**:
- Configurable paths
- Multiple usage examples (4 GPUs, single GPU, debug mode)
- Comments explaining each option

**Usage**:
```bash
# Edit paths in the script, then run:
./example_run.sh
```

### 6. Documentation
**File**: [`README.md`](README.md)

**Purpose**: Comprehensive documentation for using the benchmark scripts

**Contents**:
- Overview and directory structure
- Prerequisites and quick start guide
- Command line arguments reference
- Configuration file explanation
- Output format and results interpretation
- Advanced usage examples
- Troubleshooting guide

## Architecture

### Distributed Evaluation Flow

```
Main Process (Rank 0)
    ├── Load models and environment
    ├── Distribute sequences across 4 GPUs
    ├── Evaluate domain pairs in parallel
    │   ├── GPU 0: Sequences 0, 4, 8, ...
    │   ├── GPU 1: Sequences 1, 5, 9, ...
    │   ├── GPU 2: Sequences 2, 6, 10, ...
    │   └── GPU 3: Sequences 3, 7, 11, ...
    ├── Gather results from all processes
    ├── Calculate metrics
    └── Save results and log to wandb
```

### Domain Transfer Evaluation

For each domain pair (source -> target):
1. Load source domain initial state
2. Execute task sequence in target domain
3. Record success/failure for each task
4. Calculate success rates and transfer metrics

### Metrics Calculated

1. **Average Sequence Length**: Mean number of successful tasks (0-5)
2. **Chain Success Rates**: Success rate for i tasks in a row (i=1..5)
3. **Transfer Gap**: Performance drop when transferring domains
4. **Domain Similarity**: Similarity score based on transfer performance

## Key Features

### 1. Multi-GPU Support
- Uses accelerate for distributed training
- Automatic sequence distribution across GPUs
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

### 5. Error Handling
- Graceful handling of missing domains
- Validation of configuration and paths
- Clear error messages

## Usage Workflow

### Step 1: Setup
```bash
# Install dependencies
pip install -r ../video-prediction-policy/requirements.txt
pip install accelerate

# Download models and dataset
# (See README.md for details)
```

### Step 2: Test Setup
```bash
python test_d2d_setup.py
```

### Step 3: Run Benchmark
```bash
# Edit paths in example_run.sh or run directly:
accelerate launch --num_processes=4 run_calvin_d2d_benchmark.py \
    --config config/calvin_d2d_config.yaml \
    --video_model_path /path/to/svd-robot-calvin \
    --action_model_folder /path/to/dp-calvin \
    --clip_model_path /path/to/clip-vit-base-patch32 \
    --calvin_d2d_dir /path/to/calvin/task_D_D
```

### Step 4: Analyze Results
```bash
# Results saved to: ./logs/calvin_d2d/YYYY-MM-DD_HH-MM-SS/
# - d2d_results.json: Detailed results
# - d2d_summary.txt: Human-readable summary
# - wandb/: Wandb logs (if enabled)
```

## Customization

### Add New Domains
1. Add domain definition in [`utils/calvin_d2d_utils.py`](utils/calvin_d2d_utils.py)
2. Add domain pairs in [`config/calvin_d2d_config.yaml`](config/calvin_d2d_config.yaml)

### Modify Evaluation Parameters
Edit [`config/calvin_d2d_config.yaml`](config/calvin_d2d_config.yaml):
- `num_sequences`: Number of evaluation sequences
- `ep_len`: Episode length
- `num_videos`: Number of videos to record
- `num_sampling_steps`: Video prediction sampling steps

### Change GPU Configuration
Edit [`config/calvin_d2d_config.yaml`](config/calvin_d2d_config.yaml):
- `devices`: Number of GPUs
- `gpu_ids`: GPU IDs to use
- `num_processes`: Number of processes (should match devices)

## Performance Considerations

### Memory Optimization
- Reduce `batch_size` if OOM
- Enable `gradient_checkpointing`
- Use fewer GPUs with larger batch size

### Speed Optimization
- Reduce `num_sequences` for testing
- Disable video recording (`num_videos: 0`)
- Reduce `ep_len` for faster episodes
- Use more GPUs if available

### Result Quality
- Use more sequences for reliable statistics
- Enable video recording for debugging
- Use same-domain results as baseline

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

## Future Enhancements

Potential improvements:
1. Support for more domains
2. Adaptive domain selection
3. Real-time visualization
4. Comparison with other methods
5. Automated hyperparameter tuning
6. Domain adaptation strategies

## References

- VPP Paper: https://arxiv.org/abs/2412.14803
- VPP GitHub: https://github.com/mees/calvin
- Stable Video Diffusion: https://github.com/Stability-AI/generative-models
- MDT Policy: https://github.com/intuitive-robots/mdt_policy

## Contact

For questions or issues, please refer to the main VPP repository or open an issue.
