# GPU Compatibility Guide for Calvin D -> D Benchmark

This guide explains GPU compatibility and configuration for running VPP on Calvin D -> D benchmark.

## Overview

The Calvin D -> D benchmark is designed to run on 4 GPUs. The configuration includes GPU-specific batch size settings to optimize performance based on available VRAM.

## GPU Specifications

| GPU Type | VRAM | Compute Capability | Recommended Batch Size |
|-----------|------|-------------------|----------------------|
| RTX 5090 | 24GB | 8.6 | 16-20 |
| RTX 4090 | 24GB | 8.6 | 20-24 |
| A100 40GB | 40GB | 8.0 | 32-40 |
| A100 80GB | 80GB | 8.0 | 64-80 |

## Configuration for RTX 5090 (24GB VRAM)

### Step 1: Update Configuration File

Edit [`config/calvin_d2d_config.yaml`](config/calvin_d2d_config.yaml) and set:

```yaml
# For RTX 5090 (24GB VRAM)
batch_size: 18  # Or 16-20 depending on available memory
devices: 4
gpu_ids: "0,1,2,3"
```

### Step 2: Run Benchmark

```bash
accelerate launch --num_processes=4 run_calvin_d2d_benchmark.py \
    --config config/calvin_d2d_config.yaml \
    --video_model_path /path/to/models/svd-robot-calvin \
    --action_model_folder /path/to/models/dp-calvin \
    --clip_model_path /path/to/models/clip-vit-base-patch32 \
    --calvin_d2d_dir /path/to/calvin/task_D_D
```

### Step 3: Monitor GPU Memory

During evaluation, monitor GPU memory usage:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

If you encounter CUDA out of memory errors:
1. Reduce `batch_size` to 16 or lower
2. Enable gradient checkpointing: `gradient_checkpointing: true`
3. Reduce `num_sampling_steps` in config

## Configuration for A100 GPUs

### Step 1: Update Configuration File

Edit [`config/calvin_d2d_config.yaml`](config/calvin_d2d_config.yaml) and set:

```yaml
# For A100 40GB
batch_size: 32
devices: 4
gpu_ids: "0,1,2,3"

# For A100 80GB
# batch_size: 64
# devices: 4
# gpu_ids: "0,1,2,3"
```

**Note**: Default configuration (`batch_size: 38`) is optimized for A100 40GB and should work well.

### Step 2: Run Benchmark

```bash
accelerate launch --num_processes=4 run_calvin_d2d_benchmark.py \
    --config config/calvin_d2d_config.yaml \
    --video_model_path /path/to/models/svd-robot-calvin \
    --action_model_folder /path/to/models/dp-calvin \
    --clip_model_path /path/to/models/clip-vit-base-patch32 \
    --calvin_d2d_dir /path/to/calvin/task_D_D
```

## Performance Comparison

### Expected Performance

| GPU Type | Expected Speed | Memory Efficiency |
|-----------|---------------|------------------|
| RTX 5090 | Baseline | Good |
| RTX 4090 | Slightly faster | Good |
| A100 40GB | 1.5-2x faster | Excellent |
| A100 80GB | 2-3x faster | Excellent |

### Memory Usage

The benchmark uses approximately:
- **Model loading**: ~8GB (SVD video model)
- **Action model**: ~1GB
- **CLIP model**: ~600MB
- **Per-GPU memory**: Depends on batch size

For RTX 5090 with `batch_size: 18`:
- Total model memory: ~10GB
- Per-GPU memory: ~2-3GB (including activations)
- Available for data: ~21GB

For A100 40GB with `batch_size: 32`:
- Total model memory: ~10GB
- Per-GPU memory: ~4-6GB (including activations)
- Available for data: ~34GB

## Troubleshooting

### Issue: CUDA Out of Memory on RTX 5090

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**Solutions**:
1. Reduce batch size:
   ```yaml
   batch_size: 16  # Try 16, then 14 if still OOM
   ```

2. Enable gradient checkpointing:
   ```yaml
   performance:
     gradient_checkpointing: true
   ```

3. Reduce sampling steps:
   ```yaml
   num_sampling_steps: 8  # Reduce from 10
   ```

4. Use mixed precision:
   ```yaml
   performance:
     use_amp: true
   ```

### Issue: Slow Performance on RTX 5090

**Symptoms**: Evaluation takes longer than expected.

**Solutions**:
1. Reduce number of sequences for testing:
   ```bash
   --num_sequences 100  # Instead of 1000
   ```

2. Disable video recording:
   ```bash
   --num_videos 0  # Instead of 10
   ```

3. Reduce episode length:
   ```yaml
   ep_len: 180  # Instead of 360
   ```

### Issue: GPU Not Detected

**Symptoms**:
```
RuntimeError: CUDA device not found
```

**Solutions**:
1. Check GPU availability:
   ```bash
   nvidia-smi
   ```

2. Verify CUDA installation:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. Check GPU IDs:
   ```bash
   # List available GPUs
   nvidia-smi --list-gpus
   
   # Update config with correct GPU IDs
   # For 4 GPUs: gpu_ids: "0,1,2,3"
   # For 2 GPUs: gpu_ids: "0,1"
   ```

## Advanced Configuration

### Multi-Node Setup

If you have multiple nodes, update [`config/calvin_d2d_config.yaml`](config/calvin_d2d_config.yaml):

```yaml
# For 2 nodes with 4 GPUs each
num_machines: 2
num_processes: 8  # 4 GPUs per node
```

### Custom GPU Selection

If you want to use specific GPUs:

```yaml
# Use GPUs 0, 2, 4, 6 instead of 0, 1, 2, 3
gpu_ids: "0,2,4,6"
```

### Performance Tuning

For optimal performance, experiment with:

1. **Batch size**: Start with recommended, adjust based on memory usage
2. **Number of workers**: `num_workers: 12` (default), adjust based on CPU cores
3. **Pin memory**: `pin_memory: true` (default) for faster data loading
4. **Non-blocking transfer**: `non_blocking: true` (default) for async data transfer

## Verification

After configuration, verify GPU setup:

```bash
# Check GPU availability
nvidia-smi

# Expected output for 4 GPUs:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 560.28.01    Driver Version: 560.28.01    CUDA Version: 12.2 |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================+===================|
# |   0  NVIDIA RTX ...  Off  | 00000000:00:00.0 Off |                  N/A      0%      0%      Default |
# |   1  NVIDIA RTX ...  Off  | 00000000:00:00.0 Off |                  N/A      0%      0%      Default |
# |   2  NVIDIA RTX ...  Off  | 00000000:00:00.0 Off |                  N/A      0%      0%      Default |
# |   3  NVIDIA RTX ...  Off  | 00000000:00:00.0 Off |                  N/A      0%      0%      Default |
# +-----------------------------------------------------------------------------+
```

## Next Steps

1. **Choose your GPU type** and follow corresponding configuration above
2. **Update config file** with appropriate batch size
3. **Run benchmark** and monitor performance
4. **Adjust settings** if needed based on memory/performance

For complete setup instructions, see [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md).

For usage instructions, see [`README.md`](README.md).
