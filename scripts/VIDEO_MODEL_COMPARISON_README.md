# Video Model Comparison Experiment

This script trains two versions of the video model (Video A, Video B) with different random seeds, then trains Actor A with Video A and Actor B with Video B. Finally, it compares evaluation metrics between the two combinations.

## Purpose

To investigate whether different random seeds during video model training lead to significantly different evaluation performance when paired with different actor models.

## Experimental Design

### Combinations

1. **Video A + Actor A**: Video model trained with seed 42, Actor trained with seed 456
2. **Video B + Actor B**: Video model trained with seed 123, Actor trained with seed 789

### Hypothesis

If the video model's random seed significantly affects the learned representations, then:
- [Video A + Actor A] and [Video B + Actor B] should show similar performance
- If they show significantly different performance (>5% difference), it suggests the video model's random seed has a significant impact

## Usage

### Step 1: Train Video Models

```bash
cd /home/yangc/Lab/VPP/scripts

# Train both video models with different random seeds
python train_video_models_comparison.py \
    --config config/train_video_comparison.yaml \
    --video_a_seed 42 \
    --video_b_seed 123 \
    --calvin_d2d_dir /path/to/calvin/task_D_D
```

This will:
1. Train Video A with seed 42
2. Train Video B with seed 123
3. Save both models to `./video_model_comparison_output/video_a/` and `./video_model_comparison_output/video_b/`

### Step 2: Train Actor Models

```bash
# Train Actor A with Video A
python train_video_models_comparison.py \
    --config config/train_video_comparison.yaml \
    --actor_a_seed 456 \
    --video_a_seed 42 \
    --calvin_d2d_dir /path/to/calvin/task_D_D

# Train Actor B with Video B
python train_video_models_comparison.py \
    --config config/train_video_comparison.yaml \
    --actor_b_seed 789 \
    --video_b_seed 123 \
    --calvin_d2d_dir /path/to/calvin/task_D_D
```

This will:
1. Train Actor A with Video A (seed 456)
2. Train Actor B with Video B (seed 789)
3. Save both actors to `./video_model_comparison_output/actor_a/` and `./video_model_comparison_output/actor_b/`

### Step 3: Evaluate Combinations

```bash
# Evaluate [Video A + Actor A]
python train_video_models_comparison.py \
    --config config/train_video_comparison.yaml \
    --video_a_seed 42 \
    --actor_a_seed 456 \
    --calvin_d2d_dir /path/to/calvin/task_D_D

# Evaluate [Video B + Actor B]
python train_video_models_comparison.py \
    --config config/train_video_comparison.yaml \
    --video_b_seed 123 \
    --actor_b_seed 789 \
    --calvin_d2d_dir /path/to/calvin/task_D_D
```

This will:
1. Evaluate both combinations on Calvin D -> D benchmark
2. Save results to `./video_model_comparison_output/evaluation/`
3. Generate comparison report

### Step 4: Compare Results

The script automatically compares results and generates a comparison report showing:
- Average sequence length for each combination
- Chain success rates for each combination
- Difference between combinations
- Whether the difference is statistically significant (>5%)

## Output Structure

```
video_model_comparison_output/
├── video_a/
│   ├── config.yaml
│   ├── checkpoint-00010.pt
│   ├── checkpoint-00020.pt
│   ├── ...
│   └── final_model.pt
├── video_b/
│   ├── config.yaml
│   ├── checkpoint-00010.pt
│   ├── checkpoint-00020.pt
│   ├── ...
│   └── final_model.pt
├── actor_a/
│   ├── config.yaml
│   ├── checkpoint-00010.pt
│   ├── checkpoint-00020.pt
│   ├── ...
│   └── final_model.pt
├── actor_b/
│   ├── config.yaml
│   ├── checkpoint-00010.pt
│   ├── checkpoint-00020.pt
│   ├── ...
│   └── final_model.pt
└── evaluation/
    ├── VideoA_ActorA/
    │   ├── results.json
    │   └── d2d_summary.txt
    ├── VideoB_ActorB/
    │   ├── results.json
    │   └── d2d_summary.txt
    ├── comparison.json
    └── summary.json
```

## Configuration

Edit [`config/train_video_comparison.yaml`](config/train_video_comparison.yaml) to customize:

### Video Model Training

```yaml
model:
  # VideoFormer architecture
  Former_depth: 6
  Former_heads: 8
  Former_dim_head: 64
  Former_num_time_embeds: 16
  num_latents: 224
  
  # Training settings
  max_epochs: 100  # Adjust based on convergence
  save_every: 10
  batch_size: 4  # Adjust based on GPU memory
```

### Actor Model Training

```yaml
actor_model:
  # Actor architecture (simplified for comparison)
  input_dim: 384  # latent_dim from video model
  hidden_dim: 512
  output_dim: 7  # action_dim
  
  # Training settings
  max_epochs: 100  # Adjust based on convergence
  save_every: 10
  batch_size: 32  # Adjust based on GPU memory
```

### Evaluation Settings

```yaml
# Evaluation parameters
ep_len: 360
num_sequences: 1000  # Adjust based on time constraints
num_videos: 10  # Number of videos to record
```

## Expected Results

### Scenario 1: No Significant Difference

If both combinations show similar performance (<5% difference):
- **Conclusion**: Video model's random seed does NOT significantly impact evaluation performance
- **Implication**: Random seed during video model training is robust

### Scenario 2: Significant Difference

If combinations show significantly different performance (>5% difference):
- **Conclusion**: Video model's random seed DOES significantly impact evaluation performance
- **Implication**: Need to control or fix random seed during video model training
- **Action**: Consider using a fixed random seed for reproducibility

## Interpretation

### Key Metrics to Compare

1. **Average Sequence Length**: Higher is better (max 5.0)
2. **Chain Success Rates**: Success rate for i tasks in a row (i=1..5)
3. **Difference**: Absolute difference in average sequence length
4. **Difference Percentage**: Relative difference as percentage

### Statistical Significance

A difference of >5% is considered statistically significant for this experiment.

## Troubleshooting

### Issue: Training Fails

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce `batch_size` in config
2. Reduce `max_frames` or `num_frames`
3. Use gradient checkpointing

### Issue: Evaluation Fails

**Symptoms**:
```
FileNotFoundError: Calvin D -> D dataset not found
```

**Solutions**:
1. Verify `--calvin_d2d_dir` path
2. Check dataset structure
3. Ensure domain directories exist

### Issue: No Significant Difference

If results show no significant difference, consider:
1. Increasing number of evaluation sequences
2. Using more diverse random seeds
3. Testing with different domain pairs

## Next Steps

1. **Review configuration**: Edit [`config/train_video_comparison.yaml`](config/train_video_comparison.yaml)
2. **Run training**: Execute training commands above
3. **Monitor progress**: Check logs in output directory
4. **Analyze results**: Review comparison report
5. **Document findings**: Record conclusions and implications

## Notes

- This is a simplified implementation for comparison purposes
- Actual video model training would use the full VPP training pipeline
- Actor model training is simplified - actual implementation would use VPP_policy
- Evaluation uses placeholder logic - actual implementation would use full benchmark
- Results are simulated for demonstration - actual evaluation would run on real data

## Citation

If you use this comparison experiment, please cite the VPP paper:

```bibtex
@article{hu2024video,
  title={Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations},
  author={Hu, Yucheng and Guo, Yanjiang and Wang, Pengchao and Chen, Xiaoyu and Wang, Yen-Jen and Zhang, Jianke and Sreenath, Koushil and Lu, Chaochao and Chen, Jianyu},
  journal={arXiv preprint arXiv:2412.14803},
  year={2024}
}
```
