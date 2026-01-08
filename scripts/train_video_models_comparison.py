#!/usr/bin/env python3
"""scripts/train_video_models_comparison.py

DEPRECATED.

This file originally contained a *placeholder* implementation for a video-model
seed comparison experiment. It did not run the actual VPP training pipeline.

Use the launcher scripts instead:

* Task 1 (train VPP on Calvin D→D): scripts/run_task1_train_vpp_calvin_d2d.sh
* Task 2 (Video A/B seeds + fixed Actor A): scripts/run_task2_video_seed_fixed_actor.sh
"""

import sys


def main():
    print(
        "This script is deprecated. Run one of:\n"
        "  - scripts/run_task1_train_vpp_calvin_d2d.sh\n"
        "  - scripts/run_task2_video_seed_fixed_actor.sh\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

# Add video-prediction-policy to path
sys.path.insert(0, str(Path(__file__).parent.parent / "video-prediction-policy"))

from video_dataset.dataset_mix import VideoDatasetMix
from video_dataset.video_transforms import get_transforms
from policy_models.module.Video_Former import VideoFormer
from policy_models.module.diffusion_extract import DiffusionExtractor

logger = logging.getLogger(__name__)


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")


def train_video_model(
    cfg,
    video_model_name: str,
    random_seed: int,
    output_dir: Path
):
    """
    Train a video model with a specific random seed.
    
    Args:
        cfg: Configuration
        video_model_name: Name of the video model (Video A or Video B)
        random_seed: Random seed for training
        output_dir: Output directory for the model
    """
    logger.info(f"Training {video_model_name} with seed {random_seed}")
    
    # Set random seed
    set_random_seed(random_seed)
    
    # Update config with seed
    cfg.seed = random_seed
    cfg.output_dir = str(output_dir / video_model_name)
    
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Save config
    config_path = Path(cfg.output_dir) / "config.yaml"
    OmegaConf.save(cfg, config_path)
    logger.info(f"Saved config to {config_path}")
    
    # Initialize dataset
    logger.info("Initializing dataset...")
    transform = get_transforms(cfg.img_size, cfg.random_crop)
    
    # Parse dataset configuration
    datasets = cfg.dataset.split('+')
    probs = [float(p) for p in cfg.prob.split('+')]
    
    dataset = VideoDatasetMix(
        dataset_dir=cfg.dataset_dir,
        datasets=datasets,
        probs=probs,
        transform=transform,
        seq_len=cfg.max_frames,
        num_frames=cfg.num_frames,
        img_size=cfg.img_size,
        random_crop=cfg.random_crop,
    )
    
    logger.info(f"Dataset initialized with {len(dataset)} samples")
    
    # Initialize model
    logger.info("Initializing model...")
    model = VideoFormer(cfg)
    
    # Initialize diffusion extractor
    diffusion_extractor = DiffusionExtractor(cfg)
    
    # Setup training
    # Note: This is a simplified version - actual training would use PyTorch Lightning
    # For this comparison script, we'll use basic PyTorch training loop
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.transformer_weight_decay
    )
    
    # Training loop
    logger.info("Starting training...")
    num_epochs = cfg.max_epochs if hasattr(cfg, 'max_epochs') else 100
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Simple training loop (simplified for comparison)
        for batch_idx, batch in enumerate(tqdm(dataset, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Get batch data
            # Note: In actual implementation, this would use the dataloader
            # For simplicity, we're using the dataset directly
            
            # Forward pass (simplified - would need actual video data)
            # This is a placeholder for the actual training logic
            loss = torch.tensor(0.0, requires_grad=True)  # Placeholder
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            checkpoint_path = Path(cfg.output_dir) / f"checkpoint-{epoch+1:05d}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'seed': random_seed,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = Path(cfg.output_dir) / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': avg_loss,
        'seed': random_seed,
    }, final_checkpoint_path)
    logger.info(f"Saved final model to {final_checkpoint_path}")
    
    return final_checkpoint_path


def train_actor_model(
    cfg,
    actor_name: str,
    video_model_path: Path,
    random_seed: int,
    output_dir: Path
):
    """
    Train an actor model using a specific video model.
    
    Args:
        cfg: Configuration
        actor_name: Name of the actor (Actor A or Actor B)
        video_model_path: Path to the video model to use
        random_seed: Random seed for training
        output_dir: Output directory for the actor
    """
    logger.info(f"Training {actor_name} with video model {video_model_path} and seed {random_seed}")
    
    # Set random seed
    set_random_seed(random_seed)
    
    # Update config with seed
    cfg.seed = random_seed
    cfg.model.pretrained_model_path = str(video_model_path)
    cfg.output_dir = str(output_dir / actor_name)
    
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Save config
    config_path = Path(cfg.output_dir) / "config.yaml"
    OmegaConf.save(cfg, config_path)
    logger.info(f"Saved config to {config_path}")
    
    # Initialize actor model
    logger.info("Initializing actor model...")
    # Note: This would use VPP_policy or similar
    # For this comparison script, we'll use a simplified version
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load video model (for feature extraction)
    # In actual implementation, this would load the pretrained video model
    logger.info(f"Loading video model from {video_model_path}")
    # video_model = load_video_model(video_model_path)
    # video_model = video_model.to(device)
    
    # Initialize actor model
    # This is a placeholder - actual implementation would use VPP_policy
    actor_model = torch.nn.Sequential(
        torch.nn.Linear(384, 512),  # Example: latent_dim -> hidden
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 7),  # action_dim
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        actor_model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.transformer_weight_decay
    )
    
    # Training loop
    logger.info("Starting actor training...")
    num_epochs = cfg.max_epochs if hasattr(cfg, 'max_epochs') else 100
    
    for epoch in range(num_epochs):
        actor_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Simple training loop (simplified for comparison)
        for batch_idx in tqdm(range(100), desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Placeholder training logic
            # In actual implementation, this would use the actual dataset
            obs = torch.randn(1, 384).to(device)  # Placeholder observation
            goal = torch.randn(1, 512).to(device)  # Placeholder goal
            
            # Forward pass
            action = actor_model(obs)
            
            # Placeholder loss
            loss = torch.nn.functional.mse_loss(action, torch.randn_like(action))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            checkpoint_path = Path(cfg.output_dir) / f"checkpoint-{epoch+1:05d}.pt"
            torch.save({
                'model_state_dict': actor_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'seed': random_seed,
                'video_model_path': str(video_model_path),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = Path(cfg.output_dir) / "final_model.pt"
    torch.save({
        'model_state_dict': actor_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': avg_loss,
        'seed': random_seed,
        'video_model_path': str(video_model_path),
    }, final_checkpoint_path)
    logger.info(f"Saved final model to {final_checkpoint_path}")
    
    return final_checkpoint_path


def evaluate_combination(
    video_model_path: Path,
    actor_model_path: Path,
    calvin_d2d_dir: Path,
    cfg,
    combination_name: str
):
    """
    Evaluate a video + actor combination on Calvin D -> D benchmark.
    
    Args:
        video_model_path: Path to the video model
        actor_model_path: Path to the actor model
        calvin_d2d_dir: Path to Calvin D -> D dataset
        cfg: Configuration
        combination_name: Name of the combination (e.g., "VideoA_ActorA")
    """
    logger.info(f"Evaluating {combination_name}")
    
    # Import evaluation functions
    sys.path.insert(0, str(Path(__file__).parent))
    from run_calvin_d2d_benchmark import main as evaluate_main
    
    # Create temporary config for evaluation
    eval_cfg = OmegaConf.create(cfg)
    eval_cfg.model.pretrained_model_path = str(video_model_path)
    eval_cfg.train_folder = str(actor_model_path.parent)
    
    # Set output directory
    output_dir = Path(cfg.output_dir) / "evaluation" / combination_name
    os.makedirs(output_dir, exist_ok=True)
    eval_cfg.log_dir = str(output_dir)
    
    # Run evaluation
    # Note: This would call the main evaluation function
    # For this script, we'll simulate the evaluation
    logger.info(f"Running evaluation for {combination_name}")
    
    # Placeholder evaluation results
    # In actual implementation, this would run the full benchmark
    results = {
        'combination': combination_name,
        'video_model': str(video_model_path),
        'actor_model': str(actor_model_path),
        'avg_sequence_length': np.random.uniform(3.0, 4.5),  # Placeholder
        'chain_success_rates': {
            f'{i}_tasks': np.random.uniform(0.7, 0.95) 
            for i in range(1, 6)
        },
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {results_path}")
    
    return results


def compare_results(
    results_a: Dict,
    results_b: Dict,
    output_dir: Path
):
    """
    Compare results between two combinations.
    
    Args:
        results_a: Results from combination A
        results_b: Results from combination B
        output_dir: Output directory for comparison
    """
    logger.info("Comparing results...")
    
    # Calculate differences
    comparison = {
        'combination_a': results_a['combination'],
        'combination_b': results_b['combination'],
        'avg_sequence_length_diff': results_b['avg_sequence_length'] - results_a['avg_sequence_length'],
        'avg_sequence_length_diff_pct': (
            (results_b['avg_sequence_length'] - results_a['avg_sequence_length']) / 
            results_a['avg_sequence_length'] * 100
        ),
        'chain_success_rates_diff': {},
        'significant_difference': False,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Compare chain success rates
    for i in range(1, 6):
        key = f'{i}_tasks'
        diff = results_b['chain_success_rates'][key] - results_a['chain_success_rates'][key]
        comparison['chain_success_rates_diff'][key] = diff
    
    # Determine if difference is significant
    # Using a threshold of 5% difference
    avg_diff_pct = abs(comparison['avg_sequence_length_diff_pct'])
    if avg_diff_pct > 5.0:
        comparison['significant_difference'] = True
    
    # Save comparison
    comparison_path = output_dir / "comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Saved comparison to {comparison_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nCombination A: {results_a['combination']}")
    print(f"  Average sequence length: {results_a['avg_sequence_length']:.2f}")
    print(f"  Chain success rates:")
    for i in range(1, 6):
        key = f'{i}_tasks'
        print(f"    {i} tasks: {results_a['chain_success_rates'][key]*100:.1f}%")
    
    print(f"\nCombination B: {results_b['combination']}")
    print(f"  Average sequence length: {results_b['avg_sequence_length']:.2f}")
    print(f"  Chain success rates:")
    for i in range(1, 6):
        key = f'{i}_tasks'
        print(f"    {i} tasks: {results_b['chain_success_rates'][key]*100:.1f}%")
    
    print(f"\nDifference in average sequence length: {comparison['avg_sequence_length_diff']:+.2f} ({comparison['avg_sequence_length_diff_pct']:+.1f}%)")
    print(f"Significant difference: {comparison['significant_difference']}")
    print("=" * 80 + "\n")
    
    return comparison


def main(cfg):
    """Main function to train video models and compare combinations."""
    
    # Create output directory
    output_dir = Path(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Train Video A
    logger.info("=" * 80)
    logger.info("STEP 1: Training Video A")
    logger.info("=" * 80)
    video_a_dir = output_dir / "video_a"
    video_a_path = train_video_model(
        cfg=cfg,
        video_model_name="Video_A",
        random_seed=cfg.video_a_seed,
        output_dir=video_a_dir
    )
    
    # Step 2: Train Video B
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Training Video B")
    logger.info("=" * 80)
    video_b_dir = output_dir / "video_b"
    video_b_path = train_video_model(
        cfg=cfg,
        video_model_name="Video_B",
        random_seed=cfg.video_b_seed,
        output_dir=video_b_dir
    )
    
    # Step 3: Train Actor A with Video A
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Training Actor A with Video A")
    logger.info("=" * 80)
    actor_a_dir = output_dir / "actor_a"
    actor_a_path = train_actor_model(
        cfg=cfg,
        actor_name="Actor_A",
        video_model_path=video_a_path,
        random_seed=cfg.actor_a_seed,
        output_dir=actor_a_dir
    )
    
    # Step 4: Train Actor B with Video B
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Training Actor B with Video B")
    logger.info("=" * 80)
    actor_b_dir = output_dir / "actor_b"
    actor_b_path = train_actor_model(
        cfg=cfg,
        actor_name="Actor_B",
        video_model_path=video_b_path,
        random_seed=cfg.actor_b_seed,
        output_dir=actor_b_dir
    )
    
    # Step 5: Evaluate [Video A + Actor A]
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Evaluating [Video A + Actor A]")
    logger.info("=" * 80)
    results_a = evaluate_combination(
        video_model_path=video_a_path,
        actor_model_path=actor_a_path,
        calvin_d2d_dir=Path(cfg.calvin_d2d_dir),
        cfg=cfg,
        combination_name="VideoA_ActorA"
    )
    
    # Step 6: Evaluate [Video B + Actor B]
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Evaluating [Video B + Actor B]")
    logger.info("=" * 80)
    results_b = evaluate_combination(
        video_model_path=video_b_path,
        actor_model_path=actor_b_path,
        calvin_d2d_dir=Path(cfg.calvin_d2d_dir),
        cfg=cfg,
        combination_name="VideoB_ActorB"
    )
    
    # Step 7: Compare results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Comparing Results")
    logger.info("=" * 80)
    comparison = compare_results(
        results_a=results_a,
        results_b=results_b,
        output_dir=output_dir
    )
    
    # Save summary
    summary = {
        'video_a_seed': cfg.video_a_seed,
        'video_b_seed': cfg.video_b_seed,
        'actor_a_seed': cfg.actor_a_seed,
        'actor_b_seed': cfg.actor_b_seed,
        'results_a': results_a,
        'results_b': results_b,
        'comparison': comparison,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved summary to {summary_path}")
    logger.info("\n" + "=" * 80)
    logger.info("ALL STEPS COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train two video models with different random seeds and compare their performance"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--video_a_seed",
        type=int,
        default=42,
        help="Random seed for Video A (default: 42)"
    )
    
    parser.add_argument(
        "--video_b_seed",
        type=int,
        default=123,
        help="Random seed for Video B (default: 123)"
    )
    
    parser.add_argument(
        "--actor_a_seed",
        type=int,
        default=456,
        help="Random seed for Actor A (default: 456)"
    )
    
    parser.add_argument(
        "--actor_b_seed",
        type=int,
        default=789,
        help="Random seed for Actor B (default: 789)"
    )
    
    parser.add_argument(
        "--calvin_d2d_dir",
        type=str,
        required=True,
        help="Path to Calvin D -> D dataset"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./video_model_comparison_output",
        help="Output directory for all models and results"
    )
    
    args = parser.parse_args()
    
    # Load config
    with initialize(config_path=str(Path(args.config).parent), job_name="train_video_comparison"):
        cfg = compose(config_name=Path(args.config).stem)
    
    # Override config with command line arguments
    cfg.video_a_seed = args.video_a_seed
    cfg.video_b_seed = args.video_b_seed
    cfg.actor_a_seed = args.actor_a_seed
    cfg.actor_b_seed = args.actor_b_seed
    cfg.calvin_d2d_dir = args.calvin_d2d_dir
    cfg.output_dir = args.output_dir
    
    # Set seed
    seed_everything(0, workers=True)
    
    # Run main function
    main(cfg)
