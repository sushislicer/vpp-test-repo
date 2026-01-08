#!/usr/bin/env python3
"""
Calvin D -> D Benchmark Script for Video Prediction Policy (VPP)

This script evaluates VPP on the Calvin D -> D (Domain to Domain) benchmark,
which tests generalization across different visual domains (lighting, camera angles, backgrounds).

Usage:
    accelerate launch --num_processes=4 run_calvin_d2d_benchmark.py \
        --config config/calvin_d2d_config.yaml \
        --video_model_path /path/to/svd-robot-calvin \
        --action_model_folder /path/to/dp-calvin \
        --clip_model_path /path/to/clip-vit-base-patch32 \
        --calvin_d2d_dir /path/to/calvin/task_D_D
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
from tqdm.auto import tqdm
import wandb

# Add video-prediction-policy to path
sys.path.insert(0, str(Path(__file__).parent.parent / "video-prediction-policy"))

from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition
from policy_models.utils.utils import get_last_checkpoint
from policy_models.rollout.rollout_video import RolloutVideo

# Import D -> D specific utilities
from utils.calvin_d2d_utils import (
    get_d2d_domains,
    load_d2d_dataset,
    evaluate_d2d_sequence,
    count_d2d_success,
    print_d2d_results,
    save_d2d_results
)

logger = logging.getLogger(__name__)


def setup_distributed():
    """Setup distributed training if available."""
    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size()
    return 0, 1


def get_video_tag(i, rank=0, world_size=1):
    """Generate video tag for distributed evaluation."""
    return f"_d2d_long_horizon/sequence_{i * world_size + rank}"


def get_log_dir(log_dir, benchmark_name="calvin_d2d"):
    """Create and return log directory."""
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "evaluation" / benchmark_name
    else:
        log_dir = Path(log_dir)
    
    log_dir = log_dir / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}")
    return log_dir


def evaluate_policy_on_domain(
    model,
    env,
    lang_embeddings,
    cfg,
    source_domain: str,
    target_domain: str,
    num_videos: int = 0,
    save_dir: Path = None,
    rank: int = 0,
    world_size: int = 1
) -> Tuple[List[int], Dict]:
    """
    Evaluate policy on a specific domain transfer (source -> target).
    
    Args:
        model: VPP policy model
        env: Calvin environment
        lang_embeddings: Language embeddings
        cfg: Configuration
        source_domain: Source domain name
        target_domain: Target domain name
        num_videos: Number of videos to record
        save_dir: Directory to save videos
        rank: Process rank for distributed evaluation
        world_size: Total number of processes
    
    Returns:
        results: List of success counts for each sequence
        plans: Dictionary of plans
    """
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    
    # Setup video recording
    if num_videos > 0 and rank == 0:
        rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=save_dir,
            resolution_scale=1,
        )
    else:
        rollout_video = None
    
    # Get evaluation sequences
    eval_sequences = get_sequences(cfg.num_sequences)
    
    # Distribute sequences across processes
    if world_size > 1:
        eval_sequences = eval_sequences[rank::world_size]
    
    results = []
    plans = defaultdict(list)
    
    if not cfg.debug:
        eval_sequences = tqdm(eval_sequences, position=rank, leave=(rank == 0))
    
    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        record = (i < num_videos) and (rank == 0)
        
        # Evaluate sequence with domain transfer
        result = evaluate_d2d_sequence(
            env=env,
            model=model,
            task_checker=task_oracle,
            initial_state=initial_state,
            eval_sequence=eval_sequence,
            lang_embeddings=lang_embeddings,
            val_annotations=val_annotations,
            cfg=cfg,
            source_domain=source_domain,
            target_domain=target_domain,
            record=record,
            rollout_video=rollout_video,
            sequence_idx=i,
            rank=rank,
            world_size=world_size
        )
        
        results.append(result)
        
        if record and rollout_video is not None:
            rollout_video.write_to_tmp()
        
        if not cfg.debug:
            success_rates = count_d2d_success(results)
            average_rate = sum(success_rates) / len(success_rates) * 5
            description = " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_rates)])
            description += f" Average: {average_rate:.1f} |"
            eval_sequences.set_description(description)
        
        if result < 4 and record and rollout_video is not None:
            rollout_video._log_currentvideos_to_file(i, save_as_video=True)
    
    return results, plans


def evaluate_all_domains(
    model,
    env,
    lang_embeddings,
    cfg,
    domains: List[Tuple[str, str]],
    num_videos: int = 0,
    save_dir: Path = None,
    rank: int = 0,
    world_size: int = 1
) -> Dict[Tuple[str, str], Tuple[List[int], Dict]]:
    """
    Evaluate policy on all domain transfer pairs.
    
    Args:
        model: VPP policy model
        env: Calvin environment
        lang_embeddings: Language embeddings
        cfg: Configuration
        domains: List of (source_domain, target_domain) tuples
        num_videos: Number of videos to record per domain
        save_dir: Directory to save videos
        rank: Process rank for distributed evaluation
        world_size: Total number of processes
    
    Returns:
        all_results: Dictionary mapping domain pairs to (results, plans)
    """
    all_results = {}
    
    for source_domain, target_domain in domains:
        logger.info(f"Evaluating {source_domain} -> {target_domain}")
        
        domain_save_dir = save_dir / f"{source_domain}_to_{target_domain}" if save_dir else None
        if domain_save_dir:
            os.makedirs(domain_save_dir, exist_ok=True)
        
        results, plans = evaluate_policy_on_domain(
            model=model,
            env=env,
            lang_embeddings=lang_embeddings,
            cfg=cfg,
            source_domain=source_domain,
            target_domain=target_domain,
            num_videos=num_videos,
            save_dir=domain_save_dir,
            rank=rank,
            world_size=world_size
        )
        
        all_results[(source_domain, target_domain)] = (results, plans)
        
        # Print intermediate results
        if rank == 0:
            avg_seq_len = np.mean(results)
            logger.info(f"{source_domain} -> {target_domain}: Avg sequence length = {avg_seq_len:.2f}")
    
    return all_results


def main(cfg):
    """Main evaluation function."""
    # Setup distributed training
    rank, world_size = setup_distributed()
    
    # Set seed
    seed_everything(cfg.seed, workers=True)
    
    # Setup logging
    log_wandb = cfg.log_wandb
    if log_wandb and rank == 0:
        log_dir = get_log_dir(cfg.log_dir, cfg.benchmark_name)
        os.makedirs(log_dir / "wandb", exist_ok=True)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.benchmark_name}_{time.strftime('%Y%m%d_%H%M%S')}",
            config=OmegaConf.to_container(cfg),
            dir=str(log_dir / "wandb")
        )
    else:
        log_dir = get_log_dir(cfg.log_dir, cfg.benchmark_name)
    
    # Set device
    device_id = cfg.device if world_size == 1 else rank
    torch.cuda.set_device(device_id)
    
    logger.info(f"Rank {rank}/{world_size}, Device: cuda:{device_id}")
    
    # Load checkpoints
    checkpoints = [get_last_checkpoint(Path(cfg.train_folder))]
    
    # Get D -> D domains
    domains = get_d2d_domains(cfg.domains)
    logger.info(f"Evaluating on {len(domains)} domain transfer pairs: {domains}")
    
    # Load environment and language embeddings
    env = None
    lang_embeddings = None
    
    for checkpoint in checkpoints:
        logger.info(f"Loading checkpoint: {checkpoint}")
        
        env, _, lang_embeddings = get_default_beso_and_env(
            cfg.train_folder,
            cfg.root_data_dir,
            checkpoint,
            env=env,
            lang_embeddings=lang_embeddings,
            eval_cfg_overwrite=cfg.eval_cfg_overwrite,
            device_id=device_id,
            cfg=cfg,
        )
        
        # Load model
        ckpt_path = os.path.join(cfg.train_folder, 'saved_models')
        ckpt_files = [f for f in os.listdir(ckpt_path) if f.endswith('.pt')]
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_path}")
        
        ckpt = os.path.join(ckpt_path, ckpt_files[0])
        logger.info(f"Loading model from {ckpt}")
        
        state_dict = torch.load(ckpt, map_location='cpu')
        device = torch.device(f"cuda:{device_id}")
        
        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict['model'], strict=False)
        model.freeze()
        model = model.cuda(device)
        
        # Set model parameters
        model.num_sampling_steps = cfg.num_sampling_steps
        model.sampler_type = cfg.sampler_type
        model.multistep = cfg.multistep
        if cfg.sigma_min is not None:
            model.sigma_min = cfg.sigma_min
        if cfg.sigma_max is not None:
            model.sigma_max = cfg.sigma_max
        if cfg.noise_scheduler is not None:
            model.noise_scheduler = cfg.noise_scheduler
        
        if cfg.cfg_value != 1:
            raise NotImplementedError("cfg_value != 1 not implemented yet")
        
        model.process_device()
        model.eval()
        
        logger.info(f"Model parameters: num_sampling_steps={cfg.num_sampling_steps}, "
                   f"sampler_type={cfg.sampler_type}, multistep={cfg.multistep}")
        
        # Evaluate on all domains
        save_dir = log_dir / "videos" if log_wandb else None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        all_results = evaluate_all_domains(
            model=model,
            env=env,
            lang_embeddings=lang_embeddings,
            cfg=cfg,
            domains=domains,
            num_videos=cfg.num_videos,
            save_dir=save_dir,
            rank=rank,
            world_size=world_size
        )
        
        # Gather results from all processes
        if world_size > 1:
            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, all_results)
            if rank == 0:
                # Merge results from all processes
                merged_results = {}
                for domain_pair in domains:
                    merged_results[domain_pair] = ([], {})
                    for proc_results in gathered_results:
                        if domain_pair in proc_results:
                            merged_results[domain_pair][0].extend(proc_results[domain_pair][0])
                            merged_results[domain_pair][1].update(proc_results[domain_pair][1])
                all_results = merged_results
        else:
            gathered_results = [all_results]
        
        # Print and save results (only on rank 0)
        if rank == 0:
            print_d2d_results(all_results, cfg, log_dir)
            save_d2d_results(all_results, cfg, log_dir)
            
            if log_wandb:
                # Log to wandb
                for domain_pair, (results, _) in all_results.items():
                    source_domain, target_domain = domain_pair
                    avg_seq_len = np.mean(results)
                    chain_sr = {i + 1: sr for i, sr in enumerate(count_d2d_success(results))}
                    
                    wandb.log({
                        f"d2d/{source_domain}_to_{target_domain}/avg_seq_len": avg_seq_len,
                        f"d2d/{source_domain}_to_{target_domain}/chain_sr": chain_sr
                    })
                
                # Calculate overall statistics
                all_results_list = []
                for results, _ in all_results.values():
                    all_results_list.extend(results)
                
                overall_avg = np.mean(all_results_list)
                overall_chain_sr = {i + 1: sr for i, sr in enumerate(count_d2d_success(all_results_list))}
                
                wandb.log({
                    "d2d/overall/avg_seq_len": overall_avg,
                    "d2d/overall/chain_sr": overall_chain_sr
                })
                
                wandb.finish()
        
        logger.info("Evaluation completed!")


if __name__ == "__main__":
    # Set environment variables for distributed training
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run VPP on Calvin D -> D benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--video_model_path", type=str, required=True, help="Path to video model")
    parser.add_argument("--action_model_folder", type=str, required=True, help="Path to action model folder")
    parser.add_argument("--clip_model_path", type=str, required=True, help="Path to CLIP model")
    parser.add_argument("--calvin_d2d_dir", type=str, required=True, help="Path to Calvin D -> D dataset")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory to save logs")
    parser.add_argument("--num_videos", type=int, default=10, help="Number of videos to record")
    parser.add_argument("--num_sequences", type=int, default=1000, help="Number of evaluation sequences")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with initialize(config_path=str(config_path.parent), job_name="calvin_d2d_benchmark"):
        cfg = compose(config_name=config_path.stem)
    
    # Override config with command line arguments
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.train_folder = args.action_model_folder
    cfg.model.text_encoder_path = args.clip_model_path
    cfg.root_data_dir = args.calvin_d2d_dir
    cfg.log_dir = args.log_dir
    cfg.num_videos = args.num_videos
    cfg.num_sequences = args.num_sequences
    cfg.debug = args.debug
    cfg.log_wandb = not args.no_wandb
    
    # Run main evaluation
    main(cfg)
