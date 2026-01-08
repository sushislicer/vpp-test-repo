"""
Utility functions for Calvin D -> D (Domain to Domain) benchmark evaluation.

This module provides helper functions for:
- Loading and managing D -> D domain configurations
- Evaluating sequences with domain transfer
- Calculating and displaying D -> D specific metrics
- Saving and loading D -> D results
"""

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from termcolor import colored

logger = logging.getLogger(__name__)


# Domain definitions for Calvin D -> D benchmark
DOMAIN_DEFINITIONS = {
    "A": {
        "name": "Domain A",
        "description": "Standard lighting and camera angle",
        "lighting": "normal",
        "camera_angle": "default",
        "background": "standard"
    },
    "B": {
        "name": "Domain B",
        "description": "Dim lighting conditions",
        "lighting": "dim",
        "camera_angle": "default",
        "background": "standard"
    },
    "C": {
        "name": "Domain C",
        "description": "Bright lighting conditions",
        "lighting": "bright",
        "camera_angle": "default",
        "background": "standard"
    },
    "D": {
        "name": "Domain D",
        "description": "Alternate camera angle",
        "lighting": "normal",
        "camera_angle": "alternate",
        "background": "standard"
    }
}


def get_d2d_domains(domains_config: List[List[str]]) -> List[Tuple[str, str]]:
    """
    Parse domain configuration and return list of (source, target) domain pairs.
    
    Args:
        domains_config: List of [source_domain, target_domain] pairs from config
    
    Returns:
        List of (source_domain, target_domain) tuples
    """
    domain_pairs = []
    for pair in domains_config:
        if len(pair) == 2:
            source, target = pair
            if source in DOMAIN_DEFINITIONS and target in DOMAIN_DEFINITIONS:
                domain_pairs.append((source, target))
            else:
                logger.warning(f"Invalid domain pair: {pair}. Skipping.")
        else:
            logger.warning(f"Invalid domain pair format: {pair}. Skipping.")
    
    logger.info(f"Loaded {len(domain_pairs)} domain transfer pairs")
    return domain_pairs


def get_domain_info(domain: str) -> Dict:
    """
    Get information about a specific domain.
    
    Args:
        domain: Domain identifier (A, B, C, or D)
    
    Returns:
        Dictionary with domain information
    """
    if domain not in DOMAIN_DEFINITIONS:
        raise ValueError(f"Unknown domain: {domain}. Available domains: {list(DOMAIN_DEFINITIONS.keys())}")
    return DOMAIN_DEFINITIONS[domain]


def load_d2d_dataset(root_data_dir: str, domain: str) -> Dict:
    """
    Load Calvin D -> D dataset for a specific domain.
    
    Args:
        root_data_dir: Root directory of Calvin D -> D dataset
        domain: Domain identifier (A, B, C, or D)
    
    Returns:
        Dictionary containing dataset information
    """
    domain_dir = Path(root_data_dir) / f"domain_{domain}"
    
    if not domain_dir.exists():
        logger.warning(f"Domain directory not found: {domain_dir}")
        return {
            "domain": domain,
            "path": str(domain_dir),
            "exists": False,
            "episodes": []
        }
    
    # Load episode information
    episodes = []
    if (domain_dir / "episodes.json").exists():
        with open(domain_dir / "episodes.json", "r") as f:
            episodes = json.load(f)
    
    return {
        "domain": domain,
        "path": str(domain_dir),
        "exists": True,
        "episodes": episodes,
        "num_episodes": len(episodes)
    }


def evaluate_d2d_sequence(
    env,
    model,
    task_checker,
    initial_state,
    eval_sequence,
    lang_embeddings,
    val_annotations,
    cfg,
    source_domain: str,
    target_domain: str,
    record: bool = False,
    rollout_video=None,
    sequence_idx: int = 0,
    rank: int = 0,
    world_size: int = 1
) -> int:
    """
    Evaluate a single sequence with domain transfer from source to target domain.
    
    Args:
        env: Calvin environment
        model: VPP policy model
        task_checker: Task oracle for checking task completion
        initial_state: Initial state for the sequence
        eval_sequence: List of tasks to evaluate
        lang_embeddings: Language embeddings
        val_annotations: Validation annotations
        cfg: Configuration
        source_domain: Source domain identifier
        target_domain: Target domain identifier
        record: Whether to record video
        rollout_video: Video recording object
        sequence_idx: Index of the sequence
        rank: Process rank
        world_size: Total number of processes
    
    Returns:
        Number of successfully completed tasks in the sequence
    """
    from policy_evaluation.utils import get_env_state_for_initial_condition
    
    # Reset environment with initial state
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    
    if record and rollout_video is not None:
        caption = f"{source_domain} -> {target_domain}: " + " | ".join(eval_sequence)
        rollout_video.new_video(tag=f"_d2d_{source_domain}_to_{target_domain}/sequence_{sequence_idx}", caption=caption)
    
    success_counter = 0
    
    if cfg.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {source_domain} -> {target_domain}: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    
    for subtask in eval_sequence:
        if record and rollout_video is not None:
            rollout_video.new_subtask()
        
        # Rollout for this subtask
        success = rollout_d2d(
            env=env,
            model=model,
            task_oracle=task_checker,
            cfg=cfg,
            subtask=subtask,
            lang_embeddings=lang_embeddings,
            val_annotations=val_annotations,
            source_domain=source_domain,
            target_domain=target_domain,
            record=record,
            rollout_video=rollout_video
        )
        
        if record and rollout_video is not None:
            rollout_video.draw_outcome(success)
        
        if success:
            success_counter += 1
        else:
            return success_counter
    
    return success_counter


def rollout_d2d(
    env,
    model,
    task_oracle,
    cfg,
    subtask,
    lang_embeddings,
    val_annotations,
    source_domain: str,
    target_domain: str,
    record: bool = False,
    rollout_video=None
) -> bool:
    """
    Rollout a single subtask with domain transfer.
    
    Args:
        env: Calvin environment
        model: VPP policy model
        task_oracle: Task oracle for checking task completion
        cfg: Configuration
        subtask: Subtask to execute
        lang_embeddings: Language embeddings
        val_annotations: Validation annotations
        source_domain: Source domain identifier
        target_domain: Target domain identifier
        record: Whether to record video
        rollout_video: Video recording object
    
    Returns:
        True if task was completed successfully, False otherwise
    """
    if cfg.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    
    obs = env.get_obs()
    
    # Get language annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    
    # Get language goal embedding
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal['lang_text'] = val_annotations[subtask][0]
    
    # Add domain information to goal
    goal['source_domain'] = source_domain
    goal['target_domain'] = target_domain
    
    model.reset()
    start_info = env.get_info()
    
    for step in range(cfg.ep_len):
        # Get action from model
        action = model.step(obs, goal)
        
        # Execute action
        obs, _, _, current_info = env.step(action)
        
        if cfg.debug:
            img = env.render(mode="rgb_array")
            # Could add domain-specific visualization here
            # join_vis_lang(img, lang_annotation)
        
        if record and rollout_video is not None:
            rollout_video.update(obs["rgb_obs"]["rgb_static"])
        
        # Check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if cfg.debug:
                print(colored("success", "green"), end=" ")
            if record and rollout_video is not None:
                rollout_video.add_language_instruction(lang_annotation)
            return True
    
    if cfg.debug:
        print(colored("fail", "red"), end=" ")
    if record and rollout_video is not None:
        rollout_video.add_language_instruction(lang_annotation)
    
    return False


def count_d2d_success(results: List[int]) -> List[float]:
    """
    Calculate success rates for different sequence lengths.
    
    Args:
        results: List of success counts for each sequence
    
    Returns:
        List of success rates for sequences of length 1, 2, 3, 4, 5
    """
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results) if len(results) > 0 else 0.0
        step_success.append(sr)
    return step_success


def calculate_transfer_metrics(all_results: Dict[Tuple[str, str], Tuple[List[int], Dict]]) -> Dict:
    """
    Calculate transfer learning metrics across domain pairs.
    
    Args:
        all_results: Dictionary mapping domain pairs to (results, plans)
    
    Returns:
        Dictionary containing transfer metrics
    """
    metrics = {
        "same_domain": {},
        "cross_domain": {},
        "transfer_gap": {},
        "domain_similarity": {}
    }
    
    # Calculate same-domain performance (baseline)
    for domain_pair, (results, _) in all_results.items():
        source, target = domain_pair
        if source == target:
            avg_seq_len = np.mean(results)
            metrics["same_domain"][source] = avg_seq_len
    
    # Calculate cross-domain performance
    for domain_pair, (results, _) in all_results.items():
        source, target = domain_pair
        if source != target:
            avg_seq_len = np.mean(results)
            if source not in metrics["cross_domain"]:
                metrics["cross_domain"][source] = {}
            metrics["cross_domain"][source][target] = avg_seq_len
    
    # Calculate transfer gap (performance drop when transferring)
    for source in metrics["same_domain"]:
        baseline = metrics["same_domain"][source]
        if source in metrics["cross_domain"]:
            for target in metrics["cross_domain"][source]:
                transfer_perf = metrics["cross_domain"][source][target]
                gap = baseline - transfer_perf
                metrics["transfer_gap"][(source, target)] = gap
    
    # Calculate domain similarity based on transfer performance
    # Higher similarity = smaller transfer gap
    for (source, target), gap in metrics["transfer_gap"].items():
        similarity = max(0, 1 - gap / 5.0)  # Normalize to [0, 1]
        metrics["domain_similarity"][(source, target)] = similarity
    
    return metrics


def print_d2d_results(all_results: Dict[Tuple[str, str], Tuple[List[int], Dict]], cfg, log_dir: Path):
    """
    Print D -> D evaluation results in a formatted way.
    
    Args:
        all_results: Dictionary mapping domain pairs to (results, plans)
        cfg: Configuration
        log_dir: Log directory
    """
    print("\n" + "=" * 80)
    print("CALVIN D -> D BENCHMARK RESULTS")
    print("=" * 80)
    
    # Print results for each domain pair
    for domain_pair, (results, _) in sorted(all_results.items()):
        source, target = domain_pair
        avg_seq_len = np.mean(results)
        chain_sr = {i + 1: sr for i, sr in enumerate(count_d2d_success(results))}
        
        print(f"\n{source} -> {target}:")
        print(f"  Average sequence length: {avg_seq_len:.2f}")
        print(f"  Success rates:")
        for i, sr in chain_sr.items():
            print(f"    {i} tasks in a row: {sr * 100:.1f}%")
    
    # Calculate and print transfer metrics
    transfer_metrics = calculate_transfer_metrics(all_results)
    
    print("\n" + "-" * 80)
    print("TRANSFER METRICS")
    print("-" * 80)
    
    if transfer_metrics["same_domain"]:
        print("\nSame-domain performance (baseline):")
        for domain, perf in sorted(transfer_metrics["same_domain"].items()):
            print(f"  Domain {domain}: {perf:.2f}")
    
    if transfer_metrics["transfer_gap"]:
        print("\nTransfer gap (performance drop):")
        for (source, target), gap in sorted(transfer_metrics["transfer_gap"].items()):
            print(f"  {source} -> {target}: {gap:.2f}")
    
    if transfer_metrics["domain_similarity"]:
        print("\nDomain similarity (based on transfer performance):")
        for (source, target), similarity in sorted(transfer_metrics["domain_similarity"].items()):
            print(f"  {source} -> {target}: {similarity:.2f}")
    
    # Calculate overall statistics
    all_results_list = []
    for results, _ in all_results.values():
        all_results_list.extend(results)
    
    overall_avg = np.mean(all_results_list)
    overall_chain_sr = {i + 1: sr for i, sr in enumerate(count_d2d_success(all_results_list))}
    
    print("\n" + "-" * 80)
    print("OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total sequences evaluated: {len(all_results_list)}")
    print(f"Overall average sequence length: {overall_avg:.2f}")
    print(f"Overall success rates:")
    for i, sr in overall_chain_sr.items():
        print(f"  {i} tasks in a row: {sr * 100:.1f}%")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {log_dir}")
    print("=" * 80 + "\n")


def save_d2d_results(all_results: Dict[Tuple[str, str], Tuple[List[int], Dict]], cfg, log_dir: Path):
    """
    Save D -> D evaluation results to JSON files.
    
    Args:
        all_results: Dictionary mapping domain pairs to (results, plans)
        cfg: Configuration
        log_dir: Log directory
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Prepare results data
    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_sequences": cfg.num_sequences,
            "ep_len": cfg.ep_len,
            "num_sampling_steps": cfg.num_sampling_steps,
            "sampler_type": cfg.sampler_type,
            "multistep": cfg.multistep
        },
        "domain_results": {},
        "transfer_metrics": {},
        "overall_stats": {}
    }
    
    # Add results for each domain pair
    for domain_pair, (results, _) in all_results.items():
        source, target = domain_pair
        avg_seq_len = np.mean(results)
        chain_sr = {i + 1: sr for i, sr in enumerate(count_d2d_success(results))}
        
        results_data["domain_results"][f"{source}_to_{target}"] = {
            "source_domain": source,
            "target_domain": target,
            "num_sequences": len(results),
            "avg_sequence_length": float(avg_seq_len),
            "chain_success_rates": {f"{i}_tasks": float(sr) for i, sr in chain_sr.items()},
            "results": results
        }
    
    # Add transfer metrics
    transfer_metrics = calculate_transfer_metrics(all_results)
    results_data["transfer_metrics"] = {
        "same_domain": {k: float(v) for k, v in transfer_metrics["same_domain"].items()},
        "cross_domain": {f"{k[0]}_to_{k[1]}": float(v) for k, v in transfer_metrics["cross_domain"].items()},
        "transfer_gap": {f"{k[0]}_to_{k[1]}": float(v) for k, v in transfer_metrics["transfer_gap"].items()},
        "domain_similarity": {f"{k[0]}_to_{k[1]}": float(v) for k, v in transfer_metrics["domain_similarity"].items()}
    }
    
    # Add overall statistics
    all_results_list = []
    for results, _ in all_results.values():
        all_results_list.extend(results)
    
    overall_avg = np.mean(all_results_list)
    overall_chain_sr = {i + 1: sr for i, sr in enumerate(count_d2d_success(all_results_list))}
    
    results_data["overall_stats"] = {
        "total_sequences": len(all_results_list),
        "avg_sequence_length": float(overall_avg),
        "chain_success_rates": {f"{i}_tasks": float(sr) for i, sr in overall_chain_sr.items()}
    }
    
    # Save results
    results_file = log_dir / "d2d_results.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Also save a summary file
    summary_file = log_dir / "d2d_summary.txt"
    with open(summary_file, "w") as f:
        f.write("CALVIN D -> D BENCHMARK SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {results_data['timestamp']}\n")
        f.write(f"Total sequences: {results_data['overall_stats']['total_sequences']}\n")
        f.write(f"Overall avg sequence length: {results_data['overall_stats']['avg_sequence_length']:.2f}\n\n")
        
        f.write("DOMAIN PAIR RESULTS:\n")
        f.write("-" * 80 + "\n")
        for domain_pair, data in results_data["domain_results"].items():
            f.write(f"\n{domain_pair}:\n")
            f.write(f"  Avg sequence length: {data['avg_sequence_length']:.2f}\n")
            f.write(f"  Success rates:\n")
            for i, sr in data["chain_success_rates"].items():
                f.write(f"    {i}: {sr * 100:.1f}%\n")
        
        f.write("\n\nTRANSFER METRICS:\n")
        f.write("-" * 80 + "\n")
        if results_data["transfer_metrics"]["same_domain"]:
            f.write("\nSame-domain performance:\n")
            for domain, perf in results_data["transfer_metrics"]["same_domain"].items():
                f.write(f"  Domain {domain}: {perf:.2f}\n")
        
        if results_data["transfer_metrics"]["transfer_gap"]:
            f.write("\nTransfer gap:\n")
            for pair, gap in results_data["transfer_metrics"]["transfer_gap"].items():
                f.write(f"  {pair}: {gap:.2f}\n")
    
    logger.info(f"Summary saved to {summary_file}")


def load_d2d_results(results_file: Path) -> Dict:
    """
    Load D -> D evaluation results from JSON file.
    
    Args:
        results_file: Path to results JSON file
    
    Returns:
        Dictionary containing loaded results
    """
    with open(results_file, "r") as f:
        results = json.load(f)
    return results


def compare_d2d_results(results1: Dict, results2: Dict) -> Dict:
    """
    Compare two D -> D evaluation results.
    
    Args:
        results1: First results dictionary
        results2: Second results dictionary
    
    Returns:
        Dictionary containing comparison metrics
    """
    comparison = {
        "timestamp1": results1.get("timestamp", "unknown"),
        "timestamp2": results2.get("timestamp", "unknown"),
        "domain_comparison": {},
        "overall_comparison": {}
    }
    
    # Compare overall statistics
    overall1 = results1.get("overall_stats", {})
    overall2 = results2.get("overall_stats", {})
    
    comparison["overall_comparison"] = {
        "avg_sequence_length": {
            "results1": overall1.get("avg_sequence_length", 0),
            "results2": overall2.get("avg_sequence_length", 0),
            "difference": overall2.get("avg_sequence_length", 0) - overall1.get("avg_sequence_length", 0)
        }
    }
    
    # Compare domain pair results
    domain_results1 = results1.get("domain_results", {})
    domain_results2 = results2.get("domain_results", {})
    
    for domain_pair in set(domain_results1.keys()) | set(domain_results2.keys()):
        if domain_pair in domain_results1 and domain_pair in domain_results2:
            avg1 = domain_results1[domain_pair]["avg_sequence_length"]
            avg2 = domain_results2[domain_pair]["avg_sequence_length"]
            comparison["domain_comparison"][domain_pair] = {
                "results1": avg1,
                "results2": avg2,
                "difference": avg2 - avg1
            }
    
    return comparison
