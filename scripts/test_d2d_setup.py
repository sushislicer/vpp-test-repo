#!/usr/bin/env python3
"""
Test script to verify Calvin D -> D benchmark setup.

This script checks:
1. Configuration file validity
2. Model paths existence
3. Dataset structure
4. Import dependencies
"""

import sys
from pathlib import Path
import json

# Add video-prediction-policy to path
sys.path.insert(0, str(Path(__file__).parent.parent / "video-prediction-policy"))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        import torch
        import hydra
        import numpy as np
        from omegaconf import OmegaConf
        print("✓ Basic imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    try:
        from utils.calvin_d2d_utils import (
            get_d2d_domains,
            get_domain_info,
            load_d2d_dataset,
            count_d2d_success,
            calculate_transfer_metrics
        )
        print("✓ D -> D utils imports successful")
    except ImportError as e:
        print(f"✗ D -> D utils import failed: {e}")
        return False
    
    return True


def test_config(config_path: Path):
    """Test if configuration file is valid."""
    print(f"\nTesting configuration file: {config_path}")
    
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        print("✓ Config file loaded successfully")
        
        # Check required fields
        required_fields = ['model', 'domains', 'num_sequences', 'ep_len']
        for field in required_fields:
            if field not in cfg:
                print(f"✗ Missing required field: {field}")
                return False
        print(f"✓ All required fields present")
        
        # Check domains
        domains = cfg.domains
        print(f"✓ Found {len(domains)} domain pairs")
        
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_model_paths(video_model_path: str, action_model_folder: str, clip_model_path: str):
    """Test if model paths exist."""
    print("\nTesting model paths...")
    
    video_path = Path(video_model_path)
    if video_path.exists():
        print(f"✓ Video model path exists: {video_model_path}")
    else:
        print(f"✗ Video model path not found: {video_model_path}")
    
    action_path = Path(action_model_folder)
    if action_path.exists():
        print(f"✓ Action model folder exists: {action_model_folder}")
    else:
        print(f"✗ Action model folder not found: {action_model_folder}")
    
    clip_path = Path(clip_model_path)
    if clip_path.exists():
        print(f"✓ CLIP model path exists: {clip_model_path}")
    else:
        print(f"✗ CLIP model path not found: {clip_model_path}")


def test_dataset_structure(calvin_d2d_dir: str):
    """Test if Calvin D -> D dataset has correct structure."""
    print(f"\nTesting dataset structure: {calvin_d2d_dir}")
    
    dataset_path = Path(calvin_d2d_dir)
    if not dataset_path.exists():
        print(f"✗ Dataset directory not found: {calvin_d2d_dir}")
        return False
    
    print(f"✓ Dataset directory exists")
    
    # Check for domain directories
    expected_domains = ['domain_A', 'domain_B', 'domain_C', 'domain_D']
    found_domains = []
    
    for domain in expected_domains:
        domain_path = dataset_path / domain
        if domain_path.exists():
            found_domains.append(domain)
            print(f"✓ Found {domain}")
        else:
            print(f"⚠ {domain} not found (optional)")
    
    if found_domains:
        print(f"✓ Found {len(found_domains)} domain directories")
    else:
        print(f"⚠ No domain directories found")
    
    return True


def test_d2d_utils():
    """Test D -> D utility functions."""
    print("\nTesting D -> D utility functions...")
    
    try:
        from utils.calvin_d2d_utils import (
            get_d2d_domains,
            get_domain_info,
            count_d2d_success,
            calculate_transfer_metrics
        )
        
        # Test get_d2d_domains
        test_domains = [["A", "A"], ["A", "B"], ["B", "A"]]
        domain_pairs = get_d2d_domains(test_domains)
        print(f"✓ get_d2d_domains: {len(domain_pairs)} pairs")
        
        # Test get_domain_info
        domain_info = get_domain_info("A")
        print(f"✓ get_domain_info: {domain_info['name']}")
        
        # Test count_d2d_success
        test_results = [5, 4, 3, 2, 1, 0]
        success_rates = count_d2d_success(test_results)
        print(f"✓ count_d2d_success: {success_rates}")
        
        # Test calculate_transfer_metrics
        all_results = {
            ("A", "A"): ([5, 4, 3], {}),
            ("A", "B"): ([4, 3, 2], {}),
            ("B", "A"): ([4, 3, 2], {})
        }
        metrics = calculate_transfer_metrics(all_results)
        print(f"✓ calculate_transfer_metrics: {len(metrics)} metric categories")
        
        return True
    except Exception as e:
        print(f"✗ D -> D utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("CALVIN D -> D BENCHMARK SETUP TEST")
    print("=" * 80)
    
    # Test imports
    if not test_imports():
        print("\n✗ Import tests failed. Please install required dependencies.")
        return False
    
    # Test config
    config_path = Path(__file__).parent / "config" / "calvin_d2d_config.yaml"
    if not test_config(config_path):
        print("\n✗ Config test failed. Please check configuration file.")
        return False
    
    # Test D -> D utils
    if not test_d2d_utils():
        print("\n✗ D -> D utils test failed. Please check utility functions.")
        return False
    
    # Test model paths (optional - requires actual paths)
    print("\n" + "=" * 80)
    print("OPTIONAL TESTS (require actual model/dataset paths)")
    print("=" * 80)
    print("\nTo test model and dataset paths, run:")
    print("python test_d2d_setup.py \\")
    print("    --video_model_path /path/to/svd-robot-calvin \\")
    print("    --action_model_folder /path/to/dp-calvin \\")
    print("    --clip_model_path /path/to/clip-vit-base-patch32 \\")
    print("    --calvin_d2d_dir /path/to/calvin/task_D_D")
    
    print("\n" + "=" * 80)
    print("✓ ALL BASIC TESTS PASSED!")
    print("=" * 80)
    print("\nYour setup is ready to run the Calvin D -> D benchmark.")
    print("Run the benchmark with:")
    print("accelerate launch --num_processes=4 run_calvin_d2d_benchmark.py \\")
    print("    --config config/calvin_d2d_config.yaml \\")
    print("    --video_model_path /path/to/svd-robot-calvin \\")
    print("    --action_model_folder /path/to/dp-calvin \\")
    print("    --clip_model_path /path/to/clip-vit-base-patch32 \\")
    print("    --calvin_d2d_dir /path/to/calvin/task_D_D")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Calvin D -> D benchmark setup")
    parser.add_argument("--video_model_path", type=str, default=None, help="Path to video model")
    parser.add_argument("--action_model_folder", type=str, default=None, help="Path to action model folder")
    parser.add_argument("--clip_model_path", type=str, default=None, help="Path to CLIP model")
    parser.add_argument("--calvin_d2d_dir", type=str, default=None, help="Path to Calvin D -> D dataset")
    
    args = parser.parse_args()
    
    # Run basic tests
    success = main()
    
    # Run optional tests if paths provided
    if args.video_model_path and args.action_model_folder and args.clip_model_path:
        test_model_paths(args.video_model_path, args.action_model_folder, args.clip_model_path)
    
    if args.calvin_d2d_dir:
        test_dataset_structure(args.calvin_d2d_dir)
    
    sys.exit(0 if success else 1)
