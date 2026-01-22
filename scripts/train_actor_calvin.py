#!/usr/bin/env python3
"""Train the VPP actor/policy on a CALVIN dataset without modifying upstream code.

Why this wrapper exists:
* The upstream entrypoint [`video-prediction-policy/step2_train_action_calvin.py`](../video-prediction-policy/step2_train_action_calvin.py)
  hard-codes some runtime behavior in its `__main__` block.
* This wrapper imports and calls its [`train()`](../video-prediction-policy/step2_train_action_calvin.py:78)
  function directly, so GPU selection, logging directories, and overrides can be
  controlled externally (e.g. via `accelerate launch`).

Typical usage (single node):

```bash
accelerate launch --num_processes=8 scripts/train_actor_calvin.py \
  --root_data_dir /data/calvin/task_D_D \
  --video_model_path /exp/video_seed_42 \
  --text_encoder_path /models/clip-vit-base-patch32 \
  --log_dir /exp/actor_seed_456 \
  --seed 456 \
  batch_size=28
```
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

from omegaconf import OmegaConf


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VPP actor (CALVIN) via wrapper")
    p.add_argument(
        "--vpp_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "video-prediction-policy"),
        help="Path to video-prediction-policy checkout",
    )
    p.add_argument(
        "--policy_conf_dir",
        type=str,
        default=None,
        help="Override policy_conf directory (defaults to <vpp_dir>/policy_conf)",
    )
    p.add_argument(
        "--config_name",
        type=str,
        default="VPP_Calvinabc_train",
        help="Hydra config name under policy_conf/ (default: VPP_Calvinabc_train)",
    )

    p.add_argument("--root_data_dir", type=str, required=True)
    p.add_argument("--video_model_path", type=str, required=True)
    p.add_argument("--text_encoder_path", type=str, required=True)
    p.add_argument("--log_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=None)

    # Hydra dotlist overrides go here, e.g. "batch_size=12" "max_epochs=50"
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    vpp_dir = Path(args.vpp_dir).resolve()
    policy_conf_dir = (
        Path(args.policy_conf_dir).resolve()
        if args.policy_conf_dir is not None
        else (vpp_dir / "policy_conf")
    )

    # Import upstream training function without executing its __main__ block.
    sys.path.insert(0, str(vpp_dir))
    
    # Ensure calvin_env is importable (check PYTHONPATH if missing)
    try:
        import calvin_env
    except ImportError:
        print("Error: calvin_env not found. Please set PYTHONPATH to include CALVIN_ROOT.", file=sys.stderr)
        print(f"Current PYTHONPATH: {os.environ.get('PYTHONPATH', '')}", file=sys.stderr)
        return 1

    from step2_train_action_calvin import train  # noqa: WPS433

    from hydra import compose, initialize  # noqa: WPS433

    with initialize(config_path=str(policy_conf_dir), job_name=args.config_name):
        cfg = compose(config_name=args.config_name)

    # Required path overrides
    cfg.root_data_dir = args.root_data_dir
    cfg.datamodule.root_data_dir = args.root_data_dir
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.text_encoder_path
    cfg.log_dir = args.log_dir

    # Seed override
    if args.seed is not None:
        cfg.seed = args.seed
        # keep model.seed consistent with top-level seed if present
        if "seed" in cfg.model:
            cfg.model.seed = args.seed

    # Apply optional dotlist overrides
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
