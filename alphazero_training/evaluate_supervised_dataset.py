#!/usr/bin/env python3
"""Evaluate one supervised V3 dataset with a deterministic group split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .train_alphazero import Config, PolicyValueNet
from .train_v3_supervised import DatasetPool, evaluate, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-key", default="best_model")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.model_key not in checkpoint:
        raise ValueError(f"checkpoint has no model key {args.model_key!r}")
    config = Config(**checkpoint["config"])
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = PolicyValueNet(
        config.board_size, config.channels, config.residual_blocks
    ).to(device)
    model.load_state_dict(checkpoint[args.model_key], strict=True)
    pool = DatasetPool(args.dataset, args.seed, args.validation_fraction)
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "model_key": args.model_key,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        **evaluate(model, [pool], device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
