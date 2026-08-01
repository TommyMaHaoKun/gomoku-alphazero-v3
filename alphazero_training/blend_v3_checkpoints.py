#!/usr/bin/env python3
"""Create a parent-anchored weight-space blend of two evaluated V3 candidates."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping

import torch


_ARCHITECTURE_CONFIG_KEYS = (
    "board_size",
    "channels",
    "residual_blocks",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256.lower():
        raise ValueError(
            f"checkpoint SHA256 mismatch for {path}: expected "
            f"{expected_sha256.lower()}, got {actual}"
        )
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return payload, actual


def model_state(
    checkpoint: Mapping[str, Any], key: str, *, label: str
) -> dict[str, torch.Tensor]:
    raw = checkpoint.get(key)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{label} has no non-empty {key!r} model state")
    state: dict[str, torch.Tensor] = {}
    for name, tensor in raw.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{label} {key!r} contains a non-tensor parameter")
        state[name] = tensor.detach().cpu()
    return state


def blend_states(
    anchor: Mapping[str, torch.Tensor],
    update: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and within [0, 1]")
    if set(anchor) != set(update):
        raise ValueError("model states have different parameter names")
    blended: dict[str, torch.Tensor] = {}
    for name in anchor:
        left = anchor[name]
        right = update[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"model parameter mismatch: {name}")
        if left.is_floating_point() or left.is_complex():
            tensor = torch.lerp(left, right, alpha)
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"blended parameter is non-finite: {name}")
        else:
            tensor = right if alpha >= 0.5 else left
        blended[name] = tensor.clone()
    return blended


def verify_model_compatibility(
    anchor: Mapping[str, Any], update: Mapping[str, Any]
) -> None:
    """Reject architectural mismatches while allowing runtime-search differences."""
    if anchor.get("model_spec") != update.get("model_spec"):
        raise ValueError("checkpoint model specifications differ")
    anchor_config = anchor.get("config")
    update_config = update.get("config")
    if not isinstance(anchor_config, Mapping) or not isinstance(update_config, Mapping):
        raise ValueError("both checkpoints must contain mapping configs")
    mismatched = [
        key
        for key in _ARCHITECTURE_CONFIG_KEYS
        if anchor_config.get(key) != update_config.get(key)
    ]
    if mismatched:
        raise ValueError(
            "checkpoint architecture configurations differ: " + ", ".join(mismatched)
        )


def verify_parent_link(
    checkpoint: Mapping[str, Any],
    checkpoint_hash: str,
    parent_hash: str,
    *,
    label: str,
    allow_parent_identity: bool = False,
    parent_aliases: Iterable[str] = (),
) -> None:
    """Require parent provenance, allowing the approved parent itself as anchor."""
    if allow_parent_identity and checkpoint_hash == parent_hash:
        return
    recorded = checkpoint.get("source_parent_checkpoint_sha256")
    if recorded is None:
        recorded = checkpoint.get("parent_checkpoint_sha256")
    trusted_hashes = {parent_hash, *(str(value).lower() for value in parent_aliases)}
    if not isinstance(recorded, str) or recorded.lower() not in trusted_hashes:
        raise ValueError(
            f"{label} checkpoint is not linked to approved parent {parent_hash}"
        )


def approved_parent_aliases(checkpoint: Mapping[str, Any]) -> set[str]:
    """Return source hashes explicitly vouched for by an approved wrapper."""
    aliases = set()
    for key in ("source_checkpoint_sha256", "source_parent_checkpoint_sha256"):
        value = checkpoint.get(key)
        if isinstance(value, str) and len(value) == 64:
            aliases.add(value.lower())
    return aliases


def save_no_clobber(payload: Mapping[str, Any], output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output: {output}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite existing output: {output}") from error
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--update", type=Path, required=True)
    parser.add_argument("--update-sha256", required=True)
    parser.add_argument("--anchor-model-key", default="best_model")
    parser.add_argument("--update-model-key", default="best_model")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    anchor, anchor_hash = load_verified(args.anchor, args.anchor_sha256)
    update, update_hash = load_verified(args.update, args.update_sha256)
    if anchor.get("format_version") != 3 or update.get("format_version") != 3:
        raise ValueError("both inputs must be format-v3 checkpoints")
    verify_model_compatibility(anchor, update)
    parent_hash = args.parent_sha256.lower()
    verify_parent_link(
        anchor,
        anchor_hash,
        parent_hash,
        label="anchor",
        allow_parent_identity=True,
    )
    verify_parent_link(
        update,
        update_hash,
        parent_hash,
        label="update",
        parent_aliases=approved_parent_aliases(anchor),
    )
    state = blend_states(
        model_state(anchor, args.anchor_model_key, label="anchor"),
        model_state(update, args.update_model_key, label="update"),
        args.alpha,
    )
    payload = {
        "format_version": 3,
        "iteration": max(int(anchor.get("iteration", 0)), int(update.get("iteration", 0))),
        "global_step": max(
            int(anchor.get("global_step", 0)), int(update.get("global_step", 0))
        ),
        "v3_stage": "tactical_expert_warmstart",
        "config": dict(anchor["config"]),
        "model_spec": dict(anchor["model_spec"]),
        "train_model": state,
        "best_model": {name: tensor.clone() for name, tensor in state.items()},
        "parent_checkpoint_sha256": parent_hash,
        "warmstart_config": {
            "method": "linear_weight_space_blend",
            "alpha_update": float(args.alpha),
        },
        "blend_provenance": {
            "anchor": str(args.anchor.resolve()),
            "anchor_sha256": anchor_hash,
            "anchor_model_key": args.anchor_model_key,
            "update": str(args.update.resolve()),
            "update_sha256": update_hash,
            "update_model_key": args.update_model_key,
        },
        "saved_at_unix": time.time(),
    }
    save_no_clobber(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
                "alpha_update": args.alpha,
                "parent_checkpoint_sha256": parent_hash,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
