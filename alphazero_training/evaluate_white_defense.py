#!/usr/bin/env python3
"""Independently evaluate a checkpoint on the held-out white-defense archive.

The evaluator deliberately accepts only the ``artifacts.eval`` archive named
by a verified white-defense manifest.  Policy probabilities are renormalized
inside each record's ``candidate_mask`` before safe/unsafe mass is measured;
actions outside that bounded candidate set are reported separately and are
never counted as unsafe.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from .train_alphazero import PolicyValueNet
    from .white_defense_dataset import (
        ACTION_COUNT,
        BOARD_SIZE,
        EVAL_SPLIT,
        WhiteDefenseDataset,
        stable_json_sha256,
        validate_dataset,
    )
except ImportError:  # pragma: no cover - direct script execution convenience.
    from train_alphazero import PolicyValueNet  # type: ignore
    from white_defense_dataset import (  # type: ignore
        ACTION_COUNT,
        BOARD_SIZE,
        EVAL_SPLIT,
        WhiteDefenseDataset,
        stable_json_sha256,
        validate_dataset,
    )


MODEL_KEYS = ("candidate_model", "train_model", "best_model")
SUPPORTED_MODEL_KEYS = frozenset(MODEL_KEYS)
TRAINING_PROHIBITION = "never pass this archive to a trainer"
V3_AUTO_PRIORITIES = {
    # During self-play, best_model is deliberately the carried old champion.
    "selfplay": ("candidate_model", "train_model", "best_model"),
    # Supervised warm-start output is the network to train next, not a deployed
    # champion.  Current writers make train_model and best_model identical.
    "tactical_expert_warmstart": ("train_model", "best_model"),
    # A frozen candidate is explicitly under evaluation.
    "candidate_eval": ("candidate_model", "train_model", "best_model"),
    # Approved/screened V3 checkpoints use best_model as the playable model.
    "development_screened": ("best_model", "candidate_model", "train_model"),
    "external_champion": ("best_model", "candidate_model", "train_model"),
}
REPORT_SCHEMA_VERSION = 1


class WhiteDefenseEvaluationError(ValueError):
    """An input failed an integrity, schema, model, or numerical check."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_sha256(value: object, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise WhiteDefenseEvaluationError(f"{label} is not a SHA256 digest")
    return digest


def _strict_count(value: object, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WhiteDefenseEvaluationError(f"{label} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise WhiteDefenseEvaluationError(f"{label} must be {qualifier}")
    return value


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))


def _read_verified_manifest(manifest_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise WhiteDefenseEvaluationError(f"manifest does not exist: {manifest_path}")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise WhiteDefenseEvaluationError(f"manifest SHA256 sidecar does not exist: {sidecar}")

    manifest_data = manifest_path.read_bytes()
    actual_manifest_sha256 = sha256_bytes(manifest_data)
    try:
        sidecar_text = sidecar.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as error:
        raise WhiteDefenseEvaluationError("manifest sidecar is not UTF-8") from error
    if len(sidecar_text) < 66 or not sidecar_text[64].isspace():
        raise WhiteDefenseEvaluationError(
            "manifest sidecar must contain exactly '<sha256> <filename>'"
        )
    expected_manifest_sha256 = _normalized_sha256(
        sidecar_text[:64], label="manifest sidecar hash"
    )
    recorded_filename = sidecar_text[64:].strip()
    if not recorded_filename or Path(recorded_filename).name != manifest_path.name:
        raise WhiteDefenseEvaluationError("manifest sidecar names a different file")
    if expected_manifest_sha256 != actual_manifest_sha256:
        raise WhiteDefenseEvaluationError(
            "manifest SHA256 mismatch: "
            f"expected {expected_manifest_sha256}, actual {actual_manifest_sha256}"
        )

    try:
        manifest = json.loads(manifest_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WhiteDefenseEvaluationError(f"manifest is not valid UTF-8 JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise WhiteDefenseEvaluationError("manifest root must be a JSON object")
    recorded_payload_hash = _normalized_sha256(
        manifest.get("manifest_payload_sha256"), label="manifest payload hash"
    )
    payload_without_hash = dict(manifest)
    payload_without_hash.pop("manifest_payload_sha256", None)
    actual_payload_hash = stable_json_sha256(payload_without_hash)
    if recorded_payload_hash != actual_payload_hash:
        raise WhiteDefenseEvaluationError(
            "manifest payload SHA256 mismatch: "
            f"expected {recorded_payload_hash}, actual {actual_payload_hash}"
        )
    return manifest, {
        "manifest_sha256": actual_manifest_sha256,
        "manifest_payload_sha256": actual_payload_hash,
        "sidecar": str(sidecar),
    }


def _artifact_path(raw_path: object, *, manifest_path: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise WhiteDefenseEvaluationError("manifest artifacts.eval.path must be a path")
    path = Path(raw_path)
    if path.is_absolute():
        # Never probe a manifest-controlled absolute/UNC path: on Windows that
        # can trigger an SMB lookup before any hash is checked.  Legacy bundles
        # relocate by basename beside the explicitly selected manifest.
        return (manifest_path.parent / path.name).resolve()
    windows_path = PureWindowsPath(raw_path)
    if windows_path.is_absolute():
        # A Windows absolute path is merely a filename on POSIX.  Resolve its
        # basename beside the manifest so the signed bundle is portable.
        return (manifest_path.parent / windows_path.name).resolve()
    return (manifest_path.parent / path).resolve()


def _load_verified_eval_archive(
    eval_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[WhiteDefenseDataset, dict[str, Any]]:
    eval_path = eval_path.resolve()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get(EVAL_SPLIT), Mapping):
        raise WhiteDefenseEvaluationError("manifest has no artifacts.eval record")
    artifact = artifacts[EVAL_SPLIT]
    assert isinstance(artifact, Mapping)
    declared_path = _artifact_path(artifact.get("path"), manifest_path=manifest_path)
    if not _same_path(eval_path, declared_path):
        raise WhiteDefenseEvaluationError(
            "selected NPZ is not manifest artifacts.eval: "
            f"selected {eval_path}, declared {declared_path}"
        )
    train_artifact = artifacts.get("train")
    if isinstance(train_artifact, Mapping) and train_artifact.get("path") is not None:
        declared_train = _artifact_path(train_artifact.get("path"), manifest_path=manifest_path)
        if _same_path(eval_path, declared_train):
            raise WhiteDefenseEvaluationError("manifest aliases train and eval artifacts")
    prohibition = artifact.get("training_prohibition")
    if prohibition != TRAINING_PROHIBITION:
        raise WhiteDefenseEvaluationError(
            "manifest artifacts.eval has an invalid training_prohibition declaration"
        )
    if not eval_path.is_file():
        raise WhiteDefenseEvaluationError(f"eval NPZ does not exist: {eval_path}")

    expected_bytes = _strict_count(artifact.get("bytes"), label="artifacts.eval.bytes", positive=True)
    actual_bytes = eval_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise WhiteDefenseEvaluationError(
            f"eval NPZ size mismatch: expected {expected_bytes}, actual {actual_bytes}"
        )
    expected_hash = _normalized_sha256(
        artifact.get("sha256"), label="artifacts.eval.sha256"
    )
    actual_hash = sha256_file(eval_path)
    if actual_hash != expected_hash:
        raise WhiteDefenseEvaluationError(
            f"eval NPZ SHA256 mismatch: expected {expected_hash}, actual {actual_hash}"
        )

    required_fields = {
        field.name for field in fields(WhiteDefenseDataset) if field.name != "summary"
    }
    try:
        with np.load(eval_path, allow_pickle=False) as archive:
            actual_fields = set(archive.files)
            if actual_fields != required_fields:
                missing = sorted(required_fields - actual_fields)
                extra = sorted(actual_fields - required_fields)
                raise WhiteDefenseEvaluationError(
                    f"eval NPZ schema mismatch; missing={missing}, extra={extra}"
                )
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, WhiteDefenseEvaluationError):
            raise
        raise WhiteDefenseEvaluationError(f"could not load eval NPZ: {error}") from error

    records = len(arrays["states"])
    if records <= 0:
        raise WhiteDefenseEvaluationError("eval NPZ contains zero samples")
    declared_records = _strict_count(
        artifact.get("records"), label="artifacts.eval.records", positive=True
    )
    if records != declared_records:
        raise WhiteDefenseEvaluationError(
            f"eval record count mismatch: expected {declared_records}, actual {records}"
        )
    split_section = manifest.get("split")
    exported_records = (
        split_section.get("exported_records") if isinstance(split_section, Mapping) else None
    )
    if not isinstance(exported_records, Mapping):
        raise WhiteDefenseEvaluationError("manifest lacks split.exported_records")
    if _strict_count(
        exported_records.get(EVAL_SPLIT),
        label="split.exported_records.eval",
        positive=True,
    ) != records:
        raise WhiteDefenseEvaluationError("split.exported_records.eval disagrees with NPZ")
    validation = manifest.get("validation")
    if not isinstance(validation, Mapping):
        raise WhiteDefenseEvaluationError("manifest lacks validation records")
    if _strict_count(
        validation.get("eval_records"), label="validation.eval_records", positive=True
    ) != records:
        raise WhiteDefenseEvaluationError("validation.eval_records disagrees with NPZ")

    split_values = np.asarray(arrays["split"]).astype(str)
    if split_values.shape != (records,) or not np.all(split_values == EVAL_SPLIT):
        raise WhiteDefenseEvaluationError("selected NPZ is not a pure eval split")
    try:
        dataset = WhiteDefenseDataset(summary=dict(manifest), **arrays)
        deep_validation = validate_dataset(dataset)
    except (TypeError, ValueError) as error:
        raise WhiteDefenseEvaluationError(f"eval NPZ failed deep validation: {error}") from error
    if int(deep_validation.get("eval_records", -1)) != records:
        raise WhiteDefenseEvaluationError("deep validation did not confirm every record as eval")
    return dataset, {
        "path": str(eval_path),
        "bytes": actual_bytes,
        "sha256": actual_hash,
        "records": records,
        "training_prohibition": prohibition,
        "deep_validation": deep_validation,
    }


def _load_checkpoint_and_model(
    checkpoint_path: Path,
    *,
    requested_model_key: str,
    device: torch.device,
) -> tuple[PolicyValueNet, dict[str, Any]]:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise WhiteDefenseEvaluationError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint_data = checkpoint_path.read_bytes()
    checkpoint_hash = sha256_bytes(checkpoint_data)
    try:
        checkpoint = torch.load(
            io.BytesIO(checkpoint_data), map_location="cpu", weights_only=True
        )
    except Exception as error:  # torch emits several deserialization exception types.
        raise WhiteDefenseEvaluationError(f"could not load checkpoint: {error}") from error
    if not isinstance(checkpoint, Mapping):
        raise WhiteDefenseEvaluationError("checkpoint root must be a mapping")

    available = [key for key in MODEL_KEYS if key in checkpoint]
    raw_v3_stage = checkpoint.get("v3_stage")
    if raw_v3_stage is not None and not isinstance(raw_v3_stage, str):
        raise WhiteDefenseEvaluationError("checkpoint v3_stage must be a string when present")
    if requested_model_key == "auto":
        if raw_v3_stage is None:
            # Legacy/V2 approved checkpoints are what the desktop player
            # consumes, and that player loads best_model.  train_model can be
            # an unapproved training state, so it is only a fallback.
            auto_priority = ("best_model", "train_model")
            selection_reason = "legacy approved checkpoint: match playable best_model semantics"
        else:
            auto_priority = V3_AUTO_PRIORITIES.get(raw_v3_stage)
            if auto_priority is None:
                raise WhiteDefenseEvaluationError(
                    f"unsupported v3_stage {raw_v3_stage!r} for automatic model selection; "
                    "pass --model-key explicitly"
                )
            selection_reason = f"stage-aware automatic selection for v3_stage={raw_v3_stage}"
        selected_key = next((key for key in auto_priority if key in checkpoint), None)
        if selected_key is None:
            raise WhiteDefenseEvaluationError(
                "checkpoint has none of the model keys allowed by its automatic stage policy: "
                + ", ".join(auto_priority)
            )
    else:
        auto_priority = (
            ("best_model", "train_model")
            if raw_v3_stage is None
            else V3_AUTO_PRIORITIES.get(raw_v3_stage, ())
        )
        selection_reason = "explicit model-key override"
        if requested_model_key not in SUPPORTED_MODEL_KEYS:
            raise WhiteDefenseEvaluationError(
                f"unsupported model key {requested_model_key!r}"
            )
        if requested_model_key not in checkpoint:
            raise WhiteDefenseEvaluationError(
                f"checkpoint does not contain requested model key {requested_model_key!r}"
            )
        selected_key = requested_model_key

    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, Mapping):
        raise WhiteDefenseEvaluationError("checkpoint has no config mapping")
    try:
        board_size = int(raw_config["board_size"])
        win_length = int(raw_config["win_length"])
        channels = int(raw_config["channels"])
        residual_blocks = int(raw_config["residual_blocks"])
    except (KeyError, TypeError, ValueError) as error:
        raise WhiteDefenseEvaluationError("checkpoint config lacks a valid model architecture") from error
    if board_size != BOARD_SIZE or win_length != 5:
        raise WhiteDefenseEvaluationError(
            f"checkpoint rules are {board_size}x{board_size} win-{win_length}; expected 19x19 win-5"
        )
    if channels <= 0 or residual_blocks < 0:
        raise WhiteDefenseEvaluationError("checkpoint model dimensions must be positive")
    state = checkpoint.get(selected_key)
    if not isinstance(state, Mapping) or not state:
        raise WhiteDefenseEvaluationError(f"checkpoint {selected_key} is not a model state mapping")
    if any(not isinstance(name, str) or not isinstance(tensor, torch.Tensor) for name, tensor in state.items()):
        raise WhiteDefenseEvaluationError(f"checkpoint {selected_key} contains non-tensor state")
    try:
        model = PolicyValueNet(board_size, channels, residual_blocks)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
    except (RuntimeError, TypeError, ValueError) as error:
        raise WhiteDefenseEvaluationError(
            f"checkpoint {selected_key} is incompatible with its config: {error}"
        ) from error

    warning = None
    if (
        selected_key == "best_model"
        and requested_model_key == "auto"
        and raw_v3_stage == "selfplay"
    ):
        warning = (
            "self-play auto selection fell back to best_model because candidate_model/"
            "train_model were absent; V3 self-play best_model is the carried old champion"
        )
    elif (
        selected_key == "train_model"
        and requested_model_key == "auto"
        and raw_v3_stage is None
    ):
        warning = (
            "legacy auto selection fell back to train_model because playable best_model "
            "was absent"
        )
    return model, {
        "path": str(checkpoint_path),
        "bytes": len(checkpoint_data),
        "sha256": checkpoint_hash,
        "requested_model_key": requested_model_key,
        "checkpoint_model_key": selected_key,
        "auto_priority": list(auto_priority),
        "available_model_keys": available,
        "selection_reason": selection_reason,
        "selection_warning": warning,
        "iteration": checkpoint.get("iteration"),
        "format_version": checkpoint.get("format_version"),
        "v3_stage": raw_v3_stage,
        "stage": checkpoint.get("stage"),
        "architecture": {
            "board_size": board_size,
            "win_length": win_length,
            "channels": channels,
            "residual_blocks": residual_blocks,
        },
    }


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as error:
        raise WhiteDefenseEvaluationError(f"invalid device {requested!r}: {error}") from error
    if device.type not in {"cpu", "cuda"}:
        raise WhiteDefenseEvaluationError("device must be cpu, cuda, cuda:N, or auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise WhiteDefenseEvaluationError("CUDA was requested but is unavailable")
    return device


def _policy_metrics(
    model: PolicyValueNet,
    dataset: WhiteDefenseDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    records = len(dataset.states)
    logits_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, records, batch_size):
            states = torch.from_numpy(dataset.states[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            logits, _values = model(states)
            logits_batches.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))
    logits = np.concatenate(logits_batches, axis=0)
    if logits.shape != (records, ACTION_COUNT):
        raise WhiteDefenseEvaluationError(
            f"network returned policy shape {logits.shape}; expected {(records, ACTION_COUNT)}"
        )
    if not np.all(np.isfinite(logits)):
        raise WhiteDefenseEvaluationError("network returned non-finite policy logits")

    candidate = dataset.candidate_mask.astype(bool)
    safe = dataset.safe_mask.astype(bool)
    unsafe = candidate & ~safe
    candidate_counts = candidate.sum(axis=1)
    if np.any(candidate_counts <= 0):
        raise WhiteDefenseEvaluationError("a record has an empty candidate_mask")
    candidate_logits = np.where(candidate, logits, -np.inf)
    candidate_top1 = np.argmax(candidate_logits, axis=1)
    global_top1 = np.argmax(logits, axis=1)
    rows = np.arange(records)

    maxima = candidate_logits[rows, candidate_top1]
    exponentials = np.zeros_like(logits, dtype=np.float64)
    exponentials[candidate] = np.exp(
        (logits - maxima[:, None])[candidate]
    )
    denominators = exponentials.sum(axis=1)
    if np.any(~np.isfinite(denominators)) or np.any(denominators <= 0):
        raise WhiteDefenseEvaluationError("candidate policy normalization failed")
    probabilities = exponentials / denominators[:, None]
    safe_mass_per_record = (probabilities * safe).sum(axis=1)
    unsafe_mass_per_record = (probabilities * unsafe).sum(axis=1)
    mass_error = np.abs(safe_mass_per_record + unsafe_mass_per_record - 1.0)
    if np.any(mass_error > 1e-10):
        raise WhiteDefenseEvaluationError("safe and unsafe candidate mass do not sum to one")

    top1_safe = safe[rows, candidate_top1]
    global_in_candidate = candidate[rows, global_top1]
    return {
        "records": records,
        "top1_in_safe_set": float(top1_safe.mean()),
        "top1_in_safe_set_count": int(top1_safe.sum()),
        "safe_probability_mass": float(safe_mass_per_record.mean()),
        "unsafe_mass": float(unsafe_mass_per_record.mean()),
        "global_top1_in_candidate": float(global_in_candidate.mean()),
        "global_top1_in_candidate_count": int(global_in_candidate.sum()),
        "candidate_probability_normalization": "softmax renormalized within candidate_mask per record",
        "unsafe_definition": "candidate_mask AND NOT safe_mask; candidate-external actions are excluded",
        "per_record_mass_identity_max_abs_error": float(mass_error.max()),
        "safe_probability_mass_min": float(safe_mass_per_record.min()),
        "safe_probability_mass_max": float(safe_mass_per_record.max()),
        "unsafe_mass_min": float(unsafe_mass_per_record.min()),
        "unsafe_mass_max": float(unsafe_mass_per_record.max()),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{os.urandom(6).hex()}"
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_white_defense(
    checkpoint_path: Path,
    eval_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    model_key: str = "auto",
    device: str = "cpu",
    batch_size: int = 128,
) -> dict[str, Any]:
    """Verify all inputs, evaluate raw policy logits, and atomically write JSON."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise WhiteDefenseEvaluationError("batch_size must be a positive integer")
    checkpoint_path = checkpoint_path.resolve()
    eval_path = eval_path.resolve()
    manifest_path = manifest_path.resolve()
    output_path = output_path.resolve()
    protected = {
        checkpoint_path,
        eval_path,
        manifest_path,
        manifest_path.with_suffix(manifest_path.suffix + ".sha256"),
    }
    if output_path in protected:
        raise WhiteDefenseEvaluationError("output path must not overwrite an input")

    manifest, manifest_integrity = _read_verified_manifest(manifest_path)
    dataset, artifact_audit = _load_verified_eval_archive(
        eval_path, manifest_path, manifest
    )
    resolved_device = _resolve_device(device)
    model, checkpoint_audit = _load_checkpoint_and_model(
        checkpoint_path,
        requested_model_key=model_key,
        device=resolved_device,
    )
    metrics = _policy_metrics(
        model,
        dataset,
        device=resolved_device,
        batch_size=batch_size,
    )
    report: dict[str, Any] = {
        "format": "gomoku_white_defense_evaluation",
        "schema_version": REPORT_SCHEMA_VERSION,
        "checkpoint_model_key": checkpoint_audit["checkpoint_model_key"],
        "checkpoint": checkpoint_audit,
        "eval_dataset": artifact_audit,
        "manifest": {
            "path": str(manifest_path),
            **manifest_integrity,
            "schema_version": manifest.get("schema_version"),
            "source": manifest.get("source"),
        },
        "device": {
            "requested": device,
            "resolved": str(resolved_device),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "metrics": metrics,
        "interpretation": {
            "scope": "raw network policy on held-out white-to-move positions",
            "safe_claim": "bounded safe set supplied by the verified eval archive",
            "candidate_external_policy": (
                "excluded from safe/unsafe mass; measured only by global_top1_in_candidate"
            ),
        },
    }
    # Reject accidental NaN/Infinity before publication.
    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise WhiteDefenseEvaluationError(f"evaluation report is not finite JSON: {error}") from error
    _atomic_write_json(output_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-npz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-key",
        default="auto",
        choices=("auto", *MODEL_KEYS),
        help="auto uses stage-aware semantics; explicit keys override that policy",
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N, or auto")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = evaluate_white_defense(
            args.checkpoint,
            args.eval_npz,
            args.manifest,
            args.output,
            model_key=args.model_key,
            device=args.device,
            batch_size=args.batch_size,
        )
    except WhiteDefenseEvaluationError as error:
        raise SystemExit(f"white-defense evaluation failed: {error}") from error
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "checkpoint_model_key": report["checkpoint"]["checkpoint_model_key"],
                "metrics": report["metrics"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
