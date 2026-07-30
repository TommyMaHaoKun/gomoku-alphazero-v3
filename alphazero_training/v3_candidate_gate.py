#!/usr/bin/env python3
"""Freeze and promote V3 Gomoku candidates without touching training state.

The training checkpoint deliberately keeps the externally approved model in
``best_model`` while the network being trained lives in ``candidate_model``.
Legacy evaluators and the desktop player load ``best_model``.  This utility
therefore provides two explicit, one-way operations:

``freeze``
    Copy ``candidate_model`` from a format-v3 self-play checkpoint, or
    ``train_model`` from a format-v3 ``tactical_expert_warmstart`` checkpoint,
    into an immutable evaluation checkpoint's ``best_model`` key.  Warmstart
    freezing additionally requires the expected parent-checkpoint SHA256; it
    never falls back to the warmstart's ``best_model``.  The result is
    conspicuously marked ``not_approved`` and is suitable only for independent
    evaluators.

``promote``
    Verify that independent legal-tactics and paired-DDQK JSON reports are
    cryptographically bound to that exact evaluation checkpoint, apply the
    configured gates, and create a new permanent champion checkpoint.  It
    never edits either the training checkpoint or the candidate snapshot.

Both commands require an expected SHA256 supplied out-of-band.  Checkpoints
are read from the exact bytes that were hashed, and output publication is
atomic and no-clobber.

For example, freeze a V3F supervised result with both links in its provenance
chain supplied independently::

    python -m alphazero_training.v3_candidate_gate freeze \
      --source run_v3f/latest.pt \
      --expected-source-sha256 <sha256-of-run_v3f/latest.pt> \
      --expected-parent-sha256 <sha256-of-init-checkpoint> \
      --output candidates/v3f_candidate_eval.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch

from .ddqk_replay_export import (
    rebuild_benchmark_openings,
    validate_benchmark_game_history,
)
from .train_alphazero import Config, PolicyValueNet


FORMAT_VERSION = 3
SELFPLAY_STAGE = "selfplay"
WARMSTART_STAGE = "tactical_expert_warmstart"
CANDIDATE_STAGE = "candidate_eval"
CHAMPION_STAGE = "external_champion"
DEVELOPMENT_STAGE = "development_screened"
DEVELOPMENT_MODE = "development"
FINAL_CERTIFICATION_MODE = "final-certification"
FINAL_MIN_PAIRS = 600
FINAL_MIN_SCORE = 0.995
FINAL_MIN_COLOR_SCORE = 0.99
FINAL_MIN_CI_LOW = 0.95
FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95 = 0.995
CONFIDENCE_ALPHA = 0.05
EVALUATION_CODE_FILES = {
    "play_agent.py",
    "train_alphazero.py",
    "v3_search.py",
    "tactical_solver.py",
    "ddqk_adapter.py",
    "benchmark_ddqk.py",
}
LEGACY_EVALUATION_CODE_FILES = {
    "play_agent.py",
    "v3_search.py",
    "tactical_solver.py",
    "benchmark_ddqk.py",
}
DDQK_DECISION_ASSET_FILES = {
    "dll.so",
    "guess_data.txt",
    "black_calculated_value_19.txt",
    "white_calculated_value_19.txt",
}


class GateError(ValueError):
    """A candidate or evaluation failed an integrity or quality gate."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _normalized_sha256(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise GateError(f"{label} must be a 64-character SHA256 digest")
    return digest


def _read_verified_checkpoint(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    if not path.is_file():
        raise GateError(f"{label} does not exist: {path}")
    expected = _normalized_sha256(expected_sha256, label=f"expected {label} SHA256")
    data = path.read_bytes()
    actual = sha256_bytes(data)
    if actual != expected:
        raise GateError(
            f"{label} SHA256 mismatch: expected {expected}, actual {actual}"
        )
    # Loading the exact byte string that was hashed avoids a path mutation
    # between integrity verification and deserialization.
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise GateError(f"{label} must contain a checkpoint mapping")
    return payload, actual


def _read_json_with_sha(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    if not path.is_file():
        raise GateError(f"{label} does not exist: {path}")
    data = path.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise GateError(f"{label} root must be a JSON object")
    return payload, sha256_bytes(data)


def _state_dict(value: object, *, label: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise GateError(f"{label} must be a non-empty model state mapping")
    state: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise GateError(f"{label} contains a non-tensor parameter")
        state[key] = tensor
    return state


def _states_equal(left: object, right: object, *, left_label: str, right_label: str) -> bool:
    lhs = _state_dict(left, label=left_label)
    rhs = _state_dict(right, label=right_label)
    return set(lhs) == set(rhs) and all(
        torch.equal(lhs[name].detach().cpu(), rhs[name].detach().cpu())
        for name in lhs
    )


def _cpu_state_copy(value: object, *, label: str) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in _state_dict(value, label=label).items()
    }


def _validated_config_and_state(
    checkpoint: Mapping[str, Any],
    state_key: str,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        raise GateError(f"{label} has no valid config mapping")
    try:
        config = Config(**raw_config)
        model = PolicyValueNet(config.board_size, config.channels, config.residual_blocks)
        state = _cpu_state_copy(checkpoint.get(state_key), label=f"{label} {state_key}")
        model.load_state_dict(state, strict=True)
    except (TypeError, RuntimeError, ValueError) as error:
        raise GateError(f"{label} {state_key} is incompatible with config: {error}") from error
    return asdict(config), state


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_no_clobber_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    """Publish a complete torch file atomically and refuse any overwrite."""

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise GateError(f"refusing to overwrite immutable output: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        # Windows requires a writable descriptor for FlushFileBuffers, which
        # backs ``os.fsync`` there.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        # A hard link is an atomic, no-clobber publication on the same volume.
        # It prevents a race from replacing an existing permanent champion.
        os.link(temporary, destination)
    except FileExistsError as error:
        raise GateError(f"refusing to overwrite immutable output: {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_distinct_output(output: Path, forbidden: Sequence[Path], *, champion: bool) -> None:
    output = output.resolve()
    for path in forbidden:
        if output == path.resolve():
            raise GateError(f"output must not overwrite input checkpoint: {output}")
    if champion and output.name.lower() == "latest.pt":
        raise GateError("champion output must use a permanent name, not latest.pt")


def freeze_candidate(
    source_path: Path,
    output_path: Path,
    *,
    expected_source_sha256: str,
    expected_parent_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze a self-play or supervised candidate as an unapproved eval file.

    Existing self-play callers remain compatible: they need only provide the
    source SHA256, and the candidate comes from ``candidate_model`` exactly as
    before.  A supervised warmstart has no ``candidate_model``; its sole
    candidate source is ``train_model`` and its parent SHA256 must also be
    supplied out-of-band.
    """

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    _ensure_distinct_output(output_path, [source_path], champion=False)
    source, source_sha = _read_verified_checkpoint(
        source_path, expected_source_sha256, label="source checkpoint"
    )
    if source.get("format_version") != FORMAT_VERSION:
        raise GateError("source must be a format-v3 training checkpoint")
    source_stage = source.get("v3_stage")
    prior_approved_sha: str | None
    if source_stage == SELFPLAY_STAGE:
        if "candidate_model" not in source:
            raise GateError("self-play checkpoint is missing required candidate_model")
        if "train_model" not in source:
            raise GateError("self-play checkpoint is missing required train_model")
        if not _states_equal(
            source["candidate_model"],
            source["train_model"],
            left_label="candidate_model",
            right_label="train_model",
        ):
            raise GateError("candidate_model does not match current train_model")
        if "approved_model" not in source or "best_model" not in source:
            raise GateError("self-play checkpoint has no separately approved champion")
        if not _states_equal(
            source["approved_model"],
            source["best_model"],
            left_label="approved_model",
            right_label="best_model",
        ):
            raise GateError("source best_model is not the carried approved_model")
        config, candidate = _validated_config_and_state(
            source, "candidate_model", label="self-play checkpoint"
        )
        source_model_key = "candidate_model"
        parent_sha = str(source.get("parent_checkpoint_sha256", ""))
        prior_approved_sha = _normalized_sha256(
            source.get("approved_checkpoint_sha256", ""),
            label="source approved_checkpoint_sha256",
        )
    elif source_stage == WARMSTART_STAGE:
        if "train_model" not in source:
            raise GateError("warmstart checkpoint is missing required train_model")
        if expected_parent_sha256 is None:
            raise GateError(
                "warmstart freeze requires expected_parent_sha256 supplied out-of-band"
            )
        recorded_parent_sha = _normalized_sha256(
            source.get("parent_checkpoint_sha256", ""),
            label="warmstart parent_checkpoint_sha256",
        )
        expected_parent_sha = _normalized_sha256(
            expected_parent_sha256,
            label="expected parent checkpoint SHA256",
        )
        if recorded_parent_sha != expected_parent_sha:
            raise GateError(
                "warmstart parent checkpoint SHA256 mismatch: "
                f"expected {expected_parent_sha}, recorded {recorded_parent_sha}"
            )
        # Deliberately do not inspect or fall back to best_model.  train_model
        # is the newly trained V3F network; best_model may be a carried parent
        # in other checkpoint writers.
        config, candidate = _validated_config_and_state(
            source, "train_model", label="warmstart checkpoint"
        )
        if any(
            not bool(torch.isfinite(tensor).all().item())
            for tensor in candidate.values()
        ):
            raise GateError("warmstart checkpoint train_model contains non-finite values")
        warmstart_model_spec = source.get("model_spec")
        if not isinstance(warmstart_model_spec, dict):
            raise GateError("warmstart checkpoint has no valid model_spec mapping")
        expected_model_spec = {
            "board_size": config["board_size"],
            "channels": config["channels"],
            "residual_blocks": config["residual_blocks"],
            "input_planes": 4,
        }
        if any(
            warmstart_model_spec.get(name) != value
            for name, value in expected_model_spec.items()
        ):
            raise GateError("warmstart checkpoint model_spec does not match config")
        source_model_key = "train_model"
        parent_sha = recorded_parent_sha
        prior_approved_sha = None
    else:
        raise GateError(
            "source v3_stage must be 'selfplay' or 'tactical_expert_warmstart'"
        )

    iteration_value = source.get("iteration", -1)
    global_step_value = source.get("global_step", 0)
    if (
        isinstance(iteration_value, bool)
        or not isinstance(iteration_value, int)
        or iteration_value < 0
    ):
        raise GateError("source checkpoint has an invalid iteration")
    if (
        isinstance(global_step_value, bool)
        or not isinstance(global_step_value, int)
        or global_step_value < 0
    ):
        raise GateError("source checkpoint has an invalid global_step")
    iteration = iteration_value

    raw_model_spec = source.get("model_spec", {})
    if not isinstance(raw_model_spec, dict):
        raise GateError("source checkpoint has no valid model_spec mapping")
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "v3_stage": CANDIDATE_STAGE,
        "iteration": iteration,
        "global_step": global_step_value,
        "config": config,
        "model_spec": dict(raw_model_spec),
        # Evaluation tools and play_agent intentionally load best_model.
        # Keep independent mappings so a corrupted/mis-keyed checkpoint can
        # be detected by promotion instead of silently mutating aliases.
        "best_model": _cpu_state_copy(candidate, label="frozen best_model"),
        "candidate_model": _cpu_state_copy(candidate, label="frozen candidate_model"),
        "train_model": _cpu_state_copy(candidate, label="frozen train_model"),
        "approval_status": "not_approved",
        "is_approved": False,
        "external_evaluation": {
            "status": "pending_independent_evaluation",
            "candidate_status": "not_approved",
        },
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint_stage": source_stage,
        "source_model_key": source_model_key,
        "source_parent_checkpoint_sha256": parent_sha,
        "prior_approved_checkpoint_sha256": prior_approved_sha,
        "frozen_at_utc": _utc_now(),
    }
    _atomic_no_clobber_torch_save(payload, output_path)
    return {
        "status": "frozen_not_approved",
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "source_sha256": source_sha,
        "source_stage": source_stage,
        "source_model_key": source_model_key,
        "source_parent_sha256": parent_sha,
        "iteration": iteration,
    }


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise GateError(f"{label} must be finite")
    return result


def _probability(value: object, *, label: str) -> float:
    result = _number(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise GateError(f"{label} must be in [0, 1]")
    return result


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateError(f"{label} must be an integer")
    return int(value)


def _report_checkpoint_sha(report: Mapping[str, Any], *, label: str) -> str:
    direct = report.get("checkpoint_sha256")
    signature = report.get("signature")
    nested = signature.get("checkpoint_sha256") if isinstance(signature, Mapping) else None
    value = direct if direct is not None else nested
    if value is None:
        raise GateError(
            f"{label} has no checkpoint_sha256; rerun evaluation with SHA provenance"
        )
    return _normalized_sha256(value, label=f"{label} checkpoint_sha256")


def _validate_tactical_report(
    report: Mapping[str, Any],
    *,
    candidate_sha256: str,
    min_top1: float,
    min_samples: int,
) -> dict[str, Any]:
    if _report_checkpoint_sha(report, label="tactical report") != candidate_sha256:
        raise GateError("tactical report was not produced from this candidate checkpoint")
    if report.get("split") != "eval":
        raise GateError("tactical report must use the held-out eval split")
    if report.get("checkpoint_model_key") != "best_model":
        raise GateError("tactical report must evaluate the frozen best_model key")
    dataset_sha = _normalized_sha256(
        report.get("dataset_sha256", ""), label="tactical dataset_sha256"
    )
    samples = _integer(report.get("samples"), label="tactical samples")
    if samples < min_samples:
        raise GateError(f"tactical samples {samples} is below required {min_samples}")
    raw = report.get("raw_network")
    if not isinstance(raw, Mapping):
        raise GateError("tactical report has no raw_network metrics")
    top1 = _probability(raw.get("top1"), label="raw tactical top1")
    if top1 < min_top1:
        raise GateError(f"raw tactical top1 {top1:.6f} is below required {min_top1:.6f}")
    family_macro = _probability(
        raw.get("family_macro_top1"), label="raw tactical family_macro_top1"
    )
    if family_macro < min_top1:
        raise GateError(
            "raw tactical family-macro top1 "
            f"{family_macro:.6f} is below required {min_top1:.6f}"
        )
    return {
        "samples": samples,
        "raw_top1": top1,
        "family_macro_top1": family_macro,
        "dataset_sha256": dataset_sha,
    }


def _bounded_mean_one_sided_lower95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    radius = math.sqrt(math.log(1.0 / CONFIDENCE_ALPHA) / (2.0 * len(values)))
    return max(0.0, float(mean - radius))


def _binomial_upper_tail_probability(
    successes: int,
    trials: int,
    probability: float,
) -> float:
    """Stable P[X >= successes] for an exact binomial gate recheck."""

    if successes <= 0:
        return 1.0
    if successes > trials:
        return 0.0
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    log_probability = math.log(probability)
    log_failure = math.log1p(-probability)
    terms = [
        math.lgamma(trials + 1)
        - math.lgamma(outcome + 1)
        - math.lgamma(trials - outcome + 1)
        + outcome * log_probability
        + (trials - outcome) * log_failure
        for outcome in range(successes, trials + 1)
    ]
    largest = max(terms)
    if largest < math.log(sys.float_info.min):
        return 0.0
    return min(
        1.0,
        float(math.exp(largest) * sum(math.exp(term - largest) for term in terms)),
    )


def _exact_binomial_one_sided_lower95(successes: int, trials: int) -> float:
    """Recompute the exact one-sided Clopper-Pearson lower bound."""

    if not 0 <= successes <= trials:
        raise GateError("DDQK pair-sweep successes must be between zero and trials")
    if trials == 0 or successes == 0:
        return 0.0
    if successes == trials:
        return float(CONFIDENCE_ALPHA ** (1.0 / trials))
    low = 0.0
    high = successes / trials
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_upper_tail_probability(successes, trials, midpoint) < CONFIDENCE_ALPHA:
            low = midpoint
        else:
            high = midpoint
    return float(low)


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _validate_ddqk_signature(
    report: Mapping[str, Any],
    *,
    candidate_sha256: str,
    certification_mode: str,
    expected_ddqk_source_sha256: str | None,
    expected_ddqk_dll_sha256: str | None,
    expected_ddqk_assets_bundle_sha256: str | None,
    expected_ddqk_depth: int | None,
    expected_simulations: int | None,
    expected_opening_plies: int | None,
    expected_max_moves: int | None,
    expected_seed: int | None,
    expected_evaluation_bundle_sha256: str | None,
) -> dict[str, Any]:
    if report.get("format_version") != 3:
        raise GateError("DDQK report must use provenance-complete format_version 3")
    signature = report.get("signature")
    if not isinstance(signature, Mapping):
        raise GateError("DDQK report has no immutable signature")
    if _report_checkpoint_sha(report, label="DDQK report") != candidate_sha256:
        raise GateError("DDQK report was not produced from this candidate checkpoint")

    source = signature.get("ddqk_source")
    if not isinstance(source, str) or not source.strip():
        raise GateError("DDQK signature has no ddqk_source path")
    source_sha = _normalized_sha256(
        signature.get("ddqk_source_sha256", ""), label="DDQK source SHA256"
    )
    dll_sha = _normalized_sha256(
        signature.get("ddqk_dll_sha256", ""), label="DDQK DLL SHA256"
    )
    depth = _integer(signature.get("ddqk_depth"), label="DDQK depth")
    simulations = _integer(signature.get("simulations"), label="DDQK simulations")
    opening_plies = _integer(signature.get("opening_plies"), label="DDQK opening_plies")
    max_moves = _integer(signature.get("max_moves"), label="DDQK max_moves")
    seed = _integer(signature.get("seed"), label="DDQK seed")
    pairs = _integer(signature.get("pairs"), label="DDQK signature pairs")
    signature_mode = signature.get("certification_mode")
    if signature_mode != certification_mode:
        raise GateError(
            "DDQK certification_mode drift: "
            f"expected {certification_mode!r}, report has {signature_mode!r}"
        )

    code = signature.get("evaluation_code")
    if not isinstance(code, Mapping):
        raise GateError("DDQK signature has no evaluation_code bundle")
    raw_file_hashes = code.get("files")
    if not isinstance(raw_file_hashes, Mapping):
        raise GateError("DDQK evaluation_code has no file hashes")
    code_file_names = set(raw_file_hashes)
    allowed_code_sets = (EVALUATION_CODE_FILES, LEGACY_EVALUATION_CODE_FILES)
    if code_file_names not in allowed_code_sets or (
        certification_mode == FINAL_CERTIFICATION_MODE
        and code_file_names != EVALUATION_CODE_FILES
    ):
        raise GateError(
            "DDQK evaluation_code must hash exactly the six decision files: "
            "play_agent.py, train_alphazero.py, v3_search.py, tactical_solver.py, "
            "ddqk_adapter.py, and benchmark_ddqk.py"
        )
    file_hashes = {
        str(name): _normalized_sha256(value, label=f"DDQK code hash {name}")
        for name, value in raw_file_hashes.items()
    }
    bundle_sha = _normalized_sha256(
        code.get("bundle_sha256", ""), label="DDQK evaluation bundle SHA256"
    )
    if bundle_sha != _stable_json_sha256(file_hashes):
        raise GateError("DDQK evaluation bundle hash does not match its file hashes")

    raw_assets = signature.get("ddqk_assets")
    require_current_provenance = (
        certification_mode == FINAL_CERTIFICATION_MODE
        or code_file_names == EVALUATION_CODE_FILES
    )
    asset_file_hashes: dict[str, str] | None = None
    asset_bundle_sha: str | None = None
    if raw_assets is None:
        if require_current_provenance:
            raise GateError("DDQK signature has no complete ddqk_assets bundle")
    else:
        if not isinstance(raw_assets, Mapping):
            raise GateError("DDQK ddqk_assets must be an object")
        raw_asset_hashes = raw_assets.get("files")
        if (
            not isinstance(raw_asset_hashes, Mapping)
            or set(raw_asset_hashes) != DDQK_DECISION_ASSET_FILES
        ):
            raise GateError(
                "DDQK assets must hash exactly dll.so, guess_data.txt, "
                "black_calculated_value_19.txt, and white_calculated_value_19.txt"
            )
        asset_file_hashes = {
            str(name): _normalized_sha256(value, label=f"DDQK asset hash {name}")
            for name, value in raw_asset_hashes.items()
        }
        asset_bundle_sha = _normalized_sha256(
            raw_assets.get("bundle_sha256", ""),
            label="DDQK asset bundle SHA256",
        )
        if asset_bundle_sha != _stable_json_sha256(asset_file_hashes):
            raise GateError("DDQK asset bundle hash does not match its file hashes")
        if asset_file_hashes["dll.so"] != dll_sha:
            raise GateError("DDQK DLL SHA256 disagrees with the signed asset bundle")

    expected_fields: tuple[tuple[str, object | None, object], ...] = (
        ("ddqk_source_sha256", expected_ddqk_source_sha256, source_sha),
        ("ddqk_dll_sha256", expected_ddqk_dll_sha256, dll_sha),
        (
            "ddqk_assets_bundle_sha256",
            expected_ddqk_assets_bundle_sha256,
            asset_bundle_sha,
        ),
        ("ddqk_depth", expected_ddqk_depth, depth),
        ("simulations", expected_simulations, simulations),
        ("opening_plies", expected_opening_plies, opening_plies),
        ("max_moves", expected_max_moves, max_moves),
        ("seed", expected_seed, seed),
        ("evaluation_bundle_sha256", expected_evaluation_bundle_sha256, bundle_sha),
    )
    if certification_mode == FINAL_CERTIFICATION_MODE:
        missing = [name for name, expected, _ in expected_fields if expected is None]
        if missing:
            raise GateError(
                "final certification requires CLI expectations for: " + ", ".join(missing)
            )
    for name, expected, actual in expected_fields:
        if expected is None:
            continue
        if name.endswith("sha256"):
            expected = _normalized_sha256(str(expected), label=f"expected {name}")
        elif isinstance(actual, int):
            expected = _integer(expected, label=f"expected {name}")
        if expected != actual:
            raise GateError(
                f"DDQK {name} drift: expected {expected!r}, report has {actual!r}"
            )
    return {
        "ddqk_source": source,
        "ddqk_source_sha256": source_sha,
        "ddqk_dll_sha256": dll_sha,
        "ddqk_asset_files": asset_file_hashes,
        "ddqk_assets_bundle_sha256": asset_bundle_sha,
        "ddqk_depth": depth,
        "simulations": simulations,
        "opening_plies": opening_plies,
        "max_moves": max_moves,
        "seed": seed,
        "pairs": pairs,
        "evaluation_code_files": file_hashes,
        "evaluation_bundle_sha256": bundle_sha,
        "current_provenance_schema": code_file_names == EVALUATION_CODE_FILES,
    }


def _validate_ddqk_report(
    report: Mapping[str, Any],
    *,
    candidate_sha256: str,
    min_score: float,
    min_black: float,
    min_white: float,
    min_ci_low: float,
    min_pairs: int,
    certification_mode: str,
    expected_ddqk_source_sha256: str | None = None,
    expected_ddqk_dll_sha256: str | None = None,
    expected_ddqk_assets_bundle_sha256: str | None = None,
    expected_ddqk_depth: int | None = None,
    expected_simulations: int | None = None,
    expected_opening_plies: int | None = None,
    expected_max_moves: int | None = None,
    expected_seed: int | None = None,
    expected_evaluation_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    provenance = _validate_ddqk_signature(
        report,
        candidate_sha256=candidate_sha256,
        certification_mode=certification_mode,
        expected_ddqk_source_sha256=expected_ddqk_source_sha256,
        expected_ddqk_dll_sha256=expected_ddqk_dll_sha256,
        expected_ddqk_assets_bundle_sha256=expected_ddqk_assets_bundle_sha256,
        expected_ddqk_depth=expected_ddqk_depth,
        expected_simulations=expected_simulations,
        expected_opening_plies=expected_opening_plies,
        expected_max_moves=expected_max_moves,
        expected_seed=expected_seed,
        expected_evaluation_bundle_sha256=expected_evaluation_bundle_sha256,
    )
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise GateError("DDQK report has no summary")
    requested_pairs = _integer(
        summary.get("requested_pairs"), label="DDQK requested_pairs"
    )
    if requested_pairs != provenance["pairs"]:
        raise GateError("DDQK summary and signature pair counts differ")
    complete_pairs = _integer(summary.get("complete_pairs"), label="DDQK complete_pairs")
    incomplete_pairs = _integer(
        summary.get("incomplete_pairs"), label="DDQK incomplete_pairs"
    )
    errors = _integer(summary.get("errors"), label="DDQK errors")
    truncated = _integer(summary.get("truncated"), label="DDQK truncated")
    completed_games = _integer(
        summary.get("completed_games"), label="DDQK completed_games"
    )
    scored_games = _integer(summary.get("scored_games"), label="DDQK scored_games")
    if requested_pairs < min_pairs:
        raise GateError(f"DDQK pairs {requested_pairs} is below required {min_pairs}")
    if (
        complete_pairs != requested_pairs
        or incomplete_pairs != 0
        or errors != 0
        or truncated != 0
        or completed_games != 2 * requested_pairs
        or scored_games != completed_games
    ):
        raise GateError("DDQK report is incomplete, truncated, or contains errors")

    openings = report.get("openings")
    games = report.get("games")
    if not isinstance(openings, list) or len(openings) != requested_pairs:
        raise GateError("DDQK opening manifest does not match requested pair count")
    if not isinstance(games, list) or len(games) != 2 * requested_pairs:
        raise GateError("DDQK game records do not contain exactly two games per pair")
    manifest_sha = _normalized_sha256(
        report["signature"].get("opening_manifest_sha256", ""),
        label="DDQK opening manifest SHA256",
    )
    if manifest_sha != _stable_json_sha256(openings):
        raise GateError("DDQK opening manifest hash mismatch")
    try:
        expected_openings = rebuild_benchmark_openings(
            seed=int(provenance["seed"]),
            pairs=requested_pairs,
            opening_plies=int(provenance["opening_plies"]),
        )
    except ValueError as exc:
        raise GateError(f"DDQK opening recipe is invalid: {exc}") from exc
    if openings != expected_openings:
        raise GateError(
            "DDQK opening manifest cannot be reproduced from its signed seed recipe"
        )
    if certification_mode == FINAL_CERTIFICATION_MODE:
        opening_hashes = [_stable_json_sha256(opening) for opening in openings]
        if len(set(opening_hashes)) != requested_pairs:
            raise GateError("final certification requires distinct paired openings")

    records: dict[tuple[int, int], float] = {}
    recorded_order: list[tuple[int, int]] = []
    for index, raw_game in enumerate(games):
        if not isinstance(raw_game, Mapping):
            raise GateError(f"DDQK game {index} is not an object")
        pair_index = _integer(raw_game.get("pair_index"), label=f"DDQK game {index} pair_index")
        color = _integer(raw_game.get("model_color"), label=f"DDQK game {index} model_color")
        if not 0 <= pair_index < requested_pairs or color not in (1, 2):
            raise GateError(f"DDQK game {index} has invalid pair index or model color")
        key = (pair_index, color)
        if key in records:
            raise GateError(f"DDQK report contains duplicate game key {key}")
        recorded_order.append(key)
        if raw_game.get("opening") != openings[pair_index]:
            raise GateError(f"DDQK game {index} opening differs from its manifest")
        if raw_game.get("error") is not None or raw_game.get("termination") not in (
            "win",
            "full_board_draw",
        ):
            raise GateError(f"DDQK game {index} is incomplete or contains an engine error")
        try:
            history = validate_benchmark_game_history(
                raw_game,
                expected_opening=expected_openings[pair_index],
                expected_opening_plies=int(provenance["opening_plies"]),
                max_moves=int(provenance["max_moves"]),
                label=f"DDQK game {index}",
            )
        except ValueError as exc:
            raise GateError(str(exc)) from exc
        raw_derived_result = history["model_result"]
        if raw_derived_result not in (0.0, 0.5, 1.0):
            raise GateError(f"DDQK game {index} result must be win, draw, or loss")
        result = float(raw_derived_result)
        records[key] = result
    expected_keys = {
        (pair_index, color)
        for pair_index in range(requested_pairs)
        for color in (1, 2)
    }
    if set(records) != expected_keys:
        raise GateError("DDQK report is missing one or more color-swapped games")
    expected_order = [
        (pair_index, color)
        for pair_index in range(requested_pairs)
        for color in (1, 2)
    ]
    if recorded_order != expected_order:
        raise GateError("DDQK games are not in signed paired-opening order")

    black_results = [records[(pair_index, 1)] for pair_index in range(requested_pairs)]
    white_results = [records[(pair_index, 2)] for pair_index in range(requested_pairs)]
    pair_scores = [
        (black_results[pair_index] + white_results[pair_index]) / 2.0
        for pair_index in range(requested_pairs)
    ]
    score = sum(pair_scores) / requested_pairs
    black = sum(black_results) / requested_pairs
    white = sum(white_results) / requested_pairs
    for label, reported, recomputed in (
        ("overall score", _probability(summary.get("score"), label="DDQK overall score"), score),
        (
            "black score",
            _probability(summary.get("by_color", {}).get("black", {}).get("score"), label="DDQK black score")
            if isinstance(summary.get("by_color"), Mapping)
            and isinstance(summary.get("by_color", {}).get("black"), Mapping)
            else -1.0,
            black,
        ),
        (
            "white score",
            _probability(summary.get("by_color", {}).get("white", {}).get("score"), label="DDQK white score")
            if isinstance(summary.get("by_color"), Mapping)
            and isinstance(summary.get("by_color", {}).get("white"), Mapping)
            else -1.0,
            white,
        ),
    ):
        if reported < 0.0 or not _same_number(reported, recomputed):
            raise GateError(f"DDQK reported {label} does not match game records")

    colors = summary.get("by_color")
    if not isinstance(colors, Mapping):
        raise GateError("DDQK report has no by_color summary")
    for name in ("black", "white"):
        color_record = colors.get(name)
        if not isinstance(color_record, Mapping):
            raise GateError(f"DDQK report has no {name} summary")
        if _integer(color_record.get("games"), label=f"DDQK {name} games") != requested_pairs:
            raise GateError("DDQK color-swapped pair counts are inconsistent")

    lower = _bounded_mean_one_sided_lower95(pair_scores)
    reported_lower = _probability(
        summary.get("one_sided_95_lower_bound"),
        label="DDQK one-sided 95% lower bound",
    )
    method = summary.get("one_sided_95_lower_bound_method")
    if not isinstance(method, Mapping) or (
        method.get("name") != "hoeffding_bounded_mean"
        or not _same_number(_number(method.get("alpha"), label="DDQK bound alpha"), CONFIDENCE_ALPHA)
        or method.get("independent_unit") != "paired_opening"
        or _integer(method.get("sample_size"), label="DDQK bound sample size") != requested_pairs
    ):
        raise GateError("DDQK one-sided confidence-bound method metadata is invalid")
    if not _same_number(reported_lower, lower):
        raise GateError("DDQK one-sided 95% lower bound does not match game records")

    pair_sweep_successes = sum(
        black_results[index] == 1.0 and white_results[index] == 1.0
        for index in range(requested_pairs)
    )
    pair_sweep_trials = requested_pairs
    observed_pair_sweep_rate = pair_sweep_successes / pair_sweep_trials
    exact_pair_sweep_lower95 = _exact_binomial_one_sided_lower95(
        pair_sweep_successes,
        pair_sweep_trials,
    )
    exact_fields = {
        "pair_sweep_successes",
        "pair_sweep_trials",
        "observed_pair_sweep_rate",
        "exact_pair_sweep_lower95",
        "exact_pair_sweep_lower95_method",
    }
    present_exact_fields = exact_fields.intersection(summary)
    require_exact_fields = bool(
        certification_mode == FINAL_CERTIFICATION_MODE
        or provenance["current_provenance_schema"]
        or present_exact_fields
    )
    if require_exact_fields:
        missing_exact_fields = sorted(exact_fields.difference(summary))
        if missing_exact_fields:
            raise GateError(
                "DDQK exact pair-sweep fields are incomplete: "
                + ", ".join(missing_exact_fields)
            )
        reported_successes = _integer(
            summary.get("pair_sweep_successes"),
            label="DDQK pair-sweep successes",
        )
        reported_trials = _integer(
            summary.get("pair_sweep_trials"),
            label="DDQK pair-sweep trials",
        )
        reported_sweep_rate = _probability(
            summary.get("observed_pair_sweep_rate"),
            label="DDQK observed pair-sweep rate",
        )
        reported_exact_lower = _probability(
            summary.get("exact_pair_sweep_lower95"),
            label="DDQK exact pair-sweep lower bound",
        )
        exact_method = summary.get("exact_pair_sweep_lower95_method")
        if not isinstance(exact_method, Mapping) or (
            exact_method.get("name") != "clopper_pearson_exact_binomial"
            or not _same_number(
                _number(exact_method.get("alpha"), label="DDQK exact-bound alpha"),
                CONFIDENCE_ALPHA,
            )
            or exact_method.get("success_definition")
            != "model_wins_both_color_swapped_games"
            or exact_method.get("independent_unit") != "paired_opening"
            or _integer(
                exact_method.get("successes"),
                label="DDQK exact-bound successes",
            )
            != pair_sweep_successes
            or _integer(
                exact_method.get("trials"),
                label="DDQK exact-bound trials",
            )
            != pair_sweep_trials
        ):
            raise GateError("DDQK exact pair-sweep confidence method metadata is invalid")
        if reported_successes != pair_sweep_successes:
            raise GateError("DDQK pair-sweep successes do not match game records")
        if reported_trials != pair_sweep_trials:
            raise GateError("DDQK pair-sweep trials do not match game records")
        if not _same_number(reported_sweep_rate, observed_pair_sweep_rate):
            raise GateError("DDQK observed pair-sweep rate does not match game records")
        if not _same_number(reported_exact_lower, exact_pair_sweep_lower95):
            raise GateError("DDQK exact pair-sweep lower bound does not match game records")

    if (
        certification_mode == FINAL_CERTIFICATION_MODE
        and exact_pair_sweep_lower95 < FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95
    ):
        raise GateError(
            "DDQK exact pair-sweep gate failed: "
            f"{exact_pair_sweep_lower95:.6f} < "
            f"{FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95:.6f}"
        )

    certification = report.get("certification")
    if not isinstance(certification, Mapping) or certification.get("mode") != certification_mode:
        raise GateError("DDQK report certification metadata is missing or has drifted")
    if certification_mode == FINAL_CERTIFICATION_MODE:
        if certification.get("final_certified") is not True:
            raise GateError("DDQK report did not pass final-certification benchmark requirements")
        if certification.get("status") != "benchmark_final_requirements_passed":
            raise GateError("DDQK final-certification status is invalid")
        requirements = certification.get("requirements")
        if not isinstance(requirements, Mapping) or (
            _integer(
                requirements.get("minimum_independent_paired_openings"),
                label="DDQK final minimum paired openings",
            )
            != FINAL_MIN_PAIRS
            or not _same_number(
                _probability(
                    requirements.get("minimum_observed_score"),
                    label="DDQK final minimum observed score",
                ),
                FINAL_MIN_SCORE,
            )
            or not _same_number(
                _probability(
                    requirements.get("minimum_observed_black_score"),
                    label="DDQK final minimum black score",
                ),
                FINAL_MIN_COLOR_SCORE,
            )
            or not _same_number(
                _probability(
                    requirements.get("minimum_observed_white_score"),
                    label="DDQK final minimum white score",
                ),
                FINAL_MIN_COLOR_SCORE,
            )
            or not _same_number(
                _probability(
                    requirements.get(
                        "minimum_exact_pair_sweep_one_sided_95_lower_bound"
                    ),
                    label="DDQK final minimum exact pair-sweep lower bound",
                ),
                FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95,
            )
            or requirements.get("pair_sweep_success_definition")
            != "model_wins_both_color_swapped_games"
            or requirements.get("requires_zero_errors") is not True
            or requirements.get("requires_zero_truncated_games") is not True
        ):
            raise GateError("DDQK final-certification requirements metadata is invalid")
    elif certification.get("final_certified") is not False:
        raise GateError("development DDQK report must be marked not_final_certified")

    raw_ci = summary.get("paired_bootstrap_ci95")
    if not isinstance(raw_ci, list) or len(raw_ci) != 2:
        raise GateError("DDQK report has no descriptive paired_bootstrap_ci95 interval")
    bootstrap_ci = [
        _probability(raw_ci[0], label="DDQK bootstrap lower bound"),
        _probability(raw_ci[1], label="DDQK bootstrap upper bound"),
    ]
    if bootstrap_ci[0] > bootstrap_ci[1]:
        raise GateError("DDQK paired bootstrap interval is reversed")

    failures: list[str] = []
    for name, actual, required in (
        ("overall", score, min_score),
        ("black", black, min_black),
        ("white", white, min_white),
        ("one-sided 95% lower bound", lower, min_ci_low),
    ):
        if actual < required:
            failures.append(f"{name} {actual:.6f} < {required:.6f}")
    if failures:
        raise GateError("DDQK gate failed: " + "; ".join(failures))
    return {
        "certification_mode": certification_mode,
        "final_certified": certification_mode == FINAL_CERTIFICATION_MODE,
        "pairs": requested_pairs,
        "score": score,
        "black_score": black,
        "white_score": white,
        "one_sided_95_lower_bound": lower,
        "one_sided_95_lower_bound_method": "hoeffding_bounded_mean",
        "pair_sweep_successes": pair_sweep_successes,
        "pair_sweep_trials": pair_sweep_trials,
        "observed_pair_sweep_rate": observed_pair_sweep_rate,
        "exact_pair_sweep_lower95": exact_pair_sweep_lower95,
        "exact_pair_sweep_lower95_method": "clopper_pearson_exact_binomial",
        "paired_bootstrap_ci95_descriptive_only": bootstrap_ci,
        "opening_manifest_sha256": manifest_sha,
        **provenance,
    }


def promote_candidate(
    candidate_path: Path,
    tactical_json: Path,
    ddqk_json: Path,
    output_path: Path,
    *,
    expected_candidate_sha256: str,
    min_tactical_top1: float = 0.995,
    min_tactical_samples: int = 48,
    min_ddqk_score: float = 0.995,
    min_ddqk_black: float = 0.99,
    min_ddqk_white: float = 0.99,
    min_ddqk_ci_low: float = 0.0,
    min_ddqk_pairs: int = 50,
    certification_mode: str = DEVELOPMENT_MODE,
    expected_ddqk_source_sha256: str | None = None,
    expected_ddqk_dll_sha256: str | None = None,
    expected_ddqk_assets_bundle_sha256: str | None = None,
    expected_ddqk_depth: int | None = None,
    expected_simulations: int | None = None,
    expected_opening_plies: int | None = None,
    expected_max_moves: int | None = None,
    expected_seed: int | None = None,
    expected_evaluation_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a final champion or an explicitly non-final development screen."""

    candidate_path = candidate_path.resolve()
    output_path = output_path.resolve()
    _ensure_distinct_output(output_path, [candidate_path], champion=True)
    if certification_mode not in (DEVELOPMENT_MODE, FINAL_CERTIFICATION_MODE):
        raise GateError(
            f"certification_mode must be {DEVELOPMENT_MODE!r} or "
            f"{FINAL_CERTIFICATION_MODE!r}"
        )
    for name, value in (
        ("min_tactical_top1", min_tactical_top1),
        ("min_ddqk_score", min_ddqk_score),
        ("min_ddqk_black", min_ddqk_black),
        ("min_ddqk_white", min_ddqk_white),
        ("min_ddqk_ci_low", min_ddqk_ci_low),
    ):
        _probability(value, label=name)
    if min_tactical_samples <= 0 or min_ddqk_pairs <= 0:
        raise GateError("minimum sample and pair counts must be positive")
    if certification_mode == FINAL_CERTIFICATION_MODE:
        # Final certification minima are invariants, not weakening knobs.
        min_ddqk_pairs = max(int(min_ddqk_pairs), FINAL_MIN_PAIRS)
        min_ddqk_score = max(float(min_ddqk_score), FINAL_MIN_SCORE)
        min_ddqk_black = max(float(min_ddqk_black), FINAL_MIN_COLOR_SCORE)
        min_ddqk_white = max(float(min_ddqk_white), FINAL_MIN_COLOR_SCORE)
        min_ddqk_ci_low = max(float(min_ddqk_ci_low), FINAL_MIN_CI_LOW)

    candidate, candidate_sha = _read_verified_checkpoint(
        candidate_path,
        expected_candidate_sha256,
        label="candidate evaluation checkpoint",
    )
    if (
        candidate.get("format_version") != FORMAT_VERSION
        or candidate.get("v3_stage") != CANDIDATE_STAGE
        or candidate.get("approval_status") != "not_approved"
        or candidate.get("is_approved") is not False
    ):
        raise GateError("candidate must be a format-v3 not_approved evaluation checkpoint")
    for key in ("candidate_model", "train_model", "best_model"):
        if key not in candidate:
            raise GateError(f"candidate checkpoint is missing required {key}")
    if not _states_equal(
        candidate["candidate_model"],
        candidate["best_model"],
        left_label="candidate_model",
        right_label="best_model",
    ) or not _states_equal(
        candidate["train_model"],
        candidate["best_model"],
        left_label="train_model",
        right_label="best_model",
    ):
        raise GateError("candidate evaluation model keys do not contain identical weights")
    config, candidate_state = _validated_config_and_state(
        candidate, "best_model", label="candidate evaluation checkpoint"
    )

    tactical_report, tactical_sha = _read_json_with_sha(
        tactical_json, label="tactical evaluation JSON"
    )
    ddqk_report, ddqk_sha = _read_json_with_sha(
        ddqk_json, label="DDQK evaluation JSON"
    )
    tactical_metrics = _validate_tactical_report(
        tactical_report,
        candidate_sha256=candidate_sha,
        min_top1=min_tactical_top1,
        min_samples=min_tactical_samples,
    )
    ddqk_metrics = _validate_ddqk_report(
        ddqk_report,
        candidate_sha256=candidate_sha,
        min_score=min_ddqk_score,
        min_black=min_ddqk_black,
        min_white=min_ddqk_white,
        min_ci_low=min_ddqk_ci_low,
        min_pairs=min_ddqk_pairs,
        certification_mode=certification_mode,
        expected_ddqk_source_sha256=expected_ddqk_source_sha256,
        expected_ddqk_dll_sha256=expected_ddqk_dll_sha256,
        expected_ddqk_assets_bundle_sha256=expected_ddqk_assets_bundle_sha256,
        expected_ddqk_depth=expected_ddqk_depth,
        expected_simulations=expected_simulations,
        expected_opening_plies=expected_opening_plies,
        expected_max_moves=expected_max_moves,
        expected_seed=expected_seed,
        expected_evaluation_bundle_sha256=expected_evaluation_bundle_sha256,
    )
    thresholds = {
        "min_tactical_raw_top1": float(min_tactical_top1),
        "min_tactical_samples": int(min_tactical_samples),
        "min_ddqk_score": float(min_ddqk_score),
        "min_ddqk_black": float(min_ddqk_black),
        "min_ddqk_white": float(min_ddqk_white),
        "min_ddqk_ci_low": float(min_ddqk_ci_low),
        "min_ddqk_exact_pair_sweep_lower95": (
            FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95
            if certification_mode == FINAL_CERTIFICATION_MODE
            else None
        ),
        "min_ddqk_pairs": int(min_ddqk_pairs),
        "certification_mode": certification_mode,
    }
    final_certified = certification_mode == FINAL_CERTIFICATION_MODE
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "v3_stage": CHAMPION_STAGE if final_certified else DEVELOPMENT_STAGE,
        "iteration": int(candidate.get("iteration", -1)),
        "global_step": int(candidate.get("global_step", 0)),
        "config": config,
        "model_spec": dict(candidate.get("model_spec", {})),
        "best_model": _cpu_state_copy(candidate_state, label="champion best_model"),
        "approved_model": _cpu_state_copy(candidate_state, label="champion approved_model"),
        "candidate_model": _cpu_state_copy(candidate_state, label="champion candidate_model"),
        "train_model": _cpu_state_copy(candidate_state, label="champion train_model"),
        "approval_status": "approved" if final_certified else "not_final_certified",
        "is_approved": final_certified,
        "approved_at_utc": _utc_now() if final_certified else None,
        "screened_at_utc": _utc_now(),
        "provenance": {
            "candidate_evaluation_checkpoint": str(candidate_path),
            "candidate_evaluation_checkpoint_sha256": candidate_sha,
            "source_checkpoint": candidate.get("source_checkpoint"),
            "source_checkpoint_sha256": candidate.get(
                "source_checkpoint_sha256"
            ),
            "source_checkpoint_stage": candidate.get("source_checkpoint_stage"),
            "source_model_key": candidate.get("source_model_key"),
            # Retain the legacy self-play provenance aliases so existing
            # consumers can continue reading previously established keys.
            "source_selfplay_checkpoint": candidate.get("source_checkpoint"),
            "source_selfplay_checkpoint_sha256": candidate.get(
                "source_checkpoint_sha256"
            ),
            "source_parent_checkpoint_sha256": candidate.get(
                "source_parent_checkpoint_sha256"
            ),
            "prior_approved_checkpoint_sha256": candidate.get(
                "prior_approved_checkpoint_sha256"
            ),
        },
        "external_evaluation": {
            "status": (
                "passed_final_certification"
                if final_certified
                else "passed_development_screen_not_final_certified"
            ),
            "final_certified": final_certified,
            "thresholds": thresholds,
            "tactical": {
                "report": str(tactical_json.resolve()),
                "report_sha256": tactical_sha,
                **tactical_metrics,
            },
            "ddqk": {
                "report": str(ddqk_json.resolve()),
                "report_sha256": ddqk_sha,
                **ddqk_metrics,
            },
        },
    }
    _atomic_no_clobber_torch_save(payload, output_path)
    return {
        "status": (
            "approved_champion_created"
            if final_certified
            else "development_screen_passed_not_final_certified"
        ),
        "final_certified": final_certified,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "candidate_sha256": candidate_sha,
        "iteration": payload["iteration"],
        "metrics": {"tactical": tactical_metrics, "ddqk": ddqk_metrics},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser(
        "freeze",
        help="freeze a self-play candidate or supervised warmstart train_model",
    )
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--expected-source-sha256", required=True)
    freeze.add_argument(
        "--expected-parent-sha256",
        help=(
            "required for tactical_expert_warmstart sources; must match the "
            "parent_checkpoint_sha256 recorded in the verified source"
        ),
    )
    freeze.add_argument("--output", type=Path, required=True)

    promote = commands.add_parser("promote", help="promote a fully evaluated candidate")
    promote.add_argument("--candidate", type=Path, required=True)
    promote.add_argument("--expected-candidate-sha256", required=True)
    promote.add_argument("--tactical-json", type=Path, required=True)
    promote.add_argument("--ddqk-json", type=Path, required=True)
    promote.add_argument("--output", type=Path, required=True)
    promote.add_argument("--min-tactical-top1", type=float, default=0.995)
    promote.add_argument("--min-tactical-samples", type=int, default=48)
    promote.add_argument("--min-ddqk-score", type=float, default=0.995)
    promote.add_argument("--min-ddqk-black", type=float, default=0.99)
    promote.add_argument("--min-ddqk-white", type=float, default=0.99)
    promote.add_argument(
        "--min-ddqk-ci-low",
        type=float,
        default=0.0,
        help="development lower-bound gate; final-certification enforces at least 0.95",
    )
    promote.add_argument("--min-ddqk-pairs", type=int, default=50)
    promote.add_argument(
        "--certification-mode",
        choices=(DEVELOPMENT_MODE, FINAL_CERTIFICATION_MODE),
        default=DEVELOPMENT_MODE,
    )
    promote.add_argument("--expected-ddqk-source-sha256")
    promote.add_argument("--expected-ddqk-dll-sha256")
    promote.add_argument("--expected-ddqk-assets-bundle-sha256")
    promote.add_argument("--expected-ddqk-depth", type=int)
    promote.add_argument("--expected-simulations", type=int)
    promote.add_argument("--expected-opening-plies", type=int)
    promote.add_argument("--expected-max-moves", type=int)
    promote.add_argument("--expected-seed", type=int)
    promote.add_argument("--expected-evaluation-bundle-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_candidate(
                args.source,
                args.output,
                expected_source_sha256=args.expected_source_sha256,
                expected_parent_sha256=args.expected_parent_sha256,
            )
        else:
            result = promote_candidate(
                args.candidate,
                args.tactical_json,
                args.ddqk_json,
                args.output,
                expected_candidate_sha256=args.expected_candidate_sha256,
                min_tactical_top1=args.min_tactical_top1,
                min_tactical_samples=args.min_tactical_samples,
                min_ddqk_score=args.min_ddqk_score,
                min_ddqk_black=args.min_ddqk_black,
                min_ddqk_white=args.min_ddqk_white,
                min_ddqk_ci_low=args.min_ddqk_ci_low,
                min_ddqk_pairs=args.min_ddqk_pairs,
                certification_mode=args.certification_mode,
                expected_ddqk_source_sha256=args.expected_ddqk_source_sha256,
                expected_ddqk_dll_sha256=args.expected_ddqk_dll_sha256,
                expected_ddqk_assets_bundle_sha256=(
                    args.expected_ddqk_assets_bundle_sha256
                ),
                expected_ddqk_depth=args.expected_ddqk_depth,
                expected_simulations=args.expected_simulations,
                expected_opening_plies=args.expected_opening_plies,
                expected_max_moves=args.expected_max_moves,
                expected_seed=args.expected_seed,
                expected_evaluation_bundle_sha256=(
                    args.expected_evaluation_bundle_sha256
                ),
            )
    except GateError as error:
        parser_error = {"status": "rejected", "reason": str(error)}
        print(json.dumps(parser_error, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
