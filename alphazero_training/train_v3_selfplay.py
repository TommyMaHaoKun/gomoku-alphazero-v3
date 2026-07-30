#!/usr/bin/env python3
"""V3 mixed-replay self-play trainer for 19x19 freestyle Gomoku.

This trainer deliberately lives beside, rather than modifying, the V2
``train_alphazero.py`` loop.  V3 self-play uses the shared tactical-aware root
search, then trains on an explicit mixture of fresh self-play, DDQK teacher
positions, the tactical curriculum, and an optional manifest-authenticated
white-defense curriculum.  Active games share tactical routing, root
evaluation, and leaf inference batches while keeping the replay and checkpoint
contracts auditable.

Architecture / 代码架构
-----------------------
This V3 pipeline extends the base model without changing the legacy trainer.
``V3RootSearch`` generates tactically filtered self-play. Static replay sources
and the circular self-play replay are combined by ``SourceMixer``. Mixed
batches train ``PolicyValueNet``; manifests, replay chunks, RNG states, and
atomic checkpoints make every resume auditable.

V3 流水线在不改动旧训练器的情况下扩展基础模型。``V3RootSearch`` 生成经过战术
过滤的自我对弈；静态数据源与循环自我对弈回放由 ``SourceMixer`` 按比例混合；
混合批次训练 ``PolicyValueNet``。清单、回放分块、随机数状态和原子检查点让每次
恢复都可以审计。

Key algorithms / 重要算法
-------------------------
Parallel self-play shares batched neural inference. Source quotas control the
ratio of fresh self-play, teacher, tactical, and optional white-defense data.
Weighted policy/value losses and an optional safe-vs-unsafe margin loss update
the network with AdamW and a warmup/cosine schedule. ``candidate_model`` is the
new network under evaluation; ``best_model`` remains the accepted champion
until an external gate approves promotion.

并行自我对弈共享批量神经网络推理。数据源配额控制新鲜自我对弈、教师、战术和可选
白棋防守数据的比例。加权策略/价值损失及可选的安全-危险间隔损失通过 AdamW 与
预热/余弦学习率更新网络。``candidate_model`` 是待评估的新网络，``best_model``
在外部门控批准晋级前始终保留为已接受冠军。
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .train_alphazero import (
    BLACK,
    EMPTY,
    WHITE,
    Config,
    GomokuGame,
    PolicyValueNet,
)
from .v3_search import V3RootSearch


STOP_REQUESTED = False
SOURCE_ORDER = ("selfplay", "ddqk", "tactical", "white_defense")
WHITE_DEFENSE_SCHEMA_VERSION = 1
WHITE_DEFENSE_SOURCE = "ddqk_benchmark_format3_white_model_losses"
WHITE_DEFENSE_TRAINING_PROHIBITION = "never pass this archive to a trainer"
WHITE_DEFENSE_REQUIRED_ARRAYS = {
    "states",
    "policies",
    "values",
    "policy_weights",
    "value_weights",
    "source",
    "priority",
    "group_id",
    "split",
    "report_sha256",
    "opening_sha256",
    "state_hash",
    "candidate_mask",
    "safe_mask",
    "vcf_unknown_mask",
    "unsafe_immediate_mask",
    "unsafe_three_ply_mask",
    "unsafe_vcf_mask",
}


@dataclass
class V3SelfplayConfig:
    """Training-loop settings kept separate from the legacy search config."""

    iterations: int = 100
    selfplay_games: int = 32
    parallel_games: int = 32
    simulations: int = 384
    temperature_moves: int = 12
    max_game_plies: int = 361
    train_steps: int = 200
    batch_size: int = 384
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 1e-4
    replay_capacity: int = 250_000
    max_replay_chunks: int = 150
    selfplay_quota: float = 0.50
    ddqk_quota: float = 0.25
    tactical_quota: float = 0.25
    # Zero keeps V3/V3E checkpoints and launchers exactly backward compatible.
    white_defense_quota: float = 0.0
    # Zero preserves the historical mixed-training objective exactly.
    safe_hard_negative_scale: float = 0.0
    safe_hard_negative_margin: float = 1.0
    selfplay_policy_weight: float = 1.0
    selfplay_value_weight: float = 1.0
    seed: int = 20260722
    log_every_steps: int = 25

    def quotas(self) -> dict[str, float]:
        return {
            "selfplay": self.selfplay_quota,
            "ddqk": self.ddqk_quota,
            "tactical": self.tactical_quota,
            "white_defense": self.white_defense_quota,
        }

    def validate(self) -> None:
        integer_positive = {
            "iterations": self.iterations,
            "selfplay_games": self.selfplay_games,
            "parallel_games": self.parallel_games,
            "simulations": self.simulations,
            "temperature_moves": self.temperature_moves,
            "max_game_plies": self.max_game_plies,
            "train_steps": self.train_steps,
            "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
            "max_replay_chunks": self.max_replay_chunks,
            "log_every_steps": self.log_every_steps,
        }
        bad = [name for name, value in integer_positive.items() if value <= 0]
        if bad:
            raise ValueError(f"positive values required for: {', '.join(bad)}")
        if self.min_learning_rate < 0 or self.learning_rate <= 0:
            raise ValueError("learning rates must be non-negative and base LR positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("minimum learning rate cannot exceed base learning rate")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if any(weight < 0 for weight in self.quotas().values()):
            raise ValueError("source quotas cannot be negative")
        if not sum(self.quotas().values()) > 0:
            raise ValueError("at least one source quota must be positive")
        if self.selfplay_policy_weight < 0 or self.selfplay_value_weight < 0:
            raise ValueError("self-play loss weights cannot be negative")
        if (
            not math.isfinite(self.safe_hard_negative_scale)
            or self.safe_hard_negative_scale < 0
        ):
            raise ValueError("safe hard-negative scale must be finite and non-negative")
        if (
            not math.isfinite(self.safe_hard_negative_margin)
            or self.safe_hard_negative_margin < 0
        ):
            raise ValueError("safe hard-negative margin must be finite and non-negative")
        if self.safe_hard_negative_scale > 0 and self.white_defense_quota <= 0:
            raise ValueError(
                "safe hard-negative loss requires a positive white-defense quota"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a SHA256 hex digest")
    return digest


def _stable_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_int(value: object, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error


def validate_white_defense_manifest(
    dataset_path: Path,
    manifest_path: Path,
    *,
    board_size: int,
) -> dict[str, object]:
    """Authenticate a white-defense *train* archive and its provenance.

    The separate eval archive is deliberately never opened as training data.
    Its manifest entry must carry the generator's explicit prohibition, while
    the selected NPZ must hash-identify as the train artifact.
    """

    dataset_path = dataset_path.resolve()
    manifest_path = manifest_path.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"white-defense dataset not found: {dataset_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"white-defense manifest not found: {manifest_path}")

    sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"white-defense manifest SHA256 sidecar not found: {sidecar_path}"
        )
    sidecar_parts = sidecar_path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if len(sidecar_parts) != 2:
        raise ValueError("white-defense manifest SHA256 sidecar is malformed")
    declared_manifest_sha = _require_sha256(
        sidecar_parts[0], label="white-defense manifest sidecar digest"
    )
    if Path(sidecar_parts[1].strip()).name != manifest_path.name:
        raise ValueError("white-defense manifest SHA256 sidecar names another file")
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != declared_manifest_sha:
        raise ValueError("white-defense manifest SHA256 does not match its sidecar")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("white-defense manifest is not valid UTF-8 JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("white-defense manifest must be a JSON object")
    if _manifest_int(
        manifest.get("schema_version", -1), label="white-defense schema_version"
    ) != WHITE_DEFENSE_SCHEMA_VERSION:
        raise ValueError("white-defense manifest schema_version is unsupported")
    if manifest.get("source") != WHITE_DEFENSE_SOURCE:
        raise ValueError("white-defense manifest source/provenance is invalid")

    claimed_payload_sha = _require_sha256(
        manifest.get("manifest_payload_sha256"),
        label="white-defense manifest payload digest",
    )
    payload_without_digest = dict(manifest)
    payload_without_digest.pop("manifest_payload_sha256", None)
    if _stable_json_sha256(payload_without_digest) != claimed_payload_sha:
        raise ValueError("white-defense manifest payload SHA256 does not match")

    report_sha = _require_sha256(
        manifest.get("report_sha256"), label="white-defense report provenance"
    )
    benchmark_audit = manifest.get("benchmark_audit")
    if not isinstance(benchmark_audit, Mapping):
        raise ValueError("white-defense manifest has no benchmark provenance audit")
    provenance_generation = str(benchmark_audit.get("provenance_generation", ""))
    if provenance_generation not in {"legacy4", "current6"}:
        raise ValueError("white-defense benchmark provenance generation is invalid")

    rules = manifest.get("rules")
    if not isinstance(rules, Mapping) or (
        _manifest_int(
            rules.get("board_size", -1), label="white-defense board_size"
        )
        != board_size
        or _manifest_int(
            rules.get("win_length", -1), label="white-defense win_length"
        )
        != 5
        or rules.get("freestyle") is not True
        or rules.get("side_to_move") != "white"
    ):
        raise ValueError("white-defense manifest rules are incompatible with this run")
    claim_boundary = manifest.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping) or claim_boundary.get("label") != (
        "bounded_non_loss_within_search_candidates"
    ):
        raise ValueError("white-defense bounded-label provenance is invalid")
    split_summary = manifest.get("split")
    if not isinstance(split_summary, Mapping) or (
        split_summary.get("assigned_before_replay_or_tactical_labelling") is not True
        or split_summary.get("augmentation") != "none"
    ):
        raise ValueError("white-defense train/eval split provenance is invalid")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("white-defense manifest has no artifacts map")
    train_artifact = artifacts.get("train")
    eval_artifact = artifacts.get("eval")
    if not isinstance(train_artifact, Mapping) or not isinstance(eval_artifact, Mapping):
        raise ValueError("white-defense manifest must describe train and eval artifacts")
    if eval_artifact.get("training_prohibition") != WHITE_DEFENSE_TRAINING_PROHIBITION:
        raise ValueError("white-defense eval artifact is not explicitly training-prohibited")

    train_sha = _require_sha256(
        train_artifact.get("sha256"), label="white-defense train artifact"
    )
    eval_sha = _require_sha256(
        eval_artifact.get("sha256"), label="white-defense eval artifact"
    )
    if train_sha == eval_sha:
        raise ValueError("white-defense train and eval artifacts cannot have the same digest")
    actual_dataset_sha = sha256_file(dataset_path)
    if actual_dataset_sha != train_sha:
        if actual_dataset_sha == eval_sha:
            raise ValueError("refusing to train on the white-defense eval artifact")
        raise ValueError("white-defense train archive SHA256 does not match manifest")
    if _manifest_int(
        train_artifact.get("bytes", -1), label="white-defense train bytes"
    ) != dataset_path.stat().st_size:
        raise ValueError("white-defense train archive byte size does not match manifest")

    with np.load(dataset_path, allow_pickle=False) as archive:
        missing = sorted(WHITE_DEFENSE_REQUIRED_ARRAYS - set(archive.files))
        if missing:
            raise ValueError(f"white-defense archive is missing schema arrays {missing}")
        count = len(archive["states"])
        if _manifest_int(
            train_artifact.get("records", -1), label="white-defense train records"
        ) != count:
            raise ValueError("white-defense train record count does not match manifest")
        if count <= 0:
            raise ValueError("white-defense train archive is empty")

        for name in (
            "source",
            "group_id",
            "split",
            "report_sha256",
            "opening_sha256",
            "state_hash",
        ):
            if archive[name].shape != (count,) or archive[name].dtype.kind not in "US":
                raise ValueError(f"white-defense {name} provenance array is invalid")
        splits = np.asarray([str(value).strip().lower() for value in archive["split"]])
        if np.any(splits != "train"):
            raise ValueError("white-defense archive is not a pure train split")
        report_hashes = np.asarray(
            [str(value).lower() for value in archive["report_sha256"]]
        )
        if np.any(report_hashes != report_sha):
            raise ValueError("white-defense row provenance disagrees with manifest report")
        expected_source_prefix = f"white_defense|report={report_sha[:16]}|"
        if any(
            not str(value).startswith(expected_source_prefix)
            for value in archive["source"]
        ):
            raise ValueError("white-defense row source provenance is invalid")
        if any(report_sha not in str(value) for value in archive["group_id"]):
            raise ValueError("white-defense group provenance omits the report digest")
        for name in ("opening_sha256", "state_hash"):
            for value in archive[name]:
                _require_sha256(value, label=f"white-defense {name}")

        states = np.asarray(archive["states"])
        if states.shape != (count, 4, board_size, board_size):
            raise ValueError("white-defense state schema is invalid")
        if np.any(states[:, 3] != 0):
            raise ValueError("white-defense archive contains a non-white-to-move state")
        for index, declared_state_hash in enumerate(archive["state_hash"]):
            if hashlib.sha256(states[index].tobytes()).hexdigest() != str(
                declared_state_hash
            ).lower():
                raise ValueError("white-defense state provenance hash does not match")

        values = np.asarray(archive["values"])
        value_weights = np.asarray(archive["value_weights"])
        if np.any(values != 0) or np.any(value_weights != 0):
            raise ValueError("white-defense curriculum must remain policy-only")

        action_count = board_size * board_size
        masks: dict[str, np.ndarray] = {}
        for name in (
            "candidate_mask",
            "safe_mask",
            "vcf_unknown_mask",
            "unsafe_immediate_mask",
            "unsafe_three_ply_mask",
            "unsafe_vcf_mask",
        ):
            mask = np.asarray(archive[name])
            if mask.shape != (count, action_count) or not np.all(
                (mask == 0) | (mask == 1)
            ):
                raise ValueError(f"white-defense {name} schema is invalid")
            masks[name] = mask.astype(bool)
        unsafe = (
            masks["unsafe_immediate_mask"]
            | masks["unsafe_three_ply_mask"]
            | masks["unsafe_vcf_mask"]
        )
        if np.any(masks["safe_mask"] & unsafe) or not np.array_equal(
            masks["candidate_mask"], masks["safe_mask"] | unsafe
        ):
            raise ValueError("white-defense candidate safety classification is invalid")
        if np.any(masks["vcf_unknown_mask"] & ~masks["safe_mask"]):
            raise ValueError("white-defense UNKNOWN_BUDGET actions were marked unsafe")
        policies = np.asarray(archive["policies"])
        if policies.shape != (count, action_count) or not np.array_equal(
            policies > 0, masks["safe_mask"]
        ):
            raise ValueError("white-defense policy support does not match safe actions")

        # A self-consistent manifest must not authenticate contradictory labels
        # for an identical encoded position.  Such conflicts can otherwise be
        # produced by repeated wall-clock-bounded tactical queries.
        labels_by_state: dict[str, tuple[bytes, ...]] = {}
        for index, raw_state_hash in enumerate(archive["state_hash"]):
            state_hash = str(raw_state_hash).lower()
            signature = tuple(
                np.ascontiguousarray(masks[name][index]).tobytes()
                for name in (
                    "candidate_mask",
                    "safe_mask",
                    "vcf_unknown_mask",
                    "unsafe_immediate_mask",
                    "unsafe_three_ply_mask",
                    "unsafe_vcf_mask",
                )
            )
            previous = labels_by_state.setdefault(state_hash, signature)
            if previous != signature:
                raise ValueError(
                    "white-defense archive gives one state conflicting tactical labels"
                )

    validation = manifest.get("validation")
    if not isinstance(validation, Mapping) or _manifest_int(
        validation.get("train_records", -1),
        label="white-defense validation train_records",
    ) != count:
        raise ValueError("white-defense validation summary disagrees with train archive")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "manifest_payload_sha256": claimed_payload_sha,
        "schema_version": WHITE_DEFENSE_SCHEMA_VERSION,
        "source": WHITE_DEFENSE_SOURCE,
        "report_sha256": report_sha,
        "provenance_generation": provenance_generation,
        "eval_training_prohibition": WHITE_DEFENSE_TRAINING_PROHIBITION,
        "train_sha256": train_sha,
        "train_records": count,
    }


def allocate_source_counts(total: int, quotas: Mapping[str, float]) -> dict[str, int]:
    """Allocate an exact batch by deterministic largest remainder rounding."""

    if total <= 0:
        raise ValueError("total must be positive")
    names = list(quotas)
    weights = np.asarray([float(quotas[name]) for name in names], dtype=np.float64)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("quotas must be finite, non-negative, and have positive sum")
    exact = total * weights / weights.sum()
    counts = np.floor(exact).astype(np.int64)
    missing = total - int(counts.sum())
    # Stable sorting makes ties follow the caller's source order.
    remainder_order = np.argsort(-(exact - counts), kind="stable")
    counts[remainder_order[:missing]] += 1
    return {name: int(count) for name, count in zip(names, counts)}


def weighted_loss(
    logits: torch.Tensor,
    predicted_values: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    policy_weights: torch.Tensor,
    value_weights: torch.Tensor,
    safe_set_rows: torch.Tensor | None = None,
    candidate_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Policy/value loss with optional white-defense safe-set supervision.

    Ordinary rows retain the historical soft-target cross entropy.  A marked
    white-defense row instead minimizes the negative log safe probability
    after renormalizing within its candidate mask; probability outside the
    bounded candidate scope and redistribution within the safe set are both
    intentionally unpenalized.
    """

    log_probabilities = F.log_softmax(logits, dim=1)
    policy_per_position = -(target_policies * log_probabilities).sum(dim=1)
    if safe_set_rows is not None:
        safe_set_rows = safe_set_rows.to(device=logits.device, dtype=torch.bool)
        if safe_set_rows.shape != (len(logits),):
            raise ValueError("safe_set_rows must have one flag per policy row")
        if bool(safe_set_rows.any()):
            safe_actions = target_policies > 0
            if bool(torch.any(safe_set_rows & ~safe_actions.any(dim=1))):
                raise ValueError("every safe-set row must contain at least one safe action")
            if candidate_masks is None:
                candidate_actions = torch.ones_like(safe_actions)
            else:
                candidate_actions = candidate_masks.to(
                    device=logits.device, dtype=torch.bool
                )
                if candidate_actions.shape != safe_actions.shape:
                    raise ValueError("candidate_masks must match the policy shape")
            if bool(
                torch.any(
                    safe_set_rows
                    & (
                        ~candidate_actions.any(dim=1)
                        | (safe_actions & ~candidate_actions).any(dim=1)
                    )
                )
            ):
                raise ValueError("safe-set rows require a nonempty candidate superset")
            safe_log_mass = torch.logsumexp(
                log_probabilities.masked_fill(~safe_actions, -torch.inf),
                dim=1,
            )
            candidate_log_mass = torch.logsumexp(
                log_probabilities.masked_fill(~candidate_actions, -torch.inf),
                dim=1,
            )
            policy_per_position = torch.where(
                safe_set_rows,
                candidate_log_mass - safe_log_mass,
                policy_per_position,
            )
    policy_denominator = policy_weights.sum()
    if float(policy_denominator.detach()) > 0:
        policy_loss = (policy_per_position * policy_weights).sum() / policy_denominator
    else:
        policy_loss = logits.sum() * 0.0

    value_per_position = (predicted_values - target_values).square()
    value_denominator = value_weights.sum()
    if float(value_denominator.detach()) > 0:
        value_loss = (value_per_position * value_weights).sum() / value_denominator
    else:
        value_loss = predicted_values.sum() * 0.0
    return policy_loss + value_loss, policy_loss, value_loss


def safe_hard_negative_margin_loss(
    logits: torch.Tensor,
    target_policies: torch.Tensor,
    policy_weights: torch.Tensor,
    safe_set_rows: torch.Tensor,
    candidate_masks: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Rank one proven-safe move above the hardest unsafe search candidate.

    White-defense labels only certify actions inside ``candidate_masks``.  The
    positive is therefore the highest-logit safe action and the hard negative
    is the highest-logit candidate outside the safe set.  This directly targets
    candidate-restricted top-1 safety without inventing labels for actions that
    were outside the bounded tactical search.

    Rows with no unsafe candidate contribute zero.  Callers can preserve the
    historical objective exactly by leaving the external loss scale at zero.
    """

    if logits.ndim != 2 or target_policies.shape != logits.shape:
        raise ValueError("target_policies must match the two-dimensional logits")
    batch_size = len(logits)
    safe_set_rows = safe_set_rows.to(device=logits.device, dtype=torch.bool)
    candidate_masks = candidate_masks.to(device=logits.device, dtype=torch.bool)
    policy_weights = policy_weights.to(device=logits.device, dtype=torch.float32)
    if safe_set_rows.shape != (batch_size,):
        raise ValueError("safe_set_rows must have one flag per policy row")
    if candidate_masks.shape != logits.shape:
        raise ValueError("candidate_masks must match the policy shape")
    if policy_weights.shape != (batch_size,):
        raise ValueError("policy_weights must have one weight per policy row")
    if margin < 0 or not math.isfinite(margin):
        raise ValueError("safe hard-negative margin must be finite and non-negative")

    safe_actions = target_policies.to(device=logits.device) > 0
    if bool(torch.any(safe_set_rows & ~safe_actions.any(dim=1))):
        raise ValueError("every safe-set row must contain a safe action")
    if bool(
        torch.any(
            safe_set_rows
            & (
                ~candidate_masks.any(dim=1)
                | (safe_actions & ~candidate_masks).any(dim=1)
            )
        )
    ):
        raise ValueError("safe-set rows require a nonempty candidate superset")

    unsafe_candidates = candidate_masks & ~safe_actions
    eligible = safe_set_rows & unsafe_candidates.any(dim=1)
    if not bool(eligible.any()):
        return logits.float().sum() * 0.0

    stable_logits = logits.float()[eligible]
    eligible_safe_actions = safe_actions[eligible]
    eligible_unsafe_candidates = unsafe_candidates[eligible]
    best_safe = stable_logits.masked_fill(
        ~eligible_safe_actions, -torch.inf
    ).max(dim=1).values
    hardest_unsafe = stable_logits.masked_fill(
        ~eligible_unsafe_candidates, -torch.inf
    ).max(dim=1).values
    violations = F.relu(float(margin) + hardest_unsafe - best_safe)
    eligible_weights = policy_weights[eligible]
    return (
        (violations * eligible_weights).sum()
        / eligible_weights.sum().clamp_min(1e-8)
    )


def _validate_arrays(arrays: Mapping[str, np.ndarray], board_size: int, label: str) -> None:
    count = len(arrays["states"])
    expected_shapes = {
        "states": (count, 4, board_size, board_size),
        "policies": (count, board_size * board_size),
        "values": (count,),
        "policy_weights": (count,),
        "value_weights": (count,),
        "priority": (count,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{label}: {name} shape {arrays[name].shape}, expected {shape}")
    if count == 0:
        raise ValueError(f"{label}: dataset is empty")
    states = np.asarray(arrays["states"])
    policies = np.asarray(arrays["policies"], dtype=np.float32)
    try:
        states_finite = np.all(np.isfinite(states))
    except TypeError as error:
        raise ValueError(f"{label}: states must be numeric") from error
    if not states_finite or not np.all((states == 0) | (states == 1)):
        raise ValueError(f"{label}: state planes must be finite binary values")
    if np.any((states[:, 0] != 0) & (states[:, 1] != 0)):
        raise ValueError(f"{label}: current/opponent stone planes overlap")

    side_sums = states[:, 3].sum(axis=(1, 2))
    board_area = board_size * board_size
    if np.any((side_sums != 0) & (side_sums != board_area)):
        raise ValueError(f"{label}: side-to-move plane must be uniformly zero or one")
    own_counts = states[:, 0].sum(axis=(1, 2))
    opponent_counts = states[:, 1].sum(axis=(1, 2))
    black_to_move = side_sums == board_area
    legal_counts = np.where(
        black_to_move,
        own_counts == opponent_counts,
        opponent_counts == own_counts + 1,
    )
    if not np.all(legal_counts):
        raise ValueError(f"{label}: stone counts are not reachable by alternating play")

    occupied = (states[:, 0] != 0) | (states[:, 1] != 0)
    occupied_counts = occupied.sum(axis=(1, 2))
    if np.any(occupied_counts >= board_area):
        raise ValueError(f"{label}: training positions must have a legal empty action")
    last_counts = states[:, 2].sum(axis=(1, 2))
    expected_last_counts = (occupied_counts > 0).astype(last_counts.dtype)
    if not np.array_equal(last_counts, expected_last_counts):
        raise ValueError(
            f"{label}: last-move plane must be empty only on an empty board and one-hot otherwise"
        )
    if np.any((states[:, 2] != 0) & (states[:, 1] == 0)):
        raise ValueError(f"{label}: last move must mark an opponent stone")

    if not np.all(np.isfinite(policies)) or np.any(policies < 0):
        raise ValueError(f"{label}: policies must be finite and non-negative")
    policy_sums = policies.sum(axis=1)
    if not np.allclose(policy_sums, 1.0, atol=2e-3):
        raise ValueError(f"{label}: every policy must sum to one")
    if np.any(np.where(occupied.reshape(count, -1), policies, 0.0) > 1e-6):
        raise ValueError(f"{label}: policy assigns mass to an occupied point")
    for name in ("values", "policy_weights", "value_weights", "priority"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"{label}: {name} contains a non-finite value")
    if np.any(np.abs(arrays["values"]) > 1.0 + 1e-6):
        raise ValueError(f"{label}: values must lie in [-1, 1]")
    if np.any(arrays["policy_weights"] < 0) or np.any(arrays["value_weights"] < 0):
        raise ValueError(f"{label}: loss weights cannot be negative")
    if np.any(arrays["priority"] <= 0):
        raise ValueError(f"{label}: priorities must be positive")


class StaticReplaySource:
    """An immutable, priority-sampled NPZ source."""

    def __init__(self, name: str, path: Path, board_size: int, seed: int):
        self.name = name
        self.path = path.resolve()
        with np.load(self.path, allow_pickle=False) as archive:
            required = {"states", "policies", "values"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"{self.path}: missing arrays {missing}")
            policy_key = "policy_weights" if "policy_weights" in archive.files else "policy_weight"
            value_key = "value_weights" if "value_weights" in archive.files else "value_weight"
            if policy_key not in archive.files or value_key not in archive.files:
                raise ValueError(f"{self.path}: missing policy/value weight arrays")
            count = len(archive["states"])
            if "split" in archive.files:
                split = np.asarray(archive["split"])
                if split.shape != (count,):
                    raise ValueError(f"{self.path}: split must have shape {(count,)}")
                normalized_split = np.asarray(
                    [str(value).strip().lower() for value in split]
                )
                if np.any(normalized_split != "train"):
                    found = sorted(set(map(str, normalized_split.tolist())))
                    raise ValueError(
                        f"{self.path}: static replay accepts only split=train, found {found}"
                    )
                split_contract = "train"
            else:
                split_contract = "absent"
            raw_arrays = {
                "states": np.array(archive["states"], copy=True),
                "policies": np.array(archive["policies"], copy=True),
                "values": np.array(archive["values"], copy=True),
                "policy_weights": np.array(archive[policy_key], copy=True),
                "value_weights": np.array(archive[value_key], copy=True),
                "priority": (
                    np.array(archive["priority"], dtype=np.float64, copy=True)
                    if "priority" in archive.files
                    else np.ones(count, dtype=np.float64)
                ),
            }
            candidate_masks = (
                np.array(archive["candidate_mask"], dtype=bool, copy=True)
                if "candidate_mask" in archive.files
                else None
            )
            if "pair_index" in archive.files:
                groups = archive["pair_index"].astype(np.int64, copy=True)
                group_key = "pair_index"
            elif name == "ddqk" and "game_index" in archive.files:
                # DDQK benchmark games are written consecutively in
                # colour-swapped pairs.  Treat the pair/opening as the unit,
                # not each correlated game as an independent group.
                groups = archive["game_index"].astype(np.int64, copy=True) // 2
                group_key = "derived_pair_from_game_index"
            elif "group_id" in archive.files:
                groups = archive["group_id"].copy()
                group_key = "group_id"
            elif "source" in archive.files and archive["source"].dtype.kind in "US":
                # Tactical D4/translation variants share the template prefix.
                groups = np.asarray(
                    [str(value).split("|", 1)[0] for value in archive["source"]]
                )
                group_key = "source_template_prefix"
            else:
                groups = np.arange(count, dtype=np.int64)
                group_key = "position"
        # Validate before narrowing dtypes.  Otherwise values such as -1 in a
        # state plane would silently wrap to uint8(255), hiding corrupt input.
        _validate_arrays(raw_arrays, board_size, str(self.path))
        arrays = {
            "states": raw_arrays["states"].astype(np.uint8, copy=False),
            "policies": raw_arrays["policies"].astype(np.float16, copy=False),
            "values": raw_arrays["values"].astype(np.float32, copy=False),
            "policy_weights": raw_arrays["policy_weights"].astype(np.float32, copy=False),
            "value_weights": raw_arrays["value_weights"].astype(np.float32, copy=False),
            "priority": raw_arrays["priority"].astype(np.float64, copy=False),
        }
        if groups.shape != (len(arrays["states"]),):
            raise ValueError(f"{self.path}: invalid group array shape {groups.shape}")
        self.arrays = arrays
        if candidate_masks is not None and candidate_masks.shape != (
            len(arrays["states"]),
            board_size * board_size,
        ):
            raise ValueError(f"{self.path}: invalid candidate_mask shape")
        self.candidate_masks = candidate_masks
        self.groups = groups
        self.group_key = group_key
        self.split_contract = split_contract
        # First equalize total mass across global groups, then preserve the
        # source's priority ratios within a group.  This stops a long DDQK
        # game or a template with many symmetries from silently dominating.
        _, inverse = np.unique(groups, return_inverse=True)
        group_priority_totals = np.bincount(
            inverse, weights=arrays["priority"], minlength=int(inverse.max()) + 1
        )
        group_balanced_priority = arrays["priority"] / group_priority_totals[inverse]
        self.probabilities = group_balanced_priority / group_balanced_priority.sum()
        self.rng = np.random.default_rng(seed)
        self.sha256 = sha256_file(self.path)

    def __len__(self) -> int:
        return len(self.arrays["states"])

    def sample(self, count: int) -> dict[str, np.ndarray]:
        indices = self.rng.choice(len(self), size=count, replace=True, p=self.probabilities)
        result = {
            name: array[indices].copy()
            for name, array in self.arrays.items()
            if name != "priority"
        }
        result["candidate_masks"] = (
            self.candidate_masks[indices].copy()
            if self.candidate_masks is not None
            else ~(
                (result["states"][:, 0] != 0) | (result["states"][:, 1] != 0)
            ).reshape(count, -1)
        )
        return result

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": str(self.path),
            "sha256": self.sha256,
            "samples": len(self),
            "groups": int(len(np.unique(self.groups))),
            "group_key": self.group_key,
            "split": self.split_contract,
            "sampling": "equal_group_mass_then_position_priority",
        }


class WhiteDefenseReplaySource(StaticReplaySource):
    """Manifest-authenticated policy-only data from white model losses."""

    def __init__(
        self,
        path: Path,
        manifest_path: Path,
        board_size: int,
        seed: int,
    ):
        self.provenance = validate_white_defense_manifest(
            path,
            manifest_path,
            board_size=board_size,
        )
        super().__init__("white_defense", path, board_size, seed)
        if self.split_contract != "train":
            raise ValueError("white-defense archive must explicitly declare split=train")
        # The generator's leakage boundary is report + opening hash.  Its
        # pair_index is useful provenance, but must not take precedence over
        # the explicit group_id when balancing replay sampling.
        with np.load(self.path, allow_pickle=False) as archive:
            groups = archive["group_id"].copy()
        if groups.shape != (len(self),):
            raise ValueError("white-defense group_id has an invalid shape")
        self.groups = groups
        self.group_key = "group_id"
        _, inverse = np.unique(groups, return_inverse=True)
        group_priority_totals = np.bincount(
            inverse,
            weights=self.arrays["priority"],
            minlength=int(inverse.max()) + 1,
        )
        balanced = self.arrays["priority"] / group_priority_totals[inverse]
        self.probabilities = balanced / balanced.sum()

    def manifest(self) -> dict[str, object]:
        result = super().manifest()
        result.update(
            {
                "provenance_manifest_path": self.provenance["manifest_path"],
                "provenance_manifest_sha256": self.provenance["manifest_sha256"],
                "manifest_payload_sha256": self.provenance[
                    "manifest_payload_sha256"
                ],
                "schema_version": self.provenance["schema_version"],
                "source_provenance": self.provenance["source"],
                "report_sha256": self.provenance["report_sha256"],
                "provenance_generation": self.provenance[
                    "provenance_generation"
                ],
                "eval_training_prohibition": self.provenance[
                    "eval_training_prohibition"
                ],
            }
        )
        return result


class CircularSelfplayReplay:
    """Fixed-size in-memory replay preserving policy and value loss weights."""

    def __init__(self, capacity: int, board_size: int, seed: int):
        self.capacity = int(capacity)
        self.board_size = int(board_size)
        action_count = board_size * board_size
        self.states = np.empty((capacity, 4, board_size, board_size), dtype=np.uint8)
        self.policies = np.empty((capacity, action_count), dtype=np.float16)
        self.values = np.empty(capacity, dtype=np.float32)
        self.policy_weights = np.empty(capacity, dtype=np.float32)
        self.value_weights = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, arrays: Mapping[str, np.ndarray]) -> None:
        normalized = {
            "states": np.asarray(arrays["states"], dtype=np.uint8),
            "policies": np.asarray(arrays["policies"], dtype=np.float16),
            "values": np.asarray(arrays["values"], dtype=np.float32),
            "policy_weights": np.asarray(arrays["policy_weights"], dtype=np.float32),
            "value_weights": np.asarray(arrays["value_weights"], dtype=np.float32),
            "priority": np.ones(len(arrays["states"]), dtype=np.float32),
        }
        _validate_arrays(normalized, self.board_size, "selfplay replay addition")
        count = len(normalized["states"])
        if count >= self.capacity:
            for name in ("states", "policies", "values", "policy_weights", "value_weights"):
                normalized[name] = normalized[name][-self.capacity :]
            count = self.capacity
        first = min(count, self.capacity - self.position)
        for name in ("states", "policies", "values", "policy_weights", "value_weights"):
            destination = getattr(self, name)
            destination[self.position : self.position + first] = normalized[name][:first]
            remaining = count - first
            if remaining:
                destination[:remaining] = normalized[name][first:]
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, count: int) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("cannot sample an empty self-play replay")
        indices = self.rng.integers(0, self.size, size=count)
        result = {
            "states": self.states[indices].copy(),
            "policies": self.policies[indices].copy(),
            "values": self.values[indices].copy(),
            "policy_weights": self.policy_weights[indices].copy(),
            "value_weights": self.value_weights[indices].copy(),
        }
        result["candidate_masks"] = ~(
            (result["states"][:, 0] != 0) | (result["states"][:, 1] != 0)
        ).reshape(count, -1)
        return result


class SourceMixer:
    def __init__(
        self,
        sources: Mapping[str, object],
        quotas: Mapping[str, float],
        seed: int,
    ):
        self.sources = dict(sources)
        self.quotas = {name: float(quotas.get(name, 0.0)) for name in SOURCE_ORDER}
        self.rng = np.random.default_rng(seed)
        unknown = sorted(set(self.sources) - set(SOURCE_ORDER))
        if unknown:
            raise ValueError(f"unknown replay sources: {unknown}")
        missing_positive = [
            name
            for name in SOURCE_ORDER
            if self.quotas[name] > 0 and name not in self.sources
        ]
        if missing_positive:
            raise ValueError(
                f"positive quota assigned to missing sources: {missing_positive}"
            )

    def counts(self, batch_size: int) -> dict[str, int]:
        counts = allocate_source_counts(batch_size, self.quotas)
        unavailable = [
            name
            for name, count in counts.items()
            if count
            and (
                name not in self.sources
                or len(self.sources[name]) == 0  # type: ignore[arg-type]
            )
        ]
        if unavailable:
            raise ValueError(f"positive quota assigned to empty sources: {unavailable}")
        return counts

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        counts = self.counts(batch_size)
        parts: list[dict[str, np.ndarray]] = []
        labels: list[np.ndarray] = []
        for name in SOURCE_ORDER:
            count = counts[name]
            if count == 0:
                continue
            source = self.sources[name]
            part = source.sample(count)  # type: ignore[attr-defined]
            if "candidate_masks" not in part:
                part["candidate_masks"] = ~(
                    (part["states"][:, 0] != 0) | (part["states"][:, 1] != 0)
                ).reshape(count, -1)
            parts.append(part)
            labels.append(np.full(count, name, dtype=f"U{max(map(len, SOURCE_ORDER))}"))
        batch = {
            key: np.concatenate([part[key] for part in parts], axis=0)
            for key in parts[0]
        }
        batch["states"] = augment_batch_with_candidate_masks(
            batch["states"],
            batch["policies"],
            batch["candidate_masks"],
            self.rng,
        )
        source_names = np.concatenate(labels)
        order = self.rng.permutation(batch_size)
        result = {key: value[order] for key, value in batch.items()}
        result["source_names"] = source_names[order]
        return result


def augment_batch_with_candidate_masks(
    states: np.ndarray,
    policies: np.ndarray,
    candidate_masks: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one shared board symmetry to state, target, and candidate scope."""

    size = states.shape[-1]
    expected = (len(states), size * size)
    if policies.shape != expected or candidate_masks.shape != expected:
        raise ValueError("policy/candidate mask shape does not match states")
    for index in range(len(states)):
        rotations = int(rng.integers(0, 4))
        flip = bool(rng.integers(0, 2))
        state = np.rot90(states[index], rotations, axes=(-2, -1))
        policy = np.rot90(policies[index].reshape(size, size), rotations)
        candidates = np.rot90(
            candidate_masks[index].reshape(size, size), rotations
        )
        if flip:
            state = np.flip(state, axis=-1)
            policy = np.flip(policy, axis=-1)
            candidates = np.flip(candidates, axis=-1)
        states[index] = np.ascontiguousarray(state)
        policies[index] = np.ascontiguousarray(policy).reshape(-1)
        candidate_masks[index] = np.ascontiguousarray(candidates).reshape(-1)
    return states


def _safe_policy(game: GomokuGame, action: int, policy: np.ndarray) -> np.ndarray:
    expected = game.size * game.size
    target = np.asarray(policy, dtype=np.float32).reshape(-1).copy()
    if target.shape != (expected,):
        raise ValueError(f"search policy shape {target.shape}, expected {(expected,)}")
    if action < 0 or action >= expected or game.board.ravel()[action] != EMPTY:
        raise ValueError(f"V3 search returned illegal action {action}")
    target[game.board.ravel() != EMPTY] = 0.0
    target[~np.isfinite(target)] = 0.0
    target[target < 0] = 0.0
    total = float(target.sum())
    if total <= 0:
        target.fill(0)
        target[action] = 1.0
    else:
        target /= total
    return target


def generate_v3_selfplay(
    model: PolicyValueNet,
    search_config: Config,
    loop_config: V3SelfplayConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Generate tactical-aware self-play with multi-game batched MCTS."""

    model.eval()
    searcher = V3RootSearch(model, search_config, device, rng=rng)
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    policy_weights: list[float] = []
    value_weights: list[float] = []
    results = Counter()
    reasons = Counter()
    proven_count = 0
    search_seconds = 0.0
    search_waves = 0
    active_game_sum = 0
    peak_active_games = 0
    direct_search_positions = 0
    mcts_search_positions = 0
    inference_calls = 0
    evaluated_positions = 0
    max_inference_batch_size = 0
    inference_batch_histogram: Counter[int] = Counter()
    started = time.perf_counter()

    @dataclass
    class ActiveGame:
        game_index: int
        game: GomokuGame
        history: list[tuple[np.ndarray, np.ndarray, int]]

    active: list[ActiveGame] = []
    launched_games = 0
    completed_games = 0

    def launch_games() -> None:
        nonlocal launched_games
        launch_limit = loop_config.parallel_games
        # Direct callers can set STOP_REQUESTED before generation; still
        # produce one complete, valid replay game just like sequential mode.
        if STOP_REQUESTED and launched_games == 0:
            launch_limit = 1
        while (
            len(active) < launch_limit
            and launched_games < loop_config.selfplay_games
            and (not STOP_REQUESTED or launched_games == 0)
        ):
            active.append(
                ActiveGame(
                    launched_games,
                    GomokuGame(search_config.board_size, search_config.win_length),
                    [],
                )
            )
            launched_games += 1

    def finish_game(slot: ActiveGame) -> None:
        nonlocal completed_games
        game = slot.game
        history = slot.history
        truncated = not game.terminal
        if truncated:
            results["truncated"] += 1
            winner = EMPTY
        elif game.winner == BLACK:
            results["black"] += 1
            winner = BLACK
        elif game.winner == WHITE:
            results["white"] += 1
            winner = WHITE
        else:
            results["draw"] += 1
            winner = EMPTY

        for state, policy, player in history:
            states.append(state)
            policies.append(policy)
            values.append(0.0 if winner == EMPTY else (1.0 if winner == player else -1.0))
            policy_weights.append(loop_config.selfplay_policy_weight)
            # A ply limit is a censoring boundary, not a drawn result.  Keep
            # the policy search target but mask the unknown outcome entirely.
            value_weights.append(
                0.0 if truncated else loop_config.selfplay_value_weight
            )
        completed_games += 1
        logging.info(
            "selfplay game=%d/%d plies=%d result=%s",
            slot.game_index + 1,
            loop_config.selfplay_games,
            len(history),
            "truncated" if truncated else int(winner),
        )

    launch_games()
    stop_logged = False
    while active:
        peak_active_games = max(peak_active_games, len(active))
        active_game_sum += len(active)
        search_waves += 1
        temperatures = [
            1.0 if slot.game.move_count < loop_config.temperature_moves else 0.0
            for slot in active
        ]
        search_started = time.perf_counter()
        decisions, batch_stats = searcher.decide_batch(
            [slot.game for slot in active],
            simulations=loop_config.simulations,
            add_noise=True,
            temperature=temperatures,
        )
        search_seconds += time.perf_counter() - search_started
        direct_search_positions += batch_stats.direct_positions
        mcts_search_positions += batch_stats.mcts_positions
        inference_calls += batch_stats.inference_calls
        evaluated_positions += batch_stats.evaluated_positions
        max_inference_batch_size = max(
            max_inference_batch_size, batch_stats.max_inference_batch_size
        )
        for batch_size, count in batch_stats.inference_batch_histogram:
            inference_batch_histogram[batch_size] += count

        survivors: list[ActiveGame] = []
        for slot, decision in zip(active, decisions):
            game = slot.game
            policy = _safe_policy(game, decision.action, decision.policy)
            slot.history.append((game.encode(), policy, game.player))
            reasons[decision.reason] += 1
            proven_count += int(decision.proven)
            game.play(decision.action)
            if game.terminal or game.move_count >= loop_config.max_game_plies:
                finish_game(slot)
            else:
                survivors.append(slot)
        active = survivors

        if STOP_REQUESTED:
            if active and not stop_logged:
                logging.warning(
                    "stop requested; finishing %d active games without launching replacements",
                    len(active),
                )
                stop_logged = True
        else:
            launch_games()

    elapsed = time.perf_counter() - started
    if not states:
        raise RuntimeError("self-play produced no positions")
    arrays = {
        "states": np.stack(states).astype(np.uint8, copy=False),
        "policies": np.stack(policies).astype(np.float16, copy=False),
        "values": np.asarray(values, dtype=np.float32),
        "policy_weights": np.asarray(policy_weights, dtype=np.float32),
        "value_weights": np.asarray(value_weights, dtype=np.float32),
    }
    stats: dict[str, object] = {
        "mode": "parallel_tactical_batched",
        "games": completed_games,
        "requested_games": loop_config.selfplay_games,
        "launched_games": launched_games,
        "parallel_games": loop_config.parallel_games,
        "peak_active_games": peak_active_games,
        "search_waves": search_waves,
        "mean_active_games_per_wave": active_game_sum / max(search_waves, 1),
        "stopped_early": completed_games < loop_config.selfplay_games,
        "positions": len(states),
        "seconds": elapsed,
        "search_seconds": search_seconds,
        "positions_per_second": len(states) / max(elapsed, 1e-9),
        "searches_per_second": len(states) / max(search_seconds, 1e-9),
        "direct_search_positions": direct_search_positions,
        "mcts_search_positions": mcts_search_positions,
        "network_inference_calls": inference_calls,
        "network_evaluated_positions": evaluated_positions,
        "mean_inference_batch_size": evaluated_positions / max(inference_calls, 1),
        "max_inference_batch_size": max_inference_batch_size,
        "inference_batch_histogram": {
            str(size): count for size, count in sorted(inference_batch_histogram.items())
        },
        "proven_fraction": proven_count / len(states),
        "results": dict(results),
        "decision_reasons": dict(reasons),
    }
    return arrays, stats


def train_mixed_steps(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    mixer: SourceMixer,
    loop_config: V3SelfplayConfig,
    device: torch.device,
) -> tuple[dict[str, object], int]:
    model.train()
    totals = Counter()
    source_counts = Counter()
    policy_weight_sums = Counter()
    value_weight_sums = Counter()
    started = time.perf_counter()
    completed_steps = 0

    for local_step in range(1, loop_config.train_steps + 1):
        if STOP_REQUESTED:
            logging.warning("stop requested; ending optimization between steps")
            break
        batch = mixer.sample(loop_config.batch_size)
        states = torch.from_numpy(batch["states"]).to(device=device, dtype=torch.float32)
        policies = torch.from_numpy(batch["policies"].astype(np.float32)).to(device=device)
        values = torch.from_numpy(batch["values"]).to(device=device, dtype=torch.float32)
        policy_weights = torch.from_numpy(batch["policy_weights"]).to(device=device)
        value_weights = torch.from_numpy(batch["value_weights"]).to(device=device)
        safe_set_rows = torch.from_numpy(
            batch["source_names"] == "white_defense"
        ).to(device=device, dtype=torch.bool)
        candidate_masks = torch.from_numpy(batch["candidate_masks"]).to(
            device=device, dtype=torch.bool
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, predicted_values = model(states)
            loss, policy_loss, value_loss = weighted_loss(
                logits,
                predicted_values,
                policies,
                values,
                policy_weights,
                value_weights,
                safe_set_rows=safe_set_rows,
                candidate_masks=candidate_masks,
            )
            if loop_config.safe_hard_negative_scale:
                safe_margin_loss = safe_hard_negative_margin_loss(
                    logits,
                    policies,
                    policy_weights,
                    safe_set_rows,
                    candidate_masks,
                    margin=loop_config.safe_hard_negative_margin,
                )
                loss = loss + loop_config.safe_hard_negative_scale * safe_margin_loss
            else:
                safe_margin_loss = predicted_values.new_zeros(())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite mixed loss at local step {local_step}")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        completed_steps += 1

        totals["loss"] += float(loss.detach())
        totals["policy_loss"] += float(policy_loss.detach())
        totals["value_loss"] += float(value_loss.detach())
        totals["safe_hard_negative_loss"] += float(safe_margin_loss.detach())
        totals["gradient_norm"] += float(gradient_norm)
        for source_name in SOURCE_ORDER:
            mask = batch["source_names"] == source_name
            source_counts[source_name] += int(mask.sum())
            policy_weight_sums[source_name] += float(batch["policy_weights"][mask].sum())
            value_weight_sums[source_name] += float(batch["value_weights"][mask].sum())
        if local_step == 1 or local_step % loop_config.log_every_steps == 0:
            logging.info(
                "train step=%d/%d loss=%.4f policy=%.4f value=%.4f "
                "safe_margin=%.4f lr=%.6g",
                local_step,
                loop_config.train_steps,
                totals["loss"] / local_step,
                totals["policy_loss"] / local_step,
                totals["value_loss"] / local_step,
                totals["safe_hard_negative_loss"] / local_step,
                optimizer.param_groups[0]["lr"],
            )

    elapsed = time.perf_counter() - started
    metrics: dict[str, object] = {
        key: totals[key] / max(completed_steps, 1)
        for key in (
            "loss",
            "policy_loss",
            "value_loss",
            "safe_hard_negative_loss",
            "gradient_norm",
        )
    }
    metrics.update(
        {
            "seconds": elapsed,
            "completed_steps": completed_steps,
            "requested_steps": loop_config.train_steps,
            "stopped_early": completed_steps < loop_config.train_steps,
            "steps_per_second": completed_steps / max(elapsed, 1e-9),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "source_samples": dict(source_counts),
            "source_policy_weight_sums": dict(policy_weight_sums),
            "source_value_weight_sums": dict(value_weight_sums),
        }
    )
    return metrics, completed_steps


def cosine_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    if warmup_steps and step < warmup_steps:
        return max(step + 1, 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    loop_config: V3SelfplayConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = loop_config.iterations * loop_config.train_steps
    minimum_ratio = loop_config.min_learning_rate / loop_config.learning_rate
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_factor(
            step,
            total_steps,
            loop_config.warmup_steps,
            minimum_ratio,
        ),
    )


def _cpu_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _cpu_state_copy(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
    }


def _model_states_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left
    )


def atomic_torch_save(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def save_v3_checkpoint(
    destination: Path,
    *,
    iteration: int,
    global_step: int,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    search_config: Config,
    loop_config: V3SelfplayConfig,
    replay_size: int,
    replay_manifest: Sequence[Mapping[str, object]],
    rng: np.random.Generator,
    dataset_manifest: Sequence[Mapping[str, object]],
    parent_checkpoint_sha256: str,
    approved_model_state: Mapping[str, torch.Tensor],
    approved_checkpoint_sha256: str,
    metrics: Mapping[str, object],
    sampler_rng_state: Mapping[str, object] | None = None,
    external_evaluation: Mapping[str, object] | None = None,
) -> None:
    candidate_state = _cpu_model_state(model)
    approved_state = _cpu_state_copy(approved_model_state)
    approved_checkpoint_sha256 = approved_checkpoint_sha256.lower()
    if len(approved_checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in approved_checkpoint_sha256
    ):
        raise ValueError("approved checkpoint provenance must be a SHA256 digest")
    normalized_replay_manifest = validate_replay_manifest(
        replay_manifest,
        checkpoint_iteration=iteration,
    )
    represented_positions = sum(
        int(item["positions"]) for item in normalized_replay_manifest
    )
    if min(represented_positions, loop_config.replay_capacity) != replay_size:
        raise ValueError(
            "replay manifest does not reconstruct checkpoint replay_size: "
            f"represented={represented_positions} capacity={loop_config.replay_capacity} "
            f"expected={replay_size}"
        )
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    payload: dict[str, object] = {
        "format_version": 3,
        "v3_stage": "selfplay",
        "iteration": int(iteration),
        "global_step": int(global_step),
        "config": asdict(search_config),
        "v3_selfplay_config": asdict(loop_config),
        "model_spec": {
            "board_size": search_config.board_size,
            "channels": search_config.channels,
            "residual_blocks": search_config.residual_blocks,
            "input_planes": 4,
        },
        # ``train_model``/``candidate_model`` are allowed to improve or
        # regress.  ``best_model`` remains the separately approved champion
        # because the desktop player and legacy evaluators load that key.
        "train_model": candidate_state,
        "candidate_model": candidate_state,
        "best_model": approved_state,
        "approved_model": approved_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "replay_size": int(replay_size),
        "replay_manifest_version": 1,
        "replay_manifest": normalized_replay_manifest,
        "dataset_manifest": [dict(item) for item in dataset_manifest],
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "approved_checkpoint_sha256": approved_checkpoint_sha256,
        "metrics": dict(metrics),
        "external_evaluation": dict(
            external_evaluation
            or {
                "status": "not_run_for_current_checkpoint",
                "candidate_status": "not_run_for_current_checkpoint",
                "best_model_status": "carried_forward_approved_checkpoint",
                "required_gate": "paired DDQK benchmark and held-out tactical suite",
            }
        ),
        "sampler_rng_state": copy.deepcopy(dict(sampler_rng_state or {})),
        "rng_state": {
            "numpy_generator": copy.deepcopy(rng.bit_generator.state),
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": cuda_rng,
        },
        "saved_at_unix": time.time(),
    }
    atomic_torch_save(payload, destination)


def restore_v3_checkpoint(
    checkpoint_path: Path,
    *,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 3 or checkpoint.get("v3_stage") != "selfplay":
        raise ValueError(f"{checkpoint_path} is not a format-v3 self-play checkpoint")
    if "approved_model" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} predates candidate/champion separation and cannot be resumed safely"
        )
    if not _model_states_equal(checkpoint["best_model"], checkpoint["approved_model"]):
        raise ValueError(f"{checkpoint_path} has inconsistent approved/best model weights")
    model.load_state_dict(checkpoint["train_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    state = checkpoint.get("rng_state", {})
    if "numpy_generator" in state:
        rng.bit_generator.state = state["numpy_generator"]
    if "python" in state:
        random.setstate(state["python"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda" and state.get("torch_cuda"):
        # ``torch.load(..., map_location=cuda)`` also moves the serialized
        # CUDA RNG byte tensors onto the GPU, while set_rng_state_all requires
        # CPU ByteTensors.  Normalize them explicitly so a CUDA checkpoint can
        # be resumed on the same device that loaded its model/optimizer state.
        torch.cuda.set_rng_state_all(
            [rng_state.detach().cpu() for rng_state in state["torch_cuda"]]
        )
    return checkpoint


def sampler_rng_state(
    replay: CircularSelfplayReplay,
    static_sources: Mapping[str, StaticReplaySource],
    mixer: SourceMixer,
) -> dict[str, object]:
    state = {
        "selfplay": copy.deepcopy(replay.rng.bit_generator.state),
        "mixer": copy.deepcopy(mixer.rng.bit_generator.state),
    }
    state.update(
        {
            name: copy.deepcopy(source.rng.bit_generator.state)
            for name, source in static_sources.items()
        }
    )
    return state


def restore_sampler_rng_state(
    checkpoint: Mapping[str, object],
    replay: CircularSelfplayReplay,
    static_sources: Mapping[str, StaticReplaySource],
    mixer: SourceMixer,
) -> None:
    state = checkpoint.get("sampler_rng_state", {})
    if not isinstance(state, Mapping):
        return
    generators = {
        "selfplay": replay.rng,
        "mixer": mixer.rng,
        **{name: source.rng for name, source in static_sources.items()},
    }
    for name, generator in generators.items():
        if name in state:
            generator.bit_generator.state = copy.deepcopy(state[name])


def save_selfplay_chunk(
    replay_dir: Path,
    iteration: int,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, object]:
    replay_dir.mkdir(parents=True, exist_ok=True)
    destination = replay_dir / f"selfplay_{iteration:06d}.npz"
    temporary = destination.with_name(destination.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)
    return {
        "iteration": int(iteration),
        "filename": destination.name,
        "positions": int(len(arrays["states"])),
        "bytes": int(destination.stat().st_size),
        "sha256": sha256_file(destination),
    }


def validate_replay_manifest(
    manifest: Sequence[Mapping[str, object]],
    *,
    checkpoint_iteration: int,
) -> list[dict[str, object]]:
    if checkpoint_iteration < 0:
        raise ValueError("checkpoint iteration cannot be negative")
    normalized: list[dict[str, object]] = []
    previous_iteration = -1
    filenames: set[str] = set()
    for raw in manifest:
        try:
            iteration = int(raw["iteration"])
            filename = str(raw["filename"])
            positions = int(raw["positions"])
            byte_count = int(raw["bytes"])
            digest = str(raw["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid replay manifest entry") from error
        expected_filename = f"selfplay_{iteration:06d}.npz"
        if iteration <= 0 or iteration <= previous_iteration or iteration > checkpoint_iteration:
            raise ValueError("replay manifest iterations must be strictly increasing and committed")
        if filename != expected_filename or Path(filename).name != filename:
            raise ValueError(f"unsafe or inconsistent replay filename: {filename!r}")
        if filename in filenames:
            raise ValueError(f"duplicate replay filename: {filename}")
        if positions <= 0 or byte_count <= 0:
            raise ValueError("replay positions and byte size must be positive")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("replay sha256 must be 64 lowercase hexadecimal characters")
        normalized.append(
            {
                "iteration": iteration,
                "filename": filename,
                "positions": positions,
                "bytes": byte_count,
                "sha256": digest,
            }
        )
        previous_iteration = iteration
        filenames.add(filename)
    if normalized and normalized[-1]["iteration"] != checkpoint_iteration:
        raise ValueError("latest replay chunk must match checkpoint iteration")
    if checkpoint_iteration > 0 and not normalized:
        raise ValueError("trained checkpoint has no committed replay chunks")
    if checkpoint_iteration == 0 and normalized:
        raise ValueError("iteration-zero checkpoint cannot contain replay chunks")
    return normalized


def retained_replay_manifest(
    previous: Sequence[Mapping[str, object]],
    new_entry: Mapping[str, object],
    *,
    replay_size: int,
    replay_capacity: int,
    max_chunks: int,
) -> list[dict[str, object]]:
    entries = [dict(item) for item in previous] + [dict(new_entry)]
    normalized = validate_replay_manifest(
        entries,
        checkpoint_iteration=int(new_entry["iteration"]),
    )
    if replay_size <= 0 or replay_size > replay_capacity:
        raise ValueError("invalid replay size while committing a chunk")
    retained_reversed: list[dict[str, object]] = []
    represented = 0
    for item in reversed(normalized):
        retained_reversed.append(item)
        represented += int(item["positions"])
        if represented >= replay_size:
            break
    if represented < replay_size:
        raise ValueError("available replay chunks cannot reconstruct the in-memory replay")
    retained = list(reversed(retained_reversed))
    if len(retained) > max_chunks:
        # Data integrity wins over a soft file-count target.  Deleting any of
        # these chunks would make a later resume observably different.
        logging.warning(
            "replay needs %d chunks to reconstruct %d positions; exceeding max_replay_chunks=%d",
            len(retained),
            replay_size,
            max_chunks,
        )
    if min(sum(int(item["positions"]) for item in retained), replay_capacity) != replay_size:
        raise ValueError("retained replay suffix does not exactly reconstruct replay_size")
    return retained


def prune_uncommitted_replay_chunks(
    replay_dir: Path,
    committed_manifest: Sequence[Mapping[str, object]],
) -> None:
    keep = {str(item["filename"]) for item in committed_manifest}
    for path in replay_dir.glob("selfplay_*.npz"):
        if path.name not in keep:
            path.unlink(missing_ok=True)


def load_selfplay_chunks(
    replay_dir: Path,
    replay: CircularSelfplayReplay,
    manifest: Sequence[Mapping[str, object]],
    expected_replay_size: int,
) -> int:
    normalized = validate_replay_manifest(
        manifest,
        checkpoint_iteration=(int(manifest[-1]["iteration"]) if manifest else 0),
    )
    loaded = 0
    for item in normalized:
        path = replay_dir / str(item["filename"])
        if not path.is_file():
            raise FileNotFoundError(f"committed replay chunk is missing: {path}")
        if path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"committed replay chunk size changed: {path}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"committed replay chunk hash changed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "states",
                "policies",
                "values",
                "policy_weights",
                "value_weights",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"{path}: missing replay arrays {missing}")
            arrays = {name: np.array(archive[name], copy=True) for name in required}
        positions = len(arrays["states"])
        if positions != int(item["positions"]):
            raise ValueError(
                f"committed replay chunk position count changed: {path} "
                f"({positions} != {item['positions']})"
            )
        replay.add(arrays)
        loaded += positions
    if len(replay) != expected_replay_size:
        raise ValueError(
            "reconstructed replay size disagrees with checkpoint: "
            f"{len(replay)} != {expected_replay_size}"
        )
    return loaded


def _config_from_mapping(mapping: Mapping[str, object]) -> Config:
    allowed = {field.name for field in fields(Config)}
    return Config(**{name: value for name, value in mapping.items() if name in allowed})


def validated_model_state(
    raw_state: Mapping[str, torch.Tensor],
    config: Config,
    *,
    label: str,
) -> dict[str, torch.Tensor]:
    probe = PolicyValueNet(
        config.board_size,
        config.channels,
        config.residual_blocks,
    )
    try:
        probe.load_state_dict(raw_state, strict=True)
    except Exception as error:
        raise ValueError(f"{label} model architecture does not match the V3 run") from error
    return _cpu_model_state(probe)


def _loop_config(args: argparse.Namespace, resume: Mapping[str, object] | None) -> V3SelfplayConfig:
    if resume is not None:
        saved = resume.get("v3_selfplay_config")
        if not isinstance(saved, Mapping):
            raise ValueError("resume checkpoint has no v3_selfplay_config")
        allowed = {field.name for field in fields(V3SelfplayConfig)}
        config = V3SelfplayConfig(**{name: value for name, value in saved.items() if name in allowed})
    else:
        config = V3SelfplayConfig()
    if resume is not None:
        if args.smoke:
            raise ValueError("--smoke cannot be used while resuming a real training run")
        # AdamW moments and LambdaLR phase are checkpoint state.  Pretending a
        # different base LR/decay took effect while immediately restoring the
        # old optimizer made checkpoint metadata false.  A deliberate
        # optimizer retune must start a new run instead.
        for name in (
            "learning_rate",
            "min_learning_rate",
            "warmup_steps",
            "weight_decay",
            "safe_hard_negative_scale",
            "safe_hard_negative_margin",
        ):
            requested = getattr(args, name, None)
            saved_value = getattr(config, name)
            if requested is not None and not math.isclose(
                float(requested), float(saved_value), rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError(
                    f"cannot change --{name.replace('_', '-')} while resuming; "
                    "start a new run to retune optimizer, scheduler, or loss settings"
                )
    for name in (
        "iterations",
        "selfplay_games",
        "parallel_games",
        "simulations",
        "temperature_moves",
        "max_game_plies",
        "train_steps",
        "batch_size",
        "learning_rate",
        "min_learning_rate",
        "warmup_steps",
        "weight_decay",
        "replay_capacity",
        "max_replay_chunks",
        "selfplay_quota",
        "ddqk_quota",
        "tactical_quota",
        "white_defense_quota",
        "safe_hard_negative_scale",
        "safe_hard_negative_margin",
        "selfplay_policy_weight",
        "selfplay_value_weight",
        "seed",
        "log_every_steps",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(config, name, value)
    if args.smoke:
        config.iterations = 1
        config.selfplay_games = 1
        config.parallel_games = 1
        config.simulations = 2
        config.temperature_moves = 4
        config.max_game_plies = 8
        config.train_steps = 1
        config.batch_size = 8
        config.replay_capacity = 256
        config.max_replay_chunks = 2
        config.warmup_steps = 0
        config.log_every_steps = 1
    config.validate()
    counts = allocate_source_counts(config.batch_size, config.quotas())
    if any(config.quotas()[name] > 0 and counts[name] == 0 for name in SOURCE_ORDER):
        raise ValueError("batch size is too small to represent every positive source quota")
    return config


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train_v3.log", encoding="utf-8"),
        ],
        force=True,
    )


def handle_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logging.warning("received signal %s; stopping after the current game/optimizer step", signum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--approved-checkpoint",
        type=Path,
        help="immutable externally accepted champion carried in best_model",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--expert-npz", type=Path)
    parser.add_argument("--tactical-npz", type=Path)
    parser.add_argument(
        "--white-defense-npz",
        type=Path,
        help="pure-train NPZ written by white_defense_dataset.py",
    )
    parser.add_argument(
        "--white-defense-manifest",
        type=Path,
        help="manifest JSON for --white-defense-npz (including its .sha256 sidecar)",
    )
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--selfplay-games", type=int)
    parser.add_argument("--parallel-games", type=int)
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--temperature-moves", type=int)
    parser.add_argument("--max-game-plies", type=int)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--replay-capacity", type=int)
    parser.add_argument("--max-replay-chunks", type=int)
    parser.add_argument("--selfplay-quota", type=float)
    parser.add_argument("--ddqk-quota", type=float)
    parser.add_argument("--tactical-quota", type=float)
    parser.add_argument("--white-defense-quota", type=float)
    parser.add_argument(
        "--safe-hard-negative-scale",
        type=float,
        help=(
            "weight for candidate-restricted safe-vs-unsafe hard-negative "
            "margin loss on white-defense rows (default: 0, disabled)"
        ),
    )
    parser.add_argument(
        "--safe-hard-negative-margin",
        type=float,
        help=(
            "required logit lead of the best safe action over the hardest "
            "unsafe candidate (default: 1)"
        ),
    )
    parser.add_argument("--selfplay-policy-weight", type=float)
    parser.add_argument("--selfplay-value-weight", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-every-steps", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _dataset_path(
    explicit: Path | None,
    name: str,
    resume_checkpoint: Mapping[str, object] | None,
) -> Path:
    if explicit is not None:
        return explicit.resolve()
    if resume_checkpoint is not None:
        for item in resume_checkpoint.get("dataset_manifest", []):
            if item.get("name") == name:
                return Path(item["path"]).resolve()
    raise ValueError(f"--{'expert-npz' if name == 'ddqk' else 'tactical-npz'} is required")


def _optional_dataset_path(
    explicit: Path | None,
    name: str,
    resume_checkpoint: Mapping[str, object] | None,
) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    if resume_checkpoint is not None:
        for item in resume_checkpoint.get("dataset_manifest", []):
            if item.get("name") == name:
                return Path(str(item["path"])).resolve()
    return None


def _white_defense_manifest_path(
    explicit: Path | None,
    resume_checkpoint: Mapping[str, object] | None,
) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    if resume_checkpoint is not None:
        for item in resume_checkpoint.get("dataset_manifest", []):
            if item.get("name") == "white_defense":
                stored = item.get("provenance_manifest_path")
                if not stored:
                    raise ValueError(
                        "resume white-defense dataset has no provenance manifest path"
                    )
                return Path(str(stored)).resolve()
    return None


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    configure_logging(output_dir)
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    resume_path: Path | None = None
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint.resolve()
    elif args.resume:
        resume_path = output_dir / "latest.pt"
    resume_preview: dict[str, object] | None = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resume_preview = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_preview.get("format_version") != 3 or resume_preview.get("v3_stage") != "selfplay":
            raise ValueError(f"not a V3 self-play checkpoint: {resume_path}")
        if args.approved_checkpoint is not None:
            raise ValueError(
                "--approved-checkpoint cannot change during resume; start a new run after promotion"
            )

    loop_config = _loop_config(args, resume_preview)
    random.seed(loop_config.seed)
    np.random.seed(loop_config.seed)
    torch.manual_seed(loop_config.seed)
    rng = np.random.default_rng(loop_config.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    elif args.allow_cpu or args.smoke:
        device = torch.device("cpu")
        logging.warning("CUDA unavailable: CPU is allowed only for smoke/debug use")
    else:
        raise RuntimeError("CUDA is required; pass --allow-cpu only for a tiny smoke run")

    if resume_preview is not None:
        search_config = _config_from_mapping(resume_preview["config"])
        parent_sha256 = str(resume_preview.get("parent_checkpoint_sha256", ""))
        if "approved_model" not in resume_preview:
            raise ValueError(
                "resume checkpoint predates candidate/champion separation and is unsafe"
            )
        approved_checkpoint_sha256 = str(
            resume_preview.get("approved_checkpoint_sha256", "")
        )
        if not approved_checkpoint_sha256:
            raise ValueError("resume checkpoint has no approved champion provenance")
    else:
        if args.init_checkpoint is None:
            raise ValueError("--init-checkpoint is required for a new V3 run")
        init_path = args.init_checkpoint.resolve()
        warm = torch.load(init_path, map_location="cpu", weights_only=False)
        if (
            warm.get("format_version") != 3
            or warm.get("v3_stage") != "tactical_expert_warmstart"
        ):
            raise ValueError(
                "V3 self-play must start from a format-v3 "
                "tactical_expert_warmstart checkpoint"
            )
        search_config = _config_from_mapping(warm["config"])
        parent_sha256 = sha256_file(init_path)
    search_config.simulations = loop_config.simulations

    if resume_preview is not None:
        approved_state = validated_model_state(
            resume_preview["approved_model"],
            search_config,
            label="resumed approved",
        )
        if not _model_states_equal(resume_preview["best_model"], approved_state):
            raise ValueError("resume checkpoint best_model is not the approved champion")
    elif args.approved_checkpoint is not None:
        approved_path = args.approved_checkpoint.resolve()
        approved_checkpoint = torch.load(
            approved_path, map_location="cpu", weights_only=False
        )
        if "best_model" not in approved_checkpoint:
            raise ValueError(f"approved checkpoint has no best_model: {approved_path}")
        approved_state = validated_model_state(
            approved_checkpoint.get(
                "approved_model", approved_checkpoint["best_model"]
            ),
            search_config,
            label="approved checkpoint",
        )
        approved_checkpoint_sha256 = sha256_file(approved_path)
    elif "approved_model" in warm and warm.get("approved_checkpoint_sha256"):
        approved_state = validated_model_state(
            warm["approved_model"],
            search_config,
            label="warm-start embedded approved",
        )
        approved_checkpoint_sha256 = str(warm["approved_checkpoint_sha256"])
    else:
        raise ValueError(
            "a new self-play run requires --approved-checkpoint; the warm-start "
            "candidate cannot certify itself as best_model"
        )

    model = PolicyValueNet(
        search_config.board_size,
        search_config.channels,
        search_config.residual_blocks,
    ).to(device)
    if resume_preview is None:
        model.load_state_dict(
            warm.get("candidate_model", warm.get("train_model", warm["best_model"]))
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=loop_config.learning_rate,
        weight_decay=loop_config.weight_decay,
    )
    scheduler = make_scheduler(optimizer, loop_config)

    expert_path = _dataset_path(args.expert_npz, "ddqk", resume_preview)
    tactical_path = _dataset_path(args.tactical_npz, "tactical", resume_preview)
    white_defense_path = _optional_dataset_path(
        args.white_defense_npz, "white_defense", resume_preview
    )
    white_defense_manifest_path = _white_defense_manifest_path(
        args.white_defense_manifest, resume_preview
    )
    if (white_defense_path is None) != (white_defense_manifest_path is None):
        raise ValueError(
            "--white-defense-npz and --white-defense-manifest must be provided together"
        )
    if loop_config.white_defense_quota > 0 and white_defense_path is None:
        raise ValueError(
            "positive --white-defense-quota requires an authenticated white-defense dataset"
        )

    static_sources: dict[str, StaticReplaySource] = {
        "ddqk": StaticReplaySource("ddqk", expert_path, search_config.board_size, loop_config.seed + 11),
        "tactical": StaticReplaySource(
            "tactical", tactical_path, search_config.board_size, loop_config.seed + 23
        ),
    }
    if white_defense_path is not None and white_defense_manifest_path is not None:
        static_sources["white_defense"] = WhiteDefenseReplaySource(
            white_defense_path,
            white_defense_manifest_path,
            search_config.board_size,
            loop_config.seed + 29,
        )
    manifests = [source.manifest() for source in static_sources.values()]
    if resume_preview is not None:
        expected_manifests = {
            item["name"]: item
            for item in resume_preview.get("dataset_manifest", [])
        }
        if set(expected_manifests) != set(static_sources):
            raise ValueError("resume checkpoint has an incomplete static dataset manifest")
        for item in manifests:
            expected = expected_manifests[item["name"]]
            if item["sha256"] != expected.get("sha256"):
                raise ValueError(f"resume dataset changed: {item['name']}")
            if item["name"] == "white_defense":
                for key in (
                    "provenance_manifest_sha256",
                    "manifest_payload_sha256",
                    "schema_version",
                    "source_provenance",
                    "report_sha256",
                    "provenance_generation",
                    "eval_training_prohibition",
                ):
                    if item.get(key) != expected.get(key):
                        raise ValueError(
                            f"resume white-defense provenance changed: {key}"
                        )

    replay = CircularSelfplayReplay(
        loop_config.replay_capacity,
        search_config.board_size,
        loop_config.seed + 37,
    )
    replay_dir = output_dir / "replay"
    replay_manifest: list[dict[str, object]] = []
    if resume_preview is not None:
        if resume_preview.get("replay_manifest_version") != 1:
            raise ValueError("resume checkpoint has no transactional replay manifest")
        replay_manifest = validate_replay_manifest(
            resume_preview.get("replay_manifest", []),
            checkpoint_iteration=int(resume_preview["iteration"]),
        )
        loaded = load_selfplay_chunks(
            replay_dir,
            replay,
            replay_manifest,
            int(resume_preview.get("replay_size", -1)),
        )
        # Files outside the committed manifest are crash leftovers.  They are
        # safe to remove only after the old checkpoint has been verified.
        prune_uncommitted_replay_chunks(replay_dir, replay_manifest)
    else:
        stale_chunks = sorted(replay_dir.glob("selfplay_*.npz"))
        if stale_chunks:
            raise ValueError(
                "new run output directory already contains replay chunks; "
                "use --resume or choose a clean output directory"
            )
        loaded = 0
    mixer = SourceMixer(
        {"selfplay": replay, **static_sources},
        loop_config.quotas(),
        loop_config.seed + 41,
    )
    start_iteration = 1
    global_step = 0
    if resume_path is not None:
        restored = restore_v3_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rng=rng,
            device=device,
        )
        restore_sampler_rng_state(restored, replay, static_sources, mixer)
        if any(
            not math.isclose(
                float(group["weight_decay"]),
                loop_config.weight_decay,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for group in optimizer.param_groups
        ):
            raise ValueError("restored optimizer weight_decay disagrees with saved config")
        if any(
            not math.isclose(
                float(base_lr),
                loop_config.learning_rate,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for base_lr in scheduler.base_lrs
        ):
            raise ValueError("restored scheduler base LR disagrees with saved config")
        start_iteration = int(restored["iteration"]) + 1
        global_step = int(restored["global_step"])
        logging.info(
            "resumed iteration=%d global_step=%d replay_loaded=%d",
            start_iteration - 1,
            global_step,
            loaded,
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logging.info(
        "V3 self-play start device=%s parameters=%d search_config=%s loop_config=%s",
        device,
        parameter_count,
        asdict(search_config),
        asdict(loop_config),
    )
    logging.info("dataset_manifest=%s", manifests)

    latest_path = output_dir / "latest.pt"
    if resume_path is None:
        # An atomic iteration-zero checkpoint makes startup itself resumable;
        # importantly, the optimizer is fresh and never inherited from the
        # supervised warm-start stage.
        save_v3_checkpoint(
            latest_path,
            iteration=0,
            global_step=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            search_config=search_config,
            loop_config=loop_config,
            replay_size=len(replay),
            replay_manifest=replay_manifest,
            rng=rng,
            dataset_manifest=manifests,
            parent_checkpoint_sha256=parent_sha256,
            approved_model_state=approved_state,
            approved_checkpoint_sha256=approved_checkpoint_sha256,
            metrics={"status": "initialized", "replay_loaded": loaded},
            sampler_rng_state=sampler_rng_state(replay, static_sources, mixer),
        )
    for iteration in range(start_iteration, loop_config.iterations + 1):
        if STOP_REQUESTED:
            break
        iteration_started = time.perf_counter()
        generated, selfplay_metrics = generate_v3_selfplay(
            model,
            search_config,
            loop_config,
            device,
            rng,
        )
        replay.add(generated)
        chunk_entry = save_selfplay_chunk(
            replay_dir,
            iteration,
            generated,
        )
        pending_replay_manifest = retained_replay_manifest(
            replay_manifest,
            chunk_entry,
            replay_size=len(replay),
            replay_capacity=loop_config.replay_capacity,
            max_chunks=loop_config.max_replay_chunks,
        )
        train_metrics, completed_steps = train_mixed_steps(
            model,
            optimizer,
            scheduler,
            mixer,
            loop_config,
            device,
        )
        global_step += completed_steps
        metrics: dict[str, object] = {
            "selfplay": selfplay_metrics,
            "training": train_metrics,
            "iteration_seconds": time.perf_counter() - iteration_started,
            "replay_size": len(replay),
            "replay_chunk": dict(chunk_entry),
        }
        save_v3_checkpoint(
            latest_path,
            iteration=iteration,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            search_config=search_config,
            loop_config=loop_config,
            replay_size=len(replay),
            replay_manifest=pending_replay_manifest,
            rng=rng,
            dataset_manifest=manifests,
            parent_checkpoint_sha256=parent_sha256,
            approved_model_state=approved_state,
            approved_checkpoint_sha256=approved_checkpoint_sha256,
            metrics=metrics,
            sampler_rng_state=sampler_rng_state(replay, static_sources, mixer),
        )
        replay_manifest = pending_replay_manifest
        # Pruning happens only after the new checkpoint atomically commits its
        # manifest.  A crash before this point leaves harmless extra files.
        prune_uncommitted_replay_chunks(replay_dir, replay_manifest)
        logging.info(
            "iteration=%d complete metrics=%s checkpoint=%s",
            iteration,
            json.dumps(metrics, ensure_ascii=False),
            latest_path,
        )
        if STOP_REQUESTED:
            break

    logging.info("V3 self-play stopped cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
