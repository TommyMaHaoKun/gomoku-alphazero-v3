#!/usr/bin/env python3
"""Tactical/expert warm-start stage for the Gomoku V3 model.

This is deliberately a bounded pre-training stage, not the full self-play
loop.  It raises low-ranked tactical and DDQK teacher moves before expensive
V3 self-play starts, supports masked value targets, and writes a checkpoint
that remains loadable by the desktop agent.  Authenticated white-defense
archives use candidate-restricted safe-set supervision rather than treating a
uniform safe-action label as a desired probability distribution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .train_alphazero import Config, PolicyValueNet
from .train_v3_selfplay import (
    safe_hard_negative_margin_loss,
    validate_white_defense_manifest,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DatasetPool:
    def __init__(
        self,
        path: Path,
        seed: int,
        validation_fraction: float,
        white_defense_manifest: Path | None = None,
    ):
        self.path = path.resolve()
        self.white_defense_provenance: dict[str, object] | None = None
        if white_defense_manifest is not None:
            self.white_defense_provenance = validate_white_defense_manifest(
                self.path,
                white_defense_manifest,
                board_size=19,
            )
        with np.load(self.path) as data:
            required = {"states", "policies", "values", "priority"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"{self.path}: missing arrays {missing}")
            policy_weight_key = (
                "policy_weights" if "policy_weights" in data.files else "policy_weight"
            )
            value_weight_key = (
                "value_weights" if "value_weights" in data.files else "value_weight"
            )
            if policy_weight_key not in data.files or value_weight_key not in data.files:
                raise ValueError(
                    f"{self.path}: missing policy_weight(s) or value_weight(s)"
                )
            raw_states = np.asarray(data["states"])
            try:
                states_are_finite = bool(np.all(np.isfinite(raw_states)))
            except TypeError as error:
                raise ValueError(f"{self.path}: states must be numeric") from error
            if not states_are_finite or not np.all(
                (raw_states == 0) | (raw_states == 1)
            ):
                raise ValueError(f"{self.path}: state planes must be finite binary values")
            self.states = raw_states.astype(np.uint8, copy=True)
            self.policies = data["policies"].astype(np.float32, copy=True)
            self.values = data["values"].astype(np.float32, copy=True)
            self.policy_weights = data[policy_weight_key].astype(np.float32, copy=True)
            self.value_weights = data[value_weight_key].astype(np.float32, copy=True)
            self.priority = data["priority"].astype(np.float64, copy=True)
            self.mistake_actions = (
                data["mistake_action"].astype(np.int64, copy=True)
                if "mistake_action" in data.files
                else np.full(len(self.states), -1, dtype=np.int64)
            )
            has_safe_mask = "safe_mask" in data.files
            has_candidate_mask = "candidate_mask" in data.files
            has_white_source = (
                "source" in data.files
                and data["source"].dtype.kind in "US"
                and all(
                    str(value).startswith("white_defense|")
                    for value in data["source"]
                )
            )
            if has_safe_mask != has_white_source or (
                has_safe_mask and not has_candidate_mask
            ):
                raise ValueError(
                    f"{self.path}: incomplete white-defense safe-set schema"
                )
            self.dataset_kind = "white_defense" if has_safe_mask else "ordinary"
            self.safe_masks = (
                data["safe_mask"].astype(bool, copy=True)
                if has_safe_mask
                else np.zeros_like(self.policies, dtype=bool)
            )
            self.candidate_masks = (
                data["candidate_mask"].astype(bool, copy=True)
                if has_candidate_mask
                else ~(
                    (self.states[:, 0] != 0) | (self.states[:, 1] != 0)
                ).reshape(len(self.states), -1)
            )
            self.safe_set_rows = np.full(
                len(self.states),
                has_safe_mask,
                dtype=bool,
            )
            declared_split = data["split"].copy() if "split" in data.files else None
            if "group_id" in data.files:
                groups = data["group_id"].copy()
                self.group_key = "group_id"
            elif "pair_index" in data.files:
                # DDQK games are generated in colour-swapped pairs from the
                # same opening.  Splitting by individual game leaks that
                # opening across train and validation.
                groups = data["pair_index"].astype(np.int64, copy=True)
                self.group_key = "pair_index"
            elif "game_index" in data.files:
                groups = data["game_index"].astype(np.int64, copy=True)
                self.group_key = "game_index"
            elif "source" in data.files and data["source"].dtype.kind in "US":
                groups = np.asarray(
                    [str(value).split("|", 1)[0] for value in data["source"]]
                )
                self.group_key = "source_prefix"
            else:
                groups = np.arange(len(self.states), dtype=np.int64)
                self.group_key = "sample_index"

        count = len(self.states)
        if declared_split is not None:
            if declared_split.shape != (count,):
                raise ValueError(
                    f"{self.path}: split shape {declared_split.shape}, expected {(count,)}"
                )
            split_values = set(map(str, declared_split.tolist()))
            if split_values != {"train"}:
                raise ValueError(
                    f"{self.path}: supervised training requires a pure train split; "
                    f"found {sorted(split_values)}"
                )
        expected = (count, 4, 19, 19)
        if self.states.shape != expected:
            raise ValueError(f"{self.path}: states shape {self.states.shape}, expected {expected}")
        if self.policies.shape != (count, 361):
            raise ValueError(f"{self.path}: invalid policies shape {self.policies.shape}")
        if self.safe_masks.shape != (count, 361):
            raise ValueError(f"{self.path}: invalid white-defense safe_mask shape")
        if self.candidate_masks.shape != (count, 361):
            raise ValueError(f"{self.path}: invalid white-defense candidate_mask shape")
        for name, array in (
            ("values", self.values),
            ("policy_weights", self.policy_weights),
            ("value_weights", self.value_weights),
            ("priority", self.priority),
            ("mistake_actions", self.mistake_actions),
            ("groups", groups),
        ):
            if array.shape != (count,):
                raise ValueError(f"{self.path}: {name} shape {array.shape}, expected {(count,)}")
        if not np.all((self.states == 0) | (self.states == 1)):
            raise ValueError(f"{self.path}: state planes must be binary")
        if np.any((self.states[:, 0] != 0) & (self.states[:, 1] != 0)):
            raise ValueError(f"{self.path}: current/opponent stone planes overlap")
        side_sums = self.states[:, 3].sum(axis=(1, 2))
        board_area = 19 * 19
        if np.any((side_sums != 0) & (side_sums != board_area)):
            raise ValueError(f"{self.path}: side-to-move plane must be uniform")
        own_counts = self.states[:, 0].sum(axis=(1, 2))
        opponent_counts = self.states[:, 1].sum(axis=(1, 2))
        black_to_move = side_sums == board_area
        reachable = np.where(
            black_to_move,
            own_counts == opponent_counts,
            opponent_counts == own_counts + 1,
        )
        if not np.all(reachable):
            raise ValueError(f"{self.path}: stone counts are not reachable")
        occupied = (self.states[:, 0] != 0) | (self.states[:, 1] != 0)
        occupied_counts = occupied.sum(axis=(1, 2))
        last_counts = self.states[:, 2].sum(axis=(1, 2))
        expected_last = (occupied_counts > 0).astype(last_counts.dtype)
        if not np.array_equal(last_counts, expected_last):
            raise ValueError(f"{self.path}: invalid last-move plane")
        if np.any((self.states[:, 2] != 0) & (self.states[:, 1] == 0)):
            raise ValueError(f"{self.path}: last move must mark an opponent stone")
        if not np.all(np.isfinite(self.policies)) or np.any(self.policies < 0):
            raise ValueError(f"{self.path}: policies must be finite and non-negative")
        policy_sums = self.policies.sum(axis=1)
        if not np.allclose(policy_sums, 1.0, atol=2e-3):
            raise ValueError(f"{self.path}: policy rows must sum to one")
        if np.any(
            np.where(occupied.reshape(count, -1), self.policies, 0.0) > 1e-6
        ):
            raise ValueError(f"{self.path}: policy assigns mass to an occupied point")
        if self.dataset_kind == "white_defense":
            if not np.array_equal(self.policies > 0, self.safe_masks):
                raise ValueError(
                    f"{self.path}: white-defense policy support must equal safe_mask"
                )
            if np.any(self.safe_masks & ~self.candidate_masks) or np.any(
                ~self.candidate_masks.any(axis=1)
            ):
                raise ValueError(
                    f"{self.path}: white-defense safe_mask must lie inside candidate_mask"
                )
            if np.any(self.values != 0) or np.any(self.value_weights != 0):
                raise ValueError(
                    f"{self.path}: white-defense curriculum must remain policy-only"
                )
        elif self.white_defense_provenance is not None:
            raise ValueError(
                f"{self.path}: authenticated white-defense archive lacks its safe-set schema"
            )
        if not np.all(np.isfinite(self.values)) or np.any(np.abs(self.values) > 1.0 + 1e-6):
            raise ValueError(f"{self.path}: values must be finite and in [-1, 1]")
        if np.any(self.policy_weights < 0) or np.any(self.value_weights < 0):
            raise ValueError(f"{self.path}: loss weights cannot be negative")
        if np.any(self.priority <= 0) or not np.all(np.isfinite(self.priority)):
            raise ValueError(f"{self.path}: priorities must be finite and positive")
        mistake_rows = self.mistake_actions >= 0
        if np.any(self.mistake_actions < -1) or np.any(self.mistake_actions >= 361):
            raise ValueError(f"{self.path}: mistake_action must be -1 or a legal action")
        if np.any(mistake_rows & occupied.reshape(count, -1)[
            np.arange(count), np.maximum(self.mistake_actions, 0)
        ]):
            raise ValueError(f"{self.path}: mistake_action points to an occupied cell")
        if np.any(
            mistake_rows
            & (
                self.policies[
                    np.arange(count), np.maximum(self.mistake_actions, 0)
                ]
                > 1e-6
            )
        ):
            raise ValueError(
                f"{self.path}: mistake_action must differ from every teacher target"
            )
        if self.dataset_kind == "white_defense" and np.any(mistake_rows):
            raise ValueError(
                f"{self.path}: white-defense safe-set rows cannot carry mistake actions"
            )
        if not 0.0 <= validation_fraction < 0.5:
            raise ValueError("validation fraction must be in [0, 0.5)")

        rng = np.random.default_rng(seed)
        unique_groups = np.unique(groups)
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)
        validation_groups = (
            max(1, round(len(shuffled) * validation_fraction))
            if validation_fraction > 0 and len(shuffled) > 1
            else 0
        )
        validation_set = set(shuffled[:validation_groups].tolist())
        validation_mask = np.fromiter(
            (group in validation_set for group in groups.tolist()),
            dtype=bool,
            count=count,
        )
        self.validation_indices = np.flatnonzero(validation_mask)
        self.training_indices = np.flatnonzero(~validation_mask)
        if not len(self.training_indices):
            raise ValueError(f"{self.path}: validation split left no training samples")
        training_priority = self.priority[self.training_indices]
        training_groups = groups[self.training_indices]
        _, inverse = np.unique(training_groups, return_inverse=True)
        group_priority_totals = np.bincount(
            inverse,
            weights=training_priority,
            minlength=int(inverse.max()) + 1,
        )
        group_balanced_priority = training_priority / group_priority_totals[inverse]
        self.training_probabilities = (
            group_balanced_priority / group_balanced_priority.sum()
        )
        self.group_count = int(len(unique_groups))
        self.rng = np.random.default_rng(seed + 1)

    def sample(self, count: int) -> dict[str, np.ndarray]:
        chosen = self.rng.choice(
            self.training_indices,
            size=count,
            replace=True,
            p=self.training_probabilities,
        )
        return self.take(chosen)

    def take(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "states": self.states[indices],
            "policies": self.policies[indices],
            "values": self.values[indices],
            "policy_weights": self.policy_weights[indices],
            "value_weights": self.value_weights[indices],
            "safe_masks": self.safe_masks[indices],
            "candidate_masks": self.candidate_masks[indices],
            "safe_set_rows": self.safe_set_rows[indices],
            "mistake_actions": self.mistake_actions[indices],
        }

    def manifest(self) -> dict[str, object]:
        result = {
            "path": str(self.path),
            "sha256": sha256_file(self.path),
            "samples": len(self.states),
            "training_samples": len(self.training_indices),
            "validation_samples": len(self.validation_indices),
            "split_group_key": self.group_key,
            "groups": self.group_count,
            "sampling": "equal_group_mass_then_position_priority",
            "dataset_kind": self.dataset_kind,
            "mistake_rows": int(np.sum(self.mistake_actions >= 0)),
        }
        if self.white_defense_provenance is not None:
            result.update(
                {
                    "provenance_manifest_path": self.white_defense_provenance[
                        "manifest_path"
                    ],
                    "provenance_manifest_sha256": self.white_defense_provenance[
                        "manifest_sha256"
                    ],
                    "manifest_payload_sha256": self.white_defense_provenance[
                        "manifest_payload_sha256"
                    ],
                    "schema_version": self.white_defense_provenance[
                        "schema_version"
                    ],
                    "source_provenance": self.white_defense_provenance["source"],
                    "report_sha256": self.white_defense_provenance["report_sha256"],
                    "provenance_generation": self.white_defense_provenance[
                        "provenance_generation"
                    ],
                    "eval_training_prohibition": self.white_defense_provenance[
                        "eval_training_prohibition"
                    ],
                }
            )
        return result


def allocate_counts(total: int, weights: np.ndarray) -> np.ndarray:
    exact = total * weights / weights.sum()
    counts = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - counts))[: total - int(counts.sum())]:
        counts[index] += 1
    return counts


def concatenate_batch(parts: list[dict[str, np.ndarray]], rng: np.random.Generator) -> dict[str, np.ndarray]:
    batch = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
    order = rng.permutation(len(batch["states"]))
    return {key: value[order] for key, value in batch.items()}


def _transform_actions_d4(actions: np.ndarray, symmetries: np.ndarray) -> np.ndarray:
    transformed = actions.copy()
    valid = transformed >= 0
    x = np.where(valid, transformed % 19, 0)
    y = np.where(valid, transformed // 19, 0)
    reflected = symmetries >= 4
    x = np.where(reflected & valid, 18 - x, x)
    rotations = symmetries % 4
    for rotation in (1, 2, 3):
        selected = valid & (rotations >= rotation)
        old_x = x.copy()
        x = np.where(selected, 18 - y, x)
        y = np.where(selected, old_x, y)
    transformed[valid] = y[valid] * 19 + x[valid]
    return transformed


def d4_augment_batch(
    batch: dict[str, np.ndarray], symmetries: np.ndarray
) -> dict[str, np.ndarray]:
    symmetries = np.asarray(symmetries, dtype=np.int8)
    if symmetries.shape != (len(batch["states"]),):
        raise ValueError("one D4 symmetry is required per batch row")
    if np.any(symmetries < 0) or np.any(symmetries > 7):
        raise ValueError("D4 symmetries must be integers in [0, 7]")
    result = {key: value.copy() for key, value in batch.items()}
    spatial_keys = ("states", "policies", "safe_masks", "candidate_masks")
    for symmetry in range(1, 8):
        indices = np.flatnonzero(symmetries == symmetry)
        if not len(indices):
            continue
        reflected = symmetry >= 4
        rotations = symmetry % 4
        for key in spatial_keys:
            source = batch[key][indices]
            original_shape = source.shape
            if key != "states":
                source = source.reshape(-1, 19, 19)
            if reflected:
                source = np.flip(source, axis=-1)
            if rotations:
                source = np.rot90(source, k=-rotations, axes=(-2, -1))
            result[key][indices] = np.ascontiguousarray(source).reshape(original_shape)
    result["mistake_actions"] = _transform_actions_d4(
        batch["mistake_actions"], symmetries
    )
    return result


def weighted_loss(
    logits: torch.Tensor,
    predicted_values: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    policy_weights: torch.Tensor,
    value_weights: torch.Tensor,
    value_loss_scale: float = 1.0,
    safe_set_rows: torch.Tensor | None = None,
    candidate_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probabilities = F.log_softmax(logits, dim=1)
    policy_per_sample = -(target_policies * log_probabilities).sum(dim=1)
    if safe_set_rows is not None:
        safe_set_rows = safe_set_rows.to(device=logits.device, dtype=torch.bool)
        if safe_set_rows.shape != (len(logits),):
            raise ValueError("safe_set_rows must have one flag per policy row")
        if bool(safe_set_rows.any()):
            safe_actions = target_policies > 0
            if bool(torch.any(safe_set_rows & ~safe_actions.any(dim=1))):
                raise ValueError("every safe-set row must contain a safe action")
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
            policy_per_sample = torch.where(
                safe_set_rows,
                candidate_log_mass - safe_log_mass,
                policy_per_sample,
            )
    policy_loss = (policy_per_sample * policy_weights).sum() / policy_weights.sum().clamp_min(1e-8)
    value_per_sample = (predicted_values - target_values).square()
    value_loss = (value_per_sample * value_weights).sum() / value_weights.sum().clamp_min(1e-8)
    return policy_loss + value_loss_scale * value_loss, policy_loss, value_loss


def policy_distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """Return KL(teacher || student) in a float32 log-probability space.

    Explicitly converting logits to float32 keeps this stable when the caller
    runs the model forward pass under bf16 autocast.  ``log_target=True`` also
    avoids materialising a low-precision teacher softmax.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student and teacher policy logits must have the same shape; "
            f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    student_log_probs = F.log_softmax(student_logits.float(), dim=1)
    teacher_log_probs = F.log_softmax(teacher_logits.detach().float(), dim=1)
    return F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="batchmean",
        log_target=True,
    )


def mistake_hard_negative_margin_loss(
    logits: torch.Tensor,
    target_policies: torch.Tensor,
    policy_weights: torch.Tensor,
    mistake_actions: torch.Tensor,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    """Require the teacher action to outrank the student's recorded mistake."""

    if logits.shape != target_policies.shape:
        raise ValueError("target_policies must match logits")
    if policy_weights.shape != (len(logits),):
        raise ValueError("policy_weights must have one value per row")
    if mistake_actions.shape != (len(logits),):
        raise ValueError("mistake_actions must have one action per row")
    if margin < 0 or not math.isfinite(margin):
        raise ValueError("margin must be finite and non-negative")
    mistake_actions = mistake_actions.to(device=logits.device, dtype=torch.long)
    rows = mistake_actions >= 0
    if not bool(rows.any()):
        return logits.sum() * 0.0
    selected = mistake_actions[rows]
    if bool(torch.any(selected >= logits.shape[1])):
        raise ValueError("mistake action lies outside policy logits")
    teacher_actions = target_policies[rows].argmax(dim=1)
    if bool(torch.any(teacher_actions == selected)):
        raise ValueError("mistake action cannot equal the teacher action")
    row_index = torch.arange(len(selected), device=logits.device)
    teacher_logits = logits[rows][row_index, teacher_actions]
    mistake_logits = logits[rows][row_index, selected]
    losses = F.relu(float(margin) - teacher_logits + mistake_logits)
    weights = policy_weights[rows]
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def hold_batchnorm_fixed(model: nn.Module) -> None:
    """Keep pretrained BN statistics stable on the small synthetic curriculum."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def configure_selective_freeze(
    model: PolicyValueNet,
    train_last_residual_blocks: int,
) -> None:
    """Freeze stem/early tower while leaving heads and the last N blocks open."""

    block_count = len(model.tower)
    if not 0 <= train_last_residual_blocks <= block_count:
        raise ValueError(
            "train_last_residual_blocks must be between 0 and the tower depth"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in model.stem.parameters():
        parameter.requires_grad_(False)
    frozen_blocks = block_count - train_last_residual_blocks
    for index, block in enumerate(model.tower):
        requires_grad = index >= frozen_blocks
        for parameter in block.parameters():
            parameter.requires_grad_(requires_grad)


def unfreeze_all_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)


@torch.inference_mode()
def evaluate(
    model: PolicyValueNet,
    pools: list[DatasetPool],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    results: list[dict[str, object]] = []
    for pool in pools:
        indices = pool.validation_indices
        if not len(indices):
            indices = pool.training_indices[: min(512, len(pool.training_indices))]
        correct = 0
        global_top1_in_candidate = 0
        policy_loss_sum = 0.0
        policy_weight_sum = 0.0
        safe_probability_sum = 0.0
        value_absolute_error_sum = 0.0
        value_square_error_sum = 0.0
        value_weight_sum = 0.0
        mistake_rows = 0
        teacher_over_mistake = 0
        teacher_over_mistake_margin_one = 0
        teacher_minus_mistake_logit_sum = 0.0
        for start in range(0, len(indices), 512):
            batch = pool.take(indices[start : start + 512])
            states = torch.from_numpy(batch["states"]).to(device=device, dtype=torch.float32)
            targets = torch.from_numpy(batch["policies"]).to(device=device)
            weights = torch.from_numpy(batch["policy_weights"]).to(device=device)
            target_values = torch.from_numpy(batch["values"]).to(device=device)
            value_weights = torch.from_numpy(batch["value_weights"]).to(device=device)
            safe_masks = torch.from_numpy(batch["safe_masks"]).to(
                device=device, dtype=torch.bool
            )
            candidate_masks = torch.from_numpy(batch["candidate_masks"]).to(
                device=device, dtype=torch.bool
            )
            safe_set_rows = torch.from_numpy(batch["safe_set_rows"]).to(
                device=device, dtype=torch.bool
            )
            mistake_actions = torch.from_numpy(batch["mistake_actions"]).to(
                device=device, dtype=torch.long
            )
            logits, predicted_values = model(states)
            global_chosen = logits.argmax(dim=1)
            row = torch.arange(len(global_chosen), device=device)
            log_probabilities = F.log_softmax(logits, dim=1)
            has_mistake = mistake_actions >= 0
            if bool(has_mistake.any()):
                teacher_actions = targets.argmax(dim=1)
                mistake_rows_batch = row[has_mistake]
                margins = (
                    logits[mistake_rows_batch, teacher_actions[has_mistake]]
                    - logits[mistake_rows_batch, mistake_actions[has_mistake]]
                )
                mistake_rows += int(has_mistake.sum())
                teacher_over_mistake += int((margins > 0).sum())
                teacher_over_mistake_margin_one += int((margins >= 1.0).sum())
                teacher_minus_mistake_logit_sum += float(margins.sum())
            if pool.dataset_kind == "white_defense":
                if not bool(safe_set_rows.all()):
                    raise ValueError("white-defense evaluation batch lost its source flags")
                safe_log_mass = torch.logsumexp(
                    log_probabilities.masked_fill(~safe_masks, -torch.inf),
                    dim=1,
                )
                candidate_log_mass = torch.logsumexp(
                    log_probabilities.masked_fill(~candidate_masks, -torch.inf),
                    dim=1,
                )
                per_sample = candidate_log_mass - safe_log_mass
                safe_probability_sum += float(
                    (safe_log_mass - candidate_log_mass).exp().sum()
                )
                candidate_chosen = logits.masked_fill(
                    ~candidate_masks, -torch.inf
                ).argmax(dim=1)
                correct += int(safe_masks[row, candidate_chosen].sum())
                global_top1_in_candidate += int(
                    candidate_masks[row, global_chosen].sum()
                )
            else:
                if bool(safe_set_rows.any()):
                    raise ValueError("ordinary evaluation batch has safe-set source flags")
                target_max = targets.max(dim=1).values
                correct += int(
                    (targets[row, global_chosen] >= target_max - 1e-6).sum()
                )
                per_sample = -(targets * log_probabilities).sum(dim=1)
            policy_loss_sum += float((per_sample * weights).sum())
            policy_weight_sum += float(weights.sum())
            value_errors = predicted_values - target_values
            value_absolute_error_sum += float(
                (value_errors.abs() * value_weights).sum()
            )
            value_square_error_sum += float(
                (value_errors.square() * value_weights).sum()
            )
            value_weight_sum += float(value_weights.sum())
        result: dict[str, object] = {
            "dataset": str(pool.path),
            "dataset_kind": pool.dataset_kind,
            "samples": len(indices),
            "policy_loss": policy_loss_sum / max(policy_weight_sum, 1e-8),
            "value_mae": (
                value_absolute_error_sum / value_weight_sum
                if value_weight_sum > 0
                else None
            ),
            "value_mse": (
                value_square_error_sum / value_weight_sum
                if value_weight_sum > 0
                else None
            ),
            "value_weight_sum": value_weight_sum,
        }
        if pool.dataset_kind == "white_defense":
            safe_probability_mass = min(
                1.0,
                max(0.0, safe_probability_sum / max(len(indices), 1)),
            )
            result.update(
                {
                    "policy_metric": "safe_set_probability_mass",
                    "probability_scope": "renormalized_within_candidate_mask",
                    "top1_in_safe_set": correct / max(len(indices), 1),
                    "global_top1_in_candidate": global_top1_in_candidate
                    / max(len(indices), 1),
                    "safe_probability_mass": safe_probability_mass,
                    "unsafe_mass": max(0.0, 1.0 - safe_probability_mass),
                }
            )
        else:
            result["policy_top1"] = correct / max(len(indices), 1)
            if mistake_rows:
                result.update(
                    {
                        "mistake_rows": mistake_rows,
                        "teacher_over_mistake_rate": teacher_over_mistake
                        / mistake_rows,
                        "teacher_over_mistake_margin_one_rate": (
                            teacher_over_mistake_margin_one / mistake_rows
                        ),
                        "teacher_minus_mistake_logit_mean": (
                            teacher_minus_mistake_logit_sum / mistake_rows
                        ),
                    }
                )
        results.append(result)
    return {"datasets": results}


def save_checkpoint(
    path: Path,
    *,
    parent: dict[str, object],
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: Config,
    step: int,
    parent_sha256: str,
    manifests: list[dict[str, object]],
    metrics: dict[str, object],
    value_loss_scale: float,
    value_distill_scale: float,
    policy_distill_scale: float,
    freeze_trunk_steps: int,
    train_last_residual_blocks_during_freeze: int,
    safe_hard_negative_scale: float = 0.0,
    safe_hard_negative_margin: float = 1.0,
    mistake_hard_negative_scale: float = 0.0,
    mistake_hard_negative_margin: float = 1.0,
    random_d4_augmentation: bool = False,
    white_defense_union_audit: dict[str, object] | None = None,
) -> None:
    payload = {
        "format_version": 3,
        "iteration": int(parent.get("iteration", 0)),
        "global_step": step,
        "v3_stage": "tactical_expert_warmstart",
        "config": vars(config),
        "warmstart_config": {
            "value_loss_scale": float(value_loss_scale),
            "value_distill_scale": float(value_distill_scale),
            "policy_distill_scale": float(policy_distill_scale),
            "safe_hard_negative_scale": float(safe_hard_negative_scale),
            "safe_hard_negative_margin": float(safe_hard_negative_margin),
            "mistake_hard_negative_scale": float(mistake_hard_negative_scale),
            "mistake_hard_negative_margin": float(mistake_hard_negative_margin),
            "random_d4_augmentation": bool(random_d4_augmentation),
            "freeze_trunk_steps": int(freeze_trunk_steps),
            "train_last_residual_blocks_during_freeze": int(
                train_last_residual_blocks_during_freeze
            ),
        },
        "model_spec": {
            "board_size": config.board_size,
            "channels": config.channels,
            "residual_blocks": config.residual_blocks,
            "input_planes": 4,
        },
        "train_model": copy.deepcopy(model.state_dict()),
        "best_model": copy.deepcopy(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "parent_checkpoint_sha256": parent_sha256,
        "dataset_manifest": manifests,
        "white_defense_union_audit": dict(white_defense_union_audit or {}),
        "validation": metrics,
        "saved_at_unix": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resolve_dataset_mix(
    ordinary_paths: Sequence[Path],
    ordinary_weights: Sequence[float] | None,
    white_paths: Sequence[Path] | None,
    white_manifests: Sequence[Path] | None,
    white_weights: Sequence[float] | None,
) -> tuple[list[tuple[Path, Path | None]], np.ndarray]:
    """Pair every white archive with one manifest and one optional weight."""

    normal = list(ordinary_paths)
    normal_weights = np.asarray(
        list(ordinary_weights) if ordinary_weights is not None else [1.0] * len(normal),
        dtype=np.float64,
    )
    if (
        len(normal_weights) != len(normal)
        or np.any(~np.isfinite(normal_weights))
        or np.any(normal_weights <= 0)
    ):
        raise ValueError("provide one positive --dataset-weight per --dataset")

    white = list(white_paths or [])
    manifests = list(white_manifests or [])
    if len(white) != len(manifests):
        raise ValueError(
            "provide one --white-defense-manifest per --white-defense-npz"
        )
    resolved_white_weights = np.asarray(
        list(white_weights) if white_weights is not None else [1.0] * len(white),
        dtype=np.float64,
    )
    if (
        len(resolved_white_weights) != len(white)
        or np.any(~np.isfinite(resolved_white_weights))
        or np.any(resolved_white_weights <= 0)
    ):
        raise ValueError(
            "provide one positive --white-defense-weight per --white-defense-npz"
        )
    specs = [(path, None) for path in normal] + list(zip(white, manifests))
    weights = np.concatenate([normal_weights, resolved_white_weights])
    return specs, weights


def validate_white_defense_pool_union(
    pools: Sequence[DatasetPool],
) -> dict[str, object]:
    """Reject contradictory safe-set supervision across white sources.

    Individual manifests authenticate one archive at a time.  This additional
    gate binds the actual training semantics across all repeatable white inputs:
    the same encoded state must have the same candidate and safe masks wherever
    it appears.
    """

    labels_by_state: dict[str, tuple[bytes, bytes]] = {}
    source_by_state: dict[str, str] = {}
    records = 0
    duplicate_rows = 0
    sources = 0
    for pool in pools:
        if pool.dataset_kind != "white_defense":
            continue
        sources += 1
        records += len(pool.states)
        for index, state in enumerate(pool.states):
            state_hash = hashlib.sha256(
                np.ascontiguousarray(state).tobytes()
            ).hexdigest()
            signature = (
                np.ascontiguousarray(pool.candidate_masks[index]).tobytes(),
                np.ascontiguousarray(pool.safe_masks[index]).tobytes(),
            )
            previous = labels_by_state.setdefault(state_hash, signature)
            if previous != signature:
                raise ValueError(
                    "cross-source white-defense conflict for state "
                    f"{state_hash}: {source_by_state[state_hash]} versus {pool.path}"
                )
            if state_hash in source_by_state:
                duplicate_rows += 1
            else:
                source_by_state[state_hash] = str(pool.path)
    return {
        "sources": sources,
        "records": records,
        "unique_states": len(labels_by_state),
        "consistent_duplicate_rows": duplicate_rows,
        "conflicts": 0,
        "signature": "sha256(encoded_state)->(candidate_mask,safe_mask)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--dataset-weight", type=float, action="append")
    parser.add_argument(
        "--white-defense-npz",
        type=Path,
        action="append",
        help="repeatable pure-train white-defense NPZ; each needs one manifest",
    )
    parser.add_argument(
        "--white-defense-manifest",
        type=Path,
        action="append",
        help="repeat once per white NPZ; JSON requires its adjacent .sha256 sidecar",
    )
    parser.add_argument(
        "--white-defense-weight",
        type=float,
        action="append",
        help="repeat once per white-defense NPZ (default for each: 1)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument(
        "--random-d4-augmentation",
        action="store_true",
        help="apply an independent random rotation/reflection to every training row",
    )
    parser.add_argument("--freeze-trunk-steps", type=int, default=0)
    parser.add_argument(
        "--train-last-residual-blocks-during-freeze",
        type=int,
        default=0,
        help=(
            "while --freeze-trunk-steps is active, keep the last N residual "
            "blocks trainable with the policy/value heads (default: 0)"
        ),
    )
    parser.add_argument(
        "--value-loss-scale",
        type=float,
        default=1.0,
        help="set to 0 for a policy-only warm-start that preserves the value head",
    )
    parser.add_argument(
        "--value-distill-scale",
        type=float,
        default=0.0,
        help="MSE weight that keeps values close to the frozen parent model",
    )
    parser.add_argument(
        "--policy-distill-scale",
        type=float,
        default=0.0,
        help="KL weight that keeps policy probabilities close to the frozen parent model",
    )
    parser.add_argument(
        "--safe-hard-negative-scale",
        type=float,
        default=0.0,
        help=(
            "weight for candidate-restricted safe-vs-unsafe hard-negative "
            "margin loss on white-defense rows (default: 0, disabled)"
        ),
    )
    parser.add_argument(
        "--safe-hard-negative-margin",
        type=float,
        default=1.0,
        help=(
            "required logit lead of the best safe action over the hardest "
            "unsafe candidate when the margin loss is enabled (default: 1)"
        ),
    )
    parser.add_argument(
        "--mistake-hard-negative-scale",
        type=float,
        default=0.0,
        help=(
            "weight for teacher-vs-recorded-mistake logit margin loss "
            "(default: 0, disabled)"
        ),
    )
    parser.add_argument(
        "--mistake-hard-negative-margin",
        type=float,
        default=1.0,
        help="required teacher-logit lead over each recorded mistake",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--update-batchnorm",
        action="store_true",
        help="update BatchNorm on curriculum data (disabled by default to avoid drift)",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    if args.steps <= 0 or args.batch_size <= 0:
        raise SystemExit("steps and batch size must be positive")
    if args.freeze_trunk_steps < 0:
        raise SystemExit("--freeze-trunk-steps cannot be negative")
    if (
        args.value_loss_scale < 0
        or args.value_distill_scale < 0
        or args.policy_distill_scale < 0
        or args.safe_hard_negative_scale < 0
        or args.mistake_hard_negative_scale < 0
    ):
        raise SystemExit("value and policy loss scales cannot be negative")
    if not math.isfinite(args.safe_hard_negative_scale):
        raise SystemExit("--safe-hard-negative-scale must be finite")
    if not math.isfinite(args.mistake_hard_negative_scale):
        raise SystemExit("--mistake-hard-negative-scale must be finite")
    if (
        args.safe_hard_negative_margin < 0
        or not math.isfinite(args.safe_hard_negative_margin)
    ):
        raise SystemExit("--safe-hard-negative-margin must be finite and non-negative")
    if (
        args.mistake_hard_negative_margin < 0
        or not math.isfinite(args.mistake_hard_negative_margin)
    ):
        raise SystemExit(
            "--mistake-hard-negative-margin must be finite and non-negative"
        )
    if args.eval_every <= 0:
        raise SystemExit("--eval-every must be positive")
    try:
        dataset_specs, weights = resolve_dataset_mix(
            args.dataset,
            args.dataset_weight,
            args.white_defense_npz,
            args.white_defense_manifest,
            args.white_defense_weight,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    counts = allocate_counts(args.batch_size, weights)
    if np.any(counts == 0):
        raise SystemExit("batch size is too small for the requested dataset mix")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    init_path = args.init_checkpoint.resolve()
    parent = torch.load(init_path, map_location="cpu", weights_only=False)
    config = Config(**parent["config"])
    model = PolicyValueNet(config.board_size, config.channels, config.residual_blocks).to(device)
    if not 0 <= args.train_last_residual_blocks_during_freeze <= len(model.tower):
        raise SystemExit(
            "--train-last-residual-blocks-during-freeze exceeds the model tower depth"
        )
    model.load_state_dict(parent.get("best_model", parent["train_model"]))
    teacher: PolicyValueNet | None = None
    if args.value_distill_scale or args.policy_distill_scale:
        teacher = copy.deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    pools = [
        DatasetPool(
            path,
            args.seed + 1009 * index,
            args.validation_fraction,
            white_defense_manifest=manifest,
        )
        for index, (path, manifest) in enumerate(dataset_specs)
    ]
    if any(
        pool.dataset_kind == "white_defense"
        for pool in pools[: len(args.dataset)]
    ):
        raise SystemExit(
            "white-defense data must use --white-defense-npz and its authenticated manifest"
        )
    try:
        white_defense_union_audit = validate_white_defense_pool_union(pools)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.safe_hard_negative_scale and not any(
        pool.dataset_kind == "white_defense" for pool in pools
    ):
        raise SystemExit(
            "--safe-hard-negative-scale requires authenticated white-defense data"
        )
    if args.mistake_hard_negative_scale and not any(
        np.any(pool.mistake_actions >= 0) for pool in pools
    ):
        raise SystemExit(
            "--mistake-hard-negative-scale requires a dataset with mistake_action"
        )
    manifests = [pool.manifest() for pool in pools]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=config.weight_decay,
    )

    def schedule(step: int) -> float:
        if step < args.warmup_steps:
            return max(step + 1, 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        minimum = args.min_learning_rate / args.learning_rate
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if args.freeze_trunk_steps:
        configure_selective_freeze(
            model,
            args.train_last_residual_blocks_during_freeze,
        )
    if not args.update_batchnorm:
        hold_batchnorm_fixed(model)

    metrics = evaluate(model, pools, device)
    print(json.dumps({"step": 0, "validation": metrics}, ensure_ascii=False), flush=True)
    for step in range(1, args.steps + 1):
        if step == args.freeze_trunk_steps + 1:
            unfreeze_all_parameters(model)
        model.train()
        if not args.update_batchnorm:
            hold_batchnorm_fixed(model)
        batch = concatenate_batch(
            [pool.sample(int(count)) for pool, count in zip(pools, counts)],
            rng,
        )
        if args.random_d4_augmentation:
            batch = d4_augment_batch(
                batch,
                rng.integers(0, 8, size=len(batch["states"]), dtype=np.int8),
            )
        states = torch.from_numpy(batch["states"]).to(device=device, dtype=torch.float32)
        policies = torch.from_numpy(batch["policies"]).to(device=device)
        values = torch.from_numpy(batch["values"]).to(device=device)
        policy_weights = torch.from_numpy(batch["policy_weights"]).to(device=device)
        value_weights = torch.from_numpy(batch["value_weights"]).to(device=device)
        safe_set_rows = torch.from_numpy(batch["safe_set_rows"]).to(
            device=device, dtype=torch.bool
        )
        candidate_masks = torch.from_numpy(batch["candidate_masks"]).to(
            device=device, dtype=torch.bool
        )
        mistake_actions = torch.from_numpy(batch["mistake_actions"]).to(
            device=device, dtype=torch.long
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
                args.value_loss_scale,
                safe_set_rows=safe_set_rows,
                candidate_masks=candidate_masks,
            )
            if args.safe_hard_negative_scale:
                safe_margin_loss = safe_hard_negative_margin_loss(
                    logits,
                    policies,
                    policy_weights,
                    safe_set_rows,
                    candidate_masks,
                    margin=args.safe_hard_negative_margin,
                )
                loss = loss + args.safe_hard_negative_scale * safe_margin_loss
            else:
                safe_margin_loss = predicted_values.new_zeros(())
            if args.mistake_hard_negative_scale:
                mistake_margin_loss = mistake_hard_negative_margin_loss(
                    logits,
                    policies,
                    policy_weights,
                    mistake_actions,
                    margin=args.mistake_hard_negative_margin,
                )
                loss = loss + args.mistake_hard_negative_scale * mistake_margin_loss
            else:
                mistake_margin_loss = predicted_values.new_zeros(())
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits, teacher_values = teacher(states)
                if args.value_distill_scale:
                    value_distill_loss = F.mse_loss(predicted_values, teacher_values)
                    loss = loss + args.value_distill_scale * value_distill_loss
                else:
                    value_distill_loss = predicted_values.new_zeros(())
                if args.policy_distill_scale:
                    policy_distill_loss = policy_distillation_kl(logits, teacher_logits)
                    loss = loss + args.policy_distill_scale * policy_distill_loss
                else:
                    policy_distill_loss = predicted_values.new_zeros(())
            else:
                value_distill_loss = predicted_values.new_zeros(())
                policy_distill_loss = predicted_values.new_zeros(())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 25 == 0:
            print(
                f"step={step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"policy={float(policy_loss.detach()):.4f} "
                f"value={float(value_loss.detach()):.4f} "
                f"value_kd={float(value_distill_loss.detach()):.4f} "
                f"policy_kd={float(policy_distill_loss.detach()):.4f} "
                f"safe_margin={float(safe_margin_loss.detach()):.4f} "
                f"mistake_margin={float(mistake_margin_loss.detach()):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, pools, device)
            print(json.dumps({"step": step, "validation": metrics}, ensure_ascii=False), flush=True)
            save_checkpoint(
                args.output.resolve(),
                parent=parent,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                step=step,
                parent_sha256=sha256_file(init_path),
                manifests=manifests,
                metrics=metrics,
                value_loss_scale=args.value_loss_scale,
                value_distill_scale=args.value_distill_scale,
                policy_distill_scale=args.policy_distill_scale,
                safe_hard_negative_scale=args.safe_hard_negative_scale,
                safe_hard_negative_margin=args.safe_hard_negative_margin,
                mistake_hard_negative_scale=args.mistake_hard_negative_scale,
                mistake_hard_negative_margin=args.mistake_hard_negative_margin,
                random_d4_augmentation=args.random_d4_augmentation,
                freeze_trunk_steps=args.freeze_trunk_steps,
                train_last_residual_blocks_during_freeze=(
                    args.train_last_residual_blocks_during_freeze
                ),
                white_defense_union_audit=white_defense_union_audit,
            )

    print(f"checkpoint={args.output.resolve()}")


if __name__ == "__main__":
    main()
