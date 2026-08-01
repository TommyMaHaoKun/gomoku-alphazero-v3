"""Promote the Round7 candidate only after every authenticated hard gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .paired_model_arena import canonical_sha256
from .v3_candidate_gate import (
    CANDIDATE_STAGE,
    CHAMPION_STAGE,
    FORMAT_VERSION,
    GateError,
    _atomic_no_clobber_torch_save,
    _cpu_state_copy,
    _read_verified_checkpoint,
    _states_equal,
    _utc_now,
    _validated_config_and_state,
    sha256_file,
)


def read_sidecar_verified(path: Path) -> tuple[Any, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise GateError(f"missing SHA256 sidecar for {path}")
    expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
    if digest != expected:
        raise GateError(f"SHA256 sidecar mismatch for {path}")
    return json.loads(path.read_text(encoding="utf-8")), digest


def promote(
    candidate_path: Path,
    candidate_sha256: str,
    static_summary_path: Path,
    rapfi_final_path: Path,
    direct_arena_path: Path,
    output_path: Path,
    *,
    candidate_name: str = "round7_a035",
) -> dict[str, Any]:
    candidate, candidate_sha = _read_verified_checkpoint(
        candidate_path.resolve(), candidate_sha256, label="Round7 candidate"
    )
    if (
        candidate.get("format_version") != FORMAT_VERSION
        or candidate.get("v3_stage") != CANDIDATE_STAGE
        or candidate.get("approval_status") != "not_approved"
        or candidate.get("is_approved") is not False
    ):
        raise GateError("candidate is not an immutable unapproved evaluation checkpoint")
    for key in ("best_model", "candidate_model", "train_model"):
        if key not in candidate:
            raise GateError(f"candidate is missing {key}")
    if not _states_equal(
        candidate["best_model"], candidate["candidate_model"],
        left_label="best_model", right_label="candidate_model",
    ) or not _states_equal(
        candidate["best_model"], candidate["train_model"],
        left_label="best_model", right_label="train_model",
    ):
        raise GateError("candidate model states differ")
    config, state = _validated_config_and_state(
        candidate, "best_model", label="Round7 candidate"
    )

    static, static_sha = read_sidecar_verified(static_summary_path)
    row = static.get(candidate_name)
    if not isinstance(row, dict) or row.get("checkpoint_sha256") != candidate_sha:
        raise GateError("static summary is not bound to the candidate")
    if not (
        row.get("hard_gate_passed") is True
        and int(row.get("raw_tactics", 0)) >= 47
        and int(row.get("deployed_tactics", 0)) >= 48
        and int(row.get("white_safe_count", 0)) >= 16
        and float(row.get("white_safe_probability_mass", 0.0)) >= 0.7707617002141821
    ):
        raise GateError("Round7 static hard gate failed")

    rapfi, rapfi_sha = read_sidecar_verified(rapfi_final_path)
    if rapfi.get("candidate_checkpoint_sha256") != candidate_sha:
        raise GateError("Rapfi report is not bound to the candidate")
    candidate_colors = rapfi.get("candidate_by_color", {})
    parent_colors = rapfi.get("parent_by_color", {})
    if not (
        int(rapfi.get("pairs", 0)) >= 1024
        and float(rapfi.get("candidate_score", 0.0)) > float(rapfi.get("parent_score", 1.0))
        and float(rapfi.get("two_sided_exact_sign_p", 1.0)) < 0.05
        and all(
            float(candidate_colors[color]["score"]) + 1e-12
            >= float(parent_colors[color]["score"])
            for color in ("black", "white")
        )
    ):
        raise GateError("final independent Rapfi gate failed")

    arena, arena_sha = read_sidecar_verified(direct_arena_path)
    recorded_hash = arena.pop("report_sha256", None)
    if recorded_hash != canonical_sha256(arena):
        raise GateError("direct arena embedded report hash mismatch")
    arena["report_sha256"] = recorded_hash
    if (
        arena.get("candidate_sha256") != candidate_sha
        or arena.get("complete") is not True
        or arena.get("summary", {}).get("hard_gate_passed") is not True
        or int(arena.get("summary", {}).get("pairs", 0)) < 1024
        or float(arena.get("summary", {}).get("two_sided_exact_sign_p", 1.0)) >= 0.05
        or arena.get("summary", {}).get("color_non_regression") is not True
    ):
        raise GateError("direct candidate-versus-champion arena gate failed")

    now = _utc_now()
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "v3_stage": CHAMPION_STAGE,
        "iteration": int(candidate.get("iteration", -1)),
        "global_step": int(candidate.get("global_step", 0)),
        "config": config,
        "model_spec": dict(candidate.get("model_spec", {})),
        "best_model": _cpu_state_copy(state, label="champion best_model"),
        "approved_model": _cpu_state_copy(state, label="champion approved_model"),
        "candidate_model": _cpu_state_copy(state, label="champion candidate_model"),
        "train_model": _cpu_state_copy(state, label="champion train_model"),
        "approval_status": "approved",
        "is_approved": True,
        "approved_at_utc": now,
        "model_version": "gomoku-v3-round7-20260801",
        "provenance": {
            "candidate_evaluation_checkpoint": str(candidate_path.resolve()),
            "candidate_evaluation_checkpoint_sha256": candidate_sha,
            "source_checkpoint": candidate.get("source_checkpoint"),
            "source_checkpoint_sha256": candidate.get("source_checkpoint_sha256"),
            "source_parent_checkpoint_sha256": candidate.get("source_parent_checkpoint_sha256"),
            "prior_approved_checkpoint_sha256": candidate.get("prior_approved_checkpoint_sha256"),
        },
        "external_evaluation": {
            "status": "passed_round7_hard_gates",
            "final_certified": True,
            "static": {"report": str(static_summary_path.resolve()), "report_sha256": static_sha, **row},
            "rapfi_final": {"report": str(rapfi_final_path.resolve()), "report_sha256": rapfi_sha, **rapfi},
            "direct_arena": {"report": str(direct_arena_path.resolve()), "report_sha256": arena_sha, **arena},
        },
    }
    _atomic_no_clobber_torch_save(payload, output_path.resolve())
    return {
        "status": "approved_round7_champion_created",
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path.resolve()),
        "candidate_sha256": candidate_sha,
        "model_version": payload["model_version"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--rapfi-final", type=Path, required=True)
    parser.add_argument("--direct-arena", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = promote(
        args.candidate, args.candidate_sha256, args.static_summary,
        args.rapfi_final, args.direct_arena, args.output,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
