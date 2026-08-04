"""Roll out a Rapfi continuation from a reviewed desktop-loss correction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from . import rapfi_counterfactual as _counterfactual_module
from .build_desktop_loss_correction import sha256_file
from .rapfi_counterfactual import (
    BranchTask,
    _rollout,
    _worker_initialize,
    export_branch_dataset,
)
from .rapfi_distill import BLACK, WHITE, canonical_sha256


def build_counterfactual(
    metadata_path: Path,
    correction_path: Path,
    engine_path: Path,
    output_report: Path,
    output_dataset: Path,
    *,
    max_branch_plies: int,
    timeout_turn_ms: int,
    max_nodes: int,
    engine_threads: int,
    engine_memory_mb: int,
) -> dict[str, object]:
    metadata_path = metadata_path.resolve()
    correction_path = correction_path.resolve()
    engine_path = engine_path.resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sidecar_path = correction_path.with_suffix(correction_path.suffix + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("source_metadata_sha256") != sha256_file(metadata_path):
        raise ValueError("correction provenance does not match desktop metadata")
    if sidecar.get("npz_sha256") != sha256_file(correction_path):
        raise ValueError("correction dataset SHA256 does not match its sidecar")
    if max_branch_plies <= 0:
        raise ValueError("max_branch_plies must be positive")

    with np.load(correction_path, allow_pickle=False) as archive:
        move_number = int(archive["move_number"][0])
        teacher_action = int(archive["teacher_action"][0])
        mistake_action = int(archive["mistake_action"][0])
        logged_player = int(archive["player"][0])
    expected_logged_player = 1 if move_number % 2 == 1 else -1
    if logged_player != expected_logged_player:
        raise ValueError("correction player disagrees with alternating move order")
    # Desktop/self-play archives encode white as -1; the Rapfi protocol and
    # its counterfactual records encode white as 2.
    player = BLACK if logged_player == 1 else WHITE
    history_before = [
        [
            int(move["x"]),
            int(move["y"]),
            BLACK if str(move["color"]) == "black" else WHITE,
        ]
        for move in metadata["moves"][: move_number - 1]
    ]
    teacher_y, teacher_x = divmod(teacher_action, 19)
    mistake_y, mistake_x = divmod(mistake_action, 19)
    distance = int(metadata["plies"]) - move_number
    priority = 2.0 if player == WHITE else 1.0
    if distance <= 8:
        priority *= 2.0
    elif distance <= 16:
        priority *= 1.5
    task = BranchTask(
        task_index=0,
        pair_index=0,
        game_index=0,
        source_ply=move_number - 1,
        student_color=player,
        player=player,
        history_before=history_before,
        teacher_x=teacher_x,
        teacher_y=teacher_y,
        mistake_x=mistake_x,
        mistake_y=mistake_y,
        original_moves_to_end=distance,
        priority=priority,
    )
    _worker_initialize(
        str(engine_path),
        timeout_turn_ms,
        max_nodes,
        engine_threads,
        engine_memory_mb,
    )
    try:
        branch = _rollout(task, max_branch_plies)
    finally:
        if _counterfactual_module._WORKER_ENGINE is not None:
            _counterfactual_module._WORKER_ENGINE.close()
            _counterfactual_module._WORKER_ENGINE = None
    dataset_metadata = export_branch_dataset([branch], output_dataset.resolve())
    payload: dict[str, object] = {
        "format_version": 1,
        "report_type": "desktop_loss_counterfactual_correction",
        "complete": branch.error is None,
        "signature": {
            "source_metadata": str(metadata_path),
            "source_metadata_sha256": sha256_file(metadata_path),
            "correction_dataset": str(correction_path),
            "correction_dataset_sha256": sha256_file(correction_path),
            "correction_sidecar_sha256": sha256_file(sidecar_path),
            "engine": str(engine_path),
            "engine_sha256": sha256_file(engine_path),
            "max_branch_plies": max_branch_plies,
            "timeout_turn_ms": timeout_turn_ms,
            "max_nodes": max_nodes,
            "engine_threads": engine_threads,
        },
        "summary": {
            "termination": branch.termination,
            "winner": branch.winner,
            "moves": len(branch.moves),
            "error": branch.error,
            "dataset": dataset_metadata,
        },
        "branch": asdict(branch),
    }
    payload["report_sha256"] = canonical_sha256(payload)
    output_report = output_report.resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--max-branch-plies", type=int, default=96)
    parser.add_argument("--timeout-turn-ms", type=int, default=300)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--engine-threads", type=int, default=4)
    parser.add_argument("--engine-memory-mb", type=int, default=2048)
    args = parser.parse_args()
    payload = build_counterfactual(
        args.metadata,
        args.correction,
        args.engine,
        args.output_report,
        args.output_dataset,
        max_branch_plies=args.max_branch_plies,
        timeout_turn_ms=args.timeout_turn_ms,
        max_nodes=args.max_nodes,
        engine_threads=args.engine_threads,
        engine_memory_mb=args.engine_memory_mb,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
