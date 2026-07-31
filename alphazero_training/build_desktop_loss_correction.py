"""Build a teacher-relabelled hard-negative sample from a desktop AI loss.

Desktop replays intentionally preserve the moves that were actually played.
Those one-hot policies are an audit log, not safe policy supervision for an AI
loss: replaying the raw archive would also reinforce the AI's mistake.  This
tool replaces one reviewed AI move with an independently supplied teacher move
and exports all eight board symmetries for supervised correction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


BOARD_SIZE = 19
FORMAT_VERSION = 1


def _transform_xy(x: int, y: int, symmetry: int) -> tuple[int, int]:
    """Apply the same D4 indexing convention as the supervised trainer."""
    limit = BOARD_SIZE - 1
    if symmetry >= 4:
        x = limit - x
        symmetry -= 4
    for _ in range(symmetry):
        x, y = limit - y, x
    return x, y


def _transform_state(state: np.ndarray, symmetry: int) -> np.ndarray:
    transformed = np.zeros_like(state)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            tx, ty = _transform_xy(x, y, symmetry)
            transformed[:, ty, tx] = state[:, y, x]
    return transformed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_correction(
    metadata_path: Path,
    replay_path: Path,
    output: Path,
    *,
    move_number: int,
    teacher_x: int,
    teacher_y: int,
    teacher_name: str,
    teacher_engine_sha256: str,
    teacher_node_budgets: list[int],
    priority: float = 8.0,
) -> dict[str, object]:
    metadata_path = metadata_path.resolve()
    replay_path = replay_path.resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "gargantua_desktop_replay":
        raise ValueError("metadata is not a Gargantua desktop replay")
    if metadata.get("ai_result") != "loss":
        raise ValueError("desktop correction input must be a completed AI loss")
    if metadata.get("replay_sha256") != sha256_file(replay_path):
        raise ValueError("desktop replay SHA256 does not match its metadata")
    if move_number <= 0 or move_number > int(metadata.get("plies", 0)):
        raise ValueError("move_number is outside the recorded game")
    if not (0 <= teacher_x < BOARD_SIZE and 0 <= teacher_y < BOARD_SIZE):
        raise ValueError("teacher move is outside the board")
    if priority <= 0 or not np.isfinite(priority):
        raise ValueError("priority must be finite and positive")
    if not teacher_node_budgets or any(nodes <= 0 for nodes in teacher_node_budgets):
        raise ValueError("teacher_node_budgets must contain positive values")

    recorded_move = metadata["moves"][move_number - 1]
    if int(recorded_move["move_number"]) != move_number:
        raise ValueError("metadata move numbering is inconsistent")
    mistake_x = int(recorded_move["x"])
    mistake_y = int(recorded_move["y"])
    mistake_action = mistake_y * BOARD_SIZE + mistake_x
    teacher_action = teacher_y * BOARD_SIZE + teacher_x
    if teacher_action == mistake_action:
        raise ValueError("teacher move must differ from the recorded mistake")

    with np.load(replay_path, allow_pickle=False) as archive:
        required = {"states", "actions", "players", "move_numbers", "game_id"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"desktop replay is missing arrays: {missing}")
        index = move_number - 1
        state = np.asarray(archive["states"][index], dtype=np.uint8)
        action = int(archive["actions"][index])
        player = int(archive["players"][index])
        archived_move_number = int(archive["move_numbers"][index])
        game_id = str(archive["game_id"][index])
    if state.shape != (4, BOARD_SIZE, BOARD_SIZE):
        raise ValueError("desktop replay state has the wrong shape")
    if action != mistake_action or archived_move_number != move_number:
        raise ValueError("desktop replay arrays disagree with metadata")
    if np.any(state[:, teacher_y, teacher_x]):
        raise ValueError("teacher move points to an occupied location")

    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    mistake_actions: list[int] = []
    teacher_actions: list[int] = []
    sources: list[str] = []
    for symmetry in range(8):
        tx, ty = _transform_xy(teacher_x, teacher_y, symmetry)
        mx, my = _transform_xy(mistake_x, mistake_y, symmetry)
        transformed_teacher = ty * BOARD_SIZE + tx
        transformed_mistake = my * BOARD_SIZE + mx
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        policy[transformed_teacher] = 1.0
        states.append(_transform_state(state, symmetry))
        policies.append(policy)
        mistake_actions.append(transformed_mistake)
        teacher_actions.append(transformed_teacher)
        sources.append(
            f"desktop_loss_correction|game={game_id}|move={move_number}|sym={symmetry}"
        )

    count = len(states)
    arrays = {
        "states": np.stack(states),
        "policies": np.stack(policies),
        "values": np.zeros(count, dtype=np.float32),
        "policy_weights": np.ones(count, dtype=np.float32),
        "value_weights": np.zeros(count, dtype=np.float32),
        "priority": np.full(count, priority, dtype=np.float32),
        "mistake_action": np.asarray(mistake_actions, dtype=np.int16),
        "teacher_action": np.asarray(teacher_actions, dtype=np.int16),
        "group_id": np.full(count, game_id),
        "player": np.full(count, player, dtype=np.int8),
        "move_number": np.full(count, move_number, dtype=np.int16),
        "source": np.asarray(sources),
        "split": np.full(count, "train", dtype="<U5"),
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)

    result: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "dataset_kind": "desktop_loss_teacher_correction",
        "game_id": game_id,
        "source_metadata": str(metadata_path),
        "source_metadata_sha256": sha256_file(metadata_path),
        "source_replay": str(replay_path),
        "source_replay_sha256": sha256_file(replay_path),
        "ai_color": metadata["ai_color"],
        "move_number": move_number,
        "player": player,
        "recorded_mistake": {
            "x": mistake_x,
            "y": mistake_y,
            "action": mistake_action,
        },
        "teacher_correction": {
            "x": teacher_x,
            "y": teacher_y,
            "action": teacher_action,
        },
        "teacher_evidence": {
            "name": teacher_name,
            "engine_sha256": teacher_engine_sha256,
            "node_budgets": teacher_node_budgets,
            "consensus": True,
        },
        "samples": count,
        "symmetries": 8,
        "priority": priority,
        "value_target": "masked_policy_only",
        "npz_sha256": sha256_file(output),
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    temporary_json = sidecar.with_name(sidecar.name + f".tmp-{os.getpid()}")
    temporary_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_json, sidecar)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--move-number", type=int, required=True)
    parser.add_argument("--teacher-x", type=int, required=True)
    parser.add_argument("--teacher-y", type=int, required=True)
    parser.add_argument("--teacher-name", default="Rapfi")
    parser.add_argument("--teacher-engine-sha256", required=True)
    parser.add_argument("--teacher-node-budgets", type=int, nargs="+", required=True)
    parser.add_argument("--priority", type=float, default=8.0)
    args = parser.parse_args()
    result = build_correction(
        args.metadata,
        args.replay,
        args.output,
        move_number=args.move_number,
        teacher_x=args.teacher_x,
        teacher_y=args.teacher_y,
        teacher_name=args.teacher_name,
        teacher_engine_sha256=args.teacher_engine_sha256,
        teacher_node_budgets=args.teacher_node_budgets,
        priority=args.priority,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
