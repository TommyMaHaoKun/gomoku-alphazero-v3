"""Evaluate a checkpoint on one reviewed desktop-loss correction position."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .build_desktop_loss_correction import sha256_file
from .play_agent import AlphaZeroGomokuAgent
from .train_alphazero import Config, PolicyValueNet


BOARD_SIZE = 19


def evaluate_correction(
    checkpoint_path: Path,
    dataset_path: Path,
    *,
    simulations: int,
) -> dict[str, object]:
    checkpoint_path = checkpoint_path.resolve()
    dataset_path = dataset_path.resolve()
    with np.load(dataset_path, allow_pickle=False) as archive:
        state = np.asarray(archive["states"][0], dtype=np.uint8)
        teacher_action = int(archive["teacher_action"][0])
        mistake_action = int(archive["mistake_action"][0])
        move_number = int(archive["move_number"][0])
        source = str(archive["source"][0])
    occupied = (state[0] != 0) | (state[1] != 0)
    legal = np.flatnonzero(~occupied.reshape(-1))
    if teacher_action not in legal or mistake_action not in legal:
        raise ValueError("correction actions must both be legal in the source state")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = Config(**checkpoint["config"])
    model = PolicyValueNet(
        config.board_size, config.channels, config.residual_blocks
    ).to(device)
    model.load_state_dict(checkpoint["best_model"], strict=True)
    model.eval()
    with torch.no_grad():
        logits, value = model(
            torch.from_numpy(state[None]).to(device=device, dtype=torch.float32)
        )
        probabilities = torch.softmax(logits[0], dim=0).cpu().numpy()
    legal_order = legal[np.argsort(-probabilities[legal])]
    legal_ranks = {int(action): rank for rank, action in enumerate(legal_order, start=1)}

    black_to_move = bool(np.all(state[3] == 1))
    current_ui = 1 if black_to_move else 2
    opponent_ui = 2 if black_to_move else 1
    grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    grid[state[0] != 0] = current_ui
    grid[state[1] != 0] = opponent_ui
    last_locations = np.argwhere(state[2] != 0)
    last_move = None
    if len(last_locations) == 1:
        last_y, last_x = map(int, last_locations[0])
        last_move = (last_x, last_y)
    elif len(last_locations) > 1:
        raise ValueError("source state has more than one last-move marker")

    agent = AlphaZeroGomokuAgent(checkpoint_path, simulations=simulations)
    deployed_move = agent.choose_move(
        grid.tolist(), last_move=last_move, ai_color=current_ui
    )
    deployed_action = (
        None
        if deployed_move is None
        else int(deployed_move[1] * BOARD_SIZE + deployed_move[0])
    )
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "source": source,
        "move_number": move_number,
        "teacher_action": teacher_action,
        "teacher_xy": [teacher_action % BOARD_SIZE, teacher_action // BOARD_SIZE],
        "mistake_action": mistake_action,
        "mistake_xy": [mistake_action % BOARD_SIZE, mistake_action // BOARD_SIZE],
        "raw_network": {
            "teacher_legal_rank": legal_ranks[teacher_action],
            "teacher_probability": float(probabilities[teacher_action]),
            "mistake_legal_rank": legal_ranks[mistake_action],
            "mistake_probability": float(probabilities[mistake_action]),
            "top5_legal": [
                {
                    "action": int(action),
                    "xy": [int(action % BOARD_SIZE), int(action // BOARD_SIZE)],
                    "probability": float(probabilities[action]),
                }
                for action in legal_order[:5]
            ],
            "value": float(value.reshape(-1)[0].item()),
        },
        "deployed_search": {
            "simulations": simulations,
            "action": deployed_action,
            "xy": None if deployed_move is None else list(map(int, deployed_move)),
            "reason": agent.last_decision_reason,
        },
        "passed": deployed_action == teacher_action,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.simulations <= 0:
        raise ValueError("simulations must be positive")
    result = evaluate_correction(
        args.checkpoint, args.dataset, simulations=args.simulations
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
