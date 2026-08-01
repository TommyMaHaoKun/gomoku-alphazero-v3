"""Run a resumable, paired color-swapped arena between two V3 checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .train_alphazero import (
    BLACK,
    EMPTY,
    WHITE,
    Config,
    GomokuGame,
    Node,
    PolicyValueNet,
    run_mcts_batch,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def exact_two_sided_sign_p(gains: int, losses: int) -> float:
    trials = gains + losses
    if trials == 0:
        return 1.0
    lower = min(gains, losses)
    tail = sum(math.comb(trials, count) for count in range(lower + 1)) / 2**trials
    return min(1.0, 2.0 * tail)


def load_model(path: Path, expected_sha256: str, device: torch.device) -> tuple[PolicyValueNet, dict[str, Any]]:
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise ValueError(f"checkpoint SHA256 mismatch for {path}: {actual}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 3:
        raise ValueError(f"{path} is not a format-v3 checkpoint")
    spec = checkpoint.get("model_spec")
    if not isinstance(spec, dict):
        raise ValueError(f"{path} has no model_spec")
    state = checkpoint.get("best_model")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path} has no best_model state")
    model = PolicyValueNet(
        int(spec["board_size"]), int(spec["channels"]), int(spec["residual_blocks"])
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def build_config(checkpoint: dict[str, Any], simulations: int) -> Config:
    raw = checkpoint.get("config")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint config must be a mapping")
    allowed = {field.name for field in fields(Config)}
    values = {key: value for key, value in raw.items() if key in allowed}
    config = Config(**values)
    config.arena_simulations = simulations
    return config


def make_opening(config: Config, seed: int, pair_index: int, plies: int) -> tuple[GomokuGame, list[int]]:
    rng = np.random.default_rng(seed + pair_index * 1_000_003)
    game = GomokuGame(config.board_size, config.win_length)
    moves: list[int] = []
    for _ in range(plies):
        choices = game.candidate_actions(config.candidate_radius)
        action = int(rng.choice(choices))
        game.play(action)
        moves.append(action)
        if game.terminal:
            raise ValueError(f"opening became terminal for pair {pair_index}")
    return game, moves


def play_batch(
    candidate: PolicyValueNet,
    champion: PolicyValueNet,
    openings: list[tuple[int, GomokuGame, list[int]]],
    config: Config,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    games: list[GomokuGame] = []
    candidate_is_black: list[bool] = []
    metadata: list[tuple[int, list[int]]] = []
    for pair_index, opening, moves in openings:
        games.extend((opening.clone(), opening.clone()))
        candidate_is_black.extend((True, False))
        metadata.extend(((pair_index, moves), (pair_index, moves)))
    rng = np.random.default_rng(seed + openings[0][0] * 2_000_033)
    active = list(range(len(games)))
    while active:
        candidate_turn: list[int] = []
        champion_turn: list[int] = []
        for index in active:
            game = games[index]
            uses_candidate = (
                game.player == BLACK and candidate_is_black[index]
            ) or (
                game.player == WHITE and not candidate_is_black[index]
            )
            (candidate_turn if uses_candidate else champion_turn).append(index)
        for indices, model in ((candidate_turn, candidate), (champion_turn, champion)):
            if not indices:
                continue
            selected = [games[index] for index in indices]
            roots = [Node(1.0, game.player) for game in selected]
            run_mcts_batch(
                model,
                selected,
                roots,
                config.arena_simulations,
                config,
                device,
                rng,
                add_noise=False,
            )
            for index, root in zip(indices, roots):
                visits = {action: child.visit_count for action, child in root.children.items()}
                if not visits:
                    raise RuntimeError("MCTS returned no legal action")
                games[index].play(max(visits, key=visits.get))
        active = [index for index in active if not games[index].terminal]

    records: list[dict[str, Any]] = []
    for index, game in enumerate(games):
        pair_index, opening_moves = metadata[index]
        candidate_color = BLACK if candidate_is_black[index] else WHITE
        result = 0.5 if game.winner == EMPTY else float(game.winner == candidate_color)
        record = {
            "pair_index": pair_index,
            "candidate_color": "black" if candidate_color == BLACK else "white",
            "opening_actions": opening_moves,
            "winner": int(game.winner),
            "candidate_result": result,
            "plies": int(game.move_count),
        }
        record["record_sha256"] = canonical_sha256(record)
        records.append(record)
    return records


def load_progress(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    records: dict[tuple[int, str], dict[str, Any]] = {}
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        digest = record.pop("record_sha256", None)
        if digest != canonical_sha256(record):
            raise ValueError(f"progress hash mismatch at line {line_number}")
        record["record_sha256"] = digest
        key = (int(record["pair_index"]), str(record["candidate_color"]))
        if key in records:
            raise ValueError(f"duplicate progress record {key}")
        records[key] = record
    return records


def append_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def color_summary(records: list[dict[str, Any]], color: str, candidate: bool) -> dict[str, float | int]:
    selected = [r for r in records if r["candidate_color"] == color]
    scores = [float(r["candidate_result"]) for r in selected]
    if not candidate:
        scores = [1.0 - score for score in scores]
    return {
        "games": len(scores),
        "wins": sum(score == 1.0 for score in scores),
        "draws": sum(score == 0.5 for score in scores),
        "losses": sum(score == 0.0 for score in scores),
        "score": sum(scores) / len(scores),
    }


def summarize(records: list[dict[str, Any]], pairs: int) -> dict[str, Any]:
    indexed = {(int(r["pair_index"]), str(r["candidate_color"])): r for r in records}
    expected = {(pair, color) for pair in range(pairs) for color in ("black", "white")}
    if set(indexed) != expected:
        raise ValueError("arena progress is incomplete or has unexpected keys")
    pair_scores = [
        (float(indexed[pair, "black"]["candidate_result"]) + float(indexed[pair, "white"]["candidate_result"])) / 2.0
        for pair in range(pairs)
    ]
    gains = sum(score > 0.5 for score in pair_scores)
    losses = sum(score < 0.5 for score in pair_scores)
    unchanged = pairs - gains - losses
    p_value = exact_two_sided_sign_p(gains, losses)
    candidate_by_color = {
        "black": color_summary(records, "black", True),
        "white": color_summary(records, "white", True),
    }
    champion_by_color = {
        "black": color_summary(records, "white", False),
        "white": color_summary(records, "black", False),
    }
    color_non_regression = all(
        float(candidate_by_color[color]["score"]) + 0.02
        >= float(champion_by_color[color]["score"])
        for color in ("black", "white")
    )
    score = sum(float(r["candidate_result"]) for r in records) / len(records)
    return {
        "pairs": pairs,
        "games": len(records),
        "candidate_score": score,
        "champion_score": 1.0 - score,
        "score_delta": 2.0 * score - 1.0,
        "paired_gains": gains,
        "paired_losses": losses,
        "paired_unchanged": unchanged,
        "two_sided_exact_sign_p": p_value,
        "statistically_significant_at_0_05": p_value < 0.05,
        "candidate_by_color": candidate_by_color,
        "champion_by_color": champion_by_color,
        "color_non_regression_tolerance": 0.02,
        "color_non_regression": color_non_regression,
        "hard_gate_passed": score > 0.5 and p_value < 0.05 and color_non_regression,
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--champion-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=1024)
    parser.add_argument("--batch-pairs", type=int, default=16)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20280808)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pairs < 1 or args.batch_pairs < 1 or args.simulations < 1:
        raise ValueError("pairs, batch-pairs, and simulations must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_float32_matmul_precision("high")
    candidate, candidate_checkpoint = load_model(args.candidate, args.candidate_sha256, device)
    champion, champion_checkpoint = load_model(args.champion, args.champion_sha256, device)
    if candidate_checkpoint.get("model_spec") != champion_checkpoint.get("model_spec"):
        raise ValueError("candidate and champion model specifications differ")
    config = build_config(champion_checkpoint, args.simulations)
    progress_path = args.output.with_suffix(".jsonl")
    existing = load_progress(progress_path)
    completed_pairs = {
        pair for pair in range(args.pairs)
        if (pair, "black") in existing and (pair, "white") in existing
    }
    started = time.time()
    for start in range(0, args.pairs, args.batch_pairs):
        pair_indices = [
            pair for pair in range(start, min(args.pairs, start + args.batch_pairs))
            if pair not in completed_pairs
        ]
        if not pair_indices:
            continue
        openings = [
            (pair, *make_opening(config, args.seed, pair, args.opening_plies))
            for pair in pair_indices
        ]
        records = play_batch(candidate, champion, openings, config, device, args.seed)
        append_records(progress_path, records)
        completed_pairs.update(pair_indices)
        print(
            f"pairs {len(completed_pairs)}/{args.pairs} elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    records = list(load_progress(progress_path).values())
    summary = summarize(records, args.pairs)
    report: dict[str, Any] = {
        "format_version": 1,
        "report_type": "paired_model_arena",
        "complete": True,
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": args.candidate_sha256.lower(),
        "champion": str(args.champion.resolve()),
        "champion_sha256": args.champion_sha256.lower(),
        "seed": args.seed,
        "opening_plies": args.opening_plies,
        "simulations": args.simulations,
        "summary": summary,
        "records_sha256": sha256_file(progress_path),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
