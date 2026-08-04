#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN="rapfi_distillation/round7_league_selfplay"
CANDIDATE="$RUN/round7_a035_eval.pt"
CANDIDATE_SHA256="3a2f343985f56e6212118936cf6df67beff01b7bfaaef9538d97c3d4fa37075b"
CHAMPION="alphazero_training/latest.pt"
CHAMPION_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"
ARENA="$RUN/direct_arena_seed20280808"

cd "$ROOT"
mkdir -p "$ARENA" "$RUN/logs"
[[ "$(sha256sum "$CANDIDATE" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]]
[[ "$(sha256sum "$CHAMPION" | awk '{print $1}')" == "$CHAMPION_SHA256" ]]
sha256sum -c "$RUN/static_gate_summary.json.sha256"
sha256sum -c "$RUN/rapfi_screen_seed20280806/summary.json.sha256"
sha256sum -c "$RUN/rapfi_final_seed20280807/comparison.json.sha256"

"$PYTHON_BIN" -u -m alphazero_training.paired_model_arena \
  --candidate "$CANDIDATE" --candidate-sha256 "$CANDIDATE_SHA256" \
  --champion "$CHAMPION" --champion-sha256 "$CHAMPION_SHA256" \
  --output "$ARENA/report.json" --pairs 1024 --batch-pairs 16 \
  --opening-plies 6 --simulations 64 --seed 20280808 --device cuda
sha256sum "$ARENA/report.json" > "$ARENA/report.json.sha256"
sha256sum "$ARENA/report.jsonl" > "$ARENA/report.jsonl.sha256"
touch "$ARENA/complete.done"
