#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"
ENGINE_SHA256="95cea7a20374f8fa2795d4b410b215a7f058c9e2b659d8c0670bcb86761e6f46"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"
CANDIDATE="rapfi_distillation/round7_league_selfplay/round7_a035_eval.pt"
CANDIDATE_SHA256="3a2f343985f56e6212118936cf6df67beff01b7bfaaef9538d97c3d4fa37075b"
RUN="rapfi_distillation/round7_league_selfplay"
FINAL="$RUN/rapfi_final_seed20280807"
PAIRS=1024
SEED=20280807

cd "$ROOT"
mkdir -p "$FINAL" "$RUN/logs"
[[ "$(sha256sum "$ENGINE" | awk '{print $1}')" == "$ENGINE_SHA256" ]]
[[ "$(sha256sum "$PARENT" | awk '{print $1}')" == "$PARENT_SHA256" ]]
[[ "$(sha256sum "$CANDIDATE" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]]
sha256sum -c "$RUN/static_gate_summary.json.sha256"
sha256sum -c "$RUN/rapfi_screen_seed20280806/summary.json.sha256"

run_report() {
  local name="$1" checkpoint="$2" target
  target="$FINAL/$name"
  mkdir -p "$target"
  if [[ ! -s "$target/complete.done" ]]; then
    "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
      --checkpoint "$checkpoint" --engine "$ENGINE" \
      --report "$target/games.json" --dataset "$target/policy_unused.npz" \
      --ai-loss-dir "$target/pending_training/ai_losses" \
      --pairs "$PAIRS" --opening-plies 6 --simulations 64 --workers 12 \
      --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed "$SEED" \
      2>&1 | tee "$RUN/logs/rapfi_final_${name}.log"
    sha256sum "$target/games.json" > "$target/games.json.sha256"
    sha256sum "$target/policy_unused.npz" > "$target/policy_unused.npz.sha256"
    touch "$target/complete.done"
  fi
}

run_report parent "$PARENT"
run_report round7_a035 "$CANDIDATE"
"$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
  --parent-report "$FINAL/parent/games.json" \
  --candidate-report "$FINAL/round7_a035/games.json" \
  --output "$FINAL/comparison.json"
sha256sum "$FINAL/comparison.json" > "$FINAL/comparison.json.sha256"
touch "$FINAL/complete.done"
