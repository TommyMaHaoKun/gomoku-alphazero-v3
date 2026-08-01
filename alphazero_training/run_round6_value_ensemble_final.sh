#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"
ENGINE_SHA256="95cea7a20374f8fa2795d4b410b215a7f058c9e2b659d8c0670bcb86761e6f46"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"
CANDIDATE="rapfi_distillation/round6_regret_restart/training/r6_value_ensemble_a015_eval.pt"
CANDIDATE_SHA256="cb892b5c8a64b0e737a035631ec20fb565fcd13971a4a1712458be8682229d99"
RUN="rapfi_distillation/round6_regret_restart"
FINAL="$RUN/value_ensemble_final_seed20280804"
PAIRS=1024
SEED=20280804

cd "$ROOT"
mkdir -p "$FINAL" "$RUN/logs"
[[ "$(sha256sum "$PARENT" | awk '{print $1}')" == "$PARENT_SHA256" ]]
[[ "$(sha256sum "$CANDIDATE" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]]
[[ "$(sha256sum "$ENGINE" | awk '{print $1}')" == "$ENGINE_SHA256" ]]

run_report() {
  local name="$1" checkpoint="$2"
  local target="$FINAL/$name"
  mkdir -p "$target"
  if [[ ! -s "$target/complete.done" ]]; then
    "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
      --checkpoint "$checkpoint" --engine "$ENGINE" \
      --report "$target/games.json" --dataset "$target/policy_unused.npz" \
      --ai-loss-dir "$target/pending_training/ai_losses" \
      --pairs "$PAIRS" --opening-plies 6 --simulations 64 --workers 12 \
      --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed "$SEED" \
      2>&1 | tee "$RUN/logs/value_ensemble_final_${name}.log"
    sha256sum "$target/games.json" > "$target/games.json.sha256"
    touch "$target/complete.done"
  fi
}

run_report parent "$PARENT"
run_report r6_value_ensemble_a015 "$CANDIDATE"
"$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
  --parent-report "$FINAL/parent/games.json" \
  --candidate-report "$FINAL/r6_value_ensemble_a015/games.json" \
  --output "$FINAL/comparison.json"
sha256sum "$FINAL/comparison.json" > "$FINAL/comparison.json.sha256"
touch "$FINAL/complete.done"
