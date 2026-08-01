#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"
ENGINE_SHA256="95cea7a20374f8fa2795d4b410b215a7f058c9e2b659d8c0670bcb86761e6f46"
RUN="rapfi_distillation/round6_regret_restart"
ROOT_REPORT="$RUN/value_final_seed20280604/parent/games.json"
CANDIDATE="rapfi_distillation/round6_regret_restart/training/r6_value_a075_eval.pt"
CANDIDATE_SHA256="1899be5bf9847ce000d88521a42884adeadd71529acaa8c3969d496fb168ed65"
TARGET="$RUN/value_secondary_seed20280604/r6_value_a075"
PAIRS=1024
SEED=20280604

cd "$ROOT"
mkdir -p "$TARGET" "$RUN/logs"
[[ -s "$ROOT_REPORT" ]]
[[ "$(sha256sum "$CANDIDATE" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]]
[[ "$(sha256sum "$ENGINE" | awk '{print $1}')" == "$ENGINE_SHA256" ]]

if [[ ! -s "$TARGET/complete.done" ]]; then
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
    --checkpoint "$CANDIDATE" --engine "$ENGINE" \
    --report "$TARGET/games.json" --dataset "$TARGET/policy_unused.npz" \
    --ai-loss-dir "$TARGET/pending_training/ai_losses" \
    --pairs "$PAIRS" --opening-plies 6 --simulations 64 --workers 12 \
    --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed "$SEED" \
    2>&1 | tee "$RUN/logs/value_secondary_a075.log"
  sha256sum "$TARGET/games.json" > "$TARGET/games.json.sha256"
  touch "$TARGET/complete.done"
fi

"$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
  --parent-report "$ROOT_REPORT" --candidate-report "$TARGET/games.json" \
  --output "$RUN/value_secondary_seed20280604/comparison.json"
sha256sum "$RUN/value_secondary_seed20280604/comparison.json" \
  > "$RUN/value_secondary_seed20280604/comparison.json.sha256"
touch "$RUN/value_secondary_seed20280604/complete.done"
