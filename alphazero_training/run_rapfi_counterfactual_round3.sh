#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SOURCE="rapfi_distillation/longrun_round2_dagger2048/games.json"
RUN="rapfi_distillation/longrun_counterfactual_round3"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"

cd "$ROOT"
mkdir -p "$RUN"

while pgrep -f '^bash alphazero_training/run_rapfi_joint_longrun_pipeline.sh$' >/dev/null; do
  echo "[$(date -Is)] waiting for round2 joint pipeline before counterfactual rollout"
  sleep 60
done

if [[ ! -f "$SOURCE" ]]; then
  echo "source round2 report is missing: $SOURCE" >&2
  exit 2
fi

if [[ ! -f "$RUN/branches.npz" ]]; then
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_counterfactual \
    --report "$SOURCE" \
    --engine "$ENGINE" \
    --output-report "$RUN/branches.json" \
    --output-dataset "$RUN/branches.npz" \
    --max-tasks 4096 \
    --max-branch-plies 64 \
    --workers 8 \
    --timeout-turn-ms 300 \
    --max-nodes 100000 \
    --engine-threads 4 \
    --seed 20270501
fi

sha256sum "$RUN/branches.json" "$RUN/branches.npz" | tee "$RUN/SHA256SUMS"
echo "[$(date -Is)] counterfactual round3 data complete"
