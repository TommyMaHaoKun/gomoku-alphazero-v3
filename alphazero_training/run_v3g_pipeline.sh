#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE_DIR="${V3G_PIPELINE_DIR:-run_v3g_pipeline}"
WARM_DIR="${V3G_WARM_DIR:-run_v3g_white_margin}"
SELFPLAY_DIR="${V3G_SELFPLAY_DIR:-run_v3g_white_selfplay}"

cd "$ROOT"
if ! mkdir "$PIPELINE_DIR" 2>/dev/null; then
  echo "Refusing to overwrite an existing V3G pipeline: $PIPELINE_DIR" >&2
  exit 2
fi
LOG="$PIPELINE_DIR/pipeline.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] V3G pipeline begins"
echo "warm_output=$WARM_DIR selfplay_output=$SELFPLAY_DIR"

V3F_WHITE_PROFILE=v3g \
V3F_WHITE_ROOT=alphazero_training/_v3g_white_train_det \
V3F_OUT_DIR="$WARM_DIR" \
V3F_STEPS=800 \
V3F_TRAIN_SEED=20260821 \
V3F_FINAL_EVAL_SEED=20260817 \
V3F_WHITE_WEIGHT=0.03 \
V3F_LEARNING_RATE=5e-6 \
V3F_MIN_LEARNING_RATE=1e-6 \
V3F_POLICY_DISTILL_SCALE=0.2 \
V3F_VALUE_LOSS_SCALE=0.5 \
V3F_VALUE_DISTILL_SCALE=0.25 \
V3F_FREEZE_TRUNK_STEPS=550 \
V3F_TRAIN_LAST_RESIDUAL_BLOCKS=2 \
V3F_SAFE_HARD_NEGATIVE_SCALE=0.5 \
V3F_SAFE_HARD_NEGATIVE_MARGIN=0.5 \
  bash alphazero_training/run_v3f_supervised.sh

echo "[$(date -Is)] V3G margin warm start complete; running development diagnostics"
"$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
  --checkpoint "$WARM_DIR/latest.pt" \
  --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
  --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
  --output "$WARM_DIR/white_defense_dev_eval.json" \
  --model-key train_model \
  --device cuda

"$PYTHON_BIN" -m alphazero_training.v3_legal_tactics evaluate \
  --checkpoint "$WARM_DIR/latest.pt" \
  --dataset alphazero_training/v3_legal_tactics_eval.npz \
  --model-key train_model \
  --split eval \
  --simulations 64 \
  --device cuda \
  --json-out "$WARM_DIR/raw_tactics_dev_eval.json"

echo "[$(date -Is)] starting V3G self-play reinforcement stage"
V3G_INIT_CHECKPOINT="$WARM_DIR/latest.pt" \
V3G_OUT_DIR="$SELFPLAY_DIR" \
V3G_ITERATIONS=100 \
  bash alphazero_training/run_v3g_selfplay.sh

echo "[$(date -Is)] V3G pipeline completed cleanly"
