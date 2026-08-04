#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN="rapfi_distillation/loss_correction_round1"
OUT_DIR="$RUN/training"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e"
LOSS_DATA="$RUN/loss_hard_negative.npz"
GENERAL_RAPFI="rapfi_distillation/run_20260730_dagger128/rapfi_dagger_policy.npz"
TACTICS="alphazero_training/v3_legal_tactics_train.npz"
DDQK="alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz"
OUTPUT="$OUT_DIR/candidate_repair_s400.pt"
LOG="$OUT_DIR/train_repair_s400.log"

cd "$ROOT"
mkdir -p "$OUT_DIR"

actual_parent="$(sha256sum "$PARENT" | awk '{print $1}')"
if [[ "$actual_parent" != "$PARENT_SHA256" ]]; then
  echo "approved parent SHA256 mismatch: $actual_parent" >&2
  exit 2
fi

white_args=()
for name in p2 p4 p8 seed20260724_parent seed20260730_parent; do
  white_args+=(
    --white-defense-npz "alphazero_training/_v3f_white_train_det/$name/train.npz"
    --white-defense-manifest "alphazero_training/_v3f_white_train_det/$name/manifest.json"
    --white-defense-weight 0.06
  )
done

echo "[$(date -Is)] training balanced loss-repair candidate output=$OUTPUT"
"$PYTHON_BIN" -u -m alphazero_training.train_v3_supervised \
  --init-checkpoint "$PARENT" \
  --dataset "$LOSS_DATA" \
  --dataset "$GENERAL_RAPFI" \
  --dataset "$TACTICS" \
  --dataset "$DDQK" \
  --dataset-weight 0.30 \
  --dataset-weight 0.20 \
  --dataset-weight 0.15 \
  --dataset-weight 0.05 \
  "${white_args[@]}" \
  --output "$OUTPUT" \
  --steps 400 \
  --batch-size 512 \
  --learning-rate 6e-6 \
  --min-learning-rate 3e-6 \
  --warmup-steps 80 \
  --freeze-trunk-steps 350 \
  --train-last-residual-blocks-during-freeze 2 \
  --value-loss-scale 0 \
  --value-distill-scale 0.2 \
  --policy-distill-scale 0.25 \
  --safe-hard-negative-scale 0.25 \
  --safe-hard-negative-margin 1.0 \
  --mistake-hard-negative-scale 0.4 \
  --mistake-hard-negative-margin 1.0 \
  --validation-fraction 0.1 \
  --eval-every 400 \
  --seed 20261202 2>&1 | tee "$LOG"

sha256sum "$OUTPUT" | tee "$OUTPUT.sha256"
