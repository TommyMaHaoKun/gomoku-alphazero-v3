#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
OUT_DIR="${RAPFI_LOSS_OUT:-rapfi_distillation/loss_correction_round1/training}"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e"
LOSS_DATA="rapfi_distillation/loss_correction_round1/loss_hard_negative.npz"
GENERAL_RAPFI="rapfi_distillation/run_20260730_dagger128/rapfi_dagger_policy.npz"
TACTICS="alphazero_training/v3_legal_tactics_train.npz"
DDQK="alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz"

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
    --white-defense-weight 0.03
  )
done

for steps in 200 400 600; do
  case "$steps" in
    200) warmup=40; freeze=200 ;;
    400) warmup=80; freeze=300 ;;
    600) warmup=100; freeze=300 ;;
  esac
  output="$OUT_DIR/candidate_s${steps}.pt"
  log="$OUT_DIR/train_s${steps}.log"
  echo "[$(date -Is)] training steps=$steps warmup=$warmup freeze=$freeze output=$output"
  "$PYTHON_BIN" -u -m alphazero_training.train_v3_supervised \
    --init-checkpoint "$PARENT" \
    --dataset "$LOSS_DATA" \
    --dataset "$GENERAL_RAPFI" \
    --dataset "$TACTICS" \
    --dataset "$DDQK" \
    --dataset-weight 0.45 \
    --dataset-weight 0.20 \
    --dataset-weight 0.15 \
    --dataset-weight 0.05 \
    "${white_args[@]}" \
    --output "$output" \
    --steps "$steps" \
    --batch-size 512 \
    --learning-rate 8e-6 \
    --min-learning-rate 4e-6 \
    --warmup-steps "$warmup" \
    --freeze-trunk-steps "$freeze" \
    --train-last-residual-blocks-during-freeze 2 \
    --value-loss-scale 0 \
    --value-distill-scale 0.2 \
    --policy-distill-scale 0.2 \
    --mistake-hard-negative-scale 0.5 \
    --mistake-hard-negative-margin 1.0 \
    --validation-fraction 0.1 \
    --eval-every "$steps" \
    --seed 20261201 2>&1 | tee "$log"
  sha256sum "$output" | tee "$output.sha256"
done
