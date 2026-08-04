#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN="rapfi_distillation/round6_regret_restart"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"
OUTPUT="${ROUND6_VALUE_OUTPUT:-$RUN/training/r6_value_repair_s8000.pt}"
TRAIN_SEED="${ROUND6_VALUE_SEED:-20280405}"

cd "$ROOT"
[[ "$(sha256sum "$PARENT" | awk '{print $1}')" == "$PARENT_SHA256" ]]
[[ -s "$RUN/DATA_STAGE_COMPLETE.json" ]]

"$PYTHON_BIN" -u -m alphazero_training.train_v3_supervised \
  --init-checkpoint "$PARENT" \
  --dataset "$RUN/datasets/joint_current_champion.npz" \
  --dataset "$RUN/counterfactual/branches.npz" \
  --dataset rapfi_distillation/longrun_counterfactual_round3/branches.npz \
  --dataset alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz \
  --dataset rapfi_distillation/desktop_loss_corrections/move14_counterfactual_branch.npz \
  --dataset-weight 0.40 \
  --dataset-weight 0.30 \
  --dataset-weight 0.10 \
  --dataset-weight 0.15 \
  --dataset-weight 0.05 \
  --output "$OUTPUT" \
  --steps 8000 --batch-size 1024 \
  --learning-rate 1e-5 --min-learning-rate 1e-6 \
  --warmup-steps 400 \
  --freeze-trunk-steps 8000 \
  --train-last-residual-blocks-during-freeze 0 \
  --train-heads value \
  --value-loss-scale 1.0 \
  --value-distill-scale 0.40 \
  --policy-distill-scale 0.0 \
  --random-d4-augmentation \
  --validation-fraction 0.1 \
  --eval-every 2000 \
  --seed "$TRAIN_SEED"

sha256sum "$OUTPUT" > "$OUTPUT.sha256"
touch "$OUTPUT.done"
