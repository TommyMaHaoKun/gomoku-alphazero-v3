#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN="rapfi_distillation/round6_regret_restart"
OUTPUT="$RUN/training/r6_guard_s10000.pt"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"

cd "$ROOT"
actual_parent_sha="$(sha256sum "$PARENT" | awk '{print $1}')"
[[ "$actual_parent_sha" == "$PARENT_SHA256" ]] || {
  echo "parent SHA256 mismatch: $actual_parent_sha" >&2
  exit 2
}
[[ -s "$RUN/DATA_STAGE_COMPLETE.json" ]] || {
  echo "Round6 data stage is not complete" >&2
  exit 3
}

white_args=()
for white_name in p2 p4 p8 seed20260724_parent seed20260730_parent; do
  white_args+=(
    --white-defense-npz "alphazero_training/_v3f_white_train_det/$white_name/train.npz"
    --white-defense-manifest "alphazero_training/_v3f_white_train_det/$white_name/manifest.json"
    --white-defense-weight 0.08
  )
done

"$PYTHON_BIN" -u -m alphazero_training.train_v3_supervised \
  --init-checkpoint "$PARENT" \
  --dataset "$RUN/datasets/joint_current_champion.npz" \
  --dataset "$RUN/counterfactual/branches.npz" \
  --dataset "$RUN/datasets/regret_loss_curriculum.npz" \
  --dataset rapfi_distillation/longrun_counterfactual_round3/branches.npz \
  --dataset alphazero_training/v3_legal_tactics_train.npz \
  --dataset alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz \
  --dataset alphazero_training/ddqk_expert_50pairs_policy.npz \
  --dataset rapfi_distillation/desktop_loss_corrections/full_review/desktop_full_loss_curriculum.npz \
  --dataset rapfi_distillation/desktop_loss_corrections/move14_counterfactual_branch.npz \
  --dataset-weight 0.16 \
  --dataset-weight 0.20 \
  --dataset-weight 0.10 \
  --dataset-weight 0.04 \
  --dataset-weight 0.14 \
  --dataset-weight 0.03 \
  --dataset-weight 0.02 \
  --dataset-weight 0.05 \
  --dataset-weight 0.03 \
  "${white_args[@]}" \
  --output "$OUTPUT" \
  --steps 10000 --batch-size 1024 \
  --learning-rate 2e-6 --min-learning-rate 2e-7 \
  --warmup-steps 500 \
  --freeze-trunk-steps 7500 \
  --train-last-residual-blocks-during-freeze 2 \
  --value-loss-scale 0.12 \
  --value-distill-scale 0.25 \
  --policy-distill-scale 0.90 \
  --safe-hard-negative-scale 1.20 \
  --safe-hard-negative-margin 1.2 \
  --mistake-hard-negative-scale 1.00 \
  --mistake-hard-negative-margin 1.0 \
  --random-d4-augmentation \
  --validation-fraction 0.1 \
  --eval-every 2500 \
  --seed 20280403

sha256sum "$OUTPUT" > "$OUTPUT.sha256"
touch "$OUTPUT.done"
