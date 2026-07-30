#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$ROOT"

mkdir -p run_v3c
LOG="run_v3c/warm.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] V3C supervised warm-start begins"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON_BIN" --version

sha256sum -c <<'CHECKSUMS'
ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e  alphazero_training/latest.pt
24878729b53753ef60c2f36a438c9ecc19ec752a65ddf53ade45ba5d6c567b44  alphazero_training/v3_legal_tactics_train.npz
8014e027f2349010bd504a488d3646f0440567a85153d645d87366a9c5b1d5fc  alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz
144a751e8002e69d756ea6627564bb689f9b88cb98c60a3626aad726cbe79dfc  alphazero_training/ddqk_expert_50pairs_policy.npz
CHECKSUMS

for steps in 100 300 600; do
  output="run_v3c/warm_s${steps}.pt"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing candidate: $output" >&2
    exit 2
  fi
  echo "[$(date -Is)] training candidate steps=$steps"
  "$PYTHON_BIN" -m alphazero_training.train_v3_supervised \
    --init-checkpoint alphazero_training/latest.pt \
    --dataset alphazero_training/v3_legal_tactics_train.npz \
    --dataset alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz \
    --dataset alphazero_training/ddqk_expert_50pairs_policy.npz \
    --dataset-weight 0.05 \
    --dataset-weight 0.75 \
    --dataset-weight 0.20 \
    --output "$output" \
    --steps "$steps" \
    --batch-size 512 \
    --learning-rate 5e-5 \
    --min-learning-rate 5e-6 \
    --warmup-steps 25 \
    --freeze-trunk-steps 25 \
    --validation-fraction 0.15 \
    --eval-every 50 \
    --value-loss-scale 0.5 \
    --value-distill-scale 0.05 \
    --seed 20260730
  sha256sum "$output"
done

echo "[$(date -Is)] V3C supervised warm-start complete"
