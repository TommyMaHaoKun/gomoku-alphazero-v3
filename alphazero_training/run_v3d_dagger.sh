#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$ROOT"

mkdir -p run_v3d
LOG="run_v3d/dagger.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] V3D conservative DAgger correction begins"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON_BIN" --version

sha256sum -c <<'CHECKSUMS'
ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e  alphazero_training/latest.pt
24878729b53753ef60c2f36a438c9ecc19ec752a65ddf53ade45ba5d6c567b44  alphazero_training/v3_legal_tactics_train.npz
8014e027f2349010bd504a488d3646f0440567a85153d645d87366a9c5b1d5fc  alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz
144a751e8002e69d756ea6627564bb689f9b88cb98c60a3626aad726cbe79dfc  alphazero_training/ddqk_expert_50pairs_policy.npz
5dea0e542832038e750f8c256ffdf3b8a48ac9eccaa0093476d1f8023860fd75  alphazero_training/v3c_candidates_20260723/dagger_round1/parent_p2_failures_policy.npz
7bf17d8d084b183275031d5bb63531c609b1506922b7043818c6f3406c849646  alphazero_training/v3c_candidates_20260723/dagger_round1/parent_p4_failures_policy.npz
ce4e3c0171fa93ba6f18b3fd0f2a5cebcb4c8212bfde85bd58160ae43dc6b36a  alphazero_training/v3c_candidates_20260723/dagger_round1/parent_p8_failures_policy.npz
e2c4cd663f79c8ddf73142efcb56f86fd2b3951fcbab1caef6838c5ee828b56a  alphazero_training/v3c_candidates_20260723/ddqk_dev_seed20260724/parent_failures_policy.npz
01f7aca0d95b3575ef41864b3e688baaf37de0a38527b4660144ab3a57b3b9d7  alphazero_training/v3c_candidates_20260723/ddqk_dev_seed20260724/s100_failures_policy.npz
2580d09a002f1788059d726c920b5cd152051db3487035bda3f741def1eec2e1  alphazero_training/v3c_candidates_20260723/ddqk_dev_seed20260724/s300_failures_policy.npz
c1e74433f689bed81decfdef98360266bbd1759a210b5200cb25d6bac5c988f8  alphazero_training/v3c_candidates_20260723/ddqk_dev_seed20260724/s600_failures_policy.npz
CHECKSUMS

for steps in 25 75 150; do
  output="run_v3d/dagger_s${steps}.pt"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing candidate: $output" >&2
    exit 2
  fi
  echo "[$(date -Is)] training conservative candidate steps=$steps"
  "$PYTHON_BIN" -m alphazero_training.train_v3_supervised \
    --init-checkpoint alphazero_training/latest.pt \
    --dataset alphazero_training/v3_legal_tactics_train.npz \
    --dataset alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz \
    --dataset alphazero_training/ddqk_expert_50pairs_policy.npz \
    --dataset alphazero_training/v3c_candidates_20260723/dagger_round1/parent_p2_failures_policy.npz \
    --dataset alphazero_training/v3c_candidates_20260723/dagger_round1/parent_p4_failures_policy.npz \
    --dataset alphazero_training/v3c_candidates_20260723/dagger_round1/parent_p8_failures_policy.npz \
    --dataset alphazero_training/v3c_candidates_20260723/ddqk_dev_seed20260724/parent_failures_policy.npz \
    --dataset alphazero_training/v3c_candidates_20260723/ddqk_dev_seed20260724/s100_failures_policy.npz \
    --dataset alphazero_training/v3c_candidates_20260723/ddqk_dev_seed20260724/s300_failures_policy.npz \
    --dataset alphazero_training/v3c_candidates_20260723/ddqk_dev_seed20260724/s600_failures_policy.npz \
    --dataset-weight 0.04 \
    --dataset-weight 0.20 \
    --dataset-weight 0.16 \
    --dataset-weight 0.12 \
    --dataset-weight 0.10 \
    --dataset-weight 0.10 \
    --dataset-weight 0.08 \
    --dataset-weight 0.07 \
    --dataset-weight 0.07 \
    --dataset-weight 0.06 \
    --output "$output" \
    --steps "$steps" \
    --batch-size 512 \
    --learning-rate 1e-5 \
    --min-learning-rate 2e-6 \
    --warmup-steps 10 \
    --freeze-trunk-steps "$steps" \
    --validation-fraction 0.15 \
    --eval-every 25 \
    --value-loss-scale 0.10 \
    --value-distill-scale 0.50 \
    --policy-distill-scale 0.50 \
    --seed 20260726
  sha256sum "$output"
done

echo "[$(date -Is)] V3D conservative DAgger correction complete"
