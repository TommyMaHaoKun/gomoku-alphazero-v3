#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"
INIT="rapfi_distillation/desktop_loss_repair_round5c/training/last3_repair_s8000.pt"
INIT_SHA256="eedaf5723ae15d32f490684982c8243baa6b68e477dedcb3f7c646543136a5c9"
ROUND6="rapfi_distillation/round6_regret_restart"
OUT="rapfi_distillation/round7_league_selfplay"

cd "$ROOT"
mkdir -p "$OUT"
[[ "$(sha256sum "$PARENT" | awk '{print $1}')" == "$PARENT_SHA256" ]]
[[ "$(sha256sum "$INIT" | awk '{print $1}')" == "$INIT_SHA256" ]]
[[ -s "$ROUND6/DATA_STAGE_COMPLETE.json" ]]

"$PYTHON_BIN" -u -m alphazero_training.train_v3_selfplay \
  --output-dir "$OUT" \
  --init-checkpoint "$INIT" \
  --approved-checkpoint "$PARENT" \
  --expert-npz "$ROUND6/datasets/joint_current_champion.npz" \
  --tactical-npz alphazero_training/v3_legal_tactics_train.npz \
  --white-defense-npz alphazero_training/_v3f_white_train_det/p2/train.npz \
  --white-defense-manifest alphazero_training/_v3f_white_train_det/p2/manifest.json \
  --iterations 50 \
  --selfplay-games 16 \
  --parallel-games 16 \
  --simulations 128 \
  --temperature-moves 12 \
  --max-game-plies 361 \
  --train-steps 100 \
  --batch-size 512 \
  --learning-rate 2e-5 \
  --min-learning-rate 2e-6 \
  --warmup-steps 200 \
  --weight-decay 1e-4 \
  --replay-capacity 150000 \
  --max-replay-chunks 60 \
  --selfplay-quota 0.45 \
  --ddqk-quota 0.25 \
  --tactical-quota 0.15 \
  --white-defense-quota 0.15 \
  --safe-hard-negative-scale 0.60 \
  --safe-hard-negative-margin 1.0 \
  --selfplay-policy-weight 1.0 \
  --selfplay-value-weight 1.0 \
  --seed 20280805 \
  --log-every-steps 25
