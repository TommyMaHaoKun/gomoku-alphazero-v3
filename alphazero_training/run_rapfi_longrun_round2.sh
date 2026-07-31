#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN="rapfi_distillation/longrun_round2_dagger2048"
CANDIDATE="rapfi_distillation/loss_correction_round1/blend_sweep/candidate_blend_a060_eval.pt"
CANDIDATE_SHA256="378f5715541c95972cde396f4023b63b3f1aa83de6a04aa78af934e5253adada"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"
ENGINE_SHA256="95cea7a20374f8fa2795d4b410b215a7f058c9e2b659d8c0670bcb86761e6f46"

cd "$ROOT"
mkdir -p "$RUN"

actual_candidate="$(sha256sum "$CANDIDATE" | awk '{print $1}')"
actual_engine="$(sha256sum "$ENGINE" | awk '{print $1}')"
if [[ "$actual_candidate" != "$CANDIDATE_SHA256" ]]; then
  echo "candidate SHA256 mismatch: $actual_candidate" >&2
  exit 2
fi
if [[ "$actual_engine" != "$ENGINE_SHA256" ]]; then
  echo "Rapfi engine SHA256 mismatch: $actual_engine" >&2
  exit 2
fi

"$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
  --checkpoint "$CANDIDATE" \
  --engine "$ENGINE" \
  --report "$RUN/games.json" \
  --dataset "$RUN/rapfi_policy.npz" \
  --ai-loss-dir "$RUN/pending_training/ai_losses" \
  --pairs 2048 \
  --opening-plies 4 \
  --simulations 64 \
  --workers 8 \
  --timeout-turn-ms 300 \
  --max-nodes 100000 \
  --engine-threads 4 \
  --seed 20270101
