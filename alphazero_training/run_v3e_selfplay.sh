#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${V3E_OUT_DIR:-run_v3e_s384}"
SNAPSHOT_INTERVAL="${V3E_SNAPSHOT_INTERVAL:-5}"

cd "$ROOT"

sha256sum -c <<'CHECKSUMS'
ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e  alphazero_training/latest.pt
c333d6ed0395fae9105328378bf0c57586bd39755af206da1ac01e0f6d74f1f8  run_v3d/dagger_s25.pt
8014e027f2349010bd504a488d3646f0440567a85153d645d87366a9c5b1d5fc  alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz
24878729b53753ef60c2f36a438c9ecc19ec752a65ddf53ade45ba5d6c567b44  alphazero_training/v3_legal_tactics_train.npz
CHECKSUMS

if [[ ! "$SNAPSHOT_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "V3E_SNAPSHOT_INTERVAL must be a positive integer" >&2
  exit 2
fi
if [[ -e "$OUT_DIR/latest.pt" || -d "$OUT_DIR/replay" ]]; then
  echo "Refusing to overwrite or restart existing run: $OUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR/snapshots"

trainer_pid=""
forward_stop() {
  if [[ -n "$trainer_pid" ]] && kill -0 "$trainer_pid" 2>/dev/null; then
    kill -TERM "$trainer_pid"
  fi
}
trap forward_stop INT TERM

snapshot_latest() {
  local log_file="$OUT_DIR/train_v3.log"
  [[ -f "$log_file" && -f "$OUT_DIR/latest.pt" ]] || return 0

  local iteration
  iteration="$(sed -n 's/.*iteration=\([0-9][0-9]*\) complete.*/\1/p' "$log_file" | tail -n 1)"
  [[ -n "$iteration" ]] || return 0
  (( iteration > 0 && iteration % SNAPSHOT_INTERVAL == 0 )) || return 0

  local label destination temporary
  printf -v label '%04d' "$iteration"
  destination="$OUT_DIR/snapshots/iteration_${label}.pt"
  [[ ! -e "$destination" ]] || return 0
  temporary="${destination}.tmp.$$"
  cp --reflink=auto "$OUT_DIR/latest.pt" "$temporary"
  if mv -n "$temporary" "$destination"; then
    sha256sum "$destination" > "${destination}.sha256"
    echo "[$(date -Is)] preserved candidate snapshot: $destination"
  else
    rm -f "$temporary"
  fi
}

"$PYTHON_BIN" -u -m alphazero_training.train_v3_selfplay \
  --output-dir "$OUT_DIR" \
  --init-checkpoint run_v3d/dagger_s25.pt \
  --approved-checkpoint alphazero_training/latest.pt \
  --expert-npz alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz \
  --tactical-npz alphazero_training/v3_legal_tactics_train.npz \
  --iterations 100 \
  --selfplay-games 32 \
  --parallel-games 32 \
  --simulations 384 \
  --temperature-moves 15 \
  --max-game-plies 361 \
  --train-steps 200 \
  --batch-size 384 \
  --learning-rate 1e-4 \
  --min-learning-rate 1e-5 \
  --warmup-steps 200 \
  --weight-decay 1e-4 \
  --replay-capacity 250000 \
  --max-replay-chunks 150 \
  --selfplay-quota 0.50 \
  --ddqk-quota 0.35 \
  --tactical-quota 0.15 \
  --selfplay-policy-weight 1.0 \
  --selfplay-value-weight 1.0 \
  --seed 20260727 \
  --log-every-steps 25 &
trainer_pid="$!"

status=0
while kill -0 "$trainer_pid" 2>/dev/null; do
  snapshot_latest
  sleep 30 &
  wait "$!" || true
done
wait "$trainer_pid" || status="$?"
snapshot_latest
exit "$status"
