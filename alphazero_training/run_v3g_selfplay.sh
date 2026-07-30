#!/usr/bin/env bash
set -euo pipefail

# V3G continues from the bounded white-margin warm start while carrying the
# externally approved champion unchanged in best_model/approved_model.
ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INIT_CHECKPOINT="${V3G_INIT_CHECKPOINT:-run_v3g_white_margin/latest.pt}"
OUT_DIR="${V3G_OUT_DIR:-run_v3g_white_selfplay}"
ITERATIONS="${V3G_ITERATIONS:-100}"
SNAPSHOT_INTERVAL="${V3G_SNAPSHOT_INTERVAL:-5}"
RESUME="${V3G_RESUME:-0}"

APPROVED_CHECKPOINT="alphazero_training/latest.pt"
APPROVED_SHA256="ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e"
EXPERT_DATA="alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz"
TACTICAL_DATA="alphazero_training/v3_legal_tactics_train.npz"
WHITE_DATA="alphazero_training/_v3g_white_train_det/p2/train.npz"
WHITE_MANIFEST="alphazero_training/_v3g_white_train_det/p2/manifest.json"

if [[ ! "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "V3G_ITERATIONS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$SNAPSHOT_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "V3G_SNAPSHOT_INTERVAL must be a positive integer" >&2
  exit 2
fi
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "V3G_RESUME must be 0 or 1" >&2
  exit 2
fi

cd "$ROOT"

sha256sum -c <<'CHECKSUMS'
ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e  alphazero_training/latest.pt
8014e027f2349010bd504a488d3646f0440567a85153d645d87366a9c5b1d5fc  alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz
24878729b53753ef60c2f36a438c9ecc19ec752a65ddf53ade45ba5d6c567b44  alphazero_training/v3_legal_tactics_train.npz
48c03543f104ab2ba701de2b80ca34b8335353f2ad837b512e4e0998ddee5dc1  alphazero_training/_v3g_white_train_det/p2/train.npz
b798e0b90436f2b365dc05630a34c6c069075fedecf9c6d636ee9f000b755615  alphazero_training/_v3g_white_train_det/p2/manifest.json
CHECKSUMS

"$PYTHON_BIN" - \
  "$INIT_CHECKPOINT" \
  "$APPROVED_SHA256" \
  "$WHITE_DATA" \
  "$WHITE_MANIFEST" <<'PY'
import hashlib
import sys
from pathlib import Path

import torch

from alphazero_training.train_alphazero import Config, PolicyValueNet
from alphazero_training.train_v3_selfplay import validate_white_defense_manifest

checkpoint_path = Path(sys.argv[1])
approved_sha = sys.argv[2]
white_data = Path(sys.argv[3])
white_manifest = Path(sys.argv[4])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if checkpoint.get("format_version") != 3:
    raise SystemExit("V3G init checkpoint is not format-version 3")
if checkpoint.get("v3_stage") != "tactical_expert_warmstart":
    raise SystemExit("V3G init checkpoint is not a supervised warm start")
if checkpoint.get("parent_checkpoint_sha256") != approved_sha:
    raise SystemExit("V3G warm start does not descend directly from the approved champion")
warm = checkpoint.get("warmstart_config")
if not isinstance(warm, dict) or float(warm.get("safe_hard_negative_scale", 0.0)) <= 0:
    raise SystemExit("V3G warm start did not enable the hard-negative objective")
config = Config(**checkpoint["config"])
probe = PolicyValueNet(config.board_size, config.channels, config.residual_blocks)
probe.load_state_dict(checkpoint["train_model"], strict=True)
validate_white_defense_manifest(white_data, white_manifest, board_size=config.board_size)
digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
print(f"validated_v3g_init={checkpoint_path} sha256={digest}")
PY

if [[ "$RESUME" == "1" ]]; then
  if [[ ! -f "$OUT_DIR/latest.pt" || ! -d "$OUT_DIR/replay" ]]; then
    echo "V3G resume requires an existing latest.pt and replay directory" >&2
    exit 2
  fi
  mkdir -p "$OUT_DIR/snapshots"
else
  if ! mkdir "$OUT_DIR" 2>/dev/null; then
    echo "Refusing to overwrite or restart existing V3G output: $OUT_DIR" >&2
    exit 2
  fi
  mkdir "$OUT_DIR/snapshots"
fi

trainer_pid=""
forward_stop() {
  if [[ -n "$trainer_pid" ]] && kill -0 "$trainer_pid" 2>/dev/null; then
    kill -TERM "$trainer_pid"
  fi
}
trap forward_stop EXIT INT TERM

snapshot_latest() {
  local latest="$OUT_DIR/latest.pt"
  [[ -f "$latest" ]] || return 0

  local temporary="$OUT_DIR/snapshots/.latest.$$.tmp"
  rm -f "$temporary"
  cp --reflink=auto "$latest" "$temporary"

  local iteration
  if ! iteration="$($PYTHON_BIN - "$temporary" "$APPROVED_SHA256" <<'PY'
import sys
import torch

path, approved_sha = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
if checkpoint.get("format_version") != 3 or checkpoint.get("v3_stage") != "selfplay":
    raise SystemExit("not a V3 self-play checkpoint")
if checkpoint.get("approved_checkpoint_sha256") != approved_sha:
    raise SystemExit("approved champion provenance changed")
iteration = checkpoint.get("iteration")
if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
    raise SystemExit("checkpoint iteration is invalid")
if checkpoint.get("best_model") is None or checkpoint.get("candidate_model") is None:
    raise SystemExit("checkpoint lost champion/candidate separation")
print(iteration)
PY
  )"; then
    rm -f "$temporary"
    echo "A published V3G latest.pt failed validation" >&2
    return 1
  fi

  # The trainer deliberately publishes an authenticated iteration-zero
  # checkpoint before game one.  It is a valid resume boundary, but not a
  # candidate snapshot.
  if (( iteration == 0 )); then
    rm -f "$temporary"
    return 0
  fi

  if (( iteration % SNAPSHOT_INTERVAL != 0 )); then
    rm -f "$temporary"
    return 0
  fi

  local label destination sidecar digest sidecar_tmp
  printf -v label '%04d' "$iteration"
  destination="$OUT_DIR/snapshots/iteration_${label}.pt"
  sidecar="${destination}.sha256"
  digest="$(sha256sum "$temporary" | awk '{print $1}')"

  if ! ln "$temporary" "$destination" 2>/dev/null; then
    local existing
    existing="$(sha256sum "$destination" | awk '{print $1}')"
    if [[ "$existing" != "$digest" ]]; then
      rm -f "$temporary"
      echo "Immutable V3G snapshot collision at iteration $iteration" >&2
      return 1
    fi
  fi
  rm -f "$temporary"

  sidecar_tmp="${sidecar}.tmp.$$"
  printf '%s  %s\n' "$digest" "$destination" > "$sidecar_tmp"
  if ! ln "$sidecar_tmp" "$sidecar" 2>/dev/null; then
    if ! cmp -s "$sidecar_tmp" "$sidecar"; then
      rm -f "$sidecar_tmp"
      echo "Immutable V3G SHA256 sidecar collision at iteration $iteration" >&2
      return 1
    fi
  fi
  rm -f "$sidecar_tmp"
  echo "[$(date -Is)] preserved V3G candidate snapshot: $destination sha256=$digest"
}

trainer_args=(--output-dir "$OUT_DIR")
if [[ "$RESUME" == "1" ]]; then
  trainer_args+=(--resume)
else
  trainer_args+=(
    --init-checkpoint "$INIT_CHECKPOINT"
    --approved-checkpoint "$APPROVED_CHECKPOINT"
    --expert-npz "$EXPERT_DATA"
    --tactical-npz "$TACTICAL_DATA"
    --white-defense-npz "$WHITE_DATA"
    --white-defense-manifest "$WHITE_MANIFEST"
    --iterations "$ITERATIONS"
    --selfplay-games 32
    --parallel-games 32
    --simulations 384
    --temperature-moves 15
    --max-game-plies 361
    --train-steps 200
    --batch-size 384
    --learning-rate 5e-5
    --min-learning-rate 5e-6
    --warmup-steps 200
    --weight-decay 1e-4
    --replay-capacity 250000
    --max-replay-chunks 150
    --selfplay-quota 0.45
    --ddqk-quota 0.25
    --tactical-quota 0.15
    --white-defense-quota 0.15
    --safe-hard-negative-scale 0.15
    --safe-hard-negative-margin 0.5
    --selfplay-policy-weight 1.0
    --selfplay-value-weight 1.0
    --seed 20260811
    --log-every-steps 25
  )
fi

"$PYTHON_BIN" -u -m alphazero_training.train_v3_selfplay "${trainer_args[@]}" &
trainer_pid="$!"

status=0
while kill -0 "$trainer_pid" 2>/dev/null; do
  snapshot_latest
  sleep 30 &
  wait "$!" || true
done
wait "$trainer_pid" || status="$?"
trainer_pid=""
snapshot_latest
exit "$status"
