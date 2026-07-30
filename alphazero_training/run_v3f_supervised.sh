#!/usr/bin/env bash
set -euo pipefail

# V3F is a bounded white-defence supervised stage.  The approved checkpoint
# below is deliberately not configurable: this run must never inherit V3E.
ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WHITE_ROOT="${V3F_WHITE_ROOT:-alphazero_training/_v3f_white_train_det}"
OUT_DIR="${V3F_OUT_DIR:-run_v3f_white_supervised}"
STEPS="${V3F_STEPS:-500}"
TRAIN_SEED="${V3F_TRAIN_SEED:-20260803}"
FINAL_EVAL_SEED="${V3F_FINAL_EVAL_SEED:-20260817}"
SNAPSHOT_POLL_SECONDS="${V3F_SNAPSHOT_POLL_SECONDS:-0.5}"
WHITE_WEIGHT="${V3F_WHITE_WEIGHT:-0.09}"
LEARNING_RATE="${V3F_LEARNING_RATE:-1e-5}"
MIN_LEARNING_RATE="${V3F_MIN_LEARNING_RATE:-5e-6}"
POLICY_DISTILL_SCALE="${V3F_POLICY_DISTILL_SCALE:-0.1}"
VALUE_LOSS_SCALE="${V3F_VALUE_LOSS_SCALE:-0.5}"
VALUE_DISTILL_SCALE="${V3F_VALUE_DISTILL_SCALE:-0.1}"
FREEZE_TRUNK_STEPS="${V3F_FREEZE_TRUNK_STEPS:-300}"
TRAIN_LAST_RESIDUAL_BLOCKS="${V3F_TRAIN_LAST_RESIDUAL_BLOCKS:-2}"
SAFE_HARD_NEGATIVE_SCALE="${V3F_SAFE_HARD_NEGATIVE_SCALE:-0}"
SAFE_HARD_NEGATIVE_MARGIN="${V3F_SAFE_HARD_NEGATIVE_MARGIN:-1.0}"

APPROVED_CHECKPOINT="alphazero_training/latest.pt"
APPROVED_SHA256="ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e"
TACTICAL_DATA="alphazero_training/v3_legal_tactics_train.npz"
TEACHER_DATA="alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz"
EXPERT_DATA="alphazero_training/ddqk_expert_50pairs_policy.npz"

white_names=(
  p2
  p4
  p8
  seed20260724_parent
  seed20260730_parent
)
white_manifest_sha256=(
  b798e0b90436f2b365dc05630a34c6c069075fedecf9c6d636ee9f000b755615
  b5aaf117f3aa43ba9d4cb637a8af283bf263688df59b6a5d50084baa906e11e9
  dfd29ede024d2554d5b5c2ba8d6b71895fbb1d887862d97a91f8279fc2d220d2
  97505a56e41901d6ee39af5749449add2b66873a3420d6ae059710ee87bebf4c
  9d9edddcba8ce0a8f2853fb28559b1f5ee874bc350dd1ef5ede963b205cf2b2c
)
white_train_sha256=(
  48c03543f104ab2ba701de2b80ca34b8335353f2ad837b512e4e0998ddee5dc1
  0462140c843ebc149c1b4fb9c05888ed2037a6986b2edeb0e9cfa1956d34ee1b
  6ea8d45fff42be9342a27e5ab5003c643b2ed070e3e02632348dbc737cdf30e7
  058cacfe6cd87797666febc7eed573a1e76565c3b5b1f44095282a1a70ea8104
  d38bde5a538ea3e98e6f4fe0b99b4ce884b3d8853fee5d1eb904b8a1e799ab7c
)
white_report_sha256=(
  3a027375f446f69796c848c9f446c29d9fa7dc182169814727e252a9faa63790
  c33167ad0f0fde82e325cf9da110b28fbfee0ac9e3534aec0afaa90853fcac35
  672e1559b6db69207e88032041be4088acf16640c9234c2b3612b4bf30682a09
  b3ba30279ba95844ba53d644474faa9d97c1ff4cfedd8143e0b101e6d98fda46
  8bee658c85669483bfecc6cfd984a92ecb1db8e0e13679a3c9c2b3a60426f01e
)

if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "V3F_STEPS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$TRAIN_SEED" =~ ^[0-9]+$ || ! "$FINAL_EVAL_SEED" =~ ^[0-9]+$ ]]; then
  echo "V3F_TRAIN_SEED and V3F_FINAL_EVAL_SEED must be non-negative integers" >&2
  exit 2
fi
if [[ "$TRAIN_SEED" == "$FINAL_EVAL_SEED" ]]; then
  echo "Training and final-evaluation seeds must be different" >&2
  exit 2
fi
if ! [[ "$SNAPSHOT_POLL_SECONDS" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "V3F_SNAPSHOT_POLL_SECONDS must be a non-negative number" >&2
  exit 2
fi

cd "$ROOT"

"$PYTHON_BIN" - \
  "$WHITE_WEIGHT" \
  "$LEARNING_RATE" \
  "$MIN_LEARNING_RATE" \
  "$POLICY_DISTILL_SCALE" \
  "$VALUE_LOSS_SCALE" \
  "$VALUE_DISTILL_SCALE" \
  "$FREEZE_TRUNK_STEPS" \
  "$TRAIN_LAST_RESIDUAL_BLOCKS" \
  "$SAFE_HARD_NEGATIVE_SCALE" \
  "$SAFE_HARD_NEGATIVE_MARGIN" <<'PY'
import math
import sys

labels = (
    "V3F_WHITE_WEIGHT",
    "V3F_LEARNING_RATE",
    "V3F_MIN_LEARNING_RATE",
    "V3F_POLICY_DISTILL_SCALE",
    "V3F_VALUE_LOSS_SCALE",
    "V3F_VALUE_DISTILL_SCALE",
    "V3F_FREEZE_TRUNK_STEPS",
    "V3F_TRAIN_LAST_RESIDUAL_BLOCKS",
    "V3F_SAFE_HARD_NEGATIVE_SCALE",
    "V3F_SAFE_HARD_NEGATIVE_MARGIN",
)
try:
    values = [float(value) for value in sys.argv[1:]]
except ValueError as error:
    raise SystemExit(f"V3F numeric configuration is invalid: {error}") from error
if not all(math.isfinite(value) for value in values):
    raise SystemExit("V3F numeric configuration must be finite")
(
    white_weight,
    learning_rate,
    minimum_rate,
    policy_kd,
    value_scale,
    value_kd,
    freeze_steps,
    last_blocks,
    safe_scale,
    safe_margin,
) = values
if white_weight <= 0 or learning_rate <= 0 or minimum_rate < 0:
    raise SystemExit("V3F data weight and learning rates are invalid")
if minimum_rate > learning_rate:
    raise SystemExit("V3F_MIN_LEARNING_RATE cannot exceed V3F_LEARNING_RATE")
if policy_kd < 0 or value_scale < 0 or value_kd < 0 or safe_scale < 0 or safe_margin < 0:
    raise SystemExit("V3F distillation and hard-negative settings cannot be negative")
if freeze_steps < 0 or not freeze_steps.is_integer():
    raise SystemExit("V3F_FREEZE_TRUNK_STEPS must be a non-negative integer")
if last_blocks < 0 or not last_blocks.is_integer():
    raise SystemExit("V3F_TRAIN_LAST_RESIDUAL_BLOCKS must be a non-negative integer")
print("validated configurable V3F/V3G optimization settings")
PY

# mkdir itself is the non-overwrite lock.  This avoids a check-then-create race
# between concurrent launchers; an existing path is never reused or resumed.
if ! mkdir "$OUT_DIR" 2>/dev/null; then
  if [[ -e "$OUT_DIR" ]]; then
    echo "Refusing to overwrite or resume an existing V3F output: $OUT_DIR" >&2
  else
    echo "Unable to atomically create V3F output directory: $OUT_DIR" >&2
  fi
  exit 2
fi
if ! mkdir "$OUT_DIR/snapshots"; then
  echo "Unable to create V3F snapshot directory" >&2
  exit 2
fi
LOG="$OUT_DIR/train_v3f.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] V3F white-defence supervised stage begins"
echo "root=$(pwd)"
echo "approved_checkpoint=$APPROVED_CHECKPOINT"
echo "output=$OUT_DIR/latest.pt steps=$STEPS"
echo "train_seed=$TRAIN_SEED reserved_final_eval_seed=$FINAL_EVAL_SEED"
echo "white_root=$WHITE_ROOT"
echo "white_weight_each=$WHITE_WEIGHT learning_rate=$LEARNING_RATE min_learning_rate=$MIN_LEARNING_RATE"
echo "policy_distill_scale=$POLICY_DISTILL_SCALE value_loss_scale=$VALUE_LOSS_SCALE value_distill_scale=$VALUE_DISTILL_SCALE"
echo "freeze_trunk_steps=$FREEZE_TRUNK_STEPS train_last_residual_blocks=$TRAIN_LAST_RESIDUAL_BLOCKS"
echo "safe_hard_negative_scale=$SAFE_HARD_NEGATIVE_SCALE safe_hard_negative_margin=$SAFE_HARD_NEGATIVE_MARGIN"
"$PYTHON_BIN" --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# Authenticate the immutable parent and the three ordinary curriculum assets.
sha256sum -c <<'CHECKSUMS'
ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e  alphazero_training/latest.pt
24878729b53753ef60c2f36a438c9ecc19ec752a65ddf53ade45ba5d6c567b44  alphazero_training/v3_legal_tactics_train.npz
8014e027f2349010bd504a488d3646f0440567a85153d645d87366a9c5b1d5fc  alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz
144a751e8002e69d756ea6627564bb689f9b88cb98c60a3626aad726cbe79dfc  alphazero_training/ddqk_expert_50pairs_policy.npz
CHECKSUMS

approved_resolved="$($PYTHON_BIN -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$APPROVED_CHECKPOINT")"
output_resolved="$($PYTHON_BIN -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$OUT_DIR/latest.pt")"
if [[ "$approved_resolved" == "$output_resolved" ]]; then
  echo "V3F output would overwrite the approved checkpoint" >&2
  exit 2
fi

white_args=()
white_datasets=()
for index in "${!white_names[@]}"; do
  name="${white_names[$index]}"
  dataset="$WHITE_ROOT/$name/train.npz"
  manifest="$WHITE_ROOT/$name/manifest.json"
  "$PYTHON_BIN" - \
    "$dataset" \
    "$manifest" \
    "${white_manifest_sha256[$index]}" \
    "${white_train_sha256[$index]}" \
    "${white_report_sha256[$index]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from alphazero_training.train_v3_selfplay import validate_white_defense_manifest

dataset = Path(sys.argv[1])
manifest = Path(sys.argv[2])
expected_manifest_sha = sys.argv[3]
expected_train_sha = sys.argv[4]
expected_report_sha = sys.argv[5]
provenance = validate_white_defense_manifest(dataset, manifest, board_size=19)
payload = json.loads(manifest.read_text(encoding="utf-8"))
config = payload.get("config")
if not isinstance(config, dict):
    raise SystemExit(f"white-defense manifest has no config object: {manifest}")
if config.get("deterministic_node_budget_only") is not True:
    raise SystemExit(f"white-defense manifest is not deterministic node-budget-only: {manifest}")
vcf_time_ms = config.get("vcf_time_ms_per_query")
if (
    isinstance(vcf_time_ms, bool)
    or not isinstance(vcf_time_ms, (int, float))
    or float(vcf_time_ms) != 0.0
):
    raise SystemExit(f"white-defense manifest has a non-zero wall-clock VCF budget: {manifest}")
actual_manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
if actual_manifest_sha != expected_manifest_sha:
    raise SystemExit(
        f"fixed manifest SHA256 mismatch for {manifest}: {actual_manifest_sha}"
    )
if provenance["train_sha256"] != expected_train_sha:
    raise SystemExit(
        f"fixed train SHA256 mismatch for {dataset}: {provenance['train_sha256']}"
    )
if provenance["report_sha256"] != expected_report_sha:
    raise SystemExit(
        f"fixed report SHA256 mismatch for {manifest}: {provenance['report_sha256']}"
    )
records = payload["artifacts"]["train"]["records"]
print(
    "authenticated_white_source="
    f"{dataset} records={records} "
    f"manifest_sha256={actual_manifest_sha} "
    f"train_sha256={provenance['train_sha256']} "
    f"report_sha256={provenance['report_sha256']} "
    f"provenance_generation={provenance['provenance_generation']}"
)
PY
  white_args+=(
    --white-defense-npz "$dataset"
    --white-defense-manifest "$manifest"
    --white-defense-weight "$WHITE_WEIGHT"
  )
  white_datasets+=("$dataset")
done

# Individual manifests can each be self-consistent while disagreeing with one
# another.  Authenticate the union as one deterministic curriculum as well.
"$PYTHON_BIN" - "${white_datasets[@]}" <<'PY'
import sys
from pathlib import Path

import numpy as np

mask_names = (
    "candidate_mask",
    "safe_mask",
    "vcf_unknown_mask",
    "unsafe_immediate_mask",
    "unsafe_three_ply_mask",
    "unsafe_vcf_mask",
)
labels_by_state = {}
source_by_state = {}
duplicate_states = 0
records = 0
for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    with np.load(path, allow_pickle=False) as archive:
        records += len(archive["states"])
        for index, raw_state_hash in enumerate(archive["state_hash"]):
            state_hash = str(raw_state_hash).lower()
            signature = tuple(
                np.ascontiguousarray(archive[name][index]).tobytes()
                for name in mask_names
            )
            if state_hash in labels_by_state:
                duplicate_states += 1
                if labels_by_state[state_hash] != signature:
                    raise SystemExit(
                        "cross-source white-defense conflict for state "
                        f"{state_hash}: {source_by_state[state_hash]} versus {path}"
                    )
            else:
                labels_by_state[state_hash] = signature
                source_by_state[state_hash] = str(path)
print(
    f"authenticated_white_union=records:{records} "
    f"unique_states:{len(labels_by_state)} consistent_duplicates:{duplicate_states}"
)
PY

echo "ordinary_mix=tactical:0.30,ddqk_teacher:0.15,expert:0.10"
echo "white_mix=p2:$WHITE_WEIGHT,p4:$WHITE_WEIGHT,p8:$WHITE_WEIGHT,seed20260724_parent:$WHITE_WEIGHT,seed20260730_parent:$WHITE_WEIGHT"
echo "[$(date -Is)] preflight complete; launching trainer from approved SHA256=$APPROVED_SHA256"

trainer_pid=""
forward_stop() {
  if [[ -n "$trainer_pid" ]] && kill -0 "$trainer_pid" 2>/dev/null; then
    kill -TERM "$trainer_pid"
  fi
}
trap forward_stop EXIT INT TERM

# Validate both checkpoint semantics and its exact sidecar.  This is also the
# losing-concurrent-launcher path: an existing target is accepted only when it
# is already the intended immutable snapshot with the intended digest.
validate_preserved_snapshot() {
  local path="$1"
  local sidecar="$2"
  local expected_step="$3"
  local expected_snapshot_sha="$4"
  "$PYTHON_BIN" - \
    "$path" \
    "$sidecar" \
    "$expected_step" \
    "$expected_snapshot_sha" \
    "$APPROVED_SHA256" <<'PY'
import hashlib
import re
import sys
from pathlib import Path

import torch

from alphazero_training.train_alphazero import Config, PolicyValueNet

path = Path(sys.argv[1])
sidecar = Path(sys.argv[2])
expected_step = int(sys.argv[3])
expected_snapshot_sha = sys.argv[4]
expected_parent_sha = sys.argv[5]
if not path.is_file() or not sidecar.is_file():
    raise SystemExit("preserved snapshot or SHA256 sidecar is missing")
lines = sidecar.read_text(encoding="utf-8").splitlines()
if len(lines) != 1:
    raise SystemExit("snapshot SHA256 sidecar must contain exactly one line")
parts = lines[0].split(maxsplit=1)
if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
    raise SystemExit("snapshot SHA256 sidecar is malformed")
declared_sha = parts[0].lower()
declared_path = parts[1].lstrip("*")
if declared_path != str(path):
    raise SystemExit("snapshot SHA256 sidecar names another path")
actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
if actual_sha != declared_sha:
    raise SystemExit("existing snapshot SHA256 does not match its sidecar")
if actual_sha != expected_snapshot_sha:
    raise SystemExit("existing snapshot differs from the checkpoint being preserved")

checkpoint = torch.load(path, map_location="cpu", weights_only=False)
if checkpoint.get("format_version") != 3:
    raise SystemExit("existing snapshot is not format-version 3")
if checkpoint.get("v3_stage") != "tactical_expert_warmstart":
    raise SystemExit("existing snapshot is not a V3 supervised warm-start")
if checkpoint.get("parent_checkpoint_sha256") != expected_parent_sha:
    raise SystemExit("existing snapshot does not descend from the approved champion")
if checkpoint.get("global_step") != expected_step:
    raise SystemExit("existing snapshot has the wrong global_step")
config = Config(**checkpoint["config"])
model = PolicyValueNet(config.board_size, config.channels, config.residual_blocks)
state = checkpoint.get("train_model")
if not isinstance(state, dict):
    raise SystemExit("existing snapshot has no V3 supervised train_model")
model.load_state_dict(state, strict=True)
model.eval()
print(f"validated_snapshot={path} sha256={actual_sha} global_step={expected_step}")
PY
}

# Copy an atomically published latest.pt to a private temporary name, validate
# that stable copy, and then claim immutable names with hard links.  link(2)
# fails atomically when a destination exists, unlike GNU `mv -n`'s zero status.
last_seen_signature=""
snapshot_latest() {
  local latest="$OUT_DIR/latest.pt"
  [[ -f "$latest" ]] || return 0

  local signature
  signature="$(stat -c '%i:%s:%y' "$latest")"
  [[ "$signature" != "$last_seen_signature" ]] || return 0
  last_seen_signature="$signature"

  local temporary="$OUT_DIR/snapshots/.latest.$$.tmp"
  rm -f "$temporary"
  cp --reflink=auto "$latest" "$temporary"

  local step
  if ! step="$($PYTHON_BIN - "$temporary" "$APPROVED_SHA256" <<'PY'
import sys
from pathlib import Path

import torch

from alphazero_training.train_alphazero import Config, PolicyValueNet

path = Path(sys.argv[1])
expected_parent_sha = sys.argv[2]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
if checkpoint.get("format_version") != 3:
    raise SystemExit("snapshot is not a format-version 3 checkpoint")
if checkpoint.get("v3_stage") != "tactical_expert_warmstart":
    raise SystemExit("snapshot is not a V3 supervised warm-start checkpoint")
if checkpoint.get("parent_checkpoint_sha256") != expected_parent_sha:
    raise SystemExit("snapshot does not descend directly from the approved champion")
step = checkpoint.get("global_step")
if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
    raise SystemExit("snapshot has no valid positive global_step")
config = Config(**checkpoint["config"])
model = PolicyValueNet(config.board_size, config.channels, config.residual_blocks)
state = checkpoint.get("train_model")
if not isinstance(state, dict):
    raise SystemExit("snapshot has no V3 supervised train_model candidate state")
model.load_state_dict(state, strict=True)
model.eval()
print(step)
PY
  )"; then
    rm -f "$temporary"
    echo "A newly published latest.pt failed load/provenance validation" >&2
    return 1
  fi

  if (( step % 250 != 0 )); then
    rm -f "$temporary"
    return 0
  fi

  local label destination sidecar temporary_sidecar
  printf -v label '%06d' "$step"
  destination="$OUT_DIR/snapshots/global_step_${label}.pt"
  sidecar="${destination}.sha256"
  temporary_sidecar="${sidecar}.tmp.$$"

  local digest
  digest="$(sha256sum "$temporary" | awk '{print $1}')"
  printf '%s  %s\n' "$digest" "$destination" > "$temporary_sidecar"

  if ! ln "$temporary" "$destination" 2>/dev/null; then
    rm -f "$temporary" "$temporary_sidecar"
    validate_preserved_snapshot "$destination" "$sidecar" "$step" "$digest"
    echo "[$(date -Is)] retained valid concurrent immutable snapshot: $destination"
    return 0
  fi
  rm -f "$temporary"

  if ! ln "$temporary_sidecar" "$sidecar" 2>/dev/null; then
    rm -f "$temporary_sidecar"
    validate_preserved_snapshot "$destination" "$sidecar" "$step" "$digest"
    echo "[$(date -Is)] retained valid concurrent SHA256 sidecar: $sidecar"
    return 0
  fi
  rm -f "$temporary_sidecar"
  validate_preserved_snapshot "$destination" "$sidecar" "$step" "$digest"
  echo "[$(date -Is)] preserved immutable snapshot: $destination"
}

"$PYTHON_BIN" -u -m alphazero_training.train_v3_supervised \
  --init-checkpoint "$APPROVED_CHECKPOINT" \
  --dataset "$TACTICAL_DATA" \
  --dataset "$TEACHER_DATA" \
  --dataset "$EXPERT_DATA" \
  --dataset-weight 0.30 \
  --dataset-weight 0.15 \
  --dataset-weight 0.10 \
  "${white_args[@]}" \
  --output "$OUT_DIR/latest.pt" \
  --steps "$STEPS" \
  --batch-size 512 \
  --learning-rate "$LEARNING_RATE" \
  --min-learning-rate "$MIN_LEARNING_RATE" \
  --warmup-steps 100 \
  --freeze-trunk-steps "$FREEZE_TRUNK_STEPS" \
  --train-last-residual-blocks-during-freeze "$TRAIN_LAST_RESIDUAL_BLOCKS" \
  --value-loss-scale "$VALUE_LOSS_SCALE" \
  --value-distill-scale "$VALUE_DISTILL_SCALE" \
  --policy-distill-scale "$POLICY_DISTILL_SCALE" \
  --safe-hard-negative-scale "$SAFE_HARD_NEGATIVE_SCALE" \
  --safe-hard-negative-margin "$SAFE_HARD_NEGATIVE_MARGIN" \
  --validation-fraction 0.1 \
  --eval-every 250 \
  --seed "$TRAIN_SEED" &
trainer_pid="$!"

status=0
while kill -0 "$trainer_pid" 2>/dev/null; do
  snapshot_latest
  sleep "$SNAPSHOT_POLL_SECONDS" &
  wait "$!" || true
done
wait "$trainer_pid" || status="$?"
trainer_pid=""
snapshot_latest
if (( status != 0 )); then
  echo "[$(date -Is)] trainer failed with exit status $status" >&2
  exit "$status"
fi

final_step="$($PYTHON_BIN - "$OUT_DIR/latest.pt" "$APPROVED_SHA256" <<'PY'
import sys
from pathlib import Path
import torch

from alphazero_training.train_alphazero import Config, PolicyValueNet

path = Path(sys.argv[1])
expected_parent_sha = sys.argv[2]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
if checkpoint.get("format_version") != 3:
    raise SystemExit("final checkpoint is not format-version 3")
if checkpoint.get("v3_stage") != "tactical_expert_warmstart":
    raise SystemExit("final checkpoint is not a V3 supervised warm-start")
if checkpoint.get("parent_checkpoint_sha256") != expected_parent_sha:
    raise SystemExit("final checkpoint parent SHA256 mismatch")
state = checkpoint.get("train_model")
if not isinstance(state, dict):
    raise SystemExit("final checkpoint has no V3 supervised train_model candidate state")
config = Config(**checkpoint["config"])
model = PolicyValueNet(config.board_size, config.channels, config.residual_blocks)
model.load_state_dict(state, strict=True)
model.eval()
print(checkpoint.get("global_step", -1))
PY
)"
if [[ "$final_step" != "$STEPS" ]]; then
  echo "Final checkpoint global_step=$final_step, expected $STEPS" >&2
  exit 1
fi
sha256sum "$OUT_DIR/latest.pt" > "$OUT_DIR/latest.pt.sha256"
sha256sum -c "$OUT_DIR/latest.pt.sha256"
echo "[$(date -Is)] V3F supervised stage complete at global_step=$final_step"
