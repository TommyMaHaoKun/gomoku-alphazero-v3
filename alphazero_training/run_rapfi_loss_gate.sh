#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN="rapfi_distillation/loss_correction_round1"
TRAIN="$RUN/training"
GATE="$RUN/standard_gate"
PARENT_SHA256="ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e"

cd "$ROOT"
mkdir -p "$GATE"

for steps in 200 400 600; do
  source="$TRAIN/candidate_s${steps}.pt"
  source_hash="$(sha256sum "$source" | awk '{print $1}')"
  frozen="$TRAIN/candidate_s${steps}_eval.pt"
  if [[ ! -f "$frozen" ]]; then
    "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
      --source "$source" \
      --expected-source-sha256 "$source_hash" \
      --expected-parent-sha256 "$PARENT_SHA256" \
      --output "$frozen"
  fi
  "$PYTHON_BIN" -m alphazero_training.v3_legal_tactics evaluate \
    --checkpoint "$frozen" \
    --dataset alphazero_training/v3_legal_tactics_eval.npz \
    --model-key best_model --split eval --simulations 64 --device cuda \
    --json-out "$GATE/candidate_s${steps}_tactics.json" >/dev/null
  "$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
    --checkpoint "$frozen" \
    --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
    --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
    --output "$GATE/candidate_s${steps}_white.json" \
    --model-key best_model --device cuda >/dev/null
done

repair_source="$TRAIN/candidate_repair_s400.pt"
if [[ -f "$repair_source" ]]; then
  repair_source_hash="$(sha256sum "$repair_source" | awk '{print $1}')"
  repair_frozen="$TRAIN/candidate_repair_s400_eval.pt"
  if [[ ! -f "$repair_frozen" ]]; then
    "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
      --source "$repair_source" \
      --expected-source-sha256 "$repair_source_hash" \
      --expected-parent-sha256 "$PARENT_SHA256" \
      --output "$repair_frozen"
  fi
  "$PYTHON_BIN" -m alphazero_training.v3_legal_tactics evaluate \
    --checkpoint "$repair_frozen" \
    --dataset alphazero_training/v3_legal_tactics_eval.npz \
    --model-key best_model --split eval --simulations 64 --device cuda \
    --json-out "$GATE/candidate_repair_s400_tactics.json" >/dev/null
  "$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
    --checkpoint "$repair_frozen" \
    --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
    --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
    --output "$GATE/candidate_repair_s400_white.json" \
    --model-key best_model --device cuda >/dev/null
fi

"$PYTHON_BIN" - "$GATE" <<'PY'
import json
from pathlib import Path
import sys

gate = Path(sys.argv[1])
summary = {}
for steps in (200, 400, 600):
    tactics = json.loads((gate / f"candidate_s{steps}_tactics.json").read_text())
    white = json.loads((gate / f"candidate_s{steps}_white.json").read_text())
    summary[str(steps)] = {
        "checkpoint_sha256": tactics["checkpoint_sha256"],
        "raw_tactics_count": round(
            tactics["raw_network"]["top1"] * tactics["samples"]
        ),
        "raw_tactics_total": tactics["samples"],
        "deployed_tactics_count": round(
            tactics["v3_search_with_exact_oracle"]["accuracy"] * tactics["samples"]
        ),
        "white_safe_count": white["metrics"]["top1_in_safe_set_count"],
        "white_safe_total": white["metrics"]["records"],
        "white_safe_probability_mass": white["metrics"]["safe_probability_mass"],
    }
repair_tactics_path = gate / "candidate_repair_s400_tactics.json"
repair_white_path = gate / "candidate_repair_s400_white.json"
if repair_tactics_path.exists() and repair_white_path.exists():
    tactics = json.loads(repair_tactics_path.read_text())
    white = json.loads(repair_white_path.read_text())
    summary["repair_s400"] = {
        "checkpoint_sha256": tactics["checkpoint_sha256"],
        "raw_tactics_count": round(
            tactics["raw_network"]["top1"] * tactics["samples"]
        ),
        "raw_tactics_total": tactics["samples"],
        "deployed_tactics_count": round(
            tactics["v3_search_with_exact_oracle"]["accuracy"]
            * tactics["samples"]
        ),
        "white_safe_count": white["metrics"]["top1_in_safe_set_count"],
        "white_safe_total": white["metrics"]["records"],
        "white_safe_probability_mass": white["metrics"]["safe_probability_mass"],
    }
(gate / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
