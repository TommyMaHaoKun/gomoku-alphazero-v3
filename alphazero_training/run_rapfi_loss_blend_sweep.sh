#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN="rapfi_distillation/loss_correction_round1"
OUT="$RUN/blend_sweep"
ANCHOR="rapfi_distillation/run_20260730_dagger128/training/candidate_balanced_s400_eval.pt"
ANCHOR_SHA256="8df9df14c22cb596fcd7f31aa8b0e111e62ae1803ce329dd96b96e0f6026a35c"
UPDATE="$RUN/training/candidate_s600_eval.pt"
UPDATE_SHA256="aaeee639231cd2f4fd545df6b316d974b5d565453f31d0c91391d069dd247d5f"
PARENT_SHA256="ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e"

cd "$ROOT"
mkdir -p "$OUT"

for entry in 025:0.25 040:0.40 050:0.50 060:0.60; do
  tag="${entry%%:*}"
  alpha="${entry#*:}"
  source="$OUT/candidate_blend_a${tag}.pt"
  frozen="$OUT/candidate_blend_a${tag}_eval.pt"
  if [[ ! -f "$source" ]]; then
    "$PYTHON_BIN" -m alphazero_training.blend_v3_checkpoints \
      --anchor "$ANCHOR" --anchor-sha256 "$ANCHOR_SHA256" \
      --update "$UPDATE" --update-sha256 "$UPDATE_SHA256" \
      --alpha "$alpha" --parent-sha256 "$PARENT_SHA256" \
      --output "$source"
  fi
  source_hash="$(sha256sum "$source" | awk '{print $1}')"
  if [[ ! -f "$frozen" ]]; then
    "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
      --source "$source" --expected-source-sha256 "$source_hash" \
      --expected-parent-sha256 "$PARENT_SHA256" --output "$frozen"
  fi
  "$PYTHON_BIN" -m alphazero_training.v3_legal_tactics evaluate \
    --checkpoint "$frozen" \
    --dataset alphazero_training/v3_legal_tactics_eval.npz \
    --model-key best_model --split eval --simulations 64 --device cuda \
    --json-out "$OUT/a${tag}_tactics.json" >/dev/null
  "$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
    --checkpoint "$frozen" \
    --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
    --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
    --output "$OUT/a${tag}_white.json" \
    --model-key best_model --device cuda >/dev/null
done

"$PYTHON_BIN" - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
summary = {}
for tag, alpha in (("025", 0.25), ("040", 0.40), ("050", 0.50), ("060", 0.60)):
    tactics = json.loads((out / f"a{tag}_tactics.json").read_text())
    white = json.loads((out / f"a{tag}_white.json").read_text())
    summary[tag] = {
        "alpha_loss_candidate": alpha,
        "checkpoint_sha256": tactics["checkpoint_sha256"],
        "raw_tactics_count": round(
            tactics["raw_network"]["top1"] * tactics["samples"]
        ),
        "deployed_tactics_count": round(
            tactics["v3_search_with_exact_oracle"]["accuracy"]
            * tactics["samples"]
        ),
        "tactics_total": tactics["samples"],
        "white_safe_count": white["metrics"]["top1_in_safe_set_count"],
        "white_safe_total": white["metrics"]["records"],
        "white_safe_probability_mass": white["metrics"]["safe_probability_mass"],
    }
(out / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
