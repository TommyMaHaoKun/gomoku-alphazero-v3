#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN="rapfi_distillation/round6_regret_restart"
TRAIN="$RUN/training"
GATE="$RUN/value_ensemble_gate"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"
MOVE14="rapfi_distillation/desktop_loss_corrections/desktop_20260731T024514_932773Z_c48f1c86_move14_rapfi.npz"

cd "$ROOT"
mkdir -p "$GATE"

freeze() {
  local source="$1" output="$2" source_sha
  if [[ ! -s "$output" ]]; then
    source_sha="$(sha256sum "$source" | awk '{print $1}')"
    "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
      --source "$source" --expected-source-sha256 "$source_sha" \
      --expected-parent-sha256 "$PARENT_SHA256" --output "$output"
  fi
}

SEED1="$TRAIN/r6_value_repair_s8000_eval.pt"
freeze "$TRAIN/r6_value_repair_seed2_s8000.pt" "$TRAIN/r6_value_repair_seed2_s8000_eval.pt"
freeze "$TRAIN/r6_value_repair_seed3_s8000.pt" "$TRAIN/r6_value_repair_seed3_s8000_eval.pt"
SEED2="$TRAIN/r6_value_repair_seed2_s8000_eval.pt"
SEED3="$TRAIN/r6_value_repair_seed3_s8000_eval.pt"

if [[ ! -s "$TRAIN/r6_value_ensemble12.pt" ]]; then
  "$PYTHON_BIN" -m alphazero_training.blend_v3_checkpoints \
    --anchor "$SEED1" --anchor-sha256 "$(sha256sum "$SEED1" | awk '{print $1}')" \
    --update "$SEED2" --update-sha256 "$(sha256sum "$SEED2" | awk '{print $1}')" \
    --alpha 0.5 --parent-sha256 "$PARENT_SHA256" \
    --output "$TRAIN/r6_value_ensemble12.pt"
fi
if [[ ! -s "$TRAIN/r6_value_ensemble_full.pt" ]]; then
  "$PYTHON_BIN" -m alphazero_training.blend_v3_checkpoints \
    --anchor "$TRAIN/r6_value_ensemble12.pt" \
    --anchor-sha256 "$(sha256sum "$TRAIN/r6_value_ensemble12.pt" | awk '{print $1}')" \
    --update "$SEED3" --update-sha256 "$(sha256sum "$SEED3" | awk '{print $1}')" \
    --alpha 0.3333333333333333 --parent-sha256 "$PARENT_SHA256" \
    --output "$TRAIN/r6_value_ensemble_full.pt"
fi
freeze "$TRAIN/r6_value_ensemble_full.pt" "$TRAIN/r6_value_ensemble_full_eval.pt"
ENSEMBLE="$TRAIN/r6_value_ensemble_full_eval.pt"
ENSEMBLE_SHA="$(sha256sum "$ENSEMBLE" | awk '{print $1}')"

: > "$RUN/value_ensemble_candidates.tsv"
for alpha_spec in 015:0.15 025:0.25 035:0.35; do
  IFS=: read -r tag alpha <<<"$alpha_spec"
  name="r6_value_ensemble_a${tag}"
  source="$TRAIN/${name}.pt"
  frozen="$TRAIN/${name}_eval.pt"
  if [[ ! -s "$source" ]]; then
    "$PYTHON_BIN" -m alphazero_training.blend_v3_checkpoints \
      --anchor "$PARENT" --anchor-sha256 "$PARENT_SHA256" \
      --update "$ENSEMBLE" --update-sha256 "$ENSEMBLE_SHA" \
      --alpha "$alpha" --parent-sha256 "$PARENT_SHA256" --output "$source"
  fi
  freeze "$source" "$frozen"
  printf '%s\t%s\n' "$name" "$frozen" >> "$RUN/value_ensemble_candidates.tsv"
done

"$PYTHON_BIN" - "$PARENT" "$RUN/value_ensemble_candidates.tsv" "$GATE/parameter_invariance.json" <<'PY'
import json, sys
from pathlib import Path
import torch

parent = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)["best_model"]
rows = {}
for line in Path(sys.argv[2]).read_text().splitlines():
    name, path = line.split("\t")
    state = torch.load(Path(path), map_location="cpu", weights_only=False)["best_model"]
    changed = [key for key in parent if not torch.equal(parent[key], state[key])]
    rows[name] = {
        "changed_parameters": changed,
        "non_value_changes": [key for key in changed if not key.startswith("value_")],
    }
if any(row["non_value_changes"] or not row["changed_parameters"] for row in rows.values()):
    raise SystemExit("value ensemble parameter invariance failed")
Path(sys.argv[3]).write_text(json.dumps(rows, indent=2) + "\n")
PY

while IFS=$'\t' read -r name checkpoint; do
  "$PYTHON_BIN" -m alphazero_training.evaluate_desktop_correction \
    --checkpoint "$checkpoint" --dataset "$MOVE14" --simulations 256 \
    --output "$GATE/${name}_move14.json" >/dev/null || true
  "$PYTHON_BIN" -m alphazero_training.v3_legal_tactics evaluate \
    --checkpoint "$checkpoint" --dataset alphazero_training/v3_legal_tactics_eval.npz \
    --model-key best_model --split eval --simulations 64 --device cuda \
    --json-out "$GATE/${name}_tactics.json" >/dev/null
  "$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
    --checkpoint "$checkpoint" \
    --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
    --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
    --output "$GATE/${name}_white.json" --model-key best_model --device cuda >/dev/null
done < "$RUN/value_ensemble_candidates.tsv"

"$PYTHON_BIN" - "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path

run = Path(sys.argv[1]); gate = run / "value_ensemble_gate"; rows = {}
for line in (run / "value_ensemble_candidates.tsv").read_text().splitlines():
    name, checkpoint = line.split("\t")
    tactics = json.loads((gate / f"{name}_tactics.json").read_text())
    white = json.loads((gate / f"{name}_white.json").read_text())
    move14 = json.loads((gate / f"{name}_move14.json").read_text())
    row = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "raw_tactics": round(tactics["raw_network"]["top1"] * tactics["samples"]),
        "deployed_tactics": round(tactics["v3_search_with_exact_oracle"]["accuracy"] * tactics["samples"]),
        "white_safe_count": white["metrics"]["top1_in_safe_set_count"],
        "white_safe_probability_mass": white["metrics"]["safe_probability_mass"],
        "move14_passed": move14["passed"],
    }
    row["hard_gate_passed"] = (
        row["raw_tactics"] >= 47 and row["deployed_tactics"] >= 48
        and row["white_safe_count"] >= 16
        and row["white_safe_probability_mass"] >= 0.7707617002141821
    )
    rows[name] = row
output = run / "value_ensemble_static_gate_summary.json"
output.write_text(json.dumps(rows, indent=2) + "\n")
(output.with_suffix(output.suffix + ".sha256")).write_text(
    hashlib.sha256(output.read_bytes()).hexdigest() + "  " + str(output) + "\n"
)
print(json.dumps(rows, indent=2))
PY
