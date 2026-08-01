#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN="rapfi_distillation/round6_regret_restart"
TRAIN="$RUN/training"
GATE="$RUN/value_gate"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"
SOURCE="$TRAIN/r6_value_repair_s8000.pt"
FROZEN="$TRAIN/r6_value_repair_s8000_eval.pt"
MOVE14="rapfi_distillation/desktop_loss_corrections/desktop_20260731T024514_932773Z_c48f1c86_move14_rapfi.npz"

cd "$ROOT"
mkdir -p "$GATE"
source_sha="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [[ ! -s "$FROZEN" ]]; then
  "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
    --source "$SOURCE" --expected-source-sha256 "$source_sha" \
    --expected-parent-sha256 "$PARENT_SHA256" --output "$FROZEN"
fi

"$PYTHON_BIN" - "$PARENT" "$FROZEN" "$GATE/parameter_invariance.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch

parent_path, candidate_path, output_path = map(Path, sys.argv[1:])
parent = torch.load(parent_path, map_location="cpu", weights_only=False)["best_model"]
candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)["best_model"]
changed = []
for name in parent:
    same = torch.equal(parent[name], candidate[name])
    if not same:
        changed.append(name)
non_value_changes = [name for name in changed if not name.startswith("value_")]
result = {
    "parent_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
    "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    "changed_parameters": changed,
    "non_value_changes": non_value_changes,
    "passed": not non_value_changes and bool(changed),
}
output_path.write_text(json.dumps(result, indent=2) + "\n")
if not result["passed"]:
    raise SystemExit("value-only parameter invariance failed")
PY

: > "$RUN/value_candidate_manifest.tsv"
printf 'r6_value_full\t%s\n' "$FROZEN" >> "$RUN/value_candidate_manifest.tsv"
update_sha="$(sha256sum "$FROZEN" | awk '{print $1}')"
for alpha_spec in 025:0.25 050:0.50 075:0.75; do
  IFS=: read -r tag alpha <<<"$alpha_spec"
  name="r6_value_a${tag}"
  blended="$TRAIN/${name}.pt"
  eval_checkpoint="$TRAIN/${name}_eval.pt"
  if [[ ! -s "$blended" ]]; then
    "$PYTHON_BIN" -m alphazero_training.blend_v3_checkpoints \
      --anchor "$PARENT" --anchor-sha256 "$PARENT_SHA256" \
      --update "$FROZEN" --update-sha256 "$update_sha" \
      --alpha "$alpha" --parent-sha256 "$PARENT_SHA256" --output "$blended"
  fi
  if [[ ! -s "$eval_checkpoint" ]]; then
    blended_sha="$(sha256sum "$blended" | awk '{print $1}')"
    "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
      --source "$blended" --expected-source-sha256 "$blended_sha" \
      --expected-parent-sha256 "$PARENT_SHA256" --output "$eval_checkpoint"
  fi
  printf '%s\t%s\n' "$name" "$eval_checkpoint" >> "$RUN/value_candidate_manifest.tsv"
done

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
done < "$RUN/value_candidate_manifest.tsv"

"$PYTHON_BIN" - "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path

run = Path(sys.argv[1]); gate = run / "value_gate"; rows = {}
for line in (run / "value_candidate_manifest.tsv").read_text().splitlines():
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
output = run / "value_static_gate_summary.json"
output.write_text(json.dumps(rows, indent=2) + "\n")
(output.with_suffix(output.suffix + ".sha256")).write_text(
    hashlib.sha256(output.read_bytes()).hexdigest() + "  " + str(output) + "\n"
)
print(json.dumps(rows, indent=2))
PY
