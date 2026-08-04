#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN="rapfi_distillation/round6_regret_restart"
TRAIN="$RUN/training"
GATE="$RUN/guard_gate"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"
SOURCE="$TRAIN/r6_guard_s10000.pt"
FROZEN="$TRAIN/r6_guard_s10000_eval.pt"
MOVE14="rapfi_distillation/desktop_loss_corrections/desktop_20260731T024514_932773Z_c48f1c86_move14_rapfi.npz"

cd "$ROOT"
mkdir -p "$GATE"
source_sha="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [[ ! -s "$FROZEN" ]]; then
  "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
    --source "$SOURCE" --expected-source-sha256 "$source_sha" \
    --expected-parent-sha256 "$PARENT_SHA256" --output "$FROZEN"
fi

: > "$RUN/guard_candidate_manifest.tsv"
printf 'r6_guard\t%s\n' "$FROZEN" >> "$RUN/guard_candidate_manifest.tsv"
update_sha="$(sha256sum "$FROZEN" | awk '{print $1}')"
for alpha_spec in 010:0.10 020:0.20 030:0.30; do
  IFS=: read -r tag alpha <<<"$alpha_spec"
  name="r6_guard_a${tag}"
  blended="$TRAIN/${name}.pt"
  eval_checkpoint="$TRAIN/${name}_eval.pt"
  if [[ ! -s "$blended" ]]; then
    "$PYTHON_BIN" -m alphazero_training.blend_v3_checkpoints \
      --anchor "$PARENT" --anchor-sha256 "$PARENT_SHA256" \
      --update "$FROZEN" --update-sha256 "$update_sha" \
      --alpha "$alpha" --parent-sha256 "$PARENT_SHA256" \
      --output "$blended"
  fi
  if [[ ! -s "$eval_checkpoint" ]]; then
    blended_sha="$(sha256sum "$blended" | awk '{print $1}')"
    "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
      --source "$blended" --expected-source-sha256 "$blended_sha" \
      --expected-parent-sha256 "$PARENT_SHA256" --output "$eval_checkpoint"
  fi
  printf '%s\t%s\n' "$name" "$eval_checkpoint" >> "$RUN/guard_candidate_manifest.tsv"
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
done < "$RUN/guard_candidate_manifest.tsv"

"$PYTHON_BIN" - "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path

run = Path(sys.argv[1]); gate = run / "guard_gate"
rows = {}
for line in (run / "guard_candidate_manifest.tsv").read_text().splitlines():
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
output = run / "guard_static_gate_summary.json"
output.write_text(json.dumps(rows, indent=2) + "\n")
(output.with_suffix(output.suffix + ".sha256")).write_text(
    hashlib.sha256(output.read_bytes()).hexdigest() + "  " + str(output) + "\n"
)
print(json.dumps(rows, indent=2))
PY
