#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN="rapfi_distillation/round7_league_selfplay"
GATE="$RUN/static_gate"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"
SOURCE="$RUN/latest.pt"
MOVE14="rapfi_distillation/desktop_loss_corrections/desktop_20260731T024514_932773Z_c48f1c86_move14_rapfi.npz"

cd "$ROOT"
mkdir -p "$GATE" "$RUN/logs"
[[ "$(sha256sum "$PARENT" | awk '{print $1}')" == "$PARENT_SHA256" ]]

freeze() {
  local source="$1" output="$2" source_sha
  source_sha="$(sha256sum "$source" | awk '{print $1}')"
  if [[ ! -s "$output" ]]; then
    "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
      --source "$source" --expected-source-sha256 "$source_sha" \
      --expected-parent-sha256 "$PARENT_SHA256" --output "$output"
  fi
}

FULL="$RUN/round7_full_eval.pt"
freeze "$SOURCE" "$FULL"
FULL_SHA="$(sha256sum "$FULL" | awk '{print $1}')"

: > "$RUN/static_candidates.tsv"
printf 'round7_full\t%s\n' "$FULL" >> "$RUN/static_candidates.tsv"
for alpha_spec in 010:0.10 020:0.20 035:0.35 050:0.50; do
  IFS=: read -r tag alpha <<<"$alpha_spec"
  name="round7_a${tag}"
  blended="$RUN/${name}.pt"
  frozen="$RUN/${name}_eval.pt"
  if [[ ! -s "$blended" ]]; then
    "$PYTHON_BIN" -m alphazero_training.blend_v3_checkpoints \
      --anchor "$PARENT" --anchor-sha256 "$PARENT_SHA256" \
      --update "$FULL" --update-sha256 "$FULL_SHA" \
      --alpha "$alpha" --parent-sha256 "$PARENT_SHA256" --output "$blended"
  fi
  freeze "$blended" "$frozen"
  printf '%s\t%s\n' "$name" "$frozen" >> "$RUN/static_candidates.tsv"
done

while IFS=$'\t' read -r name checkpoint; do
  "$PYTHON_BIN" -m alphazero_training.evaluate_desktop_correction \
    --checkpoint "$checkpoint" --dataset "$MOVE14" --simulations 256 \
    --output "$GATE/${name}_move14.json" >"$RUN/logs/${name}_move14.log" 2>&1 || true
  "$PYTHON_BIN" -m alphazero_training.v3_legal_tactics evaluate \
    --checkpoint "$checkpoint" --dataset alphazero_training/v3_legal_tactics_eval.npz \
    --model-key best_model --split eval --simulations 64 --device cuda \
    --json-out "$GATE/${name}_tactics.json" >"$RUN/logs/${name}_tactics.log" 2>&1
  "$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
    --checkpoint "$checkpoint" \
    --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
    --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
    --output "$GATE/${name}_white.json" --model-key best_model --device cuda \
    >"$RUN/logs/${name}_white.log" 2>&1
done < "$RUN/static_candidates.tsv"

"$PYTHON_BIN" - "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path

run = Path(sys.argv[1])
gate = run / "static_gate"
rows = {}
for line in (run / "static_candidates.tsv").read_text().splitlines():
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
        row["raw_tactics"] >= 47
        and row["deployed_tactics"] >= 48
        and row["white_safe_count"] >= 16
        and row["white_safe_probability_mass"] >= 0.7707617002141821
    )
    rows[name] = row
output = run / "static_gate_summary.json"
output.write_text(json.dumps(rows, indent=2) + "\n")
digest = hashlib.sha256(output.read_bytes()).hexdigest()
(output.with_suffix(output.suffix + ".sha256")).write_text(f"{digest}  {output}\n")
print(json.dumps(rows, indent=2))
PY

touch "$GATE/complete.done"
