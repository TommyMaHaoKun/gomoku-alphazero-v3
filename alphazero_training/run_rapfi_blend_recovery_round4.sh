#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"
ANCHOR="rapfi_distillation/loss_correction_round1/blend_sweep/candidate_blend_a060_eval.pt"
ANCHOR_SHA256="378f5715541c95972cde396f4023b63b3f1aa83de6a04aa78af934e5253adada"
CONS="rapfi_distillation/longrun_counterfactual_train_round3/training/candidate_cf_cons_s12000_eval.pt"
CONS_SHA256="2ca26148e30c0898d48b9ff118bd8be63fbac8e7cf4d7fab5620d4f0b9552266"
AGGR="rapfi_distillation/longrun_counterfactual_train_round3/training/candidate_cf_aggr_s12000_eval.pt"
AGGR_SHA256="9bdc4e4d2b3ab32064b1746a7690bac75743bcae28dc040f75e8f33aeaa6c005"
JOINT="rapfi_distillation/longrun_joint_round2/joint_policy_value.npz"
BRANCH="rapfi_distillation/longrun_counterfactual_round3/branches.npz"
LOSS_DATA="rapfi_distillation/loss_correction_round1/loss_hard_negative.npz"
RUN="rapfi_distillation/longrun_blend_recovery_round4"

cd "$ROOT"
mkdir -p "$RUN/training" "$RUN/gate" "$RUN/loss_eval" "$RUN/joint_eval" \
  "$RUN/branch_eval"

verify_file() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

verify_file "$PARENT" "$PARENT_SHA256"
verify_file "$ANCHOR" "$ANCHOR_SHA256"
verify_file "$CONS" "$CONS_SHA256"
verify_file "$AGGR" "$AGGR_SHA256"

evaluate_checkpoint() {
  local name="$1"
  local checkpoint="$2"
  if [[ ! -s "$RUN/gate/${name}_tactics.json" ]]; then
    "$PYTHON_BIN" -m alphazero_training.v3_legal_tactics evaluate \
      --checkpoint "$checkpoint" \
      --dataset alphazero_training/v3_legal_tactics_eval.npz \
      --model-key best_model --split eval --simulations 64 --device cuda \
      --json-out "$RUN/gate/${name}_tactics.json" >/dev/null
  fi
  if [[ ! -s "$RUN/gate/${name}_white.json" ]]; then
    "$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
      --checkpoint "$checkpoint" \
      --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
      --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
      --output "$RUN/gate/${name}_white.json" \
      --model-key best_model --device cuda >/dev/null
  fi
  for dataset in loss joint branch; do
    local path seed output
    case "$dataset" in
      loss) path="$LOSS_DATA"; seed=20261201 ;;
      joint) path="$JOINT"; seed=20270111 ;;
      branch) path="$BRANCH"; seed=20270601 ;;
    esac
    output="$RUN/${dataset}_eval/${name}.json"
    if [[ ! -s "$output" ]]; then
      "$PYTHON_BIN" -m alphazero_training.evaluate_supervised_dataset \
        --checkpoint "$checkpoint" --model-key best_model --dataset "$path" \
        --seed "$seed" --validation-fraction 0.1 --device cuda \
        --output "$output" >/dev/null
    fi
  done
}

evaluate_checkpoint parent "$PARENT"
evaluate_checkpoint anchor "$ANCHOR"

families=(
  "cons:$CONS:$CONS_SHA256"
  "aggr:$AGGR:$AGGR_SHA256"
)
alphas=(
  "015:0.15" "025:0.25" "035:0.35" "045:0.45"
  "055:0.55" "065:0.65" "075:0.75" "085:0.85"
)

for family_spec in "${families[@]}"; do
  IFS=: read -r family update update_sha <<<"$family_spec"
  for alpha_spec in "${alphas[@]}"; do
    IFS=: read -r alpha_tag alpha <<<"$alpha_spec"
    name="${family}_a${alpha_tag}"
    source="$RUN/training/${name}.pt"
    frozen="$RUN/training/${name}_eval.pt"
    if [[ ! -f "$source" ]]; then
      "$PYTHON_BIN" -m alphazero_training.blend_v3_checkpoints \
        --anchor "$ANCHOR" --anchor-sha256 "$ANCHOR_SHA256" \
        --update "$update" --update-sha256 "$update_sha" \
        --alpha "$alpha" --parent-sha256 "$PARENT_SHA256" \
        --output "$source"
    fi
    source_sha="$(sha256sum "$source" | awk '{print $1}')"
    if [[ ! -f "$frozen" ]]; then
      "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
        --source "$source" --expected-source-sha256 "$source_sha" \
        --expected-parent-sha256 "$PARENT_SHA256" --output "$frozen"
    fi
    evaluate_checkpoint "$name" "$frozen"
  done
done

"$PYTHON_BIN" - "$RUN" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
names = ["parent", "anchor"] + sorted(
    path.name.removesuffix("_tactics.json")
    for path in (run / "gate").glob("*_a*_tactics.json")
)
summary = {}
for name in names:
    tactics = json.loads((run / "gate" / f"{name}_tactics.json").read_text())
    white = json.loads((run / "gate" / f"{name}_white.json").read_text())
    datasets = {
        kind: json.loads((run / f"{kind}_eval" / f"{name}.json").read_text())["datasets"][0]
        for kind in ("loss", "joint", "branch")
    }
    row = {
        "checkpoint_sha256": tactics["checkpoint_sha256"],
        "raw_tactics": round(tactics["raw_network"]["top1"] * tactics["samples"]),
        "deployed_tactics": round(
            tactics["v3_search_with_exact_oracle"]["accuracy"] * tactics["samples"]
        ),
        "tactics_total": tactics["samples"],
        "white_safe_count": white["metrics"]["top1_in_safe_set_count"],
        "white_safe_total": white["metrics"]["records"],
        "white_safe_probability_mass": white["metrics"]["safe_probability_mass"],
    }
    for kind, metrics in datasets.items():
        row[f"{kind}_policy_top1"] = metrics["policy_top1"]
        row[f"{kind}_value_mse"] = metrics["value_mse"]
        row[f"{kind}_teacher_over_mistake"] = metrics.get("teacher_over_mistake_rate")
    summary[name] = row

parent = summary["parent"]
passed = []
for name, row in summary.items():
    row["hard_gate_passed"] = bool(
        name != "parent"
        and row["raw_tactics"] >= parent["raw_tactics"]
        and row["deployed_tactics"] >= parent["deployed_tactics"]
        and row["white_safe_count"] >= parent["white_safe_count"]
        and row["white_safe_probability_mass"] >= parent["white_safe_probability_mass"]
    )
    if row["hard_gate_passed"] and "_a" in name:
        passed.append(name)

def rank_key(name):
    row = summary[name]
    return (
        row["raw_tactics"],
        row["loss_policy_top1"],
        row["branch_policy_top1"],
        -(row["branch_value_mse"] or 999.0),
        row["joint_policy_top1"],
    )

ranked = sorted(passed, key=rank_key, reverse=True)
selected = []
for family in ("cons_", "aggr_"):
    candidate = next((name for name in ranked if name.startswith(family)), None)
    if candidate is not None:
        selected.append(candidate)
for name in ranked:
    if name not in selected:
        selected.append(name)
    if len(selected) >= 6:
        break

(run / "standard_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
(run / "passed_candidates.txt").write_text(
    "\n".join(ranked) + ("\n" if ranked else "")
)
(run / "screen_candidates.txt").write_text(
    "\n".join(selected) + ("\n" if selected else "")
)
print(json.dumps({"passed": ranked, "selected_for_screen": selected}, indent=2))
if not selected:
    raise SystemExit("no Round4 blend passed the hard gate")
PY

verify_file "$PARENT" "$PARENT_SHA256"
SCREEN="$RUN/screen_seed20270801"
mkdir -p "$SCREEN/parent"
if [[ ! -s "$SCREEN/parent/games.json" ]]; then
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
    --checkpoint "$PARENT" --engine "$ENGINE" \
    --report "$SCREEN/parent/games.json" --dataset "$SCREEN/parent/policy_unused.npz" \
    --ai-loss-dir "$SCREEN/parent/pending_training/ai_losses" \
    --pairs 256 --opening-plies 4 --simulations 64 --workers 8 \
    --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed 20270801
fi

while read -r name; do
  [[ -n "$name" ]] || continue
  mkdir -p "$SCREEN/$name"
  if [[ ! -s "$SCREEN/$name/games.json" ]]; then
    "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
      --checkpoint "$RUN/training/${name}_eval.pt" --engine "$ENGINE" \
      --report "$SCREEN/$name/games.json" --dataset "$SCREEN/$name/policy_unused.npz" \
      --ai-loss-dir "$SCREEN/$name/pending_training/ai_losses" \
      --pairs 256 --opening-plies 4 --simulations 64 --workers 8 \
      --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed 20270801
  fi
  "$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
    --parent-report "$SCREEN/parent/games.json" \
    --candidate-report "$SCREEN/$name/games.json" \
    --output "$SCREEN/${name}_comparison.json"
done < "$RUN/screen_candidates.txt"

"$PYTHON_BIN" - "$SCREEN" "$RUN/screen_candidates.txt" <<'PY'
import json
from pathlib import Path
import sys

screen = Path(sys.argv[1])
names = [name for name in Path(sys.argv[2]).read_text().split() if name]
rows = {
    name: json.loads((screen / f"{name}_comparison.json").read_text())
    for name in names
}
(screen / "screen_summary.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
)
positive = [name for name in names if rows[name]["score_delta"] > 0]
if not positive:
    raise SystemExit("no Round4 blend beat parent in the 256-pair screen")
best = max(
    positive,
    key=lambda name: (rows[name]["score_delta"], -rows[name]["two_sided_exact_sign_p"]),
)
(screen / "best_candidate.txt").write_text(best + "\n")
print(json.dumps({"best_candidate": best, **rows[best]}, ensure_ascii=False, indent=2))
PY

best="$(tr -d '\r\n' < "$SCREEN/best_candidate.txt")"
EXPANDED="$RUN/expanded_seed20270901"
mkdir -p "$EXPANDED/parent" "$EXPANDED/$best"
if [[ ! -s "$EXPANDED/parent/games.json" ]]; then
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
    --checkpoint "$PARENT" --engine "$ENGINE" \
    --report "$EXPANDED/parent/games.json" --dataset "$EXPANDED/parent/policy_unused.npz" \
    --ai-loss-dir "$EXPANDED/parent/pending_training/ai_losses" \
    --pairs 1024 --opening-plies 4 --simulations 64 --workers 8 \
    --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed 20270901
fi
if [[ ! -s "$EXPANDED/$best/games.json" ]]; then
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
    --checkpoint "$RUN/training/${best}_eval.pt" --engine "$ENGINE" \
    --report "$EXPANDED/$best/games.json" --dataset "$EXPANDED/$best/policy_unused.npz" \
    --ai-loss-dir "$EXPANDED/$best/pending_training/ai_losses" \
    --pairs 1024 --opening-plies 4 --simulations 64 --workers 8 \
    --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed 20270901
fi
"$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
  --parent-report "$EXPANDED/parent/games.json" \
  --candidate-report "$EXPANDED/$best/games.json" \
  --output "$EXPANDED/comparison.json"
sha256sum "$RUN/training/${best}_eval.pt" "$EXPANDED/comparison.json" > "$RUN/FINAL_SHA256SUMS"
echo "[$(date -Is)] Round4 blend recovery complete; finalist=$best"
