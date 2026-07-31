#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"
ROUND2="rapfi_distillation/longrun_joint_round2"
COUNTER="rapfi_distillation/longrun_counterfactual_round3"
RUN="rapfi_distillation/longrun_counterfactual_train_round3"
JOINT="$ROUND2/joint_policy_value.npz"
BRANCH="$COUNTER/branches.npz"
LOSS_DATA="rapfi_distillation/loss_correction_round1/loss_hard_negative.npz"
TACTICS="alphazero_training/v3_legal_tactics_train.npz"
DDQK_TEACHER="alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz"
DDQK_EXPERT="alphazero_training/ddqk_expert_50pairs_policy.npz"

cd "$ROOT"
mkdir -p "$RUN/training" "$RUN/gate" "$RUN/loss_eval" "$RUN/joint_eval" "$RUN/branch_eval"

while pgrep -f '^bash alphazero_training/run_rapfi_counterfactual_round3.sh$' >/dev/null; do
  echo "[$(date -Is)] waiting for counterfactual round3 data"
  sleep 60
done

for path in "$JOINT" "$BRANCH" "$LOSS_DATA"; do
  if [[ ! -f "$path" ]]; then
    echo "required training data missing: $path" >&2
    exit 2
  fi
done

actual_parent="$(sha256sum "$PARENT" | awk '{print $1}')"
if [[ "$actual_parent" != "$PARENT_SHA256" ]]; then
  echo "approved parent changed before Round3: $actual_parent" >&2
  exit 3
fi

white_args=()
for name in p2 p4 p8 seed20260724_parent seed20260730_parent; do
  white_args+=(
    --white-defense-npz "alphazero_training/_v3f_white_train_det/$name/train.npz"
    --white-defense-manifest "alphazero_training/_v3f_white_train_det/$name/manifest.json"
    --white-defense-weight 0.03
  )
done

variants=(
  "cf_cons:0.25:0.15:6e-6:6e-7:0.35:4000:20270611"
  "cf_aggr:0.40:0.15:1.2e-5:1.2e-6:0.25:2500:20270612"
)

for spec in "${variants[@]}"; do
  IFS=: read -r name value_scale value_kd lr min_lr policy_kd freeze seed <<<"$spec"
  output="$RUN/training/candidate_${name}_s12000.pt"
  if [[ ! -f "$output.done" ]]; then
    echo "[$(date -Is)] Round3 training $name"
    "$PYTHON_BIN" -u -m alphazero_training.train_v3_supervised \
      --init-checkpoint "$PARENT" \
      --dataset "$JOINT" \
      --dataset "$BRANCH" \
      --dataset "$LOSS_DATA" \
      --dataset "$TACTICS" \
      --dataset "$DDQK_TEACHER" \
      --dataset "$DDQK_EXPERT" \
      --dataset-weight 0.40 \
      --dataset-weight 0.20 \
      --dataset-weight 0.05 \
      --dataset-weight 0.10 \
      --dataset-weight 0.05 \
      --dataset-weight 0.05 \
      "${white_args[@]}" \
      --output "$output" \
      --steps 12000 \
      --batch-size 1024 \
      --learning-rate "$lr" \
      --min-learning-rate "$min_lr" \
      --warmup-steps 750 \
      --freeze-trunk-steps "$freeze" \
      --train-last-residual-blocks-during-freeze 3 \
      --value-loss-scale "$value_scale" \
      --value-distill-scale "$value_kd" \
      --policy-distill-scale "$policy_kd" \
      --safe-hard-negative-scale 0.30 \
      --safe-hard-negative-margin 1.0 \
      --mistake-hard-negative-scale 0.40 \
      --mistake-hard-negative-margin 1.0 \
      --random-d4-augmentation \
      --validation-fraction 0.1 \
      --eval-every 3000 \
      --seed "$seed" 2>&1 | tee "$RUN/training/train_${name}_s12000.log"
    sha256sum "$output" | tee "$output.sha256"
    touch "$output.done"
  fi
done

evaluate_candidate() {
  local name="$1"
  local source="$2"
  local source_hash frozen
  source_hash="$(sha256sum "$source" | awk '{print $1}')"
  frozen="$RUN/training/candidate_${name}_s12000_eval.pt"
  if [[ ! -f "$frozen" ]]; then
    "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
      --source "$source" --expected-source-sha256 "$source_hash" \
      --expected-parent-sha256 "$PARENT_SHA256" --output "$frozen"
  fi
  "$PYTHON_BIN" -m alphazero_training.v3_legal_tactics evaluate \
    --checkpoint "$frozen" --dataset alphazero_training/v3_legal_tactics_eval.npz \
    --model-key best_model --split eval --simulations 64 --device cuda \
    --json-out "$RUN/gate/${name}_tactics.json" >/dev/null
  "$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
    --checkpoint "$frozen" \
    --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
    --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
    --output "$RUN/gate/${name}_white.json" --model-key best_model --device cuda >/dev/null
  for dataset in loss joint branch; do
    case "$dataset" in
      loss) path="$LOSS_DATA"; eval_seed=20261201 ;;
      joint) path="$JOINT"; eval_seed=20270111 ;;
      branch) path="$BRANCH"; eval_seed=20270601 ;;
    esac
    "$PYTHON_BIN" -m alphazero_training.evaluate_supervised_dataset \
      --checkpoint "$frozen" --model-key best_model --dataset "$path" \
      --seed "$eval_seed" --validation-fraction 0.1 --device cuda \
      --output "$RUN/${dataset}_eval/${name}.json" >/dev/null
  done
}

"$PYTHON_BIN" -m alphazero_training.v3_legal_tactics evaluate \
  --checkpoint "$PARENT" --dataset alphazero_training/v3_legal_tactics_eval.npz \
  --model-key best_model --split eval --simulations 64 --device cuda \
  --json-out "$RUN/gate/parent_tactics.json" >/dev/null
"$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
  --checkpoint "$PARENT" \
  --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
  --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
  --output "$RUN/gate/parent_white.json" --model-key best_model --device cuda >/dev/null
for dataset in loss joint branch; do
  case "$dataset" in
    loss) path="$LOSS_DATA"; eval_seed=20261201 ;;
    joint) path="$JOINT"; eval_seed=20270111 ;;
    branch) path="$BRANCH"; eval_seed=20270601 ;;
  esac
  "$PYTHON_BIN" -m alphazero_training.evaluate_supervised_dataset \
    --checkpoint "$PARENT" --model-key best_model --dataset "$path" \
    --seed "$eval_seed" --validation-fraction 0.1 --device cuda \
    --output "$RUN/${dataset}_eval/parent.json" >/dev/null
done

for spec in "${variants[@]}"; do
  IFS=: read -r name _ _ _ _ _ _ _ <<<"$spec"
  evaluate_candidate "$name" "$RUN/training/candidate_${name}_s12000.pt"
done

"$PYTHON_BIN" - "$RUN" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
names = ("cf_cons", "cf_aggr")
summary = {}
for name in ("parent", *names):
    tactics = json.loads((run / "gate" / f"{name}_tactics.json").read_text())
    white = json.loads((run / "gate" / f"{name}_white.json").read_text())
    datasets = {
        kind: json.loads((run / f"{kind}_eval" / f"{name}.json").read_text())["datasets"][0]
        for kind in ("loss", "joint", "branch")
    }
    summary[name] = {
        "checkpoint_sha256": tactics["checkpoint_sha256"],
        "raw_tactics": round(tactics["raw_network"]["top1"] * tactics["samples"]),
        "deployed_tactics": round(tactics["v3_search_with_exact_oracle"]["accuracy"] * tactics["samples"]),
        "tactics_total": tactics["samples"],
        "white_safe_count": white["metrics"]["top1_in_safe_set_count"],
        "white_safe_total": white["metrics"]["records"],
        "white_safe_probability_mass": white["metrics"]["safe_probability_mass"],
        **{
            f"{kind}_policy_top1": metrics["policy_top1"]
            for kind, metrics in datasets.items()
        },
        **{
            f"{kind}_value_mse": metrics["value_mse"]
            for kind, metrics in datasets.items()
        },
    }
parent = summary["parent"]
passed = []
for name in names:
    row = summary[name]
    row["hard_gate_passed"] = bool(
        row["raw_tactics"] >= parent["raw_tactics"]
        and row["deployed_tactics"] >= parent["deployed_tactics"]
        and row["white_safe_count"] >= parent["white_safe_count"]
        and row["white_safe_probability_mass"] >= parent["white_safe_probability_mass"]
    )
    if row["hard_gate_passed"]:
        passed.append(name)
(run / "standard_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
(run / "passed_candidates.txt").write_text("\n".join(passed) + ("\n" if passed else ""))
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not passed:
    raise SystemExit("no Round3 candidate passed the hard gate")
PY

SCREEN="$RUN/screen_seed20270601"
mkdir -p "$SCREEN/parent"
if [[ ! -f "$SCREEN/parent/games.json" ]]; then
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
    --checkpoint "$PARENT" --engine "$ENGINE" \
    --report "$SCREEN/parent/games.json" --dataset "$SCREEN/parent/policy_unused.npz" \
    --ai-loss-dir "$SCREEN/parent/pending_training/ai_losses" \
    --pairs 256 --opening-plies 4 --simulations 64 --workers 8 \
    --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed 20270601
fi
while read -r name; do
  [[ -n "$name" ]] || continue
  mkdir -p "$SCREEN/$name"
  if [[ ! -f "$SCREEN/$name/games.json" ]]; then
    "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
      --checkpoint "$RUN/training/candidate_${name}_s12000_eval.pt" --engine "$ENGINE" \
      --report "$SCREEN/$name/games.json" --dataset "$SCREEN/$name/policy_unused.npz" \
      --ai-loss-dir "$SCREEN/$name/pending_training/ai_losses" \
      --pairs 256 --opening-plies 4 --simulations 64 --workers 8 \
      --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed 20270601
  fi
  "$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
    --parent-report "$SCREEN/parent/games.json" --candidate-report "$SCREEN/$name/games.json" \
    --output "$SCREEN/${name}_comparison.json"
done < "$RUN/passed_candidates.txt"

"$PYTHON_BIN" - "$SCREEN" "$RUN/passed_candidates.txt" <<'PY'
import json
from pathlib import Path
import sys

screen = Path(sys.argv[1])
names = [name for name in Path(sys.argv[2]).read_text().split() if name]
rows = {name: json.loads((screen / f"{name}_comparison.json").read_text()) for name in names}
best = max(rows, key=lambda name: (rows[name]["score_delta"], -rows[name]["two_sided_exact_sign_p"]))
(screen / "screen_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
if rows[best]["score_delta"] <= 0:
    raise SystemExit("no Round3 candidate beat parent in screen")
(screen / "best_candidate.txt").write_text(best + "\n")
print(json.dumps({"best_candidate": best, **rows[best]}, ensure_ascii=False, indent=2))
PY

best="$(tr -d '\r\n' < "$SCREEN/best_candidate.txt")"
EXPANDED="$RUN/expanded_seed20270701"
mkdir -p "$EXPANDED/parent" "$EXPANDED/$best"
if [[ ! -f "$EXPANDED/parent/games.json" ]]; then
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
    --checkpoint "$PARENT" --engine "$ENGINE" \
    --report "$EXPANDED/parent/games.json" --dataset "$EXPANDED/parent/policy_unused.npz" \
    --ai-loss-dir "$EXPANDED/parent/pending_training/ai_losses" \
    --pairs 1024 --opening-plies 4 --simulations 64 --workers 8 \
    --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed 20270701
fi
if [[ ! -f "$EXPANDED/$best/games.json" ]]; then
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
    --checkpoint "$RUN/training/candidate_${best}_s12000_eval.pt" --engine "$ENGINE" \
    --report "$EXPANDED/$best/games.json" --dataset "$EXPANDED/$best/policy_unused.npz" \
    --ai-loss-dir "$EXPANDED/$best/pending_training/ai_losses" \
    --pairs 1024 --opening-plies 4 --simulations 64 --workers 8 \
    --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed 20270701
fi
"$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
  --parent-report "$EXPANDED/parent/games.json" \
  --candidate-report "$EXPANDED/$best/games.json" \
  --output "$EXPANDED/comparison.json"
echo "[$(date -Is)] Round3 counterfactual training complete; finalist=$best"
