#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"
ROUND1="rapfi_distillation/run_20260730_dagger128/games.json"
ROUND2="rapfi_distillation/longrun_round2_dagger2048/games.json"
ROUND2_LOG="rapfi_distillation/longrun_round2_dagger2048.launch.log"
RUN="rapfi_distillation/longrun_joint_round2"
JOINT="$RUN/joint_policy_value.npz"
LOSS_DATA="rapfi_distillation/loss_correction_round1/loss_hard_negative.npz"
TACTICS="alphazero_training/v3_legal_tactics_train.npz"
DDQK_TEACHER="alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz"
DDQK_EXPERT="alphazero_training/ddqk_expert_50pairs_policy.npz"

cd "$ROOT"
mkdir -p "$RUN/training" "$RUN/gate" "$RUN/loss_eval" "$RUN/joint_eval"

actual_parent="$(sha256sum "$PARENT" | awk '{print $1}')"
if [[ "$actual_parent" != "$PARENT_SHA256" ]]; then
  echo "approved parent SHA256 mismatch: $actual_parent" >&2
  exit 2
fi

while pgrep -f "[p]ython .*alphazero_training.rapfi_distill.*longrun_round2_dagger2048" >/dev/null; do
  completed="$(grep -c '^pair ' "$ROUND2_LOG" 2>/dev/null || true)"
  echo "[$(date -Is)] waiting for round2 DAgger: ${completed}/2048 pairs"
  sleep 60
done

if [[ ! -f "$ROUND2" ]]; then
  echo "round2 report missing after data process exited" >&2
  exit 3
fi

if [[ ! -f "$JOINT" ]]; then
  "$PYTHON_BIN" -m alphazero_training.build_rapfi_joint_dataset \
    --report "$ROUND1" \
    --report "$ROUND2" \
    --output "$JOINT" \
    --symmetries 1 | tee "$RUN/build_joint.log"
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
  "v015:0.15:0.10:20270111"
  "v030:0.30:0.15:20270112"
  "v050:0.50:0.20:20270113"
)

for spec in "${variants[@]}"; do
  IFS=: read -r name value_scale value_kd seed <<<"$spec"
  output="$RUN/training/candidate_${name}_s8000.pt"
  done_marker="$output.done"
  if [[ ! -f "$done_marker" ]]; then
    echo "[$(date -Is)] training $name value_scale=$value_scale value_kd=$value_kd"
    "$PYTHON_BIN" -u -m alphazero_training.train_v3_supervised \
      --init-checkpoint "$PARENT" \
      --dataset "$JOINT" \
      --dataset "$LOSS_DATA" \
      --dataset "$TACTICS" \
      --dataset "$DDQK_TEACHER" \
      --dataset "$DDQK_EXPERT" \
      --dataset-weight 0.55 \
      --dataset-weight 0.10 \
      --dataset-weight 0.10 \
      --dataset-weight 0.05 \
      --dataset-weight 0.05 \
      "${white_args[@]}" \
      --output "$output" \
      --steps 8000 \
      --batch-size 1024 \
      --learning-rate 2e-5 \
      --min-learning-rate 2e-6 \
      --warmup-steps 500 \
      --freeze-trunk-steps 2000 \
      --train-last-residual-blocks-during-freeze 3 \
      --value-loss-scale "$value_scale" \
      --value-distill-scale "$value_kd" \
      --policy-distill-scale 0.25 \
      --safe-hard-negative-scale 0.20 \
      --safe-hard-negative-margin 1.0 \
      --mistake-hard-negative-scale 0.35 \
      --mistake-hard-negative-margin 1.0 \
      --random-d4-augmentation \
      --validation-fraction 0.1 \
      --eval-every 2000 \
      --seed "$seed" 2>&1 | tee "$RUN/training/train_${name}_s8000.log"
    sha256sum "$output" | tee "$output.sha256"
    touch "$done_marker"
  fi
done

evaluate_standard() {
  local name="$1"
  local source="$2"
  local source_hash frozen
  source_hash="$(sha256sum "$source" | awk '{print $1}')"
  frozen="$RUN/training/candidate_${name}_s8000_eval.pt"
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
    --json-out "$RUN/gate/${name}_tactics.json" >/dev/null
  "$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
    --checkpoint "$frozen" \
    --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
    --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
    --output "$RUN/gate/${name}_white.json" \
    --model-key best_model --device cuda >/dev/null
  "$PYTHON_BIN" -m alphazero_training.evaluate_supervised_dataset \
    --checkpoint "$frozen" --model-key best_model \
    --dataset "$LOSS_DATA" --seed 20261201 --validation-fraction 0.1 \
    --device cuda --output "$RUN/loss_eval/${name}.json" >/dev/null
  "$PYTHON_BIN" -m alphazero_training.evaluate_supervised_dataset \
    --checkpoint "$frozen" --model-key best_model \
    --dataset "$JOINT" --seed 20270111 --validation-fraction 0.1 \
    --device cuda --output "$RUN/joint_eval/${name}.json" >/dev/null
}

"$PYTHON_BIN" -m alphazero_training.v3_legal_tactics evaluate \
  --checkpoint "$PARENT" \
  --dataset alphazero_training/v3_legal_tactics_eval.npz \
  --model-key best_model --split eval --simulations 64 --device cuda \
  --json-out "$RUN/gate/parent_tactics.json" >/dev/null
"$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
  --checkpoint "$PARENT" \
  --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
  --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
  --output "$RUN/gate/parent_white.json" \
  --model-key best_model --device cuda >/dev/null
"$PYTHON_BIN" -m alphazero_training.evaluate_supervised_dataset \
  --checkpoint "$PARENT" --model-key best_model \
  --dataset "$LOSS_DATA" --seed 20261201 --validation-fraction 0.1 \
  --device cuda --output "$RUN/loss_eval/parent.json" >/dev/null
"$PYTHON_BIN" -m alphazero_training.evaluate_supervised_dataset \
  --checkpoint "$PARENT" --model-key best_model \
  --dataset "$JOINT" --seed 20270111 --validation-fraction 0.1 \
  --device cuda --output "$RUN/joint_eval/parent.json" >/dev/null

for spec in "${variants[@]}"; do
  IFS=: read -r name _ _ _ <<<"$spec"
  evaluate_standard "$name" "$RUN/training/candidate_${name}_s8000.pt"
done

"$PYTHON_BIN" - "$RUN" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
names = ("v015", "v030", "v050")
summary = {}
for name in ("parent", *names):
    tactics = json.loads((run / "gate" / f"{name}_tactics.json").read_text())
    white = json.loads((run / "gate" / f"{name}_white.json").read_text())
    loss = json.loads((run / "loss_eval" / f"{name}.json").read_text())["datasets"][0]
    joint = json.loads((run / "joint_eval" / f"{name}.json").read_text())["datasets"][0]
    summary[name] = {
        "checkpoint_sha256": tactics["checkpoint_sha256"],
        "raw_tactics": round(tactics["raw_network"]["top1"] * tactics["samples"]),
        "deployed_tactics": round(
            tactics["v3_search_with_exact_oracle"]["accuracy"] * tactics["samples"]
        ),
        "tactics_total": tactics["samples"],
        "white_safe_count": white["metrics"]["top1_in_safe_set_count"],
        "white_safe_total": white["metrics"]["records"],
        "white_safe_probability_mass": white["metrics"]["safe_probability_mass"],
        "loss_policy_top1": loss["policy_top1"],
        "loss_teacher_over_mistake": loss["teacher_over_mistake_rate"],
        "joint_policy_top1": joint["policy_top1"],
        "joint_value_mse": joint["value_mse"],
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
(run / "standard_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(run / "passed_candidates.txt").write_text("\n".join(passed) + ("\n" if passed else ""))
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not passed:
    raise SystemExit("no joint candidate passed the hard gate")
PY

MATCH="$RUN/screen_seed20270301"
mkdir -p "$MATCH/parent"
if [[ ! -f "$MATCH/parent/games.json" ]]; then
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
    --checkpoint "$PARENT" --engine "$ENGINE" \
    --report "$MATCH/parent/games.json" \
    --dataset "$MATCH/parent/policy_unused.npz" \
    --ai-loss-dir "$MATCH/parent/pending_training/ai_losses" \
    --pairs 128 --opening-plies 4 --simulations 64 --workers 8 \
    --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 \
    --seed 20270301
fi

while read -r name; do
  [[ -n "$name" ]] || continue
  mkdir -p "$MATCH/$name"
  if [[ ! -f "$MATCH/$name/games.json" ]]; then
    "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
      --checkpoint "$RUN/training/candidate_${name}_s8000_eval.pt" \
      --engine "$ENGINE" \
      --report "$MATCH/$name/games.json" \
      --dataset "$MATCH/$name/policy_unused.npz" \
      --ai-loss-dir "$MATCH/$name/pending_training/ai_losses" \
      --pairs 128 --opening-plies 4 --simulations 64 --workers 8 \
      --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 \
      --seed 20270301
  fi
  "$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
    --parent-report "$MATCH/parent/games.json" \
    --candidate-report "$MATCH/$name/games.json" \
    --output "$MATCH/${name}_comparison.json"
done < "$RUN/passed_candidates.txt"

"$PYTHON_BIN" - "$MATCH" "$RUN/passed_candidates.txt" <<'PY'
import json
from pathlib import Path
import sys

match = Path(sys.argv[1])
names = [line.strip() for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()]
rows = {name: json.loads((match / f"{name}_comparison.json").read_text()) for name in names}
best = max(rows, key=lambda name: (rows[name]["score_delta"], -rows[name]["two_sided_exact_sign_p"]))
(match / "screen_summary.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
if rows[best]["score_delta"] <= 0:
    raise SystemExit("no hard-gate candidate beat the parent in the Rapfi screen")
(match / "best_candidate.txt").write_text(best + "\n", encoding="utf-8")
print(json.dumps({"best_candidate": best, **rows[best]}, ensure_ascii=False, indent=2))
PY

best="$(tr -d '\r\n' < "$MATCH/best_candidate.txt")"
EXPANDED="$RUN/expanded_seed20270401"
mkdir -p "$EXPANDED/parent" "$EXPANDED/$best"
if [[ ! -f "$EXPANDED/parent/games.json" ]]; then
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
    --checkpoint "$PARENT" --engine "$ENGINE" \
    --report "$EXPANDED/parent/games.json" \
    --dataset "$EXPANDED/parent/policy_unused.npz" \
    --ai-loss-dir "$EXPANDED/parent/pending_training/ai_losses" \
    --pairs 512 --opening-plies 4 --simulations 64 --workers 8 \
    --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 \
    --seed 20270401
fi
if [[ ! -f "$EXPANDED/$best/games.json" ]]; then
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
    --checkpoint "$RUN/training/candidate_${best}_s8000_eval.pt" \
    --engine "$ENGINE" \
    --report "$EXPANDED/$best/games.json" \
    --dataset "$EXPANDED/$best/policy_unused.npz" \
    --ai-loss-dir "$EXPANDED/$best/pending_training/ai_losses" \
    --pairs 512 --opening-plies 4 --simulations 64 --workers 8 \
    --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 \
    --seed 20270401
fi
"$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
  --parent-report "$EXPANDED/parent/games.json" \
  --candidate-report "$EXPANDED/$best/games.json" \
  --output "$EXPANDED/comparison.json"

echo "[$(date -Is)] long-run joint pipeline complete; finalist=$best"
