#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e"
ANCHOR="rapfi_distillation/longrun_blend_recovery_round4/training/cons_a055_eval.pt"
ANCHOR_SHA256="92c9291d0371d7df5e1fc9eadda674a99c03f97f6fad22d8a7586e0d8efb6c7b"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"
JOINT="rapfi_distillation/longrun_joint_round2/joint_policy_value.npz"
BRANCH="rapfi_distillation/longrun_counterfactual_round3/branches.npz"
LOSS_DATA="rapfi_distillation/loss_correction_round1/loss_hard_negative.npz"
TACTICS="alphazero_training/v3_legal_tactics_train.npz"
DDQK_TEACHER="alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz"
DDQK_EXPERT="alphazero_training/ddqk_expert_50pairs_policy.npz"
DESKTOP_FULL="rapfi_distillation/desktop_loss_corrections/full_review/desktop_full_loss_curriculum.npz"
DESKTOP_EXACT="rapfi_distillation/desktop_loss_corrections/desktop_20260731T024514_932773Z_c48f1c86_move14_rapfi.npz"
DESKTOP_BRANCH="rapfi_distillation/desktop_loss_corrections/move14_counterfactual_branch.npz"
RUN="rapfi_distillation/desktop_loss_repair_round5"

cd "$ROOT"
mkdir -p "$RUN/training" "$RUN/gate" "$RUN/loss_eval" \
  "$RUN/branch_eval" "$RUN/screen_seed20271101"

verify_file() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

verify_file "$PARENT" "$PARENT_SHA256"
verify_file "$ANCHOR" "$ANCHOR_SHA256"
for path in "$JOINT" "$BRANCH" "$LOSS_DATA" "$TACTICS" "$DDQK_TEACHER" \
  "$DDQK_EXPERT" "$DESKTOP_FULL" "$DESKTOP_EXACT" "$DESKTOP_BRANCH"; do
  [[ -s "$path" ]] || { echo "missing Round5 input: $path" >&2; exit 2; }
done

white_args=()
for name in p2 p4 p8 seed20260724_parent seed20260730_parent; do
  white_args+=(
    --white-defense-npz "alphazero_training/_v3f_white_train_det/$name/train.npz"
    --white-defense-manifest "alphazero_training/_v3f_white_train_det/$name/manifest.json"
    --white-defense-weight 0.03
  )
done

UPDATE="$RUN/training/desktop_repair_update_s6000.pt"
if [[ ! -f "$UPDATE.done" ]]; then
  "$PYTHON_BIN" -u -m alphazero_training.train_v3_supervised \
    --init-checkpoint "$PARENT" \
    --dataset "$JOINT" \
    --dataset "$BRANCH" \
    --dataset "$LOSS_DATA" \
    --dataset "$TACTICS" \
    --dataset "$DDQK_TEACHER" \
    --dataset "$DDQK_EXPERT" \
    --dataset "$DESKTOP_FULL" \
    --dataset "$DESKTOP_BRANCH" \
    --dataset-weight 0.32 \
    --dataset-weight 0.16 \
    --dataset-weight 0.05 \
    --dataset-weight 0.10 \
    --dataset-weight 0.05 \
    --dataset-weight 0.05 \
    --dataset-weight 0.06 \
    --dataset-weight 0.06 \
    "${white_args[@]}" \
    --output "$UPDATE" \
    --steps 6000 \
    --batch-size 1024 \
    --learning-rate 6e-6 \
    --min-learning-rate 6e-7 \
    --warmup-steps 300 \
    --freeze-trunk-steps 4000 \
    --train-last-residual-blocks-during-freeze 3 \
    --value-loss-scale 0.25 \
    --value-distill-scale 0.15 \
    --policy-distill-scale 0.35 \
    --safe-hard-negative-scale 0.30 \
    --safe-hard-negative-margin 1.0 \
    --mistake-hard-negative-scale 0.50 \
    --mistake-hard-negative-margin 1.0 \
    --random-d4-augmentation \
    --validation-fraction 0.1 \
    --eval-every 1500 \
    --seed 20271031 2>&1 | tee "$RUN/training/train_s6000.log"
  sha256sum "$UPDATE" | tee "$UPDATE.sha256"
  touch "$UPDATE.done"
fi

update_sha="$(sha256sum "$UPDATE" | awk '{print $1}')"
alphas=("025:0.25" "050:0.50" "075:0.75" "100:1.00")
for spec in "${alphas[@]}"; do
  IFS=: read -r tag alpha <<<"$spec"
  source="$RUN/training/repair_a${tag}.pt"
  frozen="$RUN/training/repair_a${tag}_eval.pt"
  if [[ ! -s "$source" ]]; then
    "$PYTHON_BIN" -m alphazero_training.blend_v3_checkpoints \
      --anchor "$ANCHOR" --anchor-sha256 "$ANCHOR_SHA256" \
      --update "$UPDATE" --update-sha256 "$update_sha" \
      --alpha "$alpha" --parent-sha256 "$PARENT_SHA256" --output "$source"
  fi
  source_sha="$(sha256sum "$source" | awk '{print $1}')"
  if [[ ! -s "$frozen" ]]; then
    "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
      --source "$source" --expected-source-sha256 "$source_sha" \
      --expected-parent-sha256 "$PARENT_SHA256" --output "$frozen"
  fi
done

evaluate_static() {
  local name="$1" checkpoint="$2"
  "$PYTHON_BIN" -m alphazero_training.v3_legal_tactics evaluate \
    --checkpoint "$checkpoint" --dataset alphazero_training/v3_legal_tactics_eval.npz \
    --model-key best_model --split eval --simulations 64 --device cuda \
    --json-out "$RUN/gate/${name}_tactics.json" >/dev/null
  "$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
    --checkpoint "$checkpoint" \
    --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
    --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
    --output "$RUN/gate/${name}_white.json" --model-key best_model --device cuda >/dev/null
  "$PYTHON_BIN" -m alphazero_training.evaluate_supervised_dataset \
    --checkpoint "$checkpoint" --model-key best_model --dataset "$LOSS_DATA" \
    --seed 20261201 --validation-fraction 0.1 --device cuda \
    --output "$RUN/loss_eval/${name}.json" >/dev/null
  "$PYTHON_BIN" -m alphazero_training.evaluate_supervised_dataset \
    --checkpoint "$checkpoint" --model-key best_model --dataset "$BRANCH" \
    --seed 20270601 --validation-fraction 0.1 --device cuda \
    --output "$RUN/branch_eval/${name}.json" >/dev/null
}

evaluate_static parent "$PARENT"
evaluate_static anchor "$ANCHOR"
for spec in "${alphas[@]}"; do
  tag="${spec%%:*}"
  name="repair_a${tag}"
  checkpoint="$RUN/training/${name}_eval.pt"
  set +e
  "$PYTHON_BIN" -m alphazero_training.evaluate_desktop_correction \
    --checkpoint "$checkpoint" --dataset "$DESKTOP_EXACT" --simulations 256 \
    --output "$RUN/gate/${name}_desktop_move14.json" >/dev/null
  exact_status=$?
  set -e
  evaluate_static "$name" "$checkpoint"
  echo "$exact_status" > "$RUN/gate/${name}_desktop_move14.exit"
done

"$PYTHON_BIN" - "$RUN" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
names = [f"repair_a{tag}" for tag in ("025", "050", "075", "100")]

def static(name):
    tactics = json.loads((run / "gate" / f"{name}_tactics.json").read_text())
    white = json.loads((run / "gate" / f"{name}_white.json").read_text())
    loss = json.loads((run / "loss_eval" / f"{name}.json").read_text())["datasets"][0]
    branch = json.loads((run / "branch_eval" / f"{name}.json").read_text())["datasets"][0]
    return {
        "raw_tactics": round(tactics["raw_network"]["top1"] * tactics["samples"]),
        "deployed_tactics": round(tactics["v3_search_with_exact_oracle"]["accuracy"] * tactics["samples"]),
        "tactics_total": tactics["samples"],
        "white_safe_count": white["metrics"]["top1_in_safe_set_count"],
        "white_safe_probability_mass": white["metrics"]["safe_probability_mass"],
        "loss_policy_top1": loss["policy_top1"],
        "branch_policy_top1": branch["policy_top1"],
    }

parent = static("parent")
anchor = static("anchor")
summary = {"parent": parent, "anchor": anchor}
passed = []
for name in names:
    row = static(name)
    exact = json.loads((run / "gate" / f"{name}_desktop_move14.json").read_text())
    row["desktop_move14"] = exact
    row["passed"] = bool(
        exact["passed"]
        and row["raw_tactics"] >= parent["raw_tactics"]
        and row["deployed_tactics"] >= parent["deployed_tactics"]
        and row["white_safe_count"] >= parent["white_safe_count"]
        and row["white_safe_probability_mass"] >= parent["white_safe_probability_mass"]
    )
    summary[name] = row
    if row["passed"]:
        passed.append(name)
passed.sort(key=lambda name: (
    summary[name]["desktop_move14"]["raw_network"]["teacher_probability"],
    summary[name]["loss_policy_top1"],
    summary[name]["branch_policy_top1"],
), reverse=True)
(run / "gate" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(run / "gate" / "passed.txt").write_text("\n".join(passed[:2]) + ("\n" if passed else ""))
print(json.dumps({"passed": passed, "screen": passed[:2]}, indent=2))
if not passed:
    raise SystemExit("no repaired blend passed exact and static gates")
PY

SCREEN="$RUN/screen_seed20271101"
mkdir -p "$SCREEN/parent" "$SCREEN/anchor"
for baseline in parent anchor; do
  if [[ "$baseline" == parent ]]; then checkpoint="$PARENT"; else checkpoint="$ANCHOR"; fi
  if [[ ! -s "$SCREEN/$baseline/games.json" ]]; then
    "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
      --checkpoint "$checkpoint" --engine "$ENGINE" \
      --report "$SCREEN/$baseline/games.json" --dataset "$SCREEN/$baseline/policy_unused.npz" \
      --ai-loss-dir "$SCREEN/$baseline/pending_training/ai_losses" \
      --pairs 256 --opening-plies 4 --simulations 64 --workers 8 \
      --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed 20271101
  fi
done

while read -r name; do
  [[ -n "$name" ]] || continue
  mkdir -p "$SCREEN/$name"
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
    --checkpoint "$RUN/training/${name}_eval.pt" --engine "$ENGINE" \
    --report "$SCREEN/$name/games.json" --dataset "$SCREEN/$name/policy_unused.npz" \
    --ai-loss-dir "$SCREEN/$name/pending_training/ai_losses" \
    --pairs 256 --opening-plies 4 --simulations 64 --workers 8 \
    --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed 20271101
  "$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
    --parent-report "$SCREEN/parent/games.json" --candidate-report "$SCREEN/$name/games.json" \
    --output "$SCREEN/${name}_vs_parent.json"
  "$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
    --parent-report "$SCREEN/anchor/games.json" --candidate-report "$SCREEN/$name/games.json" \
    --output "$SCREEN/${name}_vs_anchor.json"
done < "$RUN/gate/passed.txt"

"$PYTHON_BIN" - "$RUN" <<'PY'
import hashlib, json
from pathlib import Path
import sys
run = Path(sys.argv[1]); screen = run / "screen_seed20271101"
names = [name for name in (run / "gate" / "passed.txt").read_text().split() if name]
rows = {}
for name in names:
    parent = json.loads((screen / f"{name}_vs_parent.json").read_text())
    anchor = json.loads((screen / f"{name}_vs_anchor.json").read_text())
    rows[name] = {"vs_parent": parent, "vs_anchor": anchor}
eligible = [name for name in names if rows[name]["vs_parent"]["score_delta"] > 0]
best = max(eligible, key=lambda n: (rows[n]["vs_anchor"]["score_delta"], rows[n]["vs_parent"]["score_delta"]), default=None)
payload = {"candidates": rows, "best": best}
(run / "screen_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
if best:
    checkpoint = run / "training" / f"{best}_eval.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (run / "SCREEN_WINNER.txt").write_text(f"{best}\n{digest}\n")
print(json.dumps(payload, indent=2))
if best is None:
    raise SystemExit("no repaired candidate retained a positive parent screen")
PY

echo "[$(date -Is)] Round5 desktop loss repair screen complete; latest.pt unchanged"
