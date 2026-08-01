#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"
ENGINE_SHA256="95cea7a20374f8fa2795d4b410b215a7f058c9e2b659d8c0670bcb86761e6f46"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"
RUN="rapfi_distillation/round6_regret_restart"
TRAIN="$RUN/training"
GATE="$RUN/gate"
SCREEN="$RUN/screen_seed20280201"
FINAL="$RUN/final_seed20280301"

cd "$ROOT"
mkdir -p "$TRAIN" "$GATE" "$RUN/logs" "$SCREEN" "$FINAL"

verify_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  }
}

verify_sha "$PARENT" "$PARENT_SHA256"
verify_sha "$ENGINE" "$ENGINE_SHA256"

while [[ ! -s "$RUN/DATA_STAGE_COMPLETE.json" ]]; do
  data_pid="$(cat "$RUN/data_pipeline.pid" 2>/dev/null || true)"
  if [[ -n "$data_pid" ]] && ! kill -0 "$data_pid" 2>/dev/null; then
    echo "Round6 data pipeline stopped before completing" >&2
    exit 3
  fi
  echo "[$(date -Is)] waiting for Round6 data stage"
  sleep 60
done

NEW_JOINT="$RUN/datasets/joint_current_champion.npz"
NEW_COUNTER="$RUN/counterfactual/branches.npz"
NEW_LOSS="$RUN/datasets/regret_loss_curriculum.npz"
OLD_COUNTER="rapfi_distillation/longrun_counterfactual_round3/branches.npz"
TACTICS="alphazero_training/v3_legal_tactics_train.npz"
DDQK_TEACHER="alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz"
DDQK_EXPERT="alphazero_training/ddqk_expert_50pairs_policy.npz"
DESKTOP_FULL="rapfi_distillation/desktop_loss_corrections/full_review/desktop_full_loss_curriculum.npz"
MOVE14_BRANCH="rapfi_distillation/desktop_loss_corrections/move14_counterfactual_branch.npz"
MOVE14_EXACT="rapfi_distillation/desktop_loss_corrections/desktop_20260731T024514_932773Z_c48f1c86_move14_rapfi.npz"

for path in "$NEW_JOINT" "$NEW_COUNTER" "$NEW_LOSS" "$OLD_COUNTER" \
  "$TACTICS" "$DDQK_TEACHER" "$DDQK_EXPERT" "$DESKTOP_FULL" \
  "$MOVE14_BRANCH" "$MOVE14_EXACT"; do
  [[ -s "$path" ]] || { echo "required Round6 artifact missing: $path" >&2; exit 4; }
done

train_candidate() {
  local name="$1" lr="$2" min_lr="$3" white_weight="$4" seed="$5"
  shift 5
  local weights=("$@")
  local output="$TRAIN/${name}_s12000.pt"
  local frozen="$TRAIN/${name}_s12000_eval.pt"
  if [[ ! -f "$output.done" ]]; then
    white_args=()
    for white_name in p2 p4 p8 seed20260724_parent seed20260730_parent; do
      white_args+=(
        --white-defense-npz "alphazero_training/_v3f_white_train_det/$white_name/train.npz"
        --white-defense-manifest "alphazero_training/_v3f_white_train_det/$white_name/manifest.json"
        --white-defense-weight "$white_weight"
      )
    done
    "$PYTHON_BIN" -u -m alphazero_training.train_v3_supervised \
      --init-checkpoint "$PARENT" \
      --dataset "$NEW_JOINT" \
      --dataset "$NEW_COUNTER" \
      --dataset "$NEW_LOSS" \
      --dataset "$OLD_COUNTER" \
      --dataset "$TACTICS" \
      --dataset "$DDQK_TEACHER" \
      --dataset "$DDQK_EXPERT" \
      --dataset "$DESKTOP_FULL" \
      --dataset "$MOVE14_BRANCH" \
      --dataset-weight "${weights[0]}" \
      --dataset-weight "${weights[1]}" \
      --dataset-weight "${weights[2]}" \
      --dataset-weight "${weights[3]}" \
      --dataset-weight "${weights[4]}" \
      --dataset-weight "${weights[5]}" \
      --dataset-weight "${weights[6]}" \
      --dataset-weight "${weights[7]}" \
      --dataset-weight "${weights[8]}" \
      "${white_args[@]}" \
      --output "$output" \
      --steps 12000 --batch-size 1024 \
      --learning-rate "$lr" --min-learning-rate "$min_lr" \
      --warmup-steps 600 \
      --freeze-trunk-steps 8000 \
      --train-last-residual-blocks-during-freeze 3 \
      --value-loss-scale 0.12 \
      --value-distill-scale 0.25 \
      --policy-distill-scale 0.80 \
      --safe-hard-negative-scale 0.60 \
      --safe-hard-negative-margin 1.0 \
      --mistake-hard-negative-scale 1.00 \
      --mistake-hard-negative-margin 1.0 \
      --random-d4-augmentation \
      --validation-fraction 0.1 \
      --eval-every 3000 \
      --seed "$seed" 2>&1 | tee "$RUN/logs/train_${name}.log"
    sha256sum "$output" > "$output.sha256"
    touch "$output.done"
  fi
  if [[ ! -s "$frozen" ]]; then
    local source_sha
    source_sha="$(sha256sum "$output" | awk '{print $1}')"
    "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
      --source "$output" --expected-source-sha256 "$source_sha" \
      --expected-parent-sha256 "$PARENT_SHA256" --output "$frozen"
  fi
}

train_candidate r6_balanced 3e-6 3e-7 0.03 20280401 \
  0.22 0.22 0.10 0.05 0.12 0.04 0.03 0.05 0.02
train_candidate r6_white 4e-6 4e-7 0.02 20280402 \
  0.16 0.30 0.14 0.05 0.10 0.03 0.02 0.06 0.04

: > "$RUN/candidate_manifest.tsv"
for family in r6_balanced r6_white; do
  update="$TRAIN/${family}_s12000_eval.pt"
  update_sha="$(sha256sum "$update" | awk '{print $1}')"
  printf '%s\t%s\n' "$family" "$update" >> "$RUN/candidate_manifest.tsv"
  for alpha_spec in 035:0.35 045:0.45 055:0.55 065:0.65; do
    IFS=: read -r tag alpha <<<"$alpha_spec"
    name="${family}_a${tag}"
    source="$TRAIN/${name}.pt"
    frozen="$TRAIN/${name}_eval.pt"
    if [[ ! -s "$source" ]]; then
      "$PYTHON_BIN" -m alphazero_training.blend_v3_checkpoints \
        --anchor "$PARENT" --anchor-sha256 "$PARENT_SHA256" \
        --update "$update" --update-sha256 "$update_sha" \
        --alpha "$alpha" --parent-sha256 "$PARENT_SHA256" \
        --output "$source"
    fi
    if [[ ! -s "$frozen" ]]; then
      source_sha="$(sha256sum "$source" | awk '{print $1}')"
      "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
        --source "$source" --expected-source-sha256 "$source_sha" \
        --expected-parent-sha256 "$PARENT_SHA256" --output "$frozen"
    fi
    printf '%s\t%s\n' "$name" "$frozen" >> "$RUN/candidate_manifest.tsv"
  done
done

evaluate_candidate() {
  local name="$1" checkpoint="$2"
  if [[ ! -s "$GATE/${name}_move14.json" ]]; then
    set +e
    "$PYTHON_BIN" -m alphazero_training.evaluate_desktop_correction \
      --checkpoint "$checkpoint" --dataset "$MOVE14_EXACT" \
      --simulations 256 --output "$GATE/${name}_move14.json" >/dev/null
    echo "$?" > "$GATE/${name}_move14.exit"
    set -e
  fi
  if [[ ! -s "$GATE/${name}_tactics.json" ]]; then
    "$PYTHON_BIN" -m alphazero_training.v3_legal_tactics evaluate \
      --checkpoint "$checkpoint" \
      --dataset alphazero_training/v3_legal_tactics_eval.npz \
      --model-key best_model --split eval --simulations 64 --device cuda \
      --json-out "$GATE/${name}_tactics.json" >/dev/null
  fi
  if [[ ! -s "$GATE/${name}_white.json" ]]; then
    "$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
      --checkpoint "$checkpoint" \
      --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
      --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
      --output "$GATE/${name}_white.json" --model-key best_model --device cuda >/dev/null
  fi
}

while IFS=$'\t' read -r name checkpoint; do
  evaluate_candidate "$name" "$checkpoint"
done < "$RUN/candidate_manifest.tsv"

"$PYTHON_BIN" - "$RUN" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
gate = run / "gate"
manifest = {}
for line in (run / "candidate_manifest.tsv").read_text().splitlines():
    name, path = line.split("\t")
    manifest[name] = path
summary = {}
for name, checkpoint in manifest.items():
    tactics = json.loads((gate / f"{name}_tactics.json").read_text())
    white = json.loads((gate / f"{name}_white.json").read_text())
    move14 = json.loads((gate / f"{name}_move14.json").read_text())
    row = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": tactics["checkpoint_sha256"],
        "raw_tactics": round(tactics["raw_network"]["top1"] * tactics["samples"]),
        "deployed_tactics": round(
            tactics["v3_search_with_exact_oracle"]["accuracy"] * tactics["samples"]
        ),
        "white_safe_count": white["metrics"]["top1_in_safe_set_count"],
        "white_safe_probability_mass": white["metrics"]["safe_probability_mass"],
        "move14_passed": move14["passed"],
        "move14_teacher_rank": move14["raw_network"]["teacher_legal_rank"],
        "move14_teacher_probability": move14["raw_network"]["teacher_probability"],
    }
    row["hard_gate_passed"] = bool(
        row["raw_tactics"] >= 47
        and row["deployed_tactics"] >= 48
        and row["white_safe_count"] >= 16
        and row["white_safe_probability_mass"] >= 0.7707617002141821
    )
    summary[name] = row
passed = [name for name, row in summary.items() if row["hard_gate_passed"]]
passed.sort(
    key=lambda name: (
        summary[name]["raw_tactics"],
        summary[name]["white_safe_count"],
        summary[name]["white_safe_probability_mass"],
        summary[name]["move14_passed"],
        summary[name]["move14_teacher_probability"],
    ),
    reverse=True,
)
selected = passed[:4]
(run / "static_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(run / "screen_candidates.txt").write_text("\n".join(selected) + ("\n" if selected else ""))
print(json.dumps({"passed": passed, "selected_for_screen": selected}, indent=2))
if not selected:
    raise SystemExit("no Round6 candidate passed the static non-regression gate")
PY

run_report() {
  local root="$1" name="$2" checkpoint="$3" pairs="$4" seed="$5"
  mkdir -p "$root/$name"
  if [[ ! -s "$root/$name/complete.done" ]]; then
    "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
      --checkpoint "$checkpoint" --engine "$ENGINE" \
      --report "$root/$name/games.json" \
      --dataset "$root/$name/policy_unused.npz" \
      --ai-loss-dir "$root/$name/pending_training/ai_losses" \
      --pairs "$pairs" --opening-plies 6 --simulations 64 --workers 12 \
      --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed "$seed" \
      2>&1 | tee "$RUN/logs/${name}_pairs${pairs}_seed${seed}.log"
    touch "$root/$name/complete.done"
  fi
}

run_report "$SCREEN" parent "$PARENT" 256 20280201
while read -r name; do
  [[ -n "$name" ]] || continue
  checkpoint="$(awk -F '\t' -v n="$name" '$1==n {print $2}' "$RUN/candidate_manifest.tsv")"
  run_report "$SCREEN" "$name" "$checkpoint" 256 20280201
  "$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
    --parent-report "$SCREEN/parent/games.json" \
    --candidate-report "$SCREEN/$name/games.json" \
    --output "$SCREEN/${name}_comparison.json"
done < "$RUN/screen_candidates.txt"

"$PYTHON_BIN" - "$RUN" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1]); screen = run / "screen_seed20280201"
names = [line.strip() for line in (run / "screen_candidates.txt").read_text().splitlines() if line.strip()]
rows = {name: json.loads((screen / f"{name}_comparison.json").read_text()) for name in names}
eligible = [name for name, row in rows.items() if row["score_delta"] > 0]
if not eligible:
    raise SystemExit("no Round6 candidate beat the parent in the fresh Rapfi screen")
best = max(eligible, key=lambda name: (rows[name]["score_delta"], -rows[name]["two_sided_exact_sign_p"]))
(screen / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
(screen / "best_candidate.txt").write_text(best + "\n")
print(json.dumps({"best_candidate": best, **rows[best]}, indent=2))
PY

best="$(tr -d '\r\n' < "$SCREEN/best_candidate.txt")"
best_checkpoint="$(awk -F '\t' -v n="$best" '$1==n {print $2}' "$RUN/candidate_manifest.tsv")"
run_report "$FINAL" parent "$PARENT" 1024 20280301
run_report "$FINAL" "$best" "$best_checkpoint" 1024 20280301
"$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
  --parent-report "$FINAL/parent/games.json" \
  --candidate-report "$FINAL/$best/games.json" \
  --output "$FINAL/comparison.json"

"$PYTHON_BIN" - "$ROOT" "$RUN" "$best" "$best_checkpoint" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]); run = Path(sys.argv[2]); name = sys.argv[3]
checkpoint = Path(sys.argv[4])
comparison = json.loads((run / "final_seed20280301/comparison.json").read_text())
static = json.loads((run / "static_gate_summary.json").read_text())[name]
parent = json.loads((run / "final_seed20280301/parent/games.json").read_text())
candidate = json.loads((run / f"final_seed20280301/{name}/games.json").read_text())
complete = all(
    payload.get("complete")
    and payload["summary"]["complete_pairs"] == 1024
    and payload["summary"]["errors"] == 0
    and payload["summary"]["truncated"] == 0
    for payload in (parent, candidate)
)
colours_ok = all(
    comparison["candidate_by_color"][colour]["score"]
    >= comparison["parent_by_color"][colour]["score"]
    for colour in ("black", "white")
)
passed = bool(
    static["hard_gate_passed"]
    and complete
    and colours_ok
    and comparison["score_delta"] > 0
    and comparison["two_sided_exact_sign_p"] < 0.05
)
payload = {
    "rapfi_eligible_pending_direct_arena": passed,
    "candidate_name": name,
    "candidate_checkpoint": str(checkpoint),
    "candidate_sha256": hashlib.sha256((root / checkpoint).read_bytes()).hexdigest(),
    "static_gate": static,
    "reports_complete": complete,
    "colours_ok": colours_ok,
    "comparison": comparison,
}
(run / "ROUND6_FINAL_GATE.json").write_text(json.dumps(payload, indent=2) + "\n")
if passed:
    (run / "RAPFI_ELIGIBLE_PENDING_DIRECT_ARENA").write_text(payload["candidate_sha256"] + "\n")
print(json.dumps(payload, indent=2))
if not passed:
    raise SystemExit("Round6 finalist failed the independent Rapfi gate")
PY

echo "[$(date -Is)] Round6 finalist passed Rapfi gate; direct champion arena still required"
