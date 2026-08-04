#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e"
ANCHOR="rapfi_distillation/longrun_blend_recovery_round4/training/cons_a055_eval.pt"
ANCHOR_SHA256="92c9291d0371d7df5e1fc9eadda674a99c03f97f6fad22d8a7586e0d8efb6c7b"
JOINT="rapfi_distillation/longrun_joint_round2/joint_policy_value.npz"
BRANCH="rapfi_distillation/longrun_counterfactual_round3/branches.npz"
LOSS_DATA="rapfi_distillation/loss_correction_round1/loss_hard_negative.npz"
TACTICS="alphazero_training/v3_legal_tactics_train.npz"
DDQK_TEACHER="alphazero_training/ddqk_teacher_selfplay_g128x8_seed20260726.npz"
DDQK_EXPERT="alphazero_training/ddqk_expert_50pairs_policy.npz"
DESKTOP_FULL="rapfi_distillation/desktop_loss_corrections/full_review/desktop_full_loss_curriculum.npz"
DESKTOP_EXACT="rapfi_distillation/desktop_loss_corrections/desktop_20260731T024514_932773Z_c48f1c86_move14_rapfi.npz"
DESKTOP_BRANCH="rapfi_distillation/desktop_loss_corrections/move14_counterfactual_branch.npz"
RUN="rapfi_distillation/desktop_loss_repair_round5b"
OUTPUT="$RUN/training/head_repair_s8000.pt"
FROZEN="$RUN/training/head_repair_s8000_eval.pt"

cd "$ROOT"
mkdir -p "$RUN/training" "$RUN/gate"
[[ "$(sha256sum "$PARENT" | awk '{print $1}')" == "$PARENT_SHA256" ]]
[[ "$(sha256sum "$ANCHOR" | awk '{print $1}')" == "$ANCHOR_SHA256" ]]

white_args=()
for name in p2 p4 p8 seed20260724_parent seed20260730_parent; do
  white_args+=(
    --white-defense-npz "alphazero_training/_v3f_white_train_det/$name/train.npz"
    --white-defense-manifest "alphazero_training/_v3f_white_train_det/$name/manifest.json"
    --white-defense-weight 0.04
  )
done

if [[ ! -f "$OUTPUT.done" ]]; then
  "$PYTHON_BIN" -u -m alphazero_training.train_v3_supervised \
    --init-checkpoint "$ANCHOR" \
    --dataset "$JOINT" --dataset "$BRANCH" --dataset "$LOSS_DATA" \
    --dataset "$TACTICS" --dataset "$DDQK_TEACHER" --dataset "$DDQK_EXPERT" \
    --dataset "$DESKTOP_FULL" --dataset "$DESKTOP_BRANCH" \
    --dataset-weight 0.25 --dataset-weight 0.10 --dataset-weight 0.05 \
    --dataset-weight 0.15 --dataset-weight 0.04 --dataset-weight 0.04 \
    --dataset-weight 0.12 --dataset-weight 0.05 \
    "${white_args[@]}" \
    --output "$OUTPUT" --steps 8000 --batch-size 1024 \
    --learning-rate 3e-6 --min-learning-rate 3e-7 --warmup-steps 400 \
    --freeze-trunk-steps 8000 --train-last-residual-blocks-during-freeze 0 \
    --value-loss-scale 0.10 --value-distill-scale 0.20 \
    --policy-distill-scale 0.50 \
    --safe-hard-negative-scale 0.40 --safe-hard-negative-margin 1.0 \
    --mistake-hard-negative-scale 0.60 --mistake-hard-negative-margin 1.0 \
    --random-d4-augmentation --validation-fraction 0.1 --eval-every 2000 \
    --seed 20271115 2>&1 | tee "$RUN/training/train_s8000.log"
  sha256sum "$OUTPUT" > "$OUTPUT.sha256"
  touch "$OUTPUT.done"
fi

source_sha="$(sha256sum "$OUTPUT" | awk '{print $1}')"
if [[ ! -s "$FROZEN" ]]; then
  "$PYTHON_BIN" -m alphazero_training.v3_candidate_gate freeze \
    --source "$OUTPUT" --expected-source-sha256 "$source_sha" \
    --expected-parent-sha256 "$ANCHOR_SHA256" --output "$FROZEN"
fi

set +e
"$PYTHON_BIN" -m alphazero_training.evaluate_desktop_correction \
  --checkpoint "$FROZEN" --dataset "$DESKTOP_EXACT" --simulations 256 \
  --output "$RUN/gate/desktop_move14.json" >/dev/null
exact_status=$?
set -e
echo "$exact_status" > "$RUN/gate/desktop_move14.exit"

"$PYTHON_BIN" -m alphazero_training.v3_legal_tactics evaluate \
  --checkpoint "$FROZEN" --dataset alphazero_training/v3_legal_tactics_eval.npz \
  --model-key best_model --split eval --simulations 64 --device cuda \
  --json-out "$RUN/gate/tactics.json" >/dev/null
"$PYTHON_BIN" -m alphazero_training.evaluate_white_defense \
  --checkpoint "$FROZEN" \
  --eval-npz alphazero_training/_v3f_white_eval_det/seed20260801_parent/eval.npz \
  --manifest alphazero_training/_v3f_white_eval_det/seed20260801_parent/manifest.json \
  --output "$RUN/gate/white.json" --model-key best_model --device cuda >/dev/null

"$PYTHON_BIN" - "$RUN" <<'PY'
import json
from pathlib import Path
import sys
run=Path(sys.argv[1])
exact=json.loads((run/'gate/desktop_move14.json').read_text())
tactics=json.loads((run/'gate/tactics.json').read_text())
white=json.loads((run/'gate/white.json').read_text())
summary={
 'desktop_move14_passed':exact['passed'],
 'desktop_teacher_rank':exact['raw_network']['teacher_legal_rank'],
 'desktop_teacher_probability':exact['raw_network']['teacher_probability'],
 'raw_tactics':round(tactics['raw_network']['top1']*tactics['samples']),
 'deployed_tactics':round(tactics['v3_search_with_exact_oracle']['accuracy']*tactics['samples']),
 'white_safe_count':white['metrics']['top1_in_safe_set_count'],
 'white_safe_probability_mass':white['metrics']['safe_probability_mass'],
}
summary['passed_static_and_exact']=bool(
 summary['desktop_move14_passed'] and summary['raw_tactics']>=47
 and summary['deployed_tactics']>=48 and summary['white_safe_count']>=16
 and summary['white_safe_probability_mass']>=0.7245202151762786
)
(run/'gate/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
if not summary['passed_static_and_exact']:
 raise SystemExit('Round5b candidate failed exact or static gate')
PY

echo "[$(date -Is)] Round5b static and exact gates passed; latest.pt unchanged"
