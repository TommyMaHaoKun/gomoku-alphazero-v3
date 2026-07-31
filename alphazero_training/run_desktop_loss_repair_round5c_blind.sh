#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e"
ANCHOR="rapfi_distillation/longrun_blend_recovery_round4/training/cons_a055_eval.pt"
ANCHOR_SHA256="92c9291d0371d7df5e1fc9eadda674a99c03f97f6fad22d8a7586e0d8efb6c7b"
CANDIDATE="rapfi_distillation/desktop_loss_repair_round5c/training/last3_repair_s8000_eval.pt"
CANDIDATE_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"
RUN="rapfi_distillation/desktop_loss_repair_round5c"
SCREEN="$RUN/blind_seed20271201"
EXPANDED="$RUN/blind_seed20271301"

cd "$ROOT"
mkdir -p "$SCREEN" "$EXPANDED"

verify_file() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  }
}
verify_file "$PARENT" "$PARENT_SHA256"
verify_file "$ANCHOR" "$ANCHOR_SHA256"
verify_file "$CANDIDATE" "$CANDIDATE_SHA256"

run_report() {
  local root="$1" name="$2" checkpoint="$3" pairs="$4" seed="$5"
  mkdir -p "$root/$name"
  if [[ ! -s "$root/$name/games.json" ]]; then
    "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
      --checkpoint "$checkpoint" --engine "$ENGINE" \
      --report "$root/$name/games.json" \
      --dataset "$root/$name/policy_unused.npz" \
      --ai-loss-dir "$root/$name/pending_training/ai_losses" \
      --pairs "$pairs" --opening-plies 4 --simulations 64 --workers 8 \
      --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed "$seed"
  fi
}

run_report "$SCREEN" parent "$PARENT" 256 20271201
run_report "$SCREEN" anchor "$ANCHOR" 256 20271201
run_report "$SCREEN" candidate "$CANDIDATE" 256 20271201
"$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
  --parent-report "$SCREEN/parent/games.json" \
  --candidate-report "$SCREEN/candidate/games.json" \
  --output "$SCREEN/candidate_vs_parent.json"
"$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
  --parent-report "$SCREEN/anchor/games.json" \
  --candidate-report "$SCREEN/candidate/games.json" \
  --output "$SCREEN/candidate_vs_anchor.json"

"$PYTHON_BIN" - "$SCREEN" <<'PY'
import json
from pathlib import Path
import sys
p=Path(sys.argv[1])
parent=json.loads((p/'candidate_vs_parent.json').read_text())
anchor=json.loads((p/'candidate_vs_anchor.json').read_text())
passed=bool(parent['score_delta']>0 and anchor['score_delta']>=-0.015625)
payload={'passed':passed,'candidate_vs_parent':parent,'candidate_vs_anchor':anchor}
(p/'screen_gate.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
if not passed: raise SystemExit('Round5c candidate failed fresh 256-pair screen')
PY

run_report "$EXPANDED" parent "$PARENT" 1024 20271301
run_report "$EXPANDED" candidate "$CANDIDATE" 1024 20271301
"$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
  --parent-report "$EXPANDED/parent/games.json" \
  --candidate-report "$EXPANDED/candidate/games.json" \
  --output "$EXPANDED/candidate_vs_parent.json"

"$PYTHON_BIN" - "$ROOT" "$RUN" "$PARENT_SHA256" "$CANDIDATE_SHA256" <<'PY'
import hashlib, json
from pathlib import Path
import sys
root=Path(sys.argv[1]); run=Path(sys.argv[2]); parent_sha=sys.argv[3]; candidate_sha=sys.argv[4]
comparison=json.loads((run/'blind_seed20271301/candidate_vs_parent.json').read_text())
static=json.loads((run/'gate/summary.json').read_text())
parent_report=json.loads((run/'blind_seed20271301/parent/games.json').read_text())
candidate_report=json.loads((run/'blind_seed20271301/candidate/games.json').read_text())
reports_complete=bool(
 parent_report.get('complete') and candidate_report.get('complete')
 and parent_report['summary']['complete_pairs']==1024
 and candidate_report['summary']['complete_pairs']==1024
 and parent_report['summary']['errors']==0 and candidate_report['summary']['errors']==0
 and parent_report['summary']['truncated']==0 and candidate_report['summary']['truncated']==0
)
colours_ok=all(
 comparison['candidate_by_color'][colour]['score'] >= comparison['parent_by_color'][colour]['score']
 for colour in ('black','white')
)
passed=bool(
 static['passed_static_and_exact'] and reports_complete and colours_ok
 and comparison['score_delta']>0
 and comparison['two_sided_exact_sign_p']<0.05
 and comparison['parent_checkpoint_sha256']==parent_sha
 and comparison['candidate_checkpoint_sha256']==candidate_sha
)
payload={
 'eligible_for_deployment':passed,
 'static_and_exact':static,
 'reports_complete':reports_complete,
 'colours_ok':colours_ok,
 'comparison':comparison,
 'artifact_sha256':{
  'candidate':hashlib.sha256((root/'rapfi_distillation/desktop_loss_repair_round5c/training/last3_repair_s8000_eval.pt').read_bytes()).hexdigest(),
  'static_gate':hashlib.sha256((run/'gate/summary.json').read_bytes()).hexdigest(),
  'screen_gate':hashlib.sha256((run/'blind_seed20271201/screen_gate.json').read_bytes()).hexdigest(),
  'expanded_comparison':hashlib.sha256((run/'blind_seed20271301/candidate_vs_parent.json').read_bytes()).hexdigest(),
 }
}
(run/'FINAL_DEPLOYMENT_GATE.json').write_text(json.dumps(payload,indent=2)+'\n')
if passed:
 (run/'ELIGIBLE_FOR_DEPLOYMENT').write_text(candidate_sha+'\n')
print(json.dumps(payload,indent=2))
if not passed: raise SystemExit('Round5c candidate failed final independent deployment gate')
PY

echo "[$(date -Is)] Round5c final blind gate passed; eligible but latest.pt unchanged"
