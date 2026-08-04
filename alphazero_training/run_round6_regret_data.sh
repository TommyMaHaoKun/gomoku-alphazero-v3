#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"
ENGINE_SHA256="95cea7a20374f8fa2795d4b410b215a7f058c9e2b659d8c0670bcb86761e6f46"
CHAMPION="alphazero_training/latest.pt"
CHAMPION_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"
SOURCE_REPORT="rapfi_distillation/desktop_loss_repair_round5c/blind_seed20271301/candidate/games.json"
SOURCE_REPORT_SHA256="0473750d8c9b13f8a9c7843af4d39efa59ac95828e7d1f57668d7c4db623f77c"
RUN="rapfi_distillation/round6_regret_restart"

cd "$ROOT"
mkdir -p "$RUN"/{counterfactual,dagger64,dagger128,datasets,logs}

verify_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

verify_sha "$ENGINE" "$ENGINE_SHA256"
verify_sha "$CHAMPION" "$CHAMPION_SHA256"
verify_sha "$SOURCE_REPORT" "$SOURCE_REPORT_SHA256"

if [[ ! -f "$RUN/datasets/regret_loss_curriculum.done" ]]; then
  "$PYTHON_BIN" -m alphazero_training.build_rapfi_loss_curriculum \
    --report "$SOURCE_REPORT" \
    --output "$RUN/datasets/regret_loss_curriculum.npz" \
    --symmetries 8 | tee "$RUN/logs/build_loss_curriculum.log"
  touch "$RUN/datasets/regret_loss_curriculum.done"
fi

run_counterfactual() {
  if [[ -f "$RUN/counterfactual/complete.done" ]]; then
    return
  fi
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_counterfactual \
    --report "$SOURCE_REPORT" \
    --engine "$ENGINE" \
    --output-report "$RUN/counterfactual/branches.json" \
    --output-dataset "$RUN/counterfactual/branches.npz" \
    --max-tasks 8192 \
    --max-branch-plies 64 \
    --workers 12 \
    --timeout-turn-ms 300 \
    --max-nodes 100000 \
    --engine-threads 4 \
    --seed 20280101 2>&1 | tee "$RUN/logs/counterfactual.log"
  sha256sum "$RUN/counterfactual/branches.json" \
    "$RUN/counterfactual/branches.npz" > "$RUN/counterfactual/SHA256SUMS"
  touch "$RUN/counterfactual/complete.done"
}

run_dagger() {
  local name="$1" pairs="$2" simulations="$3" seed="$4"
  local root="$RUN/$name"
  if [[ -f "$root/complete.done" ]]; then
    return
  fi
  "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
    --checkpoint "$CHAMPION" \
    --engine "$ENGINE" \
    --report "$root/games.json" \
    --dataset "$root/rapfi_policy.npz" \
    --ai-loss-dir "$root/pending_training/ai_losses" \
    --pairs "$pairs" \
    --opening-plies 6 \
    --simulations "$simulations" \
    --workers 12 \
    --timeout-turn-ms 300 \
    --max-nodes 100000 \
    --engine-threads 4 \
    --seed "$seed" 2>&1 | tee "$RUN/logs/${name}.log"
  sha256sum "$root/games.json" "$root/rapfi_policy.npz" > "$root/SHA256SUMS"
  touch "$root/complete.done"
}

run_counterfactual &
counter_pid=$!
(
  run_dagger dagger64 3072 64 20280102
  run_dagger dagger128 1024 128 20280103
) &
dagger_pid=$!

wait "$counter_pid"
wait "$dagger_pid"

if [[ ! -f "$RUN/datasets/joint_current_champion.done" ]]; then
  "$PYTHON_BIN" -m alphazero_training.build_rapfi_joint_dataset \
    --report "$SOURCE_REPORT" \
    --report "$RUN/dagger64/games.json" \
    --report "$RUN/dagger128/games.json" \
    --output "$RUN/datasets/joint_current_champion.npz" \
    --symmetries 1 | tee "$RUN/logs/build_joint.log"
  touch "$RUN/datasets/joint_current_champion.done"
fi

"$PYTHON_BIN" - "$RUN" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
reports = [run / "dagger64/games.json", run / "dagger128/games.json"]
report_summaries = []
for path in reports:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert payload.get("complete") is True
    assert summary["complete_pairs"] == summary["requested_pairs"]
    assert summary["errors"] == 0 and summary["truncated"] == 0
    report_summaries.append(summary)

sidecars = [
    run / "datasets/regret_loss_curriculum.npz.json",
    run / "counterfactual/branches.npz.json",
    run / "datasets/joint_current_champion.npz.json",
]
metadata = [json.loads(path.read_text(encoding="utf-8")) for path in sidecars]
for path, item in zip(sidecars, metadata):
    npz_path = path.with_suffix("")
    actual = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    assert actual == item["npz_sha256"], (npz_path, actual, item["npz_sha256"])

payload = {
    "complete": True,
    "champion_sha256": "04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894",
    "dagger_reports": report_summaries,
    "loss_curriculum_samples": metadata[0]["samples"],
    "counterfactual_samples": metadata[1]["samples"],
    "counterfactual_branches": metadata[1]["branches"],
    "joint_samples": metadata[2]["samples"],
    "artifact_sha256": {
        str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            run / "datasets/regret_loss_curriculum.npz",
            run / "counterfactual/branches.npz",
            run / "datasets/joint_current_champion.npz",
        ]
    },
}
(run / "DATA_STAGE_COMPLETE.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

echo "[$(date -Is)] Round6 regret/restart data stage complete"
