#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"
ENGINE_SHA256="95cea7a20374f8fa2795d4b410b215a7f058c9e2b659d8c0670bcb86761e6f46"
PARENT="alphazero_training/latest.pt"
PARENT_SHA256="04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894"
RUN="rapfi_distillation/round6_regret_restart"
SCREEN="$RUN/guard_screen_seed20280404"
PAIRS=256
SEED=20280404

cd "$ROOT"
mkdir -p "$SCREEN" "$RUN/logs"
[[ "$(sha256sum "$PARENT" | awk '{print $1}')" == "$PARENT_SHA256" ]]
[[ "$(sha256sum "$ENGINE" | awk '{print $1}')" == "$ENGINE_SHA256" ]]

cat > "$RUN/guard_screen_candidates.tsv" <<'EOF'
r6_guard_a010	rapfi_distillation/round6_regret_restart/training/r6_guard_a010_eval.pt
r6_guard_a020	rapfi_distillation/round6_regret_restart/training/r6_guard_a020_eval.pt
r6_guard_a030	rapfi_distillation/round6_regret_restart/training/r6_guard_a030_eval.pt
EOF

run_report() {
  local name="$1" checkpoint="$2"
  local target="$SCREEN/$name"
  mkdir -p "$target"
  if [[ ! -s "$target/complete.done" ]]; then
    "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
      --checkpoint "$checkpoint" --engine "$ENGINE" \
      --report "$target/games.json" \
      --dataset "$target/policy_unused.npz" \
      --ai-loss-dir "$target/pending_training/ai_losses" \
      --pairs "$PAIRS" --opening-plies 6 --simulations 64 --workers 12 \
      --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed "$SEED" \
      2>&1 | tee "$RUN/logs/guard_screen_${name}.log"
    sha256sum "$target/games.json" > "$target/games.json.sha256"
    touch "$target/complete.done"
  fi
}

run_report parent "$PARENT"
while IFS=$'\t' read -r name checkpoint; do
  run_report "$name" "$checkpoint"
  "$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
    --parent-report "$SCREEN/parent/games.json" \
    --candidate-report "$SCREEN/$name/games.json" \
    --output "$SCREEN/${name}_comparison.json"
  sha256sum "$SCREEN/${name}_comparison.json" > "$SCREEN/${name}_comparison.json.sha256"
done < "$RUN/guard_screen_candidates.tsv"

"$PYTHON_BIN" - "$SCREEN" <<'PY'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = {}
for report in sorted(root.glob("*_comparison.json")):
    rows[report.stem.removesuffix("_comparison")] = json.loads(report.read_text())
summary = root / "summary.json"
summary.write_text(json.dumps(rows, indent=2) + "\n")
(summary.with_suffix(summary.suffix + ".sha256")).write_text(
    hashlib.sha256(summary.read_bytes()).hexdigest() + "  " + str(summary) + "\n"
)
print(json.dumps(rows, indent=2))
PY
touch "$SCREEN/complete.done"
