#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
ENGINE="/root/autodl-tmp/teachers/rapfi-runtime/pbrain-rapfi"
ENGINE_SHA256="95cea7a20374f8fa2795d4b410b215a7f058c9e2b659d8c0670bcb86761e6f46"
RUN="rapfi_distillation/round6_regret_restart"
SCREEN="$RUN/value_screen_seed20280404"
PARENT_REPORT="$RUN/guard_screen_seed20280404/parent/games.json"
PAIRS=256
SEED=20280404

cd "$ROOT"
mkdir -p "$SCREEN" "$RUN/logs"
[[ -s "$PARENT_REPORT" ]]
[[ "$(sha256sum "$ENGINE" | awk '{print $1}')" == "$ENGINE_SHA256" ]]

cat > "$RUN/value_screen_candidates.tsv" <<'EOF'
r6_value_a025	rapfi_distillation/round6_regret_restart/training/r6_value_a025_eval.pt
r6_value_a050	rapfi_distillation/round6_regret_restart/training/r6_value_a050_eval.pt
r6_value_a075	rapfi_distillation/round6_regret_restart/training/r6_value_a075_eval.pt
r6_value_full	rapfi_distillation/round6_regret_restart/training/r6_value_repair_s8000_eval.pt
EOF

while IFS=$'\t' read -r name checkpoint; do
  target="$SCREEN/$name"
  mkdir -p "$target"
  if [[ ! -s "$target/complete.done" ]]; then
    "$PYTHON_BIN" -u -m alphazero_training.rapfi_distill \
      --checkpoint "$checkpoint" --engine "$ENGINE" \
      --report "$target/games.json" --dataset "$target/policy_unused.npz" \
      --ai-loss-dir "$target/pending_training/ai_losses" \
      --pairs "$PAIRS" --opening-plies 6 --simulations 64 --workers 12 \
      --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4 --seed "$SEED" \
      2>&1 | tee "$RUN/logs/value_screen_${name}.log"
    sha256sum "$target/games.json" > "$target/games.json.sha256"
    touch "$target/complete.done"
  fi
  "$PYTHON_BIN" -m alphazero_training.compare_rapfi_reports \
    --parent-report "$PARENT_REPORT" --candidate-report "$target/games.json" \
    --output "$SCREEN/${name}_comparison.json"
  sha256sum "$SCREEN/${name}_comparison.json" > "$SCREEN/${name}_comparison.json.sha256"
done < "$RUN/value_screen_candidates.tsv"

"$PYTHON_BIN" - "$SCREEN" <<'PY'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1]); rows = {}
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
