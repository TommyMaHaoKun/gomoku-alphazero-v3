#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
RUN="$ROOT/rapfi_distillation/round7_league_selfplay"
LOG="$RUN/logs/direct_arena_pipeline.log"

cd "$ROOT"
if pgrep -f 'alphazero_training\.paired_model_arena.*direct_arena_seed20280808/report\.json' >/dev/null; then
  echo "Round7 direct arena is already running."
  exit 0
fi
if [[ -f "$RUN/direct_arena_seed20280808/complete.done" ]]; then
  echo "Round7 direct arena is already complete."
  exit 0
fi

nohup bash alphazero_training/run_round7_direct_arena.sh >> "$LOG" 2>&1 < /dev/null &
pid=$!
printf '{"state":"running","pid":%s,"timestamp_utc":"%s"}\n' \
  "$pid" "$(date -u +%FT%TZ)" > "$RUN/direct_arena_seed20280808/resume_state.json.tmp"
mv "$RUN/direct_arena_seed20280808/resume_state.json.tmp" "$RUN/direct_arena_seed20280808/resume_state.json"
echo "Resumed Round7 direct arena as PID $pid."
