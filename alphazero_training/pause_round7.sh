#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
RUN="$ROOT/rapfi_distillation/round7_league_selfplay"
STATE="$RUN/direct_arena_seed20280808"

cd "$ROOT"
mapfile -t pids < <(pgrep -f 'alphazero_training\.paired_model_arena.*direct_arena_seed20280808/report\.json' || true)
if ((${#pids[@]} == 0)); then
  echo "Round7 direct arena is not running. Committed progress remains intact."
  exit 0
fi

kill -TERM "${pids[@]}"
for _ in {1..30}; do
  if ! kill -0 "${pids[0]}" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "${pids[0]}" 2>/dev/null; then
  echo "Process is finishing its current operation; no forced kill was issued."
  exit 1
fi

lines=0
if [[ -f "$STATE/report.jsonl" ]]; then
  lines="$(wc -l < "$STATE/report.jsonl")"
fi
printf '{"state":"paused","committed_games":%s,"committed_pairs":%s,"timestamp_utc":"%s"}\n' \
  "$lines" "$((lines / 2))" "$(date -u +%FT%TZ)" > "$STATE/pause_state.json.tmp"
mv "$STATE/pause_state.json.tmp" "$STATE/pause_state.json"
echo "Paused safely at $((lines / 2))/1024 committed pairs."
