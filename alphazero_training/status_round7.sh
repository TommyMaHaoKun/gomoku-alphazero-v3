#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOMOKU_ROOT:-/root/autodl-tmp/gomoku_v3}"
RUN="$ROOT/rapfi_distillation/round7_league_selfplay"
STATE="$RUN/direct_arena_seed20280808"

cd "$ROOT"
if pgrep -f 'alphazero_training\.paired_model_arena.*direct_arena_seed20280808/report\.json' >/dev/null; then
  state="running"
elif [[ -f "$STATE/complete.done" ]]; then
  state="complete"
else
  state="paused_or_stopped"
fi
lines=0
if [[ -f "$STATE/report.jsonl" ]]; then
  lines="$(wc -l < "$STATE/report.jsonl")"
fi
printf 'state=%s committed_pairs=%s/1024 committed_games=%s\n' \
  "$state" "$((lines / 2))" "$lines"
sha256sum alphazero_training/latest.pt
