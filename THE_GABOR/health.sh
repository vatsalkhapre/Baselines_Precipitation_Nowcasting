#!/bin/bash
# One-line-per-GPU health snapshot + per-run status. Written every 30 min by cron
# so a stall is obvious without digging through logs.
cd "$(dirname "$0")/.."
OUT=THE_GABOR/logs/_runlogs/HEALTH.txt
{
  echo "=== $(hostname) $(date '+%F %T')  uptime:$(uptime -p 2>/dev/null) ==="
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    | awk '{printf "GPU %s\n", $0}'
  echo "chains alive: $(pgrep -cf 'orchestrate.sh' )"
  for f in THE_GABOR/logs/_runlogs/{Stage1,Stage2,Ablation,NoStage1,PartF}_*.log; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .log)
    ep=$(tr '\r' '\n' < "$f" | grep -oE 'epoch [0-9]+/[0-9]+' | tail -1)
    if grep -q '^\[done\]' "$f"; then st=DONE
    elif [ -n "$(find "$f" -mmin -20 2>/dev/null)" ]; then st=running
    else st=STALLED; fi
    printf "%-46s %-14s %s\n" "$n" "$ep" "$st"
  done
  ls THE_GABOR/logs/_runlogs/_failed/*.log >/dev/null 2>&1 && {
    echo "--- archived failures ---"; ls -1 THE_GABOR/logs/_runlogs/_failed/ | tail -5; }
} > $OUT 2>/dev/null
