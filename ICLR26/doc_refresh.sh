#!/bin/bash
# Keep ICLR26/PROGRESS.md's live-status section current without hand-editing:
# regenerates an appended STATUS block every 30 min from the actual logs.
cd /home/vatsal/NWM/Baselines_Precipitation_Nowcasting
OUT=ICLR26/STATUS_LIVE.md
{
  echo "# Live status — regenerated automatically"
  echo "_$(date '+%F %T')_"
  echo
  for H in 10.24.52.88:.88 10.24.52.66:.66 10.24.52.205:.205; do
    IP=${H%%:*}; L=${H##*:}
    echo "## $L"
    if [ "$IP" = "10.24.52.88" ]; then
      timeout 20 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/    GPU /'
      for f in THE_GABOR/logs/_runlogs/{Stage1,Stage2,Ablation,NoStage1,PartF}_*.log; do
        [ -f "$f" ] || continue
        printf '    %-46s %-13s %s\n' "$(basename "$f" .log)" \
          "$(tr '\r' '\n' < "$f" | grep -oE 'epoch [0-9]+/50' | tail -1)" \
          "$(grep -q '^\[done\]' "$f" && echo DONE || echo running)"
      done
    else
      timeout 45 ssh -o BatchMode=yes -o ConnectTimeout=10 vatsal@$IP \
        "cd ~/NWM/Baselines_Precipitation_Nowcasting
         timeout 20 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/    GPU /'
         for f in THE_GABOR/logs/_runlogs/{Stage1,Stage2,Ablation,NoStage1,PartF}_*.log; do
           [ -f \"\$f\" ] || continue
           printf '    %-46s %-13s %s\n' \"\$(basename \$f .log)\" \
             \"\$(tr '\r' '\n' < \$f | grep -oE 'epoch [0-9]+/50' | tail -1)\" \
             \"\$(grep -q '^\[done\]' \$f && echo DONE || echo running)\"
         done" 2>/dev/null
    fi
    echo
  done
  echo "## evaluations completed"
  ls ICLR26/eval/*.json 2>/dev/null | wc -l | sed 's/^/    local jsons: /'
} > $OUT 2>/dev/null
rsync -a $OUT vatsal@10.24.52.66:~/NWM/Baselines_Precipitation_Nowcasting/ICLR26/ 2>/dev/null
