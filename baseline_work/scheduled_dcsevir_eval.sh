#!/bin/bash
# Deferred DiffCast-SEVIR test evaluation.
#   usage: scheduled_dcsevir_eval.sh "YYYY-MM-DD HH:MM" <gpu>
# Sleeps until the target time, then runs the eval. Detached, so it survives
# the launching session. ~10.5 h of diffusion sampling over 1013 SEVIR samples.
set -u
TARGET="$1"; GPU="$2"
R=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting
PY=/home/vatsal/miniconda3/envs/earthformer/bin/python
NOTE=Diffphydnet_falfcl_sevir_diffcast_falfcl_on_sevir

now=$(date +%s); then_=$(date -d "$TARGET" +%s)
wait=$(( then_ - now ))
echo "$(date '+%F %T') scheduled for $TARGET (sleeping ${wait}s)"
[ "$wait" -gt 0 ] && sleep "$wait"

# don't stomp on whatever else may have claimed the card in the meantime
busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F, -v g="$GPU" '$1+0==g{print $2}' | tr -d ' ')
if echo "$busy" | grep -q "$uuid"; then
  echo "$(date '+%F %T') WARNING: gpu$GPU is busy; starting anyway would contend. Aborting."
  exit 1
fi

echo "$(date '+%F %T') launching DiffCast-SEVIR test eval on gpu$GPU"
cd $R && $PY baseline_work/eval_round2.py diffcast_falfcl $NOTE $GPU
echo "$(date '+%F %T') DCSEVIR_EVAL_FINISHED rc=$?"
