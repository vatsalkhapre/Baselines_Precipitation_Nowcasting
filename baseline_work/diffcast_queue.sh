#!/bin/bash
# Launch the remaining DiffCast+FACL cells as GPUs free up.
# Polls for a GPU with NO compute process (free memory alone races a starting
# job) and that is not listed in reserved_gpus.txt.
set -u
R=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting
RES=/home/vatsal/Dataserver2/Neurips/baseline_manifest/reserved_gpus.txt
QUEUE="meteo cikm shanghai"      # sevir already launched (longest, started first)
HOSTS="10.24.52.66 10.24.52.205 10.24.52.88"

free_gpu_on() {   # echoes "host gpu" for the first genuinely idle, unreserved GPU
  local h=$1
  local reserved busy
  reserved=$(grep -E "^$h " $RES 2>/dev/null | awk '{print $2}' | tr '\n' ' ')
  busy=$(ssh -o BatchMode=yes -o ConnectTimeout=10 -n vatsal@$h \
        "nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u" 2>/dev/null)
  ssh -o BatchMode=yes -o ConnectTimeout=10 -n vatsal@$h \
      "nvidia-smi --query-gpu=index,uuid --format=csv,noheader" 2>/dev/null | while IFS=, read -r i u; do
        i=$(echo $i | tr -d ' '); u=$(echo $u | tr -d ' ')
        echo "$reserved" | grep -qw "$i" && continue
        echo "$busy" | grep -q "$u" && continue
        echo "$i"; break
      done
}

for ds in $QUEUE; do
  while true; do
    for h in $HOSTS; do
      g=$(free_gpu_on $h | head -1)
      if [ -n "$g" ]; then
        echo "$(date +%H:%M:%S) LAUNCH diffcast_falfcl $ds -> $h gpu$g"
        ssh -o BatchMode=yes -n vatsal@$h \
          "cd $R && (setsid nohup bash baseline_work/launch_diffcast_falfcl.sh $ds $g > /tmp/dc_$ds.log 2>&1 < /dev/null &)"
        sleep 90     # let it allocate before the next probe sees the card as free
        break 2
      fi
    done
    sleep 120
  done
done
echo ALL_DIFFCAST_LAUNCHED
