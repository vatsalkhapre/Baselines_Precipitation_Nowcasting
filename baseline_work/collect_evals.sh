#!/bin/bash
# The DAWN-Cast eval outputs land in each host's LOCAL Exps/dawncast_eval
# (only Exps/baselines_falfcl is symlinked to shared storage), so they have to
# be gathered before the table generator can see them.
R=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting
S=/home/vatsal/Dataserver2/Neurips/dawncast_eval_collected
mkdir -p $S
for h in 66 205; do
  for d in $(ssh -o BatchMode=yes -n vatsal@10.24.52.$h "ls -d $R/Exps/dawncast_eval/*/ 2>/dev/null"); do
    n=$(basename $d)
    if ssh -o BatchMode=yes -n vatsal@10.24.52.$h "grep -q 'Test Results:' $d/logs/log.log 2>/dev/null"; then
      mkdir -p $S/$n/logs
      scp -q vatsal@10.24.52.$h:$d/logs/log.log $S/$n/logs/log.log
      echo "  collected $n from .$h"
    else
      echo "  SKIP $n on .$h (no Test Results yet)"
    fi
  done
done
