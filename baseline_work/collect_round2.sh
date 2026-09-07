#!/bin/bash
# round2/ and diffcast_falfcl/ live in each host's LOCAL Exps/ (only
# Exps/baselines_falfcl is symlinked to shared storage), so their logs must be
# gathered before the table generator can see them. Only runs that actually
# produced a TEST evaluation are collected.
R=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting
S=/home/vatsal/Dataserver2/Neurips/round2_collected
mkdir -p $S
for h in 88 66 205; do
  for sub in round2 diffcast_falfcl; do
    for d in $(ssh -o BatchMode=yes -n vatsal@10.24.52.$h "ls -d $R/Exps/$sub/*/ 2>/dev/null"); do
      n=$(basename $d)
      if ssh -o BatchMode=yes -n vatsal@10.24.52.$h "grep -q 'Test Results:' $d/logs/log.log 2>/dev/null"; then
        mkdir -p "$S/$n/logs"
        scp -q vatsal@10.24.52.$h:$d/logs/log.log "$S/$n/logs/log.log"
        echo "  collected $n (.$h)"
      fi
    done
  done
done
