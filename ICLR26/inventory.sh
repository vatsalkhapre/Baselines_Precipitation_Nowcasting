#!/bin/bash
# Inventory every trained checkpoint across all three servers -> ICLR26/CHECKPOINTS.md
cd /home/vatsal/NWM/Baselines_Precipitation_Nowcasting
OUT=ICLR26/CHECKPOINTS.md
R='~/NWM/Baselines_Precipitation_Nowcasting'
list_one() {   # $1 = host label, $2 = ssh target ("" = local)
  echo "### $1"
  echo '```'
  local CMD='cd ~/NWM/Baselines_Precipitation_Nowcasting 2>/dev/null || exit 0
    for d in THE_GABOR/checkpoints/*/; do
      n=$(basename "$d"); [ "$n" = "_initial" ] && continue
      b="$d/checkpoints/best_model.pt"; f="$d/checkpoints/final_model.pt"
      [ -f "$b" ] || [ -f "$f" ] || continue
      sz=$(du -sh "$d/checkpoints" 2>/dev/null | cut -f1)
      st=$(python - "$b" <<PY 2>/dev/null
import sys,torch
try:
    c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); print(c.get("step","?"))
except Exception: print("?")
PY
)
      printf "%-52s best_step=%-8s %s\n" "$n" "$st" "$sz"
    done'
  if [ -z "$2" ]; then bash -c "$CMD"; else timeout 300 ssh -o BatchMode=yes "$2" "$CMD" 2>/dev/null; fi
  echo '```'
  echo
}
{
  echo "# Checkpoint inventory"
  echo "_generated $(date '+%F %T')_"
  echo
  echo "Root on every server: \`~/NWM/Baselines_Precipitation_Nowcasting/THE_GABOR/checkpoints/<run_name>/checkpoints/\`"
  echo "Each run directory holds \`initial_model.pt\`, \`best_model.pt\` (best val CSI),"
  echo "\`last_model.pt\`, \`final_model.pt\` and the \`gabor_state*.pt\` files."
  echo
  echo "Naming: \`Stage1_pixel_<ds>_seed<N>\` (stage 1) · \`Stage2_pixel_<ds>_seed<N>\` (2-stage)"
  echo "\`Ablation_pixel_<ds>_<variant>_seed0\` · \`NoStage1_pixel_<ds>_seed0\` (item D)"
  echo "\`PartF_sevir_<target>_gaborinit_<donor>_seed0\` (item F)"
  echo "\`Gabor_pixel_SEVIR_{storm,random}_seed0\` (pre-existing Part-F donors)"
  echo
  echo "Published best models (reference, read-only), on **.205**:"
  echo '```'
  echo "/home/vatsal/Dataserver2/ICLR26/Unaliased_dataset/Best_ckpt_pixel/CIKM/CIKM_pixel_flow22.74_fhigh95.56/"
  echo "/home/vatsal/Dataserver2/ICLR26/Unaliased_dataset/Best_ckpt_pixel/Meteonet/Meteonet_pixel_flow1.09_fhigh1.12/"
  echo "/home/vatsal/Dataserver2/ICLR26/Unaliased_dataset/Best_ckpt_pixel/Shanghai/Shanghai_pixel_flow1.09_fhigh4.43/"
  echo "/home/vatsal/Dataserver2/ICLR26/Unaliased_dataset/Best_ckpt_pixel/SEVIR/dawncast_sevir_pixel/"
  echo '```'
  echo
  list_one ".88 (questlab-shell)" ""
  list_one ".66 (resiliente-2091)" "vatsal@10.24.52.66"
  list_one ".205 (questlab)" "vatsal@10.24.52.205"
} > $OUT
rsync -a $OUT vatsal@10.24.52.66:~/NWM/Baselines_Precipitation_Nowcasting/ICLR26/ 2>/dev/null
echo "wrote $OUT"
