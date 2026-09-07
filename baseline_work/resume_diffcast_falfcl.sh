#!/bin/bash
# Resume a DiffCast+FACL run later, when GPUs free up.
#
#   usage: resume_diffcast_falfcl.sh <dataset> <gpu> <new_total_epochs>
#   e.g.   resume_diffcast_falfcl.sh sevir 0 45
#
# WHAT THIS DOES DIFFERENTLY FROM A FRESH LAUNCH
#  * --ckpt_milestone ckpt-last.pt + --res_opt: without --res_opt the optimizer
#    and LR schedule silently restart from step 0.
#  * <new_total_epochs> is the TOTAL, not an increment. Passing a larger value
#    rebuilds the cosine LR schedule and the FACL curriculum over the longer
#    horizon rather than continuing a schedule sized for the original budget --
#    a schedule that never finishes decaying is a different training regime.
#  * ckpt-best.pt is copied aside first. If the extension does not beat the
#    existing best, restore it from the .pre-resume copy.
set -u
DS=$1; GPU=$2; EP=$3
R=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting
PY=/home/vatsal/miniconda3/envs/earthformer/bin/python
D=$R/Exps/diffcast_falfcl/diffcast_falfcl_on_$DS
FO=20; [ "$DS" = "cikm" ] && FO=10
CK=$D/checkpoints/ckpt-last.pt
[ -f "$CK" ] || { echo "FAIL: no checkpoint at $CK"; exit 1; }
cp -n $D/checkpoints/ckpt-best.pt $D/checkpoints/ckpt-best.pre-resume.pt 2>/dev/null
$PY - <<PY
import torch; d=torch.load("$CK",map_location='cpu',weights_only=False)
print(f"resuming $DS from epoch {d['epoch']+1}, step {d['step']}, max_csi={d.get('max_csi')}")
PY
exec env CUDA_VISIBLE_DEVICES=$GPU $PY $R/run_diffcast_falfcl.py \
  --exp_dir diffcast_falfcl --exp_note diffcast_falfcl_on_$DS \
  --backbone phydnet_falfcl --use_diff --dataset $DS \
  --batch_size 4 --seq_len 25 --frames_in 5 --frames_out $FO --img_size 128 \
  --lr 1e-4 --epochs $EP --valid --num_workers 8 \
  --ckpt_milestone $CK --res_opt \
  --wandb_state online --wandb_project_name ICLR26_FACL_runs \
  --run_name diffcast_falfcl_${DS}_resume${EP}
