#!/bin/bash
# One top-up: resume a completed FALFCL run to the uniform epoch budget.
#   usage: topup_run.sh <dataset_dir> <run_name> <target_epochs> <frames_out> <gpu>
# e.g.   topup_run.sh cikm_falfcl mau_on_cikm 80 10 1
#
# Safeguards (see topup_existing_runs.py for why each exists):
#  * ckpt-best.pt is copied to ckpt-best.pre-topup.pt BEFORE anything runs.
#    Old checkpoints carry no max_csi, so the first validation after resume
#    beats max_csi=0.0 and overwrites ckpt-best.pt with a WORSE checkpoint.
#    The caller compares afterwards and restores if the top-up did not improve.
#  * --res_opt is mandatory: without it the optimiser and LR schedule restart.
#  * The FALFCL curriculum position is reconstructed by run_baselines.py from
#    the checkpoint's step field (owner-approved option b), so the extra epochs
#    continue the loss schedule instead of restarting it at pure-FCL.
set -u
DSDIR=$1; RUN=$2; EP=$3; FOUT=$4; GPU=$5
R=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting
M=/home/vatsal/Dataserver2/Neurips/Models_falfcl
PY=/home/vatsal/miniconda3/envs/earthformer/bin/python
D=$M/$DSDIR/$RUN
case "$DSDIR" in cikm*) DS=cikm;; sevir*) DS=sevir;; shanghai*) DS=shanghai;; meteo*) DS=meteo;; *) echo "unknown dataset dir"; exit 1;; esac
BB=$($PY -c "import yaml;print(yaml.safe_load(open('$D/params.yaml'))['backbone'])" 2>/dev/null)
[ -z "$BB" ] && { echo "FAIL $RUN: cannot read backbone from params.yaml"; exit 1; }
[ -f "$D/checkpoints/ckpt-last.pt" ] || { echo "FAIL $RUN: no ckpt-last.pt"; exit 1; }
# 1. preserve the pre-top-up best
cp -n $D/checkpoints/ckpt-best.pt $D/checkpoints/ckpt-best.pre-topup.pt 2>/dev/null
# 2. expose the run under Exps/ so the runner writes back into the same directory
ln -sfn $M/$DSDIR $R/Exps/topup_$DSDIR
echo "TOPUP $RUN backbone=$BB dataset=$DS -> $EP epochs on gpu$GPU"
CUDA_VISIBLE_DEVICES=$GPU $PY $R/run_baselines.py \
  --exp_dir topup_$DSDIR --exp_note $RUN \
  --backbone $BB --dataset $DS --batch_size 4 --seq_len 25 \
  --frames_in 5 --frames_out $FOUT --img_size 128 --epochs $EP \
  --valid --num_workers 8 --wandb_state online \
  --wandb_project_name ICLR26_FACL_runs --run_name topup_$RUN \
  --ckpt_milestone $D/checkpoints/ckpt-last.pt --res_opt \
  --results_csv /home/vatsal/Dataserver2/Neurips/csv_files/models_falfcl.csv
