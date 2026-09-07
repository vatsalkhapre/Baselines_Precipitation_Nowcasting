#!/bin/bash
# Mandatory gate 1: CIKM smoke test at a tiny step count, per backbone.
BB=$1; GPU=${2:-0}; STEPS=${3:-25}; EP=${4:-5}
cd /home/vatsal/NWM/Baselines_Precipitation_Nowcasting
CUDA_VISIBLE_DEVICES=$GPU /home/vatsal/miniconda3/envs/earthformer/bin/python run_baselines.py \
  --exp_dir smoke --exp_note smoke_${BB} \
  --backbone $BB --dataset cikm \
  --batch_size 4 --seq_len 25 --frames_in 5 --frames_out 10 \
  --epochs $EP --smoke_steps $STEPS \
  --valid --valid_limit --vlnum 3 \
  --num_workers 4 --wandb_state offline \
  --wandb_project_name ICLR26_FACL_runs --run_name smoke_${BB} \
  --results_csv /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/baseline_work/SMOKE_ONLY.csv
