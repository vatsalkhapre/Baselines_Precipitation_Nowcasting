#!/bin/bash
# Round 2 launches. Each cell states WHY its settings differ from round 1, so the
# baseline treatment is auditable rather than incidental.
#   usage: launch_round2.sh <cell> <gpu>
set -u
R=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting
PY=/home/vatsal/miniconda3/envs/earthformer/bin/python
CSV=/home/vatsal/Dataserver2/Neurips/csv_files/models_falfcl.csv
CELL=$1; GPU=$2
common="--seq_len 25 --img_size 128 --frames_in 5 --num_workers 8 --valid \
 --wandb_state online --wandb_project_name ICLR26_FACL_runs --results_csv $CSV"

case $CELL in
  # ---- exPreCast: ONLY change vs round 1 is drop_path_rate 0 -> 0.2, the
  #      official default (exPreCast/model.py:600). bs=16 and lr=1e-3 are the
  #      paper's own values and are preserved. Epochs 200 -> 50 because every
  #      dataset peaked by epoch 30 and then declined for 170 epochs.
  exprecast_*)
    ds=${CELL#exprecast_}; fo=20; [ "$ds" = "cikm" ] && fo=10
    exec env CUDA_VISIBLE_DEVICES=$GPU $PY $R/run_baselines.py \
      --exp_dir round2 --exp_note $CELL --backbone exPreCast --dataset $ds \
      --frames_out $fo --batch_size 16 --lr 1e-3 --epochs 50 \
      --drop_path_rate 0.2 --embed_dim 96 --depths 2,6,2,2 --num_heads 3,6,12,24 \
      --skip_connection add --run_name $CELL $common ;;

  # ---- WADEPre Shanghai: refine_hidden_dim 480 -> 560 so it matches its three
  #      siblings (30.4M params, not 25.0M). Native loss and curriculum
  #      unchanged -- WADEPre is not a FACL model. 100 epochs; the original ran
  #      150 and peaked at 49.
  wadepre_shanghai)
    exec env CUDA_VISIBLE_DEVICES=$GPU $PY $R/run_baselines.py \
      --exp_dir round2 --exp_note wadepre_on_shanghai --backbone wadepre \
      --dataset shanghai --frames_out 20 --batch_size 4 --lr 1.5e-4 --epochs 100 \
      --refine_hidden_dim 576 --run_name wadepre_shanghai_rhd560 $common ;;

  *) echo "unknown cell $CELL"; exit 1 ;;
esac
