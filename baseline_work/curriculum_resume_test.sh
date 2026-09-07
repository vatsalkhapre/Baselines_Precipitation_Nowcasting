#!/bin/bash
# Verifies the FALFCL curriculum is reconstructed on resume for an OLD-format
# checkpoint, i.e. one written by a model file that does NOT persist facl_step_buf.
# traj_gru_falfcl (v1) is exactly such a file.
BB=traj_gru_falfcl; GPU=${1:-1}
R=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting
EXP=$R/Exps/curriculum/curr_${BB}
PY=/home/vatsal/miniconda3/envs/earthformer/bin/python
rm -rf $EXP
run() { CUDA_VISIBLE_DEVICES=$GPU $PY $R/run_baselines.py \
  --exp_dir curriculum --exp_note curr_${BB} --backbone $BB --dataset cikm \
  --batch_size 4 --seq_len 25 --frames_in 5 --frames_out 10 \
  --epochs 20 --smoke_steps 20 --valid --valid_limit --vlnum 2 \
  --num_workers 4 --wandb_state offline --wandb_project_name ICLR26_FACL_runs \
  --run_name curr_${BB} --results_csv $R/baseline_work/SMOKE_ONLY.csv "$@"; }
echo "--- phase 1 ---"; run > /tmp/curr_p1.log 2>&1 &
P=$!; for i in $(seq 1 60); do sleep 5; ls $EXP/checkpoints/*.pt >/dev/null 2>&1 && break; done
sleep 10; kill -9 $P 2>/dev/null; sleep 3
CK=$(ls -t $EXP/checkpoints/*.pt 2>/dev/null | head -1)
[ -z "$CK" ] && { echo "FAIL: no checkpoint"; exit 1; }
echo "resuming from $(basename $CK)"
$PY -c "
import torch;d=torch.load('$CK',map_location='cpu',weights_only=False)
print('  checkpoint step =',d['step'])
print('  has facl_step_buf in state_dict?', any('facl_step_buf' in k for k in d['model']))"
echo "--- phase 2 (resume) ---"; run --ckpt_milestone $CK --res_opt > /tmp/curr_p2.log 2>&1
grep -E "Seeded loss-curriculum|Loss curriculum:" /tmp/curr_p2.log | head -2
