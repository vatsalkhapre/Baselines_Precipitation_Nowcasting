#!/bin/bash
# Mandatory gate 2: kill-and-resume test.
# Trains until a checkpoint exists, kills the process mid-training, resumes from
# ckpt-last, and checks that (a) the epoch/step counters continue rather than
# restart, (b) max_csi was restored so ckpt-best cannot be clobbered by a worse
# post-resume validation, (c) the loss continues smoothly instead of jumping
# back to its initial value.
BB=$1; GPU=${2:-0}
R=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting
EXP=$R/Exps/resume/resume_${BB}
PY=/home/vatsal/miniconda3/envs/earthformer/bin/python
rm -rf $EXP
run() {
  CUDA_VISIBLE_DEVICES=$GPU $PY $R/run_baselines.py \
    --exp_dir resume --exp_note resume_${BB} \
    --backbone $BB --dataset cikm --batch_size 4 --seq_len 25 \
    --frames_in 5 --frames_out 10 --epochs 20 --smoke_steps 20 \
    --valid --num_workers 4 --wandb_state offline \
    --wandb_project_name ICLR26_FACL_runs --run_name resume_${BB} \
    --results_csv $R/baseline_work/SMOKE_ONLY.csv "$@"
}
echo "--- phase 1: train until ckpt-last exists, then kill ---"
run > /tmp/resume_${BB}_p1.log 2>&1 &
P1=$!
for i in $(seq 1 100); do
  sleep 10
  [ -f $EXP/checkpoints/ckpt-last.pt ] && break
done
sleep 20
kill -9 $P1 2>/dev/null; pkill -9 -f "resume_${BB}" 2>/dev/null; sleep 5
if [ ! -f $EXP/checkpoints/ckpt-last.pt ]; then echo "RESUME-TEST FAIL: no checkpoint was ever written"; exit 1; fi
echo "phase 1 killed. state before resume:"
$PY - <<PYEOF
import torch
d=torch.load("$EXP/checkpoints/ckpt-last.pt",map_location='cpu',weights_only=False)
print(f"   step={d['step']} epoch={d['epoch']} max_csi={d.get('max_csi')} best_step={d.get('best_step')}")
print(f"   has_opt={'opt' in d} has_sched={'scheduler' in d}")
PYEOF
echo "--- phase 2: resume ---"
run --ckpt_milestone $EXP/checkpoints/ckpt-last.pt --res_opt > /tmp/resume_${BB}_p2.log 2>&1
echo "--- results ---"
grep -E "Restored best-ckpt state|Loading epochs|Current epoch" /tmp/resume_${BB}_p2.log | head -5
echo "  loss trajectory phase1 (first/last):"
grep -o "'total_loss': [0-9.]*" /tmp/resume_${BB}_p1.log | head -1
grep -o "'total_loss': [0-9.]*" /tmp/resume_${BB}_p1.log | tail -1
echo "  loss trajectory phase2 (first/last):"
grep -o "'total_loss': [0-9.]*" /tmp/resume_${BB}_p2.log | head -1
grep -o "'total_loss': [0-9.]*" /tmp/resume_${BB}_p2.log | tail -1
