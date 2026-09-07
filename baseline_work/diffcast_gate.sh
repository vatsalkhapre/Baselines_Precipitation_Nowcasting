#!/bin/bash
# DiffCast+FACL mandatory gates: bounded smoke, then kill-and-resume.
GPU=${1:-0}
R=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting
PY=/home/vatsal/miniconda3/envs/earthformer/bin/python
EXP=$R/Exps/diffgate/Diffphydnet_falfcl_cikm_diffgate
rm -rf $R/Exps/diffgate
run(){ CUDA_VISIBLE_DEVICES=$GPU $PY $R/run_diffcast_falfcl.py \
  --exp_dir diffgate --exp_note diffgate --backbone phydnet_falfcl --use_diff \
  --dataset cikm --batch_size 4 --seq_len 25 --frames_in 5 --frames_out 10 \
  --img_size 128 --epochs 20 --smoke_steps 12 --valid --valid_limit --vlnum 2 \
  --num_workers 4 --wandb_state offline "$@"; }
echo "--- phase 1: train until a checkpoint exists, then kill ---"
run > /tmp/diffgate_p1.log 2>&1 &
P=$!
for i in $(seq 1 90); do sleep 10; ls $EXP/checkpoints/*.pt >/dev/null 2>&1 && break; done
sleep 10; kill -9 $P 2>/dev/null; pkill -9 -f "exp_note diffgate" 2>/dev/null; sleep 4
CK=$(ls -t $EXP/checkpoints/*.pt 2>/dev/null | head -1)
[ -z "$CK" ] && { echo "GATE FAIL: no checkpoint written"; tail -5 /tmp/diffgate_p1.log; exit 1; }
echo "checkpoint: $(basename $CK)"
$PY -c "
import torch; d=torch.load('$CK',map_location='cpu',weights_only=False)
print(f'  step={d[\"step\"]} epoch={d[\"epoch\"]} max_csi={d.get(\"max_csi\")}')"
echo "--- phase 2: resume ---"
run --ckpt_milestone $CK --res_opt > /tmp/diffgate_p2.log 2>&1
echo "--- gate results ---"
grep -E "Restored best-ckpt state|Seeded loss-curriculum|Loss curriculum:" /tmp/diffgate_p2.log | head -3
echo "  p1 epoch losses:"; grep -oE "Epoch [0-9]+ avg train loss: [0-9.]+" /tmp/diffgate_p1.log | tail -3
echo "  p2 epoch losses:"; grep -oE "Epoch [0-9]+ avg train loss: [0-9.]+" /tmp/diffgate_p2.log | head -3
echo "  tracebacks p1=$(grep -c Traceback /tmp/diffgate_p1.log) p2=$(grep -c Traceback /tmp/diffgate_p2.log)"
echo DIFFCAST_GATES_DONE
