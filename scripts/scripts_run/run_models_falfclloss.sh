# CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
#     --exp_dir shanghai_falfcl \
#     --exp_note earthformer_on_shanghai \
#     --batch_size 4 \
#     --backbone earthformer_falfcl \
#     --dataset shanghai \
#     --seq_len 25 \
#     --valid \
#     --epochs 70 \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "earthformer_shanghai_falfcl" 

CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir shanghai_falfcl \
    --exp_note earthformer_on_shanghai \
    --batch_size 4 \
    --backbone earthformer_falfcl \
    --dataset shanghai \
    --seq_len 25 \
    --eval \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "earthformer_shanghai_falfcl" 


CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir shanghai_falfcl \
    --exp_note simvp_on_shanghai \
    --batch_size 4 \
    --backbone simvp_falfcl \
    --dataset shanghai \
    --seq_len 25 \
    --valid \
    --epochs 70 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "simvp_shanghai_falfcl" 

CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir shanghai_falfcl \
    --exp_note simvp_on_shanghai \
    --batch_size 4 \
    --backbone simvp_falfcl \
    --dataset shanghai \
    --seq_len 25 \
    --eval \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "simvp_shanghai_falfcl" 

CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir cikm_falfcl \
    --exp_note simvp_on_cikm \
    --batch_size 8 \
    --backbone simvp_falfcl \
    --dataset cikm \
    --seq_len 15 \
    --epochs 70 \
    --valid \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "simvp_cikm_falfcl" 

CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir cikm_falfcl \
    --exp_note simvp_on_cikm \
    --batch_size 8 \
    --backbone simvp_falfcl \
    --dataset cikm \
    --seq_len 15 \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' 
