# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
#     --exp_dir meteonet \
#     --exp_note Simvp_on_meteonet \
#     --batch_size 16 \
#     --backbone simvp \
#     --dataset meteo \
#     --seq_len 25 \
#     --epochs 50 \
#     --valid \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "simvp_meteonet" 


# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
#     --exp_dir shanghai \
#     --exp_note Simvp_on_shanghai \
#     --batch_size 16 \
#     --backbone simvp \
#     --dataset shanghai \
#     --seq_len 25 \
#     --valid \
#     --epochs 50 \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "simvp_shanghai" 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
    --exp_dir cikm \
    --exp_note Simvp_on_cikm \
    --batch_size 16 \
    --backbone simvp \
    --dataset cikm \
    --seq_len 15 \
    --valid \
    --epochs 70 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "simvp_cikm" 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
    --exp_dir cikm \
    --exp_note Simvp_on_cikm \
    --batch_size 16 \
    --backbone simvp \
    --dataset cikm \
    --seq_len 15 \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "simvp_cikm" 