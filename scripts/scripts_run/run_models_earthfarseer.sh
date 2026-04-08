CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir sevir \
    --exp_note earthfarseer_on_sevir \
    --batch_size 4 \
    --backbone earthfarseer \
    --dataset sevir \
    --seq_len 25 \
    --valid \
    --epochs 40 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "earthfarseer_sevir" 

# CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
#     --exp_dir sevir \
#     --exp_note earthfarseer_on_sevir \
#     --batch_size 8 \
#     --backbone earthfarseer \
#     --dataset sevir \
#     --seq_len 25 \
#     --valid \
#     --epochs 40 \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "earthfarseer_sevir" 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir shanghai \
    --exp_note earthfarseer_on_shanghai \
    --batch_size 4 \
    --backbone earthfarseer \
    --dataset shanghai \
    --seq_len 25 \
    --valid \
    --epochs 70 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "earthfarseer_shanghai" 


# CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
#     --exp_dir meteonet \
#     --exp_note earthfarseer_on_meteonet \
#     --batch_size 8 \
#     --backbone earthfarseer \
#     --dataset meteo \
#     --seq_len 25 \
#     --valid \
#     --epochs 50 \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "earthfarseer_meteonet" 

# CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
#     --exp_dir cikm \
#     --exp_note earthfarseer_on_cikm \
#     --batch_size 8 \
#     --backbone earthfarseer \
#     --dataset cikm \
#     --seq_len 15 \
#     --valid \
#     --epochs 70 \
#     --frames_in 5 \
#     --frames_out 10 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "earthfarseer_cikm" 