CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir meteo_falfcl \
    --exp_note mau_on_meteo \
    --batch_size 4 \
    --backbone mau_falfcl \
    --dataset meteo \
    --seq_len 25 \
    --valid \
    --epochs 50 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "mau_meteo_falfcl" 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir meteo_falfcl \
    --exp_note mau_on_meteo \
    --batch_size 4 \
    --backbone mau_falfcl \
    --dataset meteo \
    --seq_len 25 \
    --eval \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "mau_meteo_falfcl" 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir meteo_falfcl \
    --exp_note alphapre_on_meteo \
    --batch_size 4 \
    --backbone alphapre_falfcl \
    --dataset meteo \
    --seq_len 25 \
    --valid \
    --epochs 50 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "alphapre_meteo_falfcl" 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir meteo_falfcl \
    --exp_note alphapre_on_meteo \
    --batch_size 4 \
    --backbone alphapre_falfcl \
    --dataset meteo \
    --seq_len 25 \
    --eval \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "alphapre_meteo_falfcl" 