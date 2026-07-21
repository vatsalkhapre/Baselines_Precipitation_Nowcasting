CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir meteo_falfcl \
    --exp_note earthformer_on_meteonet \
    --batch_size 4 \
    --backbone earthformer_falfcl \
    --dataset meteo \
    --seq_len 25 \
    --valid \
    --epochs 50 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "earthformer_meteo_falfcl" 

CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir meteo_falfcl \
    --exp_note earthformer_on_meteo \
    --batch_size 4 \
    --backbone earthformer_falfcl \
    --dataset meteo \
    --seq_len 25 \
    --eval \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "earthformer_meteo_falfcl" 




CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir meteo_falfcl \
    --exp_note simvp_on_meteo \
    --batch_size 4 \
    --backbone simvp_falfcl \
    --dataset meteo \
    --seq_len 25 \
    --valid \
    --epochs 50 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "simvp_meteo_falfcl" 

CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir meteo_falfcl \
    --exp_note simvp_on_meteo \
    --batch_size 4 \
    --backbone simvp_falfcl \
    --dataset meteo \
    --seq_len 25 \
    --eval \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "simvp_meteo_falfcl" 

CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir meteo_falfcl \
    --exp_note trajgru_on_meteo \
    --batch_size 4 \
    --backbone traj_gru_falfcl \
    --dataset meteo \
    --seq_len 25 \
    --valid \
    --epochs 50 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "trajgru_meteo_falfcl" 

CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir meteo_falfcl \
    --exp_note trajgru_on_meteo \
    --batch_size 4 \
    --backbone traj_gru_falfcl \
    --dataset meteo \
    --seq_len 25 \
    --eval \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "trajgru_meteo_falfcl" 


CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir sevir_falfcl \
    --exp_note trajgru_on_sevir \
    --batch_size 4 \
    --backbone traj_gru_falfcl \
    --dataset sevir \
    --seq_len 25 \
    --epochs 20 \
    --valid \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "trajgru_sevir_falfcl" 


CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir sevir_falfcl \
    --exp_note simvp_on_sevir \
    --batch_size 4 \
    --backbone simvp_falfcl \
    --dataset sevir \
    --seq_len 25 \
    --epochs 20 \
    --valid \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "simvp_sevir_falfcl" 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir sevir_falfcl \
    --exp_note earthformer_on_sevir \
    --batch_size 4 \
    --backbone earthformer_falfcl \
    --dataset sevir \
    --seq_len 25 \
    --epochs 20 \
    --valid \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "earthformer_sevir_falfcl" 