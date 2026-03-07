CUDA_VISIBLE_DEVICES=0 python3 run_diffcast.py \
    --exp_dir meteonet \
    --exp_note Diffcast_on_meteonet \
    --batch_size 6 \
    --backbone phydnet \
    --use_diff \
    --dataset meteo \
    --seq_len 25 \
    --epochs 50 \
    --valid \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "Diffcast_meteonet" 


CUDA_VISIBLE_DEVICES=0 python3 run_diffcast.py \
    --exp_dir meteonet \
    --exp_note Diffcast_on_meteonet \
    --batch_size 6 \
    --backbone phydnet \
    --use_diff \
    --dataset shanghai \
    --seq_len 25 \
    --valid \
    --epochs 22 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "Diffcast_shanghai" 