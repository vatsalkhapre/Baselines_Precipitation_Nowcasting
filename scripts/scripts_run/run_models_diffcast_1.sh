# CUDA_VISIBLE_DEVICES=0 python3 run_diffcast.py \
#     --exp_dir meteonet \
#     --exp_note Diffcast_on_meteonet \
#     --batch_size 6 \
#     --backbone phydnet \
#     --use_diff \
#     --dataset meteo \
#     --seq_len 25 \
#     --epochs 30 \
#     --valid \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "Diffcast_meteonet" 


CUDA_VISIBLE_DEVICES=1 python3 run_diffcast.py \
    --exp_dir shanghai \
    --exp_note Diffcast_on_shanghai \
    --batch_size 6 \
    --backbone phydnet \
    --use_diff \
    --dataset shanghai \
    --seq_len 25 \
    --valid \
    --res_opt \
    --ckpt_milestone /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/shanghai/Diffphydnet_shanghai_Diffcast_on_shanghai/checkpoints/ckpt-last.pt \
    --epochs 75 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "Diffcast_shanghai" 


# CUDA_VISIBLE_DEVICES=0 python3 run_diffcast.py \
#     --exp_dir cikm \
#     --exp_note Diffcast_on_cikm \
#     --batch_size 6 \
#     --backbone phydnet \
#     --use_diff \
#     --dataset shanghai \
#     --seq_len 15 \
#     --valid \
#     --epochs 60 \
#     --frames_in 5 \
#     --frames_out 10 \[]
#     --num_workers 8 \
#     --wandb_state 'offline' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "Diffcast_shanghai" 