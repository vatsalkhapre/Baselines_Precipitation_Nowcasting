# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
#     --exp_dir cikm \
#     --exp_note phydnet_on_cikm \
#     --batch_size 16 \
#     --backbone phydnet \
#     --dataset cikm \
#     --seq_len 15 \
#     --valid \
#     --epochs 70 \
#     --frames_in 5 \
#     --frames_out 10 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "phydnet_cikm" 


CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir sevir \
    --exp_note phydnet_on_sevir \
    --batch_size 16 \
    --backbone phydnet \
    --dataset sevir \
    --seq_len 25 \
    --ckpt_milestone /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/shanghai/phydnet_on_shanghai/checkpoints/ckpt-last.pt \
    --res_opt \
    --valid \
    --epochs 70 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "phydnet_shanghai" 


# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
#     --exp_dir cikm \
#     --exp_note phydnet_on_cikm \
#     --batch_size 16 \
#     --backbone phydnet \
#     --dataset cikm \
#     --seq_len 15 \
#     --valid \
#     --epochs 70 \
#     --frames_in 5 \
#     --frames_out 10 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "phydnet_cikm" 


CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir sevir \
    --exp_note phydnet_on_sevir \
    --batch_size 16 \
    --backbone phydnet \
    --dataset sevir \
    --seq_len 25 \
    --valid \
    --epochs 40 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "phydnet_sevir" 

