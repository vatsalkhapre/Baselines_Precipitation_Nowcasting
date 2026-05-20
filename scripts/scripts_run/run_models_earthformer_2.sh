# CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
#     --exp_dir cikm \
#     --exp_note earthformer_on_cikm \
#     --batch_size 8 \
#     --backbone earthformer \
#     --dataset cikm \
#     --seq_len 15 \
#     --epochs 70 \
#     --valid \
#     --frames_in 5 \
#     --frames_out 10 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "earthformer_cikm" 


# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
#     --exp_dir shanghai \
#     --exp_note earthformer_on_shanghai \
#     --batch_size 4 \
#     --backbone earthformer \
#     --dataset shanghai \
#     --seq_len 25 \
#     --valid \
#     --epochs 50 \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "earthformer_shanghai" 


# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
#     --exp_dir shanghai \
#     --exp_note earthformer_on_shanghai \
#     --batch_size 8 \
#     --backbone earthformer \
#     --dataset shanghai \
#     --seq_len 25 \
#     --epochs 100 \
#     --ckpt_milestone /home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Earthformer/earthformer_on_shanghai/checkpoints/ckpt-best.pt \
#     --res_opt \
#     --valid \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "earthformer_shanghai" 


# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
#     --exp_dir shanghai \
#     --exp_note earthformer_on_shanghai \
#     --batch_size 8 \
#     --backbone earthformer \
#     --dataset shanghai \
#     --seq_len 25 \
#     --ckpt_milestone /home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Earthformer/earthformer_on_shanghai/checkpoints/ckpt-best.pt \
#     --eval \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --wandb_state 'offline' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "earthformer_shanghai" 


CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
    --exp_dir meteonet \
    --exp_note earthformer_on_meteonet \
    --batch_size 8 \
    --backbone earthformer \
    --dataset meteo \
    --seq_len 25 \
    --ckpt_milestone /home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Earthformer/earthformer_on_meteonet_2/checkpoints/ckpt-best.pt \
    --eval \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "earthformer_shanghai" 


