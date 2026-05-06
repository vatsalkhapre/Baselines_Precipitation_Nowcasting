# CUDA_VISIBLE_DEVICES=0 python3 run_diffcast.py \
#     --exp_dir all_models_eval \
#     --exp_note Diffcast_on_sevir \
#     --batch_size 4 \
#     --backbone phydnet \
#     --use_diff \
#     --dataset cikm \
#     --seq_len 15 \
#     --eval \
#     --res_opt \
#     --ckpt_milestone /home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Diffcast/Diffphydnet_cikm_Diffcast_on_cikm/checkpoints/ckpt-best.pt \
#     --frames_in 5 \
#     --frames_out 10 \
#     --num_workers 8 \
#     --wandb_state 'offline' 


CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir all_models_eval \
    --exp_note earthformer \
    --batch_size 4 \
    --dataset cikm \
    --backbone earthformer \
    --ckpt_milestone "/home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Earthformer/earthformer_on_cikm/checkpoints/ckpt-best.pt" \
    --seq_len 15 \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' 




CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir all_models_eval \
    --exp_note mau \
    --batch_size 4 \
    --dataset cikm \
    --backbone mau \
    --ckpt_milestone "/home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/MAU/mau_on_cikm/checkpoints/ckpt-best.pt" \
    --seq_len 15 \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir all_models_eval \
    --exp_note phydnet \
    --batch_size 4 \
    --dataset cikm \
    --backbone phydnet \
    --ckpt_milestone "/home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Phydnet/phydnet_on_cikm/checkpoints/ckpt-best.pt" \
    --seq_len 15 \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir all_models_eval \
    --exp_note simvp \
    --batch_size 4 \
    --dataset cikm \
    --backbone simvp \
    --ckpt_milestone "/home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Simvp/Simvp_on_cikm/checkpoints/ckpt-best.pt" \
    --seq_len 15 \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir all_models_eval \
    --exp_note traj_gru \
    --batch_size 4 \
    --dataset cikm \
    --backbone traj_gru \
    --ckpt_milestone "/home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Traj_gru/traj_gru_on_cikm/checkpoints/ckpt-best.pt" \
    --seq_len 15 \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' 



#todo
# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
#     --exp_dir all_models_eval \
#     --exp_note dawncast \
#     --batch_size 4 \
#     --dataset cikm \
#     --backbone trajgru \
#     --ckpt_milestone "/home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Traj_gru/traj_gru_on_cikm//checkpoints/ckpt-best.pt" \
#     --seq_len 15 \
#     --eval \
#     --frames_in 5 \
#     --frames_out 10 \
#     --num_workers 8 \
#     --wandb_state 'offline' 


CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir all_models_eval \
    --exp_note earthformer_falfcl \
    --batch_size 4 \
    --dataset cikm \
    --backbone earthformer_falfcl \
    --ckpt_milestone "/home/vatsal/Dataserver2/Neurips/Models_falfcl/cikm_falfcl/earthformer_on_cikm/checkpoints/ckpt-best.pt" \
    --seq_len 15 \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir all_models_eval \
    --exp_note mau_falfcl \
    --batch_size 4 \
    --dataset cikm \
    --backbone mau_falfcl \
    --ckpt_milestone "/home/vatsal/Dataserver2/Neurips/Models_falfcl/cikm_falfcl/mau_on_cikm/checkpoints/ckpt-best.pt" \
    --seq_len 15 \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --exp_dir all_models_eval \
    --exp_note dawncast \
    --batch_size 4 \
    --dataset cikm \
    --backbone dawncast \
    --ckpt_milestone "/home/vatsal/Dataserver2/Neurips/Current_best_models/CIKM/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_cikm_latent_32_configA_beta100_freq0.1/checkpoints/ckpt-best.pt" \
    --seq_len 15 \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline'