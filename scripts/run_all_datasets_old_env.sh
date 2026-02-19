# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2.3.13.3.2 \
#     --dataset cikm_latent_32 \
#     --exp_dir cikm_latent_32_best_model \
#     --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2_old_env" \
#     --epochs 50 \
#     --seq_len 15 \
#     --frames_in 5 \
#     --frames_out 10 \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#     --valid \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name 'amplinet_latent_falfcl_only_2.3.13.3.2_cikm_old_env'

# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2.3.13.3.2 \
#     --dataset cikm_latent_32 \
#     --exp_dir cikm_latent_32_best_model \
#     --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2_old_env" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#     --seq_len 15 \
#     --frames_in 5 \
#     --frames_out 10 \
#     --eval \
#     --num_workers 8 \
#     --wandb_state 'offline'

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2.3.13.3.2 \
#     --dataset shanghai_lr_latent_32 \
#     --exp_dir shanghai_lr_latent_32_best_model \
#     --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2_old_env" \
#     --epochs 50 \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
#     --valid \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name 'amplinet_latent_falfcl_only_2.3.13.3.2_shanghai_old_env'

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2.3.13.3.2 \
#     --dataset shanghai_lr_latent_32 \
#     --exp_dir shanghai_lr_latent_32_best_model \
#     --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2_old_env" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
#     --eval \
#     --num_workers 8 \
#     --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.2 \
    --dataset meteo_lr_latent_32 \
    --exp_dir meteo_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2_old_env" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.3.2_meteonet_old_env'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.2 \
    --dataset meteo_lr_latent_32 \
    --exp_dir meteo_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2_old_env" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'

