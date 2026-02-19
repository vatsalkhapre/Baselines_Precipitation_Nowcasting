# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2_3_13_2_gabor \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_2_3_13_2_gabor" \
#     --epochs 50 \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --valid \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name 'amplinet_latent_falfcl_only_2_3_13_2_gabor'

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2_3_13_2_gabor \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_2_3_13_2_gabor" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --eval \
#     --num_workers 8 \
#     --wandb_state 'offline' 


# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2_3_13_2_gabor \
#     --dataset shanghai_lr_latent_32 \
#     --exp_dir shanghai_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_2_3_13_2_gabor" \
#     --epochs 50 \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
#     --valid \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name 'amplinet_latent_falfcl_only_2_3_13_2_gabor_shanghai'

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2_3_13_2_gabor \
#     --dataset shanghai_lr_latent_32 \
#     --exp_dir shanghai_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_2_3_13_2_gabor" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
#     --eval \
#     --num_workers 8 \
#     --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.2_hfl_hybridloss \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_model_parts \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.2_hfl_hybridloss_{falfcl_weight}" \
    --epochs 1 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --valid \
    --seq_len 15 \
    --falfcl_weight 1 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.2_hfl_hybridloss_cikm_{falfclweight}'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.2_hfl_hybridloss \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_model_parts \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.2_hfl_hybridloss_{falfcl_weight}" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'