
CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_1 \
    --dataset meteo_lr_latent_32 \
    --exp_dir meteo_lr_latent_32_model_parts \
    --exp_note "amplinet_latent_falfcl_only_1" \
    --eval \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
    --num_workers 8 \
    --wandb_state 'offline' 
    

# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2.1 \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_2.1" \
#     --eval \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --num_workers 8 \
#     --wandb_state 'offline' 
    

# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2.2 \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_2.2" \
#     --eval \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --num_workers 8 \
#     --wandb_state 'offline' 

# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2.3 \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_2.3" \
#     --eval \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --num_workers 8 \
#     --wandb_state 'offline' 


# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2.3.1 \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_2.3.1" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --num_workers 8 \
#     --wandb_state 'offline' \
#     --eval 


# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2.3.2.1 \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_2.3.2.1" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --num_workers 8 \
#     --wandb_state 'offline' \
#     --eval 
    
