CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl \
    --dataset meteo_lr_latent_32 \
    --exp_dir meteo_lr_latent_32_model_parts \
    --exp_note "amplinet_latent_falfcl" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline' 