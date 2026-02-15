python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone alpha_afnoamplinet_latent_falfcl \
    --dataset meteo_lr_latent_32 \
    --exp_dir meteo_lr_latent_32 \
    --exp_note "Testing_Integrity_with_afno_amplinet_0.01_1.0_.21model" \
    --ckpt_milestone "/home/vatsal/Dataserver/ICML26/BestModels/METEONET_LATENT/meteo_lr_latent_32/alpha_afnoamplinet_latent_falfcl_meteo_lr_latent_32_Testing_Integrity_with_afno_amplinet_0.01_1.0/checkpoints/ckpt-best.pt" \
    --eval \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
    --num_workers 8 \
    --wandb_state 'offline'