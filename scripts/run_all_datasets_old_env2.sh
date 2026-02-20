CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.2 \
    --dataset sevir_lr_latent_32 \
    --exp_dir sevir_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.2_old_env" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.2_sevir_old_env'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.2 \
    --dataset sevir_lr_latent_32 \
    --exp_dir sevir_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.2_old_env" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'