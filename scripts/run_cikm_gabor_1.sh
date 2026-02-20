CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2_3_13_2_gabor2 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_model_parts \
    --exp_note "amplinet_latent_falfcl_only_2_3_13_2_gabor2" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --valid \
    --seq_len 15 \
    --falfcl_weight 1 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 16 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2_3_13_2_gabor2_cikm'

