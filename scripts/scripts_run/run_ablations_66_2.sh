CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm_sevir_lr_latent_20epochs.py \
    --backbone AFNO_ablation \
    --dataset sevir_lr_latent_32 \
    --exp_dir sevir_lr_latent_32_ablations \
    --exp_note "AFNO_ablation" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
    --valid \
    --seq_len 25 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name AFNO_ablation_sevir_${weight_scale}_${a}_${b}_${f}

# CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm_sevir_lr_latent_20epochs.py \
#     --backbone AFNO_ablation \
#     --dataset sevir_lr_latent_32 \
#     --exp_dir sevir_lr_latent_32_ablations \
#     --exp_note "AFNO_ablation" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
#     --eval \
#     --seq_len 25 \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --wandb_state 'offline'