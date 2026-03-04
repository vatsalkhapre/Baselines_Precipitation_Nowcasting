CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_20epochs.py \
    --backbone FNO_ablation \
    --dataset sevir_lr_latent_32 \
    --exp_dir sevir_lr_latent_32_ablations \
    --exp_note "FNO_4layers_Ablation" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
    --valid \
    --seq_len 25 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre' \
    --run_name FNO_4layers_Ablation_sevir

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_20epochs.py \
#     --backbone FNO_ablation \
#     --dataset sevir_lr_latent_32 \
#     --exp_dir sevir_lr_latent_32_ablations \
#     --exp_note "FNO_Ablation" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
#     --eval \
#     --seq_len 25 \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --wandb_state 'offline' 