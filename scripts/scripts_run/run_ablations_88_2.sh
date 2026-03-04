CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone WNO_ablation \
    --dataset sevir_lr_latent_32 \
    --exp_dir sevir_lr_latent_32_ablations \
    --exp_note "WNO_ablation" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
    --valid \
    --hidden_size 96 \
    --seq_len 25 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name WNO_ablation_sevir_${weight_scale}_${a}_${b}_${f}

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone WNO_ablation \
    --dataset sevir_lr_latent_32 \
    --exp_dir sevir_lr_latent_32_ablations \
    --exp_note "WNO_ablation" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
    --eval \
    --hidden_size 96 \
    --seq_len 25 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' 