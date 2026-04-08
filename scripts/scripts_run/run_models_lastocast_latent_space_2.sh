# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --exp_dir lastocast_mse \
#     --exp_note lastocast_on_sevir_latent_space \
#     --batch_size 4 \
#     --hidden_dim 64 \
#     --size_factor 1.0 \
#     --backbone LASTOCast_mse \
#     --dataset sevir_lr_latent_32 \
#     --seq_len 25 \
#     --weight_scale 1.5 \
#     --alpha 1.0\
#     --beta 1.0 \
#     --freq_multiplier 2.0 \
#     --valid \
#     --epochs 50 \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name "lastocast_mse_sevir_latent_space" 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --exp_dir lastocast_mse \
    --exp_note lastocast_on_sevir_latent_space \
    --batch_size 4 \
    --hidden_dim 64 \
    --size_factor 1.0 \
    --backbone LASTOCast_mse \
    --dataset sevir_lr_latent_32 \
    --seq_len 25 \
    --weight_scale 1.5 \
    --alpha 1.0\
    --beta 1.0 \
    --freq_multiplier 2.0 \
    --eval \
    --epochs 50 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
    --wandb_state 'offline' 