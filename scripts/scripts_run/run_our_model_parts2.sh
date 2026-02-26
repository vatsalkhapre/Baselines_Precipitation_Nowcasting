CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2_3_13_2_MLP_linear_gabor2 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_our_model_parts \
    --exp_note "amplinet_latent_falfcl_only_2_3_13_2_MLP_linear_gabor2_1.0_1.0_1.0_1.5" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --epochs 50 \
    --valid \
    --seq_len 15 \
    --falfcl_weight 1 \
    --frames_in 5 \
    --frames_out 10 \
    --weight_scale 1.0 \
    --alpha 1.0 \
    --beta 1.0 \
    --freq_multiplier 1.5 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name amplinet_latent_falfcl_only_2_3_13_2_MLP_linear_gabor2_cikm1.0_1.0_1.0_1.5

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2_3_13_2_MLP_linear_gabor2 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_our_model_parts \
    --exp_note "amplinet_latent_falfcl_only_2_3_13_2_MLP_linear_gabor2_1.0_1.0_1.0_1.5" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --eval \
    --seq_len 15 \
    --falfcl_weight 1 \
    --frames_in 5 \
    --frames_out 10 \
    --weight_scale 1.0 \
    --alpha 1.0 \
    --beta 1.0 \
    --freq_multiplier 1.5 \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_2.py \
    --backbone MLP \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_our_model_parts \
    --exp_note "MLP" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --epochs 50 \
    --valid \
    --seq_len 15 \
    --falfcl_weight 1 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name amplinet_latent_falfcl_only_MLP

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_2.py \
    --backbone MLP \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_our_model_parts \
    --exp_note "MLP" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --eval \
    --seq_len 15 \
    --falfcl_weight 1 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_2.py \
    --backbone ConvMLP \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_our_model_parts \
    --exp_note "ConvMLP" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --epochs 50 \
    --valid \
    --seq_len 15 \
    --falfcl_weight 1 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name amplinet_latent_falfcl_only_ConvMLP

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_2.py \
    --backbone ConvMLP \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_our_model_parts \
    --exp_note "ConvMLP" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --eval \
    --seq_len 15 \
    --falfcl_weight 1 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_2.py \
    --backbone ConvMLP_activation \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_our_model_parts \
    --exp_note "ConvMLP_activation" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --epochs 50 \
    --valid \
    --seq_len 15 \
    --falfcl_weight 1 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name amplinet_latent_falfcl_only_ConvMLP_activation

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_2.py \
    --backbone ConvMLP_activation \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_our_model_parts \
    --exp_note "ConvMLP_activation" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --eval \
    --seq_len 15 \
    --falfcl_weight 1 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline'

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_2.py \
#     --backbone Conv_MLP_Gabor_activation \
#     --dataset cikm_latent_32 \
#     --exp_dir cikm_latent_32_our_model_parts \
#     --exp_note "Conv_MLP_Gabor_activation" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#     --epochs 50 \
#     --valid \
#     --seq_len 15 \
#     --falfcl_weight 1 \
#     --frames_in 5 \
#     --frames_out 10 \
#     --weight_scale 1.0 \
#     --alpha 1.0 \
#     --beta 1.0 \
#     --freq_multiplier 1.5 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name amplinet_latent_falfcl_only_Conv_MLP_Gabor_activation

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_2.py \
#     --backbone Conv_MLP_Gabor_activation \
#     --dataset cikm_latent_32 \
#     --exp_dir cikm_latent_32_our_model_parts \
#     --exp_note "Conv_MLP_Gabor_activation" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#     --eval \
#     --seq_len 15 \
#     --falfcl_weight 1 \
#     --frames_in 5 \
#     --frames_out 10 \
#     --weight_scale 1.0 \
#     --alpha 1.0 \
#     --beta 1.0 \
#     --freq_multiplier 1.5 \
#     --num_workers 8 \
#     --wandb_state 'offline'

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_2.py \
#     --backbone Conv_MLP_Gabor_activation_conv \
#     --dataset cikm_latent_32 \
#     --exp_dir cikm_latent_32_our_model_parts \
#     --exp_note "Conv_MLP_Gabor_activation_conv" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#     --epochs 50 \
#     --valid \
#     --seq_len 15 \
#     --falfcl_weight 1 \
#     --frames_in 5 \
#     --frames_out 10 \
#     --weight_scale 1.0 \
#     --alpha 1.0 \
#     --beta 1.0 \
#     --freq_multiplier 1.5 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name amplinet_latent_falfcl_only_Conv_MLP_Gabor_activation_conv

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_2.py \
#     --backbone Conv_MLP_Gabor_activation_conv \
#     --dataset cikm_latent_32 \
#     --exp_dir cikm_latent_32_our_model_parts \
#     --exp_note "Conv_MLP_Gabor_activation_conv" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#     --eval \
#     --seq_len 15 \
#     --falfcl_weight 1 \
#     --frames_in 5 \
#     --frames_out 10 \
#     --weight_scale 1.0 \
#     --alpha 1.0 \
#     --beta 1.0 \
#     --freq_multiplier 1.5 \
#     --num_workers 8 \
#     --wandb_state 'offline'

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_2.py \
#     --backbone Conv_MLP_Gabor_activation_conv_residual \
#     --dataset cikm_latent_32 \
#     --exp_dir cikm_latent_32_our_model_parts \
#     --exp_note "Conv_MLP_Gabor_activation_conv_residual" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#     --epochs 50 \
#     --valid \
#     --seq_len 15 \
#     --falfcl_weight 1 \
#     --frames_in 5 \
#     --frames_out 10 \
#     --weight_scale 1.0 \
#     --alpha 1.0 \
#     --beta 1.0 \
#     --freq_multiplier 1.5 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name amplinet_latent_falfcl_only_Conv_MLP_Gabor_activation_conv_residual

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_2.py \
#     --backbone Conv_MLP_Gabor_activation_conv_residual \
#     --dataset cikm_latent_32 \
#     --exp_dir cikm_latent_32_our_model_parts \
#     --exp_note "Conv_MLP_Gabor_activation_conv_residual" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#     --eval \
#     --seq_len 15 \
#     --falfcl_weight 1 \
#     --frames_in 5 \
#     --frames_out 10 \
#     --weight_scale 1.0 \
#     --alpha 1.0 \
#     --beta 1.0 \
#     --freq_multiplier 1.5 \
#     --num_workers 8 \
#     --wandb_state 'offline'