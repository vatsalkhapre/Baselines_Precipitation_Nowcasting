#!/bin/bash

echo "Starting evaluation on 3 GPUs in parallel..."

#========================================CIKM (GPU 0)============================================
CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting_consecutive.py \
      --backbone LPCast \
      --dataset cikm_latent_32 \
      --eval \
      --plot \
      --seq_len 15 \
      --frames_in 5 \
      --frames_out 10 \
      --hidden_dim 64 \
      --mlp_size_factor 1.0 \
      --lift_dims 64 64 64 \
      --proj_dims 64 64 64 4 \
      --facl_const_ratio 0.1 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ACML/Best_model/CIKM/LPCast_cikm_latent_32_hd64_lift64-64-64_proj64-64-64-4/checkpoints/ckpt-best.pt \
      --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
      --wandb_state 'offline' &

#========================================SEVIR (GPU 1)============================================
CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting_consecutive.py \
      --backbone LPCast \
      --dataset sevir_lr_latent_32 \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --hidden_dim 64 \
      --mlp_size_factor 1.0 \
      --lift_dims 64 64 64 \
      --proj_dims 64 64 4 \
      --facl_const_ratio 0.1 \
      --num_workers 8 \
      --plot_stride 40 \
      --ckpt_milestone /home/vatsal/Dataserver2/ACML/Best_model/SEVIR/LPCast_sevir_lr_latent_32_hd64_lift64-64-64_proj64-64-4/checkpoints/ckpt-best.pt \
      --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
      --wandb_state 'offline' &

#========================================SHANGHAI (GPU 2)============================================
CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting_consecutive.py \
      --backbone LPCast \
      --dataset shanghai_lr_latent_32 \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --hidden_dim 64 \
      --mlp_size_factor 1.0 \
      --lift_dims 64 64 64 \
      --proj_dims 64 64 4 \
      --facl_const_ratio 0.1 \
      --num_workers 8 \
      --plot_stride 10 \
      --ckpt_milestone /home/vatsal/Dataserver2/ACML/Best_model/SHANGHAI/LPCast_shanghai_lr_latent_32_hd64_lift64-64-64_proj64-64-4/checkpoints/ckpt-best.pt \
      --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
      --wandb_state 'offline' &

#========================================METEONET (GPU 0)============================================
CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting_consecutive.py \
      --backbone LPCast \
      --dataset meteo_lr_latent_32 \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --hidden_dim 64 \
      --mlp_size_factor 1.0 \
      --lift_dims 64 64 64 \
      --proj_dims 64 64 4 \
      --facl_const_ratio 0.1 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ACML/Best_model/METEONET/LPCast_meteo_lr_latent_32_hd64_lift64-64-64_proj64-64-4/checkpoints/ckpt-best.pt \
      --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
      --wandb_state 'offline' &

wait

echo "All evaluations finished!"