# #========================================CIKM============================================
# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting_consecutive.py \
#       --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final \
#       --dataset cikm_latent_32 \
#       --eval \
#       --plot \
#       --seq_len 15 \
#       --frames_in 5 \
#       --frames_out 10 \
#       --weight_scale_low 0.1 \
#       --alpha_low 1.0 \
#       --beta_low 100 \
#       --freq_multiplier_low 0.1 \
#       --weight_scale_high 0.25 \
#       --alpha_high 1.0 \
#       --beta_high 100  \
#       --freq_multiplier_high 0.1 \
#       --wave "db4"  \
#       --wavelet_level 2 \
#       --hidden_dim 64 \
#       --afno_blocks 1 \
#       --afno_sparsity_threshold 0.01 \
#       --afno2D_hidden_size_factor 1 \
#       --conv_kernel 7 \
#       --num_workers 8 \
#       --plot_stride 20 \
#       --ckpt_milestone /home/vatsal/Dataserver2/Neurips/Current_best_models/CIKM/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_cikm_latent_32_configA_beta100_freq0.1/checkpoints/ckpt-best.pt \
#       --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#       --plot_stride 20 \
#       --wandb_state 'offline' 
# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting_consecutive.py \
#       --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final \
#       --dataset cikm_latent_32 \
#       --eval \
#       --plot \
#       --seq_len 15 \
#       --frames_in 5 \
#       --frames_out 10 \
#       --weight_scale_low 0.1 \
#       --alpha_low 1.0 \
#       --beta_low 100 \
#       --freq_multiplier_low 0.1 \
#       --weight_scale_high 0.25 \
#       --alpha_high 1.0 \
#       --beta_high 100  \
#       --freq_multiplier_high 0.1 \
#       --wave "db4"  \
#       --wavelet_level 2 \
#       --hidden_dim 64 \
#       --afno_blocks 1 \
#       --afno_sparsity_threshold 0.01 \
#       --afno2D_hidden_size_factor 1 \
#       --conv_kernel 7 \
#       --num_workers 8 \
#       --plot_stride 20 \
#       --ckpt_milestone /home/vatsal/Dataserver2/Neurips/Current_best_models/CIKM/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_cikm_latent_32_configA_beta100_freq0.1/checkpoints/ckpt-best.pt \
#       --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#       --plot_stride 20 \
#       --wandb_state 'offline' 

#========================================SEVIR============================================
CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting_consecutive.py \
      --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final \
      --dataset sevir_lr_latent_32 \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --weight_scale_low 0.1 \
      --alpha_low 1.0 \
      --beta_low 0.17 \
      --freq_multiplier_low 0.1 \
      --weight_scale_high 1.0 \
      --alpha_high 1.0 \
      --beta_high 0.17  \
      --freq_multiplier_high 4.0 \
      --wave "db6"  \
      --wavelet_level 2 \
      --hidden_dim 64 \
      --afno_blocks 4 \
      --afno_sparsity_threshold 0.01 \
      --afno2D_hidden_size_factor 4 \
      --conv_kernel 3 \
      --num_workers 8 \
      --plot_stride 40 \
      --ckpt_milestone /home/vatsal/Dataserver2/Neurips/Current_best_models/Sevir/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_sevir_lr_latent_32_config5_flow0.1_fhigh4.0_b0.17_db6_J2_separate/checkpoints/ckpt-best.pt \
      --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
      --wandb_state 'offline' 

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
#       --backbone alphapre \
#       --dataset sevir \
#       --eval \
#       --plot \
#       --seq_len 25 \
#       --frames_in 5 \
#       --frames_out 20 \
#       --num_workers 8 \
#       --plot_stride 40 \
#       --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Baselines/Alphapre_sevir/checkpoints/AlphaPre_sevir128.pt \
#       --plot_stride 40 \
#       --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Baselines/Alphapre_sevir/checkpoints/AlphaPre_sevir128.pt \
#       --wandb_state 'offline' 


#========================================SHANGHAI============================================

# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting_consecutive.py \
#       --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final \
#       --dataset shanghai_lr_latent_32 \
#       --eval \
#       --plot \
#       --seq_len 25 \
#       --frames_in 5 \
#       --frames_out 20 \
#       --weight_scale_low 0.1 \
#       --alpha_low 1.0 \
#       --beta_low 0.17 \
#       --freq_multiplier_low 4.0 \
#       --weight_scale_high 1.0 \
#       --alpha_high 1.0 \
#       --beta_high 0.17  \
#       --freq_multiplier_high 4.0 \
#       --wave "db6"  \
#       --wavelet_level 3 \
#       --hidden_dim 64 \
#       --afno_blocks 4 \
#       --afno_sparsity_threshold 0.01 \
#       --afno2D_hidden_size_factor 3 \
#       --conv_kernel 3 \
#       --num_workers 8 \
#       --plot_stride 10 \
#       --ckpt_milestone /home/vatsal/Dataserver2/Neurips/Current_best_models/Shanghai/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_shanghai_lr_latent_32_configC_beta0.17_freq4.0/checkpoints/ckpt-best.pt \
#       --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
#       --wandb_state 'offline' 

# # #========================================METEONET============================================

# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting_consecutive.py \
#       --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final \
#       --dataset shanghai_lr_latent_32 \
#       --eval \
#       --plot \
#       --seq_len 25 \
#       --frames_in 5 \
#       --frames_out 20 \
#       --weight_scale_low 0.1 \
#       --alpha_low 1.0 \
#       --beta_low 0.17 \
#       --freq_multiplier_low 4.0 \
#       --weight_scale_high 1.0 \
#       --alpha_high 1.0 \
#       --beta_high 0.17  \
#       --freq_multiplier_high 4.0 \
#       --wave "db6"  \
#       --wavelet_level 3 \
#       --hidden_dim 64 \
#       --afno_blocks 4 \
#       --afno_sparsity_threshold 0.01 \
#       --afno2D_hidden_size_factor 3 \
#       --conv_kernel 3 \
#       --num_workers 8 \
#       --plot_stride 10 \
#       --ckpt_milestone /home/vatsal/Dataserver2/Neurips/Current_best_models/Shanghai/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_shanghai_lr_latent_32_configC_beta0.17_freq4.0/checkpoints/ckpt-best.pt \
#       --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
#       --wandb_state 'offline' 

# # #========================================METEONET============================================

# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting_consecutive.py \
#       --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final \
#       --dataset meteo_lr_latent_32 \
#       --eval \
#       --plot \
#       --seq_len 25 \
#       --frames_in 5 \
#       --frames_out 20 \
#       --weight_scale_low 2.0 \
#       --alpha_low 1.0 \
#       --beta_low 1.0 \
#       --freq_multiplier_low 2.0 \
#       --weight_scale_high 1.0 \
#       --alpha_high 1.0 \
#       --beta_high 1.0  \
#       --freq_multiplier_high 0.75 \
#       --wave "db6"  \
#       --wavelet_level 1 \
#       --hidden_dim 64 \
#       --afno_blocks 4 \
#       --afno_sparsity_threshold 0.01 \
#       --afno2D_hidden_size_factor 6 \
#       --conv_kernel 3 \
#       --num_workers 8 \
#       --plot_stride 20 \
#       --ckpt_milestone /home/vatsal/Dataserver2/Neurips/Current_best_models/Meteonet/Best_not_included/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_meteo_lr_latent_32_afno_b4_f6_db6_J1_separate/checkpoints/ckpt-best.pt \
#       --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#       --wandb_state 'offline' 