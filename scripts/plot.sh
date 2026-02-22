# #========================================CIKM============================================
# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting.py \
#       --backbone amplinet_latent_falfcl_only_2_3_13_2_gabor2 \
#       --dataset cikm_latent_32 \
#       --eval \
#       --plot \
#       --seq_len 15 \
#       --frames_in 5 \
#       --frames_out 10 \
#       --weight_scale 1.0 \
#       --alpha 1.0 \
#       --beta 1.0 \
#       --freq_multiplier 1.5 \
#       --num_workers 8 \
#       --plot_stride 10 \
#       --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Best_models/cikm_latent/amplinet_latent_falfcl_only_2_3_13_2_gabor2_cikm_latent_32_amplinet_latent_falfcl_only_2_3_13_2_gabor2_1.0_1.0_1.0_1.5/checkpoints/ckpt-best.pt \
#       --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#       --wandb_state 'offline' 

# #========================================SEVIR============================================
# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting.py \
#       --backbone amplinet_latent_falfcl_only_2_3_13_2_gabor2 \
#       --dataset sevir_lr_latent_32 \
#       --eval \
#       --plot \
#       --seq_len 25 \
#       --frames_in 5 \
#       --frames_out 20 \
#       --weight_scale 1.0 \
#       --alpha 1.0 \
#       --beta 1.0 \
#       --freq_multiplier 1.5 \
#       --num_workers 8 \
#       --plot_stride 40 \
#       --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Best_models/sevir_latent/amplinet_latent_falfcl_only_2_3_13_2_gabor2_sevir_lr_latent_32_amplinet_latent_falfcl_only_2_3_13_2_gabor2_1.0_1.0_1.0_1.5/checkpoints/ckpt-best.pt \
#       --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
#       --wandb_state 'offline' 

# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_with_plotting.py \
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
#       --wandb_state 'offline' 

#========================================SHANGHAI============================================

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting.py \
      --backbone amplinet_latent_falfcl_only_2_3_13_2_gabor2 \
      --dataset shanghai_lr_latent_32 \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --weight_scale 1.0 \
      --alpha 1.0 \
      --beta 1.0 \
      --freq_multiplier 1.5 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Best_models/shanghai_latent/amplinet_latent_falfcl_only_2_3_13_2_gabor2_shanghai_lr_latent_32_amplinet_latent_falfcl_only_2_3_13_2_gabor2_1.0_1.0_1.0_1.5/checkpoints/ckpt-best.pt \
      --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
      --wandb_state 'offline' 

#========================================METEONET============================================

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting.py \
      --backbone amplinet_latent_falfcl_only_2_3_13_2_gabor2 \
      --dataset meteo_lr_latent_32 \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --weight_scale 1.0 \
      --alpha 1.0 \
      --beta 1.0 \
      --freq_multiplier 1.5 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Best_models/meteonet_latent/amplinet_latent_falfcl_only_2_3_13_2_gabor2_meteo_lr_latent_32_amplinet_latent_falfcl_only_2_3_13_2_gabor2_1.0_1.0_1.0_1.5/checkpoints/ckpt-best.pt \
      --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
      --wandb_state 'offline' 