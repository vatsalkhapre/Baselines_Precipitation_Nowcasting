# #========================================CIKM============================================
CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting.py \
      --backbone amplinet_latent_falfcl_only_2_3_13_2_3Dconv \
      --dataset cikm_latent_32 \
      --eval \
      --plot \
      --seq_len 15 \
      --frames_in 5 \
      --frames_out 10 \
      --num_workers 8 \
      --plot_stride 10 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Ablations/W_o_Gabor/cikm_latent/amplinet_latent_falfcl_only_2_3_13_2_3Dconv_cikm_latent_32_amplinet_latent_falfcl_only_shanghai_2_3_13_2_3Dconv_corrected/checkpoints/ckpt-best.pt \
      --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
      --wandb_state 'offline' 

# #========================================SEVIR============================================
# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting.py \
#       --backbone amplinet_latent_falfcl_only_2_3_13_2_3Dconv \
#       --dataset sevir_lr_latent_32 \
#       --eval \
#       --plot \
#       --seq_len 25 \
#       --frames_in 5 \
#       --frames_out 20 \
#       --num_workers 8 \
#       --plot_stride 40 \
#       --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Best_models/sevir_latent/amplinet_latent_falfcl_only_2_3_13_2_3Dconv_sevir_lr_latent_32_amplinet_latent_falfcl_only_2_3_13_2_3Dconv_1.0_1.0_1.0_1.5/checkpoints/ckpt-best.pt \
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

# CUDA_VISIBLE_DEVICES=0 python3 run_diffcast_plotting.py \
#       --backbone phydnet \
#       --use_diff \
#       --dataset sevir \
#       --eval \
#       --plot \
#       --seq_len 25 \
#       --frames_in 5 \
#       --frames_out 20 \
#       --num_workers 8 \
#       --plot_stride 40 \
#       --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Baselines/Diffcast_sevir/checkpoints/diffcast_phydnet_sevir128.pt \
#       --wandb_state 'offline' 
#========================================SHANGHAI============================================

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting.py \
      --backbone amplinet_latent_falfcl_only_2_3_13_2_3Dconv \
      --dataset shanghai_lr_latent_32 \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Ablations/W_o_Gabor/sevir_latent/amplinet_latent_falfcl_only_2_3_13_2_3Dconv_shanghai_lr_latent_32_amplinet_latent_falfcl_only_shanghai_2_3_13_2_3Dconv_corrected/checkpoints/ckpt-best.pt \
      --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
      --wandb_state 'offline' 

#========================================METEONET============================================

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_with_plotting.py \
      --backbone amplinet_latent_falfcl_only_2_3_13_2_3Dconv \
      --dataset meteo_lr_latent_32 \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Ablations/W_o_Gabor/meteonet_latent/amplinet_latent_falfcl_only_2_3_13_2_3Dconv_meteo_lr_latent_32_amplinet_latent_falfcl_only_meteonet_2_3_13_2_3Dconv_corrected/checkpoints/ckpt-best.pt \
      --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
      --wandb_state 'offline' 