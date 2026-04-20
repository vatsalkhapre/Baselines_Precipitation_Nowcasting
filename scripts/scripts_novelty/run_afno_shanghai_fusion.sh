# JOB_COUNT=0
# for f1 in 1.75
# do
#   for a1 in 1.0
#   do 
#     for b1 in 1.0
#     do
#       for weight_scale1 in 0.1
#       do
#         for f2 in 0.75
#         do
#           for a2 in 1.0
#           do 
#             for b2 in 1.0
#             do
#               for weight_scale2 in 1.0
#               do
#                 for wave in "db4" 
#                 do 
#                   for wavelet_level in 2
#                   do 
#                     for afno_blocks in 4 
#                     do
#                       for afno2D_hidden_size_factor in 3
#                       do
#                         for sparsity_threshold in 0
#                         do
#                           for conv_kernel in 9
#                           do 
#                             for norm_before in False
#                             do
#                               for adapt_fusion in False
#                               do
#                                 for channel_mixing in False 
#                                 do
#                                   for if_residual in True
#                                   do

#                                     GPU_ID=$((JOB_COUNT % 2))
#                                     JOB_COUNT=$((JOB_COUNT + 1))

#                                     (
#                                         CUDA_VISIBLE_DEVICES=${GPU_ID} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
#                                         --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor2 \
#                                         --dataset shanghai_lr_latent_32 \
#                                         --exp_dir gabor_afno_wavelet_fusion_model \
#                                         --exp_note "AFNO2D_relu_convparallelwaveletafnogabor2_shanghai_${conv_kernel}_${norm_before}_${adapt_fusion}_${channel_mixing}" \
#                                         --epochs 50 \
#                                         --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
#                                         --valid \
#                                         --seq_len 25 \
#                                         --frames_in 5 \
#                                         --frames_out 20 \
#                                         --weight_scale_low ${weight_scale1} \
#                                         --alpha_low ${a1} \
#                                         --beta_low ${b1} \
#                                         --freq_multiplier_low ${f1} \
#                                         --weight_scale_high ${weight_scale2} \
#                                         --alpha_high ${a2} \
#                                         --beta_high ${b2} \
#                                         --freq_multiplier_high ${f2} \
#                                         --wave ${wave} \
#                                         --wavelet_level ${wavelet_level} \
#                                         --afno_blocks ${afno_blocks} \
#                                         --afno2D_hidden_size_factor ${afno2D_hidden_size_factor} \
#                                         --afno_sparsity_threshold ${sparsity_threshold} \
#                                         --conv_kernel ${conv_kernel} \
#                                         --norm_before ${norm_before} \
#                                         --use_residual ${if_residual} \
#                                         --adaptive_fusion ${adapt_fusion} \
#                                         --channel_mixing ${channel_mixing} \
#                                         --num_workers 8 \
#                                         --hf_mode 'separate' \
#                                         --wandb_state 'online' \
#                                         --wandb_project_name 'Alphapre' \
#                                         --run_name AFNO2D_relu_convparallelwaveletafnogabor2_shanghai__${conv_kernel}_${norm_before}_${adapt_fusion}_${channel_mixing}

#                                         CUDA_VISIBLE_DEVICES=${GPU_ID} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
#                                         --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor2 \
#                                         --dataset shanghai_lr_latent_32 \
#                                         --exp_dir gabor_afno_wavelet_fusion_model \
#                                         --exp_note "AFNO2D_relu_convparallelwaveletafnogabor2_shanghai_${conv_kernel}_${norm_before}_${adapt_fusion}_${channel_mixing}" \
#                                         --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
#                                         --eval \
#                                         --seq_len 25 \
#                                         --frames_in 5 \
#                                         --frames_out 20 \
#                                         --weight_scale_low ${weight_scale1} \
#                                         --alpha_low ${a1} \
#                                         --beta_low ${b1} \
#                                         --freq_multiplier_low ${f1} \
#                                         --weight_scale_high ${weight_scale2} \
#                                         --alpha_high ${a2} \
#                                         --beta_high ${b2} \
#                                         --freq_multiplier_high ${f2} \
#                                         --wave ${wave} \
#                                         --wavelet_level ${wavelet_level} \
#                                         --afno_blocks ${afno_blocks} \
#                                         --afno2D_hidden_size_factor ${afno2D_hidden_size_factor} \
#                                         --afno_sparsity_threshold ${sparsity_threshold} \
#                                         --conv_kernel ${conv_kernel} \
#                                         --norm_before ${norm_before} \
#                                         --use_residual ${if_residual} \
#                                         --adaptive_fusion ${adapt_fusion} \
#                                         --channel_mixing ${channel_mixing} \
#                                         --num_workers 8 \
#                                         --hf_mode 'separate' \
#                                         --wandb_state 'offline'
#                                     ) &

#                                     if (( JOB_COUNT % 2 == 0 )); then
#                                       wait
#                                     fi
                                    
#                                   done
#                                 done
#                               done
#                             done
#                           done    
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done


run_job() {
    local conv_kernel=$1
    local norm_before=$2
    local adapt_fusion=$3
    local channel_mixing=$4
    local GPU_ID=$5

    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
        --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor2 \
        --dataset shanghai_lr_latent_32 \
        --exp_dir gabor_afno_wavelet_fusion_model \
        --exp_note "AFNO2D_relu_convparallelwaveletafnogabor2_shanghai_${conv_kernel}_${norm_before}_${adapt_fusion}_${channel_mixing}" \
        --epochs 50 \
        --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
        --valid \
        --seq_len 25 \
        --frames_in 5 \
        --frames_out 20 \
        --weight_scale_low 0.1 \
        --alpha_low 1.0 \
        --beta_low 1.0 \
        --freq_multiplier_low 1.75 \
        --weight_scale_high 1.0 \
        --alpha_high 1.0 \
        --beta_high 1.0 \
        --freq_multiplier_high 0.75 \
        --wave db4 \
        --wavelet_level 2 \
        --afno_blocks 4 \
        --afno2D_hidden_size_factor 3 \
        --afno_sparsity_threshold 0 \
        --conv_kernel ${conv_kernel} \
        --norm_before ${norm_before} \
        --use_residual False \
        --adaptive_fusion ${adapt_fusion} \
        --channel_mixing ${channel_mixing} \
        --num_workers 8 \
        --hf_mode 'separate' \
        --wandb_state 'online' \
        --wandb_project_name 'Alphapre' \
        --run_name AFNO2D_relu_convparallelwaveletafnogabor2_shanghai__${conv_kernel}_${norm_before}_${adapt_fusion}_${channel_mixing}

    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
        --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor2 \
        --dataset shanghai_lr_latent_32 \
        --exp_dir gabor_afno_wavelet_fusion_model \
        --exp_note "AFNO2D_relu_convparallelwaveletafnogabor2_shanghai_${conv_kernel}_${norm_before}_${adapt_fusion}_${channel_mixing}" \
        --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
        --eval \
        --seq_len 25 \
        --frames_in 5 \
        --frames_out 20 \
        --weight_scale_low 0.1 \
        --alpha_low 1.0 \
        --beta_low 1.0 \
        --freq_multiplier_low 1.75 \
        --weight_scale_high 1.0 \
        --alpha_high 1.0 \
        --beta_high 1.0 \
        --freq_multiplier_high 0.75 \
        --wave db4 \
        --wavelet_level 2 \
        --afno_blocks 4 \
        --afno2D_hidden_size_factor 3 \
        --afno_sparsity_threshold 0 \
        --conv_kernel ${conv_kernel} \
        --norm_before ${norm_before} \
        --use_residual False \
        --adaptive_fusion ${adapt_fusion} \
        --channel_mixing ${channel_mixing} \
        --num_workers 8 \
        --hf_mode 'separate' \
        --wandb_state 'offline'
}

# Config 1 → GPU 0, Config 2 → GPU 1 (run in parallel)
( run_job 7 True False False 0 ) &
( run_job 7 True  True False 1 ) &
wait


echo "All jobs done!"