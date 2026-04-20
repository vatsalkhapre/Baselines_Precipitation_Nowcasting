# wait  # Wait for any remaining background jobs
# echo "All jobs done!"

JOB_COUNT=0
mkdir -p logs_multiprocessing
run_job() {
    local conv_kernel=$1
    local norm_before=$2
    local adapt_fusion=$3
    local channel_mixing=$4
    local GPU_ID=$5

    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
        --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor2 \
        --dataset cikm_latent_32 \
        --exp_dir gabor_afno_wavelet_fusion_model \
        --exp_note "AFNO2D_relu_convparallelwaveletafnogabor2_cikmbestparams_${conv_kernel}_${norm_before}_${adapt_fusion}_${channel_mixing}" \
        --epochs 50 \
        --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
        --valid \
        --seq_len 15 \
        --frames_in 5 \
        --frames_out 10 \
        --weight_scale_low 0.1 \
        --alpha_low 1.0 \
        --beta_low 1.0 \
        --freq_multiplier_low 0.75 \
        --weight_scale_high 0.25 \
        --alpha_high 1.0 \
        --beta_high 1.0 \
        --freq_multiplier_high 1.0 \
        --wave db4 \
        --wavelet_level 2 \
        --afno_blocks 2 \
        --afno2D_hidden_size_factor 2 \
        --afno_sparsity_threshold 0.01 \
        --conv_kernel ${conv_kernel} \
        --norm_before ${norm_before} \
        --use_residual False \
        --adaptive_fusion ${adapt_fusion} \
        --channel_mixing ${channel_mixing} \
        --num_workers 8 \
        --hf_mode 'separate' \
        --wandb_state 'online' \
        --wandb_project_name 'Alphapre' \
        --run_name AFNO2D_relu_convparallelwaveletafnogabor2_cikm_bestparams_${conv_kernel}_${norm_before}_${adapt_fusion}_${channel_mixing}

    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
        --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor2 \
        --dataset cikm_latent_32 \
        --exp_dir gabor_afno_wavelet_fusion_model \
        --exp_note "AFNO2D_relu_convparallelwaveletafnogabor2_cikmbestparams_${conv_kernel}_${norm_before}_${adapt_fusion}_${channel_mixing}" \
        --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
        --eval \
        --seq_len 15 \
        --frames_in 5 \
        --frames_out 10 \
        --weight_scale_low 0.1 \
        --alpha_low 1.0 \
        --beta_low 1.0 \
        --freq_multiplier_low 0.75 \
        --weight_scale_high 0.25 \
        --alpha_high 1.0 \
        --beta_high 1.0 \
        --freq_multiplier_high 1.0 \
        --wave db4 \
        --wavelet_level 2 \
        --afno_blocks 2 \
        --afno2D_hidden_size_factor 2 \
        --afno_sparsity_threshold 0.01 \
        --conv_kernel ${conv_kernel} \
        --norm_before ${norm_before} \
        --use_residual False \
        --adaptive_fusion ${adapt_fusion} \
        --channel_mixing ${channel_mixing} \
        --num_workers 8 \
        --hf_mode 'separate' \
        --wandb_state 'offline'
}

# 3 configs — assign GPUs manually
# Config 1 → GPU 0
# Config 1 → GPU 0
( run_job 7 True True False 0 ) &

# Config 2 → GPU 1
( run_job 7 True False False 1 ) &

wait

# Config 3 → GPU 0
( run_job 7 True False True 0 ) &

wait
echo "All jobs done!"

wait
echo "All jobs done!"