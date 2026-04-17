for f1 in 0.75
do
    for a1 in 1.0
    do 
        for b1 in 1.0
        do
            for weight_scale1 in 0.1
            do
                for f2 in 1.0
                do
                    for a2 in 1.0
                    do 
                        for b2 in 1.0
                        do
                            for weight_scale2 in 0.25
                            do
                                for wave in "db4" 
                                do 
                                    for wavelet_level in 2
                                    do 
                                        for afno_blocks in 2 
                                        do
                                            for afno2D_hidden_size_factor in 2
                                            do
                                                for sparsity_threshold in 0.01
                                                do
                                                    for conv_kernel in 3 5 7 9
                                                    do 
                                                        for norm_before in True False
                                                        do
                                                            for adapt_fusion in True False
                                                            do
                                                                for channel_mixing in True False 
                                                                do
                                                                    for if_residual in False
                                                                    do
                                                                    CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
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
                                                                        --weight_scale_low ${weight_scale1} \
                                                                        --alpha_low ${a1} \
                                                                        --beta_low ${b1} \
                                                                        --freq_multiplier_low ${f1} \
                                                                        --weight_scale_high ${weight_scale2} \
                                                                        --alpha_high ${a2} \
                                                                        --beta_high ${b2} \
                                                                        --freq_multiplier_high ${f2} \
                                                                        --wave ${wave} \
                                                                        --wavelet_level ${wavelet_level} \
                                                                        --afno_blocks ${afno_blocks} \
                                                                        --afno2D_hidden_size_factor ${afno2D_hidden_size_factor} \
                                                                        --afno_sparsity_threshold ${sparsity_threshold} \
                                                                        --conv_kernel ${conv_kernel} \
                                                                        --norm_before ${norm_before} \
                                                                        --use_residual ${if_residual} \
                                                                        --adaptive_fusion ${adapt_fusion} \
                                                                        --channel_mixing ${channel_mixing} \
                                                                        --num_workers 8 \
                                                                        --hf_mode 'separate' \
                                                                        --wandb_state 'offline' \
                                                                        --wandb_project_name 'Alphapre' \
                                                                        --run_name AFNO2D_relu_convparallelwaveletafnogabor2_cikm_${afno_blocks}_${afno2D_hidden_size_factor}_${sparsity_threshold}

                                                                    CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
                                                                        --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor2 \
                                                                        --dataset cikm_latent_32 \
                                                                        --exp_dir gabor_afno_wavelet_fusion_model \
                                                                        --exp_note "AFNO2D_relu_convparallelwaveletafnogabor2_cikmbestparams_${conv_kernel}_${norm_before}_${adapt_fusion}_${channel_mixing}" \
                                                                        --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
                                                                        --eval \
                                                                        --seq_len 15 \
                                                                        --frames_in 5 \
                                                                        --frames_out 10 \
                                                                        --weight_scale_low ${weight_scale1} \
                                                                        --alpha_low ${a1} \
                                                                        --beta_low ${b1} \
                                                                        --freq_multiplier_low ${f1} \
                                                                        --weight_scale_high ${weight_scale2} \
                                                                        --alpha_high ${a2} \
                                                                        --beta_high ${b2} \
                                                                        --freq_multiplier_high ${f2} \
                                                                        --wave ${wave} \
                                                                        --wavelet_level 2 \
                                                                        --afno_blocks ${afno_blocks} \
                                                                        --afno2D_hidden_size_factor ${afno2D_hidden_size_factor} \
                                                                        --afno_sparsity_threshold ${sparsity_threshold} \
                                                                        --conv_kernel ${conv_kernel} \
                                                                        --norm_before ${norm_before} \
                                                                        --use_residual ${if_residual} \
                                                                        --adaptive_fusion ${adapt_fusion} \
                                                                        --channel_mixing ${channel_mixing} \
                                                                        --num_workers 8 \
                                                                        --hf_mode 'separate' \
                                                                        --wandb_state 'offline'
                                                                    done
                                                                done
                                                            done
                                                        done
                                                    done    
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# --exp_note "amplinet_latent_falfcl_only_2_3_13_2_afno_less_full_mlp_waveletsgabor2_${weight_scale}_${a}_${b}_${f}" \