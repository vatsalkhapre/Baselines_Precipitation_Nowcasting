for f1 in 1.75
do
    for a1 in 1.0
    do 
        for b1 in 1.0
        do
            for weight_scale1 in 0.1
            do
                for f2 in 0.75
                do
                    for a2 in 1.0
                    do 
                        for b2 in 1.0
                        do
                            for weight_scale2 in 1.0
                            do
                                for wave in "db4" 
                                do 
                                    for wavelet_level in 2
                                    do 
                                        for afno_blocks in 1 
                                        do
                                            for afno2D_hidden_size_factor in 3  
                                            do
                                                for sparsity_threshold in 0.005
                                                do
                                                CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
                                                    --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_waveletafnogabor2 \
                                                    --dataset meteo_lr_latent_32 \
                                                    --exp_dir gabor_afno_wavelet_model_METEONET \
                                                    --exp_note "AFNO2D_relu_waveletafnogabor2_${afno_blocks}_${afno2D_hidden_size_factor}_${sparsity_threshold}" \
                                                    --epochs 50 \
                                                    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
                                                    --valid \
                                                    --seq_len 25 \
                                                    --frames_in 5 \
                                                    --frames_out 20 \
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
                                                    --num_workers 8 \
                                                    --hf_mode 'separate' \
                                                    --wandb_state 'online' \
                                                    --wandb_project_name 'Alphapre' \
                                                    --run_name AFNO2D_relu_waveletafnogabor2_${afno_blocks}_${afno2D_hidden_size_factor}_${sparsity_threshold}

                                                CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
                                                    --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_waveletafnogabor2 \
                                                    --dataset meteo_lr_latent_32 \
                                                    --exp_dir gabor_afno_wavelet_model_METEONET \
                                                    --exp_note "AFNO2D_relu_waveletafnogabor2_${afno_blocks}_${afno2D_hidden_size_factor}_${sparsity_threshold}" \
                                                    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
                                                    --eval \
                                                    --seq_len 25 \
                                                    --frames_in 5 \
                                                    --frames_out 20 \
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


# --exp_note "amplinet_latent_falfcl_only_2_3_13_2_afno_less_full_mlp_waveletsgabor2_${weight_scale}_${a}_${b}_${f}" \