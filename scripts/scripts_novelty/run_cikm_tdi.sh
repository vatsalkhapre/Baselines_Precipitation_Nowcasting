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
                                    CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
                                        --backbone amplinet_latent_falfcl_only_2_3_13_2_conv_less_full_mlp_tdi_waveletsgabor2 \
                                        --dataset cikm_latent_32 \
                                        --exp_dir gabor_wavelet_tdi_model \
                                        --exp_note "conv_less_full_mlp_tdi_waveletsgabor2_cikm" \
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
                                        --num_workers 8 \
                                        --hf_mode 'separate' \
                                        --wandb_state 'online' \
                                        --wandb_project_name 'Alphapre' \
                                        --run_name conv_less_full_mlp_tdi_waveletsgabor2_bestcikmparams_cikm

                                    CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
                                        --backbone amplinet_latent_falfcl_only_2_3_13_2_conv_less_full_mlp_tdi_waveletsgabor2 \
                                        --dataset cikm_latent_32 \
                                        --exp_dir gabor_wavelet_tdi_model \
                                        --exp_note "conv_less_full_mlp_tdi_waveletsgabor2_cikm" \
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
                                        --wavelet_level ${wavelet_level} \
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
                                                CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
                                                    --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_tdi_waveletafnogabor2 \
                                                    --dataset cikm_latent_32 \
                                                    --exp_dir gabor_wavelet_tdi_model \
                                                    --exp_note "AFNO2D_relu_tdi_waveletafnogabor2_cikm" \
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
                                                    --wavelet_level 2 \
                                                    --afno_blocks ${afno_blocks} \
                                                    --afno2D_hidden_size_factor ${afno2D_hidden_size_factor} \
                                                    --afno_sparsity_threshold ${sparsity_threshold} \
                                                    --num_workers 8 \
                                                    --hf_mode 'separate' \
                                                    --wandb_state 'online' \
                                                    --wandb_project_name 'Alphapre' \
                                                    --run_name cikm_AFNO2D_relu_tdi_waveletafnogabor2_${afno_blocks}_${afno2D_hidden_size_factor}_${sparsity_threshold}

                                                CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
                                                    --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_tdi_waveletafnogabor2 \
                                                    --dataset cikm_latent_32 \
                                                    --exp_dir gabor_wavelet_tdi_model \
                                                    --exp_note "AFNO2D_relu_tdi_waveletafnogabor2_cikm" \
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

for f in 1.5
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for weight_scale in 1.0
            do
            CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_tdi_gabor2 \
                --dataset cikm_latent_32 \
                --exp_dir gabor_wavelet_tdi_model \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_2_tdi_gabor2_cikm" \
                --epochs 50 \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
                --valid \
                --seq_len 15 \
                --frames_in 5 \
                --frames_out 10 \
                --weight_scale ${weight_scale} \
                --alpha ${a} \
                --beta ${b} \
                --freq_multiplier ${f} \
                --num_workers 8 \
                --wandb_state 'online' \
                --wandb_project_name 'Alphapre' \
                --run_name amplinet_latent_falfcl_only_2_3_13_2_tdi_gabor2_CIKM

            CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_tdi_gabor2 \
                --dataset cikm_latent_32 \
                --exp_dir gabor_wavelet_tdi_model \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_2_tdi_gabor2_cikm" \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
                --eval \
                --seq_len 15 \
                --frames_in 5 \
                --frames_out 10 \
                --weight_scale ${weight_scale} \
                --alpha ${a} \
                --beta ${b} \
                --freq_multiplier ${f} \
                --num_workers 8 \
                --wandb_state 'offline' 
            done
        done
    done
done
