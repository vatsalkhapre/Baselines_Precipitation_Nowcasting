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
                                for wave in "db4" "db6" 
                                do 
                                    for wavelet_level in 2 3
                                    do 
                                       for gfn_layers in 1 2 8 16 
                                       do 
                                        CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
                                            --backbone amplinet_latent_falfcl_only_2_3_13_2_GFN_waveletgfngabor2 \
                                            --dataset cikm_latent_32 \
                                            --exp_dir gabor_gfn_wavelet_model \
                                            --exp_note "_GFN_waveletgfngabor2_gfn_layers_${gfn_layers}_wave_${wave}_${wavelet_level}_dim128" \
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
                                            --num_gfn_layers ${gfn_layers} \
                                            --num_workers 8 \
                                            --hf_mode 'separate' \
                                            --wandb_state 'online' \
                                            --wandb_project_name 'Alphapre' \
                                            --run_name GFN_waveletgfngabor2_gfn_layers_${gfn_layers}_wave_${wave}_${wavelet_level}_dim128

                                        CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
                                            --backbone amplinet_latent_falfcl_only_2_3_13_2_GFN_waveletgfngabor2 \
                                            --dataset cikm_latent_32 \
                                            --exp_dir gabor_gfn_wavelet_model \
                                            --exp_note "_GFN_waveletgfngabor2_gfn_layers_${gfn_layers}_wave_${wave}_${wavelet_level}_dim128" \
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
                                            --num_gfn_layers ${gfn_layers} \
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
done

# --exp_note "amplinet_latent_falfcl_only_2_3_13_2_afno_less_full_mlp_waveletsgabor2_${weight_scale}_${a}_${b}_${f}" \