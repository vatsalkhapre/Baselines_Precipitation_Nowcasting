for f1 in 1.0
do
    for a1 in 1.0
    do 
        for b1 in 1.0
        do
            for weight_scale1 in 1.5
            do
                for f2 in 1.0
                do
                    for a2 in 1.0
                    do 
                        for b2 in 1.0
                        do
                            for weight_scale2 in 1.5
                            do
                                for wave in "haar" "db1" 
                                do 
                                    for wavelet_level in 1 2
                                    do 
                                    CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
                                        --backbone amplinet_latent_falfcl_only_2_3_13_2_conv_less_full_mlp_waveletsgabor2 \
                                        --dataset cikm_latent_32 \
                                        --exp_dir cikm_latent_32_ablations \
                                        --exp_note "amplinet_latent_falfcl_only_2_3_13_2_conv_less_full_mlp_waveletsgabor2_${weight_scale}_${a}_${b}_${f}" \
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
                                        --run_name amplinet_latent_falfcl_only_2_3_13_2_conv_less_full_mlp_waveletsgabor2_cikm_${weight_scale}_${a}_${b}_${f}

                                    CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
                                        --backbone amplinet_latent_falfcl_only_2_3_13_2_conv_less_full_mlp_waveletsgabor2 \
                                        --dataset cikm_latent_32 \
                                        --exp_dir cikm_latent_32_ablations \
                                        --exp_note "amplinet_latent_falfcl_only_2_3_13_2_conv_less_full_mlp_waveletsgabor2_${weight_scale}_${a}_${b}_${f}" \
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
                    done
                done
            done
        done
    done
done
