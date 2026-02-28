for weight_scale in 0.75
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for f in 2.0
            do
                for blocks in 1
                do
                    for hf in 2
                    do
                        for st in 0.005 0.02
                        do
                        CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3_25epochs.py \
                            --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_relu_afnogabor2 \
                            --dataset shanghai_lr_latent_32 \
                            --exp_dir shanghai_new_experiments \
                            --exp_note "amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_relu_afnogabor2_${weight_scale}_${a}_${b}_${f}_${blocks}_${hf}_${st}" \
                            --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
                            --epochs 50 \
                            --valid \
                            --seq_len 25 \
                            --falfcl_weight 1 \
                            --frames_in 5 \
                            --frames_out 20 \
                            --weight_scale ${weight_scale} \
                            --alpha ${a} \
                            --beta ${b} \
                            --freq_multiplier ${f} \
                            --afno_blocks ${blocks} \
                            --afno2D_hidden_size_factor ${hf} \
                            --afno_sparsity_threshold ${st} \
                            --num_workers 16 \
                            --wandb_state 'online' \
                            --wandb_project_name 'Alphapre' \
                            --run_name amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_relu_afnogabor2_shanghai${weight_scale}_${a}_${b}_${f}_${blocks}_${hf}_${st}

                        CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_3_25epochs.py \
                            --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_relu_afnogabor2 \
                            --dataset shanghai_lr_latent_32 \
                            --exp_dir shanghai_new_experiments \
                            --exp_note "amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_relu_afnogabor2_${weight_scale}_${a}_${b}_${f}_${blocks}_${hf}_${st}" \
                            --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
                            --eval \
                            --seq_len 25 \
                            --falfcl_weight 1 \
                            --frames_in 5 \
                            --frames_out 20 \
                            --weight_scale ${weight_scale} \
                            --alpha ${a} \
                            --beta ${b} \
                            --freq_multiplier ${f} \
                            --afno_blocks ${blocks} \
                            --afno2D_hidden_size_factor ${hf} \
                            --afno_sparsity_threshold ${st} \
                            --num_workers 16 \
                            --wandb_state 'offline'
                        done
                    done
                done
            done
        done
    done
done