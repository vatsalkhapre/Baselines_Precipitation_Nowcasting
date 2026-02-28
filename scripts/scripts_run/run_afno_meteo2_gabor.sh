
for weight_scale in 1.5
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for f in 1.25
            do
                for blocks in 2
                do
                    for hf in 3 4 
                    do
                        for st in 0.01
                        do
                        CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                            --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_relu_afnogabor2 \
                            --dataset meteo_lr_latent_32 \
                            --exp_dir meteonet_new_experiments \
                            --exp_note "amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_relu_afnogabor2_${weight_scale}_${a}_${b}_${f}_${blocks}_${hf}_${st}" \
                            --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
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
                            --ckpt_milestone /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/meteonet_new_experiments/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_relu_afnogabor2_meteo_lr_latent_32_amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_relu_afnogabor2_1.5_1.0_1.0_1.25_4/checkpoints/ckpt-last.pt \
                            --res_opt \
                            --wandb_state 'online' \
                            --wandb_project_name 'Alphapre' \
                            --run_name amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_relu_afnogabor2_meteonet${weight_scale}_${a}_${b}_${f}_${blocks}

                        CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                            --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_relu_afnogabor2 \
                            --dataset meteo_lr_latent_32 \
                            --exp_dir meteonet_new_experiments \
                            --exp_note "amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_relu_afnogabor2_${weight_scale}_${a}_${b}_${f}_${blocks}_${hf}_${st}" \
                            --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
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