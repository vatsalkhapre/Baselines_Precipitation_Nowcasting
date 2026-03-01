for weight_scale in 1.5
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for f in 1.25 1.5 1.75 2.0
            do
                for blocks in 1
                do
                CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_3_25epochs.py \
                    --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_convparallel_relu_afnogabor2 \
                    --dataset cikm_latent_32 \
                    --exp_dir cikm_new_experiments \
                    --exp_note "amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_convparallel_relu_afnogabor2_${weight_scale}_${a}_${b}_${f}_${blocks}" \
                    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
                    --epochs 50 \
                    --valid \
                    --seq_len 15 \
                    --falfcl_weight 1 \
                    --frames_in 5 \
                    --frames_out 10 \
                    --weight_scale ${weight_scale} \
                    --alpha ${a} \
                    --beta ${b} \
                    --freq_multiplier ${f} \
                    --afno_blocks ${blocks} \
                    --num_workers 8 \
                    --wandb_state 'online' \
                    --wandb_project_name 'Alphapre' \
                    --run_name amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_convparallel_relu_afnogabor2_cikm${weight_scale}_${a}_${b}_${f}_${blocks}

                CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_3_25epochs.py \
                    --backbone amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_convparallel_relu_afnogabor2 \
                    --dataset cikm_latent_32 \
                    --exp_dir cikm_new_experiments \
                    --exp_note "amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_AFNO3D_convparallel_relu_afnogabor2_${weight_scale}_${a}_${b}_${f}_${blocks}" \
                    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
                    --eval \
                    --seq_len 15 \
                    --falfcl_weight 1 \
                    --frames_in 5 \
                    --frames_out 10 \
                    --weight_scale ${weight_scale} \
                    --alpha ${a} \
                    --beta ${b} \
                    --freq_multiplier ${f} \
                    --afno_blocks ${blocks} \
                    --num_workers 8 \
                    --wandb_state 'offline'
                done
            done
        done
    done
done
