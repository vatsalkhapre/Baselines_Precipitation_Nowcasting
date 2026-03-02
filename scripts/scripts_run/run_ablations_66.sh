for f in 1.5
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for weight_scale in 1.5
            do
            CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_20epochs.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_w_o_spatiotemporal_gabor2 \
                --dataset sevir_lr_latent_32 \
                --exp_dir sevir_lr_latent_32_ablations \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_w_o_spatiotemporal_gabor2_${weight_scale}_${a}_${b}_${f}" \
                --epochs 50 \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
                --valid \
                --seq_len 25 \
                --frames_in 5 \
                --frames_out 20 \
                --weight_scale ${weight_scale} \
                --alpha ${a} \
                --beta ${b} \
                --freq_multiplier ${f} \
                --num_workers 8 \
                --wandb_state 'online' \
                --wandb_project_name 'Alphapre' \
                --run_name amplinet_latent_falfcl_only_2_3_13_2_w_o_spatiotemporal_gabor2_sevir_${weight_scale}_${a}_${b}_${f}

            # CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_20epochs.py \
            #     --backbone amplinet_latent_falfcl_only_2_3_13_2_w_o_spatiotemporal_gabor2 \
            #     --dataset sevir_lr_latent_32 \
            #     --exp_dir sevir_lr_latent_32_ablations \
            #     --exp_note "amplinet_latent_falfcl_only_2_3_13_2_w_o_spatiotemporal_gabor2_${weight_scale}_${a}_${b}_${f}" \
            #     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth" \
            #     --eval \
            #     --seq_len 25 \
            #     --frames_in 5 \
            #     --frames_out 20 \
            #     --weight_scale ${weight_scale} \
            #     --alpha ${a} \
            #     --beta ${b} \
            #     --freq_multiplier ${f} \
            #     --num_workers 8 \
            #     --wandb_state 'offline' 
            done
        done
    done
done



