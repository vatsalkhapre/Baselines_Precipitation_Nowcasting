for f in 1.5
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for weight_scale in 1.0
            do
            CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_mse_hf_corr_loss_gabor2 \
                --dataset cikm_latent_32 \
                --exp_dir cikm_latent_32_model_parts \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_mse_hf_corr_loss_gabor2_${weight_scale}_${a}_${b}_${f}" \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
                --valid \
                --seq_len 25 \
                --falfcl_weight 1 \
                --frames_in 5 \
                --frames_out 20 \
                --weight_scale ${weight_scale} \
                --alpha ${a} \
                --beta ${b} \
                --freq_multiplier ${f} \
                --num_workers 8 \
                --wandb_state 'online' \
                --wandb_project_name 'Alphapre' \
                --run_name amplinet_latent_falfcl_only_2_3_13_2__mse_hf_corr_loss_gabor2_cikm_${weight_scale}_${a}_${b}_${f}
            done
        done
    done
done