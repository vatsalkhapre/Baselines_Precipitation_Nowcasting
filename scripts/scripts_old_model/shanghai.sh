for cm in False
do
    for conv_groups in 1
    do 
        # CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
        #     --backbone amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2 \
        #     --dataset shanghai_lr_latent_32 \
        #     --exp_dir shanghai_lr_latent_32_old_model \
        #     --exp_note "amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2_cm_${cm}_conv_gr_${conv_groups}" \
        #     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
        #     --epochs 50 \
        #     --valid \
        #     --seq_len 25 \
        #     --falfcl_weight 1 \
        #     --frames_in 5 \
        #     --frames_out 20 \
        #     --weight_scale 1.0 \
        #     --alpha 1.0 \
        #     --beta 1.0 \
        #     --freq_multiplier 1.5 \
        #     --channel_mixing ${cm} \
        #     --conv_groups ${conv_groups} \
        #     --num_workers 8 \
        #     --wandb_state 'online' \
        #     --wandb_project_name 'Alphapre' \
        #     --run_name amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2_shanghaicm_${cm}_conv_gr_${conv_groups}

        CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
            --backbone amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2 \
            --dataset shanghai_lr_latent_32 \
            --exp_dir shanghai_lr_latent_32_old_model \
            --exp_note "amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2_cm_${cm}_conv_gr_${conv_groups}" \
            --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
            --eval \
            --seq_len 25 \
            --falfcl_weight 1 \
            --frames_in 5 \
            --frames_out 20 \
            --weight_scale 1.0 \
            --alpha 1.0 \
            --beta 1.0 \
            --freq_multiplier 1.5 \
            --channel_mixing ${cm} \
            --conv_groups ${conv_groups} \
            --num_workers 8 \
            --wandb_state 'offline' 
    done
done


for cm in False
do
    for conv_groups in 2
    do 
        # CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
        #     --backbone amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2 \
        #     --dataset shanghai_lr_latent_32 \
        #     --exp_dir shanghai_lr_latent_32_old_model \
        #     --exp_note "amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2_cm_${cm}_conv_gr_${conv_groups}" \
        #     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
        #     --epochs 50 \
        #     --valid \
        #     --seq_len 25 \
        #     --falfcl_weight 1 \
        #     --frames_in 5 \
        #     --frames_out 20 \
        #     --weight_scale 1.0 \
        #     --alpha 1.0 \
        #     --beta 1.0 \
        #     --freq_multiplier 1.5 \
        #     --channel_mixing ${cm} \
        #     --conv_groups ${conv_groups} \
        #     --num_workers 8 \
        #     --wandb_state 'online' \
        #     --wandb_project_name 'Alphapre' \
        #     --run_name amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2_shanghaicm_${cm}_conv_gr_${conv_groups}

        CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
            --backbone amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2 \
            --dataset shanghai_lr_latent_32 \
            --exp_dir shanghai_lr_latent_32_old_model \
            --exp_note "amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2_cm_${cm}_conv_gr_${conv_groups}" \
            --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
            --eval \
            --seq_len 25 \
            --falfcl_weight 1 \
            --frames_in 5 \
            --frames_out 20 \
            --weight_scale 1.0 \
            --alpha 1.0 \
            --beta 1.0 \
            --freq_multiplier 1.5 \
            --channel_mixing ${cm} \
            --conv_groups ${conv_groups} \
            --num_workers 8 \
            --wandb_state 'offline' 
    done
done

for cm in True
do
    for conv_groups in 2 4 8 16
    do 
        # CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
        #     --backbone amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2 \
        #     --dataset shanghai_lr_latent_32 \
        #     --exp_dir shanghai_lr_latent_32_old_model \
        #     --exp_note "amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2_cm_${cm}_conv_gr_${conv_groups}" \
        #     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
        #     --epochs 50 \
        #     --valid \
        #     --seq_len 25 \
        #     --falfcl_weight 1 \
        #     --frames_in 5 \
        #     --frames_out 20 \
        #     --weight_scale 1.0 \
        #     --alpha 1.0 \
        #     --beta 1.0 \
        #     --freq_multiplier 1.5 \
        #     --channel_mixing ${cm} \
        #     --conv_groups ${conv_groups} \
        #     --num_workers 8 \
        #     --wandb_state 'online' \
        #     --wandb_project_name 'Alphapre' \
        #     --run_name amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2_shanghaicm_${cm}_conv_gr_${conv_groups}

        CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
            --backbone amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2 \
            --dataset shanghai_lr_latent_32 \
            --exp_dir shanghai_lr_latent_32_old_model \
            --exp_note "amplinet_latent_falfcl_only_2_3_13_2_convgrpscmgabor2_cm_${cm}_conv_gr_${conv_groups}" \
            --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
            --eval \
            --seq_len 25 \
            --falfcl_weight 1 \
            --frames_in 5 \
            --frames_out 20 \
            --weight_scale 1.0 \
            --alpha 1.0 \
            --beta 1.0 \
            --freq_multiplier 1.5 \
            --channel_mixing ${cm} \
            --conv_groups ${conv_groups} \
            --num_workers 8 \
            --wandb_state 'offline' 
    done
done



