
for f in 1.0
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for weight_scale in 1.5
            do
            # CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
            #     --backbone amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2 \
            #     --dataset cikm_latent_32 \
            #     --exp_dir cikm_latent_32_ablations \
            #     --exp_note "amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2_${weight_scale}_${a}_${b}_${f}" \
            #     --epochs 50 \
            #     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
            #     --valid \
            #     --seq_len 15 \
            #     --frames_in 5 \
            #     --frames_out 10 \
            #     --weight_scale ${weight_scale} \
            #     --alpha ${a} \
            #     --beta ${b} \
            #     --freq_multiplier ${f} \
            #     --num_workers 8 \
            #     --wandb_state 'online' \
            #     --wandb_project_name 'Alphapre' \
            #     --run_name amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2_cikm_${weight_scale}_${a}_${b}_${f}

            CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2 \
                --dataset cikm_latent_32 \
                --exp_dir cikm_latent_32_ablations \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2_${weight_scale}_${a}_${b}_${f}" \
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



for f in 1.0
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for weight_scale in 1.5
            do
            CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2 \
                --dataset shanghai_lr_latent_32 \
                --exp_dir shanghai_lr_latent_32_ablations \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_2_fourier_replaced_gabor2_${weight_scale}_${a}_${b}_${f}" \
                --epochs 50 \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
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
                --run_name amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2_shanghai_${weight_scale}_${a}_${b}_${f}

            CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2 \
                --dataset shanghai_lr_latent_32 \
                --exp_dir shanghai_lr_latent_32_ablations \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_2_fourier_replaced_gabor2_${weight_scale}_${a}_${b}_${f}" \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
                --eval \
                --seq_len 25 \
                --frames_in 5 \
                --frames_out 20 \
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

# for f in 1.0
# do
#     for a in 1.0
#     do 
#         for b in 1.0
#         do
#             for weight_scale in 1.5
#             do
#             CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#                 --backbone amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2 \
#                 --dataset meteo_lr_latent_32 \
#                 --exp_dir meteo_lr_latent_32_ablations \
#                 --exp_note "amplinet_latent_falfcl_only_2_3_13_2_2_fourier_replaced_gabor2_${weight_scale}_${a}_${b}_${f}" \
#                 --epochs 50 \
#                 --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#                 --valid \
#                 --seq_len 25 \
#                 --frames_in 5 \
#                 --frames_out 20 \
#                 --weight_scale ${weight_scale} \
#                 --alpha ${a} \
#                 --beta ${b} \
#                 --freq_multiplier ${f} \
#                 --num_workers 8 \
#                 --wandb_state 'online' \
#                 --wandb_project_name 'Alphapre' \
#                 --run_name amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2_meteonet_${weight_scale}_${a}_${b}_${f}

#             CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#                 --backbone amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2 \
#                 --dataset meteo_lr_latent_32 \
#                 --exp_dir meteo_lr_latent_32_ablations \
#                 --exp_note "amplinet_latent_falfcl_only_2_3_13_2_2_fourier_replaced_gabor2_${weight_scale}_${a}_${b}_${f}" \
#                 --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#                 --eval \
#                 --seq_len 25 \
#                 --frames_in 5 \
#                 --frames_out 20 \
#                 --weight_scale ${weight_scale} \
#                 --alpha ${a} \
#                 --beta ${b} \
#                 --freq_multiplier ${f} \
#                 --num_workers 8 \
#                 --wandb_state 'offline' 
#             done
#         done
#     done
# done

for f in 1.5
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for weight_scale in 1.5
            do
            CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2 \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteo_lr_latent_32_ablations \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_2_fourier_replaced_gabor2_${weight_scale}_${a}_${b}_${f}" \
                --epochs 50 \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
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
                --run_name amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2_meteonet_${weight_scale}_${a}_${b}_${f}

            CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_fourier_replaced_gabor2 \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteo_lr_latent_32_ablations \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_2_fourier_replaced_gabor2_${weight_scale}_${a}_${b}_${f}" \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
                --eval \
                --seq_len 25 \
                --frames_in 5 \
                --frames_out 20 \
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