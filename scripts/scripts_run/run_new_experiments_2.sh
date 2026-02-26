for weight_scale in 1.0 
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for f in 1.5
            do
                for m in 8 16
                do
                CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                    --backbone amplinet_latent_falfcl_only_2_3_13_2_conv_by_conv2Dspectralgabor2 \
                    --dataset meteo_lr_latent_32 \
                    --exp_dir meteonet_new_experiments \
                    --exp_note "amplinet_latent_falfcl_only_2_3_13_2_conv_by_conv2Dspectralgabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
                    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
                    --epochs 50 \
                    --valid \
                    --seq_len 25 \
                    --falfcl_weight 1 \
                    --frames_in 5 \
                    --frames_out 20 \
                    --modes ${m} \
                    --weight_scale ${weight_scale} \
                    --alpha ${a} \
                    --beta ${b} \
                    --freq_multiplier ${f} \
                    --num_workers 8 \
                    --wandb_state 'online' \
                    --wandb_project_name 'Alphapre' \
                    --run_name amplinet_latent_falfcl_only_2_3_13_2_conv_by_conv2Dspectralgabor2_METEONET${weight_scale}_${a}_${b}_${f}_${m}

                CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                    --backbone amplinet_latent_falfcl_only_2_3_13_2_conv_by_conv2Dspectralgabor2 \
                    --dataset meteo_lr_latent_32 \
                    --exp_dir meteonet_new_experiments \
                    --exp_note "amplinet_latent_falfcl_only_2_3_13_2_conv_by_conv2Dspectralgabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
                    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
                    --eval \
                    --seq_len 25 \
                    --falfcl_weight 1 \
                    --frames_in 5 \
                    --frames_out 20 \
                    --modes ${m} \
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



for weight_scale in 1.0 
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for f in 1.5
            do
                for m in 8 16
                do
                CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                    --backbone amplinet_latent_falfcl_only_2_3_13_2_conv3D_by_conv3Dspectralgabor2 \
                    --dataset meteo_lr_latent_32 \
                    --exp_dir meteonet_new_experiments \
                    --exp_note "amplinet_latent_falfcl_only_2_3_13_2_conv3D_by_conv3Dspectralgabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
                    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
                    --epochs 50 \
                    --valid \
                    --seq_len 25 \
                    --falfcl_weight 1 \
                    --frames_in 5 \
                    --frames_out 20 \
                    --modes ${m} \
                    --weight_scale ${weight_scale} \
                    --alpha ${a} \
                    --beta ${b} \
                    --freq_multiplier ${f} \
                    --num_workers 8 \
                    --wandb_state 'online' \
                    --wandb_project_name 'Alphapre' \
                    --run_name amplinet_latent_falfcl_only_2_3_13_2_conv3D_by_conv3Dspectralgabor2_METEONET${weight_scale}_${a}_${b}_${f}_${m}

                CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                    --backbone amplinet_latent_falfcl_only_2_3_13_2_conv3D_by_conv3Dspectralgabor2 \
                    --dataset meteo_lr_latent_32 \
                    --exp_dir meteonet_new_experiments \
                    --exp_note "amplinet_latent_falfcl_only_2_3_13_2_conv3D_by_conv3Dspectralgabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
                    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
                    --eval \
                    --seq_len 25 \
                    --falfcl_weight 1 \
                    --frames_in 5 \
                    --frames_out 20 \
                    --modes ${m} \
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



for weight_scale in 1.0 
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for f in 1.5
            do
            CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_more_conv_gabor2 \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteonet_new_experiments \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_more_conv_gabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
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
                --num_workers 8 \
                --wandb_state 'online' \
                --wandb_project_name 'Alphapre' \
                --run_name amplinet_latent_falfcl_only_2_3_13_2_more_conv_gabor2_METEONET${weight_scale}_${a}_${b}_${f}_${m}

            CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_more_conv_gabor2 \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteonet_new_experiments \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_more_conv_gabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
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
                --num_workers 8 \
                --wandb_state 'offline'
            done
        done
    done
done




for weight_scale in 1.0 
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for f in 1.5
            do
            CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_more_conv_residual_gabor2 \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteonet_new_experiments \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_more_conv_residual_gabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
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
                --num_workers 8 \
                --wandb_state 'online' \
                --wandb_project_name 'Alphapre' \
                --run_name amplinet_latent_falfcl_only_2_3_13_2_more_conv_residual_gabor2_METEONET${weight_scale}_${a}_${b}_${f}_${m}

            CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_more_conv_residual_gabor2 \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteonet_new_experiments \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_more_conv_residual_gabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
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
                --num_workers 8 \
                --wandb_state 'offline'
            done
        done
    done
done


for weight_scale in 1.0 
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for f in 1.5
            do
            CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_more_gabor_gabor2 \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteonet_new_experiments \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_more_gabor_gabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
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
                --num_workers 8 \
                --wandb_state 'online' \
                --wandb_project_name 'Alphapre' \
                --run_name amplinet_latent_falfcl_only_2_3_13_2_more_gabor_gabor2_METEONET${weight_scale}_${a}_${b}_${f}_${m}

            CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_more_gabor_gabor2 \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteonet_new_experiments \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_more_gabor_gabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
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
                --num_workers 8 \
                --wandb_state 'offline'
            done
        done
    done
done



for weight_scale in 1.0 
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for f in 1.5
            do
            CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_more_gabor_n_its_residual_gabor2 \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteonet_new_experiments \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_more_gabor_n_its_residual_gabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
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
                --num_workers 8 \
                --wandb_state 'online' \
                --wandb_project_name 'Alphapre' \
                --run_name amplinet_latent_falfcl_only_2_3_13_2_more_gabor_n_its_residual_gabor2_METEONET${weight_scale}_${a}_${b}_${f}_${m}

            CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_more_gabor_n_its_residual_gabor2 \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteonet_new_experiments \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_more_gabor_n_its_residual_gabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
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
                --num_workers 8 \
                --wandb_state 'offline'
            done
        done
    done
done



for weight_scale in 1.0 
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for f in 1.5
            do
            CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_residual_with_fusion_gabor2 \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteonet_new_experiments \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_residual_with_fusion_gabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
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
                --num_workers 8 \
                --wandb_state 'online' \
                --wandb_project_name 'Alphapre' \
                --run_name amplinet_latent_falfcl_only_2_3_13_2_residual_with_fusion_gabor2_METEONET${weight_scale}_${a}_${b}_${f}_${m}

            CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent_3.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_residual_with_fusion_gabor2 \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteonet_new_experiments \
                --exp_note "amplinet_latent_falfcl_only_2_3_13_2_residual_with_fusion_gabor2_${weight_scale}_${a}_${b}_${f}_${m}" \
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
                --num_workers 8 \
                --wandb_state 'offline'
            done
        done
    done
done


