
# for wave in "db4" "db6" 
# do 
#     for wavelet_level in 2 3
#     do 
#         for sf in 1
#         do
#         CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
#             --backbone amplinet_latent_falfcl_only_2_3_13_2_conv_less_full_1mlpwavelets \
#             --dataset cikm_latent_32 \
#             --exp_dir onlymlp_wavelet_model \
#             --exp_note "conv_less_full_1mlpwavelets" \
#             --epochs 50 \
#             --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#             --valid \
#             --seq_len 15 \
#             --frames_in 5 \
#             --frames_out 10 \
#             --size_factor ${sf} \
#             --wave ${wave} \
#             --wavelet_level ${wavelet_level} \
#             --num_workers 8 \
#             --hf_mode 'separate' \
#             --wandb_state 'online' \
#             --wandb_project_name 'Alphapre' \
#             --run_name conv_less_full_1mlpwavelets_${wave}_${wavelet_level}

#         CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
#             --backbone amplinet_latent_falfcl_only_2_3_13_2_conv_less_full_1mlpwavelets \
#             --dataset cikm_latent_32 \
#             --exp_dir onlymlp_wavelet_model \
#             --exp_note "conv_less_full_1mlpwavelets" \
#             --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#             --eval \
#             --seq_len 15 \
#             --frames_in 5 \
#             --frames_out 10 \
#             --size_factor ${sf} \
#             --wave ${wave} \
#             --wavelet_level ${wavelet_level} \
#             --num_workers 8 \
#             --hf_mode 'separate' \
#             --wandb_state 'offline' 
#         done
#     done
# done

for wave in "db4" "db6" 
do 
    for wavelet_level in 2 3
    do 
    CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
        --backbone amplinet_latent_falfcl_only_2_3_13_2_conv_less_full_2mlpwavelets \
        --dataset cikm_latent_32 \
        --exp_dir onlymlp_wavelet_model \
        --exp_note "conv_less_full_2mlpwavelets" \
        --epochs 50 \
        --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
        --valid \
        --seq_len 15 \
        --frames_in 5 \
        --frames_out 10 \
        --wave ${wave} \
        --wavelet_level ${wavelet_level} \
        --num_workers 8 \
        --hf_mode 'separate' \
        --wandb_state 'online' \
        --wandb_project_name 'Alphapre' \
        --run_name conv_less_full_2mlpwavelets_${wave}_${wavelet_level}

    CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
        --backbone amplinet_latent_falfcl_only_2_3_13_2_conv_less_full_2mlpwavelets \
        --dataset cikm_latent_32 \
        --exp_dir onlymlp_wavelet_model \
        --exp_note "conv_less_full_2mlpwavelets" \
        --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
        --eval \
        --seq_len 15 \
        --frames_in 5 \
        --frames_out 10 \
        --wave ${wave} \
        --wavelet_level ${wavelet_level} \
        --num_workers 8 \
        --hf_mode 'separate' \
        --wandb_state 'offline' 
    done
done

# --exp_note "amplinet_latent_falfcl_only_2_3_13_2_afno_less_full_mlp_waveletsgabor2_${weight_scale}_${a}_${b}_${f}" \