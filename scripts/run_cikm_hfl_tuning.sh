for falfcl_weight in 0.5 0.75 1.0 1.25 1.5 2.0
do
CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.2_hfl_hybridloss \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_model_parts \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.2_hfl_hybridloss_${falfcl_weight}" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --valid \
    --seq_len 15 \
    --falfcl_weight ${falfcl_weight} \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name "amplinet_latent_falfcl_only_2.3.13.2_hfl_hybridloss_cikm_${falfcl_weight}"
done


for falfcl_weight in 0.5 0.75 1.0 1.25 1.5 2.0
do
CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.2_hfl_hybridloss \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_model_parts \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.2_hfl_hybridloss_${falfcl_weight}" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --eval \
    --falfcl_weight ${falfcl_weight} \
    --num_workers 8 \
    --wandb_state 'offline'
done