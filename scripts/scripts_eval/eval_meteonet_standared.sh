for falfcl_weight in 1.25 0.5 0.75 1.0 1.5 2.0
do
CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.2_hfl_hybridloss \
    --dataset meteo_lr_latent_32 \
    --exp_dir meteo_lr_latent_32_model_parts \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.2_hfl_hybridloss" \
    --epochs 50 \
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