python3 inspect_gabor_drift.py \
    --ckpt /home/vatsal/Dataserver2/Neurips/Current_best_models/CIKM/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_cikm_latent_32_configA_beta100_freq0.1/checkpoints/ckpt-best.pt \
    --dataset cikm \
    --model_type lastocast \
    --hf_mode separate \
    --level 2 \
    --beta_low 100 \
    --beta_high 100 \
    --freq_multiplier_low 0.1 \
    --freq_multiplier_high 0.1 \
    --save_fig \
    --save_csv \
    --out_dir /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Gabor_testing/cikm/

python3 inspect_gabor_drift.py \
    --ckpt /home/vatsal/Dataserver2/Neurips/Current_best_models/Meteonet/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_meteo_lr_latent_32_flow0.1_fhigh4.0_b0.17_db6_J1_separate/checkpoints/ckpt-best.pt \
    --dataset meteonet \
    --model_type lastocast \
    --hf_mode separate \
    --level 1 \
    --beta_low 0.17 \
    --beta_high 0.17 \
    --freq_multiplier_low 0.1 \
    --freq_multiplier_high 4.0 \
    --weight_scale_low 0.1 \
    --weight_scale_high 1.0 \
    --save_fig \
    --save_csv \
    --out_dir /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Gabor_testing/meteonet/

python3 inspect_gabor_drift.py \
    --ckpt /home/vatsal/Dataserver2/Neurips/Current_best_models/Sevir/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_sevir_lr_latent_32_config5_flow0.1_fhigh4.0_b0.17_db6_J2_separate/checkpoints/ckpt-best.pt \
    --dataset sevir \
    --model_type lastocast \
    --hf_mode separate \
    --level 2 \
    --beta_low 0.17 \
    --beta_high 0.17 \
    --freq_multiplier_low 0.1 \
    --freq_multiplier_high 4.0 \
    --weight_scale_low 0.1 \
    --weight_scale_high 1.0 \
    --save_fig \
    --save_csv \
    --out_dir /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Gabor_testing/sevir/

python3 inspect_gabor_drift.py \
    --ckpt /home/vatsal/Dataserver2/Neurips/Current_best_models/Shanghai/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_shanghai_lr_latent_32_configC_beta0.17_freq4.0/checkpoints/ckpt-best.pt \
    --dataset shanghai \
    --model_type lastocast \
    --hf_mode separate \
    --level 3 \
    --beta_low 0.17 \
    --beta_high 0.17 \
    --freq_multiplier_low 4.0 \
    --freq_multiplier_high 4.0 \
    --weight_scale_low 0.1 \
    --weight_scale_high 1.0 \
    --save_fig \
    --save_csv \
    --out_dir /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Gabor_testing/shanghai/
