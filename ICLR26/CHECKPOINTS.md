# Checkpoint inventory
_generated 2026-09-07 13:17:46_

Root on every server: `~/NWM/Baselines_Precipitation_Nowcasting/THE_GABOR/checkpoints/<run_name>/checkpoints/`
Each run directory holds `initial_model.pt`, `best_model.pt` (best val CSI),
`last_model.pt`, `final_model.pt` and the `gabor_state*.pt` files.

Naming: `Stage1_pixel_<ds>_seed<N>` (stage 1) · `Stage2_pixel_<ds>_seed<N>` (2-stage)
`Ablation_pixel_<ds>_<variant>_seed0` · `NoStage1_pixel_<ds>_seed0` (item D)
`PartF_sevir_<target>_gaborinit_<donor>_seed0` (item F)
`Gabor_pixel_SEVIR_{storm,random}_seed0` (pre-existing Part-F donors)

Published best models (reference, read-only), on **.205**:
```
/home/vatsal/Dataserver2/ICLR26/Unaliased_dataset/Best_ckpt_pixel/CIKM/CIKM_pixel_flow22.74_fhigh95.56/
/home/vatsal/Dataserver2/ICLR26/Unaliased_dataset/Best_ckpt_pixel/Meteonet/Meteonet_pixel_flow1.09_fhigh1.12/
/home/vatsal/Dataserver2/ICLR26/Unaliased_dataset/Best_ckpt_pixel/Shanghai/Shanghai_pixel_flow1.09_fhigh4.43/
/home/vatsal/Dataserver2/ICLR26/Unaliased_dataset/Best_ckpt_pixel/SEVIR/dawncast_sevir_pixel/
```

### .88 (questlab-shell)
```
Ablation_pixel_meteo_a_no_wavelet_seed0              best_step=31540    907M
Ablation_pixel_meteo_b_shared_fat_seed0              best_step=47310    908M
Ablation_pixel_meteo_c_no_gabor_seed0                best_step=47310    908M
Ablation_pixel_meteo_d_no_str_seed0                  best_step=55195    507M
Ablation_pixel_meteo_e_no_spatial_seed0              best_step=39425    681M
Ablation_pixel_meteo_f_no_srst_seed0                 best_step=55195    5.5M
Ablation_pixel_meteo_g_no_wgtm_seed0                 best_step=39425    4.2M
DAWNCast_latent_random_gaborinit_random_seed0        best_step=50000    909M
DAWNCast_latent_random_gaborinit_storm_seed0         best_step=50000    909M
DAWNCast_latent_storm_freezegabor_storm_seed0        best_step=18630    909M
Gabor_latent_SEVIR_random_seed0                      best_step=80000    6.7M
Gabor_latent_SEVIR_storm_seed0                       best_step=49680    6.7M
Gabor_pixel_SEVIR_random_seed0                       best_step=70000    6.7M
Gabor_pixel_SEVIR_storm_seed0                        best_step=37260    6.7M
Stage1_pixel_meteo_seed0                             best_step=55195    5.5M
Stage1_pixel_shanghai_seed0                          best_step=5745     7.8M
Stage2_pixel_meteo_seed0                             best_step=31540    908M
Stage2_pixel_shanghai_seed0                          best_step=9575     785M
```

### .66 (resiliente-2091)
```
Ablation_pixel_cikm_a_no_wavelet_seed0               best_step=         232M
Ablation_pixel_cikm_b_shared_fat_seed0               best_step=         233M
Ablation_pixel_cikm_c_no_gabor_seed0                 best_step=         234M
Ablation_pixel_cikm_d_no_str_seed0                   best_step=         134M
Ablation_pixel_cikm_e_no_spatial_seed0               best_step=         233M
Ablation_pixel_cikm_f_no_srst_seed0                  best_step=         6.6M
Ablation_pixel_cikm_g_no_wgtm_seed0                  best_step=         4.2M
DAWNCast_latent_random_gaborinit_random_seed0        best_step=         909M
DAWNCast_latent_random_gaborinit_storm_seed0         best_step=         909M
DAWNCast_latent_storm_freezegabor_storm_seed0        best_step=         909M
DAWNCast_latent_storm_frozen_storm_seed0             best_step=         909M
DAWNCast_latent_storm_gaborinit_random_seed0         best_step=         909M
DAWNCast_latent_storm_gaborinit_storm_seed0          best_step=         909M
Gabor_latent_SEVIR_random_seed0                      best_step=         6.7M
Gabor_latent_SEVIR_storm_seed0                       best_step=         6.7M
Gabor_pixel_SEVIR_random_seed0                       best_step=         6.7M
Gabor_pixel_SEVIR_storm_seed0                        best_step=         6.7M
NoStage1_pixel_cikm_seed0                            best_step=         234M
Stage1_pixel_cikm_seed0                              best_step=         6.6M
Stage1_pixel_cikm_seed1                              best_step=         6.6M
Stage1_pixel_cikm_seed2                              best_step=         6.6M
Stage1_pixel_cikm_seed3                              best_step=         6.6M
Stage1_pixel_cikm_seed4                              best_step=         6.6M
Stage2_pixel_cikm_seed0                              best_step=         234M
Stage2_pixel_cikm_seed1                              best_step=         234M
Stage2_pixel_cikm_seed2                              best_step=         234M
Stage2_pixel_cikm_seed3                              best_step=         234M
Stage2_pixel_cikm_seed4                              best_step=         234M
```

### .205 (questlab)
```
DAWNCast_latent_random_gaborinit_random_seed0        best_step=         909M
DAWNCast_latent_random_gaborinit_storm_seed0         best_step=         909M
DAWNCast_latent_storm_freezegabor_storm_seed0        best_step=         909M
Gabor_latent_SEVIR_random_seed0                      best_step=         6.7M
Gabor_latent_SEVIR_storm_seed0                       best_step=         6.7M
Gabor_pixel_SEVIR_random_seed0                       best_step=         6.7M
Gabor_pixel_SEVIR_storm_seed0                        best_step=         6.7M
NoStage1_pixel_meteo_seed0                           best_step=         681M
Stage1_pixel_meteo_seed0                             best_step=         5.5M
Stage1_pixel_sevir_seed0                             best_step=         6.7M
Stage1_pixel_shanghai_seed0                          best_step=         7.8M
Stage2_pixel_sevir_seed0                             best_step=         909M
```

