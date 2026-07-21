#!/bin/bash

SRC=/home/vatsal/Dataserver2/Neurips/Gabor_sweeps/cikm/gabor_sweep_cikm/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_cikm_latent_32_freq_22.74_24.34_cikm_betas_43.1034_4.8193/checkpoints/ckpt-best.pt
DST=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/Converted_DAWNCAST/cikm/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_cikm_latent_32_freq_22.74_24.34_cikm_betas_43.1034_4.8193/ckpt-best.pt

python convert_amplinet_to_dawncast.py "$SRC" "$DST" \
    --verify \
    --hf_mode shared \
    --level 2 \
    --wave db4 \
    --dim 64 \
    --t_in 5 \
    --t_out 10 \
    --img_channels 4 \
    --afno_blocks 1 \
    --hidden_size_factor 1 \
    --sparsity_threshold 0.01 \
    --k_spatial 7
