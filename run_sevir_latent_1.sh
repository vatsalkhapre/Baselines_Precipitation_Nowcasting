#!/bin/bash

# Define your list of weight pairs here.
# Format: "w1 w2"
# w1 = mse_weight
# w2 = falfcl_weight
declare -a weight_pairs=(
    "0.25 0.75"
)

# Loop through each pair
for pair in "${weight_pairs[@]}"; do
    # Split the pair string into two variables
    read -r w1 w2 <<< "$pair"

    echo "---------------------------------------------------"
    echo "Running experiment with MSE_Weight=$w1 and FALFCL_Weight=$w2"
    echo "---------------------------------------------------"

    # Run the python command
    CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
        --valid \
        --exp_note "amplinet_latent_hybrid_${w1}_mse_${w2}_falfcl" \
        --mse_weight "$w1" \
        --falfcl_weight "$w2" \
        --run_name "amplinet_latent32_hybridloss_${w1}_mse_${w2}_falfcl"

done

echo "All experiments finished!"