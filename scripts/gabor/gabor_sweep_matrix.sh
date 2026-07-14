python /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exp_paper/Gabor_exp/Gabor_codes/gabor_sweep_matrix.py \
    --logs_glob 'Exps/gabor_exp_cikm/*/logs/log.log' \
    --dataset cikm --thresholds 35 40 \
    --out_dir /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exp_paper/Gabor_exp/Gabor_explainabillity_plots/cikm_gamma_stat

python /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exp_paper/Gabor_exp/Gabor_codes/gabor_sweep_matrix.py \
    --logs_glob 'Exps/gabor_exp_shanghai/*/logs/log.log' \
    --dataset cikm --thresholds 35 40 \
    --out_dir /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exp_paper/Gabor_exp/Gabor_explainabillity_plots/shanghai_gamma_static


