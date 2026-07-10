python /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Gabor_exp/Gabor_codes/gabor_sweep_matrix.py \
    --logs_glob 'Exps/multiseed_cikm/*/logs/log.log' \
    --dataset cikm --thresholds 35 40 \
    --out_dir /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Gabor_exp/Gabor_explainabillity_plots/cikm

python /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exp_paper/Gabor_exp/Gabor_codes/gabor_sweep_matrix.py \
    --logs_glob 'Exps/gabor_exp_shanghai/*/logs/log.log' \
    --dataset shanghai --thresholds 35 40 \
    --out_dir /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exp_paper/Gabor_exp/Gabor_explainabillity_plots/stat_gamma_shanghai

python /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exp_paper/Gabor_exp/Gabor_codes/gabor_sweep_matrix.py \
    --logs_glob 'Exps/gabor_exp_meteo/*/logs/log.log' \
    --dataset shanghai --thresholds 24 32 \
    --out_dir /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exp_paper/Gabor_exp/Gabor_explainabillity_plots/meteonet
    
python /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exp_paper/Gabor_exp/Gabor_codes/gabor_sweep_matrix.py \
    --logs_glob 'Exps/gabor_exp_meteo/*/logs/log.log' \
    --dataset meteonet --thresholds 24 32 \
    --out_dir /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exp_paper/Gabor_exp/Gabor_explainabillity_plots/meteonet_const_gamma