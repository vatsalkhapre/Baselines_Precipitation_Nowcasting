python Exp_paper/Gabor_exp/Gabor_codes/gabor_regime_calibrator.py \
    --dataset sevir \
    --weight_scale_low 0.1 --gamma_learned_low 9.7925 \
    --weight_scale_high 1.0 --gamma_learned_high 5.8113 \
    --t_in 5 --t_out 20

python Exp_paper/Gabor_exp/Gabor_codes/gabor_regime_calibrator.py \
    --dataset cikm \
    --weight_scale_low 0.1 --gamma_learned_low 0.0232 \
    --weight_scale_high 0.25 --gamma_learned_high 0.2075 \
    --t_in 5 --t_out 10

python Exp_paper/Gabor_exp/Gabor_codes/gabor_regime_calibrator.py \
    --dataset meteonet \
    --weight_scale_low 0.1 --gamma_learned_low 10.0460 \
    --weight_scale_high 1.0 --gamma_learned_high 6.0882 \
    --t_in 5 --t_out 20

python Exp_paper/Gabor_exp/Gabor_codes/gabor_regime_calibrator.py \
    --dataset shanghai \
    --weight_scale_low 0.1 --gamma_learned_low 10.0481 \
    --weight_scale_high 1.0 --gamma_learned_high 6.0480 \
    --t_in 5 --t_out 20