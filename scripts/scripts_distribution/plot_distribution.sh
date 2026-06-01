python3 Experiments_for_acml/Latent_local_norm_analysis/latent_norm_analysis.py \
  --dataset sevir \
  --pixel_path '/home/vatsal/Dataserver2/Datasets/sevir/' \
  --latent_path '/home/vatsal/NWM/Dataset/sevir_lr_latent_32_normalize_resize/' \
  --img_size 128 \
  --split test \
  --max_samples 1000 \
  --outdir out_sevir

python3 Experiments_for_acml/Latent_local_norm_analysis/latent_norm_analysis.py \
  --dataset meteo \
  --pixel_path '/home/vatsal/NWM/Dataset/Meteonet/meteo_radar.h5' \
  --latent_path '/home/vatsal/NWM/Dataset/meteonet_latent_32/meteonet_latent32.h5' \
  --img_size 128 \
  --split test \
  --max_samples 1000 \
  --outdir out_meteo

python3 Experiments_for_acml/Latent_local_norm_analysis/latent_norm_analysis.py \
  --dataset shanghai \
  --pixel_path '/home/vatsal/NWM/Dataset/Shanghai_Radar/shanghai.h5' \
  --latent_path '/home/vatsal/NWM/Dataset/shanghai_latent_32/shanghai_latent_data.h5' \
  --img_size 128 \
  --split test \
  --max_samples 1000 \
  --outdir out_shanghai