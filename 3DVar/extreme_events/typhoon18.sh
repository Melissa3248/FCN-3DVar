#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J 18typhoon
#SBATCH -t 10:00:00
#SBATCH --account=m4212_g
#SBATCH --output=../../out_files/typhoon18.out


# get forecasts for ground truth, upscaled observations, and ens3dvar analysis initializations
python forecasting_hurricane_2023.py --config 'perturbations' --yaml_config '../../config/AFNO-ERA5-GLOBAL.yaml' --weights '../../weights/backbone.ckpt' --img_shape_x '720' --img_shape_y '1440'  --save_dir '../../results/typhoon/factor{}' --analysis '../../results/vary_obs/res_q0.5r0.0001n1factor{}GLOBAL/update566.npy' --y_obs_upscale '../../results/vary_obs/res_q0.5r0.0001n1factor{}GLOBAL/obs_upscaled566.npy' --clean_init '../../data/2023.h5' --horizon 40 --factor 18

# get eye of hurricane and wind speed
python get_eye_over_t.py --factor 18 --dataset 'obs' --save_dir '../../results/typhoon/factor{}'
python get_eye_over_t.py --factor 18 --dataset 'clean' --save_dir '../../results/typhoon/factor{}'
python get_eye_over_t.py --factor 18 --dataset 'analysis' --save_dir '../../results/typhoon/factor{}'