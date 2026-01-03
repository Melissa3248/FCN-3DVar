#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J f_upsc_obs18
#SBATCH -t 03:00:00
#SBATCH --account=m4212_g
#SBATCH --output=../../out_files/upsc_obs_init_errors18.out


python ../horizon_dataloader_040324.py --config 'forecasting_given_upsc_obs_init' --yaml_config '../../config/AFNO-ERA5-GLOBAL.yaml' --weights '../../weights/backbone.ckpt'  --save_folder '../../results/forecasting_errors' --tag 'obs_upscaled' --init_filepath "../../results/vary_obs/res_q0.5r0.0001n1factor18GLOBAL"