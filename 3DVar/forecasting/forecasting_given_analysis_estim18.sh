#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J 18errors
#SBATCH -t 04:00:00
#SBATCH --account=m4212_g
#SBATCH --output=../../out_files/filter_init_errors18.out


python ../horizon_dataloader_040324.py --config 'forecasting_given_filter_init' --yaml_config '../../config/AFNO-ERA5-GLOBAL.yaml' --weights '../../weights/backbone.ckpt'  --save_folder '../../results/forecasting_errors' --tag 'update' --init_filepath "../../results/vary_obs/res_q0.5r0.0001n1factor18GLOBAL"
