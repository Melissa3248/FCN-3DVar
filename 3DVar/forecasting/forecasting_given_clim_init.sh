#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J f_climatology
#SBATCH -t 04:00:00
#SBATCH --account=m4212_g
#SBATCH --output=../../out_files/forecast_climatology.out


python ../horizon_dataloader_040324.py --config 'forecasting_given_clim_init' --yaml_config '../../config/AFNO-ERA5-GLOBAL.yaml' --weights '../../weights/backbone.ckpt'  --save_folder '../../results/forecasting_errors' --tag 'clim_init' --init_filepath "../../data/individual_files_clim"