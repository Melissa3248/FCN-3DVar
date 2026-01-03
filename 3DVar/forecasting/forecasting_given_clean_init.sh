#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J fcn-preds
#SBATCH -t 04:00:00
#SBATCH --account=m4212_g
#SBATCH --output=../../out_files/fcn-preds.out


python ../horizon_dataloader_040324.py --config 'forecasting_given_clean_init' --yaml_config '../../config/AFNO-ERA5-GLOBAL.yaml' --weights '../../weights/backbone.ckpt'  --save_folder '../../results/forecasting_errors' --tag 'clean_init'