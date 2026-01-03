#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J global20-fcn-enkf
#SBATCH -t 00:45:00
#SBATCH --account=m4212_g
#SBATCH --output=out_files/global-fcn-enkf.out


python ../fcn-3dvar-GLOBAL.py --config 'perturbations' --yaml_config '../../config/AFNO-ERA5-GLOBAL.yaml' --weights '../../weights/backbone.ckpt' --img_shape_x '720' --img_shape_y '1440'  --y_obs '../../data/observations_{}/2023_obs.h5' --q '0.5' --factor 20 --save_directory '../../results/vary_obs/res' --n_perturbations 1 --X0_init "../../data/filter_init_factor{}.npy"