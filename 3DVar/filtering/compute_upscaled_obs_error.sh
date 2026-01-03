#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J upsc_obs_error
#SBATCH -t 02:00:00
#SBATCH --output=upsc_obs_error.out

# vary observations

python compute_upscaled_obs_error.py --factor 8 --folder '../../results/vary_obs/res_q0.5r0.0001n1factor{}GLOBAL' 

python compute_upscaled_obs_error.py --factor 18 --folder '../../results/vary_obs/res_q0.5r0.0001n1factor{}GLOBAL' 

python compute_upscaled_obs_error.py --factor 10 --folder '../../results/vary_obs/res_q0.5r0.0001n1factor{}GLOBAL' 

python compute_upscaled_obs_error.py --factor 20 --folder '../../results/vary_obs/res_q0.5r0.0001n1factor{}GLOBAL' 