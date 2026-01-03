#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J upsc_obs
#SBATCH -t 00:30:00
#SBATCH --output=upsc_obs.out

# vary observations

python upscale_obs.py --factor 8 --n_perturbations 1 --folder '../../results/vary_obs/res_q0.5r0.0001n{}factor{}GLOBAL'

python upscale_obs.py --factor 10 --n_perturbations 1 --folder '../../results/vary_obs/res_q0.5r0.0001n{}factor{}GLOBAL'

python upscale_obs.py --factor 18 --n_perturbations 1 --folder '../../results/vary_obs/res_q0.5r0.0001n{}factor{}GLOBAL'

python upscale_obs.py --factor 20 --n_perturbations 1 --folder '../../results/vary_obs/res_q0.5r0.0001n{}factor{}GLOBAL'