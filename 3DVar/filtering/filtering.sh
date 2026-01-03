#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J filtering_err
#SBATCH -t 01:00:00
#SBATCH --output=filtering_err.out

# vary observations

python filtering.py --data '../../data/2023.h5' --path '../../results/vary_obs/res_q0.5r0.0001n1factor{}GLOBAL' --max_t 1460 --file_tag "factor{}" --factor 8

python filtering.py --data '../../data/2023.h5' --path '../../results/vary_obs/res_q0.5r0.0001n1factor{}GLOBAL' --max_t 1460 --file_tag "factor{}" --factor 10

python filtering.py --data '../../data/2023.h5' --path '../../results/vary_obs/res_q0.5r0.0001n1factor{}GLOBAL' --max_t 1460 --file_tag "factor{}" --factor 18

python filtering.py --data '../../data/2023.h5' --path '../../results/vary_obs/res_q0.5r0.0001n1factor{}GLOBAL' --max_t 1460 --file_tag "factor{}" --factor 20

