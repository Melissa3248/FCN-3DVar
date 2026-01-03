#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J a_typhoon
#SBATCH -t 06:00:00

python get_eye_over_t.py --factor 18 --dataset 'analysis'