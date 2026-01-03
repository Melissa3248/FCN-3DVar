#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J climatology
#SBATCH -t 02:00:00
#SBATCH --account=m4212_g
#SBATCH --output=../out_files/clim.out


python create_single_files_clim.py