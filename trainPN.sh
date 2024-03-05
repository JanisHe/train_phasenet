#!/bin/bash

#SBATCH --cpus-per-task=6
#SBATCH --job-name="trainPhaseNet"
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1

# sbatch trainPN.sh
# Start training for all parfiles given as arguments

# Activate conda environment
ml conda/2022
conda activate seisbench

python core/run_pn_parfile.py "$1"
