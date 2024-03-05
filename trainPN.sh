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

count=1

for var in "$@"
do
    echo "Start training for $var"
    python core/run_pn_parfile.py $var &
    pids[${count}]=$!
    sleep 1
    (( count++ ))
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done