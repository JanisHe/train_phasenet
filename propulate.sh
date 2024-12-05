#!/bin/bash

#SBATCH --account=seismiccentralasia1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4


export "MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n $((SLUMR_PROCID / 4)) )"
export MASTER_ADDR="$MASTER_ADDR"i

# Load modules
module purge
module load GCC OpenMPI

# https://github.com/meta-llama/llama-recipes/issues/65
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO

# Settings by Daniel
# export UCX_MEMTYPE_CACHE=0
# export NCCL_IB_TIMEOUT=100
# export SHARP_COLL_LOG_LEVEL=3
# export OMPI_MCA_coll_hcoll_enable=0
# export NCCL_COLLNET_ENABLE=0

# Activate conda environment
source /p/home/jusers/heuel1/juwels/micromamba/etc/profile.d/mamba.sh
micromamba activate seisbench_propulate

# Start Python script
srun python -u propulate_main.py $@
