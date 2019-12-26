#!/bin/bash
#SBATCH --job-name=Plasticity
#SBATCH --nodes=1 --ntasks=1
#SBATCH --output=logs/multiscale%A_%a.out
#SBATCH --error=logs/multiscale%A_%a.err
#SBATCH --partition CPU
#SBATCH --array=0-15
echo "$SLURM_ARRAY_TASK_ID"

tid=(100 101 102 103 104 200 201 202 203 204)
julia Data_NNPlatePull.jl ${tid[$SLURM_ARRAY_TASK_ID]} 5.0 1 2