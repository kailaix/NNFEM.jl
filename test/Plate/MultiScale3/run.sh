#!/bin/bash
#SBATCH --job-name=MultiScale3
#SBATCH --nodes=1 --ntasks=1
#SBATCH --output=logs/multiscale%A_%a.out
#SBATCH --error=logs/multiscale%A_%a.err
#SBATCH --partition CPU
#SBATCH --array=0-11
echo "$SLURM_ARRAY_TASK_ID"

tid=(100 101 102 103 104 200 201 202 203 204 300)
julia Data_NNPlatePull.jl ${tid[$SLURM_ARRAY_TASK_ID]} 5.0 2 2