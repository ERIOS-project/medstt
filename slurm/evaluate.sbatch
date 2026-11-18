#!/bin/bash
#SBATCH --job-name=medstt-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/%x-%j.out

source "${SLURM_SUBMIT_DIR}/slurm/env.sh"

medstt evaluate "$@"
