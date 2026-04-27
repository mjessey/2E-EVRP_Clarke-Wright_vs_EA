#!/bin/bash

#SBATCH --account=bs-mojessey42
#SBATCH --job-name=csc2400_2E-EVRP_research_project
#SBATCH --output=logs/evrp_%A_%a.out
#SBATCH --error=logs/evrp_%A_%a.err
#SBATCH --partition=batch-impulse
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --export=ALL

set -eo pipefail

source ~/.bash_profile
conda activate evrp_env

cd /work/projects/bs-mojessey42/mojessey42/2E-EVRP_Clarke-Wright_vs_EA

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TMPDIR="${SLURM_TMPDIR:-/tmp}"

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-none}"

python code/main.py < benchmark_1.txt
