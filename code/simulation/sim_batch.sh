#!/bin/bash -l

#SBATCH -o /users/k1201869/wang_model/estimate_logs/sims/%a.out
#SBATCH --time=0-84:00
#SBATCH --mem=8000
#SBATCH --job-name=sims
#SBATCH -p brc

echo $SLURM_ARRAY_TASK_ID
source activate python38
cd /users/k1201869/wang_model/code/simulation
python -u sims_from_estimated_multi.py $SLURM_ARRAY_TASK_ID
