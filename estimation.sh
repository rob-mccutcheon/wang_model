#!/bin/bash -l
#SBATCH --output=/users/k1201869/wang_model/estimate_logs/%j.out
#SBATCH --output=/users/k1201869/wang_model/estimate_logs/%a.out
#SBATCH --time=0-84:00
#SBATCH --mem=12000

echo $SLURM_ARRAY_TASK_ID
source activate python38
cd /users/k1201869/wang_model
python -u estimation_imag_fix.py $SLURM_ARRAY_TASK_ID