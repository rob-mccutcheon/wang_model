#!/bin/bash -l
#SBATCH --output=/users/k1201869/wang_model/estimate_logs/%j_2.out
#SBATCH --output=/users/k1201869/wang_model/estimate_logs/%a_2.out
#SBATCH --time=0-84:00
#SBATCH --mem=30000

echo $SLURM_ARRAY_TASK_ID

# source "/users/k1201869/miniconda3/envs/python38/bin/python"
source activate python38
cd /users/k1201869/wang_model
python -u estimation_imag_fix2.py $SLURM_ARRAY_TASK_ID