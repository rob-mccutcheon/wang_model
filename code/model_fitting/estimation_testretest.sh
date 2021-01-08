#!/bin/bash -l

#SBATCH -o /users/k1201869/wang_model/estimate_logs/hcp_testretest/grouptest/%a.out
#SBATCH --time=0-84:00
#SBATCH --mem=20000
#SBATCH --job-name=grouptest
#SBATCH -p brc

echo $SLURM_ARRAY_TASK_ID
source activate python38
cd /users/k1201869/wang_model/code/model_fitting
python -u estimation_imag_fix_testretest.py $SLURM_ARRAY_TASK_ID 'test'
