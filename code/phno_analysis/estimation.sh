#!/bin/bash -l

#SBATCH -o /users/k1201869/wang_model/estimate_logs/ril_fit/%a.out
#SBATCH --time=0-84:00
#SBATCH --mem=20000
#SBATCH --job-name=ril_fit
#SBATCH -p brc

echo $SLURM_ARRAY_TASK_ID
source activate python38
cd /users/k1201869/wang_model/code/riluzole_analysis
python -u estimation.py $SLURM_ARRAY_TASK_ID
