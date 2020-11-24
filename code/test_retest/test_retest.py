import numpy as np
import os
import seaborn as sns
import pingouin as pg
import pandas as pd
from functions import test_retest_funcs as tr

home_dir = '/users/k1201869/wang_model'
data_dir = f'{home_dir}/data'
results_dir = f'{home_dir}/results'

# Subjects
subjects = os.listdir(f'{data_dir}/hcp_testretest/test')
subjects.sort()

# Load z-scored connectivity matrices and calculate nodal strengths
test_cmzs, retest_cmzs = tr.load_cmzs(subjects, f'{data_dir}/hcp_testretest/', 'cm_z_combined')
test_strengths = np.sum(test_cmzs, axis=1)
retest_strengths = np.sum(retest_cmzs, axis=1)

# Load Parameters
parameter_choice = [68,138]
rand_test_params, rand_retest_params = tr.load_parameters(subjects, 'rand', parameter_choice, f'{results_dir}/hcp_testretest')
max_test_params, max_retest_params = tr.load_parameters(subjects, 'max', parameter_choice, f'{results_dir}/hcp_testretest')
mean_test_params, mean_retest_params = tr.load_parameters(subjects, 'mean', parameter_choice, f'{results_dir}/hcp_testretest')

# Look at correlation between each test scan and the retest (each scan a vector of strength values or parameters)
strength_rankings, strength_fingerprints = tr.retest_correlation(test_strengths, retest_strengths)
rand_param_rankings, rand_param_fingerprints = tr.retest_correlation(rand_test_params, rand_retest_params)
max_param_rankings, max_param_fingerprints = tr.retest_correlation(max_test_params, max_retest_params)
mean_param_rankings, mean_param_fingerprints = tr.retest_correlation(mean_test_params, mean_retest_params)

sns.distplot(strength_rankings, kde_kws={'cut':0})
sns.distplot(mean_param_rankings, kde_kws={'cut':0})
sns.distplot(max_param_rankings, kde_kws={'cut':0})

sns.heatmap(degree_fingerprints, cmap='Reds', vmax=0.95, vmin=0.7)
sns.heatmap(mean_param_fingerprints, cmap='RdBu_r', vmax=1, vmin=-1)
sns.heatmap(mean_param_fingerprints, cmap='Reds', vmax=1, vmin=0.98)

# test retest of individual nodes
strength_reliabilities = tr.retest_reliability(subjects, test_strengths, retest_strengths)
rand_param_reliabilities = tr.retest_reliability(subjects, rand_test_params, rand_retest_params)
max_param_reliabilities = tr.retest_reliability(subjects, max_test_params, max_retest_params)
mean_param_reliabilities = tr.retest_reliability(subjects, mean_test_params, mean_retest_params)
sns.distplot(strength_reliabilities)
sns.distplot(max_param_reliabilities)
sns.distplot(mean_param_reliabilities)

np.men(mean_param_reliabilities)



# Reliability of simulation findings