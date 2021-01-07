import numpy as np
import os
import sys
sys.path.append('/users/k1201869/wang_model/code')
# os.chdir('/users/k1201869/wang_model/code')
import seaborn as sns
import pingouin as pg
import pandas as pd
import  matplotlib.pyplot as plt
from functions import test_retest_funcs as tr
from functions import simulation as sim
import pickle

home_dir = '/users/k1201869/wang_model'
data_dir = f'{home_dir}/data'
results_dir = f'{home_dir}/results'

# Subjects
subjects = os.listdir(f'{data_dir}/hcp_testretest/test')
subjects.sort()

# # Load z-scored connectivity matrices and calculate nodal strengths
# test_cmzs, retest_cmzs = tr.load_cmzs(subjects, f'{data_dir}/hcp_testretest/', 'cm_z_combined')
# test_strengths = np.sum(test_cmzs, axis=1)
# retest_strengths = np.sum(retest_cmzs, axis=1)

# # Load Parameters
# parameter_choice = [68,136]
# rand_test_params, rand_retest_params = tr.load_parameters(subjects, 'rand', parameter_choice, f'{results_dir}/hcp_testretest')
# max_test_params, max_retest_params = tr.load_parameters(subjects, 'max', parameter_choice, f'{results_dir}/hcp_testretest')
# mean_test_params, mean_retest_params = tr.load_parameters(subjects, 'mean', parameter_choice, f'{results_dir}/hcp_testretest')

# # Look at correlation between each test scan and the retest (each scan a vector of strength values or parameters)
# strength_rankings, strength_fingerprints = tr.retest_correlation(test_strengths, retest_strengths)
# rand_param_rankings, rand_param_fingerprints = tr.retest_correlation(rand_test_params, rand_retest_params)
# max_param_rankings, max_param_fingerprints = tr.retest_correlation(max_test_params, max_retest_params)
# mean_param_rankings, mean_param_fingerprints = tr.retest_correlation(mean_test_params, mean_retest_params)

# sns.distplot(strength_rankings, kde_kws={'cut':0})
# plt.xlabel('ranking')
# np.median(strength_rankings)

# sns.set_theme(style="whitegrid")
# sns.distplot(rand_param_rankings, kde_kws={'cut':0}, label='random')
# sns.distplot(mean_param_rankings, kde_kws={'cut':0}, label='mean')
# sns.distplot(max_param_rankings, kde_kws={'cut':0}, label='max')
# plt.xlabel('ranking')
# plt.legend()
# np.median(mean_param_rankings)

# sns.heatmap(strength_fingerprints, cmap='Reds', vmax=0.95, vmin=0.7)

# sns.heatmap(mean_param_fingerprints, cmap='RdBu_r', vmax=1, vmin=-1)
# plt.title('Mean')

# # test retest of individual nodes
# strength_reliabilities = tr.retest_reliability(subjects, test_strengths, retest_strengths)
# rand_param_reliabilities = tr.retest_reliability(subjects, rand_test_params, rand_retest_params)
# max_param_reliabilities = tr.retest_reliability(subjects, max_test_params, max_retest_params)
# mean_param_reliabilities = tr.retest_reliability(subjects, mean_test_params, mean_retest_params)
# sns.distplot(strength_reliabilities)
# plt.xlabel('ICC')
# np.median(strength_reliabilities)

# sns.distplot(strength_reliabilities)
# sns.distplot(rand_param_reliabilities, label='random')
# sns.distplot(mean_param_reliabilities, label='mean')
# sns.distplot(max_param_reliabilities, label='max')
# plt.legend()
# plt.xlabel('ICC')

# np.median(mean_param_reliabilities)


# Reliability of simulation findings
parameter_choice = [0,138]
# all_test_params, all_retest_params = tr.load_parameters(subjects, 'mean', parameter_choice, f'{results_dir}/hcp_testretest')
NState = 0
SC = np.loadtxt(open(f"/users/k1201869/wang_model/data/SC_dk.csv", "rb"), delimiter=",")
SC = (SC/np.max(np.max(SC)))*0.2


# i=0
# for param_set, firing_dict in zip([all_test_params, all_retest_params], [test_firing, retest_firing]):
#     for ParaE in param_set:
#         print(i)
#         ParaE = np.atleast_2d(ParaE).T
#         y,h,x = sim.firing_rate(ParaE, SC, NState)
#         firing_dict['y_mean'].append(np.mean(y, axis=1))
#         firing_dict['y_sd'].append(np.std(y, axis=1))
#         firing_dict['h_mean'].append(np.mean(h, axis=1))
#         firing_dict['h_sd'].append(np.std(h, axis=1))
#         firing_dict['x_mean'].append(np.mean(x, axis=1))
#         firing_dict['x_sd'].append(np.std(x, axis=1))
#         i=i+1

# pickle.dump(test_firing, open(f'{results_dir}/hcp_testretest/secondary_analysis/test_firing_state{NState}.pkl', "wb"))
# pickle.dump(retest_firing, open(f'{results_dir}/hcp_testretest/secondary_analysis/retest_firing_state{NState}.pkl', "wb"))

# retest_firing1 = pickle.load(open(f'{results_dir}/hcp_testretest/secondary_analysis/retest_firing_state1.pkl', "rb"))
# test_firing1 = pickle.load(open(f'{results_dir}/hcp_testretest/secondary_analysis/test_firing_state1.pkl', "rb"))

# mean_param_rankings_h, mean_param_fingerprints_h = tr.retest_correlation(np.array(test_firing['h_mean']), np.array(retest_firing['h_mean']))
# mean_param_rankings_y, mean_param_fingerprints_y = tr.retest_correlation(np.array(test_firing['y_mean']), np.array(retest_firing['y_mean']))
# mean_param_rankings_x, mean_param_fingerprints_x = tr.retest_correlation(np.array(test_firing['x_mean']), np.array(retest_firing['x_mean']))
# sns.heatmap(np.array(mean_param_fingerprints_x).astype('float64'), cmap='RdBu_r', vmax=1, vmin=-1)

# sns.distplot(mean_param_rankings_y)
# sns.distplot(mean_param_rankings_h)
# sns.distplot(mean_param_rankings_x)
# np.median(mean_param_rankings_h)

# y_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['y_mean']), np.array(retest_firing['y_mean']))
# h_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['h_mean']), np.array(retest_firing['h_mean']))
# x_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['x_mean']), np.array(retest_firing['x_mean']))
# rec_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['rec_mean']), np.array(retest_firing['rec_mean']))
# inter_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['inter_mean']), np.array(retest_firing['inter_mean']))
# sns.distplot(h_reliabilities)
# sns.distplot(y_reliabilities)
# sns.distplot(x_reliabilities)
# sns.distplot(rec_reliabilities)
# sns.distplot(inter_reliabilities)

# np.median(rec_reliabilities)
# calculate firing rate for severa parameter sets for each subject and mean them
counter=0
for j in range(4,5):
    test_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
    retest_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
    for session, firing_dict in zip(['test', 'retest'], [test_firing, retest_firing]):
        for subject in subjects:
            print(counter)
            y_l, h_l, x_l, rec_l, inter_l =[],[],[], [], []
            for i in range(j+1):
                try:
                    ParaE = np.loadtxt(f'{results_dir}/hcp_testretest/{session}/output_{subject}_{i}.txt')
                    ParaE = np.atleast_2d(ParaE).T
                    y,h,x,rec,inter = sim.firing_rate(ParaE, SC, NState)
                    y_l.append(y)
                    h_l.append(h)
                    x_l.append(x)
                    rec_l.append(rec)
                    inter_l.append(inter)
                except:
                    pass
            y = np.mean(y_l,axis=0)
            h = np.mean(h_l,axis=0)
            x = np.mean(x_l,axis=0)
            rec = np.mean(rec_l,axis=0)
            inter = np.mean(inter_l,axis=0)
            firing_dict['y_mean'].append(np.mean(y, axis=1))
            firing_dict['y_sd'].append(np.std(y, axis=1))
            firing_dict['h_mean'].append(np.mean(h, axis=1))
            firing_dict['h_sd'].append(np.std(h, axis=1))
            firing_dict['x_mean'].append(np.mean(x, axis=1))
            firing_dict['x_sd'].append(np.std(x, axis=1))
            firing_dict['rec_mean'].append(np.mean(rec, axis=1))
            firing_dict['rec_sd'].append(np.std(rec, axis=1))
            firing_dict['inter_mean'].append(np.mean(inter, axis=1))
            firing_dict['inter_sd'].append(np.std(inter, axis=1))
            counter+=1
    pickle.dump(test_firing, open(f'{results_dir}/hcp_testretest/secondary_analysis/test_firing_state_multipara{j+1}.pkl', "wb"))
    pickle.dump(retest_firing, open(f'{results_dir}/hcp_testretest/secondary_analysis/retest_firing_state_multipara{j+1}.pkl', "wb"))