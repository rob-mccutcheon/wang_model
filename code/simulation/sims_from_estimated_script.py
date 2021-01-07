import numpy as np
import os
import sys
sys.path.append('/users/k1201869/wang_model/code')
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

mean_test_firing = pickle.load(open(f'/users/k1201869/wang_model/results/hcp_testretest/secondary_analysis/indiv_connect/mean_test_firing.pkl', "rb"))
mean_retest_firing = pickle.load(open(f'/users/k1201869/wang_model/results/hcp_testretest/secondary_analysis/indiv_connect/mean_retest_firing.pkl', "rb"))
