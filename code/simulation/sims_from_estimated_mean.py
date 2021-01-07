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
subjects=np.array(subjects)
#SC
SC = np.loadtxt(open(f"{home_dir}/data/hcp_testretest/dti_collated_retest/group_retest_SC.csv", "rb"), delimiter=" ")
SC = (SC/np.max(np.max(SC)))*0.2

# calculate firing rate for single mean parameter set for each subject and mean them
counter=0
test_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
retest_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
data_path = f'/users/k1201869/wang_model/results/hcp_testretest/groupSC'
NState=0
for session, firing_dict in zip(['test', 'retest'], [test_firing, retest_firing]):
    for subject in subjects:
        print(counter)
        y_l, h_l, x_l, rec_l, inter_l =[],[],[], [], []
        mean_test_params, mean_retest_params = tr.load_parameters([subject], 'mean', [0,138], data_path)
        if session == 'test': ParaE  = mean_test_params
        if session == 'retest': ParaE  = mean_retest_params
        ParaE = np.atleast_2d(ParaE).T
        y,h,x,rec,inter = sim.firing_rate(ParaE, SC, NState)
        y_l.append(y)
        h_l.append(h)
        x_l.append(x)
        rec_l.append(rec)
        inter_l.append(inter)
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
pickle.dump(test_firing, open(f'{results_dir}/hcp_testretest/secondary_analysis/groupSC/test_firing_state_meanpara.pkl', "wb"))
pickle.dump(retest_firing, open(f'{results_dir}/hcp_testretest/secondary_analysis/groupSC/retest_firing_state_meanpara.pkl', "wb"))

mean_test_firing = pickle.load(open(f'/users/k1201869/wang_model/results/hcp_testretest/secondary_analysis/groupSC/test_firing_state_meanpara.pkl', "rb"))
mean_retest_firing = pickle.load(open(f'/users/k1201869/wang_model/results/hcp_testretest/secondary_analysis/groupSC/retest_firing_state_meanpara.pkl', "rb"))
mean_test_firing = pickle.load(open(f'/users/k1201869/wang_model/results/hcp_testretest/secondary_analysis/indiv_connect/test_firing.pkl', "rb"))
mean_retest_firing = pickle.load(open(f'/users/k1201869/wang_model/results/hcp_testretest/secondary_analysis/indiv_connect/retest_firing.pkl', "rb"))


y_reliabilities = tr.retest_reliability(list(subjects), np.array(mean_test_firing['y_mean']), np.array(mean_retest_firing['y_mean']))
h_reliabilities = tr.retest_reliability(list(subjects), np.array(mean_test_firing['h_mean']), np.array(mean_retest_firing['h_mean']))
x_reliabilities = tr.retest_reliability(list(subjects), np.array(mean_test_firing['x_mean']), np.array(mean_retest_firing['x_mean']))
rec_reliabilities = tr.retest_reliability(list(subjects), np.array(mean_test_firing['rec_mean']), np.array(mean_retest_firing['rec_mean']))
inter_reliabilities = tr.retest_reliability(list(subjects), np.array(mean_test_firing['inter_mean']), np.array(mean_retest_firing['inter_mean']))

np.median(y_reliabilities)
np.median(h_reliabilities)
np.median(x_reliabilities)
np.median(rec_reliabilities)
np.median(inter_reliabilities)

