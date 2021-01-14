import numpy as np
import os
import sys
sys.path.append('/users/k1201869/wang_model/code')
import pingouin as pg
import pandas as pd
from functions import test_retest_funcs as tr
from functions import simulation as sim
import pickle

home_dir = '/users/k1201869/wang_model'
data_dir = f'{home_dir}/data'
results_dir = f'{home_dir}/results'
session = 'test'

#subject
subject_idx = int(sys.argv[1])-1
# subject_list = pd.read_table("/users/k1201869/wang_model/data/subjects_testretest.list")
subject_list = pd.read_table("/users/k1201869/wang_model/data/subjects.list")

subject = subject_list.iloc[subject_idx].values[0]
print(f'subject {subject}')

# calculate firing rate for severa parameter sets for each subject and mean them
counter=0
SC = np.loadtxt(open(f"{home_dir}/data/hcp_testretest/dti_collated_retest/group_retest_SC.csv", "rb"), delimiter=" ")
# SC = np.loadtxt(open(f"{home_dir}/data/hcp_testretest/dti_collated_retest/{subject}_SC.csv", "rb"), delimiter=" ")

SC = (SC/np.max(np.max(SC)))*0.2
NState = 0
firing_dict = {} #{'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
y_l, h_l, x_l, rec_l, inter_l, FC_l =[], [], [], [], [], []
for g in range(1,7):
    for i in range(g):
        print(counter)
        try:
            ParaE = np.loadtxt(f'{home_dir}/results/hcpep/testretestSC/output_{subject}_{i}.txt')[:-1]
            # ParaE = np.loadtxt(f'{home_dir}/results/hcp_testretest/indiv_connect/{session}/output_{subject}_{i}.txt')[:-1]
            ParaE = np.atleast_2d(ParaE).T
            y,h,x,rec,inter,FCsim = sim.firing_rate(ParaE, SC, NState, TBOLD=0.72)
            y_l.append(y)
            h_l.append(h)
            x_l.append(x)
            rec_l.append(rec)
            inter_l.append(inter)
            FC_l.append(FCsim.real)
        except:
            pass
        counter+=1
    try:
        y = np.mean(y_l,axis=0)
        h = np.mean(h_l,axis=0)
        x = np.mean(x_l,axis=0)
        rec = np.mean(rec_l,axis=0)
        inter = np.mean(inter_l,axis=0)
        FC = np.mean(FC_l, axis=0)

        firing_dict['y_mean'] = np.mean(y, axis=1)
        firing_dict['y_sd']=np.std(y, axis=1)
        firing_dict['h_mean']=np.mean(h, axis=1)
        firing_dict['h_sd']=np.std(h, axis=1)
        firing_dict['x_mean']=np.mean(x, axis=1)
        firing_dict['x_sd']=np.std(x, axis=1)
        firing_dict['rec_mean']=np.mean(rec, axis=1)
        firing_dict['rec_sd']=np.std(rec, axis=1)
        firing_dict['inter_mean']=np.mean(inter, axis=1)
        firing_dict['inter_sd']=np.std(inter, axis=1)
        firing_dict['FCsim'] = FC
        
    except:
        pass
        

    # pickle.dump(firing_dict, open(f'{results_dir}/hcp_testretest/indiv_connect/secondary_analysis/{session}/firing_mean{g}_indiv_para_{subject}.pkl', "wb"))
    pickle.dump(firing_dict, open(f'{results_dir}/hcpep/testretestSC/secondary_analysis/firing_mean{g}_indiv_para_{subject}.pkl', "wb"))
