import numpy as np
import os
import seaborn as sns
import pingouin as pg
import pandas as pd
import  matplotlib.pyplot as plt
from functions import test_retest_funcs as tr
from functions import simulation as sim
import pickle
import bct as bct

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

sns.scatterplot(test_strengths[i,:], test_cc[i,:])

test_cc = np.zeros([45,68])
retest_cc = np.zeros([45,68])
for i in range(45):
    # ci = bct.modularity_louvain_und_sign(test_cmzs[i,:,:])
    # test_cc[i,:] = bct.participation_coef(test_cmzs[i,:,:], ci[0])
    test_cc[i,:] = bct.efficiency_wei(test_cmzs[i,:,:], local=True)
    # rci = bct.modularity_louvain_und_sign(retest_cmzs[i,:,:])
    retest_cc[i,:] = bct.efficiency_wei(retest_cmzs[i,:,:], local=True)
    # retest_cc[i,:] = bct.clustering_coef_wu(retest_cmzs[i,:,:])

# Load Parameters - w:[0,68], i:[68, 136], g:[136, 137], s:[137, 138], corr = [138,139]
parameter_choice = [0,68]
match_test_params, match_retest_params = tr.load_parameters(subjects, 'match', parameter_choice, data_path=f'{results_dir}/hcp_testretest/groupSC',num_sets=1,test_idx=0)
rand_test_params, rand_retest_params = tr.load_parameters(subjects, 'rand', parameter_choice, f'{results_dir}/hcp_testretest')
max_test_params, max_retest_params = tr.load_parameters(subjects, 'max', parameter_choice, f'{results_dir}/hcp_testretest')
mean_test_params, mean_retest_params = tr.load_parameters(subjects, 'mean', parameter_choice, f'{results_dir}/hcp_testretest/groupSC')


#long params
test_params = []
retest_params = []
test_idxs = []
retest_idxs=[]
for subject in subjects:
    test_param = []
    retest_param = []
    for i in range(6):
        try:
            test_param.append(np.loadtxt(f'{results_dir}/hcp_testretest/single_params/test/output_{subject}_{i}.txt'))
            if i==5:
                test_idxs.append(5)
        except:
            test_idxs.append(i)
            break
    for j in range(6):    
        try:
            retest_param.append(np.loadtxt(f'{results_dir}/hcp_testretest/single_params/retest/output_{subject}_{j}.txt'))
            if j==5:
                retest_idxs.append(5)
        except:
            retest_idxs.append(j)
            break
    if i!=0 and j!=0:
        test_params.append(np.mean((np.array(test_param)[:, parameter_choice[0]:parameter_choice[1]]), axis=0))
        retest_params.append(np.mean((np.array(retest_param)[:, parameter_choice[0]:parameter_choice[1]]), axis=0))
tr.retest_reliability(list(np.arange(len(test_params))), np.array(test_params), np.array(retest_params))
subjects = 

        reliability_df = pd.DataFrame({'subjects': subjects+subjects,
                                    'session': len(subjects)*['test']+len(subjects)*['retest'],
                                    'strength': np.hstack([test_data[:,i], retest_data[:,i]])})

# Look at correlation between each test scan and the retest (each scan a vector of strength values or parameters)
cc_rankings, cc_fingerprints = tr.retest_correlation(test_cc, retest_cc)
strength_rankings, strength_fingerprints = tr.retest_correlation(test_strengths, retest_strengths)
match_param_rankings, match_param_fingerprints = tr.retest_correlation(match_test_params, match_retest_params)
rand_param_rankings, rand_param_fingerprints = tr.retest_correlation(rand_test_params, rand_retest_params)
max_param_rankings, max_param_fingerprints = tr.retest_correlation(max_test_params, max_retest_params)
mean_param_rankings, mean_param_fingerprints = tr.retest_correlation(mean_test_params, mean_retest_params)

#strength heatmap
sns.heatmap(strength_fingerprints, cmap='Reds', vmax=0.95, vmin=0.7)

#strength rankings kde
sns.distplot(strength_rankings, kde_kws={'cut':0})
plt.xlabel('ranking')
np.median(strength_rankings)

# param heatmap
sns.heatmap(mean_param_fingerprints, cmap='RdBu_r', vmax=1, vmin=-1)

# param rankings kde
sns.set_theme(style="whitegrid")
sns.distplot(match_param_rankings, kde_kws={'cut':0}, label='single')
# sns.distplot(rand_param_rankings, kde_kws={'cut':0}, label='random')
sns.distplot(mean_param_rankings, kde_kws={'cut':0}, label='mean')
sns.distplot(max_param_rankings, kde_kws={'cut':0}, label='max')
plt.xlabel('ranking')
plt.legend()
for i in range(5):
    match_test_params, match_retest_params = tr.load_parameters(subjects, 'match', parameter_choice, f'{results_dir}/hcp_testretest',test_idx=i)
    match_param_reliabilities = tr.retest_reliability(subjects, match_test_params, match_retest_params)
    print(np.median(match_param_reliabilities))
np.median(mean_param_rankings)



# test retest of individual nodes
max,mean,rand, match = [], [], [], []
for i in range(5):
    print(i)
    match_test_params, match_retest_params = tr.load_parameters(subjects, 'match', parameter_choice, f'{results_dir}/hcp_testretest/short_iterations', num_sets=(i+1), test_idx=i)
    # rand_test_params, rand_retest_params = tr.load_parameters(subjects, 'rand', parameter_choice, f'{results_dir}/hcp_testretest', num_sets=(i+1))
    # max_test_params, max_retest_params = tr.load_parameters(subjects, 'max', parameter_choice, f'{results_dir}/hcp_testretest', num_sets=(i+1))
    mean_test_params, mean_retest_params = tr.load_parameters(subjects, 'mean', parameter_choice, f'{results_dir}/hcp_testretest/short_iterations', num_sets=(i+1))
    # cc_reliabilities = tr.retest_reliability(subjects, test_cc, retest_cc)

    # strength_reliabilities = tr.retest_reliability(subjects, test_strengths, retest_strengths)
    match_param_reliabilities = tr.retest_reliability(subjects, match_test_params, match_retest_params)
    # rand_param_reliabilities = tr.retest_reliability(subjects, rand_test_params, rand_retest_params)
    # max_param_reliabilities = tr.retest_reliability(subjects, max_test_params, max_retest_params)
    mean_param_reliabilities = tr.retest_reliability(subjects, mean_test_params, mean_retest_params)
    match.append(np.median(match_param_reliabilities))
    # max.append(np.median(max_param_reliabilities))
    mean.append(np.median(mean_param_reliabilities))
    # rand.append(np.median(rand_param_reliabilities))
a=tr.retest_reliability(subjects, test_strengths, test_strengths)
# Strenght ICC
sns.distplot(strength_reliabilities)
plt.xlabel('ICC')
sns.distplot(cc_reliabilities)
np.median(strength_reliabilities)
np.median(cc_reliabilities)
sns.scatterplot(test_cc[0,:],test_strengths[0,:])

# Param ICCs
sns.distplot(strength_reliabilities)
sns.distplot(match_param_reliabilities, label='single')
# sns.distplot(rand_param_reliabilities, label='random')
sns.distplot(mean_param_reliabilities[0:68], label='mean')
sns.distplot(max_param_reliabilities, label='max')
plt.legend()
plt.xlabel('ICC')
np.median(mean_param_reliabilities)
# How increasing the numbr of parameter sets improves ICC
sns.lineplot(np.arange(1,7), mean, label='mean')
sns.lineplot(np.arange(1,7), match, label='single')
# sns.lineplot(np.arange(1,7), max, label='max')
plt.xlabel('number of parameter sets')
plt.ylabel('median ICC across all nodes')


# Reliability of simulation findings
parameter_choice = [0,138]
SC = np.loadtxt(open(f"/users/k1201869/wang_model/data/SC_dk.csv", "rb"), delimiter=",")
SC = (SC/np.max(np.max(SC)))*0.2
NState = 0
j=0
for j in range(5):
    print(j)
    # match_test_params, match_retest_params = tr.load_parameters(subjects, 'match', parameter_choice, f'{results_dir}/hcp_testretest', num_sets=(j+1), test_idx=j)
    # max_test_params, max_retest_params = tr.load_parameters(subjects, 'max', parameter_choice, f'{results_dir}/hcp_testretest', num_sets=(j+1))
    # mean_test_params, mean_retest_params = tr.load_parameters(subjects, 'mean', parameter_choice, f'{results_dir}/hcp_testretest', num_sets=(j+1))
    mean_test_params_fix, mean_retest_params_fix = tr.load_parameters(subjects, 'mean', parameter_choice, f'{results_dir}/hcp_testretest', num_sets=(j+1))
    mean_test_params_fix[:,136]=1.67
    mean_retest_params_fix[:,136]=1.67
    # param_dict = {'match_test_params':match_test_params, 'match_retest_params':match_retest_params, 'max_test_params':max_test_params, 'max_retest_params':max_retest_params, 'mean_test_params':mean_test_params, 'mean_retest_params':mean_retest_params, 'mean_test_params_fix':mean_test_params_fix, 'mean_retest_params_fix':mean_retest_params_fix}
    param_dict = {'mean_test_params_fix':mean_test_params_fix, 'mean_retest_params_fix':mean_retest_params_fix}

    # for test_params, retest_params in zip(['match_test_params','max_test_params','mean_test_params', 'mean_test_params_fix'],['match_retest_params','max_retest_params','mean_retest_params', 'mean_retest_params_fix']):
    for test_params, retest_params in zip([ 'mean_test_params_fix'],['mean_retest_params_fix']):    
        test_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
        retest_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
        i=0
        for param_set, firing_dict in zip([param_dict[test_params], param_dict[retest_params]], [test_firing, retest_firing]):
            for ParaE in param_set:
                print(i)
                ParaE = np.atleast_2d(ParaE).T
                y,h,x,rec, inter = sim.firing_rate(ParaE, SC, NState)
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
                i=i+1
        pickle.dump(test_firing, open(f'{results_dir}/hcp_testretest/secondary_analysis/test_firing_state_{test_params}_{j}.pkl', "wb"))
        pickle.dump(retest_firing, open(f'{results_dir}/hcp_testretest/secondary_analysis/retest_firing_state_{test_params}_{j}.pkl', "wb"))


i=0
match_test_params, match_retest_params = tr.load_parameters(subjects, 'match', parameter_choice, f'{results_dir}/hcp_testretest', num_sets=(j+1))
param_dict = {'match_test_params':match_test_params, 'match_retest_params':match_retest_params}
test_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
retest_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
for param_set, firing_dict in zip([match_test_params, match_retest_params], [test_firing, retest_firing]):
    for ParaE in param_set:
        print(i)
        ParaE = np.atleast_2d(ParaE).T
        y,h,x,rec, inter = sim.firing_rate(ParaE, SC, NState)
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
        i=i+1

#
import copy
test_firing_match = copy.deepcopy(test_firing)
retest_firing_match = copy.deepcopy(retest_firing)

for i in range(45):
    print(test_firing_match['y_mean'][i][0])
test_firing_match['y_mean'][0]

# How simulation reliabilities vary for a single run based on the use of mean/max or rand parameters
test_params = 'mean_test_params'#,'max_test_params','mean_test_params']:
test_firing = pickle.load(open(f'{results_dir}/hcp_testretest/secondary_analysis/test_firing_state_{test_params}_5.pkl', "rb"))
retest_firing = pickle.load(open(f'{results_dir}/hcp_testretest/secondary_analysis/retest_firing_state_{test_params}_5.pkl', "rb"))
y_l,h_l,x_l, rec_l, inter_l =[],[],[],[],[]
y_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['y_mean']), np.array(retest_firing['y_mean']))
h_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['h_mean']), np.array(retest_firing['h_mean']))
x_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['x_mean']), np.array(retest_firing['x_mean']))
rec_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['rec_mean']), np.array(retest_firing['rec_mean']))
inter_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['inter_mean']), np.array(retest_firing['inter_mean']))
y_l.append(np.median(rec_reliabilities))
np.median(x_reliabilities)

# Plot for each of max, meann and rand by changing at line 128
sns.distplot(h_reliabilities, label='Firing rate')
sns.distplot(y_reliabilities, label = 'Synaptic activity')
sns.distplot(x_reliabilities, label = 'Input Current')
sns.distplot(rec_reliabilities, label = 'Recurrent input')
sns.distplot(inter_reliabilities, label ='Interareal input')
plt.legend()
plt.title(test_params)
plt.xlabel('ICC')


# calculate firing rate for several parameter sets for each subject and mean them
counter=0
j=0
for j in range(5):
    test_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
    retest_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
    for session, firing_dict in zip(['test', 'retest'], [test_firing, retest_firing]):
        for subject in subjects:
            print(counter)
            y_l, h_l, x_l, rec_l, inter_l =[],[],[], [], []
            for i in range(j+1):
                try:
                    ParaE = np.loadtxt(f'{results_dir}/hcp_testretest/{session}/output_{subject}_{i}.txt')[:-1]
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
    pickle.dump(test_firing, open(f'{results_dir}/hcp_testretest/secondary_analysis/test_firing_state_multipara_correct_{j+1}.pkl', "wb"))
    pickle.dump(retest_firing, open(f'{results_dir}/hcp_testretest/secondary_analysis/retest_firing_state_multipara_correct_{j+1}.pkl', "wb"))

# Calculate the ICC varying the number of simulation runs that are meaned
y_l,h_l,x_l, rec_l, inter_l =[],[],[],[],[]
for j in range(1,6):
    retest_firing = pickle.load(open(f'{results_dir}/hcp_testretest/secondary_analysis/retest_firing_state_multipara_correct_{j}.pkl', "rb"))
    test_firing = pickle.load(open(f'{results_dir}/hcp_testretest/secondary_analysis/test_firing_state_multipara_correct_{j}.pkl', "rb"))
    y_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['y_mean']), np.array(retest_firing['y_mean']))
    h_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['h_mean']), np.array(retest_firing['h_mean']))
    x_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['x_mean']), np.array(retest_firing['x_mean']))
    rec_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['rec_mean']), np.array(retest_firing['rec_mean']))
    inter_reliabilities = tr.retest_reliability(subjects, np.array(test_firing['inter_mean']), np.array(retest_firing['inter_mean']))
    y_l.append(np.median(y_reliabilities))
    h_l.append(np.median(h_reliabilities))
    x_l.append(np.median(x_reliabilities))
    rec_l.append(np.median(rec_reliabilities))
    inter_l.append(np.median(inter_reliabilities))

# Plot the ICC distribution for when we combien the max number of runs (i.5. 5)
sns.distplot(h_reliabilities, label='Firing rate')
sns.distplot(y_reliabilities, label = 'Synaptic activity')
sns.distplot(x_reliabilities, label = 'Input Current')
sns.distplot(rec_reliabilities, label = 'Recurrent input')
sns.distplot(inter_reliabilities, label ='Interareal input')
plt.legend()
plt.xlabel('ICC')

np.median(rec_reliabilities)
#  How much of an ICC boost are we getting by combining runs
sns.lineplot(np.arange(1,6), h_l[:], label='Firing rate')
sns.lineplot(np.arange(1,6), y_l[:], label = 'Synaptic activity')
sns.lineplot(np.arange(1,6), x_l[:], label = 'Input Current')
sns.lineplot(np.arange(1,6), rec_l[:], label = 'Recurrent input')
sns.lineplot(np.arange(1,6), inter_l[:], label ='Interareal input')
plt.legend()
plt.ylabel('median ICC')
plt.xlabel('number of parameter sets')

mean_param_rankings, mean_param_fingerprints = tr.retest_correlation(np.array(test_firing['y_mean']), np.array(retest_firing['y_mean']))
a= np.corrcoef(np.array(test_firing['y_mean']).T, np.array(retest_firing['y_mean']).T)
sns.heatmap(mean_param_fingerprints)
np.array(retest_firing['y_mean']).T.shape
a= pickle.load(open(f'{results_dir}/hcp_testretest/secondary_analysis/retest_firing_state_multipara5.pkl', "rb"))


# what about inter to recurren ratio
sim_dic = {}
for sim1 in ['rec_mean', 'inter_mean', 'y_mean', 'x_mean', 'h_mean']:
    for sim2 in ['rec_mean', 'inter_mean', 'y_mean', 'x_mean', 'h_mean']:
        test_rec_ratio = np.array(test_firing[sim1])/np.array(test_firing[sim2])
        retest_rec_ratio = np.array(retest_firing[sim1])/np.array(retest_firing[sim2])
        rec_ratio_reliabilities = tr.retest_reliability(subjects, test_rec_ratio, retest_rec_ratio)
        sim_dic[f'{sim1}_{sim2}'] = np.median(rec_ratio_reliabilities)
sns.distplot(rec_ratio_reliabilities)
np.median(rec_ratio_reliabilities)

subjects

data_path = f'{home_dir}/results/hcp_testretest'

corr2 = []
subjects2=[]
for subject in subjects2:
    try:
        params = np.loadtxt(f'{data_path}/test/output_{subject}_0.txt')
        # subjects2.append(subject)
        corr2.append(params[-1])
    except:
        pass

group = list(np.zeros(len(corr)))+list(np.ones(len(corr)))
sns.scatterplot(group, (corr+corr2))





# firing rate - mean

subjects = subjects[:26]
for i in range(1,6):
    a=[]
    for item in ['x']:#, 'y', 'h', 'rec', 'inter']:
        test_mean = []
        retest_mean = []
        for subject in subjects:
            #firing rates
            test_firing_dict = pickle.load(open(f'{results_dir}/hcp_testretest/groupSC/secondary_analysis/test/firing_mean{i}_indiv_para_{subject}.pkl', "rb"))
            retest_firing_dict = pickle.load(open(f'{results_dir}/hcp_testretest/groupSC/secondary_analysis/retest/firing_mean{i}_indiv_para_{subject}.pkl', "rb"))
            test_mean.append(test_firing_dict[f'{item}_mean'])
            retest_mean.append(retest_firing_dict[f'{item}_mean'])
        a.append(tr.retest_reliability(subjects, np.array(test_mean), np.array(retest_mean)))
        print(np.median(a))
        # print(tr.retest_reliability(subjects, np.atleast_2d(np.mean(np.array(test_mean),axis=1)).T, np.atleast_2d(np.mean(np.array(retest_mean), axis=1)).T))


tr.retest_reliability(subjects, np.atleast_2d(np.mean(np.array(test_mean),axis=1)).T, np.atleast_2d(np.mean(np.array(retest_mean), axis=1)).T)

np.mean(np.array(test_mean),axis=1).shape
sns.distplot(a[4])

