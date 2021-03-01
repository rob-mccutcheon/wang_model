from functions import wang_functions_imag_fix as wf
from functions import simulation as sim
from functions import utils
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy.stats import pearsonr
import bct
from functions import test_retest_funcs as tr

# Get empirical SC ad FC
home_dir = '/users/k1201869/wang_model/data/glucog'
subject='105923'
subject='glucog01a'
data_dir = f'{home_dir}'
#SC = np.loadtxt(open(f"{home_dir}/dti_collated_retest/{subject}_SC.csv", "rb"), delimiter=" ")
SC = np.loadtxt(open(f"/users/k1201869/wang_model/data/hcp_testretest/dti_collated_retest/group_retest_SC.csv", "rb"), delimiter=" ")
delete_nodes = [0,30,31,34,64,65]
SC = np.delete(SC, delete_nodes, 0)
SC = np.delete(SC, delete_nodes, 1)
SC = (SC/np.max(np.max(SC)))*0.2

FC = np.loadtxt(open(f'{data_dir}/{subject}_cm.txt', "rb"), delimiter=" ")
FC = FC[34:][:,34:]
np.fill_diagonal(FC,1)

# Load fitted parameters
results_dir = '/users/k1201869/wang_model/results/glucog/testretestSC'
Para_E = np.loadtxt(open(f'{results_dir}/output_{subject}_0.txt'))[:-1]
Para_E = np.atleast_2d(Para_E).T
FC_mask = np.tril(np.ones([np.size(SC, 0), np.size(SC, 0)]), 0)
y = FC[~FC_mask.astype('bool')]

# Simulate FC
Nstate=0
def funcP(Para_E, Prior_E=0,SC=SC,y=y,FC_mask=FC_mask,Nstate=Nstate,Tepochlong=14.4,TBOLD=0.72,ifRepara=0):
    FC_simR, CC_check = wf.CBIG_MFMem_rfMRI_nsolver_eul_sto(Para_E,Prior_E,SC,y,FC_mask,Nstate,Tepochlong,TBOLD,ifRepara)
    return FC_simR, CC_check

# Para_E = np.atleast_2d(mean_test_params[0,:]).T
[h_output, CC_check] = funcP(Para_E)

# Empirical SC heatmap
figures_dir = '/users/k1201869/wang_model/results/figures/pilot_figures'
sns.heatmap(SC, cmap='magma', xticklabels="", yticklabels="")
plt.xlabel('68 cortical regions')
plt.ylabel('68 cortical regions')
plt.title('Anatomical Connectivity')
plt.savefig(f'{figures_dir}/SC_heat.png')

# Empirical FC heatmap
sns.heatmap(utils.vec2mat(y), cmap='magma', xticklabels="", yticklabels="")
plt.xlabel('68 cortical regions')
plt.ylabel('68 cortical regions')
plt.title('Empirical Functional Connectivity')
plt.savefig(f'{figures_dir}/FC_emp_heat.png')

# Simulated FC heatmap
sns.heatmap(utils.vec2mat(h_output), cmap='magma', xticklabels="", yticklabels="")
plt.xlabel('68 cortical regions')
plt.ylabel('68 cortical regions')
plt.title('Simulated Functional Connectivity')
plt.savefig(f'{figures_dir}/FC_sim_heat.png')

pearsonr(h_output,y)
pearsonr(SC1[np.triu_indices(SC.shape[0])], SC2[np.triu_indices(SC.shape[0])])
pearsonr(h_output3, h_output1)

# Sim vs empirical
plt.figure(figsize=(5,5))
a = sns.regplot(x=y, y=h_output, scatter_kws={'s':0.2})
a.set(xlim=(0,1))
a.set(ylim=(0,1))
plt.xlabel('Empirical FC')
plt.ylabel('Simulated FC')
plt.savefig(f'{figures_dir}/emp_sim_scatter.png')



subject='105923'
subject='146129'
subject='103818'
h_outputs = []
for i in range(4):
    Para_E = np.loadtxt(open(f'{results_dir}/output_{subject}_{i}.txt'))[:-1]
    Para_E = np.atleast_2d(Para_E).T
    SC = np.loadtxt(open(f"{home_dir}/dti_collated_retest/{subject}_SC.csv", "rb"), delimiter=" ")
    SC = (SC/np.max(np.max(SC)))*0.2
    Nstate=0
    def funcP(Para_E, Prior_E=0,SC=SC,y=y,FC_mask=FC_mask,Nstate=Nstate,Tepochlong=14.4,TBOLD=0.72,ifRepara=0):
        FC_simR, CC_check = wf.CBIG_MFMem_rfMRI_nsolver_eul_sto(Para_E,Prior_E,SC,y,FC_mask,Nstate,Tepochlong,TBOLD,ifRepara)
        return FC_simR, CC_check
    [h_output, CC_check] = funcP(Para_E)
    h_outputs.append(h_output)
h_outputs.append(np.mean(h_outputs,axis=0))
h_outputs.append(y)
sns.heatmap(utils.vec2mat(h_output2), cmap='magma', xticklabels="", yticklabels="")
sns.heatmap(np.corrcoef(h_outputs).real)
pearsonr((h_output2+h_output0)/2, y)
shs

np.corrcoef(h_outputs).real
# mopdular strcuture
coms, q = bct.community_louvain(FC,gamma=1.05,B='negative_sym', seed=123)
idx=[]
for i in range(np.max(coms)):
    idx=idx+list(np.where(coms==i)[0])
np.fill_diagonal(FC,0)
sns.heatmap(FC[idx,:][:,idx], cmap='viridis',vmax=0.8)



subjects = np.loadtxt(open('/users/k1201869/wang_model/data/subjects_testretest.list'))
home_dir = '/users/k1201869/wang_model/results/hcp_testretest/'
results_dir = f'{home_dir}/groupSC'

retest = []
test=[]
subjects_valid=[]
for i,subject in enumerate(subjects):
    subject=str(int(subject))
    try:
        b= np.loadtxt(open(f'{results_dir}/retest/output_{subject}_0.txt'))[:-1]
        c= np.loadtxt(open(f'{results_dir}/test/output_{subject}_0.txt'))[:-1]
        retest.append(b[:68])
        test.append(c[:68])
        subjects_valid.append(subject)
    except:
        print(i+1)
        print(subject)

a=tr.retest_reliability(subjects_valid, np.array(test), np.array(retest))
np.median(a)
sns.distplot(a)
sns.heatmap(np.corrcoef(np.array(a)))


# Reliability of simulation findings
parameter_choice = [68,138]
data_path = f'/users/k1201869/wang_model/results/hcp_testretest/indiv_connect'
mean_test_params, mean_retest_params = tr.load_parameters(subjects.astype(int), 'mean', parameter_choice, data_path)
parameter_choice2 = [68,136]
mean_test_params2, mean_retest_params2 = tr.load_parameters(subjects.astype(int), 'mean', parameter_choice2, data_path)


mean_retest_params=np.mean(mean_retest_params, axis=1)
mean_test_params=np.mean(mean_test_params, axis=1)

a=tr.retest_reliability(list(subjects.astype(int)), np.array(mean_test_params2), np.array(mean_retest_params2))
a=tr.retest_reliability(list(subjects.astype(int)), np.array(mean_test_params/mean_test_params2), np.array(mean_retest_params/mean_retest_params2))

a=tr.retest_reliability(list(subjects.astype(int)), np.atleast_2d(mean_test_params).T, np.atleast_2d(mean_retest_params).T)

sns.distplot(a)

np.median(a)
num_sets=5
NState = 0
j=0

mean_test_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
mean_retest_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}

i=0
results_dir = '/users/k1201869/wang_model/results/hcp_testretest/indiv_connect/'
for subject in subjects_valid:#subjects.astype(int):
    # SC = np.loadtxt(open(f"/users/k1201869/wang_model/data/hcp_testretest/dti_collated_retest/{subject}_SC.csv", "rb"), delimiter=" ")
    SC = np.loadtxt(open(f"/users/k1201869/wang_model/data/hcp_testretest/dti_collated_retest/group_retest_SC.csv", "rb"), delimiter=" ")
    SC = (SC/np.max(np.max(SC)))*0.2
    test_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
    retest_firing = {'y_mean':[],'y_sd':[],'h_mean':[],'h_sd':[],'x_mean':[],'x_sd':[], 'rec_mean':[],'rec_sd':[], 'inter_mean':[],'inter_sd':[]}
    for session, firing_dict in zip(['test', 'retest'], [test_firing, retest_firing]):
        for i in range(1):#5):
            try:
                params = np.loadtxt(open(f'{results_dir}/{session}/output_{subject}_{i}.txt'))[:-1]
            except:
                continue
            ParaE = np.atleast_2d(params).T
            y,h,x,rec, inter = sim.firing_rate(ParaE, SC, NState)
            firing_dict['y_mean'].append(np.mean(y, axis=1))
            firing_dict['h_mean'].append(np.mean(h, axis=1))
            firing_dict['x_mean'].append(np.mean(x, axis=1))
            firing_dict['rec_mean'].append(np.mean(rec, axis=1))
            firing_dict['inter_mean'].append(np.mean(inter, axis=1))
            firing_dict['y_sd'].append(np.std(y, axis=1))
            firing_dict['h_sd'].append(np.std(h, axis=1))
            firing_dict['x_sd'].append(np.std(x, axis=1))
            firing_dict['rec_sd'].append(np.std(rec, axis=1))
            firing_dict['inter_sd'].append(np.std(inter, axis=1))
            print(i)
    print(subject)
    mean_test_firing['y_mean'].append(np.mean(test_firing['y_mean'], axis=0))
    mean_test_firing['h_mean'].append(np.mean(test_firing['h_mean'], axis=0))
    mean_test_firing['x_mean'].append(np.mean(test_firing['x_mean'], axis=0))
    mean_test_firing['rec_mean'].append(np.mean(test_firing['rec_mean'], axis=0))
    mean_test_firing['inter_mean'].append(np.mean(test_firing['inter_mean'], axis=0))
    mean_test_firing['y_sd'].append(np.mean(test_firing['y_sd'], axis=0))
    mean_test_firing['h_sd'].append(np.mean(test_firing['h_sd'], axis=0))
    mean_test_firing['x_sd'].append(np.mean(test_firing['x_sd'], axis=0))
    mean_test_firing['rec_sd'].append(np.mean(test_firing['rec_sd'], axis=0))
    mean_test_firing['inter_sd'].append(np.mean(test_firing['inter_sd'], axis=0))
    mean_retest_firing['h_mean'].append(np.mean(retest_firing['h_mean'], axis=0))
    mean_retest_firing['x_mean'].append(np.mean(retest_firing['x_mean'], axis=0))
    mean_retest_firing['rec_mean'].append(np.mean(retest_firing['rec_mean'], axis=0))
    mean_retest_firing['y_mean'].append(np.mean(retest_firing['y_mean'], axis=0))
    mean_retest_firing['inter_mean'].append(np.mean(retest_firing['inter_mean'], axis=0))
    mean_retest_firing['y_sd'].append(np.mean(retest_firing['y_sd'], axis=0))
    mean_retest_firing['h_sd'].append(np.mean(retest_firing['h_sd'], axis=0))
    mean_retest_firing['x_sd'].append(np.mean(retest_firing['x_sd'], axis=0))
    mean_retest_firing['rec_sd'].append(np.mean(retest_firing['rec_sd'], axis=0))
    mean_retest_firing['inter_sd'].append(np.mean(retest_firing['inter_sd'], axis=0))



pickle.dump(mean_test_firing, open(f'/users/k1201869/wang_model/results/hcp_testretest/secondary_analysis/indiv_connect/mean_test_firing.pkl', "wb"))
pickle.dump(mean_retest_firing, open(f'/users/k1201869/wang_model/results/hcp_testretest/secondary_analysis/indiv_connect/mean_retest_firing.pkl', "wb"))



mean_test_firing = pickle.load(open(f'/users/k1201869/wang_model/results/hcp_testretest/secondary_analysis/indiv_connect/mean_test_firing.pkl', "rb"))
mean_retest_firing = pickle.load(open(f'/users/k1201869/wang_model/results/hcp_testretest/secondary_analysis/indiv_connect/mean_retest_firing.pkl', "rb"))


retest_firing['y_mean']

len(mean_test_firing['y_mean'][0].shape])

res = []

for measure in ['y', 'h', 'x', 'rec', 'inter']:
    a=tr.retest_reliability(subjects_valid, np.array(mean_test_firing[f'{measure}_mean']), np.array(mean_retest_firing[f'{measure}_mean']))
    #a=tr.retest_reliability(list(subjects.astype(int)), np.array(mean_test_firing[f'{measure}_mean']), np.array(mean_retest_firing[f'{measure}_mean']))
    print(np.median(a))
    print(f'{measure}')
    sns.distplot(a)
    res.append(a)
len(e)
sns.distplot(e)
np.median(e)


measure2='inter'
measure1='x'
a=tr.retest_reliability(list(subjects.astype(int)), np.array(mean_test_firing[f'{measure1}_mean'])/ np.array(mean_test_firing[f'{measure2}_mean']), np.array(mean_retest_firing[f'{measure1}_mean'])/np.array(mean_retest_firing[f'{measure2}_mean']))

np.median(a)

for i in range(10000000):
    555*5636346346346*i
    for session, firing_dict in zip(['test', 'retest'], [test_firing, retest_firing]):
        firing_dict['y_mean'].append(1)