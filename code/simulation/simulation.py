from functions import wang_functions_imag_fix as wf
from functions import simulation as sim
import numpy as np
import seaborn as sns

subject = '1009_01_MR'
data_dir = f'/users/k1201869/wang_model/data/hcp_scz/{subject}'

# Get simulated FC
params = np.loadtxt(f'../../results/output_{subject}_0.txt')[:-1]
FC = np.loadtxt(open(f'{data_dir}/{subject}_dk_pearson.csv', "rb"), delimiter=",")

Para_E =  np.atleast_2d(params).T
Prior_E = None
SC = np.loadtxt(open(f"/users/k1201869/wang_model/data/SC_dk.csv", "rb"), delimiter=",")
SC = (SC/np.max(np.max(SC)))*0.2
FC_mask = np.tril(np.ones([np.size(SC, 0), np.size(SC, 0)]), 0)
y = FC[~FC_mask.astype('bool')] 
Nstate = 0
FC_simR, CC_check = wf.CBIG_MFMem_rfMRI_nsolver_eul_sto(Para_E,Prior_E,SC,y,FC_mask,Nstate,Tepochlong=14.4,TBOLD=0.72,ifRepara=0)

y_l = []
H_l = []
for subject in ['1001_01_MR','1003_01_MR','1004_01_MR','1005_01_MR','1006_01_MR','1009_01_MR']:
    for version in [0,1,2,3,4,5,6]:
        Para_E =  np.atleast_2d(np.loadtxt(f'../../results/output_{subject}_{version}.txt')[:-1]).T
        y,h = sim.firing_rate(Para_E, SC)
        y_l.append(y)
        H_l.append(h)

np.array(y_l).shape

num=7
sub_len = 6
y_l_mean = np.mean(np.array(y_l), axis=2)
a=np.corrcoef(y_l_mean).astype('float64')
ax=sns.heatmap(a, cmap='RdBu_r', center=0.4)
for i in range(sub_len):
    ax.hlines([num*i], *ax.get_xlim())
    ax.vlines([num*i], *ax.get_xlim())
