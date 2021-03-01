import sys
import os
import numpy as np
sys.path.insert(0, '../model_fitting')
import wang_functions_imag_fix as wf
from joblib import Parallel, delayed
import pandas as pd
import dask
from dask import distributed
from dask_jobqueue import SLURMCluster
from datetime import datetime
import shutil
import copy
import pickle
import numpy as np
import seaborn as sns


delete_nodes = [0,30,31,34,64,65]
SC = np.loadtxt(open(f"/users/k1201869/wang_model/data/hcp_testretest/dti_collated_retest/group_retest_SC.csv", "rb"), delimiter=" ")
SC = np.delete(SC, delete_nodes, 0)
SC = np.delete(SC, delete_nodes, 1)
SC = (SC/np.max(np.max(SC)))*0.2

#hcp_subjects
home_dir = '/users/k1201869/wang_model'
results_dir = f'{home_dir}/results'
hcp_subjects = os.listdir(f'{home_dir}/data/hcp_scz')
hcp_subjects.sort()
hcp_subjects = hcp_subjects[:-6]

hcp_fcs = []
subject = '1051_01_MR'
fc_file = f'/users/k1201869/wang_model/data/glucog/hcp_vol/{subject}_dk.txt'
FCv1 = np.loadtxt(open(fc_file, "rb"), delimiter=" ")[42:][:,42:]
sns.heatmap(FCv[42:][:,42:])
sns.heatmap(FC)


for subject in hcp_subjects:
    fc_file = f'/users/k1201869/wang_model/data/hcp_scz/{subject}/{subject}_dk_pearson.csv'
    FC = np.loadtxt(open(fc_file, "rb"), delimiter=",")
    # FC = np.delete(FC, delete_nodes, 0)
    # FC = np.delete(FC, delete_nodes, 1)
    np.fill_diagonal(FC,1)
    hcp_fcs.append(FC)


#Glucog subjects
glu_subjects = np.squeeze(pd.read_table("/users/k1201869/wang_model/code/riluzole_analysis/subjects.list", header=None).values)
glu_subjects = glu_subjects[:-1]

glu_fcs = []
for subject in glu_subjects:
    if subject[-1]=='a':
        fc_file = f'/users/k1201869/wang_model/data/glucog/{subject}_cm.txt'
        # fc_file = f'/users/k1201869/wang_model/data/glucog/dk_matrices2/{subject}_dk_cm.txt'
        FC = np.loadtxt(open(fc_file, "rb"), delimiter=" ")
        np.fill_diagonal(FC,1)
        FC = FC[34:][:,34:]
        # FC = FC[42:][:,42:]
        glu_fcs.append(FC)


sns.heatmap(np.mean(glu_fcs, axis=0))

sns.heatmap(np.mean(hcp_fcs, axis=0))

from scipy.stats import pearsonr
u_idx = np.triu_indices(71, k=1)

pearsonr(np.mean(hcp_fcs, axis=0)[u_idx], np.mean(glu_fcs, axis=0)[u_idx])
pearsonr(np.mean(hcp_fcs[70:], axis=0)[u_idx], np.mean(hcp_fcs[:70], axis=0)[u_idx])


pearsonr(FCv1[u_idx], FCv0[u_idx])

rs=[]
dataset = glu_fcs
for i in range(len(dataset)):
    for j in range(len(dataset)):
        rs.append(pearsonr(dataset[i][u_idx], dataset[j][u_idx])[0])

sns.distplot(rs)

import scipy.io as sio
gordons = '/users/k1201869/wang_model/data/glucog/connectomes.mat'
a = sio.loadmat(gordons)

cms = a['connectomes_glucog']

u_idx = np.triu_indices(333, k=1)
pearsonr(cms[:,:,2][2],cms[:,:,1][2])

np.mean(cms[:,:,2])

sns.heatmap(np.mean(cms, axis=2), cmap='RdBu_r', center=0, vmax=0.8)

cm1 = np.mean(cms, axis=2)

sns.heatmap(cm1[dmn,:][:,dmn], cmap='RdBu_r', center=0, vmax=0.8)
sns.heatmap(cm1[con,:][:,dmn], cmap='RdBu_r', center=0, vmax=0.8)

sns.heatmap(cm1[np.hstack([con,dmn]),:][:,np.hstack([con,dmn])], cmap='RdBu_r', center=0, vmax=0.8)


pearsonr(cms[:,:,3][np.hstack([con,dmn]),:][:,np.hstack([con,dmn])].flatten(), cms[:,:,2][np.hstack([con,dmn]),:][:,np.hstack([con,dmn])].flatten())
dmn = np.array([1,4,6,25,26,44,94,114,116,117,126,127,145,146,150,151,152,154,156,157,162,165,184,186,200,220,225,257,259,278,279,290,315,316,321,322,323,324,325,326,331])-1

con = np.array([21,22,27,28,34,40,63,71,72,76,81,82,84,101,103,105,111,112,147,153,180,181,185,187,188,192,196,198,219,223,234,235,238,245,246,248,249,274,317,318])-1


sns.heatmap(cms[:,:,3][np.hstack([con,dmn]),:][:,np.hstack([con,dmn])]-cms[:,:,2][np.hstack([con,dmn]),:][:,np.hstack([con,dmn])], cmap='RdBu_r', center=0, vmax=0.8)


sns.scatterplot(cms[:,:,3][np.hstack([con,dmn]),:][:,np.hstack([con,dmn])].flatten(), cms[:,:,2][np.hstack([con,dmn]),:][:,np.hstack([con,dmn])].flatten(), size=0.1)