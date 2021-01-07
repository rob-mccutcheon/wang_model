import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ttest_ind
import pickle

#define subjects
home_dir = '/users/k1201869/wang_model'
results_dir = f'{home_dir}/results'
subjects = os.listdir(f'{home_dir}/data/hcp_scz')
subjects.sort()
subjects = subjects[:-4]

# make df
df = pd.read_csv(f'{home_dir}/data/hcp_scz/ndar_subject01.txt', delimiter="\t")

# add empty columns
for i in range(138):
    df[f"param_{i}"] = 0
for i in range(68):
    df[f"strength_{i}"] = 0

items = ['y_mean', 'h_mean', 'x_mean', 'rec_mean', 'inter_mean']
for item in items:
    for i in range(68):
        df[f"{item}_{i}"] = 0


# fill with parameters and strength
for subject in subjects:
    # params
    params = []
    # for i in range(5):
    #     try:
    #         params.append(np.loadtxt(f'{home_dir}/results/hcpep/testretestSC/output_{subject}_{i}.txt')[:-1])
    #     except:
    #         continue
    try:
        params.append(np.loadtxt(f'{home_dir}/results/hcpep/testretestSC/output_{subject}_0.txt')[:-1])
    except:
        continue
    df.loc[df['src_subject_id'] == subject[:4],'param_0':'param_137'] = np.mean(params, 0)
    
    # strength
    cmz = np.loadtxt(f'{home_dir}/data/hcp_scz/{subject}/{subject}_dk_pearson.csv', delimiter=',')
    np.fill_diagonal(cmz, 0)
    strength = np.sum(cmz, axis=1)
    df.loc[df['src_subject_id'] == subject[:4],'strength_0':'strength_67'] = strength

    #firing rates
    firing_dict = pickle.load(open(f'{results_dir}/hcpep/testretestSC/secondary_analysis/firing_mean_indiv_para_{subject}.pkl', "rb"))
    for item in items:
        df.loc[df['src_subject_id'] == subject[:4], f'{item}_0':f'{item}_67'] = firing_dict[item]

df.iloc[90,:]
#  Only keep subjects with data
df=df[df['param_0']>0]

# pt and control df
df_con = df[df['phenotype']=='Control']
df_pt = df[df['phenotype']=='Patient']

#t-test
param_p=[]
for i in range(138):
    param_p.append(ttest_ind(df_con[f'param_{i}'], df_pt[f'param_{i}'])[1])


np.w

strength_p=[]
for i in range(68):
    strength_p.append(ttest_ind(df_con[f'strength_{i}'], df_pt[f'strength_{i}'])[1])

items_p = {}
for item in items:
    plist = []
    for i in range(68):
        plist.append(ttest_ind(df_con[f'{item}_{i}'], df_pt[f'{item}_{i}'])[0])
        items_p[item] = plist
    print(np.nanmin(fdrcorrection(plist[:])[1]))


items_p['y_mean']

sns.distplot(items_p['inter_mean'])
np.argmin(items_p['y_mean'])

np.sum(np.array(items_p['inter_mean'])>0)/68

sns.scatterplot(np.hstack((df_con['y_mean_46'].values, df_pt['y_mean_46'].values)), np.hstack((np.ones(38), np.zeros(63))))

sns.distplot(df_con['y_mean_46'].values)
sns.distplot(df_pt['y_mean_46'].values)


np.nanmin(fdrcorrection(param_p[:])[1])

for item in items:
    a=df_con.loc[:,f'{item}_0':f'{item}_67'].mean(axis=1).values
    b=df_pt.loc[:,f'{item}_0':f'{item}_67'].mean(axis=1).values
    print(ttest_ind(a, b))

a=df_con.loc[:,'strength_0':'strength_67'].mean(axis=1).values
b=df_pt.loc[:,'strength_0':'strength_67'].mean(axis=1).values
ttest_ind(a, b)


#for relaoding import initailly form within the figures package
from importlib import reload
from figures import fs_figures
reload(fs_figures)

fs_figures.plot_grid(items_p['h_mean'], vmin=-2.5, vmax=2.5)

np.sum(np.array(items_p['rec_mean'])>0)vmin