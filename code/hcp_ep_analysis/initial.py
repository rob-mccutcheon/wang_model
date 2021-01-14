import numpy as np
import os
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ttest_ind, pearsonr
import pickle

# define subjects
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
for i in range(68):
    df[f"strength_sim_{i}"] = 0

items = ['y_mean', 'h_mean', 'x_mean', 'rec_mean', 'inter_mean']
for item in items:
    for i in range(68):
        df[f"{item}_{i}"] = 0


# fill with parameters and strength
for subject in subjects:
    # params
    params = []
    for i in range(5):
        try:
            params.append(np.loadtxt(f'{home_dir}/results/hcpep/testretestSC/output_{subject}_{i}.txt')[:-1])
        except:
            continue
    # try:
    #     params.append(np.loadtxt(f'{home_dir}/results/hcpep/testretestSC/output_{subject}_0.txt')[:-1])
    # except:
    #     continue
    df.loc[df['src_subject_id'] == subject[:4],'param_0':'param_137'] = np.mean(params, 0)
    
    # strength
    cmz = np.loadtxt(f'{home_dir}/data/hcp_scz/{subject}/{subject}_dk_pearson.csv', delimiter=',')
    np.fill_diagonal(cmz, 0)
    strength = np.sum(cmz, axis=1)
    df.loc[df['src_subject_id'] == subject[:4],'strength_0':'strength_67'] = strength

    #firing rates
    firing_dict = pickle.load(open(f'{results_dir}/hcpep/testretestSC/secondary_analysis/firing_mean5_indiv_para_{subject}.pkl', "rb"))
    for item in items:
        df.loc[df['src_subject_id'] == subject[:4], f'{item}_0':f'{item}_67'] = firing_dict[item]
        cmz = np.loadtxt(f'{home_dir}/data/hcp_scz/{subject}/{subject}_dk_pearson.csv', delimiter=',')
    
    #sim strength
    cm_sim = firing_dict['FCsim']
    np.fill_diagonal(cm_sim, 0)
    strength_sim = np.sum(cm_sim, axis=1)
    df.loc[df['src_subject_id'] == subject[:4],'strength_sim_0':'strength_sim_67'] = strength_sim


#multilevel
import statsmodels.api as sm
import statsmodels.formula.api as smf

plist = []
for h in range(138):
    df = pd.read_csv(f'{home_dir}/data/hcp_scz/ndar_subject01.txt', delimiter="\t")
    df['param_set'] = np.nan
    df['test_param'] = np.nan
    for subject in subjects:
        for i in range(5):
            try:
                para = np.loadtxt(f'{home_dir}/results/hcpep/testretestSC/output_{subject}_{i}.txt')[:-1]
                df = df.append(pd.Series(), ignore_index = True)
                df.loc[[len(df)-1], 'src_subject_id'] = subject[:4]
                df.loc[[len(df)-1], 'param_set'] = i
                df.loc[[len(df)-1], 'test_param'] = para[h]
                df.loc[[len(df)-1], 'phenotype'] = df[df['src_subject_id']==subject[:4]].iloc[0,:]['phenotype']
            except:
                continue

    df_sub = df[df['test_param']>0]
    md = smf.mixedlm("test_param ~ phenotype", df_sub, groups=df_sub["src_subject_id"])
    mdf=md.fit()
    print(mdf.pvalues['phenotype[T.Patient]'])
    plist.append(mdf.pvalues['phenotype[T.Patient]'])


#  Only keep subjects with data
df=df[df['param_0']>0]

# pt and control df
df_con = df[df['phenotype']=='Control']
df_pt = df[df['phenotype']=='Patient']

#t-test
param_p=[]
for i in range(138):
    param_p.append(ttest_ind(df_con[f'param_{i}'], df_pt[f'param_{i}'])[0])
np.sum(fdrcorrection(param_p[68:])[0])
np.sum(np.array(param_p)<0.05)


# True and simulated strength
strength_p=[]
for i in range(68):
    strength_p.append(ttest_ind(df_con[f'strength_{i}'], df_pt[f'strength_{i}'])[0])
np.sum(fdrcorrection(strength_p)[0])
np.sum(np.array(strength_p)<0.05)


simdf=df[df['strength_sim_0']>0]
simdf_con = simdf[simdf['phenotype']=='Control']
simdf_pt = simdf[simdf['phenotype']=='Patient']
strength_sim_p=[]
for i in range(68):
    strength_sim_p.append(ttest_ind(simdf_con[f'strength_sim_{i}'], simdf_pt[f'strength_sim_{i}'])[0])

sns.scatterplot(strength_p, strength_sim_p)
pearsonr(strength_p, strength_sim_p)

df_con[f'strength_{i}']

sns.distplot(fdrcorrection(strength_p)[1])
sns.distplot(df_con['strength_22'])
sns.distplot(df_pt['strength_22'])

items_p = {}
for item in items:
    plist = []
    for i in range(68):
        plist.append(ttest_ind(df_pt[f'{item}_{i}'], df_con[f'{item}_{i}'])[0])
        items_p[item] = plist
    print(np.nanmin(fdrcorrection(plist[:])[1]))

fdrcorrection(items_p['x_mean'])
sns.distplot(items_p['rec_mean'])
np.argmin(items_p['y_mean'])

np.sum(np.array(items_p['inter_mean'])>0)/68

sns.scatterplot(np.hstack((df_con['y_mean_46'].values, df_pt['y_mean_46'].values)), np.hstack((np.ones(38), np.zeros(63))))

sns.distplot(df_con['y_mean_46'].values)
sns.distplot(df_pt['y_mean_46'].values)


np.nanmin(fdrcorrection(param_p[:])[1])

#mean
item='rec_mean'
for item in items:
    a=df_con.loc[:,f'{item}_0':f'{item}_67'].mean(axis=1).values
    b=df_pt.loc[:,f'{item}_0':f'{item}_67'].mean(axis=1).values
    print(ttest_ind(a, b))

a=df_con.loc[:,f'param_0':'param_67'].mean(axis=1).values
b=df_pt.loc[:,f'param_0':'param_67'].mean(axis=1).values
a=df_con.loc[:,f'param_67':'param_135'].mean(axis=1).values
b=df_pt.loc[:,f'param_67':'param_135'].mean(axis=1).values
print(ttest_ind(a, b))


a=df_con.loc[:,'strength_0':'strength_67'].mean(axis=1).values
b=df_pt.loc[:,'strength_0':'strength_67'].mean(axis=1).values
ttest_ind(a, b)


#for relaoding import initailly form within the figures package
from importlib import reload
from figures import fs_figures
reload(fs_figures)

np.max(items_p['rec_mean'])

fs_figures.plot_grid(items_p['rec_mean'], vmin=-3.1, vmax=3.1)
plt.savefig(f'{results_dir}/figures/pilot_figures/rec_test.png', dpi=300)

fs_figures.plot_grid(np.linspace(1,-1,68), vmin=-68, vmax=68)
fs_figures.plot_grid(np.arange(68), vmin=0, vmax=68)


np.min(con_rec)

con_rec = df_con.loc[:,'h_mean_0':'h_mean_67'].mean().values
con_rec1 = df_con.loc[:,'h_mean_0':'h_mean_67'].iloc[:25].mean().values
con_rec2 = df_con.loc[:,'h_mean_0':'h_mean_67'].iloc[25:].mean().values
pt_rec = df_pt.loc[:,'h_mean_0':'h_mean_67'].mean().values

pearsonr(con_rec1, con_rec2)
pearsonr(con_rec2, pt_rec)



con_recw = df_con.loc[:,'param_0':'param_67'].mean().values
con_reci = df_con.loc[:,'param_68':'param_135'].mean().values
con_rec1 = df_con.loc[:,'param_0':'param_67'].iloc[:25].mean().values
con_rec2 = df_con.loc[:,'param_0':'param_67'].iloc[25:].mean().values
pt_rec = df_pt.loc[:,'param_0':'param_67'].mean().values


con_rec = df_con.loc[:,'strength_0':'strength_67'].mean().values
con_rec1 = df_con.loc[:,'strength_0':'strength_67'].iloc[:25].mean().values
con_rec2 = df_con.loc[:,'strength_0':'strength_67'].iloc[25:].mean().values
pt_rec = df_pt.loc[:,'strength_0':'strength_67'].mean().values
sns.distplot(con_rec)
sns.distplot(pt_rec)

sns.scatterplot(con_recw, con_reci)
sns.lineplot(np.linspace(np.min(con_rec), np.max(con_rec),5), np.linspace(np.min(con_rec), np.max(con_rec),5))


df_con.loc[:,'rec_mean_'].values
np.argmax(items_p['rec_mean'])
sns.scatterplot()

values = np.hstack([df_con.loc[:,'rec_mean_8'].values,  df_pt.loc[:,'rec_mean_8'].values])
values = np.hstack([a,b])
group = ['Control']*df_con.loc[:,'rec_mean_2'].shape[0]+ ['Patient']*df_pt.loc[:,'rec_mean_2'].shape[0]

sns.set_style('white')
PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}
plt.rcParams['figure.figsize'] = [5, 5]
sns.stripplot(group, values, palette={'Control': 'blue', 'Patient':'red'})
plt.ylabel('Recurrent Input')
sns.boxplot(group,values, palette={'Control': 'white', 'Patient':'white'}, whis=2, **PROPS)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig(f'{results_dir}/figures/pilot_figures/rec8_scatter.png', dpi=300)


ttest_ind(df_con.loc[:,'rec_mean_8'].values,  df_pt.loc[:,'rec_mean_8'].values)