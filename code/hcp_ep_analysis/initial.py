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
subjects = subjects[:-6]

# make df
df = pd.read_csv(f'{home_dir}/data/hcp_scz/ndar_subject01.txt', delimiter="\t")


# add empty columns
for i in range(138):
    df[f"param_{i}"] = 0
for i in range(68):
    df[f"strength_{i}"] = 0
for i in range(68):
    df[f"strength_sim_{i}"] = 0

items = ['y_mean', 'h_mean', 'x_mean', 'rec_mean', 'inter_mean', 'rec_ratio_mean']
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
    firing_dict = pickle.load(open(f'{results_dir}/hcpep/testretestSC/secondary_analysis/firing_mean6_indiv_para_{subject}.pkl', "rb"))
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
    param_p.append(ttest_ind(df_con[f'param_{i}'], df_pt[f'param_{i}'])[1])
    param_p.append(ttest_ind(df_con[f'param_{i}'], df_pt[f'param_{i}'])[1])

a=df_con.loc[:,'param_0':'param_67'].mean(axis=1).values
b=df_pt.loc[:,'param_0':'param_67'].mean(axis=1).values
print(ttest_ind(a, b))

a=df_con.loc[:,'param_68':'param_135'].mean(axis=1).values
b=df_pt.loc[:,'param_68':'param_135'].mean(axis=1).values
print(ttest_ind(a, b))

np.sum(fdrcorrection(param_p[:68])[0])
np.sum(np.array(param_p)<0.05)


# True and simulated strength
strength_p=[]
for i in range(68):
    strength_p.append(ttest_ind(df_con[f'strength_{i}'], df_pt[f'strength_{i}'])[1])
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

# need to index[1] for p value and [0] for t value for freesurfer diagram
items_p = {}
for item in items:
    plist = []
    for i in range(68):
        plist.append(ttest_ind(df_pt[f'{item}_{i}'], df_con[f'{item}_{i}'])[1])
        items_p[item] = plist
    print(item)
    print(np.nanmin(fdrcorrection(plist[:])[1]))
    print(np.nanmin(plist[:]))

fdrcorrection(items_p['rec_mean'])
np.min(items_p['rec_mean'])
sns.distplot(items_p['rec_mean'])
np.argmin(items_p['y_mean'])

items_p['rec_mean'][13]

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

temp_idx = ['rec_mean_0','rec_mean_7','rec_mean_13','rec_mean_14','rec_mean_28','rec_mean_34','rec_mean_41','rec_mean_47','rec_mean_48','rec_mean_62',  'rec_mean_4','rec_mean_38','rec_mean_8','rec_mean_42','rec_mean_31','rec_mean_65','rec_mean_32','rec_mean_66','rec_mean_33','rec_mean_67']
frontal_idx = ['rec_mean_1','rec_mean_2','rec_mean_10', 'rec_mean_12', 'rec_mean_16','rec_mean_17','rec_mean_18','rec_mean_25','rec_mean_26','rec_mean_30',
            'rec_mean_35','rec_mean_36','rec_mean_44','rec_mean_46','rec_mean_50','rec_mean_51','rec_mean_52','rec_mean_59','rec_mean_60']
a=df_con.loc[:,(frontal_idx+temp_idx)].mean(axis=1).values
b=df_pt.loc[:,(frontal_idx+temp_idx)].mean(axis=1).values
print(ttest_ind(a, b))

np.std(a)
np.std(b)
len(b)

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

values = np.hstack([df_con.loc[:,'rec_mean_7'].values,  df_pt.loc[:,'rec_mean_7'].values])
values = np.hstack([df_con.loc[:,'rec_mean_0':'rec_mean_67'].mean(axis=1).values,  df_pt.loc[:,'rec_mean_0':'rec_mean_67'].mean(axis=1).values])

values = np.hstack([a,b])
group = ['Control']*df_con.loc[:,'rec_mean_2'].shape[0]+ ['Patient']*df_pt.loc[:,'rec_mean_2'].shape[0]

sns.set_style('white')
PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'},
    'showfliers':{False}
}
plt.rcParams['figure.figsize'] = [5, 5]
sns.stripplot(group, values, palette={'Control': 'blue', 'Patient':'red'})
plt.ylabel('Recurrent Input')
sns.boxplot(group,values, palette={'Control': 'white', 'Patient':'white'}, whis=2,fliersize=0.1, **PROPS)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig(f'{results_dir}/figures/pilot_figures/rec8_scatter.png', dpi=300)


ttest_ind(df_con.loc[:,'rec_mean_13'].values,  df_pt.loc[:,'rec_mean_13'].values)

# relationship with symptoms
cains = pd.read_csv(f'{home_dir}/data/hcp_scz/clinical/cains01.txt', delimiter="\t")
acpt = pd.read_csv(f'{home_dir}/data/hcp_scz/clinical/acpt01.txt', delimiter="\t")
wasi = pd.read_csv(f'{home_dir}/data/hcp_scz/clinical/wasi201.txt', delimiter="\t")
dd = pd.read_csv(f'{home_dir}/data/hcp_scz/clinical/deldisk01.txt', delimiter="\t")
df_full = df.set_index('src_subject_id').join(wasi.set_index('src_subject_id'), lsuffix='', rsuffix='_add')


df_full_pt = df_full[df_full['phenotype']=='Patient']
df_full_pt['rec_mean_overall'] = df_full_pt.loc[:,'rec_mean_0':'rec_mean_67'].mean(axis=1).values

temp_idx = ['rec_mean_7','rec_mean_13','rec_mean_28','rec_mean_41','rec_mean_47','rec_mean_62']
temp_idx = ['rec_mean_7','rec_mean_13','rec_mean_14','rec_mean_28','rec_mean_41','rec_mean_47','rec_mean_48','rec_mean_62']
temp_idx = ['rec_mean_7','rec_mean_13','rec_mean_14','rec_mean_28','rec_mean_41','rec_mean_47','rec_mean_48','rec_mean_62',  'rec_mean_4','rec_mean_38','rec_mean_8','rec_mean_42','rec_mean_31','rec_mean_65','rec_mean_32','rec_mean_66','rec_mean_33','rec_mean_67']
temp_idx = ['rec_mean_7','rec_mean_13','rec_mean_14','rec_mean_28','rec_mean_41','rec_mean_47','rec_mean_48','rec_mean_62',  'rec_mean_4','rec_mean_38','rec_mean_8','rec_mean_42','rec_mean_31','rec_mean_65','rec_mean_32','rec_mean_66','rec_mean_33','rec_mean_67']
temp_idx = ['rec_mean_4','rec_mean_7','rec_mean_8','rec_mean_13','rec_mean_14','rec_mean_28']


temp_idx = ['rec_mean_0','rec_mean_7','rec_mean_13','rec_mean_14','rec_mean_28','rec_mean_34','rec_mean_41','rec_mean_47','rec_mean_48','rec_mean_62',  'rec_mean_4','rec_mean_38','rec_mean_8','rec_mean_42','rec_mean_31','rec_mean_65','rec_mean_32','rec_mean_66','rec_mean_33','rec_mean_67']
frontal_idx = ['rec_mean_1','rec_mean_2','rec_mean_10', 'rec_mean_12', 'rec_mean_16','rec_mean_17','rec_mean_18','rec_mean_25','rec_mean_26','rec_mean_30',
            'rec_mean_35','rec_mean_36','rec_mean_44','rec_mean_46','rec_mean_50','rec_mean_51','rec_mean_52','rec_mean_59','rec_mean_60']
df_full_pt['rec_mean_temporal'] = df_full_pt.loc[:,(temp_idx+frontal_idx)].mean(axis=1).values

#cains_ssum
#'matrix_totalrawscore', 'profilesubtest_performancemr', 'wasi_matrix_perc'
# auditory_t14
# auc_200
cog_item = 'matrix_totalrawscore'
df_full_pt = df_full_pt.dropna(subset=[cog_item])
df_full = df_full.dropna(subset=[cog_item])

pearsonr(df_full_pt[f'rec_mean_temporal'], df_full_pt[cog_item].astype(np.float))
# I think 14 is middle temporal, 7 is inf temp
sns.scatterplot(df_full_pt['rec_mean_8'], df_full_pt[cog_item].astype(np.float))
pearsonr(df_full_pt['rec_mean_14'], df_full_pt[cog_item].astype(np.float))


p_perm_collec = []
perm_idx= np.arange(86)
for h in range(100):
    np.random.shuffle(perm_idx)
    p_perms = []
    ps=[]
    for i in range(68): 
        p = pearsonr(df_full_pt[f'rec_mean_{i}'], df_full_pt[cog_item].astype(np.float))
        p_perm = pearsonr(df_full_pt[f'rec_mean_{i}'].values[perm_idx], df_full_pt[cog_item].astype(np.float).values)
        ps.append(p[1])
        p_perms.append(p_perm[0])
    p_perm_collec.append(p_perms)
fdrcorrection(ps)



sns.regplot(df_full_pt[f'rec_mean_temporal'], df_full_pt[cog_item].astype(np.float))

for idx in temp_idx:
    print(idx)
    print(spearmanr(df_full_pt[idx], df_full_pt[cog_item].astype(np.float)))
sns.scatterplot(df_full_pt[f'rec_mean_overall'], df_full_pt[cog_item].astype(np.float))

fs_figures.plot_grid(ps, vmin=-0.3, vmax=0.3)

pearsonr(items_p['rec_mean'], ps)

perm_results=[]
for i in range(100):
    perm_results.append(abs(pearsonr(items_p['rec_mean'], p_perm_collec[i])[0]))

np.sum(np.array(perm_results)>0.37)
sns.scatterplot(ps, items_p['rec_mean'])