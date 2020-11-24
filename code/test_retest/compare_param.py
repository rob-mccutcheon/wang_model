import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg

from functions import test_retest_funcs2 as tr

subjects = ['01','02','03','04','05','06','09','10','12']
controls = ['01','02','03','04','05','10']
patients = ['06','09','12']

params = []
for subject in subjects:
    param_file = f'../results/output_10{subject}_01_MR_1.txt'
    params.append(np.loadtxt(param_file))

param_df = pd.DataFrame(data=np.array(params).T, columns = subjects)

ds1 = []
for i in param_df.index:
    c_param = param_df.loc[i, controls]
    p_param = param_df.loc[i, patients]
    d = pg.compute_effsize(c_param, p_param, eftype='cohen')
    ds1.append(d)

sns.distplot(ds1)
sns.scatterplot(np.arange(139), ds1)

sns.scatterplot(ds0, ds1)



# micah's results

subjects = ['01','02','03','04','05','06','09','10','12', '13', '15', '17', '18', '19', '20', '21', '22', '24', '25', '26', '27', '28', '29', '30']
controls = ['01','02','03','04','05','10',  '17',  '24',  '26', '27', '28']
patients = ['06','09','12', '13', '15', '18', '19', '20', '21', '22','25', '29', '30']

params = []
for subject in subjects:
    param_file = f'../results/micah_results/results/output_10{subject}_01_MR.txt'
    params.append(np.loadtxt(param_file))

param_df = pd.DataFrame(data=np.array(params).T, columns = subjects)

ds1 = []
for i in param_df.index:
    c_param = param_df.loc[i, controls]
    p_param = param_df.loc[i, patients]
    d = pg.compute_effsize(c_param, p_param, eftype='cohen')
    ds1.append(d)

sns.distplot(ds1)
sns.scatterplot(np.arange(139), ds1)

sns.scatterplot(ds0, ds1)



# FC measures
degs = []
for subject in subjects:
    cm_file = f'../data/hcp_scz/10{subject}_01_MR/10{subject}_01_MR_dk_pearson.csv'
    cm = pd.read_csv(cm_file, header=None).values
    np.fill_diagonal(cm, 0)
    degs.append(np.sum(cm, axis=0))

deg_df = pd.DataFrame(data=np.array(degs).T, columns = subjects)

deg_ds = []
for i in deg_df.index:
    c_param = deg_df.loc[i, controls]
    p_param = deg_df.loc[i, patients]
    d = pg.compute_effsize(c_param, p_param, eftype='cohen')
    deg_ds.append(d)

sns.distplot(deg_ds)
sns.scatterplot(np.arange(len(deg_ds)), deg_ds)



# Compare within and between participant parameter correlations
corrs=[]
res_list = []
subject='06'


for subject in ['01','02', '03','04','05','06','09','10','12']:
    res0 = np.loadtxt(f'../results/output_10{subject}_01_MR_0.txt')[:64]
    res_list.append(res0)
    res1 = np.loadtxt(f'../results/output_10{subject}_01_MR_1.txt')[:64]
    res_list.append(res1)
    res2 = np.loadtxt(f'../results/output_10{subject}_01_MR_2.txt')[:64]
    res_list.append(res2)
    res3 = np.loadtxt(f'../results/output_10{subject}_01_MR_3.txt')[:64]
    res_list.append(res3)
    res4 = np.loadtxt(f'../results/output_10{subject}_01_MR_4.txt')[:64]
    res_list.append(res4)
    # res5 = np.loadtxt(f'../results/output_10{subject}_01_MR_5.txt')[:64]
    # res_list.append(res5)
    corrs.append(np.corrcoef(res0, res1)[0,1])


num=4
sub_len = int(len(res_list)/num)
a=np.corrcoef(np.array(res_list))
ax=sns.heatmap(a, cmap='RdBu_r', center=0)
for i in range(sub_len):
    ax.hlines([num*i], *ax.get_xlim())
    ax.vlines([num*i], *ax.get_xlim())

a[26]