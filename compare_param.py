import numpy as np
import seaborn as sns

subject = '1001_01_MR'
run=0

res = np.loadtxt(f'./results/output_{subject}_{run}.txt')[:64]


subject = '1001_01_MR'
run=2
res1 = np.loadtxt(f'./results/output_{subject}_{run}.txt')[:64]

np.corrcoef(res, res1)

len(res1)

corrs=[]
res_list = []
subject='06'
for subject in ['01','02','03','04','05','06','09','10']:
    res0 = np.loadtxt(f'./results/output_10{subject}_01_MR_0.txt')[64:138]
    res_list.append(res0)
    res1 = np.loadtxt(f'./results/output_10{subject}_01_MR_1.txt')[64:138]
    res_list.append(res1)
    res2 = np.loadtxt(f'./results/output_10{subject}_01_MR_2.txt')[64:138]
    res_list.append(res2)
    corrs.append(np.corrcoef(res0, res1)[0,1])

a=np.corrcoef(np.array(res_list))
sns.heatmap(a, cmap='RdBu_r', vmin=0.95)

corrs=[]
res_list = []
subject='06'
#for subject in ['01','02','03','04','05','06','09','10']:
for subject in ['01','03','04','05','06','09']:
    res0 = np.loadtxt(f'./results/output_10{subject}_01_MR_0.txt')[:64]
    res_list.append(res0)
    res1 = np.loadtxt(f'./results/output_10{subject}_01_MR_1.txt')[:64]
    res_list.append(res1)
    res2 = np.loadtxt(f'./results/output_10{subject}_01_MR_2.txt')[:64]
    res_list.append(res2)
    res3 = np.loadtxt(f'./results/output_10{subject}_01_MR_3.txt')[:64]
    res_list.append(res3)
    corrs.append(np.corrcoef(res0, res1)[0,1])


num=4
sub_len = int(len(res_list)/num)
a=np.corrcoef(np.array(res_list))
ax=sns.heatmap(a, cmap='RdBu_r', center=0)
for i in range(sub_len):
    ax.hlines([num*i], *ax.get_xlim())
    ax.vlines([num*i], *ax.get_xlim())

