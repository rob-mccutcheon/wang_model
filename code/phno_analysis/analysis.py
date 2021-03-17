import pandas as pd
import numpy as np
import pickle
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection

# load glucog details, 'participant'  is the glucog id
data_dir = '/users/k1201869/wang_model/data/glucog'
glucog_df = pd.read_csv(f'{data_dir}/mrs_data.csv')

results_dir = '/users/k1201869/wang_model/results/glucog/testretestSC/secondary_analysis'

subject_list = np.squeeze(pd.read_table("/users/k1201869/wang_model/code/riluzole_analysis/subjects.list", header=None).values)

new_subject_list = []
for subject in subject_list[:-1]:
    new_subject_list.append(subject[:-1])
new_subject_list = np.unique(new_subject_list)

partics = glucog_df['participant'].values
partics = [str(i).zfill(2) for i in partics]
patients = []
for subject in new_subject_list:
    patients.append((glucog_df[subject[-2:] == np.array(partics)]['patient']).values[0])
patients=np.array(patients)==1



bl_list = []
fu_list = []
for subject in new_subject_list:#[patients]:
    try:
        bl_list.append(pickle.load(open(f'{results_dir}/firing_mean4_indiv_para_{subject}a.pkl', "rb"))['rec_mean'])
        fu_list.append(pickle.load(open(f'{results_dir}/firing_mean4_indiv_para_{subject}b.pkl', "rb"))['rec_mean'])
    except:
        print(subject)

ps = []
for i in range(62):
    t = ttest_rel(np.array(bl_list)[:,i], np.array(fu_list)[:,i])[1]
    ps.append([t, labs[i]])

ttest_rel(np.mean((np.array(bl_list)[:,a]), axis=1), np.mean(np.array(fu_list)[:,a], axis=1))

np.array(bl_list)[:,a]
fdrcorrection(ps)

np.array(bl_list)[:,a].shape

#load atlas
import nibabel as nib
a= nib.load('/users/k1201869/glucog/OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_2mm_v2.nii.gz')
labs = np.unique(a.get_data())[35:]

a=[]
for region in [1009,1015,1016,1030,2009,2015,2016,2030]:
for region in [1003,1012,1014,1018,1019,1020,1027,1028, 1032, 2003,2012,2014,2018,2019,2020,2027,2028, 2032,1009,1015,1016,1030,2009,2015,2016,2030]:
for region in both:
    try:
        a.append(np.where(labs==region)[0][0])
    except:
        print(region)

temp_idx = ['rec_mean_0','rec_mean_7','rec_mean_13','rec_mean_14','rec_mean_28','rec_mean_34','rec_mean_41','rec_mean_47','rec_mean_48','rec_mean_62',  'rec_mean_4','rec_mean_38','rec_mean_8','rec_mean_42','rec_mean_31','rec_mean_65','rec_mean_32','rec_mean_66','rec_mean_33','rec_mean_67']
frontal_idx = ['rec_mean_1','rec_mean_2','rec_mean_10', 'rec_mean_12', 'rec_mean_16','rec_mean_17','rec_mean_18','rec_mean_25','rec_mean_26','rec_mean_30',
            'rec_mean_35','rec_mean_36','rec_mean_44','rec_mean_46','rec_mean_50','rec_mean_51','rec_mean_52','rec_mean_59','rec_mean_60']



frontal = [1002, 1014, 1024, 1026,1027, 1028, 2002,2014, 2024, 2026, 2027, 2028]
temporal = [1009, 1015, 1030,2009, 2015, 2030]
both = frontal+temporal
b=[0,3,4,6,7,8,10,11,12,15,17,19,26,27]
labs[np.array(b)]



before = np.mean((np.array(bl_list)[:,a]), axis=1)
after = np.mean(np.array(fu_list)[:,a], axis=1)

before-after