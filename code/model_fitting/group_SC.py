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
home_dir = '/users/k1201869/wang_model/data/hcp_testretest/'
subjects = np.loadtxt(open('/users/k1201869/wang_model/data/subjects_testretest.list'))

SCs=[]
for subject in subjects.astype(int):
    SCs.append(np.loadtxt(open(f"{home_dir}/dti_collated_retest/{subject}_SC.csv", "rb"), delimiter=" "))

group_SC = np.mean(np.array(SCs), axis=0)

np.savetxt(f"{home_dir}/dti_collated_retest/group_retest_SC.csv", group_SC,delimiter=" ")