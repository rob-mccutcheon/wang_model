import sys
sys.path.append("/users/k1201869/wang_model/code")
from functions import wang_functions_imag_fix as wf
from functions import simulation as sim
import numpy as np
import seaborn as sns
import pickle
import pandas as pd

#subject
number = int(sys.argv[1])-1
subject_idx = int(sys.argv[1])-1
subject_list = pd.read_table("/users/k1201869/wang_model/data/param_combo.list", dtype=object)
w_idx = int(subject_list.iloc[subject_idx].values[0][0:2])
i_para_idx = int(subject_list.iloc[subject_idx].values[0][2:])


# Get SC
SC = np.loadtxt(open(f"/users/k1201869/wang_model/data/SC_glasser.csv", "rb"), delimiter=",")
# SC = (SC/np.max(np.max(SC)))*0.2
SC = SC[:,:360][:360, :]*0.2
SC = SC**0.5

# Parameters set
FC_mask = np.tril(np.ones([np.size(SC, 0), np.size(SC, 0)]), 0)
y = SC[~FC_mask.astype('bool')]

# find out number of brain regions
NumC = len(np.diag(SC))

## prepare the model parameters. The test retest data has w:0.2-0.9, I0.2-0.445, s 0.0004-0.001, G 1.4-2.6
# set up prior for G(globle scaling of SC), w(self-connection strength/excitatory),Sigma(noise level),Io(background input)
p = 2*NumC + 2 # number of estimated parametera

w_s = np.linspace(0.3, 0.8, 20)
i_s = np.linspace(0.2, 0.4, 20)
g_s = np.linspace(1.65, 1.9, 5)

w = w_s[w_idx]
i_para = i_s[i_para_idx]

results = np.zeros([5, SC.shape[0], SC.shape[0]])
results_emp_corr = np.zeros([5])

#random subject FC
# data_dir = f'/users/k1201869/wang_model/data/hcp_testretest/test/917255'
# fc_file = f'{data_dir}/cm_combined.csv'
# FC = np.loadtxt(open(fc_file, "rb"), delimiter=" ")

FC = np.loadtxt(open(f"/users/k1201869/wang_model/data/hcp_scz/1002_01_MR/1002_01_MR_glasser_pearson.csv", "rb"), delimiter=",")
np.fill_diagonal(FC,1)


for o,g in enumerate(g_s):
    print(o)
    Para_E = np.zeros([p,1])
    Para_E[0:NumC] = w # w (0.9 in deco paper, 0.5 is wang defaults)
    Para_E[NumC:NumC+NumC] = i_para #I0
    Para_E[2*NumC] = g #G
    Para_E[2*NumC+1] = 0.01 #sigma
    Nstate = 0
    Prior_E = None
    parameter = Para_E


    results_dir = f'/users/k1201869/wang_model/results/hcp_testretest/test'
    # parameter = np.atleast_2d(np.loadtxt(f'{results_dir}/output_103818_1.txt')[:-1]).T
    # parameter[2*NumC] = 1.7
    # parameter[0:NumC] = 0.5

    # simulation time
    Tepochlong=14.4
    kstart = 0  #s
    Tpre = 60*2 #s
    kend = Tpre+60*Tepochlong #s

    dt_l = 0.01    #s  integration time step
    dt = dt_l  #s  time step for neuro
    dtt = 0.01 #s, time step for BOLD


    # sampling ratio
    k_P = np.arange(kstart, kend+dt, dt)
    k_PP = np.arange(kstart, kend+dtt, dtt)

    # initial
    Nnodes = np.size(SC,0)
    Nsamples = len(k_P)
    Bsamples = len(k_PP)

    # for neural activity y0 = 0
    yT = np.zeros([Nnodes, 1])

    # for hemodynamic activity z0 = 0, f0 = v0 = q0 =1
    zT = np.zeros([Nnodes, Bsamples], dtype=np.complex128)
    fT = np.zeros([Nnodes,Bsamples], dtype=np.complex128)
    fT[:, 0] = 1
    vT = np.zeros([Nnodes,Bsamples], dtype=np.complex128)
    vT[:, 0] = 1
    qT = np.zeros([Nnodes,Bsamples], dtype=np.complex128)
    qT[:,0] = 1

    F = np.array([zT[:,0], fT[:,0], vT[:,0], qT[:,0]]).T
    yT[:, 0] = 0.001

    # wiener process
    w_coef = parameter[-1]/np.sqrt(0.001)
    w_dt = dt #s
    w_L = len(k_P)
    np.random.seed(Nstate)
    dW = np.sqrt(w_dt)*np.random.standard_normal(size=(Nnodes,w_L+1000)) #plus 1000 warm-up

    ## solver: Euler
    # warm-up
    for i in range(1000):
        dy = wf.CBIG_MFMem_rfMRI_mfm_ode1_fixed_node(yT,parameter,SC)
        yT = yT + dy*dt + (np.atleast_2d(w_coef*dW[:,i])).T

    ## main body: calculation
    j=0 
    y_neuro = np.zeros([Nnodes, len(k_P)])
    y_neuro = y_neuro.astype(np.complex128)
    for i in range(0,len(k_P)):
        dy = wf.CBIG_MFMem_rfMRI_mfm_ode1_fixed_node(yT,parameter,SC)
        yT = yT + dy*dt + (np.atleast_2d(w_coef*dW[:,i+1000])).T
        y_neuro[:,j] = np.squeeze(yT)
        j = j+1


    # bold balloon
    for i in range (1, len(k_PP)):
        dF = wf.CBIG_MFMem_rfMRI_rfMRI_BW_ode1(y_neuro[:,i-1],F,Nnodes)
        F = F + dF*dtt
        zT[:,i] = F[:, 0]
        fT[:,i] = F[:, 1]
        vT[:,i] = F[:, 2]
        qT[:,i] = F[:, 3]

    p1 = 0.34
    v0 = 0.02
    k1 = 4.3*28.265*3*0.0331*p
    k2 = 0.47*110*0.0331*p
    k3 = 1-0.47
    y_BOLD = 100/p1*v0*(k1*(1-qT) + k2*(1-qT/vT) + k3*(1-vT))

    Time = k_PP

    cut_indx = np.where(Time == Tpre)[0][0] # after xx s
    BOLD_cut = y_BOLD[:, cut_indx:]
    y_neuro_cut = y_neuro[:, cut_indx:]
    Time_cut = Time[cut_indx:]
    TBOLD=0.72
    BOLD_d = wf.CBIG_MFMem_rfMRI_simBOLD_downsampling(BOLD_cut,TBOLD/dtt) #down sample 

    FC_sim = np.corrcoef(BOLD_d)
    results[o] = FC_sim
    results_emp_corr[o] = np.corrcoef(FC_sim[np.tril_indices(FC.shape[0],k=-1)], FC[np.tril_indices(FC.shape[0],k=-1)])[1][0]
final = [results, results_emp_corr]   
pickle.dump(final, open(f'/users/k1201869/wang_model/results/single_param_batch/result_{w}_{i_para}.pkl', 'wb'))
print('finished')
print(final)