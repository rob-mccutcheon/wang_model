from functions import wang_functions_imag_fix as wf
import numpy as np
import seaborn as sns

subject = '1009_01_MR'
data_dir = f'/users/k1201869/wang_model/data/hcp_scz/{subject}'

# Get simulated FC
params = np.load_txt(f'../../results/output_{subject}_0.txt')[:-1]
Para_E =  np.atleast_2d(params).T
Prior_E = None
SC = np.loadtxt(open(f"/users/k1201869/wang_model/data/SC_dk.csv", "rb"), delimiter=",")
SC = (SC/np.max(np.max(SC)))*0.2
FC = np.loadtxt(open(f'{data_dir}/{subject}_dk_pearson.csv', "rb"), delimiter=",")
FC_mask = np.tril(np.ones([np.size(SC, 0), np.size(SC, 0)]), 0)
y = FC[~FC_mask.astype('bool')] 
Nstate = 0
FC_simR, CC_check = wf.CBIG_MFMem_rfMRI_nsolver_eul_sto(Para_E,Prior_E,SC,y,FC_mask,Nstate,Tepochlong=14.4,TBOLD=0.72,ifRepara=0)

y_l = []
H_l = []
for subject in ['1001_01_MR','1003_01_MR','1004_01_MR','1005_01_MR','1006_01_MR','1009_01_MR']:
    for version in [0,1,2,3,4,5,6]:
        Para_E =  np.atleast_2d(np.loadtxt(f'../../results/output_{subject}_{version}.txt')[:-1]).T
        y,h = firing_rate(Para_E, SC)
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


def firing_rate(para, SC):
    # Get firing rates
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
    yT[:, 0] = 0.001

    w_coef = para[-1]/np.sqrt(0.001)
    w_dt = dt #s
    w_L = len(k_P)
    np.random.seed(Nstate)
    dW = np.sqrt(w_dt)*np.random.standard_normal(size=(Nnodes,w_L+1000)) #plus 1000 warm-up

    j = 0

    for i in range(1000):
        dy = wf.CBIG_MFMem_rfMRI_mfm_ode1(yT,para,SC)
        yT = yT + dy*dt + (np.atleast_2d(w_coef*dW[:,i])).T

    ## main body: calculation 
    y_neuro = np.zeros([Nnodes, len(k_P)])
    H_neuro = np.zeros([Nnodes, len(k_P)])
    y_neuro = y_neuro.astype(np.complex128)
    for i in range(0,len(k_P)):
        dy, H = wf.CBIG_MFMem_rfMRI_mfm_ode1b(yT,para,SC, )
        yT = yT + dy*dt + (np.atleast_2d(w_coef*dW[:,i+1000])).T
        if i%(dtt/dt) == 0:
            y_neuro[:,j] = np.squeeze(yT)
            H_neuro[:,j] = np.squeeze(H)
            j = j+1     
    return y_neuro, H_neuro
