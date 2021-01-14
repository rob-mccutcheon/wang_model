from functions import wang_functions_imag_fix as wf
import numpy as np

def firing_rate(para, SC, Nstate, TBOLD=0.72):
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
    x_neuro = np.zeros([Nnodes, len(k_P)])
    rec_neuro = np.zeros([Nnodes, len(k_P)])
    inter_neuro = np.zeros([Nnodes, len(k_P)])
    # y_neuro = y_neuro.astype(np.complex128)
    for i in range(0,len(k_P)):
        dy, H, x, rec, inter = wf.CBIG_MFMem_rfMRI_mfm_ode1b(yT,para,SC, )
        yT = yT + dy*dt + (np.atleast_2d(w_coef*dW[:,i+1000])).T
        if i%(dtt/dt) == 0:
            y_neuro[:,j] = np.squeeze(yT)
            H_neuro[:,j] = np.squeeze(H)
            x_neuro[:,j] = np.squeeze(x)
            rec_neuro[:,j] = np.squeeze(rec)
            inter_neuro[:,j] = np.squeeze(inter)
            j = j+1

    for i in range (1, len(k_PP)):
        dF = wf.CBIG_MFMem_rfMRI_rfMRI_BW_ode1(y_neuro[:,i-1],F,Nnodes)
        F = F + dF*dtt
        zT[:,i] = F[:, 0]
        fT[:,i] = F[:, 1]
        vT[:,i] = F[:, 2]
        qT[:,i] = F[:, 3]
    
    p = 0.34
    v0 = 0.02
    k1 = 4.3*28.265*3*0.0331*p
    k2 = 0.47*110*0.0331*p
    k3 = 1-0.47
    y_BOLD = 100/p*v0*(k1*(1-qT) + k2*(1-qT/vT) + k3*(1-vT))

    Time = k_PP
    
    # (b)&(c) compute simulated FC and correlation of 2 FCs
    # get the static part
    cut_indx = np.where(Time == Tpre)[0][0] # after xx s
    BOLD_cut = y_BOLD[:, cut_indx:]
    y_neuro_cut = y_neuro[:, cut_indx:]
    Time_cut = Time[cut_indx:]

    BOLD_d = wf.CBIG_MFMem_rfMRI_simBOLD_downsampling(BOLD_cut,TBOLD/dtt) #down sample 

    FC_sim = np.corrcoef(BOLD_d)     

    return y_neuro, H_neuro, x_neuro, rec_neuro, inter_neuro, FC_sim

def vec2mat(a):
    '''convert connectivity vector to original 2D matrix form
    '''
    n = int(np.sqrt(len(a)*2))+1
    mask = np.tri(n,dtype=bool, k=-1).T # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n,n),dtype='float64')
    out[mask] = a
    out = out + out.T
    return out


