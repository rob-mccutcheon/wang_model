from functions import wang_functions_imag_fix as wf
import numpy as np

def firing_rate(para, SC, Nstate):
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
    x_neuro = np.zeros([Nnodes, len(k_P)])
    # y_neuro = y_neuro.astype(np.complex128)
    for i in range(0,len(k_P)):
        dy, H, x, rec, inter = wf.CBIG_MFMem_rfMRI_mfm_ode1b(yT,para,SC, )
        yT = yT + dy*dt + (np.atleast_2d(w_coef*dW[:,i+1000])).T
        if i%(dtt/dt) == 0:
            y_neuro[:,j] = np.squeeze(yT)
            H_neuro[:,j] = np.squeeze(H)
            x_neuro[:,j] = np.squeeze(x)
            j = j+1     
    return y_neuro, H_neuro, x_neuro, rec, inter
