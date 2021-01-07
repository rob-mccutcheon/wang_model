import numpy as np
import copy
import scipy

def CBIG_MFMem_rfMRI_mfm_ode1(y,parameter,SC,single_param=False):
    '''
    %-----------------------------------------------------------------------------
    % dy = CBIG_MFMem_rfMRI_MFMem_ode1(y,parameter,SC)
    %
    % Function for dynamical mean field model diffiential equation 1st order 
    %
    % Input:
    %     - y:        current neural state
    %     - paremter: model parameter vector {p x 1}, in order [w;I0;G], w: self-connection, I0: background, G: global scaling of SC   
    %     - SC:       structural connectivity matrix 
    %
    % Output:
    %     - dy:       change in neural state
    %
    % Reference: 
    %     (Deco 2013), Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations.
    %
    % Written by Peng Wang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    Translated to python by Rob McCutcheon Oct 2020
    %----------------------------------------------------------------------------
    '''

    if np.size(parameter, 1) > 1:
        error('Input argument ''parameter'' should be a column vector');

    # parameters for inputs and couplings
    J = 0.2609 #nA
    if single_param==False:
        w = parameter[0:np.size(SC,0)]
        G = parameter[-2]
        I0 = parameter[np.size(SC,0):2*np.size(SC,0)]
    if single_param==True:
        w = parameter[0]
        G = parameter[2]
        I0 = parameter[1]

    # parameters for firing rate
    a = 270 #pC
    b = 108 #kHz
    d = 0.154  #ms

    # parameters for synaptic activity/currents
    tau_s = 0.1  #s 
    gamma_s = 0.641 

    ## total input x
    x = J*w*y+J*G*np.matmul(SC,y)+I0
    
    ## firing rate
    c = (1-np.exp(-d*(a*x-b)))
    if np.sum(c) == 0:
        error('error, check firing rate function')
    else:
        H = (a*x-b)/c
    
    ## synaptic activity / currents
    dy = -1/tau_s*y + gamma_s*(1-y)*H

    return dy

def CBIG_MFMem_rfMRI_mfm_ode1_fixed_node(y,parameter,SC):
    '''
    %-----------------------------------------------------------------------------
    % dy = CBIG_MFMem_rfMRI_MFMem_ode1(y,parameter,SC)
    %
    % Function for dynamical mean field model diffiential equation 1st order 
    %
    % Input:
    %     - y:        current neural state
    %     - paremter: model parameter vector {p x 1}, in order [w;I0;G], w: self-connection, I0: background, G: global scaling of SC   
    %     - SC:       structural connectivity matrix 
    %
    % Output:
    %     - dy:       change in neural state
    %
    % Reference: 
    %     (Deco 2013), Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations.
    %
    % Written by Peng Wang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    Translated to python by Rob McCutcheon Oct 2020
    %----------------------------------------------------------------------------
    '''

    if np.size(parameter, 1) > 1:
        error('Input argument ''parameter'' should be a column vector');

    # parameters for inputs and couplings
    J = 0.2609 #nA
    w = parameter[0]
    G = parameter[-2]
    I0 = parameter[np.size(SC,0)]

    # parameters for firing rate
    a = 270 #pC
    b = 108 #kHz
    d = 0.154  #ms

    # parameters for synaptic activity/currents
    tau_s = 0.1  #s 
    gamma_s = 0.641 

    ## total input x
    x = J*w*y+J*G*np.matmul(SC,y)+I0
    
    ## firing rate
    c = (1-np.exp(-d*(a*x-b)))
    if np.sum(c) == 0:
        error('error, check firing rate function')
    else:
        H = (a*x-b)/c
    
    ## synaptic activity / currents
    dy = -1/tau_s*y + gamma_s*(1-y)*H

    return dy

def CBIG_MFMem_rfMRI_mfm_ode1b(y,parameter,SC):
    '''
    %-----------------------------------------------------------------------------
    % dy = CBIG_MFMem_rfMRI_MFMem_ode1(y,parameter,SC)
    %
    % Function for dynamical mean field model diffiential equation 1st order 
    %
    % Input:
    %     - y:        current neural state
    %     - paremter: model parameter vector {p x 1}, in order [w;I0;G], w: self-connection, I0: background, G: global scaling of SC   
    %     - SC:       structural connectivity matrix 
    %
    % Output:
    %     - dy:       change in neural state
    %
    % Reference: 
    %     (Deco 2013), Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations.
    %
    % Written by Peng Wang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    Translated to python by Rob McCutcheon Oct 2020
    %----------------------------------------------------------------------------
    '''

    if np.size(parameter, 1) > 1:
        error('Input argument ''parameter'' should be a column vector');

    # parameters for inputs and couplings
    J = 0.2609 #nA
    w = parameter[0:np.size(SC,0)]
    G = parameter[-2]
    I0 = parameter[np.size(SC,0):2*np.size(SC,0)]

    # parameters for firing rate
    a = 270 #pC
    b = 108 #kHz
    d = 0.154  #ms

    # parameters for synaptic activity/currents
    tau_s = 0.1  #s 
    gamma_s = 0.641 

    ## total input x
    x = J*w*y+J*G*np.matmul(SC,y)+I0
    rec = J*w*y
    inter = J*G*np.matmul(SC,y)
    
    ## firing rate
    c = (1-np.exp(-d*(a*x-b)))
    if np.sum(c) == 0:
        error('error, check firing rate function')
    else:
        H = (a*x-b)/c
    
    ## synaptic activity / currents
    dy = -1/tau_s*y + gamma_s*(1-y)*H

    return dy, H,  x, rec, inter



def CBIG_MFMem_rfMRI_rfMRI_BW_ode1(y,F,Nnodes):
    '''
    %-----------------------------------------------------------------------------
    % dF = CBIG_MFMem_rfMRI_rfMRI_BW_ode1(y,F,Nnodes)
    %
    % Function for hemodynamic model diffiential equation 1st order 
    %
    % Input:
    %     - y:        current neural activiy 
    %     - F:        current hemodynamic state      
    %     - Nnodes:   number of brain regions
    %
    % Output:
    %     - dF:       change in hemodynamic state
    %
    % Reference: 
    %     (Friston 2000), Nonlinear responses in fMRI: the Balloon model, Volterra kernels, and other hemodynamics.
    %
    % Written by Peng Wang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    % Translated to python by Rob McCutcheon
    %----------------------------------------------------------------------------
    '''

    #parameters
    beta = 1/0.65 #per s
    gamma = 1/0.41 #per s
    tau = 0.98 #s
    alpha = 0.33
    p = 0.34
    dF = np.zeros([Nnodes,4])  #F-> Nodes x 4states[dz,df,dv,dq]
    dF = dF.astype(np.complex128)
    dF[:,0] = np.squeeze(np.atleast_2d(y).T - np.atleast_2d(beta*F[:,0]).T - np.atleast_2d(gamma*(F[:,1]-1)).T) #  dz: signal
    dF[:,1] = F[:,0]                             # df: flow 
    dF[:,2] = 1/tau*(F[:,1]-F[:,2]**(1/alpha))  # dv: volume
    dF[:,3] = 1/tau*(F[:, 1]/p*(1-(1-p)**(1./F[:,1]))-F[:,3]/F[:,2]*F[:,2]**(1/alpha)) #dq: deoxyHb

    return dF


def CBIG_MFMem_rfMRI_simBOLD_downsampling(x,bin):
    '''
    %--------------------------------------------------------------------------
    % y = CBIG_MFMem_rfMRI_simBOLD_downsampling(x,bin)
    %
    % Function for reducing data samples of x {n x samples} columns , i.e. bin=8, 
    % put 8 points together, only count the first number of the 8 points. if the 
    % residual is less than 8 points, count the first number of the residual.
    %
    % Input:
    %     - x:      input data matrix {n x samples}
    %     - bin:    put "bin" points in samples together, 
    %
    %  Output
    %      - y:     reduced data matrix   
    %
    % 
    %  Example:
    %    y = CBIG_MFMem_rfMRI_simBOLD_downsampling(x,bin)
    %    suppose:
    %          - x = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18;
    %                 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18]                  ]
    %          - bin = 4
    %    y = CBIG_MFMem_rfMRI_simBOLD_downsampling(x,bin)
    %     
    %    1 2 3 4 -> 1;  5 6 7 8 -> 5; 9 10 11 12 -> 9; 13 14 15 16 -> 13;
    %    17 18 -> 17,
    %
    %    y = [1 5 9 13 17; 1 5 9 13 17]
    %
    % Written by Peng Wang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    %--------------------------------------------------------------------------
    '''
    n = np.size(x,0)

    if np.size(x,1)%bin == 0:
        T = int(np.size(x,1)/bin)
    else:
        T = int(np.floor(np.size(x,1)/bin)+1)
    
    y = np.zeros([n, T], dtype=np.complex128)

    for i in range(T):
        y[:,i] = np.atleast_2d(x[:, int(bin*(i))])

    return y


def CBIG_MFMem_rfMRI_nsolver_eul_sto(parameter,prior,SC,y_FC,FC_mask,Nstate,Tepochlong,TBOLD,ifRepara,single_param=False):
    '''
    %-----------------------------------------------------------------------------
    % FC_simR, CC_check] = CBIG_MFMem_rfMRI_nsolver_eul_sto(parameter,prior,SC,y_FC,FC_mask,Nstate,Tepochlong,TBOLD,ifRepara)
    %
    % Function to 
    %  (a)solve diffitial equation of dynamic mean field and hemodynamic model using stochastic Euler method 
    %  (b)caculate simulated functional connectivity (FC) from simulated BOLD
    %  (c)caculate correlation between emprical FC and simulated FC 
    %
    % Input:
    %     - SC:        structural connectivity matrix
    %     - y_FC:      functional connectivity vector (after mask)
    %     - FC_mask:   used to select uppertragular elements
    %     - Nstate:    noise randon seed
    %     - Tepochlong:simulation long in [min], exclusive 2min pre-simulation(casted)
    %     - TBOLD:     BOLD-signal time resolution
    %     - ifRepara:  re-parameter turn on=1/off=0
    %
    % Output:
    %     - FC_simR:  simulated FC, only entries above main diagonal, in vector form
    %     - CC_check: cross correlation of 2 FCs 
    %
    % Reference:
    %     [1](Deco 2013), Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations.
    %     [2](Friston 2000), Nonlinear responses in fMRI: the Balloon model, Volterra kernels, and other hemodynamics.
    %
    % Written by Peng Wang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    % Translated to python by Rob McCutcheon Oct 2020
    %----------------------------------------------------------------------------
    '''
    # import wang_functions as wf
    # first calculate [BOLD,yT,fT,qT,vT,zT,Time], then calculate FC_stim

    ##(a) solve diffitial equation of dynamic mean field and hemodynamic model using stochastic Euler method 

    if np.size(parameter, 1) > 1:
        error('Input argument ''parameter'' should be a column vector');

    if ifRepara == 1:
        parameter = np.exp(parameter)*prior

    ## initial system
    
    # simulation time
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

    j = 0

    ## solver: Euler
    # warm-up
    for i in range(1000):
        dy = CBIG_MFMem_rfMRI_mfm_ode1(yT,parameter,SC, single_param=single_param)
        yT = yT + dy*dt + (np.atleast_2d(w_coef*dW[:,i])).T
    
    ## main body: calculation 
    y_neuro = np.zeros([Nnodes, len(k_P)])
    y_neuro = y_neuro.astype(np.complex128)
    for i in range(0,len(k_P)):
        dy = CBIG_MFMem_rfMRI_mfm_ode1(yT,parameter,SC,single_param=single_param)
        yT = yT + dy*dt + (np.atleast_2d(w_coef*dW[:,i+1000])).T
        if i%(dtt/dt) == 0:
            y_neuro[:,j] = np.squeeze(yT)
            j = j+1     

    for i in range (1, len(k_PP)):
        dF = CBIG_MFMem_rfMRI_rfMRI_BW_ode1(y_neuro[:,i-1],F,Nnodes)
        F = F + dF*dtt
        zT[:,i] = F[:, 0]
        fT[:,i] = F[:, 1]
        vT[:,i] = F[:, 2]
        qT[:,i] = F[:, 3]


    # Parameter for Balloom-Windkessel model, we updated  this model and its
    # parameter according to Stephan et al 2007, NeuroImage 38:387-401 and
    # Heinzle et al. 2016 NeuroImage 125:556-570, Parameter are set for the 3T
    # and TE=0.0331s
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

    BOLD_d = CBIG_MFMem_rfMRI_simBOLD_downsampling(BOLD_cut,TBOLD/dtt) #down sample 

    FC_sim = np.corrcoef(BOLD_d)
    FC_simR = FC_sim[~FC_mask.astype('bool')]

    CC_check = np.corrcoef(np.arctanh(FC_simR),np.arctanh(y_FC))[0,1]

    return FC_simR, CC_check

def CBIG_MFMem_rfMRI_diff_P1(func,para,houtput,num):
    '''
    %------------------------------------------------------------------------
    % j = CBIG_mfm_rfMRI_diff_P1(func,para,houtput,num)
    %
    % Function for Newton-forwards first order derivative approximation
    %
    %                       f(x0+h)-f(x0)
    %     diff(f(x0) =      --------------
    %                             h
    %
    % Input:
    %     - func:     model equation for derivation, f   
    %     - para:     model parameter      
    %     - houtput:  model output at "para" 
    %     - num:      chosen parameter index, ith-parameter from para vector, 
    %
    % Output:
    %     - j:        derivative f at chosen parameter, df(x0)/x0
    %
    % Example:
    %    j = CBIG_MFMem_rfMRI_diff_P1(func,para,houtput,num)
    %    suppose: 
    %          - model function Y with parameter vector X: Y = model(X)
    %          - parameter vector X = [x1,x2,x3]
    %          - funcY:  funcY = @(X) model(X); 
    %    
    %    then the derivative model Y at x1 (dY/dx1) is computed by:       
    %          j_x1 = CBIG_mfm_rfMRI_diff_P1(funcY,X,Y,1)
    %
    % Written by Peng Wang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    %----------------------------------------------------------------------------
    '''
    Mesp = 2.2*1e-16

    para_1 = copy.deepcopy(para)
    if para[num] == 0:
        h = np.sqrt(Mesp)
    else:
        h = np.sqrt(Mesp)*para[num]
    
    para_1[num] = para_1[num]+h
    
    houtput_new, _ = func(para_1)
    
    j = (houtput_new - houtput)/(para_1[num]-para[num])
        
    return j

def CBIG_MFMem_rfMRI_diff_PC1(func,para,num):
    '''
    %-----------------------------------------------------------------------------
    % Function for Complex-step first order derivative approximation
    %
    %                       Im[f(x0+ih)]
    %     diff(f(x0) =    --------------
    %                             h
    %
    % Input:
    %     - func:     model equation for derivation, f   
    %     - para:     model parameter      
    %     - num:      chosen parameter index, ith-parameter from para vector, 
    %
    % Output:
    %     - j:        derivative f at chosen parameter 
    %
    % Example:
    %    j = CBIG_MFMem_rfMRI_diff_PC1(func,para,houtput,num)
    %    suppose: 
    %          - model function Y with parameter vector X: Y = model(X)
    %          - parameter vector X = [x1,x2,x3]
    %          - funcY:  funcY = @(X) model(X); 
    %    
    %    then the derivative model Y at x1 (dY/dx1) is computed by:       
    %          j_x1 = CBIG_mfm_rfMRI_diff_PC1(funcY,X,1)
    %
    % Reference: 
    %    (Martins 2003), The complex-step derivative approximation.
    %
    % Written by Peng Wang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    %----------------------------------------------------------------------------
    '''
    h1i = 1e-20j
    h1 = 1e-20

    para_1 = copy.deepcopy(para)
    para_1[num] = para_1[num]+h1i
    
    houtput_new_complex, _ = func(para_1)
    houtput_new = houtput_new_complex.imag
    
    j = houtput_new/(h1)
   
    return j

def CBIG_MFMem_rfMRI_matrixQ(i,n,T):
    '''
    %------------------------------------------------------------------------
    % q = CBIG_MFMem_rfMRI_matrixQ(i,n,T)
    % 
    % function for generation specitial diagnal matrix used in estimation
    %
    % Input:
    %     - i   is the index to selected i-th trail
    %     - n   is total availble data trails
    %     - T   is number of samples of each data trail
    %
    % Output:
    %     - q   is a matrix, q:{nT  x nT}
    %
    % Example:
    % q = CBIG_EM_Q(2,2,3)
    %                       trails  samples  selected
    % q =[0 0 0 0 0 0               T1
    %     0 0 0 0 0 0        n1     T2         No
    %     0 0 0 0 0 0               T3
    %     0 0 0 1 0 0               T1       
    %     0 0 0 0 1 0        n2     T2         Yes
    %     0 0 0 0 0 1]              T3        
    %
    % Written by Peng Wang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    %-----------------------------------------------------------------------
    '''
    q_dig = np.zeros([1,n*T])
    q_dig[T*(i):T*i+T] = 1
    q = np.diag(np.squeeze(q_dig))
    return q

def CBIG_MFMem_rfMRI_Trace_AXB(A,B):
    '''
    %------------------------------------------------------------------------
    % traceAB = CBIG_MFMem_rfMRI_Trace_AXB(A,B)
    %
    % This is a function to caculate y = trace(AB)
    %
    % Input:
    %      - A: matrix A
    %      - B: matrix B
    %
    % Output:
    %      - traceAB: trace of AB
    %
    % Example:
    %     y = CBIG_mfm_rfMRI_Trace_AXB(A,B)
    %     suppose:  A = [1 2; 3 4]; B = [4 5; 6 7];
    %     y = 69
    %
    % Written by Peng Wang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    %------------------------------------------------------------------------
    '''
    Num = np.size(A,0)
    tr = np.zeros(Num)

    for i in range(Num):
        temp = A[i,:]*B[:,i].T
        tr[i] = np.sum(temp)

    traceAB = np.sum(tr)

    return traceAB


def matlab_divide(A,b):
    ''' implementation of matlab '\''''
    import numpy as np
    from itertools import combinations

    num_vars = A.shape[1]
    rank = np.linalg.matrix_rank(A)
    if rank == num_vars:              
        sol = np.linalg.lstsq(A, b)[0]    # not under-determined
    else:
        # for nz in combinations(range(num_vars), rank):    # the variables not set to zero
        #     print(nz)
        #     try: 
        #         sol = np.zeros((num_vars, 1))  
        #         sol[nz, :] = np.asarray(np.linalg.solve(np.squeeze(A[:, nz]), b))
        #         print(sol)
        #     except np.linalg.LinAlgError:     
        #         pass
        sol = np.asarray(np.linalg.solve(np.squeeze(A[:, :]), b))
    return sol