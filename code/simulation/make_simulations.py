from functions import wang_functions_imag_fix as wf
from functions import simulation as sim
import numpy as np
import seaborn as sns
import pickle


# Get SC
subject='105923'
SC = np.loadtxt(open(f"/users/k1201869/wang_model/data/hcp_testretest/dti_collated_retest/{subject}_SC.csv", "rb"), delimiter=" ")
SC = (SC/np.max(np.max(SC)))*0.2

# Parameters set
FC_mask = np.tril(np.ones([np.size(SC, 0), np.size(SC, 0)]), 0)
y = SC[~FC_mask.astype('bool')]

# find out number of brain regions
NumC = len(np.diag(SC))

## prepare the model parameters. The test retest data has w:0.2-0.9, I0.2-0.445, s 0.0004-0.001, G 1.4-2.6
# set up prior for G(globle scaling of SC), w(self-connection strength/excitatory),Sigma(noise level),Io(background input)
p = 2*NumC + 2 # number of estimated parametera

w_s = np.linspace(0.4, 0.75, 20)
i_s = np.linspace(0.28, 0.38, 20)
g_s = np.linspace(1, 4, 5)
results = np.zeros([15,15,3,68,68])
results_emp_corr = np.zeros([15,15,3])

#random subject FC
data_dir = f'/users/k1201869/wang_model/data/hcp_testretest/test/917255'
fc_file = f'{data_dir}/cm_combined.csv'
FC = np.loadtxt(open(fc_file, "rb"), delimiter=" ")

for m,w in enumerate(w_s):
    for n,i_para in enumerate(i_s):
        for o,g in enumerate(g_s):
            print(m)
            Para_E = np.zeros([p,1])
            Para_E[0:NumC] = w # w (0.9 in deco paper, 0.5 is wang defaults)
            Para_E[NumC:NumC+NumC] = i_para #I0
            Para_E[2*NumC] = g #G
            Para_E[2*NumC+1] = 0.001 #sigma
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
            results[m,n,o] = FC_sim
            results_emp_corr[m,n,o] = np.corrcoef(FC_sim[np.tril_indices(68,k=-1)], FC[np.tril_indices(68,k=-1)])[1][0]


results_emp_corr = np.zeros([20,20,5])
w_s = np.linspace(0.4, 0.75, 20)
i_s = np.linspace(0.28, 0.38, 20)
for a,i_para in enumerate(i_s):
    for b,w in enumerate(w_s):
        r = pickle.load(open(f'/users/k1201869/wang_model/results/single_param_batch/result_{w}_{i_para}.pkl', 'rb'))
        results_emp_corr[a,b,:]= r[1]


yticks=[str(i)[:4] for i in i_s]
xticks=[str(w)[:4] for w in w_s]

sns.heatmap(results_emp_corr[:,:,3], vmin=0, vmax=0.4, yticklabels=yticks, xticklabels=xticks)
plt.xlabel('w')
plt.ylabel('I')
plt.title('G=1.78')

array([1.5  , 1.625, 1.75 , 1.875, 2.   ])

1.75, 2.5, 3.25, 4

np.max(results_emp_corr)

results[5,5,0,:,:][np.tril_indices(68,k=-1)]

results[9,9,0,:,:]

np.corrcoef(results[9,9,1,:,:].flatten(),results[9,1,2,:,:].flatten())

np.corrcoef(FC_sim_w0_5[np.triu_indices(NumC, k=1)], FC_sim[np.triu_indices(NumC, k=1)])
#ode1










results_emp_corr[0,1,1]
























Gs = np.linspace(1.4, 2.6, num=50)
for i in range(50):
    Para_E[2*NumC] = Gs[i]
    Para_E[2*NumC] = 0.1
    FC_simR2, CC_check = wf.CBIG_MFMem_rfMRI_nsolver_eul_sto(Para_E,Prior_E,SC,y,FC_mask,Nstate,Tepochlong=14.4,TBOLD=0.72,ifRepara=0)
    np.savetxt(f"/users/k1201869/wang_model/data/simulated/G_sim_{i}.csv", sim.vec2mat(FC_simR))


a0 = np.loadtxt(f"/users/k1201869/wang_model/data/simulated/G_sim_1.csv")
a40 = np.loadtxt(f"/users/k1201869/wang_model/data/simulated/G_sim_40.csv")
a49 = np.loadtxt(f"/users/k1201869/wang_model/data/simulated/G_sim_49.csv")

sns.heatmap(sim.vec2mat(FC_simR2), vmin=-0.1, vmax=0.3)

np.corrcoef(a40[np.triu_indices(NumC, k=1)], a0[np.triu_indices(NumC, k=1)])

np.corrcoef(a49[np.triu_indices(NumC, k=-1)], FC_simR.real)

sns.scatterplot(a40[np.triu_indices(NumC, k=1)], FC_simR.real)
sns.scatterplot(a40[np.tril_indices(NumC, k=-1)], a0[np.tril_indices(NumC, k=-1)])

np.min(max_test_params[:,137])

sns.scatterplot(FC_simR2, FC_simR1)

np.std(FC_simR2)
#use

# 



params = 