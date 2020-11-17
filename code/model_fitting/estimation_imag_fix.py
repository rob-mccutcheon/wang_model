import sys
import os
import numpy as np
import wang_functions_imag_fix as wf
from joblib import Parallel, delayed
import pandas as pd
import dask
from dask import distributed
from dask_jobqueue import SLURMCluster
from datetime import datetime
import shutil
import copy
import pickle

os.chdir('/users/k1201869/wang_model/')

#subject
subject_idx = int(sys.argv[1])-1
subject_list = pd.read_table("./data/subjects.list")
subject = subject_list.iloc[subject_idx].values[0]
print(f'subject {subject}')

# Set paths for where structural and functional connectivity matrices are stored
data_dir = f'./data/hcp_scz/{subject}'

#Set where to save parameters and correlation
results_dir = './results'

# Number of fitting iterations
EstimationMaxStep = 150

# Load FC and SC
fc_file = f'{data_dir}/{subject}_dk_pearson.csv'
SC = np.loadtxt(open(f"./data/SC_dk.csv", "rb"), delimiter=",")
FC = np.loadtxt(open(fc_file, "rb"), delimiter=",")
SC = (SC/np.max(np.max(SC)))*0.2

# find out number of brain regions
NumC = len(np.diag(SC))

FC_mask = np.tril(np.ones([np.size(FC, 0), np.size(FC, 0)]), 0)
y = FC[~FC_mask.astype('bool')] # use the elements above the main diagnal, y becomes a vector {samples x 1} 
n = 1            # only one FC  
T = len(y)    # samples of FC 
nT = n*T         # number of data samples

## prepare the model parameters
# set up prior for G(globle scaling of SC), w(self-connection strength/excitatory),Sigma(noise level),Io(background input)
p = 2*NumC + 2 # number of estimated parametera

init_version = 0
try:
    init_version = int(max(os.listdir(f'./temp/{subject}')))
    saved_variables_filename = os.listdir(f'./temp/{subject}/{init_version}')[0]
    saved_variables = pickle.load(open(f'./temp/{subject}/{init_version}/{saved_variables_filename}', 'rb'))
    print('loaded saved variables')
except:
    print('no variables loaded')


for version in range(init_version, 10):
    print(f'version {version}')
    Prior_E = np.zeros([p,1])

    # Basic value / expectation value     
    Prior_E[0:NumC] = 0.5 # w
    Prior_E[NumC:NumC+NumC] = 0.3 #I0
    Prior_E[2*NumC] = 1 #G
    Prior_E[2*NumC+1] = 0.001 #sigma

    # Prior for Re-Parameter A,  Parameter_model = Prior_E.*exp(A), A~Normal(E=0,C)
    A_Prior_C = 1/4*np.ones([p]) # variance for parameter
    A_Prior_C = np.diag(A_Prior_C)
    A_Prior_E = np.zeros([p,1]) 
    invPrior_C = np.linalg.inv(A_Prior_C)

    #initial Parameter  
    Para_E = Prior_E
    Para_E_new = Para_E

    # re-paramter of Para_E 
    A = np.log(Para_E/Prior_E)
    A = A.astype(np.complex128)

    # randomise init parameters
    np.random.seed(version)
    A_start = -1/4+(1/4+1/4)*np.random.rand(2*NumC+2,1)
    A = A_start.astype(np.complex128)
    Para_E = np.exp(A)*Prior_E
    Para_E_new = Para_E

    ## begin estimation
    step = 0 #counter
    CC_check_step = np.squeeze(np.zeros([1,EstimationMaxStep+1]) )    #save the fitting criterion, here is the goodness of fit, same as rrr below
    lembda_step_save = np.zeros([n,EstimationMaxStep])    #save the Ce
    rrr = np.zeros([1,EstimationMaxStep])                 #save the goodness of fit
    rrr_z = np.zeros([1,EstimationMaxStep])              #save the correlation between emprical FC and simulated FC, z-transfered
    Para_E_step_save = np.zeros([p,EstimationMaxStep])    #save the estimated parameter
    
    # If we have already ran some of this continue from where we left off
    try:
        vars().update(saved_variables)
    except:
        pass

    if version>init_version:
        step=0

    while step <= (EstimationMaxStep-1):
        print(step)
        Nstate = step
        Para_E_step = Para_E
        #np.random.seed(step)
        mem = '2 GB'
        cluster = SLURMCluster(cores=1, memory=mem, 
        queue='brc', interface='em1',
        log_directory=f'./dask_logs/dask_logs_{subject}')
        cluster.scale(jobs=30)
        client = distributed.Client(cluster)

        # calculation h_output {nT x 1}
        def funcP(Para_E, Prior_E=Prior_E,SC=SC,y=y,FC_mask=FC_mask,Nstate=Nstate,Tepochlong=14.4,TBOLD=0.72,ifRepara=0):
            FC_simR, CC_check = wf.CBIG_MFMem_rfMRI_nsolver_eul_sto(Para_E,Prior_E,SC,y,FC_mask,Nstate,Tepochlong,TBOLD,ifRepara)
            return FC_simR, CC_check

        def funcA(A, Prior_E=Prior_E,SC=SC,y=y,FC_mask=FC_mask,Nstate=Nstate,Tepochlong=14.4,TBOLD=0.72,ifRepara=1):
            FC_simR, CC_check = wf.CBIG_MFMem_rfMRI_nsolver_eul_sto(A,Prior_E,SC,y,FC_mask,Nstate,Tepochlong,TBOLD,ifRepara)
            return FC_simR, CC_check

        [h_output, CC_check] = funcP(Para_E) #h _ouput is the connectivity vector, CC_check its correlation with the empirical (After both are z-transformed)

        # calculation of Jacobian, JF, JK, {nT x p }     
        JF = np.zeros([nT,p])
        JK = JF
        JFK = np.concatenate([JF, JK],1)
        JFK = JFK.astype(np.complex128)

        # Add dimensions to A and h_output as they lose them during the lopp
        if len(A.shape)==1:
            A=np.atleast_2d(A).T
            
    #    # try to parallelise
    #    for i in range(2*p):
    #        print("inner-for", i)
    #        if i<p:
    #            JFK[:,i] = wf.CBIG_MFMem_rfMRI_diff_P1(funcA,A,h_output,i)
    #        else:
    #            JFK[:,i] = wf.CBIG_MFMem_rfMRI_diff_PC1(funcA,A,i-p)
    #    ## If wanting to test script quickly can use saved version of jfk and comment out the above loop
    #    # import pickle
    #    # JFK=pickle.load(open('./jfk', 'rb'))
    #
    #    JF = JFK[:, :p] # {nT x p}
    #    JK = JFK[:, p:]
        res1 = []
        for i in range(2*p):
            if i<p: # it is alright to get warning about losing the imaginary part for P1 but not alright for PC1
                res = dask.delayed(wf.CBIG_MFMem_rfMRI_diff_P1)(funcA,A,h_output,i)
                res1.append(res)
            else:
                res = dask.delayed(wf.CBIG_MFMem_rfMRI_diff_PC1)(funcA,A,i-p)
                res1.append(res)
        print(f'parrallel submitted {datetime.now().strftime("%H:%M:%S")}')
        res1 = dask.compute(*res1)
        JFK[:,:]=np.array(res1).T
        client.shutdown()
        shutil.rmtree(f'/users/k1201869/wang_model/dask_logs/dask_logs_{subject}')
        JF = JFK[:, :p] # {nT x p}
        JK = JFK[:, p:]
        print(f'parrallel finished {datetime.now().strftime("%H:%M:%S")}')

        # calculation of r, difference between emprical data y and model output h_output  
        r = y - h_output # {n*T x 1} 

        # prepare parallel computing of EM-algorithm
        A_old = A
        A_FDK = np.zeros([p,2])
        h_output_FDK = np.zeros([nT,2])
        r_FDK = r
        lembda_FDK = np.zeros([n,2])
        dlddpara_FDK = np.zeros([p,p,2])
        dldpara_FDK = np.zeros([p, 2])
        CC_check_FDK = np.zeros(2)
        CC_check_reg = np.zeros(4)

        LM_reg_on = np.array([1, 1]) # switcher of Levenberg-Marquardt regulation, started if correlation between FCs > 0.4
        #Estimation using Gauss-Newton and EM begin here,
        # try to parallelise
        for ppi in range(2):
            if ppi == 0:   #first ,J = JF     
                J = JF
                r = r_FDK
            else:     
                J = JK
                r = r_FDK

            # prepare lembda for lembda, Ce
            lembda = -3*np.ones([n])
            DiagCe = np.ones([nT]) #only have main diagonal entries
            for i in range(n):
                DiagCe[T*(i):T*(i)+T] = np.exp(lembda[i])
            #inv(Ce):  
            inv_DiagCe = DiagCe**-1  #try to only use diagonal element

            # preparation g, H, for g & H, see [2]
            g = np.zeros([n,1]) # initialization
            H = np.zeros([n,n]) # initialization

        #-------------------commence M-step loop %----------------------------------
            for mi in range(16):
                # first computing: pinv(J'*inv(Ce)*J)
                inv_JinvCeJ = np.zeros([p,nT])
                
                #step1: J'*inv(Ce)
                for i in range(p):
                    inv_JinvCeJ[i,:] = J[:,i].T * inv_DiagCe
                    # inv_JinvCeJ[i,:] = bsxfun(@times,J(:,i)', inv_DiagCe);
                
                #step2: J'*inv(Ce)*J              
                inv_JinvCeJ = np.matmul(inv_JinvCeJ, J)

                #step3: pinv(J'*inv(Ce)*J)
                inv_JinvCeJ = np.linalg.pinv(inv_JinvCeJ)

                # now computing:  %inv(Ce) * J * inv_JinvCeJ * J' * invCe
                P = np.zeros([nT,p])
                #step1: inv(Ce) * J  
                for i in range(p):
                    P[:,i] = (J[:,i]*np.squeeze(inv_DiagCe))
                #step2: (inv(Ce) * J) * inv_JinvCeJ * J'   
                P = np.matmul(np.matmul(P,inv_JinvCeJ),J.T)
                #step3:  -(inv(Ce) * J * inv_JinvCeJ * J') * inv(Ce)   
                # for i in range(nT):
                #     P[:,i] = P[:,i]* np.squeeze(-inv_DiagCe) #bsxfun(@times, P(:,i), -inv_DiagCe');
                P[:,:P.shape[0]//2]=(P[:,:P.shape[0]//2].T*np.squeeze(-inv_DiagCe)).T
                P[:,P.shape[0]//2:]=(P[:,P.shape[0]//2:].T*np.squeeze(-inv_DiagCe)).T
                #step4: invCe - (inv(Ce) * J * inv_JinvCeJ * J' * inv(Ce) )  
                np.fill_diagonal(P, np.diag(P)+inv_DiagCe)    

                # P = single(P);   %memory trade off
                # g(i) = -0.5*trace(P*exp(lembda(i))*Q(i))+0.5*r'*invCe*exp(lembda(i))*Q(i)*invCe*r;  {n x 1}
                #                         d  Ce
                # exp(lembda(i))*Q(i) =  -- ---
                #                         d  lembda(i)
                # see [2,3]
                
                for i in range(n):                                           
                    #step1: 0.5*r'*invCe*exp(lembda(i))*Q(i) 
                    g[i] = -0.5*np.exp(lembda[i])*np.trace(P[T*i:T*i+T][:, T*i:T*i+T])
                    #step2: (0.5*r'*invCe*exp(lembda(i))*Q(i))*invCe*r                                                      
                    g_rest = 0.5*np.matmul((r.T*inv_DiagCe)*np.exp(lembda[i]), wf.CBIG_MFMem_rfMRI_matrixQ(i,n,T))# CBIG_MFMem_rfMRI_matrixQ is used to caculate Q(i)
                    g_rest = np.matmul((g_rest*inv_DiagCe), r)
                    #step3:
                    g[i] = g[i] + g_rest

                #H(i,j) = 0.5*trace(P*exp(lembda(i))*Q(i)*P*exp(lembda(j))*Q(j)); {n x n}
                # see [2,3]

                for i in range(n):
                    for j in range(n):
                        Pij = P[T*i:T*i+T,:][:,T*j:T*j+T]
                        Pji = P[T*j:T*j+T,:][:,T*i:T*i+T]
                        H[i,j] = 0.5*np.exp(lembda[i])*np.exp(lembda[j])*wf.CBIG_MFMem_rfMRI_Trace_AXB(Pij,Pji)

                #clear P Pij Pji
                P = []
                Pij = []
                Pji = []

                #update lembda
                d_lembda = g/H #This is ok for scalars H\g # % delta lembda 

                lembda = lembda + d_lembda;

                for i in range(n):
                    if lembda[i] >= 0:
                        lembda[i] = np.min([lembda[i], 10])
                    else:
                        lembda[i] = np.max([lembda[i], -10])

                # update Ce for E-step
                DiagCe = np.ones([1,nT])
                for i in range(n):
                    DiagCe[T*i:T*i+T] = np.exp(lembda[i])

                inv_DiagCe = DiagCe**-1

                # abort criterium of m-step
                if np.max(abs(d_lembda)) < 1e-2:
                    break

        #-------------------end M-step loop %----------------------------------
            print(lembda)
            lembda_FDK[:,ppi] = lembda
        #----------------E-step-----------------------------------------------
            #-------------------------------------------------------------------
            # 
            #dldpara:   1st. derivative, {p x 1}, used in Gauss-Newton search
            #           dldpara = J'*inv(Ce)*r + inv(Prior_C)*(A_Prior_E - A); 
            #
            #dlddpara:  inv, negativ, 2nd. derivative, {p x p}, used in Gauss-Newton search   
            #           dlddpara = (J'*inv(Ce)*J + inv(Prior_C));     
            #see [2,3]
            #-------------------------------------------------------------------
            
            JinvCe = np.zeros([p,nT])# %J'invCe
            for i in range(p):
                JinvCe[i,:] = J[:,i] * inv_DiagCe #;% J'%invCe <----- p x nT

            dlddpara = np.matmul(JinvCe,J) + invPrior_C # inv, negativ, von 2nd. derivative {p x p}  

            dldpara = np.matmul(JinvCe,r) + np.squeeze(np.matmul(invPrior_C, (np.squeeze(A_Prior_E) - np.squeeze(A))))# % 1st. derivative, {p x 1}

            JinvCe = []# %save the memory

            d_A = wf.matlab_divide(dlddpara, dldpara)
            # np.savetxt('./temp/dlddpara.csv', dlddpara.astype('float128'))

            A_FDK[:,ppi] = np.squeeze(A) + np.squeeze(d_A) # %newton-gauss, fisher scoring, update Para_E
            Para_E_new = np.exp(A_FDK[:,ppi])*np.squeeze(Prior_E)

            dPara_E_new = abs(Para_E_new - np.squeeze(Para_E))

            if any(dPara_E_new>0.5): #%paramter should not improve too much
                d_A = wf.matlab_divide(dlddpara+10*np.diag(np.diag(dlddpara)), dldpara)
                A_FDK[:,ppi] = np.squeeze(A) + d_A #newton-gauss, fisher scoring, update Para_E
                Para_E_new = np.exp(A_FDK[:,ppi])*np.squeeze(Prior_E)
                LM_reg_on[ppi] = 0
            
            h_output_FDK[:,ppi], CC_check_FDK[ppi] = funcP(np.atleast_2d(Para_E_new).T)
            r = y - h_output_FDK[:,ppi]

            dlddpara_FDK[:,:,ppi] = dlddpara
            dldpara_FDK[:,ppi] = dldpara
        #-----------------------end parallel computiong---------------------

        # ---comparision the Fitting improvement between using JF and JK, choose the better one---
        F_comparison = CC_check_FDK[0]
        K_comparison = CC_check_FDK[1]

        if F_comparison >= K_comparison:
            print('real won')
            A = (A_FDK[:,0])
            h_output = h_output_FDK[:,0]
            CC_check_step[step+1] = CC_check_FDK[0]
            lembda_step_save[:,step] = lembda_FDK[:,0]
            dlddpara = dlddpara_FDK[:,:,0]
            dldpara = dldpara_FDK[:,0]
            
            if CC_check_step[step+1] > 0.4:   #Levenberg-Marquardt regulation, started if correlation between FCs > 0.4
                LM_on = LM_reg_on[0]
            else:
                LM_on = 0
            
        else:
            print('imaginary won')
            A = A_FDK[:,1]
            h_output = h_output_FDK[:,1]
            CC_check_step[step+1] = CC_check_FDK[1]
            lembda_step_save[:,step] = lembda_FDK[:,1]
            dlddpara = dlddpara_FDK[:,:,1]
            dldpara = dldpara_FDK[:,1]
            
            if CC_check_step[step+1] > 0.4: #Levenberg-Marquardt regulation, started if correlation between FCs > 0.4
                LM_on = LM_reg_on[1]
            else:
                LM_on = 0
        # -----------------End comparision------------------------------------------------

        #now adding levenberg-Maquadrat
        if LM_on == 1:
            lembda = lembda_step_save[:,step]
            DiagCe = np.ones(nT)
            for i in range(n):
                DiagCe[T*i:T*i+T] = np.exp(lembda[i])
            
            inv_DiagCe = DiagCe**-1  #try to only use diagonal element
            
            #regulation value table
            reg_reg = [0,1,10,100]
            Nreg = len(reg_reg)

            A_reg = np.zeros([p,Nreg])
            h_output_reg = np.zeros([nT,Nreg])
            lembda_reg = np.zeros([n,Nreg])

            # transfer results for reg = 0  
            A_reg[:,0] = A
            h_output_reg[:,0] = h_output
            CC_check_reg[0] = CC_check_step[step+1]

            #<--------begin parallel computing-------------------------------
            for ppi in range(1,Nreg):
                reg = reg_reg[ppi]
                A = A_old
                d_A = wf.matlab_divide((dlddpara+reg*np.diag(np.diag(dlddpara))), dldpara) #LM
                A_reg[:,ppi] = np.squeeze(A) + d_A #newton-gauss, fisher scoring, update Para_E
                Para_E_new = np.exp(A_reg[:,ppi])*np.squeeze(Prior_E)
                h_output_reg[:,ppi], CC_check_reg[ppi] = funcP(np.atleast_2d(Para_E_new).T)
                r = y - h_output_reg[:,ppi]
            #<--------------------end parallel computing------------------------------
            
            del DiagCe 
            del inv_DiagCe

            T_comparision = CC_check_reg
            # CC_check_step_save[step+1] = np.max(T_comparision)
            T_comparision_indx = np.argmax(T_comparision)

            #disp(['chosen reg is: ' num2str(reg_reg(T_comparision_indx))]);
            A = A_reg[:,T_comparision_indx]
            h_output = h_output_reg[:,T_comparision_indx]
            CC_check_step[step+1] = CC_check_reg[T_comparision_indx]  
        #--------------------------------------------------------------------------------   

        #update results, check abbort criterium
        Para_E = np.atleast_2d(np.exp(A)*np.squeeze(Prior_E)).T
        d_Para_E = Para_E - Para_E_step

        dN = np.sqrt(sum(d_Para_E**2))

        rrr[:,step] = 1-(np.var(y-h_output)/np.var(y)) #goodness of fit
        rrr_z[:,step]  = np.corrcoef(np.arctanh(h_output), np.arctanh(y))[1,0] # %correlation between 2 FCs

        Para_E_step_save[:,step] = np.squeeze(Para_E)

        print(f'goodness of fit correlation {rrr_z[:,step]}')

        #Abort criterium of total estimation
        if ((step>5)and(rrr[:,step] >= 0.99 or (dN < 1e-5 and rrr_z[:,step] > 0.4))):
            break
        if ((step>100)and(rrr_z[:,step] - rrr_z[:,step-1]<=-0.10)):
            break #% stop if we find a bifucation edge, it should be a good solution (Deco et al., 2013)

         
        
        # Save variables so in case of crash we can recover
        temp_directory = f'./temp/{subject}/{version}'
        if not os.path.exists(temp_directory):
            os.makedirs(temp_directory)
        saved_variables = {'step': step, 'Para_E': Para_E, 'rrr':rrr, 'rrr_z':rrr_z, 'Para_E_step_save':Para_E_step_save, 'lembda_step_save':lembda_step_save, 'CC_check_step': CC_check_step}
        pickle.dump(saved_variables, open(f'{temp_directory}/saved_variables_{step}.pkl', 'wb'))
        try:
            os.remove(f'{temp_directory}/saved_variables_{step-1}.pkl')
        except:
            pass
        
        #counter
        step = step + 1
    #<-----------------------------------------End while loop, End estimation ---------

    # End estimation, save result

    #find the best results
    rrr_z_max = np.max(rrr_z)
    indx_max = np.argmax(rrr_z)
    Para_E = Para_E_step_save[:,indx_max]

    print(rrr_z_max)
    print(Para_E)

    #save estimated parameter and correlation between FCs (z-transfered) to a text file
    final_results = np.append(Para_E, rrr_z_max)
    outfile = f'{results_dir}/output_{subject}_{version}.txt'
    np.savetxt(outfile, final_results)