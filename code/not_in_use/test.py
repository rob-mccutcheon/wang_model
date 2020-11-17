SC = np.loadtxt(open(f"{data_dir}/SC.csv", "rb"), delimiter=",")
SC_thresh = (SC/np.max(np.max(SC)))*0.2
SC_glasser = np.loadtxt(open(f"{data_dir}/SC_glasser.csv", "rb"), delimiter=",")


import seaborn as sns

np.max(SC_thresh)
np.sum(SC_thresh>0)/(SC_thresh.shape[0]**2-SC_thresh.shape[0])
np.sum(SC_glasser>0)/(SC_glasser.shape[0]**2-SC_glasser.shape[0])

sns.distplot(SC_thresh.flatten())
sns.distplot((SC_glasser*0.2).flatten())


import matplotlib.pyplot as plt
plt.imshow(SC_glasser[:,:360][:360,:]>0)
plt.imshow(SC_glasser[:,350:][350:,:]>0)
plt.imshow(FC<-0)


plt.imshow(SC_thresh)


np.mean(SC_thresh)/np.mean(SC_glasser)
np.mean(SC)
np.mean(SC_glasser)


import dask
import numpy as np

# Set up client
cluster = SLURMCluster(cores=1, memory='16 GB', 
                queue='brc', interface='em1',
                log_directory='./dask_logs')
cluster.scale(jobs=2)
client = distributed.Client(cluster)

# Fuction to be  parrallelised
def nT_loop(i, P,inv_DiagCe):
    x = P[:,i]* np.squeeze(-inv_DiagCe)
    return x

P = np.random.rand(64620, 64620)
inv_DiagCe = np.random.rand(64620)
P2 = da.from_array(P, chunks=100)

# Run loop
res1=[]
for ppi in range(2):
    res = dask.delayed(parloop)(ppi, JF, JK, r_FDK, n, nT, p)
    res1.append(res)

# Compute results
res1 = dask.compute(*res1)


def f(x):
    return x.sum()

N = 10000
x = np.random.randn(N, N)
x.sum()
r1 = client.submit(f, x).result()

x_scattered = client.scatter(x)
r2 = client.submit(f, x_scattered).result()




print(f'start mi{mi} range(n) 2loop')
print(datetime.now().strftime("%H:%M:%S"))  
def loop2(i, j, lembda,Pij,Pji):
    import wang_functions as wf
    a= 0.5*np.exp(lembda[i])*np.exp(lembda[j])*wf.CBIG_MFMem_rfMRI_Trace_AXB(Pij,Pji)
    return a
res1=[]
n=10
for i in range(n):
    for j in range(n):
        Pij = P[T*i:T*i+T,:][:,T*j:T*j+T]
        Pji = P[T*j:T*j+T,:][:,T*i:T*i+T]
        # H[i,j] = 0.5*np.exp(lembda[i])*np.exp(lembda[j])*wf.CBIG_MFMem_rfMRI_Trace_AXB(Pij,Pji)
        res = dask.delayed(loop2)(i, j, lembda,Pij,Pji)
        res1.append(res)
        res1.append('a')
res1 = dask.compute(*res1)
        print(f'finish mi{mi} range(n) 2loop')
print(datetime.now().strftime("%H:%M:%S"))  


import numpy as np
import pandas as pd
import time
import random

def costly_simulation(list_param):
    time.sleep(random.random())
    return sum(list_param)

input_params = pd.DataFrame(np.random.random(size=(500, 4)),
                            columns=['param_a', 'param_b', 'param_c', 'param_d'])
input_params.head()
results=[]
for parameters in input_params.values[:10]:
    result = costly_simulation(parameters)
    results.append(result)

lazy_results = []
%%time

for parameters in input_params.values[:10]:
    lazy_result = dask.delayed(costly_simulation)(parameters)
    lazy_results.append(lazy_result)
dask.compute(*lazy_results)

wf.CBIG_MFMem_rfMRI_Trace_AXB(Pij,Pji)

np.trace(np.matmul(Pij, Pji))