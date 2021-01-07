import numpy as np

def vec2mat(a):
    n = int(np.sqrt(len(a)*2))+1
    mask = ~np.tri(n,dtype=bool, k=0) # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n,n),dtype='float64')
    out[mask] = a
    out = out + out.T
    return out

def mat2vec(a):
    return a[np.triu_indices(a.shape[0], k=1)]