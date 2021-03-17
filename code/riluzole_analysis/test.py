import numpy as np

a= np.loadtxt('../../data/hcp_scz/1001_01_MR/1001_01_MR_glasser_pearson.csv', delimiter=',')
b= np.loadtxt('../../data/hcp_scz/1002_01_MR/1002_01_MR_glasser_pearson.csv', delimiter=',')

np.corrcoef(a.flatten(), b.flatten())
