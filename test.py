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