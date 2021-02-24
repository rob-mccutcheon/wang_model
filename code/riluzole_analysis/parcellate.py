
#This all pasted into tobys desktop, using the environment python36
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import clean_img
import numpy as np
import os
import nibabel as nib

masker = NiftiLabelsMasker(labels_img='/home/k1593571/Desktop/OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_2mm_v2.nii', standardize=False)

subjects = os.listdir('/home/k1593571/Desktop/resting_state/subjects/')[:-1]

for subject in subjects:
    try:
        img = f'/home/k1593571/Desktop/resting_state/subjects/{subject}/wdenoised_func_data.nii'
        cleaned_img = clean_img(img, detrend=False, standardize=False, ensure_finite=True)
        time_series = masker.fit_transform(cleaned_img)
        cm = np.corrcoef(time_series.T)
        np.savetxt(f'/home/k1593571/Desktop/resting_state/dk_matrices/{subject}_cm.txt',cm)
    except:
        print(subject)


#explor mindboggle atlas
atlas = nib.load('/users/k1201869/glucog/OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_2mm_v2.nii.gz')

labels = np.unique(atlas.get_data())[1:]


labels[34:] #62 rois

#the SC has 698 rois - need to delete 6
[0,30,31,34,63,64]
0,

unknowns = np.isin(atlas.get_data(), [91,92,630,631,632])

plt.imshow(unknowns[:,:,30])
plt.imshow(atlas.get_data()[:,:,30])
plt.imshow(unknowns[:,:,40])