
#This all pasted into tobys desktop, using the environment python36
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import clean_img
import numpy as np
import os
import nibabel as nib

# masker = NiftiLabelsMasker(labels_img='/home/k1593571/Desktop/OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_2mm_v2.nii', standardize=False)
masker = NiftiLabelsMasker(labels_img='/home/k1593571/Desktop/aparcaseg.nii.gz', standardize=False)


subjects = os.listdir('/home/k1593571/Desktop/resting_state/subjects/')[:-1]

for subject in subjects:
    try:
        img = f'/home/k1593571/Desktop/resting_state/subjects/{subject}/wdenoised_func_data.nii'
        cleaned_img = clean_img(img, detrend=False, standardize=False, ensure_finite=True)
        time_series = masker.fit_transform(cleaned_img)
        cm = np.corrcoef(time_series.T)
        np.savetxt(f'/home/k1593571/Desktop/resting_state/dk_matrices2/{subject}_dk_cm.txt',cm)
    except:
        print(subject)



#This all pasted into rob desktop, using the environment r_ennv
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import clean_img
import numpy as np
import os
import nibabel as nib

# masker = NiftiLabelsMasker(labels_img='/home/k1593571/Desktop/OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_2mm_v2.nii', standardize=False)
masker = NiftiLabelsMasker(labels_img='/home/k1201869/Desktop/aparcaseg.nii.gz', standardize=False)

subjects = ['1050_01_MR', '1051_01_MR','1052_01_MR']
for subject in subjects:
    try:
        img = f'/home/k1201869/e_i_modelling/data/processed/{subject}/MNINonLinear/Results/concat/concat_hp2000_clean.nii.gz'
        cleaned_img = clean_img(img, detrend=False, standardize=False, ensure_finite=True)
        time_series = masker.fit_transform(cleaned_img)
        cm = np.corrcoef(time_series.T)
        np.savetxt(f'/home/k1201869/e_i_modelling/data/dk_vol/{subject}_dk.txt',cm)
    except:
        print(subject)



#explor mindboggle atlas
atlas = nib.load('/users/k1201869/glucog/OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_2mm_v2.nii.gz')
atlas = nib.load('/users/k1201869/glucog/aparcaseg.nii.gz')

labels = np.unique(atlas.get_data())[42:]

len(labels)


labels[34:] #62 rois

#the SC has 698 rois - need to delete 6
[0,30,31,34,63,64]
0,

unknowns = np.isin(atlas.get_data(), [91,92,630,631,632])

plt.imshow(unknowns[:,:,30])
plt.imshow(atlas.get_data()[:,:,30])
plt.imshow(unknowns[:,:,40])