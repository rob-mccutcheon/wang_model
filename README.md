# WANG MODEL
Python 3 translation of the circuit model and fitting algorithm   originally in matlab at: https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/fMRI_dynamics/Wang2018_MFMem

NB functional connectivity matrices should not be supplied containig untransformed pearson coeficients.

## DATASETS
### HCP TEST-RETEST

1. Test and retest data downloaded from https://db.humanconnectome.org/ to NaN $HOME/e_i_modelling/data/hcp_test_retest
2. Processing code is on the NaN $HOME/e_i_modelling/code/hcp_test_retest: a)unzip.py, b)demean.sh c) parcellate.py
3. Connnectivity matrices transferred to rosalind form NaN with scp

### HCP EP
Data processed using HCP pipeline, scripts are at $HOME/hcp/code/HCPpipelines-4.2.0-rc.1/Examples/Scripts/

I was able to get the gradient coefficients file off the steiner prisma

In terms of modules you generally want freesurfer 6.0.0, fsl 6.0.1 and connectomewb 1.4.2. Matlab depends on the scripts

1. SetUpHCPPipeline.sh: Amend to link to required code (e.g. need to have downloaded Fix)
2. Prefreesurfer - need to be in a python environment with gradunwarp installed (r_env has this)
3. Freesurfer
4. Post Freesurfer
5. Generic Volume - may need to be in python environment with gradunwapr  (e.g. r_env)
6. Generic Surface
7. ICA Fix - use matlab runtime, make sure that anaconda 3.7 is loaded and that matlab unloaded, need to set enough memory:
setenv FSLQUEUE_OPTS "-l h_vmem=40G". Fix settings.sh needs to have: FSL_FIX_MCRROOT="${HOME}/hcp/code/matlabruntime/mr". Seems that evenn with this sometimes wrong verison of R loaded - try installing a package perhaps if not working, should be R 4.0
8. Post Fix - use matlab module - make sure matlab 9.5.0 loaded
9. MSMAll
10. Dedrift and resample
11. Parcellate

#### Fix
I am running it using matlab compiled runtime (download from mathworks), and r that is in anaconda 3.7, nneed to have installed the specific packages specifiedin the fix readme using devtools install_version

#### Specific subjects
1007, 1011 - missinng a unprocessed tw1 so excluded
1006, 1001 - spin echo field maps had to be renamed from '2' to '1'