import numpy as np
import pandas as pd
import pingouin as pg

def load_cmzs(subjects, data_path, connectivity_type):
    '''load hcp test retest connectivity matrices, connectivity type are cm(pearson), pcm (partial)
    , and the z -transformed versions of each'''
    test_cmzs = []
    retest_cmzs = []
    for subject in subjects:
        test_cmz = np.loadtxt(f'{data_path}/test/{subject}/{connectivity_type}.csv')
        retest_cmz = np.loadtxt(f'{data_path}/retest/{subject}/{connectivity_type}.csv')
        np.fill_diagonal(test_cmz, 0)
        np.fill_diagonal(retest_cmz, 0)
        test_cmzs.append(np.loadtxt(f'{data_path}/test/{subject}/{connectivity_type}.csv'))
        retest_cmzs.append(np.loadtxt(f'{data_path}/retest/{subject}/{connectivity_type}.csv'))
    test_cmzs = np.array(test_cmzs)
    retest_cmzs = np.array(retest_cmzs)
    return test_cmzs, retest_cmzs

def retest_correlation(test_data, retest_data):
    ''' test/retest arrays should be num_nodes * num_subjects.
        returns  'rankings' - where the correlation between test scan 
        and retest scan fell in the entire retest dataset (i.e. rank of 0
        means was fingerprinted correctly. And the corrleations matrix - correl array'''
    correl_array = []
    rankings = []
    for i in range(test_data.shape[0]):
        test = test_data[i,]
        corrs = np.corrcoef(test, retest_data)[1:,0]
        correl_array.append(corrs)
        # find ranking
        true = corrs[i]
        rank = np.where(np.sort(corrs)[::-1]==true)
        rankings.append(rank)
    return rankings, correl_array

def retest_reliability(subjects, test_data, retest_data):
    node_reliabilities = []
    for i in range(len(test_data[1,:])):
        reliability_df = pd.DataFrame({'subjects': subjects+subjects,
                                    'session': len(subjects)*['test']+len(subjects)*['retest'],
                                    'strength': np.hstack([test_data[:,i], retest_data[:,i]])})

        data = pg.read_dataset('icc')
        icc = pg.intraclass_corr(data=reliability_df, targets='subjects', raters='session',
                                ratings='strength').round(3)
        icc1 = icc[icc['Type']=='ICC1k']['ICC'].values[0]
        node_reliabilities.append(icc1)
    return node_reliabilities


def load_parameters(subjects, method, parameter_choice, data_path):
    test_params = []
    retest_params = []
    for subject in subjects:
        test_param = []
        retest_param = []
        for i in range(10):
            try:
                test_param.append(np.loadtxt(f'{data_path}/test/output_{subject}_{i}.txt'))
            except OSError:
                pass
            try:
                retest_param.append(np.loadtxt(f'{data_path}/retest/output_{subject}_{i}.txt'))
            except OSError:
                pass
        if method == 'max':
            test_idx = np.argmax(np.array(test_param)[:,-1])
            test_params.append(test_param[test_idx][parameter_choice[0]:parameter_choice[1]])
            retest_idx = np.argmax(np.array(retest_param)[:,-1])
            retest_params.append(retest_param[retest_idx][parameter_choice[0]:parameter_choice[1]])
        if method == 'mean':
            test_params.append(np.mean((np.array(test_param)[:, parameter_choice[0]:parameter_choice[1]]), axis=0))
            retest_params.append(np.mean((np.array(retest_param)[:, parameter_choice[0]:parameter_choice[1]]), axis=0))
        if method == 'rand':
            test_idx = np.random.randint(len(test_param))
            retest_idx = np.random.randint(len(retest_param))
            test_params.append(test_param[test_idx][parameter_choice[0]:parameter_choice[1]])
            retest_params.append(retest_param[retest_idx][parameter_choice[0]:parameter_choice[1]])
    test_params = np.array(test_params)
    retest_params = np.array(retest_params)
    return test_params, retest_params