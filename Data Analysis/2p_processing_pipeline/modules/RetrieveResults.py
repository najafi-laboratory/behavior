#!/usr/bin/env python3

import os
import h5py
import numpy as np


# read saved ops.npy given a folder in ./results.
'''
for save_folder in [
        'FN6_P_spont1_121623',
        'FN6_P_spont2_121623',
        'FN8_P_pp_020824',
        'FN8_P_spont1_121823',
        'FN8_P_spont2_121823',
        'FN9_P_pp_020824',
        'FN9_P_pp_021424',
        'FN12_C_omi_021324',
        'FN13_P_omi_021324',
        'FN13_P_pp_020724',
        'FN13_P_pp_021424']:

    plot_fig5(ops)

'''
# ops = read_ops('FN12_C_omi_021324')
def read_ops(save_folder):
    ops = np.load(
        os.path.join('./results', save_folder, 'ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join('./results', save_folder)
    return ops


# read raw_traces.h5.

def read_raw_traces(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'raw_traces.h5'),
        'r')
    raw_traces = dict()
    for k in f['raw'].keys():
        raw_traces[k] = np.array(f['raw'][k])
    f.close()
    return raw_traces


# read raw_voltages.h5

def read_raw_voltages(ops):
    try:
        f = h5py.File(
            os.path.join(ops['save_path0'], 'raw_voltages.h5'),
            'r')
        raw_voltages = dict()
        for k in f['raw'].keys():
            raw_voltages[k] = np.array(f['raw'][k])
        f.close()
        return raw_voltages
    except:
        print('Fail to read voltage data')
        return None


# read neural_trials.h5

def read_neural_trials(ops):
    try:
        f = h5py.File(
            os.path.join(ops['save_path0'], 'neural_trials.h5'),
            'r')
        neural_trials = dict()
        for trial in f['trial_id'].keys():
            neural_trials[trial] = dict()
            for data in f['trial_id'][trial].keys():
                neural_trials[trial][data] = np.array(f['trial_id'][trial][data])
        f.close()
        neural_trials = add_trial_types(neural_trials)
        return neural_trials
    except:
        print('Fail to read trial data')
        return None


# tentative function to hard code trial types.

def add_trial_types(neural_trials):
    from sklearn.mixture import GaussianMixture
    def frame_dur(stim, time):
        diff_stim = np.diff(stim, prepend=0)
        idx_up   = np.where(diff_stim == 1)[0]
        idx_down = np.where(diff_stim == -1)[0]
        dur_high = time[idx_down] - time[idx_up]
        dur_low  = time[idx_up[1:]] - time[idx_down[:-1]]
        return [dur_high, dur_low]
    def get_mean_std(isi):
        gmm = GaussianMixture(n_components=2)
        gmm.fit(isi.reshape(-1,1))
        std = np.mean(np.sqrt(gmm.covariances_.flatten()))
        return std
    thres = 25
    trial_type = []
    for i in range(len(neural_trials)):
        stim = neural_trials[str(i)]['stim']
        time = neural_trials[str(i)]['time']
        [_, isi] = frame_dur(stim, time)
        std = get_mean_std(isi)
        trial_type.append(std)
    trial_type = np.array(trial_type)
    trial_type[trial_type<thres] = 2
    trial_type[trial_type>thres] = 1
    neural_trials['trial_type'] = trial_type
    return neural_trials


# read mask.h5.

def read_mask(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'mask.h5'),
        'r')
    mask = dict()
    mask['label'] = np.array(f['label'])
    for ch in f['mask'].keys():
        mask[ch] = dict()
        for k in f['mask'][ch].keys():
            mask[ch][k] = np.array(f['mask'][ch][k])
    f.close()
    return mask


# main function to read completed results.

def run(ops):
    mask = read_mask(ops)
    raw_traces = read_raw_traces(ops)
    raw_voltages = read_raw_voltages(ops)
    neural_trials = read_neural_trials(ops)
    return [mask, raw_traces, raw_voltages, neural_trials]

