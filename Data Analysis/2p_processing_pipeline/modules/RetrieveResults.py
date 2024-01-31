#!/usr/bin/env python3

import os
import h5py
import numpy as np


# read saved ops.npy given a folder in ./results.

def read_ops(save_folder):
    ops = np.load(
        os.path.join('./results', save_folder, 'ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join('./results', save_folder)
    return ops


# read neural_trial.h5.

def read_neural_trial(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'neural_trial.h5'),
        'r')
    neural_trial = dict()
    for trial in f.keys():
        neural_trial[trial] = dict()
        for data in f[trial].keys():
            neural_trial[trial][data] = np.array(f[trial][data])
    f.close()
    neural_trial = add_trial_types(neural_trial)
    return neural_trial


# tentative function to hard code trial types.

def add_trial_types(neural_trial):
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
    for i in range(len(neural_trial)-1):
        stim = neural_trial[str(i)]['stim']
        time = neural_trial[str(i)]['time']
        [_, isi] = frame_dur(stim, time)
        std = get_mean_std(isi)
        trial_type.append(std)
    trial_type = np.array(trial_type)
    trial_type[trial_type<thres] = 2
    trial_type[trial_type>thres] = 1
    neural_trial['trial_type'] = trial_type
    return neural_trial


# read mask.h5.

def read_mask(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'mask.h5'),
        'r')
    mask = dict()
    for ch in f['mask'].keys():
        mask[ch] = dict()
        for k in f['mask'][ch].keys():
            mask[ch][k] = np.array(f['mask'][ch][k])
    f.close()
    return mask


# main function to read completed results.

def run(ops):
    # ops = read_ops('FN8_PPC_011824')
    neural_trial = read_neural_trial(ops)
    mask = read_mask(ops)
    return [neural_trial, mask]

