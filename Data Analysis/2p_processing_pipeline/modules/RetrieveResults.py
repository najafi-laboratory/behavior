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
    print('===============================================')
    print('======= read saved trial data and masks =======')
    print('===============================================')
    # ops = read_ops('FN8_PPC_121923')
    neural_trial = read_neural_trial(ops)
    mask = read_mask(ops)
    return [neural_trial, mask]

