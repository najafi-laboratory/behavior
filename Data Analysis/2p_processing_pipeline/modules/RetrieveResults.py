#!/usr/bin/env python3

import os
import h5py
import numpy as np
from datetime import datetime


def read_ops(save_folder):
    ops = np.load(
        os.path.join('./results', save_folder, 'ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join('./results', save_folder)
    return ops


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


def run(save_folder):
    ops = read_ops(save_folder)
    neural_trial = read_neural_trial(ops)
    mask = read_mask(ops)
    return [ops, neural_trial, mask]

