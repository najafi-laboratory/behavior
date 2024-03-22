#!/usr/bin/env python3

import os
import h5py
import numpy as np

'''
ops = read_ops('C2')
ops['spike_thres'] = 1
'''


# read saved ops.npy given a folder in ./results.

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
        return neural_trials
    except:
        print('Fail to read trial data')
        return None


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
    mask = read_mask(ops)
    raw_traces = read_raw_traces(ops)
    raw_voltages = read_raw_voltages(ops)
    neural_trials = read_neural_trials(ops)
    return [mask, raw_traces, raw_voltages, neural_trials]
