#!/usr/bin/env python3

import gc
import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from datetime import datetime
from suite2p import extraction_wrapper
from suite2p.extraction import oasis


# extract fluorescence signals from masks given as stat file.

def get_fluorescence(
        ops,
        stat,
        f_reg_ch1,
        f_reg_ch2
        ):
    stat, F_ch1, Fneu_ch1, F_ch2, Fneu_ch2 = extraction_wrapper(
        stat = stat,
        f_reg = f_reg_ch1,
        f_reg_chan2 = f_reg_ch2,
        ops = ops)
    return [stat, F_ch1, Fneu_ch1, F_ch2, Fneu_ch2]


# process the fluorescence signals.

def get_dff(
        ops,
        fluo,
        neuropil,
        ):
    # correct with neuropil signals.
    fluo = fluo.copy() - ops['neucoeff']*neuropil
    # baseline subtraction with window to compute df/f.
    f0 = gaussian_filter(fluo, [0., ops['sig_baseline']])
    # scale with individual variance.
    for j in range(fluo.shape[0]):
        fluo[j,:] = (fluo[j,:] - f0[j,:]) / ( f0[j,:] * np.std(f0[j,:]) )
    return fluo


# run spike detection on fluorescence signals.

def spike_detect(
        ops,
        fluo
        ):
    # oasis for spike detection.
    spikes = oasis(
        F=fluo,
        batch_size=ops['batch_size'],
        tau=ops['tau'],
        fs=ops['fs'])
    # threshold spikes based on variance.
    for i in range(spikes.shape[0]):
        s = spikes[i,:].copy()
        thres = np.mean(fluo[i,:]) + ops['spike_thres'] * np.std(fluo)
        s[s<thres] = 0
        spikes[i,:] = s
    return spikes


# save the trace data.

def save_traces(
        ops,
        fluo_ch1, neuropil_ch1, spikes_ch1,
        fluo_ch2, neuropil_ch2, spikes_ch2
        ):
    # file structure:
    # ops['save_path0'] / raw_traces.h5
    # -- raw
    # ---- fluo_ch1
    # ---- fluo_ch2
    # ---- neuropil_ch1
    # ---- neuropil_ch2
    # ---- spikes_ch1
    # ---- spikes_ch2
    f = h5py.File(os.path.join(
        ops['save_path0'], 'raw_traces.h5'), 'w')
    dict_group = f.create_group('raw')
    dict_group['fluo_ch1'] = fluo_ch1
    dict_group['fluo_ch2'] = fluo_ch2
    dict_group['neuropil_ch1'] = neuropil_ch1
    dict_group['neuropil_ch2'] = neuropil_ch2
    dict_group['spikes_ch1'] = spikes_ch1
    dict_group['spikes_ch2'] = spikes_ch2
    f.close()


# delete registration binary files.

def clean_reg_bin(
        ops,
        f_reg_ch1, f_reg_ch2
        ):
    del f_reg_ch1
    del f_reg_ch2
    gc.collect()
    os.remove(os.path.join(ops['save_path0'], 'temp', 'reg_ch1.bin'))
    os.remove(os.path.join(ops['save_path0'], 'temp', 'reg_ch2.bin'))
    
    
# main function for fluorescence signal extraction from ROIs.

def run(ops, stat_func, f_reg_ch1, f_reg_ch2):
    print('===============================================')
    print('======= extracting fluorescence signals =======')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    [stat,
     fluo_ch1, neuropil_ch1,
     fluo_ch2, neuropil_ch2] = get_fluorescence(
         ops,
         stat_func,
         f_reg_ch1,
         f_reg_ch2)
    print('Fluorescence extraction completed')

    fluo_ch1 = get_dff(ops, fluo_ch1, neuropil_ch1)
    fluo_ch2 = get_dff(ops, fluo_ch2, neuropil_ch2)
    print('Signal dff completed')

    spikes_ch1 = spike_detect(ops, fluo_ch1)
    spikes_ch2 = spike_detect(ops, fluo_ch2)
    print('Spike deconvolution completed')

    save_traces(ops,
        fluo_ch1, neuropil_ch1, spikes_ch1,
        fluo_ch2, neuropil_ch2, spikes_ch2)
    print('Traces data saved')
    
    clean_reg_bin(ops, f_reg_ch1, f_reg_ch2)

