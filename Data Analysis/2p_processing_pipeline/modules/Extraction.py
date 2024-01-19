#!/usr/bin/env python3

import os
import h5py
import numpy as np
from datetime import datetime
from suite2p import extraction_wrapper
from suite2p.extraction import preprocess
from suite2p.extraction import oasis


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


def normalization(
        ops,
        F,
        Fneu
        ):
    fluo = F.copy() - ops['neucoeff']*Fneu
    fluo = (fluo - np.min(fluo, axis=1).reshape(-1,1)) / (
        np.max(fluo, axis=1).reshape(-1,1) - np.min(fluo, axis=1).reshape(-1,1))
    fluo = preprocess(
            F=fluo,
            baseline=ops['baseline'],
            win_baseline=ops['win_baseline'],
            sig_baseline=ops['sig_baseline'],
            fs=ops['fs'],
            prctile_baseline=ops['prctile_baseline']
        )
    return fluo


def moving_average(
        fluo,
        win
        ):
    conv = lambda x: np.convolve(x, np.ones(win)/win, mode='same')
    mean_fluo = np.apply_along_axis(conv, axis=1, arr=fluo)
    return mean_fluo


def spike_detect(
        ops,
        fluo
        ):
    spikes = oasis(
        F=fluo,
        batch_size=ops['batch_size'],
        tau=0.5,
        fs=ops['fs'])
    return spikes


def save_traces(
        ops,
        fluo_ch1, mean_fluo_ch1, spikes_ch1,
        fluo_ch2, mean_fluo_ch2, spikes_ch2
        ):
    f = h5py.File(os.path.join(
        ops['save_path0'], 'temp', 'traces.h5'), 'w')
    dict_group = f.create_group('traces')
    dict_group['fluo_ch1'] = fluo_ch1
    dict_group['fluo_ch2'] = fluo_ch2
    dict_group['mean_fluo_ch1'] = mean_fluo_ch1
    dict_group['mean_fluo_ch2'] = mean_fluo_ch2
    dict_group['spikes_ch1'] = spikes_ch1
    dict_group['spikes_ch2'] = spikes_ch2
    f.close()


def run(ops, stat_ref, f_reg_ch1, f_reg_ch2):
    print('===============================================')
    print('======= extracting fluorescence signals =======')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    [stat,
     F_ch1, Fneu_ch1,
     F_ch2, Fneu_ch2] = get_fluorescence(
         ops,
         stat_ref,
         f_reg_ch1,
         f_reg_ch2)
    print('Fluorescence extraction completed')
    fluo_ch1 = normalization(ops, F_ch1, Fneu_ch1)
    fluo_ch2 = normalization(ops, F_ch2, Fneu_ch2)
    mean_fluo_ch1 = moving_average(fluo_ch1, ops['average_window'])
    mean_fluo_ch2 = moving_average(fluo_ch2, ops['average_window'])
    print('Signal normalization completed')
    spikes_ch1 = spike_detect(ops, mean_fluo_ch1)
    spikes_ch2 = spike_detect(ops, mean_fluo_ch2)
    print('Spike deconvolution completed')
    save_traces(ops,
        fluo_ch1, mean_fluo_ch1, spikes_ch1,
        fluo_ch2, mean_fluo_ch2, spikes_ch2)
    print('Traces data saved')
    return []

