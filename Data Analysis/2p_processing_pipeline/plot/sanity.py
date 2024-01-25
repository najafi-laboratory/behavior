#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def mean_trace_sess(
        neural_trial,
        ):
    time = np.arange(len(neural_trial['raw']['vol_img_bin']))
    vol_img_bin = neural_trial['raw']['vol_img_bin']
    vol_img_bin[vol_img_bin>1] = 1
    vol_img_bin[vol_img_bin<1] = 0
    diff_vol_img_bin = np.diff(vol_img_bin, append=0)
    idx_up = np.where(diff_vol_img_bin == 1)[0]+1
    time_img = time[idx_up]
    fluo = neural_trial['raw']['fluo_ch2']
    vol_stim_bin = neural_trial['raw']['vol_stim_bin']
    mean_fluo = np.mean(fluo, axis=0)
    fig, axs = plt.subplots(1, figsize=(10, 8))
    axs.plot(time, vol_stim_bin, color='dodgerblue', label='vol_stim_bin')
    axs.plot(time_img, mean_fluo*5, color='coral', label='mean_fluo')
    axs.set_xlabel('time / (ms)')
    axs.legend()
    fig.suptitle('Mean response for {} neurons'.format(fluo.shape[0]))


def alignment_check(
        neural_trial,
        ):
    time = np.arange(len(neural_trial['raw']['vol_img_bin']))
    vol_img_bin = neural_trial['raw']['vol_img_bin']
    diff_vol_img_bin = np.diff(vol_img_bin, append=0)
    idx_up = np.where(diff_vol_img_bin == 1)[0]+1
    time_1 = time[idx_up]
    fluo = neural_trial['raw']['fluo_ch2']
    vol_stim_bin = neural_trial['raw']['vol_stim_bin']
    mean_fluo_1 = np.mean(fluo, axis=0)
    fluo = neural_trial['0']['fluo_ch2']
    mean_fluo_2 = np.mean(fluo, axis=0)
    time_2 = neural_trial['0']['time']
    stim = neural_trial['0']['stim']
    fig, axs = plt.subplots(1, figsize=(10, 8))
    axs.plot(time, vol_img_bin, label='org img', lw=0.2)
    axs.plot(time, vol_stim_bin, label='org stim')
    axs.plot(time_2, stim, linestyle='--', label='aligned stim')
    axs.plot(time_1, mean_fluo_1*5, label='org mean_fluo', marker='.', markersize=5)
    axs.plot(time_2, mean_fluo_2*5, label='trial mean_fluo', marker='.', markersize=5)
    axs.set_xlabel('time / (ms)')
    #axs.set_xlim([20900, 21060])
    axs.legend()
       


def frame_dur(vol_img_bin):
    diff_vol_img_bin = np.diff(vol_img_bin, append=0)
    time_idx = np.where(diff_vol_img_bin == 1)[0]+1
    diff_time_idx = np.diff(time_idx)
    fig, axs = plt.subplots(1, figsize=(10, 8))
    axs.hist(diff_time_idx, bins=1000)

