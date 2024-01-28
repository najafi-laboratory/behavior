#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

from modules import RetrieveResults


# add zeros to both sides of a sequence.
def pad_zeros(data, l_frames, r_frames):
    n, t = data.shape
    padded_data = np.zeros((n, l_frames + t + r_frames))
    padded_data[:, l_frames:l_frames + t] = data
    return padded_data


# extract response around stimulus.
def get_stim_response(
        neural_trial,
        ch,
        trial_idx,
        l_frames,
        r_frames,
        ):
    response = []
    grating = []
    for trials in trial_idx:
        # read trial data.
        fluo = neural_trial[str(trials)][ch]
        stim = neural_trial[str(trials)]['stim'].reshape(1,-1)
        # add zeros to prevent indexing issue.
        fluo = pad_zeros(fluo, l_frames, r_frames)
        stim = pad_zeros(stim, l_frames, r_frames).reshape(-1)
        # compute stimulus start point.
        diff_stim = np.diff(stim, append=0)
        grat_start = np.where(diff_stim == 1)[0] + 1
        for idx in grat_start:
            # find signal response.
            r_ch = fluo[:, idx-l_frames : idx+r_frames]
            r_ch = np.expand_dims(r_ch, axis=0)
            # find stimulus.
            s = stim[idx-l_frames : idx+r_frames]
            s = np.expand_dims(s, axis=0)
            response.append(r_ch)
            grating.append(s)
    # concatenate results.
    response = np.concatenate(response, axis=0)
    grating = np.concatenate(grating, axis=0)
    grating, _ = mode(grating, axis=0)
    return response, grating


# average across grating and neurons.
def get_mean_response(response):
    response = np.mean(response, axis=0)
    response = np.mean(response, axis=0)
    return response


def plot_fig4(
        ops,
        ch = 'fluo_ch2',
        l_frames = 25,
        r_frames = 25,
        ):
    [neural_trial, _] = RetrieveResults.run(ops)
    # set trial id for jitter and fix.
    jitter_idx = [0, 2]
    fix_idx = [1, 3]
    # find grating response.
    response_fix, grating_fix = get_stim_response(
        neural_trial, ch, fix_idx, l_frames, r_frames)
    response_jitter, _ = get_stim_response(
        neural_trial, ch, jitter_idx, l_frames, r_frames)
    # compute average.
    response_fix = get_mean_response(response_fix)
    response_jitter = get_mean_response(response_jitter)
    # time axis.
    t = np.arange(-l_frames, r_frames)
    fig, axs = plt.subplots(1, 1, figsize=(16, 8))
    for k in range(len(grating_fix)-1):
        axs.fill_between(
            [t[k], t[k+1]], 0.02, 0.06,
            color='lightgrey' if grating_fix[k] == 1 else 'white',
            label='fixed grating')
    axs.plot(
        t,
        response_fix,
        color='dodgerblue',
        label='mean traces on fix')
    axs.plot(
        t,
        response_jitter,
        color='coral',
        label='mean traces on jitter')
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('frames since stimulus')
    axs.set_ylabel('response')
    axs.set_xlim([-l_frames, r_frames])
    axs.set_ylim([0.035, 0.05])
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles[-3:], labels[-3:], loc='upper right')
    fig.suptitle('Grating aligned traces comparison for neuron and trial averaqe')
    fig.savefig(os.path.join(ops['save_path0'], 'fig4_align_jitter_fixed.png'), dpi=300)
    plt.close()
    