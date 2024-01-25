#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from modules import RetrieveResults


save_folder = 'FN8_PPC_121923'


def pad_zeros(data, l_frames, r_frames):
    n, t = data.shape
    padded_data = np.zeros((n, l_frames + t + r_frames))
    padded_data[:, l_frames:l_frames + t] = data
    return padded_data

    
def get_stim_response(
        neural_trial,
        ch,
        trial_idx,
        l_frames,
        r_frames,
        ):
    response = []
    for trials in trial_idx:
        fluo = neural_trial[str(trials)][ch]
        stim = neural_trial[str(trials)]['stim'].reshape(1,-1)
        fluo = pad_zeros(fluo, l_frames, r_frames)
        stim = pad_zeros(stim, l_frames, r_frames).reshape(-1)
        diff_stim = np.diff(stim, append=0)
        grat_start = np.where(diff_stim == 1)[0] + 1
        for idx in grat_start:
            r_ch = fluo[:, idx-l_frames : idx+r_frames]
            r_ch = np.expand_dims(r_ch, axis=0)
            response.append(r_ch)
    response = np.concatenate(response, axis=0)
    return response


def get_mean_response(response):
    response = np.mean(response, axis=0)
    response = np.mean(response, axis=0)
    return response


def get_mean_stim_dur(neural_trial):
    stim = []
    for trials in range(len(neural_trial)-1):
        stim.append(neural_trial[str(trials)]['stim'])
    stim = np.concatenate(stim)
    diff_stim = np.diff(stim, append=0)
    grat_end = np.where(diff_stim == -1)[0]
    grat_start = np.where(diff_stim == 1)[0]
    dur_list = []
    for i in range(len(grat_end)):
        dur_list.append((grat_end[i] - grat_start[i]))
    stim_dur = int(np.median(dur_list))
    return stim_dur


def plot_fig5(
        ch = 'fluo_ch2',
        l_frames = 20,
        r_frames = 25,
        ):
    [_, neural_trial, _] = RetrieveResults.run(save_folder)
    jitter_idx = [0, 2]
    fix_idx = [1, 3]
    response_fix = get_stim_response(
        neural_trial, ch, fix_idx, l_frames, r_frames)
    response_jitter = get_stim_response(
        neural_trial, ch, jitter_idx, l_frames, r_frames)
    response_fix = get_mean_response(response_fix)
    response_jitter = get_mean_response(response_jitter)
    stim_dur = get_mean_stim_dur(neural_trial)
    fig, axs = plt.subplots(1, 1, figsize=(16, 8))
    axs.fill_between(
        [0, stim_dur], 0.03, 0.08,
        color='lightgrey',
        label='grating')
    axs.plot(
        np.arange(-l_frames, r_frames),
        response_fix,
        color='coral',
        label='mean traces on fix')
    axs.plot(
        np.arange(-l_frames, r_frames),
        response_jitter,
        color='dodgerblue',
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
    fig.tight_layout()
    fig.savefig('./figures/fig5_align_jitter_fixed.pdf', dpi=300)
    fig.savefig('./figures/fig5_align_jitter_fixed.png', dpi=300)
    
    
plot_fig5()