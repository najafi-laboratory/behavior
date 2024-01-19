#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from modules import RetrieveResults


save_folder = 'FN8_PPC_011724'


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


def get_mean_stim_dur(neural_trial):
    stim = []
    for trials in neural_trial.keys():
        stim.append(neural_trial[trials]['stim'])
    stim = np.concatenate(stim)
    diff_stim = np.diff(stim, append=0)
    grat_end = np.where(diff_stim == -1)[0]
    grat_start = np.where(diff_stim == 1)[0]
    dur_list = []
    for i in range(len(grat_end)):
        dur_list.append((grat_end[i] - grat_start[i]))
    stim_dur = int(np.median(dur_list))
    return stim_dur


def plot_fig3(
        ch = 'fluo_ch1',
        num_neuron = 6,
        rows = 4,
        l_frames = 20,
        r_frames = 30,
        amp = 4,
        ):
    [_, neural_trial, _] = RetrieveResults.run(save_folder)
    fix_idx = [0, 2]
    jitter_idx = [1, 3]
    response_fix = get_stim_response(
        neural_trial, ch, fix_idx, l_frames, r_frames)
    response_jitter = get_stim_response(
        neural_trial, ch, jitter_idx, l_frames, r_frames)
    stim_dur = get_mean_stim_dur(neural_trial)
    fig, axs = plt.subplots(rows, num_neuron, figsize=(20, 8))
    plt.subplots_adjust(hspace=0.8)
    plt.subplots_adjust(wspace=0.4)
    for i in range(rows):
        for j in range(num_neuron):
            idx = i*num_neuron + j
            axs[i,j].fill_between(
                [0, stim_dur], 0.1, 0.4,
                color='lightgrey')
            axs[i,j].plot(
                np.arange(-l_frames, r_frames),
                np.mean(response_jitter[:,idx,:], axis=0) * amp,
                color='springgreen',
                label='mean traces on jitter')
            axs[i,j].plot(
                np.arange(-l_frames, r_frames),
                np.mean(response_fix[:,idx,:], axis=0) * amp,
                color='dodgerblue',
                label='mean traces on fix')
    for i in range(rows):
        for j in range(num_neuron):
            axs[i,j].tick_params(tick1On=False)
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].set_xlabel('frames since stimulus')
            axs[i,j].set_ylabel('response')
            axs[i,j].set_xlim([-l_frames, r_frames])
            axs[i,j].set_ylim([0.1, 0.4])
            axs[i,j].set_title('neuron # {}'.format(i*num_neuron + j))
    handles, labels = axs[i,j].get_legend_handles_labels()
    fig.legend(handles[-3:], labels[-3:], loc='upper right')
    fig.suptitle('Grating aligned traces for {} neurons'.format(
        int(rows*num_neuron)))
    fig.tight_layout()
    fig.savefig('./figures/fig3_align_grating.pdf', dpi=300)
    fig.savefig('./figures/fig3_align_grating.png', dpi=300)