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


def get_omi_response(
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
        time = neural_trial[str(trials)]['time'].reshape(1,-1)
        fluo = pad_zeros(fluo, l_frames, r_frames)
        stim = pad_zeros(stim, l_frames, r_frames).reshape(-1)
        time = pad_zeros(time, l_frames, r_frames).reshape(-1)
        omi_idx = find_omission_idx(stim, time)
        for idx in omi_idx:
            r_ch = fluo[:, idx-l_frames : idx+r_frames]
            r_ch = np.expand_dims(r_ch, axis=0)
            response.append(r_ch)
    response = np.concatenate(response, axis=0)
    return response


def find_omission_idx(stim, time):
    diff_stim = np.diff(stim, append=0)
    grat_end = np.where(diff_stim == -1)[0][:-1]
    grat_start = np.where(diff_stim == 1)[0][1:]
    isi_list = []
    for i in range(len(grat_end)):
        if i<len(grat_start):
            isi = (time[grat_start[i]] - time[grat_end[i]])
            isi_list.append(isi)
    isi_list = np.array(isi_list)
    idx = np.where(isi_list > 1000)[0]
    omi_frames = int(500/np.median(np.diff(time, append=0)))
    omi_idx = grat_end[idx] + omi_frames
    return omi_idx


def plot_fig4(
        ch = 'fluo_ch1',
        num_neuron = 6,
        rows = 4,
        l_frames = 30,
        r_frames = 50,
        ):
    [_, neural_trial, _] = RetrieveResults.run(save_folder)
    jitter_idx = [0, 2]
    fix_idx = [1, 3]
    response_fix = get_omi_response(
        neural_trial, ch, fix_idx, l_frames, r_frames)
    response_jitter = get_omi_response(
        neural_trial, ch, jitter_idx, l_frames, r_frames)
    fig, axs = plt.subplots(rows, num_neuron, figsize=(24, 10))
    plt.subplots_adjust(hspace=0.8)
    plt.subplots_adjust(wspace=0.4)
    for i in range(rows):
        for j in range(num_neuron):
            idx = i*num_neuron + j
            mean_jitter = np.mean(response_jitter[:,idx,:], axis=0)
            mean_fix = np.mean(response_fix[:,idx,:], axis=0)
            axs[i,j].axvline(
                0,
                color='grey',
                lw=2,
                label='omission',
                linestyle='--')
            axs[i,j].plot(
                np.arange(-l_frames, r_frames),
                mean_jitter,
                color='coral',
                label='mean traces on jitter')
            axs[i,j].plot(
                np.arange(-l_frames, r_frames),
                mean_fix,
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
            axs[i,j].set_ylim([0.02, 0.12])
            axs[i,j].set_title('neuron # {}'.format(i*num_neuron + j))
    handles, labels = axs[i,j].get_legend_handles_labels()
    fig.legend(handles[-3:], labels[-3:], loc='upper right')
    fig.suptitle('Omission aligned traces for {} neurons for '.format(
        int(rows*num_neuron)) + ch)
    fig.tight_layout()
    fig.savefig('./figures/fig4_align_omission.pdf', dpi=300)
    fig.savefig('./figures/fig4_align_omission.png', dpi=300)
    
    
plot_fig4()