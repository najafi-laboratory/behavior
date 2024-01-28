#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

from modules import RetrieveResults


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


def plot_omi(
        ops,
        neural_trial,
        ch = 'fluo_ch1',
        cols = 5,
        rows = 4,
        l_frames = 30,
        r_frames = 50,
        ):
    jitter_idx = np.array([0, 2], dtype='int32')
    fix_idx = np.array([1, 3], dtype='int32')
    response_fix = get_omi_response(
        neural_trial, ch, fix_idx, l_frames, r_frames)
    response_jitter = get_omi_response(
        neural_trial, ch, jitter_idx, l_frames, r_frames)
    num_figs = int(response_fix.shape[1] / (cols * rows))
    fig, axs = plt.subplots(rows, cols, figsize=(24, 10))
    plt.subplots_adjust(hspace=0.8)
    plt.subplots_adjust(wspace=0.4)
    for i in range(rows):
        for j in range(cols):
            idx = i*cols + j
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
        for j in range(cols):
            axs[i,j].tick_params(tick1On=False)
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].set_xlabel('frames since stimulus')
            axs[i,j].set_ylabel('response')
            axs[i,j].set_xlim([-l_frames, r_frames])
            axs[i,j].set_ylim([0.02, 0.12])
            axs[i,j].set_title('neuron # {}'.format(i*cols + j))
    handles, labels = axs[i,j].get_legend_handles_labels()
    fig.legend(handles[-3:], labels[-3:], loc='upper right')
    fig.suptitle('Omission aligned traces for {} neurons for '.format(
        int(rows*cols)) + ch)
    fig.tight_layout()
    fig.savefig(os.path.join(ops['save_path0'], 'fig5_align_task.png'), dpi=300)
    plt.close()


def get_prepost_response(
        neural_trial,
        ch,
        fix_idx,
        ):
    fluo = [np.expand_dims(neural_trial[str(trials)][ch], axis=0)
            for trials in fix_idx]
    time = [neural_trial[str(trials)]['time']
            for trials in fix_idx]
    stim = [neural_trial[str(trials)]['stim'].reshape(1,-1)
            for trials in fix_idx]
    min_len = np.min([len(t) for t in time])
    fluo = [f[:,:,:min_len] for f in fluo]
    time = [t[:min_len] for t in time]
    stim = [s[:,:min_len] for s in stim]
    fluo = np.concatenate(fluo, axis=0)
    stim = np.concatenate(stim, axis=0)
    stim, _ = mode(stim, axis=0)
    return fluo, stim


def plot_prepost(
        ops,
        neural_trial,
        ch = 'fluo_ch1',
        cols = 3,
        rows = 5,
        ):
    #fix_idx = np.concatenate((np.arange(0,60),np.arange(120,180)))
    #jitter_idx = np.concatenate((np.arange(60,120),np.arange(180,240)))
    jitter_idx = np.concatenate((np.arange(0,60),np.arange(120,180)))
    fix_idx = np.concatenate((np.arange(60,120),np.arange(180,240)))
    fluo, stim = get_prepost_response(neural_trial, ch, fix_idx)
    mean_fluo = np.mean(fluo, axis=0)
    t = np.arange(0, len(stim))
    fig, axs = plt.subplots(rows, cols, figsize=(24, 10))
    plt.subplots_adjust(hspace=0.8)
    plt.subplots_adjust(wspace=0.4)
    for i in range(rows):
        for j in range(cols):
            idx = i*cols + j
            axs[i,j].plot(
                t,
                mean_fluo[idx, :],
                color='coral',
                label='pre-post fixed trial average trace')
            for k in range(len(stim)-1):
                axs[i,j].fill_between(
                    [t[k], t[k+1]], 0.0, 0.1,
                    color='lightgrey' if stim[k] == 1 else 'white',
                    label='fixed grating')
    for i in range(rows):
        for j in range(cols):
            axs[i,j].tick_params(tick1On=False)
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
            axs[i,j].set_xlabel('frames since stimulus')
            axs[i,j].set_ylabel('response')
            axs[i,j].set_title('neuron # {}'.format(i*cols + j))
    handles, labels = axs[i,j].get_legend_handles_labels()
    fig.legend(handles[-2:], labels[-2:], loc='upper right')
    fig.suptitle('Pre-pose trial average traces for {} neurons for '.format(
        int(rows*cols)) + ch)
    fig.tight_layout()
    fig.savefig(os.path.join(ops['save_path0'], 'fig5_align_task.png'), dpi=300)
    plt.close()
    
    
def plot_fig5(
        ops,
        ):
    [neural_trial, _] = RetrieveResults.run(ops)
    if len(neural_trial) <= 5:
        plot_omi(ops, neural_trial)
    else:
        plot_prepost(ops, neural_trial)
    
    
    