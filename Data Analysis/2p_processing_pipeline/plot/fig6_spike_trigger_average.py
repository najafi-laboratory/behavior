#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

from modules import PostProcess


# normalize data into [0,1].

def norm01(data):
    data = (data - np.min(data)) / ( np.max(data) - np.min(data) )
    return data


#%% plot roi


# find a patch contains roi.

def get_roi_img(
        ops,
        mask,
        size = 128
        ):
    func_masks = mask['ch'+str(ops['functional_chan'])]
    func_img = []
    roi_mask = []
    for i in np.unique(func_masks)[1:]:
        rows, cols = np.where(mask == i)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        center_row = (min_row + max_row) // 2
        center_col = (min_col + max_col) // 2
        start_row = max(0, min(mask.shape[0] - size, center_row - size/2))
        start_col = max(0, min(mask.shape[1] - size, center_col - size/2))
        func_img.append(mask['ch'+str(ops['functional_chan'])]['input_img']
                        [start_row:start_row + size, start_col:start_col + size])
        roi_mask.append(mask['ch'+str(ops['functional_chan'])]['masks']
                        [start_row:start_row + size, start_col:start_col + size])
    return func_img, roi_mask


# automatical adjustment of contrast.

def adjust_contrast(org_img, lower_percentile=75, upper_percentile=99):
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    img = np.clip((org_img - lower) * 255 / (upper - lower), 0, 255)
    img = img.astype(np.uint8)
    return img


# main function

def plot_roi(
        axs,
        masks,
        col = 0,
        ):
    0




#%% plot average fluorescence signal


# extract fluo around spikes.

def get_stim_response(
        fluo,
        spikes,
        l_frames,
        r_frames,
        num_spikes = 1,
        ):
    all_tri_fluo = []

    # check neuron one by one.
    for neu_idx in range(fluo.shape[0]):
        neu_tri_fluo = []
        # find the largest num_spikes spikes.
        neu_spike_time = np.nonzero(spikes[neu_idx,:])[0]
        neu_spike_amp = spikes[neu_idx,neu_spike_time]
        if len(neu_spike_amp) > num_spikes:
            neu_spike_time[np.argsort(-neu_spike_amp)[:num_spikes]]
        # find spikes.
        if len(neu_spike_time) > 0:
            for neu_t in neu_spike_time:
                if (neu_t - l_frames > 0 and
                    neu_t + r_frames < fluo.shape[1]):
                    f = fluo[neu_idx,neu_t - l_frames:neu_t + r_frames]
                    f = f.reshape(1, -1)
                    f = f / (spikes[neu_idx,neu_t]+1e-5)
                    neu_tri_fluo.append(f)
            if len(neu_tri_fluo) > 0:
                neu_tri_fluo = np.concatenate(neu_tri_fluo, axis=0)
            else:
                neu_tri_fluo = np.zeros((1, l_frames+r_frames))
        else:
            neu_tri_fluo = np.zeros((1, l_frames+r_frames))
        all_tri_fluo.append(neu_tri_fluo)
    return all_tri_fluo


# main function

def plot_fluo(
        axs,
        ops, fluo, spikes, label,
        l_frames, r_frames,
        col = 1,
        ):

    # get response.

    all_tri_fluo = get_stim_response(
        fluo, spikes, l_frames, r_frames)
    frame = np.arange(-l_frames, r_frames)

    # individual neuron response.
    color = ['seagreen', 'coral']
    for i in range(fluo.shape[0]):
        # individual triggered traces.
        if np.sum(all_tri_fluo[i]) != 0:
            fluo_mean = np.mean(all_tri_fluo[i], axis=0)
            fluo_sem = sem(all_tri_fluo[i], axis=0)
            # aveerage for one neuron.
            axs[i+1,col].plot(
                frame,
                fluo_mean,
                color=color[label[i]],
                label='mean trace')
            axs[i+1,col].fill_between(
                frame,
                fluo_mean - fluo_sem,
                fluo_mean + fluo_sem,
                color=color[label[i]],
                alpha=0.2)
            axs[i+1,col].set_title(
                'mean spike trigger average of neuron # '+ str(i).zfill(3))

    # mean response.
    tri_fluo = np.concatenate(all_tri_fluo, axis=0)
    tri_fluo = tri_fluo[
        np.where(np.sum(tri_fluo, axis=1)!=0)[0], :]
    fluo_mean = np.mean(tri_fluo, axis=0)
    fluo_sem = sem(tri_fluo, axis=0)
    axs[0,col].plot(
        frame,
        fluo_mean,
        color='coral',
        label='mean trace')
    axs[0,col].fill_between(
        frame,
        fluo_mean - fluo_sem,
        fluo_mean + fluo_sem,
        color='coral',
        alpha=0.2)
    axs[0,col].set_title(
        'mean spike trigger average across {} neurons'.format(
            fluo.shape[0]))

    # adjust layout.
    for i in range(axs.shape[0]):
        axs[i,col].tick_params(axis='y', tick1On=False)
        axs[i,col].spines['left'].set_visible(False)
        axs[i,col].spines['right'].set_visible(False)
        axs[i,col].spines['top'].set_visible(False)
        axs[i,col].set_xlabel('frame')
        axs[i,col].set_ylabel('response')


#%% main

def plot_fig6(
        ops,
        l_frames = 10,
        r_frames = 30,
        ):

    try:
        print('plotting fig6 spike trigger fluorescence average')

        [mask, raw_traces, _, _] = PostProcess.run(ops)
        fluo = raw_traces['fluo_ch'+str(ops['functional_chan'])]
        spikes = raw_traces['spikes_ch'+str(ops['functional_chan'])]
        label = mask['label']

        # create canvas.
        num_subplots = fluo.shape[0] + 1
        fig, axs = plt.subplots(num_subplots, 3, figsize=(20, 10))
        plt.subplots_adjust(hspace=0.2)

        # plot average fluorescence signal.
        plot_fluo(
            axs,
            ops, fluo, spikes, label,
            l_frames, r_frames)

        # adjust layout.
        fig.set_size_inches(10, num_subplots*3)
        fig.tight_layout()

        # save figure
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures', 'fig6_spike_trigger_average.pdf'),
            dpi=300)
        plt.close()

    except:
        print('plotting fig6 failed')

