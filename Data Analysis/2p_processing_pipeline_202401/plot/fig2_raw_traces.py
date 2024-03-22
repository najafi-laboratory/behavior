#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from postprocess import ReadData
from postprocess import LabelExcInh
from postprocess import TracesSpikes


# read and process traces

def read_data(
        ops
        ):
    mask         = ReadData.read_mask(ops)
    raw_traces   = ReadData.read_raw_traces(ops)
    raw_voltages = ReadData.read_raw_voltages(ops)
    label        = LabelExcInh.run(ops, mask)
    fluo = raw_traces['fluo']
    neuropil = raw_traces['neuropil']
    dff, spikes = TracesSpikes.run(ops, fluo, neuropil)
    if raw_voltages is not None:
        vol_img_bin = raw_voltages['vol_img_bin']
        vol_stim_bin = raw_voltages['vol_stim_bin']
        vol_time = raw_voltages['vol_time']
        time_img = get_img_time(vol_time, vol_img_bin)
        return [label, dff, spikes, time_img, vol_stim_bin, vol_time]
    else:
        return [label, dff, spikes, None, None, None]


# find imaging trigger time stamps

def get_img_time(
        vol_time,
        vol_img_bin
        ):
    diff_vol = np.diff(vol_img_bin, append=0)
    idx_up = np.where(diff_vol == 1)[0]+1
    img_time = vol_time[idx_up]
    return img_time


# get subsequence index with given start and end.

def get_sub_time_idx(
        time,
        start,
        end
        ):
    idx = np.where((time >= start) &(time <= end))[0]
    return idx


# main function for plot

def plot_fig2(
        ops,
        max_ms = 180000,
        ):
    try:
        print('plotting fig2 raw traces')

        [label, dff, spikes, time_img, vol_stim_bin, vol_time] = read_data(ops)
        mean_fluo_0 = np.mean(dff[label==0, :], axis=0)
        mean_fluo_1 = np.mean(dff[label==1, :], axis=0)

        # plot figs.
        if np.max(vol_time) < max_ms:
            num_figs = 1
        else:
            num_figs = int(np.max(vol_time)/max_ms)
        num_subplots = dff.shape[0] + 2
        for f in tqdm(range(num_figs)):

            # find sequence start and end timestamps.
            start = f * max_ms
            end   = (f+1) * max_ms

            # get subplot range.
            sub_vol_time_idx = get_sub_time_idx(vol_time, start, end)
            sub_time_img_idx = get_sub_time_idx(time_img, start, end)
            sub_vol_time     = vol_time[sub_vol_time_idx]
            sub_time_img     = time_img[sub_time_img_idx]
            sub_dff          = dff[:, sub_time_img_idx]
            sub_spikes       = spikes[:, sub_time_img_idx]
            sub_mean_fluo_0  = mean_fluo_0[sub_time_img_idx]
            sub_mean_fluo_1  = mean_fluo_1[sub_time_img_idx]
            sub_vol_stim_bin = vol_stim_bin[sub_vol_time_idx]

            # create new figure.
            fig, axs = plt.subplots(num_subplots, 1, figsize=(24, 16))
            plt.subplots_adjust(hspace=0.6)

            # plot mean excitory fluo on functional only.
            axs[0].plot(
                sub_vol_time,
                sub_vol_stim_bin,
                color='grey',
                label='stimulus',
                lw=0.5)
            axs[0].plot(
                sub_time_img, sub_mean_fluo_0,
                color='dodgerblue',
                label='mean of excitory',
                lw=0.5)
            axs[0].set_title(
                'mean trace of {} excitory neurons'.format(
                    np.sum(label==0)))

            # plot mean inhibitory fluo on functional and anatomical channels.
            axs[1].plot(
                sub_vol_time,
                sub_vol_stim_bin,
                color='grey',
                label='stimulus',
                lw=0.5)
            axs[1].plot(
                sub_time_img, sub_mean_fluo_1,
                color='dodgerblue',
                label='mean of inhibitory',
                lw=0.5)
            axs[1].set_title(
                'mean trace of {} inhibitory neurons'.format(
                    np.sum(label==1)))

            # plot individual traces.
            fluo_color = ['seagreen', 'coral']
            fluo_label = ['excitory', 'inhibitory']
            spikes_color = ['dodgerblue', 'violet']
            for i in range(dff.shape[0]):
                
                '''
                spikes_idx = np.nonzero(sub_spikes[i,:])[0]
                spikes_data = sub_spikes[i,spikes_idx]
                spikes_time = sub_time_img[spikes_idx]
                axs[i+2].vlines(
                    x=spikes_time,
                    ymin=0,
                    ymax=spikes_data,
                    color=spikes_color[label[i]],
                    label=fluo_label[label[i]]+'_spikes',
                    alpha=0.5,
                    lw=0.5)
                '''

                axs[i+2].plot(
                    sub_vol_time,
                    sub_vol_stim_bin * np.max(sub_dff[i,:]),
                    color='grey',
                    label='stimulus',
                    lw=0.5)
                axs[i+2].plot(
                    sub_time_img, sub_dff[i,:],
                    color=fluo_color[label[i]],
                    label=fluo_label[label[i]],
                    lw=0.5)

                axs[i+2].set_title('raw trace of neuron # '+ str(i).zfill(3))

            # adjust layout.
            for i in range(num_subplots):
                axs[i].tick_params(axis='y', tick1On=False)
                axs[i].spines['left'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].set_xlabel('time / ms')
                axs[i].set_xlim([start, end])
                axs[i].autoscale(axis='y')
                axs[i].set_xticks(f * max_ms + np.arange(0,max_ms/5000+1) * 5000)
                axs[i].legend(loc='upper left')
            fig.set_size_inches(max_ms/2000, num_subplots*2)
            fig.tight_layout()

            # save figure.
            fig.savefig(os.path.join(
                ops['save_path0'], 'figures',
                'fig2_raw_traces'+str(f).zfill(2)+'.pdf'),
                dpi=300)
            plt.close()

    except:
        print('plotting fig2 failed')
