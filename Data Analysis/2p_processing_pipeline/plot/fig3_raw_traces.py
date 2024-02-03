#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

from modules import RetrieveResults


# read and process traces

def read_data(
        ops
        ):
    [_, raw_traces, raw_voltages, _] = RetrieveResults.run(ops)
    ch = 'fluo_ch' + str(ops['functional_chan'])
    fluo = raw_traces[ch]
    vol_img_bin = raw_voltages['vol_img_bin']
    vol_stim_bin = raw_voltages['vol_stim_bin']
    vol_time = raw_voltages['vol_time']
    time_img = get_img_time(vol_time, vol_img_bin)
    return [fluo, time_img, vol_stim_bin, vol_time]


# find imaging trigger time stamps

def get_img_time(
        time_vol,
        vol_img_bin
        ):
    diff_vol = np.diff(vol_img_bin, append=0)
    idx_up = np.where(diff_vol == 1)[0]+1
    img_time = time_vol[idx_up]
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

def plot_fig3(
        ops,
        max_ms = 180000,
        ):
    try:
        print('plotting fig3 raw traces')

        [fluo, time_img, vol_stim_bin, time_vol] = read_data(ops)
        fluo = (fluo - np.min(fluo)) / (np.max(fluo) - np.min(fluo) + 1e-8)
        mean_fluo = np.mean(fluo, axis=0)

        # plot figs.
        if len(time_vol) < max_ms:
            num_figs = 1
        else:
            num_figs = int(len(time_vol)/max_ms)
        num_subplots = fluo.shape[0] + 1
        for f in range(num_figs):

            # find sequence start and end timestamps.
            start = f * max_ms
            end   = (f+1) * max_ms

            # get subplot range.
            sub_time_vol_idx = get_sub_time_idx(time_vol, start, end)
            sub_time_img_idx = get_sub_time_idx(time_img, start, end)
            sub_time_vol     = time_vol[sub_time_vol_idx]
            sub_time_img     = time_img[sub_time_img_idx]
            sub_vol_stim_bin = vol_stim_bin[sub_time_vol_idx]
            sub_fluo         = fluo[:, sub_time_img_idx]
            sub_mean_fluo    = mean_fluo[sub_time_img_idx]

            # create new figure.
            fig, axs = plt.subplots(num_subplots, 1, figsize=(24, 16))
            plt.subplots_adjust(hspace=0.6)

            # plot stimulus.
            for i in range(num_subplots):
                axs[i].plot(
                    sub_time_vol, sub_vol_stim_bin,
                    color='dodgerblue', lw=0.5)

            # plot mean fluo.
            axs[0].plot(
                sub_time_img, sub_mean_fluo,
                color='coral', lw=0.5)
            axs[0].set_title('mean trace of {} neurons'.format(fluo.shape[0]))

            # plot traces.
            for i in range(fluo.shape[0]):
                axs[i+1].plot(
                    sub_time_img, sub_fluo[i,:],
                    color='black', lw=0.5)
                axs[i+1].set_title('raw trace of neuron # '+ str(i).zfill(3))

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
                axs[i].set_yticks([])
            fig.set_size_inches(max_ms/2000, num_subplots*2)
            fig.tight_layout()

            # save figure.
            fig.savefig(os.path.join(
                ops['save_path0'], 'figures',
                'fig3_raw_traces'+str(f).zfill(2)+'.pdf'),
                dpi=300)
            plt.close()

    except:
        print('plotting fig3 failed')
