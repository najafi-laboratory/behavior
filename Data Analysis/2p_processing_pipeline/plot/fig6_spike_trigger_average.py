#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

from modules import RetrieveResults


# normalize data into [0,1]

def norm01(data):
    data = (data - np.min(data)) / ( np.max(data) - np.min(data) )
    return data


# extract fluo around spikes.

def get_stim_response(
        fluo,
        spikes,
        l_frames,
        r_frames,
        ):
    all_tri_fluo = []
    # check neuron one by one.
    for neu_idx in range(fluo.shape[0]):
        neu_tri_fluo = []
        neu_spike_time = np.where(spikes[neu_idx,:] != 0)[0]
        # find spikes.
        if len(neu_spike_time) > 0:
            for neu_t in neu_spike_time:
                if (neu_t - l_frames > 0 and 
                    neu_t + r_frames < fluo.shape[1]):
                    f = fluo[neu_idx,neu_t - l_frames:neu_t + r_frames]
                    f = f.reshape(1, -1)
                    f = f / spikes[neu_idx,neu_t]
                    neu_tri_fluo.append(f)
                else:
                    f = np.empty((1, l_frames+r_frames))
                    neu_tri_fluo.append(f)
            neu_tri_fluo = np.concatenate(neu_tri_fluo, axis=0)
        else:
            neu_tri_fluo = np.zeros((1, l_frames+r_frames))
        all_tri_fluo.append(neu_tri_fluo)
    return all_tri_fluo
    


# main function for plot

def plot_fig6(
        ops,
        l_frames = 20,
        r_frames = 40,
        ):

    try:
        print('plotting fig6 spike trigger fluorescence average')

        [_, raw_traces, _, _] = RetrieveResults.run(ops)
        fluo = raw_traces['fluo_ch'+str(ops['functional_chan'])]
        spikes = raw_traces['spikes_ch'+str(ops['functional_chan'])]
        
        # threshold spikes.
        spikes[norm01(spikes) < ops['spike_thres']] = 0
        
        # get response.
        all_tri_fluo = get_stim_response(
            fluo, spikes, l_frames, r_frames)
        frame = np.arange(-l_frames, r_frames)

        # plot signals.
        num_subplots = fluo.shape[0] + 1
        fig, axs = plt.subplots(num_subplots, 1, figsize=(20, 10))
        plt.subplots_adjust(hspace=0.2)
        
        # individual neuron response.
        for i in range(fluo.shape[0]):
            # individual triggered traces.
            if np.sum(all_tri_fluo[i]) != 0:
                for j in range(all_tri_fluo[i].shape[0]):
                    axs[i+1].plot(
                        frame,
                        all_tri_fluo[i][j,:],
                        color='grey',
                        label='triggered trace',
                        alpha=0.1)
                # aveerage for one neuron.
                axs[i+1].plot(
                    frame,
                    np.mean(all_tri_fluo[i], axis=0),
                    color='dodgerblue',
                    label='mean trace')
                axs[i+1].set_title(
                    'mean spike trigger average of neuron # '+ str(i).zfill(3))

        # mean response.
        tri_fluo = np.concatenate(all_tri_fluo, axis=0)
        tri_fluo = tri_fluo[
            np.where(np.sum(tri_fluo, axis=1)!=0)[0], :]
        for i in range(tri_fluo.shape[0]):
            axs[0].plot(
                frame,
                tri_fluo[i,:],
                color='grey',
                label='triggered trace',
                alpha=0.01)
        axs[0].plot(
            frame,
            np.mean(tri_fluo, axis=0),
            color='coral',
            label='mean trace')
        axs[0].set_title(
            'mean spike trigger average across {} neurons'.format(
                fluo.shape[0]))

        # adjust layout.
        for i in range(num_subplots):
            axs[i].tick_params(axis='y', tick1On=False)
            axs[i].spines['left'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)
            axs[i].set_xlabel('frame')
            axs[i].set_ylabel('response')
        fig.set_size_inches(6, num_subplots*4)
        fig.tight_layout()

        # save figure
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures', 'fig6_spike_trigger_average.pdf'),
            dpi=300)
        plt.close()

    except:
        print('plotting fig6 failed')
