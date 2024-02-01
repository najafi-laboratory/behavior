#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

from modules import RetrieveResults


# extract response around stimulus.

def get_stim_response(
        neural_trial,
        ch,
        trial_idx,
        l_frames,
        r_frames,
        ):
    neu_seq  = []
    neu_time = []
    stim_vol  = []
    stim_time = []
    vol_stim_bin = neural_trial['raw']['vol_stim_bin']
    #vol_time = neural_trial['raw']['vol_time']
    vol_time = np.arange(0, len(vol_stim_bin))
    for trials in trial_idx:
        # read trial data.
        fluo = neural_trial[str(trials)][ch]
        stim = neural_trial[str(trials)]['stim']
        time = neural_trial[str(trials)]['time']
        # compute stimulus start point.
        grat_start = get_grating_start(stim)
        for idx in grat_start:
            if idx > l_frames and idx < len(time)-r_frames:
                # reshape for concatenate into matrix.
                stim = stim.reshape(1,-1)
                time = time.reshape(1,-1)
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[:, idx-l_frames : idx+r_frames] - time[:, idx]
                neu_time.append(t)
                # voltage recordings.
                vidx = np.where(
                    (vol_time > time[:,idx-l_frames]) &
                    (vol_time < time[:,idx+r_frames]))[0]
                sv = vol_stim_bin[vidx].reshape(1,-1)
                stim_vol.append(sv)
                # voltage time stamps.
                st = vol_time[vidx].reshape(1,-1) - time[:, idx]
                stim_time.append(st)
    # correct voltage recordings to the same length.
    min_len = np.min([sv.shape[1] for sv in stim_vol])
    stim_vol = [sv[:,:min_len] for sv in stim_vol]
    stim_time = [st[:,:min_len] for st in stim_time]
    # concatenate results.
    neu_seq  = np.concatenate(neu_seq, axis=0)
    neu_time = np.concatenate(neu_time, axis=0)
    stim_vol  = np.concatenate(stim_vol, axis=0)
    stim_time = np.concatenate(stim_time, axis=0)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    # get mode stimulus.
    stim_vol, _ = mode(stim_vol, axis=0)
    # scale stimulus sequence.
    neu_max = np.mean(neu_seq) + 3 * np.std(neu_seq)
    neu_min = np.mean(neu_seq) - 3 * np.std(neu_seq)
    stim_vol = stim_vol*(neu_max-neu_min) + neu_min
    return [neu_seq, neu_time, stim_vol, stim_time]


# find grating start.

def get_grating_start(stim):
    diff_stim = np.diff(stim, prepend=0)
    grat_start = np.where(diff_stim == 1)[0]
    return grat_start


# normalize sequence.

def norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)


# main function for plot

def plot_fig4(
        ops,
        l_frames = 30,
        r_frames = 30,
        ):
    print('plotting fig4 grating aligned traces')
    try:
        [neural_trial, _] = RetrieveResults.run(ops)
        ch = 'fluo_ch'+str(ops['functional_chan'])

        # find trial id for jitter and fix.
        fix_idx = np.where(neural_trial['trial_type']==2)[0]
        jitter_idx = np.where(neural_trial['trial_type']==1)[0]

        # fix data
        [fix_neu_seq,  fix_neu_time,
         fix_stim_vol, fix_stim_time] = get_stim_response(
            neural_trial, ch, fix_idx, l_frames, r_frames)
        # jitter data
        [jitter_neu_seq,  jitter_neu_time,
         jitter_stim_vol, jitter_stim_time] = get_stim_response(
            neural_trial, ch, jitter_idx, l_frames, r_frames)

        # plot signals.
        num_subplots = fix_neu_seq.shape[1] + 1
        fig, axs = plt.subplots(num_subplots, 1, figsize=(20, 10))
        plt.subplots_adjust(hspace=1.2)

        # mean response.
        axs[0].plot(
            fix_stim_time,
            norm(fix_stim_vol),
            color='grey',
            label='fix stim')
        axs[0].plot(
            fix_neu_time,
            norm(np.mean(np.mean(fix_neu_seq, axis=0), axis=0)),
            color='springgreen',
            marker='.',
            markersize=5,
            label='fix')
        axs[0].plot(
            jitter_neu_time,
            norm(np.mean(np.mean(jitter_neu_seq, axis=0), axis=0)),
            color='violet',
            marker='.',
            markersize=5,
            label='jitter')
        axs[0].set_title(
            'grating average trace of {} neurons'.format(
            fix_neu_seq.shape[1]))

        # individual neuron response.
        for i in range(fix_neu_seq.shape[1]):
            axs[i+1].plot(
                fix_stim_time,
                norm(fix_stim_vol),
                color='grey',
                label='fix stim')
            axs[i+1].plot(
                fix_neu_time,
                norm(np.mean(fix_neu_seq[:,i,:], axis=0)),
                color='dodgerblue',
                marker='.',
                markersize=5,
                label='fix')
            axs[i+1].plot(
                jitter_neu_time,
                norm(np.mean(jitter_neu_seq[:,i,:], axis=0)),
                color='coral',
                marker='.',
                markersize=5,
                label='jitter')
            axs[i+1].set_title(
                'grating average trace of neuron # '+ str(i).zfill(3))

        # adjust layout.
        for i in range(num_subplots):
            axs[i].tick_params(axis='y', tick1On=False)
            axs[i].spines['left'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)
            axs[i].set_xlabel('time / ms')
            axs[i].set_xlabel('time since center grating start / ms')
            axs[i].set_ylabel('response')
            axs[i].set_xlim([np.min(fix_stim_time), np.max(fix_stim_time)])
            axs[i].set_ylim([0, 1])
            axs[i].set_yticks([])
        handles1, labels1 = axs[0].get_legend_handles_labels()
        handles2, labels2 = axs[-1].get_legend_handles_labels()
        fig.legend(handles1+handles2[1:], labels1+labels2[1:], loc='upper right')
        fig.set_size_inches(8, num_subplots*4)
        fig.tight_layout()

        # save figure
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures', 'fig4_align_grating.pdf'),
            dpi=300)
        plt.close()

    except:
        print('plotting fig4 failed')
