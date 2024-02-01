#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

from modules import RetrieveResults


#%% utils


# compute duration of sequence.

def frame_dur(stim, time):
    diff_stim = np.diff(stim, prepend=0)
    idx_up   = np.where(diff_stim == 1)[0]
    idx_down = np.where(diff_stim == -1)[0]
    dur_high = time[idx_down] - time[idx_up]
    dur_low  = time[idx_up[1:]] - time[idx_down[:-1]]
    return [idx_up, idx_down, dur_high, dur_low]


# normalize sequence.

def norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)


#%% omission


# compute omission start point.

def get_omi_idx(stim, time):
    [_, grat_end, _, isi] = frame_dur(stim, time)
    idx = np.where(isi > 1000)[0]
    omi_frames = int(500/np.median(np.diff(time, append=0)))
    omi_start = grat_end[idx] + omi_frames
    return omi_start


# extract response around omissions.

def get_omi_response(
        neural_trial,
        ch,
        trial_idx,
        l_frames,
        r_frames,
        ):
    # read data.
    vol_stim_bin = neural_trial['raw']['vol_stim_bin']
    #vol_time = neural_trial['raw']['vol_time']
    vol_time = np.arange(0, len(vol_stim_bin))
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_vol  = []
    stim_time = []
    # loop over trials.
    for trials in trial_idx:
        # read trial data.
        fluo = neural_trial[str(trials)][ch]
        stim = neural_trial[str(trials)]['stim']
        time = neural_trial[str(trials)]['time']
        # compute stimulus start point.
        omi_start = get_omi_idx(stim, time)
        for idx in omi_start:
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


# main function for plot.

def plot_omi(
        ops,
        l_frames = 30,
        r_frames = 50,
        ):
    print('plotting fig5 omission aligned response')
    [neural_trial, _] = RetrieveResults.run(ops)
    ch = 'fluo_ch'+str(ops['functional_chan'])

    # find trial id for jitter and fix.
    fix_idx = np.where(neural_trial['trial_type']==2)[0]
    jitter_idx = np.where(neural_trial['trial_type']==1)[0]

    # fix data.
    [fix_neu_seq,  fix_neu_time,
     fix_stim_vol, fix_stim_time] = get_omi_response(
        neural_trial, ch, fix_idx, l_frames, r_frames)
    # jitter data.
    [jitter_neu_seq,  jitter_neu_time,
     jitter_stim_vol, jitter_stim_time] = get_omi_response(
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
    axs[0].axvline(
        0,
        color='red',
        lw=2,
        label='omission',
        linestyle='--')
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
        axs[i+1].axvline(
            0,
            color='red',
            lw=2,
            label='omission',
            linestyle='--')
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
        axs[i].set_xlabel('time since omission / ms')
        axs[i].set_ylabel('response')
        axs[i].set_xlim([np.min(fix_stim_time), np.max(fix_stim_time)])
        axs[i].set_ylim([0, 1])
        axs[i].set_yticks([])
    handles1, labels1 = axs[0].get_legend_handles_labels()
    handles2, labels2 = axs[-1].get_legend_handles_labels()
    fig.legend(handles1+handles2[1:], labels1+labels2[1:], loc='upper right')
    fig.set_size_inches(8, num_subplots*4)
    fig.tight_layout()

    # save figure.
    fig.savefig(os.path.join(
        ops['save_path0'], 'figures', 'fig5_align_omission.pdf'),
        dpi=300)
    plt.close()



#%% prepost


# find pre perturbation repeatition.

def get_post_num_grat(
        neural_trial,
        fix_idx,
        ):
    time = neural_trial[str(fix_idx[0])]['time']
    stim = neural_trial[str(fix_idx[0])]['stim']
    [_, _, _, dur_low] = frame_dur(stim, time)
    num_down = np.where(dur_low>np.mean(dur_low))[0][0]
    return num_down


# cut sequence into the same length as the shortest one given pivots.

def trim_seq(
        data,
        pivots,
        ):
    if len(data[0].shape) == 1:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i])-pivots[i] for i in range(len(data))])
        data = [data[i][pivots[i]-len_l_min:pivots[i]+len_r_min]
                for i in range(len(data))]
    if len(data[0].shape) == 3:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i][0,0,:])-pivots[i] for i in range(len(data))])
        data = [data[i][:, :, pivots[i]-len_l_min:pivots[i]+len_r_min]
                for i in range(len(data))]
    return data


# extract response around perturbation.

def get_prepost_response(
        neural_trial,
        ch,
        trial_idx,
        num_down,
        ):
    # read data.
    vol_stim_bin = neural_trial['raw']['vol_stim_bin']
    #vol_time = neural_trial['raw']['vol_time']
    vol_time = np.arange(0, len(vol_stim_bin))
    neu_seq = [np.expand_dims(neural_trial[str(trials)][ch], axis=0)
            for trials in trial_idx]
    neu_time = [neural_trial[str(trials)]['time']
            for trials in trial_idx]
    stim = [neural_trial[str(trials)]['stim']
            for trials in trial_idx]
    # initialize list.
    stim_vol  = []
    stim_time = []
    # find perturbation point.
    post_start_idx = []
    for s,t in zip(stim, neu_time):
        [_, idx_down, _, _] = frame_dur(s, t)
        post_start_idx.append(idx_down[num_down])
    # trim sequences.
    neu_seq = trim_seq(neu_seq, post_start_idx)
    neu_time = trim_seq(neu_time, post_start_idx)
    # find voltage recordings.
    stim_vol  = []
    stim_time = []
    for t in neu_time:
        # voltage recordings.
        vidx = np.where(
            (vol_time > np.min(t)) &
            (vol_time < np.max(t)))[0]
        sv = vol_stim_bin[vidx].reshape(1,-1)
        stim_vol.append(sv)
        # voltage time stamps.
        st = vol_time[vidx].reshape(1,-1)
        stim_time.append(st)
    # correct voltage recordings to the same length.
    min_len = np.min([sv.shape[1] for sv in stim_vol])
    stim_vol = [sv[:,:min_len].reshape(1,-1) for sv in stim_vol]
    stim_time = [st[:,:min_len].reshape(1,-1) for st in stim_time]
    stim_time  = [stim_time[i] - neu_time[i][post_start_idx[i]]
                 for i in range(len(stim_time))]
    # correct neuron time stamps centering at perturbation.
    neu_time  = [neu_time[i] - neu_time[i][post_start_idx[i]]
                 for i in range(len(neu_time))]
    neu_time  = [t.reshape(1,-1) for t in neu_time]
    # concatenate results.
    neu_seq   = np.concatenate(neu_seq, axis=0)
    neu_time  = np.concatenate(neu_time, axis=0)
    stim_vol  = np.concatenate(stim_vol, axis=0)
    stim_time = np.concatenate(stim_time, axis=0)
    # get mean time stamps.
    neu_time = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    # get mode stimulus.
    stim_vol, _ = mode(stim_vol, axis=0)
    # scale stimulus sequence.
    neu_max = np.mean(neu_seq) + 3 * np.std(neu_seq)
    neu_min = np.mean(neu_seq) - 3 * np.std(neu_seq)
    stim_vol = stim_vol*(neu_max-neu_min) + neu_min
    return [neu_seq, neu_time, stim_vol, stim_time]


# main function for plot.

def plot_prepost(
        ops,
        ):
    print('plotting fig5 prepost aligned response')
    [neural_trial, _] = RetrieveResults.run(ops)
    ch = 'fluo_ch'+str(ops['functional_chan'])

    # find trial id for jitter and fix.
    fix_idx = np.where(neural_trial['trial_type']==2)[0]
    jitter_idx = np.where(neural_trial['trial_type']==1)[0]

    # find pre perturbation repeatition.
    num_down = get_post_num_grat(neural_trial, fix_idx)

    # fix data.
    [fix_neu_seq,  fix_neu_time,
     fix_stim_vol, fix_stim_time] = get_prepost_response(
         neural_trial, ch, fix_idx, num_down)
    # jitter data.
    [jitter_neu_seq,  jitter_neu_time,
     jitter_stim_vol, jitter_stim_time] = get_prepost_response(
         neural_trial, ch, jitter_idx, num_down)

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
    axs[0].axvline(
        0,
        color='red',
        lw=2,
        label='perturbation',
        linestyle='--')
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
        'perturbation average trace of {} neurons'.format(
        fix_neu_seq.shape[1]))

    # individual neuron response.
    for i in range(fix_neu_seq.shape[1]):
        axs[i+1].plot(
            fix_stim_time,
            norm(fix_stim_vol),
            color='grey',
            label='fix stim')
        axs[i+1].axvline(
            0,
            color='red',
            lw=2,
            label='perturbation',
            linestyle='--')
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
            'perturbation average trace of neuron # '+ str(i).zfill(3))

    # adjust layout.
    for i in range(num_subplots):
        axs[i].tick_params(axis='y', tick1On=False)
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].set_xlabel('time / ms')
        axs[i].set_xlabel('time since perturbation / ms')
        axs[i].set_ylabel('response')
        axs[i].set_xlim([np.min(fix_stim_time), np.max(fix_stim_time)])
        axs[i].set_ylim([0, 1])
        axs[i].set_yticks([])
    handles1, labels1 = axs[0].get_legend_handles_labels()
    handles2, labels2 = axs[-1].get_legend_handles_labels()
    fig.legend(handles1+handles2[2:], labels1+labels2[2:], loc='upper right')
    fig.set_size_inches(16, num_subplots*2)
    fig.tight_layout()

    # save figure.
    fig.savefig(os.path.join(
        ops['save_path0'], 'figures', 'fig5_align_prepost.pdf'),
        dpi=300)
    plt.close()


#%% main


def plot_fig5(
        ops,
        ):
    try:
        [neural_trial, _] = RetrieveResults.run(ops)
        if len(neural_trial['trial_type']) <= 5:
            plot_omi(ops)
        else:
            plot_prepost(ops)
    except:
        print('plotting fig5 failed')

