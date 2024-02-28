#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.stats import sem

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


#%% omission


# compute omission start point.

def get_omi_idx(stim, time):
    [grat_start, grat_end, _, isi] = frame_dur(stim, time)
    idx = np.where(isi >= 950)[0]
    omi_start = grat_end[idx] + int(500/np.median(np.diff(time, prepend=0))) + 1
    #omi_start = grat_start[idx+1]
    return omi_start


# extract response around omissions.

def get_omi_response(
        raw_voltages,
        neural_trials,
        ch,
        trial_idx,
        l_frames,
        r_frames,
        ):
    # read data.
    vol_stim_bin = raw_voltages['vol_stim_bin']
    vol_time = raw_voltages['vol_time']
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_vol  = []
    stim_time = []
    # loop over trials.
    for trials in trial_idx:
        # read trial data.
        fluo = neural_trials[str(trials)][ch]
        stim = neural_trials[str(trials)]['stim']
        time = neural_trials[str(trials)]['time']
        # compute stimulus start point.
        omi_start = get_omi_idx(stim, time)
        for idx in omi_start:
            if idx > l_frames and idx < len(time)-r_frames:
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # voltage recordings.
                vidx = np.where(
                    (vol_time > time[idx-l_frames]) &
                    (vol_time < time[idx+r_frames]))[0]
                stim_vol.append(vol_stim_bin[vidx])
                # voltage time stamps.
                stim_time.append(vol_time[vidx] - time[idx])
    # correct voltage recordings centering at perturbation.
    stim_time_zero = [np.argmin(np.abs(st)) for st in stim_time]
    stim_time = trim_seq(stim_time, stim_time_zero)
    stim_vol = trim_seq(stim_vol, stim_time_zero)
    # correct neuron time stamps centering at perturbation.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # concatenate results.
    neu_time  = [nt.reshape(1,-1) for nt in neu_time]
    stim_time = [st.reshape(1,-1) for st in stim_time]
    stim_vol  = [sv.reshape(1,-1) for sv in stim_vol]
    neu_seq   = np.concatenate(neu_seq, axis=0)
    neu_time  = np.concatenate(neu_time, axis=0)
    stim_vol  = np.concatenate(stim_vol, axis=0)
    stim_time = np.concatenate(stim_time, axis=0)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    # get mode stimulus.
    stim_vol, _ = mode(stim_vol, axis=0)
    return [neu_seq, neu_time, stim_vol, stim_time]


# main function for plot.

def plot_omi(
        ops,
        l_frames = 80,
        r_frames = 150,
        ):
    print('plotting fig5 omission aligned response')
    [mask, raw_traces, raw_voltages, neural_trials] = RetrieveResults.run(ops)
    ch = 'fluo_ch'+str(ops['functional_chan'])
    label = mask['label']

    # find trial id for jitter and fix.
    fix_idx = np.where(neural_trials['trial_type']==2)[0]
    jitter_idx = np.where(neural_trials['trial_type']==1)[0]

    # fix data.
    [fix_neu_seq,  fix_neu_time,
     fix_stim_vol, fix_stim_time] = get_omi_response(
        raw_voltages, neural_trials,
        ch, fix_idx, l_frames, r_frames)
    # jitter data.
    [jitter_neu_seq,  jitter_neu_time,
     jitter_stim_vol, jitter_stim_time] = get_omi_response(
        raw_voltages, neural_trials,
        ch, jitter_idx, l_frames, r_frames)

    # plot signals.
    color_fix = ['dodgerblue', 'coral']
    color_jitter = ['violet', 'brown']
    fluo_label = ['excitory', 'inhibitory']
    num_subplots = fix_neu_seq.shape[1] + 2
    fig, axs = plt.subplots(num_subplots, 1, figsize=(20, 10))
    plt.subplots_adjust(hspace=1.2)

    # mean response of excitory.
    mean_exc_fix = np.mean(
        np.mean(fix_neu_seq[:, label==0, :], axis=0), axis=0)
    mean_exc_jitter = np.mean(
        np.mean(jitter_neu_seq[:, label==0, :], axis=0), axis=0)
    scale = np.mean(mean_exc_fix) + 2*np.std(mean_exc_fix)
    axs[0].plot(
        fix_stim_time,
        fix_stim_vol * scale,
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
        mean_exc_fix,
        color=color_fix[0],
        label='excitory_fix')
    axs[0].plot(
        jitter_neu_time,
        mean_exc_jitter,
        color=color_jitter[0],
        label='excitory_jitter')
    axs[0].set_title(
        'omission average trace of {} excitory neurons'.format(
        np.sum(label==0)))
    
    # mean response of inhibitory.
    mean_inh_fix = np.mean(
        np.mean(fix_neu_seq[:, label==1, :], axis=0), axis=0)
    mean_inh_jitter = np.mean(
        np.mean(jitter_neu_seq[:, label==1, :], axis=0), axis=0)
    scale = np.mean(mean_inh_fix) + 3*np.std(mean_inh_fix)
    axs[1].plot(
        fix_stim_time,
        fix_stim_vol * scale,
        color='grey',
        label='fix stim')
    axs[1].axvline(
        0,
        color='red',
        lw=2,
        label='omission',
        linestyle='--')
    axs[1].plot(
        fix_neu_time,
        mean_inh_fix,
        color=color_fix[1],
        label='inhibitory_fix')
    axs[1].plot(
        jitter_neu_time,
        mean_inh_jitter,
        color=color_jitter[1],
        label='inhibitory_jitter')
    axs[1].set_title(
        'omission average trace of {} inhibitory neurons'.format(
        np.sum(label==1)))
    
    # individual neuron response.
    for i in range(fix_neu_seq.shape[1]):
        fix_mean = np.mean(fix_neu_seq[:,i,:], axis=0)
        fix_sem = sem(fix_neu_seq[:,i,:], axis=0)
        jitter_mean = np.mean(jitter_neu_seq[:,i,:], axis=0)
        jitter_sem = sem(jitter_neu_seq[:,i,:], axis=0)
        scale = np.mean(fix_mean) + 3*np.std(fix_mean)
        axs[i+2].plot(
            fix_stim_time,
            fix_stim_vol * scale,
            color='grey',
            label='fix stim')
        axs[i+2].axvline(
            0,
            color='red',
            lw=2,
            label='perturbation',
            linestyle='--')
        axs[i+2].plot(
            fix_neu_time,
            fix_mean,
            color=color_fix[label[i]],
            label=fluo_label[label[i]]+'_fix')
        axs[i+2].fill_between(
            fix_neu_time,
            fix_mean - fix_sem,
            fix_mean + fix_sem,
            color=color_fix[label[i]],
            alpha=0.2)
        axs[i+2].plot(
            jitter_neu_time,
            jitter_mean,
            color=color_jitter[label[i]],
            label=fluo_label[label[i]]+'_jitter')
        axs[i+2].fill_between(
            jitter_neu_time,
            jitter_mean - jitter_sem,
            jitter_mean + jitter_sem,
            color=color_jitter[label[i]],
            alpha=0.2)
        axs[i+2].set_title(
            'omission average trace of neuron # '+ str(i).zfill(3))

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
        axs[i].autoscale(axis='y')
        axs[i].legend(loc='upper left')
    fig.set_size_inches(12, num_subplots*6)
    fig.tight_layout()

    # save figure.
    fig.savefig(os.path.join(
        ops['save_path0'], 'figures', 'fig5_align_omission.pdf'),
        dpi=300)
    plt.close()



#%% prepost


# find pre perturbation repeatition.

def get_post_num_grat(
        neural_trials,
        fix_idx,
        ):
    time = neural_trials[str(fix_idx[0])]['time']
    stim = neural_trials[str(fix_idx[0])]['stim']
    [_, _, _, dur_low] = frame_dur(stim, time)
    num_down = np.where(dur_low>np.mean(dur_low))[0][0]
    return num_down


# extract response around perturbation.

def get_prepost_response(
        raw_voltages,
        neural_trials,
        ch,
        trial_idx,
        num_down,
        ):
    # read data.
    vol_stim_bin = raw_voltages['vol_stim_bin']
    vol_time = raw_voltages['vol_time']
    neu_seq = [np.expand_dims(neural_trials[str(trials)][ch], axis=0)
            for trials in trial_idx]
    neu_time = [neural_trials[str(trials)]['time']
            for trials in trial_idx]
    stim = [neural_trials[str(trials)]['stim']
            for trials in trial_idx]
    # find voltage recordings.
    stim_vol  = []
    stim_time = []
    for t in neu_time:
        vidx = np.where(
            (vol_time > np.min(t)) &
            (vol_time < np.max(t)))[0]
        stim_vol.append(vol_stim_bin[vidx])
        stim_time.append(vol_time[vidx])
    # find perturbation point.
    post_start_idx = []
    for s,t in zip(stim, neu_time):
        [_, idx_down, _, _] = frame_dur(s, t)
        post_start_idx.append(idx_down[num_down])
    # correct voltage recordings centering at perturbation.
    stim_time  = [stim_time[i] - neu_time[i][post_start_idx[i]]
                 for i in range(len(stim_time))]
    stim_time_zero = [np.argmin(np.abs(st)) for st in stim_time]
    stim_time = trim_seq(stim_time, stim_time_zero)
    stim_vol = trim_seq(stim_vol, stim_time_zero)
    # correct neuron time stamps centering at perturbation.
    neu_time = [neu_time[i] - neu_time[i][post_start_idx[i]]
                for i in range(len(neu_time))]
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, post_start_idx)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # concatenate results.
    neu_time  = [nt.reshape(1,-1) for nt in neu_time]
    stim_time = [st.reshape(1,-1) for st in stim_time]
    stim_vol  = [sv.reshape(1,-1) for sv in stim_vol]
    neu_seq   = np.concatenate(neu_seq, axis=0)
    neu_time  = np.concatenate(neu_time, axis=0)
    stim_vol  = np.concatenate(stim_vol, axis=0)
    stim_time = np.concatenate(stim_time, axis=0)
    # get mean time stamps.
    neu_time = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    # get mode stimulus.
    stim_vol, _ = mode(stim_vol, axis=0)
    #stim_vol = np.mean(stim_vol, axis=0)
    return [neu_seq, neu_time, stim_vol, stim_time]


# main function for plot.

def plot_prepost(
        ops,
        ):
    print('plotting fig5 prepost aligned response')
    [mask, raw_traces, raw_voltages, neural_trials] = RetrieveResults.run(ops)
    ch = 'fluo_ch'+str(ops['functional_chan'])
    label = mask['label']
    
    # find trial id for jitter and fix.
    fix_idx = np.where(neural_trials['trial_type']==2)[0]
    jitter_idx = np.where(neural_trials['trial_type']==1)[0]

    # find pre perturbation repeatition.
    num_down = get_post_num_grat(neural_trials, fix_idx)

    # fix data.
    [fix_neu_seq,  fix_neu_time,
     fix_stim_vol, fix_stim_time] = get_prepost_response(
         raw_voltages, neural_trials,
         ch, fix_idx, num_down)
    # jitter data.
    [jitter_neu_seq,  jitter_neu_time,
     jitter_stim_vol, jitter_stim_time] = get_prepost_response(
         raw_voltages, neural_trials,
         ch, jitter_idx, num_down)

    # plot signals.
    color_fix = ['dodgerblue', 'coral']
    color_jitter = ['violet', 'brown']
    fluo_label = ['excitory', 'inhibitory']
    num_subplots = fix_neu_seq.shape[1] + 2
    fig, axs = plt.subplots(num_subplots, 1, figsize=(20, 10))
    plt.subplots_adjust(hspace=1.2)

    # mean response of excitory.
    mean_exc_fix = np.mean(
        np.mean(fix_neu_seq[:, label==0, :], axis=0), axis=0)
    mean_exc_jitter = np.mean(
        np.mean(jitter_neu_seq[:, label==0, :], axis=0), axis=0)
    scale = np.mean(mean_exc_fix) + 2*np.std(mean_exc_fix)
    axs[0].plot(
        fix_stim_time,
        fix_stim_vol * scale,
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
        mean_exc_fix,
        color=color_fix[0],
        label='excitory_fix')
    axs[0].plot(
        jitter_neu_time,
        mean_exc_jitter,
        color=color_jitter[0],
        label='excitory_jitter')
    axs[0].set_title(
        'perturbation average trace of {} excitory neurons'.format(
        np.sum(label==0)))
    
    # mean response of inhibitory.
    mean_inh_fix = np.mean(
        np.mean(fix_neu_seq[:, label==1, :], axis=0), axis=0)
    mean_inh_jitter = np.mean(
        np.mean(jitter_neu_seq[:, label==1, :], axis=0), axis=0)
    scale = np.mean(mean_inh_fix) + 3*np.std(mean_inh_fix)
    axs[1].plot(
        fix_stim_time,
        fix_stim_vol * scale,
        color='grey',
        label='fix stim')
    axs[1].axvline(
        0,
        color='red',
        lw=2,
        label='perturbation',
        linestyle='--')
    axs[1].plot(
        fix_neu_time,
        mean_inh_fix,
        color=color_fix[1],
        label='inhibitory_fix')
    axs[1].plot(
        jitter_neu_time,
        mean_inh_jitter,
        color=color_jitter[1],
        label='inhibitory_jitter')
    axs[1].set_title(
        'perturbation average trace of {} inhibitory neurons'.format(
        np.sum(label==1)))

    # individual neuron response.
    for i in range(fix_neu_seq.shape[1]):
        fix_mean = np.mean(fix_neu_seq[:,i,:], axis=0)
        fix_sem = sem(fix_neu_seq[:,i,:], axis=0)
        jitter_mean = np.mean(jitter_neu_seq[:,i,:], axis=0)
        jitter_sem = sem(jitter_neu_seq[:,i,:], axis=0)
        scale = np.mean(fix_mean) + 3*np.std(fix_mean)
        axs[i+2].plot(
            fix_stim_time,
            fix_stim_vol * scale,
            color='grey',
            label='fix stim')
        axs[i+2].axvline(
            0,
            color='red',
            lw=2,
            label='perturbation',
            linestyle='--')
        axs[i+2].plot(
            fix_neu_time,
            fix_mean,
            color=color_fix[label[i]],
            label=fluo_label[label[i]]+'_fix')
        axs[i+2].fill_between(
            fix_neu_time,
            fix_mean - fix_sem,
            fix_mean + fix_sem,
            color=color_fix[label[i]],
            alpha=0.2)
        axs[i+2].plot(
            jitter_neu_time,
            jitter_mean,
            color=color_jitter[label[i]],
            label=fluo_label[label[i]]+'_jitter')
        axs[i+2].fill_between(
            jitter_neu_time,
            jitter_mean - jitter_sem,
            jitter_mean + jitter_sem,
            color=color_jitter[label[i]],
            alpha=0.2)
        axs[i+2].set_title(
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
        axs[i].autoscale(axis='y')
        axs[i].legend(loc='upper left')
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
        [_, _, _, neural_trials] = RetrieveResults.run(ops)
        if len(neural_trials['trial_type']) <= 25:
            plot_omi(ops)
        else:
            plot_prepost(ops)
    except:
        print('plotting fig5 failed')

