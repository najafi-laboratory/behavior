#!/usr/bin/env python3

import os
import h5py
import numpy as np


def thres_binary(
        data,
        thres
        ):
    data[data<thres] = 0
    data[data>thres] = 1
    return data


def vol_to_binary(
        vol_start,
        vol_stim,
        vol_img
        ):
    vol_start = thres_binary(vol_start, 1)
    vol_stim  = thres_binary(vol_stim, 1)
    vol_img   = thres_binary(vol_img, 1)
    return vol_start, vol_stim, vol_img


def get_trigger_time(
        time,
        vol
        ):
    diff_vol = np.diff(vol, append=0)
    time_idx = np.where(diff_vol == 1)[0]+1
    time_tri = time[time_idx]
    return time_tri


def align_stim(
        time_img,
        vol_stim
        ):
    stim = vol_stim[time_img]
    return stim


def get_trial_start_end(
        time_start
        ):
    start = []
    end = []
    for i in range(len(time_start)):
        s = time_start[i]
        e = time_start[i+1] if i != len(time_start)-1 else -1
        start.append(s)
        end.append(e)
    return start, end


def trial_split(
        data,
        start,
        end
        ):
    trial_data = []
    for i in range(len(start)):
        if len(data.shape) == 1:
            trial_data.append(data[start[i]:end[i]].astype('float64'))
        if len(data.shape) == 2:
            trial_data.append(data[:,start[i]:end[i]].astype('float64'))
    return trial_data


def save_trials(
        ops,
        trial_stim,
        trial_fluo_ch1,
        trial_fluo_ch2,
        trial_mean_fluo_ch1,
        trial_mean_fluo_ch2,
        trial_spikes_ch1,
        trial_spikes_ch2
        ):
    hf = h5py.File(os.path.join(
        ops['save_path0'], 'neural_trial.h5'), 'w')
    for i in range(len(trial_stim)):
        trial_group = hf.create_group(str(i))
        trial_group['stim'] = trial_stim[i]
        trial_group['fluo_ch1'] = trial_fluo_ch1[i]
        trial_group['fluo_ch2'] = trial_fluo_ch2[i]
        trial_group['mean_fluo_ch1'] = trial_mean_fluo_ch1[i]
        trial_group['mean_fluo_ch2'] = trial_mean_fluo_ch2[i]
        trial_group['spikes_ch1'] = trial_spikes_ch1[i]
        trial_group['spikes_ch2'] = trial_spikes_ch2[i]
    hf.close()


def run(
        ops,
        time, vol_start, vol_stim, vol_img,
        fluo_ch1, mean_fluo_ch1, spikes_ch1,
        fluo_ch2, mean_fluo_ch2, spikes_ch2
        ):
    print('===============================================')
    print('===== reconstructing synchronized signals =====')
    print('===============================================')
    print('Comnputing signal trigger time stamps')
    vol_start, vol_stim, vol_img = vol_to_binary(
        vol_start, vol_stim, vol_img)
    time_img   = get_trigger_time(time, vol_img)
    time_start = get_trigger_time(time, vol_start)
    print('Aligning stimulus input')
    stim = align_stim(time_img, vol_stim)
    print('Spliting trial data')
    start, end = get_trial_start_end(time_start)
    trial_stim = trial_split(stim, start, end)
    trial_fluo_ch1 = trial_split(fluo_ch1, start, end)
    trial_fluo_ch2 = trial_split(fluo_ch2, start, end)
    trial_mean_fluo_ch1 = trial_split(mean_fluo_ch1, start, end)
    trial_mean_fluo_ch2 = trial_split(mean_fluo_ch2, start, end)
    trial_spikes_ch1 = trial_split(spikes_ch1, start, end)
    trial_spikes_ch2 = trial_split(spikes_ch2, start, end)
    print('Merging obtained trial data')
    save_trials(
        ops, trial_stim,
        trial_fluo_ch1,
        trial_fluo_ch2,
        trial_mean_fluo_ch1,
        trial_mean_fluo_ch2,
        trial_spikes_ch1,
        trial_spikes_ch2
        )
    print('Trial data saved')
    return [time_img,
            trial_stim,
            trial_fluo_ch1,
            trial_fluo_ch2,
            trial_mean_fluo_ch1,
            trial_mean_fluo_ch2,
            trial_spikes_ch1,
            trial_spikes_ch2]


