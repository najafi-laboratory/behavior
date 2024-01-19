#!/usr/bin/env python3

import os
import h5py
import numpy as np
from datetime import datetime


def read_traces(
        ops
        ):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'temp', 'traces.h5'),
        'r')
    traces = dict()
    for k in f['traces'].keys():
        traces[k] = np.array(f['traces'][k])
    f.close()
    return traces


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
        traces,
        stim,
        time_img,
        start, end
        ):
    neural_trial = dict()
    for i in range(len(start)):
        start_idx = np.where(time_img > start[i])[0][0]
        end_idx = np.where(time_img < end[i])[0][-1] if end[i] != -1 else -1
        neural_trial[str(i)] = dict()
        neural_trial[str(i)]['time'] = time_img[start_idx:end_idx]
        neural_trial[str(i)]['stim'] = stim[start_idx:end_idx]
        for k in traces.keys():
            neural_trial[str(i)][k] = traces[k][:,start_idx:end_idx]
    return neural_trial


def save_trials(
        ops,
        neural_trial
        ):
    f = h5py.File(os.path.join(
        ops['save_path0'], 'neural_trial.h5'), 'w')
    for trial in range(len(neural_trial)):
        trial_group = f.create_group(str(trial))
        for k in neural_trial[str(trial)].keys():
            trial_group[k] = neural_trial[str(trial)][k]
    f.close()


def run(
        ops,
        time,
        vol_start, vol_stim, vol_img,
        ):
    print('===============================================')
    print('===== reconstructing synchronized signals =====')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print('Loading traces data')
    traces = read_traces(ops)
    print('Comnputing signal trigger time stamps')
    vol_start, vol_stim, vol_img = vol_to_binary(
        vol_start, vol_stim, vol_img)
    time_img   = get_trigger_time(time, vol_img)
    time_start = get_trigger_time(time, vol_start)
    print('Aligning stimulus input')
    stim = align_stim(time_img, vol_stim)
    print('Spliting trial data')
    start, end = get_trial_start_end(time_start)
    neural_trial = trial_split(traces, stim, time_img, start, end)
    print('Merging obtained trial data')
    save_trials(ops, neural_trial)
    print('Trial data saved')
    return [neural_trial]


