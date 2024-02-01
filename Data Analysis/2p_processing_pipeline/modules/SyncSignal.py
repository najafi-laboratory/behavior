#!/usr/bin/env python3

import os
import h5py
import numpy as np
from datetime import datetime


# read raw traces.

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


# threshold the continuous voltage recordings to 01 series.

def thres_binary(
        data,
        thres
        ):
    data_bin = data.copy()
    data_bin[data_bin<thres] = 0
    data_bin[data_bin>thres] = 1
    return data_bin


# convert all voltage recordings to binary series.

def vol_to_binary(
        vol_start,
        vol_stim,
        vol_img
        ):
    vol_start_bin = thres_binary(vol_start, 1)
    vol_stim_bin  = thres_binary(vol_stim, 1)
    vol_img_bin   = thres_binary(vol_img, 1)
    return vol_start_bin, vol_stim_bin, vol_img_bin


# detect the rising edge and falling edge of binary series.

def get_trigger_time(
        vol_time,
        vol_bin
        ):
    # find the edge with np.diff and correct it by preappend one 0.
    diff_vol = np.diff(vol_bin, prepend=0)
    idx_up = np.where(diff_vol == 1)[0]
    idx_down = np.where(diff_vol == -1)[0]
    # select the indice for risging and falling.
    # give the edges in ms.
    time_up   = vol_time[idx_up]
    time_down = vol_time[idx_down]
    return time_up, time_down


# correct the fluorescence signal timing.

def correct_time_img_center(time_img):
    # find the frame internal.
    diff_time_img = np.diff(time_img, append=0)
    # correct the last element.
    diff_time_img[-1] = np.mean(diff_time_img[:-1])
    # move the image timing to the center of photon integration interval.
    diff_time_img = diff_time_img / 2
    # correct each individual timing.
    time_neuro = time_img + diff_time_img
    return time_neuro


# align the stimulus sequence with fluorescence signal.

def align_stim(
        vol_time,
        time_neuro,
        vol_stim_bin,
        ):
    # find the rising and falling time of stimulus.
    stim_time_up, stim_time_down = get_trigger_time(
        vol_time, vol_stim_bin)
    # assign the start and end time to fluorescence frames.
    stim_start = []
    stim_end = []
    for i in range(len(stim_time_up)):
        # find the nearest frame that stimulus start or end.
        stim_start.append(
            np.argmin(np.abs(time_neuro - stim_time_up[i])))
        stim_end.append(
            np.argmin(np.abs(time_neuro - stim_time_down[i])))
    # reconstruct stimulus sequence.
    stim = np.zeros(len(time_neuro))
    for i in range(len(stim_start)):
        stim[stim_start[i]:stim_end[i]] = 1
    return stim


# process trial start signal.

def get_trial_start_end(
        time_start
        ):
    start = []
    end = []
    # assume the current trial end at the next start point.
    for i in range(len(time_start)):
        s = time_start[i]
        e = time_start[i+1] if i != len(time_start)-1 else -1
        start.append(s)
        end.append(e)
    return start, end


# trial segmentation.

def trial_split(
        traces,
        stim,
        time_neuro,
        start, end
        ):
    neural_trial = dict()
    for i in range(len(start)):
        # find start and end timing index.
        start_idx = np.where(time_neuro > start[i])[0][0]
        end_idx = np.where(time_neuro < end[i])[0][-1] if end[i] != -1 else -1
        # save time and stimulus for each trial.
        neural_trial[str(i)] = dict()
        neural_trial[str(i)]['time'] = time_neuro[start_idx:end_idx]
        neural_trial[str(i)]['stim'] = stim[start_idx:end_idx]
        # save traces for each trial.
        for k in traces.keys():
            neural_trial[str(i)][k] = traces[k][:,start_idx:end_idx]
    return neural_trial


# save neural data.

def save_raw(
        ops,
        traces,
        vol_time, vol_start_bin, vol_stim_bin, vol_img_bin,
        ):
    # file structure:
    # -- raw
    # ---- vol_time
    # ---- vol_start_bin
    # ---- vol_stim_bin
    # ---- vol_img_bin
    # ---- fluo_ch1
    # ---- fluo_ch2
    # ---- mean_fluo_ch1
    # ---- mean_fluo_ch2
    # ---- spikes_ch1
    # ---- spikes_ch2
    f = h5py.File(os.path.join(
        ops['save_path0'], 'neural_trial.h5'), 'w')
    raw_g = f.create_group('raw')
    raw_g['vol_time']      = vol_time
    raw_g['vol_start_bin'] = vol_start_bin
    raw_g['vol_stim_bin']  = vol_stim_bin
    raw_g['vol_img_bin']   = vol_img_bin
    for k in traces.keys():
        raw_g[k] = traces[k]
    f.close()


# save trial neural data.

def save_trials(
        ops,
        neural_trial
        ):
    # file structure:
    # -- 1
    # ---- fluo_ch1
    # ---- fluo_ch2
    # ---- mean_fluo_ch1
    # ---- mean_fluo_ch2
    # ---- spikes_ch1
    # ---- spikes_ch2
    # -- 2
    # ...
    f = h5py.File(os.path.join(
        ops['save_path0'], 'neural_trial.h5'), 'w')
    for trial in range(len(neural_trial)):
        trial_group = f.create_group(str(trial))
        for k in neural_trial[str(trial)].keys():
            trial_group[k] = neural_trial[str(trial)][k]
    f.close()


# main function for series alignment and trial segmentation.

def run(
        ops,
        vol_time,
        vol_start, vol_stim, vol_img,
        ):
    print('===============================================')
    print('===== reconstructing synchronized signals =====')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # read the raw traces.
    print('Loading traces data')
    traces = read_traces(ops)

    # process the voltage recordings.
    vol_start_bin, vol_stim_bin, vol_img_bin = vol_to_binary(
        vol_start, vol_stim, vol_img)
    save_raw(
        ops, traces,
        vol_time, vol_start_bin, vol_stim_bin, vol_img_bin)

    try:

        print('Comnputing signal trigger time stamps')
        time_img, _   = get_trigger_time(vol_time, vol_img_bin)
        time_start, _ = get_trigger_time(vol_time, vol_start_bin)

        # correct imaging timing
        time_neuro = correct_time_img_center(time_img)

        # stimulus alignment.
        print('Aligning stimulus input')
        stim = align_stim(vol_time, time_neuro, vol_stim_bin)

        # trial segmentation.
        print('Spliting trial data')
        start, end = get_trial_start_end(time_start)
        neural_trial = trial_split(traces, stim, time_neuro, start, end)

        # save the final data.
        print('Merging obtained trial data')
        save_trials(
            ops,
            neural_trial)
        print('Trial data saved')

    except:

        print('Trialize data failed due to voltage recordings issue')

    return []


