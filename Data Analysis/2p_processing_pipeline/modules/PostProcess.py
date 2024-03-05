#!/usr/bin/env python3

import os
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from suite2p.extraction import oasis


#%% process voltage recordings


# read the voltage recording file.

def read_vol_to_np(
        ops,
        ):
    # voltage: SESSION_Cycle00001_VoltageRecording_000NUM.csv.
    vol_record = [f for f in os.listdir(ops['data_path'])
                  if 'VoltageRecording' in f and '.csv' in f]
    df_vol = pd.read_csv(
        os.path.join(ops['data_path'], vol_record[0]),
        engine='python')
    # column 0: time index in ms.
    # column 1: trial start signal from bpod.
    # column 2: stimulus signal from photodiode.
    # column 3: BNC2 not in use.
    # column 4: image trigger signal from 2p scope camera.
    vol_time  = df_vol.iloc[:,0].to_numpy()
    vol_start = df_vol.iloc[:,1].to_numpy()
    vol_stim  = df_vol.iloc[:,2].to_numpy()
    vol_img   = df_vol.iloc[:,4].to_numpy()
    return vol_time, vol_start, vol_stim, vol_img


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


# save voltage data.

def save_vol(
        ops,
        vol_time, vol_start_bin, vol_stim_bin, vol_img_bin,
        ):
    # file structure:
    # ops['save_path0'] / raw_voltages.h5
    # -- raw
    # ---- vol_time
    # ---- vol_start_bin
    # ---- vol_stim_bin
    # ---- vol_img_bin
    f = h5py.File(os.path.join(
        ops['save_path0'], 'raw_voltages.h5'), 'w')
    grp = f.create_group('raw')
    grp['vol_time']      = vol_time
    grp['vol_start_bin'] = vol_start_bin
    grp['vol_stim_bin']  = vol_stim_bin
    grp['vol_img_bin']   = vol_img_bin
    f.close()
    

# main function to process voltage recordings

def run_vol(ops):
    vol_time, vol_start, vol_stim, vol_img = read_vol_to_np(ops)
    vol_start_bin, vol_stim_bin, vol_img_bin = vol_to_binary(
        vol_start, vol_stim, vol_img)
    save_vol(ops, vol_time, vol_start_bin, vol_stim_bin, vol_img_bin)


#%% trace normalization and spike detection


# compute dff from raw fluorescence signals.

def get_dff(
        ops,
        fluo,
        neuropil,
        ):
    # correct with neuropil signals.
    dff = fluo.copy() - ops['neucoeff']*neuropil
    # baseline subtraction with window to compute df/f.
    f0 = gaussian_filter(fluo, [0., ops['sig_baseline']])
    # scale with individual variance.
    for j in range(dff.shape[0]):
        dff[j,:] = (dff[j,:] - f0[j,:]) / (f0[j,:] * np.std(dff[j,:]))
        dff[j,:] = dff[j,:] - np.mean(dff[j,:])
    return dff


# run spike detection on fluorescence signals.

def spike_detect(
        ops,
        fluo
        ):
    # oasis for spike detection.
    spikes = oasis(
        F=fluo,
        batch_size=ops['batch_size'],
        tau=ops['tau'],
        fs=ops['fs'])
    return spikes


# threshold spikes based on variance.

def thres_spikes(
        ops,
        fluo, spikes
        ):
    for i in range(spikes.shape[0]):
        s = spikes[i,:].copy()
        thres = np.mean(fluo[i,:]) + ops['spike_thres'] * np.std(fluo)
        s[s<thres] = 0
        spikes[i,:] = s
    return spikes


#%% trialization


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
    # avoid going up but not down again at the end.
    stim_time_up = stim_time_up[:len(stim_time_down)]
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
        vol_time,
        vol_start_bin,
        ):
    time_up, time_down = get_trigger_time(vol_time, vol_start_bin)
    # find the impulse start signal.
    time_start = []
    for i in range(len(time_up)):
        if time_down[i] - time_up[i] < 200:
            time_start.append(time_up[i])
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
        raw_traces,
        stim,
        time_neuro,
        start, end
        ):
    neural_trials = dict()
    for i in range(len(start)):
        # find start and end timing index.
        start_idx = np.where(time_neuro > start[i])[0][0]
        end_idx = np.where(time_neuro < end[i])[0][-1] if end[i] != -1 else -1
        # save time and stimulus for each trial.
        neural_trials[str(i)] = dict()
        neural_trials[str(i)]['time'] = time_neuro[start_idx:end_idx]
        neural_trials[str(i)]['stim'] = stim[start_idx:end_idx]
        # save traces for each trial.
        for k in raw_traces.keys():
            neural_trials[str(i)][k] = raw_traces[k][:,start_idx:end_idx]
    return neural_trials



# save trial neural data.

def save_trials(
        ops,
        neural_trials
        ):
    # file structure:
    # ops['save_path0'] / neural_trials.h5
    # -- trial_id
    # ---- 1
    # ------ fluo_ch1
    # ------ fluo_ch2
    # ------ neuropil_ch1
    # ------ neuropil_ch2
    # ---- 2
    # ...
    f = h5py.File(
        os.path.join(ops['save_path0'], 'neural_trials.h5'),
        'w')
    grp = f.create_group('trial_id')
    for trial in range(len(neural_trials)):
        trial_group = grp.create_group(str(trial))
        for k in neural_trials[str(trial)].keys():
            trial_group[k] = neural_trials[str(trial)][k]
    f.close()


# main function for trialization

def run_trialization(ops):
    
    print('Reading raw traces and voltage recordings')
    raw_traces = read_raw_traces(ops)
    raw_voltages = read_raw_voltages(ops)
    vol_time      = raw_voltages['vol_time']
    vol_start_bin = raw_voltages['vol_start_bin']
    vol_stim_bin  = raw_voltages['vol_stim_bin']
    vol_img_bin   = raw_voltages['vol_img_bin']
    
    print('Comnputing signal trigger time stamps')
    time_img, _   = get_trigger_time(vol_time, vol_img_bin)

    # correct imaging timing
    time_neuro = correct_time_img_center(time_img)

    # stimulus alignment.
    print('Aligning stimulus input')
    stim = align_stim(vol_time, time_neuro, vol_stim_bin)

    # trial segmentation.
    print('Spliting trial data')
    start, end = get_trial_start_end(vol_time, vol_start_bin)
    neural_trials = trial_split(raw_traces, stim, time_neuro, start, end)

    # save the final data.
    print('Merging obtained trial data')
    save_trials(
        ops,
        neural_trials)
    print('Trial data saved')
    

#%% read saved results


# read saved ops.npy given a folder in ./results.

# ops = read_ops('testdata')
def read_ops(save_folder):
    ops = np.load(
        os.path.join('./results', save_folder, 'ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join('./results', save_folder)
    return ops


# read raw_traces.h5.

def read_raw_traces(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'raw_traces.h5'),
        'r')
    raw_traces = dict()
    for k in f['raw'].keys():
        raw_traces[k] = np.array(f['raw'][k])
    f.close()
    return raw_traces


# read raw_voltages.h5

def read_raw_voltages(ops):
    try:
        f = h5py.File(
            os.path.join(ops['save_path0'], 'raw_voltages.h5'),
            'r')
        raw_voltages = dict()
        for k in f['raw'].keys():
            raw_voltages[k] = np.array(f['raw'][k])
        f.close()
        return raw_voltages
    except:
        print('Fail to read voltage data')
        return None


# read neural_trials.h5

def read_neural_trials(ops):
    try:
        f = h5py.File(
            os.path.join(ops['save_path0'], 'neural_trials.h5'),
            'r')
        neural_trials = dict()
        for trial in f['trial_id'].keys():
            neural_trials[trial] = dict()
            for data in f['trial_id'][trial].keys():
                neural_trials[trial][data] = np.array(f['trial_id'][trial][data])
        f.close()
        neural_trials = add_jitter_flag(neural_trials)
        return neural_trials
    except:
        print('Fail to read trial data')
        return None


# tentative function to hard code trial types.

def add_jitter_flag(neural_trials):
    from sklearn.mixture import GaussianMixture
    def frame_dur(stim, time):
        diff_stim = np.diff(stim, prepend=0)
        idx_up   = np.where(diff_stim == 1)[0]
        idx_down = np.where(diff_stim == -1)[0]
        dur_high = time[idx_down] - time[idx_up]
        dur_low  = time[idx_up[1:]] - time[idx_down[:-1]]
        return [dur_high, dur_low]
    def get_mean_std(isi):
        gmm = GaussianMixture(n_components=2)
        gmm.fit(isi.reshape(-1,1))
        std = np.mean(np.sqrt(gmm.covariances_.flatten()))
        return std
    thres = 25
    jitter_flag = []
    for i in range(len(neural_trials)):
        stim = neural_trials[str(i)]['stim']
        time = neural_trials[str(i)]['time']
        [_, isi] = frame_dur(stim, time)
        std = get_mean_std(isi)
        jitter_flag.append(std)
    jitter_flag = np.array(jitter_flag)
    jitter_flag[jitter_flag<thres] = 0
    jitter_flag[jitter_flag>thres] = 1
    neural_trials['jitter_flag'] = jitter_flag
    return neural_trials


# read mask.h5.

def read_mask(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'mask.h5'),
        'r')
    mask = dict()
    mask['label'] = np.array(f['label'])
    for ch in f['mask'].keys():
        mask[ch] = dict()
        for k in f['mask'][ch].keys():
            mask[ch][k] = np.array(f['mask'][ch][k])
    f.close()
    return mask


# main function to read completed results.

def run_read_results(ops):
    mask = read_mask(ops)
    raw_traces = read_raw_traces(ops)
    raw_voltages = read_raw_voltages(ops)
    neural_trials = read_neural_trials(ops)
    return [mask, raw_traces, raw_voltages, neural_trials]


#%% bpod session data processing


# read bpod session data.

def read_bpod_mat_data(fname):
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], sio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d
    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d
    def _tolist(ndarray):
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    bpod_sess_data = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    bpod_sess_data = _check_keys(bpod_sess_data)
    bpod_sess_data = bpod_sess_data['SessionData']
    return bpod_sess_data



