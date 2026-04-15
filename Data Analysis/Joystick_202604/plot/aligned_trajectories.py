# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:20:47 2025

@author: saminnaji3
"""
import os
import fitz
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
#from pypdf import PdfMerger
#from pypdf import PdfReader, PdfWriter
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import warnings
start_trial = 10
def IsSelfTimed(session_data):
    isSelfTime = session_data['isSelfTimedMode']
    
    VG = []
    ST = []
    for i in range(0 , len(isSelfTime)):
        if isSelfTime[i][start_trial] == 1 or isSelfTime[i][start_trial] == np.nan:
            ST.append(i)
        else:
            VG.append(i)
    return ST , VG

def lay_out(axs, x_label = '', y_label = '', title = '', x_lim = [], y_lim = [], legend = 0):
    if not len(x_label) == 0:
        axs.set_xlabel(x_label)
    if not len(y_label) == 0:
        axs.set_ylabel(y_label)
    if not len(title) == 0:
        axs.set_title(title)
    if not len(x_lim) == 0:
        axs.set_xlim(x_lim)
    if not len(y_lim) == 0:
        axs.set_ylim(y_lim)
    if not legend == 0:
        axs.legend()
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
def plot_trajectory(axs, align_data, align_time, state, color_tag = 'k'):
    
    trajectory_mean = np.nanmean(align_data , axis = 0)
    trajectory_sem = np.nanstd(align_data , axis = 0)/np.sqrt(len(align_data))
    axs.fill_between(align_time/1000 ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = color_tag , alpha = 0.2)
    axs.plot(align_time/1000 , trajectory_mean , color = color_tag, linewidth = 0.1, label = 'n = ' + str(len(align_data)))
    
    if state == 'vis1' or state == 'vis2':
        axs.axvspan(0, 0.1, alpha=0.2, color='gray')
    else:
        axs.axvline(x = 0, color = 'gray', linestyle='--', linewidth = 1)
    if state == 'retract1':
        axs.axvline(x = 0.43, color = 'y', linestyle='--', linewidth = 1, alpha = 0.3)
        axs.axvline(x = 0.93, color = 'b', linestyle='--', linewidth = 1, alpha = 0.3)
        
# pad sequence with time stamps to the longest length with nan.
# def pad_seq(align_data, align_time):
#     pad_time = np.arange(
#         np.min([np.min(t) for t in align_time]),
#         np.max([np.max(t) for t in align_time]) + 1)
#     pad_data = []
#     for data, time in zip(align_data, align_time):
#         aligned_seq = np.full_like(pad_time, np.nan, dtype=float)
#         idx = np.searchsorted(pad_time, time)
#         aligned_seq[idx] = data
#         pad_data.append(aligned_seq)
#     return pad_data, pad_time

import numpy as np
from typing import List, Tuple, Union

def pad_seq(
    align_data: List[np.ndarray],
    align_time: List[np.ndarray],
    path: str = "pad.dat",
    dtype: np.dtype = np.float32,
    dense: bool = False,
    mode: str = "w+",      # "w+" create/overwrite; "r+" update existing; "r" read-only
) -> Tuple[np.memmap, np.ndarray]:
    """
    Pads each (data, time) trial onto a common time axis and stores the
    result in a disk-backed np.memmap array of shape (n_trials, T).

    Returns
    -------
    out : np.memmap
        Disk-backed array; call out.flush() to sync to disk.
    pad_time : np.ndarray
        The shared time axis.

    Notes
    -----
    - Use dense=False if your timestamps are sparse; it uses the union of observed times
      instead of the full [tmin..tmax] range, saving space.
    - dtype=np.float32 halves memory vs float64.
    """

    if not align_data or not align_time:
        # minimal empty memmap
        out = np.memmap(path, mode="w+", dtype=dtype, shape=(0, 0))
        pad_time = np.empty((0,), dtype=np.int64)
        return out, pad_time

    if len(align_data) != len(align_time):
        raise ValueError("align_data and align_time must have the same number of trials")

    # Build shared time axis
    if dense:
        tmin = min(np.min(t) for t in align_time)
        tmax = max(np.max(t) for t in align_time)
        pad_time = np.arange(tmin, tmax + 1, dtype=np.asarray(align_time[0]).dtype)
    else:
        pad_time = np.unique(np.concatenate([np.asarray(t) for t in align_time]))

    n_trials = len(align_data)
    T = pad_time.size

    # Create the disk-backed matrix and fill with NaN
    out = np.memmap(path, mode=mode, dtype=dtype, shape=(n_trials, T))
    out[:] = np.nan  # initialize

    # Fill row by row (no big temporaries)
    for i, (data, time) in enumerate(zip(align_data, align_time)):
        data = np.asarray(data, dtype=dtype)
        time = np.asarray(time)

        if data.size != time.size:
            raise ValueError(f"Trial {i}: data and time lengths differ ({data.size} vs {time.size})")

        # positions in shared axis
        idx = np.searchsorted(pad_time, time)
        out[i, idx] = data

    out.flush()  # ensure data hits disk
    return out, pad_time


def get_js_pos(event_timing, state):
    interval = 1
    js_time = event_timing['js_time']
    js_pos = event_timing['js_pos']
    
    trial_type = event_timing['trial_types']
    
    opto = event_timing['opto']
    
    inter_time = []
    inter_pos = []
    for (pos, time) in zip(js_pos, js_time):
        interpolator = interp1d(time, pos, bounds_error=False)
        new_time = np.arange(np.min(time), np.max(time), interval)
        new_pos = interpolator(new_time)
        inter_time.append(new_time)
        inter_pos.append(new_pos)
    if np.size(event_timing[state][0]) == 1:
        time_state = event_timing[state] 
    if np.size(event_timing[state][0]) == 2:
        time_state = [event_timing[state][t][0] for t in range(len(event_timing['js_time']))]
    #print(len(trial_type), len(time_state))
    #print(time_state)
    trial_type = np.array([trial_type[i]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i]])
    opto_tag = np.array([opto[i]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i]])

    zero_state = [np.argmin(np.abs(inter_time[i] - time_state[i]))
                  for i in range(len(inter_time))
                  if not np.isnan(time_state)[i]]
    align_data = [inter_pos[i] - inter_pos[i][0]
                  for i in range(len(inter_pos))
                  if not np.isnan(time_state)[i]]
    align_time = [inter_time[i]
                  for i in range(len(inter_time))
                  if not np.isnan(time_state)[i]]
    align_time = [align_time[i] - align_time[i][zero_state[i]]
                  for i in range(len(align_time))]
    if len(align_data) > 0:
        align_data, align_time = pad_seq(align_data, align_time)
        align_data = np.array(align_data)
    else:
        align_data = np.array([[np.nan]])
        align_time = np.array([np.nan])
        trial_type = np.array([np.nan])
        opto_tag = np.array([np.nan])
    return align_data, align_time, trial_type, opto_tag

def selected_trials_js_pos(align_data, ref1, con1, ref2= np.nan, con2= np.nan, random = 0):
    selected_trial_id = np.where(ref1 == con1)[0]
    if random:
        n = selected_trial_id.shape[0] // 2
        idx = np.random.choice(selected_trial_id.shape[0], n, replace=False)
        selected_trial_id = selected_trial_id[idx]
    selected_align_data = align_data[selected_trial_id , :]
    if not np.isnan(con2):
        selected_ref2 = ref2[selected_trial_id]
        selected_trial_id = np.where(selected_ref2 == con2)[0]
        selected_align_data = selected_align_data[selected_trial_id , :]
    return selected_align_data

def aligned_data(axs1, axs2, axs3, axs4, session_data, s_l = np.nan, opto = np.nan, st = 0):
    all_states = ['vis1' , 'push1' , 'retract1_init' , 'retract1', 'vis2' , 'wait2', 'push2', 'reward', 'reward_lick', 'punish' ,'retract2']
    event_timing = session_data['events_timing']
    ST , VG = IsSelfTimed(session_data)
    num_session = len(event_timing)
    if st == 1:
        selected_sessions = np.array(ST)
        num_session = len(ST)
    elif st == 2:
        selected_sessions = np.array(VG)
        num_session = len(VG)
    else:
        selected_sessions = np.arange(num_session)
    cmap_control = plt.cm.Greys
    cmap_opto = plt.cm.Blues
    norm = plt.Normalize(vmin=0, vmax=num_session+1)
    num = 0
    x_range = [-1.5, 2.5]
    y_range = [-0.05, 3]
    for state in all_states:
        
        for sess in selected_sessions:
            print('plotting for: ', state, ' of session:' , str(sess))
            session_event_timing = event_timing[sess]
            align_data, align_time, trial_type, opto_tag = get_js_pos(session_event_timing, state)
            if  np.isnan(s_l) and np.isnan(opto):
                plot_trajectory(axs1[num], align_data, align_time, state, color_tag = cmap_control(norm(np.where(selected_sessions==sess)[0][0]+1)))
                lay_out(axs1[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' All', x_lim = x_range, y_lim = y_range, legend = 0)
            elif np.isnan(s_l) and not(np.isnan(opto)):
                align_data1 =selected_trials_js_pos(align_data, opto_tag, 0)
                plot_trajectory(axs1[num], align_data1, align_time, state, color_tag = cmap_control(norm(np.where(selected_sessions==sess)[0][0]+1)))
                lay_out(axs1[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Cotrol', x_lim = x_range, y_lim = y_range, legend = 0)
                align_data2 =selected_trials_js_pos(align_data, opto_tag, 1)
                plot_trajectory(axs2[num], align_data2, align_time, state, color_tag = cmap_opto(norm(np.where(selected_sessions==sess)[0][0]+1)))
                lay_out(axs2[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Opto', x_lim = x_range, y_lim = y_range, legend = 0)
            elif not(np.isnan(s_l)) and np.isnan(opto):
                align_data1 =selected_trials_js_pos(align_data, trial_type, 1)
                plot_trajectory(axs1[num], align_data1, align_time, state, color_tag = cmap_control(norm(np.where(selected_sessions==sess)[0][0]+1)))
                lay_out(axs1[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Short', x_lim = x_range, y_lim = y_range, legend = 0)
                align_data2 =selected_trials_js_pos(align_data, trial_type, 2)
                plot_trajectory(axs2[num], align_data2, align_time, state, color_tag = cmap_control(norm(np.where(selected_sessions==sess)[0][0]+1)))
                lay_out(axs2[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Long', x_lim = x_range, y_lim = y_range, legend = 0)
            else:
                align_data1 =selected_trials_js_pos(align_data, trial_type, 1, opto_tag, 0)
                plot_trajectory(axs1[num], align_data1, align_time, state, color_tag = cmap_control(norm(np.where(selected_sessions==sess)[0][0]+1)))
                lay_out(axs1[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Short Control', x_lim = x_range, y_lim = y_range, legend = 0)
                align_data1 =selected_trials_js_pos(align_data, trial_type, 1, opto_tag, 1)
                plot_trajectory(axs2[num], align_data1, align_time, state, color_tag = cmap_opto(norm(np.where(selected_sessions==sess)[0][0]+1)))
                lay_out(axs2[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Short Opto', x_lim = x_range, y_lim = y_range, legend = 0)
                align_data1 =selected_trials_js_pos(align_data, trial_type, 2, opto_tag, 0)
                plot_trajectory(axs3[num], align_data1, align_time, state, color_tag = cmap_control(norm(np.where(selected_sessions==sess)[0][0]+1)))
                lay_out(axs3[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Long Control', x_lim = x_range, y_lim = y_range, legend = 0)
                align_data1 =selected_trials_js_pos(align_data, trial_type, 2, opto_tag, 1)
                plot_trajectory(axs4[num], align_data1, align_time, state, color_tag = cmap_opto(norm(np.where(selected_sessions==sess)[0][0])))
                lay_out(axs4[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Long Opto', x_lim = x_range, y_lim = y_range, legend = 0)
        num = num + 1
        
def concat(align_date):
    output = []
    length = []
    for sublist in align_date:
        for a in sublist:
            length.append(len(a))
            
    mini = np.min(length)
    #align_time1 = align_time[:mini]
    for sublist in align_date:
        for a in sublist:
            output.append(a[:mini])
    return output

def aligned_data_mega_session(axs1, axs2, session_data, s_l = np.nan, opto = np.nan, st = 0, random = 0):
    all_states = ['vis1' , 'push1' , 'retract1_init' , 'retract1', 'vis2' , 'wait2', 'push2', 'reward', 'reward_lick', 'punish' ,'retract2']
    all_states = ['vis1' , 'push1', 'retract1', 'vis2' , 'wait2', 'push2', 'reward', 'reward_lick' ,'retract2']
    event_timing = session_data['events_timing']
    ST , VG = IsSelfTimed(session_data)
    num_session = len(event_timing)
    if st == 1:
        selected_sessions = np.array(ST)
        num_session = len(ST)
    elif st == 2:
        selected_sessions = np.array(VG)
        num_session = len(VG)
    else:
        selected_sessions = np.arange(num_session)
            
    num = 0
    x_range = [-1.5, 2.5]
    y_range = [-0.05, 3]
    for state in all_states:
        print('plotting mega session for: ', state)
        concat_state = []
        concat_time = []
        concat_type = []
        concat_opto = []
        concat_js_pos = []
        for sess in selected_sessions:
            session_event_timing = event_timing[sess]
            concat_state.extend(session_event_timing[state])
            concat_time.extend(session_event_timing['js_time'])
            concat_type.extend(session_event_timing['trial_types'])
            concat_opto.extend(session_event_timing['opto'])
            concat_js_pos.extend(session_event_timing['js_pos'])
        mega_session = {
            'trial_types' : concat_type,
            'opto'        : concat_opto,
            state         : concat_state,
            'js_pos'      : concat_js_pos,
            'js_time'     : concat_time}
        align_data, align_time, trial_type, opto_tag = get_js_pos(mega_session, state)
        print('start plotting')
        if  np.isnan(s_l) and np.isnan(opto):
            plot_trajectory(axs1[num], align_data, align_time, state, color_tag = 'k')
            lay_out(axs1[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' All', x_lim = x_range, y_lim = y_range, legend = 1)
        elif np.isnan(s_l) and not(np.isnan(opto)):
            align_data1 =selected_trials_js_pos(align_data, opto_tag, 0,random = random)
            plot_trajectory(axs1[num], align_data1, align_time, state, color_tag = 'k')
            lay_out(axs1[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Cotrol/Opto', x_lim = x_range, y_lim = y_range, legend = 1)
            align_data2 =selected_trials_js_pos(align_data, opto_tag, 1,random = random)
            plot_trajectory(axs1[num], align_data2, align_time, state, color_tag = 'b')
            lay_out(axs1[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Cotrol/Opto', x_lim = x_range, y_lim = y_range, legend = 1)
        elif not(np.isnan(s_l)) and np.isnan(opto):
            align_data1 =selected_trials_js_pos(align_data, trial_type, 1,random = random)
            plot_trajectory(axs1[num], align_data1, align_time, state, color_tag = 'k')
            lay_out(axs1[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Short', x_lim = x_range, y_lim = y_range, legend = 1)
            align_data2 =selected_trials_js_pos(align_data, trial_type, 2,random = random)
            plot_trajectory(axs2[num], align_data2, align_time, state, color_tag = 'k')
            lay_out(axs2[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Long', x_lim = x_range, y_lim = y_range, legend = 1)
        else:
            align_data1 =selected_trials_js_pos(align_data, trial_type, 1, opto_tag, 0,random = random)
            plot_trajectory(axs1[num], align_data1, align_time, state, color_tag = 'k')
            lay_out(axs1[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Short Control/Opto', x_lim = x_range, y_lim = y_range, legend = 1)
            align_data1 =selected_trials_js_pos(align_data, trial_type, 1, opto_tag, 1,random = random)
            plot_trajectory(axs1[num], align_data1, align_time, state, color_tag = 'b')
            lay_out(axs1[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Short Control/Opto', x_lim = x_range, y_lim = y_range, legend = 1)
            align_data1 =selected_trials_js_pos(align_data, trial_type, 2, opto_tag, 0,random = random)
            plot_trajectory(axs2[num], align_data1, align_time, state, color_tag = 'k')
            lay_out(axs2[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Long Control/Opto', x_lim = x_range, y_lim = y_range, legend = 1)
            align_data1 =selected_trials_js_pos(align_data, trial_type, 2, opto_tag, 1,random = random)
            plot_trajectory(axs2[num], align_data1, align_time, state, color_tag = 'b')
            lay_out(axs2[num], x_label = 'Time (S)', y_label = 'Jostick def', title = state + ' Long Control/Opto', x_lim = x_range, y_lim = y_range, legend = 1)
        num = num + 1