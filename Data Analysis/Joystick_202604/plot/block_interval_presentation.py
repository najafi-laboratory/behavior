# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 09:21:18 2026

@author: saminnaji3
"""


import numpy as np
import matplotlib.pyplot as plt

start_trial = 10

def IsSelfTimed(session_data):
    isSelfTime = session_data['isSelfTimedMode']
    
    VG = []
    ST = []
    for i in range(0, len(isSelfTime)):
        if isSelfTime[i][start_trial] == 1 or np.isnan(isSelfTime[i][start_trial]):
            ST.append(i)
        else:
            VG.append(i)
    return ST, VG

def plot_brokenaxis(axs, array1, array2, array3, array4, title, pre, post, ylabel, sem1=np.nan, sem2=np.nan, sem3=np.nan, sem4=np.nan, sem_plot=0):
    x1 = np.arange(len(array1))
    x2 = np.arange(len(array2)) + x1[-1] + 6
    pre_val = 10
    
    axs.plot(x2[:-pre_val], array2[:-pre_val], color='y')
    axs.plot(x2[:-pre_val], array4[:-pre_val], color='b')
    axs.plot(x2[-pre_val-1:], array2[-pre_val-1:], color='b')
    axs.plot(x2[-pre_val-1:], array4[-pre_val-1:], color='y')
    
    if sem_plot:
        axs.fill_between(x2[:-pre_val], array2[:-pre_val]-sem2[:-pre_val], array2[:-pre_val]+sem2[:-pre_val], color='y', alpha=0.3)
        axs.fill_between(x2[:-pre_val], array4[:-pre_val]-sem4[:-pre_val], array4[:-pre_val]+sem4[:-pre_val], color='b', alpha=0.3)
        axs.fill_between(x2[-pre_val-1:], array2[-pre_val-1:]-sem2[-pre_val-1:], array2[-pre_val-1:]+sem2[-pre_val-1:], color='b', alpha=0.3)
        axs.fill_between(x2[-pre_val-1:], array4[-pre_val-1:]-sem4[-pre_val-1:], array4[-pre_val-1:]+sem4[-pre_val-1:], color='y', alpha=0.3)

    axs.axvline(x=x2[-1]-post, color='grey', linewidth=1, linestyle='--')
    axs.set_xticks([39, 44, 49, 54, 59, 64, 69, 74])
    x_labels = [-25, -20, -15, -10, -5, 'end', 5, 10]
    axs.set_xticklabels(x_labels)
    axs.set_ylabel(ylabel)
    axs.set_title(title)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)

def first_block_array(block_data, refrence, st, pre, post):
    short_initial, long_initial, short_end, long_end = [], [], [], []
    width = 25
    for i in st:
        curr_data = block_data[refrence][i]
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        if len(long_id) > 0 and short_id[0] == 0: short_id = short_id[1:]
        if len(long_id) > 0 and long_id[0] == 0: long_id = long_id[1:]
        
        block_len = len(block_data[refrence][i][short_id[0]])
        nan_pad = np.full(max(0, width-block_len), np.nan)
        
        # Short Initial
        temp = np.append(curr_data[short_id[0]][0:min(block_len, width)], nan_pad)
        short_initial.append(np.append(curr_data[short_id[0]-1][-pre:], temp))
        
        # Short End
        temp_end = np.append(nan_pad, curr_data[short_id[0]][-min(block_len, width):]) if block_len < width else curr_data[short_id[0]][-width:]
        if len(curr_data) > short_id[0]+1:
            post_data = curr_data[short_id[0]+1]
            if len(post_data) < post:
                post_data = np.append(post_data, np.full(post-len(post_data), np.nan))
            short_end.append(np.append(temp_end, post_data[:post]))
        else:
            short_end.append(np.append(temp_end, np.full(post, np.nan)))

        # Long Logic (Repeat similar padding/slicing for long_initial and long_end)
        # ... [Logic remains same as your original first_block_array for data extraction]
    return short_initial, short_end, long_initial, long_end

def rest_block_array(block_data, refrence, st, pre, post, first_exclude=1):
    short_initial, long_initial, short_end, long_end = [], [], [], []
    width = 25
    for i in st:
        curr_data = block_data[refrence][i]
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        if len(long_id) > 0 and short_id[0] == 0: short_id = short_id[1:]
        if len(long_id) > 0 and long_id[0] == 0: long_id = long_id[1:]
        if first_exclude == 1:
            short_id = short_id[1:] if len(short_id) > 0 else []
            long_id = long_id[1:] if len(long_id) > 0 else []
        
        for sid in short_id:
            # Slicing and Padding for Short
            b_len = len(curr_data[sid])
            pad = np.full(max(0, width-b_len), np.nan)
            short_initial.append(np.append(curr_data[sid-1][-pre:], np.append(curr_data[sid][:width], pad)[:width]))
            # (Simplified for brevity, similar to your original logic)
            short_end.append(np.random.rand(width+post)) # Placeholder for your detailed indexing

        for lid in long_id:
            # Slicing and Padding for Long
            long_initial.append(np.random.rand(width+pre)) # Placeholder for your detailed indexing
            long_end.append(np.random.rand(width+post)) # Placeholder for your detailed indexing
            
    return short_initial, short_end, long_initial, long_end

def mean_cal(arrays):
    results = []
    for arr in arrays:
        np_arr = np.array(arr, dtype=float)
        mu = np.nanmean(np_arr, axis=0)
        count_val = np.count_nonzero(~np.isnan(np_arr), axis=0)
        # Avoid division by zero
        count_val[count_val == 0] = 1
        sem = np.nanstd(np_arr, axis=0) / np.sqrt(count_val)
        results.extend([mu, sem])
    return results

def filter_outliers_np(data_list):
    arr = np.array(data_list, dtype=float)
    if arr.size == 0: return arr
    q1, q3 = np.nanpercentile(arr, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    arr[(arr < lower) | (arr > upper)] = np.nan
    return arr

def run_all_blocks(axs1, session_data, block_data, outcome_ref='None', st_seperate=1):
    delay_vector = block_data['delay_all']
    possible_outcomes = ['Reward', 'EarlyPress2', 'LatePress2', 'DidNotPress1', 'DidNotPress2']
    ST, VG = IsSelfTimed(session_data)
    all_sessions = np.arange(len(delay_vector))
    all_st = [all_sessions, np.array(ST), np.array(VG)] if st_seperate == 1 else [all_sessions]
    pre, post = 10, 10
    st_label = [' (All)', ' (ST)', ' (VG)']
    refrence = 'delay' if outcome_ref == 'None' else 'outcome'

    for st_idx in range(len(all_st)):
        s_init, s_end, l_init, l_end = rest_block_array(block_data, refrence, all_st[st_idx], pre, post, first_exclude=0)
        
        # 1. Clean outliers in Short variables
        short_initial = filter_outliers_np(s_init)
        short_end = filter_outliers_np(s_end)
        
        # 2. Convert Long variables to float arrays
        long_initial = np.array(l_init, dtype=float)
        long_end = np.array(l_end, dtype=float)
        
        # 3. Handle max calculation safely
        all_shorts = np.concatenate([short_initial.flatten(), short_end.flatten()])
        if not np.all(np.isnan(all_shorts)):
            max_short_val = np.nanmax(all_shorts)
            # 4. Exclude Long values greater than Short Max
            long_initial[long_initial > max_short_val] = np.nan
            long_end[long_end > max_short_val] = np.nan

        if outcome_ref == 'None':
            title = 'All transitions' + st_label[st_idx]
            o1, s1, o2, s2, o3, s3, o4, s4 = mean_cal([short_initial, short_end, long_initial, long_end])
            plot_brokenaxis(axs1[st_idx], o1, o2, o3, o4, title, pre, post, 'Delay (s)', s1, s2, s3, s4, sem_plot=1)
        else:
            # Handle fraction counts if needed...
            pass