# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 10:13:23 2025

@author: saminnaji3
"""
import os
import fitz
import csv
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import copy
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

def plot_layout(axs, title):
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    #axs.set_xlabel('Delay (s)')
    axs.set_ylabel('Percentage')
    axs.set_title(title)
    #axs.legend()
    
    
def plot_dist(axs, delay_data, color_tag, xlabel = 'Delay', alpha = 0.2):
    delay_data = delay_data[~np.isnan(delay_data)]
    #print(len(delay_data))
    #mid = np.percentile(delay_data, 50)
    if len(delay_data) > 0:
        lim = np.percentile(delay_data, 97)
        delay_data = delay_data[delay_data<lim]
        percentile_90 = np.percentile(delay_data, 90)
        bin_initial = np.linspace(np.nanmin(delay_data), percentile_90, 30)
        bin_end = np.linspace(percentile_90, np.nanmax(delay_data), 40)
        hist , bin_edge = np.histogram(delay_data, bins=np.append(bin_initial, bin_end))
        bin_center = (bin_edge[1:]+bin_edge[:-1])/2
        hist = hist/len(delay_data)
        if alpha == 1:
            axs.plot(bin_center, hist, color = color_tag, alpha = alpha, label = 'n = ' + str(len(delay_data)))
        else:
            axs.plot(bin_center, hist, color = color_tag, alpha = alpha)
    #plt.axvline(x = np.percentile(delay_data, 30), color = color_tag,alpha = alpha, linewidth = 1, linestyle='--')
    #plt.axvline(x = np.percentile(delay_data, 70), color = color_tag,alpha = alpha, linewidth = 1, linestyle=':')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel(xlabel)
    axs.set_ylabel('Probability')
    axs.set_xlim([0, 2])
    axs.legend()
    
def create_data(session_data, block_data, interval_data, sessions):
    #num_sessions = len(block_data['delay_all'])
    session_raw = session_data['raw']
    vis1_push1 = []
    vis2_push2 = []
    push1_push2 = []
    retract1_init_push2 = []
    retract1_end_push2 = []
    vis1_push2 = []
    wait2_push2 = []
    delay = []
    next_delay = []
    block_opto = []
    opto_tag = []
    short_long = []
    
    for sess in sessions:
        vis1_push1.append(np.array([item for sublist in interval_data['vis1_push1'][sess] for item in sublist]))
        vis2_push2.append(np.array([item for sublist in interval_data['vis2_push2'][sess] for item in sublist]))
        push1_push2.append(np.array([item for sublist in interval_data['push1_push2'][sess] for item in sublist]))
        retract1_init_push2.append(np.array([item for sublist in interval_data['retract1_init_push2'][sess] for item in sublist]))
        retract1_end_push2.append(np.array([item for sublist in interval_data['retract1_end_push2'][sess] for item in sublist]))
        vis1_push2.append(np.array([item for sublist in interval_data['vis1_push2'][sess] for item in sublist]))
        wait2_push2.append(np.array([item for sublist in interval_data['wait2_push2'][sess] for item in sublist]))
        next_delay.append(np.array([item for sublist in block_data['next_delay'][sess] for item in sublist]))
        delay.append(np.array(block_data['delay_all'][sess][block_data['start'][sess][0]:]))
        block_opto.append(np.array(block_data['block_opto'][sess][block_data['start'][sess][0]:]))
        opto_tag.append(np.array([item for sublist in block_data['OptoTag'][sess] for item in sublist]))
        short_long.append(np.array(session_raw[sess]['TrialTypes'][block_data['start'][sess][0]:block_data['end'][sess][-1]]))
        
    all_labels = [block_opto, opto_tag, short_long]
    #all_labels = np.array(all_labels)
    all_data = [vis1_push1, vis2_push2, push1_push2, retract1_init_push2, retract1_end_push2, vis1_push2, wait2_push2, next_delay, delay]
    #all_data = np.array(all_data)
    all_data_name = ['vis1_push1', 'vis2_push2', 'push1_push2', 'retract1_init_push2', 'retract1_end_push2', 'vis1_push2', 'wait2_push2', 'next_delay', 'delay']
    return all_labels, all_data, all_data_name

def selected_trials_multi(data, labels, conditions, combine='and'):
    """
    Select trials matching multiple per-row conditions.

    Parameters
    ----------
    neu_seq : ndarray, shape (n_trials, n_neurons, n_timepoints)
    labels  : ndarray, shape (n_rows, n_trials)  # here n_rows should be 7 (rows 0..6)
    conditions : dict[int, Any]
        Mapping row_id -> condition. Each condition can be:
          - scalar (int/float): equality (==)
          - tuple (op, value) where op in {'==','!=','>','>=','<','<='}
          - tuple ('in', iterable_of_values)
          - callable(arr1d) -> boolean mask of length n_trials
    combine : {'and','or'}
        How to combine multiple row conditions across trials.

    Returns
    -------
    selected_neu_seq : ndarray, shape (n_selected, n_neurons, n_timepoints)
    selected_labels  : ndarray, shape (n_rows, n_selected)
    idx              : ndarray, shape (n_selected,)
        Indices of selected trials (useful for debugging).
    """
    n_rows, n_trials = labels.shape
    if combine not in ('and','or'):
        raise ValueError("combine must be 'and' or 'or'")

    mask = np.ones(n_trials, dtype=bool) if combine == 'and' else np.zeros(n_trials, dtype=bool)

    for row_id, cond in conditions.items():
        if not (0 <= row_id < n_rows):
            raise IndexError(f"row_id {row_id} out of bounds for labels with {n_rows} rows")
        row = labels[row_id, :]

        # build a mask for this single row condition
        if callable(cond):
            m = np.asarray(cond(row), dtype=bool)
            if m.shape != (n_trials,):
                raise ValueError(f"callable condition for row {row_id} must return shape ({n_trials},)")
        elif isinstance(cond, tuple):
            op = cond[0]
            if op in ('==','!=','>','>=','<','<='):
                val = cond[1]
                if op == '==': m = (row == val)
                elif op == '!=': m = (row != val)
                elif op ==  '>': m = (row  > val)
                elif op == '>=': m = (row >= val)
                elif op ==  '<': m = (row  < val)
                else:            m = (row <= val)
            elif op == 'in':
                vals = np.array(list(cond[1]))
                m = np.isin(row, vals)
            elif op == 'between':
                # usage: ('between', low, high, True)  # inclusive (default True)
                low, high = cond[1], cond[2]
                inclusive = True if len(cond) < 4 else bool(cond[3])
                if inclusive:
                    m = (row >= low) & (row <= high)
                else:
                    m = (row >  low) & (row <  high)
            else:
                raise ValueError(f"unsupported operator: {op}")
        else:
            # scalar -> equality
            m = (row == cond)

        mask = (mask & m) if combine == 'and' else (mask | m)

    idx = np.where(mask)[0]
    selected_data = data[:, idx]
    selected_labels  = labels[:, idx]
    return selected_data, selected_labels, idx
    
def concat_all(all_labels, all_data):
    for i in range(len(all_labels)):
        all_labels[i] = np.concatenate(all_labels[i])
        #print(all_labels[i].shape)
    all_labels = np.array(all_labels)
    for i in range(len(all_data)):
        all_data[i] = np.concatenate(all_data[i])
        #print(all_data[i].shape)
    all_data = np.array(all_data)
    return all_labels, all_data
    
def run_dist(axs1, axs2, axs3, axs4, session_data, delay_data, interval_data, plot_type, st):
    st_title = ['(All)' , '(Self Timed)' , '(Visually Guided)']
    delay_temp = delay_data['delay_all']
    num_session = len(delay_temp)
    ST , VG = IsSelfTimed(session_data)
    if st == 1:
        selected_sessions = np.array(ST)
        num_session = len(ST)
    elif st == 2:
        selected_sessions = np.array(VG)
        num_session = len(VG)
    else:
        selected_sessions = np.arange(num_session)
        
    all_labels, all_data, all_data_name = create_data(session_data, delay_data, interval_data, selected_sessions)
    
    #delay = [delay_temp[i] for i in selected_sessions]
    # cmap = plt.cm.inferno
    # norm = plt.Normalize(vmin=0, vmax=len(delay)+1)
    # for i in range(len(delay)):
    #     color_tag = cmap(norm(len(delay)+1-i))
    #     plot_dist(axs, np.array(delay[i]), color_tag)
    
    all_labels, all_data = concat_all(all_labels, all_data)
    #delay_all = [item for sublist in delay for item in sublist]
    #print(np.array(delay_all))
    if plot_type == 'single_opto':
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {1: 0}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs1[plotting], selected_data[plotting], 'k', xlabel = all_data_name[plotting],alpha = 1)
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {1: 1}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs1[plotting], selected_data[plotting], 'dodgerblue', xlabel = all_data_name[plotting],alpha = 1)
        axs1[0].set_title('control vs opto')
    elif plot_type == 'double':
        for plotting in range(len(all_data_name)):
            plot_dist(axs1[plotting], all_data[plotting], 'k', xlabel = all_data_name[plotting],alpha = 1)
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {2: 1}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs1[plotting], selected_data[plotting], 'y', xlabel = all_data_name[plotting],alpha = 1)
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {2: 2}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs1[plotting], selected_data[plotting], 'b', xlabel = all_data_name[plotting],alpha = 1)
        axs1[0].set_title('Short vs Long')
    elif plot_type == 'double_opto_block':
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {0:0, 2: 1}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs1[plotting], selected_data[plotting], 'k', xlabel = all_data_name[plotting],alpha = 1)
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {0:1, 2: 1}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs1[plotting], selected_data[plotting], 'dodgerblue', xlabel = all_data_name[plotting],alpha = 1)
        axs1[0].set_title('Short (control vs opto blocks)')
            
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {0:0, 2: 2}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs2[plotting], selected_data[plotting], 'k', xlabel = all_data_name[plotting],alpha = 1)
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {0:1, 2: 2}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs2[plotting], selected_data[plotting], 'dodgerblue', xlabel = all_data_name[plotting],alpha = 1)
        axs2[0].set_title('Long (control vs opto blocks)')
            
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {1:0, 2: 1}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs3[plotting], selected_data[plotting], 'k', xlabel = all_data_name[plotting],alpha = 1)
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {1:1, 2: 1}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs3[plotting], selected_data[plotting], 'dodgerblue', xlabel = all_data_name[plotting],alpha = 1)
        axs3[0].set_title('Short (control vs opto trials)')
        
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {1:0, 2: 2}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs4[plotting], selected_data[plotting], 'k', xlabel = all_data_name[plotting],alpha = 1)
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {1:1, 2: 2}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs4[plotting], selected_data[plotting], 'dodgerblue', xlabel = all_data_name[plotting],alpha = 1)
        axs4[0].set_title('Long (control vs opto trials)')
            
    elif plot_type == 'double_opto':
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {1:0, 2: 1}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs1[plotting], selected_data[plotting], 'k', xlabel = all_data_name[plotting],alpha = 1)
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {1:1, 2: 1}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs1[plotting], selected_data[plotting], 'dodgerblue', xlabel = all_data_name[plotting],alpha = 1)
        axs1[0].set_title('Short (control vs opto)')
        
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {1:0, 2: 2}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs2[plotting], selected_data[plotting], 'k', xlabel = all_data_name[plotting],alpha = 1)
        selected_data, selected_labels, idx = selected_trials_multi(all_data, all_labels, conditions = {1:1, 2: 2}, combine='and')
        for plotting in range(len(all_data_name)):
            plot_dist(axs2[plotting], selected_data[plotting], 'dodgerblue', xlabel = all_data_name[plotting],alpha = 1)
        axs2[0].set_title('Long (control vs opto)')
    
    