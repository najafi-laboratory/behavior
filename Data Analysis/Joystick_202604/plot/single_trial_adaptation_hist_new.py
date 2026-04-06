# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 15:23:44 2025

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
from matplotlib.patches import FancyArrowPatch
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

def create_label(delay_vector , outcomes):
    label = copy.deepcopy(outcomes)
    for i in range(len(outcomes)):
        outcome = outcomes[i]
        delay_data = delay_vector[i][~np.isnan(np.array(delay_vector[i]))]
        condition_early = np.percentile(delay_data, 30)
        condition_late = np.percentile(delay_data, 70)
        #print(condition)
        for j in range(len(outcome)):
            if not np.isnan(delay_vector[i][j]):
                if delay_vector[i][j] <= condition_early:
                    label[i][j] = 'Short'
                elif delay_vector[i][j] <= condition_late:
                    label[i][j] = 'Long'
            if not label[i][j] in ['Short' , 'Long' , 'DidNotPress1' , 'DidNotPress2']:
                label[i][j] = 'Other'
    return label


def devide(ref_array, condition, curr_array, raw_data):
    index = []
    values = []
    trial_type = []
    for i in range(len(ref_array)):
        #temp_index = np.where(ref_array[i] == condition)[0]
        temp_index = [t for t, s in enumerate(ref_array[i]) if s == condition]
        #print(np.where(ref_array[i] == condition))
        #print(temp_index)
        temp_values = []
        curr_trial_type = raw_data[i]['TrialTypes']
        temp_trial_type = []
        if len(temp_index) > 0:
            temp_values = [curr_array[i][j] for j in temp_index]
            temp_trial_type = [curr_trial_type[j] for j in temp_index]
        index.append(temp_index)
        values.append(temp_values)
        trial_type.append(temp_trial_type)
    return index, values, trial_type

def plot_dist(axs, delay_data, color_tag, alpha = 0.2):
    delay_data = delay_data[~np.isnan(delay_data)]
    print('all trials count: ',len(delay_data))
    #mid = np.percentile(delay_data, 50)
    lim = np.percentile(delay_data, 97)
    delay_data = delay_data[delay_data<lim]
    percentile_90 = np.percentile(delay_data, 90)
    bin_initial = np.linspace(np.nanmin(delay_data), percentile_90, 30)
    bin_end = np.linspace(percentile_90, np.nanmax(delay_data), 40)
    hist , bin_edge = np.histogram(delay_data, bins=np.append(bin_initial, bin_end))
    bin_center = (bin_edge[1:]+bin_edge[:-1])/2
    hist = hist/len(delay_data)
    if alpha == 1:
        plt.plot(bin_center, hist, color = color_tag, alpha = alpha, label = 'n = ' + str(len(delay_data)))
    else:
        plt.plot(bin_center, hist, color = color_tag, alpha = alpha)
    plt.axvline(x = np.percentile(delay_data, 30), color = color_tag,alpha = alpha, linewidth = 1, linestyle='--')
    plt.axvline(x = np.percentile(delay_data, 70), color = color_tag,alpha = alpha, linewidth = 1, linestyle=':')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Delay (s)')
    axs.set_ylabel('Probability')
    axs.legend()

def next_trial(ref_array, index, trial_type, st, seprate = 0):
    output = []
    #print(index[0])
    for i in st:
        for j in range(len(index[i])):
            if seprate == 0:
                #print(len(ref_array) , i)
                if index[i][j]+1 < len(ref_array[i]):
                    output.append(ref_array[i][index[i][j]+1])
            elif seprate == 1:
                if trial_type[i][j] == seprate:
                    if index[i][j]+1 < len(ref_array[i]):
                        output.append(ref_array[i][index[i][j]+1])
            else:
                if trial_type[i][j] == seprate:
                    if index[i][j]+1 < len(ref_array[i]):
                        output.append(ref_array[i][index[i][j]+1])
    return output

def current_trial(ref_array, index, trial_type, st, seprate = 0):
    output = []
    for i in st:
        for j in range(len(index[i])):
            if seprate == 0:
                output.append(ref_array[i][index[i][j]])
            elif seprate == 1:
                if trial_type[i][j] == seprate:
                    output.append(ref_array[i][index[i][j]])
            else:
                if trial_type[i][j] == seprate:
                    output.append(ref_array[i][index[i][j]])
    return output

def count_label(all_label , ref_array):
    all_counts = np.zeros(len(all_label))
    num = 0
    for label in all_label:
        all_counts[num] = np.sum(np.array(ref_array) == label) / len(ref_array)
        num = num + 1
    return all_counts

def data_spliter(delay_data, data_name, selected_sessions):
    short_data = []
    long_data = []
    
    for session in selected_sessions:
        data_init = delay_data[data_name][session]
        short_data_init = [np.array(data_init[i]) for i in delay_data['short_id'][session]]
        if len(short_data_init) > 0:
            short_data_init = np.concatenate(short_data_init)
        long_data_init = [np.array(data_init[i]) for i in delay_data['long_id'][session]]
        if len(long_data_init) > 0:
            long_data_init = np.concatenate(long_data_init)
        
        short_data.append(short_data_init)
        long_data.append(long_data_init)
        
    return short_data, long_data

def average_boundries(delay_data, selected_sessions):
    short_low, long_low = data_spliter(delay_data, 'LowerBand', selected_sessions)
    short_low_avg = [np.nanmean(i) for i in short_low]
    long_low_avg = [np.nanmean(i) for i in long_low]
    short_up, long_up = data_spliter(delay_data, 'UpperBand', selected_sessions)
    short_up_avg = [np.nanmean(i) for i in short_up]
    long_up_avg = [np.nanmean(i) for i in long_up]
    
    return short_low_avg, short_up_avg, long_low_avg, long_up_avg

def find_bin(lower, upper, delay_all, bin_size = 0.1):
    max_val = np.nanmax(np.concatenate(delay_all))
    lower_mean = np.nanmean(lower)
    upper_mean = np.nanmean(upper)
    bin1 = np.linspace(0, lower_mean, int(lower_mean//bin_size + 1))
    bin2 = np.linspace(lower_mean, upper_mean, int((upper_mean-lower_mean)//bin_size + 1))
    bin3 = np.linspace(upper_mean, max_val, int((max_val-upper_mean)//bin_size + 1))
    bins = [bin1, bin2[1:], bin3[1:]]
    bins = np.concatenate(bins)
    return bins
    
def run_dist(axs, session_data, delay_data, st):
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
    
    delay = [delay_temp[i] for i in selected_sessions]
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=len(delay)+1)
    for i in range(len(delay)):
        color_tag = cmap(norm(len(delay)+1-i))
        #plot_dist(axs, np.array(delay[i]), color_tag)
    delay_all = [item for sublist in delay for item in sublist]
    #print(np.array(delay_all))
    plot_dist(axs, np.array(delay_all), 'k', alpha = 1)
    axs.set_title('Distribution of Inter Press Interval '+ st_title[st])

def plot_arrow(axs, delays, next_delays, outcome_array, opto, session_opto_type, bins_all, short_low_avg, short_up_avg):
    outcome_color = ['blue', 'pink', 'green']
    possible_outcome = ['EarlyPress2', 'LatePress2','Reward']
    possible_opto = ['EarlyPressITI', 'LatePressITI','RewardITI']
    delays_all = np.concatenate(delays)
    next_delays_all = np.concatenate(next_delays)
    outcome_all = np.concatenate(outcome_array)
    opto_effective_tag = []
    for i in range(len(opto)):
        temp = np.zeros(len(opto[i]))
        if 'EarlyPressITI' in session_opto_type[i]:
            indx1 = np.where(outcome_array[i] == 'EarlyPress2')[0]
            indx2 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1) & set(indx2))
            temp[indx] = 1
        if 'LatePressITI' in session_opto_type[i]:
            indx1 = np.where(outcome_array[i] == 'LatePress2')[0]
            indx2 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1) & set(indx2))
            temp[indx] = 1
        if 'RewardITI' in session_opto_type[i]:
            indx1 = np.where(outcome_array[i] == 'Reward')[0]
            indx2 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1) & set(indx2))
            temp[indx] = 1
        if 'trial' in session_opto_type[i]:
            indx1 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1))
            temp[indx] = 1
        opto_effective_tag.append(temp)
    opto_all = np.concatenate(opto_effective_tag)
    labels = np.digitize(delays_all, bins_all)
    for bins in range(len(bins_all)):
        if bins == len(bins_all) -1:
            end_bin = bins_all[-1]+0.05
        else:
            end_bin = bins_all[bins+1]
        
        axs.axvline(x=bins_all[bins], color='gray', linewidth=1, linestyle='--', alpha = 0.5)
        for outcome in [0,1,2]:
            #print(outcome)
            indx1 = np.where(labels == bins+1)[0]
            indx2 = np.where(outcome_all == possible_outcome[outcome])[0]
            indx3 = np.where(opto_all == 1)[0]
            indx4 = np.where(opto_all == 0)[0]
            indx_opto = list(set(indx1) & set(indx2) & set(indx3))
            indx_control = list(set(indx1) & set(indx2) & set(indx4))
            if len(indx_control)>0:
                next_temp = next_delays_all[indx_control]
                x_tr = (bins_all[bins]+end_bin)/2
                x_tr = np.nanmean(delays_all[indx_control])
                arrow = FancyArrowPatch((x_tr, len(indx_control)/len(labels)), (np.nanmean(next_temp), np.sum(~np.isnan(next_temp))/len(labels)),
                                        connectionstyle="arc3,rad=.2",  # curvature
                                        arrowstyle='->',                # arrow style
                                        mutation_scale=3,              # size of the arrowhead
                                        linewidth=0.5,
                                        color=outcome_color[outcome])
                axs.add_patch(arrow)
            if len(indx_opto)>0:
                #print('opto')
                next_temp = next_delays_all[indx_opto]
                x_tr = (bins_all[bins]+end_bin)/2
                x_tr = np.nanmean(delays_all[indx_opto])
                arrow = FancyArrowPatch((x_tr, len(indx_opto)/len(labels)), (np.nanmean(next_temp), np.sum(~np.isnan(next_temp))/len(labels)),
                                        connectionstyle="arc3,rad=.2",  # curvature
                                        arrowstyle='->',                # arrow style
                                        mutation_scale=3,              # size of the arrowhead
                                        linewidth=0.5,
                                        linestyle='--',
                                        color=outcome_color[outcome])

                axs.add_patch(arrow)
    #axs.fill_betweenx(y=[-1, 1], x1=np.nanmin(short_low_avg), x2=np.nanmax(short_low_avg), color='y', alpha=0.2)
    axs.fill_betweenx(y=[-1, 1], x1=np.nanmax(short_low_avg), x2=np.nanmean(short_up_avg), color='g', alpha=0.1)
    #axs.fill_betweenx(y=[-1, 1], x1=np.nanmin(short_up_avg), x2=np.nanmax(short_up_avg), color='y', alpha=0.2)
    axs.set_xlim(0, 5)
    axs.set_ylim(-0.01, 0.5)
    plot_layout(axs, 'Delay adaptation')
    axs.set_xlabel('Delay (s)')
    axs.set_ylabel('Probability')

def count_outcome(outcome_array_selected, outcome_labels):
    count_array = []
    for i in range(len(outcome_labels)):
        indx = np.where(outcome_array_selected == outcome_labels[i])[0]
        count_array.append(len(indx)/len(outcome_array_selected))
    return(np.array(count_array))

def plot_scatter_outcome(axs, count_list, outcome_labels, shape, color_tag):
    #print(len(count_list), len(count_list[0]), len(count_list[1]), len(count_list[2]))
    count_array = np.array(count_list)
    mean_all = np.nanmean(count_array, axis = 0)
    std_all = np.nanstd(count_array, axis = 0)/count_array.shape[0]
    axs.scatter(np.arange(len(outcome_labels)), mean_all, color=color_tag,s=10, marker = shape)
    axs.errorbar(np.arange(len(outcome_labels)), mean_all, yerr=std_all, fmt='none', ecolor=color_tag, elinewidth=1)
    axs.set_xlabel('outcome')
    axs.set_ylabel('Outcome percentages')
    axs.set_xticks(np.arange(len(outcome_labels)))
    axs.set_xticklabels(outcome_labels, rotation='vertical')

def plot_outcome(axs, outcome_array, outcome_next, opto, session_opto_type):
    outcome_color = ['blue', 'pink', 'green']
    possible_outcome = ['EarlyPress2', 'LatePress2','Reward']
    possible_opto = ['EarlyPressITI', 'LatePressITI','RewardITI']
    outcome_labels = ['Reward', 'EarlyPress2', 'LatePress2', 'LatePress1', 'DidNotPress1', 'DidNotPress2']
    control_array = []
    opto_array = []
    opto_pos_array = []
    control_rew = []
    opto_rew = []
    control_ep = []
    opto_ep = []
    control_lp = []
    opto_lp = []
    opto_effective_tag = []
    nan_array = np.zeros(len(outcome_labels))
    nan_array[:] = np.nan
    for i in range(len(opto)):
        temp = np.zeros(len(opto[i]))
        if 'EarlyPressITI' in session_opto_type[i]:
            indx1 = np.where(outcome_array[i] == 'EarlyPress2')[0]
            indx2 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1) & set(indx2))
            temp[indx] = 1
        if 'LatePressITI' in session_opto_type[i]:
            indx1 = np.where(outcome_array[i] == 'LatePress2')[0]
            indx2 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1) & set(indx2))
            temp[indx] = 1
        if 'RewardITI' in session_opto_type[i]:
            indx1 = np.where(outcome_array[i] == 'Reward')[0]
            indx2 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1) & set(indx2))
            temp[indx] = 1
        if 'trial' in session_opto_type[i]:
            indx1 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1))
            temp[indx] = 1
        opto_effective_tag.append(temp)
        
    for i in range(len(outcome_array)):
        control_indx = list(set(np.where(np.logical_and(opto_effective_tag[i] == 0, np.roll(opto_effective_tag[i],1) == 0))[0]))
        opto_indx = list(set(np.where(opto_effective_tag[i] == 1)[0]))
        opto_pos_index = list(set(np.where(np.roll(opto_effective_tag[i],1) == 1)[0]))
        if len(control_indx)>0:
            control_array.append(count_outcome(outcome_array[i][control_indx], outcome_labels))
        else:
            control_array.append(nan_array)
        if len(opto_indx)>0:
            opto_array.append(count_outcome(outcome_array[i][opto_indx], outcome_labels))
        else:
            opto_array.append(nan_array)
        if len(opto_pos_index)>0:
            opto_pos_array.append(count_outcome(outcome_array[i][opto_pos_index], outcome_labels))
        else:
            opto_pos_array.append(nan_array)
            
        control_indx = np.where(np.logical_and(opto_effective_tag[i] == 0, np.roll(opto_effective_tag[i],1) == 0))[0]
        opto_indx = np.where(np.roll(opto_effective_tag[i],1) == 1)[0]
        #opto_pos_index = np.where(opto_effective_tag[max(i-1, 0)] == 1)[0]
        
        rew_indx = np.where(outcome_array[i] == 'Reward')[0]
        ep_indx = np.where(outcome_array[i] == 'EarlyPress2')[0]
        lp_indx = np.where(outcome_array[i] == 'LatePress2')[0]
        
        control_rew_indx = list(set(control_indx) & set(rew_indx))
        control_ep_indx = list(set(control_indx) & set(ep_indx))
        control_lp_indx= list(set(control_indx) & set(lp_indx))
        opto_rew_indx = list(set(opto_indx) & set(rew_indx))
        opto_ep_indx = list(set(opto_indx) & set(ep_indx))
        opto_lp_indx = list(set(opto_indx) & set(lp_indx))
        if len(control_rew_indx)>0:
            control_rew.append(count_outcome(outcome_next[i][control_rew_indx], outcome_labels))
            #print(control_rew)
        else:
            control_rew.append(nan_array)
        if len(opto_rew_indx)>0:
            opto_rew.append(count_outcome(outcome_next[i][opto_rew_indx], outcome_labels))
        else:
            opto_rew.append(nan_array)
            
        if len(control_ep_indx)>0:
            control_ep.append(count_outcome(outcome_next[i][control_ep_indx], outcome_labels))
        else:
            control_ep.append(nan_array)
        if len(opto_ep_indx)>0:
            opto_ep.append(count_outcome(outcome_next[i][opto_ep_indx], outcome_labels))
        else:
            opto_ep.append(nan_array)
            
        if len(control_lp_indx)>0:
            control_lp.append(count_outcome(outcome_next[i][control_lp_indx], outcome_labels))
        else:
            control_lp.append(nan_array)
        if len(opto_lp_indx)>0:
            opto_lp.append(count_outcome(outcome_next[i][opto_lp_indx], outcome_labels))
        else:
            opto_lp.append(nan_array)
    plot_scatter_outcome(axs[0], control_array, outcome_labels, 'o', 'k')
    plot_scatter_outcome(axs[0], opto_array, outcome_labels, '^', 'dodgerblue')
    plot_scatter_outcome(axs[0], opto_pos_array, outcome_labels, '^', 'k')
    plot_scatter_outcome(axs[1], control_rew, outcome_labels, 'o', 'k')
    plot_scatter_outcome(axs[1], opto_rew, outcome_labels, '^', 'dodgerblue')
    plot_scatter_outcome(axs[2], control_ep, outcome_labels, 'o', 'k')
    plot_scatter_outcome(axs[2], opto_ep, outcome_labels, '^', 'dodgerblue')
    plot_scatter_outcome(axs[3], control_lp, outcome_labels, 'o', 'k')
    plot_scatter_outcome(axs[3], opto_lp, outcome_labels, '^', 'dodgerblue')
    plot_layout(axs[0], 'outcome for tr opto vs control vs tr+1 opto')
    plot_layout(axs[1], 'outcome for reward tr+1 opto vs tr+1 control')
    plot_layout(axs[2], 'outcome for earlypress tr+1 opto vs tr+1 control')
    plot_layout(axs[3], 'outcome for latepress tr+1 opto vs tr+1 control')
    
    
def plot_trajectory(axs, outcome_array, outcome_next, opto, session_opto_type):
    outcome_color = ['blue', 'pink', 'green']
    possible_outcome = ['EarlyPress2', 'LatePress2','Reward']
    possible_opto = ['EarlyPressITI', 'LatePressITI','RewardITI']
    outcome_labels = ['Reward', 'EarlyPress2', 'LatePress2', 'LatePress1', 'DidNotPress1', 'DidNotPress2']
    control_array = []
    opto_array = []
    opto_pos_array = []
    control_rew = []
    opto_rew = []
    control_ep = []
    opto_ep = []
    control_lp = []
    opto_lp = []
    opto_effective_tag = []
    nan_array = np.zeros(len(outcome_labels))
    nan_array[:] = np.nan
    for i in range(len(opto)):
        temp = np.zeros(len(opto[i]))
        if 'EarlyPressITI' in session_opto_type[i]:
            indx1 = np.where(outcome_array[i] == 'EarlyPress2')[0]
            indx2 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1) & set(indx2))
            temp[indx] = 1
        if 'LatePressITI' in session_opto_type[i]:
            indx1 = np.where(outcome_array[i] == 'LatePress2')[0]
            indx2 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1) & set(indx2))
            temp[indx] = 1
        if 'RewardITI' in session_opto_type[i]:
            indx1 = np.where(outcome_array[i] == 'Reward')[0]
            indx2 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1) & set(indx2))
            temp[indx] = 1
        if 'trial' in session_opto_type[i]:
            indx1 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1))
            temp[indx] = 1
        opto_effective_tag.append(temp)
        
    for i in range(len(outcome_array)):
        control_indx = list(set(np.where(np.logical_and(opto_effective_tag[i] == 0, np.roll(opto_effective_tag[i],1) == 0))[0]))
        opto_indx = list(set(np.where(opto_effective_tag[i] == 1)[0]))
        opto_pos_index = list(set(np.where(np.roll(opto_effective_tag[i],1) == 1)[0]))
        if len(control_indx)>0:
            control_array.append(count_outcome(outcome_array[i][control_indx], outcome_labels))
        else:
            control_array.append(nan_array)
        if len(opto_indx)>0:
            opto_array.append(count_outcome(outcome_array[i][opto_indx], outcome_labels))
        else:
            opto_array.append(nan_array)
        if len(opto_pos_index)>0:
            opto_pos_array.append(count_outcome(outcome_array[i][opto_pos_index], outcome_labels))
        else:
            opto_pos_array.append(nan_array)
            
        control_indx = np.where(np.logical_and(opto_effective_tag[i] == 0, np.roll(opto_effective_tag[i],1) == 0))[0]
        opto_indx = np.where(np.roll(opto_effective_tag[i],1) == 1)[0]
        #opto_pos_index = np.where(opto_effective_tag[max(i-1, 0)] == 1)[0]
        
        rew_indx = np.where(outcome_array[i] == 'Reward')[0]
        ep_indx = np.where(outcome_array[i] == 'EarlyPress2')[0]
        lp_indx = np.where(outcome_array[i] == 'LatePress2')[0]
        
        control_rew_indx = list(set(control_indx) & set(rew_indx))
        control_ep_indx = list(set(control_indx) & set(ep_indx))
        control_lp_indx= list(set(control_indx) & set(lp_indx))
        opto_rew_indx = list(set(opto_indx) & set(rew_indx))
        opto_ep_indx = list(set(opto_indx) & set(ep_indx))
        opto_lp_indx = list(set(opto_indx) & set(lp_indx))
        if len(control_rew_indx)>0:
            control_rew.append(count_outcome(outcome_next[i][control_rew_indx], outcome_labels))
            #print(control_rew)
        else:
            control_rew.append(nan_array)
        if len(opto_rew_indx)>0:
            opto_rew.append(count_outcome(outcome_next[i][opto_rew_indx], outcome_labels))
        else:
            opto_rew.append(nan_array)
            
        if len(control_ep_indx)>0:
            control_ep.append(count_outcome(outcome_next[i][control_ep_indx], outcome_labels))
        else:
            control_ep.append(nan_array)
        if len(opto_ep_indx)>0:
            opto_ep.append(count_outcome(outcome_next[i][opto_ep_indx], outcome_labels))
        else:
            opto_ep.append(nan_array)
            
        if len(control_lp_indx)>0:
            control_lp.append(count_outcome(outcome_next[i][control_lp_indx], outcome_labels))
        else:
            control_lp.append(nan_array)
        if len(opto_lp_indx)>0:
            opto_lp.append(count_outcome(outcome_next[i][opto_lp_indx], outcome_labels))
        else:
            opto_lp.append(nan_array)
    plot_scatter_outcome(axs[0], control_array, outcome_labels, 'o', 'k')
    plot_scatter_outcome(axs[0], opto_array, outcome_labels, '^', 'dodgerblue')
    plot_scatter_outcome(axs[0], opto_pos_array, outcome_labels, '^', 'k')
    plot_scatter_outcome(axs[1], control_rew, outcome_labels, 'o', 'k')
    plot_scatter_outcome(axs[1], opto_rew, outcome_labels, '^', 'dodgerblue')
    plot_scatter_outcome(axs[2], control_ep, outcome_labels, 'o', 'k')
    plot_scatter_outcome(axs[2], opto_ep, outcome_labels, '^', 'dodgerblue')
    plot_scatter_outcome(axs[3], control_lp, outcome_labels, 'o', 'k')
    plot_scatter_outcome(axs[3], opto_lp, outcome_labels, '^', 'dodgerblue')
    plot_layout(axs[0], 'outcome for tr opto vs control vs tr+1 opto')
    plot_layout(axs[1], 'outcome for reward tr+1 opto vs tr+1 control')
    plot_layout(axs[2], 'outcome for earlypress tr+1 opto vs tr+1 control')
    plot_layout(axs[3], 'outcome for latepress tr+1 opto vs tr+1 control')
    
def plot_scatter(axs, delays, next_delays, outcome_array, opto, session_opto_type, bins_all, short_low_avg, short_up_avg, legend):
    outcome_color = ['blue', 'pink', 'green']
    possible_outcome = ['EarlyPress2', 'LatePress2','Reward']
    possible_opto = ['EarlyPressITI', 'LatePressITI','RewardITI']
    delays_all = np.concatenate(delays)
    next_delays_all = np.concatenate(next_delays)
    outcome_all = np.concatenate(outcome_array)
    opto_effective_tag = []
    for i in range(len(opto)):
        temp = np.zeros(len(opto[i]))
        #print(session_opto_type[i])
        if 'EarlyPressITI' in session_opto_type[i]:
            indx1 = np.where(outcome_array[i] == 'EarlyPress2')[0]
            indx2 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1) & set(indx2))
            temp[indx] = 1
        if 'LatePressITI' in session_opto_type[i]:
            indx1 = np.where(outcome_array[i] == 'LatePress2')[0]
            indx2 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1) & set(indx2))
            temp[indx] = 1
        if 'RewardITI' in session_opto_type[i]:
            indx1 = np.where(outcome_array[i] == 'Reward')[0]
            indx2 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1) & set(indx2))
            temp[indx] = 1
        if 'trial' in session_opto_type[i]:
            indx1 = np.where(opto[i] == 1)[0]
            indx = list(set(indx1))
            temp[indx] = 1
        opto_effective_tag.append(temp)
    opto_all = np.concatenate(opto_effective_tag)
    labels = np.digitize(delays_all, bins_all)
    size_weight = 400
    for bins in range(len(bins_all)):
        if bins == len(bins_all) -1:
            end_bin = bins_all[-1]+0.05
        else:
            end_bin = bins_all[bins+1]
        
        for outcome in [0,1,2]:
            indx1 = np.where(labels == bins+1)[0]
            indx2 = np.where(outcome_all == possible_outcome[outcome])[0]
            indx3 = np.where(opto_all == 1)[0]
            indx4 = np.where(opto_all == 0)[0]
            indx_opto = list(set(indx1) & set(indx2) & set(indx3))
            indx_control = list(set(indx1) & set(indx2) & set(indx4))
            if len(indx_control)>0:
                tr1_delay = np.nanmean(next_delays_all[indx_control])
                tr_delay = np.nanmean(delays_all[indx_control])
                tr_num = len(indx_control)/len(labels)
                tr1_num = np.sum(~np.isnan(next_delays_all[indx_control]))/len(labels)
                
                axs.scatter(tr_delay, tr1_delay, s = tr_num*size_weight, color = outcome_color[outcome], alpha = 0.5, label = str(np.sum(~np.isnan(next_delays_all[indx_control]))))
                axs.scatter(tr_delay, tr1_delay, s = tr1_num*size_weight, color = outcome_color[outcome], label = str(len(indx_control)))
                
            if len(indx_opto)>0:
                tr1_delay = np.nanmean(next_delays_all[indx_opto])
                tr_delay = np.nanmean(delays_all[indx_opto])
                tr_num = len(indx_opto)/len(labels)
                tr1_num = np.sum(~np.isnan(next_delays_all[indx_opto]))/len(labels)
                
                axs.scatter(tr_delay, tr1_delay, s = tr_num*size_weight, color = outcome_color[outcome], alpha = 0.5, marker = '^', label = str(np.sum(~np.isnan(next_delays_all[indx_opto]))))
                axs.scatter(tr_delay, tr1_delay, s = tr1_num*size_weight, color = outcome_color[outcome], marker = '^', label = str(len(indx_opto)))
    axs.set_xlim([0,3.5])            
    lims = [
    np.min([axs.set_xlim(), axs.set_ylim()]),  # min of both axes
    np.max([axs.set_xlim(), axs.set_ylim()]),  # max of both axes
    ]
    axs.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Unity line')  # dashed black line
    axs.fill_between(x=[0,3.5], y1=np.nanmax(short_low_avg), y2=np.nanmean(short_up_avg), color='g', alpha=0.1)
    plot_layout(axs, 'Delay adaptation')
    axs.set_xlabel('Delay tr (s)')
    axs.set_ylabel('Delay tr+1 (s)')   
    if legend:
        axs.legend()

def find_opto(session_data, selected_sessions):
    raw = session_data['raw']
    opot_type_all = []
    for i in selected_sessions:
        gui = raw[i]['TrialSettings'][5]['GUI']
        opto = gui['SessionType']
        if opto == 1:
            opto_type = []
            if 'OptoRewardITI' in gui.keys():
                if gui['OptoRewardITI'] == 1:
                    opto_type.append('RewardITI')
                if gui['OptoEarlyPressITI'] == 1:
                    opto_type.append('EarlyPressITI')
                if gui['OptoLatePressITI'] == 1:
                    opto_type.append('LatePressITI')
                if not (gui['OptoLatePressITI'] == 1 or gui['OptoEarlyPressITI'] == 1 or gui['OptoRewardITI'] == 1):
                    opto_type.append('trial')
            elif gui['OptoITI'] == 1:
                opto_type = ['EarlyPressITI', 'LatePressITI','RewardITI']
            else:
                opto_type.append('trial')
        else:
            opto_type = ['NonOpto']
        opot_type_all.append(opto_type)
    return opot_type_all
        
             
def run(axs, session_data, delay_data,bin_size_all, st, block_type, outcome_plot = 1, label = 0):
    # finding the v isually guided or self timed sessions
    num_session = len(session_data['IDs'])
    ST , VG = IsSelfTimed(session_data)
    if st == 1:
        selected_sessions = np.array(ST)
        num_session = len(ST)
    elif st == 2:
        selected_sessions = np.array(VG)
        num_session = len(VG)
    else:
        selected_sessions = np.arange(num_session)
    
    # creating a list of delays for each session (selected sessions) 
    # delay: num_selected_session each an array of num_trials
    delay_temp = delay_data['delay_all']
    delay_all = [delay_temp[i] for i in selected_sessions]
    session_opto_type = find_opto(session_data, selected_sessions) # finding each session is what opto setting type
     
    
    # find limits for short vs long press for each session a single value
    short_low_avg, short_up_avg, long_low_avg, long_up_avg = average_boundries(delay_data, selected_sessions)
    
    # split all data
    delay_short, delay_long = data_spliter(delay_data, 'delay', selected_sessions)
    delay_short_next, delay_long_next = data_spliter(delay_data, 'next_delay', selected_sessions)
    outcome_short_next, outcome_long_next = data_spliter(delay_data, 'next_outcome', selected_sessions)
    outcome_short, outcome_long = data_spliter(delay_data, 'outcome', selected_sessions)
    opto_short, opto_long = data_spliter(delay_data, 'OptoTag', selected_sessions)
    
    
    # mega session
    if block_type == 'short':
        bins_short = find_bin(short_low_avg, short_up_avg, delay_all, bin_size = bin_size_all[0])
        #plot_arrow(axs[0], delay_short, delay_short_next, outcome_short, opto_short, session_opto_type, bins_short, short_low_avg, short_up_avg)
        bins_short = find_bin(short_low_avg, short_up_avg, delay_all, bin_size = bin_size_all[1])
        #plot_scatter(axs[1], delay_short, delay_short_next, outcome_short, opto_short, session_opto_type, bins_short, short_low_avg, short_up_avg, label)
        if outcome_plot == 1:
            plot_outcome(axs, outcome_short, outcome_short_next, opto_short, session_opto_type)
    elif block_type == 'long':
        bins_long = find_bin(short_low_avg, short_up_avg, delay_all, bin_size = bin_size_all[0])
        #plot_arrow(axs[0], delay_long, delay_long_next, outcome_long, opto_long, session_opto_type, bins_long, long_low_avg, long_up_avg)
        bins_long = find_bin(short_low_avg, short_up_avg, delay_all, bin_size = bin_size_all[1])
        #plot_scatter(axs[1], delay_long, delay_long_next, outcome_long, opto_long, session_opto_type, bins_long, long_low_avg, long_up_avg, label)
        if outcome_plot == 1:
            plot_outcome(axs, outcome_long, outcome_long_next, opto_long, session_opto_type)
            