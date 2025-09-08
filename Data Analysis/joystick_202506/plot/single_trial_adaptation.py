# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:57:12 2025

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

def IsSelfTimed(session_data):
    isSelfTime = session_data['isSelfTimedMode']
    
    VG = []
    ST = []
    for i in range(0 , len(isSelfTime)):
        if isSelfTime[i][5] == 1 or isSelfTime[i][5] == np.nan:
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

def create_opto_label(opto , outcomes):
    label = copy.deepcopy(outcomes)
    for i in range(len(outcomes)):
        outcome = outcomes[i]
        opto_tag = opto[i]
        for j in range(len(outcome)):
            if outcome[j] == 'Reward':
                if opto_tag[j] == 1:
                    label[i][j] = 'Opto_Reward'
                else:
                    label[i][j] = 'Control_Reward'
            else:
                label[i][j] = 'Control'
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
    mid = np.percentile(delay_data, 50)
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
        plot_dist(axs, np.array(delay[i]), color_tag)
    delay_all = [item for sublist in delay for item in sublist]
    #print(np.array(delay_all))
    plot_dist(axs, np.array(delay_all), 'k', alpha = 1)
    axs.set_title('Distribution of Inter Press Interval '+ st_title[st])
    
def run_count(axs, axs1, session_data, delay_data, press_type):
    delay_vector = delay_data['delay_all']
    outcomes = copy.deepcopy(session_data['outcomes'])
    raw_data = session_data['raw']
    #outcome_array = ' '.join([item for sublist in outcomes for item in sublist])
    index_all = []
    trial_type_all = []
    for i in range(len(outcomes)):
        index_all.append(np.arange(len(outcomes[i])))
        trial_type_all.append(raw_data[i]['TrialTypes'])
    #print(session_data['outcomes'][0][0:3])
    labels = create_label(delay_vector , outcomes)
    all_labels = ['Short' , 'Long' , 'DidNotPress1' , 'DidNotPress2' ,'Other']
    label_color = ['dodgerblue' , 'gold', '#FFB74D','#FB8C00', 'grey']
    all_outcomes = ['Reward' , 'DidNotPress1' , 'DidNotPress2' , 'EarlyPress' , 'EarlyPress1' , 'EarlyPress2' ,'VisStimInterruptDetect1' , 'VisStimInterruptDetect2' , 'VisStimInterruptGray1' , 'VisStimInterruptGray2' , 'LatePress1' , 'LatePress2' , 'Other', 'Warmup', 'Assisted']
    outcome_color = ['#4CAF50','#FFB74D','#FB8C00','r','#64B5F6','#1976D2','#967bb6','#9932CC','#800080','#2E003E','pink','deeppink','cyan','grey', 'darkgray']
    #print(session_data['outcomes'][0][0:3])
    if press_type in all_labels:
        ref_array = copy.deepcopy(labels)
    else:
        ref_array = copy.deepcopy(outcomes)
    index , values , trial_type = devide(ref_array, press_type, delay_vector, raw_data)
    ST , VG = IsSelfTimed(session_data)
    all_sessions = np.arange(len(delay_vector))
    all_st = [all_sessions , np.array(ST) , np.array(VG)]
    x_values = [[0, 1 , 2] , [4 , 5 , 6] , [8 , 9 , 10]]
    width = 1
    for S_L in range(3):
        for st in range(3):
            count_array = next_trial(labels, index, trial_type, all_st[st], S_L)
            final_count = count_label(all_labels , count_array)
            bottom = np.cumsum(final_count)
            bottom[1:] = bottom[:-1]
            bottom[0] = 0
            for i in range(len(all_labels)):
                axs[0].bar(x_values[S_L][st], final_count[i], bottom=bottom[i],width=width,color=label_color[i],label=all_labels[i])
                
            count_array_outome = next_trial(session_data['outcomes'], index, trial_type, all_st[st], S_L)
            final_count_outcome = count_label(all_outcomes , count_array_outome)
            #print(set(count_array_outome))
            bottom = np.cumsum(final_count_outcome)
            bottom[1:] = bottom[:-1]
            bottom[0] = 0
            for i in range(len(all_outcomes)):
                axs[1].bar(x_values[S_L][st], final_count_outcome[i], bottom=bottom[i],width=width,color=outcome_color[i],label=all_outcomes[i])
            
            count_array_outome_all = current_trial(session_data['outcomes'], index_all, trial_type_all, all_st[st], S_L)
            final_count_outcome_all = count_label(all_outcomes , count_array_outome_all)
            bottom = np.cumsum(final_count_outcome_all)
            bottom[1:] = bottom[:-1]
            bottom[0] = 0
            for i in range(len(all_outcomes)):
                axs1.bar(x_values[S_L][st], final_count_outcome_all[i], bottom=bottom[i],width=width,color=outcome_color[i],label=all_outcomes[i])
            
    x_labels = ['All(All)' , 'All(ST)' , 'All(VG)' , 'Short blocks(All)' , 'Short blocks(ST)' , 'Short blocks(VG)' , 'Long blocks(All)' , 'Long blocks(ST)' , 'Long blocks(VG)']
    axs[0].set_xticks([0, 1 ,2 , 4, 5, 6, 8, 9, 10])
    axs[0].set_xticklabels(x_labels, rotation='vertical')
    
    title = 'Press Type for post ' + press_type + ' trial'
    plot_layout(axs[0], title)
    
    axs[1].set_xticks([0, 1 ,2 , 4, 5, 6, 8, 9, 10])
    axs[1].set_xticklabels(x_labels, rotation='vertical')
    title = 'Outcome for post ' + press_type + ' trial'
    plot_layout(axs[1], title)
    
    axs1.set_xticks([0, 1 ,2 , 4, 5, 6, 8, 9, 10])
    axs1.set_xticklabels(x_labels, rotation='vertical')
    title = 'Outcome All trials'
    plot_layout(axs1, title)
     
def run_delay(axs, session_data, delay_data, press_type):
    delay_vector = delay_data['delay_all']
    outcomes = copy.deepcopy(session_data['outcomes'])
    raw_data = session_data['raw']
    #print(session_data['outcomes'][0][0:3])
    labels = create_label(delay_vector , outcomes)
    all_labels = ['Short' , 'Long' , 'DidNotPress1' , 'DidNotPress2' ,'Other']
    
    if press_type in all_labels:
        ref_array = copy.deepcopy(labels)
    else:
        ref_array = copy.deepcopy(outcomes)
    index , values , trial_type = devide(ref_array, press_type, delay_vector, raw_data)
        
    ST , VG = IsSelfTimed(session_data)
    all_sessions = np.arange(len(delay_vector))
    all_st = [all_sessions , np.array(ST) , np.array(VG)]
    x_values_curr = np.array([[0, 1 , 2] , [4 , 5 , 6] , [8 , 9 , 10]])-0.5
    x_values_next = np.array([[0, 1 , 2] , [4 , 5 , 6] , [8 , 9 , 10]])+0.5
    
    for S_L in range(3):
        for st in range(3):
            next_array = next_trial(delay_vector, index, trial_type, all_st[st], S_L)
            next_array = np.array(next_array)
            curr_array = current_trial(delay_vector, index, trial_type, all_st[st], S_L)
            curr_array = np.array(curr_array)
            mean_curr = np.nanmean(curr_array)
            std_curr = np.nanstd(curr_array)/np.sqrt(len(curr_array[~np.isnan(curr_array)]))
            # if press_type == 'LatePress2':
            #     print(len(curr_array[~np.isnan(curr_array)]) , len(curr_array))
            axs.scatter(x_values_curr[S_L][st] , mean_curr, color = 'k' , s = 2)
            axs.errorbar(x_values_curr[S_L][st] , mean_curr, yerr=std_curr , color = 'k' , alpha = 0.6)
            mean_next = np.nanmean(next_array)
            std_next = np.nanstd(next_array)/np.sqrt(len(next_array[~np.isnan(next_array)]))
            axs.scatter(x_values_next[S_L][st] , mean_next, color = 'k' , s = 2)
            axs.errorbar(x_values_next[S_L][st] , mean_next, yerr=std_next , color = 'k' , alpha = 0.6)
            axs.plot([x_values_curr[S_L][st], x_values_next[S_L][st]] , [mean_curr , mean_next] , color = 'k')
            
            # noise_curr = np.random.normal(loc=0.0, scale=0.15, size=len(curr_array))
            # x_curr = x_values_curr[S_L][st]*np.ones(len(curr_array)) + noise_curr
            # noise_next = np.random.normal(loc=0.0, scale=0.25, size=len(next_array))
            # x_next = x_values_next[S_L][st]*np.ones(len(next_array)) + noise_next
            
            # axs.scatter(x_curr, curr_array, color = 'k' , s = 0.2 , alpha = 0.3)
            # axs.scatter(x_next, next_array, color = 'k' , s = 0.2 , alpha = 0.3)

            
        x_labels = ['All(All)' , 'All(ST)' , 'All(VG)' , 'Short blocks(All)' , 'Short blocks(ST)' , 'Short blocks(VG)' , 'Long blocks(All)' , 'Long blocks(ST)' , 'Long blocks(VG)']
        #axs.set_xticks(np.sort(np.append(np.array([0, 1 ,2 , 4, 5, 6, 8, 9, 10])-0.1, np.array([0, 1 ,2 , 4, 5, 6, 8, 9, 10])+0.1)))
        axs.set_xticks([0, 1 ,2 , 4, 5, 6, 8, 9, 10])
        axs.set_xticklabels(x_labels, rotation='vertical')
        
        title = 'Delay adaptation for a ' + press_type + ' trial'
        plot_layout(axs, title)
        axs.set_ylabel('Time (S)')
        
def run_delay_concat(axs, session_data, delay_data):
    delay_vector = delay_data['delay_all']
    outcomes = copy.deepcopy(session_data['outcomes'])
    raw_data = session_data['raw']
    #print(session_data['outcomes'][0][0:3])
    labels = create_label(delay_vector , outcomes)
    all_labels = ['Short' , 'Long' , 'DidNotPress1' , 'DidNotPress2' ,'Other']
    press_color = ['gold', 'deeppink' , '#4CAF50' , '#1976D2' , 'pink']
    #press_type_all = ['Short' , 'Long' , 'Reward' , 'EarlyPress2' , 'LatePress2']
    press_type_all = ['Short' , 'Long' , 'Reward' , 'EarlyPress2' ]
    num = -1
    for press_type in press_type_all:
        num = num + 1
        if press_type in all_labels:
            ref_array = copy.deepcopy(labels)
        else:
            ref_array = copy.deepcopy(outcomes)
        index , values , trial_type = devide(ref_array, press_type, delay_vector, raw_data)
            
        ST , VG = IsSelfTimed(session_data)
        all_sessions = np.arange(len(delay_vector))
        all_st = [all_sessions , np.array(ST) , np.array(VG)]
        x_values_curr = np.array([0,4,8])-0.5
        x_values_next = np.array([0,4,8])+0.5
        for S_L in range(1,3):
            for st in range(3):
                next_array = next_trial(delay_vector, index, trial_type, all_st[st], S_L)
                next_array = np.array(next_array)
                curr_array = current_trial(delay_vector, index, trial_type, all_st[st], S_L)
                curr_array = np.array(curr_array)
                mean_curr = np.nanmean(curr_array)
                std_curr = np.nanstd(curr_array)/np.sqrt(len(curr_array[~np.isnan(curr_array)]))
                # if press_type == 'LatePress2':
                #     print(len(curr_array[~np.isnan(curr_array)]) , len(curr_array))
                axs[st].scatter(x_values_curr[S_L] , mean_curr, color = press_color[num] , s = 8)
                axs[st].errorbar(x_values_curr[S_L] , mean_curr, yerr=std_curr , color = press_color[num] , alpha = 1)
                mean_next = np.nanmean(next_array)
                std_next = np.nanstd(next_array)/np.sqrt(len(next_array[~np.isnan(next_array)]))
                axs[st].scatter(x_values_next[S_L] , mean_next, color = press_color[num] , s = 8)
                axs[st].errorbar(x_values_next[S_L] , mean_next, yerr=std_next , color = press_color[num] , alpha = 1)
                axs[st].plot([x_values_curr[S_L], x_values_next[S_L]] , [mean_curr , mean_next] , color = press_color[num])
                
                x_labels = ['Short blocks','Long blocks']
                #axs.set_xticks(np.sort(np.append(np.array([0, 1 ,2 , 4, 5, 6, 8, 9, 10])-0.1, np.array([0, 1 ,2 , 4, 5, 6, 8, 9, 10])+0.1)))
                axs[st].set_xticks([ 4, 8])
                axs[st].set_xticklabels(x_labels, rotation='vertical')
                
                title = 'Delay adaptation'
                plot_layout(axs[st], title)
                axs[st].set_ylabel('Time (S)')
                
def run_delay_concat_s_l(axs, session_data, delay_data):
    delay_vector = delay_data['delay_all']
    outcomes = copy.deepcopy(session_data['outcomes'])
    raw_data = session_data['raw']
    #print(session_data['outcomes'][0][0:3])
    labels = create_label(delay_vector , outcomes)
    all_labels = ['Short' , 'Long' , 'DidNotPress1' , 'DidNotPress2' ,'Other']
    press_color = ['gold', 'deeppink' , '#4CAF50' , '#1976D2' , 'pink']
    #press_type_all = ['Short' , 'Long' , 'Reward' , 'EarlyPress2' , 'LatePress2']
    press_type_all = ['Short' , 'Long' , 'Reward' , 'EarlyPress2' ]
    st_label = [' All', ' Self Timed' , ' Vissually Guided']
    num = -1
    x_values_curr = np.array([[0,1,1], [5,6,6], [10,11,11], [15,16,16]])-1
    x_values_next = np.array([[0,1,1], [5,6,6], [10,11,11], [15,16,16]])+1
    LS = ['salid' , 'dashed','solid']
    for press_type in press_type_all:
        num = num + 1
        if press_type in all_labels:
            ref_array = copy.deepcopy(labels)
        else:
            ref_array = copy.deepcopy(outcomes)
        index , values , trial_type = devide(ref_array, press_type, delay_vector, raw_data)
            
        ST , VG = IsSelfTimed(session_data)
        all_sessions = np.arange(len(delay_vector))
        all_st = [all_sessions , np.array(ST) , np.array(VG)]
        
        for S_L in range(1,3):
            for st in range(3):
                next_array = next_trial(delay_vector, index, trial_type, all_st[st], S_L)
                next_array = np.array(next_array)
                curr_array = current_trial(delay_vector, index, trial_type, all_st[st], S_L)
                curr_array = np.array(curr_array)
                mean_curr = np.nanmean(curr_array)
                std_curr = np.nanstd(curr_array)/np.sqrt(len(curr_array[~np.isnan(curr_array)]))
                # if press_type == 'LatePress2':
                #     print(len(curr_array[~np.isnan(curr_array)]) , len(curr_array))
                axs[st].scatter(x_values_curr[num][S_L] , mean_curr, color = press_color[num] , s = 8)
                axs[st].errorbar(x_values_curr[num][S_L] , mean_curr, yerr=std_curr , color = press_color[num] , alpha = 1)
                mean_next = np.nanmean(next_array)
                std_next = np.nanstd(next_array)/np.sqrt(len(next_array[~np.isnan(next_array)]))
                axs[st].scatter(x_values_next[num][S_L] , mean_next, color = press_color[num] , s = 8)
                axs[st].errorbar(x_values_next[num][S_L] , mean_next, yerr=std_next , color = press_color[num] , alpha = 1)
                axs[st].plot([x_values_curr[num][S_L], x_values_next[num][S_L]] , [mean_curr , mean_next] , color = press_color[num], linestyle = LS[S_L])
                
                x_labels = ['Fast Press','Post Fast Press', 'Slow Press','Post Slow Press','Reward','Post Reward','Early Press','Post Early Press']
                axs[st].set_xticks([ 0 , 2, 5, 7, 10, 12, 15, 17])
                
                #axs.set_xticks(np.sort(np.append(np.array([0, 1 ,2 , 4, 5, 6, 8, 9, 10])-0.1, np.array([0, 1 ,2 , 4, 5, 6, 8, 9, 10])+0.1)))
                #axs[st].set_xticks([0,1,2,5,6,7,10,11,12,15,16,17])
                axs[st].set_xticklabels(x_labels, rotation='vertical')
                
                title = 'Delay adaptation' + st_label[st]
                plot_layout(axs[st], title)
                axs[st].set_ylabel('Time (S)')
                
def run_delay_concat_init_end(axs, session_data, delay_data):
    delay_vector = delay_data['delay_all']
    outcomes = copy.deepcopy(session_data['outcomes'])
    raw_data = session_data['raw']
    #print(session_data['outcomes'][0][0:3])
    labels = create_label(delay_vector , outcomes)
    all_labels = ['Short' , 'Long' , 'DidNotPress1' , 'DidNotPress2' ,'Other']
    press_color = ['gold', 'deeppink' , '#4CAF50' , '#1976D2' , 'pink']
    press_type_all = ['Short' , 'Long' , 'Reward' , 'EarlyPress2' , 'LatePress2']
    title_str = ['Fast Press' , 'Slow Press' , 'Reward' , 'Early Press', 'Late Press']  
    #press_type_all = ['Short' , 'Long' , 'Reward' , 'EarlyPress2' ]
    st_label = [' first', ' second' , ' third']
    num = -1
    x_values_curr = np.array([0 , 3, 6, 9])-0.5
    x_values_next = np.array([0 , 3, 6, 9])+0.5
    LS = ['salid' , 'dashed','solid']
    ST , VG = IsSelfTimed(session_data)
    all_sessions = np.arange(len(delay_vector))
    all_st = [np.arange(4) , np.arange(4)+ 5,np.arange(4)+ 10, np.arange(4)+ 20]
    for press_type in press_type_all:
        num = num + 1
        if press_type in all_labels:
            ref_array = copy.deepcopy(labels)
        else:
            ref_array = copy.deepcopy(outcomes)
        index , values , trial_type = devide(ref_array, press_type, delay_vector, raw_data)
        
        for S_L in range(1,3):
            for st in range(len(all_st)):
                next_array = next_trial(delay_vector, index, trial_type, all_st[st], S_L)
                next_array = np.array(next_array)
                curr_array = current_trial(delay_vector, index, trial_type, all_st[st], S_L)
                curr_array = np.array(curr_array)
                mean_curr = np.nanmean(curr_array)
                std_curr = np.nanstd(curr_array)/np.sqrt(len(curr_array[~np.isnan(curr_array)]))
                # if press_type == 'LatePress2':
                #     print(len(curr_array[~np.isnan(curr_array)]) , len(curr_array))
                axs[num].scatter(x_values_curr[st] , mean_curr, color = press_color[num] , s = 8)
                axs[num].errorbar(x_values_curr[st] , mean_curr, yerr=std_curr , color = press_color[num] , alpha = 1)
                mean_next = np.nanmean(next_array)
                std_next = np.nanstd(next_array)/np.sqrt(len(next_array[~np.isnan(next_array)]))
                axs[num].scatter(x_values_next[st] , mean_next, color = press_color[num] , s = 8)
                axs[num].errorbar(x_values_next[st] , mean_next, yerr=std_next , color = press_color[num] , alpha = 1)
                axs[num].plot([x_values_curr[st], x_values_next[st]] , [mean_curr , mean_next] , color = press_color[num], linestyle = LS[S_L])
                
                x_labels = ['session 1-5','session 5-10', 'session 10-15','session 20-25']
                axs[num].set_xticks([0 , 3, 6, 9])
                
                #axs.set_xticks(np.sort(np.append(np.array([0, 1 ,2 , 4, 5, 6, 8, 9, 10])-0.1, np.array([0, 1 ,2 , 4, 5, 6, 8, 9, 10])+0.1)))
                #axs[st].set_xticks([0,1,2,5,6,7,10,11,12,15,16,17])
                axs[num].set_xticklabels(x_labels, rotation='vertical')
                
                title = 'Delay adaptation ' + title_str[num] 
                plot_layout(axs[num], title)
                axs[num].set_ylabel('Time (S)')
                
def run_delay_concat_meta_learning(axs, session_data, delay_data):
    delay_vector = delay_data['delay_all']
    outcomes = copy.deepcopy(session_data['outcomes'])
    raw_data = session_data['raw']
    #print(session_data['outcomes'][0][0:3])
    labels = create_label(delay_vector , outcomes)
    all_labels = ['Short' , 'Long' , 'DidNotPress1' , 'DidNotPress2' ,'Other']
    press_color = ['gold', 'deeppink' , '#4CAF50' , '#1976D2' , 'pink']
    press_type_all = ['Short' , 'Long' , 'Reward' , 'EarlyPress2' , 'LatePress2']
    #press_type_all = ['Short' , 'Long' , 'Reward' , 'EarlyPress2' ]
    st_label = [' first', ' second' , ' third']
    num = -1
    x_values_curr = np.array([0 , 0, 0, 0])-0.5
    x_values_next = np.array([0 , 0, 0, 0])+0.5
    LS = ['salid' , 'dashed','solid']
    ST , VG = IsSelfTimed(session_data)
    all_sessions = np.arange(len(delay_vector))
    all_st = [np.arange(4) , np.arange(4)+ 5,np.arange(4)+ 10, np.arange(4)+ 20]
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=4+1)
    
    #axs.plot(time , trajectory , color = cmap(norm(max_num+1-(num)))
    for press_type in press_type_all:
        num = num + 1
        if press_type in all_labels:
            ref_array = copy.deepcopy(labels)
        else:
            ref_array = copy.deepcopy(outcomes)
        index , values , trial_type = devide(ref_array, press_type, delay_vector, raw_data)
        
        for S_L in range(1,3):
            for st in range(len(all_st)):
                color_tag = cmap(norm(4+1-(st)))
                next_array = next_trial(delay_vector, index, trial_type, all_st[st], S_L)
                next_array = np.array(next_array)
                curr_array = current_trial(delay_vector, index, trial_type, all_st[st], S_L)
                curr_array = np.array(curr_array)
                mean_curr = np.nanmean(curr_array)
                std_curr = np.nanstd(curr_array)/np.sqrt(len(curr_array[~np.isnan(curr_array)]))
                # if press_type == 'LatePress2':
                #     print(len(curr_array[~np.isnan(curr_array)]) , len(curr_array))
                axs[num].scatter(x_values_curr[st] , mean_curr, color = color_tag , s = 8)
                axs[num].errorbar(x_values_curr[st] , mean_curr, yerr=std_curr , color = color_tag)
                mean_next = np.nanmean(next_array)
                std_next = np.nanstd(next_array)/np.sqrt(len(next_array[~np.isnan(next_array)]))
                axs[num].scatter(x_values_next[st] , mean_next, color = color_tag , s = 8 )
                axs[num].errorbar(x_values_next[st] , mean_next, yerr=std_next , color = color_tag)
                axs[num].plot([x_values_curr[st], x_values_next[st]] , [mean_curr , mean_next] , color = color_tag, linestyle = LS[S_L])
                
                x_labels = ['current','post']
                axs[num].set_xticks([-0.5 ,0.5])
                
                #axs.set_xticks(np.sort(np.append(np.array([0, 1 ,2 , 4, 5, 6, 8, 9, 10])-0.1, np.array([0, 1 ,2 , 4, 5, 6, 8, 9, 10])+0.1)))
                #axs[st].set_xticks([0,1,2,5,6,7,10,11,12,15,16,17])
                axs[num].set_xticklabels(x_labels, rotation='vertical')
    title_str = ['Fast Press' , 'Slow Press' , 'Reward' , 'Early Press', 'Late Press']    
    for num in range(len(axs)):
        axs[num].set_xlim([-2 ,2])
        title = 'Delay adaptation ' + title_str[num] 
        plot_layout(axs[num], title)
        axs[num].set_ylabel('Time (S)')