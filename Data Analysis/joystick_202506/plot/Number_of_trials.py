# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:22:42 2025

@author: saminnaji3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from datetime import date
from matplotlib.lines import Line2D
import re
import seaborn as sns

def IsSelfTimed(session_data):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    isSelfTime = session_data['isSelfTimedMode']
    
    VG = []
    ST = []
    for i in range(0 , len(isSelfTime)):
        if isSelfTime[i][5] == 1 or isSelfTime[i][5] == np.nan:
            ST.append(i)
        else:
            VG.append(i)
    return ST , VG

def count_label(session_data, session_label, plotting_sessions, norm=True):
    
    num_session = len(plotting_sessions)
    counts = np.zeros((2 , num_session))
    num = 0
    
    for i in plotting_sessions:
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        for j in range(len(session_label[i])):
            if norm:
                if session_label[i][j] == 'Reward':
                    if trial_types[j] == 1:
                        counts[0 , num] = counts[0 , num] + 1
                    else:
                        counts[1 , num] = counts[1 , num] + 1
        num = num + 1
    return counts

def count_delay(session_data, plotting_sessions):
    
    num_session = len(plotting_sessions)
    delay_mean = np.zeros((2 , num_session))
    delay_std = np.zeros((2 , num_session))
    num = 0
    
    for i in plotting_sessions:
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        delay = raw_data['PrePress2Delay']
        delay_short = []
        delay_long = []
        for j in range(len(trial_types)):
            if trial_types[j] == 1:
                delay_short.append(delay[j])
            else:
                delay_long.append(delay[j])
        delay_mean[0 , num] = np.mean(np.array(delay_short))
        delay_mean[1 , num] = np.mean(np.array(delay_long))
        delay_std[0 , num] = np.std(np.array(delay_short))
        delay_std[1 , num] = np.std(np.array(delay_long))
        num = num + 1
    return delay_mean , delay_std

def plot_dist_indvi(axs, short_delay_data , long_delay_data , num , max_num , title):
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=max_num+1)
    
    if not title == 'rewarded':
        if len(short_delay_data) > 0: 
            short_all = np.concatenate(short_delay_data)
        else:
            short_all = [np.nan]
        if len(long_delay_data) > 0: 
           long_all = np.concatenate(long_delay_data) 
        else:
            long_all = [np.nan]
    else:
        short_all = short_delay_data
        long_all = long_delay_data
     
    sns.distplot(a=short_all, hist=False , color = cmap(norm(max_num+1-num))  , ax = axs[0])
    sns.distplot(a=long_all, hist=False , color = cmap(norm(max_num+1-num))  , ax = axs[1])
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].set_xlabel('Delay (s)')
    axs[0].set_ylabel('Density')
    axs[0].set_xlim([0 , 4])
    axs[0].set_title('Delay Distribution ' +title+' (short)')
    
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].set_xlabel('Delay (s)')
    axs[1].set_ylabel('Density')
    axs[1].set_xlim([0 , 4])
    axs[1].set_title('Delay Distribution ' +title+' (long)')

def run_dist(axs , session_data , block_data , st):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    
    num_session = len(block_data['NumBlocks'])
    ST , VG = IsSelfTimed(session_data)
    
    if st == 1:
        plotting_sessions = np.array(ST)
        num_session = len(ST)
    elif st == 2:
        plotting_sessions = np.array(VG)
        num_session = len(VG)
    else:
        plotting_sessions = np.arange(num_session)
    dates1 = []
    num = 0
    max_num = 30
    num_plotting =  min(max_num , num_session)
    for i in plotting_sessions:
        #print(i)
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        delay_data = block_data['delay'][i]
        
        
        short_delay_data = []
        long_delay_data = []
        for j in range(len(short_id)):
            short_delay_data.append(block_data['delay'][i][short_id[j]])
                
        for j in range(len(long_id)):
            long_delay_data.append(block_data['delay'][i][long_id[j]])

        plot_dist_indvi(axs , short_delay_data , long_delay_data , num , num_plotting , '')
        num = num + 1
        
def run_dist_rewarded(axs , session_data , block_data , st):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    
    num_session = len(block_data['NumBlocks'])
    ST , VG = IsSelfTimed(session_data)
    
    if st == 1:
        plotting_sessions = np.array(ST)
        num_session = len(ST)
    elif st == 2:
        plotting_sessions = np.array(VG)
        num_session = len(VG)
    else:
        plotting_sessions = np.arange(num_session)
    dates1 = []
    num = 0
    max_num = 30
    num_plotting =  min(max_num , num_session)
    for i in plotting_sessions:
        short_delay_data = []
        long_delay_data = []
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        delay = session_data['session_comp_delay'][i]
        for j in range(len(outcomes[i])):
            if outcomes[i][j] == 'Reward':
                if trial_types[j] == 1:
                    short_delay_data.append(delay[j])
                else:
                    long_delay_data.append(delay[j])

        plot_dist_indvi(axs , short_delay_data , long_delay_data , num , num_plotting , 'rewarded')
        num = num + 1
        
def run_trial_num(axs , session_data , st):
    
    max_sessions=100
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    isSelfTime = session_data['isSelfTimedMode']
    start_idx = 0
    num_session = len(dates)
    ST , VG = IsSelfTimed(session_data)
    
    if st == 1:
        plotting_sessions = np.array(ST)
        num_session = len(ST)
    elif st == 2:
        plotting_sessions = np.array(VG)
        num_session = len(VG)
    else:
        plotting_sessions = np.arange(num_session)
    dates1 = []
    num = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    dates = dates[start_idx:]  
    new_dates = []
    for date_item in dates:
        new_dates.append(date_item[2:])
    dates = new_dates
    
    raw_data = session_data['raw']
    num_trials_short = []
    num_trials_long = []
    for sess in plotting_sessions:
        trial_types = raw_data[sess]['TrialTypes']
        num_trials_short.append(trial_types.count(1))
    for sess in plotting_sessions:
        trial_types = raw_data[sess]['TrialTypes']
        num_trials_long.append(trial_types.count(2))
    
        
        
    session_id = np.arange(len(plotting_sessions)) + 1
    width = 0.5
    
    axs.bar(
        session_id, num_trials_short,
        width=width,
        color='y',
        label='short')
    
    axs.bar(
        session_id , num_trials_long,
        bottom=num_trials_short,
        width=width,
        color='b',
        label='long')
    
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Training session')
    
    axs.legend()
    axs.set_ylabel('Number of trials')
    
    axs.set_xticks(np.arange(len(plotting_sessions))+1)
    
    dates_label = []
    
    for i in plotting_sessions:
        if isSelfTime[i][0] == 1:
            dates_label.append( dates[i] + '-ST')
        else:
            dates_label.append(dates[i] + '-VG')
    axs.set_xticklabels(dates_label, rotation='vertical' , fontsize =8)

    axs.set_title('Number of trials across sessions')
    
def run_reward_num(axs , session_data , st):
    
    max_sessions=100
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    isSelfTime = session_data['isSelfTimedMode']
    start_idx = 0
    num_session = len(dates)
    ST , VG = IsSelfTimed(session_data)
    
    if st == 1:
        plotting_sessions = np.array(ST)
        num_session = len(ST)
    elif st == 2:
        plotting_sessions = np.array(VG)
        num_session = len(VG)
    else:
        plotting_sessions = np.arange(num_session)
    dates1 = []
    num = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    dates = dates[start_idx:]  
    new_dates = []
    for date_item in dates:
        new_dates.append(date_item[2:])
    dates = new_dates
    
    raw_data = session_data['raw']
    num_trials_short = count_label(session_data, outcomes, plotting_sessions)[0 , :]
    num_trials_long = count_label(session_data, outcomes, plotting_sessions)[1 , :]
    
    session_id = np.arange(len(plotting_sessions)) + 1
    width = 0.5
    
    axs.bar(
        session_id, num_trials_short,
        width=width,
        color='limegreen',
        label='short')
    
    axs.bar(
        session_id , num_trials_long,
        bottom=num_trials_short,
        width=width,
        color='DarkGreen',
        label='long')
    
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Training session')
    
    axs.legend()
    axs.set_ylabel('Number of rewarded trials')
    
    axs.set_xticks(np.arange(len(plotting_sessions))+1)
    
    dates_label = []
    
    for i in plotting_sessions:
        if isSelfTime[i][0] == 1:
            dates_label.append( dates[i] + '-ST')
        else:
            dates_label.append(dates[i] + '-VG')
    axs.set_xticklabels(dates_label, rotation='vertical' , fontsize =8)

    axs.set_title('Number of rewarded trials across sessions')
    
def run_delay(axs , session_data , st):
    
    max_sessions=100
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    chemo_labels = session_data['chemo']
    isSelfTime = session_data['isSelfTimedMode']
    num_session = len(dates)
    ST , VG = IsSelfTimed(session_data)
    
    if st == 1:
        plotting_sessions = np.array(ST)
        num_session = len(ST)
    elif st == 2:
        plotting_sessions = np.array(VG)
        num_session = len(VG)
    else:
        plotting_sessions = np.arange(num_session)
        
    delay_mean , delay_std = count_delay(session_data, plotting_sessions)
    session_id = np.arange(len(plotting_sessions)) + 1
    
    axs.scatter(session_id , delay_mean[0 , :] , color = 'y' , s = 5)
    axs.scatter(session_id , delay_mean[1 , :] , color = 'b' , s = 5)
    
    axs.plot(session_id , delay_mean[0 , :] , color = 'y' , label = 'Short')
    axs.plot(session_id , delay_mean[1 , :] , color = 'b' , label = 'Long')
    
    axs.set_ylabel('pre press2 delay')
    
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xticks(np.arange(len(outcomes))+1)
    axs.set_xticks(np.arange(len(plotting_sessions))+1)
    
    dates_label = []
    
    for i in plotting_sessions:
        if isSelfTime[i][0] == 1:
            dates_label.append( dates[i] + '-ST')
        else:
            dates_label.append(dates[i] + '-VG')
    axs.set_xticklabels(dates_label, rotation='vertical' , fontsize =8)

    
    axs.set_title('Pre Press2 Delay across sessions')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)