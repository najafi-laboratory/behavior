# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:26:01 2024

@author: saminnaji3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from datetime import date
from matplotlib.lines import Line2D
import re
import seaborn as sns


states = [
    'Reward' , 
    'DidNotPress1' , 
    'DidNotPress2' , 
    'EarlyPress' , 
    'EarlyPress1' , 
    'EarlyPress2' ,
    'VisStimInterruptDetect1' ,         #int1
    'VisStimInterruptDetect2' ,         #int2
    'VisStimInterruptGray1' ,           #int3
    'VisStimInterruptGray2' ,           #int4
    'LatePress1' ,
    'LatePress2' ,
    'Other']                            #int
states_name = [
    'Reward' , 
    'DidNotPress1' , 
    'DidNotPress2' , 
    'EarlyPress' , 
    'EarlyPress1' , 
    'EarlyPress2' , 
    'VisStimInterruptDetect1' ,
    'VisStimInterruptDetect2' ,
    'VisStimInterruptGray1' ,
    'VisStimInterruptGray2' ,
    'LatePress1' ,
    'LatePress2' ,
    'VisInterrupt']
colors = [
    '#4CAF50',
    '#FFB74D',
    '#FB8C00',
    'r',
    '#64B5F6',
    '#1976D2',
    '#967bb6',
    '#9932CC',
    '#800080',
    '#2E003E',
    'pink',
    'deeppink',
    'cyan',
    'grey']

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

def count_label(session_label, states, plotting_sessions,norm=True):
    num_session = len(plotting_sessions)
    counts = np.zeros((num_session, len(states)))
    num = 0
    for i in plotting_sessions:
        
        for j in range(len(states)):
            if norm:
                total_outcomes = len(session_label[i])- session_label[i].count('Assisted') - session_label[i].count('Warmup')
                counts[num,j] = np.sum(
                    np.array(session_label[i]) == states[j]
                    ) / total_outcomes
            else:
                counts[num,j] = np.sum(
                    np.array(session_label[i]) == states[j]
                    )
        num = num + 1
    return counts

def plot_dist_indvi(axs, dates, delay_data , num , max_num):
    colors = ['#4CAF50','#FFB74D','pink','r','#64B5F6','#1976D2']
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=max_num+1)
    #print(len(delay_data))
     
    if num == 1:
        sns.distplot(a=delay_data[0], hist=False , color = cmap(norm(max_num+1-(num))) , label = 'Early Session' , ax = axs)
    elif num == max_num-1:
        sns.distplot(a=delay_data[0], hist=False , color = cmap(norm(max_num+1-(num))) , label = 'Late Session' , ax = axs)
    else:
        sns.distplot(a=delay_data[0], hist=False , color = cmap(norm(max_num+1-(num)))  , ax = axs)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Delay (s)')
    axs.set_ylabel('Density')
    axs.set_xlim([0 , 6])
    axs.set_title('Delay Distribution')
    axs.legend()
    
def plot_trajectory(axs, dates, time , trajectory , num , max_num , title):
   
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=max_num+1)
    
    axs.plot(time , trajectory , color = cmap(norm(max_num+1-(num))) , label = dates[num])
    axs.axvline(x = 0, color = 'r', linestyle='--')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Time from Press2 (s)')
    axs.set_ylabel('joystick deflection (deg)')
    #axs.set_xlim([0 , 2])
    axs.set_title(title)
    #axs.legend()

      
def run_indvi(axs, session_data , block_data, st):
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
    for i in plotting_sessions:
        dates1.append(dates[i])
    #print(dates1)
    
    max_num = 30
    num_plotting =  min(max_num , num_session)
    initial_start = max(num_session - 3*max_num , 0)
    
    
    for i in plotting_sessions:
        delay_data = block_data['delay'][i]
        plot_dist_indvi(axs, dates1 , delay_data , num, num_plotting)
        num = num + 1
        
def run_trajectory(axs, session_data , block_data, st):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    
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
    for i in plotting_sessions:
        dates1.append(dates[i])
    #print(dates1)
    
    max_num = 30
    num_plotting =  min(max_num , num_session)
    initial_start = max(num_session - 3*max_num , 0)
    
    for i in plotting_sessions:
        
       outcome_sess = outcomes[i]
       session_raw = session_data['raw'][i]['RawEvents']['Trial']
       delay = session_data['session_comp_delay'][i]
       encoder_data_aligned = session_data['encoder_pos_aligned'][i]
       # print(len(encoder_data_aligned))
       trajectory_reward = []
       trajectory_early = []
       trajectory_late = []
       num = num+1
       for j in range(len(outcome_sess)):
           outcome = outcome_sess[j]
           if outcome == 'Reward':
               press = delay[j] + session_raw[j]['States']['LeverRetract1'][1]
               pos_time = int(np.round(press*1000))
               left = 2000
               right = 1500
               time_reward = np.linspace(-left/1000, right/1000 , right+left)
               if pos_time < left:
                   nan_pad = np.zeros(left-pos_time)
                   nan_pad[:] = np.nan
                   trajectory_reward.append(np.append(nan_pad, encoder_data_aligned[j][0:pos_time+right]))
               else:
                   trajectory_reward.append(encoder_data_aligned[j][pos_time-left:pos_time+right])
           elif outcome == 'EarlyPress2':
               press = delay[j] + session_raw[j]['States']['LeverRetract1'][1]
               pos_time = int(np.round(press*1000))
               left = 1500
               right = 1500
               time_early = np.linspace(-left/1000, right/1000 , right+left)
               if pos_time < left:
                   nan_pad = np.zeros(left-pos_time)
                   nan_pad[:] = np.nan
                   trajectory_early.append(np.append(nan_pad, encoder_data_aligned[j][0:pos_time+right]))
               else:
                   trajectory_early.append(encoder_data_aligned[j][pos_time-left:pos_time+right])
           elif outcome == 'LatePress2':
               
               press = delay[j] + session_raw[j]['States']['LeverRetract1'][1]
               pos_time = int(np.round(press*1000))
               left = 4000
               right = 1500
               time_late = np.linspace(-left/1000, right/1000 , right+left)
               if pos_time < left:
                   nan_pad = np.zeros(left-pos_time)
                   nan_pad[:] = np.nan
                   trajectory_late.append(np.append(nan_pad, encoder_data_aligned[j][0:pos_time+right]))
               else:
                   #print(pos_time-left , pos_time+right)
                   trajectory_late.append(encoder_data_aligned[j][pos_time-left:pos_time+right])
           
       if len(trajectory_early) == 0:
           left = 1500
           right = 1500
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_early.append(nan_pad)
           
       if len(trajectory_reward) == 0:
           left = 2000
           right = 1500
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_reward.append(nan_pad)
           
       if len(trajectory_late) == 0:
           left = 4000
           right = 1500
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_late.append(nan_pad)
           
       plot_trajectory(axs[0], dates , time_reward , np.mean(np.array(trajectory_reward) , axis = 0) , num-1 , num_plotting , 'Rewarded trials')
       plot_trajectory(axs[1], dates , time_early , np.mean(np.array(trajectory_early) , axis = 0), num-1 , num_plotting , 'EarlyPress2 trials')
       plot_trajectory(axs[2], dates , time_late , np.mean(np.array(trajectory_late) , axis = 0), num-1 , num_plotting , 'LatePress2 trials')
                
       
def run_outcome(axs , session_data, st, label = 0):
    
    max_sessions=100
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    isSelfTime = session_data['isSelfTimedMode']
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    dates = dates[start_idx:]  
    new_dates = []
    for date_item in dates:
        new_dates.append(date_item[2:])
    dates = new_dates
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
    
    axs.axhline(0.5 , linestyle = '--' , color = 'gray')
    counts = count_label(outcomes, states, plotting_sessions)
    session_id = np.arange(len(plotting_sessions)) + 1
    bottom = np.cumsum(counts, axis=1)
    bottom[:,1:] = bottom[:,:-1]
    bottom[:,0] = 0
    width = 0.5
    for i in range(len(states)):
        axs.bar(
            session_id, counts[:,i],
            bottom=bottom[:,i],
            width=width,
            color=colors[i],
            label=states_name[i])
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Training session')
    
    axs.set_ylabel('Outcome percentages')
    
    axs.set_xticks(np.arange(len(plotting_sessions))+1)
    
    dates_label = []
    
    num = 0
    for i in plotting_sessions:
        num = num + 1
        if isSelfTime[i][0] == 1:
            dates_label.append(dates[i] + '-ST')
        else:
            dates_label.append(dates[i] + '-VG')
    axs.set_xticklabels(dates_label, rotation='vertical')
    
    if label == 1:
        dates_label = []
        
        num = 0
        for i in plotting_sessions:
            num = num + 1
            dates_label.append('Day' + str(num))
        axs.set_xticklabels(dates_label, rotation='vertical')
    
    
    axs.set_title('Reward percentage for completed trials across sessions')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
def run_trial_num(axs , session_data, st):
    
    max_sessions=100
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    isSelfTime = session_data['isSelfTimedMode']
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    dates = dates[start_idx:]  
    new_dates = []
    for date_item in dates:
        new_dates.append(date_item[2:])
    dates = new_dates
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
    num_trials = []
    for sess in plotting_sessions:
        num_trials.append(len(outcomes[sess]))
        
        
    session_id = np.arange(len(plotting_sessions)) + 1
    
    width = 0.5
   
    axs.bar(
        session_id, num_trials,
        width=width,
        color='gray',)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Training session')
    
    axs.set_ylabel('Number of trials')
    
    axs.set_xticks(np.arange(len(plotting_sessions))+1)
    
    dates_label = []
    
    for i in plotting_sessions:
        if isSelfTime[i][0] == 1:
            dates_label.append(dates[i] + '-ST')
        else:
            dates_label.append(dates[i] + '-VG')
    axs.set_xticklabels(dates_label, rotation='vertical')
    
    
    
    
    axs.set_title('Number of trials across sessions')
    

def run_delay(axs , session_data, st):
    
    max_sessions=100
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    chemo_labels = session_data['chemo']
    isSelfTime = session_data['isSelfTimedMode']
    
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    dates = dates[start_idx:]  
    new_dates = []
    for date_item in dates:
        new_dates.append(date_item[2:])
    dates = new_dates
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
    
    initial_delay = []
    final_delay = []
    for sess in plotting_sessions:
        predelay2 = session_raw[sess]['PrePress2Delay']
        initial_delay.append(np.mean(predelay2[1:30]))
        final_delay.append(np.mean(predelay2[-30:]))
        
        
    session_id = np.arange(len(plotting_sessions)) + 1
    
    
    
    axs.scatter(session_id , initial_delay , color = 'gray' , s = 5)
    axs.scatter(session_id + 0.2, final_delay , color = 'k' , s = 5)
    
    axs.plot(session_id , initial_delay , color = 'gray' , label = 'initial prepress2delay')
    axs.plot(session_id + 0.2, final_delay , color = 'k' , label = 'final prepress2delay')
    axs.set_ylabel('pre press2 delay')
    
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xticks(np.arange(len(plotting_sessions))+1)
    dates_label = []
    
    for i in plotting_sessions:
        if isSelfTime[i][0] == 1:
            dates_label.append(dates[i] + '-ST')
        else:
            dates_label.append(dates[i] + '-VG')
    axs.set_xticklabels(dates_label, rotation='vertical')
    
    
    
    
    axs.set_title('Pre Press2 Delay across sessions')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
    
def plot_dist(axs, delay_data, color_tag, alpha = 0.95):
    delay_data = delay_data[~np.isnan(delay_data)]
    mid = np.percentile(delay_data, 50)
    #lim = np.percentile(delay_data, 95)
    delay_data = delay_data[delay_data<3]
    percentile_90 = np.percentile(delay_data, 90)
    # bin_initial = np.linspace(np.nanmin(delay_data), percentile_90,  15)
    # bin_end = np.linspace(percentile_90, np.nanmax(delay_data), 15)
    # hist , bin_edge = np.histogram(delay_data, bins=np.append(bin_initial, bin_end))
    bin_initial = np.linspace(np.nanmin(delay_data), np.nanmax(delay_data),  27)
    #bin_end = np.linspace(percentile_90, np.nanmax(delay_data), 15)
    hist , bin_edge = np.histogram(delay_data, bins=bin_initial)
    bin_center = (bin_edge[1:]+bin_edge[:-1])/2
    hist = hist/len(delay_data)
    if alpha == 1:
        plt.plot(bin_center, hist, color = color_tag, alpha = alpha, label = 'n = ' + str(len(delay_data)))
    else:
        plt.plot(bin_center, hist, color = color_tag, alpha = alpha)
    #plt.axvline(x = np.percentile(delay_data, 30), color = color_tag,alpha = alpha, linewidth = 1, linestyle='--')
    #plt.axvline(x = np.percentile(delay_data, 70), color = color_tag,alpha = alpha, linewidth = 1, linestyle=':')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Delay (s)')
    axs.set_ylabel('Probability')
    axs.set_xlim([0 , 3])
    #axs.legend()
    
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
    #plot_dist(axs, np.array(delay_all), 'k', alpha = 1)
    axs.set_title('Distribution of Inter Press Interval '+ st_title[st])