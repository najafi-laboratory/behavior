# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 14:53:50 2024

@author: saminnaji3
"""

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
import pandas as pd
from sklearn.preprocessing import normalize


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


def count_label(session_data, session_label, states, plotting_sessions, norm=True):
    
    num_session = len(plotting_sessions)
    counts = np.zeros((3 , num_session, len(states)))
    numtrials = np.zeros((3 , num_session))
    num = 0
    for i in plotting_sessions:
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        for j in range(len(session_label[i])):
            if norm:
                if session_label[i][j] in states:
                    k = states.index(session_label[i][j])
                    if trial_types[j] == 1:
                        counts[0 , num , k] = counts[0 , num , k] + 1
                        numtrials[0 , num] = numtrials[0 , num] + 1
                    else:
                        counts[1 , num , k] = counts[1 , num , k] + 1
                        numtrials[1 , num] = numtrials[1 , num] + 1
                    
                    
        for j in range(len(states)):
            counts[0 , num , j] = counts[0 , num , j]/numtrials[0 , num]
            counts[1 , num , j] = counts[1 , num , j]/numtrials[1 , num]
        num = num + 1
            
    return counts

def moving_average(df , w):
    average = []
    trial = df['time'].unique()
    for i in trial:
        temp = df[(df['time'].between(i , i + w))]
        average.append(np.mean(temp['delay']))
    return average

def plot_dist_indvi(axs, dates, delay_data , num , max_num):
    colors = ['#4CAF50','#FFB74D','pink','r','#64B5F6','#1976D2']
    cmap = plt.cm.Greys
    norm = plt.Normalize(vmin=0, vmax=max_num)
    
    sns.distplot(a=delay_data, hist=False , color = cmap(norm(num)) , label = dates[num] , ax = axs)
    #axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Delay (s)')
    axs.set_ylabel('Density')
    #axs.set_xlim([0 , 2])
    axs.set_title('Delay Distribution')
    axs.legend()
    
def plot_trajectory(axs, dates, time , trajectory , num , max_num , title , legend, sem = 0):
   
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=max_num+1)
    trajectory_mean = np.mean(trajectory , axis = 0)
    if sem == 1:
        trajectory_sem = np.std(trajectory , axis = 0)/np.sqrt(len(trajectory))
        axs.fill_between(time ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = cmap(norm(max_num+1-num)) , alpha = 0.1)
    
    axs.plot(time , trajectory_mean , color = cmap(norm(max_num+1-num)) , label = dates[num])
    axs.axvline(x = 0, color = 'gray', linestyle='--')
    #axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Time from Press2 (s)')
    axs.set_ylabel('joystick deflection (deg) (mean +/- sem)')
    axs.set_xlim([-2 , 1.5])
    axs.set_title(title)
    if legend:
        axs.legend()

def plot_delay_adat(axs , data , num , max_num , title , lim , all = 0):
    
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=max_num+1)
    
    #axs.scatter(data['time'] , data['delay'] , color = cmap(norm(num)) , alpha = 0.2 , s = 2)
    if all: 
        w = 10
        df_sorted = data.sort_values(by='time')
        filtered_df = df_sorted[~df_sorted['delay'].isin([np.nan])]
        average = moving_average(filtered_df , w)
        x_value = np.arange(1 , len(average)+ 1)
        
        axs.plot(x_value , average , color = 'k' , linewidth = 4)
    else:
        w = 10
        df_sorted = data.sort_values(by='time')
        filtered_df = df_sorted[~df_sorted['delay'].isin([np.nan])]
        average = moving_average(filtered_df , w)
        x_value = np.arange(1 , len(average)+ 1)
        
        axs.plot(x_value , average , color = cmap(norm(num)))
        
        
        axs.tick_params(tick1On=False)
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.set_xlabel('trial number')
        axs.set_ylabel('normalized delay (s)')
        axs.set_title(title)
        
        if lim:
            axs.set_ylim([0.5 , 3])
        
        #axs.set_xlim([0 , 23])
def run_indvi(axs, session_data , block_data):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    
    num_session = len(block_data['NumBlocks'])
    max_num = 20
    num_plotting =  min(max_num , num_session)
    initial_start = max(num_session - 3*max_num , 0)
    for k in range(num_session//max_num + 1):
        
        start = k*num_plotting + initial_start
        end = min((k+1)*num_plotting , num_session)+ initial_start
        date = dates[start:end] 
        for i in range(start , end):
            
            delay_data = block_data['delay'][i]
            
            
            plot_dist_indvi(axs, date , delay_data , (i-start)-(i-start)//num_plotting , num_plotting)
        
        
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
       trajectory_reward_short = []
       trajectory_reward_long = []
       raw_data = session_data['raw'][i]
       trial_types = raw_data['TrialTypes']
       num = num + 1
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
                   if trial_types[j] == 1:
                       trajectory_reward_short.append(np.append(nan_pad, encoder_data_aligned[j][0:pos_time+right]))
                   else:
                       trajectory_reward_long.append(np.append(nan_pad, encoder_data_aligned[j][0:pos_time+right]))
               else:
                   if trial_types[j] == 1:
                       trajectory_reward_short.append(encoder_data_aligned[j][pos_time-left:pos_time+right])
                   else:
                       trajectory_reward_long.append(encoder_data_aligned[j][pos_time-left:pos_time+right])
                   
       
           
       if len(trajectory_reward_short) == 0:
           left = 2000
           right = 1500
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_reward_short.append(nan_pad)
           
       if len(trajectory_reward_long) == 0:
           left = 2000
           right = 1500
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_reward_long.append(nan_pad)
           
       plot_trajectory(axs[0], dates1 , time_reward , np.array(trajectory_reward_short) , num-1 , num_plotting , 'Rewarded trials (short)' , 0, sem = 1)
       plot_trajectory(axs[1], dates1 , time_reward , np.array(trajectory_reward_long), num-1 , num_plotting , 'Rewarded trials (long)' , 0, sem = 1)
       
       
def run_outcome(axs , session_data, st):
    
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
    num_session = len(dates)
    for date_item in dates:
        new_dates.append(date_item[2:])
    dates = new_dates
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
    
    counts = count_label(session_data, outcomes, states, plotting_sessions)
    session_id = np.arange(len(plotting_sessions)) + 1
    
    short_c_bottom = np.cumsum(counts[0 , : , :], axis=1)
    short_c_bottom[:,1:] = short_c_bottom[:,:-1]
    short_c_bottom[:,0] = 0
    
    long_c_bottom = np.cumsum(counts[1 , : , :], axis=1)
    long_c_bottom[:,1:] = long_c_bottom[:,:-1]
    long_c_bottom[:,0] = 0
    
    width = 0.2
    
    top_ticks = []
    
    axs.axhline(0.5 , linestyle = '--' , color = 'gray')
    for i in range(len(plotting_sessions)):
        top_ticks.append('Short|Long')
    
    for i in range(len(states)):
        axs.bar(
            session_id , counts[0,:,i],
            bottom=short_c_bottom[:,i],
            width=width,
            color=colors[i],
            label=states_name[i])
        axs.bar(
            session_id + width, counts[1,:,i],
            bottom=long_c_bottom[:,i],
            width=width,
            color=colors[i])
    
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Training session')
    
    axs.set_ylabel('Outcome percentages')
    
    dates_label = []
    label_loc = []
    num = 0
    for i in plotting_sessions:
        num = num+1
        #dates_label.append('Short')
        #label_loc.append(num)
        if isSelfTime[i][5] == 1:
            dates_label.append(dates[i] + '-ST')
        else:
            dates_label.append(dates[i] + '-VG')
        label_loc.append(num+width/2)
        #dates_label.append('Long')
        #label_loc.append(num+width)
        
    axs.set_xticks(np.array(label_loc))
    axs.set_xticklabels(dates_label, rotation='vertical', fontsize=8)
    
    axs.set_title('Reward percentage for completed trials across sessions (short: left, long: right)')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
def run_block_len(axs , session_data , block_data, st):
    
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
    
    
    len_block_short = []
    len_block_long = []
    cmap = plt.cm.inferno
    max_color = len(plotting_sessions)+1
    norm = plt.Normalize(vmin=0, vmax=max_color)
    color_short = []
    color_long = []
    for i in plotting_sessions:
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        delay_data = block_data['delay'][i]
        short_len = []
        long_len = []
        
        if len(long_id) > 0:
            if short_id[0] == 0:
                short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
            
            
        for j in range(len(short_id)):
            len_block_short.append(len(block_data['delay'][i][short_id[j]]))
            color_short.append(cmap(norm(max_color-num)))
            short_len.append(len(block_data['delay'][i][short_id[j]]))
                
        for j in range(len(long_id)):
            len_block_long.append(len(block_data['delay'][i][long_id[j]]))
            color_long.append(cmap(norm(max_color-num)))
            long_len.append(len(block_data['delay'][i][long_id[j]]))
            
        if i < len(outcomes) - 1:   
            axs[0].axvline(len(len_block_short) - 0.5 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)  
            axs[1].axvline(len(len_block_long) - 0.5 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)  
        num = num + 1
        
        if len(short_len) > 1:
            s , b = np.polyfit(np.arange(len(short_len)), np.array(short_len), 1)
            axs[0].plot(np.arange(len(short_len))+len(color_short)-len(short_len), s*(np.arange(len(short_len)))+b, color = 'r', alpha = 0.3)
        
        if len(long_len) > 1:
            s , b = np.polyfit(np.arange(len(long_len)), np.array(long_len), 1)
            axs[1].plot(np.arange(len(long_len))+len(color_long)-len(long_len), s*(np.arange(len(long_len)))+b, color = 'r', alpha = 0.3)
    
    axs[0].set_ylabel('Number of trials')
    axs[1].set_ylabel('Number of trials')
    axs[0].set_xlabel('Blocks (concatenated sessions)')
    axs[1].set_xlabel('Blocks (concatenated sessions)')
    
    axs[0].scatter(np.arange(len(len_block_short)) , len_block_short , color = color_short , s = 20 , edgecolors= 'gray')
    axs[1].scatter(np.arange(len(len_block_long)) , len_block_long , color = color_long , s = 20, edgecolors= 'gray')
    
    axs[0].set_title('Number of trials in each short block')
    axs[1].set_title('Number of trials in each long block')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)

def run_delay(axs , axs1 ,  session_data, block_data, st):
    
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
    
    
    mean_block_short = []
    mean_block_long = []
    cmap1 = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=len(plotting_sessions)+1)
    cmap2 = plt.cm.inferno
    color_short = []
    color_long = []
    upper_short = []
    upper_long = []
    lower_short = []
    lower_long = []
    slope_short = []
    slope_long = []
    slope_color = []
    initial = 0
    for i in plotting_sessions:
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        delay_data = block_data['delay'][i]
        lower = block_data['LowerBand'][i]
        upper = block_data['UpperBand'][i]
        short_x = []
        long_x = []
        
        if len(long_id) > 0:
            if short_id[0] == 0:
                short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
        
        for j in range(len(short_id)):
            if not np.isnan(np.nanmean(delay_data[short_id[j]])):
                mean_block_short.append(np.nanmean(delay_data[short_id[j]]))
                axs[0].errorbar(len(mean_block_short)-1 , mean_block_short[-1] , yerr = np.nanstd(delay_data[short_id[j]])/np.sqrt(np.count_nonzero(~np.isnan(delay_data[short_id[j]]))) , color = cmap1(norm(num_session+1-num)) , alpha = 0.2)
                color_short.append(cmap1(norm(num_session+1-num)))
                upper_short.append(np.nanmean(upper[short_id[j]]))
                lower_short.append(np.nanmean(lower[short_id[j]]))
                if len(delay_data[short_id[j]]) > initial:
                    short_x.append(np.nanmean(delay_data[short_id[j]][initial:]))
                else:
                    short_x.append(np.nanmean(delay_data[short_id[j]]))
                
        for j in range(len(long_id)):
            if not np.isnan(np.nanmean(delay_data[long_id[j]])):
                mean_block_long.append(np.nanmean(delay_data[long_id[j]]))
                axs[1].errorbar(len(mean_block_long)-1 , mean_block_long[-1] , yerr = np.nanstd(delay_data[long_id[j]])/np.sqrt(np.count_nonzero(~np.isnan(delay_data[long_id[j]]))) , color = cmap2(norm(num_session+1-num)) , alpha = 0.2)
                color_long.append(cmap2(norm(num_session+1-num)))
                upper_long.append(np.nanmean(upper[long_id[j]]))
                lower_long.append(np.nanmean(lower[long_id[j]]))
                if len(delay_data[long_id[j]]) > initial:
                    long_x.append(np.nanmean(delay_data[long_id[j]][initial:]))
                else:
                    long_x.append(np.nanmean(delay_data[long_id[j]]))
        
        if len(short_x) > 1:
            s , b = np.polyfit(np.arange(len(short_x)), np.array(short_x), 1)
            slope_short.append(s)
            axs[0].plot(np.arange(len(short_x))+len(color_short)-len(short_x), s*(np.arange(len(short_x)))+b, color = 'r', alpha = 0.3)
        else:
            slope_short.append(np.nan)
        
        if len(long_x) > 1:
            s , b = np.polyfit(np.arange(len(long_x)), np.array(long_x), 1)
            slope_long.append(s)
            axs[1].plot(np.arange(len(long_x))+len(color_long)-len(long_x), s*(np.arange(len(long_x)))+b, color = 'r', alpha = 0.3)
        else: 
            slope_long.append(np.nan)
        slope_color.append(cmap1(norm(num_session+1-num)))
                
        if i < len(outcomes) - 1:   
            axs[0].axvline(len(mean_block_short) - 0.5 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)  
            axs[1].axvline(len(mean_block_long) - 0.5 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)  
        num = num + 1

    axs[0].set_ylabel('Delay (s) (mean +/- sem)')
    axs[1].set_ylabel('Delay (s) (mean +/- sem)')
    axs[0].set_xlabel('Blocks (concatenated sessions)')
    axs[1].set_xlabel('Blocks (concatenated sessions)')
    
    axs[0].fill_between(np.arange(len(mean_block_short)) ,lower_short , upper_short, color = 'gray' , alpha = 0.1)
    axs[1].fill_between(np.arange(len(mean_block_long)) ,lower_long , upper_long, color = 'gray' , alpha = 0.1)
    axs[0].scatter(np.arange(len(mean_block_short)) , mean_block_short , color = color_short ,  s = 20 , edgecolors= 'gray')
    axs[1].scatter(np.arange(len(mean_block_long)) , mean_block_long , color = color_long ,  s = 20 , edgecolors= 'gray')
    
    axs1.scatter(slope_short , slope_long , color = slope_color ,  s = 20 , edgecolors= 'gray')
    axs1.axvline(0 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)
    axs1.axhline(0 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)
    axs1.set_title('delay adaptation slope (each point one session)')
    axs1.spines['right'].set_visible(False)
    axs1.spines['top'].set_visible(False)
    axs1.set_ylabel('long blocks slope (s/block)')
    axs1.set_xlabel('short blocks slope (s/block)')
    
    axs[0].set_title('Average Delay in each short block')
    axs[1].set_title('Average Delay in each long block')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].set_ylim([0.5 , 2])
    
def run_epoch(axs , session_data , block_data, st):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    
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
    
    num_session = len(plotting_sessions)
    
    
    all_lens = np.concatenate(block_data['NumTrials'][:])
    unique = set(all_lens)
    max_trial_num_temp = max(all_lens)
    max_trial_num = max_trial_num_temp
    i = 1
   
    max_trial_num = max_trial_num_temp
    
    
    
    max_num = num_session
    num_plotting =  num_session
    initial_start = 0
    epoch_len = 5
    short_first1 = []
    short_last1 = []
    long_first1 = []
    long_last1 = []
    
    short_first = []
    short_last = []
    long_first = []
    long_last = []
    
    short_first_sem = []
    short_last_sem = []
    long_first_sem = []
    long_last_sem = []
    
    
    for i in plotting_sessions:
        short_delay_first = []
        short_delay_last = []
        
        long_delay_first = []
        long_delay_last = []
        
        short_first_temp = []
        short_last_temp = []
        long_first_temp = []
        long_last_temp = []
        
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        max_trial_num = np.max(block_data['NumTrials'][i])
        if short_id[0] == 0:
            short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
        for j in range(len(short_id)):
            current_block = block_data['delay'][i][short_id[j]]
            if len(current_block)>1.5*epoch_len:
                short_delay_first.append(np.nanmean(current_block[0:epoch_len]))
                short_delay_last.append(np.nanmean(current_block[epoch_len:]))
                short_first_temp.append(current_block[0:epoch_len])
                short_last_temp.append(current_block[epoch_len:])
                
        for j in range(len(long_id)):
            current_block = block_data['delay'][i][long_id[j]]
            if len(current_block)>1.5*epoch_len:
                long_delay_first.append(np.nanmean(current_block[0:epoch_len]))
                long_delay_last.append(np.nanmean(current_block[epoch_len:]))
                long_first_temp.append(current_block[0:epoch_len])
                long_last_temp.append(current_block[epoch_len:])
                
        short_first1.append(np.mean(short_delay_first))
        short_last1.append(np.mean(short_delay_last))
        long_first1.append(np.mean(long_delay_first))
        long_last1.append(np.mean(long_delay_last))
        
        short_first_temp = np.concatenate(short_first_temp)
        short_last_temp = np.concatenate(short_last_temp)
        if (len(long_first_temp) > 0):
            long_first_temp = np.concatenate(long_first_temp)
            long_last_temp = np.concatenate(long_last_temp)
        
        short_first.append(np.nanmean(short_first_temp))
        short_last.append(np.nanmean(short_last_temp))
        long_first.append(np.nanmean(long_first_temp))
        long_last.append(np.nanmean(long_last_temp))
        
        short_first_sem.append(np.nanstd(short_first_temp)/np.sqrt(np.count_nonzero(~np.isnan(short_first_temp))))
        short_last_sem.append(np.nanstd(short_last_temp)/np.sqrt(np.count_nonzero(~np.isnan(short_last_temp))))
        long_first_sem.append(np.nanstd(long_first_temp)/np.sqrt(np.count_nonzero(~np.isnan(long_first_temp))))
        long_last_sem.append(np.nanstd(long_last_temp)/np.sqrt(np.count_nonzero(~np.isnan(long_last_temp))))
        
    size = 8
    # x_first = np.arange(0 ,len(short_first))/len(short_first)
    # x_last = np.arange(0 ,len(short_last))/len(short_last)+
    x_first = np.zeros(len(short_first))
    x_last = np.zeros(len(short_last))+1.5
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=num_session+1)
    
    color1 = []
    for i in range(max(len(short_last),len(short_first))):
        color1.append(np.array(cmap(norm(num_session+1-i))))
        axs.plot([x_first[i] , x_last[i]],[short_first1[i] , short_last1[i]], color = np.array(cmap(norm(num_session+1-i))))
        axs.errorbar(x_first[i], short_first1[i], yerr=short_first_sem[i] , color = np.array(cmap(norm(num_session+1-i))), alpha = 0.2)
        axs.errorbar(x_last[i], short_last1[i], yerr=short_last_sem[i], color = np.array(cmap(norm(num_session+1-i))), alpha = 0.2)
    
    axs.scatter(x_first, short_first1, color = color1 , label = date , s = size)
    axs.scatter(x_last, short_last1, color = color1 , s = size)
    
    # x_first = np.arange(0 ,len(long_first))/len(long_first)+5
    # x_last = np.arange(0 ,len(long_last))/len(long_last)+7
    x_first = np.zeros(len(long_first))+4.5
    x_last = np.zeros(len(long_last))+6
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=num_session+1)
    color1 = []
    for i in range(max(len(long_last),len(long_first))):
        color1.append(np.array(cmap(norm(num_session+1-i))))
        axs.plot([x_first[i] , x_last[i]],[long_first1[i] , long_last1[i]], color = np.array(cmap(norm(num_session+1-i))))
        axs.errorbar(x_first[i], long_first1[i], yerr=long_first_sem[i] , color = np.array(cmap(norm(num_session+1-i))), alpha = 0.2)
        axs.errorbar(x_last[i], long_last1[i], yerr=long_last_sem[i], color = np.array(cmap(norm(num_session+1-i))), alpha = 0.2)
    
    axs.scatter(x_first, long_first1, color = color1, s= size)
    axs.scatter(x_last, long_last1, color = color1, s= size)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_ylabel('delay (s) (mean +/- sem)')
    axs.set_title('Average Delay for first and last epoches (len first epoch = ' + str(epoch_len) + ')') 
    
   
    axs.set_xticks([0 , 1.5 , 4.5 , 6])
    axs.set_xticklabels(['Short First' , 'Short Last' , 'Long First' , 'Long Last'], rotation='vertical')

def run_delay_adapt(axs, session_data, block_data, st):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
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
        
    num_session = len(plotting_sessions)
    all_short = []
    all_long = []
    all_short_time = []
    all_long_time = []
    
    for i in plotting_sessions:
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        delay_data = block_data['delay'][i]
        num = num + 1
        
        short_delay_data = []
        long_delay_data = []
        short_trial_num = []
        long_trial_num = []
        short_id_all = []
        long_id_all = []
        len_block_short = []
        len_block_long = []
        
        if short_id[0] == 0:
            short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
        
        for j in range(len(short_id)):
            short_delay_data.append(block_data['delay'][i][short_id[j]])
            short_trial_num.append(np.arange(0,len(block_data['delay'][i][short_id[j]])))
            short_id_all.append(short_id[j]*np.ones(len(block_data['delay'][i][short_id[j]])))
            len_block_short.append(len(block_data['delay'][i][short_id[j]]))
                
        for j in range(len(long_id)):
            long_delay_data.append(block_data['delay'][i][long_id[j]])
            long_trial_num.append(np.arange(0,len(block_data['delay'][i][long_id[j]])))
            long_id_all.append(long_id[j]*np.ones(len(block_data['delay'][i][long_id[j]])))
            len_block_long.append(len(block_data['delay'][i][long_id[j]]))
        
        short_delay_data = np.concatenate(short_delay_data)
        long_delay_data = np.concatenate(long_delay_data)
        short_trial_num = np.concatenate(short_trial_num)
        long_trial_num = np.concatenate(long_trial_num)
        short_id_all = np.concatenate(short_id_all)
        long_id_all = np.concatenate(long_id_all)
        
        min_val = np.nanmin(short_delay_data)
        max_val = np.nanmax(short_delay_data)
        short_delay_data = [(x - min_val) / (max_val - min_val) for x in short_delay_data]
        all_short.append(short_delay_data)
        min_val = np.nanmin(long_delay_data)
        max_val = np.nanmax(long_delay_data)
        long_delay_data = [(x - min_val) / (max_val - min_val) for x in long_delay_data]
        all_long.append(long_delay_data)
        all_short_time.append(short_trial_num)
        all_long_time.append(long_trial_num)
        
        data_short = {
            'delay':short_delay_data,
            'time':short_trial_num,
            'block_id':short_id_all}
        df_short = pd.DataFrame(data_short)
        
        data_long = {
            'delay':long_delay_data,
            'time':long_trial_num,
            'block_id':long_id_all}
        df_long = pd.DataFrame(data_long)
        
        plot_delay_adat(axs[0] , df_short , num-1 , num_session , 'Delay Adaptation (Short)' , 0 )
        plot_delay_adat(axs[1] , df_long  , num-1 , num_session , 'Delay Adaptation (long)' , 0)
    
    all_short = np.concatenate(all_short)
    all_long = np.concatenate(all_long)
    all_short_time = np.concatenate(all_short_time)
    all_long_time = np.concatenate(all_long_time)
    data_short = {
        'delay':all_short,
        'time':all_short_time}
    df_short = pd.DataFrame(data_short)
    
    data_long = {
        'delay':all_long,
        'time':all_long_time}
    df_long = pd.DataFrame(data_long)
    
    plot_delay_adat(axs[0] , df_short , num-1 , num_session , 'Delay Adaptation (Short)' , 0 , all = 1)
    plot_delay_adat(axs[1] , df_long  , num-1 , num_session , 'Delay Adaptation (long)' , 0 ,  all = 1)
    
    
def run_delay_rewarded(axs , axs1 ,  session_data, block_data, st):
    
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
    
    
    mean_block_short = []
    mean_block_long = []
    cmap1 = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=len(plotting_sessions)+1)
    cmap2 = plt.cm.inferno
    color_short = []
    color_long = []
    upper_short = []
    upper_long = []
    lower_short = []
    lower_long = []
    slope_short = []
    slope_long = []
    slope_color = []
    initial = 0
    for i in plotting_sessions:
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        delay_data = block_data['delay'][i]
        lower = block_data['LowerBand'][i]
        upper = block_data['UpperBand'][i]
        short_x = []
        long_x = []
        
        if len(long_id) > 0:
            if short_id[0] == 0:
                short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
        
        for j in range(len(short_id)):
            temp = block_data['outcome'][i][short_id[j]]
            ind = [x for x in range(len(temp)) if temp[x] == 'Reward']
            delay_temp = [delay_data[short_id[j]][x] for x in ind]
            if not np.isnan(np.nanmean(delay_temp)):
                mean_block_short.append(np.nanmean(delay_temp))
                axs[0].errorbar(len(mean_block_short)-1 , mean_block_short[-1] , yerr = np.nanstd(delay_temp)/np.sqrt(np.count_nonzero(~np.isnan(delay_temp))) , color = cmap1(norm(num_session+1-num)) , alpha = 0.2)
                color_short.append(cmap1(norm(num_session+1-num)))
                upper_short.append(np.nanmean(upper[short_id[j]]))
                lower_short.append(np.nanmean(lower[short_id[j]]))
                if len(delay_data[short_id[j]]) > initial:
                    short_x.append(np.nanmean(delay_temp))
                else:
                    short_x.append(np.nanmean(delay_temp))
                
        for j in range(len(long_id)):
            temp = block_data['outcome'][i][long_id[j]]
            ind = [x for x in range(len(temp)) if temp[x] == 'Reward']
            delay_temp = [delay_data[long_id[j]][x] for x in ind]
            if not np.isnan(np.nanmean(delay_temp)):
                mean_block_long.append(np.nanmean(delay_temp))
                axs[1].errorbar(len(mean_block_long)-1 , mean_block_long[-1] , yerr = np.nanstd(delay_temp)/np.sqrt(np.count_nonzero(~np.isnan(delay_temp))) , color = cmap2(norm(num_session+1-num)) , alpha = 0.2)
                color_long.append(cmap2(norm(num_session+1-num)))
                upper_long.append(np.nanmean(upper[long_id[j]]))
                lower_long.append(np.nanmean(lower[long_id[j]]))
                if len(delay_data[long_id[j]]) > initial:
                    long_x.append(np.nanmean(delay_temp))
                else:
                    long_x.append(np.nanmean(delay_temp))
        
        if len(short_x) > 1:
            s , b = np.polyfit(np.arange(len(short_x)), np.array(short_x), 1)
            slope_short.append(s)
            axs[0].plot(np.arange(len(short_x))+len(color_short)-len(short_x), s*(np.arange(len(short_x)))+b, color = 'r', alpha = 0.3)
        else:
            slope_short.append(np.nan)
        
        if len(long_x) > 1:
            s , b = np.polyfit(np.arange(len(long_x)), np.array(long_x), 1)
            slope_long.append(s)
            axs[1].plot(np.arange(len(long_x))+len(color_long)-len(long_x), s*(np.arange(len(long_x)))+b, color = 'r', alpha = 0.3)
        else: 
            slope_long.append(np.nan)
        slope_color.append(cmap1(norm(num_session+1-num)))
                
        if i < len(outcomes) - 1:   
            axs[0].axvline(len(mean_block_short) - 0.5 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)  
            axs[1].axvline(len(mean_block_long) - 0.5 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)  
        num = num + 1

    axs[0].set_ylabel('Delay (s) (mean +/- sem)')
    axs[1].set_ylabel('Delay (s) (mean +/- sem)')
    
    axs[0].scatter(np.arange(len(mean_block_short)) , mean_block_short , color = color_short ,  s = 20 , edgecolors= 'gray')
    axs[1].scatter(np.arange(len(mean_block_long)) , mean_block_long , color = color_long ,  s = 20 , edgecolors= 'gray')
    
    axs[0].fill_between(np.arange(len(mean_block_short)) ,lower_short , upper_short, color = 'gray' , alpha = 0.1)
    axs[1].fill_between(np.arange(len(mean_block_long)) ,lower_long , upper_long, color = 'gray' , alpha = 0.1)
    axs1.scatter(slope_short , slope_long , color = slope_color ,  s = 20 )
    
    axs[0].set_xlabel('Blocks (concatenated sessions)')
    axs[1].set_xlabel('Blocks (concatenated sessions)')
    
    axs[0].fill_between(np.arange(len(mean_block_short)) ,lower_short , upper_short, color = 'gray' , alpha = 0.1)
    axs[1].fill_between(np.arange(len(mean_block_long)) ,lower_long , upper_long, color = 'gray' , alpha = 0.1)
    axs1.scatter(slope_short , slope_long , color = slope_color ,  s = 20 , edgecolors= 'gray')
    axs1.axvline(0 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)
    axs1.axhline(0 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)
    axs1.set_title('delay adaptation slope (Rewarded Trials)')
    #axs1.tick_params(tick1On=False)
    axs1.spines['right'].set_visible(False)
    axs1.spines['top'].set_visible(False)
    axs1.set_ylabel('long blocks slope (s/block)')
    axs1.set_xlabel('short blocks slope (s/block)')
    
    axs[0].set_title('Average Delay in each short block (Rewarded Trials)')
    axs[1].set_title('Average Delay in each long block (Rewarded Trials)')
    #axs[0].tick_params(tick1On=False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    #axs[1].tick_params(tick1On=False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].set_ylim([0.5 , 2])
   