# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:36:29 2025

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

def plot_average_indvi(axs ,df , c):
    #colors = ['#4CAF50','#FFB74D','pink','r','#64B5F6','#1976D2']
    
    
    w = 20
    df_sorted = df.sort_values(by='time')
    filtered_df = df_sorted[~df_sorted['delay'].isin([np.nan])]
    average = np.convolve(filtered_df['delay'], np.ones(w), 'valid') / w
    x_value = np.linspace(0,1,len(average))
    axs.plot(x_value , average, color= c) 
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.tick_params(tick1On=False)

def run_block_realize_new(axs , session_data , block_data, st):
    
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
    
    
    block_realize_short = []
    block_realize_long = []
    cmap = plt.cm.inferno
    max_color = len(plotting_sessions)+1
    norm = plt.Normalize(vmin=0, vmax=max_color)
    color_short = []
    color_long = []
    for i in plotting_sessions:
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        delay_data = block_data['delay'][i]
        
        
        if len(long_id) > 0:
            if short_id[0] == 0:
                short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
            
            
        for j in range(len(short_id)):
            block_realize_short.append(block_data['block_adaptation'][i][short_id[j]])
            color_short.append(cmap(norm(max_color-num)))
                
        for j in range(len(long_id)):
            block_realize_long.append(block_data['block_adaptation'][i][long_id[j]])
            color_long.append(cmap(norm(max_color-num)))
        if i < len(outcomes) - 1:   
            axs[0].axvline(len(block_realize_short) - 0.5 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)  
            axs[1].axvline(len(block_realize_long) - 0.5 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)  
        num = num + 1
    
    axs[0].set_ylabel('trial number')
    axs[1].set_ylabel('trial number')
    axs[0].set_xlabel('Blocks (concatenated sessions)')
    axs[1].set_xlabel('Blocks (concatenated sessions)')
    
    axs[0].scatter(np.arange(len(block_realize_short)) , block_realize_short , color = color_short , s = 20 , edgecolors= 'gray')
    axs[1].scatter(np.arange(len(block_realize_long)) , block_realize_long , color = color_long , s = 20, edgecolors= 'gray')
    #axs[0].plot(np.arange(len(len_block_short)) , len_block_short , color = 'gray')
    #axs[1].plot(np.arange(len(len_block_long)) , len_block_long , color = 'gray')
    
    axs[0].set_title('Number of trial where average delay decreased for short block')
    axs[1].set_title('Number of trial where average delay increased for long block')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
def run_block_realize(axs , session_data , block_data, st):
    
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
    
    
    block_realize_short = []
    block_realize_long = []
    cmap = plt.cm.inferno
    max_color = len(plotting_sessions)+1
    norm = plt.Normalize(vmin=0, vmax=max_color)
    color_short = []
    color_long = []
    for i in plotting_sessions:
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        delay_data = block_data['delay'][i]
        
        
        if len(long_id) > 0:
            if short_id[0] == 0:
                short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
            
            
        for j in range(len(short_id)):
            block_realize_short.append(block_data['block_realize'][i][short_id[j]])
            color_short.append(cmap(norm(max_color-num)))
                
        for j in range(len(long_id)):
            block_realize_long.append(block_data['block_realize'][i][long_id[j]])
            color_long.append(cmap(norm(max_color-num)))
        if i < len(outcomes) - 1:   
            axs[0].axvline(len(block_realize_short) - 0.5 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)  
            axs[1].axvline(len(block_realize_long) - 0.5 , linestyle = '--' , color = 'gray' , linewidth=1 , alpha = 0.3)  
        num = num + 1
    
    axs[0].set_ylabel('trial number')
    axs[1].set_ylabel('trial number')
    axs[0].set_xlabel('Blocks (concatenated sessions)')
    axs[1].set_xlabel('Blocks (concatenated sessions)')
    
    axs[0].scatter(np.arange(len(block_realize_short)) , block_realize_short , color = color_short , s = 20 , edgecolors= 'gray')
    axs[1].scatter(np.arange(len(block_realize_long)) , block_realize_long , color = color_long , s = 20, edgecolors= 'gray')
    #axs[0].plot(np.arange(len(len_block_short)) , len_block_short , color = 'gray')
    #axs[1].plot(np.arange(len(len_block_long)) , len_block_long , color = 'gray')
    
    axs[0].set_title('Adaptation trial number for short block')
    axs[1].set_title('Adaptation trial number for long block')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    

    
def run_partial_adaptation(axs , session_data , block_data, st ):
    
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
    
    
    
    block_realize_short = []
    block_realize_long = []
    cmap = plt.cm.inferno
    max_color = len(plotting_sessions)+1
    norm = plt.Normalize(vmin=0, vmax=max_color)
    color_short = []
    color_long = []
    
    time_short = []
    time_long = []
    delay_short = []
    delay_long = []
    win = 6
    width = 14
    for i in plotting_sessions:
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        delay_data = block_data['delay'][i]
        
        
        if len(long_id) > 0:
            if short_id[0] == 0:
                short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
            
            
        for j in range(len(short_id)):
            
            block_realize_num = block_data['block_realize'][i][short_id[j]]
            if np.isnan(block_realize_num):
                break
            elif block_realize_num + win< width:
                time_short.append(np.arange(width))
                nan_pad = np.zeros(width-block_realize_num-win)
                nan_pad[:] = np.nan
                delay_short.append(np.append(nan_pad , delay_data[short_id[j]][0:block_realize_num+win]))
                #axs[0].scatter(np.arange(width)-0.05*(j-len(short_id)/2)+win-width,delay_short[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
                
            elif block_realize_num + win> len(block_data['delay'][i][short_id[j]]):
                time_short.append(np.arange(width))
                nan_pad = np.zeros(-len(block_data['delay'][i][short_id[j]])+block_realize_num+win)
                nan_pad[:] = np.nan
                delay_short.append(np.append(delay_data[short_id[j]][block_realize_num+win-width:] , nan_pad))
                #print(len(delay_short[-1]))
                #axs[0].scatter(np.arange(width)-0.05*(j-len(short_id)/2)+win-width,delay_short[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
                
            else: 
                time_short.append(np.arange(width))
                delay_short.append(delay_data[short_id[j]][block_realize_num+win-width: block_realize_num+win])
                #axs[0].scatter(np.arange(width)-0.05*(j-len(short_id)/2)+win-width , delay_short[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
                
                
        for j in range(len(long_id)):
            block_realize_num = block_data['block_realize'][i][long_id[j]]
            if np.isnan(block_realize_num):
                break
            elif block_realize_num + win< width:
                time_long.append(np.arange(width))
                nan_pad = np.zeros(width-block_realize_num-win)
                nan_pad[:] = np.nan
                delay_long.append(np.append(nan_pad , delay_data[long_id[j]][0:block_realize_num+win]))
                #axs[0].scatter(np.arange(width)-0.05*(j-len(long_id)/2)+win-width,delay_long[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
                
            elif block_realize_num + win> len(block_data['delay'][i][long_id[j]]):
                time_long.append(np.arange(width))
                nan_pad = np.zeros(-len(block_data['delay'][i][long_id[j]])+block_realize_num+win)
                nan_pad[:] = np.nan
                delay_long.append(np.append(delay_data[long_id[j]][block_realize_num+win-width:] , nan_pad))
                #axs[0].scatter(np.arange(width)-0.05*(j-len(long_id)/2)+win-width,delay_long[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
                
            else: 
                time_long.append(np.arange(width))
                delay_long.append(delay_data[long_id[j]][block_realize_num+win-width: block_realize_num+win])
                #axs[1].scatter(np.arange(width)-0.05*(j-len(long_id)/2)+win-width,delay_long[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
        
        num = num + 1
    #print(np.array(delay_short).shape)
    #print(np.nanmean(np.array(delay_short), axis=0))
    # axs[0].plot(np.arange(width)+win-width, np.nanmean(np.array(delay_short), axis=0) , color = 'k')
    # axs[1].plot(np.arange(width)+win-width, np.nanmean(np.array(delay_long), axis=0) , color = 'k')
    # axs[0].spines['right'].set_visible(False)
    # axs[0].spines['top'].set_visible(False)
    # axs[1].spines['right'].set_visible(False)
    # axs[1].spines['top'].set_visible(False)
    
    delay_short_mean = np.nanmean(np.array(delay_short), axis=0)
    delay_short_sem = np.nanstd(np.array(delay_short) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short))))
    axs.fill_between(np.arange(width)+win-width ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'y' , alpha = 0.2)
    axs.plot(np.arange(width)+win-width, delay_short_mean , color = 'y', label = 'short')
    delay_long_mean = np.nanmean(np.array(delay_long), axis=0)
    delay_long_sem = np.nanstd(delay_long , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long))))
    axs.fill_between(np.arange(width)+win-width ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'b' , alpha = 0.2)
    #print('now :' ,delay_long_mean , np.arange(width)+win-width)
    axs.plot(np.arange(width)+win-width, delay_long_mean , color = 'b', label = 'long')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.legend()
    
    # axs[0].set_xlabel('trial number')
    # axs[1].set_xlabel('trial number')
    axs.set_xlabel('trial number')
    
    # axs[0].set_ylabel('delay (s)')
    # axs[1].set_ylabel('delay (s)')
    axs.set_ylabel('delay (s)')
    
    # axs[0].set_title('Realized window delay for short blocks')
    # axs[1].set_title('Realized window delay for long blocks')
    axs.set_title('Average delay aligned on adaptation trial')
    
    
def run_initial_adaptation(axs , session_data , block_data, st):
    
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
    
    
    
    block_realize_short = []
    block_realize_long = []
    cmap = plt.cm.inferno
    max_color = len(plotting_sessions)+1
    norm = plt.Normalize(vmin=0, vmax=max_color)
    color_short = []
    color_long = []
    
    time_short = []
    time_long = []
    delay_short = []
    delay_long = []
    win = 6
    width = 14
    for i in plotting_sessions:
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        delay_data = block_data['delay'][i]
        
        
        if len(long_id) > 0:
            if short_id[0] == 0:
                short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
            
            
        for j in range(len(short_id)):
            
            block_realize_num = block_data['block_realize'][i][short_id[j]]
            block_len = len(block_data['delay'][i][short_id[j]])
            # if np.isnan(block_realize_num):
            #     break
            if block_len< width:
                time_short.append(np.arange(width))
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                delay_short.append(np.append(delay_data[short_id[j]][0:block_len] , nan_pad))
                #axs[0].scatter(np.arange(width)-0.05*(j-len(short_id)/2),delay_short[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
            else: 
                time_short.append(np.arange(width))
                delay_short.append(delay_data[short_id[j]][0:width])
                #axs[0].scatter(np.arange(width)-0.05*(j-len(short_id)/2),delay_short[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
                
                
        for j in range(len(long_id)):
            block_realize_num = block_data['block_realize'][i][long_id[j]]
            block_len = len(block_data['delay'][i][long_id[j]])
            # if np.isnan(block_realize_num):
            #     break
            if block_len< width:
                time_long.append(np.arange(width))
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                delay_long.append(np.append(delay_data[long_id[j]][0:block_len] , nan_pad))
                #axs[1].scatter(np.arange(width)-0.05*(j-len(long_id)/2),delay_long[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
            else: 
                time_long.append(np.arange(width))
                delay_long.append(delay_data[long_id[j]][0:width])
                #axs[1].scatter(np.arange(width)-0.05*(j-len(long_id)/2),delay_long[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
        
        num = num + 1
    
    # axs[0].plot(np.arange(width), np.nanmean(np.array(delay_short), axis=0) , color = 'k')
    # axs[1].plot(np.arange(width), np.nanmean(np.array(delay_long), axis=0) , color = 'k')
    # axs[0].spines['right'].set_visible(False)
    # axs[0].spines['top'].set_visible(False)
    # axs[1].spines['right'].set_visible(False)
    # axs[1].spines['top'].set_visible(False)
    
    # for i in range(len(delay_short)):
    #     print(len(delay_short[i]))
    # print(delay_short[0])
    delay_short_mean = np.nanmean(np.array(delay_short), axis=0)
    delay_short_sem = np.nanstd(np.array(delay_short) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short))))
    axs.fill_between(np.arange(width) ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'y' , alpha = 0.2)
    #print(delay_short_mean)
    #print(np.arange(width))
    axs.plot(np.arange(width), delay_short_mean , color = 'y', label = 'short')
    delay_long_mean = np.nanmean(np.array(delay_long), axis=0)
    delay_long_sem = np.nanstd(delay_long , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long))))
    axs.fill_between(np.arange(width) ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'b' , alpha = 0.2)
    axs.plot(np.arange(width), delay_long_mean , color = 'b', label = 'long')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.legend()
    
    # axs[0].set_xlabel('trial number')
    # axs[1].set_xlabel('trial number')
    axs.set_xlabel('trial number')
    
    # axs[0].set_ylabel('delay (s)')
    # axs[1].set_ylabel('delay (s)')
    axs.set_ylabel('delay (s)')
    
    # axs[0].set_title('Initial trials delay for short blocks')
    # axs[1].set_title('Initial trials delay for long blocks')
    axs.set_title('Average of initial trials delay')
    
    
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
        #print(dates[i])
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
            block_realize_num = block_data['block_realize'][i][short_id[j]]
            #print(np.nanmean(current_block[:block_realize_num]) , np.nanmean(current_block[block_realize_num:]))
            if np.isnan(block_realize_num):
                block_realize_num = 0
             
            if len(current_block)>1.5*epoch_len:
                short_delay_first.append(np.nanmean(current_block[:block_realize_num]))
                short_delay_last.append(np.nanmean(current_block[block_realize_num:]))
                short_first_temp.append(current_block[:block_realize_num])
                short_last_temp.append(current_block[block_realize_num:])
                
                
        for j in range(len(long_id)):
            current_block = block_data['delay'][i][long_id[j]]
            block_realize_num = block_data['block_realize'][i][long_id[j]]
            if np.isnan(block_realize_num):
                block_realize_num = 0
               
            if len(current_block)>1.5*epoch_len:
                long_delay_first.append(np.nanmean(current_block[:block_realize_num]))
                long_delay_last.append(np.nanmean(current_block[block_realize_num:]))
                long_first_temp.append(current_block[:block_realize_num])
                long_last_temp.append(current_block[block_realize_num:])
                
        short_first1.append(np.nanmean(short_delay_first))
        short_last1.append(np.nanmean(short_delay_last))
        long_first1.append(np.nanmean(long_delay_first))
        long_last1.append(np.nanmean(long_delay_last))
        
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
    axs.set_title('Average Delay for first and last adapted epoch (based on adaptation trial)') 
    
   
    axs.set_xticks([0 , 1.5 , 4.5 , 6])
    axs.set_xticklabels(['Short First' , 'Short Last' , 'Long First' , 'Long Last'], rotation='vertical')
    
    
def run_epoch_new(axs , session_data , block_data, st):
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
            block_realize_num = block_data['block_adaptation'][i][short_id[j]]
            #print(np.nanmean(current_block[:block_realize_num]) , np.nanmean(current_block[block_realize_num:]))
            if np.isnan(block_realize_num):
                block_realize_num = 0
             
            if len(current_block)>1.5*epoch_len:
                short_delay_first.append(np.nanmean(current_block[:block_realize_num]))
                short_delay_last.append(np.nanmean(current_block[block_realize_num:]))
                short_first_temp.append(current_block[:block_realize_num])
                short_last_temp.append(current_block[block_realize_num:])
                
                
        for j in range(len(long_id)):
            current_block = block_data['delay'][i][long_id[j]]
            block_realize_num = block_data['block_adaptation'][i][long_id[j]]
            if np.isnan(block_realize_num):
                block_realize_num = 0
               
            if len(current_block)>1.5*epoch_len:
                long_delay_first.append(np.nanmean(current_block[:block_realize_num]))
                long_delay_last.append(np.nanmean(current_block[block_realize_num:]))
                long_first_temp.append(current_block[:block_realize_num])
                long_last_temp.append(current_block[block_realize_num:])
                
        short_first1.append(np.nanmean(short_delay_first))
        short_last1.append(np.nanmean(short_delay_last))
        long_first1.append(np.nanmean(long_delay_first))
        long_last1.append(np.nanmean(long_delay_last))
        
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
    axs.set_title('Average Delay for first and last adapted epoch (based on average comparison)') 
    
   
    axs.set_xticks([0 , 1.5 , 4.5 , 6])
    axs.set_xticklabels(['Short First' , 'Short Last' , 'Long First' , 'Long Last'], rotation='vertical')
    
    
def run_partial_adaptation_new(axs , session_data , block_data, st):
    
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
    
    
    
    block_realize_short = []
    block_realize_long = []
    cmap = plt.cm.inferno
    max_color = len(plotting_sessions)+1
    norm = plt.Normalize(vmin=0, vmax=max_color)
    color_short = []
    color_long = []
    
    time_short = []
    time_long = []
    delay_short = []
    delay_long = []
    win = 6
    width = 14
    for i in plotting_sessions:
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        delay_data = block_data['delay'][i]
        
        
        if len(long_id) > 0:
            if short_id[0] == 0:
                short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
            
            
        for j in range(len(short_id)):
            
            block_realize_num = block_data['block_adaptation'][i][short_id[j]]
            if np.isnan(block_realize_num):
                break
            elif block_realize_num + win< width:
                time_short.append(np.arange(width))
                nan_pad = np.zeros(width-block_realize_num-win)
                nan_pad[:] = np.nan
                delay_short.append(np.append(nan_pad , delay_data[short_id[j]][0:block_realize_num+win]))
                #axs[0].scatter(np.arange(width)-0.05*(j-len(short_id)/2)+win-width,delay_short[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
                
            elif block_realize_num + win> len(block_data['delay'][i][short_id[j]]):
                time_short.append(np.arange(width))
                nan_pad = np.zeros(-len(block_data['delay'][i][short_id[j]])+block_realize_num+win)
                nan_pad[:] = np.nan
                delay_short.append(np.append(delay_data[short_id[j]][block_realize_num+win-width:] , nan_pad))
                #print(len(delay_short[-1]))
                #axs[0].scatter(np.arange(width)-0.05*(j-len(short_id)/2)+win-width,delay_short[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
                
            else: 
                time_short.append(np.arange(width))
                delay_short.append(delay_data[short_id[j]][block_realize_num+win-width: block_realize_num+win])
                #axs[0].scatter(np.arange(width)-0.05*(j-len(short_id)/2)+win-width , delay_short[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
                
                
        for j in range(len(long_id)):
            block_realize_num = block_data['block_adaptation'][i][long_id[j]]
            if np.isnan(block_realize_num):
                break
            elif block_realize_num + win< width:
                time_long.append(np.arange(width))
                nan_pad = np.zeros(width-block_realize_num-win)
                nan_pad[:] = np.nan
                delay_long.append(np.append(nan_pad , delay_data[long_id[j]][0:block_realize_num+win]))
                #axs[0].scatter(np.arange(width)-0.05*(j-len(long_id)/2)+win-width,delay_long[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
                
            elif block_realize_num + win> len(block_data['delay'][i][long_id[j]]):
                time_long.append(np.arange(width))
                nan_pad = np.zeros(-len(block_data['delay'][i][long_id[j]])+block_realize_num+win)
                nan_pad[:] = np.nan
                delay_long.append(np.append(delay_data[long_id[j]][block_realize_num+win-width:] , nan_pad))
                #axs[0].scatter(np.arange(width)-0.05*(j-len(long_id)/2)+win-width,delay_long[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
                
            else: 
                time_long.append(np.arange(width))
                delay_long.append(delay_data[long_id[j]][block_realize_num+win-width: block_realize_num+win])
                #axs[1].scatter(np.arange(width)-0.05*(j-len(long_id)/2)+win-width,delay_long[-1] ,color = cmap(norm(num_session+1-num)), s = 3)
        
        num = num + 1
    #print(np.array(delay_short).shape)
    #print(np.nanmean(np.array(delay_short), axis=0))
    # axs[0].plot(np.arange(width)+win-width, np.nanmean(np.array(delay_short), axis=0) , color = 'k')
    # axs[1].plot(np.arange(width)+win-width, np.nanmean(np.array(delay_long), axis=0) , color = 'k')
    # axs[0].spines['right'].set_visible(False)
    # axs[0].spines['top'].set_visible(False)
    # axs[1].spines['right'].set_visible(False)
    # axs[1].spines['top'].set_visible(False)
    
    delay_short_mean = np.nanmean(np.array(delay_short), axis=0)
    delay_short_sem = np.nanstd(np.array(delay_short) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short))))
    axs.fill_between(np.arange(width)+win-width ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'y' , alpha = 0.2)
    axs.plot(np.arange(width)+win-width, delay_short_mean , color = 'y', label = 'short')
    delay_long_mean = np.nanmean(np.array(delay_long), axis=0)
    delay_long_sem = np.nanstd(delay_long , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long))))
    axs.fill_between(np.arange(width)+win-width ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'b' , alpha = 0.2)
    axs.plot(np.arange(width)+win-width, delay_long_mean , color = 'b', label = 'long')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.legend()
    
    # axs[0].set_xlabel('trial number')
    # axs[1].set_xlabel('trial number')
    axs.set_xlabel('trial number')
    
    # axs[0].set_ylabel('delay (s)')
    # axs[1].set_ylabel('delay (s)')
    axs.set_ylabel('delay (s)')
    
    # axs[0].set_title('Realized window delay for short blocks')
    # axs[1].set_title('Realized window delay for long blocks')
    axs.set_title('Average delay aligned on mean comparison')
    