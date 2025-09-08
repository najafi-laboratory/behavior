# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:06:01 2025

@author: saminnaji3
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
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

def count_label(session_data, opto, session_label, states, plotting_sessions, norm=True):
    counts = np.zeros((4 , len(states)))
    numtrials = np.zeros(4)
    
    for i in plotting_sessions:
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        for j in range(len(session_label[i])):
            if norm:
                if session_label[i][j] in states:
                    k = states.index(session_label[i][j])
                    if j > 0:
                        if opto[i][j-1] == 1:
                            if trial_types[j] == 1:
                                counts[1 , k] = counts[1 , k] + 1
                                numtrials[1] = numtrials[1] + 1   
                            else:
                                counts[3 ,k] = counts[3 , k] + 1
                                numtrials[3] = numtrials[3] + 1   
                        else:
                            if trial_types[j] == 1:
                                counts[0 , k] = counts[0 , k] + 1
                                numtrials[0] = numtrials[0] + 1   
                            else:
                                counts[2 , k] = counts[2 , k] + 1
                                numtrials[2] = numtrials[2] + 1  
                    else:
                        if trial_types[j] == 1:
                            counts[0, k] = counts[0 , k] + 1
                            numtrials[0] = numtrials[0] + 1   
                        else:
                            counts[2 , k] = counts[2 , k] + 1
                            numtrials[2] = numtrials[2] + 1  
                    
                    
    for j in range(len(states)):
        counts[0 , j] = counts[0  , j]/numtrials[0]
        counts[1 , j] = counts[1 , j]/numtrials[1]
        counts[2 , j] = counts[2 , j]/numtrials[2]
        counts[3 , j] = counts[3 , j]/numtrials[3]
    return counts

def count_push(session_data, plotting_sessions, properties, onset):
    
    num_session = len(plotting_sessions)
    delay_mean = np.zeros((2 , num_session))
    delay_std = np.zeros((2 , num_session))
    num = 0
    
    for i in plotting_sessions:
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        delay = session_data[properties][i]
        
        delay_short = []
        delay_long = []
        for j in range(len(delay)):
            if not onset == 0:
                sub =states = raw_data['RawEvents']['Trial'][j]['States']['VisDetect1'][0]
            else:
                sub = 0
            if trial_types[j] == 1:
                delay_short.append(delay[j]-sub)
            else:
                delay_long.append(delay[j]-sub)
        delay_mean[0 , num] = np.nanmean(np.array(delay_short))
        delay_mean[1 , num] = np.nanmean(np.array(delay_long))
        delay_std[0 , num] = np.nanstd(np.array(delay_short))/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short))))
        delay_std[1 , num] = np.nanstd(np.array(delay_long))/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long))))
        num = num + 1
    return delay_mean , delay_std

def count_push_opto(session_data,opto, plotting_sessions, properties, onset):
    
    num_session = len(plotting_sessions)
    delay_mean = np.zeros((4 , num_session))
    delay_std = np.zeros((4 , num_session))
    num = 0
    
    for i in plotting_sessions:
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        delay = session_data[properties][i]
        
        delay_short = []
        delay_long = []
        delay_short_opto = []
        delay_long_opto = []
        for j in range(len(delay)):
            if not onset == 0:
                sub =states = raw_data['RawEvents']['Trial'][j]['States']['VisDetect1'][0]
            else:
                sub = 0
            if j > 0:
                if opto[i][j-1] == 1:
                    if trial_types[j] == 1:
                        delay_short_opto.append(delay[j]-sub)  
                    else:
                        delay_long_opto.append(delay[j]-sub)  
                else:
                    if trial_types[j] == 1:
                        delay_short.append(delay[j]-sub)
                    else:
                        delay_long.append(delay[j]-sub)
            else:
                if trial_types[j] == 1:
                    delay_short.append(delay[j]-sub)
                else:
                    delay_long.append(delay[j]-sub)
            
            
            
        delay_mean[0 , num] = np.nanmean(np.array(delay_short))
        delay_mean[1 , num] = np.nanmean(np.array(delay_long))
        delay_std[0 , num] = np.nanstd(np.array(delay_short))/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short))))
        delay_std[1 , num] = np.nanstd(np.array(delay_long))/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long))))
        delay_mean[2 , num] = np.nanmean(np.array(delay_short_opto))
        delay_mean[3 , num] = np.nanmean(np.array(delay_long_opto))
        delay_std[2 , num] = np.nanstd(np.array(delay_short_opto))/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short_opto))))
        delay_std[3 , num] = np.nanstd(np.array(delay_long_opto))/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long_opto))))
        num = num + 1
    return delay_mean , delay_std

def run_delay(axs , session_data , properties, st, title, onset = 0):
    
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
        
    delay_mean , delay_std = count_push(session_data, plotting_sessions, properties, onset)
    session_id = np.arange(len(plotting_sessions)) + 1
    
    axs.scatter(session_id , delay_mean[0 , :] , color = 'y' , s = 5)
    axs.scatter(session_id , delay_mean[1 , :] , color = 'b' , s = 5)
    
    axs.plot(session_id , delay_mean[0 , :] , color = 'y' , label = 'Short')
    axs.plot(session_id , delay_mean[1 , :] , color = 'b' , label = 'Long')
    
    axs.fill_between(session_id , delay_mean[0 , :] - delay_std[0 , :] , delay_mean[0 , :] + delay_std[0 , :], color = 'y' , alpha = 0.2)
    axs.fill_between(session_id , delay_mean[1 , :] - delay_std[1 , :], delay_mean[1 , :] + delay_std[1 , :], color = 'b' , alpha = 0.2)
    
    if title == 'amp1':
        axs.set_ylabel('deg')
    elif title == 'velocity1':
        axs.set_ylabel('deg/delay')
    else:
        axs.set_ylabel('delay')
    
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

    
    axs.set_title(title)
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

def run_delay_opto(axs , session_data , properties, st, title, onset = 0):
    
    max_sessions=100
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    chemo_labels = session_data['chemo']
    isSelfTime = session_data['isSelfTimedMode']
    opto = session_data['session_opto_tag']
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
        
    delay_mean , delay_std = count_push_opto(session_data,opto, plotting_sessions, properties, onset)
    session_id = np.arange(len(plotting_sessions)) + 1
    
    axs.scatter(session_id , delay_mean[0 , :] , color = 'y' , s = 5)
    axs.scatter(session_id , delay_mean[1 , :] , color = 'b' , s = 5)
    axs.scatter(session_id , delay_mean[2 , :] , color = 'y' , s = 5)
    axs.scatter(session_id , delay_mean[3 , :] , color = 'b' , s = 5)
    axs.plot(session_id , delay_mean[0 , :] , color = 'y' , label = 'Short')
    axs.plot(session_id , delay_mean[1 , :] , color = 'b' , label = 'Long')
    axs.plot(session_id , delay_mean[2 , :] , color = 'y' , label = 'Short',linestyle = '--')
    axs.plot(session_id , delay_mean[3 , :] , color = 'b' , label = 'Long',linestyle = '--')
    
    axs.fill_between(session_id , delay_mean[0 , :] - delay_std[0 , :] , delay_mean[0 , :] + delay_std[0 , :], color = 'y' , alpha = 0.2)
    axs.fill_between(session_id , delay_mean[1 , :] - delay_std[1 , :], delay_mean[1 , :] + delay_std[1 , :], color = 'b' , alpha = 0.2)
    axs.fill_between(session_id , delay_mean[2 , :] - delay_std[2 , :] , delay_mean[2 , :] + delay_std[2 , :], color = 'y' , alpha = 0.2)
    axs.fill_between(session_id , delay_mean[3 , :] - delay_std[3 , :], delay_mean[3 , :] + delay_std[3 , :], color = 'b' , alpha = 0.2)
    
    if title == 'amp1':
        axs.set_ylabel('deg')
    elif title == 'velocity1':
        axs.set_ylabel('deg/delay')
    else:
        axs.set_ylabel('delay')
    
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

    
    axs.set_title(title)
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
def set_color(opto_tags , opto , control):
    color_all = []
    for i in opto_tags:
        if i == 0:
            color_all.append(control)
        else:
            color_all.append(opto)
    return color_all

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

def run_initial_adaptation(axs , session_data , block_data, st):
    
    max_sessions=100
    outcomes = session_data['outcomes']
    dates = session_data['dates']
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
    num = 0
    time_short = []
    time_long = []
    delay_short = []
    delay_long = []
    delay_short_opto = []
    delay_long_opto = []
    
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
            block_len = len(block_data['delay'][i][short_id[j]])
            
            if block_len< width:
                time_short.append(np.arange(width))
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                if not block_data['Opto'][i][short_id[j]]:
                    delay_short.append(np.append(delay_data[short_id[j]][0:block_len] , nan_pad))
                else:
                    delay_short_opto.append(np.append(delay_data[short_id[j]][0:block_len] , nan_pad))
            else: 
                time_short.append(np.arange(width))
                if not block_data['Opto'][i][short_id[j]]:
                    delay_short.append(delay_data[short_id[j]][0:width])
                else:
                    delay_short_opto.append(delay_data[short_id[j]][0:width])
                
        for j in range(len(long_id)):
            block_len = len(block_data['delay'][i][long_id[j]])
            
            if block_len< width:
                time_long.append(np.arange(width))
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                if not block_data['Opto'][i][long_id[j]]:
                    delay_long.append(np.append(delay_data[long_id[j]][0:block_len] , nan_pad))
                else:
                    delay_long_opto.append(np.append(delay_data[long_id[j]][0:block_len] , nan_pad))
            else: 
                time_long.append(np.arange(width))
                if not block_data['Opto'][i][long_id[j]]:
                    delay_long.append(delay_data[long_id[j]][0:width])
                else:
                    delay_long_opto.append(delay_data[long_id[j]][0:width])
        num = num + 1
    
    delay_short_mean = np.nanmean(np.array(delay_short), axis=0)
    delay_short_sem = np.nanstd(np.array(delay_short) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short))))
    axs.fill_between(np.arange(width) ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'y' , alpha = 0.2)
    axs.plot(np.arange(width), delay_short_mean , color = 'y', label = 'short')
    delay_long_mean = np.nanmean(np.array(delay_long), axis=0)
    delay_long_sem = np.nanstd(delay_long , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long))))
    axs.fill_between(np.arange(width) ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'b' , alpha = 0.2)
    axs.plot(np.arange(width), delay_long_mean , color = 'b', label = 'long')
    
    delay_short_mean = np.nanmean(np.array(delay_short_opto), axis=0)
    delay_short_sem = np.nanstd(np.array(delay_short_opto) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short_opto))))
    axs.fill_between(np.arange(width) ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'y' , alpha = 0.2)
    axs.plot(np.arange(width), delay_short_mean , color = 'y', label = 'short opto', linestyle = '--')
    delay_long_mean = np.nanmean(np.array(delay_long_opto), axis=0)
    delay_long_sem = np.nanstd(delay_long_opto , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long_opto))))
    axs.fill_between(np.arange(width) ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'b' , alpha = 0.2)
    axs.plot(np.arange(width), delay_long_mean , color = 'b', label = 'long opto' , linestyle = '--')
    
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.legend()
    axs.set_xlabel('trial number')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average of initial trials delay') 

def run_epoch_fix(axs , session_data , block_data, st):
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
    num_session = len(plotting_sessions)
    
    i = 1
    epoch_len = 8
    opto_tags_short = []
    opto_tags_long = []
    
    short_delay_first = []
    short_delay_last = []
    
    long_delay_first = []
    long_delay_last = []
    
    short_first_temp = []
    short_last_temp = []
    long_first_temp = []
    long_last_temp = []
    
    short_first_temp_opto = []
    short_last_temp_opto = []
    long_first_temp_opto = []
    long_last_temp_opto = []
    
    for i in plotting_sessions:
        
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        if short_id[0] == 0:
            short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
        for j in range(len(short_id)):
            current_block = block_data['delay'][i][short_id[j]]
            block_realize_num = epoch_len
            opto_tags_short.append(block_data['Opto'][i][short_id[j]])
            if np.isnan(block_realize_num):
                block_realize_num = 0
             
            if len(current_block)>1.5*epoch_len:
                short_delay_first.append(np.nanmean(current_block[:block_realize_num]))
                short_delay_last.append(np.nanmean(current_block[block_realize_num:]))
                if not block_data['Opto'][i][short_id[j]]:
                    short_first_temp.append(current_block[:block_realize_num])
                    short_last_temp.append(current_block[block_realize_num:])
                else:
                    short_first_temp_opto.append(current_block[:block_realize_num])
                    short_last_temp_opto.append(current_block[block_realize_num:])
                
                
        for j in range(len(long_id)):
            current_block = block_data['delay'][i][long_id[j]]
            block_realize_num = epoch_len
            opto_tags_long.append(block_data['Opto'][i][long_id[j]])
            if np.isnan(block_realize_num):
                block_realize_num = 0
               
            if len(current_block)>1.5*epoch_len:
                long_delay_first.append(np.nanmean(current_block[:block_realize_num]))
                long_delay_last.append(np.nanmean(current_block[block_realize_num:]))
                if not block_data['Opto'][i][long_id[j]]:
                    long_first_temp.append(current_block[:block_realize_num])
                    long_last_temp.append(current_block[block_realize_num:])
                else:
                    long_first_temp_opto.append(current_block[:block_realize_num])
                    long_last_temp_opto.append(current_block[block_realize_num:])
                    
    color_short = set_color(opto_tags_short , 'cyan' , 'k')       
    axs.scatter(np.ones(len(short_delay_first)) , short_delay_first, color = color_short[0:len(short_delay_first)] , alpha = 0.15, s = 5)
    axs.scatter(2*np.ones(len(short_delay_last)) , short_delay_last, color = color_short[0:len(short_delay_last)] , alpha = 0.15, s = 5)
    for i in range(len(short_delay_first)):
        axs.plot([1 ,2] , [short_delay_first[i] , short_delay_last[i]] , color = color_short[i] , alpha = 0.15)
        
    #printshort_first_temp, []))
    axs.scatter(1 , np.nanmean(sum(short_first_temp, [])), color = 'k' , s = 8)
    axs.scatter(2 , np.nanmean(sum(short_last_temp, [])), color = 'k' , s = 8)
    axs.scatter(1 , np.nanmean(sum(short_first_temp_opto, [])), color = 'cyan' , s = 8)
    axs.scatter(2 , np.nanmean(sum(short_last_temp_opto, [])), color = 'cyan' , s = 8)
    axs.plot([1 ,2] , [np.nanmean(sum(short_first_temp, [])) , np.nanmean(sum(short_last_temp, []))] , color = 'k')
    axs.plot([1 ,2] , [np.nanmean(sum(short_first_temp_opto, [])) , np.nanmean(sum(short_last_temp_opto, []))] , color = 'cyan')
    sem_first = np.nanstd(sum(short_first_temp, []))/np.sqrt(np.count_nonzero(~np.isnan(sum(short_first_temp, []))))
    sem_last = np.nanstd(sum(short_last_temp, []))/np.sqrt(np.count_nonzero(~np.isnan(sum(short_last_temp, []))))
    axs.errorbar(x = 1 , y = np.nanmean(sum(short_first_temp, [])) , yerr = sem_first , ecolor = 'k')
    axs.errorbar(x = 2 , y = np.nanmean(sum(short_last_temp, [])) , yerr =sem_last , ecolor = 'k')
    sem_first = np.nanstd(sum(short_first_temp_opto, []))/np.sqrt(np.count_nonzero(~np.isnan(sum(short_first_temp_opto, []))))
    sem_last = np.nanstd(sum(short_last_temp_opto, []))/np.sqrt(np.count_nonzero(~np.isnan(sum(short_last_temp_opto, []))))
    axs.errorbar(x = 1 , y = np.nanmean(sum(short_first_temp_opto, [])) , yerr = sem_first , ecolor = 'cyan')
    axs.errorbar(x = 2 , y = np.nanmean(sum(short_last_temp_opto, [])) , yerr = sem_last, ecolor = 'cyan')
    
    color_long = set_color(opto_tags_long , 'cyan' , 'k')       
    axs.scatter(4*np.ones(len(long_delay_first)) , long_delay_first, color = color_long , alpha = 0.15, s = 5)
    axs.scatter(5*np.ones(len(long_delay_last)) , long_delay_last, color = color_long , alpha = 0.15, s = 5)
    for i in range(len(long_delay_first)):
        axs.plot([4 , 5] , [long_delay_first[i] , long_delay_last[i]] , color = color_long[i] , alpha = 0.15)
    axs.scatter(4 , np.nanmean(sum(long_first_temp, [])), color = 'k' , s = 8)
    axs.scatter(5 , np.nanmean(sum(long_last_temp, [])), color = 'k' , s = 8)
    axs.scatter(4 , np.nanmean(sum(long_first_temp_opto, [])), color = 'cyan' , s = 8)
    axs.scatter(5 , np.nanmean(sum(long_last_temp_opto, [])), color = 'cyan' , s = 8)
    axs.plot([4 ,5] , [np.nanmean(sum(long_first_temp, [])) , np.nanmean(sum(long_last_temp, []))] , color = 'k')
    axs.plot([4 ,5] , [np.nanmean(sum(long_first_temp_opto, [])) , np.nanmean(sum(long_last_temp_opto, []))] , color = 'cyan')
    sem_first = np.nanstd(sum(long_first_temp, []))/np.sqrt(np.count_nonzero(~np.isnan(sum(long_first_temp, []))))
    sem_last = np.nanstd(sum(long_last_temp, []))/np.sqrt(np.count_nonzero(~np.isnan(sum(long_last_temp, []))))
    axs.errorbar(x = 4 , y = np.nanmean(sum(long_first_temp, [])) , yerr = sem_first , ecolor = 'k')
    axs.errorbar(x = 5 , y = np.nanmean(sum(long_last_temp, [])) , yerr = sem_last , ecolor = 'k')
    sem_first = np.nanstd(sum(long_first_temp_opto, []))/np.sqrt(np.count_nonzero(~np.isnan(sum(long_first_temp_opto, []))))
    sem_last = np.nanstd(sum(long_last_temp_opto, []))/np.sqrt(np.count_nonzero(~np.isnan(sum(long_last_temp_opto, []))))
    axs.errorbar(x = 4 , y = np.nanmean(sum(long_first_temp_opto, [])) , yerr = sem_first , ecolor = 'cyan')
    axs.errorbar(x = 5 , y = np.nanmean(sum(long_last_temp_opto, [])) , yerr = sem_last , ecolor = 'cyan')
    
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_ylabel('delay (s) (mean +/- sem)')
    axs.set_title('Average Delay for first and last adapted epoch') 
    
   
    axs.set_xticks([1 , 2 , 4 , 5])
    axs.set_xticklabels(['Short First' , 'Short Last' , 'Long First' , 'Long Last'], rotation='vertical')   
    
def outcome_all(axs , session_data ,  st = 0):
    
    max_sessions=100
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    num_session = len(dates)
    opto = session_data['session_opto_tag']
    
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    dates = dates[start_idx:]  
    new_dates = []
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
   
    counts = count_label(session_data, opto, outcomes, states , plotting_sessions)
    
    session_id = np.array([1])
    
    controls_bottom = np.cumsum(counts[0 , :])
    controls_bottom[1:] = controls_bottom[:-1]
    controls_bottom[0] = 0
    
    optos_bottom = np.cumsum(counts[1  , :])
    optos_bottom[1:] = optos_bottom[:-1]
    optos_bottom[0] = 0
    
    controll_bottom = np.cumsum(counts[2 , :])
    controll_bottom[1:] = controll_bottom[:-1]
    controll_bottom[0] = 0
    
    optol_bottom = np.cumsum(counts[3, :])
    optol_bottom[1:] = optol_bottom[:-1]
    optol_bottom[0] = 0
    
    width = 0.1
    dis = 0.2
    
    top_ticks = []
    
    axs.axhline(0.5 , linestyle = '--' , color = 'gray')
    
    for i in range(len(states)):
        axs.bar(
            session_id-width-dis/2 , counts[0,i],
            bottom=controls_bottom[i],
            width=width,
            color=colors[i],
            label=states_name[i])
        axs.bar(
            session_id -dis/2, counts[1,i],
            bottom=optos_bottom[i],
            width=width,
            color=colors[i])
        axs.bar(
            session_id+dis/2 , counts[2,i],
            bottom=controll_bottom[i],
            width=width,
            color=colors[i])
        axs.bar(
            session_id +width+dis/2, counts[3,i],
            bottom=optol_bottom[i],
            width=width,
            color=colors[i])

    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Training session')
    
    axs.set_ylabel('Outcome percentages')
    tick_index = np.append(session_id , session_id-width-dis/2)
    tick_index = np.append(tick_index , session_id-dis/2)
    tick_index = np.append(tick_index , session_id+dis/2)
    tick_index = np.append(tick_index , session_id+dis/2+width)
    axs.set_xticks(np.sort(tick_index))
    
    dates_label = []
    
    for i in session_id:
        dates_label.append('short (control)')
        dates_label.append('short (opto+1)')
        dates_label.append('Mega session')
        dates_label.append('long (control)')
        dates_label.append('long (opto+1)')
    axs.set_xticklabels(dates_label, rotation='vertical')
    
    
    
    secax = axs.secondary_xaxis('top')
    secax.set_xticks([])
    secax.set_xticklabels(top_ticks)
    secax.tick_params(labelsize =  8)
    secax.spines['top'].set_visible(False)
    secax.tick_params(tick1On=False)
    
    
    
    
    axs.set_title('Reward percentage for completed trials Mega session')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

def run_aligned_adaptation(axs , session_data , block_data, st, align = 0):
    
    max_sessions=100
    outcomes = session_data['outcomes']
    dates = session_data['dates']
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
    num = 0
    time_short = []
    time_long = []
    delay_short = []
    delay_long = []
    delay_short_opto = []
    delay_long_opto = []
    
    width = 6
    for i in plotting_sessions:
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        delay_data = block_data['delay'][i]
        opto = session_data['session_opto_tag'][i]
        start_stim = 2
        end_stim = 8
        
        if len(long_id) > 0:
            if short_id[0] == 0:
                short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
            
            
        for j in range(len(short_id)):
            block_len = len(block_data['delay'][i][short_id[j]])
            
            if block_len< width:
                
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                if not block_data['Opto'][i][short_id[j]]:
                    delay_short.append(np.append(delay_data[short_id[j]][start_stim:block_len] , nan_pad))
                else:
                    all_opto = np.where(np.array(opto[block_data['start'][i][short_id[j]] : block_data['end'][i][short_id[j]]]) == 1)[0]
                    if align:
                        start_stim = all_opto[0]+1
                        end_stim = start_stim + 6
                    else:
                        end_stim = all_opto[-1]+2
                        start_stim = end_stim-6
                    delay_short_opto.append(np.append(delay_data[short_id[j]][start_stim:block_len] , nan_pad))
            else: 
               
                if not block_data['Opto'][i][short_id[j]]:
                    delay_short.append(delay_data[short_id[j]][start_stim:end_stim])
                else:
                    #print(np.where(np.array(opto[block_data['start'][i][short_id[j]] : block_data['end'][i][short_id[j]]]) == 1))
                    all_opto = np.where(np.array(opto[block_data['start'][i][short_id[j]] : block_data['end'][i][short_id[j]]]) == 1)[0]
                    if align:
                        start_stim = all_opto[0]+1
                        end_stim = start_stim + 6
                    else:
                        end_stim = all_opto[-1]+2
                        start_stim = end_stim-6
                    delay_short_opto.append(delay_data[short_id[j]][start_stim:end_stim])
               
        for j in range(len(long_id)):
            block_len = len(block_data['delay'][i][long_id[j]])
            
            if block_len< width:
                
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                if not block_data['Opto'][i][long_id[j]]:
                    delay_long.append(np.append(delay_data[long_id[j]][start_stim:block_len] , nan_pad))
                else:
                    all_opto = np.where(np.array(opto[block_data['start'][i][long_id[j]] : block_data['end'][i][long_id[j]]]) == 1)[0]
                    if align:
                        start_stim = all_opto[0]+1
                        end_stim = start_stim + 6
                    else:
                        end_stim = all_opto[-1]+2
                        start_stim = end_stim-6
                    delay_long_opto.append(np.append(delay_data[long_id[j]][start_stim:block_len] , nan_pad))
            else: 
                
                    
                if not block_data['Opto'][i][long_id[j]]:
                    delay_long.append(delay_data[long_id[j]][start_stim:end_stim])
                else:
                    all_opto = np.where(np.array(opto[block_data['start'][i][long_id[j]] : block_data['end'][i][long_id[j]]]) == 1)[0]
                    if align:
                        start_stim = all_opto[0]+1
                        end_stim = start_stim + 6
                    else:
                        end_stim = all_opto[-1]+2
                        start_stim = end_stim-6
                    delay_long_opto.append(delay_data[long_id[j]][start_stim:end_stim])
        num = num + 1
    if align:
        time = np.arange(width)
    else:
        time = np.arange(width)-width
    delay_short_mean = np.nanmean(np.array(delay_short), axis=0)
    delay_short_sem = np.nanstd(np.array(delay_short) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short))))
    axs.fill_between(time ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'y' , alpha = 0.2)
    axs.plot(time, delay_short_mean , color = 'y', label = 'short')
    delay_long_mean = np.nanmean(np.array(delay_long), axis=0)
    delay_long_sem = np.nanstd(delay_long , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long))))
    axs.fill_between(time ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'b' , alpha = 0.2)
    axs.plot(time, delay_long_mean , color = 'b', label = 'long')
    
    delay_short_mean = np.nanmean(np.array(delay_short_opto), axis=0)
    delay_short_sem = np.nanstd(np.array(delay_short_opto) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short_opto))))
    axs.fill_between(time ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'y' , alpha = 0.2)
    axs.plot(time, delay_short_mean , color = 'y', label = 'short opto', linestyle = '--')
    delay_long_mean = np.nanmean(np.array(delay_long_opto), axis=0)
    delay_long_sem = np.nanstd(delay_long_opto , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long_opto))))
    axs.fill_between(time ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'b' , alpha = 0.2)
    axs.plot(time, delay_long_mean , color = 'b', label = 'long opto' , linestyle = '--')
    
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.legend()
    axs.set_xlabel('trial number')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average of aligned trials delay') 