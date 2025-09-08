# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 06:27:13 2025

@author: Sana
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:26:59 2025

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
  
def set_color(opto_tags , opto , control):
    color_all = []
    for i in opto_tags:
        if i == 0:
            color_all.append(control)
        else:
            color_all.append(opto)
    return color_all
    
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
    
def run_final_adaptation(axs , session_data , block_data, st):
    
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
                
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                if not block_data['Opto'][i][short_id[j]]:
                    delay_short.append(np.append(nan_pad, delay_data[short_id[j]][0:block_len]))
                else:
                    delay_short_opto.append(np.append(nan_pad, delay_data[short_id[j]][0:block_len]))
            else: 
                
                if not block_data['Opto'][i][short_id[j]]:
                    delay_short.append(delay_data[short_id[j]][-width:])
                else:
                    delay_short_opto.append(delay_data[short_id[j]][-width:])
                
        for j in range(len(long_id)):
            block_len = len(block_data['delay'][i][long_id[j]])
            
            if block_len< width:
               
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                if not block_data['Opto'][i][long_id[j]]:
                    delay_long.append(np.append(nan_pad, delay_data[long_id[j]][0:block_len]))
                else:
                    delay_long_opto.append(np.append(nan_pad, delay_data[long_id[j]][0:block_len]))
            else: 
                
                if not block_data['Opto'][i][long_id[j]]:
                    delay_long.append(delay_data[long_id[j]][-width:])
                else:
                    delay_long_opto.append(delay_data[long_id[j]][-width:])
        num = num + 1
    
    delay_short_mean = np.nanmean(np.array(delay_short), axis=0)
    delay_short_sem = np.nanstd(np.array(delay_short) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short))))
    axs.fill_between(np.arange(-width,0) ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'y' , alpha = 0.2)
    axs.plot(np.arange(-width,0), delay_short_mean , color = 'y', label = 'short')
    delay_long_mean = np.nanmean(np.array(delay_long), axis=0)
    delay_long_sem = np.nanstd(delay_long , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long))))
    axs.fill_between(np.arange(-width,0) ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'b' , alpha = 0.2)
    axs.plot(np.arange(-width,0), delay_long_mean , color = 'b', label = 'long')
    
    delay_short_mean = np.nanmean(np.array(delay_short_opto), axis=0)
    delay_short_sem = np.nanstd(np.array(delay_short_opto) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short_opto))))
    axs.fill_between(np.arange(-width,0) ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'y' , alpha = 0.2)
    axs.plot(np.arange(-width,0), delay_short_mean , color = 'y', label = 'short opto', linestyle = '--')
    delay_long_mean = np.nanmean(np.array(delay_long_opto), axis=0)
    delay_long_sem = np.nanstd(delay_long_opto , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long_opto))))
    axs.fill_between(np.arange(-width,0) ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'b' , alpha = 0.2)
    axs.plot(np.arange(-width,0), delay_long_mean , color = 'b', label = 'long opto' , linestyle = '--')
    
    axs.tick_params(labelleft=False)
    axs.tick_params(axis='y',which='both',left=False,right=False, labelbottom=False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.legend()
    axs.set_xlabel('trial number aligend on last trial')
    axs.set_title('Average of final trials delay') 
    
    
    
def run_initial_adaptation_added(axs ,axs1, session_data , block_data, st):
    
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
    
    width = 20
    pre = 10
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
                    temp = np.append(delay_data[short_id[j]][0:block_len] , nan_pad)
                    delay_short.append(np.append(delay_data[short_id[j]-1][-pre:], temp))
                else:
                    temp = np.append(delay_data[short_id[j]][0:block_len] , nan_pad)
                    delay_short_opto.append(np.append(delay_data[short_id[j]-1][-pre:], temp))
            else: 
                time_short.append(np.arange(width))
                if not block_data['Opto'][i][short_id[j]]:
                    delay_short.append(np.append(delay_data[short_id[j]-1][-pre:],delay_data[short_id[j]][0:width]))
                else:
                    delay_short_opto.append(np.append(delay_data[short_id[j]-1][-pre:],delay_data[short_id[j]][0:width]))
                
        for j in range(len(long_id)):
            block_len = len(block_data['delay'][i][long_id[j]])
            
            if block_len< width:
                time_long.append(np.arange(width))
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                if not block_data['Opto'][i][long_id[j]]:
                    temp = np.append(delay_data[long_id[j]][0:block_len] , nan_pad)
                    delay_long.append(np.append(delay_data[long_id[j]-1][-pre:] , temp))
                else:
                    temp = np.append(delay_data[long_id[j]][0:block_len] , nan_pad)
                    delay_long_opto.append(np.append(delay_data[long_id[j]-1][-pre:] , temp))
                #print(len(delay_long[-1]))
            else: 
                time_long.append(np.arange(width))
                #print(len(delay_data[long_id[j]]) <width)
                if not block_data['Opto'][i][long_id[j]]:
                    delay_long.append(np.append(delay_data[long_id[j]-1][-pre:] ,delay_data[long_id[j]][0:width]))
                else:
                    delay_long_opto.append(np.append(delay_data[long_id[j]-1][-pre:] , delay_data[long_id[j]][0:width]))
                if not len(delay_long[-1]) == width+pre:
                    nan_pad = np.zeros(width+pre-len(delay_long[-1]))
                    nan_pad[:] = np.nan
                    delay_long[-1] = np.append(nan_pad , delay_long[-1])
            
        num = num + 1
    
    delay_short_mean = np.nanmean(np.array(delay_short), axis=0)
    print(len(delay_short))
    print(len(delay_long))
    delay_short_sem = np.nanstd(np.array(delay_short) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short))))
    axs.fill_between(np.arange(-pre,width) ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'y' , alpha = 0.2)
    axs.plot(np.arange(-pre,width), delay_short_mean , color = 'y', label = 'short')
    # for i in range(len(delay_long)):
    #     print(len(delay_long[i]) , i)
    delay_long_mean = np.nanmean(np.array(delay_long), axis=0)
    delay_long_sem = np.nanstd(delay_long , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long))))
    axs.fill_between(np.arange(-pre,width) ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'b' , alpha = 0.2)
    axs.plot(np.arange(-pre,width), delay_long_mean , color = 'b', label = 'long')
    
    # delay_short_mean = np.nanmean(np.array(delay_short_opto), axis=0)
    # delay_short_sem = np.nanstd(np.array(delay_short_opto) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short_opto))))
    # axs.fill_between(np.arange(-pre,width) ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'dodgerblue' , alpha = 0.2)
    # axs.plot(np.arange(-pre,width), delay_short_mean , color = 'dodgerblue', label = 'short opto')
    # delay_long_mean = np.nanmean(np.array(delay_long_opto), axis=0)
    # delay_long_sem = np.nanstd(delay_long_opto , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long_opto))))
    # axs1.fill_between(np.arange(-pre,width) ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'dodgerblue' , alpha = 0.2)
    # axs1.plot(np.arange(-pre,width), delay_long_mean , color = 'dodgerblue', label = 'long opto')
    
    axs.axvline(x = 0, color = 'gray', linestyle='--')
    #axs.axvline(x = 8, color = 'gray', linestyle='--')
    # axs1.axvline(x = 0, color = 'gray', linestyle='--')
    # axs1.axvline(x = 8, color = 'gray', linestyle='--')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.legend()
    axs.set_xlabel('trial number')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average of trials delay')
    
    # axs1.spines['right'].set_visible(False)
    # axs1.spines['top'].set_visible(False)
    # axs1.legend()
    # axs1.set_xlabel('trial number')
    # axs1.set_ylabel('delay (s)')
    # axs1.set_title('Average of initial trials delay')
    
def run_final_adaptation_added(axs ,axs1, session_data , block_data, st):
    
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
    
    width = 20
    post = 10
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
                    temp = np.append(nan_pad, delay_data[short_id[j]][0:block_len])
                    if len(delay_data)>short_id[j]+1:
                        if len(delay_data[short_id[j]+1]) > post:
                            delay_short.append(np.append(temp, delay_data[short_id[j]+1][:post]))
                        else:
                            nan_pad1 = np.zeros(post-len(delay_data[short_id[j]+1]))
                            nan_pad1[:] = np.nan
                            temp1 = np.append(temp, delay_data[short_id[j]+1])
                            delay_short.append(np.append(temp1, nan_pad1))
                    else:
                        nan_pad1 = np.zeros(post)
                        nan_pad1[:] = np.nan
                        delay_short.append(np.append(temp, nan_pad1))
            else: 
                time_short.append(np.arange(width))
                if not block_data['Opto'][i][short_id[j]]:
                    if len(delay_data)>short_id[j]+1:
                        if len(delay_data[short_id[j]+1]) > post:
                            delay_short.append(np.append(delay_data[short_id[j]][-width:], delay_data[short_id[j]+1][:post]))
                        else:
                            nan_pad1 = np.zeros(post-len(delay_data[short_id[j]+1]))
                            nan_pad1[:] = np.nan
                            temp1 = np.append(delay_data[short_id[j]][-width:], delay_data[short_id[j]+1])
                            delay_short.append(np.append(temp1, nan_pad1))
                    else:
                        nan_pad1 = np.zeros(post)
                        nan_pad1[:] = np.nan
                        delay_short.append(np.append(delay_data[short_id[j]][-width:], nan_pad1))
            
                
        for j in range(len(long_id)):
            block_len = len(block_data['delay'][i][long_id[j]])
            
            if block_len< width:
                time_long.append(np.arange(width))
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                if not block_data['Opto'][i][long_id[j]]:
                    temp = np.append(nan_pad, delay_data[long_id[j]][0:block_len])
                    if len(delay_data)>long_id[j]+1:
                        if len(delay_data[long_id[j]+1]) > post:
                            delay_long.append(np.append(temp, delay_data[long_id[j]+1][:post]))
                        else:
                            nan_pad1 = np.zeros(post-len(delay_data[long_id[j]+1]))
                            nan_pad1[:] = np.nan
                            temp1 = np.append(temp, delay_data[long_id[j]+1])
                            delay_long.append(np.append(temp1, nan_pad1))
                    else:
                        nan_pad1 = np.zeros(post)
                        nan_pad1[:] = np.nan
                        delay_long.append(np.append(temp, nan_pad1))
            else: 
                time_long.append(np.arange(width))
                if not block_data['Opto'][i][long_id[j]]:
                    if len(delay_data)>long_id[j]+1:
                        if len(delay_data[long_id[j]+1]) > post:
                            delay_long.append(np.append(delay_data[long_id[j]][-width:], delay_data[long_id[j]+1][:post]))
                        else:
                            nan_pad1 = np.zeros(post-len(delay_data[long_id[j]+1]))
                            nan_pad1[:] = np.nan
                            temp1 = np.append(delay_data[long_id[j]][-width:], delay_data[long_id[j]+1])
                            delay_long.append(np.append(temp1, nan_pad1))
                    else:
                        nan_pad1 = np.zeros(post)
                        nan_pad1[:] = np.nan
                        delay_long.append(np.append(delay_data[long_id[j]][-width:], nan_pad1))
        num = num + 1
    
    print(len(delay_short))
    print(len(delay_long))
    delay_short_mean = np.nanmean(np.array(delay_short), axis=0)
    delay_short_sem = np.nanstd(np.array(delay_short) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short))))
    axs.fill_between(np.arange(-width, post) ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'y' , alpha = 0.2)
    axs.plot(np.arange(-width, post), delay_short_mean , color = 'y', label = 'short')
    delay_long_mean = np.nanmean(np.array(delay_long), axis=0)
    delay_long_sem = np.nanstd(delay_long , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long))))
    axs.fill_between(np.arange(-width, post) ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'b' , alpha = 0.2)
    axs.plot(np.arange(-width, post), delay_long_mean , color = 'b', label = 'long')
    
    # delay_short_mean = np.nanmean(np.array(delay_short_opto), axis=0)
    # delay_short_sem = np.nanstd(np.array(delay_short_opto) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short_opto))))
    # axs.fill_between(np.arange(-width, post) ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'dodgerblue' , alpha = 0.2)
    # axs.plot(np.arange(-width, post), delay_short_mean , color = 'dodgerblue', label = 'short opto')
    # delay_long_mean = np.nanmean(np.array(delay_long_opto), axis=0)
    # delay_long_sem = np.nanstd(delay_long_opto , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long_opto))))
    # axs1.fill_between(np.arange(-width, post) ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'dodgerblue' , alpha = 0.2)
    # axs1.plot(np.arange(-width, post), delay_long_mean , color = 'dodgerblue', label = 'long opto')
    
    axs.axvline(x = 0, color = 'gray', linestyle='--')
    #axs.axvline(x = 8, color = 'gray', linestyle='--')
    # axs1.axvline(x = 0, color = 'gray', linestyle='--')
    # axs1.axvline(x = 8, color = 'gray', linestyle='--')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.legend()
    axs.set_xlabel('trial number')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average of initial trials delay')
    
    # axs1.spines['right'].set_visible(False)
    # axs1.spines['top'].set_visible(False)
    # axs1.legend()
    # axs1.set_xlabel('trial number')
    # axs1.set_ylabel('delay (s)')
    # axs1.set_title('Average of initial trials delay')