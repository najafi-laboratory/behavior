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
    

def count_label(session_data, opto, session_label, states, plotting_sessions, block, norm=True):
    num_session = len(plotting_sessions)
    counts = np.zeros((4 , num_session, len(states)))
    numtrials = np.zeros((4 , num_session))
    num = 0
    sub = block-1
    for i in plotting_sessions:
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        for j in range(len(session_label[i])):
            if norm:
                if session_label[i][j] in states:
                    k = states.index(session_label[i][j])
                    if j > 0:
                        if opto[i][j+sub] == 1:
                            if trial_types[j] == 1:
                                counts[1 , num , k] = counts[1 , num , k] + 1
                                numtrials[1 , num] = numtrials[1 , num] + 1   
                            else:
                                counts[3 , num , k] = counts[3 , num , k] + 1
                                numtrials[3 , num] = numtrials[3 , num] + 1   
                        else:
                            if trial_types[j] == 1:
                                counts[0 , num , k] = counts[0 , num , k] + 1
                                numtrials[0 , num] = numtrials[0 , num] + 1   
                            else:
                                counts[2 , num , k] = counts[2 , num , k] + 1
                                numtrials[2 , num] = numtrials[2 , num] + 1  
                    else:
                        if trial_types[j] == 1:
                            counts[0 , num , k] = counts[0 , num , k] + 1
                            numtrials[0 , num] = numtrials[0 , num] + 1   
                        else:
                            counts[2 , num , k] = counts[2 , num , k] + 1
                            numtrials[2 , num] = numtrials[2 , num] + 1  
                    
                    
        for j in range(len(states)):
            counts[0 , num , j] = counts[0 , num , j]/numtrials[0 , num]
            counts[1 , num , j] = counts[1 , num , j]/numtrials[1 , num]
            counts[2 , num , j] = counts[2 , num , j]/numtrials[2 , num]
            counts[3 , num , j] = counts[3 , num , j]/numtrials[3 , num]
        num = num + 1
    return counts
def count_grand_block_label(session_data, opto, session_label, states, plotting_sessions, block, norm=True):
    num_session = len(plotting_sessions)
    counts = np.zeros((4, len(states)))
    numtrials = np.zeros(4)
    num = 0
    sub = block-1
    for i in plotting_sessions:
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        for j in range(len(session_label[i])):
            if norm:
                if session_label[i][j] in states:
                    k = states.index(session_label[i][j])
                    if j > 0:
                        if opto[i][j+sub] == 1:
                            if trial_types[j] == 1:
                                counts[1 , k] = counts[1 , k] + 1
                                numtrials[1] = numtrials[1] + 1   
                            else:
                                counts[3 , k] = counts[3 , k] + 1
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
                            counts[0 , k] = counts[0 , k] + 1
                            numtrials[0] = numtrials[0] + 1   
                        else:
                            counts[2, k] = counts[2 , k] + 1
                            numtrials[2 ] = numtrials[2] + 1  
                    
                    
    for j in range(len(states)):
        counts[0 , j] = counts[0 , j]/numtrials[0]
        counts[1 , j] = counts[1 , j]/numtrials[1]
        counts[2 , j] = counts[2 , j]/numtrials[2]
        counts[3 , j] = counts[3 , j]/numtrials[3]
        
    return counts
def count_grand_label(session_data, opto, session_label, states, plotting_sessions, norm=True):
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

def outcome(axs , session_data ,delay_data, block = 0, st = 0):
    
    max_sessions=100
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    num_session = len(dates)
    isSelfTime = session_data['isSelfTimedMode']
    opto = session_data['session_opto_tag']
    opto_block = delay_data['block_opto']
    
    if block:
        opto_label = opto
    else:
        opto_label = opto_block
    
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
    dates1 = []
    num = 0
    
    counts = count_label(session_data, opto_label, outcomes, states , plotting_sessions,block)
    
    session_id = np.arange(len(plotting_sessions)) + 1
    
    controls_bottom = np.cumsum(counts[0 , : , :], axis=1)
    controls_bottom[:,1:] = controls_bottom[:,:-1]
    controls_bottom[:,0] = 0
    
    optos_bottom = np.cumsum(counts[1 , : , :], axis=1)
    optos_bottom[:,1:] = optos_bottom[:,:-1]
    optos_bottom[:,0] = 0
    
    controll_bottom = np.cumsum(counts[2 , : , :], axis=1)
    controll_bottom[:,1:] = controll_bottom[:,:-1]
    controll_bottom[:,0] = 0
    
    optol_bottom = np.cumsum(counts[3 , : , :], axis=1)
    optol_bottom[:,1:] = optol_bottom[:,:-1]
    optol_bottom[:,0] = 0
    
    width = 0.1
    dis = 0.2
    
    top_ticks = []
    
    axs.axhline(0.5 , linestyle = '--' , color = 'gray')
    
    for i in range(len(states)):
        axs.bar(
            session_id-width-dis/2 , counts[0,:,i],
            bottom=controls_bottom[:,i],
            width=width,
            color=colors[i],
            label=states_name[i])
        axs.bar(
            session_id -dis/2, counts[1,:,i],
            bottom=optos_bottom[:,i],
            width=width,
            color=colors[i])
        axs.bar(
            session_id+dis/2 , counts[2,:,i],
            bottom=controll_bottom[:,i],
            width=width,
            color=colors[i])
        axs.bar(
            session_id +width+dis/2, counts[3,:,i],
            bottom=optol_bottom[:,i],
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
    
    for i in plotting_sessions:
        dates_label.append('short (control)')
        if not block:
            dates_label.append('short (opto+1)')
        else:
            dates_label.append('short (opto)')
        if isSelfTime[i][5] == 1:
            dates_label.append(dates[i] + '-ST')
        else:
            dates_label.append(dates[i] + '-VG')
        dates_label.append('long (control)')
        if not block:
            dates_label.append('long (opto+1)')
        else:
            dates_label.append('long (opto)')
    axs.set_xticklabels(dates_label, rotation='vertical', fontsize=5)
    
    secax = axs.secondary_xaxis('top')
    secax.set_xticks([])
    secax.set_xticklabels(top_ticks)
    secax.tick_params(labelsize =  4)
    secax.spines['top'].set_visible(False)
    secax.tick_params(tick1On=False)
    axs.set_title('Reward percentage for completed trials across sessions')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
def outcome_all(axs , session_data , block = 0,  st = 0):
    
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
    
    dates_label = []
    session_id = np.array([1])
    for i in session_id:
        dates_label.append('short (control)')
        dates_label.append('short (opto+1)')
        dates_label.append('Mega session')
        dates_label.append('long (control)')
        dates_label.append('long (opto+1)')
    axs.set_title('Reward percentage for completed trials Mega session')
    if block == 0:
        counts = count_grand_label(session_data, opto, outcomes, states , plotting_sessions)
    else:
        counts = count_grand_block_label(session_data, opto, outcomes, states , plotting_sessions , block)
        axs.set_title('Reward percentage for completed trials Mega blocks')
        dates_label = []
        for i in session_id:
            dates_label.append('short (control)')
            dates_label.append('short (opto)')
            dates_label.append('Mega session')
            dates_label.append('long (control)')
            dates_label.append('long (opto)')

    
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
    axs.set_xticklabels(dates_label, rotation='vertical')
    
    
    
    secax = axs.secondary_xaxis('top')
    secax.set_xticks([])
    secax.set_xticklabels(top_ticks)
    secax.tick_params(labelsize =  8)
    secax.spines['top'].set_visible(False)
    secax.tick_params(tick1On=False)
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
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
    epoch_len = 11
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
    axs.scatter(np.ones(len(short_delay_first)) , short_delay_first, color = color_short[0:len(short_delay_first)] , alpha = 0.1, s = 5)
    axs.scatter(2*np.ones(len(short_delay_last)) , short_delay_last, color = color_short[0:len(short_delay_last)] , alpha = 0.1, s = 5)
    for i in range(len(short_delay_first)):
        axs.plot([1 ,2] , [short_delay_first[i] , short_delay_last[i]] , color = color_short[i] , alpha = 0.1)
    #print(len(short_first_temp) , short_first_temp)
    # for i in range(len(short_first_temp)):
    #     print(short_first_temp[i].shape)
    #     print()
    # print(short_first_temp)
    #short_first_temp = np.array(short_first_temp)
    #short_first_temp = [sublist for sublist in short_first_temp if sublist]
    axs.scatter(1 , np.nanmean(np.concatenate(short_first_temp)), color = 'k' , s = 8)
    axs.scatter(2 , np.nanmean(np.concatenate(short_last_temp)), color = 'k' , s = 8)
    axs.scatter(1 , np.nanmean(np.concatenate(short_first_temp_opto)), color = 'cyan' , s = 8)
    axs.scatter(2 , np.nanmean(np.concatenate(short_last_temp_opto)), color = 'cyan' , s = 8)
    axs.plot([1 ,2] , [np.nanmean(np.concatenate(short_first_temp)) , np.nanmean(np.concatenate(short_last_temp))] , color = 'k')
    axs.plot([1 ,2] , [np.nanmean(np.concatenate(short_first_temp_opto)) , np.nanmean(np.concatenate(short_last_temp_opto))] , color = 'cyan')
    sem_first = np.nanstd(np.concatenate(short_first_temp))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(short_first_temp))))
    sem_last = np.nanstd(np.concatenate(short_last_temp))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(short_last_temp))))
    axs.errorbar(x = 1 , y = np.nanmean(np.concatenate(short_first_temp)) , yerr = sem_first , ecolor = 'k')
    axs.errorbar(x = 2 , y = np.nanmean(np.concatenate(short_last_temp)) , yerr =sem_last , ecolor = 'k')
    sem_first = np.nanstd(np.concatenate(short_first_temp_opto))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(short_first_temp_opto))))
    sem_last = np.nanstd(np.concatenate(short_last_temp_opto))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(short_last_temp_opto))))
    axs.errorbar(x = 1 , y = np.nanmean(np.concatenate(short_first_temp_opto)) , yerr = sem_first , ecolor = 'cyan')
    axs.errorbar(x = 2 , y = np.nanmean(np.concatenate(short_last_temp_opto)) , yerr = sem_last, ecolor = 'cyan')
    
    color_long = set_color(opto_tags_long , 'cyan' , 'k')       
    axs.scatter(4*np.ones(len(long_delay_first)) , long_delay_first, color = color_long[: min(len(long_delay_first), len(color_long))] , alpha = 0.1, s = 5)
    axs.scatter(5*np.ones(len(long_delay_last)) , long_delay_last, color = color_long[: min(len(long_delay_first), len(color_long))] , alpha = 0.1, s = 5)
    for i in range(len(long_delay_first)):
        axs.plot([4 , 5] , [long_delay_first[i] , long_delay_last[i]] , color = color_long[i] , alpha = 0.1)
    axs.scatter(4 , np.nanmean(np.concatenate(long_first_temp)), color = 'k' , s = 8)
    axs.scatter(5 , np.nanmean(np.concatenate(long_last_temp)), color = 'k' , s = 8)
    axs.scatter(4 , np.nanmean(np.concatenate(long_first_temp_opto)), color = 'cyan' , s = 8)
    axs.scatter(5 , np.nanmean(np.concatenate(long_last_temp_opto)), color = 'cyan' , s = 8)
    axs.plot([4 ,5] , [np.nanmean(np.concatenate(long_first_temp)) , np.nanmean(np.concatenate(long_last_temp))] , color = 'k')
    axs.plot([4 ,5] , [np.nanmean(np.concatenate(long_first_temp_opto)) , np.nanmean(np.concatenate(long_last_temp_opto))] , color = 'cyan')
    sem_first = np.nanstd(np.concatenate(long_first_temp))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(long_first_temp))))
    sem_last = np.nanstd(np.concatenate(long_last_temp))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(long_last_temp))))
    axs.errorbar(x = 4 , y = np.nanmean(np.concatenate(long_first_temp)) , yerr = sem_first , ecolor = 'k')
    axs.errorbar(x = 5 , y = np.nanmean(np.concatenate(long_last_temp)) , yerr = sem_last , ecolor = 'k')
    sem_first = np.nanstd(np.concatenate(long_first_temp_opto))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(long_first_temp_opto))))
    sem_last = np.nanstd(np.concatenate(long_last_temp_opto))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(long_last_temp_opto))))
    axs.errorbar(x = 4 , y = np.nanmean(np.concatenate(long_first_temp_opto)) , yerr = sem_first , ecolor = 'cyan')
    axs.errorbar(x = 5 , y = np.nanmean(np.concatenate(long_last_temp_opto)) , yerr = sem_last , ecolor = 'cyan')
    
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_ylabel('delay (s) (mean +/- sem)')
    axs.set_title('Average Delay for first and last adapted epoch') 
    
   
    axs.set_xticks([1 , 2 , 4 , 5])
    axs.set_xticklabels(['Short First' , 'Short Last' , 'Long First' , 'Long Last'], rotation='vertical')   
    
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
    
    width = 20
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
    
    width = 14
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
            else: 
                time_long.append(np.arange(width))
                if not block_data['Opto'][i][long_id[j]]:
                    delay_long.append(np.append(delay_data[long_id[j]-1][-pre:] ,delay_data[long_id[j]][0:width]))
                else:
                    delay_long_opto.append(np.append(delay_data[long_id[j]-1][-pre:] , delay_data[long_id[j]][0:width]))
        num = num + 1
    
    delay_short_mean = np.nanmean(np.array(delay_short), axis=0)
    delay_short_sem = np.nanstd(np.array(delay_short) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short))))
    axs.fill_between(np.arange(-pre,width) ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'k' , alpha = 0.2)
    axs.plot(np.arange(-pre,width), delay_short_mean , color = 'k', label = 'short')
    delay_long_mean = np.nanmean(np.array(delay_long), axis=0)
    delay_long_sem = np.nanstd(delay_long , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long))))
    axs1.fill_between(np.arange(-pre,width) ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'k' , alpha = 0.2)
    axs1.plot(np.arange(-pre,width), delay_long_mean , color = 'k', label = 'long')
    
    delay_short_mean = np.nanmean(np.array(delay_short_opto), axis=0)
    delay_short_sem = np.nanstd(np.array(delay_short_opto) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_short_opto))))
    axs.fill_between(np.arange(-pre,width) ,delay_short_mean-delay_short_sem , delay_short_mean+delay_short_sem, color = 'dodgerblue' , alpha = 0.2)
    axs.plot(np.arange(-pre,width), delay_short_mean , color = 'dodgerblue', label = 'short opto')
    delay_long_mean = np.nanmean(np.array(delay_long_opto), axis=0)
    delay_long_sem = np.nanstd(delay_long_opto , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(delay_long_opto))))
    axs1.fill_between(np.arange(-pre,width) ,delay_long_mean-delay_long_sem , delay_long_mean+delay_long_sem, color = 'dodgerblue' , alpha = 0.2)
    axs1.plot(np.arange(-pre,width), delay_long_mean , color = 'dodgerblue', label = 'long opto')
    
    axs.axvline(x = 0, color = 'gray', linestyle='--')
    axs.axvline(x = 8, color = 'gray', linestyle='--')
    axs1.axvline(x = 0, color = 'gray', linestyle='--')
    axs1.axvline(x = 8, color = 'gray', linestyle='--')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.legend()
    axs.set_xlabel('trial number')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average of initial trials delay')
    
    axs1.spines['right'].set_visible(False)
    axs1.spines['top'].set_visible(False)
    axs1.legend()
    axs1.set_xlabel('trial number')
    axs1.set_ylabel('delay (s)')
    axs1.set_title('Average of initial trials delay')
    
    
    
def run_first_epoch(axs , session_data , block_data, st, shaded = 1,  sep_short = 1, sep_long = 1):
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
    epoch_len = 11
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
            block_realize_num = sep_short
            opto_tags_short.append(block_data['Opto'][i][short_id[j]])
            # if np.isnan(block_realize_num):
            #     block_realize_num = 0
             
            if len(current_block)>1.5*epoch_len:
                short_delay_first.append(np.nanmean(current_block[:block_realize_num]))
                short_delay_last.append(np.nanmean(current_block[block_realize_num:epoch_len]))
                if not block_data['Opto'][i][short_id[j]]:
                    short_first_temp.append(current_block[:block_realize_num])
                    short_last_temp.append(current_block[block_realize_num:epoch_len])
                else:
                    short_first_temp_opto.append(current_block[:block_realize_num])
                    short_last_temp_opto.append(current_block[block_realize_num:epoch_len])
                
                
        for j in range(len(long_id)):
            current_block = block_data['delay'][i][long_id[j]]
            block_realize_num = sep_long
            opto_tags_long.append(block_data['Opto'][i][long_id[j]])
            # if np.isnan(block_realize_num):
            #     block_realize_num = 0
               
            if len(current_block)>1.5*epoch_len:
                long_delay_first.append(np.nanmean(current_block[:block_realize_num]))
                long_delay_last.append(np.nanmean(current_block[block_realize_num:epoch_len]))
                if not block_data['Opto'][i][long_id[j]]:
                    long_first_temp.append(current_block[:block_realize_num])
                    long_last_temp.append(current_block[block_realize_num:epoch_len])
                else:
                    long_first_temp_opto.append(current_block[:block_realize_num])
                    long_last_temp_opto.append(current_block[block_realize_num:epoch_len])
    
    size = 12
    color_short = set_color(opto_tags_short , 'cyan' , 'k') 
    if shaded:      
        axs.scatter(np.ones(len(short_delay_first)) , short_delay_first, color = color_short[0:len(short_delay_first)] , alpha = 0.1, s = 5)
        axs.scatter(2*np.ones(len(short_delay_last)) , short_delay_last, color = color_short[0:len(short_delay_last)] , alpha = 0.1, s = 5)
        for i in range(len(short_delay_first)):
            axs.plot([1 ,2] , [short_delay_first[i] , short_delay_last[i]] , color = color_short[i] , alpha = 0.1)
    #print(len(short_first_temp) , short_first_temp)
    # for i in range(len(short_first_temp)):
    #     print(short_first_temp[i].shape)
    #     print()
    # print(short_first_temp)
    #short_first_temp = np.array(short_first_temp)
    #short_first_temp = [sublist for sublist in short_first_temp if sublist]
    axs.scatter(1 , np.nanmean(np.concatenate(short_first_temp)), color = 'k' , s = size)
    axs.scatter(2 , np.nanmean(np.concatenate(short_last_temp)), color = 'k' , s = size)
    axs.scatter(1 , np.nanmean(np.concatenate(short_first_temp_opto)), color = 'cyan' , s = size)
    axs.scatter(2 , np.nanmean(np.concatenate(short_last_temp_opto)), color = 'cyan' , s = size)
    axs.plot([1 ,2] , [np.nanmean(np.concatenate(short_first_temp)) , np.nanmean(np.concatenate(short_last_temp))] , color = 'k')
    axs.plot([1 ,2] , [np.nanmean(np.concatenate(short_first_temp_opto)) , np.nanmean(np.concatenate(short_last_temp_opto))] , color = 'cyan')
    sem_first = np.nanstd(np.concatenate(short_first_temp))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(short_first_temp))))
    sem_last = np.nanstd(np.concatenate(short_last_temp))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(short_last_temp))))
    axs.errorbar(x = 1 , y = np.nanmean(np.concatenate(short_first_temp)) , yerr = sem_first , ecolor = 'k')
    axs.errorbar(x = 2 , y = np.nanmean(np.concatenate(short_last_temp)) , yerr =sem_last , ecolor = 'k')
    sem_first = np.nanstd(np.concatenate(short_first_temp_opto))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(short_first_temp_opto))))
    sem_last = np.nanstd(np.concatenate(short_last_temp_opto))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(short_last_temp_opto))))
    axs.errorbar(x = 1 , y = np.nanmean(np.concatenate(short_first_temp_opto)) , yerr = sem_first , ecolor = 'cyan')
    axs.errorbar(x = 2 , y = np.nanmean(np.concatenate(short_last_temp_opto)) , yerr = sem_last, ecolor = 'cyan')
    
    color_long = set_color(opto_tags_long , 'cyan' , 'k') 
    if shaded:      
        axs.scatter(4*np.ones(len(long_delay_first)) , long_delay_first, color = color_long[: min(len(long_delay_first), len(color_long))] , alpha = 0.1, s = 5)
        axs.scatter(5*np.ones(len(long_delay_last)) , long_delay_last, color = color_long[: min(len(long_delay_first), len(color_long))] , alpha = 0.1, s = 5)
        for i in range(len(long_delay_first)):
            axs.plot([4 , 5] , [long_delay_first[i] , long_delay_last[i]] , color = color_long[i] , alpha = 0.1)
    axs.scatter(4 , np.nanmean(np.concatenate(long_first_temp)), color = 'k' , s = size)
    axs.scatter(5 , np.nanmean(np.concatenate(long_last_temp)), color = 'k' , s = size)
    axs.scatter(4 , np.nanmean(np.concatenate(long_first_temp_opto)), color = 'cyan' , s = size)
    axs.scatter(5 , np.nanmean(np.concatenate(long_last_temp_opto)), color = 'cyan' , s = size)
    axs.plot([4 ,5] , [np.nanmean(np.concatenate(long_first_temp)) , np.nanmean(np.concatenate(long_last_temp))] , color = 'k')
    axs.plot([4 ,5] , [np.nanmean(np.concatenate(long_first_temp_opto)) , np.nanmean(np.concatenate(long_last_temp_opto))] , color = 'cyan')
    sem_first = np.nanstd(np.concatenate(long_first_temp))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(long_first_temp))))
    sem_last = np.nanstd(np.concatenate(long_last_temp))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(long_last_temp))))
    axs.errorbar(x = 4 , y = np.nanmean(np.concatenate(long_first_temp)) , yerr = sem_first , ecolor = 'k')
    axs.errorbar(x = 5 , y = np.nanmean(np.concatenate(long_last_temp)) , yerr = sem_last , ecolor = 'k')
    sem_first = np.nanstd(np.concatenate(long_first_temp_opto))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(long_first_temp_opto))))
    sem_last = np.nanstd(np.concatenate(long_last_temp_opto))/np.sqrt(np.count_nonzero(~np.isnan(np.concatenate(long_last_temp_opto))))
    axs.errorbar(x = 4 , y = np.nanmean(np.concatenate(long_first_temp_opto)) , yerr = sem_first , ecolor = 'cyan')
    axs.errorbar(x = 5 , y = np.nanmean(np.concatenate(long_last_temp_opto)) , yerr = sem_last , ecolor = 'cyan')
    
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_ylabel('delay (s) (mean +/- sem)')
    axs.set_title('Average Delay for first and last epoch' + 'frist epoch short = ' + str(sep_short) + ' long = '  + str(sep_long)) 
    
   
    axs.set_xticks([1 , 2 , 4 , 5])
    axs.set_xticklabels(['Short First' , 'Short Last' , 'Long First' , 'Long Last'], rotation='vertical')  