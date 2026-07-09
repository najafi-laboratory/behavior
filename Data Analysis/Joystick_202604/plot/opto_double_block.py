# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:18:05 2025

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

def opto_delay(session_data, opto, session_label, delay, plotting_sessions, norm=True):
    short_opto = []
    short_pos = []
    
    long_opto = []
    long_pos = []
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
                                short_pos.append(delay[i][j])   
                            else:
                                long_pos.append(delay[i][j])   
                        if opto[i][j] == 1:
                            if trial_types[j] == 1:
                                short_opto.append(delay[i][j])   
                            else:
                                long_opto.append(delay[i][j])   
                    else:
                        if opto[i][j] == 1:
                            if trial_types[j] == 1:
                                short_opto.append(delay[i][j])   
                            else:
                                long_opto.append(delay[i][j])   
    return short_opto, short_pos, long_opto, long_pos

def plot_trajectory(axs, dates, time , trajectory , num , max_num , title , legend):
   
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=max_num+1)
    
    trajectory_mean = np.mean(np.array(trajectory) , axis = 0) 
    trajectory_std = np.std(np.array(trajectory) , axis = 0)/ np.sqrt(len(np.array(trajectory)))
    
    axs.plot(time , trajectory_mean , color = 'k')
    axs.fill_between(time ,trajectory_mean-trajectory_std , trajectory_mean+trajectory_std, color = 'k' , alpha = 0.2)
    #axs.plot(time , trajectory , color = cmap(norm(max_num+1-num)) )
    axs.axvline(x = 0, color = 'gray', linestyle='--' , label = legend + '(' + str(len(trajectory)) + ')')
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Time(s)')
    axs.set_ylabel('joystick deflection (deg)')
    #axs.set_ylim([0.05 , 3])
    axs.set_title(title)
    axs.legend()
    if legend:
        axs.set_xlabel('Time(s)')
        axs.set_ylabel('joystick deflection (deg)')
        #axs.set_ylim([0.05 , 2])
    
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
    
    # axs.set_xticks(np.arange(5*len(plotting_sessions))*0.1+0.8)
    # tick_index = np.arange(5*len(plotting_sessions))*0.1+0.8
    # axs.set_xticks(tick_index)
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
    axs.set_xticklabels(dates_label, rotation='vertical')
    
    
    
    secax = axs.secondary_xaxis('top')
    secax.set_xticks([])
    secax.set_xticklabels(top_ticks)
    secax.tick_params(labelsize =  8)
    secax.spines['top'].set_visible(False)
    secax.tick_params(tick1On=False)
    
    
    
    
    axs.set_title('Reward percentage for completed trials across sessions')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    

def run_trajectory(axs, session_data , st = 0):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    
    
    num_session = len(dates)
    max_num = 20
    
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
    
    num_plotting =  min(max_num , num_session)
    initial_start = max(num_session - 3*max_num , 0)
    
    for i in plotting_sessions:
       dates1.append(dates[i])
       num = num + 1
       outcome_sess = outcomes[i]
       session_raw = session_data['raw'][i]['RawEvents']['Trial']
       delay = session_data['session_comp_delay'][i]
       encoder_data_aligned = session_data['encoder_pos_aligned'][i]
       opto = session_data['session_opto_tag'][i]
       
       trajectory_opto = []
       trajectory_control = []
       trajectory_opto_led = []
       
       raw_data = session_data['raw'][i]
       events = raw_data['RawEvents']['Trial']
       trial_types = raw_data['TrialTypes']
       led_end = []
       left = 1000
       right = 5000
       for j in range(len(outcome_sess)):
           outcome = outcome_sess[j]
           if opto[j] == 1:
               if 'GlobalTimer2_End' in events[j]['Events'].keys():
                   #print(events[j]['Events']['GlobalTimer1_End'])
                   if isinstance(events[j]['Events']['GlobalTimer2_Start'], float):
                       temp = events[j]['Events']['GlobalTimer2_Start']-events[j]['States']['VisualStimulus1'][1]
                   else:
                       temp = events[j]['Events']['GlobalTimer2_Start'][-1]-events[j]['States']['VisualStimulus1'][1]
                   #print(temp)
                   led_end.append(temp)
                   if not np.isnan(events[j]['States']['VisualStimulus1'][1]):
                       align = events[j]['States']['VisualStimulus1'][1]
                       pos_time = int(np.round(align*1000))
                       
                       time_reward = np.linspace(-left/1000, right/1000 , right+left)
                       if pos_time < left:
                           nan_pad = np.zeros(left-pos_time)
                           nan_pad[:] = np.nan
                           trajectory_opto.append(np.append(nan_pad, encoder_data_aligned[j][0:pos_time+right]))
                       else:
                           trajectory_opto.append(encoder_data_aligned[j][pos_time-left:pos_time+right])
                           
                   if isinstance(events[j]['Events']['GlobalTimer2_Start'], float):
                       temp = events[j]['Events']['GlobalTimer2_Start']
                   else:
                       temp = events[j]['Events']['GlobalTimer2_Start'][-1]
                   
                   if not np.isnan(temp):
                       align = events[j]['States']['VisualStimulus1'][1]
                       pos_time = int(np.round(temp*1000))
                       left1 = 2000
                       right1 = 3000
                       time_led = np.linspace(-left1/1000, right1/1000 , right1+left1)
                       if pos_time < left1:
                           nan_pad = np.zeros(left1-pos_time)
                           nan_pad[:] = np.nan
                           trajectory_opto_led.append(np.append(nan_pad, encoder_data_aligned[j][0:pos_time+right1]))
                       else:
                           trajectory_opto_led.append(encoder_data_aligned[j][pos_time-left1:pos_time+right1])
           else:
               if 'VisualStimulus1' in events[j]['States'].keys():
                   align = events[j]['States']['VisualStimulus1'][1]
                   pos_time = int(np.round(align*1000))
                   
                   time_reward = np.linspace(-left/1000, right/1000 , right+left)
                   if pos_time < left:
                       nan_pad = np.zeros(left-pos_time)
                       nan_pad[:] = np.nan
                       trajectory_control.append(np.append(nan_pad, encoder_data_aligned[j][0:pos_time+right]))
                   else:
                       trajectory_control.append(encoder_data_aligned[j][pos_time-left:pos_time+right])
                       
       
           
       if len(trajectory_opto) == 0:
           
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_opto.append(nan_pad)
           
       if len(trajectory_control) == 0:
           
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_control.append(nan_pad)
         
       title_str = ['Probe-1' , 'Probe' , 'Probe+1']
       
       plot_trajectory(axs[0], dates1 , time_led , trajectory_opto_led ,num , num_plotting , 'opto' , 'Laser Offset')
       plot_trajectory(axs[1], dates1 , time_reward , trajectory_opto ,num , num_plotting , 'opto' , 'Vis1')
       #print(led_end)
       #axs[1].axvline(x = np.nanmean(np.array(led_end)), color = 'gray', linestyle='--')
       plot_trajectory(axs[2], dates1 , time_reward , trajectory_control, num , num_plotting , 'control' , 'Vis1')
       
       
def run_trajectory_outcome(axs, session_data , outcome_select, st = 0):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    
    
    num_session = len(dates)
    max_num = 20
    
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
    
    num_plotting =  min(max_num , num_session)
    initial_start = max(num_session - 3*max_num , 0)
    
    for i in plotting_sessions:
       dates1.append(dates[i])
       num = num + 1
       outcome_sess = outcomes[i]
       session_raw = session_data['raw'][i]['RawEvents']['Trial']
       delay = session_data['session_comp_delay'][i]
       encoder_data_aligned = session_data['encoder_pos_aligned'][i]
       opto = session_data['session_opto_tag'][i]
       
       trajectory_opto = []
       trajectory_control = []
       trajectory_opto_led = []
       
       raw_data = session_data['raw'][i]
       events = raw_data['RawEvents']['Trial']
       trial_types = raw_data['TrialTypes']
       led_end = []
       left = 1000
       right = 5000
       for j in range(len(outcome_sess)):
           outcome = outcome_sess[j]
           if outcome == outcome_select:
               if opto[j] == 1:
                   if 'GlobalTimer2_End' in events[j]['Events'].keys():
                       #print(events[j]['Events']['GlobalTimer1_End'])
                       if isinstance(events[j]['Events']['GlobalTimer2_Start'], float):
                           temp = events[j]['Events']['GlobalTimer2_Start']-events[j]['States']['VisualStimulus1'][1]
                       else:
                           temp = events[j]['Events']['GlobalTimer2_Start'][-1]-events[j]['States']['VisualStimulus1'][1]
                       #print(temp)
                       led_end.append(temp)
                       if not np.isnan(events[j]['States']['VisualStimulus1'][1]):
                           align = events[j]['States']['VisualStimulus1'][1]
                           pos_time = int(np.round(align*1000))
                           
                           time_reward = np.linspace(-left/1000, right/1000 , right+left)
                           if pos_time < left:
                               nan_pad = np.zeros(left-pos_time)
                               nan_pad[:] = np.nan
                               trajectory_opto.append(np.append(nan_pad, encoder_data_aligned[j][0:pos_time+right]))
                           else:
                               trajectory_opto.append(encoder_data_aligned[j][pos_time-left:pos_time+right])
                               
                       if isinstance(events[j]['Events']['GlobalTimer2_Start'], float):
                           temp = events[j]['Events']['GlobalTimer2_Start']
                       else:
                           temp = events[j]['Events']['GlobalTimer2_Start'][-1]
                       
                       if not np.isnan(temp):
                           align = events[j]['States']['VisualStimulus1'][1]
                           pos_time = int(np.round(temp*1000))
                           left1 = 2000
                           right1 = 3000
                           time_led = np.linspace(-left1/1000, right1/1000 , right1+left1)
                           if pos_time < left1:
                               nan_pad = np.zeros(left1-pos_time)
                               nan_pad[:] = np.nan
                               trajectory_opto_led.append(np.append(nan_pad, encoder_data_aligned[j][0:pos_time+right1]))
                           else:
                               trajectory_opto_led.append(encoder_data_aligned[j][pos_time-left1:pos_time+right1])
               else:
                   if 'VisualStimulus1' in events[j]['States'].keys():
                       align = events[j]['States']['VisualStimulus1'][1]
                       pos_time = int(np.round(align*1000))
                       
                       time_reward = np.linspace(-left/1000, right/1000 , right+left)
                       if pos_time < left:
                           nan_pad = np.zeros(left-pos_time)
                           nan_pad[:] = np.nan
                           trajectory_control.append(np.append(nan_pad, encoder_data_aligned[j][0:pos_time+right]))
                       else:
                           trajectory_control.append(encoder_data_aligned[j][pos_time-left:pos_time+right])
                       
       
           
       if len(trajectory_opto) == 0:
           
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_opto.append(nan_pad)
           
       if len(trajectory_control) == 0:
           
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_control.append(nan_pad)
         
       title_str = ['Probe-1' , 'Probe' , 'Probe+1']
       plot_trajectory(axs[0], dates1 , time_led , trajectory_opto_led ,num , num_plotting , 'opto (' +  outcome_select + ')', 'Laser Offset')
       plot_trajectory(axs[1], dates1 , time_reward , trajectory_opto ,num , num_plotting , 'opto (' +  outcome_select + ')', 'Vis1')
       #print(led_end)
       #axs[1].axvline(x = np.nanmean(np.array(led_end)), color = 'gray', linestyle='--')
       plot_trajectory(axs[2], dates1 , time_reward , trajectory_control, num , num_plotting , 'control (' +  outcome_select + ')' , 'Vis1')
       
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
    
    
    
    for i in plotting_sessions:
        short_first = []
        short_last = []
        
        long_first = []
        long_last = []
        
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        max_trial_num = np.max(block_data['NumTrials'][i])
        # if short_id[0] == 0:
        #     short_id = short_id[1:]
        # if len(long_id) > 0:
        #     if long_id[0] ==0:
        #         long_id = long_id[1:]
        for j in range(len(short_id)):
            current_block = block_data['delay'][i][short_id[j]]
            
            if len(current_block)>1.5*epoch_len:
                short_first.append(np.nanmean(current_block[0:epoch_len]))
                short_last.append(np.nanmean(current_block[epoch_len:]))
                
        for j in range(len(long_id)):
            current_block = block_data['delay'][i][long_id[j]]
            if len(current_block)>1.5*epoch_len:
                long_first.append(np.nanmean(current_block[0:epoch_len]))
                long_last.append(np.nanmean(current_block[epoch_len:]))
                
        
    size = 3
    x_first = np.arange(0 ,len(short_first))/len(short_first)
    x_last = np.arange(0 ,len(short_last))/len(short_last)+2
    cmap = plt.cm.inferno
    lens = max(len(short_last),len(short_first))
    norm = plt.Normalize(vmin=0, vmax=lens+1)
    
    color1 = []
    lens = max(len(short_last),len(short_first))
    for i in range(max(len(short_last),len(short_first))):
        color1.append(np.array(cmap(norm(lens+1-i))))
        axs.plot([x_first[i] , x_last[i]],[short_first[i] , short_last[i]], color = np.array(cmap(norm(lens+1-i))))
    
    axs.scatter(x_first, short_first, color = color1 , label = date , s = size)
    axs.scatter(x_last, short_last, color = color1 , s = size)
    
    x_first = np.arange(0 ,len(long_first))/len(long_first)+5
    x_last = np.arange(0 ,len(long_last))/len(long_last)+7
    cmap = plt.cm.inferno
    lens = max(len(long_last),len(long_first))
    norm = plt.Normalize(vmin=0, vmax=lens+1)
    color1 = []
    for i in range(max(len(long_last),len(long_first))):
        color1.append(np.array(cmap(norm(lens+1-i))))
        axs.plot([x_first[i] , x_last[i]],[long_first[i] , long_last[i]], color = np.array(cmap(norm(lens+1-i))))
    
    axs.scatter(x_first, long_first, color = color1, s= size)
    axs.scatter(x_last, long_last, color = color1, s= size)
    
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_ylabel('delay (s)')
    axs.set_title('Average Delay for first and last epoches ') 
    
   
    axs.set_xticks([0.5 , 2.5 , 5.5 , 7.5])
    axs.set_xticklabels(['Short First' , 'Short Last' , 'Long First' , 'Long Last'], rotation='vertical')
    
def run_opto_delay(axs , session_data , block_data, st):
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
    epoch_len = 7
    
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    num_session = len(dates)
    isSelfTime = session_data['isSelfTimedMode']
    opto = session_data['session_opto_tag']
    short_first, short_last, long_first, long_last = opto_delay(session_data, opto, outcomes, session_data['session_comp_delay'], plotting_sessions, norm=True)  
    #(session_data, opto, session_label, delay, plotting_sessions, norm=True)
    
    size = 3
    x_first = np.arange(0 ,len(short_first))/len(short_first)
    x_last = np.arange(0 ,len(short_last))/len(short_last)+2
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=num_session+1)
    axs.plot([x_first[0] , x_last[0]],[np.nanmean(np.array(short_first)) , np.nanmean(np.array(short_last))], color = 'k')
    axs.scatter([x_first[0] , x_last[0]],[np.nanmean(np.array(short_first)) , np.nanmean(np.array(short_last))], color = 'k' , s= size+2)
    
    axs.scatter(x_first, short_first, color = 'gray' , label = date , s = size)
    axs.scatter(x_last, short_last, color = 'gray' , s = size)
    
    x_first = np.arange(0 ,len(long_first))/len(long_first)+5
    x_last = np.arange(0 ,len(long_last))/len(long_last)+7
    
    axs.scatter(x_first, long_first, color = 'gray', s= size)
    axs.scatter(x_last, long_last, color = 'gray', s= size)
    
    axs.plot([x_first[0] , x_last[0]],[np.nanmean(np.array(long_first)) , np.nanmean(np.array(long_last))], color = 'k')
    axs.scatter([x_first[0] , x_last[0]],[np.nanmean(np.array(long_first)) , np.nanmean(np.array(long_last))], color = 'k' , s= size+2)
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_ylabel('delay (s)')
    axs.set_title('Average Delay for opto vs opto+1') 
    
   
    axs.set_xticks([0.5 , 2.5 , 5.5 , 7.5])
    axs.set_xticklabels(['Short Opto' , 'Short Opto+1' , 'Long Opto' , 'Long Opto+1'], rotation='vertical')
    
def run_epoch_compare(axs , session_data , block_data, st):
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
    short_first = []
    short_last = []
    long_first = []
    long_last = []
    short_first_sem = []
    short_last_sem = []
    long_first_sem = []
    long_last_sem = []
    color_easy = ['k' , 'cyan']
    
    for i in plotting_sessions:
        short_delay_first = []
        short_delay_last = []
        
        long_delay_first = []
        long_delay_last = []
        
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
                
        for j in range(len(long_id)):
            current_block = block_data['delay'][i][long_id[j]]
            if len(current_block)>1.5*epoch_len:
                long_delay_first.append(np.nanmean(current_block[0:epoch_len]))
                long_delay_last.append(np.nanmean(current_block[epoch_len:]))
                
        short_first.append(np.mean(short_delay_first))
        short_last.append(np.mean(short_delay_last))
        long_first.append(np.mean(long_delay_first))
        long_last.append(np.mean(long_delay_last))
        short_first_sem.append(np.std(short_delay_first)/ np.sqrt(len(short_delay_first)))
        short_last_sem.append(np.std(short_delay_last)/ np.sqrt(len(short_delay_last)))
        long_first_sem.append(np.std(long_delay_first)/ np.sqrt(len(long_delay_first)))
        long_last_sem.append(np.std(long_delay_last)/ np.sqrt(len(long_delay_last)))
        
    size = 3
    x_first = np.arange(0 ,len(short_first))/len(short_first)
    x_last = np.arange(0 ,len(short_last))/len(short_last)+2
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=num_session+1)
    
    color1 = []
    for i in range(max(len(short_last),len(short_first))):
        color1.append(np.array(cmap(norm(num_session+1-i))))
        axs.plot([x_first[i] , x_last[i]],[short_first[i] , short_last[i]], color = color_easy[i])
    
    axs.scatter(x_first, short_first, color = color_easy , label = date , s = size)
    axs.scatter(x_last, short_last, color = color_easy , s = size)
    axs.errorbar(x_first[0], short_first[0], yerr=short_first_sem[0] , color = 'k')
    axs.errorbar(x_last[0], short_last[0], yerr=short_last_sem[0], color = 'k')
    axs.errorbar(x_first[1], short_first[1], yerr=short_first_sem[1] , color = 'cyan')
    axs.errorbar(x_last[1], short_last[1], yerr=short_last_sem[1], color = 'cyan')
    
    x_first = np.arange(0 ,len(long_first))/len(long_first)+5
    x_last = np.arange(0 ,len(long_last))/len(long_last)+7
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=num_session+1)
    color1 = []
    for i in range(max(len(long_last),len(long_first))):
        color1.append(np.array(cmap(norm(num_session+1-i))))
        axs.plot([x_first[i] , x_last[i]],[long_first[i] , long_last[i]], color = color_easy[i])
    
    axs.scatter(x_first, long_first, color = color_easy, s= size)
    axs.scatter(x_last, long_last, color = color_easy, s= size)
    axs.errorbar(x_first[0], long_first[0], yerr=long_first_sem[0] , color = 'k')
    axs.errorbar(x_last[0], long_last[0], yerr=long_last_sem[0], color = 'k')
    axs.errorbar(x_first[1], long_first[1], yerr=long_first_sem[1] , color = 'cyan')
    axs.errorbar(x_last[1], long_last[1], yerr=long_last_sem[1], color = 'cyan')
    
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('trial number')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average Delay for first and last epoches ') 
    
   
    axs.set_xticks([0.5 , 2.5 , 5.5 , 7.5])
    axs.set_xticklabels(['Short First' , 'Short Last' , 'Long First' , 'Long Last'], rotation='vertical')