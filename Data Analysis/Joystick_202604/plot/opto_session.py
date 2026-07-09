# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:18:53 2025

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
states_mod = [
    'Reward' , 
    'DidNotPress' , 
    'EarlyPress' , 
    'LatePress' ,
    'Other']  
states_name_mod = [
    'Reward' , 
    'DidNotPress' , 
    'EarlyPress' , 
    'LatePress' ,
    'Other']   
colors_mod = [
    '#4CAF50',
    '#FFB74D',
    '#64B5F6',
    'deeppink',
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
    
def change_label(session_label, plotting_sessions, norm = True):
    num_session = len(plotting_sessions)
    counts = np.zeros((2 , num_session, len(states_mod)))
    numtrials = np.zeros((2 , num_session))
    num = 0
    label_mod = []
    for i in plotting_sessions:
        temp = []
        for j in range(len(session_label[i])):
            if norm:
                if session_label[i][j] in ['EarlyPress2' , 'Reward' , 'LatePress2' , 'DidNotPress2']:
                    temp.append('Reward') 
                elif session_label[i][j] == 'EarlyPress1':
                    temp.append('EarlyPress')
                elif session_label[i][j] == 'DidNotPress1':
                    temp.append('DidNotPress')
                elif session_label[i][j] == 'LatePress1':
                    temp.append('LatePress')
                elif session_label[i][j] == 'Warmup':
                    temp.append('Warmup')
                else:
                    temp.append('Other')
                    
        label_mod.append(temp)
       
    return label_mod

def count_label(session_data, opto, session_label, states, plotting_sessions, norm=True):
    num_session = len(plotting_sessions)
    counts = np.zeros((2 , num_session, len(states)))
    numtrials = np.zeros((2 , num_session))
    num = 0
    for i in plotting_sessions:
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        for j in range(len(session_label[i])):
            if norm:
                if session_label[i][j] in states:
                    k = states.index(session_label[i][j])
                    if j > 0:
                        if opto[i][j-1] == 1:
                            counts[1 , num , k] = counts[1 , num , k] + 1
                            numtrials[1 , num] = numtrials[1 , num] + 1    
                        else:
                            counts[0 , num , k] = counts[0 , num , k] + 1
                            numtrials[0 , num] = numtrials[0 , num] + 1 
                    else:
                        counts[0 , num , k] = counts[0 , num , k] + 1
                        numtrials[0 , num] = numtrials[0 , num] + 1 
                    
                    
        for j in range(len(states)):
            counts[0 , num , j] = counts[0 , num , j]/numtrials[0 , num]
            counts[1 , num , j] = counts[1 , num , j]/numtrials[1 , num]
        num = num + 1
    return counts


def count_label_mod(session_data, opto, session_label, states, plotting_sessions, norm=True):
    num_session = len(plotting_sessions)
    counts = np.zeros((2 , num_session, len(states_mod)))
    numtrials = np.zeros((2 , num_session))
    num = 0
    session_label_mod = change_label(session_label, plotting_sessions)
    #print(session_label_mod)
    for i in plotting_sessions:
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        for j in range(len(session_label_mod[i])):
            if norm:
                #print(session_label_mod[i][j])
                if session_label_mod[i][j] in states_mod:
                    k = states_mod.index(session_label_mod[i][j])
                    if opto[i][j] == 1:
                        counts[1 , num , k] = counts[1 , num , k] + 1
                        numtrials[1 , num] = numtrials[1 , num] + 1    
                    else:
                        counts[0 , num , k] = counts[0 , num , k] + 1
                        numtrials[0 , num] = numtrials[0 , num] + 1    
                    
                    
        for j in range(len(states_mod)):
            counts[0 , num , j] = counts[0 , num , j]/numtrials[0 , num]
            counts[1 , num , j] = counts[1 , num , j]/numtrials[1 , num]
        num = num + 1
    return counts


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
    
def outcome(axs , session_data , st = 0):
    
    max_sessions=100
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    num_session = len(dates)
    isSelfTime = session_data['isSelfTimedMode']
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
    dates1 = []
    num = 0
    
    counts = count_label(session_data, opto, outcomes, states , plotting_sessions)
    
    session_id = np.arange(len(plotting_sessions)) + 1
    
    control_bottom = np.cumsum(counts[0 , : , :], axis=1)
    control_bottom[:,1:] = control_bottom[:,:-1]
    control_bottom[:,0] = 0
    
    opto_bottom = np.cumsum(counts[1 , : , :], axis=1)
    opto_bottom[:,1:] = opto_bottom[:,:-1]
    opto_bottom[:,0] = 0
    
    width = 0.2
    
    top_ticks = []
    
    axs.axhline(0.5 , linestyle = '--' , color = 'gray')
    
    
    for i in range(len(states)):
        axs.bar(
            session_id-width , counts[0,:,i],
            bottom=control_bottom[:,i],
            width=width,
            color=colors[i],
            label=states_name[i])
        axs.bar(
            session_id + width, counts[1,:,i],
            bottom=opto_bottom[:,i],
            width=width,
            color=colors[i])
        
        
        
    
    #axs.tick_params(tick1On=False)
    #axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Training session')
    
    axs.set_ylabel('Outcome percentages')
    
    axs.set_xticks(np.arange(3*len(plotting_sessions))*(width)+0.8)
    tick_index = np.arange(3*len(plotting_sessions))*(width)+0.8
    axs.set_xticks(tick_index)
    
    dates_label = []
    
    for i in plotting_sessions:
        dates_label.append('control')
        if isSelfTime[i][5] == 1:
            dates_label.append(dates[i] + '-ST')
        else:
            dates_label.append(dates[i] + '-VG')
        dates_label.append('opto+1')
    axs.set_xticklabels(dates_label, rotation='vertical')
    
    
    
    secax = axs.secondary_xaxis('top')
    secax.set_xticks([])
    secax.set_xticklabels(top_ticks)
    secax.tick_params(labelsize =  8)
    secax.spines['top'].set_visible(False)
    secax.tick_params(tick1On=False)
    
    
    
    
    axs.set_title('Reward percentage for completed trials across sessions')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
def outcome_mod(axs , session_data , st = 0):
    
    max_sessions=100
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    num_session = len(dates)
    isSelfTime = session_data['isSelfTimedMode']
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
    dates1 = []
    num = 0
    
    counts = count_label_mod(session_data, opto, outcomes, states , plotting_sessions)
    
    session_id = np.arange(len(plotting_sessions)) + 1
    
    control_bottom = np.cumsum(counts[0 , : , :], axis=1)
    control_bottom[:,1:] = control_bottom[:,:-1]
    control_bottom[:,0] = 0
    
    opto_bottom = np.cumsum(counts[1 , : , :], axis=1)
    opto_bottom[:,1:] = opto_bottom[:,:-1]
    opto_bottom[:,0] = 0
    
    width = 0.2
    
    top_ticks = []
    
    axs.axhline(0.5 , linestyle = '--' , color = 'gray')
    for i in range(len(plotting_sessions)):
        top_ticks.append('Control|Opto')
    
    for i in range(len(states_mod)):
        axs.bar(
            session_id-width , counts[0,:,i],
            bottom=control_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors_mod[i],
            label=states_name_mod[i])
        axs.bar(
            session_id+width, counts[1,:,i],
            bottom=opto_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors_mod[i])
        
        
        
    
    axs.tick_params(tick1On=False)
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Training session')
    
    axs.set_ylabel('Outcome percentages')
    
    axs.set_xticks(np.arange(len(plotting_sessions))+1)
    tick_index = np.arange(len(plotting_sessions))+1
    axs.set_xticks(tick_index + width/2)
    
    dates_label = []
    
    for i in plotting_sessions:
        if isSelfTime[i][5] == 1:
            dates_label.append(dates[i] + '-ST')
        else:
            dates_label.append(dates[i] + '-VG')
    axs.set_xticklabels(dates_label, rotation='vertical')
    secax = axs.secondary_xaxis('top')
    secax.set_xticks(tick_index + width/4)
    secax.set_xticklabels(top_ticks)
    secax.tick_params(labelsize = 8)
    
    
    
    
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
       # plot_trajectory(axs[0], dates1 , time_led , np.mean(np.array(trajectory_opto_led) , axis = 0) ,num , num_plotting , 'opto' , 1)
       # plot_trajectory(axs[1], dates1 , time_reward , np.mean(np.array(trajectory_opto) , axis = 0) ,num , num_plotting , 'opto' , 0)
       # #print(led_end)
       # axs[1].axvline(x = np.nanmean(np.array(led_end)), color = 'gray', linestyle='--')
       # plot_trajectory(axs[2], dates1 , time_reward , np.mean(np.array(trajectory_control) , axis = 0), num , num_plotting , 'control' , 0)
       
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
    session_label_mod = change_label(outcomes, plotting_sessions)
    
    for i in plotting_sessions:
       dates1.append(dates[i])
       num = num + 1
       #outcome_sess = session_label_mod[i]
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
       # plot_trajectory(axs[0], dates1 , time_led , np.mean(np.array(trajectory_opto_led) , axis = 0) ,num , num_plotting , 'opto' , 1)
       # plot_trajectory(axs[1], dates1 , time_reward , np.mean(np.array(trajectory_opto) , axis = 0) ,num , num_plotting , 'opto' , 0)
       # #print(led_end)
       # axs[1].axvline(x = np.nanmean(np.array(led_end)), color = 'gray', linestyle='--')
       # plot_trajectory(axs[2], dates1 , time_reward , np.mean(np.array(trajectory_control) , axis = 0), num , num_plotting , 'control' , 0)
       #axs[0].set_title(outcome_select)
       plot_trajectory(axs[0], dates1 , time_led , trajectory_opto_led ,num , num_plotting , 'opto (' +  outcome_select + ')', 'Laser Offset')
       plot_trajectory(axs[1], dates1 , time_reward , trajectory_opto ,num , num_plotting , 'opto (' +  outcome_select + ')', 'Vis1')
       #print(led_end)
       #axs[1].axvline(x = np.nanmean(np.array(led_end)), color = 'gray', linestyle='--')
       plot_trajectory(axs[2], dates1 , time_reward , trajectory_control, num , num_plotting , 'control (' +  outcome_select + ')' , 'Vis1')