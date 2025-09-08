# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:06:25 2025

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


def plot_trajectory(axs, time , trajectory, title , legend, Title1):
   
    trajectory_mean = np.mean(np.array(trajectory) , axis = 0) 
    trajectory_std = np.std(np.array(trajectory) , axis = 0) / np.sqrt(len(trajectory))
    if title == 'short': 
        axs.plot(time , trajectory_mean , color = 'y' , label = title)
        axs.fill_between(time ,trajectory_mean-trajectory_std , trajectory_mean+trajectory_std, color = 'y' , alpha = 0.2)
        axs.axvline(x = 0, color = 'gray', linestyle='--')
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.set_xlabel('Time from Press2 (s)')
        axs.set_ylabel('joystick deflection (deg) (mean +/- sem)')
        axs.set_xlim([-2 , 1.5])
        axs.set_title(Title1)
    else: 
        axs.plot(time , trajectory_mean , color = 'b' , label = title)
        axs.fill_between(time ,trajectory_mean-trajectory_std , trajectory_mean+trajectory_std, color = 'b' , alpha = 0.2)
    if legend:
        axs.legend()


def run_trajectory(axs, session_data, st, title):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    num_session = len(dates)
    ST , VG = IsSelfTimed(session_data)
    trajectory_short = []
    trajectory_long = []
    
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
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_reward_short.append(nan_pad)
           
       if len(trajectory_reward_long) == 0:
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_reward_long.append(nan_pad)
           
       time_reward = np.linspace(-left/1000, right/1000 , right+left)
           
       trajectory_short.append(list(np.mean(np.array(trajectory_reward_short) , axis = 0)))
       trajectory_long.append(list(np.mean(np.array(trajectory_reward_long) , axis = 0)))
           
    plot_trajectory(axs , time_reward , trajectory_short , 'short' , 1 , title)
    plot_trajectory(axs , time_reward , trajectory_long , 'long' , 1 , title)
       
def run_trajectory_probe(axs, session_data,probe_ind, st, title):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    num_session = len(dates)
    ST , VG = IsSelfTimed(session_data)
    trajectory_short = []
    trajectory_long = []
    
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
    
    max_num = 30
    num_plotting =  min(max_num , num_session)
    initial_start = max(num_session - 3*max_num , 0)
    probe_sign = [1 , 0 , -1]
    for i in plotting_sessions:
        
       outcome_sess = outcomes[i]
       session_raw = session_data['raw'][i]['RawEvents']['Trial']
       delay = session_data['session_comp_delay'][i]
       encoder_data_aligned = session_data['encoder_pos_aligned'][i]
       probe = session_data['session_probe'][i]
       trajectory_reward_short = []
       trajectory_reward_long = []
       raw_data = session_data['raw'][i]
       trial_types = raw_data['TrialTypes']
       num = num + 1
       for j in range(len(outcome_sess)):
           outcome = outcome_sess[j]
           if j+probe_sign[probe_ind] > 0 and j+probe_sign[probe_ind]<len(outcome_sess):
               if outcome == 'Reward' :
                   press = delay[j] + session_raw[j]['States']['LeverRetract1'][1]
                   pos_time = int(np.round(press*1000))
                   left = 2000
                   right = 1500
                   time_reward = np.linspace(-left/1000, right/1000 , right+left)
                   if pos_time < left:
                       nan_pad = np.zeros(left-pos_time)
                       nan_pad[:] = np.nan
                      
                       if probe[j+probe_sign[probe_ind]] == 1:
                           if trial_types[j] == 1:
                               trajectory_reward_short.append(np.append(nan_pad, encoder_data_aligned[j][0:pos_time+right]))
                           else:
                               trajectory_reward_long.append(np.append(nan_pad, encoder_data_aligned[j][0:pos_time+right]))
                   else:
                       if probe[j+probe_sign[probe_ind]] == 1:
                           if trial_types[j] == 1:
                               trajectory_reward_short.append(encoder_data_aligned[j][pos_time-left:pos_time+right])
                           else:
                               trajectory_reward_long.append(encoder_data_aligned[j][pos_time-left:pos_time+right])
                   
       
           
       if len(trajectory_reward_short) == 0:
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_reward_short.append(nan_pad)
           
       if len(trajectory_reward_long) == 0:
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_reward_long.append(nan_pad)
           
       time_reward = np.linspace(-left/1000, right/1000 , right+left)
           
       trajectory_short.append(list(np.mean(np.array(trajectory_reward_short) , axis = 0)))
       trajectory_long.append(list(np.mean(np.array(trajectory_reward_long) , axis = 0)))
           
    plot_trajectory(axs , time_reward , trajectory_short , 'short' , 1 , title)
    plot_trajectory(axs , time_reward , trajectory_long , 'long' , 1 , title)
       
       