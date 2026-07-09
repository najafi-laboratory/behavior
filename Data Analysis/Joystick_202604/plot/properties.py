# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:35:54 2025

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
from scipy.signal import savgol_filter

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
           pos_smooth = savgol_filter(encoder_data_aligned[j], window_length=40, polyorder=3)
           velocity = np.diff(pos_smooth)
           velocity_smooth = savgol_filter(velocity, window_length=40, polyorder=1)
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
       
       