# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 14:01:18 2025

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
   
    trajectory_mean = np.nanmean(np.array(trajectory) , axis = 0) 
    trajectory_std = np.nanstd(np.array(trajectory) , axis = 0) / np.sqrt(len(trajectory))
    if title == 'pre': 
        axs.plot(time , trajectory_mean , color = 'dodgerblue' , label = title)
        axs.fill_between(time ,trajectory_mean-trajectory_std , trajectory_mean+trajectory_std, color = 'dodgerblue' , alpha = 0.2)
        axs.axvline(x = 0, color = 'gray', linestyle='--')
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.set_xlabel('Time from Press2 (s)')
        axs.set_ylabel('joystick deflection (deg) (mean +/- sem)')
        #axs.set_xlim([-2 , 1.5])
        #axs.set_title(Title1)
    elif title == 'late': 
        axs.plot(time , trajectory_mean , color = 'gold' , label = title)
        axs.fill_between(time ,trajectory_mean-trajectory_std , trajectory_mean+trajectory_std, color = 'gold' , alpha = 0.2)
    else:
        axs.plot(time , trajectory_mean , color = 'k' , label = title)
        axs.fill_between(time ,trajectory_mean-trajectory_std , trajectory_mean+trajectory_std, color = 'k' , alpha = 0.2)


def run_trajectory(axs, session_data, block_data, st, title):
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
    
    
    trajectory_opto_short = []
    trajectory_opto_long = []
    trajectory_cont_short = []
    trajectory_cont_long = []
    
    trajectory_opto_short_pre = []
    trajectory_opto_long_pre = []
    trajectory_cont_short_pre = []
    trajectory_cont_long_pre = []
    
    trajectory_opto_short_late = []
    trajectory_opto_long_late = []
    trajectory_cont_short_late = []
    trajectory_cont_long_late = []
    
    group = 7
    
    
    for i in plotting_sessions:
        
       outcome_sess = outcomes[i]
       session_raw = session_data['raw'][i]['RawEvents']['Trial']
       delay = session_data['session_comp_delay'][i]
       encoder_data_aligned = session_data['encoder_pos_aligned'][i]
       opto_block_tag = block_data['block_opto'][i]
       raw_data = session_data['raw'][i]
       trial_types = raw_data['TrialTypes']
       num = num + 1
       
       
       start = block_data['start'][i]
       end = block_data['end'][i]
       max_trial_num = np.max(block_data['NumTrials'][i])
       
       for j in range(len(start)):
           if trial_types[start[j]] == 1:
               if opto_block_tag[start[j]] == 1:
                   for t in range(min(start[j]-group, 0), start[j]):
                       press = delay[t] + session_raw[t]['States']['LeverRetract1'][1]
                       if not np.isnan(press):
                           pos_time = int(np.round(press*1000))
                           left = 2000
                           right = 1500
                           if pos_time < left:
                               nan_pad = np.zeros(left-pos_time)
                               nan_pad[:] = np.nan
                               trajectory_opto_short_pre.append(np.append(nan_pad, encoder_data_aligned[t][0:pos_time+right]))
                           else:
                               trajectory_opto_short_pre.append(encoder_data_aligned[t][pos_time-left:pos_time+right])
                    
                   for t in range(start[j], start[j]+group):
                        press = delay[t] + session_raw[t]['States']['LeverRetract1'][1]
                        if not np.isnan(press):
                            pos_time = int(np.round(press*1000))
                            left = 2000
                            right = 1500
                            if pos_time < left:
                                nan_pad = np.zeros(left-pos_time)
                                nan_pad[:] = np.nan
                                trajectory_opto_short.append(np.append(nan_pad, encoder_data_aligned[t][0:pos_time+right]))
                            else:
                                
                                trajectory_opto_short.append(encoder_data_aligned[t][pos_time-left:pos_time+right])
                                
                   for t in range(end[j]-group, end[j]):
                         press = delay[t] + session_raw[t]['States']['LeverRetract1'][1]
                         if not np.isnan(press):
                             pos_time = int(np.round(press*1000))
                             left = 2000
                             right = 1500
                             if pos_time < left:
                                 nan_pad = np.zeros(left-pos_time)
                                 nan_pad[:] = np.nan
                                 trajectory_opto_short_late.append(np.append(nan_pad, encoder_data_aligned[t][0:pos_time+right]))
                             else:
                                 trajectory_opto_short_late.append(encoder_data_aligned[t][pos_time-left:pos_time+right])
                        
               else:
                    for t in range(min(start[j]-group, 0), start[j]):
                        press = delay[t] + session_raw[t]['States']['LeverRetract1'][1]
                        if not np.isnan(press):
                            pos_time = int(np.round(press*1000))
                            left = 2000
                            right = 1500
                            if pos_time < left:
                                nan_pad = np.zeros(left-pos_time)
                                nan_pad[:] = np.nan
                                trajectory_cont_short_pre.append(np.append(nan_pad, encoder_data_aligned[t][0:pos_time+right]))
                            else:
                                trajectory_cont_short_pre.append(encoder_data_aligned[t][pos_time-left:pos_time+right])
                     
                    for t in range(start[j], min(start[j]+group,len(delay))):
                         press = delay[t] + session_raw[t]['States']['LeverRetract1'][1]
                         if not np.isnan(press):
                             pos_time = int(np.round(press*1000))
                             left = 2000
                             right = 1500
                             if pos_time < left:
                                 nan_pad = np.zeros(left-pos_time)
                                 nan_pad[:] = np.nan
                                 trajectory_cont_short.append(np.append(nan_pad, encoder_data_aligned[t][0:pos_time+right]))
                             else:
                                 trajectory_cont_short.append(encoder_data_aligned[t][pos_time-left:pos_time+right])
                                 
                    for t in range(end[j]-group, end[j]):
                          press = delay[t] + session_raw[t]['States']['LeverRetract1'][1]
                          if not np.isnan(press):
                              pos_time = int(np.round(press*1000))
                              left = 2000
                              right = 1500
                              if pos_time < left:
                                  nan_pad = np.zeros(left-pos_time)
                                  nan_pad[:] = np.nan
                                  trajectory_cont_short_late.append(np.append(nan_pad, encoder_data_aligned[t][0:pos_time+right]))
                              else:
                                  trajectory_cont_short_late.append(encoder_data_aligned[t][pos_time-left:pos_time+right])
                                  
                                  
           else:
                if opto_block_tag[start[j]] == 1:
                    for t in range(min(start[j]-group, 0), start[j]):
                        press = delay[t] + session_raw[t]['States']['LeverRetract1'][1]
                        if not np.isnan(press):
                            pos_time = int(np.round(press*1000))
                            left = 2000
                            right = 1500
                            if pos_time < left:
                                nan_pad = np.zeros(left-pos_time)
                                nan_pad[:] = np.nan
                                trajectory_opto_long_pre.append(np.append(nan_pad, encoder_data_aligned[t][0:pos_time+right]))
                            else:
                                trajectory_opto_long_pre.append(encoder_data_aligned[t][pos_time-left:pos_time+right])
                     
                    for t in range(start[j], min(len(delay),start[j]+group)):
                         press = delay[t] + session_raw[t]['States']['LeverRetract1'][1]
                         if not np.isnan(press):
                             pos_time = int(np.round(press*1000))
                             left = 2000
                             right = 1500
                             if pos_time < left:
                                 nan_pad = np.zeros(left-pos_time)
                                 nan_pad[:] = np.nan
                                 trajectory_opto_long.append(np.append(nan_pad, encoder_data_aligned[t][0:pos_time+right]))
                             else:
                                 trajectory_opto_long.append(encoder_data_aligned[t][pos_time-left:pos_time+right])
                                 
                    for t in range(end[j]-group, end[j]):
                          press = delay[t] + session_raw[t]['States']['LeverRetract1'][1]
                          if not np.isnan(press):
                              pos_time = int(np.round(press*1000))
                              left = 2000
                              right = 1500
                              if pos_time < left:
                                  nan_pad = np.zeros(left-pos_time)
                                  nan_pad[:] = np.nan
                                  trajectory_opto_long_late.append(np.append(nan_pad, encoder_data_aligned[t][0:pos_time+right]))
                              else:
                                  #print(len(encoder_data_aligned[t][pos_time-left:pos_time+right]))
                                  trajectory_opto_long_late.append(encoder_data_aligned[t][pos_time-left:pos_time+right])
                         
                else:
                     for t in range(min(start[j]-group, 0), start[j]):
                         press = delay[t] + session_raw[t]['States']['LeverRetract1'][1]
                         if not np.isnan(press):
                             pos_time = int(np.round(press*1000))
                             left = 2000
                             right = 1500
                             if pos_time < left:
                                 nan_pad = np.zeros(left-pos_time)
                                 nan_pad[:] = np.nan
                                 trajectory_cont_long_pre.append(np.append(nan_pad, encoder_data_aligned[t][0:pos_time+right]))
                             else:
                                 trajectory_cont_long_pre.append(encoder_data_aligned[t][pos_time-left:pos_time+right])
                      
                     for t in range(start[j], min(start[j]+group,len(delay))):
                          press = delay[t] + session_raw[t]['States']['LeverRetract1'][1]
                          if not np.isnan(press):
                              pos_time = int(np.round(press*1000))
                              left = 2000
                              right = 1500
                              if pos_time < left:
                                  nan_pad = np.zeros(left-pos_time)
                                  nan_pad[:] = np.nan
                                  trajectory_cont_long.append(np.append(nan_pad, encoder_data_aligned[t][0:pos_time+right]))
                              else:
                                  trajectory_cont_long.append(encoder_data_aligned[t][pos_time-left:pos_time+right])
                                  
                     for t in range(end[j]-group, end[j]):
                           press = delay[t] + session_raw[t]['States']['LeverRetract1'][1]
                           if not np.isnan(press):
                               pos_time = int(np.round(press*1000))
                               left = 2000
                               right = 1500
                               if pos_time < left:
                                   nan_pad = np.zeros(left-pos_time)
                                   nan_pad[:] = np.nan
                                   trajectory_cont_long_late.append(np.append(nan_pad, encoder_data_aligned[t][0:pos_time+right]))
                               else:
                                   trajectory_cont_long_late.append(encoder_data_aligned[t][pos_time-left:pos_time+right])
                
               
       
           
    time_reward = np.linspace(-left/1000, right/1000 , right+left)
    plot_trajectory(axs[0] , time_reward , trajectory_cont_short_late , 'late' , 1 , title)
    plot_trajectory(axs[0] , time_reward , trajectory_cont_short , 'early' , 1 , title)
    plot_trajectory(axs[0] , time_reward , trajectory_cont_short_pre , 'pre' , 1 , title)
    axs[0].set_title('control short block')
    
    plot_trajectory(axs[1] , time_reward , trajectory_opto_short_late , 'late' , 1 , title)
    plot_trajectory(axs[1] , time_reward , trajectory_opto_short , 'early' , 1 , title)
    plot_trajectory(axs[1] , time_reward , trajectory_opto_short_pre , 'pre' , 1 , title)
    axs[1].set_title('opto short block')
    
    plot_trajectory(axs[2] , time_reward , trajectory_cont_long_late , 'late' , 1 , title)
    plot_trajectory(axs[2] , time_reward , trajectory_cont_long , 'early' , 1 , title)
    plot_trajectory(axs[2] , time_reward , trajectory_cont_long_pre , 'pre' , 1 , title)
    axs[2].set_title('control long block')
    
    plot_trajectory(axs[3] , time_reward , trajectory_opto_long_late , 'late' , 1 , title)
    plot_trajectory(axs[3] , time_reward , trajectory_opto_long , 'early' , 1 , title)
    plot_trajectory(axs[3] , time_reward , trajectory_opto_long_pre , 'pre' , 1 , title)
    axs[3].set_title('opto long block')
       

       
       