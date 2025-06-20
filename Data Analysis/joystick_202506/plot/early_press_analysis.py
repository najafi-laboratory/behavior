# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:57:52 2025

@author: saminnaji3
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 14:05:37 2024

@author: saminnaji3
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from datetime import date
from matplotlib.lines import Line2D
import re
import seaborn as sns





def IsSelfTimed(session_data):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    isSelfTime = session_data['isSelfTimedMode']
    
    VG = []
    ST = []
    for i in range(0 , len(isSelfTime)):
        if isSelfTime[i][5] == 1:
            ST.append(i)
        else:
            VG.append(i)
    return ST , VG
    
    
def early_detect(outcome_sess):
    probe = outcome_sess
    for i in range(len(outcome_sess)):
        if outcome_sess[i] == 'EarlyPress2':
            probe[i] = 1
            #print(1)
        else:
            probe[i] = 0
    return probe


def plot_trajectory(axs, dates, time , trajectory , num , max_num , title , legend):
   
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=max_num+1)
    
    axs.plot(time , trajectory , color = cmap(norm(max_num+1-num)) )
    axs.axvline(x = 0, color = 'gray', linestyle='--')
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Time from Press2 (s)')
    axs.set_ylabel('joystick deflection (deg)')
    axs.set_xlim([-2 , 1.5])
    axs.set_title(title)
    # if legend:
    #     axs.legend()

def plot_dist(axs, dates ,delay_preprobe ,delay_probe, delay_postprobe, num , max_num , title_str):
    
    axs.set_xlabel('Delay (s)')
    axs.set_ylabel('Density')
    #axs.set_xlim([0 , 5])
    axs.set_title('Delay Distribution (short dashed, long solid)')
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    # cmap_pre = plt.cm.YlOrBr
    # norm_pre = plt.Normalize(vmin=0, vmax=max_num+1)
    
    # cmap = plt.cm.Grays
    # norm = plt.Normalize(vmin=0, vmax=max_num+1)
    
    # cmap_post = plt.cm.Blues
    # norm_post = plt.Normalize(vmin=0, vmax=max_num+1)
    
    # sns.distplot(delay_preprobe, hist=False , color = cmap_pre(norm_pre(max_num+1-num)), ax = axs)
    # sns.distplot(delay_probe, hist=False , color = cmap(norm(max_num+1-num)), ax = axs)
    # sns.distplot(delay_postprobe, hist=False , color = cmap_post(norm_post(max_num+1-num)), ax = axs)
    
    if title_str == 0:
    
        sns.distplot(delay_preprobe, hist=False , color = 'gold',kde_kws={'linestyle':'--'}, ax = axs)
        sns.distplot(delay_probe, hist=False , color = 'k',kde_kws={'linestyle':'--'}, ax = axs)
        sns.distplot(delay_postprobe, hist=False , color = 'dodgerblue',kde_kws={'linestyle':'--'}, ax = axs)
    else: 
        sns.distplot(delay_preprobe, hist=False , color = 'gold', label = 'EarlyPress-1' , ax = axs)
        sns.distplot(delay_probe, hist=False , color = 'k', label = 'EarlyPress', ax = axs)
        sns.distplot(delay_postprobe, hist=False , color = 'dodgerblue', label = 'EarlyPress+1', ax = axs)
    axs.legend()

    
def run_trajectory(axs, session_data , block_data , probe_ind , st = 0):
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
       probe = early_detect(outcome_sess)
       # print(len(encoder_data_aligned))
       trajectory_reward_short = []
       trajectory_reward_long = []
       raw_data = session_data['raw'][i]
       trial_types = raw_data['TrialTypes']
       probe_sign = [1 , 0 , -1]
       for j in range(len(outcome_sess)):
           outcome = outcome_sess[j]
           if j+probe_sign[probe_ind] > 0 and j+probe_sign[probe_ind]<len(outcome_sess):
               if outcome == 'Reward' or  outcome == 'EaryPress2':
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
           left = 2000
           right = 1500
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_reward_short.append(nan_pad)
           time_reward = np.linspace(-left/1000, right/1000 , right+left)
           
       if len(trajectory_reward_long) == 0:
           left = 2000
           right = 1500
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           time_reward = np.linspace(-left/1000, right/1000 , right+left)
         
       title_str = ['Early Press-1' , 'Early Press' , 'Early Press+1']
       plot_trajectory(axs[0], dates1 , time_reward , np.mean(np.array(trajectory_reward_short) , axis = 0) ,num , num_plotting , title_str[probe_ind] +' (short)' , 1)
       plot_trajectory(axs[1], dates1 , time_reward , np.mean(np.array(trajectory_reward_long) , axis = 0), num , num_plotting , title_str[probe_ind] +' (long)' , 0)
       
def run_dist(axs, session_data , block_data , st = 0):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    
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
    short_delay_preprobe = []
    long_delay_preprobe = []
    short_delay_probe = []
    long_delay_probe = []
    short_delay_postprobe = []
    long_delay_postprobe = []
    
    for i in plotting_sessions:
       dates1.append(dates[i])
       num = num + 1
       outcome_sess = outcomes[i]
       #print(outcome_sess)
       session_raw = session_data['raw'][i]['RawEvents']['Trial']
       delay = session_data['session_comp_delay'][i]
       encoder_data_aligned = session_data['encoder_pos_aligned'][i]
       probe = early_detect(outcome_sess)
       
       raw_data = session_data['raw'][i]
       trial_types = raw_data['TrialTypes']
       probe_sign = [1 , 0 , -1]
       for j in range(len(outcome_sess)):
           outcome = outcome_sess[j]
           
           if probe[j] == 1:
               if trial_types[j] == 1:
                   short_delay_probe.append(delay[j])
               else:
                   long_delay_probe.append(delay[j])
                   
           if j > 0:
               if probe[j-1] == 1:
                   if trial_types[j] == 1:
                       short_delay_postprobe.append(delay[j])
                   else:
                       long_delay_postprobe.append(delay[j])
              
           if j <len(outcome_sess)-1:
               if probe[j+1] == 1:
                   if trial_types[j] == 1:
                       short_delay_preprobe.append(delay[j])
                   else:
                       long_delay_preprobe.append(delay[j])
    plot_dist(axs, dates1 ,short_delay_preprobe ,short_delay_probe, short_delay_postprobe,num -1 , num_plotting , 0)
    plot_dist(axs, dates1 ,long_delay_preprobe ,long_delay_probe, long_delay_postprobe, num-1 , num_plotting , 1)