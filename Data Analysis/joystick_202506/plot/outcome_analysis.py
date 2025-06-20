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
        if isSelfTime[i][5] == 1 or isSelfTime[i][5] == np.nan:
            ST.append(i)
        else:
            VG.append(i)
    return ST , VG
    
    
def outcome_detect(outcome_sess , state):
    probe = np.zeros(len(outcome_sess))
    for i in range(len(outcome_sess)):
        if outcome_sess[i] == state:
            probe[i] = 1
        
    return probe



def plot_dist(axs, dates ,delay_preprobe ,delay_probe, delay_postprobe, num , max_num , title_str , state):
    
    axs.set_xlabel('Delay (s)')
    axs.set_ylabel('Density')
    axs.set_title('Delay Distribution (short dashed, long solid)')
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    
    if title_str == 0:
    
        sns.distplot(delay_preprobe, hist=False , color = 'gold',kde_kws={'linestyle':'--'}, ax = axs)
        sns.distplot(delay_probe, hist=False , color = 'k',kde_kws={'linestyle':'--'}, ax = axs)
        sns.distplot(delay_postprobe, hist=False , color = 'dodgerblue',kde_kws={'linestyle':'--'}, ax = axs)
    else: 
        sns.distplot(delay_preprobe, hist=False , color = 'gold', label = state + '-1' , ax = axs)
        sns.distplot(delay_probe, hist=False , color = 'k', label = state, ax = axs)
        sns.distplot(delay_postprobe, hist=False , color = 'dodgerblue', label = state + '+1', ax = axs)
    axs.legend()

    

def run_dist(axs, session_data , block_data , state, st):
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
       
       delay = session_data['session_comp_delay'][i]
       probe = outcome_detect(outcome_sess , state)
       #probe = session_data['session_probe'][i]
       trial_types = session_data['raw'][i]['TrialTypes']
       probe_sign = [1 , 0 , -1]
       for j in range(block_data['short_id'][i][0]+1 ,len(outcome_sess)):
           
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
    #print(np.nanmax(np.array(short_delay_probe)))
    #print(np.nanmin(np.array(long_delay_probe)))
    plot_dist(axs, dates1 ,short_delay_preprobe ,short_delay_probe, short_delay_postprobe,num -1 , num_plotting , 0 , state)
    plot_dist(axs, dates1 ,long_delay_preprobe ,long_delay_probe, long_delay_postprobe, num-1 , num_plotting , 1 , state)
    
    
def run_max_min(axs, session_data , block_data , state, st):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    isSelfTime = session_data['isSelfTimedMode']
    
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
    
    short_delay_postprobe = []
    long_delay_postprobe = []
    
    for i in plotting_sessions:
       dates1.append(dates[i])
       num = num + 1
       outcome_sess = outcomes[i]
       
       delay = session_data['session_comp_delay'][i]
       probe = outcome_detect(outcome_sess , state)
       #probe = session_data['session_probe'][i]
       trial_types = session_data['raw'][i]['TrialTypes']
       probe_sign = [1 , 0 , -1]
       short_delay_probe = []
       long_delay_probe = []
       #print(block_data['end'][i][0]+1)
       
       for j in range(block_data['end'][i][0]+1 , len(outcome_sess)):
           
           if probe[j] == 1:
               if trial_types[j] == 1:
                   short_delay_probe.append(delay[j])
               else:
                   long_delay_probe.append(delay[j])
       
       axs.scatter([num] , [np.nanmax(np.array(short_delay_probe))], color = 'y' , s = 8)
       axs.scatter([num] , [np.nanmin(np.array(long_delay_probe))], color = 'b' , s = 8)
           
    axs.set_xlabel('session')
    axs.set_ylabel('delay (s)')
    #axs.set_title('Delay Distribution (short dashed, long solid)')
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xticks(np.arange(len(plotting_sessions))+1)
    
    dates_label = []
    
    for i in plotting_sessions:
        if isSelfTime[i][5] == 1:
            dates_label.append(dates[i] + '-ST')
        else:
            dates_label.append(dates[i] + '-VG')
    axs.set_xticklabels(dates_label, rotation='vertical')