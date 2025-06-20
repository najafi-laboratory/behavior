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
    
    
def count_label(session_data, probe, session_label, states, plotting_sessions, block_type , norm=True):
    num_session = len(plotting_sessions)
    counts1 = np.zeros((3 , num_session, len(states)))
    counts = np.zeros((3 , num_session, len(states)))
    count_all = np.zeros((3, len(states)))
    numtrials = np.zeros((3 , num_session))
    num = 0
    if block_type == 'short':
        short_ind = 1
    else:
        short_ind = 2
    for i in plotting_sessions:
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        for j in range(len(session_label[i])):
            if norm:
                if trial_types[j] == short_ind:
                    if session_label[i][j] in states:
                        k = states.index(session_label[i][j])
                        if j <len(session_label[i])-1:
                            if probe[i][j+1] == 1:
                                counts[0 , num , k] = counts[0 , num , k] + 1
                                numtrials[0 , num] = numtrials[0 , num] + 1
                        if probe[i][j] == 1:
                            counts[1 , num , k] = counts[1 , num , k] + 1
                            numtrials[1 , num] = numtrials[1 , num] + 1    
                        if j > 0:
                            if probe[i][j-1] == 1:
                                counts[2 , num , k] = counts[2 , num , k] + 1
                                numtrials[2 , num] = numtrials[2 , num] + 1
                    
                    
        for j in range(len(states)):
            counts1[0 , num , j] = counts[0 , num , j]/numtrials[0 , num]
            counts1[1 , num , j] = counts[1 , num , j]/numtrials[1 , num]
            counts1[2 , num , j] = counts[2 , num , j]/numtrials[2 , num]
            
        num = num + 1
    for j in range(len(states)):
        count_all[0 , j] = np.sum(counts[0 , : , j])/np.sum(numtrials[0 , :])
        count_all[1 , j] = np.sum(counts[1 , : , j])/np.sum(numtrials[1 , :])
        count_all[2 , j] = np.sum(counts[2 , : , j])/np.sum(numtrials[2 , :])
    return counts1, count_all


def plot_trajectory(axs, dates, time , trajectory , num , max_num , title , legend ,sem):
   
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=max_num+1)
    trajectory_mean = np.mean(trajectory , axis = 0)
    if sem == 1:
        trajectory_sem = np.std(trajectory , axis = 0)/np.sqrt(len(trajectory))
        axs.fill_between(time ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = cmap(norm(max_num+1-num)) , alpha = 0.1)
    
    axs.plot(time , trajectory_mean , color = cmap(norm(max_num+1-num)))
    axs.axvline(x = 0, color = 'gray', linestyle='--')
    #axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Time from Press2 (s)')
    axs.set_ylabel('joystick deflection (deg)')
    axs.set_xlim([-2 , 1.5])
    axs.set_title(title)
    # if legend:
    #     axs.legend()
def plot_grand_trajectory(axs, dates, time , trajectory , num , labels, title , legend ,sem, first = 0):
   
    colors = ['gold' , 'k' , 'dodgerblue']
    trajectory_mean = np.mean(trajectory , axis = 0)
    if sem == 1:
        trajectory_sem = np.std(trajectory , axis = 0)/np.sqrt(len(trajectory))
        axs.fill_between(time ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = colors[num] , alpha = 0.1)
    
    axs.plot(time , trajectory_mean , color = colors[num] , label = labels)
    axs.axvline(x = 0, color = 'gray', linestyle='--')
    #axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Time from Press2 (s)')
    axs.set_ylabel('joystick deflection (deg)')
    if first == 0:
        axs.set_xlim([-2 , 1.5])
    else:
        axs.set_xlim([-0.6 , 2.5])
    axs.set_title(title)
    axs.legend()
    
def plot_dist(axs, dates ,delay_preprobe ,delay_probe, delay_postprobe, num , max_num , title_str):
    
    axs.set_xlabel('Delay (s)')
    axs.set_ylabel('Density')
    #axs.set_xlim([0 , 5])
    axs.set_title('Delay Distribution (short dashed, long solid)')
    #axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    
    if title_str == 0:
    
        sns.distplot(delay_preprobe, hist=False , color = 'gold',kde_kws={'linestyle':'--'}, ax = axs)
        sns.distplot(delay_probe, hist=False , color = 'k',kde_kws={'linestyle':'--'}, ax = axs)
        sns.distplot(delay_postprobe, hist=False , color = 'dodgerblue',kde_kws={'linestyle':'--'}, ax = axs)
    else: 
        sns.distplot(delay_preprobe, hist=False , color = 'gold', label = 'Probe-1' , ax = axs)
        sns.distplot(delay_probe, hist=False , color = 'k', label = 'Probe', ax = axs)
        sns.distplot(delay_postprobe, hist=False , color = 'dodgerblue', label = 'Probe+1', ax = axs)
    axs.legend()
    
def plot_grand_dist(axs, dates ,delay_preprobe ,delay_probe, delay_postprobe, num , max_num , title_str):
    
    axs.set_xlabel('Delay (s)')
    axs.set_ylabel('Density')
    #axs.set_xlim([0 , 5])
    
    #axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    
    if title_str == 0:
        axs.set_title('Delay Distribution (short)')
        sns.distplot(delay_preprobe, hist=False , color = 'gold', label = 'Probe-1', ax = axs)
        sns.distplot(delay_probe, hist=False , color = 'k', label = 'Probe', ax = axs)
        sns.distplot(delay_postprobe, hist=False , color = 'dodgerblue',label = 'Probe+1', ax = axs)
        
    else: 
        axs.set_title('Delay Distribution (long)')
        sns.distplot(delay_preprobe, hist=False , color = 'gold', label = 'Probe-1' , ax = axs)
        sns.distplot(delay_probe, hist=False , color = 'k', label = 'Probe', ax = axs)
        sns.distplot(delay_postprobe, hist=False , color = 'dodgerblue', label = 'Probe+1', ax = axs)
    axs.legend()
    
    
def run_outcome(axs , session_data ,  block_type , st = 0):
    
    max_sessions=100
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    num_session = len(dates)
    isSelfTime = session_data['isSelfTimedMode']
    probe = session_data['session_probe']
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
    
    counts, count_all = count_label(session_data, probe, outcomes, states , plotting_sessions , block_type)
    
    session_id = np.arange(len(plotting_sessions)) + 1
    
    probe_neg_bottom = np.cumsum(counts[0 , : , :], axis=1)
    probe_neg_bottom[:,1:] = probe_neg_bottom[:,:-1]
    probe_neg_bottom[:,0] = 0
    
    probe_bottom = np.cumsum(counts[1 , : , :], axis=1)
    probe_bottom[:,1:] = probe_bottom[:,:-1]
    probe_bottom[:,0] = 0
    
    probe_pos_bottom = np.cumsum(counts[2 , : , :], axis=1)
    probe_pos_bottom[:,1:] = probe_pos_bottom[:,:-1]
    probe_pos_bottom[:,0] = 0
    
    width = 0.2
    
    top_ticks = []
    
    axs.axhline(0.5 , linestyle = '--' , color = 'gray')
    
    
    for i in range(len(states)):
        axs.bar(
            session_id-width , counts[0,:,i],
            bottom=probe_neg_bottom[:,i],
            width=width,
            color=colors[i],
            label=states_name[i])
        axs.bar(
            session_id + width, counts[1,:,i],
            bottom=probe_bottom[:,i],
            width=width,
            color=colors[i])
        axs.bar(
            session_id , counts[2,:,i],
            bottom=probe_pos_bottom[:,i],
            width=width,
            color=colors[i])
   
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
    axs.set_xticklabels(dates_label, rotation='vertical' , fontsize = 8)
    axs.set_title('Reward percentage for completed trials across sessions ' + block_type +' (probe-1|probe|probe+1)')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
    
def run_grand_outcome(axs , session_data ,  block_type , st = 0):
    
    max_sessions=100
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    num_session = len(dates)
    isSelfTime = session_data['isSelfTimedMode']
    probe = session_data['session_probe']
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
    
    counts, count_all = count_label(session_data, probe, outcomes, states , plotting_sessions , block_type)
    
    session_id = np.arange(1) + 1
    
    probe_neg_bottom = np.cumsum(count_all[0  , :])
    probe_neg_bottom[1:] = probe_neg_bottom[:-1]
    probe_neg_bottom[0] = 0
    
    probe_bottom = np.cumsum(count_all[1 , :])
    probe_bottom[1:] = probe_bottom[:-1]
    probe_bottom[0] = 0
    
    probe_pos_bottom = np.cumsum(count_all[2 , :])
    probe_pos_bottom[1:] = probe_pos_bottom[:-1]
    probe_pos_bottom[0] = 0
    
    width = 0.2
    
    top_ticks = []
    
    axs.axhline(0.5 , linestyle = '--' , color = 'gray')
    
    
    for i in range(len(states)):
        axs.bar(
            session_id-width , count_all[0,i],
            bottom=probe_neg_bottom[i],
            width=width,
            color=colors[i],
            label=states_name[i])
        axs.bar(
            session_id , count_all[1,i],
            bottom=probe_bottom[i],
            width=width,
            color=colors[i])
        axs.bar(
            session_id + width, count_all[2,i],
            bottom=probe_pos_bottom[i],
            width=width,
            color=colors[i])
   
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Training session')
    
    axs.set_ylabel('Outcome percentages')
    
    axs.set_xticks(np.arange(1)+1)
    tick_index = np.arange(1)+1
    axs.set_xticks(tick_index)
    
    dates_label = []
    
    
    if st == 1:
        dates_label.append('Grand session' + '-ST')
    elif st == 2:
        dates_label.append('Grand session' + '-VG')
    else:
        dates_label.append('Grand session')
    axs.set_xticklabels(dates_label, rotation='vertical' , fontsize = 8)
    axs.set_title('Reward percentage for completed trials all' + block_type +' (probe-1|probe|probe+1)')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)  
    
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
       probe = session_data['session_probe'][i]
       # print(len(encoder_data_aligned))
       trajectory_reward_short = []
       trajectory_reward_long = []
       raw_data = session_data['raw'][i]
       trial_types = raw_data['TrialTypes']
       probe_sign = [1 , 0 , -1]
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
           left = 2000
           right = 1500
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_reward_short.append(nan_pad)
           
       if len(trajectory_reward_long) == 0:
           left = 2000
           right = 1500
           nan_pad = np.zeros(left+right)
           nan_pad[:] = np.nan
           trajectory_reward_long.append(nan_pad)
         
       title_str = ['Probe-1' , 'Probe' , 'Probe+1']
       plot_trajectory(axs[0], dates1 , time_reward , np.array(trajectory_reward_short) ,num-1 , num_plotting , title_str[probe_ind] +' (short)' , 1 , 1)
       plot_trajectory(axs[1], dates1 , time_reward , np.array(trajectory_reward_long) , num-1 , num_plotting , title_str[probe_ind] +' (long)' , 0 , 1)

def run_grand_trajectory(axs, session_data , block_data , probe_ind , st = 0):
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
    trajectory_reward_short = []
    trajectory_reward_long = []
    num_plotting =  min(max_num , num_session)
    initial_start = max(num_session - 3*max_num , 0)
    for i in plotting_sessions:
       dates1.append(dates[i])
       num = num + 1
       outcome_sess = outcomes[i]
       session_raw = session_data['raw'][i]['RawEvents']['Trial']
       delay = session_data['session_comp_delay'][i]
       encoder_data_aligned = session_data['encoder_pos_aligned'][i]
       probe = session_data['session_probe'][i]
       # print(len(encoder_data_aligned))
       
       raw_data = session_data['raw'][i]
       trial_types = raw_data['TrialTypes']
       probe_sign = [1 , 0 , -1]
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
        left = 2000
        right = 1500
        nan_pad = np.zeros(left+right)
        nan_pad[:] = np.nan
        trajectory_reward_short.append(nan_pad)
        
    if len(trajectory_reward_long) == 0:
        left = 2000
        right = 1500
        nan_pad = np.zeros(left+right)
        nan_pad[:] = np.nan
        trajectory_reward_long.append(nan_pad)
      
    title_str = ['Probe-1' , 'Probe' , 'Probe+1']
    plot_grand_trajectory(axs[0], dates1 , time_reward , np.array(trajectory_reward_short) ,probe_ind , title_str[probe_ind] ,'Short' , 1 , 1)
    plot_grand_trajectory(axs[1], dates1 , time_reward , np.array(trajectory_reward_long) , probe_ind  , title_str[probe_ind] ,'Long' , 0 , 1)        


def run_grand_trajectory_first_press(axs, session_data,probe_ind, st):
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
    trajectory_reward_short = []
    trajectory_reward_long = []
    for i in plotting_sessions:
        
       outcome_sess = outcomes[i]
       session_raw = session_data['raw'][i]['RawEvents']['Trial']
       delay = session_data['session_comp_delay'][i]
       encoder_data_aligned = session_data['encoder_pos_aligned'][i]
       probe = session_data['session_probe'][i]
       
       raw_data = session_data['raw'][i]
       trial_types = raw_data['TrialTypes']
       num = num + 1
       for j in range(len(outcome_sess)):
           outcome = outcome_sess[j]
           if j+probe_sign[probe_ind] > 0 and j+probe_sign[probe_ind]<len(outcome_sess):
               if outcome == 'Reward' or outcome == 'DidNotPress2' or outcome == 'latePress2' :
                   press = session_raw[j]['States']['Press1'][0]
                   pos_time = int(np.round(press*1000))
                   left = 1000
                   right = 3500
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
   
    title_str = ['Probe-1' , 'Probe' , 'Probe+1']
    plot_grand_trajectory(axs[0], dates1 , time_reward , np.array(trajectory_reward_short) ,probe_ind , title_str[probe_ind] ,'Short' , 1 , 1, 1)
    plot_grand_trajectory(axs[1], dates1 , time_reward , np.array(trajectory_reward_long) , probe_ind  , title_str[probe_ind] ,'Long' , 0 , 1, 1) 
    
    
def run_grand_trajectory_first_press_rewarded(axs, session_data,probe_ind, st):
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
    trajectory_reward_short = []
    trajectory_reward_long = []
    for i in plotting_sessions:
        
       outcome_sess = outcomes[i]
       session_raw = session_data['raw'][i]['RawEvents']['Trial']
       delay = session_data['session_comp_delay'][i]
       encoder_data_aligned = session_data['encoder_pos_aligned'][i]
       probe = session_data['session_probe'][i]
       
       raw_data = session_data['raw'][i]
       trial_types = raw_data['TrialTypes']
       num = num + 1
       for j in range(len(outcome_sess)):
           outcome = outcome_sess[j]
           if j+probe_sign[probe_ind] > 0 and j+probe_sign[probe_ind]<len(outcome_sess):
               if outcome == 'Reward':
                   press = session_raw[j]['States']['Press1'][0]
                   pos_time = int(np.round(press*1000))
                   left = 1000
                   right = 3500
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
   
    title_str = ['Probe-1' , 'Probe' , 'Probe+1']
    plot_grand_trajectory(axs[0], dates1 , time_reward , np.array(trajectory_reward_short) ,probe_ind , title_str[probe_ind] ,'Short' , 1 , 1, 1)
    plot_grand_trajectory(axs[1], dates1 , time_reward , np.array(trajectory_reward_long) , probe_ind  , title_str[probe_ind] ,'Long' , 0 , 1, 1) 

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
       session_raw = session_data['raw'][i]['RawEvents']['Trial']
       delay = session_data['session_comp_delay'][i]
       encoder_data_aligned = session_data['encoder_pos_aligned'][i]
       probe = session_data['session_probe'][i]
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
    
def run_grand_dist(axs, session_data , block_data , st = 0):
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
       session_raw = session_data['raw'][i]['RawEvents']['Trial']
       delay = session_data['session_comp_delay'][i]
       encoder_data_aligned = session_data['encoder_pos_aligned'][i]
       probe = session_data['session_probe'][i]
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
    plot_grand_dist(axs[0], dates1 ,short_delay_preprobe ,short_delay_probe, short_delay_postprobe,num -1 , num_plotting , 0)
    plot_grand_dist(axs[1], dates1 ,long_delay_preprobe ,long_delay_probe, long_delay_postprobe, num-1 , num_plotting , 1)