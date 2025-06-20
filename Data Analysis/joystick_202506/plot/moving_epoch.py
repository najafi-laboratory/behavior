# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:40:18 2025

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
        if isSelfTime[i][5] == 1:
            ST.append(i)
        else:
            VG.append(i)
    return ST , VG

def moving_average(df , w):
    average = []
    trial = [0 , 4 , 9 , 14]
    for i in trial:
        temp = df[(df['time'].between(i , i + w))]
        average.append(temp['delay'])
    return average

def plot_delay_adat(axs , data , num , max_num , title , lim , all = 0):
    
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=max_num+1)
    
    #axs.scatter(data['time'] , data['delay'] , color = cmap(norm(num)) , alpha = 0.2 , s = 2)
    if all: 
        w = 7
        df_sorted = data.sort_values(by='time')
        filtered_df = df_sorted[~df_sorted['delay'].isin([np.nan])]
        average = moving_average(filtered_df , w)
        for k in range(len(average)):
            if lim:
                sns.distplot(a=average[k], hist=False , color = cmap(norm(max_num+1-(num))) , kde_kws={'linestyle':'--' , 'linewidth' :4}, ax = axs[k])
            else:
                sns.distplot(a=average[k], hist=False , color = cmap(norm(max_num+1-(num))) , kde_kws={'linewidth' :4},ax = axs[k])
        
        
    else:
        w = 7
        df_sorted = data.sort_values(by='time')
        filtered_df = df_sorted[~df_sorted['delay'].isin([np.nan])]
        average = moving_average(filtered_df , w)
        #axs[0].set_ylim([ 0, 7])
        for k in range(len(average)):
            if lim:
                sns.distplot(a=average[k], hist=False , color = cmap(norm(max_num+1-(num))) , kde_kws={'linestyle':'--'}, ax = axs[k])
            else:
                sns.distplot(a=average[k], hist=False , color = cmap(norm(max_num+1-(num))) , ax = axs[k])
        
            
            axs[k].tick_params(tick1On=False)
            axs[k].spines['right'].set_visible(False)
            axs[k].spines['top'].set_visible(False)
            axs[k].set_xlabel('Time from Press2 (s)')
            axs[k].set_ylabel('density')
            axs[k].set_title('Delay Distribution (short dashed , long solid) Epoch' + str(k))
            axs[k].set_xlim([-0.5 , 1.6])
        
        
def run_delay_adapt(axs, session_data, block_data, st):
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
    for i in plotting_sessions:
        dates1.append(dates[i])
        
    num_session = len(plotting_sessions)
    all_short = []
    all_long = []
    all_short_time = []
    all_long_time = []
    
    for i in plotting_sessions:
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        delay_data = block_data['delay'][i]
        num = num + 1
        
        short_delay_data = []
        long_delay_data = []
        short_trial_num = []
        long_trial_num = []
        short_id_all = []
        long_id_all = []
        len_block_short = []
        len_block_long = []
        
        if short_id[0] == 0:
            short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
        
        for j in range(len(short_id)):
            short_delay_data.append(block_data['delay'][i][short_id[j]])
            short_trial_num.append(np.arange(0,len(block_data['delay'][i][short_id[j]])))
            short_id_all.append(short_id[j]*np.ones(len(block_data['delay'][i][short_id[j]])))
            len_block_short.append(len(block_data['delay'][i][short_id[j]]))
                
        for j in range(len(long_id)):
            long_delay_data.append(block_data['delay'][i][long_id[j]])
            long_trial_num.append(np.arange(0,len(block_data['delay'][i][long_id[j]])))
            long_id_all.append(long_id[j]*np.ones(len(block_data['delay'][i][long_id[j]])))
            len_block_long.append(len(block_data['delay'][i][long_id[j]]))
        
        short_delay_data = np.concatenate(short_delay_data)
        long_delay_data = np.concatenate(long_delay_data)
        short_trial_num = np.concatenate(short_trial_num)
        long_trial_num = np.concatenate(long_trial_num)
        short_id_all = np.concatenate(short_id_all)
        long_id_all = np.concatenate(long_id_all)
        
        min_val = np.nanmin(short_delay_data)
        max_val = np.nanmax(short_delay_data)
        short_delay_data = [(x - min_val) / (max_val - min_val) for x in short_delay_data]
        all_short.append(short_delay_data)
        min_val = np.nanmin(long_delay_data)
        max_val = np.nanmax(long_delay_data)
        long_delay_data = [(x - min_val) / (max_val - min_val) for x in long_delay_data]
        all_long.append(long_delay_data)
        all_short_time.append(short_trial_num)
        all_long_time.append(long_trial_num)
        
        data_short = {
            'delay':short_delay_data,
            'time':short_trial_num,
            'block_id':short_id_all}
        df_short = pd.DataFrame(data_short)
        
        data_long = {
            'delay':long_delay_data,
            'time':long_trial_num,
            'block_id':long_id_all}
        df_long = pd.DataFrame(data_long)
        
        plot_delay_adat(axs , df_short , num-1 , num_session , 'Delay Adaptation (Short)' , 0 )
        plot_delay_adat(axs , df_long  , num-1 , num_session , 'Delay Adaptation (long)' , 1)
    
    all_short = np.concatenate(all_short)
    all_long = np.concatenate(all_long)
    all_short_time = np.concatenate(all_short_time)
    all_long_time = np.concatenate(all_long_time)
    data_short = {
        'delay':all_short,
        'time':all_short_time}
    df_short = pd.DataFrame(data_short)
    
    data_long = {
        'delay':all_long,
        'time':all_long_time}
    df_long = pd.DataFrame(data_long)
    
    plot_delay_adat(axs , df_short , num-1 , num_session , 'Delay Adaptation (Short)' , 0  , 1)
    plot_delay_adat(axs , df_long  , num-1 , num_session , 'Delay Adaptation (long)' , 1 , 1)