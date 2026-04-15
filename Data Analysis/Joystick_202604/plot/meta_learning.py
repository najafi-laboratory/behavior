# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:10:49 2025

@author: saminnaji3
"""

import numpy as np
import matplotlib.pyplot as plt


def IsSelfTimed(session_data):
    isSelfTime = session_data['isSelfTimedMode']
    
    VG = []
    ST = []
    for i in range(0 , len(isSelfTime)):
        if isSelfTime[i][5] == 1 or isSelfTime[i][5] == np.nan:
            ST.append(i)
        else:
            VG.append(i)
    return ST , VG

def plot(axs, trial_number, short_id, plotting_sessions, title):
    session_id = np.arange(0, len(plotting_sessions))
    store_all = []
    store_short = []
    store_long = []
    num = 0
    for i in plotting_sessions:
        data_now = trial_number[i]
        short = []
        long = []
        noise_short = np.random.normal(loc=0.0, scale=0.1, size=len(short_id[i]))
        noise_long = np.random.normal(loc=0.0, scale=0.1, size=len(data_now)-len(short_id[i]))
        for j in range(len(data_now)):
            if j in short_id[i]:
                short.append(data_now[j])
            else:
                long.append(data_now[j])
        axs.scatter(session_id[num]-0.2, np.nanmean(short), color = 'y', s= 12)
        axs.scatter(session_id[num]+0.2, np.nanmean(long), color = 'b', s= 12)
        axs.scatter(session_id[num], np.nanmean(data_now), color = 'k', s= 12)
        axs.scatter(session_id[num]-0.2+noise_short, short, color = 'y', alpha= np.arange(0.2,1,len(short)), s= 3)
        axs.scatter(session_id[num]+0.2+noise_long, long, color = 'b', alpha= np.arange(0.2,1,len(long)), s= 3)
        store_all.append(np.nanmean(data_now))
        store_short.append(np.nanmean(short))
        store_long.append(np.nanmean(long))
        num = num + 1
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('session IDs')
    axs.set_ylabel('Trial Number')
    axs.set_title(title)
    store_short = np.array(store_short)
    store_long = np.array(store_long)
    #axs.plot(session_id , store_all, color = 'k')
    short_clean = ~np.isnan(store_short)
    #print(session_id[short_clean]-0.2 , store_short[short_clean])
    axs.plot(session_id[short_clean]-0.2 , store_short[short_clean], color = 'y')
    long_clean = ~np.isnan(store_long)
    axs.plot(session_id[long_clean]+0.2 , store_long[long_clean], color = 'b')
    axs.set_xticks(np.arange(0, len(plotting_sessions), len(plotting_sessions)//4))
    x_labels = np.arange(0, len(plotting_sessions), len(plotting_sessions)//4)
    axs.set_xticklabels(x_labels)
    
def run(axs1, axs2, axs3, delay_data, session_data, st_seperate = 1):
    delay_vector = delay_data['delay_all']
    
    ST , VG = IsSelfTimed(session_data)
    all_sessions = np.arange(len(delay_vector))
    if st_seperate == 1:
        all_st = [all_sessions , np.array(ST) , np.array(VG)]
    else:
        all_st = [all_sessions]
    all_labels = [' (All)', ' (ST)', ' (VG)']
    
    for st in range(len(all_st)):
        title = 'Adaptation trial number (based on performance)' + all_labels[st]
        plot(axs1[st], delay_data['block_realize'], delay_data['short_id'], all_st[st], title)
        title = 'Adaptation trial number (based on delay change)' + all_labels[st]
        plot(axs2[st], delay_data['delay_change'], delay_data['short_id'], all_st[st], title)
        title = 'Adaptation trial number (based on delay ttset)' + all_labels[st]
        plot(axs3[st], delay_data['delay_ttest'], delay_data['short_id'], all_st[st], title)