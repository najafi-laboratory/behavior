# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:43:05 2024

@author: saminnaji3
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from datetime import date
from matplotlib.lines import Line2D
import re
import seaborn as sns

def plot_curves_all(axs, subject ,short_delay_data, long_delay_data , max_trial_num):
    
    mean_all_short = []
    error_all_short = []
    
    mean_all_long = []
    error_all_long = []
    
    for i in range(max_trial_num):
        if (i < len(short_delay_data)):
            mean_all_short.append(np.nanmean(short_delay_data[: , i]))
            error_all_short.append(sem(short_delay_data[: , i], nan_policy='omit'))
              
        if (i < len(long_delay_data)):
            mean_all_long.append(np.nanmean(long_delay_data[: , i]))
            error_all_long.append(sem(long_delay_data[: , i], nan_policy='omit'))
            
         
            
    x_values_short = np.arange(1 ,len(mean_all_short) + 1)
    axs.errorbar(x_values_short , mean_all_short , error_all_short , color = 'y', fmt="o")
    
    x_values_long = np.arange(1 ,len(mean_all_long) + 1)+0.15
    axs.errorbar(x_values_long , mean_all_long , error_all_long , color = 'b', fmt="o")
    
    
    legend_elements = [
    Line2D([0], [0], color='y', marker='o', linestyle='None', markersize=10, label='short'),
    Line2D([0], [0], color='b', marker='o', linestyle='None', markersize=10, label='long')]

    axs.legend(handles=legend_elements, loc='upper right', ncol=5, bbox_to_anchor=(0.5,1))
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('trial number')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average Delay for each trial position across sessions')
    

def plot_curves_indivi(axs,dates, subject ,short_delay_data, short_block_lowerband, short_block_upperband, long_delay_data, long_block_lowerband, long_block_upperband , max_trial_num, num , max_num):
    #colors = ['#4CAF50','#FFB74D','pink','r','#64B5F6','#1976D2']
    mean_all_short = []
    error_all_short = []
    mean_short_lower = []
    mean_short_upper = []
    
    mean_all_long = []
    error_all_long = []
    mean_long_lower = []
    mean_long_upper = []
    
    
    for i in range(max_trial_num):
        mean_all_short.append(np.nanmean(short_delay_data[: , i]))
        error_all_short.append(sem(short_delay_data[: , i], nan_policy='omit'))
        mean_short_lower.append(np.nanmean(short_block_lowerband[: , i]))
        mean_short_upper.append(np.nanmean(short_block_upperband[: , i]))
        
        mean_all_long.append(np.nanmean(long_delay_data[: , i]))
        error_all_long.append(sem(long_delay_data[: , i], nan_policy='omit'))
        mean_long_lower.append(np.nanmean(long_block_lowerband[: , i]))
        mean_long_upper.append(np.nanmean(long_block_upperband[: , i]))
        
    limit_up = max(np.nanmax(mean_all_long) , np.nanmax(mean_all_short))
    limit_down = min(np.nanmin(mean_all_long) , np.nanmin(mean_all_short))
    #axs.set_ylim([min(0 , limit_down-0.01) , max(limit_up+0.01 , 2)])        
    x_values_short = np.arange(1 ,len(mean_all_short) + 1)
    #axs.errorbar(x_values_short , mean_all_short , error_all_short , color = colors[num], fmt="o")
    axs.plot(x_values_short , mean_all_short , color = 'y')
    axs.scatter(x_values_short , mean_all_short, color= 'y')
    short_upper = np.minimum(mean_short_upper , (limit_up+0.01)*np.ones(len(mean_short_upper)))
    axs.fill_between(x_values_short , mean_short_lower,short_upper, color= 'y' , alpha = 0.1)
    
    x_values_long = np.arange(1 ,len(mean_all_long) + 1)+0.15
    #axs.errorbar(x_values_long , mean_all_long , error_all_long , color = colors[num],fmt="o", mfc='none', mec= colors[num])
    axs.scatter(x_values_long , mean_all_long , color = 'b')
    axs.plot(x_values_long , mean_all_long , color = 'b')
    long_upper = np.minimum(mean_long_upper , (limit_up+0.01)*np.ones(len(mean_long_upper)))
    axs.fill_between(x_values_long , mean_long_lower,long_upper, color= 'b' , alpha = 0.1)
    
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('trial number')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average Delay for each trial position ' + dates[num])  

    
def run(axs , session_data , block_data):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    
    num_session = len(block_data['NumBlocks'])
    all_lens = np.concatenate(block_data['NumTrials'][:])
    unique = set(all_lens)
    max_trial_num_temp = max(all_lens)
    max_trial_num = max_trial_num_temp
    i = 1
    while len(np.where(all_lens >= max_trial_num_temp)[0])/len(all_lens) < 0.5:
        max_trial_num_temp = all_lens[-i-1]
        i = i-1 
    max_trial_num = max_trial_num_temp
            
    
    short_delay_data = []
    long_delay_data = []
    for i in range(num_session):
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        for j in range(len(short_id)):
            current_block = block_data['delay'][i][short_id[j]]
            if len(current_block) < max_trial_num:
                nan_pad = np.zeros(max_trial_num-len(current_block))
                nan_pad[:] = np.nan
                current_block = np.append(current_block , nan_pad)
            short_delay_data.append(current_block[0:max_trial_num])
                
                
        for j in range(len(long_id)):
            current_block = block_data['delay'][i][long_id[j]]
            if len(current_block) < max_trial_num:
                nan_pad = np.zeros(max_trial_num-len(current_block))
                nan_pad[:] = np.nan
                current_block = np.append(current_block , nan_pad)
            long_delay_data.append(current_block[0:max_trial_num])
            
     
    short_block_data = np.zeros([1 , max_trial_num]) 
    short_block_data[:] = np.nan
    short_block_data = np.array(short_block_data)
    long_block_data = np.zeros([1 , max_trial_num]) 
    long_block_data[:] = np.nan
    
    for i in range(len(short_delay_data)):
        short_block_data = np.insert(short_block_data , 0 , np.array(short_delay_data[i]) , axis = 0)
        
        
    for i in range(len(long_delay_data)):
        long_block_data = np.insert(long_block_data , 0 , long_delay_data[i], axis = 0)
    
    
    plot_curves_all(axs, subject ,short_block_data, long_block_data , max_trial_num)
    #plot_dist_all(axs1, axs2 , subject ,short_block_data, long_block_data)
    
def run_exclude_first_block(axs , session_data , block_data):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    
    num_session = len(block_data['NumBlocks'])
    all_lens = np.concatenate(block_data['NumTrials'][:])
    unique = set(all_lens)
    max_trial_num_temp = max(all_lens)
    max_trial_num = max_trial_num_temp
    i = 1
    while len(np.where(all_lens >= max_trial_num_temp)[0])/len(all_lens) < 0.5:
        max_trial_num_temp = all_lens[-i-1]
        i = i-1 
    max_trial_num = max_trial_num_temp
    
    short_delay_data = []
    
    long_delay_data = []
    for i in range(num_session):
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        if short_id[0] == 0:
            short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
        for j in range(len(short_id)):
            current_block = block_data['delay'][i][short_id[j]]
            
            if len(current_block) < max_trial_num:
                nan_pad = np.zeros(max_trial_num-len(current_block))
                nan_pad[:] = np.nan
                current_block = np.append(current_block , nan_pad)
            short_delay_data.append(current_block[0:max_trial_num])
                
                
        for j in range(len(long_id)):
            current_block = block_data['delay'][i][long_id[j]]
            if len(current_block) < max_trial_num:
                nan_pad = np.zeros(max_trial_num-len(current_block))
                nan_pad[:] = np.nan
                current_block = np.append(current_block , nan_pad)
            long_delay_data.append(current_block[0:max_trial_num])
            
     
    short_block_data = np.zeros([1 , max_trial_num]) 
    short_block_data[:] = np.nan
    short_block_data = np.array(short_block_data)
    long_block_data = np.zeros([1 , max_trial_num]) 
    long_block_data[:] = np.nan
    
    for i in range(len(short_delay_data)):
        short_block_data = np.insert(short_block_data , 0 , np.array(short_delay_data[i]) , axis = 0)
        
        
    for i in range(len(long_delay_data)):
        long_block_data = np.insert(long_block_data , 0 , long_delay_data[i], axis = 0)
    
    
    plot_curves_all(axs, subject ,short_block_data, long_block_data , max_trial_num)
    axs.set_title('Average Delay for each trial position across sessions (First block excluded)')
    #plot_dist_all(axs1, axs2 , subject ,short_block_data, long_block_data)
    
    
    
    
def run_indivi(axs , session_data , block_data):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    
    num_session = len(block_data['NumBlocks'])
    
    
    all_lens = np.concatenate(block_data['NumTrials'][:])
    unique = set(all_lens)
    max_trial_num_temp = max(all_lens)
    max_trial_num = max_trial_num_temp
    i = 1
   
    max_trial_num = max_trial_num_temp
    
    
    max_num = 6
    num_plotting =  min(max_num , num_session)
    initial_start = max(num_session - 3*max_num , 0)
    for k in range(num_session//max_num + 1):
        
        start = k*num_plotting + initial_start
        end = min((k+1)*num_plotting , num_session)+ initial_start
        date = dates[start:end] 
    
        for i in range(start , end):
            short_delay_data = []
            short_lowerband = []
            short_upperband = []
            
            long_delay_data = []
            long_lowerband = []
            long_upperband = []
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
                current_lowerband = block_data['LowerBand'][i][short_id[j]]
                current_upperband = block_data['UpperBand'][i][short_id[j]]
                if len(current_block) < max_trial_num:
                    nan_pad = np.zeros(max_trial_num-len(current_block))
                    nan_pad[:] = np.nan
                    current_block = np.append(current_block , nan_pad)
                    current_lowerband = np.append(current_lowerband , nan_pad)
                    current_upperband = np.append(current_upperband , nan_pad)
                short_delay_data.append(current_block[0:max_trial_num])
                short_lowerband.append(current_lowerband[0:max_trial_num])
                short_upperband.append(current_upperband[0:max_trial_num])
                    
            for j in range(len(long_id)):
                current_block = block_data['delay'][i][long_id[j]]
                current_lowerband = block_data['LowerBand'][i][long_id[j]]
                current_upperband = block_data['UpperBand'][i][long_id[j]]
                if len(current_block) < max_trial_num:
                    nan_pad = np.zeros(max_trial_num-len(current_block))
                    nan_pad[:] = np.nan
                    current_block = np.append(current_block , nan_pad)
                    current_lowerband = np.append(current_lowerband , nan_pad)
                    current_upperband = np.append(current_upperband , nan_pad)
                long_delay_data.append(current_block[0:max_trial_num])
                long_lowerband.append(current_lowerband[0:max_trial_num])
                long_upperband.append(current_upperband[0:max_trial_num])
                
                
         
            short_block_data = np.zeros([1 , max_trial_num]) 
            short_block_data[:] = np.nan
            short_block_data = np.array(short_block_data)
            long_block_data = np.zeros([1 , max_trial_num]) 
            long_block_data[:] = np.nan
            
            short_block_lowerband = np.zeros([1 , max_trial_num]) 
            short_block_lowerband[:] = np.nan
            short_block_lowerband = np.array(short_block_data)
            long_block_lowerband = np.zeros([1 , max_trial_num]) 
            long_block_lowerband[:] = np.nan
            
            short_block_upperband = np.zeros([1 , max_trial_num]) 
            short_block_upperband[:] = np.nan
            short_block_upperband = np.array(short_block_data)
            long_block_upperband = np.zeros([1 , max_trial_num]) 
            long_block_upperband[:] = np.nan
            
            for s in range(len(short_delay_data)):
                short_block_data = np.insert(short_block_data , 0 , np.array(short_delay_data[s]) , axis = 0)
                short_block_lowerband = np.insert(short_block_lowerband , 0 , np.array(short_lowerband[s]) , axis = 0)
                short_block_upperband = np.insert(short_block_upperband , 0 , np.array(short_upperband[s]) , axis = 0)
                
                
            for s in range(len(long_delay_data)):
                long_block_data = np.insert(long_block_data , 0 , long_delay_data[s], axis = 0)
                long_block_lowerband = np.insert(long_block_lowerband , 0 , np.array(long_lowerband[s]) , axis = 0)
                long_block_upperband = np.insert(long_block_upperband , 0 , np.array(long_upperband[s]) , axis = 0)
            
            
            plot_curves_indivi(axs[k][(i-start)-(i-start)//num_plotting],date, subject ,short_block_data, short_block_lowerband, short_block_upperband, long_block_data, long_block_lowerband, long_block_upperband , max_trial_num,(i-start)-(i-start)//num_plotting , num_plotting)
            