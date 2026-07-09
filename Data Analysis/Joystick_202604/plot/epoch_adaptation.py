# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:46:52 2024

@author: saminnaji3
"""

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

def plot_curves_all(axs,date, subject ,short_delay_first, short_delay_last, long_delay_first, long_delay_last):
    
    size = 3
    x_first = np.arange(0 ,len(short_delay_first))/len(short_delay_first)
    x_last = np.arange(0 ,len(short_delay_last))/len(short_delay_last)+2
    cmap = plt.cm.YlOrBr
    norm = plt.Normalize(vmin=0, vmax=max(len(short_delay_last),len(short_delay_last))+1)
    color1 = []
    for i in range(max(len(short_delay_last),len(short_delay_last))):
        color1.append(np.array(cmap(norm(i+1))))
        axs.plot([x_first[i] , x_last[i]],[short_delay_first[i] , short_delay_last[i]], color = np.array(cmap(norm(i))))
    
    axs.scatter(x_first, short_delay_first, color = color1 , label = date , s = size)
    axs.scatter(x_last, short_delay_last, color = color1 , s = size)
    
    x_first = np.arange(0 ,len(long_delay_first))/len(long_delay_first)+5
    x_last = np.arange(0 ,len(long_delay_last))/len(long_delay_last)+7
    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=0, vmax=max(len(long_delay_last),len(long_delay_last))+1)
    color1 = []
    for i in range(max(len(long_delay_last),len(long_delay_last))):
        color1.append(np.array(cmap(norm(i+1))))
        axs.plot([x_first[i] , x_last[i]],[long_delay_first[i] , long_delay_last[i]], color = np.array(cmap(norm(i))))
    
    axs.scatter(x_first, long_delay_first, color = color1, s= size)
    axs.scatter(x_last, long_delay_last, color = color1, s= size)
    
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('trial number')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average Delay for first and last epoches ') 
    
   
    axs.set_xticks([0.5 , 2.5 , 5.5 , 7.5])
    axs.set_xticklabels(['Short First' , 'Short Last' , 'Long First' , 'Long Last'], rotation='vertical')
   

def plot_curves_indivi(axs,dates, subject ,short_delay_first, short_delay_last, long_delay_first, long_delay_last, max_trial_num, num , max_num):
    #colors = ['#4CAF50','#FFB74D','pink','r','#64B5F6','#1976D2']
    x_first = np.arange(0 ,len(short_delay_first))/len(short_delay_first)
    x_last = np.arange(0 ,len(short_delay_last))/len(short_delay_last)+2
    cmap = plt.cm.YlOrBr
    norm = plt.Normalize(vmin=0, vmax=max(len(short_delay_last),len(short_delay_last))+1)
    color1 = []
    size = 3
    for i in range(max(len(short_delay_last),len(short_delay_last))):
        color1.append(np.array(cmap(norm(i+1))))
        axs.plot([x_first[i] , x_last[i]],[short_delay_first[i] , short_delay_last[i]], color = np.array(cmap(norm(i))))
    
    axs.scatter(x_first, short_delay_first, color = color1, s = size)
    axs.scatter(x_last, short_delay_last, color = color1, s = size)
    
    x_first = np.arange(0 ,len(long_delay_first))/len(long_delay_first)+5
    x_last = np.arange(0 ,len(long_delay_last))/len(long_delay_last)+7
    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=0, vmax=max(len(long_delay_last),len(long_delay_last))+1)
    color1 = []
    for i in range(max(len(long_delay_last),len(long_delay_last))):
        color1.append(np.array(cmap(norm(i+1))))
        axs.plot([x_first[i] , x_last[i]],[long_delay_first[i] , long_delay_last[i]], color = np.array(cmap(norm(i))))
    
    axs.scatter(x_first, long_delay_first, color = color1, s = size)
    axs.scatter(x_last, long_delay_last, color = color1, s = size)
    
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('trial number')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average Delay for first and last epoches ' + dates[num]) 
    
   
    axs.set_xticks([0.5 , 2.5 , 5.5 , 7.5])
    axs.set_xticklabels(['Short First' , 'Short Last' , 'Long First' , 'Long Last'], rotation='vertical')
    
    
    
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
   
    max_trial_num = max_trial_num_temp
    
    
    max_num = num_session
    num_plotting =  num_session
    initial_start = 0
    short_first = []
    short_last = []
    long_first = []
    long_last = []
    for k in range(num_session//max_num + 1):
        
        start = k*num_plotting + initial_start
        end = min((k+1)*num_plotting , num_session)+ initial_start
        date = dates[start:end] 
        epoch_len = 10
    
        for i in range(start , end):
            short_delay_first = []
            short_delay_last = []
            
            long_delay_first = []
            long_delay_last = []
            
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
                
                if len(current_block)>1.5*epoch_len:
                    short_delay_first.append(np.nanmean(current_block[0:epoch_len]))
                    short_delay_last.append(np.nanmean(current_block[-epoch_len:]))
                    
            for j in range(len(long_id)):
                current_block = block_data['delay'][i][long_id[j]]
                if len(current_block)>1.5*epoch_len:
                    long_delay_first.append(np.nanmean(current_block[0:epoch_len]))
                    long_delay_last.append(np.nanmean(current_block[epoch_len:]))
                    
            short_first.append(np.mean(short_delay_first))
            short_last.append(np.mean(short_delay_last))
            long_first.append(np.mean(long_delay_first))
            long_last.append(np.mean(long_delay_last))
        plot_curves_all(axs,date, subject ,short_first, short_last, long_first, long_last)
    

    
    
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
        epoch_len = 10
    
        for i in range(start , end):
            short_delay_first = []
            short_delay_last = []
            
            long_delay_first = []
            long_delay_last = []
            
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
                
                if len(current_block)>1.5*epoch_len:
                    short_delay_first.append(np.nanmean(current_block[0:epoch_len]))
                    short_delay_last.append(np.nanmean(current_block[epoch_len:]))
                    
            for j in range(len(long_id)):
                current_block = block_data['delay'][i][long_id[j]]
                if len(current_block)>1.5*epoch_len:
                    long_delay_first.append(np.nanmean(current_block[0:epoch_len]))
                    long_delay_last.append(np.nanmean(current_block[-epoch_len:]))
                
            
            
            plot_curves_indivi(axs[k][(i-start)-(i-start)//num_plotting],date, subject ,short_delay_first, short_delay_last, long_delay_first, long_delay_last,  max_trial_num,(i-start)-(i-start)//num_plotting , num_plotting)
            