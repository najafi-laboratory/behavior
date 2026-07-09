# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:24:03 2024

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


def plot_curves_indvi(axs,dates ,df, color , num , max_num):
    #colors = ['#4CAF50','#FFB74D','pink','r','#64B5F6','#1976D2']
    
    
    
    df_sorted = df.sort_values(by='time')
    block_num = df_sorted['block_id'].unique()
    print(sorted(block_num))
    block_num = sorted(block_num)
        
    if color ==1 :
        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=-0.2, vmax=len(block_num))
        for i in range(len(block_num)):
            print(norm(i))
            axs.scatter(df_sorted[~df_sorted['block_id'].isin([block_num[i]])]['time'], df_sorted[~df_sorted['block_id'].isin([block_num[i]])]['delay'], color= cmap(norm(i)),s = 3)
        
    else:
        cmap = plt.cm.Blues
        norm = plt.Normalize(vmin=0, vmax=len(block_num))
        for i in range(len(block_num)):
            axs.scatter(df_sorted[~df_sorted['block_id'].isin([block_num[i]])]['time'], df_sorted[~df_sorted['block_id'].isin([block_num[i]])]['delay'], color= cmap(norm(i)),s = 3)
        #axs.colorbar(orientation="horizontal")
    
    axs.set_ylim([0,4])
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('trial number')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average Delay for each trial position ' + dates[num])  

def plot_average_indvi(axs,dates ,df, color , num , max_num):
    #colors = ['#4CAF50','#FFB74D','pink','r','#64B5F6','#1976D2']
    
    
    w = 10
    df_sorted = df.sort_values(by='time')
    filtered_df = df_sorted[~df_sorted['delay'].isin([np.nan])]
    average = np.convolve(filtered_df['delay'], np.ones(w), 'valid') / w
    x_value = np.linspace(0,1,len(average))
        
    if color ==1 :
        axs.plot(x_value , average, color= 'y', label = 'short')
        axs.tick_params(tick1On=False)
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.set_xlabel('trial number')
        axs.set_ylabel('delay (s)')
        axs.set_title('Moving Average Delay for each trial position ' + dates[num])  
    else:
        axs.plot(x_value , average, color= 'b', label =  'long')
        axs.legend()

    
    

def plot_var_indvi(axs ,dates ,df, color , num , max_num):
    #colors = ['#4CAF50','#FFB74D','pink','r','#64B5F6','#1976D2']
    
    
    w = 10
    df_sorted = df.sort_values(by='time')
    filtered_df = df_sorted[~df_sorted['delay'].isin([np.nan])]
    std = filtered_df['delay'].rolling(w).std()
    x_value = np.linspace(0,1,len(std))
        
    if color ==1 :
        axs.plot(x_value , std, color= 'y', label = 'short')
        axs.tick_params(tick1On=False)
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.set_xlabel('trial number')
        axs.set_ylabel('delay (s)')
        axs.set_title('STD of Delay over block ' + dates[num])  
    else:
        axs.plot(x_value , std, color= 'b', label =  'long')
        axs.legend()
    

def run_indvi(axs,axs1,axs2,session_data , block_data):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    
    num_session = len(block_data['NumBlocks'])
    max_num = 6
    num_plotting =  min(max_num , num_session)
    initial_start = max(num_session - 3*max_num , 0)
    for k in range(num_session//max_num + 1):
        
        start = k*num_plotting + initial_start
        end = min((k+1)*num_plotting , num_session)+ initial_start
        date = dates[start:end] 
        for i in range(start , end):
            short_id = block_data['short_id'][i]
            long_id = block_data['long_id'][i]
            delay_data = block_data['delay'][i]
            
            short_delay_data = []
            long_delay_data = []
            short_trial_num = []
            long_trial_num = []
            short_id_all = []
            long_id_all = []
            
            for j in range(len(short_id)):
                short_delay_data.append(block_data['delay'][i][short_id[j]])
                short_trial_num.append(np.linspace(0,1,len(block_data['delay'][i][short_id[j]])))
                short_id_all.append(short_id[j]*np.ones(len(block_data['delay'][i][short_id[j]])))
                    
            for j in range(len(long_id)):
                long_delay_data.append(block_data['delay'][i][long_id[j]])
                long_trial_num.append(np.linspace(0,1,len(block_data['delay'][i][long_id[j]])))
                long_id_all.append(long_id[j]*np.ones(len(block_data['delay'][i][long_id[j]])))
            
            short_delay_data = np.concatenate(short_delay_data)
            long_delay_data = np.concatenate(long_delay_data)
            short_trial_num = np.concatenate(short_trial_num)
            long_trial_num = np.concatenate(long_trial_num)
            short_id_all = np.concatenate(short_id_all)
            long_id_all = np.concatenate(long_id_all)
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
            
            plot_curves_indvi(axs[0][(i-start)-(i-start)//num_plotting], date ,df_short, 1, (i-start)-(i-start)//num_plotting , num_plotting)
            plot_curves_indvi(axs[1][(i-start)-(i-start)//num_plotting], date , df_long,2 ,  (i-start)-(i-start)//num_plotting , num_plotting)
            plot_average_indvi(axs1[(i-start)-(i-start)//num_plotting], date ,df_short, 1, (i-start)-(i-start)//num_plotting , num_plotting)
            plot_average_indvi(axs1[(i-start)-(i-start)//num_plotting], date ,df_long,2 , (i-start)-(i-start)//num_plotting , num_plotting)
            plot_var_indvi(axs2[(i-start)-(i-start)//num_plotting], date ,df_short,1 , (i-start)-(i-start)//num_plotting , num_plotting)
            plot_var_indvi(axs2[(i-start)-(i-start)//num_plotting], date ,df_long,2 , (i-start)-(i-start)//num_plotting , num_plotting)