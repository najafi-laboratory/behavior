# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:50:20 2024

@author: saminnaji3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from datetime import date
from matplotlib.lines import Line2D
import re
import seaborn as sns

def plot_curves_all(axs, subject ,short_delay_data, long_delay_data):
    
    mean_all = []
    error_all = []
    color_all = []
    
    for i in range(max(len(short_delay_data) , len(long_delay_data))):
        if (i < len(short_delay_data)):
            if len(short_delay_data[i]) > 30:
                mean_all.append(np.nanmean(short_delay_data[i]))
                error_all.append(sem(short_delay_data[i], nan_policy='omit'))
                color_all.append('y')
        if (i < len(long_delay_data)):
            if len(long_delay_data[i]) > 30:
                mean_all.append(np.nanmean(long_delay_data[i]))
                error_all.append(sem(long_delay_data[i], nan_policy='omit'))
                color_all.append('b')
         
            
    x_values = np.arange(1 ,len(mean_all) + 1)
    
    axs.scatter(x_values , mean_all , color = color_all)
    
    for pos, y, err, colors in zip(x_values, mean_all, mean_all, color_all):
        axs.errorbar(pos, y, err, color = colors, fmt="o")
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xticks(x_values)
    axs.set_xlabel('block number')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average Delay across sessions')
    
    legend_elements = [
    Line2D([0], [0], color='y', marker='o', linestyle='None', markersize=10, label='short'),
    Line2D([0], [0], color='b', marker='o', linestyle='None', markersize=10, label='long')]

    axs.legend(handles=legend_elements, loc='upper right', ncol=5, bbox_to_anchor=(0.5,1))

def plot_curves_indvi(axs, dates ,delay_data, short_id , long_id , num , max_num):
    
    mean_all = []
    error_all = []
    color_all = []
    
    for i in range(len(delay_data)):
        mean_all.append(np.nanmean(delay_data[i]))
        error_all.append(sem(delay_data[i], nan_policy='omit'))
        if i in short_id:
            color_all.append('y')
        else:
            color_all.append('b')
         
            
    x_values = np.arange(0 ,len(mean_all))*0.17 + (num-1)*5
    
    axs.scatter(x_values , mean_all , color = color_all)
    
    for pos, y, err, colors in zip(x_values, mean_all, mean_all, color_all):
        axs.errorbar(pos, y, err, color = colors, fmt="o")
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    
    axs.set_xticks((np.arange(max_num)-1)*5+1.5)
    
    if num == 0: 
        dates_label = []
        for k in range(max_num):
            
            if len(dates) > k:
                dates_label.append(dates[k])
            else:
                dates_label.append('')
            
        axs.set_xticklabels(dates_label, rotation='vertical')
    axs.set_xlabel('Dates')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average Delay each session')
    
    legend_elements = [
    Line2D([0], [0], color='y', marker='o', linestyle='None', markersize=10, label='short'),
    Line2D([0], [0], color='b', marker='o', linestyle='None', markersize=10, label='long')]

    axs.legend(handles=legend_elements, loc='upper right', ncol=5, bbox_to_anchor=(0.5,1))
    
    
def plot_QC_indvi(axs, dates ,delay_data, short_id , long_id , num , max_num):
    
    mean_all = []
    error_all = []
    color_all = []
    
    for i in range(len(delay_data)):
        mean_all.append(delay_data[i])
        if i in short_id:
            color_all.append('y')
        else:
            color_all.append('b')
         
            
    x_values = np.arange(0 ,len(mean_all))*0.17 + (num-1)*5
    
    axs.scatter(x_values , mean_all , color = color_all)
    
   
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    
    axs.set_xticks((np.arange(max_num)-1)*5+1.5)
    
    if num == 0: 
        dates_label = []
        for k in range(max_num):
            
            if len(dates) > k:
                dates_label.append(dates[k])
            else:
                dates_label.append('')
            
        axs.set_xticklabels(dates_label, rotation='vertical')
    axs.set_xlabel('Dates')
    axs.set_ylabel('delay (s)')
    axs.set_title('Quality Control index')
    
    legend_elements = [
    Line2D([0], [0], color='y', marker='o', linestyle='None', markersize=10, label='short'),
    Line2D([0], [0], color='b', marker='o', linestyle='None', markersize=10, label='long')]

    #axs.legend(handles=legend_elements, loc='upper right', ncol=5, bbox_to_anchor=(0.5,1))
    
    
def plot_dist_all(axs, subject ,short_delay_data, long_delay_data):
 
    for i in range(max(len(short_delay_data) , len(long_delay_data))):
        if (i < len(short_delay_data)):
            if len(short_delay_data[i]) > 30:
                sns.distplot(a=short_delay_data[i], hist=False , color = [1 , 1 , 0 , (i+1)/(len(short_delay_data)-1)])
        if (i < len(long_delay_data)):
            if len(long_delay_data[i]) > 30:
                sns.distplot(a=long_delay_data[i], hist=False , color = [0 , 0.7 , 0.7 , (i+1)/(len(long_delay_data)-1)])
    short_all = np.concatenate(short_delay_data)
    long_all = np.concatenate(long_delay_data)  
    sns.distplot(a=short_all, hist=False , color = 'y',kde_kws={"linewidth": 4})
    sns.distplot(a=long_all, hist=False , color = 'b',kde_kws={"linewidth": 4})
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Delay (s)')
    axs.set_ylabel('Density')
    axs.set_xlim([0 , 2])
    axs.set_title('Delay Distribution for each block all sessions')
    
    
def plot_dist_indvi(axs, dates, short_delay_data , long_delay_data , num , max_num):
    colors = ['#4CAF50','#FFB74D','pink','r','#64B5F6','#1976D2']
    
    
    if len(short_delay_data) > 0: 
        short_all = np.concatenate(short_delay_data)
    else:
        short_all = [np.nan]
    if len(long_delay_data) > 0: 
       long_all = np.concatenate(long_delay_data) 
    else:
        long_all = [np.nan]
     
    sns.distplot(a=short_all, hist=False , color = colors[num] , label = dates[num] , ax = axs)
    sns.distplot(a=long_all, hist=False , color = colors[num],kde_kws={'linestyle':'--'} , ax = axs)
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Delay (s)')
    axs.set_ylabel('Density')
    axs.set_xlim([0 , 2])
    axs.set_title('Delay Distribution (short:solid, long:dashed)')
    axs.legend()
    
    
def run(axs , axs1 , session_data , block_data):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    
    num_session = len(block_data['NumBlocks'])
    
    
    short_delay_data = []
    long_delay_data = []
    for i in range(num_session):
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        for j in range(len(short_id)):
            if len(short_delay_data) < j+1:
                short_delay_data.append(block_data['delay'][i][short_id[j]])
            else:
                short_delay_data[j] = np.append(short_delay_data[j] , block_data['delay'][i][short_id[j]])
                
        for j in range(len(long_id)):
            if len(long_delay_data) < j+1:
                long_delay_data.append(block_data['delay'][i][long_id[j]])
            else:
                long_delay_data[j] = np.append(long_delay_data[j] , block_data['delay'][i][long_id[j]])
    
    plot_curves_all(axs, subject ,short_delay_data, long_delay_data)
    plot_dist_all(axs1, subject ,short_delay_data, long_delay_data)
    
    
def run_indvi(axs , axs1 , session_data , block_data):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    
    num_session = len(block_data['NumBlocks'])
    #print(len(block_data['short_id']))
    max_num = 6
    num_plotting =  min(max_num , num_session)
    initial_start = max(num_session - 3*max_num , 0)
    for k in range(num_session//max_num ):
        
        start = k*num_plotting + initial_start
        end = min((k+1)*num_plotting , num_session)+ initial_start
        date = dates[start:end] 
        for i in range(start , end):
            #print(i)
            short_id = block_data['short_id'][i]
            long_id = block_data['long_id'][i]
            delay_data = block_data['delay'][i]
            
            short_delay_data = []
            long_delay_data = []
            for j in range(len(short_id)):
                short_delay_data.append(block_data['delay'][i][short_id[j]])
                    
            for j in range(len(long_id)):
                long_delay_data.append(block_data['delay'][i][long_id[j]])
            
            
            plot_curves_indvi(axs[k], date ,delay_data, short_id , long_id , (i-start)-(i-start)//num_plotting , num_plotting)
            plot_dist_indvi(axs1[k], date , short_delay_data , long_delay_data , (i-start)-(i-start)//num_plotting , num_plotting)
        
def run_QC(axs , session_data , block_data):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    
    num_session = len(block_data['NumBlocks'])
    max_num = 6
    num_plotting =  min(max_num , num_session)
    initial_start = max(num_session - 3*max_num , 0)
    for k in range(num_session//max_num):
        
        start = k*num_plotting + initial_start
        end = min((k+1)*num_plotting , num_session)+ initial_start
        date = dates[start:end] 
        for i in range(start , end):
            short_id = block_data['short_id'][i]
            long_id = block_data['long_id'][i]
            QC = block_data['QC_index'][i]
            
            
            plot_QC_indvi(axs[k], date ,QC, short_id , long_id , (i-start)-(i-start)//num_plotting , num_plotting)
            
    