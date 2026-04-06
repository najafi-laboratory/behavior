# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 08:07:54 2024

@author: saminnaji3
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import random
import re

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

def deduplicate_chemo(strings):
    result = []
    for string in strings:
        # Find all occurrences of (chemo)
        chemo_occurrences = re.findall(r'\(chemo\)', string)
        # If more than one (chemo) found, replace all but the first with empty string
        if len(chemo_occurrences) > 1:
            # Keep only one (chemo)
            string = re.sub(r'\(chemo\)', '', string)
            string = string + '(chemo)'
        result.append(string)
    return result

def count_label(session_data, probe, session_label, states, norm=True):
    num_session = len(session_label)
    counts = np.zeros((4 , num_session, len(states)))
    numtrials = np.zeros((4 , num_session))
    for i in range(num_session):
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        for j in range(len(session_label[i])):
            if norm:
                if session_label[i][j] in states:
                    k = states.index(session_label[i][j])
                    if probe[i][j] == 0:
                        if trial_types[j] == 1:
                            counts[0 , i , k] = counts[0 , i , k] + 1
                            numtrials[0 , i] = numtrials[0 , i] + 1
                        else:
                            counts[1 , i , k] = counts[1 , i , k] + 1
                            numtrials[1 , i] = numtrials[1 , i] + 1
                    else: 
                        if trial_types[j] == 1:
                            counts[2 , i , k] = counts[2 , i , k] + 1
                            numtrials[2 , i] = numtrials[2 , i] + 1
                        else:
                            counts[3 , i , k] = counts[3 , i , k] + 1
                            numtrials[3 , i] = numtrials[3 , i] + 1
                    
        for j in range(len(states)):
            counts[0 , i , j] = counts[0 , i , j]/numtrials[0 , i]
            counts[1 , i , j] = counts[1 , i , j]/numtrials[1 , i]
            counts[2 , i , j] = counts[2 , i , j]/numtrials[2 , i]
            counts[3 , i , j] = counts[3 , i , j]/numtrials[3 , i]
    return counts


def plot_fig1_probe(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):
    
    max_sessions=100
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    probe = session_data['session_probe']
    chemo_labels = session_data['chemo']
    
    
    
    
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    dates = dates[start_idx:]
    dates = deduplicate_chemo(dates)      
    new_dates = []
    for date_item in dates:
        new_dates.append(date_item[2:])
    dates = new_dates
        
    counts = count_label(session_data, probe, outcomes, states)
    session_id = np.arange(len(outcomes)) + 1
    fig, axs = plt.subplots(1, figsize=(len(session_id)*2 + 2 , 4))
    plt.subplots_adjust(hspace=0.7)
    short_c_bottom = np.cumsum(counts[0 , : , :], axis=1)
    short_c_bottom[:,1:] = short_c_bottom[:,:-1]
    short_c_bottom[:,0] = 0
    short_o_bottom = np.cumsum(counts[1 , : , :], axis=1)
    short_o_bottom[:,1:] = short_o_bottom[:,:-1]
    short_o_bottom[:,0] = 0
    long_c_bottom = np.cumsum(counts[2 , : , :], axis=1)
    long_c_bottom[:,1:] = long_c_bottom[:,:-1]
    long_c_bottom[:,0] = 0
    long_o_bottom = np.cumsum(counts[3 , : , :], axis=1)
    long_o_bottom[:,1:] = long_o_bottom[:,:-1]
    long_o_bottom[:,0] = 0
    width = 0.2
    
    top_ticks = []
    for i in range(len(session_id)):
        top_ticks.append('Short|Long')
    
    for i in range(len(states)):
        axs.bar(
            session_id , counts[0,:,i],
            bottom=short_c_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i],
            label=states_name[i])
        # axs.bar(
        #     session_id, counts[2,:,i],
        #     bottom=long_c_bottom[:,i],
        #     edgecolor='white',
        #     width=width,
        #     color=colors[i])
        axs.bar(
            session_id + width, counts[1,:,i],
            bottom=short_o_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i])
        # axs.bar(
        #     session_id+ 2*width, counts[3,:,i],
        #     bottom=long_o_bottom[:,i],
        #     edgecolor='white',
        #     width=width,
        #     color=colors[i])
            
            
        
        
        
    axs.tick_params(tick1On=False)
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Training session')
    
    axs.set_ylabel('Outcome percentages')
    
    tick_index = np.arange(len(outcomes))+1
    axs.set_xticks(tick_index + width/2)
    dates_label = dates
    for i in range(0 , len(chemo_labels)):
        if chemo_labels[i] == 1:
            dates_label[i] = dates[i] + '(chemo)'
    axs.set_xticklabels(dates_label, rotation='vertical')
    ind = 0
    for xtick in axs.get_xticklabels():
        if chemo_labels[ind] == 1:
            xtick.set_color('r')
        ind = ind + 1
    
    secax = axs.secondary_xaxis('top')
    secax.set_xticks(tick_index + width/2)
    secax.set_xticklabels(top_ticks)
    secax.tick_params(labelsize = 5)
    
    
    axs.set_title(subject)
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig.suptitle('Reward percentage for completed trials across sessions')
    fig.tight_layout()
    print('Completed fig1 outcome percentages for ' + subject)
    print()
    
    # Saving the reference of the standard output
    original_stdout = sys.stdout
    today = date.today()
    today_formatted = str(today)[2:]
    year = today_formatted[0:2]
    month = today_formatted[3:5]
    day = today_formatted[6:]
    today_string = year + month + day
    output_dir = output_dir_onedrive
    output_logs_dir = output_dir +'logs/'
    output_logs_fname = output_logs_dir + subject + 'outcome_log_' + today_string + '.txt'
    os.makedirs(output_logs_dir, exist_ok = True)
    
    
    with open(output_logs_fname, 'w') as f:
        sys.stdout = f

           
    # Reset the standard output
    sys.stdout = original_stdout 

    
    output_figs_dir = output_dir_onedrive + subject + '/'    
    output_imgs_dir = output_dir_local + subject + '/outcome_imgs/'    
    os.makedirs(output_figs_dir, exist_ok = True)
    os.makedirs(output_imgs_dir, exist_ok = True)
    fig.savefig(output_figs_dir + subject + '_Outcome_Short_Long.pdf', dpi=300)
    fig.savefig(output_imgs_dir + subject + '_Outcome_Short_Long.png', dpi=300)
    
    plt.close()
    

