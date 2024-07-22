#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import random
from matplotlib.lines import Line2D
import math

states = [
    'Reward' , 
    'DidNotPress1' , 
    'DidNotPress2' , 
    'EarlyPress' , 
    'EarlyPress1' , 
    'EarlyPress2' , 
    'Other']
states_name = [
    'Reward' , 
    'DidNotPress1' , 
    'DidNotPress2' , 
    'EarlyPress' , 
    'EarlyPress1' , 
    'EarlyPress2' , 
    'VisInterrupt']
colors = [
    'limegreen',
    'coral',
    'lightcoral',
    'dodgerblue',
    'orange',
    'deeppink',
    'violet',
    'mediumorchid',
    'purple',
    'deeppink',
    'grey']

def plot_fig1_4(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    
    session_id = np.arange(len(outcomes)) + 1
    session_block_result = []
    session_block_state = []
    session_block_start = []
    for i in range(0 , len(session_id)):
        outcome = outcomes[i]
        s = session_data['session_press_delay'][i]
        start_block = [0]
        if s[0] == 0:
            state_block = [0]
            a = 0
        else:
            state_block = [1]
            a = 1
        for i in range(0 , len(s)-1):
            if a == 0:
                if not s[i+1] == 0:
                    start_block.append(i+1)
                    state_block.append(1)
                    a = 1
            else:
                if s[i+1] == 0:
                    start_block.append(i+1)
                    state_block.append(0)
                    a = 0
        num_blocks = len(start_block)
        block_result = np.zeros((num_blocks , len(states)))
        for j in range(0 , num_blocks):
            if j == num_blocks-1:
                last = len(outcome)
            else:
                last = start_block[j+1]
            block_outcome = outcome[start_block[j]:last]
            for k in range(len(states)):
                block_result[j , k] =  block_outcome.count(states[k])/len(block_outcome)
        session_block_result.append(block_result)
        session_block_state.append(state_block)
        session_block_start.append(start_block)
    
    
    fig = plt.figure(1, figsize=(10, 5))
    fig.suptitle('Reward percentage for completed blocks across sessions')
    c = ['S' , 'L']
    custom_lines = []
    for j in range(len(states)):
        custom_lines.append(Line2D([0], [0], color=colors[j], lw=4))
    fig.legend(custom_lines,states_name,fontsize=7,loc="upper left")
    a = len(session_block_result)
    b = 1
    if len(session_block_result)>5:
        b = 2
        a = 5
    for i in range(0 , len(session_block_result)):
        axs = fig.add_subplot(b, a, i+1)  
        x = np.arange(len(session_block_result[i]))+1
        c1 = np.array([int(j) for j in session_block_state[i]])
        xlabels = []
        for j in range(len(c1)):
            xlabels.append(str(j+1)+'.'+c[c1[j]])
        bottom = np.cumsum(session_block_result[i], axis=1)
        bottom[:,1:] = bottom[:,:-1]
        bottom[:,0] = 0
        for j in range(0 , len(states)):
            plt.bar(xlabels , session_block_result[i][: , j], bottom=bottom[:,j], color = colors[j])
        axs.set_ylim(0 , 1.0)
        axs.yaxis.set_visible(False) 
        if i == 0 or i == 5:
            axs.set_ylabel('Reward percentages')
            axs.yaxis.set_visible(True)
        if chemo_labels[i] == 1:
            axs.set_title(dates[i] + '(chemo)' , fontsize = 5 , color = 'r')
        else:
            axs.set_title(dates[i] , fontsize = 5)
        plt.xticks(fontsize = 5)
    
    
    
    
    original_stdout = sys.stdout
    today = date.today()
    today_formatted = str(today)[2:]
    year = today_formatted[0:2]
    month = today_formatted[3:5]
    day = today_formatted[6:]
    today_string = year + month + day
    output_dir = 'C:\\data analysis\\behavior\\joystick\\'
    output_logs_dir = output_dir +'logs\\'
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
    fig.savefig(output_figs_dir + subject + '_Outcome__over_blocks.pdf', dpi=300)
    fig.savefig(output_imgs_dir + subject + '_Outcome__over_blocks.png', dpi=300)
   
    plt.close()

