#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from datetime import date
import random
from matplotlib.backends.backend_pdf import PdfPages # type: ignore
import re

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

def process_matrix(matrix):
    count_ones = 0
    count_zeros = 0
    
    for row in matrix:
        for element in row:
            if element == 1:
                count_ones += 1
            elif element == 0:
                count_zeros += 1
    
    if count_ones > count_zeros:
        print("selftime")
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 0:
                    matrix[i][j] = np.nan
    elif count_zeros > count_ones:
        print("VisGuided")
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 1:
                    matrix[i][j] = np.nan
    else:
        print("The number of 1s and 0s is equal")
    
    return matrix

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
    '#4B0082',
    '#2E003E',
    'purple',
    'deeppink',
    'grey']

def count_label(session_data, session_label, states, norm=True):
    num_session = len(session_label)
    counts = np.zeros((2 , num_session, len(states)))
    numtrials = np.zeros((2 , num_session))
    for i in range(num_session):
        raw_data = session_data['raw'][i]
        trial_types = raw_data['TrialTypes']
        for j in range(len(session_label[i])):
            if norm:
                k = states.index(session_label[i][j])
                if trial_types[j] == 1:
                    counts[0 , i , k] = counts[0 , i , k] + 1
                    numtrials[0 , i] = numtrials[0 , i] + 1
                else: 
                    counts[1 , i , k] = counts[1 , i , k] + 1
                    numtrials[1 , i] = numtrials[1 , i] + 1
                    
        for j in range(len(states)):
            counts[0 , i , j] = counts[0 , i , j]/numtrials[0 , i]
            counts[1 , i , j] = counts[1 , i , j]/numtrials[1 , i]
    return counts


def plot_fig1_2(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):
    
    max_sessions=100
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    
    delays = session_data['session_press_delay']
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
        
    counts = count_label(session_data, outcomes, states)
    session_id = np.arange(len(outcomes)) + 1
    fig, axs = plt.subplots(figsize=(len(session_id)*1 + 2 , 8))
    plt.subplots_adjust(hspace=0.7)
    short_bottom = np.cumsum(counts[0 , : , :], axis=1)
    short_bottom[:,1:] = short_bottom[:,:-1]
    short_bottom[:,0] = 0
    long_bottom = np.cumsum(counts[1 , : , :], axis=1)
    long_bottom[:,1:] = long_bottom[:,:-1]
    long_bottom[:,0] = 0
    width = 0.4

    top_ticks = []
    for i in range(len(session_id)):
        top_ticks.append('S|L')

    for i in range(len(states)):
        axs.bar(
            session_id, counts[0,:,i],
            bottom=short_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i],
            label=states_name[i])
        axs.bar(
            session_id + width, counts[1,:,i],
            bottom=long_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i])
            
        
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
            
    dates_label = deduplicate_chemo(dates_label)
    axs.set_xticklabels(dates_label, rotation='vertical')
    ind = 0
    for xtick in axs.get_xticklabels():
        if chemo_labels[ind] == 1:
            xtick.set_color('r')
        ind = ind + 1

    secax = axs.secondary_xaxis('top')
    secax.set_xticks(tick_index + width/2)
    secax.set_xticklabels(top_ticks)


    axs.set_title(subject)
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig.suptitle('Reward percentage for completed trials across sessions')
    fig.tight_layout()
    # plt.close(fig)
    print('Completed fig1 outcome percentages for ' + subject)
    print()


    print('########################################################')
    print('################# mega session analysis ################')
    print('########################################################')
    fig2, axs2 = plt.subplots(figsize=(10, 5))
    fig2.suptitle('Reward percentage for pooled sessions')
    width = 0.1
    isSelfTimedMode  = process_matrix (session_data['isSelfTimedMode'])
    ########################## Chemo ##########################################
    short_reward_chemo = 0
    long_reward_chemo = 0

    short_DidNotPress1_chemo = 0
    long_DidNotPress1_chemo = 0

    short_DidNotPress2_chemo = 0
    long_DidNotPress2_chemo = 0

    short_EarlyPress_chemo = 0
    long_EarlyPress_chemo = 0

    short_EarlyPress1_chemo = 0
    long_EarlyPress1_chemo = 0

    short_EarlyPress2_chemo = 0
    long_EarlyPress2_chemo = 0
    
    short_VisInterrupt1_chemo = 0
    long_VisInterrupt1_chemo = 0
    
    short_VisInterrupt2_chemo = 0
    long_VisInterrupt2_chemo = 0
    
    short_VisInterrupt3_chemo = 0
    long_VisInterrupt3_chemo = 0
    
    short_VisInterrupt4_chemo = 0
    long_VisInterrupt4_chemo = 0

    short_VisInterrupt_chemo = 0
    long_VisInterrupt_chemo = 0

    ########################### Control ####################################
    short_reward_control = 0
    long_reward_control = 0

    short_DidNotPress1_control = 0
    long_DidNotPress1_control = 0

    short_DidNotPress2_control = 0
    long_DidNotPress2_control = 0

    short_EarlyPress_control = 0
    long_EarlyPress_control = 0

    short_EarlyPress1_control = 0
    long_EarlyPress1_control = 0

    short_EarlyPress2_control = 0
    long_EarlyPress2_control = 0
    
    short_VisInterrupt1_control = 0
    long_VisInterrupt1_control = 0
    
    short_VisInterrupt2_control = 0
    long_VisInterrupt2_control = 0
    
    short_VisInterrupt3_control = 0
    long_VisInterrupt3_control = 0
    
    short_VisInterrupt4_control = 0
    long_VisInterrupt4_control = 0

    short_VisInterrupt_control = 0
    long_VisInterrupt_control = 0

    ########################## Opto ############################################
    short_reward_opto = 0
    long_reward_opto = 0

    short_DidNotPress1_opto = 0
    long_DidNotPress1_opto = 0

    short_DidNotPress2_opto = 0
    long_DidNotPress2_opto = 0

    short_EarlyPress_opto = 0
    long_EarlyPress_opto = 0

    short_EarlyPress1_opto = 0
    long_EarlyPress1_opto = 0

    short_EarlyPress2_opto = 0
    long_EarlyPress2_opto = 0
    
    short_VisInterrupt1_opto = 0
    long_VisInterrupt1_opto = 0
    
    short_VisInterrupt2_opto = 0
    long_VisInterrupt2_opto = 0
    
    short_VisInterrupt3_opto = 0
    long_VisInterrupt3_opto = 0
    
    short_VisInterrupt4_opto = 0
    long_VisInterrupt4_opto = 0

    short_VisInterrupt_opto = 0
    long_VisInterrupt_opto = 0

    ####################################################### Grand Average ########################################################
    G_c_r_s = []
    G_c_r_l = []
    G_c_dp1_s = []
    G_c_dp1_l = []
    G_c_dp2_s = []
    G_c_dp2_l = []
    G_c_ep1_s = []
    G_c_ep1_l = []
    G_c_ep2_s = []
    G_c_ep2_l = []

    G_co_r_s = []
    G_op_r_s = []
    G_co_r_l = []
    G_op_r_l = []

    G_co_dp1_s = []
    G_co_dp1_l = []
    G_op_dp1_s = []
    G_op_dp1_l = []

    G_co_dp2_s = []
    G_co_dp2_l = []
    G_op_dp2_s = []
    G_op_dp2_l = []

    G_co_ep1_s = []
    G_co_ep1_l = []
    G_op_ep1_s = []
    G_op_ep1_l = []

    G_co_ep2_s = []
    G_co_ep2_l = []
    G_op_ep2_s = []
    G_op_ep2_l = []

    for i in range(0 , len(session_id)):
        TrialOutcomes = session_data['outcomes'][i]
        # We have Raw data and extract every thing from it (Times)
        raw_data = session_data['raw'][i]
        session_date = dates[i][2:]
        trial_types = raw_data['TrialTypes']
        opto = session_data['session_opto_tag'][i]
        
        ########################################################## Grand Average #################################################
        c_r_s = 0
        c_r_l = 0
        c_dp1_s = 0
        c_dp1_l = 0
        c_dp2_s = 0
        c_dp2_l = 0
        c_ep1_s = 0
        c_ep1_l = 0
        c_ep2_s = 0
        c_ep2_l = 0
        
        co_r_s = 0
        op_r_s = 0
        co_r_l = 0
        op_r_l = 0
        
        co_dp1_s = 0
        co_dp1_l = 0
        op_dp1_s = 0
        op_dp1_l = 0

        co_dp2_s = 0
        co_dp2_l = 0
        op_dp2_s = 0
        op_dp2_l = 0

        co_ep1_s = 0
        co_ep1_l = 0
        op_ep1_s = 0
        op_ep1_l = 0

        co_ep2_s = 0
        co_ep2_l = 0
        op_ep2_s = 0
        op_ep2_l = 0
        
        count_short_con = 0
        count_short_opto = 0
        count_long_con = 0
        count_long_opto = 0
        
        if chemo_labels[i] == 1:
            for trial in range(0,len(TrialOutcomes)):
                
                if np.isnan(isSelfTimedMode[i][trial]):
                    print('nan in session, trial:',i, trial)
                    continue
                
                if TrialOutcomes[trial] == 'Reward':
                    if trial_types[trial] == 1:
                        short_reward_chemo = short_reward_chemo + 1
                        c_r_s = c_r_s + 1
                    else:
                        long_reward_chemo = long_reward_chemo + 1
                        c_r_l = c_r_l + 1
                        
                elif TrialOutcomes[trial] == 'DidNotPress1':
                    if trial_types[trial] == 1:
                        short_DidNotPress1_chemo = short_DidNotPress1_chemo + 1
                        c_dp1_s = c_dp1_s + 1
                    else:
                        long_DidNotPress1_chemo = long_DidNotPress1_chemo + 1
                        c_dp1_l = c_dp1_l + 1
                        
                elif TrialOutcomes[trial] == 'DidNotPress2':
                    if trial_types[trial] == 1:
                        short_DidNotPress2_chemo = short_DidNotPress2_chemo + 1
                        c_dp2_s = c_dp2_s + 1
                    else:
                        long_DidNotPress2_chemo = long_DidNotPress2_chemo + 1
                        c_dp2_l = c_dp2_l + 1
                        
                elif TrialOutcomes[trial] == 'EarlyPress':
                    if trial_types[trial] == 1:
                        short_EarlyPress_chemo = short_EarlyPress_chemo + 1
                    else:
                        long_EarlyPress_chemo = long_EarlyPress_chemo + 1
                        
                elif TrialOutcomes[trial] == 'EarlyPress1':
                    if trial_types[trial] == 1:
                        short_EarlyPress1_chemo = short_EarlyPress1_chemo + 1
                        c_ep1_s = c_ep1_s + 1
                    else:
                        long_EarlyPress1_chemo = long_EarlyPress1_chemo + 1
                        c_ep1_1 = c_ep1_1 + 1
                        
                elif TrialOutcomes[trial] == 'EarlyPress2':
                    if trial_types[trial] == 1:
                        short_EarlyPress2_chemo = short_EarlyPress2_chemo + 1
                        c_ep2_s = c_ep2_s + 1
                    else:
                        long_EarlyPress2_chemo = long_EarlyPress2_chemo + 1
                        c_ep2_l = c_ep2_l + 1
                        
                else:
                    if trial_types[trial] == 1:
                        short_VisInterrupt_chemo = short_VisInterrupt_chemo + 1
                    else:
                        long_VisInterrupt_chemo = long_VisInterrupt_chemo + 1            
        else:
            
            for trial in range(0,len(TrialOutcomes)):
                
                if np.isnan(isSelfTimedMode[i][trial]):
                    print('nan in CO/Opto session, trial:',i, trial)
                    continue
                
                
                if trial_types[trial] == 1 and opto[trial] == 0:
                    count_short_con = count_short_con + 1
                elif trial_types[trial] == 1 and opto[trial] == 1:
                    count_short_opto = count_short_opto + 1
                elif trial_types[trial] == 2 and opto[trial] == 0:
                    count_long_con = count_long_con + 1
                elif trial_types[trial] == 2 and opto[trial] == 1:
                    count_long_opto = count_long_opto + 1
                
                
                if TrialOutcomes[trial] == 'Reward':
                    if trial_types[trial] == 1 and opto[trial] == 0:
                        short_reward_control = short_reward_control + 1
                        co_r_s = co_r_s + 1
                    elif trial_types[trial] == 1 and opto[trial] == 1:
                        short_reward_opto = short_reward_opto + 1
                        op_r_s = op_r_s + 1
                    elif trial_types[trial] == 2 and opto[trial] == 0:
                        long_reward_control = long_reward_control + 1
                        co_r_l = co_r_l + 1
                    elif trial_types[trial] == 2 and opto[trial] == 1:
                        long_reward_opto = long_reward_opto + 1
                        op_r_l = op_r_l + 1
                        
                elif TrialOutcomes[trial] == 'DidNotPress1':
                    if trial_types[trial] == 1 and opto[trial] == 0:
                        short_DidNotPress1_control = short_DidNotPress1_control + 1
                        co_dp1_s = co_dp1_s + 1
                    elif trial_types[trial] == 1 and opto[trial] == 1:
                        short_DidNotPress1_opto = short_DidNotPress1_opto + 1
                        op_dp1_s = op_dp1_s + 1
                    elif trial_types[trial] == 2 and opto[trial] == 0:
                        long_DidNotPress1_control = long_DidNotPress1_control + 1
                        co_dp1_l = co_dp1_l + 1
                    elif trial_types[trial] == 2 and opto[trial] == 1:
                        long_DidNotPress1_opto = long_DidNotPress1_opto + 1
                        op_dp1_l = op_dp1_l + 1
                        
                elif TrialOutcomes[trial] == 'DidNotPress2':
                    if trial_types[trial] == 1 and opto[trial] == 0:
                        short_DidNotPress2_control = short_DidNotPress2_control + 1
                        co_dp2_s = co_dp2_s + 1
                    elif trial_types[trial] == 1 and opto[trial] == 1:
                        short_DidNotPress2_opto = short_DidNotPress2_opto + 1
                        op_dp2_s = op_dp2_s + 1
                    elif trial_types[trial] == 2 and opto[trial] == 0:
                        long_DidNotPress2_control = long_DidNotPress2_control + 1
                        co_dp2_l = co_dp2_l + 1
                    elif trial_types[trial] == 2 and opto[trial] == 1:
                        long_DidNotPress2_opto = long_DidNotPress2_opto + 1
                        op_dp2_l = op_dp2_l + 1
                        
                elif TrialOutcomes[trial] == 'EarlyPress':
                    if trial_types[trial] == 1 and opto[trial] == 0:
                        short_EarlyPress_control = short_EarlyPress_control + 1
                    elif trial_types[trial] == 1 and opto[trial] == 1:
                        short_EarlyPress_opto = short_EarlyPress_opto + 1
                    elif trial_types[trial] == 2 and opto[trial] == 0:
                        long_EarlyPress_control = long_EarlyPress_control + 1
                    elif trial_types[trial] == 2 and opto[trial] == 1:
                        long_EarlyPress_opto = long_EarlyPress_opto + 1
                        
                elif TrialOutcomes[trial] == 'EarlyPress1':
                    if trial_types[trial] == 1 and opto[trial] == 0:
                        short_EarlyPress1_control = short_EarlyPress1_control + 1
                        co_ep1_s = co_ep1_s + 1
                    elif trial_types[trial] == 1 and opto[trial] == 1:
                        short_EarlyPress1_opto = short_EarlyPress1_opto + 1
                        op_ep1_s = op_ep1_s + 1
                    elif trial_types[trial] == 2 and opto[trial] == 0:
                        long_EarlyPress1_control = long_EarlyPress1_control + 1
                        co_ep1_l = co_ep1_l + 1
                    elif trial_types[trial] == 2 and opto[trial] == 1:
                        long_EarlyPress1_opto = long_EarlyPress1_opto + 1
                        op_ep1_l = op_ep1_l + 1
                        
                elif TrialOutcomes[trial] == 'EarlyPress2':
                    if trial_types[trial] == 1 and opto[trial] == 0:
                        short_EarlyPress2_control = short_EarlyPress2_control + 1
                        co_ep2_s = co_ep2_s + 1
                    elif trial_types[trial] == 1 and opto[trial] == 1:
                        short_EarlyPress2_opto = short_EarlyPress2_opto + 1
                        op_ep2_s = op_ep2_s + 1
                    elif trial_types[trial] == 2 and opto[trial] == 0:
                        long_EarlyPress2_control = long_EarlyPress2_control + 1
                        co_ep2_l = co_ep2_l + 1
                    elif trial_types[trial] == 2 and opto[trial] == 1:
                        long_EarlyPress2_opto = long_EarlyPress2_opto + 1
                        op_ep2_l = op_ep2_l + 1
                        
                ####
                elif TrialOutcomes[trial] == 'VisStimInterruptDetect1':
                    if trial_types[trial] == 1 and opto[trial] == 0:
                        short_VisInterrupt1_control = short_VisInterrupt1_control + 1
                    elif trial_types[trial] == 1 and opto[trial] == 1:
                        short_VisInterrupt1_opto = short_VisInterrupt1_opto + 1
                    elif trial_types[trial] == 2 and opto[trial] == 0:
                        long_VisInterrupt1_control = long_VisInterrupt1_control + 1
                    elif trial_types[trial] == 2 and opto[trial] == 1:
                        long_VisInterrupt1_opto = long_VisInterrupt1_opto + 1
                elif TrialOutcomes[trial] == 'VisStimInterruptDetect2':
                    if trial_types[trial] == 1 and opto[trial] == 0:
                        short_VisInterrupt2_control = short_VisInterrupt2_control + 1
                    elif trial_types[trial] == 1 and opto[trial] == 1:
                        short_VisInterrupt2_opto = short_VisInterrupt2_opto + 1
                    elif trial_types[trial] == 2 and opto[trial] == 0:
                        long_VisInterrupt2_control = long_VisInterrupt2_control + 1
                    elif trial_types[trial] == 2 and opto[trial] == 1:
                        long_VisInterrupt2_opto = long_VisInterrupt2_opto + 1
                elif TrialOutcomes[trial] == 'VisStimInterruptGray1':
                    if trial_types[trial] == 1 and opto[trial] == 0:
                        short_VisInterrupt3_control = short_VisInterrupt3_control + 1
                    elif trial_types[trial] == 1 and opto[trial] == 1:
                        short_VisInterrupt3_opto = short_VisInterrupt3_opto + 1
                    elif trial_types[trial] == 2 and opto[trial] == 0:
                        long_VisInterrupt3_control = long_VisInterrupt3_control + 1
                    elif trial_types[trial] == 2 and opto[trial] == 1:
                        long_VisInterrupt3_opto = long_VisInterrupt3_opto + 1
                elif TrialOutcomes[trial] == 'VisStimInterruptGray2':
                    if trial_types[trial] == 1 and opto[trial] == 0:
                        short_VisInterrupt4_control = short_VisInterrupt4_control + 1
                    elif trial_types[trial] == 1 and opto[trial] == 1:
                        short_VisInterrupt4_opto = short_VisInterrupt4_opto + 1
                    elif trial_types[trial] == 2 and opto[trial] == 0:
                        long_VisInterrupt4_control = long_VisInterrupt4_control + 1
                    elif trial_types[trial] == 2 and opto[trial] == 1:
                        long_VisInterrupt4_opto = long_VisInterrupt4_opto + 1
                ####
                        
                else:
                    if trial_types[trial] == 1 and opto[trial] == 0:
                        short_VisInterrupt_control = short_VisInterrupt_control + 1
                    elif trial_types[trial] == 1 and opto[trial] == 1:
                        short_VisInterrupt_opto = short_VisInterrupt_opto + 1
                    elif trial_types[trial] == 2 and opto[trial] == 0:
                        long_VisInterrupt_control = long_VisInterrupt_control + 1
                    elif trial_types[trial] == 2 and opto[trial] == 1:
                        long_VisInterrupt_opto = long_VisInterrupt_opto + 1
            
        if trial_types.count(1) > 0:            
            if c_r_s > 0:
                G_c_r_s.append(c_r_s/trial_types.count(1))
            if c_dp1_s > 0:
                G_c_dp1_s.append(c_dp1_s/trial_types.count(1))
            if c_dp2_s > 0:
                G_c_dp2_s.append(c_dp2_s/trial_types.count(1))
            if c_ep1_s > 0:
                G_c_ep1_s.append(c_ep1_s/trial_types.count(1))
            if c_ep2_s > 0:
                G_c_ep2_s.append(c_ep2_s/trial_types.count(1))
            
        if trial_types.count(2) > 0 :
            if c_r_l > 0:
                G_c_r_l.append(c_r_l/trial_types.count(2))
            if c_dp1_l > 0:
                G_c_dp1_l.append(c_dp1_l/trial_types.count(2))
            if c_dp2_l > 0:
                G_c_dp2_l.append(c_dp2_l/trial_types.count(2))
            if c_ep1_l > 0:
                G_c_ep1_l.append(c_ep1_l/trial_types.count(2))
            if c_ep2_l > 0:
                G_c_ep2_l.append(c_ep2_l/trial_types.count(2))
                
        if count_short_con > 0 :
            if co_r_s > 0:
                G_co_r_s.append(co_r_s/count_short_con)
            
            if co_dp1_s > 0:
                G_co_dp1_s.append(co_dp1_s/count_short_con)
            
            if co_dp2_s > 0:
                G_co_dp2_s.append(co_dp2_s/count_short_con)
            
            if co_ep1_s > 0:
                G_co_ep1_s.append(co_ep1_s/count_short_con)
            
            if co_ep2_s > 0:
                G_co_ep2_s.append(co_ep2_s/count_short_con)
        
        if count_short_opto > 0 :    
            if op_r_s > 0:
                G_op_r_s.append(op_r_s/count_short_opto)
            if op_dp1_s > 0:
                G_op_dp1_s.append(op_dp1_s/count_short_opto)
            if op_dp2_s > 0:
                G_op_dp2_s.append(op_dp2_s/count_short_opto)
            if op_ep1_s > 0:
                G_op_ep1_s.append(op_ep1_s/count_short_opto)
            if op_ep2_s > 0:
                G_op_ep2_s.append(op_ep2_s/count_short_opto)
        
        if count_long_con > 0 :
            if co_r_l > 0:
                G_co_r_l.append(co_r_l/count_long_con)
            
            if co_dp1_l > 0:
                G_co_dp1_l.append(co_dp1_l/count_long_con)
            
            if co_dp2_l > 0:
                G_co_dp2_l.append(co_dp2_l/count_long_con)
            
            if co_ep1_l > 0:
                G_co_ep1_l.append(co_ep1_l/count_long_con)
            
            if co_ep2_l > 0:
                G_co_ep2_l.append(co_ep2_l/count_long_con)
        
        if count_long_opto > 0 :
            if op_r_l > 0:
                G_op_r_l.append(op_r_l/count_long_opto)
            if op_dp1_l > 0:
                G_op_dp1_l.append(op_dp1_l/count_long_opto)
            if op_dp2_l > 0:
                G_op_dp2_l.append(op_dp2_l/count_long_opto)
            if op_ep1_l > 0:
                G_op_ep1_l.append(op_ep1_l/count_long_opto)
            if op_ep2_l > 0:
                G_op_ep2_l.append(op_ep2_l/count_long_opto)

    ################# plotting ###################
    top_ticks_mega = []
    width = 0.6
    session_id_mega = np.arange(2) + 1
    short_chemo = [short_reward_chemo,short_DidNotPress1_chemo,short_DidNotPress2_chemo,short_EarlyPress_chemo,short_EarlyPress1_chemo,short_EarlyPress2_chemo,short_VisInterrupt1_chemo,short_VisInterrupt2_chemo,short_VisInterrupt3_chemo,short_VisInterrupt4_chemo,short_VisInterrupt_chemo]
    long_chemo = [long_reward_chemo,long_DidNotPress1_chemo,long_DidNotPress2_chemo,long_EarlyPress_chemo,long_EarlyPress1_chemo,long_EarlyPress2_chemo,long_VisInterrupt1_chemo,long_VisInterrupt2_chemo,long_VisInterrupt3_chemo,long_VisInterrupt4_chemo,long_VisInterrupt_chemo]
    short_control = [short_reward_control,short_DidNotPress1_control,short_DidNotPress2_control,short_EarlyPress_control,short_EarlyPress1_control,short_EarlyPress2_control,short_VisInterrupt1_control,short_VisInterrupt2_control,short_VisInterrupt3_control,short_VisInterrupt4_control,short_VisInterrupt_control] 
    long_control = [long_reward_control,long_DidNotPress1_control,long_DidNotPress2_control,long_EarlyPress_control,long_EarlyPress1_control,long_EarlyPress2_control,long_VisInterrupt1_control,long_VisInterrupt2_control,long_VisInterrupt3_control,long_VisInterrupt4_control,long_VisInterrupt_control]
    short_opto = [short_reward_opto, short_DidNotPress1_opto, short_DidNotPress2_opto, short_EarlyPress_opto, short_EarlyPress1_opto, short_EarlyPress2_opto, short_VisInterrupt1_opto,short_VisInterrupt2_opto,short_VisInterrupt3_opto,short_VisInterrupt4_opto,short_VisInterrupt_opto]
    long_opto = [long_reward_opto, long_DidNotPress1_opto, long_DidNotPress2_opto, long_EarlyPress_opto, long_EarlyPress1_opto, long_EarlyPress2_opto, long_VisInterrupt1_opto,long_VisInterrupt2_opto,long_VisInterrupt3_opto,long_VisInterrupt4_opto,long_VisInterrupt_opto]

    short = [ short_control/np.sum(short_control),short_chemo/np.sum(short_chemo), short_opto/np.sum(short_opto)]
    long = [long_control/np.sum(long_control), long_chemo/np.sum(long_chemo),long_opto/np.sum(long_opto)]

    short_bottom = np.cumsum(short, axis=1)
    short_bottom[:,1:] = short_bottom[:,:-1]
    short_bottom[:,0] = 0

    long_bottom = np.cumsum(long, axis=1)
    long_bottom[:,1:] = long_bottom[:,:-1]
    long_bottom[:,0] = 0

    for i in range(0,2):
        top_ticks_mega.append('Control|Chemo|Opto')                    

    axs2.tick_params(tick1On=False)
    axs2.spines['left'].set_visible(False)
    axs2.spines['right'].set_visible(False)
    axs2.spines['top'].set_visible(False)
    axs2.set_xlabel('Mega session')

    axs2.set_ylabel('Outcome percentages')

    tick_index = np.arange(2)+1
    axs2.set_xticks(tick_index + width/3)
    dates_label = ['Short','Long']

    secax = axs2.secondary_xaxis('top')
    secax.set_xticks(tick_index + width/3)
    secax.set_xticklabels(top_ticks_mega)

    axs2.set_title(subject)

    axs2.set_xticklabels(dates_label, rotation='vertical')
    # ind = 0
    # for xtick in axs2.get_xticklabels():
    #     if dates_label[ind] == 'Chemo':
    #         xtick.set_color('r')
    #     elif dates_label[ind] == 'Opto':
    #         xtick.set_color('deepskyblue')        
    #     ind = ind + 1
    
 
        
    for i in range(len(states)):
        axs2.bar(
            session_id_mega[0], short[0][i],
            bottom=short_bottom[0,i],
            edgecolor='white',
            width=width/3,
            color=colors[i],
            label=states_name[i])
        axs2.bar(
            session_id_mega[0]+ width/3, short[1][i],
            bottom=short_bottom[1,i],
            edgecolor='white',
            width=width/3,
            color=colors[i])
        axs2.bar(
            session_id_mega[0]+ 2*width/3, short[2][i],
            bottom=short_bottom[2,i],
            edgecolor='white',
            width=width/3,
            color=colors[i])
        axs2.bar(
            session_id_mega[1], long[0][i],
            bottom=long_bottom[0,i],
            edgecolor='white',
            width=width/3,
            color=colors[i])
        axs2.bar(
            session_id_mega[1] + width/3, long[1][i],
            bottom=long_bottom[1,i],
            edgecolor='white',
            width=width/3,
            color=colors[i])
        axs2.bar(
            session_id_mega[1] + 2*width/3, long[2][i],
            bottom=long_bottom[2,i],
            edgecolor='white',
            width=width/3,
            color=colors[i])
    axs2.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

    fig2.tight_layout()


    ################################################# Plotting Grand ################################################
    fig3, axs3 = plt.subplots(figsize=(10, 5))
    fig3.suptitle('Reward percentage (Grand Average)')
    # Base x-values for the categories
    x_rewarded = 0
    x_dnp1 = 1
    x_dnp2 = 2
    x_ep1 = 3
    x_ep2 = 4

    # Plot with shifted x-values
    offset = 0.1
    # Define custom colors
    dim_black = 'dimgray'
    dark_black = 'black'
    dim_red = 'lightcoral'
    dark_red = 'red'
    dim_deepskyblue = 'lightblue'
    dark_deepskyblue = 'deepskyblue'

    # Plot error bars with updated formatting and colors
    # Control group (G_co)
    axs3.errorbar(x_rewarded - 2.5*offset, np.nanmean(G_co_r_s, axis=0), np.nanstd(G_co_r_s, ddof=1) / np.sqrt(len(G_co_r_s)), color=dim_black, fmt='o', capsize=4, label='Control_short')
    axs3.errorbar(x_rewarded + 1.5*offset, np.nanmean(G_co_r_l, axis=0), np.nanstd(G_co_r_l, ddof=1) / np.sqrt(len(G_co_r_l)), color=dark_black, fmt='o', capsize=4, label='Control_long')

    axs3.errorbar(x_dnp1 - 2.5*offset, np.nanmean(G_co_dp1_s, axis=0), np.nanstd(G_co_dp1_s, ddof=1) / np.sqrt(len(G_co_dp1_s)), color=dim_black, fmt='o', capsize=4)
    axs3.errorbar(x_dnp1 + 1.5*offset, np.nanmean(G_co_dp1_l, axis=0), np.nanstd(G_co_dp1_l, ddof=1) / np.sqrt(len(G_co_dp1_l)), color=dark_black, fmt='o', capsize=4)

    axs3.errorbar(x_dnp2 - 2.5*offset, np.nanmean(G_co_dp2_s, axis=0), np.nanstd(G_co_dp2_s, ddof=1) / np.sqrt(len(G_co_dp2_s)), color=dim_black, fmt='o', capsize=4)
    axs3.errorbar(x_dnp2 + 1.5*offset, np.nanmean(G_co_dp2_l, axis=0), np.nanstd(G_co_dp2_l, ddof=1) / np.sqrt(len(G_co_dp2_l)), color=dark_black, fmt='o', capsize=4)

    axs3.errorbar(x_ep1 - 2.5*offset, np.nanmean(G_co_ep1_s, axis=0), np.nanstd(G_co_ep1_s, ddof=1) / np.sqrt(len(G_co_ep1_s)), color=dim_black, fmt='o', capsize=4)
    axs3.errorbar(x_ep1 + 1.5*offset, np.nanmean(G_co_ep1_l, axis=0), np.nanstd(G_co_ep1_l, ddof=1) / np.sqrt(len(G_co_ep1_l)), color=dark_black, fmt='o', capsize=4)

    axs3.errorbar(x_ep2 - 2.5*offset, np.nanmean(G_co_ep2_s, axis=0), np.nanstd(G_co_ep2_s, ddof=1) / np.sqrt(len(G_co_ep2_s)), color=dim_black, fmt='o', capsize=4)
    axs3.errorbar(x_ep2 + 1.5*offset, np.nanmean(G_co_ep2_l, axis=0), np.nanstd(G_co_ep2_l, ddof=1) / np.sqrt(len(G_co_ep2_l)), color=dark_black, fmt='o', capsize=4)

    # Chemo group (G_c_)
    axs3.errorbar(x_rewarded - 2*offset, np.nanmean(G_c_r_s, axis=0), np.nanstd(G_c_r_s, ddof=1) / np.sqrt(len(G_c_r_s)), color=dim_red, fmt='o', capsize=4, label='Chemo_short')
    axs3.errorbar(x_rewarded + 2 * offset, np.nanmean(G_c_r_l, axis=0), np.nanstd(G_c_r_l, ddof=1) / np.sqrt(len(G_c_r_l)), color=dark_red, fmt='o', capsize=4, label='Chemo_long')

    axs3.errorbar(x_dnp1 - 2*offset, np.nanmean(G_c_dp1_s, axis=0), np.nanstd(G_c_dp1_s, ddof=1) / np.sqrt(len(G_c_dp1_s)), color=dim_red, fmt='o', capsize=4)
    axs3.errorbar(x_dnp1 + 2 * offset, np.nanmean(G_c_dp1_l, axis=0), np.nanstd(G_c_dp1_l, ddof=1) / np.sqrt(len(G_c_dp1_l)), color=dark_red, fmt='o', capsize=4)

    axs3.errorbar(x_dnp2 - 2*offset, np.nanmean(G_c_dp2_s, axis=0), np.nanstd(G_c_dp2_s, ddof=1) / np.sqrt(len(G_c_dp2_s)), color=dim_red, fmt='o', capsize=4)
    axs3.errorbar(x_dnp2 + 2 * offset, np.nanmean(G_c_dp2_l, axis=0), np.nanstd(G_c_dp2_l, ddof=1) / np.sqrt(len(G_c_dp2_l)), color=dark_red, fmt='o', capsize=4)

    axs3.errorbar(x_ep1 - 2*offset, np.nanmean(G_c_ep1_s, axis=0), np.nanstd(G_c_ep1_s, ddof=1) / np.sqrt(len(G_c_ep1_s)), color=dim_red, fmt='o', capsize=4)
    axs3.errorbar(x_ep1 + 2 * offset, np.nanmean(G_c_ep1_l, axis=0), np.nanstd(G_c_ep1_l, ddof=1) / np.sqrt(len(G_c_ep1_l)), color=dark_red, fmt='o', capsize=4)

    axs3.errorbar(x_ep2 - 2*offset, np.nanmean(G_c_ep2_s, axis=0), np.nanstd(G_c_ep2_s, ddof=1) / np.sqrt(len(G_c_ep2_s)), color=dim_red, fmt='o', capsize=4)
    axs3.errorbar(x_ep2 + 2 * offset, np.nanmean(G_c_ep2_l, axis=0), np.nanstd(G_c_ep2_l, ddof=1) / np.sqrt(len(G_c_ep2_l)), color=dark_red, fmt='o', capsize=4)

    # Opto group (G_op)
    axs3.errorbar(x_rewarded -1.5 * offset, np.nanmean(G_op_r_s, axis=0), np.nanstd(G_op_r_s, ddof=1) / np.sqrt(len(G_op_r_s)), color=dim_deepskyblue, fmt='o', capsize=4, label='Opto_short')
    axs3.errorbar(x_rewarded + 2.5 * offset, np.nanmean(G_op_r_l, axis=0), np.nanstd(G_op_r_l, ddof=1) / np.sqrt(len(G_op_r_l)), color=dark_deepskyblue, fmt='o', capsize=4, label='Opto_long')

    axs3.errorbar(x_dnp1 - 1.5 * offset, np.nanmean(G_op_dp1_s, axis=0), np.nanstd(G_op_dp1_s, ddof=1) / np.sqrt(len(G_op_dp1_s)), color=dim_deepskyblue, fmt='o', capsize=4)
    axs3.errorbar(x_dnp1 + 2.5 * offset, np.nanmean(G_op_dp1_l, axis=0), np.nanstd(G_op_dp1_l, ddof=1) / np.sqrt(len(G_op_dp1_l)), color=dark_deepskyblue, fmt='o', capsize=4)

    axs3.errorbar(x_dnp2 - 1.5 * offset, np.nanmean(G_op_dp2_s, axis=0), np.nanstd(G_op_dp2_s, ddof=1) / np.sqrt(len(G_op_dp2_s)), color=dim_deepskyblue, fmt='o', capsize=4)
    axs3.errorbar(x_dnp2 + 2.5 * offset, np.nanmean(G_op_dp2_l, axis=0), np.nanstd(G_op_dp2_l, ddof=1) / np.sqrt(len(G_op_dp2_l)), color=dark_deepskyblue, fmt='o', capsize=4)

    axs3.errorbar(x_ep1 - 1.5 * offset, np.nanmean(G_op_ep1_s, axis=0), np.nanstd(G_op_ep1_s, ddof=1) / np.sqrt(len(G_op_ep1_s)), color=dim_deepskyblue, fmt='o', capsize=4)
    axs3.errorbar(x_ep1 + 2.5 * offset, np.nanmean(G_op_ep1_l, axis=0), np.nanstd(G_op_ep1_l, ddof=1) / np.sqrt(len(G_op_ep1_l)), color=dark_deepskyblue, fmt='o', capsize=4)

    axs3.errorbar(x_ep2 - 1.5 * offset, np.nanmean(G_op_ep2_s, axis=0), np.nanstd(G_op_ep2_s, ddof=1) / np.sqrt(len(G_op_ep2_s)), color=dim_deepskyblue, fmt='o', capsize=4)
    axs3.errorbar(x_ep2 + 2.5 * offset, np.nanmean(G_op_ep2_l, axis=0), np.nanstd(G_op_ep2_l, ddof=1) / np.sqrt(len(G_op_ep2_l)), color=dark_deepskyblue, fmt='o', capsize=4)

    # Set x-ticks to show the original category labels
    axs3.set_xticks([x_rewarded, x_dnp1, x_dnp2, x_ep1, x_ep2])
    axs3.set_xticklabels(['Rewarded', 'DidNotPress1', 'DidNotPress2', 'EarlyPress1', 'EarlyPress2'])

    # Change the color of x-tick labels
    # xtick_labels = axs3.get_xticklabels()
    # xtick_labels[0].set_color('limegreen')  # 'Rewarded'
    # xtick_labels[1].set_color('coral')      # 'DidNotPress1'
    # xtick_labels[2].set_color('lightcoral') # 'DidNotPress2'
    # xtick_labels[3].set_color('orange')     # 'EarlyPress1'
    # xtick_labels[4].set_color('deeppink')   # 'EarlyPress2'

    axs3.set_ylim(-0.1,1)
    axs3.spines['right'].set_visible(False)
    axs3.spines['top'].set_visible(False)
    axs3.set_title(subject)
    axs3.set_ylabel('Mean outcome percentages +/- SEM')
    axs3.legend(loc='best')
    fig3.tight_layout()

    output_figs_dir = output_dir_onedrive + subject + '/'  
    pdf_path = os.path.join(output_figs_dir, subject + '_Outcome_Short_Long.pdf')
    
    plt.rcParams['pdf.fonttype'] = 42  # Ensure text is kept as text (not outlines)
    plt.rcParams['ps.fonttype'] = 42   # For compatibility with EPS as well, if needed

    # Save both plots into a single PDF file with each on a separate page
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
        pdf.savefig(fig3)

    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)