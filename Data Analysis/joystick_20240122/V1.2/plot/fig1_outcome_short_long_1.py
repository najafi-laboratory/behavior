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

def count_label(session_delay, session_label, states, norm=True):
    num_session = len(session_label)
    counts = np.zeros((2 , num_session, len(states)))
    numtrials = np.zeros((2 , num_session))
    for i in range(num_session):
        for j in range(len(session_label[i])):
            if norm:
                k = states.index(session_label[i][j])
                if session_delay[i][j] == 0:
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
    fig, axs = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(hspace=0.7)
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
    new_dates = []
    for date_item in dates:
        new_dates.append(date_item[2:])
    dates = new_dates
        
    counts = count_label(delays, outcomes, states)
    session_id = np.arange(len(outcomes)) + 1
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
        
        if chemo_labels[i] == 1:
            for trial in range(0,len(TrialOutcomes)):
                
                if np.isnan(isSelfTimedMode[i][trial]):
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
                        
            G_c_r_s.append(c_r_s/trial_types.count(1))
            G_c_r_l.append(c_r_l/trial_types.count(2))
            G_c_dp1_s.append(c_dp1_s/trial_types.count(1))
            G_c_dp1_l.append(c_dp1_l/trial_types.count(2))
            G_c_dp2_s.append(c_dp2_s/trial_types.count(1))
            G_c_dp2_l.append(c_dp2_l/trial_types.count(2))
            G_c_ep1_s.append(c_ep1_s/trial_types.count(1))
            G_c_ep1_l.append(c_ep1_l/trial_types.count(2))
            G_c_ep2_s.append(c_ep2_s/trial_types.count(1))
            G_c_ep2_l.append(c_ep2_l/trial_types.count(2))
            
        else:
            for trial in range(0,len(TrialOutcomes)):
                
                if np.isnan(isSelfTimedMode[i][trial]):
                    continue
                
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
                        
                else:
                    if trial_types[trial] == 1 and opto[trial] == 0:
                        short_VisInterrupt_control = short_VisInterrupt_control + 1
                    elif trial_types[trial] == 1 and opto[trial] == 1:
                        short_VisInterrupt_opto = short_VisInterrupt_opto + 1
                    elif trial_types[trial] == 2 and opto[trial] == 0:
                        long_VisInterrupt_control = long_VisInterrupt_control + 1
                    elif trial_types[trial] == 2 and opto[trial] == 1:
                        long_VisInterrupt_opto = long_VisInterrupt_opto + 1
            
            if trial_types.count(1) > 0 :
                G_co_r_s.append(co_r_s/trial_types.count(1))
                G_op_r_s.append(op_r_s/trial_types.count(1))
                G_co_dp1_s.append(co_dp1_s/trial_types.count(1))
                G_op_dp1_s.append(op_dp1_s/trial_types.count(1))
                G_co_dp2_s.append(co_dp2_s/trial_types.count(1))
                G_op_dp2_s.append(op_dp2_s/trial_types.count(1))
                G_co_ep1_s.append(co_ep1_s/trial_types.count(1))
                G_op_ep1_s.append(op_ep1_s/trial_types.count(1))
                G_co_ep2_s.append(co_ep2_s/trial_types.count(1))
                G_op_ep2_s.append(op_ep2_s/trial_types.count(1))
            
            if trial_types.count(2) > 0 :
                G_co_r_l.append(co_r_l/trial_types.count(2))
                G_op_r_l.append(op_r_l/trial_types.count(2))
                G_co_dp1_l.append(co_dp1_l/trial_types.count(2))
                G_op_dp1_l.append(op_dp1_l/trial_types.count(2))
                G_co_dp2_l.append(co_dp2_l/trial_types.count(2))
                G_op_dp2_l.append(op_dp2_l/trial_types.count(2))
                G_co_ep1_l.append(co_ep1_l/trial_types.count(2))
                G_op_ep1_l.append(op_ep1_l/trial_types.count(2))
                G_co_ep2_l.append(co_ep2_l/trial_types.count(2))
                G_op_ep2_l.append(op_ep2_l/trial_types.count(2))
            
    ################# plotting ###################
    top_ticks_mega = []
    session_id_mega = np.arange(3) + 1
    short_chemo = [short_reward_chemo,short_DidNotPress1_chemo,short_DidNotPress2_chemo,short_EarlyPress_chemo,short_EarlyPress1_chemo,short_EarlyPress2_chemo,short_VisInterrupt_chemo]
    long_chemo = [long_reward_chemo,long_DidNotPress1_chemo,long_DidNotPress2_chemo,long_EarlyPress_chemo,long_EarlyPress1_chemo,long_EarlyPress2_chemo,long_VisInterrupt_chemo]
    short_control = [short_reward_control,short_DidNotPress1_control,short_DidNotPress2_control,short_EarlyPress_control,short_EarlyPress1_control,short_EarlyPress2_control,short_VisInterrupt_control] 
    long_control = [long_reward_control,long_DidNotPress1_control,long_DidNotPress2_control,long_EarlyPress_control,long_EarlyPress1_control,long_EarlyPress2_control,long_VisInterrupt_control]
    short_opto = [short_reward_opto, short_DidNotPress1_opto, short_DidNotPress2_opto, short_EarlyPress_opto, short_EarlyPress1_opto, short_EarlyPress2_opto, short_VisInterrupt_opto]
    long_opto = [long_reward_opto, long_DidNotPress1_opto, long_DidNotPress2_opto, long_EarlyPress_opto, long_EarlyPress1_opto, long_EarlyPress2_opto, long_VisInterrupt_opto]

    short = [short_chemo/np.sum(short_chemo), short_control/np.sum(short_control), short_opto/np.sum(short_opto)]
    long = [long_chemo/np.sum(long_chemo),long_control/np.sum(long_control), long_opto/np.sum(long_opto)]

    short_bottom = np.cumsum(short, axis=1)
    short_bottom[:,1:] = short_bottom[:,:-1]
    short_bottom[:,0] = 0

    long_bottom = np.cumsum(long, axis=1)
    long_bottom[:,1:] = long_bottom[:,:-1]
    long_bottom[:,0] = 0

    for i in range(0,3):
        top_ticks_mega.append('S|L')                    

    axs2.tick_params(tick1On=False)
    axs2.spines['left'].set_visible(False)
    axs2.spines['right'].set_visible(False)
    axs2.spines['top'].set_visible(False)
    axs2.set_xlabel('Mega session')

    axs2.set_ylabel('Outcome percentages')

    tick_index = np.arange(3)+1
    axs2.set_xticks(tick_index + width/2)
    dates_label = ['Chemo','Control','Opto']

    secax = axs2.secondary_xaxis('top')
    secax.set_xticks(tick_index + width/2)
    secax.set_xticklabels(top_ticks_mega)

    axs2.set_title(subject)

    axs2.set_xticklabels(dates_label, rotation='vertical')
    ind = 0
    for xtick in axs2.get_xticklabels():
        if dates_label[ind] == 'Chemo':
            xtick.set_color('r')
        elif dates_label[ind] == 'Opto':
            xtick.set_color('deepskyblue')        
        ind = ind + 1
        
    for i in range(len(states)):
        axs2.bar(
            session_id_mega, [short[0][i],short[1][i],short[2][i]],
            bottom=short_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i],
            label=states_name[i])
        axs2.bar(
            session_id_mega + width, [long[0][i],long[1][i], long[2][i]],
            bottom=long_bottom[:,i],
            edgecolor='white',
            width=width,
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
    axs3.errorbar(x_rewarded - 2*offset, np.mean(G_c_r_s, axis=0), np.std(G_c_r_s, ddof=1) / np.sqrt(len(G_c_r_s)), color='r', fmt='o', capsize=4, label='Chemo_short')
    axs3.errorbar(x_rewarded - offset, np.mean(G_c_r_l, axis=0), np.std(G_c_r_l, ddof=1) / np.sqrt(len(G_c_r_l)), color='r', fmt='_', capsize=4, label='Chemo_long')
    axs3.errorbar(x_rewarded, np.mean(G_co_r_s, axis=0), np.std(G_co_r_s, ddof=1) / np.sqrt(len(G_co_r_s)), color='k', fmt='o', capsize=4, label='Control_short')
    axs3.errorbar(x_rewarded + offset, np.mean(G_co_r_l, axis=0), np.std(G_co_r_l, ddof=1) / np.sqrt(len(G_co_r_l)), color='k', fmt='_', capsize=4, label='Control_long')
    axs3.errorbar(x_rewarded + 2*offset, np.mean(G_op_r_s, axis=0), np.std(G_op_r_s, ddof=1) / np.sqrt(len(G_op_r_s)), color='deepskyblue', fmt='o', capsize=4, label='Opto_short')
    axs3.errorbar(x_rewarded + 3*offset, np.mean(G_op_r_l, axis=0), np.std(G_op_r_l, ddof=1) / np.sqrt(len(G_op_r_l)), color='deepskyblue', fmt='_', capsize=4, label='Opto_long')

    axs3.errorbar(x_dnp1 - 2*offset, np.mean(G_c_dp1_s, axis=0), np.std(G_c_dp1_s, ddof=1) / np.sqrt(len(G_c_dp1_s)), color='r', fmt='o', capsize=4)
    axs3.errorbar(x_dnp1 - offset, np.mean(G_c_dp1_l, axis=0), np.std(G_c_dp1_l, ddof=1) / np.sqrt(len(G_c_dp1_l)), color='r', fmt='_', capsize=4)
    axs3.errorbar(x_dnp1, np.mean(G_co_dp1_s, axis=0), np.std(G_co_dp1_s, ddof=1) / np.sqrt(len(G_co_dp1_s)), color='k', fmt='o', capsize=4)
    axs3.errorbar(x_dnp1 + offset, np.mean(G_co_dp1_l, axis=0), np.std(G_co_dp1_l, ddof=1) / np.sqrt(len(G_co_dp1_l)), color='k', fmt='_', capsize=4)
    axs3.errorbar(x_dnp1 + 2*offset, np.mean(G_op_dp1_s, axis=0), np.std(G_op_dp1_s, ddof=1) / np.sqrt(len(G_op_dp1_s)), color='deepskyblue', fmt='o', capsize=4)
    axs3.errorbar(x_dnp1 + 3*offset, np.mean(G_op_dp1_l, axis=0), np.std(G_op_dp1_l, ddof=1) / np.sqrt(len(G_op_dp1_l)), color='deepskyblue', fmt='_', capsize=4)

    axs3.errorbar(x_dnp2 - 2*offset, np.mean(G_c_dp2_s, axis=0), np.std(G_c_dp2_s, ddof=1) / np.sqrt(len(G_c_dp2_s)), color='r', fmt='o', capsize=4)
    axs3.errorbar(x_dnp2 - offset, np.mean(G_c_dp2_l, axis=0), np.std(G_c_dp2_l, ddof=1) / np.sqrt(len(G_c_dp2_l)), color='r', fmt='_', capsize=4)
    axs3.errorbar(x_dnp2, np.mean(G_co_dp2_s, axis=0), np.std(G_co_dp2_s, ddof=1) / np.sqrt(len(G_co_dp2_s)), color='k', fmt='o', capsize=4)
    axs3.errorbar(x_dnp2 + offset, np.mean(G_co_dp2_l, axis=0), np.std(G_co_dp2_l, ddof=1) / np.sqrt(len(G_co_dp2_l)), color='k', fmt='_', capsize=4)
    axs3.errorbar(x_dnp2 + 2*offset, np.mean(G_op_dp2_s, axis=0), np.std(G_op_dp2_s, ddof=1) / np.sqrt(len(G_op_dp2_s)), color='deepskyblue', fmt='o', capsize=4)
    axs3.errorbar(x_dnp2 + 3*offset, np.mean(G_op_dp2_l, axis=0), np.std(G_op_dp2_l, ddof=1) / np.sqrt(len(G_op_dp2_l)), color='deepskyblue', fmt='_', capsize=4)

    axs3.errorbar(x_ep1 - 2*offset, np.mean(G_c_ep1_s, axis=0), np.std(G_c_ep1_s, ddof=1) / np.sqrt(len(G_c_ep1_s)), color='r', fmt='o', capsize=4)
    axs3.errorbar(x_ep1 - offset, np.mean(G_c_ep1_l, axis=0), np.std(G_c_ep1_l, ddof=1) / np.sqrt(len(G_c_ep1_l)), color='r', fmt='_', capsize=4)
    axs3.errorbar(x_ep1, np.mean(G_co_ep1_s, axis=0), np.std(G_co_ep1_s, ddof=1) / np.sqrt(len(G_co_ep1_s)), color='k', fmt='o', capsize=4)
    axs3.errorbar(x_ep1 + offset, np.mean(G_co_ep1_l, axis=0), np.std(G_co_ep1_l, ddof=1) / np.sqrt(len(G_co_ep1_l)), color='k', fmt='_', capsize=4)
    axs3.errorbar(x_ep1 + 2*offset, np.mean(G_op_ep1_s, axis=0), np.std(G_op_ep1_s, ddof=1) / np.sqrt(len(G_op_ep1_s)), color='deepskyblue', fmt='o', capsize=4)
    axs3.errorbar(x_ep1 + 3*offset, np.mean(G_op_ep1_l, axis=0), np.std(G_op_ep1_l, ddof=1) / np.sqrt(len(G_op_ep1_l)), color='deepskyblue', fmt='_', capsize=4)

    axs3.errorbar(x_ep2 - 2*offset, np.mean(G_c_ep2_s, axis=0), np.std(G_c_ep2_s, ddof=1) / np.sqrt(len(G_c_ep2_s)), color='r', fmt='o', capsize=4)
    axs3.errorbar(x_ep2 - offset, np.mean(G_c_ep2_l, axis=0), np.std(G_c_ep2_l, ddof=1) / np.sqrt(len(G_c_ep2_l)), color='r', fmt='_', capsize=4)
    axs3.errorbar(x_ep2, np.mean(G_co_ep2_s, axis=0), np.std(G_co_ep2_s, ddof=1) / np.sqrt(len(G_co_ep2_s)), color='k', fmt='o', capsize=4)
    axs3.errorbar(x_ep2 + offset, np.mean(G_co_ep2_l, axis=0), np.std(G_co_ep2_l, ddof=1) / np.sqrt(len(G_co_ep2_l)), color='k', fmt='_', capsize=4)
    axs3.errorbar(x_ep2 + 2*offset, np.mean(G_op_ep2_s, axis=0), np.std(G_op_ep2_s, ddof=1) / np.sqrt(len(G_op_ep2_s)), color='deepskyblue', fmt='o', capsize=4)
    axs3.errorbar(x_ep2 + 3*offset, np.mean(G_op_ep2_l, axis=0), np.std(G_op_ep2_l, ddof=1) / np.sqrt(len(G_op_ep2_l)), color='deepskyblue', fmt='_', capsize=4)

    # Set x-ticks to show the original category labels
    axs3.set_xticks([x_rewarded, x_dnp1, x_dnp2, x_ep1, x_ep2])
    axs3.set_xticklabels(['Rewarded', 'DidNotPress1', 'DidNotPress2', 'EarlyPress1', 'EarlyPress2'])

    # Change the color of x-tick labels
    xtick_labels = axs3.get_xticklabels()
    xtick_labels[0].set_color('limegreen')  # 'Rewarded'
    xtick_labels[1].set_color('coral')      # 'DidNotPress1'
    xtick_labels[2].set_color('lightcoral') # 'DidNotPress2'
    xtick_labels[3].set_color('orange')     # 'EarlyPress1'
    xtick_labels[4].set_color('deeppink')   # 'EarlyPress2'

    axs3.set_ylim(-0.1,1)
    axs3.spines['right'].set_visible(False)
    axs3.spines['top'].set_visible(False)
    axs3.set_title(subject)
    axs3.set_ylabel('Mean outcome percentages +/- SEM')
    axs3.legend(loc='best')
    fig3.tight_layout()

    output_figs_dir = output_dir_onedrive + subject + '/'  
    pdf_path = os.path.join(output_figs_dir, subject + '_Outcome_Short_Long.pdf')

    # Save both plots into a single PDF file with each on a separate page
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
        pdf.savefig(fig3)

    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)