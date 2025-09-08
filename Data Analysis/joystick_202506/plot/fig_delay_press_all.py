# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:08:10 2024

@author: saminnaji3
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.lines import Line2D
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

def push_delay(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):

    max_sessions=100
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    start_idx = 0
    dates = session_data['dates']
    #opto = session_data['session_opto_tag']
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    session_id = np.arange(len(outcomes)) + 1

    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    dates = deduplicate_chemo(dates)
    session_id = np.arange(len(outcomes)) + 1
    #isSelfTimedMode  = process_matrix (session_data['isSelfTimedMode'])


    
    if len(session_id) > 0:
        fig1,axs1 = plt.subplots(nrows= len(dates), ncols=1, figsize= (60,20 + 1.5* len(dates)))
        fig1.suptitle(3*'\n' + subject +'\n')

        
        for i in range(0 , len(session_id)):
            TrialOutcomes = session_data['outcomes'][i]
            # We have Raw data and extract every thing from it (Times)
            raw_data = session_data['raw'][i]
            session_date = dates[i][2:]
            trial_types = raw_data['TrialTypes']
            block_state = np.diff(trial_types)
            block_start = np.where(block_state !=0)[0]
            delay_comp = session_data['session_comp_delay'][i]
            prepressdelay = []
            window = []
            opto = session_data['session_opto_tag'][i]
            
            print('delay for session:' + session_date)

            if 'chemo' in session_date:
                axs1[i].set_title('\n' + session_date, color = 'red')
            else:
                axs1[i].set_title('\n' + session_date, color = 'k')
                
            axs1[i].set_ylabel('delay (s)')
            axs1[i].set_xlabel('trials')
            axs1[i].spines['top'].set_visible(False)
            axs1[i].spines['right'].set_visible(False)
            
            for trial in range(0,len(TrialOutcomes)):
                
                trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                PreVis2Delay = delay_comp[trial]
                if trial_types[trial] == 1:
                    if 'Press2WindowShort_s' in raw_data['TrialSettings'][trial]['GUI'].keys():
                        window.append(raw_data['TrialSettings'][trial]['GUI']['Press2WindowShort_s'])
                    else:
                        window.append(raw_data['TrialSettings'][trial]['GUI']['Press2Window_s'])
                else:
                    if ('Press2WindowLong_s' in raw_data['TrialSettings'][trial]['GUI'].keys()):
                        window.append(raw_data['TrialSettings'][trial]['GUI']['Press2WindowLong_s'])
                    else:
                        window.append(raw_data['TrialSettings'][trial]['GUI']['Press2Window_s'])
                    
                #window.append(raw_data['TrialSettings'][trial]['GUI']['Press2Window_s'])
                prepressdelay.append(raw_data['PrePress2Delay'][trial])
                
                opto_tag = opto[trial]
                if opto_tag == 1:
                    marker_shape = '^'
                else:
                    marker_shape = 'o'
        
                if TrialOutcomes[trial] == 'Reward':
                    axs1[i].scatter(trial, PreVis2Delay, color = '#4efd54', marker=marker_shape)
                    
                elif TrialOutcomes[trial] == 'DidNotPress1':
                    axs1[i].scatter(trial, PreVis2Delay, color='red', marker=marker_shape)
                    
                elif TrialOutcomes[trial] == 'DidNotPress2':
                    axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='red', marker=marker_shape)
                    
                elif TrialOutcomes[trial] == 'EarlyPress1':
                    axs1[i].scatter(trial, PreVis2Delay, color='blue', marker=marker_shape)
                elif TrialOutcomes[trial] == 'EarlyPress2':
                    axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='blue', marker=marker_shape)
                elif TrialOutcomes[trial] == 'LatePress1':
                    axs1[i].scatter(trial, PreVis2Delay, color='pink', marker=marker_shape)                    
                elif TrialOutcomes[trial] == 'LatePress2':
                    axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='pink', marker=marker_shape)     
                elif TrialOutcomes[trial] == 'EarlyPress':
                    axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='gray', marker=marker_shape)
                elif TrialOutcomes[trial] == 'VisStimInterruptDetect1':
                    axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='#D3D3D3', marker=marker_shape)
                elif TrialOutcomes[trial] == 'VisStimInterruptDetect2':
                    axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='#C0C0C0', marker=marker_shape)
                elif TrialOutcomes[trial] == 'VisStimInterruptGray1':
                    axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='#808080', marker=marker_shape)
                elif TrialOutcomes[trial] == 'VisStimInterruptGray2':
                    axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='#696969', marker=marker_shape)
                else:
                    axs1[i].scatter(trial, PreVis2Delay, color = 'k')
                    
                
            #if session_data['isSelfTimedMode'][i][0] == 1:
            if raw_data['TrialSettings'][0]['GUI']['SelfTimedMode'] == 1:
                axs1[i].plot(prepressdelay , color = 'k')
                axs1[i].plot(np.array(window)+np.array(prepressdelay) , color = 'k')
            else:
                axs1[i].plot(np.array(prepressdelay)+0.15 , color = 'k')
                axs1[i].plot(np.array(window)+np.array(prepressdelay)+0.12 , color = 'k')
            if len(block_start) > 0:       
                if  block_state[block_start[0]] != 1:
                    axs1[i].axvspan(0 , block_start[0] , alpha = 0.2 , color = 'b')
                else:
                    axs1[i].axvspan(0 , block_start[0] , alpha = 0.2 , color = 'y')
                if  block_state[block_start[-1]] != 1:
                    axs1[i].axvspan(block_start[-1] , len(TrialOutcomes) , alpha = 0.2 , color = 'y')
                else:
                    axs1[i].axvspan(block_start[-1] , len(TrialOutcomes) , alpha = 0.2 , color = 'b')
               
                for block in range(len(block_start)-1):
                    if block_state[block_start[block]] ==1:
                        axs1[i].axvspan(block_start[block] , block_start[block+1] , alpha = 0.2 , color = 'b')
                    else:
                        axs1[i].axvspan(block_start[block] , block_start[block+1] , alpha = 0.2 , color = 'y')
                    if block == 0 and block_state[block_start[0]] == -1:
                        axs1[i].axvspan(0 , block_start[0] , alpha = 0.2 , color = 'b')
        # Create custom legend with markers
        legend_elements = [
        Line2D([0], [0], color='#4efd54', marker='o', linestyle='None', markersize=10, label='Rewarded'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, label='DidNotPress1'),
        Line2D([0], [0], color='pink', marker='o', linestyle='None', markersize=10, label='LatePress1'),
        Line2D([0], [0], color='pink', marker='o',fillstyle = 'none', linestyle='None', markersize=10, label='LatePress2'),
        Line2D([0], [0], color='red', marker='o',fillstyle = 'none', linestyle='None', markersize=10, label='DidNotPress2'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=10, label='EarlyPress1'),
        Line2D([0], [0], color='blue', marker='o',fillstyle = 'none', linestyle='None', markersize=10, label='EarlyPress2'),
        Line2D([0], [0], color='#D3D3D3', marker='o', linestyle='None', markersize=10, label='VisStimInterruptDetect1'),
        Line2D([0], [0], color='#C0C0C0', marker='o', linestyle='None', markersize=10, label='VisStimInterruptDetect2'),
        Line2D([0], [0], color='#808080', marker='o', linestyle='None', markersize=10, label='VisStimInterruptGray1'),
        Line2D([0], [0], color='#696969', marker='o', linestyle='None', markersize=10, label='VisStimInterruptGray2'),
        Line2D([0], [0], color='k', marker='o', linestyle='None', markersize=10, label='Visinterup')]

        fig1.legend(handles=legend_elements, loc='upper center', ncol=5, bbox_to_anchor=(0.5,1))
        fig1.tight_layout()
        output_figs_dir = output_dir_onedrive + subject + '/'    
        output_imgs_dir = output_dir_local + subject + '/bpod_imgs/'
        os.makedirs(output_figs_dir, exist_ok = True)
        os.makedirs(output_imgs_dir, exist_ok = True)
        fig1.savefig(output_figs_dir + subject + '_delay.pdf', dpi=300, bbox_inches='tight')
        fig1.savefig(output_imgs_dir + subject + '_bpod.png', dpi=300)
        plt.close(fig1)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
       