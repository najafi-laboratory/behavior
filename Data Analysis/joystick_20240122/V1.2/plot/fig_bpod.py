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

def plot_bpod(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):

    max_sessions=10
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    start_idx = 0
    dates = session_data['dates']
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    session_id = np.arange(len(outcomes)) + 1

    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    dates = deduplicate_chemo(dates)
    session_id = np.arange(len(outcomes)) + 1
    isSelfTimedMode  = process_matrix (session_data['isSelfTimedMode'])



    if any(0 in row for row in isSelfTimedMode):
        print('Visually Guided')
        fig1,axs1 = plt.subplots(nrows=10, ncols=1, figsize= (30,20))
        fig1.suptitle('\n' + subject)

        for i in range(0 , len(session_id)):
            TrialOutcomes = session_data['outcomes'][i]
            # We have Raw data and extract every thing from it (Times)
            raw_data = session_data['raw'][i]
            session_date = dates[i][2:]
            trial_types = raw_data['TrialTypes']
            print('bpod for session:' + session_date)

            axs1[i].set_yticks([1, 2])
            axs1[i].set_yticklabels(['2', '1'])
            axs1[i].set_title('\n' + session_date)
            axs1[i].set_ylabel('Trial Type')
            axs1[i].set_xlabel('trials')
            axs1[i].spines['top'].set_visible(False)
            axs1[i].spines['right'].set_visible(False)
            
            for trial in range(0,len(TrialOutcomes)):
                
                if np.isnan(isSelfTimedMode[i][trial]):
                    continue
                
                if trial_types[trial] == 2:
                    trial_types[trial] = 1
                else:
                    trial_types[trial] = 2
                    
                if TrialOutcomes[trial] == 'Reward':
                    axs1[i].scatter(trial, trial_types[trial], color = 'green')
                    
                elif TrialOutcomes[trial] == 'DidNotPress1':
                    axs1[i].scatter(trial, trial_types[trial], color='red')
                    
                elif TrialOutcomes[trial] == 'DidNotPress2':
                    axs1[i].scatter(trial, trial_types[trial], c='none', edgecolors='red', marker='o')
                    
                elif TrialOutcomes[trial] == 'EarlyPress1':
                    axs1[i].scatter(trial, trial_types[trial], color='blue')
                elif TrialOutcomes[trial] == 'EarlyPress2':
                    axs1[i].scatter(trial, trial_types[trial], c='none', edgecolors='blue', marker='o')
                elif TrialOutcomes[trial] == 'EarlyPress':
                    axs1[i].scatter(trial, trial_types[trial], c='none', edgecolors='gray', marker='o')
                else:
                    axs1[i].scatter(trial, trial_types[trial], color = 'gray')
        
        # Create custom legend with markers
        legend_elements = [
        Line2D([0], [0], color='green', marker='o', linestyle='None', markersize=10, label='Rewarded'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, label='DidNotPress1'),
        Line2D([0], [0], color='red', marker='o',fillstyle = 'none', linestyle='None', markersize=10, label='DidNotPress2'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=10, label='EarlyPress1'),
        Line2D([0], [0], color='blue', marker='o',fillstyle = 'none', linestyle='None', markersize=10, label='EarlyPress2'),
        Line2D([0], [0], color='gray', marker='o', linestyle='None', markersize=10, label='Visinterup')]

        fig1.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5,1))
        fig1.tight_layout()
        output_figs_dir = output_dir_onedrive + subject + '/'    
        output_imgs_dir = output_dir_local + subject + '/bpod_imgs/'
        os.makedirs(output_figs_dir, exist_ok = True)
        os.makedirs(output_imgs_dir, exist_ok = True)
        fig1.savefig(output_figs_dir + subject + '_bpod.pdf', dpi=300, bbox_inches='tight')
        fig1.savefig(output_imgs_dir + subject + '_bpod.png', dpi=300)
        plt.close(fig1)
        
    else:
        print('self-time')
        fig1,axs1 = plt.subplots(nrows=10, ncols=1, figsize= (60,40))
        fig1.suptitle('\n' + subject)
        
        for i in range(0 , len(session_id)):
            TrialOutcomes = session_data['outcomes'][i]
            # We have Raw data and extract every thing from it (Times)
            raw_data = session_data['raw'][i]
            session_date = dates[i][2:]
            trial_types = raw_data['TrialTypes']
            print('bpod for session:' + session_date)

            axs1[i].set_yticks([1, 2])
            axs1[i].set_yticklabels(['2', '1'])
            if 'chemo' in session_date:
                axs1[i].set_title('\n' + session_date, color = 'red')
            else:
                axs1[i].set_title('\n' + session_date, color = 'k')
                
            axs1[i].set_ylabel('Trial Type')
            axs1[i].set_xlabel('trials')
            axs1[i].spines['top'].set_visible(False)
            axs1[i].spines['right'].set_visible(False)
            
            for trial in range(0,len(TrialOutcomes)):
                
                if np.isnan(isSelfTimedMode[i][trial]):
                    continue
                
                if trial_types[trial] == 2:
                    trial_types[trial] = 1
                else:
                    trial_types[trial] = 2
                    
                if TrialOutcomes[trial] == 'Reward':
                    axs1[i].scatter(trial, trial_types[trial], color = 'green')
                    
                elif TrialOutcomes[trial] == 'DidNotPress1':
                    axs1[i].scatter(trial, trial_types[trial], color='red')
                    
                elif TrialOutcomes[trial] == 'DidNotPress2':
                    axs1[i].scatter(trial, trial_types[trial], c='none', edgecolors='red', marker='o')
                    
                elif TrialOutcomes[trial] == 'EarlyPress1':
                    axs1[i].scatter(trial, trial_types[trial], color='blue')
                elif TrialOutcomes[trial] == 'EarlyPress2':
                    axs1[i].scatter(trial, trial_types[trial], c='none', edgecolors='blue', marker='o')
                elif TrialOutcomes[trial] == 'EarlyPress':
                    axs1[i].scatter(trial, trial_types[trial], c='none', edgecolors='gray', marker='o')
                else:
                    axs1[i].scatter(trial, trial_types[trial], color = 'gray')
        
        # Create custom legend with markers
        legend_elements = [
        Line2D([0], [0], color='green', marker='o', linestyle='None', markersize=10, label='Rewarded'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, label='DidNotPress1'),
        Line2D([0], [0], color='red', marker='o',fillstyle = 'none', linestyle='None', markersize=10, label='DidNotPress2'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=10, label='EarlyPress1'),
        Line2D([0], [0], color='blue', marker='o',fillstyle = 'none', linestyle='None', markersize=10, label='EarlyPress2'),
        Line2D([0], [0], color='gray', marker='o', linestyle='None', markersize=10, label='Visinterup')]

        fig1.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5,1))
        fig1.tight_layout()
        output_figs_dir = output_dir_onedrive + subject + '/'    
        output_imgs_dir = output_dir_local + subject + '/bpod_imgs/'
        os.makedirs(output_figs_dir, exist_ok = True)
        os.makedirs(output_imgs_dir, exist_ok = True)
        fig1.savefig(output_figs_dir + subject + '_bpod.pdf', dpi=300, bbox_inches='tight')
        fig1.savefig(output_imgs_dir + subject + '_bpod.png', dpi=300)
        plt.close(fig1)