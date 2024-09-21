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


    if len(session_id) == 1:
        if any(0 in row for row in isSelfTimedMode):
            print('Visually Guided')
            fig1,axs1 = plt.subplots(figsize= (60,10))
            fig1.suptitle(3*'\n' + subject +'\n')

            for i in range(0 , len(session_id)):
                TrialOutcomes = session_data['outcomes'][i]
                # We have Raw data and extract every thing from it (Times)
                raw_data = session_data['raw'][i]
                session_date = dates[i][2:]
                trial_types = raw_data['TrialTypes']
                print('delay for session:' + session_date)

                if 'chemo' in session_date:
                    axs1.set_title('\n' + session_date, color = 'red')
                else:
                    axs1.set_title('\n' + session_date, color = 'k')
                    
                axs1.set_ylabel('delay (s)')
                axs1.set_xlabel('trials')
                axs1.spines['top'].set_visible(False)
                axs1.spines['right'].set_visible(False)
                
                for trial in range(0,len(TrialOutcomes)):
                    
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    PreVis2Delay_1 = trial_states['PreVis2Delay'][0]
                    PreVis2Delay_2 = trial_states['PreVis2Delay'][1]
                    PreVis2Delay = PreVis2Delay_2 - PreVis2Delay_1
                    
                    if np.isnan(isSelfTimedMode[i][trial]):
                        continue
                    
                        
                    if TrialOutcomes[trial] == 'Reward':
                        axs1.scatter(trial, PreVis2Delay, color = '#4efd54')
                        
                    elif TrialOutcomes[trial] == 'DidNotPress1':
                        axs1.scatter(trial, PreVis2Delay, color='red')
                        
                    elif TrialOutcomes[trial] == 'DidNotPress2':
                        axs1.scatter(trial, PreVis2Delay, c='none', edgecolors='red', marker='o')
                        
                    elif TrialOutcomes[trial] == 'EarlyPress1':
                        axs1.scatter(trial, PreVis2Delay, color='blue')
                    elif TrialOutcomes[trial] == 'EarlyPress2':
                        axs1.scatter(trial, PreVis2Delay, c='none', edgecolors='blue', marker='o')
                    elif TrialOutcomes[trial] == 'EarlyPress':
                        axs1.scatter(trial, PreVis2Delay, c='none', edgecolors='gray', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptDetect1':
                        axs1.scatter(trial, PreVis2Delay, c='none', edgecolors='#D3D3D3', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptDetect2':
                        axs1.scatter(trial, PreVis2Delay, c='none', edgecolors='#C0C0C0', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptGray1':
                        axs1.scatter(trial, PreVis2Delay, c='none', edgecolors='#808080', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptGray2':
                        axs1.scatter(trial, PreVis2Delay, c='none', edgecolors='#696969', marker='o')
                    else:
                        axs1.scatter(trial, PreVis2Delay, color = 'k')
            
            # Create custom legend with markers
            legend_elements = [
            Line2D([0], [0], color='#4efd54', marker='o', linestyle='None', markersize=10, label='Rewarded'),
            Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, label='DidNotPress1'),
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
            
        else:
            print('self-time')
            fig1,axs1 = plt.subplots(figsize= (60,10))
            fig1.suptitle(3*'\n' + subject +'\n')
            
            for i in range(0 , len(session_id)):
                TrialOutcomes = session_data['outcomes'][i]
                # We have Raw data and extract every thing from it (Times)
                raw_data = session_data['raw'][i]
                session_date = dates[i][2:]
                trial_types = raw_data['TrialTypes']
                print('delay for session:' + session_date)

                if 'chemo' in session_date:
                    axs1.set_title('\n' + session_date, color = 'red')
                else:
                    axs1.set_title('\n' + session_date, color = 'k')
                    
                axs1.set_ylabel('delay (s)')
                axs1.set_xlabel('trials')
                axs1.spines['top'].set_visible(False)
                axs1.spines['right'].set_visible(False)
                
                for trial in range(0,len(TrialOutcomes)):
                    
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    
                    if np.isnan(isSelfTimedMode[i][trial]):
                        continue
                    
                    PrePress2Delay_1 = trial_states['PrePress2Delay'][0]
                    PrePress2Delay_2 = trial_states['PrePress2Delay'][1]
                    PrePress2Delay = PrePress2Delay_2 - PrePress2Delay_1
                        
                    if TrialOutcomes[trial] == 'Reward':
                        axs1.scatter(trial, PrePress2Delay, color = '#4efd54')                    
                    elif TrialOutcomes[trial] == 'DidNotPress1':
                        axs1.scatter(trial, PrePress2Delay, color='red')                    
                    elif TrialOutcomes[trial] == 'DidNotPress2':
                        axs1.scatter(trial, PrePress2Delay, c='none', edgecolors='red', marker='o')                    
                    elif TrialOutcomes[trial] == 'EarlyPress1':
                        axs1.scatter(trial, PrePress2Delay, color='blue')
                    elif TrialOutcomes[trial] == 'EarlyPress2':
                        axs1.scatter(trial, PrePress2Delay, c='none', edgecolors='blue', marker='o')
                    elif TrialOutcomes[trial] == 'EarlyPress':
                        axs1.scatter(trial, PrePress2Delay, c='none', edgecolors='gray', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptDetect1':
                        axs1.scatter(trial,PrePress2Delay, c='none', edgecolors='#D3D3D3', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptDetect2':
                        axs1.scatter(trial, PrePress2Delay, c='none', edgecolors='#C0C0C0', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptGray1':
                        axs1.scatter(trial, PrePress2Delay, c='none', edgecolors='#808080', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptGray2':
                        axs1.scatter(trial, PrePress2Delay, c='none', edgecolors='#696969', marker='o')
                    else:
                        axs1.scatter(trial, PrePress2Delay, color = 'k')
            
            # Create custom legend with markers
            legend_elements = [
            Line2D([0], [0], color='#4efd54', marker='o', linestyle='None', markersize=10, label='Rewarded'),
            Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, label='DidNotPress1'),
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
            output_imgs_dir = output_dir_local + subject + '/delay_imgs/'
            os.makedirs(output_figs_dir, exist_ok = True)
            os.makedirs(output_imgs_dir, exist_ok = True)
            fig1.savefig(output_figs_dir + subject + '_delay.pdf', dpi=300, bbox_inches='tight')
            fig1.savefig(output_imgs_dir + subject + '_delay.png', dpi=300)
            plt.close(fig1)
        
    else:
        if any(0 in row for row in isSelfTimedMode):
            print('Visually Guided')
            fig1,axs1 = plt.subplots(nrows= len(dates), ncols=1, figsize= (60,20 + 1.5* len(dates)))
            fig1.suptitle(3*'\n' + subject +'\n')

            for i in range(0 , len(session_id)):
                TrialOutcomes = session_data['outcomes'][i]
                # We have Raw data and extract every thing from it (Times)
                raw_data = session_data['raw'][i]
                session_date = dates[i][2:]
                trial_types = raw_data['TrialTypes']
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
                    PreVis2Delay_1 = trial_states['PreVis2Delay'][0]
                    PreVis2Delay_2 = trial_states['PreVis2Delay'][1]
                    PreVis2Delay = PreVis2Delay_2 - PreVis2Delay_1
                    
                    if np.isnan(isSelfTimedMode[i][trial]):
                        continue
                    
                        
                    if TrialOutcomes[trial] == 'Reward':
                        axs1[i].scatter(trial, PreVis2Delay, color = '#4efd54')
                        
                    elif TrialOutcomes[trial] == 'DidNotPress1':
                        axs1[i].scatter(trial, PreVis2Delay, color='red')
                        
                    elif TrialOutcomes[trial] == 'DidNotPress2':
                        axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='red', marker='o')
                        
                    elif TrialOutcomes[trial] == 'EarlyPress1':
                        axs1[i].scatter(trial, PreVis2Delay, color='blue')
                    elif TrialOutcomes[trial] == 'EarlyPress2':
                        axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='blue', marker='o')
                    elif TrialOutcomes[trial] == 'EarlyPress':
                        axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='gray', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptDetect1':
                        axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='#D3D3D3', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptDetect2':
                        axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='#C0C0C0', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptGray1':
                        axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='#808080', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptGray2':
                        axs1[i].scatter(trial, PreVis2Delay, c='none', edgecolors='#696969', marker='o')
                    else:
                        axs1[i].scatter(trial, PreVis2Delay, color = 'k')
            
            # Create custom legend with markers
            legend_elements = [
            Line2D([0], [0], color='#4efd54', marker='o', linestyle='None', markersize=10, label='Rewarded'),
            Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, label='DidNotPress1'),
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
            
        else:
            print('self-time')
            fig1,axs1 = plt.subplots(nrows= len(dates), ncols=1, figsize= (60,20 + 1.5* len(dates)))
            fig1.suptitle(3*'\n' + subject +'\n')
            
            for i in range(0 , len(session_id)):
                TrialOutcomes = session_data['outcomes'][i]
                # We have Raw data and extract every thing from it (Times)
                raw_data = session_data['raw'][i]
                session_date = dates[i][2:]
                trial_types = raw_data['TrialTypes']
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
                    
                    if np.isnan(isSelfTimedMode[i][trial]):
                        continue
                    
                    PrePress2Delay_1 = trial_states['PrePress2Delay'][0]
                    PrePress2Delay_2 = trial_states['PrePress2Delay'][1]
                    PrePress2Delay = PrePress2Delay_2 - PrePress2Delay_1
                        
                    if TrialOutcomes[trial] == 'Reward':
                        axs1[i].scatter(trial, PrePress2Delay, color = '#4efd54')                    
                    elif TrialOutcomes[trial] == 'DidNotPress1':
                        axs1[i].scatter(trial, PrePress2Delay, color='red')                    
                    elif TrialOutcomes[trial] == 'DidNotPress2':
                        axs1[i].scatter(trial, PrePress2Delay, c='none', edgecolors='red', marker='o')                    
                    elif TrialOutcomes[trial] == 'EarlyPress1':
                        axs1[i].scatter(trial, PrePress2Delay, color='blue')
                    elif TrialOutcomes[trial] == 'EarlyPress2':
                        axs1[i].scatter(trial, PrePress2Delay, c='none', edgecolors='blue', marker='o')
                    elif TrialOutcomes[trial] == 'EarlyPress':
                        axs1[i].scatter(trial, PrePress2Delay, c='none', edgecolors='gray', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptDetect1':
                        axs1[i].scatter(trial,PrePress2Delay, c='none', edgecolors='#D3D3D3', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptDetect2':
                        axs1[i].scatter(trial, PrePress2Delay, c='none', edgecolors='#C0C0C0', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptGray1':
                        axs1[i].scatter(trial, PrePress2Delay, c='none', edgecolors='#808080', marker='o')
                    elif TrialOutcomes[trial] == 'VisStimInterruptGray2':
                        axs1[i].scatter(trial, PrePress2Delay, c='none', edgecolors='#696969', marker='o')
                    else:
                        axs1[i].scatter(trial, PrePress2Delay, color = 'k')
            
            # Create custom legend with markers
            legend_elements = [
            Line2D([0], [0], color='#4efd54', marker='o', linestyle='None', markersize=10, label='Rewarded'),
            Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, label='DidNotPress1'),
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
            output_imgs_dir = output_dir_local + subject + '/delay_imgs/'
            os.makedirs(output_figs_dir, exist_ok = True)
            os.makedirs(output_imgs_dir, exist_ok = True)
            fig1.savefig(output_figs_dir + subject + '_delay.pdf', dpi=300, bbox_inches='tight')
            # fig1.savefig(output_imgs_dir + subject + '_delay.png', dpi=300)
            plt.close(fig1)