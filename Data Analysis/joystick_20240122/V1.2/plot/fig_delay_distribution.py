
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import date
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


def plot_delay_distribution(
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

    today = date.today()
    today_formatted = str(today)[2:]
    year = today_formatted[0:2]
    month = today_formatted[3:5]
    day = today_formatted[6:]
    today_string = year + month + day
    numSessions = session_data['total_sessions']

    encoder_time_max = 100
    ms_per_s = 1000
    savefiles = 1
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_id = np.arange(len(outcomes)) + 1
    isSelfTimedMode  = process_matrix (session_data['isSelfTimedMode'])

    mean_PreVis2Delay_session_short = []
    std_PreVis2Delay_session_short = []
    mean_PreVis2Delay_session_long = []
    std_PreVis2Delay_session_long = []

    mean_PrePress2Delay_session_short = []
    std_PrePress2Delay_session_short = []
    mean_PrePress2Delay_session_long = []
    std_PrePress2Delay_session_long = []

    offset = 0.1

    chemo_labels = session_data['chemo']
    for i in range(0 , len(chemo_labels)):
            if chemo_labels[i] == 1:
                dates[i] = dates[i] + '(chemo)'

    dates = deduplicate_chemo(dates)
    numeric_dates = np.arange(len(dates))

    if any(0 in row for row in isSelfTimedMode):
        print('Visually Guided')
        fig1,axs1 = plt.subplots()
        fig1.suptitle(subject)
        
        
        for i in range(0 , len(session_id)):
            TrialOutcomes = session_data['outcomes'][i]
            # We have Raw data and extract every thing from it (Times)
            raw_data = session_data['raw'][i]
            session_date = dates[i][2:]
            trial_types = raw_data['TrialTypes']
            print('analysis for delay of session:' + session_date)
            PreVis2Delay_session_short = []
            PreVis2Delay_session_long = []
            for trial in range(0,len(TrialOutcomes)):
                
                if np.isnan(isSelfTimedMode[i][trial]):
                    continue
                
                if TrialOutcomes[trial] == 'Reward' or TrialOutcomes[trial] == 'EarlyPress2' or TrialOutcomes[trial] == 'DidNotPress2':
                    encoder_data = raw_data['EncoderData'][trial]              
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)               
                    
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    PreVis2Delay_1 = trial_states['PreVis2Delay'][0]
                    PreVis2Delay_2 = trial_states['PreVis2Delay'][1]
                    PreVis2Delay = PreVis2Delay_2 - PreVis2Delay_1
            
                    if trial_types[trial] == 1:
                        PreVis2Delay_session_short.append(PreVis2Delay)
                    else:
                        PreVis2Delay_session_long.append(PreVis2Delay)
            # creating error bars
            std_error_1 = np.std(PreVis2Delay_session_short, ddof=1) / np.sqrt(len(PreVis2Delay_session_short))
            std_PreVis2Delay_session_short.append(std_error_1)
            mean_PreVis2Delay_session_short.append(np.mean(PreVis2Delay_session_short))
            
            std_error_2 = np.std(PreVis2Delay_session_long, ddof=1) / np.sqrt(len(PreVis2Delay_session_long))
            std_PreVis2Delay_session_long.append(std_error_2)
            mean_PreVis2Delay_session_long.append(np.mean(PreVis2Delay_session_long))
        
        axs1.errorbar(numeric_dates, mean_PreVis2Delay_session_short, yerr=std_PreVis2Delay_session_short, fmt='o', capsize=4, color = 'blue', label = 'short')
        axs1.errorbar(numeric_dates + offset, mean_PreVis2Delay_session_long, yerr=std_PreVis2Delay_session_long, fmt='o', capsize=4, color = 'green', label = 'long')
        axs1.legend()
        axs1.set_title('Previs2Delay')
        axs1.set_ylabel('Mean +/- SEM of preVis2Delay (s)')
        axs1.set_xlabel('Sessions')
        axs1.spines['top'].set_visible(False)
        axs1.spines['right'].set_visible(False)
        axs1.set_xticks(numeric_dates)
        axs1.set_xticklabels(dates, rotation=90, ha = 'center')
        fig1.tight_layout()
        output_figs_dir = output_dir_onedrive + subject + '/'    
        output_imgs_dir = output_dir_local + subject + '/Delay_Distribution_imgs/'
        os.makedirs(output_figs_dir, exist_ok = True)
        os.makedirs(output_imgs_dir, exist_ok = True)
        fig1.savefig(output_figs_dir + subject + '_Delay_Distribution_analysis_oversessions.pdf', dpi=300)
        fig1.savefig(output_imgs_dir + subject + '_Delay_Distribution_analysis_imgs_oversessions.png', dpi=300)
        plt.close(fig1)
    # Analysis for the self-time mode
    else:
        print('self-time')
        fig1,axs1 = plt.subplots()
        fig1.suptitle(subject)
        
        for i in range(0 , len(session_id)):
            TrialOutcomes = session_data['outcomes'][i]
            # We have Raw data and extract every thing from it (Times)
            raw_data = session_data['raw'][i]
            session_date = dates[i][2:]
            trial_types = raw_data['TrialTypes']
            print('analysis for delay of session:' + session_date)
            PrePress2Delay_session_short = []
            PrePress2Delay_session_long = []
            
            for trial in range(0,len(TrialOutcomes)):
                
                if np.isnan(isSelfTimedMode[i][trial]):
                    continue
                
                if TrialOutcomes[trial] == 'Reward' or TrialOutcomes[trial] == 'EarlyPress2' or TrialOutcomes[trial] == 'DidNotPress2':
                    encoder_data = raw_data['EncoderData'][trial]              
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)               
                    
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    PrePress2Delay_1 = trial_states['PrePress2Delay'][0]
                    PrePress2Delay_2 = trial_states['PrePress2Delay'][1]
                    PrePress2Delay = PrePress2Delay_2 - PrePress2Delay_1
            
                    if trial_types[trial] == 1:
                        PrePress2Delay_session_short.append(PrePress2Delay)
                    else:
                        PrePress2Delay_session_long.append(PrePress2Delay)
            # creating error bars
            std_error_1 = np.std(PrePress2Delay_session_short, ddof=1) / np.sqrt(len(PrePress2Delay_session_short))
            std_PrePress2Delay_session_short.append(std_error_1)
            mean_PrePress2Delay_session_short.append(np.mean(PrePress2Delay_session_short))
            
            std_error_2 = np.std(PrePress2Delay_session_long, ddof=1) / np.sqrt(len(PrePress2Delay_session_long))
            std_PrePress2Delay_session_long.append(std_error_2)
            mean_PrePress2Delay_session_long.append(np.mean(PrePress2Delay_session_long))
        
        axs1.errorbar(numeric_dates, mean_PrePress2Delay_session_short, yerr=std_PrePress2Delay_session_short, fmt='o', capsize=4, color = 'blue', label = 'short')
        axs1.errorbar(numeric_dates + offset, mean_PrePress2Delay_session_long, yerr=std_PrePress2Delay_session_long, fmt='o', capsize=4, color = 'green', label = 'long')
        axs1.legend()
        axs1.set_title('PrePress2Delay')
        axs1.set_ylabel('Mean +/- SEM of PrePress2Delay (s)')
        axs1.set_xlabel('Sessions')
        axs1.spines['top'].set_visible(False)
        axs1.spines['right'].set_visible(False)
        axs1.set_xticks(numeric_dates)
        axs1.set_xticklabels(dates, rotation=90, ha = 'center')
        fig1.tight_layout()
        output_figs_dir = output_dir_onedrive + subject + '/'    
        output_imgs_dir = output_dir_local + subject + '/Delay_Distribution_imgs/'
        os.makedirs(output_figs_dir, exist_ok = True)
        os.makedirs(output_imgs_dir, exist_ok = True)
        fig1.savefig(output_figs_dir + subject + '_Delay_Distribution_analysis_oversessions.pdf', dpi=300)
        fig1.savefig(output_imgs_dir + subject + '_Delay_Distribution_analysis_imgs_oversessions.png', dpi=300)
        plt.close(fig1)