import os
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.backends.backend_pdf import PdfPages # type: ignore
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfWriter, PdfReader # type: ignore
from datetime import date
from statistics import mean 
from scipy.signal import savgol_filter # type: ignore
from scipy.signal import find_peaks # type: ignore
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

def is_chemo_session(date_str):
  return 'chemo' in date_str.lower() 


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

# For finding onsets related to position traces (DONT need for velocity onset)
def first_touch_onset(arr, value, start, end):
    indices = []

    # Ensure start and end are within the bounds of the list
    start = max(0, start)
    end = min(len(arr), end)

    for i in range(start, end):
        if arr[i-1] <= value and arr[i] >= value:
            indices.append(i)  # Collect all indices where the value is met

    # If indices are found within the current range, return the last one
    if indices:
        return indices[-1]

    # If not found, extend the search range and call the function recursively
    new_start = start - 300
    new_end = end + 300

    # Check if the extended range is within the list's bounds
    if new_start >= 0 or new_end <= len(arr):
        return first_touch_onset(arr, value, new_start, new_end)
    else:
        return None  # Return None if the value is not found within the extended range


# For finding the base point after a peak
def find_half_peak_point_after(velocity, peak_idx):
    
    half_peak_value = 2.5
    for i in range(peak_idx + 1, len(velocity)):
        if velocity[i] <= half_peak_value:
            return i
    return 0


# For finding the base point before a peak
def find_half_peak_point_before(velocity, peak_idx):

    half_peak_value = 2.5
    for i in range(peak_idx - 1, -1, -1):
        if velocity[i] <= half_peak_value:
            return i
    return 0


# We defined Velocity onset As 40% of the ideal peak amplitude
def find_onethird_peak_point_before(velocity, peak_idx):
    peak_value = velocity[peak_idx]
    onethird_peak_value = peak_value * 0.4
    for i in range(peak_idx - 1, -1, -1):
        if velocity[i] <= onethird_peak_value:
            return i
    return 0

# Use this function for finding onset in a proper intervall (This function need the 3 previous functions)
def velocity_onset(arr, start, end):
    start = max(0,start)
    end = min(len(arr), end)
    
    # Now we are finding local peaks within interval with at least 100ms away from each others and minimum 5 height
    peaks,_ = find_peaks(arr[start:end],distance=65, height=5) # can be modified for better results
    
    # we add start to fix the idx for global arr
    if len(peaks) >= 1:
        peaks = peaks + start
    
    # for situation we dont have any peaks and the peak happens after interval (so peaks happend in target point)
    if len(peaks) == 0:
        peaks = end
        onset4velocity = find_onethird_peak_point_before(arr,peaks)
        return onset4velocity
    
    # if len(peaks) == 1:
    #     onset4velocity = find_half_peak_point_before(arr,peaks[0])
    #     return onset4velocity
    
    if len(peaks) >= 1:
        peaks = np.hstack((peaks,end))
        for i in range(len(peaks) - 1 , 0 , -1):
            # if interval less than 100 datapoints the mouse pause very briefly
            x2 = find_half_peak_point_before(arr,peaks[i])
            x1 = find_half_peak_point_after(arr,peaks[i-1])
            result = all(value > -2 for value in arr[x1:x2])
            if (find_half_peak_point_before(arr,peaks[i] - find_half_peak_point_after(arr,peaks[i-1]))) <= 110 and result:
                onset4velocity = find_onethird_peak_point_before(arr,peaks[i-1])
                continue
            else:
                onset4velocity = find_onethird_peak_point_before(arr, peaks[i])
                break
        
        return onset4velocity



def plot_time_analysis(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):
    
    max_sessions=100
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    start_idx = 0
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
    
    all_onset_press1 = []
    all_stimulations1_peak = []
    all_onset_press2 = []
    all_stimulations2_peak = []
    all_onset_to_peak_1 = []
    all_onset_to_peak_2 = []
    mean_targetTime_press1 = []
    mean_targetTime_press2_reward = []
    mean_targetTime_press2_early = []
    std_targetTime_press1 = []
    std_targetTime_press2_reward = []
    std_targetTime_press2_early = []
    all_vis1_to_press1 = []
    all_vis2_to_press2 = []
    mean_amp_press1_s = []
    mean_amp_press2_reward_s = []
    mean_amp_press2_early_s = []
    std_amp_press1_s = []
    std_amp_press2_reward_s = []
    std_amp_press2_early_s = []

    mean_amp_press1_l = []
    mean_amp_press2_reward_l = []
    mean_amp_press2_early_l = []
    std_amp_press1_l = []
    std_amp_press2_reward_l = []
    std_amp_press2_early_l = []

    std_interval_early = []
    mean_interval_early = []
    std_interval_reward = []
    mean_interval_reward = []
    # Velocity
    std_target1_velocity_s = []
    mean_target1_velocity_s = []

    std_target1_velocity_l = []
    mean_target1_velocity_l = []

    std_target2_velocity_reward_s = []
    mean_target2_velocity_reward_s = []

    std_target2_velocity_reward_l = []
    mean_target2_velocity_reward_l = []

    std_target2_velocity_early_s = []
    mean_target2_velocity_early_s = []

    std_target2_velocity_early_l = []
    mean_target2_velocity_early_l = []

    std_interval_velocity_reward_s = []
    mean_interval_velocity_reward_s = []

    std_interval_velocity_reward_l = []
    mean_interval_velocity_reward_l = []

    std_interval_velocity_early_s = []
    mean_interval_velocity_early_s = []

    std_interval_velocity_early_l = []
    mean_interval_velocity_early_l = []


    mean_PreVis2Delay_session_short = []
    std_PreVis2Delay_session_short = []
    mean_PreVis2Delay_session_long = []
    std_PreVis2Delay_session_long = []

    mean_PrePress2Delay_session_short = []
    std_PrePress2Delay_session_short = []
    mean_PrePress2Delay_session_long = []
    std_PrePress2Delay_session_long = []

    mean_max_press1_vel_s = []
    std_max_press1_vel_s = []
    std_max_press1_vel_l = []
    mean_max_press1_vel_l = []
    std_max_press2_vel_early_s = []
    mean_max_press2_vel_early_s = []
    std_max_press2_vel_early_l = []
    mean_max_press2_vel_early_l = []
    std_max_press2_vel_reward_s = []
    mean_max_press2_vel_reward_s = []
    std_max_press2_vel_reward_l = []
    mean_max_press2_vel_reward_l = []

    std_velaton1_s = []
    mean_velaton1_s = []
    std_velaton1_l = []
    mean_velaton1_l = []
    std_velaton2_early_s = []
    mean_velaton2_early_s = []
    std_velaton2_early_l = []
    mean_velaton2_early_l = []
    std_velaton2_reward_s = []
    mean_velaton2_reward_s= []
    std_velaton2_reward_l = []
    mean_velaton2_reward_l = []

    # GRAND ######################################
    G_chemo_target1_velocity_s = []
    G_chemo_target2_velocity_s = []
    G_chemo_interval_velocity_s = []
    G_chemo_PreVis2Delay_session_short = []
    G_chemo_amp_press1_s = []
    G_chemo_amp_press2_s = []
    G_chemo_max_press1_vel_s = []
    G_chemo_max_press2_vel_s = []
    G_chemo_velaton1_s = []
    G_chemo_velaton2_s = []

    G_opto_target1_velocity_s = []
    G_opto_target2_velocity_s = []
    G_opto_interval_velocity_s = []
    G_opto_PreVis2Delay_session_short = []
    G_opto_amp_press1_s = []
    G_opto_amp_press2_s = []
    G_opto_max_press1_vel_s = []
    G_opto_max_press2_vel_s = []
    G_opto_velaton1_s = []
    G_opto_velaton2_s = []

    G_con_target1_velocity_s = []
    G_con_target2_velocity_s = []
    G_con_interval_velocity_s = []
    G_con_PreVis2Delay_session_short = []
    G_con_amp_press1_s = []
    G_con_amp_press2_s = []
    G_con_max_press1_vel_s = []
    G_con_max_press2_vel_s = []
    G_con_velaton1_s = []
    G_con_velaton2_s = []
    #### Long ####
    G_chemo_target1_velocity_l = []
    G_chemo_target2_velocity_l = []
    G_chemo_interval_velocity_l = []
    G_chemo_PreVis2Delay_session_long = []
    G_chemo_amp_press1_l = []
    G_chemo_amp_press2_l = []
    G_chemo_max_press1_vel_l = []
    G_chemo_max_press2_vel_l = []
    G_chemo_velaton1_l = []
    G_chemo_velaton2_l = []

    G_opto_target1_velocity_l = []
    G_opto_target2_velocity_l = []
    G_opto_interval_velocity_l = []
    G_opto_PreVis2Delay_session_long = []
    G_opto_amp_press1_l = []
    G_opto_amp_press2_l = []
    G_opto_max_press1_vel_l = []
    G_opto_max_press2_vel_l = []
    G_opto_velaton1_l = []
    G_opto_velaton2_l = []

    G_con_target1_velocity_l = []
    G_con_target2_velocity_l = []
    G_con_interval_velocity_l = []
    G_con_PreVis2Delay_session_long = []
    G_con_amp_press1_l = []
    G_con_amp_press2_l = []
    G_con_max_press1_vel_l = []
    G_con_max_press2_vel_l = []
    G_con_velaton1_l = []
    G_con_velaton2_l = []

    # POOLED ######################################
    P_chemo_target1_velocity_s = []
    P_chemo_target2_velocity_s = []
    P_chemo_interval_velocity_s = []
    P_chemo_PreVis2Delay_session_short = []
    P_chemo_amp_press1_s = []
    P_chemo_amp_press2_s = []
    P_chemo_max_press1_vel_s = []
    P_chemo_max_press2_vel_s = []
    P_chemo_velaton1_s = []
    P_chemo_velaton2_s = []

    P_opto_target1_velocity_s = []
    P_opto_target2_velocity_s = []
    P_opto_interval_velocity_s = []
    P_opto_PreVis2Delay_session_short = []
    P_opto_amp_press1_s = []
    P_opto_amp_press2_s = []
    P_opto_max_press1_vel_s = []
    P_opto_max_press2_vel_s = []
    P_opto_velaton1_s = []
    P_opto_velaton2_s = []

    P_con_target1_velocity_s = []
    P_con_target2_velocity_s = []
    P_con_interval_velocity_s = []
    P_con_PreVis2Delay_session_short = []
    P_con_amp_press1_s = []
    P_con_amp_press2_s = []
    P_con_max_press1_vel_s = []
    P_con_max_press2_vel_s = []
    P_con_velaton1_s = []
    P_con_velaton2_s = []
    #### Long ####
    P_chemo_target1_velocity_l = []
    P_chemo_target2_velocity_l = []
    P_chemo_interval_velocity_l = []
    P_chemo_PreVis2Delay_session_long = []
    P_chemo_amp_press1_l = []
    P_chemo_amp_press2_l = []
    P_chemo_max_press1_vel_l = []
    P_chemo_max_press2_vel_l = []
    P_chemo_velaton1_l = []
    P_chemo_velaton2_l = []

    P_opto_target1_velocity_l = []
    P_opto_target2_velocity_l = []
    P_opto_interval_velocity_l = []
    P_opto_PreVis2Delay_session_long = []
    P_opto_amp_press1_l = []
    P_opto_amp_press2_l = []
    P_opto_max_press1_vel_l = []
    P_opto_max_press2_vel_l = []
    P_opto_velaton1_l = []
    P_opto_velaton2_l = []

    P_con_target1_velocity_l = []
    P_con_target2_velocity_l = []
    P_con_interval_velocity_l = []
    P_con_PreVis2Delay_session_long = []
    P_con_amp_press1_l = []
    P_con_amp_press2_l = []
    P_con_max_press1_vel_l = []
    P_con_max_press2_vel_l = []
    P_con_velaton1_l = []
    P_con_velaton2_l = []


    isSelfTimedMode  = process_matrix (session_data['isSelfTimedMode'])

    chemo_labels = session_data['chemo']
    for i in range(0 , len(chemo_labels)):
            if chemo_labels[i] == 1:
                dates[i] = dates[i] + '(chemo)'

    offset = 0.15
    dates = deduplicate_chemo(dates)
    numeric_dates = np.arange(len(dates))

    # Loop through dates and assign colors based on 'chemo' presence
    red_color = (1,1,1)
    black_color = (1,0,0)  # Assuming default color was black
    date_colors = [red_color if is_chemo_session(date) else black_color for date in dates]

    if any(0 in row for row in isSelfTimedMode):
        print('Visually Guided')
        fig1,axs1 = plt.subplots(nrows=5, ncols=2, figsize=(15, 25))
        fig1.suptitle(subject + ' \n ' + 'Time analysis over sessions' + '\n')


        for i in range(0 , len(session_id)):
            TrialOutcomes = session_data['outcomes'][i]
            opto = session_data['session_opto_tag'][i]
            reward_count = TrialOutcomes.count('Reward')
            onset_press1 = []
            # rotatory1_1 = []
            onset_press2 = []
            stimulations2_peak = []
            targetTime_press1 = []
            targetTime_press2 = []
            targetTime_press2_reward = []
            targetTime_press2_early = []
            press_vis_2 = []
            amp_press1_s = []
            amp_press2_s = []
            amp_press1_l = []
            amp_press2_l = []
            amp_press2_reward_s = []
            amp_press2_early_s = []
            amp_press2_reward_l = []
            amp_press2_early_l = []
            
            # VisualStimulus11 = []
            interval_early = []
            interval_reward = []
            target1_velocity_s = []
            target2_velocity_s = []
            target2_velocity_early_s = []
            target2_velocity_reward_s = []
            interval_velocity_reward_s = []
            interval_velocity_early_s = []
            interval_velocity_s = []
            interval = []
            
            # interval_early_l = []
            # interval_reward_l = []
            target1_velocity_l = []
            target2_velocity_l = []
            target2_velocity_early_l = []
            target2_velocity_reward_l = []
            interval_velocity_reward_l = []
            interval_velocity_early_l = []
            interval_velocity_l = []
            # interval_l = []
            
            PreVis2Delay_session_short = []
            PreVis2Delay_session_long = []
            
            velaton1_s = []
            max_press1_vel_s = []
            velaton1_l = []
            max_press1_vel_l = []
            velaton2_reward_s = []
            max_press2_vel_reward_s = []
            velaton2_reward_l = []
            max_press2_vel_reward_l = []
            velaton2_early_s = []
            max_press2_vel_early_s = []
            velaton2_early_l = []
            max_press2_vel_early_l = []
            
            j = 0
            # GRAND ######################################
            chemo_target1_velocity_s = []
            chemo_target2_velocity_s = []
            chemo_interval_velocity_s = []
            chemo_PreVis2Delay_session_short = []
            chemo_amp_press1_s = []
            chemo_amp_press2_s = []
            chemo_max_press1_vel_s = []
            chemo_max_press2_vel_s = []
            chemo_velaton1_s = []
            chemo_velaton2_s = []

            opto_target1_velocity_s = []
            opto_target2_velocity_s = []
            opto_interval_velocity_s = []
            opto_PreVis2Delay_session_short = []
            opto_amp_press1_s = []
            opto_amp_press2_s = []
            opto_max_press1_vel_s = []
            opto_max_press2_vel_s = []
            opto_velaton1_s = []
            opto_velaton2_s = []

            con_target1_velocity_s = []
            con_target2_velocity_s = []
            con_interval_velocity_s = []
            con_PreVis2Delay_session_short = []
            con_amp_press1_s = []
            con_amp_press2_s = []
            con_max_press1_vel_s = []
            con_max_press2_vel_s = []
            con_velaton1_s = []
            con_velaton2_s = []
            #### Long ####
            chemo_target1_velocity_l = []
            chemo_target2_velocity_l = []
            chemo_interval_velocity_l = []
            chemo_PreVis2Delay_session_long = []
            chemo_amp_press1_l = []
            chemo_amp_press2_l = []
            chemo_max_press1_vel_l = []
            chemo_max_press2_vel_l = []
            chemo_velaton1_l = []
            chemo_velaton2_l = []

            opto_target1_velocity_l = []
            opto_target2_velocity_l = []
            opto_interval_velocity_l = []
            opto_PreVis2Delay_session_long = []
            opto_amp_press1_l = []
            opto_amp_press2_l = []
            opto_max_press1_vel_l = []
            opto_max_press2_vel_l = []
            opto_velaton1_l = []
            opto_velaton2_l = []

            con_target1_velocity_l = []
            con_target2_velocity_l = []
            con_interval_velocity_l = []
            con_PreVis2Delay_session_long = []
            con_amp_press1_l = []
            con_amp_press2_l = []
            con_max_press1_vel_l = []
            con_max_press2_vel_l = []
            con_velaton1_l = []
            con_velaton2_l = []
            
            # We have Raw data and extract every thing from it (Times)
            raw_data = session_data['raw'][i]
            session_date = dates[i][2:]
            
            pdf_paths = []
            # Creating figures for each session
            fig, axs = plt.subplots(nrows=8, ncols=2, figsize=(10, 32))   # should be changed
            fig.suptitle(subject + ' - ' + session_date + ' \n ' + 'Time analysis for Reward, Earlypress2 and Didnotpress2 trials' + '\n')
            step_size = 0.5
            
            print('timing analysis of session:' + session_date)
            # The loop for each session
            for trial in range(0,len(TrialOutcomes)):
                
                if np.isnan(isSelfTimedMode[i][trial]):
                    continue
                
                trial_types = raw_data['TrialTypes']
                        
                if TrialOutcomes[trial] == 'Reward':
                    
                    trial_type = trial_types[trial]
                    encoder_data = raw_data['EncoderData'][trial]
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                    
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
                    
                    # Check to avoid nan                 
                    # Extract the relevant values
                    values_to_check = [
                    trial_states['WaitForPress1'][0],
                    trial_states['VisDetect2'][0],
                    trial_states['VisDetect1'][0],
                    trial_states['VisualStimulus1'][0],
                    trial_states['VisualStimulus2'][0],
                    trial_states['LeverRetract2'][1],
                    trial_states['LeverRetract1'][1],
                    trial_states['LeverRetract1'][0],
                    trial_states['Reward'][0]]
                    # Check for NaN values
                    if any(np.isnan(value) for value in values_to_check):
                        print("At least one of the values is NaN in ", trial)
                        continue
                    
                    waitforpress1 = int(trial_states['WaitForPress1'][0]*1000)  # press1 interval finding
                    VisDetect2 = int(trial_states['VisDetect2'][0]*1000) 
                    VisDetect1 = int(trial_states['VisDetect1'][0]*1000) 
                    VisualStimulus1 = int(trial_states['VisualStimulus1'][0]*1000) 
                    VisualStimulus2 = int(trial_states['VisualStimulus2'][0]*1000) 
                    LeverRetract2 = int(trial_states['LeverRetract2'][1]*1000) 
                    LeverRetract1 = int(trial_states['LeverRetract1'][1]*1000) 
                    # Get the base line for the 500 datapoints before the vis1
                    base_line = encoder_positions_aligned_vis1[:VisualStimulus1]
                    base_line = base_line[~np.isnan(base_line)]
                    base_line = np.mean(base_line)
                    
                    if base_line >= 0.9:
                        print('trial starts with larger value:', trial + 1)
                        continue
                    
                    threshold_press1 = base_line + 0.1
                    rotatory1 = int(trial_states['LeverRetract1'][0]*1000) 
                    rotatory2 = int(trial_states['Reward'][0]*1000) 
                    
                    
                    amp1 = np.argmax(encoder_positions_aligned_vis1[rotatory1:VisDetect2])
                    amp1 = amp1 + rotatory1
                    onset1_position = first_touch_onset(encoder_positions_aligned_vis1, threshold_press1,0,amp1)
                    onset_press1.append(onset1_position)
                    # VisualStimulus11.append(VisualStimulus1)
                    
                    # rotatory1_1.append(rotatory1)  # The retract 1 time is the target
                    
                    # Target 1 to Onset_press 1
                    targetTime_press1.append(np.abs(rotatory1 - onset1_position)/1000) # type: ignore
                    
                    ##### press1
                    amp_press1 = (np.max(encoder_positions_aligned_vis1[rotatory1:VisDetect2]) - base_line)

                    ####### Calculations for vis 2
                    threshold_press2 = base_line + 0.25
                    amp2 = np.argmax(encoder_positions_aligned_vis1[VisualStimulus2:LeverRetract2])
                    amp2 = amp2 + VisualStimulus2
                    onset2_position = first_touch_onset(encoder_positions_aligned_vis1, threshold_press2,VisDetect2,amp2)
                    onset_press2.append(onset2_position)
                    
                    # Target 2 to Onset_press 2
                    targetTime_press2.append(np.abs(rotatory2 - onset2_position)/1000) # type: ignore
                    targetTime_press2_reward.append(np.abs(rotatory2 - onset2_position)/1000) # type: ignore
                    ##### press2
                    amp_press2 = (np.max(encoder_positions_aligned_vis1[VisualStimulus2:LeverRetract2]) - base_line)
                    amp_press2_reward = (np.max(encoder_positions_aligned_vis1[VisualStimulus2:LeverRetract2]) - base_line)
                    ##### Interval onset
                    intervall = onset2_position - onset1_position # type: ignore
                    interval_reward.append(intervall/1000)
                    interval.append(intervall/1000)
                    
                    encoder_positions_aligned_vis1 = savgol_filter(encoder_positions_aligned_vis1, window_length=40, polyorder=3)
                    velocity = np.gradient(encoder_positions_aligned_vis1, encoder_times_vis1)
                    #### Velocity onset
                    velocity = savgol_filter(velocity, window_length=40, polyorder=1)
                    on1_vel = velocity_onset(velocity,int(VisDetect1),rotatory1) # type: ignore
                    on2_vel = velocity_onset(velocity,int(VisDetect2),rotatory2) # type: ignore
                    intervall_vel = on2_vel - on1_vel # type: ignore
                    
                    PreVis2Delay_1 = trial_states['PreVis2Delay'][0]
                    PreVis2Delay_2 = trial_states['PreVis2Delay'][1]
                    PreVis2Delay = PreVis2Delay_2 - PreVis2Delay_1
                    
                    # Getting max velocity for press 1 and press 2 //// velocity at onset1 and onset2
                    if amp1 >= on1_vel: # type: ignore
                        max_press1_vel = np.max(velocity[on1_vel:amp1])
                    else:
                        max_press1_vel = np.max(velocity[amp1:on1_vel])
                        
                    if amp2 >= on2_vel:                 # type: ignore
                        max_press2_vel = np.max(velocity[on2_vel:amp2])
                    else:
                        max_press2_vel = np.max(velocity[amp2:on2_vel])
                        
                    velaton1 = velocity[on1_vel]
                    velaton2 = velocity[on2_vel]
                    
                    if trial_type == 1:
                        target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                        target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                        target2_velocity_reward_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore                    
                        interval_velocity_s.append(intervall_vel/1000)
                        interval_velocity_reward_s.append(intervall_vel/1000)
                        PreVis2Delay_session_short.append(PreVis2Delay)
                        amp_press1_s.append(amp_press1)
                        amp_press2_s.append(amp_press2)
                        amp_press2_reward_s.append(amp_press2_reward)
                        max_press1_vel_s.append(max_press1_vel)
                        max_press2_vel_reward_s.append(max_press2_vel)
                        velaton1_s.append(velaton1)
                        velaton2_reward_s.append(velaton2)
                        if session_data['chemo'][i] == 1:
                            chemo_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            chemo_target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                            chemo_interval_velocity_s.append(intervall_vel/1000)
                            chemo_PreVis2Delay_session_short.append(PreVis2Delay)
                            chemo_amp_press1_s.append(amp_press1)
                            chemo_amp_press2_s.append(amp_press2)
                            chemo_max_press1_vel_s.append(max_press1_vel)
                            chemo_max_press2_vel_s.append(max_press2_vel)
                            chemo_velaton1_s.append(velaton1)
                            chemo_velaton2_s.append(velaton2)
                            
                            # POOLED
                            P_chemo_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            P_chemo_target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                            P_chemo_interval_velocity_s.append(intervall_vel/1000)
                            P_chemo_PreVis2Delay_session_short.append(PreVis2Delay)
                            P_chemo_amp_press1_s.append(amp_press1)
                            P_chemo_amp_press2_s.append(amp_press2)
                            P_chemo_max_press1_vel_s.append(max_press1_vel)
                            P_chemo_max_press2_vel_s.append(max_press2_vel)
                            P_chemo_velaton1_s.append(velaton1)
                            P_chemo_velaton2_s.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                opto_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                opto_target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                opto_interval_velocity_s.append(intervall_vel/1000)
                                opto_PreVis2Delay_session_short.append(PreVis2Delay)
                                opto_amp_press1_s.append(amp_press1)
                                opto_amp_press2_s.append(amp_press2)
                                opto_max_press1_vel_s.append(max_press1_vel)
                                opto_max_press2_vel_s.append(max_press2_vel)
                                opto_velaton1_s.append(velaton1)
                                opto_velaton2_s.append(velaton2)
                                
                                # POOLED
                                P_opto_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_opto_target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                P_opto_interval_velocity_s.append(intervall_vel/1000)
                                P_opto_PreVis2Delay_session_short.append(PreVis2Delay)
                                P_opto_amp_press1_s.append(amp_press1)
                                P_opto_amp_press2_s.append(amp_press2)
                                P_opto_max_press1_vel_s.append(max_press1_vel)
                                P_opto_max_press2_vel_s.append(max_press2_vel)
                                P_opto_velaton1_s.append(velaton1)
                                P_opto_velaton2_s.append(velaton2)
                            else:
                                con_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                con_target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                con_interval_velocity_s.append(intervall_vel/1000)
                                con_PreVis2Delay_session_short.append(PreVis2Delay)
                                con_amp_press1_s.append(amp_press1)
                                con_amp_press2_s.append(amp_press2)
                                con_max_press1_vel_s.append(max_press1_vel)
                                con_max_press2_vel_s.append(max_press2_vel)
                                con_velaton1_s.append(velaton1)
                                con_velaton2_s.append(velaton2)
                                
                                # POOLED
                                P_con_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_con_target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                P_con_interval_velocity_s.append(intervall_vel/1000)
                                P_con_PreVis2Delay_session_short.append(PreVis2Delay)
                                P_con_amp_press1_s.append(amp_press1)
                                P_con_amp_press2_s.append(amp_press2)
                                P_con_max_press1_vel_s.append(max_press1_vel)
                                P_con_max_press2_vel_s.append(max_press2_vel)
                                P_con_velaton1_s.append(velaton1)
                                P_con_velaton2_s.append(velaton2)
                        
                    else:
                    
                        target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                        target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                        target2_velocity_reward_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                        interval_velocity_l.append(intervall_vel/1000)
                        interval_velocity_reward_l.append(intervall_vel/1000)
                        PreVis2Delay_session_long.append(PreVis2Delay)
                        amp_press1_l.append(amp_press1)
                        amp_press2_l.append(amp_press2)
                        amp_press2_reward_l.append(amp_press2_reward)
                        max_press1_vel_l.append(max_press1_vel)
                        max_press2_vel_reward_l.append(max_press2_vel)
                        velaton1_l.append(velaton1)
                        velaton2_reward_l.append(velaton1)
                        
                        if session_data['chemo'][i] == 1:
                            chemo_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            chemo_target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                            chemo_interval_velocity_l.append(intervall_vel/1000)
                            chemo_PreVis2Delay_session_long.append(PreVis2Delay)
                            chemo_amp_press1_l.append(amp_press1)
                            chemo_amp_press2_l.append(amp_press2)
                            chemo_max_press1_vel_l.append(max_press1_vel)
                            chemo_max_press2_vel_l.append(max_press2_vel)
                            chemo_velaton1_l.append(velaton1)
                            chemo_velaton2_l.append(velaton2)
                            
                            # POOLED
                            P_chemo_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            P_chemo_target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                            P_chemo_interval_velocity_l.append(intervall_vel/1000)
                            P_chemo_PreVis2Delay_session_long.append(PreVis2Delay)
                            P_chemo_amp_press1_l.append(amp_press1)
                            P_chemo_amp_press2_l.append(amp_press2)
                            P_chemo_max_press1_vel_l.append(max_press1_vel)
                            P_chemo_max_press2_vel_l.append(max_press2_vel)
                            P_chemo_velaton1_l.append(velaton1)
                            P_chemo_velaton2_l.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                opto_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                opto_target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                opto_interval_velocity_l.append(intervall_vel/1000)
                                opto_PreVis2Delay_session_long.append(PreVis2Delay)
                                opto_amp_press1_l.append(amp_press1)
                                opto_amp_press2_l.append(amp_press2)
                                opto_max_press1_vel_l.append(max_press1_vel)
                                opto_max_press2_vel_l.append(max_press2_vel)
                                opto_velaton1_l.append(velaton1)
                                opto_velaton2_l.append(velaton2)
                                
                                # POOLED
                                P_opto_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_opto_target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                P_opto_interval_velocity_l.append(intervall_vel/1000)
                                P_opto_PreVis2Delay_session_long.append(PreVis2Delay)
                                P_opto_amp_press1_l.append(amp_press1)
                                P_opto_amp_press2_l.append(amp_press2)
                                P_opto_max_press1_vel_l.append(max_press1_vel)
                                P_opto_max_press2_vel_l.append(max_press2_vel)
                                P_opto_velaton1_l.append(velaton1)
                                P_opto_velaton2_l.append(velaton2)
                            else:
                                con_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                con_target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                con_interval_velocity_l.append(intervall_vel/1000)
                                con_PreVis2Delay_session_long.append(PreVis2Delay)
                                con_amp_press1_l.append(amp_press1)
                                con_amp_press2_l.append(amp_press2)
                                con_max_press1_vel_l.append(max_press1_vel)
                                con_max_press2_vel_l.append(max_press2_vel)
                                con_velaton1_l.append(velaton1)
                                con_velaton2_l.append(velaton2)
                                
                                # POOLED
                                P_con_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_con_target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                P_con_interval_velocity_l.append(intervall_vel/1000)
                                P_con_PreVis2Delay_session_long.append(PreVis2Delay)
                                P_con_amp_press1_l.append(amp_press1)
                                P_con_amp_press2_l.append(amp_press2)
                                P_con_max_press1_vel_l.append(max_press1_vel)
                                P_con_max_press2_vel_l.append(max_press2_vel)
                                P_con_velaton1_l.append(velaton1)
                                P_con_velaton2_l.append(velaton2)
                        
                    j = j+1
                # For DidNotPress2 (WE have the press one only)
                elif TrialOutcomes[trial] == 'DidNotPress2':
                    trial_type = trial_types[trial]
                    encoder_data = raw_data['EncoderData'][trial]              
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                    ####
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
                    
                    # Check to avoid nan                 
                    # Extract the relevant values
                    values_to_check = [
                    trial_states['VisDetect2'][0],
                    trial_states['VisDetect1'][0],
                    trial_states['VisualStimulus1'][0],
                    trial_states['LeverRetract1'][0]
                    ]
                    # Check for NaN values
                    if any(np.isnan(value) for value in values_to_check):
                        print("At least one of the values is NaN in ", trial)
                        continue
                    
                    VisDetect2 = int(trial_states['VisDetect2'][0]*1000) #needed
                    VisDetect1 = int(trial_states['VisDetect1'][0]*1000) #needed
                    VisualStimulus1 = int(trial_states['VisualStimulus1'][0]*1000) #needed
                    # Get the base line for the 500 datapoints before the vis1
                    base_line = encoder_positions_aligned_vis1[:VisualStimulus1]
                    base_line = base_line[~np.isnan(base_line)]
                    base_line = np.mean(base_line)
                    
                    if base_line >= 0.9:
                        print('trial starts with larger value:', trial + 1)
                        continue
                    
                    threshold_press1 = base_line + 0.1
                    rotatory1 = int(trial_states['LeverRetract1'][0]*1000)  
                    
                    
                    amp1 = np.argmax(encoder_positions_aligned_vis1[rotatory1:VisDetect2])
                    amp1 = amp1 + rotatory1
                    onset1_position = first_touch_onset(encoder_positions_aligned_vis1, threshold_press1,0,amp1)
                    onset_press1.append(onset1_position)
                    
                    # Target 1 to Onset_press 1
                    targetTime_press1.append(np.abs(rotatory1 - onset1_position)/1000) # type: ignore
                    
                    ##### press1
                    amp_press1 = (np.max(encoder_positions_aligned_vis1[rotatory1:VisDetect2]) - base_line)
                    
                    velocity = np.gradient(encoder_positions_aligned_vis1,encoder_times_vis1)
                    velocity = savgol_filter(velocity, window_length=40, polyorder=1)
                    #### Velocity onset
                    on1_vel = velocity_onset(velocity,int(VisDetect1),rotatory1) # type: ignore
                    
                    PreVis2Delay_1 = trial_states['PreVis2Delay'][0]
                    PreVis2Delay_2 = trial_states['PreVis2Delay'][1]
                    PreVis2Delay = PreVis2Delay_2 - PreVis2Delay_1
                    
                    # Getting max velocity for press 1 and press 2 //// velocity at onset1 and onset2
                    if amp1 >= on1_vel: # type: ignore
                        max_press1_vel = np.max(velocity[on1_vel:amp1])
                    else:
                        max_press1_vel = np.max(velocity[amp1:on1_vel])
                        
                    velaton1 = velocity[on1_vel]
                    
                    if trial_type == 1:
                        target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                        PreVis2Delay_session_short.append(PreVis2Delay)
                        amp_press1_s.append(amp_press1)
                        max_press1_vel_s.append(max_press1_vel)
                        velaton1_s.append(velaton1)
                        if session_data['chemo'][i] == 1:
                            chemo_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                            chemo_PreVis2Delay_session_short.append(PreVis2Delay)
                            chemo_amp_press1_s.append(amp_press1)
                            chemo_max_press1_vel_s.append(max_press1_vel)
                            chemo_velaton1_s.append(velaton1)
                            
                            # POOLED
                            P_chemo_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                            P_chemo_PreVis2Delay_session_short.append(PreVis2Delay)
                            P_chemo_amp_press1_s.append(amp_press1)
                            P_chemo_max_press1_vel_s.append(max_press1_vel)
                            P_chemo_velaton1_s.append(velaton1)
                        else:
                            if opto[trial] == 1:
                                opto_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                opto_PreVis2Delay_session_short.append(PreVis2Delay)
                                opto_amp_press1_s.append(amp_press1)
                                opto_max_press1_vel_s.append(max_press1_vel)
                                opto_velaton1_s.append(velaton1)
                                
                                # POOLED
                                P_opto_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                P_opto_PreVis2Delay_session_short.append(PreVis2Delay)
                                P_opto_amp_press1_s.append(amp_press1)
                                P_opto_max_press1_vel_s.append(max_press1_vel)
                                P_opto_velaton1_s.append(velaton1)
                            else:
                                con_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                con_PreVis2Delay_session_short.append(PreVis2Delay)
                                con_amp_press1_s.append(amp_press1)
                                con_max_press1_vel_s.append(max_press1_vel)
                                con_velaton1_s.append(velaton1)
                                
                                # POOLED
                                P_con_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                P_con_PreVis2Delay_session_short.append(PreVis2Delay)
                                P_con_amp_press1_s.append(amp_press1)
                                P_con_max_press1_vel_s.append(max_press1_vel)
                                P_con_velaton1_s.append(velaton1)

                        
                    else:
                        target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                        PreVis2Delay_session_long.append(PreVis2Delay)
                        amp_press1_l.append(amp_press1)
                        max_press1_vel_l.append(max_press1_vel)
                        velaton1_l.append(velaton1)
                        if session_data['chemo'][i] == 1:
                            chemo_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                            chemo_PreVis2Delay_session_long.append(PreVis2Delay)
                            chemo_amp_press1_l.append(amp_press1)
                            chemo_max_press1_vel_l.append(max_press1_vel)
                            chemo_velaton1_l.append(velaton1)
                            
                            # POOLED
                            P_chemo_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                            P_chemo_PreVis2Delay_session_long.append(PreVis2Delay)
                            P_chemo_amp_press1_l.append(amp_press1)
                            P_chemo_max_press1_vel_l.append(max_press1_vel)
                            P_chemo_velaton1_l.append(velaton1)
                        else:
                            if opto[trial] == 1:
                                opto_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                opto_PreVis2Delay_session_long.append(PreVis2Delay)
                                opto_amp_press1_l.append(amp_press1)
                                opto_max_press1_vel_l.append(max_press1_vel)
                                opto_velaton1_l.append(velaton1)
                                
                                # POOLED
                                P_opto_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                P_opto_PreVis2Delay_session_long.append(PreVis2Delay)
                                P_opto_amp_press1_l.append(amp_press1)
                                P_opto_max_press1_vel_l.append(max_press1_vel)
                                P_opto_velaton1_l.append(velaton1)
                            else:
                                con_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                con_PreVis2Delay_session_long.append(PreVis2Delay)
                                con_amp_press1_l.append(amp_press1)
                                con_max_press1_vel_l.append(max_press1_vel)
                                con_velaton1_l.append(velaton1)
                                
                                # POOLED
                                P_con_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                P_con_PreVis2Delay_session_long.append(PreVis2Delay)
                                P_con_amp_press1_l.append(amp_press1)
                                P_con_max_press1_vel_l.append(max_press1_vel)
                                P_con_velaton1_l.append(velaton1)
                    
                    j = j + 1
                # For EarlyPress2, We have both onset1 and 2
                elif TrialOutcomes[trial] == 'EarlyPress2':
                    trial_type = trial_types[trial]
                    encoder_data = raw_data['EncoderData'][trial]              
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                    ####
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
                    
                    # Check to avoid nan                 
                    # Extract the relevant values
                    values_to_check = [
                    trial_states['WaitForPress1'][0],
                    trial_states['VisualStimulus1'][0],
                    trial_states['LeverRetract1'][1],
                    trial_states['VisDetect1'][0],
                    trial_states['Punish'][0],
                    trial_states['Punish'][1],
                    trial_states['PreVis2Delay'][0],
                    trial_states['LeverRetract1'][0]
                    ]
                    # Check for NaN values
                    if any(np.isnan(value) for value in values_to_check):
                        print("At least one of the values is NaN in ", trial)
                        continue
                    
                    waitforpress1 = int(trial_states['WaitForPress1'][0]*1000)  # press1 interval finding 
                    VisualStimulus1 = int(trial_states['VisualStimulus1'][0]*1000)  #
                    LeverRetract1 = int(trial_states['LeverRetract1'][1]*1000) #
                    VisDetect1 = int(trial_states['VisDetect1'][0]*1000) #needed               
                    punish1 = int(trial_states['Punish'][0]*1000)#
                    punish2 = int(trial_states['Punish'][1]*1000)#
                    PreVis2Delay = int(trial_states['PreVis2Delay'][0]*1000)#
                    # Get the base line for the 500 datapoints before the vis1
                    base_line = encoder_positions_aligned_vis1[:VisualStimulus1]
                    base_line = base_line[~np.isnan(base_line)]
                    base_line = np.mean(base_line)
                    
                    if base_line >= 0.9:
                        print('trial starts with larger value:', trial + 1)
                        continue
                    
                    threshold_press1 = base_line + 0.1
                    rotatory1 = int(trial_states['LeverRetract1'][0]*1000) #
                    
                    
                    amp1 = np.argmax(encoder_positions_aligned_vis1[rotatory1:LeverRetract1])
                    amp1 = amp1 +rotatory1
                    onset1_position = first_touch_onset(encoder_positions_aligned_vis1, threshold_press1,0,amp1)
                    
                    # Target 1 to Onset_press 1
                    targetTime_press1.append(np.abs(rotatory1 - onset1_position)/1000) # type: ignore
                    
                    ##### press1
                    amp_press1 = (np.max(encoder_positions_aligned_vis1[rotatory1:LeverRetract1]) - base_line)
                    

                    ####### Calculations for vis 2
                    threshold_press2 = base_line + 0.25
                    amp2 = np.argmax(encoder_positions_aligned_vis1[punish1:punish2])
                    amp2 = amp2 + punish1
                    onset2_position = first_touch_onset(encoder_positions_aligned_vis1, threshold_press2,PreVis2Delay,amp2)
                    
                    # Target 2 to Onset_press 2
                    targetTime_press2.append(np.abs(punish1 - onset2_position)/1000) # type: ignore
                    targetTime_press2_early.append(np.abs(punish1 - onset2_position)/1000) # type: ignore
                    ##### press2
                    amp_press2 = (np.max(encoder_positions_aligned_vis1[punish1:punish2]) - base_line)
                    amp_press2_early = (np.max(encoder_positions_aligned_vis1[punish1:punish2]) - base_line)
                    ##### Interval onset
                    intervall = onset2_position - onset1_position # type: ignore
                    interval_early.append(intervall/1000)
                    interval.append(intervall/1000)
                    
                    encoder_positions_aligned_vis1 = savgol_filter(encoder_positions_aligned_vis1, window_length=40, polyorder=3)
                    velocity = np.gradient(encoder_positions_aligned_vis1, encoder_times_vis1)
                    velocity = savgol_filter(velocity, window_length=40, polyorder=1)
                    #### Velocity onset
                    on1_vel = velocity_onset(velocity,int(VisDetect1),rotatory1) # type: ignore
                    on2_vel = velocity_onset(velocity,int(PreVis2Delay),punish1) # type: ignore
                    intervall_vel = on2_vel - on1_vel # type: ignore
                    
                    PreVis2Delay_1 = trial_states['PreVis2Delay'][0]
                    PreVis2Delay_2 = trial_states['PreVis2Delay'][1]
                    PreVis2Delay = PreVis2Delay_2 - PreVis2Delay_1
                    # Getting max velocity for press 1 and press 2 //// velocity at onset1 and onset2
                    if amp1 >= on1_vel:                     # type: ignore
                        max_press1_vel = np.max(velocity[on1_vel:amp1])
                    else:
                        max_press1_vel = np.max(velocity[amp1:on1_vel])
                        
                    if amp2 >= on2_vel: # type: ignore
                        max_press2_vel = np.max(velocity[on2_vel:amp2])
                    else:
                        max_press2_vel = np.max(velocity[amp2:on2_vel])
                        
                    velaton1 = velocity[on1_vel]
                    velaton2 = velocity[on2_vel]
                    
                    if trial_type == 1:
                        target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                        target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                        target2_velocity_early_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                        interval_velocity_s.append(intervall_vel/1000)
                        interval_velocity_early_s.append(intervall_vel/1000)
                        PreVis2Delay_session_short.append(PreVis2Delay)
                        amp_press1_s.append(amp_press1)
                        amp_press2_s.append(amp_press2)
                        amp_press2_early_s.append(amp_press2_early)
                        max_press1_vel_s.append(max_press1_vel)
                        max_press2_vel_early_s.append(max_press2_vel)
                        velaton1_s.append(velaton1)
                        velaton2_early_s.append(velaton2)
                        if session_data['chemo'][i] == 1:
                            chemo_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            chemo_target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                            chemo_interval_velocity_s.append(intervall_vel/1000)
                            chemo_PreVis2Delay_session_short.append(PreVis2Delay)
                            chemo_amp_press1_s.append(amp_press1)
                            chemo_amp_press2_s.append(amp_press2)
                            chemo_max_press1_vel_s.append(max_press1_vel)
                            chemo_max_press2_vel_s.append(max_press2_vel)
                            chemo_velaton1_s.append(velaton1)
                            chemo_velaton2_s.append(velaton2)
                            
                            # POOLED
                            P_chemo_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            P_chemo_target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                            P_chemo_interval_velocity_s.append(intervall_vel/1000)
                            P_chemo_PreVis2Delay_session_short.append(PreVis2Delay)
                            P_chemo_amp_press1_s.append(amp_press1)
                            P_chemo_amp_press2_s.append(amp_press2)
                            P_chemo_max_press1_vel_s.append(max_press1_vel)
                            P_chemo_max_press2_vel_s.append(max_press2_vel)
                            P_chemo_velaton1_s.append(velaton1)
                            P_chemo_velaton2_s.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                opto_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                opto_target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                opto_interval_velocity_s.append(intervall_vel/1000)
                                opto_PreVis2Delay_session_short.append(PreVis2Delay)
                                opto_amp_press1_s.append(amp_press1)
                                opto_amp_press2_s.append(amp_press2)
                                opto_max_press1_vel_s.append(max_press1_vel)
                                opto_max_press2_vel_s.append(max_press2_vel)
                                opto_velaton1_s.append(velaton1)
                                opto_velaton2_s.append(velaton2)
                                
                                # POOLED
                                P_opto_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_opto_target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                P_opto_interval_velocity_s.append(intervall_vel/1000)
                                P_opto_PreVis2Delay_session_short.append(PreVis2Delay)
                                P_opto_amp_press1_s.append(amp_press1)
                                P_opto_amp_press2_s.append(amp_press2)
                                P_opto_max_press1_vel_s.append(max_press1_vel)
                                P_opto_max_press2_vel_s.append(max_press2_vel)
                                P_opto_velaton1_s.append(velaton1)
                                P_opto_velaton2_s.append(velaton2)
                            else:
                                con_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                con_target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                con_interval_velocity_s.append(intervall_vel/1000)
                                con_PreVis2Delay_session_short.append(PreVis2Delay)
                                con_amp_press1_s.append(amp_press1)
                                con_amp_press2_s.append(amp_press2)
                                con_max_press1_vel_s.append(max_press1_vel)
                                con_max_press2_vel_s.append(max_press2_vel)
                                con_velaton1_s.append(velaton1)
                                con_velaton2_s.append(velaton2)
                                
                                # POOLED
                                P_con_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_con_target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                P_con_interval_velocity_s.append(intervall_vel/1000)
                                P_con_PreVis2Delay_session_short.append(PreVis2Delay)
                                P_con_amp_press1_s.append(amp_press1)
                                P_con_amp_press2_s.append(amp_press2)
                                P_con_max_press1_vel_s.append(max_press1_vel)
                                P_con_max_press2_vel_s.append(max_press2_vel)
                                P_con_velaton1_s.append(velaton1)
                                P_con_velaton2_s.append(velaton2)
                        
                    else:
                        target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                        target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                        target2_velocity_early_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                        interval_velocity_l.append(intervall_vel/1000)
                        interval_velocity_early_l.append(intervall_vel/1000)
                        PreVis2Delay_session_long.append(PreVis2Delay)
                        amp_press1_l.append(amp_press1)
                        amp_press2_l.append(amp_press2)
                        amp_press2_early_l.append(amp_press2_early)
                        max_press1_vel_l.append(max_press1_vel)
                        max_press2_vel_early_l.append(max_press2_vel)
                        velaton1_l.append(velaton1)
                        velaton2_early_l.append(velaton1)
                        if session_data['chemo'][i] == 1:
                            chemo_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            chemo_target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                            chemo_interval_velocity_l.append(intervall_vel/1000)
                            chemo_PreVis2Delay_session_long.append(PreVis2Delay)
                            chemo_amp_press1_l.append(amp_press1)
                            chemo_amp_press2_l.append(amp_press2)
                            chemo_max_press1_vel_l.append(max_press1_vel)
                            chemo_max_press2_vel_l.append(max_press2_vel)
                            chemo_velaton1_l.append(velaton1)
                            chemo_velaton2_l.append(velaton2)
                            
                            # POOLED
                            P_chemo_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            P_chemo_target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                            P_chemo_interval_velocity_l.append(intervall_vel/1000)
                            P_chemo_PreVis2Delay_session_long.append(PreVis2Delay)
                            P_chemo_amp_press1_l.append(amp_press1)
                            P_chemo_amp_press2_l.append(amp_press2)
                            P_chemo_max_press1_vel_l.append(max_press1_vel)
                            P_chemo_max_press2_vel_l.append(max_press2_vel)
                            P_chemo_velaton1_l.append(velaton1)
                            P_chemo_velaton2_l.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                opto_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                opto_target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                opto_interval_velocity_l.append(intervall_vel/1000)
                                opto_PreVis2Delay_session_long.append(PreVis2Delay)
                                opto_amp_press1_l.append(amp_press1)
                                opto_amp_press2_l.append(amp_press2)
                                opto_max_press1_vel_l.append(max_press1_vel)
                                opto_max_press2_vel_l.append(max_press2_vel)
                                opto_velaton1_l.append(velaton1)
                                opto_velaton2_l.append(velaton2)
                                
                                # POOLED
                                P_opto_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_opto_target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                P_opto_interval_velocity_l.append(intervall_vel/1000)
                                P_opto_PreVis2Delay_session_long.append(PreVis2Delay)
                                P_opto_amp_press1_l.append(amp_press1)
                                P_opto_amp_press2_l.append(amp_press2)
                                P_opto_max_press1_vel_l.append(max_press1_vel)
                                P_opto_max_press2_vel_l.append(max_press2_vel)
                                P_opto_velaton1_l.append(velaton1)
                                P_opto_velaton2_l.append(velaton2)
                            else:
                                con_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                con_target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                con_interval_velocity_l.append(intervall_vel/1000)
                                con_PreVis2Delay_session_long.append(PreVis2Delay)
                                con_amp_press1_l.append(amp_press1)
                                con_amp_press2_l.append(amp_press2)
                                con_max_press1_vel_l.append(max_press1_vel)
                                con_max_press2_vel_l.append(max_press2_vel)
                                con_velaton1_l.append(velaton1)
                                con_velaton2_l.append(velaton2)
                                
                                # POOLED
                                P_con_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_con_target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                P_con_interval_velocity_l.append(intervall_vel/1000)
                                P_con_PreVis2Delay_session_long.append(PreVis2Delay)
                                P_con_amp_press1_l.append(amp_press1)
                                P_con_amp_press2_l.append(amp_press2)
                                P_con_max_press1_vel_l.append(max_press1_vel)
                                P_con_max_press2_vel_l.append(max_press2_vel)
                                P_con_velaton1_l.append(velaton1)
                                P_con_velaton2_l.append(velaton2)
                    
                    j = j + 1 
                    
            # Plotting for each session (onset to peak for both stimulations)
            axs[0,0].hist(targetTime_press1, bins=100, rwidth= 0.5)
            axs[0,0].set_title('onset1_position to target1 histogram')
            axs[0,0].set_xlabel('Interval Time (s)')
            axs[0,0].set_ylabel('Count of Rewarded Trials')
            
            axs[0,0].spines['top'].set_visible(False)
            axs[0,0].spines['right'].set_visible(False)
                    
            axs[1,0].plot(targetTime_press1) 
            axs[1,0].set_title('onset1_position to target1 for each trial')
            axs[1,0].set_xlabel('Trials')
            axs[1,0].set_ylabel('Interval Time (s)')
            
            axs[1,0].spines['top'].set_visible(False)
            axs[1,0].spines['right'].set_visible(False)
            
            axs[0,1].hist(targetTime_press2, bins=100, rwidth= 0.5)
            axs[0,1].set_title('onset2_position to target2 histogram')
            axs[0,1].set_xlabel('Interval Time (s)')
            axs[0,1].set_ylabel('Count of Rewarded Trials')
            
            axs[0,1].spines['top'].set_visible(False)
            axs[0,1].spines['right'].set_visible(False)

            axs[1,1].plot(targetTime_press2) 
            axs[1,1].set_title('onset2_position to target2 for each trial')
            axs[1,1].set_xlabel('Trials')
            axs[1,1].set_ylabel('Interval Time (s)')
            
            axs[1,1].spines['top'].set_visible(False)
            axs[1,1].spines['right'].set_visible(False)
            
            axs[2,0].hist(target1_velocity_s, bins=100, rwidth= 0.5, color = 'blue',label = 'short')
            axs[2,0].hist(target1_velocity_l, bins=100, rwidth= 0.5, color = 'green',label = 'long')
            axs[2,0].set_title('onset1_velocity to target1 histogram')
            axs[2,0].set_xlabel('Interval Time (s)')
            axs[2,0].set_ylabel('Count of Rewarded Trials')
            axs[2,0].legend()
            
            axs[2,0].spines['top'].set_visible(False)
            axs[2,0].spines['right'].set_visible(False)
            
            axs[3,0].plot(target1_velocity_s, color = 'blue', label = 'short')        
            axs[3,0].plot(target1_velocity_l, color = 'green', label = 'long') 
            axs[3,0].set_title('onset1_velocity to target1 for each trial')
            axs[3,0].set_xlabel('Trials')
            axs[3,0].set_ylabel('Interval Time (s)')
            
            axs[3,0].spines['top'].set_visible(False)
            axs[3,0].spines['right'].set_visible(False)
            
            axs[2,1].hist(target2_velocity_s, bins=100, rwidth= 0.5, color = 'blue',label = 'short')
            axs[2,1].hist(target2_velocity_l, bins=100, rwidth= 0.5, color = 'green',label = 'long')
            axs[2,1].set_title('onset2_velocity to target2 histogram')
            axs[2,1].set_xlabel('Interval Time (s)')
            axs[2,1].set_ylabel('Count of Rewarded Trials')
            axs[2,1].legend()
            
            axs[2,1].spines['top'].set_visible(False)
            axs[2,1].spines['right'].set_visible(False)

            axs[3,1].plot(target2_velocity_s, color = 'blue', label = 'short')
            axs[3,1].plot(target2_velocity_l, color = 'green', label = 'long') 
            axs[3,1].set_title('onset2_velocity to target2 for each trial')
            axs[3,1].set_xlabel('Trials')
            axs[3,1].set_ylabel('Interval Time (s)')
            axs[3,1].legend()
            
            axs[3,1].spines['top'].set_visible(False)
            axs[3,1].spines['right'].set_visible(False)
                    
            axs[4,0].hist(amp_press1_s, bins=20, rwidth= 0.5, color = 'blue', label = 'short')
            axs[4,0].hist(amp_press1_l, bins=20, rwidth= 0.5, color = 'green', label = 'long')
            axs[4,0].set_title('amp difference baseline and peak1 histogram')
            axs[4,0].set_xlabel('Amplitude (deg)')
            axs[4,0].set_ylabel('Count of Rewarded Trials')
            axs[4,0].legend()
            
            axs[4,0].spines['top'].set_visible(False)
            axs[4,0].spines['right'].set_visible(False)

            axs[4,1].hist(amp_press2_s, bins=20, rwidth= 0.5, color = 'blue', label = 'short') 
            axs[4,1].hist(amp_press2_l, bins=20, rwidth= 0.5, color = 'green', label = 'long')
            axs[4,1].set_title('amp difference baseline and peak2 histogram')
            axs[4,1].set_xlabel('Amplitude (deg)')
            axs[4,1].set_ylabel('Count of Rewarded Trials')
            axs[4,1].legend()
            
            axs[4,1].spines['top'].set_visible(False)
            axs[4,1].spines['right'].set_visible(False)
        
            axs[5,0].plot(amp_press1_s, color = 'blue', label = 'short')
            axs[5,0].plot(amp_press1_l, color = 'green', label = 'long')
            axs[5,0].set_title('amp difference baseline and peak1 for each trial')
            axs[5,0].set_xlabel('Trials')
            axs[5,0].set_ylabel('Amplitude (deg)')
            axs[5,0].legend()
            
            axs[5,0].spines['top'].set_visible(False)
            axs[5,0].spines['right'].set_visible(False)
        
            axs[5,1].plot(amp_press2_s, color = 'blue', label = 'short') 
            axs[5,1].plot(amp_press2_s, color = 'green', label = 'long') 
            axs[5,1].set_title('amp difference baseline and peak2 for each trial')
            axs[5,1].set_xlabel('Trials')
            axs[5,1].set_ylabel('Amplitude (deg)')
            axs[5,1].legend()
            
            axs[5,1].spines['top'].set_visible(False)
            axs[5,1].spines['right'].set_visible(False)
            
            axs[6,0].hist(interval,bins=100, rwidth= 0.5)
            axs[6,0].set_title('Time Interval between onset1 & 2 Position histogram')
            axs[6,0].set_xlabel('time (s)')
            axs[6,0].set_ylabel('Count of Rewarded Trials')
            
            axs[6,0].spines['top'].set_visible(False)
            axs[6,0].spines['right'].set_visible(False)
            
            
            axs[7,0].plot(interval) 
            axs[7,0].set_title('Time Interval between onset1 & 2 Position')
            axs[7,0].set_xlabel('Trials')
            axs[7,0].set_ylabel('Time Interval (s)')
            
            axs[7,0].spines['top'].set_visible(False)
            axs[7,0].spines['right'].set_visible(False)
        
            axs[6,1].hist(interval_velocity_s,bins=100, rwidth= 0.5, color = 'blue',label = 'short')
            axs[6,1].hist(interval_velocity_l,bins=100, rwidth= 0.5, color = 'green',label = 'long')
            axs[6,1].set_title('Time Interval between onset1 & 2 Velocity histogram')
            axs[6,1].set_xlabel('time (s)')
            axs[6,1].set_ylabel('Count of Rewarded Trials')
            axs[6,1].legend()
            
            axs[6,1].spines['top'].set_visible(False)
            axs[6,1].spines['right'].set_visible(False)
            
            axs[7,1].plot(interval_velocity_s, color = 'blue',label = 'short')
            axs[7,1].plot(interval_velocity_l, color = 'green',label = 'long') 
            axs[7,1].set_title('Time Interval between onset1 & 2 Velocity')
            axs[7,1].set_xlabel('Trials')
            axs[7,1].set_ylabel('Time Interval (s)')
            axs[7,1].legend()
            
            axs[7,1].spines['top'].set_visible(False)
            axs[7,1].spines['right'].set_visible(False)
            
            fig.tight_layout()
            
            ################
            G_chemo_target1_velocity_s.append(np.mean(chemo_target1_velocity_s, axis = 0))
            G_chemo_target2_velocity_s.append(np.mean(chemo_target2_velocity_s, axis = 0))
            G_chemo_interval_velocity_s.append(np.mean(chemo_interval_velocity_s, axis = 0))
            G_chemo_PreVis2Delay_session_short.append(np.mean(chemo_PreVis2Delay_session_short, axis = 0))
            G_chemo_amp_press1_s.append(np.mean(chemo_amp_press1_s, axis = 0))
            G_chemo_amp_press2_s.append(np.mean(chemo_amp_press2_s, axis = 0))
            G_chemo_max_press1_vel_s.append(np.mean(chemo_max_press1_vel_s, axis = 0))
            G_chemo_max_press2_vel_s.append(np.mean(chemo_max_press2_vel_s, axis = 0))
            G_chemo_velaton1_s.append(np.mean(chemo_velaton1_s, axis = 0))
            G_chemo_velaton2_s.append(np.mean(chemo_velaton2_s, axis = 0))

            G_opto_target1_velocity_s.append(np.mean(opto_target1_velocity_s, axis = 0))
            G_opto_target2_velocity_s.append(np.mean(opto_target2_velocity_s, axis = 0))
            G_opto_interval_velocity_s.append(np.mean(opto_interval_velocity_s, axis = 0))
            G_opto_PreVis2Delay_session_short.append(np.mean(opto_PreVis2Delay_session_short, axis = 0))
            G_opto_amp_press1_s.append(np.mean(opto_amp_press1_s, axis = 0))
            G_opto_amp_press2_s.append(np.mean(opto_amp_press2_s, axis = 0))
            G_opto_max_press1_vel_s.append(np.mean(opto_max_press1_vel_s, axis = 0))
            G_opto_max_press2_vel_s.append(np.mean(opto_max_press2_vel_s, axis = 0))
            G_opto_velaton1_s.append(np.mean(opto_velaton1_s, axis = 0))
            G_opto_velaton2_s.append(np.mean(opto_velaton2_s, axis = 0))

            G_con_target1_velocity_s.append(np.mean(con_target1_velocity_s, axis = 0))
            G_con_target2_velocity_s.append(np.mean(con_target2_velocity_s, axis = 0))
            G_con_interval_velocity_s.append(np.mean(con_interval_velocity_s, axis = 0))
            G_con_PreVis2Delay_session_short.append(np.mean(con_PreVis2Delay_session_short, axis = 0))
            G_con_amp_press1_s.append(np.mean(con_amp_press1_s, axis = 0))
            G_con_amp_press2_s.append(np.mean(con_amp_press2_s, axis = 0))
            G_con_max_press1_vel_s.append(np.mean(con_max_press1_vel_s, axis = 0))
            G_con_max_press2_vel_s.append(np.mean(con_max_press2_vel_s, axis = 0))
            G_con_velaton1_s.append(np.mean(con_velaton1_s, axis = 0))
            G_con_velaton2_s.append(np.mean(con_velaton2_s, axis = 0))
            #### Long ####
            G_chemo_target1_velocity_l.append(np.mean(chemo_target1_velocity_l, axis = 0))
            G_chemo_target2_velocity_l.append(np.mean(chemo_target2_velocity_l, axis = 0))
            G_chemo_interval_velocity_l.append(np.mean(chemo_interval_velocity_l, axis = 0))
            G_chemo_PreVis2Delay_session_long.append(np.mean(chemo_PreVis2Delay_session_long, axis = 0))
            G_chemo_amp_press1_l.append(np.mean(chemo_amp_press1_l, axis = 0))
            G_chemo_amp_press2_l.append(np.mean(chemo_amp_press2_l, axis = 0))
            G_chemo_max_press1_vel_l.append(np.mean(chemo_max_press1_vel_l, axis = 0))
            G_chemo_max_press2_vel_l.append(np.mean(chemo_max_press2_vel_l, axis = 0))
            G_chemo_velaton1_l.append(np.mean(chemo_velaton1_l, axis = 0))
            G_chemo_velaton2_l.append(np.mean(chemo_velaton2_l, axis = 0))

            G_opto_target1_velocity_l.append(np.mean(opto_target1_velocity_l, axis = 0))
            G_opto_target2_velocity_l.append(np.mean(opto_target2_velocity_l, axis = 0))
            G_opto_interval_velocity_l.append(np.mean(opto_interval_velocity_l, axis = 0))
            G_opto_PreVis2Delay_session_long.append(np.mean(opto_PreVis2Delay_session_long, axis = 0))
            G_opto_amp_press1_l.append(np.mean(opto_amp_press1_l, axis = 0))
            G_opto_amp_press2_l.append(np.mean(opto_amp_press2_l, axis = 0))
            G_opto_max_press1_vel_l.append(np.mean(opto_max_press1_vel_l, axis = 0))
            G_opto_max_press2_vel_l.append(np.mean(opto_max_press2_vel_l, axis = 0))
            G_opto_velaton1_l.append(np.mean(opto_velaton1_l, axis = 0))
            G_opto_velaton2_l.append(np.mean(opto_velaton2_l, axis = 0))

            G_con_target1_velocity_l.append(np.mean(con_target1_velocity_l, axis = 0))
            G_con_target2_velocity_l.append(np.mean(con_target2_velocity_l, axis = 0))
            G_con_interval_velocity_l.append(np.mean(con_interval_velocity_l, axis = 0))
            G_con_PreVis2Delay_session_long.append(np.mean(con_PreVis2Delay_session_long, axis = 0))
            G_con_amp_press1_l.append(np.mean(con_amp_press1_l, axis = 0))
            G_con_amp_press2_l.append(np.mean(con_amp_press2_l, axis = 0))
            G_con_max_press1_vel_l.append(np.mean(con_max_press1_vel_l, axis = 0))
            G_con_max_press2_vel_l.append(np.mean(con_max_press2_vel_l, axis = 0))
            G_con_velaton1_l.append(np.mean(con_velaton1_l, axis = 0))
            G_con_velaton2_l.append(np.mean(con_velaton2_l, axis = 0))
            
            #####################
            
            std_error_1 = np.std(targetTime_press1, ddof=1) / np.sqrt(len(targetTime_press1))
            std_targetTime_press1.append(std_error_1)
            mean_targetTime_press1.append(np.mean(targetTime_press1))
            
            std_error_2 = np.std(targetTime_press2_reward, ddof=1) / np.sqrt(len(targetTime_press2_reward))
            std_targetTime_press2_reward.append(std_error_2)
            mean_targetTime_press2_reward.append(np.mean(targetTime_press2_reward))
            
            std_error_13 = np.std(targetTime_press2_early, ddof=1) / np.sqrt(len(targetTime_press2_early))
            std_targetTime_press2_early.append(std_error_13)
            mean_targetTime_press2_early.append(np.mean(targetTime_press2_early))
            
            std_error_3 = np.std(amp_press1_s, ddof=1) / np.sqrt(len(amp_press1_s))
            std_amp_press1_s.append(std_error_3)
            mean_amp_press1_s.append(np.mean(amp_press1_s))
            
            std_error_3 = np.std(amp_press1_l, ddof=1) / np.sqrt(len(amp_press1_l))
            std_amp_press1_l.append(std_error_3)
            mean_amp_press1_l.append(np.mean(amp_press1_l))
            
            std_error_4 = np.std(amp_press2_reward_s, ddof=1) / np.sqrt(len(amp_press2_reward_s))
            std_amp_press2_reward_s.append(std_error_4)
            mean_amp_press2_reward_s.append(np.mean(amp_press2_reward_s))
            
            std_error_4 = np.std(amp_press2_reward_l, ddof=1) / np.sqrt(len(amp_press2_reward_l))
            std_amp_press2_reward_l.append(std_error_4)
            mean_amp_press2_reward_l.append(np.mean(amp_press2_reward_l))
            
            std_error_11 = np.std(amp_press2_early_s, ddof=1) / np.sqrt(len(amp_press2_early_s))
            std_amp_press2_early_s.append(std_error_11)
            mean_amp_press2_early_s.append(np.mean(amp_press2_early_s))
            
            std_error_11 = np.std(amp_press2_early_l, ddof=1) / np.sqrt(len(amp_press2_early_l))
            std_amp_press2_early_l.append(std_error_11)
            mean_amp_press2_early_l.append(np.mean(amp_press2_early_l))
            
            std_error_5 = np.std(interval_reward, ddof=1) / np.sqrt(len(interval_reward))
            std_interval_reward.append(std_error_5)
            mean_interval_reward.append(np.mean(interval_reward))
            
            std_error_10 = np.std(interval_early, ddof=1) / np.sqrt(len(interval_early))
            std_interval_early.append(std_error_10)
            mean_interval_early.append(np.mean(interval_early))
            
            std_error_6  = np.std(target1_velocity_s, ddof=1) / np.sqrt(len(target1_velocity_s))
            std_target1_velocity_s.append(std_error_6)
            mean_target1_velocity_s.append(np.mean(target1_velocity_s))
            
            std_error_7  = np.std(target2_velocity_reward_s, ddof=1) / np.sqrt(len(target2_velocity_reward_s))
            std_target2_velocity_reward_s.append(std_error_7)
            mean_target2_velocity_reward_s.append(np.mean(target2_velocity_reward_s))
            
            std_error_12  = np.std(target2_velocity_early_s, ddof=1) / np.sqrt(len(target2_velocity_early_s))
            std_target2_velocity_early_s.append(std_error_12)
            mean_target2_velocity_early_s.append(np.mean(target2_velocity_early_s))
            
            std_error_8  = np.std(interval_velocity_reward_s, ddof=1) / np.sqrt(len(interval_velocity_reward_s))
            std_interval_velocity_reward_s.append(std_error_8)
            mean_interval_velocity_reward_s.append(np.mean(interval_velocity_reward_s))
            
            std_error_9  = np.std(interval_velocity_early_s, ddof=1) / np.sqrt(len(interval_velocity_early_s))
            std_interval_velocity_early_s.append(std_error_9)
            mean_interval_velocity_early_s.append(np.mean(interval_velocity_early_s))
            ###############################################################
            std_error_14  = np.std(target1_velocity_l, ddof=1) / np.sqrt(len(target1_velocity_l))
            std_target1_velocity_l.append(std_error_14)
            mean_target1_velocity_l.append(np.mean(target1_velocity_l))
            
            std_error_15  = np.std(target2_velocity_reward_l, ddof=1) / np.sqrt(len(target2_velocity_reward_l))
            std_target2_velocity_reward_l.append(std_error_15)
            mean_target2_velocity_reward_l.append(np.mean(target2_velocity_reward_l))
            
            std_error_16  = np.std(target2_velocity_early_l, ddof=1) / np.sqrt(len(target2_velocity_early_l))
            std_target2_velocity_early_l.append(std_error_16)
            mean_target2_velocity_early_l.append(np.mean(target2_velocity_early_l))
            
            std_error_17  = np.std(interval_velocity_reward_l, ddof=1) / np.sqrt(len(interval_velocity_reward_l))
            std_interval_velocity_reward_l.append(std_error_17)
            mean_interval_velocity_reward_l.append(np.mean(interval_velocity_reward_l))
            
            std_error_18  = np.std(interval_velocity_early_l, ddof=1) / np.sqrt(len(interval_velocity_early_l))
            std_interval_velocity_early_l.append(std_error_18)
            mean_interval_velocity_early_l.append(np.mean(interval_velocity_early_l))
            ###################################################################
            std_error = np.std(PreVis2Delay_session_short, ddof=1) / np.sqrt(len(PreVis2Delay_session_short))
            std_PreVis2Delay_session_short.append(std_error)
            mean_PreVis2Delay_session_short.append(np.mean(PreVis2Delay_session_short))
            
            std_error = np.std(PreVis2Delay_session_long, ddof=1) / np.sqrt(len(PreVis2Delay_session_long))
            std_PreVis2Delay_session_long.append(std_error)
            mean_PreVis2Delay_session_long.append(np.mean(PreVis2Delay_session_long))
            ##############################################################################
            std_error = np.std(max_press1_vel_s, ddof=1) / np.sqrt(len(max_press1_vel_s))
            std_max_press1_vel_s.append(std_error)
            mean_max_press1_vel_s.append(np.mean(max_press1_vel_s))
            
            std_error = np.std(max_press1_vel_l, ddof=1) / np.sqrt(len(max_press1_vel_l))
            std_max_press1_vel_l.append(std_error)
            mean_max_press1_vel_l.append(np.mean(max_press1_vel_l))
            
            std_error = np.std(max_press2_vel_early_s, ddof=1) / np.sqrt(len(max_press2_vel_early_s))
            std_max_press2_vel_early_s.append(std_error)
            mean_max_press2_vel_early_s.append(np.mean(max_press2_vel_early_s))
            
            std_error = np.std(max_press2_vel_early_l, ddof=1) / np.sqrt(len(max_press2_vel_early_l))
            std_max_press2_vel_early_l.append(std_error)
            mean_max_press2_vel_early_l.append(np.mean(max_press2_vel_early_l))
            
            std_error = np.std(max_press2_vel_reward_s, ddof=1) / np.sqrt(len(max_press2_vel_reward_s))
            std_max_press2_vel_reward_s.append(std_error)
            mean_max_press2_vel_reward_s.append(np.mean(max_press2_vel_reward_s))
            
            std_error = np.std(max_press2_vel_reward_l, ddof=1) / np.sqrt(len(max_press2_vel_reward_l))
            std_max_press2_vel_reward_l.append(std_error)
            mean_max_press2_vel_reward_l.append(np.mean(max_press2_vel_reward_l))
            #############################################################################
            std_error = np.std(velaton1_s, ddof=1) / np.sqrt(len(velaton1_s))
            std_velaton1_s.append(std_error)
            mean_velaton1_s.append(np.mean(velaton1_s))
            
            std_error = np.std(velaton1_l, ddof=1) / np.sqrt(len(velaton1_l))
            std_velaton1_l.append(std_error)
            mean_velaton1_l.append(np.mean(velaton1_l))
            
            std_error = np.std(velaton2_early_s, ddof=1) / np.sqrt(len(velaton2_early_s))
            std_velaton2_early_s.append(std_error)
            mean_velaton2_early_s.append(np.mean(velaton2_early_s))
            
            std_error = np.std(velaton2_early_l, ddof=1) / np.sqrt(len(velaton2_early_l))
            std_velaton2_early_l.append(std_error)
            mean_velaton2_early_l.append(np.mean(velaton2_early_l))
            
            std_error = np.std(velaton2_reward_s, ddof=1) / np.sqrt(len(velaton2_reward_s))
            std_velaton2_reward_s.append(std_error)
            mean_velaton2_reward_s.append(np.mean(velaton2_reward_s))
            
            std_error = np.std(velaton2_reward_l, ddof=1) / np.sqrt(len(velaton2_reward_l))
            std_velaton2_reward_l.append(std_error)
            mean_velaton2_reward_l.append(np.mean(velaton2_reward_l))
            
            output_figs_dir = output_dir_onedrive + subject + '/'    
            output_imgs_dir = output_dir_local + subject + '/time analysis_imgs/'    
            os.makedirs(output_figs_dir, exist_ok = True)
            os.makedirs(output_imgs_dir, exist_ok = True)
            fig.savefig(output_figs_dir + subject + '_' + session_date + '_timing_amp_analysis_' + '.pdf', dpi=300)
            # fig.savefig(output_imgs_dir + subject + '_time_analysis_'+ session_date +'.png', dpi=300)
            plt.close(fig)

        # Plotting for anlyze over all sessions
        axs1[2,0].errorbar(numeric_dates, mean_amp_press1_s, yerr=std_amp_press1_s, fmt='o', capsize=4, color = 'blue', label = 'Rew & Early2_S')
        axs1[2,0].errorbar(numeric_dates + offset, mean_amp_press1_l, yerr=std_amp_press1_l, fmt='o', capsize=4, color = 'green', label = 'Rew & Early2_L')
        axs1[2,0].set_title('baseline to peak press1')
        axs1[2,0].set_ylabel('Mean joystick deflection (deg) +/- SEM')
        axs1[2,0].set_xlabel('Sessions')
        axs1[2,0].set_xticks(numeric_dates)
        axs1[2,0].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[2,0].spines['top'].set_visible(False)
        axs1[2,0].spines['right'].set_visible(False)
        axs1[2,0].legend()

        axs1[2,1].errorbar(numeric_dates, mean_amp_press2_reward_s, yerr=std_amp_press2_reward_s, fmt='o', capsize=4, color= 'blue', label = 'Rewarded_S')
        axs1[2,1].errorbar(numeric_dates + offset, mean_amp_press2_reward_l, yerr=std_amp_press2_reward_l, fmt='o', capsize=4, color= 'green', label = 'Rewarded_L')
        axs1[2,1].errorbar(numeric_dates + 2*offset, mean_amp_press2_early_s, yerr=std_amp_press2_early_s, fmt='_', capsize=4, color= 'blue', label = 'Early_Press_S')
        axs1[2,1].errorbar(numeric_dates + 3*offset, mean_amp_press2_early_l, yerr=std_amp_press2_early_l, fmt='_', capsize=4, color= 'green', label = 'Early_Press_L')
        axs1[2,1].set_title('baseline to peak press2')
        axs1[2,1].set_ylabel('Mean joystick deflection (deg) +/- SEM')
        axs1[2,1].set_xlabel('Sessions')
        axs1[2,1].set_xticks(numeric_dates)
        axs1[2,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[2,1].legend()
        axs1[2,1].spines['top'].set_visible(False)
        axs1[2,1].spines['right'].set_visible(False)
        
        axs1[0,0].errorbar(numeric_dates, mean_target1_velocity_s, yerr=std_target1_velocity_s, fmt='o', capsize=4, color = 'blue', label = 'Rew & Early2_S')
        axs1[0,0].errorbar(numeric_dates + offset, mean_target1_velocity_l, yerr=std_target1_velocity_l, fmt='o', capsize=4, color = 'green', label = 'Rew & Early2_L')
        axs1[0,0].set_title('onset1 to target1')
        axs1[0,0].set_ylabel('Mean time Interval (s) +/- SEM')
        axs1[0,0].set_xlabel('Sessions')
        axs1[0,0].set_xticks(numeric_dates)
        axs1[0,0].set_xticklabels(dates, rotation=90, ha = 'center') 
        axs1[0,0].legend()
        axs1[0,0].spines['top'].set_visible(False)
        axs1[0,0].spines['right'].set_visible(False)

        axs1[0,1].errorbar(numeric_dates, mean_target2_velocity_reward_s, yerr=std_target2_velocity_reward_s, fmt='o', capsize=4, color= 'blue', label = 'Rewarded_S')
        axs1[0,1].errorbar(numeric_dates + offset, mean_target2_velocity_reward_l, yerr=std_target2_velocity_reward_l, fmt='o', capsize=4, color= 'green', label = 'Rewarded_L')
        axs1[0,1].errorbar(numeric_dates + 2*offset, mean_target2_velocity_early_s, yerr=std_target2_velocity_early_s, fmt='_', capsize=4, color= 'blue', label = 'Early_Press_S')
        axs1[0,1].errorbar(numeric_dates + 3*offset, mean_target2_velocity_early_l, yerr=std_target2_velocity_early_l, fmt='_', capsize=4, color= 'green', label = 'Early_Press_L')
        axs1[0,1].set_title('onset2 to target2')
        axs1[0,1].set_ylabel('Mean time Interval (s) +/- SEM ')
        axs1[0,1].set_xlabel('Sessions')
        axs1[0,1].set_xticks(numeric_dates)
        axs1[0,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[0,1].legend()
        axs1[0,1].spines['top'].set_visible(False)
        axs1[0,1].spines['right'].set_visible(False)
        
        axs1[1,0].errorbar(numeric_dates, mean_interval_velocity_reward_s, yerr=std_interval_velocity_reward_s, fmt='o', capsize=4, color= 'blue', label = 'Rewarded_S')
        axs1[1,0].errorbar(numeric_dates + offset, mean_interval_velocity_reward_l, yerr=std_interval_velocity_reward_l, fmt='o', capsize=4, color= 'green', label = 'Rewarded_L')
        axs1[1,0].errorbar(numeric_dates + 2*offset, mean_interval_velocity_early_s, yerr=std_interval_velocity_early_s, fmt='_', capsize=4, color= 'blue', label = 'Early_Press_S')
        axs1[1,0].errorbar(numeric_dates + 3*offset, mean_interval_velocity_early_l, yerr=std_interval_velocity_early_l, fmt='_', capsize=4, color= 'green', label = 'Early_Press_L')
        axs1[1,0].set_title('Interval time between onset1 & 2')
        axs1[1,0].set_ylabel('Mean Time (s) +/- SEM')
        axs1[1,0].set_xlabel('Sessions')
        axs1[1,0].set_xticks(numeric_dates)
        axs1[1,0].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[1,0].legend()
        axs1[1,0].spines['top'].set_visible(False)
        axs1[1,0].spines['right'].set_visible(False)
        
        axs1[1,1].errorbar(numeric_dates, mean_PreVis2Delay_session_short, yerr=std_PreVis2Delay_session_short, fmt='o', capsize=4, color = 'blue', label = 'short')
        axs1[1,1].errorbar(numeric_dates + offset, mean_PreVis2Delay_session_long, yerr=std_PreVis2Delay_session_long, fmt='o', capsize=4, color = 'green', label = 'long')
        axs1[1,1].set_title('Previs2Delay')
        axs1[1,1].set_ylabel('Mean preVis2Delay (s) +/- SEM')
        axs1[1,1].set_xlabel('Sessions')
        axs1[1,1].set_xticks(numeric_dates)
        axs1[1,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[1,1].legend()
        axs1[1,1].spines['top'].set_visible(False)
        axs1[1,1].spines['right'].set_visible(False)
        
        axs1[3,0].errorbar(numeric_dates, mean_max_press1_vel_s, yerr=std_max_press1_vel_s, fmt='o', capsize=4, color = 'blue', label = 'Rew & Early2_S')
        axs1[3,0].errorbar(numeric_dates + offset, mean_max_press1_vel_l, yerr=std_max_press1_vel_l, fmt='o', capsize=4, color = 'green', label = 'Rew & Early2_L')
        axs1[3,0].set_title('peak velocity at press1')
        axs1[3,0].set_ylabel('Mean velocity (deg/s) +/- SEM ')
        axs1[3,0].set_xlabel('Sessions')
        axs1[3,0].set_xticks(numeric_dates)
        axs1[3,0].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[3,0].legend()
        axs1[3,0].spines['top'].set_visible(False)
        axs1[3,0].spines['right'].set_visible(False)

        
        axs1[3,1].errorbar(numeric_dates, mean_max_press2_vel_reward_s, yerr=std_max_press2_vel_reward_s, fmt='o', capsize=4, color= 'blue', label = 'Rewarded_S')
        axs1[3,1].errorbar(numeric_dates + offset, mean_max_press2_vel_reward_l, yerr=std_max_press2_vel_reward_l, fmt='o', capsize=4, color= 'green', label = 'Rewarded_L')
        axs1[3,1].errorbar(numeric_dates + 2*offset, mean_max_press2_vel_early_s, yerr=std_max_press2_vel_early_s, fmt='_', capsize=4, color= 'blue', label = 'Early_Press2_S')
        axs1[3,1].errorbar(numeric_dates + 3*offset, mean_max_press2_vel_early_l, yerr=std_max_press2_vel_early_l, fmt='_', capsize=4, color= 'green', label = 'Early_Press2_L')
        axs1[3,1].set_title('peak velocity at press2')
        axs1[3,1].set_ylabel('Mean velocity (deg/s) +/- SEM ')
        axs1[3,1].set_xlabel('Sessions')
        axs1[3,1].set_xticks(numeric_dates)
        axs1[3,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[3,1].legend()
        axs1[3,1].spines['top'].set_visible(False)
        axs1[3,1].spines['right'].set_visible(False)

        axs1[4,0].errorbar(numeric_dates, mean_velaton1_s, yerr=std_velaton1_s, fmt='o', capsize=4, color = 'blue', label = 'Rew & Early2_S')
        axs1[4,0].errorbar(numeric_dates + offset, mean_velaton1_l, yerr=std_velaton1_l, fmt='o', capsize=4, color = 'green', label = 'Rew & Early2_L')
        axs1[4,0].set_title('velocity at onset1')
        axs1[4,0].set_ylabel('Mean velocity (deg/s) +/- SEM ')
        axs1[4,0].set_xlabel('Sessions')
        axs1[4,0].set_xticks(numeric_dates)
        axs1[4,0].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[4,0].legend()
        axs1[4,0].spines['top'].set_visible(False)
        axs1[4,0].spines['right'].set_visible(False)

        
        axs1[4,1].errorbar(numeric_dates, mean_velaton2_reward_s, yerr=std_velaton2_reward_s, fmt='o', capsize=4, color= 'blue', label = 'Rewarded_S')
        axs1[4,1].errorbar(numeric_dates + offset, mean_velaton2_reward_l, yerr=std_velaton2_reward_l, fmt='o', capsize=4, color= 'green', label = 'Rewarded_L')
        axs1[4,1].errorbar(numeric_dates + 2*offset, mean_velaton2_early_s, yerr=std_velaton2_early_s, fmt='_', capsize=4, color= 'blue', label = 'Early_Press2_S')
        axs1[4,1].errorbar(numeric_dates + 3*offset, mean_velaton2_early_l, yerr=std_velaton2_early_l, fmt='_', capsize=4, color= 'green', label = 'Early_Press2_L')
        axs1[4,1].set_title('velocity at onset2')
        axs1[4,1].set_ylabel('Mean velocity (deg/s) +/- SEM ')
        axs1[4,1].set_xlabel('Sessions')
        axs1[4,1].set_xticks(numeric_dates)
        axs1[4,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[4,1].legend()
        axs1[4,1].spines['top'].set_visible(False)
        axs1[4,1].spines['right'].set_visible(False)    
        
        fig1.tight_layout()
        
        # os.makedirs(output_figs_dir, exist_ok = True)
        # os.makedirs(output_imgs_dir, exist_ok = True)
        # fig1.savefig(output_figs_dir + subject + '_time_analysis_oversessions.pdf', dpi=300)
        # fig1.savefig(output_imgs_dir + subject + '_time_analysis_imgs_oversessions.png', dpi=300)
        # plt.close(fig1)
        
    else:
        print('selftime')
        
        fig1,axs1 = plt.subplots(nrows=5, ncols=2, figsize=(15, 25))
        fig1.suptitle(subject + ' \n ' + 'Timeing analysis over sessions' + '\n')


        for i in range(0 , len(session_id)):
            TrialOutcomes = session_data['outcomes'][i]
            reward_count = TrialOutcomes.count('Reward')
            opto = session_data['session_opto_tag'][i]
            onset_press1 = []
            # rotatory1_1 = []
            onset_press2 = []
            stimulations2_peak = []
            targetTime_press1 = []
            targetTime_press2 = []
            targetTime_press2_reward = []
            targetTime_press2_early = []
            press_vis_2 = []
            amp_press1_s = []
            amp_press2_s = []
            amp_press1_l = []
            amp_press2_l = []
            amp_press2_reward_s = []
            amp_press2_early_s = []
            amp_press2_reward_l = []
            amp_press2_early_l = []
            
            # VisualStimulus11 = []
            interval_early = []
            interval_reward = []
            target1_velocity_s = []
            target2_velocity_s = []
            target2_velocity_early_s = []
            target2_velocity_reward_s = []
            interval_velocity_reward_s = []
            interval_velocity_early_s = []
            interval_velocity_s = []
            interval = []
            
            # interval_early_l = []
            # interval_reward_l = []
            target1_velocity_l = []
            target2_velocity_l = []
            target2_velocity_early_l = []
            target2_velocity_reward_l = []
            interval_velocity_reward_l = []
            interval_velocity_early_l = []
            interval_velocity_l = []
            # interval_l = []
            
            PreVis2Delay_session_short = []
            PreVis2Delay_session_long = []
            
            velaton1_s = []
            max_press1_vel_s = []
            velaton1_l = []
            max_press1_vel_l = []
            velaton2_reward_s = []
            max_press2_vel_reward_s = []
            velaton2_reward_l = []
            max_press2_vel_reward_l = []
            velaton2_early_s = []
            max_press2_vel_early_s = []
            velaton2_early_l = []
            max_press2_vel_early_l = []
            x = []
            j = 0
            # GRAND ######################################
            chemo_target1_velocity_s = []
            chemo_target2_velocity_s = []
            chemo_interval_velocity_s = []
            chemo_PreVis2Delay_session_short = []
            chemo_amp_press1_s = []
            chemo_amp_press2_s = []
            chemo_max_press1_vel_s = []
            chemo_max_press2_vel_s = []
            chemo_velaton1_s = []
            chemo_velaton2_s = []

            opto_target1_velocity_s = []
            opto_target2_velocity_s = []
            opto_interval_velocity_s = []
            opto_PreVis2Delay_session_short = []
            opto_amp_press1_s = []
            opto_amp_press2_s = []
            opto_max_press1_vel_s = []
            opto_max_press2_vel_s = []
            opto_velaton1_s = []
            opto_velaton2_s = []

            con_target1_velocity_s = []
            con_target2_velocity_s = []
            con_interval_velocity_s = []
            con_PreVis2Delay_session_short = []
            con_amp_press1_s = []
            con_amp_press2_s = []
            con_max_press1_vel_s = []
            con_max_press2_vel_s = []
            con_velaton1_s = []
            con_velaton2_s = []
            #### Long ####
            chemo_target1_velocity_l = []
            chemo_target2_velocity_l = []
            chemo_interval_velocity_l = []
            chemo_PreVis2Delay_session_long = []
            chemo_amp_press1_l = []
            chemo_amp_press2_l = []
            chemo_max_press1_vel_l = []
            chemo_max_press2_vel_l = []
            chemo_velaton1_l = []
            chemo_velaton2_l = []

            opto_target1_velocity_l = []
            opto_target2_velocity_l = []
            opto_interval_velocity_l = []
            opto_PreVis2Delay_session_long = []
            opto_amp_press1_l = []
            opto_amp_press2_l = []
            opto_max_press1_vel_l = []
            opto_max_press2_vel_l = []
            opto_velaton1_l = []
            opto_velaton2_l = []

            con_target1_velocity_l = []
            con_target2_velocity_l = []
            con_interval_velocity_l = []
            con_PreVis2Delay_session_long = []
            con_amp_press1_l = []
            con_amp_press2_l = []
            con_max_press1_vel_l = []
            con_max_press2_vel_l = []
            con_velaton1_l = []
            con_velaton2_l = []
            # We have Raw data and extract every thing from it (Times)
            raw_data = session_data['raw'][i]
            session_date = dates[i][2:]
            
            # Creating figures for each session
            fig, axs = plt.subplots(nrows=8, ncols=2, figsize=(10, 32))   # should be changed
            fig.suptitle(subject + ' - ' + session_date + ' \n ' + 'Time analysis for Reward trials' + '\n')
            
            print('timing analysis of session:' + session_date)
            # The loop for each session
            for trial in range(0,len(TrialOutcomes)):
                if np.isnan(isSelfTimedMode[i][trial]):
                    continue
                
                trial_types = raw_data['TrialTypes']
                
                if TrialOutcomes[trial] == 'Reward':
                    
                    trial_type = trial_types[trial]
                    encoder_data = raw_data['EncoderData'][trial]              
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
                    
                    # Check to avoid nan                 
                    # Extract the relevant values
                    values_to_check = [
                    trial_states['WaitForPress1'][0],
                    trial_states['WaitForPress2'][0],
                    trial_states['VisDetect1'][0],
                    trial_states['VisualStimulus1'][0],
                    trial_states['LeverRetract2'][1],
                    trial_states['LeverRetract1'][1],
                    trial_event['RotaryEncoder1_1'][0],
                    trial_states['Reward'][0]]
                    # Check for NaN values
                    if any(np.isnan(value) for value in values_to_check):
                        print("At least one of the values is NaN in ", trial)
                        continue
                    
                    waitforpress1 = int(trial_states['WaitForPress1'][0]*1000)  # press1 interval finding
                    waitforpress2 = int(trial_states['WaitForPress2'][0]*1000) 
                    VisDetect1 = int(trial_states['VisDetect1'][0]*1000) #needed
                    VisualStimulus1 = int(trial_states['VisualStimulus1'][0]*1000) 
                    LeverRetract2 = int(trial_states['LeverRetract2'][1]*1000) 
                    LeverRetract1 = int(trial_states['LeverRetract1'][1]*1000) 
                    
                    # Get the base line for the 500 datapoints before the vis1
                    base_line = encoder_positions_aligned_vis1[:VisualStimulus1]
                    base_line = base_line[~np.isnan(base_line)]
                    base_line = np.mean(base_line)
                    
                    if base_line >= 0.9:
                            print('trial starts with larger value:', trial + 1)
                            continue
                    
                    threshold_press1 = base_line + 0.1
                    rotatory1 = int(trial_event['RotaryEncoder1_1'][0]*1000) 
                    rotatory2 = int(trial_states['Reward'][0]*1000) 
                    
                    
                    amp1 = np.argmax(encoder_positions_aligned_vis1[rotatory1:LeverRetract1])
                    amp1 = amp1 +rotatory1
                    onset1_position = first_touch_onset(encoder_positions_aligned_vis1, threshold_press1,0,amp1)
                                    
                    # Target 1 to Onset_press 1
                    targetTime_press1.append(np.abs(rotatory1 - onset1_position)/1000) # type: ignore
                    
                    ##### press1
                    amp_press1 = (np.max(encoder_positions_aligned_vis1[rotatory1:LeverRetract1]) - base_line)
                    

                    ####### Calculations for vis 2
                    threshold_press2 = base_line + 0.25
                    amp2 = np.argmax(encoder_positions_aligned_vis1[LeverRetract1:LeverRetract2])
                    amp2 = amp2 + LeverRetract1
                    onset2_position = first_touch_onset(encoder_positions_aligned_vis1, threshold_press2,LeverRetract1,amp2)
                    
                    # Target 2 to Onset_press 2
                    targetTime_press2.append(np.abs(rotatory2 - onset2_position)/1000) # type: ignore
                    targetTime_press2_reward.append(np.abs(rotatory2 - onset2_position)/1000) # type: ignore
                    ##### press2
                    amp_press2 = (np.max(encoder_positions_aligned_vis1[LeverRetract1:LeverRetract2]) - base_line)
                    amp_press2_reward = (np.max(encoder_positions_aligned_vis1[LeverRetract1:LeverRetract2]) - base_line)
                    
                    ##### Interval onset
                    intervall = onset2_position - onset1_position # type: ignore
                    interval.append(intervall/1000)
                    interval_reward.append(intervall/1000)
                    
                    encoder_positions_aligned_vis1 = savgol_filter(encoder_positions_aligned_vis1, window_length=40, polyorder=3)
                    velocity = np.gradient(encoder_positions_aligned_vis1, encoder_times_vis1)
                    velocity = savgol_filter(velocity, window_length=40, polyorder=1)
                    #### Velocity onset
                    on1_vel = velocity_onset(velocity,int(VisDetect1),rotatory1) # type: ignore
                    on2_vel = velocity_onset(velocity,waitforpress2,rotatory2) # type: ignore
                    intervall_vel = on2_vel - on1_vel # type: ignore
                    
                    PrePress2Delay_1 = trial_states['PrePress2Delay'][0]
                    PrePress2Delay_2 = trial_states['PrePress2Delay'][1]
                    PrePress2Delay = PrePress2Delay_2 - PrePress2Delay_1
                    
                    # Getting max velocity for press 1 and press 2 //// velocity at onset1 and onset2
                    if on1_vel <= amp1: # type: ignore
                        max_press1_vel = np.max(velocity[on1_vel:amp1])
                    else:
                        max_press1_vel = np.max(velocity[amp1:on1_vel])

                    if on2_vel <= amp2: # type: ignore
                        max_press2_vel = np.max(velocity[on2_vel:amp2])
                    else:
                        max_press2_vel = np.max(velocity[amp2:on2_vel])
                        
                    velaton1 = velocity[on1_vel]
                    velaton2 = velocity[on2_vel]
                    
                    if trial_type == 1:
                        target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                        target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                        target2_velocity_reward_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                        interval_velocity_s.append(intervall_vel/1000)
                        interval_velocity_reward_s.append(intervall_vel/1000)
                        PreVis2Delay_session_short.append(PrePress2Delay)
                        amp_press1_s.append(amp_press1)
                        amp_press2_s.append(amp_press2)
                        amp_press2_reward_s.append(amp_press2_reward)
                        max_press1_vel_s.append(max_press1_vel)
                        max_press2_vel_reward_s.append(max_press2_vel)
                        velaton1_s.append(velaton1)
                        velaton2_reward_s.append(velaton2)
                        if session_data['chemo'][i] == 1:
                            chemo_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            chemo_target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                            chemo_interval_velocity_s.append(intervall_vel/1000)
                            chemo_PreVis2Delay_session_short.append(PrePress2Delay)
                            chemo_amp_press1_s.append(amp_press1)
                            chemo_amp_press2_s.append(amp_press2)
                            chemo_max_press1_vel_s.append(max_press1_vel)
                            chemo_max_press2_vel_s.append(max_press2_vel)
                            chemo_velaton1_s.append(velaton1)
                            chemo_velaton2_s.append(velaton2)
                            
                            # POOLED
                            P_chemo_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            P_chemo_target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                            P_chemo_interval_velocity_s.append(intervall_vel/1000)
                            P_chemo_PreVis2Delay_session_short.append(PrePress2Delay)
                            P_chemo_amp_press1_s.append(amp_press1)
                            P_chemo_amp_press2_s.append(amp_press2)
                            P_chemo_max_press1_vel_s.append(max_press1_vel)
                            P_chemo_max_press2_vel_s.append(max_press2_vel)
                            P_chemo_velaton1_s.append(velaton1)
                            P_chemo_velaton2_s.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                opto_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                opto_target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                opto_interval_velocity_s.append(intervall_vel/1000)
                                opto_PreVis2Delay_session_short.append(PrePress2Delay)
                                opto_amp_press1_s.append(amp_press1)
                                opto_amp_press2_s.append(amp_press2)
                                opto_max_press1_vel_s.append(max_press1_vel)
                                opto_max_press2_vel_s.append(max_press2_vel)
                                opto_velaton1_s.append(velaton1)
                                opto_velaton2_s.append(velaton2)
                                
                                # POOLED
                                P_opto_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_opto_target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                P_opto_interval_velocity_s.append(intervall_vel/1000)
                                P_opto_PreVis2Delay_session_short.append(PrePress2Delay)
                                P_opto_amp_press1_s.append(amp_press1)
                                P_opto_amp_press2_s.append(amp_press2)
                                P_opto_max_press1_vel_s.append(max_press1_vel)
                                P_opto_max_press2_vel_s.append(max_press2_vel)
                                P_opto_velaton1_s.append(velaton1)
                                P_opto_velaton2_s.append(velaton2)
                            else:
                                con_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                con_target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                con_interval_velocity_s.append(intervall_vel/1000)
                                con_PreVis2Delay_session_short.append(PrePress2Delay)
                                con_amp_press1_s.append(amp_press1)
                                con_amp_press2_s.append(amp_press2)
                                con_max_press1_vel_s.append(max_press1_vel)
                                con_max_press2_vel_s.append(max_press2_vel)
                                con_velaton1_s.append(velaton1)
                                con_velaton2_s.append(velaton2)
                                
                                # POOLED
                                P_con_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_con_target2_velocity_s.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                P_con_interval_velocity_s.append(intervall_vel/1000)
                                P_con_PreVis2Delay_session_short.append(PrePress2Delay)
                                P_con_amp_press1_s.append(amp_press1)
                                P_con_amp_press2_s.append(amp_press2)
                                P_con_max_press1_vel_s.append(max_press1_vel)
                                P_con_max_press2_vel_s.append(max_press2_vel)
                                P_con_velaton1_s.append(velaton1)
                                P_con_velaton2_s.append(velaton2)
                            
                    else:
                    
                        target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                        target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                        target2_velocity_reward_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                        interval_velocity_l.append(intervall_vel/1000)
                        interval_velocity_reward_l.append(intervall_vel/1000)
                        PreVis2Delay_session_long.append(PrePress2Delay)
                        amp_press1_l.append(amp_press1)
                        amp_press2_l.append(amp_press2)
                        amp_press2_reward_l.append(amp_press2_reward)
                        max_press1_vel_l.append(max_press1_vel)
                        max_press2_vel_reward_l.append(max_press2_vel)
                        velaton1_l.append(velaton1)
                        velaton2_reward_l.append(velaton1)
                        if session_data['chemo'][i] == 1:
                            chemo_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            chemo_target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                            chemo_interval_velocity_l.append(intervall_vel/1000)
                            chemo_PreVis2Delay_session_long.append(PrePress2Delay)
                            chemo_amp_press1_l.append(amp_press1)
                            chemo_amp_press2_l.append(amp_press2)
                            chemo_max_press1_vel_l.append(max_press1_vel)
                            chemo_max_press2_vel_l.append(max_press2_vel)
                            chemo_velaton1_l.append(velaton1)
                            chemo_velaton2_l.append(velaton2)
                            
                            # POOLED
                            P_chemo_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            P_chemo_target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                            P_chemo_interval_velocity_l.append(intervall_vel/1000)
                            P_chemo_PreVis2Delay_session_long.append(PrePress2Delay)
                            P_chemo_amp_press1_l.append(amp_press1)
                            P_chemo_amp_press2_l.append(amp_press2)
                            P_chemo_max_press1_vel_l.append(max_press1_vel)
                            P_chemo_max_press2_vel_l.append(max_press2_vel)
                            P_chemo_velaton1_l.append(velaton1)
                            P_chemo_velaton2_l.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                opto_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                opto_target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                opto_interval_velocity_l.append(intervall_vel/1000)
                                opto_PreVis2Delay_session_long.append(PrePress2Delay)
                                opto_amp_press1_l.append(amp_press1)
                                opto_amp_press2_l.append(amp_press2)
                                opto_max_press1_vel_l.append(max_press1_vel)
                                opto_max_press2_vel_l.append(max_press2_vel)
                                opto_velaton1_l.append(velaton1)
                                opto_velaton2_l.append(velaton2)
                                
                                # POOLED
                                P_opto_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_opto_target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                P_opto_interval_velocity_l.append(intervall_vel/1000)
                                P_opto_PreVis2Delay_session_long.append(PrePress2Delay)
                                P_opto_amp_press1_l.append(amp_press1)
                                P_opto_amp_press2_l.append(amp_press2)
                                P_opto_max_press1_vel_l.append(max_press1_vel)
                                P_opto_max_press2_vel_l.append(max_press2_vel)
                                P_opto_velaton1_l.append(velaton1)
                                P_opto_velaton2_l.append(velaton2)
                            else:
                                con_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                con_target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                con_interval_velocity_l.append(intervall_vel/1000)
                                con_PreVis2Delay_session_long.append(PrePress2Delay)
                                con_amp_press1_l.append(amp_press1)
                                con_amp_press2_l.append(amp_press2)
                                con_max_press1_vel_l.append(max_press1_vel)
                                con_max_press2_vel_l.append(max_press2_vel)
                                con_velaton1_l.append(velaton1)
                                con_velaton2_l.append(velaton2)
                                
                                # POOLED
                                P_con_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_con_target2_velocity_l.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                                P_con_interval_velocity_l.append(intervall_vel/1000)
                                P_con_PreVis2Delay_session_long.append(PrePress2Delay)
                                P_con_amp_press1_l.append(amp_press1)
                                P_con_amp_press2_l.append(amp_press2)
                                P_con_max_press1_vel_l.append(max_press1_vel)
                                P_con_max_press2_vel_l.append(max_press2_vel)
                                P_con_velaton1_l.append(velaton1)
                                P_con_velaton2_l.append(velaton2)
            
                    
                    j = j+1            
                # For DidNotPress2 (WE have the press one only)
                elif TrialOutcomes[trial] == 'DidNotPress2':
                    trial_type = trial_types[trial]
                    encoder_data = raw_data['EncoderData'][trial]              
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                    ####
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
                    
                    # Check to avoid nan                 
                    # Extract the relevant values
                    values_to_check = [
                    trial_states['VisualStimulus1'][0],
                    trial_states['WaitForPress2'][0],
                    trial_states['VisDetect1'][0],
                    trial_states['LeverRetract1'][0]
                    ]
                    # Check for NaN values
                    if any(np.isnan(value) for value in values_to_check):
                        print("At least one of the values is NaN in ", trial)
                        continue
                    
                    VisualStimulus1 = int(trial_states['VisualStimulus1'][0]*1000) #needed
                    WaitForPress2 = int(trial_states['WaitForPress2'][0]*1000) #needed
                    VisDetect1 = int(trial_states['VisDetect1'][0]*1000) #needed
                    # Get the base line for the 500 datapoints before the vis1
                    base_line = encoder_positions_aligned_vis1[:VisualStimulus1]
                    base_line = base_line[~np.isnan(base_line)]
                    base_line = np.mean(base_line)
                    
                    if base_line >= 0.9:
                        print('trial starts with larger value:', trial + 1)
                        continue
                    
                    threshold_press1 = base_line + 0.1
                    rotatory1 = int(trial_states['LeverRetract1'][0]*1000)  
                    
                    
                    amp1 = np.argmax(encoder_positions_aligned_vis1[rotatory1:WaitForPress2])
                    amp1 = amp1 + rotatory1
                    onset1_position = first_touch_onset(encoder_positions_aligned_vis1, threshold_press1,0,amp1)
                    onset_press1.append(onset1_position)
                    
                    # Target 1 to Onset_press 1
                    targetTime_press1.append(np.abs(rotatory1 - onset1_position)/1000) # type: ignore
                    
                    ##### press1
                    amp_press1 = (np.max(encoder_positions_aligned_vis1[rotatory1:WaitForPress2]) - base_line)
                    
                    encoder_positions_aligned_vis1 = savgol_filter(encoder_positions_aligned_vis1, window_length=40, polyorder=3)
                    velocity = np.gradient(encoder_positions_aligned_vis1,encoder_times_vis1)
                    velocity = savgol_filter(velocity, window_length=40, polyorder=1)
                    #### Velocity onset
                    on1_vel = velocity_onset(velocity,int(VisDetect1),rotatory1) # type: ignore
                    
                    PrePress2Delay_1 = trial_states['PrePress2Delay'][0]
                    PrePress2Delay_2 = trial_states['PrePress2Delay'][1]
                    PrePress2Delay = PrePress2Delay_2 - PrePress2Delay_1
                    # Getting max velocity for press 1 and press 2 //// velocity at onset1 and onset2
                    if on1_vel <= amp1: # type: ignore
                        max_press1_vel = np.max(velocity[on1_vel:amp1])
                    else:
                        max_press1_vel = np.max(velocity[amp1:on1_vel])
                    velaton1 = velocity[on1_vel]
                    
                    if trial_type == 1:
                        target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                        PreVis2Delay_session_short.append(PrePress2Delay)
                        amp_press1_s.append(amp_press1)
                        max_press1_vel_s.append(max_press1_vel)
                        velaton1_s.append(velaton1)
                        if session_data['chemo'][i] == 1:
                            chemo_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                            chemo_PreVis2Delay_session_short.append(PrePress2Delay)
                            chemo_amp_press1_s.append(amp_press1)
                            chemo_max_press1_vel_s.append(max_press1_vel)
                            chemo_velaton1_s.append(velaton1)
                            
                            # POOLED
                            P_chemo_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                            P_chemo_PreVis2Delay_session_short.append(PrePress2Delay)
                            P_chemo_amp_press1_s.append(amp_press1)
                            P_chemo_max_press1_vel_s.append(max_press1_vel)
                            P_chemo_velaton1_s.append(velaton1)
                        else:
                            if opto[trial] == 1:
                                opto_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                opto_PreVis2Delay_session_short.append(PrePress2Delay)
                                opto_amp_press1_s.append(amp_press1)
                                opto_max_press1_vel_s.append(max_press1_vel)
                                opto_velaton1_s.append(velaton1)
                                
                                # POOLED
                                P_opto_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                P_opto_PreVis2Delay_session_short.append(PrePress2Delay)
                                P_opto_amp_press1_s.append(amp_press1)
                                P_opto_max_press1_vel_s.append(max_press1_vel)
                                P_opto_velaton1_s.append(velaton1)
                            else:
                                con_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                con_PreVis2Delay_session_short.append(PrePress2Delay)
                                con_amp_press1_s.append(amp_press1)
                                con_max_press1_vel_s.append(max_press1_vel)
                                con_velaton1_s.append(velaton1)
                                
                                # POOLED
                                P_con_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                P_con_PreVis2Delay_session_short.append(PrePress2Delay)
                                P_con_amp_press1_s.append(amp_press1)
                                P_con_max_press1_vel_s.append(max_press1_vel)
                                P_con_velaton1_s.append(velaton1)
                                
                                
                            
                    else:
                        target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                        PreVis2Delay_session_long.append(PrePress2Delay)
                        amp_press1_l.append(amp_press1)
                        max_press1_vel_l.append(max_press1_vel)
                        velaton1_l.append(velaton1)
                        if session_data['chemo'][i] == 1:
                            chemo_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                            chemo_PreVis2Delay_session_long.append(PrePress2Delay)
                            chemo_amp_press1_l.append(amp_press1)
                            chemo_max_press1_vel_l.append(max_press1_vel)
                            chemo_velaton1_l.append(velaton1)
                            
                            # POOLED
                            P_chemo_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                            P_chemo_PreVis2Delay_session_long.append(PrePress2Delay)
                            P_chemo_amp_press1_l.append(amp_press1)
                            P_chemo_max_press1_vel_l.append(max_press1_vel)
                            P_chemo_velaton1_l.append(velaton1)
                        else:
                            if opto[trial] == 1:
                                opto_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                opto_PreVis2Delay_session_long.append(PrePress2Delay)
                                opto_amp_press1_l.append(amp_press1)
                                opto_max_press1_vel_l.append(max_press1_vel)
                                opto_velaton1_l.append(velaton1)
                                
                                # POOLED
                                P_opto_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                P_opto_PreVis2Delay_session_long.append(PrePress2Delay)
                                P_opto_amp_press1_l.append(amp_press1)
                                P_opto_max_press1_vel_l.append(max_press1_vel)
                                P_opto_velaton1_l.append(velaton1)
                            else:
                                con_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                con_PreVis2Delay_session_long.append(PrePress2Delay)
                                con_amp_press1_l.append(amp_press1)
                                con_max_press1_vel_l.append(max_press1_vel)
                                con_velaton1_l.append(velaton1)
                                
                                # POOLED
                                P_con_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore  
                                P_con_PreVis2Delay_session_long.append(PrePress2Delay)
                                P_con_amp_press1_l.append(amp_press1)
                                P_con_max_press1_vel_l.append(max_press1_vel)
                                P_con_velaton1_l.append(velaton1)
                    
                    j = j + 1
                # For EarlyPress2, We have both onset1 and 2
                elif TrialOutcomes[trial] == 'EarlyPress2':
                    trial_type = trial_types[trial]
                    encoder_data = raw_data['EncoderData'][trial]              
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                    ####
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
                    
                    # Check to avoid nan                 
                    # Extract the relevant values
                    values_to_check = [
                    trial_states['WaitForPress1'][0],
                    trial_states['VisualStimulus1'][0],
                    trial_states['LeverRetract1'][1],
                    trial_states['VisDetect1'][0],
                    trial_states['Punish'][0],
                    trial_states['Punish'][1],
                    trial_states['LeverRetract1'][0]
                    ]
                    # Check for NaN values
                    if any(np.isnan(value) for value in values_to_check):
                        print("At least one of the values is NaN in ", trial)
                        continue
                    
                    waitforpress1 = int(trial_states['WaitForPress1'][0]*1000)  # press1 interval finding 
                    VisualStimulus1 = int(trial_states['VisualStimulus1'][0]*1000)  #
                    LeverRetract1 = int(trial_states['LeverRetract1'][1]*1000) #
                    VisDetect1 = int(trial_states['VisDetect1'][0]*1000) #needed
                    punish1 = int(trial_states['Punish'][0]*1000)#
                    punish2 = int(trial_states['Punish'][1]*1000)#
                    # Get the base line for the 500 datapoints before the vis1
                    base_line = encoder_positions_aligned_vis1[:VisualStimulus1]
                    base_line = base_line[~np.isnan(base_line)]
                    base_line = np.mean(base_line)
                    
                    if base_line >= 0.9:
                        print('trial starts with larger value:', trial + 1)
                        continue
                    
                    threshold_press1 = base_line + 0.1
                    rotatory1 = int(trial_states['LeverRetract1'][0]*1000) #
                    
                    
                    amp1 = np.argmax(encoder_positions_aligned_vis1[rotatory1:LeverRetract1])
                    amp1 = amp1 +rotatory1
                    onset1_position = first_touch_onset(encoder_positions_aligned_vis1, threshold_press1,0,amp1)
                    
                    # Target 1 to Onset_press 1
                    targetTime_press1.append(np.abs(rotatory1 - onset1_position)/1000) # type: ignore
                    
                    ##### press1
                    amp_press1 = (np.max(encoder_positions_aligned_vis1[rotatory1:LeverRetract1]) - base_line)
                    

                    ####### Calculations for vis 2
                    threshold_press2 = base_line + 0.25
                    amp2 = np.argmax(encoder_positions_aligned_vis1[punish1:punish2])
                    amp2 = amp2 + punish1
                    onset2_position = first_touch_onset(encoder_positions_aligned_vis1, threshold_press2,LeverRetract1,amp2)
                    
                    # Target 2 to Onset_press 2
                    targetTime_press2.append(np.abs(punish1 - onset2_position)/1000) # type: ignore
                    targetTime_press2_early.append(np.abs(punish1 - onset2_position)/1000) # type: ignore
                    ##### press2
                    amp_press2 = (np.max(encoder_positions_aligned_vis1[punish1:punish2]) - base_line)
                    amp_press2_early =(np.max(encoder_positions_aligned_vis1[punish1:punish2]) - base_line)
                    
                    ##### Interval onset
                    intervall = onset2_position - onset1_position # type: ignore
                    interval.append(intervall/1000)
                    interval_early.append(intervall/1000)
                    
                    encoder_positions_aligned_vis1 = savgol_filter(encoder_positions_aligned_vis1, window_length=40, polyorder=3)
                    velocity = np.gradient(encoder_positions_aligned_vis1,encoder_times_vis1)
                    velocity = savgol_filter(velocity, window_length=40, polyorder=1)
                    #### Velocity onset
                    # print(trial)
                    on1_vel = velocity_onset(velocity,int(VisDetect1),rotatory1) # type: ignore
                    on2_vel = velocity_onset(velocity,int(LeverRetract1),punish1) # type: ignore
                    intervall_vel = on2_vel - on1_vel # type: ignore
                    
                    PrePress2Delay_1 = trial_states['PrePress2Delay'][0]
                    PrePress2Delay_2 = trial_states['PrePress2Delay'][1]
                    PrePress2Delay = PrePress2Delay_2 - PrePress2Delay_1
                    # Getting max velocity for press 1 and press 2 //// velocity at onset1 and onset2
                    if on1_vel <= amp1: # type: ignore
                        max_press1_vel = np.max(velocity[on1_vel:amp1])
                    else:
                        max_press1_vel = np.max(velocity[amp1:on1_vel])
                        
                    if on2_vel <= amp2: # type: ignore
                        max_press2_vel = np.max(velocity[on2_vel:amp2])
                    else:
                        max_press2_vel = np.max(velocity[amp2:on2_vel])
                        
                    velaton1 = velocity[on1_vel]
                    velaton2 = velocity[on2_vel]
                    
                    if trial_type == 1:
                        target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                        target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                        target2_velocity_early_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                        interval_velocity_s.append(intervall_vel/1000)
                        interval_velocity_early_s.append(intervall_vel/1000)
                        PreVis2Delay_session_short.append(PrePress2Delay)
                        amp_press1_s.append(amp_press1)
                        amp_press2_s.append(amp_press2)
                        amp_press2_early_s.append(amp_press2_early)
                        max_press1_vel_s.append(max_press1_vel)
                        max_press2_vel_early_s.append(max_press2_vel)
                        velaton1_s.append(velaton1)
                        velaton2_early_s.append(velaton2)
                        if session_data['chemo'][i] == 1:
                            chemo_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            chemo_target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                            chemo_interval_velocity_s.append(intervall_vel/1000)
                            chemo_PreVis2Delay_session_short.append(PrePress2Delay)
                            chemo_amp_press1_s.append(amp_press1)
                            chemo_amp_press2_s.append(amp_press2)
                            chemo_max_press1_vel_s.append(max_press1_vel)
                            chemo_max_press2_vel_s.append(max_press2_vel)
                            chemo_velaton1_s.append(velaton1)
                            chemo_velaton2_s.append(velaton2)
                            
                            # POOLED
                            P_chemo_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            P_chemo_target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                            P_chemo_interval_velocity_s.append(intervall_vel/1000)
                            P_chemo_PreVis2Delay_session_short.append(PrePress2Delay)
                            P_chemo_amp_press1_s.append(amp_press1)
                            P_chemo_amp_press2_s.append(amp_press2)
                            P_chemo_max_press1_vel_s.append(max_press1_vel)
                            P_chemo_max_press2_vel_s.append(max_press2_vel)
                            P_chemo_velaton1_s.append(velaton1)
                            P_chemo_velaton2_s.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                opto_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                opto_target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                opto_interval_velocity_s.append(intervall_vel/1000)
                                opto_PreVis2Delay_session_short.append(PrePress2Delay)
                                opto_amp_press1_s.append(amp_press1)
                                opto_amp_press2_s.append(amp_press2)
                                opto_max_press1_vel_s.append(max_press1_vel)
                                opto_max_press2_vel_s.append(max_press2_vel)
                                opto_velaton1_s.append(velaton1)
                                opto_velaton2_s.append(velaton2)
                                
                                # POOLED
                                P_opto_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_opto_target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                P_opto_interval_velocity_s.append(intervall_vel/1000)
                                P_opto_PreVis2Delay_session_short.append(PrePress2Delay)
                                P_opto_amp_press1_s.append(amp_press1)
                                P_opto_amp_press2_s.append(amp_press2)
                                P_opto_max_press1_vel_s.append(max_press1_vel)
                                P_opto_max_press2_vel_s.append(max_press2_vel)
                                P_opto_velaton1_s.append(velaton1)
                                P_opto_velaton2_s.append(velaton2)
                            else:
                                con_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                con_target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                con_interval_velocity_s.append(intervall_vel/1000)
                                con_PreVis2Delay_session_short.append(PrePress2Delay)
                                con_amp_press1_s.append(amp_press1)
                                con_amp_press2_s.append(amp_press2)
                                con_max_press1_vel_s.append(max_press1_vel)
                                con_max_press2_vel_s.append(max_press2_vel)
                                con_velaton1_s.append(velaton1)
                                con_velaton2_s.append(velaton2)
                                
                                # POOLED
                                P_con_target1_velocity_s.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_con_target2_velocity_s.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                P_con_interval_velocity_s.append(intervall_vel/1000)
                                P_con_PreVis2Delay_session_short.append(PrePress2Delay)
                                P_con_amp_press1_s.append(amp_press1)
                                P_con_amp_press2_s.append(amp_press2)
                                P_con_max_press1_vel_s.append(max_press1_vel)
                                P_con_max_press2_vel_s.append(max_press2_vel)
                                P_con_velaton1_s.append(velaton1)
                                P_con_velaton2_s.append(velaton2)
                                
                            
                        
                    else:
                        target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                        target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                        target2_velocity_early_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                        interval_velocity_l.append(intervall_vel/1000)
                        interval_velocity_early_l.append(intervall_vel/1000)
                        PreVis2Delay_session_long.append(PrePress2Delay)
                        amp_press1_l.append(amp_press1)
                        amp_press2_l.append(amp_press2)
                        amp_press2_early_l.append(amp_press2_early)
                        max_press1_vel_l.append(max_press1_vel)
                        max_press2_vel_early_l.append(max_press2_vel)
                        velaton1_l.append(velaton1)
                        velaton2_early_l.append(velaton1)
                        if session_data['chemo'][i] == 1:
                            chemo_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            chemo_target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                            chemo_interval_velocity_l.append(intervall_vel/1000)
                            chemo_PreVis2Delay_session_long.append(PrePress2Delay)
                            chemo_amp_press1_l.append(amp_press1)
                            chemo_amp_press2_l.append(amp_press2)
                            chemo_max_press1_vel_l.append(max_press1_vel)
                            chemo_max_press2_vel_l.append(max_press2_vel)
                            chemo_velaton1_l.append(velaton1)
                            chemo_velaton2_l.append(velaton2)
                            
                            # POOLED
                            P_chemo_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                            P_chemo_target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                            P_chemo_interval_velocity_l.append(intervall_vel/1000)
                            P_chemo_PreVis2Delay_session_long.append(PrePress2Delay)
                            P_chemo_amp_press1_l.append(amp_press1)
                            P_chemo_amp_press2_l.append(amp_press2)
                            P_chemo_max_press1_vel_l.append(max_press1_vel)
                            P_chemo_max_press2_vel_l.append(max_press2_vel)
                            P_chemo_velaton1_l.append(velaton1)
                            P_chemo_velaton2_l.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                opto_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                opto_target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                opto_interval_velocity_l.append(intervall_vel/1000)
                                opto_PreVis2Delay_session_long.append(PrePress2Delay)
                                opto_amp_press1_l.append(amp_press1)
                                opto_amp_press2_l.append(amp_press2)
                                opto_max_press1_vel_l.append(max_press1_vel)
                                opto_max_press2_vel_l.append(max_press2_vel)
                                opto_velaton1_l.append(velaton1)
                                opto_velaton2_l.append(velaton2)
                                
                                # POOLED
                                P_opto_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_opto_target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                P_opto_interval_velocity_l.append(intervall_vel/1000)
                                P_opto_PreVis2Delay_session_long.append(PrePress2Delay)
                                P_opto_amp_press1_l.append(amp_press1)
                                P_opto_amp_press2_l.append(amp_press2)
                                P_opto_max_press1_vel_l.append(max_press1_vel)
                                P_opto_max_press2_vel_l.append(max_press2_vel)
                                P_opto_velaton1_l.append(velaton1)
                                P_opto_velaton2_l.append(velaton2)
                            else:
                                con_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                con_target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                con_interval_velocity_l.append(intervall_vel/1000)
                                con_PreVis2Delay_session_long.append(PrePress2Delay)
                                con_amp_press1_l.append(amp_press1)
                                con_amp_press2_l.append(amp_press2)
                                con_max_press1_vel_l.append(max_press1_vel)
                                con_max_press2_vel_l.append(max_press2_vel)
                                con_velaton1_l.append(velaton1)
                                con_velaton2_l.append(velaton2)
                                
                                # POOLED
                                P_con_target1_velocity_l.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                                P_con_target2_velocity_l.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                                P_con_interval_velocity_l.append(intervall_vel/1000)
                                P_con_PreVis2Delay_session_long.append(PrePress2Delay)
                                P_con_amp_press1_l.append(amp_press1)
                                P_con_amp_press2_l.append(amp_press2)
                                P_con_max_press1_vel_l.append(max_press1_vel)
                                P_con_max_press2_vel_l.append(max_press2_vel)
                                P_con_velaton1_l.append(velaton1)
                                P_con_velaton2_l.append(velaton2)
                    
                    j = j + 1
                    
            # Plotting for each session (onset to peak for both stimulations)
            axs[0,0].hist(targetTime_press1, bins=100, rwidth= 0.5)
            axs[0,0].set_title('onset1_position to target1 histogram')
            axs[0,0].set_xlabel('Interval Time (s)')
            axs[0,0].set_ylabel('Count of Rewarded Trials')
            
            axs[0,0].spines['top'].set_visible(False)
            axs[0,0].spines['right'].set_visible(False)
                    
            axs[1,0].plot(targetTime_press1) 
            axs[1,0].set_title('onset1_position to target1 for each trial')
            axs[1,0].set_xlabel('Trials')
            axs[1,0].set_ylabel('Interval Time (s)')
            
            axs[1,0].spines['top'].set_visible(False)
            axs[1,0].spines['right'].set_visible(False)
            
            axs[0,1].hist(targetTime_press2, bins=100, rwidth= 0.5)
            axs[0,1].set_title('onset2_position to target2 histogram')
            axs[0,1].set_xlabel('Interval Time (s)')
            axs[0,1].set_ylabel('Count of Rewarded Trials')
            
            axs[0,1].spines['top'].set_visible(False)
            axs[0,1].spines['right'].set_visible(False)

            axs[1,1].plot(targetTime_press2) 
            axs[1,1].set_title('onset2_position to target2 for each trial')
            axs[1,1].set_xlabel('Trials')
            axs[1,1].set_ylabel('Interval Time (s)')
            
            axs[1,1].spines['top'].set_visible(False)
            axs[1,1].spines['right'].set_visible(False)
            
            axs[2,0].hist(target1_velocity_s, bins=100, rwidth= 0.5, color = 'blue',label = 'short')
            axs[2,0].hist(target1_velocity_l, bins=100, rwidth= 0.5, color = 'green',label = 'long')
            axs[2,0].set_title('onset1_velocity to target1 histogram')
            axs[2,0].set_xlabel('Interval Time (s)')
            axs[2,0].set_ylabel('Count of Rewarded Trials')
            axs[2,0].legend()
            
            axs[2,0].spines['top'].set_visible(False)
            axs[2,0].spines['right'].set_visible(False)
            
            axs[3,0].plot(target1_velocity_s, color = 'blue', label = 'short')        
            axs[3,0].plot(target1_velocity_l, color = 'green', label = 'long') 
            axs[3,0].set_title('onset1_velocity to target1 for each trial')
            axs[3,0].set_xlabel('Trials')
            axs[3,0].set_ylabel('Interval Time (s)')
            
            axs[3,0].spines['top'].set_visible(False)
            axs[3,0].spines['right'].set_visible(False)
            
            axs[2,1].hist(target2_velocity_s, bins=100, rwidth= 0.5, color = 'blue',label = 'short')
            axs[2,1].hist(target2_velocity_l, bins=100, rwidth= 0.5, color = 'green',label = 'long')
            axs[2,1].set_title('onset2_velocity to target2 histogram')
            axs[2,1].set_xlabel('Interval Time (s)')
            axs[2,1].set_ylabel('Count of Rewarded Trials')
            axs[2,1].legend()
            
            axs[2,1].spines['top'].set_visible(False)
            axs[2,1].spines['right'].set_visible(False)

            axs[3,1].plot(target2_velocity_s, color = 'blue', label = 'short')
            axs[3,1].plot(target2_velocity_l, color = 'green', label = 'long') 
            axs[3,1].set_title('onset2_velocity to target2 for each trial')
            axs[3,1].set_xlabel('Trials')
            axs[3,1].set_ylabel('Interval Time (s)')
            axs[3,1].legend()
            
            axs[3,1].spines['top'].set_visible(False)
            axs[3,1].spines['right'].set_visible(False)
                    
            axs[4,0].hist(amp_press1_s, bins=20, rwidth= 0.5, color = 'blue', label = 'short')
            axs[4,0].hist(amp_press1_l, bins=20, rwidth= 0.5, color = 'green', label = 'long')
            axs[4,0].set_title('amp difference baseline and peak1 histogram')
            axs[4,0].set_xlabel('Amplitude (deg)')
            axs[4,0].set_ylabel('Count of Rewarded Trials')
            axs[4,0].legend()
            
            axs[4,0].spines['top'].set_visible(False)
            axs[4,0].spines['right'].set_visible(False)

            axs[4,1].hist(amp_press2_s, bins=20, rwidth= 0.5, color = 'blue', label = 'short') 
            axs[4,1].hist(amp_press2_l, bins=20, rwidth= 0.5, color = 'green', label = 'long')
            axs[4,1].set_title('amp difference baseline and peak2 histogram')
            axs[4,1].set_xlabel('Amplitude (deg)')
            axs[4,1].set_ylabel('Count of Rewarded Trials')
            axs[4,1].legend()
            
            axs[4,1].spines['top'].set_visible(False)
            axs[4,1].spines['right'].set_visible(False)
        
            axs[5,0].plot(amp_press1_s, color = 'blue', label = 'short')
            axs[5,0].plot(amp_press1_l, color = 'green', label = 'long')
            axs[5,0].set_title('amp difference baseline and peak1 for each trial')
            axs[5,0].set_xlabel('Trials')
            axs[5,0].set_ylabel('Amplitude (deg)')
            axs[5,0].legend()
            
            axs[5,0].spines['top'].set_visible(False)
            axs[5,0].spines['right'].set_visible(False)
        
            axs[5,1].plot(amp_press2_s, color = 'blue', label = 'short') 
            axs[5,1].plot(amp_press2_s, color = 'green', label = 'long') 
            axs[5,1].set_title('amp difference baseline and peak2 for each trial')
            axs[5,1].set_xlabel('Trials')
            axs[5,1].set_ylabel('Amplitude (deg)')
            axs[5,1].legend()
            
            axs[5,1].spines['top'].set_visible(False)
            axs[5,1].spines['right'].set_visible(False)
            
            axs[6,0].hist(interval,bins=100, rwidth= 0.5)
            axs[6,0].set_title('Time Interval between onset1 & 2 Position histogram')
            axs[6,0].set_xlabel('time (s)')
            axs[6,0].set_ylabel('Count of Rewarded Trials')
            
            axs[6,0].spines['top'].set_visible(False)
            axs[6,0].spines['right'].set_visible(False)
            
            
            axs[7,0].plot(interval) 
            axs[7,0].set_title('Time Interval between onset1 & 2 Position')
            axs[7,0].set_xlabel('Trials')
            axs[7,0].set_ylabel('Time Interval (s)')
            
            axs[7,0].spines['top'].set_visible(False)
            axs[7,0].spines['right'].set_visible(False)
        
            axs[6,1].hist(interval_velocity_s,bins=100, rwidth= 0.5, color = 'blue',label = 'short')
            axs[6,1].hist(interval_velocity_l,bins=100, rwidth= 0.5, color = 'green',label = 'long')
            axs[6,1].set_title('Time Interval between onset1 & 2 Velocity histogram')
            axs[6,1].set_xlabel('time (s)')
            axs[6,1].set_ylabel('Count of Rewarded Trials')
            axs[6,1].legend()
            
            axs[6,1].spines['top'].set_visible(False)
            axs[6,1].spines['right'].set_visible(False)
            
            axs[7,1].plot(interval_velocity_s, color = 'blue',label = 'short')
            axs[7,1].plot(interval_velocity_l, color = 'green',label = 'long') 
            axs[7,1].set_title('Time Interval between onset1 & 2 Velocity')
            axs[7,1].set_xlabel('Trials')
            axs[7,1].set_ylabel('Time Interval (s)')
            axs[7,1].legend()
            
            axs[7,1].spines['top'].set_visible(False)
            axs[7,1].spines['right'].set_visible(False)
            
            fig.tight_layout()
            
            ################
            G_chemo_target1_velocity_s.append(np.mean(chemo_target1_velocity_s, axis = 0))
            G_chemo_target2_velocity_s.append(np.mean(chemo_target2_velocity_s, axis = 0))
            G_chemo_interval_velocity_s.append(np.mean(chemo_interval_velocity_s, axis = 0))
            G_chemo_PreVis2Delay_session_short.append(np.mean(chemo_PreVis2Delay_session_short, axis = 0))
            G_chemo_amp_press1_s.append(np.mean(chemo_amp_press1_s, axis = 0))
            G_chemo_amp_press2_s.append(np.mean(chemo_amp_press2_s, axis = 0))
            G_chemo_max_press1_vel_s.append(np.mean(chemo_max_press1_vel_s, axis = 0))
            G_chemo_max_press2_vel_s.append(np.mean(chemo_max_press2_vel_s, axis = 0))
            G_chemo_velaton1_s.append(np.mean(chemo_velaton1_s, axis = 0))
            G_chemo_velaton2_s.append(np.mean(chemo_velaton2_s, axis = 0))

            G_opto_target1_velocity_s.append(np.mean(opto_target1_velocity_s, axis = 0))
            G_opto_target2_velocity_s.append(np.mean(opto_target2_velocity_s, axis = 0))
            G_opto_interval_velocity_s.append(np.mean(opto_interval_velocity_s, axis = 0))
            G_opto_PreVis2Delay_session_short.append(np.mean(opto_PreVis2Delay_session_short, axis = 0))
            G_opto_amp_press1_s.append(np.mean(opto_amp_press1_s, axis = 0))
            G_opto_amp_press2_s.append(np.mean(opto_amp_press2_s, axis = 0))
            G_opto_max_press1_vel_s.append(np.mean(opto_max_press1_vel_s, axis = 0))
            G_opto_max_press2_vel_s.append(np.mean(opto_max_press2_vel_s, axis = 0))
            G_opto_velaton1_s.append(np.mean(opto_velaton1_s, axis = 0))
            G_opto_velaton2_s.append(np.mean(opto_velaton2_s, axis = 0))

            G_con_target1_velocity_s.append(np.mean(con_target1_velocity_s, axis = 0))
            G_con_target2_velocity_s.append(np.mean(con_target2_velocity_s, axis = 0))
            G_con_interval_velocity_s.append(np.mean(con_interval_velocity_s, axis = 0))
            G_con_PreVis2Delay_session_short.append(np.mean(con_PreVis2Delay_session_short, axis = 0))
            G_con_amp_press1_s.append(np.mean(con_amp_press1_s, axis = 0))
            G_con_amp_press2_s.append(np.mean(con_amp_press2_s, axis = 0))
            G_con_max_press1_vel_s.append(np.mean(con_max_press1_vel_s, axis = 0))
            G_con_max_press2_vel_s.append(np.mean(con_max_press2_vel_s, axis = 0))
            G_con_velaton1_s.append(np.mean(con_velaton1_s, axis = 0))
            G_con_velaton2_s.append(np.mean(con_velaton2_s, axis = 0))
            #### Long ####
            G_chemo_target1_velocity_l.append(np.mean(chemo_target1_velocity_l, axis = 0))
            G_chemo_target2_velocity_l.append(np.mean(chemo_target2_velocity_l, axis = 0))
            G_chemo_interval_velocity_l.append(np.mean(chemo_interval_velocity_l, axis = 0))
            G_chemo_PreVis2Delay_session_long.append(np.mean(chemo_PreVis2Delay_session_long, axis = 0))
            G_chemo_amp_press1_l.append(np.mean(chemo_amp_press1_l, axis = 0))
            G_chemo_amp_press2_l.append(np.mean(chemo_amp_press2_l, axis = 0))
            G_chemo_max_press1_vel_l.append(np.mean(chemo_max_press1_vel_l, axis = 0))
            G_chemo_max_press2_vel_l.append(np.mean(chemo_max_press2_vel_l, axis = 0))
            G_chemo_velaton1_l.append(np.mean(chemo_velaton1_l, axis = 0))
            G_chemo_velaton2_l.append(np.mean(chemo_velaton2_l, axis = 0))

            G_opto_target1_velocity_l.append(np.mean(opto_target1_velocity_l, axis = 0))
            G_opto_target2_velocity_l.append(np.mean(opto_target2_velocity_l, axis = 0))
            G_opto_interval_velocity_l.append(np.mean(opto_interval_velocity_l, axis = 0))
            G_opto_PreVis2Delay_session_long.append(np.mean(opto_PreVis2Delay_session_long, axis = 0))
            G_opto_amp_press1_l.append(np.mean(opto_amp_press1_l, axis = 0))
            G_opto_amp_press2_l.append(np.mean(opto_amp_press2_l, axis = 0))
            G_opto_max_press1_vel_l.append(np.mean(opto_max_press1_vel_l, axis = 0))
            G_opto_max_press2_vel_l.append(np.mean(opto_max_press2_vel_l, axis = 0))
            G_opto_velaton1_l.append(np.mean(opto_velaton1_l, axis = 0))
            G_opto_velaton2_l.append(np.mean(opto_velaton2_l, axis = 0))

            G_con_target1_velocity_l.append(np.mean(con_target1_velocity_l, axis = 0))
            G_con_target2_velocity_l.append(np.mean(con_target2_velocity_l, axis = 0))
            G_con_interval_velocity_l.append(np.mean(con_interval_velocity_l, axis = 0))
            G_con_PreVis2Delay_session_long.append(np.mean(con_PreVis2Delay_session_long, axis = 0))
            G_con_amp_press1_l.append(np.mean(con_amp_press1_l, axis = 0))
            G_con_amp_press2_l.append(np.mean(con_amp_press2_l, axis = 0))
            G_con_max_press1_vel_l.append(np.mean(con_max_press1_vel_l, axis = 0))
            G_con_max_press2_vel_l.append(np.mean(con_max_press2_vel_l, axis = 0))
            G_con_velaton1_l.append(np.mean(con_velaton1_l, axis = 0))
            G_con_velaton2_l.append(np.mean(con_velaton2_l, axis = 0))
            
            ################
            
            std_error_1 = np.std(targetTime_press1, ddof=1) / np.sqrt(len(targetTime_press1))
            std_targetTime_press1.append(std_error_1)
            mean_targetTime_press1.append(np.mean(targetTime_press1))
            
            std_error_2 = np.std(targetTime_press2_reward, ddof=1) / np.sqrt(len(targetTime_press2_reward))
            std_targetTime_press2_reward.append(std_error_2)
            mean_targetTime_press2_reward.append(np.mean(targetTime_press2_reward))
            
            std_error_13 = np.std(targetTime_press2_early, ddof=1) / np.sqrt(len(targetTime_press2_early))
            std_targetTime_press2_early.append(std_error_13)
            mean_targetTime_press2_early.append(np.mean(targetTime_press2_early))
            
            std_error_3 = np.std(amp_press1_s, ddof=1) / np.sqrt(len(amp_press1_s))
            std_amp_press1_s.append(std_error_3)
            mean_amp_press1_s.append(np.mean(amp_press1_s))
            
            std_error_3 = np.std(amp_press1_l, ddof=1) / np.sqrt(len(amp_press1_l))
            std_amp_press1_l.append(std_error_3)
            mean_amp_press1_l.append(np.mean(amp_press1_l))
            
            std_error_4 = np.std(amp_press2_reward_s, ddof=1) / np.sqrt(len(amp_press2_reward_s))
            std_amp_press2_reward_s.append(std_error_4)
            mean_amp_press2_reward_s.append(np.mean(amp_press2_reward_s))
            
            std_error_4 = np.std(amp_press2_reward_l, ddof=1) / np.sqrt(len(amp_press2_reward_l))
            std_amp_press2_reward_l.append(std_error_4)
            mean_amp_press2_reward_l.append(np.mean(amp_press2_reward_l))
            
            std_error_11 = np.std(amp_press2_early_s, ddof=1) / np.sqrt(len(amp_press2_early_s))
            std_amp_press2_early_s.append(std_error_11)
            mean_amp_press2_early_s.append(np.mean(amp_press2_early_s))
            
            std_error_11 = np.std(amp_press2_early_l, ddof=1) / np.sqrt(len(amp_press2_early_l))
            std_amp_press2_early_l.append(std_error_11)
            mean_amp_press2_early_l.append(np.mean(amp_press2_early_l))
            
            std_error_5 = np.std(interval_reward, ddof=1) / np.sqrt(len(interval_reward))
            std_interval_reward.append(std_error_5)
            mean_interval_reward.append(np.mean(interval_reward))
            
            std_error_10 = np.std(interval_early, ddof=1) / np.sqrt(len(interval_early))
            std_interval_early.append(std_error_10)
            mean_interval_early.append(np.mean(interval_early))
            
            std_error_6  = np.std(target1_velocity_s, ddof=1) / np.sqrt(len(target1_velocity_s))
            std_target1_velocity_s.append(std_error_6)
            mean_target1_velocity_s.append(np.mean(target1_velocity_s))
            
            std_error_7  = np.std(target2_velocity_reward_s, ddof=1) / np.sqrt(len(target2_velocity_reward_s))
            std_target2_velocity_reward_s.append(std_error_7)
            mean_target2_velocity_reward_s.append(np.mean(target2_velocity_reward_s))
            
            std_error_12  = np.std(target2_velocity_early_s, ddof=1) / np.sqrt(len(target2_velocity_early_s))
            std_target2_velocity_early_s.append(std_error_12)
            mean_target2_velocity_early_s.append(np.mean(target2_velocity_early_s))
            
            std_error_8  = np.std(interval_velocity_reward_s, ddof=1) / np.sqrt(len(interval_velocity_reward_s))
            std_interval_velocity_reward_s.append(std_error_8)
            mean_interval_velocity_reward_s.append(np.mean(interval_velocity_reward_s))
            
            std_error_9  = np.std(interval_velocity_early_s, ddof=1) / np.sqrt(len(interval_velocity_early_s))
            std_interval_velocity_early_s.append(std_error_9)
            mean_interval_velocity_early_s.append(np.mean(interval_velocity_early_s))
            ###############################################################
            std_error_14  = np.std(target1_velocity_l, ddof=1) / np.sqrt(len(target1_velocity_l))
            std_target1_velocity_l.append(std_error_14)
            mean_target1_velocity_l.append(np.mean(target1_velocity_l))
            
            std_error_15  = np.std(target2_velocity_reward_l, ddof=1) / np.sqrt(len(target2_velocity_reward_l))
            std_target2_velocity_reward_l.append(std_error_15)
            mean_target2_velocity_reward_l.append(np.mean(target2_velocity_reward_l))
            
            std_error_16  = np.std(target2_velocity_early_l, ddof=1) / np.sqrt(len(target2_velocity_early_l))
            std_target2_velocity_early_l.append(std_error_16)
            mean_target2_velocity_early_l.append(np.mean(target2_velocity_early_l))
            
            std_error_17  = np.std(interval_velocity_reward_l, ddof=1) / np.sqrt(len(interval_velocity_reward_l))
            std_interval_velocity_reward_l.append(std_error_17)
            mean_interval_velocity_reward_l.append(np.mean(interval_velocity_reward_l))
            
            std_error_18  = np.std(interval_velocity_early_l, ddof=1) / np.sqrt(len(interval_velocity_early_l))
            std_interval_velocity_early_l.append(std_error_18)
            mean_interval_velocity_early_l.append(np.mean(interval_velocity_early_l))
            ###################################################################
            std_error = np.std(PreVis2Delay_session_short, ddof=1) / np.sqrt(len(PreVis2Delay_session_short))
            std_PreVis2Delay_session_short.append(std_error)
            mean_PreVis2Delay_session_short.append(np.mean(PreVis2Delay_session_short))
            
            std_error = np.std(PreVis2Delay_session_long, ddof=1) / np.sqrt(len(PreVis2Delay_session_long))
            std_PreVis2Delay_session_long.append(std_error)
            mean_PreVis2Delay_session_long.append(np.mean(PreVis2Delay_session_long))
            ##############################################################################
            std_error = np.std(max_press1_vel_s, ddof=1) / np.sqrt(len(max_press1_vel_s))
            std_max_press1_vel_s.append(std_error)
            mean_max_press1_vel_s.append(np.mean(max_press1_vel_s))
            
            std_error = np.std(max_press1_vel_l, ddof=1) / np.sqrt(len(max_press1_vel_l))
            std_max_press1_vel_l.append(std_error)
            mean_max_press1_vel_l.append(np.mean(max_press1_vel_l))
            
            std_error = np.std(max_press2_vel_early_s, ddof=1) / np.sqrt(len(max_press2_vel_early_s))
            std_max_press2_vel_early_s.append(std_error)
            mean_max_press2_vel_early_s.append(np.mean(max_press2_vel_early_s))
            
            std_error = np.std(max_press2_vel_early_l, ddof=1) / np.sqrt(len(max_press2_vel_early_l))
            std_max_press2_vel_early_l.append(std_error)
            mean_max_press2_vel_early_l.append(np.mean(max_press2_vel_early_l))
            
            std_error = np.std(max_press2_vel_reward_s, ddof=1) / np.sqrt(len(max_press2_vel_reward_s))
            std_max_press2_vel_reward_s.append(std_error)
            mean_max_press2_vel_reward_s.append(np.mean(max_press2_vel_reward_s))
            
            std_error = np.std(max_press2_vel_reward_l, ddof=1) / np.sqrt(len(max_press2_vel_reward_l))
            std_max_press2_vel_reward_l.append(std_error)
            mean_max_press2_vel_reward_l.append(np.mean(max_press2_vel_reward_l))
            #############################################################################
            std_error = np.std(velaton1_s, ddof=1) / np.sqrt(len(velaton1_s))
            std_velaton1_s.append(std_error)
            mean_velaton1_s.append(np.mean(velaton1_s))
            
            std_error = np.std(velaton1_l, ddof=1) / np.sqrt(len(velaton1_l))
            std_velaton1_l.append(std_error)
            mean_velaton1_l.append(np.mean(velaton1_l))
            
            std_error = np.std(velaton2_early_s, ddof=1) / np.sqrt(len(velaton2_early_s))
            std_velaton2_early_s.append(std_error)
            mean_velaton2_early_s.append(np.mean(velaton2_early_s))
            
            std_error = np.std(velaton2_early_l, ddof=1) / np.sqrt(len(velaton2_early_l))
            std_velaton2_early_l.append(std_error)
            mean_velaton2_early_l.append(np.mean(velaton2_early_l))
            
            std_error = np.std(velaton2_reward_s, ddof=1) / np.sqrt(len(velaton2_reward_s))
            std_velaton2_reward_s.append(std_error)
            mean_velaton2_reward_s.append(np.mean(velaton2_reward_s))
            
            std_error = np.std(velaton2_reward_l, ddof=1) / np.sqrt(len(velaton2_reward_l))
            std_velaton2_reward_l.append(std_error)
            mean_velaton2_reward_l.append(np.mean(velaton2_reward_l))
            
            output_figs_dir = output_dir_onedrive + subject + '/'    
            output_imgs_dir = output_dir_local + subject + '/time analysis_imgs/' 
            output_data_dir  = output_dir_local + subject + 'data of onset'
            os.makedirs(output_figs_dir, exist_ok = True)
            os.makedirs(output_imgs_dir, exist_ok = True)
            fig.savefig(output_figs_dir + subject + '_' + session_date + '_timing_analysis_Amp'+'.pdf', dpi=300)
            # fig.savefig(output_imgs_dir + subject + '_time_analysis_'+ session_date +'.png', dpi=300)
            plt.close(fig)

        # Plotting for anlyze over all sessions
        axs1[2,0].errorbar(numeric_dates, mean_amp_press1_s, yerr=std_amp_press1_s, fmt='o', capsize=4, color = 'blue', label = 'Rew & Early2_S')
        axs1[2,0].errorbar(numeric_dates + offset, mean_amp_press1_l, yerr=std_amp_press1_l, fmt='o', capsize=4, color = 'green', label = 'Rew & Early2_L')
        axs1[2,0].set_title('baseline to peak press1')
        axs1[2,0].set_ylabel('Mean Press deflection (deg) +/- SEM')
        axs1[2,0].set_xlabel('Sessions')
        axs1[2,0].set_xticks(numeric_dates)
        axs1[2,0].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[2,0].spines['top'].set_visible(False)
        axs1[2,0].spines['right'].set_visible(False)
        axs1[2,0].legend()

        axs1[2,1].errorbar(numeric_dates, mean_amp_press2_reward_s, yerr=std_amp_press2_reward_s, fmt='o', capsize=4, color= 'blue', label = 'Rewarded_S')
        axs1[2,1].errorbar(numeric_dates + offset, mean_amp_press2_reward_l, yerr=std_amp_press2_reward_l, fmt='o', capsize=4, color= 'green', label = 'Rewarded_L')
        axs1[2,1].errorbar(numeric_dates + 2*offset, mean_amp_press2_early_s, yerr=std_amp_press2_early_s, fmt='_', capsize=4, color= 'blue', label = 'Early_Press_S')
        axs1[2,1].errorbar(numeric_dates + 3*offset, mean_amp_press2_early_l, yerr=std_amp_press2_early_l, fmt='_', capsize=4, color= 'green', label = 'Early_Press_L')
        axs1[2,1].set_title('baseline to peak press2')
        axs1[2,1].set_ylabel('Mean time Interval (s) +/- SEM')
        axs1[2,1].set_xlabel('Sessions')
        axs1[2,1].set_xticks(numeric_dates)
        axs1[2,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[2,1].legend()
        axs1[2,1].spines['top'].set_visible(False)
        axs1[2,1].spines['right'].set_visible(False)
        
        axs1[0,0].errorbar(numeric_dates, mean_target1_velocity_s, yerr=std_target1_velocity_s, fmt='o', capsize=4, color = 'blue', label = 'Rew & Early2_S')
        axs1[0,0].errorbar(numeric_dates + offset, mean_target1_velocity_l, yerr=std_target1_velocity_l, fmt='o', capsize=4, color = 'green', label = 'Rew & Early2_L')
        axs1[0,0].set_title('onset1 to target1')
        axs1[0,0].set_ylabel('Mean time Interval (s) +/- SEM')
        axs1[0,0].set_xlabel('Sessions')
        axs1[0,0].set_xticks(numeric_dates)
        axs1[0,0].set_xticklabels(dates, rotation=90, ha = 'center') 
        axs1[0,0].legend()
        axs1[0,0].spines['top'].set_visible(False)
        axs1[0,0].spines['right'].set_visible(False)

        axs1[0,1].errorbar(numeric_dates, mean_target2_velocity_reward_s, yerr=std_target2_velocity_reward_s, fmt='o', capsize=4, color= 'blue', label = 'Rewarded_S')
        axs1[0,1].errorbar(numeric_dates + offset, mean_target2_velocity_reward_l, yerr=std_target2_velocity_reward_l, fmt='o', capsize=4, color= 'green', label = 'Rewarded_L')
        axs1[0,1].errorbar(numeric_dates + 2*offset, mean_target2_velocity_early_s, yerr=std_target2_velocity_early_s, fmt='_', capsize=4, color= 'blue', label = 'Early_Press_S')
        axs1[0,1].errorbar(numeric_dates + 3*offset, mean_target2_velocity_early_l, yerr=std_target2_velocity_early_l, fmt='_', capsize=4, color= 'green', label = 'Early_Press_L')
        axs1[0,1].set_title('onset2 to target2')
        axs1[0,1].set_ylabel('Mean time Interval (s) +/- SEM ')
        axs1[0,1].set_xlabel('Sessions')
        axs1[0,1].set_xticks(numeric_dates)
        axs1[0,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[0,1].legend()
        axs1[0,1].spines['top'].set_visible(False)
        axs1[0,1].spines['right'].set_visible(False)
        
        axs1[1,0].errorbar(numeric_dates, mean_interval_velocity_reward_s, yerr=std_interval_velocity_reward_s, fmt='o', capsize=4, color= 'blue', label = 'Rewarded_S')
        axs1[1,0].errorbar(numeric_dates + offset, mean_interval_velocity_reward_l, yerr=std_interval_velocity_reward_l, fmt='o', capsize=4, color= 'green', label = 'Rewarded_L')
        axs1[1,0].errorbar(numeric_dates + 2*offset, mean_interval_velocity_early_s, yerr=std_interval_velocity_early_s, fmt='_', capsize=4, color= 'blue', label = 'Early_Press_S')
        axs1[1,0].errorbar(numeric_dates + 3*offset, mean_interval_velocity_early_l, yerr=std_interval_velocity_early_l, fmt='_', capsize=4, color= 'green', label = 'Early_Press_L')
        axs1[1,0].set_title('Interval time between onset1 & 2')
        axs1[1,0].set_ylabel('Mean Interval (s) +/- SEM')
        axs1[1,0].set_xlabel('Sessions')
        axs1[1,0].set_xticks(numeric_dates)
        axs1[1,0].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[1,0].legend()
        axs1[1,0].spines['top'].set_visible(False)
        axs1[1,0].spines['right'].set_visible(False)
        
        axs1[1,1].errorbar(numeric_dates, mean_PreVis2Delay_session_short, yerr=std_PreVis2Delay_session_short, fmt='o', capsize=4, color = 'blue', label = 'short')
        axs1[1,1].errorbar(numeric_dates + offset, mean_PreVis2Delay_session_long, yerr=std_PreVis2Delay_session_long, fmt='o', capsize=4, color = 'green', label = 'long')
        axs1[1,1].set_title('PrePress2Delay')
        axs1[1,1].set_ylabel('Mean preVis2Delay (s) +/- SEM')
        axs1[1,1].set_xlabel('Sessions')
        axs1[1,1].set_xticks(numeric_dates)
        axs1[1,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[1,1].legend()
        axs1[1,1].spines['top'].set_visible(False)
        axs1[1,1].spines['right'].set_visible(False)
        
        axs1[3,0].errorbar(numeric_dates, mean_max_press1_vel_s, yerr=std_max_press1_vel_s, fmt='o', capsize=4, color = 'blue', label = 'Rew & Early2_S')
        axs1[3,0].errorbar(numeric_dates + offset, mean_max_press1_vel_l, yerr=std_max_press1_vel_l, fmt='o', capsize=4, color = 'green', label = 'Rew & Early2_L')
        axs1[3,0].set_title('peak velocity at press1')
        axs1[3,0].set_ylabel('Mean velocity (deg/s) +/- SEM ')
        axs1[3,0].set_xlabel('Sessions')
        axs1[3,0].set_xticks(numeric_dates)
        axs1[3,0].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[3,0].legend()
        axs1[3,0].spines['top'].set_visible(False)
        axs1[3,0].spines['right'].set_visible(False)

        
        axs1[3,1].errorbar(numeric_dates, mean_max_press2_vel_reward_s, yerr=std_max_press2_vel_reward_s, fmt='o', capsize=4, color= 'blue', label = 'Rewarded_S')
        axs1[3,1].errorbar(numeric_dates + offset, mean_max_press2_vel_reward_l, yerr=std_max_press2_vel_reward_l, fmt='o', capsize=4, color= 'green', label = 'Rewarded_L')
        axs1[3,1].errorbar(numeric_dates + 2*offset, mean_max_press2_vel_early_s, yerr=std_max_press2_vel_early_s, fmt='_', capsize=4, color= 'blue', label = 'Early_Press2_S')
        axs1[3,1].errorbar(numeric_dates + 3*offset, mean_max_press2_vel_early_l, yerr=std_max_press2_vel_early_l, fmt='_', capsize=4, color= 'green', label = 'Early_Press2_L')
        axs1[3,1].set_title('peak velocity at press2')
        axs1[3,1].set_ylabel('Mean velocity (deg/s) +/- SEM ')
        axs1[3,1].set_xlabel('Sessions')
        axs1[3,1].set_xticks(numeric_dates)
        axs1[3,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[3,1].legend()
        axs1[3,1].spines['top'].set_visible(False)
        axs1[3,1].spines['right'].set_visible(False)

        axs1[4,0].errorbar(numeric_dates, mean_velaton1_s, yerr=std_velaton1_s, fmt='o', capsize=4, color = 'blue', label = 'Rew & Early2_S')
        axs1[4,0].errorbar(numeric_dates + offset, mean_velaton1_l, yerr=std_velaton1_l, fmt='o', capsize=4, color = 'green', label = 'Rew & Early2_L')
        axs1[4,0].set_title('velocity at onset1')
        axs1[4,0].set_ylabel('Mean velocity (deg/s) +/- SEM ')
        axs1[4,0].set_xlabel('Sessions')
        axs1[4,0].set_xticks(numeric_dates)
        axs1[4,0].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[4,0].legend()
        axs1[4,0].spines['top'].set_visible(False)
        axs1[4,0].spines['right'].set_visible(False)

        
        axs1[4,1].errorbar(numeric_dates, mean_velaton2_reward_s, yerr=std_velaton2_reward_s, fmt='o', capsize=4, color= 'blue', label = 'Rewarded_S')
        axs1[4,1].errorbar(numeric_dates + offset, mean_velaton2_reward_l, yerr=std_velaton2_reward_l, fmt='o', capsize=4, color= 'green', label = 'Rewarded_L')
        axs1[4,1].errorbar(numeric_dates + 2*offset, mean_velaton2_early_s, yerr=std_velaton2_early_s, fmt='_', capsize=4, color= 'blue', label = 'Early_Press2_S')
        axs1[4,1].errorbar(numeric_dates + 3*offset, mean_velaton2_early_l, yerr=std_velaton2_early_l, fmt='_', capsize=4, color= 'green', label = 'Early_Press2_L')
        axs1[4,1].set_title('velocity at onset2')
        axs1[4,1].set_ylabel('Mean velocity (deg/s) +/- SEM ')
        axs1[4,1].set_xlabel('Sessions')
        axs1[4,1].set_xticks(numeric_dates)
        axs1[4,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[4,1].legend()
        axs1[4,1].spines['top'].set_visible(False)
        axs1[4,1].spines['right'].set_visible(False)    
        
        fig1.tight_layout()
        
        # os.makedirs(output_figs_dir, exist_ok = True)
        # os.makedirs(output_imgs_dir, exist_ok = True)
        # fig1.savefig(output_figs_dir + subject + '_time_analysis_oversessions.pdf', dpi=300)
        # fig1.savefig(output_imgs_dir + subject + '_time_analysis_imgs_oversessions.png', dpi=300)
        # plt.close(fig1)
        
        
    ########################################### PLOTS for POOLED sessions ###############################################################
    print('##########################')
    print('#########POOLED###########')
    print('##########################')

    fig2, axs2 = plt.subplots(nrows=3, figsize=(20, 10))   # should be changed
    fig2.suptitle(subject + ' - ' + session_date + ' \n ' + 'Timing analysis for Rewarded trials (POOLED SESSIONS)' + '\n')

    offset = 0.1

    x01 = 0
    x02 = 1
    x03 = 2

    axs2[0].errorbar(x01 - 2.5*offset, np.mean(P_chemo_target1_velocity_s, axis=0), np.std(P_chemo_target1_velocity_s,axis=0), color = 'r',fmt='_', capsize=4, label = 'Chemo_Short')
    axs2[0].errorbar(x01 - 1.5*offset, np.mean(P_chemo_target1_velocity_l, axis=0), np.std(P_chemo_target1_velocity_l,axis=0), color = 'r',fmt='o', capsize=4, label = 'Chemo_Long')

    axs2[0].errorbar(x01 - 0.5*offset, np.mean(P_con_target1_velocity_s, axis=0), np.std(P_con_target1_velocity_s,axis=0), color = 'k',fmt='_', capsize=4, label = 'Control_Short')
    axs2[0].errorbar(x01 + 0.5*offset, np.mean(P_con_target1_velocity_l, axis=0), np.std(P_con_target1_velocity_l,axis=0), color = 'k',fmt='o', capsize=4, label = 'Control_Long')

    axs2[0].errorbar(x01 + 1.5*offset, np.mean(P_opto_target1_velocity_s, axis=0), np.std(P_opto_target1_velocity_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4, label = 'Opto_Short')
    axs2[0].errorbar(x01 + 2.5*offset, np.mean(P_opto_target1_velocity_l, axis=0), np.std(P_opto_target1_velocity_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4, label = 'Opto_Long')

    axs2[0].errorbar(x02 - 2.5*offset, np.mean(P_chemo_target2_velocity_s, axis=0), np.std(P_chemo_target2_velocity_s,axis=0), color = 'r',fmt='_', capsize=4)
    axs2[0].errorbar(x02 - 1.5*offset, np.mean(P_chemo_target2_velocity_l, axis=0), np.std(P_chemo_target2_velocity_l,axis=0), color = 'r',fmt='o', capsize=4)

    axs2[0].errorbar(x02 - 0.5*offset, np.mean(P_con_target2_velocity_s, axis=0), np.std(P_con_target2_velocity_s,axis=0), color = 'k',fmt='_', capsize=4)
    axs2[0].errorbar(x02 + 0.5*offset, np.mean(P_con_target2_velocity_l, axis=0), np.std(P_con_target2_velocity_l,axis=0), color = 'k',fmt='o', capsize=4)

    axs2[0].errorbar(x02 + 1.5*offset, np.mean(P_opto_target2_velocity_s, axis=0), np.std(P_opto_target2_velocity_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4)
    axs2[0].errorbar(x02 + 2.5*offset, np.mean(P_opto_target2_velocity_l, axis=0), np.std(P_opto_target2_velocity_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4)

    axs2[0].errorbar(x03 - 2.5*offset, np.mean(P_chemo_interval_velocity_s, axis=0), np.std(P_chemo_interval_velocity_s,axis=0), color = 'r',fmt='_', capsize=4)
    axs2[0].errorbar(x03 - 1.5*offset, np.mean(P_chemo_interval_velocity_l, axis=0), np.std(P_chemo_interval_velocity_l,axis=0), color = 'r',fmt='o', capsize=4)

    axs2[0].errorbar(x03 - 0.5*offset, np.mean(P_con_interval_velocity_s, axis=0), np.std(P_con_interval_velocity_s,axis=0), color = 'k',fmt='_', capsize=4)
    axs2[0].errorbar(x03 + 0.5*offset, np.mean(P_con_interval_velocity_l, axis=0), np.std(P_con_interval_velocity_l,axis=0), color = 'k',fmt='o', capsize=4)

    axs2[0].errorbar(x03 + 1.5*offset, np.mean(P_opto_interval_velocity_s, axis=0), np.std(P_opto_interval_velocity_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4)
    axs2[0].errorbar(x03 + 2.5*offset, np.mean(P_opto_interval_velocity_l, axis=0), np.std(P_opto_interval_velocity_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4)

    # Set x-ticks to show the original category labels
    axs2[0].set_xticks([x01, x02, x03])
    axs2[0].set_xticklabels(['onset1_target1', 'onset2_target2', 'onset1_onset2_interval'])

    axs2[0].spines['right'].set_visible(False)
    axs2[0].spines['top'].set_visible(False)
    axs2[0].set_title('\n'+ subject)
    axs2[0].set_ylabel('Mean onset time (s) +/- SEM')
    axs2[0].legend(loc='best')

    x11 = 0
    x12 = 1

    axs2[1].errorbar(x11 - 2.5*offset, np.mean(P_chemo_amp_press1_s, axis=0), np.std(P_chemo_amp_press1_s,axis=0), color = 'r',fmt='_', capsize=4, label = 'Chemo_Short')
    axs2[1].errorbar(x11 - 1.5*offset, np.mean(P_chemo_amp_press1_l, axis=0), np.std(P_chemo_amp_press1_l,axis=0), color = 'r',fmt='o', capsize=4, label = 'Chemo_Long')

    axs2[1].errorbar(x11 - 0.5*offset, np.mean(P_con_amp_press1_s, axis=0), np.std(P_con_amp_press1_s,axis=0), color = 'k',fmt='_', capsize=4, label = 'Control_Short')
    axs2[1].errorbar(x11 + 0.5*offset, np.mean(P_con_amp_press1_l, axis=0), np.std(P_con_amp_press1_l,axis=0), color = 'k',fmt='o', capsize=4, label = 'Control_Long')

    axs2[1].errorbar(x11 + 1.5*offset, np.mean(P_opto_amp_press1_s, axis=0), np.std(P_opto_amp_press1_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4, label = 'Opto_Short')
    axs2[1].errorbar(x11 + 2.5*offset, np.mean(P_opto_amp_press1_l, axis=0), np.std(P_opto_amp_press1_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4, label = 'Opto_Long')

    axs2[1].errorbar(x12 - 2.5*offset, np.mean(P_chemo_amp_press2_s, axis=0), np.std(P_chemo_amp_press2_s,axis=0), color = 'r',fmt='_', capsize=4)
    axs2[1].errorbar(x12 - 1.5*offset, np.mean(P_chemo_amp_press2_l, axis=0), np.std(P_chemo_amp_press2_l,axis=0), color = 'r',fmt='o', capsize=4)

    axs2[1].errorbar(x12 - 0.5*offset, np.mean(P_con_amp_press2_s, axis=0), np.std(P_con_amp_press2_s,axis=0), color = 'k',fmt='_', capsize=4)
    axs2[1].errorbar(x12 + 0.5*offset, np.mean(P_con_amp_press2_l, axis=0), np.std(P_con_amp_press2_l,axis=0), color = 'k',fmt='o', capsize=4)

    axs2[1].errorbar(x12 + 1.5*offset, np.mean(P_opto_amp_press2_s, axis=0), np.std(P_opto_amp_press2_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4)
    axs2[1].errorbar(x12 + 2.5*offset, np.mean(P_opto_amp_press2_l, axis=0), np.std(P_opto_amp_press2_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4)

    # Set x-ticks to show the original category labels
    axs2[1].set_xticks([x11, x12])
    axs2[1].set_xticklabels(['baseline_peak1', 'baseline_peak2'])

    axs2[1].spines['right'].set_visible(False)
    axs2[1].spines['top'].set_visible(False)
    axs2[1].set_title('\n'+ 'Baseline to onset of presses')
    axs2[1].set_ylabel('Mean amplitude (deg) +/- SEM')
    axs2[1].legend(loc='best')

    x21 = 0
    x22 = 1
    x23 = 2
    x24 = 3

    axs2[2].errorbar(x21 - 2.5*offset, np.mean(P_chemo_max_press1_vel_s, axis=0), np.std(P_chemo_max_press1_vel_s,axis=0), color = 'r',fmt='_', capsize=4, label = 'Chemo_Short')
    axs2[2].errorbar(x21 - 1.5*offset, np.mean(P_chemo_max_press1_vel_l, axis=0), np.std(P_chemo_max_press1_vel_l,axis=0), color = 'r',fmt='o', capsize=4, label = 'Chemo_Long')

    axs2[2].errorbar(x21 - 0.5*offset, np.mean(P_con_max_press1_vel_s, axis=0), np.std(P_con_max_press1_vel_s,axis=0), color = 'k',fmt='_', capsize=4, label = 'Control_Short')
    axs2[2].errorbar(x21 + 0.5*offset, np.mean(P_con_max_press1_vel_l, axis=0), np.std(P_con_max_press1_vel_l,axis=0), color = 'k',fmt='o', capsize=4, label = 'Control_Long')

    axs2[2].errorbar(x21 + 1.5*offset, np.mean(P_opto_max_press1_vel_s, axis=0), np.std(P_opto_max_press1_vel_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4, label = 'Opto_Short')
    axs2[2].errorbar(x21 + 2.5*offset, np.mean(P_opto_max_press1_vel_l, axis=0), np.std(P_opto_max_press1_vel_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4, label = 'Opto_Long')

    axs2[2].errorbar(x22 - 2.5*offset, np.mean(P_chemo_max_press2_vel_s, axis=0), np.std(P_chemo_max_press2_vel_s,axis=0), color = 'r',fmt='_', capsize=4)
    axs2[2].errorbar(x22 - 1.5*offset, np.mean(P_chemo_max_press2_vel_l, axis=0), np.std(P_chemo_max_press2_vel_l,axis=0), color = 'r',fmt='o', capsize=4)

    axs2[2].errorbar(x22 - 0.5*offset, np.mean(P_con_max_press2_vel_s, axis=0), np.std(P_con_max_press2_vel_s,axis=0), color = 'k',fmt='_', capsize=4)
    axs2[2].errorbar(x22 + 0.5*offset, np.mean(P_con_max_press2_vel_l, axis=0), np.std(P_con_max_press2_vel_l,axis=0), color = 'k',fmt='o', capsize=4)

    axs2[2].errorbar(x22 + 1.5*offset, np.mean(P_opto_max_press2_vel_s, axis=0), np.std(P_opto_max_press2_vel_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4)
    axs2[2].errorbar(x22 + 2.5*offset, np.mean(P_opto_max_press2_vel_l, axis=0), np.std(P_opto_max_press2_vel_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4)

    axs2[2].errorbar(x23 - 2.5*offset, np.mean(P_chemo_velaton1_s, axis=0), np.std(P_chemo_velaton1_s,axis=0), color = 'r',fmt='_', capsize=4)
    axs2[2].errorbar(x23 - 1.5*offset, np.mean(P_chemo_velaton1_l, axis=0), np.std(P_chemo_velaton1_l,axis=0), color = 'r',fmt='o', capsize=4)

    axs2[2].errorbar(x23 - 0.5*offset, np.mean(P_con_velaton1_s, axis=0), np.std(P_con_velaton1_s,axis=0), color = 'k',fmt='_', capsize=4)
    axs2[2].errorbar(x23 + 0.5*offset, np.mean(P_con_velaton1_l, axis=0), np.std(P_con_velaton1_l,axis=0), color = 'k',fmt='o', capsize=4)

    axs2[2].errorbar(x23 + 1.5*offset, np.mean(P_opto_velaton1_s, axis=0), np.std(P_opto_velaton1_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4)
    axs2[2].errorbar(x23 + 2.5*offset, np.mean(P_opto_velaton1_l, axis=0), np.std(P_opto_velaton1_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4)

    axs2[2].errorbar(x24 - 2.5*offset, np.mean(P_chemo_velaton2_s, axis=0), np.std(P_chemo_velaton2_s,axis=0), color = 'r',fmt='_', capsize=4)
    axs2[2].errorbar(x24 - 1.5*offset, np.mean(P_chemo_velaton2_l, axis=0), np.std(P_chemo_velaton2_l,axis=0), color = 'r',fmt='o', capsize=4)

    axs2[2].errorbar(x24 - 0.5*offset, np.mean(P_con_velaton2_s, axis=0), np.std(P_con_velaton2_s,axis=0), color = 'k',fmt='_', capsize=4)
    axs2[2].errorbar(x24 + 0.5*offset, np.mean(P_con_velaton2_l, axis=0), np.std(P_con_velaton2_l,axis=0), color = 'k',fmt='o', capsize=4)

    axs2[2].errorbar(x24 + 1.5*offset, np.mean(P_opto_velaton2_s, axis=0), np.std(P_opto_velaton2_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4)
    axs2[2].errorbar(x24 + 2.5*offset, np.mean(P_opto_velaton2_l, axis=0), np.std(P_opto_velaton2_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4)

    # Set x-ticks to show the original category labels
    axs2[2].set_xticks([x21, x22, x23, x24])
    axs2[2].set_xticklabels(['peak_velocity_press1', 'peak_velocity_press2', 'velocity at press1', 'velocity at press2'])

    axs2[2].spines['right'].set_visible(False)
    axs2[2].spines['top'].set_visible(False)
    axs2[2].set_title('\n'+ 'peak velocity and velocity at presses')
    axs2[2].set_ylabel('Mean velocity (deg/s) +/- SEM')
    axs2[2].legend(loc='best')
    fig2.tight_layout()

    ########################################### PLOTS for GRAND sessions ###############################################################
    print('##########################')
    print('#########GRAND###########')
    print('##########################')

    fig3, axs3 = plt.subplots(nrows=3, figsize=(20, 10))   # should be changed
    fig3.suptitle(subject + ' - ' + session_date + ' \n ' + 'Timing analysis for Rewarded trials (GRANDED AVG SESSIONS)' + '\n')

    offset = 0.1

    x01 = 0
    x02 = 1
    x03 = 2

    axs3[0].errorbar(x01 - 2.5*offset, np.nanmean(G_chemo_target1_velocity_s, axis=0), np.nanstd(G_chemo_target1_velocity_s,axis=0), color = 'r',fmt='_', capsize=4, label = 'Chemo_Short')
    axs3[0].errorbar(x01 - 1.5*offset, np.nanmean(G_chemo_target1_velocity_l, axis=0), np.nanstd(G_chemo_target1_velocity_l,axis=0), color = 'r',fmt='o', capsize=4, label = 'Chemo_Long')

    axs3[0].errorbar(x01 - 0.5*offset, np.nanmean(G_con_target1_velocity_s, axis=0), np.nanstd(G_con_target1_velocity_s,axis=0), color = 'k',fmt='_', capsize=4, label = 'Control_Short')
    axs3[0].errorbar(x01 + 0.5*offset, np.nanmean(G_con_target1_velocity_l, axis=0), np.nanstd(G_con_target1_velocity_l,axis=0), color = 'k',fmt='o', capsize=4, label = 'Control_Long')

    axs3[0].errorbar(x01 + 1.5*offset, np.nanmean(G_opto_target1_velocity_s, axis=0), np.nanstd(G_opto_target1_velocity_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4, label = 'Opto_Short')
    axs3[0].errorbar(x01 + 2.5*offset, np.nanmean(G_opto_target1_velocity_l, axis=0), np.nanstd(G_opto_target1_velocity_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4, label = 'Opto_Long')

    axs3[0].errorbar(x02 - 2.5*offset, np.nanmean(G_chemo_target2_velocity_s, axis=0), np.nanstd(G_chemo_target2_velocity_s,axis=0), color = 'r',fmt='_', capsize=4)
    axs3[0].errorbar(x02 - 1.5*offset, np.nanmean(G_chemo_target2_velocity_l, axis=0), np.nanstd(G_chemo_target2_velocity_l,axis=0), color = 'r',fmt='o', capsize=4)

    axs3[0].errorbar(x02 - 0.5*offset, np.nanmean(G_con_target2_velocity_s, axis=0), np.nanstd(G_con_target2_velocity_s,axis=0), color = 'k',fmt='_', capsize=4)
    axs3[0].errorbar(x02 + 0.5*offset, np.nanmean(G_con_target2_velocity_l, axis=0), np.nanstd(G_con_target2_velocity_l,axis=0), color = 'k',fmt='o', capsize=4)

    axs3[0].errorbar(x02 + 1.5*offset, np.nanmean(G_opto_target2_velocity_s, axis=0), np.nanstd(G_opto_target2_velocity_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4)
    axs3[0].errorbar(x02 + 2.5*offset, np.nanmean(G_opto_target2_velocity_l, axis=0), np.nanstd(G_opto_target2_velocity_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4)

    axs3[0].errorbar(x03 - 2.5*offset, np.nanmean(G_chemo_interval_velocity_s, axis=0), np.nanstd(G_chemo_interval_velocity_s,axis=0), color = 'r',fmt='_', capsize=4)
    axs3[0].errorbar(x03 - 1.5*offset, np.nanmean(G_chemo_interval_velocity_l, axis=0), np.nanstd(G_chemo_interval_velocity_l,axis=0), color = 'r',fmt='o', capsize=4)

    axs3[0].errorbar(x03 - 0.5*offset, np.nanmean(G_con_interval_velocity_s, axis=0), np.nanstd(G_con_interval_velocity_s,axis=0), color = 'k',fmt='_', capsize=4)
    axs3[0].errorbar(x03 + 0.5*offset, np.nanmean(G_con_interval_velocity_l, axis=0), np.nanstd(G_con_interval_velocity_l,axis=0), color = 'k',fmt='o', capsize=4)

    axs3[0].errorbar(x03 + 1.5*offset, np.nanmean(G_opto_interval_velocity_s, axis=0), np.nanstd(G_opto_interval_velocity_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4)
    axs3[0].errorbar(x03 + 2.5*offset, np.nanmean(G_opto_interval_velocity_l, axis=0), np.nanstd(G_opto_interval_velocity_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4)

    # Set x-ticks to show the original category labels
    axs3[0].set_xticks([x01, x02, x03])
    axs3[0].set_xticklabels(['onset1_target1', 'onset2_target2', 'onset1_onset2_interval'])

    axs3[0].spines['right'].set_visible(False)
    axs3[0].spines['top'].set_visible(False)
    axs3[0].set_title('\n'+ subject)
    axs3[0].set_ylabel('Mean onset time (s) +/- SEM')
    axs3[0].legend(loc='best')

    x11 = 0
    x12 = 1

    axs3[1].errorbar(x11 - 2.5*offset, np.nanmean(G_chemo_amp_press1_s, axis=0), np.nanstd(G_chemo_amp_press1_s,axis=0), color = 'r',fmt='_', capsize=4, label = 'Chemo_Short')
    axs3[1].errorbar(x11 - 1.5*offset, np.nanmean(G_chemo_amp_press1_l, axis=0), np.nanstd(G_chemo_amp_press1_l,axis=0), color = 'r',fmt='o', capsize=4, label = 'Chemo_Long')

    axs3[1].errorbar(x11 - 0.5*offset, np.nanmean(G_con_amp_press1_s, axis=0), np.nanstd(G_con_amp_press1_s,axis=0), color = 'k',fmt='_', capsize=4, label = 'Control_Short')
    axs3[1].errorbar(x11 + 0.5*offset, np.nanmean(G_con_amp_press1_l, axis=0), np.nanstd(G_con_amp_press1_l,axis=0), color = 'k',fmt='o', capsize=4, label = 'Control_Long')

    axs3[1].errorbar(x11 + 1.5*offset, np.nanmean(G_opto_amp_press1_s, axis=0), np.nanstd(G_opto_amp_press1_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4, label = 'Opto_Short')
    axs3[1].errorbar(x11 + 2.5*offset, np.nanmean(G_opto_amp_press1_l, axis=0), np.nanstd(G_opto_amp_press1_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4, label = 'Opto_Long')

    axs3[1].errorbar(x12 - 2.5*offset, np.nanmean(G_chemo_amp_press2_s, axis=0), np.nanstd(G_chemo_amp_press2_s,axis=0), color = 'r',fmt='_', capsize=4)
    axs3[1].errorbar(x12 - 1.5*offset, np.nanmean(G_chemo_amp_press2_l, axis=0), np.nanstd(G_chemo_amp_press2_l,axis=0), color = 'r',fmt='o', capsize=4)

    axs3[1].errorbar(x12 - 0.5*offset, np.nanmean(G_con_amp_press2_s, axis=0), np.nanstd(G_con_amp_press2_s,axis=0), color = 'k',fmt='_', capsize=4)
    axs3[1].errorbar(x12 + 0.5*offset, np.nanmean(G_con_amp_press2_l, axis=0), np.nanstd(G_con_amp_press2_l,axis=0), color = 'k',fmt='o', capsize=4)

    axs3[1].errorbar(x12 + 1.5*offset, np.nanmean(G_opto_amp_press2_s, axis=0), np.nanstd(G_opto_amp_press2_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4)
    axs3[1].errorbar(x12 + 2.5*offset, np.nanmean(G_opto_amp_press2_l, axis=0), np.nanstd(G_opto_amp_press2_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4)

    # Set x-ticks to show the original category labels
    axs3[1].set_xticks([x11, x12])
    axs3[1].set_xticklabels(['baseline_peak1', 'baseline_peak2'])

    axs3[1].spines['right'].set_visible(False)
    axs3[1].spines['top'].set_visible(False)
    axs3[1].set_title('\n'+ 'Baseline to onset of presses')
    axs3[1].set_ylabel('Mean amplitude (deg) +/- SEM')
    axs3[1].legend(loc='best')

    x21 = 0
    x22 = 1
    x23 = 2
    x24 = 3

    axs3[2].errorbar(x21 - 2.5*offset, np.nanmean(G_chemo_max_press1_vel_s, axis=0), np.nanstd(G_chemo_max_press1_vel_s,axis=0), color = 'r',fmt='_', capsize=4, label = 'Chemo_Short')
    axs3[2].errorbar(x21 - 1.5*offset, np.nanmean(G_chemo_max_press1_vel_l, axis=0), np.nanstd(G_chemo_max_press1_vel_l,axis=0), color = 'r',fmt='o', capsize=4, label = 'Chemo_Long')

    axs3[2].errorbar(x21 - 0.5*offset, np.nanmean(G_con_max_press1_vel_s, axis=0), np.nanstd(G_con_max_press1_vel_s,axis=0), color = 'k',fmt='_', capsize=4, label = 'Control_Short')
    axs3[2].errorbar(x21 + 0.5*offset, np.nanmean(G_con_max_press1_vel_l, axis=0), np.nanstd(G_con_max_press1_vel_l,axis=0), color = 'k',fmt='o', capsize=4, label = 'Control_Long')

    axs3[2].errorbar(x21 + 1.5*offset, np.nanmean(G_opto_max_press1_vel_s, axis=0), np.nanstd(G_opto_max_press1_vel_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4, label = 'Opto_Short')
    axs3[2].errorbar(x21 + 2.5*offset, np.nanmean(G_opto_max_press1_vel_l, axis=0), np.nanstd(G_opto_max_press1_vel_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4, label = 'Opto_Long')

    axs3[2].errorbar(x22 - 2.5*offset, np.nanmean(G_chemo_max_press2_vel_s, axis=0), np.nanstd(G_chemo_max_press2_vel_s,axis=0), color = 'r',fmt='_', capsize=4)
    axs3[2].errorbar(x22 - 1.5*offset, np.nanmean(G_chemo_max_press2_vel_l, axis=0), np.nanstd(G_chemo_max_press2_vel_l,axis=0), color = 'r',fmt='o', capsize=4)

    axs3[2].errorbar(x22 - 0.5*offset, np.nanmean(G_con_max_press2_vel_s, axis=0), np.nanstd(G_con_max_press2_vel_s,axis=0), color = 'k',fmt='_', capsize=4)
    axs3[2].errorbar(x22 + 0.5*offset, np.nanmean(G_con_max_press2_vel_l, axis=0), np.nanstd(G_con_max_press2_vel_l,axis=0), color = 'k',fmt='o', capsize=4)

    axs3[2].errorbar(x22 + 1.5*offset, np.nanmean(G_opto_max_press2_vel_s, axis=0), np.nanstd(G_opto_max_press2_vel_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4)
    axs3[2].errorbar(x22 + 2.5*offset, np.nanmean(G_opto_max_press2_vel_l, axis=0), np.nanstd(G_opto_max_press2_vel_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4)

    axs3[2].errorbar(x23 - 2.5*offset, np.nanmean(G_chemo_velaton1_s, axis=0), np.nanstd(G_chemo_velaton1_s,axis=0), color = 'r',fmt='_', capsize=4)
    axs3[2].errorbar(x23 - 1.5*offset, np.nanmean(G_chemo_velaton1_l, axis=0), np.nanstd(G_chemo_velaton1_l,axis=0), color = 'r',fmt='o', capsize=4)

    axs3[2].errorbar(x23 - 0.5*offset, np.nanmean(G_con_velaton1_s, axis=0), np.nanstd(G_con_velaton1_s,axis=0), color = 'k',fmt='_', capsize=4)
    axs3[2].errorbar(x23 + 0.5*offset, np.nanmean(G_con_velaton1_l, axis=0), np.nanstd(G_con_velaton1_l,axis=0), color = 'k',fmt='o', capsize=4)

    axs3[2].errorbar(x23 + 1.5*offset, np.nanmean(G_opto_velaton1_s, axis=0), np.nanstd(G_opto_velaton1_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4)
    axs3[2].errorbar(x23 + 2.5*offset, np.nanmean(G_opto_velaton1_l, axis=0), np.nanstd(G_opto_velaton1_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4)

    axs3[2].errorbar(x24 - 2.5*offset, np.nanmean(G_chemo_velaton2_s, axis=0), np.nanstd(G_chemo_velaton2_s,axis=0), color = 'r',fmt='_', capsize=4)
    axs3[2].errorbar(x24 - 1.5*offset, np.nanmean(G_chemo_velaton2_l, axis=0), np.nanstd(G_chemo_velaton2_l,axis=0), color = 'r',fmt='o', capsize=4)

    axs3[2].errorbar(x24 - 0.5*offset, np.nanmean(G_con_velaton2_s, axis=0), np.nanstd(G_con_velaton2_s,axis=0), color = 'k',fmt='_', capsize=4)
    axs3[2].errorbar(x24 + 0.5*offset, np.nanmean(G_con_velaton2_l, axis=0), np.nanstd(G_con_velaton2_l,axis=0), color = 'k',fmt='o', capsize=4)

    axs3[2].errorbar(x24 + 1.5*offset, np.nanmean(G_opto_velaton2_s, axis=0), np.nanstd(G_opto_velaton2_s,axis=0), color = 'deepskyblue',fmt='_', capsize=4)
    axs3[2].errorbar(x24 + 2.5*offset, np.nanmean(G_opto_velaton2_l, axis=0), np.nanstd(G_opto_velaton2_l,axis=0), color = 'deepskyblue',fmt='o', capsize=4)

    # Set x-ticks to show the original category labels
    axs3[2].set_xticks([x21, x22, x23, x24])
    axs3[2].set_xticklabels(['peak_velocity_press1', 'peak_velocity_press2', 'velocity at press1', 'velocity at press2'])

    axs3[2].spines['right'].set_visible(False)
    axs3[2].spines['top'].set_visible(False)
    axs3[2].set_title('\n'+ 'peak velocity and velocity at presses')
    axs3[2].set_ylabel('Mean velocity (deg/s) +/- SEM')
    axs3[2].legend(loc='best')
    fig3.tight_layout()


    output_figs_dir = output_dir_onedrive + subject + '/'  
    pdf_path = os.path.join(output_figs_dir, subject + '_Timing_Amp_Sum_Analysis.pdf')

    # Save both plots into a single PDF file with each on a separate page
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)