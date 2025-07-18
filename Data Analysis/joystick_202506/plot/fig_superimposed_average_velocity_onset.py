import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfWriter, PdfReader
from datetime import date
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import re
import math

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

def find_half_peak_point_after(velocity, peak_idx):
    
    half_peak_value = 2.5
    for i in range(peak_idx + 1, len(velocity)):
        if velocity[i] <= half_peak_value:
            return i
    return 0


def find_half_peak_point_before(velocity, peak_idx):

    half_peak_value = 2.5
    for i in range(peak_idx - 1, -1, -1):
        if velocity[i] <= half_peak_value:
            return i
    return 0


def find_onethird_peak_point_before(velocity, peak_idx):
    peak_value = velocity[peak_idx]
    onethird_peak_value = peak_value * 0.4
    for i in range(peak_idx - 1, -1, -1):
        if velocity[i] <= onethird_peak_value:
            return i
    return 0


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


# Function to filter sublists by the length of the first sublist
def filter_sublists_by_length(data):
    if not data or (isinstance(data[0], (list, tuple)) and len(data[0]) == 0):
        return []  # Return an empty list if the input list is empty or the first sublist is empty
    
    first_length = len(data[0])
    filtered_data = [sublist for sublist in data if len(sublist) == first_length]
    return filtered_data

def plot_aligend_onset (
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

    chemo_labels = session_data['chemo']
    print(chemo_labels)
    for i in range(1 , len(chemo_labels)):
            if chemo_labels[i] == 1:
                dates[i] = dates[i] + '(chemo)'

    num_traces = len(session_id)
    # Create color gradients with more distinct shades
    red_colors = [(1 - i*0.1, 0, 0) for i in range(num_traces)]  # Shades of red
    gray_colors = [(0.9 - i * 0.1, 0.9 - i * 0.1, 0.9 - i * 0.1) for i in range(num_traces)]  # Shades of gray

    mean_encoder_positions_aligned_vel1_all = []
    mean_encoder_positions_aligned_vel2_all = []
    mean_encoder_positions_aligned_vel1_S = []
    mean_encoder_positions_aligned_vel2_S =[]
    mean_encoder_positions_aligned_vel2_L = []
    mean_encoder_positions_aligned_vel1_L = []

    mega_encoder_positions_aligned_vel1_con_S = []
    mega_encoder_positions_aligned_vel1_con_L = []
    mega_encoder_positions_aligned_vel2_con_L = []
    mega_encoder_positions_aligned_vel2_con_S = []

    mega_encoder_positions_aligned_vel1_chem_S = []
    mega_encoder_positions_aligned_vel1_chem_L = []
    mega_encoder_positions_aligned_vel2_chem_L = []
    mega_encoder_positions_aligned_vel2_chem_S = []

    mega_encoder_positions_aligned_vel2_con_all = []
    mega_encoder_positions_aligned_vel1_con_all = []
    mega_encoder_positions_aligned_vel2_chem_all = []
    mega_encoder_positions_aligned_vel1_chem_all = []

    # mean_mega_encoder_positions_aligned_vel1_con_S = []
    # mean_mega_encoder_positions_aligned_vel1_con_L = []
    # mean_mega_encoder_positions_aligned_vel2_con_L = []
    # mean_mega_encoder_positions_aligned_vel2_con_S = []

    # mean_mega_encoder_positions_aligned_vel1_chem_S = []
    # mean_mega_encoder_positions_aligned_vel1_chem_L = []
    # mean_mega_encoder_positions_aligned_vel2_chem_L = []
    # mega_encoder_positions_aligned_vel2_chem_S = []

    # mean_mega_encoder_positions_aligned_vel2_con_all = []
    # mean_mega_encoder_positions_aligned_vel1_con_all = []
    # mean_mega_encoder_positions_aligned_vel2_chem_all = []
    # mean_mega_encoder_positions_aligned_vel1_chem_all = []
                
    if any(0 in row for row in isSelfTimedMode):
        print('Visually Guided')
        fig1,axs1 = plt.subplots(nrows= 3, ncols= 2,figsize =(10,10))
        fig1.suptitle(subject + '\n' + 'Superimposed averaged rewarded trials' + '\n')
        
        
        
        encoder_times = np.linspace(0, 1.3, num= 1300)
        encoder_times2 = np.linspace(0, 1.5, num= 1500)
        dates = deduplicate_chemo(dates)
        
        for i in range(0 , len(session_id)):
            TrialOutcomes = session_data['outcomes'][i]
            # We have Raw data and extract every thing from it (Times)
            raw_data = session_data['raw'][i]
            session_date = dates[i][2:]
            trial_types = raw_data['TrialTypes']
            print('analysis for delay of session:' + session_date)
            onset_press1 = []
            # rotatory1_1 = []
            onset_press2 = []
            stimulations2_peak = []
            targetTime_press1 = []
            targetTime_press2 = []
            targetTime_press2_reward = []
            targetTime_press2_early = []
            press_vis_2 = []
            amp_press1 = []
            amp_press2 = []
            amp_press2_reward = []
            amp_press2_early = []
            # VisualStimulus11 = []
            interval_early = []
            interval_reward = []
            target1_velocity = []
            target2_velocity = []
            target2_velocity_early = []
            target2_velocity_reward = []
            interval_velocity_reward = []
            interval_velocity_early = []
            interval_velocity = []
            interval = []
            encoder_positions_aligned_vel1_all = []
            encoder_positions_aligned_vel2_all = []
            encoder_positions_aligned_vel1_S = []
            encoder_positions_aligned_vel1_L = []
            encoder_positions_aligned_vel2_L = []
            encoder_positions_aligned_vel2_S = []
            on1_vel_mean = []
            on2_vel_mean = []
            on1_vel_S_mean = []
            on1_vel_L_mean = []
            on2_vel_S_mean = []
            on2_vel_L_mean = []
            
            for trial in range(0,len(TrialOutcomes)):
                
                if np.isnan(isSelfTimedMode[i][trial]):
                    continue
                
                if TrialOutcomes[trial] == 'Reward':
                    encoder_data = raw_data['EncoderData'][trial]              
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                    
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
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
                    amp_press1.append(np.max(encoder_positions_aligned_vis1[rotatory1:VisDetect2]) - base_line)
                    

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
                    amp_press2.append(np.max(encoder_positions_aligned_vis1[VisualStimulus2:LeverRetract2]) - base_line)
                    amp_press2_reward.append(np.max(encoder_positions_aligned_vis1[VisualStimulus2:LeverRetract2]) - base_line)
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
                    target1_velocity.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                    target2_velocity.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                    target2_velocity_reward.append(np.abs(rotatory2 - on2_vel)/1000) # type: ignore
                    intervall_vel = on2_vel - on1_vel # type: ignore
                    interval_velocity.append(intervall_vel/1000)
                    interval_velocity_reward.append(intervall_vel/1000)
                    
                    on1_vel_mean.append(on1_vel)
                    on2_vel_mean.append(on2_vel)
                    
                    if on1_vel < 300: # type: ignore
                        # Calculate the number of zeros to prepend
                        num_zeros = 300 - on1_vel # type: ignore
                    
                        # Pad with zeros at the beginning
                        encoder_positions_aligned_vis1 = np.concatenate((np.zeros(num_zeros), encoder_positions_aligned_vis1))
                        on1_vel = 300
                    
                    encoder_positions_aligned_vel1_all.append(encoder_positions_aligned_vis1[on1_vel - 300 : on1_vel + 1000]) # type: ignore
                    encoder_positions_aligned_vel2_all.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                    
                    if 'chemo' in dates[i]:
                        mega_encoder_positions_aligned_vel1_chem_all.append(encoder_positions_aligned_vis1[on1_vel - 300 : on1_vel + 1000]) # type: ignore
                        mega_encoder_positions_aligned_vel2_chem_all.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                    else:
                        mega_encoder_positions_aligned_vel1_con_all.append(encoder_positions_aligned_vis1[on1_vel - 300 : on1_vel + 1000]) # type: ignore
                        mega_encoder_positions_aligned_vel2_con_all.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                    
                    if trial_types[trial] == 1:
                        encoder_positions_aligned_vel1_S.append(encoder_positions_aligned_vis1[on1_vel -300 : on1_vel + 1000]) # type: ignore
                        encoder_positions_aligned_vel2_S.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                        on2_vel_mean.append(on2_vel)
                        on2_vel_mean.append(on2_vel)
                        if 'chemo' in dates[i]:
                            mega_encoder_positions_aligned_vel1_chem_S.append(encoder_positions_aligned_vis1[on1_vel -300 : on1_vel + 1000]) # type: ignore
                            mega_encoder_positions_aligned_vel2_chem_S.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                        else:
                            mega_encoder_positions_aligned_vel1_con_S.append(encoder_positions_aligned_vis1[on1_vel -300 : on1_vel + 1000]) # type: ignore
                            mega_encoder_positions_aligned_vel2_con_S.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                    else:
                        encoder_positions_aligned_vel1_L.append(encoder_positions_aligned_vis1[on1_vel - 300 : on1_vel + 1000]) # type: ignore
                        encoder_positions_aligned_vel2_L.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                        on2_vel_mean.append(on2_vel)
                        on2_vel_mean.append(on2_vel)
                        if 'chemo' in dates[i]:
                            mega_encoder_positions_aligned_vel1_chem_L.append(encoder_positions_aligned_vis1[on1_vel -300 : on1_vel + 1000]) # type: ignore
                            mega_encoder_positions_aligned_vel2_chem_L.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                        else:
                            mega_encoder_positions_aligned_vel1_con_L.append(encoder_positions_aligned_vis1[on1_vel -300 : on1_vel + 1000]) # type: ignore
                            mega_encoder_positions_aligned_vel2_con_L.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
        

            # Compute the mean along the rows
            encoder_positions_aligned_vel1_all = filter_sublists_by_length(encoder_positions_aligned_vel1_all)
            encoder_positions_aligned_vel2_all = filter_sublists_by_length(encoder_positions_aligned_vel2_all)
            encoder_positions_aligned_vel1_S = filter_sublists_by_length(encoder_positions_aligned_vel1_S)
            encoder_positions_aligned_vel2_S = filter_sublists_by_length(encoder_positions_aligned_vel2_S)
            encoder_positions_aligned_vel1_L = filter_sublists_by_length(encoder_positions_aligned_vel1_L)
            encoder_positions_aligned_vel2_L = filter_sublists_by_length(encoder_positions_aligned_vel2_L)
                    
            mean_encoder_positions_aligned_vel1_all.append(np.mean(encoder_positions_aligned_vel1_all, axis=0))
            mean_encoder_positions_aligned_vel2_all.append(np.mean(encoder_positions_aligned_vel2_all, axis=0))
            mean_encoder_positions_aligned_vel1_S.append(np.mean(encoder_positions_aligned_vel1_S, axis=0))
            mean_encoder_positions_aligned_vel2_S.append(np.mean(encoder_positions_aligned_vel2_S, axis=0))
            mean_encoder_positions_aligned_vel1_L.append(np.mean(encoder_positions_aligned_vel1_L, axis=0))
            mean_encoder_positions_aligned_vel2_L.append(np.mean(encoder_positions_aligned_vel2_L, axis=0))
        
        
            dates = deduplicate_chemo(dates)
            output_figs_dir = output_dir_onedrive + subject + '/'    
            output_imgs_dir = output_dir_local + subject + '/Superimposed average velocity onset/'
            
            if 'chemo' in dates[i]:
                # Plotting
                axs1[0,0].plot(encoder_times - 0.3,mean_encoder_positions_aligned_vel1_all[i], color = red_colors[i % len(red_colors)], label = dates[i])
                axs1[0,0].set_title('aligned all trials on onset1')
                axs1[0,0].set_xlabel('time (s)')
                axs1[0,0].set_ylabel('joystick deflection (deg)')
                axs1[0,0].set_xlim(-0.5,1.5)
                axs1[0,0].axvline(x = 0, color = 'r', linestyle='--')
                axs1[0,0].legend()
                axs1[0,0].spines['top'].set_visible(False)
                axs1[0,0].spines['right'].set_visible(False)
            
                axs1[0,1].plot(encoder_times2 - 0.5 ,mean_encoder_positions_aligned_vel2_all[i], color = red_colors[i % len(red_colors)], label = dates[i])
                axs1[0,1].set_title('aligned all trials on onset2')
                axs1[0,1].set_xlabel('time (s)')
                axs1[0,1].set_ylabel('joystick deflection (deg)')
                axs1[0,1].set_xlim(-0.5,1.5)
                axs1[0,1].axvline(x = 0, color = 'r', linestyle='--')
                axs1[0,1].legend()
                axs1[0,1].spines['top'].set_visible(False)
                axs1[0,1].spines['right'].set_visible(False)
            
                axs1[1,0].plot(encoder_times - 0.3,mean_encoder_positions_aligned_vel1_S[i], color = red_colors[i % len(red_colors)], label = dates[i])
                axs1[1,0].set_title('aligned short trials on onset1')
                axs1[1,0].set_xlabel('time (s)')
                axs1[1,0].set_ylabel('joystick deflection (deg)')
                axs1[1,0].set_xlim(-0.5,1.5)
                axs1[1,0].axvline(x = 0, color = 'r', linestyle='--')
                axs1[1,0].legend()
                axs1[1,0].spines['top'].set_visible(False)
                axs1[1,0].spines['right'].set_visible(False)
            
                axs1[1,1].plot(encoder_times2 - 0.5,mean_encoder_positions_aligned_vel2_S[i],color = red_colors[i % len(red_colors)], label = dates[i])
                axs1[1,1].set_title('aligned short trials on onset2')
                axs1[1,1].set_xlabel('time (s)')
                axs1[1,1].set_ylabel('joystick deflection (deg)')
                axs1[1,1].set_xlim(-0.5,1.5)
                axs1[1,1].axvline(x = 0, color = 'r', linestyle='--')
                axs1[1,1].legend()
                axs1[1,1].spines['top'].set_visible(False)
                axs1[1,1].spines['right'].set_visible(False)
            
                axs1[2,0].plot(encoder_times - 0.3,mean_encoder_positions_aligned_vel1_L[i], color = red_colors[i % len(red_colors)], label = dates[i])
                axs1[2,0].set_title('aligned long trials on onset1')
                axs1[2,0].set_xlabel('time (s)')
                axs1[2,0].set_ylabel('joystick deflection (deg)')
                axs1[2,0].set_xlim(-0.5,1.5)
                axs1[2,0].axvline(x = 0, color = 'r', linestyle='--')
                axs1[2,0].legend()
                axs1[2,0].spines['top'].set_visible(False)
                axs1[2,0].spines['right'].set_visible(False)
            
                axs1[2,1].plot(encoder_times2 - 0.5,mean_encoder_positions_aligned_vel2_L[i], color = red_colors[i % len(red_colors)], label = dates[i])
                axs1[2,1].set_title('aligned long trials on onset2')
                axs1[2,1].set_xlabel('time (s)')
                axs1[2,1].set_ylabel('joystick deflection (deg)')
                axs1[2,1].set_xlim(-0.5,1.5)
                axs1[2,1].axvline(x = 0, color = 'r', linestyle='--')
                axs1[2,1].legend()
                axs1[2,1].spines['top'].set_visible(False)
                axs1[2,1].spines['right'].set_visible(False)
            
            else:
                # Plotting
                print(i)
                axs1[0,0].plot(encoder_times - 0.3,mean_encoder_positions_aligned_vel1_all[i], color = gray_colors[i % len(gray_colors)], label = dates[i])
                axs1[0,0].set_title('aligned all trials on onset1')
                axs1[0,0].set_xlabel('time (s)')
                axs1[0,0].set_ylabel('joystick deflection (deg)')
                axs1[0,0].set_xlim(-0.5,1.5)
                axs1[0,0].axvline(x = 0, color = 'r', linestyle='--')
                axs1[0,0].spines['top'].set_visible(False)
                axs1[0,0].spines['right'].set_visible(False)
                axs1[0,0].legend()
                
                axs1[0,1].plot(encoder_times2 - 0.5 ,mean_encoder_positions_aligned_vel2_all[i], color = gray_colors[i % len(gray_colors)], label = dates[i])
                axs1[0,1].set_title('aligned all trials on onset2')
                axs1[0,1].set_xlabel('time (s)')
                axs1[0,1].set_ylabel('joystick deflection (deg)')
                axs1[0,1].set_xlim(-0.5,1.5)
                axs1[0,1].axvline(x = 0, color = 'r', linestyle='--')
                axs1[0,1].legend()
                axs1[0,1].spines['top'].set_visible(False)
                axs1[0,1].spines['right'].set_visible(False)
            
                axs1[1,0].plot(encoder_times - 0.3,mean_encoder_positions_aligned_vel1_S[i], color = gray_colors[i % len(gray_colors)], label = dates[i])
                axs1[1,0].set_title('aligned short trials on onset1')
                axs1[1,0].set_xlabel('time (s)')
                axs1[1,0].set_ylabel('joystick deflection (deg)')
                axs1[1,0].set_xlim(-0.5,1.5)
                axs1[1,0].axvline(x = 0, color = 'r', linestyle='--')
                axs1[1,0].legend()
                axs1[1,0].spines['top'].set_visible(False)
                axs1[1,0].spines['right'].set_visible(False)
            
                axs1[1,1].plot(encoder_times2 - 0.5,mean_encoder_positions_aligned_vel2_S[i],color = gray_colors[i % len(gray_colors)], label = dates[i])
                axs1[1,1].set_title('aligned short trials on onset2')
                axs1[1,1].set_xlabel('time (s)')
                axs1[1,1].set_ylabel('joystick deflection (deg)')
                axs1[1,1].set_xlim(-0.5,1.5)
                axs1[1,1].axvline(x = 0, color = 'r', linestyle='--')
                axs1[1,1].legend()
                axs1[1,1].spines['top'].set_visible(False)
                axs1[1,1].spines['right'].set_visible(False)
            
                axs1[2,0].plot(encoder_times - 0.3,mean_encoder_positions_aligned_vel1_L[i], color = gray_colors[i % len(gray_colors)], label = dates[i])
                axs1[2,0].set_title('aligned long trials on onset1')
                axs1[2,0].set_xlabel('time (s)')
                axs1[2,0].set_ylabel('joystick deflection (deg)')
                axs1[2,0].set_xlim(-0.5,1.5)
                axs1[2,0].axvline(x = 0, color = 'r', linestyle='--')
                axs1[2,0].legend()
                axs1[2,0].spines['top'].set_visible(False)
                axs1[2,0].spines['right'].set_visible(False)
            
                axs1[2,1].plot(encoder_times2 - 0.5,mean_encoder_positions_aligned_vel2_L[i], color = gray_colors[i % len(gray_colors)], label = dates[i])
                axs1[2,1].set_title('aligned long trials on onset2')
                axs1[2,1].set_xlabel('time (s)')
                axs1[2,1].set_ylabel('joystick deflection (deg)')
                axs1[2,1].set_xlim(-0.5,1.5)
                axs1[2,1].axvline(x = 0, color = 'r', linestyle='--')
                axs1[2,1].legend()
                axs1[2,1].spines['top'].set_visible(False)
                axs1[2,1].spines['right'].set_visible(False)
            
        
        fig1.tight_layout()
        # os.makedirs(output_figs_dir, exist_ok = True)
        # os.makedirs(output_imgs_dir, exist_ok = True)
        # fig1.savefig(output_figs_dir + subject + '_superimposed_averaged_velocity_onset.pdf', dpi=300)
        # fig1.savefig(output_imgs_dir + subject + '_superimposed_averaged_velocity_onset.png', dpi=300)
        # plt.close(fig1)
        
    else:
        print('selftime')
        
        fig1,axs1 = plt.subplots(nrows= 3, ncols= 2,figsize =(10,10))
        fig1.suptitle(subject + '\n' + 'Superimposed averaged rewarded trials' + '\n')
        
        encoder_times = np.linspace(0, 1.3, num= 1300)
        encoder_times2 = np.linspace(0, 1.5, num= 1500)
        dates = deduplicate_chemo(dates)
        
        for i in range(0 , len(session_id)):
            TrialOutcomes = session_data['outcomes'][i]
            # We have Raw data and extract every thing from it (Times)
            raw_data = session_data['raw'][i]
            session_date = dates[i][2:]
            trial_types = raw_data['TrialTypes']
            print('analysis for delay of session:' + session_date)
            
            onset_press1 = []
            # rotatory1_1 = []
            onset_press2 = []
            stimulations2_peak = []
            targetTime_press1 = []
            targetTime_press2 = []
            targetTime_press2_reward = []
            targetTime_press2_early = []
            press_vis_2 = []
            amp_press1 = []
            amp_press2 = []
            amp_press2_reward = []
            amp_press2_early = []
            # VisualStimulus11 = []
            interval_early = []
            interval_reward = []
            target1_velocity = []
            target2_velocity = []
            target2_velocity_early = []
            target2_velocity_reward = []
            interval_velocity_reward = []
            interval_velocity_early = []
            interval_velocity = []
            interval = []
            encoder_positions_aligned_vel1_all = []
            encoder_positions_aligned_vel2_all = []
            encoder_positions_aligned_vel1_S = []
            encoder_positions_aligned_vel1_L = []
            encoder_positions_aligned_vel2_L = []
            encoder_positions_aligned_vel2_S = []
            on1_vel_mean = []
            on2_vel_mean = []
            on1_vel_S_mean = []
            on1_vel_L_mean = []
            on2_vel_S_mean = []
            on2_vel_L_mean = []
            
            for trial in range(0,len(TrialOutcomes)):
                
                if np.isnan(isSelfTimedMode[i][trial]):
                    continue
                
                if TrialOutcomes[trial] == 'Reward':
                    encoder_data = raw_data['EncoderData'][trial]              
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                    
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
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
                    amp_press1.append(np.max(encoder_positions_aligned_vis1[rotatory1:LeverRetract1]) - base_line)
                    

                    ####### Calculations for vis 2
                    threshold_press2 = base_line + 0.25
                    amp2 = np.argmax(encoder_positions_aligned_vis1[LeverRetract1:LeverRetract2])
                    amp2 = amp2 + LeverRetract1
                    onset2_position = first_touch_onset(encoder_positions_aligned_vis1, threshold_press2,LeverRetract1,amp2)
                    
                    # Target 2 to Onset_press 2
                    targetTime_press2.append(np.abs(rotatory2 - onset2_position)/1000) # type: ignore
                    targetTime_press2_reward.append(np.abs(rotatory2 - onset2_position)/1000) # type: ignore
                    ##### press2
                    amp_press2.append(np.max(encoder_positions_aligned_vis1[LeverRetract1:LeverRetract2]) - base_line)
                    amp_press2_reward.append(np.max(encoder_positions_aligned_vis1[LeverRetract1:LeverRetract2]) - base_line)
                    
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
                    target1_velocity.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                    target2_velocity.append(np.abs(LeverRetract2 - on2_vel)/1000) # type: ignore
                    target2_velocity_reward.append(np.abs(LeverRetract2 - on2_vel)/1000) # type: ignore
                    intervall_vel = on2_vel - on1_vel # type: ignore
                    interval_velocity.append(intervall_vel/1000)
                    interval_velocity_reward.append(intervall_vel/1000)
                    
                    on1_vel_mean.append(on1_vel)
                    on2_vel_mean.append(on2_vel)
                    
                    if on1_vel < 300: # type: ignore
                        # Calculate the number of zeros to prepend
                        num_zeros = 300 - on1_vel # type: ignore
                    
                        # Pad with zeros at the beginning
                        encoder_positions_aligned_vis1 = np.concatenate((np.zeros(num_zeros), encoder_positions_aligned_vis1))
                        on1_vel = 300
                    
                    encoder_positions_aligned_vel1_all.append(encoder_positions_aligned_vis1[on1_vel - 300 : on1_vel + 1000]) # type: ignore
                    encoder_positions_aligned_vel2_all.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                    
                    if 'chemo' in dates[i]:
                        mega_encoder_positions_aligned_vel1_chem_all.append(encoder_positions_aligned_vis1[on1_vel - 300 : on1_vel + 1000]) # type: ignore
                        mega_encoder_positions_aligned_vel2_chem_all.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                    else:
                        mega_encoder_positions_aligned_vel1_con_all.append(encoder_positions_aligned_vis1[on1_vel - 300 : on1_vel + 1000]) # type: ignore
                        mega_encoder_positions_aligned_vel2_con_all.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                    
                    if trial_types[trial] == 1:
                        encoder_positions_aligned_vel1_S.append(encoder_positions_aligned_vis1[on1_vel -300 : on1_vel + 1000]) # type: ignore
                        encoder_positions_aligned_vel2_S.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                        on2_vel_mean.append(on2_vel)
                        on2_vel_mean.append(on2_vel)
                        if 'chemo' in dates[i]:
                            mega_encoder_positions_aligned_vel1_chem_S.append(encoder_positions_aligned_vis1[on1_vel -300 : on1_vel + 1000]) # type: ignore
                            mega_encoder_positions_aligned_vel2_chem_S.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                        else:
                            mega_encoder_positions_aligned_vel1_con_S.append(encoder_positions_aligned_vis1[on1_vel -300 : on1_vel + 1000]) # type: ignore
                            mega_encoder_positions_aligned_vel2_con_S.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                    else:
                        encoder_positions_aligned_vel1_L.append(encoder_positions_aligned_vis1[on1_vel - 300 : on1_vel + 1000]) # type: ignore
                        encoder_positions_aligned_vel2_L.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                        on2_vel_mean.append(on2_vel)
                        on2_vel_mean.append(on2_vel)
                        if 'chemo' in dates[i]:
                            mega_encoder_positions_aligned_vel1_chem_L.append(encoder_positions_aligned_vis1[on1_vel -300 : on1_vel + 1000]) # type: ignore
                            mega_encoder_positions_aligned_vel2_chem_L.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
                        else:
                            mega_encoder_positions_aligned_vel1_con_L.append(encoder_positions_aligned_vis1[on1_vel -300 : on1_vel + 1000]) # type: ignore
                            mega_encoder_positions_aligned_vel2_con_L.append(encoder_positions_aligned_vis1[on2_vel - 500 : on2_vel + 1000]) # type: ignore
        

            # Compute the mean along the rows
            encoder_positions_aligned_vel1_all = filter_sublists_by_length(encoder_positions_aligned_vel1_all)
            encoder_positions_aligned_vel2_all = filter_sublists_by_length(encoder_positions_aligned_vel2_all)
            encoder_positions_aligned_vel1_S = filter_sublists_by_length(encoder_positions_aligned_vel1_S)
            encoder_positions_aligned_vel2_S = filter_sublists_by_length(encoder_positions_aligned_vel2_S)
            encoder_positions_aligned_vel1_L = filter_sublists_by_length(encoder_positions_aligned_vel1_L)
            encoder_positions_aligned_vel2_L = filter_sublists_by_length(encoder_positions_aligned_vel2_L)
                    
            mean_encoder_positions_aligned_vel1_all.append(np.mean(encoder_positions_aligned_vel1_all, axis=0))
            mean_encoder_positions_aligned_vel2_all.append(np.mean(encoder_positions_aligned_vel2_all, axis=0))
            mean_encoder_positions_aligned_vel1_S.append(np.mean(encoder_positions_aligned_vel1_S, axis=0))
            mean_encoder_positions_aligned_vel2_S.append(np.mean(encoder_positions_aligned_vel2_S, axis=0))
            mean_encoder_positions_aligned_vel1_L.append(np.mean(encoder_positions_aligned_vel1_L, axis=0))
            mean_encoder_positions_aligned_vel2_L.append(np.mean(encoder_positions_aligned_vel2_L, axis=0))
        
        
            dates = deduplicate_chemo(dates)
            output_figs_dir = output_dir_onedrive + subject + '/'
            output_imgs_dir = output_dir_local + subject + '/Superimposed average velocity onset/'
            
            if 'chemo' in dates[i]:
                # Plotting
                axs1[0,0].plot(encoder_times - 0.3,mean_encoder_positions_aligned_vel1_all[i], color = red_colors[i % len(red_colors)], label = dates[i])
                axs1[0,0].set_title('aligned all trials on onset1')
                axs1[0,0].set_xlabel('time (s)')
                axs1[0,0].set_ylabel('joystick deflection (deg)')
                axs1[0,0].set_xlim(-0.5,1.5)
                axs1[0,0].axvline(x = 0, color = 'r', linestyle='--')
                axs1[0,0].legend()
                axs1[0,0].spines['top'].set_visible(False)
                axs1[0,0].spines['right'].set_visible(False)
            
                
                axs1[0,1].plot(encoder_times2 - 0.5 ,mean_encoder_positions_aligned_vel2_all[i], color = red_colors[i % len(red_colors)], label = dates[i])
                axs1[0,1].set_title('aligned all trials on onset2')
                axs1[0,1].set_xlabel('time (s)')
                axs1[0,1].set_ylabel('joystick deflection (deg)')
                axs1[0,1].set_xlim(-0.5,1.5)
                axs1[0,1].axvline(x = 0, color = 'r', linestyle='--')
                axs1[0,1].legend()
                axs1[0,1].spines['top'].set_visible(False)
                axs1[0,1].spines['right'].set_visible(False)
                
                axs1[1,0].plot(encoder_times - 0.3,mean_encoder_positions_aligned_vel1_S[i], color = red_colors[i % len(red_colors)], label = dates[i])
                axs1[1,0].set_title('aligned short trials on onset1')
                axs1[1,0].set_xlabel('time (s)')
                axs1[1,0].set_ylabel('joystick deflection (deg)')
                axs1[1,0].set_xlim(-0.5,1.5)
                axs1[1,0].axvline(x = 0, color = 'r', linestyle='--')
                axs1[1,0].legend()
                axs1[1,0].spines['top'].set_visible(False)
                axs1[1,0].spines['right'].set_visible(False)
                
                axs1[1,1].plot(encoder_times2 - 0.5,mean_encoder_positions_aligned_vel2_S[i],color = red_colors[i % len(red_colors)], label = dates[i])
                axs1[1,1].set_title('aligned short trials on onset2')
                axs1[1,1].set_xlabel('time (s)')
                axs1[1,1].set_ylabel('joystick deflection (deg)')
                axs1[1,1].set_xlim(-0.5,1.5)
                axs1[1,1].axvline(x = 0, color = 'r', linestyle='--')
                axs1[1,1].legend()
                axs1[1,1].spines['top'].set_visible(False)
                axs1[1,1].spines['right'].set_visible(False)
                
                axs1[2,0].plot(encoder_times - 0.3,mean_encoder_positions_aligned_vel1_L[i], color = red_colors[i % len(red_colors)], label = dates[i])
                axs1[2,0].set_title('aligned long trials on onset1')
                axs1[2,0].set_xlabel('time (s)')
                axs1[2,0].set_ylabel('joystick deflection (deg)')
                axs1[2,0].set_xlim(-0.5,1.5)
                axs1[2,0].axvline(x = 0, color = 'r', linestyle='--')
                axs1[2,0].legend()
                axs1[2,0].spines['top'].set_visible(False)
                axs1[2,0].spines['right'].set_visible(False)
                
                axs1[2,1].plot(encoder_times2 - 0.5,mean_encoder_positions_aligned_vel2_L[i], color = red_colors[i % len(red_colors)], label = dates[i])
                axs1[2,1].set_title('aligned long trials on onset2')
                axs1[2,1].set_xlabel('time (s)')
                axs1[2,1].set_ylabel('joystick deflection (deg)')
                axs1[2,1].set_xlim(-0.5,1.5)
                axs1[2,1].axvline(x = 0, color = 'r', linestyle='--')
                axs1[2,1].legend()
                axs1[2,1].spines['top'].set_visible(False)
                axs1[2,1].spines['right'].set_visible(False)
                
            else:
                # Plotting
                axs1[0,0].plot(encoder_times - 0.3,mean_encoder_positions_aligned_vel1_all[i], color = gray_colors[i % len(gray_colors)], label = dates[i])
                axs1[0,0].set_title('aligned all trials on onset1')
                axs1[0,0].set_xlabel('time (s)')
                axs1[0,0].set_ylabel('joystick deflection (deg)')
                axs1[0,0].set_xlim(-0.5,1.5)
                axs1[0,0].axvline(x = 0, color = 'r', linestyle='--')
                axs1[0,0].legend()
                axs1[0,0].spines['top'].set_visible(False)
                axs1[0,0].spines['right'].set_visible(False)
                
                axs1[0,1].plot(encoder_times2 - 0.5 ,mean_encoder_positions_aligned_vel2_all[i], color = gray_colors[i % len(gray_colors)], label = dates[i])
                axs1[0,1].set_title('aligned all trials on onset2')
                axs1[0,1].set_xlabel('time (s)')
                axs1[0,1].set_ylabel('joystick deflection (deg)')
                axs1[0,1].set_xlim(-0.5,1.5)
                axs1[0,1].axvline(x = 0, color = 'r', linestyle='--')
                axs1[0,1].legend()
                axs1[0,1].spines['top'].set_visible(False)
                axs1[0,1].spines['right'].set_visible(False)
                
                axs1[1,0].plot(encoder_times - 0.3,mean_encoder_positions_aligned_vel1_S[i], color = gray_colors[i % len(gray_colors)], label = dates[i])
                axs1[1,0].set_title('aligned short trials on onset1')
                axs1[1,0].set_xlabel('time (s)')
                axs1[1,0].set_ylabel('joystick deflection (deg)')
                axs1[1,0].set_xlim(-0.5,1.5)
                axs1[1,0].axvline(x = 0, color = 'r', linestyle='--')
                axs1[1,0].legend()
                axs1[1,0].spines['top'].set_visible(False)
                axs1[1,0].spines['right'].set_visible(False)
                
                axs1[1,1].plot(encoder_times2 - 0.5,mean_encoder_positions_aligned_vel2_S[i],color = gray_colors[i % len(gray_colors)], label = dates[i])
                axs1[1,1].set_title('aligned short trials on onset2')
                axs1[1,1].set_xlabel('time (s)')
                axs1[1,1].set_ylabel('joystick deflection (deg)')
                axs1[1,1].set_xlim(-0.5,1.5)
                axs1[1,1].axvline(x = 0, color = 'r', linestyle='--')
                axs1[1,1].legend()
                axs1[1,1].spines['top'].set_visible(False)
                axs1[1,1].spines['right'].set_visible(False)
                
                axs1[2,0].plot(encoder_times - 0.3,mean_encoder_positions_aligned_vel1_L[i], color = gray_colors[i % len(gray_colors)], label = dates[i])
                axs1[2,0].set_title('aligned long trials on onset1')
                axs1[2,0].set_xlabel('time (s)')
                axs1[2,0].set_ylabel('joystick deflection (deg)')
                axs1[2,0].set_xlim(-0.5,1.5)
                axs1[2,0].axvline(x = 0, color = 'r', linestyle='--')
                axs1[2,0].legend()
                axs1[2,0].spines['top'].set_visible(False)
                axs1[2,0].spines['right'].set_visible(False)
                
                axs1[2,1].plot(encoder_times2 - 0.5,mean_encoder_positions_aligned_vel2_L[i], color = gray_colors[i % len(gray_colors)], label = dates[i])
                axs1[2,1].set_title('aligned long trials on onset2')
                axs1[2,1].set_xlabel('time (s)')
                axs1[2,1].set_ylabel('joystick deflection (deg)')
                axs1[2,1].set_xlim(-0.5,1.5)
                axs1[2,1].axvline(x = 0, color = 'r', linestyle='--')
                axs1[2,1].legend()
                axs1[2,1].spines['top'].set_visible(False)
                axs1[2,1].spines['right'].set_visible(False)
                
            fig1.tight_layout()

        # fig1.savefig(output_figs_dir + subject + '_superimposed_averaged_velocity_onset.pdf', dpi=300)
        # fig1.savefig(output_imgs_dir + subject + '_superimposed_averaged_velocity_onset.png', dpi=300)
        
    ############################### Mega session analysis ###############################################
    print('####### Mega ########')

    mega_encoder_positions_aligned_vel1_chem_all = filter_sublists_by_length(mega_encoder_positions_aligned_vel1_chem_all)
    mega_encoder_positions_aligned_vel1_con_all  = filter_sublists_by_length(mega_encoder_positions_aligned_vel1_con_all)
    mega_encoder_positions_aligned_vel2_chem_all = filter_sublists_by_length(mega_encoder_positions_aligned_vel2_chem_all)
    mega_encoder_positions_aligned_vel2_con_all = filter_sublists_by_length(mega_encoder_positions_aligned_vel2_con_all)
    mega_encoder_positions_aligned_vel1_con_S = filter_sublists_by_length(mega_encoder_positions_aligned_vel1_con_S)
    mega_encoder_positions_aligned_vel1_con_L = filter_sublists_by_length(mega_encoder_positions_aligned_vel1_con_L)
    mega_encoder_positions_aligned_vel2_con_L = filter_sublists_by_length(mega_encoder_positions_aligned_vel2_con_L)
    mega_encoder_positions_aligned_vel2_con_S = filter_sublists_by_length(mega_encoder_positions_aligned_vel2_con_S)
    mega_encoder_positions_aligned_vel1_chem_S = filter_sublists_by_length(mega_encoder_positions_aligned_vel1_chem_S)
    mega_encoder_positions_aligned_vel1_chem_L = filter_sublists_by_length(mega_encoder_positions_aligned_vel1_chem_L)
    mega_encoder_positions_aligned_vel2_chem_L = filter_sublists_by_length(mega_encoder_positions_aligned_vel2_chem_L)
    mega_encoder_positions_aligned_vel2_chem_S = filter_sublists_by_length(mega_encoder_positions_aligned_vel2_chem_S)


    mean_mega_encoder_positions_aligned_vel1_con_S = np.mean(mega_encoder_positions_aligned_vel1_con_S, axis=0)
    mean_mega_encoder_positions_aligned_vel1_con_L = np.mean(mega_encoder_positions_aligned_vel1_con_L, axis=0)
    mean_mega_encoder_positions_aligned_vel2_con_L = np.mean(mega_encoder_positions_aligned_vel2_con_L, axis=0)
    mean_mega_encoder_positions_aligned_vel2_con_S = np.mean(mega_encoder_positions_aligned_vel2_con_S, axis=0)

    mean_mega_encoder_positions_aligned_vel1_chem_S = np.mean(mega_encoder_positions_aligned_vel1_chem_S, axis=0)
    mean_mega_encoder_positions_aligned_vel1_chem_L = np.mean(mega_encoder_positions_aligned_vel1_chem_L, axis=0)
    mean_mega_encoder_positions_aligned_vel2_chem_L = np.mean(mega_encoder_positions_aligned_vel2_chem_L, axis=0)
    mean_mega_encoder_positions_aligned_vel2_chem_S = np.mean(mega_encoder_positions_aligned_vel2_chem_S, axis=0)

    mean_mega_encoder_positions_aligned_vel2_con_all = np.mean(mega_encoder_positions_aligned_vel2_con_all, axis=0)
    mean_mega_encoder_positions_aligned_vel1_con_all = np.mean(mega_encoder_positions_aligned_vel1_con_all, axis=0)
    mean_mega_encoder_positions_aligned_vel2_chem_all = np.mean(mega_encoder_positions_aligned_vel2_chem_all, axis=0)
    mean_mega_encoder_positions_aligned_vel1_chem_all = np.mean(mega_encoder_positions_aligned_vel1_chem_all, axis=0)

    fig2, axs2 = plt.subplots(nrows= 3, ncols= 2,figsize =(10,10))
    fig2.suptitle(subject + '\n' + 'Superimposed averaged rewarded Mega session' + '\n')

    # Plotting
    axs2[0,0].plot(encoder_times - 0.3,mean_mega_encoder_positions_aligned_vel1_con_all, color = 'k', label = 'control')
    if mean_mega_encoder_positions_aligned_vel1_chem_all.size > 1:
        axs2[0,0].plot(encoder_times - 0.3,mean_mega_encoder_positions_aligned_vel1_chem_all, color = 'r', label = 'chemo')
    axs2[0,0].set_title('aligned all trials on onset1')
    axs2[0,0].set_xlabel('time (s)')
    axs2[0,0].set_ylabel('joystick deflection (deg)')
    axs2[0,0].set_xlim(-0.5,1.5)
    axs2[0,0].axvline(x = 0, color = 'r', linestyle='--')
    axs2[0,0].legend()
    axs2[0,0].spines['top'].set_visible(False)
    axs2[0,0].spines['right'].set_visible(False)

    axs2[0,1].plot(encoder_times2 - 0.5 ,mean_mega_encoder_positions_aligned_vel2_con_all, color = 'k', label = 'control')
    if mean_mega_encoder_positions_aligned_vel2_chem_all.size > 1:
        axs2[0,1].plot(encoder_times2 - 0.5 ,mean_mega_encoder_positions_aligned_vel2_chem_all, color = 'r', label = 'chemo')
    axs2[0,1].set_title('aligned all trials on onset2')
    axs2[0,1].set_xlabel('time (s)')
    axs2[0,1].set_ylabel('joystick deflection (deg)')
    axs2[0,1].set_xlim(-0.5,1.5)
    axs2[0,1].axvline(x = 0, color = 'r', linestyle='--')
    axs2[0,1].legend()
    axs2[0,1].spines['top'].set_visible(False)
    axs2[0,1].spines['right'].set_visible(False)

    axs2[1,0].plot(encoder_times - 0.3,mean_mega_encoder_positions_aligned_vel1_con_S, color = 'k', label = 'control')
    if mean_mega_encoder_positions_aligned_vel1_chem_S.size > 1:
        axs2[1,0].plot(encoder_times - 0.3,mean_mega_encoder_positions_aligned_vel1_chem_S, color = 'r', label = 'chemo')
    axs2[1,0].set_title('aligned short trials on onset1')
    axs2[1,0].set_xlabel('time (s)')
    axs2[1,0].set_ylabel('joystick deflection (deg)')
    axs2[1,0].set_xlim(-0.5,1.5)
    axs2[1,0].axvline(x = 0, color = 'r', linestyle='--')
    axs2[1,0].legend()
    axs2[1,0].spines['top'].set_visible(False)
    axs2[1,0].spines['right'].set_visible(False)

    axs2[1,1].plot(encoder_times2 - 0.5,mean_mega_encoder_positions_aligned_vel2_con_S,color = 'k', label = 'control')
    if mean_mega_encoder_positions_aligned_vel2_chem_S.size > 1:
        axs2[1,1].plot(encoder_times2 - 0.5,mean_mega_encoder_positions_aligned_vel2_chem_S,color = 'r', label = 'chemo')
    axs2[1,1].set_title('aligned short trials on onset2')
    axs2[1,1].set_xlabel('time (s)')
    axs2[1,1].set_ylabel('joystick deflection (deg)')
    axs2[1,1].set_xlim(-0.5,1.5)
    axs2[1,1].axvline(x = 0, color = 'r', linestyle='--')
    axs2[1,1].legend()
    axs2[1,1].spines['top'].set_visible(False)
    axs2[1,1].spines['right'].set_visible(False)

    axs2[2,0].plot(encoder_times - 0.3,mean_mega_encoder_positions_aligned_vel1_con_L, color = 'k', label = 'control')
    if mean_mega_encoder_positions_aligned_vel1_chem_L.size > 1:
        axs2[2,0].plot(encoder_times - 0.3,mean_mega_encoder_positions_aligned_vel1_chem_L, color = 'r', label = 'chemo')
    axs2[2,0].set_title('aligned long trials on onset1')
    axs2[2,0].set_xlabel('time (s)')
    axs2[2,0].set_ylabel('joystick deflection (deg)')
    axs2[2,0].set_xlim(-0.5,1.5)
    axs2[2,0].axvline(x = 0, color = 'r', linestyle='--')
    axs2[2,0].legend()
    axs2[2,0].spines['top'].set_visible(False)
    axs2[2,0].spines['right'].set_visible(False)

    axs2[2,1].plot(encoder_times2 - 0.5,mean_mega_encoder_positions_aligned_vel2_con_L, color = 'k', label = 'control')
    if mean_mega_encoder_positions_aligned_vel2_chem_L.size > 1:
        axs2[2,1].plot(encoder_times2 - 0.5,mean_mega_encoder_positions_aligned_vel2_chem_L, color = 'r', label = 'chemo')
    axs2[2,1].set_title('aligned long trials on onset2')
    axs2[2,1].set_xlabel('time (s)')
    axs2[2,1].set_ylabel('joystick deflection (deg)')
    axs2[2,1].set_xlim(-0.5,1.5)
    axs2[2,1].axvline(x = 0, color = 'r', linestyle='--')
    axs2[2,1].legend()
    axs2[2,1].spines['top'].set_visible(False)
    axs2[2,1].spines['right'].set_visible(False)

    fig2.tight_layout()

    os.makedirs(output_figs_dir, exist_ok = True)
    os.makedirs(output_imgs_dir, exist_ok = True)

    pdf_path = os.path.join(output_figs_dir, subject + '_superimposed_averaged_velocity_onset.pdf')
    # Save both plots into a single PDF file with each on a separate page
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)

    # Close the figures after saving to free up memory
    plt.close(fig1)
    plt.close(fig2)
