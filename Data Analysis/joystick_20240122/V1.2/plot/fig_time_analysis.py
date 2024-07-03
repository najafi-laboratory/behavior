import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
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

def plot_time_analysis(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):
    
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

    max_sessions=10
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

    mean_targetTime_press1 = []
    mean_targetTime_press2_reward = []
    mean_targetTime_press2_early = []
    std_targetTime_press1 = []
    std_targetTime_press2_reward = []
    std_targetTime_press2_early = []
    mean_amp_press1 = []
    mean_amp_press2_reward = []
    mean_amp_press2_early = []
    std_amp_press1 = []
    std_amp_press2_reward = []
    std_amp_press2_early = []
    std_interval_early = []
    mean_interval_early = []
    std_interval_reward = []
    mean_interval_reward = []
    # Velocity
    std_target1_velocity = []
    mean_target1_velocity = []

    std_target2_velocity_reward = []
    mean_target2_velocity_reward = []
    std_target2_velocity_early = []
    mean_target2_velocity_early = []

    std_interval_velocity_reward = []
    mean_interval_velocity_reward = []

    std_interval_velocity_early = []
    mean_interval_velocity_early = []
    isSelfTimedMode  = process_matrix (session_data['isSelfTimedMode'])

    chemo_labels = session_data['chemo']
    for i in range(0 , len(chemo_labels)):
            if chemo_labels[i] == 1:
                dates[i] = dates[i] + '(chemo)'

    dates = deduplicate_chemo(dates)
    offset = 0.15
    numeric_dates = np.arange(len(dates))

    if any(0 in row for row in isSelfTimedMode):
        print('Visually Guided')
        fig1,axs1 = plt.subplots(nrows=4, ncols=2, figsize=(10, 20))
        fig1.suptitle(subject + ' \n ' + 'Time analysis over sessions' + '\n')


        for i in range(0 , len(session_id)):
            TrialOutcomes = session_data['outcomes'][i]
            
            onset_press1 = []
            onset_press2 = []
            targetTime_press1 = []
            targetTime_press2 = []
            targetTime_press2_reward = []
            targetTime_press2_early = []
            amp_press1 = []
            amp_press2 = []
            amp_press2_reward = []
            amp_press2_early = []
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
            j = 0
            
            # We have Raw data and extract every thing from it (Times)
            raw_data = session_data['raw'][i]
            session_date = dates[i][2:]
            
            # Creating figures for each session
            fig, axs = plt.subplots(nrows=8, ncols=2, figsize=(10, 32))   # should be changed
            fig.suptitle(subject + ' - ' + session_date + ' \n ' + 'Time analysis for Reward, Earlypress2 and Didnotpress2 trials' + '\n')
            
            print('time analysis of session:' + session_date)
            # The loop for each session
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
                    
                    if base_line >= 0.8:
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

                    j = j+1
                # For DidNotPress2 (WE have the press one only)
                elif TrialOutcomes[trial] == 'DidNotPress2':
                    encoder_data = raw_data['EncoderData'][trial]              
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                    ####
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
                    VisDetect2 = int(trial_states['VisDetect2'][0]*1000) #needed
                    VisDetect1 = int(trial_states['VisDetect1'][0]*1000) #needed
                    VisualStimulus1 = int(trial_states['VisualStimulus1'][0]*1000) #needed
                    # Get the base line for the 500 datapoints before the vis1
                    base_line = encoder_positions_aligned_vis1[:VisualStimulus1]
                    base_line = base_line[~np.isnan(base_line)]
                    base_line = np.mean(base_line)
                    
                    if base_line >= 0.8:
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
                    amp_press1.append(np.max(encoder_positions_aligned_vis1[rotatory1:VisDetect2]) - base_line)
                    
                    velocity = np.gradient(encoder_positions_aligned_vis1,encoder_times_vis1)
                    velocity = savgol_filter(velocity, window_length=40, polyorder=1)
                    #### Velocity onset
                    on1_vel = velocity_onset(velocity,int(VisDetect1),rotatory1) # type: ignore
                    target1_velocity.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                    
                    j = j + 1
                # For EarlyPress2, We have both onset1 and 2
                elif TrialOutcomes[trial] == 'EarlyPress2':
                    encoder_data = raw_data['EncoderData'][trial]              
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                    ####
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
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
                    
                    if base_line >= 0.8:
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
                    amp_press1.append(np.max(encoder_positions_aligned_vis1[rotatory1:LeverRetract1]) - base_line)
                    

                    ####### Calculations for vis 2
                    threshold_press2 = base_line + 0.25
                    amp2 = np.argmax(encoder_positions_aligned_vis1[punish1:punish2])
                    amp2 = amp2 + punish1
                    onset2_position = first_touch_onset(encoder_positions_aligned_vis1, threshold_press2,PreVis2Delay,amp2)
                    
                    # Target 2 to Onset_press 2
                    targetTime_press2.append(np.abs(punish1 - onset2_position)/1000) # type: ignore
                    targetTime_press2_early.append(np.abs(punish1 - onset2_position)/1000) # type: ignore
                    ##### press2
                    amp_press2.append(np.max(encoder_positions_aligned_vis1[punish1:punish2]) - base_line)
                    amp_press2_early.append(np.max(encoder_positions_aligned_vis1[punish1:punish2]) - base_line)
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
                    target1_velocity.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                    target2_velocity.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                    target2_velocity_early.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                    intervall_vel = on2_vel - on1_vel # type: ignore
                    interval_velocity.append(intervall_vel/1000)
                    interval_velocity_early.append(intervall_vel/1000)
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
            
            axs[2,0].hist(target1_velocity, bins=100, rwidth= 0.5)
            axs[2,0].set_title('onset1_velocity to target1 histogram')
            axs[2,0].set_xlabel('Interval Time (s)')
            axs[2,0].set_ylabel('Count of Rewarded Trials')
            
            axs[2,0].spines['top'].set_visible(False)
            axs[2,0].spines['right'].set_visible(False)
                    
            axs[3,0].plot(target1_velocity) 
            axs[3,0].set_title('onset1_velocity to target1 for each trial')
            axs[3,0].set_xlabel('Trials')
            axs[3,0].set_ylabel('Interval Time (s)')
            
            axs[3,0].spines['top'].set_visible(False)
            axs[3,0].spines['right'].set_visible(False)
            
            axs[2,1].hist(target2_velocity, bins=100, rwidth= 0.5)
            axs[2,1].set_title('onset2_velocity to target2 histogram')
            axs[2,1].set_xlabel('Interval Time (s)')
            axs[2,1].set_ylabel('Count of Rewarded Trials')
            
            axs[2,1].spines['top'].set_visible(False)
            axs[2,1].spines['right'].set_visible(False)

            axs[3,1].plot(target2_velocity) 
            axs[3,1].set_title('onset2_velocity to target2 for each trial')
            axs[3,1].set_xlabel('Trials')
            axs[3,1].set_ylabel('Interval Time (s)')
            
            axs[3,1].spines['top'].set_visible(False)
            axs[3,1].spines['right'].set_visible(False)
                    
            axs[4,0].hist(amp_press1, bins=20, rwidth= 0.5)
            axs[4,0].set_title('amp difference baseline and peak1 histogram')
            axs[4,0].set_xlabel('Amplitude (deg)')
            axs[4,0].set_ylabel('Count of Rewarded Trials')
            
            axs[4,0].spines['top'].set_visible(False)
            axs[4,0].spines['right'].set_visible(False)

            axs[4,1].hist(amp_press2, bins=20, rwidth= 0.5) 
            axs[4,1].set_title('amp difference baseline and peak2 histogram')
            axs[4,1].set_xlabel('Amplitude (deg)')
            axs[4,1].set_ylabel('Count of Rewarded Trials')
            
            axs[4,1].spines['top'].set_visible(False)
            axs[4,1].spines['right'].set_visible(False)
        
            axs[5,0].plot(amp_press1) 
            axs[5,0].set_title('amp difference baseline and peak1 for each trial')
            axs[5,0].set_xlabel('Trials')
            axs[5,0].set_ylabel('Amplitude (deg)')
            
            axs[5,0].spines['top'].set_visible(False)
            axs[5,0].spines['right'].set_visible(False)
        
            axs[5,1].plot(amp_press2) 
            axs[5,1].set_title('amp difference baseline and peak2 for each trial')
            axs[5,1].set_xlabel('Trials')
            axs[5,1].set_ylabel('Amplitude (deg)')
            
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
        
            axs[6,1].hist(interval_velocity,bins=100, rwidth= 0.5)
            axs[6,1].set_title('Time Interval between onset1 & 2 Velocity histogram')
            axs[6,1].set_xlabel('time (s)')
            axs[6,1].set_ylabel('Count of Rewarded Trials')
            
            axs[6,1].spines['top'].set_visible(False)
            axs[6,1].spines['right'].set_visible(False)
            
            axs[7,1].plot(interval_velocity) 
            axs[7,1].set_title('Time Interval between onset1 & 2 Velocity')
            axs[7,1].set_xlabel('Trials')
            axs[7,1].set_ylabel('Time Interval (s)')
            
            axs[7,1].spines['top'].set_visible(False)
            axs[7,1].spines['right'].set_visible(False)
            
            fig.tight_layout()
            
            std_error_1 = np.std(targetTime_press1, ddof=1) / np.sqrt(len(targetTime_press1))
            std_targetTime_press1.append(std_error_1)
            mean_targetTime_press1.append(np.mean(targetTime_press1))
            
            std_error_2 = np.std(targetTime_press2_reward, ddof=1) / np.sqrt(len(targetTime_press2_reward))
            std_targetTime_press2_reward.append(std_error_2)
            mean_targetTime_press2_reward.append(np.mean(targetTime_press2_reward))
            
            std_error_13 = np.std(targetTime_press2_early, ddof=1) / np.sqrt(len(targetTime_press2_early))
            std_targetTime_press2_early.append(std_error_13)
            mean_targetTime_press2_early.append(np.mean(targetTime_press2_early))
            
            std_error_3 = np.std(amp_press1, ddof=1) / np.sqrt(len(amp_press1))
            std_amp_press1.append(std_error_3)
            mean_amp_press1.append(np.mean(amp_press1))
            
            std_error_4 = np.std(amp_press2_reward, ddof=1) / np.sqrt(len(amp_press2_reward))
            std_amp_press2_reward.append(std_error_4)
            mean_amp_press2_reward.append(np.mean(amp_press2_reward))
            
            std_error_11 = np.std(amp_press2_early, ddof=1) / np.sqrt(len(amp_press2_early))
            std_amp_press2_early.append(std_error_11)
            mean_amp_press2_early.append(np.mean(amp_press2_early))
            
            std_error_5 = np.std(interval_reward, ddof=1) / np.sqrt(len(interval_reward))
            std_interval_reward.append(std_error_5)
            mean_interval_reward.append(np.mean(interval_reward))
            
            std_error_10 = np.std(interval_early, ddof=1) / np.sqrt(len(interval_early))
            std_interval_early.append(std_error_10)
            mean_interval_early.append(np.mean(interval_early))
            
            std_error_6  = np.std(target1_velocity, ddof=1) / np.sqrt(len(target1_velocity))
            std_target1_velocity.append(std_error_6)
            mean_target1_velocity.append(np.mean(target1_velocity))
            
            std_error_7  = np.std(target2_velocity_reward, ddof=1) / np.sqrt(len(target2_velocity_reward))
            std_target2_velocity_reward.append(std_error_7)
            mean_target2_velocity_reward.append(np.mean(target2_velocity_reward))
            
            std_error_12  = np.std(target2_velocity_early, ddof=1) / np.sqrt(len(target2_velocity_early))
            std_target2_velocity_early.append(std_error_12)
            mean_target2_velocity_early.append(np.mean(target2_velocity_early))
            
            std_error_8  = np.std(interval_velocity_reward, ddof=1) / np.sqrt(len(interval_velocity_reward))
            std_interval_velocity_reward.append(std_error_8)
            mean_interval_velocity_reward.append(np.mean(interval_velocity_reward))
            
            std_error_9  = np.std(interval_velocity_early, ddof=1) / np.sqrt(len(interval_velocity_early))
            std_interval_velocity_early.append(std_error_9)
            mean_interval_velocity_early.append(np.mean(interval_velocity_early))
            
            output_figs_dir = output_dir_onedrive + subject + '/'    
            output_imgs_dir = output_dir_local + subject + '/time analysis_imgs/'    
            os.makedirs(output_figs_dir, exist_ok = True)
            os.makedirs(output_imgs_dir, exist_ok = True)
            fig.savefig(output_figs_dir + subject + '_time_analysis_'+ session_date +'.pdf', dpi=300)
            fig.savefig(output_imgs_dir + subject + '_time_analysis_'+ session_date +'.png', dpi=300)
            plt.close(fig)

        # Plotting for anlyze over all sessions
        axs1[0,0].errorbar(numeric_dates, mean_targetTime_press1, yerr=std_targetTime_press1, fmt='o', capsize=4, color = 'blue')
        axs1[0,0].set_title('onset1_postion to target1')
        axs1[0,0].set_ylabel('Mean +/- SEM of time Interval (s)')
        axs1[0,0].set_xlabel('Sessions')
        axs1[0,0].set_xticks(numeric_dates)
        axs1[0,0].set_xticklabels(dates, rotation=90, ha = 'center')
        
        axs1[0,0].spines['top'].set_visible(False)
        axs1[0,0].spines['right'].set_visible(False)

        axs1[0,1].errorbar(numeric_dates, mean_targetTime_press2_reward, yerr=std_targetTime_press2_reward, fmt='o', capsize=4, color= 'blue', label = 'Rewarded')
        axs1[0,1].errorbar(numeric_dates + offset, mean_targetTime_press2_early, yerr=std_targetTime_press2_early, fmt='o', capsize=4, color= 'green', label = 'Early_Press')
        axs1[0,1].set_title('onset2_position to target2')
        axs1[0,1].set_ylabel('Mean +/- SEM of time Interval (s)')
        axs1[0,1].set_xlabel('Sessions')
        axs1[0,1].set_xticks(numeric_dates)
        axs1[0,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[0,1].legend()
        axs1[0,1].spines['top'].set_visible(False)
        axs1[0,1].spines['right'].set_visible(False)
        
        axs1[1,0].errorbar(numeric_dates, mean_target1_velocity, yerr=std_target1_velocity, fmt='o', capsize=4, color = 'blue')
        axs1[1,0].set_title('onset1_velocity to target1')
        axs1[1,0].set_ylabel('Mean +/- SEM of time Interval (s)')
        axs1[1,0].set_xlabel('Sessions')
        axs1[1,0].set_xticks(numeric_dates)
        axs1[1,0].set_xticklabels(dates, rotation=90, ha = 'center')
        
        axs1[1,0].spines['top'].set_visible(False)
        axs1[1,0].spines['right'].set_visible(False)

        axs1[1,1].errorbar(numeric_dates, mean_target2_velocity_reward, yerr=std_target2_velocity_reward, fmt='o', capsize=4, color= 'blue', label = 'Rewarded')
        axs1[1,1].errorbar(numeric_dates + offset, mean_target2_velocity_early, yerr=std_target2_velocity_early, fmt='o', capsize=4, color= 'green', label = 'Early_Press')
        axs1[1,1].set_title('onset2_velocity to target2')
        axs1[1,1].set_ylabel('Mean +/- SEM of time Interval (s)')
        axs1[1,1].set_xlabel('Sessions')
        axs1[1,1].set_xticks(numeric_dates)
        axs1[1,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[1,1].legend()
        axs1[1,1].spines['top'].set_visible(False)
        axs1[1,1].spines['right'].set_visible(False)
        
        axs1[2,0].errorbar(numeric_dates, mean_amp_press1, yerr=std_amp_press1, fmt='o', capsize=4, color = 'blue')
        axs1[2,0].set_title('Amplitude of Press1 (position)')
        axs1[2,0].set_ylabel('Mean +/- SEM of press amplitude (deg)')
        axs1[2,0].set_xlabel('Sessions')
        axs1[2,0].set_xticks(numeric_dates)
        axs1[2,0].set_xticklabels(dates, rotation=90, ha = 'center')
        
        axs1[2,0].spines['top'].set_visible(False)
        axs1[2,0].spines['right'].set_visible(False)

        axs1[2,1].errorbar(numeric_dates, mean_amp_press2_reward, yerr=std_amp_press2_reward, fmt='o', capsize=4, color= 'blue', label = 'Rewarded')
        axs1[2,1].errorbar(numeric_dates + offset, mean_amp_press2_early, yerr=std_amp_press2_early, fmt='o', capsize=4, color= 'green', label = 'Early_Press')
        axs1[2,1].set_title('Amplitude of Press2 (position)')
        axs1[2,1].set_ylabel('Mean +/- SEM of press amplitude (deg)')
        axs1[2,1].set_xlabel('Sessions')
        axs1[2,1].set_xticks(numeric_dates)
        axs1[2,1].set_xticklabels(session_data['dates'], rotation=90, ha = 'center')
        axs1[2,1].legend()

        axs1[2,1].spines['top'].set_visible(False)
        axs1[2,1].spines['right'].set_visible(False)
        
        axs1[3,0].errorbar(numeric_dates, mean_interval_reward, yerr=std_interval_reward, fmt='o', capsize=4, color= 'blue', label = 'Rewarded')
        axs1[3,0].errorbar(numeric_dates + offset, mean_interval_early, yerr=std_interval_early, fmt='o', capsize=4, color= 'green', label = 'Early_Press')
        axs1[3,0].set_title('Interval time between onset1 & 2 Postion')
        axs1[3,0].set_ylabel('Mean +/- SEM of Interval (s)')
        axs1[3,0].set_xlabel('Sessions')
        axs1[3,0].set_xticks(numeric_dates)
        axs1[3,0].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[3,0].legend()
        
        axs1[3,0].spines['top'].set_visible(False)
        axs1[3,0].spines['right'].set_visible(False)
        
        axs1[3,1].errorbar(numeric_dates, mean_interval_velocity_reward, yerr=std_interval_velocity_reward, fmt='o', capsize=4, color= 'blue', label = 'Rewarded')
        axs1[3,1].errorbar(numeric_dates + offset, mean_interval_velocity_early, yerr=std_interval_velocity_early, fmt='o', capsize=4, color= 'green', label = 'Early_Press')
        axs1[3,1].set_title('Interval time between onset1 & 2 Velocity')
        axs1[3,1].set_ylabel('Mean +/- SEM of Interval (s)')
        axs1[3,1].set_xlabel('Sessions')
        axs1[3,0].set_xticks(numeric_dates)
        axs1[3,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[3,1].legend()
        
        axs1[3,1].spines['top'].set_visible(False)
        axs1[3,1].spines['right'].set_visible(False)
        
        fig1.tight_layout()
        
        os.makedirs(output_figs_dir, exist_ok = True)
        os.makedirs(output_imgs_dir, exist_ok = True)
        fig1.savefig(output_figs_dir + subject + '_time_analysis_oversessions.pdf', dpi=300)
        fig1.savefig(output_imgs_dir + subject + '_time_analysis_imgs_oversessions.png', dpi=300)
        plt.close(fig1)
        
    else:
        print('selftime')
        
        fig1,axs1 = plt.subplots(nrows=4, ncols=2, figsize=(10, 15))
        fig1.suptitle(subject + ' \n ' + 'Time analysis over sessions' + '\n')


        for i in range(0 , len(session_id)):
            TrialOutcomes = session_data['outcomes'][i]
            reward_count = TrialOutcomes.count('Reward')
            onset_press1 = []
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
            interval = []
            interval_reward = []
            interval_early = []
            target1_velocity = []
            target2_velocity = []
            target2_velocity_reward = []
            target2_velocity_early = []
            interval_velocity = []
            interval_velocity_reward = []
            interval_velocity_early = []
            j = 0
            
            # We have Raw data and extract every thing from it (Times)
            raw_data = session_data['raw'][i]
            session_date = dates[i][2:]
            
            # Creating figures for each session
            fig, axs = plt.subplots(nrows=8, ncols=2, figsize=(10, 32))   # should be changed
            fig.suptitle(subject + ' - ' + session_date + ' \n ' + 'Time analysis for Reward trials' + '\n')
            
            print('analysis of session:' + session_date)
            # The loop for each session
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
                    
                    if base_line >= 0.8:
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
                    
                    j = j+1            
                # For DidNotPress2 (WE have the press one only)
                elif TrialOutcomes[trial] == 'DidNotPress2':
                    encoder_data = raw_data['EncoderData'][trial]              
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                    ####
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
                    VisualStimulus1 = int(trial_states['VisualStimulus1'][0]*1000) #needed
                    WaitForPress2 = int(trial_states['WaitForPress2'][0]*1000) #needed
                    VisDetect1 = int(trial_states['VisDetect1'][0]*1000) #needed
                    # Get the base line for the 500 datapoints before the vis1
                    base_line = encoder_positions_aligned_vis1[:VisualStimulus1]
                    base_line = base_line[~np.isnan(base_line)]
                    base_line = np.mean(base_line)
                    
                    if base_line >= 0.8:
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
                    amp_press1.append(np.max(encoder_positions_aligned_vis1[rotatory1:WaitForPress2]) - base_line)
                    
                    encoder_positions_aligned_vis1 = savgol_filter(encoder_positions_aligned_vis1, window_length=40, polyorder=3)
                    velocity = np.gradient(encoder_positions_aligned_vis1,encoder_times_vis1)
                    velocity = savgol_filter(velocity, window_length=40, polyorder=1)
                    #### Velocity onset
                    on1_vel = velocity_onset(velocity,int(VisDetect1),rotatory1) # type: ignore
                    target1_velocity.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                    
                    j = j + 1
                # For EarlyPress2, We have both onset1 and 2
                elif TrialOutcomes[trial] == 'EarlyPress2':
                    encoder_data = raw_data['EncoderData'][trial]              
                    times = encoder_data['Times']
                    positions = encoder_data['Positions']
                    encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                    encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                    ####
                    trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                    trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
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
                    
                    if base_line >= 0.8:
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
                    amp_press1.append(np.max(encoder_positions_aligned_vis1[rotatory1:LeverRetract1]) - base_line)
                    

                    ####### Calculations for vis 2
                    threshold_press2 = base_line + 0.25
                    amp2 = np.argmax(encoder_positions_aligned_vis1[punish1:punish2])
                    amp2 = amp2 + punish1
                    onset2_position = first_touch_onset(encoder_positions_aligned_vis1, threshold_press2,LeverRetract1,amp2)
                    
                    # Target 2 to Onset_press 2
                    targetTime_press2.append(np.abs(punish1 - onset2_position)/1000) # type: ignore
                    targetTime_press2_early.append(np.abs(punish1 - onset2_position)/1000) # type: ignore
                    ##### press2
                    amp_press2.append(np.max(encoder_positions_aligned_vis1[punish1:punish2]) - base_line)
                    amp_press2_early.append(np.max(encoder_positions_aligned_vis1[punish1:punish2]) - base_line)
                    
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
                    target1_velocity.append(np.abs(rotatory1 - on1_vel)/1000) # type: ignore
                    target2_velocity.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                    target2_velocity_early.append(np.abs(punish1 - on2_vel)/1000) # type: ignore
                    intervall_vel = on2_vel - on1_vel # type: ignore
                    interval_velocity.append(intervall_vel/1000)
                    interval_velocity_early.append(intervall_vel/1000)
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
            
            axs[2,0].hist(target1_velocity, bins=100, rwidth= 0.5)
            axs[2,0].set_title('onset1_velocity to target1 histogram')
            axs[2,0].set_xlabel('Interval Time (s)')
            axs[2,0].set_ylabel('Count of Rewarded Trials')
            
            axs[2,0].spines['top'].set_visible(False)
            axs[2,0].spines['right'].set_visible(False)
                    
            axs[3,0].plot(target1_velocity) 
            axs[3,0].set_title('onset1_velocity to target1 for each trial')
            axs[3,0].set_xlabel('Trials')
            axs[3,0].set_ylabel('Interval Time (s)')
            
            axs[3,0].spines['top'].set_visible(False)
            axs[3,0].spines['right'].set_visible(False)
            
            axs[2,1].hist(target2_velocity, bins=100, rwidth= 0.5)
            axs[2,1].set_title('onset2_velocity to target2 histogram')
            axs[2,1].set_xlabel('Interval Time (s)')
            axs[2,1].set_ylabel('Count of Rewarded Trials')
            
            axs[2,1].spines['top'].set_visible(False)
            axs[2,1].spines['right'].set_visible(False)

            axs[3,1].plot(target2_velocity) 
            axs[3,1].set_title('onset2_velocity to target2 for each trial')
            axs[3,1].set_xlabel('Trials')
            axs[3,1].set_ylabel('Interval Time (s)')
            
            axs[3,1].spines['top'].set_visible(False)
            axs[3,1].spines['right'].set_visible(False)
                    
            axs[4,0].hist(amp_press1, bins=20, rwidth= 0.5)
            axs[4,0].set_title('amp difference baseline and peak1 histogram')
            axs[4,0].set_xlabel('Amplitude (deg)')
            axs[4,0].set_ylabel('Count of Rewarded Trials')
            
            axs[4,0].spines['top'].set_visible(False)
            axs[4,0].spines['right'].set_visible(False)

            axs[4,1].hist(amp_press2, bins=20, rwidth= 0.5) 
            axs[4,1].set_title('amp difference baseline and peak2 histogram')
            axs[4,1].set_xlabel('Amplitude (deg)')
            axs[4,1].set_ylabel('Count of Rewarded Trials')
            
            axs[4,1].spines['top'].set_visible(False)
            axs[4,1].spines['right'].set_visible(False)
        
            axs[5,0].plot(amp_press1) 
            axs[5,0].set_title('amp difference baseline and peak1 for each trial')
            axs[5,0].set_xlabel('Trials')
            axs[5,0].set_ylabel('Amplitude (deg)')
            
            axs[5,0].spines['top'].set_visible(False)
            axs[5,0].spines['right'].set_visible(False)
        
            axs[5,1].plot(amp_press2) 
            axs[5,1].set_title('amp difference baseline and peak2 for each trial')
            axs[5,1].set_xlabel('Trials')
            axs[5,1].set_ylabel('Amplitude (deg)')
            
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
        
            axs[6,1].hist(interval_velocity,bins=100, rwidth= 0.5)
            axs[6,1].set_title('Time Interval between onset1 & 2 Velocity histogram')
            axs[6,1].set_xlabel('time (s)')
            axs[6,1].set_ylabel('Count of Rewarded Trials')
            
            axs[6,1].spines['top'].set_visible(False)
            axs[6,1].spines['right'].set_visible(False)
            
            axs[7,1].plot(interval_velocity) 
            axs[7,1].set_title('Time Interval between onset1 & 2 Velocity')
            axs[7,1].set_xlabel('Trials')
            axs[7,1].set_ylabel('Time Interval (s)')
            
            axs[7,1].spines['top'].set_visible(False)
            axs[7,1].spines['right'].set_visible(False)
            
            fig.tight_layout()
            
            std_error_1 = np.std(targetTime_press1, ddof=1) / np.sqrt(len(targetTime_press1))
            std_targetTime_press1.append(std_error_1)
            mean_targetTime_press1.append(np.mean(targetTime_press1))
            
            std_error_2 = np.std(targetTime_press2_reward, ddof=1) / np.sqrt(len(targetTime_press2_reward))
            std_targetTime_press2_reward.append(std_error_2)
            mean_targetTime_press2_reward.append(np.mean(targetTime_press2_reward))
            
            std_error_13 = np.std(targetTime_press2_early, ddof=1) / np.sqrt(len(targetTime_press2_early))
            std_targetTime_press2_early.append(std_error_13)
            mean_targetTime_press2_early.append(np.mean(targetTime_press2_early))
            
            std_error_3 = np.std(amp_press1, ddof=1) / np.sqrt(len(amp_press1))
            std_amp_press1.append(std_error_3)
            mean_amp_press1.append(np.mean(amp_press1))
            
            std_error_4 = np.std(amp_press2_reward, ddof=1) / np.sqrt(len(amp_press2_reward))
            std_amp_press2_reward.append(std_error_4)
            mean_amp_press2_reward.append(np.mean(amp_press2_reward))
            
            std_error_11 = np.std(amp_press2_early, ddof=1) / np.sqrt(len(amp_press2_early))
            std_amp_press2_early.append(std_error_11)
            mean_amp_press2_early.append(np.mean(amp_press2_early))
            
            std_error_5 = np.std(interval_reward, ddof=1) / np.sqrt(len(interval_reward))
            std_interval_reward.append(std_error_5)
            mean_interval_reward.append(np.mean(interval_reward))
            
            std_error_10 = np.std(interval_early, ddof=1) / np.sqrt(len(interval_early))
            std_interval_early.append(std_error_10)
            mean_interval_early.append(np.mean(interval_early))
            
            std_error_6  = np.std(target1_velocity, ddof=1) / np.sqrt(len(target1_velocity))
            std_target1_velocity.append(std_error_6)
            mean_target1_velocity.append(np.mean(target1_velocity))
            
            std_error_7  = np.std(target2_velocity_reward, ddof=1) / np.sqrt(len(target2_velocity_reward))
            std_target2_velocity_reward.append(std_error_7)
            mean_target2_velocity_reward.append(np.mean(target2_velocity_reward))
            
            std_error_12  = np.std(target2_velocity_early, ddof=1) / np.sqrt(len(target2_velocity_early))
            std_target2_velocity_early.append(std_error_12)
            mean_target2_velocity_early.append(np.mean(target2_velocity_early))
            
            std_error_8  = np.std(interval_velocity_reward, ddof=1) / np.sqrt(len(interval_velocity_reward))
            std_interval_velocity_reward.append(std_error_8)
            mean_interval_velocity_reward.append(np.mean(interval_velocity_reward))
            
            std_error_9  = np.std(interval_velocity_early, ddof=1) / np.sqrt(len(interval_velocity_early))
            std_interval_velocity_early.append(std_error_9)
            mean_interval_velocity_early.append(np.mean(interval_velocity_early))
            
            output_figs_dir = output_dir_onedrive + subject + '/'    
            output_imgs_dir = output_dir_local + subject + '/time analysis_imgs/' 
            # output_data_dir  = output_dir_local + subject + 'data of onset'
            os.makedirs(output_figs_dir, exist_ok = True)
            os.makedirs(output_imgs_dir, exist_ok = True)
            fig.savefig(output_figs_dir + subject + '_time_analysis_'+ session_date +'.pdf', dpi=300)
            fig.savefig(output_imgs_dir + subject + '_time_analysis_'+ session_date +'.png', dpi=300)
            plt.close(fig)

        # Plotting for anlyze over all sessions
        axs1[0,0].errorbar(numeric_dates, mean_targetTime_press1, yerr=std_targetTime_press1, fmt='o', capsize=4, color = 'blue')
        axs1[0,0].set_title('onset1_postion to target1')
        axs1[0,0].set_ylabel('Mean +/- SEM of time Interval (s)')
        axs1[0,0].set_xlabel('Sessions')
        axs1[0,0].set_xticks(numeric_dates)
        axs1[0,0].set_xticklabels(dates, rotation=90, ha = 'center')
        
        axs1[0,0].spines['top'].set_visible(False)
        axs1[0,0].spines['right'].set_visible(False)

        axs1[0,1].errorbar(numeric_dates, mean_targetTime_press2_reward, yerr=std_targetTime_press2_reward, fmt='o', capsize=4, color= 'blue', label = 'Rewarded')
        axs1[0,1].errorbar(numeric_dates + offset, mean_targetTime_press2_early, yerr=std_targetTime_press2_early, fmt='o', capsize=4, color= 'green', label = 'Early_Press')
        axs1[0,1].set_title('onset2_position to target2')
        axs1[0,1].set_ylabel('Mean +/- SEM of time Interval (s)')
        axs1[0,1].set_xlabel('Sessions')
        axs1[0,1].set_xticks(numeric_dates)
        axs1[0,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[0,1].legend()
        axs1[0,1].spines['top'].set_visible(False)
        axs1[0,1].spines['right'].set_visible(False)
        
        axs1[1,0].errorbar(numeric_dates, mean_target1_velocity, yerr=std_target1_velocity, fmt='o', capsize=4, color = 'blue')
        axs1[1,0].set_title('onset1_velocity to target1')
        axs1[1,0].set_ylabel('Mean +/- SEM of time Interval (s)')
        axs1[1,0].set_xlabel('Sessions')
        axs1[1,0].set_xticks(numeric_dates)
        axs1[1,0].set_xticklabels(dates, rotation=90, ha = 'center')
        
        axs1[1,0].spines['top'].set_visible(False)
        axs1[1,0].spines['right'].set_visible(False)

        axs1[1,1].errorbar(numeric_dates, mean_target2_velocity_reward, yerr=std_target2_velocity_reward, fmt='o', capsize=4, color= 'blue', label = 'Rewarded')
        axs1[1,1].errorbar(numeric_dates + offset, mean_target2_velocity_early, yerr=std_target2_velocity_early, fmt='o', capsize=4, color= 'green', label = 'Early_Press')
        axs1[1,1].set_title('onset2_velocity to target2')
        axs1[1,1].set_ylabel('Mean +/- SEM of time Interval (s)')
        axs1[1,1].set_xlabel('Sessions')
        axs1[1,1].set_xticks(numeric_dates)
        axs1[1,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[1,1].legend()
        axs1[1,1].spines['top'].set_visible(False)
        axs1[1,1].spines['right'].set_visible(False)
        
        axs1[2,0].errorbar(numeric_dates, mean_amp_press1, yerr=std_amp_press1, fmt='o', capsize=4, color = 'blue')
        axs1[2,0].set_title('Amplitude of Press1 (position)')
        axs1[2,0].set_ylabel('Mean +/- SEM of press amplitude (deg)')
        axs1[2,0].set_xlabel('Sessions')
        axs1[2,0].set_xticks(numeric_dates)
        axs1[2,0].set_xticklabels(dates, rotation=90, ha = 'center')
        
        axs1[2,0].spines['top'].set_visible(False)
        axs1[2,0].spines['right'].set_visible(False)

        axs1[2,1].errorbar(numeric_dates, mean_amp_press2_reward, yerr=std_amp_press2_reward, fmt='o', capsize=4, color= 'blue', label = 'Rewarded')
        axs1[2,1].errorbar(numeric_dates + offset, mean_amp_press2_early, yerr=std_amp_press2_early, fmt='o', capsize=4, color= 'green', label = 'Early_Press')
        axs1[2,1].set_title('Amplitude of Press2 (position)')
        axs1[2,1].set_ylabel('Mean +/- SEM of press amplitude (deg)')
        axs1[2,1].set_xlabel('Sessions')
        axs1[2,1].set_xticks(numeric_dates)
        axs1[2,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[2,1].legend()

        axs1[2,1].spines['top'].set_visible(False)
        axs1[2,1].spines['right'].set_visible(False)
        
        axs1[3,0].errorbar(numeric_dates, mean_interval_reward, yerr=std_interval_reward, fmt='o', capsize=4, color= 'blue', label = 'Rewarded')
        axs1[3,0].errorbar(numeric_dates + offset, mean_interval_early, yerr=std_interval_early, fmt='o', capsize=4, color= 'green', label = 'Early_Press')
        axs1[3,0].set_title('Interval time between onset1 & 2 Postion')
        axs1[3,0].set_ylabel('Mean +/- SEM of Interval (s)')
        axs1[3,0].set_xlabel('Sessions')
        axs1[3,0].set_xticks(numeric_dates)
        axs1[3,0].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[3,0].legend()
        
        axs1[3,0].spines['top'].set_visible(False)
        axs1[3,0].spines['right'].set_visible(False)
        
        axs1[3,1].errorbar(numeric_dates, mean_interval_velocity_reward, yerr=std_interval_velocity_reward, fmt='o', capsize=4, color= 'blue', label = 'Rewarded')
        axs1[3,1].errorbar(numeric_dates + offset, mean_interval_velocity_early, yerr=std_interval_velocity_early, fmt='o', capsize=4, color= 'green', label = 'Early_Press')
        axs1[3,1].set_title('Interval time between onset1 & 2 Velocity')
        axs1[3,1].set_ylabel('Mean +/- SEM of Interval (s)')
        axs1[3,1].set_xlabel('Sessions')
        axs1[3,1].set_xticks(numeric_dates)
        axs1[3,1].set_xticklabels(dates, rotation=90, ha = 'center')
        axs1[3,1].legend()
        
        axs1[3,1].spines['top'].set_visible(False)
        axs1[3,1].spines['right'].set_visible(False)
        
        fig1.tight_layout()
        
        os.makedirs(output_figs_dir, exist_ok = True)
        os.makedirs(output_imgs_dir, exist_ok = True)
        fig1.savefig(output_figs_dir + subject + '_time_analysis_oversessions.pdf', dpi=300)
        fig1.savefig(output_imgs_dir + subject + '_time_analysis_imgs_oversessions.png', dpi=300)
        plt.close(fig1)