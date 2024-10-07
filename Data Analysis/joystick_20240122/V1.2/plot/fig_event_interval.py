import os
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.backends.backend_pdf import PdfPages # type: ignore
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfWriter, PdfReader # type: ignore
from datetime import date
from scipy.signal import savgol_filter # type: ignore
from scipy.signal import find_peaks # type: ignore
import re
import math
from scipy import stats
from matplotlib.lines import Line2D


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


def event_interval(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_id = np.arange(len(outcomes)) + 1
    sess_num = np.zeros((3 , len(session_id)))
    encoder_time_max = 100
    ms_per_s = 1000
    chemo_labels = session_data['chemo']
                
    dates = deduplicate_chemo(dates)            
    isSelfTimedMode  = process_matrix (session_data['isSelfTimedMode'])

    if any(0 in row for row in isSelfTimedMode):
        print('Visually Guided')
    else:
        print('Selftime')
    
    # Plotting sessions
    # Define colors
    black_shades = [(150, 150, 150), (100, 100, 100), (50, 50, 50)]
    red_shades = [(255, 102, 102), (255, 51, 51), (204, 0, 0)]
    skyblue_shades = [(135, 206, 235), (70, 130, 180), (0, 105, 148)]
    # Normalize the colors to [0, 1] range for matplotlib
    black_shades = [tuple(c/255 for c in shade) for shade in black_shades]
    red_shades = [tuple(c/255 for c in shade) for shade in red_shades]
    skyblue_shades = [tuple(c/255 for c in shade) for shade in skyblue_shades]

    xxx = np.arange(9) + 1
    if any(0 in row for row in isSelfTimedMode):
        xxx_labels = ['vis1 to Onset1', 'Onset1 to Target1', 'Onset1 to Peak1', 'Onset1 to LeverRetract1_End','LeverRetract1_End to Vis2', 'Vis2 to Onset2', 'Onset2 to Reward', 'Vis1 to Punish', 'Vis2 to Punish']
    else:
        xxx_labels = ['vis1 to Onset1', 'Onset1 to Target1', 'Onset1 to Peak1', 'Onset1 to LeverRetract1_End','LeverRetract1_End to Vis2', 'WaitforPress2 to Onset2', 'Onset2 to Reward', 'Vis1 to Punish', 'WaitforPress2 to Punish']

    # Initialize the loop for analysis #####################
    for i in range(0 , len(session_id)):
        TrialOutcomes = session_data['outcomes'][i]
        # We have Raw data and extract every thing from it (Times)
        raw_data = session_data['raw'][i]
        session_date = dates[i][2:]
        trial_types = raw_data['TrialTypes']
        opto = session_data['session_opto_tag'][i]
        
        # if len(TrialOutcomes) < 10:
        #     print('Not enough trial for analysis in session: ' + session_date)
        #     continue
        
        print('Time Interval Analysis:' + session_date)
        # SESSION VARIABLES ####################################################
        vis1_on1_chemo_s = []
        on1_targ1_chemo_s = []
        on1_amp1_chemo_s = []
        on1_levret1end_chemo_s = []
        levret1end_vis2_chemo_s = []
        vis2_on2_chemo_s = []
        on2_targ2_chemo_s = []
        
        vis1_on1_opto_s = []
        on1_targ1_opto_s = []
        on1_amp1_opto_s = []
        on1_levret1end_opto_s = []
        levret1end_vis2_opto_s = []
        vis2_on2_opto_s = []
        on2_targ2_opto_s = []
        
        vis1_on1_con_s = []
        on1_targ1_con_s = []
        on1_amp1_con_s = []
        on1_levret1end_con_s = []
        levret1end_vis2_con_s = []
        vis2_on2_con_s = []
        on2_targ2_con_s = []
        
        vis1_on1_chemo_l = []
        on1_targ1_chemo_l = []
        on1_amp1_chemo_l = []
        on1_levret1end_chemo_l = []
        levret1end_vis2_chemo_l = []
        vis2_on2_chemo_l = []
        on2_targ2_chemo_l = []
        
        vis1_on1_opto_l = []
        on1_targ1_opto_l = []
        on1_amp1_opto_l = []
        on1_levret1end_opto_l = []
        levret1end_vis2_opto_l = []
        vis2_on2_opto_l = []
        on2_targ2_opto_l = []
        
        vis1_on1_con_l = []
        on1_targ1_con_l = []
        on1_amp1_con_l = []
        on1_levret1end_con_l = []
        levret1end_vis2_con_l = []
        vis2_on2_con_l = []
        on2_targ2_con_l = []
        
        
        vis1_pun_chemo_s = []
        vis1_pun_opto_s = []
        vis1_pun_con_s = []
        
        vis1_pun_chemo_l = []
        vis1_pun_opto_l = []
        vis1_pun_con_l = []
        
        vis2_pun_chemo_s = []
        vis2_pun_opto_s = []
        vis2_pun_con_s = []
        
        vis2_pun_chemo_l = []
        vis2_pun_opto_l = []
        vis2_pun_con_l = []
        ############################################
        for trial in range(0 ,len(TrialOutcomes)):
                
            if np.isnan(isSelfTimedMode[i][trial]):
                continue
            
            # # Passing warmup trials 
            # if iswarmup == 1:
            #     continue
            
            if TrialOutcomes[trial] == 'Reward':
                # define the trajectory based on each trial
                encoder_data = raw_data['EncoderData'][trial]              
                times = encoder_data['Times']
                positions = encoder_data['Positions']
                encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                # NEED THIS VARIABLES
                trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
                
                VisDetect1 = int(trial_states['VisDetect1'][0]*1000)
                VisualStimulus1 = int(trial_states['VisualStimulus1'][0]*1000) 
                # End of lever retracts
                string = dates[i]
                string = string.split('(')[0]
                if int(string) <= 20240804 :
                    if np.isnan(trial_states['LeverRetract2'][1]*1000):
                        continue
                    LeverRetract2 = int(trial_states['LeverRetract2'][1]*1000) 
                elif 20240804 < int(string):
                    LeverRetract2 = int(trial_event['SoftCode1'][-1]*1000)
                
                if LeverRetract2 > 100000:
                    print('trial durations is larger than 100s: ', trial + 1)
                    continue
                LeverRetract1 = int(trial_states['LeverRetract1'][1]*1000) # End of lever retract1
                # First section of lever pushing for rewarded trials
                rotatory1 = int(trial_event['RotaryEncoder1_1'][0]*1000) 
                rotatory2 = int(trial_event['RotaryEncoder1_1'][1]*1000) 
                
                if LeverRetract2 < rotatory2:
                    print('LeverRetract2End smaller than target2',trial)
                    continue     
                    
                base_line = encoder_positions_aligned_vis1[:VisualStimulus1]
                base_line = base_line[~np.isnan(base_line)]
                base_line = np.mean(base_line)
                
                # Checking the baseline value
                if base_line >= 0.9:
                    print('trial starts with larger value:', trial + 1)
                    continue
                
                encoder_positions_aligned_vis1 = savgol_filter(encoder_positions_aligned_vis1, window_length=40, polyorder=3)
                velocity = np.gradient(encoder_positions_aligned_vis1, encoder_times_vis1)
                velocity = savgol_filter(velocity, window_length=40, polyorder=1)
                
                # Onset1
                on1_vel = velocity_onset(velocity,VisDetect1,rotatory1)
                # Vis1 to Onset1
                vis1_on1 = np.abs(on1_vel - VisualStimulus1)/1000  # type: ignore
                # Onset1 to target1 (rotatory1)
                on1_targ1 = np.abs(rotatory1 - on1_vel)/1000 # type: ignore
                # Amplitude Max1
                amp1 = np.max(encoder_positions_aligned_vis1[rotatory1:LeverRetract1])
                # Onset1 to Amp1 (Time)
                on1_amp1 = np.abs((np.argmax(encoder_positions_aligned_vis1[rotatory1:LeverRetract1]) + rotatory1) - on1_vel)/1000 # type: ignore
                # baseline to peak1
                base_amp1 = np.abs(amp1 - base_line)
                # Max velocity1
                if LeverRetract1 >= on1_vel: # type: ignore
                    max_vel1 = np.max(velocity[on1_vel:LeverRetract1])
                else:
                    max_vel1 = np.max(velocity[LeverRetract1:on1_vel])
                # Onset1 Velocity
                velaton1 = velocity[on1_vel]
                
                # variables we need fo visguided mice
                if any(0 in row for row in isSelfTimedMode):
                    VisDetect2 = int(trial_states['VisDetect2'][0]*1000)
                    VisualStimulus2 = int(trial_states['VisualStimulus2'][0]*1000)
                    # Onset2
                    on2_vel = velocity_onset(velocity,VisDetect2,rotatory2)
                    # Vis2 to Onset2
                    vis2_on2 = np.abs(on2_vel - VisualStimulus2)/1000 # type: ignore
                    
                    PreVis2Delay_1 = trial_states['PreVis2Delay'][0]
                    PreVis2Delay_2 = trial_states['PreVis2Delay'][1]
                    PreVis2Delay = PreVis2Delay_2 - PreVis2Delay_1
                # variables just neded for selftime mice
                else:
                    waitforpress2 = int(trial_states['WaitForPress2'][0]*1000)
                    # Onset2
                    on2_vel = velocity_onset(velocity,waitforpress2,rotatory2) # type: ignore
                    #NOTE: I make the waitpress2 equal to vis2 to make code smaller
                    VisualStimulus2 = waitforpress2
                    # waitpress2 to Onset2
                    vis2_on2 = np.abs(on2_vel - VisualStimulus2)/1000 # type: ignore
                    
                    PrePress2Delay_1 = trial_states['PrePress2Delay'][0]
                    PrePress2Delay_2 = trial_states['PrePress2Delay'][1]
                    PrePress2Delay = PrePress2Delay_2 - PrePress2Delay_1
                    PreVis2Delay = PrePress2Delay #NOTE: I make the PrePress2Delay equal to PreVis2Delay to make code smaller
                
                
                # Amplitude Max2
                amp2 = np.max(encoder_positions_aligned_vis1[rotatory2:LeverRetract2])
                # Onset2 to Amp2 (Time)
                on2_amp2 = np.abs((np.argmax(encoder_positions_aligned_vis1[rotatory2:LeverRetract2]) + rotatory2) - on2_vel)/1000 # type: ignore
                # Onset2 to target2 (Reward)
                on2_targ2 = np.abs(rotatory2 - on2_vel)/1000 # type: ignore
                # Onset Interval
                on_interval = np.abs(on2_vel - on1_vel)/1000 # type: ignore
                # Lever Retract1 End to Onset2
                levret1end_on2 = np.abs(on2_vel - LeverRetract1)/1000 # type: ignore
                on1_levret1end = np.abs(LeverRetract1 - on1_vel)/1000 # type: ignore
                levret1end_vis2 = np.abs(VisualStimulus2 - LeverRetract1)/1000
                # baseline to peak1
                base_amp2 = np.abs(amp2 - base_line)
                # Max velocity2
                if LeverRetract2 >= on2_vel:                 # type: ignore
                    max_vel2 = np.max(velocity[on2_vel:LeverRetract2])
                else:
                    max_vel2 = np.max(velocity[LeverRetract2:on2_vel])
                # Onset2 Velocity
                velaton2 = velocity[on2_vel]
                if trial_types[trial] == 1:
                    if chemo_labels[i] == 1:
                        vis1_on1_chemo_s.append(vis1_on1)
                        on1_targ1_chemo_s.append(on1_targ1)
                        on1_amp1_chemo_s.append(on1_amp1)
                        on1_levret1end_chemo_s.append(on1_levret1end)
                        levret1end_vis2_chemo_s.append(levret1end_vis2)
                        vis2_on2_chemo_s.append(vis2_on2)
                        on2_targ2_chemo_s.append(on2_targ2)
                    else:
                        if opto[trial] == 1:
                            vis1_on1_opto_s.append(vis1_on1)
                            on1_targ1_opto_s.append(on1_targ1)
                            on1_amp1_opto_s.append(on1_amp1)
                            on1_levret1end_opto_s.append(on1_levret1end)
                            levret1end_vis2_opto_s.append(levret1end_vis2)
                            vis2_on2_opto_s.append(vis2_on2)
                            on2_targ2_opto_s.append(on2_targ2)
                        else:
                            vis1_on1_con_s.append(vis1_on1)
                            on1_targ1_con_s.append(on1_targ1)
                            on1_amp1_con_s.append(on1_amp1)
                            on1_levret1end_con_s.append(on1_levret1end)
                            levret1end_vis2_con_s.append(levret1end_vis2)
                            vis2_on2_con_s.append(vis2_on2)
                            on2_targ2_con_s.append(on2_targ2)
                else:
                    if chemo_labels[i] == 1:
                        vis1_on1_chemo_l.append(vis1_on1)
                        on1_targ1_chemo_l.append(on1_targ1)
                        on1_amp1_chemo_l.append(on1_amp1)
                        on1_levret1end_chemo_l.append(on1_levret1end)
                        levret1end_vis2_chemo_l.append(levret1end_vis2)
                        vis2_on2_chemo_l.append(vis2_on2)
                        on2_targ2_chemo_l.append(on2_targ2)
                    else:
                        if opto[trial] == 1:
                            vis1_on1_opto_l.append(vis1_on1)
                            on1_targ1_opto_l.append(on1_targ1)
                            on1_amp1_opto_l.append(on1_amp1)
                            on1_levret1end_opto_l.append(on1_levret1end)
                            levret1end_vis2_opto_l.append(levret1end_vis2)
                            vis2_on2_opto_l.append(vis2_on2)
                            on2_targ2_opto_l.append(on2_targ2)
                        else:
                            vis1_on1_con_l.append(vis1_on1)
                            on1_targ1_con_l.append(on1_targ1)
                            on1_amp1_con_l.append(on1_amp1)
                            on1_levret1end_con_l.append(on1_levret1end)
                            levret1end_vis2_con_l.append(levret1end_vis2)
                            vis2_on2_con_l.append(vis2_on2)
                            on2_targ2_con_l.append(on2_targ2)   

                
            elif TrialOutcomes[trial] == 'DidNotPress1':
                # NEED THIS VARIABLES
                trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
                
                VisualStimulus1 = int(trial_states['VisualStimulus1'][0]*1000) 
                punish = int(trial_states['Punish'][0]*1000)
                vis1_pun = np.abs(punish - VisualStimulus1)/1000
                
                if trial_types[trial] == 1:
                    if chemo_labels[i] == 1:
                        vis1_pun_chemo_s.append(vis1_pun)
                    else:
                        if opto[trial] == 1:
                            vis1_pun_opto_s.append(vis1_pun)
                        else:
                            vis1_pun_con_s.append(vis1_pun)
                else:
                    if chemo_labels[i] == 1:
                        vis1_pun_chemo_l.append(vis1_pun)
                    else:
                        if opto[trial] == 1:
                            vis1_pun_opto_l.append(vis1_pun)
                        else:
                            vis1_pun_con_l.append(vis1_pun)
                
            elif TrialOutcomes[trial] == 'DidNotPress2':
                # NEED THIS VARIABLES
                trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                trial_event = raw_data['RawEvents']['Trial'][trial]['Events']
                punish = int(trial_states['Punish'][0]*1000)
                
                if any(0 in row for row in isSelfTimedMode):
                    VisualStimulus2 = int(trial_states['VisualStimulus2'][0]*1000)
                else:
                    waitforpress2 = int(trial_states['WaitForPress2'][0]*1000)
                    VisualStimulus2 = waitforpress2
                    
                vis2_pun = np.abs(punish - VisualStimulus2)/1000
                
                if trial_types[trial] == 1:
                    if chemo_labels[i] == 1:
                        vis2_pun_chemo_s.append(vis2_pun)
                    else:
                        if opto[trial] == 1:
                            vis2_pun_opto_s.append(vis2_pun)
                        else:
                            vis2_pun_con_s.append(vis2_pun)
                else:
                    if chemo_labels[i] == 1:
                        vis2_pun_chemo_l.append(vis2_pun)
                    else:
                        if opto[trial] == 1:
                            vis2_pun_opto_l.append(vis2_pun)
                        else:
                            vis2_pun_con_l.append(vis2_pun)
        # PLOTTING #################################################
        offset = 2
        fig1, axs1 = plt.subplots(nrows=9, ncols= 2, figsize=(20, 70))
        fig1.subplots_adjust(hspace=0.7)
        fig1.suptitle( 1* '\n'+subject+ '\n Even Interval\n' + dates[i] + 5*'\n')
        
        for k in range(0,9):
            axs1[k,0].set_title('Short Trials. \n')
            axs1[k,0].spines['right'].set_visible(False)
            axs1[k,0].spines['top'].set_visible(False)
            axs1[k,0].set_xlabel('Event')
            axs1[k,0].set_ylabel('Time Interval(s) Mean +/- SEM')
            
            axs1[k,1].set_title('Long Trials. \n')
            axs1[k,1].spines['right'].set_visible(False)
            axs1[k,1].spines['top'].set_visible(False)
            axs1[k,1].set_xlabel('Event')
            axs1[k,1].set_ylabel('Time Interval(s) Mean +/- SEM')

    
        
        
        if len(vis1_on1_con_s) > 0:
            axs1[0,0].boxplot(vis1_on1_con_s, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[0]),labels=[xxx_labels[0]])
        if len(vis1_on1_chemo_s) > 0:
            axs1[0,0].boxplot(vis1_on1_chemo_s, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[0]))
        if len(vis1_on1_opto_s) > 0:
            axs1[0,0].boxplot(vis1_on1_opto_s, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[0]))
            
        if len(on1_targ1_con_s) > 0:
            axs1[1,0].boxplot(on1_targ1_con_s, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[0]), labels=[xxx_labels[1]])
        if len(on1_targ1_chemo_s) > 0:
            axs1[1,0].boxplot(on1_targ1_chemo_s, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[0]))
        if len(on1_targ1_opto_s) > 0:
            axs1[1,0].boxplot(on1_targ1_opto_s, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[0]), )
            
        if len(on1_amp1_con_s) > 0:
            axs1[2,0].boxplot(on1_amp1_con_s, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[0]), labels=[xxx_labels[2]])
        if len(on1_amp1_chemo_s) > 0:
            axs1[2,0].boxplot(on1_amp1_chemo_s, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[0]))
        if len(on1_amp1_opto_s) > 0:
            axs1[2,0].boxplot(on1_amp1_opto_s, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[0]))
            
        if len(on1_levret1end_con_s) > 0:
            axs1[3,0].boxplot(on1_levret1end_con_s, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[0]), labels=[xxx_labels[3]])
        if len(on1_levret1end_chemo_s) > 0:
            axs1[3,0].boxplot(on1_levret1end_chemo_s, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[0]))
        if len(on1_levret1end_opto_s) > 0:
            axs1[3,0].boxplot(on1_levret1end_opto_s, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[0]))
            
        if len(levret1end_vis2_con_s) > 0:
            axs1[4,0].boxplot(levret1end_vis2_con_s, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[0]), labels=[xxx_labels[4]])
        if len(levret1end_vis2_chemo_s) > 0:
            axs1[4,0].boxplot(levret1end_vis2_chemo_s, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[0]))
        if len(levret1end_vis2_opto_s) > 0:
            axs1[4,0].boxplot(levret1end_vis2_opto_s, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[0]))
            
        if len(vis2_on2_con_s) > 0:
            axs1[5,0].boxplot(vis2_on2_con_s, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[0]), labels=[xxx_labels[5]])
        if len(vis2_on2_chemo_s) > 0:
            axs1[5,0].boxplot(vis2_on2_chemo_s, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[0]))
        if len(vis2_on2_opto_s) > 0:
            axs1[5,0].boxplot(vis2_on2_opto_s, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[0]))
            
        if len(on2_targ2_con_s) > 0:
            axs1[6,0].boxplot(on2_targ2_con_s, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[0]), labels=[xxx_labels[6]])
        if len(on2_targ2_chemo_s) > 0:
            axs1[6,0].boxplot(on2_targ2_chemo_s, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[0]))
        if len(on2_targ2_opto_s) > 0:
            axs1[6,0].boxplot(on2_targ2_opto_s, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[0]))
            
        if len(vis1_pun_con_s) > 0:
            axs1[7,0].boxplot(vis1_pun_con_s, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[0]), labels=[xxx_labels[7]])
        if len(vis1_pun_chemo_s) > 0:
            axs1[7,0].boxplot(vis1_pun_chemo_s, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[0]))
        if len(vis1_pun_opto_s) > 0:
            axs1[7,0].boxplot(vis1_pun_opto_s, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[0]))
            
        if len(vis2_pun_con_s) > 0:
            axs1[8,0].boxplot(vis2_pun_con_s, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[0]), labels=[xxx_labels[8]])
        if len(vis2_pun_chemo_s) > 0:
            axs1[8,0].boxplot(vis2_pun_chemo_s, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[0]))
        if len(vis2_pun_opto_s) > 0:
            axs1[8,0].boxplot(vis2_pun_opto_s, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[0]))         

        if len(vis1_on1_con_l) > 0:
            axs1[0,1].boxplot(vis1_on1_con_l, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[1]), labels=[xxx_labels[0]])
        if len(vis1_on1_chemo_l) > 0:
            axs1[0,1].boxplot(vis1_on1_chemo_l, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[1]))
        if len(vis1_on1_opto_l) > 0:
            axs1[0,1].boxplot(vis1_on1_opto_l, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[1]))
            
        if len(on1_targ1_con_l) > 0:
            axs1[1,1].boxplot(on1_targ1_con_l, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[1]), labels=[xxx_labels[1]])
        if len(on1_targ1_chemo_l) > 0:
            axs1[1,1].boxplot(on1_targ1_chemo_l, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[1]))
        if len(on1_targ1_opto_l) > 0:
            axs1[1,1].boxplot(on1_targ1_opto_l, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[1]))
            
        if len(on1_amp1_con_l) > 0:
            axs1[2,1].boxplot(on1_amp1_con_l, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[1]), labels=[xxx_labels[2]])
        if len(on1_amp1_chemo_l) > 0:
            axs1[2,1].boxplot(on1_amp1_chemo_l, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[1]))
        if len(on1_amp1_opto_l) > 0:
            axs1[2,1].boxplot(on1_amp1_opto_l, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[1]))
            
        if len(on1_levret1end_con_l) > 0:
            axs1[3,1].boxplot(on1_levret1end_con_l, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[1]), labels=[xxx_labels[3]])
        if len(on1_levret1end_chemo_l) > 0:
            axs1[3,1].boxplot(on1_levret1end_chemo_l, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[1]))
        if len(on1_levret1end_opto_l) > 0:
            axs1[3,1].boxplot(on1_levret1end_opto_l, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[1]))
            
        if len(levret1end_vis2_con_l) > 0:
            axs1[4,1].boxplot(levret1end_vis2_con_l, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[1]), labels=[xxx_labels[4]])
        if len(levret1end_vis2_chemo_l) > 0:
            axs1[4,1].boxplot(levret1end_vis2_chemo_l, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[1]))
        if len(levret1end_vis2_opto_l) > 0:
            axs1[4,1].boxplot(levret1end_vis2_opto_l, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[1]))
            
        if len(vis2_on2_con_l) > 0:
            axs1[5,1].boxplot(vis2_on2_con_l, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[1]), labels=[xxx_labels[5]])
        if len(vis2_on2_chemo_l) > 0:
            axs1[5,1].boxplot(vis2_on2_chemo_l, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[1]))
        if len(vis2_on2_opto_l) > 0:
            axs1[5,1].boxplot(vis2_on2_opto_l, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[1]))
            
        if len(on2_targ2_con_l) > 0:
            axs1[6,1].boxplot(on2_targ2_con_l, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[1]), labels=[xxx_labels[6]])
        if len(on2_targ2_chemo_l) > 0:
            axs1[6,1].boxplot(on2_targ2_chemo_l, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[1]))
        if len(on2_targ2_opto_l) > 0:
            axs1[6,1].boxplot(on2_targ2_opto_l, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[1]))
            
        if len(vis1_pun_con_l) > 0:
            axs1[7,1].boxplot(vis1_pun_con_l, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[1]), labels=[xxx_labels[7]])
        if len(vis1_pun_chemo_l) > 0:
            axs1[7,1].boxplot(vis1_pun_chemo_l, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[1]))
        if len(vis1_pun_opto_l) > 0:
            axs1[7,1].boxplot(vis1_pun_opto_l, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[1]))
            
        if len(vis2_pun_con_l) > 0:
            axs1[8,1].boxplot(vis2_pun_con_l, positions=[xxx[0] - 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=black_shades[1]), labels=[xxx_labels[8]])
        if len(vis2_pun_chemo_l) > 0:
            axs1[8,1].boxplot(vis2_pun_chemo_l, positions=[xxx[0]], widths=0.1, patch_artist=True, boxprops=dict(facecolor=red_shades[1]))
        if len(vis2_pun_opto_l) > 0:
            axs1[8,1].boxplot(vis2_pun_opto_l, positions=[xxx[0] + 0.1 * offset], widths=0.1, patch_artist=True, boxprops=dict(facecolor=skyblue_shades[1]))
        
        xxx = np.arange(1) + 1
        axs1[0,0].set_xticks(xxx)
        axs1[0,0].set_xticklabels([xxx_labels[0]], rotation=90, ha = 'center')
        
        axs1[1,0].set_xticks(xxx)
        axs1[1,0].set_xticklabels([xxx_labels[1]], rotation=90, ha = 'center')
        
        axs1[2,0].set_xticks(xxx)
        axs1[2,0].set_xticklabels([xxx_labels[2]], rotation=90, ha = 'center')
        
        axs1[3,0].set_xticks(xxx)
        axs1[3,0].set_xticklabels([xxx_labels[3]], rotation=90, ha = 'center')
        
        axs1[4,0].set_xticks(xxx)
        axs1[4,0].set_xticklabels([xxx_labels[4]], rotation=90, ha = 'center')
        
        axs1[5,0].set_xticks(xxx)
        axs1[5,0].set_xticklabels([xxx_labels[5]], rotation=90, ha = 'center')
        
        axs1[6,0].set_xticks(xxx)
        axs1[6,0].set_xticklabels([xxx_labels[6]], rotation=90, ha = 'center')
        
        axs1[7,0].set_xticks(xxx)
        axs1[7,0].set_xticklabels([xxx_labels[7]], rotation=90, ha = 'center')
        
        axs1[8,0].set_xticks(xxx)
        axs1[8,0].set_xticklabels([xxx_labels[8]], rotation=90, ha = 'center')
        
        axs1[0,1].set_xticks(xxx)
        axs1[0,1].set_xticklabels([xxx_labels[0]], rotation=90, ha = 'center')
        
        axs1[1,1].set_xticks(xxx)
        axs1[1,1].set_xticklabels([xxx_labels[1]], rotation=90, ha = 'center')
        
        axs1[2,1].set_xticks(xxx)
        axs1[2,1].set_xticklabels([xxx_labels[2]], rotation=90, ha = 'center')
        
        axs1[3,1].set_xticks(xxx)
        axs1[3,1].set_xticklabels([xxx_labels[3]], rotation=90, ha = 'center')
        
        axs1[4,1].set_xticks(xxx)
        axs1[4,1].set_xticklabels([xxx_labels[4]], rotation=90, ha = 'center')
        
        axs1[5,1].set_xticks(xxx)
        axs1[5,1].set_xticklabels([xxx_labels[5]], rotation=90, ha = 'center')
        
        axs1[6,1].set_xticks(xxx)
        axs1[6,1].set_xticklabels([xxx_labels[6]], rotation=90, ha = 'center')
        
        axs1[7,1].set_xticks(xxx)
        axs1[7,1].set_xticklabels([xxx_labels[7]], rotation=90, ha = 'center')
        
        axs1[8,1].set_xticks(xxx)
        axs1[8,1].set_xticklabels([xxx_labels[8]], rotation=90, ha = 'center')
        
        # Create custom legend with markers
        legend_elements = [
        Line2D([0], [0], color=black_shades[0], marker='o', linestyle='None', markersize=10, label='Control'),
        Line2D([0], [0], color=red_shades[0], marker='o', linestyle='None', markersize=10, label='Chemo'),
        Line2D([0], [0], color=skyblue_shades[0], marker='o', linestyle='None', markersize=10, label='Opto')]

        fig1.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5,1))
        
        fig1.tight_layout()

        output_figs_dir = output_dir_onedrive + subject + '/'  
        pdf_path = os.path.join(output_figs_dir, subject + '_'+ dates[i] +'_Event_interval_boxplot.pdf')
        plt.rcParams['pdf.fonttype'] = 42  # Ensure text is kept as text (not outlines)
        plt.rcParams['ps.fonttype'] = 42   # For compatibility with EPS as well, if needed

        # Save both plots into a single PDF file with each on a separate page
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig1)

        plt.close(fig1)