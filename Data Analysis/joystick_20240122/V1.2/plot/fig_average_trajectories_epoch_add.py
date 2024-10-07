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

def epoching_session(nTrials , Block_size):
    epoch1_block_size = int(np.ceil(Block_size/5))
    epoch2_block_size = int(np.ceil((Block_size - epoch1_block_size)/2))
    epoch3_block_size = Block_size - (epoch1_block_size + epoch2_block_size)

    # define the count of block in a session
    nBlock = int(np.ceil(nTrials/Block_size))
    epoch1 = []
    epoch2 = []
    epoch3 = []
    for i in range(0,nBlock):
        epoch1 = np.concatenate((epoch1, np.arange(i * Block_size, ((i * Block_size) + epoch1_block_size))))
        epoch2 = np.concatenate((epoch2, np.arange((i * Block_size) + epoch1_block_size , (i * Block_size) + epoch1_block_size + epoch2_block_size)))
        epoch3 = np.concatenate((epoch3, np.arange((i * Block_size) + epoch1_block_size + epoch2_block_size , (i * Block_size) + epoch1_block_size + epoch2_block_size + epoch3_block_size)))
        
    # Now we have arrays for epochs, just to drop the trials larger than nTrials

    epoch1 = epoch1[epoch1 <= nTrials]
    epoch2 = epoch2[epoch2 <= nTrials]
    epoch3 = epoch3[epoch3 <= nTrials]
    return epoch1, epoch2, epoch3

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

def generate_grayscale_shades(n):
    shades = []
    for i in range(1, n + 1):  # Start from 1 instead of 0
        intensity = 255 * (1 - i / n)
        shade = (intensity, intensity, intensity)  # Grayscale has equal R, G, B values
        shades.append(shade)
    shades = [(x/255, y/255, z/255) for x, y, z in shades]
    return shades

# Function to generate redscale shades starting from shade 1
def generate_redscale_shades(n):
    shades = []
    for i in range(0, n):  # Start from 1 instead of 0
        intensity = 255 * (1 - i / n)
        shade = (intensity, 0, 0)  # Redscale has varying R value with G, B fixed at 0
        shades.append(shade)
    shades = [(x/255, y/255, z/255) for x, y, z in shades]
    return shades

def average_superimposed_epoch_add(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):
    ###################################################### POOLED VARIABLES ##########################################
    # EPOCH1 #####
    # all
    p_epoch1_vis1_con_all = []
    p_epoch1_vis2_con_all = []
    p_epoch1_rew_con_all = []
    p_epoch1_on1_con_all = []
    p_epoch1_on2_con_all = []

    p_epoch1_vis1_chemo_all = []
    p_epoch1_vis2_chemo_all = []
    p_epoch1_rew_chemo_all = []
    p_epoch1_on1_chemo_all = []
    p_epoch1_on2_chemo_all = []

    p_epoch1_vis1_opto_all = []
    p_epoch1_vis2_opto_all = []
    p_epoch1_rew_opto_all = []
    p_epoch1_on1_opto_all = []
    p_epoch1_on2_opto_all = []
    # short
    p_epoch1_vis1_con_s = []
    p_epoch1_vis2_con_s = []
    p_epoch1_rew_con_s = []
    p_epoch1_on1_con_s = []
    p_epoch1_on2_con_s = []

    p_epoch1_vis1_chemo_s = []
    p_epoch1_vis2_chemo_s = []
    p_epoch1_rew_chemo_s = []
    p_epoch1_on1_chemo_s = []
    p_epoch1_on2_chemo_s = []

    p_epoch1_vis1_opto_s = []
    p_epoch1_vis2_opto_s = []
    p_epoch1_rew_opto_s = []
    p_epoch1_on1_opto_s = []
    p_epoch1_on2_opto_s = []
    # long
    p_epoch1_vis1_con_l = []
    p_epoch1_vis2_con_l = []
    p_epoch1_rew_con_l = []
    p_epoch1_on1_con_l = []
    p_epoch1_on2_con_l = []

    p_epoch1_vis1_chemo_l = []
    p_epoch1_vis2_chemo_l = []
    p_epoch1_rew_chemo_l = []
    p_epoch1_on1_chemo_l = []
    p_epoch1_on2_chemo_l = []

    p_epoch1_vis1_opto_l = []
    p_epoch1_vis2_opto_l = []
    p_epoch1_rew_opto_l = []
    p_epoch1_on1_opto_l = []
    p_epoch1_on2_opto_l = []

    # EPOCH2 #####
    # all
    p_epoch2_vis1_con_all = []
    p_epoch2_vis2_con_all = []
    p_epoch2_rew_con_all = []
    p_epoch2_on1_con_all = []
    p_epoch2_on2_con_all = []

    p_epoch2_vis1_chemo_all = []
    p_epoch2_vis2_chemo_all = []
    p_epoch2_rew_chemo_all = []
    p_epoch2_on1_chemo_all = []
    p_epoch2_on2_chemo_all = []

    p_epoch2_vis1_opto_all = []
    p_epoch2_vis2_opto_all = []
    p_epoch2_rew_opto_all = []
    p_epoch2_on1_opto_all = []
    p_epoch2_on2_opto_all = []
    # short
    p_epoch2_vis1_con_s = []
    p_epoch2_vis2_con_s = []
    p_epoch2_rew_con_s = []
    p_epoch2_on1_con_s = []
    p_epoch2_on2_con_s = []

    p_epoch2_vis1_chemo_s = []
    p_epoch2_vis2_chemo_s = []
    p_epoch2_rew_chemo_s = []
    p_epoch2_on1_chemo_s = []
    p_epoch2_on2_chemo_s = []

    p_epoch2_vis1_opto_s = []
    p_epoch2_vis2_opto_s = []
    p_epoch2_rew_opto_s = []
    p_epoch2_on1_opto_s = []
    p_epoch2_on2_opto_s = []
    # long
    p_epoch2_vis1_con_l = []
    p_epoch2_vis2_con_l = []
    p_epoch2_rew_con_l = []
    p_epoch2_on1_con_l = []
    p_epoch2_on2_con_l = []

    p_epoch2_vis1_chemo_l = []
    p_epoch2_vis2_chemo_l = []
    p_epoch2_rew_chemo_l = []
    p_epoch2_on1_chemo_l = []
    p_epoch2_on2_chemo_l = []

    p_epoch2_vis1_opto_l = []
    p_epoch2_vis2_opto_l = []
    p_epoch2_rew_opto_l = []
    p_epoch2_on1_opto_l = []
    p_epoch2_on2_opto_l = []

    # EPOCH3 #####
    # all
    p_epoch3_vis1_con_all = []
    p_epoch3_vis2_con_all = []
    p_epoch3_rew_con_all = []
    p_epoch3_on1_con_all = []
    p_epoch3_on2_con_all = []

    p_epoch3_vis1_chemo_all = []
    p_epoch3_vis2_chemo_all = []
    p_epoch3_rew_chemo_all = []
    p_epoch3_on1_chemo_all = []
    p_epoch3_on2_chemo_all = []

    p_epoch3_vis1_opto_all = []
    p_epoch3_vis2_opto_all = []
    p_epoch3_rew_opto_all = []
    p_epoch3_on1_opto_all = []
    p_epoch3_on2_opto_all = []
    # short
    p_epoch3_vis1_con_s = []
    p_epoch3_vis2_con_s = []
    p_epoch3_rew_con_s = []
    p_epoch3_on1_con_s = []
    p_epoch3_on2_con_s = []

    p_epoch3_vis1_chemo_s = []
    p_epoch3_vis2_chemo_s = []
    p_epoch3_rew_chemo_s = []
    p_epoch3_on1_chemo_s = []
    p_epoch3_on2_chemo_s = []

    p_epoch3_vis1_opto_s = []
    p_epoch3_vis2_opto_s = []
    p_epoch3_rew_opto_s = []
    p_epoch3_on1_opto_s = []
    p_epoch3_on2_opto_s = []
    # long
    p_epoch3_vis1_con_l = []
    p_epoch3_vis2_con_l = []
    p_epoch3_rew_con_l = []
    p_epoch3_on1_con_l = []
    p_epoch3_on2_con_l = []

    p_epoch3_vis1_chemo_l = []
    p_epoch3_vis2_chemo_l = []
    p_epoch3_rew_chemo_l = []
    p_epoch3_on1_chemo_l = []
    p_epoch3_on2_chemo_l = []

    p_epoch3_vis1_opto_l = []
    p_epoch3_vis2_opto_l = []
    p_epoch3_rew_opto_l = []
    p_epoch3_on1_opto_l = []
    p_epoch3_on2_opto_l = []
    ##################################################################################################################
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_id = np.arange(len(outcomes)) + 1
    sess_num = np.zeros((3 , len(session_id)))
    encoder_time_max = 100
    ms_per_s = 1000
    chemo_labels = session_data['chemo']
    chemo_conter = 0
    for i in range(0 , len(chemo_labels)):
            if chemo_labels[i] == 1:
                dates[i] = dates[i] + '(chemo)'
                chemo_conter = chemo_conter + 1
                
    dates = deduplicate_chemo(dates)            
    isSelfTimedMode  = process_matrix (session_data['isSelfTimedMode'])


    encoder_times = np.linspace(0, 3, num= 3000)

    if any(0 in row for row in isSelfTimedMode):
        print('Visually Guided')
    else:
        print('Selftime')
        
    stop_idx = session_data['total_sessions'] - 1
    target_thresh = session_data['session_target_thresh']
    t = []
    for i in range(0 , stop_idx+1):
        t.append(np.mean(target_thresh[i]))
    target_thresh = np.mean(t)

    # Plotting:
    red = generate_redscale_shades(chemo_conter)
    black = generate_grayscale_shades(len(session_id) - chemo_conter)
    r = -1
    k = -1

    fig, axs = plt.subplots(nrows=9, ncols=5, figsize=(27, 36))
    fig.subplots_adjust(hspace=0.7)
    fig.suptitle(subject + '\n Average Trajectories superimposed for Rewarded, EarlyPress2 and DidnotPress2 Trials\n')
    # ALL TRIALS
    axs[0,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[0,0].set_title('VisStim1 Aligned Epoch1.\n') 
    axs[0,0].axvline(x = 0, color = 'r', linestyle='--')
    axs[0,0].set_xlim((-1, 2))
    axs[0,0].spines['right'].set_visible(False)
    axs[0,0].spines['top'].set_visible(False)
    axs[0,0].set_xlabel('Time from VisStim1 (s)')
    axs[0,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[0,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[0,1].axvline(x = 0, color = 'r', linestyle='--')
    axs[0,1].set_xlim((-1, 2))
    axs[0,1].spines['right'].set_visible(False)
    axs[0,1].spines['top'].set_visible(False)
    axs[0,1].set_title('VisStim2 Aligned Epoch1.\n') 
    axs[0,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs[0,1].set_title('waitforpress2 Aligned Epoch1.\n') 
        axs[0,1].set_xlabel('Time from waitforpress2 (s)')
    axs[0,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[0,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[0,2].set_title('\n ALL TRIALS \n Reward Aligned Epoch1.\n') 
    axs[0,2].axvline(x = 0, color = 'r', linestyle='--')
    axs[0,2].set_xlim((-1, 2))
    axs[0,2].spines['right'].set_visible(False)
    axs[0,2].spines['top'].set_visible(False)
    axs[0,2].set_xlabel('Time from Reward (s)')
    axs[0,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[0,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[0,3].set_title('Onset1 Aligned Epoch1.\n') 
    axs[0,3].axvline(x = 0, color = 'r', linestyle='--')
    axs[0,3].set_xlim((-1, 2))
    axs[0,3].spines['right'].set_visible(False)
    axs[0,3].spines['top'].set_visible(False)
    axs[0,3].set_xlabel('Time from Onset1 (s)')
    axs[0,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[0,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[0,4].set_title('Onset2 Aligned Epoch1.\n') 
    axs[0,4].axvline(x = 0, color = 'r', linestyle='--')
    axs[0,4].set_xlim((-1, 2))
    axs[0,4].spines['right'].set_visible(False)
    axs[0,4].spines['top'].set_visible(False)
    axs[0,4].set_xlabel('Time from Onset2 (s)')
    axs[0,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[1,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[1,0].set_title('VisStim1 Aligned Epoch2.\n') 
    axs[1,0].axvline(x = 0, color = 'r', linestyle='--')
    axs[1,0].set_xlim((-1, 2))
    axs[1,0].spines['right'].set_visible(False)
    axs[1,0].spines['top'].set_visible(False)
    axs[1,0].set_xlabel('Time from VisStim1 (s)')
    axs[1,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[1,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[1,1].axvline(x = 0, color = 'r', linestyle='--')
    axs[1,1].set_xlim((-1, 2))
    axs[1,1].spines['right'].set_visible(False)
    axs[1,1].spines['top'].set_visible(False)
    axs[1,1].set_title('VisStim2 Aligned Epoch2.\n') 
    axs[1,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs[1,1].set_title('waitforpress2 Aligned Epoch2.\n') 
        axs[1,1].set_xlabel('Time from waitforpress2 (s)')
    axs[1,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[1,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[1,2].set_title('\n Reward Aligned Epoch2.\n') 
    axs[1,2].axvline(x = 0, color = 'r', linestyle='--')
    axs[1,2].set_xlim((-1, 2))
    axs[1,2].spines['right'].set_visible(False)
    axs[1,2].spines['top'].set_visible(False)
    axs[1,2].set_xlabel('Time from Reward (s)')
    axs[1,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[1,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[1,3].set_title('Onset1 Aligned Epoch2.\n') 
    axs[1,3].axvline(x = 0, color = 'r', linestyle='--')
    axs[1,3].set_xlim((-1, 2))
    axs[1,3].spines['right'].set_visible(False)
    axs[1,3].spines['top'].set_visible(False)
    axs[1,3].set_xlabel('Time from Onset1 (s)')
    axs[1,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[1,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[1,4].set_title('Onset2 Aligned Epoch2.\n') 
    axs[1,4].axvline(x = 0, color = 'r', linestyle='--')
    axs[1,4].set_xlim((-1, 2))
    axs[1,4].spines['right'].set_visible(False)
    axs[1,4].spines['top'].set_visible(False)
    axs[1,4].set_xlabel('Time from Onset2 (s)')
    axs[1,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[2,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[2,0].set_title('VisStim1 Aligned Epoch3.\n') 
    axs[2,0].axvline(x = 0, color = 'r', linestyle='--')
    axs[2,0].set_xlim((-1, 2))
    axs[2,0].spines['right'].set_visible(False)
    axs[2,0].spines['top'].set_visible(False)
    axs[2,0].set_xlabel('Time from VisStim1 (s)')
    axs[2,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[2,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[2,1].axvline(x = 0, color = 'r', linestyle='--')
    axs[2,1].set_xlim((-1, 2))
    axs[2,1].spines['right'].set_visible(False)
    axs[2,1].spines['top'].set_visible(False)
    axs[2,1].set_title('VisStim2 Aligned Epoch3.\n') 
    axs[2,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs[2,1].set_title('waitforpress2 Aligned Epoch3.\n') 
        axs[2,1].set_xlabel('Time from waitforpress2 (s)')
    axs[2,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[2,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[2,2].set_title('\n Reward Aligned Epoch3.\n') 
    axs[2,2].axvline(x = 0, color = 'r', linestyle='--')
    axs[2,2].set_xlim((-1, 2))
    axs[2,2].spines['right'].set_visible(False)
    axs[2,2].spines['top'].set_visible(False)
    axs[2,2].set_xlabel('Time from Reward (s)')
    axs[2,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[2,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[2,3].set_title('Onset1 Aligned Epoch3.\n') 
    axs[2,3].axvline(x = 0, color = 'r', linestyle='--')
    axs[2,3].set_xlim((-1, 2))
    axs[2,3].spines['right'].set_visible(False)
    axs[2,3].spines['top'].set_visible(False)
    axs[2,3].set_xlabel('Time from Onset1 (s)')
    axs[2,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[2,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[2,4].set_title('Onset2 Aligned Epoch3.\n') 
    axs[2,4].axvline(x = 0, color = 'r', linestyle='--')
    axs[2,4].set_xlim((-1, 2))
    axs[2,4].spines['right'].set_visible(False)
    axs[2,4].spines['top'].set_visible(False)
    axs[2,4].set_xlabel('Time from Onset2 (s)')
    axs[2,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    # SHORT TRIALS
    axs[3,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[3,0].set_title('VisStim1 Aligned Epoch1.\n') 
    axs[3,0].axvline(x = 0, color = 'r', linestyle='--')
    axs[3,0].set_xlim((-1, 2))
    axs[3,0].spines['right'].set_visible(False)
    axs[3,0].spines['top'].set_visible(False)
    axs[3,0].set_xlabel('Time from VisStim1 (s)')
    axs[3,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[3,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[3,1].axvline(x = 0, color = 'r', linestyle='--')
    axs[3,1].set_xlim((-1, 2))
    axs[3,1].spines['right'].set_visible(False)
    axs[3,1].spines['top'].set_visible(False)
    axs[3,1].set_title('VisStim2 Aligned Epoch1.\n') 
    axs[3,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs[3,1].set_title('waitforpress2 Aligned Epoch1.\n') 
        axs[3,1].set_xlabel('Time from waitforpress2 (s)')
    axs[3,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[3,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[3,2].set_title('\n SHORT TRIALS \n Reward Aligned Epoch1.\n') 
    axs[3,2].axvline(x = 0, color = 'r', linestyle='--')
    axs[3,2].set_xlim((-1, 2))
    axs[3,2].spines['right'].set_visible(False)
    axs[3,2].spines['top'].set_visible(False)
    axs[3,2].set_xlabel('Time from Reward (s)')
    axs[3,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[3,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[3,3].set_title('Onset1 Aligned Epoch1.\n') 
    axs[3,3].axvline(x = 0, color = 'r', linestyle='--')
    axs[3,3].set_xlim((-1, 2))
    axs[3,3].spines['right'].set_visible(False)
    axs[3,3].spines['top'].set_visible(False)
    axs[3,3].set_xlabel('Time from Onset1 (s)')
    axs[3,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[3,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[3,4].set_title('Onset2 Aligned Epoch1.\n') 
    axs[3,4].axvline(x = 0, color = 'r', linestyle='--')
    axs[3,4].set_xlim((-1, 2))
    axs[3,4].spines['right'].set_visible(False)
    axs[3,4].spines['top'].set_visible(False)
    axs[3,4].set_xlabel('Time from Onset2 (s)')
    axs[3,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[4,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[4,0].set_title('VisStim1 Aligned Epoch2.\n') 
    axs[4,0].axvline(x = 0, color = 'r', linestyle='--')
    axs[4,0].set_xlim((-1, 2))
    axs[4,0].spines['right'].set_visible(False)
    axs[4,0].spines['top'].set_visible(False)
    axs[4,0].set_xlabel('Time from VisStim1 (s)')
    axs[4,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[4,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[4,1].axvline(x = 0, color = 'r', linestyle='--')
    axs[4,1].set_xlim((-1, 2))
    axs[4,1].spines['right'].set_visible(False)
    axs[4,1].spines['top'].set_visible(False)
    axs[4,1].set_title('VisStim2 Aligned Epoch2.\n') 
    axs[4,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs[4,1].set_title('waitforpress2 Aligned Epoch2.\n') 
        axs[4,1].set_xlabel('Time from waitforpress2 (s)')
    axs[4,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[4,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[4,2].set_title('\n Reward Aligned Epoch2.\n') 
    axs[4,2].axvline(x = 0, color = 'r', linestyle='--')
    axs[4,2].set_xlim((-1, 2))
    axs[4,2].spines['right'].set_visible(False)
    axs[4,2].spines['top'].set_visible(False)
    axs[4,2].set_xlabel('Time from Reward (s)')
    axs[4,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[4,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[4,3].set_title('Onset1 Aligned Epoch2.\n') 
    axs[4,3].axvline(x = 0, color = 'r', linestyle='--')
    axs[4,3].set_xlim((-1, 2))
    axs[4,3].spines['right'].set_visible(False)
    axs[4,3].spines['top'].set_visible(False)
    axs[4,3].set_xlabel('Time from Onset1 (s)')
    axs[4,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[4,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[4,4].set_title('Onset2 Aligned Epoch2.\n') 
    axs[4,4].axvline(x = 0, color = 'r', linestyle='--')
    axs[4,4].set_xlim((-1, 2))
    axs[4,4].spines['right'].set_visible(False)
    axs[4,4].spines['top'].set_visible(False)
    axs[4,4].set_xlabel('Time from Onset2 (s)')
    axs[4,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[5,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[5,0].set_title('VisStim1 Aligned Epoch3.\n') 
    axs[5,0].axvline(x = 0, color = 'r', linestyle='--')
    axs[5,0].set_xlim((-1, 2))
    axs[5,0].spines['right'].set_visible(False)
    axs[5,0].spines['top'].set_visible(False)
    axs[5,0].set_xlabel('Time from VisStim1 (s)')
    axs[5,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[5,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[5,1].axvline(x = 0, color = 'r', linestyle='--')
    axs[5,1].set_xlim((-1, 2))
    axs[5,1].spines['right'].set_visible(False)
    axs[5,1].spines['top'].set_visible(False)
    axs[5,1].set_title('VisStim2 Aligned Epoch3.\n') 
    axs[5,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs[5,1].set_title('waitforpress2 Aligned Epoch3.\n') 
        axs[5,1].set_xlabel('Time from waitforpress2 (s)')
    axs[5,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[5,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[5,2].set_title('\n Reward Aligned Epoch3.\n') 
    axs[5,2].axvline(x = 0, color = 'r', linestyle='--')
    axs[5,2].set_xlim((-1, 2))
    axs[5,2].spines['right'].set_visible(False)
    axs[5,2].spines['top'].set_visible(False)
    axs[5,2].set_xlabel('Time from Reward (s)')
    axs[5,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[5,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[5,3].set_title('Onset1 Aligned Epoch3.\n') 
    axs[5,3].axvline(x = 0, color = 'r', linestyle='--')
    axs[5,3].set_xlim((-1, 2))
    axs[5,3].spines['right'].set_visible(False)
    axs[5,3].spines['top'].set_visible(False)
    axs[5,3].set_xlabel('Time from Onset1 (s)')
    axs[5,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[5,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[5,4].set_title('Onset2 Aligned Epoch3.\n') 
    axs[5,4].axvline(x = 0, color = 'r', linestyle='--')
    axs[5,4].set_xlim((-1, 2))
    axs[5,4].spines['right'].set_visible(False)
    axs[5,4].spines['top'].set_visible(False)
    axs[5,4].set_xlabel('Time from Onset2 (s)')
    axs[5,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    # LONG TRIALS
    axs[6,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[6,0].set_title('VisStim1 Aligned Epoch1.\n') 
    axs[6,0].axvline(x = 0, color = 'r', linestyle='--')
    axs[6,0].set_xlim((-1, 2))
    axs[6,0].spines['right'].set_visible(False)
    axs[6,0].spines['top'].set_visible(False)
    axs[6,0].set_xlabel('Time from VisStim1 (s)')
    axs[6,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[6,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[6,1].axvline(x = 0, color = 'r', linestyle='--')
    axs[6,1].set_xlim((-1, 2))
    axs[6,1].spines['right'].set_visible(False)
    axs[6,1].spines['top'].set_visible(False)
    axs[6,1].set_title('VisStim2 Aligned Epoch1.\n') 
    axs[6,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs[6,1].set_title('waitforpress2 Aligned Epoch1.\n') 
        axs[6,1].set_xlabel('Time from waitforpress2 (s)')
    axs[6,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[6,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[6,2].set_title('\n LONG TRIALS \n Reward Aligned Epoch1.\n') 
    axs[6,2].axvline(x = 0, color = 'r', linestyle='--')
    axs[6,2].set_xlim((-1, 2))
    axs[6,2].spines['right'].set_visible(False)
    axs[6,2].spines['top'].set_visible(False)
    axs[6,2].set_xlabel('Time from Reward (s)')
    axs[6,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[6,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[6,3].set_title('Onset1 Aligned Epoch1.\n') 
    axs[6,3].axvline(x = 0, color = 'r', linestyle='--')
    axs[6,3].set_xlim((-1, 2))
    axs[6,3].spines['right'].set_visible(False)
    axs[6,3].spines['top'].set_visible(False)
    axs[6,3].set_xlabel('Time from Onset1 (s)')
    axs[6,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[6,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[6,4].set_title('Onset2 Aligned Epoch1.\n') 
    axs[6,4].axvline(x = 0, color = 'r', linestyle='--')
    axs[6,4].set_xlim((-1, 2))
    axs[6,4].spines['right'].set_visible(False)
    axs[6,4].spines['top'].set_visible(False)
    axs[6,4].set_xlabel('Time from Onset2 (s)')
    axs[6,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[7,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[7,0].set_title('VisStim1 Aligned Epoch2.\n') 
    axs[7,0].axvline(x = 0, color = 'r', linestyle='--')
    axs[7,0].set_xlim((-1, 2))
    axs[7,0].spines['right'].set_visible(False)
    axs[7,0].spines['top'].set_visible(False)
    axs[7,0].set_xlabel('Time from VisStim1 (s)')
    axs[7,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[7,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[7,1].axvline(x = 0, color = 'r', linestyle='--')
    axs[7,1].set_xlim((-1, 2))
    axs[7,1].spines['right'].set_visible(False)
    axs[7,1].spines['top'].set_visible(False)
    axs[7,1].set_title('VisStim2 Aligned Epoch2.\n') 
    axs[7,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs[7,1].set_title('waitforpress2 Aligned Epoch2.\n') 
        axs[7,1].set_xlabel('Time from waitforpress2 (s)')
    axs[7,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[7,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[7,2].set_title('\n Reward Aligned Epoch2.\n') 
    axs[7,2].axvline(x = 0, color = 'r', linestyle='--')
    axs[7,2].set_xlim((-1, 2))
    axs[7,2].spines['right'].set_visible(False)
    axs[7,2].spines['top'].set_visible(False)
    axs[7,2].set_xlabel('Time from Reward (s)')
    axs[7,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[7,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[7,3].set_title('Onset1 Aligned Epoch2.\n') 
    axs[7,3].axvline(x = 0, color = 'r', linestyle='--')
    axs[7,3].set_xlim((-1, 2))
    axs[7,3].spines['right'].set_visible(False)
    axs[7,3].spines['top'].set_visible(False)
    axs[7,3].set_xlabel('Time from Onset1 (s)')
    axs[7,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[7,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[7,4].set_title('Onset2 Aligned Epoch2.\n') 
    axs[7,4].axvline(x = 0, color = 'r', linestyle='--')
    axs[7,4].set_xlim((-1, 2))
    axs[7,4].spines['right'].set_visible(False)
    axs[7,4].spines['top'].set_visible(False)
    axs[7,4].set_xlabel('Time from Onset2 (s)')
    axs[7,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[8,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[8,0].set_title('VisStim1 Aligned Epoch3.\n') 
    axs[8,0].axvline(x = 0, color = 'r', linestyle='--')
    axs[8,0].set_xlim((-1, 2))
    axs[8,0].spines['right'].set_visible(False)
    axs[8,0].spines['top'].set_visible(False)
    axs[8,0].set_xlabel('Time from VisStim1 (s)')
    axs[8,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[8,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[8,1].axvline(x = 0, color = 'r', linestyle='--')
    axs[8,1].set_xlim((-1, 2))
    axs[8,1].spines['right'].set_visible(False)
    axs[8,1].spines['top'].set_visible(False)
    axs[8,1].set_title('VisStim2 Aligned Epoch3.\n') 
    axs[8,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs[8,1].set_title('waitforpress2 Aligned Epoch3.\n') 
        axs[8,1].set_xlabel('Time from waitforpress2 (s)')
    axs[8,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[8,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[8,2].set_title('\n Reward Aligned Epoch3.\n') 
    axs[8,2].axvline(x = 0, color = 'r', linestyle='--')
    axs[8,2].set_xlim((-1, 2))
    axs[8,2].spines['right'].set_visible(False)
    axs[8,2].spines['top'].set_visible(False)
    axs[8,2].set_xlabel('Time from Reward (s)')
    axs[8,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[8,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[8,3].set_title('Onset1 Aligned Epoch3.\n') 
    axs[8,3].axvline(x = 0, color = 'r', linestyle='--')
    axs[8,3].set_xlim((-1, 2))
    axs[8,3].spines['right'].set_visible(False)
    axs[8,3].spines['top'].set_visible(False)
    axs[8,3].set_xlabel('Time from Onset1 (s)')
    axs[8,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs[8,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs[8,4].set_title('Onset2 Aligned Epoch3.\n') 
    axs[8,4].axvline(x = 0, color = 'r', linestyle='--')
    axs[8,4].set_xlim((-1, 2))
    axs[8,4].spines['right'].set_visible(False)
    axs[8,4].spines['top'].set_visible(False)
    axs[8,4].set_xlabel('Time from Onset2 (s)')
    axs[8,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    # Initialize the loop for analysis #####################
    for i in range(0 , len(session_id)):
        TrialOutcomes = session_data['outcomes'][i]
        # We have Raw data and extract every thing from it (Times)
        raw_data = session_data['raw'][i]
        session_date = dates[i][2:]
        trial_types = raw_data['TrialTypes']
        # iswarmup = raw_data['IsWarmupTrial']
        opto = session_data['session_opto_tag'][i]
        print('Epoch based analysis:' + session_date)
        # Variables for single session#######################
        # EPOCH1 #####
        # all
        epoch1_vis1_all = []
        epoch1_vis2_all = []
        epoch1_rew_all = []
        epoch1_on1_all = []
        epoch1_on2_all = []
        
        epoch1_vis1_chemo_all = []
        epoch1_vis2_chemo_all = []
        epoch1_rew_chemo_all = []
        epoch1_on1_chemo_all = []
        epoch1_on2_chemo_all = []
        # short
        epoch1_vis1_s = []
        epoch1_vis2_s = []
        epoch1_rew_s = []
        epoch1_on1_s = []
        epoch1_on2_s = []
        
        epoch1_vis1_chemo_s = []
        epoch1_vis2_chemo_s = []
        epoch1_rew_chemo_s = []
        epoch1_on1_chemo_s = []
        epoch1_on2_chemo_s = []
        # long
        epoch1_vis1_l = []
        epoch1_vis2_l = []
        epoch1_rew_l = []
        epoch1_on1_l = []
        epoch1_on2_l = []
        
        epoch1_vis1_chemo_l = []
        epoch1_vis2_chemo_l = []
        epoch1_rew_chemo_l = []
        epoch1_on1_chemo_l = []
        epoch1_on2_chemo_l = []
        # EPOCH2 #####
        # all
        epoch2_vis1_all = []
        epoch2_vis2_all = []
        epoch2_rew_all = []
        epoch2_on1_all = []
        epoch2_on2_all = []
        
        epoch2_vis1_chemo_all = []
        epoch2_vis2_chemo_all = []
        epoch2_rew_chemo_all = []
        epoch2_on1_chemo_all = []
        epoch2_on2_chemo_all = []
        # short
        epoch2_vis1_s = []
        epoch2_vis2_s = []
        epoch2_rew_s = []
        epoch2_on1_s = []
        epoch2_on2_s = []
        
        epoch2_vis1_chemo_s = []
        epoch2_vis2_chemo_s = []
        epoch2_rew_chemo_s = []
        epoch2_on1_chemo_s = []
        epoch2_on2_chemo_s = []
        # long
        epoch2_vis1_l = []
        epoch2_vis2_l = []
        epoch2_rew_l = []
        epoch2_on1_l = []
        epoch2_on2_l = []
        
        epoch2_vis1_chemo_l = []
        epoch2_vis2_chemo_l = []
        epoch2_rew_chemo_l = []
        epoch2_on1_chemo_l = []
        epoch2_on2_chemo_l = []
        # EPOCH3 #####
        # all
        epoch3_vis1_all = []
        epoch3_vis2_all = []
        epoch3_rew_all = []
        epoch3_on1_all = []
        epoch3_on2_all = []
        
        epoch3_vis1_chemo_all = []
        epoch3_vis2_chemo_all = []
        epoch3_rew_chemo_all = []
        epoch3_on1_chemo_all = []
        epoch3_on2_chemo_all = []
        # short
        epoch3_vis1_s = []
        epoch3_vis2_s = []
        epoch3_rew_s = []
        epoch3_on1_s = []
        epoch3_on2_s = []
        
        epoch3_vis1_chemo_s = []
        epoch3_vis2_chemo_s = []
        epoch3_rew_chemo_s = []
        epoch3_on1_chemo_s = []
        epoch3_on2_chemo_s = []
        # long
        epoch3_vis1_l = []
        epoch3_vis2_l = []
        epoch3_rew_l = []
        epoch3_on1_l = []
        epoch3_on2_l = []
        
        epoch3_vis1_chemo_l = []
        epoch3_vis2_chemo_l = []
        epoch3_rew_chemo_l = []
        epoch3_on1_chemo_l = []
        epoch3_on2_chemo_l = []
        #####################################################
        Block_size = session_data['raw'][i]['TrialSettings'][0]['GUI']['NumTrialsPerBlock']
        nTrials = session_data['raw'][i]['nTrials']
        epoch1 , epoch2 , epoch3 = epoching_session(nTrials, Block_size)
        
        if nTrials <= Block_size:
            print('not adequate trials for analysis in session:',i)
            continue
        
        # NOTE: we pass the first block
        for trial in range(Block_size ,len(TrialOutcomes)):
            
            if np.isnan(isSelfTimedMode[i][trial]):
                continue
            
            # Passing warmup trials 
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
                reward = int(trial_states['Reward'][0]*1000)
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
                # First section of lever pushing for rewarded trials
                rotatory1 = int(trial_event['RotaryEncoder1_1'][0]*1000) 
                rotatory2 = int(trial_event['RotaryEncoder1_1'][1]*1000) 
                    
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
                
                # variables we need fo visguided mice
                if any(0 in row for row in isSelfTimedMode):
                    VisDetect2 = int(trial_states['VisDetect2'][0]*1000)
                    VisualStimulus2 = int(trial_states['VisualStimulus2'][0]*1000)
                    # Onset2
                    on2_vel = velocity_onset(velocity,VisDetect2,rotatory2)
                # variables just neded for selftime mice
                else:
                    waitforpress2 = int(trial_states['WaitForPress2'][0]*1000)
                    # Onset2
                    on2_vel = velocity_onset(velocity,waitforpress2,rotatory2) # type: ignore
                    #NOTE: I make the waitpress2 equal to vis2 to make code smaller
                    VisualStimulus2 = waitforpress2
                
                if VisualStimulus1 > on1_vel: # type: ignore
                    continue
                
                    
                if VisualStimulus1 < 1000: # type: ignore
                    # Calculate the number of zeros to prepend
                    num_zeros = 1000 - VisualStimulus1 # type: ignore
                    # Pad with zeros at the beginning
                    encoder_positions_aligned_vis1 = np.concatenate((np.zeros(num_zeros), encoder_positions_aligned_vis1))
                    VisualStimulus1 = 1000
                    VisualStimulus2 = VisualStimulus2 + num_zeros
                    on1_vel = on1_vel + num_zeros # type: ignore
                    on2_vel = on2_vel + num_zeros # type: ignore
                    rotatory2 = rotatory2 + num_zeros
                    reward = reward + num_zeros
                    
                # Now defining vis1, vis2, reward (rotatory2), onset1 and onset2 for states: chemo, non_chemo,// control, chemo, opto for 3 epochs and 2 Long and Short
                if trial in epoch1:
                    # All trials
                    if chemo_labels[i] == 1:
                        # Needed for single session plots
                        epoch1_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch1_vis2_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                        epoch1_rew_chemo_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                        epoch1_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        epoch1_on2_chemo_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        # Pooled
                        p_epoch1_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        p_epoch1_vis2_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                        p_epoch1_rew_chemo_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                        p_epoch1_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        p_epoch1_on2_chemo_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                    else:
                        # NEEDED for single session plots
                        epoch1_vis1_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch1_vis2_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                        epoch1_rew_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                        epoch1_on1_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        epoch1_on2_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                        if opto[trial] == 1:
                            # Pooled
                            p_epoch1_vis1_opto_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch1_vis2_opto_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            p_epoch1_rew_opto_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            p_epoch1_on1_opto_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch1_on2_opto_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        else:
                            # Pooled
                            p_epoch1_vis1_con_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch1_vis2_con_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            p_epoch1_rew_con_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            p_epoch1_on1_con_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch1_on2_con_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                    # Short trials   
                    if trial_types[trial] == 1:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch1_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch1_vis2_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            epoch1_rew_chemo_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            epoch1_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch1_on2_chemo_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch1_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch1_vis2_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            p_epoch1_rew_chemo_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            p_epoch1_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch1_on2_chemo_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch1_vis1_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch1_vis2_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            epoch1_rew_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            epoch1_on1_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch1_on2_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch1_vis1_opto_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch1_vis2_opto_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                                p_epoch1_rew_opto_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                                p_epoch1_on1_opto_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch1_on2_opto_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch1_vis1_con_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch1_vis2_con_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                                p_epoch1_rew_con_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                                p_epoch1_on1_con_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch1_on2_con_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                    # Long trials
                    else:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch1_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch1_vis2_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            epoch1_rew_chemo_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            epoch1_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch1_on2_chemo_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch1_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch1_vis2_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            p_epoch1_rew_chemo_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            p_epoch1_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch1_on2_chemo_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch1_vis1_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch1_vis2_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            epoch1_rew_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            epoch1_on1_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch1_on2_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch1_vis1_opto_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch1_vis2_opto_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                                p_epoch1_rew_opto_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                                p_epoch1_on1_opto_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch1_on2_opto_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch1_vis1_con_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch1_vis2_con_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                                p_epoch1_rew_con_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                                p_epoch1_on1_con_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch1_on2_con_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                elif trial in epoch2:
                    # All trials
                    if chemo_labels[i] == 1:
                        # Needed for single session plots
                        epoch2_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch2_vis2_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                        epoch2_rew_chemo_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                        epoch2_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        epoch2_on2_chemo_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        # Pooled
                        p_epoch2_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        p_epoch2_vis2_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                        p_epoch2_rew_chemo_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                        p_epoch2_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        p_epoch2_on2_chemo_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                    else:
                        # NEEDED for single session plots
                        epoch2_vis1_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch2_vis2_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                        epoch2_rew_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                        epoch2_on1_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        epoch2_on2_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                        if opto[trial] == 1:
                            # Pooled
                            p_epoch2_vis1_opto_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch2_vis2_opto_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            p_epoch2_rew_opto_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            p_epoch2_on1_opto_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch2_on2_opto_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        else:
                            # Pooled
                            p_epoch2_vis1_con_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch2_vis2_con_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            p_epoch2_rew_con_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            p_epoch2_on1_con_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch2_on2_con_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                    # Short trials   
                    if trial_types[trial] == 1:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch2_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch2_vis2_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            epoch2_rew_chemo_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            epoch2_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch2_on2_chemo_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch2_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch2_vis2_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            p_epoch2_rew_chemo_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            p_epoch2_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch2_on2_chemo_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch2_vis1_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch2_vis2_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            epoch2_rew_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            epoch2_on1_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch2_on2_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch2_vis1_opto_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch2_vis2_opto_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                                p_epoch2_rew_opto_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                                p_epoch2_on1_opto_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch2_on2_opto_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch2_vis1_con_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch2_vis2_con_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                                p_epoch2_rew_con_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                                p_epoch2_on1_con_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch2_on2_con_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                    # Long trials
                    else:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch2_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch2_vis2_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            epoch2_rew_chemo_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            epoch2_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch2_on2_chemo_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch2_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch2_vis2_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            p_epoch2_rew_chemo_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            p_epoch2_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch2_on2_chemo_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                        else:
                            # NEEDED for single session plots
                            epoch2_vis1_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch2_vis2_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            epoch2_rew_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            epoch2_on1_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch2_on2_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch2_vis1_opto_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch2_vis2_opto_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                                p_epoch2_rew_opto_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                                p_epoch2_on1_opto_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch2_on2_opto_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch2_vis1_con_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch2_vis2_con_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                                p_epoch2_rew_con_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                                p_epoch2_on1_con_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch2_on2_con_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                elif trial in epoch3:
                    # All trials
                    if chemo_labels[i] == 1:
                        # Needed for single session plots
                        epoch3_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch3_vis2_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                        epoch3_rew_chemo_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                        epoch3_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        epoch3_on2_chemo_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        # Pooled
                        p_epoch3_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        p_epoch3_vis2_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                        p_epoch3_rew_chemo_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                        p_epoch3_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        p_epoch3_on2_chemo_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                    else:
                        # NEEDED for single session plots
                        epoch3_vis1_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch3_vis2_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                        epoch3_rew_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                        epoch3_on1_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        epoch3_on2_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                        if opto[trial] == 1:
                            # Pooled
                            p_epoch3_vis1_opto_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch3_vis2_opto_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            p_epoch3_rew_opto_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            p_epoch3_on1_opto_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch3_on2_opto_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        else:
                            # Pooled
                            p_epoch3_vis1_con_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch3_vis2_con_all.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            p_epoch3_rew_con_all.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            p_epoch3_on1_con_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch3_on2_con_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                    # Short trials   
                    if trial_types[trial] == 1:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch3_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch3_vis2_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            epoch3_rew_chemo_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            epoch3_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch3_on2_chemo_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch3_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch3_vis2_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            p_epoch3_rew_chemo_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            p_epoch3_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch3_on2_chemo_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch3_vis1_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch3_vis2_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            epoch3_rew_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            epoch3_on1_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch3_on2_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch3_vis1_opto_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch3_vis2_opto_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                                p_epoch3_rew_opto_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                                p_epoch3_on1_opto_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch3_on2_opto_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch3_vis1_con_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch3_vis2_con_s.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                                p_epoch3_rew_con_s.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                                p_epoch3_on1_con_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch3_on2_con_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                    # Long trials
                    else:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch3_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch3_vis2_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            epoch3_rew_chemo_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            epoch3_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch3_on2_chemo_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch3_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch3_vis2_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            p_epoch3_rew_chemo_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            p_epoch3_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch3_on2_chemo_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch3_vis1_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch3_vis2_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                            epoch3_rew_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                            epoch3_on1_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch3_on2_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch3_vis1_opto_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch3_vis2_opto_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                                p_epoch3_rew_opto_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                                p_epoch3_on1_opto_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch3_on2_opto_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch3_vis1_con_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch3_vis2_con_l.append(encoder_positions_aligned_vis1[VisualStimulus2-1000:VisualStimulus2+2000])
                                p_epoch3_rew_con_l.append(encoder_positions_aligned_vis1[reward-1000:reward+2000])
                                p_epoch3_on1_con_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch3_on2_con_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
            elif TrialOutcomes[trial] == 'EarlyPress2':
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
                if int(string) < 20240804:
                    punish1 = int(trial_states['Punish'][0]*1000)
                    punish2 = int(trial_states['Punish'][1]*1000)
                    if isinstance( trial_event['RotaryEncoder1_1'], float):
                        rotatory1 = int(trial_event['RotaryEncoder1_1'] * 1000)
                    else:
                        rotatory1 =  int(trial_event['RotaryEncoder1_1'][0] * 1000)
                elif 20240804 <= int(string):
                    punish1 = int(trial_states['EarlyPress2Punish'][0]*1000)
                    punish2 = int(trial_event['SoftCode1'][-1]*1000)
                    if isinstance( trial_event['RotaryEncoder1_1'], float):
                        rotatory1 = int(trial_event['RotaryEncoder1_1'] * 1000)
                    else:
                        rotatory1 =  int(trial_event['RotaryEncoder1_1'][0] * 1000)
                    
                if punish1 > punish2: # type: ignore
                    print('Punish start is larger than punish end', trial + 1)
                
                base_line = encoder_positions_aligned_vis1[:VisualStimulus1]
                base_line = base_line[~np.isnan(base_line)]
                base_line = np.mean(base_line)
                
                # Checking the baseline value
                if base_line >= 0.9:
                    print('trial starts with larger value:', trial + 1)
                    continue
                
                if rotatory1 > 100000:
                    continue
                encoder_positions_aligned_vis1 = savgol_filter(encoder_positions_aligned_vis1, window_length=40, polyorder=3)
                velocity = np.gradient(encoder_positions_aligned_vis1, encoder_times_vis1)
                velocity = savgol_filter(velocity, window_length=40, polyorder=1)
                
                # Onset1
                on1_vel = velocity_onset(velocity,VisDetect1,rotatory1)
                
                # variables we need fo visguided mice
                if any(0 in row for row in isSelfTimedMode):
                    PreVis2Delay = int(trial_states['PreVis2Delay'][0]*1000)
                    # Onset2
                    on2_vel = velocity_onset(velocity,PreVis2Delay,punish1) # type: ignore
                    # variables just neded for selftime mice
                else:
                    PrePress2Delay = int(trial_states['PrePress2Delay'][0]*1000)
                    # Onset2
                    on2_vel = velocity_onset(velocity,PrePress2Delay,punish1) # type: ignore
                    
                if VisualStimulus1 > on1_vel: # type: ignore
                    continue
                
                if VisualStimulus1 < 1000: # type: ignore
                    # Calculate the number of zeros to prepend
                    num_zeros = 1000 - VisualStimulus1 # type: ignore
                    # Pad with zeros at the beginning
                    encoder_positions_aligned_vis1 = np.concatenate((np.zeros(num_zeros), encoder_positions_aligned_vis1))
                    VisualStimulus1 = 1000
                    on1_vel = on1_vel + num_zeros # type: ignore
                    on2_vel = on2_vel + num_zeros # type: ignore
                
                # Now defining vis1, vis2, reward (rotatory2), onset1 and onset2 for states: chemo, non_chemo,// control, chemo, opto for 3 epochs and 2 Long and Short
                if trial in epoch1:
                    # All trials
                    if chemo_labels[i] == 1:
                        # Needed for single session plots
                        epoch1_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch1_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        epoch1_on2_chemo_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        # Pooled
                        p_epoch1_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        p_epoch1_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        p_epoch1_on2_chemo_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                    else:
                        # NEEDED for single session plots
                        epoch1_vis1_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch1_on1_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        epoch1_on2_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                        if opto[trial] == 1:
                            # Pooled
                            p_epoch1_vis1_opto_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch1_on1_opto_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch1_on2_opto_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        else:
                            # Pooled
                            p_epoch1_vis1_con_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch1_on1_con_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch1_on2_con_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                    # Short trials   
                    if trial_types[trial] == 1:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch1_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch1_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch1_on2_chemo_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch1_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch1_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch1_on2_chemo_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch1_vis1_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch1_on1_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch1_on2_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch1_vis1_opto_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch1_on1_opto_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch1_on2_opto_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch1_vis1_con_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch1_on1_con_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch1_on2_con_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                    # Long trials
                    else:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch1_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch1_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch1_on2_chemo_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch1_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch1_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch1_on2_chemo_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch1_vis1_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch1_on1_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch1_on2_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch1_vis1_opto_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch1_on1_opto_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch1_on2_opto_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch1_vis1_con_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch1_on1_con_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch1_on2_con_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                elif trial in epoch2:
                    # All trials
                    if chemo_labels[i] == 1:
                        # Needed for single session plots
                        epoch2_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch2_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        epoch2_on2_chemo_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        # Pooled
                        p_epoch2_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        p_epoch2_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        p_epoch2_on2_chemo_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                    else:
                        # NEEDED for single session plots
                        epoch2_vis1_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch2_on1_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        epoch2_on2_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                        if opto[trial] == 1:
                            # Pooled
                            p_epoch2_vis1_opto_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch2_on1_opto_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch2_on2_opto_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        else:
                            # Pooled
                            p_epoch2_vis1_con_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch2_on1_con_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch2_on2_con_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                    # Short trials   
                    if trial_types[trial] == 1:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch2_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch2_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch2_on2_chemo_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch2_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch2_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch2_on2_chemo_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch2_vis1_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch2_on1_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch2_on2_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch2_vis1_opto_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch2_on1_opto_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch2_on2_opto_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch2_vis1_con_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch2_on1_con_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch2_on2_con_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                    # Long trials
                    else:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch2_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch2_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch2_on2_chemo_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch2_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch2_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch2_on2_chemo_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                        else:
                            # NEEDED for single session plots
                            epoch2_vis1_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch2_on1_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch2_on2_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch2_vis1_opto_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch2_on1_opto_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch2_on2_opto_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch2_vis1_con_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch2_on1_con_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch2_on2_con_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                elif trial in epoch3:
                    # All trials
                    if chemo_labels[i] == 1:
                        # Needed for single session plots
                        epoch3_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch3_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        epoch3_on2_chemo_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        # Pooled
                        p_epoch3_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        p_epoch3_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        p_epoch3_on2_chemo_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                    else:
                        # NEEDED for single session plots
                        epoch3_vis1_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch3_on1_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        epoch3_on2_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        
                        if opto[trial] == 1:
                            # Pooled
                            p_epoch3_vis1_opto_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch3_on1_opto_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch3_on2_opto_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                        else:
                            # Pooled
                            p_epoch3_vis1_con_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch3_on1_con_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch3_on2_con_all.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                    # Short trials   
                    if trial_types[trial] == 1:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch3_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch3_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch3_on2_chemo_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch3_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch3_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch3_on2_chemo_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch3_vis1_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch3_on1_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch3_on2_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch3_vis1_opto_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch3_on1_opto_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch3_on2_opto_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch3_vis1_con_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch3_on1_con_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch3_on2_con_s.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                    # Long trials
                    else:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch3_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch3_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch3_on2_chemo_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch3_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch3_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            p_epoch3_on2_chemo_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch3_vis1_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch3_on1_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            epoch3_on2_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch3_vis1_opto_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch3_on1_opto_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch3_on2_opto_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch3_vis1_con_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch3_on1_con_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                                p_epoch3_on2_con_l.append(encoder_positions_aligned_vis1[on2_vel-1000:on2_vel+2000]) # type: ignore 
            # For DidNotPress2 (WE have the press one only)
            elif TrialOutcomes[trial] == 'DidNotPress2':
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
                
                LeverRetract1 = int(trial_states['LeverRetract1'][1]*1000) # End of lever retract1
                # First section of lever pushing for rewarded trials
                if isinstance( trial_event['RotaryEncoder1_1'], float):
                    rotatory1 = int(trial_event['RotaryEncoder1_1'] * 1000)
                else:
                    rotatory1 =  int(trial_event['RotaryEncoder1_1'][0] * 1000) 
                
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
                
                if rotatory1 > 100000:
                    print('Didnotpress2 is longer than 100 s: ', trial + 1)
                    continue
                # Onset1
                on1_vel = velocity_onset(velocity,VisDetect1,rotatory1)
                
                
                if VisualStimulus1 > on1_vel: # type: ignore
                    continue
                
                if VisualStimulus1 < 1000: # type: ignore
                    # Calculate the number of zeros to prepend
                    num_zeros = 1000 - VisualStimulus1 # type: ignore
                    # Pad with zeros at the beginning
                    encoder_positions_aligned_vis1 = np.concatenate((np.zeros(num_zeros), encoder_positions_aligned_vis1))
                    VisualStimulus1 = 1000
                    on1_vel = on1_vel + num_zeros # type: ignore
                
                # Now defining vis1, vis2, reward (rotatory2), onset1 and onset2 for states: chemo, non_chemo,// control, chemo, opto for 3 epochs and 2 Long and Short
                if trial in epoch1:
                    # All trials
                    if chemo_labels[i] == 1:
                        # Needed for single session plots
                        epoch1_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch1_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        # Pooled
                        p_epoch1_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        p_epoch1_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        
                    else:
                        # NEEDED for single session plots
                        epoch1_vis1_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch1_on1_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        
                        if opto[trial] == 1:
                            # Pooled
                            p_epoch1_vis1_opto_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch1_on1_opto_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        else:
                            # Pooled
                            p_epoch1_vis1_con_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch1_on1_con_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                    # Short trials   
                    if trial_types[trial] == 1:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch1_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch1_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch1_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch1_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch1_vis1_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch1_on1_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch1_vis1_opto_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch1_on1_opto_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch1_vis1_con_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch1_on1_con_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                    # Long trials
                    else:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch1_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch1_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch1_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch1_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch1_vis1_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch1_on1_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch1_vis1_opto_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch1_on1_opto_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch1_vis1_con_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch1_on1_con_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                elif trial in epoch2:
                    # All trials
                    if chemo_labels[i] == 1:
                        # Needed for single session plots
                        epoch2_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch2_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        # Pooled
                        p_epoch2_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        p_epoch2_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        
                    else:
                        # NEEDED for single session plots
                        epoch2_vis1_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch2_on1_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        
                        if opto[trial] == 1:
                            # Pooled
                            p_epoch2_vis1_opto_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch2_on1_opto_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        else:
                            # Pooled
                            p_epoch2_vis1_con_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch2_on1_con_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                    # Short trials   
                    if trial_types[trial] == 1:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch2_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch2_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch2_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch2_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch2_vis1_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch2_on1_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch2_vis1_opto_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch2_on1_opto_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch2_vis1_con_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch2_on1_con_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                    # Long trials
                    else:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch2_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch2_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch2_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch2_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        
                        else:
                            # NEEDED for single session plots
                            epoch2_vis1_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch2_on1_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch2_vis1_opto_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch2_on1_opto_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch2_vis1_con_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch2_on1_con_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                elif trial in epoch3:
                    # All trials
                    if chemo_labels[i] == 1:
                        # Needed for single session plots
                        epoch3_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch3_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        # Pooled
                        p_epoch3_vis1_chemo_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        p_epoch3_on1_chemo_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        
                    else:
                        # NEEDED for single session plots
                        epoch3_vis1_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                        epoch3_on1_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        
                        if opto[trial] == 1:
                            # Pooled
                            p_epoch3_vis1_opto_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch3_on1_opto_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                        else:
                            # Pooled
                            p_epoch3_vis1_con_all.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch3_on1_con_all.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                    # Short trials   
                    if trial_types[trial] == 1:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch3_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch3_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch3_vis1_chemo_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch3_on1_chemo_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch3_vis1_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch3_on1_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch3_vis1_opto_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch3_on1_opto_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch3_vis1_con_s.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch3_on1_con_s.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                    # Long trials
                    else:
                        if chemo_labels[i] == 1:
                            # Needed for single session plots
                            epoch3_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch3_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            # Pooled
                            p_epoch3_vis1_chemo_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            p_epoch3_on1_chemo_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            
                        else:
                            # NEEDED for single session plots
                            epoch3_vis1_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                            epoch3_on1_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            
                            if opto[trial] == 1:
                                # Pooled
                                p_epoch3_vis1_opto_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch3_on1_opto_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore
                            else:
                                # Pooled
                                p_epoch3_vis1_con_l.append(encoder_positions_aligned_vis1[VisualStimulus1-1000:VisualStimulus1+2000])
                                p_epoch3_on1_con_l.append(encoder_positions_aligned_vis1[on1_vel-1000:on1_vel+2000]) # type: ignore 
        # plotting single sessions
        if chemo_labels[i] == 1:
            r = r + 1
        else:
            k = k + 1
        ######## epoch1 all
        if len(epoch1_vis1_all) > 0: 
            axs[0,0].plot(encoder_times - 1,np.mean(epoch1_vis1_all, axis=0), color = black[k])
        if len(epoch1_vis1_chemo_all) > 0:
            axs[0,0].plot(encoder_times - 1,np.mean(epoch1_vis1_chemo_all, axis=0), color = red[r]) 
            
        if len(epoch1_vis2_all) > 0: 
            axs[0,1].plot(encoder_times - 1,np.mean(epoch1_vis2_all, axis=0), color = black[k])
        if len(epoch1_vis2_chemo_all) > 0:
            axs[0,1].plot(encoder_times - 1,np.mean(epoch1_vis2_chemo_all, axis=0), color = red[r]) 
        
        if len(epoch1_rew_all) > 0: 
            axs[0,2].plot(encoder_times - 1,np.mean(epoch1_rew_all, axis=0), color = black[k])
        if len(epoch1_rew_chemo_all) > 0:
            axs[0,2].plot(encoder_times - 1,np.mean(epoch1_rew_chemo_all, axis=0), color = red[r])
            
        if len(epoch1_on1_all) > 0: 
            axs[0,3].plot(encoder_times - 1,np.mean(epoch1_on1_all, axis=0), color = black[k])
        if len(epoch1_on1_chemo_all) > 0:
            axs[0,3].plot(encoder_times - 1,np.mean(epoch1_on1_chemo_all, axis=0), color = red[r])
            
        if len(epoch1_on2_all) > 0: 
            axs[0,4].plot(encoder_times - 1,np.mean(epoch1_on2_all, axis=0), color = black[k], label = dates[i][4:])
        if len(epoch1_on2_chemo_all) > 0:
            axs[0,4].plot(encoder_times - 1,np.mean(epoch1_on2_chemo_all, axis=0), color = red[r], label = dates[i][4:])
        ######## epoch2 all
        if len(epoch2_vis1_all) > 0: 
            axs[1,0].plot(encoder_times - 1,np.mean(epoch2_vis1_all, axis=0), color = black[k])
        if len(epoch2_vis1_chemo_all) > 0:
            axs[1,0].plot(encoder_times - 1,np.mean(epoch2_vis1_chemo_all, axis=0), color = red[r]) 
            
        if len(epoch2_vis2_all) > 0: 
            axs[1,1].plot(encoder_times - 1,np.mean(epoch2_vis2_all, axis=0), color = black[k])
        if len(epoch2_vis2_chemo_all) > 0:
            axs[1,1].plot(encoder_times - 1,np.mean(epoch2_vis2_chemo_all, axis=0), color = red[r]) 
        
        if len(epoch2_rew_all) > 0: 
            axs[1,2].plot(encoder_times - 1,np.mean(epoch2_rew_all, axis=0), color = black[k])
        if len(epoch2_rew_chemo_all) > 0:
            axs[1,2].plot(encoder_times - 1,np.mean(epoch2_rew_chemo_all, axis=0), color = red[r])
            
        if len(epoch2_on1_all) > 0: 
            axs[1,3].plot(encoder_times - 1,np.mean(epoch2_on1_all, axis=0), color = black[k])
        if len(epoch2_on1_chemo_all) > 0:
            axs[1,3].plot(encoder_times - 1,np.mean(epoch2_on1_chemo_all, axis=0), color = red[r])
            
        if len(epoch2_on2_all) > 0: 
            axs[1,4].plot(encoder_times - 1,np.mean(epoch2_on2_all, axis=0), color = black[k], label = dates[i][4:])
        if len(epoch2_on2_chemo_all) > 0:
            axs[1,4].plot(encoder_times - 1,np.mean(epoch2_on2_chemo_all, axis=0), color = red[r], label = dates[i][4:])    
        ######## epoch3 all
        if len(epoch3_vis1_all) > 0: 
            axs[2,0].plot(encoder_times - 1,np.mean(epoch3_vis1_all, axis=0), color = black[k])
        if len(epoch3_vis1_chemo_all) > 0:
            axs[2,0].plot(encoder_times - 1,np.mean(epoch3_vis1_chemo_all, axis=0), color = red[r]) 
            
        if len(epoch3_vis2_all) > 0: 
            axs[2,1].plot(encoder_times - 1,np.mean(epoch3_vis2_all, axis=0), color = black[k])
        if len(epoch3_vis2_chemo_all) > 0:
            axs[2,1].plot(encoder_times - 1,np.mean(epoch3_vis2_chemo_all, axis=0), color = red[r]) 
        
        if len(epoch3_rew_all) > 0: 
            axs[2,2].plot(encoder_times - 1,np.mean(epoch3_rew_all, axis=0), color = black[k])
        if len(epoch3_rew_chemo_all) > 0:
            axs[2,2].plot(encoder_times - 1,np.mean(epoch3_rew_chemo_all, axis=0), color = red[r])
            
        if len(epoch3_on1_all) > 0: 
            axs[2,3].plot(encoder_times - 1,np.mean(epoch3_on1_all, axis=0), color = black[k])
        if len(epoch3_on1_chemo_all) > 0:
            axs[2,3].plot(encoder_times - 1,np.mean(epoch3_on1_chemo_all, axis=0), color = red[r])
            
        if len(epoch3_on2_all) > 0: 
            axs[2,4].plot(encoder_times - 1,np.mean(epoch3_on2_all, axis=0), color = black[k], label = dates[i][4:])
        if len(epoch3_on2_chemo_all) > 0:
            axs[2,4].plot(encoder_times - 1,np.mean(epoch3_on2_chemo_all, axis=0), color = red[r], label = dates[i][4:])    
            
        ######## epoch1 short
        if len(epoch1_vis1_s) > 0: 
            axs[3,0].plot(encoder_times - 1,np.mean(epoch1_vis1_s, axis=0), color = black[k])
        if len(epoch1_vis1_chemo_s) > 0:
            axs[3,0].plot(encoder_times - 1,np.mean(epoch1_vis1_chemo_s, axis=0), color = red[r]) 
            
        if len(epoch1_vis2_s) > 0: 
            axs[3,1].plot(encoder_times - 1,np.mean(epoch1_vis2_s, axis=0), color = black[k])
        if len(epoch1_vis2_chemo_s) > 0:
            axs[3,1].plot(encoder_times - 1,np.mean(epoch1_vis2_chemo_s, axis=0), color = red[r]) 
        
        if len(epoch1_rew_s) > 0: 
            axs[3,2].plot(encoder_times - 1,np.mean(epoch1_rew_s, axis=0), color = black[k])
        if len(epoch1_rew_chemo_s) > 0:
            axs[3,2].plot(encoder_times - 1,np.mean(epoch1_rew_chemo_s, axis=0), color = red[r])
            
        if len(epoch1_on1_s) > 0: 
            axs[3,3].plot(encoder_times - 1,np.mean(epoch1_on1_s, axis=0), color = black[k])
        if len(epoch1_on1_chemo_s) > 0:
            axs[3,3].plot(encoder_times - 1,np.mean(epoch1_on1_chemo_s, axis=0), color = red[r])
            
        if len(epoch1_on2_s) > 0: 
            axs[3,4].plot(encoder_times - 1,np.mean(epoch1_on2_s, axis=0), color = black[k], label = dates[i][4:])
        if len(epoch1_on2_chemo_s) > 0:
            axs[3,4].plot(encoder_times - 1,np.mean(epoch1_on2_chemo_s, axis=0), color = red[r], label = dates[i][4:])
        ######## epoch2 short
        if len(epoch2_vis1_s) > 0: 
            axs[4,0].plot(encoder_times - 1,np.mean(epoch2_vis1_s, axis=0), color = black[k])
        if len(epoch2_vis1_chemo_s) > 0:
            axs[4,0].plot(encoder_times - 1,np.mean(epoch2_vis1_chemo_s, axis=0), color = red[r]) 
            
        if len(epoch2_vis2_s) > 0: 
            axs[4,1].plot(encoder_times - 1,np.mean(epoch2_vis2_s, axis=0), color = black[k])
        if len(epoch2_vis2_chemo_s) > 0:
            axs[4,1].plot(encoder_times - 1,np.mean(epoch2_vis2_chemo_s, axis=0), color = red[r]) 
        
        if len(epoch2_rew_s) > 0: 
            axs[4,2].plot(encoder_times - 1,np.mean(epoch2_rew_s, axis=0), color = black[k])
        if len(epoch2_rew_chemo_s) > 0:
            axs[4,2].plot(encoder_times - 1,np.mean(epoch2_rew_chemo_s, axis=0), color = red[r])
            
        if len(epoch2_on1_s) > 0: 
            axs[4,3].plot(encoder_times - 1,np.mean(epoch2_on1_s, axis=0), color = black[k])
        if len(epoch2_on1_chemo_s) > 0:
            axs[4,3].plot(encoder_times - 1,np.mean(epoch2_on1_chemo_s, axis=0), color = red[r])
            
        if len(epoch2_on2_s) > 0:
            axs[4,4].plot(encoder_times - 1,np.mean(epoch2_on2_s, axis=0), color = black[k], label = dates[i][4:])
        if len(epoch2_on2_chemo_s) > 0:
            axs[4,4].plot(encoder_times - 1,np.mean(epoch2_on2_chemo_s, axis=0), color = red[r], label = dates[i][4:])    
        ######## epoch3 short
        if len(epoch3_vis1_s) > 0: 
            axs[5,0].plot(encoder_times - 1,np.mean(epoch3_vis1_s, axis=0), color = black[k])
        if len(epoch3_vis1_chemo_s) > 0:
            axs[5,0].plot(encoder_times - 1,np.mean(epoch3_vis1_chemo_s, axis=0), color = red[r]) 
            
        if len(epoch3_vis2_s) > 0: 
            axs[5,1].plot(encoder_times - 1,np.mean(epoch3_vis2_s, axis=0), color = black[k])
        if len(epoch3_vis2_chemo_s) > 0:
            axs[5,1].plot(encoder_times - 1,np.mean(epoch3_vis2_chemo_s, axis=0), color = red[r]) 
        
        if len(epoch3_rew_s) > 0: 
            axs[5,2].plot(encoder_times - 1,np.mean(epoch3_rew_s, axis=0), color = black[k])
        if len(epoch3_rew_chemo_s) > 0:
            axs[5,2].plot(encoder_times - 1,np.mean(epoch3_rew_chemo_s, axis=0), color = red[r])
            
        if len(epoch3_on1_s) > 0: 
            axs[5,3].plot(encoder_times - 1,np.mean(epoch3_on1_s, axis=0), color = black[k])
        if len(epoch3_on1_chemo_s) > 0:
            axs[5,3].plot(encoder_times - 1,np.mean(epoch3_on1_chemo_s, axis=0), color = red[r])
            
        if len(epoch3_on2_s) > 0: 
            axs[5,4].plot(encoder_times - 1,np.mean(epoch3_on2_s, axis=0), color = black[k], label = dates[i][4:])
        if len(epoch3_on2_chemo_s) > 0:
            axs[5,4].plot(encoder_times - 1,np.mean(epoch3_on2_chemo_s, axis=0), color = red[r], label = dates[i][4:])    
            
        ######## epoch1 long
        if len(epoch1_vis1_l) > 0: 
            axs[6,0].plot(encoder_times - 1,np.mean(epoch1_vis1_l, axis=0), color = black[k])
        if len(epoch1_vis1_chemo_l) > 0:
            axs[6,0].plot(encoder_times - 1,np.mean(epoch1_vis1_chemo_l, axis=0), color = red[r]) 
            
        if len(epoch1_vis2_l) > 0: 
            axs[6,1].plot(encoder_times - 1,np.mean(epoch1_vis2_l, axis=0), color = black[k])
        if len(epoch1_vis2_chemo_l) > 0:
            axs[6,1].plot(encoder_times - 1,np.mean(epoch1_vis2_chemo_l, axis=0), color = red[r]) 
        
        if len(epoch1_rew_l) > 0: 
            axs[6,2].plot(encoder_times - 1,np.mean(epoch1_rew_l, axis=0), color = black[k])
        if len(epoch1_rew_chemo_l) > 0:
            axs[6,2].plot(encoder_times - 1,np.mean(epoch1_rew_chemo_l, axis=0), color = red[r])
            
        if len(epoch1_on1_l) > 0: 
            axs[6,3].plot(encoder_times - 1,np.mean(epoch1_on1_l, axis=0), color = black[k])
        if len(epoch1_on1_chemo_l) > 0:
            axs[6,3].plot(encoder_times - 1,np.mean(epoch1_on1_chemo_l, axis=0), color = red[r])
            
        if len(epoch1_on2_l) > 0: 
            axs[6,4].plot(encoder_times - 1,np.mean(epoch1_on2_l, axis=0), color = black[k], label = dates[i][4:])
        if len(epoch1_on2_chemo_l) > 0:
            axs[6,4].plot(encoder_times - 1,np.mean(epoch1_on2_chemo_l, axis=0), color = red[r], label = dates[i][4:])
        ######## epoch2 long
        if len(epoch2_vis1_l) > 0: 
            axs[7,0].plot(encoder_times - 1,np.mean(epoch2_vis1_l, axis=0), color = black[k])
        if len(epoch2_vis1_chemo_l) > 0:
            axs[7,0].plot(encoder_times - 1,np.mean(epoch2_vis1_chemo_l, axis=0), color = red[r]) 
            
        if len(epoch2_vis2_l) > 0: 
            axs[7,1].plot(encoder_times - 1,np.mean(epoch2_vis2_l, axis=0), color = black[k])
        if len(epoch2_vis2_chemo_l) > 0:
            axs[7,1].plot(encoder_times - 1,np.mean(epoch2_vis2_chemo_l, axis=0), color = red[r]) 
        
        if len(epoch2_rew_l) > 0: 
            axs[7,2].plot(encoder_times - 1,np.mean(epoch2_rew_l, axis=0), color = black[k])
        if len(epoch2_rew_chemo_l) > 0:
            axs[7,2].plot(encoder_times - 1,np.mean(epoch2_rew_chemo_l, axis=0), color = red[r])
            
        if len(epoch2_on1_l) > 0: 
            axs[7,3].plot(encoder_times - 1,np.mean(epoch2_on1_l, axis=0), color = black[k])
        if len(epoch2_on1_chemo_l) > 0:
            axs[7,3].plot(encoder_times - 1,np.mean(epoch2_on1_chemo_l, axis=0), color = red[r])
            
        if len(epoch2_on2_l) > 0: 
            axs[7,4].plot(encoder_times - 1,np.mean(epoch2_on2_l, axis=0), color = black[k], label = dates[i][4:])
        if len(epoch2_on2_chemo_l) > 0:
            axs[7,4].plot(encoder_times - 1,np.mean(epoch2_on2_chemo_l, axis=0), color = red[r], label = dates[i][4:])    
        ######## epoch3 long
        if len(epoch3_vis1_l) > 0: 
            axs[8,0].plot(encoder_times - 1,np.mean(epoch3_vis1_l, axis=0), color = black[k])
        if len(epoch3_vis1_chemo_l) > 0:
            axs[8,0].plot(encoder_times - 1,np.mean(epoch3_vis1_chemo_l, axis=0), color = red[r]) 
            
        if len(epoch3_vis2_l) > 0: 
            axs[8,1].plot(encoder_times - 1,np.mean(epoch3_vis2_l, axis=0), color = black[k])
        if len(epoch3_vis2_chemo_l) > 0:
            axs[8,1].plot(encoder_times - 1,np.mean(epoch3_vis2_chemo_l, axis=0), color = red[r]) 
        
        if len(epoch3_rew_l) > 0: 
            axs[8,2].plot(encoder_times - 1,np.mean(epoch3_rew_l, axis=0), color = black[k])
        if len(epoch3_rew_chemo_l) > 0:
            axs[8,2].plot(encoder_times - 1,np.mean(epoch3_rew_chemo_l, axis=0), color = red[r])
            
        if len(epoch3_on1_l) > 0: 
            axs[8,3].plot(encoder_times - 1,np.mean(epoch3_on1_l, axis=0), color = black[k])
        if len(epoch3_on1_chemo_l) > 0:
            axs[8,3].plot(encoder_times - 1,np.mean(epoch3_on1_chemo_l, axis=0), color = red[r])
            
        if len(epoch3_on2_l) > 0: 
            axs[8,4].plot(encoder_times - 1,np.mean(epoch3_on2_l, axis=0), color = black[k], label = dates[i][4:])
        if len(epoch3_on2_chemo_l) > 0:
            axs[8,4].plot(encoder_times - 1,np.mean(epoch3_on2_chemo_l, axis=0), color = red[r], label = dates[i][4:]) 



    axs[0,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs[1,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs[2,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs[3,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs[4,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)   
    axs[5,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs[6,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)   
    axs[7,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)   
    axs[8,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)                   
    fig.tight_layout()                
    
    ################### POOLDE ####################################################
    fig1, axs1 = plt.subplots(nrows=9, ncols=5, figsize=(27, 36))
    fig1.subplots_adjust(hspace=0.7)
    fig1.suptitle(subject + '\n Average Trajectories Pooled Trials\n')
    # ALL TRIALS
    axs1[0,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[0,0].set_title('VisStim1 Aligned Epoch1.\n') 
    axs1[0,0].axvline(x = 0, color = 'r', linestyle='--')
    axs1[0,0].set_xlim((-1, 2))
    axs1[0,0].spines['right'].set_visible(False)
    axs1[0,0].spines['top'].set_visible(False)
    axs1[0,0].set_xlabel('Time from VisStim1 (s)')
    axs1[0,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[0,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[0,1].axvline(x = 0, color = 'r', linestyle='--')
    axs1[0,1].set_xlim((-1, 2))
    axs1[0,1].spines['right'].set_visible(False)
    axs1[0,1].spines['top'].set_visible(False)
    axs1[0,1].set_title('VisStim2 Aligned Epoch1.\n') 
    axs1[0,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs1[0,1].set_title('waitforpress2 Aligned Epoch1.\n') 
        axs1[0,1].set_xlabel('Time from waitforpress2 (s)')
    axs1[0,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[0,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[0,2].set_title('\n ALL TRIALS \n Reward Aligned Epoch1.\n') 
    axs1[0,2].axvline(x = 0, color = 'r', linestyle='--')
    axs1[0,2].set_xlim((-1, 2))
    axs1[0,2].spines['right'].set_visible(False)
    axs1[0,2].spines['top'].set_visible(False)
    axs1[0,2].set_xlabel('Time from Reward (s)')
    axs1[0,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[0,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[0,3].set_title('Onset1 Aligned Epoch1.\n') 
    axs1[0,3].axvline(x = 0, color = 'r', linestyle='--')
    axs1[0,3].set_xlim((-1, 2))
    axs1[0,3].spines['right'].set_visible(False)
    axs1[0,3].spines['top'].set_visible(False)
    axs1[0,3].set_xlabel('Time from Onset1 (s)')
    axs1[0,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[0,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[0,4].set_title('Onset2 Aligned Epoch1.\n') 
    axs1[0,4].axvline(x = 0, color = 'r', linestyle='--')
    axs1[0,4].set_xlim((-1, 2))
    axs1[0,4].spines['right'].set_visible(False)
    axs1[0,4].spines['top'].set_visible(False)
    axs1[0,4].set_xlabel('Time from Onset2 (s)')
    axs1[0,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[1,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[1,0].set_title('VisStim1 Aligned Epoch2.\n') 
    axs1[1,0].axvline(x = 0, color = 'r', linestyle='--')
    axs1[1,0].set_xlim((-1, 2))
    axs1[1,0].spines['right'].set_visible(False)
    axs1[1,0].spines['top'].set_visible(False)
    axs1[1,0].set_xlabel('Time from VisStim1 (s)')
    axs1[1,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[1,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[1,1].axvline(x = 0, color = 'r', linestyle='--')
    axs1[1,1].set_xlim((-1, 2))
    axs1[1,1].spines['right'].set_visible(False)
    axs1[1,1].spines['top'].set_visible(False)
    axs1[1,1].set_title('VisStim2 Aligned Epoch2.\n') 
    axs1[1,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs1[1,1].set_title('waitforpress2 Aligned Epoch2.\n') 
        axs1[1,1].set_xlabel('Time from waitforpress2 (s)')
    axs1[1,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[1,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[1,2].set_title('\n Reward Aligned Epoch2.\n') 
    axs1[1,2].axvline(x = 0, color = 'r', linestyle='--')
    axs1[1,2].set_xlim((-1, 2))
    axs1[1,2].spines['right'].set_visible(False)
    axs1[1,2].spines['top'].set_visible(False)
    axs1[1,2].set_xlabel('Time from Reward (s)')
    axs1[1,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[1,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[1,3].set_title('Onset1 Aligned Epoch2.\n') 
    axs1[1,3].axvline(x = 0, color = 'r', linestyle='--')
    axs1[1,3].set_xlim((-1, 2))
    axs1[1,3].spines['right'].set_visible(False)
    axs1[1,3].spines['top'].set_visible(False)
    axs1[1,3].set_xlabel('Time from Onset1 (s)')
    axs1[1,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[1,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[1,4].set_title('Onset2 Aligned Epoch2.\n') 
    axs1[1,4].axvline(x = 0, color = 'r', linestyle='--')
    axs1[1,4].set_xlim((-1, 2))
    axs1[1,4].spines['right'].set_visible(False)
    axs1[1,4].spines['top'].set_visible(False)
    axs1[1,4].set_xlabel('Time from Onset2 (s)')
    axs1[1,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[2,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[2,0].set_title('VisStim1 Aligned Epoch3.\n') 
    axs1[2,0].axvline(x = 0, color = 'r', linestyle='--')
    axs1[2,0].set_xlim((-1, 2))
    axs1[2,0].spines['right'].set_visible(False)
    axs1[2,0].spines['top'].set_visible(False)
    axs1[2,0].set_xlabel('Time from VisStim1 (s)')
    axs1[2,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[2,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[2,1].axvline(x = 0, color = 'r', linestyle='--')
    axs1[2,1].set_xlim((-1, 2))
    axs1[2,1].spines['right'].set_visible(False)
    axs1[2,1].spines['top'].set_visible(False)
    axs1[2,1].set_title('VisStim2 Aligned Epoch3.\n') 
    axs1[2,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs1[2,1].set_title('waitforpress2 Aligned Epoch3.\n') 
        axs1[2,1].set_xlabel('Time from waitforpress2 (s)')
    axs1[2,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[2,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[2,2].set_title('\n Reward Aligned Epoch3.\n') 
    axs1[2,2].axvline(x = 0, color = 'r', linestyle='--')
    axs1[2,2].set_xlim((-1, 2))
    axs1[2,2].spines['right'].set_visible(False)
    axs1[2,2].spines['top'].set_visible(False)
    axs1[2,2].set_xlabel('Time from Reward (s)')
    axs1[2,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[2,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[2,3].set_title('Onset1 Aligned Epoch3.\n') 
    axs1[2,3].axvline(x = 0, color = 'r', linestyle='--')
    axs1[2,3].set_xlim((-1, 2))
    axs1[2,3].spines['right'].set_visible(False)
    axs1[2,3].spines['top'].set_visible(False)
    axs1[2,3].set_xlabel('Time from Onset1 (s)')
    axs1[2,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[2,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[2,4].set_title('Onset2 Aligned Epoch3.\n') 
    axs1[2,4].axvline(x = 0, color = 'r', linestyle='--')
    axs1[2,4].set_xlim((-1, 2))
    axs1[2,4].spines['right'].set_visible(False)
    axs1[2,4].spines['top'].set_visible(False)
    axs1[2,4].set_xlabel('Time from Onset2 (s)')
    axs1[2,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    # SHORT TRIALS
    axs1[3,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[3,0].set_title('VisStim1 Aligned Epoch1.\n') 
    axs1[3,0].axvline(x = 0, color = 'r', linestyle='--')
    axs1[3,0].set_xlim((-1, 2))
    axs1[3,0].spines['right'].set_visible(False)
    axs1[3,0].spines['top'].set_visible(False)
    axs1[3,0].set_xlabel('Time from VisStim1 (s)')
    axs1[3,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[3,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[3,1].axvline(x = 0, color = 'r', linestyle='--')
    axs1[3,1].set_xlim((-1, 2))
    axs1[3,1].spines['right'].set_visible(False)
    axs1[3,1].spines['top'].set_visible(False)
    axs1[3,1].set_title('VisStim2 Aligned Epoch1.\n') 
    axs1[3,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs1[3,1].set_title('waitforpress2 Aligned Epoch1.\n') 
        axs1[3,1].set_xlabel('Time from waitforpress2 (s)')
    axs1[3,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[3,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[3,2].set_title('\n SHORT TRIALS \n Reward Aligned Epoch1.\n') 
    axs1[3,2].axvline(x = 0, color = 'r', linestyle='--')
    axs1[3,2].set_xlim((-1, 2))
    axs1[3,2].spines['right'].set_visible(False)
    axs1[3,2].spines['top'].set_visible(False)
    axs1[3,2].set_xlabel('Time from Reward (s)')
    axs1[3,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[3,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[3,3].set_title('Onset1 Aligned Epoch1.\n') 
    axs1[3,3].axvline(x = 0, color = 'r', linestyle='--')
    axs1[3,3].set_xlim((-1, 2))
    axs1[3,3].spines['right'].set_visible(False)
    axs1[3,3].spines['top'].set_visible(False)
    axs1[3,3].set_xlabel('Time from Onset1 (s)')
    axs1[3,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[3,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[3,4].set_title('Onset2 Aligned Epoch1.\n') 
    axs1[3,4].axvline(x = 0, color = 'r', linestyle='--')
    axs1[3,4].set_xlim((-1, 2))
    axs1[3,4].spines['right'].set_visible(False)
    axs1[3,4].spines['top'].set_visible(False)
    axs1[3,4].set_xlabel('Time from Onset2 (s)')
    axs1[3,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[4,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[4,0].set_title('VisStim1 Aligned Epoch2.\n') 
    axs1[4,0].axvline(x = 0, color = 'r', linestyle='--')
    axs1[4,0].set_xlim((-1, 2))
    axs1[4,0].spines['right'].set_visible(False)
    axs1[4,0].spines['top'].set_visible(False)
    axs1[4,0].set_xlabel('Time from VisStim1 (s)')
    axs1[4,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[4,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[4,1].axvline(x = 0, color = 'r', linestyle='--')
    axs1[4,1].set_xlim((-1, 2))
    axs1[4,1].spines['right'].set_visible(False)
    axs1[4,1].spines['top'].set_visible(False)
    axs1[4,1].set_title('VisStim2 Aligned Epoch2.\n') 
    axs1[4,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs1[4,1].set_title('waitforpress2 Aligned Epoch2.\n') 
        axs1[4,1].set_xlabel('Time from waitforpress2 (s)')
    axs1[4,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[4,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[4,2].set_title('\n Reward Aligned Epoch2.\n') 
    axs1[4,2].axvline(x = 0, color = 'r', linestyle='--')
    axs1[4,2].set_xlim((-1, 2))
    axs1[4,2].spines['right'].set_visible(False)
    axs1[4,2].spines['top'].set_visible(False)
    axs1[4,2].set_xlabel('Time from Reward (s)')
    axs1[4,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[4,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[4,3].set_title('Onset1 Aligned Epoch2.\n') 
    axs1[4,3].axvline(x = 0, color = 'r', linestyle='--')
    axs1[4,3].set_xlim((-1, 2))
    axs1[4,3].spines['right'].set_visible(False)
    axs1[4,3].spines['top'].set_visible(False)
    axs1[4,3].set_xlabel('Time from Onset1 (s)')
    axs1[4,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[4,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[4,4].set_title('Onset2 Aligned Epoch2.\n') 
    axs1[4,4].axvline(x = 0, color = 'r', linestyle='--')
    axs1[4,4].set_xlim((-1, 2))
    axs1[4,4].spines['right'].set_visible(False)
    axs1[4,4].spines['top'].set_visible(False)
    axs1[4,4].set_xlabel('Time from Onset2 (s)')
    axs1[4,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[5,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[5,0].set_title('VisStim1 Aligned Epoch3.\n') 
    axs1[5,0].axvline(x = 0, color = 'r', linestyle='--')
    axs1[5,0].set_xlim((-1, 2))
    axs1[5,0].spines['right'].set_visible(False)
    axs1[5,0].spines['top'].set_visible(False)
    axs1[5,0].set_xlabel('Time from VisStim1 (s)')
    axs1[5,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[5,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[5,1].axvline(x = 0, color = 'r', linestyle='--')
    axs1[5,1].set_xlim((-1, 2))
    axs1[5,1].spines['right'].set_visible(False)
    axs1[5,1].spines['top'].set_visible(False)
    axs1[5,1].set_title('VisStim2 Aligned Epoch3.\n') 
    axs1[5,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs1[5,1].set_title('waitforpress2 Aligned Epoch3.\n') 
        axs1[5,1].set_xlabel('Time from waitforpress2 (s)')
    axs1[5,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[5,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[5,2].set_title('\n Reward Aligned Epoch3.\n') 
    axs1[5,2].axvline(x = 0, color = 'r', linestyle='--')
    axs1[5,2].set_xlim((-1, 2))
    axs1[5,2].spines['right'].set_visible(False)
    axs1[5,2].spines['top'].set_visible(False)
    axs1[5,2].set_xlabel('Time from Reward (s)')
    axs1[5,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[5,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[5,3].set_title('Onset1 Aligned Epoch3.\n') 
    axs1[5,3].axvline(x = 0, color = 'r', linestyle='--')
    axs1[5,3].set_xlim((-1, 2))
    axs1[5,3].spines['right'].set_visible(False)
    axs1[5,3].spines['top'].set_visible(False)
    axs1[5,3].set_xlabel('Time from Onset1 (s)')
    axs1[5,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[5,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[5,4].set_title('Onset2 Aligned Epoch3.\n') 
    axs1[5,4].axvline(x = 0, color = 'r', linestyle='--')
    axs1[5,4].set_xlim((-1, 2))
    axs1[5,4].spines['right'].set_visible(False)
    axs1[5,4].spines['top'].set_visible(False)
    axs1[5,4].set_xlabel('Time from Onset2 (s)')
    axs1[5,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    # LONG TRIALS
    axs1[6,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[6,0].set_title('VisStim1 Aligned Epoch1.\n') 
    axs1[6,0].axvline(x = 0, color = 'r', linestyle='--')
    axs1[6,0].set_xlim((-1, 2))
    axs1[6,0].spines['right'].set_visible(False)
    axs1[6,0].spines['top'].set_visible(False)
    axs1[6,0].set_xlabel('Time from VisStim1 (s)')
    axs1[6,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[6,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[6,1].axvline(x = 0, color = 'r', linestyle='--')
    axs1[6,1].set_xlim((-1, 2))
    axs1[6,1].spines['right'].set_visible(False)
    axs1[6,1].spines['top'].set_visible(False)
    axs1[6,1].set_title('VisStim2 Aligned Epoch1.\n') 
    axs1[6,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs1[6,1].set_title('waitforpress2 Aligned Epoch1.\n') 
        axs1[6,1].set_xlabel('Time from waitforpress2 (s)')
    axs1[6,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[6,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[6,2].set_title('\n LONG TRIALS \n Reward Aligned Epoch1.\n') 
    axs1[6,2].axvline(x = 0, color = 'r', linestyle='--')
    axs1[6,2].set_xlim((-1, 2))
    axs1[6,2].spines['right'].set_visible(False)
    axs1[6,2].spines['top'].set_visible(False)
    axs1[6,2].set_xlabel('Time from Reward (s)')
    axs1[6,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[6,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[6,3].set_title('Onset1 Aligned Epoch1.\n') 
    axs1[6,3].axvline(x = 0, color = 'r', linestyle='--')
    axs1[6,3].set_xlim((-1, 2))
    axs1[6,3].spines['right'].set_visible(False)
    axs1[6,3].spines['top'].set_visible(False)
    axs1[6,3].set_xlabel('Time from Onset1 (s)')
    axs1[6,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[6,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[6,4].set_title('Onset2 Aligned Epoch1.\n') 
    axs1[6,4].axvline(x = 0, color = 'r', linestyle='--')
    axs1[6,4].set_xlim((-1, 2))
    axs1[6,4].spines['right'].set_visible(False)
    axs1[6,4].spines['top'].set_visible(False)
    axs1[6,4].set_xlabel('Time from Onset2 (s)')
    axs1[6,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[7,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[7,0].set_title('VisStim1 Aligned Epoch2.\n') 
    axs1[7,0].axvline(x = 0, color = 'r', linestyle='--')
    axs1[7,0].set_xlim((-1, 2))
    axs1[7,0].spines['right'].set_visible(False)
    axs1[7,0].spines['top'].set_visible(False)
    axs1[7,0].set_xlabel('Time from VisStim1 (s)')
    axs1[7,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[7,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[7,1].axvline(x = 0, color = 'r', linestyle='--')
    axs1[7,1].set_xlim((-1, 2))
    axs1[7,1].spines['right'].set_visible(False)
    axs1[7,1].spines['top'].set_visible(False)
    axs1[7,1].set_title('VisStim2 Aligned Epoch2.\n') 
    axs1[7,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs1[7,1].set_title('waitforpress2 Aligned Epoch2.\n') 
        axs1[7,1].set_xlabel('Time from waitforpress2 (s)')
    axs1[7,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[7,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[7,2].set_title('\n Reward Aligned Epoch2.\n') 
    axs1[7,2].axvline(x = 0, color = 'r', linestyle='--')
    axs1[7,2].set_xlim((-1, 2))
    axs1[7,2].spines['right'].set_visible(False)
    axs1[7,2].spines['top'].set_visible(False)
    axs1[7,2].set_xlabel('Time from Reward (s)')
    axs1[7,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[7,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[7,3].set_title('Onset1 Aligned Epoch2.\n') 
    axs1[7,3].axvline(x = 0, color = 'r', linestyle='--')
    axs1[7,3].set_xlim((-1, 2))
    axs1[7,3].spines['right'].set_visible(False)
    axs1[7,3].spines['top'].set_visible(False)
    axs1[7,3].set_xlabel('Time from Onset1 (s)')
    axs1[7,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[7,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[7,4].set_title('Onset2 Aligned Epoch2.\n') 
    axs1[7,4].axvline(x = 0, color = 'r', linestyle='--')
    axs1[7,4].set_xlim((-1, 2))
    axs1[7,4].spines['right'].set_visible(False)
    axs1[7,4].spines['top'].set_visible(False)
    axs1[7,4].set_xlabel('Time from Onset2 (s)')
    axs1[7,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[8,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[8,0].set_title('VisStim1 Aligned Epoch3.\n') 
    axs1[8,0].axvline(x = 0, color = 'r', linestyle='--')
    axs1[8,0].set_xlim((-1, 2))
    axs1[8,0].spines['right'].set_visible(False)
    axs1[8,0].spines['top'].set_visible(False)
    axs1[8,0].set_xlabel('Time from VisStim1 (s)')
    axs1[8,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[8,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[8,1].axvline(x = 0, color = 'r', linestyle='--')
    axs1[8,1].set_xlim((-1, 2))
    axs1[8,1].spines['right'].set_visible(False)
    axs1[8,1].spines['top'].set_visible(False)
    axs1[8,1].set_title('VisStim2 Aligned Epoch3.\n') 
    axs1[8,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs1[8,1].set_title('waitforpress2 Aligned Epoch3.\n') 
        axs1[8,1].set_xlabel('Time from waitforpress2 (s)')
    axs1[8,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[8,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[8,2].set_title('\n Reward Aligned Epoch3.\n') 
    axs1[8,2].axvline(x = 0, color = 'r', linestyle='--')
    axs1[8,2].set_xlim((-1, 2))
    axs1[8,2].spines['right'].set_visible(False)
    axs1[8,2].spines['top'].set_visible(False)
    axs1[8,2].set_xlabel('Time from Reward (s)')
    axs1[8,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[8,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[8,3].set_title('Onset1 Aligned Epoch3.\n') 
    axs1[8,3].axvline(x = 0, color = 'r', linestyle='--')
    axs1[8,3].set_xlim((-1, 2))
    axs1[8,3].spines['right'].set_visible(False)
    axs1[8,3].spines['top'].set_visible(False)
    axs1[8,3].set_xlabel('Time from Onset1 (s)')
    axs1[8,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs1[8,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs1[8,4].set_title('Onset2 Aligned Epoch3.\n') 
    axs1[8,4].axvline(x = 0, color = 'r', linestyle='--')
    axs1[8,4].set_xlim((-1, 2))
    axs1[8,4].spines['right'].set_visible(False)
    axs1[8,4].spines['top'].set_visible(False)
    axs1[8,4].set_xlabel('Time from Onset2 (s)')
    axs1[8,4].set_ylabel('Joystick deflection (deg) Rewarded trials')
    ##############
    # epoch1 ALL
    if len(p_epoch1_vis1_con_all) > 0:
        x = np.mean(p_epoch1_vis1_con_all, axis=0)
        y = stats.sem(p_epoch1_vis1_con_all, axis=0)
        axs1[0,0].plot(encoder_times - 1, x , color = 'k')
        axs1[0,0].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_vis1_chemo_all) > 0:
        x = np.mean(p_epoch1_vis1_chemo_all, axis=0)
        y = stats.sem(p_epoch1_vis1_chemo_all, axis=0)
        axs1[0,0].plot(encoder_times - 1, x , color = 'r')
        axs1[0,0].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_vis1_opto_all) > 0:
        x = np.mean(p_epoch1_vis1_opto_all, axis=0)
        y = stats.sem(p_epoch1_vis1_opto_all, axis=0)
        axs1[0,0].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[0,0].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch1_vis2_con_all) > 0:
        x = np.mean(p_epoch1_vis2_con_all, axis=0)
        y = stats.sem(p_epoch1_vis2_con_all, axis=0)
        axs1[0,1].plot(encoder_times - 1, x , color = 'k')
        axs1[0,1].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_vis2_chemo_all) > 0:
        x = np.mean(p_epoch1_vis2_chemo_all, axis=0)
        y = stats.sem(p_epoch1_vis2_chemo_all, axis=0)
        axs1[0,1].plot(encoder_times - 1, x , color = 'r')
        axs1[0,1].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_vis2_opto_all) > 0:
        x = np.mean(p_epoch1_vis2_opto_all, axis=0)
        y = stats.sem(p_epoch1_vis2_opto_all, axis=0)
        axs1[0,1].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[0,1].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch1_rew_con_all) > 0:
        x = np.mean(p_epoch1_rew_con_all, axis=0)
        y = stats.sem(p_epoch1_rew_con_all, axis=0)
        axs1[0,2].plot(encoder_times - 1, x , color = 'k')
        axs1[0,2].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_rew_chemo_all) > 0:
        x = np.mean(p_epoch1_rew_chemo_all, axis=0)
        y = stats.sem(p_epoch1_rew_chemo_all, axis=0)
        axs1[0,2].plot(encoder_times - 1, x , color = 'r')
        axs1[0,2].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_rew_opto_all) > 0:
        x = np.mean(p_epoch1_rew_opto_all, axis=0)
        y = stats.sem(p_epoch1_rew_opto_all, axis=0)
        axs1[0,2].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[0,2].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch1_on1_con_all) > 0:
        x = np.mean(p_epoch1_on1_con_all, axis=0)
        y = stats.sem(p_epoch1_on1_con_all, axis=0)
        axs1[0,3].plot(encoder_times - 1, x , color = 'k')
        axs1[0,3].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_on1_chemo_all) > 0:
        x = np.mean(p_epoch1_on1_chemo_all, axis=0)
        y = stats.sem(p_epoch1_on1_chemo_all, axis=0)
        axs1[0,3].plot(encoder_times - 1, x , color = 'r')
        axs1[0,3].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_on1_opto_all) > 0:
        x = np.mean(p_epoch1_on1_opto_all, axis=0)
        y = stats.sem(p_epoch1_on1_opto_all, axis=0)
        axs1[0,3].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[0,3].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)

    if len(p_epoch1_on2_con_all) > 0:
        x = np.mean(p_epoch1_on2_con_all, axis=0)
        y = stats.sem(p_epoch1_on2_con_all, axis=0)
        axs1[0,4].plot(encoder_times - 1, x , color = 'k', label = 'Control_All_epoch1')
        axs1[0,4].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_on2_chemo_all) > 0:
        x = np.mean(p_epoch1_on2_chemo_all, axis=0)
        y = stats.sem(p_epoch1_on2_chemo_all, axis=0)
        axs1[0,4].plot(encoder_times - 1, x , color = 'r',label = 'Chemo_All_epoch1')
        axs1[0,4].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_on2_opto_all) > 0:
        x = np.mean(p_epoch1_on2_opto_all, axis=0)
        y = stats.sem(p_epoch1_on2_opto_all, axis=0)
        axs1[0,4].plot(encoder_times - 1, x , color = 'deepskyblue', label = 'Opto_All_epoch1')
        axs1[0,4].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    # epoch2 ALL
    if len(p_epoch2_vis1_con_all) > 0:
        x = np.mean(p_epoch2_vis1_con_all, axis=0)
        y = stats.sem(p_epoch2_vis1_con_all, axis=0)
        axs1[1,0].plot(encoder_times - 1, x , color = 'k')
        axs1[1,0].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_vis1_chemo_all) > 0:
        x = np.mean(p_epoch2_vis1_chemo_all, axis=0)
        y = stats.sem(p_epoch2_vis1_chemo_all, axis=0)
        axs1[1,0].plot(encoder_times - 1, x , color = 'r')
        axs1[1,0].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_vis1_opto_all) > 0:
        x = np.mean(p_epoch2_vis1_opto_all, axis=0)
        y = stats.sem(p_epoch2_vis1_opto_all, axis=0)
        axs1[1,0].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[1,0].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch2_vis2_con_all) > 0:
        x = np.mean(p_epoch2_vis2_con_all, axis=0)
        y = stats.sem(p_epoch2_vis2_con_all, axis=0)
        axs1[1,1].plot(encoder_times - 1, x , color = 'k')
        axs1[1,1].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_vis2_chemo_all) > 0:
        x = np.mean(p_epoch2_vis2_chemo_all, axis=0)
        y = stats.sem(p_epoch2_vis2_chemo_all, axis=0)
        axs1[1,1].plot(encoder_times - 1, x , color = 'r')
        axs1[1,1].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_vis2_opto_all) > 0:
        x = np.mean(p_epoch2_vis2_opto_all, axis=0)
        y = stats.sem(p_epoch2_vis2_opto_all, axis=0)
        axs1[1,1].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[1,1].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch2_rew_con_all) > 0:
        x = np.mean(p_epoch2_rew_con_all, axis=0)
        y = stats.sem(p_epoch2_rew_con_all, axis=0)
        axs1[1,2].plot(encoder_times - 1, x , color = 'k')
        axs1[1,2].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_rew_chemo_all) > 0:
        x = np.mean(p_epoch2_rew_chemo_all, axis=0)
        y = stats.sem(p_epoch2_rew_chemo_all, axis=0)
        axs1[1,2].plot(encoder_times - 1, x , color = 'r')
        axs1[1,2].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_rew_opto_all) > 0:
        x = np.mean(p_epoch2_rew_opto_all, axis=0)
        y = stats.sem(p_epoch2_rew_opto_all, axis=0)
        axs1[1,2].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[1,2].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch2_on1_con_all) > 0:
        x = np.mean(p_epoch2_on1_con_all, axis=0)
        y = stats.sem(p_epoch2_on1_con_all, axis=0)
        axs1[1,3].plot(encoder_times - 1, x , color = 'k')
        axs1[1,3].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_on1_chemo_all) > 0:
        x = np.mean(p_epoch2_on1_chemo_all, axis=0)
        y = stats.sem(p_epoch2_on1_chemo_all, axis=0)
        axs1[1,3].plot(encoder_times - 1, x , color = 'r')
        axs1[1,3].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_on1_opto_all) > 0:
        x = np.mean(p_epoch2_on1_opto_all, axis=0)
        y = stats.sem(p_epoch2_on1_opto_all, axis=0)
        axs1[1,3].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[1,3].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)

    if len(p_epoch2_on2_con_all) > 0:
        x = np.mean(p_epoch2_on2_con_all, axis=0)
        y = stats.sem(p_epoch2_on2_con_all, axis=0)
        axs1[1,4].plot(encoder_times - 1, x , color = 'k', label = 'Control_All_epoch2')
        axs1[1,4].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_on2_chemo_all) > 0:
        x = np.mean(p_epoch2_on2_chemo_all, axis=0)
        y = stats.sem(p_epoch2_on2_chemo_all, axis=0)
        axs1[1,4].plot(encoder_times - 1, x , color = 'r',label = 'Chemo_All_epoch2')
        axs1[1,4].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_on2_opto_all) > 0:
        x = np.mean(p_epoch2_on2_opto_all, axis=0)
        y = stats.sem(p_epoch2_on2_opto_all, axis=0)
        axs1[1,4].plot(encoder_times - 1, x , color = 'deepskyblue', label = 'Opto_All_epoch2')
        axs1[1,4].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    # epoch3 ALL
    if len(p_epoch3_vis1_con_all) > 0:
        x = np.mean(p_epoch3_vis1_con_all, axis=0)
        y = stats.sem(p_epoch3_vis1_con_all, axis=0)
        axs1[2,0].plot(encoder_times - 1, x , color = 'k')
        axs1[2,0].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_vis1_chemo_all) > 0:
        x = np.mean(p_epoch3_vis1_chemo_all, axis=0)
        y = stats.sem(p_epoch3_vis1_chemo_all, axis=0)
        axs1[2,0].plot(encoder_times - 1, x , color = 'r')
        axs1[2,0].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_vis1_opto_all) > 0:
        x = np.mean(p_epoch3_vis1_opto_all, axis=0)
        y = stats.sem(p_epoch3_vis1_opto_all, axis=0)
        axs1[2,0].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[2,0].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch3_vis2_con_all) > 0:
        x = np.mean(p_epoch3_vis2_con_all, axis=0)
        y = stats.sem(p_epoch3_vis2_con_all, axis=0)
        axs1[2,1].plot(encoder_times - 1, x , color = 'k')
        axs1[2,1].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_vis2_chemo_all) > 0:
        x = np.mean(p_epoch3_vis2_chemo_all, axis=0)
        y = stats.sem(p_epoch3_vis2_chemo_all, axis=0)
        axs1[2,1].plot(encoder_times - 1, x , color = 'r')
        axs1[2,1].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_vis2_opto_all) > 0:
        x = np.mean(p_epoch3_vis2_opto_all, axis=0)
        y = stats.sem(p_epoch3_vis2_opto_all, axis=0)
        axs1[2,1].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[2,1].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch3_rew_con_all) > 0:
        x = np.mean(p_epoch3_rew_con_all, axis=0)
        y = stats.sem(p_epoch3_rew_con_all, axis=0)
        axs1[2,2].plot(encoder_times - 1, x , color = 'k')
        axs1[2,2].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_rew_chemo_all) > 0:
        x = np.mean(p_epoch3_rew_chemo_all, axis=0)
        y = stats.sem(p_epoch3_rew_chemo_all, axis=0)
        axs1[2,2].plot(encoder_times - 1, x , color = 'r')
        axs1[2,2].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_rew_opto_all) > 0:
        x = np.mean(p_epoch3_rew_opto_all, axis=0)
        y = stats.sem(p_epoch3_rew_opto_all, axis=0)
        axs1[2,2].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[2,2].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch3_on1_con_all) > 0:
        x = np.mean(p_epoch3_on1_con_all, axis=0)
        y = stats.sem(p_epoch3_on1_con_all, axis=0)
        axs1[2,3].plot(encoder_times - 1, x , color = 'k')
        axs1[2,3].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_on1_chemo_all) > 0:
        x = np.mean(p_epoch3_on1_chemo_all, axis=0)
        y = stats.sem(p_epoch3_on1_chemo_all, axis=0)
        axs1[2,3].plot(encoder_times - 1, x , color = 'r')
        axs1[2,3].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_on1_opto_all) > 0:
        x = np.mean(p_epoch3_on1_opto_all, axis=0)
        y = stats.sem(p_epoch3_on1_opto_all, axis=0)
        axs1[2,3].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[2,3].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)

    if len(p_epoch3_on2_con_all) > 0:
        x = np.mean(p_epoch3_on2_con_all, axis=0)
        y = stats.sem(p_epoch3_on2_con_all, axis=0)
        axs1[2,4].plot(encoder_times - 1, x , color = 'k', label = 'Control_All_epoch3')
        axs1[2,4].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_on2_chemo_all) > 0:
        x = np.mean(p_epoch3_on2_chemo_all, axis=0)
        y = stats.sem(p_epoch3_on2_chemo_all, axis=0)
        axs1[2,4].plot(encoder_times - 1, x , color = 'r',label = 'Chemo_All_epoch3')
        axs1[2,4].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_on2_opto_all) > 0:
        x = np.mean(p_epoch3_on2_opto_all, axis=0)
        y = stats.sem(p_epoch3_on2_opto_all, axis=0)
        axs1[2,4].plot(encoder_times - 1, x , color = 'deepskyblue', label = 'Opto_All_epoch3')
        axs1[2,4].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    # epoch1 short
    if len(p_epoch1_vis1_con_s) > 0:
        x = np.mean(p_epoch1_vis1_con_s, axis=0)
        y = stats.sem(p_epoch1_vis1_con_s, axis=0)
        axs1[3,0].plot(encoder_times - 1, x , color = 'k')
        axs1[3,0].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_vis1_chemo_s) > 0:
        x = np.mean(p_epoch1_vis1_chemo_s, axis=0)
        y = stats.sem(p_epoch1_vis1_chemo_s, axis=0)
        axs1[3,0].plot(encoder_times - 1, x , color = 'r')
        axs1[3,0].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_vis1_opto_s) > 0:
        x = np.mean(p_epoch1_vis1_opto_s, axis=0)
        y = stats.sem(p_epoch1_vis1_opto_s, axis=0)
        axs1[3,0].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[3,0].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch1_vis2_con_s) > 0:
        x = np.mean(p_epoch1_vis2_con_s, axis=0)
        y = stats.sem(p_epoch1_vis2_con_s, axis=0)
        axs1[3,1].plot(encoder_times - 1, x , color = 'k')
        axs1[3,1].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_vis2_chemo_s) > 0:
        x = np.mean(p_epoch1_vis2_chemo_s, axis=0)
        y = stats.sem(p_epoch1_vis2_chemo_s, axis=0)
        axs1[3,1].plot(encoder_times - 1, x , color = 'r')
        axs1[3,1].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_vis2_opto_s) > 0:
        x = np.mean(p_epoch1_vis2_opto_s, axis=0)
        y = stats.sem(p_epoch1_vis2_opto_s, axis=0)
        axs1[3,1].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[3,1].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch1_rew_con_s) > 0:
        x = np.mean(p_epoch1_rew_con_s, axis=0)
        y = stats.sem(p_epoch1_rew_con_s, axis=0)
        axs1[3,2].plot(encoder_times - 1, x , color = 'k')
        axs1[3,2].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_rew_chemo_s) > 0:
        x = np.mean(p_epoch1_rew_chemo_s, axis=0)
        y = stats.sem(p_epoch1_rew_chemo_s, axis=0)
        axs1[3,2].plot(encoder_times - 1, x , color = 'r')
        axs1[3,2].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_rew_opto_s) > 0:
        x = np.mean(p_epoch1_rew_opto_s, axis=0)
        y = stats.sem(p_epoch1_rew_opto_s, axis=0)
        axs1[3,2].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[3,2].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch1_on1_con_s) > 0:
        x = np.mean(p_epoch1_on1_con_s, axis=0)
        y = stats.sem(p_epoch1_on1_con_s, axis=0)
        axs1[3,3].plot(encoder_times - 1, x , color = 'k')
        axs1[3,3].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_on1_chemo_s) > 0:
        x = np.mean(p_epoch1_on1_chemo_s, axis=0)
        y = stats.sem(p_epoch1_on1_chemo_s, axis=0)
        axs1[3,3].plot(encoder_times - 1, x , color = 'r')
        axs1[3,3].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_on1_opto_s) > 0:
        x = np.mean(p_epoch1_on1_opto_s, axis=0)
        y = stats.sem(p_epoch1_on1_opto_s, axis=0)
        axs1[3,3].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[3,3].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)

    if len(p_epoch1_on2_con_s) > 0:
        x = np.mean(p_epoch1_on2_con_s, axis=0)
        y = stats.sem(p_epoch1_on2_con_s, axis=0)
        axs1[3,4].plot(encoder_times - 1, x , color = 'k', label = 'Control_Short_epoch1')
        axs1[3,4].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_on2_chemo_s) > 0:
        x = np.mean(p_epoch1_on2_chemo_s, axis=0)
        y = stats.sem(p_epoch1_on2_chemo_s, axis=0)
        axs1[3,4].plot(encoder_times - 1, x , color = 'r',label = 'Chemo_Short_epoch1')
        axs1[3,4].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_on2_opto_s) > 0:
        x = np.mean(p_epoch1_on2_opto_s, axis=0)
        y = stats.sem(p_epoch1_on2_opto_s, axis=0)
        axs1[3,4].plot(encoder_times - 1, x , color = 'deepskyblue', label = 'Opto_Short_epoch1')
        axs1[3,4].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    # epoch2 short
    if len(p_epoch2_vis1_con_s) > 0:
        x = np.mean(p_epoch2_vis1_con_s, axis=0)
        y = stats.sem(p_epoch2_vis1_con_s, axis=0)
        axs1[4,0].plot(encoder_times - 1, x , color = 'k')
        axs1[4,0].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_vis1_chemo_s) > 0:
        x = np.mean(p_epoch2_vis1_chemo_s, axis=0)
        y = stats.sem(p_epoch2_vis1_chemo_s, axis=0)
        axs1[4,0].plot(encoder_times - 1, x , color = 'r')
        axs1[4,0].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_vis1_opto_s) > 0:
        x = np.mean(p_epoch2_vis1_opto_s, axis=0)
        y = stats.sem(p_epoch2_vis1_opto_s, axis=0)
        axs1[4,0].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[4,0].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch2_vis2_con_s) > 0:
        x = np.mean(p_epoch2_vis2_con_s, axis=0)
        y = stats.sem(p_epoch2_vis2_con_s, axis=0)
        axs1[4,1].plot(encoder_times - 1, x , color = 'k')
        axs1[4,1].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_vis2_chemo_s) > 0:
        x = np.mean(p_epoch2_vis2_chemo_s, axis=0)
        y = stats.sem(p_epoch2_vis2_chemo_s, axis=0)
        axs1[4,1].plot(encoder_times - 1, x , color = 'r')
        axs1[4,1].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_vis2_opto_s) > 0:
        x = np.mean(p_epoch2_vis2_opto_s, axis=0)
        y = stats.sem(p_epoch2_vis2_opto_s, axis=0)
        axs1[4,1].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[4,1].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch2_rew_con_s) > 0:
        x = np.mean(p_epoch2_rew_con_s, axis=0)
        y = stats.sem(p_epoch2_rew_con_s, axis=0)
        axs1[4,2].plot(encoder_times - 1, x , color = 'k')
        axs1[4,2].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_rew_chemo_s) > 0:
        x = np.mean(p_epoch2_rew_chemo_s, axis=0)
        y = stats.sem(p_epoch2_rew_chemo_s, axis=0)
        axs1[4,2].plot(encoder_times - 1, x , color = 'r')
        axs1[4,2].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_rew_opto_s) > 0:
        x = np.mean(p_epoch2_rew_opto_s, axis=0)
        y = stats.sem(p_epoch2_rew_opto_s, axis=0)
        axs1[4,2].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[4,2].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch2_on1_con_s) > 0:
        x = np.mean(p_epoch2_on1_con_s, axis=0)
        y = stats.sem(p_epoch2_on1_con_s, axis=0)
        axs1[4,3].plot(encoder_times - 1, x , color = 'k')
        axs1[4,3].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_on1_chemo_s) > 0:
        x = np.mean(p_epoch2_on1_chemo_s, axis=0)
        y = stats.sem(p_epoch2_on1_chemo_s, axis=0)
        axs1[4,3].plot(encoder_times - 1, x , color = 'r')
        axs1[4,3].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_on1_opto_s) > 0:
        x = np.mean(p_epoch2_on1_opto_s, axis=0)
        y = stats.sem(p_epoch2_on1_opto_s, axis=0)
        axs1[4,3].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[4,3].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)

    if len(p_epoch2_on2_con_s) > 0:
        x = np.mean(p_epoch2_on2_con_s, axis=0)
        y = stats.sem(p_epoch2_on2_con_s, axis=0)
        axs1[4,4].plot(encoder_times - 1, x , color = 'k', label = 'Control_Short_epoch2')
        axs1[4,4].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_on2_chemo_s) > 0:
        x = np.mean(p_epoch2_on2_chemo_s, axis=0)
        y = stats.sem(p_epoch2_on2_chemo_s, axis=0)
        axs1[4,4].plot(encoder_times - 1, x , color = 'r',label = 'Chemo_Short_epoch2')
        axs1[4,4].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_on2_opto_s) > 0:
        x = np.mean(p_epoch2_on2_opto_s, axis=0)
        y = stats.sem(p_epoch2_on2_opto_s, axis=0)
        axs1[4,4].plot(encoder_times - 1, x , color = 'deepskyblue', label = 'Opto_Short_epoch2')
        axs1[4,4].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    # epoch3 short
    if len(p_epoch3_vis1_con_s) > 0:
        x = np.mean(p_epoch3_vis1_con_s, axis=0)
        y = stats.sem(p_epoch3_vis1_con_s, axis=0)
        axs1[5,0].plot(encoder_times - 1, x , color = 'k')
        axs1[5,0].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_vis1_chemo_s) > 0:
        x = np.mean(p_epoch3_vis1_chemo_s, axis=0)
        y = stats.sem(p_epoch3_vis1_chemo_s, axis=0)
        axs1[5,0].plot(encoder_times - 1, x , color = 'r')
        axs1[5,0].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_vis1_opto_s) > 0:
        x = np.mean(p_epoch3_vis1_opto_s, axis=0)
        y = stats.sem(p_epoch3_vis1_opto_s, axis=0)
        axs1[5,0].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[5,0].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch3_vis2_con_s) > 0:
        x = np.mean(p_epoch3_vis2_con_s, axis=0)
        y = stats.sem(p_epoch3_vis2_con_s, axis=0)
        axs1[5,1].plot(encoder_times - 1, x , color = 'k')
        axs1[5,1].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_vis2_chemo_s) > 0:
        x = np.mean(p_epoch3_vis2_chemo_s, axis=0)
        y = stats.sem(p_epoch3_vis2_chemo_s, axis=0)
        axs1[5,1].plot(encoder_times - 1, x , color = 'r')
        axs1[5,1].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_vis2_opto_s) > 0:
        x = np.mean(p_epoch3_vis2_opto_s, axis=0)
        y = stats.sem(p_epoch3_vis2_opto_s, axis=0)
        axs1[5,1].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[5,1].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch3_rew_con_s) > 0:
        x = np.mean(p_epoch3_rew_con_s, axis=0)
        y = stats.sem(p_epoch3_rew_con_s, axis=0)
        axs1[5,2].plot(encoder_times - 1, x , color = 'k')
        axs1[5,2].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_rew_chemo_s) > 0:
        x = np.mean(p_epoch3_rew_chemo_s, axis=0)
        y = stats.sem(p_epoch3_rew_chemo_s, axis=0)
        axs1[5,2].plot(encoder_times - 1, x , color = 'r')
        axs1[5,2].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_rew_opto_s) > 0:
        x = np.mean(p_epoch3_rew_opto_s, axis=0)
        y = stats.sem(p_epoch3_rew_opto_s, axis=0)
        axs1[5,2].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[5,2].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch3_on1_con_s) > 0:
        x = np.mean(p_epoch3_on1_con_s, axis=0)
        y = stats.sem(p_epoch3_on1_con_s, axis=0)
        axs1[5,3].plot(encoder_times - 1, x , color = 'k')
        axs1[5,3].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_on1_chemo_s) > 0:
        x = np.mean(p_epoch3_on1_chemo_s, axis=0)
        y = stats.sem(p_epoch3_on1_chemo_s, axis=0)
        axs1[5,3].plot(encoder_times - 1, x , color = 'r')
        axs1[5,3].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_on1_opto_s) > 0:
        x = np.mean(p_epoch3_on1_opto_s, axis=0)
        y = stats.sem(p_epoch3_on1_opto_s, axis=0)
        axs1[5,3].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[5,3].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)

    if len(p_epoch3_on2_con_s) > 0:
        x = np.mean(p_epoch3_on2_con_s, axis=0)
        y = stats.sem(p_epoch3_on2_con_s, axis=0)
        axs1[5,4].plot(encoder_times - 1, x , color = 'k', label = 'Control_Short_epoch3')
        axs1[5,4].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_on2_chemo_s) > 0:
        x = np.mean(p_epoch3_on2_chemo_s, axis=0)
        y = stats.sem(p_epoch3_on2_chemo_s, axis=0)
        axs1[5,4].plot(encoder_times - 1, x , color = 'r',label = 'Chemo_Short_epoch3')
        axs1[5,4].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_on2_opto_s) > 0:
        x = np.mean(p_epoch3_on2_opto_s, axis=0)
        y = stats.sem(p_epoch3_on2_opto_s, axis=0)
        axs1[5,4].plot(encoder_times - 1, x , color = 'deepskyblue', label = 'Opto_Short_epoch3')
        axs1[5,4].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)

    # epoch1 long
    if len(p_epoch1_vis1_con_l) > 0:
        x = np.mean(p_epoch1_vis1_con_l, axis=0)
        y = stats.sem(p_epoch1_vis1_con_l, axis=0)
        axs1[6,0].plot(encoder_times - 1, x , color = 'k')
        axs1[6,0].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_vis1_chemo_l) > 0:
        x = np.mean(p_epoch1_vis1_chemo_l, axis=0)
        y = stats.sem(p_epoch1_vis1_chemo_l, axis=0)
        axs1[6,0].plot(encoder_times - 1, x , color = 'r')
        axs1[6,0].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_vis1_opto_l) > 0:
        x = np.mean(p_epoch1_vis1_opto_l, axis=0)
        y = stats.sem(p_epoch1_vis1_opto_l, axis=0)
        axs1[6,0].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[6,0].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch1_vis2_con_l) > 0:
        x = np.mean(p_epoch1_vis2_con_l, axis=0)
        y = stats.sem(p_epoch1_vis2_con_l, axis=0)
        axs1[6,1].plot(encoder_times - 1, x , color = 'k')
        axs1[6,1].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_vis2_chemo_l) > 0:
        x = np.mean(p_epoch1_vis2_chemo_l, axis=0)
        y = stats.sem(p_epoch1_vis2_chemo_l, axis=0)
        axs1[6,1].plot(encoder_times - 1, x , color = 'r')
        axs1[6,1].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_vis2_opto_l) > 0:
        x = np.mean(p_epoch1_vis2_opto_l, axis=0)
        y = stats.sem(p_epoch1_vis2_opto_l, axis=0)
        axs1[6,1].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[6,1].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch1_rew_con_l) > 0:
        x = np.mean(p_epoch1_rew_con_l, axis=0)
        y = stats.sem(p_epoch1_rew_con_l, axis=0)
        axs1[6,2].plot(encoder_times - 1, x , color = 'k')
        axs1[6,2].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_rew_chemo_l) > 0:
        x = np.mean(p_epoch1_rew_chemo_l, axis=0)
        y = stats.sem(p_epoch1_rew_chemo_l, axis=0)
        axs1[6,2].plot(encoder_times - 1, x , color = 'r')
        axs1[6,2].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_rew_opto_l) > 0:
        x = np.mean(p_epoch1_rew_opto_l, axis=0)
        y = stats.sem(p_epoch1_rew_opto_l, axis=0)
        axs1[6,2].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[6,2].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch1_on1_con_l) > 0:
        x = np.mean(p_epoch1_on1_con_l, axis=0)
        y = stats.sem(p_epoch1_on1_con_l, axis=0)
        axs1[6,3].plot(encoder_times - 1, x , color = 'k')
        axs1[6,3].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_on1_chemo_l) > 0:
        x = np.mean(p_epoch1_on1_chemo_l, axis=0)
        y = stats.sem(p_epoch1_on1_chemo_l, axis=0)
        axs1[6,3].plot(encoder_times - 1, x , color = 'r')
        axs1[6,3].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_on1_opto_l) > 0:
        x = np.mean(p_epoch1_on1_opto_l, axis=0)
        y = stats.sem(p_epoch1_on1_opto_l, axis=0)
        axs1[6,3].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[6,3].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)

    if len(p_epoch1_on2_con_l) > 0:
        x = np.mean(p_epoch1_on2_con_l, axis=0)
        y = stats.sem(p_epoch1_on2_con_l, axis=0)
        axs1[6,4].plot(encoder_times - 1, x , color = 'k', label = 'Control_Long_epoch1')
        axs1[6,4].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch1_on2_chemo_l) > 0:
        x = np.mean(p_epoch1_on2_chemo_l, axis=0)
        y = stats.sem(p_epoch1_on2_chemo_l, axis=0)
        axs1[6,4].plot(encoder_times - 1, x , color = 'r',label = 'Chemo_Long_epoch1')
        axs1[6,4].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch1_on2_opto_l) > 0:
        x = np.mean(p_epoch1_on2_opto_l, axis=0)
        y = stats.sem(p_epoch1_on2_opto_l, axis=0)
        axs1[6,4].plot(encoder_times - 1, x , color = 'deepskyblue', label = 'Opto_Long_epoch1')
        axs1[6,4].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    # epoch2 long
    if len(p_epoch2_vis1_con_l) > 0:
        x = np.mean(p_epoch2_vis1_con_l, axis=0)
        y = stats.sem(p_epoch2_vis1_con_l, axis=0)
        axs1[7,0].plot(encoder_times - 1, x , color = 'k')
        axs1[7,0].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_vis1_chemo_l) > 0:
        x = np.mean(p_epoch2_vis1_chemo_l, axis=0)
        y = stats.sem(p_epoch2_vis1_chemo_l, axis=0)
        axs1[7,0].plot(encoder_times - 1, x , color = 'r')
        axs1[7,0].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_vis1_opto_l) > 0:
        x = np.mean(p_epoch2_vis1_opto_l, axis=0)
        y = stats.sem(p_epoch2_vis1_opto_l, axis=0)
        axs1[7,0].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[7,0].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch2_vis2_con_l) > 0:
        x = np.mean(p_epoch2_vis2_con_l, axis=0)
        y = stats.sem(p_epoch2_vis2_con_l, axis=0)
        axs1[7,1].plot(encoder_times - 1, x , color = 'k')
        axs1[7,1].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_vis2_chemo_l) > 0:
        x = np.mean(p_epoch2_vis2_chemo_l, axis=0)
        y = stats.sem(p_epoch2_vis2_chemo_l, axis=0)
        axs1[7,1].plot(encoder_times - 1, x , color = 'r')
        axs1[7,1].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_vis2_opto_l) > 0:
        x = np.mean(p_epoch2_vis2_opto_l, axis=0)
        y = stats.sem(p_epoch2_vis2_opto_l, axis=0)
        axs1[7,1].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[7,1].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch2_rew_con_l) > 0:
        x = np.mean(p_epoch2_rew_con_l, axis=0)
        y = stats.sem(p_epoch2_rew_con_l, axis=0)
        axs1[7,2].plot(encoder_times - 1, x , color = 'k')
        axs1[7,2].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_rew_chemo_l) > 0:
        x = np.mean(p_epoch2_rew_chemo_l, axis=0)
        y = stats.sem(p_epoch2_rew_chemo_l, axis=0)
        axs1[7,2].plot(encoder_times - 1, x , color = 'r')
        axs1[7,2].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_rew_opto_l) > 0:
        x = np.mean(p_epoch2_rew_opto_l, axis=0)
        y = stats.sem(p_epoch2_rew_opto_l, axis=0)
        axs1[7,2].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[7,2].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch2_on1_con_l) > 0:
        x = np.mean(p_epoch2_on1_con_l, axis=0)
        y = stats.sem(p_epoch2_on1_con_l, axis=0)
        axs1[7,3].plot(encoder_times - 1, x , color = 'k')
        axs1[7,3].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_on1_chemo_l) > 0:
        x = np.mean(p_epoch2_on1_chemo_l, axis=0)
        y = stats.sem(p_epoch2_on1_chemo_l, axis=0)
        axs1[7,3].plot(encoder_times - 1, x , color = 'r')
        axs1[7,3].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_on1_opto_l) > 0:
        x = np.mean(p_epoch2_on1_opto_l, axis=0)
        y = stats.sem(p_epoch2_on1_opto_l, axis=0)
        axs1[7,3].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[7,3].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)

    if len(p_epoch2_on2_con_l) > 0:
        x = np.mean(p_epoch2_on2_con_l, axis=0)
        y = stats.sem(p_epoch2_on2_con_l, axis=0)
        axs1[7,4].plot(encoder_times - 1, x , color = 'k', label = 'Control_Long_epoch2')
        axs1[7,4].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch2_on2_chemo_l) > 0:
        x = np.mean(p_epoch2_on2_chemo_l, axis=0)
        y = stats.sem(p_epoch2_on2_chemo_l, axis=0)
        axs1[7,4].plot(encoder_times - 1, x , color = 'r',label = 'Chemo_Long_epoch2')
        axs1[7,4].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch2_on2_opto_l) > 0:
        x = np.mean(p_epoch2_on2_opto_l, axis=0)
        y = stats.sem(p_epoch2_on2_opto_l, axis=0)
        axs1[7,4].plot(encoder_times - 1, x , color = 'deepskyblue', label = 'Opto_Long_epoch2')
        axs1[7,4].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    # epoch3 long
    if len(p_epoch3_vis1_con_l) > 0:
        x = np.mean(p_epoch3_vis1_con_l, axis=0)
        y = stats.sem(p_epoch3_vis1_con_l, axis=0)
        axs1[8,0].plot(encoder_times - 1, x , color = 'k')
        axs1[8,0].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_vis1_chemo_l) > 0:
        x = np.mean(p_epoch3_vis1_chemo_l, axis=0)
        y = stats.sem(p_epoch3_vis1_chemo_l, axis=0)
        axs1[8,0].plot(encoder_times - 1, x , color = 'r')
        axs1[8,0].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_vis1_opto_l) > 0:
        x = np.mean(p_epoch3_vis1_opto_l, axis=0)
        y = stats.sem(p_epoch3_vis1_opto_l, axis=0)
        axs1[8,0].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[8,0].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch3_vis2_con_l) > 0:
        x = np.mean(p_epoch3_vis2_con_l, axis=0)
        y = stats.sem(p_epoch3_vis2_con_l, axis=0)
        axs1[8,1].plot(encoder_times - 1, x , color = 'k')
        axs1[8,1].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_vis2_chemo_l) > 0:
        x = np.mean(p_epoch3_vis2_chemo_l, axis=0)
        y = stats.sem(p_epoch3_vis2_chemo_l, axis=0)
        axs1[8,1].plot(encoder_times - 1, x , color = 'r')
        axs1[8,1].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_vis2_opto_l) > 0:
        x = np.mean(p_epoch3_vis2_opto_l, axis=0)
        y = stats.sem(p_epoch3_vis2_opto_l, axis=0)
        axs1[8,1].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[8,1].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch3_rew_con_l) > 0:
        x = np.mean(p_epoch3_rew_con_l, axis=0)
        y = stats.sem(p_epoch3_rew_con_l, axis=0)
        axs1[8,2].plot(encoder_times - 1, x , color = 'k')
        axs1[8,2].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_rew_chemo_l) > 0:
        x = np.mean(p_epoch3_rew_chemo_l, axis=0)
        y = stats.sem(p_epoch3_rew_chemo_l, axis=0)
        axs1[8,2].plot(encoder_times - 1, x , color = 'r')
        axs1[8,2].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_rew_opto_l) > 0:
        x = np.mean(p_epoch3_rew_opto_l, axis=0)
        y = stats.sem(p_epoch3_rew_opto_l, axis=0)
        axs1[8,2].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[8,2].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)
        
    if len(p_epoch3_on1_con_l) > 0:
        x = np.mean(p_epoch3_on1_con_l, axis=0)
        y = stats.sem(p_epoch3_on1_con_l, axis=0)
        axs1[8,3].plot(encoder_times - 1, x , color = 'k')
        axs1[8,3].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_on1_chemo_l) > 0:
        x = np.mean(p_epoch3_on1_chemo_l, axis=0)
        y = stats.sem(p_epoch3_on1_chemo_l, axis=0)
        axs1[8,3].plot(encoder_times - 1, x , color = 'r')
        axs1[8,3].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_on1_opto_l) > 0:
        x = np.mean(p_epoch3_on1_opto_l, axis=0)
        y = stats.sem(p_epoch3_on1_opto_l, axis=0)
        axs1[8,3].plot(encoder_times - 1, x , color = 'deepskyblue')
        axs1[8,3].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)

    if len(p_epoch3_on2_con_l) > 0:
        x = np.mean(p_epoch3_on2_con_l, axis=0)
        y = stats.sem(p_epoch3_on2_con_l, axis=0)
        axs1[8,4].plot(encoder_times - 1, x , color = 'k', label = 'Control_Long_epoch3')
        axs1[8,4].fill_between(encoder_times - 1, x-y , x+y , color='k', alpha=0.3)
        
    if len(p_epoch3_on2_chemo_l) > 0:
        x = np.mean(p_epoch3_on2_chemo_l, axis=0)
        y = stats.sem(p_epoch3_on2_chemo_l, axis=0)
        axs1[8,4].plot(encoder_times - 1, x , color = 'r',label = 'Chemo_Long_epoch3')
        axs1[8,4].fill_between(encoder_times - 1, x-y , x+y , color='r', alpha=0.3)
        
    if len(p_epoch3_on2_opto_l) > 0:
        x = np.mean(p_epoch3_on2_opto_l, axis=0)
        y = stats.sem(p_epoch3_on2_opto_l, axis=0)
        axs1[8,4].plot(encoder_times - 1, x , color = 'deepskyblue', label = 'Opto_Long_epoch3')
        axs1[8,4].fill_between(encoder_times - 1, x-y , x+y , color='deepskyblue', alpha=0.3)


    ##############
    axs1[0,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs1[1,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs1[2,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs1[3,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs1[4,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)   
    axs1[5,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs1[6,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)   
    axs1[7,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)   
    axs1[8,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)  
    fig1.tight_layout() 


    ################################## POOLED STATE ROWS ######################################
    fig2, axs2 = plt.subplots(nrows=9, ncols=5, figsize=(27, 36))
    fig2.subplots_adjust(hspace=0.7)
    fig2.suptitle(subject + '\n Average Trajectories Pooled Trials (Rows Condition)\n')

    # Define colors
    black_shades = [(150, 150, 150), (100, 100, 100), (50, 50, 50)]
    red_shades = [(255, 102, 102), (255, 51, 51), (204, 0, 0)]
    skyblue_shades = [(135, 206, 235), (70, 130, 180), (0, 105, 148)]
    # Normalize the colors to [0, 1] range for matplotlib
    black_shades = [tuple(c/255 for c in shade) for shade in black_shades]
    red_shades = [tuple(c/255 for c in shade) for shade in red_shades]
    skyblue_shades = [tuple(c/255 for c in shade) for shade in skyblue_shades]

    # ALL TRIALS
    axs2[0,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[0,0].set_title('VisStim1 Aligned Control over Epochs.\n') 
    axs2[0,0].axvline(x = 0, color = 'r', linestyle='--')
    axs2[0,0].set_xlim((-1, 2))
    axs2[0,0].spines['right'].set_visible(False)
    axs2[0,0].spines['top'].set_visible(False)
    axs2[0,0].set_xlabel('Time from VisStim1 (s)')
    axs2[0,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[0,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[0,1].axvline(x = 0, color = 'r', linestyle='--')
    axs2[0,1].set_xlim((-1, 2))
    axs2[0,1].spines['right'].set_visible(False)
    axs2[0,1].spines['top'].set_visible(False)
    axs2[0,1].set_title('VisStim2 Aligned Control over Epochs.\n') 
    axs2[0,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs2[0,1].set_title('waitforpress2 Aligned Control over Epochs.\n') 
        axs2[0,1].set_xlabel('Time from waitforpress2 (s)')
    axs2[0,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[0,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[0,2].set_title('\n ALL TRIALS \n Reward Aligned Control over Epochs.\n') 
    axs2[0,2].axvline(x = 0, color = 'r', linestyle='--')
    axs2[0,2].set_xlim((-1, 2))
    axs2[0,2].spines['right'].set_visible(False)
    axs2[0,2].spines['top'].set_visible(False)
    axs2[0,2].set_xlabel('Time from Reward (s)')
    axs2[0,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[0,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[0,3].set_title('Onset1 Aligned Control over Epochs.\n') 
    axs2[0,3].axvline(x = 0, color = 'r', linestyle='--')
    axs2[0,3].set_xlim((-1, 2))
    axs2[0,3].spines['right'].set_visible(False)
    axs2[0,3].spines['top'].set_visible(False)
    axs2[0,3].set_xlabel('Time from Onset1 (s)')
    axs2[0,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[0,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[0,4].set_title('Onset2 Aligned Control over Epochs.\n') 
    axs2[0,4].axvline(x = 0, color = 'r', linestyle='--')
    axs2[0,4].set_xlim((-1, 2))
    axs2[0,4].spines['right'].set_visible(False)
    axs2[0,4].spines['top'].set_visible(False)
    axs2[0,4].set_xlabel('Time from Onset2 (s)')
    axs2[0,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[1,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[1,0].set_title('VisStim1 Aligned Chemo over Epochs..\n') 
    axs2[1,0].axvline(x = 0, color = 'r', linestyle='--')
    axs2[1,0].set_xlim((-1, 2))
    axs2[1,0].spines['right'].set_visible(False)
    axs2[1,0].spines['top'].set_visible(False)
    axs2[1,0].set_xlabel('Time from VisStim1 (s)')
    axs2[1,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[1,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[1,1].axvline(x = 0, color = 'r', linestyle='--')
    axs2[1,1].set_xlim((-1, 2))
    axs2[1,1].spines['right'].set_visible(False)
    axs2[1,1].spines['top'].set_visible(False)
    axs2[1,1].set_title('VisStim2 Aligned Chemo over Epochs..\n') 
    axs2[1,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs2[1,1].set_title('waitforpress2 Aligned Chemo over Epochs..\n') 
        axs2[1,1].set_xlabel('Time from waitforpress2 (s)')
    axs2[1,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[1,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[1,2].set_title('\n Reward Aligned Chemo over Epochs..\n') 
    axs2[1,2].axvline(x = 0, color = 'r', linestyle='--')
    axs2[1,2].set_xlim((-1, 2))
    axs2[1,2].spines['right'].set_visible(False)
    axs2[1,2].spines['top'].set_visible(False)
    axs2[1,2].set_xlabel('Time from Reward (s)')
    axs2[1,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[1,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[1,3].set_title('Onset1 Aligned Chemo over Epochs..\n') 
    axs2[1,3].axvline(x = 0, color = 'r', linestyle='--')
    axs2[1,3].set_xlim((-1, 2))
    axs2[1,3].spines['right'].set_visible(False)
    axs2[1,3].spines['top'].set_visible(False)
    axs2[1,3].set_xlabel('Time from Onset1 (s)')
    axs2[1,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[1,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[1,4].set_title('Onset2 Aligned Chemo over Epochs..\n') 
    axs2[1,4].axvline(x = 0, color = 'r', linestyle='--')
    axs2[1,4].set_xlim((-1, 2))
    axs2[1,4].spines['right'].set_visible(False)
    axs2[1,4].spines['top'].set_visible(False)
    axs2[1,4].set_xlabel('Time from Onset2 (s)')
    axs2[1,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[2,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[2,0].set_title('VisStim1 Aligned Opto over Epochs..\n') 
    axs2[2,0].axvline(x = 0, color = 'r', linestyle='--')
    axs2[2,0].set_xlim((-1, 2))
    axs2[2,0].spines['right'].set_visible(False)
    axs2[2,0].spines['top'].set_visible(False)
    axs2[2,0].set_xlabel('Time from VisStim1 (s)')
    axs2[2,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[2,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[2,1].axvline(x = 0, color = 'r', linestyle='--')
    axs2[2,1].set_xlim((-1, 2))
    axs2[2,1].spines['right'].set_visible(False)
    axs2[2,1].spines['top'].set_visible(False)
    axs2[2,1].set_title('VisStim2 Aligned Opto over Epochs..\n') 
    axs2[2,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs2[2,1].set_title('waitforpress2 Aligned Opto over Epochs..\n') 
        axs2[2,1].set_xlabel('Time from waitforpress2 (s)')
    axs2[2,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[2,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[2,2].set_title('\n Reward Aligned Opto over Epochs..\n') 
    axs2[2,2].axvline(x = 0, color = 'r', linestyle='--')
    axs2[2,2].set_xlim((-1, 2))
    axs2[2,2].spines['right'].set_visible(False)
    axs2[2,2].spines['top'].set_visible(False)
    axs2[2,2].set_xlabel('Time from Reward (s)')
    axs2[2,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[2,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[2,3].set_title('Onset1 Aligned Opto over Epochs..\n') 
    axs2[2,3].axvline(x = 0, color = 'r', linestyle='--')
    axs2[2,3].set_xlim((-1, 2))
    axs2[2,3].spines['right'].set_visible(False)
    axs2[2,3].spines['top'].set_visible(False)
    axs2[2,3].set_xlabel('Time from Onset1 (s)')
    axs2[2,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[2,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[2,4].set_title('Onset2 Aligned Opto over Epochs..\n') 
    axs2[2,4].axvline(x = 0, color = 'r', linestyle='--')
    axs2[2,4].set_xlim((-1, 2))
    axs2[2,4].spines['right'].set_visible(False)
    axs2[2,4].spines['top'].set_visible(False)
    axs2[2,4].set_xlabel('Time from Onset2 (s)')
    axs2[2,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    # SHORT TRIALS
    axs2[3,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[3,0].set_title('VisStim1 Aligned Control over Epochs.\n') 
    axs2[3,0].axvline(x = 0, color = 'r', linestyle='--')
    axs2[3,0].set_xlim((-1, 2))
    axs2[3,0].spines['right'].set_visible(False)
    axs2[3,0].spines['top'].set_visible(False)
    axs2[3,0].set_xlabel('Time from VisStim1 (s)')
    axs2[3,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[3,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[3,1].axvline(x = 0, color = 'r', linestyle='--')
    axs2[3,1].set_xlim((-1, 2))
    axs2[3,1].spines['right'].set_visible(False)
    axs2[3,1].spines['top'].set_visible(False)
    axs2[3,1].set_title('VisStim2 Aligned Control over Epochs.\n') 
    axs2[3,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs2[3,1].set_title('waitforpress2 Aligned Control over Epochs.\n') 
        axs2[3,1].set_xlabel('Time from waitforpress2 (s)')
    axs2[3,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[3,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[3,2].set_title('\n SHORT TRIALS \n Reward Aligned Control over Epochs.\n') 
    axs2[3,2].axvline(x = 0, color = 'r', linestyle='--')
    axs2[3,2].set_xlim((-1, 2))
    axs2[3,2].spines['right'].set_visible(False)
    axs2[3,2].spines['top'].set_visible(False)
    axs2[3,2].set_xlabel('Time from Reward (s)')
    axs2[3,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[3,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[3,3].set_title('Onset1 Aligned Control over Epochs.\n') 
    axs2[3,3].axvline(x = 0, color = 'r', linestyle='--')
    axs2[3,3].set_xlim((-1, 2))
    axs2[3,3].spines['right'].set_visible(False)
    axs2[3,3].spines['top'].set_visible(False)
    axs2[3,3].set_xlabel('Time from Onset1 (s)')
    axs2[3,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[3,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[3,4].set_title('Onset2 Aligned Control over Epochs.\n') 
    axs2[3,4].axvline(x = 0, color = 'r', linestyle='--')
    axs2[3,4].set_xlim((-1, 2))
    axs2[3,4].spines['right'].set_visible(False)
    axs2[3,4].spines['top'].set_visible(False)
    axs2[3,4].set_xlabel('Time from Onset2 (s)')
    axs2[3,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[4,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[4,0].set_title('VisStim1 Aligned Chemo over Epochs..\n') 
    axs2[4,0].axvline(x = 0, color = 'r', linestyle='--')
    axs2[4,0].set_xlim((-1, 2))
    axs2[4,0].spines['right'].set_visible(False)
    axs2[4,0].spines['top'].set_visible(False)
    axs2[4,0].set_xlabel('Time from VisStim1 (s)')
    axs2[4,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[4,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[4,1].axvline(x = 0, color = 'r', linestyle='--')
    axs2[4,1].set_xlim((-1, 2))
    axs2[4,1].spines['right'].set_visible(False)
    axs2[4,1].spines['top'].set_visible(False)
    axs2[4,1].set_title('VisStim2 Aligned Chemo over Epochs..\n') 
    axs2[4,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs2[4,1].set_title('waitforpress2 Aligned Chemo over Epochs..\n') 
        axs2[4,1].set_xlabel('Time from waitforpress2 (s)')
    axs2[4,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[4,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[4,2].set_title('\n Reward Aligned Chemo over Epochs..\n') 
    axs2[4,2].axvline(x = 0, color = 'r', linestyle='--')
    axs2[4,2].set_xlim((-1, 2))
    axs2[4,2].spines['right'].set_visible(False)
    axs2[4,2].spines['top'].set_visible(False)
    axs2[4,2].set_xlabel('Time from Reward (s)')
    axs2[4,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[4,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[4,3].set_title('Onset1 Aligned Chemo over Epochs..\n') 
    axs2[4,3].axvline(x = 0, color = 'r', linestyle='--')
    axs2[4,3].set_xlim((-1, 2))
    axs2[4,3].spines['right'].set_visible(False)
    axs2[4,3].spines['top'].set_visible(False)
    axs2[4,3].set_xlabel('Time from Onset1 (s)')
    axs2[4,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[4,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[4,4].set_title('Onset2 Aligned Chemo over Epochs..\n') 
    axs2[4,4].axvline(x = 0, color = 'r', linestyle='--')
    axs2[4,4].set_xlim((-1, 2))
    axs2[4,4].spines['right'].set_visible(False)
    axs2[4,4].spines['top'].set_visible(False)
    axs2[4,4].set_xlabel('Time from Onset2 (s)')
    axs2[4,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[5,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[5,0].set_title('VisStim1 Aligned Opto over Epochs..\n') 
    axs2[5,0].axvline(x = 0, color = 'r', linestyle='--')
    axs2[5,0].set_xlim((-1, 2))
    axs2[5,0].spines['right'].set_visible(False)
    axs2[5,0].spines['top'].set_visible(False)
    axs2[5,0].set_xlabel('Time from VisStim1 (s)')
    axs2[5,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[5,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[5,1].axvline(x = 0, color = 'r', linestyle='--')
    axs2[5,1].set_xlim((-1, 2))
    axs2[5,1].spines['right'].set_visible(False)
    axs2[5,1].spines['top'].set_visible(False)
    axs2[5,1].set_title('VisStim2 Aligned Opto over Epochs..\n') 
    axs2[5,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs2[5,1].set_title('waitforpress2 Aligned Opto over Epochs..\n') 
        axs2[5,1].set_xlabel('Time from waitforpress2 (s)')
    axs2[5,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[5,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[5,2].set_title('\n Reward Aligned Opto over Epochs..\n') 
    axs2[5,2].axvline(x = 0, color = 'r', linestyle='--')
    axs2[5,2].set_xlim((-1, 2))
    axs2[5,2].spines['right'].set_visible(False)
    axs2[5,2].spines['top'].set_visible(False)
    axs2[5,2].set_xlabel('Time from Reward (s)')
    axs2[5,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[5,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[5,3].set_title('Onset1 Aligned Opto over Epochs..\n') 
    axs2[5,3].axvline(x = 0, color = 'r', linestyle='--')
    axs2[5,3].set_xlim((-1, 2))
    axs2[5,3].spines['right'].set_visible(False)
    axs2[5,3].spines['top'].set_visible(False)
    axs2[5,3].set_xlabel('Time from Onset1 (s)')
    axs2[5,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[5,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[5,4].set_title('Onset2 Aligned Opto over Epochs..\n') 
    axs2[5,4].axvline(x = 0, color = 'r', linestyle='--')
    axs2[5,4].set_xlim((-1, 2))
    axs2[5,4].spines['right'].set_visible(False)
    axs2[5,4].spines['top'].set_visible(False)
    axs2[5,4].set_xlabel('Time from Onset2 (s)')
    axs2[5,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    # LONG TRIALS
    axs2[6,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[6,0].set_title('VisStim1 Aligned Control over Epochs.\n') 
    axs2[6,0].axvline(x = 0, color = 'r', linestyle='--')
    axs2[6,0].set_xlim((-1, 2))
    axs2[6,0].spines['right'].set_visible(False)
    axs2[6,0].spines['top'].set_visible(False)
    axs2[6,0].set_xlabel('Time from VisStim1 (s)')
    axs2[6,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[6,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[6,1].axvline(x = 0, color = 'r', linestyle='--')
    axs2[6,1].set_xlim((-1, 2))
    axs2[6,1].spines['right'].set_visible(False)
    axs2[6,1].spines['top'].set_visible(False)
    axs2[6,1].set_title('VisStim2 Aligned Control over Epochs.\n') 
    axs2[6,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs2[6,1].set_title('waitforpress2 Aligned Control over Epochs.\n') 
        axs2[6,1].set_xlabel('Time from waitforpress2 (s)')
    axs2[6,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[6,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[6,2].set_title('\n LONG TRIALS \n Reward Aligned Control over Epochs.\n') 
    axs2[6,2].axvline(x = 0, color = 'r', linestyle='--')
    axs2[6,2].set_xlim((-1, 2))
    axs2[6,2].spines['right'].set_visible(False)
    axs2[6,2].spines['top'].set_visible(False)
    axs2[6,2].set_xlabel('Time from Reward (s)')
    axs2[6,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[6,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[6,3].set_title('Onset1 Aligned Control over Epochs.\n') 
    axs2[6,3].axvline(x = 0, color = 'r', linestyle='--')
    axs2[6,3].set_xlim((-1, 2))
    axs2[6,3].spines['right'].set_visible(False)
    axs2[6,3].spines['top'].set_visible(False)
    axs2[6,3].set_xlabel('Time from Onset1 (s)')
    axs2[6,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[6,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[6,4].set_title('Onset2 Aligned Control over Epochs.\n') 
    axs2[6,4].axvline(x = 0, color = 'r', linestyle='--')
    axs2[6,4].set_xlim((-1, 2))
    axs2[6,4].spines['right'].set_visible(False)
    axs2[6,4].spines['top'].set_visible(False)
    axs2[6,4].set_xlabel('Time from Onset2 (s)')
    axs2[6,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[7,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[7,0].set_title('VisStim1 Aligned Chemo over Epochs..\n') 
    axs2[7,0].axvline(x = 0, color = 'r', linestyle='--')
    axs2[7,0].set_xlim((-1, 2))
    axs2[7,0].spines['right'].set_visible(False)
    axs2[7,0].spines['top'].set_visible(False)
    axs2[7,0].set_xlabel('Time from VisStim1 (s)')
    axs2[7,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[7,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[7,1].axvline(x = 0, color = 'r', linestyle='--')
    axs2[7,1].set_xlim((-1, 2))
    axs2[7,1].spines['right'].set_visible(False)
    axs2[7,1].spines['top'].set_visible(False)
    axs2[7,1].set_title('VisStim2 Aligned Chemo over Epochs..\n') 
    axs2[7,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs2[7,1].set_title('waitforpress2 Aligned Chemo over Epochs..\n') 
        axs2[7,1].set_xlabel('Time from waitforpress2 (s)')
    axs2[7,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[7,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[7,2].set_title('\n Reward Aligned Chemo over Epochs..\n') 
    axs2[7,2].axvline(x = 0, color = 'r', linestyle='--')
    axs2[7,2].set_xlim((-1, 2))
    axs2[7,2].spines['right'].set_visible(False)
    axs2[7,2].spines['top'].set_visible(False)
    axs2[7,2].set_xlabel('Time from Reward (s)')
    axs2[7,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[7,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[7,3].set_title('Onset1 Aligned Chemo over Epochs..\n') 
    axs2[7,3].axvline(x = 0, color = 'r', linestyle='--')
    axs2[7,3].set_xlim((-1, 2))
    axs2[7,3].spines['right'].set_visible(False)
    axs2[7,3].spines['top'].set_visible(False)
    axs2[7,3].set_xlabel('Time from Onset1 (s)')
    axs2[7,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[7,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[7,4].set_title('Onset2 Aligned Chemo over Epochs..\n') 
    axs2[7,4].axvline(x = 0, color = 'r', linestyle='--')
    axs2[7,4].set_xlim((-1, 2))
    axs2[7,4].spines['right'].set_visible(False)
    axs2[7,4].spines['top'].set_visible(False)
    axs2[7,4].set_xlabel('Time from Onset2 (s)')
    axs2[7,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[8,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[8,0].set_title('VisStim1 Aligned Opto over Epochs..\n') 
    axs2[8,0].axvline(x = 0, color = 'r', linestyle='--')
    axs2[8,0].set_xlim((-1, 2))
    axs2[8,0].spines['right'].set_visible(False)
    axs2[8,0].spines['top'].set_visible(False)
    axs2[8,0].set_xlabel('Time from VisStim1 (s)')
    axs2[8,0].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[8,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[8,1].axvline(x = 0, color = 'r', linestyle='--')
    axs2[8,1].set_xlim((-1, 2))
    axs2[8,1].spines['right'].set_visible(False)
    axs2[8,1].spines['top'].set_visible(False)
    axs2[8,1].set_title('VisStim2 Aligned Opto over Epochs..\n') 
    axs2[8,1].set_xlabel('Time from VisStim2 (s)')
    if not(any(0 in row for row in isSelfTimedMode)):
        axs2[8,1].set_title('waitforpress2 Aligned Opto over Epochs..\n') 
        axs2[8,1].set_xlabel('Time from waitforpress2 (s)')
    axs2[8,1].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[8,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[8,2].set_title('\n Reward Aligned Opto over Epochs..\n') 
    axs2[8,2].axvline(x = 0, color = 'r', linestyle='--')
    axs2[8,2].set_xlim((-1, 2))
    axs2[8,2].spines['right'].set_visible(False)
    axs2[8,2].spines['top'].set_visible(False)
    axs2[8,2].set_xlabel('Time from Reward (s)')
    axs2[8,2].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[8,3].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[8,3].set_title('Onset1 Aligned Opto over Epochs..\n') 
    axs2[8,3].axvline(x = 0, color = 'r', linestyle='--')
    axs2[8,3].set_xlim((-1, 2))
    axs2[8,3].spines['right'].set_visible(False)
    axs2[8,3].spines['top'].set_visible(False)
    axs2[8,3].set_xlabel('Time from Onset1 (s)')
    axs2[8,3].set_ylabel('Joystick deflection (deg) Rewarded trials')

    axs2[8,4].axhline(y = target_thresh, color = '0.6', linestyle='--')
    axs2[8,4].set_title('Onset2 Aligned Opto over Epochs..\n') 
    axs2[8,4].axvline(x = 0, color = 'r', linestyle='--')
    axs2[8,4].set_xlim((-1, 2))
    axs2[8,4].spines['right'].set_visible(False)
    axs2[8,4].spines['top'].set_visible(False)
    axs2[8,4].set_xlabel('Time from Onset2 (s)')
    axs2[8,4].set_ylabel('Joystick deflection (deg) Rewarded trials')

    # Control ALL
    if len(p_epoch1_vis1_con_all) > 0:
        x = np.mean(p_epoch1_vis1_con_all, axis=0)
        y = stats.sem(p_epoch1_vis1_con_all, axis=0)
        axs2[0,0].plot(encoder_times - 1, x , color = black_shades[0])
        axs2[0,0].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis1_con_all) > 0:
        x = np.mean(p_epoch2_vis1_con_all, axis=0)
        y = stats.sem(p_epoch2_vis1_con_all, axis=0)
        axs2[0,0].plot(encoder_times - 1, x , color = black_shades[1])
        axs2[0,0].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis1_con_all) > 0:
        x = np.mean(p_epoch3_vis1_con_all, axis=0)
        y = stats.sem(p_epoch3_vis1_con_all, axis=0)
        axs2[0,0].plot(encoder_times - 1, x , color = black_shades[2])
        axs2[0,0].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)
        
    if len(p_epoch1_vis2_con_all) > 0:
        x = np.mean(p_epoch1_vis2_con_all, axis=0)
        y = stats.sem(p_epoch1_vis2_con_all, axis=0)
        axs2[0,1].plot(encoder_times - 1, x , color = black_shades[0])
        axs2[0,1].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis2_con_all) > 0:
        x = np.mean(p_epoch2_vis2_con_all, axis=0)
        y = stats.sem(p_epoch2_vis2_con_all, axis=0)
        axs2[0,1].plot(encoder_times - 1, x , color = black_shades[1])
        axs2[0,1].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis2_con_all) > 0:
        x = np.mean(p_epoch3_vis2_con_all, axis=0)
        y = stats.sem(p_epoch3_vis2_con_all, axis=0)
        axs2[0,1].plot(encoder_times - 1, x , color = black_shades[2])
        axs2[0,1].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)
        
    if len(p_epoch1_rew_con_all) > 0:
        x = np.mean(p_epoch1_rew_con_all, axis=0)
        y = stats.sem(p_epoch1_rew_con_all, axis=0)
        axs2[0,2].plot(encoder_times - 1, x , color = black_shades[0])
        axs2[0,2].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_rew_con_all) > 0:
        x = np.mean(p_epoch2_rew_con_all, axis=0)
        y = stats.sem(p_epoch2_rew_con_all, axis=0)
        axs2[0,2].plot(encoder_times - 1, x , color = black_shades[1])
        axs2[0,2].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_rew_con_all) > 0:
        x = np.mean(p_epoch3_rew_con_all, axis=0)
        y = stats.sem(p_epoch3_rew_con_all, axis=0)
        axs2[0,2].plot(encoder_times - 1, x , color = black_shades[2])
        axs2[0,2].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)
        
    if len(p_epoch1_on1_con_all) > 0:
        x = np.mean(p_epoch1_on1_con_all, axis=0)
        y = stats.sem(p_epoch1_on1_con_all, axis=0)
        axs2[0,3].plot(encoder_times - 1, x , color = black_shades[0])
        axs2[0,3].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_on1_con_all) > 0:
        x = np.mean(p_epoch2_on1_con_all, axis=0)
        y = stats.sem(p_epoch2_on1_con_all, axis=0)
        axs2[0,3].plot(encoder_times - 1, x , color = black_shades[1])
        axs2[0,3].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_on1_con_all) > 0:
        x = np.mean(p_epoch3_on1_con_all, axis=0)
        y = stats.sem(p_epoch3_on1_con_all, axis=0)
        axs2[0,3].plot(encoder_times - 1, x , color = black_shades[2])
        axs2[0,3].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)

    if len(p_epoch1_on2_con_all) > 0:
        x = np.mean(p_epoch1_on2_con_all, axis=0)
        y = stats.sem(p_epoch1_on2_con_all, axis=0)
        axs2[0,4].plot(encoder_times - 1, x , color = black_shades[0], label = 'Control_All_Epoch1')
        axs2[0,4].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_on2_con_all) > 0:
        x = np.mean(p_epoch2_on2_con_all, axis=0)
        y = stats.sem(p_epoch2_on2_con_all, axis=0)
        axs2[0,4].plot(encoder_times - 1, x , color = black_shades[1],label = 'Control_All_Epoch2')
        axs2[0,4].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_on2_con_all) > 0:
        x = np.mean(p_epoch3_on2_con_all, axis=0)
        y = stats.sem(p_epoch3_on2_con_all, axis=0)
        axs2[0,4].plot(encoder_times - 1, x , color = black_shades[2], label = 'Control_All_Epoch3')
        axs2[0,4].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)
    # Chemo ALL
    if len(p_epoch1_vis1_chemo_all) > 0:
        x = np.mean(p_epoch1_vis1_chemo_all, axis=0)
        y = stats.sem(p_epoch1_vis1_chemo_all, axis=0)
        axs2[1,0].plot(encoder_times - 1, x , color = red_shades[0])
        axs2[1,0].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis1_chemo_all) > 0:
        x = np.mean(p_epoch2_vis1_chemo_all, axis=0)
        y = stats.sem(p_epoch2_vis1_chemo_all, axis=0)
        axs2[1,0].plot(encoder_times - 1, x , color = red_shades[1])
        axs2[1,0].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis1_chemo_all) > 0:
        x = np.mean(p_epoch3_vis1_chemo_all, axis=0)
        y = stats.sem(p_epoch3_vis1_chemo_all, axis=0)
        axs2[1,0].plot(encoder_times - 1, x , color = red_shades[2])
        axs2[1,0].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)
        
    if len(p_epoch1_vis2_chemo_all) > 0:
        x = np.mean(p_epoch1_vis2_chemo_all, axis=0)
        y = stats.sem(p_epoch1_vis2_chemo_all, axis=0)
        axs2[1,1].plot(encoder_times - 1, x , color = red_shades[0])
        axs2[1,1].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis2_chemo_all) > 0:
        x = np.mean(p_epoch2_vis2_chemo_all, axis=0)
        y = stats.sem(p_epoch2_vis2_chemo_all, axis=0)
        axs2[1,1].plot(encoder_times - 1, x , color = red_shades[1])
        axs2[1,1].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis2_chemo_all) > 0:
        x = np.mean(p_epoch3_vis2_chemo_all, axis=0)
        y = stats.sem(p_epoch3_vis2_chemo_all, axis=0)
        axs2[1,1].plot(encoder_times - 1, x , color = red_shades[2])
        axs2[1,1].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)
        
    if len(p_epoch1_rew_chemo_all) > 0:
        x = np.mean(p_epoch1_rew_chemo_all, axis=0)
        y = stats.sem(p_epoch1_rew_chemo_all, axis=0)
        axs2[1,2].plot(encoder_times - 1, x , color = red_shades[0])
        axs2[1,2].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_rew_chemo_all) > 0:
        x = np.mean(p_epoch2_rew_chemo_all, axis=0)
        y = stats.sem(p_epoch2_rew_chemo_all, axis=0)
        axs2[1,2].plot(encoder_times - 1, x , color = red_shades[1])
        axs2[1,2].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_rew_chemo_all) > 0:
        x = np.mean(p_epoch3_rew_chemo_all, axis=0)
        y = stats.sem(p_epoch3_rew_chemo_all, axis=0)
        axs2[1,2].plot(encoder_times - 1, x , color = red_shades[2])
        axs2[1,2].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)
        
    if len(p_epoch1_on1_chemo_all) > 0:
        x = np.mean(p_epoch1_on1_chemo_all, axis=0)
        y = stats.sem(p_epoch1_on1_chemo_all, axis=0)
        axs2[1,3].plot(encoder_times - 1, x , color = red_shades[0])
        axs2[1,3].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_on1_chemo_all) > 0:
        x = np.mean(p_epoch2_on1_chemo_all, axis=0)
        y = stats.sem(p_epoch2_on1_chemo_all, axis=0)
        axs2[1,3].plot(encoder_times - 1, x , color = red_shades[1])
        axs2[1,3].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_on1_chemo_all) > 0:
        x = np.mean(p_epoch3_on1_chemo_all, axis=0)
        y = stats.sem(p_epoch3_on1_chemo_all, axis=0)
        axs2[1,3].plot(encoder_times - 1, x , color = red_shades[2])
        axs2[1,3].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)

    if len(p_epoch1_on2_chemo_all) > 0:
        x = np.mean(p_epoch1_on2_chemo_all, axis=0)
        y = stats.sem(p_epoch1_on2_chemo_all, axis=0)
        axs2[1,4].plot(encoder_times - 1, x , color = red_shades[0], label = 'Chemo_All_Epoch1')
        axs2[1,4].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_on2_chemo_all) > 0:
        x = np.mean(p_epoch2_on2_chemo_all, axis=0)
        y = stats.sem(p_epoch2_on2_chemo_all, axis=0)
        axs2[1,4].plot(encoder_times - 1, x , color = red_shades[1],label = 'Chemo_All_Epoch2')
        axs2[1,4].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_on2_chemo_all) > 0:
        x = np.mean(p_epoch3_on2_chemo_all, axis=0)
        y = stats.sem(p_epoch3_on2_chemo_all, axis=0)
        axs2[1,4].plot(encoder_times - 1, x , color = red_shades[2], label = 'Chemo_All_Epoch3')
        axs2[1,4].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)
    # Opto ALL
    if len(p_epoch1_vis1_opto_all) > 0:
        x = np.mean(p_epoch1_vis1_opto_all, axis=0)
        y = stats.sem(p_epoch1_vis1_opto_all, axis=0)
        axs2[2,0].plot(encoder_times - 1, x , color = skyblue_shades[0])
        axs2[2,0].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis1_opto_all) > 0:
        x = np.mean(p_epoch2_vis1_opto_all, axis=0)
        y = stats.sem(p_epoch2_vis1_opto_all, axis=0)
        axs2[2,0].plot(encoder_times - 1, x , color = skyblue_shades[1])
        axs2[2,0].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis1_opto_all) > 0:
        x = np.mean(p_epoch3_vis1_opto_all, axis=0)
        y = stats.sem(p_epoch3_vis1_opto_all, axis=0)
        axs2[2,0].plot(encoder_times - 1, x , color = skyblue_shades[2])
        axs2[2,0].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)
        
    if len(p_epoch1_vis2_opto_all) > 0:
        x = np.mean(p_epoch1_vis2_opto_all, axis=0)
        y = stats.sem(p_epoch1_vis2_opto_all, axis=0)
        axs2[2,1].plot(encoder_times - 1, x , color = skyblue_shades[0])
        axs2[2,1].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis2_opto_all) > 0:
        x = np.mean(p_epoch2_vis2_opto_all, axis=0)
        y = stats.sem(p_epoch2_vis2_opto_all, axis=0)
        axs2[2,1].plot(encoder_times - 1, x , color = skyblue_shades[1])
        axs2[2,1].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis2_opto_all) > 0:
        x = np.mean(p_epoch3_vis2_opto_all, axis=0)
        y = stats.sem(p_epoch3_vis2_opto_all, axis=0)
        axs2[2,1].plot(encoder_times - 1, x , color = skyblue_shades[2])
        axs2[2,1].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)
        
    if len(p_epoch1_rew_opto_all) > 0:
        x = np.mean(p_epoch1_rew_opto_all, axis=0)
        y = stats.sem(p_epoch1_rew_opto_all, axis=0)
        axs2[2,2].plot(encoder_times - 1, x , color = skyblue_shades[0])
        axs2[2,2].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_rew_opto_all) > 0:
        x = np.mean(p_epoch2_rew_opto_all, axis=0)
        y = stats.sem(p_epoch2_rew_opto_all, axis=0)
        axs2[2,2].plot(encoder_times - 1, x , color = skyblue_shades[1])
        axs2[2,2].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_rew_opto_all) > 0:
        x = np.mean(p_epoch3_rew_opto_all, axis=0)
        y = stats.sem(p_epoch3_rew_opto_all, axis=0)
        axs2[2,2].plot(encoder_times - 1, x , color = skyblue_shades[2])
        axs2[2,2].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)
        
    if len(p_epoch1_on1_opto_all) > 0:
        x = np.mean(p_epoch1_on1_opto_all, axis=0)
        y = stats.sem(p_epoch1_on1_opto_all, axis=0)
        axs2[2,3].plot(encoder_times - 1, x , color = skyblue_shades[0])
        axs2[2,3].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_on1_opto_all) > 0:
        x = np.mean(p_epoch2_on1_opto_all, axis=0)
        y = stats.sem(p_epoch2_on1_opto_all, axis=0)
        axs2[2,3].plot(encoder_times - 1, x , color = skyblue_shades[1])
        axs2[2,3].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_on1_opto_all) > 0:
        x = np.mean(p_epoch3_on1_opto_all, axis=0)
        y = stats.sem(p_epoch3_on1_opto_all, axis=0)
        axs2[2,3].plot(encoder_times - 1, x , color = skyblue_shades[2])
        axs2[2,3].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)

    if len(p_epoch1_on2_opto_all) > 0:
        x = np.mean(p_epoch1_on2_opto_all, axis=0)
        y = stats.sem(p_epoch1_on2_opto_all, axis=0)
        axs2[2,4].plot(encoder_times - 1, x , color = skyblue_shades[0], label = 'Opto_All_Epoch1')
        axs2[2,4].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_on2_opto_all) > 0:
        x = np.mean(p_epoch2_on2_opto_all, axis=0)
        y = stats.sem(p_epoch2_on2_opto_all, axis=0)
        axs2[2,4].plot(encoder_times - 1, x , color = skyblue_shades[1],label = 'Opto_All_Epoch2')
        axs2[2,4].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_on2_opto_all) > 0:
        x = np.mean(p_epoch3_on2_opto_all, axis=0)
        y = stats.sem(p_epoch3_on2_opto_all, axis=0)
        axs2[2,4].plot(encoder_times - 1, x , color = skyblue_shades[2], label = 'Opto_All_Epoch3')
        axs2[2,4].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)

    # Control ALL
    if len(p_epoch1_vis1_con_s) > 0:
        x = np.mean(p_epoch1_vis1_con_s, axis=0)
        y = stats.sem(p_epoch1_vis1_con_s, axis=0)
        axs2[3,0].plot(encoder_times - 1, x , color = black_shades[0])
        axs2[3,0].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis1_con_s) > 0:
        x = np.mean(p_epoch2_vis1_con_s, axis=0)
        y = stats.sem(p_epoch2_vis1_con_s, axis=0)
        axs2[3,0].plot(encoder_times - 1, x , color = black_shades[1])
        axs2[3,0].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis1_con_s) > 0:
        x = np.mean(p_epoch3_vis1_con_s, axis=0)
        y = stats.sem(p_epoch3_vis1_con_s, axis=0)
        axs2[3,0].plot(encoder_times - 1, x , color = black_shades[2])
        axs2[3,0].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)
        
    if len(p_epoch1_vis2_con_s) > 0:
        x = np.mean(p_epoch1_vis2_con_s, axis=0)
        y = stats.sem(p_epoch1_vis2_con_s, axis=0)
        axs2[3,1].plot(encoder_times - 1, x , color = black_shades[0])
        axs2[3,1].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis2_con_s) > 0:
        x = np.mean(p_epoch2_vis2_con_s, axis=0)
        y = stats.sem(p_epoch2_vis2_con_s, axis=0)
        axs2[3,1].plot(encoder_times - 1, x , color = black_shades[1])
        axs2[3,1].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis2_con_s) > 0:
        x = np.mean(p_epoch3_vis2_con_s, axis=0)
        y = stats.sem(p_epoch3_vis2_con_s, axis=0)
        axs2[3,1].plot(encoder_times - 1, x , color = black_shades[2])
        axs2[3,1].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)
        
    if len(p_epoch1_rew_con_s) > 0:
        x = np.mean(p_epoch1_rew_con_s, axis=0)
        y = stats.sem(p_epoch1_rew_con_s, axis=0)
        axs2[3,2].plot(encoder_times - 1, x , color = black_shades[0])
        axs2[3,2].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_rew_con_s) > 0:
        x = np.mean(p_epoch2_rew_con_s, axis=0)
        y = stats.sem(p_epoch2_rew_con_s, axis=0)
        axs2[3,2].plot(encoder_times - 1, x , color = black_shades[1])
        axs2[3,2].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_rew_con_s) > 0:
        x = np.mean(p_epoch3_rew_con_s, axis=0)
        y = stats.sem(p_epoch3_rew_con_s, axis=0)
        axs2[3,2].plot(encoder_times - 1, x , color = black_shades[2])
        axs2[3,2].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)
        
    if len(p_epoch1_on1_con_s) > 0:
        x = np.mean(p_epoch1_on1_con_s, axis=0)
        y = stats.sem(p_epoch1_on1_con_s, axis=0)
        axs2[3,3].plot(encoder_times - 1, x , color = black_shades[0])
        axs2[3,3].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_on1_con_s) > 0:
        x = np.mean(p_epoch2_on1_con_s, axis=0)
        y = stats.sem(p_epoch2_on1_con_s, axis=0)
        axs2[3,3].plot(encoder_times - 1, x , color = black_shades[1])
        axs2[3,3].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_on1_con_s) > 0:
        x = np.mean(p_epoch3_on1_con_s, axis=0)
        y = stats.sem(p_epoch3_on1_con_s, axis=0)
        axs2[3,3].plot(encoder_times - 1, x , color = black_shades[2])
        axs2[3,3].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)

    if len(p_epoch1_on2_con_s) > 0:
        x = np.mean(p_epoch1_on2_con_s, axis=0)
        y = stats.sem(p_epoch1_on2_con_s, axis=0)
        axs2[3,4].plot(encoder_times - 1, x , color = black_shades[0], label = 'Control_Short_Epoch1')
        axs2[3,4].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_on2_con_s) > 0:
        x = np.mean(p_epoch2_on2_con_s, axis=0)
        y = stats.sem(p_epoch2_on2_con_s, axis=0)
        axs2[3,4].plot(encoder_times - 1, x , color = black_shades[1],label = 'Control_Short_Epoch2')
        axs2[3,4].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_on2_con_s) > 0:
        x = np.mean(p_epoch3_on2_con_s, axis=0)
        y = stats.sem(p_epoch3_on2_con_s, axis=0)
        axs2[3,4].plot(encoder_times - 1, x , color = black_shades[2], label = 'Control_Short_Epoch3')
        axs2[3,4].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)
    # Chemo ALL
    if len(p_epoch1_vis1_chemo_s) > 0:
        x = np.mean(p_epoch1_vis1_chemo_s, axis=0)
        y = stats.sem(p_epoch1_vis1_chemo_s, axis=0)
        axs2[4,0].plot(encoder_times - 1, x , color = red_shades[0])
        axs2[4,0].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis1_chemo_s) > 0:
        x = np.mean(p_epoch2_vis1_chemo_s, axis=0)
        y = stats.sem(p_epoch2_vis1_chemo_s, axis=0)
        axs2[4,0].plot(encoder_times - 1, x , color = red_shades[1])
        axs2[4,0].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis1_chemo_s) > 0:
        x = np.mean(p_epoch3_vis1_chemo_s, axis=0)
        y = stats.sem(p_epoch3_vis1_chemo_s, axis=0)
        axs2[4,0].plot(encoder_times - 1, x , color = red_shades[2])
        axs2[4,0].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)
        
    if len(p_epoch1_vis2_chemo_s) > 0:
        x = np.mean(p_epoch1_vis2_chemo_s, axis=0)
        y = stats.sem(p_epoch1_vis2_chemo_s, axis=0)
        axs2[4,1].plot(encoder_times - 1, x , color = red_shades[0])
        axs2[4,1].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis2_chemo_s) > 0:
        x = np.mean(p_epoch2_vis2_chemo_s, axis=0)
        y = stats.sem(p_epoch2_vis2_chemo_s, axis=0)
        axs2[4,1].plot(encoder_times - 1, x , color = red_shades[1])
        axs2[4,1].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis2_chemo_s) > 0:
        x = np.mean(p_epoch3_vis2_chemo_s, axis=0)
        y = stats.sem(p_epoch3_vis2_chemo_s, axis=0)
        axs2[4,1].plot(encoder_times - 1, x , color = red_shades[2])
        axs2[4,1].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)
        
    if len(p_epoch1_rew_chemo_s) > 0:
        x = np.mean(p_epoch1_rew_chemo_s, axis=0)
        y = stats.sem(p_epoch1_rew_chemo_s, axis=0)
        axs2[4,2].plot(encoder_times - 1, x , color = red_shades[0])
        axs2[4,2].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_rew_chemo_s) > 0:
        x = np.mean(p_epoch2_rew_chemo_s, axis=0)
        y = stats.sem(p_epoch2_rew_chemo_s, axis=0)
        axs2[4,2].plot(encoder_times - 1, x , color = red_shades[1])
        axs2[4,2].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_rew_chemo_s) > 0:
        x = np.mean(p_epoch3_rew_chemo_s, axis=0)
        y = stats.sem(p_epoch3_rew_chemo_s, axis=0)
        axs2[4,2].plot(encoder_times - 1, x , color = red_shades[2])
        axs2[4,2].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)
        
    if len(p_epoch1_on1_chemo_s) > 0:
        x = np.mean(p_epoch1_on1_chemo_s, axis=0)
        y = stats.sem(p_epoch1_on1_chemo_s, axis=0)
        axs2[4,3].plot(encoder_times - 1, x , color = red_shades[0])
        axs2[4,3].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_on1_chemo_s) > 0:
        x = np.mean(p_epoch2_on1_chemo_s, axis=0)
        y = stats.sem(p_epoch2_on1_chemo_s, axis=0)
        axs2[4,3].plot(encoder_times - 1, x , color = red_shades[1])
        axs2[4,3].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_on1_chemo_s) > 0:
        x = np.mean(p_epoch3_on1_chemo_s, axis=0)
        y = stats.sem(p_epoch3_on1_chemo_s, axis=0)
        axs2[4,3].plot(encoder_times - 1, x , color = red_shades[2])
        axs2[4,3].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)

    if len(p_epoch1_on2_chemo_s) > 0:
        x = np.mean(p_epoch1_on2_chemo_s, axis=0)
        y = stats.sem(p_epoch1_on2_chemo_s, axis=0)
        axs2[4,4].plot(encoder_times - 1, x , color = red_shades[0], label = 'Chemo_Short_Epoch1')
        axs2[4,4].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_on2_chemo_s) > 0:
        x = np.mean(p_epoch2_on2_chemo_s, axis=0)
        y = stats.sem(p_epoch2_on2_chemo_s, axis=0)
        axs2[4,4].plot(encoder_times - 1, x , color = red_shades[1],label = 'Chemo_Short_Epoch2')
        axs2[4,4].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_on2_chemo_s) > 0:
        x = np.mean(p_epoch3_on2_chemo_s, axis=0)
        y = stats.sem(p_epoch3_on2_chemo_s, axis=0)
        axs2[4,4].plot(encoder_times - 1, x , color = red_shades[2], label = 'Chemo_Short_Epoch3')
        axs2[4,4].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)
    # Opto Short
    if len(p_epoch1_vis1_opto_s) > 0:
        x = np.mean(p_epoch1_vis1_opto_s, axis=0)
        y = stats.sem(p_epoch1_vis1_opto_s, axis=0)
        axs2[5,0].plot(encoder_times - 1, x , color = skyblue_shades[0])
        axs2[5,0].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis1_opto_s) > 0:
        x = np.mean(p_epoch2_vis1_opto_s, axis=0)
        y = stats.sem(p_epoch2_vis1_opto_s, axis=0)
        axs2[5,0].plot(encoder_times - 1, x , color = skyblue_shades[1])
        axs2[5,0].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis1_opto_s) > 0:
        x = np.mean(p_epoch3_vis1_opto_s, axis=0)
        y = stats.sem(p_epoch3_vis1_opto_s, axis=0)
        axs2[5,0].plot(encoder_times - 1, x , color = skyblue_shades[2])
        axs2[5,0].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)
        
    if len(p_epoch1_vis2_opto_s) > 0:
        x = np.mean(p_epoch1_vis2_opto_s, axis=0)
        y = stats.sem(p_epoch1_vis2_opto_s, axis=0)
        axs2[5,1].plot(encoder_times - 1, x , color = skyblue_shades[0])
        axs2[5,1].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis2_opto_s) > 0:
        x = np.mean(p_epoch2_vis2_opto_s, axis=0)
        y = stats.sem(p_epoch2_vis2_opto_s, axis=0)
        axs2[5,1].plot(encoder_times - 1, x , color = skyblue_shades[1])
        axs2[5,1].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis2_opto_s) > 0:
        x = np.mean(p_epoch3_vis2_opto_s, axis=0)
        y = stats.sem(p_epoch3_vis2_opto_s, axis=0)
        axs2[5,1].plot(encoder_times - 1, x , color = skyblue_shades[2])
        axs2[5,1].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)
        
    if len(p_epoch1_rew_opto_s) > 0:
        x = np.mean(p_epoch1_rew_opto_s, axis=0)
        y = stats.sem(p_epoch1_rew_opto_s, axis=0)
        axs2[5,2].plot(encoder_times - 1, x , color = skyblue_shades[0])
        axs2[5,2].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_rew_opto_s) > 0:
        x = np.mean(p_epoch2_rew_opto_s, axis=0)
        y = stats.sem(p_epoch2_rew_opto_s, axis=0)
        axs2[5,2].plot(encoder_times - 1, x , color = skyblue_shades[1])
        axs2[5,2].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_rew_opto_s) > 0:
        x = np.mean(p_epoch3_rew_opto_s, axis=0)
        y = stats.sem(p_epoch3_rew_opto_s, axis=0)
        axs2[5,2].plot(encoder_times - 1, x , color = skyblue_shades[2])
        axs2[5,2].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)
        
    if len(p_epoch1_on1_opto_s) > 0:
        x = np.mean(p_epoch1_on1_opto_s, axis=0)
        y = stats.sem(p_epoch1_on1_opto_s, axis=0)
        axs2[5,3].plot(encoder_times - 1, x , color = skyblue_shades[0])
        axs2[5,3].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_on1_opto_s) > 0:
        x = np.mean(p_epoch2_on1_opto_s, axis=0)
        y = stats.sem(p_epoch2_on1_opto_s, axis=0)
        axs2[5,3].plot(encoder_times - 1, x , color = skyblue_shades[1])
        axs2[5,3].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_on1_opto_s) > 0:
        x = np.mean(p_epoch3_on1_opto_s, axis=0)
        y = stats.sem(p_epoch3_on1_opto_s, axis=0)
        axs2[5,3].plot(encoder_times - 1, x , color = skyblue_shades[2])
        axs2[5,3].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)

    if len(p_epoch1_on2_opto_s) > 0:
        x = np.mean(p_epoch1_on2_opto_s, axis=0)
        y = stats.sem(p_epoch1_on2_opto_s, axis=0)
        axs2[5,4].plot(encoder_times - 1, x , color = skyblue_shades[0], label = 'Opto_Short_Epoch1')
        axs2[5,4].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_on2_opto_s) > 0:
        x = np.mean(p_epoch2_on2_opto_s, axis=0)
        y = stats.sem(p_epoch2_on2_opto_s, axis=0)
        axs2[5,4].plot(encoder_times - 1, x , color = skyblue_shades[1],label = 'Opto_Short_Epoch2')
        axs2[5,4].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_on2_opto_s) > 0:
        x = np.mean(p_epoch3_on2_opto_s, axis=0)
        y = stats.sem(p_epoch3_on2_opto_s, axis=0)
        axs2[5,4].plot(encoder_times - 1, x , color = skyblue_shades[2], label = 'Opto_Short_Epoch3')
        axs2[5,4].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)    
        
    # Control ALL
    if len(p_epoch1_vis1_con_l) > 0:
        x = np.mean(p_epoch1_vis1_con_l, axis=0)
        y = stats.sem(p_epoch1_vis1_con_l, axis=0)
        axs2[6,0].plot(encoder_times - 1, x , color = black_shades[0])
        axs2[6,0].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis1_con_l) > 0:
        x = np.mean(p_epoch2_vis1_con_l, axis=0)
        y = stats.sem(p_epoch2_vis1_con_l, axis=0)
        axs2[6,0].plot(encoder_times - 1, x , color = black_shades[1])
        axs2[6,0].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis1_con_l) > 0:
        x = np.mean(p_epoch3_vis1_con_l, axis=0)
        y = stats.sem(p_epoch3_vis1_con_l, axis=0)
        axs2[6,0].plot(encoder_times - 1, x , color = black_shades[2])
        axs2[6,0].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)
        
    if len(p_epoch1_vis2_con_l) > 0:
        x = np.mean(p_epoch1_vis2_con_l, axis=0)
        y = stats.sem(p_epoch1_vis2_con_l, axis=0)
        axs2[6,1].plot(encoder_times - 1, x , color = black_shades[0])
        axs2[6,1].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis2_con_l) > 0:
        x = np.mean(p_epoch2_vis2_con_l, axis=0)
        y = stats.sem(p_epoch2_vis2_con_l, axis=0)
        axs2[6,1].plot(encoder_times - 1, x , color = black_shades[1])
        axs2[6,1].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis2_con_l) > 0:
        x = np.mean(p_epoch3_vis2_con_l, axis=0)
        y = stats.sem(p_epoch3_vis2_con_l, axis=0)
        axs2[6,1].plot(encoder_times - 1, x , color = black_shades[2])
        axs2[6,1].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)
        
    if len(p_epoch1_rew_con_l) > 0:
        x = np.mean(p_epoch1_rew_con_l, axis=0)
        y = stats.sem(p_epoch1_rew_con_l, axis=0)
        axs2[6,2].plot(encoder_times - 1, x , color = black_shades[0])
        axs2[6,2].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_rew_con_l) > 0:
        x = np.mean(p_epoch2_rew_con_l, axis=0)
        y = stats.sem(p_epoch2_rew_con_l, axis=0)
        axs2[6,2].plot(encoder_times - 1, x , color = black_shades[1])
        axs2[6,2].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_rew_con_l) > 0:
        x = np.mean(p_epoch3_rew_con_l, axis=0)
        y = stats.sem(p_epoch3_rew_con_l, axis=0)
        axs2[6,2].plot(encoder_times - 1, x , color = black_shades[2])
        axs2[6,2].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)
        
    if len(p_epoch1_on1_con_l) > 0:
        x = np.mean(p_epoch1_on1_con_l, axis=0)
        y = stats.sem(p_epoch1_on1_con_l, axis=0)
        axs2[6,3].plot(encoder_times - 1, x , color = black_shades[0])
        axs2[6,3].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_on1_con_l) > 0:
        x = np.mean(p_epoch2_on1_con_l, axis=0)
        y = stats.sem(p_epoch2_on1_con_l, axis=0)
        axs2[6,3].plot(encoder_times - 1, x , color = black_shades[1])
        axs2[6,3].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_on1_con_l) > 0:
        x = np.mean(p_epoch3_on1_con_l, axis=0)
        y = stats.sem(p_epoch3_on1_con_l, axis=0)
        axs2[6,3].plot(encoder_times - 1, x , color = black_shades[2])
        axs2[6,3].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)

    if len(p_epoch1_on2_con_l) > 0:
        x = np.mean(p_epoch1_on2_con_l, axis=0)
        y = stats.sem(p_epoch1_on2_con_l, axis=0)
        axs2[6,4].plot(encoder_times - 1, x , color = black_shades[0], label = 'Control_Long_Epoch1')
        axs2[6,4].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[0], alpha=0.3)
        
    if len(p_epoch2_on2_con_l) > 0:
        x = np.mean(p_epoch2_on2_con_l, axis=0)
        y = stats.sem(p_epoch2_on2_con_l, axis=0)
        axs2[6,4].plot(encoder_times - 1, x , color = black_shades[1],label = 'Control_Long_Epoch2')
        axs2[6,4].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[1], alpha=0.3)
        
    if len(p_epoch3_on2_con_l) > 0:
        x = np.mean(p_epoch3_on2_con_l, axis=0)
        y = stats.sem(p_epoch3_on2_con_l, axis=0)
        axs2[6,4].plot(encoder_times - 1, x , color = black_shades[2], label = 'Control_Long_Epoch3')
        axs2[6,4].fill_between(encoder_times - 1, x-y , x+y , color=black_shades[2], alpha=0.3)
    # Chemo ALL
    if len(p_epoch1_vis1_chemo_l) > 0:
        x = np.mean(p_epoch1_vis1_chemo_l, axis=0)
        y = stats.sem(p_epoch1_vis1_chemo_l, axis=0)
        axs2[7,0].plot(encoder_times - 1, x , color = red_shades[0])
        axs2[7,0].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis1_chemo_l) > 0:
        x = np.mean(p_epoch2_vis1_chemo_l, axis=0)
        y = stats.sem(p_epoch2_vis1_chemo_l, axis=0)
        axs2[7,0].plot(encoder_times - 1, x , color = red_shades[1])
        axs2[7,0].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis1_chemo_l) > 0:
        x = np.mean(p_epoch3_vis1_chemo_l, axis=0)
        y = stats.sem(p_epoch3_vis1_chemo_l, axis=0)
        axs2[7,0].plot(encoder_times - 1, x , color = red_shades[2])
        axs2[7,0].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)
        
    if len(p_epoch1_vis2_chemo_l) > 0:
        x = np.mean(p_epoch1_vis2_chemo_l, axis=0)
        y = stats.sem(p_epoch1_vis2_chemo_l, axis=0)
        axs2[7,1].plot(encoder_times - 1, x , color = red_shades[0])
        axs2[7,1].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis2_chemo_l) > 0:
        x = np.mean(p_epoch2_vis2_chemo_l, axis=0)
        y = stats.sem(p_epoch2_vis2_chemo_l, axis=0)
        axs2[7,1].plot(encoder_times - 1, x , color = red_shades[1])
        axs2[7,1].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis2_chemo_l) > 0:
        x = np.mean(p_epoch3_vis2_chemo_l, axis=0)
        y = stats.sem(p_epoch3_vis2_chemo_l, axis=0)
        axs2[7,1].plot(encoder_times - 1, x , color = red_shades[2])
        axs2[7,1].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)
        
    if len(p_epoch1_rew_chemo_l) > 0:
        x = np.mean(p_epoch1_rew_chemo_l, axis=0)
        y = stats.sem(p_epoch1_rew_chemo_l, axis=0)
        axs2[7,2].plot(encoder_times - 1, x , color = red_shades[0])
        axs2[7,2].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_rew_chemo_l) > 0:
        x = np.mean(p_epoch2_rew_chemo_l, axis=0)
        y = stats.sem(p_epoch2_rew_chemo_l, axis=0)
        axs2[7,2].plot(encoder_times - 1, x , color = red_shades[1])
        axs2[7,2].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_rew_chemo_l) > 0:
        x = np.mean(p_epoch3_rew_chemo_l, axis=0)
        y = stats.sem(p_epoch3_rew_chemo_l, axis=0)
        axs2[7,2].plot(encoder_times - 1, x , color = red_shades[2])
        axs2[7,2].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)
        
    if len(p_epoch1_on1_chemo_l) > 0:
        x = np.mean(p_epoch1_on1_chemo_l, axis=0)
        y = stats.sem(p_epoch1_on1_chemo_l, axis=0)
        axs2[7,3].plot(encoder_times - 1, x , color = red_shades[0])
        axs2[7,3].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_on1_chemo_l) > 0:
        x = np.mean(p_epoch2_on1_chemo_l, axis=0)
        y = stats.sem(p_epoch2_on1_chemo_l, axis=0)
        axs2[7,3].plot(encoder_times - 1, x , color = red_shades[1])
        axs2[7,3].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_on1_chemo_l) > 0:
        x = np.mean(p_epoch3_on1_chemo_l, axis=0)
        y = stats.sem(p_epoch3_on1_chemo_l, axis=0)
        axs2[7,3].plot(encoder_times - 1, x , color = red_shades[2])
        axs2[7,3].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)

    if len(p_epoch1_on2_chemo_l) > 0:
        x = np.mean(p_epoch1_on2_chemo_l, axis=0)
        y = stats.sem(p_epoch1_on2_chemo_l, axis=0)
        axs2[7,4].plot(encoder_times - 1, x , color = red_shades[0], label = 'Chemo_Long_Epoch1')
        axs2[7,4].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[0], alpha=0.3)
        
    if len(p_epoch2_on2_chemo_l) > 0:
        x = np.mean(p_epoch2_on2_chemo_l, axis=0)
        y = stats.sem(p_epoch2_on2_chemo_l, axis=0)
        axs2[7,4].plot(encoder_times - 1, x , color = red_shades[1],label = 'Chemo_Long_Epoch2')
        axs2[7,4].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[1], alpha=0.3)
        
    if len(p_epoch3_on2_chemo_l) > 0:
        x = np.mean(p_epoch3_on2_chemo_l, axis=0)
        y = stats.sem(p_epoch3_on2_chemo_l, axis=0)
        axs2[7,4].plot(encoder_times - 1, x , color = red_shades[2], label = 'Chemo_Long_Epoch3')
        axs2[7,4].fill_between(encoder_times - 1, x-y , x+y , color=red_shades[2], alpha=0.3)
    # Opto Short
    if len(p_epoch1_vis1_opto_l) > 0:
        x = np.mean(p_epoch1_vis1_opto_l, axis=0)
        y = stats.sem(p_epoch1_vis1_opto_l, axis=0)
        axs2[8,0].plot(encoder_times - 1, x , color = skyblue_shades[0])
        axs2[8,0].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis1_opto_l) > 0:
        x = np.mean(p_epoch2_vis1_opto_l, axis=0)
        y = stats.sem(p_epoch2_vis1_opto_l, axis=0)
        axs2[8,0].plot(encoder_times - 1, x , color = skyblue_shades[1])
        axs2[8,0].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis1_opto_l) > 0:
        x = np.mean(p_epoch3_vis1_opto_l, axis=0)
        y = stats.sem(p_epoch3_vis1_opto_l, axis=0)
        axs2[8,0].plot(encoder_times - 1, x , color = skyblue_shades[2])
        axs2[8,0].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)
        
    if len(p_epoch1_vis2_opto_l) > 0:
        x = np.mean(p_epoch1_vis2_opto_l, axis=0)
        y = stats.sem(p_epoch1_vis2_opto_l, axis=0)
        axs2[8,1].plot(encoder_times - 1, x , color = skyblue_shades[0])
        axs2[8,1].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_vis2_opto_l) > 0:
        x = np.mean(p_epoch2_vis2_opto_l, axis=0)
        y = stats.sem(p_epoch2_vis2_opto_l, axis=0)
        axs2[8,1].plot(encoder_times - 1, x , color = skyblue_shades[1])
        axs2[8,1].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_vis2_opto_l) > 0:
        x = np.mean(p_epoch3_vis2_opto_l, axis=0)
        y = stats.sem(p_epoch3_vis2_opto_l, axis=0)
        axs2[8,1].plot(encoder_times - 1, x , color = skyblue_shades[2])
        axs2[8,1].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)
        
    if len(p_epoch1_rew_opto_l) > 0:
        x = np.mean(p_epoch1_rew_opto_l, axis=0)
        y = stats.sem(p_epoch1_rew_opto_l, axis=0)
        axs2[8,2].plot(encoder_times - 1, x , color = skyblue_shades[0])
        axs2[8,2].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_rew_opto_l) > 0:
        x = np.mean(p_epoch2_rew_opto_l, axis=0)
        y = stats.sem(p_epoch2_rew_opto_l, axis=0)
        axs2[8,2].plot(encoder_times - 1, x , color = skyblue_shades[1])
        axs2[8,2].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_rew_opto_l) > 0:
        x = np.mean(p_epoch3_rew_opto_l, axis=0)
        y = stats.sem(p_epoch3_rew_opto_l, axis=0)
        axs2[8,2].plot(encoder_times - 1, x , color = skyblue_shades[2])
        axs2[8,2].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)
        
    if len(p_epoch1_on1_opto_l) > 0:
        x = np.mean(p_epoch1_on1_opto_l, axis=0)
        y = stats.sem(p_epoch1_on1_opto_l, axis=0)
        axs2[8,3].plot(encoder_times - 1, x , color = skyblue_shades[0])
        axs2[8,3].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_on1_opto_l) > 0:
        x = np.mean(p_epoch2_on1_opto_l, axis=0)
        y = stats.sem(p_epoch2_on1_opto_l, axis=0)
        axs2[8,3].plot(encoder_times - 1, x , color = skyblue_shades[1])
        axs2[8,3].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_on1_opto_l) > 0:
        x = np.mean(p_epoch3_on1_opto_l, axis=0)
        y = stats.sem(p_epoch3_on1_opto_l, axis=0)
        axs2[8,3].plot(encoder_times - 1, x , color = skyblue_shades[2])
        axs2[8,3].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)

    if len(p_epoch1_on2_opto_l) > 0:
        x = np.mean(p_epoch1_on2_opto_l, axis=0)
        y = stats.sem(p_epoch1_on2_opto_l, axis=0)
        axs2[8,4].plot(encoder_times - 1, x , color = skyblue_shades[0], label = 'Opto_Long_Epoch1')
        axs2[8,4].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[0], alpha=0.3)
        
    if len(p_epoch2_on2_opto_l) > 0:
        x = np.mean(p_epoch2_on2_opto_l, axis=0)
        y = stats.sem(p_epoch2_on2_opto_l, axis=0)
        axs2[8,4].plot(encoder_times - 1, x , color = skyblue_shades[1],label = 'Opto_Long_Epoch2')
        axs2[8,4].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[1], alpha=0.3)
        
    if len(p_epoch3_on2_opto_l) > 0:
        x = np.mean(p_epoch3_on2_opto_l, axis=0)
        y = stats.sem(p_epoch3_on2_opto_l, axis=0)
        axs2[8,4].plot(encoder_times - 1, x , color = skyblue_shades[2], label = 'Opto_Long_Epoch3')
        axs2[8,4].fill_between(encoder_times - 1, x-y , x+y , color=skyblue_shades[2], alpha=0.3)      
    ##############
    axs2[0,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[1,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[2,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[3,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[4,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)   
    axs2[5,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[6,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)   
    axs2[7,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)   
    axs2[8,4].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)  
    fig2.tight_layout()     

    output_dir_onedrive = 'E:\\Ph.D\\Georgia Tech\\Behavior\\figures'
    output_figs_dir = output_dir_onedrive + subject + '/'  
    pdf_path = os.path.join(output_figs_dir, subject + '_Learning_Epoch_Superimposed_average_trajectories_PLUS.pdf')
    
    plt.rcParams['pdf.fonttype'] = 42  # Ensure text is kept as text (not outlines)
    plt.rcParams['ps.fonttype'] = 42   # For compatibility with EPS as well, if needed

    # Save both plots into a single PDF file with each on a separate page
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig1)
        pdf.savefig(fig2)

    plt.close(fig)
    plt.close(fig1)
    plt.close(fig2)               

        