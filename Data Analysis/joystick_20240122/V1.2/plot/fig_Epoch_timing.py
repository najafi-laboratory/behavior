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

#############################################################################




def Timing_epoch(
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
    chemo_conter = 0
    for i in range(0 , len(chemo_labels)):
            if chemo_labels[i] == 1:
                dates[i] = dates[i] + '(chemo)'
                chemo_conter = chemo_conter + 1
                
    dates = deduplicate_chemo(dates)            
    isSelfTimedMode  = process_matrix (session_data['isSelfTimedMode'])

    if any(0 in row for row in isSelfTimedMode):
        print('Visually Guided')
    else:
        print('Selftime')
    # POOLED VARIABLES ###########################################################
    # Epoch1
    p_epoch1_previs2delay_con_s = []
    p_epoch1_vis1_on1_con_s = []
    p_epoch1_vis2_on2_con_s = []
    p_epoch1_on1_targ1_con_s = []
    p_epoch1_on2_targ2_con_s = []
    p_epoch1_on1_amp1_con_s = []
    p_epoch1_on2_amp2_con_s = []
    p_epoch1_on_interval_con_s = []
    p_epoch1_levret1end_on2_con_s = []
    p_epoch1_base_amp1_con_s = []
    p_epoch1_base_amp2_con_s = []
    p_epoch1_max_vel1_con_s = []
    p_epoch1_max_vel2_con_s = []
    p_epoch1_velaton1_con_s = []
    p_epoch1_velaton2_con_s = []

    p_epoch1_previs2delay_chemo_s = []
    p_epoch1_vis1_on1_chemo_s = []
    p_epoch1_vis2_on2_chemo_s = []
    p_epoch1_on1_targ1_chemo_s = []
    p_epoch1_on2_targ2_chemo_s = []
    p_epoch1_on1_amp1_chemo_s = []
    p_epoch1_on2_amp2_chemo_s = []
    p_epoch1_on_interval_chemo_s = []
    p_epoch1_levret1end_on2_chemo_s = []
    p_epoch1_base_amp1_chemo_s = []
    p_epoch1_base_amp2_chemo_s = []
    p_epoch1_max_vel1_chemo_s = []
    p_epoch1_max_vel2_chemo_s = []
    p_epoch1_velaton1_chemo_s = []
    p_epoch1_velaton2_chemo_s = []

    p_epoch1_previs2delay_opto_s = []
    p_epoch1_vis1_on1_opto_s = []
    p_epoch1_vis2_on2_opto_s = []
    p_epoch1_on1_targ1_opto_s = []
    p_epoch1_on2_targ2_opto_s = []
    p_epoch1_on1_amp1_opto_s = []
    p_epoch1_on2_amp2_opto_s = []
    p_epoch1_on_interval_opto_s = []
    p_epoch1_levret1end_on2_opto_s = []
    p_epoch1_base_amp1_opto_s = []
    p_epoch1_base_amp2_opto_s = []
    p_epoch1_max_vel1_opto_s = []
    p_epoch1_max_vel2_opto_s = []
    p_epoch1_velaton1_opto_s = []
    p_epoch1_velaton2_opto_s = []

    p_epoch1_previs2delay_con_l = []
    p_epoch1_vis1_on1_con_l = []
    p_epoch1_vis2_on2_con_l = []
    p_epoch1_on1_targ1_con_l = []
    p_epoch1_on2_targ2_con_l = []
    p_epoch1_on1_amp1_con_l = []
    p_epoch1_on2_amp2_con_l = []
    p_epoch1_on_interval_con_l = []
    p_epoch1_levret1end_on2_con_l = []
    p_epoch1_base_amp1_con_l = []
    p_epoch1_base_amp2_con_l = []
    p_epoch1_max_vel1_con_l = []
    p_epoch1_max_vel2_con_l = []
    p_epoch1_velaton1_con_l = []
    p_epoch1_velaton2_con_l = []

    p_epoch1_previs2delay_chemo_l = []
    p_epoch1_vis1_on1_chemo_l = []
    p_epoch1_vis2_on2_chemo_l = []
    p_epoch1_on1_targ1_chemo_l = []
    p_epoch1_on2_targ2_chemo_l = []
    p_epoch1_on1_amp1_chemo_l = []
    p_epoch1_on2_amp2_chemo_l = []
    p_epoch1_on_interval_chemo_l = []
    p_epoch1_levret1end_on2_chemo_l = []
    p_epoch1_base_amp1_chemo_l = []
    p_epoch1_base_amp2_chemo_l = []
    p_epoch1_max_vel1_chemo_l = []
    p_epoch1_max_vel2_chemo_l = []
    p_epoch1_velaton1_chemo_l = []
    p_epoch1_velaton2_chemo_l = []

    p_epoch1_previs2delay_opto_l = []
    p_epoch1_vis1_on1_opto_l = []
    p_epoch1_vis2_on2_opto_l = []
    p_epoch1_on1_targ1_opto_l = []
    p_epoch1_on2_targ2_opto_l = []
    p_epoch1_on1_amp1_opto_l = []
    p_epoch1_on2_amp2_opto_l = []
    p_epoch1_on_interval_opto_l = []
    p_epoch1_levret1end_on2_opto_l = []
    p_epoch1_base_amp1_opto_l = []
    p_epoch1_base_amp2_opto_l = []
    p_epoch1_max_vel1_opto_l = []
    p_epoch1_max_vel2_opto_l = []
    p_epoch1_velaton1_opto_l = []
    p_epoch1_velaton2_opto_l = []
    # Epoch2
    p_epoch2_previs2delay_con_s = []
    p_epoch2_vis1_on1_con_s = []
    p_epoch2_vis2_on2_con_s = []
    p_epoch2_on1_targ1_con_s = []
    p_epoch2_on2_targ2_con_s = []
    p_epoch2_on1_amp1_con_s = []
    p_epoch2_on2_amp2_con_s = []
    p_epoch2_on_interval_con_s = []
    p_epoch2_levret1end_on2_con_s = []
    p_epoch2_base_amp1_con_s = []
    p_epoch2_base_amp2_con_s = []
    p_epoch2_max_vel1_con_s = []
    p_epoch2_max_vel2_con_s = []
    p_epoch2_velaton1_con_s = []
    p_epoch2_velaton2_con_s = []

    p_epoch2_previs2delay_chemo_s = []
    p_epoch2_vis1_on1_chemo_s = []
    p_epoch2_vis2_on2_chemo_s = []
    p_epoch2_on1_targ1_chemo_s = []
    p_epoch2_on2_targ2_chemo_s = []
    p_epoch2_on1_amp1_chemo_s = []
    p_epoch2_on2_amp2_chemo_s = []
    p_epoch2_on_interval_chemo_s = []
    p_epoch2_levret1end_on2_chemo_s = []
    p_epoch2_base_amp1_chemo_s = []
    p_epoch2_base_amp2_chemo_s = []
    p_epoch2_max_vel1_chemo_s = []
    p_epoch2_max_vel2_chemo_s = []
    p_epoch2_velaton1_chemo_s = []
    p_epoch2_velaton2_chemo_s = []

    p_epoch2_previs2delay_opto_s = []
    p_epoch2_vis1_on1_opto_s = []
    p_epoch2_vis2_on2_opto_s = []
    p_epoch2_on1_targ1_opto_s = []
    p_epoch2_on2_targ2_opto_s = []
    p_epoch2_on1_amp1_opto_s = []
    p_epoch2_on2_amp2_opto_s = []
    p_epoch2_on_interval_opto_s = []
    p_epoch2_levret1end_on2_opto_s = []
    p_epoch2_base_amp1_opto_s = []
    p_epoch2_base_amp2_opto_s = []
    p_epoch2_max_vel1_opto_s = []
    p_epoch2_max_vel2_opto_s = []
    p_epoch2_velaton1_opto_s = []
    p_epoch2_velaton2_opto_s = []

    p_epoch2_previs2delay_con_l = []
    p_epoch2_vis1_on1_con_l = []
    p_epoch2_vis2_on2_con_l = []
    p_epoch2_on1_targ1_con_l = []
    p_epoch2_on2_targ2_con_l = []
    p_epoch2_on1_amp1_con_l = []
    p_epoch2_on2_amp2_con_l = []
    p_epoch2_on_interval_con_l = []
    p_epoch2_levret1end_on2_con_l = []
    p_epoch2_base_amp1_con_l = []
    p_epoch2_base_amp2_con_l = []
    p_epoch2_max_vel1_con_l = []
    p_epoch2_max_vel2_con_l = []
    p_epoch2_velaton1_con_l = []
    p_epoch2_velaton2_con_l = []

    p_epoch2_previs2delay_chemo_l = []
    p_epoch2_vis1_on1_chemo_l = []
    p_epoch2_vis2_on2_chemo_l = []
    p_epoch2_on1_targ1_chemo_l = []
    p_epoch2_on2_targ2_chemo_l = []
    p_epoch2_on1_amp1_chemo_l = []
    p_epoch2_on2_amp2_chemo_l = []
    p_epoch2_on_interval_chemo_l = []
    p_epoch2_levret1end_on2_chemo_l = []
    p_epoch2_base_amp1_chemo_l = []
    p_epoch2_base_amp2_chemo_l = []
    p_epoch2_max_vel1_chemo_l = []
    p_epoch2_max_vel2_chemo_l = []
    p_epoch2_velaton1_chemo_l = []
    p_epoch2_velaton2_chemo_l = []

    p_epoch2_previs2delay_opto_l = []
    p_epoch2_vis1_on1_opto_l = []
    p_epoch2_vis2_on2_opto_l = []
    p_epoch2_on1_targ1_opto_l = []
    p_epoch2_on2_targ2_opto_l = []
    p_epoch2_on1_amp1_opto_l = []
    p_epoch2_on2_amp2_opto_l = []
    p_epoch2_on_interval_opto_l = []
    p_epoch2_levret1end_on2_opto_l = []
    p_epoch2_base_amp1_opto_l = []
    p_epoch2_base_amp2_opto_l = []
    p_epoch2_max_vel1_opto_l = []
    p_epoch2_max_vel2_opto_l = []
    p_epoch2_velaton1_opto_l = []
    p_epoch2_velaton2_opto_l = []
    # Epoch3
    p_epoch3_previs2delay_con_s = []
    p_epoch3_vis1_on1_con_s = []
    p_epoch3_vis2_on2_con_s = []
    p_epoch3_on1_targ1_con_s = []
    p_epoch3_on2_targ2_con_s = []
    p_epoch3_on1_amp1_con_s = []
    p_epoch3_on2_amp2_con_s = []
    p_epoch3_on_interval_con_s = []
    p_epoch3_levret1end_on2_con_s = []
    p_epoch3_base_amp1_con_s = []
    p_epoch3_base_amp2_con_s = []
    p_epoch3_max_vel1_con_s = []
    p_epoch3_max_vel2_con_s = []
    p_epoch3_velaton1_con_s = []
    p_epoch3_velaton2_con_s = []

    p_epoch3_previs2delay_chemo_s = []
    p_epoch3_vis1_on1_chemo_s = []
    p_epoch3_vis2_on2_chemo_s = []
    p_epoch3_on1_targ1_chemo_s = []
    p_epoch3_on2_targ2_chemo_s = []
    p_epoch3_on1_amp1_chemo_s = []
    p_epoch3_on2_amp2_chemo_s = []
    p_epoch3_on_interval_chemo_s = []
    p_epoch3_levret1end_on2_chemo_s = []
    p_epoch3_base_amp1_chemo_s = []
    p_epoch3_base_amp2_chemo_s = []
    p_epoch3_max_vel1_chemo_s = []
    p_epoch3_max_vel2_chemo_s = []
    p_epoch3_velaton1_chemo_s = []
    p_epoch3_velaton2_chemo_s = []

    p_epoch3_previs2delay_opto_s = []
    p_epoch3_vis1_on1_opto_s = []
    p_epoch3_vis2_on2_opto_s = []
    p_epoch3_on1_targ1_opto_s = []
    p_epoch3_on2_targ2_opto_s = []
    p_epoch3_on1_amp1_opto_s = []
    p_epoch3_on2_amp2_opto_s = []
    p_epoch3_on_interval_opto_s = []
    p_epoch3_levret1end_on2_opto_s = []
    p_epoch3_base_amp1_opto_s = []
    p_epoch3_base_amp2_opto_s = []
    p_epoch3_max_vel1_opto_s = []
    p_epoch3_max_vel2_opto_s = []
    p_epoch3_velaton1_opto_s = []
    p_epoch3_velaton2_opto_s = []

    p_epoch3_previs2delay_con_l = []
    p_epoch3_vis1_on1_con_l = []
    p_epoch3_vis2_on2_con_l = []
    p_epoch3_on1_targ1_con_l = []
    p_epoch3_on2_targ2_con_l = []
    p_epoch3_on1_amp1_con_l = []
    p_epoch3_on2_amp2_con_l = []
    p_epoch3_on_interval_con_l = []
    p_epoch3_levret1end_on2_con_l = []
    p_epoch3_base_amp1_con_l = []
    p_epoch3_base_amp2_con_l = []
    p_epoch3_max_vel1_con_l = []
    p_epoch3_max_vel2_con_l = []
    p_epoch3_velaton1_con_l = []
    p_epoch3_velaton2_con_l = []

    p_epoch3_previs2delay_chemo_l = []
    p_epoch3_vis1_on1_chemo_l = []
    p_epoch3_vis2_on2_chemo_l = []
    p_epoch3_on1_targ1_chemo_l = []
    p_epoch3_on2_targ2_chemo_l = []
    p_epoch3_on1_amp1_chemo_l = []
    p_epoch3_on2_amp2_chemo_l = []
    p_epoch3_on_interval_chemo_l = []
    p_epoch3_levret1end_on2_chemo_l = []
    p_epoch3_base_amp1_chemo_l = []
    p_epoch3_base_amp2_chemo_l = []
    p_epoch3_max_vel1_chemo_l = []
    p_epoch3_max_vel2_chemo_l = []
    p_epoch3_velaton1_chemo_l = []
    p_epoch3_velaton2_chemo_l = []

    p_epoch3_previs2delay_opto_l = []
    p_epoch3_vis1_on1_opto_l = []
    p_epoch3_vis2_on2_opto_l = []
    p_epoch3_on1_targ1_opto_l = []
    p_epoch3_on2_targ2_opto_l = []
    p_epoch3_on1_amp1_opto_l = []
    p_epoch3_on2_amp2_opto_l = []
    p_epoch3_on_interval_opto_l = []
    p_epoch3_levret1end_on2_opto_l = []
    p_epoch3_base_amp1_opto_l = []
    p_epoch3_base_amp2_opto_l = []
    p_epoch3_max_vel1_opto_l = []
    p_epoch3_max_vel2_opto_l = []
    p_epoch3_velaton1_opto_l = []
    p_epoch3_velaton2_opto_l = []
    # GRAND AVERAGE ###########################################################
    # Epoch1
    G_epoch1_previs2delay_con_s = []
    G_epoch1_vis1_on1_con_s = []
    G_epoch1_vis2_on2_con_s = []
    G_epoch1_on1_targ1_con_s = []
    G_epoch1_on2_targ2_con_s = []
    G_epoch1_on1_amp1_con_s = []
    G_epoch1_on2_amp2_con_s = []
    G_epoch1_on_interval_con_s = []
    G_epoch1_levret1end_on2_con_s = []
    G_epoch1_base_amp1_con_s = []
    G_epoch1_base_amp2_con_s = []
    G_epoch1_max_vel1_con_s = []
    G_epoch1_max_vel2_con_s = []
    G_epoch1_velaton1_con_s = []
    G_epoch1_velaton2_con_s = []

    G_epoch1_previs2delay_chemo_s = []
    G_epoch1_vis1_on1_chemo_s = []
    G_epoch1_vis2_on2_chemo_s = []
    G_epoch1_on1_targ1_chemo_s = []
    G_epoch1_on2_targ2_chemo_s = []
    G_epoch1_on1_amp1_chemo_s = []
    G_epoch1_on2_amp2_chemo_s = []
    G_epoch1_on_interval_chemo_s = []
    G_epoch1_levret1end_on2_chemo_s = []
    G_epoch1_base_amp1_chemo_s = []
    G_epoch1_base_amp2_chemo_s = []
    G_epoch1_max_vel1_chemo_s = []
    G_epoch1_max_vel2_chemo_s = []
    G_epoch1_velaton1_chemo_s = []
    G_epoch1_velaton2_chemo_s = []

    G_epoch1_previs2delay_opto_s = []
    G_epoch1_vis1_on1_opto_s = []
    G_epoch1_vis2_on2_opto_s = []
    G_epoch1_on1_targ1_opto_s = []
    G_epoch1_on2_targ2_opto_s = []
    G_epoch1_on1_amp1_opto_s = []
    G_epoch1_on2_amp2_opto_s = []
    G_epoch1_on_interval_opto_s = []
    G_epoch1_levret1end_on2_opto_s = []
    G_epoch1_base_amp1_opto_s = []
    G_epoch1_base_amp2_opto_s = []
    G_epoch1_max_vel1_opto_s = []
    G_epoch1_max_vel2_opto_s = []
    G_epoch1_velaton1_opto_s = []
    G_epoch1_velaton2_opto_s = []

    G_epoch1_previs2delay_con_l = []
    G_epoch1_vis1_on1_con_l = []
    G_epoch1_vis2_on2_con_l = []
    G_epoch1_on1_targ1_con_l = []
    G_epoch1_on2_targ2_con_l = []
    G_epoch1_on1_amp1_con_l = []
    G_epoch1_on2_amp2_con_l = []
    G_epoch1_on_interval_con_l = []
    G_epoch1_levret1end_on2_con_l = []
    G_epoch1_base_amp1_con_l = []
    G_epoch1_base_amp2_con_l = []
    G_epoch1_max_vel1_con_l = []
    G_epoch1_max_vel2_con_l = []
    G_epoch1_velaton1_con_l = []
    G_epoch1_velaton2_con_l = []

    G_epoch1_previs2delay_chemo_l = []
    G_epoch1_vis1_on1_chemo_l = []
    G_epoch1_vis2_on2_chemo_l = []
    G_epoch1_on1_targ1_chemo_l = []
    G_epoch1_on2_targ2_chemo_l = []
    G_epoch1_on1_amp1_chemo_l = []
    G_epoch1_on2_amp2_chemo_l = []
    G_epoch1_on_interval_chemo_l = []
    G_epoch1_levret1end_on2_chemo_l = []
    G_epoch1_base_amp1_chemo_l = []
    G_epoch1_base_amp2_chemo_l = []
    G_epoch1_max_vel1_chemo_l = []
    G_epoch1_max_vel2_chemo_l = []
    G_epoch1_velaton1_chemo_l = []
    G_epoch1_velaton2_chemo_l = []

    G_epoch1_previs2delay_opto_l = []
    G_epoch1_vis1_on1_opto_l = []
    G_epoch1_vis2_on2_opto_l = []
    G_epoch1_on1_targ1_opto_l = []
    G_epoch1_on2_targ2_opto_l = []
    G_epoch1_on1_amp1_opto_l = []
    G_epoch1_on2_amp2_opto_l = []
    G_epoch1_on_interval_opto_l = []
    G_epoch1_levret1end_on2_opto_l = []
    G_epoch1_base_amp1_opto_l = []
    G_epoch1_base_amp2_opto_l = []
    G_epoch1_max_vel1_opto_l = []
    G_epoch1_max_vel2_opto_l = []
    G_epoch1_velaton1_opto_l = []
    G_epoch1_velaton2_opto_l = []
    # Epoch2
    G_epoch2_previs2delay_con_s = []
    G_epoch2_vis1_on1_con_s = []
    G_epoch2_vis2_on2_con_s = []
    G_epoch2_on1_targ1_con_s = []
    G_epoch2_on2_targ2_con_s = []
    G_epoch2_on1_amp1_con_s = []
    G_epoch2_on2_amp2_con_s = []
    G_epoch2_on_interval_con_s = []
    G_epoch2_levret1end_on2_con_s = []
    G_epoch2_base_amp1_con_s = []
    G_epoch2_base_amp2_con_s = []
    G_epoch2_max_vel1_con_s = []
    G_epoch2_max_vel2_con_s = []
    G_epoch2_velaton1_con_s = []
    G_epoch2_velaton2_con_s = []

    G_epoch2_previs2delay_chemo_s = []
    G_epoch2_vis1_on1_chemo_s = []
    G_epoch2_vis2_on2_chemo_s = []
    G_epoch2_on1_targ1_chemo_s = []
    G_epoch2_on2_targ2_chemo_s = []
    G_epoch2_on1_amp1_chemo_s = []
    G_epoch2_on2_amp2_chemo_s = []
    G_epoch2_on_interval_chemo_s = []
    G_epoch2_levret1end_on2_chemo_s = []
    G_epoch2_base_amp1_chemo_s = []
    G_epoch2_base_amp2_chemo_s = []
    G_epoch2_max_vel1_chemo_s = []
    G_epoch2_max_vel2_chemo_s = []
    G_epoch2_velaton1_chemo_s = []
    G_epoch2_velaton2_chemo_s = []

    G_epoch2_previs2delay_opto_s = []
    G_epoch2_vis1_on1_opto_s = []
    G_epoch2_vis2_on2_opto_s = []
    G_epoch2_on1_targ1_opto_s = []
    G_epoch2_on2_targ2_opto_s = []
    G_epoch2_on1_amp1_opto_s = []
    G_epoch2_on2_amp2_opto_s = []
    G_epoch2_on_interval_opto_s = []
    G_epoch2_levret1end_on2_opto_s = []
    G_epoch2_base_amp1_opto_s = []
    G_epoch2_base_amp2_opto_s = []
    G_epoch2_max_vel1_opto_s = []
    G_epoch2_max_vel2_opto_s = []
    G_epoch2_velaton1_opto_s = []
    G_epoch2_velaton2_opto_s = []

    G_epoch2_previs2delay_con_l = []
    G_epoch2_vis1_on1_con_l = []
    G_epoch2_vis2_on2_con_l = []
    G_epoch2_on1_targ1_con_l = []
    G_epoch2_on2_targ2_con_l = []
    G_epoch2_on1_amp1_con_l = []
    G_epoch2_on2_amp2_con_l = []
    G_epoch2_on_interval_con_l = []
    G_epoch2_levret1end_on2_con_l = []
    G_epoch2_base_amp1_con_l = []
    G_epoch2_base_amp2_con_l = []
    G_epoch2_max_vel1_con_l = []
    G_epoch2_max_vel2_con_l = []
    G_epoch2_velaton1_con_l = []
    G_epoch2_velaton2_con_l = []

    G_epoch2_previs2delay_chemo_l = []
    G_epoch2_vis1_on1_chemo_l = []
    G_epoch2_vis2_on2_chemo_l = []
    G_epoch2_on1_targ1_chemo_l = []
    G_epoch2_on2_targ2_chemo_l = []
    G_epoch2_on1_amp1_chemo_l = []
    G_epoch2_on2_amp2_chemo_l = []
    G_epoch2_on_interval_chemo_l = []
    G_epoch2_levret1end_on2_chemo_l = []
    G_epoch2_base_amp1_chemo_l = []
    G_epoch2_base_amp2_chemo_l = []
    G_epoch2_max_vel1_chemo_l = []
    G_epoch2_max_vel2_chemo_l = []
    G_epoch2_velaton1_chemo_l = []
    G_epoch2_velaton2_chemo_l = []

    G_epoch2_previs2delay_opto_l = []
    G_epoch2_vis1_on1_opto_l = []
    G_epoch2_vis2_on2_opto_l = []
    G_epoch2_on1_targ1_opto_l = []
    G_epoch2_on2_targ2_opto_l = []
    G_epoch2_on1_amp1_opto_l = []
    G_epoch2_on2_amp2_opto_l = []
    G_epoch2_on_interval_opto_l = []
    G_epoch2_levret1end_on2_opto_l = []
    G_epoch2_base_amp1_opto_l = []
    G_epoch2_base_amp2_opto_l = []
    G_epoch2_max_vel1_opto_l = []
    G_epoch2_max_vel2_opto_l = []
    G_epoch2_velaton1_opto_l = []
    G_epoch2_velaton2_opto_l = []
    # Epoch3
    G_epoch3_previs2delay_con_s = []
    G_epoch3_vis1_on1_con_s = []
    G_epoch3_vis2_on2_con_s = []
    G_epoch3_on1_targ1_con_s = []
    G_epoch3_on2_targ2_con_s = []
    G_epoch3_on1_amp1_con_s = []
    G_epoch3_on2_amp2_con_s = []
    G_epoch3_on_interval_con_s = []
    G_epoch3_levret1end_on2_con_s = []
    G_epoch3_base_amp1_con_s = []
    G_epoch3_base_amp2_con_s = []
    G_epoch3_max_vel1_con_s = []
    G_epoch3_max_vel2_con_s = []
    G_epoch3_velaton1_con_s = []
    G_epoch3_velaton2_con_s = []

    G_epoch3_previs2delay_chemo_s = []
    G_epoch3_vis1_on1_chemo_s = []
    G_epoch3_vis2_on2_chemo_s = []
    G_epoch3_on1_targ1_chemo_s = []
    G_epoch3_on2_targ2_chemo_s = []
    G_epoch3_on1_amp1_chemo_s = []
    G_epoch3_on2_amp2_chemo_s = []
    G_epoch3_on_interval_chemo_s = []
    G_epoch3_levret1end_on2_chemo_s = []
    G_epoch3_base_amp1_chemo_s = []
    G_epoch3_base_amp2_chemo_s = []
    G_epoch3_max_vel1_chemo_s = []
    G_epoch3_max_vel2_chemo_s = []
    G_epoch3_velaton1_chemo_s = []
    G_epoch3_velaton2_chemo_s = []

    G_epoch3_previs2delay_opto_s = []
    G_epoch3_vis1_on1_opto_s = []
    G_epoch3_vis2_on2_opto_s = []
    G_epoch3_on1_targ1_opto_s = []
    G_epoch3_on2_targ2_opto_s = []
    G_epoch3_on1_amp1_opto_s = []
    G_epoch3_on2_amp2_opto_s = []
    G_epoch3_on_interval_opto_s = []
    G_epoch3_levret1end_on2_opto_s = []
    G_epoch3_base_amp1_opto_s = []
    G_epoch3_base_amp2_opto_s = []
    G_epoch3_max_vel1_opto_s = []
    G_epoch3_max_vel2_opto_s = []
    G_epoch3_velaton1_opto_s = []
    G_epoch3_velaton2_opto_s = []

    G_epoch3_previs2delay_con_l = []
    G_epoch3_vis1_on1_con_l = []
    G_epoch3_vis2_on2_con_l = []
    G_epoch3_on1_targ1_con_l = []
    G_epoch3_on2_targ2_con_l = []
    G_epoch3_on1_amp1_con_l = []
    G_epoch3_on2_amp2_con_l = []
    G_epoch3_on_interval_con_l = []
    G_epoch3_levret1end_on2_con_l = []
    G_epoch3_base_amp1_con_l = []
    G_epoch3_base_amp2_con_l = []
    G_epoch3_max_vel1_con_l = []
    G_epoch3_max_vel2_con_l = []
    G_epoch3_velaton1_con_l = []
    G_epoch3_velaton2_con_l = []

    G_epoch3_previs2delay_chemo_l = []
    G_epoch3_vis1_on1_chemo_l = []
    G_epoch3_vis2_on2_chemo_l = []
    G_epoch3_on1_targ1_chemo_l = []
    G_epoch3_on2_targ2_chemo_l = []
    G_epoch3_on1_amp1_chemo_l = []
    G_epoch3_on2_amp2_chemo_l = []
    G_epoch3_on_interval_chemo_l = []
    G_epoch3_levret1end_on2_chemo_l = []
    G_epoch3_base_amp1_chemo_l = []
    G_epoch3_base_amp2_chemo_l = []
    G_epoch3_max_vel1_chemo_l = []
    G_epoch3_max_vel2_chemo_l = []
    G_epoch3_velaton1_chemo_l = []
    G_epoch3_velaton2_chemo_l = []

    G_epoch3_previs2delay_opto_l = []
    G_epoch3_vis1_on1_opto_l = []
    G_epoch3_vis2_on2_opto_l = []
    G_epoch3_on1_targ1_opto_l = []
    G_epoch3_on2_targ2_opto_l = []
    G_epoch3_on1_amp1_opto_l = []
    G_epoch3_on2_amp2_opto_l = []
    G_epoch3_on_interval_opto_l = []
    G_epoch3_levret1end_on2_opto_l = []
    G_epoch3_base_amp1_opto_l = []
    G_epoch3_base_amp2_opto_l = []
    G_epoch3_max_vel1_opto_l = []
    G_epoch3_max_vel2_opto_l = []
    G_epoch3_velaton1_opto_l = []
    G_epoch3_velaton2_opto_l = []
    ########################################################
    # Plotting sessions
    # Define colors
    black_shades = [(150, 150, 150), (100, 100, 100), (50, 50, 50)]
    red_shades = [(255, 102, 102), (255, 51, 51), (204, 0, 0)]
    skyblue_shades = [(135, 206, 235), (70, 130, 180), (0, 105, 148)]
    # Normalize the colors to [0, 1] range for matplotlib
    black_shades = [tuple(c/255 for c in shade) for shade in black_shades]
    red_shades = [tuple(c/255 for c in shade) for shade in red_shades]
    skyblue_shades = [tuple(c/255 for c in shade) for shade in skyblue_shades]

    offset = 0.1
    fig1, axs1 = plt.subplots(nrows=16, ncols=2, figsize=(10 + 1.7 *len(session_id) , 80))
    fig1.subplots_adjust(hspace=0.7)
    fig1.suptitle(subject + '\n Time, Amplitude and Kinematic Quatification over sessions\n')
    # prepress / previs delay
    if any(0 in row for row in isSelfTimedMode):
        axs1[0,0].set_title('Short Trials PreVis2Delay\n') 
    else:
        axs1[0,0].set_title(6*'\n' + 'Short Trials PrePress2Delay.\n') 
    axs1[0,0].spines['right'].set_visible(False)
    axs1[0,0].spines['top'].set_visible(False)
    axs1[0,0].set_xlabel('Sessions')
    axs1[0,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[0,0].set_ylim((-0.05,0.2))
    axs1[0,0].set_xticks(session_id)
    axs1[0,0].set_xticklabels(dates, rotation=90, ha = 'center')
        
    if any(0 in row for row in isSelfTimedMode):
        axs1[1,0].set_title('Long Trials PreVis2Delay\n') 
    else:
        axs1[1,0].set_title('Long Trials PrePress2Delay.\n') 
    axs1[1,0].spines['right'].set_visible(False)
    axs1[1,0].spines['top'].set_visible(False)
    axs1[1,0].set_xlabel('Sessions')
    axs1[1,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[1,0].set_xticks(session_id)
    axs1[1,0].set_xticklabels(dates, rotation=90, ha = 'center')
    # Vis to Onset
    axs1[2,0].set_title('Short Trials Vis1 to Onset1\n') 
    axs1[2,0].spines['right'].set_visible(False)
    axs1[2,0].spines['top'].set_visible(False)
    axs1[2,0].set_xlabel('Sessions')
    axs1[2,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[2,0].set_xticks(session_id)
    axs1[2,0].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[3,0].set_title('Long Trials Vis1 to Onset1\n') 
    axs1[3,0].spines['right'].set_visible(False)
    axs1[3,0].spines['top'].set_visible(False)
    axs1[3,0].set_xlabel('Sessions')
    axs1[3,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[3,0].set_xticks(session_id)
    axs1[3,0].set_xticklabels(dates, rotation=90, ha = 'center')

    if any(0 in row for row in isSelfTimedMode):
        axs1[2,1].set_title('Short Trials Vis2 to Onset2\n') 
    else:
        axs1[2,1].set_title('Short Trials WaitforPress2 to Onset2.\n') 
    axs1[2,1].spines['right'].set_visible(False)
    axs1[2,1].spines['top'].set_visible(False)
    axs1[2,1].set_xlabel('Sessions')
    axs1[2,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[2,1].set_xticks(session_id)
    axs1[2,1].set_xticklabels(dates, rotation=90, ha = 'center')

    if any(0 in row for row in isSelfTimedMode):
        axs1[3,1].set_title('Long Trials Vis2 to Onset2\n') 
    else:
        axs1[3,1].set_title('Long Trials WaitforPress2 to Onset2.\n') 
    axs1[3,1].spines['right'].set_visible(False)
    axs1[3,1].spines['top'].set_visible(False)
    axs1[3,1].set_xlabel('Sessions')
    axs1[3,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[3,1].set_xticks(session_id)
    axs1[3,1].set_xticklabels(dates, rotation=90, ha = 'center')
    # Onset to Target
    axs1[4,0].set_title('Short Trials Onset1 to Target1\n') 
    axs1[4,0].spines['right'].set_visible(False)
    axs1[4,0].spines['top'].set_visible(False)
    axs1[4,0].set_xlabel('Sessions')
    axs1[4,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[4,0].set_xticks(session_id)
    axs1[4,0].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[5,0].set_title('Long Trials Onset1 to Target1\n') 
    axs1[5,0].spines['right'].set_visible(False)
    axs1[5,0].spines['top'].set_visible(False)
    axs1[5,0].set_xlabel('Sessions')
    axs1[5,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[5,0].set_xticks(session_id)
    axs1[5,0].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[4,1].set_title('Short Trials Onset2 to Target2\n') 
    axs1[4,1].spines['right'].set_visible(False)
    axs1[4,1].spines['top'].set_visible(False)
    axs1[4,1].set_xlabel('Sessions')
    axs1[4,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[4,1].set_xticks(session_id)
    axs1[4,1].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[5,1].set_title('Long Trials Onset2 to Target2\n') 
    axs1[5,1].spines['right'].set_visible(False)
    axs1[5,1].spines['top'].set_visible(False)
    axs1[5,1].set_xlabel('Sessions')
    axs1[5,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[5,1].set_xticks(session_id)
    axs1[5,1].set_xticklabels(dates, rotation=90, ha = 'center')
    # Onset to peak time
    axs1[6,0].set_title('Short Trials Onset1 to Peak1\n') 
    axs1[6,0].spines['right'].set_visible(False)
    axs1[6,0].spines['top'].set_visible(False)
    axs1[6,0].set_xlabel('Sessions')
    axs1[6,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[6,0].set_xticks(session_id)
    axs1[6,0].set_xticklabels(dates, rotation=90, ha = 'center')
        
    axs1[7,0].set_title('Long Trials Onset1 to Peak1\n') 
    axs1[7,0].spines['right'].set_visible(False)
    axs1[7,0].spines['top'].set_visible(False)
    axs1[7,0].set_xlabel('Sessions')
    axs1[7,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[7,0].set_xticks(session_id)
    axs1[7,0].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[6,1].set_title('Short Trials Onset2 to Peak2\n') 
    axs1[6,1].spines['right'].set_visible(False)
    axs1[6,1].spines['top'].set_visible(False)
    axs1[6,1].set_xlabel('Sessions')
    axs1[6,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[6,1].set_xticks(session_id)
    axs1[6,1].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[7,1].set_title('Long Trials Onset2 to Peak2\n') 
    axs1[7,1].spines['right'].set_visible(False)
    axs1[7,1].spines['top'].set_visible(False)
    axs1[7,1].set_xlabel('Sessions')
    axs1[7,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[7,1].set_xticks(session_id)
    axs1[7,1].set_xticklabels(dates, rotation=90, ha = 'center')
    # Onset Interval
    axs1[8,0].set_title('Short Trials Onset1 and Onset2 Interval\n') 
    axs1[8,0].spines['right'].set_visible(False)
    axs1[8,0].spines['top'].set_visible(False)
    axs1[8,0].set_xlabel('Sessions')
    axs1[8,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[8,0].set_xticks(session_id)
    axs1[8,0].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[9,0].set_title('Long Trials Onset1 and Onset2 Interval\n') 
    axs1[9,0].spines['right'].set_visible(False)
    axs1[9,0].spines['top'].set_visible(False)
    axs1[9,0].set_xlabel('Sessions')
    axs1[9,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[9,0].set_xticks(session_id)
    axs1[9,0].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[8,1].set_title('Short Trials LeverRetract1_End to Onset2\n') 
    axs1[8,1].spines['right'].set_visible(False)
    axs1[8,1].spines['top'].set_visible(False)
    axs1[8,1].set_xlabel('Sessions')
    axs1[8,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[8,1].set_xticks(session_id)
    axs1[8,1].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[9,1].set_title('Long Trials LeverRetract1_End to Onset2\n') 
    axs1[9,1].spines['right'].set_visible(False)
    axs1[9,1].spines['top'].set_visible(False)
    axs1[9,1].set_xlabel('Sessions')
    axs1[9,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs1[9,1].set_xticks(session_id)
    axs1[9,1].set_xticklabels(dates, rotation=90, ha = 'center')
    # Baseline to Peak
    axs1[10,0].set_title('Short Trials Baseline to Peak1\n') 
    axs1[10,0].spines['right'].set_visible(False)
    axs1[10,0].spines['top'].set_visible(False)
    axs1[10,0].set_xlabel('Sessions')
    axs1[10,0].set_ylabel('Lever Deflection(deg) Mean +/- SEM')
    axs1[10,0].set_xticks(session_id)
    axs1[10,0].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[11,0].set_title('Long Trials Baseline to Peak1\n') 
    axs1[11,0].spines['right'].set_visible(False)
    axs1[11,0].spines['top'].set_visible(False)
    axs1[11,0].set_xlabel('Sessions')
    axs1[11,0].set_ylabel('Lever Deflection(deg) Mean +/- SEM')
    axs1[11,0].set_xticks(session_id)
    axs1[11,0].set_xticklabels(dates, rotation=90, ha = 'center')
        
    axs1[10,1].set_title('Short Trials Baseline to Peak2\n') 
    axs1[10,1].spines['right'].set_visible(False)
    axs1[10,1].spines['top'].set_visible(False)
    axs1[10,1].set_xlabel('Sessions')
    axs1[10,1].set_ylabel('Lever Deflection(deg) Mean +/- SEM')
    axs1[10,1].set_xticks(session_id)
    axs1[10,1].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[11,1].set_title('Long Trials Baseline to Peak2\n') 
    axs1[11,1].spines['right'].set_visible(False)
    axs1[11,1].spines['top'].set_visible(False)
    axs1[11,1].set_xlabel('Sessions')
    axs1[11,1].set_ylabel('Lever Deflection(deg) Mean +/- SEM')
    axs1[11,1].set_xticks(session_id)
    axs1[11,1].set_xticklabels(dates, rotation=90, ha = 'center')
    # Max Velocity Press
    axs1[12,0].set_title('Short Trials Max Velocity Push1\n') 
    axs1[12,0].spines['right'].set_visible(False)
    axs1[12,0].spines['top'].set_visible(False)
    axs1[12,0].set_xlabel('Sessions')
    axs1[12,0].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs1[12,0].set_xticks(session_id)
    axs1[12,0].set_xticklabels(dates, rotation=90, ha = 'center')
        
    axs1[13,0].set_title('Long Trials Max Velocity Push1\n') 
    axs1[13,0].spines['right'].set_visible(False)
    axs1[13,0].spines['top'].set_visible(False)
    axs1[13,0].set_xlabel('Sessions')
    axs1[13,0].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs1[13,0].set_xticks(session_id)
    axs1[13,0].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[12,1].set_title('Short Trials Max Velocity Push2\n') 
    axs1[12,1].spines['right'].set_visible(False)
    axs1[12,1].spines['top'].set_visible(False)
    axs1[12,1].set_xlabel('Sessions')
    axs1[12,1].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs1[12,1].set_xticks(session_id)
    axs1[12,1].set_xticklabels(dates, rotation=90, ha = 'center')
        
    axs1[13,1].set_title('Long Trials Max Velocity Push2\n') 
    axs1[13,1].spines['right'].set_visible(False)
    axs1[13,1].spines['top'].set_visible(False)
    axs1[13,1].set_xlabel('Sessions')
    axs1[13,1].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs1[13,1].set_xticks(session_id)
    axs1[13,1].set_xticklabels(dates, rotation=90, ha = 'center')
    # Velocity at press
    axs1[14,0].set_title('Short Trials Velocity at Onset1\n') 
    axs1[14,0].spines['right'].set_visible(False)
    axs1[14,0].spines['top'].set_visible(False)
    axs1[14,0].set_xlabel('Sessions')
    axs1[14,0].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs1[14,0].set_xticks(session_id)
    axs1[14,0].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[15,0].set_title('Long Trials Velocity at Onset1\n') 
    axs1[15,0].spines['right'].set_visible(False)
    axs1[15,0].spines['top'].set_visible(False)
    axs1[15,0].set_xlabel('Sessions')
    axs1[15,0].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs1[15,0].set_xticks(session_id)
    axs1[15,0].set_xticklabels(dates, rotation=90, ha = 'center')
        
    axs1[14,1].set_title('Short Trials Velocity at Onset2\n') 
    axs1[14,1].spines['right'].set_visible(False)
    axs1[14,1].spines['top'].set_visible(False)
    axs1[14,1].set_xlabel('Sessions')
    axs1[14,1].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs1[14,1].set_xticks(session_id)
    axs1[14,1].set_xticklabels(dates, rotation=90, ha = 'center')

    axs1[15,1].set_title('Long Trials Velocity at Onset2\n') 
    axs1[15,1].spines['right'].set_visible(False)
    axs1[15,1].spines['top'].set_visible(False)
    axs1[15,1].set_xlabel('Sessions')
    axs1[15,1].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs1[15,1].set_xticks(session_id)
    axs1[15,1].set_xticklabels(dates, rotation=90, ha = 'center')
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
        #####################################################
        Block_size = session_data['raw'][i]['TrialSettings'][0]['GUI']['NumTrialsPerBlock']
        nTrials = session_data['raw'][i]['nTrials']
        epoch1 , epoch2 , epoch3 = epoching_session(nTrials, Block_size)
        
        if nTrials <= Block_size:
            print('not adequate trials for analysis in session:',i)
            continue
        # SESSION VARIABLES ####################################################
        # Epoch1
        epoch1_previs2delay_con_s = []
        epoch1_vis1_on1_con_s = []
        epoch1_vis2_on2_con_s = []
        epoch1_on1_targ1_con_s = []
        epoch1_on2_targ2_con_s = []
        epoch1_on1_amp1_con_s = []
        epoch1_on2_amp2_con_s = []
        epoch1_on_interval_con_s = []
        epoch1_levret1end_on2_con_s = []
        epoch1_base_amp1_con_s = []
        epoch1_base_amp2_con_s = []
        epoch1_max_vel1_con_s = []
        epoch1_max_vel2_con_s = []
        epoch1_velaton1_con_s = []
        epoch1_velaton2_con_s = []
        
        epoch1_previs2delay_chemo_s = []
        epoch1_vis1_on1_chemo_s = []
        epoch1_vis2_on2_chemo_s = []
        epoch1_on1_targ1_chemo_s = []
        epoch1_on2_targ2_chemo_s = []
        epoch1_on1_amp1_chemo_s = []
        epoch1_on2_amp2_chemo_s = []
        epoch1_on_interval_chemo_s = []
        epoch1_levret1end_on2_chemo_s = []
        epoch1_base_amp1_chemo_s = []
        epoch1_base_amp2_chemo_s = []
        epoch1_max_vel1_chemo_s = []
        epoch1_max_vel2_chemo_s = []
        epoch1_velaton1_chemo_s = []
        epoch1_velaton2_chemo_s = []
        
        epoch1_previs2delay_opto_s = []
        epoch1_vis1_on1_opto_s = []
        epoch1_vis2_on2_opto_s = []
        epoch1_on1_targ1_opto_s = []
        epoch1_on2_targ2_opto_s = []
        epoch1_on1_amp1_opto_s = []
        epoch1_on2_amp2_opto_s = []
        epoch1_on_interval_opto_s = []
        epoch1_levret1end_on2_opto_s = []
        epoch1_base_amp1_opto_s = []
        epoch1_base_amp2_opto_s = []
        epoch1_max_vel1_opto_s = []
        epoch1_max_vel2_opto_s = []
        epoch1_velaton1_opto_s = []
        epoch1_velaton2_opto_s = []
        
        epoch1_previs2delay_con_l = []
        epoch1_vis1_on1_con_l = []
        epoch1_vis2_on2_con_l = []
        epoch1_on1_targ1_con_l = []
        epoch1_on2_targ2_con_l = []
        epoch1_on1_amp1_con_l = []
        epoch1_on2_amp2_con_l = []
        epoch1_on_interval_con_l = []
        epoch1_levret1end_on2_con_l = []
        epoch1_base_amp1_con_l = []
        epoch1_base_amp2_con_l = []
        epoch1_max_vel1_con_l = []
        epoch1_max_vel2_con_l = []
        epoch1_velaton1_con_l = []
        epoch1_velaton2_con_l = []
        
        epoch1_previs2delay_chemo_l = []
        epoch1_vis1_on1_chemo_l = []
        epoch1_vis2_on2_chemo_l = []
        epoch1_on1_targ1_chemo_l = []
        epoch1_on2_targ2_chemo_l = []
        epoch1_on1_amp1_chemo_l = []
        epoch1_on2_amp2_chemo_l = []
        epoch1_on_interval_chemo_l = []
        epoch1_levret1end_on2_chemo_l = []
        epoch1_base_amp1_chemo_l = []
        epoch1_base_amp2_chemo_l = []
        epoch1_max_vel1_chemo_l = []
        epoch1_max_vel2_chemo_l = []
        epoch1_velaton1_chemo_l = []
        epoch1_velaton2_chemo_l = []
        
        epoch1_previs2delay_opto_l = []
        epoch1_vis1_on1_opto_l = []
        epoch1_vis2_on2_opto_l = []
        epoch1_on1_targ1_opto_l = []
        epoch1_on2_targ2_opto_l = []
        epoch1_on1_amp1_opto_l = []
        epoch1_on2_amp2_opto_l = []
        epoch1_on_interval_opto_l = []
        epoch1_levret1end_on2_opto_l = []
        epoch1_base_amp1_opto_l = []
        epoch1_base_amp2_opto_l = []
        epoch1_max_vel1_opto_l = []
        epoch1_max_vel2_opto_l = []
        epoch1_velaton1_opto_l = []
        epoch1_velaton2_opto_l = []
        # Epoch2
        epoch2_previs2delay_con_s = []
        epoch2_vis1_on1_con_s = []
        epoch2_vis2_on2_con_s = []
        epoch2_on1_targ1_con_s = []
        epoch2_on2_targ2_con_s = []
        epoch2_on1_amp1_con_s = []
        epoch2_on2_amp2_con_s = []
        epoch2_on_interval_con_s = []
        epoch2_levret1end_on2_con_s = []
        epoch2_base_amp1_con_s = []
        epoch2_base_amp2_con_s = []
        epoch2_max_vel1_con_s = []
        epoch2_max_vel2_con_s = []
        epoch2_velaton1_con_s = []
        epoch2_velaton2_con_s = []
        
        epoch2_previs2delay_chemo_s = []
        epoch2_vis1_on1_chemo_s = []
        epoch2_vis2_on2_chemo_s = []
        epoch2_on1_targ1_chemo_s = []
        epoch2_on2_targ2_chemo_s = []
        epoch2_on1_amp1_chemo_s = []
        epoch2_on2_amp2_chemo_s = []
        epoch2_on_interval_chemo_s = []
        epoch2_levret1end_on2_chemo_s = []
        epoch2_base_amp1_chemo_s = []
        epoch2_base_amp2_chemo_s = []
        epoch2_max_vel1_chemo_s = []
        epoch2_max_vel2_chemo_s = []
        epoch2_velaton1_chemo_s = []
        epoch2_velaton2_chemo_s = []
        
        epoch2_previs2delay_opto_s = []
        epoch2_vis1_on1_opto_s = []
        epoch2_vis2_on2_opto_s = []
        epoch2_on1_targ1_opto_s = []
        epoch2_on2_targ2_opto_s = []
        epoch2_on1_amp1_opto_s = []
        epoch2_on2_amp2_opto_s = []
        epoch2_on_interval_opto_s = []
        epoch2_levret1end_on2_opto_s = []
        epoch2_base_amp1_opto_s = []
        epoch2_base_amp2_opto_s = []
        epoch2_max_vel1_opto_s = []
        epoch2_max_vel2_opto_s = []
        epoch2_velaton1_opto_s = []
        epoch2_velaton2_opto_s = []
        
        epoch2_previs2delay_con_l = []
        epoch2_vis1_on1_con_l = []
        epoch2_vis2_on2_con_l = []
        epoch2_on1_targ1_con_l = []
        epoch2_on2_targ2_con_l = []
        epoch2_on1_amp1_con_l = []
        epoch2_on2_amp2_con_l = []
        epoch2_on_interval_con_l = []
        epoch2_levret1end_on2_con_l = []
        epoch2_base_amp1_con_l = []
        epoch2_base_amp2_con_l = []
        epoch2_max_vel1_con_l = []
        epoch2_max_vel2_con_l = []
        epoch2_velaton1_con_l = []
        epoch2_velaton2_con_l = []
        
        epoch2_previs2delay_chemo_l = []
        epoch2_vis1_on1_chemo_l = []
        epoch2_vis2_on2_chemo_l = []
        epoch2_on1_targ1_chemo_l = []
        epoch2_on2_targ2_chemo_l = []
        epoch2_on1_amp1_chemo_l = []
        epoch2_on2_amp2_chemo_l = []
        epoch2_on_interval_chemo_l = []
        epoch2_levret1end_on2_chemo_l = []
        epoch2_base_amp1_chemo_l = []
        epoch2_base_amp2_chemo_l = []
        epoch2_max_vel1_chemo_l = []
        epoch2_max_vel2_chemo_l = []
        epoch2_velaton1_chemo_l = []
        epoch2_velaton2_chemo_l = []
        
        epoch2_previs2delay_opto_l = []
        epoch2_vis1_on1_opto_l = []
        epoch2_vis2_on2_opto_l = []
        epoch2_on1_targ1_opto_l = []
        epoch2_on2_targ2_opto_l = []
        epoch2_on1_amp1_opto_l = []
        epoch2_on2_amp2_opto_l = []
        epoch2_on_interval_opto_l = []
        epoch2_levret1end_on2_opto_l = []
        epoch2_base_amp1_opto_l = []
        epoch2_base_amp2_opto_l = []
        epoch2_max_vel1_opto_l = []
        epoch2_max_vel2_opto_l = []
        epoch2_velaton1_opto_l = []
        epoch2_velaton2_opto_l = []
        # Epoch3
        epoch3_previs2delay_con_s = []
        epoch3_vis1_on1_con_s = []
        epoch3_vis2_on2_con_s = []
        epoch3_on1_targ1_con_s = []
        epoch3_on2_targ2_con_s = []
        epoch3_on1_amp1_con_s = []
        epoch3_on2_amp2_con_s = []
        epoch3_on_interval_con_s = []
        epoch3_levret1end_on2_con_s = []
        epoch3_base_amp1_con_s = []
        epoch3_base_amp2_con_s = []
        epoch3_max_vel1_con_s = []
        epoch3_max_vel2_con_s = []
        epoch3_velaton1_con_s = []
        epoch3_velaton2_con_s = []
        
        epoch3_previs2delay_chemo_s = []
        epoch3_vis1_on1_chemo_s = []
        epoch3_vis2_on2_chemo_s = []
        epoch3_on1_targ1_chemo_s = []
        epoch3_on2_targ2_chemo_s = []
        epoch3_on1_amp1_chemo_s = []
        epoch3_on2_amp2_chemo_s = []
        epoch3_on_interval_chemo_s = []
        epoch3_levret1end_on2_chemo_s = []
        epoch3_base_amp1_chemo_s = []
        epoch3_base_amp2_chemo_s = []
        epoch3_max_vel1_chemo_s = []
        epoch3_max_vel2_chemo_s = []
        epoch3_velaton1_chemo_s = []
        epoch3_velaton2_chemo_s = []
        
        epoch3_previs2delay_opto_s = []
        epoch3_vis1_on1_opto_s = []
        epoch3_vis2_on2_opto_s = []
        epoch3_on1_targ1_opto_s = []
        epoch3_on2_targ2_opto_s = []
        epoch3_on1_amp1_opto_s = []
        epoch3_on2_amp2_opto_s = []
        epoch3_on_interval_opto_s = []
        epoch3_levret1end_on2_opto_s = []
        epoch3_base_amp1_opto_s = []
        epoch3_base_amp2_opto_s = []
        epoch3_max_vel1_opto_s = []
        epoch3_max_vel2_opto_s = []
        epoch3_velaton1_opto_s = []
        epoch3_velaton2_opto_s = []
        
        epoch3_previs2delay_con_l = []
        epoch3_vis1_on1_con_l = []
        epoch3_vis2_on2_con_l = []
        epoch3_on1_targ1_con_l = []
        epoch3_on2_targ2_con_l = []
        epoch3_on1_amp1_con_l = []
        epoch3_on2_amp2_con_l = []
        epoch3_on_interval_con_l = []
        epoch3_levret1end_on2_con_l = []
        epoch3_base_amp1_con_l = []
        epoch3_base_amp2_con_l = []
        epoch3_max_vel1_con_l = []
        epoch3_max_vel2_con_l = []
        epoch3_velaton1_con_l = []
        epoch3_velaton2_con_l = []
        
        epoch3_previs2delay_chemo_l = []
        epoch3_vis1_on1_chemo_l = []
        epoch3_vis2_on2_chemo_l = []
        epoch3_on1_targ1_chemo_l = []
        epoch3_on2_targ2_chemo_l = []
        epoch3_on1_amp1_chemo_l = []
        epoch3_on2_amp2_chemo_l = []
        epoch3_on_interval_chemo_l = []
        epoch3_levret1end_on2_chemo_l = []
        epoch3_base_amp1_chemo_l = []
        epoch3_base_amp2_chemo_l = []
        epoch3_max_vel1_chemo_l = []
        epoch3_max_vel2_chemo_l = []
        epoch3_velaton1_chemo_l = []
        epoch3_velaton2_chemo_l = []
        
        epoch3_previs2delay_opto_l = []
        epoch3_vis1_on1_opto_l = []
        epoch3_vis2_on2_opto_l = []
        epoch3_on1_targ1_opto_l = []
        epoch3_on2_targ2_opto_l = []
        epoch3_on1_amp1_opto_l = []
        epoch3_on2_amp2_opto_l = []
        epoch3_on_interval_opto_l = []
        epoch3_levret1end_on2_opto_l = []
        epoch3_base_amp1_opto_l = []
        epoch3_base_amp2_opto_l = []
        epoch3_max_vel1_opto_l = []
        epoch3_max_vel2_opto_l = []
        epoch3_velaton1_opto_l = []
        epoch3_velaton2_opto_l = []
        ########################################################################
        # NOTE: we pass the first block
        for trial in range(Block_size ,len(TrialOutcomes)):
            
            if np.isnan(isSelfTimedMode[i][trial]):
                continue
            
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
                # baseline to peak1
                base_amp2 = np.abs(amp2 - base_line)
                # Max velocity2
                if LeverRetract2 >= on2_vel:                 # type: ignore
                    max_vel2 = np.max(velocity[on2_vel:LeverRetract2])
                else:
                    max_vel2 = np.max(velocity[LeverRetract2:on2_vel])
                # Onset2 Velocity
                velaton2 = velocity[on2_vel]
                
                if trial in epoch1:
                    if trial_types[trial] == 1:
                        if chemo_labels[i] == 1:
                            epoch1_previs2delay_chemo_s.append(PreVis2Delay)
                            epoch1_vis1_on1_chemo_s.append(vis1_on1)
                            epoch1_vis2_on2_chemo_s.append(vis2_on2)
                            epoch1_on1_targ1_chemo_s.append(on1_targ1)
                            epoch1_on2_targ2_chemo_s.append(on2_targ2)
                            epoch1_on1_amp1_chemo_s.append(on1_amp1)
                            epoch1_on2_amp2_chemo_s.append(on2_amp2)
                            epoch1_on_interval_chemo_s.append(on_interval)
                            epoch1_levret1end_on2_chemo_s.append(levret1end_on2)
                            epoch1_base_amp1_chemo_s.append(base_amp1)
                            epoch1_base_amp2_chemo_s.append(base_amp2)
                            epoch1_max_vel1_chemo_s.append(max_vel1)
                            epoch1_max_vel2_chemo_s.append(max_vel2)
                            epoch1_velaton1_chemo_s.append(velaton1)
                            epoch1_velaton2_chemo_s.append(velaton2)
                            # POOLED
                            p_epoch1_previs2delay_chemo_s.append(PreVis2Delay)
                            p_epoch1_vis1_on1_chemo_s.append(vis1_on1)
                            p_epoch1_vis2_on2_chemo_s.append(vis2_on2)
                            p_epoch1_on1_targ1_chemo_s.append(on1_targ1)
                            p_epoch1_on2_targ2_chemo_s.append(on2_targ2)
                            p_epoch1_on1_amp1_chemo_s.append(on1_amp1)
                            p_epoch1_on2_amp2_chemo_s.append(on2_amp2)
                            p_epoch1_on_interval_chemo_s.append(on_interval)
                            p_epoch1_levret1end_on2_chemo_s.append(levret1end_on2)
                            p_epoch1_base_amp1_chemo_s.append(base_amp1)
                            p_epoch1_base_amp2_chemo_s.append(base_amp2)
                            p_epoch1_max_vel1_chemo_s.append(max_vel1)
                            p_epoch1_max_vel2_chemo_s.append(max_vel2)
                            p_epoch1_velaton1_chemo_s.append(velaton1)
                            p_epoch1_velaton2_chemo_s.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                epoch1_previs2delay_opto_s.append(PreVis2Delay)
                                epoch1_vis1_on1_opto_s.append(vis1_on1)
                                epoch1_vis2_on2_opto_s.append(vis2_on2)
                                epoch1_on1_targ1_opto_s.append(on1_targ1)
                                epoch1_on2_targ2_opto_s.append(on2_targ2)
                                epoch1_on1_amp1_opto_s.append(on1_amp1)
                                epoch1_on2_amp2_opto_s.append(on2_amp2)
                                epoch1_on_interval_opto_s.append(on_interval)
                                epoch1_levret1end_on2_opto_s.append(levret1end_on2)
                                epoch1_base_amp1_opto_s.append(base_amp1)
                                epoch1_base_amp2_opto_s.append(base_amp2)
                                epoch1_max_vel1_opto_s.append(max_vel1)
                                epoch1_max_vel2_opto_s.append(max_vel2)
                                epoch1_velaton1_opto_s.append(velaton1)
                                epoch1_velaton2_opto_s.append(velaton2)
                                # POOLED
                                p_epoch1_previs2delay_opto_s.append(PreVis2Delay)
                                p_epoch1_vis1_on1_opto_s.append(vis1_on1)
                                p_epoch1_vis2_on2_opto_s.append(vis2_on2)
                                p_epoch1_on1_targ1_opto_s.append(on1_targ1)
                                p_epoch1_on2_targ2_opto_s.append(on2_targ2)
                                p_epoch1_on1_amp1_opto_s.append(on1_amp1)
                                p_epoch1_on2_amp2_opto_s.append(on2_amp2)
                                p_epoch1_on_interval_opto_s.append(on_interval)
                                p_epoch1_levret1end_on2_opto_s.append(levret1end_on2)
                                p_epoch1_base_amp1_opto_s.append(base_amp1)
                                p_epoch1_base_amp2_opto_s.append(base_amp2)
                                p_epoch1_max_vel1_opto_s.append(max_vel1)
                                p_epoch1_max_vel2_opto_s.append(max_vel2)
                                p_epoch1_velaton1_opto_s.append(velaton1)
                                p_epoch1_velaton2_opto_s.append(velaton2)
                            else:
                                epoch1_previs2delay_con_s.append(PreVis2Delay)
                                epoch1_vis1_on1_con_s.append(vis1_on1)
                                epoch1_vis2_on2_con_s.append(vis2_on2)
                                epoch1_on1_targ1_con_s.append(on1_targ1)
                                epoch1_on2_targ2_con_s.append(on2_targ2)
                                epoch1_on1_amp1_con_s.append(on1_amp1)
                                epoch1_on2_amp2_con_s.append(on2_amp2)
                                epoch1_on_interval_con_s.append(on_interval)
                                epoch1_levret1end_on2_con_s.append(levret1end_on2)
                                epoch1_base_amp1_con_s.append(base_amp1)
                                epoch1_base_amp2_con_s.append(base_amp2)
                                epoch1_max_vel1_con_s.append(max_vel1)
                                epoch1_max_vel2_con_s.append(max_vel2)
                                epoch1_velaton1_con_s.append(velaton1)
                                epoch1_velaton2_con_s.append(velaton2)
                                # POOLED
                                p_epoch1_previs2delay_con_s.append(PreVis2Delay)
                                p_epoch1_vis1_on1_con_s.append(vis1_on1)
                                p_epoch1_vis2_on2_con_s.append(vis2_on2)
                                p_epoch1_on1_targ1_con_s.append(on1_targ1)
                                p_epoch1_on2_targ2_con_s.append(on2_targ2)
                                p_epoch1_on1_amp1_con_s.append(on1_amp1)
                                p_epoch1_on2_amp2_con_s.append(on2_amp2)
                                p_epoch1_on_interval_con_s.append(on_interval)
                                p_epoch1_levret1end_on2_con_s.append(levret1end_on2)
                                p_epoch1_base_amp1_con_s.append(base_amp1)
                                p_epoch1_base_amp2_con_s.append(base_amp2)
                                p_epoch1_max_vel1_con_s.append(max_vel1)
                                p_epoch1_max_vel2_con_s.append(max_vel2)
                                p_epoch1_velaton1_con_s.append(velaton1)
                                p_epoch1_velaton2_con_s.append(velaton2)
                    else:
                        if chemo_labels[i] == 1:
                            epoch1_previs2delay_chemo_l.append(PreVis2Delay)
                            epoch1_vis1_on1_chemo_l.append(vis1_on1)
                            epoch1_vis2_on2_chemo_l.append(vis2_on2)
                            epoch1_on1_targ1_chemo_l.append(on1_targ1)
                            epoch1_on2_targ2_chemo_l.append(on2_targ2)
                            epoch1_on1_amp1_chemo_l.append(on1_amp1)
                            epoch1_on2_amp2_chemo_l.append(on2_amp2)
                            epoch1_on_interval_chemo_l.append(on_interval)
                            epoch1_levret1end_on2_chemo_l.append(levret1end_on2)
                            epoch1_base_amp1_chemo_l.append(base_amp1)
                            epoch1_base_amp2_chemo_l.append(base_amp2)
                            epoch1_max_vel1_chemo_l.append(max_vel1)
                            epoch1_max_vel2_chemo_l.append(max_vel2)
                            epoch1_velaton1_chemo_l.append(velaton1)
                            epoch1_velaton2_chemo_l.append(velaton2)
                            # POOLED
                            p_epoch1_previs2delay_chemo_l.append(PreVis2Delay)
                            p_epoch1_vis1_on1_chemo_l.append(vis1_on1)
                            p_epoch1_vis2_on2_chemo_l.append(vis2_on2)
                            p_epoch1_on1_targ1_chemo_l.append(on1_targ1)
                            p_epoch1_on2_targ2_chemo_l.append(on2_targ2)
                            p_epoch1_on1_amp1_chemo_l.append(on1_amp1)
                            p_epoch1_on2_amp2_chemo_l.append(on2_amp2)
                            p_epoch1_on_interval_chemo_l.append(on_interval)
                            p_epoch1_levret1end_on2_chemo_l.append(levret1end_on2)
                            p_epoch1_base_amp1_chemo_l.append(base_amp1)
                            p_epoch1_base_amp2_chemo_l.append(base_amp2)
                            p_epoch1_max_vel1_chemo_l.append(max_vel1)
                            p_epoch1_max_vel2_chemo_l.append(max_vel2)
                            p_epoch1_velaton1_chemo_l.append(velaton1)
                            p_epoch1_velaton2_chemo_l.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                epoch1_previs2delay_opto_l.append(PreVis2Delay)
                                epoch1_vis1_on1_opto_l.append(vis1_on1)
                                epoch1_vis2_on2_opto_l.append(vis2_on2)
                                epoch1_on1_targ1_opto_l.append(on1_targ1)
                                epoch1_on2_targ2_opto_l.append(on2_targ2)
                                epoch1_on1_amp1_opto_l.append(on1_amp1)
                                epoch1_on2_amp2_opto_l.append(on2_amp2)
                                epoch1_on_interval_opto_l.append(on_interval)
                                epoch1_levret1end_on2_opto_l.append(levret1end_on2)
                                epoch1_base_amp1_opto_l.append(base_amp1)
                                epoch1_base_amp2_opto_l.append(base_amp2)
                                epoch1_max_vel1_opto_l.append(max_vel1)
                                epoch1_max_vel2_opto_l.append(max_vel2)
                                epoch1_velaton1_opto_l.append(velaton1)
                                epoch1_velaton2_opto_l.append(velaton2)
                                # POOLED
                                p_epoch1_previs2delay_opto_l.append(PreVis2Delay)
                                p_epoch1_vis1_on1_opto_l.append(vis1_on1)
                                p_epoch1_vis2_on2_opto_l.append(vis2_on2)
                                p_epoch1_on1_targ1_opto_l.append(on1_targ1)
                                p_epoch1_on2_targ2_opto_l.append(on2_targ2)
                                p_epoch1_on1_amp1_opto_l.append(on1_amp1)
                                p_epoch1_on2_amp2_opto_l.append(on2_amp2)
                                p_epoch1_on_interval_opto_l.append(on_interval)
                                p_epoch1_levret1end_on2_opto_l.append(levret1end_on2)
                                p_epoch1_base_amp1_opto_l.append(base_amp1)
                                p_epoch1_base_amp2_opto_l.append(base_amp2)
                                p_epoch1_max_vel1_opto_l.append(max_vel1)
                                p_epoch1_max_vel2_opto_l.append(max_vel2)
                                p_epoch1_velaton1_opto_l.append(velaton1)
                                p_epoch1_velaton2_opto_l.append(velaton2)
                            else:
                                epoch1_previs2delay_con_l.append(PreVis2Delay)
                                epoch1_vis1_on1_con_l.append(vis1_on1)
                                epoch1_vis2_on2_con_l.append(vis2_on2)
                                epoch1_on1_targ1_con_l.append(on1_targ1)
                                epoch1_on2_targ2_con_l.append(on2_targ2)
                                epoch1_on1_amp1_con_l.append(on1_amp1)
                                epoch1_on2_amp2_con_l.append(on2_amp2)
                                epoch1_on_interval_con_l.append(on_interval)
                                epoch1_levret1end_on2_con_l.append(levret1end_on2)
                                epoch1_base_amp1_con_l.append(base_amp1)
                                epoch1_base_amp2_con_l.append(base_amp2)
                                epoch1_max_vel1_con_l.append(max_vel1)
                                epoch1_max_vel2_con_l.append(max_vel2)
                                epoch1_velaton1_con_l.append(velaton1)
                                epoch1_velaton2_con_l.append(velaton2)
                                # POOLED
                                p_epoch1_previs2delay_con_l.append(PreVis2Delay)
                                p_epoch1_vis1_on1_con_l.append(vis1_on1)
                                p_epoch1_vis2_on2_con_l.append(vis2_on2)
                                p_epoch1_on1_targ1_con_l.append(on1_targ1)
                                p_epoch1_on2_targ2_con_l.append(on2_targ2)
                                p_epoch1_on1_amp1_con_l.append(on1_amp1)
                                p_epoch1_on2_amp2_con_l.append(on2_amp2)
                                p_epoch1_on_interval_con_l.append(on_interval)
                                p_epoch1_levret1end_on2_con_l.append(levret1end_on2)
                                p_epoch1_base_amp1_con_l.append(base_amp1)
                                p_epoch1_base_amp2_con_l.append(base_amp2)
                                p_epoch1_max_vel1_con_l.append(max_vel1)
                                p_epoch1_max_vel2_con_l.append(max_vel2)
                                p_epoch1_velaton1_con_l.append(velaton1)
                                p_epoch1_velaton2_con_l.append(velaton2)
                elif trial in epoch2:
                    if trial_types[trial] == 1:
                        if chemo_labels[i] == 1:
                            epoch2_previs2delay_chemo_s.append(PreVis2Delay)
                            epoch2_vis1_on1_chemo_s.append(vis1_on1)
                            epoch2_vis2_on2_chemo_s.append(vis2_on2)
                            epoch2_on1_targ1_chemo_s.append(on1_targ1)
                            epoch2_on2_targ2_chemo_s.append(on2_targ2)
                            epoch2_on1_amp1_chemo_s.append(on1_amp1)
                            epoch2_on2_amp2_chemo_s.append(on2_amp2)
                            epoch2_on_interval_chemo_s.append(on_interval)
                            epoch2_levret1end_on2_chemo_s.append(levret1end_on2)
                            epoch2_base_amp1_chemo_s.append(base_amp1)
                            epoch2_base_amp2_chemo_s.append(base_amp2)
                            epoch2_max_vel1_chemo_s.append(max_vel1)
                            epoch2_max_vel2_chemo_s.append(max_vel2)
                            epoch2_velaton1_chemo_s.append(velaton1)
                            epoch2_velaton2_chemo_s.append(velaton2)
                            # POOLED
                            p_epoch2_previs2delay_chemo_s.append(PreVis2Delay)
                            p_epoch2_vis1_on1_chemo_s.append(vis1_on1)
                            p_epoch2_vis2_on2_chemo_s.append(vis2_on2)
                            p_epoch2_on1_targ1_chemo_s.append(on1_targ1)
                            p_epoch2_on2_targ2_chemo_s.append(on2_targ2)
                            p_epoch2_on1_amp1_chemo_s.append(on1_amp1)
                            p_epoch2_on2_amp2_chemo_s.append(on2_amp2)
                            p_epoch2_on_interval_chemo_s.append(on_interval)
                            p_epoch2_levret1end_on2_chemo_s.append(levret1end_on2)
                            p_epoch2_base_amp1_chemo_s.append(base_amp1)
                            p_epoch2_base_amp2_chemo_s.append(base_amp2)
                            p_epoch2_max_vel1_chemo_s.append(max_vel1)
                            p_epoch2_max_vel2_chemo_s.append(max_vel2)
                            p_epoch2_velaton1_chemo_s.append(velaton1)
                            p_epoch2_velaton2_chemo_s.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                epoch2_previs2delay_opto_s.append(PreVis2Delay)
                                epoch2_vis1_on1_opto_s.append(vis1_on1)
                                epoch2_vis2_on2_opto_s.append(vis2_on2)
                                epoch2_on1_targ1_opto_s.append(on1_targ1)
                                epoch2_on2_targ2_opto_s.append(on2_targ2)
                                epoch2_on1_amp1_opto_s.append(on1_amp1)
                                epoch2_on2_amp2_opto_s.append(on2_amp2)
                                epoch2_on_interval_opto_s.append(on_interval)
                                epoch2_levret1end_on2_opto_s.append(levret1end_on2)
                                epoch2_base_amp1_opto_s.append(base_amp1)
                                epoch2_base_amp2_opto_s.append(base_amp2)
                                epoch2_max_vel1_opto_s.append(max_vel1)
                                epoch2_max_vel2_opto_s.append(max_vel2)
                                epoch2_velaton1_opto_s.append(velaton1)
                                epoch2_velaton2_opto_s.append(velaton2)
                                # POOLED
                                p_epoch2_previs2delay_opto_s.append(PreVis2Delay)
                                p_epoch2_vis1_on1_opto_s.append(vis1_on1)
                                p_epoch2_vis2_on2_opto_s.append(vis2_on2)
                                p_epoch2_on1_targ1_opto_s.append(on1_targ1)
                                p_epoch2_on2_targ2_opto_s.append(on2_targ2)
                                p_epoch2_on1_amp1_opto_s.append(on1_amp1)
                                p_epoch2_on2_amp2_opto_s.append(on2_amp2)
                                p_epoch2_on_interval_opto_s.append(on_interval)
                                p_epoch2_levret1end_on2_opto_s.append(levret1end_on2)
                                p_epoch2_base_amp1_opto_s.append(base_amp1)
                                p_epoch2_base_amp2_opto_s.append(base_amp2)
                                p_epoch2_max_vel1_opto_s.append(max_vel1)
                                p_epoch2_max_vel2_opto_s.append(max_vel2)
                                p_epoch2_velaton1_opto_s.append(velaton1)
                                p_epoch2_velaton2_opto_s.append(velaton2)
                            else:
                                epoch2_previs2delay_con_s.append(PreVis2Delay)
                                epoch2_vis1_on1_con_s.append(vis1_on1)
                                epoch2_vis2_on2_con_s.append(vis2_on2)
                                epoch2_on1_targ1_con_s.append(on1_targ1)
                                epoch2_on2_targ2_con_s.append(on2_targ2)
                                epoch2_on1_amp1_con_s.append(on1_amp1)
                                epoch2_on2_amp2_con_s.append(on2_amp2)
                                epoch2_on_interval_con_s.append(on_interval)
                                epoch2_levret1end_on2_con_s.append(levret1end_on2)
                                epoch2_base_amp1_con_s.append(base_amp1)
                                epoch2_base_amp2_con_s.append(base_amp2)
                                epoch2_max_vel1_con_s.append(max_vel1)
                                epoch2_max_vel2_con_s.append(max_vel2)
                                epoch2_velaton1_con_s.append(velaton1)
                                epoch2_velaton2_con_s.append(velaton2)
                                # POOLED
                                p_epoch2_previs2delay_con_s.append(PreVis2Delay)
                                p_epoch2_vis1_on1_con_s.append(vis1_on1)
                                p_epoch2_vis2_on2_con_s.append(vis2_on2)
                                p_epoch2_on1_targ1_con_s.append(on1_targ1)
                                p_epoch2_on2_targ2_con_s.append(on2_targ2)
                                p_epoch2_on1_amp1_con_s.append(on1_amp1)
                                p_epoch2_on2_amp2_con_s.append(on2_amp2)
                                p_epoch2_on_interval_con_s.append(on_interval)
                                p_epoch2_levret1end_on2_con_s.append(levret1end_on2)
                                p_epoch2_base_amp1_con_s.append(base_amp1)
                                p_epoch2_base_amp2_con_s.append(base_amp2)
                                p_epoch2_max_vel1_con_s.append(max_vel1)
                                p_epoch2_max_vel2_con_s.append(max_vel2)
                                p_epoch2_velaton1_con_s.append(velaton1)
                                p_epoch2_velaton2_con_s.append(velaton2)
                    else:
                        if chemo_labels[i] == 1:
                            epoch2_previs2delay_chemo_l.append(PreVis2Delay)
                            epoch2_vis1_on1_chemo_l.append(vis1_on1)
                            epoch2_vis2_on2_chemo_l.append(vis2_on2)
                            epoch2_on1_targ1_chemo_l.append(on1_targ1)
                            epoch2_on2_targ2_chemo_l.append(on2_targ2)
                            epoch2_on1_amp1_chemo_l.append(on1_amp1)
                            epoch2_on2_amp2_chemo_l.append(on2_amp2)
                            epoch2_on_interval_chemo_l.append(on_interval)
                            epoch2_levret1end_on2_chemo_l.append(levret1end_on2)
                            epoch2_base_amp1_chemo_l.append(base_amp1)
                            epoch2_base_amp2_chemo_l.append(base_amp2)
                            epoch2_max_vel1_chemo_l.append(max_vel1)
                            epoch2_max_vel2_chemo_l.append(max_vel2)
                            epoch2_velaton1_chemo_l.append(velaton1)
                            epoch2_velaton2_chemo_l.append(velaton2)
                            # POOLED
                            p_epoch2_previs2delay_chemo_l.append(PreVis2Delay)
                            p_epoch2_vis1_on1_chemo_l.append(vis1_on1)
                            p_epoch2_vis2_on2_chemo_l.append(vis2_on2)
                            p_epoch2_on1_targ1_chemo_l.append(on1_targ1)
                            p_epoch2_on2_targ2_chemo_l.append(on2_targ2)
                            p_epoch2_on1_amp1_chemo_l.append(on1_amp1)
                            p_epoch2_on2_amp2_chemo_l.append(on2_amp2)
                            p_epoch2_on_interval_chemo_l.append(on_interval)
                            p_epoch2_levret1end_on2_chemo_l.append(levret1end_on2)
                            p_epoch2_base_amp1_chemo_l.append(base_amp1)
                            p_epoch2_base_amp2_chemo_l.append(base_amp2)
                            p_epoch2_max_vel1_chemo_l.append(max_vel1)
                            p_epoch2_max_vel2_chemo_l.append(max_vel2)
                            p_epoch2_velaton1_chemo_l.append(velaton1)
                            p_epoch2_velaton2_chemo_l.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                epoch2_previs2delay_opto_l.append(PreVis2Delay)
                                epoch2_vis1_on1_opto_l.append(vis1_on1)
                                epoch2_vis2_on2_opto_l.append(vis2_on2)
                                epoch2_on1_targ1_opto_l.append(on1_targ1)
                                epoch2_on2_targ2_opto_l.append(on2_targ2)
                                epoch2_on1_amp1_opto_l.append(on1_amp1)
                                epoch2_on2_amp2_opto_l.append(on2_amp2)
                                epoch2_on_interval_opto_l.append(on_interval)
                                epoch2_levret1end_on2_opto_l.append(levret1end_on2)
                                epoch2_base_amp1_opto_l.append(base_amp1)
                                epoch2_base_amp2_opto_l.append(base_amp2)
                                epoch2_max_vel1_opto_l.append(max_vel1)
                                epoch2_max_vel2_opto_l.append(max_vel2)
                                epoch2_velaton1_opto_l.append(velaton1)
                                epoch2_velaton2_opto_l.append(velaton2)
                                # POOLED
                                p_epoch2_previs2delay_opto_l.append(PreVis2Delay)
                                p_epoch2_vis1_on1_opto_l.append(vis1_on1)
                                p_epoch2_vis2_on2_opto_l.append(vis2_on2)
                                p_epoch2_on1_targ1_opto_l.append(on1_targ1)
                                p_epoch2_on2_targ2_opto_l.append(on2_targ2)
                                p_epoch2_on1_amp1_opto_l.append(on1_amp1)
                                p_epoch2_on2_amp2_opto_l.append(on2_amp2)
                                p_epoch2_on_interval_opto_l.append(on_interval)
                                p_epoch2_levret1end_on2_opto_l.append(levret1end_on2)
                                p_epoch2_base_amp1_opto_l.append(base_amp1)
                                p_epoch2_base_amp2_opto_l.append(base_amp2)
                                p_epoch2_max_vel1_opto_l.append(max_vel1)
                                p_epoch2_max_vel2_opto_l.append(max_vel2)
                                p_epoch2_velaton1_opto_l.append(velaton1)
                                p_epoch2_velaton2_opto_l.append(velaton2)
                            else:
                                epoch2_previs2delay_con_l.append(PreVis2Delay)
                                epoch2_vis1_on1_con_l.append(vis1_on1)
                                epoch2_vis2_on2_con_l.append(vis2_on2)
                                epoch2_on1_targ1_con_l.append(on1_targ1)
                                epoch2_on2_targ2_con_l.append(on2_targ2)
                                epoch2_on1_amp1_con_l.append(on1_amp1)
                                epoch2_on2_amp2_con_l.append(on2_amp2)
                                epoch2_on_interval_con_l.append(on_interval)
                                epoch2_levret1end_on2_con_l.append(levret1end_on2)
                                epoch2_base_amp1_con_l.append(base_amp1)
                                epoch2_base_amp2_con_l.append(base_amp2)
                                epoch2_max_vel1_con_l.append(max_vel1)
                                epoch2_max_vel2_con_l.append(max_vel2)
                                epoch2_velaton1_con_l.append(velaton1)
                                epoch2_velaton2_con_l.append(velaton2)
                                # POOLED
                                p_epoch2_previs2delay_con_l.append(PreVis2Delay)
                                p_epoch2_vis1_on1_con_l.append(vis1_on1)
                                p_epoch2_vis2_on2_con_l.append(vis2_on2)
                                p_epoch2_on1_targ1_con_l.append(on1_targ1)
                                p_epoch2_on2_targ2_con_l.append(on2_targ2)
                                p_epoch2_on1_amp1_con_l.append(on1_amp1)
                                p_epoch2_on2_amp2_con_l.append(on2_amp2)
                                p_epoch2_on_interval_con_l.append(on_interval)
                                p_epoch2_levret1end_on2_con_l.append(levret1end_on2)
                                p_epoch2_base_amp1_con_l.append(base_amp1)
                                p_epoch2_base_amp2_con_l.append(base_amp2)
                                p_epoch2_max_vel1_con_l.append(max_vel1)
                                p_epoch2_max_vel2_con_l.append(max_vel2)
                                p_epoch2_velaton1_con_l.append(velaton1)
                                p_epoch2_velaton2_con_l.append(velaton2)
                elif trial in epoch3:
                    if trial_types[trial] == 1:
                        if chemo_labels[i] == 1:
                            epoch3_previs2delay_chemo_s.append(PreVis2Delay)
                            epoch3_vis1_on1_chemo_s.append(vis1_on1)
                            epoch3_vis2_on2_chemo_s.append(vis2_on2)
                            epoch3_on1_targ1_chemo_s.append(on1_targ1)
                            epoch3_on2_targ2_chemo_s.append(on2_targ2)
                            epoch3_on1_amp1_chemo_s.append(on1_amp1)
                            epoch3_on2_amp2_chemo_s.append(on2_amp2)
                            epoch3_on_interval_chemo_s.append(on_interval)
                            epoch3_levret1end_on2_chemo_s.append(levret1end_on2)
                            epoch3_base_amp1_chemo_s.append(base_amp1)
                            epoch3_base_amp2_chemo_s.append(base_amp2)
                            epoch3_max_vel1_chemo_s.append(max_vel1)
                            epoch3_max_vel2_chemo_s.append(max_vel2)
                            epoch3_velaton1_chemo_s.append(velaton1)
                            epoch3_velaton2_chemo_s.append(velaton2)
                            # POOLED
                            p_epoch3_previs2delay_chemo_s.append(PreVis2Delay)
                            p_epoch3_vis1_on1_chemo_s.append(vis1_on1)
                            p_epoch3_vis2_on2_chemo_s.append(vis2_on2)
                            p_epoch3_on1_targ1_chemo_s.append(on1_targ1)
                            p_epoch3_on2_targ2_chemo_s.append(on2_targ2)
                            p_epoch3_on1_amp1_chemo_s.append(on1_amp1)
                            p_epoch3_on2_amp2_chemo_s.append(on2_amp2)
                            p_epoch3_on_interval_chemo_s.append(on_interval)
                            p_epoch3_levret1end_on2_chemo_s.append(levret1end_on2)
                            p_epoch3_base_amp1_chemo_s.append(base_amp1)
                            p_epoch3_base_amp2_chemo_s.append(base_amp2)
                            p_epoch3_max_vel1_chemo_s.append(max_vel1)
                            p_epoch3_max_vel2_chemo_s.append(max_vel2)
                            p_epoch3_velaton1_chemo_s.append(velaton1)
                            p_epoch3_velaton2_chemo_s.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                epoch3_previs2delay_opto_s.append(PreVis2Delay)
                                epoch3_vis1_on1_opto_s.append(vis1_on1)
                                epoch3_vis2_on2_opto_s.append(vis2_on2)
                                epoch3_on1_targ1_opto_s.append(on1_targ1)
                                epoch3_on2_targ2_opto_s.append(on2_targ2)
                                epoch3_on1_amp1_opto_s.append(on1_amp1)
                                epoch3_on2_amp2_opto_s.append(on2_amp2)
                                epoch3_on_interval_opto_s.append(on_interval)
                                epoch3_levret1end_on2_opto_s.append(levret1end_on2)
                                epoch3_base_amp1_opto_s.append(base_amp1)
                                epoch3_base_amp2_opto_s.append(base_amp2)
                                epoch3_max_vel1_opto_s.append(max_vel1)
                                epoch3_max_vel2_opto_s.append(max_vel2)
                                epoch3_velaton1_opto_s.append(velaton1)
                                epoch3_velaton2_opto_s.append(velaton2)
                                # POOLED
                                p_epoch3_previs2delay_opto_s.append(PreVis2Delay)
                                p_epoch3_vis1_on1_opto_s.append(vis1_on1)
                                p_epoch3_vis2_on2_opto_s.append(vis2_on2)
                                p_epoch3_on1_targ1_opto_s.append(on1_targ1)
                                p_epoch3_on2_targ2_opto_s.append(on2_targ2)
                                p_epoch3_on1_amp1_opto_s.append(on1_amp1)
                                p_epoch3_on2_amp2_opto_s.append(on2_amp2)
                                p_epoch3_on_interval_opto_s.append(on_interval)
                                p_epoch3_levret1end_on2_opto_s.append(levret1end_on2)
                                p_epoch3_base_amp1_opto_s.append(base_amp1)
                                p_epoch3_base_amp2_opto_s.append(base_amp2)
                                p_epoch3_max_vel1_opto_s.append(max_vel1)
                                p_epoch3_max_vel2_opto_s.append(max_vel2)
                                p_epoch3_velaton1_opto_s.append(velaton1)
                                p_epoch3_velaton2_opto_s.append(velaton2)
                            else:
                                epoch3_previs2delay_con_s.append(PreVis2Delay)
                                epoch3_vis1_on1_con_s.append(vis1_on1)
                                epoch3_vis2_on2_con_s.append(vis2_on2)
                                epoch3_on1_targ1_con_s.append(on1_targ1)
                                epoch3_on2_targ2_con_s.append(on2_targ2)
                                epoch3_on1_amp1_con_s.append(on1_amp1)
                                epoch3_on2_amp2_con_s.append(on2_amp2)
                                epoch3_on_interval_con_s.append(on_interval)
                                epoch3_levret1end_on2_con_s.append(levret1end_on2)
                                epoch3_base_amp1_con_s.append(base_amp1)
                                epoch3_base_amp2_con_s.append(base_amp2)
                                epoch3_max_vel1_con_s.append(max_vel1)
                                epoch3_max_vel2_con_s.append(max_vel2)
                                epoch3_velaton1_con_s.append(velaton1)
                                epoch3_velaton2_con_s.append(velaton2)
                                # POOLED
                                p_epoch3_previs2delay_con_s.append(PreVis2Delay)
                                p_epoch3_vis1_on1_con_s.append(vis1_on1)
                                p_epoch3_vis2_on2_con_s.append(vis2_on2)
                                p_epoch3_on1_targ1_con_s.append(on1_targ1)
                                p_epoch3_on2_targ2_con_s.append(on2_targ2)
                                p_epoch3_on1_amp1_con_s.append(on1_amp1)
                                p_epoch3_on2_amp2_con_s.append(on2_amp2)
                                p_epoch3_on_interval_con_s.append(on_interval)
                                p_epoch3_levret1end_on2_con_s.append(levret1end_on2)
                                p_epoch3_base_amp1_con_s.append(base_amp1)
                                p_epoch3_base_amp2_con_s.append(base_amp2)
                                p_epoch3_max_vel1_con_s.append(max_vel1)
                                p_epoch3_max_vel2_con_s.append(max_vel2)
                                p_epoch3_velaton1_con_s.append(velaton1)
                                p_epoch3_velaton2_con_s.append(velaton2)
                    else:
                        if chemo_labels[i] == 1:
                            epoch3_previs2delay_chemo_l.append(PreVis2Delay)
                            epoch3_vis1_on1_chemo_l.append(vis1_on1)
                            epoch3_vis2_on2_chemo_l.append(vis2_on2)
                            epoch3_on1_targ1_chemo_l.append(on1_targ1)
                            epoch3_on2_targ2_chemo_l.append(on2_targ2)
                            epoch3_on1_amp1_chemo_l.append(on1_amp1)
                            epoch3_on2_amp2_chemo_l.append(on2_amp2)
                            epoch3_on_interval_chemo_l.append(on_interval)
                            epoch3_levret1end_on2_chemo_l.append(levret1end_on2)
                            epoch3_base_amp1_chemo_l.append(base_amp1)
                            epoch3_base_amp2_chemo_l.append(base_amp2)
                            epoch3_max_vel1_chemo_l.append(max_vel1)
                            epoch3_max_vel2_chemo_l.append(max_vel2)
                            epoch3_velaton1_chemo_l.append(velaton1)
                            epoch3_velaton2_chemo_l.append(velaton2)
                            # POOLED
                            p_epoch3_previs2delay_chemo_l.append(PreVis2Delay)
                            p_epoch3_vis1_on1_chemo_l.append(vis1_on1)
                            p_epoch3_vis2_on2_chemo_l.append(vis2_on2)
                            p_epoch3_on1_targ1_chemo_l.append(on1_targ1)
                            p_epoch3_on2_targ2_chemo_l.append(on2_targ2)
                            p_epoch3_on1_amp1_chemo_l.append(on1_amp1)
                            p_epoch3_on2_amp2_chemo_l.append(on2_amp2)
                            p_epoch3_on_interval_chemo_l.append(on_interval)
                            p_epoch3_levret1end_on2_chemo_l.append(levret1end_on2)
                            p_epoch3_base_amp1_chemo_l.append(base_amp1)
                            p_epoch3_base_amp2_chemo_l.append(base_amp2)
                            p_epoch3_max_vel1_chemo_l.append(max_vel1)
                            p_epoch3_max_vel2_chemo_l.append(max_vel2)
                            p_epoch3_velaton1_chemo_l.append(velaton1)
                            p_epoch3_velaton2_chemo_l.append(velaton2)
                        else:
                            if opto[trial] == 1:
                                epoch3_previs2delay_opto_l.append(PreVis2Delay)
                                epoch3_vis1_on1_opto_l.append(vis1_on1)
                                epoch3_vis2_on2_opto_l.append(vis2_on2)
                                epoch3_on1_targ1_opto_l.append(on1_targ1)
                                epoch3_on2_targ2_opto_l.append(on2_targ2)
                                epoch3_on1_amp1_opto_l.append(on1_amp1)
                                epoch3_on2_amp2_opto_l.append(on2_amp2)
                                epoch3_on_interval_opto_l.append(on_interval)
                                epoch3_levret1end_on2_opto_l.append(levret1end_on2)
                                epoch3_base_amp1_opto_l.append(base_amp1)
                                epoch3_base_amp2_opto_l.append(base_amp2)
                                epoch3_max_vel1_opto_l.append(max_vel1)
                                epoch3_max_vel2_opto_l.append(max_vel2)
                                epoch3_velaton1_opto_l.append(velaton1)
                                epoch3_velaton2_opto_l.append(velaton2)
                                # POOLED
                                p_epoch3_previs2delay_opto_l.append(PreVis2Delay)
                                p_epoch3_vis1_on1_opto_l.append(vis1_on1)
                                p_epoch3_vis2_on2_opto_l.append(vis2_on2)
                                p_epoch3_on1_targ1_opto_l.append(on1_targ1)
                                p_epoch3_on2_targ2_opto_l.append(on2_targ2)
                                p_epoch3_on1_amp1_opto_l.append(on1_amp1)
                                p_epoch3_on2_amp2_opto_l.append(on2_amp2)
                                p_epoch3_on_interval_opto_l.append(on_interval)
                                p_epoch3_levret1end_on2_opto_l.append(levret1end_on2)
                                p_epoch3_base_amp1_opto_l.append(base_amp1)
                                p_epoch3_base_amp2_opto_l.append(base_amp2)
                                p_epoch3_max_vel1_opto_l.append(max_vel1)
                                p_epoch3_max_vel2_opto_l.append(max_vel2)
                                p_epoch3_velaton1_opto_l.append(velaton1)
                                p_epoch3_velaton2_opto_l.append(velaton2)
                            else:
                                epoch3_previs2delay_con_l.append(PreVis2Delay)
                                epoch3_vis1_on1_con_l.append(vis1_on1)
                                epoch3_vis2_on2_con_l.append(vis2_on2)
                                epoch3_on1_targ1_con_l.append(on1_targ1)
                                epoch3_on2_targ2_con_l.append(on2_targ2)
                                epoch3_on1_amp1_con_l.append(on1_amp1)
                                epoch3_on2_amp2_con_l.append(on2_amp2)
                                epoch3_on_interval_con_l.append(on_interval)
                                epoch3_levret1end_on2_con_l.append(levret1end_on2)
                                epoch3_base_amp1_con_l.append(base_amp1)
                                epoch3_base_amp2_con_l.append(base_amp2)
                                epoch3_max_vel1_con_l.append(max_vel1)
                                epoch3_max_vel2_con_l.append(max_vel2)
                                epoch3_velaton1_con_l.append(velaton1)
                                epoch3_velaton2_con_l.append(velaton2)
                                # POOLED
                                p_epoch3_previs2delay_con_l.append(PreVis2Delay)
                                p_epoch3_vis1_on1_con_l.append(vis1_on1)
                                p_epoch3_vis2_on2_con_l.append(vis2_on2)
                                p_epoch3_on1_targ1_con_l.append(on1_targ1)
                                p_epoch3_on2_targ2_con_l.append(on2_targ2)
                                p_epoch3_on1_amp1_con_l.append(on1_amp1)
                                p_epoch3_on2_amp2_con_l.append(on2_amp2)
                                p_epoch3_on_interval_con_l.append(on_interval)
                                p_epoch3_levret1end_on2_con_l.append(levret1end_on2)
                                p_epoch3_base_amp1_con_l.append(base_amp1)
                                p_epoch3_base_amp2_con_l.append(base_amp2)
                                p_epoch3_max_vel1_con_l.append(max_vel1)
                                p_epoch3_max_vel2_con_l.append(max_vel2)
                                p_epoch3_velaton1_con_l.append(velaton1)
                                p_epoch3_velaton2_con_l.append(velaton2)
        # [0,0]
        if len(epoch1_previs2delay_con_s) > 0 :
            G_epoch1_previs2delay_con_s.append(np.mean(epoch1_previs2delay_con_s, axis=0))                    
            axs1[0,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_previs2delay_con_s , axis=0), yerr=stats.sem(epoch1_previs2delay_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_previs2delay_con_s) > 0 :
            G_epoch2_previs2delay_con_s.append(np.mean(epoch2_previs2delay_con_s, axis=0))                    
            axs1[0,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_previs2delay_con_s , axis=0), yerr=stats.sem(epoch2_previs2delay_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_previs2delay_con_s) > 0 :
            G_epoch3_previs2delay_con_s.append(np.mean(epoch3_previs2delay_con_s, axis=0))                    
            axs1[0,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_previs2delay_con_s , axis=0), yerr=stats.sem(epoch3_previs2delay_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_previs2delay_chemo_s) > 0 :
            G_epoch1_previs2delay_chemo_s.append(np.mean(epoch1_previs2delay_chemo_s, axis=0))                    
            axs1[0,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_previs2delay_chemo_s , axis=0), yerr=stats.sem(epoch1_previs2delay_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_previs2delay_chemo_s) > 0 :
            G_epoch2_previs2delay_chemo_s.append(np.mean(epoch2_previs2delay_chemo_s, axis=0))                    
            axs1[0,0].errorbar(i + 1 , np.mean(epoch2_previs2delay_chemo_s , axis=0), yerr=stats.sem(epoch2_previs2delay_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_previs2delay_chemo_s) > 0 :
            G_epoch3_previs2delay_chemo_s.append(np.mean(epoch3_previs2delay_chemo_s, axis=0))                    
            axs1[0,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_previs2delay_chemo_s , axis=0), yerr=stats.sem(epoch3_previs2delay_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_previs2delay_opto_s) > 0 :
            G_epoch1_previs2delay_opto_s.append(np.mean(epoch1_previs2delay_opto_s, axis=0))                    
            axs1[0,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_previs2delay_opto_s , axis=0), yerr=stats.sem(epoch1_previs2delay_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_previs2delay_opto_s) > 0 :
            G_epoch2_previs2delay_opto_s.append(np.mean(epoch2_previs2delay_opto_s, axis=0))                    
            axs1[0,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_previs2delay_opto_s , axis=0), yerr=stats.sem(epoch2_previs2delay_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_previs2delay_opto_s) > 0 :
            G_epoch3_previs2delay_opto_s.append(np.mean(epoch3_previs2delay_opto_s, axis=0))                    
            axs1[0,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_previs2delay_opto_s , axis=0), yerr=stats.sem(epoch3_previs2delay_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        
        # [1,0]
        if len(epoch1_previs2delay_con_l) > 0 :
            G_epoch1_previs2delay_con_l.append(np.mean(epoch1_previs2delay_con_l, axis=0))                    
            axs1[1,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_previs2delay_con_l , axis=0), yerr=stats.sem(epoch1_previs2delay_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_previs2delay_con_l) > 0 :
            G_epoch2_previs2delay_con_l.append(np.mean(epoch2_previs2delay_con_l, axis=0))                    
            axs1[1,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_previs2delay_con_l , axis=0), yerr=stats.sem(epoch2_previs2delay_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_previs2delay_con_l) > 0 :
            G_epoch3_previs2delay_con_l.append(np.mean(epoch3_previs2delay_con_l, axis=0))                    
            axs1[1,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_previs2delay_con_l , axis=0), yerr=stats.sem(epoch3_previs2delay_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_previs2delay_chemo_l) > 0 :
            G_epoch1_previs2delay_chemo_l.append(np.mean(epoch1_previs2delay_chemo_l, axis=0))                    
            axs1[1,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_previs2delay_chemo_l , axis=0), yerr=stats.sem(epoch1_previs2delay_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_previs2delay_chemo_l) > 0 :
            G_epoch2_previs2delay_chemo_l.append(np.mean(epoch2_previs2delay_chemo_l, axis=0))                    
            axs1[1,0].errorbar(i + 1 , np.mean(epoch2_previs2delay_chemo_l , axis=0), yerr=stats.sem(epoch2_previs2delay_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_previs2delay_chemo_l) > 0 :
            G_epoch3_previs2delay_chemo_l.append(np.mean(epoch3_previs2delay_chemo_l, axis=0))                    
            axs1[1,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_previs2delay_chemo_l , axis=0), yerr=stats.sem(epoch3_previs2delay_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_previs2delay_opto_l) > 0 :
            G_epoch1_previs2delay_opto_l.append(np.mean(epoch1_previs2delay_opto_l, axis=0))                    
            axs1[1,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_previs2delay_opto_l , axis=0), yerr=stats.sem(epoch1_previs2delay_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_previs2delay_opto_l) > 0 :
            G_epoch2_previs2delay_opto_l.append(np.mean(epoch2_previs2delay_opto_l, axis=0))                    
            axs1[1,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_previs2delay_opto_l , axis=0), yerr=stats.sem(epoch2_previs2delay_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_previs2delay_opto_l) > 0 :
            G_epoch3_previs2delay_opto_l.append(np.mean(epoch3_previs2delay_opto_l, axis=0))                    
            axs1[1,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_previs2delay_opto_l , axis=0), yerr=stats.sem(epoch3_previs2delay_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        # [2,0]
        if len(epoch1_vis1_on1_con_s) > 0 :
            G_epoch1_vis1_on1_con_s.append(np.mean(epoch1_vis1_on1_con_s, axis=0))                    
            axs1[2,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_vis1_on1_con_s , axis=0), yerr=stats.sem(epoch1_vis1_on1_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_vis1_on1_con_s) > 0 :
            G_epoch2_vis1_on1_con_s.append(np.mean(epoch2_vis1_on1_con_s, axis=0))                    
            axs1[2,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_vis1_on1_con_s , axis=0), yerr=stats.sem(epoch2_vis1_on1_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_vis1_on1_con_s) > 0 :
            G_epoch3_vis1_on1_con_s.append(np.mean(epoch3_vis1_on1_con_s, axis=0))                    
            axs1[2,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_vis1_on1_con_s , axis=0), yerr=stats.sem(epoch3_vis1_on1_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_vis1_on1_chemo_s) > 0 :
            G_epoch1_vis1_on1_chemo_s.append(np.mean(epoch1_vis1_on1_chemo_s, axis=0))                    
            axs1[2,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_vis1_on1_chemo_s , axis=0), yerr=stats.sem(epoch1_vis1_on1_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_vis1_on1_chemo_s) > 0 :
            G_epoch2_vis1_on1_chemo_s.append(np.mean(epoch2_vis1_on1_chemo_s, axis=0))                    
            axs1[2,0].errorbar(i + 1 , np.mean(epoch2_vis1_on1_chemo_s , axis=0), yerr=stats.sem(epoch2_vis1_on1_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_vis1_on1_chemo_s) > 0 :
            G_epoch3_vis1_on1_chemo_s.append(np.mean(epoch3_vis1_on1_chemo_s, axis=0))                    
            axs1[2,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_vis1_on1_chemo_s , axis=0), yerr=stats.sem(epoch3_vis1_on1_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_vis1_on1_opto_s) > 0 :
            G_epoch1_vis1_on1_opto_s.append(np.mean(epoch1_vis1_on1_opto_s, axis=0))                    
            axs1[2,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_vis1_on1_opto_s , axis=0), yerr=stats.sem(epoch1_vis1_on1_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_vis1_on1_opto_s) > 0 :
            G_epoch2_vis1_on1_opto_s.append(np.mean(epoch2_vis1_on1_opto_s, axis=0))                    
            axs1[2,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_vis1_on1_opto_s , axis=0), yerr=stats.sem(epoch2_vis1_on1_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_vis1_on1_opto_s) > 0 :
            G_epoch3_vis1_on1_opto_s.append(np.mean(epoch3_vis1_on1_opto_s, axis=0))                    
            axs1[2,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_vis1_on1_opto_s , axis=0), yerr=stats.sem(epoch3_vis1_on1_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
            
        # [3,0]
        if len(epoch1_vis1_on1_con_l) > 0 :
            G_epoch1_vis1_on1_con_l.append(np.mean(epoch1_vis1_on1_con_l, axis=0))                    
            axs1[3,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_vis1_on1_con_l , axis=0), yerr=stats.sem(epoch1_vis1_on1_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_vis1_on1_con_l) > 0 :
            G_epoch2_vis1_on1_con_l.append(np.mean(epoch2_vis1_on1_con_l, axis=0))                    
            axs1[3,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_vis1_on1_con_l , axis=0), yerr=stats.sem(epoch2_vis1_on1_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_vis1_on1_con_l) > 0 :
            G_epoch3_vis1_on1_con_l.append(np.mean(epoch3_vis1_on1_con_l, axis=0))                    
            axs1[3,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_vis1_on1_con_l , axis=0), yerr=stats.sem(epoch3_vis1_on1_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_vis1_on1_chemo_l) > 0 :
            G_epoch1_vis1_on1_chemo_l.append(np.mean(epoch1_vis1_on1_chemo_l, axis=0))                    
            axs1[3,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_vis1_on1_chemo_l , axis=0), yerr=stats.sem(epoch1_vis1_on1_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_vis1_on1_chemo_l) > 0 :
            G_epoch2_vis1_on1_chemo_l.append(np.mean(epoch2_vis1_on1_chemo_l, axis=0))                    
            axs1[3,0].errorbar(i + 1 , np.mean(epoch2_vis1_on1_chemo_l , axis=0), yerr=stats.sem(epoch2_vis1_on1_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_vis1_on1_chemo_l) > 0 :
            G_epoch3_vis1_on1_chemo_l.append(np.mean(epoch3_vis1_on1_chemo_l, axis=0))                    
            axs1[3,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_vis1_on1_chemo_l , axis=0), yerr=stats.sem(epoch3_vis1_on1_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_vis1_on1_opto_l) > 0 :
            G_epoch1_vis1_on1_opto_l.append(np.mean(epoch1_vis1_on1_opto_l, axis=0))                    
            axs1[3,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_vis1_on1_opto_l , axis=0), yerr=stats.sem(epoch1_vis1_on1_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_vis1_on1_opto_l) > 0 :
            G_epoch2_vis1_on1_opto_l.append(np.mean(epoch2_vis1_on1_opto_l, axis=0))                    
            axs1[3,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_vis1_on1_opto_l , axis=0), yerr=stats.sem(epoch2_vis1_on1_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_vis1_on1_opto_l) > 0 :
            G_epoch3_vis1_on1_opto_l.append(np.mean(epoch3_vis1_on1_opto_l, axis=0))                    
            axs1[3,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_vis1_on1_opto_l , axis=0), yerr=stats.sem(epoch3_vis1_on1_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        
        # [2,1]
        if len(epoch1_vis2_on2_con_s) > 0 :
            G_epoch1_vis2_on2_con_s.append(np.mean(epoch1_vis2_on2_con_s, axis=0))                    
            axs1[2,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_vis2_on2_con_s , axis=0), yerr=stats.sem(epoch1_vis2_on2_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_vis2_on2_con_s) > 0 :
            G_epoch2_vis2_on2_con_s.append(np.mean(epoch2_vis2_on2_con_s, axis=0))                    
            axs1[2,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_vis2_on2_con_s , axis=0), yerr=stats.sem(epoch2_vis2_on2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_vis2_on2_con_s) > 0 :
            G_epoch3_vis2_on2_con_s.append(np.mean(epoch3_vis2_on2_con_s, axis=0))                    
            axs1[2,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_vis2_on2_con_s , axis=0), yerr=stats.sem(epoch3_vis2_on2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_vis2_on2_chemo_s) > 0 :
            G_epoch1_vis2_on2_chemo_s.append(np.mean(epoch1_vis2_on2_chemo_s, axis=0))                    
            axs1[2,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_vis2_on2_chemo_s , axis=0), yerr=stats.sem(epoch1_vis2_on2_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_vis2_on2_chemo_s) > 0 :
            G_epoch2_vis2_on2_chemo_s.append(np.mean(epoch2_vis2_on2_chemo_s, axis=0))                    
            axs1[2,1].errorbar(i + 1 , np.mean(epoch2_vis2_on2_chemo_s , axis=0), yerr=stats.sem(epoch2_vis2_on2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_vis2_on2_chemo_s) > 0 :
            G_epoch3_vis2_on2_chemo_s.append(np.mean(epoch3_vis2_on2_chemo_s, axis=0))                    
            axs1[2,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_vis2_on2_chemo_s , axis=0), yerr=stats.sem(epoch3_vis2_on2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_vis2_on2_opto_s) > 0 :
            G_epoch1_vis2_on2_opto_s.append(np.mean(epoch1_vis2_on2_opto_s, axis=0))                    
            axs1[2,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_vis2_on2_opto_s , axis=0), yerr=stats.sem(epoch1_vis2_on2_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_vis2_on2_opto_s) > 0 :
            G_epoch2_vis2_on2_opto_s.append(np.mean(epoch2_vis2_on2_opto_s, axis=0))                    
            axs1[2,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_vis2_on2_opto_s , axis=0), yerr=stats.sem(epoch2_vis2_on2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_vis2_on2_opto_s) > 0 :
            G_epoch3_vis2_on2_opto_s.append(np.mean(epoch3_vis2_on2_opto_s, axis=0))                    
            axs1[2,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_vis2_on2_opto_s , axis=0), yerr=stats.sem(epoch3_vis2_on2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        # [3,1]
        if len(epoch1_vis2_on2_con_l) > 0 :
            G_epoch1_vis2_on2_con_l.append(np.mean(epoch1_vis2_on2_con_l, axis=0))                    
            axs1[3,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_vis2_on2_con_l , axis=0), yerr=stats.sem(epoch1_vis2_on2_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_vis2_on2_con_l) > 0 :
            G_epoch2_vis2_on2_con_l.append(np.mean(epoch2_vis2_on2_con_l, axis=0))                    
            axs1[3,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_vis2_on2_con_l , axis=0), yerr=stats.sem(epoch2_vis2_on2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_vis2_on2_con_l) > 0 :
            G_epoch3_vis2_on2_con_l.append(np.mean(epoch3_vis2_on2_con_l, axis=0))                    
            axs1[3,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_vis2_on2_con_l , axis=0), yerr=stats.sem(epoch3_vis2_on2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_vis2_on2_chemo_l) > 0 :
            G_epoch1_vis2_on2_chemo_l.append(np.mean(epoch1_vis2_on2_chemo_l, axis=0))                    
            axs1[3,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_vis2_on2_chemo_l , axis=0), yerr=stats.sem(epoch1_vis2_on2_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_vis2_on2_chemo_l) > 0 :
            G_epoch2_vis2_on2_chemo_l.append(np.mean(epoch2_vis2_on2_chemo_l, axis=0))                    
            axs1[3,1].errorbar(i + 1 , np.mean(epoch2_vis2_on2_chemo_l , axis=0), yerr=stats.sem(epoch2_vis2_on2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_vis2_on2_chemo_l) > 0 :
            G_epoch3_vis2_on2_chemo_l.append(np.mean(epoch3_vis2_on2_chemo_l, axis=0))                    
            axs1[3,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_vis2_on2_chemo_l , axis=0), yerr=stats.sem(epoch3_vis2_on2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_vis2_on2_opto_l) > 0 :
            G_epoch1_vis2_on2_opto_l.append(np.mean(epoch1_vis2_on2_opto_l, axis=0))                    
            axs1[3,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_vis2_on2_opto_l , axis=0), yerr=stats.sem(epoch1_vis2_on2_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_vis2_on2_opto_l) > 0 :
            G_epoch2_vis2_on2_opto_l.append(np.mean(epoch2_vis2_on2_opto_l, axis=0))                    
            axs1[3,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_vis2_on2_opto_l , axis=0), yerr=stats.sem(epoch2_vis2_on2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_vis2_on2_opto_l) > 0 :
            G_epoch3_vis2_on2_opto_l.append(np.mean(epoch3_vis2_on2_opto_l, axis=0))                    
            axs1[3,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_vis2_on2_opto_l , axis=0), yerr=stats.sem(epoch3_vis2_on2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        
        # [4,0]
        if len(epoch1_on1_targ1_con_s) > 0 :
            G_epoch1_on1_targ1_con_s.append(np.mean(epoch1_on1_targ1_con_s, axis=0))                    
            axs1[4,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_on1_targ1_con_s , axis=0), yerr=stats.sem(epoch1_on1_targ1_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_on1_targ1_con_s) > 0 :
            G_epoch2_on1_targ1_con_s.append(np.mean(epoch2_on1_targ1_con_s, axis=0))                    
            axs1[4,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_on1_targ1_con_s , axis=0), yerr=stats.sem(epoch2_on1_targ1_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_on1_targ1_con_s) > 0 :
            G_epoch3_on1_targ1_con_s.append(np.mean(epoch3_on1_targ1_con_s, axis=0))                    
            axs1[4,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_on1_targ1_con_s , axis=0), yerr=stats.sem(epoch3_on1_targ1_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_on1_targ1_chemo_s) > 0 :
            G_epoch1_on1_targ1_chemo_s.append(np.mean(epoch1_on1_targ1_chemo_s, axis=0))                    
            axs1[4,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_on1_targ1_chemo_s , axis=0), yerr=stats.sem(epoch1_on1_targ1_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_on1_targ1_chemo_s) > 0 :
            G_epoch2_on1_targ1_chemo_s.append(np.mean(epoch2_on1_targ1_chemo_s, axis=0))                    
            axs1[4,0].errorbar(i + 1 , np.mean(epoch2_on1_targ1_chemo_s , axis=0), yerr=stats.sem(epoch2_on1_targ1_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_on1_targ1_chemo_s) > 0 :
            G_epoch3_on1_targ1_chemo_s.append(np.mean(epoch3_on1_targ1_chemo_s, axis=0))                    
            axs1[4,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_on1_targ1_chemo_s , axis=0), yerr=stats.sem(epoch3_on1_targ1_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_on1_targ1_opto_s) > 0 :
            G_epoch1_on1_targ1_opto_s.append(np.mean(epoch1_on1_targ1_opto_s, axis=0))                    
            axs1[4,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_on1_targ1_opto_s , axis=0), yerr=stats.sem(epoch1_on1_targ1_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_on1_targ1_opto_s) > 0 :
            G_epoch2_on1_targ1_opto_s.append(np.mean(epoch2_on1_targ1_opto_s, axis=0))                    
            axs1[4,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_on1_targ1_opto_s , axis=0), yerr=stats.sem(epoch2_on1_targ1_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_on1_targ1_opto_s) > 0 :
            G_epoch3_on1_targ1_opto_s.append(np.mean(epoch3_on1_targ1_opto_s, axis=0))                    
            axs1[4,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_on1_targ1_opto_s , axis=0), yerr=stats.sem(epoch3_on1_targ1_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
            
        # [5,0]
        if len(epoch1_on1_targ1_con_l) > 0 :
            G_epoch1_on1_targ1_con_l.append(np.mean(epoch1_on1_targ1_con_l, axis=0))                    
            axs1[5,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_on1_targ1_con_l , axis=0), yerr=stats.sem(epoch1_on1_targ1_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_on1_targ1_con_l) > 0 :
            G_epoch2_on1_targ1_con_l.append(np.mean(epoch2_on1_targ1_con_l, axis=0))                    
            axs1[5,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_on1_targ1_con_l , axis=0), yerr=stats.sem(epoch2_on1_targ1_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_on1_targ1_con_l) > 0 :
            G_epoch3_on1_targ1_con_l.append(np.mean(epoch3_on1_targ1_con_l, axis=0))                    
            axs1[5,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_on1_targ1_con_l , axis=0), yerr=stats.sem(epoch3_on1_targ1_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_on1_targ1_chemo_l) > 0 :
            G_epoch1_on1_targ1_chemo_l.append(np.mean(epoch1_on1_targ1_chemo_l, axis=0))                    
            axs1[5,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_on1_targ1_chemo_l , axis=0), yerr=stats.sem(epoch1_on1_targ1_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_on1_targ1_chemo_l) > 0 :
            G_epoch2_on1_targ1_chemo_l.append(np.mean(epoch2_on1_targ1_chemo_l, axis=0))                    
            axs1[5,0].errorbar(i + 1 , np.mean(epoch2_on1_targ1_chemo_l , axis=0), yerr=stats.sem(epoch2_on1_targ1_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_on1_targ1_chemo_l) > 0 :
            G_epoch3_on1_targ1_chemo_l.append(np.mean(epoch3_on1_targ1_chemo_l, axis=0))                    
            axs1[5,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_on1_targ1_chemo_l , axis=0), yerr=stats.sem(epoch3_on1_targ1_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_on1_targ1_opto_l) > 0 :
            G_epoch1_on1_targ1_opto_l.append(np.mean(epoch1_on1_targ1_opto_l, axis=0))                    
            axs1[5,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_on1_targ1_opto_l , axis=0), yerr=stats.sem(epoch1_on1_targ1_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_on1_targ1_opto_l) > 0 :
            G_epoch2_on1_targ1_opto_l.append(np.mean(epoch2_on1_targ1_opto_l, axis=0))                    
            axs1[5,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_on1_targ1_opto_l , axis=0), yerr=stats.sem(epoch2_on1_targ1_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_on1_targ1_opto_l) > 0 :
            G_epoch3_on1_targ1_opto_l.append(np.mean(epoch3_on1_targ1_opto_l, axis=0))                    
            axs1[5,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_on1_targ1_opto_l , axis=0), yerr=stats.sem(epoch3_on1_targ1_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        
        # [4,1]
        if len(epoch1_on2_targ2_con_s) > 0 :
            G_epoch1_on2_targ2_con_s.append(np.mean(epoch1_on2_targ2_con_s, axis=0))                    
            axs1[4,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_on2_targ2_con_s , axis=0), yerr=stats.sem(epoch1_on2_targ2_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_on2_targ2_con_s) > 0 :
            G_epoch2_on2_targ2_con_s.append(np.mean(epoch2_on2_targ2_con_s, axis=0))                    
            axs1[4,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_on2_targ2_con_s , axis=0), yerr=stats.sem(epoch2_on2_targ2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_on2_targ2_con_s) > 0 :
            G_epoch3_on2_targ2_con_s.append(np.mean(epoch3_on2_targ2_con_s, axis=0))                    
            axs1[4,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_on2_targ2_con_s , axis=0), yerr=stats.sem(epoch3_on2_targ2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_on2_targ2_chemo_s) > 0 :
            G_epoch1_on2_targ2_chemo_s.append(np.mean(epoch1_on2_targ2_chemo_s, axis=0))                    
            axs1[4,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_on2_targ2_chemo_s , axis=0), yerr=stats.sem(epoch1_on2_targ2_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_on2_targ2_chemo_s) > 0 :
            G_epoch2_on2_targ2_chemo_s.append(np.mean(epoch2_on2_targ2_chemo_s, axis=0))                    
            axs1[4,1].errorbar(i + 1 , np.mean(epoch2_on2_targ2_chemo_s , axis=0), yerr=stats.sem(epoch2_on2_targ2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_on2_targ2_chemo_s) > 0 :
            G_epoch3_on2_targ2_chemo_s.append(np.mean(epoch3_on2_targ2_chemo_s, axis=0))                    
            axs1[4,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_on2_targ2_chemo_s , axis=0), yerr=stats.sem(epoch3_on2_targ2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_on2_targ2_opto_s) > 0 :
            G_epoch1_on2_targ2_opto_s.append(np.mean(epoch1_on2_targ2_opto_s, axis=0))                    
            axs1[4,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_on2_targ2_opto_s , axis=0), yerr=stats.sem(epoch1_on2_targ2_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_on2_targ2_opto_s) > 0 :
            G_epoch2_on2_targ2_opto_s.append(np.mean(epoch2_on2_targ2_opto_s, axis=0))                    
            axs1[4,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_on2_targ2_opto_s , axis=0), yerr=stats.sem(epoch2_on2_targ2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_on2_targ2_opto_s) > 0 :
            G_epoch3_on2_targ2_opto_s.append(np.mean(epoch3_on2_targ2_opto_s, axis=0))                    
            axs1[4,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_on2_targ2_opto_s , axis=0), yerr=stats.sem(epoch3_on2_targ2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        # [5,1]
        if len(epoch1_on2_targ2_con_l) > 0 :
            G_epoch1_on2_targ2_con_l.append(np.mean(epoch1_on2_targ2_con_l, axis=0))                    
            axs1[5,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_on2_targ2_con_l , axis=0), yerr=stats.sem(epoch1_on2_targ2_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_on2_targ2_con_l) > 0 :
            G_epoch2_on2_targ2_con_l.append(np.mean(epoch2_on2_targ2_con_l, axis=0))                    
            axs1[5,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_on2_targ2_con_l , axis=0), yerr=stats.sem(epoch2_on2_targ2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_on2_targ2_con_l) > 0 :
            G_epoch3_on2_targ2_con_l.append(np.mean(epoch3_on2_targ2_con_l, axis=0))                    
            axs1[5,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_on2_targ2_con_l , axis=0), yerr=stats.sem(epoch3_on2_targ2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_on2_targ2_chemo_l) > 0 :
            G_epoch1_on2_targ2_chemo_l.append(np.mean(epoch1_on2_targ2_chemo_l, axis=0))                    
            axs1[5,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_on2_targ2_chemo_l , axis=0), yerr=stats.sem(epoch1_on2_targ2_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_on2_targ2_chemo_l) > 0 :
            G_epoch2_on2_targ2_chemo_l.append(np.mean(epoch2_on2_targ2_chemo_l, axis=0))                    
            axs1[5,1].errorbar(i + 1 , np.mean(epoch2_on2_targ2_chemo_l , axis=0), yerr=stats.sem(epoch2_on2_targ2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_on2_targ2_chemo_l) > 0 :
            G_epoch3_on2_targ2_chemo_l.append(np.mean(epoch3_on2_targ2_chemo_l, axis=0))                    
            axs1[5,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_on2_targ2_chemo_l , axis=0), yerr=stats.sem(epoch3_on2_targ2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_on2_targ2_opto_l) > 0 :
            G_epoch1_on2_targ2_opto_l.append(np.mean(epoch1_on2_targ2_opto_l, axis=0))                    
            axs1[5,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_on2_targ2_opto_l , axis=0), yerr=stats.sem(epoch1_on2_targ2_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_on2_targ2_opto_l) > 0 :
            G_epoch2_on2_targ2_opto_l.append(np.mean(epoch2_on2_targ2_opto_l, axis=0))                    
            axs1[5,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_on2_targ2_opto_l , axis=0), yerr=stats.sem(epoch2_on2_targ2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_on2_targ2_opto_l) > 0 :
            G_epoch3_on2_targ2_opto_l.append(np.mean(epoch3_on2_targ2_opto_l, axis=0))                    
            axs1[5,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_on2_targ2_opto_l , axis=0), yerr=stats.sem(epoch3_on2_targ2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        # [6,0]
        if len(epoch1_on1_amp1_con_s) > 0 :
            G_epoch1_on1_amp1_con_s.append(np.mean(epoch1_on1_amp1_con_s, axis=0))                    
            axs1[6,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_on1_amp1_con_s , axis=0), yerr=stats.sem(epoch1_on1_amp1_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_on1_amp1_con_s) > 0 :
            G_epoch2_on1_amp1_con_s.append(np.mean(epoch2_on1_amp1_con_s, axis=0))                    
            axs1[6,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_on1_amp1_con_s , axis=0), yerr=stats.sem(epoch2_on1_amp1_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_on1_amp1_con_s) > 0 :
            G_epoch3_on1_amp1_con_s.append(np.mean(epoch3_on1_amp1_con_s, axis=0))                    
            axs1[6,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_on1_amp1_con_s , axis=0), yerr=stats.sem(epoch3_on1_amp1_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_on1_amp1_chemo_s) > 0 :
            G_epoch1_on1_amp1_chemo_s.append(np.mean(epoch1_on1_amp1_chemo_s, axis=0))                    
            axs1[6,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_on1_amp1_chemo_s , axis=0), yerr=stats.sem(epoch1_on1_amp1_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_on1_amp1_chemo_s) > 0 :
            G_epoch2_on1_amp1_chemo_s.append(np.mean(epoch2_on1_amp1_chemo_s, axis=0))                    
            axs1[6,0].errorbar(i + 1 , np.mean(epoch2_on1_amp1_chemo_s , axis=0), yerr=stats.sem(epoch2_on1_amp1_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_on1_amp1_chemo_s) > 0 :
            G_epoch3_on1_amp1_chemo_s.append(np.mean(epoch3_on1_amp1_chemo_s, axis=0))                    
            axs1[6,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_on1_amp1_chemo_s , axis=0), yerr=stats.sem(epoch3_on1_amp1_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_on1_amp1_opto_s) > 0 :
            G_epoch1_on1_amp1_opto_s.append(np.mean(epoch1_on1_amp1_opto_s, axis=0))                    
            axs1[6,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_on1_amp1_opto_s , axis=0), yerr=stats.sem(epoch1_on1_amp1_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_on1_amp1_opto_s) > 0 :
            G_epoch2_on1_amp1_opto_s.append(np.mean(epoch2_on1_amp1_opto_s, axis=0))                    
            axs1[6,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_on1_amp1_opto_s , axis=0), yerr=stats.sem(epoch2_on1_amp1_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_on1_amp1_opto_s) > 0 :
            G_epoch3_on1_amp1_opto_s.append(np.mean(epoch3_on1_amp1_opto_s, axis=0))                    
            axs1[6,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_on1_amp1_opto_s , axis=0), yerr=stats.sem(epoch3_on1_amp1_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
            
        # [7,0]
        if len(epoch1_on1_amp1_con_l) > 0 :
            G_epoch1_on1_amp1_con_l.append(np.mean(epoch1_on1_amp1_con_l, axis=0))                    
            axs1[7,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_on1_amp1_con_l , axis=0), yerr=stats.sem(epoch1_on1_amp1_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_on1_amp1_con_l) > 0 :
            G_epoch2_on1_amp1_con_l.append(np.mean(epoch2_on1_amp1_con_l, axis=0))                    
            axs1[7,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_on1_amp1_con_l , axis=0), yerr=stats.sem(epoch2_on1_amp1_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_on1_amp1_con_l) > 0 :
            G_epoch3_on1_amp1_con_l.append(np.mean(epoch3_on1_amp1_con_l, axis=0))                    
            axs1[7,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_on1_amp1_con_l , axis=0), yerr=stats.sem(epoch3_on1_amp1_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_on1_amp1_chemo_l) > 0 :
            G_epoch1_on1_amp1_chemo_l.append(np.mean(epoch1_on1_amp1_chemo_l, axis=0))                    
            axs1[7,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_on1_amp1_chemo_l , axis=0), yerr=stats.sem(epoch1_on1_amp1_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_on1_amp1_chemo_l) > 0 :
            G_epoch2_on1_amp1_chemo_l.append(np.mean(epoch2_on1_amp1_chemo_l, axis=0))                    
            axs1[7,0].errorbar(i + 1 , np.mean(epoch2_on1_amp1_chemo_l , axis=0), yerr=stats.sem(epoch2_on1_amp1_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_on1_amp1_chemo_l) > 0 :
            G_epoch3_on1_amp1_chemo_l.append(np.mean(epoch3_on1_amp1_chemo_l, axis=0))                    
            axs1[7,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_on1_amp1_chemo_l , axis=0), yerr=stats.sem(epoch3_on1_amp1_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_on1_amp1_opto_l) > 0 :
            G_epoch1_on1_amp1_opto_l.append(np.mean(epoch1_on1_amp1_opto_l, axis=0))                    
            axs1[7,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_on1_amp1_opto_l , axis=0), yerr=stats.sem(epoch1_on1_amp1_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_on1_amp1_opto_l) > 0 :
            G_epoch2_on1_amp1_opto_l.append(np.mean(epoch2_on1_amp1_opto_l, axis=0))                    
            axs1[7,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_on1_amp1_opto_l , axis=0), yerr=stats.sem(epoch2_on1_amp1_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_on1_amp1_opto_l) > 0 :
            G_epoch3_on1_amp1_opto_l.append(np.mean(epoch3_on1_amp1_opto_l, axis=0))                    
            axs1[7,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_on1_amp1_opto_l , axis=0), yerr=stats.sem(epoch3_on1_amp1_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        
        # [6,1]
        if len(epoch1_on2_amp2_con_s) > 0 :
            G_epoch1_on2_amp2_con_s.append(np.mean(epoch1_on2_amp2_con_s, axis=0))                    
            axs1[6,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_on2_amp2_con_s , axis=0), yerr=stats.sem(epoch1_on2_amp2_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_on2_amp2_con_s) > 0 :
            G_epoch2_on2_amp2_con_s.append(np.mean(epoch2_on2_amp2_con_s, axis=0))                    
            axs1[6,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_on2_amp2_con_s , axis=0), yerr=stats.sem(epoch2_on2_amp2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_on2_amp2_con_s) > 0 :
            G_epoch3_on2_amp2_con_s.append(np.mean(epoch3_on2_amp2_con_s, axis=0))                    
            axs1[6,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_on2_amp2_con_s , axis=0), yerr=stats.sem(epoch3_on2_amp2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_on2_amp2_chemo_s) > 0 :
            G_epoch1_on2_amp2_chemo_s.append(np.mean(epoch1_on2_amp2_chemo_s, axis=0))                    
            axs1[6,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_on2_amp2_chemo_s , axis=0), yerr=stats.sem(epoch1_on2_amp2_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_on2_amp2_chemo_s) > 0 :
            G_epoch2_on2_amp2_chemo_s.append(np.mean(epoch2_on2_amp2_chemo_s, axis=0))                    
            axs1[6,1].errorbar(i + 1 , np.mean(epoch2_on2_amp2_chemo_s , axis=0), yerr=stats.sem(epoch2_on2_amp2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_on2_amp2_chemo_s) > 0 :
            G_epoch3_on2_amp2_chemo_s.append(np.mean(epoch3_on2_amp2_chemo_s, axis=0))                    
            axs1[6,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_on2_amp2_chemo_s , axis=0), yerr=stats.sem(epoch3_on2_amp2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_on2_amp2_opto_s) > 0 :
            G_epoch1_on2_amp2_opto_s.append(np.mean(epoch1_on2_amp2_opto_s, axis=0))                    
            axs1[6,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_on2_amp2_opto_s , axis=0), yerr=stats.sem(epoch1_on2_amp2_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_on2_amp2_opto_s) > 0 :
            G_epoch2_on2_amp2_opto_s.append(np.mean(epoch2_on2_amp2_opto_s, axis=0))                    
            axs1[6,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_on2_amp2_opto_s , axis=0), yerr=stats.sem(epoch2_on2_amp2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_on2_amp2_opto_s) > 0 :
            G_epoch3_on2_amp2_opto_s.append(np.mean(epoch3_on2_amp2_opto_s, axis=0))                    
            axs1[6,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_on2_amp2_opto_s , axis=0), yerr=stats.sem(epoch3_on2_amp2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        # [7,1]
        if len(epoch1_on2_amp2_con_l) > 0 :
            G_epoch1_on2_amp2_con_l.append(np.mean(epoch1_on2_amp2_con_l, axis=0))                    
            axs1[7,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_on2_amp2_con_l , axis=0), yerr=stats.sem(epoch1_on2_amp2_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_on2_amp2_con_l) > 0 :
            G_epoch2_on2_amp2_con_l.append(np.mean(epoch2_on2_amp2_con_l, axis=0))                    
            axs1[7,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_on2_amp2_con_l , axis=0), yerr=stats.sem(epoch2_on2_amp2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_on2_amp2_con_l) > 0 :
            G_epoch3_on2_amp2_con_l.append(np.mean(epoch3_on2_amp2_con_l, axis=0))                    
            axs1[7,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_on2_amp2_con_l , axis=0), yerr=stats.sem(epoch3_on2_amp2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_on2_amp2_chemo_l) > 0 :
            G_epoch1_on2_amp2_chemo_l.append(np.mean(epoch1_on2_amp2_chemo_l, axis=0))                    
            axs1[7,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_on2_amp2_chemo_l , axis=0), yerr=stats.sem(epoch1_on2_amp2_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_on2_amp2_chemo_l) > 0 :
            G_epoch2_on2_amp2_chemo_l.append(np.mean(epoch2_on2_amp2_chemo_l, axis=0))                    
            axs1[7,1].errorbar(i + 1 , np.mean(epoch2_on2_amp2_chemo_l , axis=0), yerr=stats.sem(epoch2_on2_amp2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_on2_amp2_chemo_l) > 0 :
            G_epoch3_on2_amp2_chemo_l.append(np.mean(epoch3_on2_amp2_chemo_l, axis=0))                    
            axs1[7,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_on2_amp2_chemo_l , axis=0), yerr=stats.sem(epoch3_on2_amp2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_on2_amp2_opto_l) > 0 :
            G_epoch1_on2_amp2_opto_l.append(np.mean(epoch1_on2_amp2_opto_l, axis=0))                    
            axs1[7,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_on2_amp2_opto_l , axis=0), yerr=stats.sem(epoch1_on2_amp2_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_on2_amp2_opto_l) > 0 :
            G_epoch2_on2_amp2_opto_l.append(np.mean(epoch2_on2_amp2_opto_l, axis=0))                    
            axs1[7,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_on2_amp2_opto_l , axis=0), yerr=stats.sem(epoch2_on2_amp2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_on2_amp2_opto_l) > 0 :
            G_epoch3_on2_amp2_opto_l.append(np.mean(epoch3_on2_amp2_opto_l, axis=0))                    
            axs1[7,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_on2_amp2_opto_l , axis=0), yerr=stats.sem(epoch3_on2_amp2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        # [8,0]
        if len(epoch1_on_interval_con_s) > 0 :
            G_epoch1_on_interval_con_s.append(np.mean(epoch1_on_interval_con_s, axis=0))                    
            axs1[8,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_on_interval_con_s , axis=0), yerr=stats.sem(epoch1_on_interval_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_on_interval_con_s) > 0 :
            G_epoch2_on_interval_con_s.append(np.mean(epoch2_on_interval_con_s, axis=0))                    
            axs1[8,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_on_interval_con_s , axis=0), yerr=stats.sem(epoch2_on_interval_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_on_interval_con_s) > 0 :
            G_epoch3_on_interval_con_s.append(np.mean(epoch3_on_interval_con_s, axis=0))                    
            axs1[8,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_on_interval_con_s , axis=0), yerr=stats.sem(epoch3_on_interval_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_on_interval_chemo_s) > 0 :
            G_epoch1_on_interval_chemo_s.append(np.mean(epoch1_on_interval_chemo_s, axis=0))                    
            axs1[8,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_on_interval_chemo_s , axis=0), yerr=stats.sem(epoch1_on_interval_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_on_interval_chemo_s) > 0 :
            G_epoch2_on_interval_chemo_s.append(np.mean(epoch2_on_interval_chemo_s, axis=0))                    
            axs1[8,0].errorbar(i + 1 , np.mean(epoch2_on_interval_chemo_s , axis=0), yerr=stats.sem(epoch2_on_interval_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_on_interval_chemo_s) > 0 :
            G_epoch3_on_interval_chemo_s.append(np.mean(epoch3_on_interval_chemo_s, axis=0))                    
            axs1[8,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_on_interval_chemo_s , axis=0), yerr=stats.sem(epoch3_on_interval_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_on_interval_opto_s) > 0 :
            G_epoch1_on_interval_opto_s.append(np.mean(epoch1_on_interval_opto_s, axis=0))                    
            axs1[8,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_on_interval_opto_s , axis=0), yerr=stats.sem(epoch1_on_interval_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_on_interval_opto_s) > 0 :
            G_epoch2_on_interval_opto_s.append(np.mean(epoch2_on_interval_opto_s, axis=0))                    
            axs1[8,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_on_interval_opto_s , axis=0), yerr=stats.sem(epoch2_on_interval_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_on_interval_opto_s) > 0 :
            G_epoch3_on_interval_opto_s.append(np.mean(epoch3_on_interval_opto_s, axis=0))                    
            axs1[8,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_on_interval_opto_s , axis=0), yerr=stats.sem(epoch3_on_interval_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
            
        # [9,0]
        if len(epoch1_on_interval_con_l) > 0 :
            G_epoch1_on_interval_con_l.append(np.mean(epoch1_on_interval_con_l, axis=0))                    
            axs1[9,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_on_interval_con_l , axis=0), yerr=stats.sem(epoch1_on_interval_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_on_interval_con_l) > 0 :
            G_epoch2_on_interval_con_l.append(np.mean(epoch2_on_interval_con_l, axis=0))                    
            axs1[9,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_on_interval_con_l , axis=0), yerr=stats.sem(epoch2_on_interval_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_on_interval_con_l) > 0 :
            G_epoch3_on_interval_con_l.append(np.mean(epoch3_on_interval_con_l, axis=0))                    
            axs1[9,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_on_interval_con_l , axis=0), yerr=stats.sem(epoch3_on_interval_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_on_interval_chemo_l) > 0 :
            G_epoch1_on_interval_chemo_l.append(np.mean(epoch1_on_interval_chemo_l, axis=0))                    
            axs1[9,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_on_interval_chemo_l , axis=0), yerr=stats.sem(epoch1_on_interval_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_on_interval_chemo_l) > 0 :
            G_epoch2_on_interval_chemo_l.append(np.mean(epoch2_on_interval_chemo_l, axis=0))                    
            axs1[9,0].errorbar(i + 1 , np.mean(epoch2_on_interval_chemo_l , axis=0), yerr=stats.sem(epoch2_on_interval_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_on_interval_chemo_l) > 0 :
            G_epoch3_on_interval_chemo_l.append(np.mean(epoch3_on_interval_chemo_l, axis=0))                    
            axs1[9,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_on_interval_chemo_l , axis=0), yerr=stats.sem(epoch3_on_interval_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_on_interval_opto_l) > 0 :
            G_epoch1_on_interval_opto_l.append(np.mean(epoch1_on_interval_opto_l, axis=0))                    
            axs1[9,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_on_interval_opto_l , axis=0), yerr=stats.sem(epoch1_on_interval_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_on_interval_opto_l) > 0 :
            G_epoch2_on_interval_opto_l.append(np.mean(epoch2_on_interval_opto_l, axis=0))                    
            axs1[9,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_on_interval_opto_l , axis=0), yerr=stats.sem(epoch2_on_interval_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_on_interval_opto_l) > 0 :
            G_epoch3_on_interval_opto_l.append(np.mean(epoch3_on_interval_opto_l, axis=0))                    
            axs1[9,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_on_interval_opto_l , axis=0), yerr=stats.sem(epoch3_on_interval_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        
        # [8,1]
        if len(epoch1_levret1end_on2_con_s) > 0 :
            G_epoch1_levret1end_on2_con_s.append(np.mean(epoch1_levret1end_on2_con_s, axis=0))                    
            axs1[8,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_levret1end_on2_con_s , axis=0), yerr=stats.sem(epoch1_levret1end_on2_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_levret1end_on2_con_s) > 0 :
            G_epoch2_levret1end_on2_con_s.append(np.mean(epoch2_levret1end_on2_con_s, axis=0))                    
            axs1[8,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_levret1end_on2_con_s , axis=0), yerr=stats.sem(epoch2_levret1end_on2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_levret1end_on2_con_s) > 0 :
            G_epoch3_levret1end_on2_con_s.append(np.mean(epoch3_levret1end_on2_con_s, axis=0))                    
            axs1[8,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_levret1end_on2_con_s , axis=0), yerr=stats.sem(epoch3_levret1end_on2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_levret1end_on2_chemo_s) > 0 :
            G_epoch1_levret1end_on2_chemo_s.append(np.mean(epoch1_levret1end_on2_chemo_s, axis=0))                    
            axs1[8,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_levret1end_on2_chemo_s , axis=0), yerr=stats.sem(epoch1_levret1end_on2_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_levret1end_on2_chemo_s) > 0 :
            G_epoch2_levret1end_on2_chemo_s.append(np.mean(epoch2_levret1end_on2_chemo_s, axis=0))                    
            axs1[8,1].errorbar(i + 1 , np.mean(epoch2_levret1end_on2_chemo_s , axis=0), yerr=stats.sem(epoch2_levret1end_on2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_levret1end_on2_chemo_s) > 0 :
            G_epoch3_levret1end_on2_chemo_s.append(np.mean(epoch3_levret1end_on2_chemo_s, axis=0))                    
            axs1[8,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_levret1end_on2_chemo_s , axis=0), yerr=stats.sem(epoch3_levret1end_on2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_levret1end_on2_opto_s) > 0 :
            G_epoch1_levret1end_on2_opto_s.append(np.mean(epoch1_levret1end_on2_opto_s, axis=0))                    
            axs1[8,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_levret1end_on2_opto_s , axis=0), yerr=stats.sem(epoch1_levret1end_on2_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_levret1end_on2_opto_s) > 0 :
            G_epoch2_levret1end_on2_opto_s.append(np.mean(epoch2_levret1end_on2_opto_s, axis=0))                    
            axs1[8,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_levret1end_on2_opto_s , axis=0), yerr=stats.sem(epoch2_levret1end_on2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_levret1end_on2_opto_s) > 0 :
            G_epoch3_levret1end_on2_opto_s.append(np.mean(epoch3_levret1end_on2_opto_s, axis=0))                    
            axs1[8,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_levret1end_on2_opto_s , axis=0), yerr=stats.sem(epoch3_levret1end_on2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        # [9,1]
        if len(epoch1_levret1end_on2_con_l) > 0 :
            G_epoch1_levret1end_on2_con_l.append(np.mean(epoch1_levret1end_on2_con_l, axis=0))                    
            axs1[9,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_levret1end_on2_con_l , axis=0), yerr=stats.sem(epoch1_levret1end_on2_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_levret1end_on2_con_l) > 0 :
            G_epoch2_levret1end_on2_con_l.append(np.mean(epoch2_levret1end_on2_con_l, axis=0))                    
            axs1[9,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_levret1end_on2_con_l , axis=0), yerr=stats.sem(epoch2_levret1end_on2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_levret1end_on2_con_l) > 0 :
            G_epoch3_levret1end_on2_con_l.append(np.mean(epoch3_levret1end_on2_con_l, axis=0))                    
            axs1[9,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_levret1end_on2_con_l , axis=0), yerr=stats.sem(epoch3_levret1end_on2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_levret1end_on2_chemo_l) > 0 :
            G_epoch1_levret1end_on2_chemo_l.append(np.mean(epoch1_levret1end_on2_chemo_l, axis=0))                    
            axs1[9,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_levret1end_on2_chemo_l , axis=0), yerr=stats.sem(epoch1_levret1end_on2_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_levret1end_on2_chemo_l) > 0 :
            G_epoch2_levret1end_on2_chemo_l.append(np.mean(epoch2_levret1end_on2_chemo_l, axis=0))                    
            axs1[9,1].errorbar(i + 1 , np.mean(epoch2_levret1end_on2_chemo_l , axis=0), yerr=stats.sem(epoch2_levret1end_on2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_levret1end_on2_chemo_l) > 0 :
            G_epoch3_levret1end_on2_chemo_l.append(np.mean(epoch3_levret1end_on2_chemo_l, axis=0))                    
            axs1[9,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_levret1end_on2_chemo_l , axis=0), yerr=stats.sem(epoch3_levret1end_on2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_levret1end_on2_opto_l) > 0 :
            G_epoch1_levret1end_on2_opto_l.append(np.mean(epoch1_levret1end_on2_opto_l, axis=0))                    
            axs1[9,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_levret1end_on2_opto_l , axis=0), yerr=stats.sem(epoch1_levret1end_on2_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_levret1end_on2_opto_l) > 0 :
            G_epoch2_levret1end_on2_opto_l.append(np.mean(epoch2_levret1end_on2_opto_l, axis=0))                    
            axs1[9,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_levret1end_on2_opto_l , axis=0), yerr=stats.sem(epoch2_levret1end_on2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_levret1end_on2_opto_l) > 0 :
            G_epoch3_levret1end_on2_opto_l.append(np.mean(epoch3_levret1end_on2_opto_l, axis=0))                    
            axs1[9,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_levret1end_on2_opto_l , axis=0), yerr=stats.sem(epoch3_levret1end_on2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        # [10,0]
        if len(epoch1_base_amp1_con_s) > 0 :
            G_epoch1_base_amp1_con_s.append(np.mean(epoch1_base_amp1_con_s, axis=0))                    
            axs1[10,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_base_amp1_con_s , axis=0), yerr=stats.sem(epoch1_base_amp1_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_base_amp1_con_s) > 0 :
            G_epoch2_base_amp1_con_s.append(np.mean(epoch2_base_amp1_con_s, axis=0))                    
            axs1[10,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_base_amp1_con_s , axis=0), yerr=stats.sem(epoch2_base_amp1_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_base_amp1_con_s) > 0 :
            G_epoch3_base_amp1_con_s.append(np.mean(epoch3_base_amp1_con_s, axis=0))                    
            axs1[10,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_base_amp1_con_s , axis=0), yerr=stats.sem(epoch3_base_amp1_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_base_amp1_chemo_s) > 0 :
            G_epoch1_base_amp1_chemo_s.append(np.mean(epoch1_base_amp1_chemo_s, axis=0))                    
            axs1[10,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_base_amp1_chemo_s , axis=0), yerr=stats.sem(epoch1_base_amp1_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_base_amp1_chemo_s) > 0 :
            G_epoch2_base_amp1_chemo_s.append(np.mean(epoch2_base_amp1_chemo_s, axis=0))                    
            axs1[10,0].errorbar(i + 1 , np.mean(epoch2_base_amp1_chemo_s , axis=0), yerr=stats.sem(epoch2_base_amp1_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_base_amp1_chemo_s) > 0 :
            G_epoch3_base_amp1_chemo_s.append(np.mean(epoch3_base_amp1_chemo_s, axis=0))                    
            axs1[10,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_base_amp1_chemo_s , axis=0), yerr=stats.sem(epoch3_base_amp1_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_base_amp1_opto_s) > 0 :
            G_epoch1_base_amp1_opto_s.append(np.mean(epoch1_base_amp1_opto_s, axis=0))                    
            axs1[10,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_base_amp1_opto_s , axis=0), yerr=stats.sem(epoch1_base_amp1_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_base_amp1_opto_s) > 0 :
            G_epoch2_base_amp1_opto_s.append(np.mean(epoch2_base_amp1_opto_s, axis=0))                    
            axs1[10,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_base_amp1_opto_s , axis=0), yerr=stats.sem(epoch2_base_amp1_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_base_amp1_opto_s) > 0 :
            G_epoch3_base_amp1_opto_s.append(np.mean(epoch3_base_amp1_opto_s, axis=0))                    
            axs1[10,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_base_amp1_opto_s , axis=0), yerr=stats.sem(epoch3_base_amp1_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
            
        # [11,0]
        if len(epoch1_base_amp1_con_l) > 0 :
            G_epoch1_base_amp1_con_l.append(np.mean(epoch1_base_amp1_con_l, axis=0))                    
            axs1[11,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_base_amp1_con_l , axis=0), yerr=stats.sem(epoch1_base_amp1_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_base_amp1_con_l) > 0 :
            G_epoch2_base_amp1_con_l.append(np.mean(epoch2_base_amp1_con_l, axis=0))                    
            axs1[11,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_base_amp1_con_l , axis=0), yerr=stats.sem(epoch2_base_amp1_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_base_amp1_con_l) > 0 :
            G_epoch3_base_amp1_con_l.append(np.mean(epoch3_base_amp1_con_l, axis=0))                    
            axs1[11,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_base_amp1_con_l , axis=0), yerr=stats.sem(epoch3_base_amp1_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_base_amp1_chemo_l) > 0 :
            G_epoch1_base_amp1_chemo_l.append(np.mean(epoch1_base_amp1_chemo_l, axis=0))                    
            axs1[11,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_base_amp1_chemo_l , axis=0), yerr=stats.sem(epoch1_base_amp1_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_base_amp1_chemo_l) > 0 :
            G_epoch2_base_amp1_chemo_l.append(np.mean(epoch2_base_amp1_chemo_l, axis=0))                    
            axs1[11,0].errorbar(i + 1 , np.mean(epoch2_base_amp1_chemo_l , axis=0), yerr=stats.sem(epoch2_base_amp1_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_base_amp1_chemo_l) > 0 :
            G_epoch3_base_amp1_chemo_l.append(np.mean(epoch3_base_amp1_chemo_l, axis=0))                    
            axs1[11,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_base_amp1_chemo_l , axis=0), yerr=stats.sem(epoch3_base_amp1_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_base_amp1_opto_l) > 0 :
            G_epoch1_base_amp1_opto_l.append(np.mean(epoch1_base_amp1_opto_l, axis=0))                    
            axs1[11,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_base_amp1_opto_l , axis=0), yerr=stats.sem(epoch1_base_amp1_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_base_amp1_opto_l) > 0 :
            G_epoch2_base_amp1_opto_l.append(np.mean(epoch2_base_amp1_opto_l, axis=0))                    
            axs1[11,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_base_amp1_opto_l , axis=0), yerr=stats.sem(epoch2_base_amp1_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_base_amp1_opto_l) > 0 :
            G_epoch3_base_amp1_opto_l.append(np.mean(epoch3_base_amp1_opto_l, axis=0))                    
            axs1[11,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_base_amp1_opto_l , axis=0), yerr=stats.sem(epoch3_base_amp1_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        
        # [10,1]
        if len(epoch1_base_amp2_con_s) > 0 :
            G_epoch1_base_amp2_con_s.append(np.mean(epoch1_base_amp2_con_s, axis=0))                    
            axs1[10,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_base_amp2_con_s , axis=0), yerr=stats.sem(epoch1_base_amp2_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_base_amp2_con_s) > 0 :
            G_epoch2_base_amp2_con_s.append(np.mean(epoch2_base_amp2_con_s, axis=0))                    
            axs1[10,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_base_amp2_con_s , axis=0), yerr=stats.sem(epoch2_base_amp2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_base_amp2_con_s) > 0 :
            G_epoch3_base_amp2_con_s.append(np.mean(epoch3_base_amp2_con_s, axis=0))                    
            axs1[10,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_base_amp2_con_s , axis=0), yerr=stats.sem(epoch3_base_amp2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_base_amp2_chemo_s) > 0 :
            G_epoch1_base_amp2_chemo_s.append(np.mean(epoch1_base_amp2_chemo_s, axis=0))                    
            axs1[10,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_base_amp2_chemo_s , axis=0), yerr=stats.sem(epoch1_base_amp2_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_base_amp2_chemo_s) > 0 :
            G_epoch2_base_amp2_chemo_s.append(np.mean(epoch2_base_amp2_chemo_s, axis=0))                    
            axs1[10,1].errorbar(i + 1 , np.mean(epoch2_base_amp2_chemo_s , axis=0), yerr=stats.sem(epoch2_base_amp2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_base_amp2_chemo_s) > 0 :
            G_epoch3_base_amp2_chemo_s.append(np.mean(epoch3_base_amp2_chemo_s, axis=0))                    
            axs1[10,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_base_amp2_chemo_s , axis=0), yerr=stats.sem(epoch3_base_amp2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_base_amp2_opto_s) > 0 :
            G_epoch1_base_amp2_opto_s.append(np.mean(epoch1_base_amp2_opto_s, axis=0))                    
            axs1[10,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_base_amp2_opto_s , axis=0), yerr=stats.sem(epoch1_base_amp2_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_base_amp2_opto_s) > 0 :
            G_epoch2_base_amp2_opto_s.append(np.mean(epoch2_base_amp2_opto_s, axis=0))                    
            axs1[10,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_base_amp2_opto_s , axis=0), yerr=stats.sem(epoch2_base_amp2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_base_amp2_opto_s) > 0 :
            G_epoch3_base_amp2_opto_s.append(np.mean(epoch3_base_amp2_opto_s, axis=0))                    
            axs1[10,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_base_amp2_opto_s , axis=0), yerr=stats.sem(epoch3_base_amp2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        # [11,1]
        if len(epoch1_base_amp2_con_l) > 0 :
            G_epoch1_base_amp2_con_l.append(np.mean(epoch1_base_amp2_con_l, axis=0))                    
            axs1[11,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_base_amp2_con_l , axis=0), yerr=stats.sem(epoch1_base_amp2_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_base_amp2_con_l) > 0 :
            G_epoch2_base_amp2_con_l.append(np.mean(epoch2_base_amp2_con_l, axis=0))                    
            axs1[11,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_base_amp2_con_l , axis=0), yerr=stats.sem(epoch2_base_amp2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_base_amp2_con_l) > 0 :
            G_epoch3_base_amp2_con_l.append(np.mean(epoch3_base_amp2_con_l, axis=0))                    
            axs1[11,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_base_amp2_con_l , axis=0), yerr=stats.sem(epoch3_base_amp2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_base_amp2_chemo_l) > 0 :
            G_epoch1_base_amp2_chemo_l.append(np.mean(epoch1_base_amp2_chemo_l, axis=0))                    
            axs1[11,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_base_amp2_chemo_l , axis=0), yerr=stats.sem(epoch1_base_amp2_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_base_amp2_chemo_l) > 0 :
            G_epoch2_base_amp2_chemo_l.append(np.mean(epoch2_base_amp2_chemo_l, axis=0))                    
            axs1[11,1].errorbar(i + 1 , np.mean(epoch2_base_amp2_chemo_l , axis=0), yerr=stats.sem(epoch2_base_amp2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_base_amp2_chemo_l) > 0 :
            G_epoch3_base_amp2_chemo_l.append(np.mean(epoch3_base_amp2_chemo_l, axis=0))                    
            axs1[11,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_base_amp2_chemo_l , axis=0), yerr=stats.sem(epoch3_base_amp2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_base_amp2_opto_l) > 0 :
            G_epoch1_base_amp2_opto_l.append(np.mean(epoch1_base_amp2_opto_l, axis=0))                    
            axs1[11,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_base_amp2_opto_l , axis=0), yerr=stats.sem(epoch1_base_amp2_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_base_amp2_opto_l) > 0 :
            G_epoch2_base_amp2_opto_l.append(np.mean(epoch2_base_amp2_opto_l, axis=0))                    
            axs1[11,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_base_amp2_opto_l , axis=0), yerr=stats.sem(epoch2_base_amp2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_base_amp2_opto_l) > 0 :
            G_epoch3_base_amp2_opto_l.append(np.mean(epoch3_base_amp2_opto_l, axis=0))                    
            axs1[11,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_base_amp2_opto_l , axis=0), yerr=stats.sem(epoch3_base_amp2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        # [12,0]
        if len(epoch1_max_vel1_con_s) > 0 :
            G_epoch1_max_vel1_con_s.append(np.mean(epoch1_max_vel1_con_s, axis=0))                    
            axs1[12,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_max_vel1_con_s , axis=0), yerr=stats.sem(epoch1_max_vel1_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_max_vel1_con_s) > 0 :
            G_epoch2_max_vel1_con_s.append(np.mean(epoch2_max_vel1_con_s, axis=0))                    
            axs1[12,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_max_vel1_con_s , axis=0), yerr=stats.sem(epoch2_max_vel1_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_max_vel1_con_s) > 0 :
            G_epoch3_max_vel1_con_s.append(np.mean(epoch3_max_vel1_con_s, axis=0))                    
            axs1[12,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_max_vel1_con_s , axis=0), yerr=stats.sem(epoch3_max_vel1_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_max_vel1_chemo_s) > 0 :
            G_epoch1_max_vel1_chemo_s.append(np.mean(epoch1_max_vel1_chemo_s, axis=0))                    
            axs1[12,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_max_vel1_chemo_s , axis=0), yerr=stats.sem(epoch1_max_vel1_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_max_vel1_chemo_s) > 0 :
            G_epoch2_max_vel1_chemo_s.append(np.mean(epoch2_max_vel1_chemo_s, axis=0))                    
            axs1[12,0].errorbar(i + 1 , np.mean(epoch2_max_vel1_chemo_s , axis=0), yerr=stats.sem(epoch2_max_vel1_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_max_vel1_chemo_s) > 0 :
            G_epoch3_max_vel1_chemo_s.append(np.mean(epoch3_max_vel1_chemo_s, axis=0))                    
            axs1[12,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_max_vel1_chemo_s , axis=0), yerr=stats.sem(epoch3_max_vel1_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_max_vel1_opto_s) > 0 :
            G_epoch1_max_vel1_opto_s.append(np.mean(epoch1_max_vel1_opto_s, axis=0))                    
            axs1[12,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_max_vel1_opto_s , axis=0), yerr=stats.sem(epoch1_max_vel1_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_max_vel1_opto_s) > 0 :
            G_epoch2_max_vel1_opto_s.append(np.mean(epoch2_max_vel1_opto_s, axis=0))                    
            axs1[12,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_max_vel1_opto_s , axis=0), yerr=stats.sem(epoch2_max_vel1_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_max_vel1_opto_s) > 0 :
            G_epoch3_max_vel1_opto_s.append(np.mean(epoch3_max_vel1_opto_s, axis=0))                    
            axs1[12,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_max_vel1_opto_s , axis=0), yerr=stats.sem(epoch3_max_vel1_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
            
        # [13,0]
        if len(epoch1_max_vel1_con_l) > 0 :
            G_epoch1_max_vel1_con_l.append(np.mean(epoch1_max_vel1_con_l, axis=0))                    
            axs1[13,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_max_vel1_con_l , axis=0), yerr=stats.sem(epoch1_max_vel1_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_max_vel1_con_l) > 0 :
            G_epoch2_max_vel1_con_l.append(np.mean(epoch2_max_vel1_con_l, axis=0))                    
            axs1[13,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_max_vel1_con_l , axis=0), yerr=stats.sem(epoch2_max_vel1_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_max_vel1_con_l) > 0 :
            G_epoch3_max_vel1_con_l.append(np.mean(epoch3_max_vel1_con_l, axis=0))                    
            axs1[13,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_max_vel1_con_l , axis=0), yerr=stats.sem(epoch3_max_vel1_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_max_vel1_chemo_l) > 0 :
            G_epoch1_max_vel1_chemo_l.append(np.mean(epoch1_max_vel1_chemo_l, axis=0))                    
            axs1[13,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_max_vel1_chemo_l , axis=0), yerr=stats.sem(epoch1_max_vel1_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_max_vel1_chemo_l) > 0 :
            G_epoch2_max_vel1_chemo_l.append(np.mean(epoch2_max_vel1_chemo_l, axis=0))                    
            axs1[13,0].errorbar(i + 1 , np.mean(epoch2_max_vel1_chemo_l , axis=0), yerr=stats.sem(epoch2_max_vel1_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_max_vel1_chemo_l) > 0 :
            G_epoch3_max_vel1_chemo_l.append(np.mean(epoch3_max_vel1_chemo_l, axis=0))                    
            axs1[13,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_max_vel1_chemo_l , axis=0), yerr=stats.sem(epoch3_max_vel1_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_max_vel1_opto_l) > 0 :
            G_epoch1_max_vel1_opto_l.append(np.mean(epoch1_max_vel1_opto_l, axis=0))                    
            axs1[13,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_max_vel1_opto_l , axis=0), yerr=stats.sem(epoch1_max_vel1_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_max_vel1_opto_l) > 0 :
            G_epoch2_max_vel1_opto_l.append(np.mean(epoch2_max_vel1_opto_l, axis=0))                    
            axs1[13,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_max_vel1_opto_l , axis=0), yerr=stats.sem(epoch2_max_vel1_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_max_vel1_opto_l) > 0 :
            G_epoch3_max_vel1_opto_l.append(np.mean(epoch3_max_vel1_opto_l, axis=0))                    
            axs1[13,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_max_vel1_opto_l , axis=0), yerr=stats.sem(epoch3_max_vel1_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        
        # [12,1]
        if len(epoch1_max_vel2_con_s) > 0 :
            G_epoch1_max_vel2_con_s.append(np.mean(epoch1_max_vel2_con_s, axis=0))                    
            axs1[12,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_max_vel2_con_s , axis=0), yerr=stats.sem(epoch1_max_vel2_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_max_vel2_con_s) > 0 :
            G_epoch2_max_vel2_con_s.append(np.mean(epoch2_max_vel2_con_s, axis=0))                    
            axs1[12,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_max_vel2_con_s , axis=0), yerr=stats.sem(epoch2_max_vel2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_max_vel2_con_s) > 0 :
            G_epoch3_max_vel2_con_s.append(np.mean(epoch3_max_vel2_con_s, axis=0))                    
            axs1[12,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_max_vel2_con_s , axis=0), yerr=stats.sem(epoch3_max_vel2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_max_vel2_chemo_s) > 0 :
            G_epoch1_max_vel2_chemo_s.append(np.mean(epoch1_max_vel2_chemo_s, axis=0))                    
            axs1[12,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_max_vel2_chemo_s , axis=0), yerr=stats.sem(epoch1_max_vel2_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_max_vel2_chemo_s) > 0 :
            G_epoch2_max_vel2_chemo_s.append(np.mean(epoch2_max_vel2_chemo_s, axis=0))                    
            axs1[12,1].errorbar(i + 1 , np.mean(epoch2_max_vel2_chemo_s , axis=0), yerr=stats.sem(epoch2_max_vel2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_max_vel2_chemo_s) > 0 :
            G_epoch3_max_vel2_chemo_s.append(np.mean(epoch3_max_vel2_chemo_s, axis=0))                    
            axs1[12,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_max_vel2_chemo_s , axis=0), yerr=stats.sem(epoch3_max_vel2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_max_vel2_opto_s) > 0 :
            G_epoch1_max_vel2_opto_s.append(np.mean(epoch1_max_vel2_opto_s, axis=0))                    
            axs1[12,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_max_vel2_opto_s , axis=0), yerr=stats.sem(epoch1_max_vel2_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_max_vel2_opto_s) > 0 :
            G_epoch2_max_vel2_opto_s.append(np.mean(epoch2_max_vel2_opto_s, axis=0))                    
            axs1[12,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_max_vel2_opto_s , axis=0), yerr=stats.sem(epoch2_max_vel2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_max_vel2_opto_s) > 0 :
            G_epoch3_max_vel2_opto_s.append(np.mean(epoch3_max_vel2_opto_s, axis=0))                    
            axs1[12,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_max_vel2_opto_s , axis=0), yerr=stats.sem(epoch3_max_vel2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        # [13,1]
        if len(epoch1_max_vel2_con_l) > 0 :
            G_epoch1_max_vel2_con_l.append(np.mean(epoch1_max_vel2_con_l, axis=0))                    
            axs1[13,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_max_vel2_con_l , axis=0), yerr=stats.sem(epoch1_max_vel2_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_max_vel2_con_l) > 0 :
            G_epoch2_max_vel2_con_l.append(np.mean(epoch2_max_vel2_con_l, axis=0))                    
            axs1[13,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_max_vel2_con_l , axis=0), yerr=stats.sem(epoch2_max_vel2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_max_vel2_con_l) > 0 :
            G_epoch3_max_vel2_con_l.append(np.mean(epoch3_max_vel2_con_l, axis=0))                    
            axs1[13,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_max_vel2_con_l , axis=0), yerr=stats.sem(epoch3_max_vel2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_max_vel2_chemo_l) > 0 :
            G_epoch1_max_vel2_chemo_l.append(np.mean(epoch1_max_vel2_chemo_l, axis=0))                    
            axs1[13,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_max_vel2_chemo_l , axis=0), yerr=stats.sem(epoch1_max_vel2_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_max_vel2_chemo_l) > 0 :
            G_epoch2_max_vel2_chemo_l.append(np.mean(epoch2_max_vel2_chemo_l, axis=0))                    
            axs1[13,1].errorbar(i + 1 , np.mean(epoch2_max_vel2_chemo_l , axis=0), yerr=stats.sem(epoch2_max_vel2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_max_vel2_chemo_l) > 0 :
            G_epoch3_max_vel2_chemo_l.append(np.mean(epoch3_max_vel2_chemo_l, axis=0))                    
            axs1[13,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_max_vel2_chemo_l , axis=0), yerr=stats.sem(epoch3_max_vel2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_max_vel2_opto_l) > 0 :
            G_epoch1_max_vel2_opto_l.append(np.mean(epoch1_max_vel2_opto_l, axis=0))                    
            axs1[13,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_max_vel2_opto_l , axis=0), yerr=stats.sem(epoch1_max_vel2_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_max_vel2_opto_l) > 0 :
            G_epoch2_max_vel2_opto_l.append(np.mean(epoch2_max_vel2_opto_l, axis=0))                    
            axs1[13,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_max_vel2_opto_l , axis=0), yerr=stats.sem(epoch2_max_vel2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_max_vel2_opto_l) > 0 :
            G_epoch3_max_vel2_opto_l.append(np.mean(epoch3_max_vel2_opto_l, axis=0))                    
            axs1[13,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_max_vel2_opto_l , axis=0), yerr=stats.sem(epoch3_max_vel2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        # [14,0]
        if len(epoch1_velaton1_con_s) > 0 :
            G_epoch1_velaton1_con_s.append(np.mean(epoch1_velaton1_con_s, axis=0))                    
            axs1[14,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_velaton1_con_s , axis=0), yerr=stats.sem(epoch1_velaton1_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_velaton1_con_s) > 0 :
            G_epoch2_velaton1_con_s.append(np.mean(epoch2_velaton1_con_s, axis=0))                    
            axs1[14,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_velaton1_con_s , axis=0), yerr=stats.sem(epoch2_velaton1_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_velaton1_con_s) > 0 :
            G_epoch3_velaton1_con_s.append(np.mean(epoch3_velaton1_con_s, axis=0))                    
            axs1[14,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_velaton1_con_s , axis=0), yerr=stats.sem(epoch3_velaton1_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_velaton1_chemo_s) > 0 :
            G_epoch1_velaton1_chemo_s.append(np.mean(epoch1_velaton1_chemo_s, axis=0))                    
            axs1[14,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_velaton1_chemo_s , axis=0), yerr=stats.sem(epoch1_velaton1_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_velaton1_chemo_s) > 0 :
            G_epoch2_velaton1_chemo_s.append(np.mean(epoch2_velaton1_chemo_s, axis=0))                    
            axs1[14,0].errorbar(i + 1 , np.mean(epoch2_velaton1_chemo_s , axis=0), yerr=stats.sem(epoch2_velaton1_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_velaton1_chemo_s) > 0 :
            G_epoch3_velaton1_chemo_s.append(np.mean(epoch3_velaton1_chemo_s, axis=0))                    
            axs1[14,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_velaton1_chemo_s , axis=0), yerr=stats.sem(epoch3_velaton1_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_velaton1_opto_s) > 0 :
            G_epoch1_velaton1_opto_s.append(np.mean(epoch1_velaton1_opto_s, axis=0))                    
            axs1[14,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_velaton1_opto_s , axis=0), yerr=stats.sem(epoch1_velaton1_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_velaton1_opto_s) > 0 :
            G_epoch2_velaton1_opto_s.append(np.mean(epoch2_velaton1_opto_s, axis=0))                    
            axs1[14,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_velaton1_opto_s , axis=0), yerr=stats.sem(epoch2_velaton1_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_velaton1_opto_s) > 0 :
            G_epoch3_velaton1_opto_s.append(np.mean(epoch3_velaton1_opto_s, axis=0))                    
            axs1[14,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_velaton1_opto_s , axis=0), yerr=stats.sem(epoch3_velaton1_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
            
        # [15,0]
        if len(epoch1_velaton1_con_l) > 0 :
            G_epoch1_velaton1_con_l.append(np.mean(epoch1_velaton1_con_l, axis=0))                    
            axs1[15,0].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_velaton1_con_l , axis=0), yerr=stats.sem(epoch1_velaton1_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_velaton1_con_l) > 0 :
            G_epoch2_velaton1_con_l.append(np.mean(epoch2_velaton1_con_l, axis=0))                    
            axs1[15,0].errorbar(i + 1 - 2*offset, np.mean(epoch2_velaton1_con_l , axis=0), yerr=stats.sem(epoch2_velaton1_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_velaton1_con_l) > 0 :
            G_epoch3_velaton1_con_l.append(np.mean(epoch3_velaton1_con_l, axis=0))                    
            axs1[15,0].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_velaton1_con_l , axis=0), yerr=stats.sem(epoch3_velaton1_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_velaton1_chemo_l) > 0 :
            G_epoch1_velaton1_chemo_l.append(np.mean(epoch1_velaton1_chemo_l, axis=0))                    
            axs1[15,0].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_velaton1_chemo_l , axis=0), yerr=stats.sem(epoch1_velaton1_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_velaton1_chemo_l) > 0 :
            G_epoch2_velaton1_chemo_l.append(np.mean(epoch2_velaton1_chemo_l, axis=0))                    
            axs1[15,0].errorbar(i + 1 , np.mean(epoch2_velaton1_chemo_l , axis=0), yerr=stats.sem(epoch2_velaton1_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_velaton1_chemo_l) > 0 :
            G_epoch3_velaton1_chemo_l.append(np.mean(epoch3_velaton1_chemo_l, axis=0))                    
            axs1[15,0].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_velaton1_chemo_l , axis=0), yerr=stats.sem(epoch3_velaton1_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_velaton1_opto_l) > 0 :
            G_epoch1_velaton1_opto_l.append(np.mean(epoch1_velaton1_opto_l, axis=0))                    
            axs1[15,0].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_velaton1_opto_l , axis=0), yerr=stats.sem(epoch1_velaton1_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_velaton1_opto_l) > 0 :
            G_epoch2_velaton1_opto_l.append(np.mean(epoch2_velaton1_opto_l, axis=0))                    
            axs1[15,0].errorbar(i + 1 + 2*offset, np.mean(epoch2_velaton1_opto_l , axis=0), yerr=stats.sem(epoch2_velaton1_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_velaton1_opto_l) > 0 :
            G_epoch3_velaton1_opto_l.append(np.mean(epoch3_velaton1_opto_l, axis=0))                    
            axs1[15,0].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_velaton1_opto_l , axis=0), yerr=stats.sem(epoch3_velaton1_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        
        # [14,1]
        if len(epoch1_velaton2_con_s) > 0 :
            G_epoch1_velaton2_con_s.append(np.mean(epoch1_velaton2_con_s, axis=0))                    
            axs1[14,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_velaton2_con_s , axis=0), yerr=stats.sem(epoch1_velaton2_con_s, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_velaton2_con_s) > 0 :
            G_epoch2_velaton2_con_s.append(np.mean(epoch2_velaton2_con_s, axis=0))                    
            axs1[14,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_velaton2_con_s , axis=0), yerr=stats.sem(epoch2_velaton2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_velaton2_con_s) > 0 :
            G_epoch3_velaton2_con_s.append(np.mean(epoch3_velaton2_con_s, axis=0))                    
            axs1[14,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_velaton2_con_s , axis=0), yerr=stats.sem(epoch3_velaton2_con_s, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_velaton2_chemo_s) > 0 :
            G_epoch1_velaton2_chemo_s.append(np.mean(epoch1_velaton2_chemo_s, axis=0))                    
            axs1[14,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_velaton2_chemo_s , axis=0), yerr=stats.sem(epoch1_velaton2_chemo_s, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_velaton2_chemo_s) > 0 :
            G_epoch2_velaton2_chemo_s.append(np.mean(epoch2_velaton2_chemo_s, axis=0))                    
            axs1[14,1].errorbar(i + 1 , np.mean(epoch2_velaton2_chemo_s , axis=0), yerr=stats.sem(epoch2_velaton2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_velaton2_chemo_s) > 0 :
            G_epoch3_velaton2_chemo_s.append(np.mean(epoch3_velaton2_chemo_s, axis=0))                    
            axs1[14,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_velaton2_chemo_s , axis=0), yerr=stats.sem(epoch3_velaton2_chemo_s, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_velaton2_opto_s) > 0 :
            G_epoch1_velaton2_opto_s.append(np.mean(epoch1_velaton2_opto_s, axis=0))                    
            axs1[14,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_velaton2_opto_s , axis=0), yerr=stats.sem(epoch1_velaton2_opto_s, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_velaton2_opto_s) > 0 :
            G_epoch2_velaton2_opto_s.append(np.mean(epoch2_velaton2_opto_s, axis=0))                    
            axs1[14,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_velaton2_opto_s , axis=0), yerr=stats.sem(epoch2_velaton2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_velaton2_opto_s) > 0 :
            G_epoch3_velaton2_opto_s.append(np.mean(epoch3_velaton2_opto_s, axis=0))                    
            axs1[14,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_velaton2_opto_s , axis=0), yerr=stats.sem(epoch3_velaton2_opto_s, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])
        # [15,1]
        if len(epoch1_velaton2_con_l) > 0 :
            G_epoch1_velaton2_con_l.append(np.mean(epoch1_velaton2_con_l, axis=0))                    
            axs1[15,1].errorbar(i + 1 - 2.5*offset, np.mean(epoch1_velaton2_con_l , axis=0), yerr=stats.sem(epoch1_velaton2_con_l, axis=0), fmt='o', capsize=4, color = black_shades[0])
        if len(epoch2_velaton2_con_l) > 0 :
            G_epoch2_velaton2_con_l.append(np.mean(epoch2_velaton2_con_l, axis=0))                    
            axs1[15,1].errorbar(i + 1 - 2*offset, np.mean(epoch2_velaton2_con_l , axis=0), yerr=stats.sem(epoch2_velaton2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[1])
        if len(epoch3_velaton2_con_l) > 0 :
            G_epoch3_velaton2_con_l.append(np.mean(epoch3_velaton2_con_l, axis=0))                    
            axs1[15,1].errorbar(i + 1 - 1.5*offset, np.mean(epoch3_velaton2_con_l , axis=0), yerr=stats.sem(epoch3_velaton2_con_l, axis = 0), fmt='o', capsize=4, color = black_shades[2])
        
        if len(epoch1_velaton2_chemo_l) > 0 :
            G_epoch1_velaton2_chemo_l.append(np.mean(epoch1_velaton2_chemo_l, axis=0))                    
            axs1[15,1].errorbar(i + 1 - 0.5*offset, np.mean(epoch1_velaton2_chemo_l , axis=0), yerr=stats.sem(epoch1_velaton2_chemo_l, axis=0), fmt='o', capsize=4, color = red_shades[0])
        if len(epoch2_velaton2_chemo_l) > 0 :
            G_epoch2_velaton2_chemo_l.append(np.mean(epoch2_velaton2_chemo_l, axis=0))                    
            axs1[15,1].errorbar(i + 1 , np.mean(epoch2_velaton2_chemo_l , axis=0), yerr=stats.sem(epoch2_velaton2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[1])
        if len(epoch3_velaton2_chemo_l) > 0 :
            G_epoch3_velaton2_chemo_l.append(np.mean(epoch3_velaton2_chemo_l, axis=0))                    
            axs1[15,1].errorbar(i + 1 + 0.5*offset, np.mean(epoch3_velaton2_chemo_l , axis=0), yerr=stats.sem(epoch3_velaton2_chemo_l, axis = 0), fmt='o', capsize=4, color = red_shades[2])
        
        if len(epoch1_velaton2_opto_l) > 0 :
            G_epoch1_velaton2_opto_l.append(np.mean(epoch1_velaton2_opto_l, axis=0))                    
            axs1[15,1].errorbar(i + 1 + 1.5*offset, np.mean(epoch1_velaton2_opto_l , axis=0), yerr=stats.sem(epoch1_velaton2_opto_l, axis=0), fmt='o', capsize=4, color = skyblue_shades[0])
        if len(epoch2_velaton2_opto_l) > 0 :
            G_epoch2_velaton2_opto_l.append(np.mean(epoch2_velaton2_opto_l, axis=0))                    
            axs1[15,1].errorbar(i + 1 + 2*offset, np.mean(epoch2_velaton2_opto_l , axis=0), yerr=stats.sem(epoch2_velaton2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[1])
        if len(epoch3_velaton2_opto_l) > 0 :
            G_epoch3_velaton2_opto_l.append(np.mean(epoch3_velaton2_opto_l, axis=0))                    
            axs1[15,1].errorbar(i + 1 + 2.5*offset, np.mean(epoch3_velaton2_opto_l , axis=0), yerr=stats.sem(epoch3_velaton2_opto_l, axis = 0), fmt='o', capsize=4, color = skyblue_shades[2])

    # Create custom legend with markers
    legend_elements = [
        Line2D([0], [0], color=black_shades[0], marker='o', markersize=8, markeredgewidth=2, linestyle='-', lw=2, label='Control_Epoch1'),
        Line2D([0], [0], color=black_shades[1], marker='o', markersize=8, markeredgewidth=2, linestyle='-', lw=2, label='Control_Epoch2'),
        Line2D([0], [0], color=black_shades[2], marker='o', markersize=8, markeredgewidth=2, linestyle='-', lw=2, label='Control_Epoch3'),
        Line2D([0], [0], color=red_shades[0], marker='o', markersize=8, markeredgewidth=2, linestyle='-', lw=2, label='Chemo_Epoch1'),
        Line2D([0], [0], color=red_shades[1], marker='o', markersize=8, markeredgewidth=2, linestyle='-', lw=2, label='Chemo_Epoch2'),
        Line2D([0], [0], color=red_shades[2], marker='o', markersize=8, markeredgewidth=2, linestyle='-', lw=2, label='Chemo_Epoch3'),
        Line2D([0], [0], color=skyblue_shades[0], marker='o', markersize=8, markeredgewidth=2, linestyle='-', lw=2, label='Opto_Epoch1'),
        Line2D([0], [0], color=skyblue_shades[1], marker='o', markersize=8, markeredgewidth=2, linestyle='-', lw=2, label='Opto_Epoch2'),
        Line2D([0], [0], color=skyblue_shades[2], marker='o', markersize=8, markeredgewidth=2, linestyle='-', lw=2, label='Opto_Epoch3')
    ]
    fig1.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5,1))
    fig1.tight_layout()

    ############ POOLED ############################################################
    print('########### POOLED ##########')
    offset = 0.1
    session_id = np.arange(2) + 1
    dates = ['Short','Long']
    fig2, axs2 = plt.subplots(nrows=7, ncols=2, figsize=(16 , 35))
    fig2.subplots_adjust(hspace=0.7)
    fig2.suptitle(subject + '\n Time, Amplitude and Kinematic Quatification Pooled Session\n')
    # Vis to Onset
    axs2[0,0].set_title(3*'\n' + 'Vis1 to Onset1\n') 
    axs2[0,0].spines['right'].set_visible(False)
    axs2[0,0].spines['top'].set_visible(False)
    axs2[0,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs2[0,0].set_xticks(session_id)
    axs2[0,0].set_xticklabels(dates, ha = 'center')

    if any(0 in row for row in isSelfTimedMode):
        axs2[0,1].set_title('Vis2 to Onset2\n') 
    else:
        axs2[0,1].set_title('WaitforPress2 to Onset2\n') 
    axs2[0,1].spines['right'].set_visible(False)
    axs2[0,1].spines['top'].set_visible(False)
    axs2[0,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs2[0,1].set_xticks(session_id)
    axs2[0,1].set_xticklabels(dates, ha = 'center')
    # Onset to Target
    axs2[1,0].set_title('Onset1 to Target1\n') 
    axs2[1,0].spines['right'].set_visible(False)
    axs2[1,0].spines['top'].set_visible(False)
    axs2[1,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs2[1,0].set_xticks(session_id)
    axs2[1,0].set_xticklabels(dates, ha = 'center')

    axs2[1,1].set_title('Onset2 to Target2\n') 
    axs2[1,1].spines['right'].set_visible(False)
    axs2[1,1].spines['top'].set_visible(False)
    axs2[1,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs2[1,1].set_xticks(session_id)
    axs2[1,1].set_xticklabels(dates, ha = 'center')
    # Onset to peak time
    axs2[2,0].set_title('Onset1 to Peak1\n') 
    axs2[2,0].spines['right'].set_visible(False)
    axs2[2,0].spines['top'].set_visible(False)
    axs2[2,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs2[2,0].set_xticks(session_id)
    axs2[2,0].set_xticklabels(dates, ha = 'center')

    axs2[2,1].set_title('Onset2 to Peak2\n') 
    axs2[2,1].spines['right'].set_visible(False)
    axs2[2,1].spines['top'].set_visible(False)
    axs2[2,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs2[2,1].set_xticks(session_id)
    axs2[2,1].set_xticklabels(dates, ha = 'center')
    # Onset Interval
    axs2[3,0].set_title('Onset1 and Onset2 Interval\n') 
    axs2[3,0].spines['right'].set_visible(False)
    axs2[3,0].spines['top'].set_visible(False)
    axs2[3,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs2[3,0].set_xticks(session_id)
    axs2[3,0].set_xticklabels(dates, ha = 'center')

    axs2[3,1].set_title('LeverRetract1_End to Onset2\n') 
    axs2[3,1].spines['right'].set_visible(False)
    axs2[3,1].spines['top'].set_visible(False)
    axs2[3,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs2[3,1].set_xticks(session_id)
    axs2[3,1].set_xticklabels(dates, ha = 'center')
    # Baseline to Peak
    axs2[4,0].set_title('Baseline to Peak1\n') 
    axs2[4,0].spines['right'].set_visible(False)
    axs2[4,0].spines['top'].set_visible(False)
    axs2[4,0].set_ylabel('Lever Deflection(deg) Mean +/- SEM')
    axs2[4,0].set_xticks(session_id)
    axs2[4,0].set_xticklabels(dates, ha = 'center')

    axs2[4,1].set_title('Baseline to Peak2\n') 
    axs2[4,1].spines['right'].set_visible(False)
    axs2[4,1].spines['top'].set_visible(False)
    axs2[4,1].set_ylabel('Lever Deflection(deg) Mean +/- SEM')
    axs2[4,1].set_xticks(session_id)
    axs2[4,1].set_xticklabels(dates, ha = 'center')
    # Max Velocity Press
    axs2[5,0].set_title('Max Velocity Push1\n') 
    axs2[5,0].spines['right'].set_visible(False)
    axs2[5,0].spines['top'].set_visible(False)
    axs2[5,0].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs2[5,0].set_xticks(session_id)
    axs2[5,0].set_xticklabels(dates, ha = 'center')

    axs2[5,1].set_title('Max Velocity Push2\n') 
    axs2[5,1].spines['right'].set_visible(False)
    axs2[5,1].spines['top'].set_visible(False)
    axs2[5,1].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs2[5,1].set_xticks(session_id)
    axs2[5,1].set_xticklabels(dates, ha = 'center')
    # Velocity at Onset
    axs2[6,0].set_title('Velocity at Onset1\n') 
    axs2[6,0].spines['right'].set_visible(False)
    axs2[6,0].spines['top'].set_visible(False)
    axs2[6,0].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs2[6,0].set_xticks(session_id)
    axs2[6,0].set_xticklabels(dates, ha = 'center')

    axs2[6,1].set_title('Velocity at Onset2\n') 
    axs2[6,1].spines['right'].set_visible(False)
    axs2[6,1].spines['top'].set_visible(False)
    axs2[6,1].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs2[6,1].set_xticks(session_id)
    axs2[6,1].set_xticklabels(dates, ha = 'center')

    # Plotting
    # [0,0]
    if len(p_epoch1_vis1_on1_con_s) > 0:
        axs2[0,0].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_vis1_on1_con_s,axis=0) , yerr = stats.sem(p_epoch1_vis1_on1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_vis1_on1_con_s) > 0:
        axs2[0,0].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_vis1_on1_con_s,axis=0) , yerr = stats.sem(p_epoch2_vis1_on1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_vis1_on1_con_s) > 0:
        axs2[0,0].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_vis1_on1_con_s,axis=0) , yerr = stats.sem(p_epoch3_vis1_on1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_vis1_on1_chemo_s) > 0:
        axs2[0,0].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_vis1_on1_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_vis1_on1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_vis1_on1_chemo_s) > 0:
        axs2[0,0].errorbar(session_id[0], np.mean(p_epoch2_vis1_on1_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_vis1_on1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_vis1_on1_chemo_s) > 0:
        axs2[0,0].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_vis1_on1_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_vis1_on1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_vis1_on1_opto_s) > 0:
        axs2[0,0].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_vis1_on1_opto_s,axis=0) , yerr = stats.sem(p_epoch1_vis1_on1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_vis1_on1_opto_s) > 0:
        axs2[0,0].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_vis1_on1_opto_s,axis=0) , yerr = stats.sem(p_epoch2_vis1_on1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_vis1_on1_opto_s) > 0:
        axs2[0,0].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_vis1_on1_opto_s,axis=0) , yerr = stats.sem(p_epoch3_vis1_on1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_vis1_on1_con_l) > 0:
        axs2[0,0].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_vis1_on1_con_l,axis=0) , yerr = stats.sem(p_epoch1_vis1_on1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_vis1_on1_con_l) > 0:
        axs2[0,0].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_vis1_on1_con_l,axis=0) , yerr = stats.sem(p_epoch2_vis1_on1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_vis1_on1_con_l) > 0:
        axs2[0,0].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_vis1_on1_con_l,axis=0) , yerr = stats.sem(p_epoch3_vis1_on1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_vis1_on1_chemo_l) > 0:
        axs2[0,0].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_vis1_on1_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_vis1_on1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_vis1_on1_chemo_l) > 0:
        axs2[0,0].errorbar(session_id[1], np.mean(p_epoch2_vis1_on1_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_vis1_on1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_vis1_on1_chemo_l) > 0:
        axs2[0,0].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_vis1_on1_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_vis1_on1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_vis1_on1_opto_l) > 0:
        axs2[0,0].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_vis1_on1_opto_l,axis=0) , yerr = stats.sem(p_epoch1_vis1_on1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_vis1_on1_opto_l) > 0:
        axs2[0,0].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_vis1_on1_opto_l,axis=0) , yerr = stats.sem(p_epoch2_vis1_on1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_vis1_on1_opto_l) > 0:
        axs2[0,0].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_vis1_on1_opto_l,axis=0) , yerr = stats.sem(p_epoch3_vis1_on1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [0,1]
    if len(p_epoch1_vis2_on2_con_s) > 0:
        axs2[0,1].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_vis2_on2_con_s,axis=0) , yerr = stats.sem(p_epoch1_vis2_on2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_vis2_on2_con_s) > 0:
        axs2[0,1].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_vis2_on2_con_s,axis=0) , yerr = stats.sem(p_epoch2_vis2_on2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_vis2_on2_con_s) > 0:
        axs2[0,1].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_vis2_on2_con_s,axis=0) , yerr = stats.sem(p_epoch3_vis2_on2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_vis2_on2_chemo_s) > 0:
        axs2[0,1].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_vis2_on2_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_vis2_on2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_vis2_on2_chemo_s) > 0:
        axs2[0,1].errorbar(session_id[0], np.mean(p_epoch2_vis2_on2_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_vis2_on2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_vis2_on2_chemo_s) > 0:
        axs2[0,1].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_vis2_on2_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_vis2_on2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_vis2_on2_opto_s) > 0:
        axs2[0,1].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_vis2_on2_opto_s,axis=0) , yerr = stats.sem(p_epoch1_vis2_on2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(p_epoch2_vis2_on2_opto_s) > 0:
        axs2[0,1].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_vis2_on2_opto_s,axis=0) , yerr = stats.sem(p_epoch2_vis2_on2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(p_epoch3_vis2_on2_opto_s) > 0:
        axs2[0,1].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_vis2_on2_opto_s,axis=0) , yerr = stats.sem(p_epoch3_vis2_on2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(p_epoch1_vis2_on2_con_l) > 0:
        axs2[0,1].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_vis2_on2_con_l,axis=0) , yerr = stats.sem(p_epoch1_vis2_on2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_vis2_on2_con_l) > 0:
        axs2[0,1].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_vis2_on2_con_l,axis=0) , yerr = stats.sem(p_epoch2_vis2_on2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_vis2_on2_con_l) > 0:
        axs2[0,1].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_vis2_on2_con_l,axis=0) , yerr = stats.sem(p_epoch3_vis2_on2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_vis2_on2_chemo_l) > 0:
        axs2[0,1].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_vis2_on2_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_vis2_on2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_vis2_on2_chemo_l) > 0:
        axs2[0,1].errorbar(session_id[1], np.mean(p_epoch2_vis2_on2_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_vis2_on2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_vis2_on2_chemo_l) > 0:
        axs2[0,1].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_vis2_on2_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_vis2_on2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_vis2_on2_opto_l) > 0:
        axs2[0,1].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_vis2_on2_opto_l,axis=0) , yerr = stats.sem(p_epoch1_vis2_on2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_vis2_on2_opto_l) > 0:
        axs2[0,1].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_vis2_on2_opto_l,axis=0) , yerr = stats.sem(p_epoch2_vis2_on2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_vis2_on2_opto_l) > 0:
        axs2[0,1].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_vis2_on2_opto_l,axis=0) , yerr = stats.sem(p_epoch3_vis2_on2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [1,0]
    if len(p_epoch1_on1_targ1_con_s) > 0:
        axs2[1,0].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_on1_targ1_con_s,axis=0) , yerr = stats.sem(p_epoch1_on1_targ1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_on1_targ1_con_s) > 0:
        axs2[1,0].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_on1_targ1_con_s,axis=0) , yerr = stats.sem(p_epoch2_on1_targ1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_on1_targ1_con_s) > 0:
        axs2[1,0].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_on1_targ1_con_s,axis=0) , yerr = stats.sem(p_epoch3_on1_targ1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_on1_targ1_chemo_s) > 0:
        axs2[1,0].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_on1_targ1_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_on1_targ1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_on1_targ1_chemo_s) > 0:
        axs2[1,0].errorbar(session_id[0], np.mean(p_epoch2_on1_targ1_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_on1_targ1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_on1_targ1_chemo_s) > 0:
        axs2[1,0].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_on1_targ1_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_on1_targ1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_on1_targ1_opto_s) > 0:
        axs2[1,0].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_on1_targ1_opto_s,axis=0) , yerr = stats.sem(p_epoch1_on1_targ1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(p_epoch2_on1_targ1_opto_s) > 0:
        axs2[1,0].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_on1_targ1_opto_s,axis=0) , yerr = stats.sem(p_epoch2_on1_targ1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(p_epoch3_on1_targ1_opto_s) > 0:
        axs2[1,0].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_on1_targ1_opto_s,axis=0) , yerr = stats.sem(p_epoch3_on1_targ1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(p_epoch1_on1_targ1_con_l) > 0:
        axs2[1,0].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_on1_targ1_con_l,axis=0) , yerr = stats.sem(p_epoch1_on1_targ1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_on1_targ1_con_l) > 0:
        axs2[1,0].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_on1_targ1_con_l,axis=0) , yerr = stats.sem(p_epoch2_on1_targ1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_on1_targ1_con_l) > 0:
        axs2[1,0].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_on1_targ1_con_l,axis=0) , yerr = stats.sem(p_epoch3_on1_targ1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_on1_targ1_chemo_l) > 0:
        axs2[1,0].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_on1_targ1_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_on1_targ1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_on1_targ1_chemo_l) > 0:
        axs2[1,0].errorbar(session_id[1], np.mean(p_epoch2_on1_targ1_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_on1_targ1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_on1_targ1_chemo_l) > 0:
        axs2[1,0].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_on1_targ1_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_on1_targ1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_on1_targ1_opto_l) > 0:
        axs2[1,0].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_on1_targ1_opto_l,axis=0) , yerr = stats.sem(p_epoch1_on1_targ1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_on1_targ1_opto_l) > 0:
        axs2[1,0].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_on1_targ1_opto_l,axis=0) , yerr = stats.sem(p_epoch2_on1_targ1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_on1_targ1_opto_l) > 0:
        axs2[1,0].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_on1_targ1_opto_l,axis=0) , yerr = stats.sem(p_epoch3_on1_targ1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [1,1]
    if len(p_epoch1_on2_targ2_con_s) > 0:
        axs2[1,1].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_on2_targ2_con_s,axis=0) , yerr = stats.sem(p_epoch1_on2_targ2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_on2_targ2_con_s) > 0:
        axs2[1,1].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_on2_targ2_con_s,axis=0) , yerr = stats.sem(p_epoch2_on2_targ2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_on2_targ2_con_s) > 0:
        axs2[1,1].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_on2_targ2_con_s,axis=0) , yerr = stats.sem(p_epoch3_on2_targ2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_on2_targ2_chemo_s) > 0:
        axs2[1,1].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_on2_targ2_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_on2_targ2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_on2_targ2_chemo_s) > 0:
        axs2[1,1].errorbar(session_id[0], np.mean(p_epoch2_on2_targ2_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_on2_targ2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_on2_targ2_chemo_s) > 0:
        axs2[1,1].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_on2_targ2_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_on2_targ2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_on2_targ2_opto_s) > 0:
        axs2[1,1].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_on2_targ2_opto_s,axis=0) , yerr = stats.sem(p_epoch1_on2_targ2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(p_epoch2_on2_targ2_opto_s) > 0:
        axs2[1,1].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_on2_targ2_opto_s,axis=0) , yerr = stats.sem(p_epoch2_on2_targ2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(p_epoch3_on2_targ2_opto_s) > 0:
        axs2[1,1].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_on2_targ2_opto_s,axis=0) , yerr = stats.sem(p_epoch3_on2_targ2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(p_epoch1_on2_targ2_con_l) > 0:
        axs2[1,1].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_on2_targ2_con_l,axis=0) , yerr = stats.sem(p_epoch1_on2_targ2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_on2_targ2_con_l) > 0:
        axs2[1,1].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_on2_targ2_con_l,axis=0) , yerr = stats.sem(p_epoch2_on2_targ2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_on2_targ2_con_l) > 0:
        axs2[1,1].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_on2_targ2_con_l,axis=0) , yerr = stats.sem(p_epoch3_on2_targ2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_on2_targ2_chemo_l) > 0:
        axs2[1,1].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_on2_targ2_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_on2_targ2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_on2_targ2_chemo_l) > 0:
        axs2[1,1].errorbar(session_id[1], np.mean(p_epoch2_on2_targ2_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_on2_targ2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_on2_targ2_chemo_l) > 0:
        axs2[1,1].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_on2_targ2_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_on2_targ2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_on2_targ2_opto_l) > 0:
        axs2[1,1].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_on2_targ2_opto_l,axis=0) , yerr = stats.sem(p_epoch1_on2_targ2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_on2_targ2_opto_l) > 0:
        axs2[1,1].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_on2_targ2_opto_l,axis=0) , yerr = stats.sem(p_epoch2_on2_targ2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_on2_targ2_opto_l) > 0:
        axs2[1,1].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_on2_targ2_opto_l,axis=0) , yerr = stats.sem(p_epoch3_on2_targ2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])

    # [2,0]
    if len(p_epoch1_on1_amp1_con_s) > 0:
        axs2[2,0].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_on1_amp1_con_s,axis=0) , yerr = stats.sem(p_epoch1_on1_amp1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_on1_amp1_con_s) > 0:
        axs2[2,0].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_on1_amp1_con_s,axis=0) , yerr = stats.sem(p_epoch2_on1_amp1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_on1_amp1_con_s) > 0:
        axs2[2,0].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_on1_amp1_con_s,axis=0) , yerr = stats.sem(p_epoch3_on1_amp1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_on1_amp1_chemo_s) > 0:
        axs2[2,0].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_on1_amp1_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_on1_amp1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_on1_amp1_chemo_s) > 0:
        axs2[2,0].errorbar(session_id[0], np.mean(p_epoch2_on1_amp1_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_on1_amp1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_on1_amp1_chemo_s) > 0:
        axs2[2,0].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_on1_amp1_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_on1_amp1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_on1_amp1_opto_s) > 0:
        axs2[2,0].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_on1_amp1_opto_s,axis=0) , yerr = stats.sem(p_epoch1_on1_amp1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(p_epoch2_on1_amp1_opto_s) > 0:
        axs2[2,0].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_on1_amp1_opto_s,axis=0) , yerr = stats.sem(p_epoch2_on1_amp1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(p_epoch3_on1_amp1_opto_s) > 0:
        axs2[2,0].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_on1_amp1_opto_s,axis=0) , yerr = stats.sem(p_epoch3_on1_amp1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(p_epoch1_on1_amp1_con_l) > 0:
        axs2[2,0].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_on1_amp1_con_l,axis=0) , yerr = stats.sem(p_epoch1_on1_amp1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_on1_amp1_con_l) > 0:
        axs2[2,0].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_on1_amp1_con_l,axis=0) , yerr = stats.sem(p_epoch2_on1_amp1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_on1_amp1_con_l) > 0:
        axs2[2,0].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_on1_amp1_con_l,axis=0) , yerr = stats.sem(p_epoch3_on1_amp1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_on1_amp1_chemo_l) > 0:
        axs2[2,0].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_on1_amp1_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_on1_amp1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_on1_amp1_chemo_l) > 0:
        axs2[2,0].errorbar(session_id[1], np.mean(p_epoch2_on1_amp1_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_on1_amp1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_on1_amp1_chemo_l) > 0:
        axs2[2,0].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_on1_amp1_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_on1_amp1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_on1_amp1_opto_l) > 0:
        axs2[2,0].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_on1_amp1_opto_l,axis=0) , yerr = stats.sem(p_epoch1_on1_amp1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_on1_amp1_opto_l) > 0:
        axs2[2,0].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_on1_amp1_opto_l,axis=0) , yerr = stats.sem(p_epoch2_on1_amp1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_on1_amp1_opto_l) > 0:
        axs2[2,0].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_on1_amp1_opto_l,axis=0) , yerr = stats.sem(p_epoch3_on1_amp1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [2,1]
    if len(p_epoch1_on2_amp2_con_s) > 0:
        axs2[2,1].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_on2_amp2_con_s,axis=0) , yerr = stats.sem(p_epoch1_on2_amp2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_on2_amp2_con_s) > 0:
        axs2[2,1].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_on2_amp2_con_s,axis=0) , yerr = stats.sem(p_epoch2_on2_amp2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_on2_amp2_con_s) > 0:
        axs2[2,1].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_on2_amp2_con_s,axis=0) , yerr = stats.sem(p_epoch3_on2_amp2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_on2_amp2_chemo_s) > 0:
        axs2[2,1].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_on2_amp2_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_on2_amp2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_on2_amp2_chemo_s) > 0:
        axs2[2,1].errorbar(session_id[0], np.mean(p_epoch2_on2_amp2_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_on2_amp2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_on2_amp2_chemo_s) > 0:
        axs2[2,1].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_on2_amp2_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_on2_amp2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_on2_amp2_opto_s) > 0:
        axs2[2,1].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_on2_amp2_opto_s,axis=0) , yerr = stats.sem(p_epoch1_on2_amp2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(p_epoch2_on2_amp2_opto_s) > 0:
        axs2[2,1].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_on2_amp2_opto_s,axis=0) , yerr = stats.sem(p_epoch2_on2_amp2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(p_epoch3_on2_amp2_opto_s) > 0:
        axs2[2,1].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_on2_amp2_opto_s,axis=0) , yerr = stats.sem(p_epoch3_on2_amp2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(p_epoch1_on2_amp2_con_l) > 0:
        axs2[2,1].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_on2_amp2_con_l,axis=0) , yerr = stats.sem(p_epoch1_on2_amp2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_on2_amp2_con_l) > 0:
        axs2[2,1].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_on2_amp2_con_l,axis=0) , yerr = stats.sem(p_epoch2_on2_amp2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_on2_amp2_con_l) > 0:
        axs2[2,1].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_on2_amp2_con_l,axis=0) , yerr = stats.sem(p_epoch3_on2_amp2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_on2_amp2_chemo_l) > 0:
        axs2[2,1].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_on2_amp2_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_on2_amp2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_on2_amp2_chemo_l) > 0:
        axs2[2,1].errorbar(session_id[1], np.mean(p_epoch2_on2_amp2_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_on2_amp2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_on2_amp2_chemo_l) > 0:
        axs2[2,1].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_on2_amp2_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_on2_amp2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_on2_amp2_opto_l) > 0:
        axs2[2,1].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_on2_amp2_opto_l,axis=0) , yerr = stats.sem(p_epoch1_on2_amp2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_on2_amp2_opto_l) > 0:
        axs2[2,1].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_on2_amp2_opto_l,axis=0) , yerr = stats.sem(p_epoch2_on2_amp2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_on2_amp2_opto_l) > 0:
        axs2[2,1].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_on2_amp2_opto_l,axis=0) , yerr = stats.sem(p_epoch3_on2_amp2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [3,0]
    if len(p_epoch1_on_interval_con_s) > 0:
        axs2[3,0].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_on_interval_con_s,axis=0) , yerr = stats.sem(p_epoch1_on_interval_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_on_interval_con_s) > 0:
        axs2[3,0].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_on_interval_con_s,axis=0) , yerr = stats.sem(p_epoch2_on_interval_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_on_interval_con_s) > 0:
        axs2[3,0].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_on_interval_con_s,axis=0) , yerr = stats.sem(p_epoch3_on_interval_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_on_interval_chemo_s) > 0:
        axs2[3,0].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_on_interval_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_on_interval_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_on_interval_chemo_s) > 0:
        axs2[3,0].errorbar(session_id[0], np.mean(p_epoch2_on_interval_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_on_interval_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_on_interval_chemo_s) > 0:
        axs2[3,0].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_on_interval_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_on_interval_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_on_interval_opto_s) > 0:
        axs2[3,0].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_on_interval_opto_s,axis=0) , yerr = stats.sem(p_epoch1_on_interval_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(p_epoch2_on_interval_opto_s) > 0:
        axs2[3,0].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_on_interval_opto_s,axis=0) , yerr = stats.sem(p_epoch2_on_interval_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(p_epoch3_on_interval_opto_s) > 0:
        axs2[3,0].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_on_interval_opto_s,axis=0) , yerr = stats.sem(p_epoch3_on_interval_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(p_epoch1_on_interval_con_l) > 0:
        axs2[3,0].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_on_interval_con_l,axis=0) , yerr = stats.sem(p_epoch1_on_interval_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_on_interval_con_l) > 0:
        axs2[3,0].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_on_interval_con_l,axis=0) , yerr = stats.sem(p_epoch2_on_interval_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_on_interval_con_l) > 0:
        axs2[3,0].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_on_interval_con_l,axis=0) , yerr = stats.sem(p_epoch3_on_interval_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_on_interval_chemo_l) > 0:
        axs2[3,0].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_on_interval_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_on_interval_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_on_interval_chemo_l) > 0:
        axs2[3,0].errorbar(session_id[1], np.mean(p_epoch2_on_interval_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_on_interval_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_on_interval_chemo_l) > 0:
        axs2[3,0].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_on_interval_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_on_interval_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_on_interval_opto_l) > 0:
        axs2[3,0].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_on_interval_opto_l,axis=0) , yerr = stats.sem(p_epoch1_on_interval_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_on_interval_opto_l) > 0:
        axs2[3,0].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_on_interval_opto_l,axis=0) , yerr = stats.sem(p_epoch2_on_interval_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_on_interval_opto_l) > 0:
        axs2[3,0].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_on_interval_opto_l,axis=0) , yerr = stats.sem(p_epoch3_on_interval_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [3,1]
    if len(p_epoch1_levret1end_on2_con_s) > 0:
        axs2[3,1].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_levret1end_on2_con_s,axis=0) , yerr = stats.sem(p_epoch1_levret1end_on2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_levret1end_on2_con_s) > 0:
        axs2[3,1].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_levret1end_on2_con_s,axis=0) , yerr = stats.sem(p_epoch2_levret1end_on2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_levret1end_on2_con_s) > 0:
        axs2[3,1].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_levret1end_on2_con_s,axis=0) , yerr = stats.sem(p_epoch3_levret1end_on2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_levret1end_on2_chemo_s) > 0:
        axs2[3,1].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_levret1end_on2_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_levret1end_on2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_levret1end_on2_chemo_s) > 0:
        axs2[3,1].errorbar(session_id[0], np.mean(p_epoch2_levret1end_on2_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_levret1end_on2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_levret1end_on2_chemo_s) > 0:
        axs2[3,1].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_levret1end_on2_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_levret1end_on2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_levret1end_on2_opto_s) > 0:
        axs2[3,1].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_levret1end_on2_opto_s,axis=0) , yerr = stats.sem(p_epoch1_levret1end_on2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(p_epoch2_levret1end_on2_opto_s) > 0:
        axs2[3,1].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_levret1end_on2_opto_s,axis=0) , yerr = stats.sem(p_epoch2_levret1end_on2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(p_epoch3_levret1end_on2_opto_s) > 0:
        axs2[3,1].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_levret1end_on2_opto_s,axis=0) , yerr = stats.sem(p_epoch3_levret1end_on2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(p_epoch1_levret1end_on2_con_l) > 0:
        axs2[3,1].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_levret1end_on2_con_l,axis=0) , yerr = stats.sem(p_epoch1_levret1end_on2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_levret1end_on2_con_l) > 0:
        axs2[3,1].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_levret1end_on2_con_l,axis=0) , yerr = stats.sem(p_epoch2_levret1end_on2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_levret1end_on2_con_l) > 0:
        axs2[3,1].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_levret1end_on2_con_l,axis=0) , yerr = stats.sem(p_epoch3_levret1end_on2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_levret1end_on2_chemo_l) > 0:
        axs2[3,1].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_levret1end_on2_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_levret1end_on2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_levret1end_on2_chemo_l) > 0:
        axs2[3,1].errorbar(session_id[1], np.mean(p_epoch2_levret1end_on2_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_levret1end_on2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_levret1end_on2_chemo_l) > 0:
        axs2[3,1].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_levret1end_on2_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_levret1end_on2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_levret1end_on2_opto_l) > 0:
        axs2[3,1].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_levret1end_on2_opto_l,axis=0) , yerr = stats.sem(p_epoch1_levret1end_on2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_levret1end_on2_opto_l) > 0:
        axs2[3,1].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_levret1end_on2_opto_l,axis=0) , yerr = stats.sem(p_epoch2_levret1end_on2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_levret1end_on2_opto_l) > 0:
        axs2[3,1].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_levret1end_on2_opto_l,axis=0) , yerr = stats.sem(p_epoch3_levret1end_on2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [4,0]
    if len(p_epoch1_base_amp1_con_s) > 0:
        axs2[4,0].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_base_amp1_con_s,axis=0) , yerr = stats.sem(p_epoch1_base_amp1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_base_amp1_con_s) > 0:
        axs2[4,0].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_base_amp1_con_s,axis=0) , yerr = stats.sem(p_epoch2_base_amp1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_base_amp1_con_s) > 0:
        axs2[4,0].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_base_amp1_con_s,axis=0) , yerr = stats.sem(p_epoch3_base_amp1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_base_amp1_chemo_s) > 0:
        axs2[4,0].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_base_amp1_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_base_amp1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_base_amp1_chemo_s) > 0:
        axs2[4,0].errorbar(session_id[0], np.mean(p_epoch2_base_amp1_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_base_amp1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_base_amp1_chemo_s) > 0:
        axs2[4,0].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_base_amp1_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_base_amp1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_base_amp1_opto_s) > 0:
        axs2[4,0].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_base_amp1_opto_s,axis=0) , yerr = stats.sem(p_epoch1_base_amp1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(p_epoch2_base_amp1_opto_s) > 0:
        axs2[4,0].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_base_amp1_opto_s,axis=0) , yerr = stats.sem(p_epoch2_base_amp1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(p_epoch3_base_amp1_opto_s) > 0:
        axs2[4,0].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_base_amp1_opto_s,axis=0) , yerr = stats.sem(p_epoch3_base_amp1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(p_epoch1_base_amp1_con_l) > 0:
        axs2[4,0].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_base_amp1_con_l,axis=0) , yerr = stats.sem(p_epoch1_base_amp1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_base_amp1_con_l) > 0:
        axs2[4,0].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_base_amp1_con_l,axis=0) , yerr = stats.sem(p_epoch2_base_amp1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_base_amp1_con_l) > 0:
        axs2[4,0].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_base_amp1_con_l,axis=0) , yerr = stats.sem(p_epoch3_base_amp1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_base_amp1_chemo_l) > 0:
        axs2[4,0].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_base_amp1_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_base_amp1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_base_amp1_chemo_l) > 0:
        axs2[4,0].errorbar(session_id[1], np.mean(p_epoch2_base_amp1_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_base_amp1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_base_amp1_chemo_l) > 0:
        axs2[4,0].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_base_amp1_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_base_amp1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_base_amp1_opto_l) > 0:
        axs2[4,0].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_base_amp1_opto_l,axis=0) , yerr = stats.sem(p_epoch1_base_amp1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_base_amp1_opto_l) > 0:
        axs2[4,0].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_base_amp1_opto_l,axis=0) , yerr = stats.sem(p_epoch2_base_amp1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_base_amp1_opto_l) > 0:
        axs2[4,0].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_base_amp1_opto_l,axis=0) , yerr = stats.sem(p_epoch3_base_amp1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [4,1]
    if len(p_epoch1_base_amp2_con_s) > 0:
        axs2[4,1].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_base_amp2_con_s,axis=0) , yerr = stats.sem(p_epoch1_base_amp2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_base_amp2_con_s) > 0:
        axs2[4,1].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_base_amp2_con_s,axis=0) , yerr = stats.sem(p_epoch2_base_amp2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_base_amp2_con_s) > 0:
        axs2[4,1].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_base_amp2_con_s,axis=0) , yerr = stats.sem(p_epoch3_base_amp2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_base_amp2_chemo_s) > 0:
        axs2[4,1].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_base_amp2_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_base_amp2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_base_amp2_chemo_s) > 0:
        axs2[4,1].errorbar(session_id[0], np.mean(p_epoch2_base_amp2_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_base_amp2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_base_amp2_chemo_s) > 0:
        axs2[4,1].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_base_amp2_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_base_amp2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_base_amp2_opto_s) > 0:
        axs2[4,1].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_base_amp2_opto_s,axis=0) , yerr = stats.sem(p_epoch1_base_amp2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(p_epoch2_base_amp2_opto_s) > 0:
        axs2[4,1].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_base_amp2_opto_s,axis=0) , yerr = stats.sem(p_epoch2_base_amp2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(p_epoch3_base_amp2_opto_s) > 0:
        axs2[4,1].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_base_amp2_opto_s,axis=0) , yerr = stats.sem(p_epoch3_base_amp2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(p_epoch1_base_amp2_con_l) > 0:
        axs2[4,1].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_base_amp2_con_l,axis=0) , yerr = stats.sem(p_epoch1_base_amp2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_base_amp2_con_l) > 0:
        axs2[4,1].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_base_amp2_con_l,axis=0) , yerr = stats.sem(p_epoch2_base_amp2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_base_amp2_con_l) > 0:
        axs2[4,1].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_base_amp2_con_l,axis=0) , yerr = stats.sem(p_epoch3_base_amp2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_base_amp2_chemo_l) > 0:
        axs2[4,1].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_base_amp2_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_base_amp2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_base_amp2_chemo_l) > 0:
        axs2[4,1].errorbar(session_id[1], np.mean(p_epoch2_base_amp2_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_base_amp2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_base_amp2_chemo_l) > 0:
        axs2[4,1].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_base_amp2_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_base_amp2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_base_amp2_opto_l) > 0:
        axs2[4,1].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_base_amp2_opto_l,axis=0) , yerr = stats.sem(p_epoch1_base_amp2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_base_amp2_opto_l) > 0:
        axs2[4,1].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_base_amp2_opto_l,axis=0) , yerr = stats.sem(p_epoch2_base_amp2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_base_amp2_opto_l) > 0:
        axs2[4,1].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_base_amp2_opto_l,axis=0) , yerr = stats.sem(p_epoch3_base_amp2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [5,0]
    if len(p_epoch1_max_vel1_con_s) > 0:
        axs2[5,0].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_max_vel1_con_s,axis=0) , yerr = stats.sem(p_epoch1_max_vel1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_max_vel1_con_s) > 0:
        axs2[5,0].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_max_vel1_con_s,axis=0) , yerr = stats.sem(p_epoch2_max_vel1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_max_vel1_con_s) > 0:
        axs2[5,0].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_max_vel1_con_s,axis=0) , yerr = stats.sem(p_epoch3_max_vel1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_max_vel1_chemo_s) > 0:
        axs2[5,0].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_max_vel1_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_max_vel1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_max_vel1_chemo_s) > 0:
        axs2[5,0].errorbar(session_id[0], np.mean(p_epoch2_max_vel1_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_max_vel1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_max_vel1_chemo_s) > 0:
        axs2[5,0].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_max_vel1_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_max_vel1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_max_vel1_opto_s) > 0:
        axs2[5,0].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_max_vel1_opto_s,axis=0) , yerr = stats.sem(p_epoch1_max_vel1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(p_epoch2_max_vel1_opto_s) > 0:
        axs2[5,0].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_max_vel1_opto_s,axis=0) , yerr = stats.sem(p_epoch2_max_vel1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(p_epoch3_max_vel1_opto_s) > 0:
        axs2[5,0].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_max_vel1_opto_s,axis=0) , yerr = stats.sem(p_epoch3_max_vel1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(p_epoch1_max_vel1_con_l) > 0:
        axs2[5,0].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_max_vel1_con_l,axis=0) , yerr = stats.sem(p_epoch1_max_vel1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_max_vel1_con_l) > 0:
        axs2[5,0].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_max_vel1_con_l,axis=0) , yerr = stats.sem(p_epoch2_max_vel1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_max_vel1_con_l) > 0:
        axs2[5,0].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_max_vel1_con_l,axis=0) , yerr = stats.sem(p_epoch3_max_vel1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_max_vel1_chemo_l) > 0:
        axs2[5,0].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_max_vel1_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_max_vel1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_max_vel1_chemo_l) > 0:
        axs2[5,0].errorbar(session_id[1], np.mean(p_epoch2_max_vel1_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_max_vel1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_max_vel1_chemo_l) > 0:
        axs2[5,0].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_max_vel1_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_max_vel1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_max_vel1_opto_l) > 0:
        axs2[5,0].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_max_vel1_opto_l,axis=0) , yerr = stats.sem(p_epoch1_max_vel1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_max_vel1_opto_l) > 0:
        axs2[5,0].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_max_vel1_opto_l,axis=0) , yerr = stats.sem(p_epoch2_max_vel1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_max_vel1_opto_l) > 0:
        axs2[5,0].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_max_vel1_opto_l,axis=0) , yerr = stats.sem(p_epoch3_max_vel1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [5,1]
    if len(p_epoch1_max_vel2_con_s) > 0:
        axs2[5,1].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_max_vel2_con_s,axis=0) , yerr = stats.sem(p_epoch1_max_vel2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_max_vel2_con_s) > 0:
        axs2[5,1].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_max_vel2_con_s,axis=0) , yerr = stats.sem(p_epoch2_max_vel2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_max_vel2_con_s) > 0:
        axs2[5,1].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_max_vel2_con_s,axis=0) , yerr = stats.sem(p_epoch3_max_vel2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_max_vel2_chemo_s) > 0:
        axs2[5,1].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_max_vel2_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_max_vel2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_max_vel2_chemo_s) > 0:
        axs2[5,1].errorbar(session_id[0], np.mean(p_epoch2_max_vel2_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_max_vel2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_max_vel2_chemo_s) > 0:
        axs2[5,1].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_max_vel2_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_max_vel2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_max_vel2_opto_s) > 0:
        axs2[5,1].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_max_vel2_opto_s,axis=0) , yerr = stats.sem(p_epoch1_max_vel2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(p_epoch2_max_vel2_opto_s) > 0:
        axs2[5,1].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_max_vel2_opto_s,axis=0) , yerr = stats.sem(p_epoch2_max_vel2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(p_epoch3_max_vel2_opto_s) > 0:
        axs2[5,1].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_max_vel2_opto_s,axis=0) , yerr = stats.sem(p_epoch3_max_vel2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(p_epoch1_max_vel2_con_l) > 0:
        axs2[5,1].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_max_vel2_con_l,axis=0) , yerr = stats.sem(p_epoch1_max_vel2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_max_vel2_con_l) > 0:
        axs2[5,1].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_max_vel2_con_l,axis=0) , yerr = stats.sem(p_epoch2_max_vel2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_max_vel2_con_l) > 0:
        axs2[5,1].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_max_vel2_con_l,axis=0) , yerr = stats.sem(p_epoch3_max_vel2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_max_vel2_chemo_l) > 0:
        axs2[5,1].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_max_vel2_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_max_vel2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_max_vel2_chemo_l) > 0:
        axs2[5,1].errorbar(session_id[1], np.mean(p_epoch2_max_vel2_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_max_vel2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_max_vel2_chemo_l) > 0:
        axs2[5,1].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_max_vel2_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_max_vel2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_max_vel2_opto_l) > 0:
        axs2[5,1].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_max_vel2_opto_l,axis=0) , yerr = stats.sem(p_epoch1_max_vel2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_max_vel2_opto_l) > 0:
        axs2[5,1].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_max_vel2_opto_l,axis=0) , yerr = stats.sem(p_epoch2_max_vel2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_max_vel2_opto_l) > 0:
        axs2[5,1].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_max_vel2_opto_l,axis=0) , yerr = stats.sem(p_epoch3_max_vel2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [6,0]
    if len(p_epoch1_velaton1_con_s) > 0:
        axs2[6,0].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_velaton1_con_s,axis=0) , yerr = stats.sem(p_epoch1_velaton1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_velaton1_con_s) > 0:
        axs2[6,0].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_velaton1_con_s,axis=0) , yerr = stats.sem(p_epoch2_velaton1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_velaton1_con_s) > 0:
        axs2[6,0].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_velaton1_con_s,axis=0) , yerr = stats.sem(p_epoch3_velaton1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_velaton1_chemo_s) > 0:
        axs2[6,0].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_velaton1_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_velaton1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_velaton1_chemo_s) > 0:
        axs2[6,0].errorbar(session_id[0], np.mean(p_epoch2_velaton1_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_velaton1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_velaton1_chemo_s) > 0:
        axs2[6,0].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_velaton1_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_velaton1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_velaton1_opto_s) > 0:
        axs2[6,0].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_velaton1_opto_s,axis=0) , yerr = stats.sem(p_epoch1_velaton1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(p_epoch2_velaton1_opto_s) > 0:
        axs2[6,0].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_velaton1_opto_s,axis=0) , yerr = stats.sem(p_epoch2_velaton1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(p_epoch3_velaton1_opto_s) > 0:
        axs2[6,0].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_velaton1_opto_s,axis=0) , yerr = stats.sem(p_epoch3_velaton1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(p_epoch1_velaton1_con_l) > 0:
        axs2[6,0].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_velaton1_con_l,axis=0) , yerr = stats.sem(p_epoch1_velaton1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_velaton1_con_l) > 0:
        axs2[6,0].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_velaton1_con_l,axis=0) , yerr = stats.sem(p_epoch2_velaton1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_velaton1_con_l) > 0:
        axs2[6,0].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_velaton1_con_l,axis=0) , yerr = stats.sem(p_epoch3_velaton1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_velaton1_chemo_l) > 0:
        axs2[6,0].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_velaton1_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_velaton1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_velaton1_chemo_l) > 0:
        axs2[6,0].errorbar(session_id[1], np.mean(p_epoch2_velaton1_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_velaton1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_velaton1_chemo_l) > 0:
        axs2[6,0].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_velaton1_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_velaton1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_velaton1_opto_l) > 0:
        axs2[6,0].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_velaton1_opto_l,axis=0) , yerr = stats.sem(p_epoch1_velaton1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_velaton1_opto_l) > 0:
        axs2[6,0].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_velaton1_opto_l,axis=0) , yerr = stats.sem(p_epoch2_velaton1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_velaton1_opto_l) > 0:
        axs2[6,0].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_velaton1_opto_l,axis=0) , yerr = stats.sem(p_epoch3_velaton1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [6,1]
    if len(p_epoch1_velaton2_con_s) > 0:
        axs2[6,1].errorbar(session_id[0] - 2.5*offset, np.mean(p_epoch1_velaton2_con_s,axis=0) , yerr = stats.sem(p_epoch1_velaton2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(p_epoch2_velaton2_con_s) > 0:
        axs2[6,1].errorbar(session_id[0] - 2.2*offset, np.mean(p_epoch2_velaton2_con_s,axis=0) , yerr = stats.sem(p_epoch2_velaton2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(p_epoch3_velaton2_con_s) > 0:
        axs2[6,1].errorbar(session_id[0] - 1.9*offset, np.mean(p_epoch3_velaton2_con_s,axis=0) , yerr = stats.sem(p_epoch3_velaton2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(p_epoch1_velaton2_chemo_s) > 0:
        axs2[6,1].errorbar(session_id[0] - 0.3*offset, np.mean(p_epoch1_velaton2_chemo_s,axis=0) , yerr = stats.sem(p_epoch1_velaton2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(p_epoch2_velaton2_chemo_s) > 0:
        axs2[6,1].errorbar(session_id[0], np.mean(p_epoch2_velaton2_chemo_s,axis=0) , yerr = stats.sem(p_epoch2_velaton2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(p_epoch3_velaton2_chemo_s) > 0:
        axs2[6,1].errorbar(session_id[0] + 0.3*offset, np.mean(p_epoch3_velaton2_chemo_s,axis=0) , yerr = stats.sem(p_epoch3_velaton2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(p_epoch1_velaton2_opto_s) > 0:
        axs2[6,1].errorbar(session_id[0] + 1.9*offset, np.mean(p_epoch1_velaton2_opto_s,axis=0) , yerr = stats.sem(p_epoch1_velaton2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(p_epoch2_velaton2_opto_s) > 0:
        axs2[6,1].errorbar(session_id[0] + 2.2*offset, np.mean(p_epoch2_velaton2_opto_s,axis=0) , yerr = stats.sem(p_epoch2_velaton2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(p_epoch3_velaton2_opto_s) > 0:
        axs2[6,1].errorbar(session_id[0] + 2.5*offset, np.mean(p_epoch3_velaton2_opto_s,axis=0) , yerr = stats.sem(p_epoch3_velaton2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(p_epoch1_velaton2_con_l) > 0:
        axs2[6,1].errorbar(session_id[1] - 2.5*offset, np.mean(p_epoch1_velaton2_con_l,axis=0) , yerr = stats.sem(p_epoch1_velaton2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(p_epoch2_velaton2_con_l) > 0:
        axs2[6,1].errorbar(session_id[1] - 2.2*offset, np.mean(p_epoch2_velaton2_con_l,axis=0) , yerr = stats.sem(p_epoch2_velaton2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(p_epoch3_velaton2_con_l) > 0:
        axs2[6,1].errorbar(session_id[1] - 1.9*offset, np.mean(p_epoch3_velaton2_con_l,axis=0) , yerr = stats.sem(p_epoch3_velaton2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(p_epoch1_velaton2_chemo_l) > 0:
        axs2[6,1].errorbar(session_id[1] - 0.3*offset, np.mean(p_epoch1_velaton2_chemo_l,axis=0) , yerr = stats.sem(p_epoch1_velaton2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(p_epoch2_velaton2_chemo_l) > 0:
        axs2[6,1].errorbar(session_id[1], np.mean(p_epoch2_velaton2_chemo_l,axis=0) , yerr = stats.sem(p_epoch2_velaton2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(p_epoch3_velaton2_chemo_l) > 0:
        axs2[6,1].errorbar(session_id[1] + 0.3*offset, np.mean(p_epoch3_velaton2_chemo_l,axis=0) , yerr = stats.sem(p_epoch3_velaton2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(p_epoch1_velaton2_opto_l) > 0:
        axs2[6,1].errorbar(session_id[1] + 1.9*offset, np.mean(p_epoch1_velaton2_opto_l,axis=0) , yerr = stats.sem(p_epoch1_velaton2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(p_epoch2_velaton2_opto_l) > 0:
        axs2[6,1].errorbar(session_id[1] + 2.2*offset, np.mean(p_epoch2_velaton2_opto_l,axis=0) , yerr = stats.sem(p_epoch2_velaton2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(p_epoch3_velaton2_opto_l) > 0:
        axs2[6,1].errorbar(session_id[1] + 2.5*offset, np.mean(p_epoch3_velaton2_opto_l,axis=0) , yerr = stats.sem(p_epoch3_velaton2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])

    axs2[0,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[0,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[1,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[1,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[2,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[2,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[3,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[3,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[4,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[4,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[5,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[5,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[6,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[6,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

    fig2.tight_layout()

    # GRAND AVERAGE ###########################################################################
    print('####### Grand Average ######')
    fig3, axs3 = plt.subplots(nrows=7, ncols=2, figsize=(16 , 35))
    fig3.subplots_adjust(hspace=0.7)
    fig3.suptitle(subject + '\n Time, Amplitude and Kinematic Quatification Grand Average\n')
    # Vis to Onset
    axs3[0,0].set_title(3*'\n' + 'Vis1 to Onset1\n') 
    axs3[0,0].spines['right'].set_visible(False)
    axs3[0,0].spines['top'].set_visible(False)
    axs3[0,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs3[0,0].set_xticks(session_id)
    axs3[0,0].set_xticklabels(dates, ha = 'center')

    if any(0 in row for row in isSelfTimedMode):
        axs3[0,1].set_title('Vis2 to Onset2\n') 
    else:
        axs3[0,1].set_title('WaitforPress2 to Onset2\n') 
    axs3[0,1].spines['right'].set_visible(False)
    axs3[0,1].spines['top'].set_visible(False)
    axs3[0,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs3[0,1].set_xticks(session_id)
    axs3[0,1].set_xticklabels(dates, ha = 'center')
    # Onset to Target
    axs3[1,0].set_title('Onset1 to Target1\n') 
    axs3[1,0].spines['right'].set_visible(False)
    axs3[1,0].spines['top'].set_visible(False)
    axs3[1,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs3[1,0].set_xticks(session_id)
    axs3[1,0].set_xticklabels(dates, ha = 'center')

    axs3[1,1].set_title('Onset2 to Target2\n') 
    axs3[1,1].spines['right'].set_visible(False)
    axs3[1,1].spines['top'].set_visible(False)
    axs3[1,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs3[1,1].set_xticks(session_id)
    axs3[1,1].set_xticklabels(dates, ha = 'center')
    # Onset to peak time
    axs3[2,0].set_title('Onset1 to Peak1\n') 
    axs3[2,0].spines['right'].set_visible(False)
    axs3[2,0].spines['top'].set_visible(False)
    axs3[2,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs3[2,0].set_xticks(session_id)
    axs3[2,0].set_xticklabels(dates, ha = 'center')

    axs3[2,1].set_title('Onset2 to Peak2\n') 
    axs3[2,1].spines['right'].set_visible(False)
    axs3[2,1].spines['top'].set_visible(False)
    axs3[2,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs3[2,1].set_xticks(session_id)
    axs3[2,1].set_xticklabels(dates, ha = 'center')
    # Onset Interval
    axs3[3,0].set_title('Onset1 and Onset2 Interval\n') 
    axs3[3,0].spines['right'].set_visible(False)
    axs3[3,0].spines['top'].set_visible(False)
    axs3[3,0].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs3[3,0].set_xticks(session_id)
    axs3[3,0].set_xticklabels(dates, ha = 'center')

    axs3[3,1].set_title('LeverRetract1_End to Onset2\n') 
    axs3[3,1].spines['right'].set_visible(False)
    axs3[3,1].spines['top'].set_visible(False)
    axs3[3,1].set_ylabel('Time Interval(s) Mean +/- SEM')
    axs3[3,1].set_xticks(session_id)
    axs3[3,1].set_xticklabels(dates, ha = 'center')
    # Baseline to Peak
    axs3[4,0].set_title('Baseline to Peak1\n') 
    axs3[4,0].spines['right'].set_visible(False)
    axs3[4,0].spines['top'].set_visible(False)
    axs3[4,0].set_ylabel('Lever Deflection(deg) Mean +/- SEM')
    axs3[4,0].set_xticks(session_id)
    axs3[4,0].set_xticklabels(dates, ha = 'center')

    axs3[4,1].set_title('Baseline to Peak2\n') 
    axs3[4,1].spines['right'].set_visible(False)
    axs3[4,1].spines['top'].set_visible(False)
    axs3[4,1].set_ylabel('Lever Deflection(deg) Mean +/- SEM')
    axs3[4,1].set_xticks(session_id)
    axs3[4,1].set_xticklabels(dates, ha = 'center')
    # Max Velocity Press
    axs3[5,0].set_title('Max Velocity Press1\n') 
    axs3[5,0].spines['right'].set_visible(False)
    axs3[5,0].spines['top'].set_visible(False)
    axs3[5,0].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs3[5,0].set_xticks(session_id)
    axs3[5,0].set_xticklabels(dates, ha = 'center')

    axs3[5,1].set_title('Max Velocity Press2\n') 
    axs3[5,1].spines['right'].set_visible(False)
    axs3[5,1].spines['top'].set_visible(False)
    axs3[5,1].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs3[5,1].set_xticks(session_id)
    axs3[5,1].set_xticklabels(dates, ha = 'center')
    # Velocity at Onset
    axs3[6,0].set_title('Velocity at Onset1\n') 
    axs3[6,0].spines['right'].set_visible(False)
    axs3[6,0].spines['top'].set_visible(False)
    axs3[6,0].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs3[6,0].set_xticks(session_id)
    axs3[6,0].set_xticklabels(dates, ha = 'center')

    axs3[6,1].set_title('Velocity at Onset2\n') 
    axs3[6,1].spines['right'].set_visible(False)
    axs3[6,1].spines['top'].set_visible(False)
    axs3[6,1].set_ylabel('Lever Deflection Velocity(deg/s) Mean +/- SEM')
    axs3[6,1].set_xticks(session_id)
    axs3[6,1].set_xticklabels(dates, ha = 'center')

    # Plotting
    # [0,0]
    if len(G_epoch1_vis1_on1_con_s) > 0:
        axs3[0,0].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_vis1_on1_con_s,axis=0) , yerr = stats.sem(G_epoch1_vis1_on1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_vis1_on1_con_s) > 0:
        axs3[0,0].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_vis1_on1_con_s,axis=0) , yerr = stats.sem(G_epoch2_vis1_on1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_vis1_on1_con_s) > 0:
        axs3[0,0].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_vis1_on1_con_s,axis=0) , yerr = stats.sem(G_epoch3_vis1_on1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_vis1_on1_chemo_s) > 0:
        axs3[0,0].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_vis1_on1_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_vis1_on1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_vis1_on1_chemo_s) > 0:
        axs3[0,0].errorbar(session_id[0], np.mean(G_epoch2_vis1_on1_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_vis1_on1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_vis1_on1_chemo_s) > 0:
        axs3[0,0].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_vis1_on1_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_vis1_on1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_vis1_on1_opto_s) > 0:
        axs3[0,0].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_vis1_on1_opto_s,axis=0) , yerr = stats.sem(G_epoch1_vis1_on1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_vis1_on1_opto_s) > 0:
        axs3[0,0].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_vis1_on1_opto_s,axis=0) , yerr = stats.sem(G_epoch2_vis1_on1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_vis1_on1_opto_s) > 0:
        axs3[0,0].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_vis1_on1_opto_s,axis=0) , yerr = stats.sem(G_epoch3_vis1_on1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_vis1_on1_con_l) > 0:
        axs3[0,0].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_vis1_on1_con_l,axis=0) , yerr = stats.sem(G_epoch1_vis1_on1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_vis1_on1_con_l) > 0:
        axs3[0,0].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_vis1_on1_con_l,axis=0) , yerr = stats.sem(G_epoch2_vis1_on1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_vis1_on1_con_l) > 0:
        axs3[0,0].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_vis1_on1_con_l,axis=0) , yerr = stats.sem(G_epoch3_vis1_on1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_vis1_on1_chemo_l) > 0:
        axs3[0,0].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_vis1_on1_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_vis1_on1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_vis1_on1_chemo_l) > 0:
        axs3[0,0].errorbar(session_id[1], np.mean(G_epoch2_vis1_on1_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_vis1_on1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_vis1_on1_chemo_l) > 0:
        axs3[0,0].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_vis1_on1_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_vis1_on1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_vis1_on1_opto_l) > 0:
        axs3[0,0].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_vis1_on1_opto_l,axis=0) , yerr = stats.sem(G_epoch1_vis1_on1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_vis1_on1_opto_l) > 0:
        axs3[0,0].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_vis1_on1_opto_l,axis=0) , yerr = stats.sem(G_epoch2_vis1_on1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_vis1_on1_opto_l) > 0:
        axs3[0,0].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_vis1_on1_opto_l,axis=0) , yerr = stats.sem(G_epoch3_vis1_on1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [0,1]
    if len(G_epoch1_vis2_on2_con_s) > 0:
        axs3[0,1].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_vis2_on2_con_s,axis=0) , yerr = stats.sem(G_epoch1_vis2_on2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_vis2_on2_con_s) > 0:
        axs3[0,1].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_vis2_on2_con_s,axis=0) , yerr = stats.sem(G_epoch2_vis2_on2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_vis2_on2_con_s) > 0:
        axs3[0,1].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_vis2_on2_con_s,axis=0) , yerr = stats.sem(G_epoch3_vis2_on2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_vis2_on2_chemo_s) > 0:
        axs3[0,1].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_vis2_on2_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_vis2_on2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_vis2_on2_chemo_s) > 0:
        axs3[0,1].errorbar(session_id[0], np.mean(G_epoch2_vis2_on2_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_vis2_on2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_vis2_on2_chemo_s) > 0:
        axs3[0,1].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_vis2_on2_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_vis2_on2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_vis2_on2_opto_s) > 0:
        axs3[0,1].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_vis2_on2_opto_s,axis=0) , yerr = stats.sem(G_epoch1_vis2_on2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_vis2_on2_opto_s) > 0:
        axs3[0,1].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_vis2_on2_opto_s,axis=0) , yerr = stats.sem(G_epoch2_vis2_on2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_vis2_on2_opto_s) > 0:
        axs3[0,1].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_vis2_on2_opto_s,axis=0) , yerr = stats.sem(G_epoch3_vis2_on2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_vis2_on2_con_l) > 0:
        axs3[0,1].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_vis2_on2_con_l,axis=0) , yerr = stats.sem(G_epoch1_vis2_on2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_vis2_on2_con_l) > 0:
        axs3[0,1].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_vis2_on2_con_l,axis=0) , yerr = stats.sem(G_epoch2_vis2_on2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_vis2_on2_con_l) > 0:
        axs3[0,1].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_vis2_on2_con_l,axis=0) , yerr = stats.sem(G_epoch3_vis2_on2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_vis2_on2_chemo_l) > 0:
        axs3[0,1].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_vis2_on2_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_vis2_on2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_vis2_on2_chemo_l) > 0:
        axs3[0,1].errorbar(session_id[1], np.mean(G_epoch2_vis2_on2_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_vis2_on2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_vis2_on2_chemo_l) > 0:
        axs3[0,1].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_vis2_on2_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_vis2_on2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_vis2_on2_opto_l) > 0:
        axs3[0,1].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_vis2_on2_opto_l,axis=0) , yerr = stats.sem(G_epoch1_vis2_on2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_vis2_on2_opto_l) > 0:
        axs3[0,1].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_vis2_on2_opto_l,axis=0) , yerr = stats.sem(G_epoch2_vis2_on2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_vis2_on2_opto_l) > 0:
        axs3[0,1].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_vis2_on2_opto_l,axis=0) , yerr = stats.sem(G_epoch3_vis2_on2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [1,0]
    if len(G_epoch1_on1_targ1_con_s) > 0:
        axs3[1,0].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_on1_targ1_con_s,axis=0) , yerr = stats.sem(G_epoch1_on1_targ1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_on1_targ1_con_s) > 0:
        axs3[1,0].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_on1_targ1_con_s,axis=0) , yerr = stats.sem(G_epoch2_on1_targ1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_on1_targ1_con_s) > 0:
        axs3[1,0].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_on1_targ1_con_s,axis=0) , yerr = stats.sem(G_epoch3_on1_targ1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_on1_targ1_chemo_s) > 0:
        axs3[1,0].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_on1_targ1_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_on1_targ1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_on1_targ1_chemo_s) > 0:
        axs3[1,0].errorbar(session_id[0], np.mean(G_epoch2_on1_targ1_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_on1_targ1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_on1_targ1_chemo_s) > 0:
        axs3[1,0].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_on1_targ1_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_on1_targ1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_on1_targ1_opto_s) > 0:
        axs3[1,0].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_on1_targ1_opto_s,axis=0) , yerr = stats.sem(G_epoch1_on1_targ1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(G_epoch2_on1_targ1_opto_s) > 0:
        axs3[1,0].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_on1_targ1_opto_s,axis=0) , yerr = stats.sem(G_epoch2_on1_targ1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(G_epoch3_on1_targ1_opto_s) > 0:
        axs3[1,0].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_on1_targ1_opto_s,axis=0) , yerr = stats.sem(G_epoch3_on1_targ1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(G_epoch1_on1_targ1_con_l) > 0:
        axs3[1,0].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_on1_targ1_con_l,axis=0) , yerr = stats.sem(G_epoch1_on1_targ1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_on1_targ1_con_l) > 0:
        axs3[1,0].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_on1_targ1_con_l,axis=0) , yerr = stats.sem(G_epoch2_on1_targ1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_on1_targ1_con_l) > 0:
        axs3[1,0].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_on1_targ1_con_l,axis=0) , yerr = stats.sem(G_epoch3_on1_targ1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_on1_targ1_chemo_l) > 0:
        axs3[1,0].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_on1_targ1_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_on1_targ1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_on1_targ1_chemo_l) > 0:
        axs3[1,0].errorbar(session_id[1], np.mean(G_epoch2_on1_targ1_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_on1_targ1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_on1_targ1_chemo_l) > 0:
        axs3[1,0].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_on1_targ1_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_on1_targ1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_on1_targ1_opto_l) > 0:
        axs3[1,0].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_on1_targ1_opto_l,axis=0) , yerr = stats.sem(G_epoch1_on1_targ1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_on1_targ1_opto_l) > 0:
        axs3[1,0].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_on1_targ1_opto_l,axis=0) , yerr = stats.sem(G_epoch2_on1_targ1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_on1_targ1_opto_l) > 0:
        axs3[1,0].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_on1_targ1_opto_l,axis=0) , yerr = stats.sem(G_epoch3_on1_targ1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [1,1]
    if len(G_epoch1_on2_targ2_con_s) > 0:
        axs3[1,1].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_on2_targ2_con_s,axis=0) , yerr = stats.sem(G_epoch1_on2_targ2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_on2_targ2_con_s) > 0:
        axs3[1,1].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_on2_targ2_con_s,axis=0) , yerr = stats.sem(G_epoch2_on2_targ2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_on2_targ2_con_s) > 0:
        axs3[1,1].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_on2_targ2_con_s,axis=0) , yerr = stats.sem(G_epoch3_on2_targ2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_on2_targ2_chemo_s) > 0:
        axs3[1,1].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_on2_targ2_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_on2_targ2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_on2_targ2_chemo_s) > 0:
        axs3[1,1].errorbar(session_id[0], np.mean(G_epoch2_on2_targ2_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_on2_targ2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_on2_targ2_chemo_s) > 0:
        axs3[1,1].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_on2_targ2_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_on2_targ2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_on2_targ2_opto_s) > 0:
        axs3[1,1].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_on2_targ2_opto_s,axis=0) , yerr = stats.sem(G_epoch1_on2_targ2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(G_epoch2_on2_targ2_opto_s) > 0:
        axs3[1,1].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_on2_targ2_opto_s,axis=0) , yerr = stats.sem(G_epoch2_on2_targ2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(G_epoch3_on2_targ2_opto_s) > 0:
        axs3[1,1].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_on2_targ2_opto_s,axis=0) , yerr = stats.sem(G_epoch3_on2_targ2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(G_epoch1_on2_targ2_con_l) > 0:
        axs3[1,1].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_on2_targ2_con_l,axis=0) , yerr = stats.sem(G_epoch1_on2_targ2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_on2_targ2_con_l) > 0:
        axs3[1,1].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_on2_targ2_con_l,axis=0) , yerr = stats.sem(G_epoch2_on2_targ2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_on2_targ2_con_l) > 0:
        axs3[1,1].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_on2_targ2_con_l,axis=0) , yerr = stats.sem(G_epoch3_on2_targ2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_on2_targ2_chemo_l) > 0:
        axs3[1,1].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_on2_targ2_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_on2_targ2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_on2_targ2_chemo_l) > 0:
        axs3[1,1].errorbar(session_id[1], np.mean(G_epoch2_on2_targ2_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_on2_targ2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_on2_targ2_chemo_l) > 0:
        axs3[1,1].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_on2_targ2_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_on2_targ2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_on2_targ2_opto_l) > 0:
        axs3[1,1].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_on2_targ2_opto_l,axis=0) , yerr = stats.sem(G_epoch1_on2_targ2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_on2_targ2_opto_l) > 0:
        axs3[1,1].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_on2_targ2_opto_l,axis=0) , yerr = stats.sem(G_epoch2_on2_targ2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_on2_targ2_opto_l) > 0:
        axs3[1,1].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_on2_targ2_opto_l,axis=0) , yerr = stats.sem(G_epoch3_on2_targ2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])

    # [2,0]
    if len(G_epoch1_on1_amp1_con_s) > 0:
        axs3[2,0].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_on1_amp1_con_s,axis=0) , yerr = stats.sem(G_epoch1_on1_amp1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_on1_amp1_con_s) > 0:
        axs3[2,0].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_on1_amp1_con_s,axis=0) , yerr = stats.sem(G_epoch2_on1_amp1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_on1_amp1_con_s) > 0:
        axs3[2,0].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_on1_amp1_con_s,axis=0) , yerr = stats.sem(G_epoch3_on1_amp1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_on1_amp1_chemo_s) > 0:
        axs3[2,0].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_on1_amp1_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_on1_amp1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_on1_amp1_chemo_s) > 0:
        axs3[2,0].errorbar(session_id[0], np.mean(G_epoch2_on1_amp1_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_on1_amp1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_on1_amp1_chemo_s) > 0:
        axs3[2,0].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_on1_amp1_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_on1_amp1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_on1_amp1_opto_s) > 0:
        axs3[2,0].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_on1_amp1_opto_s,axis=0) , yerr = stats.sem(G_epoch1_on1_amp1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(G_epoch2_on1_amp1_opto_s) > 0:
        axs3[2,0].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_on1_amp1_opto_s,axis=0) , yerr = stats.sem(G_epoch2_on1_amp1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(G_epoch3_on1_amp1_opto_s) > 0:
        axs3[2,0].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_on1_amp1_opto_s,axis=0) , yerr = stats.sem(G_epoch3_on1_amp1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(G_epoch1_on1_amp1_con_l) > 0:
        axs3[2,0].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_on1_amp1_con_l,axis=0) , yerr = stats.sem(G_epoch1_on1_amp1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_on1_amp1_con_l) > 0:
        axs3[2,0].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_on1_amp1_con_l,axis=0) , yerr = stats.sem(G_epoch2_on1_amp1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_on1_amp1_con_l) > 0:
        axs3[2,0].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_on1_amp1_con_l,axis=0) , yerr = stats.sem(G_epoch3_on1_amp1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_on1_amp1_chemo_l) > 0:
        axs3[2,0].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_on1_amp1_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_on1_amp1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_on1_amp1_chemo_l) > 0:
        axs3[2,0].errorbar(session_id[1], np.mean(G_epoch2_on1_amp1_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_on1_amp1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_on1_amp1_chemo_l) > 0:
        axs3[2,0].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_on1_amp1_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_on1_amp1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_on1_amp1_opto_l) > 0:
        axs3[2,0].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_on1_amp1_opto_l,axis=0) , yerr = stats.sem(G_epoch1_on1_amp1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_on1_amp1_opto_l) > 0:
        axs3[2,0].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_on1_amp1_opto_l,axis=0) , yerr = stats.sem(G_epoch2_on1_amp1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_on1_amp1_opto_l) > 0:
        axs3[2,0].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_on1_amp1_opto_l,axis=0) , yerr = stats.sem(G_epoch3_on1_amp1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [2,1]
    if len(G_epoch1_on2_amp2_con_s) > 0:
        axs3[2,1].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_on2_amp2_con_s,axis=0) , yerr = stats.sem(G_epoch1_on2_amp2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_on2_amp2_con_s) > 0:
        axs3[2,1].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_on2_amp2_con_s,axis=0) , yerr = stats.sem(G_epoch2_on2_amp2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_on2_amp2_con_s) > 0:
        axs3[2,1].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_on2_amp2_con_s,axis=0) , yerr = stats.sem(G_epoch3_on2_amp2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_on2_amp2_chemo_s) > 0:
        axs3[2,1].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_on2_amp2_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_on2_amp2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_on2_amp2_chemo_s) > 0:
        axs3[2,1].errorbar(session_id[0], np.mean(G_epoch2_on2_amp2_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_on2_amp2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_on2_amp2_chemo_s) > 0:
        axs3[2,1].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_on2_amp2_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_on2_amp2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_on2_amp2_opto_s) > 0:
        axs3[2,1].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_on2_amp2_opto_s,axis=0) , yerr = stats.sem(G_epoch1_on2_amp2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(G_epoch2_on2_amp2_opto_s) > 0:
        axs3[2,1].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_on2_amp2_opto_s,axis=0) , yerr = stats.sem(G_epoch2_on2_amp2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(G_epoch3_on2_amp2_opto_s) > 0:
        axs3[2,1].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_on2_amp2_opto_s,axis=0) , yerr = stats.sem(G_epoch3_on2_amp2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(G_epoch1_on2_amp2_con_l) > 0:
        axs3[2,1].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_on2_amp2_con_l,axis=0) , yerr = stats.sem(G_epoch1_on2_amp2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_on2_amp2_con_l) > 0:
        axs3[2,1].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_on2_amp2_con_l,axis=0) , yerr = stats.sem(G_epoch2_on2_amp2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_on2_amp2_con_l) > 0:
        axs3[2,1].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_on2_amp2_con_l,axis=0) , yerr = stats.sem(G_epoch3_on2_amp2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_on2_amp2_chemo_l) > 0:
        axs3[2,1].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_on2_amp2_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_on2_amp2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_on2_amp2_chemo_l) > 0:
        axs3[2,1].errorbar(session_id[1], np.mean(G_epoch2_on2_amp2_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_on2_amp2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_on2_amp2_chemo_l) > 0:
        axs3[2,1].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_on2_amp2_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_on2_amp2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_on2_amp2_opto_l) > 0:
        axs3[2,1].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_on2_amp2_opto_l,axis=0) , yerr = stats.sem(G_epoch1_on2_amp2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_on2_amp2_opto_l) > 0:
        axs3[2,1].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_on2_amp2_opto_l,axis=0) , yerr = stats.sem(G_epoch2_on2_amp2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_on2_amp2_opto_l) > 0:
        axs3[2,1].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_on2_amp2_opto_l,axis=0) , yerr = stats.sem(G_epoch3_on2_amp2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [3,0]
    if len(G_epoch1_on_interval_con_s) > 0:
        axs3[3,0].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_on_interval_con_s,axis=0) , yerr = stats.sem(G_epoch1_on_interval_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_on_interval_con_s) > 0:
        axs3[3,0].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_on_interval_con_s,axis=0) , yerr = stats.sem(G_epoch2_on_interval_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_on_interval_con_s) > 0:
        axs3[3,0].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_on_interval_con_s,axis=0) , yerr = stats.sem(G_epoch3_on_interval_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_on_interval_chemo_s) > 0:
        axs3[3,0].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_on_interval_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_on_interval_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_on_interval_chemo_s) > 0:
        axs3[3,0].errorbar(session_id[0], np.mean(G_epoch2_on_interval_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_on_interval_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_on_interval_chemo_s) > 0:
        axs3[3,0].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_on_interval_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_on_interval_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_on_interval_opto_s) > 0:
        axs3[3,0].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_on_interval_opto_s,axis=0) , yerr = stats.sem(G_epoch1_on_interval_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(G_epoch2_on_interval_opto_s) > 0:
        axs3[3,0].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_on_interval_opto_s,axis=0) , yerr = stats.sem(G_epoch2_on_interval_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(G_epoch3_on_interval_opto_s) > 0:
        axs3[3,0].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_on_interval_opto_s,axis=0) , yerr = stats.sem(G_epoch3_on_interval_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(G_epoch1_on_interval_con_l) > 0:
        axs3[3,0].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_on_interval_con_l,axis=0) , yerr = stats.sem(G_epoch1_on_interval_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_on_interval_con_l) > 0:
        axs3[3,0].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_on_interval_con_l,axis=0) , yerr = stats.sem(G_epoch2_on_interval_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_on_interval_con_l) > 0:
        axs3[3,0].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_on_interval_con_l,axis=0) , yerr = stats.sem(G_epoch3_on_interval_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_on_interval_chemo_l) > 0:
        axs3[3,0].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_on_interval_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_on_interval_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_on_interval_chemo_l) > 0:
        axs3[3,0].errorbar(session_id[1], np.mean(G_epoch2_on_interval_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_on_interval_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_on_interval_chemo_l) > 0:
        axs3[3,0].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_on_interval_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_on_interval_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_on_interval_opto_l) > 0:
        axs3[3,0].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_on_interval_opto_l,axis=0) , yerr = stats.sem(G_epoch1_on_interval_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_on_interval_opto_l) > 0:
        axs3[3,0].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_on_interval_opto_l,axis=0) , yerr = stats.sem(G_epoch2_on_interval_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_on_interval_opto_l) > 0:
        axs3[3,0].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_on_interval_opto_l,axis=0) , yerr = stats.sem(G_epoch3_on_interval_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [3,1]
    if len(G_epoch1_levret1end_on2_con_s) > 0:
        axs3[3,1].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_levret1end_on2_con_s,axis=0) , yerr = stats.sem(G_epoch1_levret1end_on2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_levret1end_on2_con_s) > 0:
        axs3[3,1].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_levret1end_on2_con_s,axis=0) , yerr = stats.sem(G_epoch2_levret1end_on2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_levret1end_on2_con_s) > 0:
        axs3[3,1].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_levret1end_on2_con_s,axis=0) , yerr = stats.sem(G_epoch3_levret1end_on2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_levret1end_on2_chemo_s) > 0:
        axs3[3,1].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_levret1end_on2_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_levret1end_on2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_levret1end_on2_chemo_s) > 0:
        axs3[3,1].errorbar(session_id[0], np.mean(G_epoch2_levret1end_on2_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_levret1end_on2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_levret1end_on2_chemo_s) > 0:
        axs3[3,1].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_levret1end_on2_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_levret1end_on2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_levret1end_on2_opto_s) > 0:
        axs3[3,1].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_levret1end_on2_opto_s,axis=0) , yerr = stats.sem(G_epoch1_levret1end_on2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(G_epoch2_levret1end_on2_opto_s) > 0:
        axs3[3,1].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_levret1end_on2_opto_s,axis=0) , yerr = stats.sem(G_epoch2_levret1end_on2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(G_epoch3_levret1end_on2_opto_s) > 0:
        axs3[3,1].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_levret1end_on2_opto_s,axis=0) , yerr = stats.sem(G_epoch3_levret1end_on2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(G_epoch1_levret1end_on2_con_l) > 0:
        axs3[3,1].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_levret1end_on2_con_l,axis=0) , yerr = stats.sem(G_epoch1_levret1end_on2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_levret1end_on2_con_l) > 0:
        axs3[3,1].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_levret1end_on2_con_l,axis=0) , yerr = stats.sem(G_epoch2_levret1end_on2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_levret1end_on2_con_l) > 0:
        axs3[3,1].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_levret1end_on2_con_l,axis=0) , yerr = stats.sem(G_epoch3_levret1end_on2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_levret1end_on2_chemo_l) > 0:
        axs3[3,1].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_levret1end_on2_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_levret1end_on2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_levret1end_on2_chemo_l) > 0:
        axs3[3,1].errorbar(session_id[1], np.mean(G_epoch2_levret1end_on2_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_levret1end_on2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_levret1end_on2_chemo_l) > 0:
        axs3[3,1].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_levret1end_on2_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_levret1end_on2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_levret1end_on2_opto_l) > 0:
        axs3[3,1].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_levret1end_on2_opto_l,axis=0) , yerr = stats.sem(G_epoch1_levret1end_on2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_levret1end_on2_opto_l) > 0:
        axs3[3,1].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_levret1end_on2_opto_l,axis=0) , yerr = stats.sem(G_epoch2_levret1end_on2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_levret1end_on2_opto_l) > 0:
        axs3[3,1].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_levret1end_on2_opto_l,axis=0) , yerr = stats.sem(G_epoch3_levret1end_on2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [4,0]
    if len(G_epoch1_base_amp1_con_s) > 0:
        axs3[4,0].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_base_amp1_con_s,axis=0) , yerr = stats.sem(G_epoch1_base_amp1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_base_amp1_con_s) > 0:
        axs3[4,0].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_base_amp1_con_s,axis=0) , yerr = stats.sem(G_epoch2_base_amp1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_base_amp1_con_s) > 0:
        axs3[4,0].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_base_amp1_con_s,axis=0) , yerr = stats.sem(G_epoch3_base_amp1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_base_amp1_chemo_s) > 0:
        axs3[4,0].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_base_amp1_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_base_amp1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_base_amp1_chemo_s) > 0:
        axs3[4,0].errorbar(session_id[0], np.mean(G_epoch2_base_amp1_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_base_amp1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_base_amp1_chemo_s) > 0:
        axs3[4,0].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_base_amp1_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_base_amp1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_base_amp1_opto_s) > 0:
        axs3[4,0].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_base_amp1_opto_s,axis=0) , yerr = stats.sem(G_epoch1_base_amp1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(G_epoch2_base_amp1_opto_s) > 0:
        axs3[4,0].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_base_amp1_opto_s,axis=0) , yerr = stats.sem(G_epoch2_base_amp1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(G_epoch3_base_amp1_opto_s) > 0:
        axs3[4,0].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_base_amp1_opto_s,axis=0) , yerr = stats.sem(G_epoch3_base_amp1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(G_epoch1_base_amp1_con_l) > 0:
        axs3[4,0].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_base_amp1_con_l,axis=0) , yerr = stats.sem(G_epoch1_base_amp1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_base_amp1_con_l) > 0:
        axs3[4,0].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_base_amp1_con_l,axis=0) , yerr = stats.sem(G_epoch2_base_amp1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_base_amp1_con_l) > 0:
        axs3[4,0].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_base_amp1_con_l,axis=0) , yerr = stats.sem(G_epoch3_base_amp1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_base_amp1_chemo_l) > 0:
        axs3[4,0].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_base_amp1_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_base_amp1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_base_amp1_chemo_l) > 0:
        axs3[4,0].errorbar(session_id[1], np.mean(G_epoch2_base_amp1_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_base_amp1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_base_amp1_chemo_l) > 0:
        axs3[4,0].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_base_amp1_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_base_amp1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_base_amp1_opto_l) > 0:
        axs3[4,0].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_base_amp1_opto_l,axis=0) , yerr = stats.sem(G_epoch1_base_amp1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_base_amp1_opto_l) > 0:
        axs3[4,0].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_base_amp1_opto_l,axis=0) , yerr = stats.sem(G_epoch2_base_amp1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_base_amp1_opto_l) > 0:
        axs3[4,0].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_base_amp1_opto_l,axis=0) , yerr = stats.sem(G_epoch3_base_amp1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [4,1]
    if len(G_epoch1_base_amp2_con_s) > 0:
        axs3[4,1].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_base_amp2_con_s,axis=0) , yerr = stats.sem(G_epoch1_base_amp2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_base_amp2_con_s) > 0:
        axs3[4,1].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_base_amp2_con_s,axis=0) , yerr = stats.sem(G_epoch2_base_amp2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_base_amp2_con_s) > 0:
        axs3[4,1].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_base_amp2_con_s,axis=0) , yerr = stats.sem(G_epoch3_base_amp2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_base_amp2_chemo_s) > 0:
        axs3[4,1].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_base_amp2_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_base_amp2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_base_amp2_chemo_s) > 0:
        axs3[4,1].errorbar(session_id[0], np.mean(G_epoch2_base_amp2_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_base_amp2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_base_amp2_chemo_s) > 0:
        axs3[4,1].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_base_amp2_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_base_amp2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_base_amp2_opto_s) > 0:
        axs3[4,1].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_base_amp2_opto_s,axis=0) , yerr = stats.sem(G_epoch1_base_amp2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(G_epoch2_base_amp2_opto_s) > 0:
        axs3[4,1].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_base_amp2_opto_s,axis=0) , yerr = stats.sem(G_epoch2_base_amp2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(G_epoch3_base_amp2_opto_s) > 0:
        axs3[4,1].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_base_amp2_opto_s,axis=0) , yerr = stats.sem(G_epoch3_base_amp2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(G_epoch1_base_amp2_con_l) > 0:
        axs3[4,1].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_base_amp2_con_l,axis=0) , yerr = stats.sem(G_epoch1_base_amp2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_base_amp2_con_l) > 0:
        axs3[4,1].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_base_amp2_con_l,axis=0) , yerr = stats.sem(G_epoch2_base_amp2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_base_amp2_con_l) > 0:
        axs3[4,1].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_base_amp2_con_l,axis=0) , yerr = stats.sem(G_epoch3_base_amp2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_base_amp2_chemo_l) > 0:
        axs3[4,1].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_base_amp2_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_base_amp2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_base_amp2_chemo_l) > 0:
        axs3[4,1].errorbar(session_id[1], np.mean(G_epoch2_base_amp2_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_base_amp2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_base_amp2_chemo_l) > 0:
        axs3[4,1].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_base_amp2_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_base_amp2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_base_amp2_opto_l) > 0:
        axs3[4,1].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_base_amp2_opto_l,axis=0) , yerr = stats.sem(G_epoch1_base_amp2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_base_amp2_opto_l) > 0:
        axs3[4,1].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_base_amp2_opto_l,axis=0) , yerr = stats.sem(G_epoch2_base_amp2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_base_amp2_opto_l) > 0:
        axs3[4,1].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_base_amp2_opto_l,axis=0) , yerr = stats.sem(G_epoch3_base_amp2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [5,0]
    if len(G_epoch1_max_vel1_con_s) > 0:
        axs3[5,0].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_max_vel1_con_s,axis=0) , yerr = stats.sem(G_epoch1_max_vel1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_max_vel1_con_s) > 0:
        axs3[5,0].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_max_vel1_con_s,axis=0) , yerr = stats.sem(G_epoch2_max_vel1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_max_vel1_con_s) > 0:
        axs3[5,0].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_max_vel1_con_s,axis=0) , yerr = stats.sem(G_epoch3_max_vel1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_max_vel1_chemo_s) > 0:
        axs3[5,0].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_max_vel1_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_max_vel1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_max_vel1_chemo_s) > 0:
        axs3[5,0].errorbar(session_id[0], np.mean(G_epoch2_max_vel1_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_max_vel1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_max_vel1_chemo_s) > 0:
        axs3[5,0].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_max_vel1_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_max_vel1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_max_vel1_opto_s) > 0:
        axs3[5,0].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_max_vel1_opto_s,axis=0) , yerr = stats.sem(G_epoch1_max_vel1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(G_epoch2_max_vel1_opto_s) > 0:
        axs3[5,0].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_max_vel1_opto_s,axis=0) , yerr = stats.sem(G_epoch2_max_vel1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(G_epoch3_max_vel1_opto_s) > 0:
        axs3[5,0].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_max_vel1_opto_s,axis=0) , yerr = stats.sem(G_epoch3_max_vel1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(G_epoch1_max_vel1_con_l) > 0:
        axs3[5,0].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_max_vel1_con_l,axis=0) , yerr = stats.sem(G_epoch1_max_vel1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_max_vel1_con_l) > 0:
        axs3[5,0].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_max_vel1_con_l,axis=0) , yerr = stats.sem(G_epoch2_max_vel1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_max_vel1_con_l) > 0:
        axs3[5,0].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_max_vel1_con_l,axis=0) , yerr = stats.sem(G_epoch3_max_vel1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_max_vel1_chemo_l) > 0:
        axs3[5,0].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_max_vel1_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_max_vel1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_max_vel1_chemo_l) > 0:
        axs3[5,0].errorbar(session_id[1], np.mean(G_epoch2_max_vel1_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_max_vel1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_max_vel1_chemo_l) > 0:
        axs3[5,0].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_max_vel1_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_max_vel1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_max_vel1_opto_l) > 0:
        axs3[5,0].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_max_vel1_opto_l,axis=0) , yerr = stats.sem(G_epoch1_max_vel1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_max_vel1_opto_l) > 0:
        axs3[5,0].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_max_vel1_opto_l,axis=0) , yerr = stats.sem(G_epoch2_max_vel1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_max_vel1_opto_l) > 0:
        axs3[5,0].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_max_vel1_opto_l,axis=0) , yerr = stats.sem(G_epoch3_max_vel1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [5,1]
    if len(G_epoch1_max_vel2_con_s) > 0:
        axs3[5,1].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_max_vel2_con_s,axis=0) , yerr = stats.sem(G_epoch1_max_vel2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_max_vel2_con_s) > 0:
        axs3[5,1].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_max_vel2_con_s,axis=0) , yerr = stats.sem(G_epoch2_max_vel2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_max_vel2_con_s) > 0:
        axs3[5,1].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_max_vel2_con_s,axis=0) , yerr = stats.sem(G_epoch3_max_vel2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_max_vel2_chemo_s) > 0:
        axs3[5,1].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_max_vel2_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_max_vel2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_max_vel2_chemo_s) > 0:
        axs3[5,1].errorbar(session_id[0], np.mean(G_epoch2_max_vel2_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_max_vel2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_max_vel2_chemo_s) > 0:
        axs3[5,1].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_max_vel2_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_max_vel2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_max_vel2_opto_s) > 0:
        axs3[5,1].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_max_vel2_opto_s,axis=0) , yerr = stats.sem(G_epoch1_max_vel2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(G_epoch2_max_vel2_opto_s) > 0:
        axs3[5,1].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_max_vel2_opto_s,axis=0) , yerr = stats.sem(G_epoch2_max_vel2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(G_epoch3_max_vel2_opto_s) > 0:
        axs3[5,1].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_max_vel2_opto_s,axis=0) , yerr = stats.sem(G_epoch3_max_vel2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(G_epoch1_max_vel2_con_l) > 0:
        axs3[5,1].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_max_vel2_con_l,axis=0) , yerr = stats.sem(G_epoch1_max_vel2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_max_vel2_con_l) > 0:
        axs3[5,1].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_max_vel2_con_l,axis=0) , yerr = stats.sem(G_epoch2_max_vel2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_max_vel2_con_l) > 0:
        axs3[5,1].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_max_vel2_con_l,axis=0) , yerr = stats.sem(G_epoch3_max_vel2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_max_vel2_chemo_l) > 0:
        axs3[5,1].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_max_vel2_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_max_vel2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_max_vel2_chemo_l) > 0:
        axs3[5,1].errorbar(session_id[1], np.mean(G_epoch2_max_vel2_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_max_vel2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_max_vel2_chemo_l) > 0:
        axs3[5,1].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_max_vel2_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_max_vel2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_max_vel2_opto_l) > 0:
        axs3[5,1].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_max_vel2_opto_l,axis=0) , yerr = stats.sem(G_epoch1_max_vel2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_max_vel2_opto_l) > 0:
        axs3[5,1].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_max_vel2_opto_l,axis=0) , yerr = stats.sem(G_epoch2_max_vel2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_max_vel2_opto_l) > 0:
        axs3[5,1].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_max_vel2_opto_l,axis=0) , yerr = stats.sem(G_epoch3_max_vel2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [6,0]
    if len(G_epoch1_velaton1_con_s) > 0:
        axs3[6,0].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_velaton1_con_s,axis=0) , yerr = stats.sem(G_epoch1_velaton1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_velaton1_con_s) > 0:
        axs3[6,0].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_velaton1_con_s,axis=0) , yerr = stats.sem(G_epoch2_velaton1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_velaton1_con_s) > 0:
        axs3[6,0].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_velaton1_con_s,axis=0) , yerr = stats.sem(G_epoch3_velaton1_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_velaton1_chemo_s) > 0:
        axs3[6,0].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_velaton1_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_velaton1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_velaton1_chemo_s) > 0:
        axs3[6,0].errorbar(session_id[0], np.mean(G_epoch2_velaton1_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_velaton1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_velaton1_chemo_s) > 0:
        axs3[6,0].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_velaton1_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_velaton1_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_velaton1_opto_s) > 0:
        axs3[6,0].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_velaton1_opto_s,axis=0) , yerr = stats.sem(G_epoch1_velaton1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(G_epoch2_velaton1_opto_s) > 0:
        axs3[6,0].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_velaton1_opto_s,axis=0) , yerr = stats.sem(G_epoch2_velaton1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(G_epoch3_velaton1_opto_s) > 0:
        axs3[6,0].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_velaton1_opto_s,axis=0) , yerr = stats.sem(G_epoch3_velaton1_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(G_epoch1_velaton1_con_l) > 0:
        axs3[6,0].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_velaton1_con_l,axis=0) , yerr = stats.sem(G_epoch1_velaton1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_velaton1_con_l) > 0:
        axs3[6,0].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_velaton1_con_l,axis=0) , yerr = stats.sem(G_epoch2_velaton1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_velaton1_con_l) > 0:
        axs3[6,0].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_velaton1_con_l,axis=0) , yerr = stats.sem(G_epoch3_velaton1_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_velaton1_chemo_l) > 0:
        axs3[6,0].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_velaton1_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_velaton1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_velaton1_chemo_l) > 0:
        axs3[6,0].errorbar(session_id[1], np.mean(G_epoch2_velaton1_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_velaton1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_velaton1_chemo_l) > 0:
        axs3[6,0].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_velaton1_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_velaton1_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_velaton1_opto_l) > 0:
        axs3[6,0].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_velaton1_opto_l,axis=0) , yerr = stats.sem(G_epoch1_velaton1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_velaton1_opto_l) > 0:
        axs3[6,0].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_velaton1_opto_l,axis=0) , yerr = stats.sem(G_epoch2_velaton1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_velaton1_opto_l) > 0:
        axs3[6,0].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_velaton1_opto_l,axis=0) , yerr = stats.sem(G_epoch3_velaton1_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])
    # [6,1]
    if len(G_epoch1_velaton2_con_s) > 0:
        axs3[6,1].errorbar(session_id[0] - 2.5*offset, np.mean(G_epoch1_velaton2_con_s,axis=0) , yerr = stats.sem(G_epoch1_velaton2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[0], label = 'Control_Epoch1')
    if len(G_epoch2_velaton2_con_s) > 0:
        axs3[6,1].errorbar(session_id[0] - 2.2*offset, np.mean(G_epoch2_velaton2_con_s,axis=0) , yerr = stats.sem(G_epoch2_velaton2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[1], label = 'Control_Epoch2')
    if len(G_epoch3_velaton2_con_s) > 0:
        axs3[6,1].errorbar(session_id[0] - 1.9*offset, np.mean(G_epoch3_velaton2_con_s,axis=0) , yerr = stats.sem(G_epoch3_velaton2_con_s, axis=0),fmt='o', capsize=4,color = black_shades[2], label = 'Control_Epoch3')

    if len(G_epoch1_velaton2_chemo_s) > 0:
        axs3[6,1].errorbar(session_id[0] - 0.3*offset, np.mean(G_epoch1_velaton2_chemo_s,axis=0) , yerr = stats.sem(G_epoch1_velaton2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[0], label = 'Chemo_Epoch1')
    if len(G_epoch2_velaton2_chemo_s) > 0:
        axs3[6,1].errorbar(session_id[0], np.mean(G_epoch2_velaton2_chemo_s,axis=0) , yerr = stats.sem(G_epoch2_velaton2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[1], label = 'Chemo_Epoch2')
    if len(G_epoch3_velaton2_chemo_s) > 0:
        axs3[6,1].errorbar(session_id[0] + 0.3*offset, np.mean(G_epoch3_velaton2_chemo_s,axis=0) , yerr = stats.sem(G_epoch3_velaton2_chemo_s, axis=0),fmt='o', capsize=4,color = red_shades[2], label = 'Chemo_Epoch3')

    if len(G_epoch1_velaton2_opto_s) > 0:
        axs3[6,1].errorbar(session_id[0] + 1.9*offset, np.mean(G_epoch1_velaton2_opto_s,axis=0) , yerr = stats.sem(G_epoch1_velaton2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[0], label = 'Opto_Epoch1')
    if len(G_epoch2_velaton2_opto_s) > 0:
        axs3[6,1].errorbar(session_id[0] + 2.2*offset, np.mean(G_epoch2_velaton2_opto_s,axis=0) , yerr = stats.sem(G_epoch2_velaton2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[1], label = 'Opto_Epoch2')
    if len(G_epoch3_velaton2_opto_s) > 0:
        axs3[6,1].errorbar(session_id[0] + 2.5*offset, np.mean(G_epoch3_velaton2_opto_s,axis=0) , yerr = stats.sem(G_epoch3_velaton2_opto_s, axis=0),fmt='o', capsize=4,color = skyblue_shades[2], label = 'Opto_Epoch3')

    if len(G_epoch1_velaton2_con_l) > 0:
        axs3[6,1].errorbar(session_id[1] - 2.5*offset, np.mean(G_epoch1_velaton2_con_l,axis=0) , yerr = stats.sem(G_epoch1_velaton2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[0])
    if len(G_epoch2_velaton2_con_l) > 0:
        axs3[6,1].errorbar(session_id[1] - 2.2*offset, np.mean(G_epoch2_velaton2_con_l,axis=0) , yerr = stats.sem(G_epoch2_velaton2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[1])
    if len(G_epoch3_velaton2_con_l) > 0:
        axs3[6,1].errorbar(session_id[1] - 1.9*offset, np.mean(G_epoch3_velaton2_con_l,axis=0) , yerr = stats.sem(G_epoch3_velaton2_con_l, axis=0),fmt='o', capsize=4,color = black_shades[2])

    if len(G_epoch1_velaton2_chemo_l) > 0:
        axs3[6,1].errorbar(session_id[1] - 0.3*offset, np.mean(G_epoch1_velaton2_chemo_l,axis=0) , yerr = stats.sem(G_epoch1_velaton2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[0])
    if len(G_epoch2_velaton2_chemo_l) > 0:
        axs3[6,1].errorbar(session_id[1], np.mean(G_epoch2_velaton2_chemo_l,axis=0) , yerr = stats.sem(G_epoch2_velaton2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[1])
    if len(G_epoch3_velaton2_chemo_l) > 0:
        axs3[6,1].errorbar(session_id[1] + 0.3*offset, np.mean(G_epoch3_velaton2_chemo_l,axis=0) , yerr = stats.sem(G_epoch3_velaton2_chemo_l, axis=0),fmt='o', capsize=4,color = red_shades[2])

    if len(G_epoch1_velaton2_opto_l) > 0:
        axs3[6,1].errorbar(session_id[1] + 1.9*offset, np.mean(G_epoch1_velaton2_opto_l,axis=0) , yerr = stats.sem(G_epoch1_velaton2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[0])
    if len(G_epoch2_velaton2_opto_l) > 0:
        axs3[6,1].errorbar(session_id[1] + 2.2*offset, np.mean(G_epoch2_velaton2_opto_l,axis=0) , yerr = stats.sem(G_epoch2_velaton2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[1])
    if len(G_epoch3_velaton2_opto_l) > 0:
        axs3[6,1].errorbar(session_id[1] + 2.5*offset, np.mean(G_epoch3_velaton2_opto_l,axis=0) , yerr = stats.sem(G_epoch3_velaton2_opto_l, axis=0),fmt='o', capsize=4,color = skyblue_shades[2])

    axs3[0,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[0,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[1,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[1,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[2,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[2,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[3,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[3,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[4,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[4,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[5,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[5,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[6,0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[6,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

    fig3.tight_layout()

    output_dir_onedrive = 'E:\\Ph.D\\Georgia Tech\\Behavior\\figures'
    output_figs_dir = output_dir_onedrive + subject + '/'  
    pdf_path = os.path.join(output_figs_dir, subject + '_Learning_Epoch_time_amp_quantification.pdf')
    
    plt.rcParams['pdf.fonttype'] = 42  # Ensure text is kept as text (not outlines)
    plt.rcParams['ps.fonttype'] = 42   # For compatibility with EPS as well, if needed

    # Save both plots into a single PDF file with each on a separate page
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)