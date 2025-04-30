#!/usr/bin/env python
# coding: utf-8

# In[1]:



from scipy.stats import sem
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfWriter, PdfReader
from scipy.interpolate import interp1d
from datetime import date
from statistics import mean 
import math
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.patches as mpatches #ohoolahan doge, daq, dip switches, diva, and doogie hausarus cavedog?
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from utils.util import get_figsize_from_pdf_spec

print_debug = 0

def gaussian_smooth(x, sigma=2):
    return gaussian_filter1d(x, sigma=sigma)

def ensure_list(var):
    return var if isinstance(var, list) else [var]

def save_image(filename): 
    
    p = PdfPages(filename+'.pdf') 
    fig_nums = plt.get_fignums()   
    figs = [plt.figure(n) for n in fig_nums] 
      
    for fig in figs:  
        
        fig.savefig(p, format='pdf', dpi=300)
           
    p.close() 
    
def remove_substrings(s, substrings):
    for sub in substrings:
        s = s.replace(sub, "")
    return s

def flip_underscore_parts(s):
    parts = s.split("_", 1)  # Split into two parts at the first underscore
    if len(parts) < 2:
        return s  # Return original string if no underscore is found
    return f"{parts[1]}_{parts[0]}"

def lowercase_h(s):
    return s.replace('H', 'h')

def get_earliest_lick(row):
    left = row['licks_left_start']
    right = row['licks_right_start']

    all_licks = []

    if isinstance(left, list):
        all_licks.extend([v for v in left if not np.isnan(v)])
    if isinstance(right, list):
        all_licks.extend([v for v in right if not np.isnan(v)])

    return min(all_licks) if all_licks else np.nan

def process_session_licks(session_data, session_idx):
    """
    Process licks for a single session.
    
    Args:
        session_data (dict): Contains left/right lick times and opto flags for trials.
    
    Returns:
        dict: Processed licks categorized into left/right and opto/non-opto.
    """
    processed_licks = {}
    
    conditions = ["left_trial_left_opto", "left_trial_left_no_opto", "left_trial_right_opto", "left_trial_right_no_opto",
                  "right_trial_left_opto", "right_trial_left_no_opto", "right_trial_right_opto", "right_trial_right_no_opto"]
    
    for condition in conditions:
        processed_licks[f"{condition}_starts"] = []
        processed_licks[f"{condition}_stops"] = []
    
    raw_data = session_data['raw']
    
    numTrials = raw_data[session_idx]['nTrials']
        
    trial_types = session_data['trial_type'][session_idx]
    opto_flags = session_data['opto_trial'][session_idx]
    
    for trial in range(numTrials):
           
        licks = {}
        
        if not 'Port1In' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_left_start'] = [np.nan]
        elif type(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']) == float:
            licks['licks_left_start'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']]
        else:
            licks['licks_left_start'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']
            
        if not 'Port1Out' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_left_stop'] = [np.nan]
        elif type(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']) == float:
            licks['licks_left_stop'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']]
        else:
            licks['licks_left_stop'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']        
    
        if not 'Port3In' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_right_start'] = [np.nan]
        elif type(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']) == float:
            licks['licks_right_start'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']]
        else:
            licks['licks_right_start'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']         
        
        if not 'Port3Out' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_right_stop'] = [np.nan]
        elif type(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']) == float:
            licks['licks_right_stop'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']]
        else:
            licks['licks_right_stop'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']         
        
        trial_type = "left_trial" if trial_types[trial] == 1 else "right_trial" 
        
        is_opto = opto_flags[trial]
        
        for side in ["left", "right"]:
            key = f"{trial_type}_{side}_opto" if is_opto else f"{trial_type}_{side}_no_opto"
            processed_licks[f"{key}_starts"].append(licks[f"licks_{side}_start"])
            processed_licks[f"{key}_stops"].append(licks[f"licks_{side}_stop"])
    
    # Sorting each condition by the first lick time
    processed_licks_sorted = {}
    for condition in conditions:
        starts = processed_licks[f"{condition}_starts"]
        stops = processed_licks[f"{condition}_stops"]
        
        # Convert lists to arrays for sorting
        starts_array = np.array(starts, dtype=object)
        stops_array = np.array(stops, dtype=object)
    
        # Sort by first valid element, placing NaNs at the end
        valid_indices = [i for i in range(len(starts_array)) if not np.isnan(starts_array[i][0])] # fix for 'empty' lists TypeError: 'int' object is not subscriptable
        nan_indices = [i for i in range(len(starts_array)) if np.isnan(starts_array[i][0])]
    
        sorted_valid_indices = sorted(valid_indices, key=lambda i: starts_array[i][0])
        sorted_indices = sorted_valid_indices + nan_indices  # NaNs go at the end
    
        processed_licks_sorted[f"{condition}_starts"] = [starts_array[i] for i in sorted_indices]
        processed_licks_sorted[f"{condition}_stops"] = [stops_array[i] for i in sorted_indices]    
                
    return processed_licks, processed_licks_sorted



def process_session_licks_with_valve_times(session_data, session_idx):
    """
    Process licks for a single session.
    
    Args:
        session_data (dict): Contains left/right lick times and opto flags for trials.
    
    Returns:
        dict: Processed licks categorized into left/right and opto/non-opto.
    """
    processed_licks = {}
    
    conditions = ["left_trial_left_opto", "left_trial_left_no_opto", "left_trial_right_opto", "left_trial_right_no_opto",
                  "right_trial_left_opto", "right_trial_left_no_opto", "right_trial_right_opto", "right_trial_right_no_opto"]
    
    for condition in conditions:
        processed_licks[f"{condition}_starts"] = []
        processed_licks[f"{condition}_stops"] = []
        processed_licks[f"{condition}_valve_starts"] = []
        processed_licks[f"{condition}_valve_stops"] = []
    
    raw_data = session_data['raw']
    
    numTrials = raw_data[session_idx]['nTrials']
        
    trial_types = session_data['trial_type'][session_idx]
    opto_flags = session_data['opto_trial'][session_idx]
    
    for trial in range(numTrials):
           
        licks = {}
        valve_times = []
        
        if not 'Port1In' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_left_start'] = [np.float64(np.nan)]
        elif type(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']) == float:
            licks['licks_left_start'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']]
        else:
            licks['licks_left_start'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']
            
        if not 'Port1Out' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_left_stop'] = [np.float64(np.nan)]
        elif type(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']) == float:
            licks['licks_left_stop'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']]
        else:
            licks['licks_left_stop'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']        
    
        if not 'Port3In' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_right_start'] = [np.float64(np.nan)]
        elif type(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']) == float:
            licks['licks_right_start'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']]
        else:
            licks['licks_right_start'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']         
        
        if not 'Port3Out' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_right_stop'] = [np.float64(np.nan)]
        elif type(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']) == float:
            licks['licks_right_stop'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']]
        else:
            licks['licks_right_stop'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']  

        trial_type = "left_trial" if trial_types[trial] == 1 else "right_trial" 
        
        # Track valve open/close times for the trial (start/stop)
        if 'Reward' in raw_data[session_idx]['RawEvents']['Trial'][trial]['States']:
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][0])
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][1])  
        else:
            valve_times.append(np.float64(np.nan))
            valve_times.append(np.float64(np.nan))
        
        is_opto = opto_flags[trial]
        
        for side in ["left", "right"]:
            key = f"{trial_type}_{side}_opto" if is_opto else f"{trial_type}_{side}_no_opto"
            processed_licks[f"{key}_starts"].append(licks[f"licks_{side}_start"])
            processed_licks[f"{key}_stops"].append(licks[f"licks_{side}_stop"])
            processed_licks[f"{key}_valve_starts"].append(valve_times[0])
            processed_licks[f"{key}_valve_stops"].append(valve_times[1])

    return processed_licks

def process_session_licks_with_valve_times_df(session_data, session_idx):
    """
    Process licks for a single session.
    
    Args:
        session_data (dict): Contains left/right lick times and opto flags for trials.
    
    Returns:
        dict: Processed licks categorized into left/right and opto/non-opto.
    """
    processed_licks = []
    
    # conditions = ["left_trial_left_opto", "left_trial_left_no_opto", "left_trial_right_opto", "left_trial_right_no_opto",
    #               "right_trial_left_opto", "right_trial_left_no_opto", "right_trial_right_opto", "right_trial_right_no_opto"]
    
    # for condition in conditions:
    #     processed_licks[f"{condition}_starts"] = []
    #     processed_licks[f"{condition}_stops"] = []
    #     processed_licks[f"{condition}_valve_starts"] = []
    #     processed_licks[f"{condition}_valve_stops"] = []
    
    raw_data = session_data['raw']
    
    numTrials = raw_data[session_idx]['nTrials']
        
    trial_types = session_data['trial_type'][session_idx]
    opto_flags = session_data['opto_trial'][session_idx]
    
    for trial in range(numTrials):
           
        licks = {}
        valve_times = []
        rewarded = 0
        isNaive = 0
        no_lick = 0
        
        alignment = 0
        # alignment = outcome_time[trial]
        alignment = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['WindowChoice'][0]          
        
        if not 'Port1In' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_left_start'] = [np.float64(np.nan)]
        elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In'], (float, int)):
            licks['licks_left_start'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']]
        else:
            licks['licks_left_start'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']
            
        if not 'Port1Out' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_left_stop'] = [np.float64(np.nan)]
        elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out'], (float, int)):
            licks['licks_left_stop'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']]
        else:
            licks['licks_left_stop'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']        
    
        if not 'Port3In' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_right_start'] = [np.float64(np.nan)]
        elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In'], (float, int)):
            licks['licks_right_start'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']]
        else:
            licks['licks_right_start'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']         
        
        if not 'Port3Out' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_right_stop'] = [np.float64(np.nan)]
        elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out'], (float, int)):
            licks['licks_right_stop'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']]
        else:
            licks['licks_right_stop'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']  

        # np.array([x - alignment for x in licks['licks_left_start']])
        licks['licks_left_start'] = [x - alignment for x in licks['licks_left_start']]
        licks['licks_left_stop'] = [x - alignment for x in licks['licks_left_stop']]
        licks['licks_right_start'] = [x - alignment for x in licks['licks_right_start']]
        licks['licks_right_stop'] = [x - alignment for x in licks['licks_right_stop']]


        trial_type = "left" if trial_types[trial] == 1 else "right" 
        
        # Track valve open/close times for the trial (start/stop)
        if 'Reward' in raw_data[session_idx]['RawEvents']['Trial'][trial]['States']:
            is_naive = 0
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][0] - alignment)
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][1] - alignment)
            if not np.isnan(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][0]):
                rewarded = 1
            else:
                rewarded = 0
        elif 'NaiveRewardDeliver' in raw_data[session_idx]['RawEvents']['Trial'][trial]['States']:  
            is_naive = 1
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['NaiveRewardDeliver'][0] - alignment)
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['NaiveRewardDeliver'][1] - alignment)
            if not np.isnan(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['NaiveRewardDeliver'][0]):
                rewarded = 1
            else:
                rewarded = 0
        else:
            print('what this????')
            valve_times.append(np.float64(np.nan))
            valve_times.append(np.float64(np.nan))
            
        if 'DidNotChoose' in raw_data[session_idx]['RawEvents']['Trial'][trial]['States']:  
            if not np.isnan(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['DidNotChoose'][0]):
                no_lick = 1
            

        is_opto = opto_flags[trial]   

        processed_licks.append({
            "trial": trial,
            "trial_side": trial_type,
            "is_opto": is_opto,
            "is_naive": is_naive,
            "rewarded": rewarded,
            "no_lick": no_lick,            
            "licks_left_start": licks['licks_left_start'],
            "licks_left_stop": licks['licks_left_stop'],
            "licks_right_start": licks['licks_right_start'],
            "licks_right_stop": licks['licks_right_stop'],
            "valve_start": valve_times[0],
            "valve_stop": valve_times[1]
        })
        
    return pd.DataFrame(processed_licks)


    # return processed_licks


def sort_licks_left_right_opto_non(processed_licks):
    processed_licks_sorted = {}
    
    for trial_side in ["left", "right"]:
        condition_pairs = [(f"{trial_side}_trial_left_no_opto", f"{trial_side}_trial_right_no_opto"),
                           (f"{trial_side}_trial_left_opto", f"{trial_side}_trial_right_opto")]
        
        for left_cond, right_cond in condition_pairs:
            left_starts = processed_licks[f"{left_cond}_starts"]
            right_starts = processed_licks[f"{right_cond}_starts"]
            left_stops = processed_licks[f"{left_cond}_stops"]
            right_stops = processed_licks[f"{right_cond}_stops"]
            left_valve_starts = processed_licks[f"{left_cond}_valve_starts"]
            right_valve_starts = processed_licks[f"{right_cond}_valve_starts"]
            left_valve_stops = processed_licks[f"{left_cond}_valve_stops"]
            right_valve_stops = processed_licks[f"{right_cond}_valve_stops"]
            
            if trial_side == "left":
                sort_reference = left_starts
                backup_reference = right_starts
            else:
                sort_reference = right_starts
                backup_reference = left_starts
            
            valid_indices = [i for i in range(len(sort_reference)) if isinstance(sort_reference[i], (list, np.ndarray)) and len(sort_reference[i]) > 0 and not np.isnan(sort_reference[i][0])]
            nan_indices = [i for i in range(len(sort_reference)) if i not in valid_indices]
            
            sorted_valid_indices = sorted(valid_indices, key=lambda i: sort_reference[i][0])
            nan_sorted_indices = sorted(nan_indices, key=lambda i: backup_reference[i][0] if isinstance(backup_reference[i], (list, np.ndarray)) and len(backup_reference[i]) > 0 and not np.isnan(backup_reference[i][0]) else float('inf'))
            
            sorted_indices = sorted_valid_indices + nan_sorted_indices
            
            processed_licks_sorted[f"{left_cond}_starts"] = [left_starts[i] for i in sorted_indices]
            processed_licks_sorted[f"{right_cond}_starts"] = [right_starts[i] for i in sorted_indices]
            processed_licks_sorted[f"{left_cond}_stops"] = [left_stops[i] for i in sorted_indices]
            processed_licks_sorted[f"{right_cond}_stops"] = [right_stops[i] for i in sorted_indices]
            processed_licks_sorted[f"{left_cond}_valve_starts"] = [left_valve_starts[i] for i in sorted_indices]
            processed_licks_sorted[f"{right_cond}_valve_starts"] = [right_valve_starts[i] for i in sorted_indices]
            processed_licks_sorted[f"{left_cond}_valve_stops"] = [left_valve_stops[i] for i in sorted_indices]
            processed_licks_sorted[f"{right_cond}_valve_stops"] = [right_valve_stops[i] for i in sorted_indices]
            
    return processed_licks_sorted

def sort_licks_df(licks_df):
    """
    Sorts the licks DataFrame by the earliest lick start time within each trial.
    
    Args:
        licks_df (pd.DataFrame): DataFrame containing lick times.
    
    Returns:
        pd.DataFrame: Sorted DataFrame.
    """
    # licks_df['earliest_lick'] = licks_df[['licks_left_start', 'licks_right_start']].min(axis=1)
    licks_df['earliest_lick'] = licks_df.apply(get_earliest_lick, axis=1)      
    sorted_df = licks_df.sort_values(by='earliest_lick', ascending=True, na_position='last').drop(columns=['earliest_lick'])
    return sorted_df


def plot_lick_traces_df(lick_data, trial_side='all', title="Lick Traces", ax=None):
    """
    Plot lick traces for the specified trial side ("left" or "right"), ensuring left and right licks
    from the same trial appear on the same row.
    
    Args:
        lick_data (pd.DataFrame): The DataFrame containing lick start and stop times, and other trial info.
        trial_side (str): 'left', 'right', or 'all'. Filters trials by side.
        title (str): The title of the plot.
    """
    # Define colors for better visual distinction
    # colors = {
    #     "left_no_opto": "#1f77b4",   # Blue
    #     "right_no_opto": "#ff7f0e",  # Orange
    #     "left_opto": "#2ca02c",      # Green
    #     "right_opto": "#d62728",     # Red
    #     "valve": "#9467bd"           # Purple for valve
    # }
    
    colors = {
        "left_no_opto": "#1f77b4",   # Blue
        "right_no_opto": "#d62728",  # Red (updated)
        "left_opto": "#17becf",      # Cyan (updated)
        "right_opto": "#9467bd",     # Violet (updated)
        "valve": "#2ca02c"           # Green for valve
    }    
    
    # Set up the plot
    if ax ==  None:
        fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trials")

    # Filter trials based on trial_side
    if trial_side == 'left':
        filtered_licks = lick_data[lick_data['trial_side'] == 'left']
    elif trial_side == 'right':
        filtered_licks = lick_data[lick_data['trial_side'] == 'right']
    elif trial_side == 'all':
        filtered_licks = lick_data
    else:
        raise ValueError(f"Invalid trial_side: {trial_side}. Must be 'left', 'right', or 'all'.")


    # Define colors for 'left' and 'right' trials
    # side_colors = {
    #     "left": "#1f77b4",   # Blue for left trials
    #     "right": "#d62728",  # Red for right trials
    # }
    # side_colors = {
    #     "left": "#0000CD",   # mediumblue for left trials
    #     "right": "#FF00FF",  # magenta for right trials
    # }
    # side_colors = {
    #     "left": "#0000CD",   # mediumblue for left trials
    #     "right": "#d62728",  # red for right trials
    # }
    side_colors = {
        "left": "#1f77b4",   # mediumblue for left trials   # (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
        "right": "#d62728",  # red for right trials   # (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    }   



    # Store trial sides and their respective colors
    trial_sides = []  # Store side info for plotting
    trial_colors = []  # Store color for each trial
    trial_offset = 1  # Initialize the trial offset
    
    # Loop through the filtered trials and plot the lick traces
    for idx, row in filtered_licks.iterrows():
        # Get the color for this trial based on its side
        color = [mcolors.hex2color(side_colors[row['trial_side']])]
        # Store the trial's side and color
        trial_sides.append(row['trial_side'])
        trial_colors.append(color)  

      
        trial_num = row['trial']
        lick_left_start = row['licks_left_start']
        lick_left_stop = row['licks_left_stop']
        lick_right_start = row['licks_right_start']
        lick_right_stop = row['licks_right_stop']
        is_opto = row['is_opto']
        
        # Determine trial type (left or right side) and condition (opto or no opto)
        if row['trial_side'] == 'left':
            left_cond = 'left_opto' if is_opto else 'left_no_opto'
            right_cond = 'right_opto' if is_opto else 'right_no_opto'   
            # ax.axhline(y=trial_offset, color='blue', linestyle='--', linewidth=0.5, alpha=0.7)
        else:
            left_cond = 'left_opto' if is_opto else 'left_no_opto'
            right_cond = 'right_opto' if is_opto else 'right_no_opto'
            # ax.axhline(y=trial_offset, color='r', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Plot left licks
        if len(lick_left_start) > 0 and len(lick_left_stop) > 0:
            for start, stop in zip(lick_left_start, lick_left_stop):
                if not np.isnan(start) and not np.isnan(stop):
                    ax.plot([start, stop], [trial_offset] * 2, color=colors[left_cond], alpha=0.7, lw=0.5)
        
        # Plot right licks
        if len(lick_right_start) > 0 and len(lick_right_stop) > 0:
            for start, stop in zip(lick_right_start, lick_right_stop):
                if not np.isnan(start) and not np.isnan(stop):
                    ax.plot([start, stop], [trial_offset] * 2, color=colors[right_cond], alpha=0.9, lw=0.5)                     

        # Plot valve times
        valve_start = row['valve_start']
        valve_stop = row['valve_stop']
        if not np.isnan(valve_start) and not np.isnan(valve_stop):
            ax.plot([valve_start, valve_stop], [trial_offset, trial_offset], color=colors["valve"], lw=2, alpha=0.3)

        trial_offset += 1  # Increment the trial offset

    # Create a new axis for the colorbar-like row labels
    cbar_ax = ax.inset_axes([1.00, 0, 0.02, 1], transform=ax.transAxes)  # x, y, w, h
    cbar_ax.set_yticks([])  # Remove y-axis ticks
    cbar_ax.set_xticks([])  # Remove x-axis ticks
    
    # Add text labels directly on the colorbar
    # cbar_ax.text(1.5, len(lick_data) - 1, "Left", va="center", ha="left", fontsize=12, color="blue", fontweight="bold")
    # cbar_ax.text(1.5, 0, "Right", va="center", ha="left", fontsize=12, color="red", fontweight="bold")

    # Create a color matrix (each row is a color corresponding to a trial)
    color_matrix = np.array(trial_colors, dtype=float).reshape(len(trial_colors), 1, 3)  # Convert hex to RGB

    # Display the color matrix as an image
    # cbar_ax.imshow(color_matrix, aspect='auto', origin='lower', extent=[0, 1, 0, len(filtered_licks)])
    cbar_ax.imshow(color_matrix, aspect='auto', origin='lower')

    # Explicitly set the colorbar y-limits to match the number of rows
    cbar_ax.set_ylim(-1, len(filtered_licks) + 1)
    
    # ** Add a separate legend for the colorbar **
    legend_ax = ax.inset_axes([1.0, 0.8, 0.15, 0.1], transform=ax.transAxes)  # x, y, w, h
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.axis("off")

    # Create manual patches for legend
    left_patch = mpatches.Patch(color=side_colors["left"], label="Left Trials")
    right_patch = mpatches.Patch(color=side_colors["right"], label="Right Trials")   
    
    legend_ax.legend(handles=[left_patch, right_patch], loc="center", fontsize=10, frameon=False)

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color=colors["left_no_opto"], lw=2, label='Left Lick Control'),
        Line2D([0], [0], color=colors["right_no_opto"], lw=2, label='Right Lick Control'),
        Line2D([0], [0], color=colors["left_opto"], lw=2, label='Left Lick Opto'),
        Line2D([0], [0], color=colors["right_opto"], lw=2, label='Right Lick Opto'),
        Line2D([0], [0], color=colors["valve"], lw=2, label="Valve Open")
    ]
    
    ax.legend(handles=legend_elements)
    
    ax.set_xlim(x_lim_left, x_lim_right)    
    ax.set_ylim(0, len(filtered_licks) + 1)
    
    # # Adding the colorbar
    # norm = plt.Normalize(vmin=0, vmax=3)  # Adjust based on number of categories
    # cmap = cm.get_cmap("coolwarm")  # Choose a colormap (or you could define your own)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # Empty array, as we're just using the color map for the colorbar
    # cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    # cbar.set_label('Trial Categories')  # Label for the colorbar    
    
    # Define colors for 'left' and 'right' trials
    # side_colors = {
    #     "left": "#1f77b4",   # Blue
    #     "right": "#ff7f0e",  # Orange
    # }    
    
    # Add the colorbar based on trial side
    # trial_side_mapping = {'left': 0, 'right': 1}  # Mapping of trial sides to numeric values for colorbar
    # norm = mcolors.BoundaryNorm([0, 1, 2], 2)  # Normalize the sides to range 0 to 1 for colorbar
    # cmap = mcolors.ListedColormap([side_colors['left'], side_colors['right']])
    
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # Empty array, as we're just using the color map for the colorbar
    # cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    # cbar.set_ticks([0, 1])
    # cbar.set_ticklabels(['Left Trials', 'Right Trials'])
    # cbar.set_label('Trial Side')  # Label for the colorbar    

    if ax ==  None:
        plt.show()

def plot_avg_lick_traces(avg_lick_times, trial_side='all', title="Lick Traces", ax=None):   
    """
    Plot average lick traces for different trial conditions using the same color scheme.
    
    Args:
        avg_lick_times (dict): Dictionary containing average lick start and stop times for different trial conditions.
        title (str): Plot title.
    """
    # fig, ax = plt.subplots(figsize=(12, 6))

    if trial_side == 'left':
        condition_colors = {
            "control_rewarded": "#1f77b4",   # Blue
            "opto_rewarded": "#17becf",      # Cyan 
        }
    elif trial_side == 'right':
        condition_colors = {
            "control_rewarded": "#d62728",  # Red
            "opto_rewarded": "#9467bd",     # Violet
        }    
    

    # Plot the average lick traces
    for idx, (condition, times) in enumerate(avg_lick_times.items()):
        lick_starts = times["starts"]
        lick_stops = times["stops"]
                     

        if (isinstance(lick_starts, np.ndarray) and lick_starts.dtype == np.float64) and \
            (isinstance(lick_stops, np.ndarray) and lick_stops.dtype == np.float64):
        
            try:
                if len(lick_starts) > 0 and len(lick_stops) > 0:
                # if lick_starts and lick_stops:
                    for start, stop in zip(lick_starts, lick_stops):
                        if not np.isnan(start) and not np.isnan(stop):
                            ax.plot([start, stop], [idx, idx], color=condition_colors.get(condition, "black"), alpha=0.9, lw=3)  
                        
            except Exception as e:
                print(f"An error occurred: {e}")                    
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Conditions")
    ax.set_yticks(range(len(avg_lick_times)))
    ax.set_yticklabels([cond.replace("_", " ").title() for cond in avg_lick_times.keys()])

    # Adjust the y-limits to provide more space in the center of the plot
    ax.set_ylim(-1, len(avg_lick_times))  # Set y-limits with a bit of padding above and below the traces

    # Create custom legend
    if trial_side == 'left':
        legend_elements = [
            Line2D([0], [0], color=condition_colors["control_rewarded"], lw=2, label='Left Lick Control'),
            Line2D([0], [0], color=condition_colors["opto_rewarded"], lw=2, label='Left Lick Opto'),
        ]
    elif trial_side == 'right':
        legend_elements = [
            Line2D([0], [0], color=condition_colors["control_rewarded"], lw=2, label='Right Lick Control'),
            Line2D([0], [0], color=condition_colors["opto_rewarded"], lw=2, label='Right Lick Opto'),
        ]        
    
    ax.legend(handles=legend_elements, loc='upper right')    
    
    if ax ==  None:
        plt.show()    

def plot_lick_traces(lick_data, trial_side, title="Lick Traces"):
    """
    Plot lick traces for the specified trial side ("left" or "right").
    
    Args:
        lick_data (dict): Processed lick data.
        trial_side (str): Either "left" or "right" to specify which trials to plot.
        title (str): Plot title.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    conditions = [f"{trial_side}_trial_left_no_opto", f"{trial_side}_trial_left_opto", 
                  f"{trial_side}_trial_right_no_opto", f"{trial_side}_trial_right_opto"]
    colors = ['green', 'lightgreen', 'black', 'darkgray']
    labels = [f"{trial_side.capitalize()} Trial - Left No Opto", f"{trial_side.capitalize()} Trial - Left Opto", 
              f"{trial_side.capitalize()} Trial - Right No Opto", f"{trial_side.capitalize()} Trial - Right Opto"]
    
      
    trial_offset = 0  # Reset trial offset for each side
    for cond, color, label in zip(conditions, colors, labels):
        for trial_starts, trial_stops in zip(lick_data[f"{cond}_starts"], lick_data[f"{cond}_stops"]):
            valid_trials = False  # Flag to check if there are any valid start/stop pairs
            
            for start, stop in zip(trial_starts, trial_stops):
                if not np.isnan(start) and not np.isnan(stop):
                    ax.plot([start, stop], [trial_offset] * 2, color=color, alpha=0.7)
                    valid_trials = True  # Mark as a valid trial
            
            # Increment the trial_offset only if valid trials were plotted
            if valid_trials:
                trial_offset += 1    
    
    # Create custom legend with correct colors
    legend_elements = [Line2D([0], [0], color='green', lw=2, label=labels[0]),
                       Line2D([0], [0], color='lightgreen', lw=2, label=labels[1]),
                       Line2D([0], [0], color='black', lw=2, label=labels[2]),
                       Line2D([0], [0], color='lightgray', lw=2, label=labels[3])]    
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trials")
    ax.set_title(title)
    ax.legend(handles=legend_elements, loc='upper right')
    plt.show()

def plot_avg_lick_traces_from_matrix(avg_lick_traces, trial_side='all', title="Lick Traces", ax=None):
    """
    Plot average lick traces as intensity rows where color reflects lick probability.
    
    Args:
        avg_lick_traces (dict): Keys = condition labels, values = 1D np.arrays of avg lick probability (0–1).
        time_vector (np.array): 1D array of timepoints.
        trial_side (str): 'left', 'right', or 'all' for color scheme.
        title (str): Plot title.
        ax (matplotlib.axes.Axes): Optional axis to plot into.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Define condition colors
    if trial_side == 'left':
        condition_colors = {
            "control_rewarded": "#1f77b4",   # Blue
            "opto_rewarded": "#17becf",      # Cyan 
        }
    elif trial_side == 'right':
        condition_colors = {
            "control_rewarded": "#d62728",  # Red
            "opto_rewarded": "#9467bd",     # Violet
        }
    else:
        condition_colors = {k: f"C{i}" for i, k in enumerate(avg_lick_traces.keys())}

    # Plot as intensity row (fill_between with alpha based on value)
    for idx, (condition, avg_trace) in enumerate(avg_lick_traces.items()):
        if isinstance(avg_trace, np.ndarray) and avg_trace.ndim == 1:
            base_y = np.full_like(time_vector, idx, dtype=float)
            color = condition_colors.get(condition, "black")
            avg_trace = gaussian_smooth(avg_trace*100)
            
            ax.plot(time_vector, avg_trace, color=color, alpha=0.7, lw=1)                        
            # for t, val in zip(time_vector, avg_trace):
            #     if not np.isnan(val) and val > 0:
            #         ax.plot([t, t], [idx - 0.4, idx + 0.4], color=color, alpha=val, lw=1)
            #         # ax.plot(t, idx, color=color, alpha=val, lw=1)

    # Axis setup
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(x_lim_left, x_lim_right)    
    ax.set_ylabel("Lick Probability")
    # ax.set_yticks(range(len(avg_lick_traces)))
    # ax.set_yticklabels([cond.replace("_", " ").title() for cond in avg_lick_traces.keys()])
    # ax.set_ylim(-1, len(avg_lick_traces))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0', '25', '50', '75', '100'])
    ax.set_ylim(0, 100)    

    # Legend
    if trial_side in ['left', 'right']:
        legend_elements = [
            Line2D([0], [0], color=condition_colors["control_rewarded"], lw=2, label=f'{trial_side.title()} Lick Control'),
            Line2D([0], [0], color=condition_colors["opto_rewarded"], lw=2, label=f'{trial_side.title()} Lick Opto'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    if ax is None:
        plt.show()
    
def plot_lick_traces_valves_side_type(lick_data, trial_side, title="Lick Traces"):
    """
    Plot lick traces for the specified trial side ("left" or "right"), ensuring left and right licks
    from the same trial appear on the same row.
    
    Args:
        lick_data (dict): Processed lick data.
        trial_side (str): Either "left" or "right" to specify which trials to plot.
        title (str): Plot title.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set title based on trial_side
    if trial_side == "left":
        title = "Left Trials - Short ISI - " + title
    else:
        title = "Right Trials - Long ISI - " + title
    
    
    # Define colors for better visual distinction
    colors = {
        "left_no_opto": "#1f77b4",   # Blue
        "right_no_opto": "#ff7f0e",  # Orange
        "left_opto": "#2ca02c",     # Green
        "right_opto": "#d62728" ,     # Red
        "valve": "#9467bd"           #
    }
    
    # Organize conditions by opto/non-opto first, then left/right
    conditions = [(f"{trial_side}_trial_left_no_opto", f"{trial_side}_trial_right_no_opto", colors["left_no_opto"], colors["right_no_opto"]),
                  (f"{trial_side}_trial_left_opto", f"{trial_side}_trial_right_opto", colors["left_opto"], colors["right_opto"])]        
    
    # # Organize conditions by opto/non-opto first, then left/right
    # conditions = [(f"{trial_side}_trial_left_no_opto", f"{trial_side}_trial_right_no_opto", 'green', 'black'),
    #               (f"{trial_side}_trial_left_opto", f"{trial_side}_trial_right_opto", 'lightgreen', 'darkgray')]    
    
    trial_offset = 0  # Initialize the trial offset
    
    for (left_cond, right_cond, left_color, right_color) in conditions:
        for trial_idx in range(len(lick_data[f"{left_cond}_starts"])):
            
            # Get left licks
            left_starts = lick_data[f"{left_cond}_starts"][trial_idx]
            left_stops = lick_data[f"{left_cond}_stops"][trial_idx]
            
            # Get right licks
            right_starts = lick_data[f"{right_cond}_starts"][trial_idx]
            right_stops = lick_data[f"{right_cond}_stops"][trial_idx]
            
            # Get valve timings
            valve_starts = lick_data[f"{left_cond}_valve_starts"][trial_idx]  # Assume same valve times for both
            valve_stops = lick_data[f"{left_cond}_valve_stops"][trial_idx]
            
            valid_trials = False  # Flag to check if there are valid trials
            
            # Plot left licks
            if len(left_starts) > 0 and len(left_stops) > 0:
                for start, stop in zip(left_starts, left_stops):
                    if not np.isnan(start) and not np.isnan(stop):
                        ax.plot([start, stop], [trial_offset] * 2, color=left_color, alpha=0.7, lw=0.5)
                        valid_trials = True
            
            # Plot right licks
            if len(right_starts) > 0 and len(right_stops) > 0:
                for start, stop in zip(right_starts, right_stops):
                    if not np.isnan(start) and not np.isnan(stop):
                        ax.plot([start, stop], [trial_offset] * 2, color=right_color, alpha=0.9, lw=0.5)
                        valid_trials = True
            
            # Plot valve times
            if valid_trials and not np.isnan(valve_starts) and not np.isnan(valve_stops):
                # for v_start, v_stop in zip(valve_starts, valve_stops):
                # if not np.isnan(v_start) and not np.isnan(v_stop):
                ax.plot([valve_starts, valve_stops], [trial_offset, trial_offset], color=colors["valve"], lw=2, alpha=0.3)
            
            # Increment trial offset only if there were valid trials
            if valid_trials:
                trial_offset += 1
    
    labels = ["Control", "Opto"]    
    
    # Create custom legend
    legend_elements = [Line2D([0], [0], color=colors["left_no_opto"], lw=2, label='Left Lick ' + labels[0]),
                       Line2D([0], [0], color=colors["right_no_opto"], lw=2, label='Right Lick ' + labels[0]),
                       Line2D([0], [0], color=colors["left_opto"], lw=2, label='Left Lick ' + labels[1]),
                       Line2D([0], [0], color=colors["right_opto"], lw=2, label='Right Lick ' + labels[1]),
                       Line2D([0], [0], color=colors["valve"], lw=2, label="Valve Open/Close")]    
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trials")
    ax.set_title(title)
    ax.legend(handles=legend_elements)             
    plt.show()                


def plot_lick_traces_valves(lick_data, trial_side, title="Lick Traces"):
    """
    Plot lick traces for the specified trial side ("left" or "right"), ensuring left and right licks
    from the same trial appear on the same row.
    
    Args:
        lick_data (dict): Processed lick data.
        trial_side (str): Either "left" or "right" to specify which trials to plot.
        title (str): Plot title.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set title based on trial_side
    if trial_side == "left":
        title = "Left Trials - Short ISI - " + title
    else:
        title = "Right Trials - Long ISI - " + title
    
    
    # Define colors for better visual distinction
    colors = {
        "left_no_opto": "#1f77b4",   # Blue
        "right_no_opto": "#ff7f0e",  # Orange
        "left_opto": "#2ca02c",     # Green
        "right_opto": "#d62728" ,     # Red
        "valve": "#9467bd"           #
    }
    
    # Organize conditions by opto/non-opto first, then left/right
    conditions = [(f"{trial_side}_trial_left_no_opto", f"{trial_side}_trial_right_no_opto", colors["left_no_opto"], colors["right_no_opto"]),
                  (f"{trial_side}_trial_left_opto", f"{trial_side}_trial_right_opto", colors["left_opto"], colors["right_opto"])]        

    conditions = [([("left_trial_left_no_opto", "left_trial_right_no_opto"),("right_trial_left_no_opto", "right_trial_right_no_opto")], colors["left_no_opto"], colors["right_no_opto"]),
                  (f"{trial_side}_trial_left_opto", f"{trial_side}_trial_right_opto", colors["left_opto"], colors["right_opto"])]        
    
    # # Organize conditions by opto/non-opto first, then left/right
    # conditions = [(f"{trial_side}_trial_left_no_opto", f"{trial_side}_trial_right_no_opto", 'green', 'black'),
    #               (f"{trial_side}_trial_left_opto", f"{trial_side}_trial_right_opto", 'lightgreen', 'darkgray')]    
    
    trial_offset = 0  # Initialize the trial offset
    
    for (left_cond, right_cond, left_color, right_color) in conditions:
        for trial_idx in range(len(lick_data[f"{left_cond}_starts"])):
            
            # Get left licks
            left_starts = lick_data[f"{left_cond}_starts"][trial_idx]
            left_stops = lick_data[f"{left_cond}_stops"][trial_idx]
            
            # Get right licks
            right_starts = lick_data[f"{right_cond}_starts"][trial_idx]
            right_stops = lick_data[f"{right_cond}_stops"][trial_idx]
            
            # Get valve timings
            valve_starts = lick_data[f"{left_cond}_valve_starts"][trial_idx]  # Assume same valve times for both
            valve_stops = lick_data[f"{left_cond}_valve_stops"][trial_idx]
            
            valid_trials = False  # Flag to check if there are valid trials
            
            # Plot left licks
            if len(left_starts) > 0 and len(left_stops) > 0:
                for start, stop in zip(left_starts, left_stops):
                    if not np.isnan(start) and not np.isnan(stop):
                        ax.plot([start, stop], [trial_offset] * 2, color=left_color, alpha=0.7, lw=0.5)
                        valid_trials = True
            
            # Plot right licks
            if len(right_starts) > 0 and len(right_stops) > 0:
                for start, stop in zip(right_starts, right_stops):
                    if not np.isnan(start) and not np.isnan(stop):
                        ax.plot([start, stop], [trial_offset] * 2, color=right_color, alpha=0.9, lw=0.5)
                        valid_trials = True
            
            # Plot valve times
            if valid_trials and not np.isnan(valve_starts) and not np.isnan(valve_stops):
                # for v_start, v_stop in zip(valve_starts, valve_stops):
                # if not np.isnan(v_start) and not np.isnan(v_stop):
                ax.plot([valve_starts, valve_stops], [trial_offset, trial_offset], color=colors["valve"], lw=2, alpha=0.3)
            
            # Increment trial offset only if there were valid trials
            if valid_trials:
                trial_offset += 1
    
    # # labels = [f"{trial_side.capitalize()} Trial - No Opto", f"{trial_side.capitalize()} Trial - Opto"]
    # labels = ["Control", "Opto"]    
    # 
    # Create custom legend
    # legend_elements = [Line2D([0], [0], color='green', lw=2, label='Left Lick ' + labels[0]),
    #                    Line2D([0], [0], color='black', lw=2, label='Right Lick ' + labels[0]),
    #                    Line2D([0], [0], color='lightgreen', lw=2, label='Left Lick ' + labels[1]),
    #                    Line2D([0], [0], color='darkgray', lw=2, label='Right Lick ' + labels[1]),
    #                    Line2D([0], [0], color='blue', lw=2, label="Valve Open/Close")]
    
    # # Update legend colors
    # legend_elements = [Line2D([0], [0], color='#1f77b4', lw=2, label='Left Lick Control'),
    #                    Line2D([0], [0], color='#ff7f0e', lw=2, label='Right Lick Control'),
    #                    Line2D([0], [0], color='#17becf', lw=2, label='Left Lick Opto'),
    #                    Line2D([0], [0], color='#d62728', lw=2, label='Right Lick Opto'),
    #                    Line2D([0], [0], color='#9467bd', lw=2, label="Valve Open/Close")]    
    
    # labels = [f"{trial_side.capitalize()} Trial - No Opto", f"{trial_side.capitalize()} Trial - Opto"]
    labels = ["Control", "Opto"]    
    
    # Create custom legend
    legend_elements = [Line2D([0], [0], color=colors["left_no_opto"], lw=2, label='Left Lick ' + labels[0]),
                       Line2D([0], [0], color=colors["right_no_opto"], lw=2, label='Right Lick ' + labels[0]),
                       Line2D([0], [0], color=colors["left_opto"], lw=2, label='Left Lick ' + labels[1]),
                       Line2D([0], [0], color=colors["right_opto"], lw=2, label='Right Lick ' + labels[1]),
                       Line2D([0], [0], color=colors["valve"], lw=2, label="Valve Open/Close")]    
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trials")
    ax.set_title(title)
    ax.legend(handles=legend_elements)             
    plt.show()     

def licks_to_binary_times(lick_starts_list, lick_stops_list, time_resolution_ms=1):
    """
    Converts lists of lick start/stop times into binary time arrays.
    
    Args:
        lick_starts_list: list of lists of float64 start times
        lick_stops_list:  list of lists of float64 stop times
        time_resolution_ms: time resolution in milliseconds
    
    Returns:
        binary_lick_matrix: list of np.array (each is binary per-ms lick timeline)
        time_vector: 1D array of timepoints (shared across trials)
    """
    
    # Determine the global max time across all trials to fix the length
    # max_time = 0.0
    # for starts, stops in zip(lick_starts_list, lick_stops_list):
    #     if starts and stops:
    #         trial_max = max(max(starts), max(stops))
    #         max_time = max(max_time, trial_max)

    # max_time = 10.0 # 10 seconds to match across sides/sessions
    # # Create a shared time vector (e.g., 0 ms to max_time in ms steps)
    # time_vector = np.arange(0, max_time + time_resolution_ms / 1000, time_resolution_ms / 1000)
    n_timepoints = len(time_vector)

    binary_lick_matrix = []

    

    for starts, stops in zip(lick_starts_list, lick_stops_list):
        binary_trace = np.zeros(n_timepoints, dtype=int)
        for start, stop in zip(starts, stops):
            # Find indices in the time_vector that fall within lick window
            mask = (time_vector >= start) & (time_vector <= stop)
            binary_trace[mask] = 1
        binary_lick_matrix.append(binary_trace)

    if not binary_lick_matrix:
        binary_trace = np.zeros(n_timepoints, dtype=int)
        binary_lick_matrix.append(binary_trace)

    return binary_lick_matrix

def average_lick_trace(binary_lick_matrix):
    """
    Computes the average lick probability across trials.
    
    Args:
        binary_lick_matrix: list of 1D numpy arrays (binary per-trial lick traces)
    
    Returns:
        avg_lick_trace: 1D array of mean lick probability at each timepoint
    """
    if not binary_lick_matrix:
        return np.array([])  # Handle empty input safely
    
    # Stack into 2D array: shape (n_trials, n_timepoints)
    lick_array = np.stack(binary_lick_matrix)
    
    # Compute average across trials (axis=0)
    avg_lick_trace = np.nanmean(lick_array, axis=0)
    
    return avg_lick_trace

def session_avg_licks_interpolated(processed_licks, side):  

    avg_lick_times ={}    

    side_trials_df = processed_licks[processed_licks['trial_side'] == side]
    
    # split trials according to rewarded and punished
    rewarded_trials_df = side_trials_df[side_trials_df['rewarded'] == True]
    punished_trials_df = side_trials_df[side_trials_df['rewarded'] == False]
    
    # non_opto_rewarded
    non_opto_rewarded = rewarded_trials_df[rewarded_trials_df['is_opto'] == False]            

    # Extract the column with lists of lick start times
    lick_starts_list = non_opto_rewarded[f'licks_{side}_start'].tolist()
    lick_stops_list = non_opto_rewarded[f'licks_{side}_stop'].tolist()
    
    binary_lick_matrix = licks_to_binary_times(lick_starts_list, lick_stops_list)
    non_opto_rewarded_avg_lick_trace = average_lick_trace(binary_lick_matrix)
        
    # opto_rewarded
    opto_rewarded = rewarded_trials_df[rewarded_trials_df['is_opto'] == True]    
    
    # Extract the column with lists of lick start times
    lick_starts_list = opto_rewarded[f'licks_{side}_start'].tolist()
    lick_stops_list = opto_rewarded[f'licks_{side}_stop'].tolist() 
    
    binary_lick_matrix = licks_to_binary_times(lick_starts_list, lick_stops_list)
    opto_rewarded_avg_lick_trace = average_lick_trace(binary_lick_matrix)    

    avg_lick_times = {
        "control_rewarded": non_opto_rewarded_avg_lick_trace,
        "opto_rewarded": opto_rewarded_avg_lick_trace,
    }  
    
    # avg_lick_times = {
    #     "control_rewarded": {
    #         "starts": non_opto_rewarded_avg_lick_starts,
    #         "stops": non_opto_rewarded_avg_lick_stops
    #     },
    #     "opto_rewarded": {
    #         "starts": opto_rewarded_avg_lick_starts,
    #         "stops": opto_rewarded_avg_lick_stops
    #     },
    # }            

    return avg_lick_times   

def sort_type_1(processed_licks):  

    left_side_trials_df = processed_licks[processed_licks['trial_side'] == 'left']
    
    # Create a temporary column 'min_lick_start' which holds the minimum lick start time from the arrays
    left_side_trials_df['min_lick_left_start'] = left_side_trials_df['licks_left_start'].apply(lambda x: min(x) if isinstance(x, list) else x)
    left_side_trials_df['min_lick_right_start'] = left_side_trials_df['licks_right_start'].apply(lambda x: min(x) if isinstance(x, list) else x)
    
    # Now sort based on the 'min_lick_start' column, but keep the original arrays intact
    rewarded_trials_df = left_side_trials_df[left_side_trials_df['rewarded'] == True]
    nonrewarded_trials_df = left_side_trials_df[left_side_trials_df['rewarded'] == False]
    
    # rewarded_trials_df = left_side_trials_df[left_side_trials_df['rewarded'] == True]
    # nonrewarded_trials_df = left_side_trials_df[left_side_trials_df['rewarded'] == False]
    
    rewarded_trials_sorted_df = rewarded_trials_df.sort_values(by=['is_opto', 'min_lick_left_start'], ascending=[True, True])
    nonrewarded_trials_sorted_df = nonrewarded_trials_df.sort_values(by=['is_opto', 'min_lick_right_start'], ascending=[True, True])
    
    # Combine back into one DataFrame
    sorted_left_side_trials_df = pd.concat([rewarded_trials_sorted_df, nonrewarded_trials_sorted_df])

    return sorted_left_side_trials_df

def sort_type_2(processed_licks):     

    right_side_trials_df = processed_licks[processed_licks['trial_side'] == 'right']
    
    # Create a temporary column 'min_lick_start' which holds the minimum lick start time from the arrays
    right_side_trials_df['min_lick_left_start'] = right_side_trials_df['licks_left_start'].apply(lambda x: min(x) if isinstance(x, list) else x)
    right_side_trials_df['min_lick_right_start'] = right_side_trials_df['licks_right_start'].apply(lambda x: min(x) if isinstance(x, list) else x)
    
    # Now sort based on the 'min_lick_start' column, but keep the original arrays intact
    rewarded_trials_df = right_side_trials_df[right_side_trials_df['rewarded'] == True]
    nonrewarded_trials_df = right_side_trials_df[right_side_trials_df['rewarded'] == False]
    
    # rewarded_trials_df = right_side_trials_df[right_side_trials_df['rewarded'] == True]
    # nonrewarded_trials_df = right_side_trials_df[right_side_trials_df['rewarded'] == False]
    
    rewarded_trials_sorted_df = rewarded_trials_df.sort_values(by=['is_opto', 'min_lick_right_start'], ascending=[True, True])
    nonrewarded_trials_sorted_df = nonrewarded_trials_df.sort_values(by=['is_opto', 'min_lick_left_start'], ascending=[True, True])
    
    # Combine back into one DataFrame
    sorted_right_side_trials_df = pd.concat([rewarded_trials_sorted_df, nonrewarded_trials_sorted_df])

    return sorted_right_side_trials_df

def sort_type_3(processed_licks):     
    
    left_side_trials_df = processed_licks[processed_licks['trial_side'] == 'left']
    sorted_left_side_trials_df = sort_type_1(processed_licks)
    
    right_side_trials_df = processed_licks[processed_licks['trial_side'] == 'right']
    sorted_right_side_trials_df = sort_type_2(processed_licks)   
    
    # Combine back into one DataFrame
    sorted_trials_df = pd.concat([sorted_left_side_trials_df, sorted_right_side_trials_df])
    
    return sorted_trials_df

def session_avg_licks(processed_licks, side):  

    avg_lick_times ={}    

    side_trials_df = processed_licks[processed_licks['trial_side'] == side]
    
    # split trials according to rewarded and punished
    rewarded_trials_df = side_trials_df[side_trials_df['rewarded'] == True]
    punished_trials_df = side_trials_df[side_trials_df['rewarded'] == False]
    
    # non_opto_rewarded
    non_opto_rewarded = rewarded_trials_df[rewarded_trials_df['is_opto'] == False]            

    # Extract the column with lists of lick start times
    lick_starts_list = non_opto_rewarded[f'licks_{side}_start'].tolist()
    lick_stops_list = non_opto_rewarded[f'licks_{side}_stop'].tolist()
    
    # Pad lists to the same length with NaNs
    padded_starts = pad_list(lick_starts_list)        
    # Compute element-wise mean, ignoring NaNs
    non_opto_rewarded_avg_lick_starts = np.nanmean(padded_starts, axis=0)
    
    # Pad lists to the same length with NaNs
    padded_stops = pad_list(lick_stops_list)
    # Compute element-wise mean, ignoring NaNs
    non_opto_rewarded_avg_lick_stops = np.nanmean(padded_stops, axis=0)            
    
    if print_debug:
        print(f"Avg Lick Start Times: {non_opto_rewarded_avg_lick_starts}")            
        print(f"Avg Lick Stop Times: {non_opto_rewarded_avg_lick_stops}")     
    
    # opto_rewarded
    opto_rewarded = rewarded_trials_df[rewarded_trials_df['is_opto'] == True]
    
    # Extract the column with lists of lick start times
    lick_starts_list = opto_rewarded[f'licks_{side}_start'].tolist()
    lick_stops_list = opto_rewarded[f'licks_{side}_stop'].tolist()
    
    # Pad lists to the same length with NaNs
    padded_starts = pad_list(lick_starts_list)              
    # Compute element-wise mean, ignoring NaNs
    opto_rewarded_avg_lick_starts = np.nanmean(padded_starts, axis=0)
    
    # Pad lists to the same length with NaNs
    padded_stops = pad_list(lick_stops_list)            
    # Compute element-wise mean, ignoring NaNs
    opto_rewarded_avg_lick_stops = np.nanmean(padded_stops, axis=0)            
    
    if print_debug:
        print(f"Avg Lick Start Times: {opto_rewarded_avg_lick_starts}")            
        print(f"Avg Lick Stop Times: {opto_rewarded_avg_lick_stops}")             
    
    # Compute element-wise mean
    # avg_lick_starts = np.nanmean(lick_starts_array, axis=0)  # Use nanmean to handle unequal lengths
    # avg_lick_stops = np.nanmean(lick_stops_array, axis=0)
    
    # print(f"Avg Lick Starts: {avg_lick_starts}")
    # print(f"Avg Lick Stops: {avg_lick_stops}")            
    

    # # non_opto_punished
    # non_opto_punished = punished_trials_df[punished_trials_df['is_opto'] == False]    
    
    # # Extract the column with lists of lick start times
    # lick_starts_list = non_opto_punished[f'licks_{side}_start'].tolist()
    # lick_stops_list = non_opto_punished[f'licks_{side}_stop'].tolist()
    
    # # Pad lists to the same length with NaNs
    # padded_starts = pad_list(lick_starts_list)             
    # # Compute element-wise mean, ignoring NaNs
    # non_opto_punished_avg_lick_starts = np.nanmean(padded_starts, axis=0)
    
    # # Pad lists to the same length with NaNs
    # padded_stops = pad_list(lick_stops_list)          
    # # Compute element-wise mean, ignoring NaNs
    # non_opto_punished_avg_lick_stops = np.nanmean(padded_stops, axis=0)            
    
    # print(f"Avg Lick Start Times: {non_opto_punished_avg_lick_starts}")            
    # print(f"Avg Lick Stop Times: {non_opto_punished_avg_lick_stops}")     
    
    # # opto_punished
    # opto_punished = punished_trials_df[punished_trials_df['is_opto'] == True]
    
    # # Extract the column with lists of lick start times
    # lick_starts_list = opto_punished[f'licks_{side}_start'].tolist()
    # lick_stops_list = opto_punished[f'licks_{side}_stop'].tolist()
    
    # # Pad lists to the same length with NaNs
    # padded_starts = pad_list(lick_starts_list)         
    # # Compute element-wise mean, ignoring NaNs
    # opto_punished_avg_lick_starts = np.nanmean(padded_starts, axis=0)
    
    # # Pad lists to the same length with NaNs
    # padded_stops = pad_list(lick_stops_list)         
    # # Compute element-wise mean, ignoring NaNs
    # opto_punished_avg_lick_stops = np.nanmean(padded_stops, axis=0)            
    
    # print(f"Avg Lick Start Times: {opto_punished_avg_lick_starts}")            
    # print(f"Avg Lick Stop Times: {opto_punished_avg_lick_stops}")             
          
    avg_lick_times = {
        "control_rewarded": {
            "starts": non_opto_rewarded_avg_lick_starts,
            "stops": non_opto_rewarded_avg_lick_stops
        },
        "opto_rewarded": {
            "starts": opto_rewarded_avg_lick_starts,
            "stops": opto_rewarded_avg_lick_stops
        },
        # "non_opto_punished": {
        #     "starts": non_opto_punished_avg_lick_starts,
        #     "stops": non_opto_punished_avg_lick_stops
        # },
        # "opto_punished": {
        #     "starts": opto_punished_avg_lick_starts,
        #     "stops": opto_punished_avg_lick_stops
        # }
    }            

    return avg_lick_times 

def grand_avg_licks(lick_list):
    """
    Compute the grand average lick start and stop times for control and opto conditions.
    
    :param lick_list: List of dictionaries containing lick start and stop times.
    :return: Dictionary with grand averaged lick times for control and opto.
    """
    grand_avg_lick_times = {
        "control_rewarded": {
            "starts": [],
            "stops": []
        },
        "opto_rewarded": {
            "starts": [],
            "stops": []
        }
    }
    
    # Collect all start and stop times for each condition
    control_starts = []
    control_stops = []
    opto_starts = []
    opto_stops = []
    
    for lick in lick_list:
        control_starts.append(lick["control_rewarded"]["starts"].tolist())
        control_stops.append(lick["control_rewarded"]["stops"].tolist())
        opto_starts.append(lick["opto_rewarded"]["starts"].tolist())
        opto_stops.append(lick["opto_rewarded"]["stops"].tolist())
    
    
    
    control_starts = pad_list(control_starts)
    control_stops = pad_list(control_stops)
    opto_starts = pad_list(opto_starts)
    opto_stops = pad_list(opto_stops)
    
    # Compute grand average using numpy
    grand_avg_lick_times["control_rewarded"]["starts"] = np.nanmean(control_starts, axis=0)
    grand_avg_lick_times["control_rewarded"]["stops"] = np.nanmean(control_stops, axis=0)
    grand_avg_lick_times["opto_rewarded"]["starts"] = np.nanmean(opto_starts, axis=0)
    grand_avg_lick_times["opto_rewarded"]["stops"] = np.nanmean(opto_stops, axis=0)      
      
    return grand_avg_lick_times 

def pad_to_length(arr, length):
    padded = np.pad(arr, (0, length - len(arr)), constant_values=np.nan)
    return np.asarray(padded, dtype=np.float64)


def grand_avg_licks_matrix(lick_list):
    """
    Compute the grand average lick start and stop times for control and opto conditions.
    
    :param lick_list: List of dictionaries containing lick start and stop times.
    :return: Dictionary with grand averaged lick times for control and opto.
    """
    if not lick_list:
        return np.array([])  # Handle empty input safely    
    
    grand_avg_lick_times = {
        "control_rewarded": {
        },
        "opto_rewarded": {
        }
    }
    
    # Collect all start and stop times for each condition
    # control_starts = []
    # control_stops = []
    # opto_starts = []
    # opto_stops = []
    
    control = []
    opto = []
    
    for lick in lick_list:
        control.append(lick["control_rewarded"].tolist())
        opto.append(lick["opto_rewarded"].tolist())       
                 
    grand_avg_lick_times["control_rewarded"] = np.nanmean(control, axis=0)
    grand_avg_lick_times["opto_rewarded"] = np.nanmean(opto, axis=0)   
      
    return grand_avg_lick_times 

def check_if_lists(lst):
    return all(isinstance(item, list) for item in lst)

def pad_list(licks_list):        
    padded_list = []
    try:            
        # if licks_list and len(licks_list) > 1:
        if licks_list:
            # Ensure all elements are float64 lists
            valid_lists = [lst for lst in licks_list if isinstance(lst, list) and all(isinstance(x, float) for x in lst)]
            
            if not valid_lists:  # Handle case where there are no valid lists
                # return np.array([])  # Return an empty array to prevent errors                
                return padded_list  # Return an empty array to prevent errors   
            
            # Find the max length
            max_len = max(len(lst) for lst in valid_lists)
        
            # Pad lists to max_len with NaNs
            padded_list = [lst + [np.nan] * (max_len - len(lst)) for lst in valid_lists]
            
            # valid_arrays = [arr for arr in licks_list if isinstance(arr, np.ndarray) and arr.dtype == np.float64]
            
            # if valid_arrays:  # Ensure there's at least one valid array
            #     max_len = max(arr.shape[0] for arr in valid_arrays)  # Get max length along axis 0
            # else:
            #     max_len = 0  # Handle case where no valid arrays exist                
            # # Pad each array with NaNs to max_len
            # padded_arrays = [np.pad(arr, (0, max_len - arr.shape[0]), constant_values=np.nan) for arr in valid_arrays]
            
            # # Convert to a NumPy array
            # padded_array = np.array(padded_arrays, dtype=np.float64)                
            
            
            # max_len = max(len(lst) for lst in licks_list)
            # max_len = max((len(lst) for lst in licks_list if isinstance(lst, list)), default=0)
            # valid_lists = [lst for lst in licks_list if isinstance(lst, list)]
            
            
            # if valid_arrays:  # Ensure there's at least one valid list
            #     max_len = max(len(lst) for lst in valid_lists)
            # else:
            #     max_len = 0  # Handle the case where no valid lists exist                
            # string combo, cheddar, up up down down left left right right a b a b start
            
            
            # padded_list = np.array([lst + [np.nan] * (max_len - len(lst)) if isinstance(lst, list) else lst.tolist() + [np.nan] * (max_len - len(lst)) for lst in licks_list])

        else:
            padded_list = licks_list
            # padded_list = np.array(np.nan, dtype=np.float64)
            # # make sure lick vars are lists               
            # padded_list = padded_list if padded_list else [[np.nan]]  # Ensures it's at least a 1x1 list
            # padded_list = np.array(padded_list, dtype=np.float64)
            # if padded_list.ndim == 0:
            #     padded_list = np.array([padded_list], dtype=np.float64)                 
            # padded_list = [np.nan]
        # if not isinstance(padded_list, list):
        #     padded_list.tolist() 
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return padded_list

max_time = 5.0 # 10 seconds to match across sides/sessions
time_resolution_ms = 1
# Create a shared time vector (e.g., 0 ms to max_time in ms steps)
time_vector = np.arange(0, max_time + time_resolution_ms / 1000, time_resolution_ms / 1000)

x_lim_left = 0.0
x_lim_right = max_time

def plot_licking_opto_avg(M, config, subjectIdx, show_plot=1):
    # figure meta
    rowspan, colspan = 2, 4
    fig_size = get_figsize_from_pdf_spec(rowspan, colspan, config['pdf_spec']['pdf_pg_cover'])    
    fig, ax = plt.subplots(figsize=fig_size)   
    
  
    
  
    max_sessions = 20
    subject = M['subject']
    dates = M['dates']
    session_id = np.arange(len(dates)) + 1
    jitter_flag = M['jitter_flag']
    raw_data = M['raw']
    outcomes = M['outcomes']
    outcomes_time = M['outcomes_time']
    #categories = M['isi_post_emp']
        

    
    subject = remove_substrings(subject, ['_opto', '_reg'])
    subject = flip_underscore_parts(subject)
    subject = lowercase_h(subject)
        
    trial_type = M['trial_type']
    opto_flag = M['opto_trial']
    
    row = 4 
    col = 5

    # alignments = ['1st flash' , '3th flash' , '4th flash' , 'choice window' , 'outcome']
    alignments = ['Choice Window' , 'Outcome']
    row_names = ['rewarded short' , 'rewarded long' , 'punished short' , 'punished long']
    n_bins = 500        

    # Define the relative height ratios
    # height_ratios = [1, 0.125, 0.125, 0.125]  # The 4th row will have twice the height of the others
    height_ratios = [1, 0.75, 0.75]  # The 4th row will have twice the height of the others
    
    # fig, axs = plt.subplots(nrows=4, ncols=len(alignments), figsize=(40, 30), 
    #                 gridspec_kw={'height_ratios': height_ratios})
    fig, axs = plt.subplots(nrows=3, ncols=len(alignments), figsize=(40, 30), 
                    gridspec_kw={'height_ratios': height_ratios})    
    
    #fig2, axs1 = plt.subplots(nrows=4, ncols=len(alignments), figsize=(40, 30))
    pdf_streams = []
    pdf_paths = []
    # numTrials = raw_data[i]['nTrials']
    # outcome = outcomes[i]
    # outcome_time = outcomes_time[i]
    # session_date = dates[i]
    


    # pooled avg
    all_processed_licks = pd.DataFrame()
    for i in range(len(dates)):
        # print(i)        
        numTrials = raw_data[i]['nTrials']
        # print(numTrials)
        processed_licks = pd.DataFrame()
        
        # get session processed licks
        processed_licks = process_session_licks_with_valve_times_df(M, i)
        
        # Combine back into one DataFrame
        all_processed_licks = pd.concat([all_processed_licks, processed_licks])
           

    debug = 0


    # remove trials without licks
    all_processed_licks = all_processed_licks[all_processed_licks['no_lick'] == 0]
    # remove naive trials
    all_processed_licks = all_processed_licks[all_processed_licks['is_naive'] == 0]

    row = 0
        
    if debug:
        plot_lick_traces_df(all_processed_licks, 'all', 'All Trials - Ordered by Trial')
    else:
        plot_lick_traces_df(all_processed_licks, 'all', 'All Trials - Ordered by Trial', axs[row, 0])
    
    sorted_trials_df = sort_type_3(all_processed_licks)
    print(f"Plotting all session licks...{subject}")
    plot_lick_traces_df(sorted_trials_df, 'all', 'All Trials - Ordered by Earliest Lick', axs[row, 1])    
                                                                                    
    row = row+1
    
    # # left side trial ordered
    # left_side_trials_df = processed_licks[processed_licks['trial_side'] == 'left']
    # plot_lick_traces_df(left_side_trials_df, 'left', 'Left Trials - Short ISI - Ordered by Trial', axs[row, 0])
    
    # # right side trial ordered
    # right_side_trials_df = processed_licks[processed_licks['trial_side'] == 'right']
    # plot_lick_traces_df(right_side_trials_df, 'left', 'Right Trials - Long ISI - Ordered by Trial', axs[row, 1])        
    
    # opto_df = processed_licks[(processed_licks['is_opto'] ==  True)]
    # opto_left = opto_df[(opto_df['trial_side'] ==  'left')]
    
    # opto_df = all_processed_licks[(all_processed_licks['is_opto'] ==  True)]
    # opto_left = opto_df[(opto_df['trial_side'] ==  'left')]    
    
    print(f"Plotting pooled average licks...{subject}")
    left_side_avg_licks = session_avg_licks_interpolated(all_processed_licks, 'left')    
    # sessions_left_side_avg_licks.append(left_side_avg_licks)
    plot_avg_lick_traces_from_matrix(left_side_avg_licks, 'left', 'Left Trials - Short ISI - Session Lick Pooled Average', axs[row, 0])       
    
    # left_side_avg_licks = session_avg_licks(all_processed_licks, 'left')
    # plot_avg_lick_traces(left_side_avg_licks, 'left', 'Left Trials - Short ISI - Session Lick Pooled Average', axs[row, 0])        
    
    right_side_avg_licks = session_avg_licks_interpolated(all_processed_licks, 'right')
    # sessions_right_side_avg_licks.append(right_side_avg_licks)
    plot_avg_lick_traces_from_matrix(right_side_avg_licks, 'right', 'Right Trials - Long ISI - Session Lick Pooled Average', axs[row, 1])            
        
    # right_side_avg_licks = session_avg_licks(all_processed_licks, 'right')
    # plot_avg_lick_traces(right_side_avg_licks, 'right', 'Right Trials - Long ISI - Session Lick Pooled Average', axs[row, 1])     
    
    row = row+1
    
    # grand avg
    sessions_left_side_avg_licks = []
    sessions_right_side_avg_licks = []
    for i in range(len(dates)):
        # print(i)        
        numTrials = raw_data[i]['nTrials']
        # print(numTrials)
        processed_licks = pd.DataFrame()
        
        # get session processed licks
        processed_licks = process_session_licks_with_valve_times_df(M, i)                
        # remove trials without licks
        processed_licks = processed_licks[processed_licks['no_lick'] == 0]
        # remove naive trials
        processed_licks = processed_licks[processed_licks['is_naive'] == 0]        
        
        
        
        # left_side_avg_licks = session_avg_licks(processed_licks, 'left')
        
        left_side_avg_licks = session_avg_licks_interpolated(processed_licks, 'left')
        sessions_left_side_avg_licks.append(left_side_avg_licks)
        # plot_avg_lick_traces(left_side_avg_licks, 'left', 'Left Trials - Short ISI - Session Lick Average', axs[row, 0])        
                
        # if i == 2:
            # print('')
        # right_side_avg_licks = session_avg_licks(processed_licks, 'right')
        
        right_side_avg_licks = session_avg_licks_interpolated(processed_licks, 'right')
        sessions_right_side_avg_licks.append(right_side_avg_licks)
        # plot_avg_lick_traces(right_side_avg_licks, 'right', 'Right Trials - Long ISI - Session Lick Average', axs[row, 1])  
     
    print(f"Plotting grand average licks...{subject}")
    # grand_left_side_avg_licks = grand_avg_licks(sessions_left_side_avg_licks)
    grand_left_side_avg_licks = grand_avg_licks_matrix(sessions_left_side_avg_licks)
    # plot_avg_lick_traces(grand_left_side_avg_licks, 'left', 'Left Trials - Short ISI - Session Lick Grand Average', axs[row, 0])        
    # plot_avg_lick_traces(grand_left_side_avg_licks, 'left', 'Left Trials - Short ISI - Session Lick Grand Average', axs[row, 0])            
    plot_avg_lick_traces_from_matrix(grand_left_side_avg_licks, 'left', 'Left Trials - Short ISI - Session Lick Grand Average', axs[row, 0])
    
    # grand_right_side_avg_licks = grand_avg_licks(sessions_right_side_avg_licks)
    grand_right_side_avg_licks = grand_avg_licks_matrix(sessions_right_side_avg_licks)
    # plot_avg_lick_traces(grand_right_side_avg_licks, 'right', 'Right Trials - Long ISI - Session Lick Grand Average', axs[row, 1])
    plot_avg_lick_traces_from_matrix(grand_right_side_avg_licks, 'right', 'Right Trials - Long ISI - Session Lick Grand Average', axs[row, 1])
    
    # # left side first lick ordered
    # sorted_left_side_trials_df = sort_type_1(processed_licks)
    # plot_lick_traces_df(sorted_left_side_trials_df, 'right', 'Left Trials - Short ISI - Ordered by Earliest Lick', axs[row, 0])
    
    # # right side first lick ordered
    # sorted_right_side_trials_df = sort_type_2(processed_licks)
    # plot_lick_traces_df(sorted_right_side_trials_df, 'right', 'Right Trials - Long ISI - Ordered by Earliest Lick', axs[row, 1])
    
    row = row+1
    
    # left_side_avg_licks = session_avg_licks(processed_licks, 'left')
    # plot_avg_lick_traces(left_side_avg_licks, 'left', 'Left Trials - Short ISI - Session Lick Average', axs[row, 0])        
    
    # right_side_avg_licks = session_avg_licks(processed_licks, 'right')
    # plot_avg_lick_traces(right_side_avg_licks, 'right', 'Right Trials - Long ISI - Session Lick Average', axs[row, 1])        
    

    # output_dir_onedrive, 
    # output_dir_local

    # output_pdf_dir =  output_dir_onedrive + subject + '/'
    # output_pdf_pages_dir = output_dir_local + subject + '/lick_traces/'
    # os.makedirs(output_pdf_dir, exist_ok = True)
    # os.makedirs(output_pdf_pages_dir, exist_ok = True)
    # output_pdf_filename = output_pdf_pages_dir + subject + '_lick_traces_all_sessions_pooled_report_' + dates[0] + '_' + dates[-1]
    # pdf_paths.append(output_pdf_filename + '.pdf')
    # save_image(output_pdf_filename)        
    # plt.close(fig)
        
    
    # output = PdfWriter()
    # pdf_files = []
    # for pdf_path in pdf_paths:            
    #     f = open(pdf_path, "rb")
    #     pdf_streams.append(PdfReader(f))
    #     pdf_files.append(f)

    # for pdf_file_stream in pdf_streams:
    #     output.add_page(pdf_file_stream.pages[0])

    # for pdf_file in pdf_files:
    #     pdf_file.close()


    # if upload:
    #     outputStream = open(r'' + output_pdf_dir + '/lick/' + subject + '_lick_traces_all_sessions' + '.pdf', "wb")
    #     output.write(outputStream)
    #     outputStream.close()
        
    if show_plot:
        plt.show()
        
    subject = config['list_config'][subjectIdx]['subject_name']
    output_dir = os.path.join(config['paths']['figure_dir_local'] + subject)
    figure_id = f"{subject}_licking_avg"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)    
   
    # return {
    #     'figure_id': figure_id,
    #     'path': out_path,
    #     'caption': f"Decision time plot for {subject}",
    #     'subject': subject,
    #     'tags': ['performance', 'bias'],
    #     "layout": {
    #       "page": 0,
    #       "page_key": "pdf_pg_cover", 
    #       "row": 0,
    #       "col": 6,
    #       "rowspan": rowspan,
    #       "colspan": colspan,
    #     }        
    # }        
    return out_path   



