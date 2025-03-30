#!/usr/bin/env python
# coding: utf-8

# In[1]:



from scipy.stats import sem
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfWriter, PdfReader
from scipy.interpolate import interp1d
from datetime import date
from statistics import mean 
import math
from matplotlib.lines import Line2D

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
        
        
    # # Organize conditions by opto/non-opto first, then left/right
    # conditions = [(f"{trial_side}_trial_left_no_opto", f"{trial_side}_trial_right_no_opto"),
    #               (f"{trial_side}_trial_left_opto", f"{trial_side}_trial_right_opto")]
    # labels = [f"{trial_side.capitalize()} Trial - No Opto", f"{trial_side.capitalize()} Trial - Opto"]
    
    # trial_offset = 0  # Initialize the trial offset
    
    # for (left_cond, right_cond) in conditions:
        
            
    return processed_licks, processed_licks_sorted

# def sort_licks_left_right_opto_non(processed_licks):
#     # Sorting each condition by the first lick time
#     processed_licks_sorted = {}
#     for condition in conditions:
#         starts = processed_licks[f"{condition}_starts"]
#         stops = processed_licks[f"{condition}_stops"]
        
#         # Convert lists to arrays for sorting
#         starts_array = np.array(starts, dtype=object)
#         stops_array = np.array(stops, dtype=object)
    
#         # Sort by first valid element, placing NaNs at the end
#         valid_indices = [i for i in range(len(starts_array)) if not np.isnan(starts_array[i][0])]
#         nan_indices = [i for i in range(len(starts_array)) if np.isnan(starts_array[i][0])]
    
#         sorted_valid_indices = sorted(valid_indices, key=lambda i: starts_array[i][0])
#         sorted_indices = sorted_valid_indices + nan_indices  # NaNs go at the end
    
#         processed_licks_sorted[f"{condition}_starts"] = [starts_array[i] for i in sorted_indices]
#         processed_licks_sorted[f"{condition}_stops"] = [stops_array[i] for i in sorted_indices]   
#     return processed_licks_sorted

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
            # label category
            # if not np.isnan(valve_times[0]):
            #     label = 'Rewarded ' + side.capitalize()                
            # else:
            #     label = 'Punished ' + side.capitalize()
            # processed_licks[f"{key}_category"].append(label)
    
    # # Sorting each condition by the first lick time
    # processed_licks_sorted = {}
    # for condition in conditions:
    #     starts = processed_licks[f"{condition}_starts"]
    #     stops = processed_licks[f"{condition}_stops"]
    #     valve_starts = processed_licks[f"{condition}_valve_starts"]
    #     valve_stops = processed_licks[f"{condition}_valve_stops"]
        
    #     # Convert lists to arrays for sorting
    #     starts_array = np.array(starts, dtype=object)
    #     stops_array = np.array(stops, dtype=object)
    #     valve_starts_array = np.array(valve_starts, dtype=object)
    #     valve_stops_array = np.array(valve_stops, dtype=object)        
    
    #     # Sort by first valid element, placing NaNs at the end
    #     valid_indices = [i for i in range(len(starts_array)) if not np.isnan(starts_array[i][0])]
    #     nan_indices = [i for i in range(len(starts_array)) if np.isnan(starts_array[i][0])]
    
    #     sorted_valid_indices = sorted(valid_indices, key=lambda i: starts_array[i][0])
    #     sorted_indices = sorted_valid_indices + nan_indices  # NaNs go at the end
    
    #     processed_licks_sorted[f"{condition}_starts"] = [starts_array[i] for i in sorted_indices]
    #     processed_licks_sorted[f"{condition}_stops"] = [stops_array[i] for i in sorted_indices]        
    #     processed_licks_sorted[f"{condition}_valve_starts"] = [valve_starts_array[i] for i in sorted_indices]
    #     processed_licks_sorted[f"{condition}_valve_stops"] = [valve_stops_array[i] for i in sorted_indices]      
     
    # return processed_licks, processed_licks_sorted
    return processed_licks

# def sort_licks_left_right_opto_non(processed_licks):
#     conditions = ["left_trial_left_opto", "left_trial_left_no_opto", "left_trial_right_opto", "left_trial_right_no_opto",
#                   "right_trial_left_opto", "right_trial_left_no_opto", "right_trial_right_opto", "right_trial_right_no_opto"]    
    
#     # Sorting each condition by the first lick time
#     processed_licks_sorted = {}
#     for condition in conditions:
#         starts = processed_licks[f"{condition}_starts"]
#         stops = processed_licks[f"{condition}_stops"]
#         valve_starts = processed_licks[f"{condition}_valve_starts"]
#         valve_stops = processed_licks[f"{condition}_valve_stops"]
        
#         # Convert lists to arrays for sorting
#         starts_array = np.array(starts, dtype=object)
#         stops_array = np.array(stops, dtype=object)
#         valve_starts_array = np.array(valve_starts, dtype=object)
#         valve_stops_array = np.array(valve_stops, dtype=object)        
    
#         # Sort by first valid element, placing NaNs at the end
#         valid_indices = [i for i in range(len(starts_array)) if not np.isnan(starts_array[i][0])]
#         nan_indices = [i for i in range(len(starts_array)) if np.isnan(starts_array[i][0])]
    
#         sorted_valid_indices = sorted(valid_indices, key=lambda i: starts_array[i][0])
#         sorted_indices = sorted_valid_indices + nan_indices  # NaNs go at the end
    
#         processed_licks_sorted[f"{condition}_starts"] = [starts_array[i] for i in sorted_indices]
#         processed_licks_sorted[f"{condition}_stops"] = [stops_array[i] for i in sorted_indices]        
#         processed_licks_sorted[f"{condition}_valve_starts"] = [valve_starts_array[i] for i in sorted_indices]
#         processed_licks_sorted[f"{condition}_valve_stops"] = [valve_stops_array[i] for i in sorted_indices]   
#     return processed_licks_sorted


# def sort_licks_left_right_opto_non(processed_licks):
#     processed_licks_sorted = {}
    
#     for trial_side in ["left", "right"]:
#         condition_pairs = [(f"{trial_side}_trial_left_no_opto", f"{trial_side}_trial_right_no_opto"),
#                            (f"{trial_side}_trial_left_opto", f"{trial_side}_trial_right_opto")]
        
#         for left_cond, right_cond in condition_pairs:
#             left_starts = processed_licks[f"{left_cond}_starts"]
#             right_starts = processed_licks[f"{right_cond}_starts"]
#             left_stops = processed_licks[f"{left_cond}_stops"]
#             right_stops = processed_licks[f"{right_cond}_stops"]
#             left_valve_starts = processed_licks[f"{left_cond}_valve_starts"]
#             right_valve_starts = processed_licks[f"{right_cond}_valve_starts"]
#             left_valve_stops = processed_licks[f"{left_cond}_valve_stops"]
#             right_valve_stops = processed_licks[f"{right_cond}_valve_stops"]
            
#             if trial_side == "left":
#                 sort_reference = left_starts
#             else:
#                 sort_reference = right_starts
            
#             starts_array = np.array(left_starts + right_starts, dtype=object)
#             stops_array = np.array(left_stops + right_stops, dtype=object)
#             valve_starts_array = np.array(left_valve_starts + right_valve_starts, dtype=object)
#             valve_stops_array = np.array(left_valve_stops + right_valve_stops, dtype=object)
            
#             valid_indices = [i for i in range(len(sort_reference)) if isinstance(sort_reference[i], (list, np.ndarray)) and len(sort_reference[i]) > 0 and not np.isnan(sort_reference[i][0])]
#             nan_indices = [i for i in range(len(sort_reference)) if i not in valid_indices]
            
#             sorted_valid_indices = sorted(valid_indices, key=lambda i: sort_reference[i][0])
#             sorted_indices = sorted_valid_indices + nan_indices
            
#             processed_licks_sorted[f"{left_cond}_starts"] = [left_starts[i] for i in sorted_indices]
#             processed_licks_sorted[f"{right_cond}_starts"] = [right_starts[i] for i in sorted_indices]
#             processed_licks_sorted[f"{left_cond}_stops"] = [left_stops[i] for i in sorted_indices]
#             processed_licks_sorted[f"{right_cond}_stops"] = [right_stops[i] for i in sorted_indices]
#             processed_licks_sorted[f"{left_cond}_valve_starts"] = [left_valve_starts[i] for i in sorted_indices]
#             processed_licks_sorted[f"{right_cond}_valve_starts"] = [right_valve_starts[i] for i in sorted_indices]
#             processed_licks_sorted[f"{left_cond}_valve_stops"] = [left_valve_stops[i] for i in sorted_indices]
#             processed_licks_sorted[f"{right_cond}_valve_stops"] = [right_valve_stops[i] for i in sorted_indices]
            
#     return processed_licks_sorted

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


# def process_session_licks_with_valve_times(session_data, session_idx):
#     """
#     Process licks for a single session and track valve open/close times for each trial,
#     while sorting the lick data and valve times (start/stop), excluding NaN values.
    
#     Args:
#         session_data (dict): Contains left/right lick times, opto flags, and valve events for trials.
    
#     Returns:
#         dict: Processed licks categorized into left/right and opto/non-opto, 
#               along with sorted valve open/close times for each trial.
#     """
#     processed_licks = {}
    
#     conditions = ["left_trial_left_opto", "left_trial_left_no_opto", 
#                   "left_trial_right_opto", "left_trial_right_no_opto",
#                   "right_trial_left_opto", "right_trial_left_no_opto", 
#                   "right_trial_right_opto", "right_trial_right_no_opto"]
    
#     for condition in conditions:
#         processed_licks[f"{condition}_starts"] = []
#         processed_licks[f"{condition}_stops"] = []
#         processed_licks[f"{condition}_valve_starts"] = []
#         processed_licks[f"{condition}_valve_stops"] = []
    
#     raw_data = session_data['raw']
    
#     numTrials = raw_data[session_idx]['nTrials']
        
#     trial_types = session_data['trial_type'][session_idx]
#     opto_flags = session_data['opto_trial'][session_idx]
    
#     for trial in range(numTrials):
#         licks = {}
#         valve_open_times = []  # List to store valve open/close times for the current trial
        
#         # Process left and right lick start/stop times
#         if 'Port1In' not in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']:
#             licks['licks_left_start'] = []
#         elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In'], float):
#             licks['licks_left_start'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']]
#         else:
#             licks['licks_left_start'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']
            
#         if 'Port1Out' not in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']:
#             licks['licks_left_stop'] = []
#         elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out'], float):
#             licks['licks_left_stop'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']]
#         else:
#             licks['licks_left_stop'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']        
    
#         if 'Port3In' not in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']:
#             licks['licks_right_start'] = []
#         elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In'], float):
#             licks['licks_right_start'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']]
#         else:
#             licks['licks_right_start'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']         
        
#         if 'Port3Out' not in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']:
#             licks['licks_right_stop'] = []
#         elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out'], float):
#             licks['licks_right_stop'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']]
#         else:
#             licks['licks_right_stop'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']         
        
#         # Track valve open/close times for the trial (start/stop)
#         if 'ValveOpen' in raw_data[session_idx]['RawEvents']['Trial'][trial]['States']:
#             valve_open_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][0])
#             valve_open_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][1])
                   
#         trial_type = "left_trial" if trial_types[trial] == 1 else "right_trial" 
        
#         is_opto = opto_flags[trial]
        
#         for side in ["left", "right"]:
#             key = f"{trial_type}_{side}_opto" if is_opto else f"{trial_type}_{side}_no_opto"
#             processed_licks[f"{key}_starts"].append(licks[f"licks_{side}_start"])
#             processed_licks[f"{key}_stops"].append(licks[f"licks_{side}_stop"])
        
#         # Remove NaN values from lick times
#         licks_left_start = [x for x in licks['licks_left_start'] if not np.isnan(x)]
#         licks_left_stop = [x for x in licks['licks_left_stop'] if not np.isnan(x)]
#         licks_right_start = [x for x in licks['licks_right_start'] if not np.isnan(x)]
#         licks_right_stop = [x for x in licks['licks_right_stop'] if not np.isnan(x)]
        
#         # Sort the licks for this trial (start and stop times)
#         sorted_left_starts, sorted_left_stops = zip(*sorted(zip(licks_left_start, licks_left_stop), key=lambda x: x[0])) if licks_left_start else ([], [])
#         sorted_right_starts, sorted_right_stops = zip(*sorted(zip(licks_right_start, licks_right_stop), key=lambda x: x[0])) if licks_right_start else ([], [])
        
#         processed_licks[f"{key}_starts"][-1] = sorted_left_starts + sorted_right_starts
#         processed_licks[f"{key}_stops"][-1] = sorted_left_stops + sorted_right_stops
        
#         # Sort and filter out NaN valve times
#         sorted_valve_times = sorted([x for x in valve_open_times if not np.isnan(x)])
        
#         # Update processed_licks dict with valve open/close times for the specific condition
#         processed_licks[f"{key}_valve_starts"].append(sorted_valve_times[::2])  # Valve open times
#         processed_licks[f"{key}_valve_stops"].append(sorted_valve_times[1::2])  # Valve close times
    
#     return processed_licks



def pool_sessions(all_sessions):
    """
    Pool lick data across all sessions.
    
    Args:
        all_sessions (list): List of processed session dictionaries.
    
    Returns:
        dict: Pooled lick data.
    """
    pooled_licks = {
        "left_opto": [], "left_no_opto": [],
        "right_opto": [], "right_no_opto": []
    }
    
    for session in all_sessions:
        for key in pooled_licks:
            pooled_licks[key].extend(session[key])
    
    return pooled_licks

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
    ax.legend(handles=legend_elements)
    plt.show()

# def plot_lick_traces_valves(lick_data, trial_side, title="Lick Traces"):
#     """
#     Plot lick traces for the specified trial side ("left" or "right").
    
#     Args:
#         lick_data (dict): Processed lick data.
#         trial_side (str): Either "left" or "right" to specify which trials to plot.
#         title (str): Plot title.
#     """
#     fig, ax = plt.subplots(figsize=(12, 8))
    
#     conditions = [f"{trial_side}_trial_left_no_opto", f"{trial_side}_trial_left_opto", 
#                   f"{trial_side}_trial_right_no_opto", f"{trial_side}_trial_right_opto"]
#     colors = ['green', 'lightgreen', 'black', 'darkgray']
#     labels = [f"{trial_side.capitalize()} Trial - Left No Opto", f"{trial_side.capitalize()} Trial - Left Opto", 
#               f"{trial_side.capitalize()} Trial - Right No Opto", f"{trial_side.capitalize()} Trial - Right Opto"]
      
#     # trial_offset = 0  # Reset trial offset for each side
#     # for cond, color, label in zip(conditions, colors, labels):
#     #     for trial_idx, (trial_starts, trial_stops, valve_starts, valve_stops) in enumerate(zip(
#     #             lick_data[f"{cond}_starts"], lick_data[f"{cond}_stops"],
#     #             lick_data[f"{cond}_valve_starts"], lick_data[f"{cond}_valve_stops"])):
            
#     #         # Ensure valve times are iterable (even if they're a single float)
#     #         if isinstance(valve_starts, float):
#     #             valve_starts = [valve_starts]
#     #         if isinstance(valve_stops, float):
#     #             valve_stops = [valve_stops]

#     #         valid_trials = False  # Flag to check if there are any valid start/stop pairs
            
#     #         # Plot lick traces (start/stop)
#     #         for start, stop in zip(trial_starts, trial_stops):
#     #             if not np.isnan(start) and not np.isnan(stop):
#     #                 ax.plot([start, stop], [trial_offset] * 2, color=color, alpha=0.7)
#     #                 valid_trials = True  # Mark as a valid trial
            
#     #         # Plot valve open/close times as horizontal lines
#     #         for v_start, v_stop in zip(valve_starts, valve_stops):
#     #             if not np.isnan(v_start) and not np.isnan(v_stop) and valid_trials:
#     #                 ax.plot([v_start, v_stop], [trial_offset] * 2, color='red', lw=2, alpha=0.7)

#     #         # Increment the trial_offset only if valid trials were plotted
#     #         if valid_trials:
#     #             trial_offset += 1 
    
    
    
#     # trial_offset = 0  # Reset trial offset for each side
#     # for cond, color, label in zip(conditions, colors, labels):
#     #     for trial_idx, (trial_starts, trial_stops, valve_starts, valve_stops) in enumerate(zip(
#     #             lick_data[f"{cond}_starts"], lick_data[f"{cond}_stops"],
#     #             lick_data[f"{cond}_valve_starts"], lick_data[f"{cond}_valve_stops"])):

#     #         # Ensure valve times are iterable (even if they're a single float)
#     #         if isinstance(valve_starts, float):
#     #             valve_starts = [valve_starts]
#     #         if isinstance(valve_stops, float):
#     #             valve_stops = [valve_stops]

#     #         valid_trials = False  # Flag to check if there are any valid start/stop pairs
            
#     #         # Plot lick traces (start/stop)
#     #         for start, stop in zip(trial_starts, trial_stops):
#     #             if not np.isnan(start) and not np.isnan(stop):
#     #                 ax.plot([start, stop], [trial_offset] * 2, color=color, alpha=0.7)
#     #                 valid_trials = True  # Mark as a valid trial

#     #         # Plot valve open/close times as horizontal lines, associated with the current trial
#     #         for v_start, v_stop in zip(valve_starts, valve_stops):
#     #             if not np.isnan(v_start) and not np.isnan(v_stop) and valid_trials:
#     #                 ax.plot([v_start, v_stop], [trial_offset] * 2, color='blue', lw=2, alpha=0.7)

#     #         # Increment the trial_offset only if valid trials were plotted
#     #         if valid_trials:
#     #             trial_offset += 1  # Only increment if there were valid trials    
    
#     # trial_offset = 0  # Reset trial offset for each side    
#     # for cond, color, label in zip(conditions, colors, labels):
#     #     for trial_idx, (trial_starts, trial_stops, valve_starts, valve_stops) in enumerate(zip(
#     #             lick_data[f"{cond}_starts"], lick_data[f"{cond}_stops"],
#     #             lick_data[f"{cond}_valve_starts"], lick_data[f"{cond}_valve_stops"])):

#     #         # Ensure valve times are iterable (even if they're a single float)
#     #         if isinstance(valve_starts, float):
#     #             valve_starts = [valve_starts]
#     #         if isinstance(valve_stops, float):
#     #             valve_stops = [valve_stops]

#     #         valid_trials = False  # Flag to check if there are any valid start/stop pairs
#     #         current_trial_offset = trial_offset  # Store the current trial offset to use for both sides
            
#     #         # Plot left side lick traces (start/stop)
#     #         if len(trial_starts) > 0 and len(trial_stops) > 0:  # If left lick data exists
#     #             for start, stop in zip(trial_starts, trial_stops):
#     #                 if not np.isnan(start) and not np.isnan(stop):
#     #                     ax.plot([start, stop], [current_trial_offset] * 2, color=color, alpha=0.7)
#     #                     valid_trials = True  # Mark as a valid trial

#     #         # Plot right side lick traces (start/stop)
#     #         if len(trial_starts) > 0 and len(trial_stops) > 0:  # If right lick data exists
#     #             for start, stop in zip(trial_starts, trial_stops):
#     #                 if not np.isnan(start) and not np.isnan(stop):
#     #                     ax.plot([start, stop], [current_trial_offset + 0.5] * 2, color=color, alpha=0.7)
#     #                     valid_trials = True  # Mark as a valid trial
                        
#     #         # Plot valve open/close times as horizontal lines for both left and right sides
#     #         for v_start, v_stop in zip(valve_starts, valve_stops):
#     #             if not np.isnan(v_start) and not np.isnan(v_stop) and valid_trials:
#     #                 ax.plot([v_start, v_stop], [current_trial_offset, current_trial_offset], color='red', lw=1, alpha=0.7)
#     #                 ax.plot([v_start, v_stop], [current_trial_offset + 0.5, current_trial_offset + 0.5], color='red', lw=1, alpha=0.7)
            
#     #         # Increment the trial_offset only if valid trials were plotted
#     #         if valid_trials:
#     #             trial_offset += 1  # Only increment if there were valid trials    
    
#     trial_offset = 0  # Initialize the trial offset
    
#     for cond, color, label in zip(conditions, colors, labels):
#         for trial_idx, (trial_starts, trial_stops, valve_starts, valve_stops) in enumerate(zip(
#                 lick_data[f"{cond}_starts"], lick_data[f"{cond}_stops"],
#                 lick_data[f"{cond}_valve_starts"], lick_data[f"{cond}_valve_stops"])):

#             # Ensure valve times are iterable (even if they're a single float)
#             if isinstance(valve_starts, float):
#                 valve_starts = [valve_starts]
#             if isinstance(valve_stops, float):
#                 valve_stops = [valve_stops]

#             valid_trials = False  # Flag to check if there are any valid start/stop pairs
#             current_trial_offset = trial_offset  # Store the current trial offset to use for both sides
            
#             # Plot left side lick traces (start/stop)
#             if len(trial_starts) > 0 and len(trial_stops) > 0:  # If left lick data exists
#                 for start, stop in zip(trial_starts, trial_stops):
#                     if not np.isnan(start) and not np.isnan(stop):
#                         ax.plot([start, stop], [current_trial_offset] * 2, color=color, alpha=0.7, lw=0.5)
#                         valid_trials = True  # Mark as a valid trial

#             # Plot right side lick traces (start/stop)
#             if len(trial_starts) > 0 and len(trial_stops) > 0:  # If right lick data exists
#                 for start, stop in zip(trial_starts, trial_stops):
#                     if not np.isnan(start) and not np.isnan(stop):
#                         ax.plot([start, stop], [current_trial_offset + 0.5] * 2, color=color, alpha=0.7, lw=0.5)
#                         valid_trials = True  # Mark as a valid trial
                        
#             # Plot valve open/close times as horizontal lines for both left and right sides
#             for v_start, v_stop in zip(valve_starts, valve_stops):
#                 if not np.isnan(v_start) and not np.isnan(v_stop) and valid_trials:
#                     ax.plot([v_start, v_stop], [current_trial_offset, current_trial_offset], color='red', lw=2, alpha=0.7)
#                     ax.plot([v_start, v_stop], [current_trial_offset + 0.5, current_trial_offset + 0.5], color='red', lw=2, alpha=0.7)
            
#             # Increment the trial_offset only if valid trials were plotted
#             if valid_trials:
#                 trial_offset += 1  # Only increment if there were valid trials    
    
#     # Create custom legend with correct colors
#     legend_elements = [Line2D([0], [0], color='green', lw=2, label=labels[0]),
#                        Line2D([0], [0], color='lightgreen', lw=2, label=labels[1]),
#                        Line2D([0], [0], color='black', lw=2, label=labels[2]),
#                        Line2D([0], [0], color='lightgray', lw=2, label=labels[3])]    
    
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Trials")
#     ax.set_title(title)
#     ax.legend(handles=legend_elements)
#     plt.show()
    
def plot_lick_traces_valves_type(lick_data, trial_side, title="Lick Traces"):
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

def analyze_licks(dates, sessions):
    """
    Process and plot lick traces for multiple sessions.
    
    Args:
        dates (list): List of session dates.
        sessions (list): List of session data dictionaries.
    """
    all_processed = []
    
    for i in range(len(dates)):
        session_licks = process_session_licks(sessions[i])
        all_processed.append(session_licks)
        plot_lick_traces(session_licks, title=f"Session {dates[i]}")
    
    pooled_licks = pool_sessions(all_processed)
    plot_lick_traces(pooled_licks, title="Pooled Sessions")


def run(subject_session_data,output_dir_onedrive, output_dir_local):
    max_sessions = 20
    subject = subject_session_data['subject']
    dates = subject_session_data['dates']
    session_id = np.arange(len(dates)) + 1
    jitter_flag = subject_session_data['jitter_flag']
    raw_data = subject_session_data['raw']
    outcomes = subject_session_data['outcomes']
    outcomes_time = subject_session_data['outcomes_time']
    #categories = subject_session_data['isi_post_emp']
    
    
    subject = remove_substrings(subject, ['_opto', '_reg'])
    subject = flip_underscore_parts(subject)
    subject = lowercase_h(subject)
    
    
    trial_type = subject_session_data['trial_type']
    opto_flag = subject_session_data['opto_trial']
    
    row = 4 
    col = 5


    # alignments = ['1st flash' , '3th flash' , '4th flash' , 'choice window' , 'outcome']
    alignments = ['Choice Window' , 'Outcome']
    row_names = ['rewarded short' , 'rewarded long' , 'punished short' , 'punished long']
    n_bins = 500
    
    processed_licks = []
    
    for i in range(len(dates)):
        print(i)
        fig, axs = plt.subplots(nrows=4, ncols=len(alignments), figsize=(40, 30))
        #fig2, axs1 = plt.subplots(nrows=4, ncols=len(alignments), figsize=(40, 30))
        pdf_streams = []
        pdf_paths = []
        numTrials = raw_data[i]['nTrials']
        outcome = outcomes[i]
        outcome_time = outcomes_time[i]
        session_date = dates[i]
        
        processed_licks, processed_licks_sorted = process_session_licks(subject_session_data, i)
        # processed_licks = process_session_licks(subject_session_data, i)
        # processed_licks_sorted = sort_licks_left_right_opto_non(processed_licks)
        
        
        plot_lick_traces(processed_licks, 'left')
        plot_lick_traces(processed_licks_sorted, 'left')
        
        # processed_licks, processed_licks_sorted = process_session_licks_with_valve_times(subject_session_data, i)
        processed_licks = process_session_licks_with_valve_times(subject_session_data, i)
        processed_licks_sorted = sort_licks_left_right_opto_non(processed_licks)
        
        # processed_licks = process_session_licks(subject_session_data, i)
        # processed_licks_sorted = sort_licks_left_right_opto_non(processed_licks)        
        
        # left licks
        plot_lick_traces_valves_type(processed_licks, trial_side='left', title='Ordered by Trial')
        plot_lick_traces_valves_type(processed_licks_sorted, trial_side='left', title='Ordered by Earliest Lick')
        
        
        # right licks
        plot_lick_traces_valves_type(processed_licks, 'right', 'Ordered by Trial')
        plot_lick_traces_valves_type(processed_licks_sorted, 'right', 'Ordered by Earliest Lick')
        
        plot_lick_traces_valves(processed_licks, title='Ordered by Trial')
        
        
        print('')
        if 0:
            row = 0
            for j in range(len(alignments)):
                series_right_rl = []
                series_right_rs = []
                series_right_ps = []
                series_right_pl = []
                
                series_right_rl_num = []
                series_right_rs_num = []
                series_right_ps_num = []
                series_right_pl_num = []            
                
                series_right_rl_opto = []
                series_right_rs_opto = []
                series_right_ps_opto = []
                series_right_pl_opto = []            
    
                series_right_rl_opto_num = []
                series_right_rs_opto_num = []
                series_right_ps_opto_num = []
                series_right_pl_opto_num = [] 
                
                series_center_rl = []
                series_center_rs = []
                series_center_ps = []
                series_center_pl = []
         
                series_center_rl_opto = []
                series_center_rs_opto = []
                series_center_ps_opto = []
                series_center_pl_opto = []            
                
                series_left_rl = []
                series_left_rs = []
                series_left_ps = []
                series_left_pl = []
                
                series_left_rl_num = []
                series_left_rs_num = []
                series_left_ps_num = []
                series_left_pl_num = []
                                   
                series_left_rl_opto = []
                series_left_rs_opto = []
                series_left_ps_opto = []
                series_left_pl_opto = []              
                
                series_left_rl_opto_num = []
                series_left_rs_opto_num = []
                series_left_ps_opto_num = []
                series_left_pl_opto_num = []
                
                colors = []
                
                print(i)
                for trial in range(numTrials):
                    print(trial)
                    
                    stim_seq = np.divide(subject_session_data['stim_seq'][i][trial],1000)
                    step = 10000
                    start = 0
                    # category = 1000*np.mean(raw_data[i]['ProcessedSessionData'][trial]['trial_isi']['PostISI'])
                    # category = 1000*np.mean(raw_data[i]['TrialSettings'][0]['GUI']['ISIOrig_s'])
                    
    
                    if not 'Port1In' in raw_data[i]['RawEvents']['Trial'][trial]['Events'].keys():
                            port1 = []
                    elif type(raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port1In']) == float:
                        port1 = [raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port1In']]
                    else:
                        port1 = raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port1In']
    
                    if not 'Port2In' in raw_data[i]['RawEvents']['Trial'][trial]['Events'].keys():
                        port2= []
                    elif type(raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port2In']) == float:
                        port2 = [raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port2In']]
                    else:
                        port2 = raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port2In']
    
                    if not 'Port3In' in raw_data[i]['RawEvents']['Trial'][trial]['Events'].keys():
                        port3= []
                    elif type(raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port3In']) == float:
                        port3 = [raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port3In']]
                    else:
                        port3 = raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port3In']
                        
                    # make sure port vars are list type
                    port1 = ensure_list(port1)
                    port2 = ensure_list(port2)
                    port3 = ensure_list(port3)                
                    
                    # if j == 4:
                    #     alignment = outcome_time[trial]
                    # elif j == 3:
                    #     alignment = raw_data[i]['RawEvents']['Trial'][trial]['States']['WindowChoice'][0]
                    # elif j == 2:
                    #     if len(stim_seq[1 , :]) > 2:
                    #         alignment = stim_seq[1 , 2]
                    #     else:
                    #         alignment = 'nan'
                    # elif j == 1:
                    #     if len(stim_seq[1 , :]) > 3:
                    #         alignment = stim_seq[1 , 3]
                    #     else:
                    #         alignment = 'nan'
                    # elif j == 0:
                    #     if len(stim_seq[1 , :]) > 0:
                    #         alignment = stim_seq[1 , 0]
                    #     else:
                    #         alignment = 'nan'
                    
                    if j == 1:
                        alignment = outcome_time[trial]                    
                    elif j == 0:
                        alignment = raw_data[i]['RawEvents']['Trial'][trial]['States']['WindowChoice'][0]                    
                            
                    if opto_flag[i][trial]:
                        print(trial)
                        
                    if not alignment == 'nan':
                        if outcome[trial] == 'Reward':
                            if trial_type[i][trial] == 1:
                                if opto_flag[i][trial]:
                                    series_right_rs_opto.append([x - alignment for x in port3])
                                    series_center_rs_opto.append([x - alignment for x in port2])
                                    series_left_rs_opto.append([x - alignment for x in port1])   
                                    
                                    series_right_rs_opto_num.append(trial)
                                    series_left_rs_opto_num.append(trial)
                                else:
                                    series_right_rs.append([x - alignment for x in port3])
                                    series_center_rs.append([x - alignment for x in port2])
                                    series_left_rs.append([x - alignment for x in port1])
                                    
                                    series_right_rs_num.append(trial)
                                    series_left_rs_num.append(trial)                                
                            else:
                                if opto_flag[i][trial]:       
                                    series_right_rl_opto.append([x - alignment for x in port3])
                                    series_center_rl_opto.append([x - alignment for x in port2])
                                    series_left_rl_opto.append([x - alignment for x in port1])   
                                    
                                    series_right_rl_opto_num.append(trial)
                                    series_left_rl_opto_num.append(trial)
                                else:
                                    series_right_rl.append([x - alignment for x in port3])
                                    series_center_rl.append([x - alignment for x in port2])
                                    series_left_rl.append([x - alignment for x in port1])
                                    
                                    series_right_rl_num.append(trial)
                                    series_left_rl_num.append(trial)                                
                                 
                                    
                            # if category < 500:
                            #     if len(stim_seq[1 , :]) > 3:
                            #         series_right_rs.append([x - alignment for x in port3])
                            #         series_center_rs.append([x - alignment for x in port2])
                            #         series_left_rs.append([x - alignment for x in port1])
                            # if category > 500:
                            #     if len(stim_seq[1 , :]) > 3:
                            #         colors.append('red')
                            #         series_right_rl.append([x - alignment for x in port3])
                            #         series_center_rl.append([x - alignment for x in port2])
                            #         series_left_rl.append([x - alignment for x in port1])
    
                        if outcome[trial] == 'Punish':
                            if trial_type[i][trial] == 1:
                                if opto_flag[i][trial]:
                                    series_right_ps_opto.append([x - alignment for x in port3])
                                    series_center_ps_opto.append([x - alignment for x in port2])
                                    series_left_ps_opto.append([x - alignment for x in port1])   
                                    
                                    
                                else:
                                    series_right_ps.append([x - alignment for x in port3])
                                    series_center_ps.append([x - alignment for x in port2])
                                    series_left_ps.append([x - alignment for x in port1])
                                    
                                    
                            else:
                                if opto_flag[i][trial]:
                                    series_right_pl_opto.append([x - alignment for x in port3])
                                    series_center_pl_opto.append([x - alignment for x in port2])
                                    series_left_pl_opto.append([x - alignment for x in port1])  
                                    
                                    
                                else:                        
                                    series_right_pl.append([x - alignment for x in port3])
                                    series_center_pl.append([x - alignment for x in port2])
                                    series_left_pl.append([x - alignment for x in port1])    
                                    
                                    
                                            
                            
                            # if category < 500:
                            #     if len(stim_seq[1 , :]) > 3:
                            #         colors.append('yellow')
                            #         series_right_ps.append([x - alignment for x in port3])
                            #         series_center_ps.append([x - alignment for x in port2])
                            #         series_left_ps.append([x - alignment for x in port1])
    
                            # if category > 500:
                            #     if len(stim_seq[1 , :]) > 3:
                            #         colors.append('green')
                            #         series_right_pl.append([x - alignment for x in port3])
                            #         series_center_pl.append([x - alignment for x in port2])
                            #         series_left_pl.append([x - alignment for x in port1])
                
                '''
                For blue and light blue, common color names in string format are:
    
                Blue → "blue"
                Light Blue → "lightblue"
                Other variations include:
                
                Sky Blue → "skyblue"
                Powder Blue → "powderblue"
                Deep Sky Blue → "deepskyblue"
                Dodger Blue → "dodgerblue"
                Steel Blue → "steelblue"
                
                
                For red and light red, the common color names in string format are:
                
                Red → "red"
                Light Red → No standard "lightred", but similar options include:
                Salmon → "salmon"
                Light Coral → "lightcoral"
                Indian Red → "indianred"
                Tomato → "tomato"   
                
                
                For green, dark green, and light green in Matplotlib, you can use the following color names as strings:
                
                Green → "green" or "#008000"
                Dark Green → "darkgreen" or "#006400"
                Light Green → "lightgreen" or "#90EE90"
                Lime Green (a bright alternative) → "limegreen" or "#32CD32"
                Pale Green (a very light alternative) → "palegreen" or "#98FB98"            
                
                For black, dark black, and light black (grayish tones) in Matplotlib, use these color names:
                
                Black → "black" or "#000000"
                Dark Gray (near black) → "dimgray" or "#696969"
                Light Gray (faded black) → "lightgray" or "#D3D3D3"
                Very Dark Gray → "gray" or "#808080"
                Charcoal (deep dark gray) → "slategray" or "#708090"            
                
                '''
                
                xlim_left = -0.1
                xlim_right = 4
                
                # SHORT DUR
                #  averages
                ################################################################################################################  
                
                # Define colors: one color per group
                colorlist = ['green', 'lightgreen', 'dimgray', 'lightgray']
                
                row = 0
                
                # Combine all events
                all_events = series_left_rs + series_left_rs_opto + series_right_ps + series_right_ps_opto # Concatenate both groups        
                
                # Step 1: Find the minimum number of licks across all trials
                min_length_series_left_rs = min(len(x) for x in series_left_rs)
                min_length_series_left_rs_opto = min(len(x) for x in series_left_rs_opto)
                min_length_series_right_ps = min(len(x) for x in series_right_ps)
                min_length_series_right_ps_opto = min(len(x) for x in series_right_ps_opto)
                
    
                # Step 2: Limit each trial to the minimum number of licks
                limited_licks_series_left_rs = [x[:min_length_series_left_rs] for x in series_left_rs]
                
                limited_licks_series_left_rs_opto = [x[:min_length_series_left_rs_opto] for x in series_left_rs_opto]
                limited_licks_series_right_ps = [x[:min_length_series_right_ps] for x in series_right_ps]
                limited_licks_series_right_ps_opto = [x[:min_length_series_right_ps_opto] for x in series_right_ps_opto]
                
                # Step 3: Transpose to group licks by index (1st, 2nd, 3rd lick)
                licks_transposed_series_left_rs = np.array(limited_licks_series_left_rs).T            
                licks_transposed_series_left_rs_opto = np.array(limited_licks_series_left_rs_opto).T
                licks_transposed_series_right_ps = np.array(limited_licks_series_right_ps).T
                licks_transposed_series_right_ps_opto = np.array(limited_licks_series_right_ps_opto).T
                
                # Step 4: Calculate the element-wise average of the licks across trials
                avg_lick_trace_series_left_rs = np.mean(licks_transposed_series_left_rs, axis=1)                          
                avg_lick_trace_series_left_rs_opto = np.mean(licks_transposed_series_left_rs_opto, axis=1)              
                avg_lick_trace_series_right_ps = np.mean(licks_transposed_series_right_ps, axis=1)              
                avg_lick_trace_series_right_ps_opto = np.mean(licks_transposed_series_right_ps_opto, axis=1)     
    
                
                avg_lick_traces = [
                    avg_lick_trace_series_left_rs,
                    avg_lick_trace_series_left_rs_opto,
                    avg_lick_trace_series_right_ps,
                    avg_lick_trace_series_right_ps_opto
                ]
                
                # Step 1: Find the maximum length of the traces
                max_length = max(len(trace) for trace in avg_lick_traces)
                
                
                padded_lick_traces = [np.pad(trace, (0, max_length - len(trace)), mode='constant', constant_values=np.nan).tolist() for trace in avg_lick_traces]
                y_positions = np.arange(len(padded_lick_traces))
                num_traces = len(padded_lick_traces)
                   
                # axs[row , j].eventplot(padded_lick_traces, lineoffsets=y_positions, color=colorlist, linelengths = 0.3)            
                
                # axs[row , j].set_xlim([xlim_left,xlim_right])
                # axs[row,  j].set_ylim(-1, num_traces)  # Ensure all rows fit within view            
                # axs[row , j].set_title('Type: Short ISI, Sorted: Trial Order, Aligned: ' + alignments[j])
    
                # # Define y-ticks and labels for each sublist
                
                # yticklabels = ['Left Avg', 'Left Opto Avg', 'Right Avg', 'Right Opto Avg']
                # yticks = np.arange(len(yticklabels))
                
                # Set yticks and labels
                # axs[row , j].set_yticks(yticks)
                # axs[row , j].set_yticklabels(yticklabels)
    
                # # # Add one empty plot for each color-label pair
    
                # # Add legend
                # axs[row , j].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)            
                
                # # Get the figure handle from the axes
                # figure_handle = axs[row , j].figure
                # figure_handle.show()
                
                ################################################################################################################  
                
                labels = ['Left Lick', \
                          'Left Lick - Opto', \
                          'Right Lick', \
                          'Right Lick - Opto',]            
                
                # Flattened y-positions: each sublist gets a unique row
                num_sublists_1 = len(series_left_rs)
                num_sublists_2 = len(series_left_rs_opto)
                num_sublists_3 = len(series_right_ps)
                num_sublists_4 = len(series_right_ps_opto)
    
                # num_sublists_1 = len(series_right_rs)
                # num_sublists_2 = len(series_right_rs_opto)
                
                y_positions = list(range(num_sublists_1)) + \
                    list(range(num_sublists_1, num_sublists_1 + num_sublists_2)) + \
                    list(range(num_sublists_1 + num_sublists_2, num_sublists_1 + num_sublists_2 + num_sublists_3)) + \
                    list(range(num_sublists_1 + num_sublists_2 + num_sublists_3, num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4))
                
                # y_ticks = list(range(num_sublists_1)) + \
                #     list(range(num_sublists_2)) + \
                #     list(range(num_sublists_3)) + \
                #     list(range(num_sublists_4))
                
                # y_ticks = [x + 1 for x in y_ticks]
                
                # Updated y_positions with space for averages
                y_positions = []
                modified_traces = []  # Will contain avg traces + original traces
                
                sublists = [series_left_rs, series_left_rs_opto, series_right_ps, series_right_ps_opto]
                
                y_offset = 0  # Keeps track of row index
                for idx, sublist in enumerate(sublists):
                    # Insert the average trace above the sublist
                    modified_traces.append(padded_lick_traces[idx])
                    y_positions.append(y_offset)  # Position for average
                    y_offset += 1  # Shift position
                
                    # Add the original traces with updated positions
                    for trace in sublist:
                        modified_traces.append(trace)
                        y_positions.append(y_offset)
                        y_offset += 1  # Shift for next row            
                
                y_ticks = list(range(len(modified_traces)))
                
                # Define colors: one color per group
                colorlist = ['green', 'lightgreen', 'dimgray', 'lightgray']
                colors = ['green'] * num_sublists_1 + ['lightgreen'] * num_sublists_2  + \
                    ['dimgray'] * num_sublists_3 + ['lightgray'] * num_sublists_4 
                
                colors = ['green'] * num_sublists_1 + \
                    ['red'] * 1 + \
                    ['lightgreen'] * num_sublists_2  + \
                    ['red'] * 1 + \
                    ['dimgray'] * num_sublists_3 + \
                    ['red'] * 1 + \
                    ['lightgray'] * num_sublists_4 + \
                    ['red'] * 1
              
                # # Step 1: Concatenate all the lists
                # concatenated_data = np.concatenate(series_left_rs)
                
                # # Step 2: Compute the histogram
                # # `bins` defines the range of values (e.g., from min to max of the data)
                # bins = 100  # Number of bins you want to divide the range into
                # hist, bin_edges = np.histogram(concatenated_data, bins=bins, density=True)
                
                # hist = hist * 100
                
                # # Step 3: Normalize the histogram to get the probability density
                # # This is already handled by setting `density=True` in np.histogram
                
                # # Step 4: Plot the probability density curve
                # # Compute the midpoints of the bins
                # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # if j == 1:
                #     axs[row , j].plot(bin_centers, hist, label="Probability Density")
                
                
                # 'reward, short, ' + alignments[j]
                # trial ordered
                ################################################################################################################
                
                axs[row , j].vlines(0 ,len(y_positions), 0, linestyle='--', color='grey')
                # axs[row , j].eventplot(series_center_rs, color='black', linelengths = 0.3)
                # axs[row , j].eventplot(series_right_rs, color='red', linelengths = 0.3)
                # axs[row , j].eventplot(series_left_rs, color='limegreen', linelengths = 0.3)
                
                # axs[row , j].eventplot(series_right_rs_opto, color='red', linelengths = 0.3)
                # axs[row , j].eventplot(series_left_rs_opto, color='indigo', linelengths = 0.3)            
                
                # axs[row , j].eventplot(all_events, color=colors, lineoffsets=y_positions, linelengths = 0.6)      
                
                axs[row,  j].eventplot(modified_traces, color=colors, linelengths=0.6, lineoffsets=y_positions)
                
                axs[row , j].set_xlim([xlim_left,xlim_right])
                axs[row , j].set_title('Type: Short ISI, Sorted: Trial Order, Aligned: ' + alignments[j])
                # if len(series_right_rs) > 0:
                    
                # axs[row , j].hist(np.concatenate(series_center_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black', alpha = 0.2)
                # axs[row , j].hist(np.concatenate(series_right_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red', alpha = 0.2)
                
                # axs[row , j].hist(np.concatenate(series_left_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen', alpha = 0.2)
    
                # Add one empty plot for each color-label pair
                for k in range(len(colorlist)):
                    axs[row , j].plot([], [], color=colorlist[k], label=labels[k])
    
                # Add legend
                axs[row , j].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
               
                # Define y-ticks and labels for each sublist
                yticks = [num_sublists_1 / 2, num_sublists_1 + num_sublists_2 / 2, 
                          num_sublists_1 + num_sublists_2 + num_sublists_3 / 2, 
                          num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4 / 2]
                
                yticklabels = ['Left', 'Left Opto', 'Right', 'Right Opto']
                
                # Set yticks and labels
                axs[row , j].set_yticks(yticks)
                axs[row , j].set_yticklabels(yticklabels)
                
                line_thickness = 0.5  # Adjust line thickness here
                
                # Draw partition lines to divide sublists
                axs[row , j].axhline(y=num_sublists_1+0.5, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2+1.5, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + num_sublists_3+2.5, color='grey', linestyle='--', linewidth=line_thickness)          
               
                # Create a twin y-axis
                axs2 = axs[row , j].twinx() 
                
                # Set different tick marks (e.g., transformed scale)
                axs2.set_ylim(axs[row , j].get_ylim())  # Keep same limits
                axs2.set_yticks(y_positions)  # tick positions
                axs2.set_yticklabels(y_ticks)  # tick labels
                axs2.set_ylabel("Trials", color='black')
                axs2.tick_params(axis='y', labelcolor='black',  size=2, labelsize=3)        
                
                
                
                # SHORT DUR
                # 
                ################################################################################################################            
                
                row = row + 1
                
                labels = ['Left Lick', \
                          'Left Lick - Opto', \
                          'Right Lick', \
                          'Right Lick - Opto',]            
                
                # Flattened y-positions: each sublist gets a unique row
                num_sublists_1 = len(series_left_rs)
                num_sublists_2 = len(series_left_rs_opto)
                num_sublists_3 = len(series_right_ps)
                num_sublists_4 = len(series_right_ps_opto)
    
                # num_sublists_1 = len(series_right_rs)
                # num_sublists_2 = len(series_right_rs_opto)
                
                y_positions = list(range(num_sublists_1)) + list(range(num_sublists_1, num_sublists_1 + num_sublists_2)) + \
                    list(range(num_sublists_1 + num_sublists_2, num_sublists_1 + num_sublists_2 + num_sublists_3)) + \
                    list(range(num_sublists_1 + num_sublists_2 + num_sublists_3, num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4))
                
                y_ticks = list(range(num_sublists_1)) + list(range(num_sublists_2)) + list(range(num_sublists_3)) + list(range(num_sublists_4))
                
                y_ticks = [x + 1 for x in y_ticks]
                
                
                # Define colors: one color per group
                colorlist = ['green', 'lightgreen', 'dimgray', 'lightgray']
                colors = ['green'] * num_sublists_1 + ['lightgreen'] * num_sublists_2  + \
                    ['dimgray'] * num_sublists_3 + ['lightgray'] * num_sublists_4 
                
              
                # # Step 1: Concatenate all the lists
                # concatenated_data = np.concatenate(series_left_rs)
                
                # # Step 2: Compute the histogram
                # # `bins` defines the range of values (e.g., from min to max of the data)
                # bins = 100  # Number of bins you want to divide the range into
                # hist, bin_edges = np.histogram(concatenated_data, bins=bins, density=True)
                
                # hist = hist * 100
                
                # # Step 3: Normalize the histogram to get the probability density
                # # This is already handled by setting `density=True` in np.histogram
                
                # # Step 4: Plot the probability density curve
                # # Compute the midpoints of the bins
                # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # if j == 1:
                #     axs[row , j].plot(bin_centers, hist, label="Probability Density")
                
                
                # 'reward, short, ' + alignments[j]
                # trial ordered
                ################################################################################################################
                
                axs[row , j].vlines(0 ,len(y_positions), 0, linestyle='--', color='grey')
                # axs[row , j].eventplot(series_center_rs, color='black', linelengths = 0.3)
                # axs[row , j].eventplot(series_right_rs, color='red', linelengths = 0.3)
                # axs[row , j].eventplot(series_left_rs, color='limegreen', linelengths = 0.3)
                
                # axs[row , j].eventplot(series_right_rs_opto, color='red', linelengths = 0.3)
                # axs[row , j].eventplot(series_left_rs_opto, color='indigo', linelengths = 0.3)            
                
                axs[row , j].eventplot(all_events, color=colors, lineoffsets=y_positions, linelengths = 0.6)            
                
                axs[row , j].set_xlim([xlim_left,xlim_right])
                axs[row , j].set_title('Type: Short ISI, Sorted: Trial Order, Aligned: ' + alignments[j])
                # if len(series_right_rs) > 0:
                    
                # axs[row , j].hist(np.concatenate(series_center_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black', alpha = 0.2)
                # axs[row , j].hist(np.concatenate(series_right_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red', alpha = 0.2)
                
                # axs[row , j].hist(np.concatenate(series_left_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen', alpha = 0.2)
    
                # Add one empty plot for each color-label pair
                for k in range(len(colorlist)):
                    axs[row , j].plot([], [], color=colorlist[k], label=labels[k])
    
                # Add legend
                axs[row , j].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
               
                # Define y-ticks and labels for each sublist
                yticks = [num_sublists_1 / 2, num_sublists_1 + num_sublists_2 / 2, 
                          num_sublists_1 + num_sublists_2 + num_sublists_3 / 2, 
                          num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4 / 2]
                
                yticklabels = ['Left', 'Left Opto', 'Right', 'Right Opto']
                
                # Set yticks and labels
                axs[row , j].set_yticks(yticks)
                axs[row , j].set_yticklabels(yticklabels)
                
                line_thickness = 0.5  # Adjust line thickness here
                
                # Draw partition lines to divide sublists
                axs[row , j].axhline(y=num_sublists_1-0.5, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2-0.5, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + num_sublists_3-0.5, color='grey', linestyle='--', linewidth=line_thickness)          
               
                # Create a twin y-axis
                axs2 = axs[row , j].twinx() 
                
                # Set different tick marks (e.g., transformed scale)
                axs2.set_ylim(axs[row , j].get_ylim())  # Keep same limits
                axs2.set_yticks(y_positions)  # tick positions
                axs2.set_yticklabels(y_ticks)  # tick labels
                axs2.set_ylabel("Trials", color='black')
                axs2.tick_params(axis='y', labelcolor='black',  size=2, labelsize=3)            
               
               
    
                # 'reward, long, ' + alignments[j]
                # trial ordered
                ################################################################################################################
                
                row = row + 1
    
                labels = ['Left Lick', \
                          'Left Lick - Opto', \
                          'Right Lick', \
                          'Right Lick - Opto',]
           
                ##############            
                # Flattened y-positions: each sublist gets a unique row
                num_sublists_1 = len(series_left_pl)
                num_sublists_2 = len(series_left_pl_opto)
                num_sublists_3 = len(series_right_rl)
                num_sublists_4 = len(series_right_rl_opto)
            
                y_positions = list(range(num_sublists_1)) + list(range(num_sublists_1, num_sublists_1 + num_sublists_2)) + \
                    list(range(num_sublists_1 + num_sublists_2, num_sublists_1 + num_sublists_2 + num_sublists_3)) + \
                    list(range(num_sublists_1 + num_sublists_2 + num_sublists_3, num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4))
                
                y_ticks = list(range(num_sublists_1)) + list(range(num_sublists_2)) + list(range(num_sublists_3)) + list(range(num_sublists_4))
                
                y_ticks = [x + 1 for x in y_ticks]               
                
                # Define colors: one color per group
                colors = ['green'] * num_sublists_1 + ['lightgreen'] * num_sublists_2  + \
                    ['dimgray'] * num_sublists_3 + ['lightgray'] * num_sublists_4 
                
                # Combine all events
                all_events = series_left_pl + series_left_pl_opto + series_right_rl + series_right_rl_opto # Concatenate both groups   
    
                axs[row , j].vlines(0 ,len(y_positions), 0, linestyle='--', color='grey')
                          
                axs[row , j].eventplot(all_events, color=colors, lineoffsets=y_positions, linelengths = 0.6)
                
                axs[row , j].set_xlim([xlim_left,xlim_right])
                axs[row , j].set_title('Type: Long ISI, Sorted: Trial Order, Aligned: ' + alignments[j])
    
                # Add one empty plot for each color-label pair
                for k in range(len(colorlist)):
                    axs[row , j].plot([], [], color=colorlist[k], label=labels[k])
    
                # Add legend
                axs[row , j].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
                # Define y-ticks and labels for each sublist
                yticks = [num_sublists_1 / 2, num_sublists_1 + num_sublists_2 / 2, 
                          num_sublists_1 + num_sublists_2 + num_sublists_3 / 2, 
                          num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4 / 2]
                
                yticklabels = ['Left', 'Left Opto', 'Right', 'Right Opto']
                
                # Set yticks and labels
                axs[row , j].set_yticks(yticks)
                axs[row , j].set_yticklabels(yticklabels)
                
                line_thickness = 0.5  # Adjust line thickness here
                
                # Draw partition lines to divide sublists
                axs[row , j].axhline(y=num_sublists_1, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + num_sublists_3, color='grey', linestyle='--', linewidth=line_thickness) 
    
    
                # Create a twin y-axis
                axs2 = axs[row , j].twinx() 
                
                # Set different tick marks (e.g., transformed scale)
                axs2.set_ylim(axs[row , j].get_ylim())  # Keep same limits
                axs2.set_yticks(y_positions)  # tick positions
                axs2.set_yticklabels(y_ticks)  # tick labels
                axs2.set_ylabel("Trials", color='black')
                axs2.tick_params(axis='y', labelcolor='black',  size=2, labelsize=3)      
    
                # 'reward, short, ' + alignments[j]
                # time-to-first-lick ordered
                ################################################################################################################
                   
                row = row + 1
                   
                labels = ['Left Lick', \
                          'Left Lick - Opto', \
                          'Right Lick', \
                          'Right Lick - Opto',]
                   
                series_left_rs_sorted = sorted(series_left_rs, key=lambda x: x[0])
                series_left_rs_opto_sorted = sorted(series_left_rs_opto, key=lambda x: x[0])            
                series_right_ps_sorted = sorted(series_right_ps, key=lambda x: x[0])
                series_right_ps_opto_sorted = sorted(series_right_ps_opto, key=lambda x: x[0])
                
                
                ##############            
                # Flattened y-positions: each sublist gets a unique row
                num_sublists_1 = len(series_left_rs_sorted)
                num_sublists_2 = len(series_left_rs_opto_sorted)
                num_sublists_3 = len(series_right_ps_sorted)
                num_sublists_4 = len(series_right_ps_opto_sorted)
                            
                y_positions = list(range(num_sublists_1)) + list(range(num_sublists_1, num_sublists_1 + num_sublists_2)) + \
                    list(range(num_sublists_1 + num_sublists_2, num_sublists_1 + num_sublists_2 + num_sublists_3)) + \
                    list(range(num_sublists_1 + num_sublists_2 + num_sublists_3, num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4))
                
                y_ticks = list(range(num_sublists_1)) + list(range(num_sublists_2)) + list(range(num_sublists_3)) + list(range(num_sublists_4))
                
                y_ticks = [x + 1 for x in y_ticks]            
                
                # Define colors: one color per group
                colors = ['green'] * num_sublists_1 + ['lightgreen'] * num_sublists_2  + \
                    ['dimgray'] * num_sublists_3 + ['lightgray'] * num_sublists_4 
                
                # Combine all events
                all_events = series_left_rs_sorted + series_left_rs_opto_sorted + series_right_ps_sorted + series_right_ps_opto_sorted # Concatenate both groups                                        
                
                axs[row , j].eventplot(all_events, color=colors, lineoffsets=y_positions, linelengths = 0.6)
                   
                axs[row , j].vlines(0 ,len(y_positions), 0, linestyle='--', color='grey')
                                    
                axs[row , j].set_xlim([xlim_left,xlim_right])
                axs[row , j].set_title('Type: Short ISI, Sorted: First Lick, Aligned: ' + alignments[j])                
                   
                # Add one empty plot for each color-label pair
                for k in range(len(colorlist)):
                    axs[row , j].plot([], [], color=colorlist[k], label=labels[k])
                   
                # Add legend
                axs[row , j].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
                   
                # Define y-ticks and labels for each sublist
                yticks = [num_sublists_1 / 2, num_sublists_1 + num_sublists_2 / 2, 
                          num_sublists_1 + num_sublists_2 + num_sublists_3 / 2, 
                          num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4 / 2]
                
                yticklabels = ['Left', 'Left Opto', 'Right', 'Right Opto']
                
                # Set yticks and labels
                axs[row , j].set_yticks(yticks)
                axs[row , j].set_yticklabels(yticklabels)
                
                line_thickness = 0.5  # Adjust line thickness here
                
                # Draw partition lines to divide sublists
                axs[row , j].axhline(y=num_sublists_1, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + num_sublists_3, color='grey', linestyle='--', linewidth=line_thickness) 
                   
                # Create a twin y-axis
                axs2 = axs[row , j].twinx() 
                
                # Set different tick marks (e.g., transformed scale)
                axs2.set_ylim(axs[row , j].get_ylim())  # Keep same limits
                axs2.set_yticks(y_positions)  # tick positions
                axs2.set_yticklabels(y_ticks)  # tick labels
                axs2.set_ylabel("Trials", color='black')
                axs2.tick_params(axis='y', labelcolor='black',  size=2, labelsize=3)      
    
    
                # 'reward, long, ' + alignments[j]
                # time-to-first-lick ordered
                ################################################################################################################
                
                if 0:
                    row = row + 1
        
                    labels = ['Left Lick', \
                              'Left Lick - Opto', \
                              'Right Lick', \
                              'Right Lick - Opto',]
        
                    series_left_pl_sorted = sorted(series_left_pl, key=lambda x: x[0])
                    series_left_pl_opto_sorted = sorted(series_left_pl_opto, key=lambda x: x[0])            
                    series_right_rl_sorted = sorted(series_right_rl, key=lambda x: x[0])
                    series_right_rl_opto_sorted = sorted(series_right_rl_opto, key=lambda x: x[0])
                    
                    
                    ##############            
                    # Flattened y-positions: each sublist gets a unique row
                    num_sublists_1 = len(series_left_pl_sorted)
                    num_sublists_2 = len(series_left_pl_opto_sorted)
                    num_sublists_3 = len(series_right_rl_sorted)
                    num_sublists_4 = len(series_right_rl_opto_sorted)
                  
                    y_positions = list(range(num_sublists_1)) + list(range(num_sublists_1, num_sublists_1 + num_sublists_2)) + \
                        list(range(num_sublists_1 + num_sublists_2, num_sublists_1 + num_sublists_2 + num_sublists_3)) + \
                        list(range(num_sublists_1 + num_sublists_2 + num_sublists_3, num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4))
                    
                    y_ticks = list(range(num_sublists_1)) + list(range(num_sublists_2)) + list(range(num_sublists_3)) + list(range(num_sublists_4))
                    
                    y_ticks = [x + 1 for x in y_ticks]               
                    
                    # Define colors: one color per group
                    colors = ['green'] * num_sublists_1 + ['lightgreen'] * num_sublists_2  + \
                        ['dimgray'] * num_sublists_3 + ['lightgray'] * num_sublists_4 
                    
                    # Combine all events
                    all_events = series_left_pl_sorted + series_left_pl_opto_sorted + series_right_rl_sorted + series_right_rl_opto_sorted # Concatenate both groups        
        
                    axs[row , j].vlines(0 ,len(y_positions), 0, linestyle='--', color='grey')
                              
                    axs[row , j].eventplot(all_events, color=colors, lineoffsets=y_positions, linelengths = 0.6)
                    
                    axs[row , j].set_xlim([xlim_left,xlim_right])
                    axs[row , j].set_title('Type: Long ISI, Sorted: First Lick, Aligned: ' + alignments[j])
        
                    # Add one empty plot for each color-label pair
                    for k in range(len(colorlist)):
                        axs[row , j].plot([], [], color=colorlist[k], label=labels[k])
        
                    # Add legend
                    axs[row , j].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
        
                    # Define y-ticks and labels for each sublist
                    yticks = [num_sublists_1 / 2, num_sublists_1 + num_sublists_2 / 2, 
                              num_sublists_1 + num_sublists_2 + num_sublists_3 / 2, 
                              num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4 / 2]
                    
                    yticklabels = ['Left', 'Left Opto', 'Right', 'Right Opto']
                    
                    # Set yticks and labels
                    axs[row , j].set_yticks(yticks)
                    axs[row , j].set_yticklabels(yticklabels)
                    
                    line_thickness = 0.5  # Adjust line thickness here
                    
                    # Draw partition lines to divide sublists
                    axs[row , j].axhline(y=num_sublists_1, color='grey', linestyle='--', linewidth=line_thickness)
                    axs[row , j].axhline(y=num_sublists_1 + num_sublists_2, color='grey', linestyle='--', linewidth=line_thickness)
                    axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + num_sublists_3, color='grey', linestyle='--', linewidth=line_thickness) 
        
                    # Create a twin y-axis
                    axs2 = axs[row , j].twinx() 
                    
                    # Set different tick marks (e.g., transformed scale)
                    axs2.set_ylim(axs[row , j].get_ylim())  # Keep same limits
                    axs2.set_yticks(y_positions)  # tick positions
                    axs2.set_yticklabels(y_ticks)  # tick labels
                    axs2.set_ylabel("Trials", color='black')
                    axs2.tick_params(axis='y', labelcolor='black',  size=2, labelsize=3)      
                # 'punish, short, ' + alignments[j]
                # trial ordered
                ################################################################################################################
    
    
                # 'punish, short, ' + alignments[j]
                # time-to-first-lick ordered
                ################################################################################################################
    
    
    
                if 0:
                    axs[2 , j].vlines(0 ,len(series_left_ps), 0, linestyle='--', color='grey')
                    axs[2 , j].eventplot(series_center_ps, color='black', linelengths = 0.3)
                    axs[2 , j].eventplot(series_right_ps, color='red', linelengths = 0.3)
                    axs[2 , j].eventplot(series_left_ps, color='limegreen', linelengths = 0.3)
                    axs[2 , j].set_xlim([xlim_left,xlim_right])
                    axs[2 , j].set_title('punish, short, ' + alignments[j])
                    # if len(series_center_ps) > 0:
                        
                    axs[2 , j].hist(np.concatenate(series_center_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black', alpha = 0.4)
                    axs[2 , j].hist(np.concatenate(series_right_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red', alpha = 0.4)
                    axs[2 , j].hist(np.concatenate(series_left_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'green', alpha = 0.4)
    
    
    
    
    
    
    
                    axs[3 , j].vlines(0 ,len(series_left_pl), 0, linestyle='--', color='grey')
                    axs[3 , j].eventplot(series_center_pl, color='black', linelengths = 0.3)
                    axs[3 , j].eventplot(series_right_pl, color='red', linelengths = 0.3)
                    axs[3 , j].eventplot(series_left_pl, color='limegreen', linelengths = 0.3)
                    axs[3 , j].set_xlim([xlim_left,xlim_right])
                    axs[3 , j].set_title('punish, long, ' + alignments[j])
                    # if len(series_center_pl) > 0:
                        
                    axs[3 , j].hist(np.concatenate(series_center_pl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black' , alpha = 0.2)
                    axs[3 , j].hist(np.concatenate(series_right_pl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red' , alpha = 0.2)
                    axs[3 , j].hist(np.concatenate(series_left_pl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen' , alpha = 0.2)
                    
                
                
                
                
                
                
                
                
                
                
    #             axs1[0 , j].vlines(0 ,len(series_left_rs), 0, linestyle='--', color='grey')
    #             axs1[0 , j].set_xlim([-2,6])
    #             axs1[0 , j].set_title('reward, short, ' + alignments[j])
    #             if len(series_center_rs) > 0:
    #                 axs1[0 , j].hist(np.concatenate(series_center_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black')
    #                 axs1[0 , j].hist(np.concatenate(series_right_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red')
    #                 axs1[0 , j].hist(np.concatenate(series_left_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen')
    
    #             axs1[1 , j].vlines(0 ,len(series_left_rl), 0, linestyle='--', color='grey')
    #             axs1[1 , j].set_xlim([-2,6])
    #             axs1[1 , j].set_title('reward, long, ' + alignments[j])
    #             if len(series_center_rl) > 0:
    #                 axs1[1 , j].hist(np.concatenate(series_center_rl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black')
    #                 axs1[1 , j].hist(np.concatenate(series_right_rl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red')
    #                 axs1[1 , j].hist(np.concatenate(series_left_rl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen')
    
    #             axs1[2 , j].vlines(0 ,len(series_left_ps), 0, linestyle='--', color='grey')
    #             axs1[2 , j].set_xlim([-2,6])
    #             axs1[2 , j].set_title('punish, short, ' + alignments[j])
    #             if len(series_center_ps) > 0:
    #                 axs1[2 , j].hist(np.concatenate(series_center_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black')
    #                 axs1[2 , j].hist(np.concatenate(series_right_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red')
    #                 axs1[2 , j].hist(np.concatenate(series_left_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen')
    
    
    #             axs1[3 , j].vlines(0 ,len(series_left_pl), 0, linestyle='--', color='grey')
    #             axs1[3 , j].set_xlim([-2,6])
    #             axs1[3 , j].set_title('punish, long, ' + alignments[j])
    #             if len(series_center_pl) > 0:
    #                 axs1[3 , j].hist(np.concatenate(series_center_pl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black')
    #                 axs1[3 , j].hist(np.concatenate(series_right_pl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red')
    #                 axs1[3 , j].hist(np.concatenate(series_left_pl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen')
    
    






        output_dir_onedrive, 
        output_dir_local

        output_pdf_dir =  output_dir_onedrive + subject + '/'
        output_pdf_pages_dir = output_dir_local + subject + '/lick_traces/lick_traces_' + session_date + '/'
        os.makedirs(output_pdf_dir, exist_ok = True)
        os.makedirs(output_pdf_pages_dir, exist_ok = True)
        output_pdf_filename = output_pdf_pages_dir + subject +  session_date + '_lick_traces' + str(i)
        pdf_paths.append(output_pdf_filename + '.pdf')
        save_image(output_pdf_filename)        
        plt.close(fig)
            
        
        output = PdfWriter()
        pdf_files = []
        for pdf_path in pdf_paths:            
            f = open(pdf_path, "rb")
            pdf_streams.append(PdfReader(f))
            pdf_files.append(f)

        for pdf_file_stream in pdf_streams:
            output.add_page(pdf_file_stream.pages[0])

        for pdf_file in pdf_files:
            pdf_file.close()


        outputStream = open(r'' + output_pdf_dir + subject + '_' + session_date + '_lick_traces' + '.pdf', "wb")
        output.write(outputStream)
        outputStream.close()
        




