# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 20:28:17 2025

@author: timst
"""
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
from pathlib import Path
import re
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from utils.util import get_figsize_from_pdf_spec
from scipy.stats import norm
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from matplotlib.collections import LineCollection


print_debug = 0

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

def get_decision_df(session_data, session_idx):
    """
    Process licks for a single session.
    
    Args:
        session_data (dict): Contains left/right lick times and opto flags for trials.
    
    Returns:
        dict: Processed licks categorized into left/right and opto/non-opto.
    """
    processed_dec = []
    
    
    raw_data = session_data['raw']
    
    numTrials = raw_data[session_idx]['nTrials']
    
    outcomes_time = session_data['outcomes_time']
    outcome_time = outcomes_time[session_idx]
        
    trial_types = session_data['trial_type'][session_idx]
    opto_flags = session_data['opto_trial'][session_idx]
    
    opto_encode = np.nan
    
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

        ###
  
        # np.array([x - alignment for x in licks['licks_left_start']])
        licks['licks_left_start'] = [(x - alignment)*1000 for x in licks['licks_left_start']]
        licks['licks_left_stop'] = [(x - alignment)*1000 for x in licks['licks_left_stop']]
        licks['licks_right_start'] = [(x - alignment)*1000 for x in licks['licks_right_start']]
        licks['licks_right_stop'] = [(x - alignment)*1000 for x in licks['licks_right_stop']]
  
        # check for licks or spout touches before choice window
        # if licks['licks_left_start'][0] < -0.1:
        #     licks['licks_left_start'] = [np.float64(np.nan)]
        # if licks['licks_left_stop'][0] < -0.1:
        #     licks['licks_left_stop'] = [np.float64(np.nan)]
        # if licks['licks_right_start'][0] < -0.1:
        #     licks['licks_right_start'] = [np.float64(np.nan)]
        # if licks['licks_right_stop'][0] < -0.1:
        #     licks['licks_right_stop'] = [np.float64(np.nan)]            
    
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
        
        if is_opto and not is_naive:
            opto_encode = 0
            
        isi = session_data['isi_post_emp'][session_idx][trial]
        
        move_correct_spout = session_data['move_correct_spout_flag'][session_idx][trial]
            
        processed_dec.append({
            "trial": trial,
            "trial_side": trial_type,
            "isi": isi,
            "is_opto": is_opto,
            "is_naive": is_naive,
            "rewarded": rewarded,
            "no_lick": no_lick,
            "opto_encode": opto_encode,
            "move_correct_spout": move_correct_spout,
            "licks_left_start": licks['licks_left_start'],
            "licks_left_stop": licks['licks_left_stop'],
            "licks_right_start": licks['licks_right_start'],
            "licks_right_stop": licks['licks_right_stop'],
            "valve_start": valve_times[0],
            "valve_stop": valve_times[1]
        })
        
        opto_encode = opto_encode + 1
        
    return pd.DataFrame(processed_dec)

def get_earliest_lick(row):
    left = row['licks_left_start']
    right = row['licks_right_start']

    all_licks = []

    if isinstance(left, list):
        all_licks.extend([v for v in left if not np.isnan(v)])
    if isinstance(right, list):
        all_licks.extend([v for v in right if not np.isnan(v)])

    return min(all_licks) if all_licks else np.nan   

def smooth_reward_curve(response_times, rewards, x_vals, bandwidth=0.2):
    smoothed = []
    for x in x_vals:
        weights = norm.pdf(response_times - x, scale=bandwidth)
        if weights.sum() > 0:
            smoothed.append(np.sum(weights * rewards) / np.sum(weights))
        else:
            smoothed.append(np.nan)  # Or interpolate later
    return np.array(smoothed)

def filter_df(processed_dec):
    # filter tags
    filtered_df = processed_dec[(processed_dec['is_naive'] == False)]
    filtered_df = filtered_df[(filtered_df['no_lick'] == False)]
    filtered_df = filtered_df[(filtered_df['move_correct_spout'] == False)]   
         
    
    return filtered_df

# def plot_variable_linewidth(x, y, weights, ax=None, label=None, color='blue', linewidth_range=(0.5, 10)):
#     if ax is None:
#         fig, ax = plt.subplots()

#     # Normalize weights to a range of line widths
#     w_norm = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-6)
#     linewidths = linewidth_range[0] + w_norm * (linewidth_range[1] - linewidth_range[0])

#     # Build line segments
#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)

#     lc = LineCollection(segments, linewidths=linewidths[:-1], color=color, label=label, alpha=0.9)
#     ax.add_collection(lc)
#     ax.set_xlim(x.min(), x.max())
#     ax.set_ylim(y.min(), y.max())
#     return ax

def plot_variable_linewidth(
    x, y, weights, ax=None, label=None, color='blue',
    linewidth_range=(0.3, 13),
    alpha_range=(0.001, 1.0)
):
    if ax is None:
        fig, ax = plt.subplots()

    # if not np.all(np.isfinite(weights)):
    #     raise ValueError("weights contains NaN or Inf")
    # Handle constant or invalid weights
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        raise ValueError("weights contain NaN or Inf")
    
#     w_min = np.min(weights)
#     w_max = np.max(weights)
#     if w_max == w_min:
#         w_norm = np.ones_like(weights)
#     else:
#         w_norm = (weights - w_min) / (w_max - w_min)

#     # # Normalize weights
#     # w_norm = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-6)

#     # Scale linewidth and alpha from normalized weights
#     linewidths = linewidth_range[0] + w_norm * (linewidth_range[1] - linewidth_range[0])
#     alphas = alpha_range[0] + w_norm * (alpha_range[1] - alpha_range[0])

# # alpha_min, alpha_max = 0.05, 1.0
# # alphas = alpha_min + w_norm * (alpha_max - alpha_min)

#     # Prepare segments
#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)

#     # Create one line per segment to allow varying alpha
#     for i, seg in enumerate(segments):
#         lc = LineCollection([seg], linewidths=[linewidths[i]], colors=[color], alpha=alphas[i])
#         ax.add_collection(lc)


#     ax.set_xlim(np.min(x), np.max(x))
#     ax.set_ylim(np.min(y), np.max(y))

    # Assuming weights is same length as x/y
    weights = weights[:-1]  # Now length N-1 to match segments

    # Normalize weights
    w_norm = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-6)
    w_norm = w_norm**2  # Or try **3 for more dramatic effect
    
    
    linewidths = linewidth_range[0] + w_norm * (linewidth_range[1] - linewidth_range[0])
    alphas = alpha_range[0] + w_norm * (alpha_range[1] - alpha_range[0])

    # Prepare segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Convert alpha into RGBA color array
    from matplotlib.colors import to_rgba
    base_rgba = np.array(to_rgba(color))
    colors = np.tile(base_rgba, (len(segments), 1))
    colors[:, -1] = alphas  # set alpha channel

    # One LineCollection for all segments
    lc = LineCollection(segments, linewidths=linewidths, colors=colors)
    # Create a proxy artist with full alpha for the legend
    # legend_handle = Line2D([0], [0], color=color, linewidth=np.mean(linewidths), label=label, alpha=1.0)
    ax.add_collection(lc)
    # ax.legend(handles=[legend_handle])
    
    # # Append legend proxy with full alpha
    if label is not None:        
        # Step 1: Get existing legend entries
        # handles, labels = ax.get_legend_handles_labels()
        current_handles = []
        current_labels = []
        legend = ax.get_legend()
        if legend:
            current_handles = legend.legend_handles
            current_labels = [text.get_text() for text in legend.get_texts()]        
        
        # Step 2: Create your custom handle
        legend_handle = Line2D([0], [0], color=color, linewidth=1.0, label=label, alpha=1.0)
        
        # Step 3: Append if label is not already present (optional)
        if label not in current_labels:
            current_handles.append(legend_handle)
            current_labels.append(label)
        
        # Step 4: Update the legend on ax
        ax.legend(current_handles, current_labels)
    
        legend = ax.get_legend()
        if legend:
            current_handles = legend.legend_handles
            current_labels = [text.get_text() for text in legend.get_texts()]          
    
        for handle in current_handles:
            print(f"Label: {handle.get_label()}, Color: {handle.get_color()}, Linewidth: {handle.get_linewidth()}, Alpha: {handle.get_alpha()}")     
    

    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    return ax


def plot_decision_time_kernel_opto_diff(M, config, subjectIdx, sessionIdx=-1, figure_id=None, show_plot=1, opto=False, side='both'):
    # figure meta
    rowspan, colspan = 2, 2
    fig_size = get_figsize_from_pdf_spec(rowspan, colspan, config['pdf_spec']['pdf_pg_opto_psychometric'])    
    fig, ax = plt.subplots(figsize=fig_size) 

    subject = config['list_config'][subjectIdx]['subject_name']

    is_avg = 0
    if sessionIdx != -1:
        decision_df = get_decision_df(M, sessionIdx)
        dates = M['dates'][sessionIdx]
        if not isinstance(dates, list):
            dates = [dates]
        date_str = dates[0]
    else:
        is_avg = 1
        dates = M['dates']
        combined_df = pd.DataFrame()
        for sessionIdx in range(0, len(dates)):
            decision_df = get_decision_df(M, sessionIdx)
            combined_df = pd.concat([combined_df, decision_df], ignore_index=True)
        decision_df = combined_df
        if not isinstance(dates, list):
            dates = [dates]
        date_str = f"{dates[0]} - {dates[-1]}"
    
    num_residuals = 5
    
    filtered_df = filter_df(decision_df)
    filtered_df['earliest_lick'] = filtered_df.apply(get_earliest_lick, axis=1) 
    
    # earliest_lick = np.nanmin(filtered_df[['licks_left_start', 'licks_right_start']].values, axis=1)
    # filtered_df['earliest_lick'] = earliest_lick
    
    # filtered_df['earliest_lick'] = filtered_df[['licks_left_start', 'licks_right_start']].min(axis=1, skipna=True)
    
    control_df = filtered_df[(filtered_df['is_opto'] == False)]
    
    if opto:
        opto_df = filtered_df[(filtered_df['is_opto'] == True)]
    
    max_time = 1000
    incr_t = 0.1  # change as needed      
    t_vals = np.arange(0, max_time + incr_t, incr_t)
    
    def response_curve(df, t_vals, bandwidth=20, bw_method=0.025):
        response_times = df['earliest_lick']
        rewards = df['rewarded']
                
        response_curve = smooth_reward_curve(response_times, rewards, t_vals, bandwidth=20)   

        # # response_times: 1D array of trial response times
        # kde = gaussian_kde(response_times, bw_method=0.025)  # Adjust bw_method as needed
        
        # # x_vals = np.linspace(min(response_times), max(response_times), 200)
        # density = kde(t_vals)
        
        # # Probability density function. Normalize to percentage if you want
        # trials_percent_curve = density / density.sum()  # Area under curve sums to 1    
        
        # # Approximate percentage of trials per response time
        # bin_width = t_vals[1] - t_vals[0]  # Or use your desired smoothing resolution
        # percent_density = kde(t_vals) * len(response_times) * bin_width * 100        
                     
        # kde = gaussian_kde(response_times, bw_method=bw_method)
        # density = kde(t_vals)
        
        # bin_width = t_vals[1] - t_vals[0]
        # percent_density = density * bin_width * 100  # Percentage of trials in each bin

        # counts, _ = np.histogram(response_times, bins=t_vals)
        # percent_per_bin = counts / counts.sum() * 100  
        # percent_density = percent_per_bin
        
        # counts, bin_edges = np.histogram(response_times, bins=t_vals)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # percent_per_bin = counts / counts.sum() * 100   
        
        # KDE estimate over response times
        kde = gaussian_kde(response_times, bw_method=bw_method)
        density = kde(t_vals)  # Unit: 1/ms
    
        # Scale density to match total number of trials (area under curve ≈ len(response_times))
        bin_width = t_vals[1] - t_vals[0]
        smoothed_counts = density * len(response_times) * bin_width  # Trials per bin        
        
        percent_density = smoothed_counts
        # percent_density.append()
        
        return response_curve, percent_density
    
    control_response_curve, control_percent_density = response_curve(control_df, t_vals, bw_method=0.25)
    
    if side == 'both':
        control_response_curve_left, control_percent_density_left = response_curve(control_df[(control_df['trial_side'] == 'left')], t_vals, bw_method=1.0)
        control_response_curve_right, control_percent_density_right = response_curve(control_df[(control_df['trial_side'] == 'right')], t_vals, bw_method=1.0)
    elif side == 'left':
        control_response_curve_left, control_percent_density_left = response_curve(control_df[(control_df['trial_side'] == 'left')], t_vals, bw_method=1.0)
    elif side == 'right':
        control_response_curve_right, control_percent_density_right = response_curve(control_df[(control_df['trial_side'] == 'right')], t_vals, bw_method=1.0)
    
    if opto:
        opto_response_curve, opto_percent_density = response_curve(opto_df, t_vals, bw_method=1.0)        
        opto_response_effect = opto_response_curve - control_response_curve
        if side == 'both':
            opto_response_curve_left, opto_percent_density_left = response_curve(opto_df[(opto_df['trial_side'] == 'left')], t_vals, bw_method=1.0)
            opto_response_curve_right, opto_percent_density_right = response_curve(opto_df[(opto_df['trial_side'] == 'right')], t_vals, bw_method=1.0)                
        elif side == 'left':
            opto_response_curve_left, opto_percent_density_left = response_curve(opto_df[(opto_df['trial_side'] == 'left')], t_vals, bw_method=1.0)
            opto_response_effect_left = opto_response_curve_left - control_response_curve_left
        elif side == 'right':
            opto_response_curve_right, opto_percent_density_right = response_curve(opto_df[(opto_df['trial_side'] == 'right')], t_vals, bw_method=1.0)    
            opto_response_effect_right = opto_response_curve_right - control_response_curve_right

    
    

    # response_times = control_df['earliest_lick']
    # rewards = control_df['rewarded']
    
    # max_time = 1000
    # incr_t = 0.1  # change as needed      
    # t_vals = np.arange(0, max_time + incr_t, incr_t)
    
    # control_response_curve = smooth_reward_curve(response_times, rewards, t_vals, bandwidth=20)
    
    # # response_times: 1D array of trial response times
    # kde = gaussian_kde(response_times, bw_method=0.025)  # Adjust bw_method as needed
    
    # # x_vals = np.linspace(min(response_times), max(response_times), 200)
    # density = kde(t_vals)
    
    # # Probability density function. Normalize to percentage if you want
    # trials_percent_curve = density / density.sum()  # Area under curve sums to 1    
    
    # # Approximate percentage of trials per response time
    # bin_width = t_vals[1] - t_vals[0]  # Or use your desired smoothing resolution
    # percent_density = kde(t_vals) * len(response_times) * bin_width * 100
    
    # lowess = sm.nonparametric.lowess
    # smoothed = lowess(rewards, response_times, frac=0.2)  # frac controls smoothness    
    
    # Create a second axis on the right side with a different scale
    # ax2 = ax.twinx()
    # ax2.set_ylabel('Percentage of Trials per Bin')
    # ax.set_ylim([0.00, 1.00])
    
    # if opto:
    #     ax2.plot(
    #         t_vals,
    #         control_percent_density,
    #         color='gray',
    #         marker='.',
    #         label='control % trials',
    #         markersize=2,
    #         linewidth=1)  
    #     ax2.plot(
    #         t_vals,
    #         opto_percent_density,
    #         color='lightgray',
    #         linestyle='--', 
    #         marker='o',
    #         label='opto % trials',
    #         markersize=2,
    #         linewidth=1)        
    # else:
    #     ax2.plot(
    #         t_vals,
    #         control_percent_density,
    #         color='gray',
    #         marker='.',
    #         label='control % trials',
    #         markersize=2,
    #         linewidth=1)    
    
    # ax2.legend(loc='upper right', ncol=1, bbox_to_anchor=(1, 1))    
    
    # ax.plot(
    #     t_vals,
    #     control_response_curve,
    #     color='black',
    #     marker='.',
    #     label='Control',
    #     markersize=2)
    
    # plot_variable_linewidth(t_vals, control_response_curve, control_percent_density, ax, color='black', label='Control')    

    
    # if side == 'both':
    #     # ax.plot(
    #     #     t_vals,
    #     #     control_response_curve_left,
    #     #     color='blue',
    #     #     marker='.',
    #     #     label='Control Left',
    #     #     markersize=2)    
    #     # ax.plot(
    #     #     t_vals,
    #     #     control_response_curve_right,
    #     #     color='red',
    #     #     marker='.',
    #     #     label='Control Right',
    #     #     markersize=2)
        
    #     # plot_variable_linewidth(t_vals, control_response_curve_left, control_percent_density_left, ax, color='blue', label='Control Left')            
    #     # plot_variable_linewidth(t_vals, control_response_curve_right, control_percent_density_right, ax, color='red', label='Control Right')      
    #     # plot_variable_linewidth(t_vals, opto_response_effect, control_percent_density, ax, color='violet', label='Opto - Control')
        
    #     print('')
        
    # elif side == 'left':
    #     # ax.plot(
    #     #     t_vals,
    #     #     control_response_curve_left,
    #     #     color='blue',
    #     #     marker='.',
    #     #     label='Control Left',
    #     #     markersize=2)    
        
    #     plot_variable_linewidth(t_vals, control_response_curve_left, control_percent_density_left, ax, color='blue', label='Contro Leftl')           
    # elif side == 'right':
    #     # ax.plot(
    #     #     t_vals,
    #     #     control_response_curve_right,
    #     #     color='red',
    #     #     marker='.',
    #     #     label='Control Right',
    #     #     markersize=2)    
    #     plot_variable_linewidth(t_vals, control_response_curve_right, control_percent_density_right, ax, color='red', label='Control Right')    

    # Step 3: Create your manual entries (e.g., outcome annotation)
    # if opto and side == None:        
    #     extra = [
    #         mpatches.Patch(color='gray', label='Control - Trials %'),
    #         mpatches.Patch(color='lightgray', label='Opto - Trials %')
    #     ]    
    # else:
    #     extra = [
    #         mpatches.Patch(color='gray', label='Control - Trials %'),
    #     ]            
    # extra = []
                    
    if opto:
        # ax.plot(
        #     t_vals,
        #     opto_response_curve,
        #     color='violet',
        #     marker='.',
        #     label='Opto',
        #     markersize=2)    
        
        
        if side == 'both':  
            # ax.plot(
            #     t_vals,
            #     opto_response_curve_left,
            #     color='lightblue',
            #     marker='.',
            #     label='Opto Left',
            #     markersize=2)        
            # ax.plot(
            #     t_vals,
            #     opto_response_curve_right,
            #     color='lightcoral',
            #     marker='.',
            #     label='Opto Right',
            #     markersize=2)    
            
            # plot_variable_linewidth(t_vals, opto_response_curve_left, opto_percent_density_left, ax, color='lightblue', label='Opto Left')     
            # plot_variable_linewidth(t_vals, opto_response_curve_right, opto_percent_density_right, ax, color='lightcoral', label='Opto Right')                 
            
            plot_variable_linewidth(t_vals, opto_response_effect, control_percent_density, ax, label='Opto - Control', color='violet')  
            # extra.append(mpatches.Patch(color='violet', label='Opto - Control'))
             
            
        elif side == 'left':         
            # ax.plot(
            #     t_vals,
            #     opto_response_curve_left,
            #     color='lightblue',
            #     marker='.',
            #     label='Opto Left',
            #     markersize=2)      
            
            # plot_variable_linewidth(t_vals, opto_response_curve_left, opto_percent_density_left, ax, color='lightblue', label='Opto Left')   
            plot_variable_linewidth(t_vals, opto_response_effect_left, control_percent_density, ax, label='Opto - Control (Left)', color='lightblue')
            
            # extra.append(mpatches.Patch(color='lightblue', label='Opto - Control (Left)'))
            
        elif side == 'right':            
            # ax.plot(
            #     t_vals,
            #     opto_response_curve_right,
            #     color='lightcoral',
            #     marker='.',
            #     label='Opto Right',
            #     markersize=2)    
            # plot_variable_linewidth(t_vals, opto_response_curve_right, opto_percent_density_right, ax, color='lightcoral', label='Opto Right')                             
            plot_variable_linewidth(t_vals, opto_response_effect_right, control_percent_density, ax, label='Opto - Control (Right)', color='lightcoral')
            
            # extra.append(mpatches.Patch(color='lightcoral', label='Opto - Control (Right)'))
        elif side == None:
            plot_variable_linewidth(t_vals, opto_response_curve, opto_percent_density, ax, label='Opto', color='violet')    
            # extra.append(mpatches.Patch(color='violet', label='Opto'))
    
    ax.hlines(
        [-0.25, 0.00, 0.25], 0.0, max_time,
        linestyle=':', color='grey')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([0, 650])
    ax.set_ylim([-0.50, 0.5])
    ax.set_xlabel('Decision Time (Since Choice Window Onset) / ms')
    ax.set_ylabel('Correct Prob. (Opto - Control)')
    ax.set_xticks(np.arange(0, 1000, 100))
    ax.tick_params(axis='x', rotation=45)
    # ax.set_yticks([-0.25, 0.00, 0.25, 1])
    ax.set_yticks([-0.50, -0.25, 0.00, 0.25, 0.50])
    

    # ax.legend(loc='upper right', ncol=1, bbox_to_anchor=(1, 1))
    ax.set_title('Average Decision Time Curve - Opto Effect ' + date_str)
    
    # # Step 2: Get existing handles and labels
    # handles, labels = ax.get_legend_handles_labels()
    
    # # Step 3: Create your manual entries (e.g., outcome annotation)
    # # if opto:        
    # #     extra = [
    # #         mpatches.Patch(color='gray', label='Control - Trials %'),
    # #         mpatches.Patch(color='lightgray', label='Opto - Trials %')
    # #     ]    
    # # else:
    # #     extra = [
    # #         mpatches.Patch(color='gray', label='Control - Trials %'),
    # #     ]            
    # extra = []
    
    # # Step 4: Combine and apply
    # ax.legend(handles + extra, labels + [e.get_label() for e in extra], loc='upper right')
       
    handles1 = []
    labels1 = []
    legend = ax.get_legend()
    if legend:
        handles1 = legend.legend_handles
        labels1 = [text.get_text() for text in legend.get_texts()]  
    
    # handles1, labels1 = ax.get_legend_handles_labels()
    
    for handle in handles1:
        print('ax1')
        print(f"Label: {handle.get_label()}, Color: {handle.get_color()}, Linewidth: {handle.get_linewidth()}, Alpha: {handle.get_alpha()}") 
        
    # Add legend to one of the axes (or both if needed)
    ax.legend(handles1, labels1, loc='upper right')    
    
    if show_plot:
        plt.show()
        
    subject = config['list_config'][subjectIdx]['subject_name']
    output_dir = os.path.join(config['paths']['figure_dir_local'] + subject)
    figure_id = f"{subject}_decision_time_curve_opto_effect"
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
  

