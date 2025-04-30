# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 21:30:10 2025

@author: timst
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import pandas as pd
from types import SimpleNamespace
import os
from utils.util import get_figsize_from_pdf_spec

# region_map = {
#     0: 'Control',
#     1: 'RLat',
#     2: 'LLat',
#     3: 'RIntA',
#     4: 'LIntA',
#     5: 'LPPC',
#     6: 'RPPC',
#     7: 'mPFC',
#     8: 'LPost',
#     9: 'RPost'
# }

region_map = {
    0: {'name': 'Control','location': 'center'},
    1: {'name': 'RLat',   'location': 'right'},
    2: {'name': 'LLat',   'location': 'left'},
    3: {'name': 'RIntA',  'location': 'right'},
    4: {'name': 'LIntA',  'location': 'left'},
    5: {'name': 'LPPC',   'location': 'left'},
    6: {'name': 'RPPC',   'location': 'right'},
    7: {'name': 'mPFC',   'location': 'center'},
    8: {'name': 'LPost',  'location': 'left'},
    9: {'name': 'RPost',  'location': 'right'},
}

left_regions = ['LIntA', 'LLat', 'LPPC', 'LPost']
right_regions = ['RIntA', 'RLat', 'RPPC', 'RPost']
center_regions = ['mPFC']

print_debug = 0
# bin the data with timestamps.

def get_bin_stat(decision, session_settings, isi='post'):
    bin_size=100
    least_trials=1
    # set bins across isi range
    # short ISI: [50, 400, 750]ms.  associated with left lick
    # long ISI: [750, 1100, 1450]ms.  associated with right lick
    isi_long_mean = session_settings['ISILongMean_s'] * 1000
    bin_right = isi_long_mean + 400
    bins = np.arange(0, bin_right + bin_size, bin_size)
    bins = bins - bin_size / 2
    if isi=='pre':
        row = 4
    if isi=='post':
        row = 5
    bin_indices = np.digitize(decision[row,:], bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        direction = decision[1, bin_indices == i].copy()
        m = np.mean(direction) if len(direction) > least_trials else np.nan
        s = sem(direction) if len(direction) > least_trials else np.nan
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem  = np.array(bin_sem)
    bin_isi  = bins[:-1] + (bins[1]-bins[0]) / 2
    non_nan  = (1-np.isnan(bin_mean)).astype('bool')
    bin_mean = bin_mean[non_nan]
    bin_sem  = bin_sem[non_nan]
    bin_isi  = bin_isi[non_nan]
    return bin_mean, bin_sem, bin_isi


def separate_fix_jitter(decision):
    decision_fix = decision[:,decision[3,:]==0]
    decision_jitter = decision[:,decision[3,:]==1]
    decision_chemo = decision[:,decision[3,:]==2]
    decision_opto = decision[:,decision[3,:]==3]
    decision_opto_left = decision[:,decision[6,:]==1]
    decision_opto_right = decision[:,decision[6,:]==2]
    return decision_fix, decision_jitter, decision_chemo, decision_opto, decision_opto_left, decision_opto_right

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
        licks['licks_left_start'] = [x - alignment for x in licks['licks_left_start']]
        licks['licks_left_stop'] = [x - alignment for x in licks['licks_left_stop']]
        licks['licks_right_start'] = [x - alignment for x in licks['licks_right_start']]
        licks['licks_right_stop'] = [x - alignment for x in licks['licks_right_stop']]
  
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

def bin_stats(filtered_df):
    # Define bin edges
    max_isi = filtered_df['isi'].max() + 200
    print(filtered_df['isi'].max())
    incr_isi = 20  # change as needed
      
    
    bins = np.arange(0, max_isi + incr_isi, incr_isi)
    bin_labels = bins[:-1]  # label bins by left edge
    bin_centers = bins[:-1] + incr_isi / 2
    bin_indices = np.arange(0, len(bins))
    
    means = [np.nan] * len(bins)
    sems = [np.nan] * len(bins)

    reward_stats = (
        filtered_df
        .groupby(['bin_idx', 'trial_side'])
        .agg(
            percent_reward=('rewarded', lambda g: (g == True).mean()),
            sem_reward=('rewarded', lambda g: g.std(ddof=1) / np.sqrt(len(g)))
        )
        .reset_index()
    )        
            
    # convert rewarded percent on left to probability of licking right spout
    reward_stats.loc[reward_stats['trial_side'] == 'left', 'percent_reward'] = \
        1 - reward_stats.loc[reward_stats['trial_side'] == 'left', 'percent_reward']
        
    for _, row in reward_stats.iterrows():
        means[int(row['bin_idx'])] = row['percent_reward']            
        sems[int(row['bin_idx'])] = row['sem_reward']
        
    means = np.asarray(means)
    sems = np.asarray(sems)
    isi = np.asarray(bins)
    
    nan_indices = np.where(np.isnan(means))[0]
    means = np.delete(means, nan_indices)
    sems = np.delete(sems, nan_indices)
    isis = np.delete(isi, nan_indices)
    
    # opto_residual_means.append(means)            
    # opto_residual_sems.append(sems)
    # opto_residual_isis.append(isi)
       
    return means, sems, isis

# # DataFrame Accessor class
# @pd.api.extensions.register_dataframe_accessor("opto")
# class OptoTools:
#     def __init__(self, df):
#         self._df = df

#     def assign_bins(self, bin_col='isi', bin_size=0.1, max_val=None):
#         max_val = max_val or self._df[bin_col].max()
#         bins = np.arange(0, max_val + bin_size, bin_size)
#         bin_indices = np.digitize(self._df[bin_col], bins) - 1
#         self._df['bin_idx'] = bin_indices
#         self._df['bin_edge'] = bins[bin_indices]
#         return self._df

#     def reward_stats(self):
#         grouped = self._df.groupby(['bin_idx', 'trial_side'])
#         result = grouped.apply(lambda g: pd.Series({
#             'percent_reward': (g['rewarded'] == True).mean() * 100,
#             'sem_reward': (g['rewarded'] == True).std(ddof=1) / np.sqrt(len(g))
#         }))
#         return result.reset_index()


def filter_df(processed_dec):
    # filter tags
    filtered_df = processed_dec[(processed_dec['is_naive'] == False)]
    filtered_df = filtered_df[(filtered_df['no_lick'] == False)]
    filtered_df = filtered_df[(filtered_df['move_correct_spout'] == False)]   
         
    
    return filtered_df

def bin_control(processed_dec):
    filtered_df = filter_df(processed_dec)
    
    # get control trials
    control_df = filtered_df[(filtered_df['is_opto'] == False)]
    # # get opto trials
    # opto_trials = filtered_df[(filtered_df['is_opto'] == True)]
    
    # Define bin edges
    max_isi = filtered_df['isi'].max() + 200
    incr_isi = 20  # change as needed
      
    bins = np.arange(0, max_isi + incr_isi, incr_isi)
    bin_labels = bins[:-1]  # label bins by left edge
    bin_centers = bins[:-1] + incr_isi / 2
    bin_indices = np.arange(0, len(bins))
    
    # Assign each row to a bin
    control_df['bin_idx'] = np.digitize(control_df['isi'], bins) - 1    
    
    control_means = []
    control_sems = []
    control_isis = []
    
    means, sems, isis = bin_stats(control_df)
    
    control_means.append(means)            
    control_sems.append(sems)
    control_isis.append(isis)    
    
    return control_means, control_sems, control_isis

def bin_opto_residuals(processed_dec, num_residuals):
    filtered_df = filter_df(processed_dec)
    
    max_val = filtered_df['opto_encode'].max()
    if pd.notna(max_val):
        max_post_opto_encode = int(max_val)
    else:
        max_post_opto_encode = 0  # or another default value    
    
    # # get max number of post opto trials
    # max_post_opto_encode = int(filtered_df['opto_encode'].max())




    # Define bin edges
    max_isi = filtered_df['isi'].max() + 200
    incr_isi = 20  # change as needed
      
    bins = np.arange(0, max_isi + incr_isi, incr_isi)
    bin_labels = bins[:-1]  # label bins by left edge
    bin_centers = bins[:-1] + incr_isi / 2
    bin_indices = np.arange(0, len(bins))
    
    # Assign each row to a bin
    filtered_df['bin_idx'] = np.digitize(filtered_df['isi'], bins) - 1

    opto_residuals = []
    for post_op_idx in range(0, max_post_opto_encode + 1):
        if print_debug == 1:
            print(f"Extracting post opto trials {post_op_idx}")
        trials_at_idx = filtered_df[(filtered_df['opto_encode'] == post_op_idx)]
        opto_residuals.append(trials_at_idx)
    
    opto_residual_means = []
    opto_residual_sems = []
    opto_residual_isis = []
    
    # for opto_residual in opto_residuals:
    for idx in range(0, num_residuals):        
        # means = [np.nan] * len(bins)
        # sems = [np.nan] * len(bins)
 
        # reward_stats = (
        #     opto_residual
        #     .groupby(['bin_idx', 'trial_side'])
        #     .agg(
        #         percent_reward=('rewarded', lambda g: (g == True).mean()),
        #         sem_reward=('rewarded', lambda g: g.std(ddof=1) / np.sqrt(len(g)))
        #     )
        #     .reset_index()
        # )        
                
        # # convert rewarded percent on left to probability of licking right spout
        # reward_stats.loc[reward_stats['trial_side'] == 'left', 'percent_reward'] = \
        #     1 - reward_stats.loc[reward_stats['trial_side'] == 'left', 'percent_reward']
            
        # for _, row in reward_stats.iterrows():
        #     means[int(row['bin_idx'])] = row['percent_reward']            
        #     sems[int(row['bin_idx'])] = row['sem_reward']
            
        # means = np.asarray(means)
        # sems = np.asarray(sems)
        # isi = np.asarray(bins)
        
        # nan_indices = np.where(np.isnan(means))[0]
        # means = np.delete(means, nan_indices)
        # sems = np.delete(sems, nan_indices)
        # isi = np.delete(isi, nan_indices)
        
        # opto_residual_list = opto_residuals[0:idx+1]   # combine opto + trials for residual
        opto_residual_list = opto_residuals[idx]   # keep each index separate
        opto_residual_list = opto_residual_list if isinstance(opto_residual_list, list) else [opto_residual_list]
        opto_residual = pd.concat([x for x in opto_residual_list], ignore_index=True)
        means, sems, isis = bin_stats(opto_residual)
        
        opto_residual_means.append(means)            
        opto_residual_sems.append(sems)
        opto_residual_isis.append(isis)
        
        
    return opto_residual_means, opto_residual_sems, opto_residual_isis

# def clear_nan(array):    
#     return array[~np.isnan(array)]

def get_decision(subject_session_data, session_num):
    decision = subject_session_data['decision'][session_num]
    # decision = [np.concatenate(d, axis=1) for d in decision]
    decision = np.concatenate(decision, axis=1)
    jitter_flag = subject_session_data['jitter_flag'][session_num]
    # jitter_flag = np.concatenate(jitter_flag).reshape(1,-1)
    jitter_flag = np.array(jitter_flag).reshape(1,-1)
    # opto_flag = subject_session_data['opto_flag']
    opto_flag = subject_session_data['opto_trial'][session_num]
    opto_flag = np.array(opto_flag).reshape(1,-1)
    jitter_flag[0 , :] = jitter_flag[0 , :] + opto_flag[0 , :]*3
    # jitter_flag = jitter_flag + opto_flag*3
    # jitter_flag = [j + o * 3 for j, o in zip(jitter_flag, opto_flag)]
    opto_side = subject_session_data['opto_side'][session_num]
    opto_side = np.array(opto_side).reshape(1,-1)
    outcomes = subject_session_data['outcomes'][session_num]
    all_trials = 0
    chemo_labels = subject_session_data['Chemo'][session_num]
    # for j in range(len(chemo_labels)):
    #     if chemo_labels[j] == 1:
    #         jitter_flag[0 , all_trials:all_trials+len(outcomes[j])] = 2*np.ones(len(outcomes[j]))
    #     all_trials += len(outcomes[j])
    isi_pre_emp = subject_session_data['isi_pre_emp'][session_num]
    # isi_pre_emp = np.concatenate(isi_pre_emp).reshape(1,-1)
    isi_pre_emp = np.array(isi_pre_emp).reshape(1,-1)
    
    isi_post_emp = subject_session_data['isi_post_emp'][session_num]
    isi_post_emp = np.array(isi_post_emp).reshape(1,-1)
    # isi_post_emp = np.concatenate(isi_post_emp).reshape(1,-1)
    decision = np.concatenate([decision, jitter_flag, isi_pre_emp, isi_post_emp, opto_side], axis=0)
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    decision = decision[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: jitter flag.
    # row 4: pre pert isi.
    # row 5: post pert isi.
    # row 6: opto side
    
    decision_fix, decision_jitter, decision_chemo, decision_opto, decision_opto_left, decision_opto_right = separate_fix_jitter(decision)
    return decision_fix, decision_jitter, decision_chemo, decision_opto, decision_opto_left, decision_opto_right


def plot_psychometric_opto_epoch_residual(M, config, subjectIdx, sessionIdx, figure_id=None, show_plot=1):

    # figure meta
    rowspan, colspan = 2, 2
    fig_size = get_figsize_from_pdf_spec(rowspan, colspan, config['pdf_spec']['pdf_pg_opto_psychometric'])    
    fig, ax = plt.subplots(figsize=fig_size) 

    subject = config['list_config'][subjectIdx]['subject_name']
    
    OptoRegionIdx = M['raw'][sessionIdx]['TrialSettings'][0]['GUI']['OptoRegion']
    OptoRegion = region_map[OptoRegionIdx]['name']
    OptoSide = region_map[OptoRegionIdx]['location']
    
    # OptoPwrs = M['raw'][sessionIdx]['TrialSettings'][0]['GUI']['OptoRegion']
    
    # OptoSide = M['opto_side'][sessionIdx][0]   
    
    # OptoSide = M['opto_side'][sessionIdx][0]        

    dates = M['dates'][sessionIdx]
    print(dates)
    
    session_settings = M['session_settings'][sessionIdx]
    isi_short_mean = session_settings['ISIShortMean_s'] * 1000
    isi_long_mean = session_settings['ISILongMean_s'] * 1000
    # isi_orig = session_settings['ISIOrig_s'] * 1000
    isi_orig = (isi_short_mean + isi_long_mean) / 2
    
    is_avg = 0
    if sessionIdx != -1:
        decision_df = get_decision_df(M, sessionIdx)
        
    else:
        is_avg = 1
        combined_df = pd.DataFrame()
        for sessionIdx in range(0, len(dates)):
            decision_df = get_decision_df(M, sessionIdx)
            combined_df = pd.concat([combined_df, decision_df], ignore_index=True)
        decision_df = combined_df
    
    num_residuals = 5
    
    
    
    
    opto_residual_means, opto_residual_sems, opto_residual_isis = bin_opto_residuals(decision_df, num_residuals)
    
    control_means, control_sems, control_isis = bin_control(decision_df)
    
    # decision_fix, decision_jitter, decision_chemo, decision_opto, decision_opto_left, decision_opto_right = get_decision(subject_session_data_copy, session_num)
    # bin_mean_fix, bin_sem_fix, bin_isi_fix = get_bin_stat(decision_fix, session_settings)
    # bin_mean_jitter, bin_sem_jitter, bin_isi_jitter = get_bin_stat(decision_jitter, session_settings)
    # bin_mean_chemo, bin_sem_chemo, bin_isi_chemo = get_bin_stat(decision_chemo, session_settings)
    # bin_mean_opto, bin_sem_opto, bin_isi_opto = get_bin_stat(decision_opto, session_settings)
    
    # bin_mean_opto_left, bin_sem_opto_left, bin_isi_opto_left = get_bin_stat(decision_opto_left, session_settings)
    # bin_mean_opto_right, bin_sem_opto_right, bin_isi_opto_right = get_bin_stat(decision_opto_right, session_settings)
    
    
    left_label = 'Opto ' + OptoRegion
    right_label = 'Opto ' + OptoRegion
    
    def generate_purple_colors(n, min_red=100, max_red=230, min_blue=150, max_blue=255, shift=0):
        """Generate n purple shades with optional brightness shift."""
        red_vals = np.linspace(max_red, min_red, n).astype(int) - shift
        blue_vals = np.linspace(max_blue, min_blue, n).astype(int) - shift
        
        # Clip to [0, 255] to prevent RGB overflow
        red_vals = np.clip(red_vals, 0, 255)
        blue_vals = np.clip(blue_vals, 0, 255)
    
        purples = [np.array([r, 0, b]) / 255 for r, b in zip(red_vals, blue_vals)]
        return purples    
    
    # Function to generate 'n' green colors, from light to dark
    def generate_green_colors(n, min_green=120, max_green=230, shift=0):
        greens = []
        for i in range(n):
            # # The green component ranges from 255 (light) to 0 (dark)
            # green_value = int(255 * ( i / (n - 1)))  # Linear interpolation
            # # greens.append((0, green_value, 0))  # RGB format: (Red, Green, Blue)
            # greens.append(np.array([0, green_value, 0]) / 255)  # Normalize to [0, 1]
            
            """Generate n green shades between min_green and max_green intensity."""
            green_vals = np.linspace(max_green, min_green, n).astype(int)  # light to dark
            green_vals = green_vals - shift
            greens = [np.array([0, g, 0]) / 255 for g in green_vals]            
            
        return greens
    
    def generate_blue_colors(n, min_blue=100, max_blue=255, shift=0):
        """Generate n blue shades between min_blue and max_blue, with optional brightness shift."""
        blue_vals = np.linspace(max_blue, min_blue, n).astype(int) - shift
        blue_vals = np.clip(blue_vals, 0, 255)
    
        blues = [np.array([0, 0, b]) / 255 for b in blue_vals]
        return blues       
    
    # def generate_blue_colors(n):
    #     blues = []
    #     for i in range(n):
    #         # The green component ranges from 255 (light) to 0 (dark)
    #         blue_value = int(255 * ( i / (n - 1)))  # Linear interpolation
    #         # greens.append((0, green_value, 0))  # RGB format: (Red, Green, Blue)
    #         blues.append(np.array([blue_value, 0, 0]) / 255)  # Normalize to [0, 1]
    #     return blues    
    
 
    
    mean_greens = generate_green_colors(num_residuals)
    sem_greens = generate_green_colors(num_residuals, shift=100)
    
    mean_blues = generate_blue_colors(num_residuals)
    sem_blues = generate_blue_colors(num_residuals, shift=100)    
    
    mean_purples = generate_purple_colors(num_residuals)
    sem_purples = generate_purple_colors(num_residuals, shift=100)      
    
    def generate_opto_labels(num):
        labels = []
        labels.append('Opto ' + OptoRegion)
        for i in range(0, num):        
            if i != 0:
                labels.append('Opto + ' + str(i))
        return labels
    
    labels = generate_opto_labels(num_residuals)
    
    if OptoRegion in left_regions:
        mean_color = mean_blues
        sem_color = sem_blues
    elif OptoRegion in right_regions:
        mean_color = mean_greens
        sem_color = sem_greens
    else:
        mean_color = mean_purples
        sem_color = sem_purples    
    
    # opto_side
    
    # plot opto residuals
    for idx in range(0, num_residuals):
        # fix
        
        # ax.plot(
        #     opto_residual_isis[idx],
        #     opto_residual_means[idx],
        #     color=mean_greens[idx], marker='.', label=labels[idx], markersize=4)
        # ax.fill_between(
        #     opto_residual_isis[idx],
        #     opto_residual_means[idx] - opto_residual_sems[idx],
        #     opto_residual_means[idx] + opto_residual_sems[idx],
        #     color='honeydew', alpha=0.2) 
        #     # color=sem_greens[idx], alpha=0.2)    
        ax.plot(
            opto_residual_isis[idx],
            opto_residual_means[idx],
            color=mean_color[idx], marker='.', label=labels[idx], markersize=4)
        # ax.fill_between(
        #     opto_residual_isis[idx],
        #     opto_residual_means[idx] - opto_residual_sems[idx],
        #     opto_residual_means[idx] + opto_residual_sems[idx],
            # color=sem_color[idx], alpha=0.02) 
            # color=sem_greens[idx], alpha=0.2)            
    
    
    # fix
    ax.plot(
        control_isis[0],
        control_means[0],
        color='black', marker='.', label='Control', markersize=4)
    ax.fill_between(
        control_isis[0],
        control_means[0] - control_sems[0],
        control_means[0] + control_sems[0],
        color='grey', alpha=0.2)
    
    # opto
    # all
    # ax.plot(
    #     bin_isi_opto,
    #     bin_mean_opto,
    #     color='indigo', marker='.', label='opto', markersize=4)
    # ax.fill_between(
    #     bin_isi_opto,
    #     bin_mean_opto - bin_sem_opto,
    #     bin_mean_opto + bin_sem_opto,
    #     color='violet', alpha=0.2)    
    
    
    # left_label = 'opto left'
    # right_label = 'opto right'
    
    # if subject not in ['LCHR_TS01_opto', 'LCHR_TS02_opto']:
    #     # left_label = subject
    #     # right_label = subject
    #     left_label = 'opto'
    #     right_label = 'opto'
    
    
    # if len(bin_isi_opto_left) > 0:
    #     # left
    #     ax.plot(
    #         bin_isi_opto_left,
    #         bin_mean_opto_left,
    #         color='blue', marker='.', label=left_label, markersize=4)
    #     ax.fill_between(
    #         bin_isi_opto_left,
    #         bin_mean_opto_left - bin_sem_opto_left,
    #         bin_mean_opto_left + bin_sem_opto_left,
    #         color='violet', alpha=0.2)   

    # if len(bin_isi_opto_right) > 0:
    #     # right
    #     ax.plot(
    #         bin_isi_opto_right,
    #         bin_mean_opto_right,
    #         color='green', marker='.', label=right_label, markersize=4)
    #     ax.fill_between(
    #         bin_isi_opto_right,
    #         bin_mean_opto_right - bin_sem_opto_right,
    #         bin_mean_opto_right + bin_sem_opto_right,
    #         color='lightgreen', alpha=0.2)   
    
    
     
    # ax.plot(
    #     bin_isi_jitter,
    #     bin_mean_jitter,
    #     color='limegreen', marker='.', label='jitter', markersize=4)
    # ax.fill_between(
    #     bin_isi_jitter,
    #     bin_mean_jitter - bin_sem_jitter,
    #     bin_mean_jitter + bin_sem_jitter,
    #     color='limegreen', alpha=0.2)
    # ax.plot(
    #     bin_isi_chemo,
    #     bin_mean_chemo,
    #     color='red', marker='.', label='chemo', markersize=4)
    # ax.fill_between(
    #     bin_isi_chemo,
    #     bin_mean_chemo - bin_sem_chemo,
    #     bin_mean_chemo + bin_sem_chemo,
    #     color='red', alpha=0.2)
    # ax.plot(
    #     bin_isi_opto,
    #     bin_mean_opto,
    #     color='dodgerblue', marker='.', label='opto', markersize=4)
    # ax.fill_between(
    #     bin_isi_opto,
    #     bin_mean_opto - bin_sem_opto,
    #     bin_mean_opto + bin_sem_opto,
    #     color='dodgerblue', alpha=0.2)
    
    x_left = isi_short_mean - 100
    x_right = isi_long_mean + 100
    cat = isi_orig
    x_left = 0
    x_right = 2*cat
    
    ax.vlines(
        cat, 0.0, 1.0,
        linestyle='--', color='mediumseagreen',
        label='Category Boundary')
    ax.hlines(0.5, x_left, x_right, linestyle='--', color='grey')
    # ax.vlines(500, 0.0, 1.0, linestyle=':', color='grey')
    ax.tick_params(tick1On=False)
    ax.tick_params(axis='x', rotation=45)    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([x_left,x_right])
    ax.set_ylim([-0.05,1.05])
    # ax.set_xticks(np.arange(6)*200)
    # ax.set_xticks(np.arange(11)*150)
    ax.set_xticks(np.arange(0,x_right,250))
    ax.set_yticks(np.arange(5)*0.25)
    ax.set_xlabel('ISI')
    ax.set_ylabel('Prob. of Choosing the Right Side (mean$\pm$sem)')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    start_from = []
    start_date = []
    date = M['dates'][sessionIdx]
    if start_from=='start_date':
        ax.set_title('average psychometric function from ' + start_date)
    elif start_from=='non_naive':
        ax.set_title('average psychometric function non-naive')
    else:
        if not is_avg:
            ax.set_title('Residual Opto Psychometric Function ' + date)
        else:
            ax.set_title('Average Residual Opto Psychometric Function ' + dates[0] + '-' + dates[-1])
        
    if show_plot:
        plt.show()
        
    subject = config['list_config'][subjectIdx]['subject_name']
    output_dir = os.path.join(config['paths']['figure_dir_local'] + subject)
    
    if figure_id is None:
        figure_id = f"{subject}_psychometric_opto_epoch_residual_s{sessionIdx}"
    
    # figure_id = f"{subject}_psychometric_opto_epoch"
    
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)    

    # return {
    #     'figure_id': figure_id,
    #     'path': out_path,
    #     'caption': f"Psychometric residual plot for {subject}",
    #     'subject': subject,
    #     'tags': ['performance', 'bias'],
    #     "layout": {
    #       "page": 1,
    #       "page_key": "pdf_pg_opto_psychometric_residual", 
    #       "row": 0,
    #       "col": 0,
    #       "rowspan": rowspan,
    #       "colspan": colspan,
    #     }        
    # }       

    return out_path