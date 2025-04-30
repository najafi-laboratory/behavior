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
    return decision_fix, decision_jitter, decision_chemo, decision_opto

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

def get_bin_stat(decision, max_time, filtered_df):
    # bin_size=250
    bin_size=25
    # bin_size=5    
    # bin_size=100
    least_trials=3
    bins = np.arange(0, max_time + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(decision[0,:], bins) - 1
    
    
    filtered_df = filter_df(filtered_df)   
    filtered_df = filtered_df[(filtered_df['is_opto'] == False)]
    filtered_df['earliest_lick'] = filtered_df.apply(get_earliest_lick, axis=1)
    
    max_val = filtered_df['opto_encode'].max()
    if pd.notna(max_val):
        max_post_opto_encode = int(max_val)
    else:
        max_post_opto_encode = 0  # or another default value       
                
   
    # Assign each row to a bin
    filtered_df['bin_idx'] = np.digitize(filtered_df['earliest_lick'], bins) - 1    
    bin_indices_df = np.digitize(filtered_df.earliest_lick, bins) - 1
    
    bin_mean = []
    bin_sem = []
    trials_per_bin = []
    
    
    means = []
    sems = []    
    trials_per_bin_df = []
    time = []    
    
    # print('')
    # print('arr')
    for i in range(len(bins)-1):  
        if print_debug:
            print('')
            print('df')
        trials_df = filtered_df[(filtered_df['bin_idx'] == i)]
        correctness_df = trials_df['rewarded']
        m_df = np.mean(correctness_df) if len(correctness_df) > least_trials else np.nan
        s_df = sem(correctness_df) if len(correctness_df) > least_trials else np.nan
        num_trials_df = len(trials_df) if len(correctness_df) > least_trials else np.nan
        trials_per_bin_df.append(num_trials_df)  
        means.append(m_df)
        sems.append(s_df)
        time.append(bins[i])
        if print_debug:
            print(f'i:{i}, m_df:{m_df}, t:{trials_per_bin_df}')    
        
        if print_debug:
            print('')
            print('arr')
        correctness = decision[2, bin_indices == i].copy()
        test = decision[:, bin_indices == i].copy()
        m = np.mean(correctness) if len(correctness) > least_trials else np.nan
        s = sem(correctness) if len(correctness) > least_trials else np.nan
        # num_trials = np.sum(bin_indices[bin_indices == i]) if len(correctness) > least_trials else np.nan
        num_trials = len(bin_indices[bin_indices == i]) if len(correctness) > least_trials else np.nan
        trials_per_bin.append(num_trials)  
        bin_mean.append(m)
        bin_sem.append(s)
        if print_debug:
            print(f'i:{i}, m:{m}, t:{trials_per_bin}')
   
    means = np.asarray(means)
    sems = np.asarray(sems)
    time = np.asarray(time)
    trials_per_bin = np.asarray(trials_per_bin)
    
    nan_indices = np.where(np.isnan(means))[0]
    means = np.delete(means, nan_indices)
    sems = np.delete(sems, nan_indices)
    times = np.delete(time, nan_indices)
    trials_per_bin = np.delete(trials_per_bin, nan_indices)    
   
    
   
    
    bin_mean = np.array(bin_mean)
    bin_sem  = np.array(bin_sem)
    bin_time = bins[:-1] + (bins[1]-bins[0]) / 2
    non_nan  = (1-np.isnan(bin_mean)).astype('bool')
    bin_mean = bin_mean[non_nan]
    bin_sem  = bin_sem[non_nan]
    bin_time = bin_time[non_nan]
    trials_per_bin = np.array(trials_per_bin)
    trials_per_bin = trials_per_bin[non_nan]
    return bin_mean, bin_sem, bin_time, trials_per_bin

def bin_stats(filtered_df, isOpto):
    # Define bin edges
    max_isi = 1000
    incr_isi = 25  # change as needed
      
    
    
    bins = np.arange(0, max_isi + incr_isi, incr_isi)
    bin_labels = bins[:-1]  # label bins by left edge
    # bin_centers = bins[:-1] - incr_isi / 2
    bins = bins - incr_isi / 2
    # bin_indices = np.arange(0, len(bins))
    # bin_indices = np.digitize(decision[0,:], bins) - 1
    # means = [np.nan] * len(bins)
    # sems = [np.nan] * len(bins)

# filtered_df['EarliestLick']
    bin_indices = np.digitize(filtered_df.earliest_lick, bins) - 1
    
    if isOpto:
        least_trials=1
    else:
        least_trials=3


    means = []
    sems = []    
    trials_per_bin = []
    time = []
    if print_debug:
        print('')
        print('df')
    for i in range(len(bins)-1):    
        trials = filtered_df[(filtered_df['bin_idx'] == i)]
        correctness = trials['rewarded']
        m = np.mean(correctness) if len(correctness) > least_trials else np.nan
        s = sem(correctness) if len(correctness) > least_trials else np.nan
        num_trials = len(trials) if len(correctness) > least_trials else np.nan
        trials_per_bin.append(num_trials)  
        means.append(m)
        sems.append(s)
        time.append(bins[i])
        if print_debug:
            print(f'i:{i}, m:{m}, t:{trials_per_bin}')
        
    # reward_stats = (
    #     filtered_df
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
        
    means = np.asarray(means)
    sems = np.asarray(sems)
    time = np.asarray(time)
    trials_per_bin = np.asarray(trials_per_bin)
    
    nan_indices = np.where(np.isnan(means))[0]
    means = np.delete(means, nan_indices)
    sems = np.delete(sems, nan_indices)
    times = np.delete(time, nan_indices)
    trials_per_bin = np.delete(trials_per_bin, nan_indices)
    
    # opto_residual_means.append(means)            
    # opto_residual_sems.append(sems)
    # opto_residual_isis.append(isi)
       
    return means, sems, times, trials_per_bin

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
    control_means = []
    control_sems = []
    control_times = []
    control_trials = []
    
    filtered_df = filter_df(processed_dec)   
    filtered_df = filtered_df[(filtered_df['is_opto'] == False)]
    filtered_df['earliest_lick'] = filtered_df.apply(get_earliest_lick, axis=1)
    
    max_val = filtered_df['opto_encode'].max()
    if pd.notna(max_val):
        max_post_opto_encode = int(max_val)
    else:
        max_post_opto_encode = 0  # or another default value       
                
    # Define bin edges
    # max_isi = filtered_df['isi'].max() + 200
    max_time = 1000
    incr_isi = 25  # change as needed
      
    bins = np.arange(0, max_time + incr_isi, incr_isi)
    bins = bins[:-1] - incr_isi / 2
    
    # Assign each row to a bin
    filtered_df['bin_idx'] = np.digitize(filtered_df['earliest_lick'], bins) - 1      
    
    means, sems, times, trials = bin_stats(filtered_df, isOpto=False)
    
    control_means.append(means)            
    control_sems.append(sems)
    control_times.append(times)   
    control_trials.append(trials)
        
    return control_means, control_sems, control_times, control_trials

def get_earliest_lick(row):
    left = row['licks_left_start']
    right = row['licks_right_start']

    all_licks = []

    if isinstance(left, list):
        all_licks.extend([v for v in left if not np.isnan(v)])
    if isinstance(right, list):
        all_licks.extend([v for v in right if not np.isnan(v)])

    return min(all_licks) if all_licks else np.nan


def bin_opto_residuals(processed_dec, num_residuals):
   
    opto_residual_means = []
    opto_residual_sems = []
    opto_residual_times = []
    opto_trials = []
    
    filtered_df = filter_df(processed_dec)
    
    # earliest_lick = np.nanmin(filtered_df[['licks_left_start', 'licks_right_start']].values, axis=1)
    # filtered_df['earliest_lick'] = earliest_lick
    
    # filtered_df['earliest_lick'] = filtered_df[['licks_left_start', 'licks_right_start']].min(axis=1, skipna=True)
    
    filtered_df = filtered_df[(filtered_df['is_opto'] == True)]
    
    filtered_df['earliest_lick'] = filtered_df.apply(get_earliest_lick, axis=1)
    # sorted_df = filtered_df.sort_values(by='earliest_lick', ascending=True, na_position='last')
    
    # test = filtered_df.iloc[17]
    # test = filtered_df[['licks_left_start', 'licks_right_start']].min(axis=1, skipna=True))
    
    
    max_val = filtered_df['opto_encode'].max()
    if pd.notna(max_val):
        max_post_opto_encode = int(max_val)
    else:
        max_post_opto_encode = 0  # or another default value       
        
        
    # Define bin edges
    # max_isi = filtered_df['isi'].max() + 200
    max_time = 1000
    incr_isi = 25  # change as needed
      
    bins = np.arange(0, max_time + incr_isi, incr_isi)
    bins = bins[:-1] - incr_isi / 2

    # Assign each row to a bin
    filtered_df['bin_idx'] = np.digitize(filtered_df['earliest_lick'], bins) - 1    
    
    means, sems, times, trials = bin_stats(filtered_df, isOpto=True)

    opto_residual_means.append(means)            
    opto_residual_sems.append(sems)
    opto_residual_times.append(times)   
    opto_trials.append(trials)
  
    return opto_residual_means, opto_residual_sems, opto_residual_times

# def clear_nan(array):    
#     return array[~np.isnan(array)]

def get_decision(M):
    decision = M['decision']
    decision = [np.concatenate(d, axis=1) for d in decision]
    # decision = np.concatenate(decision, axis=1)
    
    if len(decision) > 0:
        decision = np.concatenate(decision, axis=1)
    else:
        decision = np.array([])  # or handle this case another way    
    
    jitter_flag = M['jitter_flag']
    jitter_flag = np.concatenate(jitter_flag).reshape(1,-1)
    opto_flag = M['opto_flag']
    opto_flag = np.concatenate(opto_flag).reshape(1,-1)
    jitter_flag[0 , :] = jitter_flag[0 , :] + opto_flag[0 , :]*3
    outcomes = M['outcomes']
    all_trials = 0
    # chemo_labels = M['Chemo']
    # for j in range(len(chemo_labels)):
    #     if chemo_labels[j] == 1:
    #         jitter_flag[0 , all_trials:all_trials+len(outcomes[j])] = 2*np.ones(len(outcomes[j]))
    #     all_trials += len(outcomes[j])
    pre_isi = M['pre_isi']
    pre_isi = np.concatenate(pre_isi).reshape(1,-1)
    post_isi_mean = M['isi_post_emp']
    post_isi_mean = np.concatenate(post_isi_mean).reshape(1,-1)
    choice_start = M['choice_start']
    choice_start = np.concatenate(choice_start).reshape(-1)     
    # stim_start = M['stim_start']
    # stim_start = np.concatenate(stim_start).reshape(-1)
    # decision = np.concatenate([decision, jitter_flag, pre_isi, post_isi_mean], axis=0)
    decision = np.concatenate([decision, jitter_flag, post_isi_mean], axis=0)    
    # decision[0,:] -= stim_start
    # decision[0,:] -= choice_start
    decision[0,:] -= 1000*choice_start
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    decision = decision[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: jitter flag.
    # row 4: pre pert isi.
    # row 5: post pert isi.
    decision_fix, decision_jitter, decision_chemo, decision_opto = separate_fix_jitter(decision)
    return decision_fix, decision_jitter, decision_chemo, decision_opto


def plot_decision_time_opto(M, config, subjectIdx, sessionIdx=-1, figure_id=None, show_plot=1):

    # figure meta
    rowspan, colspan = 2, 2
    fig_size = get_figsize_from_pdf_spec(rowspan, colspan, config['pdf_spec']['pdf_pg_opto_psychometric'])    
    fig, ax = plt.subplots(figsize=fig_size) 

    subject = config['list_config'][subjectIdx]['subject_name']
    
    session_settings = M['session_settings'][sessionIdx]
    isi_short_mean = session_settings['ISIShortMean_s'] * 1000
    isi_long_mean = session_settings['ISILongMean_s'] * 1000
    isi_orig = session_settings['ISIOrig_s'] * 1000
    
    is_avg = 0
    if sessionIdx != -1:
        decision_df = get_decision_df(M, sessionIdx)
        dates = M['dates'][sessionIdx]
    else:
        is_avg = 1
        dates = M['dates']
        combined_df = pd.DataFrame()
        for sessionIdx in range(0, len(dates)):
            decision_df = get_decision_df(M, sessionIdx)
            combined_df = pd.concat([combined_df, decision_df], ignore_index=True)
        decision_df = combined_df
    
    num_residuals = 5
    
    opto_residual_means, opto_residual_sems, opto_residual_times = bin_opto_residuals(decision_df, num_residuals)
    
    control_means, control_sems, control_times, control_trials = bin_control(decision_df)
    
    max_time = 1000 # choice window is 5s, although most licks are 1s or less
    decision_fix, decision_jitter, decision_chemo, decision_opto = get_decision(M)
    # bin_mean_fix, bin_sem_fix, bin_time_fix, trials_per_bin_fix = get_bin_stat(decision_fix, max_time, decision_df)
    # bin_mean_jitter, bin_sem_jitter, bin_time_jitter, trials_per_bin_jitter = get_bin_stat(decision_jitter, max_time)
    # bin_mean_chemo, bin_sem_chemo, bin_time_chemo, trials_per_bin_chemo = get_bin_stat(decision_chemo, max_time)
    # bin_mean_opto, bin_sem_opto, bin_time_opto, trials_per_bin_opto = get_bin_stat(decision_opto, max_time)

    
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
    
    def generate_blue_colors(n):
        blues = []
        for i in range(n):
            # The green component ranges from 255 (light) to 0 (dark)
            blue_value = int(255 * ( i / (n - 1)))  # Linear interpolation
            # greens.append((0, green_value, 0))  # RGB format: (Red, Green, Blue)
            blues.append(np.array([blue_value, 0, 0]) / 255)  # Normalize to [0, 1]
        return blues    
    
    mean_greens = generate_green_colors(num_residuals)
    sem_greens = generate_green_colors(num_residuals, shift=100)
    
    def generate_opto_labels(num):
        labels = []
        labels.append('opto')
        for i in range(0, num):        
            if i != 0:
                labels.append('opto + ' + str(i))
        return labels
    
    labels = generate_opto_labels(num_residuals)
    

    # opto_side
    
    # # plot opto residuals
    # for idx in range(0, num_residuals):
    #     # fix
        
    #     ax.plot(
    #         opto_residual_isis[idx],
    #         opto_residual_means[idx],
    #         color=mean_greens[idx], marker='.', label=labels[idx], markersize=4)
    #     ax.fill_between(
    #         opto_residual_isis[idx],
    #         opto_residual_means[idx] - opto_residual_sems[idx],
    #         opto_residual_means[idx] + opto_residual_sems[idx],
    #         color='honeydew', alpha=0.2) 
    #         # color=sem_greens[idx], alpha=0.2)    
    
    for idx in range(len(control_means)):    
        ax.plot(
            control_times[idx],
            control_means[idx],
            color='indigo',
            marker='.',
            label='control',
            markersize=4)
        
        ax.fill_between(
            control_times[idx],
            control_means[idx] - control_sems[idx],
            control_means[idx] + control_sems[idx],
            color='violet',
            alpha=0.2)    
    

    for idx in range(len(opto_residual_means)):     
        ax.plot(
            opto_residual_times[idx],
            opto_residual_means[idx],
            color='blue',
            marker='.',
            label='opto',
            markersize=4)
        
        ax.fill_between(
            opto_residual_times[idx],
            opto_residual_means[idx] - opto_residual_sems[idx],
            opto_residual_means[idx] + opto_residual_sems[idx],
            color='lightblue',
            alpha=0.2)    
    # # fix
    # for idx in range(len(control_means)):   
    #     ax.plot(
    #         control_times[idx],
    #         control_means[0],
    #         color='black', marker='.', label='control', markersize=4)
    #     ax.fill_between(
    #         control_times[idx],
    #         control_means[0] - control_sems[0],
    #         control_means[0] + control_sems[0],
    #         color='grey', alpha=0.2)
        max_time = 1000
    
    ax.hlines(
        0.5, 0.0, max_time,
        linestyle=':', color='grey')
    # ax.vlines(
    #     1300, 0.0, 1.0,
    #     linestyle=':', color='mediumseagreen')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_xlim([0, max_time])
    ax.set_xlim([0, 650])
    # ax.set_xlim([0, 1000])
    ax.set_ylim([0.20, 1.05])
    ax.set_xlabel('decision time (since choice window onset) / s')
    ax.set_ylabel('correct prob.')
    # ax.set_xticks(np.arange(0, max_time, 1000))
    # ax.set_xticks(np.arange(0, max_time, 100))
    ax.set_xticks(np.arange(0, 650, 100))
    ax.tick_params(axis='x', rotation=45)
    # ax.set_xticklabels(rotation=45)
    ax.set_yticks([0.25, 0.50, 0.75, 1])
    
    
    # Create a second axis on the right side with a different scale
    # ax2 = ax.figure.add_axes(ax.get_position())  # Copy position from ax1
    ax2 = ax.twinx()
    # ax2.set_frame_on(False)  # Hide the box of the second axis
    # ax2.plot(x, y2, 'b-', label='2*cos(x)')
    ax2.set_ylabel('trials per bin')
    # ax2.tick_params(axis='y', labelcolor='b')    
    
    for idx in range(len(control_means)): 
        ax2.plot(
            control_times[idx],
            control_trials[idx],
            color='gray',
            marker='.',
            label='control',
            markersize=4)
    
    
    # ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1, 1))
    ax.legend(loc='best', ncol=1, bbox_to_anchor=(1, 1))
    ax.set_title('average decision time curve')
    
    # if start_from=='start_date':
    #     ax.set_title('average decision time curve from ' + start_date)
    # elif start_from=='non_naive':
    #     ax.set_title('average decision time curve non-naive')
    # else:
    ax.set_title('average decision time curve')      
        
    if show_plot:
        plt.show()
        
    subject = config['list_config'][subjectIdx]['subject_name']
    output_dir = os.path.join(config['paths']['figure_dir_local'] + subject)
    figure_id = f"{subject}_decision_time_opto"
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
    
    # x_left = isi_short_mean - 100
    # x_right = isi_long_mean + 100
    # cat = isi_orig
    # x_left = 0
    # x_right = 2*cat
    
    # ax.vlines(
    #     cat, 0.0, 1.0,
    #     linestyle='--', color='mediumseagreen',
    #     label='Category Boundary')
    # ax.hlines(0.5, x_left, x_right, linestyle='--', color='grey')
    # # ax.vlines(500, 0.0, 1.0, linestyle=':', color='grey')
    # ax.tick_params(tick1On=False)
    # ax.tick_params(axis='x', rotation=45)    
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.set_xlim([x_left,x_right])
    # ax.set_ylim([-0.05,1.05])
    # # ax.set_xticks(np.arange(6)*200)
    # # ax.set_xticks(np.arange(11)*150)
    # ax.set_xticks(np.arange(0,x_right,250))
    # ax.set_yticks(np.arange(5)*0.25)
    # ax.set_xlabel('post perturbation isi')
    # ax.set_ylabel('prob. of choosing the right side (mean$\pm$sem)')
    # ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    # start_from = []
    # start_date = []
    # date = M['dates'][sessionIdx]
    # if start_from=='start_date':
    #     ax.set_title('average psychometric function from ' + start_date)
    # elif start_from=='non_naive':
    #     ax.set_title('average psychometric function non-naive')
    # else:
    #     if not is_avg:
    #         ax.set_title('residual opto psychometric function ' + date)
    #     else:
    #         ax.set_title('average residual opto psychometric function ' + dates[0] + '-' + dates[-1])
        
    # if show_plot:
    #     plt.show()
        
    # subject = config['list_config'][subjectIdx]['subject_name']
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject)
    
    # if figure_id is None:
    #     figure_id = f"{subject}_psychometric_opto_epoch_residual_s{sessionIdx}"
    
    # # figure_id = f"{subject}_psychometric_opto_epoch"
    
    # file_ext = '.pdf'
    # filename = figure_id + file_ext
    # os.makedirs(output_dir, exist_ok=True)
    # out_path = os.path.join(output_dir, filename)
    # fig.savefig(out_path, bbox_inches='tight', dpi=300)
    # plt.close(fig)    

    # # return {
    # #     'figure_id': figure_id,
    # #     'path': out_path,
    # #     'caption': f"Psychometric residual plot for {subject}",
    # #     'subject': subject,
    # #     'tags': ['performance', 'bias'],
    # #     "layout": {
    # #       "page": 1,
    # #       "page_key": "pdf_pg_opto_psychometric_residual", 
    # #       "row": 0,
    # #       "col": 0,
    # #       "rowspan": rowspan,
    # #       "colspan": colspan,
    # #     }        
    # # }       

    # return out_path