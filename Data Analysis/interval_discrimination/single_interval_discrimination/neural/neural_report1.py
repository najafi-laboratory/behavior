# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:22:43 2025

@author: timst
"""
import os
import h5py
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
# from matplotlib.collections import BrokenBarHCollection
from scipy.cluster.vq import kmeans2

def get_session_df(session_data, session_idx):
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
        
        reward_start = np.nan
        punish_start = np.nan
        
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
                licks['licks_left_start'] = [np.float64(-1)]
                licks['licks_left_stop'] = [np.float64(-1)]
                licks['licks_right_start'] = [np.float64(-1)]
                licks['licks_right_stop'] = [np.float64(-1)]

        trial_start = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Start'][0]
        trial_stop = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['ITI'][1]
            
        choice_start =  raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['WindowChoice'][0]
        stim_start_bpod = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['AudStimTrigger'][0]
        stim_stop_bpod = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['AudStimTrigger'][1]
        

        if rewarded:
            if not is_naive:
                reward_start = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][0]
            else:
                reward_start = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['NaiveRewardDeliver'][0]
        else:
            if not is_naive:
                punish_start = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Punish'][0]
            else:
                punish_start = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['PunishNaive'][0]
        
        choice_stop =  raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['WindowChoice'][1]
        lick_time_s = choice_stop
        
        # reward_start = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][0]
        # punish_start = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Punish'][0]
        
        is_opto = opto_flags[trial]   
        
        if is_opto and not is_naive:
            opto_encode = 0
            
        isi = session_data['isi_post_emp'][session_idx][trial]
        
        move_correct_spout = session_data['move_correct_spout_flag'][session_idx][trial]
                  
        processed_dec.append({
            "trial_index": trial,
            "trial_start": trial_start,
            "trial_stop": trial_stop,
            "choice_start": choice_start,
            "choice_stop": choice_stop,
            "stim_start_bpod": stim_start_bpod,
            "stim_stop_bpod": stim_stop_bpod,
            
            "reward_start": reward_start,
            "punish_start": punish_start,
            
            "trial_side": trial_type,
            "stim_duration": isi,
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
        all_licks.extend([v for v in left if not np.isnan(v) and v >= 0])
    if isinstance(right, list):
        all_licks.extend([v for v in right if not np.isnan(v) and v >= 0])

    return min(all_licks) if all_licks else np.nan

# 1 = mouse chose right
# 0 = mouse chose left
# We'll infer based on whether rewarded and which side was correct
def infer_choice(row):
    if row['rewarded'] == 1:
        return row['is_right']  # Mouse must have chosen correctly
    else:
        return 1 - row['is_right']  # Mouse chose the wrong side

def add_model_features(df, exp_alpha=0.7):
    """
    Adds features to a trial-by-trial dataframe for modeling behavior.
    
    Parameters:
        df (pd.DataFrame): Combined trial data across sessions.
    
    Returns:
        pd.DataFrame: DataFrame with new model features.
    """
    df = df.copy()

    # ---- Crecombinase Features ----
    # Encode trial type    
    df['is_right'] = (df['trial_side'] == 'right').astype(int)
    df['mouse_choice'] = df.apply(infer_choice, axis=1)
    # Opto flag as int
    df['is_opto'] = df['is_opto'].astype(int)


    # Normalize trial index within session
    df['norm_trial_index'] = df.groupby('session_id')['trial_index'].transform(
        lambda x: x / x.max()
    )

    # # ---- Optional / Temporal Dynamics ----
    # # Previous choice (encoded as 1 = right, 0 = left, np.nan for first trial of session)
    # df['prev_choice'] = df.groupby('session_id')['is_right'].shift(1)
    # # Previous reward (1 = rewarded, 0 = not rewarded)
    # df['prev_reward'] = df.groupby('session_id')['rewarded'].shift(1)
    # # Previous opto
    # df['prev_opto'] = df.groupby('session_id')['is_opto'].shift(1)
    
    # --- Add multi-back history features ---
    for n in range(1, 4):
        df[f'choice_{n}back'] = df.groupby('session_id')['is_right'].shift(n)
        df[f'reward_{n}back'] = df.groupby('session_id')['rewarded'].shift(n)
        df[f'opto_{n}back'] = df.groupby('session_id')['is_opto'].shift(n)      
    
    # --- Stay vs switch from previous choice ---
    df['stay_from_1back'] = df['is_right'] == df['choice_1back']
    df['stay_from_1back'] = df['stay_from_1back'].astype(float)  # Make it 0.0 / 1.0 (or NaN)


    # ---- Response Times ----
    df['lick_time'] = df.apply(get_earliest_lick, axis=1)   
    # df['lick_time_s'] = df['lick_time']/1000
    df['response_time'] = df['lick_time']     
    # if 'lick_time_stim' in df.columns:
    #     df['rt_from_stim'] = df['lick_time_stim']
    # if 'lick_time_choice' in df.columns:
    #     df['rt_from_choice'] = df['lick_time_choice']        


    # Interaction: ISI x Opto
    if 'stim_duration' in df.columns:
        df['isi_opto_interaction'] = df['stim_duration'] * df['is_opto']

    # Rolling average of reward over last 5 trials (behavioral state estimate)
    df['rolling_accuracy'] = df.groupby('session_id')['rewarded'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

    # --- Rolling choice bias (last 5) ---
    df['rolling_choice_bias'] = df.groupby('session_id')['is_right'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

    # --- Exponential decay choice bias (alpha-weighted) ---
    def exp_decay(x, alpha=exp_alpha):
        ewma = [np.nan]  # First trial has no history
        for i in range(1, len(x)):
            prev = ewma[-1]
            if np.isnan(prev):
                ewma.append(x.iloc[i - 1])
            else:
                ewma.append(alpha * x.iloc[i - 1] + (1 - alpha) * prev)
        return pd.Series(ewma, index=x.index)

    df['exp_choice_bias'] = df.groupby('session_id')['is_right'].transform(
        lambda x: exp_decay(x, alpha=exp_alpha)
    )

    return df



def load_bruker_frame_times(xml_file, ops=None):
    print(f"[LOADING] {os.path.basename(xml_file)} from {os.path.dirname(xml_file)}")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    frame_times = []

    # Find all Sequence nodes with type="TSeries Timed Element"
    sequences = [seq for seq in root.findall('.//Sequence') if seq.attrib.get('type') == 'TSeries Timed Element']
    if len(sequences) == 0:
        raise ValueError("No <Sequence> with type='TSeries Timed Element' found in XML file.")

    sequence = sequences[0]  # Use the first matching Sequence
    print(f"[INFO] Found {len(sequences)} <Sequence> elements with type='TSeries Timed Element'. Using the first one.")

    # Look inside Sequence for Frames
    frames = sequence.findall('Frame')
    print(f"[INFO] Found {len(frames)} <Frame> elements inside the selected <Sequence>.")

    for frame in frames:
        rel_time_sec = float(frame.attrib.get('relativeTime', frame.attrib.get('RelativeTime', 0)))
        frame_times.append(rel_time_sec)

    frame_times = np.array(frame_times)
    print(f"[SUCCESS] Loaded {len(frame_times)} frame times from XML.")

    if ops is not None:
        frame_num = ops.get('nframes', None)
        if frame_num is not None:
            print(f"[VALIDATION] Number of frames in ops: {frame_num}")
            if frame_num != len(frame_times):
                raise ValueError(f"Mismatch between ops['nframes'] ({frame_num}) and XML frame times ({len(frame_times)}).")
            else:
                print(f"[SUCCESS] XML frame times match ops['nframes'].")

    return frame_times



def segment_trials_with_voltage_sync(dff, frame_times, trials_df, voltage_time, voltage_sync_signal, voltage_stim, threshold=0.5, ops=None, bruker_xml_path=None):
    """
    Segments DFF data trial-by-trial using synchronization signals from voltage trace.

    Args:
        dff (np.ndarray): Neural traces (neurons x frames).
        frame_times (np.ndarray or None): Frame timestamps (frames,). If None, reconstruct from ops or load from XML.
        trials_df (pd.DataFrame): Trial dataframe with 'start' and 'stop' times.
        voltage_time (np.ndarray): Timestamps of voltage recordings.
        voltage_sync_signal (np.ndarray): Voltage signal containing sync pulses.
        threshold (float): Threshold to detect leading edges in the voltage signal.
        ops (dict, optional): Suite2p ops dictionary, required if frame_times is None and no XML.
        bruker_xml_path (str, optional): Path to Bruker XML file containing frame times.

    Returns:
        trials_df (pd.DataFrame): Updated dataframe with added columns 'scope_dff' and 'scope_time'.
    """
    print("[START] Segmenting trials using voltage synchronization signals")

    if frame_times is None:
        if bruker_xml_path is not None:
            print("[INFO] frame_times not provided, loading from Bruker XML.")
            frame_times = load_bruker_frame_times(bruker_xml_path, ops=ops)
        else:
            print("[INFO] frame_times not provided, attempting to reconstruct from ops.")
            assert ops is not None, "ops must be provided if frame_times and bruker_xml_path are None."
            frame_times = np.arange(dff.shape[1]) / ops['fs']
            print(f"[RECONSTRUCTED] frame_times with shape {frame_times.shape} using fs = {ops['fs']} Hz.")

    print(f"[INFO] dff shape: {dff.shape}, frame_times shape: {frame_times.shape}")
    print(f"[INFO] voltage_time shape: {voltage_time.shape}, voltage_sync_signal shape: {voltage_sync_signal.shape}")
    print(f"[INFO] Number of trials: {len(trials_df)}")

    # convert voltage_time to s
    voltage_time = voltage_time / 1000

    # Find rising edges in voltage signal
    sync_edges = np.where(np.diff((voltage_sync_signal > threshold).astype(int)) == 1)[0]
    sync_times = voltage_time[sync_edges]
    print(f"[INFO] Detected {len(sync_times)} sync edges in voltage signal.")

    scope_dffs = []
    scope_times = []
    flash_times = []

    # plt.figure(figsize=(12, 6))
    # plt.plot(voltage_time, voltage_sync_signal)
    # plt.plot(voltage_time, voltage_stim)
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Voltage Sync Signal')
    # plt.title('Voltage Sync Signal Over Time')
    # plt.xlim([0, 50000])
    # plt.grid(True)
    # plt.show()

    for idx, (row, sync_time) in enumerate(zip(trials_df.itertuples(index=False), sync_times)):
        trial_start = row.trial_start
        trial_stop = row.trial_stop
        # print(f"[PROCESSING] Trial {idx}: start={trial_start:.3f}s, stop={trial_stop:.3f}s, sync_time={sync_time:.3f}s")

        trial_start_time = sync_time
        trial_end_time = sync_time + (trial_stop - trial_start)

        mask = (frame_times >= trial_start_time) & (frame_times <= trial_end_time)
        selected_frames = np.where(mask)[0]

        if len(selected_frames) == 0:
            print(f"[WARNING] No frames found for trial {idx}, skipping.")
            scope_dffs.append(None)
            scope_times.append(None)
            flash_times.append(None)
            continue

        trial_frame_times = frame_times[selected_frames] - trial_start_time
        trial_dff = dff[:, selected_frames]

        # print(f"[SUCCESS] Trial {idx}: trial_dff shape={trial_dff.shape}, trial_frame_times shape={trial_frame_times.shape}")

        scope_dffs.append(trial_dff)
        scope_times.append(trial_frame_times)
        
        # Find flash times from voltage_stim
        stim_mask = (voltage_time >= trial_start_time) & (voltage_time <= trial_end_time)
        trial_voltage_stim = voltage_stim[stim_mask]
        trial_voltage_time = voltage_time[stim_mask] - trial_start_time

        # if idx < 5:  # Print for first few trials only
        #     print(f"[DEBUG] Trial {idx}: trial_voltage_stim has {np.sum(trial_voltage_stim > 0.5)} ON samples")

        rising_edges = np.where(np.diff((trial_voltage_stim > 0.5).astype(int)) == 1)[0]
        falling_edges = np.where(np.diff((trial_voltage_stim > 0.5).astype(int)) == -1)[0]

        flashes = []
        for rise, fall in zip(rising_edges, falling_edges):
            flash_start = trial_voltage_time[rise]
            flash_end = trial_voltage_time[fall]
            flashes.append((flash_start, flash_end))

        flash_times.append(flashes)        

    trials_df = trials_df.copy()
    trials_df['scope_dff'] = scope_dffs
    trials_df['scope_time'] = scope_times
    trials_df['flash_times'] = flash_times

    print(f"[DONE] Segmented scope data for {len(trials_df)} trials.")
    return trials_df


def align_trials(trials_df, align_to_time_column='start_time'):
    print(f"[ALIGNING] Aligning trials to {align_to_time_column}")
    trials_df = trials_df.copy()

    if align_to_time_column not in trials_df.columns:
        raise ValueError(f"{align_to_time_column} column not found in trials_df.")

    aligned_times = []
    aligned_flashes = []
    aligned_reward_times = []
    aligned_punish_times = []

    for t, flashes, align_time, reward_time, punish_time in zip(
        trials_df['scope_time'],
        trials_df['flash_times'],
        trials_df[align_to_time_column],
        trials_df.get('reward_start', [None]*len(trials_df)),
        trials_df.get('punish_start', [None]*len(trials_df))
    ):
        if t is None or align_time is None:
            aligned_times.append(None)
            aligned_flashes.append(None)
            aligned_reward_times.append(None)
            aligned_punish_times.append(None)
        else:
            aligned_times.append(t - align_time)

            if flashes is not None:
                shifted_flashes = [(start - align_time, end - align_time) for (start, end) in flashes]
                aligned_flashes.append(shifted_flashes)
            else:
                aligned_flashes.append(None)

            if reward_time is not None:
                aligned_reward_times.append(reward_time - align_time)
            else:
                aligned_reward_times.append(None)

            if punish_time is not None:
                aligned_punish_times.append(punish_time - align_time)
            else:
                aligned_punish_times.append(None)

    trials_df['aligned_scope_time'] = aligned_times
    trials_df['aligned_flash_times'] = aligned_flashes
    trials_df['aligned_reward_time'] = aligned_reward_times
    trials_df['aligned_punish_time'] = aligned_punish_times

    # Debugging hook:
    first_valid = next((times for times in aligned_times if times is not None and len(times) > 0), None)
    if first_valid is not None:
        print(f"[DEBUG] Example aligned scope_time (first 5 points): {first_valid[:5]}")
    print("[DONE] Trials aligned.")    

    return trials_df


def plot_neural_raster(
    trials_df,
    neuron_idx=0,
    time_window=None,
    show_flash_bars=True,
    show_align_line=True,
    show_reward_lines=True,
    show_punish_lines=True,
    plot_title='Raster Plot',
    align_line_label='Alignment Event'):

    print("[PLOTTING] Neural raster plot with optional flash bars and alignment/reward/punish lines.")

    plt.figure(figsize=(12, 8))

    trial_idx = 0
    for scope_time, scope_dff, flashes, reward_time, punish_time in zip(
        trials_df['aligned_scope_time'],
        trials_df['scope_dff'],
        trials_df['aligned_flash_times'],
        trials_df['aligned_reward_time'],
        trials_df['aligned_punish_time']
    ):
        if scope_time is None or scope_dff is None:
            continue
        plt.plot(scope_time, scope_dff[neuron_idx, :] + trial_idx*2, color='black', alpha=0.5)

        if show_flash_bars and flashes is not None:
            for (flash_start, flash_end) in flashes:
                plt.fill_betweenx(
                    y=[trial_idx*2 - 1, trial_idx*2 + 1],
                    x1=flash_start, x2=flash_end,
                    color='blue', alpha=0.2
                )

        if show_reward_lines and reward_time is not None and not np.isnan(reward_time):
            plt.plot([reward_time, reward_time], [trial_idx*2 - 1, trial_idx*2 + 1], color='green', linestyle='--', linewidth=2, alpha=0.9, label='Reward Start' if trial_idx == 0 else "")

        if show_punish_lines and punish_time is not None and not np.isnan(punish_time):
            plt.plot([punish_time, punish_time], [trial_idx*2 - 1, trial_idx*2 + 1], color='red', linestyle='--', linewidth=2, alpha=0.9, label='Punish Start' if trial_idx == 0 else "")

        trial_idx += 1

    if show_align_line:
        plt.axvline(0, color='purple', linestyle='--', linewidth=2, label=align_line_label)

    plt.xlabel('Time aligned to event (s)')
    plt.ylabel('Trial # (shifted)')
    plt.title(plot_title)
    if time_window is not None:
        plt.xlim(time_window)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_trial_average(trials_df, neuron_idx=0, time_window=None):
    print("[PLOTTING] Trial-averaged neural trace.")

    all_times = []
    all_traces = []

    for scope_time, scope_dff in zip(trials_df['aligned_scope_time'], trials_df['scope_dff']):
        if scope_time is None or scope_dff is None:
            continue
        all_times.append(scope_time)
        all_traces.append(scope_dff[neuron_idx, :])

    if not all_traces:
        print("[WARNING] No valid trials to average.")
        return

    # Interpolate to a common time base
    common_time = np.linspace(time_window[0], time_window[1], 500) if time_window else np.linspace(-2, 4, 500)
    interpolated_traces = []
    for t, trace in zip(all_times, all_traces):
        interp = np.interp(common_time, t, trace, left=np.nan, right=np.nan)
        interpolated_traces.append(interp)

    interpolated_traces = np.array(interpolated_traces)
    mean_trace = np.nanmean(interpolated_traces, axis=0)
    sem_trace = np.nanstd(interpolated_traces, axis=0) / np.sqrt(np.sum(~np.isnan(interpolated_traces), axis=0))

    plt.figure(figsize=(10, 6))
    plt.plot(common_time, mean_trace, label='Mean Trace')
    plt.fill_between(common_time, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3, label='SEM')
    plt.xlabel('Time aligned to event (s)')
    plt.ylabel('dF/F')
    plt.title(f'Trial-Averaged Trace of Neuron {neuron_idx}')
    plt.legend()
    plt.grid(True)
    plt.show()

# def compute_neuron_averages(trials_df, time_window=[-2, 4], num_points=500):
#     print("[COMPUTING] Averaged traces for each neuron.")
#     common_time = np.linspace(time_window[0], time_window[1], num_points)
#     n_neurons = trials_df['scope_dff'].dropna().iloc[0].shape[0]

#     neuron_averages = []

#     for neuron_idx in range(n_neurons):
#         interpolated_traces = []

#         for scope_time, scope_dff in zip(trials_df['aligned_scope_time'], trials_df['scope_dff']):
#             if scope_time is None or scope_dff is None:
#                 continue
#             trace = scope_dff[neuron_idx, :]
#             interp = np.interp(common_time, scope_time, trace, left=np.nan, right=np.nan)
#             interpolated_traces.append(interp)

#         interpolated_traces = np.array(interpolated_traces)
#         mean_trace = np.nanmean(interpolated_traces, axis=0)
#         neuron_averages.append(mean_trace)

#     neuron_averages = np.array(neuron_averages)
#     print(f"[SUCCESS] Computed neuron averages: shape {neuron_averages.shape}")
#     return common_time, neuron_averages

# def compute_neuron_averages(trials_df, time_window=None, num_points=500):
#     print("[COMPUTING] Averaged traces for each neuron.")

#     if time_window is not None:
#         common_time = np.linspace(time_window[0], time_window[1], num_points)
#     else:
#         all_times = np.concatenate([row for row in trials_df['scope_time'].dropna() if len(row) > 0])
#         first_time = np.nanmin(all_times)
#         last_time = np.nanmax(all_times)
#         common_time = np.linspace(first_time, last_time, num_points)

#     n_neurons = trials_df['scope_dff'].dropna().iloc[0].shape[0]

#     avg_traces = np.zeros((n_neurons, len(common_time)))
#     counts = np.zeros(len(common_time))

#     for idx, row in trials_df.dropna(subset=['scope_dff']).iterrows():
#         times = row['scope_time']
#         traces = row['scope_dff']

#         if len(times) == 0 or len(traces) == 0:
#             continue

#         interp_traces = np.array([
#             np.interp(common_time, times, traces[n_idx, :], left=np.nan, right=np.nan)
#             for n_idx in range(n_neurons)
#         ])

#         valid_mask = ~np.isnan(interp_traces)
#         interp_traces[np.isnan(interp_traces)] = 0

#         avg_traces += interp_traces
#         counts += valid_mask.sum(axis=0)

#     counts[counts == 0] = 1
#     avg_traces /= counts

#     return common_time, avg_traces

# def compute_neuron_averages(trials_df, time_window=None, num_points=500):
#     print("[COMPUTING] Averaged traces for each neuron.")

#     valid_trials = trials_df.dropna(subset=['scope_time', 'scope_dff'])

#     if time_window is not None:
#         common_time = np.linspace(time_window[0], time_window[1], num_points)
#     else:
#         all_times = np.concatenate([row for row in valid_trials['scope_time'] if len(row) > 0])
#         first_time = np.nanmin(all_times)
#         last_time = np.nanmax(all_times)
#         common_time = np.linspace(first_time, last_time, num_points)

#     n_neurons = valid_trials.iloc[0]['scope_dff'].shape[0]

#     avg_traces = np.zeros((n_neurons, len(common_time)))
#     counts = np.zeros(len(common_time))

#     for idx, row in valid_trials.iterrows():
#         times = row['scope_time']
#         traces = row['scope_dff']

#         if len(times) == 0 or len(traces) == 0:
#             continue

#         interp_traces = np.array([
#             np.interp(common_time, times, traces[n_idx, :], left=np.nan, right=np.nan)
#             for n_idx in range(n_neurons)
#         ])

#         valid_mask = ~np.isnan(interp_traces)
#         interp_traces[np.isnan(interp_traces)] = 0

#         avg_traces += interp_traces
#         counts += valid_mask.sum(axis=0)

#     counts[counts == 0] = 1
#     avg_traces /= counts

#     return common_time, avg_traces

# def compute_neuron_averages(trials_df, time_window=None, num_points=500):
#     print("[COMPUTING] Averaged traces for each neuron.")

#     valid_trials = trials_df.dropna(subset=['scope_time', 'scope_dff'])

#     if time_window is None:
#         # Auto infer window based on min/max of shifted scope_times
#         all_times = np.concatenate([row for row in valid_trials['scope_time'] if len(row) > 0])
#         min_t = np.nanmin(all_times)
#         max_t = np.nanmax(all_times)
#         time_window = [min_t, max_t]

#     common_time = np.linspace(time_window[0], time_window[1], num_points)

#     n_neurons = valid_trials.iloc[0]['scope_dff'].shape[0]
#     avg_traces = np.zeros((n_neurons, len(common_time)))
#     counts = np.zeros(len(common_time))

#     for idx, row in valid_trials.iterrows():
#         times = row['scope_time']
#         traces = row['scope_dff']

#         if len(times) == 0 or len(traces) == 0:
#             continue

#         interp_traces = np.array([
#             np.interp(common_time, times, traces[n_idx, :], left=np.nan, right=np.nan)
#             for n_idx in range(n_neurons)
#         ])

#         valid_mask = ~np.isnan(interp_traces)
#         interp_traces[np.isnan(interp_traces)] = 0

#         avg_traces += interp_traces
#         counts += valid_mask.sum(axis=0)

#     counts[counts == 0] = 1
#     avg_traces /= counts

#     return common_time, avg_traces

def compute_neuron_averages(trials_df, time_window=None, num_points=500):
    print("[COMPUTING] Averaged traces for each neuron.")

    valid_trials = trials_df.dropna(subset=['aligned_scope_time', 'scope_dff'])
    print(f"[DEBUG] Valid trials after dropna: {len(valid_trials)}")



    if time_window is None:
        all_times = []
        for idx, t in enumerate(valid_trials['aligned_scope_time']):
            if t is not None and len(t) > 0:
                all_times.extend(t)
                if np.nanmin(t) < -100:   # Only print suspicious ones
                    print(f"[DEBUG] Trial {idx} suspicious min time: {np.nanmin(t):.2f}")        
        for t in valid_trials['aligned_scope_time']:
            if t is not None and len(t) > 0:
                all_times.extend(t)
        if len(all_times) == 0:
            print("[WARNING] No valid scope times found.")
            return None, None
        min_t = np.nanmin(all_times)
        max_t = np.nanmax(all_times)
        time_window = [min_t, max_t]
        print(f"[DEBUG] Inferred time_window from data: {time_window}")

    common_time = np.linspace(time_window[0], time_window[1], num_points)
    print(f"[DEBUG] Common time from {common_time[0]:.2f} to {common_time[-1]:.2f} with {len(common_time)} points")

    n_neurons = valid_trials.iloc[0]['scope_dff'].shape[0]
    print(f"[DEBUG] Number of neurons detected: {n_neurons}")

    avg_traces = np.zeros((n_neurons, len(common_time)))
    counts = np.zeros(len(common_time))

    for idx, row in valid_trials.iterrows():
        # times = row['scope_time']
        times = row['aligned_scope_time']
        traces = row['scope_dff']

        if times is None or traces is None or len(times) == 0 or len(traces) == 0:
            continue

        interp_traces = np.array([
            np.interp(common_time, times, traces[n_idx, :], left=np.nan, right=np.nan)
            for n_idx in range(n_neurons)
        ])

        valid_mask = ~np.isnan(interp_traces)
        interp_traces[np.isnan(interp_traces)] = 0

        avg_traces += interp_traces
        counts += valid_mask.sum(axis=0)

    counts[counts == 0] = 1
    avg_traces /= counts

    print(f"[SUCCESS] Computed averaged traces with shape {avg_traces.shape}")
    return common_time, avg_traces

def plot_neuron_heatmap(common_time, neuron_averages, sort_by='peak', time_window=(-2, 4), cmap='viridis', plot_title='Neuron Heatmap', align_line_label='Alignment Event', custom_sort_window=None):
    print("[PLOTTING] Neuron heatmap.")

    if sort_by == 'peak':
        sort_indices = np.argsort(np.nanargmax(neuron_averages, axis=1))
    elif sort_by == 'mean':
        sort_indices = np.argsort(np.nanmean(neuron_averages, axis=1))
    elif sort_by == 'integral':
        sort_indices = np.argsort(np.nansum(neuron_averages, axis=1))
    elif sort_by == 'max_amp':
        sort_indices = np.argsort(np.nanmax(neuron_averages, axis=1))
    elif sort_by == 'time_to_peak':
        time_to_peak = [common_time[np.nanargmax(trace)] if np.any(~np.isnan(trace)) else np.nan for trace in neuron_averages]
        sort_indices = np.argsort(time_to_peak)
    elif sort_by == 'custom_window' and custom_sort_window is not None:
        start_idx = np.searchsorted(common_time, custom_sort_window[0])
        end_idx = np.searchsorted(common_time, custom_sort_window[1])
        window_means = np.nanmean(neuron_averages[:, start_idx:end_idx], axis=1)
        sort_indices = np.argsort(window_means)
    else:
        sort_indices = np.arange(neuron_averages.shape[0])

    sorted_averages = neuron_averages[sort_indices, :]

    plt.figure(figsize=(12, 8))
    plt.imshow(sorted_averages, aspect='auto', extent=[common_time[0], common_time[-1], 0, sorted_averages.shape[0]], cmap=cmap, origin='lower')
    plt.colorbar(label='dF/F')
    plt.axvline(0, color='purple', linestyle='--', linewidth=2, label=align_line_label)
    plt.xlabel('Time (s)')
    plt.ylabel(f'Neuron (sorted by {sort_by})')
    plt.title(plot_title)
    plt.xlim(time_window)
    plt.legend()
    plt.grid(True)
    plt.show()



def kmeans_cluster_neurons(neuron_averages, k=5, normalize=True):
    print(f"[CLUSTERING] Performing k-means with k={k}.")

    # Fill NaNs and Infs with zeros
    neuron_averages_filled = np.nan_to_num(neuron_averages, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove rows that are still all zeros (invalid neurons)
    valid_mask = ~(np.all(neuron_averages_filled == 0, axis=1))
    neuron_averages_valid = neuron_averages_filled[valid_mask]
    valid_indices = np.arange(len(neuron_averages))[valid_mask]

    if len(neuron_averages_valid) == 0:
        print("[WARNING] No valid neurons for clustering.")
        return pd.DataFrame(columns=['neuron_idx', 'cluster'])

    if normalize:
        neuron_averages_valid = neuron_averages_valid / np.nanmax(np.abs(neuron_averages_valid), axis=1, keepdims=True)

    centroids, labels = kmeans2(neuron_averages_valid, k, minit='points')

    cluster_df = pd.DataFrame({
        'neuron_idx': valid_indices,
        'cluster': labels
    })
    print(f"[SUCCESS] k-means clustering complete. {len(valid_indices)} neurons clustered.")
    return cluster_df



# def plot_cluster_heatmap(common_time, neuron_averages, cluster_df, time_window, plot_title, align_line_label, events_to_plot, trials_df, custom_time_window=None):
#     plt.figure(figsize=(12, 8))
#     sorted_df = cluster_df.sort_values('cluster')
#     plt.imshow(neuron_averages[sorted_df['neuron_idx']], aspect='auto', extent=[common_time[0], common_time[-1], 0, neuron_averages.shape[0]], origin='lower', cmap='viridis')
#     plt.colorbar(label='dF/F')

#     for event_key, props in events_to_plot.items():
#         if event_key in trials_df.columns:
#             event_series = trials_df[event_key]
#             all_vlines = []
#             first_flash_starts = []
#             first_flash_ends = []
#             second_flash_starts = []
#             second_flash_ends = []

#             for flashes in event_series:
#                 if props['style'] == 'vline' and not np.isnan(flashes):
#                     all_vlines.append(flashes)
#                 elif props['style'] == 'bar' and isinstance(flashes, (list, tuple)) and len(flashes) >= 2:
#                     flash1 = flashes[0]
#                     flash2 = flashes[1]
#                     if not np.isnan(flash1[0]) and not np.isnan(flash1[1]):
#                         first_flash_starts.append(flash1[0])
#                         first_flash_ends.append(flash1[1])
#                     if not np.isnan(flash2[0]) and not np.isnan(flash2[1]):
#                         second_flash_starts.append(flash2[0])
#                         second_flash_ends.append(flash2[1])

#             if all_vlines:
#                 for t in all_vlines:
#                     plt.axvline(x=t, color=props.get('color', 'skyblue'), alpha=props.get('alpha', 0.1), linestyle='--')

#     ymin, ymax = plt.ylim()

#     if first_flash_starts and first_flash_ends:
#         plt.fill_betweenx([ymin, ymax], np.mean(first_flash_starts), np.mean(first_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)
#     if second_flash_starts and second_flash_ends:
#         plt.fill_betweenx([ymin, ymax], np.mean(second_flash_starts), np.mean(second_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)

#     plt.axvline(x=0, color='purple', linestyle='--', linewidth=2, label=align_line_label)
#     plt.title(plot_title)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Neuron (sorted by cluster)')
#     if custom_time_window:
#         plt.xlim(custom_time_window)
#     elif time_window:
#         plt.xlim(time_window)
#     else:
#         plt.xlim(common_time[0], common_time[-1])
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# def plot_cluster_averages(common_time, cluster_avg_traces, cluster_sem_traces, time_window, plot_title, align_line_label, events_to_plot, trials_df, custom_time_window=None):
#     plt.figure(figsize=(12, 8))
#     colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_avg_traces)))

#     for (idx, trace), (_, sem), color in zip(cluster_avg_traces.iterrows(), cluster_sem_traces.iterrows(), colors):
#         plt.plot(common_time, trace.values, label=f'Cluster {idx}', color=color)
#         plt.fill_between(common_time, trace.values - sem.values, trace.values + sem.values, color=color, alpha=0.3)

#     ymin, ymax = plt.ylim()

#     for event_key, props in events_to_plot.items():
#         if event_key in trials_df.columns:
#             event_series = trials_df[event_key]
#             all_vlines = []
#             first_flash_starts = []
#             first_flash_ends = []
#             second_flash_starts = []
#             second_flash_ends = []

#             for flashes in event_series:
#                 if props['style'] == 'vline' and not np.isnan(flashes):
#                     all_vlines.append(flashes)
#                 elif props['style'] == 'bar' and isinstance(flashes, (list, tuple)) and len(flashes) >= 2:
#                     flash1 = flashes[0]
#                     flash2 = flashes[1]
#                     if not np.isnan(flash1[0]) and not np.isnan(flash1[1]):
#                         first_flash_starts.append(flash1[0])
#                         first_flash_ends.append(flash1[1])
#                     if not np.isnan(flash2[0]) and not np.isnan(flash2[1]):
#                         second_flash_starts.append(flash2[0])
#                         second_flash_ends.append(flash2[1])

#             if all_vlines:
#                 for t in all_vlines:
#                     plt.axvline(x=t, color=props.get('color', 'skyblue'), alpha=props.get('alpha', 0.1), linestyle='--')

#     if first_flash_starts and first_flash_ends:
#         plt.fill_betweenx([ymin, ymax], np.mean(first_flash_starts), np.mean(first_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)
#     if second_flash_starts and second_flash_ends:
#         plt.fill_betweenx([ymin, ymax], np.mean(second_flash_starts), np.mean(second_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)

#     plt.axvline(x=0, color='purple', linestyle='--', linewidth=2, label=align_line_label)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Mean dF/F')
#     plt.title(plot_title)
#     if custom_time_window:
#         plt.xlim(custom_time_window)
#     elif time_window:
#         plt.xlim(time_window)
#     else:
#         plt.xlim(common_time[0], common_time[-1])
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def plot_cluster_heatmap(common_time, neuron_averages, cluster_df, time_window, plot_title, align_line_label, events_to_plot, trials_df, custom_time_window=None):
#     plt.figure(figsize=(12, 8))
#     sorted_df = cluster_df.sort_values('cluster')
#     plt.imshow(neuron_averages[sorted_df['neuron_idx']], aspect='auto', extent=[common_time[0], common_time[-1], 0, neuron_averages.shape[0]], origin='lower', cmap='viridis')
#     plt.colorbar(label='dF/F')

#     for event_key, props in events_to_plot.items():
#         if event_key in trials_df.columns:
#             event_series = trials_df[event_key]
#             all_vlines = []
#             first_flash_starts = []
#             first_flash_ends = []
#             second_flash_starts = []
#             second_flash_ends = []

#             for flashes in event_series:
#                 if props['style'] == 'vline' and not np.isnan(flashes):
#                     all_vlines.append(flashes)
#                 elif props['style'] == 'bar' and isinstance(flashes, (list, tuple)) and len(flashes) >= 2:
#                     flash1 = flashes[0]
#                     flash2 = flashes[1]
#                     if not np.isnan(flash1[0]) and not np.isnan(flash1[1]):
#                         first_flash_starts.append(flash1[0])
#                         first_flash_ends.append(flash1[1])
#                     if not np.isnan(flash2[0]) and not np.isnan(flash2[1]):
#                         second_flash_starts.append(flash2[0])
#                         second_flash_ends.append(flash2[1])

#             if all_vlines:
#                 for t in all_vlines:
#                     plt.axvline(x=t, color=props.get('color', 'skyblue'), alpha=props.get('alpha', 0.1), linestyle='--')

#     ymin, ymax = plt.ylim()

#     if first_flash_starts and first_flash_ends:
#         plt.fill_betweenx([ymin, ymax], np.mean(first_flash_starts), np.mean(first_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)
#     if second_flash_starts and second_flash_ends:
#         plt.fill_betweenx([ymin, ymax], np.mean(second_flash_starts), np.mean(second_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)

#     plt.axvline(x=0, color='purple', linestyle='--', linewidth=2, label=align_line_label)
#     plt.title(plot_title)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Neuron (sorted by cluster)')
#     if custom_time_window is not None:
#         plt.xlim(custom_time_window)
#     elif time_window is not None:
#         plt.xlim(time_window)
#     else:
#         plt.xlim(common_time[0], common_time[-1])
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# def plot_cluster_averages(common_time, cluster_avg_traces, cluster_sem_traces, time_window, plot_title, align_line_label, events_to_plot, trials_df, custom_time_window=None):
#     plt.figure(figsize=(12, 8))
#     colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_avg_traces)))

#     for (idx, trace), (_, sem), color in zip(cluster_avg_traces.iterrows(), cluster_sem_traces.iterrows(), colors):
#         plt.plot(common_time, trace.values, label=f'Cluster {idx}', color=color)
#         plt.fill_between(common_time, trace.values - sem.values, trace.values + sem.values, color=color, alpha=0.3)

#     ymin, ymax = plt.ylim()

#     for event_key, props in events_to_plot.items():
#         if event_key in trials_df.columns:
#             event_series = trials_df[event_key]
#             all_vlines = []
#             first_flash_starts = []
#             first_flash_ends = []
#             second_flash_starts = []
#             second_flash_ends = []

#             for flashes in event_series:
#                 if props['style'] == 'vline' and not np.isnan(flashes):
#                     all_vlines.append(flashes)
#                 elif props['style'] == 'bar' and isinstance(flashes, (list, tuple)) and len(flashes) >= 2:
#                     flash1 = flashes[0]
#                     flash2 = flashes[1]
#                     if not np.isnan(flash1[0]) and not np.isnan(flash1[1]):
#                         first_flash_starts.append(flash1[0])
#                         first_flash_ends.append(flash1[1])
#                     if not np.isnan(flash2[0]) and not np.isnan(flash2[1]):
#                         second_flash_starts.append(flash2[0])
#                         second_flash_ends.append(flash2[1])

#             if all_vlines:
#                 for t in all_vlines:
#                     plt.axvline(x=t, color=props.get('color', 'skyblue'), alpha=props.get('alpha', 0.1), linestyle='--')

#     if first_flash_starts and first_flash_ends:
#         plt.fill_betweenx([ymin, ymax], np.mean(first_flash_starts), np.mean(first_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)
#     if second_flash_starts and second_flash_ends:
#         plt.fill_betweenx([ymin, ymax], np.mean(second_flash_starts), np.mean(second_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)

#     plt.axvline(x=0, color='purple', linestyle='--', linewidth=2, label=align_line_label)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Mean dF/F')
#     plt.title(plot_title)
#     if custom_time_window is not None:
#         plt.xlim(custom_time_window)
#     elif time_window is not None:
#         plt.xlim(time_window)
#     else:
#         plt.xlim(common_time[0], common_time[-1])
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def plot_cluster_heatmap(common_time, neuron_averages, cluster_df, time_window, plot_title, align_line_label, events_to_plot, trials_df, custom_time_window=None):
#     plt.figure(figsize=(12, 8))
#     sorted_df = cluster_df.sort_values('cluster')
#     plt.imshow(neuron_averages[sorted_df['neuron_idx']], aspect='auto', extent=[common_time[0], common_time[-1], 0, neuron_averages.shape[0]], origin='lower', cmap='viridis')
#     plt.colorbar(label='dF/F')

#     for event_key, props in events_to_plot.items():
#         if event_key in trials_df.columns:
#             event_series = trials_df[event_key]
#             all_vlines = []
#             first_flash_starts = []
#             first_flash_ends = []
#             second_flash_starts = []
#             second_flash_ends = []

#             for flashes in event_series:
#                 if props['style'] == 'vline' and not np.isnan(flashes):
#                     all_vlines.append(flashes)
#                 elif props['style'] == 'bar' and isinstance(flashes, (list, tuple)) and len(flashes) >= 2:
#                     flash1 = flashes[0]
#                     flash2 = flashes[1]
#                     if not np.isnan(flash1[0]) and not np.isnan(flash1[1]):
#                         first_flash_starts.append(flash1[0])
#                         first_flash_ends.append(flash1[1])
#                     if not np.isnan(flash2[0]) and not np.isnan(flash2[1]):
#                         second_flash_starts.append(flash2[0])
#                         second_flash_ends.append(flash2[1])

#             if all_vlines:
#                 for t in all_vlines:
#                     plt.axvline(x=t, color=props.get('color', 'skyblue'), alpha=props.get('alpha', 0.1), linestyle='--')

#     ymin, ymax = plt.ylim()

#     if first_flash_starts and first_flash_ends:
#         plt.fill_betweenx([ymin, ymax], np.mean(first_flash_starts), np.mean(first_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)
#     if second_flash_starts and second_flash_ends:
#         plt.fill_betweenx([ymin, ymax], np.mean(second_flash_starts), np.mean(second_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)

#     plt.axvline(x=0, color='purple', linestyle='--', linewidth=2, label=align_line_label)
#     plt.title(plot_title)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Neuron (sorted by cluster)')
#     if custom_time_window is not None:
#         plt.xlim(custom_time_window)
#     else:
#         plt.xlim(common_time[~np.isnan(common_time)][0], common_time[~np.isnan(common_time)][-1])
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# def plot_cluster_averages(common_time, cluster_avg_traces, cluster_sem_traces, time_window, plot_title, align_line_label, events_to_plot, trials_df, custom_time_window=None):
#     plt.figure(figsize=(12, 8))
#     colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_avg_traces)))

#     for (idx, trace), (_, sem), color in zip(cluster_avg_traces.iterrows(), cluster_sem_traces.iterrows(), colors):
#         plt.plot(common_time, trace.values, label=f'Cluster {idx}', color=color)
#         plt.fill_between(common_time, trace.values - sem.values, trace.values + sem.values, color=color, alpha=0.3)

#     ymin, ymax = plt.ylim()

#     for event_key, props in events_to_plot.items():
#         if event_key in trials_df.columns:
#             event_series = trials_df[event_key]
#             all_vlines = []
#             first_flash_starts = []
#             first_flash_ends = []
#             second_flash_starts = []
#             second_flash_ends = []

#             for flashes in event_series:
#                 if props['style'] == 'vline' and not np.isnan(flashes):
#                     all_vlines.append(flashes)
#                 elif props['style'] == 'bar' and isinstance(flashes, (list, tuple)) and len(flashes) >= 2:
#                     flash1 = flashes[0]
#                     flash2 = flashes[1]
#                     if not np.isnan(flash1[0]) and not np.isnan(flash1[1]):
#                         first_flash_starts.append(flash1[0])
#                         first_flash_ends.append(flash1[1])
#                     if not np.isnan(flash2[0]) and not np.isnan(flash2[1]):
#                         second_flash_starts.append(flash2[0])
#                         second_flash_ends.append(flash2[1])

#             if all_vlines:
#                 for t in all_vlines:
#                     plt.axvline(x=t, color=props.get('color', 'skyblue'), alpha=props.get('alpha', 0.1), linestyle='--')

#     if first_flash_starts and first_flash_ends:
#         plt.fill_betweenx([ymin, ymax], np.mean(first_flash_starts), np.mean(first_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)
#     if second_flash_starts and second_flash_ends:
#         plt.fill_betweenx([ymin, ymax], np.mean(second_flash_starts), np.mean(second_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)

#     plt.axvline(x=0, color='purple', linestyle='--', linewidth=2, label=align_line_label)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Mean dF/F')
#     plt.title(plot_title)
#     if custom_time_window is not None:
#         plt.xlim(custom_time_window)
#     else:
#         plt.xlim(common_time[~np.isnan(common_time)][0], common_time[~np.isnan(common_time)][-1])
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def compute_cluster_averages(common_time, neuron_averages, cluster_df):
    print("[COMPUTING] Cluster average traces.")

    merged = pd.DataFrame(neuron_averages)
    merged['neuron_idx'] = np.arange(len(neuron_averages))
    merged = merged.merge(cluster_df, on='neuron_idx', how='inner')

    # Drop 'neuron_idx' before averaging
    merged = merged.drop(columns=['neuron_idx'])

    cluster_avg_traces = merged.groupby('cluster').mean()
    cluster_sem_traces = merged.groupby('cluster').sem()

    print(f"[SUCCESS] Computed averages for {cluster_avg_traces.shape[0]} clusters.")
    return common_time, cluster_avg_traces, cluster_sem_traces



def plot_events_on_raster(event_times, ymin, ymax, color='black', alpha=0.5, linestyle='--', label=None, style='vline'):
    if event_times is None:
        return

    if style == 'vline':
        for t in event_times:
            if np.isnan(t):
                continue
            plt.plot([t, t], [ymin, ymax], color=color, alpha=alpha, linestyle=linestyle, linewidth=1.5, zorder=3, label=label)
            label = None  # Only label first occurrence
    elif style == 'bar':
        for (start, end) in event_times:
            plt.fill_betweenx([ymin, ymax], start, end, color=color, alpha=alpha, zorder=1)
            label = None

def plot_events_on_heatmap(event_times, ymin, ymax, color='black', alpha=0.5, linestyle='--', label=None, style='vline'):
    if event_times is None:
        return

    if style == 'vline':
        for t in event_times:
            if np.isnan(t):
                continue
            plt.axvline(x=t, color=color, alpha=alpha, linestyle=linestyle, linewidth=1.5, zorder=3, label=label)
            label = None
    elif style == 'bar':
        for (start, end) in event_times:
            plt.axvspan(start, end, ymin=0, ymax=1, color=color, alpha=alpha, zorder=1)
            label = None

def filter_trials_by_condition(trials_df, condition_func):
    return trials_df[condition_func(trials_df)].reset_index(drop=True)

def save_plot(filename, folder="plots"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, filename), bbox_inches='tight')
    print(f"[SAVED] Plot saved to {folder}/{filename}")

def generate_summary_report(trials_df, cluster_df=None, output_folder="plots"):
    print("[SUMMARY] Generating session summary.")

    summary_text = f"""
    Session Summary Report
    =======================
    
    Total Trials: {len(trials_df)}
    
    Available Columns:
    {list(trials_df.columns)}
    
    Trials by Choice:
    {trials_df['mouse_choice'].value_counts(dropna=False).to_string()}
    
    Trials by Reward:
    {trials_df['rewarded'].value_counts(dropna=False).to_string()}
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    summary_path = os.path.join(output_folder, "session_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(f"[SAVED] Summary report saved to {summary_path}")

    # Plot pie charts
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    trials_df['mouse_choice'].value_counts(dropna=False).plot.pie(autopct='%1.1f%%')
    plt.title('Choices')

    plt.subplot(1, 2, 2)
    trials_df['rewarded'].value_counts(dropna=False).plot.pie(autopct='%1.1f%%')
    plt.title('Rewarded')

    save_plot("summary_pies.png", folder=output_folder)
    plt.close()

    if cluster_df is not None and not cluster_df.empty:
        plt.figure(figsize=(6, 6))
        cluster_df['cluster'].value_counts().sort_index().plot.pie(autopct='%1.1f%%')
        plt.title('Neurons per Cluster')
        save_plot("cluster_pie.png", folder=output_folder)
        plt.close()


def plot_cluster_heatmap(common_time, neuron_averages, cluster_df, time_window, plot_title, align_line_label, events_to_plot, trials_df, custom_time_window=None):
    plt.figure(figsize=(12, 8))
    sorted_df = cluster_df.sort_values('cluster')
    plt.imshow(neuron_averages[sorted_df['neuron_idx']], aspect='auto', extent=[common_time[0], common_time[-1], 0, neuron_averages.shape[0]], origin='lower', cmap='viridis')
    plt.colorbar(label='dF/F')

    for event_key, props in events_to_plot.items():
        if event_key in trials_df.columns:
            event_series = trials_df[event_key]
            all_vlines = []
            first_flash_starts = []
            first_flash_ends = []
            second_flash_starts = []
            second_flash_ends = []

            for flashes in event_series:
                if props['style'] == 'vline' and not np.isnan(flashes):
                    all_vlines.append(flashes)
                elif props['style'] == 'bar' and isinstance(flashes, (list, tuple)) and len(flashes) >= 2:
                    flash1 = flashes[0]
                    flash2 = flashes[1]
                    if not np.isnan(flash1[0]) and not np.isnan(flash1[1]):
                        first_flash_starts.append(flash1[0])
                        first_flash_ends.append(flash1[1])
                    if not np.isnan(flash2[0]) and not np.isnan(flash2[1]):
                        second_flash_starts.append(flash2[0])
                        second_flash_ends.append(flash2[1])

            if all_vlines:
                for t in all_vlines:
                    plt.axvline(x=t, color=props.get('color', 'skyblue'), alpha=props.get('alpha', 0.1), linestyle='--')

    ymin, ymax = plt.ylim()

    if first_flash_starts and first_flash_ends:
        plt.fill_betweenx([ymin, ymax], np.mean(first_flash_starts), np.mean(first_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)
    if second_flash_starts and second_flash_ends:
        plt.fill_betweenx([ymin, ymax], np.mean(second_flash_starts), np.mean(second_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)

    plt.axvline(x=0, color='purple', linestyle='--', linewidth=2, label=align_line_label)
    plt.title(plot_title)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron (sorted by cluster)')
    if custom_time_window is not None:
        plt.xlim(custom_time_window)
    else:
        plt.xlim(common_time[~np.isnan(common_time)][0], common_time[~np.isnan(common_time)][-1])
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_cluster_averages(common_time, cluster_avg_traces, cluster_sem_traces, time_window, plot_title, align_line_label, events_to_plot, trials_df, custom_time_window=None):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_avg_traces)))

    for (idx, trace), (_, sem), color in zip(cluster_avg_traces.iterrows(), cluster_sem_traces.iterrows(), colors):
        plt.plot(common_time, trace.values, label=f'Cluster {idx}', color=color)
        plt.fill_between(common_time, trace.values - sem.values, trace.values + sem.values, color=color, alpha=0.3)

    ymin, ymax = plt.ylim()

    for event_key, props in events_to_plot.items():
        if event_key in trials_df.columns:
            event_series = trials_df[event_key]
            all_vlines = []
            first_flash_starts = []
            first_flash_ends = []
            second_flash_starts = []
            second_flash_ends = []

            for flashes in event_series:
                if props['style'] == 'vline' and not np.isnan(flashes):
                    all_vlines.append(flashes)
                elif props['style'] == 'bar' and isinstance(flashes, (list, tuple)) and len(flashes) >= 2:
                    flash1 = flashes[0]
                    flash2 = flashes[1]
                    if not np.isnan(flash1[0]) and not np.isnan(flash1[1]):
                        first_flash_starts.append(flash1[0])
                        first_flash_ends.append(flash1[1])
                    if not np.isnan(flash2[0]) and not np.isnan(flash2[1]):
                        second_flash_starts.append(flash2[0])
                        second_flash_ends.append(flash2[1])

            if all_vlines:
                for t in all_vlines:
                    plt.axvline(x=t, color=props.get('color', 'skyblue'), alpha=props.get('alpha', 0.1), linestyle='--')

    if first_flash_starts and first_flash_ends:
        plt.fill_betweenx([ymin, ymax], np.mean(first_flash_starts), np.mean(first_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)
    if second_flash_starts and second_flash_ends:
        plt.fill_betweenx([ymin, ymax], np.mean(second_flash_starts), np.mean(second_flash_ends), color=props.get('color', 'skyblue'), alpha=0.3)

    plt.axvline(x=0, color='purple', linestyle='--', linewidth=2, label=align_line_label)
    plt.xlabel('Time (s)')
    plt.ylabel('Mean dF/F')
    plt.title(plot_title)
    if custom_time_window is not None:
        plt.xlim(custom_time_window)
    else:
        plt.xlim(common_time[~np.isnan(common_time)][0], common_time[~np.isnan(common_time)][-1])
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Full report generation function ---

def generate_full_report(trials_df, conditions, alignments, events_to_plot, k_clusters=5, time_window=[-10, 10], custom_time_windows=None):
    for condition_name, condition_func in conditions.items():
        trials_filtered = trials_df[condition_func(trials_df)]

        for alignment_name, align_col in alignments.items():
            
            if alignment_name == 'choice_window_start':
                print('')
            if alignment_name == 'First Lick Aligned':
                print('')                    
                
                
            
            aligned_trials = align_trials(trials_filtered, align_col)
            
            debug_align_trials(aligned_trials, align_to_time_column=align_col)
            
            if alignment_name == 'choice_window_start':
                print('')
            if alignment_name == 'First Lick Aligned':
                print('')                  

            common_time, neuron_averages = compute_neuron_averages(aligned_trials, time_window=None)
            cluster_df = kmeans_cluster_neurons(neuron_averages, k=k_clusters, normalize=True)
            plot_title = f"Clustered Heatmap - {condition_name} - {alignment_name}"

            custom_window = None
            if custom_time_windows:
                custom_window = custom_time_windows.get(alignment_name, None)

            debug_plot_cluster_heatmap(common_time, neuron_averages, cluster_df)

            plot_cluster_heatmap(
                common_time,
                neuron_averages,
                cluster_df,
                time_window,
                plot_title,
                align_line_label=alignment_name,
                events_to_plot=events_to_plot,
                trials_df=aligned_trials,
                custom_time_window=custom_window
            )

            common_time, cluster_avg_traces, cluster_sem_traces = compute_cluster_averages(common_time, neuron_averages, cluster_df)

            debug_plot_cluster_averages(common_time, cluster_avg_traces)

            plot_cluster_averages(
                common_time,
                cluster_avg_traces,
                cluster_sem_traces,
                time_window,
                plot_title=f"Cluster Averages - {condition_name} - {alignment_name}",
                align_line_label=alignment_name,
                events_to_plot=events_to_plot,
                trials_df=aligned_trials,
                custom_time_window=custom_window
            )

# --- Debug hooks for full report pipeline ---

def debug_align_trials(trials_df, align_to_time_column='start_time'):
    print(f"[DEBUG] Aligning trials to column: {align_to_time_column}")
    aligned_times = trials_df[align_to_time_column].dropna()
    print(f"[DEBUG] Alignment times (min, max): {aligned_times.min()}, {aligned_times.max()}")


def debug_plot_cluster_heatmap(common_time, neuron_averages, cluster_df):
    print(f"[DEBUG] Plotting heatmap:")
    print(f"[DEBUG] Common time range: {common_time[0]:.2f} to {common_time[-1]:.2f}")
    print(f"[DEBUG] Neuron averages shape: {neuron_averages.shape}")
    print(f"[DEBUG] Cluster assignments: {cluster_df['cluster'].value_counts().to_dict()}")


def debug_plot_cluster_averages(common_time, cluster_avg_traces):
    print(f"[DEBUG] Plotting cluster averages:")
    print(f"[DEBUG] Common time range: {common_time[0]:.2f} to {common_time[-1]:.2f}")
    print(f"[DEBUG] Cluster average traces shape: {cluster_avg_traces.shape}")

# --- New functions to compare same neurons across conditions ---

def extract_cluster_neuron_ids(cluster_df):
    cluster_to_neurons = cluster_df.groupby('cluster')['neuron_idx'].apply(list).to_dict()
    print(f"[DEBUG] Extracted neuron IDs per cluster: { {k: len(v) for k,v in cluster_to_neurons.items()} }")
    return cluster_to_neurons

# def extract_neuron_traces(neuron_averages, neuron_ids):
#     return neuron_averages[neuron_ids, :]

def extract_neuron_traces(neuron_averages, neuron_ids, normalize=False):
    traces = neuron_averages[neuron_ids, :]
    if normalize:
        traces = (traces - np.nanmean(traces, axis=1, keepdims=True)) / (np.nanstd(traces, axis=1, keepdims=True) + 1e-6)
    return traces

def plot_traces_by_cluster(common_time, neuron_traces_dict, title_prefix=""):
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(neuron_traces_dict)))

    for (cluster_id, traces), color in zip(neuron_traces_dict.items(), colors):
        mean_trace = np.nanmean(traces, axis=0)
        sem_trace = np.nanstd(traces, axis=0) / np.sqrt(traces.shape[0])

        plt.plot(common_time, mean_trace, label=f"Cluster {cluster_id}", color=color)
        plt.fill_between(common_time, mean_trace - sem_trace, mean_trace + sem_trace, color=color, alpha=0.3)

    plt.axvline(x=0, color='purple', linestyle='--', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("dF/F")
    plt.title(f"{title_prefix} Neuron Traces by Cluster")
    plt.legend()
    plt.grid(True)
    plt.show()

def overlay_traces_by_cluster(common_time, neuron_traces_left, neuron_traces_right, title_prefix=""):
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(neuron_traces_left)))

    for (cluster_id, traces_left), (_, traces_right), color in zip(neuron_traces_left.items(), neuron_traces_right.items(), colors):
        mean_trace_left = np.nanmean(traces_left, axis=0)
        sem_trace_left = np.nanstd(traces_left, axis=0) / np.sqrt(traces_left.shape[0])

        mean_trace_right = np.nanmean(traces_right, axis=0)
        sem_trace_right = np.nanstd(traces_right, axis=0) / np.sqrt(traces_right.shape[0])

        plt.plot(common_time, mean_trace_left, label=f"Cluster {cluster_id} Left", color=color, linestyle='-')
        plt.plot(common_time, mean_trace_right, label=f"Cluster {cluster_id} Right", color=color, linestyle='--')

    plt.axvline(x=0, color='purple', linestyle='--', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("dF/F")
    plt.title(f"{title_prefix} Cluster Overlay Left vs Right")
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_reward_punish(trials_df, clusters_neuron_ids, align_to='choice_start'):
    comparisons = {
        'left_rewarded': (trials_df['trial_side'] == 'left') & (trials_df['rewarded'] == 1),
        'left_punished': (trials_df['trial_side'] == 'left') & (trials_df['rewarded'] == 0),
        'right_rewarded': (trials_df['trial_side'] == 'right') & (trials_df['rewarded'] == 1),
        'right_punished': (trials_df['trial_side'] == 'right') & (trials_df['rewarded'] == 0)
    }

    extracted = {}

    for label, mask in comparisons.items():
        if label == 'right_punished':
            print('')
        
        subset = trials_df[mask]
        aligned = align_trials(subset, align_to_time_column=align_to)
        common_time, averages = compute_neuron_averages(aligned)



        traces_dict = {}
        for cluster_id, neuron_ids in clusters_neuron_ids.items():
            # traces = extract_neuron_traces(averages, neuron_ids, normalize=True)
            traces = extract_neuron_traces(averages, neuron_ids, normalize=True)
            traces_dict[cluster_id] = traces

        extracted[label] = (common_time, traces_dict)

    return extracted

def plot_comparison_overlays(extracted, clusters_to_plot=None):
    for cluster_id in clusters_to_plot if clusters_to_plot else extracted[list(extracted.keys())[0]][1].keys():
        plt.figure(figsize=(14, 8))
        colors = {'left_rewarded': 'blue', 'left_punished': 'cyan', 'right_rewarded': 'red', 'right_punished': 'orange'}

        for label, (common_time, traces_dict) in extracted.items():
            if cluster_id in traces_dict:
                mean_trace = np.nanmean(traces_dict[cluster_id], axis=0)
                plt.plot(common_time, mean_trace, label=label, color=colors.get(label, 'black'))

        plt.axvline(x=0, color='purple', linestyle='--', linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized dF/F")
        plt.title(f"Cluster {cluster_id} Comparison Across Trial Types")
        plt.legend()
        plt.grid(True)
        plt.show()

def neural_report(M, config, subjectIdx, sessionIdx=-1, figure_id=None, show_plot=1):
    print('Compiling neural report...')
    
    
    subject = M['subject']
    dates = M['dates']
    print(f"\n📦 Startingneural report construction for subject: {subject}")
    print(f"📅 Sessions to process: {dates}\n")

    # ------------------------------
    # Step 1: Load + preprocess session data
    # ------------------------------
    print("🔍 Preprocessing session data...")
    session_dfs = []
    for i in range(len(dates)):
        print(f"  - Processing session {i} ({dates[i]})")
        df = get_session_df(M, i)
        df['session_id'] = i
        df['mouse_id'] = subject
        df['date'] = dates[i]
        df = df[df['trial_stop'] <= 13].reset_index(drop=True)  # drop trials that are longer than should be
        df_proc = add_model_features(df)

        


        # Check for missing features
        feature_cols = [
            'is_right', 'stim_duration', 'is_opto', 'norm_trial_index',
            'response_time', 'rolling_accuracy',
            'choice_1back', 'choice_2back', 'choice_3back',
            'reward_1back', 'reward_2back', 'reward_3back',
            'opto_1back', 'opto_2back', 'opto_3back',
            'stay_from_1back', 'rolling_choice_bias', 'exp_choice_bias',
            'isi_opto_interaction'
        ]
        missing_features = [f for f in feature_cols if f not in df_proc.columns]
        assert not missing_features, f"❌ Missing features in session {i}: {missing_features}"

        session_dfs.append(df_proc)

    # Combine all sessions
    trials_df = pd.concat(session_dfs, ignore_index=True)
    print(f"✅ Combined all sessions into one DataFrame: {trials_df.shape}")

    # Paths (adjust to your file structure)
    
    suite2p_path = 'D:/git/2p_imaging/passive_interval_oddball_202412/results/TS03/TS03_CRBL_20250424_2AFC'
    
    # Load DFF
    print(f"[LOADING] dff.h5 from {suite2p_path}")
    with h5py.File(os.path.join(suite2p_path, 'dff.h5'), 'r') as f:
        print(f"[INFO] Available datasets in dff.h5: {list(f.keys())}")
        dff = f['dff'][:]
    print(f"[SUCCESS] Loaded dff: shape {dff.shape}, dtype {dff.dtype}")
    
    # Load Masks
    print(f"[LOADING] masks.h5 from {suite2p_path}")
    with h5py.File(os.path.join(suite2p_path, 'masks.h5'), 'r') as f:
        print(f"[INFO] Available datasets in masks.h5: {list(f.keys())}")
        masks = f['masks_func'][:]
    print(f"[SUCCESS] Loaded masks: shape {masks.shape}, dtype {masks.dtype}")
    
    # Load Move Offset
    with h5py.File(os.path.join(suite2p_path, 'move_offset.h5'), 'r') as f:
        print(f"[INFO] Available datasets in move_offset.h5: {list(f.keys())}")
        xoff = f['xoff'][:]
        yoff = f['yoff'][:]
    print(f"[SUCCESS] Loaded xoff: shape {xoff.shape}, dtype {xoff.dtype}")
    print(f"[SUCCESS] Loaded yoff: shape {yoff.shape}, dtype {yoff.dtype}")

    # Load Ops
    print(f"[LOADING] ops.npy from {suite2p_path}")
    ops = np.load(os.path.join(suite2p_path, 'ops.npy'), allow_pickle=True).item()
    print(f"[SUCCESS] Loaded ops: keys = {list(ops.keys())}")
    
    # Load Raw Voltages
    print(f"[LOADING] raw_voltages.h5 from {suite2p_path}")
    with h5py.File(os.path.join(suite2p_path, 'raw_voltages.h5'), 'r') as f:
        print(f"[INFO] Available datasets in raw_voltages.h5: {list(f['raw'].keys())}")
        raw_voltages = {k: f['raw'][k][:] for k in f['raw'].keys()}
    print(f"[SUCCESS] Loaded raw_voltages: keys = {list(raw_voltages.keys())}")


    # Load Bruker xml file
    print(f"[LOADING] TS03_CRBL_20250424_2AFC-159.xml from {suite2p_path}")
    # with h5py.File(os.path.join(suite2p_path, 'TS03_CRBL_20250424_2AFC-159_Cycle00001_VoltageRecording_001.xml'), 'r') as f:
    #     # print(f"[INFO] Available datasets in raw_voltages.h5: {list(f['raw'].keys())}")
    xml_file = os.path.join(suite2p_path, 'TS03_CRBL_20250424_2AFC-159.xml')
    # explore_large_xml(xml_file, 3000)
    frame_times = load_bruker_frame_times(xml_file, ops)
    frame_num = ops['nframes']

    # neu data tags
    # vol_2p_stim
    # vol_flir
    # vol_hifi
    # vol_img
    # vol_led
    # vol_pmt
    # vol_start
    # vol_stim_aud
    # vol_stim_vis
    # vol_time
    
    voltage_time = np.array(raw_voltages['vol_time'])
    voltage_sync_signal = np.array(raw_voltages['vol_start'])
    voltage_stim = np.array(raw_voltages['vol_stim_vis'])
    voltage_etl = np.array(raw_voltages['vol_img'])    
    
    plt.figure(figsize=(12, 6))
    plt.plot(voltage_time, voltage_sync_signal)
    plt.plot(voltage_time, voltage_stim)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage Sync Signal')
    plt.title('Voltage Sync Signal Over Time')
    plt.xlim([0, 50000])
    plt.grid(True)
    plt.show()
    
    trials_df = segment_trials_with_voltage_sync(dff, frame_times, trials_df, voltage_time, voltage_sync_signal, voltage_stim, threshold=0.5, ops=None, bruker_xml_path=None)
    
        # raw_voltages = {k: f['raw'][k][:] for k in f['raw'].keys()}
        
    # print(f"[SUCCESS] Loaded raw_voltages: keys = {list(raw_voltages.keys())}")    

    # trials_df = align_trials(trials_df, align_to_event='choice_start')
    
    # align to
    # choice_start
    # stim_start_bpod
    # stim_stop_bpod
    
    # Define your filtering conditions
    # conditions = {
    #     'all_trials': lambda df: pd.Series(True, index=df.index),
    #     'left_choices': lambda df: df['mouse_choice'] == 0,
    #     'right_choices': lambda df: df['mouse_choice'] == 1,
    #     'rewarded_trials': lambda df: df['rewarded'] == 1,
    # }
    # conditions = {
    #     # 'all_trials': lambda df: pd.Series(True, index=df.index),
    #     # 'left_choices': lambda df: df['mouse_choice'] == 0,
    #     # 'right_choices': lambda df: df['mouse_choice'] == 1,
    #     # 'rewarded_trials': lambda df: df['rewarded'] == 1,
    #     'left_rewarded': lambda df: (df['trial_side'] == 'left') & (df['rewarded'] == 1),
    #     'right_rewarded': lambda df: (df['trial_side'] == 'right') & (df['rewarded'] == 1),
    #     # 'left_not_rewarded': lambda df: (df['mouse_choice'] == 0) & (df['rewarded'] == 0),
    #     # 'right_not_rewarded': lambda df: (df['mouse_choice'] == 1) & (df['rewarded'] == 0),
    # }    
    
    # # Define your alignments
    # alignments = {
    #     'first_flash_start': 'stim_start_bpod',
    #     'choice_window_start': 'choice_start',
    #     'first_lick_start': 'lick_time_s',
    #     'reward_start': 'reward_start',
    # }
    
    # # Define your event overlays
    # events_to_plot = {
    #     'aligned_flash_times': {'color': 'blue', 'alpha': 0.2, 'style': 'bar'},
    #     'lick_time_aligned': {'color': 'black', 'alpha': 0.7, 'style': 'vline'},
    #     'reward_start_aligned': {'color': 'green', 'alpha': 0.8, 'style': 'vline'},
    #     'punish_start_aligned': {'color': 'red', 'alpha': 0.8, 'style': 'vline'},
    # }
    
    # # Run the full report
    # generate_full_report(
    #     trials_df=trials_df,
    #     conditions=conditions,
    #     alignments=alignments,
    #     events_to_plot=events_to_plot,
    #     k_clusters=5,
    #     time_window=[-7, 8]
    # )
    
    
    # Example to call generate_full_report
    
    # LEFT
    
    conditions = {
        'Left Rewarded': lambda df: (df['trial_side'] == 'left') & (df['rewarded'] == 1),
        # 'right_rewarded': lambda df: (df['trial_side'] == 'right') & (df['rewarded'] == 1),
    }
    
    alignments = {
        'Vis Stim Aligned': 'stim_start_bpod',
        'Choice Window Aligned': 'choice_start',
        'First Lick Aligned': 'choice_stop',
        'Reward Aligned': 'reward_start',
    }
    
    events_to_plot = {
        'aligned_flash_times': {'color': 'blue', 'alpha': 0.2, 'style': 'bar'},
        'lick_time_aligned': {'color': 'black', 'alpha': 0.7, 'style': 'vline'},
        'reward_start_aligned': {'color': 'green', 'alpha': 0.8, 'style': 'vline'},
        'punish_start_aligned': {'color': 'red', 'alpha': 0.8, 'style': 'vline'},
    }
    
    custom_time_windows = {
        'Vis Stim Aligned': (-1, 9),
        'Choice Window Aligned': (-4, 7),
        'First Lick Aligned': (-2, 6),
        'Reward Aligned': (-4, 6),
    }
    
    custom_time_windows = {}
    
    generate_full_report(
        trials_df=trials_df,
        conditions=conditions,
        alignments=alignments,
        events_to_plot=events_to_plot,
        k_clusters=5,
        time_window=[-10, 10],
    )       
    
    # # Run the full report
    # generate_full_report(
    #     trials_df=trials_df,
    #     conditions=conditions,
    #     alignments=alignments,
    #     events_to_plot=events_to_plot,
    #     k_clusters=5,
    #     # time_window=[-7, 8],
    #     # custom_time_windows=custom_time_windows
    # )    
    
    # RIGHT
    
    # Example to call generate_full_report
    conditions = {
        # 'left_rewarded': lambda df: (df['trial_side'] == 'left') & (df['rewarded'] == 1),
        'right_rewarded': lambda df: (df['trial_side'] == 'right') & (df['rewarded'] == 1),
    }
    
    alignments = {
        'first_flash_start': 'stim_start_bpod',
        'choice_window_start': 'choice_start',
        'first_lick_start': 'choice_stop',
        'reward_start': 'reward_start',
    }
    
    events_to_plot = {
        'aligned_flash_times': {'color': 'blue', 'alpha': 0.2, 'style': 'bar'},
        'lick_time_aligned': {'color': 'black', 'alpha': 0.7, 'style': 'vline'},
        'reward_start_aligned': {'color': 'green', 'alpha': 0.8, 'style': 'vline'},
        'punish_start_aligned': {'color': 'red', 'alpha': 0.8, 'style': 'vline'},
    }
    
    custom_time_windows = {
        'first_flash_start': (-1, 7),
        'choice_window_start': (-2, 6),
        'first_lick_start': (-1, 7),
        'reward_start': (-2, 6),
    }
       
    # Run the full report
    generate_full_report(
        trials_df=trials_df,
        conditions=conditions,
        alignments=alignments,
        events_to_plot=events_to_plot,
        k_clusters=5,
        # time_window=[-7, 8],
        custom_time_windows=custom_time_windows
    ) 
    
    
    # REWARDED LEFT AND RIGHT, SAME ROIS CLUSTERED, STIM START
    
    # --- Step 1: Filter Left Rewarded Trials ---
    left_trials_df = trials_df[(trials_df['trial_side'] == 'left') & (trials_df['rewarded'] == 1)]
    
    # --- Step 2: Align and Compute Averages for Left ---
    aligned_left = align_trials(left_trials_df, align_to_time_column='stim_start_bpod')
    common_time_left, neuron_averages_left = compute_neuron_averages(aligned_left)
    
    # --- Step 3: Cluster on Left Trials ---
    cluster_df_left = kmeans_cluster_neurons(neuron_averages_left, k=5)
    
    # --- Step 4: Save Cluster Memberships ---
    clusters_neuron_ids = extract_cluster_neuron_ids(cluster_df_left)
    
    # --- Step 5: Extract Neuron Traces for Left Clusters ---
    neuron_traces_left = {}
    
    for cluster_id, neuron_ids in clusters_neuron_ids.items():
        traces = extract_neuron_traces(neuron_averages_left, neuron_ids)
        neuron_traces_left[cluster_id] = traces
    
    # --- Step 6: Filter Right Rewarded Trials ---
    right_trials_df = trials_df[(trials_df['trial_side'] == 'right') & (trials_df['rewarded'] == 1)]
    
    # --- Step 7: Align and Compute Averages for Right ---
    # aligned_right = align_trials(right_trials_df, align_to_time_column='stim_start_bpod')
    aligned_right = align_trials(right_trials_df, align_to_time_column='stim_start_bpod')
    common_time_right, neuron_averages_right = compute_neuron_averages(aligned_right)
    
    # --- Step 8: Extract Neuron Traces for Same Neurons on Right ---
    neuron_traces_right = {}
    
    for cluster_id, neuron_ids in clusters_neuron_ids.items():
        traces = extract_neuron_traces(neuron_averages_right, neuron_ids)
        neuron_traces_right[cluster_id] = traces
    
    # --- Step 9: Overlay Traces Left vs Right ---
    overlay_traces_by_cluster(
        common_time_left,  # or common_time_right (should be close)
        neuron_traces_left,
        neuron_traces_right,
        title_prefix="Rewarded - Stim Start Aligned -"
    )


    # REWARDED LEFT AND RIGHT, SAME ROIS CLUSTERED, Choice START

    # --- Step 1: Filter Left Rewarded Trials ---
    left_trials_df = trials_df[(trials_df['trial_side'] == 'left') & (trials_df['rewarded'] == 1)]
    
    # --- Step 2: Align and Compute Averages for Left ---
    aligned_left = align_trials(left_trials_df, align_to_time_column='choice_start')
    common_time_left, neuron_averages_left = compute_neuron_averages(aligned_left)
    
    # --- Step 3: Cluster on Left Trials ---
    cluster_df_left = kmeans_cluster_neurons(neuron_averages_left, k=5)
    
    # --- Step 4: Save Cluster Memberships ---
    clusters_neuron_ids = extract_cluster_neuron_ids(cluster_df_left)
    
    # --- Step 5: Extract Neuron Traces for Left Clusters ---
    neuron_traces_left = {}
    
    for cluster_id, neuron_ids in clusters_neuron_ids.items():
        traces = extract_neuron_traces(neuron_averages_left, neuron_ids)
        neuron_traces_left[cluster_id] = traces
    
    # --- Step 6: Filter Right Rewarded Trials ---
    right_trials_df = trials_df[(trials_df['trial_side'] == 'right') & (trials_df['rewarded'] == 1)]
    
    # --- Step 7: Align and Compute Averages for Right ---
    # aligned_right = align_trials(right_trials_df, align_to_time_column='stim_start_bpod')
    aligned_right = align_trials(right_trials_df, align_to_time_column='choice_start')
    common_time_right, neuron_averages_right = compute_neuron_averages(aligned_right)
    
    # --- Step 8: Extract Neuron Traces for Same Neurons on Right ---
    neuron_traces_right = {}
    
    for cluster_id, neuron_ids in clusters_neuron_ids.items():
        traces = extract_neuron_traces(neuron_averages_right, neuron_ids)
        neuron_traces_right[cluster_id] = traces
    
    # --- Step 9: Overlay Traces Left vs Right ---
    overlay_traces_by_cluster(
        common_time_left,  # or common_time_right (should be close)
        neuron_traces_left,
        neuron_traces_right,
        title_prefix="Rewarded - Choice Start Aligned -"
    )


    # REW/PUN, LEFT/RIGHT, SAME ROIS CLUSTERED
    # --- First cluster left rewarded ---
    left_trials_df = trials_df[(trials_df['trial_side'] == 'left') & (trials_df['rewarded'] == 1)]
    aligned_left = align_trials(left_trials_df, align_to_time_column='stim_start_bpod')
    common_time_left, neuron_averages_left = compute_neuron_averages(aligned_left)
    cluster_df_left = kmeans_cluster_neurons(neuron_averages_left, k=5)
    clusters_neuron_ids = extract_cluster_neuron_ids(cluster_df_left)
    
    # --- Compare across rewarded/punished ---
    # extracted_traces = compare_reward_punish(trials_df, clusters_neuron_ids, align_to='stim_start_bpod')
    extracted_traces = compare_reward_punish(trials_df, clusters_neuron_ids, align_to='choice_start')
    
    # --- Plot the comparisons ---
    plot_comparison_overlays(extracted_traces)
    
    
    # queries
    # mouse licks left and is rewarded
    # "(mouse_choice == 0) and (rewarded == 1)"
    
    # trial type is left and is rewarded
    # "(trial_side == 'left') and (rewarded == 1)"
    
    # --- Define your filtering conditions ---
    conditions = {
        'all_trials': lambda df: pd.Series(True, index=df.index),
        'left_choices': lambda df: df['mouse_choice'] == 0,
        'right_choices': lambda df: df['mouse_choice'] == 1,
        'rewarded_trials': lambda df: df['rewarded'] == 1,
    }
    
    # --- Define your alignments ---
    alignments = {
        'first_flash_start': 'stim_start_bpod',
        'choice_window_start': 'choice_start', 
        'first_lick_start': 'lick_time',
        'reward_start': 'reward_start',
    }
    
    # --- Run full report generation ---
    generate_full_report(
        trials_df=trials_df,         # your segmented trials DataFrame
        conditions=conditions,
        alignments=alignments,
        k_clusters=5,                # number of clusters for k-means
        time_window=(-7, 8)          # plotting window around alignment
    )    
    
    
    
    # REWARDED LEFT TRIALS
    ############################################################
    
    # CHOICE WINDOW
    # 1. Align trials to choice window
    # trials_choice_align_df = align_trials(trials_df, align_to_time_column='stim_start_bpod')
    trials_reward_choice_align_df = align_trials(trials_df, align_to_time_column='choice_start')
    
    # 2. Filter the trials for left-choice and rewarded trials
    trials_reward_choice_align_df = trials_reward_choice_align_df.query("(trial_side == 'left') and (rewarded == 1)")
    
    num_neurons = dff.shape[0]
    
    time_window = [-4, 4]
    num_neurons = 10
    for neuron_idx in range(num_neurons):
        # plot_neural_raster(trials_choice_align_filtered, neuron_idx=neuron_idx, time_window=time_window, show_flash_bars=True)
        plot_neural_raster(trials_reward_choice_align_df, neuron_idx=neuron_idx, time_window=time_window, show_flash_bars=True, plot_title=f'Neuron Raster Choice Window Aligned - Rewarded Trials - Neuron {neuron_idx}', align_line_label='Choice Window Aligned')

    # 1. Align trials to stim start
    trials_reward_stim_start_align_df = align_trials(trials_df, align_to_time_column='stim_start_bpod')
    
    
    # 2. Filter the trialsfor left-choice and rewarded trials
    trials_reward_stim_start_align_df = trials_reward_stim_start_align_df.query("(trial_side == 'left') and (rewarded == 1)")
    
    num_neurons = dff.shape[0]
    
    time_window = [-2, 4]
    num_neurons = 10
    for neuron_idx in range(num_neurons):
        plot_neural_raster(trials_reward_stim_start_align_df, neuron_idx=neuron_idx, time_window=time_window, show_flash_bars=True, plot_title=f'Neuron Raster Stim Start Aligned - Rewarded Trials - Neuron {neuron_idx}', align_line_label='Stim Start Aligned')


    # 1. Align trials to reward start
    trials_reward_start_align_df = align_trials(trials_df, align_to_time_column='reward_start')
    
    
    # 2. Filter the trialsfor left-choice and rewarded trials
    trials_reward_start_align_df = trials_reward_start_align_df.query("(trial_side == 'left') and (rewarded == 1)")
    
    
    time_window = [-4, 4]
    num_neurons = 10
    # num_neurons = dff.shape[0]
    for neuron_idx in range(num_neurons):
        plot_neural_raster(trials_reward_start_align_df, neuron_idx=neuron_idx, time_window=time_window, show_flash_bars=True, plot_title=f'Neuron Raster Reward Start Aligned - Rewarded Trials - Neuron {neuron_idx}', align_line_label='Reward Start Aligned')

    
    # PUNISHED LEFT TRIALS
    ############################################################
    
    
    # CHOICE WINDOW
    # 1. Align trials to choice window
    # trials_choice_align_df = align_trials(trials_df, align_to_time_column='stim_start_bpod')
    trials_punish_choice_align_df = align_trials(trials_df, align_to_time_column='choice_start')
    
    # 2. Filter the trials for left-choice and rewarded trials
    trials_punish_choice_align_df = trials_punish_choice_align_df.query("(trial_side == 'left') and (rewarded == 0)")
    
    num_neurons = dff.shape[0]
    
    time_window = [-4, 4]
    num_neurons = 10
    for neuron_idx in range(num_neurons):
        # plot_neural_raster(trials_choice_align_filtered, neuron_idx=neuron_idx, time_window=time_window, show_flash_bars=True)
        plot_neural_raster(trials_punish_choice_align_df, neuron_idx=neuron_idx, time_window=time_window, show_flash_bars=True, plot_title=f'Neuron Raster Choice Window Aligned - Rewarded Trials - Neuron {neuron_idx}', align_line_label='Choice Window Aligned')

    # 1. Align trials to stim start
    trials_punish_stim_start_align_df = align_trials(trials_df, align_to_time_column='stim_start_bpod')
    
    
    # 2. Filter the trialsfor left-choice and rewarded trials
    trials_punish_stim_start_align_df = trials_punish_stim_start_align_df.query("(trial_side == 'left') and (rewarded == 0)")
    
    num_neurons = dff.shape[0]
    
    time_window = [-2, 4]
    num_neurons = 10
    for neuron_idx in range(num_neurons):
        plot_neural_raster(trials_punish_stim_start_align_df, neuron_idx=neuron_idx, time_window=time_window, show_flash_bars=True, plot_title=f'Neuron Raster Stim Start Aligned - Rewarded Trials - Neuron {neuron_idx}', align_line_label='Stim Start Aligned')


    # 1. Align trials to punish start
    trials_punish_start_align_df = align_trials(trials_df, align_to_time_column='punish_start')
    
    
    # 2. Filter the trials for left-choice and rewarded trials
    trials_punish_start_align_df = trials_punish_start_align_df.query("(trial_side == 'left') and (rewarded == 0)")
    
    
    time_window = [-4, 4]
    num_neurons = 10
    # num_neurons = dff.shape[0]
    for neuron_idx in range(num_neurons):
        plot_neural_raster(trials_punish_start_align_df, neuron_idx=neuron_idx, time_window=time_window, show_flash_bars=True, plot_title=f'Neuron Raster Punish Start Aligned - Rewarded Trials - Neuron {neuron_idx}', align_line_label='Punish Start Aligned')    
    
    
    
    # sort types
    # 
    # peak
    # mean
    # integral
    # max_amp
    # time_to_peak
    # custom_window
    

    sort_types = ['peak', 'mean', 'integral', 'max_amp', 'time_to_peak']    

    # HEATMAP RASTERS - AVERAGE NEURON TRACES
    
    for sort_type in sort_types:
        # STIM START ALIGNED - REWARD
        time_window = [-0.5, 6]
        common_time, neuron_averages = compute_neuron_averages(trials_reward_stim_start_align_df, time_window=time_window, num_points=500)
        plot_neuron_heatmap(common_time, neuron_averages, sort_by=sort_type, time_window=time_window, cmap='viridis', plot_title='Neuron Heatmap - Stim Start Aligned - Rewarded', align_line_label='Stim Start Aligned')
        
        # STIM START ALIGNED - PUNISH
        time_window = [-0.5, 6]
        common_time, neuron_averages = compute_neuron_averages(trials_punish_stim_start_align_df, time_window=time_window, num_points=500)
        plot_neuron_heatmap(common_time, neuron_averages, sort_by=sort_type, time_window=time_window, cmap='viridis', plot_title='Neuron Heatmap - Stim Start Aligned - Punished', align_line_label='Stim Start Aligned')    
        
        
        # CHOICE WINDOW START ALIGNED - REWARD
        time_window = [-4, 4]
        common_time, neuron_averages = compute_neuron_averages(trials_reward_choice_align_df, time_window=time_window, num_points=500)
        plot_neuron_heatmap(common_time, neuron_averages, sort_by=sort_type, time_window=time_window, cmap='viridis', plot_title='Neuron Heatmap - Choice Start Aligned - Rewarded', align_line_label='Choice Start Aligned')    
        
        # CHOICE WINDOW START ALIGNED - PUNISH
        time_window = [-4, 4]
        common_time, neuron_averages = compute_neuron_averages(trials_punish_choice_align_df, time_window=time_window, num_points=500)
        plot_neuron_heatmap(common_time, neuron_averages, sort_by=sort_type, time_window=time_window, cmap='viridis', plot_title='Neuron Heatmap - Choice Start Aligned - Punished', align_line_label='Choice Start Aligned')        
        
        # REWARD START ALIGNED
        time_window = [-4, 4]
        common_time, neuron_averages = compute_neuron_averages(trials_reward_start_align_df, time_window=time_window, num_points=500)
        plot_neuron_heatmap(common_time, neuron_averages, sort_by=sort_type, time_window=time_window, cmap='viridis', plot_title='Neuron Heatmap - Reward Start Aligned - Rewarded', align_line_label='Reward Start Aligned')
    
        # PUNISH START ALIGNED
        time_window = [-4, 4]
        common_time, neuron_averages = compute_neuron_averages(trials_punish_start_align_df, time_window=time_window, num_points=500)
        plot_neuron_heatmap(common_time, neuron_averages, sort_by=sort_type, time_window=time_window, cmap='viridis', plot_title='Neuron Heatmap - Punish Start Aligned - Punished', align_line_label='Punish Start Aligned')    
    

    # Get neuron averages
    time_window = [-0.5, 6]    
    common_time, neuron_averages = compute_neuron_averages(trials_reward_stim_start_align_df, time_window=time_window, num_points=500)

    # 2. Cluster neurons
    cluster_df = kmeans_cluster_neurons(neuron_averages, k=5, normalize=True)
        
    # 3. Plot heatmap sorted by cluster
    plot_cluster_heatmap(
        common_time,
        neuron_averages,
        cluster_df,
        time_window=time_window,
        cmap='viridis',
        plot_title='Clustered Neural Activity Aligned to Stim Start',
        align_line_label='Stim Start Aligned'
    )

  
    # 4. Compute cluster averages and SEMs
    common_time, cluster_avg_traces, cluster_sem_traces = compute_cluster_averages(common_time, neuron_averages, cluster_df)
    
    # 5. Plot cluster average traces with SEM
    plot_cluster_averages(
        common_time,
        cluster_avg_traces,
        cluster_sem_traces,
        time_window=time_window,
        plot_title='Average Neural Response per Cluster Aligned to Stim Start',
        align_line_label='Stim Start Aligned'
    )    


    # # --- Define your filtering conditions ---
    # conditions = {
    #     'all_trials': lambda df: pd.Series(True, index=df.index),
    #     'left_choices': lambda df: df['mouse_choice'] == 0,
    #     'right_choices': lambda df: df['mouse_choice'] == 1,
    #     'rewarded_trials': lambda df: df['rewarded'] == 1,
    # }
    
    # # --- Define your alignments ---
    # alignments = {
    #     'first_flash_start': 'stim_start_bpod',
    #     'choice_window_start': 'choice_start', 
    #     'first_lick_start': 'earliest_lick',
    #     'reward_start': 'reward_start',
    # }
    
    # # --- Run full report generation ---
    # generate_full_report(
    #     trials_df=trials_df,         # your segmented trials DataFrame
    #     conditions=conditions,
    #     alignments=alignments,
    #     k_clusters=5,                # number of clusters for k-means
    #     time_window=(-7, 8)          # plotting window around alignment
    # )



    # time_window = [-3, 5]
    # num_neurons = 10
    # for neuron_idx in range(num_neurons):
    #     plot_trial_average(trials_choice_align_filtered, neuron_idx=neuron_idx, time_window=time_window)
    # plot_neural_raster(trials_df, trial_filter=None, neuron_idx=0):
    # frame_times = np.arange(dff.shape[1]) / ops['fs']

    # Load the root
    # xml_path = xml_file
    # print(f"[LOADING] Parsing {xml_path}")
    
    # tree = ET.parse(xml_path)
    # root = tree.getroot()
    
    # # Print the root tag and immediate children
    # print(f"[INFO] Root tag: {root.tag}")
    # print(f"[INFO] Number of immediate children: {len(root)}")
    
    # # Print first few elements to inspect structure
    # for i, child in enumerate(root):
    #     print(f"[CHILD {i}] tag: {child.tag}, attributes: {child.attrib}")
    #     if i > 10:  # only print first 10 children
    #         break



    # segment_trials_with_voltage_sync(all_sessions_df, frame_times, trials_df, voltage_time, voltage_sync_signal, threshold=0.5)
    
    print('Checcka Flag')
    
    
    
    
    
    
    
    
    
    return