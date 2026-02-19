# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 15:43:07 2025

@author: timst
"""
import os
import numpy as np
import scipy.io as sio
import pandas as pd
import pickle
from tqdm import tqdm
from datetime import datetime

from modules.utils import sanitize_path, sanitize_and_create_dir  # assume your utilities live here

REGION_MAP = {
    0: {'abbrev': 'Control', 'full_name': 'No Stimulation',              'location': 'Center'},
    1: {'abbrev': 'RLat',    'full_name': 'Right Lateral Cerebellar Nucleus', 'location': 'Right'},
    2: {'abbrev': 'LLat',    'full_name': 'Left Lateral Cerebellar Nucleus',  'location': 'Left'},
    3: {'abbrev': 'RIntA',   'full_name': 'Right Interposed Nucleus',   'location': 'Right'},
    4: {'abbrev': 'LIntA',   'full_name': 'Left Interposed Nucleus',    'location': 'Left'},
    5: {'abbrev': 'LPPC',    'full_name': 'Left Posterior Parietal Cortex',   'location': 'Left'},
    6: {'abbrev': 'RPPC',    'full_name': 'Right Posterior Parietal Cortex',  'location': 'Right'},
    7: {'abbrev': 'mPFC',    'full_name': 'Medial Prefrontal Cortex',   'location': 'Center'},
    8: {'abbrev': 'LPost',   'full_name': 'Left Posterior Cortex',      'location': 'Left'},
    9: {'abbrev': 'RPost',   'full_name': 'Right Posterior Cortex',     'location': 'Right'},
}

SIDE_TO_NUM = {"Center": 0, "Left": 1, "Right": 2}

def decode_opto_region_fields(region_id):
    region = REGION_MAP.get(region_id)
    
    side = region['location']
    return {
        'OptoRegionCode': region_id,
        'OptoRegionShortText': region['abbrev'],
        'OptoRegionFullText': region['full_name'],
        'OptoTargetSideText': side,
        'OptoTargetSideNum': SIDE_TO_NUM[side]
    }

# labeling every trials for a subject
def states_labeling(trial_states):
    if 'ChangingMindReward' in trial_states.keys() and not np.isnan(trial_states['ChangingMindReward'][0]):
        outcome = 'ChangingMindReward'
    elif 'WrongInitiation' in trial_states.keys() and not np.isnan(trial_states['WrongInitiation'][0]):
        outcome = 'WrongInitiation'
    elif 'Punish' in trial_states.keys() and not np.isnan(trial_states['Punish'][0]):
        outcome = 'Punish'
    elif 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
        outcome = 'Reward'
    elif 'PunishNaive' in trial_states.keys() and not np.isnan(trial_states['PunishNaive'][0]):
        outcome = 'PunishNaive'
    elif 'RewardNaive' in trial_states.keys() and not np.isnan(trial_states['RewardNaive'][0]):
        outcome = 'RewardNaive'
    elif 'EarlyChoice' in trial_states.keys() and not np.isnan(trial_states['EarlyChoice'][0]):
        outcome = 'EarlyChoice'
    elif 'DidNotChoose' in trial_states.keys() and not np.isnan(trial_states['DidNotChoose'][0]):
        outcome = 'DidNotChoose'
    else:
        outcome = 'Other'
    return outcome

def compute_opto_encode(opto_col):
    encode = []
    counter = np.nan
    for val in opto_col:
        if val == 1:
            counter = 0
        elif not np.isnan(counter):
            counter += 1
        encode.append(counter)
    return encode

def get_flash_value(x, i):
    """Safely access the i-th element from a list/array/scalar."""
    if isinstance(x, (list, np.ndarray)):
        return x[i] if len(x) > i else np.nan
    elif isinstance(x, (float, int)) and i == 0:
        return x
    return np.nan

def extract_flash_times(row):
    start_flash_1 = get_flash_value(row['BNC1High'], 0)
    start_flash_2 = get_flash_value(row['BNC1High'], 1)
    end_flash_1   = get_flash_value(row['BNC1Low'], 0)
    end_flash_2   = get_flash_value(row['BNC1Low'], 1)

    all_flashes = int(
        not np.isnan(start_flash_2) and not np.isnan(end_flash_2)
    )

    return pd.Series({
        'start_flash_1': start_flash_1,
        'start_flash_2': start_flash_2,
        'end_flash_1': end_flash_1,
        'end_flash_2': end_flash_2,
        'all_flashes': all_flashes
    })

def add_flash_timings(df):
    """Add safe flash timing columns and all_flashes indicator to df."""
    flash_df = df.apply(extract_flash_times, axis=1)
    return pd.concat([df, flash_df], axis=1)

def get_earliest_lick(row, left_col, right_col):
    left = row[left_col]
    right = row[right_col]

    all_licks = []

    if isinstance(left, list):
        all_licks.extend([v for v in left if not np.isnan(v)])
    if isinstance(right, list):
        all_licks.extend([v for v in right if not np.isnan(v)])

    return min(all_licks) if all_licks else np.nan   

def sanitize_lick_times(start_times, stop_times, trial_start_time=None, trial_end_time=None, fudge_ms=1.0):
    """
    Ensures start_times and stop_times are well-formed:
    - First stop must come after first start (else drop stop).
    - Last start must come before last stop (else drop or fudge stop).
    - Lengths must match.
    - Optionally enforce trial start/end limits.
    
    Parameters:
        start_times (list of float): Lick start times.
        stop_times (list of float): Lick stop times.
        trial_start_time (float or None): Optional trial start bound.
        trial_end_time (float or None): Optional trial end bound.
        fudge_ms (float): Small offset to apply if generating synthetic stop time.
    
    Returns:
        (list, list): Matched start/stop times or ([np.nan], [np.nan]) if none valid.
    """
    # start_times = np.array(start_times, dtype=np.float64)
    # stop_times = np.array(stop_times, dtype=np.float64)

    # # Drop NaNs
    # start_times = start_times[~np.isnan(start_times)]
    # stop_times = stop_times[~np.isnan(stop_times)]

    # # Handle too-short arrays
    # if len(start_times) == 0 and len(stop_times) == 0:
    #     return [np.nan], [np.nan]

    # If first stop precedes first start → drop the first stop
    if len(stop_times) > 0 and len(start_times) > 0 and stop_times[0] < start_times[0]:
        stop_times = stop_times[1:]

    # If last start is after last stop → drop the last start OR add fake stop
    if len(start_times) > len(stop_times):
        if trial_end_time is not None:
            stop_times = np.append(stop_times, min(trial_end_time, start_times[-1] + fudge_ms))
        else:
            stop_times = np.append(stop_times, start_times[-1] + fudge_ms)

    elif len(stop_times) > len(start_times):
        # If there's an extra stop time at the beginning
        stop_times = stop_times[:len(start_times)]

    # Final truncate to match
    min_len = min(len(start_times), len(stop_times))
    start_times = start_times[:min_len]
    stop_times = stop_times[:min_len]

    # Apply trial bounds if provided
    if trial_start_time is not None:
        valid = start_times >= trial_start_time
        start_times = start_times[valid]
        stop_times = stop_times[valid]

    if trial_end_time is not None:
        valid = stop_times <= trial_end_time
        start_times = start_times[valid]
        stop_times = stop_times[valid]

    if len(start_times) == 0 or len(stop_times) == 0:
        return [np.nan], [np.nan]

    return start_times, stop_times

def filter_early_licks(start_times, stop_times, min_time_ms):
    """
    Remove licks that start before min_time_ms.
    Keeps start/stop pairing by index.
    Returns [np.nan] if no valid licks remain.
    """
    start_times = np.array(start_times, dtype=np.float64)
    stop_times = np.array(stop_times, dtype=np.float64)

    # Keep indices where start >= min_time
    keep_mask = start_times >= min_time_ms

    if np.any(keep_mask):
        return (
            start_times[keep_mask].tolist(),
            stop_times[keep_mask].tolist()            
        )
    else:
        return [np.nan], [np.nan]

def build_trial_dataframe(data):
    rows = []
    
    events_to_get = [
                    'Port3In',
                    'Port3Out',
                    'Port1In',
                    'Port1Out',
                    'Port4In',
                    'Port4Out',
                    'BNC1High',
                    'BNC1Low',                    
                   ]

    # get states/events data separately, avoids having to name every state/event explicitly
    states_events_data = data['RawEvents']['Trial']
    for idx, trial in enumerate(states_events_data):
        row = {'trial_index': idx}  # Include index in the row
        
        # get state timings
        states = trial['States']
        for state in states:
            try:
                row[state] = states[state]
            except Exception as e:
                print(f"Warning: Failed to extract '{state}' in trial {idx}: {e}")
                row[state] = None
        
        # get timing of selected events
        events = trial['Events']
        for event in events_to_get:
            if event in events:
                row[event] = events[event]
            else:
                # row[event] = [np.nan]
                row[event] = np.nan
        
        licks = {}
        valve_times = []
    
        if not 'Port1In' in events.keys():
            licks['licks_left_start'] = [np.float64(np.nan)]
        elif isinstance(events['Port1In'], (float, int)):
            licks['licks_left_start'] = [events['Port1In']]
        else:
            licks['licks_left_start'] = events['Port1In']
            
        if not 'Port1Out' in events.keys():
            licks['licks_left_stop'] = [np.float64(np.nan)]
        elif isinstance(events['Port1Out'], (float, int)):
            licks['licks_left_stop'] = [events['Port1Out']]
        else:
            licks['licks_left_stop'] = events['Port1Out']    
    
        if not 'Port3In' in events.keys():
            licks['licks_right_start'] = [np.float64(np.nan)]
        elif isinstance(events['Port3In'], (float, int)):
            licks['licks_right_start'] = [events['Port3In']]
        else:
            licks['licks_right_start'] = events['Port3In']       
        
        if not 'Port3Out' in events.keys():
            licks['licks_right_stop'] = [np.float64(np.nan)]
        elif isinstance(events['Port3Out'], (float, int)):
            licks['licks_right_stop'] = [events['Port3Out']]
        else:
            licks['licks_right_stop'] = events['Port3Out']        
    
        # convert lick times to ms
        alignment = 0        
        row['licks_left_start'] = [(x - alignment)*1000 for x in licks['licks_left_start']]
        row['licks_left_stop'] = [(x - alignment)*1000 for x in licks['licks_left_stop']]
        row['licks_right_start'] = [(x - alignment)*1000 for x in licks['licks_right_start']]
        row['licks_right_stop'] = [(x - alignment)*1000 for x in licks['licks_right_stop']]        
        
        # check start/stop times to match
        row['licks_left_start'], row['licks_left_stop'] = sanitize_lick_times(row['licks_left_start'], row['licks_left_stop'], trial_start_time=None, trial_end_time=None, fudge_ms=1.0)
        row['licks_right_start'], row['licks_right_stop'] = sanitize_lick_times(row['licks_right_start'], row['licks_right_stop'], trial_start_time=None, trial_end_time=None, fudge_ms=1.0)
        
        # filter early licks
        check_early_lick = states_events_data[idx]['States']['WindowChoice'][0] * 1000        
        row['licks_left_start'], row['licks_left_stop'] = filter_early_licks(row['licks_left_start'], row['licks_left_stop'], check_early_lick)
        row['licks_right_start'], row['licks_right_stop'] = filter_early_licks(row['licks_right_start'], row['licks_right_stop'], check_early_lick)        
          
        # test filter
        # check_early_lick = 500  # in ms
        # row['licks_left_start'] = [480, 510, 495, 525]  # ms
        # row['licks_left_stop']  = [490, 520, 505, 540]  # paired stop times
        # row['licks_left_start'], row['licks_left_stop'] = filter_early_licks(
        #     row['licks_left_start'],
        #     row['licks_left_stop'],
        #     check_early_lick
        # )
        
        # get window choice aligned lick times
        alignment = states_events_data[idx]['States']['WindowChoice'][0] * 1000  
        row['licks_left_start_choice'] = [(x - alignment) for x in row['licks_left_start']]
        row['licks_left_stop_choice'] = [(x - alignment) for x in row['licks_left_stop']]
        row['licks_right_start_choice'] = [(x - alignment) for x in row['licks_right_start']]
        row['licks_right_stop_choice'] = [(x - alignment) for x in row['licks_right_stop']]          
        
        # check for < 0 RTs
        
        # get outcome
        row['outcome'] = states_labeling(states)
        
        # set naive flag
        if row['outcome'] in ['RewardNaive', 'PunishNaive']:
            row['naive'] = 1
        else:
            row['naive'] = 0
        
        # --- Add rewarded column ---
        if row['outcome'] in ['Reward', 'RewardNaive']:
            row['rewarded'] = 1
        else:
            row['rewarded'] = 0        
            
        # --- Add punished column ---
        if row['outcome'] in ['Punish', 'PunishNaive']:
            row['punished'] = 1
        else:
            row['punished'] = 0
            
        # add lick column               
        if not row['rewarded'] and not row['punished']:
            row['lick'] =  0
        else:
            row['lick'] =  1
        
        if row['rewarded']:
            row['punish_start'] = np.nan
            row['punish_stop'] = np.nan            
            if row['naive']:
                reward_times = trial['States'].get('RewardNaive', [])                
            else:
                reward_times = trial['States'].get('Reward', [])                
            row['reward_start'] = reward_times[0] if len(reward_times) > 0 else np.nan
            row['reward_stop'] = reward_times[1] if len(reward_times) > 0 else np.nan
        else:
            row['reward_start'] = np.nan
            row['reward_stop'] = np.nan
            if row['naive']:
                punish_times = trial['States'].get('PunishNaive', [])
            else:
                punish_times = trial['States'].get('Punish', [])
            row['punish_start'] = punish_times[0] if len(punish_times) > 0 else np.nan
            row['punish_stop'] = punish_times[1] if len(punish_times) > 0 else np.nan            
                 
         # Track valve open/close times for the trial (start/stop), maybe we can decipher the world rig, eh?
        if row['rewarded']:
            if row['naive']:
                valve_times.append(states['NaiveRewardDeliver'][0] - alignment)
                valve_times.append(states['NaiveRewardDeliver'][1] - alignment)                                 
            else:
                valve_times.append(states['Reward'][0] - alignment)
                valve_times.append(states['Reward'][1] - alignment)
        else:
            valve_times.append(np.float64(np.nan))
            valve_times.append(np.float64(np.nan))
        
        row['valve_start'] = valve_times[0]
        row['valve_stop'] = valve_times[1]
        
   
        
        row['trial_start'] = states['Start'][0]
        row['trial_stop'] = states['ITI'][1]        
        row['choice_start'] =  states['WindowChoice'][0]
        row['choice_stop'] =  states['WindowChoice'][1]
        row['stim_start'] = states['AudStimTrigger'][0]
        row['stim_stop'] = states['AudStimTrigger'][1]            
            
        row['isi'] = 1000*np.mean(data['ProcessedSessionData'][idx]['trial_isi']['PostISI'][0])         

        rows.append(row)

    return pd.DataFrame(rows)

# 1 = mouse chose right
# 0 = mouse chose left
# We'll infer based on whether rewarded and which side was correct
def infer_choice(row):
    if row['rewarded'] == 1:
        return row['is_right']  # Mouse must have chosen correctly
    else:
        return 1 - row['is_right']  # Mouse chose the wrong side


def convert_date_to_yyyymmdd(date_str):
    """
    Convert date string like '17-Apr-2025' to '20250417'.
    """
    dt = datetime.strptime(date_str, "%d-%b-%Y")
    return dt.strftime("%Y%m%d")

def preprocess_session_data(session_data):
    """
    Apply preprocessing to the extracted session data.

    Args:
        session_data (dict): Contains 'df' and 'session_info'.

    Returns:
        dict: Preprocessed session data.
    """
    data = session_data
    # session_info = session_data['session_info']

    # --- Session Data ---
    # get gui settings from first trial
    GUISettings = data['TrialSettings'][0]['GUI']
    
    # get session data
    session_info = {
        'ComputerHostName' : data['ComputerHostName'],        
        'RigName' : data['RigName'],
        'nTrials' : data['nTrials'],        
        'SessionDate' : data['Info']['SessionDate'],
        'date' : convert_date_to_yyyymmdd(data['Info']['SessionDate']),
        'SessionStartTime_MATLAB' : data['Info']['SessionStartTime_MATLAB'],
        'SessionStartTime_UTC' : data['Info']['SessionStartTime_UTC'],  
        'TrialTypes' : data['TrialTypes'],
        'OptoType' : data.get('OptoType', 0),
        'GUISettings': GUISettings,
        'subject_name': data['subject_name']
    }
    session_info['TrialTypes']
    

    # --- Trial Data ---
    # get trial data variable
    trial_data = data['RawEvents']['Trial']

    # check we have the right number of trials    
    assert data['nTrials'] == len(trial_data), "Mismatch: expected ntrials does not match length of trial_data"

    # get df for trial
    df = build_trial_dataframe(data)

    # # --- Add 'naive' column ---
    # df['naive'] = np.where(
    #     df[['RewardNaive', 'PunishNaive']].notna().any(axis=1),
    #     1,
    #     0
    # )
    
    # add left/right side type
    df['trial_side'] = pd.Series(session_info['TrialTypes']).map({1: 'left', 2: 'right'})
    
    # side type encoding
    df['is_right'] = (df['trial_side'] == 'right').astype(int)

    # add mouse choice side
    df['is_right_choice'] = df.apply(infer_choice, axis=1)
    df['mouse_choice'] = pd.Series(df['is_right_choice']).map({0: 'left', 1: 'right'})
    
    # add mouse choice correct, could check rewarded
    df['mouse_correct'] = (df['is_right'] == df['is_right_choice']).astype(int)
    
    # ---- Response Times ----
    df['RT'] = df.apply(get_earliest_lick, axis=1, left_col='licks_left_start_choice', right_col='licks_right_start_choice')   
    
    # --- Add 'opto' column and region info ---
    # get opto trial list
    opto_type_list = session_info.get('OptoType', [])    
    # Make sure it's the right length
    assert len(opto_type_list) == len(df), f"OptoType list length {len(opto_type_list)} doesn't match trial dataframe length {len(df)}"    
    # Add directly to DataFrame of trials
    df['is_opto'] = pd.Series(session_info['OptoType']).astype(int)
    # encode opto residual trials
    df['opto_encode'] = compute_opto_encode(df['is_opto'].values.astype(int))
    # opto session flag
    session_info['OptoSession'] = int(df['is_opto'].any())
    # if opto session, get opto region, otherwise set to control
    session_info['OptoRegion'] = session_info.get('GUISettings', {}).get('OptoRegion', 0) if session_info.get('OptoSession', 0) else 0
    # --- Region decoding ---
    region_code = session_info.get('OptoRegion', 0)
    session_info.update(decode_opto_region_fields(region_code))

    # --- move correct spout column ---
    if 'MoveCorrectSpout' in data:
        df['MoveCorrectSpout'] = pd.Series(data['MoveCorrectSpout']).astype(int)
    else:
        df['MoveCorrectSpout'] = 0  # fills entire column with 0s
        

    # get flash timings
    df = add_flash_timings(df)
    
    # print(session_info['SessionDate'])
    
    
    # if session_info['SessionDate'] == '04-Mar-2025':
    #     print('')    
    
    # df['start_flash_1'] = df['BNC1High'].apply(lambda x: x[0] if len(x) > 0 else np.nan)
    # df['end_flash_1']   = df['BNC1Low'].apply(lambda x: x[0] if len(x) > 0 else np.nan)
    # df['start_flash_2'] = df['BNC1High'].apply(lambda x: x[1] if len(x) > 1 else np.nan)    
    # df['end_flash_2']   = df['BNC1Low'].apply(lambda x: x[1] if len(x) > 1 else np.nan)

    # df['all_flashes'] 
    
    

    # Add metadata
    # change session_data var name later after we know we have the neede sessions data to reduce file size
    session_data['df'] = df
    session_data['session_info'] = session_info
    session_data['preprocessing_version'] = 1.0

    return session_data

def load_or_preprocess_session(extracted_path, preprocessed_path, force=False):
    """
    Load preprocessed session if available, otherwise load extracted and preprocess it.

    Args:
        extracted_path (str): Path to extracted session file.
        preprocessed_path (str): Path to preprocessed session file.
        force (bool): If True, redo preprocessing even if preprocessed file exists.

    Returns:
        dict: Preprocessed session data.
    """
    # subjects_root_dir = sanitize_path(config.paths['session_data'])
    # extracted_path = sanitize_and_create_dir(config.paths['extracted_data'])
    
    if not force and os.path.exists(preprocessed_path):
        with open(preprocessed_path, 'rb') as f:
            return pickle.load(f)

    # Load raw extracted
    with open(extracted_path, 'rb') as f:
        session_data = pickle.load(f)

    # Apply preprocessing
    session_data = preprocess_session_data(session_data)

    # Save preprocessed
    os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
    with open(preprocessed_path, 'wb') as f:
        pickle.dump(session_data, f)

    return session_data

def batch_preprocess_sessions(subject_list, config, force=False):
    """
    Load and preprocess all sessions listed in the config bundle.

    Args:
        extracted_root_dir (str): Base path to extracted sessions.
        preprocessed_root_dir (str): Base path to store preprocessed sessions.
        config_bundle (dict): Contains 'list_config' (dict of subject configs).
        force (bool): Force re-preprocessing even if preprocessed file exists.

    Returns:
        dict: Preprocessed data for all subjects and sessions.
    """    
    extracted_root_dir = sanitize_and_create_dir(config.paths['extracted_data'])
    preprocessed_root_dir = sanitize_path(config.paths['preprocessed_data'])    

    all_data = {}

    # for subject_id, subject_config in config.session_config_list_2AFC["list_config"].items():
    for subject_id in subject_list:        
        subject_config = config.session_config_list_2AFC["list_config"].get(subject_id, None)
        session_list = subject_config["list_session_name"]
        extracted_subject_dir = os.path.join(extracted_root_dir, subject_id)
        preprocessed_subject_dir = os.path.join(preprocessed_root_dir, subject_id)

        subject_sessions = []
        
        # for mat_file in tqdm(mat_files, desc=f"Extracting {os.path.basename(subject_dir)}"):
        
        for session_id in tqdm(session_list, desc=f"Preprocessing {subject_id}") :
            extracted_path = os.path.join(extracted_subject_dir, f"{session_id}_extracted.pkl")
            preprocessed_path = os.path.join(preprocessed_subject_dir, f"{session_id}_preprocessed.pkl")

            if not os.path.exists(extracted_path):
                print(f"[WARNING] Extracted file missing: {extracted_path}")
                continue

            session_data = load_or_preprocess_session(extracted_path, preprocessed_path, force=force)
            subject_sessions.append(session_data)

        all_data[subject_id] = subject_sessions

    return all_data
