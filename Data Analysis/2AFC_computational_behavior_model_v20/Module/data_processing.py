"""
Module/data_processing.py

Handles the extraction, parsing, and transformation of raw .mat behavior
files into structured Pandas DataFrames ready for PyTorch model ingestion.
"""

import numpy as np
import pandas as pd

# External reader modules (Assuming these already exist in your Module/ folder)
from Module.Reader import load_mat
from Module.session_name_parse import parse_behavior_file_path
from Module.Licking_properties import extract_lick_properties

def states_labeling(trial_states):
    """Parses raw trial states to determine the outcome and initiation status."""
    outcome_priority = [
        'ChangingMindReward', 'WrongInitiation', 'Punish', 'Reward',
        'PunishNaive', 'RewardNaive', 'EarlyChoice', 'DidNotChoose'
    ]
    for state in outcome_priority:
        if (state in trial_states and 
            len(trial_states[state]) > 0 and 
            not np.isnan(trial_states[state][0])):
            return state, trial_states[state][0]
    return 'Other', np.nan

def extract_session_properties(data_single_session, subject, version, session_date):
    """Extracts trial-by-trial properties and session metadata from raw dict."""
    try:
        settings = data_single_session['TrialSettings'][0]['GUI']
        gui_meta = data_single_session.get('TrialSettings', [{}])[0].get('GUIMeta', {})
        
        number_of_trials = data_single_session.get('nTrials', 0)
        warm_up_trials = settings.get('NumNaiveWarmup', 0)
        max_same_side = settings.get('MaxSameSide', 0)
        Num_block_warmup = settings.get('NumBlockWarmup', 0)
        block_length = settings.get('BlockLength', 0)
        
        training_level_idx = settings.get('TrainingLevel', 0)
        training_level = gui_meta.get('TrainingLevel', {}).get('String', [])[training_level_idx-1] if 'TrainingLevel' in gui_meta else 'Unknown'
        
        if 'Contingency' in settings:
            contingency_idx = settings.get('Contingency', 0)
            contingency_type = gui_meta.get('Contingency', {}).get('String', [])[contingency_idx - 1] if 'Contingency' in gui_meta else 'Unknown'
        else:
            contingency_type = 'Normal - Short/Left'
            contingency_idx = 1 

        isi_settings = {
            'short': {'mean': settings.get('ISIShortMean_s', 0), 'max': settings.get('ISIShortMax_s', 0), 'min': settings.get('ISIShortMin_s', 0)},
            'long': {'mean': settings.get('ISILongMean_s', 0), 'max': settings.get('ISILongMax_s', 0), 'min': settings.get('ISILongMin_s', 0)}
        }
        isi_divider = settings.get('ISIOrig_s', 0)
        
        trials_properties = {
            'subject': subject, 'version': version, 'session_date': session_date,
            'outcome': [], 'outcome_initiate': [], 'warm_up': [], 'trial_initiation_time': [],
            'jitter_flag': [], 'opto_tag': [], 'trial_type': [], 'trial_isi': [], 'block_type': [],  
            'n_warm_up_trials': warm_up_trials, 'n_max_same_side': max_same_side,
            'difficulty': training_level, 'experimentor': settings.get('ExperimenterInitials', 'Unknown'),
            'Anti_bias': settings.get('AntiBiasServoAdjustAct', 0), 'num_warmup_blocks': Num_block_warmup,
            'block_length': block_length, 'Contingency': contingency_type, 'Contingency_idx': contingency_idx,
            'isi_settings': isi_settings, 'isi_devider': isi_divider
        }

        trial_types = data_single_session.get('TrialTypes', [])
        block_types = data_single_session.get('BlockTypes', [])
        
        for i in range(number_of_trials):
            try:
                trial_states = data_single_session['RawEvents']['Trial'][i]['States']
                outcome, outcome_initiate = states_labeling(trial_states)
                trials_properties['outcome'].append(outcome)
                trials_properties['outcome_initiate'].append(outcome_initiate)
                trials_properties['warm_up'].append(1 if i < warm_up_trials else 0)
                trials_properties['trial_initiation_time'].append(outcome_initiate)
                
                jitter_flag = data_single_session.get('JitterFlag', [])
                trials_properties['jitter_flag'].append(1 if i < len(jitter_flag) and jitter_flag[i] == 1 else 0)
                
                opto_type = data_single_session.get('OptoType', [])
                trials_properties['opto_tag'].append(opto_type[i] if i < len(opto_type) else None)
                
                trial_type = None
                if i < len(trial_types):
                    if contingency_idx == 1:
                        trial_type = trial_types[i]
                    elif contingency_idx == 2:
                        trial_type = 2 if trial_types[i] == 1 else (1 if trial_types[i] == 2 else trial_types[i])
                    else:
                        trial_type = trial_types[i]
                trials_properties['trial_type'].append(trial_type)

                block_type = None
                if i < len(block_types):
                    if contingency_idx == 1:
                        block_type = block_types[i]
                    elif contingency_idx == 2:
                        block_type = 2 if block_types[i] == 1 else (1 if block_types[i] == 2 else block_types[i])
                    else:
                        block_type = block_types[i]
                trials_properties['block_type'].append(block_type)
                
                processed_data = data_single_session.get('ProcessedSessionData', [])
                if i < len(processed_data) and isinstance(processed_data[i], dict) and 'trial_isi' in processed_data[i]:
                    trials_properties['trial_isi'].append(processed_data[i]['trial_isi'].get('PostMeanISI', None))
                else:
                    trials_properties['trial_isi'].append(None)
                    
            except (KeyError, IndexError):
                for key in ['outcome', 'outcome_initiate', 'warm_up', 'trial_initiation_time', 
                           'jitter_flag', 'opto_tag', 'trial_type', 'trial_isi', 'block_type']:
                    if len(trials_properties[key]) == i:
                        trials_properties[key].append(None)
        
        return trials_properties
    except Exception as e:
        print(f"Error extracting session properties: {e}")
        return {}

def load_session_data(path):
    """Loads a single session using standard parsing tools."""
    session = load_mat(path)
    subject, version, date = parse_behavior_file_path(path)
    return {
        'session': session, 'subject': subject, 'version': version, 'date': date,
        'trial_properties': extract_session_properties(session, subject, version, date),
        'lick_properties': extract_lick_properties(session, subject, version, date)
    }

def prepare_session_data(data_paths):
    """Compiles multiple sessions into a unified dictionary of arrays."""
    sessions_data = {
        'dates': [], 'outcomes': [], 'opto_tags': [], 'trial_types': [], 'trial_isi': [],
        'lick_properties': [], 'block_type': []
    }
    for path in data_paths:
        try:
            data = load_session_data(path)
            sessions_data['dates'].append(data['date'])
            sessions_data['outcomes'].append(data['trial_properties']['outcome'])
            sessions_data['opto_tags'].append(data['trial_properties']['opto_tag'])
            sessions_data['trial_types'].append(data['trial_properties']['trial_type'])
            sessions_data['trial_isi'].append(data['trial_properties']['trial_isi'])
            sessions_data['lick_properties'].append(data['lick_properties'])
            sessions_data['block_type'].append(data['trial_properties']['block_type'])
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    return sessions_data

def transform_data_to_dataframe(sessions_data, session_index=0):
    """Transforms raw dictionary lists into a clean pandas DataFrame for the model."""
    if session_index >= len(sessions_data['outcomes']): return pd.DataFrame()
    outcomes = np.asarray(sessions_data['outcomes'][session_index])
    trial_types_int = np.asarray(sessions_data['trial_types'][session_index])
    trial_isi = np.asarray(sessions_data['trial_isi'][session_index])
    block_types = np.asarray(sessions_data['block_type'][session_index])

    if len(outcomes) == 0: return pd.DataFrame()

    mouse_choices = []
    for outcome, tt in zip(outcomes, trial_types_int):
        rewarded = outcome in ['Reward', 'RewardNaive']
        is_short = (tt == 1)
        if rewarded:
            mouse_choices.append('left' if is_short else 'right')
        else:
            mouse_choices.append('right' if is_short else 'left')

    df = pd.DataFrame({
        'isi': trial_isi,
        'trial_type': ['short' if tt == 1 else 'long' for tt in trial_types_int],
        'block_type': ['neutral' if b == 0 else 'short_block' if b == 1 else 'long_block' for b in block_types],
        'mouse_choice': mouse_choices,
        'rewarded': [1 if o in ['Reward', 'RewardNaive'] else 0 for o in outcomes]
    })
    return df.dropna().reset_index(drop=True)