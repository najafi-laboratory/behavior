import numpy as np
import pandas as pd

def states_labeling(trial_states):
    """
    Determine the outcome type and time for a trial based on its state transitions.
    
    This function evaluates trial states in a priority order to determine the final outcome
    of the trial and when that outcome was initiated.
    
    Args:
        trial_states (dict): Dictionary containing state transitions for a single trial
        
    Returns:
        tuple: (outcome_type, outcome_time) where:
            - outcome_type (str): The type of outcome for this trial
            - outcome_time (float): The timestamp when the outcome state was initiated
    """
    # Define priority order of outcome states to check
    outcome_priority = [
        'ChangingMindReward',
        'WrongInitiation',
        'Punish',
        'Reward',
        'PunishNaive',
        'RewardNaive',
        'EarlyChoice',
        'DidNotChoose'
    ]
    
    # Check each outcome state in priority order
    for state in outcome_priority:
        # Check if the state exists and has a valid timestamp (not NaN)
        if (state in trial_states and 
            len(trial_states[state]) > 0 and 
            not np.isnan(trial_states[state][0])):
            
            return state, trial_states[state][0]
    
    # If no valid state found, return default values
    return 'Other', np.nan


def extract_session_properties(data_single_session, subject, version, session_date):
    """
    Extract and organize properties from a single experimental session.
    
    This function extracts session-level settings and trial-by-trial properties
    from the raw session data and organizes them into a structured dictionary.
    
    Args:
        data_single_session (dict): Dictionary containing data from a single recording session
        
    Returns:
        dict: Organized dictionary of session and trial properties
    """
    try:
        # Extract session-level settings (constants across all trials)
        settings = data_single_session['TrialSettings'][0]['GUI']
        gui_meta = data_single_session.get('TrialSettings', [{}])[0].get('GUIMeta', {})
        
        # Get basic session parameters
        number_of_trials = data_single_session.get('nTrials', 0)
        warm_up_trials = settings.get('NumNaiveWarmup', 0)
        max_same_side = settings.get('MaxSameSide', 0)
        
        # Get training level as a string
        training_level_idx = settings.get('TrainingLevel', 0)
        training_level = gui_meta.get('TrainingLevel', {}).get('String', [])[training_level_idx - 1] if 'TrainingLevel' in gui_meta else 'Unknown'

        # Contingency (1 = normal, 2 = reversed) 
        # If normal, means that left choose for short ISI is rewareded, if reversed, then right choose for short ISI is rewarded
        if 'Contingency' in settings:
            contingency_idx = settings.get('Contingency', 0)
            contingency_type = gui_meta.get('Contingency', {}).get('String', [])[contingency_idx - 1] if 'Contingency' in gui_meta else 'Unknown'
        else:
            contingency_type = 'Normal - Short/Left'
 
        
        # Extract ISI settings
        isi_settings = {
            'short': {
                'mean': settings.get('ISIShortMean_s', 0),
                'max': settings.get('ISIShortMax_s', 0),
                'min': settings.get('ISIShortMin_s', 0)
            },
            'long': {
                'mean': settings.get('ISILongMean_s', 0),
                'max': settings.get('ISILongMax_s', 0),
                'min': settings.get('ISILongMin_s', 0)
            }
        }

        # Left to right ISI devider
        isi_divider = settings.get('ISIOrig_s', 0)
        
        # Initialize the session_properties dictionary
        trials_properties = {
            # Session metadata
            'subject': subject,
            'version': version,
            'session_date': session_date,
            
            # Lists to store per-trial data
            'outcome': [],
            'outcome_initiate': [],
            'warm_up': [],
            'trial_initiation_time': [],
            'jitter_flag': [],
            'opto_tag': [],
            'trial_type': [],
            'trial_isi': [],
            
            # Session constants
            'n_warm_up_trials': warm_up_trials,
            'n_max_same_side': max_same_side,
            'difficulty': training_level,
            'experimentor': settings.get('ExperimenterInitials', 'Unknown'),
            'Anti_bias': settings.get('AntiBiasServoAdjustAct', 0),
            'Contingency': contingency_type,
            'Contingency_idx': contingency_idx,
            
            # ISI settings (nested for better organization)
            'isi_settings': isi_settings,
            'isi_devider': isi_divider
        }
        
        # Record trial type (2 = right, 1 = left)
        trial_types = data_single_session.get('TrialTypes', [])

        # Process each trial
        for i in range(number_of_trials):
            # Safely extract trial data with error handling
            try:
                trial_states = data_single_session['RawEvents']['Trial'][i]['States']
                trial_events = data_single_session['RawEvents']['Trial'][i]['Events']
                
                # Get outcome type and time for each trial
                outcome, outcome_initiate = states_labeling(trial_states)
                trials_properties['outcome'].append(outcome)
                trials_properties['outcome_initiate'].append(outcome_initiate)
                
                # Record warm-up status (1 = warm-up trial, 0 = regular trial)
                trials_properties['warm_up'].append(1 if i < warm_up_trials else 0)
                
                # Record trial initiation time
                trials_properties['trial_initiation_time'].append(outcome_initiate)
                
                # Record jitter flag (1 = jittered, 0 = fixed)
                jitter_flag = data_single_session.get('JitterFlag', [])
                trials_properties['jitter_flag'].append(1 if i < len(jitter_flag) and jitter_flag[i] == 1 else 0)
                
                # Record opto tag for optogenetic experiments
                opto_type = data_single_session.get('OptoType', [])
                trials_properties['opto_tag'].append(opto_type[i] if i < len(opto_type) else None)
                
                # Determine trial type based on contingency
                trial_type = None
                if i < len(trial_types):
                    if contingency_idx == 1:
                        # Normal: use trial_types as is
                        trial_type = trial_types[i]
                    elif contingency_idx == 2:
                        # Reversed: swap 1 <-> 2
                        if trial_types[i] == 1:
                            trial_type = 2
                        elif trial_types[i] == 2:
                            trial_type = 1
                        else:
                            trial_type = trial_types[i]
                    else:
                        # Unknown contingency, use as is
                        trial_type = trial_types[i]
                trials_properties['trial_type'].append(trial_type)
                
                # Record trial ISI (inter-stimulus interval)
                processed_data = data_single_session.get('ProcessedSessionData', [])
                if i < len(processed_data) and isinstance(processed_data[i], dict) and 'trial_isi' in processed_data[i]:
                    trials_properties['trial_isi'].append(processed_data[i]['trial_isi'].get('PostMeanISI', None))
                else:
                    trials_properties['trial_isi'].append(None)
                    
            except (KeyError, IndexError) as e:
                print(f"Error processing trial {i}: {e}")
                # Add placeholder values on error
                for key in ['outcome', 'outcome_initiate', 'warm_up', 'trial_initiation_time', 
                           'jitter_flag', 'opto_tag', 'trial_type', 'trial_isi']:
                    if len(trials_properties[key]) == i:  # Only append if this trial hasn't been processed yet
                        trials_properties[key].append(None)
        
        return trials_properties
        
    except Exception as e:
        print(f"Error extracting session properties: {e}")
        return {}