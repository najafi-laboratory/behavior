import numpy as np

# Function to extract session and trial properties from experimental data
def extract_session_properties(data_single_session, subject, version, session_date):
    """
    Extract and organize session-level and trial-level properties from experimental data.
    
    Processes a single session to extract metadata, trial outcomes, contingency-adjusted
    trial/block types, and ISI settings. Handles reversed contingency logic to normalize
    trial classifications when the experimental setup uses reversed stimulus-outcome mapping.
    
    Args:
        data_single_session (dict): Raw session data containing trial information, settings, and events.
        subject (str): Subject/animal identifier.
        version (str): Experimental version or protocol identifier.
        session_date (str): Date of the experimental session.
    
    Returns:
        dict: Dictionary containing session metadata and lists of per-trial properties including:
            - outcome, outcome_initiate, trial_type, block_type, trial_isi, etc.
            - session constants (difficulty, contingency, ISI settings, etc.)
            Returns empty dict on critical errors.
    """
    try:
        # Extract session-level settings (constants across all trials)
        settings = data_single_session['TrialSettings'][0]['GUI']
        gui_meta = data_single_session.get('TrialSettings', [{}])[0].get('GUIMeta', {})
        
        # Get basic session parameters
        number_of_trials = data_single_session.get('nTrials', 0)
        warm_up_trials = settings.get('NumNaiveWarmup', 0)
        max_same_side = settings.get('MaxSameSide', 0)

        # Block information
        Num_block_warmup = settings.get('NumBlockWarmup', 0)
        block_length = settings.get('BlockLength', 0)
        
        # Get training level as a string
        training_level_idx = settings.get('TrainingLevel', 0)
        training_level = gui_meta.get('TrainingLevel', {}).get('String', [])[training_level_idx-1] if 'TrainingLevel' in gui_meta else 'Unknown'
        
        # Contingency (1 = normal, 2 = reversed) 
        if 'Contingency' in settings:
            contingency_idx = settings.get('Contingency', 0)
            contingency_type = gui_meta.get('Contingency', {}).get('String', [])[contingency_idx - 1] if 'Contingency' in gui_meta else 'Unknown'
        else:
            contingency_type = 'Normal - Short/Left'
            contingency_idx = 1 # Default to normal

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
            'block_type': [],  
            
            # Session constants
            'n_warm_up_trials': warm_up_trials,
            'n_max_same_side': max_same_side,
            'difficulty': training_level,
            'experimentor': settings.get('ExperimenterInitials', 'Unknown'),
            'Anti_bias': settings.get('AntiBiasServoAdjustAct', 0),
            'num_warmup_blocks': Num_block_warmup,
            'block_length': block_length,
            'Contingency': contingency_type,
            'Contingency_idx': contingency_idx,
            
            # ISI settings
            'isi_settings': isi_settings,
            'isi_devider': isi_divider
        }

        # Record trial type (2 = right, 1 = left)
        trial_types = data_single_session.get('TrialTypes', [])

        # Record block types (0 = warmup, 1 = short, 2 = long)
        block_types = data_single_session.get('BlockTypes', [])
        
        # Process each trial
        for i in range(number_of_trials):
            # Safely extract trial data with error handling
            try:
                trial_states = data_single_session['RawEvents']['Trial'][i]['States']
                # trial_events = data_single_session['RawEvents']['Trial'][i]['Events'] # Unused here
                
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
                
                # --- APPLY CONTINGENCY LOGIC FOR TRIAL TYPE ---
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

                # --- APPLY CONTINGENCY LOGIC FOR BLOCK TYPE ---
                block_type = None
                if i < len(block_types):
                    if contingency_idx == 1:
                        # Normal: use block_types as is
                        block_type = block_types[i]
                    elif contingency_idx == 2:
                        # Reversed: swap 1 <-> 2, keep 0 unchanged
                        if block_types[i] == 1:
                            block_type = 2
                        elif block_types[i] == 2:
                            block_type = 1
                        else:
                            block_type = block_types[i]  # Keep 0 or other values unchanged
                    else:
                        # Unknown contingency, use as is
                        block_type = block_types[i]
                trials_properties['block_type'].append(block_type)
                
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
                           'jitter_flag', 'opto_tag', 'trial_type', 'trial_isi', 'block_type']:
                    if len(trials_properties[key]) == i:  # Only append if this trial hasn't been processed yet
                        trials_properties[key].append(None)
        
        return trials_properties
        
    except Exception as e:
        print(f"Error extracting session properties: {e}")
        return {}


def states_labeling(trial_states):
    """
    Determine the outcome type and timestamp for a trial based on its state transitions.
    This function identifies the primary outcome of a trial by checking for the presence of 
    outcome states in a prioritized order. The priority reflects the behavioral hierarchy, 
    where more specific or exceptional outcomes (e.g., ChangingMindReward) take precedence 
    over general outcomes (e.g., Reward).
    Args:
        trial_states (dict): A dictionary mapping state names (str) to lists of timestamps (float).
                            Each state maps to a list containing one or more timestamps when 
                            that state was entered during the trial.
    Returns:
        tuple: A tuple containing:
            - outcome_type (str): The name of the detected outcome state. If no valid state 
                                 is found, returns 'Other'.
            - timestamp (float): The timestamp (in seconds) when the outcome state occurred. 
                               Returns np.nan if no valid outcome state is found.
    Notes:
        - The function returns the first valid outcome state encountered in priority order.
        - A state is considered valid if it exists in trial_states, has at least one entry,
          and the first timestamp is not NaN.
        - Priority order (highest to lowest): ChangingMindReward, WrongInitiation, Punish, 
          Reward, PunishNaive, RewardNaive, EarlyChoice, DidNotChoose.
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