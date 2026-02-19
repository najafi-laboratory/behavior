import logging
import numpy as np
import pandas as pd
from datetime import datetime

class SingleIntervalDiscriminationPreprocessor:
    """
    Experiment-specific preprocessor for single interval discrimination data.
    Handles preprocessing specific to this experiment type.
    """
    
    # Class constants - experiment-specific mappings
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
    
    OPTO_SIDE_TO_NUM = {
        'Center': 0,
        'Left': 1,
        'Right': 2
    }
    
    def __init__(self, config_manager, logger):
        """Initialize with ConfigManager for accessing experiment config."""
        self.config_manager = config_manager
        self.logger = logger
        self.logger.info("SP-SID: Initializing SingleIntervalDiscriminationPreprocessor...")
        
        # Get experiment-specific preprocessing config
        experiment_config = config_manager.config.get('experiment_configs', {}).get(config_manager.experiment_name, {})
        self.preprocessing_config = experiment_config.get('preprocessing', {})
        
        self.logger.info("SP-SID: SingleIntervalDiscriminationPreprocessor initialized successfully")
    
    def preprocess_session_data(self, session_data, session_id):
        """
        Apply single interval discrimination specific preprocessing to session data.
        
        Args:
            session_data: Raw session data from SessionExtractor
            
        Returns:
            Processed session data with experiment-specific transformations
        """
        self.logger.info("SP-SID: Applying single interval discrimination preprocessing...")
        
        # Start with copy of original data
        processed_data = session_data.copy()
        
        # Add session ID
        processed_data['session_id'] = session_id
          
        # Add experiment-specific preprocessing steps here
        processed_data['experiment_type'] = 'single_interval_discrimination'
        processed_data['sid_preprocessing_applied'] = True
        
        # Extract session-level information
        session_info = self._extract_session_info(processed_data)
        
        # Validate trial data
        self._validate_trial_data(processed_data)
        
        # Build trial dataframe
        df = self.build_trial_dataframe(processed_data)
        
        # Add behavioral analysis columns
        df = self._add_behavioral_analysis(df, session_info)
        
        # Add optogenetics analysis
        df = self._add_optogenetics_analysis(df, session_info)
        
        # Add timing analysis
        df = self._add_timing_analysis(df)
        
        # Add additional features
        df = self._add_additional_features(df, processed_data)
        
        
        # Add trial timestamps relative to Bpod session times
        # Shift timestamps to align with start of first trial since all trial data starts at first trialss
        TrialStartTimestamp = session_data['TrialStartTimestamp']
        StartOfTrials = TrialStartTimestamp[0]
        TrialStartTimestamp = TrialStartTimestamp - StartOfTrials
        df['trial_start_timestamp'] = pd.Series(TrialStartTimestamp)
        
        TrialStopTimestamp = session_data['TrialEndTimestamp']
        TrialStopTimestamp = TrialStopTimestamp - StartOfTrials
        df['trial_stop_timestamp'] = pd.Series(TrialStopTimestamp)        
        
        df['trial_start'] = df['Start'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan)
                
        df['choice_stop'] = [trial['States']['WindowChoice'][-1] for trial in session_data['RawEvents']['Trial']]
        
        df['did_not_choose'] = (df['outcome'] == 'DidNotChoose').astype(int)

        df['time_did_not_choose'] = np.where(
            df['did_not_choose'] == 1, 
            df['choice_stop'], 
            np.nan
        )
        
        df['servo_in'] = [trial['Events']['SoftCode2'] for trial in session_data['RawEvents']['Trial']]            
        df['servo_out'] = [
            float(ev) if isinstance(ev, float) else float(ev[0])
            for ev in (trial['Events']['SoftCode1'] for trial in session_data['RawEvents']['Trial'])
        ]
            
        session_info['unique_isis'] = np.unique(df['isi'])
        
        # get mean isi, use as category boundary
        session_info['mean_isi'] = np.mean(session_info['unique_isis'])
        
        # get arrays of short and long isis
        u = np.asarray(session_info['unique_isis'])
        m = session_info['mean_isi']  # scalar
        session_info['long_isis'] = u[u > m]
        session_info['short_isis'] = u[u < m]


        df['is_short'] = (df['isi'] < session_info['mean_isi']).astype(int)

        # if df['lick'], then set df['lick_start'] to value of df['choice_stop']
        df['lick_start'] = np.where(df['lick'], df['choice_stop'], np.nan)

        # Store results
        processed_data['df_trials'] = df
        processed_data['session_info'] = session_info
        
        self.logger.info("SP-SID: Single interval discrimination preprocessing completed")
        return processed_data

    def _extract_session_info(self, processed_data):
        """Extract session-level information into structured format."""
        gui_settings = processed_data['TrialSettings'][0]['GUI']
        
        session_info = {
            'experiment_type': 'single_interval_discrimination',
            'sid_preprocessing_applied': True,
            'ComputerHostName': processed_data['ComputerHostName'],        
            'RigName': processed_data['RigName'],
            'nTrials': processed_data['nTrials'],        
            'SessionDate': processed_data['Info']['SessionDate'],
            'date': self.convert_date_to_yyyymmdd(processed_data['Info']['SessionDate']),
            'SessionStartTime_MATLAB': processed_data['Info']['SessionStartTime_MATLAB'],
            'SessionStartTime_UTC': processed_data['Info']['SessionStartTime_UTC'],  
            'TrialTypes': processed_data['TrialTypes'],
            'OptoType': processed_data.get('OptoType', 0),
            'GUIInitSettings': gui_settings,
            'subject_name': processed_data['subject_name']
        }
        
        return session_info

    def _validate_trial_data(self, processed_data):
        """Validate trial data consistency."""
        trial_data = processed_data['RawEvents']['Trial']
        expected_trials = processed_data['nTrials']
        
        if expected_trials != len(trial_data):
            raise ValueError(f"Trial count mismatch: expected {expected_trials}, got {len(trial_data)}")

    def _add_behavioral_analysis(self, df, session_info):
        """Add behavioral analysis columns to dataframe."""
        # Trial side information
        df['trial_side'] = pd.Series(session_info['TrialTypes']).map({1: 'left', 2: 'right'})
        df['is_right'] = (df['trial_side'] == 'right').astype(int)

        # Mouse choice information
        df['is_right_choice'] = df.apply(self.infer_choice, axis=1)
        df['mouse_choice'] = pd.Series(df['is_right_choice']).map({0: 'left', 1: 'right'})
        
        # Correctness
        df['mouse_correct'] = (df['is_right'] == df['is_right_choice']).astype(int)
        
        return df

    def _add_optogenetics_analysis(self, df, session_info):
        """Add optogenetics analysis columns and update session info."""
        opto_type_list = session_info.get('OptoType', [])    
        
        # Validate opto data length
        if len(opto_type_list) != len(df):
            raise ValueError(f"OptoType list length {len(opto_type_list)} doesn't match DataFrame length {len(df)}")
        
        # Add opto columns
        df['is_opto'] = pd.Series(session_info['OptoType']).astype(int)
        df['opto_encode'] = self.compute_opto_encode(df['is_opto'].values.astype(int))
        
        # Update session info
        session_info['OptoSession'] = int(df['is_opto'].any())
        session_info['OptoRegion'] = session_info.get('GUIInitSettings', {}).get('OptoRegion', 0) if session_info.get('OptoSession', 0) else 0
        
        # Add region decoding
        region_code = session_info.get('OptoRegion', 0)
        session_info.update(self.decode_opto_region_fields(region_code))
        
        return df

    def _add_timing_analysis(self, df):
        """Add timing-related analysis columns."""
        # Response times
        df['RT'] = df.apply(self.get_earliest_lick, axis=1, 
                           left_col='licks_left_start_choice', 
                           right_col='licks_right_start_choice')
        
        # Flash timings
        df = self.add_flash_timings(df)
        
        return df

    def _add_additional_features(self, df, processed_data):
        """Add additional feature columns."""
        # Move correct spout column
        if 'MoveCorrectSpout' in processed_data:
            df['MoveCorrectSpout'] = pd.Series(processed_data['MoveCorrectSpout']).astype(int)
        else:
            df['MoveCorrectSpout'] = 0
        
        return df

    # === Helper Methods for Data Processing ===
    
    def decode_opto_region_fields(self, region_id):
        """Decode opto region fields based on region ID."""
        region = self.REGION_MAP.get(region_id)
        
        if region is None:
            self.logger.warning(f"SP-SID: Unknown region ID {region_id}, using control settings")
            region = self.REGION_MAP[0]  # Default to control
        
        side = region['location']
        return {
            'OptoRegionCode': region_id,
            'OptoRegionShortText': region['abbrev'],
            'OptoRegionFullText': region['full_name'],
            'OptoTargetSideText': side,
            'OptoTargetSideNum': self.OPTO_SIDE_TO_NUM[side]
        }

    def states_labeling(self, trial_states):
        """Label trial states to determine outcome."""
        # Define outcome priority order (first match wins)
        # Use tuples in case we use different outcome labels
        outcome_checks = [
            ('ChangingMindReward', 'ChangingMindReward'),
            ('WrongInitiation', 'WrongInitiation'),
            ('Punish', 'Punish'),
            ('Reward', 'Reward'),
            ('PunishNaive', 'PunishNaive'),
            ('RewardNaive', 'RewardNaive'),
            ('EarlyChoice', 'EarlyChoice'),
            ('DidNotChoose', 'DidNotChoose')
        ]
        
        # Check each outcome in priority order
        for state_key, outcome_label in outcome_checks:
            if state_key in trial_states:
                state_value = trial_states[state_key]
                # Handle both single values and arrays
                if isinstance(state_value, (list, tuple)):
                    if len(state_value) > 0 and not np.isnan(state_value[0]):
                        return outcome_label
                else:
                    if not np.isnan(state_value):
                        return outcome_label
        
        # If no known state found, log warning and return 'Other'
        available_states = list(trial_states.keys())
        self.logger.warning(f"SP-SID: No recognized outcome state found. Available states: {available_states}")
        return 'Other'

    def compute_opto_encode(self, opto_col):
        """Compute opto encoding for trial sequence."""
        encode = []
        counter = np.nan
        for val in opto_col:
            if val == 1:
                counter = 0
            elif not np.isnan(counter):
                counter += 1
            encode.append(counter)
        return encode

    def get_flash_value(self, x, i):
        """Safely access the i-th element from a list/array/scalar."""
        if isinstance(x, (list, np.ndarray)):
            return x[i] if len(x) > i else np.nan
        elif isinstance(x, (float, int)) and i == 0:
            return x
        return np.nan

    def extract_flash_times(self, row):
        """Extract flash timing information from trial row."""
        start_flash_1 = self.get_flash_value(row['BNC1High'], 0)
        start_flash_2 = self.get_flash_value(row['BNC1High'], 1)
        end_flash_1   = self.get_flash_value(row['BNC1Low'], 0)
        end_flash_2   = self.get_flash_value(row['BNC1Low'], 1)

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

    def add_flash_timings(self, df):
        """Add safe flash timing columns and all_flashes indicator to df."""
        flash_df = df.apply(self.extract_flash_times, axis=1)
        return pd.concat([df, flash_df], axis=1)

    def get_earliest_lick(self, row, left_col, right_col):
        """Get earliest lick time from left and right columns."""
        left = row[left_col]
        right = row[right_col]

        all_licks = []

        if isinstance(left, list):
            all_licks.extend([v for v in left if not np.isnan(v)])
        if isinstance(right, list):
            all_licks.extend([v for v in right if not np.isnan(v)])

        return min(all_licks) if all_licks else np.nan   

    def sanitize_lick_times(self, start_times, stop_times, trial_start_time=None, trial_end_time=None, fudge_ms=1.0):
        """Ensure start_times and stop_times are well-formed."""

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

        # Final truncate to match lengths
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

    def filter_early_licks(self, start_times, stop_times, min_time_ms):
        """Remove licks that start before min_time_ms."""
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

    def build_trial_dataframe(self, data):
        """Build trial dataframe from session data."""
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
                    self.logger.warning(f"SP-SID: Failed to extract '{state}' in trial {idx}: {e}")
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
        
            # Convert lick times to ms (no initial alignment)
            initial_alignment = 0        
            row['licks_left_start'] = [(x - initial_alignment)*1000 for x in licks['licks_left_start']]
            row['licks_left_stop'] = [(x - initial_alignment)*1000 for x in licks['licks_left_stop']]
            row['licks_right_start'] = [(x - initial_alignment)*1000 for x in licks['licks_right_start']]
            row['licks_right_stop'] = [(x - initial_alignment)*1000 for x in licks['licks_right_stop']]        
            
            # check start/stop times to match
            row['licks_left_start'], row['licks_left_stop'] = self.sanitize_lick_times(row['licks_left_start'], row['licks_left_stop'], trial_start_time=None, trial_end_time=None, fudge_ms=1.0)
            row['licks_right_start'], row['licks_right_stop'] = self.sanitize_lick_times(row['licks_right_start'], row['licks_right_stop'], trial_start_time=None, trial_end_time=None, fudge_ms=1.0)
            
            # Filter early licks based on choice window start
            check_early_lick = states_events_data[idx]['States']['WindowChoice'][0] * 1000        
            row['licks_left_start'], row['licks_left_stop'] = self.filter_early_licks(row['licks_left_start'], row['licks_left_stop'], check_early_lick)
            row['licks_right_start'], row['licks_right_stop'] = self.filter_early_licks(row['licks_right_start'], row['licks_right_stop'], check_early_lick)        
            
            # Get window choice aligned lick times
            choice_window_alignment = states_events_data[idx]['States']['WindowChoice'][0] * 1000  
            row['licks_left_start_choice'] = [(x - choice_window_alignment) for x in row['licks_left_start']]
            row['licks_left_stop_choice'] = [(x - choice_window_alignment) for x in row['licks_left_stop']]
            row['licks_right_start_choice'] = [(x - choice_window_alignment) for x in row['licks_right_start']]
            row['licks_right_stop_choice'] = [(x - choice_window_alignment) for x in row['licks_right_stop']]          
        
            # TODO check for < 0 RTs
            # make sure to handle so that we don't lose trials, need all trials till we filter in data analysis
            
            # get outcome
            row['outcome'] = self.states_labeling(states)
            
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
                     
             # Track valve open/close times for the trial (start/stop), maybe we can decipher the world rig, eh? maybe...we'll see
            if row['rewarded']:
                if row['naive']:
                    valve_times.append(states['NaiveRewardDeliver'][0] - choice_window_alignment)
                    valve_times.append(states['NaiveRewardDeliver'][1] - choice_window_alignment)                                 
                else:
                    valve_times.append(states['Reward'][0] - choice_window_alignment)
                    valve_times.append(states['Reward'][1] - choice_window_alignment)
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
    def infer_choice(self, row):
        """Infer mouse choice based on reward and correct side."""
        if row['rewarded'] == 1:
            return row['is_right']  # Mouse must have chosen correctly
        else:
            return 1 - row['is_right']  # Mouse chose the wrong side


    def convert_date_to_yyyymmdd(self, date_str):
        """
        Convert date string like '17-Apr-2025' to '20250417'.
        """
        dt = datetime.strptime(date_str, "%d-%b-%Y")
        return dt.strftime("%Y%m%d")