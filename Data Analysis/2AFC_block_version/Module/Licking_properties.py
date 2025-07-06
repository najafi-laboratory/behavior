from Module.Trial_properties import extract_session_properties

def extract_lick_properties(data_single_session, subject, version, session_date):
    """
    Extract lick properties from a single session of a 2-alternative forced choice task.
    
    Parameters:
    -----------
    data_single_session : dict
        Single session data dictionary
    subject : str
        Subject ID
    version : str
        Version of the task
    session_date : str
        Date of the session
    
    Returns:
    --------
    lick_properties : dict
        Dictionary containing lick properties for different conditions
    """
    # Initialize the properties dictionaries for different trial types and outcomes
    short_ISI_reward_left_correct_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': []
    }
    
    short_ISI_reward_right_incorrect_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': []
    }
    
    short_ISI_punish_right_incorrect_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': []
    }
    
    short_ISI_punish_left_correct_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': []
    }
    
    long_ISI_reward_right_correct_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': []
    }
    
    long_ISI_reward_left_incorrect_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': []
    }
    
    long_ISI_punish_left_incorrect_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': []
    }
    
    long_ISI_punish_right_correct_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': []
    }
    
    # Extract session properties
    trials_properties = extract_session_properties(data_single_session, subject, version, session_date)
    
    # Extract trial properties
    outcome = trials_properties['outcome']
    trial_type = trials_properties['trial_type']
    trial_ISI = trials_properties['trial_isi']
    opto_tag = trials_properties['opto_tag']
    block_type = trials_properties['block_type']
    ISI_divider = trials_properties['isi_devider']
    ISI_setting = trials_properties['isi_settings']
    contingency_idx = 1
    
    # Get number of trials
    number_of_trials = data_single_session.get('nTrials', 0)
    
    for i in range(number_of_trials):
        trial_states = data_single_session['RawEvents']['Trial'][i]['States']
        trial_events = data_single_session['RawEvents']['Trial'][i]['Events']
        
        # Skip if trial data is incomplete
        if 'WindowChoice' not in trial_states:
            continue
        
        choice_window_start = trial_states.get('WindowChoice', [0])[0]
        # Normal contingiency 
        if contingency_idx == 1:
            # Short ISI trials
            if trial_type[i] == 1:
                # Correct left lick (rewarded)
                if outcome[i] == 'Reward':
                    # Check if Port1In (left port) exists
                    if 'Port1In' in trial_events:
                        # Process correct left licks in rewarded short ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            short_ISI_reward_left_correct_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
                    
                    # Check if Port3In (right port) exists - the mouse licked the correct port first (rewarded) 
                    # but also licked the incorrect port later
                    if 'Port3In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            short_ISI_reward_right_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
                
                # Incorrect right lick (punished)
                elif outcome[i] == 'Punish':
                    # Check if Port3In (right port) exists
                    if 'Port3In' in trial_events:
                        # Process incorrect right licks in punished short ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            short_ISI_punish_right_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
                    
                    # Check if Port1In (left port) exists - the mouse licked the incorrect port first (punished)
                    # but also licked the correct port later
                    if 'Port1In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            short_ISI_punish_left_correct_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
            
            # Long ISI trials
            elif trial_type[i] == 2:
                # Correct right lick (rewarded)
                if outcome[i] == 'Reward':
                    # Check if Port3In (right port) exists
                    if 'Port3In' in trial_events:
                        # Process correct right licks in rewarded long ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            long_ISI_reward_right_correct_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
                    
                    # Check if Port1In (left port) exists - the mouse licked correct port first (rewarded)
                    # but also licked incorrect port later
                    if 'Port1In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            long_ISI_reward_left_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
                
                # Incorrect left lick (punished)
                elif outcome[i] == 'Punish':
                    # Check if Port1In (left port) exists
                    if 'Port1In' in trial_events:
                        # Process incorrect left licks in punished long ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            long_ISI_punish_left_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
                    
                    # Check if Port3In (right port) exists - the mouse licked incorrect port first (punished)
                    # but also licked correct port later
                    if 'Port3In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            long_ISI_punish_right_correct_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
        # reverse ###########################################################################################
        elif contingency_idx == 2:
            # Short ISI trials
            if trial_type[i] == 1:
                # Correct left lick (rewarded)
                if outcome[i] == 'Reward':
                    # Check if Port3In (Right port) exists
                    if 'Port3In' in trial_events:
                        # Process correct left licks in rewarded short ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            long_ISI_reward_right_correct_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
                    
                    # Check if Port3In (right port) exists - the mouse licked the correct port first (rewarded) 
                    # but also licked the incorrect port later
                    if 'Port1In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            short_ISI_reward_right_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
                
                # Incorrect right lick (punished)
                elif outcome[i] == 'Punish':
                    # Check if Port3In (right port) exists
                    if 'Port1In' in trial_events:
                        # Process incorrect right licks in punished short ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            long_ISI_punish_left_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
                    
                    # Check if Port1In (left port) exists - the mouse licked the incorrect port first (punished)
                    # but also licked the correct port later
                    if 'Port3In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            short_ISI_punish_left_correct_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
            
            # Long ISI trials
            elif trial_type[i] == 2:
                # Correct right lick (rewarded)
                if outcome[i] == 'Reward':
                    # Check if Port3In (right port) exists
                    if 'Port1In' in trial_events:
                        # Process correct right licks in rewarded long ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            short_ISI_reward_left_correct_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
                    
                    # Check if Port1In (left port) exists - the mouse licked correct port first (rewarded)
                    # but also licked incorrect port later
                    if 'Port3In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            long_ISI_reward_left_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
                
                # Incorrect left lick (punished)
                elif outcome[i] == 'Punish':
                    # Check if Port1In (left port) exists
                    if 'Port3In' in trial_events:
                        # Process incorrect left licks in punished long ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            short_ISI_punish_right_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )
                    
                    # Check if Port3In (right port) exists - the mouse licked incorrect port first (punished)
                    # but also licked correct port later
                    if 'Port1In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            long_ISI_punish_right_correct_lick, trial_ISI[i], opto_tag[i], block_type[i]
                        )

        
            
    
    # Combine all lick data into a single dictionary for return
    lick_properties = {
        'short_ISI_reward_left_correct_lick': short_ISI_reward_left_correct_lick,
        'short_ISI_reward_right_incorrect_lick': short_ISI_reward_right_incorrect_lick,
        'short_ISI_punish_right_incorrect_lick': short_ISI_punish_right_incorrect_lick,
        'short_ISI_punish_left_correct_lick': short_ISI_punish_left_correct_lick,
        'long_ISI_reward_right_correct_lick': long_ISI_reward_right_correct_lick,
        'long_ISI_reward_left_incorrect_lick': long_ISI_reward_left_incorrect_lick,
        'long_ISI_punish_left_incorrect_lick': long_ISI_punish_left_incorrect_lick,
        'long_ISI_punish_right_correct_lick': long_ISI_punish_right_correct_lick,
        'trial_properties': trials_properties,
        'subject': subject,
        'session_date': session_date,
        'ISI_setting': ISI_setting,
        'ISI_devider': ISI_divider
    }
    
    return lick_properties


def process_lick_data(trial_events, trial_states, choice_window_start, port_in, port_out, data_dict, trial_isi, opto_tag, block_type):
    """
    Helper function to process lick data for a specific port and trial - modified to only record the first lick per trial
    """
    # Handle scalar or array inputs for port in/out
    port_in_times = trial_events.get(port_in, [])
    port_out_times = trial_events.get(port_out, [])
    
    # Convert to list if it's a scalar
    if not isinstance(port_in_times, list):
        port_in_times = [port_in_times]
    if not isinstance(port_out_times, list):
        port_out_times = [port_out_times]
    
    # Only process the first lick for each trial
    if port_in_times and port_out_times:
        start_time = port_in_times[0]
        end_time = port_out_times[0]
        
        # Calculate reaction time
        reaction_time = start_time - choice_window_start
        
        # Calculate relative times and duration
        rel_start_time = start_time - choice_window_start
        rel_end_time = end_time - choice_window_start
        duration = rel_end_time - rel_start_time
        
        # Store data
        data_dict['Lick_reaction_time'].append(reaction_time)
        data_dict['Lick_start_time'].append(rel_start_time)
        data_dict['Lick_end_time'].append(rel_end_time)
        data_dict['Lick_duration'].append(duration)
        data_dict['Trial_ISI'].append(trial_isi)
        data_dict['opto_tag'].append(opto_tag)
        data_dict['block_type'].append(block_type)