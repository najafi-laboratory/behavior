import numpy as np
from Module.Trial_properties import extract_session_properties

def extract_lick_properties(data_single_session, subject, version, session_date):
    """
    Extract lick properties from a single session of a 2-alternative forced choice task, including trial number.
    
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
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': [],
        'epoch': [], 'is_rare': [], 'is_post_opto': [], 'is_pre_opto': [], 'trial_number': []
    }
    
    short_ISI_reward_right_incorrect_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': [],
        'epoch': [], 'is_rare': [], 'is_post_opto': [], 'is_pre_opto': [], 'trial_number': []
    }
    
    short_ISI_punish_right_incorrect_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': [],
        'epoch': [], 'is_rare': [], 'is_post_opto': [], 'is_pre_opto': [], 'trial_number': []
    }
    
    short_ISI_punish_left_correct_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': [],
        'epoch': [], 'is_rare': [], 'is_post_opto': [], 'is_pre_opto': [], 'trial_number': []
    }
    
    long_ISI_reward_right_correct_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': [],
        'epoch': [], 'is_rare': [], 'is_post_opto': [], 'is_pre_opto': [], 'trial_number': []
    }
    
    long_ISI_reward_left_incorrect_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': [],
        'epoch': [], 'is_rare': [], 'is_post_opto': [], 'is_pre_opto': [], 'trial_number': []
    }
    
    long_ISI_punish_left_incorrect_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': [],
        'epoch': [], 'is_rare': [], 'is_post_opto': [], 'is_pre_opto': [], 'trial_number': []
    }
    
    long_ISI_punish_right_correct_lick = {
        'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
        'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': [],
        'epoch': [], 'is_rare': [], 'is_post_opto': [], 'is_pre_opto': [], 'trial_number': []
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
    contingency_idx = trials_properties['Contingency_idx']
    
    # Compute is_rare
    trial_types = np.array(trial_type)
    block_types = np.array(block_type)
    is_rare = np.zeros(len(trial_types), dtype=int)
    short_block_mask = block_types == 1
    long_block_mask = block_types == 2
    is_rare[short_block_mask & (trial_types == 2)] = 1  # Long trials in short blocks
    is_rare[long_block_mask & (trial_types == 1)] = 1  # Short trials in long blocks
    
    # Compute epoch (early=1, late=2)
    epoch = np.zeros(len(trial_types), dtype=int)
    for b_type in [0, 1, 2]:  # Iterate over block types
        block_mask = block_types == b_type
        block_indices = np.where(block_mask)[0]
        if len(block_indices) == 0:
            continue
        # Identify consecutive segments
        block_segments = []
        current_segment = [block_indices[0]]
        for i in range(1, len(block_indices)):
            if block_indices[i] == block_indices[i-1] + 1:
                current_segment.append(block_indices[i])
            else:
                block_segments.append(current_segment)
                current_segment = [block_indices[i]]
        block_segments.append(current_segment)
        # Assign epochs
        for segment in block_segments:
            if len(segment) < 2:
                epoch[segment] = 1  # Treat single-trial blocks as early
                continue
            mid_point = len(segment) // 2
            early_indices = segment[:mid_point]
            late_indices = segment[mid_point:]
            epoch[early_indices] = 1  # Early epoch
            epoch[late_indices] = 2   # Late epoch
    
    # Compute is_post_opto and is_pre_opto
    is_post_opto = np.zeros(len(trial_types), dtype=int)
    is_pre_opto = np.zeros(len(trial_types), dtype=int)
    opto_tag = np.array(opto_tag)
    for i in range(len(trial_types)):
        if i > 0 and opto_tag[i-1] == 1:
            is_post_opto[i] = 1
        if i < len(trial_types) - 1 and opto_tag[i+1] == 1:
            is_pre_opto[i] = 1
    
    # Get number of trials
    number_of_trials = data_single_session.get('nTrials', 0)
    
    for i in range(number_of_trials):
        trial_states = data_single_session['RawEvents']['Trial'][i]['States']
        trial_events = data_single_session['RawEvents']['Trial'][i]['Events']
        
        # Skip if trial data is incomplete
        if 'WindowChoice' not in trial_states:
            continue
        
        choice_window_start = trial_states.get('WindowChoice', [0])[0]
        # Normal contingency
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
                            short_ISI_reward_left_correct_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
                    
                    # Check if Port3In (right port) exists - incorrect lick
                    if 'Port3In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            short_ISI_reward_right_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
                
                # Incorrect right lick (punished)
                elif outcome[i] == 'Punish':
                    # Check if Port3In (right port) exists
                    if 'Port3In' in trial_events:
                        # Process incorrect right licks in punished short ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            short_ISI_punish_right_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
                    
                    # Check if Port1In (left port) exists - correct lick
                    if 'Port1In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            short_ISI_punish_left_correct_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
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
                            long_ISI_reward_right_correct_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
                    
                    # Check if Port1In (left port) exists - incorrect lick
                    if 'Port1In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            long_ISI_reward_left_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
                
                # Incorrect left lick (punished)
                elif outcome[i] == 'Punish':
                    # Check if Port1In (left port) exists
                    if 'Port1In' in trial_events:
                        # Process incorrect left licks in punished long ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            long_ISI_punish_left_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
                    
                    # Check if Port3In (right port) exists - correct lick
                    if 'Port3In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            long_ISI_punish_right_correct_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
        
        # Reverse contingency
        elif contingency_idx == 2:
            # Short ISI trials
            if trial_type[i] == 1:
                # Correct left lick (rewarded)
                if outcome[i] == 'Reward':
                    # Check if Port3In (right port) exists
                    if 'Port3In' in trial_events:
                        # Process correct left licks in rewarded short ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            long_ISI_reward_right_correct_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
                    
                    # Check if Port1In (left port) exists - incorrect lick
                    if 'Port1In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            short_ISI_reward_right_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
                
                # Incorrect right lick (punished)
                elif outcome[i] == 'Punish':
                    # Check if Port1In (left port) exists
                    if 'Port1In' in trial_events:
                        # Process incorrect right licks in punished short ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            long_ISI_punish_left_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
                    
                    # Check if Port3In (right port) exists - correct lick
                    if 'Port3In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            short_ISI_punish_left_correct_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
            
            # Long ISI trials
            elif trial_type[i] == 2:
                # Correct right lick (rewarded)
                if outcome[i] == 'Reward':
                    # Check if Port1In (left port) exists
                    if 'Port1In' in trial_events:
                        # Process correct right licks in rewarded long ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            short_ISI_reward_left_correct_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
                    
                    # Check if Port3In (right port) exists - incorrect lick
                    if 'Port3In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            long_ISI_reward_left_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
                
                # Incorrect left lick (punished)
                elif outcome[i] == 'Punish':
                    # Check if Port3In (right port) exists
                    if 'Port3In' in trial_events:
                        # Process incorrect left licks in punished long ISI trial
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port3In', 'Port3Out',
                            short_ISI_punish_right_incorrect_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
                        )
                    
                    # Check if Port1In (left port) exists - correct lick
                    if 'Port1In' in trial_events:
                        process_lick_data(
                            trial_events, trial_states, choice_window_start, 'Port1In', 'Port1Out',
                            long_ISI_punish_right_correct_lick, trial_ISI[i], opto_tag[i], block_type[i],
                            epoch[i], is_rare[i], is_post_opto[i], is_pre_opto[i], i
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

def process_lick_data(trial_events, trial_states, choice_window_start, port_in, port_out, data_dict, trial_isi, opto_tag, block_type, epoch, is_rare, is_post_opto, is_pre_opto, trial_number):
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
        data_dict['epoch'].append(epoch)
        data_dict['is_rare'].append(is_rare)
        data_dict['is_post_opto'].append(is_post_opto)
        data_dict['is_pre_opto'].append(is_pre_opto)
        data_dict['trial_number'].append(trial_number)