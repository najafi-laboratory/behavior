import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import sem
import os
import pandas as pd

def plot_reaction_time_transitions(sessions_data, subject, data_paths, save_path=None):
    """
    Plot the Reaction Time (RT) aligned on block transitions, separated by 
    Reward vs. Non-Reward outcomes, and split into Control/Opto/All columns.
    
    Grid: (n_sessions + 1) x 6
    Cols 0-1: Control (S->L, L->S)
    Cols 2-3: Opto    (S->L, L->S)
    Cols 4-5: All     (S->L, L->S)
    """

    # --- Helper Functions ---

    def reconstruct_rt_array(lick_props, n_trials):
        """
        Consolidate RTs into a single array aligned with trial indices.
        """
        rt_array = np.full(n_trials, np.nan)
        
        keys = [
            'short_ISI_reward_left_correct_lick', 'short_ISI_reward_right_incorrect_lick',
            'short_ISI_punish_right_incorrect_lick', 'short_ISI_punish_left_correct_lick',
            'long_ISI_reward_right_correct_lick', 'long_ISI_reward_left_incorrect_lick',
            'long_ISI_punish_left_incorrect_lick', 'long_ISI_punish_right_correct_lick'
        ]
        
        for key in keys:
            if key in lick_props:
                data = lick_props[key]
                trials = data.get('trial_number', [])
                rts = data.get('Lick_reaction_time', [])
                
                for t, rt in zip(trials, rts):
                    if 0 <= t < n_trials:
                        rt_array[t] = rt
                        
        return rt_array

    def get_transition_chunks(transitions, rt_array, outcome_array, opto_array, window=20):
        """
        Extract chunks of RTs, Outcomes, and Opto tags around transitions.
        Returns matrices of shape (n_transitions, 2*window+1).
        """
        rt_chunks = []
        outcome_chunks = []
        opto_chunks = []
        
        for t in transitions:
            start_idx = t - window
            end_idx = t + window + 1
            
            # Initialize chunks with NaNs/None
            rt_chunk = np.full(2 * window + 1, np.nan)
            outcome_chunk = np.full(2 * window + 1, None, dtype=object)
            opto_chunk = np.full(2 * window + 1, 0.0) # Default to 0 (control) if missing
            
            # Valid range in original data
            valid_start = max(0, start_idx)
            valid_end = min(len(rt_array), end_idx)
            
            # Map to chunk indices
            chunk_start = valid_start - start_idx
            chunk_end = chunk_start + (valid_end - valid_start)
            
            # Fill data
            rt_chunk[chunk_start:chunk_end] = rt_array[valid_start:valid_end]
            outcome_chunk[chunk_start:chunk_end] = outcome_array[valid_start:valid_end]
            opto_chunk[chunk_start:chunk_end] = opto_array[valid_start:valid_end]
            
            rt_chunks.append(rt_chunk)
            outcome_chunks.append(outcome_chunk)
            opto_chunks.append(opto_chunk)
            
        if not rt_chunks:
            return None, None, None
            
        return np.vstack(rt_chunks), np.vstack(outcome_chunks), np.vstack(opto_chunks)

    def calculate_filtered_statistics(rt_matrix, outcome_matrix, opto_matrix, filter_mode):
        """
        Calculate Mean and SEM separately for Reward and Non-Reward trials,
        after filtering for specific Opto conditions.
        
        filter_mode: 'control', 'opto', 'all'
        """
        if rt_matrix is None:
            return None, None, None, None

        # --- 1. Apply Opto Filter ---
        # We create a copy of RT matrix and set non-matching trials to NaN
        rt_filtered = rt_matrix.copy()
        
        # Clean opto matrix (handle NaNs as 0)
        opto_clean = np.nan_to_num(opto_matrix, nan=0.0)
        
        if filter_mode == 'control':
            # Keep only 0s. Set everything else to NaN
            # condition: keep if opto == 0
            mask_keep = (opto_clean == 0)
            rt_filtered[~mask_keep] = np.nan
            
        elif filter_mode == 'opto':
            # Keep only non-0s.
            mask_keep = (opto_clean != 0)
            rt_filtered[~mask_keep] = np.nan
            
        elif filter_mode == 'all':
            pass # Keep everything
            
        # If after filtering, everything is NaN (e.g. no opto trials), 
        # subsequent nanmean will return NaNs correctly (and trigger RuntimeWarning, which we can suppress or ignore)

        # --- 2. Split by Outcome ---
        reward_mask = (outcome_matrix == 'Reward')
        non_reward_mask = (outcome_matrix != 'Reward')

        # Reward Stats
        rt_reward = rt_filtered.copy()
        rt_reward[~reward_mask] = np.nan
        
        # Use nanmean to ignore filtered-out trials and NaNs
        with np.errstate(invalid='ignore'): # Suppress warnings for Mean of empty slice
            mean_rew = np.nanmean(rt_reward, axis=0)
            sem_rew = sem(rt_reward, axis=0, nan_policy='omit')
            
            # Non-Reward Stats
            rt_non = rt_filtered.copy()
            rt_non[~non_reward_mask] = np.nan
            
            mean_non = np.nanmean(rt_non, axis=0)
            sem_non = sem(rt_non, axis=0, nan_policy='omit')

        return mean_rew, sem_rew, mean_non, sem_non

    def plot_dual_rt_trace(ax, mean_rew, sem_rew, mean_non, sem_non, trial_positions, title):
        """
        Plot both Reward and Non-Reward traces on the same axis.
        """
        # Check if we have valid data (at least one non-nan value)
        has_data = False
        if mean_rew is not None and not np.all(np.isnan(mean_rew)):
            has_data = True
        if mean_non is not None and not np.all(np.isnan(mean_non)):
            has_data = True
            
        if not has_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        # Plot Reward (Green)
        ax.plot(trial_positions, mean_rew, marker='o', color='forestgreen', label='Reward', markersize=3, linewidth=1)
        # Handle cases where SEM is all NaN (single trial)
        if not np.all(np.isnan(sem_rew)):
            ax.fill_between(trial_positions, mean_rew - sem_rew, mean_rew + sem_rew, color='forestgreen', alpha=0.2)

        # Plot Non-Reward (Gray/Red)
        ax.plot(trial_positions, mean_non, marker='o', color='red', label='Non-Reward', markersize=3, linestyle='--', linewidth=1)
        if not np.all(np.isnan(sem_non)):
            ax.fill_between(trial_positions, mean_non - sem_non, mean_non + sem_non, color='red', alpha=0.2)
        
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
        
        # Labels and Style
        ax.set_title(title, fontsize=9)
        
        # Only set Y-label for the first column
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel('RT (s)')
        
        ax.set_xticks(np.arange(-window, window + 1, 5)) # Sparse ticks
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Set standardized ylim if possible, or leave auto
        # ax.set_ylim(0, 1.5) # Assuming RTs are usually < 1.5s, adjust if needed

    # --- Main Processing ---

    dates = sessions_data['dates']
    block_types_list = sessions_data['block_type']
    lick_properties_list = sessions_data['lick_properties']
    outcomes_list = sessions_data['outcomes']
    opto_list = sessions_data.get('opto_tags', [[]]*len(dates)) # Handle missing key safely
    n_sessions = len(dates)

    # Initialize figure
    # 6 Columns: Control(S->L, L->S), Opto(S->L, L->S), All(S->L, L->S)
    fig = plt.figure(figsize=(32, 3 * (n_sessions + 1)))
    gs = gridspec.GridSpec(n_sessions + 1, 6, figure=fig, wspace=0.3, hspace=0.6)
    fig.suptitle(f"Reaction Time: Reward vs Non-Reward around Transitions - {subject}", fontsize=16, y=0.99)

    # Containers for Pooled Data
    # Structure: pooled_data[condition][transition_type]
    pooled_data = {
        'control': {'s2l_rt': [], 's2l_out': [], 's2l_opto': [], 'l2s_rt': [], 'l2s_out': [], 'l2s_opto': []},
        'opto':    {'s2l_rt': [], 's2l_out': [], 's2l_opto': [], 'l2s_rt': [], 'l2s_out': [], 'l2s_opto': []},
        'all':     {'s2l_rt': [], 's2l_out': [], 's2l_opto': [], 'l2s_rt': [], 'l2s_out': [], 'l2s_opto': []}
    }

    window = 20
    trial_positions = np.arange(-window, window + 1)

    column_map = [
        ('control', 'S->L', 0), ('control', 'L->S', 1),
        ('opto',    'S->L', 2), ('opto',    'L->S', 3),
        ('all',     'S->L', 4), ('all',     'L->S', 5)
    ]

    for i, (date, lick_props, block_types, outcomes, opto_tags) in enumerate(zip(dates, lick_properties_list, block_types_list, outcomes_list, opto_list)):
        n_trials = len(block_types)
        
        # 1. Reconstruct Arrays
        rt_array = reconstruct_rt_array(lick_props, n_trials)
        outcome_array = np.array(outcomes)
        
        # Handle opto tags safely
        if opto_tags is None or len(opto_tags) == 0:
            opto_array = np.zeros(n_trials)
        else:
            # Clean and ensure numeric
            opto_raw = pd.to_numeric(opto_tags, errors='coerce')
            opto_array = np.nan_to_num(opto_raw, nan=0.0)
            # Pad if length mismatch
            if len(opto_array) < n_trials:
                opto_array = np.pad(opto_array, (0, n_trials - len(opto_array)), constant_values=0)
            elif len(opto_array) > n_trials:
                opto_array = opto_array[:n_trials]

        # 2. Identify Transitions
        s2l_transitions = []
        l2s_transitions = []
        for j in range(1, len(block_types)):
            prev_block = block_types[j-1]
            curr_block = block_types[j]
            if prev_block == 1 and curr_block == 2:
                s2l_transitions.append(j)
            elif prev_block == 2 and curr_block == 1:
                l2s_transitions.append(j)
        
        # 3. Get Chunks (RT, Outcomes, Opto)
        rt_s2l, out_s2l, opto_s2l = get_transition_chunks(s2l_transitions, rt_array, outcome_array, opto_array, window)
        rt_l2s, out_l2s, opto_l2s = get_transition_chunks(l2s_transitions, rt_array, outcome_array, opto_array, window)

        # 4. Collect for Pooling (Store ALL chunks, filtering happens at stats calculation)
        if rt_s2l is not None:
            for key in pooled_data: # Append to all containers, we filter later
                pooled_data[key]['s2l_rt'].append(rt_s2l)
                pooled_data[key]['s2l_out'].append(out_s2l)
                pooled_data[key]['s2l_opto'].append(opto_s2l)
                
        if rt_l2s is not None:
            for key in pooled_data:
                pooled_data[key]['l2s_rt'].append(rt_l2s)
                pooled_data[key]['l2s_out'].append(out_l2s)
                pooled_data[key]['l2s_opto'].append(opto_l2s)

        # 5. Process and Plot per Session
        for cond, trans_type, col_idx in column_map:
            ax = fig.add_subplot(gs[i + 1, col_idx])
            
            # Select correct data
            curr_rt = rt_s2l if trans_type == 'S->L' else rt_l2s
            curr_out = out_s2l if trans_type == 'S->L' else out_l2s
            curr_opto = opto_s2l if trans_type == 'S->L' else opto_l2s
            
            # Calculate filtered stats
            mr, sr, mn, sn = calculate_filtered_statistics(curr_rt, curr_out, curr_opto, cond)
            
            # Title
            if i == 0: # Only verbose title on first row if needed, but here we do per plot
                t_str = f"{cond.capitalize()} {trans_type}\n{date}"
            else:
                t_str = date
                
            plot_dual_rt_trace(ax, mr, sr, mn, sn, trial_positions, t_str)

    # --- Process Pooled Data ---
    
    # Titles for top row
    top_titles = [
        "Control: Short->Long", "Control: Long->Short",
        "Opto: Short->Long",    "Opto: Long->Short",
        "All: Short->Long",     "All: Long->Short"
    ]

    for (cond, trans_type, col_idx), title in zip(column_map, top_titles):
        ax_pool = fig.add_subplot(gs[0, col_idx])
        
        # Retrieve list of matrices
        key_rt = 's2l_rt' if trans_type == 'S->L' else 'l2s_rt'
        key_out = 's2l_out' if trans_type == 'S->L' else 'l2s_out'
        key_opto = 's2l_opto' if trans_type == 'S->L' else 'l2s_opto'
        
        list_rt = pooled_data[cond][key_rt]
        list_out = pooled_data[cond][key_out]
        list_opto = pooled_data[cond][key_opto]
        
        if list_rt:
            # Stack sessions
            pool_rt = np.vstack(list_rt)
            pool_out = np.vstack(list_out)
            pool_opto = np.vstack(list_opto)
            
            # Calculate Stats on Pooled Matrix
            pmr, psr, pmn, psn = calculate_filtered_statistics(pool_rt, pool_out, pool_opto, cond)
            
            plot_dual_rt_trace(ax_pool, pmr, psr, pmn, psn, trial_positions, f"{title}\n(n={pool_rt.shape[0]})")
            
            # Add Legend to the last plot of the top row
            if col_idx == 5:
                ax_pool.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        else:
            ax_pool.text(0.5, 0.5, 'No Transitions', ha='center')
            ax_pool.set_title(title)

    plt.tight_layout() # This might warn with complex grids, acceptable
    
    if save_path:
        s_str = data_paths[-1].split("_")[-2] if len(data_paths) > 0 else "end"
        e_str = data_paths[0].split("_")[-2] if len(data_paths) > 0 else "start"
        output_filename = f'Reaction_Time_Trans_SplitOpto_{subject}_{s_str}_{e_str}.pdf'
        output_path = os.path.join(save_path, output_filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close(fig)