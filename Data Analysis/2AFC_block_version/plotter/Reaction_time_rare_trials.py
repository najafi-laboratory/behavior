import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import sem
import os
import pandas as pd

def plot_reaction_time_rare_trials(sessions_data, subject, data_paths, save_path=None):
    """
    Plot the Reaction Time (RT) aligned on Rare trials, separated by Reward vs. Non-Reward outcomes.
    
    Grid Structure (6 columns):
    Cols 0-1: Control (Rare in Short, Rare in Long)
    Cols 2-3: Opto    (Rare in Short, Rare in Long)
    Cols 4-5: All     (Rare in Short, Rare in Long)
    
    Window is 3 trials before and 3 trials after the rare trial.
    """

    # --- Helper Functions ---

    def reconstruct_rt_array(lick_props, n_trials):
        """Consolidate RTs into a single array aligned with trial indices."""
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

    def identify_rare_indices(trial_types, block_types):
        """Identify indices of rare trials."""
        trial_types = np.array(trial_types)
        block_types = np.array(block_types)
        
        # Short blocks (Type 1): Rare trials are Long (Type 2)
        rare_in_short_mask = (block_types == 1) & (trial_types == 2)
        rare_in_short_indices = np.where(rare_in_short_mask)[0]
        
        # Long blocks (Type 2): Rare trials are Short (Type 1)
        rare_in_long_mask = (block_types == 2) & (trial_types == 1)
        rare_in_long_indices = np.where(rare_in_long_mask)[0]
        
        return rare_in_short_indices, rare_in_long_indices

    def get_rare_chunks(triggers, rt_array, outcome_array, opto_array, window=3):
        """
        Collect chunks of RT, Outcome, and Opto data around triggers.
        """
        rt_chunks = []
        outcome_chunks = []
        opto_chunks = []
        
        for t in triggers:
            start_idx = t - window
            end_idx = t + window + 1
            
            # Initialize chunks
            rt_chunk = np.full(2 * window + 1, np.nan)
            outcome_chunk = np.full(2 * window + 1, None, dtype=object)
            opto_chunk = np.full(2 * window + 1, 0.0)
            
            # Valid range
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
        """
        if rt_matrix is None:
            return None, None, None, None

        # --- 1. Apply Opto Filter ---
        rt_filtered = rt_matrix.copy()
        opto_clean = np.nan_to_num(opto_matrix, nan=0.0)
        
        if filter_mode == 'control':
            mask_keep = (opto_clean == 0)
            rt_filtered[~mask_keep] = np.nan
        elif filter_mode == 'opto':
            mask_keep = (opto_clean != 0)
            rt_filtered[~mask_keep] = np.nan
        elif filter_mode == 'all':
            pass

        # --- 2. Split by Outcome ---
        reward_mask = (outcome_matrix == 'Reward')
        non_reward_mask = (outcome_matrix != 'Reward')

        # Reward Stats
        rt_reward = rt_filtered.copy()
        rt_reward[~reward_mask] = np.nan
        
        with np.errstate(invalid='ignore'):
            mean_rew = np.nanmean(rt_reward, axis=0)
            sem_rew = sem(rt_reward, axis=0, nan_policy='omit')
            
            # Non-Reward Stats
            rt_non = rt_filtered.copy()
            rt_non[~non_reward_mask] = np.nan
            
            mean_non = np.nanmean(rt_non, axis=0)
            sem_non = sem(rt_non, axis=0, nan_policy='omit')

        return mean_rew, sem_rew, mean_non, sem_non

    def plot_dual_rt_trace(ax, mean_rew, sem_rew, mean_non, sem_non, trial_positions, title):
        """Plot both Reward and Non-Reward traces on the same axis."""
        
        has_data = False
        if mean_rew is not None and not np.all(np.isnan(mean_rew)): has_data = True
        if mean_non is not None and not np.all(np.isnan(mean_non)): has_data = True
            
        if not has_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        # Plot Reward (Green)
        ax.plot(trial_positions, mean_rew, marker='o', color='forestgreen', label='Reward', markersize=3, linewidth=1)
        if not np.all(np.isnan(sem_rew)):
            ax.fill_between(trial_positions, mean_rew - sem_rew, mean_rew + sem_rew, color='forestgreen', alpha=0.2)

        # Plot Non-Reward (Red/Gray)
        ax.plot(trial_positions, mean_non, marker='o', color='red', label='Non-Reward', markersize=3, linestyle='--', linewidth=1)
        if not np.all(np.isnan(sem_non)):
            ax.fill_between(trial_positions, mean_non - sem_non, mean_non + sem_non, color='red', alpha=0.2)
        
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.5, label='Rare Trial')
        
        ax.set_title(title, fontsize=9)
        
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel('RT (s)')
            
        ax.set_xticks(np.arange(-3, 4, 1))
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.set_ylim(0, 1.5) # Adjust limit as needed

    # --- Main Processing ---

    dates = sessions_data['dates']
    block_types_list = sessions_data['block_type']
    trial_types_list = sessions_data['trial_types']
    lick_properties_list = sessions_data['lick_properties']
    outcomes_list = sessions_data['outcomes']
    opto_list = sessions_data.get('opto_tags', [[]]*len(dates))
    n_sessions = len(dates)

    # Initialize figure: (n_sessions + 1) rows, 6 columns
    fig = plt.figure(figsize=(32, 3 * (n_sessions + 1)))
    gs = gridspec.GridSpec(n_sessions + 1, 6, figure=fig, wspace=0.3, hspace=0.6)
    fig.suptitle(f"Reaction Time: Reward vs Non-Reward around Rare Trials - {subject}", fontsize=16, y=0.99)

    # Containers for Pooled Data
    pooled_data = {
        'control': {'short_rt': [], 'short_out': [], 'short_opto': [], 'long_rt': [], 'long_out': [], 'long_opto': []},
        'opto':    {'short_rt': [], 'short_out': [], 'short_opto': [], 'long_rt': [], 'long_out': [], 'long_opto': []},
        'all':     {'short_rt': [], 'short_out': [], 'short_opto': [], 'long_rt': [], 'long_out': [], 'long_opto': []}
    }

    window = 3
    trial_positions = np.arange(-window, window + 1)
    
    # Map: (Filter Condition, Rare Type, Column Index)
    column_map = [
        ('control', 'Short', 0), ('control', 'Long', 1),
        ('opto',    'Short', 2), ('opto',    'Long', 3),
        ('all',     'Short', 4), ('all',     'Long', 5)
    ]

    for i, (date, lick_props, block_types, trial_types, outcomes, opto_tags) in enumerate(zip(dates, lick_properties_list, block_types_list, trial_types_list, outcomes_list, opto_list)):
        n_trials = len(block_types)
        
        # 1. Reconstruct Arrays
        rt_array = reconstruct_rt_array(lick_props, n_trials)
        outcome_array = np.array(outcomes)
        
        # Safe Opto Conversion
        if opto_tags is None or len(opto_tags) == 0:
            opto_array = np.zeros(n_trials)
        else:
            opto_raw = pd.to_numeric(opto_tags, errors='coerce')
            opto_array = np.nan_to_num(opto_raw, nan=0.0)
            if len(opto_array) < n_trials:
                opto_array = np.pad(opto_array, (0, n_trials - len(opto_array)), constant_values=0)
            elif len(opto_array) > n_trials:
                opto_array = opto_array[:n_trials]

        # 2. Identify Rare Indices
        rare_short_idx, rare_long_idx = identify_rare_indices(trial_types, block_types)
        
        # 3. Get Chunks
        rt_s, out_s, opto_s = get_rare_chunks(rare_short_idx, rt_array, outcome_array, opto_array, window)
        rt_l, out_l, opto_l = get_rare_chunks(rare_long_idx, rt_array, outcome_array, opto_array, window)

        # 4. Collect for Pooling (Store unfiltered chunks)
        if rt_s is not None:
            for key in pooled_data:
                pooled_data[key]['short_rt'].append(rt_s)
                pooled_data[key]['short_out'].append(out_s)
                pooled_data[key]['short_opto'].append(opto_s)
        if rt_l is not None:
            for key in pooled_data:
                pooled_data[key]['long_rt'].append(rt_l)
                pooled_data[key]['long_out'].append(out_l)
                pooled_data[key]['long_opto'].append(opto_l)

        # 5. Plot Individual Session
        for cond, rare_type, col_idx in column_map:
            ax = fig.add_subplot(gs[i + 1, col_idx])
            
            # Select correct data
            curr_rt = rt_s if rare_type == 'Short' else rt_l
            curr_out = out_s if rare_type == 'Short' else out_l
            curr_opto = opto_s if rare_type == 'Short' else opto_l
            
            mr, sr, mn, sn = calculate_filtered_statistics(curr_rt, curr_out, curr_opto, cond)
            
            t_str = f"{date}" if i > 0 else f"{cond.capitalize()} {rare_type}\n{date}"
            plot_dual_rt_trace(ax, mr, sr, mn, sn, trial_positions, t_str)

    # --- Process Pooled Data ---
    
    top_titles = [
        "Control: Rare in Short", "Control: Rare in Long",
        "Opto: Rare in Short",    "Opto: Rare in Long",
        "All: Rare in Short",     "All: Rare in Long"
    ]

    for (cond, rare_type, col_idx), title in zip(column_map, top_titles):
        ax_pool = fig.add_subplot(gs[0, col_idx])
        
        key_rt = 'short_rt' if rare_type == 'Short' else 'long_rt'
        key_out = 'short_out' if rare_type == 'Short' else 'long_out'
        key_opto = 'short_opto' if rare_type == 'Short' else 'long_opto'
        
        list_rt = pooled_data[cond][key_rt]
        list_out = pooled_data[cond][key_out]
        list_opto = pooled_data[cond][key_opto]
        
        if list_rt:
            pool_rt = np.vstack(list_rt)
            pool_out = np.vstack(list_out)
            pool_opto = np.vstack(list_opto)
            
            pmr, psr, pmn, psn = calculate_filtered_statistics(pool_rt, pool_out, pool_opto, cond)
            
            plot_dual_rt_trace(ax_pool, pmr, psr, pmn, psn, trial_positions, f"{title}\n(n={pool_rt.shape[0]})")
            
            if col_idx == 5:
                ax_pool.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        else:
            ax_pool.text(0.5, 0.5, 'No Data', ha='center')
            ax_pool.set_title(title)

    plt.tight_layout()
    
    if save_path:
        s_str = data_paths[-1].split("_")[-2] if len(data_paths) > 0 else "end"
        e_str = data_paths[0].split("_")[-2] if len(data_paths) > 0 else "start"
        output_filename = f'Reaction_Time_Rare_Trials_SplitOpto_{subject}_{s_str}_{e_str}.pdf'
        output_path = os.path.join(save_path, output_filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close(fig)