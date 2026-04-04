import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import sem
import os
import pandas as pd

def plot_rare_trial_performance(sessions_data, subject, data_paths, save_path=None):
    """
    Plot the fraction of outcomes aligned on Rare trials.
    
    Grid Structure (6 columns):
    Cols 0-1: Control (Rare in Short, Rare in Long) - Non-Opto triggers
    Cols 2-3: Opto    (Rare in Short, Rare in Long) - Opto triggers
    Cols 4-5: All     (Rare in Short, Rare in Long) - All triggers
    
    Analysis window is 3 trials before and 3 trials after the rare trial.
    """
    
    # --- Helper Functions ---
    def identify_rare_indices(trial_types, block_types):
        """
        Identify indices of rare trials, separated by block type.
        """
        trial_types = np.array(trial_types)
        block_types = np.array(block_types)
        
        # Short blocks (Type 1): Rare trials are Long (Type 2)
        short_block_mask = block_types == 1
        rare_in_short_mask = short_block_mask & (trial_types == 2)
        rare_in_short_indices = np.where(rare_in_short_mask)[0]
        
        # Long blocks (Type 2): Rare trials are Short (Type 1)
        long_block_mask = block_types == 2
        rare_in_long_mask = long_block_mask & (trial_types == 1)
        rare_in_long_indices = np.where(rare_in_long_mask)[0]
        
        # General rare mask for plotting the 'Rare' line trace
        rare_mask = rare_in_short_mask | rare_in_long_mask
        
        return rare_in_short_indices, rare_in_long_indices, rare_mask

    def filter_triggers_by_opto(triggers, opto_tags, condition):
        """
        Filter trigger indices based on opto condition.
        """
        if len(triggers) == 0:
            return np.array([])
            
        # Ensure opto_tags matches data length or handle safely
        # Assuming opto_tags is aligned with trials
        
        current_optos = opto_tags[triggers]
        
        if condition == 'control':
            # Keep triggers where opto is 0
            return triggers[current_optos == 0]
        elif condition == 'opto':
            # Keep triggers where opto is NOT 0
            return triggers[current_optos != 0]
        else: # 'all'
            return triggers

    def calculate_fractions(triggers, outcomes, rare_mask, window=3):
        """
        Calculate fractions and SEM around specific trigger events.
        """
        outcome_types = ['Reward', 'Punish', 'DidNotChoose', 'Rare']
        fractions = {outcome: np.zeros(2 * window + 1) for outcome in outcome_types}
        fractions_sem = {outcome: np.zeros(2 * window + 1) for outcome in outcome_types}
        trial_positions = np.arange(-window, window + 1)
        
        # Pre-convert outcomes to array for faster indexing if possible, 
        # but list indexing is fine for small windows.
        
        for outcome in outcome_types:
            for pos in range(-window, window + 1):
                if len(triggers) == 0:
                    continue
                    
                outcome_counts = []
                for t in triggers:
                    trial_idx = t + pos
                    
                    # Boundary check
                    if 0 <= trial_idx < len(outcomes) and outcomes[trial_idx] is not None:
                        if outcome == 'Rare':
                            # Check against the general rare mask
                            outcome_counts.append(1 if rare_mask[trial_idx] else 0)
                        else:
                            outcome_counts.append(1 if outcomes[trial_idx] == outcome else 0)
                
                if outcome_counts:
                    fractions[outcome][pos + window] = np.mean(outcome_counts)
                    fractions_sem[outcome][pos + window] = sem(outcome_counts, nan_policy='omit')
                else:
                    fractions[outcome][pos + window] = 0
                    fractions_sem[outcome][pos + window] = 0
        
        return fractions, fractions_sem, trial_positions

    def plot_fractions(ax, fractions, fractions_sem, trial_positions, title, n_events):
        """
        Plot the fractions with SEM.
        """
        colors = {'Reward': 'green', 'Punish': 'red', 'DidNotChoose': 'gray', 'Rare': 'blue'}
        
        # Check if there is data to plot
        has_data = False
        for outcome in fractions:
            if np.any(fractions[outcome] > 0):
                has_data = True
                break
        
        if not has_data and n_events == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(title, fontsize=8)
            ax.axis('off')
            return

        for outcome, frac in fractions.items():
            ax.plot(trial_positions, frac, marker='o', label=outcome, color=colors[outcome], markersize=3)
            ax.fill_between(trial_positions, 
                            frac - fractions_sem[outcome], 
                            frac + fractions_sem[outcome], 
                            color=colors[outcome], alpha=0.2)
        
        ax.axvline(x=0, color='black', linestyle='--', label='Rare Trigger')
        ax.set_title(f"{title}\n(n={n_events})", fontsize=9)
        
        # Axis labels only on edges
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel('Fraction')
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel('Trial Relative to Rare')
            
        # Legend only on top right plot (last column of first row)
        if ax.get_subplotspec().is_first_row() and ax.get_subplotspec().is_last_col():
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        
        ax.set_xticks(np.arange(trial_positions[0], trial_positions[-1] + 1, 1))
        ax.set_ylim(-0.05, 1.05)
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # --- Main Processing ---

    dates = sessions_data['dates']
    outcomes_list = sessions_data['outcomes']
    block_types_list = sessions_data['block_type']
    trial_types_list = sessions_data.get('trial_types', [None] * len(dates))
    opto_list = sessions_data.get('opto_tags', [[]] * len(dates))
    
    n_sessions = len(dates)

    # Initialize figure: (n_sessions + 1) rows, 6 columns
    fig = plt.figure(figsize=(24, 3 * (n_sessions + 1)))
    gs = gridspec.GridSpec(n_sessions + 1, 6, figure=fig, wspace=0.3, hspace=0.6)
    
    fig.suptitle(f"Performance around Rare Trials - {subject} ({n_sessions} Sessions) \n", fontsize=16, y=0.99)

    # Configuration Map
    # (Filter Condition, Block Type, Column Index, Title Suffix)
    configs = [
        ('control', 'Short', 0, 'Control'), ('control', 'Long', 1, 'Control'),
        ('opto',    'Short', 2, 'Opto'),    ('opto',    'Long', 3, 'Opto'),
        ('all',     'Short', 4, 'All'),     ('all',     'Long', 5, 'All')
    ]

    # --- 1. POOLED DATA (Row 0) ---
    all_outcomes = np.concatenate(outcomes_list)
    all_block_types = np.concatenate(block_types_list)
    all_trial_types = np.concatenate(trial_types_list) if trial_types_list[0] is not None else None
    
    # Process Opto Tags for Pooling
    cleaned_optos = []
    for ot in opto_list:
        if ot is None or len(ot) == 0:
             # If completely missing, assume 0s length of outcomes (heuristic) or handle downstream
             # Here, safer to skip or fill if we knew length. 
             # Assuming opto_list aligned with outcomes_list, we can infer length from outcome list
             pass 
    
    # robust concatenation of opto
    all_opto_list = []
    for i, ot in enumerate(opto_list):
        n = len(outcomes_list[i])
        if ot is None or len(ot) == 0:
            all_opto_list.append(np.zeros(n))
        else:
            # clean nan
            temp = pd.to_numeric(ot, errors='coerce')
            temp = np.nan_to_num(temp, nan=0.0)
            # pad/trim
            if len(temp) < n:
                temp = np.pad(temp, (0, n-len(temp)), constant_values=0)
            elif len(temp) > n:
                temp = temp[:n]
            all_opto_list.append(temp)
    
    all_opto_tags = np.concatenate(all_opto_list)

    if all_trial_types is not None:
        rare_short_idx, rare_long_idx, rare_mask_pooled = identify_rare_indices(all_trial_types, all_block_types)
        
        for cond, blk_type, col, title_suffix in configs:
            ax = fig.add_subplot(gs[0, col])
            
            # Select triggers
            base_triggers = rare_short_idx if blk_type == 'Short' else rare_long_idx
            final_triggers = filter_triggers_by_opto(base_triggers, all_opto_tags, cond)
            
            # Title
            full_title = f"Pooled: {title_suffix} - Rare in {blk_type}"
            
            if len(final_triggers) > 0:
                fracs, sems, x_axis = calculate_fractions(final_triggers, all_outcomes, rare_mask_pooled)
                plot_fractions(ax, fracs, sems, x_axis, full_title, len(final_triggers))
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(full_title, fontsize=9)
                ax.axis('off')

    # --- 2. INDIVIDUAL SESSIONS (Rows 1..N) ---
    for i, (date, outcomes, block_types, trial_types, opto_raw) in enumerate(zip(dates, outcomes_list, block_types_list, trial_types_list, opto_list)):
        
        # Prepare Opto for this session
        n = len(outcomes)
        if opto_raw is None or len(opto_raw) == 0:
            opto_sess = np.zeros(n)
        else:
            temp = pd.to_numeric(opto_raw, errors='coerce')
            opto_sess = np.nan_to_num(temp, nan=0.0)
            if len(opto_sess) < n:
                opto_sess = np.pad(opto_sess, (0, n-len(opto_sess)), constant_values=0)
            elif len(opto_sess) > n:
                opto_sess = opto_sess[:n]
        
        rare_short_idx, rare_long_idx, rare_mask = identify_rare_indices(trial_types, block_types)

        for cond, blk_type, col, title_suffix in configs:
            ax = fig.add_subplot(gs[i + 1, col])
            
            base_triggers = rare_short_idx if blk_type == 'Short' else rare_long_idx
            final_triggers = filter_triggers_by_opto(base_triggers, opto_sess, cond)
            
            # Title for sessions usually just Date or short info
            if i == 0: # Redundant if pooled titles are clear, but helpful
                 plot_title = f"{date}"
            else:
                 plot_title = f"{date}"

            if len(final_triggers) > 0:
                fracs, sems, x_axis = calculate_fractions(final_triggers, outcomes, rare_mask)
                plot_fractions(ax, fracs, sems, x_axis, plot_title, len(final_triggers))
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(plot_title, fontsize=8)
                ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    if save_path:
        s_str = data_paths[-1].split("_")[-2] if len(data_paths) > 0 else "end"
        e_str = data_paths[0].split("_")[-2] if len(data_paths) > 0 else "start"
        output_filename = f'Rare_trial_adaptation_SplitOpto_{subject}_{s_str}_{e_str}.pdf'
        output_path = os.path.join(save_path, output_filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close(fig)