import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import sem
import os
import pandas as pd

def plot_block_transitions(sessions_data, subject, data_paths, save_path=None):
    """
    Plot the fraction of trial outcomes aligned on block transitions (S->L and L->S),
    separated by Control (Non-Opto), Opto, and All trials.
    
    Grid: (n_sessions + 1) x 6
    Cols 0-1: Control (S->L, L->S)
    Cols 2-3: Opto    (S->L, L->S)
    Cols 4-5: All     (S->L, L->S)
    
    Window: 20 trials.
    Y-Lim: (-0.05, 1.05).
    """

    # --- Helper Functions ---

    def calculate_fractions(transitions, outcomes, rare_mask, window=20):
        """
        Calculate fractions and SEM around transition events.
        """
        outcome_types = ['Reward', 'Punish', 'DidNotChoose', 'Rare']
        fractions = {outcome: np.zeros(2 * window + 1) for outcome in outcome_types}
        fractions_sem = {outcome: np.zeros(2 * window + 1) for outcome in outcome_types}
        trial_positions = np.arange(-window, window + 1)
        
        for outcome in outcome_types:
            for pos in range(-window, window + 1):
                if not transitions:
                    continue
                outcome_counts = []
                for t in transitions:
                    trial_idx = t + pos
                    if 0 <= trial_idx < len(outcomes) and outcomes[trial_idx] is not None:
                        if outcome == 'Rare':
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

    def identify_rare_vs_majority_trials(trial_types, block_types):
        """Identify rare vs majority trials."""
        trial_types = np.array(trial_types)
        block_types = np.array(block_types)
        
        rare_mask = np.zeros(len(trial_types), dtype=bool)
        
        short_block_mask = block_types == 1
        rare_mask |= short_block_mask & (trial_types == 2)
        
        long_block_mask = block_types == 2
        rare_mask |= long_block_mask & (trial_types == 1)
        
        return {'rare_mask': rare_mask}

    def filter_transitions_by_opto(transitions, opto_tags, condition):
        """
        Filter transition indices based on the opto tag of the transition trial.
        """
        if not transitions:
            return []
            
        trans_array = np.array(transitions)
        # Ensure opto_tags is safe to index
        valid_indices = trans_array[trans_array < len(opto_tags)]
        
        if len(valid_indices) == 0:
            return []

        current_optos = opto_tags[valid_indices]
        
        if condition == 'control':
            return valid_indices[current_optos == 0].tolist()
        elif condition == 'opto':
            return valid_indices[current_optos != 0].tolist()
        else: # 'all'
            return transitions

    def plot_fractions(ax, fractions, fractions_sem, trial_positions, title, n_transitions):
        """Plot fractions with SEM."""
        colors = {'Reward': 'green', 'Punish': 'red', 'DidNotChoose': 'gray', 'Rare': 'blue'}
        
        # Check for data
        has_data = False
        for outcome in fractions:
            if np.any(fractions[outcome] > 0):
                has_data = True
                break
        
        if not has_data and n_transitions == 0:
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
        
        ax.axvline(x=0, color='black', linestyle='--', label='Transition')
        ax.set_title(f"{title}\n({n_transitions} Trans)", fontsize=9)
        
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel('Fraction')
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel('Trial Relative to Transition')
            
        # Legend on top right plot
        if ax.get_subplotspec().is_first_row() and ax.get_subplotspec().is_last_col():
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
            
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

    # Initialize Figure
    fig = plt.figure(figsize=(32, 3 * (n_sessions + 1)))
    gs = gridspec.GridSpec(n_sessions + 1, 6, figure=fig, wspace=0.3, hspace=0.6)
    fig.suptitle(f"Outcomes around Block Transitions - {subject}", fontsize=16, y=0.99)

    # Configs: (Filter Condition, Transition Type, Column Index, Title Suffix)
    configs = [
        ('control', 'S->L', 0, 'Control'), ('control', 'L->S', 1, 'Control'),
        ('opto',    'S->L', 2, 'Opto'),    ('opto',    'L->S', 3, 'Opto'),
        ('all',     'S->L', 4, 'All'),     ('all',     'L->S', 5, 'All')
    ]

    window = 20

    # --- 1. POOLED DATA ---
    all_outcomes = np.concatenate(outcomes_list)
    all_block_types = np.concatenate(block_types_list)
    all_trial_types = np.concatenate(trial_types_list) if trial_types_list[0] is not None else None
    
    # Process Pooled Opto Tags
    all_opto_list = []
    for i, ot in enumerate(opto_list):
        n = len(outcomes_list[i])
        if ot is None or len(ot) == 0:
            all_opto_list.append(np.zeros(n))
        else:
            temp = pd.to_numeric(ot, errors='coerce')
            temp = np.nan_to_num(temp, nan=0.0)
            if len(temp) < n: temp = np.pad(temp, (0, n-len(temp)), constant_values=0)
            elif len(temp) > n: temp = temp[:n]
            all_opto_list.append(temp)
    all_opto_tags = np.concatenate(all_opto_list)

    # Find Pooled Transitions
    s2l_pooled = []
    l2s_pooled = []
    for i in range(1, len(all_block_types)):
        if all_block_types[i-1] == 1 and all_block_types[i] == 2:
            s2l_pooled.append(i)
        elif all_block_types[i-1] == 2 and all_block_types[i] == 1:
            l2s_pooled.append(i)

    rare_mask_pooled = None
    if all_trial_types is not None:
        rare_mask_pooled = identify_rare_vs_majority_trials(all_trial_types, all_block_types)['rare_mask']
    else:
        rare_mask_pooled = np.zeros(len(all_outcomes), dtype=bool)

    # Plot Pooled
    for cond, trans_type, col, suffix in configs:
        ax = fig.add_subplot(gs[0, col])
        
        base_trans = s2l_pooled if trans_type == 'S->L' else l2s_pooled
        final_trans = filter_transitions_by_opto(base_trans, all_opto_tags, cond)
        
        full_title = f"Pooled: {suffix} {trans_type}"
        
        if final_trans:
            f, s, t_pos = calculate_fractions(final_trans, all_outcomes, rare_mask_pooled, window=window)
            plot_fractions(ax, f, s, t_pos, full_title, len(final_trans))
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(full_title, fontsize=9)
            ax.axis('off')

    # --- 2. INDIVIDUAL SESSIONS ---
    for i, (date, outcomes, block_types, trial_types, opto_raw) in enumerate(zip(dates, outcomes_list, block_types_list, trial_types_list, opto_list)):
        
        # Prepare Opto
        n = len(outcomes)
        if opto_raw is None or len(opto_raw) == 0:
            opto_sess = np.zeros(n)
        else:
            temp = pd.to_numeric(opto_raw, errors='coerce')
            opto_sess = np.nan_to_num(temp, nan=0.0)
            if len(opto_sess) < n: opto_sess = np.pad(opto_sess, (0, n-len(opto_sess)), constant_values=0)
            elif len(opto_sess) > n: opto_sess = opto_sess[:n]
        
        # Find Transitions
        s2l = []
        l2s = []
        for j in range(1, len(block_types)):
            if block_types[j-1] == 1 and block_types[j] == 2:
                s2l.append(j)
            elif block_types[j-1] == 2 and block_types[j] == 1:
                l2s.append(j)
        
        rare_mask = identify_rare_vs_majority_trials(trial_types, block_types)['rare_mask'] if trial_types is not None else np.zeros(len(outcomes), dtype=bool)

        for cond, trans_type, col, suffix in configs:
            ax = fig.add_subplot(gs[i + 1, col])
            
            base_trans = s2l if trans_type == 'S->L' else l2s
            final_trans = filter_transitions_by_opto(base_trans, opto_sess, cond)
            
            plot_title = f"{date}"

            if final_trans:
                f, s, t_pos = calculate_fractions(final_trans, outcomes, rare_mask, window=window)
                plot_fractions(ax, f, s, t_pos, plot_title, len(final_trans))
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(plot_title, fontsize=8)
                ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    if save_path:
        s_str = data_paths[-1].split("_")[-2] if len(data_paths) > 0 else "end"
        e_str = data_paths[0].split("_")[-2] if len(data_paths) > 0 else "start"
        output_path = os.path.join(save_path, f'Trial_by_trial_adaptation_SplitOpto_{subject}_{s_str}_{e_str}.pdf')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close(fig)