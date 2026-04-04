import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import sem
import os

def plot_performance_by_stimulus(sessions_data, subject, data_paths, save_path=None):
    """
    Plot performance (Reward Rate) aligned on block transitions with 5 conditions:
    1. No Opto Trials (Cols 1-2)
    2. All Trials (Cols 3-4)
    3. Opto Trials Only (Cols 5-6)
    4. Destination is Control Block (Cols 7-8) - NEW
    5. Destination is Opto Block (Cols 9-10) - NEW
    
    Includes visual spacers between conditions.
    """

    def calculate_stimulus_performance(transitions, outcomes, trial_types, opto_tags, 
                                     mode='all', window=20):
        """
        Calculate accuracy separately for Short (1) and Long (2) trials.
        Filters based on opto_tags and mode.
        """
        # Map labels to the values found in trial_types
        stimuli = {'Short Trials': 1, 'Long Trials': 2}
        
        fractions = {stim: np.full(2 * window + 1, np.nan) for stim in stimuli}
        fractions_sem = {stim: np.full(2 * window + 1, np.nan) for stim in stimuli}
        trial_positions = np.arange(-window, window + 1)
        
        for stim_name, stim_val in stimuli.items():
            for i, pos in enumerate(range(-window -1, window)):
                if not transitions:
                    continue
                
                valid_outcomes = []
                
                for t in transitions:
                    # t is the index of the FIRST trial of the NEW block.
                    trial_idx = t + pos
                    
                    # Bounds check
                    if 0 <= trial_idx < len(outcomes):
                        # --- FILTERING LOGIC ---
                        # 1. Check valid outcome
                        # 2. Check stimulus type matches
                        if (outcomes[trial_idx] is not None) and (trial_types[trial_idx] == stim_val):
                            
                            # 3. Check Opto Condition
                            is_opto = (opto_tags[trial_idx] == 1)
                            
                            keep_trial = False
                            if mode == 'all':
                                keep_trial = True
                            elif mode == 'no_opto' and not is_opto:
                                keep_trial = True
                            elif mode == 'opto' and is_opto:
                                keep_trial = True
                                
                            if keep_trial:
                                # Calculate Performance: 1 if Reward, 0 if Punish/Other
                                is_reward = 1 if outcomes[trial_idx] == 'Reward' else 0
                                valid_outcomes.append(is_reward)

                # Only calculate statistics if we actually found trials
                if len(valid_outcomes) > 0:
                    fractions[stim_name][i] = np.mean(valid_outcomes)
                    fractions_sem[stim_name][i] = sem(valid_outcomes)
        
        return fractions, fractions_sem, trial_positions

    def classify_transitions_by_dest_block(transitions, block_types, opto_tags):
        """
        Classify transitions based on whether the DESTINATION block (the block starting at t)
        is a Control Block (no opto trials) or an Opto Block (contains opto trials).
        """
        trans_control = []
        trans_opto = []

        total_trials = len(block_types)

        for t in transitions:
            # The transition t is the start index of the new block.
            # We need to find the end of this block to check its content.
            current_block_type = block_types[t]
            
            # Find the end of the current block
            end_idx = t + 1
            while end_idx < total_trials and block_types[end_idx] == current_block_type:
                end_idx += 1
            
            # Slice the opto tags for this block
            block_opto_slice = opto_tags[t:end_idx]
            
            # Check if any trial in this block was Opto
            if np.sum(block_opto_slice) > 0:
                trans_opto.append(t)
            else:
                trans_control.append(t)
                
        return trans_control, trans_opto

    def plot_traces(ax, fractions, fractions_sem, trial_positions, title, n_transitions, show_ylabel=False):
        """
        Plot the two traces (Short vs Long) handling NaNs.
        """
        colors = {'Short Trials': '#1f77b4', 'Long Trials': '#d62728'} # Blue vs Red
        
        has_data = False
        for stim_name, val_array in fractions.items():
            # Check if there is any data to plot
            if not np.all(np.isnan(val_array)):
                has_data = True
            
            sem_array = fractions_sem[stim_name]
            
            ax.plot(trial_positions, val_array, marker='o', markersize=3, 
                    label=stim_name, color=colors[stim_name], alpha=0.8)
            
            valid_mask = ~np.isnan(val_array)
            if np.any(valid_mask):
                ax.fill_between(trial_positions[valid_mask], 
                                (val_array - sem_array)[valid_mask], 
                                (val_array + sem_array)[valid_mask], 
                                color=colors[stim_name], alpha=0.2)
        
        # Vertical line at transition
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Axis setup
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(np.arange(-20, 21, 10))
        
        if show_ylabel:
            ax.set_ylabel('Reward Rate')
        else:
            ax.set_yticklabels([]) # Hide Y ticks

        # Title
        if n_transitions > 0:
            ax.set_title(f"{title}", fontsize=9)
        else:
            ax.set_title(f"{title}\n(No Trans)", fontsize=8, color='gray')

        # Aesthetics
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True, axis='y', linestyle=':', alpha=0.3)

    # --- Main Execution Block ---

    dates = sessions_data['dates']
    outcomes_list = sessions_data['outcomes']
    block_types_list = sessions_data['block_type']
    trial_types_list = sessions_data.get('trial_types', [None] * len(dates)) 
    opto_tags_list = sessions_data.get('opto_tags', [[0]*len(o) for o in outcomes_list])
    n_sessions = len(dates)

    # Figure Layout Setup
    # 5 Groups of 2 columns each, plus 4 spacers = 14 columns
    # Groups: [NoOpto], [All], [OptoOnly], [ControlBlock], [OptoBlock]
    width_ratios = [
        1, 1, 0.2, # No Opto
        1, 1, 0.2, # All
        1, 1, 0.2, # Opto Only
        1, 1, 0.2, # Control Block (Dest)
        1, 1       # Opto Block (Dest)
    ]
    
    n_cols = len(width_ratios)
    fig = plt.figure(figsize=(55, 3.5 * (n_sessions + 1) + 2)) 
    gs = gridspec.GridSpec(n_sessions + 1, n_cols, figure=fig, width_ratios=width_ratios)

    fig.suptitle(
        f"Performance by Block Transition Type\nSubject: {subject} | Sessions: {n_sessions}\n"
        f"Groups: No Opto Trials | All Trials | Opto Trials | Dest is Control Block | Dest is Opto Block",
        fontsize=18, fontweight='bold', y=0.98
    )

    # Helper map: (Title Prefix, S->L Col, L->S Col, Trial Filter Mode, Use Block Filter?)
    # If Use Block Filter is True, we use the pre-calculated list of specific transitions.
    # If False, we use the full list of transitions.
    
    # We will construct the loop dynamically below to handle the specific lists.
    
    # --- 1. Process Pooled Data ---
    all_outcomes = np.concatenate(outcomes_list)
    all_block_types = np.concatenate(block_types_list)
    all_trial_types = np.concatenate(trial_types_list) if trial_types_list[0] is not None else None
    all_opto_tags = np.concatenate(opto_tags_list)

    if all_trial_types is not None:
        # A. Find all base transitions
        short_to_long_all = []
        long_to_short_all = []
        
        for i in range(1, len(all_block_types)):
            # Basic transition check
            if all_block_types[i-1] == 1 and all_block_types[i] == 2:
                short_to_long_all.append(i)
            elif all_block_types[i-1] == 2 and all_block_types[i] == 1:
                long_to_short_all.append(i)

        # B. Classify Transitions by Destination Block Type
        s2l_ctl, s2l_opto = classify_transitions_by_dest_block(short_to_long_all, all_block_types, all_opto_tags)
        l2s_ctl, l2s_opto = classify_transitions_by_dest_block(long_to_short_all, all_block_types, all_opto_tags)

        # Define the Groups for plotting
        # Format: (Group Name, Col Index SL, Col Index LS, Trial Filter Mode, Transition List SL, Transition List LS)
        plot_groups = [
            ("No Opto", 0, 1, 'no_opto', short_to_long_all, long_to_short_all),
            ("All Trials", 3, 4, 'all', short_to_long_all, long_to_short_all),
            ("Opto Only", 6, 7, 'opto', short_to_long_all, long_to_short_all),
            ("Control Block", 9, 10, 'all', s2l_ctl, l2s_ctl),    # New: Dest is Control
            ("Opto Block", 12, 13, 'all', s2l_opto, l2s_opto)     # New: Dest is Opto
        ]

        # Plot Pooled Row
        for name, col_sl, col_ls, mode, t_sl, t_ls in plot_groups:
            # S->L
            ax_sl = fig.add_subplot(gs[0, col_sl])
            f_sl, s_sl, tp = calculate_stimulus_performance(t_sl, all_outcomes, all_trial_types, all_opto_tags, mode=mode)
            plot_traces(ax_sl, f_sl, s_sl, tp, f"Pooled {name}: S->L", len(t_sl), show_ylabel=(col_sl==0))
            
            if col_sl == 0:
                ax_sl.legend(loc='lower right', fontsize='x-small')

            # L->S
            ax_ls = fig.add_subplot(gs[0, col_ls])
            f_ls, s_ls, tp = calculate_stimulus_performance(t_ls, all_outcomes, all_trial_types, all_opto_tags, mode=mode)
            plot_traces(ax_ls, f_ls, s_ls, tp, f"Pooled {name}: L->S", len(t_ls), show_ylabel=False)

    # --- 2. Process Individual Sessions ---
    for i, (date, outcomes, block_types, trial_types, opto_tags) in enumerate(zip(dates, outcomes_list, block_types_list, trial_types_list, opto_tags_list)):
        if trial_types is None:
            continue

        # A. Find base transitions
        s2l_all = []
        l2s_all = []
        for j in range(1, len(block_types)):
            if block_types[j-1] == 1 and block_types[j] == 2:
                s2l_all.append(j)
            elif block_types[j-1] == 2 and block_types[j] == 1:
                l2s_all.append(j)
        
        # B. Classify Transitions
        s2l_ctl_sess, s2l_opto_sess = classify_transitions_by_dest_block(s2l_all, block_types, opto_tags)
        l2s_ctl_sess, l2s_opto_sess = classify_transitions_by_dest_block(l2s_all, block_types, opto_tags)

        # Re-define groups with session-specific lists
        sess_plot_groups = [
            ("No Opto", 0, 1, 'no_opto', s2l_all, l2s_all),
            ("All", 3, 4, 'all', s2l_all, l2s_all),
            ("Opto Only", 6, 7, 'opto', s2l_all, l2s_all),
            ("Control Blk", 9, 10, 'all', s2l_ctl_sess, l2s_ctl_sess),
            ("Opto Blk", 12, 13, 'all', s2l_opto_sess, l2s_opto_sess)
        ]

        # Plot Session Row
        for name, col_sl, col_ls, mode, t_sl, t_ls in sess_plot_groups:
            # S->L
            ax_sl = fig.add_subplot(gs[i + 1, col_sl])
            f_sl, s_sl, tp = calculate_stimulus_performance(t_sl, outcomes, trial_types, opto_tags, mode=mode)
            plot_traces(ax_sl, f_sl, s_sl, tp, f"{date} {name}", len(t_sl), show_ylabel=(col_sl==0))

            # L->S
            ax_ls = fig.add_subplot(gs[i + 1, col_ls])
            f_ls, s_ls, tp = calculate_stimulus_performance(t_ls, outcomes, trial_types, opto_tags, mode=mode)
            plot_traces(ax_ls, f_ls, s_ls, tp, f"{date} {name}", len(t_ls), show_ylabel=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    if save_path:
        filename = f'Performance_by_Stimulus_BlockTypes_{subject}.pdf'
        output_path = os.path.join(save_path, filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")

    plt.close(fig)