import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import sem
import os

def plot_block_transitions(sessions_data, subject, data_paths, save_path=None):
    """
    Plot the fraction of Reward, Punish, and DidNotChoose outcomes aligned on block transitions
    (short-to-long and long-to-short) for multiple sessions, including pooled data, using GridSpec.
    Save the plots to a PDF file.

    Args:
        sessions_data (dict): Dictionary containing session data from prepare_session_data function.
        output_pdf (str): Path to save the output PDF file.

    Returns:
        None: Generates and saves plots to a PDF file.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    # Extract session data
    dates = sessions_data['dates']
    outcomes_list = sessions_data['outcomes']
    block_types_list = sessions_data['block_type']
    n_sessions = len(dates)

    def calculate_fractions(transitions, outcomes, window=10):
        """
        Calculate fractions and SEM of Reward, Punish, DidNotChoose for trials around transitions.
        
        Args:
            transitions (list): Indices of transition trials
            outcomes (np.array): Array of trial outcomes
            window (int): Number of trials before/after transition to consider
        
        Returns:
            tuple: (fractions, fractions_sem, trial_positions)
        """
        outcome_types = ['Reward', 'Punish', 'DidNotChoose']
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
                        outcome_counts.append(1 if outcomes[trial_idx] == outcome else 0)
                fractions[outcome][pos + window] = np.mean(outcome_counts) if outcome_counts else 0
                fractions_sem[outcome][pos + window] = sem(outcome_counts, nan_policy='omit') if outcome_counts else 0
        
        return fractions, fractions_sem, trial_positions

    def plot_fractions(ax, fractions, fractions_sem, trial_positions, title):
        """
        Plot the fractions with SEM for each outcome type on the given axis.
        
        Args:
            ax: Matplotlib axis to plot on
            fractions (dict): Fractions for each outcome type
            fractions_sem (dict): SEM for each outcome type
            trial_positions (np.array): Trial positions relative to transition
            title (str): Title of the plot
        """
        colors = {'Reward': 'green', 'Punish': 'red', 'DidNotChoose': 'gray'}
        for outcome, frac in fractions.items():
            ax.plot(trial_positions, frac, marker='o', label=outcome, color=colors[outcome])
            ax.fill_between(trial_positions, 
                           frac - fractions_sem[outcome], 
                           frac + fractions_sem[outcome], 
                           color=colors[outcome], alpha=0.2)
        
        ax.axvline(x=0, color='black', linestyle='--', label='Transition')
        ax.set_xlabel('Trial Relative to Transition')
        ax.set_ylabel('Fraction of Trials')
        ax.set_title(title)
        ax.legend()
        ax.set_xticks(np.arange(-10, 11, 2))
        ax.set_ylim(0, 1)
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Initialize PDF
    # Create figure with GridSpec: n_sessions + 1 rows (pooled + individual), 2 columns
    fig = plt.figure(figsize=(12, 4 * (n_sessions + 1)))
    gs = gridspec.GridSpec(n_sessions + 1, 2, figure=fig)

    # Process pooled data first
    all_outcomes = np.concatenate(outcomes_list)
    all_block_types = np.concatenate(block_types_list)
    short_to_long_transitions = []
    long_to_short_transitions = []

    # Identify transitions in pooled data
    for i in range(1, len(all_block_types)):
        prev_block = all_block_types[i-1]
        curr_block = all_block_types[i]
        if prev_block == 1 and curr_block == 2:
            short_to_long_transitions.append(i)
        elif prev_block == 2 and curr_block == 1:
            long_to_short_transitions.append(i)

    # Plot pooled data
    ax1 = fig.add_subplot(gs[0, 0])
    if short_to_long_transitions:
        fractions_s2l, sem_s2l, trial_positions = calculate_fractions(short_to_long_transitions, all_outcomes)
        plot_fractions(ax1, fractions_s2l, sem_s2l, trial_positions, 'Pooled Short to Long Transitions')
    else:
        ax1.text(0.5, 0.5, 'No Short to Long Transitions', ha='center', va='center')
        ax1.set_title('Pooled Short to Long Transitions')
        ax1.grid(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    if long_to_short_transitions:
        fractions_l2s, sem_l2s, trial_positions = calculate_fractions(long_to_short_transitions, all_outcomes)
        plot_fractions(ax2, fractions_l2s, sem_l2s, trial_positions, 'Pooled Long to Short Transitions')
    else:
        ax2.text(0.5, 0.5, 'No Long to Short Transitions', ha='center', va='center')
        ax2.set_title('Pooled Long to Short Transitions')
        ax2.grid(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

    # Process individual sessions
    for i, (date, outcomes, block_types) in enumerate(zip(dates, outcomes_list, block_types_list)):
        short_to_long_transitions = []
        long_to_short_transitions = []

        # Identify transitions for this session
        for j in range(1, len(block_types)):
            prev_block = block_types[j-1]
            curr_block = block_types[j]
            if prev_block == 1 and curr_block == 2:
                short_to_long_transitions.append(j)
            elif prev_block == 2 and curr_block == 1:
                long_to_short_transitions.append(j)

        # Plot short-to-long transitions
        ax1 = fig.add_subplot(gs[i + 1, 0])
        if short_to_long_transitions:
            fractions_s2l, sem_s2l, trial_positions = calculate_fractions(short_to_long_transitions, outcomes)
            plot_fractions(ax1, fractions_s2l, sem_s2l, trial_positions, f'Short to Long Transitions - {date}')
        else:
            ax1.text(0.5, 0.5, 'No Short to Long Transitions', ha='center', va='center')
            ax1.set_title(f'Short to Long Transitions - {date}')
            ax1.grid(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)

        # Plot long-to-short transitions
        ax2 = fig.add_subplot(gs[i + 1, 1])
        if long_to_short_transitions:
            fractions_l2s, sem_l2s, trial_positions = calculate_fractions(long_to_short_transitions, outcomes)
            plot_fractions(ax2, fractions_l2s, sem_l2s, trial_positions, f'Long to Short Transitions - {date}')
        else:
            ax2.text(0.5, 0.5, 'No Long to Short Transitions', ha='center', va='center')
            ax2.set_title(f'Long to Short Transitions - {date}')
            ax2.grid(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)

        plt.tight_layout()
        
        if save_path:
            output_path = os.path.join(save_path, f'Trial_by_trial_adaptation_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.close(fig)