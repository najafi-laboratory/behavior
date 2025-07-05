import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

def plot_session_outcomes_combined(sessions_outcome, sessions_date, ax=None, outcome_colors=None):
    """
    Plot a combined bar chart showing outcome distribution for all sessions,
    with one bar per session and x-ticks showing session dates.
    Shows outcomes as percentages and uses a specific order for outcomes.
    
    Parameters:
    -----------
    sessions_outcome : list of lists
        List containing outcome lists for each session
    sessions_date : list
        List of session dates
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.
    outcome_colors : dict, optional
        Dictionary mapping outcome types to colors
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes with the plot
    """
    if outcome_colors is None:
        outcome_colors = {
            'Reward': '#63f250',         # Green
            'RewardNaive': '#115e07',    # Darker green
            'Punish': '#e74c3c',         # Red
            'PunishNaive': '#6b201f',    # Darker red
            'WrongInitiation': '#f39c12', # Orange
            'ChangingMindReward': '#9b59b6', # Purple
            'EarlyChoice': '#3498db',    # Blue
            'DidNotChoose': '#95a5a6',   # Gray
            'Other': '#7f8c8d'           # Darker gray
        }
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Remove top, right, and left spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Format dates consistently
    formatted_dates = []
    for date in sessions_date:
        if isinstance(date, str):
            try:
                # Try to parse the date string (try different formats)
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y%m%d']:
                    try:
                        date_obj = datetime.strptime(date, fmt)
                        formatted_dates.append(date_obj)
                        break
                    except ValueError:
                        continue
                else:  # If no format works, use the original string
                    formatted_dates.append(date)
            except Exception:
                formatted_dates.append(date)
        else:
            formatted_dates.append(date)
    
    # Define specific order for outcomes (from bottom to top)
    outcome_order = [
        'Reward',
        'RewardNaive',
        'Punish',
        'PunishNaive',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]
    
    # Filter outcome_order to only include outcomes present in the data
    all_outcomes = set()
    for outcomes in sessions_outcome:
        all_outcomes.update(outcomes)
    
    ordered_outcomes = [outcome for outcome in outcome_order if outcome in all_outcomes]
    # Add any outcomes that aren't in our predefined order to the end
    for outcome in sorted(all_outcomes):
        if outcome not in ordered_outcomes:
            ordered_outcomes.append(outcome)
    
    # Set up the x-axis
    x_positions = np.arange(len(sessions_date))
    
    # Calculate total trials per session for percentage calculation
    total_trials = []
    for session_outcomes in sessions_outcome:
        total_trials.append(len(session_outcomes))
    
    # Create one stacked bar for each session
    bottom = np.zeros(len(sessions_date))

    # Bar width
    barwidth = 0.2
    
    for outcome in ordered_outcomes:
        # Calculate the percentage of this outcome in each session
        percentages = []
        for i, session_outcomes in enumerate(sessions_outcome):
            outcome_count = session_outcomes.count(outcome)
            percentage = (outcome_count / total_trials[i] * 100) if total_trials[i] > 0 else 0
            percentages.append(percentage)
        
        # Plot this outcome segment with its color
        color = outcome_colors.get(outcome, outcome_colors.get('Other', '#7f8c8d'))
        ax.bar(x_positions, percentages, bottom=bottom, label=outcome, width= barwidth, color=color)
        
        # Update the bottom for the next outcome segment
        bottom += np.array(percentages)
    
    # Set x-axis labels to session dates with consistent format
    if all(isinstance(date, datetime) for date in formatted_dates):
        date_labels = [date.strftime('%Y-%m-%d') for date in formatted_dates]
    else:
        date_labels = [str(date) for date in formatted_dates]
        
    ax.set_xticks(x_positions)
    ax.set_xticklabels(date_labels, rotation=45, ha='right')
    
    # Set labels and title
    ax.set_xlabel('Session Date')
    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title('Session Outcomes')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc= 'best')
    
    # Set y-axis to show percentages
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))
    
    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    return ax

# -------------------------------------------------------------------------------
# pooled sessions outcome plot (returns just one bar plot)
def plot_pooled_session_outcomes(sessions_outcome, ax=None, outcome_colors=None):
    """
    Plot a single bar chart showing outcome distribution for all sessions pooled together,
    with outcomes as percentages and using a specific order for outcomes.
    
    Parameters:
    -----------
    sessions_outcome : list of lists
        List containing outcome lists for each session
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.
    outcome_colors : dict, optional
        Dictionary mapping outcome types to colors
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes with the plot
    """
    if outcome_colors is None:
        outcome_colors = {
            'Reward': '#63f250',         # Green
            'RewardNaive': '#115e07',    # Darker green
            'Punish': '#e74c3c',         # Red
            'PunishNaive': '#6b201f',    # Darker red
            'WrongInitiation': '#f39c12', # Orange
            'ChangingMindReward': '#9b59b6', # Purple
            'EarlyChoice': '#3498db',    # Blue
            'DidNotChoose': '#95a5a6',   # Gray
            'Other': '#7f8c8d'           # Darker gray
        }
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Remove top, right, and left spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Define specific order for outcomes (from bottom to top)
    outcome_order = [
        'Reward',
        'RewardNaive',
        'Punish',
        'PunishNaive',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]
    
    # Pool all outcomes into a single list
    all_outcomes = []
    for session_outcomes in sessions_outcome:
        all_outcomes.extend(session_outcomes)
    
    # Filter outcome_order to only include outcomes present in the data
    unique_outcomes = set(all_outcomes)
    ordered_outcomes = [outcome for outcome in outcome_order if outcome in unique_outcomes]
    # Add any outcomes that aren't in our predefined order to the end
    for outcome in sorted(unique_outcomes):
        if outcome not in ordered_outcomes:
            ordered_outcomes.append(outcome)
    
    # Calculate total trials for percentage calculation
    total_trials = len(all_outcomes)
    
    # Calculate percentages for each outcome
    percentages = []
    for outcome in ordered_outcomes:
        outcome_count = all_outcomes.count(outcome)
        percentage = (outcome_count / total_trials * 100) if total_trials > 0 else 0
        percentages.append(percentage)
    
    # Create single stacked bar
    bottom = 0
    barwidth = 0.4
    
    for outcome, percentage in zip(ordered_outcomes, percentages):
        color = outcome_colors.get(outcome, outcome_colors.get('Other', '#7f8c8d'))
        ax.bar(0, percentage, bottom=bottom, label=outcome, width=barwidth, color=color)
        bottom += percentage
    
    # Set x-axis
    ax.set_xticks([0])
    ax.set_xticklabels(['Pooled Sessions'])
    
    # Set labels and title
    ax.set_xlabel('All Sessions')
    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title('Pooled Session Outcomes')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
    
    # Set y-axis to show percentages
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))
    
    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    return ax

# ----------------------------------------------------------------------------------
# Plotting grand average of the outcomes for all sessions
def plot_grand_average_session_outcomes(sessions_outcome, ax=None, outcome_colors=None):
    """
    Plot a single bar chart showing the grand average outcome distribution across all sessions.
    For each outcome, the percentage is calculated per session, then averaged across sessions.
    
    Parameters:
    -----------
    sessions_outcome : list of lists
        List containing outcome lists for each session
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.
    outcome_colors : dict, optional
        Dictionary mapping outcome types to colors
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes with the plot
    """
    if outcome_colors is None:
        outcome_colors = {
            'Reward': '#63f250',         # Green
            'RewardNaive': '#115e07',    # Darker green
            'Punish': '#e74c3c',         # Red
            'PunishNaive': '#6b201f',    # Darker red
            'WrongInitiation': '#f39c12', # Orange
            'ChangingMindReward': '#9b59b6', # Purple
            'EarlyChoice': '#3498db',    # Blue
            'DidNotChoose': '#95a5a6',   # Gray
            'Other': '#7f8c8d'           # Darker gray
        }
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Remove top, right, and left spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Define specific order for outcomes (from bottom to top)
    outcome_order = [
        'Reward',
        'RewardNaive',
        'Punish',
        'PunishNaive',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]
    
    # Get all unique outcomes across sessions
    all_outcomes = set()
    for session_outcomes in sessions_outcome:
        all_outcomes.update(session_outcomes)
    
    # Filter outcome_order to only include outcomes present in the data
    ordered_outcomes = [outcome for outcome in outcome_order if outcome in all_outcomes]
    # Add any outcomes that aren't in our predefined order to the end
    for outcome in sorted(all_outcomes):
        if outcome not in ordered_outcomes:
            ordered_outcomes.append(outcome)
    
    # Calculate percentages for each outcome in each session
    session_percentages = {outcome: [] for outcome in ordered_outcomes}
    for session_outcomes in sessions_outcome:
        total_trials = len(session_outcomes)
        for outcome in ordered_outcomes:
            outcome_count = session_outcomes.count(outcome)
            percentage = (outcome_count / total_trials * 100) if total_trials > 0 else 0
            session_percentages[outcome].append(percentage)
    
    # Calculate grand average percentages
    avg_percentages = []
    for outcome in ordered_outcomes:
        avg_percentage = np.mean(session_percentages[outcome]) if session_percentages[outcome] else 0
        avg_percentages.append(avg_percentage)
    
    # Create single stacked bar
    bottom = 0
    barwidth = 0.4
    
    for outcome, avg_percentage in zip(ordered_outcomes, avg_percentages):
        color = outcome_colors.get(outcome, outcome_colors.get('Other', '#7f8c8d'))
        ax.bar(0, avg_percentage, bottom=bottom, label=outcome, width=barwidth, color=color)
        bottom += avg_percentage
    
    # Set x-axis
    ax.set_xticks([0])
    ax.set_xticklabels(['Grand Average'])
    
    # Set labels and title
    ax.set_xlabel('All Sessions')
    ax.set_ylabel('Average Percentage of Trials (%)')
    ax.set_title('Grand Average Session Outcomes')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
    
    # Set y-axis to show percentages
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))
    
    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    return ax

# ----------------------------------------------------------------------------------
# Plotting function for trial outcomes by trial type (short vs long ISI)
def plot_outcomes_by_trial_type(sessions_outcome, sessions_date, sessions_trial_type, ax=None, outcome_colors=None):

    if outcome_colors is None:
        outcome_colors = {
            'Reward': '#63f250',         # Green
            'Punish': '#e74c3c',         # Red
            'WrongInitiation': '#f39c12', # Orange
            'ChangingMindReward': '#9b59b6', # Purple
            'EarlyChoice': '#3498db',    # Blue
            'DidNotChoose': '#95a5a6',   # Gray
            'Other': '#7f8c8d'           # Darker gray
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    formatted_dates = []
    for date in sessions_date:
        if isinstance(date, str):
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y%m%d']:
                try:
                    date_obj = datetime.strptime(date, fmt)
                    formatted_dates.append(date_obj)
                    break
                except ValueError:
                    continue
            else:
                formatted_dates.append(date)
        else:
            formatted_dates.append(date)

    outcome_order = [
        'Reward',
        'Punish',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]

    # Identify what outcomes are present
    filtered_outcomes = set()
    for session_outcomes in sessions_outcome:
        for outcome in session_outcomes:
            if outcome not in ['RewardNaive', 'PunishNaive']:
                filtered_outcomes.add(outcome)

    ordered_outcomes = [o for o in outcome_order if o in filtered_outcomes]
    for o in sorted(filtered_outcomes):
        if o not in ordered_outcomes:
            ordered_outcomes.append(o)

    bar_width = 0.2
    session_positions = np.arange(len(sessions_date)) * 2
    short_isi_positions = session_positions - bar_width * 0.5
    long_isi_positions = session_positions + bar_width * 0.5

    data = []
    for outcomes, trial_types in zip(sessions_outcome, sessions_trial_type):
        short_isi_outcomes = [o for i, o in enumerate(outcomes) if i < len(trial_types) and trial_types[i] == 1 and o not in ['RewardNaive', 'PunishNaive']]
        long_isi_outcomes = [o for i, o in enumerate(outcomes) if i < len(trial_types) and trial_types[i] == 2 and o not in ['RewardNaive', 'PunishNaive']]
        data.append({
            'short': short_isi_outcomes,
            'long': long_isi_outcomes,
            'total_short': len(short_isi_outcomes),
            'total_long': len(long_isi_outcomes)
        })

    # Initialize bottoms BEFORE plotting
    short_bottoms = np.zeros(len(sessions_date))
    long_bottoms = np.zeros(len(sessions_date))

    for outcome in ordered_outcomes:
        short_percentages = []
        long_percentages = []

        for session_data in data:
            short_count = session_data['short'].count(outcome)
            short_pct = (short_count / session_data['total_short'] * 100) if session_data['total_short'] > 0 else 0
            short_percentages.append(short_pct)

            long_count = session_data['long'].count(outcome)
            long_pct = (long_count / session_data['total_long'] * 100) if session_data['total_long'] > 0 else 0
            long_percentages.append(long_pct)

        color = outcome_colors.get(outcome, outcome_colors['Other'])

        ax.bar(short_isi_positions, short_percentages, width=bar_width,
               bottom=short_bottoms, color=color, label=outcome)
        ax.bar(long_isi_positions, long_percentages, width=bar_width,
               bottom=long_bottoms, color=color)

        # Update bottoms for stacking next outcome
        short_bottoms += np.array(short_percentages)
        long_bottoms += np.array(long_percentages)

    # Set main x-ticks (bottom)
    if all(isinstance(date, datetime) for date in formatted_dates):
        date_labels = [date.strftime('%Y-%m-%d') for date in formatted_dates]
    else:
        date_labels = [str(date) for date in formatted_dates]

    ax.set_xticks(session_positions)
    ax.set_xticklabels(date_labels, rotation=45, ha='right')

    # Add subtle vertical lines
    for pos in session_positions:
        ax.axvline(pos, color='white', linewidth=5)

    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Add top x-axis for S|L
    ax_sec = ax.secondary_xaxis('top')
    ax_sec.set_xticks(session_positions)
    ax_sec.set_xticklabels(['S|L' for _ in session_positions], fontsize=10)

    # Labels and title
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    ax.set_xlabel('Session Date')
    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title('Session Outcomes by Trial Type (Short vs Long ISI)')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    return ax

# ----------------------------------------------------------------------------------
#  plotting function for pooled sessions outcome trial based on trial type (short vs long ISI)

def plot_pooled_outcomes_by_trial_type(sessions_outcome, sessions_trial_type, ax=None, outcome_colors=None):
    if outcome_colors is None:
        outcome_colors = {
            'Reward': '#63f250',         # Green
            'Punish': '#e74c3c',         # Red
            'WrongInitiation': '#f39c12', # Orange
            'ChangingMindReward': '#9b59b6', # Purple
            'EarlyChoice': '#3498db',    # Blue
            'DidNotChoose': '#95a5a6',   # Gray
            'Other': '#7f8c8d'           # Darker gray
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    outcome_order = [
        'Reward',
        'Punish',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]

    # Identify what outcomes are present
    filtered_outcomes = set()
    for session_outcomes in sessions_outcome:
        for outcome in session_outcomes:
            if outcome not in ['RewardNaive', 'PunishNaive']:
                filtered_outcomes.add(outcome)

    ordered_outcomes = [o for o in outcome_order if o in filtered_outcomes]
    for o in sorted(filtered_outcomes):
        if o not in ordered_outcomes:
            ordered_outcomes.append(o)

    short_isi_outcomes = []
    long_isi_outcomes = []
    for outcomes, trial_types in zip(sessions_outcome, sessions_trial_type):
        short_isi_outcomes.extend([o for i, o in enumerate(outcomes) if i < len(trial_types) and trial_types[i] == 1 and o not in ['RewardNaive', 'PunishNaive']])
        long_isi_outcomes.extend([o for i, o in enumerate(outcomes) if i < len(trial_types) and trial_types[i] == 2 and o not in ['RewardNaive', 'PunishNaive']])

    total_short = len(short_isi_outcomes)
    total_long = len(long_isi_outcomes)

    bar_width = 0.35
    positions = [0, 1]
    short_pos = positions[0]
    long_pos = positions[1]

    # Initialize bottoms for stacking
    short_bottom = 0
    long_bottom = 0

    # Plot stacked bars
    for outcome in ordered_outcomes:
        short_count = short_isi_outcomes.count(outcome)
        short_pct = (short_count / total_short * 100) if total_short > 0 else 0

        long_count = long_isi_outcomes.count(outcome)
        long_pct = (long_count / total_long * 100) if total_long > 0 else 0

        color = outcome_colors.get(outcome, outcome_colors['Other'])

        ax.bar(short_pos, short_pct, width=bar_width, bottom=short_bottom, color=color, label=outcome)
        ax.bar(long_pos, long_pct, width=bar_width, bottom=long_bottom, color=color)

        short_bottom += short_pct
        long_bottom += long_pct

    # Customize axes
    ax.set_xticks(positions)
    ax.set_xticklabels(['Short ISI', 'Long ISI'])

    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title('Pooled Session Outcomes by Trial Type (Short vs Long ISI)\n')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


    return ax

# ----------------------------------------------------------------------------------
# plotting function for Grand average of the outcomes for all sessions by trial type (short vs long ISI)
def plot_average_outcomes_by_trial_type(sessions_outcome, sessions_trial_type, ax=None, outcome_colors=None):
    if outcome_colors is None:
        outcome_colors = {
            'Reward': '#63f250',         # Green
            'Punish': '#e74c3c',         # Red
            'WrongInitiation': '#f39c12', # Orange
            'ChangingMindReward': '#9b59b6', # Purple
            'EarlyChoice': '#3498db',    # Blue
            'DidNotChoose': '#95a5a6',   # Gray
            'Other': '#7f8c8d'           # Darker gray
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    outcome_order = [
        'Reward',
        'Punish',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]

    # Collect outcome percentages for each session
    all_short_percentages = []
    all_long_percentages = []
    filtered_outcomes = set()

    for outcomes, trial_types in zip(sessions_outcome, sessions_trial_type):
        short_isi_outcomes = [o for i, o in enumerate(outcomes) if i < len(trial_types) and trial_types[i] == 1 and o not in ['RewardNaive', 'PunishNaive']]
        long_isi_outcomes = [o for i, o in enumerate(outcomes) if i < len(trial_types) and trial_types[i] == 2 and o not in ['RewardNaive', 'PunishNaive']]
        
        total_short = len(short_isi_outcomes)
        total_long = len(long_isi_outcomes)

        short_pct = {}
        long_pct = {}

        for outcome in set(short_isi_outcomes + long_isi_outcomes):
            filtered_outcomes.add(outcome)
            short_count = short_isi_outcomes.count(outcome)
            short_pct[outcome] = (short_count / total_short * 100) if total_short > 0 else 0
            long_count = long_isi_outcomes.count(outcome)
            long_pct[outcome] = (long_count / total_long * 100) if total_long > 0 else 0

        all_short_percentages.append(short_pct)
        all_long_percentages.append(long_pct)

    # Determine ordered outcomes
    ordered_outcomes = [o for o in outcome_order if o in filtered_outcomes]
    for o in sorted(filtered_outcomes):
        if o not in ordered_outcomes:
            ordered_outcomes.append(o)

    # Calculate average percentages across sessions
    avg_short_percentages = {}
    avg_long_percentages = {}

    for outcome in ordered_outcomes:
        short_vals = [pct.get(outcome, 0) for pct in all_short_percentages]
        long_vals = [pct.get(outcome, 0) for pct in all_long_percentages]
        avg_short_percentages[outcome] = np.mean(short_vals) if short_vals else 0
        avg_long_percentages[outcome] = np.mean(long_vals) if long_vals else 0

    # Plotting
    bar_width = 0.35
    positions = [0, 1]
    short_pos = positions[0]
    long_pos = positions[1]

    short_bottom = 0
    long_bottom = 0

    for outcome in ordered_outcomes:
        short_pct = avg_short_percentages[outcome]
        long_pct = avg_long_percentages[outcome]
        color = outcome_colors.get(outcome, outcome_colors['Other'])

        ax.bar(short_pos, short_pct, width=bar_width, bottom=short_bottom, color=color, label=outcome)
        ax.bar(long_pos, long_pct, width=bar_width, bottom=long_bottom, color=color)

        short_bottom += short_pct
        long_bottom += long_pct

    # Customize axes
    ax.set_xticks(positions)
    ax.set_xticklabels(['Short ISI', 'Long ISI'])

    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    ax.set_ylabel('Average Percentage of Trials (%)')
    ax.set_title('Average Session Outcomes by Trial Type (Short vs Long ISI) \n')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    return ax

# ---------------------------------------------------------------------------------------
# Opto vs control trials among short ISI 
def plot_outcomes_by_opto_tag_short(sessions_outcome, sessions_date, sessions_trial_type, sessions_opto_tag, ax=None, outcome_colors=None):

    if outcome_colors is None:
        outcome_colors = {
            'Reward': '#63f250',         # Green
            'Punish': '#e74c3c',         # Red
            'WrongInitiation': '#f39c12', # Orange
            'ChangingMindReward': '#9b59b6', # Purple
            'EarlyChoice': '#3498db',    # Blue
            'DidNotChoose': '#95a5a6',   # Gray
            'Other': '#7f8c8d'           # Darker gray
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    formatted_dates = []
    for date in sessions_date:
        if isinstance(date, str):
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y%m%d']:
                try:
                    date_obj = datetime.strptime(date, fmt)
                    formatted_dates.append(date_obj)
                    break
                except ValueError:
                    continue
            else:
                formatted_dates.append(date)
        else:
            formatted_dates.append(date)

    outcome_order = [
        'Reward',
        'Punish',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]

    # Identify what outcomes are present
    filtered_outcomes = set()
    for session_outcomes in sessions_outcome:
        for outcome in session_outcomes:
            if outcome not in ['RewardNaive', 'PunishNaive']:
                filtered_outcomes.add(outcome)

    ordered_outcomes = [o for o in outcome_order if o in filtered_outcomes]
    for o in sorted(filtered_outcomes):
        if o not in ordered_outcomes:
            ordered_outcomes.append(o)

    bar_width = 0.2
    session_positions = np.arange(len(sessions_date)) * 2
    control_positions = session_positions - bar_width * 0.5
    opto_positions = session_positions + bar_width * 0.5

    data = []
    for outcomes, trial_types, opto_tags in zip(sessions_outcome, sessions_trial_type, sessions_opto_tag):
        control_outcomes = [o for i, o in enumerate(outcomes) 
                          if i < len(trial_types) and i < len(opto_tags) 
                          and trial_types[i] == 1 and opto_tags[i] == 0 
                          and o not in ['RewardNaive', 'PunishNaive']]
        opto_outcomes = [o for i, o in enumerate(outcomes) 
                        if i < len(trial_types) and i < len(opto_tags) 
                        and trial_types[i] == 1 and opto_tags[i] == 1 
                        and o not in ['RewardNaive', 'PunishNaive']]
        data.append({
            'control': control_outcomes,
            'opto': opto_outcomes,
            'total_control': len(control_outcomes),
            'total_opto': len(opto_outcomes)
        })

    # Initialize bottoms BEFORE plotting
    control_bottoms = np.zeros(len(sessions_date))
    opto_bottoms = np.zeros(len(sessions_date))

    for outcome in ordered_outcomes:
        control_percentages = []
        opto_percentages = []

        for session_data in data:
            control_count = session_data['control'].count(outcome)
            control_pct = (control_count / session_data['total_control'] * 100) if session_data['total_control'] > 0 else 0
            control_percentages.append(control_pct)

            opto_count = session_data['opto'].count(outcome)
            opto_pct = (opto_count / session_data['total_opto'] * 100) if session_data['total_opto'] > 0 else 0
            opto_percentages.append(opto_pct)

        color = outcome_colors.get(outcome, outcome_colors['Other'])

        ax.bar(control_positions, control_percentages, width=bar_width,
               bottom=control_bottoms, color=color, label=outcome)
        ax.bar(opto_positions, opto_percentages, width=bar_width,
               bottom=opto_bottoms, color=color)

        # Update bottoms for stacking next outcome
        control_bottoms += np.array(control_percentages)
        opto_bottoms += np.array(opto_percentages)

    # Set main x-ticks (bottom)
    if all(isinstance(date, datetime) for date in formatted_dates):
        date_labels = [date.strftime('%Y-%m-%d') for date in formatted_dates]
    else:
        date_labels = [str(date) for date in formatted_dates]

    ax.set_xticks(session_positions)
    ax.set_xticklabels(date_labels, rotation=45, ha='right')

    # Add subtle vertical lines
    for pos in session_positions:
        ax.axvline(pos, color='white', linewidth=5)

    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Add top x-axis for C|O
    ax_sec = ax.secondary_xaxis('top')
    ax_sec.set_xticks(session_positions)
    ax_sec.set_xticklabels(['C|O' for _ in session_positions], fontsize=10)

    # Labels and title
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    ax.set_xlabel('Session Date')
    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title('Session Outcomes for Short ISI Trials (Control vs Opto)')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))
    

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

# ----------------------------------------------------------------------------------
# Plotting function for pooled sessions short ISI outcome trial based on opto tag (control vs opto)
def plot_pooled_outcomes_by_opto_tag_short(sessions_outcome, sessions_trial_type, sessions_opto_tag, ax=None, outcome_colors=None):

    if outcome_colors is None:
        outcome_colors = {
            'Reward': '#63f250',         # Green
            'Punish': '#e74c3c',         # Red
            'WrongInitiation': '#f39c12', # Orange
            'ChangingMindReward': '#9b59b6', # Purple
            'EarlyChoice': '#3498db',    # Blue
            'DidNotChoose': '#95a5a6',   # Gray
            'Other': '#7f8c8d'           # Darker gray
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    outcome_order = [
        'Reward',
        'Punish',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]

    # Identify what outcomes are present
    filtered_outcomes = set()
    for session_outcomes in sessions_outcome:
        for outcome in session_outcomes:
            if outcome not in ['RewardNaive', 'PunishNaive']:
                filtered_outcomes.add(outcome)

    ordered_outcomes = [o for o in outcome_order if o in filtered_outcomes]
    for o in sorted(filtered_outcomes):
        if o not in ordered_outcomes:
            ordered_outcomes.append(o)

    # Pool all sessions for short ISI trials
    control_outcomes = []
    opto_outcomes = []
    for outcomes, trial_types, opto_tags in zip(sessions_outcome, sessions_trial_type, sessions_opto_tag):
        control_outcomes.extend([o for i, o in enumerate(outcomes) 
                               if i < len(trial_types) and i < len(opto_tags) 
                               and trial_types[i] == 1 and opto_tags[i] == 0 
                               and o not in ['RewardNaive', 'PunishNaive']])
        opto_outcomes.extend([o for i, o in enumerate(outcomes) 
                             if i < len(trial_types) and i < len(opto_tags) 
                             and trial_types[i] == 1 and opto_tags[i] == 1 
                             and o not in ['RewardNaive', 'PunishNaive']])

    total_control = len(control_outcomes)
    total_opto = len(opto_outcomes)

    bar_width = 0.35
    positions = [0, 1]
    control_pos = positions[0]
    opto_pos = positions[1]

    # Initialize bottoms
    control_bottom = 0
    opto_bottom = 0

    for outcome in ordered_outcomes:
        control_count = control_outcomes.count(outcome)
        control_pct = (control_count / total_control * 100) if total_control > 0 else 0

        opto_count = opto_outcomes.count(outcome)
        opto_pct = (opto_count / total_opto * 100) if total_opto > 0 else 0

        color = outcome_colors.get(outcome, outcome_colors['Other'])

        # Assign label to both control and opto bars to ensure legend includes all outcomes
        ax.bar(control_pos, control_pct, width=bar_width, bottom=control_bottom, 
            color=color, label=outcome)
        ax.bar(opto_pos, opto_pct, width=bar_width, bottom=opto_bottom, 
            color=color, label=outcome)

        # Update bottoms for stacking next outcome
        control_bottom += control_pct
        opto_bottom += opto_pct

    # Set x-ticks
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'Opto'])

    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Labels and title
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title('Pooled Session Outcomes for Short ISI Trials (Control vs Opto) \n')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    return ax

# ----------------------------------------------------------------------------------
# plotting function for Grand average of the short ISI outcomes for all sessions by opto tag (control vs opto)
def plot_grand_average_outcomes_by_opto_tag_short(sessions_outcome, sessions_trial_type, sessions_opto_tag, ax=None, outcome_colors=None):

    if outcome_colors is None:
        outcome_colors = {
            'Reward': '#63f250',         # Green
            'Punish': '#e74c3c',         # Red
            'WrongInitiation': '#f39c12', # Orange
            'ChangingMindReward': '#9b59b6', # Purple
            'EarlyChoice': '#3498db',    # Blue
            'DidNotChoose': '#95a5a6',   # Gray
            'Other': '#7f8c8d'           # Darker gray
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    outcome_order = [
        'Reward',
        'Punish',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]

    # Identify what outcomes are present
    filtered_outcomes = set()
    for session_outcomes in sessions_outcome:
        for outcome in session_outcomes:
            if outcome not in ['RewardNaive', 'PunishNaive']:
                filtered_outcomes.add(outcome)

    ordered_outcomes = [o for o in outcome_order if o in filtered_outcomes]
    for o in sorted(filtered_outcomes):
        if o not in ordered_outcomes:
            ordered_outcomes.append(o)

    # Collect percentages for each session
    control_percentages_by_outcome = {outcome: [] for outcome in ordered_outcomes}
    opto_percentages_by_outcome = {outcome: [] for outcome in ordered_outcomes}

    for outcomes, trial_types, opto_tags in zip(sessions_outcome, sessions_trial_type, sessions_opto_tag):
        control_outcomes = [o for i, o in enumerate(outcomes) 
                          if i < len(trial_types) and i < len(opto_tags) 
                          and trial_types[i] == 1 and opto_tags[i] == 0 
                          and o not in ['RewardNaive', 'PunishNaive']]
        opto_outcomes = [o for i, o in enumerate(outcomes) 
                        if i < len(trial_types) and i < len(opto_tags) 
                        and trial_types[i] == 1 and opto_tags[i] == 1 
                        and o not in ['RewardNaive', 'PunishNaive']]
        
        total_control = len(control_outcomes)
        total_opto = len(opto_outcomes)

        for outcome in ordered_outcomes:
            control_count = control_outcomes.count(outcome)
            control_pct = (control_count / total_control * 100) if total_control > 0 else 0
            control_percentages_by_outcome[outcome].append(control_pct)

            opto_count = opto_outcomes.count(outcome)
            opto_pct = (opto_count / total_opto * 100) if total_opto > 0 else 0
            opto_percentages_by_outcome[outcome].append(opto_pct)

    # Compute grand average percentages
    control_mean_percentages = []
    opto_mean_percentages = []
    for outcome in ordered_outcomes:
        control_mean = np.mean(control_percentages_by_outcome[outcome]) if control_percentages_by_outcome[outcome] else 0
        opto_mean = np.mean(opto_percentages_by_outcome[outcome]) if opto_percentages_by_outcome[outcome] else 0
        control_mean_percentages.append(control_mean)
        opto_mean_percentages.append(opto_mean)

    # Plotting
    bar_width = 0.35
    positions = [0, 1]
    control_pos = positions[0]
    opto_pos = positions[1]

    control_bottom = 0
    opto_bottom = 0

    for outcome, control_pct, opto_pct in zip(ordered_outcomes, control_mean_percentages, opto_mean_percentages):
        color = outcome_colors.get(outcome, outcome_colors['Other'])

        ax.bar(control_pos, control_pct, width=bar_width, bottom=control_bottom, 
               color=color, label=outcome)
        ax.bar(opto_pos, opto_pct, width=bar_width, bottom=opto_bottom, 
               color=color, label=outcome)

        control_bottom += control_pct
        opto_bottom += opto_pct

    # Set x-ticks
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'Opto'])

    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Labels and title
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    ax.set_ylabel('Mean Percentage of Trials (%)')
    ax.set_title('Grand Average Session Outcomes for Short ISI Trials (Control vs Opto)\n')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    return ax

# ----------------------------------------------------------------------------------
# Opto vs control trials among Long ISI 
def plot_outcomes_by_opto_tag_long(sessions_outcome, sessions_date, sessions_trial_type, sessions_opto_tag, ax=None, outcome_colors=None):

    if outcome_colors is None:
        outcome_colors = {
            'Reward': '#63f250',         # Green
            'Punish': '#e74c3c',         # Red
            'WrongInitiation': '#f39c12', # Orange
            'ChangingMindReward': '#9b59b6', # Purple
            'EarlyChoice': '#3498db',    # Blue
            'DidNotChoose': '#95a5a6',   # Gray
            'Other': '#7f8c8d'           # Darker gray
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    formatted_dates = []
    for date in sessions_date:
        if isinstance(date, str):
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y%m%d']:
                try:
                    date_obj = datetime.strptime(date, fmt)
                    formatted_dates.append(date_obj)
                    break
                except ValueError:
                    continue
            else:
                formatted_dates.append(date)
        else:
            formatted_dates.append(date)

    outcome_order = [
        'Reward',
        'Punish',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]

    # Identify what outcomes are present
    filtered_outcomes = set()
    for session_outcomes in sessions_outcome:
        for outcome in session_outcomes:
            if outcome not in ['RewardNaive', 'PunishNaive']:
                filtered_outcomes.add(outcome)

    ordered_outcomes = [o for o in outcome_order if o in filtered_outcomes]
    for o in sorted(filtered_outcomes):
        if o not in ordered_outcomes:
            ordered_outcomes.append(o)

    bar_width = 0.2
    session_positions = np.arange(len(sessions_date)) * 2
    control_positions = session_positions - bar_width * 0.5
    opto_positions = session_positions + bar_width * 0.5

    data = []
    for outcomes, trial_types, opto_tags in zip(sessions_outcome, sessions_trial_type, sessions_opto_tag):
        control_outcomes = [o for i, o in enumerate(outcomes) 
                          if i < len(trial_types) and i < len(opto_tags) 
                          and trial_types[i] == 2 and opto_tags[i] == 0 
                          and o not in ['RewardNaive', 'PunishNaive']]
        opto_outcomes = [o for i, o in enumerate(outcomes) 
                        if i < len(trial_types) and i < len(opto_tags) 
                        and trial_types[i] == 2 and opto_tags[i] == 1 
                        and o not in ['RewardNaive', 'PunishNaive']]
        data.append({
            'control': control_outcomes,
            'opto': opto_outcomes,
            'total_control': len(control_outcomes),
            'total_opto': len(opto_outcomes)
        })

    # Initialize bottoms BEFORE plotting
    control_bottoms = np.zeros(len(sessions_date))
    opto_bottoms = np.zeros(len(sessions_date))

    for outcome in ordered_outcomes:
        control_percentages = []
        opto_percentages = []

        for session_data in data:
            control_count = session_data['control'].count(outcome)
            control_pct = (control_count / session_data['total_control'] * 100) if session_data['total_control'] > 0 else 0
            control_percentages.append(control_pct)

            opto_count = session_data['opto'].count(outcome)
            opto_pct = (opto_count / session_data['total_opto'] * 100) if session_data['total_opto'] > 0 else 0
            opto_percentages.append(opto_pct)

        color = outcome_colors.get(outcome, outcome_colors['Other'])

        ax.bar(control_positions, control_percentages, width=bar_width,
               bottom=control_bottoms, color=color, label=outcome)
        ax.bar(opto_positions, opto_percentages, width=bar_width,
               bottom=opto_bottoms, color=color)

        # Update bottoms for stacking next outcome
        control_bottoms += np.array(control_percentages)
        opto_bottoms += np.array(opto_percentages)

    # Set main x-ticks (bottom)
    if all(isinstance(date, datetime) for date in formatted_dates):
        date_labels = [date.strftime('%Y-%m-%d') for date in formatted_dates]
    else:
        date_labels = [str(date) for date in formatted_dates]

    ax.set_xticks(session_positions)
    ax.set_xticklabels(date_labels, rotation=45, ha='right')

    # Add subtle vertical lines
    for pos in session_positions:
        ax.axvline(pos, color='white', linewidth=5)

    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Add top x-axis for C|O
    ax_sec = ax.secondary_xaxis('top')
    ax_sec.set_xticks(session_positions)
    ax_sec.set_xticklabels(['C|O' for _ in session_positions], fontsize=10)

    # Labels and title
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    ax.set_xlabel('Session Date')
    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title('Session Outcomes for Long ISI Trials (Control vs Opto)')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.tight_layout()

# ----------------------------------------------------------------------------------
# Plotting function for pooled sessions long ISI outcome trial based on opto tag (control vs opto)
def plot_pooled_outcomes_by_opto_tag_long(sessions_outcome, sessions_trial_type, sessions_opto_tag, ax=None, outcome_colors=None):

    if outcome_colors is None:
        outcome_colors = {
            'Reward': '#63f250',         # Green
            'Punish': '#e74c3c',         # Red
            'WrongInitiation': '#f39c12', # Orange
            'ChangingMindReward': '#9b59b6', # Purple
            'EarlyChoice': '#3498db',    # Blue
            'DidNotChoose': '#95a5a6',   # Gray
            'Other': '#7f8c8d'           # Darker gray
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    outcome_order = [
        'Reward',
        'Punish',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]

    # Identify what outcomes are present
    filtered_outcomes = set()
    for session_outcomes in sessions_outcome:
        for outcome in session_outcomes:
            if outcome not in ['RewardNaive', 'PunishNaive']:
                filtered_outcomes.add(outcome)

    ordered_outcomes = [o for o in outcome_order if o in filtered_outcomes]
    for o in sorted(filtered_outcomes):
        if o not in ordered_outcomes:
            ordered_outcomes.append(o)

    # Pool all sessions for short ISI trials
    control_outcomes = []
    opto_outcomes = []
    for outcomes, trial_types, opto_tags in zip(sessions_outcome, sessions_trial_type, sessions_opto_tag):
        control_outcomes.extend([o for i, o in enumerate(outcomes) 
                               if i < len(trial_types) and i < len(opto_tags) 
                               and trial_types[i] == 2 and opto_tags[i] == 0 
                               and o not in ['RewardNaive', 'PunishNaive']])
        opto_outcomes.extend([o for i, o in enumerate(outcomes) 
                             if i < len(trial_types) and i < len(opto_tags) 
                             and trial_types[i] == 2 and opto_tags[i] == 1 
                             and o not in ['RewardNaive', 'PunishNaive']])

    total_control = len(control_outcomes)
    total_opto = len(opto_outcomes)

    bar_width = 0.35
    positions = [0, 1]
    control_pos = positions[0]
    opto_pos = positions[1]

    # Initialize bottoms
    control_bottom = 0
    opto_bottom = 0

    for outcome in ordered_outcomes:
        control_count = control_outcomes.count(outcome)
        control_pct = (control_count / total_control * 100) if total_control > 0 else 0

        opto_count = opto_outcomes.count(outcome)
        opto_pct = (opto_count / total_opto * 100) if total_opto > 0 else 0

        color = outcome_colors.get(outcome, outcome_colors['Other'])

        # Assign label to both control and opto bars to ensure legend includes all outcomes
        ax.bar(control_pos, control_pct, width=bar_width, bottom=control_bottom, 
            color=color, label=outcome)
        ax.bar(opto_pos, opto_pct, width=bar_width, bottom=opto_bottom, 
            color=color, label=outcome)

        # Update bottoms for stacking next outcome
        control_bottom += control_pct
        opto_bottom += opto_pct

    # Set x-ticks
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'Opto'])

    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Labels and title
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title('Pooled Session Outcomes for Long ISI Trials (Control vs Opto)\n')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    return ax

# ----------------------------------------------------------------------------------
# plotting function for Grand average of the long ISI outcomes for all sessions by opto tag (control vs opto)
def plot_grand_average_outcomes_by_opto_tag_long(sessions_outcome, sessions_trial_type, sessions_opto_tag, ax=None, outcome_colors=None):

    if outcome_colors is None:
        outcome_colors = {
            'Reward': '#63f250',         # Green
            'Punish': '#e74c3c',         # Red
            'WrongInitiation': '#f39c12', # Orange
            'ChangingMindReward': '#9b59b6', # Purple
            'EarlyChoice': '#3498db',    # Blue
            'DidNotChoose': '#95a5a6',   # Gray
            'Other': '#7f8c8d'           # Darker gray
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    outcome_order = [
        'Reward',
        'Punish',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]

    # Identify what outcomes are present
    filtered_outcomes = set()
    for session_outcomes in sessions_outcome:
        for outcome in session_outcomes:
            if outcome not in ['RewardNaive', 'PunishNaive']:
                filtered_outcomes.add(outcome)

    ordered_outcomes = [o for o in outcome_order if o in filtered_outcomes]
    for o in sorted(filtered_outcomes):
        if o not in ordered_outcomes:
            ordered_outcomes.append(o)

    # Collect percentages for each session
    control_percentages_by_outcome = {outcome: [] for outcome in ordered_outcomes}
    opto_percentages_by_outcome = {outcome: [] for outcome in ordered_outcomes}

    for outcomes, trial_types, opto_tags in zip(sessions_outcome, sessions_trial_type, sessions_opto_tag):
        control_outcomes = [o for i, o in enumerate(outcomes) 
                          if i < len(trial_types) and i < len(opto_tags) 
                          and trial_types[i] == 2 and opto_tags[i] == 0 
                          and o not in ['RewardNaive', 'PunishNaive']]
        opto_outcomes = [o for i, o in enumerate(outcomes) 
                        if i < len(trial_types) and i < len(opto_tags) 
                        and trial_types[i] == 2 and opto_tags[i] == 1 
                        and o not in ['RewardNaive', 'PunishNaive']]
        
        total_control = len(control_outcomes)
        total_opto = len(opto_outcomes)

        for outcome in ordered_outcomes:
            control_count = control_outcomes.count(outcome)
            control_pct = (control_count / total_control * 100) if total_control > 0 else 0
            control_percentages_by_outcome[outcome].append(control_pct)

            opto_count = opto_outcomes.count(outcome)
            opto_pct = (opto_count / total_opto * 100) if total_opto > 0 else 0
            opto_percentages_by_outcome[outcome].append(opto_pct)

    # Compute grand average percentages
    control_mean_percentages = []
    opto_mean_percentages = []
    for outcome in ordered_outcomes:
        control_mean = np.mean(control_percentages_by_outcome[outcome]) if control_percentages_by_outcome[outcome] else 0
        opto_mean = np.mean(opto_percentages_by_outcome[outcome]) if opto_percentages_by_outcome[outcome] else 0
        control_mean_percentages.append(control_mean)
        opto_mean_percentages.append(opto_mean)

    # Plotting
    bar_width = 0.35
    positions = [0, 1]
    control_pos = positions[0]
    opto_pos = positions[1]

    control_bottom = 0
    opto_bottom = 0

    for outcome, control_pct, opto_pct in zip(ordered_outcomes, control_mean_percentages, opto_mean_percentages):
        color = outcome_colors.get(outcome, outcome_colors['Other'])

        ax.bar(control_pos, control_pct, width=bar_width, bottom=control_bottom, 
               color=color, label=outcome)
        ax.bar(opto_pos, opto_pct, width=bar_width, bottom=opto_bottom, 
               color=color, label=outcome)

        control_bottom += control_pct
        opto_bottom += opto_pct

    # Set x-ticks
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'Opto'])

    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Labels and title
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    ax.set_ylabel('Mean Percentage of Trials (%)')
    ax.set_title('Grand Average Session Outcomes for Long ISI Trials (Control vs Opto)\n')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    return ax


# ----------------------------------------------------------------------------------
# Function for plotting decssions for each session
def plot_decision_sidedness(sessions_outcome, sessions_date, sessions_trial_type, ax=None, right_color='#63f250', left_color='#e6c910'):
    """
    Plots proportion of "Right Decision" vs "Left Decision" per session.
    Right Decision = Reward in Long ISI + Punish in Short ISI
    Left Decision = 100% - Right Decision

    Parameters:
        sessions_outcome (list of list): Outcomes per session.
        sessions_date (list): Session dates (str or datetime).
        sessions_trial_type (list of list): Trial types per session (1 = Short ISI, 2 = Long ISI).
        ax (matplotlib.axes.Axes, optional): Axes object to plot on.
        right_color (str): Color for Right Decision.
        left_color (str): Color for Left Decision.

    Returns:
        ax (matplotlib.axes.Axes): The plotted axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Handle date formatting
    formatted_dates = []
    for date in sessions_date:
        if isinstance(date, str):
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y%m%d']:
                try:
                    date_obj = datetime.strptime(date, fmt)
                    formatted_dates.append(date_obj)
                    break
                except ValueError:
                    continue
            else:
                formatted_dates.append(date)
        else:
            formatted_dates.append(date)

    # Compute Right vs Left percentages per session
    right_percentages = []
    left_percentages = []

    for outcomes, trial_types in zip(sessions_outcome, sessions_trial_type):
        total_trials = len([o for o in outcomes if o not in ['RewardNaive', 'PunishNaive']])
        if total_trials == 0:
            right_percentages.append(0)
            left_percentages.append(100)
            continue

        right_count = 0
        for i, (outcome, trial_type) in enumerate(zip(outcomes, trial_types)):
            if outcome in ['RewardNaive', 'PunishNaive']:
                continue
            if outcome == 'Reward' and trial_type == 2:  # Long ISI
                right_count += 1
            elif outcome == 'Punish' and trial_type == 1:  # Short ISI
                right_count += 1

        right_pct = right_count / total_trials * 100
        left_pct = 100 - right_pct
        right_percentages.append(right_pct)
        left_percentages.append(left_pct)

    # Plotting
    session_positions = np.arange(len(sessions_date)) 
    ax.bar(session_positions, right_percentages, width=0.2, color=right_color, label='Right Decision')
    ax.bar(session_positions, left_percentages, width=0.2, bottom=right_percentages, color=left_color, label='Left Decision')

    # X-axis labels
    if all(isinstance(date, datetime) for date in formatted_dates):
        date_labels = [date.strftime('%Y-%m-%d') for date in formatted_dates]
    else:
        date_labels = [str(date) for date in formatted_dates]

    ax.set_xticks(session_positions)
    ax.set_xticklabels(date_labels, rotation=45, ha='right')

    # Aesthetics
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Session Date')
    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title('Right vs Left Decision Proportion by Session')
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.tight_layout()
    return ax

# ----------------------------------------------------------------------------------
# Function for pooling all session and finding decsion
def plot_pooled_decision_sidedness(sessions_outcome, sessions_trial_type, ax=None, right_color='#63f250', left_color='#e6c910'):
    """
    Pools all sessions and plots a single bar showing proportion of Right vs Left Decisions.

    Right Decision = Reward in Long ISI + Punish in Short ISI  
    Left Decision = All other valid outcomes

    Parameters:
        sessions_outcome (list of list): Outcomes per session.
        sessions_trial_type (list of list): Trial types per session (1 = Short ISI, 2 = Long ISI).
        ax (matplotlib.axes.Axes, optional): Axes object to plot on.
        right_color (str): Color for Right Decision.
        left_color (str): Color for Left Decision.

    Returns:
        ax (matplotlib.axes.Axes): The plotted axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))

    total_valid = 0
    right_count = 0

    for outcomes, trial_types in zip(sessions_outcome, sessions_trial_type):
        for outcome, trial_type in zip(outcomes, trial_types):
            if outcome in ['RewardNaive', 'PunishNaive']:
                continue
            total_valid += 1
            if (outcome == 'Reward' and trial_type == 2) or (outcome == 'Punish' and trial_type == 1):
                right_count += 1

    if total_valid == 0:
        right_pct = 0
    else:
        right_pct = right_count / total_valid * 100

    left_pct = 100 - right_pct

    ax.bar(0, right_pct, width= 0.2, color=right_color, label='Right Decision')
    ax.bar(0, left_pct, width= 0.2, bottom=right_pct, color=left_color, label='Left Decision')

    ax.set_xticks([0])
    ax.set_xticklabels(['All Sessions'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title('Pooled Decision Sidedness\n')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best')

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.tight_layout()
    return ax

# ----------------------------------------------------------------------------------
# Funciton for plotting grand averafe of deciosns across sessions
def plot_average_decision_sidedness(sessions_outcome, sessions_trial_type, ax=None, right_color='#63f250', left_color='#e6c910'):
    """
    Computes average Right vs Left Decision percentages across sessions (grand average).

    Right Decision = Reward in Long ISI + Punish in Short ISI  
    Left Decision = All other valid outcomes

    Parameters:
        sessions_outcome (list of list): Outcomes per session.
        sessions_trial_type (list of list): Trial types per session (1 = Short ISI, 2 = Long ISI).
        ax (matplotlib.axes.Axes, optional): Axes object to plot on.
        right_color (str): Color for Right Decision.
        left_color (str): Color for Left Decision.

    Returns:
        ax (matplotlib.axes.Axes): The plotted axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))

    right_percentages = []

    for outcomes, trial_types in zip(sessions_outcome, sessions_trial_type):
        total = 0
        right = 0

        for outcome, trial_type in zip(outcomes, trial_types):
            if outcome in ['RewardNaive', 'PunishNaive']:
                continue
            total += 1
            if (outcome == 'Reward' and trial_type == 2) or (outcome == 'Punish' and trial_type == 1):
                right += 1

        if total > 0:
            right_pct = right / total * 100
            right_percentages.append(right_pct)

    if not right_percentages:
        avg_right = 0
    else:
        avg_right = np.mean(right_percentages)

    avg_left = 100 - avg_right

    ax.bar(0, avg_right, width=0.2, color=right_color, label='Right Decision (avg)')
    ax.bar(0, avg_left, width=0.2, bottom=avg_right, color=left_color, label='Left Decision (avg)')

    ax.set_xticks([0])
    ax.set_xticklabels(['Grand Avg'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Average Percentage (%)')
    ax.set_title('Grand Average Decision Sidedness\n')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best')

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    return ax

# ----------------------------------------------------------------------------------
# Plotting function for all sessions using GridSpec
def plot_all_sessions_with_gridspec(sessions_outcome, sessions_date, sessions_trial_type, sessions_opto_tag, outcome_colors=None, figsize=(40, 20)):
    """
    Create a single plot with all sessions displayed as bars next to each other
    using GridSpec for better layout control.
    
    Parameters:
    -----------
    sessions_outcome : list of lists
        List containing outcome lists for each session
    sessions_date : list
        List of session dates
    outcome_colors : dict, optional
        Dictionary mapping outcome types to colors
    figsize : tuple, optional
        Size of the figure (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(5, 3,  height_ratios=[1, 1, 1, 1, 1], width_ratios=[10, 1, 1])  # Just one plot area
    ax = fig.add_subplot(gs[0, 0])
    
    # Plot all sessions in this single axis
    plot_session_outcomes_combined(sessions_outcome, sessions_date, ax=ax, outcome_colors=outcome_colors)

    # Plot pooled session outcomes in the next subplot
    ax = fig.add_subplot(gs[0, 1])
    plot_pooled_session_outcomes(sessions_outcome, ax=ax, outcome_colors=None)

    # Plot grand average session outcomes in the next subplot
    ax = fig.add_subplot(gs[0, 2])
    plot_grand_average_session_outcomes(sessions_outcome, ax=ax, outcome_colors=None)

    # Plot outcomes by trial type in the next subplot
    ax = fig.add_subplot(gs[1, 0])
    plot_outcomes_by_trial_type(sessions_outcome, sessions_date, sessions_trial_type, ax=ax, outcome_colors=outcome_colors)

    # Plot pooled outcomes by trial type in the next subplot
    ax = fig.add_subplot(gs[1, 1])
    plot_pooled_outcomes_by_trial_type(sessions_outcome, sessions_trial_type, ax=ax, outcome_colors=None)

    # plot average outcomes by trial type in the next subplot
    ax = fig.add_subplot(gs[1, 2])
    plot_average_outcomes_by_trial_type(sessions_outcome, sessions_trial_type, ax=ax, outcome_colors=None)

    # plot outcomes by opto tag for short ISI in the next subplot
    ax = fig.add_subplot(gs[2, 0])
    plot_outcomes_by_opto_tag_short(sessions_outcome, sessions_date, sessions_trial_type, sessions_opto_tag, ax=ax, outcome_colors=None)

    # plot pooled outcomes by opto tag for short ISI in the next subplot
    ax = fig.add_subplot(gs[2, 1])
    plot_pooled_outcomes_by_opto_tag_short(sessions_outcome, sessions_trial_type, sessions_opto_tag, ax=ax, outcome_colors=None)

    # plot grand average outcomes by opto tag for short ISI in the next subplot
    ax = fig.add_subplot(gs[2, 2])
    plot_grand_average_outcomes_by_opto_tag_short(sessions_outcome, sessions_trial_type, sessions_opto_tag, ax=ax, outcome_colors=None)

    # plot outcomes by opto tag for long ISI in the next subplot
    ax = fig.add_subplot(gs[3, 0])
    plot_outcomes_by_opto_tag_long(sessions_outcome, sessions_date, sessions_trial_type, sessions_opto_tag, ax=ax, outcome_colors=None)

    # plot pooled outcomes by opto tag for long ISI in the next subplot
    ax = fig.add_subplot(gs[3, 1])
    plot_pooled_outcomes_by_opto_tag_long(sessions_outcome, sessions_trial_type, sessions_opto_tag, ax=ax, outcome_colors=None)

    # plot grand average outcomes by opto tag for long ISI in the next subplot
    ax = fig.add_subplot(gs[3, 2])
    plot_grand_average_outcomes_by_opto_tag_long(sessions_outcome, sessions_trial_type, sessions_opto_tag, ax=ax, outcome_colors=None)
    
    # Plot decsions for each session
    ax = fig.add_subplot(gs[4,0])
    plot_decision_sidedness(sessions_outcome, sessions_date, sessions_trial_type, ax=ax)

    # Plot pooled decision sidedness
    ax = fig.add_subplot(gs[4,1])
    plot_pooled_decision_sidedness(sessions_outcome, sessions_trial_type, ax=ax)

    # Plot average decision sidedness
    ax = fig.add_subplot(gs[4,2])
    plot_average_decision_sidedness(sessions_outcome, sessions_trial_type, ax=ax)


    plt.subplots_adjust(hspace= 1, wspace= 2)  # Add space between subplots
    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    return fig