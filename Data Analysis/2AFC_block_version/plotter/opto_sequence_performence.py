import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Single session by session plot
def plot_session_outcomes_by_opto_sequence(sessions_outcome, sessions_opto_tag, sessions_date, ax=None, outcome_colors=None):
    """
    Plot a bar chart for each session showing outcome distribution for pre-opto, opto, and post-opto trials,
    with x-ticks showing session dates. Shows outcomes as percentages and uses a specific order for outcomes.

    Parameters:
    -----------
    sessions_outcome : list of lists
        List containing outcome lists for each session
    sessions_opto_tag : list of lists
        List containing opto tag lists (0 or 1) for each session
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
            'Punish': '#e74c3c',         # Red
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
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y%m%d']:
                    try:
                        date_obj = datetime.strptime(date, fmt)
                        formatted_dates.append(date_obj)
                        break
                    except ValueError:
                        continue
                else:
                    formatted_dates.append(date)
            except Exception:
                formatted_dates.append(date)
        else:
            formatted_dates.append(date)

    # Define specific order for outcomes
    outcome_order = [
        'Reward',
        'Punish',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]

    # Identify present outcomes
    filtered_outcomes = set()
    for session_outcomes in sessions_outcome:
        for outcome in session_outcomes:
            if outcome not in ['RewardNaive', 'PunishNaive']:
                filtered_outcomes.add(outcome)

    ordered_outcomes = [o for o in outcome_order if o in filtered_outcomes]
    for o in sorted(filtered_outcomes):
        if o not in ordered_outcomes:
            ordered_outcomes.append(o)

    # Set up x-axis positions
    valid_sessions = []
    valid_dates = []
    x_positions = []
    pos_offset = 0
    bar_width = 0.2
    gap = 0.01
    group_width = 1.0

    for i, (outcomes, opto_tags, date) in enumerate(zip(sessions_outcome, sessions_opto_tag, formatted_dates)):
        # Skip sessions with no opto trials
        if 1 not in opto_tags:
            continue
        valid_sessions.append(i)
        valid_dates.append(date)
        x_positions.extend([pos_offset, pos_offset + bar_width + gap, pos_offset + 2 * (bar_width + gap)])
        pos_offset += group_width + bar_width

    # Process each session
    pre_percentages = []
    opto_percentages = []
    post_percentages = []

    for i, (outcomes, opto_tags) in enumerate(zip(sessions_outcome, sessions_opto_tag)):
        if 1 not in opto_tags:
            continue

        pre_opto_outcomes = []
        opto_outcomes = []
        post_opto_outcomes = []

        # Assign outcomes to pre-opto, opto, or post-opto
        for j in range(len(outcomes)):
            if j < len(opto_tags):
                if opto_tags[j] == 1:
                    opto_outcomes.append(outcomes[j])
                    if j + 1 < len(outcomes):
                        post_opto_outcomes.append(outcomes[j + 1])
                elif j > 0 and opto_tags[j - 1] == 1:
                    continue
                else:
                    if j + 1 < len(opto_tags) and opto_tags[j + 1] == 1:
                        pre_opto_outcomes.append(outcomes[j])

        # Filter out naive outcomes
        pre_opto_outcomes = [o for o in pre_opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]
        opto_outcomes = [o for o in opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]
        post_opto_outcomes = [o for o in post_opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]

        # Calculate percentages
        total_pre = len(pre_opto_outcomes)
        total_opto = len(opto_outcomes)
        total_post = len(post_opto_outcomes)

        pre_pct = []
        opto_pct = []
        post_pct = []

        for outcome in ordered_outcomes:
            pre_count = pre_opto_outcomes.count(outcome)
            pre_pct.append((pre_count / total_pre * 100) if total_pre > 0 else 0)

            opto_count = opto_outcomes.count(outcome)
            opto_pct.append((opto_count / total_opto * 100) if total_opto > 0 else 0)

            post_count = post_opto_outcomes.count(outcome)
            post_pct.append((post_count / total_post * 100) if total_post > 0 else 0)

        pre_percentages.append(pre_pct)
        opto_percentages.append(opto_pct)
        post_percentages.append(post_pct)

    # Plot bars
    for i, session_idx in enumerate(valid_sessions):
        pre_bottom = 0
        opto_bottom = 0
        post_bottom = 0
        pos_idx = i * 3

        for j, outcome in enumerate(ordered_outcomes):
            color = outcome_colors.get(outcome, outcome_colors.get('Other', '#7f8c8d'))
            ax.bar(x_positions[pos_idx], pre_percentages[i][j], width=bar_width, bottom=pre_bottom,
                   color=color, label=outcome if i == 0 else None)
            ax.bar(x_positions[pos_idx + 1], opto_percentages[i][j], width=bar_width, bottom=opto_bottom,
                   color=color, label=None)
            ax.bar(x_positions[pos_idx + 2], post_percentages[i][j], width=bar_width, bottom=post_bottom,
                   color=color, label=None)

            pre_bottom += pre_percentages[i][j]
            opto_bottom += opto_percentages[i][j]
            post_bottom += post_percentages[i][j]

    # Set x-axis labels
    if all(isinstance(date, datetime) for date in valid_dates):
        date_labels = [date.strftime('%Y-%m-%d') for date in valid_dates]
    else:
        date_labels = [str(date) for date in valid_dates]

    # Create group labels centered under each group
    group_positions = [i + bar_width for i in range(0, len(x_positions), 3)]
    ax.set_xticks(group_positions)
    ax.set_xticklabels(date_labels, rotation=45, ha='right')

    # Add secondary x-axis labels for pre/opto/post
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(['Pre', 'Opto', 'Post'] * len(valid_sessions), rotation=45, ha='right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # Labels and title
    ax.set_xlabel('Session Date')
    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title('Session Outcomes by Opto Sequence (Pre-Opto vs Opto vs Post-Opto)')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    # Add horizontal line at 50%
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    return ax

# pooled data outcome

def plot_pooled_outcomes_by_opto_sequence(sessions_outcome, sessions_opto_tag, ax=None, outcome_colors=None):
    """
    Plot a bar chart showing outcome distribution for pooled pre-opto, opto, and post-opto trials
    across all sessions, with one bar each for pre-opto, opto, and post-opto.
    Shows outcomes as percentages and uses a specific order for outcomes.

    Parameters:
    -----------
    sessions_outcome : list of lists
        List containing outcome lists for each session
    sessions_opto_tag : list of lists
        List containing opto tag lists (0 or 1) for each session
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

    # Identify present outcomes
    filtered_outcomes = set()
    for session_outcomes in sessions_outcome:
        for outcome in session_outcomes:
            if outcome not in ['RewardNaive', 'PunishNaive']:
                filtered_outcomes.add(outcome)

    ordered_outcomes = [o for o in outcome_order if o in filtered_outcomes]
    for o in sorted(filtered_outcomes):
        if o not in ordered_outcomes:
            ordered_outcomes.append(o)

    # Pool trials across sessions
    pre_opto_outcomes = []
    opto_outcomes = []
    post_opto_outcomes = []

    for outcomes, opto_tags in zip(sessions_outcome, sessions_opto_tag):
        # Skip sessions with no opto trials
        if 1 not in opto_tags:
            continue

        # Assign outcomes to pre-opto, opto, or post-opto
        for i in range(len(outcomes)):
            if i < len(opto_tags):
                if opto_tags[i] == 1:
                    opto_outcomes.append(outcomes[i])
                    if i + 1 < len(outcomes):
                        post_opto_outcomes.append(outcomes[i + 1])
                elif i > 0 and opto_tags[i - 1] == 1:
                    continue
                else:
                    if i + 1 < len(opto_tags) and opto_tags[i + 1] == 1:
                        pre_opto_outcomes.append(outcomes[i])

    # Filter out naive outcomes
    pre_opto_outcomes = [o for o in pre_opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]
    opto_outcomes = [o for o in opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]
    post_opto_outcomes = [o for o in post_opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]

    # Calculate percentages for pooled data
    total_pre = len(pre_opto_outcomes)
    total_opto = len(opto_outcomes)
    total_post = len(post_opto_outcomes)

    pre_percentages = []
    opto_percentages = []
    post_percentages = []

    for outcome in ordered_outcomes:
        pre_count = pre_opto_outcomes.count(outcome)
        pre_pct = (pre_count / total_pre * 100) if total_pre > 0 else 0
        pre_percentages.append(pre_pct)

        opto_count = opto_outcomes.count(outcome)
        opto_pct = (opto_count / total_opto * 100) if total_opto > 0 else 0
        opto_percentages.append(opto_pct)

        post_count = post_opto_outcomes.count(outcome)
        post_pct = (post_count / total_post * 100) if total_post > 0 else 0
        post_percentages.append(post_pct)

    # Plotting
    bar_width = 0.4
    positions = [0, 1, 2]
    pre_pos, opto_pos, post_pos = positions

    pre_bottom = 0
    opto_bottom = 0
    post_bottom = 0

    for outcome, pre_pct, opto_pct, post_pct in zip(ordered_outcomes, pre_percentages, opto_percentages, post_percentages):
        color = outcome_colors.get(outcome, outcome_colors['Other'])

        ax.bar(pre_pos, pre_pct, width=bar_width, bottom=pre_bottom, 
               color=color, label=outcome)
        ax.bar(opto_pos, opto_pct, width=bar_width, bottom=opto_bottom, 
               color=color, label=outcome)
        ax.bar(post_pos, post_pct, width=bar_width, bottom=post_bottom, 
               color=color, label=outcome)

        pre_bottom += pre_pct
        opto_bottom += opto_pct
        post_bottom += post_pct

    # Set x-ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(['Pre-Opto', 'Opto', 'Post-Opto'])

    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Labels and title
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title('Pooled Outcomes Across Sessions (Pre-Opto vs Opto vs Post-Opto)\n')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    return ax

# Grand average
def plot_outcomes_by_opto_sequence_grand_average(sessions_outcome, sessions_opto_tag, ax=None, outcome_colors=None):
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
        fig, ax = plt.subplots(figsize=(10, 8))

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
    pre_opto_percentages = {outcome: [] for outcome in ordered_outcomes}
    opto_percentages = {outcome: [] for outcome in ordered_outcomes}
    post_opto_percentages = {outcome: [] for outcome in ordered_outcomes}

    for outcomes, opto_tags in zip(sessions_outcome, sessions_opto_tag):
        # Check if session has any opto trials
        if 1 not in opto_tags:
            continue  # Skip sessions with no opto trials

        pre_opto_outcomes = []
        opto_outcomes = []
        post_opto_outcomes = []

        # Iterate through trials to assign outcomes to pre-opto, opto, or post-opto
        for i in range(len(outcomes)):
            if i < len(opto_tags):
                if opto_tags[i] == 1:
                    # Current trial is opto
                    opto_outcomes.append(outcomes[i])
                    # If there's a next trial, it's post-opto
                    if i + 1 < len(outcomes):
                        post_opto_outcomes.append(outcomes[i + 1])
                elif i > 0 and opto_tags[i - 1] == 1:
                    # Skip post-opto trials as they're already assigned
                    continue
                else:
                    # Current trial is pre-opto if followed by an opto trial
                    if i + 1 < len(opto_tags) and opto_tags[i + 1] == 1:
                        pre_opto_outcomes.append(outcomes[i])

        # Filter out naive outcomes
        pre_opto_outcomes = [o for o in pre_opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]
        opto_outcomes = [o for o in opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]
        post_opto_outcomes = [o for o in post_opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]

        # Calculate percentages
        total_pre = len(pre_opto_outcomes)
        total_opto = len(opto_outcomes)
        total_post = len(post_opto_outcomes)

        for outcome in ordered_outcomes:
            pre_count = pre_opto_outcomes.count(outcome)
            pre_pct = (pre_count / total_pre * 100) if total_pre > 0 else 0
            pre_opto_percentages[outcome].append(pre_pct)

            opto_count = opto_outcomes.count(outcome)
            opto_pct = (opto_count / total_opto * 100) if total_opto > 0 else 0
            opto_percentages[outcome].append(opto_pct)

            post_count = post_opto_outcomes.count(outcome)
            post_pct = (post_count / total_post * 100) if total_post > 0 else 0
            post_opto_percentages[outcome].append(post_pct)

    # Compute grand average percentages
    pre_mean_percentages = []
    opto_mean_percentages = []
    post_mean_percentages = []
    for outcome in ordered_outcomes:
        pre_mean = np.mean(pre_opto_percentages[outcome]) if pre_opto_percentages[outcome] else 0
        opto_mean = np.mean(opto_percentages[outcome]) if opto_percentages[outcome] else 0
        post_mean = np.mean(post_opto_percentages[outcome]) if post_opto_percentages[outcome] else 0
        pre_mean_percentages.append(pre_mean)
        opto_mean_percentages.append(opto_mean)
        post_mean_percentages.append(post_mean)

    # Plotting
    bar_width = 0.4
    positions = [0, 1, 2]
    pre_pos, opto_pos, post_pos = positions

    pre_bottom = 0
    opto_bottom = 0
    post_bottom = 0

    for outcome, pre_pct, opto_pct, post_pct in zip(ordered_outcomes, pre_mean_percentages, opto_mean_percentages, post_mean_percentages):
        color = outcome_colors.get(outcome, outcome_colors['Other'])

        ax.bar(pre_pos, pre_pct, width=bar_width, bottom=pre_bottom, 
               color=color, label=outcome)
        ax.bar(opto_pos, opto_pct, width=bar_width, bottom=opto_bottom, 
               color=color, label=outcome)
        ax.bar(post_pos, post_pct, width=bar_width, bottom=post_bottom, 
               color=color, label=outcome)

        pre_bottom += pre_pct
        opto_bottom += opto_pct
        post_bottom += post_pct

    # Set x-ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(['Pre-Opto', 'Opto', 'Post-Opto'])

    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Labels and title
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    ax.set_ylabel('Mean Percentage of Trials (%)')
    ax.set_title('Grand Average Session Outcomes (Pre-Opto vs Opto vs Post-Opto)\n')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    return ax

##################
def plot_session_outcomes_by_opto_sequence_trial_type(sessions_outcome, sessions_opto_tag, sessions_trial_type, sessions_date, trial_type=2, ax=None, outcome_colors=None):
    """
    Plot a bar chart for each session showing outcome distribution for pre-opto, opto, and post-opto trials,
    with trial type filtering applied only to opto trials (1 for short, 2 for long), and x-ticks showing session dates.
    Shows outcomes as percentages and uses a specific order for outcomes.

    Parameters:
    -----------
    sessions_outcome : list of lists
        List containing outcome lists for each session
    sessions_opto_tag : list of lists
        List containing opto tag lists (0 or 1) for each session
    sessions_trial_type : list of lists
        List containing trial type lists (1 for short, 2 for long) for each session
    sessions_date : list
        List of session dates
    trial_type : int, optional
        Trial type to filter opto trials (1 for short, 2 for long). Default is 2 (long).
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
            'Punish': '#e74c3c',         # Red
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
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y%m%d']:
                    try:
                        date_obj = datetime.strptime(date, fmt)
                        formatted_dates.append(date_obj)
                        break
                    except ValueError:
                        continue
                else:
                    formatted_dates.append(date)
            except Exception:
                formatted_dates.append(date)
        else:
            formatted_dates.append(date)

    # Define specific order for outcomes
    outcome_order = [
        'Reward',
        'Punish',
        'WrongInitiation',
        'ChangingMindReward',
        'EarlyChoice',
        'DidNotChoose',
        'Other'
    ]

    # Identify present outcomes
    filtered_outcomes = set()
    for session_outcomes in sessions_outcome:
        for outcome in session_outcomes:
            if outcome not in ['RewardNaive', 'PunishNaive']:
                filtered_outcomes.add(outcome)

    ordered_outcomes = [o for o in outcome_order if o in filtered_outcomes]
    for o in sorted(filtered_outcomes):
        if o not in ordered_outcomes:
            ordered_outcomes.append(o)

    # Set up x-axis positions
    valid_sessions = []
    valid_dates = []
    x_positions = []
    pos_offset = 0
    bar_width = 0.2
    gap = 0.01
    group_width = 1.0

    for i, (outcomes, opto_tags, trial_types) in enumerate(zip(sessions_outcome, sessions_opto_tag, sessions_trial_type)):
        # Check if session has any opto trials of the specified trial type
        has_valid_opto = any(opto_tags[j] == 1 and trial_types[j] == trial_type for j in range(min(len(opto_tags), len(trial_types))))
        if not has_valid_opto:
            continue
        valid_sessions.append(i)
        valid_dates.append(formatted_dates[i])
        x_positions.extend([pos_offset, pos_offset + bar_width + gap, pos_offset + 2 * (bar_width + gap)])
        pos_offset += group_width + bar_width

    # Process each session
    pre_percentages = []
    opto_percentages = []
    post_percentages = []

    for i, (outcomes, opto_tags, trial_types) in enumerate(zip(sessions_outcome, sessions_opto_tag, sessions_trial_type)):
        if not any(opto_tags[j] == 1 and trial_types[j] == trial_type for j in range(min(len(opto_tags), len(trial_types)))):
            continue

        pre_opto_outcomes = []
        opto_outcomes = []
        post_opto_outcomes = []

        # Assign outcomes to pre-opto, opto, or post-opto, filtering opto trials by trial type
        for j in range(len(outcomes)):
            if j >= len(opto_tags) or j >= len(trial_types):
                continue
            if opto_tags[j] == 1 and trial_types[j] == trial_type:
                opto_outcomes.append(outcomes[j])
                if j + 1 < len(outcomes):
                    post_opto_outcomes.append(outcomes[j + 1])
            elif j + 1 < len(opto_tags) and opto_tags[j + 1] == 1 and j + 1 < len(trial_types) and trial_types[j + 1] == trial_type:
                pre_opto_outcomes.append(outcomes[j])

        # Filter out naive outcomes
        pre_opto_outcomes = [o for o in pre_opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]
        opto_outcomes = [o for o in opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]
        post_opto_outcomes = [o for o in post_opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]

        # Calculate percentages
        total_pre = len(pre_opto_outcomes)
        total_opto = len(opto_outcomes)
        total_post = len(post_opto_outcomes)

        pre_pct = []
        opto_pct = []
        post_pct = []

        for outcome in ordered_outcomes:
            pre_count = pre_opto_outcomes.count(outcome)
            pre_pct.append((pre_count / total_pre * 100) if total_pre > 0 else 0)

            opto_count = opto_outcomes.count(outcome)
            opto_pct.append((opto_count / total_opto * 100) if total_opto > 0 else 0)

            post_count = post_opto_outcomes.count(outcome)
            post_pct.append((post_count / total_post * 100) if total_post > 0 else 0)

        pre_percentages.append(pre_pct)
        opto_percentages.append(opto_pct)
        post_percentages.append(post_pct)

    # Plot bars
    for i, session_idx in enumerate(valid_sessions):
        pre_bottom = 0
        opto_bottom = 0
        post_bottom = 0
        pos_idx = i * 3

        for j, outcome in enumerate(ordered_outcomes):
            color = outcome_colors.get(outcome, outcome_colors.get('Other', '#7f8c8d'))
            ax.bar(x_positions[pos_idx], pre_percentages[i][j], width=bar_width, bottom=pre_bottom,
                   color=color, label=outcome if i == 0 else None)
            ax.bar(x_positions[pos_idx + 1], opto_percentages[i][j], width=bar_width, bottom=opto_bottom,
                   color=color, label=None)
            ax.bar(x_positions[pos_idx + 2], post_percentages[i][j], width=bar_width, bottom=post_bottom,
                   color=color, label=None)

            pre_bottom += pre_percentages[i][j]
            opto_bottom += opto_percentages[i][j]
            post_bottom += post_percentages[i][j]

    # Set x-axis labels
    if all(isinstance(date, datetime) for date in valid_dates):
        date_labels = [date.strftime('%Y-%m-%d') for date in valid_dates]
    else:
        date_labels = [str(date) for date in valid_dates]

    # Create group labels centered under each group
    group_positions = [i + bar_width for i in range(0, len(x_positions), 3)]
    ax.set_xticks(group_positions)
    ax.set_xticklabels(date_labels, rotation=45, ha='right')

    # Add secondary x-axis labels for pre/opto/post
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(['Pre', 'Opto', 'Post'] * len(valid_sessions), rotation=45, ha='right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # Labels and title
    trial_type_label = 'Long ISI' if trial_type == 2 else 'Short ISI'
    ax.set_xlabel('Session Date')
    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title(f'Session Outcomes by Opto Sequence ({trial_type_label} Opto Trials: Pre-Opto vs Opto vs Post-Opto)')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    # Add horizontal line at 50%
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    return ax

###################################################
def plot_pooled_outcomes_by_opto_sequence_trial_type(sessions_outcome, sessions_opto_tag, sessions_trial_type, trial_type=2, ax=None, outcome_colors=None):
    """
    Plot a bar chart showing outcome distribution for pooled pre-opto, opto, and post-opto trials
    across all sessions, with trial type filtering applied only to opto trials (1 for short, 2 for long).
    Shows outcomes as percentages and uses a specific order for outcomes.

    Parameters:
    -----------
    sessions_outcome : list of lists
        List containing outcome lists for each session
    sessions_opto_tag : list of lists
        List containing opto tag lists (0 or 1) for each session
    sessions_trial_type : list of lists
        List containing trial type lists (1 for short, 2 for long) for each session
    trial_type : int, optional
        Trial type to filter opto trials (1 for short, 2 for long). Default is 2 (long).
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

    # Identify present outcomes
    filtered_outcomes = set()
    for session_outcomes in sessions_outcome:
        for outcome in session_outcomes:
            if outcome not in ['RewardNaive', 'PunishNaive']:
                filtered_outcomes.add(outcome)

    ordered_outcomes = [o for o in outcome_order if o in filtered_outcomes]
    for o in sorted(filtered_outcomes):
        if o not in ordered_outcomes:
            ordered_outcomes.append(o)

    # Pool trials across sessions
    pre_opto_outcomes = []
    opto_outcomes = []
    post_opto_outcomes = []

    for outcomes, opto_tags, trial_types in zip(sessions_outcome, sessions_opto_tag, sessions_trial_type):
        # Skip sessions with no opto trials of the specified trial type
        if not any(opto_tags[j] == 1 and trial_types[j] == trial_type for j in range(min(len(opto_tags), len(trial_types)))):
            continue

        # Assign outcomes to pre-opto, opto, or post-opto, filtering opto trials by trial type
        for i in range(len(outcomes)):
            if i >= len(opto_tags) or i >= len(trial_types):
                continue
            if opto_tags[i] == 1 and trial_types[i] == trial_type:
                opto_outcomes.append(outcomes[i])
                if i + 1 < len(outcomes):
                    post_opto_outcomes.append(outcomes[i + 1])
            elif i + 1 < len(opto_tags) and opto_tags[i + 1] == 1 and i + 1 < len(trial_types) and trial_types[i + 1] == trial_type:
                pre_opto_outcomes.append(outcomes[i])

    # Filter out naive outcomes
    pre_opto_outcomes = [o for o in pre_opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]
    opto_outcomes = [o for o in opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]
    post_opto_outcomes = [o for o in post_opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]

    # Calculate percentages for pooled data
    total_pre = len(pre_opto_outcomes)
    total_opto = len(opto_outcomes)
    total_post = len(post_opto_outcomes)

    pre_percentages = []
    opto_percentages = []
    post_percentages = []

    for outcome in ordered_outcomes:
        pre_count = pre_opto_outcomes.count(outcome)
        pre_pct = (pre_count / total_pre * 100) if total_pre > 0 else 0
        pre_percentages.append(pre_pct)

        opto_count = opto_outcomes.count(outcome)
        opto_pct = (opto_count / total_opto * 100) if total_opto > 0 else 0
        opto_percentages.append(opto_pct)

        post_count = post_opto_outcomes.count(outcome)
        post_pct = (post_count / total_post * 100) if total_post > 0 else 0
        post_percentages.append(post_pct)

    # Plotting
    bar_width = 0.4
    positions = [0, 1, 2]
    pre_pos, opto_pos, post_pos = positions

    pre_bottom = 0
    opto_bottom = 0
    post_bottom = 0

    for outcome, pre_pct, opto_pct, post_pct in zip(ordered_outcomes, pre_percentages, opto_percentages, post_percentages):
        color = outcome_colors.get(outcome, outcome_colors['Other'])

        ax.bar(pre_pos, pre_pct, width=bar_width, bottom=pre_bottom, 
               color=color, label=outcome)
        ax.bar(opto_pos, opto_pct, width=bar_width, bottom=opto_bottom, 
               color=color, label=outcome)
        ax.bar(post_pos, post_pct, width=bar_width, bottom=post_bottom, 
               color=color, label=outcome)

        pre_bottom += pre_pct
        opto_bottom += opto_pct
        post_bottom += post_pct

    # Set x-ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(['Pre-Opto', 'Opto', 'Post-Opto'])

    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Labels and title
    trial_type_label = 'Long ISI' if trial_type == 2 else 'Short ISI'
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title(f'Pooled Outcomes Across Sessions ({trial_type_label} Opto Trials: Pre-Opto vs Opto vs Post-Opto)\n')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    return ax

####################################
def plot_grand_average_outcomes_by_opto_sequence_trial_type(sessions_outcome, sessions_opto_tag, sessions_trial_type, trial_type=2, ax=None, outcome_colors=None):
    """
    Plot a bar chart showing the grand average outcome distribution for pre-opto, opto, and post-opto trials
    across all sessions, with trial type filtering applied only to opto trials (1 for short, 2 for long).
    Shows outcomes as mean percentages across sessions and uses a specific order for outcomes.

    Parameters:
    -----------
    sessions_outcome : list of lists
        List containing outcome lists for each session
    sessions_opto_tag : list of lists
        List containing opto tag lists (0 or 1) for each session
    sessions_trial_type : list of lists
        List containing trial type lists (1 for short, 2 for long) for each session
    trial_type : int, optional
        Trial type to filter opto trials (1 for short, 2 for long). Default is 2 (long).
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

    # Identify present outcomes
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
    pre_opto_percentages = {outcome: [] for outcome in ordered_outcomes}
    opto_percentages = {outcome: [] for outcome in ordered_outcomes}
    post_opto_percentages = {outcome: [] for outcome in ordered_outcomes}

    for outcomes, opto_tags, trial_types in zip(sessions_outcome, sessions_opto_tag, sessions_trial_type):
        # Skip sessions with no opto trials of the specified trial type
        if not any(opto_tags[j] == 1 and trial_types[j] == trial_type for j in range(min(len(opto_tags), len(trial_types)))):
            continue

        pre_opto_outcomes = []
        opto_outcomes = []
        post_opto_outcomes = []

        # Assign outcomes to pre-opto, opto, or post-opto, filtering opto trials by trial type
        for i in range(len(outcomes)):
            if i >= len(opto_tags) or i >= len(trial_types):
                continue
            if opto_tags[i] == 1 and trial_types[i] == trial_type:
                opto_outcomes.append(outcomes[i])
                if i + 1 < len(outcomes):
                    post_opto_outcomes.append(outcomes[i + 1])
            elif i + 1 < len(opto_tags) and opto_tags[i + 1] == 1 and i + 1 < len(trial_types) and trial_types[i + 1] == trial_type:
                pre_opto_outcomes.append(outcomes[i])

        # Filter out naive outcomes
        pre_opto_outcomes = [o for o in pre_opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]
        opto_outcomes = [o for o in opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]
        post_opto_outcomes = [o for o in post_opto_outcomes if o not in ['RewardNaive', 'PunishNaive']]

        # Calculate percentages
        total_pre = len(pre_opto_outcomes)
        total_opto = len(opto_outcomes)
        total_post = len(post_opto_outcomes)

        for outcome in ordered_outcomes:
            pre_count = pre_opto_outcomes.count(outcome)
            pre_pct = (pre_count / total_pre * 100) if total_pre > 0 else 0
            pre_opto_percentages[outcome].append(pre_pct)

            opto_count = opto_outcomes.count(outcome)
            opto_pct = (opto_count / total_opto * 100) if total_opto > 0 else 0
            opto_percentages[outcome].append(opto_pct)

            post_count = post_opto_outcomes.count(outcome)
            post_pct = (post_count / total_post * 100) if total_post > 0 else 0
            post_opto_percentages[outcome].append(post_pct)

    # Compute grand average percentages
    pre_mean_percentages = []
    opto_mean_percentages = []
    post_mean_percentages = []
    for outcome in ordered_outcomes:
        pre_mean = np.mean(pre_opto_percentages[outcome]) if pre_opto_percentages[outcome] else 0
        opto_mean = np.mean(opto_percentages[outcome]) if opto_percentages[outcome] else 0
        post_mean = np.mean(post_opto_percentages[outcome]) if post_opto_percentages[outcome] else 0
        pre_mean_percentages.append(pre_mean)
        opto_mean_percentages.append(opto_mean)
        post_mean_percentages.append(post_mean)

    # Plotting
    bar_width = 0.4
    positions = [0, 1, 2]
    pre_pos, opto_pos, post_pos = positions

    pre_bottom = 0
    opto_bottom = 0
    post_bottom = 0

    for outcome, pre_pct, opto_pct, post_pct in zip(ordered_outcomes, pre_mean_percentages, opto_mean_percentages, post_mean_percentages):
        color = outcome_colors.get(outcome, outcome_colors['Other'])

        ax.bar(pre_pos, pre_pct, width=bar_width, bottom=pre_bottom, 
               color=color, label=outcome)
        ax.bar(opto_pos, opto_pct, width=bar_width, bottom=opto_bottom, 
               color=color, label=outcome)
        ax.bar(post_pos, post_pct, width=bar_width, bottom=post_bottom, 
               color=color, label=outcome)

        pre_bottom += pre_pct
        opto_bottom += opto_pct
        post_bottom += post_pct

    # Set x-ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(['Pre-Opto', 'Opto', 'Post-Opto'])

    # Add a horizontal line at 50% for reference
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Labels and title
    trial_type_label = 'Long ISI' if trial_type == 2 else 'Short ISI'
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='best')

    ax.set_ylabel('Mean Percentage of Trials (%)')
    ax.set_title(f'Grand Average Session Outcomes ({trial_type_label} Opto Trials: Pre-Opto vs Opto vs Post-Opto)\n')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    return ax

def plot_opto_seq_with_gridspec(sessions_outcome, sessions_date, sessions_trial_type, sessions_opto_tag, outcome_colors=None, figsize=(40, 30)):
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
    gs = plt.GridSpec(3, 3,  height_ratios=[1, 1, 1], width_ratios=[10, 1, 1])  # Just one plot area
    ax = fig.add_subplot(gs[0, 0])
    
    # Plot all sessions in this single axis
    plot_session_outcomes_by_opto_sequence(sessions_outcome, sessions_opto_tag, sessions_date, ax=ax, outcome_colors=outcome_colors)

    # Plot pooled session outcomes in the next subplot
    ax = fig.add_subplot(gs[0, 1])
    plot_pooled_outcomes_by_opto_sequence(sessions_outcome, sessions_opto_tag, ax=ax, outcome_colors=None)

    # Plot grand average session outcomes in the next subplot
    ax = fig.add_subplot(gs[0, 2])
    plot_outcomes_by_opto_sequence_grand_average(sessions_outcome, sessions_opto_tag, ax=ax, outcome_colors=None)

    # plot all for short opto trial
    ax = fig.add_subplot(gs[1, 0])
    plot_session_outcomes_by_opto_sequence_trial_type(sessions_outcome, sessions_opto_tag, sessions_trial_type, sessions_date, trial_type=1, ax=ax, outcome_colors=None)

    # plot pooled data for short opto trials
    ax = fig.add_subplot(gs[1,1])
    plot_pooled_outcomes_by_opto_sequence_trial_type(sessions_outcome, sessions_opto_tag, sessions_trial_type, trial_type=1, ax=ax, outcome_colors=None)

    # Plot Grand average of opto short trials
    ax = fig.add_subplot(gs[1,2])
    plot_grand_average_outcomes_by_opto_sequence_trial_type(sessions_outcome, sessions_opto_tag, sessions_trial_type, trial_type=1, ax=ax, outcome_colors=None)

    # plot all for long opto trial
    ax = fig.add_subplot(gs[2, 0])
    plot_session_outcomes_by_opto_sequence_trial_type(sessions_outcome, sessions_opto_tag, sessions_trial_type, sessions_date, trial_type=2, ax=ax, outcome_colors=None)

    # plot pooled data for long opto trials
    ax = fig.add_subplot(gs[2,1])
    plot_pooled_outcomes_by_opto_sequence_trial_type(sessions_outcome, sessions_opto_tag, sessions_trial_type, trial_type=2, ax=ax, outcome_colors=None)

    # Plot Grand average of opto long trials
    ax = fig.add_subplot(gs[2,2])
    plot_grand_average_outcomes_by_opto_sequence_trial_type(sessions_outcome, sessions_opto_tag, sessions_trial_type, trial_type=2, ax=ax, outcome_colors=None)

    plt.subplots_adjust(hspace= 1, wspace= 3)  # Add space between subplots
    plt.tight_layout()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    return fig