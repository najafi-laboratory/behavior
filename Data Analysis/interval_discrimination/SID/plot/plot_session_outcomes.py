# -*- coding: utf-8 -*-
"""
Created on Sat May  3 19:52:31 2025

@author: timst
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
    
def plot_trial_outcomes_by_session(
    df,
    split_opto=True,
    min_trials=10,
    sort_sessions=True,
    show_counts=True,
    normalize=True,
    bar_spacing=0.3,  # NEW: spacing between sessions
    ax=None,
    outcome_col='outcome',
    side_col='trial_side',
    opto_col='is_opto',
    session_col='date',
    show_plot=True
):
    """
    Plot trial outcome bars per session, grouped by trial side.
    Optionally split control/opto and normalize to percentage.

    Parameters:
    - df : pd.DataFrame
    - split_opto : bool, if True, split control vs opto
    - min_trials : int, sessions with fewer trials are excluded
    - sort_sessions : bool, sort sessions by date
    - show_counts : bool, annotate bars with trial count or percent
    - normalize : bool, plot as percentage instead of count
    - bar_spacing : float, spacing between bar groups
    - ax : matplotlib axis (optional)
    """


    outcome_colors = {
        'Reward': '#2ca02c',         # green
        'Punish': '#d62728',         # red
        'DidNotChoose': '#e0e0e0',   # very light grey
        'RewardNaive': '#98df8a',    # light green
        'PunishNaive': '#ff9896'     # light red
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 3.5))
    else:
        fig = ax.figure

    df = df.copy()

    group_cols = [session_col, side_col]
    if split_opto:
        group_cols.append(opto_col)
    group_cols.append(outcome_col)

    summary = df.groupby(group_cols).size().unstack(fill_value=0)

    session_ticks = []
    session_labels = []

    sessions = sorted(df[session_col].unique()) if sort_sessions else df[session_col].unique()
    x = 0
    label_order = ['left', 'right']

    for session in sessions:
        session_df = summary.loc[session] if session in summary.index.get_level_values(0) else None
        if session_df is None or session_df.sum().sum() < min_trials:
            continue

        x_positions = []

        # Build expected group order for this session
        expected_groups = []
        for side in label_order:
            if split_opto:
                expected_groups.append((side, 0))  # control
                expected_groups.append((side, 1))  # opto
            else:
                expected_groups.append(side)

        for group in expected_groups:
            if split_opto:
                if group not in session_df.index:
                    outcome_counts = {k: 0 for k in outcome_colors.keys()}
                else:
                    outcome_counts = session_df.loc[group].to_dict()
            else:
                if group not in session_df.index:
                    outcome_counts = {k: 0 for k in outcome_colors.keys()}
                else:
                    outcome_counts = session_df.loc[group].to_dict()

            # Sort outcome keys by color map order
            outcomes_sorted = [k for k in outcome_colors if k in outcome_counts]

            # Compute total and optionally normalize
            total = sum(outcome_counts.values())
            bottom = 0
            for outcome in outcomes_sorted:
                count = outcome_counts[outcome]
                value = (count / total) if (normalize and total > 0) else count
                ax.bar(x, value, bottom=bottom, color=outcome_colors[outcome],
                       label=outcome if x == 0 else "", zorder=3)
                bottom += value

            # Add total count or percent on top
            if show_counts:
                # label_text = f"{int(total)}" if not normalize else f"{bottom * 100:.1f}%"
                label_text = ''
                ax.text(x, bottom + 0.02, label_text, ha='center', va='bottom', fontsize=8)

            # Label bar as L / LO / R / RO
            if split_opto:
                label = f"{group[0][0].upper()}{'O' if group[1] == 1 else ''}"
            else:
                label = group[0][0].upper()
            ax.text(x, -0.05 if normalize else -3, label, ha='center', va='top', fontsize=9, clip_on=False)
            # ax.text(x, -0.15 if normalize else -3, label, ha='center', va='top', fontsize=9, clip_on=False)

            x_positions.append(x)
            x += 1


        # Add spacing between session groups
        x += bar_spacing
        center_x = np.mean(x_positions)
        session_ticks.append(center_x)
        session_labels.append(session)

    ax.set_xticks(session_ticks)
    ax.set_xticklabels(session_labels, rotation=45, ha='right')
    ax.tick_params(axis='x', pad=12)  # increase to lower the labels
    ax.set_ylabel("Trial Proportion" if normalize else "Trial Count")
    # ax.set_title("Trial Outcomes by Session")
    # ax.set_title("Trial Outcomes by Session", y=1.05)
    
    subject = df['subject_name'].unique()[0]
    
    # If 'session_date' is not already datetime, convert it
    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min().strftime("%Y-%m-%d")
    max_date = df['date'].max().strftime("%Y-%m-%d")
    dates = min_date + '-' + max_date
    
    title = f"{subject} | Sessions {min_date} to {max_date}"
    ax.set_title(title, y=1.05)    
    
    
    
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    if show_plot:
        plt.show()



    # subject = session_info['subject_name']
    # date = session_info['date']
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject + '\\' + date)
    output_dir = os.path.join('plots\\' + subject)
    figure_id = f"{subject}_{dates}_session_outcomes"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches=None, pad_inches=0.1, dpi=300)
    plt.close(fig)       

    return out_path    



# def plot_trial_outcomes_by_session(
#     df,
#     split_opto=True,
#     min_trials=10,
#     sort_sessions=True,
#     show_counts=True,
#     normalize=True,
#     ax=None,
#     outcome_col='outcome',
#     side_col='trial_side',
#     opto_col='is_opto',
#     session_col='date'
# ):
#     """
#     Plot trial outcome bars per session, grouped by trial side.
#     Optionally split control/opto and normalize to percentage.

#     Parameters:
#     - df : pd.DataFrame
#     - split_opto : bool, if True, split control vs opto
#     - min_trials : int, sessions with fewer trials are excluded
#     - sort_sessions : bool, sort sessions by date
#     - show_counts : bool, annotate bars with trial count or percent
#     - normalize : bool, plot as percentage instead of count
#     - ax : matplotlib axis (optional)
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd

#     outcome_colors = {
#         'Reward': '#2ca02c',         # green
#         'Punish': '#d62728',         # red
#         'DidNotChoose': '#e0e0e0',   # very light grey
#         'RewardNaive': '#98df8a',    # light green
#         'PunishNaive': '#ff9896'     # light red
#     }

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(14, 6))
#     else:
#         fig = ax.figure

#     df = df.copy()

#     group_cols = [session_col, side_col]
#     if split_opto:
#         group_cols.append(opto_col)
#     group_cols.append(outcome_col)

#     summary = df.groupby(group_cols).size().unstack(fill_value=0)

#     session_ticks = []
#     session_labels = []

#     sessions = sorted(df[session_col].unique()) if sort_sessions else df[session_col].unique()
#     x = 0
#     label_order = ['left', 'right']

#     for session in sessions:
#         session_df = summary.loc[session] if session in summary.index.get_level_values(0) else None
#         if session_df is None or session_df.sum().sum() < min_trials:
#             continue

#         x_positions = []

#         # Build expected group order for this session
#         expected_groups = []
#         for side in label_order:
#             if split_opto:
#                 expected_groups.append((side, 0))  # control
#                 expected_groups.append((side, 1))  # opto
#             else:
#                 expected_groups.append(side)

#         for group in expected_groups:
#             if split_opto:
#                 if group not in session_df.index:
#                     outcome_counts = {k: 0 for k in outcome_colors.keys()}
#                 else:
#                     outcome_counts = session_df.loc[group].to_dict()
#             else:
#                 if group not in session_df.index:
#                     outcome_counts = {k: 0 for k in outcome_colors.keys()}
#                 else:
#                     outcome_counts = session_df.loc[group].to_dict()

#             # Sort outcome keys by color map order
#             outcomes_sorted = [k for k in outcome_colors if k in outcome_counts]

#             # Compute total and optionally normalize
#             total = sum(outcome_counts.values())
#             bottom = 0
#             for outcome in outcomes_sorted:
#                 count = outcome_counts[outcome]
#                 value = (count / total) if (normalize and total > 0) else count
#                 ax.bar(x, value, bottom=bottom, color=outcome_colors[outcome],
#                        label=outcome if x == 0 else "", zorder=3)
#                 bottom += value

#             # Add total count or percent on top
#             if show_counts:
#                 label_text = f"{int(total)}" if not normalize else f"{bottom * 100:.1f}%"
#                 ax.text(x, bottom + 0.02, label_text, ha='center', va='bottom', fontsize=8)

#             # Label bar as L / LO / R / RO
#             if split_opto:
#                 label = f"{group[0][0].upper()}{'O' if group[1] == 1 else ''}"
#             else:
#                 label = group[0][0].upper()
#             ax.text(x, -0.05 if normalize else -3, label, ha='center', va='top', fontsize=9, clip_on=False)

#             x_positions.append(x)
#             x += 1

#         center_x = np.mean(x_positions)
#         session_ticks.append(center_x)
#         session_labels.append(session)

#     ax.set_xticks(session_ticks)
#     ax.set_xticklabels(session_labels, rotation=45, ha='right')
#     ax.set_ylabel("Trial Proportion" if normalize else "Trial Count")
#     ax.set_title("Trial Outcomes by Session")
#     ax.legend(loc='upper right', fontsize=8)
#     ax.grid(True, axis='y', alpha=0.3)
#     fig.tight_layout()

#     return ax


# def plot_trial_outcomes_by_session(
#     df,
#     split_opto=True,
#     min_trials=10,
#     sort_sessions=True,
#     show_counts=True,
#     normalize=True,
#     ax=None,
#     rewarded_col='mouse_correct',
#     side_col='trial_side',
#     opto_col='is_opto',
#     session_col='date'
# ):
#     """
#     Plot rewarded vs punished trial bars per session, grouped by trial side.
#     Optionally split control/opto and normalize to percentage.

#     Parameters:
#     - df : pd.DataFrame
#     - split_opto : bool, if True, split control vs opto
#     - min_trials : int, sessions with fewer trials are excluded
#     - sort_sessions : bool, sort sessions by date
#     - show_counts : bool, annotate bars with trial count
#     - normalize : bool, plot as percentage instead of count
#     - ax : matplotlib axis (optional)
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(14, 6))
#     else:
#         fig = ax.figure

#     df = df.copy()
#     df[rewarded_col] = df[rewarded_col].astype(int)

#     group_cols = [session_col, side_col]
#     if split_opto:
#         group_cols.append(opto_col)
#     group_cols.append(rewarded_col)

#     summary = df.groupby(group_cols).size().unstack(fill_value=0)

#     bar_labels = []
#     session_ticks = []
#     session_labels = []

#     sessions = sorted(df[session_col].unique()) if sort_sessions else df[session_col].unique()
#     x = 0
#     label_order = ['left', 'right']

#     for session in sessions:
#         session_df = summary.loc[session] if session in summary.index.get_level_values(0) else None
#         if session_df is None or session_df.sum().sum() < min_trials:
#             continue

#         group_entries = []
#         x_positions = []

#         # Build expected group order for this session
#         expected_groups = []
#         for side in label_order:
#             if split_opto:
#                 expected_groups.append((side, 0))  # control
#                 expected_groups.append((side, 1))  # opto
#             else:
#                 expected_groups.append(side)

#         for group in expected_groups:
#             if split_opto:
#                 if group not in session_df.index:
#                     rewarded = 0
#                     punished = 0
#                 else:
#                     counts = session_df.loc[group]
#                     rewarded = counts.get(1, 0)
#                     punished = counts.get(0, 0)
#                 label = f"{group[0][0].upper()} {'Opto' if group[1] == 1 else ''}".strip()
#             else:
#                 if group not in session_df.index:
#                     rewarded = 0
#                     punished = 0
#                 else:
#                     counts = session_df.loc[group]
#                     rewarded = counts.get(1, 0)
#                     punished = counts.get(0, 0)
#                 label = group[0].upper()

#             total = rewarded + punished
#             if normalize and total > 0:
#                 rewarded /= total
#                 punished /= total

#             ax.bar(x, rewarded, color='green', label='Rewarded' if x == 0 else "", zorder=3)
#             ax.bar(x, punished, bottom=rewarded, color='red', label='Punished' if x == 0 else "", zorder=3)

#             if show_counts:
#                 height = rewarded + punished
#                 label_text = f"{int(total)}" if not normalize else f"{height * 100:.1f}%"
#                 ax.text(x, height + 0.02, label_text, ha='center', va='bottom', fontsize=8)

#             bar_labels.append(label)
#             x_positions.append(x)
#             x += 1

#         # Center session label under group
#         center_x = np.mean(x_positions)
#         session_ticks.append(center_x)
#         session_labels.append(session)

#     ax.set_xticks(session_ticks)
#     ax.set_xticklabels(session_labels, rotation=45, ha='right')
#     ax.set_ylabel("Trial Proportion" if normalize else "Trial Count")
#     ax.set_title("Rewarded vs Punished Trials by Session")
#     ax.legend(loc='upper right')
#     ax.grid(True, axis='y', alpha=0.3)
#     fig.tight_layout()

#     return ax



# def plot_trial_outcomes_by_session(
#     df,
#     split_opto=True,
#     min_trials=10,
#     sort_sessions=True,
#     show_counts=True,
#     normalize=True,
#     ax=None,
#     rewarded_col='mouse_correct',
#     side_col='trial_side',
#     opto_col='is_opto',
#     session_col='date'
# ):
#     """
#     Plot rewarded vs punished trial bars per session, grouped by trial side.
#     Optionally split control/opto and normalize to percentage.

#     Parameters:
#     - df : pd.DataFrame
#     - split_opto : bool, if True, split control vs opto
#     - min_trials : int, sessions with fewer trials are excluded
#     - sort_sessions : bool, sort sessions by date
#     - show_counts : bool, annotate bars with trial count or percent
#     - normalize : bool, plot as percentage instead of count
#     - ax : matplotlib axis (optional)
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(14, 6))
#     else:
#         fig = ax.figure

#     df = df.copy()
#     df[rewarded_col] = df[rewarded_col].astype(int)

#     group_cols = [session_col, side_col]
#     if split_opto:
#         group_cols.append(opto_col)
#     group_cols.append(rewarded_col)

# #     summary = df.groupby(group_cols).size().unstack(fill_value=0)

# #     bar_labels = []
# #     session_ticks = []
# #     session_labels = []

# #     sessions = sorted(df[session_col].unique()) if sort_sessions else df[session_col].unique()
# #     x = 0
# #     label_order = ['left', 'right']

# #     for session in sessions:
# #         session_df = summary.loc[session] if session in summary.index.get_level_values(0) else None
# #         if session_df is None or session_df.sum().sum() < min_trials:
# #             continue

#     summary = df.groupby(group_cols).size().unstack(fill_value=0)

#     session_ticks = []
#     session_labels = []

#     sessions = sorted(df[session_col].unique()) if sort_sessions else df[session_col].unique()
#     x = 0
#     label_order = ['left', 'right']

#     for session in sessions:
#         session_df = summary.loc[session] if session in summary.index.get_level_values(0) else None
#         if session_df is None or session_df.sum().sum() < min_trials:
#             continue

#         x_positions = []

#         # Build expected group order for this session
#         expected_groups = []
#         for side in label_order:
#             if split_opto:
#                 expected_groups.append((side, 0))  # control
#                 expected_groups.append((side, 1))  # opto
#             else:
#                 expected_groups.append(side)

#         for group in expected_groups:
#             if split_opto:
#                 if group not in session_df.index:
#                     rewarded = 0
#                     punished = 0
#                 else:
#                     counts = session_df.loc[group]
#                     rewarded = counts.get(1, 0)
#                     punished = counts.get(0, 0)
#                 label = f"{group[0][0].upper()}{'O' if group[1] == 1 else ''}"
#             else:
#                 if group not in session_df.index:
#                     rewarded = 0
#                     punished = 0
#                 else:
#                     counts = session_df.loc[group]
#                     rewarded = counts.get(1, 0)
#                     punished = counts.get(0, 0)
#                 label = group[0][0].upper()

#             total = rewarded + punished
#             if normalize and total > 0:
#                 rewarded /= total
#                 punished /= total

#             ax.bar(x, rewarded, color='green', label='Rewarded' if x == 0 else "", zorder=3)
#             ax.bar(x, punished, bottom=rewarded, color='red', label='Punished' if x == 0 else "", zorder=3)

#             if show_counts:
#                 height = rewarded + punished
#                 label_text = f"{int(total)}" if not normalize else f"{height * 100:.1f}%"
#                 ax.text(x, height + 0.02, label_text, ha='center', va='bottom', fontsize=8)

#             # Add L / LO / R / RO under each bar
#             ax.text(x, -0.05 if normalize else -3, label, ha='center', va='top', fontsize=9, rotation=0, clip_on=False)

#             x_positions.append(x)
#             x += 1

#         # Center session label under group
#         center_x = np.mean(x_positions)
#         session_ticks.append(center_x)
#         session_labels.append(session)

#     ax.set_xticks(session_ticks)
#     ax.set_xticklabels(session_labels, rotation=45, ha='right')
#     ax.set_ylabel("Trial Proportion" if normalize else "Trial Count")
#     ax.set_title("Rewarded vs Punished Trials by Session")
#     ax.legend(loc='upper right')
#     ax.grid(True, axis='y', alpha=0.3)
#     fig.tight_layout()

#     return ax
