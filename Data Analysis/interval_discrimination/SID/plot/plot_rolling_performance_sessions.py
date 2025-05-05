# -*- coding: utf-8 -*-
"""
Created on Sat May  3 21:23:57 2025

@author: timst
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_rolling_performance_across_sessions(
    df_concat,
    outcome_col='mouse_correct',
    session_col='date',
    subject_col='subject_name',
    window_size=50,
    kde_bw=0.10,  # as a fraction of trial range, kde_bw=0.05
    bin_size=50,
    ax=None,
    show_plot=True
):
    """
    Plot rolling, KDE, and binned average performance across all sessions.

    Parameters:
    - df_concat : pd.DataFrame with concatenated sessions
    - outcome_col : str, column indicating trial success (1/0)
    - session_col : str, session date column (datetime)
    - subject_col : str, subject name column
    - window_size : int, window size for rolling average
    - kde_bw : float, bandwidth as fraction of total trial range
    - bin_size : int, trial bin size for step-average overlay
    - ax : matplotlib axis (optional)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import gaussian_kde

    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 3.5))
    else:
        fig = ax.figure

    df = df_concat.copy()
    df = df.sort_values(by=[session_col]).reset_index(drop=True)
    df['trial_num'] = np.arange(len(df))
    x_vals = df['trial_num'].values
    y_vals = df[outcome_col].astype(float).values

    # Rolling mean (centered)
    rolling_series = pd.Series(y_vals, index=x_vals)
    rolling_perf = rolling_series.rolling(window_size, min_periods=1, center=True).mean()
    ax.plot(rolling_perf.index, rolling_perf.values, label=f"Rolling (n={window_size})", color='black', linewidth=2)

    # KDE
    if len(np.unique(x_vals)) > 1 and y_vals.sum() > 0:
        # kde = gaussian_kde(x_vals, weights=y_vals, bw_method=kde_bw)
        # kde_vals = kde(x_vals)
        # kde_vals = np.clip(kde_vals, 0, 1)
        # ax.plot(x_vals, kde_vals, label=f"KDE (bw={kde_bw})", color='blue', linestyle='--', linewidth=2)
        
        correct_x = x_vals[y_vals == 1]
        kde_correct = gaussian_kde(correct_x, bw_method=kde_bw)
        kde_all = gaussian_kde(x_vals, bw_method=kde_bw)
        p_correct = kde_correct(x_vals) / (kde_all(x_vals) + 1e-6)
        p_correct = np.clip(p_correct, 0, 1)        
        
        ax.plot(x_vals, p_correct, label=f"KDE P(Correct)", linestyle='--', color='blue')
    else:
        ax.plot([], [], label="KDE (invalid data)", color='blue', linestyle='--')

    # Step-binned average
    df['trial_bin'] = df['trial_num'] // bin_size
    bin_perf = df.groupby('trial_bin')[outcome_col].mean()
    bin_centers = bin_perf.index * bin_size + bin_size // 2
    ax.step(bin_centers, bin_perf.values, where='mid', label=f"Binned Avg (n={bin_size})", color='orange', alpha=0.7)

    # Session transitions
    session_starts = df.groupby(session_col)['trial_num'].min().values
    for start in session_starts:
        ax.axvline(start, color='gray', linestyle='--', alpha=0.3)

    for session_date, start in zip(df[session_col].unique(), session_starts):
        ax.text(start, 1.05, pd.to_datetime(session_date).strftime("%m-%d"), ha='left', va='bottom', fontsize=8, rotation=45)

    # Title and labels
    subject = df[subject_col].unique()[0]
    min_date = pd.to_datetime(df[session_col]).min().strftime("%Y-%m-%d")
    max_date = pd.to_datetime(df[session_col]).max().strftime("%Y-%m-%d")
    ax.set_title(f"{subject} | Sessions {min_date} to {max_date}", fontsize=10, y=1.1)

    ax.set_xlabel("Cumulative Trial Number")
    ax.set_ylabel("Performance (P(Correct))")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if show_plot:
        plt.show()
    
    
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject + '\\' + date)
    output_dir = os.path.join('plots\\' + subject)
    figure_id = f"{subject}_rolling_perf_curve"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)       

    return out_path






# def plot_rolling_performance_across_sessions(
#     df_concat,
#     outcome_col='mouse_correct',
#     session_col='date',
#     subject_col='subject_name',
#     window_size=50,
#     kde_bw=0.05,  # fraction of total trials
#     ax=None
# ):
#     """
#     Plot both rolling and KDE-smoothed performance across all sessions.

#     Parameters:
#     - df_concat : pd.DataFrame with concatenated sessions
#     - outcome_col : str, column indicating trial success (1/0)
#     - session_col : str, session date column (datetime)
#     - subject_col : str, subject name column
#     - window_size : int, window size for rolling average
#     - kde_bw : float, bandwidth as fraction of total trial range
#     - ax : matplotlib axis (optional)
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd
#     from scipy.stats import gaussian_kde

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(12, 5))
#     else:
#         fig = ax.figure

#     df = df_concat.copy()
#     df = df.sort_values(by=[session_col]).reset_index(drop=True)
#     df['trial_num'] = np.arange(len(df))

#     x_vals = df['trial_num'].values
#     y_vals = df[outcome_col].astype(float).values

#     # Rolling mean
#     rolling_perf = pd.Series(y_vals).rolling(window_size, min_periods=1, center=True).mean().values
#     ax.plot(x_vals, rolling_perf, label=f"Rolling (n={window_size})", color='black', linewidth=2)

#     # KDE
#     if len(np.unique(x_vals)) > 1 and y_vals.sum() > 0:
#         kde = gaussian_kde(x_vals, weights=y_vals, bw_method=kde_bw)
#         kde_vals = kde(x_vals)
#         kde_vals = np.clip(kde_vals, 0, 1)
#         ax.plot(x_vals, kde_vals, label=f"KDE (bw={kde_bw})", color='blue', linestyle='--', linewidth=2)
#     else:
#         ax.plot([], [], label="KDE (invalid data)", color='blue', linestyle='--')

#     # Session transition lines
#     session_starts = df.groupby(session_col)['trial_num'].min().values
#     for start in session_starts:
#         ax.axvline(start, color='gray', linestyle='--', alpha=0.3)

#     # Session labels
#     for session_date, start in zip(df[session_col].unique(), session_starts):
#         ax.text(start, 1.05, pd.to_datetime(session_date).strftime("%m-%d"), ha='left', va='bottom', fontsize=8, rotation=45)

#     # Title and formatting
#     subject = df[subject_col].unique()[0]
#     min_date = pd.to_datetime(df[session_col]).min().strftime("%Y-%m-%d")
#     max_date = pd.to_datetime(df[session_col]).max().strftime("%Y-%m-%d")
#     ax.set_title(f"{subject} | Sessions {min_date} to {max_date}", fontsize=13, y=1.04)

#     ax.set_xlabel("Cumulative Trial Number")
#     ax.set_ylabel("Performance (P(Correct))")
#     ax.set_ylim(-0.05, 1.05)
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#     fig.tight_layout()

#     return ax









# def plot_rolling_performance_across_sessions(
#     df_concat,
#     outcome_col='mouse_correct',
#     session_col='date',
#     subject_col='subject_name',
#     smoothing_method='rolling',  # 'rolling' or 'kde'
#     window_size=50,
#     kde_bw=200,
#     ax=None
# ):
#     """
#     Plot rolling or KDE-smoothed performance (e.g., reward rate) across all sessions.

#     Parameters:
#     - df_concat : pd.DataFrame with concatenated sessions
#     - outcome_col : str, column indicating trial success (1/0)
#     - session_col : str, column indicating session date (datetime)
#     - subject_col : str, subject name column
#     - smoothing_method : 'rolling' or 'kde'
#     - window_size : int, window for rolling average
#     - kde_bw : float, bandwidth for gaussian_kde
#     - ax : matplotlib axis (optional)
#     """
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(12, 5))
#     else:
#         fig = ax.figure

#     df = df_concat.copy()
#     df = df.sort_values(by=[session_col]).reset_index(drop=True)
#     df['trial_num'] = np.arange(len(df))

#     x_vals = df['trial_num'].values
#     y_vals = df[outcome_col].astype(float).values

#     # Apply smoothing
#     if smoothing_method == 'rolling':
#         perf_vals = pd.Series(y_vals).rolling(window_size, min_periods=1).mean().values
#     elif smoothing_method == 'kde':
#         kde = gaussian_kde(x_vals, weights=y_vals, bw_method=kde_bw / len(x_vals))
#         perf_vals = kde(x_vals)
#         perf_vals = np.clip(perf_vals, 0, 1)  # keep in [0, 1] for prob
#     else:
#         raise ValueError("smoothing_method must be 'rolling' or 'kde'")

#     # Plot smoothed performance
#     ax.plot(x_vals, perf_vals, label=f"{smoothing_method.title()} Performance", color='violet', linewidth=2)

#     # Session transition lines
#     session_starts = df.groupby(session_col)['trial_num'].min().values
#     for start in session_starts:
#         ax.axvline(start, color='gray', linestyle='--', alpha=0.3)

#     # Label sessions
#     for session_date, start in zip(df[session_col].unique(), session_starts):
#         ax.text(start, 1.05, pd.to_datetime(session_date).strftime("%m-%d"), ha='left', va='bottom', fontsize=8, rotation=45)

#     subject = df[subject_col].unique()[0]
#     min_date = pd.to_datetime(df[session_col]).min().strftime("%Y-%m-%d")
#     max_date = pd.to_datetime(df[session_col]).max().strftime("%Y-%m-%d")
#     ax.set_title(f"{subject} | Performance | Sessions {min_date} to {max_date}", fontsize=10, y=1.06)

#     ax.set_xlabel("Cumulative Trial Number")
#     ax.set_ylabel("Performance (P(Correct))")
#     ax.set_ylim(-0.05, 1.05)
#     ax.grid(True, alpha=0.3)
#     ax.legend()


#     fig.tight_layout()
#     return ax

