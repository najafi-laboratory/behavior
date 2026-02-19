import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

def plot_performance_around_rare_trials(exp_df, sim_df, window=3, save_path=None):
    """
    Plot performance (accuracy) of both mouse and model around rare trials for multiple sessions.
    Args:
        exp_df (pd.DataFrame): DataFrame containing experimental session data with multi-index (session_id, trial).
        sim_df (pd.DataFrame): DataFrame containing simulation results with multi-index (session_id, trial).
        window (int): Number of trials before and after the rare trial to include in the plot.
        save_path (str, optional): Path to save the plot. If None, the plot is not saved.
    Returns:
        None
    """
    combined = exp_df.reset_index().set_index(['session_id', 'trial_in_session']).copy()
    sim_aligned = sim_df.reset_index().set_index(['session_id', 'trial_in_session'])
    
    correct_map = {'short': 'left', 'long': 'right'}
    combined['correct_mouse'] = combined.apply(
        lambda row: 1 if row['mouse_choice'] == correct_map[row['trial_type']] else 0, axis=1
    )
    combined['correct_model'] = sim_aligned['correct_model'].astype(int)
    
    is_short_blk = combined['block_type'] == 'short_block'
    is_long_blk  = combined['block_type'] == 'long_block'
    is_short_tr  = combined['trial_type'] == 'short'
    is_long_tr   = combined['trial_type'] == 'long'
    
    idx_rare_long = combined[is_short_blk & is_long_tr].index
    idx_rare_short = combined[is_long_blk & is_short_tr].index
    
    combined_flat = combined.reset_index()
    
    def get_event_traces(indices):
        mouse_traces = []
        model_traces = []
        for idx_sess, idx_trial in indices:
            matches = combined_flat.index[(combined_flat['session_id'] == idx_sess) & 
                                          (combined_flat['trial_in_session'] == idx_trial)]
            if len(matches) == 0: continue
            loc = matches[0]
            if loc - window < 0 or loc + window + 1 > len(combined_flat): continue
            slice_df = combined_flat.iloc[loc - window : loc + window + 1]
            if slice_df['session_id'].nunique() == 1:
                mouse_traces.append(slice_df['correct_mouse'].values)
                model_traces.append(slice_df['correct_model'].values)
        return mouse_traces, model_traces

    rl_mouse, rl_model = get_event_traces(idx_rare_long)
    rs_mouse, rs_model = get_event_traces(idx_rare_short)

    if not rl_mouse and not rs_mouse:
        print("Not enough rare trials for performance plot.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f'Performance Around Rare Trials (Window +/-{window})', fontsize=16)
    x = np.arange(-window, window + 1)

    def plot_trace(ax, data_list, color, label):
        if not data_list: return
        arr = np.array(data_list)
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        ax.plot(x, mean, color=color, lw=1.5, marker='o', markersize=4, label=label)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15)

    plot_trace(axs[0], rl_mouse, 'black', 'Mouse')
    plot_trace(axs[0], rl_model, 'red', 'Model')
    axs[0].set_title(f'Rare LONG (in Short Block)\n(n={len(rl_mouse)})')
    axs[0].set_ylabel('Proportion Correct')

    plot_trace(axs[1], rs_mouse, 'black', 'Mouse')
    plot_trace(axs[1], rs_model, 'red', 'Model')
    axs[1].set_title(f'Rare SHORT (in Long Block)\n(n={len(rs_mouse)})')

    for ax in axs:
        ax.set_xticks(x)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.7, label='Rare Trial')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Trials relative to Rare Event')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, ls=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        saving = os.path.join(save_path, "performance_around_rare_trials.pdf")
        plt.savefig(saving, dpi=300)

    plt.close()