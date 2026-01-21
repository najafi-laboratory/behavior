import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

def plot_rare_trial_dynamics(sim_df, window=2, save_path=None):
    """
    Plot model dynamics (p_short, w_time, w_ctx, bias) around rare trials
    Args:
        sim_df (pd.DataFrame): DataFrame containing simulation results with multi-index (session_id, trial).
        window (int): Number of trials before and after the rare trial to include in the plot.
        save_path (str, optional): Path to save the plot. If None, the plot is not saved.
    Returns:
        None
    """
    df = sim_df.reset_index().copy()
    cols_to_plot = ['p_short_pre', 'w_time_pre', 'w_ctx_pre', 'bias_pre']
    labels = ['Belief p(Short)', 'Norm. Weight (Time)', 'Norm. Weight (Context)', 'Choice Bias']
    
    is_short_blk = df['block_type'] == 'short_block'
    is_long_blk  = df['block_type'] == 'long_block'
    is_short_tr  = df['trial_type'] == 'short'
    is_long_tr   = df['trial_type'] == 'long'
    
    idx_rare_long = df[is_short_blk & is_long_tr].index
    idx_rare_short = df[is_long_blk & is_short_tr].index
    
    data_collector = {col: {'rare_long': [], 'rare_short': []} for col in cols_to_plot}

    def normalize_trace(trace):
        t_min, t_max = np.min(trace), np.max(trace)
        if t_max - t_min < 1e-9: return np.zeros_like(trace)
        return (trace - t_min) / (t_max - t_min)

    def collect_windows(indices, key_type):
        for idx in indices:
            if idx - window < 0 or idx + window + 1 > len(df): continue
            slice_df = df.iloc[idx - window : idx + window + 1]
            if slice_df['session_id'].nunique() == 1:
                for col in cols_to_plot:
                    vals = slice_df[col].values
                    if col in ['w_time_pre', 'w_ctx_pre']:
                        vals = normalize_trace(vals)
                    data_collector[col][key_type].append(vals)

    collect_windows(idx_rare_long, 'rare_long')
    collect_windows(idx_rare_short, 'rare_short')

    if not data_collector['p_short_pre']['rare_long'] and not data_collector['p_short_pre']['rare_short']:
        print("No rare trials found suitable for plotting.")
        return

    x = np.arange(-window, window + 1)
    fig, axs = plt.subplots(len(cols_to_plot), 2, figsize=(12, 3 * len(cols_to_plot)), sharex=True)
    fig.suptitle(f'Model Dynamics Around Rare Trials (Weights Normalized, Win +/-{window})', fontsize=16)
    
    warnings.simplefilter("ignore", category=RuntimeWarning)

    for i, (col, label) in enumerate(zip(cols_to_plot, labels)):
        ax1 = axs[i, 0]
        vals = np.array(data_collector[col]['rare_long'])
        if len(vals) > 0:
            mean_v = np.nanmean(vals, axis=0)
            sem_v = np.nanstd(vals, axis=0) / np.sqrt(np.sum(~np.isnan(vals), axis=0))
            ax1.plot(x, mean_v, color='purple', lw=2, marker='o', markersize=4)
            ax1.fill_between(x, mean_v - sem_v, mean_v + sem_v, color='purple', alpha=0.2)
        
        ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel(label, fontsize=11)
        if i == 0: ax1.set_title(f'Rare LONG in Short Block', fontsize=12, color='purple')

        ax2 = axs[i, 1]
        vals = np.array(data_collector[col]['rare_short'])
        if len(vals) > 0:
            mean_v = np.nanmean(vals, axis=0)
            sem_v = np.nanstd(vals, axis=0) / np.sqrt(np.sum(~np.isnan(vals), axis=0))
            ax2.plot(x, mean_v, color='orange', lw=2, marker='o', markersize=4)
            ax2.fill_between(x, mean_v - sem_v, mean_v + sem_v, color='orange', alpha=0.2)
            
        ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
        if i == 0: ax2.set_title(f'Rare SHORT in Long Block', fontsize=12, color='orange')

        for ax in [ax1, ax2]:
            ax.set_xticks(x)
            ax.grid(True, ls=':', alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if col in ['w_time_pre', 'w_ctx_pre']:
                ax.set_ylim(-0.05, 1.05)
            if i == len(cols_to_plot) - 1:
                ax.set_xlabel('Trials relative to Rare Event')

    warnings.simplefilter("default", category=RuntimeWarning)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        saving = os.path.join(save_path, "rare_trial_dynamics.pdf")
        plt.savefig(saving, dpi=300)

    plt.close()