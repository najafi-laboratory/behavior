import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

def plot_all_transition_dynamics(sim_df, exp_df, window=20, save_path=None):
    """
    Plot model dynamics (p_short, w_time, w_ctx, bias) around block transitions for multiple sessions.
    Args:
        sim_df (pd.DataFrame): DataFrame containing simulation results with multi-index (session_id, trial).
        exp_df (pd.DataFrame): DataFrame containing experimental session data with multi-index (session_id, trial).
        window (int): Number of trials before and after the transition to include in the plot.
        save_path (str, optional): Path to save the plot. If None, the plot is not saved.
    Returns:
        None
    """
    combined = exp_df.reset_index().set_index(['session_id', 'trial_in_session']).copy()
    cols_to_plot = ['p_short_pre', 'w_time_pre', 'w_ctx_pre', 'bias_pre']
    labels = ['Belief p(Short)', 'Norm. Weight (Time)', 'Norm. Weight (Context)', 'Choice Bias']
    
    sim_aligned = sim_df.reset_index().set_index(['session_id', 'trial_in_session'])[cols_to_plot]
    combined = pd.concat([combined, sim_aligned], axis=1)

    combined['block_numeric'] = combined['block_type'].map({'short_block': 1, 'long_block': 0, 'neutral': np.nan})
    combined['block_diff'] = combined.groupby('session_id')['block_numeric'].diff()
    
    s2l_transitions = combined[combined['block_diff'] == -1].index
    l2s_transitions = combined[combined['block_diff'] == 1].index
    
    data_collector = {col: {'s2l': [], 'l2s': []} for col in cols_to_plot}
    combined_flat = combined.reset_index()

    def normalize_trace(trace):
        t_min, t_max = np.min(trace), np.max(trace)
        if t_max - t_min < 1e-9: return np.zeros_like(trace) 
        return (trace - t_min) / (t_max - t_min)

    def extract_slices(transition_indices, key_type):
        for idx_session, idx_trial in transition_indices:
            matches = combined_flat.index[(combined_flat['session_id'] == idx_session) & 
                                          (combined_flat['trial_in_session'] == idx_trial)]
            if len(matches) == 0: continue
            iloc_pos = matches[0]
            if iloc_pos - window < 0 or iloc_pos + window + 1 > len(combined_flat): continue
            slice_df = combined_flat.iloc[iloc_pos - window : iloc_pos + window + 1]
            if slice_df['session_id'].nunique() == 1 and len(slice_df) == 2 * window + 1:
                for col in cols_to_plot:
                    vals = slice_df[col].values
                    if col in ['w_time_pre', 'w_ctx_pre']:
                        vals = normalize_trace(vals)
                    data_collector[col][key_type].append(vals)

    extract_slices(s2l_transitions, 's2l')
    extract_slices(l2s_transitions, 'l2s')
    
    if not data_collector['p_short_pre']['s2l']:
        print("Not enough transitions found to plot dynamics.")
        return

    x = np.arange(-window, window + 1)
    fig, axs = plt.subplots(len(cols_to_plot), 2, figsize=(14, 3.5 * len(cols_to_plot)), sharex=True)
    fig.suptitle('Model Dynamics Around Block Transitions (Weights Normalized per Window)', fontsize=18)
    
    warnings.simplefilter("ignore", category=RuntimeWarning)

    for i, (col, label) in enumerate(zip(cols_to_plot, labels)):
        ax_left = axs[i, 0]
        arr_s2l = np.array(data_collector[col]['s2l'])
        mean_s2l = np.nanmean(arr_s2l, axis=0)
        sem_s2l = np.nanstd(arr_s2l, axis=0) / np.sqrt(np.sum(~np.isnan(arr_s2l), axis=0))
        
        ax_left.plot(x, mean_s2l, color='green', lw=2)
        ax_left.fill_between(x, mean_s2l - sem_s2l, mean_s2l + sem_s2l, color='green', alpha=0.2)
        ax_left.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax_left.set_ylabel(label, fontsize=12)
        if i == 0: ax_left.set_title(f'Short -> Long (n={len(arr_s2l)})', fontsize=14)
        
        ax_right = axs[i, 1]
        arr_l2s = np.array(data_collector[col]['l2s'])
        mean_l2s = np.nanmean(arr_l2s, axis=0)
        sem_l2s = np.nanstd(arr_l2s, axis=0) / np.sqrt(np.sum(~np.isnan(arr_l2s), axis=0))
        
        ax_right.plot(x, mean_l2s, color='red', lw=2)
        ax_right.fill_between(x, mean_l2s - sem_l2s, mean_l2s + sem_l2s, color='red', alpha=0.2)
        ax_right.axvline(0, color='black', linestyle='--', alpha=0.5)
        if i == 0: ax_right.set_title(f'Long -> Short (n={len(arr_l2s)})', fontsize=14)

        for ax in [ax_left, ax_right]:
            ax.grid(True, ls=':', alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if col in ['w_time_pre', 'w_ctx_pre']:
                ax.set_ylim(-0.05, 1.05) 
            if i == len(cols_to_plot) - 1:
                ax.set_xlabel('Trials from Switch', fontsize=12)

    warnings.simplefilter("default", category=RuntimeWarning)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        saving = os.path.join(save_path, "transition_dynamics.pdf")
        plt.savefig(saving, dpi=300)

    plt.close()