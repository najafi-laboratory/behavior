import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

def plot_weight_trajectories(sim_df, save_path=None):
    """
    Plot the model's weight trajectories (w_time and w_ctx) across trials for multiple sessions
    Args:
        sim_df (pd.DataFrame): DataFrame containing simulation results with multi-index (session_id, trial).
        save_path (str, optional): Path to save the plot. If None, the plot is not saved.   
    Returns:
        None
    """
    warnings.simplefilter("ignore", category=RuntimeWarning)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Weight Trajectories', fontsize=16)

    # Individual w_time
    ax = axs[0, 0]
    for session_id in sim_df.index.get_level_values('session_id').unique():
        ax.plot(sim_df.loc[session_id]['w_time_pre'].values, color='gray', alpha=0.2)
    ax.set_title('Individual Session w_time')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('w_time')

    # Average w_time
    ax = axs[0, 1]
    temp_df = sim_df.reset_index()
    w_time_df = temp_df.pivot(index='trial_in_session', columns='session_id', values='w_time_pre')
    
    mean_w_time = w_time_df.apply(pd.to_numeric, errors='coerce').mean(axis=1)
    sem_w_time = w_time_df.apply(pd.to_numeric, errors='coerce').sem(axis=1)
    ax.plot(mean_w_time.index, mean_w_time, color='blue', lw=2, label='Mean w_time')
    ax.fill_between(mean_w_time.index, mean_w_time - sem_w_time, mean_w_time + sem_w_time,
                    color='blue', alpha=0.2, label='SEM')
    ax.set_title('Average w_time Across Sessions')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('w_time')
    ax.legend()

    # Individual w_ctx
    ax = axs[1, 0]
    for session_id in sim_df.index.get_level_values('session_id').unique():
        ax.plot(sim_df.loc[session_id]['w_ctx_pre'].values, color='gray', alpha=0.2)
    ax.set_title('Individual Session w_ctx')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('w_ctx')

    # Average w_ctx
    ax = axs[1, 1]
    w_ctx_df = temp_df.pivot(index='trial_in_session', columns='session_id', values='w_ctx_pre')
    
    mean_w_ctx = w_ctx_df.apply(pd.to_numeric, errors='coerce').mean(axis=1)
    sem_w_ctx = w_ctx_df.apply(pd.to_numeric, errors='coerce').sem(axis=1)
    ax.plot(mean_w_ctx.index, mean_w_ctx, color='green', lw=2, label='Mean w_ctx')
    ax.fill_between(mean_w_ctx.index, mean_w_ctx - sem_w_ctx, mean_w_ctx + sem_w_ctx,
                    color='green', alpha=0.2, label='SEM')
    ax.set_title('Average w_ctx Across Sessions')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('w_ctx')
    ax.legend()
    
    for ax in axs.flat:
        ax.grid(True, ls='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    warnings.simplefilter("default", category=RuntimeWarning)


    if save_path:
        os.makedirs(save_path, exist_ok=True)
        saving = os.path.join(save_path, "weight_trajectories.pdf")
        plt.savefig(saving, dpi=300)

    plt.close()