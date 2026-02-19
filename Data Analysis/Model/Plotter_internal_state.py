import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_internal_states(sim_df, save_path=None):
    """
    Plot the model's internal belief about block type (p_short_block) across trials for multiple sessions.
    Args:
        sim_df (pd.DataFrame): DataFrame containing simulation results with multi-index (session_id, trial).
        save_path (str, optional): Path to save the plot. If None, the plot is not saved.   
        returns:    
        None
    """
    sessions_to_plot = sim_df.index.get_level_values('session_id').unique()[:4]
    
    if len(sessions_to_plot) == 0:
        print("Cannot plot internal states: No session data found.")
        return

    fig, axs = plt.subplots(len(sessions_to_plot), 1, 
                            figsize=(15, 3 * len(sessions_to_plot)), 
                            sharex=True, squeeze=False)
    fig.suptitle('Model Internal Belief (p_short_block) vs. Actual Block', fontsize=16)
    
    for i, session_id in enumerate(sessions_to_plot):
        ax = axs[i, 0]
        s_df = sim_df.loc[session_id].reset_index(drop=True)
        
        ax.plot(s_df.index, s_df['p_short_pre'], label='Model p(Short)', color='blue', lw=2)
        ax.set_ylabel('p(Short)')
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7)

        block_map = {'short_block': 1, 'long_block': 0, 'neutral': 0.5}
        actual_blocks = s_df['block_type'].map(block_map)
        ax.fill_between(s_df.index, 0, 1, where=(actual_blocks == 1), 
                        color='red', alpha=0.2, label='Actual Short Block')
        ax.fill_between(s_df.index, 0, 1, where=(actual_blocks == 0), 
                        color='green', alpha=0.2, label='Actual Long Block')
        
        ax.set_title(f'Session {session_id}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, ls='--', alpha=0.5)

    ax.set_xlabel('Trial Number')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        saving = os.path.join(save_path, "internal_states.pdf")
        plt.savefig(saving, dpi=300)

    plt.close()