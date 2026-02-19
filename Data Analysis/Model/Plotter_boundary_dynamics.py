import os
import matplotlib.pyplot as plt

def plot_boundary_dynamics(sim_df, save_path=None):
    """
    Plot the evolution of the decision boundary over trials for multiple sessions.
    Args:
        sim_df (pd.DataFrame): DataFrame containing simulation results with multi-index (session_id, trial).
        save_path (str, optional): Path to save the plot. If None, the plot is not saved.
    Returns:
        None
    """
    sessions_to_plot = sim_df.index.get_level_values('session_id').unique()[:4]
    if len(sessions_to_plot) == 0: return

    fig, axs = plt.subplots(len(sessions_to_plot), 1, figsize=(15, 3 * len(sessions_to_plot)), sharex=True)
    fig.suptitle('Dynamic Decision Boundary Evolution', fontsize=16)

    # Handle single session case to ensure axs is iterable
    if len(sessions_to_plot) == 1:
        axs = [axs]

    for i, session_id in enumerate(sessions_to_plot):
        ax = axs[i]
        s_df = sim_df.loc[session_id].reset_index(drop=True)
        
        # 1. Plot Boundary Line
        ax.plot(s_df.index, s_df['boundary_pre'], color='purple', lw=2, label='Boundary')
        ax.axhline(1.25, color='gray', linestyle=':', label='Start (1.25s)')
        
        # 2. Add Block Background Shading
        # We define a range that covers the possible boundary values (0.5 to 2.5)
        y_min, y_max = 0.0, 3.0 
        
        is_short = (s_df['block_type'] == 'short_block')
        is_long = (s_df['block_type'] == 'long_block')
        
        ax.fill_between(s_df.index, y_min, y_max, where=is_short, 
                        color='red', alpha=0.1, label='Short Block')
        ax.fill_between(s_df.index, y_min, y_max, where=is_long, 
                        color='green', alpha=0.1, label='Long Block')
        
        # 3. Highlight Errors
        errors = s_df[~s_df['correct_model']]
        ax.scatter(errors.index, errors['boundary_pre'], color='red', s=10, alpha=0.6, label='Error', zorder=5)

        # 4. Formatting
        ax.set_ylabel('Boundary (s)')
        ax.set_title(f'Session {session_id}')
        ax.set_ylim(0.5, 2.0) # Focus on the active range
        ax.grid(True, alpha=0.3)
        
        # Clean Legend (avoid duplicates)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        saving = os.path.join(save_path, "boundary_dynamics.pdf")
        plt.savefig(saving, dpi=300)

    plt.close()