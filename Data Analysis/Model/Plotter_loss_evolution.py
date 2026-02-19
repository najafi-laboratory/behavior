import os
import numpy as np
import matplotlib.pyplot as plt
OPTIM_STEPS_STAGE_1 = 2000
OPTIM_STEPS_STAGE_2 = 2000
OPTIM_STEPS_STAGE_3 = 2000

def plot_loss_evolution(results, save_path=None):
    """
    Plots optimization loss separated by stage for each session.
    Creates a (N_sessions x 3) grid.
    """
    valid_results = [r for r in results if r['success']]
    if not valid_results:
        print("No valid results to plot loss.")
        return

    n_sessions = len(valid_results)
    
    # Create a grid: Rows = Sessions, Cols = 3 Stages
    # Adjust height: 3 inches per session
    fig, axs = plt.subplots(n_sessions, 3, figsize=(15, 3 * n_sessions), constrained_layout=True)
    
    # If only 1 session, axs is 1D array. Make it 2D (1, 3) for consistency.
    if n_sessions == 1:
        axs = np.expand_dims(axs, axis=0)

    # Define indices for slicing
    idx_s1_end = OPTIM_STEPS_STAGE_1
    idx_s2_end = OPTIM_STEPS_STAGE_1 + OPTIM_STEPS_STAGE_2
    
    for i, res in enumerate(valid_results):
        loss = np.array(res['loss_history'])
        sess_id = res['session_id']
        
        # Slice the data
        loss_s1 = loss[0 : idx_s1_end]
        loss_s2 = loss[idx_s1_end : idx_s2_end]
        loss_s3 = loss[idx_s2_end :]
        
        # --- Plot Stage 1: Sensory ---
        ax1 = axs[i, 0]
        ax1.plot(range(len(loss_s1)), loss_s1, color='blue')
        ax1.set_title(f'Sess {sess_id}: Stage 1 (Sensory)')
        ax1.set_ylabel('NLL')
        ax1.grid(True, alpha=0.3)
        
        # --- Plot Stage 2: Strategy ---
        ax2 = axs[i, 1]
        # X-axis continues from where S1 left off for clarity, or starts at 0. 
        # Let's start at 0 for "Step in Stage"
        ax2.plot(range(len(loss_s2)), loss_s2, color='green')
        ax2.set_title(f'Sess {sess_id}: Stage 2 (Strategy)')
        # Auto-scale Y to see small changes
        ax2.autoscale(enable=True, axis='y', tight=False)
        ax2.grid(True, alpha=0.3)

        # --- Plot Stage 3: Fine Tune ---
        ax3 = axs[i, 2]
        ax3.plot(range(len(loss_s3)), loss_s3, color='black')
        ax3.set_title(f'Sess {sess_id}: Stage 3 (FineTune)')
        ax3.autoscale(enable=True, axis='y', tight=False)
        ax3.grid(True, alpha=0.3)

    plt.suptitle('Optimization Progress per Stage (Independent Y-Scales)', fontsize=16)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        saving = os.path.join(save_path, "optimization_loss_separated.pdf")
        plt.savefig(saving, dpi=300)

    plt.close()
