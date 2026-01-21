import os
import pandas as pd
import numpy as np
import torch

def save_comprehensive_results(results, prefix='Model_Fit', save_path=None):
    """
    Save comprehensive results from model fitting and simulation to CSV files.
    Args:
        results (list of dict): Each dict contains results for a session, including:
            - 'session_id': Identifier for the session
            - 'best_params': List or array of best fitted parameters
            - 'fixed_bias': Fixed bias parameter value
            - 'nll': Final negative log-likelihood value
            - 'sim_df': DataFrame with detailed simulation results per trial
            - 'loss_history': (optional) List of loss values over optimization steps
        prefix (str): Prefix for the saved CSV files.

    Saves two files:
    1. prefix_Trials_and_Params.csv: Contains ALL trial data merged with best fitted parameters.
    2. prefix_Loss_History.csv: Contains the step-by-step loss evolution for every session.
    """
    if not results:
        print("No results to save.")
        return

    # --- 1. Define Parameter Names ---
    # Must match the order in fit_single_session_torch
    param_names = [
        'decay_rate', 'noise_param_a', 
        'alpha_reward', 'alpha_punish', 'alpha_unc',
        'alpha_ch_r', 'alpha_ch_p', 
        'beta', 'lapse',
        'p_switch', 'p_rare', 
        'alpha_boundary' 
    ]

    all_sessions_dfs = []
    loss_data = {}

    print(f"\nCompiling data for export...")
    
    for res in results:
        if not res['success']: 
            continue
            
        sess_id = res['session_id']
        params = res['best_params']
        fixed_bias = res['fixed_bias']
        if torch.is_tensor(fixed_bias): fixed_bias = fixed_bias.item()
        
        final_nll = res['nll']
        sim_df = res['sim_df'].copy()
        
        # --- A. Attach Session Metadata & Parameters to DataFrame ---
        # This repeats the parameter values for every trial in this session,
        # which makes the CSV self-contained for analysis.
        sim_df.insert(0, 'session_id', sess_id)
        sim_df['final_nll'] = final_nll
        sim_df['fixed_bias_param'] = fixed_bias
        
        for i, p_name in enumerate(param_names):
            sim_df[p_name] = params[i]
            
        all_sessions_dfs.append(sim_df)
        
        # --- B. Collect Loss History ---
        # We assume loss_history exists from our previous update
        if 'loss_history' in res:
            loss_data[f'Session_{sess_id}'] = res['loss_history']

    # --- SAVE FILE 1: TRIAL DATA + PARAMS ---
    if all_sessions_dfs:
        full_df = pd.concat(all_sessions_dfs, ignore_index=True)
        
        # Rename columns to be nice and readable if needed
        # (sim_df already has: isi, trial_type, block_type, mouse_choice, etc.)
        
        if save_path:
            prefix = os.path.join(save_path, prefix)

        csv_name = f"{prefix}_Trials_and_Params.csv"
        full_df.to_csv(csv_name, index=False)
        print(f"✔ Saved comprehensive trial data to: {csv_name}")
        print(f"  (Shape: {full_df.shape})")
    # --- SAVE FILE 2: LOSS HISTORY ---
    if loss_data:
        # Pad lists with NaN if they have different lengths (though they shouldn't)
        max_len = max(len(v) for v in loss_data.values())
        padded_data = {k: v + [np.nan]*(max_len - len(v)) for k, v in loss_data.items()}
        
        loss_df = pd.DataFrame(padded_data)
        loss_df.index.name = 'Step'

        if save_path:
            prefix = os.path.join(save_path, prefix)
        
        loss_name = f"{prefix}_Loss_History.csv"
        loss_df.to_csv(loss_name, index=True)
        print(f"✔ Saved optimization history to: {loss_name}")
        print(f"  (Shape: {loss_df.shape})")