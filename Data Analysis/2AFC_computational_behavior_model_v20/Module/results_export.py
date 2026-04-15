"""
Module/results_export.py

Formats and exports the complex model fit and trace results into clean CSV 
files tailored for PACE HPC cluster storage.
"""

import os
import torch
import numpy as np
import pandas as pd

def save_comprehensive_results(results, prefix='Model_Fit', save_path=None):
    """
    Saves trial-by-trial parameters and loss history dataframes to CSV.
    """
    if not results:
        print("No results to save.")
        return

    param_names = [
        'decay_rate', 'noise_param_a', 
        'alpha_reward', 'alpha_punish', 'gamma', 'alpha_unc_ctx', 'alpha_unc_sens',
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
        
        sim_df.insert(0, 'session_id', sess_id)
        sim_df['final_nll'] = final_nll
        sim_df['fixed_bias_param'] = fixed_bias
        sim_df['initial_boundary_param'] = 1.25
        sim_df['p_common_param'] = 1.0 - params[12]
        
        for i, p_name in enumerate(param_names):
            sim_df[p_name] = params[i]
            
        all_sessions_dfs.append(sim_df)
        
        if 'loss_history' in res:
            loss_data[f'Session_{sess_id}'] = res['loss_history']

    if all_sessions_dfs:
        full_df = pd.concat(all_sessions_dfs, ignore_index=True)
        
        if save_path:
            prefix = os.path.join(save_path, prefix)

        csv_name = f"{prefix}_Trials_and_Params.csv"
        full_df.to_csv(csv_name, index=False)
        print(f"✔ Saved comprehensive trial data to: {csv_name}")
        print(f"  (Shape: {full_df.shape})")

    if loss_data:
        max_len = max(len(v) for v in loss_data.values())
        padded_data = {k: v + [np.nan]*(max_len - len(v)) for k, v in loss_data.items()}
        
        loss_df = pd.DataFrame(padded_data)
        loss_df.index.name = 'Step'
        
        if save_path:
            # Overwrite prefix in case path is used
            prefix = os.path.join(save_path, os.path.basename(prefix))
            
        loss_name = f"{prefix}_Loss_History.csv"
        loss_df.to_csv(loss_name)
        print(f"✔ Saved optimization history to: {loss_name}")
