"""
main.py

The main orchestrator execution script specifically designed for PACE HPC cluster.
Handles multiprocessing, PyTorch thread control, data loading, and routing to 
the model fitting modules.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import multiprocessing
from tqdm import tqdm

# Import modular components
from Module.data_processing import prepare_session_data, transform_data_to_dataframe
from Module.model_fitting import fit_single_session_torch
from Module.results_export import save_comprehensive_results


def _parse_args():
    parser = argparse.ArgumentParser(description="PACE-ready entrypoint for Model_v20.")
    parser.add_argument(
        "--data-path",
        action="append",
        dest="data_paths",
        default=[],
        help="Path to one .mat file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--data-paths-file",
        default=os.environ.get("DATA_PATHS_FILE"),
        help="Text file containing one .mat path per line.",
    )
    parser.add_argument(
        "--save-path",
        default=os.environ.get("SAVE_PATH"),
        help="Directory where result CSVs will be written.",
    )
    return parser.parse_args()


def _load_configured_data_paths(args):
    data_paths = list(args.data_paths)

    env_paths = os.environ.get("DATA_PATHS", "").strip()
    if env_paths:
        data_paths.extend([p for p in env_paths.split(os.pathsep) if p.strip()])

    if args.data_paths_file:
        with open(args.data_paths_file, "r", encoding="utf-8") as handle:
            file_paths = [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
        data_paths.extend(file_paths)

    if data_paths:
        return data_paths

    # Legacy fallback preserved for parity with the notebook/script.
    return [
        # r'/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250814_121831.mat',
        # r"/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC06/MC06_SChR2_block_single_interval_discrimination_V_1_20250721_151425.mat",
        r"/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC06/MC06_SChR2_block_single_interval_discrimination_V_1_20250801_144807.mat",
    ]


if __name__ == '__main__':
    args = _parse_args()

    # --- HPC CONFIGURATION & THREAD CONTROL ---
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    
    # CRITICAL: Limit PyTorch and NumPy threads to prevent contention on HPC
    torch.set_num_threads(slurm_cpus)
    os.environ["OMP_NUM_THREADS"] = str(slurm_cpus)
    os.environ["MKL_NUM_THREADS"] = str(slurm_cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(slurm_cpus)

    print(f"--- HPC CONFIGURATION ---")
    print(f"Task ID: {task_id}")
    print(f"Allocated CPUs: {slurm_cpus}")
    print(f"-------------------------\n")

    all_data_paths = _load_configured_data_paths(args)
    save_path = args.save_path or r"/storage/home/hcoda1/1/ashamsnia6/r-fnajafi3-0/Projects/results"
    
    if task_id < len(all_data_paths):
        data_paths = [all_data_paths[task_id]]
        print(f"Processing Array Task {task_id}:")
        print(f"File: {data_paths[0]}")
    else:
        print(f"Task ID {task_id} is out of range (Total files: {len(all_data_paths)}).")
        sys.exit()
    
    print("Loading Data...")
    raw = prepare_session_data(data_paths)
    if not raw['outcomes']: 
        print("No outcomes loaded. Check paths.")
        sys.exit()
    
    sessions_df = [transform_data_to_dataframe(raw, i) for i in range(len(raw['outcomes']))]
    sessions_df = [df for df in sessions_df if not df.empty]
    print(f'Loaded {len(sessions_df)} sessions')

    # ALL 14 PARAMETERS bounds
    bounds = [
        (0.2, 1.2),      # 0: decay
        (1.0, 60.0),     # 1: noise
        (0.0, 0.2),      # 2: a_rew
        (0.0, 0.2),      # 3: a_pun
        (0.0, 1.0),      # 4: gamma
        (0.0, 2.5),      # 5: a_unc_ctx
        (0.0, 2.5),      # 6: a_unc_sens
        (0.0, 1.0),      # 7: a_ch_r
        (0.0, 1.0),      # 8: a_ch_p
        (0.1, 15.0),     # 9: beta
        (0.0, 0.3),      # 10: lapse
        (0.001, 0.25),   # 11: p_switch
        (0.01, 0.45),    # 12: p_rare
        (0.0, 0.2)       # 13: alpha_boundary
    ]

    print(f"Starting 3-Stage Fit with JIT Optimization ... ({slurm_cpus} cores)")
    
    fit_tasks = [(df, i, bounds, None) for i, df in enumerate(sessions_df)]
    with multiprocessing.Pool(processes=slurm_cpus) as pool:
        results = list(tqdm(pool.imap(fit_single_session_torch, fit_tasks), total=len(sessions_df), desc="Sessions"))
    
    sim_dfs_list = [r['sim_df'] for r in results]
    valid_params = [r['best_params'] for r in results if r['success']]
    fixed_biases = [r['fixed_bias'] for r in results]
    
    param_names_reduced = [
        'decay_rate', 'noise_param_a', 'alpha_reward', 'alpha_punish', 
        'gamma', 'alpha_unc_ctx', 'alpha_unc_sens',
        'alpha_ch_r', 'alpha_ch_p', 'beta', 'lapse',
        'p_switch', 'p_rare', 'alpha_boundary'
    ]

    if valid_params:
        mean_params = np.mean(valid_params, axis=0)
        print(f'\nMEAN PARAMETERS (3 Stages):')
        for n, v in zip(param_names_reduced, mean_params):
            print(f'   {n}: {v:.4f}')
        print(f'   Average Fixed Bias: {np.mean(fixed_biases):.4f} +- {np.std(fixed_biases):.4f}')
    
    if sessions_df:
        exp_all = pd.concat(sessions_df, keys=range(len(sessions_df)), names=['session_id', 'trial_in_session']).reset_index(drop=True)
        sim_all = pd.concat(sim_dfs_list, keys=range(len(sim_dfs_list)), names=['session_id', 'trial_in_session']).reset_index(drop=True)
        
        correct_map = {'short': 'left', 'long': 'right'}
        comparison = sim_all.copy()
        comparison['correct_mouse'] = exp_all.apply(
            lambda row: 1 if row['mouse_choice'] == correct_map[row['trial_type']] else 0, axis=1
        )
        comparison['correct_model'] = comparison['correct_model'].astype(int)
        comparison['block_type'] = exp_all['block_type']
        comparison['isi'] = exp_all['isi']

        mouse_acc = comparison['correct_mouse'].mean()
        model_acc = comparison['correct_model'].mean()
        print(f"\nACCURACY COMPARISON")
        print(f"   Mouse overall:  {mouse_acc:.3%}")
        print(f"   Model overall:  {model_acc:.3%}")
        print(f"   Difference:     {model_acc - mouse_acc:+.3%}")

        print(f"\nBY BLOCK TYPE:")
        for block in ['short_block', 'long_block', 'neutral']:
            subset = comparison[comparison['block_type'] == block]
            if len(subset) == 0: continue
            print(f"   {block:12}: Mouse {subset['correct_mouse'].mean():.3%} | Model {subset['correct_model'].mean():.3%}")

        print(f"\nBY ISI BIN:")
        isi_bins = pd.cut(comparison['isi'], bins=np.linspace(0, 2.5, 21))
        acc_by_bin = comparison.groupby(isi_bins, observed=False).agg(
            mouse_acc=('correct_mouse', 'mean'),
            model_acc=('correct_model', 'mean'),
            n=('correct_mouse', 'size')
        ).round(3)
        print(acc_by_bin)

        agreement = (comparison['mouse_choice'] == comparison['model_choice']).mean()
        print(f"\nTRIAL-BY-TRIAL CHOICE AGREEMENT: {agreement:.3%}")
        
        os.makedirs(save_path, exist_ok=True)
        
        # --- SAVE EVERYTHING ---
        save_comprehensive_results(results, prefix=f'Dynamic_Boundary_Fit_Task{task_id}', save_path=save_path)
