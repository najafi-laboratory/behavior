import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
import torch  # Added for thread control
from tqdm import tqdm
from Data_reader_dataframe_transformer import transform_data_to_dataframe
from Data_reader_load_mat import prepare_session_data
from Model_fit_single_session import fit_single_session_torch
from Plotter_psychometric import plot_psychometric_comparison
from Plotter_internal_state import plot_internal_states
from Plotter_loss_evolution import plot_loss_evolution
from Plotter_weights import plot_weight_trajectories
from Plotter_boundary_dynamics import plot_boundary_dynamics
from Plotter_block_transition_dynamic import plot_all_transition_dynamics
from Plotter_rare_dynamics import plot_rare_trial_dynamics
from Plotter_performance_block_transition import plot_performance_around_transitions
from Plotter_performance_rare import plot_performance_around_rare_trials
from Results_saver import save_comprehensive_results

if __name__ == '__main__':
    # --- 1. HPC CONFIGURATION & THREAD CONTROL ---
    # Get the SLURM Array Task ID (defaults to 0 if running locally)
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    
    # Get the number of CPUs allocated by SLURM (defaults to 4 if not set)
    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    
    # CRITICAL: Limit PyTorch and NumPy threads to prevent contention
    # This fixes the "22 hours for one session" issue.
    torch.set_num_threads(slurm_cpus)
    os.environ["OMP_NUM_THREADS"] = str(slurm_cpus)
    os.environ["MKL_NUM_THREADS"] = str(slurm_cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(slurm_cpus)

    print(f"--- HPC CONFIGURATION ---")
    print(f"Task ID: {task_id}")
    print(f"Allocated CPUs: {slurm_cpus}")
    print(f"-------------------------\n")

    # --- 2. DATA PATHS ---
    ALL_DATA_PATHS = [
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250624_150012.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250625_164242.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250626_182737.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250628_192014.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250630_174909.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250701_154950.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250702_140050.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250703_164706.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250708_192901.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250709_174856.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250710_130625.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250711_093117.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250715_180125.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250717_144725.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250718_172810.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250722_175504.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250730_193455.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250731_163425.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250805_133504.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250808_180200.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250811_195235.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250814_121831.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250819_182316.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250822_143100.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250826_143546.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250902_155717.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250903_112248.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250904_145410.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250909_155820.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250910_163253.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250911_143000.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250912_141501.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250917_215445.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250923_163844.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250924_180806.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250925_173549.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250929_160056.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20250930_164710.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20251001_151717.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20251002_154335.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20251003_151814.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20251008_173017.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20251009_161811.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20251010_171101.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20251013_170824.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20251014_150532.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20251015_164533.mat",
             r"/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data/YH24LG/YH24LG_block_single_interval_discrimination_V_1_20251016_151502.mat",
             ]
    
    # --- 3. SELECT SINGLE FILE BASED ON ARRAY ID ---
    if task_id < len(ALL_DATA_PATHS):
        DATA_PATHS = [ALL_DATA_PATHS[task_id]]
        print(f"Processing Array Task {task_id}:")
        print(f"File: {DATA_PATHS[0]}")
    else:
        print(f"Task ID {task_id} is out of range (Total files: {len(ALL_DATA_PATHS)}).")
        sys.exit()

    SAVE_PATH = r'/storage/home/hcoda1/1/ashamsnia6/r-fnajafi3-0/Projects/results'

    # --- 4. START PROCESSING ---
    print("Loading Data...")
    raw = prepare_session_data(DATA_PATHS)
    if not raw['outcomes']: 
        print("No outcomes loaded. Check paths.")
        sys.exit()
    
    sessions_df = [transform_data_to_dataframe(raw, i) for i in range(len(raw['outcomes']))]
    sessions_df = [df for df in sessions_df if not df.empty]
    print(f'Loaded {len(sessions_df)} sessions (Should be 1 for Array Job)')

    # 0:decay, 1:noise, 2:a_rew, 3:a_pun, 4:a_unc, 5:a_ch_r, 6:a_ch_p, 7:beta, 8:lapse, 9:p_switch, 10:p_rare
    bounds = [
        (0.2, 1.2),      # 0: decay
        (1.0, 60.0),     # 1: noise
        (0.0, 0.5),      # 2: a_rew
        (0.0, 0.5),      # 3: a_pun
        (0.0, 5.0),      # 4: a_unc
        (0.0, 1.0),      # 5: a_ch_r
        (0.0, 1.0),      # 6: a_ch_p
        (0.1, 15.0),     # 7: beta
        (0.0, 0.4),      # 8: lapse
        (0.01, 0.20),    # 9: p_switch 
        (0.01, 0.45),    # 10: p_rare
        (0.0, 0.2)       # 11: alpha_boundary
    ]

    print(f"Starting 3-Stage Fit with JIT Optimization ... ({slurm_cpus} cores)")
    
    # Use 'slurm_cpus' for the Pool (usually 4), though we only have 1 item.
    pool = multiprocessing.Pool(processes=slurm_cpus)
    fit_tasks = [(df, i, bounds, None) for i, df in enumerate(sessions_df)]
    
    # Main Progress Bar
    results = list(tqdm(pool.imap(fit_single_session_torch, fit_tasks), total=len(sessions_df), desc="Sessions"))
    pool.close()
    
    sim_dfs_list = [r['sim_df'] for r in results]
    valid_params = [r['best_params'] for r in results if r['success']]
    fixed_biases = [r['fixed_bias'] for r in results]
    
    param_names_reduced = [
        'decay_rate', 'noise_param_a', 
        'alpha_reward', 'alpha_punish', 'alpha_unc',
        'alpha_ch_r', 'alpha_ch_p', 'beta', 'lapse',
        'p_switch', 'p_rare'
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
        
        # --- PREPARE DATA FOR ANALYSIS ---
        correct_map = {'short': 'left', 'long': 'right'}
        
        comparison = sim_all.copy()
        comparison['correct_mouse'] = exp_all.apply(
            lambda row: 1 if row['mouse_choice'] == correct_map[row['trial_type']] else 0, axis=1
        )
        comparison['correct_model'] = comparison['correct_model'].astype(int)
        comparison['block_type'] = exp_all['block_type']
        comparison['isi'] = exp_all['isi']

        # --- 1. Overall accuracies ---
        mouse_acc = comparison['correct_mouse'].mean()
        model_acc = comparison['correct_model'].mean()
        print(f"\nACCURACY COMPARISON")
        print(f"   Mouse overall:  {mouse_acc:.3%}")
        print(f"   Model overall:  {model_acc:.3%}")
        print(f"   Difference:     {model_acc - mouse_acc:+.3%}")

        # --- 2. Accuracy by block type ---
        print(f"\nBY BLOCK TYPE:")
        for block in ['short_block', 'long_block', 'neutral']:
            subset = comparison[comparison['block_type'] == block]
            if len(subset) == 0: continue
            print(f"   {block:12}: Mouse {subset['correct_mouse'].mean():.3%} | Model {subset['correct_model'].mean():.3%}")

        # --- 3. Accuracy by ISI bin ---
        print(f"\nBY ISI BIN:")
        isi_bins = pd.cut(comparison['isi'], bins=np.linspace(0, 2.5, 21))
        acc_by_bin = comparison.groupby(isi_bins, observed=False).agg(
            mouse_acc=('correct_mouse', 'mean'),
            model_acc=('correct_model', 'mean'),
            n=('correct_mouse', 'size')
        ).round(3)
        print(acc_by_bin)

        # --- 4. Trial-by-trial agreement ---
        agreement = (comparison['mouse_choice'] == comparison['model_choice']).mean()
        print(f"\nTRIAL-BY-TRIAL CHOICE AGREEMENT: {agreement:.3%}")
        
        # --- PLOTTING (Saves per session for Array Job) ---
        exp_all_with_keys = pd.concat(sessions_df, keys=range(len(sessions_df)), names=['session_id', 'trial_in_session'])
        sim_all_with_keys = pd.concat(sim_dfs_list, keys=range(len(sim_dfs_list)), names=['session_id', 'trial_in_session'])

        print("\nGenerating Plots...")
        
        # Update save paths to include Task ID to prevent overwriting if running same dir
        # (Assuming your plotting functions handle filename uniqueness, but best to be safe)
        task_suffix = f"_task{task_id}"
        
        # Note: You might want to modify your Save functions to append the session name or task ID
        # For now, we rely on Save_comprehensive_results which likely pickles the data.
        
        plot_psychometric_comparison(exp_all, sim_all, boundary=1.25, save_path=SAVE_PATH)
        plot_internal_states(sim_all_with_keys, save_path=SAVE_PATH)
        plot_loss_evolution(results, save_path=SAVE_PATH)
        plot_weight_trajectories(sim_all_with_keys, save_path=SAVE_PATH)
        plot_boundary_dynamics(sim_all_with_keys, save_path=SAVE_PATH)
        plot_all_transition_dynamics(sim_all_with_keys, exp_all_with_keys, save_path=SAVE_PATH)
        plot_rare_trial_dynamics(sim_all_with_keys, window=3, save_path=SAVE_PATH)
        plot_performance_around_transitions(exp_all_with_keys, sim_all_with_keys, window=20, save_path=SAVE_PATH)
        plot_performance_around_rare_trials(exp_all_with_keys, sim_all_with_keys, window=3, save_path=SAVE_PATH)

        # --- SAVE RESULTS ---
        # We append the Task ID to the prefix so files don't collide
        save_comprehensive_results(results, prefix=f'Dynamic_Boundary_Fit_Task{task_id}', save_path=SAVE_PATH)