"""
Direct Python port of neuralTraj.

Prepares data and calls functions for extracting neural trajectories.
"""

import numpy as np
import os
import pickle
from typing import List, Dict, Any, Optional
from modules.GPFA.util.assignopts import assign_opts
from modules.GPFA.util.getSeq import get_seq
from modules.GPFA.core_gpfa.gpfaEngine import gpfa_engine
from modules.GPFA.core_twostage.twoStageEngine import two_stage_engine


def neural_traj(run_idx: int, 
               dat: List[Dict[str, Any]], 
               **kwargs) -> Optional[Dict[str, Any]]:
    """
    Prepares data and calls functions for extracting neural trajectories.

    INPUTS:
    run_idx     - results files will be saved in pkl_results/runXXX, where
                  XXX is run_idx
    dat         - list whose nth entry (corresponding to the nth experimental
                  trial) has fields
                    'trialId' -- unique trial identifier
                    'spikes'  -- 0/1 matrix of the raw spiking activity across 
                                all neurons.  Each row corresponds to a neuron.  
                                Each column corresponds to a 1 msec timestep.

    OUTPUTS:
    result      - dictionary containing all variables saved in pkl_results/runXXX/
                  if 'num_folds' is 0. Else, returns None.
                  
    OPTIONAL ARGUMENTS:
    method      - method for extracting neural trajectories
                  'gpfa' (default), 'fa', 'ppca', 'pca'
    bin_width   - spike bin width in msec (default: 20)
    num_folds   - number of cross-validation folds (default: 0)
                  0 indicates no cross-validation, i.e. train on all trials.
    x_dim       - state dimensionality (default: 3)

    @ 2009 Byron Yu         byronyu@stanford.edu
           John Cunningham  jcunnin@stanford.edu
    Python port by GitHub Copilot
    """
    
    method = 'gpfa'
    bin_width = 20  # in msec
    num_folds = 0
    x_dim = 8
    extra_opts = assign_opts(locals(), kwargs)

    print('\n---------------------------------------')
    
    # Create results directory
    if not os.path.exists('pkl_results'):
        os.makedirs('pkl_results')
    
    # Make a directory for this run_idx if it doesn't already exist
    run_dir = f'pkl_results/run{run_idx:03d}'
    if os.path.exists(run_dir):
        print(f'Using existing directory {run_dir}...')
    else:
        print(f'Making directory {run_dir}...')
        os.makedirs(run_dir)

    # Obtain binned spike counts
    seq = get_seq(dat, bin_width, **extra_opts)
    if not seq:
        print('Error: No valid trials. Exiting.')
        return None

    # Set cross-validation folds
    N = len(seq)
    fdiv = np.floor(np.linspace(1, N + 1, num_folds + 1)).astype(int)

    result_filename = None  # Track the filename for return

    for cvf in range(num_folds + 1):
        if cvf == 0:
            print('\n===== Training on all data =====')
        else:
            print(f'\n===== Cross-validation fold {cvf} of {num_folds} =====')

        # Specify filename where results will be saved
        fname = f'{run_dir}/{method}_xDim{x_dim:02d}'
        if cvf > 0:
            fname = f'{fname}_cv{cvf:02d}'
        
        fname_pkl = f'{fname}.pkl'
        
        if os.path.exists(fname_pkl):
            print(f'{fname_pkl} already exists. Skipping...')
            continue

        # Set cross-validation masks
        test_mask = np.zeros(N, dtype=bool)
        if cvf > 0:
            test_mask[fdiv[cvf-1]:fdiv[cvf]] = True  # Convert to 0-based indexing
        train_mask = ~test_mask

        if cvf == 0:
            # If training on all trials, keep original trial ordering
            tr = np.arange(N)
        else:
            # Randomly reorder trials before partitioning into training and test sets
            np.random.seed(0)
            tr = np.random.permutation(N)
        
        train_trial_idx = tr[train_mask]
        test_trial_idx = tr[test_mask]
        seq_train = [seq[i] for i in train_trial_idx]
        seq_test = [seq[i] for i in test_trial_idx]

        # Remove inactive units based on training set
        y_all_train = np.concatenate([trial['y'] for trial in seq_train], axis=1)
        has_spikes_bool = (np.mean(y_all_train, axis=1) != 0)

        for n in range(len(seq_train)):
            seq_train[n]['y'] = seq_train[n]['y'][has_spikes_bool, :]
        for n in range(len(seq_test)):
            seq_test[n]['y'] = seq_test[n]['y'][has_spikes_bool, :]

        # Check if training data covariance is full rank
        y_all = np.concatenate([trial['y'] for trial in seq_train], axis=1)
        y_dim = y_all.shape[0]

        if np.linalg.matrix_rank(np.cov(y_all.T, rowvar=False)) < y_dim:
            print('ERROR: Observation covariance matrix is rank deficient.')
            print('Possible causes: repeated units, not enough observations.')
            print('Exiting...')
            return None

        print(f'Number of training trials: {len(seq_train)}')
        print(f'Number of test trials: {len(seq_test)}')
        print(f'Latent space dimensionality: {x_dim}')
        print(f'Observation dimensionality: {np.sum(has_spikes_bool)}')

        # If doing cross-validation, don't use private noise variance floor
        if cvf > 0:
            extra_opts_cv = {**extra_opts, 'min_var_frac': -np.inf}
        else:
            extra_opts_cv = extra_opts

        # The following does the heavy lifting
        if method == 'gpfa':
            gpfa_engine(seq_train, seq_test, fname,
                       x_dim=x_dim, bin_width=bin_width, **extra_opts_cv)
        
        elif method in ['fa', 'ppca', 'pca']:
            two_stage_engine(seq_train, seq_test, fname,
                           typ=method, x_dim=x_dim, bin_width=bin_width, **extra_opts_cv)

        # Save additional metadata
        if os.path.exists(fname_pkl):
            # Load existing data and append metadata
            with open(fname_pkl, 'rb') as f:
                saved_data = pickle.load(f)
            
            # Add metadata
            saved_data.update({
                'method': method,
                'cvf': cvf,
                'has_spikes_bool': has_spikes_bool,
                'extra_opts': extra_opts
            })
            
            # Save back with metadata
            with open(fname_pkl, 'wb') as f:
                pickle.dump(saved_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if cvf == 0:
                result_filename = fname_pkl

    # Return result if requested and appropriate
    result = None
    if num_folds == 0 and result_filename and os.path.exists(result_filename):
        with open(result_filename, 'rb') as f:
            result = pickle.load(f)

    return result


def load_neural_traj_results(run_idx: int, 
                           method: str = 'gpfa', 
                           x_dim: int = 8, 
                           cvf: int = 0) -> Optional[Dict[str, Any]]:
    """
    Load results from a previous neural_traj run.
    
    INPUTS:
    run_idx - run index
    method  - method used ('gpfa', 'fa', 'ppca', 'pca')
    x_dim   - dimensionality used
    cvf     - cross-validation fold (0 for training on all data)
    
    OUTPUTS:
    result  - loaded results dictionary, or None if file not found
    """
    
    run_dir = f'pkl_results/run{run_idx:03d}'
    fname = f'{run_dir}/{method}_xDim{x_dim:02d}'
    
    if cvf > 0:
        fname = f'{fname}_cv{cvf:02d}'
    
    fname_pkl = f'{fname}.pkl'
    
    if os.path.exists(fname_pkl):
        with open(fname_pkl, 'rb') as f:
            return pickle.load(f)
    else:
        print(f'Results file not found: {fname_pkl}')
        return None


def list_neural_traj_results(run_idx: int) -> List[str]:
    """
    List all result files for a given run index.
    
    INPUTS:
    run_idx - run index
    
    OUTPUTS:
    files   - list of result filenames
    """
    
    run_dir = f'pkl_results/run{run_idx:03d}'
    
    if not os.path.exists(run_dir):
        print(f'Run directory not found: {run_dir}')
        return []
    
    files = []
    for filename in os.listdir(run_dir):
        if filename.endswith('.pkl'):
            files.append(os.path.join(run_dir, filename))
    
    return sorted(files)