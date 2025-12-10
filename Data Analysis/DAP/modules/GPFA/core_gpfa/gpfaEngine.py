"""
Direct Python port of gpfaEngine.m

Extract neural trajectories using GPFA.
"""

import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from modules.GPFA.util.assignopts import assign_opts
from modules.GPFA.util.cutTrials import cut_trials
from modules.GPFA.core_twostage.fastfa import fastfa
from modules.GPFA.core_gpfa.em import em
from modules.GPFA.core_gpfa.exactInferenceWithLL import exact_inference_with_ll
from modules.GPFA.core_gpfa.cosmoother_gpfa_viaOrth_fast import cosmoother_gpfa_viaorth_fast
from modules.GPFA.core_gpfa.cosmoother_gpfa_viaOrth import cosmoother_gpfa_viaorth


def gpfa_engine(seq_train: List[Dict[str, Any]], 
               seq_test: List[Dict[str, Any]], 
               fname: str, 
               **kwargs) -> None:
    """
    Extract neural trajectories using GPFA.

    INPUTS:
    seq_train      - training data structure, whose nth entry (corresponding to 
                     the nth experimental trial) has fields
                       trial_id (1 x 1)   -- unique trial identifier
                       y (# neurons x T) -- neural data
                       T (1 x 1)         -- number of timesteps
    seq_test       - test data structure (same format as seq_train)
    fname          - filename of where results are saved

    OPTIONAL ARGUMENTS:
    x_dim          - state dimensionality (default: 3)
    bin_width      - spike bin width in msec (default: 20)
    start_tau      - GP timescale initialization in msec (default: 100)
    start_eps      - GP noise variance initialization (default: 1e-3)

    @ 2009 Byron Yu         byronyu@stanford.edu
           John Cunningham  jcunnin@stanford.edu
    """
    
    x_dim = 8
    bin_width = 20  # in msec
    start_tau = 100  # in msec
    start_eps = 1e-3
    extra_opts = assign_opts(locals(), kwargs)

    # For compute efficiency, train on equal-length segments of trials
    seq_train_cut = cut_trials(seq_train, **extra_opts)
    if not seq_train_cut:
        print('WARNING: no segments extracted for training.  Defaulting to seg_length=Inf.')
        seq_train_cut = cut_trials(seq_train, seg_length=np.inf)

    # ==================================
    # Initialize state model parameters
    # ==================================
    start_params = {}
    start_params['covType'] = 'rbf'
    # GP timescale
    # Assume bin_width is the time step size.
    start_params['gamma'] = ((bin_width / start_tau)**2 * np.ones((1, x_dim)))[0]
    # GP noise variance
    start_params['eps'] = (start_eps * np.ones((1, x_dim)))[0]

    # ========================================
    # Initialize observation model parameters
    # ========================================
    print('Initializing parameters using factor analysis...')

    y_all = np.concatenate([trial['y'] for trial in seq_train_cut], axis=1)
    fa_params, fa_ll = fastfa(y_all, x_dim, **extra_opts)

    start_params['d'] = np.mean(y_all, axis=1)
    start_params['C'] = fa_params['L']
    start_params['R'] = np.diag(fa_params['Ph'])

    # Define parameter constraints
    start_params['notes'] = {
        'learnKernelParams': True,
        'learnGPNoise': False,
        'RforceDiagonal': True
    }

    current_params = start_params.copy()

    # =====================
    # Fit model parameters
    # =====================
    print('\nFitting GPFA model...')

    est_params, seq_train_cut, LL_cut, iter_time = em(current_params, seq_train_cut, **extra_opts)

    # Extract neural trajectories for original, unsegmented trials
    # using learned parameters
    seq_train, LL_train = exact_inference_with_ll(seq_train, est_params)

    # ==================================
    # Assess generalization performance
    # ==================================
    if seq_test:  # check if there are any test trials
        # Leave-neuron-out prediction on test data 
        if est_params['notes']['RforceDiagonal']:
            seq_test = cosmoother_gpfa_viaorth_fast(seq_test, est_params, list(range(1, x_dim + 1)))
        else:
            seq_test = cosmoother_gpfa_viaorth(seq_test, est_params, list(range(1, x_dim + 1)))
        # Compute log-likelihood of test data
        _, LL_test = exact_inference_with_ll(seq_test, est_params)
    else:
        LL_test = None

    # =============
    # Save results
    # =============
    # Create results dictionary with all relevant variables
    results = {
        'seq_train': seq_train,
        'seq_test': seq_test,
        'seq_train_cut': seq_train_cut,
        'est_params': est_params,
        'start_params': start_params,
        'current_params': current_params,
        'fa_params': fa_params,
        'fa_ll': fa_ll,
        'LL_cut': LL_cut,
        'LL_train': LL_train,
        'LL_test': LL_test,
        'iter_time': iter_time,
        'x_dim': x_dim,
        'bin_width': bin_width,
        'start_tau': start_tau,
        'start_eps': start_eps,
        'extra_opts': extra_opts
    }

    print(f'Saving {fname}...')
    with open(f'{fname}.pkl', 'wb') as f:
        pickle.dump(results, f)