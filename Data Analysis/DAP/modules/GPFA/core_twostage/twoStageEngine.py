"""
Direct Python port of twoStageEngine.m

Extract neural trajectories using a two-stage method.
"""

import numpy as np
import pickle
from typing import List, Dict, Any
from sklearn.decomposition import PCA
from modules.GPFA.util.assignopts import assign_opts
from modules.GPFA.util.smoother import smoother
from modules.GPFA.core_twostage.fastfa import fastfa
from modules.GPFA.core_twostage.fastfa_estep import fastfa_estep
from modules.GPFA.core_twostage.cosmoother_fa import cosmoother_fa
from modules.GPFA.core_twostage.cosmoother_pca import cosmoother_pca
from modules.GPFA.util.segmentByTrial import segment_by_trial


def two_stage_engine(seq_train: List[Dict[str, Any]], 
                    seq_test: List[Dict[str, Any]], 
                    fname: str, 
                    **kwargs) -> None:
    """
    Extract neural trajectories using a two-stage method.

    INPUTS:
    seq_train      - training data structure, whose nth entry (corresponding to
                     the nth experimental trial) has fields
                       trial_id (1 x 1)   -- unique trial identifier
                       y (# neurons x T) -- neural data
                       T (1 x 1)         -- number of timesteps
    seq_test       - test data structure (same format as seq_train)
    fname          - filename of where results are saved

    OPTIONAL ARGUMENTS:
    typ            - type of dimensionality reduction
                     'fa' (default), 'ppca', 'pca'
    x_dim          - state dimensionality (default: 3)
    bin_width      - spike bin width in msec (default: 20)
    kern_sd_list   - vector of Gaussian smoothing kernel widths to run
                     Values are standard deviations in msec (default: 20:5:80)

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    typ = 'fa'
    x_dim = 3
    bin_width = 20  # in msec
    kern_sd_list = list(range(20, 85, 5))  # 20:5:80 in msec
    extra_opts = assign_opts(locals(), kwargs)

    kern = []
    
    for k in range(len(kern_sd_list)):
        print(f"Performing 'smooth and {typ.upper()}' with kernSD={kern_sd_list[k]}...")
        
        kern_k = {}
        
        # ======================
        # Smooth data over time
        # ======================
        kern_k['kernSD'] = kern_sd_list[k]
        
        # Training data
        kern_k['seqTrain'] = []
        for n in range(len(seq_train)):
            trial = seq_train[n].copy()
            trial['y'] = smoother(seq_train[n]['y'], kern_sd_list[k], bin_width)
            kern_k['seqTrain'].append(trial)
        
        # Test data
        kern_k['seqTest'] = []
        for n in range(len(seq_test)):
            trial = seq_test[n].copy()
            trial['y'] = smoother(seq_test[n]['y'], kern_sd_list[k], bin_width)
            kern_k['seqTest'].append(trial)

        Y_train_raw = np.concatenate([trial['y'] for trial in seq_train], axis=1)
        Y_test_raw = np.concatenate([trial['y'] for trial in seq_test], axis=1)
        
        # ===============================
        # Apply dimensionality reduction
        # ===============================
        Y = np.concatenate([trial['y'] for trial in kern_k['seqTrain']], axis=1)
        
        if typ in ['fa', 'ppca']:
            est_params, LL = fastfa(Y, x_dim, typ=typ, **extra_opts)
            X = fastfa_estep(Y, est_params)
            # To save disk space, don't save posterior covariance, which is identical
            # for each data point and can be computed from the learned parameters.
            kern_k['seqTrain'] = segment_by_trial(kern_k['seqTrain'], X['mean'], 'xpost')
            kern_k['estParams'] = est_params
            kern_k['LL'] = LL
                
        elif typ == 'pca':
            # Using sklearn's PCA (equivalent to MATLAB's pca function)
            pca = PCA(n_components=x_dim)
            pc_scores = pca.fit_transform(Y.T)  # Y.T to match MATLAB convention
            pc_dirs = pca.components_.T  # Components as columns
            
            kern_k['seqTrain'] = segment_by_trial(kern_k['seqTrain'], 
                                                 pc_scores[:, :x_dim].T, 'xpost')
            
            kern_k['estParams'] = {
                'L': pc_dirs[:, :x_dim],
                'd': np.mean(Y, axis=1)
            }
        
        # ========================================
        # Leave-neuron-out prediction on test data
        # ========================================
        if seq_test:  # check if there are any test trials
            Y = np.concatenate([trial['y'] for trial in kern_k['seqTest']], axis=1)
            
            if typ in ['fa', 'ppca']:
                Ycs = cosmoother_fa(Y, kern_k['estParams'])
                kern_k['seqTest'] = segment_by_trial(kern_k['seqTest'], Ycs, 'ycs')
                
            elif typ == 'pca':
                Ycs = cosmoother_pca(Y, kern_k['estParams'])
                kern_k['seqTest'] = segment_by_trial(kern_k['seqTest'], Ycs, 'ycs')
        
        kern.append(kern_k)

    # =============
    # Save results
    # =============
    # Create results dictionary with all relevant variables
    results = {
        'kern': kern,
        'typ': typ,
        'x_dim': x_dim,
        'bin_width': bin_width,
        'kern_sd_list': kern_sd_list,
        'extra_opts': extra_opts,
        'Y_train_raw': Y_train_raw,
        'Y_test_raw': Y_test_raw
    }

    print(f'Saving {fname}...')
    with open(f'{fname}.pkl', 'wb') as f:
        pickle.dump(results, f)