"""
Direct Python port of getTrajNewTrials.m

Extract neural trajectories from a set of new trials using previously-fitted
model parameters.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .assign_opts import assign_opts
from .get_seq import get_seq
from .smoother import smoother
from .exact_inference_with_ll import exact_inference_with_ll
from .causal_inference import causal_inference
from .fastfa_estep import fastfa_estep
from .orthogonalize import orthogonalize
from .segment_by_trial import segment_by_trial


def get_traj_new_trials(ws: Dict[str, Any], 
                       dat: List[Dict[str, Any]], 
                       **kwargs) -> Optional[List[Dict[str, Any]]]:
    """
    Extract neural trajectories from a set of new trials using previously-fitted
    model parameters.

    INPUTS:
    ws          - saved workspace variables that include the previously-fitted
                  model parameters 'est_params'
    dat         - data for new trials with fields
                    'trialId' -- unique trial identifier
                    'spikes'  -- 0/1 matrix of the raw spiking activity across
                                all neurons. Each row corresponds to a neuron.
                                Each column corresponds to a 1 msec timestep.

    OUTPUTS:
    seq_new     - data structure containing orthonormalized neural trajectories
                  ('xorth') for the new trials

    OPTIONAL ARGUMENTS:
    kern_sd     - for two-stage methods, specify kernel smoothing width.  
                  By default, the function uses ws['kern'][0].
    causal      - logical indicating whether temporal smoothing should
                  include only past data (True) or all data (False)
                  (default: False)

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    kern_sd = None
    causal = False
    extra_opts = assign_opts(locals(), kwargs)

    if not ws:
        print('ERROR: Input argument is empty.')
        return None

    # Process data in the same way as in 'ws'    
    # Obtain binned spike counts
    seq_new = get_seq(dat, ws['bin_width'], **ws.get('extra_opts', {}))
    
    # Remove inactive units
    for n in range(len(seq_new)):
        seq_new[n]['y'] = seq_new[n]['y'][ws['has_spikes_bool'], :]

    if 'kern_sd_list' in ws:
        # Two-stage methods
        if kern_sd is None:
            k = 0  # Use first kernel (0-based indexing)
        else:
            try:
                k = ws['kern_sd_list'].index(kern_sd)
            except ValueError:
                print('ERROR: Selected kern_sd not found.')
                return None
        
        # Kernel smoothing
        for n in range(len(seq_new)):
            seq_new[n]['y'] = smoother(seq_new[n]['y'], 
                                     ws['kern_sd_list'][k], 
                                     ws['bin_width'], 
                                     causal=causal)

    if ws['method'] == 'gpfa' and not causal:
        seq_new, _ = exact_inference_with_ll(seq_new, ws['est_params'])
        C = ws['est_params']['C']
        X = np.concatenate([trial['xsm'] for trial in seq_new], axis=1)
        X_orth, C_orth = orthogonalize(X, C)
        seq_new = segment_by_trial(seq_new, X_orth, 'xorth')

    elif ws['method'] == 'gpfa' and causal:
        seq_new = causal_inference(seq_new, ws['est_params'], **extra_opts)
        C = ws['est_params']['C']
        X = np.concatenate([trial['xfi'] for trial in seq_new], axis=1)
        X_orth, C_orth = orthogonalize(X, C)
        seq_new = segment_by_trial(seq_new, X_orth, 'xorth')
        
    elif ws['method'] in ['fa', 'ppca']:
        Y = np.concatenate([trial['y'] for trial in seq_new], axis=1)
        X = fastfa_estep(Y, ws['kern'][k]['est_params'])
        L = ws['kern'][k]['est_params']['L']
        X_orth, L_orth = orthogonalize(X['mean'], L)
        seq_new = segment_by_trial(seq_new, X_orth, 'xorth')
        
    elif ws['method'] == 'pca':
        Y = np.concatenate([trial['y'] for trial in seq_new], axis=1)
        est_params = ws['kern'][k]['est_params']
        X_orth = est_params['L'].T @ (Y - est_params['d'][:, np.newaxis])
        seq_new = segment_by_trial(seq_new, X_orth, 'xorth')

    return seq_new