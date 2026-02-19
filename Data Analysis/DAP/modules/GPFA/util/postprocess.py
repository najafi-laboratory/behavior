"""
Direct Python port of postprocess.m

Orthonormalization and other cleanup.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from modules.GPFA.util.assignopts import assign_opts
from modules.GPFA.util.orthogonalize import orthogonalize
from modules.GPFA.util.segmentByTrial import segment_by_trial
from modules.GPFA.core_gpfa.exactInferenceWithLL import exact_inference_with_ll
from modules.GPFA.core_twostage.fastfa_estep import fastfa_estep


def postprocess(ws: Dict[str, Any], **kwargs) -> Tuple[Optional[Dict[str, Any]], 
                                                      Optional[List[Dict[str, Any]]], 
                                                      Optional[List[Dict[str, Any]]]]:
    """
    Orthonormalization and other cleanup.

    INPUTS:
    ws        - workspace variables returned by neural_traj function

    OUTPUTS:
    est_params - estimated model parameters, including 'C_orth' obtained
                 by orthonormalizing the columns of C
    seq_train  - training data structure containing new field 'x_orth', 
                 the orthonormalized neural trajectories
    seq_test   - test data structure containing orthonormalized neural
                 trajectories in 'x_orth', obtained using 'est_params'

    OPTIONAL ARGUMENTS:
    kern_sd    - for two-stage methods, this function returns seq_train
                 and est_params corresponding to kern_sd. By default, 
                 the function uses ws['kern'][0].

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    kern_sd = None
    extra_opts = assign_opts(locals(), kwargs)

    est_params = None
    seq_train = None
    seq_test = None

    if not ws:
        print('ERROR: Input argument is empty.')
        return est_params, seq_train, seq_test

    if 'kern' in ws:
        if not ws.get('kern_sd_list'):
            k = 0  # Use first kernel (0-based indexing)
        else:
            if kern_sd is None:
                k = 0
            else:
                try:
                    k = ws['kern_sd_list'].index(kern_sd)
                except ValueError:
                    print('ERROR: Selected kern_sd not found.')
                    return est_params, seq_train, seq_test

    if ws['method'] == 'gpfa':
        C = ws['est_params']['C']
        X = np.concatenate([trial['xsm'] for trial in ws['seq_train']], axis=1)
        X_orth, C_orth = orthogonalize(X, C)
        seq_train = segment_by_trial(ws['seq_train'], X_orth, 'xorth')
        
        est_params = ws['est_params'].copy()
        est_params['C_orth'] = C_orth

        if ws.get('seq_test'):
            print('Extracting neural trajectories for test data...')
            
            ws_seq_test, _ = exact_inference_with_ll(ws['seq_test'], est_params)
            X = np.concatenate([trial['xsm'] for trial in ws_seq_test], axis=1)
            X_orth, C_orth = orthogonalize(X, C)
            seq_test = segment_by_trial(ws_seq_test, X_orth, 'xorth')

    elif ws['method'] in ['fa', 'ppca']:
        L = ws['kern'][k]['est_params']['L']
        X = np.concatenate([trial['xpost'] for trial in ws['kern'][k]['seq_train']], axis=1)
        X_orth, L_orth = orthogonalize(X, L)
        seq_train = segment_by_trial(ws['kern'][k]['seq_train'], X_orth, 'xorth')
        
        # Convert to GPFA naming/formatting conventions
        est_params = {
            'C': ws['kern'][k]['est_params']['L'],
            'd': ws['kern'][k]['est_params']['d'],
            'C_orth': L_orth,
            'R': np.diag(ws['kern'][k]['est_params']['Ph'])
        }

        if ws['kern'][k].get('seq_test'):
            print('Extracting neural trajectories for test data...')
            
            Y = np.concatenate([trial['y'] for trial in ws['kern'][k]['seq_test']], axis=1)
            X = fastfa_estep(Y, ws['kern'][k]['est_params'])
            X_orth, L_orth = orthogonalize(X['mean'], L)
            seq_test = segment_by_trial(ws['kern'][k]['seq_test'], X_orth, 'xorth')

    elif ws['method'] == 'pca':
        # PCA is already orthonormalized
        X = np.concatenate([trial['xpost'] for trial in ws['kern'][k]['seq_train']], axis=1)
        seq_train = segment_by_trial(ws['kern'][k]['seq_train'], X, 'xorth')
        
        est_params = {
            'C_orth': ws['kern'][k]['est_params']['L'],
            'd': ws['kern'][k]['est_params']['d']
        }
        
        if ws['kern'][k].get('seq_test'):
            print('Extracting neural trajectories for test data...')
            
            Y = np.concatenate([trial['y'] for trial in ws['kern'][k]['seq_test']], axis=1)
            X_orth = est_params['C_orth'].T @ (Y - est_params['d'][:, np.newaxis])
            seq_test = segment_by_trial(ws['kern'][k]['seq_test'], X_orth, 'xorth')

    else:
        print('ERROR: method not recognized.')

    return est_params, seq_train, seq_test