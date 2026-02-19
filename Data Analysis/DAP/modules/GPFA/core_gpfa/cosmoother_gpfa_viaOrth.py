"""
Direct Python port of cosmoother_gpfa_viaOrth.m

Performs leave-neuron-out prediction for GPFA.
"""

import numpy as np
from typing import List, Dict, Any
from modules.GPFA.core_gpfa.exactInferenceWithLL import exact_inference_with_ll
from modules.GPFA.util.orthogonalize import orthogonalize
from modules.GPFA.util.segmentByTrial import segment_by_trial


def cosmoother_gpfa_viaorth(seq: List[Dict[str, Any]], 
                           params: Dict[str, Any], 
                           m_list: List[int]) -> List[Dict[str, Any]]:
    """
    Performs leave-neuron-out prediction for GPFA.

    INPUTS:
    seq         - test data structure
    params      - GPFA model parameters fit to training data
    m_list      - number of top orthonormal latent coordinates to use for 
                  prediction (e.g., [1, 2, 3, 4, 5])

    OUTPUTS:
    seq         - test data structure with new fields ycs_orth_XX, where XX are
                  elements of m_list. seq[n]['ycs_orth_XX'] has the same dimensions
                  as seq[n]['y'].

    @ 2009 Byron Yu         byronyu@stanford.edu
           John Cunningham  jcunnin@stanford.edu
    """
    
    y_dim, x_dim = params['C'].shape

    # Initialize output fields
    for n in range(len(seq)):
        for m in m_list:
            fn = f'ycs_orth_{m:02d}'
            seq[n][fn] = np.full((y_dim, seq[n]['T']), np.nan)

    for i in range(y_dim):
        # Indices 0:y_dim with i removed (Python 0-based indexing)
        mi = list(range(i)) + list(range(i + 1, y_dim))
        
        # Create reduced data structure (leave out neuron i)
        seq_cs = []
        for n in range(len(seq)):
            trial_cs = {
                'T': seq[n]['T'],
                'y': seq[n]['y'][mi, :]
            }
            seq_cs.append(trial_cs)
        
        # Create reduced parameter structure
        params_cs = {
            'C': params['C'][mi, :],
            'd': params['d'][mi],
            'R': params['R'][np.ix_(mi, mi)],
            # Copy other necessary parameters
            'gamma': params.get('gamma', None),
            'eps': params.get('eps', None),
            'notes': params.get('notes', {})
        }
        
        # Perform inference without neuron i
        seq_cs = exact_inference_with_ll(seq_cs, params_cs, get_ll=False)
        
        # Note: it is critical to use params['C'] here and not params_cs['C']
        xsm_all = np.concatenate([trial['xsm'] for trial in seq_cs], axis=1)
        x_orth, c_orth = orthogonalize(xsm_all, params['C'])
        seq_cs = segment_by_trial(seq_cs, x_orth, 'xorth')
        
        # Make predictions for neuron i
        for n in range(len(seq)):
            for m in m_list:
                fn = f'ycs_orth_{m:02d}'
                seq[n][fn][i, :] = (c_orth[i, :m] @ seq_cs[n]['xorth'][:m, :] + 
                                   params['d'][i])
        
        print(f'Cross-validation complete for {i+1:3d} of {y_dim} neurons\r', 
              end='', flush=True)
    
    print()  # New line after progress updates
    return seq