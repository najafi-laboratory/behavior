"""
Direct Python port of cosmoother_fa.m

Performs leave-neuron-out prediction for FA or PPCA.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


def cosmoother_fa(Y: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Performs leave-neuron-out prediction for FA or PPCA.

    INPUTS:
    Y           - test data (# neurons x # data points)
    params      - model parameters fit to training data using fastfa.py

    OUTPUTS: 
    Ycs         - leave-neuron-out prediction mean (# neurons x # data points)
    Vcs         - leave-neuron-out prediction variance (# neurons x 1)

    Note: the choice of FA vs. PPCA does not need to be specified because
    the choice is reflected in params['Ph'].

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    L = params['L']
    Ph = params['Ph']
    d = params['d']
    
    y_dim, x_dim = L.shape
    I = np.eye(x_dim)

    Ycs = np.zeros_like(Y)
    # One variance for each observed dimension
    # Doesn't depend on observed data
    Vcs = np.zeros((y_dim, 1))

    for i in range(y_dim):
        # Indices 0:y_dim with i removed (Python 0-based indexing)
        mi = list(range(i)) + list(range(i + 1, y_dim))
        
        Phinv = 1.0 / Ph[mi]  # (y_dim-1,)
        LRinv = (L[mi, :] * Phinv[:, np.newaxis]).T  # x_dim x (y_dim - 1)
        LRinvL = LRinv @ L[mi, :]  # x_dim x x_dim
        
        term2 = L[i, :] @ (I - np.linalg.solve(I + LRinvL, LRinvL))  # 1 x x_dim
        
        dif = Y[mi, :] - d[mi][:, np.newaxis]
        Ycs[i, :] = d[i] + term2 @ LRinv @ dif
        
        Vcs[i] = L[i, :] @ L[i, :].T + Ph[i] - term2 @ LRinvL @ L[i, :].T
    
    return Ycs, Vcs