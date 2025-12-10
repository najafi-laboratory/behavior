"""
Direct Python port of cosmoother_pca.m

Performs leave-neuron-out prediction for PCA.
"""

import numpy as np
from typing import Dict, Any


def cosmoother_pca(Y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Performs leave-neuron-out prediction for PCA.

    INPUTS:
    Y           - test data (# neurons x # data points)
    params      - PCA parameters fit to training data

    OUTPUTS:
    Ycs         - leave-neuron-out prediction (# neurons x # data points)

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    L = params['L']
    d = params['d']
    
    y_dim, x_dim = L.shape
    
    Ycs = np.zeros_like(Y)
    
    for i in range(y_dim):
        # Indices 0:y_dim with i removed (Python 0-based indexing)
        mi = list(range(i)) + list(range(i + 1, y_dim))
        
        Xmi = (np.linalg.inv(L[mi, :].T @ L[mi, :]) @ L[mi, :].T @ 
               (Y[mi, :] - d[mi][:, np.newaxis]))
        
        Ycs[i, :] = L[i, :] @ Xmi + d[i]
    
    return Ycs