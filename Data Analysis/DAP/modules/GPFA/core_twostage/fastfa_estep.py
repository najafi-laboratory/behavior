"""
Direct Python port of fastfa_estep.m

Compute the low-dimensional points and data likelihoods using a
previously learned FA or PPCA model.
"""

import numpy as np
from typing import Dict, Any, Tuple
from modules.GPFA.util.logdet import logdet


def fastfa_estep(X: np.ndarray, params: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Compute the low-dimensional points and data likelihoods using a
    previously learned FA or PPCA model.

      x_dim: data dimensionality
      z_dim: latent dimensionality
      N:    number of data points

    INPUTS:
    X        - data matrix (x_dim x N)
    params   - learned FA or PPCA parameters (structure with fields L, Ph, d)

    OUTPUTS:
    Z        - dictionary with fields:
                 mean (z_dim x N)       -- posterior mean
                 cov (z_dim x z_dim)    -- posterior covariance (same for all data)
    LL       - log-likelihood of data

    Note: the choice of FA vs. PPCA does not need to be specified because  
    the choice is reflected in params['Ph'].

    Code adapted from ffa.m by Zoubin Ghaharamani.

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    x_dim, N = X.shape
    z_dim = params['L'].shape[1]
    
    L = params['L']
    Ph = params['Ph']
    d = params['d']
    
    Xc = X - d[:, np.newaxis]  # bsxfun equivalent
    XcXc = Xc @ Xc.T
    
    I = np.eye(z_dim)
    
    const = -x_dim / 2 * np.log(2 * np.pi)
    
    iPh = np.diag(1.0 / Ph)
    iPhL = iPh @ L
    MM = iPh - iPhL @ np.linalg.solve(I + L.T @ iPhL, iPhL.T)
    beta = L.T @ MM  # z_dim x x_dim
    
    Z = {
        'mean': beta @ Xc,      # z_dim x N
        'cov': I - beta @ L     # z_dim x z_dim; same for all observations
    }
    
    LL = N * const + 0.5 * N * logdet(MM) - 0.5 * np.sum(MM * XcXc)
    
    return Z, LL