"""
Direct Python port of make_K_big.m

Constructs full GP covariance matrix across all state dimensions and timesteps.
"""

import numpy as np
from typing import Tuple, Dict, Any
from modules.GPFA.util.invToeplitz.invToeplitz import inv_toeplitz


def make_k_big(params: Dict[str, Any], T: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Constructs full GP covariance matrix across all state dimensions and timesteps.

    INPUTS:
    params       - GPFA model parameters
    T            - number of timesteps

    OUTPUTS:
    K_big        - GP covariance matrix with dimensions (x_dim * T) x (x_dim * T).
                   The (t1, t2) block is diagonal, has dimensions x_dim x x_dim, and 
                   represents the covariance between the state vectors at
                   timesteps t1 and t2.  K_big is sparse and striped.
    K_big_inv    - inverse of K_big
    logdet_K_big - log determinant of K_big

    @ 2009 Byron Yu         byronyu@stanford.edu
           John Cunningham  jcunnin@stanford.edu
    """
    
    x_dim = params['C'].shape[1]
    
    idx = np.arange(0, x_dim * T, x_dim)
    
    K_big = np.zeros((x_dim * T, x_dim * T))
    K_big_inv = np.zeros((x_dim * T, x_dim * T))
    
    # Create time difference matrix
    t_grid = np.arange(1, T + 1)  # 1:T in MATLAB
    Tdif = t_grid[:, np.newaxis] - t_grid[np.newaxis, :]  # repmat equivalent
    
    logdet_K_big = 0
    
    for i in range(x_dim):
        if params['covType'] == 'rbf':
            K = ((1 - params['eps'][i]) * 
                 np.exp(-params['gamma'][i] / 2 * Tdif**2) + 
                 params['eps'][i] * np.eye(T))
        
        elif params['covType'] == 'tri':
            K = (np.maximum(1 - params['eps'][i] - params['a'][i] * np.abs(Tdif), 0) + 
                 params['eps'][i] * np.eye(T))
        
        elif params['covType'] == 'logexp':
            z = (params['gamma'] * 
                 (1 - params['eps'][i] - params['a'][i] * np.abs(Tdif)))
            
            out_UL = (z > 36)
            out_LL = (z < -19)
            in_lim = ~out_UL & ~out_LL
            
            hz = np.full_like(z, np.nan)
            hz[out_UL] = z[out_UL]
            hz[out_LL] = np.exp(z[out_LL])
            hz[in_lim] = np.log(1 + np.exp(z[in_lim]))
            
            K = hz / params['gamma'] + params['eps'][i] * np.eye(T)
        
        else:
            raise ValueError(f"Unknown covariance type: {params['covType']}")
        
        # Fill diagonal blocks
        # K_big[idx + i, idx + i] = K
        K_big[np.ix_(idx + i, idx + i)] = K
        K_inv, logdet_K = inv_toeplitz(K)
        # K_big_inv[idx + i, idx + i] = K_inv
        K_big_inv[np.ix_(idx + i, idx + i)] = K_inv
        
        logdet_K_big += logdet_K
    
    return K_big, K_big_inv, logdet_K_big