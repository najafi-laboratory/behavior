"""
Direct Python port of causalInferencePrecomp.m

Perform precomputations for causal_inference.py.
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, Any, List, Tuple
from .assign_opts import assign_opts
from .make_k_big import make_k_big
from .inv_per_symm import inv_per_symm
from .fill_per_symm import fill_per_symm


def causal_inference_precomp(params: Dict[str, Any], T_max: int, **kwargs) -> Dict[str, Any]:
    """
    Perform precomputations for causal_inference.py.

    INPUT:
    params  - GPFA model parameters 
    T_max   - number of timesteps of the longest trial in dataset

    OUTPUT:
    precomp - data structure with fields
                 filt[t]['M'] (xDim x xDim*t) -- filter to obtain posterior mean 
                                                  for x_t given y_1,...,y_t
                 CRinv                         -- precomputation of C' * inv(R)

    OPTIONAL ARGUMENT:
    T_star  - number of timesteps of the longest filter to precompute
              (default: T_max)

    @ 2011 Byron Yu         byronyu@cmu.edu
    """
    
    T_star = T_max
    extra_opts = assign_opts(locals(), kwargs)
    
    y_dim, x_dim = params['C'].shape

    if params['notes']['RforceDiagonal']:
        Rinv = np.diag(1.0 / np.diag(params['R']))
    else:
        Rinv = np.linalg.inv(params['R'])
        Rinv = (Rinv + Rinv.T) / 2  # ensure symmetry
    
    CRinv = params['C'].T @ Rinv
    CRinvC = CRinv @ params['C']
    
    precomp = {'filt': []}
    
    for T in range(1, T_star + 1):
        K_big, K_big_inv = make_k_big(params, T)
        K_big = sp.csr_matrix(K_big)

        # Create block diagonal matrix of CRinvC
        blah = [CRinvC for _ in range(T)]
        blk_diag_CRinvC = sp.block_diag(blah)
        
        invM = inv_per_symm(K_big_inv + blk_diag_CRinvC, x_dim, 
                           off_diag_sparse=True)
        
        # Compute blkProd = CRinvC_big * invM efficiently
        # blkProd is block persymmetric, so just compute top half
        T_half = int(np.ceil(T / 2))
        blk_prod = np.zeros((x_dim * T_half, x_dim * T))
        idx = np.arange(0, x_dim * T_half + 1, x_dim)
        
        for t in range(T_half):
            b_idx = slice(idx[t], idx[t + 1])
            blk_prod[b_idx, :] = CRinvC @ invM[b_idx, :]
        
        # Compute just the first block row of blkProd
        eye_sparse = sp.eye(x_dim * T_half, x_dim * T)
        blk_prod = K_big[:x_dim, :] @ fill_per_symm(eye_sparse - blk_prod, x_dim, T)
        
        # Last block row of blkProd is just block-reversal of first block row
        horz_idx = np.add.outer(np.arange(x_dim), np.arange(T - 1, -1, -1) * x_dim)
        M = blk_prod[:, horz_idx.ravel()]
        
        precomp['filt'].append({'M': M})
    
    precomp['CRinv'] = CRinv
    
    return precomp