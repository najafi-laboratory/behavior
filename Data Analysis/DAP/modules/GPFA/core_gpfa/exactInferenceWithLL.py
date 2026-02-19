"""
Direct Python port of exactInferenceWithLL.m

Extracts latent trajectories given GPFA model parameters.
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Any, Tuple
from modules.GPFA.util.assignopts import assign_opts
from modules.GPFA.core_gpfa.make_K_big import make_k_big
from modules.GPFA.util.invPerSymm import inv_per_symm
from modules.GPFA.util.fillPerSymm import fill_per_symm
from modules.GPFA.util.logdet import logdet


def exact_inference_with_ll(seq: List[Dict[str, Any]], 
                           params: Dict[str, Any], 
                           **kwargs) -> Tuple[List[Dict[str, Any]], float]:
    """
    Extracts latent trajectories given GPFA model parameters.

    INPUTS:
    seq         - data structure, whose nth entry (corresponding to the nth 
                  experimental trial) has fields
                    y (y_dim x T) -- neural data
                    T (1 x 1)    -- number of timesteps
    params      - GPFA model parameters 
     
    OUTPUTS:
    seq         - data structure with new fields
                    xsm (x_dim x T)        -- posterior mean at each timepoint
                    Vsm (x_dim x x_dim x T) -- posterior covariance at each timepoint
                    VsmGP (T x T x x_dim)  -- posterior covariance of each GP
    LL          - data log likelihood

    OPTIONAL ARGUMENTS:
    get_ll      - logical that specifies whether to compute data log likelihood
                  (default: True)

    @ 2009 Byron Yu         byronyu@stanford.edu
           John Cunningham  jcunnin@stanford.edu
    """
    
    get_ll = True
    extra_opts = assign_opts(locals(), kwargs)
    
    y_dim, x_dim = params['C'].shape

    # Precomputations
    if params['notes']['RforceDiagonal']:
        Rinv = np.diag(1.0 / np.diag(params['R']))
        logdet_R = np.sum(np.log(np.diag(params['R'])))
    else:
        Rinv = np.linalg.inv(params['R'])
        Rinv = (Rinv + Rinv.T) / 2  # ensure symmetry
        logdet_R = logdet(params['R'])
    
    CRinv = params['C'].T @ Rinv
    CRinvC = CRinv @ params['C']
    
    T_all = [trial['T'] for trial in seq]
    Tu = np.unique(T_all)
    LL = 0

    # Overview:
    # - Outer loop on each elt of Tu.
    # - For each elt of Tu, find all trials with that length.
    # - Do inference and LL computation for all those trials together.
    for j in range(len(Tu)):
        T = Tu[j]

        K_big, K_big_inv, logdet_K_big = make_k_big(params, T)
        
        # There are three sparse matrices here: K_big, K_big_inv, and CRinvC_inv.
        # Choosing which one(s) to make sparse is tricky.  If all are sparse,
        # code slows down significantly.  Empirically, code runs fastest if
        # only K_big is made sparse.
        #
        # There are two problems with calling both K_big_inv and CRCinvC_big 
        # sparse:
        # 1) their sum is represented by Python as a sparse matrix and taking 
        #    its inverse is more costly than taking the inverse of the 
        #    corresponding full matrix.
        # 2) term2 has very few zero entries, but Python will represent it as a 
        #    sparse matrix.  This makes later computations with term2 inefficient.
        
        K_big = sp.csr_matrix(K_big)

        blah = [CRinvC for _ in range(T)]
        # CRinvC_big = sp.block_diag(blah)  # (x_dim*T) x (x_dim*T)
        invM, logdet_M = inv_per_symm(K_big_inv + sp.block_diag(blah), x_dim,
                                     off_diag_sparse=True)
        
        # Note that posterior covariance does not depend on observations, 
        # so can compute once for all trials with same T.
        # x_dim x x_dim posterior covariance for each timepoint
        Vsm = np.full((x_dim, x_dim, T), np.nan)
        idx = np.arange(0, x_dim * T + 1, x_dim)
        for t in range(T):
            c_idx = slice(idx[t], idx[t + 1])
            Vsm[:, :, t] = invM[c_idx, c_idx]
        
        # T x T posterior covariance for each GP
        VsmGP = np.full((T, T, x_dim), np.nan)
        idx = np.arange(0, x_dim * (T - 1) + 1, x_dim)
        for i in range(x_dim):
            VsmGP[:, :, i] = invM[idx + i, :][:, idx + i]
        
        # Process all trials with length T
        n_list = [n for n, t in enumerate(T_all) if t == T]
        
        # Concatenate y data for all trials of length T
        y_concat = np.concatenate([seq[n]['y'] for n in n_list], axis=1)
        dif = y_concat - params['d'][:, np.newaxis]  # y_dim x sum(T)
        term1_mat = (CRinv @ dif).reshape(x_dim * T, -1, order='F')  # (x_dim*T) x length(n_list)

        # Compute blk_prod = CRinvC_big * invM efficiently
        # blk_prod is block persymmetric, so just compute top half
        T_half = int(np.ceil(T / 2))
        blk_prod = np.zeros((x_dim * T_half, x_dim * T))
        idx = np.arange(0, x_dim * T_half + 1, x_dim)
        for t in range(T_half):
            b_idx = slice(idx[t], idx[t + 1])
            blk_prod[b_idx, :] = CRinvC @ invM[b_idx, :]
        
        blk_prod = (K_big[:x_dim * T_half, :] @ 
                   fill_per_symm(sp.eye(x_dim * T_half, x_dim * T) - blk_prod, x_dim, T))
        xsm_mat = fill_per_symm(blk_prod, x_dim, T) @ term1_mat  # (x_dim*T) x length(n_list)
        
        ctr = 0
        for n in n_list:
            seq[n]['xsm'] = xsm_mat[:, ctr].reshape(x_dim, T, order='F')
            seq[n]['Vsm'] = Vsm
            seq[n]['VsmGP'] = VsmGP
            ctr += 1

        if get_ll:
            # Compute data likelihood
            val = (-T * logdet_R - logdet_K_big - logdet_M -
                   y_dim * T * np.log(2 * np.pi))
            # LL = (LL + len(n_list) * val - 
            #       np.sum((Rinv @ dif) * dif) +
            #       np.sum((term1_mat.T @ invM) * term1_mat.T))
            # LL = (LL + len(n_list) * val - 
            #       np.sum((Rinv @ dif) * dif) +
            #       np.sum((term1_mat.T * invM) * term1_mat.T))            
            # LL = (LL + len(n_list) * val - 
            #       np.sum((Rinv @ dif) * dif) +
            #       np.sum(term1_mat * (invM @ term1_mat)))   
            LL = (LL + len(n_list) * val - 
                  np.sum((Rinv @ dif) * dif) +
                  np.sum((term1_mat.T @ invM).T * term1_mat.T))            

    if get_ll:
        LL = LL / 2
    else:
        LL = np.nan
    
    return seq, LL