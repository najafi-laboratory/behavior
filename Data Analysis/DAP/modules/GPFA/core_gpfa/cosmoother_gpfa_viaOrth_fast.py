"""
Direct Python port of cosmoother_gpfa_viaOrth_fast.m

Performs leave-neuron-out prediction for GPFA. This version takes 
advantage of R being diagonal for computational savings.
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Any
from modules.GPFA.core_gpfa.make_K_big import make_k_big
from modules.GPFA.util.invPerSymm import inv_per_symm
from modules.GPFA.util.fillPerSymm import fill_per_symm
from modules.GPFA.util.orthogonalize import orthogonalize


def cosmoother_gpfa_viaorth_fast(seq: List[Dict[str, Any]], 
                                params: Dict[str, Any], 
                                m_list: List[int]) -> List[Dict[str, Any]]:
    """
    Performs leave-neuron-out prediction for GPFA. This version takes 
    advantage of R being diagonal for computational savings.

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
    
    if not params['notes']['RforceDiagonal']:
        print('ERROR: R must be diagonal to use cosmoother_gpfa_viaorth_fast.')
        return seq

    y_dim, x_dim = params['C'].shape
    Rinv = np.diag(1.0 / np.diag(params['R']))
    CRinv = params['C'].T @ Rinv
    CRinvC = CRinv @ params['C']

    _, Corth, TT = orthogonalize(np.zeros((x_dim, 1)), params['C'])

    T_all = [trial['T'] for trial in seq]
    Tu = np.unique(T_all)

    # Initialize output fields
    for n in range(len(seq)):
        for m in m_list:
            fn = f'ycs_orth_{m:02d}'
            seq[n][fn] = np.full((y_dim, seq[n]['T']), np.nan)

    for j, T in enumerate(Tu):
        T_half = int(np.ceil(T / 2))

        K_big, K_big_inv, logdet_K_big = make_k_big(params, T)
        K_big = sp.csr_matrix(K_big)

        # Create block diagonal matrix
        blah = [CRinvC for _ in range(T)]
        invM = inv_per_symm(K_big_inv + sp.block_diag(blah), x_dim,
                           off_diag_sparse=True)

        # Process all trials with length T
        n_list = [n for n, t in enumerate(T_all) if t == T]
        
        # Concatenate y data for all trials of length T
        y_concat = np.concatenate([seq[n]['y'] for n in n_list], axis=1)
        dif = y_concat - params['d'][:, np.newaxis]  # yDim x sum(T)
        CRinv_dif = CRinv @ dif  # xDim x sum(T)

        for i in range(y_dim):
            # Downdate invM to remove contribution of neuron i
            ci_invM = np.full((T_half, x_dim * T), np.nan)
            ci_invM_ci = np.full((T_half, T), np.nan)
            idx = np.arange(0, x_dim * T + 1, x_dim)
            ci = params['C'][i, :] / np.sqrt(params['R'][i, i])

            for t in range(T_half):
                b_idx = slice(idx[t], idx[t + 1])
                ci_invM[t, :] = ci @ invM[b_idx, :]

            for t in range(T):
                b_idx = slice(idx[t], idx[t + 1])
                ci_invM_ci[:, t] = ci_invM[:, b_idx] @ ci

            ci_invM = fill_per_symm(ci_invM, x_dim, T, blk_size_vert=1)  # T x (xDim*T)
            term = np.linalg.solve(fill_per_symm(ci_invM_ci, 1, T) - np.eye(T), 
                                 ci_invM)  # T x (xDim*T)
            
            invM_mi = invM - ci_invM.T @ term  # (xDim*T) x (xDim*T)

            # Subtract out contribution of neuron i
            CRinvC_mi = CRinvC - np.outer(ci, ci)
            term1_data = CRinv_dif - np.outer(params['C'][i, :] / params['R'][i, i], 
                                            dif[i, :])
            term1_mat = term1_data.reshape(x_dim * T, -1, order='F')  # (xDim*T) x length(nList)

            # Compute blkProd = CRinvC_big * invM efficiently
            # blkProd is block persymmetric, so just compute top half
            blk_prod = np.zeros((x_dim * T_half, x_dim * T))
            idx = np.arange(0, x_dim * T_half + 1, x_dim)

            for t in range(T_half):
                b_idx = slice(idx[t], idx[t + 1])
                blk_prod[b_idx, :] = CRinvC_mi @ invM_mi[b_idx, :]

            blk_prod = (K_big[:x_dim * T_half, :] @ 
                       fill_per_symm(sp.eye(x_dim * T_half, x_dim * T) - blk_prod, 
                                   x_dim, T))
            xsm_mat = fill_per_symm(blk_prod, x_dim, T) @ term1_mat  # (xDim*T) x length(nList)

            ctr = 0
            for n in n_list:
                xorth = TT @ xsm_mat[:, ctr].reshape(x_dim, T, order='F')

                for m in m_list:
                    fn = f'ycs_orth_{m:02d}'
                    seq[n][fn][i, :] = Corth[i, :m] @ xorth[:m, :] + params['d'][i]

                ctr += 1

        print(f'Cross-validation complete for {j+1:3d} of {len(Tu)} trial lengths\r', 
              end='', flush=True)

    print()  # New line after progress updates
    return seq