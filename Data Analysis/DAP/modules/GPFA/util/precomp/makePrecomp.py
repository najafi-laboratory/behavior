"""
Direct Python port of makePrecomp.m

Make the precomputation matrices specified by the GPFA algorithm.

John P Cunningham
2009
Python port by GitHub Copilot
"""

import numpy as np
from typing import List, Dict, Any
import warnings


def make_precomp(seq: List[Dict[str, Any]], x_dim: int) -> List[Dict[str, Any]]:
    """
    Make the precomputation matrices specified by the GPFA algorithm.

    INPUTS:
    seq         - The sequence list of inferred latents, etc.
    x_dim       - the dimension of the latent space.
    
    NOTE: All inputs are named sensibly to those in learn_gp_params.
          This code probably should not be called from anywhere but there.

    OUTPUTS:
    precomp     - The precomp list will be updated with the posterior
                  covariance and the other requirements.

    NOTE: We bother with this method because we need this particular 
          matrix sum to be as fast as possible. Thus, no error checking
          is done here as that would add needless computation.
          Instead, the onus is on the caller (which should be 
          learn_gp_params()) to make sure this is called correctly.

    NOTE: The original MATLAB version called MEX code for efficiency.
          In Python, we use vectorized NumPy operations for similar performance.
    """
    
    #######################################################################
    # Setup
    #######################################################################
    T_all = [trial['T'] for trial in seq]
    T_max = max(T_all)
    
    # Create time difference matrices
    time_grid = np.arange(1, T_max + 1)
    T_dif = time_grid[:, np.newaxis] - time_grid[np.newaxis, :]
    
    # Initialize precomp structure for each dimension
    precomp = []
    for i in range(x_dim):
        precomp_i = {
            'absDif': np.abs(T_dif),
            'difSq': T_dif**2,
            'Tall': T_all,
            'Tu': []
        }
        precomp.append(precomp_i)
    
    # Find unique trial lengths
    T_u = sorted(list(set(T_all)))
    
    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        for j, T in enumerate(T_u):
            # Find trials with this length
            n_list = [n for n, trial_T in enumerate(T_all) if trial_T == T]
            
            tu_entry = {
                'nList': n_list,
                'T': T,
                'numTrials': len(n_list),
                'PautoSUM': np.zeros((T, T))
            }
            precomp[i]['Tu'].append(tu_entry)
    
    # At this point the basic precomp is built. The previous steps
    # should be computationally cheap. We now compute PautoSUM,
    # which was initialized as zeros.
    
    ####################################################################
    # Fill out PautoSum using vectorized operations
    ####################################################################
    try:
        # Use vectorized NumPy operations for efficiency
        _fill_pauto_sum_vectorized(precomp, seq, x_dim, T_u)
        
    except Exception as e:
        # Fall back to explicit loops if vectorized version fails
        warnings.warn(f"Vectorized computation failed ({e}), falling back to explicit loops.")
        _fill_pauto_sum_loops(precomp, seq, x_dim, T_u)
    
    return precomp


def _fill_pauto_sum_vectorized(precomp: List[Dict[str, Any]], 
                              seq: List[Dict[str, Any]], 
                              x_dim: int, 
                              T_u: List[int]) -> None:
    """
    Fill PautoSUM using vectorized operations for efficiency.
    
    This function attempts to use NumPy's vectorized operations
    to compute the sum more efficiently than explicit loops.
    """
    
    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        # Loop once for each trial length
        for j, T in enumerate(T_u):
            n_list = precomp[i]['Tu'][j]['nList']
            
            if not n_list:
                continue
            
            # Collect all VsmGP matrices and xsm vectors for this trial length
            Vsm_stack = []
            outer_products = []
            
            for n in n_list:
                # Add covariance matrix
                Vsm_stack.append(seq[n]['VsmGP'][:, :, i])
                
                # Add outer product of mean
                xsm_i = seq[n]['xsm'][i, :T]  # Ensure correct length
                outer_products.append(np.outer(xsm_i, xsm_i))
            
            # Sum all matrices at once
            if Vsm_stack and outer_products:
                precomp[i]['Tu'][j]['PautoSUM'] = (
                    np.sum(Vsm_stack, axis=0) + np.sum(outer_products, axis=0)
                )


def _fill_pauto_sum_loops(precomp: List[Dict[str, Any]], 
                         seq: List[Dict[str, Any]], 
                         x_dim: int, 
                         T_u: List[int]) -> None:
    """
    Fill PautoSUM using explicit loops (fallback implementation).
    
    This is equivalent to the original MATLAB implementation
    without MEX acceleration.
    """
    
    print('Using explicit loops for PautoSUM computation.')
    
    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        # Loop once for each trial length (each of T_u)
        for j, T in enumerate(T_u):
            # Loop once for each trial (each of nList)
            for n in precomp[i]['Tu'][j]['nList']:
                # Add covariance matrix and outer product of mean
                precomp[i]['Tu'][j]['PautoSUM'] += (
                    seq[n]['VsmGP'][:, :, i] + 
                    np.outer(seq[n]['xsm'][i, :], seq[n]['xsm'][i, :])
                )