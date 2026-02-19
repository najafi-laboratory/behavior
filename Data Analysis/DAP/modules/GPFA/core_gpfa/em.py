"""
Direct Python port of em.m

Fits GPFA model parameters using expectation-maximization (EM) algorithm.
"""

import numpy as np
import time
from typing import List, Dict, Any, Tuple
from modules.GPFA.util.assignopts import assign_opts
from modules.GPFA.core_gpfa.exactInferenceWithLL import exact_inference_with_ll
from modules.GPFA.core_gpfa.learnGPparams import learn_gp_params


def em(current_params: Dict[str, Any], 
       seq: List[Dict[str, Any]], 
       **kwargs) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float], List[float]]:
    """
    Fits GPFA model parameters using expectation-maximization (EM) algorithm.

      y_dim: number of neurons
      x_dim: state dimensionality

    INPUTS:
    current_params - GPFA model parameters at which EM algorithm is initialized
                       cov_type (string) -- type of GP covariance ('rbf')
                       gamma (1 x x_dim) -- GP timescales in milliseconds are 
                                           'step_size ./ sqrt(gamma)'
                       eps (1 x x_dim)   -- GP noise variances
                       d (y_dim x 1)     -- observation mean
                       C (y_dim x x_dim)  -- mapping between low- and high-d spaces
                       R (y_dim x y_dim)  -- observation noise covariance
    seq           - training data structure, whose nth entry (corresponding to
                   the nth experimental trial) has fields
                     trial_id      -- unique trial identifier
                     T (1 x 1)    -- number of timesteps
                     y (y_dim x T) -- neural data

    OUTPUTS:
    est_params     - learned GPFA model parameters returned by EM algorithm
                     (same format as current_params)
    seq           - training data structure with new fields
                     xsm (x_dim x T)        -- posterior mean at each timepoint
                     Vsm (x_dim x x_dim x T) -- posterior covariance at each timepoint
                     VsmGP (T x T x x_dim)  -- posterior covariance of each GP
    LL            - data log likelihood after each EM iteration
    iter_time     - computation time for each EM iteration
                   
    OPTIONAL ARGUMENTS:
    em_max_iters    - number of EM iterations to run (default: 500)
    tol           - stopping criterion for EM (default: 1e-8)
    min_var_frac    - fraction of overall data variance for each observed dimension
                     to set as the private variance floor.  This is used to combat
                     Heywood cases, where ML parameter learning returns one or more
                     zero private variances. (default: 0.01)
                     (See Martin & McDonald, Psychometrika, Dec 1975.)
    freq_ll        - data likelihood is computed every freq_ll EM iterations. 
                     freq_ll = 1 means that data likelihood is computed every 
                     iteration. (default: 5)
    verbose       - logical that specifies whether to display status messages
                     (default: false)

    @ 2009 Byron Yu         byronyu@stanford.edu
           John Cunningham  jcunnin@stanford.edu
    """
    
    em_max_iters = 500
    tol = 1e-8
    min_var_frac = 0.01
    verbose = True
    freq_ll = 10
    extra_opts = assign_opts(locals(), kwargs)
    
    N = len(seq)
    T = np.array([trial['T'] for trial in seq])
    y_dim, x_dim = current_params['C'].shape
    LL = []
    LLi = 0
    iter_time = []
    
    # Compute variance floor
    y_all = np.concatenate([trial['y'] for trial in seq], axis=1)
    var_floor = min_var_frac * np.diag(np.cov(y_all.T))
    
    # Loop once for each iteration of EM algorithm
    for i in range(1, em_max_iters + 1):
        if verbose:
            print()
        
        np.random.seed(i)
        t_start = time.time()
        
        print(f'EM iteration {i:3d} of {em_max_iters}', end='')
        if (i % freq_ll == 0) or (i <= 2):
            get_ll = True
        else:
            get_ll = False
        
        # ==== E STEP =====
        if not np.isnan(LLi):
            LL_old = LLi
        
        seq, LLi = exact_inference_with_ll(seq, current_params, get_ll=get_ll)
        LL.append(LLi)
        
        # ==== M STEP ====
        sum_p_auto = np.zeros((x_dim, x_dim))
        for n in range(N):
            sum_p_auto += (np.sum(seq[n]['Vsm'], axis=2) + 
                          seq[n]['xsm'] @ seq[n]['xsm'].T)
        
        Y = np.concatenate([trial['y'] for trial in seq], axis=1)
        Xsm = np.concatenate([trial['xsm'] for trial in seq], axis=1)
        sum_yxtrans = Y @ Xsm.T
        # sum_xall = np.sum(Xsm, axis=1, keepdims=True)
        # sum_yall = np.sum(Y, axis=1, keepdims=True)

        # Alternative if keepdims doesn't work:
        # sum_xall = np.sum(Xsm, axis=1)[:, np.newaxis]
        sum_xall = np.sum(Xsm, axis=1)
        sum_yall = np.sum(Y, axis=1)[:, np.newaxis]        
        
        term = np.block([[sum_p_auto, sum_xall],
                        [sum_xall.T, np.sum(T)]])  # (x_dim+1) x (x_dim+1)
        Cd = np.concatenate([sum_yxtrans, sum_yall], axis=1) @ np.linalg.inv(term)  # y_dim x (x_dim+1)
        
        current_params['C'] = Cd[:, :x_dim]
        current_params['d'] = Cd[:, -1]
        
        # Update R
        if current_params['notes']['RforceDiagonal']:
            sum_yytrans = np.sum(Y * Y, axis=1)
            yd = sum_yall.flatten() * current_params['d']
            term = np.sum((sum_yxtrans - current_params['d'][:, np.newaxis] @ sum_xall.T) * 
                         current_params['C'], axis=1)
            r = (current_params['d']**2 + 
                (sum_yytrans - 2*yd - term) / np.sum(T))
            
            # Set minimum private variance
            r = np.maximum(var_floor, r)
            current_params['R'] = np.diag(r)
        else:
            sum_yytrans = Y @ Y.T
            yd = sum_yall @ current_params['d'][:, np.newaxis].T
            term = ((sum_yxtrans - current_params['d'][:, np.newaxis] @ sum_xall.T) @ 
                   current_params['C'].T)
            R = (current_params['d'][:, np.newaxis] @ current_params['d'][:, np.newaxis].T + 
                (sum_yytrans - yd - yd.T - term) / np.sum(T))
            
            current_params['R'] = (R + R.T) / 2  # ensure symmetry
        
        if current_params['notes']['learnKernelParams']:
            res = learn_gp_params(seq, current_params, verbose=verbose, **extra_opts)
            
            if current_params['covType'] == 'rbf':
                current_params['gamma'] = res['gamma']
            elif current_params['covType'] == 'tri':
                current_params['a'] = res['a']
            elif current_params['covType'] == 'logexp':
                current_params['a'] = res['a']
            
            if current_params['notes']['learnGPNoise']:
                current_params['eps'] = res['eps']
        
        t_end = time.time()
        iter_time.append(t_end - t_start)
        
        # Display the most recent likelihood that was evaluated
        if verbose:
            if get_ll:
                print(f'       lik {LLi:g} ({t_end - t_start:.1f} sec)')
            else:
                print()
        else:
            if get_ll:
                print(f'       lik {LLi:g}\r', end='', flush=True)
            else:
                print('\r', end='', flush=True)
        
        # Verify that likelihood is growing monotonically
        if i <= 2:
            LL_base = LLi
        elif LLi < LL_old:
            print(f'\nError: Data likelihood has decreased from {LL_old:g} to {LLi:g}')
            breakpoint()  # Python equivalent of MATLAB's keyboard
        elif (LLi - LL_base) < (1 + tol) * (LL_old - LL_base):
            break
    
    print()
    
    if len(LL) < em_max_iters:
        print(f'Fitting has converged after {len(LL)} EM iterations.')
    
    if np.any(np.diag(current_params['R']) == var_floor):
        print('Warning: Private variance floor used for one or more observed dimensions in GPFA.')
    
    est_params = current_params.copy()
    
    return est_params, seq, LL, iter_time