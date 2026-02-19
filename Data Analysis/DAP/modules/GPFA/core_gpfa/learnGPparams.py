"""
Direct Python port of learnGPparams.m

Updates parameters of GP state model given neural trajectories.
"""

import numpy as np
from typing import Dict, Any, List
from modules.GPFA.util.assignopts import assign_opts
from modules.GPFA.util.precomp.makePrecomp import make_precomp
from modules.GPFA.util.minimize import minimize


def learn_gp_params(seq: List[Dict[str, Any]], 
                   params: Dict[str, Any], 
                   **kwargs) -> Dict[str, Any]:
    """
    Updates parameters of GP state model given neural trajectories.

    INPUTS:
    seq         - data structure containing neural trajectories
    params      - current GP state model parameters, which gives starting point
                  for gradient optimization

    OUTPUT:
    res         - updated GP state model parameters

    OPTIONAL ARGUMENTS:
    max_iters   - maximum number of line searches (if >0), maximum number 
                  of function evaluations (if <0), for minimize.py (default:-8)
    verbose     - logical that specifies whether to display status messages
                  (default: false)

    @ 2009 Byron Yu         byronyu@stanford.edu
           John Cunningham  jcunnin@stanford.edu
    """
    
    max_iters = -8  # for minimize.py
    verbose = False
    extra_opts = assign_opts(locals(), kwargs)

    cov_type = params['covType']
    
    if cov_type == 'rbf':
        # If there's more than one type of parameter, put them in the
        # second row of old_params.
        old_params = params['gamma']
        fname = 'grad_betgam'
    elif cov_type == 'tri':
        old_params = params['a']
        fname = 'grad_trislope'
    elif cov_type == 'logexp':
        old_params = params['a']
        fname = 'grad_logexpslope'
    else:
        raise ValueError(f"Unknown covariance type: {cov_type}")
    
    if params['notes']['learnGPNoise']:
        old_params = np.vstack([old_params, params['eps']])
        fname = fname + '_noise'

    x_dim = old_params.shape[1]
    precomp = make_precomp(seq, x_dim)
    
    res = {}
    
    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        const = {}
        
        if cov_type == 'logexp':
            # No constants for 'rbf' or 'tri'
            const['gamma'] = params['gamma']
        
        if not params['notes']['learnGPNoise']:
            const['eps'] = params['eps'][i]

        init_p = np.log(old_params[:, i])

        # This does the heavy lifting
        res_p, res_f, res_iters = minimize(init_p, fname, max_iters, precomp[i], const)
        
        if cov_type == 'rbf':
            if 'gamma' not in res:
                res['gamma'] = np.zeros(x_dim)
            res['gamma'][i] = np.exp(res_p[0])
        elif cov_type == 'tri':
            if 'a' not in res:
                res['a'] = np.zeros(x_dim)
            res['a'][i] = np.exp(res_p[0])
        elif cov_type == 'logexp':
            if 'a' not in res:
                res['a'] = np.zeros(x_dim)
            res['a'][i] = np.exp(res_p[0])
        
        if params['notes']['learnGPNoise']:
            if 'eps' not in res:
                res['eps'] = np.zeros(x_dim)
            res['eps'][i] = np.exp(res_p[1])
        
        if verbose:
            print(f'\nConverged p; xDim:{i}, p:{np.array2string(res_p, precision=3)}')
    
    return res