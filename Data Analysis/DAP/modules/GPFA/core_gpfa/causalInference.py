"""
Direct Python port of causalInference.m

Extracts latent trajectories using only *current and past* neural
activity given GPFA model parameters.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .assign_opts import assign_opts
from .causal_inference_precomp import causal_inference_precomp


def causal_inference(seq: List[Dict[str, Any]], params: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
    """
    Extracts latent trajectories using only *current and past* neural
    activity given GPFA model parameters.

    INPUTS:
    seq         - data structure, whose nth entry (corresponding to the nth 
                  experimental trial) has fields
                    y (yDim x T) -- neural data
                    T (1 x 1)    -- number of timesteps
    params      - GPFA model parameters 
     
    OUTPUT:
    seq         - data structure with new field
                    xfi (xDim x T) -- posterior mean at each timepoint
                                      E[x_t | y_1,...,y_t]

    @ 2011 Byron Yu         byronyu@cmu.edu

    Note: To minimize compute time, this function does not compute the
    posterior covariance of the neural trajectories.  If the posterior
    covariance is desired, call exact_inference_with_ll.py.
    """
    
    extra_opts = assign_opts(locals(), kwargs)

    # This function does most of the heavy lifting
    T_max = max([trial['T'] for trial in seq])
    precomp = causal_inference_precomp(params, T_max, **extra_opts)

    y_dim, x_dim = params['C'].shape
    T_star = len(precomp['filt'])

    # GPFA makes the approximation that x_t does not depend on y_{t-s}
    # for s >= T_star.
    print(f'T_star = {T_star}, T_max = {T_max}')
    print('GPFA is making an approximation if T_star < T_max.')
    
    for n in range(len(seq)):
        T = seq[n]['T']
        xfi = np.full((x_dim, T), np.nan)
        
        # yDim x T
        dif = seq[n]['y'] - params['d'][:, np.newaxis]
        term1 = (precomp['CRinv'] @ dif).reshape(x_dim * T, 1, order='F')
        
        for t in range(min(T, T_star)):
            xfi[:, t] = precomp['filt'][t]['M'] @ term1[:x_dim * (t + 1)]
        
        if T > T_star:
            for t in range(T_star, T):
                idx_start = x_dim * (t - T_star)
                idx_end = x_dim * (t + 1)
                xfi[:, t] = precomp['filt'][-1]['M'] @ term1[idx_start:idx_end]
        
        seq[n]['xfi'] = xfi
    
    return seq