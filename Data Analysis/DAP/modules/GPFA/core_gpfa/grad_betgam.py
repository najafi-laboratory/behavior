"""
Direct Python port of grad_betgam.m

Gradient computation for GP timescale optimization.
This function is called by minimize.py.
"""

import numpy as np
from typing import Tuple, Dict, Any
from .inv_toeplitz import inv_toeplitz


def grad_betgam(p: np.ndarray, precomp: Dict[str, Any], const: Dict[str, Any]) -> Tuple[float, float]:
    """
    Gradient computation for GP timescale optimization.
    This function is called by minimize.py.

    INPUTS:
    p           - variable with respect to which optimization is performed,
                  where p = log(1 / timescale ^2)
    precomp     - structure containing precomputations

    OUTPUTS:
    f           - value of objective function E[log P({x},{y})] at p
    df          - gradient at p    

    @ 2009 Byron Yu         byronyu@stanford.edu
           John Cunningham  jcunnin@stanford.edu
    """
    
    T_all = precomp['Tall']
    T_max = np.max(T_all)
    
    temp = (1 - const['eps']) * np.exp(-np.exp(p[0]) / 2 * precomp['difSq'])  # T_max x T_max
    K_max = temp + const['eps'] * np.eye(T_max)
    dKdgamma_max = -0.5 * temp * precomp['difSq']
    
    dEdgamma = 0
    f = 0
    
    for j in range(len(precomp['Tu'])):
        T = precomp['Tu'][j]['T']
        T_half = int(np.ceil(T / 2))
        
        Kinv, logdet_K = inv_toeplitz(K_max[:T, :T])
        
        KinvM = Kinv[:T_half, :] @ dKdgamma_max[:T, :T]  # T_half x T
        KinvMKinv = (KinvM @ Kinv).T  # T_half x T
        
        dg_KinvM = np.diag(KinvM)
        tr_KinvM = 2 * np.sum(dg_KinvM) - (T % 2) * dg_KinvM[-1]
        
        mkr = int(np.ceil(0.5 * T**2))
        
        dEdgamma = (dEdgamma - 0.5 * precomp['Tu'][j]['numTrials'] * tr_KinvM +
                   0.5 * precomp['Tu'][j]['PautoSUM'][:mkr] @ KinvMKinv[:mkr].ravel() +
                   0.5 * precomp['Tu'][j]['PautoSUM'][::-1][:T**2-mkr] @ KinvMKinv[:T**2-mkr].ravel())
        
        f = (f - 0.5 * precomp['Tu'][j]['numTrials'] * logdet_K -
             0.5 * precomp['Tu'][j]['PautoSUM'].ravel() @ Kinv.ravel())
    
    f = -f
    # exp(p) is needed because we're computing gradients with
    # respect to log(gamma), rather than gamma
    df = -dEdgamma * np.exp(p[0])
    
    return f, df