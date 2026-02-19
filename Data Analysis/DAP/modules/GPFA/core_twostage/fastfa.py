"""
Direct Python port of fastfa.m

Factor analysis and probabilistic PCA.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from modules.GPFA.util.assignopts import assign_opts


def fastfa(X: np.ndarray, z_dim: int, **kwargs) -> Tuple[Dict[str, np.ndarray], List[float]]:
    """
    Factor analysis and probabilistic PCA.

      x_dim: data dimensionality
      z_dim: latent dimensionality
      N:    number of data points

    INPUTS:
    X    - data matrix (x_dim x N)
    z_dim - number of factors

    OUTPUTS:
    est_params - dictionary with fields:
                   L  -- factor loadings (x_dim x z_dim)
                   Ph -- diagonal of uniqueness matrix (x_dim x 1)
                   d  -- data mean (x_dim x 1)
    LL         - log likelihood at each EM iteration

    OPTIONAL ARGUMENTS:
    typ        - 'fa' (default) or 'ppca'
    tol        - stopping criterion for EM (default: 1e-8)
    cyc        - maximum number of EM iterations (default: 1e8)
    min_var_frac - fraction of overall data variance for each observed dimension
                   to set as the private variance floor.  This is used to combat 
                   Heywood cases, where ML parameter learning returns one or more 
                   zero private variances. (default: 0.01)
                   (See Martin & McDonald, Psychometrika, Dec 1975.)
    verbose    - logical that specifies whether to display status messages
                 (default: false)

    Code adapted from ffa.m by Zoubin Ghahramani.

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    typ = 'fa'
    tol = 1e-8
    cyc = int(1e8)
    min_var_frac = 0.01
    verbose = True
    extra_opts = assign_opts(locals(), kwargs)

    np.random.seed(0)
    x_dim, N = X.shape
    
    # Initialization of parameters
    # cX = np.cov(X.T, ddof=0)  # ddof=0 for biased estimate like MATLAB
    cX = np.cov(X, rowvar=True, ddof=0)  # ddof=0 for biased estimate like MATLAB
    if np.linalg.matrix_rank(cX) == x_dim:
        scale = np.exp(2 * np.sum(np.log(np.diag(np.linalg.cholesky(cX)))) / x_dim)
    else:
        # cX may not be full rank because N < x_dim
        print('WARNING in fastfa.py: Data matrix is not full rank.')
        r = np.linalg.matrix_rank(cX)
        e = np.sort(np.linalg.eigvals(cX))[::-1]  # descending order
        scale = np.exp(np.mean(np.log(e[:r])))  # geometric mean
    
    L = np.random.randn(x_dim, z_dim) * np.sqrt(scale / z_dim)
    Ph = np.diag(cX)
    d = np.mean(X, axis=1)

    var_floor = min_var_frac * np.diag(cX)

    I = np.eye(z_dim)
    const = -x_dim / 2 * np.log(2 * np.pi)
    LLi = 0
    LL = []

    for i in range(1, cyc + 1):
        # =======
        # E-step
        # =======
        iPh = np.diag(1.0 / Ph)
        iPhL = iPh @ L
        MM = iPh - iPhL @ np.linalg.solve(I + L.T @ iPhL, iPhL.T)
        beta = L.T @ MM  # z_dim x x_dim
        
        cX_beta = cX @ beta.T  # x_dim x z_dim
        EZZ = I - beta @ L + beta @ cX_beta
        
        # Compute log likelihood
        LL_old = LLi
        ld_M = np.sum(np.log(np.diag(np.linalg.cholesky(MM))))
        LLi = N * const + N * ld_M - 0.5 * N * np.sum(MM * cX)
        if verbose:
            print(f'EM iteration {i:5d} lik {LLi:8.1f} \r', end='', flush=True)
        LL.append(LLi)
        
        # =======
        # M-step
        # =======
        L = cX_beta @ np.linalg.inv(EZZ)
        Ph = np.diag(cX) - np.sum(cX_beta * L, axis=1)
        
        if typ == 'ppca':
            Ph = np.mean(Ph) * np.ones(x_dim)
        elif typ == 'fa':
            # Set minimum private variance
            Ph = np.maximum(var_floor, Ph)
        
        if i <= 2:
            LL_base = LLi
        elif LLi < LL_old:
            print('VIOLATION')
        elif (LLi - LL_base) < (1 + tol) * (LL_old - LL_base):
            break
    
    if verbose:
        print()

    if np.any(Ph == var_floor):
        print('Warning: Private variance floor used for one or more observed dimensions in FA.')

    est_params = {
        'L': L,
        'Ph': Ph,
        'd': d
    }

    return est_params, LL