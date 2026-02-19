"""
Direct Python port of orthogonalize.m

Orthonormalize the columns of the loading matrix and 
apply the corresponding linear transform to the latent variables.
"""

import numpy as np
from typing import Tuple


def orthogonalize(X: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthonormalize the columns of the loading matrix and 
    apply the corresponding linear transform to the latent variables.

      y_dim: data dimensionality
      x_dim: latent dimensionality

    INPUTS:
    X        - latent variables (x_dim x T)
    L        - loading matrix (y_dim x x_dim)

    OUTPUTS:
    X_orth   - orthonormalized latent variables (x_dim x T)
    L_orth   - orthonormalized loading matrix (y_dim x x_dim)
    TT       - linear transform applied to latent variables (x_dim x x_dim)

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    x_dim = L.shape[1]
    
    if x_dim == 1:
        TT = np.sqrt(L.T @ L)
        L_orth = L / TT
        X_orth = TT * X
    else:
        UU, DD, VV = np.linalg.svd(L, full_matrices=False)
        # TT is transform matrix
        TT = np.diag(DD) @ VV
        
        L_orth = UU
        X_orth = TT @ X
    
    return X_orth, L_orth, TT