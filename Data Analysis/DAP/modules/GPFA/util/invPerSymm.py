"""
Direct Python port of invPerSymm.m

Inverts a matrix that is block persymmetric.
"""

import numpy as np
from scipy.sparse import issparse
from typing import Tuple, Optional
from modules.GPFA.util.assignopts import assign_opts
from modules.GPFA.util.fillPerSymm import fill_per_symm
from modules.GPFA.util.logdet import logdet


def inv_per_symm(M: np.ndarray, 
                blk_size: int, 
                **kwargs) -> Tuple[np.ndarray, Optional[float]]:
    """
    Inverts a matrix that is block persymmetric. This function is
    faster than calling np.linalg.inv(M) directly because it only computes the
    top half of inv(M). The bottom half of inv(M) is made up of
    elements from the top half of inv(M).

    WARNING: If the input matrix M is not block persymmetric, no
    error message will be produced and the output of this function will
    not be meaningful.

    INPUTS:
    M           - the block persymmetric matrix to be inverted
                  ((blk_size*T) x (blk_size*T)). Each block is 
                  blk_size x blk_size, arranged in a T x T grid.
    blk_size    - edge length of one block

    OUTPUTS:
    inv_M       - inverse of M ((blk_size*T) x (blk_size*T))
    logdet_M    - log determinant of M (optional)

    OPTIONAL ARGUMENTS:
    off_diag_sparse - logical that specifies whether off-diagonal blocks are
                      sparse (default: False)

    @ 2009 Byron Yu         byronyu@stanford.edu
           John Cunningham  jcunnin@stanford.edu
    """
    
    off_diag_sparse = False  # specify if A12 is sparse
    extra_opts = assign_opts(locals(), kwargs)

    T = M.shape[0] // blk_size
    T_half = int(np.ceil(T / 2))
    mkr = blk_size * T_half

    inv_A11 = np.linalg.inv(M[:mkr, :mkr])
    inv_A11 = (inv_A11 + inv_A11.T) / 2
    
    if off_diag_sparse:
        from scipy.sparse import csc_matrix
        A12 = csc_matrix(M[:mkr, mkr:])
    else:
        A12 = M[:mkr, mkr:]

    term = inv_A11 @ A12
    F22 = M[mkr:, mkr:] - A12.T @ term
    
    res12 = -np.linalg.solve(F22.T, term.T).T  # Equivalent to -term / F22
    res11 = inv_A11 - res12 @ term.T
    res11 = (res11 + res11.T) / 2
    
    # Fill in bottom half of inv_M by picking elements from res11 and res12
    inv_M = fill_per_symm(np.hstack([res11, res12]), blk_size, T)
    
    # Compute log determinant if requested
    logdet_M = None
    if True:  # Always compute since Python doesn't have nargout equivalent
        logdet_M = -logdet(inv_A11) + logdet(F22)
    
    return inv_M, logdet_M