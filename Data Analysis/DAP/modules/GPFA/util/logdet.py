"""
Direct Python port of logdet.m

Compute log(det(A)) where A is positive-definite.
This is faster and more stable than using log(det(A)).
"""

import numpy as np


def logdet(A: np.ndarray) -> float:
    """
    Compute log(det(A)) where A is positive-definite.
    This is faster and more stable than using np.log(np.linalg.det(A)).

    INPUTS:
    A           - positive-definite matrix

    OUTPUTS:
    y           - log determinant of A

    Written by Tom Minka
    (c) Microsoft Corporation. All rights reserved.
    Python port by GitHub Copilot
    """
    
    U = np.linalg.cholesky(A).T  # chol(A) in MATLAB returns upper triangular
    y = 2 * np.sum(np.log(np.diag(U)))
    
    return y