"""
Direct Python port of invToeplitzFast.m

This function provides fast Toeplitz matrix inversion using optimized algorithms.
In the original MATLAB version, this was a wrapper for C-MEX code.
In Python, we use optimized NumPy/SciPy implementations.

John P Cunningham
2009
Python port by GitHub Copilot
"""

import numpy as np
from typing import Tuple
from ..logdet import logdet
from .inv_toeplitz import inv_toeplitz


def inv_toeplitz_fast(T: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fast inversion of a positive definite (symmetric) Toeplitz matrix.
    
    This function is equivalent to the MATLAB invToeplitzFast.m, which was
    a wrapper for the MEX function invToeplitzFastZohar. In Python, we use
    the optimized Trench algorithm implementation from inv_toeplitz.
    
    The algorithm inverts a positive definite (symmetric) Toeplitz matrix 
    in O(n^2) time, which is considerably better than the O(n^3) that 
    np.linalg.inv() offers. This follows the Trench algorithm implementation 
    of Zohar 1969.
    
    This function also computes the log determinant, which is calculated 
    essentially for free as part of the calculation of the inverse.
    
    INPUTS:
    T    - the positive definite (symmetric) Toeplitz matrix, which
           does NOT need to be scaled to be 1 on the main diagonal.
    
    OUTPUTS:
    T_inv - the inverse of T
    ld    - the log determinant of T, NOT T_inv.
    
    NOTE: This code is used to speed up the Toeplitz inversion as much
    as possible. Accordingly, no error checking is done. The onus is
    on the caller (which should be inv_toeplitz.py) to pass the correct args.
    
    NOTE: Whenever possible, do not actually invert a matrix. This code is 
    written just in case you really need to do so. Otherwise, for example
    if you just want to solve inv(T)*x for some vector x, you are better off
    using a fast inversion method, like PCG with fast matrix multiplication,
    which could be something like an FFT method for the Toeplitz matrix.
    """
    
    # Use the optimized Trench algorithm (mode 1) from inv_toeplitz
    # This provides the same O(n^2) performance as the original MEX code
    T_inv, ld = inv_toeplitz(T, run_mode=1)
    
    return T_inv, ld


def inv_toeplitz_fast_zohar(T: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Python implementation of the Zohar (1969) Trench algorithm.
    
    This is a direct implementation of the algorithm from Zohar's 1969 paper,
    equivalent to the original invToeplitzFastZohar MEX function.
    
    INPUTS:
    T - symmetric positive definite Toeplitz matrix
    
    OUTPUTS:
    T_inv - inverse of T
    ld    - log determinant of T
    """
    
    n = T.shape[0]
    
    if n == 1:
        T_inv = 1.0 / T
        ld = np.log(T[0, 0])
        return T_inv, ld
    
    # Step 1: Extract the first row (same as first column for symmetric)
    r = T[0, :] / T[0, 0]  # Normalize by T[0,0]
    
    # Step 2: Solve the Yule-Walker equations using Levinson-Durbin
    # T_{n-1} * a = -r[1:n-1]
    a = _levinson_durbin_zohar(r[1:])
    
    # Step 3: Calculate gamma
    gamma = 1.0 / (1.0 + np.dot(r[1:], a))
    
    # Step 4: Build the inverse using Zohar's formula
    T_inv = np.zeros((n, n))
    
    # Fill the first row and column
    T_inv[0, 0] = gamma
    if n > 1:
        T_inv[0, 1:] = gamma * a[::-1]  # Reverse order
        T_inv[1:, 0] = T_inv[0, 1:]     # Symmetry
    
    # Fill the last row and column using persymmetry
    if n > 1:
        T_inv[-1, :-1] = T_inv[0, 1:][::-1]  # Persymmetric
        T_inv[:-1, -1] = T_inv[-1, :-1]      # Symmetry
    
    # Fill the interior using the Trench recurrence
    for i in range(1, n-1):
        for j in range(i, n-1-i):
            # Zohar's formula (equation 4.7.5)
            T_inv[i, j] = (T_inv[n-1-j, n-1-i] + 
                          (1/gamma) * (T_inv[0, i] * T_inv[0, j] - 
                                     T_inv[0, n-1-i] * T_inv[0, n-1-j]))
            
            # Fill by symmetry
            T_inv[j, i] = T_inv[i, j]
    
    # Step 5: Rescale by the original T[0,0]
    T_inv = T_inv / T[0, 0]
    
    # Step 6: Calculate log determinant
    ld = logdet(T)
    
    return T_inv, ld


def _levinson_durbin_zohar(r: np.ndarray) -> np.ndarray:
    """
    Levinson-Durbin algorithm for solving Yule-Walker equations.
    
    Solves: T * a = -r where T is Toeplitz
    
    INPUTS:
    r - autocorrelation coefficients [r1, r2, ..., rn]
    
    OUTPUTS:
    a - solution vector
    """
    
    n = len(r)
    if n == 0:
        return np.array([])
    
    a = np.zeros(n)
    E = 1.0  # Error term
    
    for i in range(n):
        # Calculate reflection coefficient
        k = (-r[i] - np.sum(a[:i] * r[i-1::-1])) / E
        
        # Update coefficients
        a[i] = k
        for j in range(i):
            a[j] = a[j] + k * a[i-1-j]
        
        # Update error
        E = E * (1 - k * k)
    
    return a