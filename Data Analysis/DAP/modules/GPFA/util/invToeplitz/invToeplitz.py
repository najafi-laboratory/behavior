"""
Direct Python port of invToeplitz.m

Invert a symmetric, real, positive definite Toeplitz matrix
using either np.linalg.inv() or the Trench algorithm.

John P Cunningham
2009
Python port by GitHub Copilot
"""

import numpy as np
from scipy.linalg import solve_toeplitz, toeplitz
from typing import Tuple, Optional
import warnings
from ..logdet import logdet


def inv_toeplitz(T: np.ndarray, run_mode: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Invert a symmetric, real, positive definite Toeplitz matrix.

    INPUTS:
    T         - the positive definite symmetric Toeplitz matrix to be inverted
    run_mode  - OPTIONAL: to force this inversion to happen using a particular method.
                -1: use np.linalg.inv()
                 0: use fast Trench algorithm (Python implementation)
                 1: use vectorized Trench algorithm
                None: automatic selection based on matrix size

    OUTPUTS:
    T_inv     - the inverted matrix, also symmetric (and persymmetric), NOT TOEPLITZ
    ld        - log determinant of T

    NOTE: We bother with this method because we need this particular matrix inversion 
    to be as fast as possible. Thus, no error checking is done here as that would add 
    needless computation. Instead, the onus is on the caller to make sure the matrix 
    is toeplitz and symmetric and real.
    """
    
    n = T.shape[0]
    
    # Automatic determination of run_mode if not specified
    if run_mode is None:
        # Always try optimized Trench first
        try_mode = 0
        # If that fails, use inv() for small matrices, vectorized Trench for large
        if n < 150:
            catch_mode = -1  # inv() is the best failure option
        else:
            catch_mode = 1   # vectorized Trench is the best failure option
    else:
        # If specified, force it
        try_mode = run_mode
        catch_mode = run_mode
    
    # Invert T with the specified method
    try:
        T_inv, ld = _inv_toeplitz_mode(T, try_mode)
    except Exception as e:
        warnings.warn(f"Primary method failed ({e}), trying fallback method.")
        T_inv, ld = _inv_toeplitz_mode(T, catch_mode)
    
    return T_inv, ld


def _inv_toeplitz_mode(T: np.ndarray, run_mode: int) -> Tuple[np.ndarray, float]:
    """
    Subfunction for try-catch block.
    """
    
    n = T.shape[0]
    
    if run_mode == -1:
        ###################################################################
        # Just call NumPy inv()
        ###################################################################
        T_inv = np.linalg.inv(T)
        ld = logdet(T)
        
    elif run_mode == 0:
        ###################################################################
        # Fast Python implementation using scipy's solve_toeplitz
        ###################################################################
        # Extract the first row of the Toeplitz matrix
        c = T[:, 0]  # First column (same as first row for symmetric Toeplitz)
        
        # Create identity matrix
        I = np.eye(n)
        
        # Solve T * T_inv = I column by column using solve_toeplitz
        T_inv = np.zeros((n, n))
        for i in range(n):
            T_inv[:, i] = solve_toeplitz(c, I[:, i])
        
        # Calculate log determinant
        ld = logdet(T)
        
    elif run_mode == 1:
        ###################################################################
        # Vectorized version of Trench algorithm
        ###################################################################
        # Set up the Trench parameters
        r, gam, v = _trench_setup(T)
        
        # Initialize T_inv
        T_inv = np.zeros((n, n))
        
        ###############################
        # Fill out the borders of T_inv
        ###############################
        # The first element
        T_inv[0, 0] = gam
        
        # The first row
        T_inv[0, 1:n] = v[n-2::-1]  # v(n-1:-1:1) in MATLAB
        
        # Use symmetry and persymmetry to fill out border of matrix
        # The first column
        T_inv[1:n, 0] = T_inv[0, 1:n]
        
        # Last column - persymmetric with 1st row
        T_inv[1:n, n-1] = T_inv[0, n-2::-1]
        
        # Last row - persymmetric with 1st column
        T_inv[n-1, 1:n-1] = T_inv[n-2::-1, 0]
        
        ###############################
        # Fill out the interior of T_inv
        ###############################
        for i in range(n-2, (n-1)//2, -1):  # (n-1):-1:(floor((n-1)/2)+1) in MATLAB
            # Convert to 0-based indexing
            end_idx = n - i
            
            # 4.7.5 equation
            idx_range = np.arange(i, end_idx - 1, -1)  # i:-1:(n-i+1) in MATLAB
            
            T_inv[i, idx_range] = (T_inv[n-1-idx_range, n-1-i] + 
                                  (1/gam) * (v[i] * v[idx_range] - 
                                           v[n-1-i] * v[n-1-idx_range]))
            
            # Symmetry
            T_inv[end_idx:i, i] = T_inv[i, end_idx:i]
            
            # Persymmetry
            if i > end_idx:
                T_inv[end_idx, i-1:end_idx-1:-1] = T_inv[end_idx+1:i+1, i]
                
                # Symmetry
                T_inv[end_idx:i, end_idx] = T_inv[end_idx, end_idx:i]
        
        ###############################
        # Renormalize
        ###############################
        T_inv = T_inv / T[0, 0]
        
        ###############################
        # Calculate the log determinant
        ###############################
        ld = logdet(T)
    
    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")
    
    return T_inv, ld


def _trench_setup(T: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Subfunction for Trench setup.
    """
    
    n = T.shape[0]
    
    # Initial setup for Trench algorithm (from Golub book)
    # Normalize the matrix to r0 = 1 w.l.o.g.
    r = T[0, 1:n] / T[0, 0]
    
    # Use scipy's levinson to solve the Yule-Walker equations
    # This replaces the MATLAB levinson function
    from scipy.linalg import solve_toeplitz
    
    # Create the autocorrelation vector for levinson
    autocorr = np.concatenate([[1], r])
    
    # Solve the Yule-Walker equations: T_{n-1} * y = -r
    # Using solve_toeplitz for efficiency
    T_sub = T[0:n-1, 0:n-1]
    rhs = -T[0, 1:n]
    y = np.linalg.solve(T_sub, rhs)
    
    # Calculate the quantity gamma
    gam = 1 / (1 + np.dot(r, y))
    
    # Now calculate v
    v = gam * y[::-1]  # gam * y(n-1:-1:1) in MATLAB
    
    return r, gam, v


def levinson_durbin(r: np.ndarray) -> np.ndarray:
    """
    Solve the Yule-Walker equations using Levinson-Durbin algorithm.
    
    This is a Python implementation of the Levinson-Durbin algorithm
    that replaces MATLAB's levinson function.
    
    INPUTS:
    r - autocorrelation sequence [r0, r1, ..., r_{n-1}]
    
    OUTPUTS:
    a - solution to the Yule-Walker equations
    """
    
    n = len(r) - 1
    a = np.zeros(n + 1)
    a[0] = 1
    
    E = r[0]
    
    for i in range(1, n + 1):
        k = (r[i] - np.sum(a[1:i] * r[i-1:0:-1])) / E
        a[i] = k
        
        for j in range(1, i):
            a[j] = a[j] - k * a[i - j]
        
        E = (1 - k * k) * E
    
    return a