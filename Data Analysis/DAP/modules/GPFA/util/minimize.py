"""
Direct Python port of minimize.m

Minimize a differentiable multivariate function using conjugate gradients.
"""

import numpy as np
from typing import Callable, Tuple, Any, List, Union


def minimize(X: np.ndarray, 
            f: Union[str, Callable], 
            length: Union[int, List[int]], 
            *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Minimize a differentiable multivariate function.

    INPUTS:
    X           - starting point (D x 1)
    f           - function name (string) or callable that returns function value and gradient
    length      - length of run: if positive, gives maximum number of line searches,
                  if negative, gives maximum allowed number of function evaluations.
                  Can be [length, red] where red is the expected reduction in 
                  function value in the first line-search (defaults to 1.0).
    *args       - additional arguments passed to function f
    **kwargs    - additional keyword arguments passed to function f

    OUTPUTS:
    X           - found solution
    fX          - vector of function values indicating progress made
    i           - number of iterations used

    The Polack-Ribiere flavour of conjugate gradients is used to compute search
    directions, and a line search using quadratic and cubic polynomial
    approximations and the Wolfe-Powell stopping criteria is used.

    Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).
    Python port by GitHub Copilot
    """

    # Constants
    INT = 0.1      # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0      # extrapolate maximum 3 times the current step-size
    MAX = 20       # max 20 function evaluations per line search
    RATIO = 10     # maximum allowed slope ratio
    SIG = 0.1      # maximum allowed absolute ratio between previous and new slopes
    RHO = SIG / 2  # minimum allowed fraction of the expected

    # Handle length parameter
    if isinstance(length, (list, tuple)) and len(length) == 2:
        red = length[1]
        length = length[0]
    else:
        red = 1.0

    if length > 0:
        S = 'Linesearch'
    else:
        S = 'Function evaluation'

    # Handle function input
    if isinstance(f, str):
        # Import function by name from appropriate module
        if f in globals():
            func = globals()[f]
        elif f in kwargs:
            func = kwargs[f]
        else:
            # Try to import from grad modules
            try:
                if f == 'grad_betgam':
                    from .grad_betgam import grad_betgam
                    func = grad_betgam
                elif f == 'grad_trislope':
                    from .grad_trislope import grad_trislope
                    func = grad_trislope
                elif f == 'grad_logexpslope':
                    from .grad_logexpslope import grad_logexpslope
                    func = grad_logexpslope
                elif f.endswith('_noise'):
                    # Handle noise variants
                    base_name = f.replace('_noise', '')
                    if base_name == 'grad_betgam':
                        from .grad_betgam_noise import grad_betgam_noise
                        func = grad_betgam_noise
                    elif base_name == 'grad_trislope':
                        from .grad_trislope_noise import grad_trislope_noise
                        func = grad_trislope_noise
                    elif base_name == 'grad_logexpslope':
                        from .grad_logexpslope_noise import grad_logexpslope_noise
                        func = grad_logexpslope_noise
                    else:
                        raise ValueError(f"Unknown function: {f}")
                else:
                    raise ValueError(f"Unknown function: {f}")
            except ImportError:
                raise ValueError(f"Cannot import function: {f}")
    else:
        func = f

    i = 0                               # zero the run length counter
    ls_failed = 0                      # no previous line search has failed
    
    f0, df0 = func(X, *args, **kwargs)  # get function value and gradient
    fX = np.array([f0])
    i = i + (length < 0)               # count epochs
    
    s = -df0                           # initial search direction (steepest)
    d0 = -s.T @ s                      # initial slope
    x3 = red / (1 - d0)               # initial step is red/(|s|+1)

    while i < abs(length):             # while not finished
        i = i + (length > 0)           # count iterations

        X0 = X.copy()                  # make a copy of current values
        F0 = f0
        dF0 = df0.copy()
        
        if length > 0:
            M = MAX
        else:
            M = min(MAX, -length - i)

        while True:                    # keep extrapolating as long as necessary
            x2 = 0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0.copy()
            success = 0
            
            while not success and M > 0:
                try:
                    M = M - 1
                    i = i + (length < 0)     # count epochs
                    f3, df3 = func(X + x3 * s, *args, **kwargs)
                    if (np.isnan(f3) or np.isinf(f3) or 
                        np.any(np.isnan(df3)) or np.any(np.isinf(df3))):
                        raise ValueError("NaN or Inf encountered")
                    success = 1
                except:                      # catch any error which occurred in f
                    x3 = (x2 + x3) / 2      # bisect and try again

            if f3 < F0:                     # keep best values
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3.copy()

            d3 = df3.T @ s                  # new slope
            
            # are we done extrapolating?
            if d3 > SIG * d0 or f3 > f0 + x3 * RHO * d0 or M == 0:
                break

            x1 = x2                         # move point 2 to point 1
            f1 = f2
            d1 = d2
            x2 = x3                         # move point 3 to point 2
            f2 = f3
            d2 = d3
            
            # make cubic extrapolation
            A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
            B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
            
            # numerical error possible, ok!
            x3 = x1 - d1 * (x2 - x1)**2 / (B + np.sqrt(B * B - A * d1 * (x2 - x1)))
            
            # num prob | wrong sign?
            if (not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0):
                x3 = x2 * EXT              # extrapolate maximum amount
            elif x3 > x2 * EXT:            # new point beyond extrapolation limit?
                x3 = x2 * EXT              # extrapolate maximum amount
            elif x3 < x2 + INT * (x2 - x1):  # new point too close to previous point?
                x3 = x2 + INT * (x2 - x1)

        # end extrapolation

        # keep interpolating
        while (abs(d3) > -SIG * d0 or f3 > f0 + x3 * RHO * d0) and M > 0:
            if d3 > 0 or f3 > f0 + x3 * RHO * d0:  # choose subinterval
                x4 = x3                     # move point 3 to point 4
                f4 = f3
                d4 = d3
            else:
                x2 = x3                     # move point 3 to point 2
                f2 = f3
                d2 = d3

            if f4 > f0:
                # quadratic interpolation
                x3 = x2 - (0.5 * d2 * (x4 - x2)**2) / (f4 - f2 - d2 * (x4 - x2))
            else:
                # cubic interpolation
                A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)
                B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                x3 = x2 + (np.sqrt(B * B - A * d2 * (x4 - x2)**2) - B) / A

            if np.isnan(x3) or np.isinf(x3):
                x3 = (x2 + x4) / 2          # if numerical problem then bisect

            # don't accept too close
            x3 = max(min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 - x2))
            
            f3, df3 = func(X + x3 * s, *args, **kwargs)
            
            if f3 < F0:                     # keep best values
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3.copy()

            M = M - 1
            i = i + (length < 0)            # count epochs
            d3 = df3.T @ s                  # new slope

        # end interpolation

        if abs(d3) < -SIG * d0 and f3 < f0 + x3 * RHO * d0:  # if line search succeeded
            X = X + x3 * s                  # update variables
            f0 = f3
            fX = np.append(fX, f0)
            
            # Polack-Ribiere CG direction
            s = (df3.T @ df3 - df0.T @ df3) / (df0.T @ df0) * s - df3
            df0 = df3.copy()                # swap derivatives
            d3 = d0
            d0 = df0.T @ s
            
            if d0 > 0:                      # new slope must be negative
                s = -df0                    # otherwise use steepest direction
                d0 = -s.T @ s

            x3 = x3 * min(RATIO, d3 / (d0 - np.finfo(float).tiny))  # slope ratio but max RATIO
            ls_failed = 0                   # this line search did not fail
        else:
            X = X0                          # restore best point so far
            f0 = F0
            df0 = dF0.copy()
            
            if ls_failed or i > abs(length):  # line search failed twice in a row
                break                       # or we ran out of time, so we give up

            s = -df0                        # try steepest
            d0 = -s.T @ s
            x3 = 1 / (1 - d0)
            ls_failed = 1                   # this line search failed

    return X, fX, i