"""
Direct Python port of smoother.m

Gaussian kernel smoothing of data across time.
"""

import numpy as np
from scipy.stats import norm
from typing import Union
from modules.GPFA.util.assignopts import assign_opts


def smoother(y_in: np.ndarray, 
            kern_sd: float, 
            step_size: float, 
            **kwargs) -> np.ndarray:
    """
    Gaussian kernel smoothing of data across time.

    INPUTS:
    y_in        - input data (y_dim x T)
    kern_sd     - standard deviation of Gaussian kernel, in msec
    step_size   - time between 2 consecutive datapoints in y_in, in msec

    OUTPUTS:
    y_out       - smoothed version of y_in (y_dim x T)

    OPTIONAL ARGUMENTS:
    causal      - logical indicating whether temporal smoothing should
                  include only past data (True) or all data (False)
                  (default: False)

    @ 2009 Byron Yu -- byronyu@stanford.edu
    Aug 21, 2011: Added option for causal smoothing
    Python port by GitHub Copilot
    """
    
    causal = False
    extra_opts = assign_opts(locals(), kwargs)

    if kern_sd == 0 or y_in.shape[1] == 1:
        return y_in

    # Filter half length
    # Go 3 standard deviations out
    flt_hl = int(np.ceil(3 * kern_sd / step_size))

    # Length of flt is 2*flt_hl + 1
    x_vals = np.arange(-flt_hl * step_size, (flt_hl + 1) * step_size, step_size)
    flt = norm.pdf(x_vals, 0, kern_sd)

    if causal:
        flt[:flt_hl] = 0

    y_dim, T = y_in.shape
    y_out = np.full((y_dim, T), np.nan)

    # Normalize by sum of filter taps actually used
    nm = np.convolve(flt, np.ones(T), mode='full')
    
    for i in range(y_dim):
        ys = np.convolve(flt, y_in[i, :], mode='full') / nm
        # Cut off edges so that result of convolution is same length 
        # as original data
        y_out[i, :] = ys[flt_hl:-flt_hl]
    
    return y_out