"""
Direct Python port of fillPerSymm.m

Fills in the bottom half of a block persymmetric matrix, given the top half.
"""

import numpy as np
from typing import Optional
from modules.GPFA.util.assignopts import assign_opts


def fill_per_symm(P_in: np.ndarray, 
                  blk_size: int, 
                  T: int, 
                  **kwargs) -> np.ndarray:
    """
    Fills in the bottom half of a block persymmetric matrix, given the top half.

    INPUTS:
    P_in         - top half of block persymmetric matrix
                   (x_dim*T_half) x (x_dim*T), where T_half = ceil(T/2)
    blk_size     - edge length of one block
    T            - number of blocks making up a row of P_in

    OUTPUTS:
    P_out        - full block persymmetric matrix
                   (x_dim*T) x (x_dim*T)

    OPTIONAL ARGUMENTS:
    blk_size_vert - vertical block edge length if blocks are not square.
                    'blk_size' is assumed to be the horizontal block edge
                    length.

    @ 2009 Byron Yu         byronyu@stanford.edu
           John Cunningham  jcunnin@stanford.edu
    """
    
    blk_size_vert = blk_size
    extra_opts = assign_opts(locals(), kwargs)
    
    # Fill in bottom half by doing blockwise fliplr and flipud
    T_half = T // 2  # floor(T/2) in MATLAB
    
    # Create index arrays using broadcasting (equivalent to bsxfun)
    # idx_half: indices for the vertical dimension
    row_indices = np.arange(1, blk_size_vert + 1)[:, np.newaxis]  # (blk_size_vert, 1)
    col_offsets_half = np.arange(T_half - 1, -1, -1) * blk_size_vert  # (T_half,)
    idx_half = row_indices + col_offsets_half  # Broadcasting: (blk_size_vert, T_half)
    
    # idx_full: indices for the horizontal dimension  
    row_indices_full = np.arange(1, blk_size + 1)[:, np.newaxis]  # (blk_size, 1)
    col_offsets_full = np.arange(T - 1, -1, -1) * blk_size  # (T,)
    idx_full = row_indices_full + col_offsets_full  # Broadcasting: (blk_size, T)
    
    # Convert to 0-based indexing for Python
    idx_half_0based = idx_half - 1
    idx_full_0based = idx_full - 1
    
    # Flatten indices for advanced indexing
    idx_half_flat = idx_half_0based.ravel()
    idx_full_flat = idx_full_0based.ravel()
    
    # Create the bottom half by indexing P_in
    bottom_half = P_in[idx_half_flat, :][:, idx_full_flat]
    
    # Concatenate top and bottom halves
    P_out = np.vstack([P_in, bottom_half])
    
    return P_out