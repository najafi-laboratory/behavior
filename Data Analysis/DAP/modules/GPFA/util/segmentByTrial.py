"""
Direct Python port of segmentByTrial.m

Segment and store data by trial.
"""

import numpy as np
from typing import List, Dict, Any


def segment_by_trial(seq: List[Dict[str, Any]], 
                    X: np.ndarray, 
                    fn: str) -> List[Dict[str, Any]]:
    """
    Segment and store data by trial.
  
    INPUTS:
    seq        - data structure that has field 'T', the number of timesteps
    X          - data to be segmented 
                 (any dimensionality x total number of timesteps)
    fn         - new field name of seq where segments of X are stored

    OUTPUTS:
    seq        - data structure with new field 'fn'

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    if sum([trial['T'] for trial in seq]) != X.shape[1]:
        print('Error: size of X incorrect.')
        return seq
    
    ctr = 0
    for n in range(len(seq)):
        T = seq[n]['T']
        idx_start = ctr
        idx_end = ctr + T
        seq[n][fn] = X[:, idx_start:idx_end]
        
        ctr = ctr + T
    
    return seq