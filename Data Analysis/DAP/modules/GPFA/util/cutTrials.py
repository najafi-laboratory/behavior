"""
Direct Python port of cutTrials.m

Extracts trial segments that are all of the same length.
"""

import numpy as np
from typing import List, Dict, Any
from modules.GPFA.util.assignopts import assign_opts


def cut_trials(seq_in: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """
    Extracts trial segments that are all of the same length. Uses
    overlapping segments if trial length is not integer multiple
    of segment length. Ignores trials with length shorter than 
    one segment length.

    INPUTS:
    seq_in      - data structure, whose nth entry (corresponding to
                  the nth experimental trial) has fields
                    trial_id      -- unique trial identifier
                    T (1 x 1)    -- number of timesteps in trial
                    y (y_dim x T) -- neural data

    OUTPUTS:
    seq_out     - data structure, whose nth entry (corresponding to
                  the nth segment) has fields
                    trial_id      -- identifier of trial from which 
                                    segment was taken
                    seg_id        -- segment identifier within trial
                    T (1 x 1)    -- number of timesteps in segment
                    y (y_dim x T) -- neural data

    OPTIONAL ARGUMENTS:
    seg_length  - length of segments to extract, in number of timesteps.
                  If infinite, entire trials are extracted, i.e., no 
                  segmenting. (default: 20)

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    seg_length = 20  # number of timesteps in each segment
    extra_opts = assign_opts(locals(), kwargs)

    if np.isinf(seg_length):
        return seq_in

    seq_out = []
    
    for n in range(len(seq_in)):
        T = seq_in[n]['T']
        
        # Skip trials that are shorter than seg_length
        if T < seg_length:
            print(f'Warning: trial_id {seq_in[n]["trialId"]:4d} shorter than one seg_length...skipping')
            continue
        
        num_seg = int(np.ceil(T / seg_length))
        
        if num_seg == 1:
            cum_ol = [0]
        else:
            total_ol = (seg_length * num_seg) - T
            probs = np.ones(num_seg - 1) / (num_seg - 1)
            # multinomial random sampling is very sensitive to sum(probs) being even slightly
            # away from 1 due to floating point round-off.
            probs[-1] = 1 - np.sum(probs[:-1])
            rand_ol = np.random.multinomial(total_ol, probs)
            cum_ol = np.concatenate([[0], np.cumsum(rand_ol)])

        seg = {
            'trialId': seq_in[n]['trialId'],
            'T': seg_length
        }
        
        for s in range(num_seg):
            t_start = -cum_ol[s] + seg_length * s  # Convert to 0-based indexing
            
            seg_copy = seg.copy()
            seg_copy['segId'] = s + 1  # Keep 1-based for compatibility
            seg_copy['y'] = seq_in[n]['y'][:, t_start:(t_start + seg_length)]
            
            seq_out.append(seg_copy)
    
    return seq_out