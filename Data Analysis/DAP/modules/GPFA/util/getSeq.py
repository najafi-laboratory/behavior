"""
Direct Python port of getSeq.m

Converts 0/1 spike trains into spike counts.
"""

import numpy as np
from typing import List, Dict, Any
from modules.GPFA.util.assignopts import assign_opts


def get_seq(dat: List[Dict[str, Any]], bin_width: int, **kwargs) -> List[Dict[str, Any]]:
    """
    Converts 0/1 spike trains into spike counts.

    INPUTS:
    dat         - list of dictionaries, whose nth entry (corresponding to the nth experimental
                  trial) has fields:
                    'trialId' -- unique trial identifier
                    'spikes'  -- 0/1 matrix of the raw spiking activity across
                                 all neurons. Each row corresponds to a neuron.
                                 Each column corresponds to a 1 msec timestep.
    bin_width   - spike bin width in msec

    OUTPUTS:
    seq         - data structure, whose nth entry (corresponding to
                  the nth experimental trial) has fields:
                    'trialId'      -- unique trial identifier
                    'T' (1 x 1)    -- number of timesteps
                    'y' (y_dim x T) -- neural data

    OPTIONAL ARGUMENTS:
    use_sqrt    - logical specifying whether or not to use square-root transform
                  on spike counts (default: True)

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    use_sqrt = True
    extra_opts = assign_opts(locals(), kwargs)

    seq = []
    
    for n in range(len(dat)):
        y_dim = dat[n]['spikes'].shape[0]
        T = dat[n]['spikes'].shape[1] // bin_width  # floor division
        
        trial_seq = {
            'trialId': dat[n]['trialId'],
            'T': T,
            'y': np.full((y_dim, T), np.nan)
        }
        
        for t in range(T):
            i_start = bin_width * t  # 0-based indexing
            i_end = bin_width * (t + 1)
            
            trial_seq['y'][:, t] = np.sum(dat[n]['spikes'][:, i_start:i_end], axis=1)
        
        if use_sqrt:
            trial_seq['y'] = np.sqrt(trial_seq['y'])
        
        seq.append(trial_seq)
    
    # Remove trials that are shorter than one bin width
    if seq:
        trials_to_keep = [trial['T'] > 0 for trial in seq]
        seq = [trial for i, trial in enumerate(seq) if trials_to_keep[i]]
    
    return seq