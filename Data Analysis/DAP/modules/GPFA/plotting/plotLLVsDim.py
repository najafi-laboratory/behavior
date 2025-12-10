"""
Direct Python port of plotLLVsDim.m

Plot cross-validated data log-likelihood versus state dimensionality for GPFA.
"""

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from modules.GPFA.util.assignopts import assign_opts
from modules.GPFA.util.parseFilename import parse_filename


def plot_ll_vs_dim(run_idx: int, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Plot cross-validated data log-likelihood versus state dimensionality for GPFA.

    INPUTS:
    run_idx - results files will be loaded from mat_results/runXXX, where
              XXX is run_idx

    OUTPUTS:
    res     - data structure containing log-likelihood values shown in plot

    OPTIONAL ARGUMENTS:
    plot_on - logical that specifies whether or not to display plot (default: True)

    @ 2013 Byron Yu -- byronyu@cmu.edu
    """
    
    plot_on = True
    extra_opts = assign_opts(locals(), kwargs)

    run_dir = f'mat_results/run{run_idx:03d}'
    if not os.path.isdir(run_dir):
        print(f'ERROR: {run_dir} does not exist.  Exiting...')
        return None
    else:
        D = glob.glob(os.path.join(run_dir, '*.pkl'))

    if not D:
        print('ERROR: No valid files.  Exiting...')
        return None

    file_info = []
    for file_path in D:
        filename = os.path.basename(file_path)
        P = parse_filename(filename)
        
        file_info.append({
            'name': filename,
            'path': file_path,
            'method': P['method'],
            'x_dim': P['x_dim'],
            'cvf': P['cvf']
        })

    # Only continue processing GPFA files that have test trials
    D_filtered = [f for f in file_info 
                  if f['method'] == 'gpfa' and f['cvf'] > 0]

    if not D_filtered:
        print('ERROR: No valid files.  Exiting...')
        return None

    for i in range(len(D_filtered)):
        print(f"Loading {run_dir}/{D_filtered[i]['name']}...")
        with open(D_filtered[i]['path'], 'rb') as f:
            ws = pickle.load(f)
        
        D_filtered[i]['LL_test'] = ws['LL_test']
        D_filtered[i]['num_trials'] = len(ws['seq_test'])

    res = {}
    res['name'] = 'gpfa'
    res['x_dim'] = sorted(list(set([f['x_dim'] for f in D_filtered])))

    # Do for each unique state dimensionality
    res['LL_test'] = []
    res['num_trials'] = []
    
    for p in range(len(res['x_dim'])):
        Dp = [f for f in D_filtered if f['x_dim'] == res['x_dim'][p]]
        
        # Sum across cross-validation folds
        res['LL_test'].append(sum([f['LL_test'] for f in Dp]))
        res['num_trials'].append(sum([f['num_trials'] for f in Dp]))

    if len(set(res['num_trials'])) != 1:
        print('ERROR: Number of test trials must be the same across')
        print('all state dimensionalities.  Exiting...')
        return None

    # =========
    # Plotting
    # =========
    if plot_on:
        col = 'k'
        
        plt.figure()
        plt.plot(res['x_dim'], res['LL_test'], col)
        plt.title('GPFA', fontsize=12)
        plt.xlabel('State dimensionality', fontsize=14)
        plt.ylabel('Cross-validated data log-likelihood', fontsize=14)
        plt.show()

    return res