"""
Direct Python port of plotPredErrorVsKernSD.m

Plot prediction error versus smoothing kernel standard deviation.
"""

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from modules.GPFA.util.assignopts import assign_opts
from modules.GPFA.util.parseFilename import parse_filename


def plot_pred_error_vs_kern_sd(run_idx: int, x_dim: int, **kwargs) -> Optional[List[Dict[str, Any]]]:
    """
    Plot prediction error versus smoothing kernel standard deviation.

    INPUTS:
    run_idx    - results files will be loaded from mat_results/runXXX, where
                 XXX is run_idx
    x_dim      - state dimensionality to be plotted for all methods

    OUTPUTS:
    method     - list of data structures containing prediction error values shown in plot

    OPTIONAL ARGUMENTS:
    plot_on    - logical that specifies whether or not to display plot (default: True)

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    plot_on = True
    extra_opts = assign_opts(locals(), kwargs)

    all_methods = ['pca', 'ppca', 'fa', 'gpfa']
    
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
        
        method_idx = None
        if P['method'] in all_methods:
            method_idx = all_methods.index(P['method']) + 1  # 1-based to match MATLAB
        
        file_info.append({
            'name': filename,
            'path': file_path,
            'method': P['method'],
            'x_dim': P['x_dim'],
            'cvf': P['cvf'],
            'method_idx': method_idx
        })

    # Apply filtering criteria
    # File has test data
    crit1 = [f['cvf'] > 0 for f in file_info]
    # File uses selected state dimensionality
    crit2 = [f['x_dim'] == x_dim for f in file_info]
    # File can be potentially used for reduced GPFA
    crit3 = [f['x_dim'] > x_dim and f['method_idx'] == 4 for f in file_info]
    
    # Only continue processing files that satisfy criteria
    D_filtered = [f for i, f in enumerate(file_info) 
                  if crit1[i] and (crit2[i] or crit3[i])]

    if not D_filtered:
        print('ERROR: No valid files.  Exiting...')
        return None
    
    # Check if there are any two-stage method files
    two_stage_methods = [f for f in D_filtered 
                        if f['method_idx'] is not None and f['method_idx'] <= 3]
    if not two_stage_methods:
        print('ERROR: No valid two-stage files.  Exiting...')
        return None

    for i in range(len(D_filtered)):
        print(f"Loading {run_dir}/{D_filtered[i]['name']}...")
        with open(D_filtered[i]['path'], 'rb') as f:
            ws = pickle.load(f)
        
        # Compute prediction error
        if D_filtered[i]['method_idx'] is not None and D_filtered[i]['method_idx'] <= 3:
            # PCA, PPCA, FA
            sse = [np.nan] * len(ws['kern'])
            for k in range(len(sse)):
                Ycs = np.concatenate([trial['ycs'] for trial in ws['kern'][k]['seq_test']], axis=1)
                sse[k] = np.sum((Ycs.ravel() - ws['Y_test_raw'].ravel())**2)
            
            D_filtered[i]['kern_sd'] = ws['kern_sd_list']
            D_filtered[i]['sse'] = sse
            D_filtered[i]['num_trials'] = len(ws['kern'][0]['seq_test'])
            
        elif D_filtered[i]['method_idx'] == 4:
            # GPFA
            Y_test_raw = np.concatenate([trial['y'] for trial in ws['seq_test']], axis=1)
            fn = f'ycs_orth_{x_dim:02d}'
            Ycs = np.concatenate([trial[fn] for trial in ws['seq_test']], axis=1)
            D_filtered[i]['sse'] = np.sum((Ycs.ravel() - Y_test_raw.ravel())**2)
            D_filtered[i]['kern_sd'] = np.nan
            D_filtered[i]['num_trials'] = len(ws['seq_test'])

    # Sum prediction error across cross-validation folds
    method = []
    for n in range(1, 5):  # 1-based to match MATLAB indexing
        method_n = {'name': all_methods[n-1].upper()}
        
        Dn = [f for f in D_filtered 
              if f['method_idx'] == n and f['x_dim'] == x_dim]
        
        if not Dn:
            method_n.update({'kern_sd': None, 'sse': None, 'num_trials': None})
            method.append(method_n)
            continue

        # Ensure that same kernSD were used in each fold
        first_kern_sd = Dn[0]['kern_sd']
        for f in Dn:
            if not np.array_equal(f['kern_sd'], first_kern_sd, equal_nan=True):
                print('ERROR: kernSD used in each cross-validation fold')
                print('must be identical.  Exiting...')
                return None
        
        method_n['kern_sd'] = Dn[0]['kern_sd']
        
        # Sum SSE across folds
        if isinstance(Dn[0]['sse'], list):
            # For methods with multiple kernel SDs
            sse_arrays = np.array([f['sse'] for f in Dn])
            method_n['sse'] = np.sum(sse_arrays, axis=0).tolist()
        else:
            # For single values
            method_n['sse'] = sum([f['sse'] for f in Dn])
        
        method_n['num_trials'] = sum([f['num_trials'] for f in Dn])
        method.append(method_n)

    # Reduced GPFA (based on GPFA files with largest x_dim)
    Dn = [f for f in D_filtered if f['method_idx'] == 4]
    if Dn:
        d_list = [f['x_dim'] for f in Dn]
        max_d = max(d_list)
        Dnn = [f for f in Dn if f['x_dim'] == max_d]
        
        method.append({
            'name': 'Reduced GPFA',
            'kern_sd': np.nan,
            'sse': sum([f['sse'] for f in Dnn]),
            'num_trials': sum([f['num_trials'] for f in Dnn])
        })

    # Check that number of trials is consistent
    num_trials_all = [m['num_trials'] for m in method if m['num_trials'] is not None]
    if len(set(num_trials_all)) != 1:
        print('ERROR: Number of test trials must be the same across')
        print('all methods.  Exiting...')
        return None

    # GPFA prediction error does not depend on kernSD, but put in same
    # format as the other methods for ease of plotting.
    kern_sd_values = []
    for m in method:
        if m['kern_sd'] is not None and not np.isnan(m['kern_sd']).all():
            if isinstance(m['kern_sd'], list):
                kern_sd_values.extend(m['kern_sd'])
            else:
                kern_sd_values.append(m['kern_sd'])
    
    if kern_sd_values:
        min_kern_sd = min(kern_sd_values)
        max_kern_sd = max(kern_sd_values)
        
        for m in method:
            if m['kern_sd'] is None or np.isnan(m['kern_sd']).all():
                m['kern_sd'] = [min_kern_sd, max_kern_sd]
                if isinstance(m['sse'], (int, float)):
                    m['sse'] = [m['sse'], m['sse']]

    # =========
    # Plotting
    # =========
    if plot_on:
        col = ['r--', 'r', 'g', 'k--', 'k']
        
        plt.figure()
        lgnd = []
        
        for n in range(len(method)):
            if (method[n]['kern_sd'] is not None and 
                method[n]['sse'] is not None):
                plt.plot(method[n]['kern_sd'], method[n]['sse'], col[n])
                lgnd.append(method[n]['name'])
        
        plt.legend(lgnd)
        plt.title(f'State dimensionality = {x_dim}', fontsize=12)
        plt.xlabel('Kernel width (ms)', fontsize=14)
        plt.ylabel('Prediction error', fontsize=14)
        plt.show()

    return method