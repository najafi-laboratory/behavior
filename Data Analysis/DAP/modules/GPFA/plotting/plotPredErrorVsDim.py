"""
Direct Python port of plotPredErrorVsDim.m

Plot prediction error versus state dimensionality.
"""

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from modules.GPFA.util.assignopts import assign_opts
from modules.GPFA.util.parseFilename import parse_filename


def plot_pred_error_vs_dim(run_idx: int, kern_sd: float, **kwargs) -> Optional[List[Dict[str, Any]]]:
    """
    Plot prediction error versus state dimensionality.

    INPUTS:
    run_idx    - results files will be loaded from mat_results/runXXX, where
                 XXX is run_idx
    kern_sd    - smoothing kernel standard deviation to use for two-stage methods

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
            method_idx = all_methods.index(P['method'])
        
        file_info.append({
            'name': filename,
            'path': file_path,
            'method': P['method'],
            'x_dim': P['x_dim'],
            'cvf': P['cvf'],
            'method_idx': method_idx
        })

    # Only continue processing files that have test trials
    D_filtered = [f for f in file_info if f['cvf'] > 0]

    if not D_filtered:
        print('ERROR: No valid files.  Exiting...')
        return None
    
    for i in range(len(D_filtered)):
        print(f"Loading {run_dir}/{D_filtered[i]['name']}...")
        with open(D_filtered[i]['path'], 'rb') as f:
            ws = pickle.load(f)
        
        # Check if selected kern_sd has been run
        if 'kern_sd_list' in ws:
            # PCA, PPCA, FA
            kern_sd_exists = kern_sd in ws['kern_sd_list']
            if kern_sd_exists:
                kidx = ws['kern_sd_list'].index(kern_sd)
        else:
            # GPFA
            kern_sd_exists = True
        
        D_filtered[i]['is_valid'] = False
        
        if kern_sd_exists:
            D_filtered[i]['is_valid'] = True
            
            # Compute prediction error
            if D_filtered[i]['method_idx'] is not None and D_filtered[i]['method_idx'] <= 2:
                # PCA, PPCA, FA
                Ycs = np.concatenate([trial['ycs'] for trial in ws['kern'][kidx]['seq_test']], axis=1)
                Y_test_raw = ws['Y_test_raw']
                D_filtered[i]['sse'] = np.sum((Ycs.ravel() - Y_test_raw.ravel())**2)
                D_filtered[i]['num_trials'] = len(ws['kern'][kidx]['seq_test'])
                
            elif D_filtered[i]['method_idx'] == 3:
                # GPFA
                Y_test_raw = np.concatenate([trial['y'] for trial in ws['seq_test']], axis=1)
                sse_orth = []
                for p in range(1, D_filtered[i]['x_dim'] + 1):
                    fn = f'ycs_orth_{p:02d}'
                    Ycs = np.concatenate([trial[fn] for trial in ws['seq_test']], axis=1)
                    sse_orth.append(np.sum((Ycs.ravel() - Y_test_raw.ravel())**2))
                
                D_filtered[i]['sse_orth'] = sse_orth
                D_filtered[i]['sse'] = sse_orth[-1]  # Last element
                D_filtered[i]['num_trials'] = len(ws['seq_test'])

    D_valid = [f for f in D_filtered if f['is_valid']]

    if not D_valid:
        print('ERROR: No valid files.  Exiting...')
        return None

    # Sum prediction error across cross-validation folds
    method = []
    for n in range(4):
        Dn = [f for f in D_valid if f['method_idx'] == n]
        
        if Dn:  # If there are files for this method
            method_n = {
                'name': all_methods[n].upper(),
                'x_dim': sorted(list(set([f['x_dim'] for f in Dn]))),
                'sse': [],
                'num_trials': []
            }
            
            # Do for each unique state dimensionality
            for p in range(len(method_n['x_dim'])):
                Dnn = [f for f in Dn if f['x_dim'] == method_n['x_dim'][p]]
                
                # Sum across cross-validation folds
                method_n['sse'].append(sum([f['sse'] for f in Dnn]))
                method_n['num_trials'].append(sum([f['num_trials'] for f in Dnn]))
            
            method.append(method_n)
        else:
            method.append({'name': all_methods[n].upper(), 'x_dim': [], 'sse': [], 'num_trials': []})

    # Reduced GPFA (based on GPFA files with largest x_dim)
    Dn = [f for f in D_valid if f['method_idx'] == 3]

    if Dn:
        d_list = [f['x_dim'] for f in Dn]
        max_d = max(d_list)
        Dnn = [f for f in Dn if f['x_dim'] == max_d]
        
        # Stack sse_orth arrays and sum across folds
        sse_orth_all = np.array([f['sse_orth'] for f in Dnn])
        
        method.append({
            'name': 'Reduced GPFA',
            'x_dim': list(range(1, max_d + 1)),
            'sse': np.sum(sse_orth_all, axis=0).tolist(),
            'num_trials': [sum([f['num_trials'] for f in Dnn])] * max_d
        })

    # Check that number of trials is consistent
    num_trials_all = []
    for m in method:
        if m['num_trials']:
            num_trials_all.extend(m['num_trials'])
    
    if len(set(num_trials_all)) != 1:
        print('ERROR: Number of test trials must be the same across')
        print('all methods and state dimensionalities.  Exiting...')
        return None

    # =========
    # Plotting
    # =========
    if plot_on:
        col = ['r--', 'r', 'g', 'k--', 'k']
        
        plt.figure()
        lgnd = []
        
        for n in range(len(method)):
            if method[n]['x_dim']:  # If not empty
                plt.plot(method[n]['x_dim'], method[n]['sse'], col[n])
                lgnd.append(method[n]['name'])
        
        plt.legend(lgnd)
        plt.title(f'For two-stage methods, kernel width = {kern_sd} ms', fontsize=12)
        plt.xlabel('State dimensionality', fontsize=14)
        plt.ylabel('Prediction error', fontsize=14)
        plt.show()

    return method