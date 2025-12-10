"""
Direct Python port of plotEachDimVsTime.m

Plot each state dimension versus time in a separate panel.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from modules.GPFA.util.assignopts import assign_opts


def plot_each_dim_vs_time(seq: List[Dict[str, Any]], 
                         xspec: str, 
                         bin_width: float, 
                         **kwargs) -> None:
    """
    Plot each state dimension versus time in a separate panel.

    INPUTS:
    seq       - data structure containing extracted trajectories
    xspec     - field name of trajectories in 'seq' to be plotted 
                (e.g., 'xorth' or 'xsm')
    bin_width - spike bin width used when fitting model

    OPTIONAL ARGUMENTS:
    n_plot_max  - maximum number of trials to plot (default: 20)
    red_trials  - vector of trial_ids whose trajectories are plotted in red
                  (default: [])
    n_cols      - number of subplot columns (default: 4)

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    n_plot_max = 20
    red_trials = []
    n_cols = 4
    extra_opts = assign_opts(locals(), kwargs)

    fig = plt.figure()
    pos = fig.get_size_inches()
    fig.set_size_inches(2 * pos[0], pos[1])

    X_all = np.concatenate([trial[xspec] for trial in seq], axis=1)
    x_max = np.ceil(10 * np.max(np.abs(X_all))) / 10  # round max value to next highest 1e-1
    
    T_max = max([trial['T'] for trial in seq])
    xtk_step = np.ceil(T_max / 25) * 5
    xtk = np.arange(1, T_max + 1, xtk_step)  # 1:xtkStep:Tmax
    xtkl = np.arange(0, T_max * bin_width, xtk_step * bin_width)  # 0:(xtkStep*binWidth):(Tmax-1)*binWidth
    ytk = [-x_max, 0, x_max]

    n_rows = int(np.ceil(X_all.shape[0] / n_cols))
    
    for n in range(min(len(seq), n_plot_max)):
        dat = seq[n][xspec]
        T = seq[n]['T']
            
        for k in range(dat.shape[0]):
            plt.subplot(n_rows, n_cols, k + 1)  # +1 for 1-based indexing in subplot
            
            if seq[n]['trialId'] in red_trials:
                col = [1, 0, 0]  # red
                lw = 3
            else:
                col = [0.2, 0.2, 0.2]  # gray
                lw = 0.05
            
            plt.plot(range(1, T + 1), dat[k, :], linewidth=lw, color=col)

    for k in range(dat.shape[0]):
        plt.subplot(n_rows, n_cols, k + 1)
        plt.axis([1, T_max, 1.1 * min(ytk), 1.1 * max(ytk)])

        if xspec == 'xorth':
            title_str = f'$\\tilde{{\\mathbf{{x}}}}_{{{k+1},:}}$'
        else:
            title_str = f'${{\\mathbf{{x}}}}_{{{k+1},:}}$'
        
        plt.title(title_str, fontsize=16)
        
        plt.xticks(xtk, xtkl)
        plt.yticks(ytk, ytk)
        plt.xlabel('Time (ms)')
    
    plt.tight_layout()
    plt.show()