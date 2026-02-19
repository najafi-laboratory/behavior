"""
Direct Python port of plot3D.m

Plot neural trajectories in a three-dimensional space.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Any, Optional
from modules.GPFA.util.assignopts import assign_opts


def plot_3d(seq: List[Dict[str, Any]], xspec: str, **kwargs) -> None:
    """
    Plot neural trajectories in a three-dimensional space.

    INPUTS:
    seq        - data structure containing extracted trajectories
    xspec      - field name of trajectories in 'seq' to be plotted 
                 (e.g., 'xorth' or 'xsm')

    OPTIONAL ARGUMENTS:
    dims_to_plot - selects three dimensions in seq[n][xspec] to plot 
                   (default: [1, 2, 3])
    n_plot_max   - maximum number of trials to plot (default: 20)
    red_trials   - list of trial_ids whose trajectories are plotted in red
                   (default: [])

    @ 2009 Byron Yu -- byronyu@stanford.edu
    """
    
    dims_to_plot = [1, 2, 3]  # 1-based indexing to match MATLAB
    n_plot_max = 20
    red_trials = []
    extra_opts = assign_opts(locals(), kwargs)

    if seq[0][xspec].shape[0] < 3:
        print('ERROR: Trajectories have less than 3 dimensions.')
        return

    fig = plt.figure()
    pos = fig.get_size_inches()
    fig.set_size_inches(1.3 * pos[0], 1.3 * pos[1])
    
    ax = fig.add_subplot(111, projection='3d')

    for n in range(min(len(seq), n_plot_max)):
        # Convert to 0-based indexing for Python
        dims_idx = [d - 1 for d in dims_to_plot]
        dat = seq[n][xspec][dims_idx, :]
        T = seq[n]['T']
        
        if seq[n]['trialId'] in red_trials:
            col = [1, 0, 0]  # red
            lw = 3
        else:
            col = [0.2, 0.2, 0.2]  # gray
            lw = 0.5
        
        ax.plot3D(dat[0, :], dat[1, :], dat[2, :], '.-', 
                 linewidth=lw, color=col)

    # Set equal aspect ratio
    # Get the data limits
    x_data = []
    y_data = []
    z_data = []
    
    for n in range(min(len(seq), n_plot_max)):
        dims_idx = [d - 1 for d in dims_to_plot]
        dat = seq[n][xspec][dims_idx, :]
        x_data.extend(dat[0, :])
        y_data.extend(dat[1, :])
        z_data.extend(dat[2, :])
    
    # Set equal scaling
    max_range = np.array([max(x_data) - min(x_data), 
                         max(y_data) - min(y_data),
                         max(z_data) - min(z_data)]).max() / 2.0
    
    mid_x = (max(x_data) + min(x_data)) * 0.5
    mid_y = (max(y_data) + min(y_data)) * 0.5
    mid_z = (max(z_data) + min(z_data)) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set labels with LaTeX formatting
    if xspec == 'xorth':
        str1 = f'$\\tilde{{\\mathbf{{x}}}}_{{{dims_to_plot[0]},:}}$'
        str2 = f'$\\tilde{{\\mathbf{{x}}}}_{{{dims_to_plot[1]},:}}$'
        str3 = f'$\\tilde{{\\mathbf{{x}}}}_{{{dims_to_plot[2]},:}}$'
    else:
        str1 = f'${{\\mathbf{{x}}}}_{{{dims_to_plot[0]},:}}$'
        str2 = f'${{\\mathbf{{x}}}}_{{{dims_to_plot[1]},:}}$'
        str3 = f'${{\\mathbf{{x}}}}_{{{dims_to_plot[2]},:}}$'
    
    ax.set_xlabel(str1, fontsize=24)
    ax.set_ylabel(str2, fontsize=24)
    ax.set_zlabel(str3, fontsize=24)
    
    plt.show()