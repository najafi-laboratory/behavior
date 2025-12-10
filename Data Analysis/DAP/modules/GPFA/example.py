"""
Direct Python port of example.m for GPFA

TIPS:
- For exploratory analysis using GPFA, we often run only Section 1
  below, and not Section 2 (which finds the optimal latent
  dimensionality).  This can provide a substantial savings in running
  time, since running Section 2 takes roughly K times as long as
  Section 1, where K is the number of cross-validation folds.  As long
  as we use a latent dimensionality that is 'large enough' in Section 1,
  we can roughly estimate the latent dimensionality by looking at
  the plot produced by plotEachDimVsTime.  The optimal latent
  dimensionality is approximately the number of top dimensions that
  have 'meaningful' temporal structure.  For visualization purposes,
  this rough dimensionality estimate is usually sufficient.

- For exploratory analysis with the two-stage methods, we MUST run
  Section 2 to obtain the optimal smoothing kernel width.  There is
  no easy way estimate the optimal smoothing kernel width from the
  results of Section 1.
"""

import numpy as np
import pickle
import os
from typing import Dict, Any, List, Optional

# Import GPFA modules (equivalent to MATLAB functions)
from modules.GPFA.neuralTraj import neural_traj
from modules.GPFA.util.postprocess import postprocess
from modules.GPFA.plotting.plot3D import plot_3d
from modules.GPFA.plotting.plotEachDimVsTime import plot_each_dim_vs_time
from modules.GPFA.plotting.plotPredErrorVsDim import plot_pred_error_vs_dim
from modules.GPFA.plotting.plotPredErrorVsKernSD import plot_pred_error_vs_kern_sd


filename = 'D:/PHD/GIT/data_analysis/DAP/modules/GPFA/mat_sample/sample_dat.pkl'

def load_sample_data(data_path: str = filename) -> Dict[str, Any]:
    """Load sample data (equivalent to MATLAB's load('mat_sample/sample_dat'))"""
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            dat = pickle.load(f)
        return dat
    else:
        raise FileNotFoundError(f"Sample data not found at {data_path}")

def main():
    """Direct port of the MATLAB example.m script"""
    
    # ===========================================
    # 1) Basic extraction of neural trajectories
    # ===========================================
    dat = load_sample_data()
    
    # Results will be saved in mat_results/runXXX/, where XXX is run_idx.
    # Use a new run_idx for each dataset.
    run_idx = 1
    
    # Select method to extract neural trajectories:
    # 'gpfa' -- Gaussian-process factor analysis
    # 'fa'   -- Smooth and factor analysis
    # 'ppca' -- Smooth and probabilistic principal components analysis
    # 'pca'  -- Smooth and principal components analysis
    method = 'gpfa'
    
    # Select number of latent dimensions
    x_dim = 8
    # NOTE: The optimal dimensionality should be found using 
    #       cross-validation (Section 2) below.
    
    # If using a two-stage method ('fa', 'ppca', or 'pca'), select
    # standard deviation (in msec) of Gaussian smoothing kernel.
    kern_sd = 30
    # NOTE: The optimal kernel width should be found using 
    #       cross-validation (Section 2) below.
    
    # Extract neural trajectories
    result = neural_traj(run_idx, dat, method=method, x_dim=x_dim, 
                        kern_sd_list=kern_sd)
    # NOTE: This function does most of the heavy lifting.
    
    # Orthonormalize neural trajectories
    est_params, seq_train = postprocess(result, kern_sd=kern_sd)
    # NOTE: The importance of orthnormalization is described on 
    #       pp.621-622 of Yu et al., J Neurophysiol, 2009.
    
    # Plot neural trajectories in 3D space
    plot_3d(seq_train, data_field='xorth', dims_to_plot=[1, 2, 3])
    # NOTES:
    # - This figure shows the time-evolution of neural population
    #   activity on a single-trial basis.  Each trajectory is extracted from
    #   the activity of all units on a single trial.
    # - This particular example is based on multi-electrode recordings
    #   in premotor and motor cortices within a 400 ms period starting 300 ms 
    #   before movement onset.  The extracted trajectories appear to
    #   follow the same general path, but there are clear trial-to-trial
    #   differences that can be related to the physical arm movement. 
    # - Analogous to Figure 8 in Yu et al., J Neurophysiol, 2009.
    # WARNING:
    # - If the optimal dimensionality (as assessed by cross-validation in 
    #   Section 2) is greater than 3, then this plot may mask important 
    #   features of the neural trajectories in the dimensions not plotted.  
    #   This motivates looking at the next plot, which shows all latent 
    #   dimensions.
    
    # Plot each dimension of neural trajectories versus time
    plot_each_dim_vs_time(seq_train, data_field='xorth', bin_width=result['bin_width'])
    # NOTES:
    # - These are the same neural trajectories as in the previous figure.
    #   The advantage of this figure is that we can see all latent
    #   dimensions (one per panel), not just three selected dimensions.  
    #   As with the previous figure, each trajectory is extracted from the 
    #   population activity on a single trial.  The activity of each unit 
    #   is some linear combination of each of the panels.  The panels are
    #   ordered, starting with the dimension of greatest covariance
    #   (in the case of 'gpfa' and 'fa') or variance (in the case of
    #   'ppca' and 'pca').
    # - From this figure, we can roughly estimate the optimal
    #   dimensionality by counting the number of top dimensions that have
    #   'meaningful' temporal structure.   In this example, the optimal 
    #   dimensionality appears to be about 5.  This can be assessed
    #   quantitatively using cross-validation in Section 2.
    # - Analogous to Figure 7 in Yu et al., J Neurophysiol, 2009.
    
    print()
    print('Basic extraction and plotting of neural trajectories is complete.')
    print('Press any key to start cross-validation...')
    print('[Depending on the dataset, this can take many minutes to hours.]')
    input()  # Python equivalent of MATLAB's pause
    
    # ========================================================
    # 2) Full cross-validation to find:
    #  - optimal state dimensionality for all methods
    #  - optimal smoothing kernel width for two-stage methods
    # ========================================================
    
    # Select number of cross-validation folds
    num_folds = 4
    
    # Perform cross-validation for different state dimensionalities.
    # Results are saved in mat_results/runXXX/, where XXX is run_idx.
    for x_dim in [2, 5, 8]:
        neural_traj(run_idx, dat, method='pca', x_dim=x_dim, num_folds=num_folds)
        neural_traj(run_idx, dat, method='ppca', x_dim=x_dim, num_folds=num_folds)
        neural_traj(run_idx, dat, method='fa', x_dim=x_dim, num_folds=num_folds)
        neural_traj(run_idx, dat, method='gpfa', x_dim=x_dim, num_folds=num_folds)
    
    print()
    # NOTES:
    # - These function calls are computationally demanding.  Cross-validation 
    #   takes a long time because a separate model has to be fit for each 
    #   state dimensionality and each cross-validation fold.
    
    # Plot prediction error versus state dimensionality.
    # Results files are loaded from mat_results/runXXX/, where XXX is run_idx.
    kern_sd = 30  # select kern_sd for two-stage methods
    plot_pred_error_vs_dim(run_idx, kern_sd)
    # NOTES:
    # - Using this figure, we i) compare the performance (i.e,,
    #   predictive ability) of different methods for extracting neural
    #   trajectories, and ii) find the optimal latent dimensionality for
    #   each method.  The optimal dimensionality is that which gives the
    #   lowest prediction error.  For the two-stage methods, the latent
    #   dimensionality and smoothing kernel width must be jointly
    #   optimized, which requires looking at the next figure.
    # - In this particular example, the optimal dimensionality is 5. This
    #   implies that, even though the raw data are evolving in a
    #   53-dimensional space (i.e., there are 53 units), the system
    #   appears to be using only 5 degrees of freedom due to firing rate
    #   correlations across the neural population.
    # - Analogous to Figure 5A in Yu et al., J Neurophysiol, 2009.
    
    # Plot prediction error versus kernel_sd.
    # Results files are loaded from mat_results/runXXX/, where XXX is run_idx.
    x_dim = 5  # select state dimensionality
    plot_pred_error_vs_kern_sd(run_idx, x_dim)
    # NOTES:
    # - This figure is used to find the optimal smoothing kernel for the
    #   two-stage methods.  The same smoothing kernel is used for all units.
    # - In this particular example, the optimal standard deviation of a
    #   Gaussian smoothing kernel with FA is 30 ms.
    # - Analogous to Figures 5B and 5C in Yu et al., J Neurophysiol, 2009.

if __name__ == "__main__":
    main()