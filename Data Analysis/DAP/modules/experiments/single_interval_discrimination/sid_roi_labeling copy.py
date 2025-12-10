from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional

from dataclasses import dataclass
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.decomposition import NMF
from scipy.optimize import minimize
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
import warnings
from matplotlib.gridspec import GridSpec

from copy import deepcopy


def prepare_trial_tensor(trial_data: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convert trial data to tensor format suitable for motif discovery.
    X shape: (n_rois, n_time_bins, n_trials)
    """
    print("=== PREPARING TRIAL TENSOR ===")
    
    df = trial_data['df_trials_with_segments']
    tensor_cfg = cfg.get('tensor_prep', {})
    
    print(f"Input data: {len(df)} trials")
    
    # Verify data structure
    if len(df) == 0:
        raise ValueError("No trials found in df_trials_with_segments")
    
    # Check first trial to understand data structure
    first_row = df.iloc[0]
    print(f"First trial columns: {list(first_row.keys())}")
    
    if 'dff_time_vector' not in first_row or 'dff_segment' not in first_row:
        raise ValueError("Missing required columns: dff_time_vector or dff_segment")
    
    first_time = np.asarray(first_row['dff_time_vector'])
    first_dff = np.asarray(first_row['dff_segment'])
    print(f"First trial - time shape: {first_time.shape}, dff shape: {first_dff.shape}")
    print(f"First trial - time range: {first_time.min():.3f} to {first_time.max():.3f}s")
    
    # Find common time grid across all trials
    print("\nAnalyzing time vectors across trials...")
    all_time_vecs = []
    time_ranges = []
    
    for trial_idx, (_, row) in enumerate(df.iterrows()):
        t_vec = np.asarray(row['dff_time_vector'])
        if len(t_vec) == 0:
            print(f"  Trial {trial_idx}: empty time vector, skipping")
            continue
        all_time_vecs.append(t_vec)
        time_ranges.append((t_vec.min(), t_vec.max()))
        
        if trial_idx < 5:  # Print first 5 for verification
            print(f"  Trial {trial_idx}: {len(t_vec)} timepoints, range {t_vec.min():.3f} to {t_vec.max():.3f}s")
    
    if len(all_time_vecs) == 0:
        raise ValueError("No valid time vectors found")
    
    # Determine global time range
    t_min = min(t_range[0] for t_range in time_ranges)
    t_max = max(t_range[1] for t_range in time_ranges)
    print(f"\nGlobal time range: {t_min:.3f} to {t_max:.3f}s ({t_max - t_min:.3f}s duration)")
    
    # Set up time grid - downsample for computational efficiency
    dt = tensor_cfg.get('dt', 0.067)  # ~15 Hz default
    time_grid = np.arange(t_min, t_max + dt/2, dt)  # Add dt/2 to ensure we capture t_max
    n_bins = len(time_grid)
    
    print(f"Time grid: {n_bins} bins at {1/dt:.1f} Hz (dt = {dt:.3f}s)")
    
    # Determine number of ROIs from first valid trial
    n_rois = None
    for _, row in df.iterrows():
        dff = np.asarray(row['dff_segment'])
        if dff.size > 0:
            n_rois = dff.shape[0]
            break
    
    if n_rois is None:
        raise ValueError("Could not determine number of ROIs")
    
    print(f"Number of ROIs: {n_rois}")
    
    # Initialize tensor with NaN
    n_trials = len(df)
    X = np.full((n_rois, n_bins, n_trials), np.nan, dtype=np.float32)
    
    print(f"\nTensor shape: {X.shape} (ROIs × time × trials)")
    print("Filling tensor by interpolating trials...")
    
    # Fill tensor by interpolating each trial
    valid_trials = 0
    interpolation_stats = []
    
    for trial_idx, (_, row) in enumerate(df.iterrows()):
        t_vec = np.asarray(row['dff_time_vector'])
        dff = np.asarray(row['dff_segment'])
        
        if dff.size == 0 or t_vec.size == 0:
            print(f"  Trial {trial_idx}: empty data, skipping")
            continue
            
        if dff.shape[0] != n_rois:
            print(f"  Trial {trial_idx}: ROI count mismatch ({dff.shape[0]} vs {n_rois}), skipping")
            continue
        
        # Ensure monotonic time (Suite2p sometimes has duplicates)
        if not np.all(np.diff(t_vec) > 0):
            sort_idx = np.argsort(t_vec)
            t_vec = t_vec[sort_idx]
            dff = dff[:, sort_idx]
            
        # Interpolate each ROI to common grid
        trial_finite_count = 0
        for roi_idx in range(n_rois):
            # Only interpolate within the native time support
            roi_data = dff[roi_idx, :]
            valid_mask = np.isfinite(roi_data) & np.isfinite(t_vec)
            
            if valid_mask.sum() < 3:  # Need at least 3 points for interpolation
                continue
                
            # Interpolate only within the data's time range
            t_valid = t_vec[valid_mask]
            roi_valid = roi_data[valid_mask]
            
            # Find time grid points within this ROI's data range
            grid_mask = (time_grid >= t_valid.min()) & (time_grid <= t_valid.max())
            
            if grid_mask.sum() > 0:
                X[roi_idx, grid_mask, trial_idx] = np.interp(
                    time_grid[grid_mask], t_valid, roi_valid
                )
                trial_finite_count += grid_mask.sum()
        
        interpolation_stats.append(trial_finite_count)
        valid_trials += 1
        
        if trial_idx % 50 == 0 or trial_idx < 10:
            print(f"  Trial {trial_idx}: {trial_finite_count} finite values interpolated")
    
    print(f"\nCompleted interpolation: {valid_trials}/{n_trials} trials processed")
    
    # Analyze tensor completeness
    total_elements = X.size
    finite_elements = np.isfinite(X).sum()
    finite_fraction = finite_elements / total_elements
    
    print(f"Tensor completeness: {finite_elements:,}/{total_elements:,} ({finite_fraction:.1%}) finite")
    
    # Per-ROI completeness
    roi_completeness = np.isfinite(X).sum(axis=(1, 2)) / (n_bins * n_trials)
    print(f"ROI completeness: min={roi_completeness.min():.1%}, "
          f"median={np.median(roi_completeness):.1%}, max={roi_completeness.max():.1%}")
    
    # Per-trial completeness  
    trial_completeness = np.isfinite(X).sum(axis=(0, 1)) / (n_rois * n_bins)
    print(f"Trial completeness: min={trial_completeness.min():.1%}, "
          f"median={np.median(trial_completeness):.1%}, max={trial_completeness.max():.1%}")
    
    # Warn about low completeness
    if finite_fraction < 0.5:
        print(f"WARNING: Low tensor completeness ({finite_fraction:.1%}). "
              "Consider adjusting time range or interpolation parameters.")
    
    # Store trial metadata for later auditing
    trial_info = {}
    
    # Extract available trial-level variables
    available_cols = df.columns.tolist()
    print(f"\nAvailable trial metadata columns: {available_cols}")
    
    for col in ['isi', 'is_right_choice', 'is_right', 'rewarded', 'punished', 'lick', 'lick_start', 'did_not_choose']:
        if col in df.columns:
            values = df[col].values
            trial_info[col] = values
            finite_count = np.isfinite(values).sum() if np.issubdtype(values.dtype, np.number) else len(values)
            print(f"  {col}: {finite_count}/{len(values)} valid values")
        else:
            print(f"  {col}: not found")
    
    # Add trial indices for reference
    trial_info['trial_indices'] = df.index.values
    
    print(f"\nTrial info extracted: {list(trial_info.keys())}")
    print("=== TENSOR PREPARATION COMPLETE ===\n")
    
    return X, time_grid, trial_info


def fit_cp_decomposition(X: np.ndarray, time_grid: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fit CP/PARAFAC decomposition to extract temporal motifs.
    X ≈ Σ_k C_k ⊗ B_k ⊗ A_k where:
    - C_k: ROI loadings (which ROIs participate)
    - B_k: motif timecourse (when it happens)  
    - A_k: trial weights (how much per trial)
    """
    print("=== FITTING CP/PARAFAC DECOMPOSITION ===")
    
    cp_cfg = cfg.get('cp_decomposition', {})
    n_motifs = cp_cfg.get('n_motifs', 8)
    max_iter = cp_cfg.get('max_iter', 100)
    tol = cp_cfg.get('tolerance', 1e-6)
    
    n_rois, n_bins, n_trials = X.shape
    print(f"Input tensor: {X.shape} (ROIs × time × trials)")
    print(f"Fitting {n_motifs} motifs")
    print(f"Max iterations: {max_iter}, tolerance: {tol}")
    
    # Analyze data characteristics
    finite_mask = np.isfinite(X)
    print(f"Data completeness: {finite_mask.sum()}/{X.size} ({finite_mask.mean():.1%}) finite")
    
    if finite_mask.sum() < 0.1 * X.size:
        print("WARNING: Very sparse data (<10% finite). Results may be unreliable.")
    
    # Initialize factor matrices with small random values
    np.random.seed(42)  # For reproducibility
    
    C = np.random.rand(n_rois, n_motifs) * 0.1      # ROI loadings
    B = np.random.rand(n_bins, n_motifs) * 0.1      # Motif timecourses
    A = np.random.rand(n_trials, n_motifs) * 0.1    # Trial weights
    
    print(f"Initialized factors: C{C.shape}, B{B.shape}, A{A.shape}")
    
    # Track convergence
    errors = []
    prev_error = np.inf
    
    print("\nStarting alternating least squares optimization...")
    
    for iteration in range(max_iter):
        # Update C (ROI loadings) - fix B and A
        C = update_factor_als(X, [B, A], [1, 2], finite_mask, 'C')
        
        # Update B (motif timecourses) - fix C and A
        B = update_factor_als(X, [C, A], [0, 2], finite_mask, 'B')
        
        # Update A (trial weights) - fix C and B
        A = update_factor_als(X, [C, B], [0, 1], finite_mask, 'A')
        
        # Compute reconstruction error every 10 iterations
        if iteration % 10 == 0:
            error = compute_reconstruction_error(X, C, B, A, finite_mask)
            errors.append(error)
            
            print(f"  Iter {iteration:3d}: error = {error:.6f}")
            
            # Check convergence
            if iteration > 0 and abs(prev_error - error) < tol:
                print(f"  Converged after {iteration} iterations!")
                break
            prev_error = error
        
        # Check for numerical issues
        if not (np.isfinite(C).all() and np.isfinite(B).all() and np.isfinite(A).all()):
            print(f"  WARNING: Non-finite values detected at iteration {iteration}")
            break
    
    # Final error computation
    final_error = compute_reconstruction_error(X, C, B, A, finite_mask)
    print(f"\nFinal reconstruction error: {final_error:.6f}")
    
    # Analyze factor characteristics
    print("\nFactor analysis:")
    
    # ROI loadings statistics
    roi_sparsity = (C > 0.01).mean(axis=0)  # Fraction of ROIs with non-trivial loading
    print(f"ROI participation per motif: {roi_sparsity}")
    print(f"  Mean: {roi_sparsity.mean():.2f}, Std: {roi_sparsity.std():.2f}")
    
    # Motif timecourse statistics
    motif_vars = np.var(B, axis=0)
    motif_peaks = np.argmax(np.abs(B), axis=0)
    motif_peak_times = time_grid[motif_peaks]
    print(f"Motif variances: {motif_vars}")
    print(f"Motif peak times: {motif_peak_times}")
    
    # Trial weight statistics
    trial_sparsity = (A > 0.01).mean(axis=0)  # Fraction of trials with non-trivial weight
    print(f"Trial participation per motif: {trial_sparsity}")
    
    # Sort motifs by explained variance (largest first)
    sort_idx = np.argsort(motif_vars)[::-1]
    
    C_sorted = C[:, sort_idx]
    B_sorted = B[:, sort_idx]
    A_sorted = A[:, sort_idx]
    
    print(f"\nMotifs sorted by variance (descending): {motif_vars[sort_idx]}")
    
    # Package results
    result = {
        'roi_loadings': C_sorted,           # (n_rois, n_motifs)
        'motif_timecourses': B_sorted,      # (n_bins, n_motifs)  
        'trial_weights': A_sorted,          # (n_trials, n_motifs)
        'time_grid': time_grid,             # (n_bins,)
        'reconstruction_error': final_error,
        'convergence_errors': errors,
        'n_iterations': min(iteration + 1, max_iter),
        'converged': iteration < max_iter - 1,
        'config': cp_cfg.copy()
    }
    
    print("=== CP/PARAFAC DECOMPOSITION COMPLETE ===\n")
    
    return result


def update_factor_als(X: np.ndarray, other_factors: List[np.ndarray], 
                     other_modes: List[int], finite_mask: np.ndarray, 
                     factor_name: str) -> np.ndarray:
    """
    Alternating least squares update for one factor in CP decomposition.
    """
    # Determine which mode we're updating
    all_modes = [0, 1, 2]  # ROI, time, trial
    update_mode = None
    for mode in all_modes:
        if mode not in other_modes:
            update_mode = mode
            break
    
    if update_mode is None:
        raise ValueError("Could not determine update mode")
    
    # Khatri-Rao product of other factors
    if len(other_factors) == 1:
        krao = other_factors[0]
    else:
        krao = khatri_rao_product(other_factors[0], other_factors[1])
    
    # Unfold tensor along the mode we're updating
    if update_mode == 0:  # ROI mode
        X_unfold = X.reshape(X.shape[0], -1)  # (n_rois, n_bins * n_trials)
        mask_unfold = finite_mask.reshape(finite_mask.shape[0], -1)
    elif update_mode == 1:  # Time mode
        X_unfold = X.transpose(1, 0, 2).reshape(X.shape[1], -1)  # (n_bins, n_rois * n_trials)
        mask_unfold = finite_mask.transpose(1, 0, 2).reshape(finite_mask.shape[1], -1)
    else:  # Trial mode
        X_unfold = X.transpose(2, 0, 1).reshape(X.shape[2], -1)  # (n_trials, n_rois * n_bins)
        mask_unfold = finite_mask.transpose(2, 0, 1).reshape(finite_mask.shape[2], -1)
    
    # Solve least squares for each row, handling missing data
    n_rows, n_cols = X_unfold.shape
    n_factors = krao.shape[1]
    updated_factor = np.zeros((n_rows, n_factors))
    
    for i in range(n_rows):
        valid_cols = mask_unfold[i, :]
        n_valid = valid_cols.sum()
        
        if n_valid < n_factors:  # Need at least as many observations as factors
            continue
            
        try:
            # Solve least squares: krao[valid, :] @ factor[i, :] = X_unfold[i, valid]
            A_valid = krao[valid_cols, :]
            b_valid = X_unfold[i, valid_cols]
            
            # Add small regularization to handle near-singular cases
            reg = 1e-8
            solution = np.linalg.solve(A_valid.T @ A_valid + reg * np.eye(n_factors), 
                                     A_valid.T @ b_valid)
            updated_factor[i, :] = solution
            
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            try:
                updated_factor[i, :] = np.linalg.pinv(krao[valid_cols, :]) @ X_unfold[i, valid_cols]
            except:
                pass  # Keep zeros for problematic rows
    
    # Apply non-negativity constraint (optional but often helpful)
    apply_nonneg = True  # Could be a config option
    if apply_nonneg:
        updated_factor = np.maximum(updated_factor, 0)
    
    return updated_factor


def khatri_rao_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute Khatri-Rao product: A ⊙ B
    Result has shape (A.shape[0] * B.shape[0], A.shape[1])
    """
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Incompatible shapes for Khatri-Rao: {A.shape} and {B.shape}")
    
    n_factors = A.shape[1]
    result_rows = A.shape[0] * B.shape[0]
    result = np.zeros((result_rows, n_factors))
    
    for k in range(n_factors):
        result[:, k] = np.kron(A[:, k], B[:, k])
    
    return result


def compute_reconstruction_error(X: np.ndarray, C: np.ndarray, B: np.ndarray, 
                               A: np.ndarray, finite_mask: np.ndarray) -> float:
    """
    Compute reconstruction error: ||X - X_recon||_F^2 on finite elements only.
    """
    # Reconstruct tensor: X_recon[i,j,k] = Σ_r C[i,r] * B[j,r] * A[k,r]
    n_rois, n_bins, n_trials = X.shape
    n_motifs = C.shape[1]
    
    X_recon = np.zeros_like(X)
    
    for r in range(n_motifs):
        # Outer product of the three factor vectors
        factor_tensor = np.outer(C[:, r], np.outer(B[:, r], A[:, r])).reshape(n_rois, n_bins, n_trials)
        X_recon += factor_tensor
    
    # Compute error only on finite elements
    diff = (X - X_recon)[finite_mask]
    mse = np.mean(diff ** 2)
    
    return mse


# def audit_motifs_against_events(cp_result: Dict[str, Any], trial_info: Dict[str, Any], 
#                                cfg: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Post-hoc auditing of discovered motifs against behavioral events.
#     This assigns functional labels without biasing the original discovery.
#     """
#     print("=== AUDITING MOTIFS AGAINST BEHAVIORAL EVENTS ===")
    
#     audit_cfg = cfg.get('motif_audit', {})
    
#     motif_timecourses = cp_result['motif_timecourses']  # (n_bins, n_motifs)
#     trial_weights = cp_result['trial_weights']          # (n_trials, n_motifs)
#     time_grid = cp_result['time_grid']                  # (n_bins,)
#     n_motifs = motif_timecourses.shape[1]
    
#     print(f"Auditing {n_motifs} motifs against behavioral events")
#     print(f"Time grid: {len(time_grid)} bins from {time_grid[0]:.3f} to {time_grid[-1]:.3f}s")
#     print(f"Available trial info: {list(trial_info.keys())}")
    
#     # Initialize results
#     motif_scores = {}
#     motif_labels = []
    
#     # Event columns to check (these should be time values relative to trial start)
#     event_columns = ['start_flash_1', 'end_flash_1', 'start_flash_2', 'end_flash_2', 'choice_start']
#     available_events = [col for col in event_columns if col in trial_info]
#     print(f"Available event columns: {available_events}")
    
#     # Process each motif
#     for k in range(n_motifs):
#         print(f"\n--- Motif {k} ---")
        
#         motif_waveform = motif_timecourses[:, k]
#         motif_trial_weights = trial_weights[:, k]
        
#         # Basic motif statistics
#         peak_idx = np.argmax(np.abs(motif_waveform))
#         peak_time = time_grid[peak_idx]
#         peak_amplitude = motif_waveform[peak_idx]
#         motif_var = np.var(motif_waveform)
        
#         print(f"  Peak time: {peak_time:.3f}s, amplitude: {peak_amplitude:.4f}, variance: {motif_var:.4f}")
        
#         scores = {}
        
#         # 1. Event enrichment analysis
#         print("  Event enrichment analysis:")
#         for event_col in available_events:
#             if event_col not in trial_info:
#                 continue
                
#             enrichment_score = score_event_enrichment(
#                 motif_trial_weights, trial_info[event_col], peak_time, event_col
#             )
#             scores[f'{event_col}_enrichment'] = enrichment_score
            
#             if enrichment_score > 2.0:
#                 print(f"    {event_col}: STRONG enrichment (z={enrichment_score:.2f})")
#             elif enrichment_score > 1.0:
#                 print(f"    {event_col}: moderate enrichment (z={enrichment_score:.2f})")
#             else:
#                 print(f"    {event_col}: weak/no enrichment (z={enrichment_score:.2f})")
        
#         # 2. ISI/expectation coupling
#         print("  ISI coupling analysis:")
#         if 'isi' in trial_info:
#             isi_score = score_isi_coupling(motif_trial_weights, trial_info['isi'])
#             scores['isi_coupling'] = isi_score
#             print(f"    ISI correlation: r²={isi_score:.4f}")
#         else:
#             scores['isi_coupling'] = 0.0
#             print("    ISI data not available")
        
#         # 3. Choice information
#         print("  Choice information analysis:")
#         if 'choice' in trial_info:
#             choice_auc = score_choice_information(motif_trial_weights, trial_info['choice'])
#             scores['choice_auc'] = choice_auc
#             print(f"    Choice AUC: {choice_auc:.4f}")
#         else:
#             scores['choice_auc'] = 0.5
#             print("    Choice data not available")
        
#         # 4. Outcome coupling
#         print("  Outcome coupling analysis:")
#         if 'outcome' in trial_info:
#             outcome_auc = score_choice_information(motif_trial_weights, trial_info['outcome'])
#             scores['outcome_auc'] = outcome_auc
#             print(f"    Outcome AUC: {outcome_auc:.4f}")
#         else:
#             scores['outcome_auc'] = 0.5
#             print("    Outcome data not available")
        
#         # Assign functional label based on scores
#         motif_label = assign_motif_label(scores, audit_cfg)
#         motif_labels.append(motif_label)
#         motif_scores[f'motif_{k}'] = scores
        
#         print(f"  → LABEL: {motif_label}")
    
#     # Summary statistics
#     print(f"\n=== MOTIF LABELING SUMMARY ===")
#     label_counts = {}
#     for label in motif_labels:
#         label_counts[label] = label_counts.get(label, 0) + 1
    
#     for label, count in label_counts.items():
#         print(f"  {label}: {count} motifs")
    
#     # Package results
#     audit_result = {
#         'motif_labels': motif_labels,
#         'motif_scores': motif_scores,
#         'label_counts': label_counts,
#         'config': audit_cfg.copy()
#     }
    
#     print("=== MOTIF AUDITING COMPLETE ===\n")
    
#     return audit_result



def audit_motifs_against_events(cp_result: Dict[str, Any], trial_info: Dict[str, Any], 
                               cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-hoc auditing of discovered motifs against behavioral events.
    This assigns functional labels without biasing the original discovery.
    """
    print("=== AUDITING MOTIFS AGAINST BEHAVIORAL EVENTS ===")
    
    audit_cfg = cfg.get('motif_audit', {})
    
    motif_timecourses = cp_result['motif_timecourses']  # (n_bins, n_motifs)
    trial_weights = cp_result['trial_weights']          # (n_trials, n_motifs)
    time_grid = cp_result['time_grid']                  # (n_bins,)
    n_motifs = motif_timecourses.shape[1]
    
    print(f"Auditing {n_motifs} motifs against behavioral events")
    print(f"Time grid: {len(time_grid)} bins from {time_grid[0]:.3f} to {time_grid[-1]:.3f}s")
    print(f"Available trial info: {list(trial_info.keys())}")
    
    # Initialize results
    motif_scores = {}
    motif_labels = []
    
    # Event columns to check (these should be time values relative to trial start)
    event_columns = ['start_flash_1', 'end_flash_1', 'start_flash_2', 'end_flash_2', 'lick_start']
    available_events = [col for col in event_columns if col in trial_info]
    print(f"Available event columns: {available_events}")
    
    # Process each motif
    for k in range(n_motifs):
        print(f"\n--- Motif {k} ---")
        
        motif_waveform = motif_timecourses[:, k]
        motif_trial_weights = trial_weights[:, k]
        
        # Basic motif statistics
        peak_idx = np.argmax(np.abs(motif_waveform))
        peak_time = time_grid[peak_idx]
        peak_amplitude = motif_waveform[peak_idx]
        motif_var = np.var(motif_waveform)
        
        print(f"  Peak time: {peak_time:.3f}s, amplitude: {peak_amplitude:.4f}, variance: {motif_var:.4f}")
        
        scores = {}
        
        # 1. Event enrichment analysis
        print("  Event enrichment analysis:")
        for event_col in available_events:
            if event_col not in trial_info:
                continue
                
            enrichment_score = score_event_enrichment(
                motif_trial_weights, trial_info[event_col], peak_time, event_col
            )
            scores[f'{event_col}_enrichment'] = enrichment_score
            
            if enrichment_score > 2.0:
                print(f"    {event_col}: STRONG enrichment (z={enrichment_score:.2f})")
            elif enrichment_score > 1.0:
                print(f"    {event_col}: moderate enrichment (z={enrichment_score:.2f})")
            else:
                print(f"    {event_col}: weak/no enrichment (z={enrichment_score:.2f})")
        
        # 2. ISI/expectation coupling
        print("  ISI coupling analysis:")
        if 'isi' in trial_info:
            isi_score = score_isi_coupling(motif_trial_weights, trial_info['isi'])
            scores['isi_coupling'] = isi_score
            print(f"    ISI correlation: r²={isi_score:.4f}")
        else:
            scores['isi_coupling'] = 0.0
            print("    ISI data not available")
        
        # 3. Choice information (actual choice made by mouse)
        print("  Choice information analysis:")
        if 'is_right_choice' in trial_info:
            choice_auc = score_choice_information(motif_trial_weights, trial_info['is_right_choice'])
            scores['choice_auc'] = choice_auc
            print(f"    Choice AUC (left vs right lick): {choice_auc:.4f}")
        else:
            scores['choice_auc'] = 0.5
            print("    Choice data (is_right_choice) not available")
        
        # 4. Trial type information (stimulus identity)
        print("  Trial type (stimulus) information analysis:")
        if 'is_right' in trial_info:
            stimulus_auc = score_choice_information(motif_trial_weights, trial_info['is_right'])
            scores['stimulus_auc'] = stimulus_auc
            print(f"    Stimulus AUC (left vs right type): {stimulus_auc:.4f}")
        else:
            scores['stimulus_auc'] = 0.5
            print("    Stimulus data (is_right) not available")
        
        # 5. Outcome coupling (correct vs incorrect)
        print("  Outcome coupling analysis:")
        if 'rewarded' in trial_info:
            outcome_auc = score_choice_information(motif_trial_weights, trial_info['rewarded'])
            scores['outcome_auc'] = outcome_auc
            print(f"    Outcome AUC (unrewarded vs rewarded): {outcome_auc:.4f}")
        else:
            scores['outcome_auc'] = 0.5
            print("    Outcome data (rewarded) not available")
        
        # 6. No-choice trials analysis
        print("  No-choice analysis:")
        if 'did_not_choose' in trial_info:
            nochoice_auc = score_choice_information(motif_trial_weights, trial_info['did_not_choose'])
            scores['nochoice_auc'] = nochoice_auc
            print(f"    No-choice AUC (chose vs did not choose): {nochoice_auc:.4f}")
        else:
            scores['nochoice_auc'] = 0.5
            print("    No-choice data (did_not_choose) not available")
        
        # Assign functional label based on scores
        motif_label = assign_motif_label(scores, audit_cfg)
        motif_labels.append(motif_label)
        motif_scores[f'motif_{k}'] = scores
        
        print(f"  → LABEL: {motif_label}")
    
    # Summary statistics
    print(f"\n=== MOTIF LABELING SUMMARY ===")
    label_counts = {}
    for label in motif_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in label_counts.items():
        print(f"  {label}: {count} motifs")
    
    # Package results
    audit_result = {
        'motif_labels': motif_labels,
        'motif_scores': motif_scores,
        'label_counts': label_counts,
        'config': audit_cfg.copy()
    }
    
    print("=== MOTIF AUDITING COMPLETE ===\n")
    
    return audit_result





































def assign_motif_label(scores: Dict[str, float], audit_cfg: Dict[str, Any]) -> str:
    """
    Assign functional label to motif based on audit scores.
    Updated to handle the specific behavioral variables in your dataset.
    """
    # Thresholds from config
    event_thresh = audit_cfg.get('event_enrichment_thresh', 2.0)
    choice_thresh = audit_cfg.get('choice_auc_thresh', 0.65)
    isi_thresh = audit_cfg.get('isi_coupling_thresh', 0.1)
    outcome_thresh = audit_cfg.get('outcome_auc_thresh', 0.65)
    stimulus_thresh = audit_cfg.get('stimulus_auc_thresh', 0.65)
    nochoice_thresh = audit_cfg.get('nochoice_auc_thresh', 0.65)
    
    labels = []
    
    # Check event enrichments
    events_with_enrichment = []
    for event in ['start_flash_1', 'end_flash_1', 'start_flash_2', 'end_flash_2', 'lick_start']:
        score_key = f'{event}_enrichment'
        if score_key in scores and scores[score_key] > event_thresh:
            if 'flash' in event:
                events_with_enrichment.append('stimulus')
            elif 'lick' in event:
                events_with_enrichment.append('choice')
    
    # Check choice information (actual lick direction)
    if scores.get('choice_auc', 0.5) > choice_thresh:
        labels.append('choice')
    
    # Check stimulus information (trial type)
    if scores.get('stimulus_auc', 0.5) > stimulus_thresh:
        labels.append('stimulus')
    
    # Check outcome information (reward)
    if scores.get('outcome_auc', 0.5) > outcome_thresh:
        labels.append('outcome')
    
    # Check no-choice information (engagement)
    if scores.get('nochoice_auc', 0.5) > nochoice_thresh:
        labels.append('engagement')
    
    # Check ISI coupling (expectation/timing)
    if scores.get('isi_coupling', 0.0) > isi_thresh:
        labels.append('expectation')
    
    # Add stimulus labels from event enrichment
    labels.extend(events_with_enrichment)
    
    # Return most specific label
    unique_labels = list(set(labels))
    
    if not unique_labels:
        return 'artifact'
    elif len(unique_labels) == 1:
        return unique_labels[0]
    else:
        # Return compound label for multi-functional motifs
        return '_'.join(sorted(unique_labels))


def assign_motif_label_strict(scores: Dict[str, float], audit_cfg: Dict[str, Any]) -> str:
    """
    STRICTER version: Assign single most significant functional label to motif.
    """
    # Thresholds from config - make them more stringent
    event_thresh = audit_cfg.get('event_enrichment_thresh', 3.0)  # Raised from 2.0
    choice_thresh = audit_cfg.get('choice_auc_thresh', 0.70)       # Raised from 0.65
    isi_thresh = audit_cfg.get('isi_coupling_thresh', 0.15)        # Raised from 0.1
    outcome_thresh = audit_cfg.get('outcome_auc_thresh', 0.70)     # Raised from 0.65
    stimulus_thresh = audit_cfg.get('stimulus_auc_thresh', 0.70)   # Raised from 0.65
    nochoice_thresh = audit_cfg.get('nochoice_auc_thresh', 0.70)   # Raised from 0.65
    
    # Collect all significant scores with their strengths
    significant_scores = []
    
    # Check event enrichments
    for event in ['start_flash_1', 'end_flash_1', 'start_flash_2', 'end_flash_2', 'lick_start']:
        score_key = f'{event}_enrichment'
        if score_key in scores and scores[score_key] > event_thresh:
            strength = scores[score_key]
            if 'flash' in event:
                significant_scores.append(('stimulus', strength))
            elif 'lick' in event:
                significant_scores.append(('choice', strength))
    
    # Check behavioral scores
    if scores.get('choice_auc', 0.5) > choice_thresh:
        # Convert AUC to "strength" (distance from chance)
        strength = abs(scores['choice_auc'] - 0.5) * 2  # Scale to 0-1
        significant_scores.append(('choice', strength))
    
    if scores.get('stimulus_auc', 0.5) > stimulus_thresh:
        strength = abs(scores['stimulus_auc'] - 0.5) * 2
        significant_scores.append(('stimulus', strength))
    
    if scores.get('outcome_auc', 0.5) > outcome_thresh:
        strength = abs(scores['outcome_auc'] - 0.5) * 2
        significant_scores.append(('outcome', strength))
    
    if scores.get('nochoice_auc', 0.5) > nochoice_thresh:
        strength = abs(scores['nochoice_auc'] - 0.5) * 2
        significant_scores.append(('engagement', strength))
    
    if scores.get('isi_coupling', 0.0) > isi_thresh:
        strength = scores['isi_coupling']  # Already 0-1 scale
        significant_scores.append(('expectation', strength))
    
    # Return the STRONGEST single label
    if not significant_scores:
        return 'artifact'
    else:
        # Sort by strength and return the strongest
        significant_scores.sort(key=lambda x: x[1], reverse=True)
        return significant_scores[0][0]


def debug_motif_scores(audit_result: Dict[str, Any]):
    """
    Print detailed breakdown of scores for each motif to understand labeling.
    """
    print("=== DEBUGGING MOTIF SCORES ===")
    
    motif_scores = audit_result['motif_scores']
    motif_labels = audit_result['motif_labels']
    
    for k, label in enumerate(motif_labels):
        scores = motif_scores[f'motif_{k}']
        print(f"\nMotif {k} (labeled: {label}):")
        
        # Print all scores in organized way
        print("  Event enrichments:")
        for score_name, score_val in scores.items():
            if 'enrichment' in score_name:
                event_name = score_name.replace('_enrichment', '')
                print(f"    {event_name}: {score_val:.3f}")
        
        print("  Behavioral coupling:")
        for score_name in ['choice_auc', 'stimulus_auc', 'outcome_auc', 'nochoice_auc', 'isi_coupling']:
            if score_name in scores:
                print(f"    {score_name}: {scores[score_name]:.3f}")
        
        # Show which thresholds were exceeded
        exceeded = []
        if any(scores.get(f'{e}_enrichment', 0) > 2.0 for e in ['start_flash_1', 'end_flash_1', 'start_flash_2', 'end_flash_2']):
            exceeded.append('stimulus_event')
        if scores.get('lick_start_enrichment', 0) > 2.0:
            exceeded.append('choice_event')
        if scores.get('choice_auc', 0.5) > 0.65:
            exceeded.append('choice_info')
        if scores.get('stimulus_auc', 0.5) > 0.65:
            exceeded.append('stimulus_info')
        if scores.get('outcome_auc', 0.5) > 0.65:
            exceeded.append('outcome_info')
        if scores.get('isi_coupling', 0) > 0.1:
            exceeded.append('timing_info')
            
        print(f"    Exceeded thresholds: {exceeded}")



def analyze_trial_info_distribution(trial_info: Dict[str, Any]):
    """
    Analyze the distribution of behavioral variables to understand the data.
    """
    print("=== TRIAL INFO ANALYSIS ===")
    
    for var_name, values in trial_info.items():
        if var_name == 'trial_indices':
            continue
            
        print(f"\n{var_name}:")
        
        if np.issubdtype(type(values[0]), np.number):
            finite_mask = np.isfinite(values)
            finite_vals = values[finite_mask]
            
            if len(finite_vals) > 0:
                print(f"  Valid values: {finite_mask.sum()}/{len(values)}")
                print(f"  Range: {finite_vals.min():.3f} to {finite_vals.max():.3f}")
                print(f"  Unique values: {len(np.unique(finite_vals))}")
                if len(np.unique(finite_vals)) <= 10:
                    unique_vals, counts = np.unique(finite_vals, return_counts=True)
                    for val, count in zip(unique_vals, counts):
                        print(f"    {val}: {count} trials")
        else:
            unique_vals, counts = np.unique(values, return_counts=True)
            print(f"  Unique values: {len(unique_vals)}")
            for val, count in zip(unique_vals, counts):
                print(f"    {val}: {count} trials")





def score_event_enrichment(trial_weights: np.ndarray, event_times: np.ndarray, 
                         motif_peak_time: float, event_name: str) -> float:
    """
    Test if motif is enriched around a specific behavioral event.
    Returns z-score of enrichment.
    """
    # Clean data
    valid_mask = np.isfinite(trial_weights) & np.isfinite(event_times)
    if valid_mask.sum() < 10:
        return 0.0
    
    weights = trial_weights[valid_mask]
    events = event_times[valid_mask]
    
    # Compute temporal distance from motif peak to event times
    distances = np.abs(motif_peak_time - events)
    
    # Test if high-weight trials have shorter distances to events
    median_weight = np.median(weights)
    high_weight_mask = weights > median_weight
    low_weight_mask = weights <= median_weight
    
    if high_weight_mask.sum() < 3 or low_weight_mask.sum() < 3:
        return 0.0
    
    high_weight_distances = distances[high_weight_mask]
    low_weight_distances = distances[low_weight_mask]
    
    # Z-score for enrichment (negative distances indicate enrichment)
    mean_diff = np.mean(low_weight_distances) - np.mean(high_weight_distances)
    pooled_std = np.std(distances)
    
    if pooled_std == 0:
        return 0.0
    
    z_score = mean_diff / pooled_std
    
    return z_score


def score_isi_coupling(trial_weights: np.ndarray, isi_values: np.ndarray) -> float:
    """
    Test if motif shows ISI-dependent activation (expectation/timing).
    Returns correlation coefficient squared.
    """
    valid_mask = np.isfinite(trial_weights) & np.isfinite(isi_values)
    if valid_mask.sum() < 10:
        return 0.0
    
    weights = trial_weights[valid_mask]
    isis = isi_values[valid_mask]
    
    try:
        correlation = np.corrcoef(weights, isis)[0, 1]
        return correlation ** 2 if np.isfinite(correlation) else 0.0
    except:
        return 0.0


def score_choice_information(trial_weights: np.ndarray, choice_values: np.ndarray) -> float:
    """
    Test if motif carries choice/outcome information using cross-validated AUC.
    """
    valid_mask = np.isfinite(trial_weights) & np.isfinite(choice_values)
    if valid_mask.sum() < 20:
        return 0.5
    
    weights = trial_weights[valid_mask]
    choices = choice_values[valid_mask]
    
    # Check if we have both choice types
    unique_choices = np.unique(choices)
    if len(unique_choices) < 2:
        return 0.5
    
    try:
        # Simple cross-validated AUC
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        
        X = weights.reshape(-1, 1)
        y = choices
        
        scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, 
                               cv=min(3, len(unique_choices)), scoring='roc_auc')
        return np.mean(scores)
    except:
        # Fallback to simple correlation-based score
        try:
            corr = np.corrcoef(weights, choices)[0, 1]
            # Convert correlation to pseudo-AUC
            return 0.5 + 0.5 * np.abs(corr) if np.isfinite(corr) else 0.5
        except:
            return 0.5


# def assign_motif_label(scores: Dict[str, float], audit_cfg: Dict[str, Any]) -> str:
#     """
#     Assign functional label to motif based on audit scores.
#     """
#     # Thresholds from config
#     event_thresh = audit_cfg.get('event_enrichment_thresh', 2.0)
#     choice_thresh = audit_cfg.get('choice_auc_thresh', 0.65)
#     isi_thresh = audit_cfg.get('isi_coupling_thresh', 0.1)
#     outcome_thresh = audit_cfg.get('outcome_auc_thresh', 0.65)
    
#     labels = []
    
#     # Check event enrichments
#     events_with_enrichment = []
#     for event in ['start_flash_1', 'end_flash_1', 'start_flash_2', 'end_flash_2', 'choice_start']:
#         score_key = f'{event}_enrichment'
#         if score_key in scores and scores[score_key] > event_thresh:
#             if 'flash' in event:
#                 events_with_enrichment.append('stimulus')
#             elif 'choice' in event:
#                 events_with_enrichment.append('choice')
    
#     # Check choice information
#     if scores.get('choice_auc', 0.5) > choice_thresh:
#         labels.append('choice')
    
#     # Check outcome information
#     if scores.get('outcome_auc', 0.5) > outcome_thresh:
#         labels.append('outcome')
    
#     # Check ISI coupling (expectation/timing)
#     if scores.get('isi_coupling', 0.0) > isi_thresh:
#         labels.append('expectation')
    
#     # Add stimulus labels
#     labels.extend(events_with_enrichment)
    
#     # Return most specific label
#     unique_labels = list(set(labels))
    
#     if not unique_labels:
#         return 'artifact'
#     elif len(unique_labels) == 1:
#         return unique_labels[0]
#     else:
#         # Return compound label for multi-functional motifs
#         return '_'.join(sorted(unique_labels))


def build_roi_sets_from_motifs(cp_result: Dict[str, Any], audit_result: Dict[str, Any], 
                              cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Convert continuous motif loadings to discrete ROI sets.
    """
    print("=== BUILDING ROI SETS FROM MOTIFS ===")
    
    roi_sets_cfg = cfg.get('roi_sets', {})
    threshold_method = roi_sets_cfg.get('threshold_method', 'percentile')
    threshold_percentile = roi_sets_cfg.get('threshold_percentile', 75)
    
    roi_loadings = cp_result['roi_loadings']  # (n_rois, n_motifs)
    motif_labels = audit_result['motif_labels']
    n_rois, n_motifs = roi_loadings.shape
    
    print(f"Input: {n_rois} ROIs, {n_motifs} motifs")
    print(f"Threshold method: {threshold_method}")
    
    roi_sets = {}
    
    # Process each motif
    for k, label in enumerate(motif_labels):
        if label == 'artifact':
            print(f"  Motif {k} ({label}): skipping artifact")
            continue
        
        loadings = roi_loadings[:, k]
        
        # Determine threshold
        if threshold_method == 'percentile':
            threshold = np.percentile(loadings, threshold_percentile)
        elif threshold_method == 'otsu':
            threshold = otsu_threshold(loadings)
        elif threshold_method == 'mean_plus_std':
            threshold = np.mean(loadings) + np.std(loadings)
        else:
            threshold = 0.1  # Default fallback
        
        # Apply threshold to get ROI set
        roi_mask = loadings > threshold
        roi_indices = np.where(roi_mask)[0]
        
        # Store with unique name if label already exists
        set_name = label
        counter = 1
        while set_name in roi_sets:
            set_name = f"{label}_{counter}"
            counter += 1
        
        roi_sets[set_name] = roi_indices
        
        print(f"  Motif {k} ({label}): {len(roi_indices)} ROIs (threshold: {threshold:.4f})")
    
    # Compute meaningful intersections
    print("\nComputing ROI set intersections:")
    intersections = compute_roi_intersections(roi_sets)
    roi_sets.update(intersections)
    
    # Summary
    print(f"\n=== ROI SETS SUMMARY ===")
    for set_name, roi_indices in roi_sets.items():
        print(f"  {set_name}: {len(roi_indices)} ROIs")
    
    print("=== ROI SET BUILDING COMPLETE ===\n")
    
    return roi_sets


def otsu_threshold(values: np.ndarray) -> float:
    """
    Simple Otsu-like threshold for continuous values.
    """
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.mean(values) if len(values) > 0 else 0.0
    
    # Use 75th percentile as approximation to Otsu threshold
    return np.percentile(values, 75)


def compute_roi_intersections(roi_sets: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute meaningful intersections between ROI sets.
    """
    intersections = {}
    
    set_names = list(roi_sets.keys())
    
    for i, name1 in enumerate(set_names):
        for name2 in set_names[i+1:]:
            if name1 != name2 and not ('∩' in name1 or '∩' in name2):  # Don't intersect intersections
                intersection = np.intersect1d(roi_sets[name1], roi_sets[name2])
                if len(intersection) >= 3:  # Only keep substantial intersections
                    intersections[f'{name1}∩{name2}'] = intersection
                    print(f"    {name1} ∩ {name2}: {len(intersection)} ROIs")
    
    return intersections




# Add these functions to the end of sid_roi_labeling.py

# def plot_motif_overview(cp_result: Dict[str, Any], audit_result: Dict[str, Any], 
#                        roi_sets: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> plt.Figure:
#     """
#     Create overview plot showing discovered motifs and their characteristics.
#     """
#     print("=== PLOTTING MOTIF OVERVIEW ===")
    
#     motif_timecourses = cp_result['motif_timecourses']  # (n_bins, n_motifs)
#     roi_loadings = cp_result['roi_loadings']            # (n_rois, n_motifs)
#     trial_weights = cp_result['trial_weights']          # (n_trials, n_motifs)
#     time_grid = cp_result['time_grid']
#     motif_labels = audit_result['motif_labels']
#     motif_scores = audit_result['motif_scores']
    
#     n_motifs = len(motif_labels)
    
#     # Create figure with subplots
#     fig = plt.figure(figsize=(16, 4 * n_motifs))
#     gs = GridSpec(n_motifs, 4, figure=fig, hspace=0.3, wspace=0.3)
    
#     for k in range(n_motifs):
#         label = motif_labels[k]
#         scores = motif_scores[f'motif_{k}']
        
#         # 1. Motif timecourse
#         ax1 = fig.add_subplot(gs[k, 0])
#         ax1.plot(time_grid, motif_timecourses[:, k], 'b-', linewidth=2)
#         ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
#         ax1.set_title(f'Motif {k}: {label}')
#         ax1.set_xlabel('Time (s)')
#         ax1.set_ylabel('Amplitude')
#         ax1.grid(True, alpha=0.3)
        
#         # 2. ROI loadings histogram
#         ax2 = fig.add_subplot(gs[k, 1])
#         loadings = roi_loadings[:, k]
#         ax2.hist(loadings, bins=30, alpha=0.7, color='orange')
        
#         # Show threshold if this motif has an ROI set
#         set_names = [name for name in roi_sets.keys() if label in name and '∩' not in name]
#         if set_names:
#             set_name = set_names[0]
#             roi_indices = roi_sets[set_name]
#             if len(roi_indices) > 0:
#                 threshold = np.min(loadings[roi_indices])
#                 ax2.axvline(threshold, color='red', linestyle='--', 
#                            label=f'Threshold\n({len(roi_indices)} ROIs)')
#                 ax2.legend()
        
#         ax2.set_title(f'ROI Loadings')
#         ax2.set_xlabel('Loading')
#         ax2.set_ylabel('Count')
#         ax2.grid(True, alpha=0.3)
        
#         # 3. Trial weights histogram
#         ax3 = fig.add_subplot(gs[k, 2])
#         weights = trial_weights[:, k]
#         ax3.hist(weights, bins=30, alpha=0.7, color='green')
#         ax3.set_title(f'Trial Weights')
#         ax3.set_xlabel('Weight')
#         ax3.set_ylabel('Count')
#         ax3.grid(True, alpha=0.3)
        
#         # 4. Audit scores
#         ax4 = fig.add_subplot(gs[k, 3])
        
#         # Extract key scores for plotting
#         score_names = []
#         score_values = []
        
#         for score_name, score_val in scores.items():
#             if 'enrichment' in score_name:
#                 score_names.append(score_name.replace('_enrichment', ''))
#                 score_values.append(score_val)
#             elif score_name in ['choice_auc', 'stimulus_auc', 'outcome_auc', 'nochoice_auc', 'isi_coupling']:
#                 score_names.append(score_name)
#                 score_values.append(score_val)
        
#         if score_names:
#             bars = ax4.barh(range(len(score_names)), score_values, alpha=0.7)
#             ax4.set_yticks(range(len(score_names)))
#             ax4.set_yticklabels(score_names, fontsize=8)
#             ax4.set_xlabel('Score')
#             ax4.set_title('Audit Scores')
#             ax4.grid(True, alpha=0.3)
            
#             # Color bars based on significance
#             for i, (bar, val) in enumerate(bars, score_values):
#                 if 'auc' in score_names[i]:
#                     if val > 0.65:
#                         bar.set_color('red')
#                     elif val < 0.35:
#                         bar.set_color('blue')
#                 elif 'coupling' in score_names[i]:
#                     if val > 0.1:
#                         bar.set_color('purple')
#                 else:  # enrichment scores
#                     if val > 2.0:
#                         bar.set_color('red')
#                     elif val > 1.0:
#                         bar.set_color('orange')
    
#     plt.suptitle('Motif Discovery Overview', fontsize=16, y=0.98)
#     print("=== MOTIF OVERVIEW PLOT COMPLETE ===\n")
    
#     return fig


# ...existing code...

def plot_motif_overview(cp_result: Dict[str, Any], audit_result: Dict[str, Any], 
                       roi_sets: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> plt.Figure:
    """
    Create overview plot showing discovered motifs and their characteristics.
    """
    print("=== PLOTTING MOTIF OVERVIEW ===")
    
    motif_timecourses = cp_result['motif_timecourses']  # (n_bins, n_motifs)
    roi_loadings = cp_result['roi_loadings']            # (n_rois, n_motifs)
    trial_weights = cp_result['trial_weights']          # (n_trials, n_motifs)
    time_grid = cp_result['time_grid']
    motif_labels = audit_result['motif_labels']
    motif_scores = audit_result['motif_scores']
    
    n_motifs = len(motif_labels)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 4 * n_motifs))
    gs = GridSpec(n_motifs, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for k in range(n_motifs):
        label = motif_labels[k]
        scores = motif_scores[f'motif_{k}']
        
        # 1. Motif timecourse
        ax1 = fig.add_subplot(gs[k, 0])
        ax1.plot(time_grid, motif_timecourses[:, k], 'b-', linewidth=2)
        ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax1.set_title(f'Motif {k}: {label}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # 2. ROI loadings histogram
        ax2 = fig.add_subplot(gs[k, 1])
        loadings = roi_loadings[:, k]
        ax2.hist(loadings, bins=30, alpha=0.7, color='orange')
        
        # Show threshold if this motif has an ROI set
        set_names = [name for name in roi_sets.keys() if label in name and '∩' not in name]
        if set_names:
            set_name = set_names[0]
            roi_indices = roi_sets[set_name]
            if len(roi_indices) > 0:
                threshold = np.min(loadings[roi_indices])
                ax2.axvline(threshold, color='red', linestyle='--', 
                           label=f'Threshold\n({len(roi_indices)} ROIs)')
                ax2.legend()
        
        ax2.set_title(f'ROI Loadings')
        ax2.set_xlabel('Loading')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trial weights histogram
        ax3 = fig.add_subplot(gs[k, 2])
        weights = trial_weights[:, k]
        ax3.hist(weights, bins=30, alpha=0.7, color='green')
        ax3.set_title(f'Trial Weights')
        ax3.set_xlabel('Weight')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)
        
        # 4. Audit scores
        ax4 = fig.add_subplot(gs[k, 3])
        
        # Extract key scores for plotting
        score_names = []
        score_values = []
        
        for score_name, score_val in scores.items():
            if 'enrichment' in score_name:
                score_names.append(score_name.replace('_enrichment', ''))
                score_values.append(score_val)
            elif score_name in ['choice_auc', 'stimulus_auc', 'outcome_auc', 'nochoice_auc', 'isi_coupling']:
                score_names.append(score_name)
                score_values.append(score_val)
        
        if score_names:
            bars = ax4.barh(range(len(score_names)), score_values, alpha=0.7)
            ax4.set_yticks(range(len(score_names)))
            ax4.set_yticklabels(score_names, fontsize=8)
            ax4.set_xlabel('Score')
            ax4.set_title('Audit Scores')
            ax4.grid(True, alpha=0.3)
            
            # Color bars based on significance - FIXED LINE
            for i, (bar, val) in enumerate(zip(bars, score_values)):
                if 'auc' in score_names[i]:
                    if val > 0.65:
                        bar.set_color('red')
                    elif val < 0.35:
                        bar.set_color('blue')
                elif 'coupling' in score_names[i]:
                    if val > 0.1:
                        bar.set_color('purple')
                else:  # enrichment scores
                    if val > 2.0:
                        bar.set_color('red')
                    elif val > 1.0:
                        bar.set_color('orange')
    
    plt.suptitle('Motif Discovery Overview', fontsize=16, y=0.98)
    print("=== MOTIF OVERVIEW PLOT COMPLETE ===\n")
    
    return fig

# ...rest of existing code...

def plot_roi_set_spatial_maps(roi_sets: Dict[str, np.ndarray], trial_data: Dict[str, Any], 
                             cfg: Dict[str, Any]) -> plt.Figure:
    """
    Plot spatial maps of ROI sets overlaid on mean image (if available).
    """
    print("=== PLOTTING ROI SET SPATIAL MAPS ===")
    
    # Try to get spatial information from trial data
    if 'imaging_metadata' in trial_data and 'mean_image' in trial_data['imaging_metadata']:
        mean_img = trial_data['imaging_metadata']['mean_image']
        print(f"Using mean image: {mean_img.shape}")
    else:
        print("No mean image found, creating dummy spatial layout")
        # Create dummy spatial layout for visualization
        n_rois = max(max(indices) for indices in roi_sets.values()) + 1
        grid_size = int(np.ceil(np.sqrt(n_rois)))
        mean_img = np.zeros((grid_size * 10, grid_size * 10))
    
    # Filter out intersection sets for cleaner display
    main_sets = {name: indices for name, indices in roi_sets.items() if '∩' not in name}
    
    n_sets = len(main_sets)
    if n_sets == 0:
        print("No ROI sets to plot")
        return None
    
    # Create subplot grid
    cols = min(4, n_sets)
    rows = (n_sets + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n_sets == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    # Color palette for different sets
    colors = plt.cm.Set1(np.linspace(0, 1, n_sets))
    
    for i, (set_name, roi_indices) in enumerate(main_sets.items()):
        ax = axes[i]
        
        # Show mean image as background
        ax.imshow(mean_img, cmap='gray', alpha=0.5)
        
        if 'imaging_metadata' in trial_data and 'roi_centroids' in trial_data['imaging_metadata']:
            # Use actual ROI positions if available
            centroids = trial_data['imaging_metadata']['roi_centroids']
            if len(centroids) > max(roi_indices):
                x_coords = [centroids[idx][1] for idx in roi_indices]  # x = column
                y_coords = [centroids[idx][0] for idx in roi_indices]  # y = row
                ax.scatter(x_coords, y_coords, c=[colors[i]], s=20, alpha=0.8)
        else:
            # Create dummy positions
            grid_size = int(np.ceil(np.sqrt(len(roi_indices))))
            positions = [(idx % grid_size, idx // grid_size) for idx in roi_indices]
            x_coords = [pos[0] * 10 + 5 for pos in positions]
            y_coords = [pos[1] * 10 + 5 for pos in positions]
            ax.scatter(x_coords, y_coords, c=[colors[i]], s=50, alpha=0.8)
        
        ax.set_title(f'{set_name}\n{len(roi_indices)} ROIs')
        ax.set_aspect('equal')
        
    # Hide unused subplots
    for i in range(n_sets, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('ROI Set Spatial Distribution', fontsize=14)
    plt.tight_layout()
    
    print("=== ROI SPATIAL MAPS COMPLETE ===\n")
    
    return fig


def plot_roi_set_overlap_matrix(roi_sets: Dict[str, np.ndarray]) -> plt.Figure:
    """
    Plot overlap matrix between ROI sets using Jaccard similarity.
    """
    print("=== PLOTTING ROI SET OVERLAP MATRIX ===")
    
    # Filter to main sets (no intersections)
    main_sets = {name: indices for name, indices in roi_sets.items() if '∩' not in name}
    
    set_names = list(main_sets.keys())
    n_sets = len(set_names)
    
    if n_sets < 2:
        print("Need at least 2 sets for overlap analysis")
        return None
    
    # Compute Jaccard similarity matrix
    jaccard_matrix = np.zeros((n_sets, n_sets))
    
    for i, name1 in enumerate(set_names):
        for j, name2 in enumerate(set_names):
            set1 = set(main_sets[name1])
            set2 = set(main_sets[name2])
            
            if len(set1) == 0 and len(set2) == 0:
                jaccard_matrix[i, j] = 1.0
            elif len(set1) == 0 or len(set2) == 0:
                jaccard_matrix[i, j] = 0.0
            else:
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                jaccard_matrix[i, j] = intersection / union if union > 0 else 0.0
    
    # Plot matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(jaccard_matrix, cmap='Blues', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(n_sets):
        for j in range(n_sets):
            text = ax.text(j, i, f'{jaccard_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=10)
    
    ax.set_xticks(range(n_sets))
    ax.set_yticks(range(n_sets))
    ax.set_xticklabels(set_names, rotation=45, ha='right')
    ax.set_yticklabels(set_names)
    
    ax.set_title('ROI Set Overlap (Jaccard Similarity)')
    ax.set_xlabel('ROI Set')
    ax.set_ylabel('ROI Set')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Jaccard Similarity')
    
    plt.tight_layout()
    
    print("=== ROI OVERLAP MATRIX COMPLETE ===\n")
    
    return fig


def plot_trial_motif_activations(cp_result: Dict[str, Any], trial_info: Dict[str, Any], 
                                audit_result: Dict[str, Any], cfg: Dict[str, Any]) -> plt.Figure:
    """
    Plot trial-by-trial motif activations grouped by behavioral conditions.
    """
    print("=== PLOTTING TRIAL MOTIF ACTIVATIONS ===")
    
    trial_weights = cp_result['trial_weights']  # (n_trials, n_motifs)
    motif_labels = audit_result['motif_labels']
    n_motifs = len(motif_labels)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # 1. Activations by ISI
    if 'isi' in trial_info:
        ax = axes[0]
        isi_values = trial_info['isi']
        valid_mask = np.isfinite(isi_values)
        
        if valid_mask.sum() > 0:
            unique_isis = np.unique(isi_values[valid_mask])
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_isis)))
            
            for k in range(min(4, n_motifs)):  # Show first 4 motifs
                for i, isi in enumerate(unique_isis):
                    isi_mask = valid_mask & (isi_values == isi)
                    weights = trial_weights[isi_mask, k]
                    
                    x_pos = k + (i - len(unique_isis)/2) * 0.1
                    ax.scatter([x_pos] * len(weights), weights, 
                             c=[colors[i]], alpha=0.6, s=20, label=f'ISI {isi:.2f}' if k == 0 else '')
            
            ax.set_xlabel('Motif')
            ax.set_ylabel('Activation Weight')
            ax.set_title('Motif Activations by ISI')
            ax.set_xticks(range(min(4, n_motifs)))
            ax.set_xticklabels([f'M{k}\n{motif_labels[k][:8]}' for k in range(min(4, n_motifs))])
            if len(unique_isis) <= 10:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Activations by choice
    if 'is_right_choice' in trial_info:
        ax = axes[1]
        choice_values = trial_info['is_right_choice']
        valid_mask = np.isfinite(choice_values)
        
        if valid_mask.sum() > 0:
            for k in range(min(4, n_motifs)):
                left_weights = trial_weights[valid_mask & (choice_values == 0), k]
                right_weights = trial_weights[valid_mask & (choice_values == 1), k]
                
                positions = [k - 0.2, k + 0.2]
                bp = ax.boxplot([left_weights, right_weights], positions=positions, widths=0.3,
                               patch_artist=True, labels=['L', 'R'])
                bp['boxes'][0].set_facecolor('blue')
                bp['boxes'][1].set_facecolor('red')
            
            ax.set_xlabel('Motif')
            ax.set_ylabel('Activation Weight')
            ax.set_title('Motif Activations by Choice (L/R)')
            ax.set_xticks(range(min(4, n_motifs)))
            ax.set_xticklabels([f'M{k}\n{motif_labels[k][:8]}' for k in range(min(4, n_motifs))])
    
    # 3. Activations by outcome
    if 'rewarded' in trial_info:
        ax = axes[2]
        reward_values = trial_info['rewarded']
        valid_mask = np.isfinite(reward_values) if np.issubdtype(reward_values.dtype, np.number) else np.ones(len(reward_values), dtype=bool)
        
        if valid_mask.sum() > 0:
            for k in range(min(4, n_motifs)):
                unrewarded_weights = trial_weights[valid_mask & (reward_values == 0), k]
                rewarded_weights = trial_weights[valid_mask & (reward_values == 1), k]
                
                positions = [k - 0.2, k + 0.2]
                bp = ax.boxplot([unrewarded_weights, rewarded_weights], positions=positions, widths=0.3,
                               patch_artist=True, labels=['No', 'Yes'])
                bp['boxes'][0].set_facecolor('gray')
                bp['boxes'][1].set_facecolor('gold')
            
            ax.set_xlabel('Motif')
            ax.set_ylabel('Activation Weight')
            ax.set_title('Motif Activations by Reward')
            ax.set_xticks(range(min(4, n_motifs)))
            ax.set_xticklabels([f'M{k}\n{motif_labels[k][:8]}' for k in range(min(4, n_motifs))])
    
    # 4. Overall motif activation distribution
    ax = axes[3]
    bp = ax.boxplot([trial_weights[:, k] for k in range(min(6, n_motifs))], 
                   patch_artist=True)
    
    # Color boxes by motif type
    type_colors = {'stimulus': 'red', 'choice': 'blue', 'outcome': 'gold', 
                  'expectation': 'purple', 'engagement': 'green', 'artifact': 'gray'}
    
    for k, box in enumerate(bp['boxes']):
        if k < len(motif_labels):
            label = motif_labels[k]
            main_type = label.split('_')[0]
            color = type_colors.get(main_type, 'lightblue')
            box.set_facecolor(color)
    
    ax.set_xlabel('Motif')
    ax.set_ylabel('Activation Weight')
    ax.set_title('Overall Motif Activation Distribution')
    ax.set_xticklabels([f'M{k}\n{motif_labels[k][:8]}' for k in range(min(6, n_motifs))])
    
    plt.tight_layout()
    
    print("=== TRIAL MOTIF ACTIVATIONS COMPLETE ===\n")
    
    return fig


def print_roi_set_summary(roi_sets: Dict[str, np.ndarray], audit_result: Dict[str, Any]):
    """
    Print detailed summary of ROI sets and their characteristics.
    """
    print("=== ROI SET SUMMARY ===")
    
    motif_scores = audit_result['motif_scores']
    
    # Separate main sets from intersections
    main_sets = {name: indices for name, indices in roi_sets.items() if '∩' not in name}
    intersection_sets = {name: indices for name, indices in roi_sets.items() if '∩' in name}
    
    print(f"\nMain ROI Sets ({len(main_sets)}):")
    print("-" * 60)
    
    for set_name, roi_indices in main_sets.items():
        print(f"\n{set_name.upper()}:")
        print(f"  ROIs: {len(roi_indices)} total")
        print(f"  Indices: {roi_indices[:10].tolist()}{'...' if len(roi_indices) > 10 else ''}")
        
        # Find corresponding motif scores
        motif_idx = None
        for key in motif_scores.keys():
            if set_name in audit_result['motif_labels'][int(key.split('_')[1])]:
                motif_idx = int(key.split('_')[1])
                break
        
        if motif_idx is not None:
            scores = motif_scores[f'motif_{motif_idx}']
            print(f"  Key scores:")
            
            # Show most relevant scores
            for score_name, score_val in scores.items():
                if 'enrichment' in score_name and score_val > 1.0:
                    print(f"    {score_name}: {score_val:.2f}")
                elif score_name in ['choice_auc', 'stimulus_auc', 'outcome_auc'] and abs(score_val - 0.5) > 0.1:
                    print(f"    {score_name}: {score_val:.3f}")
                elif score_name == 'isi_coupling' and score_val > 0.05:
                    print(f"    {score_name}: {score_val:.3f}")
    
    if intersection_sets:
        print(f"\nIntersection Sets ({len(intersection_sets)}):")
        print("-" * 60)
        
        for set_name, roi_indices in intersection_sets.items():
            parts = set_name.split('∩')
            set1_size = len(main_sets.get(parts[0], []))
            set2_size = len(main_sets.get(parts[1], []))
            
            jaccard = len(roi_indices) / (set1_size + set2_size - len(roi_indices)) if (set1_size + set2_size > len(roi_indices)) else 0
            
            print(f"\n{set_name}:")
            print(f"  ROIs: {len(roi_indices)} shared")
            print(f"  Jaccard similarity: {jaccard:.3f}")
            print(f"  Indices: {roi_indices[:10].tolist()}{'...' if len(roi_indices) > 10 else ''}")
    
    print("\n" + "=" * 60)


def save_roi_sets_to_file(roi_sets: Dict[str, np.ndarray], cp_result: Dict[str, Any], 
                         audit_result: Dict[str, Any], filepath: str):
    """
    Save ROI sets and analysis results to file for later use.
    """
    print(f"=== SAVING ROI SETS TO {filepath} ===")
    
    save_data = {
        'roi_sets': roi_sets,
        'cp_result': cp_result,
        'audit_result': audit_result,
        'metadata': {
            'n_motifs': len(audit_result['motif_labels']),
            'n_roi_sets': len(roi_sets),
            'total_rois_covered': len(np.unique(np.concatenate(list(roi_sets.values())))),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Saved ROI labeling results to: {filepath}")
    print(f"  {len(roi_sets)} ROI sets")
    print(f"  {len(audit_result['motif_labels'])} motifs")
    print(f"  {save_data['metadata']['total_rois_covered']} unique ROIs covered")


# Add a comprehensive analysis function
def run_full_roi_analysis_with_plots(trial_data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run complete ROI motif analysis with all plots and summaries.
    """
    print("=" * 80)
    print("RUNNING FULL ROI MOTIF ANALYSIS WITH VISUALIZATION")
    print("=" * 80)
    
    # Run main analysis pipeline
    X, time_grid, trial_info = prepare_trial_tensor(trial_data, cfg)
    cp_result = fit_cp_decomposition(X, time_grid, cfg)
    audit_result = audit_motifs_against_events(cp_result, trial_info, cfg)
    roi_sets = build_roi_sets_from_motifs(cp_result, audit_result, cfg)
    
    # Generate all plots
    print("\nGenerating visualization plots...")
    
    fig1 = plot_motif_overview(cp_result, audit_result, roi_sets, cfg)
    fig1.suptitle('Motif Discovery Overview', fontsize=16)
    
    fig2 = plot_roi_set_spatial_maps(roi_sets, trial_data, cfg)
    if fig2 is not None:
        fig2.suptitle('ROI Set Spatial Maps', fontsize=16)
    
    fig3 = plot_roi_set_overlap_matrix(roi_sets)
    if fig3 is not None:
        fig3.suptitle('ROI Set Overlap Matrix', fontsize=16)
    
    fig4 = plot_trial_motif_activations(cp_result, trial_info, audit_result, cfg)
    fig4.suptitle('Trial-by-Trial Motif Activations', fontsize=16)
    
    # Print summary
    print_roi_set_summary(roi_sets, audit_result)
    
    # Package results
    results = {
        'cp_result': cp_result,
        'audit_result': audit_result,
        'roi_sets': roi_sets,
        'trial_info': trial_info,
        'time_grid': time_grid,
        'config': cfg,
        'figures': {
            'motif_overview': fig1,
            'spatial_maps': fig2,
            'overlap_matrix': fig3,
            'trial_activations': fig4
        }
    }
    
    print("\n" + "=" * 80)
    print("ROI MOTIF ANALYSIS COMPLETE!")
    print("=" * 80)
    
    return results



















def plot_functional_raster_by_isi(X: np.ndarray, time_grid: np.ndarray, 
                                 trial_info: Dict[str, Any], roi_sets: Dict[str, np.ndarray],
                                 audit_result: Dict[str, Any], cp_result: Dict[str, Any],
                                 target_isi: float = None, cfg: Dict[str, Any] = None) -> plt.Figure:
    """
    Create hierarchical raster plot organized by functional ROI groups.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_rois, n_bins, n_trials)
        Neural activity tensor
    time_grid : np.ndarray
        Time vector
    trial_info : Dict
        Trial metadata including ISI, choice info
    roi_sets : Dict
        Functional ROI groupings
    target_isi : float, optional
        Specific ISI to filter trials (ms). If None, use all trials
    """
    print(f"=== PLOTTING FUNCTIONAL RASTER (ISI={target_isi}) ===")
    
    # Filter trials by ISI if specified
    if target_isi is not None and 'isi' in trial_info:
        isi_mask = np.abs(trial_info['isi'] - target_isi) < 50  # 50ms tolerance
        trial_indices = np.where(isi_mask)[0]
        print(f"Using {len(trial_indices)} trials with ISI ≈ {target_isi}ms")
    else:
        trial_indices = np.arange(X.shape[2])
        print(f"Using all {len(trial_indices)} trials")
    
    if len(trial_indices) < 5:
        print("Too few trials for meaningful raster plot")
        return None
    
    # Extract trial subset
    X_subset = X[:, :, trial_indices]
    trial_info_subset = {key: val[trial_indices] if isinstance(val, np.ndarray) else val 
                        for key, val in trial_info.items()}
    
    # Get main functional groups (exclude intersections for cleaner visualization)
    main_roi_sets = {name: indices for name, indices in roi_sets.items() if '∩' not in name}
    
    if len(main_roi_sets) == 0:
        print("No main ROI sets found")
        return None
    
    # Order ROI groups by importance (choice > stimulus > outcome > expectation > engagement)
    group_priority = {
        'choice': 1, 'stimulus': 2, 'outcome': 3, 'expectation': 4, 
        'engagement': 5, 'artifact': 99
    }
    
    def get_group_priority(name):
        for key in group_priority:
            if key in name.lower():
                return group_priority[key]
        return 50  # Default for unknown types
    
    sorted_groups = sorted(main_roi_sets.items(), key=lambda x: get_group_priority(x[0]))
    
    # Create figure with hierarchical subplots
    n_groups = len(sorted_groups)
    fig = plt.figure(figsize=(14, 2 * n_groups + 2))
    
    # Add space for behavioral traces at top
    gs = GridSpec(n_groups + 1, 1, height_ratios=[0.3] + [1] * n_groups, 
                  hspace=0.05, figure=fig)
    
    # Plot behavioral context at top
    ax_behav = fig.add_subplot(gs[0, 0])
    plot_behavioral_context(ax_behav, trial_info_subset, time_grid, target_isi)
    
    # Track ROI positions for reference
    roi_positions = {}  # Will map ROI index to (group_idx, position_in_group)
    current_roi_position = 0
    
    # Process each functional group
    for group_idx, (group_name, roi_indices) in enumerate(sorted_groups):
        print(f"Processing group: {group_name} ({len(roi_indices)} ROIs)")
        
        if len(roi_indices) == 0:
            continue
        
        # Order ROIs within group by motif loading strength
        ordered_roi_indices = order_rois_by_motif_strength(
            roi_indices, group_name, cp_result, audit_result
        )
        
        # Get neural data for this group
        group_data = X_subset[ordered_roi_indices, :, :]  # (group_rois, time, trials)
        
        # Create subplot for this group
        ax = fig.add_subplot(gs[group_idx + 1, 0])
        
        # Plot raster for this group
        plot_group_raster(ax, group_data, time_grid, trial_info_subset, 
                         group_name, ordered_roi_indices, target_isi)
        
        # Store ROI positions for reference
        for i, roi_idx in enumerate(ordered_roi_indices):
            roi_positions[roi_idx] = (group_idx, i)
        
        current_roi_position += len(ordered_roi_indices)
    
    # Add overall title and formatting
    title = f'Functional ROI Group Rasters'
    if target_isi is not None:
        title += f' (ISI = {target_isi}ms)'
    fig.suptitle(title, fontsize=14, y=0.98)
    
    # Add shared time axis
    ax.set_xlabel('Time (s)')
    
    print("=== FUNCTIONAL RASTER PLOT COMPLETE ===\n")
    
    return fig


def order_rois_by_motif_strength(roi_indices: np.ndarray, group_name: str,
                                cp_result: Dict[str, Any], audit_result: Dict[str, Any]) -> np.ndarray:
    """
    Order ROIs within a functional group by their motif participation strength.
    """
    roi_loadings = cp_result['roi_loadings']  # (n_rois, n_motifs)
    motif_labels = audit_result['motif_labels']
    
    # Find which motifs correspond to this group
    group_motifs = []
    for motif_idx, label in enumerate(motif_labels):
        if any(group_key in label.lower() for group_key in group_name.lower().split('_')):
            group_motifs.append(motif_idx)
    
    if len(group_motifs) == 0:
        # Fallback: use overall activity level
        activity_strength = np.mean(roi_loadings[roi_indices, :], axis=1)
    else:
        # Use strength in relevant motifs
        activity_strength = np.mean(roi_loadings[roi_indices, :][:, group_motifs], axis=1)
    
    # Sort by strength (strongest first)
    sort_order = np.argsort(activity_strength)[::-1]
    
    return roi_indices[sort_order]


def plot_behavioral_context(ax: plt.Axes, trial_info: Dict[str, Any], 
                           time_grid: np.ndarray, target_isi: float):
    """
    Plot behavioral context (choice patterns, ISI info) at top of raster.
    """
    n_trials = len(trial_info.get('is_right_choice', []))
    
    # Create behavioral indicator matrix
    behav_matrix = np.zeros((3, n_trials))  # 3 rows: choice, stimulus, outcome
    
    # Row 0: Choice (blue=left, red=right, gray=no choice)
    if 'is_right_choice' in trial_info and 'did_not_choose' in trial_info:
        choices = trial_info['is_right_choice']
        no_choice = trial_info['did_not_choose']
        
        behav_matrix[0, :] = choices  # 0=left, 1=right
        behav_matrix[0, no_choice == 1] = 0.5  # Gray for no choice
    
    # Row 1: Stimulus type (light blue=left type, light red=right type)
    if 'is_right' in trial_info:
        behav_matrix[1, :] = trial_info['is_right']
    
    # Row 2: Outcome (dark=unrewarded, bright=rewarded)  
    if 'rewarded' in trial_info:
        behav_matrix[2, :] = trial_info['rewarded']
    
    # Plot behavioral matrix
    im = ax.imshow(behav_matrix, aspect='auto', cmap='RdBu_r', 
                   extent=[0, n_trials-1, -0.5, 2.5], vmin=0, vmax=1)
    
    # Labels and formatting
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Choice', 'Stimulus', 'Outcome'], fontsize=10)
    ax.set_xlim(0, n_trials-1)
    ax.set_title(f'Behavioral Context (n={n_trials} trials)', fontsize=12)
    ax.set_xticks([])  # Remove x-ticks from behavioral plot
    
    # Add colorbar for reference
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Left/Unrewarded → Right/Rewarded', fontsize=8)


def plot_group_raster(ax: plt.Axes, group_data: np.ndarray, time_grid: np.ndarray,
                     trial_info: Dict[str, Any], group_name: str, roi_indices: np.ndarray,
                     target_isi: float):
    """
    Plot raster for a single functional group.
    """
    n_rois, n_bins, n_trials = group_data.shape
    
    # Smooth data for cleaner visualization
    smoothed_data = gaussian_filter1d(group_data, sigma=1.0, axis=1)
    
    # Z-score normalize each ROI for comparable visualization
    for roi_idx in range(n_rois):
        roi_data = smoothed_data[roi_idx, :, :].flatten()
        finite_mask = np.isfinite(roi_data)
        if finite_mask.sum() > 10:
            mean_val = np.mean(roi_data[finite_mask])
            std_val = np.std(roi_data[finite_mask])
            if std_val > 0:
                smoothed_data[roi_idx, :, :] = (smoothed_data[roi_idx, :, :] - mean_val) / std_val
    
    # Create raster plot
    # Reshape for plotting: (n_rois * n_trials, n_bins)
    plot_data = np.zeros((n_rois * n_trials, n_bins))
    
    for trial_idx in range(n_trials):
        start_row = trial_idx * n_rois
        end_row = start_row + n_rois
        plot_data[start_row:end_row, :] = smoothed_data[:, :, trial_idx]
    
    # Plot with trial groupings
    im = ax.imshow(plot_data, aspect='auto', cmap='RdBu_r', 
                   extent=[time_grid[0], time_grid[-1], 0, n_rois * n_trials],
                   vmin=-2, vmax=2, interpolation='nearest')
    
    # Add trial separators
    for trial_idx in range(1, n_trials):
        y_pos = trial_idx * n_rois
        ax.axhline(y_pos, color='white', linewidth=1, alpha=0.8)
    
    # Add ROI group separators (every 10 ROIs for readability)
    for roi_group in range(10, n_rois, 10):
        for trial_idx in range(n_trials):
            y_pos = trial_idx * n_rois + roi_group
            ax.axhline(y_pos, color='gray', linewidth=0.5, alpha=0.5)
    
    # Formatting
    ax.set_ylabel(f'{group_name}\n({n_rois} ROIs)', fontsize=10, rotation=0, 
                  ha='right', va='center')
    ax.set_xlim(time_grid[0], time_grid[-1])
    ax.set_ylim(0, n_rois * n_trials)
    
    # Remove x-ticks except for bottom plot
    ax.set_xticks([])
    
    # Add vertical lines for key time points if available
    if 'lick_start' in trial_info:
        # Show median lick time
        lick_times = trial_info['lick_start']
        finite_licks = lick_times[np.isfinite(lick_times)]
        if len(finite_licks) > 0:
            median_lick = np.median(finite_licks)
            ax.axvline(median_lick, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add ISI marker if specified
    if target_isi is not None:
        # Assuming stimulus 1 at 0.5s, stimulus 2 at 0.5 + ISI/1000
        f1_time = 0.5
        f2_time = f1_time + target_isi / 1000
        ax.axvline(f1_time, color='blue', linestyle='-', alpha=0.7, linewidth=1)
        ax.axvline(f2_time, color='blue', linestyle='-', alpha=0.7, linewidth=1)


def plot_isi_comparison_rasters(X: np.ndarray, time_grid: np.ndarray,
                               trial_info: Dict[str, Any], roi_sets: Dict[str, np.ndarray],
                               audit_result: Dict[str, Any], cp_result: Dict[str, Any],
                               target_isis: List[float] = None, cfg: Dict[str, Any] = None) -> plt.Figure:
    """
    Create side-by-side raster comparison across different ISI values.
    """
    print("=== PLOTTING ISI COMPARISON RASTERS ===")
    
    if target_isis is None:
        # Use a subset of available ISIs
        unique_isis = np.unique(trial_info['isi'])
        target_isis = [200, 450, 700, 1700, 2300]  # Representative range
        target_isis = [isi for isi in target_isis if isi in unique_isis]
    
    if len(target_isis) < 2:
        print("Need at least 2 ISI values for comparison")
        return None
    
    print(f"Comparing ISIs: {target_isis}")
    
    # Create subplot grid
    n_isis = len(target_isis)
    fig = plt.figure(figsize=(5 * n_isis, 12))
    
    for i, isi in enumerate(target_isis):
        print(f"  Processing ISI {isi}ms...")
        
        # Create individual raster for this ISI
        # Use tight subplot positioning
        subplot_width = 1.0 / n_isis
        subplot_left = i * subplot_width
        
        # Create temporary figure for single ISI
        temp_fig = plot_functional_raster_by_isi(
            X, time_grid, trial_info, roi_sets, audit_result, cp_result, 
            target_isi=isi, cfg=cfg
        )
        
        if temp_fig is not None:
            # Copy the content to main figure (simplified approach)
            ax_main = fig.add_subplot(1, n_isis, i + 1)
            ax_main.text(0.5, 0.5, f'ISI {isi}ms\n(Full raster)', 
                        ha='center', va='center', transform=ax_main.transAxes,
                        fontsize=12)
            ax_main.set_title(f'ISI = {isi}ms')
            
            plt.close(temp_fig)  # Clean up temporary figure
    
    fig.suptitle('Functional ROI Activity Across ISI Values', fontsize=16)
    plt.tight_layout()
    
    print("=== ISI COMPARISON RASTERS COMPLETE ===\n")
    
    return fig


def analyze_temporal_dynamics_by_group(cp_result: Dict[str, Any], audit_result: Dict[str, Any],
                                      roi_sets: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Analyze temporal dynamics of each functional ROI group.
    """
    print("=== ANALYZING TEMPORAL DYNAMICS BY GROUP ===")
    
    motif_timecourses = cp_result['motif_timecourses']  # (n_bins, n_motifs)
    time_grid = cp_result['time_grid']
    motif_labels = audit_result['motif_labels']
    
    dynamics_results = {}
    
    # Filter to main ROI sets
    main_roi_sets = {name: indices for name, indices in roi_sets.items() if '∩' not in name}
    
    for set_name, roi_indices in main_roi_sets.items():
        print(f"\nAnalyzing {set_name}:")
        
        # Find corresponding motifs for this ROI set
        corresponding_motifs = []
        for motif_idx, label in enumerate(motif_labels):
            if any(key in label.lower() for key in set_name.lower().split('_')):
                corresponding_motifs.append(motif_idx)
        
        if len(corresponding_motifs) == 0:
            print(f"  No corresponding motifs found")
            continue
        
        # Analyze temporal properties
        group_dynamics = {}
        
        for motif_idx in corresponding_motifs:
            timecourse = motif_timecourses[:, motif_idx]
            
            # Find peak time
            peak_idx = np.argmax(np.abs(timecourse))
            peak_time = time_grid[peak_idx]
            peak_amplitude = timecourse[peak_idx]
            
            # Find onset/offset times (when signal crosses 10% of peak)
            threshold = 0.1 * np.max(np.abs(timecourse))
            above_threshold = np.abs(timecourse) > threshold
            
            if above_threshold.sum() > 0:
                onset_idx = np.where(above_threshold)[0][0]
                offset_idx = np.where(above_threshold)[0][-1]
                onset_time = time_grid[onset_idx]
                offset_time = time_grid[offset_idx]
                duration = offset_time - onset_time
            else:
                onset_time = offset_time = duration = np.nan
            
            # Compute temporal center of mass
            weights = np.abs(timecourse)
            com_time = np.sum(time_grid * weights) / np.sum(weights) if np.sum(weights) > 0 else np.nan
            
            group_dynamics[f'motif_{motif_idx}'] = {
                'peak_time': peak_time,
                'peak_amplitude': peak_amplitude,
                'onset_time': onset_time,
                'offset_time': offset_time,
                'duration': duration,
                'center_of_mass': com_time,
                'label': motif_labels[motif_idx]
            }
            
            print(f"  Motif {motif_idx} ({motif_labels[motif_idx]}):")
            print(f"    Peak: {peak_time:.3f}s (amp: {peak_amplitude:.3f})")
            print(f"    Duration: {duration:.3f}s ({onset_time:.3f} - {offset_time:.3f}s)")
            print(f"    Center of mass: {com_time:.3f}s")
        
        dynamics_results[set_name] = group_dynamics
    
    print("=== TEMPORAL DYNAMICS ANALYSIS COMPLETE ===\n")
    
    return dynamics_results


# Add these to the main analysis section:
def run_raster_analysis(results: Dict[str, Any], target_isis: List[float] = None) -> Dict[str, Any]:
    """
    Run comprehensive raster analysis using discovered functional groups.
    """
    print("=" * 80)
    print("RUNNING FUNCTIONAL RASTER ANALYSIS")
    print("=" * 80)
    
    # Extract results
    X = results['X'] if 'X' in results else None
    time_grid = results['time_grid']
    trial_info = results['trial_info']
    roi_sets = results['roi_sets']
    audit_result = results['audit_result']
    cp_result = results['cp_result']
    cfg = results['config']
    
    if X is None:
        print("ERROR: Neural data tensor (X) not found in results")
        return results
    
    # Default ISI values for comparison
    if target_isis is None:
        unique_isis = np.unique(trial_info['isi'])
        target_isis = [200, 450, 700, 1700, 2300]
        target_isis = [isi for isi in target_isis if isi in unique_isis][:4]  # Limit to 4 for display
    
    # 1. Analyze temporal dynamics
    dynamics_results = analyze_temporal_dynamics_by_group(cp_result, audit_result, roi_sets)
    
    # 2. Create rasters for specific ISI values
    raster_figures = {}
    
    for isi in target_isis:
        print(f"Creating raster for ISI {isi}ms...")
        fig = plot_functional_raster_by_isi(
            X, time_grid, trial_info, roi_sets, audit_result, cp_result,
            target_isi=isi, cfg=cfg
        )
        if fig is not None:
            raster_figures[f'isi_{isi}'] = fig
    
    # 3. Create ISI comparison plot
    print("Creating ISI comparison plot...")
    comparison_fig = plot_isi_comparison_rasters(
        X, time_grid, trial_info, roi_sets, audit_result, cp_result,
        target_isis=target_isis, cfg=cfg
    )
    
    if comparison_fig is not None:
        raster_figures['isi_comparison'] = comparison_fig
    
    # Add to results
    results['raster_analysis'] = {
        'temporal_dynamics': dynamics_results,
        'raster_figures': raster_figures,
        'target_isis': target_isis
    }
    
    print("=" * 80)
    print("FUNCTIONAL RASTER ANALYSIS COMPLETE!")
    print("=" * 80)
    
    return results










































def plot_simplified_functional_raster(X: np.ndarray, time_grid: np.ndarray, 
                                     trial_info: Dict[str, Any], roi_sets: Dict[str, np.ndarray],
                                     audit_result: Dict[str, Any], cp_result: Dict[str, Any],
                                     target_isi: float = None, n_rois_per_group: int = 10) -> plt.Figure:
    """
    Create simplified, interpretable raster with fewer ROIs per group.
    """
    print(f"=== PLOTTING SIMPLIFIED FUNCTIONAL RASTER (ISI={target_isi}) ===")
    
    # Filter trials by ISI
    if target_isi is not None and 'isi' in trial_info:
        isi_mask = np.abs(trial_info['isi'] - target_isi) < 50
        trial_indices = np.where(isi_mask)[0]
        print(f"Using {len(trial_indices)} trials with ISI ≈ {target_isi}ms")
    else:
        trial_indices = np.arange(X.shape[2])
    
    if len(trial_indices) < 5:
        return None
    
    # Extract trial subset
    X_subset = X[:, :, trial_indices]
    trial_info_subset = {key: val[trial_indices] if isinstance(val, np.ndarray) else val 
                        for key, val in trial_info.items()}
    
    # Get main functional groups (exclude intersections)
    main_roi_sets = {name: indices for name, indices in roi_sets.items() if '∩' not in name}
    
    # Order groups by functional priority
    group_priority = {'choice': 1, 'stimulus': 2, 'outcome': 3, 'expectation': 4, 'engagement': 5}
    def get_priority(name):
        for key in group_priority:
            if key in name.lower():
                return group_priority[key]
        return 50
    
    sorted_groups = sorted(main_roi_sets.items(), key=lambda x: get_priority(x[0]))[:4]  # Limit to 4 groups
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    n_groups = len(sorted_groups)
    n_trials = len(trial_indices)
    
    # Grid layout: behavioral context + groups + average traces
    gs = GridSpec(n_groups + 2, 2, height_ratios=[0.4] + [1] * n_groups + [0.6], 
                  width_ratios=[3, 1], hspace=0.3, wspace=0.2, figure=fig)
    
    # 1. Behavioral context (spans both columns)
    ax_behav = fig.add_subplot(gs[0, :])
    plot_enhanced_behavioral_context(ax_behav, trial_info_subset, time_grid, target_isi)
    
    # 2. Process each functional group  
    group_averages = {}
    
    for group_idx, (group_name, roi_indices) in enumerate(sorted_groups):
        print(f"Processing {group_name}: {len(roi_indices)} ROIs")
        
        # Select top ROIs by motif strength (much fewer for clarity)
        ordered_rois = order_rois_by_motif_strength(roi_indices, group_name, cp_result, audit_result)
        selected_rois = ordered_rois[:n_rois_per_group]  # Take top 10 ROIs
        
        # Get data for selected ROIs
        group_data = X_subset[selected_rois, :, :]  # (10 ROIs, time, trials)
        
        # Create raster subplot
        ax_raster = fig.add_subplot(gs[group_idx + 1, 0])
        
        # Determine if this is the bottom raster plot
        is_bottom_plot = (group_idx == len(sorted_groups) - 1)
        
        plot_clean_group_raster(ax_raster, group_data, time_grid, trial_info_subset, 
                               group_name, selected_rois, target_isi, is_bottom_plot)
        
        # Create average trace subplot
        ax_avg = fig.add_subplot(gs[group_idx + 1, 1])
        avg_trace = plot_group_average_trace(ax_avg, group_data, time_grid, trial_info_subset, 
                                           group_name, target_isi)
        group_averages[group_name] = avg_trace
    
    # 3. Summary comparison (bottom row)
    ax_summary = fig.add_subplot(gs[-1, :])
    plot_group_comparison(ax_summary, group_averages, time_grid, target_isi)
    
    # Title
    title = f'Functional ROI Group Activity (ISI = {target_isi}ms, Top {n_rois_per_group} ROIs/group)'
    fig.suptitle(title, fontsize=14, y=0.98)
    
    return fig



# def plot_simplified_functional_raster(X: np.ndarray, time_grid: np.ndarray, 
#                                      trial_info: Dict[str, Any], roi_sets: Dict[str, np.ndarray],
#                                      audit_result: Dict[str, Any], cp_result: Dict[str, Any],
#                                      target_isi: float = None, n_rois_per_group: int = 10) -> plt.Figure:
#     """
#     Create simplified, interpretable raster with fewer ROIs per group.
#     """
#     print(f"=== PLOTTING SIMPLIFIED FUNCTIONAL RASTER (ISI={target_isi}) ===")
    
#     # Filter trials by ISI
#     if target_isi is not None and 'isi' in trial_info:
#         isi_mask = np.abs(trial_info['isi'] - target_isi) < 50
#         trial_indices = np.where(isi_mask)[0]
#         print(f"Using {len(trial_indices)} trials with ISI ≈ {target_isi}ms")
#     else:
#         trial_indices = np.arange(X.shape[2])
    
#     if len(trial_indices) < 5:
#         return None
    
#     # Extract trial subset
#     X_subset = X[:, :, trial_indices]
#     trial_info_subset = {key: val[trial_indices] if isinstance(val, np.ndarray) else val 
#                         for key, val in trial_info.items()}
    
#     # Get main functional groups (exclude intersections)
#     main_roi_sets = {name: indices for name, indices in roi_sets.items() if '∩' not in name}
    
#     # Order groups by functional priority
#     group_priority = {'choice': 1, 'stimulus': 2, 'outcome': 3, 'expectation': 4, 'engagement': 5}
#     def get_priority(name):
#         for key in group_priority:
#             if key in name.lower():
#                 return group_priority[key]
#         return 50
    
#     sorted_groups = sorted(main_roi_sets.items(), key=lambda x: get_priority(x[0]))[:4]  # Limit to 4 groups
    
#     # Create figure
#     fig = plt.figure(figsize=(16, 12))
#     n_groups = len(sorted_groups)
#     n_trials = len(trial_indices)
    
#     # Grid layout: behavioral context + groups + average traces
#     gs = GridSpec(n_groups + 2, 2, height_ratios=[0.4] + [1] * n_groups + [0.6], 
#                   width_ratios=[3, 1], hspace=0.3, wspace=0.2, figure=fig)
    
#     # 1. Behavioral context (spans both columns)
#     ax_behav = fig.add_subplot(gs[0, :])
#     plot_enhanced_behavioral_context(ax_behav, trial_info_subset, time_grid, target_isi)
    
#     # 2. Process each functional group  
#     group_averages = {}
    
#     for group_idx, (group_name, roi_indices) in enumerate(sorted_groups):
#         print(f"Processing {group_name}: {len(roi_indices)} ROIs")
        
#         # Select top ROIs by motif strength (much fewer for clarity)
#         ordered_rois = order_rois_by_motif_strength(roi_indices, group_name, cp_result, audit_result)
#         selected_rois = ordered_rois[:n_rois_per_group]  # Take top 10 ROIs
        
#         # Get data for selected ROIs
#         group_data = X_subset[selected_rois, :, :]  # (10 ROIs, time, trials)
        
#         # Create raster subplot
#         ax_raster = fig.add_subplot(gs[group_idx + 1, 0])
#         plot_clean_group_raster(ax_raster, group_data, time_grid, trial_info_subset, 
#                                group_name, selected_rois, target_isi)
        
#         # Create average trace subplot
#         ax_avg = fig.add_subplot(gs[group_idx + 1, 1])
#         avg_trace = plot_group_average_trace(ax_avg, group_data, time_grid, trial_info_subset, 
#                                            group_name, target_isi)
#         group_averages[group_name] = avg_trace
    
#     # 3. Summary comparison (bottom row)
#     ax_summary = fig.add_subplot(gs[-1, :])
#     plot_group_comparison(ax_summary, group_averages, time_grid, target_isi)
    
#     # Title
#     title = f'Functional ROI Group Activity (ISI = {target_isi}ms, Top {n_rois_per_group} ROIs/group)'
#     fig.suptitle(title, fontsize=14, y=0.98)
    
#     return fig


def plot_enhanced_behavioral_context(ax: plt.Axes, trial_info: Dict[str, Any], 
                                   time_grid: np.ndarray, target_isi: float):
    """Enhanced behavioral context with better visibility."""
    n_trials = len(trial_info.get('is_right_choice', []))
    
    # Sort trials by choice for cleaner visualization
    if 'is_right_choice' in trial_info:
        choice_vals = trial_info['is_right_choice']
        sort_order = np.argsort(choice_vals)
    else:
        sort_order = np.arange(n_trials)
    
    # Create behavioral indicator strips
    y_positions = [0, 1, 2, 3]
    labels = ['Choice', 'Stimulus', 'Outcome', 'ISI']
    
    for i, trial_idx in enumerate(sort_order):
        x_pos = i
        
        # Choice (red=right, blue=left, gray=no choice)
        if 'is_right_choice' in trial_info and 'did_not_choose' in trial_info:
            if trial_info['did_not_choose'][trial_idx]:
                color = 'gray'
            elif trial_info['is_right_choice'][trial_idx]:
                color = 'red'
            else:
                color = 'blue'
            ax.barh(y_positions[0], 1, left=x_pos, height=0.8, color=color, alpha=0.8)
        
        # Stimulus type
        if 'is_right' in trial_info:
            color = 'lightcoral' if trial_info['is_right'][trial_idx] else 'lightblue'
            ax.barh(y_positions[1], 1, left=x_pos, height=0.8, color=color, alpha=0.8)
        
        # Outcome
        if 'rewarded' in trial_info:
            color = 'gold' if trial_info['rewarded'][trial_idx] else 'lightgray'
            ax.barh(y_positions[2], 1, left=x_pos, height=0.8, color=color, alpha=0.8)
        
        # ISI value (color-coded)
        if 'isi' in trial_info:
            isi_val = trial_info['isi'][trial_idx]
            # Normalize ISI to color scale
            isi_norm = (isi_val - 200) / (2300 - 200)  # 0-1 scale
            color = plt.cm.viridis(isi_norm)
            ax.barh(y_positions[3], 1, left=x_pos, height=0.8, color=color, alpha=0.8)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, n_trials)
    ax.set_ylim(-0.5, 3.5)
    ax.set_title(f'Trial Context (n={n_trials}, sorted by choice)', fontsize=12)
    ax.set_xlabel('Trial (sorted)')


# def plot_clean_group_raster(ax: plt.Axes, group_data: np.ndarray, time_grid: np.ndarray,
#                            trial_info: Dict[str, Any], group_name: str, roi_indices: np.ndarray,
#                            target_isi: float):
#     """Plot clean, interpretable raster for a small number of ROIs."""
#     n_rois, n_bins, n_trials = group_data.shape
    
#     # Sort trials by choice for better visualization
#     if 'is_right_choice' in trial_info:
#         choice_vals = trial_info['is_right_choice']
#         sort_order = np.argsort(choice_vals)
#     else:
#         sort_order = np.arange(n_trials)
    
#     # Smooth and normalize data
#     smoothed_data = gaussian_filter1d(group_data, sigma=1.5, axis=1)
    
#     # Z-score normalize each ROI
#     for roi_idx in range(n_rois):
#         roi_data = smoothed_data[roi_idx, :, :].flatten()
#         finite_mask = np.isfinite(roi_data)
#         if finite_mask.sum() > 10:
#             mean_val = np.mean(roi_data[finite_mask])
#             std_val = np.std(roi_data[finite_mask])
#             if std_val > 0:
#                 smoothed_data[roi_idx, :, :] = (smoothed_data[roi_idx, :, :] - mean_val) / std_val
    
#     # Create raster plot with sorted trials
#     plot_data = np.zeros((n_rois * n_trials, n_bins))
    
#     for i, trial_idx in enumerate(sort_order):
#         start_row = i * n_rois
#         end_row = start_row + n_rois
#         plot_data[start_row:end_row, :] = smoothed_data[:, :, trial_idx]
    
#     # Plot with better color scale
#     im = ax.imshow(plot_data, aspect='auto', cmap='RdBu_r', 
#                    extent=[time_grid[0], time_grid[-1], 0, n_rois * n_trials],
#                    vmin=-3, vmax=3, interpolation='bilinear')
    
#     # Add clear trial separators
#     for i in range(1, n_trials):
#         y_pos = i * n_rois
#         ax.axhline(y_pos, color='white', linewidth=2, alpha=0.9)
    
#     # Add ROI separators within trials
#     for roi_idx in range(1, n_rois):
#         for i in range(n_trials):
#             y_pos = i * n_rois + roi_idx
#             ax.axhline(y_pos, color='gray', linewidth=0.5, alpha=0.3)
    
#     # Mark stimulus times
#     if target_isi is not None:
#         f1_time = 0.5
#         f2_time = f1_time + target_isi / 1000
#         ax.axvline(f1_time, color='blue', linestyle='-', alpha=0.8, linewidth=2, label='F1')
#         ax.axvline(f2_time, color='blue', linestyle='-', alpha=0.8, linewidth=2, label='F2')
    
#     # Add median lick time
#     if 'lick_start' in trial_info:
#         lick_times = trial_info['lick_start']
#         finite_licks = lick_times[np.isfinite(lick_times)]
#         if len(finite_licks) > 0:
#             median_lick = np.median(finite_licks)
#             ax.axvline(median_lick, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Lick')
    
#     # Formatting
#     clean_name = group_name.replace('_', ' ').title()
#     ax.set_ylabel(f'{clean_name}\n({n_rois} ROIs)', fontsize=10)
#     ax.set_xlim(time_grid[0], time_grid[-1])
#     ax.legend(loc='upper right', fontsize=8)
    
#     if ax.get_subplotspec().rowspan.start == ax.figure._gridspecs[0].nrows - 3:  # Bottom raster
#         ax.set_xlabel('Time (s)')


def plot_clean_group_raster(ax: plt.Axes, group_data: np.ndarray, time_grid: np.ndarray,
                           trial_info: Dict[str, Any], group_name: str, roi_indices: np.ndarray,
                           target_isi: float, is_bottom_plot: bool = False):
    """Plot clean, interpretable raster for a small number of ROIs."""
    n_rois, n_bins, n_trials = group_data.shape
    
    # Sort trials by choice for better visualization
    if 'is_right_choice' in trial_info:
        choice_vals = trial_info['is_right_choice']
        sort_order = np.argsort(choice_vals)
    else:
        sort_order = np.arange(n_trials)
    
    # Smooth and normalize data
    smoothed_data = gaussian_filter1d(group_data, sigma=1.5, axis=1)
    
    # Z-score normalize each ROI
    for roi_idx in range(n_rois):
        roi_data = smoothed_data[roi_idx, :, :].flatten()
        finite_mask = np.isfinite(roi_data)
        if finite_mask.sum() > 10:
            mean_val = np.mean(roi_data[finite_mask])
            std_val = np.std(roi_data[finite_mask])
            if std_val > 0:
                smoothed_data[roi_idx, :, :] = (smoothed_data[roi_idx, :, :] - mean_val) / std_val
    
    # Create raster plot with sorted trials
    plot_data = np.zeros((n_rois * n_trials, n_bins))
    
    for i, trial_idx in enumerate(sort_order):
        start_row = i * n_rois
        end_row = start_row + n_rois
        plot_data[start_row:end_row, :] = smoothed_data[:, :, trial_idx]
    
    # Plot with better color scale
    im = ax.imshow(plot_data, aspect='auto', cmap='RdBu_r', 
                   extent=[time_grid[0], time_grid[-1], 0, n_rois * n_trials],
                   vmin=-3, vmax=3, interpolation='bilinear')
    
    # Add clear trial separators
    for i in range(1, n_trials):
        y_pos = i * n_rois
        ax.axhline(y_pos, color='white', linewidth=2, alpha=0.9)
    
    # Add ROI separators within trials
    for roi_idx in range(1, n_rois):
        for i in range(n_trials):
            y_pos = i * n_rois + roi_idx
            ax.axhline(y_pos, color='gray', linewidth=0.5, alpha=0.3)
    
    # Mark stimulus times
    if target_isi is not None:
        f1_time = 0.5
        f2_time = f1_time + target_isi / 1000
        ax.axvline(f1_time, color='blue', linestyle='-', alpha=0.8, linewidth=2, label='F1')
        ax.axvline(f2_time, color='blue', linestyle='-', alpha=0.8, linewidth=2, label='F2')
    
    # Add median lick time
    if 'lick_start' in trial_info:
        lick_times = trial_info['lick_start']
        finite_licks = lick_times[np.isfinite(lick_times)]
        if len(finite_licks) > 0:
            median_lick = np.median(finite_licks)
            ax.axvline(median_lick, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Lick')
    
    # Formatting
    clean_name = group_name.replace('_', ' ').title()
    ax.set_ylabel(f'{clean_name}\n({n_rois} ROIs)', fontsize=10)
    ax.set_xlim(time_grid[0], time_grid[-1])
    ax.legend(loc='upper right', fontsize=8)
    
    # Only add x-axis label if this is the bottom plot
    if is_bottom_plot:
        ax.set_xlabel('Time (s)')




def plot_group_average_trace(ax: plt.Axes, group_data: np.ndarray, time_grid: np.ndarray,
                           trial_info: Dict[str, Any], group_name: str, target_isi: float) -> np.ndarray:
    """Plot average activity trace for the group."""
    
    # Compute average across ROIs and trials
    avg_trace = np.nanmean(group_data, axis=(0, 2))  # Average over ROIs and trials
    std_trace = np.nanstd(group_data, axis=(0, 2)) / np.sqrt(group_data.shape[0] * group_data.shape[2])
    
    # Smooth the average
    avg_smooth = gaussian_filter1d(avg_trace, sigma=1.0)
    std_smooth = gaussian_filter1d(std_trace, sigma=1.0)
    
    # Plot with error bars
    ax.plot(time_grid, avg_smooth, linewidth=2, label=group_name.split('_')[0].title())
    ax.fill_between(time_grid, avg_smooth - std_smooth, avg_smooth + std_smooth, alpha=0.3)
    
    # Mark events
    if target_isi is not None:
        f1_time = 0.5
        f2_time = f1_time + target_isi / 1000
        ax.axvline(f1_time, color='blue', linestyle='--', alpha=0.5)
        ax.axvline(f2_time, color='blue', linestyle='--', alpha=0.5)
    
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylabel('ΔF/F (z-score)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return avg_smooth


def plot_group_comparison(ax: plt.Axes, group_averages: Dict[str, np.ndarray], 
                         time_grid: np.ndarray, target_isi: float):
    """Compare average traces across functional groups."""
    
    colors = ['red', 'blue', 'purple', 'green', 'orange']
    
    for i, (group_name, avg_trace) in enumerate(group_averages.items()):
        clean_name = group_name.split('_')[0].title()
        ax.plot(time_grid, avg_trace, linewidth=2, color=colors[i % len(colors)], 
               label=clean_name, alpha=0.8)
    
    # Mark events
    if target_isi is not None:
        f1_time = 0.5
        f2_time = f1_time + target_isi / 1000
        ax.axvline(f1_time, color='blue', linestyle='-', alpha=0.7, linewidth=2)
        ax.axvline(f2_time, color='blue', linestyle='-', alpha=0.7, linewidth=2)
        ax.text(f1_time, ax.get_ylim()[1]*0.9, 'F1', ha='center', fontsize=10)
        ax.text(f2_time, ax.get_ylim()[1]*0.9, 'F2', ha='center', fontsize=10)
    
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Average ΔF/F (z-score)')
    ax.set_title('Functional Group Comparison')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


# Replace the main raster function call with:
def run_simplified_raster_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """Run simplified raster analysis with cleaner visualization."""
    
    X = results['X']
    time_grid = results['time_grid']
    trial_info = results['trial_info']
    roi_sets = results['roi_sets']
    audit_result = results['audit_result']
    cp_result = results['cp_result']
    
    # Test with one ISI value first
    target_isis = [200, 700, 1700]  # Just 3 ISIs for comparison
    
    simplified_figures = {}
    
    for isi in target_isis:
        print(f"Creating simplified raster for ISI {isi}ms...")
        fig = plot_simplified_functional_raster(
            X, time_grid, trial_info, roi_sets, audit_result, cp_result,
            target_isi=isi, n_rois_per_group=8  # Only 8 ROIs per group
        )
        if fig is not None:
            simplified_figures[f'isi_{isi}_simplified'] = fig
    
    results['simplified_rasters'] = simplified_figures
    return results


















# Add these new functions to replace the current CP decomposition approach:







def prepare_trial_tensor_high_res(trial_data: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convert trial data to high temporal resolution tensor format.
    Uses native imaging rate (~29.76 Hz, ~33ms bins) instead of downsampled.
    """
    print("=== PREPARING HIGH-RESOLUTION TRIAL TENSOR ===")
    
    df = trial_data['df_trials_with_segments']
    tensor_cfg = cfg.get('tensor_prep', {})
    
    print(f"Input data: {len(df)} trials")
    
    # Use NATIVE temporal resolution from imaging
    if 'imaging_fs' in trial_data:
        native_fs = trial_data['imaging_fs']
        dt = 1.0 / native_fs
        print(f"Using native imaging rate: {native_fs:.2f} Hz (dt = {dt*1000:.1f}ms)")
    else:
        # Fallback: estimate from data
        first_row = df.iloc[0]
        first_time = np.asarray(first_row['dff_time_vector'])
        dt = np.mean(np.diff(first_time))
        print(f"Estimated dt from data: {dt*1000:.1f}ms")
    
    # Get time range as before
    all_time_vecs = []
    time_ranges = []
    
    for trial_idx, (_, row) in enumerate(df.iterrows()):
        t_vec = np.asarray(row['dff_time_vector'])
        if len(t_vec) == 0:
            continue
        all_time_vecs.append(t_vec)
        time_ranges.append((t_vec.min(), t_vec.max()))
    
    t_min = min(t_range[0] for t_range in time_ranges)
    t_max = max(t_range[1] for t_range in time_ranges)
    
    # Create high-resolution time grid
    time_grid = np.arange(t_min, t_max + dt/2, dt)
    n_bins = len(time_grid)
    
    print(f"High-res time grid: {n_bins} bins at {1/dt:.1f} Hz")
    print(f"Time range: {t_min:.3f} to {t_max:.3f}s ({t_max - t_min:.3f}s duration)")
    
    # Continue with tensor filling as before...
    n_rois = None
    for _, row in df.iterrows():
        dff = np.asarray(row['dff_segment'])
        if dff.size > 0:
            n_rois = dff.shape[0]
            break
    
    n_trials = len(df)
    X = np.full((n_rois, n_bins, n_trials), np.nan, dtype=np.float32)
    
    # Fill tensor
    valid_trials = 0
    for trial_idx, (_, row) in enumerate(df.iterrows()):
        t_vec = np.asarray(row['dff_time_vector'])
        dff = np.asarray(row['dff_segment'])
        
        if dff.size == 0 or t_vec.size == 0 or dff.shape[0] != n_rois:
            continue
        
        # Ensure monotonic time
        if not np.all(np.diff(t_vec) > 0):
            sort_idx = np.argsort(t_vec)
            t_vec = t_vec[sort_idx]
            dff = dff[:, sort_idx]
        
        # Interpolate to high-res grid
        for roi_idx in range(n_rois):
            roi_data = dff[roi_idx, :]
            valid_mask = np.isfinite(roi_data) & np.isfinite(t_vec)
            
            if valid_mask.sum() < 3:
                continue
            
            t_valid = t_vec[valid_mask]
            roi_valid = roi_data[valid_mask]
            
            # Find grid points within data range
            grid_mask = (time_grid >= t_valid.min()) & (time_grid <= t_valid.max())
            
            if grid_mask.sum() > 0:
                X[roi_idx, grid_mask, trial_idx] = np.interp(
                    time_grid[grid_mask], t_valid, roi_valid
                )
        
        valid_trials += 1
    
    # Extract trial metadata with event times
    trial_info = {}
    event_columns = ['start_flash_1', 'end_flash_1', 'start_flash_2', 'end_flash_2', 'lick_start']
    
    for col in event_columns + ['isi', 'is_right_choice', 'is_right', 'rewarded', 'punished', 'did_not_choose']:
        if col in df.columns:
            trial_info[col] = df[col].values
            print(f"  {col}: extracted")
        else:
            print(f"  {col}: not found")
    
    trial_info['trial_indices'] = df.index.values
    
    print(f"High-resolution tensor: {X.shape}")
    print("=== HIGH-RES TENSOR PREPARATION COMPLETE ===\n")
    
    return X, time_grid, trial_info


def analyze_event_specific_roi_groups(X: np.ndarray, time_grid: np.ndarray, 
                                     trial_info: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Discover ROI groups locked to specific behavioral events with proper temporal resolution.
    """
    print("=== ANALYZING EVENT-SPECIFIC ROI GROUPS ===")
    
    roi_groups = {}
    n_rois, n_bins, n_trials = X.shape
    
    print(f"Input: {n_rois} ROIs, {n_bins} time bins, {n_trials} trials")
    
    # Process each primary functional group
    for group_name, group_config in PRIMARY_FUNCTIONAL_GROUPS.items():
        print(f"\nAnalyzing {group_name}...")
        
        if group_config.get('special_analysis') == 'isi_sensitivity_f1_to_f2_only':
            # Special ISI sensitivity analysis
            roi_groups[group_name] = analyze_isi_duration_sensitivity_proper(
                X, time_grid, trial_info, group_config, cfg
            )
        else:
            # Standard event-locked analysis
            roi_groups[group_name] = analyze_event_locked_responses(
                X, time_grid, trial_info, group_config, cfg
            )
    
    # Compute intersections between groups
    print("\nComputing ROI group intersections...")
    roi_groups.update(compute_roi_group_intersections(roi_groups))
    
    print(f"\nTotal ROI groups discovered: {len(roi_groups)}")
    for name, group_data in roi_groups.items():
        if 'roi_indices' in group_data:
            print(f"  {name}: {len(group_data['roi_indices'])} ROIs")
    
    print("=== EVENT-SPECIFIC ROI GROUP ANALYSIS COMPLETE ===\n")
    
    return roi_groups


def analyze_isi_duration_sensitivity_proper(X: np.ndarray, time_grid: np.ndarray, 
                                           trial_info: Dict[str, Any], group_config: Dict[str, Any], 
                                           cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Proper ISI-duration sensitivity analysis: ONLY look between F1 offset and F2 onset.
    Key insight: Use long ISI trials to identify ROIs with continued activity modulation during the wait period.
    """
    print("  ISI sensitivity analysis (F1 offset → F2 onset only)")
    
    # Check required event times
    if 'end_flash_1' not in trial_info or 'start_flash_2' not in trial_info:
        print("    ERROR: Missing F1/F2 timing information")
        return {'roi_indices': [], 'error': 'missing_event_times'}
    
    f1_end_times = trial_info['end_flash_1']
    f2_start_times = trial_info['start_flash_2']
    
    # Filter to trials with valid timing
    valid_mask = (np.isfinite(f1_end_times) & np.isfinite(f2_start_times) & 
                  np.isfinite(trial_info.get('isi', np.full(len(f1_end_times), np.nan))))
    
    if valid_mask.sum() < 20:
        print(f"    ERROR: Too few valid trials ({valid_mask.sum()})")
        return {'roi_indices': [], 'error': 'insufficient_trials'}
    
    # Get long ISI trials (top 25% of ISI values)
    isis = trial_info['isi'][valid_mask]
    long_isi_threshold = np.percentile(isis, 75)
    
    valid_trial_indices = np.where(valid_mask)[0]
    long_isi_mask = isis >= long_isi_threshold
    long_isi_trial_indices = valid_trial_indices[long_isi_mask]
    
    print(f"    Using {len(long_isi_trial_indices)} long ISI trials (ISI ≥ {long_isi_threshold:.0f}ms)")
    
    isi_sensitive_rois = []
    roi_scores = []
    
    # Analyze each ROI
    for roi_idx in range(X.shape[0]):
        roi_score = compute_isi_sensitivity_score(
            roi_idx, X, time_grid, trial_info, long_isi_trial_indices, cfg
        )
        
        roi_scores.append(roi_score)
        
        sensitivity_threshold = cfg.get('isi_sensitivity_threshold', 2.0)
        if roi_score > sensitivity_threshold:
            isi_sensitive_rois.append(roi_idx)
    
    print(f"    Found {len(isi_sensitive_rois)} ISI-sensitive ROIs (threshold: {sensitivity_threshold})")
    
    return {
        'roi_indices': np.array(isi_sensitive_rois),
        'roi_scores': np.array(roi_scores),
        'analysis_method': 'f1_to_f2_activity_modulation',
        'threshold_used': sensitivity_threshold,
        'long_isi_threshold': long_isi_threshold,
        'trials_analyzed': len(long_isi_trial_indices),
        'description': group_config['description']
    }


# def compute_isi_sensitivity_score(roi_idx: int, X: np.ndarray, time_grid: np.ndarray, 
#                                 trial_info: Dict[str, Any], long_isi_trial_indices: np.ndarray, 
#                                 cfg: Dict[str, Any]) -> float:
#     """
#     Compute ISI sensitivity score for a single ROI.
    
#     Strategy:
#     1. Align to F1 offset for each long ISI trial
#     2. Extract baseline period (before F1 offset)  
#     3. Extract ISI period (F1 offset to F2 onset - buffer)
#     4. Test for sustained modulation during ISI
#     """
    
#     f1_end_times = trial_info['end_flash_1']
#     f2_start_times = trial_info['start_flash_2']
    
#     baseline_scores = []
#     isi_scores = []
    
#     for trial_idx in long_isi_trial_indices:
#         f1_end = f1_end_times[trial_idx]
#         f2_start = f2_start_times[trial_idx]
        
#         if not (np.isfinite(f1_end) and np.isfinite(f2_start)):
#             continue
        
#         # Define time windows
#         baseline_start = f1_end - 0.3  # 300ms before F1 end
#         baseline_end = f1_end - 0.05   # 50ms before F1 end (avoid offset response)
        
#         isi_start = f1_end + 0.1       # 100ms after F1 end (avoid offset response)
#         isi_end = f2_start - 0.1       # 100ms before F2 start (avoid onset anticipation)
        
#         # Skip if ISI window is too short
#         if isi_end <= isi_start:
#             continue
        
#         # Extract baseline activity
#         baseline_mask = (time_grid >= baseline_start) & (time_grid <= baseline_end)
#         if baseline_mask.sum() > 0:
#             baseline_activity = X[roi_idx, baseline_mask, trial_idx]
#             baseline_mean = np.nanmean(baseline_activity)
#             if np.isfinite(baseline_mean):
#                 baseline_scores.append(baseline_mean)
        
#         # Extract ISI activity  
#         isi_mask = (time_grid >= isi_start) & (time_grid <= isi_end)
#         if isi_mask.sum() > 0:
#             isi_activity = X[roi_idx, isi_mask, trial_idx]
#             isi_mean = np.nanmean(isi_activity)
#             if np.isfinite(isi_mean):
#                 isi_scores.append(isi_mean)
    
#     # Compute sensitivity score
#     if len(baseline_scores) < 3 or len(isi_scores) < 3:
#         return 0.0
    
#     baseline_scores = np.array(baseline_scores)
#     isi_scores = np.array(isi_scores)
    
#     # Test for significant difference (sustained modulation)
#     baseline_mean = np.mean(baseline_scores)
#     isi_mean = np.mean(isi_scores)
#     pooled_std = np.std(np.concatenate([baseline_scores, isi_scores]))
    
#     if pooled_std == 0:
#         return 0.0
    
#     # Z-score of difference (positive = sustained suppression, negative = sustained activation)
#     sensitivity_score = abs(baseline_mean - isi_mean) / pooled_std
    
#     return sensitivity_score


def analyze_event_locked_responses(X: np.ndarray, time_grid: np.ndarray, 
                                 trial_info: Dict[str, Any], group_config: Dict[str, Any], 
                                 cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze ROI responses locked to specific behavioral events.
    """
    event_name = group_config['event']
    window = group_config['window']
    
    print(f"  Event-locked analysis: {event_name}")
    
    if event_name not in trial_info:
        print(f"    ERROR: Event '{event_name}' not found in trial info")
        return {'roi_indices': [], 'error': f'missing_event_{event_name}'}
    
    event_times = trial_info[event_name]
    valid_mask = np.isfinite(event_times)
    
    if valid_mask.sum() < 10:
        print(f"    ERROR: Too few valid trials ({valid_mask.sum()})")
        return {'roi_indices': [], 'error': 'insufficient_trials'}
    
    valid_trial_indices = np.where(valid_mask)[0]
    print(f"    Using {len(valid_trial_indices)} trials with valid {event_name} times")
    
    event_locked_rois = []
    roi_scores = []
    
    # Analyze each ROI
    for roi_idx in range(X.shape[0]):
        roi_score = compute_event_locking_score(
            roi_idx, X, time_grid, event_times, valid_trial_indices, window, cfg
        )
        
        roi_scores.append(roi_score)
        
        event_threshold = cfg.get('event_locking_threshold', 2.0)
        if roi_score > event_threshold:
            event_locked_rois.append(roi_idx)
    
    print(f"    Found {len(event_locked_rois)} {event_name}-locked ROIs")
    
    return {
        'roi_indices': np.array(event_locked_rois),
        'roi_scores': np.array(roi_scores),
        'analysis_method': f'{event_name}_event_locked',
        'event_name': event_name,
        'window': window,
        'threshold_used': cfg.get('event_locking_threshold', 2.0),
        'trials_analyzed': len(valid_trial_indices),
        'description': group_config['description']
    }


# def compute_event_locking_score(roi_idx: int, X: np.ndarray, time_grid: np.ndarray,
#                               event_times: np.ndarray, valid_trial_indices: np.ndarray,
#                               window: Tuple[float, float], cfg: Dict[str, Any]) -> float:
#     """
#     Compute event-locking score for a single ROI.
    
#     Strategy:
#     1. Align to event time for each trial
#     2. Extract pre-event baseline and post-event response windows
#     3. Test for significant modulation in response window vs baseline
#     """
    
#     pre_window, post_window = window
    
#     baseline_scores = []
#     response_scores = []
    
#     for trial_idx in valid_trial_indices:
#         event_time = event_times[trial_idx]
        
#         if not np.isfinite(event_time):
#             continue
        
#         # Define time windows relative to event
#         baseline_start = event_time + pre_window
#         baseline_end = event_time - 0.05  # 50ms before event
        
#         response_start = event_time + 0.05  # 50ms after event
#         response_end = event_time + post_window
        
#         # Extract baseline activity
#         baseline_mask = (time_grid >= baseline_start) & (time_grid <= baseline_end)
#         if baseline_mask.sum() > 0:
#             baseline_activity = X[roi_idx, baseline_mask, trial_idx]
#             baseline_mean = np.nanmean(baseline_activity)
#             if np.isfinite(baseline_mean):
#                 baseline_scores.append(baseline_mean)
        
#         # Extract response activity
#         response_mask = (time_grid >= response_start) & (time_grid <= response_end)
#         if response_mask.sum() > 0:
#             response_activity = X[roi_idx, response_mask, trial_idx]
#             response_mean = np.nanmean(response_activity)
#             if np.isfinite(response_mean):
#                 response_scores.append(response_mean)
    
#     # Compute locking score
#     if len(baseline_scores) < 3 or len(response_scores) < 3:
#         return 0.0
    
#     baseline_scores = np.array(baseline_scores)
#     response_scores = np.array(response_scores)
    
#     # Test for significant difference
#     baseline_mean = np.mean(baseline_scores)
#     response_mean = np.mean(response_scores)
#     pooled_std = np.std(np.concatenate([baseline_scores, response_scores]))
    
#     if pooled_std == 0:
#         return 0.0
    
#     # Z-score of difference (absolute value = strength of modulation)
#     locking_score = abs(response_mean - baseline_mean) / pooled_std
    
#     return locking_score


def compute_roi_group_intersections(roi_groups: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute meaningful intersections between ROI groups.
    """
    intersections = {}
    
    # Get main groups (exclude existing intersections)
    main_groups = {name: data for name, data in roi_groups.items() 
                  if '∩' not in name and 'roi_indices' in data and len(data['roi_indices']) > 0}
    
    group_names = list(main_groups.keys())
    
    for i, name1 in enumerate(group_names):
        for name2 in group_names[i+1:]:
            if name1 != name2:
                roi_set1 = set(main_groups[name1]['roi_indices'])
                roi_set2 = set(main_groups[name2]['roi_indices'])
                
                intersection = roi_set1.intersection(roi_set2)
                
                if len(intersection) >= 3:  # Only keep substantial intersections
                    intersections[f'{name1}∩{name2}'] = {
                        'roi_indices': np.array(list(intersection)),
                        'analysis_method': 'intersection',
                        'parent_groups': [name1, name2],
                        'description': f'ROIs with both {name1} and {name2} properties'
                    }
    
    return intersections


# Update the main configuration
def run_event_specific_roi_analysis(trial_data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run complete event-specific ROI analysis with proper temporal resolution.
    """
    print("=" * 80)
    print("RUNNING EVENT-SPECIFIC ROI ANALYSIS")
    print("=" * 80)
    
    # Step 1: Prepare high-resolution tensor
    X, time_grid, trial_info = prepare_trial_tensor_high_res(trial_data, cfg)
    
    # Step 2: Analyze event-specific ROI groups
    roi_groups = analyze_event_specific_roi_groups(X, time_grid, trial_info, cfg)
    
    # Step 3: Convert to ROI sets for compatibility with existing plotting
    roi_sets = {}
    for group_name, group_data in roi_groups.items():
        if 'roi_indices' in group_data and len(group_data['roi_indices']) > 0:
            roi_sets[group_name] = group_data['roi_indices']
    
    # Package results
    results = {
        'X': X,
        'time_grid': time_grid,
        'trial_info': trial_info,
        'roi_groups': roi_groups,
        'roi_sets': roi_sets,
        'config': cfg,
        'analysis_method': 'event_specific_with_proper_isi'
    }
    
    print(f"\nEvent-specific analysis complete!")
    print(f"Discovered {len(roi_sets)} functional ROI groups:")
    for name, indices in roi_sets.items():
        print(f"  {name}: {len(indices)} ROIs")
    
    print("=" * 80)
    
    return results






















# ...existing code...

def compute_event_locking_score(roi_idx: int, X: np.ndarray, time_grid: np.ndarray,
                              event_times: np.ndarray, valid_trial_indices: np.ndarray,
                              window: Tuple[float, float], cfg: Dict[str, Any]) -> float:
    """
    Compute event-locking score for a single ROI using TRIAL-AVERAGED responses.
    
    Strategy:
    1. Align all trials to event time
    2. Average across trials to get mean response
    3. Compare baseline vs response periods in the averaged trace
    4. This captures consistent but weak responses that are lost in single-trial noise
    """
    
    pre_window, post_window = window
    
    # Collect all trial segments aligned to event
    aligned_segments = []
    
    for trial_idx in valid_trial_indices:
        event_time = event_times[trial_idx]
        
        if not np.isfinite(event_time):
            continue
        
        # Define time windows relative to event
        baseline_start = event_time + pre_window
        response_end = event_time + post_window
        
        # Find time indices for this trial's segment
        segment_mask = (time_grid >= baseline_start) & (time_grid <= response_end)
        
        if segment_mask.sum() < 5:  # Need minimum segment length
            continue
        
        # Extract ROI activity for this segment
        roi_segment = X[roi_idx, segment_mask, trial_idx]
        
        if np.isfinite(roi_segment).sum() < 3:  # Need some valid data
            continue
            
        aligned_segments.append(roi_segment)
    
    if len(aligned_segments) < 10:  # Need minimum trials for averaging
        return 0.0
    
    # Pad segments to same length for averaging
    max_length = max(len(seg) for seg in aligned_segments)
    padded_segments = []
    
    for seg in aligned_segments:
        if len(seg) < max_length:
            # Pad with NaN
            padded_seg = np.full(max_length, np.nan)
            padded_seg[:len(seg)] = seg
        else:
            padded_seg = seg[:max_length]
        padded_segments.append(padded_seg)
    
    # Average across trials (ignoring NaN)
    trial_averaged_response = np.nanmean(padded_segments, axis=0)
    
    if np.isfinite(trial_averaged_response).sum() < 5:
        return 0.0
    
    # Define baseline and response periods in averaged trace
    # Assume event happens at 1/3 into the segment (based on pre_window proportion)
    total_length = len(trial_averaged_response)
    baseline_proportion = abs(pre_window) / (abs(pre_window) + post_window)
    event_idx = int(baseline_proportion * total_length)
    
    # Baseline: before event (avoid very edge)
    baseline_start_idx = max(0, int(0.1 * event_idx))
    baseline_end_idx = max(1, event_idx - int(0.1 * event_idx))
    
    # Response: after event (avoid immediate edge)
    response_start_idx = min(total_length-1, event_idx + int(0.1 * (total_length - event_idx)))
    response_end_idx = total_length
    
    if baseline_end_idx <= baseline_start_idx or response_end_idx <= response_start_idx:
        return 0.0
    
    # Extract baseline and response periods
    baseline_activity = trial_averaged_response[baseline_start_idx:baseline_end_idx]
    response_activity = trial_averaged_response[response_start_idx:response_end_idx]
    
    baseline_activity = baseline_activity[np.isfinite(baseline_activity)]
    response_activity = response_activity[np.isfinite(response_activity)]
    
    if len(baseline_activity) < 3 or len(response_activity) < 3:
        return 0.0
    
    # Compute modulation strength using trial-averaged data
    baseline_mean = np.mean(baseline_activity)
    response_mean = np.mean(response_activity)
    
    # Use pooled standard deviation across the entire averaged trace for normalization
    all_activity = trial_averaged_response[np.isfinite(trial_averaged_response)]
    pooled_std = np.std(all_activity)
    
    if pooled_std == 0:
        return 0.0
    
    # Z-score of difference (absolute value = strength of modulation)
    locking_score = abs(response_mean - baseline_mean) / pooled_std
    
    return locking_score


def compute_isi_sensitivity_score(roi_idx: int, X: np.ndarray, time_grid: np.ndarray, 
                                trial_info: Dict[str, Any], long_isi_trial_indices: np.ndarray, 
                                cfg: Dict[str, Any]) -> float:
    """
    Compute ISI sensitivity score using TRIAL-AVERAGED approach.
    
    Strategy:
    1. Align all long ISI trials to F1 offset
    2. Average across trials to get mean ISI period activity
    3. Compare baseline vs ISI periods in averaged trace
    """
    
    f1_end_times = trial_info['end_flash_1']
    f2_start_times = trial_info['start_flash_2']
    
    # Collect ISI segments aligned to F1 offset
    aligned_segments = []
    
    for trial_idx in long_isi_trial_indices:
        f1_end = f1_end_times[trial_idx]
        f2_start = f2_start_times[trial_idx]
        
        if not (np.isfinite(f1_end) and np.isfinite(f2_start)):
            continue
        
        # Define analysis window: baseline before F1 end + ISI period
        analysis_start = f1_end - 0.5  # 500ms before F1 end
        analysis_end = f2_start - 0.1   # 100ms before F2 start
        
        if analysis_end <= analysis_start:
            continue
        
        # Extract segment
        segment_mask = (time_grid >= analysis_start) & (time_grid <= analysis_end)
        
        if segment_mask.sum() < 10:  # Need reasonable segment length
            continue
        
        roi_segment = X[roi_idx, segment_mask, trial_idx]
        
        if np.isfinite(roi_segment).sum() < 5:
            continue
            
        aligned_segments.append(roi_segment)
    
    if len(aligned_segments) < 5:  # Need minimum trials
        return 0.0
    
    # Pad and average segments
    max_length = max(len(seg) for seg in aligned_segments)
    padded_segments = []
    
    for seg in aligned_segments:
        if len(seg) < max_length:
            padded_seg = np.full(max_length, np.nan)
            padded_seg[:len(seg)] = seg
        else:
            padded_seg = seg[:max_length]
        padded_segments.append(padded_seg)
    
    # Average across trials
    trial_averaged_isi = np.nanmean(padded_segments, axis=0)
    
    if np.isfinite(trial_averaged_isi).sum() < 10:
        return 0.0
    
    # Define baseline and ISI periods
    # F1 offset happens at ~50ms/(50ms + ISI_duration) into segment
    baseline_length = int(0.5 * 30)  # ~500ms at 30Hz = 15 bins
    
    if len(trial_averaged_isi) < baseline_length + 10:
        return 0.0
    
    # Baseline: first part of segment (before F1 offset)
    baseline_activity = trial_averaged_isi[:baseline_length]
    
    # ISI period: after F1 offset, avoiding edges
    isi_start_idx = baseline_length + 3  # Skip 3 bins after F1 offset
    isi_end_idx = len(trial_averaged_isi) - 3  # Skip 3 bins before F2
    
    if isi_end_idx <= isi_start_idx:
        return 0.0
    
    isi_activity = trial_averaged_isi[isi_start_idx:isi_end_idx]
    
    # Remove NaN values
    baseline_activity = baseline_activity[np.isfinite(baseline_activity)]
    isi_activity = isi_activity[np.isfinite(isi_activity)]
    
    if len(baseline_activity) < 3 or len(isi_activity) < 3:
        return 0.0
    
    # Compute sensitivity score
    baseline_mean = np.mean(baseline_activity)
    isi_mean = np.mean(isi_activity)
    
    # Use pooled standard deviation
    all_activity = trial_averaged_isi[np.isfinite(trial_averaged_isi)]
    pooled_std = np.std(all_activity)
    
    if pooled_std == 0:
        return 0.0
    
    # Z-score of difference (absolute = strength of sustained modulation)
    sensitivity_score = abs(baseline_mean - isi_mean) / pooled_std
    
    return sensitivity_score




# Also add a debug version that shows the process:
def analyze_event_locked_responses_debug(X: np.ndarray, time_grid: np.ndarray, 
                                        trial_info: Dict[str, Any], group_config: Dict[str, Any], 
                                        cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debug version that shows why ROIs are/aren't being detected.
    """
    event_name = group_config['event']
    window = group_config['window']
    
    print(f"  Event-locked analysis: {event_name} (DEBUG MODE)")
    
    if event_name not in trial_info:
        print(f"    ERROR: Event '{event_name}' not found in trial info")
        return {'roi_indices': [], 'error': f'missing_event_{event_name}'}
    
    event_times = trial_info[event_name]
    valid_mask = np.isfinite(event_times)
    
    if valid_mask.sum() < 10:
        print(f"    ERROR: Too few valid trials ({valid_mask.sum()})")
        return {'roi_indices': [], 'error': 'insufficient_trials'}
    
    valid_trial_indices = np.where(valid_mask)[0]
    print(f"    Using {len(valid_trial_indices)} trials with valid {event_name} times")
    
    event_locked_rois = []
    roi_scores = []
    
    # Test first 10 ROIs in detail
    n_test_rois = min(10, X.shape[0])
    print(f"    Testing first {n_test_rois} ROIs in detail:")
    
    for roi_idx in range(n_test_rois):
        roi_score = compute_event_locking_score(
            roi_idx, X, time_grid, event_times, valid_trial_indices, window, cfg
        )
        
        print(f"      ROI {roi_idx}: score = {roi_score:.3f}")
        
        event_threshold = cfg.get('event_locking_threshold', 0.5)
        if roi_score > event_threshold:
            print(f"        → DETECTED (above threshold {event_threshold})")
        else:
            print(f"        → Not detected (below threshold {event_threshold})")
    
    # Process all ROIs
    for roi_idx in range(X.shape[0]):
        roi_score = compute_event_locking_score(
            roi_idx, X, time_grid, event_times, valid_trial_indices, window, cfg
        )
        
        roi_scores.append(roi_score)
        
        event_threshold = cfg.get('event_locking_threshold', 0.5)
        if roi_score > event_threshold:
            event_locked_rois.append(roi_idx)
    
    print(f"    Found {len(event_locked_rois)} {event_name}-locked ROIs")
    print(f"    Score distribution: min={min(roi_scores):.3f}, max={max(roi_scores):.3f}, "
          f"mean={np.mean(roi_scores):.3f}, std={np.std(roi_scores):.3f}")
    
    return {
        'roi_indices': np.array(event_locked_rois),
        'roi_scores': np.array(roi_scores),
        'analysis_method': f'{event_name}_event_locked_trial_averaged',
        'event_name': event_name,
        'window': window,
        'threshold_used': cfg.get('event_locking_threshold', 0.5),
        'trials_analyzed': len(valid_trial_indices),
        'description': group_config['description']
    }

# Update the main analysis to use debug mode initially:
def analyze_event_specific_roi_groups_debug(X: np.ndarray, time_grid: np.ndarray, 
                                           trial_info: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debug version of event-specific analysis.
    """
    print("=== ANALYZING EVENT-SPECIFIC ROI GROUPS (DEBUG MODE) ===")
    
    roi_groups = {}
    n_rois, n_bins, n_trials = X.shape
    
    print(f"Input: {n_rois} ROIs, {n_bins} time bins, {n_trials} trials")
    
    # Test just the first few groups
    test_groups = ['f1_onset', 'lick_start']  # Start with these
    
    for group_name in test_groups:
        if group_name in PRIMARY_FUNCTIONAL_GROUPS:
            group_config = PRIMARY_FUNCTIONAL_GROUPS[group_name]
            print(f"\nAnalyzing {group_name}...")
            
            if group_config.get('special_analysis') == 'isi_sensitivity_f1_to_f2_only':
                # Special ISI sensitivity analysis
                roi_groups[group_name] = analyze_isi_duration_sensitivity_proper(
                    X, time_grid, trial_info, group_config, cfg
                )
            else:
                # Debug event-locked analysis
                roi_groups[group_name] = analyze_event_locked_responses_debug(
                    X, time_grid, trial_info, group_config, cfg
                )
    
    return roi_groups


















# Update configuration with the working thresholds
cfg_event_specific = {
    'tensor_prep': {
        'use_native_resolution': True,
    },
    'event_analysis': {
        'event_locking_threshold': 0.5,     # Working threshold from debug
        'isi_sensitivity_threshold': 0.3,   # Lower for ISI sensitivity
    },
    'roi_sets': {
        'min_group_size': 5,  # Require at least 5 ROIs per group
    }
}

# Add the missing PRIMARY_FUNCTIONAL_GROUPS at the module level
PRIMARY_FUNCTIONAL_GROUPS = {
    'f1_onset': {
        'event': 'start_flash_1', 
        'window': (-0.1, 0.2),
        'description': 'F1-onset locked ROIs'
    },
    'f1_offset_isi_start': {
        'event': 'end_flash_1',
        'window': (-0.05, 0.2),
        'description': 'F1-offset / ISI-start locked ROIs'
    },
    'f2_onset': {
        'event': 'start_flash_2',
        'window': (-0.1, 0.2),
        'description': 'F2-onset locked ROIs'  
    },
    'f2_offset_choice_start': {
        'event': 'end_flash_2',
        'window': (-0.1, 0.3),
        'description': 'F2-offset / choice-start locked ROIs'
    },
    'choice_execution_lick': {
        'event': 'lick_start',
        'window': (-0.2, 0.3),
        'description': 'Choice execution / lick-start locked ROIs'
    },
    'isi_duration_sensitive': {
        'event': 'end_flash_1',
        'window': (0.2, 'f2_onset'),  # CRITICAL: Stop at F2 onset!
        'description': 'ISI-duration sensitive ROIs',
        'special_analysis': 'isi_sensitivity_f1_to_f2_only'
    }
}

def run_complete_event_specific_analysis(trial_data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run complete event-specific ROI analysis with working detection methods.
    """
    print("=" * 80)
    print("RUNNING COMPLETE EVENT-SPECIFIC ROI ANALYSIS")
    print("=" * 80)
    
    # Step 1: Prepare high-resolution tensor
    X, time_grid, trial_info = prepare_trial_tensor_high_res(trial_data, cfg)
    
    # Step 2: Analyze all event-specific ROI groups (not just debug subset)
    roi_groups = analyze_event_specific_roi_groups(X, time_grid, trial_info, cfg)
    
    # Step 3: Convert to ROI sets for compatibility with existing plotting
    roi_sets = {}
    for group_name, group_data in roi_groups.items():
        if 'roi_indices' in group_data and len(group_data['roi_indices']) >= cfg.get('roi_sets', {}).get('min_group_size', 5):
            roi_sets[group_name] = group_data['roi_indices']
    
    # Step 4: Create mock audit result for plotting compatibility
    audit_result = create_mock_audit_from_event_groups(roi_groups)
    
    # Step 5: Create mock CP result for plotting compatibility  
    cp_result = create_mock_cp_from_event_groups(X, time_grid, roi_groups)
    
    # Package results
    results = {
        'X': X,
        'time_grid': time_grid,
        'trial_info': trial_info,
        'roi_groups': roi_groups,
        'roi_sets': roi_sets,
        'audit_result': audit_result,  # For plotting compatibility
        'cp_result': cp_result,        # For plotting compatibility
        'config': cfg,
        'analysis_method': 'event_specific_trial_averaged'
    }
    
    print(f"\nEvent-specific analysis complete!")
    print(f"Discovered {len(roi_sets)} functional ROI groups:")
    for name, indices in roi_sets.items():
        print(f"  {name}: {len(indices)} ROIs")
    
    print("=" * 80)
    
    return results

def create_mock_audit_from_event_groups(roi_groups: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create mock audit result for plotting compatibility.
    """
    motif_labels = []
    motif_scores = {}
    
    for i, (group_name, group_data) in enumerate(roi_groups.items()):
        if 'roi_indices' in group_data and len(group_data['roi_indices']) > 0:
            # Extract functional label from group name
            if 'f1' in group_name:
                label = 'stimulus'
            elif 'f2' in group_name:
                label = 'stimulus' 
            elif 'lick' in group_name or 'choice' in group_name:
                label = 'choice'
            elif 'isi' in group_name:
                label = 'expectation'
            else:
                label = 'unknown'
            
            motif_labels.append(label)
            
            # Create mock scores based on detection results
            scores = {
                'event_enrichment': group_data.get('threshold_used', 1.0),
                'choice_auc': 0.6 if 'choice' in label else 0.5,
                'stimulus_auc': 0.6 if 'stimulus' in label else 0.5,
                'isi_coupling': 0.2 if 'expectation' in label else 0.05,
                'outcome_auc': 0.5
            }
            motif_scores[f'motif_{i}'] = scores
    
    return {
        'motif_labels': motif_labels,
        'motif_scores': motif_scores,
        'label_counts': {label: motif_labels.count(label) for label in set(motif_labels)}
    }

def create_mock_cp_from_event_groups(X: np.ndarray, time_grid: np.ndarray, 
                                    roi_groups: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create mock CP result for plotting compatibility.
    """
    n_rois, n_bins, n_trials = X.shape
    valid_groups = [group for group in roi_groups.values() 
                   if 'roi_indices' in group and len(group['roi_indices']) > 0]
    n_motifs = len(valid_groups)
    
    # Create mock factor matrices
    roi_loadings = np.zeros((n_rois, n_motifs))
    motif_timecourses = np.random.randn(n_bins, n_motifs) * 0.1  # Small random variations
    trial_weights = np.random.randn(n_trials, n_motifs) * 0.1
    
    # Set loadings for detected ROIs
    for motif_idx, group_data in enumerate(valid_groups):
        roi_indices = group_data['roi_indices']
        roi_loadings[roi_indices, motif_idx] = 1.0  # High loading for detected ROIs
        
        # Create simple motif timecourse (Gaussian around peak)
        peak_time = 1.0 + motif_idx * 0.5  # Spread peaks across time
        peak_idx = np.argmin(np.abs(time_grid - peak_time))
        gaussian = np.exp(-0.5 * ((np.arange(n_bins) - peak_idx) / 5)**2)
        motif_timecourses[:, motif_idx] = gaussian
        
        # Set trial weights to 1 for simplicity
        trial_weights[:, motif_idx] = 1.0
    
    return {
        'roi_loadings': roi_loadings,
        'motif_timecourses': motif_timecourses, 
        'trial_weights': trial_weights,
        'time_grid': time_grid,
        'reconstruction_error': 0.1,
        'converged': True
    }

def plot_event_specific_raster_analysis(results: Dict[str, Any], target_isis: List[float] = None) -> Dict[str, Any]:
    """
    Create raster plots using event-specific ROI groups.
    """
    print("=== PLOTTING EVENT-SPECIFIC RASTER ANALYSIS ===")
    
    X = results['X']
    time_grid = results['time_grid']
    trial_info = results['trial_info']
    roi_sets = results['roi_sets']
    audit_result = results['audit_result']
    cp_result = results['cp_result']
    
    if target_isis is None:
        unique_isis = np.unique(trial_info['isi'])
        target_isis = [200, 700, 1700]  # Representative ISIs
        target_isis = [isi for isi in target_isis if isi in unique_isis]
    
    print(f"Creating rasters for ISIs: {target_isis}")
    
    raster_figures = {}
    
    for isi in target_isis:
        print(f"Creating raster for ISI {isi}ms...")
        fig = plot_simplified_functional_raster(
            X, time_grid, trial_info, roi_sets, audit_result, cp_result,
            target_isi=isi, n_rois_per_group=8
        )
        if fig is not None:
            raster_figures[f'event_specific_isi_{isi}'] = fig
    
    results['event_specific_rasters'] = raster_figures
    
    print("=== EVENT-SPECIFIC RASTER ANALYSIS COMPLETE ===")
    
    return results
























if __name__ == '__main__':
    print('Starting ROI motif labeling analysis...\n')
    
    
    
    
# %%

# Update the configuration section at the bottom:
cfg = {
    'tensor_prep': {
        'dt': 0.067,  # ~15 Hz sampling for computational efficiency
    },
    'cp_decomposition': {
        'n_motifs': 8,
        'max_iter': 100,
        'tolerance': 1e-6,
    },
    'motif_audit': {
        'event_enrichment_thresh': 2.0,    # Z-score threshold for event enrichment
        'choice_auc_thresh': 0.65,         # AUC threshold for choice information (is_right_choice)
        'stimulus_auc_thresh': 0.65,       # AUC threshold for stimulus information (is_right)
        'isi_coupling_thresh': 0.1,        # R² threshold for ISI coupling
        'outcome_auc_thresh': 0.65,        # AUC threshold for outcome information (rewarded)
        'nochoice_auc_thresh': 0.65,       # AUC threshold for no-choice information (did_not_choose)
    },
    'roi_sets': {
        'threshold_method': 'percentile',   # 'percentile', 'otsu', 'mean_plus_std'
        'threshold_percentile': 75,
    }
}

PRIMARY_FUNCTIONAL_GROUPS = {
    'f1_onset': {
        'event': 'start_flash_1', 
        'window': (-0.1, 0.2),
        'description': 'F1-onset locked ROIs'
    },
    'f1_offset_isi_start': {
        'event': 'end_flash_1',
        'window': (-0.05, 0.2),
        'description': 'F1-offset / ISI-start locked ROIs'
    },
    'f2_onset': {
        'event': 'start_flash_2',
        'window': (-0.1, 0.2),
        'description': 'F2-onset locked ROIs'  
    },
    'f2_offset_choice_start': {
        'event': 'end_flash_2',
        'window': (-0.1, 0.3),
        'description': 'F2-offset / choice-start locked ROIs'
    },
    'choice_execution_lick': {
        'event': 'lick_start',
        'window': (-0.2, 0.3),
        'description': 'Choice execution / lick-start locked ROIs'
    },
    'isi_duration_sensitive': {
        'event': 'end_flash_1',
        'window': (0.2, 'f2_onset'),  # CRITICAL: Stop at F2 onset!
        'description': 'ISI-duration sensitive ROIs',
        'special_analysis': 'isi_sensitivity_f1_to_f2_only'
    }
}


# %%


# path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_simplex_20250529_2afc-379/sid_imaging_segmented_data.pkl'
path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/sid_imaging_segmented_data.pkl'



path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/trial_start.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_trial_start = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_trial_start.keys())}")




path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/start_flash_1.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_start_flash_1 = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_start_flash_1.keys())}")




path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/end_flash_1.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_end_flash_1 = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_end_flash_1.keys())}")




path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/start_flash_2.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_start_flash_2 = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_start_flash_2.keys())}")






path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/end_flash_2.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_end_flash_2 = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_end_flash_2.keys())}")




path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/choice_start.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_choice_start = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_choice_start.keys())}")





path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/lick_start.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_lick_start = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_lick_start.keys())}")



trial_data_trial_start
trial_data_start_flash_1
trial_data_end_flash_1
trial_data_start_flash_2
trial_data_end_flash_2
trial_data_choice_start
trial_data_lick_start

# %%
# %%
    
# Run the analysis pipeline
try:
    # Step 1: Prepare trial tensor
    X, time_grid, trial_info = prepare_trial_tensor(trial_data, cfg)
    
    # Step 2: Fit CP decomposition (unsupervised motif discovery)
    cp_result = fit_cp_decomposition(X, time_grid, cfg)
    
    # Step 3: Audit motifs against behavioral events (post-hoc labeling)
    audit_result = audit_motifs_against_events(cp_result, trial_info, cfg)
    
    # Step 4: Build discrete ROI sets
    roi_sets = build_roi_sets_from_motifs(cp_result, audit_result, cfg)
    
    print("✓ ROI motif labeling completed successfully!")
    
    # Save results for further analysis
    results = {
        'cp_result': cp_result,
        'audit_result': audit_result, 
        'roi_sets': roi_sets,
        'trial_info': trial_info,
        'time_grid': time_grid,
        'config': cfg
    }
    
    print(f"\nResults contain {len(roi_sets)} ROI sets:")
    for name, indices in roi_sets.items():
        print(f"  {name}: {len(indices)} ROIs")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    


# %%


try:
    # Step 1: Prepare trial tensor
    X, time_grid, trial_info = prepare_trial_tensor(trial_data, cfg)    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

    
# %%


try:
    # Step 2: Fit CP decomposition (unsupervised motif discovery)
    cp_result = fit_cp_decomposition(X, time_grid, cfg)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    
# %%


try:
    # Step 3: Audit motifs against behavioral events (post-hoc labeling)
    audit_result = audit_motifs_against_events(cp_result, trial_info, cfg)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    
    
# %%


try:

    # Step 4: Build discrete ROI sets
    roi_sets = build_roi_sets_from_motifs(cp_result, audit_result, cfg)
    
    print("✓ ROI motif labeling completed successfully!")
    
    # Save results for further analysis
    results = {
        'cp_result': cp_result,
        'audit_result': audit_result, 
        'roi_sets': roi_sets,
        'trial_info': trial_info,
        'time_grid': time_grid,
        'config': cfg
    }
    
    print(f"\nResults contain {len(roi_sets)} ROI sets:")
    for name, indices in roi_sets.items():
        print(f"  {name}: {len(indices)} ROIs")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    
   
   
# %%


try:
 # Generate all plots
    print("\nGenerating visualization plots...")
    
    fig1 = plot_motif_overview(cp_result, audit_result, roi_sets, cfg)
    fig1.suptitle('Motif Discovery Overview', fontsize=16)
    
    fig2 = plot_roi_set_spatial_maps(roi_sets, trial_data, cfg)
    if fig2 is not None:
        fig2.suptitle('ROI Set Spatial Maps', fontsize=16)
    
    fig3 = plot_roi_set_overlap_matrix(roi_sets)
    if fig3 is not None:
        fig3.suptitle('ROI Set Overlap Matrix', fontsize=16)
    
    fig4 = plot_trial_motif_activations(cp_result, trial_info, audit_result, cfg)
    fig4.suptitle('Trial-by-Trial Motif Activations', fontsize=16)
    
    # Print summary
    print_roi_set_summary(roi_sets, audit_result)
    
    # Package results
    results = {
        'cp_result': cp_result,
        'audit_result': audit_result,
        'roi_sets': roi_sets,
        'trial_info': trial_info,
        'time_grid': time_grid,
        'config': cfg,
        'figures': {
            'motif_overview': fig1,
            'spatial_maps': fig2,
            'overlap_matrix': fig3,
            'trial_activations': fig4
        }
    }
    
    print("\n" + "=" * 80)
    print("ROI MOTIF ANALYSIS COMPLETE!")
    print("=" * 80)   
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    
    
# %%


try:

    # Add to your main analysis:

    # After audit_result is created, add:
    debug_motif_scores(audit_result)
    analyze_trial_info_distribution(trial_info)

    # Re-run labeling with stricter criteria
    print("\n=== RE-LABELING WITH STRICT CRITERIA ===")
    strict_cfg = cfg['motif_audit'].copy()
    strict_cfg.update({
        'event_enrichment_thresh': 3.0,
        'choice_auc_thresh': 0.70,
        'stimulus_auc_thresh': 0.70,
        'outcome_auc_thresh': 0.70,
        'isi_coupling_thresh': 0.15,
        'nochoice_auc_thresh': 0.70,
    })

    strict_labels = []
    for k in range(len(audit_result['motif_labels'])):
        scores = audit_result['motif_scores'][f'motif_{k}']
        strict_label = assign_motif_label_strict(scores, strict_cfg)
        strict_labels.append(strict_label)
        print(f"Motif {k}: {audit_result['motif_labels'][k]} -> {strict_label}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()   
   
   
   
   
   
   
   
# %%


try:

# Add this after your existing analysis in the main section:

    # Add the neural data tensor to results for raster analysis
    results['X'] = X
    
    # Run comprehensive raster analysis
    results = run_raster_analysis(results, target_isis=[200, 450, 700, 1700, 2300])
    
    # Display the raster plots
    raster_figs = results['raster_analysis']['raster_figures']
    
    for fig_name, fig in raster_figs.items():
        if fig is not None:
            fig.show()
            print(f"Displaying {fig_name}")
    
    # Print temporal dynamics summary
    dynamics = results['raster_analysis']['temporal_dynamics']
    print("\n=== TEMPORAL DYNAMICS SUMMARY ===")
    for group_name, group_dynamics in dynamics.items():
        print(f"\n{group_name.upper()}:")
        for motif_key, motif_data in group_dynamics.items():
            peak_time = motif_data['peak_time']
            duration = motif_data['duration']
            print(f"  {motif_key}: Peak at {peak_time:.3f}s, Duration {duration:.3f}s")
    
    plt.show()
    
except Exception as e:
    print(f"ERROR in raster analysis: {e}")
    import traceback
    traceback.print_exc()

  
  
# %%
  
  

# Replace the previous raster analysis with:
try:
    # Add the neural data tensor to results for raster analysis
    results['X'] = X
    
    # Run simplified raster analysis
    results = run_simplified_raster_analysis(results)
    
    # Display the simplified rasters
    for fig_name, fig in results['simplified_rasters'].items():
        fig.show()
        print(f"Displaying {fig_name}")
    
    plt.show()
    
except Exception as e:
    print(f"ERROR in simplified raster analysis: {e}")
    import traceback
    traceback.print_exc()
  
  
   
   
# %%


try:
    # Run full analysis with plots
    full_results = run_full_roi_analysis_with_plots(trial_data, cfg)
    
    # Optionally save results
    # save_filepath = path.replace('.pkl', '_roi_motif_results.pkl')
    # save_roi_sets_to_file(roi_sets, cp_result, audit_result, save_filepath)
    
    plt.show() 
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
   
   

  
# %%
# Update configuration with the working thresholds
cfg_event_specific = {
    'tensor_prep': {
        'use_native_resolution': True,
    },
    'event_analysis': {
        'event_locking_threshold': 0.5,     # Working threshold from debug
        'isi_sensitivity_threshold': 0.3,   # Lower for ISI sensitivity
    },
    'roi_sets': {
        'min_group_size': 5,  # Require at least 5 ROIs per group
    }
}



try:
    print("=== RUNNING COMPLETE EVENT-SPECIFIC ANALYSIS ===")
    
    # Run complete event-specific analysis with working parameters
    event_specific_results = run_complete_event_specific_analysis(trial_data, cfg_event_specific)
    
    # Create raster plots
    event_specific_results = plot_event_specific_raster_analysis(
        event_specific_results, target_isis=[200, 700, 1700]
    )
    
    # Display results summary
    print("\n=== EVENT-SPECIFIC ANALYSIS SUMMARY ===")
    roi_sets = event_specific_results['roi_sets']
    
    for group_name, roi_indices in roi_sets.items():
        print(f"\n{group_name.upper()}:")
        print(f"  ROIs detected: {len(roi_indices)}")
        print(f"  Representative ROIs: {roi_indices[:10].tolist()}{'...' if len(roi_indices) > 10 else ''}")
        
        # Show detection method info
        group_data = event_specific_results['roi_groups'].get(group_name, {})
        if 'analysis_method' in group_data:
            print(f"  Detection method: {group_data['analysis_method']}")
        if 'threshold_used' in group_data:
            print(f"  Threshold used: {group_data['threshold_used']}")
        if 'trials_analyzed' in group_data:
            print(f"  Trials analyzed: {group_data['trials_analyzed']}")
    
    # Display raster plots
    if 'event_specific_rasters' in event_specific_results:
        for fig_name, fig in event_specific_results['event_specific_rasters'].items():
            fig.show()
            print(f"Displaying {fig_name}")
    
    plt.show()
    
    print("\n=== EVENT-SPECIFIC ANALYSIS COMPLETE! ===")
    
except Exception as e:
    print(f"ERROR in event-specific analysis: {e}")
    import traceback
    traceback.print_exc()
  
  
  
  
  
  
  
  
  
# %%
# Update configuration with much lower thresholds for trial-averaged detection
cfg_event_specific = {
    'tensor_prep': {
        'use_native_resolution': True,
    },
    'event_analysis': {
        'event_locking_threshold': 0.5,     # Much lower - trial averaging makes signals cleaner
        'isi_sensitivity_threshold': 0.3,   # Much lower for sustained modulation
    },
    'roi_sets': {
        'min_group_size': 1,  # Allow smaller groups initially
    }
}

try:
    # event_specific_results = run_event_specific_roi_analysis(trial_data, cfg_event_specific)
    

# Run debug version to understand the detection issues

    print("=== RUNNING DEBUG EVENT-SPECIFIC ANALYSIS ===")
    
    # Use debug configuration with lower thresholds
    debug_cfg = {
        'event_analysis': {
            'event_locking_threshold': 0.3,  # Very low threshold for testing
            'isi_sensitivity_threshold': 0.2,
        }
    }
    
    # Prepare high-res tensor (if not already done)
    # if 'X' not in locals():
    X, time_grid, trial_info = prepare_trial_tensor_high_res(trial_data, cfg_event_specific)
    
    # Run debug analysis
    debug_roi_groups = analyze_event_specific_roi_groups_debug(X, time_grid, trial_info, debug_cfg)
    
    print("\nDEBUG RESULTS:")
    for group_name, group_data in debug_roi_groups.items():
        if 'roi_indices' in group_data:
            print(f"  {group_name}: {len(group_data['roi_indices'])} ROIs detected")
            if 'roi_scores' in group_data:
                scores = group_data['roi_scores']
                print(f"    Score stats: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")
    
except Exception as e:
    print(f"DEBUG ERROR: {e}")
    import traceback
    traceback.print_exc()
    
  
   
   

    
# %%


try:
    print('holder')
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()