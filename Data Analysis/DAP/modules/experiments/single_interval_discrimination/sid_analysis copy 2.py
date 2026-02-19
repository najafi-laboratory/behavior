"""
Minimal MVP for Lobule V PC timing analysis on trial-segmented dF/F.

Focus: adapter → metrics with hardcoded, sensible defaults (few configs).
- Alignment: F1-OFF (end of flash 1) = 0 s for pre-F2 analyses.
- Time grid: 0 .. max(ISI) seconds, 240 bins (uniform). Per-trial mask for t >= ISI.
- Metrics: scaling-vs-clock (ΔR²), time-resolved decoders (ISI R²(t), AUC(t), divergence),
           hazard unique variance; raster sort index.
- Optional F2-locked PE can be added later without changing the rest.

Inputs expected (from your DataFrame columns):
- dff_segment: (n_rois, n_samples)
- dff_time_vector: (n_samples,) (seconds)
- start_flash_1, end_flash_1, start_flash_2, isi (seconds)
- is_right, is_right_choice, rewarded, punished, did_not_choose, time_did_not_choose,
  servo_in, servo_out, lick, lick_start, RT
- trial_data['session_info']['unique_isis'] (seconds or milliseconds; auto-converted)

Outputs: result dict with fields described in the Spec Sheet.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, roc_auc_score






# -----------------------------
# Adapter: trial_data → M
# -----------------------------

def _to_seconds(arr_like):
    """Convert array-like values to seconds, assuming ms if values > 10."""
    arr = np.asarray(arr_like, dtype=float)
    print(f"_to_seconds input range: {np.nanmin(arr):.3f} to {np.nanmax(arr):.3f}")
    # If values look like ms (e.g., >10), convert to seconds.
    if np.nanmax(arr) > 10.0:
        arr = arr / 1000.0
        print(f"Converted from ms to seconds. New range: {np.nanmin(arr):.3f} to {np.nanmax(arr):.3f}")
    return arr


def build_M_from_trials(trial_data: Dict[str, Any], grid_bins: int = 240) -> Dict[str, Any]:
    """Construct canonical M from trial_data['df_trials_with_segments'].
    - F1-OFF aligned pre-F2 grid of length max(ISI), 240 bins.
    - Per-trial NaN mask for t >= ISI.
    """
    print("=== Starting build_M_from_trials ===")
    
    df = trial_data['df_trials_with_segments']
    n_trials = len(df)
    print(f"Number of trials: {n_trials}")

    # Extract arrays from DataFrame-like object reliably
    # Get ISIs (seconds) and allowed set
    isi_vec = _to_seconds(df['isi'].to_numpy())
    isi_allowed = _to_seconds(trial_data['session_info']['unique_isis'])
    isi_allowed = np.sort(np.unique(isi_allowed))
    
    print(f"ISI vector shape: {isi_vec.shape}, range: {np.nanmin(isi_vec):.3f} to {np.nanmax(isi_vec):.3f}")
    print(f"Allowed ISIs: {isi_allowed}")

    # Time grid: 0..max(ISI) - creates uniform time axis for all trials
    t_max = float(np.nanmax(isi_vec))
    Nt = int(grid_bins)
    time = np.linspace(0.0, t_max, Nt)
    print(f"Time grid: {Nt} bins from 0 to {t_max:.3f} seconds")
    print(f"Time resolution: {(time[1] - time[0]):.4f} seconds per bin")

    # Initialize holders (we infer n_rois from first row)
    first = df.iloc[0]
    dff0 = np.asarray(first['dff_segment'], dtype=float)
    if dff0.ndim != 2:
        raise ValueError("dff_segment must be 2D (n_rois x n_samples) per trial")
    n_rois = dff0.shape[0]
    print(f"Number of ROIs: {n_rois}")
    print(f"First trial dF/F shape: {dff0.shape}")
    
    # Initialize ROI traces array (trials x rois x time_bins)
    roi_traces = np.full((n_trials, n_rois, Nt), np.nan, dtype=np.float32)
    print(f"Initialized roi_traces shape: {roi_traces.shape}")

    # Initialize behavioral arrays
    is_right_trial = np.zeros(n_trials, dtype=bool)
    is_right_choice = np.zeros(n_trials, dtype=bool)
    rewarded = np.zeros(n_trials, dtype=bool)
    punished = np.zeros(n_trials, dtype=bool)
    did_not_choose = np.zeros(n_trials, dtype=bool)
    time_dnc = np.full(n_trials, np.nan, dtype=float)
    servo_in = np.full(n_trials, np.nan, dtype=float)
    servo_out = np.full(n_trials, np.nan, dtype=float)
    lick = np.zeros(n_trials, dtype=bool)
    lick_start = np.full(n_trials, np.nan, dtype=float)
    RT = np.full(n_trials, np.nan, dtype=float)

    F1_on = np.full(n_trials, np.nan, dtype=float)
    F1_off = np.full(n_trials, np.nan, dtype=float)

    print("Processing individual trials...")
    nan_count = 0
    interpolation_issues = 0
    
    for idx, (trial_idx, row) in enumerate(df.iterrows()):
        if idx % 50 == 0:
            print(f"Processing trial {idx}/{n_trials}")
            
        # Extract basic timing - F1_off becomes the reference point (t=0)
        F1_on[idx] = float(row['start_flash_1'])
        F1_off[idx] = float(row['end_flash_1'])
        
        # Align trial's dF/F to F1-OFF = 0
        t_vec = np.asarray(row['dff_time_vector'], dtype=float) - F1_off[idx]
        dff = np.asarray(row['dff_segment'], dtype=float)  # (rois, samples)
        
        if idx == 0:
            print(f"Trial 0 original time range: {np.min(t_vec + F1_off[idx]):.3f} to {np.max(t_vec + F1_off[idx]):.3f}")
            print(f"Trial 0 aligned time range: {np.min(t_vec):.3f} to {np.max(t_vec):.3f}")
            print(f"Trial 0 dF/F shape: {dff.shape}")
        
        # Interpolate each ROI to common grid; mask outside native coverage
        # np.interp extrapolates with endpoints; we replace out-of-range with NaN
        t_min, t_max_local = np.nanmin(t_vec), np.nanmax(t_vec)
        
        for r in range(n_rois):
            y = dff[r]
            # Guard against non-increasing t_vec - ensure monotonic time series
            if not np.all(np.diff(t_vec) > 0):
                # Sort by time if needed
                order = np.argsort(t_vec)
                tt = t_vec[order]
                yy = y[order]
                if idx == 0 and r == 0:
                    print("Warning: Non-monotonic time vector detected, sorting...")
            else:
                tt, yy = t_vec, y
                
            # Interpolate to common time grid
            vals = np.interp(time, tt, yy, left=np.nan, right=np.nan)
            # Replace extrapolated endpoints with NaN explicitly
            vals[(time < t_min) | (time > t_max_local)] = np.nan
            roi_traces[idx, r] = vals
            
            # Count NaNs for debugging
            nan_count += np.isnan(vals).sum()

        # Per-trial mask: drop samples at/after that trial's F2 (ISI from F1-OFF)
        # This ensures we only analyze pre-F2 period
        isi = isi_vec[idx]
        mask_indices = time >= isi
        roi_traces[idx, :, mask_indices] = np.nan
        
        if idx == 0:
            print(f"Trial 0 ISI: {isi:.3f}s, masked {mask_indices.sum()} time points")

        # Extract behavior labels
        is_right_trial[idx] = bool(row.get('is_right', False))
        is_right_choice[idx] = bool(row.get('is_right_choice', False))
        rewarded[idx] = bool(row.get('rewarded', False))
        punished[idx] = bool(row.get('punished', False))
        did_not_choose[idx] = bool(row.get('did_not_choose', False))
        time_dnc[idx] = float(row.get('time_did_not_choose', np.nan))
        servo_in[idx] = float(row.get('servo_in', np.nan))
        servo_out[idx] = float(row.get('servo_out', np.nan))
        lick[idx] = bool(row.get('lick', False))
        lick_start[idx] = float(row.get('lick_start', np.nan))
        RT[idx] = float(row.get('RT', np.nan))

    # Label short vs long trials based on predefined ISI sets
    short_set = set(_to_seconds([200, 325, 450, 575, 700]))
    is_short = np.array([1 if float(isi) in short_set else 0 for isi in isi_vec], dtype=np.uint8)
    
    print(f"Short trials: {is_short.sum()}/{len(is_short)}")
    print(f"Total NaN values in roi_traces: {nan_count}")
    print(f"Final roi_traces shape: {roi_traces.shape}")
    print(f"Behavioral summary:")
    print(f"  Right trials: {is_right_trial.sum()}")
    print(f"  Right choices: {is_right_choice.sum()}")
    print(f"  Rewarded: {rewarded.sum()}")
    print(f"  Punished: {punished.sum()}")
    print(f"  Did not choose: {did_not_choose.sum()}")

    M = {
        'roi_traces': roi_traces,
        'time': time.astype(np.float32),
        'F2_time': isi_vec.astype(np.float32),
        'is_short': is_short,
        'isi_allowed': isi_allowed.astype(np.float32),
        # helpful behavior/context
        'is_right_trial': is_right_trial,
        'is_right_choice': is_right_choice,
        'rewarded': rewarded,
        'punished': punished,
        'did_not_choose': did_not_choose,
        'time_did_not_choose': time_dnc,
        'servo_in': servo_in,
        'servo_out': servo_out,
        'lick': lick,
        'lick_start': lick_start,
        'RT': RT,
    }
    
    print("=== Completed build_M_from_trials ===\n")
    return M

# -----------------------------
# Core metrics (NaN-safe)
# -----------------------------

def _phase_time(T: np.ndarray, isi: float) -> np.ndarray:
    """Convert absolute time to phase (0-1) relative to ISI duration."""
    ph = T / max(1e-9, isi)
    return np.clip(ph, 0.0, 1.0)


def _build_bases(T: np.ndarray, F2: np.ndarray, knots_phase: int = 8, knots_time: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Build spline basis functions for phase-based (scaling) and time-based (clock) models.
    
    Returns:
    - B_phase: (n_trials, n_timepoints, n_phase_features) - each trial has different phase mapping
    - B_time: (n_trials, n_timepoints, n_time_features) - absolute time features (same across trials)
    """
    print(f"Building spline bases with {knots_phase} phase knots, {knots_time} time knots")
    
    n_trials = len(F2)
    print(f"Number of trials: {n_trials}")
    print(f"Time vector length: {len(T)}")
    print(f"F2 times range: {np.nanmin(F2):.3f} to {np.nanmax(F2):.3f}")
    
    # Build time-based (clock) spline basis - same for all trials
    st_time = SplineTransformer(degree=3, n_knots=knots_time, include_bias=False)
    B_time = st_time.fit_transform(T.reshape(-1,1))  # (Nt, b_time)
    print(f"Time basis shape: {B_time.shape}")
    
    # Broadcast to all trials (same time basis for each trial)
    B_time = np.broadcast_to(B_time, (n_trials, *B_time.shape))
    print(f"Broadcasted time basis shape: {B_time.shape}")

    # Build phase-based (scaling) spline basis - different for each trial based on ISI
    st_phase = SplineTransformer(degree=3, n_knots=knots_phase, include_bias=False)
    grid01 = np.linspace(0,1,len(T)).reshape(-1,1)
    st_phase.fit(grid01)  # Fit on 0-1 grid
    
    B_phase = np.zeros((n_trials, len(T), st_phase.n_features_out_))
    print(f"Phase basis shape: {B_phase.shape}")
    
    # For each trial, convert absolute time to phase based on that trial's ISI
    for i in range(n_trials):
        ph = _phase_time(T, F2[i]).reshape(-1,1)  # Convert time to 0-1 phase
        B_phase[i] = st_phase.transform(ph)
        
        if i == 0:
            print(f"Trial 0 ISI: {F2[i]:.3f}, phase range: {ph.min():.3f} to {ph.max():.3f}")
    
    return B_phase, B_time


def _cv_r2_roi_with_nans(Y: np.ndarray, B: np.ndarray, cv_folds: int = 5) -> float:
    """Trial-wise CV R² for one ROI, dropping NaNs safely.
    
    Performs cross-validation by holding out entire trials (not individual timepoints)
    to avoid data leakage between training and test sets.
    
    Y: (Ntr, Nt) - neural activity for one ROI across trials and time
    B: (Ntr, Nt, Nb) - basis functions (either phase or time)
    """
    Ntr, Nt, Nb = B.shape
    print(f"CV R2 calculation: {Ntr} trials, {Nt} timepoints, {Nb} basis functions")
    
    # Flatten to (Ntr*Nt, Nb) and (Ntr*Nt,) for regression
    X_all = B.reshape(Ntr*Nt, Nb)
    y_all = Y.reshape(Ntr*Nt)
    
    # Only use finite (non-NaN) data points
    finite = np.isfinite(y_all)
    finite_count = finite.sum()
    print(f"Finite data points: {finite_count}/{len(y_all)} ({100*finite_count/len(y_all):.1f}%)")
    
    if finite_count < 10:
        print("Insufficient finite data points for CV")
        return np.nan
        
    # Create trial IDs for each data point to enable trial-wise CV
    trial_ids = np.repeat(np.arange(Ntr), Nt)

    preds = np.full_like(y_all, np.nan, dtype=float)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
    
    successful_folds = 0
    for fold_idx, (train_tr, test_tr) in enumerate(kf.split(np.arange(Ntr))):
        # Create masks for training and test data points
        tr_mask = np.isin(trial_ids, train_tr) & finite
        te_mask = np.isin(trial_ids, test_tr) & finite
        
        tr_count = tr_mask.sum()
        te_count = te_mask.sum()
        
        if tr_count < Nb + 2 or te_count == 0:
            print(f"Fold {fold_idx}: insufficient data (train: {tr_count}, test: {te_count})")
            continue
            
        # Fit ridge regression on training data
        reg = Ridge(alpha=1.0).fit(X_all[tr_mask], y_all[tr_mask])
        preds[te_mask] = reg.predict(X_all[te_mask])
        successful_folds += 1
        
        if fold_idx == 0:
            print(f"Fold {fold_idx}: train={tr_count}, test={te_count} points")
    
    print(f"Successful CV folds: {successful_folds}/{cv_folds}")
    
    # Calculate R² on all predicted vs actual values
    ok = np.isfinite(y_all) & np.isfinite(preds)
    ok_count = ok.sum()
    
    if ok_count < 10:
        print("Insufficient valid predictions for R² calculation")
        return np.nan
        
    r2 = r2_score(y_all[ok], preds[ok])
    print(f"Final R² calculated on {ok_count} points: {r2:.4f}")
    return r2


def scaling_vs_clock(M: Dict[str, Any], cv_folds: int = 5, knots_phase: int = 8, knots_time: int = 8):
    """Compare scaling (phase-based) vs clock (time-based) models for each ROI.
    
    Scaling model: neural activity varies with phase of interval (0-1)
    Clock model: neural activity varies with absolute time from stimulus
    
    Returns:
    - delta: difference in R² (phase - time) for each ROI
    - pref: dict categorizing ROIs by their preference
    - (r2_phase, r2_ms): individual R² values for phase and time models
    """
    print("=== Starting scaling_vs_clock analysis ===")
    
    X = M['roi_traces']              # (Ntr, Nroi, Nt)
    T = M['time']
    F2 = M['F2_time']
    Ntr, Nroi, Nt = X.shape
    
    print(f"Data shape: {Ntr} trials, {Nroi} ROIs, {Nt} timepoints")
    print(f"Time range: {T[0]:.3f} to {T[-1]:.3f} seconds")
    
    # Build basis functions for both models
    Bp, Bt = _build_bases(T, F2, knots_phase, knots_time)
    print(f"Phase basis shape: {Bp.shape}")
    print(f"Time basis shape: {Bt.shape}")

    # Initialize results arrays
    r2_phase = np.zeros(Nroi)
    r2_ms = np.zeros(Nroi)
    
    print("Computing R² for each ROI...")
    
    for r in range(Nroi):
        if r % 20 == 0:
            print(f"Processing ROI {r}/{Nroi}")
            
        Y = X[:, r, :]  # (Ntr, Nt) - this ROI's activity across trials and time
        
        # Calculate cross-validated R² for both models
        r2_phase[r] = _cv_r2_roi_with_nans(Y, Bp, cv_folds)
        r2_ms[r] = _cv_r2_roi_with_nans(Y, Bt, cv_folds)
        
        if r < 3:  # Print details for first few ROIs
            print(f"ROI {r}: Phase R²={r2_phase[r]:.4f}, Time R²={r2_ms[r]:.4f}")
    
    # Calculate preference scores (ΔR² = R²_phase - R²_time)
    delta = r2_phase - r2_ms
    
    print(f"Delta R² range: {np.nanmin(delta):.4f} to {np.nanmax(delta):.4f}")
    print(f"Mean delta R²: {np.nanmean(delta):.4f}")
    
    # Categorize ROIs by preference (using 0.02 threshold)
    eps = 0.02
    scaling_rois = np.where(delta >  eps)[0]
    clock_rois = np.where(delta < -eps)[0] 
    mixed_rois = np.where(np.abs(delta) <= eps)[0]
    
    pref = {
        'scaling': scaling_rois,
        'clock':   clock_rois,
        'mixed':   mixed_rois,
    }
    
    print(f"ROI preferences:")
    print(f"  Scaling: {len(scaling_rois)} ROIs")
    print(f"  Clock: {len(clock_rois)} ROIs") 
    print(f"  Mixed: {len(mixed_rois)} ROIs")
    
    print("=== Completed scaling_vs_clock analysis ===\n")
    return delta, pref, (r2_phase, r2_ms)


def _bin_time_features(X: np.ndarray, T: np.ndarray, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Bin neural activity across time for time-resolved decoding.
    
    Divides the time axis into bins and averages activity within each bin.
    
    Returns:
    - F: (Ntr, Nroi, n_bins) - binned features
    - centers: (n_bins,) - time bin centers
    """
    print(f"Binning time features into {n_bins} bins")
    
    # Create time bin edges and centers
    edges = np.linspace(T[0], T[-1], n_bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    
    print(f"Time bins: {edges[0]:.3f} to {edges[-1]:.3f}, bin width: {edges[1]-edges[0]:.3f}")
    
    Ntr, Nroi, Nt = X.shape
    F = np.full((Ntr, Nroi, n_bins), np.nan, dtype=float)
    
    # Average activity within each time bin
    for b in range(n_bins):
        # Find timepoints within this bin
        m = (T >= edges[b]) & (T < edges[b+1])
        bin_points = m.sum()
        
        if not np.any(m):
            print(f"Warning: No timepoints in bin {b}")
            continue
            
        # NaN-safe mean across time samples in bin
        F[:, :, b] = np.nanmean(X[:, :, m], axis=2)
        
        if b == 0:
            print(f"Bin 0: {bin_points} timepoints, center at {centers[b]:.3f}s")
    
    # Count NaNs
    nan_count = np.isnan(F).sum()
    total_count = F.size
    print(f"Binned features: {nan_count}/{total_count} NaNs ({100*nan_count/total_count:.1f}%)")
    
    return F, centers


def _cv_ridge_r2_with_mask(Z: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Cross-validated Ridge regression R² with NaN handling.
    
    Used for continuous target prediction (e.g., ISI values).
    """
    # Drop rows with NaNs in features or target
    valid = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    valid_count = valid.sum()
    unique_targets = np.unique(y[valid]).size if valid_count > 0 else 0
    
    print(f"Ridge CV: {valid_count} valid samples, {unique_targets} unique targets")
    
    if valid_count < 10 or unique_targets < 2:
        print("Insufficient data for Ridge regression")
        return np.nan
        
    Z = Z[valid]
    y = y[valid]
    
    # Perform cross-validation
    kf = KFold(n_splits=min(k, len(y)), shuffle=True, random_state=0)
    preds = np.zeros_like(y, dtype=float)
    
    for fold_idx, (train, test) in enumerate(kf.split(Z)):
        m = Ridge(alpha=1.0).fit(Z[train], y[train])
        preds[test] = m.predict(Z[test])
    
    r2 = r2_score(y, preds)
    print(f"Ridge R²: {r2:.4f}")
    return r2


def _cv_logistic_auc_with_mask(Z: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Cross-validated Logistic regression AUC with NaN handling.
    
    Used for binary classification (e.g., short vs long trials).
    """
    valid = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    valid_count = valid.sum()
    
    if valid_count < 10:
        print(f"Logistic CV: insufficient data ({valid_count} samples)")
        return np.nan
        
    Z = Z[valid]
    y = y[valid].astype(int)
    
    # Require both classes present
    unique_classes = np.unique(y)
    if unique_classes.size < 2:
        print(f"Logistic CV: only {unique_classes.size} class(es) present")
        return np.nan
    
    print(f"Logistic CV: {valid_count} valid samples, classes: {unique_classes}")
    
    kf = KFold(n_splits=min(k, len(y)), shuffle=True, random_state=0)
    scores = []
    
    for fold_idx, (train, test) in enumerate(kf.split(Z)):
        # Ensure test has both classes; if not, skip that fold
        test_classes = np.unique(y[test])
        if test_classes.size < 2:
            print(f"Fold {fold_idx}: test set has only {test_classes.size} class(es), skipping")
            continue
            
        m = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, max_iter=1000)
        m.fit(Z[train], y[train])
        p = m.predict_proba(Z[test])[:,1]  # Probability of positive class
        scores.append(roc_auc_score(y[test], p))
    
    if not scores:
        print("No valid folds for AUC calculation")
        return np.nan
        
    auc = float(np.nanmean(scores))
    print(f"Logistic AUC: {auc:.4f} (from {len(scores)} folds)")
    return auc


def time_resolved_decoding(M: Dict[str, Any], n_bins: int = 20) -> Dict[str, Any]:
    """Perform time-resolved decoding of ISI and short/long classification.
    
    At each time bin, use population activity to predict:
    1. Continuous ISI values (R²)
    2. Binary short/long classification (AUC)
    
    Also estimates divergence time when decoding becomes reliable.
    """
    print("=== Starting time_resolved_decoding ===")
    
    X = M['roi_traces']
    T = M['time']
    isi = M['F2_time']
    is_short = M['is_short']
    
    print(f"Decoding from {X.shape[1]} ROIs across {len(T)} timepoints")
    print(f"ISI range: {np.nanmin(isi):.3f} to {np.nanmax(isi):.3f}")
    print(f"Short trials: {is_short.sum()}/{len(is_short)}")
    
    # Bin neural activity across time
    F, centers = _bin_time_features(X, T, n_bins)
    Ntr, Nroi, B = F.shape
    print(f"Binned features shape: {F.shape}")

    # Initialize results
    auc_t = np.full(B, np.nan)
    r2_t = np.full(B, np.nan)
    
    print("Processing each time bin...")
    
    for b in range(B):
        print(f"Time bin {b}/{B}: t={centers[b]:.3f}s")
        
        Z = F[:, :, b]  # (Ntr, Nroi) - population activity at this time bin
        
        # Standardize features across trials (only on valid rows)
        valid_rows = np.all(np.isfinite(Z), axis=1)
        valid_count = valid_rows.sum()
        
        print(f"  Valid trials: {valid_count}/{Ntr}")
        
        if valid_count < 10:
            print(f"  Skipping bin {b}: insufficient valid data")
            continue
            
        # Standardize only the valid data
        Zv = Z[valid_rows]
        scaler = StandardScaler()
        Zs = scaler.fit_transform(Zv)
        
        # Place standardized data back into full array with NaNs for invalid rows
        Z_std = np.full_like(Z, np.nan)
        Z_std[valid_rows] = Zs
        
        # Decode continuous ISI values
        r2_t[b] = _cv_ridge_r2_with_mask(Z_std, isi, k=5)
        
        # Decode binary short/long classification  
        auc_t[b] = _cv_logistic_auc_with_mask(Z_std, is_short, k=5)
        
        print(f"  Results: R²={r2_t[b]:.4f}, AUC={auc_t[b]:.4f}")

    # Calculate divergence time: earliest of 3 consecutive bins with AUC > 0.6
    print("Calculating divergence time...")
    
    divergence_time = np.nan
    thr, sustain = 0.6, 3  # Threshold and required consecutive bins
    run = 0
    
    for b in range(B):
        if np.isfinite(auc_t[b]) and auc_t[b] > thr:
            run += 1
            if run >= sustain:
                # Divergence occurs at start of sustained period
                divergence_time = centers[b - sustain + 1]
                print(f"Divergence detected at t={divergence_time:.3f}s (bins {b-sustain+1}-{b})")
                break
        else:
            run = 0
    
    if np.isnan(divergence_time):
        print("No divergence time detected")
    
    print(f"Time-resolved decoding summary:")
    print(f"  R² range: {np.nanmin(r2_t):.4f} to {np.nanmax(r2_t):.4f}")
    print(f"  AUC range: {np.nanmin(auc_t):.4f} to {np.nanmax(auc_t):.4f}")
    print(f"  Divergence time: {divergence_time:.3f}s" if np.isfinite(divergence_time) else "  No divergence detected")
    
    result = {
        'time_points': centers,
        'isi_r2_t': r2_t,
        'shortlong_auc_t': auc_t,
        'divergence_time': divergence_time,
    }
    
    print("=== Completed time_resolved_decoding ===\n")
    return result


def hazard_unique_variance(M: Dict[str, Any]) -> float:
    """Calculate unique variance explained by hazard rate beyond linear time.
    
    Hazard rate represents the instantaneous probability of interval termination
    given that it hasn't terminated yet. This tests whether neural activity
    tracks this cognitive/temporal expectation signal.
    
    Returns the unique R² contribution of hazard rate after accounting for linear time.
    """
    print("=== Starting hazard_unique_variance ===")
    
    X = M['roi_traces']
    T = M['time']
    F2 = M['F2_time']
    
    print(f"Computing hazard rate from {len(np.unique(F2))} unique ISI values")
    
    # Compute discrete hazard at each absolute time: map to next allowed ISI >= t
    allowed = np.sort(np.unique(F2))
    print(f"Allowed ISI values: {allowed}")
    
    # Empirical prior over allowed ISIs (how often each ISI occurs)
    prior = np.array([(F2 == t).mean() for t in allowed])
    prior = prior / prior.sum()  # Normalize to sum to 1
    print(f"ISI priors: {prior}")
    
    # Calculate survival function and hazard rates
    # Hazard = P(terminate at t | survived to t) = prior[t] / survival[t]
    surv = 1.0  # Start with probability 1 of surviving
    hmap = {}   # Map from ISI to hazard rate
    
    print("Computing hazard rates:")
    for p, t in zip(prior, allowed):
        hazard = p / max(1e-12, surv)  # Avoid division by zero
        hmap[t] = hazard
        print(f"  ISI {t:.3f}s: prior={p:.3f}, survival={surv:.3f}, hazard={hazard:.3f}")
        surv -= p  # Update survival probability
    
    # Map each timepoint to hazard rate of next possible termination
    next_allowed = np.array([
        allowed[np.searchsorted(allowed, tt, side='left')] 
        if np.any(allowed >= tt) else allowed[-1] 
        for tt in T
    ])
    haz_t = np.vectorize(hmap.get)(next_allowed)
    
    print(f"Hazard time series range: {np.min(haz_t):.6f} to {np.max(haz_t):.6f}")

    # Response variable: population average over ROIs, flattened over (trial,time)
    Y = np.nanmean(X, axis=1)  # (Ntr, Nt) - average across ROIs for each trial
    y = Y.reshape(-1)          # Flatten to 1D
    
    print(f"Response variable: {Y.shape} -> {y.shape}")

    # Predictor variables: linear time and hazard(t)
    lin_t = np.tile(T, X.shape[0])  # Repeat time vector for each trial
    H = np.tile(haz_t, X.shape[0])   # Repeat hazard vector for each trial
    
    print(f"Predictor shapes: lin_t={lin_t.shape}, H={H.shape}")

    # Only use finite data points
    finite = np.isfinite(y) & np.isfinite(lin_t) & np.isfinite(H)
    finite_count = finite.sum()
    
    print(f"Finite data points: {finite_count}/{len(y)} ({100*finite_count/len(y):.1f}%)")
    
    if finite_count < 20:
        print("Insufficient finite data for hazard analysis")
        return np.nan

    # Create trial IDs for cross-validation (avoid data leakage)
    Ntr, Nt = Y.shape
    trial_ids = np.repeat(np.arange(Ntr), Nt)
    kf = KFold(n_splits=min(5, Ntr), shuffle=True, random_state=0)

    def _cv_r2(Xcols):
        """Helper function for cross-validated R² calculation."""
        preds = np.full_like(y, np.nan, dtype=float)
        successful_folds = 0
        
        for tr, te in kf.split(np.arange(Ntr)):
            # Create masks for training/test trials
            m_train = np.isin(trial_ids, tr) & finite
            m_test = np.isin(trial_ids, te) & finite
            
            if m_train.sum() < 5 or m_test.sum() == 0:
                continue
                
            model = LinearRegression().fit(Xcols[m_train], y[m_train])
            preds[m_test] = model.predict(Xcols[m_test])
            successful_folds += 1
        
        ok = np.isfinite(y) & np.isfinite(preds)
        
        if ok.sum() > 10:
            r2 = r2_score(y[ok], preds[ok])
            print(f"    CV R² from {successful_folds} folds, {ok.sum()} predictions: {r2:.6f}")
            return r2
        else:
            print(f"    Insufficient predictions: {ok.sum()}")
            return np.nan

    # Compare models: time-only vs time+hazard
    print("Fitting time-only model...")
    X_time = lin_t.reshape(-1,1)
    r2_time = _cv_r2(X_time)
    
    print("Fitting time+hazard model...")
    X_both = np.column_stack([lin_t, H])
    r2_both = _cv_r2(X_both)
    
    # Calculate unique contribution of hazard rate
    if not np.isfinite(r2_time) or not np.isfinite(r2_both):
        print("Invalid R² values - cannot compute unique variance")
        return np.nan
    
    unique_r2 = max(0.0, float(r2_both - r2_time))
    
    print(f"Hazard analysis results:")
    print(f"  Time-only R²: {r2_time:.6f}")
    print(f"  Time+Hazard R²: {r2_both:.6f}")
    print(f"  Unique hazard R²: {unique_r2:.6f}")
    
    print("=== Completed hazard_unique_variance ===\n")
    return unique_r2


def sort_index(M: Dict[str, Any], delta_r2: np.ndarray) -> np.ndarray:
    """Generate sort order for ROIs based on their temporal preference.
    
    For scaling-preferring populations: sort by peak phase (using session-mean ISI)
    For clock-preferring populations: sort by absolute peak latency
    
    This creates a useful visualization order for raster plots.
    """
    print("=== Starting sort_index calculation ===")
    
    X = M['roi_traces']
    T = M['time']
    
    # Average activity across trials for each ROI
    Y = np.nanmean(X, axis=0)  # (Nroi, Nt)
    print(f"ROI-averaged activity shape: {Y.shape}")
    
    # Apply light smoothing for robustness (simple moving average)
    # Use a simple moving average of width 3 to reduce noise
    kernel = np.array([1,2,1], dtype=float)
    kernel = kernel / kernel.sum()
    Yp = np.copy(Y)
    
    print("Applying smoothing to ROI traces...")
    for r in range(Y.shape[0]):
        conv = np.convolve(Y[r], kernel, mode='same')
        Yp[r] = conv

    # Determine population preference from median delta R²
    median_delta = np.nanmedian(delta_r2)
    print(f"Median delta R²: {median_delta:.4f}")
    
    if median_delta > 0:
        print("Population preference: SCALING - sorting by peak phase")
        
        # For scaling: order by peak phase (using session-mean ISI)
        mean_isi = float(np.nanmean(M['F2_time']))
        print(f"Session mean ISI: {mean_isi:.3f}s")
        
        # Convert time to phase using session mean ISI
        ph = T / max(mean_isi, 1e-9)
        ph = np.clip(ph, 0, 1)
        print(f"Phase range: {ph.min():.3f} to {ph.max():.3f}")
        
        # Find peak phase for each ROI
        prefs = []
        for r in range(Yp.shape[0]):
            peak_idx = int(np.nanargmax(Yp[r]))
            peak_phase = ph[peak_idx]
            prefs.append(peak_phase)
            
            if r < 3:  # Show details for first few ROIs
                print(f"ROI {r}: peak at time {T[peak_idx]:.3f}s, phase {peak_phase:.3f}")
        
        # Sort by peak phase (early to late in phase)
        order = np.argsort(prefs)
        print(f"Peak phases range: {np.min(prefs):.3f} to {np.max(prefs):.3f}")
        
    else:
        print("Population preference: CLOCK - sorting by peak latency")
        
        # For clock: order by absolute peak latency
        lats = np.array([int(np.nanargmax(row)) for row in Yp])
        peak_times = T[lats]
        
        print(f"Peak latencies: {np.min(peak_times):.3f}s to {np.max(peak_times):.3f}s")
        
        # Sort by peak latency (early to late in absolute time)
        order = np.argsort(lats)
    
    print(f"Sort order computed for {len(order)} ROIs")
    print(f"First 10 ROIs in sort order: {order[:10]}")
    
    print("=== Completed sort_index calculation ===\n")
    return order


# -----------------------------
# Runner
# -----------------------------

def run_all_from_raw(trial_data: Dict[str, Any]) -> Dict[str, Any]:
    """Complete analysis pipeline from raw trial data to all metrics.
    
    This is the main entry point that runs all analyses in sequence.
    """
    print("========================================")
    print("STARTING COMPLETE SID TIMING ANALYSIS")
    print("========================================")
    
    # Step 1: Build canonical data structure
    M = build_M_from_trials(trial_data, grid_bins=240)
    
    # Step 2: Scaling vs clock analysis
    delta, pref, (r2_phase, r2_ms) = scaling_vs_clock(M, cv_folds=5, knots_phase=8, knots_time=8)
    
    # Step 3: Time-resolved decoding
    decode = time_resolved_decoding(M, n_bins=20)
    
    # Step 4: Hazard rate analysis
    hazard_uv = hazard_unique_variance(M)
    
    # Step 5: Generate sort order for visualization
    order = sort_index(M, delta)
    
    # Compile results
    results = {
        'delta_r2': delta,
        'model_pref': pref,
        'r2_phase': r2_phase,
        'r2_ms': r2_ms,
        'decode': decode,
        'hazard_unique_r2': hazard_uv,
        'sort_index': order,
        'time': M['time'],
    }
    
    print("========================================")
    print("ANALYSIS COMPLETE - SUMMARY:")
    print("========================================")
    print(f"Processed {M['roi_traces'].shape[0]} trials, {M['roi_traces'].shape[1]} ROIs")
    print(f"Scaling preference: {len(pref['scaling'])} ROIs")
    print(f"Clock preference: {len(pref['clock'])} ROIs")
    print(f"Mixed preference: {len(pref['mixed'])} ROIs")
    print(f"Divergence time: {decode['divergence_time']:.3f}s" if np.isfinite(decode['divergence_time']) else "No divergence detected")
    print(f"Hazard unique R²: {hazard_uv:.6f}")
    print("========================================")
    
    return results


if __name__ == "__main__":
    print("This module provides build_M_from_trials(...) and run_all_from_raw(...).")
    
# %%
"""
Plotting utilities for the Lobule V MVP analysis.

Main entry:
    plot_session_onepager(M, R, session_title=None, save_path=None)

Panels rendered (auto-skip if data missing):
1) ΔR² histogram with fractions (scaling/clock/mixed) + median ΔR².
2) Time-resolved decoders: ISI R²(t) and short/long AUC(t) with divergence marker.
3) Hazard unique R² bar.
4) Raster (ROIs × time) sorted by the winning story; vertical lines at allowed ISIs.

Inputs:
- M: canonical session dict produced by the adapter (build_M_from_trials)
- R: results dict from run_all_from_raw or equivalent

Notes:
- Designed for smooth calcium traces (ΔF/F), NaN-safe.
- Uses matplotlib only.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def _safe_percentile(a, q):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return np.percentile(a, q)


def plot_session_onepager(M: dict, R: dict, session_title: str | None = None, save_path: str | None = None):
    """Create a one-page summary figure for a single session.

    Parameters
    ----------
    M : dict
        Canonical session object (must contain 'roi_traces', 'time', 'isi_allowed').
    R : dict
        Results object with keys like 'delta_r2', 'model_pref', 'decode', 'hazard_unique_r2', 'sort_index'.
    session_title : str, optional
        Title for the page (mouse/session/date), by default None.
    save_path : str, optional
        If provided, path to save the figure (PNG/PDF) with dpi=300, by default None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Figure canvas
    fig = plt.figure(figsize=(12, 7.5))
    gs = GridSpec(2, 3, height_ratios=[1.0, 1.6], width_ratios=[1, 1.2, 0.8], wspace=0.35, hspace=0.35)

    # ---------------------
    # Panel 1: ΔR² histogram + fractions
    # ---------------------
    ax1 = fig.add_subplot(gs[0, 0])
    delta = np.asarray(R.get('delta_r2'))
    bins = np.linspace(np.nanmin(delta) if np.isfinite(np.nanmin(delta)) else -0.5,
                       np.nanmax(delta) if np.isfinite(np.nanmax(delta)) else 0.5, 31)
    ax1.hist(delta[np.isfinite(delta)], bins=bins, edgecolor='none')
    ax1.axvline(0, color='k', lw=1, ls='--')
    med = np.nanmedian(delta)
    ax1.axvline(med, color='k', lw=1.5)
    ax1.set_xlabel('ΔR² (phase − ms)')
    ax1.set_ylabel('ROIs')
    ax1.set_title('Scaling vs Clock')
    # Fractions text box
    pref = R.get('model_pref', {})
    n_total = np.isfinite(delta).sum()
    n_scal = len(pref.get('scaling', []))
    n_clock = len(pref.get('clock', []))
    n_mixed = len(pref.get('mixed', []))
    txt = (f"median ΔR² = {med:.03f}\n"
           f"scaling: {n_scal}/{n_total} ({(100*n_scal/max(n_total,1)):.1f}%)\n"
           f"clock:   {n_clock}/{n_total} ({(100*n_clock/max(n_total,1)):.1f}%)\n"
           f"mixed:   {n_mixed}/{n_total} ({(100*n_mixed/max(n_total,1)):.1f}%)")
    ax1.text(0.98, 0.98, txt, transform=ax1.transAxes, va='top', ha='right', fontsize=9,
             bbox=dict(boxstyle='round', fc='white', ec='0.8', alpha=0.9))

    # ---------------------
    # Panel 2: Time-resolved decoders
    # ---------------------
    ax2 = fig.add_subplot(gs[0, 1])
    dec = R.get('decode', {})
    tpts = np.asarray(dec.get('time_points'))
    r2_t = np.asarray(dec.get('isi_r2_t'))
    auc_t = np.asarray(dec.get('shortlong_auc_t'))
    # Left axis: R2(t)
    ax2.plot(tpts, r2_t, lw=2, label='ISI $R^2$(t)')
    ax2.set_xlabel('Time from F1-OFF (s)')
    ax2.set_ylabel('ISI $R^2$(t)')
    # Right axis: AUC(t)
    ax2b = ax2.twinx()
    ax2b.plot(tpts, auc_t, lw=2, ls='--', label='Short/Long AUC(t)')
    ax2b.set_ylabel('AUC(t)')
    # Divergence marker if present
    div_t = dec.get('divergence_time')
    if div_t is not None and np.isfinite(div_t):
        ax2.axvline(div_t, color='k', lw=1.5, ls=':')
        ax2.text(div_t, ax2.get_ylim()[1], 'divergence', rotation=90, va='top', ha='right', fontsize=8)
    # Legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, frameon=True)
    ax2.set_title('Time-resolved decode')

    # ---------------------
    # Panel 3: Hazard unique R² bar
    # ---------------------
    ax3 = fig.add_subplot(gs[0, 2])
    huv = R.get('hazard_unique_r2')
    if huv is None or not np.isfinite(huv):
        ax3.text(0.5, 0.5, 'Hazard UV N/A', va='center', ha='center')
        ax3.set_axis_off()
    else:
        ax3.bar(["Hazard unique $R^2$"], [huv])
        ax3.set_ylim(0, max(0.01, 1.05*huv))
        ax3.set_title('Anticipation beyond ramp')
        for i, v in enumerate([huv]):
            ax3.text(i, v, f" {v:.3f}", va='bottom', ha='left')

    # ---------------------
    # Panel 4: Raster (ROI × time) sorted by winner
    # ---------------------
    ax4 = fig.add_subplot(gs[1, :])
    X = M['roi_traces']  # (Ntr, Nroi, Nt)
    T = M['time']
    order = np.asarray(R.get('sort_index')) if 'sort_index' in R else np.arange(X.shape[1])
    # Average across trials (NaN-safe)
    mean_tr = np.nanmean(X, axis=0)  # (Nroi, Nt)
    mean_tr = mean_tr[order]
    # Color scale from robust percentiles
    vmin = _safe_percentile(mean_tr, 5)
    vmax = _safe_percentile(mean_tr, 95)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(mean_tr), np.nanmax(mean_tr)
    im = ax4.imshow(mean_tr, aspect='auto', interpolation='nearest',
                    extent=[T[0], T[-1], 0, mean_tr.shape[0]], origin='lower',
                    vmin=vmin, vmax=vmax)
    ax4.set_xlabel('Time from F1-OFF (s)')
    ax4.set_ylabel('ROIs (sorted)')
    ax4.set_title('Population raster (avg across trials)')
    # Vertical lines at allowed ISIs
    if 'isi_allowed' in M:
        for isi in np.asarray(M['isi_allowed']):
            ax4.axvline(isi, color='w', lw=0.8, ls=':', alpha=0.8)
    cbar = plt.colorbar(im, ax=ax4, fraction=0.02, pad=0.02)
    cbar.set_label('ΔF/F (a.u.)')

    # Page title
    if session_title:
        fig.suptitle(session_title, y=0.99, fontsize=13)

    # Tight layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
# %%    
    
    path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/sid_imaging_segmented_data.pkl'

    import pickle

    with open(path, 'rb') as f:
        trial_data = pickle.load(f)   # one object back (e.g., a dict)  
        
        
# %%


    print('test')
    
    M = build_M_from_trials(trial_data, grid_bins=240)
    
    print('M return')
    
    
    
# %%

# ...existing code...
delta, pref, (r2_phase, r2_ms) = scaling_vs_clock(M, cv_folds=5, knots_phase=8, knots_time=8)
# ...existing code...

# %%

# ...existing code...
decode = time_resolved_decoding(M, n_bins=20)
# ...existing code...

# %%

# ...existing code...
hazard_uv = hazard_unique_variance(M)
# ...existing code...

# %%

# ...existing code...
order = sort_index(M, delta)
# ...existing code...

# %%

# ...existing code...
R ={
        'delta_r2': delta,
        'model_pref': pref,
        'r2_phase': r2_phase,
        'r2_ms': r2_ms,
        'decode': decode,
        'hazard_unique_r2': hazard_uv,
        'sort_index': order,
        'time': M['time'],        
    }
# ...existing code...


# %%

plot_session_onepager(M, R)



