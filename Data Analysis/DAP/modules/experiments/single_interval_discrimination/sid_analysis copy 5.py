"""
Restored Page‑1 pipeline for Lobule V timing (ISI) + choice, plus group mean
trace plotting. This version mirrors the original layout you liked:

Panels:
 1. Scaling vs Clock (ΔR² histogram)
 2. Time‑resolved decode (ISI R²(t) + Short/Long AUC)
 3. Anticipation beyond ramp (Hazard unique variance)
 4. Trough/turning‑point vs ISI
 5. Fair‑window decoders (amp vs slope)
 6. RSA mean off‑diagonal corr (Abs time vs Phase)
 7. F2‑locked choice decode (AUC from F2)
 8. Population raster (avg across trials, F1‑OFF locked)

Also includes:
 - build_M_from_trials(...)  → uses your DataFrame access pattern and explicit
   short/long ISI sets; converts ms→s automatically.
 - add_choice_locked_to_M(...), add_F2_locked_to_M(...)
 - plot_roi_group_traces(...) → group mean traces like your IMG4 (spaghetti + mean)

Dependencies: numpy, matplotlib, scikit‑learn
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from dataclasses import dataclass
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, roc_auc_score

from matplotlib.gridspec import GridSpec

# -------------------- constants / ISI sets --------------------
SHORT_MS = [200, 325, 450, 575, 700]
LONG_MS  = [1700, 1850, 2000, 2150, 2300]
SHORT_S = np.round(np.asarray(SHORT_MS) / 1000.0, 3)
LONG_S  = np.round(np.asarray(LONG_MS)  / 1000.0, 3)

# ----------------------------- helpers -----------------------------

def _to_seconds(x) -> np.ndarray:
    """Convert values to seconds by heuristic (typical ISIs are in ms)."""
    arr = np.asarray(x, dtype=float)
    print(f"_to_seconds input range: {np.nanmin(arr):.3f} to {np.nanmax(arr):.3f}")
    if not np.any(np.isfinite(arr)):
        return arr
    # med = float(np.nanmedian(np.abs(arr)))
    arr = arr / 1000.0
    return arr
    # return arr / 1000.0 if med > 20.0 else arr

def nanmean_no_warn(a: np.ndarray, axis: int):
    valid = np.isfinite(a)
    cnt = valid.sum(axis=axis, keepdims=True)
    s = np.nansum(a, axis=axis, keepdims=True)
    out = s / np.maximum(cnt, 1)
    out[cnt == 0] = np.nan
    return np.squeeze(out, axis=axis)

# simple RBF basis

def _rbf_basis(z: np.ndarray, n_centers: int, zmin: float, zmax: float, add_const=True) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    centers = np.linspace(zmin, zmax, n_centers)
    width = (zmax - zmin) / max(n_centers - 1, 1)
    width = max(width, (zmax - zmin) / max(4, n_centers)) + 1e-12
    Phi = np.exp(-0.5 * ((z[:, None] - centers[None, :]) / width) ** 2)
    if add_const:
        Phi = np.column_stack([np.ones_like(z), Phi])
    return Phi

def _ridge_fit_predict(Phi_tr: np.ndarray, y_tr: np.ndarray, Phi_te: np.ndarray, alpha=1e-2) -> np.ndarray:
    A = Phi_tr.T @ Phi_tr
    w = np.linalg.solve(A + alpha * np.eye(A.shape[0]), Phi_tr.T @ y_tr)
    return Phi_te @ w

from sklearn.metrics import r2_score

def r2_nan(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3: return np.nan
    return r2_score(y_true[m], y_pred[m])

# window helpers

def auto_preF2_window(M, frac=0.9, min_width=0.35):
    T = M['time']
    earliest_f2 = float(np.nanmin(M['isi'])) if np.isfinite(np.nanmin(M['isi'])) else T[-1]
    t1 = min(frac * earliest_f2, T[-1] - 1e-6)
    t0 = max(0.0, t1 - min_width)
    if t1 <= t0: t0, t1 = 0.0, min(earliest_f2 * 0.8, T[-1])
    return (t0, t1)

def baseline_correct_inplace(M, base_win=(0.0, 0.05)):
    X, T = M['roi_traces'], M['time']
    bm = (T >= base_win[0]) & (T <= base_win[1])
    if np.any(bm):
        base = np.nanmean(X[:, :, bm], axis=2, keepdims=True)
        M['roi_traces'] = X - base
    return M

# ---------------------- build data matrices ----------------------

def build_M_from_trials(trial_data: Dict[str, Any], grid_bins: int = 240) -> Dict[str, Any]:
    print("=== Starting build_M_from_trials ===")
    df = trial_data['df_trials_with_segments']
    n_trials = len(df)
    print(f"Number of trials: {n_trials}")

    isi_vec = _to_seconds(df['isi'].to_numpy())
    isi_allowed = _to_seconds(trial_data['session_info']['unique_isis'])
    isi_allowed = np.sort(np.unique(isi_allowed))
    print(f"ISI vector shape: {isi_vec.shape}, range: {np.nanmin(isi_vec):.3f} to {np.nanmax(isi_vec):.3f}")
    print(f"Allowed ISIs: {isi_allowed}")

    t_max = float(np.nanmax(isi_vec))
    Nt = int(grid_bins)
    time = np.linspace(0.0, t_max, Nt)
    print(f"Time grid: {Nt} bins from 0 to {t_max:.3f} seconds")
    print(f"Time resolution: {(time[1]-time[0]):.4f} seconds per bin")

    first = df.iloc[0]
    dff0 = np.asarray(first['dff_segment'], dtype=float)
    if dff0.ndim != 2:
        raise ValueError("dff_segment must be 2D (n_rois x n_samples) per trial")
    n_rois = dff0.shape[0]
    print(f"Number of ROIs: {n_rois}")
    print(f"First trial dF/F shape: {dff0.shape}")

    roi_traces = np.full((n_trials, n_rois, Nt), np.nan, dtype=np.float32)
    print(f"Initialized roi_traces shape: {roi_traces.shape}")

    # behavioral arrays
    is_right_trial = np.zeros(n_trials, dtype=bool)
    is_right_choice = np.zeros(n_trials, dtype=bool)
    rewarded = np.zeros(n_trials, dtype=bool)
    punished = np.zeros(n_trials, dtype=bool)
    did_not_choose = np.zeros(n_trials, dtype=bool)
    time_dnc = np.full(n_trials, np.nan, float)
    choice_start = np.full(n_trials, np.nan, float)
    choice_stop = np.full(n_trials, np.nan, float)
    servo_in = np.full(n_trials, np.nan, float)
    servo_out = np.full(n_trials, np.nan, float)
    lick = np.zeros(n_trials, dtype=bool)
    lick_start = np.full(n_trials, np.nan, float)
    RT = np.full(n_trials, np.nan, float)
    F1_on = np.full(n_trials, np.nan, float)
    F1_off = np.full(n_trials, np.nan, float)
    F2_on = np.full(n_trials, np.nan, float)
    F2_off = np.full(n_trials, np.nan, float)

    print("Processing individual trials...")
    for i, (_, row) in enumerate(df.iterrows()):
        f1off = float(row['end_flash_1'])
        t_vec = np.asarray(row['dff_time_vector'], float) - f1off
        dff   = np.asarray(row['dff_segment'], float)
        if not np.all(np.diff(t_vec) > 0):
            order = np.argsort(t_vec); t_vec = t_vec[order]; dff = dff[:, order]
        tmin, tmax = np.nanmin(t_vec), np.nanmax(t_vec)
        for r in range(n_rois):
            vals = np.interp(time, t_vec, dff[r], left=np.nan, right=np.nan)
            vals[(time < tmin) | (time > tmax)] = np.nan
            roi_traces[i, r] = vals

        is_right_trial[i]  = bool(row.get('is_right', False))
        is_right_choice[i] = bool(row.get('is_right_choice', False))
        rewarded[i]        = bool(row.get('rewarded', False))
        punished[i]        = bool(row.get('punished', False))
        did_not_choose[i]  = bool(row.get('did_not_choose', False))
        time_dnc[i]        = float(row.get('time_did_not_choose', np.nan))
        choice_start[i]    = float(row.get('choice_start', np.nan))
        choice_stop[i]     = float(row.get('choice_stop', np.nan))
        servo_in[i]        = float(row.get('servo_in', np.nan))
        servo_out[i]      = float(row.get('servo_out', np.nan))
        lick[i]            = bool(row.get('lick', False))
        lick_start[i]      = float(row.get('lick_start', np.nan))
        RT[i]              = float(row.get('RT', np.nan))
        F1_on[i]           = float(row.get('start_flash_1', np.nan))
        F1_off[i]          = float(row.get('end_flash_1', np.nan))
        F2_on[i]           = float(row.get('start_flash_2', np.nan))
        F2_off[i]          = float(row.get('end_flash_2', np.nan))

    
    # ---- SESSION short/long from trial_data ----
    si = trial_data.get('session_info', {})

    short_levels = _to_seconds(si.get('short_isis', []))
    long_levels  = _to_seconds(si.get('long_isis',  []))

    mean_isi = si.get('mean_isi', np.nan)
    if np.isfinite(mean_isi):
        mean_isi = float(_to_seconds([mean_isi])[0])  # scalar → seconds
    else:
        mean_isi = np.nan

    def _is_member(vals, levels, tol=1e-3):
        vals = np.asarray(vals, float).reshape(-1)
        levels = np.asarray(levels, float).reshape(-1)
        if levels.size == 0:
            return np.zeros_like(vals, dtype=bool)
        return (np.abs(vals[:, None] - levels[None, :]) <= tol).any(axis=1)

    # Primary label: membership in session-provided short set
    is_short = _is_member(isi_vec, short_levels)

    # Fallbacks (in case short_levels wasn't provided):
    if not np.any(is_short):
        if np.isfinite(mean_isi):                      # boundary present
            is_short = np.asarray(isi_vec <= mean_isi, dtype=bool)
        else:                                          # last resort: split unique ISIs
            boundary = np.median(isi_allowed)
            is_short = np.asarray(isi_vec <= boundary, dtype=bool)
    # ---------------------------------------------
    
    
    

    M = dict(
        roi_traces=roi_traces,
        time=time.astype(np.float32),
        isi=isi_vec.astype(np.float32),
        isi_allowed=isi_allowed.astype(np.float32),
        is_right=is_right_trial,
        is_right_choice=is_right_choice,
        rewarded=rewarded,
        punished=punished,
        did_not_choose=did_not_choose,
        time_did_not_choose=time_dnc,
        choice_start=choice_start,
        choice_stop=choice_stop,
        servo_in=servo_in,
        servo_out=servo_out,
        lick=lick,
        lick_start=lick_start,
        RT=RT,
        F1_on=F1_on, F1_off=F1_off, F2_on=F2_on, F2_off=F2_off,
        n_trials=n_trials, n_rois=n_rois,
        is_short=is_short.astype(bool),
        short_levels=np.asarray(short_levels, float),
        long_levels=np.asarray(long_levels, float),
        mean_isi=np.float32(mean_isi),
    )
    M['F2_time'] = M['isi']  # identical here
    return M

# ---------------- alignment helpers ----------------

def add_choice_locked_to_M(M: Dict[str, Any], trial_data: Dict[str, Any], window: Tuple[float,float] = (-0.6, 0.6), grid_bins: int = 240) -> Dict[str, Any]:
    df = trial_data['df_trials_with_segments']
    rows = list(df.iterrows())
    n_trials = len(rows); n_rois = int(M['n_rois'])
    tau = np.linspace(window[0], window[1], grid_bins)
    Xc = np.full((n_trials, n_rois, grid_bins), np.nan, dtype=np.float32)
    lick_rel = np.full(n_trials, np.nan, float)

    for i, (_, row) in enumerate(rows):
        cs = row.get('choice_start', row.get('servo_in', np.nan))
        cs = float(cs) if cs is not None else np.nan
        t_vec = np.asarray(row['dff_time_vector'], float) - cs
        dff   = np.asarray(row['dff_segment'], float)
        if not np.all(np.diff(t_vec) > 0):
            order = np.argsort(t_vec); t_vec = t_vec[order]; dff = dff[:, order]
        tmin, tmax = np.nanmin(t_vec), np.nanmax(t_vec)
        for r in range(n_rois):
            vals = np.interp(tau, t_vec, dff[r], left=np.nan, right=np.nan)
            vals[(tau < tmin) | (tau > tmax)] = np.nan
            Xc[i, r] = vals
        if row.get('lick_start') is not None and np.isfinite(cs):
            lick_rel[i] = float(row['lick_start']) - cs

    M['roi_traces_choice']   = Xc
    M['time_choice']         = tau.astype(np.float32)
    M['lick_start_relChoice']= lick_rel.astype(np.float32)
    return M


def add_F2_locked_to_M(M: Dict[str, Any], trial_data: Dict[str, Any], window: Tuple[float,float]=(-0.2,0.6), grid_bins: int = 160) -> Dict[str, Any]:
    df = trial_data['df_trials_with_segments']
    rows = list(df.iterrows())
    n_trials = len(rows); n_rois = int(M['n_rois'])
    tau = np.linspace(window[0], window[1], grid_bins)
    X = np.full((n_trials, n_rois, grid_bins), np.nan, dtype=np.float32)
    choice_rel = np.full(n_trials, np.nan, float)

    lick_rel = np.full(n_trials, np.nan, float)
    # inside the loop

    for i, (_, row) in enumerate(rows):
        f2 = float(row['start_flash_2'])
        t_vec = np.asarray(row['dff_time_vector'], float) - f2
        dff   = np.asarray(row['dff_segment'], float)
        if not np.all(np.diff(t_vec) > 0):
            order = np.argsort(t_vec); t_vec = t_vec[order]; dff = dff[:, order]
        tmin, tmax = np.nanmin(t_vec), np.nanmax(t_vec)
        for r in range(n_rois):
            vals = np.interp(tau, t_vec, dff[r], left=np.nan, right=np.nan)
            vals[(tau < tmin) | (tau > tmax)] = np.nan
            X[i, r] = vals
        cs = row.get('choice_start', row.get('servo_in', np.nan))
        if cs is not None and np.isfinite(f2):
            try:
                choice_rel[i] = float(cs) - f2
            except Exception:
                pass
        if row.get('lick_start') is not None and np.isfinite(f2):
            lick_rel[i] = float(row['lick_start']) - f2
    M['roi_traces_F2locked'] = X
    M['time_F2locked'] = tau.astype(np.float32)
    M['choice_start_relF2'] = choice_rel.astype(np.float32)
    M['lick_start_relF2'] = lick_rel.astype(np.float32)
    return M

# ---------------------- analyses ----------------------

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
        print("Insufficient valid predictions for R2 calculation")
        return np.nan
        
    r2 = r2_score(y_all[ok], preds[ok])
    print(f"Final R2 calculated on {ok_count} points: {r2:.4f}")
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

# def scaling_vs_clock_cv(M: Dict[str, Any], preF2_window=None, n_bases=9, alpha=1e-2, n_splits=5) -> Dict[str, Any]:
#     X, T, isi = M['roi_traces'], M['time'], M['isi']
#     if preF2_window is None:
#         preF2_window = auto_preF2_window(M)
#     t0, t1 = preF2_window
#     sel = (T >= t0) & (T <= t1)
#     tt = T[sel]
#     if tt.size < 8:
#         raise ValueError(f"preF2 window too small: {preF2_window}")
#     Bt = _rbf_basis(tt, n_bases, tt[0], tt[-1])

#     from sklearn.model_selection import KFold
#     kf = KFold(n_splits=min(n_splits, X.shape[0]), shuffle=True, random_state=0)

#     Ntr, Nroi = X.shape[0], X.shape[1]
#     r2_phase = np.full(Nroi, np.nan); r2_time = np.full(Nroi, np.nan)

#     for r in range(Nroi):
#         yp_true=[]; yp_pred=[]; yt_true=[]; yt_pred=[]
#         for tr_idx, te_idx in kf.split(np.arange(Ntr)):
#             A_phase=[]; y_phase=[]; A_time=[]; y_time=[]
#             for ii in tr_idx:
#                 ph = (tt / (isi[ii] + 1e-9)).clip(0, 1)
#                 Bp = _rbf_basis(ph, n_bases, 0.0, 1.0)
#                 A_phase.append(Bp); A_time.append(Bt)
#                 y = X[ii, r, sel]; y_phase.append(y); y_time.append(y)
#             A_phase = np.vstack(A_phase); y_phase = np.hstack(y_phase)
#             A_time  = np.vstack(A_time);  y_time  = np.hstack(y_time)
#             m1 = np.isfinite(y_phase) & np.all(np.isfinite(A_phase), axis=1)
#             m2 = np.isfinite(y_time)  & np.all(np.isfinite(A_time),  axis=1)
#             if m1.sum() < 30 or m2.sum() < 30: continue
#             for ii in te_idx:
#                 ph = (tt / (isi[ii] + 1e-9)).clip(0, 1)
#                 Bp = _rbf_basis(ph, n_bases, 0.0, 1.0)
#                 y_true = X[ii, r, sel]
#                 yp_true.append(y_true)
#                 yp_pred.append(_ridge_fit_predict(A_phase[m1], y_phase[m1], Bp, alpha))
#                 yt_true.append(y_true)
#                 yt_pred.append(_ridge_fit_predict(A_time[m2],  y_time[m2],  Bt, alpha))
#         if yp_true:
#             r2_phase[r] = r2_nan(np.concatenate(yp_true), np.concatenate(yp_pred))
#             r2_time[r]  = r2_nan(np.concatenate(yt_true), np.concatenate(yt_pred))
#     return dict(delta_r2=r2_phase - r2_time, r2_phase=r2_phase, r2_time=r2_time, window=preF2_window)

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


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
    print(f"  R2 range: {np.nanmin(r2_t):.4f} to {np.nanmax(r2_t):.4f}")
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





# def time_resolved_isi_decode(M: Dict[str, Any], preF2_window=None, n_bins: int = 24) -> Dict[str, Any]:
#     X, T, isi = M['roi_traces'], M['time'], M['isi']
#     if preF2_window is None:
#         preF2_window = auto_preF2_window(M)
#     t0, t1 = preF2_window
#     # mwin = (T >= t0) & (T <= t1)
#     # tt = T[mwin]
#     # edges = np.linspace(tt[0], tt[-1], n_bins + 1)
#     # centers = 0.5 * (edges[:-1] + edges[1:])
#     # AFTER (full time course like the original)
#     edges   = np.linspace(T[0], T[-1], n_bins + 1)
#     centers = 0.5 * (edges[:-1] + edges[1:])


#     boundary = np.median(M['isi_allowed'])
#     # y_bin = (isi >= boundary).astype(int)
#     y_bin = M['is_short'].astype(int)   # 1 = short, 0 = long


#     r2_t = np.full(len(centers), np.nan); auc_t = np.full(len(centers), np.nan)

#     for b in range(len(centers)):
#         m = (T >= edges[b]) & (T < edges[b+1])
#         Z = np.nanmean(X[:, :, m], axis=2)
#         valid = np.all(np.isfinite(Z), axis=1)
#         if valid.sum() < 16: continue
#         Z = Z[valid]; y_reg = isi[valid]; y_clf = y_bin[valid]
#         kf = KFold(n_splits=min(5, Z.shape[0]), shuffle=True, random_state=0)
#         preds=[]; truth=[]; scores=[]
#         for tr, te in kf.split(Z):
#             Ztr, Zte = Z[tr], Z[te]
#             sc = StandardScaler().fit(Ztr); Ztr = sc.transform(Ztr); Zte = sc.transform(Zte)
#             r = Ridge(alpha=1.0).fit(Ztr, y_reg[tr])
#             preds.append(r.predict(Zte)); truth.append(y_reg[te])
#             clf = LogisticRegression(penalty='l2', solver='liblinear', C=1.0).fit(Ztr, y_clf[tr])
#             if np.unique(y_clf[te]).size >= 2:
#                 scores.append(roc_auc_score(y_clf[te], clf.predict_proba(Zte)[:,1]))
#         if preds:
#             r2_t[b] = r2_nan(np.concatenate(truth), np.concatenate(preds))
#         auc_t[b] = float(np.nanmean(scores)) if scores else np.nan
#     return dict(time_points=centers, isi_r2=r2_t, auc_shortlong=auc_t)

# # fair‑window decoders

def fair_window_decoders(M, preF2_window=None, win=0.12, step=0.03):
    X, T, isi = M['roi_traces'], M['time'], M['isi']
    if preF2_window is None:
        preF2_window = auto_preF2_window(M)
    t0, t1 = preF2_window
    centers=[]; amp_r2=[]; amp_auc=[]; slope_r2=[]; slope_auc=[]
    boundary = np.median(M['isi_allowed'])
    # y_bin = (isi >= boundary).astype(int)
    y_bin = M['is_short'].astype(int)
    t = T[(T>=t0)&(T<=t1)]; c = t0 + win/2.0
    while c + win/2.0 <= t1:
        m = (T >= (c - win/2.0)) & (T < (c + win/2.0))
        if m.sum() >= 4:
            A = np.nanmean(X[:, :, m], axis=2)  # amplitude
            tt = T[m] - T[m].mean(); denom = np.sum(tt**2) + 1e-9
            B = np.tensordot(X[:, :, m], tt, axes=(2,0)) / denom  # slope
            for F, r2_list, auc_list in [(A, amp_r2, amp_auc), (B, slope_r2, slope_auc)]:
                valid = np.all(np.isfinite(F), axis=1)
                if valid.sum() < 16:
                    r2_list.append(np.nan); auc_list.append(np.nan); continue
                Z = F[valid]; y_reg = isi[valid]; y_clf = y_bin[valid]
                kf = KFold(n_splits=min(5, Z.shape[0]), shuffle=True, random_state=0)
                preds=[]; truth=[]; scores=[]
                for tr, te in kf.split(Z):
                    Ztr, Zte = Z[tr], Z[te]
                    sc = StandardScaler().fit(Ztr); Ztr = sc.transform(Ztr); Zte = sc.transform(Zte)
                    r = Ridge(alpha=1.0).fit(Ztr, y_reg[tr])
                    preds.append(r.predict(Zte)); truth.append(y_reg[te])
                    clf = LogisticRegression(penalty='l2', solver='liblinear', C=1.0).fit(Ztr, y_clf[tr])
                    if np.unique(y_clf[te]).size >= 2:
                        scores.append(roc_auc_score(y_clf[te], clf.predict_proba(Zte)[:,1]))
                r2_list.append(r2_nan(np.concatenate(truth), np.concatenate(preds)) if preds else np.nan)
                auc_list.append(float(np.nanmean(scores)) if scores else np.nan)
            centers.append(c)
        c += step
    return dict(time_points=np.asarray(centers), amp_r2=np.asarray(amp_r2), amp_auc=np.asarray(amp_auc), slope_r2=np.asarray(slope_r2), slope_auc=np.asarray(slope_auc))

# trough vs ISI

def trough_time_vs_isi(M):
    X, T, levels, isi = M['roi_traces'], M['time'], np.unique(M['isi_allowed']), M['isi']
    trough_t = []
    for lv in levels:
        sel = np.isfinite(isi) & (np.abs(isi - lv) < 1e-6)
        if sel.sum() < 5: trough_t.append(np.nan); continue
        Y = nanmean_no_warn(X[sel], axis=0)  # (Nroi,Nt)
        y = nanmean_no_warn(Y, axis=0)
        idx = np.nanargmin(y); trough_t.append(T[idx])
    trough_t = np.asarray(trough_t, float)
    # linear fit
    x = levels; y = trough_t
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 3:
        A = np.column_stack([np.ones(m.sum()), x[m]])
        beta = np.linalg.lstsq(A, y[m], rcond=None)[0]
        yhat = A @ beta
        ss_res = np.nansum((y[m] - yhat)**2); ss_tot = np.nansum((y[m]-np.nanmean(y[m]))**2)
        r2_lin = 1 - ss_res / (ss_tot + 1e-9)
    else:
        r2_lin = np.nan
    return dict(levels=levels, trough_time=trough_t, r2_linear=r2_lin)

# hazard unique variance

# def hazard_unique_variance(M, sigma=0.06, preF2_window=None, n_splits=5):
#     X, T = M['roi_traces'], M['time']
#     if preF2_window is None:
#         preF2_window = (0.15, 0.70)  # the window you used in the original
#     mwin = (T >= preF2_window[0]) & (T <= preF2_window[1])
#     tt = T[mwin]

#     levels = np.asarray(M['isi_allowed'])
#     H = np.zeros_like(tt)
#     for a in levels:
#         H += np.exp(-0.5 * ((tt - a) / max(sigma, 1e-3)) ** 2)
#     H = (H - H.min()) / (H.max() - H.min() + 1e-9)
#     R = (tt - tt.min()) / (tt.max() - tt.min() + 1e-9)

#     from sklearn.model_selection import KFold
#     kf = KFold(n_splits=min(n_splits, max(3, tt.size//10)), shuffle=True, random_state=0)

#     Nroi = X.shape[1]
#     uv = np.full(Nroi, np.nan)

#     # session-average per ROI, then CV along time points
#     Y = np.nanmean(X, axis=0)[:, mwin]  # (Nroi, Nt_sel)

#     for r in range(Nroi):
#         y = Y[r]
#         if not np.any(np.isfinite(y)): 
#             continue
#         r2_fulls, r2_times = [], []
#         for tr_idx, te_idx in kf.split(np.arange(tt.size)):
#             Af = np.column_stack([np.ones(tr_idx.size), R[tr_idx], H[tr_idx]])
#             At = np.column_stack([np.ones(tr_idx.size), R[tr_idx]])
#             ytr = y[tr_idx]
#             if np.sum(np.isfinite(ytr)) < 10: 
#                 continue
#             # fit on train
#             bf = np.linalg.lstsq(Af, ytr, rcond=None)[0]
#             bt = np.linalg.lstsq(At, ytr, rcond=None)[0]
#             # predict on test
#             Af_te = np.column_stack([np.ones(te_idx.size), R[te_idx], H[te_idx]])
#             At_te = np.column_stack([np.ones(te_idx.size), R[te_idx]])
#             yte   = y[te_idx]
#             ypf   = Af_te @ bf
#             ypt   = At_te @ bt
#             r2_fulls.append(r2_nan(yte, ypf))
#             r2_times.append(r2_nan(yte, ypt))
#         if r2_fulls and r2_times:
#             uv[r] = max(0.0, np.nanmean(r2_fulls) - np.nanmean(r2_times))
#     return dict(uv_per_roi=uv,
#                 session_uv=float(np.nanmean(uv)),
#                 frac_positive=float(np.mean(uv > 0)))


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

# def hazard_unique_variance(M, sigma=0.06, preF2_window=None):
#     X, T = M['roi_traces'], M['time']
#     if preF2_window is None: 
#         preF2_window = auto_preF2_window(M)
#     mwin = (T >= preF2_window[0]) & (T <= preF2_window[1])
#     tt = T[mwin]; levels = np.asarray(M['isi_allowed'])
#     H = np.zeros_like(tt)
#     for a in levels:
#         H += np.exp(-0.5 * ((tt - a) / max(sigma, 1e-3)) ** 2)
#     H = (H - H.min()) / (H.max() - H.min() + 1e-9)
#     R = (tt - tt.min()) / (tt.max() - tt.min() + 1e-9)

#     Nroi = X.shape[1]
#     uv = np.full(Nroi, np.nan)
#     y = nanmean_no_warn(X, axis=0)[:, mwin]  # session average per ROI
#     for r in range(Nroi):
#         yy = y[r]
#         A = np.column_stack([np.ones_like(tt), R, H]); m = np.all(np.isfinite(A), axis=1) & np.isfinite(yy)
#         if m.sum() < 10: continue
#         beta = np.linalg.lstsq(A[m], yy[m], rcond=None)[0]
#         r2_full = r2_nan(yy[m], A[m] @ beta)
#         A0 = np.column_stack([np.ones_like(tt), R]); beta0 = np.linalg.lstsq(A0[m], yy[m], rcond=None)[0]
#         r2_time = r2_nan(yy[m], A0[m] @ beta0)
#         uv[r] = max(0.0, r2_full - r2_time)
#     return dict(uv_per_roi=uv, session_uv=float(np.nanmean(uv)), frac_positive=float(np.mean(uv > 0)))

# RSA


# -----------------------------
# 3) RSA: phase vs ms alignment
# -----------------------------

def rsa_phase_vs_time(M: Dict[str, Any], n_phase_bins: int = 20) -> Dict[str, Any]:
    """Representational similarity analysis comparing absolute-time vs phase alignment.

    For each ISI level, compute the mean ROI×time matrix (pre‑F2).
    - MS alignment: for each pair of ISIs, correlate ROI×time vectors using only time < min(ISI_i, ISI_j).
    - Phase alignment: resample each ISI’s ROI×time to fixed phase bins (0..1), then correlate.

    Returns: mean off‑diagonal correlation for MS and PHASE, plus the matrices.
    """
    X = M['roi_traces']
    T = M['time']
    F2 = M['F2_time']
    Ntr, Nroi, Nt = X.shape

    levels = np.sort(np.unique(F2[np.isfinite(F2)]))
    K = len(levels)

    # Mean per ISI level
    mean_by_level = []  # list of (Nroi, Nt)
    for isi in levels:
        sel = np.isclose(F2, isi, rtol=0, atol=1e-6)
        avg = np.nanmean(X[sel], axis=0)
        mean_by_level.append(avg)

    # Absolute‑time RSA
    C_ms = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(K):
            if i == j:
                C_ms[i, j] = 1.0
                continue
            tmax = min(levels[i], levels[j])
            valid = (T >= 0) & (T < tmax)
            if valid.sum() < 5:
                continue
            A = mean_by_level[i][:, valid].reshape(-1)
            B = mean_by_level[j][:, valid].reshape(-1)
            if np.all(np.isfinite(A)) and np.all(np.isfinite(B)):
                a = (A - A.mean()) / (A.std() + 1e-9)
                b = (B - B.mean()) / (B.std() + 1e-9)
                C_ms[i, j] = float(np.nanmean(a * b))

    # Phase RSA
    # Build phase grid and resample each ISI’s mean traces across phase
    phase_grid = np.linspace(0, 1, n_phase_bins)
    mean_by_phase = []  # list of (Nroi, n_phase_bins)
    for k, isi in enumerate(levels):
        valid = (T >= 0) & (T < isi)
        if valid.sum() < 5:
            mean_by_phase.append(np.full((Nroi, n_phase_bins), np.nan))
            continue
        # Map absolute time to phase indices
        t_rel = (T[valid] / max(isi, 1e-9)).clip(0, 1)
        # For each phase bin, average nearest times
        A = np.full((Nroi, n_phase_bins), np.nan)
        for b in range(n_phase_bins - 1):
            m = (t_rel >= phase_grid[b]) & (t_rel < phase_grid[b + 1])
            if m.sum() == 0:
                continue
            A[:, b] = np.nanmean(mean_by_level[k][:, valid][:, m], axis=1)
        # Last bin include the endpoint
        A[:, -1] = np.nanmean(mean_by_level[k][:, valid][:, t_rel >= phase_grid[-1]], axis=1)
        mean_by_phase.append(A)

    C_ph = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(K):
            if i == j:
                C_ph[i, j] = 1.0
                continue
            A = mean_by_phase[i].reshape(-1)
            B = mean_by_phase[j].reshape(-1)
            if np.all(np.isfinite(A)) and np.all(np.isfinite(B)):
                a = (A - A.mean()) / (A.std() + 1e-9)
                b = (B - B.mean()) / (B.std() + 1e-9)
                C_ph[i, j] = float(np.nanmean(a * b))

    # Mean off‑diagonal correlation
    off = ~np.eye(K, dtype=bool)
    mean_ms = float(np.nanmean(C_ms[off]))
    mean_ph = float(np.nanmean(C_ph[off]))

    return {
        'levels': levels,
        'corr_ms': C_ms,
        'corr_phase': C_ph,
        'mean_corr_ms': mean_ms,
        'mean_corr_phase': mean_ph,
    }



# def rsa_abs_vs_phase(M, preF2_window=None, n_phase_bins=20):
#     X, T, isi = M['roi_traces'], M['time'], M['isi']
#     if preF2_window is None: 
#         preF2_window = auto_preF2_window(M)
#     mwin = (T >= preF2_window[0]) & (T <= preF2_window[1])
#     tt = T[mwin]; levels = np.unique(M['isi_allowed'])
#     per_isi = {}
#     for lv in levels:
#         sel = np.isfinite(isi) & (np.abs(isi - lv) < 1e-6)
#         if sel.sum() < 5: continue
#         Y = nanmean_no_warn(X[sel], axis=0)
#         per_isi[lv] = Y[:, mwin]
#     keys = sorted(per_isi.keys())
#     if len(keys) < 3:
#         return dict(abs_time=np.nan, phase=np.nan)

#     def _mean_offdiag_corr(stack):
#         nI, _, Nt = stack.shape
#         vals = []
#         for t in range(Nt):
#             mat = np.corrcoef(stack[:, :, t])
#             if np.any(np.isnan(mat)): continue
#             off = mat[~np.eye(nI, dtype=bool)]
#             vals.append(np.nanmean(off))
#         return float(np.nanmean(vals)) if vals else np.nan

#     arr_abs = np.stack([per_isi[k] for k in keys], axis=0)
#     abs_time_score = _mean_offdiag_corr(arr_abs)

#     phase_grid = np.linspace(0, 1, n_phase_bins)
#     arr_phase = []
#     for k in keys:
#         Y = per_isi[k]
#         ph = (tt / (k + 1e-9)).clip(0, 1)
#         Yp = np.empty((Y.shape[0], n_phase_bins)); Yp[:] = np.nan
#         for r in range(Y.shape[0]):
#             Yp[r] = np.interp(phase_grid, ph, Y[r], left=np.nan, right=np.nan)
#         arr_phase.append(Yp)
#     arr_phase = np.stack(arr_phase, axis=0)
#     phase_score = _mean_offdiag_corr(arr_phase)
#     return dict(abs_time=abs_time_score, phase=phase_score)

# choice decoders

def decode_choice_F2locked_v2(M: Dict[str, Any], n_bins=24, t_window=(0.0, 0.6), baseline_window=(-0.2, -0.05), zscore_within_fold=True, stratify_by_isi=True):
    X = M.get('roi_traces_F2locked'); T = M.get('time_F2locked')
    side = M.get('is_right_choice'); isi = M.get('F2_time')
    if X is None or T is None or side is None: return None
    X = X.copy(); side = side.astype(int)
    if baseline_window is not None:
        bm = (T >= baseline_window[0]) & (T <= baseline_window[1])
        if np.any(bm):
            X -= np.nanmean(X[:, :, bm], axis=2, keepdims=True)
    mwin = (T >= t_window[0]) & (T <= t_window[1])
    T = T[mwin]; X = X[:, :, mwin]
    edges = np.linspace(T[0], T[-1], n_bins + 1); centers = 0.5 * (edges[:-1] + edges[1:])
    auc = np.full(len(centers), np.nan)

    from sklearn.model_selection import KFold, StratifiedKFold

    for b in range(len(centers)):
        m = (T >= edges[b]) & (T < edges[b+1])
        if not np.any(m): continue
        Z_full = np.nanmean(X[:, :, m], axis=2)
        valid = np.where(np.all(np.isfinite(Z_full), axis=1))[0]
        if valid.size < 12: continue
        yv = side[valid]
        if np.unique(yv).size < 2: continue
        if stratify_by_isi and isi is not None:
            # levels = np.unique(isi[np.isfinite(isi)])
            levels = np.unique(M['isi_allowed'])
            lvl = np.searchsorted(levels, np.asarray(isi)[valid])
            y_joint = yv * max(1, levels.size) + lvl       
            splitter = StratifiedKFold(n_splits=min(5, len(valid)), shuffle=True, random_state=0)
            folds = splitter.split(np.zeros_like(y_joint), y_joint)
        else:
            splitter = KFold(n_splits=min(5, len(valid)), shuffle=True, random_state=0)
            folds = splitter.split(np.arange(len(valid)))
        Z = Z_full[valid]
        scores = []
        for tr_rel, te_rel in folds:
            if tr_rel.size < 10 or te_rel.size < 4: continue
            Ztr, Zte = Z[tr_rel], Z[te_rel]
            ytr, yte = yv[tr_rel], yv[te_rel]
            if zscore_within_fold:
                sc = StandardScaler().fit(Ztr); Ztr = sc.transform(Ztr); Zte = sc.transform(Zte)
            clf = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, max_iter=1000)
            clf.fit(Ztr, ytr)
            if np.unique(yte).size < 2: continue
            p = clf.predict_proba(Zte)[:, 1]
            scores.append(roc_auc_score(yte, p))
        auc[b] = float(np.nanmean(scores)) if scores else np.nan
    return {'time_points': centers, 'auc_t': auc}

# ---------------------- run bundles ----------------------

def run_timing_analysis(M: Dict[str, Any], preF2_window=None) -> Dict[str, Any]:
    # model = scaling_vs_clock(M, preF2_window=preF2_window)
    # decode = time_resolved_decoding(M, preF2_window=preF2_window)
    # fair = fair_window_decoders(M, preF2_window=preF2_window)
    # trough = trough_time_vs_isi(M)
    # hz = hazard_unique_variance(M, preF2_window=preF2_window)
    # rsa = rsa_phase_vs_time(M, preF2_window=preF2_window)
    model = scaling_vs_clock(M)
    decode = time_resolved_decoding(M)
    fair = fair_window_decoders(M)
    trough = trough_time_vs_isi(M)
    hz = hazard_unique_variance(M)
    rsa = rsa_phase_vs_time(M)    
    return dict(model=model, decode=decode, fair=fair, trough=trough, hazard=hz, rsa=rsa)

# ---------------------- plotting ----------------------

def plot_session_report_v1(M: Dict[str, Any], T: Dict[str, Any], choice_auc_F2: Dict[str,Any] | None = None, session_title: str | None = None, save_path: str | None = None):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1,1,1.5], wspace=0.35, hspace=0.45)

    # [1] ΔR² histogram (robust)
    ax1 = fig.add_subplot(gs[0,0])
    delta = np.asarray(T['model']['delta_r2'], float)
    finite = np.isfinite(delta)
    if finite.sum() >= 2:
        lo = float(np.nanmin(delta[finite])); hi = float(np.nanmax(delta[finite]))
        if lo == hi:
            eps = max(1e-4, 0.1 * max(1e-3, abs(lo))); lo, hi = lo - eps, hi + eps
        bins = np.linspace(lo, hi, 31)
        ax1.hist(delta[finite], bins=bins, color='#4682b4', edgecolor='none')
        ax1.axvline(0, color='k', ls='--', lw=1)
        med = float(np.nanmedian(delta)); ax1.axvline(med, color='k', lw=1.5)
        n = delta.size
        clock_cnt = int(np.sum(delta < 0)); mixed_cnt = int(np.sum(np.abs(delta) <= 0.02))
        ax1.set_title('Scaling vs Clock')
        ax1.text(0.98, 0.95, f"median ΔR²={med:.3f}\nclock: {clock_cnt}/{n}\nmixed: {mixed_cnt}/{n}", transform=ax1.transAxes, ha='right', va='top', fontsize=8)
        ax1.set_xlabel('ΔR² (phase - ms)'); ax1.set_ylabel('ROIs')
    else:
        ax1.text(0.5,0.5,'ΔR² unavailable',ha='center',va='center'); ax1.set_title('Scaling vs Clock')

    # [2] time‑resolved decode
    ax2 = fig.add_subplot(gs[0,1])
    dec = T['decode']; tt = dec['time_points']
    # ax2.plot(tt, dec['isi_r2'], lw=2, label='ISI R²(t)')
    # ax2.set_ylabel('ISI R²')
    ax2.plot(tt, dec['isi_r2']*100.0, lw=2, label='ISI R²(t)')
    ax2.set_ylabel('ISI R² (%, t)')

    ax2b = ax2.twinx(); ax2b.plot(tt, dec['auc_shortlong'], lw=2, ls='--', label='Short/Long AUC', color='tab:orange')
    ax2.axhline(0, color='k', ls=':', lw=1); ax2b.axhline(0.5, color='k', ls=':', lw=1)
    ax2.set_title('Time-resolved decode'); ax2.set_xlabel('Time from F1-OFF (s)'); ax2b.set_ylabel('AUC')

    # [3] Hazard UV
    ax3 = fig.add_subplot(gs[0,2])
    hz = T['hazard']; ax3.bar(['Session UV','Frac ROIs>0'], [hz['session_uv'], hz['frac_positive']])
    ax3.set_title('Anticipation beyond ramp')


    # [4] Fair‑window decoders
    ax4 = fig.add_subplot(gs[1,0])
    fw = T.get('fair', None)
    if fw is not None and fw['time_points'].size:
        tt = fw['time_points']
        ax4.plot(tt, fw['amp_r2'], lw=1.8, label='amp ISI R²')
        ax4b = ax4.twinx(); ax4b.plot(tt, fw['slope_auc'], lw=1.8, ls='--', color='tab:orange', label='slope AUC')
        ax4.axhline(0, color='k', ls=':', lw=1); ax4b.axhline(0.5, color='k', ls=':', lw=1)
        ax4.set_title('Fair-window decoders (amp vs slope)'); ax4.set_xlabel('Time from F1-OFF (s)'); ax4.set_ylabel('ISI R²'); ax4b.set_ylabel('AUC')
    else:
        ax4.text(0.5,0.5,'fair-window N/A',ha='center',va='center'); ax4.set_title('Fair-window decoders')


    # [5] RSA summary
    ax5 = fig.add_subplot(gs[1,1])
    rs = T['rsa']; ax5.bar(['Abs time','Phase'], [rs['abs_time'], rs['phase']]); ax5.set_ylim(0,1)
    ax5.set_title('RSA mean off-diagonal corr')

    # [6] F2‑locked choice decode
    ax6 = fig.add_subplot(gs[1,2])
    if choice_auc_F2 is not None:
        t2, a2 = choice_auc_F2['time_points'], choice_auc_F2['auc_t']
        ax6.plot(t2, a2, lw=2)
        ax6.axhline(0.5, color='k', ls='--', lw=1)
        # optional first lick overlay
        try:
            c_med = np.nanmedian(M.get('choice_start_relF2'))
            l_med = np.nanmedian(M.get('lick_start_relF2'))
            if np.isfinite(c_med): 
                ax6.axvline(c_med, color='C2', lw=1, ls='--')
            if np.isfinite(l_med): 
                ax6.axvline(l_med, color='C3', lw=1, ls='--')
            
        except Exception: pass
        ax6.set_title('F2-locked choice decode')
        ax6.set_xlabel('Time from F2 (s)') 
        ax6.set_ylabel('AUC(τ)')
    else:
        ax6.text(0.5,0.5,'F2-locked choice AUC N/A',ha='center',va='center')
        ax6.set_axis_off()
        
    # Add this block (e.g., gs[1,0]) if it was removed:
    ax_tr = fig.add_subplot(gs[2,0])
    tr = T['trough']
    ax_tr.plot(tr['levels'], tr['trough_time'], 'o-', lw=2)
    ax_tr.axhline(np.nanmedian(tr['trough_time']), color='C2', ls=':', lw=1)
    ax_tr.text(0.02, 0.04, f"linear R²={tr['r2_linear']:.2f}", transform=ax_tr.transAxes)
    ax_tr.set_xlabel('ISI (s)'); ax_tr.set_ylabel('Trough time (s, F1-OFF)')
    ax_tr.set_title('Trough/turning-point vs ISI')        

    # [7/8] Population raster
    ax8 = fig.add_subplot(gs[2,1:])
    Y = nanmean_no_warn(M['roi_traces'], axis=0)  # (Nroi,Nt)
    order = np.argsort(np.nanargmin(Y, axis=1))
    im = ax8.imshow(Y[order], aspect='auto', extent=[M['time'][0], M['time'][-1], 0, Y.shape[0]], cmap='viridis', vmin=np.nanpercentile(Y, 2), vmax=np.nanpercentile(Y, 98))
    ax8.set_title('Population raster (avg across trials)')
    ax8.set_xlabel('Time from F1-OFF (s)'); ax8.set_ylabel('ROIs (sorted)')
    plt.colorbar(im, ax=ax8, fraction=0.02, pad=0.01, label='ΔF/F (a.u.)')

    if session_title: fig.suptitle(session_title, y=0.98, fontsize=11)
    if save_path: fig.savefig(save_path, dpi=180, bbox_inches='tight')
    return fig

# ---------------------- group mean traces ----------------------

def build_group_masks(M: Dict[str, Any], T: Dict[str, Any], signal_quantile=5, delta_eps=0.02, hazard_thresh=0.0):
    masks = {}
    delta = T['model']['delta_r2']
    masks['clock'] = np.isfinite(delta) & (delta < -delta_eps)
    masks['scaling'] = np.isfinite(delta) & (delta >  delta_eps)
    masks['mixed'] = np.isfinite(delta) & ~(masks['clock'] | masks['scaling'])
    masks['signal']  = masks['clock']        # <- THIS matches the original IMG4 groups

    
    # signal: strongest negative trough in pre‑F2 window (top q%)
    pre = (M['time'] >= 0.15) & (M['time'] <= min(np.nanmin(M['isi'])*0.85, M['time'][-1]))
    Y = nanmean_no_warn(M['roi_traces'], axis=0)  # (Nroi,Nt)
    minval = np.nanmin(Y[:, pre], axis=1)
    thr = np.nanpercentile(minval, signal_quantile)
    masks['signal'] = minval <= thr
    # hazard positive
    uv = T.get('hazard',{}).get('uv_per_roi', None)
    if uv is not None:
        masks['hazard_pos'] = np.isfinite(uv) & (uv > hazard_thresh)
    return masks


# ---- ROI grouping + group trace plotting ----

def get_roi_group_masks(M: dict, R: dict, extras: dict | None = None,
                        eps: float = 0.02,
                        choice_window: tuple[float, float] = (0.05, 0.25),
                        choice_es_thresh: float = 0.5,
                        hazard_thresh: float = 0.0) -> dict:
    """Return boolean masks for several ROI groups to enable behavioral filtering.

    Groups returned:
      - 'scaling', 'clock', 'mixed' from model_pref in R (threshold eps on ΔR²)
      - 'signal' := scaling ∪ clock (|ΔR²| > eps)
      - 'hazard_pos' from extras['hz_roi']['partial_r2'] > hazard_thresh (if available)
      - 'choice_mod' : post‑F2 side effect (Cohen's d > choice_es_thresh in [0.05, 0.25]s)

    Notes:
      * 'choice_mod' requires M['roi_traces_F2locked'], M['time_F2locked'], and M['is_right_choice'].
      * Masks default to False where info is unavailable.
    """
    Nroi = M['roi_traces'].shape[1]
    masks = {k: np.zeros(Nroi, dtype=bool) for k in ['scaling','clock','mixed','signal','hazard_pos','choice_mod']}

    # From ΔR²
    delta = np.asarray(R.get('delta_r2'))
    if delta.size == Nroi:
        masks['scaling'] = delta >  eps
        masks['clock']   = delta < -eps
        masks['mixed']   = ~np.isfinite(delta) | (np.abs(delta) <= eps)
        masks['signal']  = np.isfinite(delta) & (np.abs(delta) > eps)

    # Hazard per‑ROI (if provided)
    if isinstance(extras, dict) and extras.get('hz_roi') is not None:
        pr2 = np.asarray(extras['hz_roi'].get('partial_r2'))
        if pr2.size == Nroi:
            masks['hazard_pos'] = np.isfinite(pr2) & (pr2 > hazard_thresh)

    # Choice‑modulated post‑F2 (requires F2‑locked arrays)
    Xf2 = M.get('roi_traces_F2locked'); T2 = M.get('time_F2locked'); side = M.get('is_right_choice')
    if isinstance(Xf2, np.ndarray) and isinstance(T2, np.ndarray) and side is not None:
        w = (T2 >= choice_window[0]) & (T2 <= choice_window[1])
        if np.any(w):
            # Per‑ROI trial features in the window
            feat = np.nanmean(Xf2[:, :, w], axis=2)  # (Ntr, Nroi)
            side = side.astype(bool)
            if np.unique(side).size == 2:
                a = feat[ side]
                b = feat[~side]
                # Cohen's d per ROI
                ma = np.nanmean(a, axis=0); mb = np.nanmean(b, axis=0)
                va = np.nanvar(a, axis=0, ddof=1); vb = np.nanvar(b, axis=0, ddof=1)
                na = np.sum(np.isfinite(a), axis=0); nb = np.sum(np.isfinite(b), axis=0)
                # pooled sd
                sp = np.sqrt(((na-1)*va + (nb-1)*vb) / np.maximum(na+nb-2, 1))
                d = (ma - mb) / (sp + 1e-9)
                ok = (na >= 8) & (nb >= 8) & np.isfinite(d)
                masks['choice_mod'] = ok & (np.abs(d) > choice_es_thresh)
    return masks


def plot_roi_group_traces(
    M, R, extras=None, groups=('signal','mixed'), align='F1OFF',
    trial_filter=('licked','rewarded','no_dnc'),   # <- matches original figures
    equalize_isi=True,                             # <- key difference
    f2_baseline_window=(-0.20, 0.0),               # viz only
    choice_baseline_window=(-0.60, -0.20),         # viz only
    use_median=False, smooth_sigma_s=None, max_spaghetti=10,
    colors=None, title=None
):
    import numpy as np, matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    masks = get_roi_group_masks(M, R, extras)
    if colors is None:
        colors = {'signal':'C0', 'mixed':'C1', 'scaling':'C2', 'clock':'C3', 'hazard_pos':'C4', 'choice_mod':'C5'}

    # pick alignment
    if align.upper() == 'F2':
        X, T = M.get('roi_traces_F2locked'), M.get('time_F2locked'); xlabel='Time from F2 (s)'
        base_window = f2_baseline_window
    elif align.upper() == 'CHOICE':
        X, T = M.get('roi_traces_choice'), M.get('time_choice'); xlabel='Time from choice start (s)'
        base_window = choice_baseline_window
    else:
        X, T = M['roi_traces'], M['time']; xlabel='Time from F1-OFF (s)'; base_window=None
    assert X is not None and T is not None

    # build trial keep mask
    keep = np.ones(X.shape[0], bool)
    if trial_filter:
        if 'licked' in trial_filter:    keep &= M.get('lick', np.zeros_like(keep, bool))
        if 'rewarded' in trial_filter:  keep &= M.get('rewarded', np.zeros_like(keep, bool))
        if 'no_dnc' in trial_filter:    keep &= ~M.get('did_not_choose', np.zeros_like(keep, bool))
    X = X[keep]; isi_kept = M.get('F2_time')[keep]
    levels = np.asarray(M.get('isi_allowed', np.unique(isi_kept[np.isfinite(isi_kept)])))

    # baseline subtraction for viz (F2/CHOICE)
    if base_window is not None:
        bm = (T >= base_window[0]) & (T <= base_window[1])
        if np.any(bm):
            X = X - np.nanmean(X[:, :, bm], axis=2, keepdims=True)

    # equal-ISI averaging
    def avg_equal_isi(Xtr, isi):
        if not equalize_isi:
            return np.nanmedian(Xtr, 0) if use_median else np.nanmean(Xtr, 0)  # (Nroi,Nt)
        per = []
        for lv in levels:
            sel = np.isfinite(isi) & (np.abs(isi - lv) < 1e-6)
            if sel.sum() == 0: continue
            per.append(np.nanmedian(Xtr[sel], 0) if use_median else np.nanmean(Xtr[sel], 0))
        if not per: return np.full(Xtr.shape[1:], np.nan)
        return np.nanmean(np.stack(per, 0), 0)

    # ROI-by-time matrix after equal-ISI averag.
    Y = avg_equal_isi(X, isi_kept)             # (Nroi, Nt)
    if smooth_sigma_s:
        dt = np.median(np.diff(T))
        Y = gaussian_filter1d(Y, sigma=max(0.01, smooth_sigma_s/max(dt,1e-6)), axis=1, mode='nearest')

    # plot
    plt.figure(figsize=(9,3.6))
    vals_for_limits=[]
    for g in groups:
        m = masks.get(g, np.zeros(Y.shape[0], bool))
        if m.sum()==0: continue
        Yn = Y[m]                                # (nROI, Nt)
        mu = np.nanmean(Yn, 0) if not use_median else np.nanmedian(Yn, 0)
        if not use_median:
            se = np.nanstd(Yn, 0)/np.sqrt(np.maximum(1, np.sum(np.isfinite(Yn), 0)))
            plt.fill_between(T, mu-se, mu+se, alpha=0.18, color=colors.get(g,'C0'))
        else:
            q25, q75 = np.nanpercentile(Yn, [25,75], axis=0)
            plt.fill_between(T, q25, q75, alpha=0.18, color=colors.get(g,'C0'))
        plt.plot(T, mu, lw=2.2, color=colors.get(g,'C0'), label=f'{g} (n={m.sum()})')
        # light spaghetti
        for ridx in np.where(m)[0][:max_spaghetti]:
            plt.plot(T, Y[ridx], lw=0.6, alpha=0.25, color=colors.get(g,'C0'))
        vals_for_limits.append(Yn.ravel())

    # robust y-limits
    if vals_for_limits:
        arr = np.concatenate(vals_for_limits)
        lo, hi = np.nanpercentile(arr, [2, 98]); pad = (hi-lo)*0.10
        plt.ylim(lo-pad, hi+pad)

    # vertical ISI guides for F1-OFF
    if align.upper()=='F1OFF' and 'isi_allowed' in M:
        for a in np.asarray(M['isi_allowed']): plt.axvline(a, color='k', ls=':', lw=0.6, alpha=0.35)

    plt.xlabel(xlabel); plt.ylabel('ΔF/F')
    plt.title(title or f"Group mean traces ({' vs '.join(groups)})")
    plt.legend(frameon=False)
    plt.tight_layout()


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


# ---- v2 additions: NaN‑robust trough phase + RSA ----

def trough_time_vs_isi_v2(M: Dict[str, Any], smooth_w: int = 7) -> Dict[str, Any]:
    """Same as trough_time_vs_isi but also returns phase_of_trough and phase summaries.
    Uses NaN‑safe PCA fitting and robust handling of invalid bins.
    """
    X = M['roi_traces']; T = M['time']; F2 = M['F2_time']
    Ntr, Nroi, Nt = X.shape
    mean_rt = np.nanmean(X, axis=0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    strict_cols = np.all(np.isfinite(mean_rt), axis=0)
    if strict_cols.sum() >= 10:
        pca.fit(mean_rt[:, strict_cols].T)
    else:
        mean_rt_f = mean_rt.copy()
        col_means = np.nanmean(mean_rt_f, axis=0)
        nan_idx = ~np.isfinite(mean_rt_f)
        if np.any(nan_idx):
            mean_rt_f[nan_idx] = np.take(col_means, np.where(nan_idx)[1])
        pca.fit(mean_rt_f.T)
    w = pca.components_[0]
    levels = np.sort(np.unique(F2[np.isfinite(F2)]))
    trough_time = np.full(levels.shape, np.nan)
    trough_idx = np.full(levels.shape, -1, dtype=int)
    for k, isi in enumerate(levels):
        sel = np.isclose(F2, isi, rtol=0, atol=1e-6)
        if sel.sum() == 0:
            continue
        avg = np.nanmean(X[sel], axis=0)
        pop = np.dot(w, avg)
        pop = np.convolve(pop, np.array([1,2,1])/4.0, mode='same') if smooth_w <= 3 else pop
        valid = (T >= 0) & (T < isi) & np.isfinite(pop)
        if not np.any(valid):
            continue
        idx_local = np.nanargmin(pop[valid])
        global_idx = np.arange(Nt)[valid][idx_local]
        trough_idx[k] = global_idx
        trough_time[k] = T[global_idx]
    phase_of_trough = np.full_like(trough_time, np.nan)
    nz = levels > 1e-12
    phase_of_trough[nz] = trough_time[nz] / levels[nz]
    phi_vals = phase_of_trough[np.isfinite(phase_of_trough)]
    phi_median = float(np.nanmedian(phi_vals)) if phi_vals.size else np.nan
    phi_q25 = float(np.nanpercentile(phi_vals, 25)) if phi_vals.size else np.nan
    phi_q75 = float(np.nanpercentile(phi_vals, 75)) if phi_vals.size else np.nan
    # constant vs linear fit on trough_time
    y = trough_time; x = levels
    finite = np.isfinite(x) & np.isfinite(y)
    const_pred = np.full_like(y, np.nan); lin_pred = np.full_like(y, np.nan)
    r2_const = np.nan; r2_lin = np.nan
    if finite.sum() >= 3:
        yv, xv = y[finite], x[finite]
        const = float(np.nanmean(yv))
        const_pred[finite] = const
        r2_const = 1.0 - np.nanmean((yv-const)**2) / (np.nanvar(yv)+1e-12)
        A = np.column_stack([np.ones_like(xv), xv])
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
        lin = A @ beta
        lin_pred[finite] = lin
        r2_lin = 1.0 - np.nanmean((yv-lin)**2) / (np.nanvar(yv)+1e-12)
    return {
        'isi_levels': levels,
        'trough_time': trough_time,
        'trough_idx': trough_idx,
        'phase_of_trough': phase_of_trough,
        'phi_median': phi_median,
        'phi_q25': phi_q25,
        'phi_q75': phi_q75,
        'r2_constant': float(r2_const),
        'r2_linear': float(r2_lin),
        'constant_pred': const_pred,
        'linear_pred': lin_pred,
    }


def rsa_phase_vs_time_v2(M: Dict[str, Any], n_phase_bins: int = 20, window: tuple[float,float] | None = None) -> Dict[str, Any]:
    """NaN‑robust RSA : absolute time vs phase (nearest‑neighbor fill for empty phase bins).
    Optional `window=(t0,t1)` restricts the absolute-time comparison to a pre‑F2 subwindow
    (e.g., to remove the early sensory transient).
    """
    X = M['roi_traces']; T = M['time']; F2 = M['F2_time']
    Ntr, Nroi, Nt = X.shape
    levels = np.sort(np.unique(F2[np.isfinite(F2)]))
    K = len(levels)
    mean_by_level = []
    for isi in levels:
        sel = np.isclose(F2, isi, rtol=0, atol=1e-6)
        mean_by_level.append(np.nanmean(X[sel], axis=0))
    # Absolute‑time RSA
    C_ms = np.full((K,K), np.nan)
    for i in range(K):
        for j in range(K):
            if i==j:
                C_ms[i,j]=1.0; continue
            tmax = min(levels[i], levels[j])
            valid = (T>=0)&(T<tmax)
            if window is not None:
                valid = valid & (T>=window[0]) & (T<=window[1])
            if valid.sum()<5: continue
            A = mean_by_level[i][:,valid].reshape(-1)
            B = mean_by_level[j][:,valid].reshape(-1)
            a = (A-np.nanmean(A))/(np.nanstd(A)+1e-9)
            b = (B-np.nanmean(B))/(np.nanstd(B)+1e-9)
            m = np.isfinite(a)&np.isfinite(b)
            if m.sum()<5: continue
            C_ms[i,j]=float(np.nanmean(a[m]*b[m]))
    # Helper to fill phase bins
    def _fill_nearest(v):
        w=v.copy()
        if np.all(~np.isfinite(w)): return w
        for k in range(1,len(w)):
            if not np.isfinite(w[k]): w[k]=w[k-1]
        for k in range(len(w)-2,-1,-1):
            if not np.isfinite(w[k]): w[k]=w[k+1]
        return w
    phase_grid = np.linspace(0,1,n_phase_bins)
    mean_by_phase=[]
    for k,isi in enumerate(levels):
        valid=(T>=0)&(T<isi)
        A=np.full((Nroi,n_phase_bins),np.nan)
        if valid.sum()>=5:
            t_rel=(T[valid]/max(isi,1e-9)).clip(0,1)
            for b in range(n_phase_bins-1):
                m=(t_rel>=phase_grid[b])&(t_rel<phase_grid[b+1])
                if m.sum()>0:
                    A[:,b]=np.nanmean(mean_by_level[k][:,valid][:,m],axis=1)
            m_last=(t_rel>=phase_grid[-1])
            if m_last.sum()>0:
                A[:,-1]=np.nanmean(mean_by_level[k][:,valid][:,m_last],axis=1)
        for r in range(Nroi):
            A[r]=_fill_nearest(A[r])
        mean_by_phase.append(A)
    C_ph=np.full((K,K),np.nan)
    for i in range(K):
        for j in range(K):
            if i==j:
                C_ph[i,j]=1.0; continue
            A=mean_by_phase[i].reshape(-1); B=mean_by_phase[j].reshape(-1)
            a=(A-np.nanmean(A))/(np.nanstd(A)+1e-9)
            b=(B-np.nanmean(B))/(np.nanstd(B)+1e-9)
            m=np.isfinite(a)&np.isfinite(b)
            if m.sum()<5: continue
            C_ph[i,j]=float(np.nanmean(a[m]*b[m]))
    off=~np.eye(K,dtype=bool)
    mean_ms=float(np.nanmean(C_ms[off])) if K>1 else np.nan
    mean_ph=float(np.nanmean(C_ph[off])) if K>1 else np.nan
    return {'levels':levels,'corr_ms':C_ms,'corr_phase':C_ph,'mean_corr_ms':mean_ms,'mean_corr_phase':mean_ph}


def scaling_vs_clock_windowed(M: Dict[str, Any], time_window: tuple[float,float], cv_folds: int = 5, knots_phase: int = 8, knots_time: int = 8):
    """Run scaling vs clock comparison restricted to a pre‑F2 time window.
    Equivalent to lobuleV_mvp.scaling_vs_clock but on a sliced time axis.
    Returns (delta_r2, model_pref, (r2_phase, r2_ms)).
    """
    from sklearn.preprocessing import SplineTransformer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    X = M['roi_traces']
    Tfull = M['time']
    F2 = M['F2_time']
    t0, t1 = time_window
    mask_t = (Tfull >= t0) & (Tfull <= t1)
    T = Tfull[mask_t]
    X = X[:, :, mask_t]
    Ntr, Nroi, Nt = X.shape
    # Build time and phase spline bases
    st_time = SplineTransformer(degree=3, n_knots=knots_time, include_bias=False)
    B_time = st_time.fit_transform(T.reshape(-1,1))
    B_time = np.broadcast_to(B_time, (Ntr, Nt, B_time.shape[1]))
    st_phase = SplineTransformer(degree=3, n_knots=knots_phase, include_bias=False)
    st_phase.fit(np.linspace(0,1,len(T)).reshape(-1,1))
    B_phase = np.zeros((Ntr, Nt, st_phase.n_features_out_))
    for i in range(Ntr):
        ph = (T / max(F2[i], 1e-9)).reshape(-1,1)
        ph = np.clip(ph, 0.0, 1.0)
        B_phase[i] = st_phase.transform(ph)
    def _cv_r2(Y, B):
        X_all = B.reshape(Ntr*Nt, B.shape[2])
        y_all = Y.reshape(Ntr*Nt)
        finite = np.isfinite(y_all)
        if finite.sum() < 10:
            return np.nan
        trial_ids = np.repeat(np.arange(Ntr), Nt)
        preds = np.full_like(y_all, np.nan, dtype=float)
        kf = KFold(n_splits=min(cv_folds, Ntr), shuffle=True, random_state=0)
        for train_tr, test_tr in kf.split(np.arange(Ntr)):
            tr_mask = np.isin(trial_ids, train_tr) & finite
            te_mask = np.isin(trial_ids, test_tr) & finite
            if tr_mask.sum() < B.shape[2] + 2 or te_mask.sum() == 0:
                continue
            reg = Ridge(alpha=1.0).fit(X_all[tr_mask], y_all[tr_mask])
            preds[te_mask] = reg.predict(X_all[te_mask])
        ok = np.isfinite(y_all) & np.isfinite(preds)
        return r2_score(y_all[ok], preds[ok]) if ok.sum() > 10 else np.nan
    r2_phase = np.zeros(Nroi)
    r2_ms = np.zeros(Nroi)
    for r in range(Nroi):
        Y = X[:, r, :]
        r2_phase[r] = _cv_r2(Y, B_phase)
        r2_ms[r] = _cv_r2(Y, B_time)
    delta = r2_phase - r2_ms
    eps = 0.02
    pref = {
        'scaling': np.where(delta >  eps)[0],
        'clock':   np.where(delta < -eps)[0],
        'mixed':   np.where(np.abs(delta) <= eps)[0],
    }
    return delta, pref, (r2_phase, r2_ms)


def add_F2_locked_to_M(M: Dict[str, Any], trial_data: Dict[str, Any], window: tuple[float,float] = (-0.2, 0.6), grid_bins: int = 160) -> Dict[str, Any]:
    """Augment M with F2-locked arrays built from trial_data.
    - window: (start, stop) in seconds relative to F2 onset.
    - Interpolates each trial's ROI×time onto a common τ grid.
    """
    
    """Construct canonical M from trial_data['df_trials_with_segments'].
    - F1-OFF aligned pre-F2 grid of length max(ISI), 240 bins.
    - Per-trial NaN mask for t >= ISI.
    """
    print("=== Starting build_M_from_trials ===")
    
    
    df = trial_data['df_trials_with_segments']
    n_trials = len(df)
    first = df.iloc[0]
    dff0 = np.asarray(first['dff_segment'], dtype=float)
    if dff0.ndim != 2:
        raise ValueError("dff_segment must be 2D (n_rois x n_samples) per trial")
    n_rois = dff0.shape[0]
    tau = np.linspace(window[0], window[1], grid_bins)
    Xf2 = np.full((n_trials, n_rois, grid_bins), np.nan, dtype=np.float32)  
    
    servo_rel = np.full(n_trials, np.nan, dtype=float)
    lick_rel = np.full(n_trials, np.nan, dtype=float)    
          
    for idx, (trial_idx, row) in enumerate(df.iterrows()):
        t_vec = np.asarray(row['dff_time_vector'], dtype=float) - float(row['start_flash_2'])
        dff = np.asarray(row['dff_segment'], dtype=float)
        t_min, t_max_local = np.nanmin(t_vec), np.nanmax(t_vec)
        for r in range(n_rois):
            y = dff[r]
            if not np.all(np.diff(t_vec) > 0):
                order = np.argsort(t_vec)
                tt = t_vec[order]
                yy = y[order]
            else:
                tt, yy = t_vec, y
            vals = np.interp(tau, tt, yy, left=np.nan, right=np.nan)
            vals[(tau < t_min) | (tau > t_max_local)] = np.nan
            Xf2[idx, r] = vals
        # relative behavior times (if present)
        try:
            servo_rel[idx] = float(row.get('servo_in', np.nan)) - float(row['start_flash_2'])
        except Exception:
            servo_rel[idx] = np.nan
        try:
            lick_rel[idx] = float(row.get('lick_start', np.nan)) - float(row['start_flash_2'])
        except Exception:
            lick_rel[idx] = np.nan

    M['roi_traces_F2locked'] = Xf2
    M['time_F2locked'] = tau.astype(np.float32)
    M['servo_in_relF2'] = servo_rel.astype(np.float32)
    M['lick_start_relF2'] = lick_rel.astype(np.float32)
    return M



def decode_choice_F2locked_v2(M: Dict[str, Any], n_bins: int = 20,
                            baseline_window: tuple[float,float] | None = (-0.2, -0.05),
                            zscore_within_fold: bool = True,
                            restrict_preF2: bool = False) -> Dict[str, Any] | None:
    """Time‑resolved AUC(τ) for is_right_choice using F2‑locked traces.

    Robustness features:
      - Optional per‑trial, per‑ROI **baseline subtraction** over `baseline_window`.
      - **Within‑fold** standardization (fit scaler on train only) to avoid leakage.
      - Optional `restrict_preF2=True` to compute AUC only for τ<0.
    """
    if 'roi_traces_F2locked' not in M or 'time_F2locked' not in M:
        return None
    X = M['roi_traces_F2locked'].copy()
    T = M['time_F2locked']
    side = M.get('is_right_choice')
    if side is None:
        return None
    Ntr, Nroi, Nt = X.shape

    # Optional baseline subtraction
    if baseline_window is not None:
        b0, b1 = baseline_window
        bm = (T >= b0) & (T <= b1)
        if np.any(bm):
            base = np.nanmean(X[:, :, bm], axis=2, keepdims=True)  # (Ntr,Nroi,1)
            X = X - base

    # Optional restriction to pre‑F2 times only
    if restrict_preF2:
        mask_t = T < 0
        T = T[mask_t]
        X = X[:, :, mask_t]
        Nt = X.shape[2]
        if Nt < 3:
            return {'time_points': T, 'auc_t': np.full_like(T, np.nan, dtype=float)}

    # Bin τ
    edges = np.linspace(T[0], T[-1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    auc_t = np.full(len(centers), np.nan)

    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    for b in range(len(centers)):
        m = (T >= edges[b]) & (T < edges[b+1])
        if not np.any(m):
            continue
        Z = np.nanmean(X[:, :, m], axis=2)  # (Ntr, Nroi)
        valid = np.all(np.isfinite(Z), axis=1)
        y = side.astype(int)
        if valid.sum() < 12 or np.unique(y[valid]).size < 2:
            continue
        Z = Z[valid]; y = y[valid]
        kf = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=0)
        scores = []
        for tr, te in kf.split(Z):
            Ztr, Zte = Z[tr], Z[te]
            ytr, yte = y[tr], y[te]
            # within‑fold standardization
            if zscore_within_fold:
                scaler = StandardScaler().fit(Ztr)
                Ztr = scaler.transform(Ztr)
                Zte = scaler.transform(Zte)
            clf = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, max_iter=1000)
            clf.fit(Ztr, ytr)
            p = clf.predict_proba(Zte)[:, 1]
            if np.unique(yte).size < 2:
                continue
            scores.append(roc_auc_score(yte, p))
        auc_t[b] = float(np.nanmean(scores)) if scores else np.nan

    return {'time_points': centers, 'auc_t': auc_t}


# -----------------------------
# 2) Fair‑window pre‑F2 decoders (amplitude vs slope)
# -----------------------------

def _bin_time(T: np.ndarray, n_bins: int, tmin: float, tmax: float) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(tmin, tmax, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers

def decode_preF2_fairwindow(M: Dict[str, Any], tmax: float = 0.7, n_bins: int = 12, use_slope: bool = True) -> Dict[str, Any]:
    """Run decoders only in t < tmax so both categories are present.

    Returns dict with time points, ISI R^2(t), AUC(t) for amplitude features,
    and (optionally) for slope features computed via central differences.
    """
    X = M['roi_traces']   # (Ntr, Nroi, Nt)
    T = M['time']
    isi = M['F2_time']
    is_short = M['is_short']
    # Restrict to fair window
    mask_t = (T >= 0) & (T < min(tmax, np.nanmax(T)))
    Tsub = T[mask_t]
    Xsub = X[:, :, mask_t]

    # Build bins in the subwindow
    edges, centers = _bin_time(Tsub, n_bins=n_bins, tmin=Tsub[0], tmax=Tsub[-1])

    def _bin_feats(Xin: np.ndarray) -> np.ndarray:
        Ntr, Nroi, Nt = Xin.shape
        F = np.full((Ntr, Nroi, len(centers)), np.nan)
        for b in range(len(centers)):
            m = (Tsub >= edges[b]) & (Tsub < edges[b+1])
            if not np.any(m):
                continue
            F[:, :, b] = np.nanmean(Xin[:, :, m], axis=2)
        return F

    F_amp = _bin_feats(Xsub)

    results = {
        'time_points': centers,
        'amp': {'isi_r2_t': np.full(len(centers), np.nan), 'auc_t': np.full(len(centers), np.nan)},
    }

    # Helper CV decoders
    def _cv_ridge(Z, y):
        valid = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
        if valid.sum() < 12 or np.unique(y[valid]).size < 2:
            return np.nan
        Z = Z[valid]; y = y[valid]
        kf = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=0)
        preds = np.zeros_like(y, dtype=float)
        for tr, te in kf.split(Z):
            m = Ridge(alpha=1.0).fit(Z[tr], y[tr])
            preds[te] = m.predict(Z[te])
        return r2_score(y, preds)

    def _cv_auc(Z, y):
        valid = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
        if valid.sum() < 12 or np.unique(y[valid]).size < 2:
            return np.nan
        Z = Z[valid]; y = y[valid].astype(int)
        if np.unique(y).size < 2:
            return np.nan
        kf = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=0)
        scores = []
        for tr, te in kf.split(Z):
            if np.unique(y[te]).size < 2:
                continue
            clf = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, max_iter=1000)
            clf.fit(Z[tr], y[tr])
            p = clf.predict_proba(Z[te])[:, 1]
            scores.append(roc_auc_score(y[te], p))
        return float(np.nanmean(scores)) if scores else np.nan

    # Decode per bin (amplitude)
    for b in range(len(centers)):
        Z = F_amp[:, :, b]
        # Standardize across trials (valid only)
        valid = np.all(np.isfinite(Z), axis=1)
        if valid.sum() < 12:
            continue
        Zs = np.full_like(Z, np.nan)
        Zs[valid] = StandardScaler().fit_transform(Z[valid])
        results['amp']['isi_r2_t'][b] = _cv_ridge(Zs, isi)
        results['amp']['auc_t'][b] = _cv_auc(Zs, is_short)

    if use_slope:
        # Central differences along time inside fair window
        Xdiff = np.full_like(Xsub, np.nan)
        Xdiff[:, :, 1:-1] = (Xsub[:, :, 2:] - Xsub[:, :, :-2]) / np.maximum(1e-9, (Tsub[2:] - Tsub[:-2]))
        F_slope = _bin_feats(Xdiff)
        results['slope'] = {'isi_r2_t': np.full(len(centers), np.nan), 'auc_t': np.full(len(centers), np.nan)}
        for b in range(len(centers)):
            Z = F_slope[:, :, b]
            valid = np.all(np.isfinite(Z), axis=1)
            if valid.sum() < 12:
                continue
            Zs = np.full_like(Z, np.nan)
            Zs[valid] = StandardScaler().fit_transform(Z[valid])
            results['slope']['isi_r2_t'][b] = _cv_ridge(Zs, isi)
            results['slope']['auc_t'][b] = _cv_auc(Zs, is_short)

    return results

# -----------------------------
# 4) Hazard partial R^2 per ROI (discrete bumps)
# -----------------------------

def hazard_unique_variance_per_roi(M: Dict[str, Any], kernel: float = 0.12) -> Dict[str, Any]:
    """Per‑ROI partial R^2 of discrete hazard bumps beyond linear time.

    Predictors: [time] + Gaussian bumps centered at each allowed F2 time (width=kernel s).
    CV folds split by trial. Y is dF/F per ROI flattened across (trial,time).

    Returns: dict with 'partial_r2' (Nroi,), 'fraction_positive', and average bump weights.
    """
    X = M['roi_traces']
    T = M['time']
    F2 = M['F2_time']
    levels = np.sort(np.unique(F2[np.isfinite(F2)]))
    Ntr, Nroi, Nt = X.shape

    # Build design columns that depend only on absolute T
    tt = T.reshape(-1, 1)
    # Linear time
    X_time = (tt - tt.mean()) / (tt.std() + 1e-9)  # (Nt,1)
    # Gaussian bumps at each allowed time
    bumps = []
    for c in levels:
        bumps.append(np.exp(-0.5 * ((T - c) / max(kernel, 1e-3)) ** 2))
    X_bumps = np.stack(bumps, axis=1)  # (Nt, K)
    # Z‑score bumps columns
    X_bumps = (X_bumps - X_bumps.mean(axis=0)) / (X_bumps.std(axis=0) + 1e-9)

    # Tile across trials to match y
    X_time_tile = np.tile(X_time, (Ntr, 1))  # (Ntr*Nt, 1)
    X_bumps_tile = np.tile(X_bumps, (Ntr, 1))  # (Ntr*Nt, K)

    # Trial ids for CV
    trial_ids = np.repeat(np.arange(Ntr), Nt)
    kf = KFold(n_splits=min(5, Ntr), shuffle=True, random_state=0)

    partial = np.full(Nroi, np.nan)
    avg_betas = np.zeros((Nroi, len(levels)))

    for r in range(Nroi):
        Y = X[:, r, :].reshape(-1)
        finite = np.isfinite(Y)
        if finite.sum() < 50:
            continue

        def _cv_r2(Xcols):
            preds = np.full_like(Y, np.nan, dtype=float)
            for tr, te in kf.split(np.arange(Ntr)):
                m_tr = np.isin(trial_ids, tr) & finite
                m_te = np.isin(trial_ids, te) & finite
                if m_tr.sum() < Xcols.shape[1] + 2 or m_te.sum() == 0:
                    continue
                reg = LinearRegression().fit(Xcols[m_tr], Y[m_tr])
                preds[m_te] = reg.predict(Xcols[m_te])
            ok = np.isfinite(Y) & np.isfinite(preds)
            return r2_score(Y[ok], preds[ok]) if ok.sum() > 10 else np.nan

        r2_time = _cv_r2(X_time_tile)
        X_both = np.column_stack([X_time_tile, X_bumps_tile])
        r2_both = _cv_r2(X_both)
        if np.isfinite(r2_time) and np.isfinite(r2_both):
            partial[r] = max(0.0, float(r2_both - r2_time))

        # Fit once on all finite data to get bump betas (for descriptive weights)
        try:
            reg_all = LinearRegression().fit(X_both[finite], Y[finite])
            avg_betas[r] = reg_all.coef_[1:]  # skip time column
        except Exception:
            avg_betas[r] = np.nan

    fraction_pos = float(np.nanmean(partial > 0))
    mean_betas = np.nanmean(avg_betas, axis=0)  # (K,)

    return {
        'levels': levels,
        'partial_r2': partial,
        'fraction_positive': fraction_pos,
        'mean_bump_weights': mean_betas,
    }


def decode_choice_choiceLocked_v3(
    M, n_bins=24, baseline_window=(-0.6, -0.2),
    zscore_within_fold=True, restrict_preChoice=True,
    stratify_by_isi=True, random_state=0
):
    """
    Time-resolved left vs right decode around choice_start.
    Builds CV folds *per bin* on the valid subset → robust when NaNs vary with time.
    """
    X = M.get('roi_traces_choice'); T = M.get('time_choice')
    side = M.get('is_right_choice'); isi = M.get('F2_time')
    if X is None or T is None or side is None: return None
    side = side.astype(int)
    X = X.copy()

    # Baseline subtraction
    if baseline_window is not None:
        b0, b1 = baseline_window
        bm = (T >= b0) & (T <= b1)
        if np.any(bm):
            X -= np.nanmean(X[:, :, bm], axis=2, keepdims=True)

    # Restrict to pre-choice (optional)
    if restrict_preChoice:
        mpre = T < 0
        T = T[mpre]; X = X[:, :, mpre]

    # Time bins
    edges = np.linspace(T[0], T[-1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    auc_t = np.full(len(centers), np.nan)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold

    for b in range(len(centers)):
        m = (T >= edges[b]) & (T < edges[b+1])
        if not np.any(m): continue

        Z_full = np.nanmean(X[:, :, m], axis=2)           # (Ntr, Nroi)
        valid_idx = np.where(np.all(np.isfinite(Z_full), axis=1))[0]
        if valid_idx.size < 12: continue
        yv = side[valid_idx]
        if np.unique(yv).size < 2: continue

        # Build folds on the valid subset (critical fix)
        if stratify_by_isi and isi is not None:
            levels = np.unique(isi[np.isfinite(isi)])
            lvl = np.searchsorted(levels, np.asarray(isi)[valid_idx])
            y_joint = yv * max(1, levels.size) + lvl
            splitter = StratifiedKFold(n_splits=min(5, len(valid_idx)),
                                       shuffle=True, random_state=random_state)
            folds = splitter.split(np.zeros_like(y_joint), y_joint)
        else:
            splitter = KFold(n_splits=min(5, len(valid_idx)),
                             shuffle=True, random_state=random_state)
            folds = splitter.split(np.arange(len(valid_idx)))

        Z = Z_full[valid_idx]
        scores = []
        for tr_rel, te_rel in folds:
            if tr_rel.size < 10 or te_rel.size < 4: continue
            Ztr, Zte = Z[tr_rel], Z[te_rel]
            ytr, yte = yv[tr_rel], yv[te_rel]
            if zscore_within_fold:
                sc = StandardScaler().fit(Ztr)
                Ztr = sc.transform(Ztr); Zte = sc.transform(Zte)
            clf = LogisticRegression(penalty='l2', solver='liblinear',
                                     C=1.0, max_iter=1000)
            clf.fit(Ztr, ytr)
            p = clf.predict_proba(Zte)[:, 1]
            if np.unique(yte).size < 2: continue
            scores.append(roc_auc_score(yte, p))
        auc_t[b] = float(np.nanmean(scores)) if scores else np.nan

    return {'time_points': centers, 'auc_t': auc_t}


def _safe_percentile(a, q):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return np.percentile(a, q)


def _maybe_compute_extras(M, compute_extras: bool, extras_in: dict | None, preF2_window: tuple[float,float] | None):
    out = extras_in.copy() if isinstance(extras_in, dict) else {}
    if compute_extras:

        out['trough'] = trough_time_vs_isi_v2(M)
 
        out['fair'] = decode_preF2_fairwindow(M, tmax=0.7, n_bins=12, use_slope=True)
  
        # out['rsa'] = rsa_phase_vs_time_v2(M, n_phase_bins=20)
        out['rsa'] = rsa_phase_vs_time_v2(M, n_phase_bins=20, window=preF2_window)
  
        out['hz_roi'] = hazard_unique_variance_per_roi(M, kernel=0.12)

        out['choice'] = decode_choice_F2locked_v2(M)
        
        # out['choice'] = decode_choice_choiceLocked_v3(M)  # pre-choice AUC by default
    return out


def plot_session_report_v2(M: dict, R: dict, session_title: str | None = None,
                           save_path: str | None = None, compute_extras: bool = True,
                           extras: dict | None = None, preF2_window: tuple[float,float] | None = None):
    # Prepare extras
    X = M['roi_traces']
    T = M['time']
    extras = _maybe_compute_extras(M, compute_extras, extras, preF2_window)

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(3, 3, height_ratios=[1.0, 1.0, 1.4], width_ratios=[1.0, 1.0, 1.0], wspace=0.35, hspace=0.35)

    # ------ [1] ΔR² histogram ------
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
    pref = R.get('model_pref', {})
    n_total = int(np.isfinite(delta).sum())
    n_scal = len(pref.get('scaling', []))
    n_clock = len(pref.get('clock', []))
    n_mixed = len(pref.get('mixed', []))
    txt = (f"median ΔR² = {med:.03f}\n"
           f"scaling: {n_scal}/{n_total} ({(100*n_scal/max(n_total,1)):.1f}%)\n"
           f"clock:   {n_clock}/{n_total} ({(100*n_clock/max(n_total,1)):.1f}%)\n"
           f"mixed:   {n_mixed}/{n_total} ({(100*n_mixed/max(n_total,1)):.1f}%)")
    ax1.text(0.98, 0.98, txt, transform=ax1.transAxes, va='top', ha='right', fontsize=9,
             bbox=dict(boxstyle='round', fc='white', ec='0.8', alpha=0.9))

    # ------ [2] Time‑resolved decode ------
    ax2 = fig.add_subplot(gs[0, 1])
    dec = R.get('decode', {})
    tpts = np.asarray(dec.get('time_points'))
    r2_t = np.asarray(dec.get('isi_r2_t'))
    auc_t = np.asarray(dec.get('shortlong_auc_t'))
    if tpts.size > 0:
        ax2.plot(tpts, r2_t, lw=2, label='ISI $R^2$(t)')
        ax2b = ax2.twinx()
        ax2b.plot(tpts, auc_t, lw=2, ls='--', label='Short/Long AUC(t)')
        div_t = dec.get('divergence_time')
        if div_t is not None and np.isfinite(div_t):
            ax2.axvline(div_t, color='k', lw=1.5, ls=':')
            ax2.text(div_t, ax2.get_ylim()[1], 'divergence', rotation=90, va='top', ha='right', fontsize=8)
        ax2.set_xlabel('Time from F1-OFF (s)')
        ax2.set_ylabel('ISI $R^2$(t)')
        ax2.set_title('Time-resolved decode')
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, frameon=True)
    else:
        ax2.axis('off')

    # ------ [3] Hazard summary ------
    ax3 = fig.add_subplot(gs[0, 2])
    huv = R.get('hazard_unique_r2')
    hz_roi = extras.get('hz_roi') if isinstance(extras, dict) else None
    if huv is None or not np.isfinite(huv):
        ax3.text(0.5, 0.5, 'Hazard N/A', va='center', ha='center')
        ax3.set_axis_off()
    else:
        bars = [huv]
        labels = ['Session UV']
        title = 'Anticipation beyond ramp'
        if hz_roi:
            frac_pos = hz_roi.get('fraction_positive')
            if frac_pos is not None and np.isfinite(frac_pos):
                bars.append(frac_pos)
                labels.append('Frac ROIs UV>0')
        ax3.bar(labels, bars)
        ax3.set_ylim(0, max(0.01, 1.05 * np.nanmax(bars)))
        ax3.set_title(title)
        for i, v in enumerate(bars):
            ax3.text(i, v, f" {v:.3f}", va='bottom', ha='left')
        # Optional bump weights as inset
        if hz_roi and 'mean_bump_weights' in hz_roi:
            inset = ax3.inset_axes([0.55, 0.1, 0.4, 0.8])
            betas = np.asarray(hz_roi['mean_bump_weights'])
            levels = np.asarray(hz_roi.get('levels', np.arange(len(betas))))
            inset.plot(levels, betas, marker='o')
            inset.set_title('Mean bump weights', fontsize=8)
            inset.tick_params(labelsize=7)

    # ------ [4] Trough vs ISI ------
    ax4 = fig.add_subplot(gs[1, 0])
    thr = extras.get('trough') if isinstance(extras, dict) else None
    if thr:
        x = np.asarray(thr['isi_levels']); y = np.asarray(thr['trough_time'])
        ax4.plot(x, y, 'o-', lw=2, label='trough time')
        # Overplot constant vs linear fits if present
        const_pred = np.asarray(thr.get('constant_pred'))
        lin_pred = np.asarray(thr.get('linear_pred'))
        if np.any(np.isfinite(const_pred)):
            r2c = thr.get('r2_constant')
            ax4.plot(x, const_pred, ':', lw=1.5, label=f"const R²={r2c:.2f}" if r2c is not None else 'const')
        if np.any(np.isfinite(lin_pred)):
            r2l = thr.get('r2_linear')
            ax4.plot(x, lin_pred, '--', lw=1.5, label=f"linear R²={r2l:.2f}" if r2l is not None else 'linear')
        ax4.set_xlabel('ISI (s)')
        ax4.set_ylabel('Trough time (s, F1-OFF)')
        ax4.set_title('Trough/turning-point vs ISI')
        # Phase-of-trough summary
        phi_med = thr.get('phi_median'); q25 = thr.get('phi_q25'); q75 = thr.get('phi_q75')
        if phi_med is not None and np.isfinite(phi_med):
            txt = f"ϕ_trough median = {phi_med:.2f} IQR = [{q25:.2f}, {q75:.2f}]"
            ax4.text(0.98, 0.02, txt, transform=ax4.transAxes, va='bottom', ha='right', fontsize=9,
                     bbox=dict(boxstyle='round', fc='white', ec='0.8', alpha=0.9))
        ax4.legend(fontsize=8)
    else:
        ax4.axis('off')

    # ------ [5] Fair-window decoders (amp vs slope) ------
    ax5 = fig.add_subplot(gs[1, 1])
    fair = extras.get('fair') if isinstance(extras, dict) else None
    if fair:
        tpts_fw = np.asarray(fair['time_points'])
        amp_r2 = np.asarray(fair['amp']['isi_r2_t'])
        amp_auc = np.asarray(fair['amp']['auc_t'])
        ax5.plot(tpts_fw, amp_r2, lw=2, label='amp ISI $R^2$')
        ax5b = ax5.twinx()
        ax5b.plot(tpts_fw, amp_auc, lw=2, ls='--', label='amp AUC')
        if 'slope' in fair:
            sl_r2 = np.asarray(fair['slope']['isi_r2_t'])
            sl_auc = np.asarray(fair['slope']['auc_t'])
            ax5.plot(tpts_fw, sl_r2, lw=1.5, label='slope ISI $R^2$')
            ax5b.plot(tpts_fw, sl_auc, lw=1.5, ls='--', label='slope AUC')
        ax5.set_xlabel('Time from F1-OFF (s), t < 0.7 s')
        ax5.set_ylabel('ISI $R^2$')
        ax5.set_title('Fair-window decoders (amp vs slope)')
        # Annotate slope>amp if so
        try:
            mean_amp = np.nanmean(amp_r2)
            mean_slope = np.nanmean(sl_r2) if 'slope' in fair else np.nan
            if np.isfinite(mean_slope) and np.isfinite(mean_amp) and mean_slope > mean_amp:
                ax5.text(0.02, 0.02, f"slope R² > amp R² ({mean_slope:.3f} > {mean_amp:.3f})", transform=ax5.transAxes,
                         va='bottom', ha='left', fontsize=8,
                         bbox=dict(boxstyle='round', fc='white', ec='0.8', alpha=0.9))
        except Exception:
            pass
        l1, lab1 = ax5.get_legend_handles_labels()
        l2, lab2 = ax5b.get_legend_handles_labels()
        ax5.legend(l1 + l2, lab1 + lab2, loc='upper left', fontsize=8, frameon=True)
    else:
        ax5.axis('off')

    # ------ [6] RSA summary ------
    ax6 = fig.add_subplot(gs[1, 2])
    rsa = extras.get('rsa') if isinstance(extras, dict) else None
    if rsa:
        m_ms = rsa.get('mean_corr_ms'); m_ph = rsa.get('mean_corr_phase')
        ax6.bar(['Abs time', 'Phase'], [m_ms, m_ph])
        ax6.set_ylim(0, 1)
        ax6.set_title('RSA mean off-diagonal corr')
        for i, v in enumerate([m_ms, m_ph]):
            if v is not None and np.isfinite(v):
                ax6.text(i, v, f" {v:.2f}", va='bottom', ha='left')
    else:
        ax6.axis('off')

    # # ------ [7] F2-locked choice AUC(τ) ------
    # ax7 = fig.add_subplot(gs[2, 0])
    # choice = extras.get('choice') if isinstance(extras, dict) else None
    # if choice:
    #     t_tau = np.asarray(choice['time_points']); auc_tau = np.asarray(choice['auc_t'])
    #     ax7.plot(t_tau, auc_tau, lw=2)
    #     ax7.axhline(0.5, color='k', ls='--', lw=1)
    #     # Divergence time (fixed threshold rule)
    #     def _div_time(t, s, thr=0.6, sustain=3):
    #         run = 0
    #         for i, v in enumerate(s):
    #             if np.isfinite(v) and v > thr:
    #                 run += 1
    #                 if run >= sustain:
    #                     return t[i - sustain + 1]
    #             else:
    #                 run = 0
    #         return np.nan
    #     div_t = _div_time(t_tau, auc_tau)
    #     if np.isfinite(div_t):
    #         ax7.axvline(div_t, color='k', lw=1.5, ls=':')
    #         ax7.text(div_t, ax7.get_ylim()[1], 'divergence', rotation=90, va='top', ha='right', fontsize=8)
    #     # Behavior overlays if available (median servo_in and lick_start relative to F2)
    #     try:
    #         servo_med = np.nanmedian(M.get('servo_in_relF2'))
    #         if np.isfinite(servo_med):
    #             ax7.axvline(servo_med, color='C2', lw=1, ls='--')
    #             ax7.text(servo_med, ax7.get_ylim()[1], 'servo_in', rotation=90, va='top', ha='right', fontsize=8, color='C2')
    #     except Exception:
    #         pass
    #     try:
    #         lick_med = np.nanmedian(M.get('lick_start_relF2'))
    #         if np.isfinite(lick_med):
    #             ax7.axvline(lick_med, color='C3', lw=1, ls='--')
    #             ax7.text(lick_med, ax7.get_ylim()[1], 'first lick', rotation=90, va='top', ha='right', fontsize=8, color='C3')
    #     except Exception:
    #         pass
    #     ax7.set_xlabel('Time from F2 (s)')
    #     ax7.set_ylabel('Choice AUC(τ)')
    #     ax7.set_title('F2-locked choice decode')
    # else:
    #     ax7.text(0.5, 0.5, 'F2-locked choice AUC N/A', va='center', ha='center')
    #     ax7.set_axis_off()

    # ------ [7] Choice-locked left/right AUC (pre-choice) ------
    ax7 = fig.add_subplot(gs[2, 0])
    choice = extras.get('choice') if isinstance(extras, dict) else None
    if choice:
        tt, aa = np.asarray(choice['time_points']), np.asarray(choice['auc_t'])
        ax7.plot(tt, aa, lw=2)
        ax7.axhline(0.5, color='k', ls='--', lw=1)

        def _div_time(t, s, thr=0.6, sustain=3):
            run = 0
            for i,v in enumerate(s):
                if np.isfinite(v) and v > thr:
                    run += 1
                    if run >= sustain: return t[i - sustain + 1]
                else: run = 0
            return np.nan
        div_t = _div_time(tt, aa)
        if np.isfinite(div_t):
            ax7.axvline(div_t, color='k', lw=1.5, ls=':')
            ax7.text(div_t, ax7.get_ylim()[1], 'divergence', rotation=90,
                    va='top', ha='right', fontsize=8)

        # overlay first lick median (relative to choice_start)
        try:
            lk_med = np.nanmedian(M.get('lick_start_relChoice'))
            if np.isfinite(lk_med):
                ax7.axvline(lk_med, color='C3', lw=1, ls='--')
                ax7.text(lk_med, ax7.get_ylim()[1], 'first lick', rotation=90,
                        va='top', ha='right', fontsize=8, color='C3')
        except Exception:
            pass
        ax7.set_xlabel('Time from choice start (s)')
        ax7.set_ylabel('AUC(τ)')
        ax7.set_title('Choice-locked left/right decode (pre-choice)')
    else:
        ax7.text(0.5, 0.5, 'Choice-locked AUC N/A', va='center', ha='center')
        ax7.set_axis_off()



    # ------ [8-9] Sorted raster ------
    ax8 = fig.add_subplot(gs[2, 1:])
    order = np.asarray(R.get('sort_index')) if 'sort_index' in R else np.arange(X.shape[1])
    mean_tr = np.nanmean(X, axis=0)[order]
    vmin = _safe_percentile(mean_tr, 5)
    vmax = _safe_percentile(mean_tr, 95)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(mean_tr), np.nanmax(mean_tr)
    im = ax8.imshow(mean_tr, aspect='auto', interpolation='nearest',
                    extent=[T[0], T[-1], 0, mean_tr.shape[0]], origin='lower',
                    vmin=vmin, vmax=vmax)
    ax8.set_xlabel('Time from F1-OFF (s)')
    ax8.set_ylabel('ROIs (sorted)')
    ax8.set_title('Population raster (avg across trials)')
    if 'isi_allowed' in M:
        for isi in np.asarray(M['isi_allowed']):
            ax8.axvline(isi, color='w', lw=0.8, ls=':', alpha=0.8)
    cbar = plt.colorbar(im, ax=ax8, fraction=0.02, pad=0.02)
    cbar.set_label('ΔF/F (a.u.)')

    if session_title:
        fig.suptitle(session_title, y=0.995, fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

# def plot_roi_group_traces(M, masks, group_a, group_b=None,
#                           align='F1OFF', max_spaghetti=10, save_path=None):
#     # ---- choose alignment ----
#     if align.upper() == 'F1OFF':
#         X, T = M['roi_traces'], M['time']; xlabel = 'Time from F1-OFF (s)'
#         vlines = M.get('isi_allowed', [])
#     elif align.upper() == 'F2':
#         X, T = M.get('roi_traces_F2locked'), M.get('time_F2locked'); xlabel = 'Time from F2 (s)'; vlines=None
#     elif align.upper() == 'CHOICE':
#         X, T = M.get('roi_traces_choice'), M.get('time_choice'); xlabel = 'Time from choice start (s)'; vlines=None
#     else:
#         raise ValueError('align must be F1OFF, F2, or CHOICE')
#     if X is None or T is None:
#         raise ValueError('alignment arrays missing—did you call the add_*_to_M helper?')

#     # session-average per ROI (Nroi x Nt) used for spaghetti + limits
#     Y = nanmean_no_warn(X, axis=0)

#     plt.figure(figsize=(9, 3.2))
#     vals_for_limits = []  # <-- collect values from the groups we actually plotted

#     def _group_trace(mask, color, label):
#         m = np.asarray(mask, bool)
#         # mean ± sem over the selected ROIs
#         y   = nanmean_no_warn(Y[m], axis=0)
#         sem = np.nanstd(Y[m], axis=0) / np.sqrt(max(1, m.sum()))

#         # spaghetti (a few example ROIs)
#         for ridx in np.where(m)[0][:max_spaghetti]:
#             plt.plot(T, Y[ridx], alpha=0.15, lw=1, color=color)

#         # mean curve
#         plt.plot(T, y, lw=2.2, color=color, label=f"{label} (n={int(m.sum())})")
#         plt.fill_between(T, y - sem, y + sem, alpha=0.18, color=color)

#         # collect values for robust y-limits
#         if m.any():
#             vals_for_limits.append(Y[m].ravel())

#     _group_trace(masks[group_a], 'C0', group_a)
#     if group_b is not None and group_b in masks:
#         _group_trace(masks[group_b], 'C1', group_b)

#     # vertical guides at allowed ISIs (if F1-OFF aligned)
#     if vlines is not None and len(vlines):
#         for a in vlines:
#             plt.axvline(a, color='k', ls=':', lw=0.6, alpha=0.35)

#     # ----- robust y-limits from what was plotted -----
#     if len(vals_for_limits):
#         arr = np.concatenate(vals_for_limits)
#         lo, hi = np.nanpercentile(arr, [2, 98])
#         pad = (hi - lo) * 0.10
#         plt.ylim(lo - pad, hi + pad)
#     # --------------------------------------------------

#     plt.xlabel(xlabel); plt.ylabel('ΔF/F')
#     plt.title(f'Group mean traces ({group_a}' + (f' vs {group_b}' if group_b else '') + ')')
#     plt.legend(frameon=False)
#     if save_path:
#         plt.savefig(save_path, dpi=170, bbox_inches='tight')
#     return plt.gcf()




# def plot_roi_group_traces(M: Dict[str, Any], masks: Dict[str, np.ndarray], group_a: str, group_b: str | None = None, align: str = 'F1OFF', max_spaghetti: int = 10, save_path: str | None = None):
#     """Mean + SEM with light spaghetti for each ROI in the group(s)."""
#     if align.upper() == 'F1OFF':
#         X, T = M['roi_traces'], M['time']; xlabel = 'Time from F1-OFF (s)'
#         vlines = M.get('isi_allowed', [])
#     elif align.upper() == 'F2':
#         X, T = M.get('roi_traces_F2locked'), M.get('time_F2locked'); xlabel = 'Time from F2 (s)'; vlines=None
#     elif align.upper() == 'CHOICE':
#         X, T = M.get('roi_traces_choice'), M.get('time_choice'); xlabel = 'Time from choice start (s)'; vlines=None
#     else:
#         raise ValueError('align must be F1OFF, F2, or CHOICE')
#     if X is None or T is None:
#         raise ValueError('alignment arrays missing—did you call the add_*_to_M helper?')

#     def _group_trace(mask, color, label):
#         m = np.asarray(mask, bool)
#         Y = nanmean_no_warn(X, axis=0)  # (Nroi,Nt)
#         y = nanmean_no_warn(Y[m], axis=0)
#         sem = np.nanstd(Y[m], axis=0) / np.sqrt(max(1, m.sum()))
#         for ridx in np.where(m)[0][:max_spaghetti]:
#             plt.plot(T, Y[ridx], alpha=0.15, lw=1, color=color)
#         plt.plot(T, y, lw=2.2, color=color, label=f"{label} (n={int(m.sum())})")
#         plt.fill_between(T, y-sem, y+sem, alpha=0.15, color=color)
#         # light gray verticals at allowed ISIs
#         for a in M.get('isi_allowed', []):
#             plt.axvline(a, color='k', ls=':', lw=0.6, alpha=0.35)
#         plt.ylabel('ΔF/F'); 
#         # plt.ylim(np.nanpercentile(X, 1), np.nanpercentile(X, 99))
#         # after computing Y (Nroi, Nt) and y/sem for each group,
#         # collect values for ylim:
#         vals_for_limits = []
#         vals_for_limits.append(Y[m].ravel())
#         lo, hi = np.nanpercentile(np.concatenate(vals_for_limits), [2, 98])
#         pad = (hi - lo) * 0.10
#         plt.ylim(lo - pad, hi + pad)



#     plt.figure(figsize=(9,3.2))
#     _group_trace(masks[group_a], 'C0', group_a)
#     if group_b is not None and group_b in masks:
#         _group_trace(masks[group_b], 'C1', group_b)
#     if vlines is not None and len(vlines):
#         for a in vlines: plt.axvline(a, color='k', ls=':', lw=0.6, alpha=0.5)
#     plt.xlabel(xlabel); plt.ylabel('ΔF/F'); plt.title(f'Group mean traces ({group_a}' + (f' vs {group_b}' if group_b else '') + ')')
#     plt.legend(frameon=False)
#     if save_path: plt.savefig(save_path, dpi=170, bbox_inches='tight')
#     return plt.gcf()

# ---------------------- driver ----------------------

def make_session_report1_and_group_traces(
    trial_data,
    report_path="session_report_v2.png",
    preF2_window=(0.15, 0.70),
    group_align='F1OFF',                 # 'F1OFF' | 'F2' | 'CHOICE'
    group_pairs=('signal','mixed'),
    group_fig_path=None
):
    # Build M (session_info-driven short/long)
    M = build_M_from_trials(trial_data, grid_bins=240)

    # (Optional) add aligned arrays for F2 / CHOICE if you’ll plot those
    if group_align.upper() == 'F2':
        M = add_F2_locked_to_M(M, trial_data, window=(-0.2, 0.6))
    if group_align.upper() == 'CHOICE':
        M = add_choice_locked_to_M(M, trial_data, window=(-0.6, 0.6))

    # Run the original pipeline
    R = run_all_from_raw(trial_data)

    # Report 1 (original one-pager)
    plot_session_report_v2(
        M, R, session_title="Mouse • Session • Date",
        save_path=report_path, compute_extras=True, preF2_window=preF2_window
    )

    # Group mean traces (equal-ISI + rewarded/licked + no DNC)
    plot_roi_group_traces(
        M, R, extras=None, groups=group_pairs, align=group_align,
        trial_filter=('licked','rewarded','no_dnc'),
        equalize_isi=True, f2_baseline_window=(-0.20,0.0),
        choice_baseline_window=(-0.60,-0.20), max_spaghetti=10
    )
    if group_fig_path:
        import matplotlib.pyplot as plt
        plt.savefig(group_fig_path, dpi=170, bbox_inches='tight')




def run_page1_reports(trial_data: Dict[str, Any], save_prefix: str | None = None):
    # Build + baseline
    M = build_M_from_trials(trial_data, grid_bins=240)
    M = baseline_correct_inplace(M, base_win=(0.0, 0.05))
    # Add choice + F2 locks
    M = add_choice_locked_to_M(M, trial_data, window=(-0.6, 0.6), grid_bins=240)
    M = add_F2_locked_to_M(M, trial_data, window=(-0.2, 0.6), grid_bins=160)
    # Analyses
    T = run_timing_analysis(M, preF2_window=None)
    choice_auc_F2 = decode_choice_F2locked_v2(M)
    # Report page 1
    fig = plot_session_report_v1(M, T, choice_auc_F2, session_title='Mouse • Session • Date', save_path=(save_prefix + '_page1.png') if save_prefix else None)
    # Group masks + an example group plot (signal vs mixed)
    masks = build_group_masks(M, T)
    fig_g = plot_roi_group_traces(M, masks, group_a='signal', group_b='mixed', align='F1OFF', save_path=(save_prefix + '_group_signal_vs_mixed.png') if save_prefix else None)
    return M, T, choice_auc_F2, masks, fig, fig_g


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



if __name__ == '__main__':
    print('Import and call run_page1_reports(...) or individual functions.')


# %%


    path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/sid_imaging_segmented_data.pkl'

    import pickle

    with open(path, 'rb') as f:
        trial_data = pickle.load(f)   # one object back (e.g., a dict)  
        
        
        
# %%


save_prefix="session01"

# Build + baseline
M = build_M_from_trials(trial_data, grid_bins=240)
# M = baseline_correct_inplace(M, base_win=(0.0, 0.05))


# %%

# Add choice + F2 locks
M = add_choice_locked_to_M(M, trial_data, window=(-0.6, 0.6), grid_bins=240)
M = add_F2_locked_to_M(M, trial_data, window=(-0.2, 0.6), grid_bins=160)
# Analyses
PRE_F2_WINDOW = (0.15, 0.70)
T = run_timing_analysis(M, preF2_window=PRE_F2_WINDOW)
choice_auc_F2 = decode_choice_F2locked_v2(M)

# %%

# Report page 1
fig = plot_session_report_v2(M, T, choice_auc_F2, session_title='Mouse * Session * Date', save_path=(save_prefix + '_page1.png') if save_prefix else None)

# %%

# Group masks + an example group plot (signal vs mixed)
masks = build_group_masks(M, T)
fig_g = plot_roi_group_traces(M, masks, group_a='signal', group_b='mixed', align='F1OFF', save_path=(save_prefix + '_group_signal_vs_mixed.png') if save_prefix else None)
plot_roi_group_traces(M, masks, 'scaling', 'clock', align='F1OFF', save_path='session01_group_scaling_vs_clock.png')
plot_roi_group_traces(M, masks, 'hazard_pos', 'signal', align='F1OFF', save_path='session01_group_hazard_vs_signal.png')
plot_roi_group_traces(M, masks, 'signal', 'mixed', align='F2', save_path='session01_group_signal_vs_mixed_F2.png')





# %%


    """Complete analysis pipeline from raw trial data to all metrics.
    
    This is the main entry point that runs all analyses in sequence.
    """
    print("========================================")
    print("STARTING COMPLETE SID TIMING ANALYSIS")
    print("========================================")
    
    
    report_path="session_report_v2.png"
    preF2_window=(0.15, 0.70)
    group_align='F1OFF'                # 'F1OFF' | 'F2' | 'CHOICE'
    group_pairs=('signal','mixed')
    group_fig_path=None

    
    # Step 1: Build canonical data structure
    M = build_M_from_trials(trial_data, grid_bins=240)
    
    # (Optional) add aligned arrays for F2 / CHOICE if you’ll plot those
    if group_align.upper() == 'F2':
        M = add_F2_locked_to_M(M, trial_data, window=(-0.2, 0.6))
    if group_align.upper() == 'CHOICE':
        M = add_choice_locked_to_M(M, trial_data, window=(-0.6, 0.6))    
    
    # Step 2: Scaling vs clock analysis
    delta, pref, (r2_phase, r2_ms) = scaling_vs_clock(M, cv_folds=5, knots_phase=8, knots_time=8)
    
    # Step 3: Time-resolved decoding
    decode = time_resolved_decoding(M, n_bins=20)
    
    # Step 4: Hazard rate analysis
    hazard_uv = hazard_unique_variance(M)
    
    # Step 5: Generate sort order for visualization
    order = sort_index(M, delta)
    
    # Compile results
    R = {
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
    
# %%
    
    # Report 1 (original one-pager)
    plot_session_report_v2(
        M, R, session_title="Mouse • Session • Date",
        save_path=report_path, compute_extras=True, preF2_window=preF2_window
    )    
    
    
    # Group mean traces (equal-ISI + rewarded/licked + no DNC)
    plot_roi_group_traces(
        M, R, extras=None, groups=group_pairs, align=group_align,
        trial_filter=('licked','rewarded','no_dnc'),
        equalize_isi=True, f2_baseline_window=(-0.20,0.0),
        choice_baseline_window=(-0.60,-0.20), max_spaghetti=10
    )    