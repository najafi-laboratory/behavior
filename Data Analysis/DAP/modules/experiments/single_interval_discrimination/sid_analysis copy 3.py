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

"""
Extras for Lobule V PC timing analysis — targeted tests for clock vs scaling vs hazard
and post‑F2 choice dynamics. Plug-in to the MVP (`lobuleV_mvp.py`).

Functions (all NaN-safe):
- trough_time_vs_isi(M): pre‑F2 trough/turning‑point time per ISI (F1‑OFF locked)
- decode_preF2_fairwindow(M, tmax=0.7, n_bins=12, use_slope=True): decoders in the fair window
- rsa_phase_vs_time(M, n_phase_bins=20): representational similarity (phase vs ms)
- hazard_unique_variance_per_roi(M, kernel=0.12): per‑ROI partial R^2 for discrete hazard bumps
- decode_choice_F2locked(M, n_bins=20): time‑resolved F2‑locked choice decoder (if arrays present)

All functions return small dicts that can be plotted or aggregated.
"""

from sklearn.decomposition import PCA

# -----------------------------
# Utilities
# -----------------------------

def _moving_average(x: np.ndarray, w: int = 5) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w, dtype=float) / w
    return np.convolve(x, k, mode='same')


def _bin_time(T: np.ndarray, n_bins: int, tmin: float, tmax: float) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(tmin, tmax, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _zscore_rows(Z: np.ndarray) -> np.ndarray:
    # Z: trials x features
    mu = np.nanmean(Z, axis=0, keepdims=True)
    sd = np.nanstd(Z, axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (Z - mu) / sd


# -----------------------------
# 1) Trough/turning‑point vs ISI (F1‑OFF locked)
# -----------------------------

def trough_time_vs_isi(M: Dict[str, Any], smooth_w: int = 7) -> Dict[str, Any]:
    """Estimate the time of minimum (trough) of a population projection for each ISI.

    NaN‑safe PCA: we fit PC1 on columns (time bins) that have no NaNs across ROIs.
    If too few such columns exist, we impute remaining NaNs by column means as a
    fallback. The ROI weight vector `w` is then applied to each ISI's mean trace
    to get a 1D population timecourse, and the trough is found in [0, ISI).

    Returns dict with arrays:
      'isi_levels' (K,), 'trough_time' (K,), 'trough_idx' (K,),
      plus simple model fits: constant vs linear trough_time ~ ISI.
    """
    X = M['roi_traces']   # (Ntr, Nroi, Nt)
    T = M['time']
    F2 = M['F2_time']
    Ntr, Nroi, Nt = X.shape

    # Trial‑averaged ROI x time (NaN‑safe)
    mean_rt = np.nanmean(X, axis=0)  # (Nroi, Nt)

    # --- Build ROI‑space PC1 with NaN handling ---
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)

    # Prefer strict columns (no NaNs across ROIs)
    strict_cols = np.all(np.isfinite(mean_rt), axis=0)
    if strict_cols.sum() >= 10:
        pca.fit(mean_rt[:, strict_cols].T)
    else:
        # Fallback: impute NaNs by column means and fit on all columns
        mean_rt_f = mean_rt.copy()
        col_means = np.nanmean(mean_rt_f, axis=0)
        nan_idx = ~np.isfinite(mean_rt_f)
        if np.any(nan_idx):
            mean_rt_f[nan_idx] = np.take(col_means, np.where(nan_idx)[1])
        pca.fit(mean_rt_f.T)

    w = pca.components_[0]  # (Nroi,)

    # Group trials by ISI level
    levels = np.sort(np.unique(F2[np.isfinite(F2)]))
    trough_time = np.full(levels.shape, np.nan)
    trough_idx = np.full(levels.shape, -1, dtype=int)

    for k, isi in enumerate(levels):
        sel = np.isclose(F2, isi, rtol=0, atol=1e-6)
        if sel.sum() == 0:
            continue
        avg = np.nanmean(X[sel], axis=0)  # (Nroi, Nt)
        # Population projection across all time bins
        pop = np.dot(w, avg)  # (Nt,)
        # Smooth lightly for robustness
        pop = _moving_average(pop, smooth_w)
        # Valid window for this ISI: [0, isi)
        valid = (T >= 0) & (T < isi) & np.isfinite(pop)
        if not np.any(valid):
            continue
        idx_local = np.nanargmin(pop[valid])
        global_idx = np.arange(Nt)[valid][idx_local]
        trough_idx[k] = global_idx
        trough_time[k] = T[global_idx]

    # Simple model comparison: trough_time ~ constant vs ~ a + b*ISI
    y = trough_time
    x = levels
    finite = np.isfinite(x) & np.isfinite(y)
    const_pred = np.full_like(y, np.nan)
    lin_pred = np.full_like(y, np.nan)
    r2_const = np.nan
    r2_lin = np.nan
    if finite.sum() >= 3:
        yv, xv = y[finite], x[finite]
        const = float(np.nanmean(yv))
        const_pred[finite] = const
        r2_const = 1.0 - np.nanmean((yv - const) ** 2) / (np.nanvar(yv) + 1e-12)
        A = np.column_stack([np.ones_like(xv), xv])
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
        lin = A @ beta
        lin_pred[finite] = lin
        r2_lin = 1.0 - np.nanmean((yv - lin) ** 2) / (np.nanvar(yv) + 1e-12)

    return {
        'isi_levels': levels,
        'trough_time': trough_time,
        'trough_idx': trough_idx,
        'r2_constant': float(r2_const),
        'r2_linear': float(r2_lin),
        'constant_pred': const_pred,
        'linear_pred': lin_pred,
    }

# -----------------------------
# 2) Fair‑window pre‑F2 decoders (amplitude vs slope)
# -----------------------------

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

# -----------------------------
# 5) F2‑locked choice decoder (optional)
# -----------------------------

# def decode_choice_F2locked(M: Dict[str, Any], n_bins: int = 20) -> Dict[str, Any] | None:
#     """Time‑resolved AUC(τ) for is_right_choice using F2‑locked traces.
#     Returns None if required arrays are missing.
#     """
#     if 'roi_traces_F2locked' not in M or 'time_F2locked' not in M:
#         return None
#     X = M['roi_traces_F2locked']
#     T = M['time_F2locked']
#     side = M.get('is_right_choice')
#     if side is None:
#         return None
#     Ntr, Nroi, Nt = X.shape

#     # Bin τ
#     edges, centers = _bin_time(T, n_bins, T[0], T[-1])
#     auc_t = np.full(len(centers), np.nan)
#     # Build per‑bin Z features and decode
#     for b in range(len(centers)):
#         m = (T >= edges[b]) & (T < edges[b+1])
#         if not np.any(m):
#             continue
#         Z = np.nanmean(X[:, :, m], axis=2)
#         valid = np.all(np.isfinite(Z), axis=1)
#         if valid.sum() < 12 or np.unique(side[valid]).size < 2:
#             continue
#         Zs = np.full_like(Z, np.nan)
#         Zs[valid] = StandardScaler().fit_transform(Z[valid])
#         # CV AUC
#         y = side.astype(int)
#         yv = y[valid]
#         Zv = Zs[valid]
#         kf = KFold(n_splits=min(5, len(yv)), shuffle=True, random_state=0)
#         scores = []
#         for tr, te in kf.split(Zv):
#             if np.unique(yv[te]).size < 2:
#                 continue
#             clf = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, max_iter=1000)
#             clf.fit(Zv[tr], yv[tr])
#             p = clf.predict_proba(Zv[te])[:, 1]
#             scores.append(roc_auc_score(yv[te], p))
#         auc_t[b] = float(np.nanmean(scores)) if scores else np.nan

#     return {'time_points': centers, 'auc_t': auc_t}


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
Updated one‑pager report for Lobule V PC timing analysis.

Main API
--------
plot_session_report_v2(M, R, session_title=None, save_path=None, compute_extras=True, extras=None)

- M: canonical session dict from `build_M_from_trials` (lobuleV_mvp)
- R: results dict from `run_all_from_raw`
- extras: optional dict containing precomputed outputs from lobuleV_extras:
    {
      'trough': trough_time_vs_isi(M),
      'fair': decode_preF2_fairwindow(M, ...),
      'rsa': rsa_phase_vs_time(M),
      'hz_roi': hazard_unique_variance_per_roi(M, ...),
      'choice': decode_choice_F2locked(M)
    }
  If compute_extras=True (default), these will be computed on the fly.

Panels (auto-skip if unavailable)
----------------------------------
Row 1
  [1] ΔR² histogram + fractions
  [2] Time‑resolved decode (ISI R²(t), AUC(t), divergence)
  [3] Hazard: session unique R² (+ per‑ROI fraction & bump weights if provided)
Row 2
  [4] Trough/turning‑point time vs ISI (constant vs linear fits)
  [5] Fair‑window decoders (amp vs slope) for t < 0.7 s
  [6] RSA summary (bars of mean corr + optional heatmaps thumbnails)
Row 3
  [7] F2‑locked choice AUC(τ) (with optional behavior overlays)
  [8-9] Sorted raster (avg across trials)

All plots are NaN‑safe and degrade gracefully if inputs are missing.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Optional imports (extras). Safe if not installed in user context; guarded in code.



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

        # out['choice'] = decode_choice_F2locked_v2(M)
        
        out['choice'] = decode_choice_choiceLocked_v3(M)  # pre-choice AUC by default
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


def session_summary(M, R, extras):
    summ = {}
    thr = extras.get('trough', {})
    summ['phi_trough_median'] = thr.get('phi_median')
    summ['phi_trough_IQR']    = (thr.get('phi_q25'), thr.get('phi_q75'))
    summ['trough_R2_linear']  = thr.get('r2_linear')
    summ['trough_R2_const']   = thr.get('r2_constant')

    rsa = extras.get('rsa', {})
    summ['RSA_abs']  = rsa.get('mean_corr_ms')
    summ['RSA_phase']= rsa.get('mean_corr_phase')

    hz  = extras.get('hz_roi', {})
    summ['hazard_session_UV']  = R.get('hazard_unique_r2')
    summ['hazard_frac_ROIs>0'] = hz.get('fraction_positive')
    summ['hazard_peak_weight'] = (hz.get('levels', [None])[int(np.nanargmax(hz.get('mean_bump_weights', [np.nan])))]
                                  if hz.get('mean_bump_weights') is not None else None)

    choice = extras.get('choice', {})
    if choice:
        t, a = np.asarray(choice['time_points']), np.asarray(choice['auc_t'])
        def _div(t, s, thr=0.6, sustain=3):
            run = 0
            for i,v in enumerate(s):
                if np.isfinite(v) and v>thr:
                    run += 1
                    if run>=sustain: return t[i-sustain+1]
                else: run = 0
            return np.nan
        summ['choice_divergence_s'] = _div(t, a)
        summ['servo_in_med_s'] = np.nanmedian(M.get('servo_in_relF2'))
        summ['first_lick_med_s'] = np.nanmedian(M.get('lick_start_relF2'))
    return summ


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


def plot_roi_group_traces(M: dict, R: dict, extras: dict | None = None,
                          groups: tuple[str, ...] = ('signal','mixed'),
                          align: str = 'F1OFF',
                          show_sem: bool = True,
                          max_spaghetti: int = 0,
                          colors: dict | None = None,
                          title: str | None = None):
    """Plot mean ROI traces for selected groups (e.g., 'signal' vs 'mixed').

    Parameters
    ----------
    groups : names from get_roi_group_masks keys
    align  : 'F1OFF' (uses M['roi_traces'], M['time']) or 'F2' (requires F2‑locked)
    max_spaghetti : overlay up to N individual ROI means per group
    """
    masks = get_roi_group_masks(M, R, extras)
    if colors is None:
        colors = {'signal':'C0', 'mixed':'C1', 'scaling':'C2', 'clock':'C3', 'hazard_pos':'C4', 'choice_mod':'C5'}

    if align.upper() == 'F2':
        X = M.get('roi_traces_F2locked'); T = M.get('time_F2locked')
        xlabel = 'Time from F2 (s)'
    elif align.upper() == 'CHOICE':
        X = M.get('roi_traces_choice'); T = M.get('time_choice')
        xlabel = 'Time from choice start (s)'        
    else:
        X = M['roi_traces']; T = M['time']
        xlabel = 'Time from F1-OFF (s)'
    assert X is not None and T is not None, 'Required arrays not present.'

    # Precompute ROI means (avg over trials)
    roi_mean = np.nanmean(X, axis=0)  # (Nroi, Nt)

    plt.figure(figsize=(9,4))
    ax = plt.gca()
    for g in groups:
        m = masks.get(g)
        if m is None or m.sum() == 0:
            continue
        Y = roi_mean[m]  # (Ng, Nt)
        mu = np.nanmean(Y, axis=0)
        if show_sem:
            sd = np.nanstd(Y, axis=0)
            n  = np.maximum(1, np.sum(np.isfinite(Y), axis=0))
            se = sd / np.sqrt(n)
            ax.fill_between(T, mu-se, mu+se, alpha=0.15, color=colors.get(g,'C0'))
        ax.plot(T, mu, lw=2, label=f"{g} (n={m.sum()})", color=colors.get(g,'C0'))
        if max_spaghetti > 0:
            k = min(max_spaghetti, Y.shape[0])
            idx = np.linspace(0, Y.shape[0]-1, k, dtype=int)
            for ii in idx:
                ax.plot(T, Y[ii], lw=0.6, alpha=0.4, color=colors.get(g,'C0'))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('ΔF/F')
    ax.set_title(title or f"Group mean traces ({' vs '.join(groups)})")
    ax.legend(frameon=True)
    if 'isi_allowed' in M and align.upper()=='F1OFF':
        for isi in np.asarray(M['isi_allowed']):
            ax.axvline(isi, color='k', lw=0.6, ls=':', alpha=0.5)
    plt.tight_layout()
    return masks

# %%


"""
Page 2: ROI-group diagnostics for Lobule V PC timing analysis

What this renders (auto-skips panels if inputs are missing):
  [1] ROI group counts (signal, mixed, scaling, clock, hazard+, choice_mod)
  [2] Group mean traces (F1-OFF): signal vs mixed
  [3] Group mean traces (F2-locked): choice_mod vs mixed
  [4] Effect sizes (Δ pre-F2 slope; Δ post-F2 amplitude) with bootstrap CIs
  [5] Transient-free ΔR² histogram (0.15–0.70 s)
  [6] Scatter: per-ROI pre-F2 slope vs post-F2 amplitude, color by group

API
---
plot_session_page2(M, R, extras=None, save_path=None,
                   preF2_window=(0.15,0.70), postF2_window=(0.50,0.60),
                   eps=0.02, choice_window=(0.05,0.25),
                   choice_es_thresh=0.5, hazard_thresh=0.0)

Depends on helpers in lobuleV_report_v2 and lobuleV_extras.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ------------------------ utils ------------------------

def _bootstrap_ci(x: np.ndarray, fn, n=1000, alpha=0.05):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, (np.nan, np.nan)
    vals = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        samp = rng.choice(x, size=x.size, replace=True)
        vals.append(fn(samp))
    vals = np.asarray(vals)
    return float(np.nanmean(vals)), (float(np.nanpercentile(vals, 100*alpha/2)),
                                      float(np.nanpercentile(vals, 100*(1-alpha/2))))

def _bootstrap_diff_groups(a: np.ndarray, b: np.ndarray, fn=lambda x: np.nanmean(x), n=2000, alpha=0.05):
    """Bootstrap CI for difference of a summary between two groups: fn(a) - fn(b).
    Resamples within each group with replacement. Returns (diff_mean, (lo, hi))."""
    a = np.asarray(a); b = np.asarray(b)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n):
        sa = rng.choice(a, size=a.size, replace=True)
        sb = rng.choice(b, size=b.size, replace=True)
        vals.append(fn(sa) - fn(sb))
    vals = np.asarray(vals)
    return float(np.nanmean(vals)), (float(np.nanpercentile(vals, 100*alpha/2)),
                                      float(np.nanpercentile(vals, 100*(1-alpha/2))))


def _pre_slope_per_roi(M: dict, window=(0.15,0.70)):
    X, T = M['roi_traces'], M['time']
    roi_mean = np.nanmean(X, axis=0)  # (Nroi, Nt)
    m = (T>=window[0]) & (T<=window[1])
    t = T[m]
    A = np.column_stack([np.ones_like(t), t])
    # Solve for each ROI via lstsq on (t, y)
    beta = np.linalg.lstsq(A, roi_mean[:, m].T, rcond=None)[0]  # (2, Nroi)
    return beta[1]  # slope per ROI


def _post_amp_per_roi(M: dict, window=(0.50,0.60)):
    X2, T2 = M.get('roi_traces_F2locked'), M.get('time_F2locked')
    if X2 is None or T2 is None:
        return None
    m = (T2>=window[0]) & (T2<=window[1])
    if not np.any(m):
        return None
    # mean over trials, then time
    amp = np.nanmean(np.nanmean(X2[:, :, m], axis=2), axis=0)  # (Nroi,)
    return amp


# ------------------------ main page ------------------------

def plot_session_page2(M: dict, R: dict, extras: dict | None = None, save_path: str | None = None,
                       preF2_window=(0.15,0.70), postF2_window=(0.50,0.60),
                       eps: float = 0.02, choice_window=(0.05,0.25),
                       choice_es_thresh: float = 0.5, hazard_thresh: float = 0.0,
                       servo_auc_bins: int = 24, servo_pre_window=(-0.3, 0.0)):
    """Render Page 2 with ROI-group diagnostics. Returns (fig, masks dict)."""
    Nroi = M['roi_traces'].shape[1]
    # 1) Build masks (uses ΔR² from R; you can pass R with windowed ΔR² if desired)
    masks = get_roi_group_masks(M, R, extras,
                                eps=eps,
                                choice_window=choice_window,
                                choice_es_thresh=choice_es_thresh,
                                hazard_thresh=hazard_thresh)

    # Precompute ROI features
    slope = _pre_slope_per_roi(M, window=preF2_window)
    amp   = _post_amp_per_roi(M, window=postF2_window)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 3, height_ratios=[1.0, 1.1, 1.1, 1.0], wspace=0.35, hspace=0.35)

    # [1] Counts bar
    ax1 = fig.add_subplot(gs[0, 0])
    groups = ['signal','mixed','scaling','clock','hazard_pos','choice_mod']
    counts = []
    for g in groups:
        m = masks.get(g, np.zeros(Nroi, bool))
        counts.append(int(np.sum(m)))
    ax1.bar(groups, counts)
    ax1.set_ylabel('ROIs')
    ax1.set_title('ROI group counts')
    ax1.tick_params(axis='x', rotation=30)
    for i, v in enumerate(counts):
        ax1.text(i, v, f" {v}", va='bottom', ha='left')

    # [2] Group means F1-OFF: signal vs mixed
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        plot_roi_group_traces(M, R, extras, groups=('signal','mixed'), align='F1OFF', max_spaghetti=8)
        ax2.set_title('Group means (F1-OFF): signal vs mixed')
    except Exception:
        ax2.text(0.5, 0.5, 'F1-OFF traces N/A', va='center', ha='center')
    # [3] Group means F2-locked: choice_mod vs mixed
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        plot_roi_group_traces(M, R, extras, groups=('choice_mod','mixed'), align='F2', max_spaghetti=6)
        ax3.set_title('Group means (F2): choice_mod vs mixed')
    except Exception:
        ax3.text(0.5, 0.5, 'F2-locked traces N/A', va='center', ha='center')

    # [4] Effect sizes with CIs
    ax4 = fig.add_subplot(gs[1, 0])
    def _safe_bar_with_ci(ax, labels, diffs, cis):
        y = np.array([diffs.get(k, np.nan) for k in labels], dtype=float)
        yerr = np.zeros((2, len(labels)), dtype=float) * np.nan
        for i, k in enumerate(labels):
            lo, hi = cis.get(k, (np.nan, np.nan))
            if np.isfinite(y[i]) and np.isfinite(lo) and np.isfinite(hi):
                yerr[0, i] = y[i] - lo
                yerr[1, i] = hi - y[i]
        ax.bar(labels, y, yerr=yerr, capsize=4)
        ax.axhline(0, color='k', lw=1)

    diffs = {}
    cis = {}
    # Δ pre-F2 slope (signal - mixed)
    a = slope[masks.get('signal', np.zeros(Nroi,bool))]
    b = slope[masks.get('mixed',  np.zeros(Nroi,bool))]
    d_pre, ci_pre = _bootstrap_diff_groups(a, b)
    diffs['Δ pre-slope'] = d_pre; cis['Δ pre-slope'] = ci_pre
    # Δ post-F2 amp (signal - mixed)
    if amp is not None:
        a2 = amp[masks.get('signal', np.zeros(Nroi,bool))]
        b2 = amp[masks.get('mixed',  np.zeros(Nroi,bool))]
        d_post, ci_post = _bootstrap_diff_groups(a2, b2)
        diffs['Δ post-amp'] = d_post; cis['Δ post-amp'] = ci_post
    _safe_bar_with_ci(ax4, ['Δ pre-slope', 'Δ post-amp'], diffs, cis)
    ax4.set_title('Signal − Mixed (bootstrap CI)')

    # [5] Transient-free ΔR² histogram (if available)
    ax5 = fig.add_subplot(gs[1, 1])
    if scaling_vs_clock_windowed is not None:
        delta_w, pref_w, _ = scaling_vs_clock_windowed(M, time_window=preF2_window)
        bins = np.linspace(np.nanmin(delta_w) if np.isfinite(np.nanmin(delta_w)) else -0.5,
                           np.nanmax(delta_w) if np.isfinite(np.nanmax(delta_w)) else 0.5, 31)
        ax5.hist(delta_w[np.isfinite(delta_w)], bins=bins, edgecolor='none')
        ax5.axvline(0, color='k', ls='--', lw=1)
        ax5.set_title(f'Transient-free ΔR² [{preF2_window[0]:.2f},{preF2_window[1]:.2f}] s')
        med = np.nanmedian(delta_w)
        ax5.axvline(med, color='k', lw=1.5)
        ax5.text(0.98, 0.98, f"median={med:.3f}", transform=ax5.transAxes, ha='right', va='top')
    else:
        ax5.text(0.5, 0.5, 'ΔR²(windowed) N/A', va='center', ha='center')

    # [6] Scatter: pre-slope vs post-amp color by group
    ax6 = fig.add_subplot(gs[1, 2])
    if amp is not None:
        col = np.array(['#bbbbbb'] * Nroi, dtype=object)
        if 'signal' in masks:     col[masks['signal']] = 'C0'
        if 'hazard_pos' in masks: col[masks['hazard_pos']] = 'C4'
        if 'choice_mod' in masks: col[masks['choice_mod']] = 'C5'
        valid = np.isfinite(slope) & np.isfinite(amp)
        ax6.scatter(slope[valid], amp[valid], s=14, c=col[valid], alpha=0.7, edgecolors='none')
        ax6.set_xlabel('pre-F2 slope (ΔF/F per s)')
        ax6.set_ylabel('post-F2 amplitude (ΔF/F)')
        ax6.set_title('ROI features: slope vs amplitude (colored by group)')
    else:
        ax6.text(0.5, 0.5, 'F2-locked arrays N/A', va='center', ha='center')

    # [7] Servo-locked pre-choice decode (AUC)
    ax7 = fig.add_subplot(gs[2, 0])
    try:
        ser = decode_choice_servo_locked_v2(M, n_bins=servo_auc_bins, restrict_preServo=True)
        if ser is not None:
            tt, aa = np.asarray(ser['time_points']), np.asarray(ser['auc_t'])
            ax7.plot(tt, aa, lw=2)
            ax7.axhline(0.5, color='k', ls='--', lw=1)
            ax7.set_title('Servo-locked left/right AUC (pre-choice)')
            ax7.set_xlabel('Time from servo (s)')
            ax7.set_ylabel('AUC')
    except Exception:
        ax7.text(0.5, 0.5, 'Servo-locked decode N/A', va='center', ha='center')

    # [8] Timer x Choice overlap (servo-pre)
    ax8 = fig.add_subplot(gs[2, 1])
    try:
        d = roi_choice_d_servo_pre(M, window=servo_pre_window)
        ov = timing_choice_overlap(R, d, timing_eps=eps, d_thresh=0.5)
        if ov is not None:
            lab = ['timer', 'choice', 'overlap']
            vals = [ov['K_timer'], ov['M_choice'], ov['overlap']]
            ax8.bar(lab, vals)
            ax8.set_title(f"Timer ∩ Choice (p={ov['p_hypergeom']:.3g})")
    except Exception:
        ax8.text(0.5, 0.5, 'Overlap N/A', va='center', ha='center')

    # [9] ΔR²(windowed) vs pre-choice d (per ROI)
    ax9 = fig.add_subplot(gs[2, 2])
    try:
        if scaling_vs_clock_windowed is None:
            raise RuntimeError('scaling_vs_clock_windowed unavailable')
        delta_w, _, _ = scaling_vs_clock_windowed(M, time_window=preF2_window)
        d = roi_choice_d_servo_pre(M, window=servo_pre_window)
        valid = np.isfinite(delta_w) & np.isfinite(d)
        ax9.scatter(delta_w[valid], d[valid], s=12, alpha=0.7)
        ax9.axvline(0, color='k', ls=':'); ax9.axhline(0, color='k', ls=':')
        ax9.set_xlabel('ΔR² (phase − ms) [pre-F2 window]')
        ax9.set_ylabel("Cohen's d (R−L) pre-servo")
        ax9.set_title('Timer index vs pre-choice side effect per ROI')
    except Exception:
        ax9.text(0.5, 0.5, 'Scatter N/A', va='center', ha='center')

    # [10] Legend key (colors)
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')
    ax10.text(0.01, 0.6, 'Groups: signal=C0, hazard+=C4, choice_mod=C5; mixed=gray', fontsize=9)


    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, masks






# ---- Servo (choice_start) alignment + decoders + overlap ----

def add_servo_locked_to_M(M: Dict[str, Any], trial_data: Dict[str, Any],
                          window: tuple[float,float] = (-0.6, 0.6), grid_bins: int = 240) -> Dict[str, Any]:
    """Add ROI×time arrays aligned to `servo_in` / `choice_start`.
    Stores:
      - M['roi_traces_servo'] : (Ntr, Nroi, Nt)
      - M['time_servo']       : (Nt,)
      - M['lick_start_relServo'] : (Ntr,) if available
    """
    df = trial_data['df_trials_with_segments']
    n_trials = len(df)
    first = df.iloc[0]
    dff0 = np.asarray(first['dff_segment'], dtype=float)
    if dff0.ndim != 2:
        raise ValueError("dff_segment must be 2D (n_rois x n_samples) per trial")
    n_rois = dff0.shape[0]
    tau = np.linspace(window[0], window[1], grid_bins)
    Xs = np.full((n_trials, n_rois, grid_bins), np.nan, dtype=np.float32)
    lick_rel = np.full(n_trials, np.nan, dtype=float)
    
    
    for idx, (trial_idx, row) in enumerate(df.iterrows()):
        servo = float(row.get('servo_in', row.get('choice_start')))
        t_vec = np.asarray(row['dff_time_vector'], dtype=float) - servo
        dff = np.asarray(row['dff_segment'], dtype=float)
        # Ensure strictly increasing time for interpolation
        if not np.all(np.diff(t_vec) > 0):
            order = np.argsort(t_vec)
            t_vec = t_vec[order]
            dff = dff[:, order]
        t_min, t_max = np.nanmin(t_vec), np.nanmax(t_vec)
        for r in range(n_rois):
            y = dff[r]
            vals_r = np.interp(tau, t_vec, y, left=np.nan, right=np.nan)
            # keep NaN outside the trial's support
            out = (tau < t_min) | (tau > t_max)
            vals_r[out] = np.nan
            Xs[idx, r] = vals_r
        # relative first-lick time to servo, if present
        try:
            lick_rel[idx] = float(row.get('lick_start', np.nan)) - servo
        except Exception:
            lick_rel[idx] = np.nan

    M['roi_traces_servo'] = Xs
    M['time_servo'] = tau.astype(np.float32)
    M['lick_start_relServo'] = lick_rel.astype(np.float32)
    return M


def _stratified_kfold_by_side_and_isi(side: np.ndarray, isi: np.ndarray, n_splits: int = 5, random_state: int = 0):
    """Stratify by the joint (side, isi_level) label for fair pre-choice decoding."""
    side = side.astype(int)
    # Map ISI values to integer levels (stable order)
    levels = np.unique(isi[np.isfinite(isi)])
    idx = np.searchsorted(levels, isi)
    y_joint = side * max(1, levels.size) + idx
    # Use StratifiedKFold on y_joint
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=min(n_splits, len(side)), shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros_like(y_joint), y_joint))


def decode_choice_servo_locked_v2(M: Dict[str, Any], n_bins: int = 24,
                                  baseline_window: tuple[float,float] | None = (-0.6, -0.2),
                                  zscore_within_fold: bool = True,
                                  restrict_preServo: bool = True,
                                  stratify_by_isi: bool = True) -> Dict[str, Any] | None:
    """Time-resolved left-vs-right decode around servo/choice start.
    Uses within-fold scaling and optional stratification by ISI.
    If restrict_preServo=True, only τ<0 contributes.
    """
    if 'roi_traces_servo' not in M or 'time_servo' not in M or M.get('is_right_choice') is None:
        return None
    X = M['roi_traces_servo'].copy()  # (Ntr,Nroi,Nt)
    T = M['time_servo']
    side = M['is_right_choice'].astype(int)
    isi = M.get('F2_time')  # per-trial ISI

    # Baseline subtraction
    if baseline_window is not None:
        b0, b1 = baseline_window
        bm = (T >= b0) & (T <= b1)
        if np.any(bm):
            base = np.nanmean(X[:, :, bm], axis=2, keepdims=True)
            X = X - base

    # Optionally restrict to pre-Servo
    if restrict_preServo:
        mask_t = T < 0
        T = T[mask_t]
        X = X[:, :, mask_t]

    # Bin time
    edges = np.linspace(T[0], T[-1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    auc_t = np.full(len(centers), np.nan)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Precompute folds
    if stratify_by_isi and isi is not None:
        folds = _stratified_kfold_by_side_and_isi(side, np.asarray(isi), n_splits=5)
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        folds = list(kf.split(np.arange(len(side))))

    for b in range(len(centers)):
        m = (T >= edges[b]) & (T < edges[b+1])
        if not np.any(m):
            continue
        Z = np.nanmean(X[:, :, m], axis=2)  # (Ntr, Nroi)
        valid = np.all(np.isfinite(Z), axis=1)
        if valid.sum() < 12 or np.unique(side[valid]).size < 2:
            continue
        Z = Z[valid]; y = side[valid]
        scores = []
        for tr, te in folds:
            tr = np.intersect1d(tr, np.where(valid)[0])
            te = np.intersect1d(te, np.where(valid)[0])
            if tr.size < 10 or te.size < 4:
                continue
            Ztr, Zte = Z[tr], Z[te]
            ytr, yte = y[tr], y[te]
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


def roi_choice_d_servo_pre(M: Dict[str, Any], window: tuple[float,float] = (-0.3, 0.0)) -> np.ndarray | None:
    """Per-ROI Cohen's d for right vs left in a pre-servo window."""
    Xs, Ts = M.get('roi_traces_servo'), M.get('time_servo')
    side = M.get('is_right_choice')
    if Xs is None or Ts is None or side is None:
        return None
    m = (Ts >= window[0]) & (Ts <= window[1])
    if not np.any(m):
        return None
    feat = np.nanmean(Xs[:, :, m], axis=2)  # (Ntr, Nroi)
    side = side.astype(bool)
    a = feat[ side]; b = feat[~side]
    ma = np.nanmean(a, axis=0); mb = np.nanmean(b, axis=0)
    va = np.nanvar(a, axis=0, ddof=1); vb = np.nanvar(b, axis=0, ddof=1)
    na = np.sum(np.isfinite(a), axis=0); nb = np.sum(np.isfinite(b), axis=0)
    sp = np.sqrt(((na-1)*va + (nb-1)*vb) / np.maximum(na+nb-2, 1))
    d = (ma - mb) / (sp + 1e-9)
    d[~np.isfinite(d)] = np.nan
    return d


def timing_choice_overlap(R: Dict[str, Any], choice_d: np.ndarray, timing_eps: float = 0.02, d_thresh: float = 0.5):
    """Quantify overlap between timer-like ROIs (|ΔR²|>eps) and choice-modulated ROIs (|d|>d_thresh).
    Returns counts and a hypergeometric p-value for overlap ≥ observed.
    """
    delta = np.asarray(R.get('delta_r2'))
    if delta.ndim == 0:
        return None
    N = delta.size
    timer = np.isfinite(delta) & (np.abs(delta) > timing_eps)
    choice = np.isfinite(choice_d) & (np.abs(choice_d) > d_thresh)
    K = int(timer.sum()); M_ = int(choice.sum()); k = int((timer & choice).sum())
    # Hypergeometric tail p-value
    from math import comb
    def _hyp_pval(N, K, M, k):
        denom = comb(N, M)
        s = 0.0
        for i in range(k, min(K, M) + 1):
            s += comb(K, i) * comb(N - K, M - i)
        return float(s / denom) if denom > 0 else np.nan
    p = _hyp_pval(N, K, M_, k)
    return {'N':N, 'K_timer':K, 'M_choice':M_, 'overlap':k, 'p_hypergeom':p,
            'timer_mask': timer, 'choice_mask': choice}



# --- add to lobuleV_extras.py ---

def _choice_start_time(row):
    cs = row.get('choice_start', None)
    if cs is None:
        cs = row.get('servo_in', None)  # legacy alias, if present
    return float(cs) if cs is not None else float('nan')


def add_choice_locked_to_M(M, trial_data, window=(-0.6, 0.6), grid_bins=240):
    """Build M['roi_traces_choice'], M['time_choice'], M['lick_start_relChoice'].
       Time is in seconds relative to choice_start."""
    df = trial_data['df_trials_with_segments']
    n_trials = len(df)
    first = df.iloc[0]
    n_rois = int(np.asarray(first['dff_segment']).shape[0])
    tau = np.linspace(window[0], window[1], grid_bins)

    Xc = np.full((n_trials, n_rois, grid_bins), np.nan, dtype=np.float32)
    lick_rel = np.full(n_trials, np.nan, dtype=float)
    for idx, (trial_idx, row) in enumerate(df.iterrows()):
        cs = _choice_start_time(row)
        t_vec = np.asarray(row['dff_time_vector'], float) - cs
        dff   = np.asarray(row['dff_segment'], float)
        if not np.all(np.diff(t_vec) > 0):
            order = np.argsort(t_vec)
            t_vec = t_vec[order]; dff = dff[:, order]
        tmin, tmax = np.nanmin(t_vec), np.nanmax(t_vec)
        for r in range(n_rois):
            vals = np.interp(tau, t_vec, dff[r], left=np.nan, right=np.nan)
            vals[(tau < tmin) | (tau > tmax)] = np.nan
            Xc[idx, r] = vals
        if row.get('lick_start') is not None and np.isfinite(cs):
            lick_rel[idx] = float(row['lick_start']) - cs

    M['roi_traces_choice']   = Xc
    M['time_choice']         = tau.astype(np.float32)
    M['lick_start_relChoice']= lick_rel.astype(np.float32)
    return M

# --- add to lobuleV_extras.py ---

# lobuleV_extras.py
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




def roi_choice_d_preChoice(M, window=(-0.3, 0.0)):
    Xc, Tc = M.get('roi_traces_choice'), M.get('time_choice')
    side = M.get('is_right_choice')
    if Xc is None or Tc is None or side is None:
        return None
    m = (Tc >= window[0]) & (Tc <= window[1])
    if not np.any(m): return None
    feat = np.nanmean(Xc[:, :, m], axis=2)
    side = side.astype(bool)
    a, b = feat[ side], feat[~side]
    ma, mb = np.nanmean(a, axis=0), np.nanmean(b, axis=0)
    va, vb = np.nanvar(a, axis=0, ddof=1), np.nanvar(b, axis=0, ddof=1)
    na, nb = np.sum(np.isfinite(a), axis=0), np.sum(np.isfinite(b), axis=0)
    sp = np.sqrt(((na-1)*va + (nb-1)*vb) / np.maximum(na+nb-2, 1))
    d = (ma - mb) / (sp + 1e-9)
    d[~np.isfinite(d)] = np.nan
    return d


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



R = run_all_from_raw(trial_data)

# %%

trough = trough_time_vs_isi(M)
fair    = decode_preF2_fairwindow(M, tmax=0.7, use_slope=True)
rsa     = rsa_phase_vs_time(M)
hz_roi  = hazard_unique_variance_per_roi(M, kernel=0.12)
choice  = decode_choice_F2locked(M)   # returns None if F2-locked not present


# %%


# Build canonical M and baseline results
M = build_M_from_trials(trial_data)
R = run_all_from_raw(trial_data)

# (A) Windowed ΔR² to exclude the early F1-OFF transient
delta_w, pref_w, (r2p_w, r2ms_w) = scaling_vs_clock_windowed(M, time_window=(0.15, 0.70))


# %%
# (B) Add F2-locked arrays, then compute choice AUC(τ)
M = add_F2_locked_to_M(M, trial_data, window=(-0.2, 0.6))
# choice = decode_choice_F2locked(M)  # returns dict with 'time_points' and 'auc_t'

# Robust choice AUC with baseline subtraction and in-fold scaling
choice = decode_choice_F2locked_v2(
    M,
    n_bins=20,
    baseline_window=(-0.2, -0.05),   # None to disable
    zscore_within_fold=True,
    restrict_preF2=False              # True => compute only for τ<0
)

# %%
# (C) Updated report with the pre-F2 window applied to RSA
fig = plot_session_report_v2(
    M, R,
    session_title="Mouse • Session • Date",
    save_path="session_report_v2.png",
    compute_extras=True,
    preF2_window=(0.15, 0.70)
)


# %%


extras = _maybe_compute_extras(M, compute_extras=True, extras=extras, preF2_window=(0.15, 0.70))


S = session_summary(M, R, extras)  # where `extras` is what report computed
S

# %%


masks = get_roi_group_masks(M, R, extras)




groups=('choice_mod','mixed')
plot_roi_group_traces(M, R, extras, groups=groups, align='F1OFF', max_spaghetti=10)
# 3) Same comparison but F2-locked (helpful to see post-F2 differences)
plot_roi_group_traces(M, R, extras, groups=groups, align='F2')

groups=('signal','mixed')
plot_roi_group_traces(M, R, extras, groups=groups, align='F1OFF', max_spaghetti=10)



plot_roi_group_traces(M, R, extras, groups=groups, align='F2', max_spaghetti=10)


groups=('choice_mod','mixed') #→ which ROIs carry post-F2 choice info vs the rest.
plot_roi_group_traces(M, R, extras, groups=groups, align='F1OFF', max_spaghetti=10)
plot_roi_group_traces(M, R, extras, groups=groups, align='F2', max_spaghetti=10)


groups=('hazard_pos','signal') #→ anticipatory vs timer-like.
plot_roi_group_traces(M, R, extras, groups=groups, align='F1OFF', max_spaghetti=10)
plot_roi_group_traces(M, R, extras, groups=groups, align='F2', max_spaghetti=10)

groups=('scaling','clock') #→ head-to-head timer stories.
plot_roi_group_traces(M, R, extras, groups=groups, align='F1OFF', max_spaghetti=10)
plot_roi_group_traces(M, R, extras, groups=groups, align='F2', max_spaghetti=10)


# %%

delta_w, pref_w, _ = scaling_vs_clock_windowed(M, time_window=(0.15, 0.70))
R_w = {'delta_r2': delta_w, 'model_pref': pref_w}

masks = get_roi_group_masks(M, R_w, extras, eps=0.01)  # slightly looser eps

# 2) Quick group summary metrics (pre/post windows)
import numpy as np

def group_metrics(M, mask, pre=(0.15,0.70), post=(0.50,0.60)):
    X, T = M['roi_traces'], M['time']
    X2, T2 = M.get('roi_traces_F2locked'), M.get('time_F2locked')
    # pre-F2 slope (timer strength)
    mpre = (T>=pre[0]) & (T<=pre[1])
    roi_mean = np.nanmean(X, axis=0)[mask]                 # (nROI, Nt)
    tpre = T[mpre]; ypre = roi_mean[:, mpre]
    # simple slope via linear fit per ROI
    A = np.column_stack([np.ones_like(tpre), tpre])
    beta = np.linalg.lstsq(A, ypre.T, rcond=None)[0]       # (2, nROI)
    pre_slope = beta[1]
    # post-F2 amplitude
    if X2 is not None:
        mpost = (T2>=post[0]) & (T2<=post[1])
        post_amp = np.nanmean(np.nanmean(X2[:, mask, :][:, :, mpost], axis=2), axis=0)
    else:
        post_amp = np.full(pre_slope.shape, np.nan)
    return {
        'n': int(mask.sum()),
        'pre_slope_mean': float(np.nanmean(pre_slope)),
        'pre_slope_ci': np.nanpercentile(pre_slope, [25,75]).tolist(),
        'post_amp_mean': float(np.nanmean(post_amp)),
        'post_amp_ci': np.nanpercentile(post_amp, [25,75]).tolist(),
    }

summary = {
    'signal':     group_metrics(M, masks['signal']),
    'mixed':      group_metrics(M, masks['mixed']),
    'hazard_pos': group_metrics(M, masks['hazard_pos']),
    'choice_mod': group_metrics(M, masks['choice_mod']),
}
summary


# %%

# after building M, R and extras (the same extras the v2 page computed)
fig2, masks = plot_session_page2(
    M, R, extras,
    save_path="session_report_page2.png",
    preF2_window=(0.15, 0.70),
    postF2_window=(0.50, 0.60),
    eps=0.02,                 # ΔR² threshold for signal/mixed
    choice_window=(0.05,0.25),
    choice_es_thresh=0.5,     # Cohen's d threshold for choice_mod
    hazard_thresh=0.0
)


# %%


# Build servo-locked arrays
M = add_servo_locked_to_M(M, trial_data, window=(-0.6, 0.6))

# Render diagnostics page with servo panels & overlaps
fig2, masks = plot_session_page2(
    M, R, extras,
    save_path="session_report_page2.png",
    preF2_window=(0.15, 0.70),       # transient-free timer window
    postF2_window=(0.50, 0.60),      # for post-cue amplitude
    eps=0.02,                        # ΔR² threshold for 'signal'
    choice_window=(0.05, 0.25),
    choice_es_thresh=0.5,
    hazard_thresh=0.0,
    servo_auc_bins=24,
    servo_pre_window=(-0.3, 0.0)
)


# %%




# 1) Build M as usual, then add choice-locked arrays

M = add_choice_locked_to_M(M, trial_data, window=(-0.6, 0.6))

# 2) Run your existing pipeline (R, extras computed in the report call)

fig = plot_session_report_v2(M, R, session_title="Mouse • Session • Date",
                             save_path="session_report_v2.png",
                             compute_extras=True,
                             preF2_window=(0.15, 0.70))

plot_roi_group_traces(M, R, extras, groups=('choice_mod','mixed'), align='CHOICE', max_spaghetti=8)

# %%

# compute extras inside (trough vs ISI, fair-window decoders, RSA, per-ROI hazard, F2-locked choice)
fig = plot_session_report_v2(M, R,
                             session_title="Mouse • Session • Date",
                             save_path="session_report_v2.png",
                             compute_extras=True)



# %%

plot_session_onepager(M, R)



