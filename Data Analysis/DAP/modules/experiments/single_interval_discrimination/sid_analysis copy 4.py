"""
Clean, page‑1-only pipeline for Lobule V timing (ISI) and choice analyses.

Includes:
- build_M_from_trials(...)  → canonical F1‑OFF grid (0..max ISI)
- add_choice_locked_to_M(...) → arrays aligned to choice_start
- run_timing_analysis(M, ...) → ΔR² (scaling vs clock), time‑resolved decodes,
  trough vs ISI, hazard unique variance, RSA (abs‑time vs phase)
- run_choice_analysis(M, ...) → choice‑locked pre‑choice decode
- plot_timing_report(M, T, ...), plot_choice_report(M, C, ...) → page‑1 figures
- run_all_and_plot(trial_data, behavioral_data, ...) → end‑to‑end driver

Dependencies: numpy, matplotlib, scikit‑learn
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Any, Tuple

# --------------------------- helpers ---------------------------

def _to_seconds(arr_like):
    """Convert array-like values to seconds, assuming ms if values > 10."""
    arr = np.asarray(arr_like, dtype=float)
    print(f"_to_seconds input range: {np.nanmin(arr):.3f} to {np.nanmax(arr):.3f}")
    # If values look like ms (e.g., >10), convert to seconds.
    if np.nanmax(arr) > 10.0:
        arr = arr / 1000.0
        print(f"Converted from ms to seconds. New range: {np.nanmin(arr):.3f} to {np.nanmax(arr):.3f}")
    return arr

def nanmean_no_warn(a: np.ndarray, axis: int):
    valid = np.isfinite(a)
    cnt = valid.sum(axis=axis, keepdims=True)
    s = np.nansum(a, axis=axis, keepdims=True)
    out = s / np.maximum(cnt, 1)
    out[cnt == 0] = np.nan
    return np.squeeze(out, axis=axis)

# Simple Gaussian RBF basis (used for scaling/clock models)
def _rbf_basis(z: np.ndarray, n_centers: int, zmin: float, zmax: float, add_const=True) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    centers = np.linspace(zmin, zmax, n_centers)
    # width so that adjacent centers overlap sensibly
    width = (zmax - zmin) / (n_centers - 1 + 1e-9)
    width = max(width, (zmax - zmin) / max(4, n_centers))
    Phi = np.exp(-0.5 * ((z[:, None] - centers[None, :]) / (width + 1e-9)) ** 2)
    if add_const:
        Phi = np.column_stack([np.ones_like(z), Phi])
    return Phi  # (len(z), n_centers + const)

# Ridge fit/predict (no intercept—basis should include const)
def _ridge_fit_predict(Phi_tr: np.ndarray, y_tr: np.ndarray, Phi_te: np.ndarray, alpha=1e-3) -> np.ndarray:
    # Closed-form ridge: (Phi^T Phi + a I)^-1 Phi^T y
    A = Phi_tr.T @ Phi_tr
    n = A.shape[0]
    w = np.linalg.solve(A + alpha * np.eye(n), Phi_tr.T @ y_tr)
    return Phi_te @ w

# R^2 with NaN‑robustness
from sklearn.metrics import r2_score

def r2_nan(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return np.nan
    return r2_score(y_true[m], y_pred[m])

def auto_preF2_window(M, frac=0.9, min_width=0.35):
    """Return a safe pre-F2 window common to all trials."""
    T = M['time']
    earliest_f2 = float(np.nanmin(M['isi'])) if np.isfinite(np.nanmin(M['isi'])) else T[-1]
    t1 = min(frac * earliest_f2, T[-1] - 1e-6)
    t0 = max(0.0, t1 - min_width)
    if t1 <= t0:  # last resort
        t0, t1 = 0.0, min(earliest_f2 * 0.8, T[-1])
    return (t0, t1)

def baseline_correct_inplace(M, base_win=(0.0, 0.05)):
    """Subtract per-trial, per-ROI baseline (0–50 ms) from roi_traces."""
    X, T = M['roi_traces'], M['time']
    bm = (T >= base_win[0]) & (T <= base_win[1])
    if not np.any(bm): return M
    base = np.nanmean(X[:, :, bm], axis=2, keepdims=True)
    M['roi_traces'] = X - base
    return M


# ---------------------- build data matrices ----------------------

def build_M_from_trials(trial_data: Dict[str, Any], grid_bins: int = 240) -> Dict[str, Any]:
    print("=== Starting build_M_from_trials ===")
    df = trial_data['df_trials_with_segments']
    n_trials = len(df)
    print(f"Number of trials: {n_trials}")

    # ISIs and allowed set (seconds)
    isi_vec = _to_seconds(df['isi'].to_numpy())
    isi_allowed = _to_seconds(trial_data['session_info']['unique_isis'])
    isi_allowed = np.sort(np.unique(isi_allowed))
    print(f"ISI vector shape: {isi_vec.shape}, range: {np.nanmin(isi_vec):.3f} to {np.nanmax(isi_vec):.3f}")
    print(f"Allowed ISIs: {isi_allowed}")

    # Time grid 0..max(ISI) — aligned to F1‑OFF
    t_max = float(np.nanmax(isi_vec))
    Nt = int(grid_bins)
    time = np.linspace(0.0, t_max, Nt)
    print(f"Time grid: {Nt} bins from 0 to {t_max:.3f} seconds")
    print(f"Time resolution: {(time[1]-time[0]):.4f} seconds per bin")

    # Infer #ROIs from first row
    first = df.iloc[0]
    dff0 = np.asarray(first['dff_segment'], dtype=float)
    if dff0.ndim != 2:
        raise ValueError("dff_segment must be 2D (n_rois x n_samples) per trial")
    n_rois = dff0.shape[0]
    print(f"Number of ROIs: {n_rois}")
    print(f"First trial dF/F shape: {dff0.shape}")

    # Allocate
    roi_traces = np.full((n_trials, n_rois, Nt), np.nan, dtype=np.float32)
    print(f"Initialized roi_traces shape: {roi_traces.shape}")

    # Behavioral arrays
    is_right_trial   = np.zeros(n_trials, dtype=bool)
    is_right_choice  = np.zeros(n_trials, dtype=bool)
    rewarded         = np.zeros(n_trials, dtype=bool)
    punished         = np.zeros(n_trials, dtype=bool)
    did_not_choose   = np.zeros(n_trials, dtype=bool)
    time_dnc         = np.full(n_trials, np.nan, float)
    choice_start     = np.full(n_trials, np.nan, float)  # canonical
    choice_stop      = np.full(n_trials, np.nan, float)
    lick             = np.zeros(n_trials, dtype=bool)
    lick_start       = np.full(n_trials, np.nan, float)
    RT               = np.full(n_trials, np.nan, float)
    F1_on            = np.full(n_trials, np.nan, float)
    F1_off           = np.full(n_trials, np.nan, float)
    F2_on            = np.full(n_trials, np.nan, float)
    F2_off           = np.full(n_trials, np.nan, float)

    print("Processing individual trials...")
    for i, (_, row) in enumerate(df.iterrows()):
        # Align to F1‑OFF
        f1off = float(row['end_flash_1'])
        t_vec = np.asarray(row['dff_time_vector'], float) - f1off
        dff   = np.asarray(row['dff_segment'], float)
        if not np.all(np.diff(t_vec) > 0):
            order = np.argsort(t_vec)
            t_vec = t_vec[order]; dff = dff[:, order]
        tmin, tmax = np.nanmin(t_vec), np.nanmax(t_vec)
        for r in range(n_rois):
            vals = np.interp(time, t_vec, dff[r], left=np.nan, right=np.nan)
            outside = (time < tmin) | (time > tmax)
            vals[outside] = np.nan
            roi_traces[i, r] = vals

        # Copy metadata (with safe fallbacks)
        is_right_trial[i]  = bool(row.get('is_right', False))
        is_right_choice[i] = bool(row.get('is_right_choice', False))
        rewarded[i]        = bool(row.get('rewarded', False))
        punished[i]        = bool(row.get('punished', False))
        did_not_choose[i]  = bool(row.get('did_not_choose', False))
        time_dnc[i]        = float(row.get('time_did_not_choose', np.nan))
        cs = row.get('choice_start', row.get('servo_in', np.nan))
        choice_start[i]    = float(cs) if cs is not None else np.nan
        choice_stop[i]     = float(row.get('choice_stop', np.nan))
        lick[i]            = bool(row.get('lick', False))
        lick_start[i]      = float(row.get('lick_start', np.nan))
        RT[i]              = float(row.get('RT', np.nan))
        F1_on[i]           = float(row.get('start_flash_1', np.nan))
        F1_off[i]          = float(row.get('end_flash_1', np.nan))
        F2_on[i]           = float(row.get('start_flash_2', np.nan))
        F2_off[i]          = float(row.get('end_flash_2', np.nan))

    M = dict(
        roi_traces=roi_traces,
        time=time.astype(np.float32),
        isi=_to_seconds(isi_vec).astype(np.float32),
        isi_allowed=isi_allowed.astype(np.float32),
        is_right=is_right_trial,
        is_right_choice=is_right_choice,
        rewarded=rewarded,
        punished=punished,
        did_not_choose=did_not_choose,
        time_did_not_choose=time_dnc,
        choice_start=choice_start,
        choice_stop=choice_stop,
        lick=lick,
        lick_start=lick_start,
        RT=RT,
        F1_on=F1_on, F1_off=F1_off, F2_on=F2_on, F2_off=F2_off,
        n_trials=n_trials, n_rois=n_rois
    )
    # For convenience
    M['F2_time'] = M['isi']  # identical in this segmentation
    return M

# Choice‑locked arrays -------------------------------------------------

def add_choice_locked_to_M(M: Dict[str, Any], trial_data: Dict[str, Any], window: Tuple[float,float] = (-0.6, 0.6), grid_bins: int = 240) -> Dict[str, Any]:
    df = trial_data['df_trials_with_segments']
    rows = list(df.iterrows())
    n_trials = len(rows)
    n_rois = int(M['n_rois'])
    tau = np.linspace(window[0], window[1], grid_bins)
    Xc = np.full((n_trials, n_rois, grid_bins), np.nan, dtype=np.float32)
    lick_rel = np.full(n_trials, np.nan, float)

    for i, (_, row) in enumerate(rows):
        cs = row.get('choice_start', row.get('servo_in', np.nan))
        cs = float(cs) if cs is not None else np.nan
        t_vec = np.asarray(row['dff_time_vector'], float) - cs
        dff   = np.asarray(row['dff_segment'], float)
        if not np.all(np.diff(t_vec) > 0):
            order = np.argsort(t_vec)
            t_vec = t_vec[order]; dff = dff[:, order]
        tmin, tmax = np.nanmin(t_vec), np.nanmax(t_vec)
        for r in range(n_rois):
            vals = np.interp(tau, t_vec, dff[r], left=np.nan, right=np.nan)
            vals[(tau < tmin) | (tau > tmax)] = np.nan
            Xc[i, r] = vals
        if row.get('lick_start') is not None and np.isfinite(cs):
            lick_rel[i] = float(row['lick_start']) - cs

    M['roi_traces_choice'] = Xc
    M['time_choice'] = tau.astype(np.float32)
    M['lick_start_relChoice'] = lick_rel.astype(np.float32)
    return M

# ---------------------- timing analysis (ISI) ----------------------

# def scaling_vs_clock_cv(M: Dict[str, Any], preF2_window=(0.0, 0.7), n_bases=9, alpha=1e-2, n_splits=5) -> Dict[str, Any]:
#     X = M['roi_traces']  # (Ntr, Nroi, Nt)
#     T = M['time']
#     isi = M['isi']
    

#     from sklearn.model_selection import KFold
#     kf = KFold(n_splits=min(n_splits, len(isi)), shuffle=True, random_state=0)

#     Ntr, Nroi, _ = X.shape
#     r2_phase = np.full(Nroi, np.nan)
#     r2_time  = np.full(Nroi, np.nan)



#     # --- robust pre-F2 window selection ---
#     t0_req, t1_req = preF2_window
#     earliest_f2 = float(np.nanmin(M['isi'])) if np.isfinite(np.nanmin(M['isi'])) else T[-1]
#     t0 = max(0.0, float(t0_req))
#     t1 = min(float(t1_req), earliest_f2 - 1e-6)

#     Nt_sel = (T >= t0) & (T <= t1)
#     if Nt_sel.sum() < 8:
#         # fallback: keep a sensible pre-F2 slice with at least a few bins
#         t1 = min(earliest_f2 * 0.9, T[-1])
#         t0 = max(0.0, t1 - max(0.25, 0.2 * max(t1, 1e-3)))
#         Nt_sel = (T >= t0) & (T <= t1)
#         if Nt_sel.sum() < 4:
#             raise ValueError(f"PreF2 window has no samples on the grid (computed [{t0:.3f}, {t1:.3f}] s).")
#     # --------------------------------------

#     # Precompute time‑basis for the clock model (same for all trials)
#     # Nt_sel = (T >= preF2_window[0]) & (T <= preF2_window[1])
#     # Bt = _rbf_basis(T[Nt_sel], n_bases, T[Nt_sel][0], T[Nt_sel][-1])  # (Nt_sel, nb+1)
#     Bt = _rbf_basis(T[Nt_sel], n_bases, T[Nt_sel][0], T[Nt_sel][-1])  # (Nt_sel, nb+1)
    
#     for r in range(Nroi):
#         y_pred_phase_all = []; y_true_phase_all = []
#         y_pred_time_all  = []; y_true_time_all  = []
#         for tr_idx, te_idx in kf.split(np.arange(Ntr)):
#             # Phase (scaling) model: per‑trial phase grid
#             A_phase_list = []; y_phase_list = []
#             A_time_list  = []; y_time_list  = []
#             for ii in tr_idx:
#                 tt = T[Nt_sel]
#                 ph = (tt / (isi[ii] + 1e-9)).clip(0, 1)
#                 Bp = _rbf_basis(ph, n_bases, 0.0, 1.0)
#                 A_phase_list.append(Bp)
#                 A_time_list.append(Bt)
#                 y_trial = X[ii, r, Nt_sel]
#                 y_phase_list.append(y_trial)
#                 y_time_list.append(y_trial)
#             A_phase = np.vstack(A_phase_list)
#             y_phase = np.hstack(y_phase_list)
#             A_time  = np.vstack(A_time_list)
#             y_time  = np.hstack(y_time_list)

#             # Remove NaNs
#             m1 = np.isfinite(y_phase) & np.all(np.isfinite(A_phase), axis=1)
#             m2 = np.isfinite(y_time)  & np.all(np.isfinite(A_time), axis=1)
#             if m1.sum() < 20 or m2.sum() < 20:
#                 continue
#             yhat_phase_w = _ridge_fit_predict(A_phase[m1], y_phase[m1], A_phase[m1], alpha=alpha)
#             yhat_time_w  = _ridge_fit_predict(A_time[m2],  y_time[m2],  A_time[m2],  alpha=alpha)
#             # Now predict on held‑out trials
#             for ii in te_idx:
#                 tt = T[Nt_sel]
#                 ph = (tt / (isi[ii] + 1e-9)).clip(0, 1)
#                 Bp = _rbf_basis(ph, n_bases, 0.0, 1.0)
#                 y_true = X[ii, r, Nt_sel]
#                 y_pred = _ridge_fit_predict(A_phase[m1], y_phase[m1], Bp, alpha=alpha)
#                 y_pred_phase_all.append(y_pred)
#                 y_true_phase_all.append(y_true)
#                 # clock
#                 y_pred_t = _ridge_fit_predict(A_time[m2], y_time[m2], Bt, alpha=alpha)
#                 y_pred_time_all.append(y_pred_t)
#                 y_true_time_all.append(y_true)
#         if y_true_phase_all:
#             ytp = np.concatenate(y_true_phase_all)
#             ypp = np.concatenate(y_pred_phase_all)
#             r2_phase[r] = r2_nan(ytp, ypp)
#         if y_true_time_all:
#             ytt = np.concatenate(y_true_time_all)
#             ypt = np.concatenate(y_pred_time_all)
#             r2_time[r]  = r2_nan(ytt, ypt)

#     delta = r2_phase - r2_time
#     out = dict(delta_r2=delta, r2_phase=r2_phase, r2_time=r2_time,
#                window=preF2_window)
#     return out

def scaling_vs_clock_cv(M, preF2_window=None, n_bases=9, alpha=1e-2, n_splits=5):
    X, T, isi = M['roi_traces'], M['time'], M['isi']
    if preF2_window is None:
        preF2_window = auto_preF2_window(M)
    t0, t1 = preF2_window
    Nt_sel = (T >= t0) & (T <= t1)
    tt = T[Nt_sel]
    if tt.size < 8:
        raise ValueError(f"preF2 window too small: {preF2_window}")
    Bt = _rbf_basis(tt, n_bases, tt[0], tt[-1])

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=min(n_splits, X.shape[0]), shuffle=True, random_state=0)

    Ntr, Nroi = X.shape[0], X.shape[1]
    r2_phase = np.full(Nroi, np.nan); r2_time = np.full(Nroi, np.nan)

    for r in range(Nroi):
        yp_true=[]; yp_pred=[]; yt_true=[]; yt_pred=[]
        for tr_idx, te_idx in kf.split(np.arange(Ntr)):
            # build training design
            A_phase=[]; y_phase=[]; A_time=[]; y_time=[]
            for ii in tr_idx:
                ph = (tt / (isi[ii] + 1e-9)).clip(0, 1)
                Bp = _rbf_basis(ph, n_bases, 0.0, 1.0)
                A_phase.append(Bp); A_time.append(Bt)
                y = X[ii, r, Nt_sel]; y_phase.append(y); y_time.append(y)
            A_phase = np.vstack(A_phase); y_phase = np.hstack(y_phase)
            A_time  = np.vstack(A_time);  y_time  = np.hstack(y_time)

            m1 = np.isfinite(y_phase) & np.all(np.isfinite(A_phase), axis=1)
            m2 = np.isfinite(y_time)  & np.all(np.isfinite(A_time),  axis=1)
            if m1.sum() < 30 or m2.sum() < 30: continue

            # predict held-out
            for ii in te_idx:
                ph = (tt / (isi[ii] + 1e-9)).clip(0, 1)
                Bp = _rbf_basis(ph, n_bases, 0.0, 1.0)
                y_true = X[ii, r, Nt_sel]
                yp_true.append(y_true)
                yp_pred.append(_ridge_fit_predict(A_phase[m1], y_phase[m1], Bp, alpha))
                yt_true.append(y_true)
                yt_pred.append(_ridge_fit_predict(A_time[m2],  y_time[m2],  Bt, alpha))
        if yp_true:
            r2_phase[r] = r2_nan(np.concatenate(yp_true), np.concatenate(yp_pred))
            r2_time[r]  = r2_nan(np.concatenate(yt_true), np.concatenate(yt_pred))
    return dict(delta_r2=r2_phase - r2_time, r2_phase=r2_phase, r2_time=r2_time,
                window=preF2_window)


# Time‑resolved decoders (ISI regression + short/long)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# def time_resolved_isi_decode(M: Dict[str, Any], preF2_window=(0.0, 0.7), n_bins: int = 24) -> Dict[str, Any]:
#     X = M['roi_traces']; T = M['time']; isi = M['isi']
#     # mwin = (T >= preF2_window[0]) & (T <= preF2_window[1])
#     # tvec = T[mwin]
#     # edges = np.linspace(tvec[0], tvec[-1], n_bins + 1)
    
    
#     # --- robust pre-F2 window selection ---
#     t0_req, t1_req = preF2_window
#     earliest_f2 = float(np.nanmin(M['isi'])) if np.isfinite(np.nanmin(M['isi'])) else T[-1]
#     t0 = max(0.0, float(t0_req))
#     t1 = min(float(t1_req), earliest_f2 - 1e-6)
#     mwin = (T >= t0) & (T <= t1)
#     if mwin.sum() < 8:
#         t1 = min(earliest_f2 * 0.9, T[-1])
#         t0 = max(0.0, t1 - max(0.25, 0.2 * max(t1, 1e-3)))
#         mwin = (T >= t0) & (T <= t1)
#         if mwin.sum() < 4:
#             raise ValueError(f"PreF2 window has no samples on the grid (computed [{t0:.3f}, {t1:.3f}] s).")
#     # --------------------------------------
    
#     tvec = T[mwin]
#     edges = np.linspace(tvec[0], tvec[-1], n_bins + 1)
        
    
    
#     centers = 0.5 * (edges[:-1] + edges[1:])

#     # boundary between short/long: mid of allowed set
#     boundary = np.median(M['isi_allowed'])
#     y_bin = (isi >= boundary).astype(int)

#     r2_t = np.full(len(centers), np.nan)
#     auc_t = np.full(len(centers), np.nan)

#     for b in range(len(centers)):
#         m = (T >= edges[b]) & (T < edges[b+1])
#         Z = np.nanmean(X[:, :, m], axis=2)
#         valid = np.all(np.isfinite(Z), axis=1)
#         if valid.sum() < 12:
#             continue
#         Z = Z[valid]; y_reg = isi[valid]; y_clf = y_bin[valid]
#         kf = KFold(n_splits=min(5, Z.shape[0]), shuffle=True, random_state=0)
#         preds = []; truth = []
#         scores = []
#         for tr, te in kf.split(Z):
#             Ztr, Zte = Z[tr], Z[te]
#             r = Ridge(alpha=1.0).fit(Ztr, y_reg[tr])
#             preds.append(r.predict(Zte)); truth.append(y_reg[te])
#             clf = LogisticRegression(penalty='l2', solver='liblinear', C=1.0).fit(Ztr, y_clf[tr])
#             if np.unique(y_clf[te]).size >= 2:
#                 p = clf.predict_proba(Zte)[:, 1]
#                 scores.append(roc_auc_score(y_clf[te], p))
#         if preds:
#             yhat = np.concatenate(preds); ytru = np.concatenate(truth)
#             r2_t[b] = r2_nan(ytru, yhat)
#         auc_t[b] = float(np.nanmean(scores)) if scores else np.nan

#     return dict(time_points=centers, isi_r2=r2_t, auc_shortlong=auc_t)

def time_resolved_isi_decode(M, preF2_window=None, n_bins=24):
    X, T, isi = M['roi_traces'], M['time'], M['isi']
    if preF2_window is None:
        preF2_window = auto_preF2_window(M)
    t0, t1 = preF2_window
    mwin = (T >= t0) & (T <= t1)
    tt = T[mwin]
    edges = np.linspace(tt[0], tt[-1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    boundary = np.median(M['isi_allowed'])
    y_bin = (isi >= boundary).astype(int)

    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    r2_t = np.full(len(centers), np.nan); auc_t = np.full(len(centers), np.nan)

    for b in range(len(centers)):
        m = (T >= edges[b]) & (T < edges[b+1])
        Z = np.nanmean(X[:, :, m], axis=2)
        valid = np.all(np.isfinite(Z), axis=1)
        if valid.sum() < 16: continue
        Z = Z[valid]; y_reg = isi[valid]; y_clf = y_bin[valid]
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
        if preds:
            r2_t[b] = r2_nan(np.concatenate(truth), np.concatenate(preds))
        auc_t[b] = float(np.nanmean(scores)) if scores else np.nan
    return dict(time_points=centers, isi_r2=r2_t, auc_shortlong=auc_t)



# Trough vs ISI (population)

def trough_time_vs_isi(M: Dict[str, Any]) -> Dict[str, Any]:
    X = M['roi_traces']; T = M['time']; isi = M['isi']; levels = np.unique(M['isi_allowed'])
    trough_t = []
    for lv in levels:
        sel = np.isfinite(isi) & (np.abs(isi - lv) < 1e-6)
        if sel.sum() < 5:
            trough_t.append(np.nan); continue
        Y = nanmean_no_warn(X[sel], axis=0)  # (Nroi, Nt)
        y = nanmean_no_warn(Y, axis=0)       # (Nt,)
        idx = np.nanargmin(y)
        trough_t.append(T[idx])
    trough_t = np.asarray(trough_t, float)
    # fits
    x = levels; y = trough_t
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 3:
        A = np.column_stack([np.ones(m.sum()), x[m]])
        beta = np.linalg.lstsq(A, y[m], rcond=None)[0]
        yhat = A @ beta
        ss_res = np.nansum((y[m] - yhat) ** 2)
        ss_tot = np.nansum((y[m] - np.nanmean(y[m])) ** 2)
        r2_lin = 1 - ss_res / (ss_tot + 1e-9)
    else:
        r2_lin = np.nan
    return dict(levels=levels, trough_time=trough_t, r2_linear=r2_lin)

# Hazard unique variance (simple two‑regressor GLM: time vs bumps at allowed ISIs)

def hazard_unique_variance(M: Dict[str, Any], sigma: float = 0.06, preF2_window=(0.0, 0.7)) -> Dict[str, Any]:
    X = M['roi_traces']; T = M['time']; levels = np.asarray(M['isi_allowed'])
    # mwin = (T >= preF2_window[0]) & (T <= preF2_window[1])
    # tt = T[mwin]
    
    
    
    # --- robust pre-F2 window selection ---
    if preF2_window is None:
        preF2_window = auto_preF2_window(M)    
    t0_req, t1_req = preF2_window
    earliest_f2 = float(np.nanmin(M['isi'])) if np.isfinite(np.nanmin(M['isi'])) else T[-1]
    t0 = max(0.0, float(t0_req))
    t1 = min(float(t1_req), earliest_f2 - 1e-6)
    mwin = (T >= t0) & (T <= t1)
    if mwin.sum() < 8:
        t1 = min(earliest_f2 * 0.9, T[-1])
        t0 = max(0.0, t1 - max(0.25, 0.2 * max(t1, 1e-3)))
        mwin = (T >= t0) & (T <= t1)
    # --------------------------------------
    tt = T[mwin]
    
    
    
    # Build hazard bump regressor
    H = np.zeros_like(tt)
    for a in levels:
        H += np.exp(-0.5 * ((tt - a) / max(sigma, 1e-3)) ** 2)
    H = (H - H.min()) / (H.max() - H.min() + 1e-9)
    # Time ramp regressor
    R = (tt - tt.min()) / (tt.max() - tt.min() + 1e-9)

    Ntr, Nroi, _ = X.shape
    uv = np.full(Nroi, np.nan)
    for r in range(Nroi):
        y = nanmean_no_warn(X[:, r, :][:, mwin], axis=0)  # session average trace
        if np.sum(np.isfinite(y)) < 10: continue
        # full model
        A = np.column_stack([np.ones_like(tt), R, H])
        m = np.all(np.isfinite(A), axis=1) & np.isfinite(y)
        if m.sum() < 10: continue
        beta = np.linalg.lstsq(A[m], y[m], rcond=None)[0]
        yhat_full = A[m] @ beta
        r2_full = r2_nan(y[m], yhat_full)
        # time‑only
        A0 = np.column_stack([np.ones_like(tt), R])
        beta0 = np.linalg.lstsq(A0[m], y[m], rcond=None)[0]
        yhat0 = A0[m] @ beta0
        r2_time = r2_nan(y[m], yhat0)
        uv[r] = max(0.0, (r2_full - r2_time))
    return dict(uv_per_roi=uv, session_uv=float(np.nanmean(uv)), frac_positive=float(np.mean(uv > 0)))

# RSA: abs‑time vs phase (mean off‑diag corr in a preF2 window)

def rsa_abs_vs_phase(M: Dict[str, Any], preF2_window=(0.15, 0.70), n_phase_bins: int = 20) -> Dict[str, Any]:
    X = M['roi_traces']; T = M['time']; isi = M['isi']; levels = np.unique(M['isi_allowed'])
    # mwin = (T >= preF2_window[0]) & (T <= preF2_window[1])
    # tt = T[mwin]
    
    
    # --- robust pre-F2 window selection ---
    if preF2_window is None:
        preF2_window = auto_preF2_window(M)
    t0_req, t1_req = preF2_window
    earliest_f2 = float(np.nanmin(M['isi'])) if np.isfinite(np.nanmin(M['isi'])) else T[-1]
    t0 = max(0.0, float(t0_req))
    t1 = min(float(t1_req), earliest_f2 - 1e-6)
    mwin = (T >= t0) & (T <= t1)
    if mwin.sum() < 8:
        t1 = min(earliest_f2 * 0.9, T[-1])
        t0 = max(0.0, t1 - max(0.25, 0.2 * max(t1, 1e-3)))
        mwin = (T >= t0) & (T <= t1)
    # --------------------------------------
    tt = T[mwin]

    
    
    # mean ROI×time per ISI
    per_isi = {}
    for lv in levels:
        sel = np.isfinite(isi) & (np.abs(isi - lv) < 1e-6)
        if sel.sum() < 5: continue
        Y = nanmean_no_warn(X[sel], axis=0)  # (Nroi, Nt)
        per_isi[lv] = Y[:, mwin]
    keys = sorted(per_isi.keys())
    if len(keys) < 3:
        return dict(abs_time=np.nan, phase=np.nan)

    # abs‑time correlations: for each t, corr across ROIs between all ISI pairs
    def _mean_offdiag_corr(stack):  # stack shape (nISI, Nroi, Nt)
        nI = stack.shape[0]; Nt = stack.shape[2]
        vals = []
        for t in range(Nt):
            mat = np.corrcoef(stack[:, :, t])  # corr across ISIs using ROI dimension
            if np.any(np.isnan(mat)): continue
            off = mat[~np.eye(nI, dtype=bool)]
            vals.append(np.nanmean(off))
        return float(np.nanmean(vals)) if vals else np.nan

    arr_abs = np.stack([per_isi[k] for k in keys], axis=0)  # (nI,Nroi,Ntwin)
    abs_time_score = _mean_offdiag_corr(arr_abs)

    # phase: resample each ISI window onto common phase grid
    Nt = arr_abs.shape[2]
    phase_grid = np.linspace(0, 1, n_phase_bins)
    arr_phase = []
    for k in keys:
        Y = per_isi[k]
        # map absolute tt to phase in [0,1] for this ISI k
        ph = (tt / (k + 1e-9)).clip(0, 1)
        for r in range(Y.shape[0]):
            Yr = Y[r]
            Yp = np.interp(phase_grid, ph, Yr, left=np.nan, right=np.nan)
            if r == 0:
                Yp_stack = np.empty((Y.shape[0], n_phase_bins)); Yp_stack[:] = np.nan
            Yp_stack[r] = Yp
        arr_phase.append(Yp_stack)
    arr_phase = np.stack(arr_phase, axis=0)
    phase_score = _mean_offdiag_corr(arr_phase)

    return dict(abs_time=abs_time_score, phase=phase_score)

def fair_window_decoders(M, preF2_window=None, win=0.12, step=0.03):
    """Windowed features: amplitude (mean) and slope (OLS over time) per ROI;
       decode ISI (R²) and short/long (AUC) across time."""
    X, T, isi = M['roi_traces'], M['time'], M['isi']
    if preF2_window is None:
        preF2_window = auto_preF2_window(M)
    t0, t1 = preF2_window
    centers = []
    amp_r2=[]; amp_auc=[]; slope_r2=[]; slope_auc=[]
    boundary = np.median(M['isi_allowed'])
    y_bin = (isi >= boundary).astype(int)

    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    t = T[(T>=t0)&(T<=t1)]
    c = t0 + win/2.0
    while c + win/2.0 <= t1:
        m = (T >= (c - win/2.0)) & (T < (c + win/2.0))
        if m.sum() >= 4:
            # amplitude: (Ntr, Nroi)
            A = np.nanmean(X[:, :, m], axis=2)
            # slope: fit y = a + b t per ROI → take b per ROI as feature
            tt = T[m] - T[m].mean()
            denom = np.sum(tt**2) + 1e-9
            B = np.tensordot(X[:, :, m], tt, axes=(2,0)) / denom  # (Ntr,Nroi)

            for F, r2_list, auc_list in [(A, amp_r2, amp_auc), (B, slope_r2, slope_auc)]:
                valid = np.all(np.isfinite(F), axis=1)
                if valid.sum() < 16:
                    r2_list.append(np.nan); auc_list.append(np.nan)
                    continue
                Z = F[valid]; y_reg = isi[valid]; y_clf = y_bin[valid]
                kf = KFold(n_splits=min(5, Z.shape[0]), shuffle=True, random_state=0)
                preds=[]; truth=[]; scores=[]
                for tr, te in kf.split(Z):
                    Ztr, Zte = Z[tr], Z[te]
                    sc = StandardScaler().fit(Ztr)
                    Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)
                    r = Ridge(alpha=1.0).fit(Ztr, y_reg[tr])
                    preds.append(r.predict(Zte)); truth.append(y_reg[te])
                    clf = LogisticRegression(penalty='l2', solver='liblinear', C=1.0).fit(Ztr, y_clf[tr])
                    if np.unique(y_clf[te]).size >= 2:
                        scores.append(roc_auc_score(y_clf[te], clf.predict_proba(Zte)[:,1]))
                r2_list.append(r2_nan(np.concatenate(truth), np.concatenate(preds)) if preds else np.nan)
                auc_list.append(float(np.nanmean(scores)) if scores else np.nan)
            centers.append(c)
        c += step
    return dict(time_points=np.asarray(centers), amp_r2=np.asarray(amp_r2),
                amp_auc=np.asarray(amp_auc), slope_r2=np.asarray(slope_r2),
                slope_auc=np.asarray(slope_auc))



# ---------------------- choice analysis ----------------------

def decode_choice_choiceLocked_v3(M: Dict[str, Any], n_bins=24, baseline_window=(-0.6, -0.2), zscore_within_fold=True, restrict_preChoice=True, stratify_by_isi=True, random_state=0) -> Dict[str, Any] | None:
    X = M.get('roi_traces_choice'); T = M.get('time_choice')
    side = M.get('is_right_choice'); isi = M.get('F2_time')
    if X is None or T is None or side is None:
        return None
    side = side.astype(int)
    X = X.copy()
    # baseline
    if baseline_window is not None:
        b0, b1 = baseline_window
        bm = (T >= b0) & (T <= b1)
        if np.any(bm):
            X -= np.nanmean(X[:, :, bm], axis=2, keepdims=True)
    if restrict_preChoice:
        mpre = T < 0
        T = T[mpre]; X = X[:, :, mpre]
    edges = np.linspace(T[0], T[-1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    auc_t = np.full(len(centers), np.nan)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold

    for b in range(len(centers)):
        m = (T >= edges[b]) & (T < edges[b+1])
        if not np.any(m):
            continue
        Z_full = np.nanmean(X[:, :, m], axis=2)
        valid_idx = np.where(np.all(np.isfinite(Z_full), axis=1))[0]
        if valid_idx.size < 12:
            continue
        yv = side[valid_idx]
        if np.unique(yv).size < 2:
            continue
        if stratify_by_isi and isi is not None:
            levels = np.unique(isi[np.isfinite(isi)])
            lvl = np.searchsorted(levels, np.asarray(isi)[valid_idx])
            y_joint = yv * max(1, levels.size) + lvl
            splitter = StratifiedKFold(n_splits=min(5, len(valid_idx)), shuffle=True, random_state=random_state)
            folds = splitter.split(np.zeros_like(y_joint), y_joint)
        else:
            splitter = KFold(n_splits=min(5, len(valid_idx)), shuffle=True, random_state=random_state)
            folds = splitter.split(np.arange(len(valid_idx)))
        Z = Z_full[valid_idx]
        scores = []
        for tr_rel, te_rel in folds:
            if tr_rel.size < 10 or te_rel.size < 4:
                continue
            Ztr, Zte = Z[tr_rel], Z[te_rel]
            ytr, yte = yv[tr_rel], yv[te_rel]
            if zscore_within_fold:
                sc = StandardScaler().fit(Ztr)
                Ztr = sc.transform(Ztr); Zte = sc.transform(Zte)
            clf = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, max_iter=1000)
            clf.fit(Ztr, ytr)
            p = clf.predict_proba(Zte)[:, 1]
            if np.unique(yte).size < 2: continue
            scores.append(roc_auc_score(yte, p))
        auc_t[b] = float(np.nanmean(scores)) if scores else np.nan
    return {'time_points': centers, 'auc_t': auc_t}

# --------------------------- runners ---------------------------

# def run_timing_analysis(M: Dict[str, Any], preF2_window=(0.15, 0.70)) -> Dict[str, Any]:
#     print("Running timing analysis (ISI)…")
#     model = scaling_vs_clock_cv(M, preF2_window=preF2_window)
#     decode = time_resolved_isi_decode(M, preF2_window=preF2_window)
#     trough = trough_time_vs_isi(M)
#     hz = hazard_unique_variance(M, sigma=0.06, preF2_window=preF2_window)
#     rsa = rsa_abs_vs_phase(M, preF2_window=preF2_window)
#     return dict(model=model, decode=decode, trough=trough, hazard=hz, rsa=rsa)

def run_timing_analysis(M: Dict[str, Any], preF2_window=(0.15, 0.70)) -> Dict[str, Any]:
    print("Running timing analysis (ISI)…")
    model = scaling_vs_clock_cv(M, preF2_window=preF2_window)
    decode = time_resolved_isi_decode(M, preF2_window=preF2_window)
    trough = trough_time_vs_isi(M)
    hz = hazard_unique_variance(M, sigma=0.06, preF2_window=preF2_window)
    rsa = rsa_abs_vs_phase(M, preF2_window=preF2_window)
    return dict(model=model, decode=decode, trough=trough, hazard=hz, rsa=rsa)
    fw = fair_window_decoders(M, preF2_window=preF2_window)
    return dict(model=model, decode=decode, trough=trough, hazard=hz, rsa=rsa, fair=fw)


def run_choice_analysis(M: Dict[str, Any]) -> Dict[str, Any]:
    print("Running choice analysis (pre‑choice decode)…")
    ch = decode_choice_choiceLocked_v3(M)
    return dict(choice_auc=ch)

# --------------------------- plotting ---------------------------

def plot_timing_report(M: Dict[str, Any], T: Dict[str, Any], session_title: str | None = None, save_path: str | None = None):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1,1,1.5], wspace=0.35, hspace=0.45)

    # # [1] ΔR² histogram
    # ax1 = fig.add_subplot(gs[0,0])
    # delta = T['model']['delta_r2']
    # bins = np.linspace(np.nanmin(delta[~np.isnan(delta)] + [-1e-3]), np.nanmax(delta[~np.isnan(delta)] + [1e-3]), 31)
    # ax1.hist(delta[np.isfinite(delta)], bins=bins, color='#4682b4', edgecolor='none')
    # ax1.axvline(0, color='k', ls='--', lw=1)
    # med = np.nanmedian(delta)
    # ax1.axvline(med, color='k', lw=1.5)
    # n = len(delta)
    # ax1.set_title('Scaling vs Clock')
    # ax1.text(0.98, 0.95, f"median ΔR²={med:.3f}\nclock: {(delta<0).sum()}/{n}\nmixed: {(np.abs(delta)<=0.02).sum()}/{n}",
    #          transform=ax1.transAxes, ha='right', va='top', fontsize=8)
    # ax1.set_xlabel('ΔR² (phase − ms)'); ax1.set_ylabel('ROIs')
    
    # [1] ΔR² histogram (robust to all-NaN / constant arrays)
    ax1 = fig.add_subplot(gs[0, 0])
    delta = np.asarray(T['model']['delta_r2'], float)
    finite = np.isfinite(delta)

    if finite.sum() >= 2:
        lo = float(np.nanmin(delta[finite]))
        hi = float(np.nanmax(delta[finite]))
        if lo == hi:  # widen a constant distribution
            eps = max(1e-4, 0.1 * max(1e-3, abs(lo)))
            lo, hi = lo - eps, hi + eps
        bins = np.linspace(lo, hi, 31)
        ax1.hist(delta[finite], bins=bins, color='#4682b4', edgecolor='none')
        ax1.axvline(0, color='k', ls='--', lw=1)
        med = float(np.nanmedian(delta))
        if np.isfinite(med):
            ax1.axvline(med, color='k', lw=1.5)
        n = delta.size
        clock_cnt = int(np.sum(delta < 0))
        mixed_cnt = int(np.sum(np.abs(delta) <= 0.02))
        ax1.set_title('Scaling vs Clock')
        ax1.text(
            0.98, 0.95,
            f"median ΔR²={med:.3f}\nclock: {clock_cnt}/{n}\nmixed: {mixed_cnt}/{n}",
            transform=ax1.transAxes, ha='right', va='top', fontsize=8
        )
        ax1.set_xlabel('ΔR² (phase − ms)'); ax1.set_ylabel('ROIs')

    elif finite.sum() == 1:
        # Single finite value — plot a tiny bar at that point
        val = float(delta[finite][0])
        eps = max(1e-4, 0.1 * max(1e-3, abs(val)))
        bins = np.linspace(val - eps, val + eps, 5)
        ax1.hist([val], bins=bins, color='#4682b4', edgecolor='none')
        ax1.axvline(0, color='k', ls='--', lw=1)
        ax1.set_title('Scaling vs Clock (single finite ΔR²)')
        ax1.set_xlabel('ΔR² (phase − ms)'); ax1.set_ylabel('ROIs')

    else:
        # No finite values — show a placeholder
        ax1.text(0.5, 0.5, 'ΔR² unavailable', ha='center', va='center')
        ax1.set_title('Scaling vs Clock')
        ax1.set_xlim(-0.05, 0.05); ax1.set_ylim(0, 1)
        ax1.set_xlabel('ΔR² (phase − ms)'); ax1.set_ylabel('ROIs')


    # [2] time‑resolved decode
    ax2 = fig.add_subplot(gs[0,1])
    dec = T['decode']; tt = dec['time_points']
    ax2.plot(tt, dec['isi_r2'], lw=2, label='ISI R²(t)')
    ax2b = ax2.twinx(); ax2b.plot(tt, dec['auc_shortlong'], lw=2, ls='--', label='Short/Long AUC(t)', color='tab:orange')
    ax2.axhline(0, color='k', lw=1, ls=':'); ax2b.axhline(0.5, color='k', lw=1, ls=':')
    ax2.set_title('Time‑resolved decode'); ax2.set_xlabel('Time from F1‑OFF (s)'); ax2.set_ylabel('ISI R²(t)'); ax2b.set_ylabel('AUC')

    # [3] Hazard UV (session + fraction)
    ax3 = fig.add_subplot(gs[0,2])
    hz = T['hazard']
    ax3.bar(['Session UV','Frac ROIs>0'], [hz['session_uv'], hz['frac_positive']])
    ax3.set_title('Anticipation beyond ramp')

    # # [4] Trough vs ISI
    # ax4 = fig.add_subplot(gs[1,0])
    # tr = T['trough']; x = tr['levels']; y = tr['trough_time']
    # ax4.plot(x, y, 'o-', lw=2)
    # ax4.set_xlabel('ISI (s)'); ax4.set_ylabel('Trough time (s, F1‑OFF)')
    # ax4.set_title('Trough/turning‑point vs ISI')
    # ax4.text(0.02, 0.02, f"linear R²={tr['r2_linear']:.2f}", transform=ax4.transAxes, ha='left', va='bottom', fontsize=8)



    # [4] (replace current [4] with fair-window decoders panel)
    ax4 = fig.add_subplot(gs[1,0])
    fw = T.get('fair', None)
    if fw is not None and fw['time_points'].size:
        tt = fw['time_points']
        ax4.plot(tt, fw['amp_r2'],  lw=1.8, label='amp ISI R²')
        ax4b = ax4.twinx()
        ax4b.plot(tt, fw['slope_auc'], lw=1.8, ls='--', label='slope AUC', color='tab:orange')
        ax4.axhline(0, color='k', ls=':', lw=1); ax4b.axhline(0.5, color='k', ls=':', lw=1)
        ax4.set_title('Fair-window decoders (amp vs slope)')
        ax4.set_xlabel('Time from F1-OFF (s)'); ax4.set_ylabel('ISI R²'); ax4b.set_ylabel('AUC')
    else:
        ax4.text(0.5, 0.5, 'fair-window N/A', ha='center', va='center')



    # [5] RSA summary
    ax5 = fig.add_subplot(gs[1,1])
    rs = T['rsa']
    ax5.bar(['Abs time','Phase'], [rs['abs_time'], rs['phase']])
    ax5.set_ylim(0, 1)
    ax5.set_title('RSA mean off‑diagonal corr')

    # [6] Population raster (ROI means)
    ax6 = fig.add_subplot(gs[2,:])
    Y = nanmean_no_warn(M['roi_traces'], axis=0)  # (Nroi,Nt)
    # simple sort by time of minimum
    tmp = np.nanargmin(Y, axis=1)
    order = np.argsort(tmp)
    im = ax6.imshow(Y[order], aspect='auto', extent=[M['time'][0], M['time'][-1], 0, Y.shape[0]], cmap='viridis', vmin=np.nanpercentile(Y, 2), vmax=np.nanpercentile(Y, 98))
    ax6.set_title('Population raster (avg across trials)')
    ax6.set_xlabel('Time from F1‑OFF (s)'); ax6.set_ylabel('ROIs (sorted)')
    plt.colorbar(im, ax=ax6, fraction=0.02, pad=0.01, label='ΔF/F (a.u.)')

    if session_title:
        fig.suptitle(session_title, y=0.98, fontsize=11)
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
    return fig


def plot_choice_report(M: Dict[str, Any], C: Dict[str, Any], session_title: str | None = None, save_path: str | None = None):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(6.5, 4.2))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])
    ch = C['choice_auc']
    if ch is not None:
        t, a = ch['time_points'], ch['auc_t']
        ax.plot(t, a, lw=2)
        ax.axhline(0.5, color='k', ls='--', lw=1)
        # divergence
        def _div_time(t, s, thr=0.6, sustain=3):
            run = 0
            for i, v in enumerate(s):
                if np.isfinite(v) and v > thr:
                    run += 1
                    if run >= sustain: return t[i - sustain + 1]
                else: run = 0
            return np.nan
        dv = _div_time(t, a)
        if np.isfinite(dv):
            ax.axvline(dv, color='k', lw=1.5, ls=':')
            ax.text(dv, ax.get_ylim()[1], 'divergence', rotation=90, va='top', ha='right', fontsize=8)
        # overlay first lick median
        try:
            lk = np.nanmedian(M.get('lick_start_relChoice'))
            if np.isfinite(lk):
                ax.axvline(lk, color='C3', lw=1, ls='--')
                ax.text(lk, ax.get_ylim()[1], 'first lick', rotation=90, va='top', ha='right', fontsize=8, color='C3')
        except Exception:
            pass
        ax.set_xlabel('Time from choice start (s)'); ax.set_ylabel('AUC(τ)')
        ax.set_title('Choice‑locked left/right decode (pre‑choice)')
    else:
        ax.text(0.5, 0.5, 'Choice decode N/A', va='center', ha='center')
    if session_title:
        fig.suptitle(session_title, y=0.98, fontsize=11)
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
    return fig

# --------------------------- driver ---------------------------

def run_all_and_plot(trial_data: Dict[str, Any], save_prefix: str | None = None, grid_bins: int = 240):
    # Build
    M = build_M_from_trials(trial_data, grid_bins=grid_bins)
    M = baseline_correct_inplace(M, base_win=(0.0, 0.05))
    # Add choice‑locked arrays for the choice report
    M = add_choice_locked_to_M(M, trial_data, window=(-0.6, 0.6), grid_bins=240)
    # Analyses
    # T = run_timing_analysis(M, preF2_window=(0.15, 0.70))
    T = run_timing_analysis(M, preF2_window=None)
    C = run_choice_analysis(M)
    # Reports
    f1 = plot_timing_report(M, T, session_title='Mouse • Session • Date', save_path=(save_prefix + '_timing.png') if save_prefix else None)
    f2 = plot_choice_report(M, C, session_title='Mouse • Session • Date', save_path=(save_prefix + '_choice.png') if save_prefix else None)
    return M, T, C, f1, f2

if __name__ == '__main__':
    print('Import this module and call run_all_and_plot(...) or use individual functions.')


# %%    
    
    path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/sid_imaging_segmented_data.pkl'

    import pickle

    with open(path, 'rb') as f:
        trial_data = pickle.load(f)   # one object back (e.g., a dict)  
        
        
# %%


# End-to-end (saves figures if you pass a prefix):
M, T, C, fig_timing, fig_choice = run_all_and_plot(
    trial_data, save_prefix="session01"
)


# %%

# Build
M = build_M_from_trials(trial_data, grid_bins=240)
M = baseline_correct_inplace(M, base_win=(0.0, 0.05))
# Add choice‑locked arrays for the choice report
M = add_choice_locked_to_M(M, trial_data, window=(-0.6, 0.6), grid_bins=240)
# Analyses
# T = run_timing_analysis(M, preF2_window=(0.15, 0.70))
T = run_timing_analysis(M, preF2_window=None)
C = run_choice_analysis(M)


# %%
save_prefix="session01"
# Reports
f1 = plot_timing_report(M, T, session_title='Mouse • Session • Date', save_path=(save_prefix + '_timing.png') if save_prefix else None)
f2 = plot_choice_report(M, C, session_title='Mouse • Session • Date', save_path=(save_prefix + '_choice.png') if save_prefix else None)




# %%

# Or step-by-step:
M = build_M_from_trials(trial_data, grid_bins=240)
M = baseline_correct_inplace(M, base_win=(0.0, 0.05))
M = add_choice_locked_to_M(M, trial_data, window=(-0.6, 0.6), grid_bins=240)

# T = run_timing_analysis(M, preF2_window=(0.15, 0.70))
T = run_timing_analysis(M, preF2_window=None)
fig_t = plot_timing_report(M, T, session_title="Mouse • Session • Date", save_path="timing.png")

C = run_choice_analysis(M)
fig_c = plot_choice_report(M, C, session_title="Mouse • Session • Date", save_path="choice.png")