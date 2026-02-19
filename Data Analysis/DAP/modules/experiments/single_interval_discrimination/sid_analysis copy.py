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
    arr = np.asarray(arr_like, dtype=float)
    # If values look like ms (e.g., >10), convert to seconds.
    if np.nanmax(arr) > 10.0:
        arr = arr / 1000.0
    return arr


def build_M_from_trials(trial_data: Dict[str, Any], grid_bins: int = 240) -> Dict[str, Any]:
    """Construct canonical M from trial_data['df_trials_with_segments'].
    - F1-OFF aligned pre-F2 grid of length max(ISI), 240 bins.
    - Per-trial NaN mask for t >= ISI.
    """
    df = trial_data['df_trials_with_segments']
    # Extract arrays from DataFrame-like object reliably
    # We assume df supports iteration over rows with dict-like access.
    # rows = list(df)
    n_trials = len(df)


    # Get ISIs (seconds) and allowed set
    isi_vec = _to_seconds(df['isi'].to_numpy())
    isi_allowed = _to_seconds(trial_data['session_info']['unique_isis'])
    isi_allowed = np.sort(np.unique(isi_allowed))

    # Time grid: 0..max(ISI)
    t_max = float(np.nanmax(isi_vec))
    Nt = int(grid_bins)
    time = np.linspace(0.0, t_max, Nt)

    # Initialize holders (we infer n_rois from first row)
    first = df.iloc[0]
    dff0 = np.asarray(first['dff_segment'], dtype=float)
    if dff0.ndim != 2:
        raise ValueError("dff_segment must be 2D (n_rois x n_samples) per trial")
    n_rois = dff0.shape[0]
    roi_traces = np.full((n_trials, n_rois, Nt), np.nan, dtype=np.float32)

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

    for idx, (trial_idx, row) in enumerate(df.iterrows()):
        # Extract basic timing
        F1_on[idx] = float(row['start_flash_1'])
        F1_off[idx] = float(row['end_flash_1'])
        # Align trial's dF/F to F1-OFF = 0
        t_vec = np.asarray(row['dff_time_vector'], dtype=float) - F1_off[idx]
        dff = np.asarray(row['dff_segment'], dtype=float)  # (rois, samples)
        # Interpolate each ROI to common grid; mask outside native coverage
        # np.interp extrapolates with endpoints; we replace out-of-range with NaN
        t_min, t_max_local = np.nanmin(t_vec), np.nanmax(t_vec)
        for r in range(n_rois):
            y = dff[r]
            # Guard against non-increasing t_vec
            if not np.all(np.diff(t_vec) > 0):
                # Sort by time if needed
                order = np.argsort(t_vec)
                tt = t_vec[order]
                yy = y[order]
            else:
                tt, yy = t_vec, y
            vals = np.interp(time, tt, yy, left=np.nan, right=np.nan)
            # Replace extrapolated endpoints with NaN explicitly
            vals[(time < t_min) | (time > t_max_local)] = np.nan
            roi_traces[idx, r] = vals

        # Per-trial mask: drop samples at/after that trial's F2 (ISI from F1-OFF)
        isi = isi_vec[idx]
        roi_traces[idx, :, time >= isi] = np.nan

        # Behavior labels
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

    # is_short labeling via explicit sets (seconds)
    short_set = set(_to_seconds([200, 325, 450, 575, 700]))
    is_short = np.array([1 if float(isi) in short_set else 0 for isi in isi_vec], dtype=np.uint8)

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
    return M

# -----------------------------
# Core metrics (NaN-safe)
# -----------------------------

def _phase_time(T: np.ndarray, isi: float) -> np.ndarray:
    ph = T / max(1e-9, isi)
    return np.clip(ph, 0.0, 1.0)


def _build_bases(T: np.ndarray, F2: np.ndarray, knots_phase: int = 8, knots_time: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    n_trials = len(F2)
    st_time = SplineTransformer(degree=3, n_knots=knots_time, include_bias=False)
    B_time = st_time.fit_transform(T.reshape(-1,1))  # (Nt, b_time)
    B_time = np.broadcast_to(B_time, (n_trials, *B_time.shape))

    st_phase = SplineTransformer(degree=3, n_knots=knots_phase, include_bias=False)
    grid01 = np.linspace(0,1,len(T)).reshape(-1,1)
    st_phase.fit(grid01)
    B_phase = np.zeros((n_trials, len(T), st_phase.n_features_out_))
    for i in range(n_trials):
        ph = _phase_time(T, F2[i]).reshape(-1,1)
        B_phase[i] = st_phase.transform(ph)
    return B_phase, B_time


def _cv_r2_roi_with_nans(Y: np.ndarray, B: np.ndarray, cv_folds: int = 5) -> float:
    """Trial-wise CV R² for one ROI, dropping NaNs safely.
    Y: (Ntr, Nt); B: (Ntr, Nt, Nb)
    """
    Ntr, Nt, Nb = B.shape
    X_all = B.reshape(Ntr*Nt, Nb)
    y_all = Y.reshape(Ntr*Nt)
    finite = np.isfinite(y_all)
    if finite.sum() < 10:
        return np.nan
    trial_ids = np.repeat(np.arange(Ntr), Nt)

    preds = np.full_like(y_all, np.nan, dtype=float)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
    for train_tr, test_tr in kf.split(np.arange(Ntr)):
        tr_mask = np.isin(trial_ids, train_tr) & finite
        te_mask = np.isin(trial_ids, test_tr) & finite
        if tr_mask.sum() < Nb + 2 or te_mask.sum() == 0:
            continue
        reg = Ridge(alpha=1.0).fit(X_all[tr_mask], y_all[tr_mask])
        preds[te_mask] = reg.predict(X_all[te_mask])
    ok = np.isfinite(y_all) & np.isfinite(preds)
    if ok.sum() < 10:
        return np.nan
    return r2_score(y_all[ok], preds[ok])


def scaling_vs_clock(M: Dict[str, Any], cv_folds: int = 5, knots_phase: int = 8, knots_time: int = 8):
    X = M['roi_traces']              # (Ntr, Nroi, Nt)
    T = M['time']
    F2 = M['F2_time']
    Ntr, Nroi, Nt = X.shape
    Bp, Bt = _build_bases(T, F2, knots_phase, knots_time)

    r2_phase = np.zeros(Nroi)
    r2_ms = np.zeros(Nroi)
    for r in range(Nroi):
        Y = X[:, r, :]
        r2_phase[r] = _cv_r2_roi_with_nans(Y, Bp, cv_folds)
        r2_ms[r] = _cv_r2_roi_with_nans(Y, Bt, cv_folds)
    delta = r2_phase - r2_ms
    eps = 0.02
    pref = {
        'scaling': np.where(delta >  eps)[0],
        'clock':   np.where(delta < -eps)[0],
        'mixed':   np.where(np.abs(delta) <= eps)[0],
    }
    return delta, pref, (r2_phase, r2_ms)


def _bin_time_features(X: np.ndarray, T: np.ndarray, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(T[0], T[-1], n_bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    Ntr, Nroi, Nt = X.shape
    F = np.full((Ntr, Nroi, n_bins), np.nan, dtype=float)
    for b in range(n_bins):
        m = (T >= edges[b]) & (T < edges[b+1])
        if not np.any(m):
            continue
        # NaN-safe mean across time samples in bin
        F[:, :, b] = np.nanmean(X[:, :, m], axis=2)
    return F, centers


def _cv_ridge_r2_with_mask(Z: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    # Drop rows with NaNs in Z or y
    valid = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    if valid.sum() < 10 or np.unique(y[valid]).size < 2:
        return np.nan
    Z = Z[valid]
    y = y[valid]
    kf = KFold(n_splits=min(k, len(y)), shuffle=True, random_state=0)
    preds = np.zeros_like(y, dtype=float)
    for train, test in kf.split(Z):
        m = Ridge(alpha=1.0).fit(Z[train], y[train])
        preds[test] = m.predict(Z[test])
    return r2_score(y, preds)


def _cv_logistic_auc_with_mask(Z: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    valid = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    if valid.sum() < 10 or np.unique(y[valid]).size < 2:
        return np.nan
    Z = Z[valid]
    y = y[valid].astype(int)
    # Require both classes present
    if np.unique(y).size < 2:
        return np.nan
    kf = KFold(n_splits=min(k, len(y)), shuffle=True, random_state=0)
    scores = []
    for train, test in kf.split(Z):
        # Ensure test has both classes; if not, skip that fold
        if np.unique(y[test]).size < 2:
            continue
        m = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, max_iter=1000)
        m.fit(Z[train], y[train])
        p = m.predict_proba(Z[test])[:,1]
        scores.append(roc_auc_score(y[test], p))
    return float(np.nanmean(scores)) if scores else np.nan


def time_resolved_decoding(M: Dict[str, Any], n_bins: int = 20) -> Dict[str, Any]:
    X = M['roi_traces']
    T = M['time']
    isi = M['F2_time']
    is_short = M['is_short']
    F, centers = _bin_time_features(X, T, n_bins)
    Ntr, Nroi, B = F.shape

    auc_t = np.full(B, np.nan)
    r2_t = np.full(B, np.nan)
    for b in range(B):
        Z = F[:, :, b]
        # Standardize across trials on valid rows only
        valid_rows = np.all(np.isfinite(Z), axis=1)
        if valid_rows.sum() < 10:
            continue
        Zv = Z[valid_rows]
        Zs = StandardScaler().fit_transform(Zv)
        # Place back into full array with NaNs where invalid
        Z_std = np.full_like(Z, np.nan)
        Z_std[valid_rows] = Zs
        # Continuous ISI
        r2_t[b] = _cv_ridge_r2_with_mask(Z_std, isi, k=5)
        # Binary short/long
        auc_t[b] = _cv_logistic_auc_with_mask(Z_std, is_short, k=5)

    # Simple divergence rule: earliest of 3 consecutive bins with AUC > 0.6
    divergence_time = np.nan
    thr, sustain = 0.6, 3
    run = 0
    for b in range(B):
        if np.isfinite(auc_t[b]) and auc_t[b] > thr:
            run += 1
            if run >= sustain:
                divergence_time = centers[b - sustain + 1]
                break
        else:
            run = 0

    return {
        'time_points': centers,
        'isi_r2_t': r2_t,
        'shortlong_auc_t': auc_t,
        'divergence_time': divergence_time,
    }


def hazard_unique_variance(M: Dict[str, Any]) -> float:
    X = M['roi_traces']
    T = M['time']
    F2 = M['F2_time']
    # Compute discrete hazard at each absolute time: map to next allowed ISI >= t
    allowed = np.sort(np.unique(F2))
    # Empirical prior over allowed ISIs
    prior = np.array([(F2 == t).mean() for t in allowed])
    prior = prior / prior.sum()
    # Hazard map per allowed time
    surv = 1.0
    hmap = {}
    for p, t in zip(prior, allowed):
        hmap[t] = p / max(1e-12, surv)
        surv -= p
    next_allowed = np.array([allowed[np.searchsorted(allowed, tt, side='left')] if np.any(allowed >= tt) else allowed[-1] for tt in T])
    haz_t = np.vectorize(hmap.get)(next_allowed)

    # Response: population average over ROIs, flattened over (trial,time)
    Y = np.nanmean(X, axis=1)  # (Ntr, Nt)
    y = Y.reshape(-1)

    # Predictors: linear time and hazard(t)
    lin_t = np.tile(T, X.shape[0])
    H = np.tile(haz_t, X.shape[0])

    finite = np.isfinite(y) & np.isfinite(lin_t) & np.isfinite(H)
    if finite.sum() < 20:
        return np.nan

    # Trial-wise CV: build trial ids for grouping
    Ntr, Nt = Y.shape
    trial_ids = np.repeat(np.arange(Ntr), Nt)
    kf = KFold(n_splits=min(5, Ntr), shuffle=True, random_state=0)

    def _cv_r2(Xcols):
        preds = np.full_like(y, np.nan, dtype=float)
        for tr, te in kf.split(np.arange(Ntr)):
            m_train = np.isin(trial_ids, tr) & finite
            m_test = np.isin(trial_ids, te) & finite
            if m_train.sum() < 5 or m_test.sum() == 0:
                continue
            model = LinearRegression().fit(Xcols[m_train], y[m_train])
            preds[m_test] = model.predict(Xcols[m_test])
        ok = np.isfinite(y) & np.isfinite(preds)
        return r2_score(y[ok], preds[ok]) if ok.sum() > 10 else np.nan

    X_time = lin_t.reshape(-1,1)
    X_both = np.column_stack([lin_t, H])
    r2_time = _cv_r2(X_time)
    r2_both = _cv_r2(X_both)
    if not np.isfinite(r2_time) or not np.isfinite(r2_both):
        return np.nan
    return max(0.0, float(r2_both - r2_time))


def sort_index(M: Dict[str, Any], delta_r2: np.ndarray) -> np.ndarray:
    X = M['roi_traces']
    T = M['time']
    Y = np.nanmean(X, axis=0)  # (Nroi, Nt)
    # Smooth (light) for robustness without importing extra deps
    # Use a simple moving average of width 3
    kernel = np.array([1,2,1], dtype=float)
    kernel = kernel / kernel.sum()
    Yp = np.copy(Y)
    for r in range(Y.shape[0]):
        conv = np.convolve(Y[r], kernel, mode='same')
        Yp[r] = conv

    if np.nanmedian(delta_r2) > 0:
        # scaling: order by peak phase (using session-mean ISI)
        mean_isi = float(np.nanmean(M['F2_time']))
        ph = T / max(mean_isi, 1e-9)
        ph = np.clip(ph, 0, 1)
        prefs = []
        for r in range(Yp.shape[0]):
            idx = int(np.nanargmax(Yp[r]))
            prefs.append(ph[idx])
        order = np.argsort(prefs)
    else:
        # clock: order by absolute peak latency
        lats = np.array([int(np.nanargmax(row)) for row in Yp])
        order = np.argsort(lats)
    return order


# -----------------------------
# Runner
# -----------------------------

def run_all_from_raw(trial_data: Dict[str, Any]) -> Dict[str, Any]:
    M = build_M_from_trials(trial_data, grid_bins=240)
    delta, pref, (r2_phase, r2_ms) = scaling_vs_clock(M, cv_folds=5, knots_phase=8, knots_time=8)
    decode = time_resolved_decoding(M, n_bins=20)
    hazard_uv = hazard_unique_variance(M)
    order = sort_index(M, delta)
    return {
        'delta_r2': delta,
        'model_pref': pref,
        'r2_phase': r2_phase,
        'r2_ms': r2_ms,
        'decode': decode,
        'hazard_unique_r2': hazard_uv,
        'sort_index': order,
        'time': M['time'],
    }


if __name__ == "__main__":



    print("This module provides build_M_from_trials(...) and run_all_from_raw(...).")
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

delta, pref, (r2_phase, r2_ms) = scaling_vs_clock(M, cv_folds=5, knots_phase=8, knots_time=8)



# %%


decode = time_resolved_decoding(M, n_bins=20)







# %%

hazard_uv = hazard_unique_variance(M)




# %%

order = sort_index(M, delta)




# %%

results ={
        'delta_r2': delta,
        'model_pref': pref,
        'r2_phase': r2_phase,
        'r2_ms': r2_ms,
        'decode': decode,
        'hazard_unique_r2': hazard_uv,
        'sort_index': order,
        'time': M['time'],        
    }

    
# %%    
    # delta, pref, (r2_phase, r2_ms) = scaling_vs_clock(M, cv_folds=5, knots_phase=8, knots_time=8)
    # decode = time_resolved_decoding(M, n_bins=20)
    # hazard_uv = hazard_unique_variance(M)
    # order = sort_index(M, delta)
    # return {
    #     'delta_r2': delta,
    #     'model_pref': pref,
    #     'r2_phase': r2_phase,
    #     'r2_ms': r2_ms,
    #     'decode': decode,
    #     'hazard_unique_r2': hazard_uv,
    #     'sort_index': order,
    #     'time': M['time'],
    # }    
    
    
    
    
    
    
    
    
