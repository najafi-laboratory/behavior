
"""
roi_timing_analysis.py
----------------------
Tools for analyzing trial-segmented, event-aligned calcium imaging data in a
2AFC timing task (F1 -> F2 with variable ISIs, plus choice/motor epochs).

This module implements a compact, *task-grounded* set of analyses:

1) Scaling vs Clock model comparison (ΔR²) per ROI
2) Time-resolved neurometric decoding (continuous ISI and short/long)
3) Hazard unique variance (pre-F2) via Ridge GLM with partial R²
4) Prediction Error at F2 (regress F2-aligned response on timing surprise)
5) Raster sort indices consistent with the winning timing story
6) Split-half reliability helpers

All functions assume **trial-segmented** data; no continuous streams required.

--------------------------------------------------------------------------
INPUT DATA CONTRACT (adjust your data to match before using these funcs)
--------------------------------------------------------------------------
Per session, you should have:

- X_f1 : np.ndarray, shape (n_roi, n_time, n_trial)
    F1-aligned fluorescence (or deconvolved) traces. Time t=0 at F1 onset.

- t_f1 : np.ndarray, shape (n_time,)
    Time vector (seconds) relative to F1 onset for X_f1.

- ISI  : np.ndarray, shape (n_trial,)
    True F2 time (seconds) per trial; *one of the 10 allowed ISI levels.*

- F2_idx : np.ndarray, shape (n_trial,)
    Index into t_f1 that marks F2 onset per trial (i.e., t_f1[F2_idx[j]] ~ ISI[j]).

OPTIONAL (but useful):
- side : np.ndarray, shape (n_trial,), values in {'L','R'} or {0,1}
- choice_idx : np.ndarray, shape (n_trial,), index in t_f1 for choice/spouts-in
- lick_first_idx : np.ndarray, shape (n_trial,), index of first lick
- X_f2 : np.ndarray, shape (n_roi, n_time2, n_trial)  (F2-aligned traces, t=0 at F2)
- t_f2 : np.ndarray, shape (n_time2,)

All analyses in this module operate on these arrays. See each function docstring
for details and recommendations (z-scoring, masking, CV folds).

Author: ChatGPT (GPT-5 Thinking)
License: MIT
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
from scipy.interpolate import interp1d
from scipy.stats import linregress, rankdata, norm
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # sklearn >=1.1
except Exception:
    GroupKFold = None
    StratifiedGroupKFold = None

# -----------------------------
# Utilities and basic helpers
# -----------------------------

def zscore_across_trials(X: np.ndarray, baseline_idx: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Z-score each ROI across trials using a baseline window.
    Parameters
    ----------
    X : array (n_roi, n_time, n_trial)
    baseline_idx : boolean or integer index array over time; if None, uses all pre-zero times (t<0) is recommended externally.
    Returns
    -------
    Xz : array same shape as X
    Notes
    -----
    - Compute mean/std per ROI using (time, trial) samples within baseline_idx, then apply to entire (time, trial) grid.
    - If baseline_idx is None, this function z-scores per ROI across all timepoints and trials.
    """
    n_roi, n_time, n_trial = X.shape
    Xz = X.copy()
    if baseline_idx is not None:
        mask = baseline_idx.astype(bool)
        if mask.shape[0] != n_time:
            raise ValueError("baseline_idx must have length n_time")
        # compute stats per ROI over selected time samples across trials
        Xb = X[:, mask, :].reshape(n_roi, -1)
    else:
        Xb = X.reshape(n_roi, -1)

    mu = np.nanmean(Xb, axis=1, keepdims=True)
    sd = np.nanstd(Xb, axis=1, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu.reshape(-1,1,1)) / sd.reshape(-1,1,1)
    return Xz


def make_preF2_mask(t_f1: np.ndarray, F2_idx: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask of shape (n_time, n_trial) indicating frames strictly before F2 in each trial.
    """
    n_time = t_f1.shape[0]
    n_trial = F2_idx.shape[0]
    M = np.zeros((n_time, n_trial), dtype=bool)
    for j in range(n_trial):
        f2 = int(F2_idx[j])
        f2 = np.clip(f2, 0, n_time-1)
        M[:f2, j] = True  # strictly pre-F2
    return M


def resample_to_phase(X_f1: np.ndarray, t_f1: np.ndarray, ISI: np.ndarray, n_phase: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample each trial's F1-aligned trace to a common phase grid φ ∈ [0,1].
    Parameters
    ----------
    X_f1 : array (n_roi, n_time, n_trial)
    t_f1 : array (n_time,)
    ISI  : array (n_trial,), seconds to F2 (one of 10 levels)
    n_phase : number of phase bins; if None, uses n_time
    Returns
    -------
    X_phase : array (n_roi, n_phase, n_trial)
    phi_grid: array (n_phase,)
    Notes
    -----
    - Only samples with 0 <= t < ISI[j] are meaningful for pre-F2 analysis in trial j.
    - We linearly interpolate each trial onto a uniform phase grid φ = t / ISI[j].
    """
    n_roi, n_time, n_trial = X_f1.shape
    if n_phase is None:
        n_phase = n_time
    phi_grid = np.linspace(0.0, 1.0, n_phase, endpoint=False)  # exclude 1.0 (F2 itself) for pre-F2

    X_phase = np.full((n_roi, n_phase, n_trial), np.nan, dtype=float)
    for j in range(n_trial):
        valid_mask = (t_f1 >= 0) & (t_f1 < ISI[j])
        if not np.any(valid_mask):
            continue
        t_valid = t_f1[valid_mask]
        phi_valid = t_valid / ISI[j]
        phi_valid = np.clip(phi_valid, 0.0, 1.0 - 1e-9)
        for r in range(n_roi):
            y = X_f1[r, valid_mask, j]
            f = interp1d(phi_valid, y, kind='linear', bounds_error=False, fill_value='extrapolate')
            X_phase[r, :, j] = f(phi_grid)
    return X_phase, phi_grid


def gaussian_basis_1d(x: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    """
    Build a simple Gaussian RBF basis matrix for 1D inputs.
    Parameters
    ----------
    x : array (n_samples,)
    centers : array (n_centers,)
    width : float, common std for all Gaussians
    Returns
    -------
    Phi : array (n_samples, n_centers)
    """
    x = x[:, None]
    c = centers[None, :]
    Phi = np.exp(-0.5 * ((x - c)/width)**2)
    return Phi


def concat_trials_by_mask(y_trials: np.ndarray, x_trials: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate per-trial vectors into a single long vector using a boolean mask over (time, trial).
    Parameters
    ----------
    y_trials : array (n_time, n_trial)
    x_trials : array (n_time, n_trial) OR list of arrays to be stacked as columns
    mask     : boolean array (n_time, n_trial), True for samples to include
    Returns
    -------
    y_concat : array (N,)
    X_concat : array (N, n_feat)
    Notes
    -----
    - If x_trials is a list of arrays, each must be (n_time, n_trial). They will be column-stacked.
    """
    idx = np.where(mask.ravel(order='F'))[0]  # Fortran order to keep time-within-trial blocks together
    y = y_trials.reshape(-1, order='F')
    if isinstance(x_trials, list):
        Xcols = [x.reshape(-1, order='F') for x in x_trials]
        X = np.column_stack(Xcols)
    else:
        X = x_trials.reshape(-1, order='F')
        X = X[:, None]
    return y[idx], X[idx, :]


# --------------------------------------
# 1) Scaling vs Clock model comparison
# --------------------------------------

@dataclass
class ScalingClockResult:
    R2_clock: np.ndarray      # (n_roi,)
    R2_scale: np.ndarray      # (n_roi,)
    delta_R2: np.ndarray      # (n_roi,)
    alpha_opt: np.ndarray     # (n_roi,)


def _cv_curve_fit_ridge_roi(y_trials: np.ndarray, x_trials: np.ndarray, mask: np.ndarray,
                            n_centers:int=12, width: Optional[float]=None,
                            n_splits:int=5, seed:int=0, alphas:List[float]=[0.1,1.0,10.0]) -> float:
    """
    Internal helper: cross-validated curve fit for a single ROI using Gaussian bases over x.
    Returns CV R².
    """
    rng = np.random.RandomState(seed)
    # Build design per trial, then mask+concat
    x_min = np.nanmin(x_trials[mask])
    x_max = np.nanmax(x_trials[mask])
    centers = np.linspace(x_min, x_max, n_centers)
    if width is None:
        width = (x_max - x_min) / (n_centers * 1.5 + 1e-9)

    # Build per-trial basis then concat
    Phi_trials = gaussian_basis_1d(x_trials.flatten(order='F'), centers, width).reshape(*x_trials.shape, n_centers, order='F')
    # Concat by mask
    y_vec, X_mat = concat_trials_by_mask(y_trials, [Phi_trials[..., k] for k in range(n_centers)], mask)

    # Map each row to its trial id for CV splits
    n_time, n_trial = x_trials.shape
    trial_ids = np.repeat(np.arange(n_trial), n_time)[mask.ravel(order='F')]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng)

    # Select alpha by inner CV quickly (or just use a default)
    best_alpha = None
    best_score = -np.inf
    for a in alphas:
        scores = []
        for tr_idx, te_idx in kf.split(np.zeros(len(y_vec)), groups=trial_ids):
            mdl = Ridge(alpha=a, fit_intercept=True)
            mdl.fit(X_mat[tr_idx], y_vec[tr_idx])
            yhat = mdl.predict(X_mat[te_idx])
            scores.append(r2_score(y_vec[te_idx], yhat))
        m = np.mean(scores)
        if m > best_score:
            best_score = m
            best_alpha = a

    # Final CV with best alpha
    scores = []
    for tr_idx, te_idx in kf.split(np.zeros(len(y_vec)), groups=trial_ids):
        mdl = Ridge(alpha=best_alpha, fit_intercept=True)
        mdl.fit(X_mat[tr_idx], y_vec[tr_idx])
        yhat = mdl.predict(X_mat[te_idx])
        scores.append(r2_score(y_vec[te_idx], yhat))
    return float(np.mean(scores))


def scaling_vs_clock(X_f1: np.ndarray, t_f1: np.ndarray, ISI: np.ndarray, F2_idx: np.ndarray,
                     n_centers:int=12, width: Optional[float]=None, alpha_grid: Optional[np.ndarray]=None,
                     n_splits:int=5, seed:int=0) -> ScalingClockResult:
    """
    Compare a Clock (absolute time) vs Scaling (phase) model per ROI using CV R².
    Parameters
    ----------
    X_f1 : array (n_roi, n_time, n_trial)
    t_f1 : array (n_time,)
    ISI  : array (n_trial,)
    F2_idx : array (n_trial,)
    n_centers : number of Gaussian basis centers
    width : Gaussian width; if None, auto-estimated from x-range
    alpha_grid : array of candidate scaling exponents α in [0,1]; if None, uses np.linspace(0,1,11)
    n_splits : CV folds (by trials)
    seed : random seed
    Returns
    -------
    ScalingClockResult
    Notes
    -----
    - Pre-F2 frames only (causal). Uses a simple Gaussian basis curve fit and trial-wise CV.
    - For Scaling model, we use phase grid φ = t / ISI (α=1) and also search α if alpha_grid provided.
    """
    n_roi, n_time, n_trial = X_f1.shape
    M_pre = make_preF2_mask(t_f1, F2_idx)

    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 11)

    # Prepare x-axes: absolute and phase (alpha=1)
    t_mat = np.repeat(t_f1[:, None], n_trial, axis=1)  # (n_time, n_trial)
    phi_mat = t_mat / ISI[None, :]
    phi_mat = np.clip(phi_mat, 0.0, 1.0 - 1e-9)

    R2_clock = np.zeros(n_roi)
    R2_scale = np.zeros(n_roi)
    alpha_opt = np.zeros(n_roi)

    for r in range(n_roi):
        y_trials = X_f1[r, :, :]  # (n_time, n_trial)

        # Clock fit (absolute time)
        R2c = _cv_curve_fit_ridge_roi(y_trials, t_mat, M_pre,
                                      n_centers=n_centers, width=width,
                                      n_splits=n_splits, seed=seed)
        R2_clock[r] = R2c

        # Scaling fit: search α
        best = -np.inf
        best_alpha = 1.0
        for a in alpha_grid:
            x_warp = t_mat / (ISI[None, :] ** a)
            x_warp = np.clip(x_warp, 0.0, np.nanmax(x_warp[M_pre]) + 1e-9)
            R2a = _cv_curve_fit_ridge_roi(y_trials, x_warp, M_pre,
                                          n_centers=n_centers, width=width,
                                          n_splits=n_splits, seed=seed)
            if R2a > best:
                best = R2a
                best_alpha = a
        R2_scale[r] = best
        alpha_opt[r] = best_alpha

    delta = R2_scale - R2_clock
    return ScalingClockResult(R2_clock=R2_clock, R2_scale=R2_scale, delta_R2=delta, alpha_opt=alpha_opt)


# --------------------------------------
# 2) Time-resolved decoding (neurometric)
# --------------------------------------

@dataclass
class NeurometricResult:
    times: np.ndarray              # (n_time_used,)
    r2_continuous: np.ndarray      # ISI regression R² per time (NaN where not enough trials)
    auc_binary: np.ndarray         # short/long AUC per time (NaN where not enough trials)
    n_trials_available: np.ndarray # number of trials contributing per time


def time_resolved_decoding(X_f1: np.ndarray, t_f1: np.ndarray, ISI: np.ndarray,
                           F2_idx: np.ndarray, boundary: float,
                           preF2_only: bool=True, n_splits:int=5, seed:int=0) -> NeurometricResult:
    """
    Decode ISI (continuous) and short/long (binary) from population activity over time.
    Parameters
    ----------
    X_f1 : array (n_roi, n_time, n_trial)
    t_f1 : array (n_time,)
    ISI  : array (n_trial,)
    F2_idx : array (n_trial,)
    boundary : float, threshold separating short/long (e.g., midpoint between the two sets)
    preF2_only : if True, at each time use only trials where t < F2
    n_splits : CV folds (by trials)
    seed : random state
    Returns
    -------
    NeurometricResult
    Notes
    -----
    - At each time index k, feature vector is ROI activity across trials: X[:, k, trials_k].
    - Standardized per time via StandardScaler within train folds.
    """
    rng = np.random.RandomState(seed)
    n_roi, n_time, n_trial = X_f1.shape
    M_pre = make_preF2_mask(t_f1, F2_idx)

    r2_cont = np.full(n_time, np.nan)
    auc_bin = np.full(n_time, np.nan)
    n_avail = np.zeros(n_time, dtype=int)

    # Binary labels for short/long
    y_bin_all = (ISI >= boundary).astype(int)

    for k in range(n_time):
        if preF2_only:
            trials_k = np.where(M_pre[k, :])[0]
        else:
            trials_k = np.arange(n_trial)

        if len(trials_k) < max(8, n_splits):  # need enough trials
            continue

        n_avail[k] = len(trials_k)

        Xk = X_f1[:, k, trials_k].T  # (n_trials_k, n_roi)
        y_cont = ISI[trials_k]
        y_bin = y_bin_all[trials_k]

        # Continuous: Ridge regression with KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng)
        preds = np.zeros_like(y_cont)
        for tr_idx, te_idx in kf.split(Xk):
            scaler = StandardScaler().fit(Xk[tr_idx])
            Xtr = scaler.transform(Xk[tr_idx])
            Xte = scaler.transform(Xk[te_idx])
            mdl = Ridge(alpha=1.0)
            mdl.fit(Xtr, y_cont[tr_idx])
            preds[te_idx] = mdl.predict(Xte)
        r2_cont[k] = r2_score(y_cont, preds)

        # Binary: Logistic regression with StratifiedKFold
        if len(np.unique(y_bin)) > 1:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng)
            proba = np.zeros_like(y_bin, dtype=float)
            for tr_idx, te_idx in skf.split(Xk, y_bin):
                scaler = StandardScaler().fit(Xk[tr_idx])
                Xtr = scaler.transform(Xk[tr_idx])
                Xte = scaler.transform(Xk[te_idx])
                clf = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=200)
                clf.fit(Xtr, y_bin[tr_idx])
                proba[te_idx] = clf.predict_proba(Xte)[:, 1]
            auc_bin[k] = roc_auc_score(y_bin, proba)

    return NeurometricResult(times=t_f1, r2_continuous=r2_cont, auc_binary=auc_bin, n_trials_available=n_avail)


# --------------------------------------
# 3) Hazard unique variance (pre-F2 GLM)
# --------------------------------------

def build_discrete_hazard(t_f1: np.ndarray, allowed_ISI: np.ndarray) -> np.ndarray:
    """
    Build a discrete-time hazard regressor for the F1->F2 interval given the set of allowed F2 times.
    Parameters
    ----------
    t_f1 : (n_time,) grid relative to F1
    allowed_ISI : (n_levels,), sorted unique allowed F2 times (seconds)
    Returns
    -------
    h : (n_time,) piecewise-constant hazard function that steps up only at allowed times.
    Notes
    -----
    For a uniform prior over K allowed times {τ1,...,τK}, the discrete hazard at τk is:
        h(τk) = P(F2 at τk | F2 >= τk) = 1 / (K - k + 1)
    We represent this as a piecewise-constant regressor that holds the *current* hazard value
    between allowed times, stepping up at each subsequent allowed time.
    """
    allowed = np.sort(np.unique(allowed_ISI))
    K = len(allowed)
    h = np.zeros_like(t_f1, dtype=float)
    current = 0.0
    for k, tau in enumerate(allowed, start=1):
        # For t in [previous_tau, tau), hazard holds the previous value
        # At t >= tau, hazard updates to 1/(K-k+1)
        current = 1.0 / (K - k + 1)
        h[t_f1 >= tau] = current
    # For t < first allowed time, hazard is 0
    h[t_f1 < allowed[0]] = 0.0
    return h


@dataclass
class HazardGLMResult:
    r2_full: np.ndarray          # (n_roi,)
    r2_reduced_time: np.ndarray  # (n_roi,) without hazard
    partial_r2_hazard: np.ndarray# (n_roi,)


def hazard_unique_variance(X_f1: np.ndarray, t_f1: np.ndarray, ISI: np.ndarray, F2_idx: np.ndarray,
                           allowed_ISI: Optional[np.ndarray]=None,
                           n_splits:int=5, seed:int=0) -> HazardGLMResult:
    """
    Estimate unique variance explained by a discrete hazard regressor beyond linear time terms.
    Parameters
    ----------
    X_f1 : (n_roi, n_time, n_trial) F1-aligned
    t_f1 : (n_time,)
    ISI : (n_trial,)
    F2_idx : (n_trial,)
    allowed_ISI : (n_levels,), optional; if None, uses sorted unique of ISI
    n_splits : CV folds (by trials)
    seed : random state
    Returns
    -------
    HazardGLMResult with partial R² for hazard per ROI.
    Notes
    -----
    - Design matrix has columns: [time, time^2, hazard(t)]
    - We fit using Ridge on concatenated pre-F2 frames, with trial-wise CV splits.
    - Partial R² for hazard = R²_full - R²_reduced_time evaluated on held-out data.
    """
    rng = np.random.RandomState(seed)
    n_roi, n_time, n_trial = X_f1.shape
    M_pre = make_preF2_mask(t_f1, F2_idx)

    if allowed_ISI is None:
        allowed_ISI = np.sort(np.unique(ISI))

    # Precompute regressors per trial (identical across trials for time and hazard, but we mask per trial)
    t_mat = np.repeat(t_f1[:, None], n_trial, axis=1)
    time = t_mat
    time2 = t_mat**2
    hazard_vec = build_discrete_hazard(t_f1, allowed_ISI)
    hazard = np.repeat(hazard_vec[:, None], n_trial, axis=1)

    # Prepare trial ids per frame for CV grouping
    trial_ids_full = np.repeat(np.arange(n_trial), n_time).reshape(n_time, n_trial, order='F')
    trial_ids_vec = trial_ids_full.ravel(order='F')[M_pre.ravel(order='F')]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng)

    r2_full = np.zeros(n_roi)
    r2_red  = np.zeros(n_roi)

    for r in range(n_roi):
        y = X_f1[r, :, :]

        # Concat with mask for FULL model
        y_vec, X_full = concat_trials_by_mask(y, [time, time2, hazard], M_pre)
        # Reduced (time + time^2 only)
        _, X_red = concat_trials_by_mask(y, [time, time2], M_pre)

        # CV by trials
        scores_full = []
        scores_red  = []
        for tr_idx, te_idx in kf.split(np.zeros_like(trial_ids_vec), groups=trial_ids_vec):
            mdlF = Ridge(alpha=1.0)
            mdlR = Ridge(alpha=1.0)
            mdlF.fit(X_full[tr_idx], y_vec[tr_idx])
            mdlR.fit(X_red[tr_idx],  y_vec[tr_idx])
            yhatF = mdlF.predict(X_full[te_idx])
            yhatR = mdlR.predict(X_red[te_idx])
            scores_full.append(r2_score(y_vec[te_idx], yhatF))
            scores_red.append(r2_score(y_vec[te_idx], yhatR))
        r2_full[r] = np.mean(scores_full)
        r2_red[r]  = np.mean(scores_red)

    return HazardGLMResult(r2_full=r2_full, r2_reduced_time=r2_red, partial_r2_hazard=(r2_full - r2_red))


# --------------------------------------
# 4) Prediction Error at F2
# --------------------------------------

@dataclass
class PESlopeResult:
    slope: np.ndarray        # (n_roi,)
    pval: np.ndarray         # (n_roi,)
    surprise_name: str       # description of PE regressor used


def benjamini_hochberg(pvals: np.ndarray, alpha: float=0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR control (returns boolean mask of discoveries).
    """
    p = np.asarray(pvals)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n+1)
    thresh = (ranks / n) * alpha
    passed = p[order] <= thresh
    if not np.any(passed):
        return np.zeros_like(p, dtype=bool)
    kmax = np.max(np.where(passed)[0])
    cutoff = thresh[kmax]
    return p <= cutoff


def prediction_error_at_F2(X_f2: np.ndarray, t_f2: np.ndarray, ISI: np.ndarray,
                           window: Tuple[float,float]=(0.0, 0.25),
                           surprise: str='abs_deviation') -> PESlopeResult:
    """
    Regress post-F2 response on a trial-wise timing surprise metric.
    Parameters
    ----------
    X_f2 : array (n_roi, n_time2, n_trial), F2-aligned
    t_f2 : array (n_time2,)
    ISI : array (n_trial,)
    window : (start, stop) seconds relative to F2 to summarize response (AUC/mean)
    surprise : 'abs_deviation' or 'neglog_uniform'
        - 'abs_deviation': |ISI - mean(ISI)|
        - 'neglog_uniform': -log P(F2 at t) under uniform over allowed ISIs (constant across trials -> not informative)
    Returns
    -------
    PESlopeResult
    Notes
    -----
    - For uniform ISIs, -log Uniform is constant → use 'abs_deviation' by default.
    - Response summary uses mean over the window (can be changed to peak/AUC as needed).
    """
    n_roi, n_time2, n_trial = X_f2.shape
    # Build response summary
    idx = np.where((t_f2 >= window[0]) & (t_f2 <= window[1]))[0]
    if len(idx) == 0:
        raise ValueError("prediction_error_at_F2: empty window; adjust 'window' to match t_f2 range.")
    Y = np.nanmean(X_f2[:, idx, :], axis=1)  # (n_roi, n_trial)

    # Build surprise regressor
    if surprise == 'abs_deviation':
        pe = np.abs(ISI - np.mean(ISI))
        sname = "abs(ISI - mean ISI)"
    elif surprise == 'neglog_uniform':
        # NOTE: for uniform discrete ISIs, -log P is constant and uninformative; included for completeness.
        K = len(np.unique(ISI))
        pe = np.full(n_trial, -np.log(1.0 / K))
        sname = "-log P (uniform)"
    else:
        raise ValueError("Unknown surprise type. Use 'abs_deviation' or 'neglog_uniform'.")

    # Regress per ROI (simple slope + pval)
    slope = np.zeros(n_roi)
    pval  = np.ones(n_roi)
    for r in range(n_roi):
        lr = linregress(pe, Y[r, :])
        slope[r] = lr.slope
        pval[r]  = lr.pvalue

    return PESlopeResult(slope=slope, pval=pval, surprise_name=sname)


# --------------------------------------
# 5) Raster sort index
# --------------------------------------

def raster_sort_index(delta_R2: np.ndarray, alpha_opt: np.ndarray,
                      X_f1: np.ndarray, t_f1: np.ndarray, F2_idx: np.ndarray,
                      mode:str='auto') -> np.ndarray:
    """
    Build a single sort index for rasters consistent with the winning timing story.
    Parameters
    ----------
    delta_R2 : (n_roi,) R2_scale - R2_clock
    alpha_opt : (n_roi,) optimal scaling exponent per ROI
    X_f1, t_f1, F2_idx : used to compute peak latency/phase for sorting
    mode : 'auto' | 'scale' | 'clock'
        - 'auto': if median(delta_R2) > 0 → use phase; else use absolute time.
    Returns
    -------
    idx : np.ndarray of ROI indices in the desired plot order
    Notes
    -----
    - For scaling: sort by preferred phase (peak pre-F2 on phase grid).
    - For clock: sort by absolute latency to pre-F2 peak (ms).
    """
    n_roi, n_time, n_trial = X_f1.shape
    M_pre = make_preF2_mask(t_f1, F2_idx)

    if mode == 'auto':
        mode = 'scale' if np.nanmedian(delta_R2) > 0 else 'clock'

    if mode == 'scale':
        # Estimate preferred phase via normalized time within each trial; average over trials.
        t_mat = np.repeat(t_f1[:, None], n_trial, axis=1)
        phase = t_mat / np.repeat((F2_idx.astype(float) / (1.0 / (t_f1[1]-t_f1[0])))[None, :], n_time, axis=0)  # rough phase using indices
        phase = np.clip(phase, 0.0, 1.0 - 1e-9)
        pref = np.zeros(n_roi)
        for r in range(n_roi):
            Yr = X_f1[r, :, :].copy()
            Yr[~M_pre] = np.nan
            # Weighted average phase of the pre-F2 peak
            with np.errstate(invalid='ignore'):
                peak_mask = (Yr == np.nanmax(Yr, axis=0, keepdims=True))
            # For each trial, take phase at peak; then median across trials
            phases = []
            for j in range(n_trial):
                col = Yr[:, j]
                if not np.any(np.isfinite(col)): continue
                k = np.nanargmax(col)
                phases.append(phase[k, j])
            pref[r] = np.nan if len(phases)==0 else float(np.nanmedian(phases))
        order = np.argsort(np.nan_to_num(pref, nan=np.inf))

    else:  # 'clock'
        # Preferred absolute latency (ms) pre-F2
        pref = np.zeros(n_roi)
        for r in range(n_roi):
            Yr = X_f1[r, :, :].copy()
            Yr[~M_pre] = np.nan
            lats = []
            for j in range(n_trial):
                col = Yr[:, j]
                if not np.any(np.isfinite(col)): continue
                k = np.nanargmax(col)
                lats.append(t_f1[k])
            pref[r] = np.nan if len(lats)==0 else float(np.nanmedian(lats))
        order = np.argsort(np.nan_to_num(pref, nan=np.inf))

    return order


# --------------------------------------
# 6) Split-half reliability
# --------------------------------------

@dataclass
class SplitHalfResult:
    deltaR2_agreement: float
    alpha_corr: float


def split_half_reliability(X_f1: np.ndarray, t_f1: np.ndarray, ISI: np.ndarray, F2_idx: np.ndarray,
                           seed:int=0) -> SplitHalfResult:
    """
    Compute basic split-half reliability for ΔR² sign and α estimates.
    - Splits trials into halves stratified by ISI levels.
    """
    rng = np.random.RandomState(seed)
    n_roi, n_time, n_trial = X_f1.shape
    # Stratify by ISI
    levels = np.unique(ISI)
    halfA = []
    halfB = []
    for lev in levels:
        idx = np.where(ISI == lev)[0]
        rng.shuffle(idx)
        m = len(idx)//2
        halfA.extend(idx[:m])
        halfB.extend(idx[m:])
    halfA = np.array(halfA, dtype=int)
    halfB = np.array(halfB, dtype=int)
    if len(halfA)==0 or len(halfB)==0:
        return SplitHalfResult(deltaR2_agreement=np.nan, alpha_corr=np.nan)

    resA = scaling_vs_clock(X_f1[:,:,halfA], t_f1, ISI[halfA], F2_idx[halfA], seed=seed)
    resB = scaling_vs_clock(X_f1[:,:,halfB], t_f1, ISI[halfB], F2_idx[halfB], seed=seed+1)

    signA = np.sign(resA.delta_R2)
    signB = np.sign(resB.delta_R2)
    # Agreement fraction ignoring NaNs
    valid = np.isfinite(signA) & np.isfinite(signB)
    agree = np.mean((signA[valid] == signB[valid]).astype(float)) if np.any(valid) else np.nan

    # Correlation of alpha (Spearman)
    valid_a = np.isfinite(resA.alpha_opt) & np.isfinite(resB.alpha_opt)
    if np.any(valid_a):
        aA = resA.alpha_opt[valid_a]
        aB = resB.alpha_opt[valid_a]
        # Spearman rho via rank corr
        rho = np.corrcoef(rankdata(aA), rankdata(aB))[0,1]
    else:
        rho = np.nan

    return SplitHalfResult(deltaR2_agreement=agree, alpha_corr=rho)


# -----------------------------
# Convenience: binary labels
# -----------------------------

def primary_timing_label(delta_R2: np.ndarray, alpha_opt: np.ndarray, thresh: float=0.0) -> np.ndarray:
    """
    Quick label per ROI: 'scaling' if delta_R2 > thresh, else 'clock'.
    Returns an array of strings length n_roi.
    """
    labels = np.where(delta_R2 > thresh, 'scaling', 'clock').astype(object)
    # Optionally tag mixed if near zero:
    near = np.isfinite(delta_R2) & (np.abs(delta_R2) < 0.02)
    labels[near] = 'mixed'
    return labels




def _pick_trials_stratified_by_isi(isi: np.ndarray, per_level: int, random_state: int = 0) -> np.ndarray:
    """
    Pick up to 'per_level' trials per unique ISI level (stratified subsample).
    Returns indices into trials.
    """
    rng = np.random.RandomState(random_state)
    isi = np.asarray(isi)
    idx_keep = []
    for lev in np.sort(np.unique(isi)):
        idx = np.where(isi == lev)[0]
        if idx.size <= per_level:
            idx_keep.extend(idx.tolist())
        else:
            rng.shuffle(idx)
            idx_keep.extend(idx[:per_level].tolist())
    return np.array(sorted(idx_keep), dtype=int)


def thin_contract(
    X: np.ndarray,      # (n_roi, n_time, n_trial)
    t: np.ndarray,      # (n_time,)
    ISI: np.ndarray,    # (n_trial,)
    F2_idx: np.ndarray, # (n_trial,)
    roi_max: Optional[int] = None,
    trials_per_level: Optional[int] = None,
    time_stride: int = 1,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return a thinned version of the contract for quick iteration.
    - roi_max: keep first N ROIs
    - trials_per_level: keep up to N trials per ISI level (stratified)
    - time_stride: keep every k-th time sample (recompute F2_idx accordingly)

    Returns: X_thin, t_thin, ISI_thin, F2_idx_thin, roi_idx, trial_idx
    """
    X_thin = X
    t_thin = t
    ISI_thin = ISI
    F2_idx_thin = F2_idx.astype(int)

    # Subsample trials (stratified by ISI)
    trial_idx = np.arange(X.shape[2], dtype=int)
    if trials_per_level is not None:
        trial_idx = _pick_trials_stratified_by_isi(ISI, trials_per_level, random_state=random_state)
        X_thin = X_thin[:, :, trial_idx]
        ISI_thin = ISI[trial_idx]
        F2_idx_thin = F2_idx_thin[trial_idx]

    # Subsample ROIs
    roi_idx = np.arange(X_thin.shape[0], dtype=int)
    if roi_max is not None and roi_max < X_thin.shape[0]:
        roi_idx = roi_idx[:roi_max]
        X_thin = X_thin[roi_idx, :, :]

    # Downsample time
    if time_stride is not None and time_stride > 1:
        t_thin = t_thin[::time_stride]
        X_thin = X_thin[:, ::time_stride, :]
        # Recompute F2 indices under downsampling
        F2_idx_thin = np.floor(F2_idx_thin / time_stride).astype(int)
        F2_idx_thin = np.clip(F2_idx_thin, 0, len(t_thin) - 1)

    return X_thin, t_thin, ISI_thin, F2_idx_thin, roi_idx, trial_idx









def _bin_isi_for_stratification(isi: np.ndarray, n_bins: int = None) -> np.ndarray:
    """Map continuous/discrete ISI to bins for stratified CV (default: unique ISI levels)."""
    isi = np.asarray(isi, float)
    if n_bins is None:
        uniq = np.sort(np.unique(np.round(isi, 6)))
        return np.searchsorted(uniq, np.round(isi, 6))
    # optional equal-frequency binning
    qs = np.quantile(isi, np.linspace(0, 1, n_bins + 1))
    return np.clip(np.digitize(isi, qs[1:-1], right=True), 0, n_bins - 1)

def _make_cv(y: np.ndarray = None, groups: np.ndarray = None, n_splits: int = 5,
             shuffle: bool = True, random_state: int = 0):
    """
    Choose a CV splitter:
      - StratifiedGroupKFold if y and groups provided (sklearn>=1.1)
      - GroupKFold if only groups
      - StratifiedKFold if only y
      - KFold otherwise
    """
    if y is not None and groups is not None and StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if groups is not None and GroupKFold is not None:
        return GroupKFold(n_splits=n_splits)
    if y is not None:
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)



# -----------------------------
# Example usage (pseudo)
# -----------------------------
# from roi_timing_analysis import (
#     zscore_across_trials, make_preF2_mask, resample_to_phase,
#     scaling_vs_clock, time_resolved_decoding,
#     hazard_unique_variance, prediction_error_at_F2,
#     raster_sort_index, split_half_reliability, primary_timing_label
# )
#
# # 0) Ensure your df/arrays match the INPUT DATA CONTRACT at the top.
# Xz = zscore_across_trials(X_f1, baseline_idx=(t_f1<0))
#
# # 1) Scaling vs clock
# sc_res = scaling_vs_clock(Xz, t_f1, ISI, F2_idx)
# labels = primary_timing_label(sc_res.delta_R2, sc_res.alpha_opt)
#
# # 2) Neurometric decoding (choose boundary = midpoint between short/long sets)
# boundary = 0.5*(np.max(short_ISIs)+np.min(long_ISIs))
# neu = time_resolved_decoding(Xz, t_f1, ISI, F2_idx, boundary=boundary, preF2_only=True)
#
# # 3) Hazard unique variance
# haz = hazard_unique_variance(Xz, t_f1, ISI, F2_idx, allowed_ISI=np.unique(ISI))
#
# # 4) Prediction error at F2 (needs F2-aligned data)
# pe = prediction_error_at_F2(X_f2, t_f2, ISI, window=(0.0, 0.25), surprise='abs_deviation')
#
# # 5) Raster sort
# order = raster_sort_index(sc_res.delta_R2, sc_res.alpha_opt, Xz, t_f1, F2_idx, mode='auto')
#
# # 6) Split-half
# sh = split_half_reliability(Xz, t_f1, ISI, F2_idx)
#
# # You can now plot rasters using 'order', and summarize sc_res, neu, haz, pe, and sh.
