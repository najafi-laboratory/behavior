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
import pandas as pd
from scipy import stats
from matplotlib.gridspec import GridSpec


# Expected input:
# chunks = {
#   'trial_start': trial_data_trial_start,
#   'isi':          trial_data_isi,
#   'start_flash_1': trial_data_start_flash_1,
#   ...
# }
#
# Each trial_data_* has:
#   - 'df_trials_with_segments' DataFrame with identical row order across chunks
#       columns: dff_segment (n_rois x n_time_chunk), dff_time_vector (n_time_chunk,)
#                isi, is_short, is_right, is_right_choice, rewarded, punished,
#                did_not_choose, time_did_not_choose, choice_start, choice_stop,
#                servo_in, servo_out, lick, lick_start, RT,
#                start_flash_1, end_flash_1, start_flash_2, end_flash_2
#
# Only difference between chunks: alignment event & time window / length.

_CHUNK_META_DEFAULTS = {
    'trial_start':    dict(alignment='trial_start'),
    'isi':            dict(alignment='end_flash_1', mask_rule='strict_pre_F2'),
    'start_flash_1':  dict(alignment='start_flash_1'),
    'end_flash_1':    dict(alignment='end_flash_1'),
    'start_flash_2':  dict(alignment='start_flash_2'),
    'end_flash_2':    dict(alignment='end_flash_2'),
    'choice_start':   dict(alignment='choice_start'),
    'lick_start':     dict(alignment='lick_start'),
}

def _to_seconds_if_ms(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    if np.nanmax(np.abs(a)) > 10.0:  # treat as ms
        return a / 1000.0
    return a




def _stack_chunk(df, n_trials: int, n_rois: int, *, strict: bool = True, chunk_name: str = "?") -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (traces, time_vector)
      traces: (n_trials, n_rois, n_time_chunk) float32
      time_vector: (n_time_chunk,) float (canonical per chunk)
    If a row has malformed dff_segment (None, scalar, wrong ndim), it is filled with NaNs.
    """
    # canonical time from first valid row
    first_valid_idx = None
    for j in range(len(df)):
        tv_candidate = np.asarray(df.iloc[j]['dff_time_vector'], dtype=object)
        if np.ndim(tv_candidate) == 1 and len(tv_candidate) > 0:
            first_valid_idx = j
            break
    if first_valid_idx is None:
        raise ValueError(f"[{chunk_name}] No valid dff_time_vector found.")
    t0 = np.asarray(df.iloc[first_valid_idx]['dff_time_vector'], dtype=float)
    if t0.ndim != 1:
        raise ValueError(f"[{chunk_name}] Canonical time vector not 1D (ndim={t0.ndim}).")
    Nt = t0.size
    out = np.full((n_trials, n_rois, Nt), np.nan, dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        t_vec_raw = row.get('dff_time_vector', None)
        dseg_raw = row.get('dff_segment', None)

        # Convert safely
        try:
            t_vec = np.asarray(t_vec_raw, dtype=float)
        except Exception:
            if strict:
                raise ValueError(f"[{chunk_name}] Row {i}: invalid dff_time_vector type={type(t_vec_raw)}")
            continue

        dat = np.asarray(dseg_raw)  # leave dtype flexible until checked

        # Validate dat
        if dat is None or dat.ndim == 0:
            if strict:
                print(f"[warn][{chunk_name}] Row {i}: dff_segment is scalar/None; leaving NaNs.")
            continue
        if dat.ndim != 2:
            if strict:
                raise ValueError(f"[{chunk_name}] Row {i}: dff_segment ndim={dat.ndim} !=2 (shape={dat.shape})")
            else:
                print(f"[warn][{chunk_name}] Row {i}: unexpected ndim={dat.ndim}; skipping.")
                continue
        if dat.shape[0] != n_rois:
            msg = f"[{chunk_name}] Row {i}: ROI mismatch (got {dat.shape[0]}, expected {n_rois})."
            if strict:
                raise ValueError(msg)
            else:
                print("[warn]" + msg)
                continue

        # Fast path if time matches
        if t_vec.shape[0] == Nt and np.allclose(t_vec, t0, atol=1e-9, rtol=0):
            out[i] = dat.astype(np.float32, copy=False)
            continue

        # Interpolate each ROI within native support
        if t_vec.ndim != 1 or t_vec.size < 2:
            if strict:
                print(f"[warn][{chunk_name}] Row {i}: invalid time vector (size={t_vec.size}); skipping.")
            continue

        for r in range(n_rois):
            y = np.asarray(dat[r], dtype=float)
            m = np.isfinite(y)
            if m.sum() < 2:
                continue
            tv = t_vec[m]; yv = y[m]
            seg = (t0 >= tv.min()) & (t0 <= tv.max())
            if seg.any():
                out[i, r, seg] = np.interp(t0[seg], tv, yv).astype(np.float32, copy=False)
    return out, t0






# --- Simple sanity print (optional helper) ----------------------------------
def summarize_M(M: Dict[str, Any]) -> None:
    print("=== M SUMMARY ===")
    print(f"Trials={M['n_trials']}  ROIs={M['n_rois']}")
    print(f"Primary chunk: {M['primary_chunk']}")
    print(f"ISI range (s): {np.nanmin(M['isi']):.3f} .. {np.nanmax(M['isi']):.3f}")
    print(f"Unique ISIs: {M['unique_isis']}")
    for k, info in M['chunks'].items():
        d = info['data']
        print(f"  Chunk '{k}': shape={d.shape}  window={info['window']}  dt={info['dt']:.4f}s  NaN%={info['nan_frac']*100:.1f}  align={info['alignment']}")
    print("=================")






















def make_isi_phase_matrix(M: Dict[str, Any],
                          chunk_key: str = 'isi',
                          clamp: bool = True) -> np.ndarray:
    """
    Returns phase matrix (trials, time) with φ=t/ISI in [0,1]; NaN where original data NaN or t<0.
    """
    ch = M['chunks'][chunk_key]
    t = ch['time']  # (T,)
    X = ch['data']  # (trials, rois, T)
    isi = M['isi']  # (trials,)
    trials, _, T = X.shape
    phase = np.full((trials, T), np.nan, dtype=np.float32)
    for i in range(trials):
        if not np.isfinite(isi[i]) or isi[i] <= 0:
            continue
        valid = (t >= 0) & (t < isi[i])
        if not np.any(valid):
            continue
        phi = t[valid] / isi[i]
        if clamp:
            phi = np.clip(phi, 0.0, 1.0)
        phase[i, valid] = phi.astype(np.float32, copy=False)
    return phase

def trim_all_nan_columns(arr: np.ndarray, axis: int = -1) -> Tuple[np.ndarray, slice]:
    """
    Remove leading/trailing all-NaN columns along time axis.
    Returns trimmed array and slice used.
    """
    assert axis == -1
    mask_valid = ~np.all(np.isnan(arr), axis=tuple(range(arr.ndim-1)))
    if mask_valid.all():
        return arr, slice(0, arr.shape[-1])
    idx = np.where(mask_valid)[0]
    sl = slice(idx[0], idx[-1]+1)
    return arr[..., sl], sl


def build_bagged_trial_means(M: Dict[str, Any],
                             chunk: str,
                             B: int = 40,
                             sample_frac: float = 0.7,
                             stratify_isi: bool = True,
                             random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (tensor, time) where tensor shape = (rois, time, B).
    Each bag = mean over sampled trials (ignores NaNs).
    """
    rng = np.random.default_rng(random_state)
    ch = M['chunks'][chunk]
    X = ch['data']  # (trials, rois, time)
    t = ch['time']
    trials, rois, T = X.shape
    out = np.full((rois, T, B), np.nan, dtype=np.float32)

    if stratify_isi and chunk == 'isi':
        isi = M['isi']
        qs = np.quantile(isi, [0, 1/3, 2/3, 1])
        strata = [np.where((isi >= a - 1e-9) & (isi <= b + 1e-9))[0] for a, b in zip(qs[:-1], qs[1:])]
    else:
        strata = [np.arange(trials)]

    for b in range(B):
        sel_idx = []
        for s in strata:
            if len(s) == 0:
                continue
            k = max(1, int(len(s) * sample_frac))
            sel_idx.append(rng.choice(s, size=k, replace=True))
        if not sel_idx:
            continue
        sel_idx = np.concatenate(sel_idx)
        if sel_idx.size == 0:
            continue
        X_sel = X[sel_idx]  # (sel, rois, time)
        # Skip bag if everything is NaN
        if not np.any(np.isfinite(X_sel)):
            continue
        bag_data = safe_nanmean(X_sel, axis=0)  # (rois, time)
        out[:, :, b] = bag_data
    return out, t
# --- Minimal masked CP (fallback) -------------------------------------------



def extract_roi_sets(A: np.ndarray,
                     method: str = 'top_q',
                     q: float = 0.10,
                     min_rois: int = 5) -> List[np.ndarray]:
    """
    A: (rois, components)
    Returns list of ROI index arrays per component.
    """
    sets = []
    R = A.shape[1]
    for r in range(R):
        v = A[:, r]
        v_pos = np.maximum(v, 0)
        if np.all(v_pos == 0):
            sets.append(np.array([], int))
            continue
        if method == 'top_q':
            k = max(min_rois, int(len(v_pos) * q))
            idx = np.argsort(v_pos)[::-1][:k]
        elif method == 'otsu':
            from skimage.filters import threshold_otsu
            thr = threshold_otsu(v_pos[v_pos > 0])
            idx = np.where(v_pos >= thr)[0]
        elif method == 'mad':
            med = np.median(v_pos[v_pos > 0])
            mad = np.median(np.abs(v_pos[v_pos > 0] - med)) + 1e-12
            thr = med + 1.5 * mad
            idx = np.where(v_pos >= thr)[0]
        else:
            raise ValueError("method must be top_q|otsu|mad")
        sets.append(np.sort(idx))
    return sets



def clean_tensor_for_cp(tensor: np.ndarray, min_time_non_nan_frac: float = 0.01):
    """
    tensor: (rois, time, bags)
    Removes:
      - time bins with all NaN OR with < min_time_non_nan_frac * rois non-NaN across all bags
      - bags with all NaN
    Returns cleaned tensor and (time_keep_idx, bag_keep_idx)
    """
    rois, T, B = tensor.shape
    # time keep
    valid_time_counts = np.sum(np.any(~np.isnan(tensor), axis=0), axis=1)  # (T,)
    time_keep = valid_time_counts >= max(1, int(min_time_non_nan_frac * rois))
    # bag keep
    bag_keep = np.any(~np.isnan(tensor), axis=(0,1))
    cleaned = tensor[:, time_keep][:, :, bag_keep]
    return cleaned, np.where(time_keep)[0], np.where(bag_keep)[0]

def cp_decompose_masked(X: np.ndarray,
                        mask: np.ndarray,
                        rank: int,
                        n_iter: int = 150,
                        tol: float = 1e-5,
                        ridge: float = 1e-3,
                        random_state: Optional[int] = None,
                        verbose: bool = False) -> Dict[str, Any]:
    """
    ALS CP with ridge regularization and mask (True=keep).
    X: (I,J,K)
    mask: same shape
    """
    rng = np.random.default_rng(random_state)
    I, J, K = X.shape
    R = rank
    # init (scaled)
    A = rng.standard_normal((I, R)) * 0.1
    B = rng.standard_normal((J, R)) * 0.1
    C = rng.standard_normal((K, R)) * 0.1
    M = mask.astype(float)

    def khatri(m1, m2):
        return (m1[:, None, :] * m2[None, :, :]).reshape(m1.shape[0]*m2.shape[0], R)

    prev_loss = None
    eyeR = np.eye(R)

    for it in range(n_iter):
        # Update A
        Z = khatri(C, B)            # (KJ,R)
        X1 = (X * M).reshape(I, -1)
        M1 = M.reshape(I, -1)
        ZtZ = Z.T @ (Z)             # precompute global (approx); mask applied row-wise
        for i in range(I):
            W = M1[i] > 0
            if W.sum() < R:
                continue
            Zw = Z[W]
            xw = X1[i, W]
            G = Zw.T @ Zw + ridge * eyeR
            try:
                A[i] = np.linalg.solve(G, Zw.T @ xw)
            except np.linalg.LinAlgError:
                # fallback leave previous
                pass

        # Update B
        Z = khatri(C, A)            # (KI,R)
        X2 = np.moveaxis(X * M, 1, 0).reshape(J, -1)
        M2 = np.moveaxis(M, 1, 0).reshape(J, -1)
        for j in range(J):
            W = M2[j] > 0
            if W.sum() < R:
                continue
            Zw = Z[W]
            xw = X2[j, W]
            G = Zw.T @ Zw + ridge * eyeR
            try:
                B[j] = np.linalg.solve(G, Zw.T @ xw)
            except np.linalg.LinAlgError:
                pass

        # Update C
        Z = khatri(B, A)            # (JI,R)
        X3 = np.moveaxis(X * M, 2, 0).reshape(K, -1)
        M3 = np.moveaxis(M, 2, 0).reshape(K, -1)
        for k in range(K):
            W = M3[k] > 0
            if W.sum() < R:
                continue
            Zw = Z[W]
            xw = X3[k, W]
            G = Zw.T @ Zw + ridge * eyeR
            try:
                C[k] = np.linalg.solve(G, Zw.T @ xw)
            except np.linalg.LinAlgError:
                pass

        # Normalize components
        lam = np.linalg.norm(A, axis=0)
        lam[lam == 0] = 1
        A /= lam; B /= lam; C /= lam

        # Loss
        recon = np.zeros_like(X)
        for r in range(R):
            recon += np.outer(A[:, r], B[:, r])[:, :, None] * C[:, r][None, None, :]
        diff = (X - recon) * M
        loss = np.nansum(diff**2) / (M.sum() + 1e-12)

        if verbose and it % 10 == 0:
            print(f"[CP] it={it} loss={loss:.6g}")

        if prev_loss is not None:
            rel = (prev_loss - loss) / (abs(prev_loss) + 1e-12)
            if rel < tol:
                break
        prev_loss = loss

    return dict(A=A, B=B, C=C, lambda_=lam, loss=loss, it=it+1)

# def run_chunk_cp_pipeline(M: Dict[str, Any],
#                           chunk: str,
#                           ranks: List[int] = [8, 10, 12],
#                           B: int = 40,
#                           sample_frac: float = 0.7,
#                           stratify_isi: bool = True,
#                           standardize_rois: bool = True,
#                           random_state: Optional[int] = 0) -> Dict[str, Any]:
#     """
#     Bag means → clean → (optional) standardize → masked CP over candidate ranks → pick best by loss.
#     """
#     tensor, t = build_bagged_trial_means(M, chunk, B=B,
#                                          sample_frac=sample_frac,
#                                          stratify_isi=stratify_isi,
#                                          random_state=random_state)  # (rois,time,bags)

#     # Clean tensor (remove fully NaN time bins / bags)
#     tensor_clean, time_keep, bag_keep = clean_tensor_for_cp(tensor)
#     t_clean = t[time_keep]
#     if tensor_clean.size == 0:
#         raise ValueError("After cleaning, tensor is empty.")

#     # Mask & fill
#     mask = ~np.isnan(tensor_clean)
#     X = np.nan_to_num(tensor_clean, nan=0.0)

#     # Optional standardization across time per ROI (aggregate over bags)
#     if standardize_rois:
#         mean_roi = np.sum(X, axis=2).mean(axis=1, keepdims=True) / max(1, X.shape[2])
#         std_roi = np.sqrt(np.sum((X - mean_roi[:, :, None])**2, axis=(1,2)) / (X.shape[1]*X.shape[2]-1 + 1e-9))
#         std_roi[std_roi == 0] = 1
#         X = X / std_roi[:, None, None]

#     results = []
#     for r in ranks:
#         try:
#             res = cp_decompose_masked(X, mask, rank=r, random_state=random_state)
#             res['rank'] = r
#             results.append(res)
#         except Exception as e:
#             print(f"[warn] rank {r} failed: {e}")

#     if not results:
#         raise RuntimeError("All CP decompositions failed.")

#     best = min(results, key=lambda d: d['loss'])
#     roi_sets = extract_roi_sets(best['A'], method='top_q')
#     return dict(
#         chunk=chunk,
#         time=t_clean,
#         time_indices=time_keep,
#         bag_indices=bag_keep,
#         factors=best,
#         roi_sets=roi_sets,
#         tested_ranks=[r for r in ranks],
#         all_results=results
#     )
    







































# ...existing code...

def component_temporal_stats(B: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
    """
    B: (time, R)
    Returns dict of temporal variance, peak-to-peak, max abs, and normalized energy.
    """
    var = np.var(B, axis=0)
    ptp = np.ptp(B, axis=0)
    max_abs = np.max(np.abs(B), axis=0)
    energy = np.sum(B**2, axis=0)
    energy /= (energy.max() + 1e-12)
    return dict(var=var, ptp=ptp, max_abs=max_abs, energy=energy)

def loading_gini(v: np.ndarray) -> float:
    """
    Gini coefficient (0=uniform, 1=all mass in one element) for non-negative vector.
    """
    x = np.sort(np.maximum(v, 0))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    cum = np.cumsum(x)
    g = (n + 1 - 2 * (cum.sum() / cum[-1])) / n
    return g

def adaptive_threshold_one(v_pos: np.ndarray,
                           method: str = 'auto',
                           min_rois: int = 5,
                           max_rois: Optional[int] = None) -> np.ndarray:
    """
    Returns indices selected from non-negative loadings v_pos.
    method:
      - 'auto': tries Otsu; fallback to top_k based on cumulative contribution
      - 'otsu': Otsu threshold
      - 'gini': choose smallest set reaching 60% loading mass or at least min_rois
    """
    v_pos = np.asarray(v_pos)
    if v_pos.sum() == 0:
        return np.array([], int)

    # Normalize for cumulative strategies
    order = np.argsort(v_pos)[::-1]
    v_sorted = v_pos[order]
    if method == 'otsu' or method == 'auto':
        try:
            from skimage.filters import threshold_otsu
            thr = threshold_otsu(v_pos[v_pos > 0])
            idx = np.where(v_pos >= thr)[0]
            if method == 'auto' and (idx.size < min_rois):
                # fallback cumulative
                cum = np.cumsum(v_sorted) / v_sorted.sum()
                k = np.searchsorted(cum, 0.6) + 1
                k = max(k, min_rois)
                idx = order[:k]
        except Exception:
            cum = np.cumsum(v_sorted) / v_sorted.sum()
            k = np.searchsorted(cum, 0.6) + 1
            k = max(k, min_rois)
            idx = order[:k]
    elif method == 'gini':
        cum = np.cumsum(v_sorted) / v_sorted.sum()
        k = np.searchsorted(cum, 0.6) + 1
        k = max(k, min_rois)
        idx = order[:k]
    else:
        raise ValueError("method must be auto|otsu|gini")

    if max_rois is not None and idx.size > max_rois:
        idx = idx[:max_rois]
    return np.sort(idx)

def extract_roi_sets_adaptive(A: np.ndarray,
                              min_rois: int = 5,
                              method: str = 'auto') -> List[np.ndarray]:
    """
    Adaptive extraction per component using positive loadings and data-driven threshold.
    """
    sets = []
    R = A.shape[1]
    for r in range(R):
        v = np.maximum(A[:, r], 0)
        idx = adaptive_threshold_one(v, method=method, min_rois=min_rois)
        sets.append(idx)
    return sets





# ...existing code...

def prune_components(A: np.ndarray,
                     B: np.ndarray,
                     roi_sets: List[np.ndarray],
                     time_var: np.ndarray,
                     ptp: np.ndarray,
                     min_time_var: float = 1e-4,
                     min_ptp: float = 1e-2,
                     max_jaccard: float = 0.8) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Remove low-information or redundant components BEFORE stability filtering.

    Rules
    -----
    1. Drop if temporal variance < min_time_var OR peak-to-peak < min_ptp OR
       ROI set empty.
    2. Greedy redundancy: for each surviving r, if Jaccard(roi_sets[r], roi_sets[q]) > max_jaccard
       with any earlier kept q -> drop r.

    Parameters
    ----------
    A, B : factor matrices
    roi_sets : list[np.ndarray]
        ROI indices per component (from adaptive or threshold method).
    time_var, ptp : np.ndarray
        Per-component temporal stats (variance; peak-to-peak amplitude).
    max_jaccard : float
        Upper bound on allowed overlap (0..1).

    Returns
    -------
    A2, B2, roi_sets2, keep_mask

    Notes
    -----
    * keep_mask indexes original ordering.
    * Choose thresholds relative to window length (see adaptive_pruning_thresholds).
    """
    R = A.shape[1]
    keep = np.ones(R, dtype=bool)

    for r in range(R):
        if time_var[r] < min_time_var or ptp[r] < min_ptp or len(roi_sets[r]) == 0:
            keep[r] = False

    def jaccard(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 and b.size == 0:
            return 1.0
        inter = len(np.intersect1d(a, b))   # FIX: was np.intersection1d
        if inter == 0:
            return 0.0
        union = len(np.union1d(a, b))
        return inter / union if union else 0.0

    for r in range(R):
        if not keep[r]:
            continue
        for q in range(r):
            if keep[q] and jaccard(roi_sets[r], roi_sets[q]) > max_jaccard:
                keep[r] = False
                break

    A2 = A[:, keep]
    B2 = B[:, keep]
    roi_sets2 = [roi_sets[i] for i in range(R) if keep[i]]
    return A2, B2, roi_sets2, keep



def filter_components_by_stability(A: np.ndarray,
                                   B: np.ndarray,
                                   roi_sets: List[np.ndarray],
                                   stability_corr: np.ndarray,
                                   keep_mask: np.ndarray,
                                   min_abs_corr: float = 0.6) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Final stability-based pruning.

    Parameters
    ----------
    A, B, roi_sets : post-prune (already reduced) factors.
    stability_corr : (n_kept,) array
        Per-component absolute loading correlations (aligned to A columns).
    keep_mask : bool
        Historical mask mapping to original pre-prune components
        (passed for external bookkeeping; not modified here).
    min_abs_corr : float
        Components below threshold removed.

    Returns
    -------
    A2, B2, roi_sets2, stable_mask
      stable_mask length == len(roi_sets); True for retained.

    Guideline
    ---------
    * Typical threshold 0.6–0.7. Increase for stricter reproducibility.
    """
    if stability_corr.shape[0] != A.shape[1]:
        raise ValueError("Stability vector length mismatch.")
    stable = stability_corr >= min_abs_corr
    A2 = A[:, stable]
    B2 = B[:, stable]
    roi_sets2 = [roi_sets[i] for i in range(len(roi_sets)) if stable[i]]
    return A2, B2, roi_sets2, stable

# ...existing code...

# ...existing code (keep previous definitions) ...

def split_half_stability(M: Dict[str, Any],
                         chunk: str,
                         rank: int,
                         B: int = 50,
                         sample_frac: float = 0.7,
                         random_state: int = 0,
                         sign_invariant: bool = True) -> Dict[str, Any]:
    """
    Estimate component reproducibility (split-half absolute loading correlation).

    Procedure
    ---------
    1. Build bagged means tensor (B bags).
    2. Randomly partition bags into two halves.
    3. CP decompose each half at target rank.
    4. Match components between halves via Hungarian algorithm on
       |corr(column_i_half1, column_j_half2)| (sign_invariant=True).
    5. Record absolute matched correlations per component index.

    Parameters
    ----------
    rank : int
        CP rank to test (should match initial best-loss rank).
    B : int
        Total bags (will be split into ~B/2 & ~B/2).
    sample_frac : float
        Fraction of trials sampled per bag (with replacement).
    sign_invariant : bool
        If True use absolute correlations for matching (recommended).

    Returns
    -------
    dict:
      corr_matrix / abs_corr_matrix
      matched_corr / matched_abs_corr
      per_component_abs_corr  : length=rank, 0 if unmatched
      row_idx / col_idx       : Hungarian matches
      time                    : cleaned time vector from bag tensor

    Interpretation
    --------------
    per_component_abs_corr ~0.8–1.0 indicates stable motifs; <0.5 often unreliable.
    """
    tensor, t = build_bagged_trial_means(M, chunk, B=B,
                                         sample_frac=sample_frac,
                                         stratify_isi=(chunk=='isi'),
                                         random_state=random_state)
    tensor_clean, time_keep, bag_keep = clean_tensor_for_cp(tensor)
    X = np.nan_to_num(tensor_clean, nan=0.0)
    mask = ~np.isnan(tensor_clean)

    rng = np.random.default_rng(random_state)
    bag_idx = np.arange(X.shape[2])
    rng.shuffle(bag_idx)
    mid = bag_idx.size // 2
    b1, b2 = bag_idx[:mid], bag_idx[mid:]
    X1, M1 = X[:, :, b1], mask[:, :, b1]
    X2, M2 = X[:, :, b2], mask[:, :, b2]

    res1 = cp_decompose_masked(X1, M1, rank=rank, random_state=random_state)
    res2 = cp_decompose_masked(X2, M2, rank=rank, random_state=random_state+1)

    A1, A2 = res1['A'], res2['A']

    def col_corr(a, b):
        a0 = a - a.mean(axis=0, keepdims=True)
        b0 = b - b.mean(axis=0, keepdims=True)
        na = np.linalg.norm(a0, axis=0) + 1e-12
        nb = np.linalg.norm(b0, axis=0) + 1e-12
        return (a0.T @ b0) / (na[:, None] * nb[None, :])

    C = col_corr(A1, A2)
    match_matrix = np.abs(C) if sign_invariant else C

    from scipy.optimize import linear_sum_assignment
    r_ind, c_ind = linear_sum_assignment(-match_matrix)
    matched_corr = C[r_ind, c_ind]
    matched_abs = np.abs(matched_corr)

    # Sign alignment
    flip = matched_corr < 0
    if np.any(flip):
        for i, need_flip in enumerate(flip):
            if need_flip:
                A2[:, c_ind[i]] *= -1
        # Recompute C after flips (optional)
        C = col_corr(A1, A2)

    # Map per-component abs correlation (unmatched -> 0)
    per_comp_abs = np.zeros(rank, dtype=float)
    for i, comp_idx in enumerate(r_ind):
        per_comp_abs[comp_idx] = matched_abs[i]

    return dict(
        corr_matrix=C,
        abs_corr_matrix=np.abs(C),
        row_idx=r_ind,
        col_idx=c_ind,
        matched_corr=matched_corr,
        matched_abs_corr=matched_abs,
        per_component_abs_corr=per_comp_abs,
        rank=rank,
        time=t[time_keep]
    )

def run_cp_full_pipeline(M: Dict[str, Any],
                         chunk: str,
                         ranks: List[int] = [8,10,12],
                         B: int = 50,
                         sample_frac: float = 0.7,
                         stratify_isi: bool = True,
                         random_state: int = 0,
                         stability_B: int = 60,
                         min_abs_corr: float = 0.65,
                         pruning_jaccard: float = 0.75) -> Dict[str, Any]:
    """
    End‑to‑end CP analysis for a single chunk.

    Stages
    ------
    1. run_cp_with_adaptive_sets:
         * Bagged trial means tensor
         * CP rank sweep (candidate ranks)
         * Adaptive ROI subset extraction (data-driven threshold)
         * Basic pruning (temporal variance / ptp / Jaccard redundancy)
    2. split_half_stability on chosen rank (pre-prune orientation)
    3. Stability-based pruning (min_abs_corr)

    Parameters
    ----------
    ranks : list[int]       Candidate ranks (choose best by reconstruction loss)
    B : int                 # of bag samples for bagged trial means
    sample_frac : float     Fraction (approx) of trials per bag (with replacement)
    stratify_isi : bool     If True and chunk=='isi' strata sample across ISI terciles
    stability_B : int       Bag samples for split-half stability
    min_abs_corr : float    Threshold for absolute loading correlation
    pruning_jaccard : float Max Jaccard allowed during initial redundancy pruning

    Returns
    -------
    dict with keys (selected):
      final_A, final_B, final_roi_sets
      stability_vec         (abs correlations of kept pre-stability-prune comps)
      base, stability       (intermediate dictionaries)
      rank                  (best rank pre-pruning)
      params                (recorded parameter values)

    Diagnostics
    -----------
    * Examine base['temporal_stats'] for each component.
    * If stability_vec small for many comps: raise stability_B or adjust ranks.

    """
    # Step 1-3: CP + adaptive + prune
    base = run_cp_with_adaptive_sets(
        M, chunk, ranks=ranks, B=B,
        sample_frac=sample_frac,
        stratify_isi=stratify_isi,
        random_state=random_state
    )
    # Step 4: stability on full rank (before pruning indices lose meaning)
    rank_used = base['factors']['rank']
    stab = split_half_stability(M, chunk, rank=rank_used,
                                B=stability_B,
                                sample_frac=sample_frac,
                                random_state=random_state+123)

    # Map stability to pruned components
    kept_idx = np.where(base['keep_mask'])[0]
    stability_vec = stab['per_component_abs_corr'][kept_idx]

    # Step 5: stability-based pruning
    A_final, B_final, roi_sets_final, stable_mask = filter_components_by_stability(
        base['A_pruned'], base['B_pruned'], base['pruned_roi_sets'],
        stability_vec, base['keep_mask'][base['keep_mask']],
        min_abs_corr=min_abs_corr
    )

    return dict(
        chunk=chunk,
        rank=rank_used,
        base=base,
        stability=stab,
        kept_indices=kept_idx,
        stability_vec=stability_vec,
        final_A=A_final,
        final_B=B_final,
        final_roi_sets=roi_sets_final,
        final_stable_mask=stable_mask,
        params=dict(
            ranks_tested=ranks,
            B=B,
            stability_B=stability_B,
            sample_frac=sample_frac,
            min_abs_corr=min_abs_corr,
            pruning_jaccard=pruning_jaccard
        )
    )

# Example condensed usage (replace ad-hoc sequence):
# cp_full_isi = run_cp_full_pipeline(M, 'isi', ranks=[8,10,12], min_abs_corr=0.65)
# print("Final ISI components:", cp_full_isi['final_A'].shape[1])
# print("Stability (abs) for final:", np.round(cp_full_isi['stability_vec'],3))


def run_cp_with_adaptive_sets(M: Dict[str, Any],
                              chunk: str,
                              ranks: List[int] = [8,10,12],
                              **kwargs) -> Dict[str, Any]:
    """
    Wrapper: run CP pipeline, adaptive ROI set extraction, pruning, stats.
    """
    res = run_chunk_cp_pipeline(M, chunk, ranks=ranks, **kwargs)
    A = res['factors']['A']
    Bmat = res['factors']['B']
    stats_t = component_temporal_stats(Bmat, res['time'])
    # Adaptive ROI sets
    roi_sets = extract_roi_sets_adaptive(A, method='auto')
    # Prune
    A2, B2, roi_sets2, keep_mask = prune_components(
        A, Bmat, roi_sets,
        stats_t['var'], stats_t['ptp'],
        min_time_var=1e-4, min_ptp=0.01, max_jaccard=0.75
    )
    res.update(dict(
        adaptive_roi_sets=roi_sets,
        pruned_roi_sets=roi_sets2,
        keep_mask=keep_mask,
        temporal_stats=stats_t,
        A_pruned=A2,
        B_pruned=B2
    ))
    return res

























































# ...existing code...

def build_component_catalog(M: Dict[str, Any],
                            chunk_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Assemble a catalog across chunks.
    chunk_results: chunk_key -> cp_full_pipeline result (from run_cp_full_pipeline)
    Returns:
      catalog: list of dicts with fields:
        {'chunk','comp_idx','rois','n_rois'}
      incidence: (n_rois, n_components) binary matrix
      comp_labels: list ["chunk:idx", ...]
    """
    catalog = []
    comp_labels = []
    col_ptr = 0
    entries = []
    for chunk, res in chunk_results.items():
        for i, rois in enumerate(res['final_roi_sets']):
            catalog.append({
                'chunk': chunk,
                'comp_idx': i,
                'rois': rois,
                'n_rois': len(rois)
            })
            comp_labels.append(f"{chunk}:{i}")
            entries.append(rois)
    n_rois = M['n_rois']
    n_comp = len(entries)
    incidence = np.zeros((n_rois, n_comp), dtype=bool)
    for j, rois in enumerate(entries):
        incidence[rois, j] = True
    return dict(catalog=catalog,
                incidence=incidence,
                comp_labels=comp_labels)

def summarize_component_overlaps(catalog: Dict[str, Any],
                                 top_k: int = 10):
    inc = catalog['incidence']
    labels = catalog['comp_labels']
    n_comp = inc.shape[1]
    # Pairwise Jaccard
    jacc = np.zeros((n_comp, n_comp))
    for i in range(n_comp):
        a = inc[:, i]
        for j in range(i+1, n_comp):
            b = inc[:, j]
            inter = np.sum(a & b)
            union = np.sum(a | b)
            jacc[i, j] = jacc[j, i] = inter / union if union else 0.0
    # Report top overlaps
    tri = []
    for i in range(n_comp):
        for j in range(i+1, n_comp):
            tri.append((jacc[i, j], i, j))
    tri.sort(reverse=True)
    print("Top pairwise Jaccard overlaps:")
    for k, (val, i, j) in enumerate(tri[:top_k]):
        print(f"  {labels[i]} vs {labels[j]}: {val:.3f}")

def run_pipeline_for_chunks(M: Dict[str, Any],
                            chunks: List[str],
                            ranks: List[int] = [6,8,10,12],
                            min_abs_corr: float = 0.65,
                            B: int = 50,
                            stability_B: int = 60,
                            sample_frac: float = 0.7) -> Dict[str, Dict[str, Any]]:
    results = {}
    for ch in chunks:
        print(f"[run] {ch}")
        res = run_cp_full_pipeline(
            M, ch,
            ranks=ranks,
            B=B,
            stability_B=stability_B,
            sample_frac=sample_frac,
            stratify_isi=(ch=='isi'),
            min_abs_corr=min_abs_corr
        )
        print(f"  final components: {res['final_A'].shape[1]}")
        results[ch] = res
    return results

def plot_component_temporals(res: Dict[str, Any], title: str):
    """
    Quick visualization of normalized temporal factor waveforms for a chunk.

    Normalization
    -------------
    Each temporal factor B[:, i] scaled by max|B[:, i]| so that shapes are
    directly comparable (y-axis arbitrary).

    Parameters
    ----------
    res : dict
        CP result (expects res['final_B'] and res['base']['time']).
    title : str
        Figure title.

    Use
    ---
    Inspect diversity, latency structure, and possible redundancy at a glance.
    """    
    Bmat = res['final_B']  # (time, comps)
    t = res['base']['time']
    plt.figure(figsize=(5,3))
    for i in range(Bmat.shape[1]):
        w = Bmat[:, i]
        if np.max(np.abs(w)) > 0:
            w = w / (np.max(np.abs(w)) + 1e-12)
        plt.plot(t, w, label=f"{i}")
    plt.axvline(0, color='k', ls='--', lw=0.8)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Norm factor")
    if Bmat.shape[1] <= 12:
        plt.legend(fontsize=7, ncol=3)
    plt.tight_layout()
































# ...existing code...

# ...existing code...
def build_roi_role_table(M: Dict[str, Any],
                         catalog: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct ROI x component incidence and per-ROI role summary.
    Returns:
      incidence: (n_rois, n_components) bool
      labels: list of component labels
      roles_per_roi: list length n_rois; each a list of component label strings
      comp_indices_per_roi: list length n_rois; each a 1D np.ndarray of component indices
      n_roles: (# components each ROI belongs to)
      multi_role_mask: boolean mask of ROIs with >1 roles
    """
    inc = catalog['incidence']  # (n_rois, n_components)
    labels = catalog['comp_labels']
    n_rois = inc.shape[0]

    roles_per_roi: List[List[str]] = []
    comp_indices_per_roi: List[np.ndarray] = []
    for i in range(n_rois):
        comps = np.flatnonzero(inc[i])          # 1D array of indices
        comp_indices_per_roi.append(comps)
        roles_per_roi.append([labels[j] for j in comps])

    n_roles = inc.sum(axis=1)
    multi_role_mask = n_roles > 1
    return dict(
        incidence=inc,
        labels=labels,
        roles_per_roi=roles_per_roi,
        comp_indices_per_roi=comp_indices_per_roi,
        n_roles=n_roles,
        multi_role_mask=multi_role_mask
    )
# ...existing code...

# ...existing code...
def component_uniqueness_metrics(catalog: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute per-component overlap/uniqueness metrics from catalog incidence.

    Metrics
    -------
    size            : number of ROIs in component
    max_jacc_other  : max pairwise Jaccard with any OTHER component
    exclusivity     : (# ROIs unique to this component) / size
    jaccard_matrix  : full symmetric Jaccard (n_comp x n_comp)

    Interpretation
    --------------
    * Low exclusivity & high max_jacc_other -> broad / shared motif.
    * High exclusivity & low max_jacc_other -> specific / unique motif.

    Returns
    -------
    dict of arrays (aligned with catalog['comp_labels']).

    Downstream
    ----------
    Combine with build_roi_annotation_dataframe for per-ROI role tagging.
    """
    inc = catalog['incidence']
    labels = catalog['comp_labels']
    n_comp = inc.shape[1]
    sizes = inc.sum(axis=0)
    # Pairwise Jaccard
    jacc = np.zeros((n_comp, n_comp), float)
    for i in range(n_comp):
        a = inc[:, i]
        for j in range(i+1, n_comp):
            b = inc[:, j]
            inter = np.sum(a & b)
            union = np.sum(a | b)
            jac_val = inter / union if union else 0.0
            jacc[i, j] = jacc[j, i] = jac_val
    max_jacc_other = np.max(np.where(np.eye(n_comp, dtype=bool), 0.0, jacc), axis=1)
    # Exclusivity
    roi_coverage_counts = inc.sum(axis=1)
    exclusivity = []
    for c in range(n_comp):
        rois_c = inc[:, c]
        unique_c = np.sum((roi_coverage_counts == 1) & rois_c)
        exclusivity.append(unique_c / (sizes[c] + 1e-9))
    metrics = dict(
        labels=labels,
        size=sizes,
        max_jacc_other=max_jacc_other,
        exclusivity=np.array(exclusivity),
        jaccard_matrix=jacc
    )
    return metrics
# ...existing code...

def merge_high_overlap_components(catalog: Dict[str, Any],
                                  overlap_thresh: float = 0.6) -> Dict[str, Any]:
    """
    Merge components with Jaccard > overlap_thresh (greedy). Merged set is union of ROIs.
    Returns new catalog-like structure (merged).
    """
    inc = catalog['incidence']
    labels = catalog['comp_labels']
    n_comp = inc.shape[1]
    # Build Jaccard
    jacc = np.zeros((n_comp, n_comp))
    for i in range(n_comp):
        a = inc[:, i]
        for j in range(i+1, n_comp):
            b = inc[:, j]
            inter = np.sum(a & b)
            union = np.sum(a | b)
            jacc[i, j] = jacc[j, i] = inter / union if union else 0.0
    used = np.zeros(n_comp, dtype=bool)
    merged_sets = []
    merged_labels = []
    for i in range(n_comp):
        if used[i]:
            continue
        group = [i]
        overlaps = np.where(jacc[i] >= overlap_thresh)[0]
        for o in overlaps:
            if o != i and not used[o]:
                group.append(o)
        used[group] = True
        union_mask = np.any(inc[:, group], axis=1)
        merged_sets.append(np.where(union_mask)[0])
        if len(group) == 1:
            merged_labels.append(labels[i])
        else:
            merged_labels.append("+".join(labels[g] for g in group))
    # Build new incidence
    n_rois = inc.shape[0]
    new_inc = np.zeros((n_rois, len(merged_sets)), dtype=bool)
    for j, rois in enumerate(merged_sets):
        new_inc[rois, j] = True
    return dict(
        catalog=[{'label': merged_labels[j], 'rois': merged_sets[j], 'n_rois': len(merged_sets[j])}
                 for j in range(len(merged_sets))],
        incidence=new_inc,
        comp_labels=merged_labels,
        merged_from_original=labels,
        overlap_thresh=overlap_thresh
    )

def adaptive_pruning_thresholds(window_length: float,
                                base_min_ptp: float = 0.01,
                                base_min_var: float = 1e-4) -> Tuple[float, float]:
    """
    Scale pruning thresholds for short windows (fewer time points).
    For windows <0.4s reduce thresholds proportionally.
    """
    scale = 1.0
    if window_length < 0.4:
        scale = max(0.3, window_length / 0.4)  # clamp
    return base_min_ptp * scale, base_min_var * scale

# Example integration point (if you want to re-run start_flash_1 with relaxed thresholds):
def rerun_single_chunk_with_relaxed_pruning(M: Dict[str, Any],
                                            chunk: str,
                                            ranks: List[int] = [4,6,8],
                                            min_abs_corr: float = 0.55,
                                            window_override: Optional[Tuple[float,float]] = None) -> Dict[str, Any]:
    ch_info = M['chunks'][chunk]
    window_len = ch_info['window'][1] - ch_info['window'][0]
    min_ptp, min_var = adaptive_pruning_thresholds(window_len,
                                                   base_min_ptp=0.008,
                                                   base_min_var=5e-5)
    # Temporarily monkey-patch prune thresholds by wrapping run_cp_full_pipeline
    res = run_cp_full_pipeline(
        M, chunk,
        ranks=ranks,
        B=40,
        stability_B=50,
        sample_frac=0.7,
        stratify_isi=(chunk == 'isi'),
        min_abs_corr=min_abs_corr
    )
    # Post-filter with relaxed thresholds (override)
    A = res['base']['factors']['A']
    Bmat = res['base']['factors']['B']
    stats_t = component_temporal_stats(Bmat, res['base']['time'])
    roi_sets = extract_roi_sets_adaptive(A, method='auto')
    A2, B2, roi_sets2, keep_mask = prune_components(
        A, Bmat, roi_sets,
        stats_t['var'], stats_t['ptp'],
        min_time_var=min_var,
        min_ptp=min_ptp,
        max_jaccard=0.8
    )
    res['relaxed'] = dict(
        min_ptp=min_ptp,
        min_var=min_var,
        A_relaxed=A2,
        B_relaxed=B2,
        roi_sets_relaxed=roi_sets2,
        keep_mask_relaxed=keep_mask,
        stats=stats_t
    )
    return res

















































# ...existing code...

def build_roi_annotation_dataframe(M: Dict[str, Any],
                                   catalog: Dict[str, Any],
                                   uniqueness: Dict[str, Any]) -> pd.DataFrame:
    """
    Returns DataFrame with per-ROI annotations:
      roi_id, n_roles, roles (semicolon), is_multi_role, primary_role (largest component),
      broad_flag (if all roles are broad motifs)
    """
    inc = catalog['incidence']          # (n_rois, n_comp)
    labels = catalog['comp_labels']
    size = uniqueness['size']
    maxJ = uniqueness['max_jacc_other']
    excl = uniqueness['exclusivity']

    # Mark broad motifs (low exclusivity & high overlap)
    broad_mask = (excl < 0.05) & (maxJ > 0.55)
    label_to_broad = {labels[i]: bool(broad_mask[i]) for i in range(len(labels))}

    rows = []
    for roi in range(M['n_rois']):
        comps = np.flatnonzero(inc[roi])
        role_labels = [labels[c] for c in comps]
        n_roles = len(comps)
        is_multi = n_roles > 1
        # Choose primary: component with largest size (alternatively exclusivity)
        if n_roles:
            primary_idx = comps[np.argmax(size[comps])]
            primary_role = labels[primary_idx]
        else:
            primary_role = ""
        broad_flag = all(label_to_broad[l] for l in role_labels) if role_labels else False
        rows.append(dict(
            roi_id=roi,
            n_roles=n_roles,
            roles=";".join(role_labels),
            is_multi_role=is_multi,
            primary_role=primary_role,
            broad_flag=broad_flag
        ))
    return pd.DataFrame(rows)

def export_roi_annotations(df_roles: pd.DataFrame,
                           uniqueness: Dict[str, Any],
                           out_csv: str) -> None:
    """
    Writes ROI roles + component uniqueness summary to disk.
    """
    df_roles.to_csv(out_csv, index=False)
    # Also save component-level table
    comp_df = pd.DataFrame(dict(
        label=uniqueness['labels'],
        size=uniqueness['size'],
        max_jacc_other=uniqueness['max_jacc_other'],
        exclusivity=uniqueness['exclusivity']
    ))
    comp_df.to_csv(out_csv.replace('.csv', '_components.csv'), index=False)
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_csv.replace('.csv', '_components.csv')}")

# Convenience to filter catalog to canonical F2 representation
def filter_catalog_by_label_prefix(catalog: Dict[str, Any],
                                   keep_prefixes: List[str]) -> Dict[str, Any]:
    inc = catalog['incidence']
    labels = catalog['comp_labels']
    keep_cols = [i for i,l in enumerate(labels) if any(l.startswith(p) for p in keep_prefixes)]
    new_inc = inc[:, keep_cols]
    new_labels = [labels[i] for i in keep_cols]
    return dict(catalog=[], incidence=new_inc, comp_labels=new_labels)

# ...existing code...



































































# ...existing code...

def _robust_sym_vlim(mat: np.ndarray, quant: float = 0.98) -> Tuple[float, float]:
    vals = mat[np.isfinite(mat)]
    if vals.size == 0:
        return -1, 1
    lim = np.quantile(np.abs(vals), quant)
    if lim <= 0:
        lim = np.max(np.abs(vals)) if vals.size else 1.0
    if lim == 0:
        lim = 1.0
    return -lim, lim

def compute_group_session_arrays(M: Dict[str, Any],
                                 roi_idx: np.ndarray,
                                 smoothing_sigma: Optional[float] = None) -> Dict[str, Any]:
    """
    From trial_start chunk: produce
      roi_mean: (n_group_rois, T) mean across trials for each ROI
      group_trial_means: (trials, T) mean across ROIs for each trial
    """
    ch = M['chunks']['trial_start']
    X = ch['data']  # (trials, rois, T)
    t = ch['time']
    if roi_idx.size == 0:
        return dict(time=t, roi_mean=np.empty((0, t.size)), group_trial_means=np.empty((M['n_trials'], t.size)))
    sub = X[:, roi_idx, :]  # (trials, nR, T)
    # Mean across trials per ROI
    roi_mean = np.nanmean(sub, axis=0)  # (nR, T)
    # Mean across ROIs per trial
    group_trial_means = np.nanmean(sub, axis=1)  # (trials, T)
    if smoothing_sigma and smoothing_sigma > 0:
        roi_mean = gaussian_filter1d(roi_mean, axis=1, sigma=smoothing_sigma)
        group_trial_means = gaussian_filter1d(group_trial_means, axis=1, sigma=smoothing_sigma)
    return dict(time=t, roi_mean=roi_mean, group_trial_means=group_trial_means)

def plot_group_overview(M: Dict[str, Any],
                        roi_idx: np.ndarray,
                        label: str,
                        condition_split: bool = True,
                        smoothing_sigma: Optional[float] = 0.8,
                        events: Optional[List[str]] = None,
                        event_colors: Optional[Dict[str, str]] = None,
                        figsize: Tuple[float, float] = (7.5, 4.8)) -> None:
    """
    One figure: (A) ROI x time raster (mean across trials per ROI) (B) mean traces with short/long split.

    events: list of event field names in M to plot (default standard set).
    """
    if events is None:
        events = ['F1_on', 'F1_off', 'F2_on', 'F2_off', 'choice_start', 'lick_start']
    if event_colors is None:
        event_colors = {
            'F1_on': '#1f77b4',
            'F1_off': '#1f77b4',
            'F2_on': '#ff7f0e',
            'F2_off': '#ff7f0e',
            'choice_start': '#2ca02c',
            'lick_start': '#d62728'
        }

    data = compute_group_session_arrays(M, roi_idx, smoothing_sigma=smoothing_sigma)
    t = data['time']
    roi_mean = data['roi_mean']
    group_trial_means = data['group_trial_means']  # (trials, T)
    nR = roi_mean.shape[0]

    # Split conditions
    is_short = M['is_short']
    short_traces = group_trial_means[is_short] if condition_split else None
    long_traces = group_trial_means[~is_short] if condition_split else None

    # Aggregate means & SEM
    mean_all = np.nanmean(group_trial_means, axis=0) if group_trial_means.size else np.zeros_like(t)
    sem_all = np.nanstd(group_trial_means, axis=0, ddof=1) / np.sqrt(max(1, np.sum(np.isfinite(group_trial_means[:, 0]))))

    if condition_split and short_traces.size:
        mean_short = np.nanmean(short_traces, axis=0)
        sem_short = np.nanstd(short_traces, axis=0, ddof=1) / np.sqrt(max(1, short_traces.shape[0]))
        mean_long = np.nanmean(long_traces, axis=0)
        sem_long = np.nanstd(long_traces, axis=0, ddof=1) / np.sqrt(max(1, long_traces.shape[0]))
    else:
        mean_short = mean_long = sem_short = sem_long = None

    # Event stats
    event_stats = {}
    for ev in events:
        if ev in M and np.any(np.isfinite(M[ev])):
            arr = M[ev]
            mu = float(np.nanmean(arr))
            sd = float(np.nanstd(arr))
            event_stats[ev] = (mu, sd)

    vmin, vmax = _robust_sym_vlim(roi_mean)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[3.2, 1.4], hspace=0.25, figure=fig)

    # Panel A: raster ROI x time
    ax_r = fig.add_subplot(gs[0])
    if nR:
        im = ax_r.imshow(roi_mean, aspect='auto', origin='lower',
                         extent=(t[0], t[-1], 0, nR),
                         cmap='coolwarm', vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax_r, fraction=0.02, pad=0.01)
        cbar.set_label('Mean z-dF/F', fontsize=8)
    ax_r.set_ylabel(f'ROIs (n={nR})', fontsize=9)
    ax_r.set_title(f'Group {label} | ROI mean across trials', fontsize=10)

    # Event markers on raster
    for ev, (mu, sd) in event_stats.items():
        col = event_colors.get(ev, 'k')
        ax_r.axvline(mu, color=col, ls='-', lw=1.0, alpha=0.9)
        if sd > 0:
            ax_r.axvspan(mu - sd, mu + sd, color=col, alpha=0.08)

    # Panel B: mean traces
    ax_t = fig.add_subplot(gs[1], sharex=ax_r)
    ax_t.plot(t, mean_all, color='k', lw=1.5, label='All')
    ax_t.fill_between(t, mean_all - sem_all, mean_all + sem_all, color='k', alpha=0.15, linewidth=0)

    if condition_split and mean_short is not None:
        ax_t.plot(t, mean_short, color='#1f77b4', lw=1.2, label='Short')
        ax_t.fill_between(t, mean_short - sem_short, mean_short + sem_short, color='#1f77b4', alpha=0.15, linewidth=0)
        ax_t.plot(t, mean_long, color='#ff7f0e', lw=1.2, label='Long')
        ax_t.fill_between(t, mean_long - sem_long, mean_long + sem_long, color='#ff7f0e', alpha=0.15, linewidth=0)

    # Event lines again for clarity
    for ev, (mu, sd) in event_stats.items():
        col = event_colors.get(ev, 'k')
        ax_t.axvline(mu, color=col, ls='--', lw=0.9)
        if sd > 0:
            ax_t.axvspan(mu - sd, mu + sd, color=col, alpha=0.05)

    ax_t.set_xlabel('Time (s)', fontsize=9)
    ax_t.set_ylabel('Group mean', fontsize=9)
    ax_t.legend(fontsize=7, ncol=3, frameon=False)
    ax_t.grid(alpha=0.2, linewidth=0.5)
    plt.tight_layout()

def plot_all_groups(M: Dict[str, Any],
                    catalog: Dict[str, Any],
                    smoothing_sigma: float = 0.8,
                    condition_split: bool = True,
                    groups: Optional[List[str]] = None) -> None:
    """
    Iterate through catalog labels (or provided subset) and plot each group figure.
    """
    labels = catalog['comp_labels']
    inc = catalog['incidence']  # (n_rois, n_comp)
    if groups is None:
        groups = labels
    label_to_col = {lbl: j for j, lbl in enumerate(labels)}
    for lbl in groups:
        if lbl not in label_to_col:
            print(f"[warn] group '{lbl}' not found in catalog; skipping.")
            continue
        j = label_to_col[lbl]
        roi_idx = np.flatnonzero(inc[:, j])
        plot_group_overview(M, roi_idx, lbl,
                            condition_split=condition_split,
                            smoothing_sigma=smoothing_sigma)

# ...existing code...

















































# ...existing code...

def _compute_roi_onsets(t: np.ndarray,
                        roi_mean_all: np.ndarray,
                        onset_window: Tuple[float, float],
                        z_thresh: float = 1.0) -> np.ndarray:
    """
    roi_mean_all: (nR, T) mean trace per ROI (all trials collapsed).
    Returns onset time (float) per ROI; np.inf if no onset.
    Onset = first time inside onset_window where trace > baseline_mean + z_thresh*baseline_std
    Baseline = t < onset_window[0].
    """
    nR, T = roi_mean_all.shape
    onset_times = np.full(nR, np.inf, float)
    mask_baseline = t < onset_window[0]
    mask_win = (t >= onset_window[0]) & (t <= onset_window[1])
    if not mask_win.any():
        return onset_times
    # Precompute baseline stats
    if mask_baseline.any():
        base_mean = np.nanmean(roi_mean_all[:, mask_baseline], axis=1)
        base_std = np.nanstd(roi_mean_all[:, mask_baseline], axis=1) + 1e-9
    else:
        base_mean = np.nanmean(roi_mean_all, axis=1)
        base_std = np.nanstd(roi_mean_all, axis=1) + 1e-9
    thr = base_mean + z_thresh * base_std
    win_idx = np.where(mask_win)[0]
    for r in range(nR):
        y = roi_mean_all[r, win_idx]
        hits = np.where(y > thr[r])[0]
        if hits.size:
            onset_times[r] = t[win_idx[hits[0]]]
    return onset_times

def _event_stats_condition(times: np.ndarray, cond_mask: np.ndarray) -> Tuple[float, float]:
    sel = times[cond_mask]
    sel = sel[np.isfinite(sel)]
    if sel.size == 0:
        return np.nan, np.nan
    return float(np.nanmean(sel)), float(np.nanstd(sel))



def plot_group_overview_v2(M: Dict[str, Any],
                           roi_idx: np.ndarray,
                           label: str,
                           cfg: Optional[Dict[str, Any]] = None):
    """
    (Patched)
    Additions:
      cfg['trace_ylim']          : fixed y-range for traces
      cfg['show_onset_markers']  : bool (default False) scatter detected onsets on rasters
      cfg['onset_marker_kwargs'] : dict passed to scatter (defaults set below)
      cfg['zero_line']           : bool (default True) draw horizontal dashed 0 line on all trace panels
      cfg['zero_line_kwargs']    : dict (e.g., {'color':'k','lw':0.6,'ls':'--'})
    """
    if cfg is None:
        cfg = {}
    events = cfg.get('events', ['F1_on','F1_off','F2_on','F2_off','choice_start','lick_start'])
    event_colors = cfg.get('event_colors', {
        'F1_on':'#1f77b4','F1_off':'#1f77b4',
        'F2_on':'#ff7f0e','F2_off':'#ff7f0e',
        'choice_start':'#2ca02c','lick_start':'#d62728'
    })
    smoothing_sigma = cfg.get('smoothing_sigma', 0.8)
    sort_mode = cfg.get('sort_mode', 'index')
    onset_window = cfg.get('onset_window', (0.0, 3.0))
    onset_z = cfg.get('onset_z', 1.0)
    show_onsets = cfg.get('show_onset_markers', False)
    onset_marker_kwargs = cfg.get('onset_marker_kwargs', dict(marker='o', s=6, color='k', linewidths=0))
    shared_vlim = cfg.get('vlim', None)
    trace_ylim = cfg.get('trace_ylim', None)
    zero_line = cfg.get('zero_line', True)
    zero_line_kwargs = cfg.get('zero_line_kwargs', dict(color='k', lw=0.6, ls='--', alpha=0.9))
    fig_size = cfg.get('figsize', (7.2, 8.0))

    ch = M['chunks']['trial_start']
    X = ch['data']
    t = ch['time']
    if roi_idx.size == 0:
        print(f"[plot_group_overview_v2] Empty ROI set for {label}")
        return
    sub = X[:, roi_idx, :]
    if smoothing_sigma and smoothing_sigma > 0:
        sub = gaussian_filter1d(sub, axis=2, sigma=smoothing_sigma)
    is_short = M['is_short']
    short_sub = sub[is_short]
    long_sub  = sub[~is_short]

    roi_mean_short = np.nanmean(short_sub, axis=0) if short_sub.size else np.full((len(roi_idx), t.size), np.nan)
    roi_mean_long  = np.nanmean(long_sub,  axis=0) if long_sub.size  else np.full((len(roi_idx), t.size), np.nan)
    roi_mean_all   = np.nanmean(sub, axis=0)

    # --- Onset detection (always compute if markers requested or sorting by onset) ---
    onsets = None
    if sort_mode == 'onset' or show_onsets:
        onsets = _compute_roi_onsets(t, roi_mean_all, onset_window, z_thresh=onset_z)

    order = np.arange(len(roi_idx))
    if sort_mode == 'onset' and onsets is not None:
        order = np.argsort(onsets)

    roi_mean_short = roi_mean_short[order]
    roi_mean_long  = roi_mean_long[order]
    if onsets is not None:
        onsets_ordered = onsets[order]
    else:
        onsets_ordered = None

    group_trial_means = np.nanmean(sub, axis=1)
    short_trial_means = group_trial_means[is_short]
    long_trial_means  = group_trial_means[~is_short]

    mean_all  = np.nanmean(group_trial_means, axis=0)
    sem_all   = np.nanstd(group_trial_means, axis=0, ddof=1) / max(1, np.sqrt(group_trial_means.shape[0]))
    mean_short = np.nanmean(short_trial_means, axis=0)
    sem_short  = np.nanstd(short_trial_means, axis=0, ddof=1) / max(1, np.sqrt(short_trial_means.shape[0]))
    mean_long  = np.nanmean(long_trial_means, axis=0)
    sem_long   = np.nanstd(long_trial_means, axis=0, ddof=1) / max(1, np.sqrt(long_trial_means.shape[0]))
    diff_trace = mean_long - mean_short

    event_stats_short = {}
    event_stats_long  = {}
    for ev in events:
        if ev in M:
            mu_s, sd_s = _event_stats_condition(M[ev], is_short)
            mu_l, sd_l = _event_stats_condition(M[ev], ~is_short)
            event_stats_short[ev] = (mu_s, sd_s)
            event_stats_long[ev]  = (mu_l, sd_l)

    if shared_vlim is None:
        combined = np.vstack([roi_mean_short, roi_mean_long])
        vmin, vmax = _robust_sym_vlim(combined, quant=0.98)
    else:
        vmin, vmax = shared_vlim

    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(5, 1, height_ratios=[1.6,1.6,1.0,0.9,0.9], hspace=0.35, figure=fig)

    def _plot_raster(ax, mat, title, ev_stats):
        im = ax.imshow(mat, aspect='auto', origin='lower',
                       extent=(t[0], t[-1], 0, mat.shape[0]),
                       cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.set_ylabel(f'ROIs\n(n={mat.shape[0]})', fontsize=8)
        ax.set_title(title, fontsize=9)
        for ev,(mu,sd) in ev_stats.items():
            if not np.isfinite(mu): continue
            col = event_colors.get(ev,'k')
            ax.axvline(mu, color=col, lw=0.9)
            if np.isfinite(sd) and sd>0:
                ax.axvspan(mu-sd, mu+sd, color=col, alpha=0.07)
        # onset markers
        if show_onsets and onsets_ordered is not None:
            valid = np.isfinite(onsets_ordered)
            if valid.any():
                # center markers vertically in each row
                y_idx = np.arange(onsets_ordered.size)[valid] + 0.5
                ax.scatter(onsets_ordered[valid], y_idx,
                           **onset_marker_kwargs)
        return im

    ax1 = fig.add_subplot(gs[0])
    _plot_raster(ax1, roi_mean_short, f'{label} | Short trials', event_stats_short)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    im = _plot_raster(ax2, roi_mean_long, f'{label} | Long trials', event_stats_long)
    cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.015, pad=0.01)
    cbar.set_label('Mean z-dF/F', fontsize=8)

    def _plot_events(ax, ev_stats_short, ev_stats_long):
        for ev in events:
            mu_s, sd_s = ev_stats_short.get(ev,(np.nan,np.nan))
            mu_l, sd_l = ev_stats_long.get(ev,(np.nan,np.nan))
            col = event_colors.get(ev,'k')
            if np.isfinite(mu_s):
                ax.axvline(mu_s, color=col, lw=0.9)
                if np.isfinite(sd_s) and sd_s>0:
                    ax.axvspan(mu_s-sd_s, mu_s+sd_s, color=col, alpha=0.05)
            if np.isfinite(mu_l):
                ax.axvline(mu_l, color=col, lw=0.9, ls='--')
                if np.isfinite(sd_l) and sd_l>0:
                    ax.axvspan(mu_l-sd_l, mu_l+sd_l, color=col, alpha=0.04)

    # Panel 3
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t, mean_all, color='k', lw=1.4, label='All')
    ax3.fill_between(t, mean_all - sem_all, mean_all + sem_all, color='k', alpha=0.15, linewidth=0)
    ax3.plot(t, mean_short, color='#1f77b4', lw=1.1, label='Short')
    ax3.plot(t, mean_long, color='#ff7f0e', lw=1.1, label='Long')
    ax3.plot(t, diff_trace, color='0.4', lw=1.0, ls='--', label='Long-Short')
    _plot_events(ax3, event_stats_short, event_stats_long)
    if zero_line: ax3.axhline(0, **zero_line_kwargs)
    ax3.set_ylabel('Mean', fontsize=8)
    ax3.legend(fontsize=7, ncol=4, frameon=False)
    ax3.grid(alpha=0.25, linewidth=0.5)
    if trace_ylim: ax3.set_ylim(trace_ylim)

    # Panel 4
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(t, mean_short, color='#1f77b4', lw=1.3)
    ax4.fill_between(t, mean_short - sem_short, mean_short + sem_short,
                     color='#1f77b4', alpha=0.18, linewidth=0)
    for ev,(mu,sd) in event_stats_short.items():
        if not np.isfinite(mu): continue
        col = event_colors.get(ev,'k')
        ax4.axvline(mu, color=col, lw=0.9)
        if np.isfinite(sd) and sd>0:
            ax4.axvspan(mu-sd, mu+sd, color=col, alpha=0.07)
    if zero_line: ax4.axhline(0, **zero_line_kwargs)
    ax4.set_ylabel('Short', fontsize=8)
    ax4.grid(alpha=0.2, linewidth=0.4)
    if trace_ylim: ax4.set_ylim(trace_ylim)

    # Panel 5
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.plot(t, mean_long, color='#ff7f0e', lw=1.3)
    ax5.fill_between(t, mean_long - sem_long, mean_long + sem_long,
                     color='#ff7f0e', alpha=0.18, linewidth=0)
    for ev,(mu,sd) in event_stats_long.items():
        if not np.isfinite(mu): continue
        col = event_colors.get(ev,'k')
        ax5.axvline(mu, color=col, lw=0.9)
        if np.isfinite(sd) and sd>0:
            ax5.axvspan(mu-sd, mu+sd, color=col, alpha=0.07)
    if zero_line: ax5.axhline(0, **zero_line_kwargs)
    ax5.set_ylabel('Long', fontsize=8)
    ax5.set_xlabel('Time (s)', fontsize=9)
    ax5.grid(alpha=0.2, linewidth=0.4)
    if trace_ylim: ax5.set_ylim(trace_ylim)

    if sort_mode == 'onset':
        ax1.text(0.99, 0.98, 'sorted by onset', transform=ax1.transAxes,
                 ha='right', va='top', fontsize=7, color='k')
    plt.tight_layout()



# --- PATCH: shared per-chunk trace y-limits for plot_all_groups_v2 -----------------------
# def plot_group_overview_v2(M: Dict[str, Any],
#                            roi_idx: np.ndarray,
#                            label: str,
#                            cfg: Optional[Dict[str, Any]] = None):
#     """
#     (Patched) Added support for cfg['trace_ylim'] to force identical y-range across
#     all groups within a chunk when invoked via plot_all_groups_v2.
#     """
#     if cfg is None:
#         cfg = {}
#     events = cfg.get('events', ['F1_on','F1_off','F2_on','F2_off','choice_start','lick_start'])
#     event_colors = cfg.get('event_colors', {
#         'F1_on':'#1f77b4','F1_off':'#1f77b4',
#         'F2_on':'#ff7f0e','F2_off':'#ff7f0e',
#         'choice_start':'#2ca02c','lick_start':'#d62728'
#     })
#     smoothing_sigma = cfg.get('smoothing_sigma', 0.8)
#     sort_mode = cfg.get('sort_mode', 'index')
#     onset_window = cfg.get('onset_window', (0.0, 3.0))
#     onset_z = cfg.get('onset_z', 1.0)
#     shared_vlim = cfg.get('vlim', None)
#     trace_ylim = cfg.get('trace_ylim', None)   # NEW
#     fig_size = cfg.get('figsize', (7.2, 8.0))

#     ch = M['chunks']['trial_start']
#     X = ch['data']
#     t = ch['time']
#     if roi_idx.size == 0:
#         print(f"[plot_group_overview_v2] Empty ROI set for {label}")
#         return
#     sub = X[:, roi_idx, :]
#     if smoothing_sigma and smoothing_sigma > 0:
#         sub = gaussian_filter1d(sub, axis=2, sigma=smoothing_sigma)
#     is_short = M['is_short']
#     short_sub = sub[is_short]
#     long_sub  = sub[~is_short]

#     roi_mean_short = np.nanmean(short_sub, axis=0) if short_sub.size else np.full((len(roi_idx), t.size), np.nan)
#     roi_mean_long  = np.nanmean(long_sub,  axis=0) if long_sub.size  else np.full((len(roi_idx), t.size), np.nan)
#     roi_mean_all   = np.nanmean(sub, axis=0)

#     order = np.arange(len(roi_idx))
#     if sort_mode == 'onset':
#         onsets = _compute_roi_onsets(t, roi_mean_all, onset_window, z_thresh=onset_z)
#         order = np.argsort(onsets)
#     roi_mean_short = roi_mean_short[order]
#     roi_mean_long  = roi_mean_long[order]

#     group_trial_means = np.nanmean(sub, axis=1)
#     short_trial_means = group_trial_means[is_short]
#     long_trial_means  = group_trial_means[~is_short]

#     mean_all  = np.nanmean(group_trial_means, axis=0)
#     sem_all   = np.nanstd(group_trial_means, axis=0, ddof=1) / max(1, np.sqrt(group_trial_means.shape[0]))
#     mean_short = np.nanmean(short_trial_means, axis=0)
#     sem_short  = np.nanstd(short_trial_means, axis=0, ddof=1) / max(1, np.sqrt(short_trial_means.shape[0]))
#     mean_long  = np.nanmean(long_trial_means, axis=0)
#     sem_long   = np.nanstd(long_trial_means, axis=0, ddof=1) / max(1, np.sqrt(long_trial_means.shape[0]))
#     diff_trace = mean_long - mean_short

#     event_stats_short = {}
#     event_stats_long  = {}
#     for ev in events:
#         if ev in M:
#             mu_s, sd_s = _event_stats_condition(M[ev], is_short)
#             mu_l, sd_l = _event_stats_condition(M[ev], ~is_short)
#             event_stats_short[ev] = (mu_s, sd_s)
#             event_stats_long[ev]  = (mu_l, sd_l)

#     if shared_vlim is None:
#         combined = np.vstack([roi_mean_short, roi_mean_long])
#         vmin, vmax = _robust_sym_vlim(combined, quant=0.98)
#     else:
#         vmin, vmax = shared_vlim

#     fig = plt.figure(figsize=fig_size)
#     gs = GridSpec(5, 1, height_ratios=[1.6,1.6,1.0,0.9,0.9], hspace=0.35, figure=fig)

#     def _plot_raster(ax, mat, title, ev_stats):
#         im = ax.imshow(mat, aspect='auto', origin='lower',
#                        extent=(t[0], t[-1], 0, mat.shape[0]),
#                        cmap='coolwarm', vmin=vmin, vmax=vmax)
#         ax.set_ylabel(f'ROIs\n(n={mat.shape[0]})', fontsize=8)
#         ax.set_title(title, fontsize=9)
#         for ev,(mu,sd) in ev_stats.items():
#             if not np.isfinite(mu): continue
#             col = event_colors.get(ev,'k')
#             ax.axvline(mu, color=col, lw=0.9)
#             if np.isfinite(sd) and sd>0:
#                 ax.axvspan(mu-sd, mu+sd, color=col, alpha=0.07)
#         return im

#     ax1 = fig.add_subplot(gs[0])
#     _plot_raster(ax1, roi_mean_short, f'{label} | Short trials', event_stats_short)
#     ax2 = fig.add_subplot(gs[1], sharex=ax1)
#     im = _plot_raster(ax2, roi_mean_long, f'{label} | Long trials', event_stats_long)
#     cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.015, pad=0.01)
#     cbar.set_label('Mean z-dF/F', fontsize=8)

#     def _plot_events(ax, ev_stats_short, ev_stats_long):
#         for ev in events:
#             mu_s, sd_s = ev_stats_short.get(ev,(np.nan,np.nan))
#             mu_l, sd_l = ev_stats_long.get(ev,(np.nan,np.nan))
#             col = event_colors.get(ev,'k')
#             if np.isfinite(mu_s):
#                 ax.axvline(mu_s, color=col, lw=0.9)
#                 if np.isfinite(sd_s) and sd_s>0:
#                     ax.axvspan(mu_s-sd_s, mu_s+sd_s, color=col, alpha=0.05)
#             if np.isfinite(mu_l):
#                 ax.axvline(mu_l, color=col, lw=0.9, ls='--')
#                 if np.isfinite(sd_l) and sd_l>0:
#                     ax.axvspan(mu_l-sd_l, mu_l+sd_l, color=col, alpha=0.04)

#     # Panel 3
#     ax3 = fig.add_subplot(gs[2], sharex=ax1)
#     ax3.plot(t, mean_all, color='k', lw=1.4, label='All')
#     ax3.fill_between(t, mean_all - sem_all, mean_all + sem_all, color='k', alpha=0.15, linewidth=0)
#     ax3.plot(t, mean_short, color='#1f77b4', lw=1.1, label='Short')
#     ax3.plot(t, mean_long, color='#ff7f0e', lw=1.1, label='Long')
#     ax3.plot(t, diff_trace, color='0.4', lw=1.0, ls='--', label='Long-Short')
#     _plot_events(ax3, event_stats_short, event_stats_long)
#     ax3.set_ylabel('Mean', fontsize=8)
#     ax3.legend(fontsize=7, ncol=4, frameon=False)
#     ax3.grid(alpha=0.25, linewidth=0.5)
#     if trace_ylim: ax3.set_ylim(trace_ylim)

#     # Panel 4
#     ax4 = fig.add_subplot(gs[3], sharex=ax1)
#     ax4.plot(t, mean_short, color='#1f77b4', lw=1.3)
#     ax4.fill_between(t, mean_short - sem_short, mean_short + sem_short,
#                      color='#1f77b4', alpha=0.18, linewidth=0)
#     for ev,(mu,sd) in event_stats_short.items():
#         if not np.isfinite(mu): continue
#         col = event_colors.get(ev,'k')
#         ax4.axvline(mu, color=col, lw=0.9)
#         if np.isfinite(sd) and sd>0:
#             ax4.axvspan(mu-sd, mu+sd, color=col, alpha=0.07)
#     ax4.set_ylabel('Short', fontsize=8)
#     ax4.grid(alpha=0.2, linewidth=0.4)
#     if trace_ylim: ax4.set_ylim(trace_ylim)

#     # Panel 5
#     ax5 = fig.add_subplot(gs[4], sharex=ax1)
#     ax5.plot(t, mean_long, color='#ff7f0e', lw=1.3)
#     ax5.fill_between(t, mean_long - sem_long, mean_long + sem_long,
#                      color='#ff7f0e', alpha=0.18, linewidth=0)
#     for ev,(mu,sd) in event_stats_long.items():
#         if not np.isfinite(mu): continue
#         col = event_colors.get(ev,'k')
#         ax5.axvline(mu, color=col, lw=0.9)
#         if np.isfinite(sd) and sd>0:
#             ax5.axvspan(mu-sd, mu+sd, color=col, alpha=0.07)
#     ax5.set_ylabel('Long', fontsize=8)
#     ax5.set_xlabel('Time (s)', fontsize=9)
#     ax5.grid(alpha=0.2, linewidth=0.4)
#     if trace_ylim: ax5.set_ylim(trace_ylim)

#     if sort_mode == 'onset':
#         ax1.text(0.99, 0.98, 'sorted by onset', transform=ax1.transAxes,
#                  ha='right', va='top', fontsize=7, color='k')
#     plt.tight_layout()


def plot_all_groups_v2(M: Dict[str, Any],
                       catalog: Dict[str, Any],
                       groups: Optional[List[str]] = None,
                       cfg: Optional[Dict[str, Any]] = None,
                       compute_global_vlim: bool = True):
    """
    (Patched) Adds shared trace y-limits per chunk:
      cfg keys:
        trace_shared (bool, default True)
        trace_quantile (float, default 1.0) -> robust amplitude limit
        trace_sym (bool, default True) -> symmetric about 0
        trace_pad_frac (float, default 0.05) -> padding outside limits
        trace_ylim (tuple|None) -> manual override (skip auto)
    """
    if cfg is None:
        cfg = {}
    labels = catalog['comp_labels']
    inc = catalog['incidence']
    if groups is None:
        groups = labels

    # Precompute global raster color scale if requested
    if compute_global_vlim and 'vlim' not in cfg:
        mats = []
        ch = M['chunks']['trial_start']['data']
        is_short = M['is_short']
        for lbl in groups:
            if lbl not in labels:
                continue
            j = labels.index(lbl)
            roi_idx = np.flatnonzero(inc[:, j])
            if roi_idx.size == 0:
                continue
            sub = ch[:, roi_idx, :]
            short_sub = sub[is_short]
            long_sub  = sub[~is_short]
            if short_sub.size:
                mats.append(np.nanmean(short_sub, axis=0))
            if long_sub.size:
                mats.append(np.nanmean(long_sub, axis=0))
        if mats:
            all_mat = np.vstack(mats)
            vmin, vmax = _robust_sym_vlim(all_mat)
            cfg['vlim'] = (vmin, vmax)

    # Shared trace y-limits (per chunk) unless overridden
    if cfg.get('trace_shared', True) and 'trace_ylim' not in cfg:
        trace_vals = []
        ch_data = M['chunks']['trial_start']['data']
        tmask_short = M['is_short']
        tmask_long = ~M['is_short']
        q = float(cfg.get('trace_quantile', 1.0))
        for lbl in groups:
            if lbl not in labels:
                continue
            j = labels.index(lbl)
            roi_idx = np.flatnonzero(inc[:, j])
            if roi_idx.size == 0:
                continue
            # mean across ROIs per trial
            per_trial = np.nanmean(ch_data[:, roi_idx, :], axis=1)
            # all / short / long means
            trace_vals.append(np.nanmean(per_trial, axis=0))
            if tmask_short.any():
                trace_vals.append(np.nanmean(per_trial[tmask_short], axis=0))
            if tmask_long.any():
                trace_vals.append(np.nanmean(per_trial[tmask_long], axis=0))
        if trace_vals:
            concat = np.hstack(trace_vals)
            finite = concat[np.isfinite(concat)]
            if finite.size:
                if q < 1.0:
                    lo = np.quantile(finite, 1 - q)
                    hi = np.quantile(finite, q)
                else:
                    lo = finite.min(); hi = finite.max()
                pad = (hi - lo) * cfg.get('trace_pad_frac', 0.05)
                if cfg.get('trace_sym', True):
                    m = max(abs(lo), abs(hi))
                    cfg['trace_ylim'] = (-m - pad, m + pad)
                else:
                    cfg['trace_ylim'] = (lo - pad, hi + pad)

    for lbl in groups:
        if lbl not in labels:
            print(f"[warn] group '{lbl}' not in catalog; skip")
            continue
        j = labels.index(lbl)
        roi_idx = np.flatnonzero(inc[:, j])
        plot_group_overview_v2(M, roi_idx, lbl, cfg=cfg)





# def plot_group_overview_v2(M: Dict[str, Any],
#                            roi_idx: np.ndarray,
#                            label: str,
#                            cfg: Optional[Dict[str, Any]] = None):
#     """
#     Advanced per-group visualization with 5 panels:
#       1. Short ROI raster (mean across short trials per ROI)
#       2. Long  ROI raster
#       3. Combined mean traces (All, Short, Long, Difference Long-Short)
#       4. Short condition mean (with its events)
#       5. Long  condition mean (with its events)
#     """
#     if cfg is None:
#         cfg = {}
#     events = cfg.get('events', ['F1_on','F1_off','F2_on','F2_off','choice_start','lick_start'])
#     event_colors = cfg.get('event_colors', {
#         'F1_on':'#1f77b4','F1_off':'#1f77b4',
#         'F2_on':'#ff7f0e','F2_off':'#ff7f0e',
#         'choice_start':'#2ca02c','lick_start':'#d62728'
#     })
#     smoothing_sigma = cfg.get('smoothing_sigma', 0.8)
#     sort_mode = cfg.get('sort_mode', 'index')
#     onset_window = cfg.get('onset_window', (0.0, 3.0))
#     onset_z = cfg.get('onset_z', 1.0)
#     shared_vlim = cfg.get('vlim', None)

#     ch = M['chunks']['trial_start']
#     X = ch['data']  # (trials, rois, T)
#     t = ch['time']
#     if roi_idx.size == 0:
#         print(f"[plot_group_overview_v2] Empty ROI set for {label}")
#         return
#     sub = X[:, roi_idx, :]  # (trials, nR, T)
#     if smoothing_sigma and smoothing_sigma > 0:
#         sub = gaussian_filter1d(sub, axis=2, sigma=smoothing_sigma)
#     is_short = M['is_short']
#     short_sub = sub[is_short]      # (nShort, nR, T)
#     long_sub  = sub[~is_short]

#     # Mean across trials per ROI (condition specific)
#     roi_mean_short = np.nanmean(short_sub, axis=0) if short_sub.size else np.full((len(roi_idx), t.size), np.nan)
#     roi_mean_long  = np.nanmean(long_sub, axis=0)  if long_sub.size  else np.full((len(roi_idx), t.size), np.nan)
#     roi_mean_all   = np.nanmean(sub, axis=0)

#     # ROI ordering
#     order = np.arange(len(roi_idx))
#     if sort_mode == 'onset':
#         onsets = _compute_roi_onsets(t, roi_mean_all, onset_window, z_thresh=onset_z)
#         order = np.argsort(onsets)  # inf go to end
#     roi_mean_short = roi_mean_short[order]
#     roi_mean_long  = roi_mean_long[order]

#     # Group mean traces per trial (for SEM)
#     group_trial_means = np.nanmean(sub, axis=1)            # (trials, T)
#     short_trial_means = group_trial_means[is_short]
#     long_trial_means  = group_trial_means[~is_short]

#     mean_all  = np.nanmean(group_trial_means, axis=0)
#     sem_all   = np.nanstd(group_trial_means, axis=0, ddof=1) / max(1, np.sqrt(group_trial_means.shape[0]))
#     mean_short = np.nanmean(short_trial_means, axis=0)
#     sem_short  = np.nanstd(short_trial_means, axis=0, ddof=1) / max(1, np.sqrt(short_trial_means.shape[0]))
#     mean_long  = np.nanmean(long_trial_means, axis=0)
#     sem_long   = np.nanstd(long_trial_means, axis=0, ddof=1) / max(1, np.sqrt(long_trial_means.shape[0]))
#     diff_trace = mean_long - mean_short

#     # Event stats per condition
#     event_stats_short = {}
#     event_stats_long  = {}
#     for ev in events:
#         if ev in M:
#             mu_s, sd_s = _event_stats_condition(M[ev], is_short)
#             mu_l, sd_l = _event_stats_condition(M[ev], ~is_short)
#             event_stats_short[ev] = (mu_s, sd_s)
#             event_stats_long[ev]  = (mu_l, sd_l)

#     # Color scale (shared for the two rasters)
#     if shared_vlim is None:
#         combined = np.vstack([roi_mean_short, roi_mean_long])
#         vmin, vmax = _robust_sym_vlim(combined, quant=0.98)
#     else:
#         vmin, vmax = shared_vlim

#     fig = plt.figure(figsize=cfg.get('figsize', (7.2, 8.0)))
#     gs = GridSpec(5, 1, height_ratios=[1.6,1.6,1.0,0.9,0.9], hspace=0.35, figure=fig)

#     def _plot_raster(ax, mat, title, ev_stats, cond_label):
#         im = ax.imshow(mat, aspect='auto', origin='lower',
#                        extent=(t[0], t[-1], 0, mat.shape[0]),
#                        cmap='coolwarm', vmin=vmin, vmax=vmax)
#         ax.set_ylabel(f'ROIs\n(n={mat.shape[0]})', fontsize=8)
#         ax.set_title(title, fontsize=9)
#         # Condition-specific event lines
#         for ev, (mu, sd) in ev_stats.items():
#             if not np.isfinite(mu):
#                 continue
#             col = event_colors.get(ev, 'k')
#             ax.axvline(mu, color=col, lw=0.9)
#             if np.isfinite(sd) and sd > 0:
#                 ax.axvspan(mu - sd, mu + sd, color=col, alpha=0.07)
#         return im

#     ax1 = fig.add_subplot(gs[0])
#     _plot_raster(ax1, roi_mean_short, f'{label} | Short trials', event_stats_short, 'Short')

#     ax2 = fig.add_subplot(gs[1], sharex=ax1)
#     im = _plot_raster(ax2, roi_mean_long, f'{label} | Long trials', event_stats_long, 'Long')

#     cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.015, pad=0.01)
#     cbar.set_label('Mean z-dF/F', fontsize=8)

#     # Panel 3: Combined mean + diff
#     ax3 = fig.add_subplot(gs[2], sharex=ax1)
#     ax3.plot(t, mean_all, color='k', lw=1.4, label='All')
#     ax3.fill_between(t, mean_all - sem_all, mean_all + sem_all, color='k', alpha=0.15, linewidth=0)
#     ax3.plot(t, mean_short, color='#1f77b4', lw=1.1, label='Short')
#     ax3.plot(t, mean_long, color='#ff7f0e', lw=1.1, label='Long')
#     ax3.plot(t, diff_trace, color='0.4', lw=1.0, ls='--', label='Long-Short')
#     # Event lines: show both (short solid, long dashed) if means differ > small tolerance
#     for ev in events:
#         mu_s, sd_s = event_stats_short.get(ev, (np.nan, np.nan))
#         mu_l, sd_l = event_stats_long.get(ev, (np.nan, np.nan))
#         col = event_colors.get(ev, 'k')
#         if np.isfinite(mu_s):
#             ax3.axvline(mu_s, color=col, lw=0.9, alpha=0.9)
#         if np.isfinite(sd_s) and sd_s > 0:
#             ax3.axvspan(mu_s - sd_s, mu_s + sd_s, color=col, alpha=0.05)
#         if np.isfinite(mu_l):
#             ax3.axvline(mu_l, color=col, lw=0.9, ls='--', alpha=0.9)
#         if np.isfinite(sd_l) and sd_l > 0:
#             ax3.axvspan(mu_l - sd_l, mu_l + sd_l, color=col, alpha=0.04)
#     ax3.set_ylabel('Mean', fontsize=8)
#     ax3.legend(fontsize=7, ncol=4, frameon=False)
#     ax3.grid(alpha=0.25, linewidth=0.5)

#     # Panel 4: Short
#     ax4 = fig.add_subplot(gs[3], sharex=ax1)
#     ax4.plot(t, mean_short, color='#1f77b4', lw=1.3)
#     ax4.fill_between(t, mean_short - sem_short, mean_short + sem_short,
#                      color='#1f77b4', alpha=0.18, linewidth=0)
#     for ev, (mu, sd) in event_stats_short.items():
#         if not np.isfinite(mu): continue
#         col = event_colors.get(ev, 'k')
#         ax4.axvline(mu, color=col, lw=0.9)
#         if np.isfinite(sd) and sd > 0:
#             ax4.axvspan(mu - sd, mu + sd, color=col, alpha=0.07)
#     ax4.set_ylabel('Short', fontsize=8)
#     ax4.grid(alpha=0.2, linewidth=0.4)

#     # Panel 5: Long
#     ax5 = fig.add_subplot(gs[4], sharex=ax1)
#     ax5.plot(t, mean_long, color='#ff7f0e', lw=1.3)
#     ax5.fill_between(t, mean_long - sem_long, mean_long + sem_long,
#                      color='#ff7f0e', alpha=0.18, linewidth=0)
#     for ev, (mu, sd) in event_stats_long.items():
#         if not np.isfinite(mu): continue
#         col = event_colors.get(ev, 'k')
#         ax5.axvline(mu, color=col, lw=0.9)
#         if np.isfinite(sd) and sd > 0:
#             ax5.axvspan(mu - sd, mu + sd, color=col, alpha=0.07)
#     ax5.set_ylabel('Long', fontsize=8)
#     ax5.set_xlabel('Time (s)', fontsize=9)
#     ax5.grid(alpha=0.2, linewidth=0.4)

#     # Annotate sorting
#     if sort_mode == 'onset':
#         ax1.text(0.99, 0.98, 'sorted by onset', transform=ax1.transAxes,
#                  ha='right', va='top', fontsize=7, color='k')
#     plt.tight_layout()

# def plot_all_groups_v2(M: Dict[str, Any],
#                        catalog: Dict[str, Any],
#                        groups: Optional[List[str]] = None,
#                        cfg: Optional[Dict[str, Any]] = None,
#                        compute_global_vlim: bool = True):
#     """
#     Advanced catalog component visualization with consistent scaling & onset sort.

#     Panels (per group)
#     ------------------
#     1. Short raster (ROI mean across short trials)
#     2. Long  raster
#     3. Combined mean traces (All / Short / Long / Long-Short)
#     4. Short mean focus
#     5. Long mean focus

#     Parameters
#     ----------
#     M : dict
#     catalog : dict
#         Output from build_component_catalog (incidence + comp_labels).
#     groups : list[str] | None
#         Subset of labels; None -> all.
#     cfg : dict
#         Keys (optional):
#           events            : list[str]
#           event_colors      : mapping
#           smoothing_sigma   : float (temporal smoothing)
#           sort_mode         : 'index' | 'onset'
#           onset_window      : (t0,t1) for onset search when sort_mode='onset'
#           onset_z           : z threshold above baseline for onset
#           vlim              : (vmin,vmax) if manual color scale
#           figsize           : (w,h)
#     compute_global_vlim : bool
#         If True and cfg['vlim'] absent -> derive symmetric robust limits across all rasters.

#     Color Scale
#     -----------
#     Robust symmetric (quantile 0.98 of abs) unless overridden.

#     Returns
#     -------
#     None (creates figures).  Use outside loops for selective inspection.

#     Tips
#     ----
#     * Provide groups sorted by experimental interest (e.g., all choice-related first).
#     * Use subset_labels = [l for l in labels if l.startswith('choice_start:')]
#     """
#     if cfg is None:
#         cfg = {}
#     labels = catalog['comp_labels']
#     inc = catalog['incidence']
#     if groups is None:
#         groups = labels
#     # Precompute global vlim
#     if compute_global_vlim and 'vlim' not in cfg:
#         mats = []
#         ch = M['chunks']['trial_start']['data']
#         is_short = M['is_short']
#         for lbl in groups:
#             if lbl not in labels:
#                 continue
#             j = labels.index(lbl)
#             roi_idx = np.flatnonzero(inc[:, j])
#             if roi_idx.size == 0:
#                 continue
#             sub = ch[:, roi_idx, :]
#             short_sub = sub[is_short]
#             long_sub  = sub[~is_short]
#             if short_sub.size:
#                 mats.append(np.nanmean(short_sub, axis=0))
#             if long_sub.size:
#                 mats.append(np.nanmean(long_sub, axis=0))
#         if mats:
#             all_mat = np.vstack(mats)
#             vmin, vmax = _robust_sym_vlim(all_mat)
#             cfg = dict(cfg)  # copy
#             cfg['vlim'] = (vmin, vmax)
#     for lbl in groups:
#         if lbl not in labels:
#             print(f"[warn] group '{lbl}' not in catalog; skip")
#             continue
#         j = labels.index(lbl)
#         roi_idx = np.flatnonzero(inc[:, j])
#         plot_group_overview_v2(M, roi_idx, lbl, cfg=cfg)

# ...existing code...



































# ...existing code...

# ...existing code...

def build_M_from_chunks(chunks: Dict[str, Dict[str, Any]],
                        cfg: Optional[Dict[str, Any]] = None,
                        primary_chunk: str = 'trial_start') -> Dict[str, Any]:
    """
    Construct unified session dictionary M from per‑event (chunk) DataFrames.

    Input
    -----
    chunks : dict
        Mapping {chunk_key -> trial_data_chunk_dict}.  Each trial_data_chunk_dict
        must contain a DataFrame under key 'df_trials_with_segments' with
        identical trial ordering across chunks.  Required columns inside each row:
          dff_segment : (n_rois, n_time_i) array (raw or z-scored dF/F segment)
          dff_time_vector : (n_time_i,) time vector (seconds or ms)
        Additional scalar columns (isi, choice_start, etc.) are copied to top level
        ONLY from the primary_chunk DataFrame (assumed identical across chunks).
    cfg : dict | None
        Optional per‑chunk options. Example:
          cfg = {
            'isi': {'mask_after_isi': True}
          }
        mask_after_isi : if True (default) NaN‑out time samples >= trial ISI within
                         the isi chunk (prevents leakage of post‑ISI activity into
                         “anticipation” analysis).
    primary_chunk : str
        Chunk whose DataFrame defines canonical trial order and supplies the
        scalar behavioral / event timing arrays to top level of M.

    Processing
    ----------
    1. Infers n_rois from first valid (2D) dff_segment row of primary chunk.
    2. For each chunk:
         * Builds canonical time vector from first valid row.
         * Interpolates any row whose dff_time_vector differs (ROI‑wise).
         * Computes NaN fraction.
         * Applies optional ISI masking for chunk=='isi'.
    3. Converts millisecond columns to seconds automatically (if max|value|>10).

    Output (M)
    ----------
    M : dict with keys
      n_trials, n_rois
      isi, is_short, unique_isis, short_levels, long_levels
      (behavior / event arrays) e.g. choice_start, lick_start, F1_on, ...
      chunks : {
         chunk_key : {
            data : (trials, rois, time) float32
            time : (time,) float32
            dt : float
            alignment : str (from _CHUNK_META_DEFAULTS or chunk key)
            window : (t_min, t_max)
            mask_rule : str|None
            nan_frac : float
         }, ...
      }
      primary_chunk : str
      roi_traces : (trials, rois, time_primary)  (copy of primary chunk)
      time : (time_primary,)
      dt : float (primary chunk sampling)

    Notes
    -----
    * Does NOT modify data scaling (no baseline, no z-scoring).
    * Assumes all trial-level event scalars are already aligned to the
      same reference timeline (usually absolute in session; relative
      alignment is handled later per chunk when plotting).
    * If you see large nan_frac -> check interpolation or upstream export.

    Raises
    ------
    KeyError  : primary_chunk missing
    ValueError: no trials, cannot infer n_rois, or mismatched trial counts.

    """
    cfg = cfg or {}
    if primary_chunk not in chunks:
        raise KeyError(f"primary_chunk '{primary_chunk}' missing. Keys={list(chunks)}")

    df0 = chunks[primary_chunk]['df_trials_with_segments']
    n_trials = len(df0)
    if n_trials == 0:
        raise ValueError("No trials in primary chunk.")
    # Determine n_rois from first valid row
    n_rois = None
    for r in range(n_trials):
        seg = np.asarray(df0.iloc[r]['dff_segment'])
        if seg.ndim == 2:
            n_rois = seg.shape[0]
            break
    if n_rois is None:
        raise ValueError("Could not infer n_rois (no valid dff_segment rows).")

    # Trial scalar arrays
    def _col(name, dtype=float, fill=np.nan):
        if name in df0.columns:
            arr = df0[name].to_numpy()
        else:
            if dtype == bool:
                return np.zeros(n_trials, dtype=bool)
            arr = np.full(n_trials, fill, float)
        return arr.astype(dtype, copy=False)

    isi = _to_seconds_if_ms(_col('isi', dtype=float))
    is_short = _col('is_short', dtype=bool)
    unique_isis = np.sort(np.unique(np.round(isi, 6)))
    short_levels = np.sort(np.unique(np.round(isi[is_short], 6)))
    long_levels  = np.sort(np.unique(np.round(isi[~is_short], 6)))

    behavior_cols = {
        'is_right':        _col('is_right', dtype=bool),
        'is_right_choice': _col('is_right_choice', dtype=bool),
        'rewarded':        _col('rewarded', dtype=bool),
        'punished':        _col('punished', dtype=bool),
        'did_not_choose':  _col('did_not_choose', dtype=bool),
        'time_did_not_choose': _col('time_did_not_choose', dtype=float),
        'choice_start':    _col('choice_start', dtype=float),
        'choice_stop':     _col('choice_stop', dtype=float),
        'servo_in':        _col('servo_in', dtype=float),
        'servo_out':       _col('servo_out', dtype=float),
        'lick':            _col('lick', dtype=bool),
        'lick_start':      _col('lick_start', dtype=float),
        'RT':              _col('RT', dtype=float),
        'F1_on':           _col('start_flash_1', dtype=float),
        'F1_off':          _col('end_flash_1', dtype=float),
        'F2_on':           _col('start_flash_2', dtype=float),
        'F2_off':          _col('end_flash_2', dtype=float),
    }

    chunk_out = {}
    for key, td in chunks.items():
        df = td['df_trials_with_segments']
        if len(df) != n_trials:
            raise ValueError(f"Chunk '{key}' trial mismatch ({len(df)} vs {n_trials})")
        # Validate ROI count
        probe = None
        for j in range(len(df)):
            seg = np.asarray(df.iloc[j]['dff_segment'])
            if seg.ndim == 2:
                probe = seg
                break
        if probe is None or probe.ndim != 2:
            raise ValueError(f"Chunk '{key}' no valid 2D dff_segment.")
        if probe.shape[0] != n_rois:
            raise ValueError(f"Chunk '{key}' ROI count mismatch {probe.shape[0]} vs {n_rois}")

        traces, t_vec = _stack_chunk(df, n_trials, n_rois, strict=False, chunk_name=key)
        meta = _CHUNK_META_DEFAULTS.get(key, {'alignment': key})
        mask_rule = meta.get('mask_rule')

        # Optional ISI truncation
        local_cfg = cfg.get(key, {})
        if key == 'isi' and local_cfg.get('mask_after_isi', True):
            for tr in range(n_trials):
                cutoff = isi[tr]
                if np.isfinite(cutoff):
                    mask = t_vec >= cutoff
                    if mask.any():
                        traces[tr, :, mask] = np.nan

        nan_frac = float(np.isnan(traces).sum() / traces.size)
        chunk_out[key] = dict(
            data=traces,
            time=t_vec.astype(np.float32),
            dt=float(np.nanmedian(np.diff(t_vec))) if t_vec.size > 1 else np.nan,
            alignment=meta.get('alignment', key),
            window=(float(t_vec[0]), float(t_vec[-1]) if t_vec.size else 0.0),
            mask_rule=mask_rule,
            nan_frac=nan_frac
        )

    M = dict(
        n_trials=int(n_trials),
        n_rois=int(n_rois),
        isi=isi.astype(np.float32),
        is_short=is_short.astype(bool),
        unique_isis=unique_isis.astype(np.float32),
        short_levels=short_levels.astype(np.float32),
        long_levels=long_levels.astype(np.float32),
        chunks=chunk_out,
        primary_chunk=primary_chunk,
        **behavior_cols
    )

    prim = chunk_out[primary_chunk]
    M['roi_traces'] = prim['data']
    M['time'] = prim['time']
    M['dt'] = prim['dt']
    return M

# ...existing code...


# --- Signed ROI group extraction (bidirectional motifs) ---------------------

def extract_signed_roi_groups(A: np.ndarray,
                              method: str = 'abs_top_q',
                              q: float = 0.10,
                              min_rois: int = 5,
                              max_rois: Optional[int] = None,
                              separate_signs: bool = False) -> List[Dict[str, Any]]:
    """
    Derive signed ROI subsets (Positive vs Negative) from component loadings.

    Strategy
    --------
    1. Compute absolute loadings |A[:, r]|.
    2. Select top max(min_rois, q * n_rois) indices (after sorting by magnitude).
    3. Record their sign (+1 / -1).
    4. Provide boolean masks positive_mask / negative_mask within the selected
       subset so you can easily pull either side.

    Parameters
    ----------
    A : (n_rois, n_components) array
        Loading matrix (can be z-scored or raw).
    method : str
        Currently only 'abs_top_q' (reserve for future threshold styles).
    q : float
        Quantile fraction (0 < q <= 1). 0.10 keeps top 10% (minimum min_rois).
    min_rois : int
        Minimum number of ROIs to keep even if q yields less.
    max_rois : int | None
        Optional hard cap (applied after min_rois / q logic).
    separate_signs : bool
        If False (default): one dict per component containing both subsets.
        If True: produce up to TWO entries per component:
           * one with only positives (positive_mask all True)
           * one with only negatives (negative_mask all True)
        Each carries parent_comp == original component index.

    Returns
    -------
    groups : list[dict]
        Each dict fields:
          comp / parent_comp (if separate_signs)
          roi_idx        : selected ROI indices
          signs          : sign vector (len = len(roi_idx))
          positive_mask  : bool mask into roi_idx
          negative_mask  : bool mask into roi_idx

    Usage
    -----
    signed = extract_signed_roi_groups(A, q=0.08)
    for g in signed:
        pos = g['roi_idx'][g['positive_mask']]
        neg = g['roi_idx'][g['negative_mask']]

    Caveats
    -------
    * If a component is nearly uni-signed you may get empty negative subset.
    * Orientation (deciding which sign should be “Positive” wrt activity) is
      performed later via orient_signed_groups / finalize_signed_orientation.
    """
    R = A.shape[1]
    out = []
    for r in range(R):
        v = A[:, r]
        mags = np.abs(v)
        if np.all(mags == 0):
            continue
        k = max(min_rois, int(len(mags) * q)) if method == 'abs_top_q' else min_rois
        order = np.argsort(mags)[::-1]
        idx = order[:k]
        if max_rois and idx.size > max_rois:
            idx = idx[:max_rois]
        signs = np.sign(v[idx]).astype(int)
        pos_mask = signs > 0
        neg_mask = signs < 0
        if separate_signs:
            if pos_mask.any():
                out.append(dict(parent_comp=r, comp=r, roi_idx=idx[pos_mask],
                                signs=np.ones(pos_mask.sum(), int),
                                positive_mask=np.ones(pos_mask.sum(), bool),
                                negative_mask=np.zeros(pos_mask.sum(), bool)))
            if neg_mask.any():
                out.append(dict(parent_comp=r, comp=r, roi_idx=idx[neg_mask],
                                signs=-np.ones(neg_mask.sum(), int),
                                positive_mask=np.zeros(neg_mask.sum(), bool),
                                negative_mask=np.ones(neg_mask.sum(), bool)))
        else:
            out.append(dict(comp=r, roi_idx=idx, signs=signs,
                            positive_mask=pos_mask, negative_mask=neg_mask))
    return out

def print_signed_component_summary(signed_groups: List[Dict[str, Any]]):
    for g in signed_groups:
        n = len(g['roi_idx'])
        n_pos = g['positive_mask'].sum()
        n_neg = g['negative_mask'].sum()
        label = f"comp{g.get('comp')}"
        if 'parent_comp' in g:
            label += f" (parent {g['parent_comp']})"
        print(f"{label}: total={n}, +={n_pos}, -={n_neg}")

# --- Phase-resampled ISI chunk ---------------------------------------------

def add_phase_resampled_chunk(M: Dict[str, Any],
                              source_chunk: str = 'isi',
                              target_chunk: str = 'isi_phase',
                              n_phase: int = 80,
                              clamp: bool = True) -> Dict[str, Any]:
    """
    Create a new chunk with each trial's ISI segment resampled to a fixed number
    of phase bins (φ in [0,1]). Keeps NaNs for bins outside valid original support.
    """
    if target_chunk in M['chunks']:
        raise ValueError(f"Chunk '{target_chunk}' already exists.")
    if source_chunk not in M['chunks']:
        raise KeyError(f"Source chunk '{source_chunk}' not found.")
    ch = M['chunks'][source_chunk]
    X = ch['data']              # (trials, rois, time)
    t = ch['time']              # (time,)
    isi = M['isi']
    trials, rois, T = X.shape
    phase_bins = np.linspace(0, 1, n_phase)
    out = np.full((trials, rois, n_phase), np.nan, dtype=np.float32)

    for tr in range(trials):
        dur = isi[tr]
        if not (np.isfinite(dur) and dur > 0):
            continue
        # valid region: 0 <= t < dur (assuming alignment at end_flash_1)
        valid = (t >= 0) & (t < dur)
        if valid.sum() < 3:
            continue
        t_valid = t[valid]
        phi = t_valid / dur
        if clamp:
            phi = np.clip(phi, 0, 1)
        for r in range(rois):
            y = X[tr, r, valid]
            m = np.isfinite(y)
            if m.sum() < 2:
                continue
            out[tr, r] = np.interp(phase_bins, phi[m], y[m], left=np.nan, right=np.nan)

    M['chunks'][target_chunk] = dict(
        data=out,
        time=phase_bins.astype(np.float32),
        dt=float(np.median(np.diff(phase_bins))),
        alignment='phase_isi',
        window=(0.0, 1.0),
        mask_rule='phase',
        nan_frac=float(np.isnan(out).sum()/out.size)
    )
    return M

# --- Signed profile diagnostic plot ----------------------------------------

# ...existing code...
def plot_component_signed_profile(M: Dict[str, Any],
                                  res: Dict[str, Any],
                                  comp_idx: int,
                                  chunk: str,
                                  signed_groups: Optional[List[Dict[str, Any]]] = None,
                                  use_phase: bool = False):
    """
    Overlay mean traces of positive vs negative ROI subsets for a component.
    Works with normal time or phase-resampled chunk.
    """
    import matplotlib.pyplot as plt
    A = res['factors']['A']
    if comp_idx >= A.shape[1]:
        raise IndexError("comp_idx out of range.")
    if signed_groups is None:
        signed_groups = extract_signed_roi_groups(A, separate_signs=False)
    group = next((g for g in signed_groups if g['comp'] == comp_idx), None)
    if group is None:
        print("Component not found in signed groups.")
        return
    ch = M['chunks'][chunk]
    X = ch['data']
    t = ch['time']
    roi_idx = group['roi_idx']
    signs = group['signs']
    pos_idx = roi_idx[signs > 0]
    neg_idx = roi_idx[signs < 0]

    mean_pos = (safe_nanmean(X[:, pos_idx], axis=(0,1))
                if pos_idx.size and np.any(np.isfinite(X[:, pos_idx])) else None)
    mean_neg = (safe_nanmean(X[:, neg_idx], axis=(0,1))
                if neg_idx.size and np.any(np.isfinite(X[:, neg_idx])) else None)
    # ...existing code...

    plt.figure(figsize=(5,3))
    # Corrected color spec ('tab:blue', 'tab:red')
    if mean_pos is not None:
        plt.plot(t, mean_pos, color='tab:blue', label='Positive')
    if mean_neg is not None:
        plt.plot(t, mean_neg, color='tab:red', label='Negative')
    plt.axhline(0, color='k', lw=0.6)
    plt.xlabel("ISI phase (φ)" if use_phase else "Time (s)")
    plt.ylabel("Mean z-dF/F")
    plt.title(f"{chunk} comp {comp_idx} (signed subsets)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
# ...existing code...

# --- CP pipeline tweak: respect pre_zscored flag ---------------------------

def run_chunk_cp_pipeline(M: Dict[str, Any],
                          chunk: str,
                          ranks: List[int] = [8, 10, 12],
                          B: int = 40,
                          sample_frac: float = 0.7,
                          stratify_isi: bool = True,
                          standardize_rois: bool = True,
                          pre_zscored: bool = False,
                          random_state: Optional[int] = 0) -> Dict[str, Any]:
    """
    Added 'pre_zscored': if True, forces standardize_rois=False regardless of argument.
    """
    if pre_zscored:
        standardize_rois = False
    # ...existing code (unchanged until ROI standardization block)...
    tensor, t = build_bagged_trial_means(M, chunk, B=B,
                                         sample_frac=sample_frac,
                                         stratify_isi=stratify_isi,
                                         random_state=random_state)
    tensor_clean, time_keep, bag_keep = clean_tensor_for_cp(tensor)
    t_clean = t[time_keep]
    if tensor_clean.size == 0:
        raise ValueError("After cleaning, tensor is empty.")
    mask = ~np.isnan(tensor_clean)
    X = np.nan_to_num(tensor_clean, nan=0.0)

    if standardize_rois:
        # ...existing standardization code...
        mean_roi = np.sum(X, axis=2).mean(axis=1, keepdims=True) / max(1, X.shape[2])
        std_roi = np.sqrt(np.sum((X - mean_roi[:, :, None])**2, axis=(1,2)) /
                          (X.shape[1]*X.shape[2]-1 + 1e-9))
        std_roi[std_roi == 0] = 1
        X = X / std_roi[:, None, None]

    results = []
    for r in ranks:
        try:
            res = cp_decompose_masked(X, mask, rank=r, random_state=random_state)
            res['rank'] = r
            results.append(res)
        except Exception as e:
            print(f"[warn] rank {r} failed: {e}")
    if not results:
        raise RuntimeError("All CP decompositions failed.")
    best = min(results, key=lambda d: d['loss'])
    roi_sets = extract_roi_sets(best['A'], method='top_q')
    return dict(
        chunk=chunk,
        time=t_clean,
        time_indices=time_keep,
        bag_indices=bag_keep,
        factors=best,
        roi_sets=roi_sets,
        tested_ranks=[r for r in ranks],
        all_results=results,
        pre_zscored=pre_zscored
    )

# ...existing code...













# Which chunks to run full CP (time-domain)
RANKS = [6,8,10]
MIN_ABS_CORR = 0.65
# Phase CP
PHASE_BINS = 80
# Output CSV
ANNOT_CSV = "roi_roles.csv"
# --------------------------



def run_time_domain_analysis(M, chunk_keys, ranks, min_abs_corr):
    """
    Convenience wrapper to run the full CP + pruning + stability pipeline on
    multiple time-domain chunks.

    Parameters
    ----------
    M : dict
        Session structure from build_M_from_chunks.
    chunk_keys : list[str]
        Chunks to analyze (e.g. ['isi','choice_start',...]).
    ranks : list[int]
        Candidate CP ranks (passed to run_cp_full_pipeline).
    min_abs_corr : float
        Split-half absolute loading correlation threshold used during the
        stability-based pruning stage (final filter).

    Returns
    -------
    results : dict
        Mapping chunk -> cp_full_result with fields:
          final_A / final_B          : factor matrices (rois x R, time x R)
          final_roi_sets             : list[np.ndarray] ROI indices per component
          stability_vec              : abs correlations for components BEFORE
                                       stability pruning was applied
          base, stability, params    : intermediate diagnostic sub-dicts
          rank                       : chosen (best-loss) rank prior to pruning

    Tips
    ----
    * Inspect results[ch]['stability_vec'] to judge reliability.
    * If many components fall below min_abs_corr increase B or adjust ranks.
    """    
    results = {}
    for ch in chunk_keys:
        print(f"[CP full] {ch}")
        res = run_cp_full_pipeline(
            M, ch,
            ranks=ranks,
            min_abs_corr=min_abs_corr,
            B=50,
            stability_B=60,
            sample_frac=0.7,
            stratify_isi=(ch=='isi'),
            random_state=0
        )
        print(f"  final comps={res['final_A'].shape[1]}")
        results[ch] = res
    return results


# --- Helper to orient signed groups (phase & time) -------------------------
def orient_signed_groups(A: np.ndarray,
                         signed_groups: List[Dict[str, Any]],
                         M: Optional[Dict[str, Any]] = None,
                         chunk: Optional[str] = None,
                         method: str = 'loading',
                         phase_center: Tuple[float,float] = (0.05, 0.95)) -> List[Dict[str, Any]]:
    """
    Heuristically flip component signs so the 'positive' subset corresponds
    to the larger mean activity metric.

    Parameters
    ----------
    A : (n_rois, n_components)
        Loading matrix (W). This function MAY modify A in-place (sign flips).
    signed_groups : list[dict]
        Output of extract_signed_roi_groups (modified in-place).
    M : dict | None
        Session dictionary (required for trace-based methods).
    chunk : str | None
        Chunk key whose data is used for trace averaging.
    method : str
        'loading'      : compare mean positive loading vs |mean negative loading|
        'trace_mean'   : compare mean over time of trial-averaged traces (all time)
        'trace_phase'  : same but only inside phase_center window (for phase chunk)
    phase_center : (lo, hi)
        Phase window used only when method='trace_phase'.

    Behavior
    --------
    * If neg_metric > pos_metric -> multiply A[:, comp] and group.signs by -1
      and recompute masks.

    Returns
    -------
    signed_groups (same list object for chaining).

    Notes
    -----
    * Use finalize_signed_orientation afterwards to enforce final mean trace
      convention in case of ties / stochastic changes.
    * Safe to call multiple times (idempotent when no further flips needed).
    """
    if M is not None and chunk is not None:
        X = M['chunks'][chunk]['data']  # (trials, rois, time)
        t = M['chunks'][chunk]['time']
    else:
        X = None
        t = None

    for g in signed_groups:
        comp = g['comp']
        roi_idx = g['roi_idx']
        signs = g['signs']
        pos_idx = roi_idx[g['positive_mask']]
        neg_idx = roi_idx[g['negative_mask']]
        if pos_idx.size == 0 or neg_idx.size == 0:
            continue

    # ...existing code...
        def _mean_trace(idxs, use_window=False):
            if X is None or idxs.size == 0:
                return np.nan
            mt = safe_nanmean(X[:, idxs], axis=(0,1))
            if use_window and t is not None:
                lo, hi = phase_center
                mask = (t >= lo) & (t <= hi)
                if mask.any():
                    return float(np.nanmean(mt[mask]))
            return float(np.nanmean(mt))

        if method == 'loading':
            pos_metric = float(np.nanmean(A[pos_idx, comp]))
            neg_metric = float(np.nanmean(-A[neg_idx, comp]))  # make positive magnitude
        elif method == 'trace_mean':
            pos_metric = _mean_trace(pos_idx, use_window=False)
            neg_metric = _mean_trace(neg_idx, use_window=False)
        elif method == 'trace_phase':
            pos_metric = _mean_trace(pos_idx, use_window=True)
            neg_metric = _mean_trace(neg_idx, use_window=True)
        else:
            raise ValueError("Unknown method for orientation.")

        # If negative side stronger, flip
        if neg_metric > pos_metric:
            A[:, comp] *= -1
            signs *= -1
            g['signs'] = signs
            g['positive_mask'] = signs > 0
            g['negative_mask'] = signs < 0
    return signed_groups


# ...existing code...

def run_phase_analysis(M):
    print("[Phase] adding phase-resampled chunk...")
    M['chunks'].pop('isi_phase', None)
    add_phase_resampled_chunk(M, source_chunk='isi', target_chunk='isi_phase', n_phase=PHASE_BINS)
    print("[Phase] CP decomposition...")
    res_phase = run_chunk_cp_pipeline(
        M, 'isi_phase',
        ranks=[8,10,12],
        B=40,
        sample_frac=0.7,
        stratify_isi=False,
        standardize_rois=False,
        pre_zscored=True,
        random_state=0
    )
    print(f"[Phase] best rank={res_phase['factors']['rank']} loss={res_phase['factors']['loss']:.4g}")
    signed = extract_signed_roi_groups(res_phase['factors']['A'], q=0.10)
    orient_signed_groups(res_phase['factors']['A'], signed, M=M, chunk='isi_phase', method='trace_phase')
    orient_signed_groups(res_phase['factors']['A'], signed, M=M, chunk='isi_phase', method='trace_mean')
    finalize_signed_orientation(M, res_phase, signed, 'isi_phase')
    print_signed_component_summary(signed)
    for c in range(min(5, res_phase['factors']['A'].shape[1])):
        plot_component_signed_profile(M, res_phase, comp_idx=c, chunk='isi_phase',
                                      signed_groups=signed, use_phase=True)
    return res_phase, signed

# ...existing code...

def integrate_phase_components_into_catalog(catalog: Dict[str, Any],
                                            res_phase: Dict[str, Any],
                                            signed_groups: List[Dict[str, Any]],
                                            M: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Append isi_phase signed component ROI sets into existing catalog (adds two columns per component:
    positive & negative subsets). Works with original catalog structure (incidence + comp_labels).
    """
    if catalog is None or 'incidence' not in catalog or 'comp_labels' not in catalog:
        return catalog
    inc = catalog['incidence']
    labels = catalog['comp_labels']
    n_rois = inc.shape[0]
    new_cols = []
    new_labels = []
    for g in signed_groups:
        comp = g['comp']
        roi_idx = g['roi_idx']
        pos_idx = roi_idx[g['positive_mask']]
        neg_idx = roi_idx[g['negative_mask']]
        if pos_idx.size:
            col = np.zeros(n_rois, bool); col[pos_idx] = True
            new_cols.append(col); new_labels.append(f"isi_phase:comp{comp}_pos")
        if neg_idx.size:
            col = np.zeros(n_rois, bool); col[neg_idx] = True
            new_cols.append(col); new_labels.append(f"isi_phase:comp{comp}_neg")
    if new_cols:
        new_inc = np.column_stack([inc] + new_cols)
        catalog['incidence'] = new_inc
        catalog['comp_labels'].extend(new_labels)
    return catalog

# ...existing code...






# ...existing code...

def finalize_signed_orientation(M: Dict[str, Any],
                                res: Dict[str, Any],
                                signed_groups: List[Dict[str, Any]],
                                chunk: str) -> None:
    """
    Final pass enforcing Positive > Negative (scalar mean trace criterion).

    Logic
    -----
    1. For each component with both subsets:
         * Compute mean trace across trials & ROIs for positives and negatives.
         * Collapse to scalar means mp, mn.
         * If mp <= mn flip loading signs and update masks.

    Parameters
    ----------
    M : session dict
    res : CP result dict containing res['factors']['A'] (modified in-place).
    signed_groups : list[dict]
        Signed group definitions (modified in-place).
    chunk : str
        Chunk key for extracting trial x ROI x time data.

    Guarantees
    ----------
    After this step average(Positive trace) > average(Negative trace) OR
    one subset is empty (no change applied).

    Side Effects
    ------------
    * Mutates A (sign flips) and signed_groups entries.
    """
    X = M['chunks'][chunk]['data']          # (trials, rois, time/phase)
    A = res['factors']['A']
    for g in signed_groups:
        comp = g['comp']
        roi_idx = g['roi_idx']
        signs = g['signs']
        pos_idx = roi_idx[g['positive_mask']]
        neg_idx = roi_idx[g['negative_mask']]
        if pos_idx.size == 0 or neg_idx.size == 0:
            continue
        # Mean trace across trials & ROIs -> (time,)
        mp_trace = safe_nanmean(X[:, pos_idx], axis=(0,1))  # (time,)
        mn_trace = safe_nanmean(X[:, neg_idx], axis=(0,1))  # (time,)
        # Collapse to scalar (overall mean); ignore if all-NaN
        mp = float(np.nanmean(mp_trace)) if np.any(np.isfinite(mp_trace)) else np.nan
        mn = float(np.nanmean(mn_trace)) if np.any(np.isfinite(mn_trace)) else np.nan
        if np.isfinite(mp) and np.isfinite(mn) and mp <= mn:
            A[:, comp] *= -1
            signs *= -1
            g['signs'] = signs
            g['positive_mask'] = signs > 0
            g['negative_mask'] = signs < 0

#






def safe_nanmean(a: np.ndarray, axis=None, keepdims=False):
    """
    NaN-mean without RuntimeWarning; returns NaN where count==0.
    """
    a = np.asarray(a)
    with np.errstate(invalid='ignore'):
        s = np.nansum(a, axis=axis, keepdims=keepdims)
        c = np.sum(np.isfinite(a), axis=axis, keepdims=keepdims)
        out = s / np.maximum(c, 1)
        if np.isscalar(out):
            if c == 0:
                return np.nan
        else:
            out[c == 0] = np.nan
    return out















# ...existing code...

def compute_event_phase_stats(M: Dict[str, Any],
                              events: Tuple[str, ...] = ('F2_on', 'F2_off'),
                              clamp: bool = True) -> Dict[str, Tuple[float, float]]:
    """
    Convert event times to ISI phase: φ = (event_time - F1_off) / ISI.
    Returns dict: event -> (mean_phase, std_phase) for phases within [0,1].
    """
    isi = M['isi']
    F1_off = M.get('F1_off', None)
    out = {}
    if F1_off is None:
        return out
    for ev in events:
        if ev not in M:
            continue
        raw = M[ev]
        with np.errstate(invalid='ignore', divide='ignore'):
            phi = (raw - F1_off) / isi
        valid = np.isfinite(phi) & np.isfinite(isi) & (isi > 0)
        if clamp:
            phi = np.clip(phi, 0, 1)
        phi_valid = phi[valid & (phi >= 0) & (phi <= 1)]
        if phi_valid.size:
            out[ev] = (float(np.nanmean(phi_valid)), float(np.nanstd(phi_valid)))
    return out

def plot_isi_phase_components_posneg(M: Dict[str, Any],
                                     res_phase: Dict[str, Any],
                                     signed_groups: List[Dict[str, Any]],
                                     comp_indices: Optional[List[int]] = None,
                                     smooth_sigma: float = 0.0,
                                     events: Tuple[str, ...] = ('F2_on',),
                                     show_negative: bool = True,
                                     max_comps: Optional[int] = None,
                                     figsize: Tuple[float,float] = (5.2, 6.0)) -> None:
    """
    For each requested phase component create a figure with 3 subplots:
      1. All trials (Positive vs Negative ROI mean traces over phase)
      2. Short trials only
      3. Long trials only
    Event markers shown as mean (solid line) + ±SD span in phase units.
    """
    chunk_key = 'isi_phase'
    if chunk_key not in M['chunks']:
        print("[plot_isi_phase_components_posneg] Missing isi_phase chunk.")
        return
    X = M['chunks'][chunk_key]['data']          # (trials, rois, phase_bins)
    phi_axis = M['chunks'][chunk_key]['time']   # (phase_bins,)
    is_short = M['is_short']
    n_trials = X.shape[0]
    ev_stats = compute_event_phase_stats(M, events=events)

    # Map comp -> signed group entry
    comp_to_group = {g['comp']: g for g in signed_groups}
    R = res_phase['factors']['A'].shape[1]
    if comp_indices is None:
        comp_indices = list(range(R))
    if max_comps is not None:
        comp_indices = comp_indices[:max_comps]

    from scipy.ndimage import gaussian_filter1d

    for c in comp_indices:
        g = comp_to_group.get(c, None)
        if g is None:
            continue
        roi_idx = g['roi_idx']
        pos_mask = g['positive_mask']
        neg_mask = g['negative_mask']
        pos_idx = roi_idx[pos_mask]
        neg_idx = roi_idx[neg_mask]

        def _mean_trace(trial_mask: np.ndarray, idxs: np.ndarray) -> Optional[np.ndarray]:
            if idxs.size == 0 or trial_mask.sum() == 0:
                return None
            sub = X[trial_mask][:, idxs, :]  # (trials_sel, rois_sel, phase_bins)
            mt = safe_nanmean(sub, axis=(0,1))
            if smooth_sigma and mt is not None and np.isfinite(mt).any():
                mt = gaussian_filter1d(mt, sigma=smooth_sigma)
            return mt

        all_mask = np.ones(n_trials, bool)
        short_mask = is_short.astype(bool)
        long_mask = ~is_short.astype(bool)

        pos_all = _mean_trace(all_mask, pos_idx)
        neg_all = _mean_trace(all_mask, neg_idx) if show_negative else None
        pos_short = _mean_trace(short_mask, pos_idx)
        neg_short = _mean_trace(short_mask, neg_idx) if show_negative else None
        pos_long = _mean_trace(long_mask, pos_idx)
        neg_long = _mean_trace(long_mask, neg_idx) if show_negative else None

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize)
        titles = ["All trials", "Short trials", "Long trials"]
        data_pairs = [(pos_all, neg_all), (pos_short, neg_short), (pos_long, neg_long)]

        for ax, title, (pdat, ndat) in zip(axes, titles, data_pairs):
            if pdat is not None:
                ax.plot(phi_axis, pdat, color='tab:blue', lw=1.4, label='Positive')
            if ndat is not None:
                ax.plot(phi_axis, ndat, color='tab:red', lw=1.2, ls='--', label='Negative')
            # Event phase markers
            for ev, (mu, sd) in ev_stats.items():
                ax.axvline(mu, color='k', lw=0.8)
                if np.isfinite(sd) and sd > 0:
                    ax.axvspan(mu - sd, mu + sd, color='k', alpha=0.08)
            ax.set_ylabel('Mean z-dF/F', fontsize=8)
            ax.set_title(f"comp {c} | {title}", fontsize=9)
            ax.axhline(0, color='k', lw=0.5, alpha=0.6)
            ax.grid(alpha=0.2, linewidth=0.4)
        axes[-1].set_xlabel('ISI phase (φ)', fontsize=9)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(fontsize=8, frameon=False, ncol=2)
        fig.tight_layout()








def plot_isi_phase_components_matrix(M: Dict[str, Any],
                                     res_phase: Dict[str, Any],
                                     signed_groups: List[Dict[str, Any]],
                                     comps: Optional[List[int]] = None,
                                     events: Tuple[str, ...] = ('F2_on',),
                                     show_negative: bool = True,
                                     smooth_sigma: float = 0.6,
                                     layout: str = 'grid',
                                     max_comps: Optional[int] = None,
                                     dedup_corr_thresh: Optional[float] = None,
                                     corr_mode: str = 'all',
                                     figsize_per: Tuple[float,float] = (2.2, 1.5)) -> None:
    """
    Batch plot phase-resampled component POS/NEG traces (All / Short / Long).

    Key Options
    -----------
    layout : 'grid' | 'per_component'
        grid -> compact matrix; per_component -> individual vertical figures.
    dedup_corr_thresh : float | None
        If set, drop later components whose Positive (or chosen corr_mode) trace
        Pearson correlation with an earlier kept component > threshold.
        Use to suppress visually redundant motifs.
    corr_mode : 'all' | 'pos' | 'neg'
        Trace used for correlation comparison. 'all' -> fall back to pos_all.
    max_comps : int | None
        Hard cap after optional subset filtering (before dedup).

    Tips
    ----
    * For dense ranks start with dedup_corr_thresh ~0.95–0.98.
    * Combine with cluster_isi_phase_components to obtain representative set.

    """
    chunk_key = 'isi_phase'
    if chunk_key not in M['chunks']:
        print("[plot_isi_phase_components_matrix] Missing isi_phase chunk.")
        return
    X = M['chunks'][chunk_key]['data']          # (trials, rois, phase_bins)
    phi = M['chunks'][chunk_key]['time']
    is_short = M['is_short'].astype(bool)
    n_trials = X.shape[0]

    comp_map = {g['comp']: g for g in signed_groups}
    R = res_phase['factors']['A'].shape[1]
    if comps is None:
        comps = list(range(R))
    comps = [c for c in comps if c in comp_map]

    if max_comps:
        comps = comps[:max_comps]

    from scipy.ndimage import gaussian_filter1d

    def mean_trace(trial_mask, roi_idx):
        if roi_idx.size == 0 or trial_mask.sum() == 0:
            return None
        sub = X[trial_mask][:, roi_idx, :]
        mt = safe_nanmean(sub, axis=(0,1))
        if mt is not None and np.any(np.isfinite(mt)) and smooth_sigma:
            mt = gaussian_filter1d(mt, sigma=smooth_sigma)
        return mt

    # Precompute all requested traces
    traces = {}
    for c in comps:
        g = comp_map[c]
        pos_idx = g['roi_idx'][g['positive_mask']]
        neg_idx = g['roi_idx'][g['negative_mask']]
        all_mask = np.ones(n_trials, bool)
        short_mask = is_short
        long_mask = ~is_short
        traces[c] = dict(
            pos_all = mean_trace(all_mask, pos_idx),
            neg_all = mean_trace(all_mask, neg_idx) if show_negative else None,
            pos_short = mean_trace(short_mask, pos_idx),
            neg_short = mean_trace(short_mask, neg_idx) if show_negative else None,
            pos_long = mean_trace(long_mask, pos_idx),
            neg_long = mean_trace(long_mask, neg_idx) if show_negative else None
        )

    # Deduplicate if requested
    if dedup_corr_thresh is not None and dedup_corr_thresh > 0:
        kept = []
        ref_mat = []
        for c in comps:
            # choose correlation basis
            if corr_mode == 'neg' and traces[c]['neg_all'] is not None:
                basis = traces[c]['neg_all']
            else:
                basis = traces[c]['pos_all']
            if basis is None or not np.any(np.isfinite(basis)):
                continue
            basis_z = basis - np.nanmean(basis)
            denom = np.nanstd(basis_z) + 1e-12
            basis_z = basis_z / denom
            is_dup = False
            for r in ref_mat:
                # aligned finite mask
                m = np.isfinite(basis_z) & np.isfinite(r)
                if m.sum() < 5:
                    continue
                corr = np.dot(basis_z[m], r[m]) / max(1, m.sum())
                if corr > dedup_corr_thresh:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(c)
                ref_mat.append(basis_z)
        if not kept:
            print("[plot_isi_phase_components_matrix] Dedup removed all components.")
            return
        comps = kept
        print(f"[dedup] kept {len(comps)} components after correlation threshold {dedup_corr_thresh}")

    ev_stats = compute_event_phase_stats(M, events=events)

    def draw_one(ax, pos, neg, title=None):
        if pos is not None:
            ax.plot(phi, pos, color='tab:blue', lw=1.2, label='Positive')
        if neg is not None:
            ax.plot(phi, neg, color='tab:red', lw=1.0, ls='--', label='Negative')
        for ev, (mu, sd) in ev_stats.items():
            ax.axvline(mu, color='k', lw=0.7)
            if np.isfinite(sd) and sd > 0:
                ax.axvspan(mu - sd, mu + sd, color='k', alpha=0.07)
        ax.axhline(0, color='k', lw=0.4, alpha=0.6)
        if title:
            ax.set_title(title, fontsize=8)
        ax.grid(alpha=0.2, linewidth=0.4)

    if layout == 'grid':
        nC = len(comps)
        fig_w = 3 * figsize_per[0]
        fig_h = nC * figsize_per[1]
        fig, axes = plt.subplots(nC, 3, sharex=True, figsize=(fig_w, fig_h))
        if nC == 1:
            axes = np.array([axes])
        for r, c in enumerate(comps):
            tr = traces[c]
            draw_one(axes[r,0], tr['pos_all'], tr['neg_all'], f"comp {c} | All")
            draw_one(axes[r,1], tr['pos_short'], tr['neg_short'], "Short")
            draw_one(axes[r,2], tr['pos_long'], tr['neg_long'], "Long")
            if r == 0:
                axes[r,0].legend(frameon=False, fontsize=7, ncol=2)
        for ax in axes[-1]:
            ax.set_xlabel("ISI phase (φ)")
        for r in range(nC):
            axes[r,0].set_ylabel("Mean z-dF/F", fontsize=7)
        fig.tight_layout()
    else:  # per_component
        for c in comps:
            tr = traces[c]
            fig, axs = plt.subplots(3,1,sharex=True, figsize=(4.0, 5.0))
            draw_one(axs[0], tr['pos_all'], tr['neg_all'], f"comp {c} | All")
            draw_one(axs[1], tr['pos_short'], tr['neg_short'], "Short")
            draw_one(axs[2], tr['pos_long'], tr['neg_long'], "Long")
            axs[2].set_xlabel("ISI phase (φ)")
            for ax in axs:
                ax.set_ylabel("Mean")
            axs[0].legend(frameon=False, fontsize=8)
            fig.tight_layout()

# ...existing code...










# filepath: d:\PHD\GIT\data_analysis\DAP\modules\experiments\single_interval_discrimination\sid_roi_labeling.py
# ...existing code...

def list_isi_phase_component_rois(M: Dict[str, Any],
                                  res_phase: Dict[str, Any],
                                  signed_groups: List[Dict[str, Any]],
                                  roi_labels: Optional[List[str]] = None,
                                  sort_by: str = 'comp',
                                  show_overlap: bool = True,
                                  overlap_mode: str = 'pos',
                                  min_jaccard_report: float = 0.3) -> pd.DataFrame:
    """
    Return DataFrame with ROI indices (and optional labels) for Positive / Negative
    subsets of each phase component. Optionally prints pairwise Jaccard overlaps.

    overlap_mode: 'pos' | 'neg' | 'both'   (which side to compute overlap on)
    """
    rows = []
    for g in signed_groups:
        c = g['comp']
        roi_idx = g['roi_idx']
        pos = roi_idx[g['positive_mask']]
        neg = roi_idx[g['negative_mask']]
        for grp_name, arr in (('pos', pos), ('neg', neg)):
            if arr.size == 0:
                continue
            if roi_labels is not None:
                labels = [roi_labels[i] for i in arr]
            else:
                labels = [int(i) for i in arr]
            rows.append(dict(
                comp=c,
                group=grp_name,
                n_rois=arr.size,
                roi_indices=arr.tolist(),
                roi_labels=labels
            ))
    if not rows:
        return pd.DataFrame(columns=['comp','group','n_rois','roi_indices','roi_labels'])
    df = pd.DataFrame(rows)
    if sort_by == 'size':
        df = df.sort_values(['n_rois','comp','group'], ascending=[False, True, True])
    else:
        df = df.sort_values(['comp','group'])
    # Print concise summary
    print("ISI phase component ROI groups:")
    for _, r in df.iterrows():
        print(f" comp {r.comp:2d} {r.group}: n={r.n_rois}")
    # Overlap diagnostics
    if show_overlap:
        # Build list of chosen sets for specified mode
        if overlap_mode == 'pos':
            sets = {g['comp']: g['roi_idx'][g['positive_mask']] for g in signed_groups}
        elif overlap_mode == 'neg':
            sets = {g['comp']: g['roi_idx'][g['negative_mask']] for g in signed_groups}
        else:  # both -> union
            sets = {g['comp']: np.unique(np.concatenate([
                g['roi_idx'][g['positive_mask']], g['roi_idx'][g['negative_mask']]
            ])) for g in signed_groups}
        comps = sorted(sets.keys())
        print(f"\nPairwise Jaccard overlaps ({overlap_mode} sets) >= {min_jaccard_report}:")
        any_line = False
        for i, ci in enumerate(comps):
            Si = sets[ci]
            for cj in comps[i+1:]:
                Sj = sets[cj]
                if Si.size == 0 and Sj.size == 0:
                    continue
                inter = np.intersect1d(Si, Sj).size
                union = np.union1d(Si, Sj).size
                j = inter / union if union else 0.0
                if j >= min_jaccard_report:
                    print(f"  comp {ci} vs {cj}: J={j:.3f} (|∩|={inter}, |∪|={union})")
                    any_line = True
        if not any_line:
            print("  (none)")
    return df

# ...existing code...














# filepath: d:\PHD\GIT\data_analysis\DAP\modules\experiments\single_interval_discrimination\sid_roi_labeling.py
# ...existing code...

def cluster_isi_phase_components(signed_groups: List[Dict[str, Any]],
                                 jaccard_thresh: float = 0.80,
                                 mode: str = 'pos',
                                 representative: str = 'largest_pos') -> Dict[str, Any]:
    """
    Greedy clustering of phase components by ROI overlap.

    Parameters
    ----------
    signed_groups : list[dict]
        Phase signed groups (usually from run_phase_analysis).
    jaccard_thresh : float
        Merge components with set overlap J >= threshold (selected set via 'mode').
    mode : str
        'pos'  : cluster on positive subsets
        'neg'  : negative subsets
        'both' : union(pos,neg)
    representative : str
        Representative component rule within a cluster:
          'largest_pos'   (default)
          'largest_total' (largest union size)
          'first'         (first encountered / ordering)

    Returns
    -------
    clustering : dict
      clusters            : list of cluster dicts
      keep_components     : representative component indices (original order)
      dropped_components  : redundant component indices
      params              : parameter echo

    Usage
    -----
    cl = cluster_isi_phase_components(...);
    reps = build_representative_phase_groups(signed_groups, cl);
    plot_isi_phase_components_matrix(..., reps, comps=cl['keep_components'])

    Limits
    ------
    * Greedy approach; order affects clustering (seeded by positive size).
    """
    # Build per-component sets
    comp_sets = {}
    pos_sizes = {}
    neg_sizes = {}
    for g in signed_groups:
        c = g['comp']
        roi_idx = g['roi_idx']
        pos = roi_idx[g['positive_mask']]
        neg = roi_idx[g['negative_mask']]
        if mode == 'pos':
            base = pos
        elif mode == 'neg':
            base = neg
        else:
            base = np.unique(np.concatenate([pos, neg]))
        comp_sets[c] = np.unique(base)
        pos_sizes[c] = pos.size
        neg_sizes[c] = neg.size

    comps = sorted(comp_sets.keys())
    used = set()
    clusters = []

    # Order seed components (largest positive first helps form stable clusters)
    seed_order = sorted(comps, key=lambda c: pos_sizes[c], reverse=True)

    def jaccard(a, b):
        if a.size == 0 and b.size == 0: return 1.0
        if a.size == 0 or b.size == 0: return 0.0
        inter = np.intersect1d(a, b).size
        if inter == 0: return 0.0
        union = np.union1d(a, b).size
        return inter / union if union else 0.0

    for seed in seed_order:
        if seed in used:
            continue
        members = [seed]
        used.add(seed)
        S_seed = comp_sets[seed]
        for other in comps:
            if other in used:
                continue
            J = jaccard(S_seed, comp_sets[other])
            if J >= jaccard_thresh:
                members.append(other)
                used.add(other)
        # Pick representative
        if representative == 'largest_pos':
            rep = max(members, key=lambda c: pos_sizes[c])
        elif representative == 'largest_total':
            rep = max(members, key=lambda c: np.unique(
                np.concatenate([
                    signed_groups[c]['roi_idx'][signed_groups[c]['positive_mask']],
                    signed_groups[c]['roi_idx'][signed_groups[c]['negative_mask']]
                ])).size if c < len(signed_groups) else pos_sizes[c])
        else:
            rep = members[0]

        # Build unions
        pos_union = []
        neg_union = []
        for g in signed_groups:
            if g['comp'] in members:
                pos_union.append(g['roi_idx'][g['positive_mask']])
                neg_union.append(g['roi_idx'][g['negative_mask']])
        pos_union = np.unique(np.concatenate(pos_union)) if pos_union else np.array([], int)
        neg_union = np.unique(np.concatenate(neg_union)) if neg_union else np.array([], int)

        # Pairwise Jaccard with representative (on chosen mode set)
        rep_set = comp_sets[rep]
        pw = {}
        for m in members:
            if m == rep: continue
            pw[m] = jaccard(rep_set, comp_sets[m])

        clusters.append(dict(
            members=sorted(members),
            representative=rep,
            pos_union=pos_union,
            neg_union=neg_union,
            pos_sizes={m: pos_sizes[m] for m in members},
            neg_sizes={m: neg_sizes[m] for m in members},
            pairwise_jaccard_with_rep=pw
        ))

    keep_components = [c['representative'] for c in clusters]
    # Preserve original ordering of representatives
    keep_components = [c for c in comps if c in keep_components]
    dropped = [c for c in comps if c not in keep_components]

    # Console summary
    print(f"[phase clustering] jaccard_thresh={jaccard_thresh} mode={mode}")
    for cl in clusters:
        mem = ",".join(map(str, cl['members']))
        print(f"  cluster rep={cl['representative']} members=[{mem}] "
              f"pos_union={cl['pos_union'].size} neg_union={cl['neg_union'].size}")
    print(f"  kept {len(keep_components)} reps; dropped {len(dropped)} redundant components")

    return dict(
        clusters=clusters,
        keep_components=keep_components,
        dropped_components=dropped,
        params=dict(jaccard_thresh=jaccard_thresh, mode=mode, representative=representative)
    )

def build_representative_phase_groups(signed_groups: List[Dict[str, Any]],
                                      clustering: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create a reduced signed_groups list using representative comps.
    Keeps original representative group's ROI/sign assignment (no union expansion),
    so downstream plots stay consistent. (You can adapt to use unions if desired.)
    """
    reps = set(clustering['keep_components'])
    reduced = [g for g in signed_groups if g['comp'] in reps]
    # Ensure deterministic ordering
    reduced = sorted(reduced, key=lambda g: g['comp'])
    print(f"[phase clustering] reduced signed groups: {len(signed_groups)} -> {len(reduced)}")
    return reduced

# ...existing code...











# ...existing code...

def plot_signed_components_bigstack(M: Dict[str, Any],
                                    res: Dict[str, Any],
                                    chunk: str,
                                    signed_groups: Optional[List[Dict[str, Any]]] = None,
                                    comps: Optional[List[int]] = None,
                                    events: Tuple[str, ...] = ('F1_on','F1_off','F2_on','F2_off','choice_start','lick_start'),
                                    smooth_sigma: float = 0.0,
                                    show_negative: bool = True,
                                    orient: bool = False,
                                    orient_methods: Tuple[str, ...] = ('trace_mean',),
                                    height_per_panel: float = 0.85,
                                    pos_color: str = 'tab:blue',
                                    neg_color: str = 'tab:red') -> plt.Figure:
    """
    Large “atlas” figure: 3 stacked panels per component (All / Short / Long)
    using the chunk's native (aligned) time axis.

    Use Cases
    ---------
    * Quick comparative inspection across components for a single alignment.
    * Identify condition selectivity (Short vs Long divergence) within chunk window.

    Parameters (additions to earlier docs)
    --------------------------------------
    res : dict
        CP result that contains res['factors']['A'] (NOT necessarily final_A).
    orient : bool
        If True re-orients provided signed_groups (use cautiously if already finalized).
    orient_methods : tuple[str]
        Sequence of methods tried in order; first to flip wins.

    Returns
    -------
    matplotib.figure.Figure
    """
    if chunk not in M['chunks']:
        print(f"[plot_signed_components_bigstack] chunk '{chunk}' missing.")
        return
    X = M['chunks'][chunk]['data']      # (trials, rois, time)
    t = M['chunks'][chunk]['time']
    is_short = M['is_short'].astype(bool)
    n_trials = X.shape[0]

    A = res['factors']['A']
    R = A.shape[1]

    if signed_groups is None:
        signed_groups = extract_signed_roi_groups(A, q=0.10)
        if orient:
            for m in orient_methods:
                orient_signed_groups(A, signed_groups, M=M, chunk=chunk, method=m)

    # Component filter
    if comps is None:
        comps = [g['comp'] for g in signed_groups if g['comp'] < R]
    else:
        comps = [c for c in comps if c < R]

    # Build quick access dict
    sg_map = {g['comp']: g for g in signed_groups}

    # Event alignment: chunk alignment event name
    alignment_event = M['chunks'][chunk].get('alignment', None)
    # Pre-compute event mean ± sd in chunk frame
    event_stats = {}
    if alignment_event and alignment_event in M:
        align_times = M[alignment_event]
        for ev in events:
            if ev not in M: continue
            ev_times = M[ev]
            dt = ev_times - align_times
            valid = np.isfinite(dt)
            if valid.any():
                mu = float(np.nanmean(dt[valid]))
                sd = float(np.nanstd(dt[valid]))
                event_stats[ev] = (mu, sd)

    from scipy.ndimage import gaussian_filter1d

    def mean_trace(trial_mask, roi_idx):
        if roi_idx.size == 0 or trial_mask.sum() == 0:
            return None
        sub = X[trial_mask][:, roi_idx, :]  # (trials_sel, rois_sel, time)
        mt = safe_nanmean(sub, axis=(0,1))
        if mt is not None and np.any(np.isfinite(mt)) and smooth_sigma:
            mt = gaussian_filter1d(mt, sigma=smooth_sigma)
        return mt

    nC = len(comps)
    if nC == 0:
        print("[plot_signed_components_bigstack] no components to plot.")
        return

    fig_h = nC * 3 * height_per_panel
    fig_w = 5.2
    fig, axes = plt.subplots(nC*3, 1, sharex=True, figsize=(fig_w, fig_h))
    if nC == 1:
        axes = np.array(axes).reshape(3,)

    for block_i, comp in enumerate(comps):
        g = sg_map.get(comp)
        if g is None:
            continue
        roi_idx = g['roi_idx']
        pos_idx = roi_idx[g['positive_mask']]
        neg_idx = roi_idx[g['negative_mask']]

        row_all   = block_i*3
        row_short = block_i*3 + 1
        row_long  = block_i*3 + 2
        ax_all, ax_short, ax_long = axes[row_all], axes[row_short], axes[row_long]

        all_mask = np.ones(n_trials, bool)
        short_mask = is_short
        long_mask = ~is_short

        p_all = mean_trace(all_mask,   pos_idx)
        p_short = mean_trace(short_mask, pos_idx)
        p_long = mean_trace(long_mask,  pos_idx)

        n_all = mean_trace(all_mask,   neg_idx) if show_negative else None
        n_short = mean_trace(short_mask, neg_idx) if show_negative else None
        n_long = mean_trace(long_mask,  neg_idx) if show_negative else None

        def draw(ax, pdat, ndat, title):
            if pdat is not None:
                ax.plot(t, pdat, color=pos_color, lw=1.2, label='Positive')
            if ndat is not None:
                ax.plot(t, ndat, color=neg_color, lw=1.0, ls='--', label='Negative')
            for ev,(mu,sd) in event_stats.items():
                ax.axvline(mu, color='k', lw=0.7)
                if np.isfinite(sd) and sd>0:
                    ax.axvspan(mu-sd, mu+sd, color='k', alpha=0.06)
            ax.axhline(0, color='k', lw=0.5, alpha=0.6)
            ax.set_ylabel("Mean", fontsize=7)
            ax.grid(alpha=0.2, linewidth=0.4)
            ax.set_title(title, fontsize=8)

        draw(ax_all,   p_all,   n_all,   f"comp {comp} | All")
        draw(ax_short, p_short, n_short, "Short")
        draw(ax_long,  p_long,  n_long,  "Long")

        if block_i == 0:  # single legend at top
            handles, labels = ax_all.get_legend_handles_labels()
            if handles:
                ax_all.legend(frameon=False, fontsize=7, ncol=2)

    axes[-1].set_xlabel("Time (s)", fontsize=8)
    plt.tight_layout()
    return fig

# ...existing code...














def plot_phase_components_on_full_trial(M: Dict[str, Any],
                                        signed_groups: List[Dict[str, Any]],
                                        comps: Optional[List[int]] = None,
                                        trial_chunk: str = 'trial_start',
                                        events: Tuple[str, ...] = ('F1_on','F1_off','F2_on','F2_off','choice_start','lick_start'),
                                        smooth_sigma: float = 0.8,
                                        show_negative: bool = True,
                                        height_per_panel: float = 0.9,
                                        pos_color: str = 'tab:blue',
                                        neg_color: str = 'tab:red') -> Optional[plt.Figure]:
    """
    Display phase component ROI groups over the global trial timeline.

    Difference vs plot_signed_component_mean_traces_stack
    -----------------------------------------------------
    * This function is intended for phase-derived ROI memberships
      (e.g. isi_phase), but works with any signed group list.
    * Axis is always trial_chunk timeline (not phase).

    Recommended
    -----------
    Use to assess whether phase-derived motifs extend into other trial epochs.

    """
    if trial_chunk not in M['chunks']:
        print(f"[plot_phase_components_on_full_trial] trial chunk '{trial_chunk}' missing.")
        return None

    # Trial_start data (global full-trial time base)
    ch = M['chunks'][trial_chunk]
    X_full = ch['data']            # (trials, rois, T_full)
    t_full = ch['time']
    n_trials, n_rois, T = X_full.shape

    is_short = M['is_short'].astype(bool)

    # Select components
    if comps is None:
        comps = sorted({g['comp'] for g in signed_groups})
    sg_map = {g['comp']: g for g in signed_groups if g['comp'] in comps}
    comps = [c for c in comps if c in sg_map]
    if not comps:
        print("[plot_phase_components_on_full_trial] No matching components.")
        return None

    # Event statistics on full trial axis (mean ± SD of raw event times relative to alignment of trial_chunk)
    align_evt = ch.get('alignment', None)
    event_stats = {}
    if align_evt and align_evt in M:
        align_times = M[align_evt]
        for ev in events:
            if ev not in M: 
                continue
            ev_times = M[ev]
            dt = ev_times - align_times
            valid = np.isfinite(dt)
            if valid.any():
                mu = float(np.nanmean(dt[valid]))
                sd = float(np.nanstd(dt[valid]))
                event_stats[ev] = (mu, sd)

    from scipy.ndimage import gaussian_filter1d

    def mean_trace(trial_mask: np.ndarray, roi_idx: np.ndarray) -> Optional[np.ndarray]:
        if roi_idx.size == 0 or trial_mask.sum() == 0:
            return None
        sub = X_full[trial_mask][:, roi_idx, :]       # (sel_trials, sel_rois, T)
        # Mean across trials, then across ROIs (same as mean across both dims)
        # First compute per-trial ROI mean for possible future SEM (not needed here -> direct collapse)
        mt = safe_nanmean(sub, axis=(0,1))            # (T,)
        if mt is not None and np.any(np.isfinite(mt)) and smooth_sigma and smooth_sigma > 0:
            mt = gaussian_filter1d(mt, sigma=smooth_sigma)
        return mt

    nC = len(comps)
    fig_h = nC * 3 * height_per_panel
    fig_w = 6.0
    fig, axes = plt.subplots(nC * 3, 1, sharex=True, figsize=(fig_w, fig_h))
    if nC == 1:
        axes = np.array(axes).reshape(3,)

    for block_i, c in enumerate(comps):
        g = sg_map[c]
        roi_idx = g['roi_idx']
        pos_idx = roi_idx[g['positive_mask']]
        neg_idx = roi_idx[g['negative_mask']]

        row_all   = block_i * 3
        row_short = block_i * 3 + 1
        row_long  = block_i * 3 + 2
        ax_all, ax_short, ax_long = axes[row_all], axes[row_short], axes[row_long]

        all_mask   = np.ones(n_trials, bool)
        short_mask = is_short
        long_mask  = ~is_short

        p_all   = mean_trace(all_mask,   pos_idx)
        p_short = mean_trace(short_mask, pos_idx)
        p_long  = mean_trace(long_mask,  pos_idx)

        n_all   = mean_trace(all_mask,   neg_idx) if show_negative else None
        n_short = mean_trace(short_mask, neg_idx) if show_negative else None
        n_long  = mean_trace(long_mask,  neg_idx) if show_negative else None

        def draw(ax, pdat, ndat, title):
            if pdat is not None:
                ax.plot(t_full, pdat, color=pos_color, lw=1.2, label='Positive')
            if ndat is not None:
                ax.plot(t_full, ndat, color=neg_color, lw=1.0, ls='--', label='Negative')
            # Events
            for ev, (mu, sd) in event_stats.items():
                ax.axvline(mu, color='k', lw=0.75)
                if np.isfinite(sd) and sd > 0:
                    ax.axvspan(mu - sd, mu + sd, color='k', alpha=0.05)
            ax.axhline(0, color='k', lw=0.5, alpha=0.6)
            ax.set_ylabel("Mean", fontsize=7)
            ax.set_title(title, fontsize=8)
            ax.grid(alpha=0.2, linewidth=0.4)

        draw(ax_all,   p_all,   n_all,   f"comp {c} | All")
        draw(ax_short, p_short, n_short, "Short")
        draw(ax_long,  p_long,  n_long,  "Long")

        if block_i == 0:
            h, l = ax_all.get_legend_handles_labels()
            if h:
                ax_all.legend(frameon=False, fontsize=7, ncol=2)

    axes[-1].set_xlabel("Time (s)", fontsize=8)
    plt.tight_layout()
    return fig

# ...existing code...


















# ...existing code...

def plot_signed_component_mean_traces_stack(M: Dict[str, Any],
                                            signed_groups: List[Dict[str, Any]],
                                            trial_chunk: str = 'trial_start',
                                            comps: Optional[List[int]] = None,
                                            events: Tuple[str, ...] = ('F1_on','F1_off','F2_on','F2_off','choice_start','lick_start'),
                                            smooth_sigma: float = 0.8,
                                            pos_color: str = 'tab:blue',
                                            neg_color: str = 'tab:red',
                                            height_per_panel: float = 0.9,
                                            show_sem: bool = False) -> Optional[plt.Figure]:
    """
    Per-component stacked POS/NEG mean traces on the full trial timeline.

    Layout per component
    --------------------
      Row 1: All (events: short & long)
      Row 2: Short only
      Row 3: Long only

    Parameters
    ----------
    signed_groups : list[dict]
        ROI membership (assumed oriented).
    trial_chunk : str
        Chunk providing global time axis (usually 'trial_start').
    smooth_sigma : float
        Gaussian σ (samples) for temporal smoothing (0 disables).
    show_sem : bool
        If True fill ±SEM bands per subset (across trials).

    Notes
    -----
    Use when you want isolated inspection of each component pattern without
    alpha crowding (contrast to overlay view).
    """
    if trial_chunk not in M['chunks']:
        print(f"[plot_signed_component_mean_traces_stack] chunk '{trial_chunk}' missing.")
        return None

    # Data
    ch = M['chunks'][trial_chunk]
    X = ch['data']              # (trials, rois, T)
    t = ch['time']
    n_trials = X.shape[0]
    is_short = M['is_short'].astype(bool)

    # Component selection
    if comps is None:
        comps = sorted({g['comp'] for g in signed_groups})
    sg_map = {g['comp']: g for g in signed_groups if g['comp'] in comps}
    comps = [c for c in comps if c in sg_map]
    if not comps:
        print("[plot_signed_component_mean_traces_stack] no components selected.")
        return None

    # Event stats (condition-specific)
    def _event_stats(ev):
        if ev not in M: return (np.nan, np.nan), (np.nan, np.nan)
        arr = M[ev]
        mu_s, sd_s = _event_stats_condition(arr, is_short)
        mu_l, sd_l = _event_stats_condition(arr, ~is_short)
        return (mu_s, sd_s), (mu_l, sd_l)

    event_stats_short = {}
    event_stats_long  = {}
    for ev in events:
        (mu_s, sd_s), (mu_l, sd_l) = _event_stats(ev)
        event_stats_short[ev] = (mu_s, sd_s)
        event_stats_long[ev]  = (mu_l, sd_l)

    from scipy.ndimage import gaussian_filter1d

    def _cond_mean_trace(roi_idx: np.ndarray, trial_mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (mean_trace, sem_trace) for given ROI subset & trial mask.
        mean over ROIs per trial, then over trials; SEM across trials.
        """
        if roi_idx.size == 0 or trial_mask.sum() == 0:
            return None, None
        sub = X[trial_mask][:, roi_idx, :]  # (n_trials_sel, n_rois_sel, T)
        # Mean across ROIs per trial
        per_trial = safe_nanmean(sub, axis=1)  # (trials_sel, T)
        mt = safe_nanmean(per_trial, axis=0)   # (T,)
        if mt is not None and np.any(np.isfinite(mt)) and smooth_sigma > 0:
            mt = gaussian_filter1d(mt, sigma=smooth_sigma)
        if not show_sem:
            return mt, None
        # SEM
        with np.errstate(invalid='ignore'):
            sd = np.nanstd(per_trial, axis=0, ddof=1)
        n_eff = np.sum(np.isfinite(per_trial), axis=0)
        sem = sd / np.sqrt(np.maximum(n_eff, 1))
        if smooth_sigma > 0 and sem is not None:
            sem = gaussian_filter1d(sem, sigma=smooth_sigma)
        return mt, sem

    nC = len(comps)
    fig_h = nC * 3 * height_per_panel
    fig_w = 6.4
    fig, axes = plt.subplots(nC * 3, 1, sharex=True, figsize=(fig_w, fig_h))
    if nC == 1:
        axes = np.array(axes).reshape(3,)

    for block_i, comp in enumerate(comps):
        g = sg_map[comp]
        roi_idx = g['roi_idx']
        pos_idx = roi_idx[g['positive_mask']]
        neg_idx = roi_idx[g['negative_mask']]

        rows = slice(block_i * 3, block_i * 3 + 3)
        ax_all, ax_short, ax_long = axes[rows]

        masks = dict(all=np.ones(n_trials, bool),
                     short=is_short,
                     long=~is_short)

        # Compute traces
        traces = {}
        for key, m in masks.items():
            p_mean, p_sem = _cond_mean_trace(pos_idx, m)
            n_mean, n_sem = _cond_mean_trace(neg_idx, m)
            traces[key] = dict(pos=p_mean, pos_sem=p_sem, neg=n_mean, neg_sem=n_sem)

        def _draw_events_combined(ax):
            for ev in events:
                mu_s, sd_s = event_stats_short.get(ev, (np.nan, np.nan))
                mu_l, sd_l = event_stats_long.get(ev, (np.nan, np.nan))
                col = {'F1_on':'#1f77b4','F1_off':'#1f77b4',
                       'F2_on':'#ff7f0e','F2_off':'#ff7f0e',
                       'choice_start':'#2ca02c','lick_start':'#d62728'}.get(ev, 'k')
                if np.isfinite(mu_s):
                    ax.axvline(mu_s, color=col, lw=0.9)
                    if np.isfinite(sd_s) and sd_s > 0:
                        ax.axvspan(mu_s - sd_s, mu_s + sd_s, color=col, alpha=0.06)
                if np.isfinite(mu_l):
                    ax.axvline(mu_l, color=col, lw=0.9, ls='--')
                    if np.isfinite(sd_l) and sd_l > 0:
                        ax.axvspan(mu_l - sd_l, mu_l + sd_l, color=col, alpha=0.05)

        def _draw_events_single(ax, mode):
            stats = event_stats_short if mode == 'short' else event_stats_long
            for ev, (mu, sd) in stats.items():
                if not np.isfinite(mu):
                    continue
                col = {'F1_on':'#1f77b4','F1_off':'#1f77b4',
                       'F2_on':'#ff7f0e','F2_off':'#ff7f0e',
                       'choice_start':'#2ca02c','lick_start':'#d62728'}.get(ev, 'k')
                ax.axvline(mu, color=col, lw=0.9)
                if np.isfinite(sd) and sd > 0:
                    ax.axvspan(mu - sd, mu + sd, color=col, alpha=0.06)

        def _plot_panel(ax, data, title, ev_mode):
            p = data['pos']; n = data['neg']
            if p is not None:
                ax.plot(t, p, color=pos_color, lw=1.3, label='Positive')
                if show_sem and data['pos_sem'] is not None:
                    sem = data['pos_sem']
                    ax.fill_between(t, p - sem, p + sem, color=pos_color, alpha=0.18, linewidth=0)
            if n is not None:
                ax.plot(t, n, color=neg_color, lw=1.1, ls='--', label='Negative')
                if show_sem and data['neg_sem'] is not None:
                    sem = data['neg_sem']
                    ax.fill_between(t, n - sem, n + sem, color=neg_color, alpha=0.15, linewidth=0)
            if ev_mode == 'combined':
                _draw_events_combined(ax)
            else:
                _draw_events_single(ax, ev_mode)
            ax.axhline(0, color='k', lw=0.5, alpha=0.6)
            ax.set_ylabel("Mean", fontsize=7)
            ax.set_title(title, fontsize=8)
            ax.grid(alpha=0.2, linewidth=0.4)

        _plot_panel(ax_all,   traces['all'],   f"comp {comp} | All",   'combined')
        _plot_panel(ax_short, traces['short'], "Short",                'short')
        _plot_panel(ax_long,  traces['long'],  "Long",                 'long')

        if block_i == 0:
            h, l = ax_all.get_legend_handles_labels()
            if h:
                ax_all.legend(frameon=False, fontsize=7, ncol=2)

    axes[-1].set_xlabel("Time (s)", fontsize=8)
    plt.tight_layout()
    return fig
# ...existing code...




















# ...existing code...

def plot_signed_components_overlay(M: Dict[str, Any],
                                   signed_groups: List[Dict[str, Any]],
                                   trial_chunk: str = 'trial_start',
                                   comps: Optional[List[int]] = None,
                                   events: Tuple[str, ...] = ('F1_on','F1_off','F2_on','F2_off','choice_start','lick_start'),
                                   smooth_sigma: float = 0.8,
                                   energy_mode: str = 'combined',      # 'combined' | 'pos_all' | 'diff_only'
                                   min_alpha: float = 0.25,
                                   max_alpha: float = 0.95,
                                   sort_by_energy: bool = True,
                                   show_negative: bool = True,
                                   show_sem: bool = False,
                                   sem_alpha: float = 0.12,
                                   annotate: bool = True,
                                   annotate_mode: str = 'end',          # 'end' | 'peak'
                                   annotate_top_n: Optional[int] = None,
                                   pos_color: str = 'tab:blue',
                                   neg_color: str = 'tab:red',
                                   figsize: Tuple[float,float] = (7.2, 5.0),
                                   chunk_name: Optional[str] = None,
                                   energy_flat_tol: float = 1e-6,        # NEW: tolerance for “flat” energies
                                   flat_energy_alpha: str = 'max',       # 'max' | 'mid' | 'min'
                                   debug_energies: bool = False          # NEW                                   
                                   ) -> Optional[plt.Figure]:
    """
    Overlay multi-component POS/NEG mean traces in three panels (All / Short / Long).

    Intent
    ------
    Provide a compact “component motif landscape” where prominence of each
    component is conveyed by alpha (transparency) rather than color diversity.

    Panels
    -------
    1. All trials   : events for short (solid) & long (dashed) with SD shading
    2. Short trials : short-only events
    3. Long trials  : long-only events

    Parameters (Key)
    ----------------
    signed_groups : list[dict]
        ROI membership for each component (or pruned list).
    comps : list[int] | None
        Subset of component indices; None -> all in signed_groups.
    energy_mode : str
        'combined' : sum (pos - neg)^2 (default; contrast-aware)
        'pos_all'  : sum pos^2 (ignore negative)
        'diff_only': alias of 'combined'
    min_alpha / max_alpha : float
        Range for alpha scaling.
    energy_flat_tol : float
        If (maxE - minE) < tol treat energies as “flat”.
    flat_energy_alpha : str
        'max' | 'mid' | 'min' mapping all components when energy range flat.
    annotate : bool
        Add component IDs on All panel (at end or peak).
    annotate_mode : str
        'end' or 'peak' for text position.
    annotate_top_n : int | None
        Limit number of labels (highest energy first).
    show_sem : bool
        Currently reserved (SEM shading not implemented in overlay style).

    Returns
    -------
    fig | None

    Quick Use
    ---------
    plot_signed_components_overlay(M, signed_pruned['isi'],
                                   chunk_name='isi', debug_energies=True)

    Interpretation
    --------------
    * Darker solid blue = stronger positive motif (under chosen energy_mode).
    * Red dashed mirrors negative subset (if present).
    * If most lines very light: either truly low energy or you may want
      'energy_mode="pos_all"' / adjust min_alpha.

    """
    if trial_chunk not in M['chunks']:
        print(f"[plot_signed_components_overlay] chunk '{trial_chunk}' missing.")
        return None
    ch = M['chunks'][trial_chunk]
    X = ch['data']            # (trials, rois, T)
    t = ch['time']
    is_short = M['is_short'].astype(bool)
    n_trials = X.shape[0]

    # Select components
    if comps is None:
        comps = sorted({g['comp'] for g in signed_groups})
    sg_map = {g['comp']: g for g in signed_groups if g['comp'] in comps}
    comps = [c for c in comps if c in sg_map]
    if not comps:
        print("[plot_signed_components_overlay] no components selected.")
        return None

    from scipy.ndimage import gaussian_filter1d

    def mean_trace(roi_idx: np.ndarray, trial_mask: np.ndarray) -> Optional[np.ndarray]:
        if roi_idx.size == 0 or trial_mask.sum() == 0:
            return None
        sub = X[trial_mask][:, roi_idx, :]
        mt = safe_nanmean(sub, axis=(0,1))
        if mt is not None and np.any(np.isfinite(mt)) and smooth_sigma > 0:
            mt = gaussian_filter1d(mt, sigma=smooth_sigma)
        return mt

    # Event stats (condition-specific)
    event_stats_short = {}
    event_stats_long  = {}
    for ev in events:
        if ev in M:
            mu_s, sd_s = _event_stats_condition(M[ev], is_short)
            mu_l, sd_l = _event_stats_condition(M[ev], ~is_short)
            event_stats_short[ev] = (mu_s, sd_s)
            event_stats_long[ev]  = (mu_l, sd_l)

    # Gather traces & energies
    records = []
    all_mask = np.ones(n_trials, bool)
    short_mask = is_short
    long_mask = ~is_short

    for c in comps:
        g = sg_map[c]
        roi_idx = g['roi_idx']
        pos_idx = roi_idx[g['positive_mask']]
        neg_idx = roi_idx[g['negative_mask']]

        pos_all = mean_trace(pos_idx, all_mask)
        pos_short = mean_trace(pos_idx, short_mask)
        pos_long = mean_trace(pos_idx, long_mask)
        neg_all = mean_trace(neg_idx, all_mask) if show_negative else None
        neg_short = mean_trace(neg_idx, short_mask) if show_negative else None
        neg_long = mean_trace(neg_idx, long_mask) if show_negative else None

        # Energy metric
        if energy_mode in ('combined', 'diff_only'):
            if pos_all is not None and neg_all is not None:
                diff = pos_all - neg_all
                energy = float(np.nansum(diff**2))
            elif pos_all is not None:
                energy = float(np.nansum(pos_all**2))
            else:
                energy = 0.0
        elif energy_mode == 'pos_all':
            energy = float(np.nansum(pos_all**2)) if pos_all is not None else 0.0
        else:
            raise ValueError("energy_mode must be combined|diff_only|pos_all")

        records.append(dict(
            comp=c,
            pos_all=pos_all, pos_short=pos_short, pos_long=pos_long,
            neg_all=neg_all, neg_short=neg_short, neg_long=neg_long,
            energy=energy
        ))

    if not records:
        print("[plot_signed_components_overlay] no usable traces.")
        return None

    # energies = np.array([r['energy'] for r in records])
    # if np.allclose(energies, 0):
    #     normE = np.ones_like(energies)
    # else:
    #     normE = (energies - energies.min()) / (energies.max() - energies.min() + 1e-12)
    # for r, ne in zip(records, normE):
    #     r['alpha'] = min_alpha + (max_alpha - min_alpha) * ne
    energies = np.array([r['energy'] for r in records], float)
    rng = energies.max() - energies.min()
    if rng < energy_flat_tol:
        # Flat energy case
        if flat_energy_alpha == 'max':
            assigned = np.full_like(energies, max_alpha)
        elif flat_energy_alpha == 'mid':
            assigned = np.full_like(energies, 0.5*(min_alpha+max_alpha))
        else:  # 'min'
            assigned = np.full_like(energies, min_alpha)
        for r, a in zip(records, assigned):
            r['alpha'] = float(a)
    else:
        normE = (energies - energies.min()) / (rng + 1e-12)
        for r, ne in zip(records, normE):
            r['alpha'] = float(min_alpha + (max_alpha - min_alpha) * ne)

    if debug_energies:
        print("[overlay energies]")
        for r in records:
            print(f" comp {r['comp']:2d} energy={r['energy']:.4g} alpha={r['alpha']:.3f}")

    if sort_by_energy:
        records.sort(key=lambda d: d['energy'])  # low first

    if sort_by_energy:
        records.sort(key=lambda d: d['energy'])  # low first, high last (plot on top)

    # Figure
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize)
    ax_all, ax_short, ax_long = axes

    def draw_events_combined(ax):
        for ev in events:
            mu_s, sd_s = event_stats_short.get(ev, (np.nan, np.nan))
            mu_l, sd_l = event_stats_long.get(ev, (np.nan, np.nan))
            col = {'F1_on':'#1f77b4','F1_off':'#1f77b4',
                   'F2_on':'#ff7f0e','F2_off':'#ff7f0e',
                   'choice_start':'#2ca02c','lick_start':'#d62728'}.get(ev, 'k')
            if np.isfinite(mu_s):
                ax.axvline(mu_s, color=col, lw=0.8)
                if np.isfinite(sd_s) and sd_s > 0:
                    ax.axvspan(mu_s - sd_s, mu_s + sd_s, color=col, alpha=0.05)
            if np.isfinite(mu_l):
                ax.axvline(mu_l, color=col, lw=0.8, ls='--')
                if np.isfinite(sd_l) and sd_l > 0:
                    ax.axvspan(mu_l - sd_l, mu_l + sd_l, color=col, alpha=0.04)

    def draw_events_single(ax, mode):
        stats = event_stats_short if mode == 'short' else event_stats_long
        for ev, (mu, sd) in stats.items():
            if not np.isfinite(mu): continue
            col = {'F1_on':'#1f77b4','F1_off':'#1f77b4',
                   'F2_on':'#ff7f0e','F2_off':'#ff7f0e',
                   'choice_start':'#2ca02c','lick_start':'#d62728'}.get(ev, 'k')
            ax.axvline(mu, color=col, lw=0.8)
            if np.isfinite(sd) and sd > 0:
                ax.axvspan(mu - sd, mu + sd, color=col, alpha=0.05)

    # Plot lines
    print(chunk_name)
    for r in records:
        print(r['comp'], r['energy'], r['alpha'])
        a = r['alpha']
        # All
        if r['pos_all'] is not None:
            ax_all.plot(t, r['pos_all'], color=pos_color, alpha=a, lw=1.3)
        if show_negative and r['neg_all'] is not None:
            ax_all.plot(t, r['neg_all'], color=neg_color, alpha=a, lw=1.0, ls='--')
        # Short
        if r['pos_short'] is not None:
            ax_short.plot(t, r['pos_short'], color=pos_color, alpha=a, lw=1.2)
        if show_negative and r['neg_short'] is not None:
            ax_short.plot(t, r['neg_short'], color=neg_color, alpha=a, lw=1.0, ls='--')
        # Long
        if r['pos_long'] is not None:
            ax_long.plot(t, r['pos_long'], color=pos_color, alpha=a, lw=1.2)
        if show_negative and r['neg_long'] is not None:
            ax_long.plot(t, r['neg_long'], color=neg_color, alpha=a, lw=1.0, ls='--')

    # Event markers
    draw_events_combined(ax_all)
    draw_events_single(ax_short, 'short')
    draw_events_single(ax_long, 'long')

    for ax, title in zip(axes, ["All", "Short", "Long"]):
        ax.axhline(0, color='k', lw=0.5, alpha=0.6)
        ax.set_ylabel("Mean", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.4)

    axes[-1].set_xlabel("Time (s)", fontsize=9)

    # Annotation (component IDs) for clarity
    if annotate:
        # Decide ordering for annotation (highest energy first)
        ordered = sorted(records, key=lambda d: d['energy'], reverse=True)
        if annotate_top_n is not None:
            ordered = ordered[:annotate_top_n]
        for r in ordered:
            trace = r['pos_all'] if r['pos_all'] is not None else r['neg_all']
            if trace is None:
                continue
            if annotate_mode == 'peak':
                idx = int(np.nanargmax(np.abs(trace)))
            else:  # end
                idx = len(t) - 1
            x = t[idx]
            y = trace[idx]
            ax_all.text(x, y, f"{r['comp']}", fontsize=7, ha='left', va='center',
                        color='k', alpha=0.8)

    # Legend (component energy table)
    energies_sorted = sorted([(r['comp'], r['energy'], r['alpha']) for r in records],
                             key=lambda x: x[1], reverse=True)
    legend_text = "Comp (energy, α): " + "  ".join(
        f"{c}({e:.2g},{a:.2f})" for c, e, a in energies_sorted[:12]
    )
    axes[0].text(0.01, 0.98, legend_text, transform=axes[0].transAxes,
                 ha='left', va='top', fontsize=7, color='k',
                 bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=2))

    fig = plt.gcf()
    fig.suptitle(f"Overlay signed component traces (N={len(records)})  energy_mode={energy_mode}", fontsize=10, y=0.995)
    fig = plt.gcf()
    title_chunk = chunk_name if chunk_name is not None else trial_chunk
    fig.suptitle(f"{title_chunk} | overlay signed component traces (N={len(records)})  energy_mode={energy_mode}",
                 fontsize=10, y=0.995)
    plt.tight_layout(rect=[0,0,1,0.985])
    return fig
    plt.tight_layout(rect=[0,0,1,0.985])
    return fig

# ...existing code...









def build_signed_groups_for_chunks(M: Dict[str, Any],
                                   chunk_results: Dict[str, Dict[str, Any]],
                                   q: float = 0.10,
                                   use_final: bool = True,
                                   orient: bool = True,
                                   orient_method: str = 'trace_mean') -> Dict[str, List[Dict[str, Any]]]:
    """
    Batch create signed ROI groups for every chunk result.

    Parameters
    ----------
    M : dict
        Session dictionary.
    chunk_results : dict
        Mapping chunk -> CP result (must have either:
          * 'final_A'  (preferred)  OR
          * 'base']['factors']['A'] (fallback)
        Typically pass time_results plus optionally {'isi_phase': {'final_A': ...}}.
    q : float
        Quantile for absolute loading selection (see extract_signed_roi_groups).
    use_final : bool
        If True use res['final_A']; else use res['base']['factors']['A'].
    orient : bool
        If True apply orient_signed_groups + finalize_signed_orientation.
    orient_method : str
        Method argument forwarded to orient_signed_groups.

    Returns
    -------
    signed_by_chunk : dict
        chunk -> list[ signed_group dicts ].

    Notes
    -----
    * Safe to run again; will regenerate groups reflecting latest A matrices.
    * For phase chunk you may wish to re-run orientation with 'trace_phase'
      then 'trace_mean' sequentially (already done in run_phase_analysis).
    """
    out = {}
    for chunk, res in chunk_results.items():
        try:
            if use_final and 'final_A' in res:
                A = res['final_A']
            else:
                A = res['base']['factors']['A']
            if A.size == 0:
                continue
            signed = extract_signed_roi_groups(A, q=q)
            if orient:
                orient_signed_groups(A, signed, M=M, chunk=chunk, method=orient_method)
                finalize_signed_orientation(M, {'factors': {'A': A}}, signed, chunk)
            out[chunk] = signed
        except Exception as e:
            print(f"[build_signed_groups_for_chunks] chunk {chunk} failed: {e}")
    return out

def integrate_signed_groups_into_catalog(catalog: Dict[str, Any],
                                         M: Dict[str, Any],
                                         signed_groups_by_chunk: Dict[str, List[Dict[str, Any]]],
                                         label_prefix: str = '',
                                         skip_existing: bool = True) -> Dict[str, Any]:
    """
    Augment catalog incidence matrix with separate positive / negative columns
    for each signed component.

    New Labels
    ----------
    Format: "<prefix><chunk>:comp<k>_pos" and "..._neg"

    Parameters
    ----------
    catalog : dict
        Must contain 'incidence' (n_rois, n_components) and 'comp_labels'.
    M : dict
        (Unused now—placeholder for future filtering by ROI attributes).
    signed_groups_by_chunk : dict
        chunk -> list of signed groups.
    label_prefix : str
        Optional string prepended to every new label (e.g. "signed:").
    skip_existing : bool
        If True, labels already in catalog['comp_labels'] will not be duplicated.

    Returns
    -------
    catalog (modified in-place)

    Caveats
    -------
    * Negative subset may be empty; then no _neg column is added.
    * Incidence columns are appended (ordering preserved within each chunk).
    """
    if catalog is None or 'incidence' not in catalog:
        raise ValueError("Catalog missing incidence matrix.")
    inc = catalog['incidence']
    labels = catalog['comp_labels']
    n_rois = inc.shape[0]
    new_cols = []
    new_labels = []
    for chunk, groups in signed_groups_by_chunk.items():
        for g in groups:
            comp = g['comp']
            roi_idx = g['roi_idx']
            pos_idx = roi_idx[g['positive_mask']]
            neg_idx = roi_idx[g['negative_mask']]
            base = f"{chunk}:comp{comp}"
            if label_prefix:
                base = f"{label_prefix}{base}"
            if pos_idx.size:
                lbl = f"{base}_pos"
                if not (skip_existing and lbl in labels):
                    col = np.zeros(n_rois, bool); col[pos_idx] = True
                    new_cols.append(col); new_labels.append(lbl)
            if neg_idx.size:
                lbl = f"{base}_neg"
                if not (skip_existing and lbl in labels):
                    col = np.zeros(n_rois, bool); col[neg_idx] = True
                    new_cols.append(col); new_labels.append(lbl)
    if new_cols:
        catalog['incidence'] = np.column_stack([inc] + new_cols)
        catalog['comp_labels'].extend(new_labels)
        print(f"[integrate_signed_groups_into_catalog] added {len(new_labels)} signed columns.")
    else:
        print("[integrate_signed_groups_into_catalog] nothing added.")
    return catalog

def summarize_signed_group_overlap(signed_groups_by_chunk: Dict[str, List[Dict[str, Any]]],
                                    mode: str = 'pos',
                                    min_jaccard: float = 0.5) -> None:
    """
    Print pairwise Jaccard overlaps (>= threshold) across ALL signed groups over chunks.

    Parameters
    ----------
    signed_groups_by_chunk : dict
        chunk -> list of signed group dicts.
    mode : str
        'pos'  : use positive subset only
        'neg'  : use negative subset only
        'both' : union of pos and neg
    min_jaccard : float
        Report only pairs with Jaccard >= this value.

    Output
    ------
    Console lines: "<chunkA>:c<i>_pos vs <chunkB>:c<j>_pos : J=..."

    Uses
    ----
    * Identify redundant anatomical / functional motifs across alignments.
    * Guide clustering or pruning of overlapping groups.
    """
    sets = []
    names = []
    for chunk, groups in signed_groups_by_chunk.items():
        for g in groups:
            roi_idx = g['roi_idx']
            if mode == 'pos':
                idx = roi_idx[g['positive_mask']]
                tag = f"{chunk}:c{g['comp']}_pos"
            elif mode == 'neg':
                idx = roi_idx[g['negative_mask']]
                tag = f"{chunk}:c{g['comp']}_neg"
            else:
                idx = np.unique(np.concatenate([
                    roi_idx[g['positive_mask']], roi_idx[g['negative_mask']]
                ]))
                tag = f"{chunk}:c{g['comp']}_both"
            if idx.size:
                sets.append(np.sort(idx))
                names.append(tag)
    n = len(sets)
    if n == 0:
        print("[summarize_signed_group_overlap] no sets.")
        return
    print(f"[signed overlap] {n} sets (mode={mode})")
    for i in range(n):
        Si = sets[i]
        for j in range(i+1, n):
            Sj = sets[j]
            inter = np.intersect1d(Si, Sj).size
            if inter == 0:
                continue
            union = np.union1d(Si, Sj).size
            J = inter / union if union else 0.0
            if J >= min_jaccard:
                print(f"  {names[i]} vs {names[j]}: J={J:.3f} (|∩|={inter}, |∪|={union})")

# ...existing code...






















def assess_signed_groups(A: np.ndarray,
                         signed_groups: List[Dict[str, Any]],
                         min_neg_frac: float = 0.10,
                         min_neg_rel_mag: float = 0.25,
                         print_table: bool = True,
                         metric: str = 'loading',
                         M: Optional[Dict[str, Any]] = None,
                         chunk: Optional[str] = None,
                         trace_reduce: str = 'mean',
                         use_abs_trace: bool = True) -> pd.DataFrame:
    """
    Quantify whether the NEGATIVE side of each signed component is meaningful.

    Metrics
    -------
    loading       : mean(|loading|) for each side
    trace_mean    : trial-average ROI traces -> mean across time (abs or signed)
    trace_energy  : sum of squares of trial-average ROI trace (energy proxy)

    Parameters
    ----------
    A : (n_rois, n_components) array
        Loading matrix (used for metric='loading').
    signed_groups : list[dict]
        From extract_signed_roi_groups (optionally oriented).
    min_neg_frac : float
        Minimum fraction (n_neg / (n_pos+n_neg)) to consider “non-trivial”.
    min_neg_rel_mag : float
        Minimum relative magnitude (neg_mag / pos_mag) required.
    print_table : bool
        If True prints concise table.
    metric : str
        One of {'loading','trace_mean','trace_energy'}.
    M : dict
        Required for trace_* metrics (provides chunk data).
    chunk : str
        Chunk key for trace metrics.
    trace_reduce : str
        If use_abs_trace=False and metric='trace_mean':
          'mean' (default), 'max' (peak), else treated as 'mean'.
    use_abs_trace : bool
        If True (default) magnitude uses mean(abs(trace)), else signed mean.

    Returns
    -------
    DataFrame columns:
      comp, n_pos, n_neg, frac_neg, pos_mag, neg_mag, rel_mag, weak (bool)

    Weak Rule
    ---------
    weak = (frac_neg < min_neg_frac) OR (rel_mag < min_neg_rel_mag)

    Typical Workflow
    ----------------
    df = assess_signed_groups(..., metric='loading')
    df2 = assess_signed_groups(..., metric='trace_mean', M=M, chunk='choice_start')
    """
    if metric.startswith('trace'):
        if M is None or chunk is None:
            raise ValueError("metric trace_* requires M and chunk.")
        if chunk not in M['chunks']:
            raise KeyError(f"Chunk '{chunk}' not in M.")
        X = M['chunks'][chunk]['data']  # (trials, rois, time/phase)
    else:
        X = None

    rows = []
    for g in signed_groups:
        c = g['comp']
        idx = g['roi_idx']
        pos = idx[g['positive_mask']]
        neg = idx[g['negative_mask']]
        n_pos = pos.size
        n_neg = neg.size
        tot = n_pos + n_neg
        if metric == 'loading':
            v = A[:, c]
            pos_mag = np.nanmean(np.abs(v[pos])) if n_pos else 0.0
            neg_mag = np.nanmean(np.abs(v[neg])) if n_neg else 0.0
        else:
            # Build per-ROI mean trace across trials
            if n_pos:
                pos_trace = safe_nanmean(X[:, pos], axis=0)  # (time,)
                if metric == 'trace_energy':
                    pos_mag = float(np.nansum(np.square(pos_trace)))
                else:
                    if use_abs_trace:
                        pos_mag = float(np.nanmean(np.abs(pos_trace)))
                    else:
                        if trace_reduce == 'mean':
                            pos_mag = float(np.nanmean(pos_trace))
                        elif trace_reduce == 'max':
                            pos_mag = float(np.nanmax(pos_trace))
                        else:
                            pos_mag = float(np.nanmean(pos_trace))
            else:
                pos_mag = 0.0
            if n_neg:
                neg_trace = safe_nanmean(X[:, neg], axis=0)
                if metric == 'trace_energy':
                    neg_mag = float(np.nansum(np.square(neg_trace)))
                else:
                    if use_abs_trace:
                        neg_mag = float(np.nanmean(np.abs(neg_trace)))
                    else:
                        if trace_reduce == 'mean':
                            neg_mag = float(np.nanmean(neg_trace))
                        elif trace_reduce == 'max':
                            neg_mag = float(np.nanmax(neg_trace))
                        else:
                            neg_mag = float(np.nanmean(neg_trace))
            else:
                neg_mag = 0.0

        frac_neg = n_neg / tot if tot else 0.0
        rel_mag = neg_mag / (pos_mag + 1e-12)
        weak = (frac_neg < min_neg_frac) or (rel_mag < min_neg_rel_mag)
        rows.append(dict(comp=c, n_pos=n_pos, n_neg=n_neg,
                         frac_neg=frac_neg, pos_mag=pos_mag,
                         neg_mag=neg_mag, rel_mag=rel_mag, weak=weak))
    df = pd.DataFrame(rows).sort_values('comp')
    if print_table:
        print(f"[assess_signed_groups] metric={metric} chunk={chunk if metric.startswith('trace') else 'N/A'}")
        print("comp  n_pos  n_neg  frac_neg  rel_mag  weak")
        for r in df.itertuples():
            print(f"{r.comp:4d} {r.n_pos:6d} {r.n_neg:6d}  {r.frac_neg:7.3f}  {r.rel_mag:7.3f}  {r.weak}")
        print(f"Weak negatives: {df['weak'].sum()}/{len(df)}")
    return df
# ...existing code...




def prune_weak_negatives(A: np.ndarray,
                         signed_groups: List[Dict[str, Any]],
                         min_neg_frac: float = 0.10,
                         min_neg_rel_mag: float = 0.25,
                         drop_entire: bool = False) -> List[Dict[str, Any]]:
    """
    Simplify signed groups by removing weak negative subsets.

    Parameters
    ----------
    A : (n_rois, n_components)
        Loading matrix (passed to assess_signed_groups with metric='loading').
    signed_groups : list[dict]
        Modified in place (negative_mask zeroed or group dropped).
    min_neg_frac, min_neg_rel_mag : float
        Thresholds forwarded to assess_signed_groups.
    drop_entire : bool
        False (default): keep component but clear negative_mask -> positive-only.
        True: remove whole component entry from list.

    Returns
    -------
    signed_groups (modified reference) for chaining.

    Recommendation
    --------------
    * Run with drop_entire=False first (inspect effect on plots).
    * If many components become trivial singletons consider adjusting q or ranks.
    """
    df = assess_signed_groups(A, signed_groups,
                              min_neg_frac=min_neg_frac,
                              min_neg_rel_mag=min_neg_rel_mag,
                              print_table=False)
    weak_set = set(df.loc[df.weak, 'comp'])
    kept = []
    for g in signed_groups:
        if g['comp'] in weak_set:
            if drop_entire:
                continue
            # keep component, clear negative side
            g['negative_mask'][:] = False
        kept.append(g)
    if drop_entire:
        print(f"[prune_weak_negatives] dropped {len(signed_groups)-len(kept)} weak components")
    else:
        print(f"[prune_weak_negatives] converted {len(weak_set)} to single-sided (positive only)")
    return kept







def run_full_sid_analysis(chunks_raw: Dict[str, Any],
                          *,
                          primary_chunk: str = 'trial_start',
                          time_chunks: List[str] = ('isi','start_flash_1','end_flash_1',
                                                    'start_flash_2','end_flash_2',
                                                    'choice_start','lick_start'),
                          cp_ranks: List[int] = (6,8,10),
                          min_abs_corr: float = 0.65,
                          signed_q: float = 0.10,
                          phase_bins: int = 80,
                          prune_min_neg_frac: float = 0.10,
                          prune_min_neg_rel_mag: float = 0.25,
                          overlay_chunk_subset: Optional[List[str]] = None,
                          debug: bool = False) -> Dict[str, Any]:
    """
    High-level pipeline runner (starting at 'build M') with descriptive steps.

    Steps:
      1. Build M (session structure) from raw chunk DataFrames (already segmented & aligned).
      2. Summarize M (basic sanity: trial counts, windows, NaN %).
      3. Time-domain CP per requested chunks (rank sweep + pruning + stability).
      4. Build catalog of final time-domain components (incidence matrix).
      5. Overlap diagnostics across time-domain components.
      6. Phase analysis on ISI (resample ISI into φ bins; CP; dual orientation).
      7. Build signed (pos/neg) ROI groups for ALL chunks (time + phase).
      8. Redundancy summary across signed groups (pairwise Jaccard ≥ threshold).
      9. Integrate signed groups into catalog (adds *_pos / *_neg columns).
     10. Assess negative subset strength (loading & trace metrics) per chunk.
     11. Optional pruning of weak negatives (convert to single-sided).
     12. Plot:
          a) Temporal factor shapes (per chunk)
          b) Per-component stacked signed traces
          c) Overlay across components (alpha = energy)
          d) Phase component matrices (optional)
          e) Group overview rasters (catalog components)
     13. Uniqueness & ROI role tables; export (optional).

    Returns dict with all intermediate artifacts.
    """
    results: Dict[str, Any] = {}

    # ---- 1. Build M ----
    M = build_M_from_chunks(chunks_raw, cfg={'isi': {'mask_after_isi': True}},
                            primary_chunk=primary_chunk)
    results['M'] = M
    if debug: summarize_M(M)

    # ---- 2. Time-domain CP (multi-chunk) ----
    time_results = run_time_domain_analysis(
        M,
        chunk_keys=list(time_chunks),
        ranks=list(cp_ranks),
        min_abs_corr=min_abs_corr
    )
    results['time_results'] = time_results

    # ---- 3. Catalog (time-domain only) + overlaps ----
    catalog = build_component_catalog(M, time_results)
    results['catalog_initial'] = catalog
    if debug:
        print("\n[diagnostic] time-domain component overlaps:")
        summarize_component_overlaps(catalog)

    # ---- 4. ISI Phase CP ----
    M['chunks'].pop('isi_phase', None)  # reset if already present
    res_phase, signed_phase = run_phase_analysis(M)  # includes orientation + quick preview plots
    results['res_phase'] = res_phase
    results['signed_phase'] = signed_phase

    # ---- 5. Signed groups for ALL chunks (time + phase) ----
    combined_results = dict(time_results)
    combined_results['isi_phase'] = {'final_A': res_phase['factors']['A']}
    signed_all = build_signed_groups_for_chunks(
        M,
        combined_results,
        q=signed_q,
        use_final=True,
        orient=True,
        orient_method='trace_mean'
    )
    # Replace isi_phase entry with the (possibly more thoroughly oriented) signed_phase
    signed_all['isi_phase'] = signed_phase
    results['signed_all'] = signed_all

    # ---- 6. Redundancy across signed groups ----
    if debug:
        summarize_signed_group_overlap(signed_all, mode='pos', min_jaccard=0.7)

    # ---- 7. Integrate signed groups into catalog ----
    catalog_signed = integrate_signed_groups_into_catalog(
        catalog, M, signed_all, label_prefix='', skip_existing=True
    )
    results['catalog_signed'] = catalog_signed

    # ---- 8. Assess & optionally prune negative subsets (per chunk) ----
    neg_assessment = {}
    pruned_signed = {}
    for ch in list(time_chunks) + ['isi_phase']:
        A_mat = (res_phase['factors']['A'] if ch == 'isi_phase'
                 else time_results[ch]['final_A'])
        sg = signed_all[ch]
        # Loading-level assessment
        df_load = assess_signed_groups(A_mat, sg, metric='loading',
                                       print_table=False)
        # Trace mean assessment (if chunk exists; phase uses 'isi_phase')
        trace_chunk = 'isi_phase' if ch == 'isi_phase' else ch
        df_trace = assess_signed_groups(A_mat, sg, metric='trace_mean',
                                        M=M, chunk=trace_chunk,
                                        print_table=False)
        neg_assessment[ch] = dict(loading=df_load, trace=df_trace)

        # Prune weak negatives (in-place copy)
        sg_copy = [dict(**g) for g in sg]
        prune_weak_negatives(A_mat, sg_copy,
                             min_neg_frac=prune_min_neg_frac,
                             min_neg_rel_mag=prune_min_neg_rel_mag,
                             drop_entire=False)
        pruned_signed[ch] = sg_copy
    results['neg_assessment'] = neg_assessment
    results['signed_pruned'] = pruned_signed

    # ---- 9. Plot temporal factors (normalized waveforms) ----
    if debug:
        for ch in time_chunks:
            plot_component_temporals(time_results[ch], f"{ch} final temporals")

    # ---- 10. Overlay plots (compare component motifs over full trial) ----
    overlay_targets = overlay_chunk_subset or list(time_chunks)
    overlay_figs = {}
    for ch in overlay_targets:
        f = plot_signed_components_overlay(
            M,
            pruned_signed[ch],             # use pruned (neg removed if weak)
            trial_chunk='trial_start',
            chunk_name=ch,
            energy_mode='combined',
            smooth_sigma=0.6,
            sort_by_energy=True,
            annotate=True,
            debug_energies=False
        )
        overlay_figs[ch] = f
    # Phase overlay
    overlay_figs['isi_phase'] = plot_signed_components_overlay(
        M, pruned_signed['isi_phase'],
        trial_chunk='trial_start',
        chunk_name='isi_phase',
        smooth_sigma=0.6,
        energy_mode='combined'
    )
    results['overlay_figs'] = overlay_figs

    # ---- 11. Detailed per-component stacks (optional) ----
    # Example for one chunk (user can loop):
    # plot_signed_component_mean_traces_stack(M, pruned_signed['choice_start'])

    # ---- 12. Phase component matrix (pos/neg) ----
    plot_isi_phase_components_matrix(
        M, res_phase, pruned_signed['isi_phase'],
        comps=None,
        events=('F2_on','F2_off'),
        layout='grid',
        smooth_sigma=0.6
    )

    # ---- 13. Uniqueness + ROI roles (post signed integration) ----
    metrics = component_uniqueness_metrics(catalog_signed)
    roi_roles_df = build_roi_annotation_dataframe(M, catalog_signed, metrics)
    results['uniqueness'] = metrics
    results['roi_roles_df'] = roi_roles_df

    return results


# ---------------------------------------------------------------------------
# Example fully commented usage block (replace ad-hoc bottom script):
"""
# 0. Load your per-chunk pickles (already in variables trial_data_*).
chunks_raw = {
    'trial_start':   trial_data_trial_start,
    'isi':           trial_data_isi,
    'start_flash_1': trial_data_start_flash_1,
    'end_flash_1':   trial_data_end_flash_1,
    'start_flash_2': trial_data_start_flash_2,
    'end_flash_2':   trial_data_end_flash_2,
    'choice_start':  trial_data_choice_start,
    'lick_start':    trial_data_lick_start
}

# 1. Run full pipeline
results = run_full_sid_analysis(
    chunks_raw,
    primary_chunk='trial_start',
    time_chunks=['isi','start_flash_1','end_flash_1',
                 'start_flash_2','end_flash_2','choice_start','lick_start'],
    cp_ranks=[6,8,10],
    min_abs_corr=0.65,
    signed_q=0.10,
    phase_bins=80,
    prune_min_neg_frac=0.12,
    prune_min_neg_rel_mag=0.3,
    overlay_chunk_subset=['isi','choice_start','lick_start'],
    debug=True
)

# 2. Access artifacts
M              = results['M']
time_results   = results['time_results']
res_phase      = results['res_phase']
signed_phase   = results['signed_phase']
signed_all     = results['signed_pruned']     # pruned signed groups (pos / (optional) neg)
catalog_signed = results['catalog_signed']
roi_roles_df   = results['roi_roles_df']

# 3. Save ROI role table if desired
roi_roles_df.to_csv("roi_roles.csv", index=False)

# 4. Additional plots (examples):
plot_signed_component_mean_traces_stack(M, signed_all['choice_start'])
plot_phase_components_on_full_trial(M, signed_all['isi_phase'])
"""







# ============================ UpSet STYLE INTERSECTION PLOT =============================
def build_group_sets_for_upset(signed_groups_by_chunk: Dict[str, List[Dict[str, Any]]],
                               include_negative: bool = True,
                               mode: str = 'pos',
                               min_size: int = 5,
                               dedup_jaccard: float = 0.97,
                               chunks: Optional[List[str]] = None) -> Dict[str, set]:
    """
    Collect ROI sets for UpSet plotting.

    Parameters
    ----------
    signed_groups_by_chunk : dict
        chunk -> list of signed group dicts.
    include_negative : bool
        If True and mode='both', add separate _neg subsets (if non-empty).
    mode : str
        'pos'  -> only positive subsets
        'neg'  -> only negative subsets
        'both' -> include positives (and negatives if include_negative)
    min_size : int
        Reject subsets smaller than this (noise reduction).
    dedup_jaccard : float
        If two sets Jaccard >= threshold keep first, drop later (near duplicates).
    chunks : list[str] | None
        Restrict to subset of chunks (None = all).

    Returns
    -------
    dict label -> python set of ROI indices.
      label format: <chunk>:compK_pos / _neg
    """
    if chunks is None:
        chunks = list(signed_groups_by_chunk.keys())
    raw = {}
    for ch in chunks:
        for g in signed_groups_by_chunk[ch]:
            base = f"{ch}:comp{g['comp']}"
            roi_idx = g['roi_idx']
            if mode in ('pos','both'):
                pos = set(roi_idx[g['positive_mask']])
                if len(pos) >= min_size:
                    raw[base + "_pos"] = pos
            if mode in ('neg','both') and include_negative:
                neg = set(roi_idx[g['negative_mask']])
                if len(neg) >= min_size:
                    raw[base + "_neg"] = neg
    # Deduplicate (greedy)
    keep = {}
    for lbl, s in raw.items():
        dup = False
        for kl, ks in keep.items():
            inter = len(s & ks)
            if inter == 0:
                continue
            jac = inter / len(s | ks)
            if jac >= dedup_jaccard:
                dup = True
                break
        if not dup:
            keep[lbl] = s
    return keep


def compute_upset_intersections(group_sets: Dict[str, set],
                                max_combo_size: int = 4,
                                top_k: int = 30,
                                min_intersection: int = 3,
                                coverage_priority: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Enumerate set intersections (like UpSet). Returns a DataFrame describing each kept combination.

    Parameters
    ----------
    group_sets : dict label -> set
    max_combo_size : int
        Maximum number of sets in an intersection (2..).
    top_k : int
        Limit number of combinations shown (after ranking).
    min_intersection : int
        Drop combinations with < this many ROIs.
    coverage_priority : bool
        If True prefer combinations that add novel ROIs (greedy coverage), else pure size rank.

    Returns
    -------
    df : DataFrame columns:
         combo (tuple of labels),
         size (intersection size),
         roi_indices (sorted list),
         new_coverage (only if coverage_priority)
    labels_order : list
        Ordering of base set labels (used for matrix axis).
    """
    labels = list(group_sets.keys())
    sets_list = [group_sets[l] for l in labels]
    rows = []
    import itertools
    for r in range(2, min(max_combo_size, len(labels)) + 1):
        for comb in itertools.combinations(range(len(labels)), r):
            inter = set.intersection(*(sets_list[i] for i in comb))
            if len(inter) >= min_intersection:
                rows.append((tuple(labels[i] for i in comb), len(inter), sorted(inter)))
    if not rows:
        return pd.DataFrame(columns=['combo','size','roi_indices']), labels
    # Rank
    rows.sort(key=lambda x: x[1], reverse=True)
    if coverage_priority:
        covered = set()
        ranked = []
        for combo, sz, roi_idx in rows:
            new_cov = len(set(roi_idx) - covered)
            if new_cov > 0:
                ranked.append((combo, sz, roi_idx, new_cov))
                covered |= set(roi_idx)
            if len(ranked) >= top_k:
                break
        df = pd.DataFrame(ranked, columns=['combo','size','roi_indices','new_coverage'])
    else:
        rows = rows[:top_k]
        df = pd.DataFrame(rows, columns=['combo','size','roi_indices'])
    return df, labels


def plot_upset(groups_df: pd.DataFrame,
               all_labels: List[str],
               sort_by: str = 'size',
               show_set_sizes: bool = True,
               figsize: Tuple[float,float] = (10,6),
               title: str = "ROI Group Intersections (UpSet)") -> plt.Figure:
    """
    Create a minimalist UpSet-style plot (bars + combination matrix).

    Parameters
    ----------
    groups_df : DataFrame
        Output from compute_upset_intersections.
    all_labels : list[str]
        Master label ordering (rows bottom-up).
    sort_by : 'size' | 'new_coverage'
        Column to order intersection bars.
    show_set_sizes : bool
        If True add side bar with individual set sizes.
    """
    if groups_df.empty:
        raise ValueError("No intersections to plot.")
    metric = sort_by if sort_by in groups_df.columns else 'size'
    groups_df = groups_df.sort_values(metric, ascending=False).reset_index(drop=True)
    combos = groups_df['combo'].tolist()
    sizes = groups_df['size'].to_numpy()

    # Figure layout
    left_w = 0.18 if show_set_sizes else 0.02
    fig = plt.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, width_ratios=[left_w, 1 - left_w], height_ratios=[0.65, 0.35],
                  hspace=0.05, wspace=0.1, figure=fig)
    ax_bar = fig.add_subplot(gs[0,1])
    ax_matrix = fig.add_subplot(gs[1,1], sharex=ax_bar)
    ax_side = fig.add_subplot(gs[:,0]) if show_set_sizes else None

    # Main bars
    ax_bar.bar(range(len(sizes)), sizes, color='#444')
    ax_bar.set_ylabel("Intersection size")
    ax_bar.set_xticks([])
    ax_bar.set_title(title, fontsize=11)

    # Combination matrix (dot rows)
    label_index = {l:i for i,l in enumerate(all_labels)}
    for x, combo in enumerate(combos):
        ys = [label_index[l] for l in combo]
        y_min, y_max = min(ys), max(ys)
        for y in ys:
            ax_matrix.scatter(x, y, s=60, color='tab:blue', zorder=3)
        if len(ys) > 1:
            ax_matrix.plot([x,x], [y_min, y_max], color='tab:blue', lw=2, zorder=2)
    ax_matrix.set_yticks(range(len(all_labels)))
    ax_matrix.set_yticklabels(all_labels, fontsize=7)
    ax_matrix.set_xlabel("Intersection rank")
    ax_matrix.invert_yaxis()

    # Side set sizes
    if show_set_sizes:
        set_sizes = [len(group_sets[l]) for l in all_labels]  # group_sets must exist in outer scope
        ax_side.barh(range(len(all_labels)), set_sizes, color='0.65')
        ax_side.set_yticks(range(len(all_labels)))
        ax_side.set_yticklabels([])
        ax_side.invert_yaxis()
        ax_side.set_xlabel("Set size", fontsize=8)
        for i, v in enumerate(set_sizes):
            ax_side.text(v+0.5, i, str(v), va='center', fontsize=7)

    for spine in ['top','right']:
        ax_bar.spines[spine].set_visible(False)
        ax_matrix.spines[spine].set_visible(False)
        if show_set_sizes:
            ax_side.spines[spine].set_visible(False)
    return fig


def upset_roi_group_overlaps(signed_groups_by_chunk: Dict[str, List[Dict[str, Any]]],
                             chunks: Optional[List[str]] = None,
                             mode: str = 'both',
                             include_negative: bool = True,
                             min_size: int = 5,
                             dedup_jaccard: float = 0.97,
                             max_combo_size: int = 4,
                             top_k: int = 30,
                             min_intersection: int = 3,
                             coverage_priority: bool = True,
                             sort_by: str = 'size',
                             show_set_sizes: bool = True,
                             title: Optional[str] = None) -> Dict[str, Any]:
    """
    High-level one-call UpSet builder for signed ROI groups.

    Returns
    -------
    dict with:
      group_sets, intersections_df, figure
    """
    global group_sets  # allow side size function to access if needed
    group_sets = build_group_sets_for_upset(
        signed_groups_by_chunk,
        include_negative=include_negative,
        mode=mode,
        min_size=min_size,
        dedup_jaccard=dedup_jaccard,
        chunks=chunks
    )
    inter_df, labels = compute_upset_intersections(
        group_sets,
        max_combo_size=max_combo_size,
        top_k=top_k,
        min_intersection=min_intersection,
        coverage_priority=coverage_priority
    )
    if inter_df.empty:
        print("No intersections pass filters.")
        return dict(group_sets=group_sets, intersections_df=inter_df, figure=None)
    fig = plot_upset(inter_df, labels, sort_by=sort_by,
                     show_set_sizes=show_set_sizes,
                     title=title or "ROI Group Intersections")
    plt.tight_layout()
    return dict(group_sets=group_sets, intersections_df=inter_df, figure=fig)








# def plot_upset_enhanced(groups_df: pd.DataFrame,
#                         all_labels: List[str],
#                         *,
#                         sort_by: str = 'size',
#                         show_set_sizes: bool = True,
#                         wrap_labels: bool = True,
#                         wrap_width: int = 22,
#                         truncate_labels: Optional[int] = None,
#                         annotate_bars: bool = True,
#                         annotate_fmt: str = "{size}",
#                         bar_color: str = "#444",
#                         dot_color: str = "tab:blue",
#                         palette: Optional[List[str]] = None,
#                         id_prefix: str = "S",
#                         show_id_column: bool = True,
#                         matrix_use_ids: bool = True,
#                         inline_ids_with_labels: bool = False,
#                         full_label_legend: Optional[str] = None,
#                         legend_cols: int = 2,
#                         label_fontsize: int = 8,
#                         legend_fontsize: int = 8,
#                         min_font: int = 7,
#                         base_fig_height: float = 4.2,
#                         row_height: float = 0.34,
#                         bar_panel_height: float = 2.3,
#                         matrix_panel_height: float = 1.6,
#                         set_size_panel_width: float = 0.18,
#                         figsize_scale: float = 1.0,
#                         extra_width: float = 0.0,
#                         title: str = "ROI Group Intersections (UpSet)",
#                         dpi: int = 120,
#                         xrotate: int = 0,
#                         explain: bool = True,
#                         show_side_pct: bool = True,
#                         show_bar_pct: bool = False,
#                         left_pad: float = 0.14,
#                         side_bar_color: str = "0.80",
#                         panel_title_fontsize: int = 11,
#                         set_size_position: str = 'left'  # NEW: 'left' or 'right'
#                         ) -> Tuple[plt.Figure, pd.DataFrame]:
#     """
#     UpSet plot (enhanced). NEW: set_size_position='right' moves per-set size bars
#     to the right side so long labels are not clipped on the left.
#     """
#     if groups_df.empty:
#         raise ValueError("groups_df is empty.")
#     if set_size_position not in ('left','right'):
#         raise ValueError("set_size_position must be 'left' or 'right'")
#     metric = sort_by if sort_by in groups_df.columns else 'size'
#     groups_df = groups_df.sort_values(metric, ascending=False).reset_index(drop=True)

#     import textwrap
#     n_sets = len(all_labels)
#     id_map = {lab: f"{id_prefix}{i}" for i, lab in enumerate(all_labels)}

#     def _short_label(lab: str) -> str:
#         s = lab
#         if truncate_labels and truncate_labels > 0 and len(s) > truncate_labels:
#             s = s[:truncate_labels-1] + "…"
#         if wrap_labels and wrap_width and wrap_width > 4:
#             w = textwrap.wrap(s, width=wrap_width, break_long_words=False, replace_whitespace=False)
#             if w:
#                 s = "\n".join(w)
#         return s

#     short_labels = [_short_label(l) for l in all_labels]

#     # Figure size & layout
#     longest_line = max(len(max(lbl.split("\n"), key=len)) for lbl in short_labels) if short_labels else 10
#     width_auto = 8.5 + (longest_line / 6.5)
#     fig_w = (width_auto + extra_width) * figsize_scale
#     fig_h = (base_fig_height + n_sets * row_height) * figsize_scale

#     # If bars moved right, increase left padding automatically (unless user overrode)
#     if set_size_position == 'right':
#         left_pad = max(left_pad, 0.20)

#     # Determine side width fraction
#     if show_set_sizes:
#         side_w = set_size_panel_width
#     else:
#         side_w = 0.04

#     import matplotlib.gridspec as gridspec
#     if show_set_sizes and set_size_position == 'left':
#         width_ratios = [side_w, 1 - side_w]
#         side_col = 0
#         main_col = 1
#     elif show_set_sizes and set_size_position == 'right':
#         width_ratios = [1 - side_w, side_w]
#         side_col = 1
#         main_col = 0
#     else:
#         width_ratios = [1.0, 1e-6]  # dummy thin column
#         side_col = 1
#         main_col = 0

#     fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
#     gs = gridspec.GridSpec(
#         3, 2,
#         height_ratios=[bar_panel_height, matrix_panel_height, 0.08],
#         width_ratios=width_ratios,
#         hspace=0.10,
#         wspace=0.08,
#         figure=fig
#     )

#     ax_bar = fig.add_subplot(gs[0, main_col])
#     ax_matrix = fig.add_subplot(gs[1, main_col], sharex=ax_bar)
#     ax_side = fig.add_subplot(gs[:, side_col]) if show_set_sizes else None

#     combos = groups_df['combo'].tolist()
#     sizes = groups_df['size'].to_numpy().astype(float)
#     # Universe
#     universe = set()
#     for s in groups_df['roi_indices']:
#         universe.update(s)
#     total_universe = float(len(universe)) if universe else 1.0

#     # Bars
#     ax_bar.bar(range(len(sizes)), sizes, color=bar_color)
#     ax_bar.set_ylabel("Intersection size (ROIs)")
#     ax_bar.set_xticks([])
#     ax_bar.set_title(title, fontsize=panel_title_fontsize)
#     ymax_inter = sizes.max() if sizes.size else 1
#     if annotate_bars:
#         for x, sz in enumerate(sizes):
#             pct = 100.0 * sz / total_universe
#             txt = annotate_fmt.format(size=int(sz), pct=pct)
#             if show_bar_pct and "{pct" not in annotate_fmt:
#                 txt = f"{txt} ({pct:.1f}%)"
#             ax_bar.text(x, sz + 0.01 * ymax_inter, txt,
#                         ha='center', va='bottom', fontsize=legend_fontsize)

#     # Colors
#     if palette:
#         from itertools import cycle
#         cyc = cycle(palette)
#         set_colors = {lab: next(cyc) for lab in all_labels}
#     else:
#         set_colors = {lab: dot_color for lab in all_labels}

#     # Matrix
#     label_index = {l: i for i, l in enumerate(all_labels)}
#     for x, combo in enumerate(combos):
#         rows = [label_index[l] for l in combo if l in label_index]
#         if not rows:
#             continue
#         ymin, ymax_r = min(rows), max(rows)
#         ax_matrix.plot([x, x], [ymin, ymax_r], color='#555', lw=1.1, zorder=1)
#         for r in rows:
#             lab = all_labels[r]
#             ax_matrix.scatter(x, r, s=70, color=set_colors[lab],
#                               edgecolors='black', linewidths=0.4, zorder=2)

#     # Row labels
#     if matrix_use_ids:
#         if inline_ids_with_labels:
#             ytick_labels = [f"{id_map[l]} | {short_labels[i]}" for i, l in enumerate(all_labels)]
#         else:
#             ytick_labels = [id_map[l] for l in all_labels]
#     else:
#         ytick_labels = [
#             f"{id_map[l]} | {short_labels[i]}" if show_id_column else short_labels[i]
#             for i, l in enumerate(all_labels)
#         ]
#     ax_matrix.set_yticks(range(n_sets))
#     ax_matrix.set_yticklabels(ytick_labels, fontsize=max(min_font, label_fontsize))
#     ax_matrix.invert_yaxis()
#     ax_matrix.set_xlabel("Intersection rank", fontsize=label_fontsize + 1)

#     for spine in ['top', 'right']:
#         ax_bar.spines[spine].set_visible(False)
#         ax_matrix.spines[spine].set_visible(False)

#     # Side set sizes
#     mapping_rows = []
#     if show_set_sizes:
#         base_sizes = []
#         for lab in all_labels:
#             union_lab = set()
#             for combo, roi_idx in zip(groups_df['combo'], groups_df['roi_indices']):
#                 if lab in combo:
#                     union_lab.update(roi_idx)
#             base_sizes.append(len(union_lab))
#             mapping_rows.append(dict(id=id_map[lab], label=lab, size=len(union_lab)))
#         max_side = max(base_sizes) if base_sizes else 1
#         # Ensure correct orientation (horizontal barh works regardless of side position)
#         ax_side.barh(range(n_sets), base_sizes, color=side_bar_color)
#         ax_side.invert_yaxis()
#         ax_side.set_yticks(range(n_sets))
#         ax_side.set_yticklabels([])  # labels in matrix axis
#         side_xlabel = "Set size (ROIs)"
#         ax_side.set_xlabel(side_xlabel, fontsize=label_fontsize)
#         for i, v in enumerate(base_sizes):
#             pct = (100.0 * v / total_universe) if total_universe else 0.0
#             txt = f"{v}"
#             if show_side_pct:
#                 txt = f"{v} ({pct:.1f}%)"
#             offset = 0.01 * max_side
#             ax_side.text(v + offset, i, txt, va='center', fontsize=label_fontsize - 1)
#         for spine in ['top', 'right']:
#             ax_side.spines[spine].set_visible(False)
#         # Flip x-axis label side if bars on right (cosmetic)
#         if set_size_position == 'right':
#             ax_side.yaxis.tick_right()
#     else:
#         for lab in all_labels:
#             mapping_rows.append(dict(id=id_map[lab], label=lab, size=np.nan))
#     mapping_df = pd.DataFrame(mapping_rows)

#     # Explanatory box
#     if explain:
#         fig.text(0.995, 0.02,
#                  "Top bars: intersection sizes\n"
#                  f"{'Right' if set_size_position=='right' else 'Left'} bars: individual set sizes\n"
#                  "Dots column: sets in that intersection\n"
#                  "Line links included sets",
#                  ha='right', va='bottom',
#                  fontsize=legend_fontsize,
#                  bbox=dict(facecolor='white', alpha=0.65, edgecolor='0.7', pad=4))

#     if matrix_use_ids and not inline_ids_with_labels:
#         print("[UpSet] ID mapping:")
#         print(mapping_df[['id', 'label', 'size']].to_string(index=False))

#     plt.subplots_adjust(left=left_pad)
#     return fig, mapping_df



def plot_upset_enhanced(groups_df: pd.DataFrame,
                        all_labels: List[str],
                        *,
                        sort_by: str = 'size',
                        show_set_sizes: bool = True,
                        wrap_labels: bool = True,
                        wrap_width: int = 22,
                        truncate_labels: Optional[int] = None,
                        annotate_bars: bool = True,
                        annotate_fmt: str = "{size}",
                        bar_color: str = "#444",
                        dot_color: str = "tab:blue",
                        palette: Optional[List[str]] = None,
                        id_prefix: str = "S",
                        show_id_column: bool = True,
                        matrix_use_ids: bool = True,
                        inline_ids_with_labels: bool = False,
                        full_label_legend: Optional[str] = None,
                        legend_cols: int = 2,
                        label_fontsize: int = 8,
                        legend_fontsize: int = 8,
                        min_font: int = 7,
                        base_fig_height: float = 4.2,
                        row_height: float = 0.34,
                        bar_panel_height: float = 2.3,
                        matrix_panel_height: float = 1.6,
                        set_size_panel_width: float = 0.18,
                        figsize_scale: float = 1.0,
                        extra_width: float = 0.0,
                        title: str = "ROI Group Intersections (UpSet)",
                        dpi: int = 120,
                        xrotate: int = 0,
                        explain: bool = True,
                        show_side_pct: bool = True,
                        show_bar_pct: bool = False,
                        left_pad: float = 0.14,
                        side_bar_color: str = "0.80",
                        panel_title_fontsize: int = 11,
                        set_size_position: str = 'left',
                        # --- NEW ---
                        show_group_separators: bool = True,
                        group_separator_key: Optional[Callable[[str], str]] = None,
                        group_separator_color: str = 'k',
                        group_separator_lw: float = 1.0,
                        group_separator_alpha: float = 0.55,
                        intersection_count_labels: bool = True,
                        intersection_count_fmt: str = "n={size}",
                        intersection_count_fontsize: int = 7,
                        intersection_count_rotation: int = 0,
                        intersection_count_min_sets: int = 2
                        ) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Enhanced UpSet plot with optional group separators & inline intersection size labels.

    Group separators:
      Determined by group_separator_key(label) or default chunk prefix pre ':'.
      Draw horizontal lines between contiguous groups (matrix + side panel).

    Intersection labels:
      For each column (intersection) place intersection_count_fmt (e.g. 'n=123')
      centered vertically across participating rows if they span >= intersection_count_min_sets sets.
    """
    if groups_df.empty:
        raise ValueError("groups_df is empty.")
    if set_size_position not in ('left','right'):
        raise ValueError("set_size_position must be 'left' or 'right'")
    metric = sort_by if sort_by in groups_df.columns else 'size'
    groups_df = groups_df.sort_values(metric, ascending=False).reset_index(drop=True)

    import textwrap
    n_sets = len(all_labels)
    id_map = {lab: f"{id_prefix}{i}" for i, lab in enumerate(all_labels)}

    def _short_label(lab: str) -> str:
        s = lab
        if truncate_labels and truncate_labels > 0 and len(s) > truncate_labels:
            s = s[:truncate_labels-1] + "…"
        if wrap_labels and wrap_width and wrap_width > 4:
            w = textwrap.wrap(s, width=wrap_width, break_long_words=False, replace_whitespace=False)
            if w:
                s = "\n".join(w)
        return s

    short_labels = [_short_label(l) for l in all_labels]

    # Figure size
    longest_line = max(len(max(lbl.split("\n"), key=len)) for lbl in short_labels) if short_labels else 10
    width_auto = 8.5 + (longest_line / 6.5)
    fig_w = (width_auto + extra_width) * figsize_scale
    fig_h = (base_fig_height + n_sets * row_height) * figsize_scale

    # If bars moved right, increase left padding
    if set_size_position == 'right':
        left_pad = max(left_pad, 0.20)

    if show_set_sizes:
        side_w = set_size_panel_width
    else:
        side_w = 0.04

    import matplotlib.gridspec as gridspec
    if show_set_sizes and set_size_position == 'left':
        width_ratios = [side_w, 1 - side_w]; side_col = 0; main_col = 1
    elif show_set_sizes and set_size_position == 'right':
        width_ratios = [1 - side_w, side_w]; side_col = 1; main_col = 0
    else:
        width_ratios = [1.0, 1e-6]; side_col = 1; main_col = 0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = gridspec.GridSpec(3, 2,
                           height_ratios=[bar_panel_height, matrix_panel_height, 0.08],
                           width_ratios=width_ratios,
                           hspace=0.10, wspace=0.08, figure=fig)
    ax_bar    = fig.add_subplot(gs[0, main_col])
    ax_matrix = fig.add_subplot(gs[1, main_col], sharex=ax_bar)
    ax_side   = fig.add_subplot(gs[:, side_col]) if show_set_sizes else None

    combos = groups_df['combo'].tolist()
    sizes  = groups_df['size'].to_numpy().astype(float)

    # Universe (for percentages)
    universe = set()
    for s in groups_df['roi_indices']:
        universe.update(s)
    total_universe = float(len(universe)) if universe else 1.0

    # Top bars
    ax_bar.bar(range(len(sizes)), sizes, color=bar_color)
    ax_bar.set_ylabel("Intersection size (ROIs)")
    ax_bar.set_xticks([])
    ax_bar.set_title(title, fontsize=panel_title_fontsize)
    ymax_inter = sizes.max() if sizes.size else 1
    if annotate_bars:
        for x, sz in enumerate(sizes):
            pct = 100.0 * sz / total_universe
            txt = annotate_fmt.format(size=int(sz), pct=pct)
            if show_bar_pct and "{pct" not in annotate_fmt:
                txt = f"{txt} ({pct:.1f}%)"
            ax_bar.text(x, sz + 0.01 * ymax_inter, txt,
                        ha='center', va='bottom', fontsize=legend_fontsize)

    # Colors
    if palette:
        from itertools import cycle
        cyc = cycle(palette)
        set_colors = {lab: next(cyc) for lab in all_labels}
    else:
        set_colors = {lab: dot_color for lab in all_labels}

    # Matrix
    label_index = {l: i for i, l in enumerate(all_labels)}
    for x, combo in enumerate(combos):
        rows = [label_index[l] for l in combo if l in label_index]
        if not rows:
            continue
        ymin, ymax_r = min(rows), max(rows)
        ax_matrix.plot([x, x], [ymin, ymax_r], color='#555', lw=1.1, zorder=1)
        for r in rows:
            lab = all_labels[r]
            ax_matrix.scatter(x, r, s=70, color=set_colors[lab],
                              edgecolors='black', linewidths=0.4, zorder=2)
        # Intersection size label inside matrix
        if intersection_count_labels and len(rows) >= intersection_count_min_sets:
            y_mid = (ymin + ymax_r) / 2.0
            ax_matrix.text(x, y_mid,
                           intersection_count_fmt.format(size=int(sizes[x])),
                           ha='center', va='center',
                           fontsize=intersection_count_fontsize,
                           rotation=intersection_count_rotation,
                           color='black',
                           zorder=3)

    # Row labels
    if matrix_use_ids:
        if inline_ids_with_labels:
            ytick_labels = [f"{id_map[l]} | {short_labels[i]}" for i, l in enumerate(all_labels)]
        else:
            ytick_labels = [id_map[l] for l in all_labels]
    else:
        ytick_labels = [
            f"{id_map[l]} | {short_labels[i]}" if show_id_column else short_labels[i]
            for i, l in enumerate(all_labels)
        ]
    ax_matrix.set_yticks(range(n_sets))
    ax_matrix.set_yticklabels(ytick_labels,
                              fontsize=max(min_font, label_fontsize))
    ax_matrix.invert_yaxis()
    ax_matrix.set_xlabel("Intersection rank", fontsize=label_fontsize + 1)

    for spine in ['top', 'right']:
        ax_bar.spines[spine].set_visible(False)
        ax_matrix.spines[spine].set_visible(False)

    # Side set sizes + mapping
    mapping_rows = []
    if show_set_sizes:
        base_sizes = []
        for lab in all_labels:
            union_lab = set()
            for combo, roi_idx in zip(groups_df['combo'], groups_df['roi_indices']):
                if lab in combo:
                    union_lab.update(roi_idx)
            base_sizes.append(len(union_lab))
            mapping_rows.append(dict(id=id_map[lab], label=lab, size=len(union_lab)))
        max_side = max(base_sizes) if base_sizes else 1
        ax_side.barh(range(n_sets), base_sizes, color=side_bar_color)
        ax_side.invert_yaxis()
        ax_side.set_yticks(range(n_sets))
        ax_side.set_yticklabels([])
        ax_side.set_xlabel("Set size (ROIs)", fontsize=label_fontsize)
        for i, v in enumerate(base_sizes):
            pct = (100.0 * v / total_universe) if total_universe else 0.0
            txt = f"{v}" if not show_side_pct else f"{v} ({pct:.1f}%)"
            ax_side.text(v + 0.01 * max_side, i, txt,
                         va='center', fontsize=label_fontsize - 1)
        for spine in ['top', 'right']:
            ax_side.spines[spine].set_visible(False)
        if set_size_position == 'right':
            ax_side.yaxis.tick_right()
    else:
        for lab in all_labels:
            mapping_rows.append(dict(id=id_map[lab], label=lab, size=np.nan))
    mapping_df = pd.DataFrame(mapping_rows)

    # --- NEW: group separators ---
    if show_group_separators and n_sets:
        if group_separator_key is None:
            def group_separator_key(lbl: str) -> str:
                return lbl.split(':', 1)[0] if ':' in lbl else lbl
        group_keys = [group_separator_key(l) for l in all_labels]
        boundaries = []
        cur = group_keys[0]
        for i, gk in enumerate(group_keys):
            if gk != cur:
                boundaries.append(i - 0.5)
                cur = gk
        # after last group not needed visually
        x_min, x_max = -0.5, len(sizes) - 0.5
        for y in boundaries:
            ax_matrix.hlines(y, x_min, x_max,
                             colors=group_separator_color,
                             linestyles='-', lw=group_separator_lw,
                             alpha=group_separator_alpha)
            if show_set_sizes and ax_side is not None:
                ax_side.hlines(y, *ax_side.get_xlim(),
                               colors=group_separator_color,
                               linestyles='-', lw=group_separator_lw,
                               alpha=group_separator_alpha)

    # Explanatory box
    if explain:
        fig.text(0.995, 0.02,
                 f"Top bars: intersection sizes\n"
                 f"{'Right' if (show_set_sizes and set_size_position=='right') else 'Left'} bars: set sizes\n"
                 "Dots column: sets in intersection\n"
                 "Vertical line connects participating sets\n"
                 f"{'Inline n labels = intersection size' if intersection_count_labels else ''}",
                 ha='right', va='bottom',
                 fontsize=legend_fontsize,
                 bbox=dict(facecolor='white', alpha=0.65, edgecolor='0.7', pad=4))

    if matrix_use_ids and not inline_ids_with_labels:
        print("[UpSet] ID mapping:")
        print(mapping_df[['id', 'label', 'size']].to_string(index=False))

    plt.subplots_adjust(left=left_pad)
    return fig, mapping_df



# --- Convenience wrapper combining existing computation + enhanced plot -------------
def upset_plot_enhanced_workflow(signed_groups_by_chunk: Dict[str, List[Dict[str, Any]]],
                                 chunks: Optional[List[str]] = None,
                                 mode: str = 'both',
                                 include_negative: bool = True,
                                 min_size: int = 5,
                                 dedup_jaccard: float = 0.97,
                                 max_combo_size: int = 4,
                                 top_k: int = 40,
                                 min_intersection: int = 3,
                                 coverage_priority: bool = True,
                                 **plot_kwargs) -> Dict[str, Any]:
    """
    One-call pipeline:
      1. Build sets (build_group_sets_for_upset)
      2. Compute intersections (compute_upset_intersections)
      3. Plot using plot_upset_enhanced

    plot_kwargs forwarded to plot_upset_enhanced.
    """
    group_sets = build_group_sets_for_upset(
        signed_groups_by_chunk,
        include_negative=include_negative,
        mode=mode,
        min_size=min_size,
        dedup_jaccard=dedup_jaccard,
        chunks=chunks
    )
    inter_df, labels = compute_upset_intersections(
        group_sets,
        max_combo_size=max_combo_size,
        top_k=top_k,
        min_intersection=min_intersection,
        coverage_priority=coverage_priority
    )
    if inter_df.empty:
        print("[upset enhanced] No intersections passed filters.")
        return dict(group_sets=group_sets, intersections=inter_df, figure=None)
    fig = plot_upset_enhanced(inter_df, labels, **plot_kwargs)
    return dict(group_sets=group_sets, intersections=inter_df, figure=fig)
# ...existing code...







































# --- Sankey / Alluvial style visualizations for ROI signed groups -----------------------
# Two styles:
#   (A) Overlap graph: each group node; links weighted by intersection size (J>=thr or abs count).
#   (B) Hierarchy flow: Chunk  ->  Group(+/-)  ->  Cluster (coarse merged motif)
# Requires: pip install plotly

import math

def _collect_signed_sets(signed_groups_by_chunk,
                         chunks=None,
                         include_negative=True,
                         min_size=5,
                         dedup_jaccard=0.97):
    if chunks is None:
        chunks = list(signed_groups_by_chunk.keys())
    sets = {}
    for ch in chunks:
        for g in signed_groups_by_chunk[ch]:
            base = f"{ch}:c{g['comp']}"
            roi_idx = g['roi_idx']
            # positive
            pos = set(roi_idx[g['positive_mask']])
            if len(pos) >= min_size:
                sets[base + "_pos"] = pos
            if include_negative:
                neg = set(roi_idx[g['negative_mask']])
                if len(neg) >= min_size:
                    sets[base + "_neg"] = neg
    # greedy dedup
    keep = {}
    for lbl, s in sets.items():
        dup=False
        for kl, ks in keep.items():
            inter = len(s & ks)
            if not inter: continue
            jac = inter / len(s | ks)
            if jac >= dedup_jaccard:
                dup=True; break
        if not dup:
            keep[lbl]=s
    return keep

def sankey_group_overlap(group_sets,
                         jaccard_thresh=0.25,
                         max_links=250,
                         use_jaccard=False,
                         title=None):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Install plotly: pip install plotly")
        return None
    labels = list(group_sets.keys())
    sets_list = [group_sets[l] for l in labels]
    src=[]; tgt=[]; val=[]; hover=[]
    n=len(labels)
    for i in range(n):
        A=sets_list[i]
        for j in range(i+1,n):
            B=sets_list[j]
            inter = len(A & B)
            if inter==0: continue
            jac = inter / len(A | B)
            if jac >= jaccard_thresh:
                src.append(i); tgt.append(j)
                val.append(jac if use_jaccard else inter)
                hover.append(f"{labels[i]} ↔ {labels[j]}<br>|∩|={inter}  J={jac:.2f}")
    if not val:
        print("No pairwise overlaps pass threshold.")
        return None
    # limit
    if len(val) > max_links:
        order = sorted(range(len(val)), key=lambda k: val[k], reverse=True)[:max_links]
        src = [src[k] for k in order]
        tgt = [tgt[k] for k in order]
        val = [val[k] for k in order]
        hover = [hover[k] for k in order]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=14, thickness=14),
        link=dict(source=src, target=tgt, value=val, hovertemplate=hover)
    ))
    fig.update_layout(title=title or f"Signed ROI group overlaps (J≥{jaccard_thresh})")
    return fig

def _simple_cluster_sets(group_sets, jaccard_merge=0.6):
    labels=list(group_sets.keys())
    sets={l: set(group_sets[l]) for l in labels}
    clusters=[]
    used=set()
    for lbl in sorted(labels, key=lambda x: -len(sets[x])):
        if lbl in used: continue
        base=sets[lbl]
        members=[lbl]; used.add(lbl)
        for other in labels:
            if other in used: continue
            inter=len(base & sets[other])
            if inter==0: continue
            jac=inter / len(base | sets[other])
            if jac >= jaccard_merge:
                members.append(other); used.add(other)
        clusters.append(members)
    return clusters  # list of lists

def sankey_chunk_group_cluster(signed_groups_by_chunk,
                               chunks=None,
                               include_negative=True,
                               min_size=5,
                               dedup_jaccard=0.97,
                               cluster_jaccard=0.55,
                               max_groups=80,
                               title=None):
    """
    Flow: Chunk -> Group(+/-) -> Cluster (merged motif)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Install plotly: pip install plotly")
        return None
    group_sets = _collect_signed_sets(signed_groups_by_chunk,
                                      chunks=chunks,
                                      include_negative=include_negative,
                                      min_size=min_size,
                                      dedup_jaccard=dedup_jaccard)
    # optional cap
    if len(group_sets) > max_groups:
        # keep largest sets first
        ordered = sorted(group_sets.items(), key=lambda kv: -len(kv[1]))[:max_groups]
        group_sets = dict(ordered)
    # clusters
    clusters = _simple_cluster_sets(group_sets, jaccard_merge=cluster_jaccard)
    cluster_label_map={}
    for i, mem in enumerate(clusters):
        all_rois = set().union(*(group_sets[m] for m in mem))
        cluster_label_map[i]=dict(members=mem, size=len(all_rois))
    # build nodes
    chunk_nodes = sorted({lbl.split(':',1)[0] for lbl in group_sets})
    group_nodes = list(group_sets.keys())
    cluster_nodes = [f"cluster_{i}" for i in range(len(clusters))]
    node_labels = chunk_nodes + group_nodes + cluster_nodes
    idx = {lab:i for i, lab in enumerate(node_labels)}
    # links chunk -> group
    src=[]; tgt=[]; val=[]; hover=[]
    for g in group_nodes:
        chunk = g.split(':',1)[0]
        s = idx[chunk]; t=idx[g]
        size = len(group_sets[g])
        src.append(s); tgt.append(t); val.append(size)
        hover.append(f"{chunk} → {g}<br>|set|={size}")
    # links group -> cluster
    for ci, info in cluster_label_map.items():
        c_label = f"cluster_{ci}"
        for g in info['members']:
            s=idx[g]; t=idx[c_label]
            inter = len(group_sets[g])  # value = group size (flow)
            src.append(s); tgt.append(t); val.append(inter)
            hover.append(f"{g} → {c_label}<br>|group|={inter}")
    # cluster labels enhanced
    for ci, info in cluster_label_map.items():
        members = info['members']
        node_labels[idx[f"cluster_{ci}"]] = f"Cluster {ci}\n(n={info['size']})"
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=node_labels, pad=12, thickness=16),
        link=dict(source=src, target=tgt, value=val, hovertemplate=hover)
    ))
    fig.update_layout(title=title or "Chunk → Group(+/-) → Cluster flow")
    return fig

# ---- Convenience wrappers ---------------------------------------------------

def plot_sankey_overlap(signed_groups_by_chunk,
                        chunks=None,
                        include_negative=True,
                        min_size=5,
                        dedup_jaccard=0.97,
                        jaccard_thresh=0.25,
                        use_jaccard=False):
    gs = _collect_signed_sets(signed_groups_by_chunk, chunks, include_negative, min_size, dedup_jaccard)
    return sankey_group_overlap(gs,
                                jaccard_thresh=jaccard_thresh,
                                use_jaccard=use_jaccard,
                                title="Signed ROI group overlap Sankey")

def plot_sankey_hierarchy(signed_groups_by_chunk,
                          chunks=None,
                          include_negative=True,
                          min_size=5,
                          dedup_jaccard=0.97,
                          cluster_jaccard=0.55):
    return sankey_chunk_group_cluster(signed_groups_by_chunk,
                                      chunks=chunks,
                                      include_negative=include_negative,
                                      min_size=min_size,
                                      dedup_jaccard=dedup_jaccard,
                                      cluster_jaccard=cluster_jaccard)
# --------------------------------------------------------------------------------
# Example (after you have 'signed_pruned'):
# fig1 = plot_sankey_overlap(signed_pruned, chunks=['isi','choice_start','lick_start'], jaccard_thresh=0.3)
# fig1.show()
# fig2 = plot_sankey_hierarchy(signed_pruned, chunks




















# Rationale: Sankey (Plotly) is interactive; this gives a pure matplotlib
# exportable “alluvial” style: vertical strata (stages) with colored flows.

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    inter = len(a & b)
    if inter == 0: return 0.0
    return inter / len(a | b)

def _cluster_group_sets(group_sets: Dict[str,set],
                        jaccard_thresh: float = 0.55) -> Dict[str,str]:
    """
    Greedy cluster of group_sets (label->set) by J >= thresh.
    Returns mapping group_label -> cluster_id (string).
    """
    labels = sorted(group_sets.keys(), key=lambda l: -len(group_sets[l]))
    assigned = {}
    clusters = []
    for lbl in labels:
        if lbl in assigned: continue
        seed = group_sets[lbl]
        members = [lbl]
        for other in labels:
            if other in assigned: continue
            if other == lbl: continue
            if _jaccard(seed, group_sets[other]) >= jaccard_thresh:
                members.append(other)
        cid = f"C{len(clusters)}"
        for m in members:
            assigned[m] = cid
        clusters.append(members)
    return assigned  # label -> cluster id

def build_alluvial_spec(signed_groups_by_chunk: Dict[str,List[Dict[str,Any]]],
                        *,
                        chunks: Optional[List[str]] = None,
                        include_negative: bool = True,
                        min_size: int = 5,
                        dedup_jaccard: float = 0.97,
                        cluster_jaccard: float = 0.55) -> Dict[str,Any]:
    """
    Stage 0: chunk
    Stage 1: group label (chunk:compK_pos / _neg)
    Stage 2: cluster (auto Jaccard merge)
    """
    if chunks is None:
        chunks = list(signed_groups_by_chunk.keys())
    # Collect sets
    raw = {}
    for ch in chunks:
        for g in signed_groups_by_chunk[ch]:
            base = f"{ch}:c{g['comp']}"
            roi_idx = g['roi_idx']
            pos = set(roi_idx[g['positive_mask']])
            if len(pos) >= min_size:
                raw[base+"_pos"] = pos
            if include_negative:
                neg = set(roi_idx[g['negative_mask']])
                if len(neg) >= min_size:
                    raw[base+"_neg"] = neg
    # Dedup
    keep = {}
    for lbl, s in raw.items():
        dup=False
        for k2, s2 in keep.items():
            j = _jaccard(s,s2)
            if j >= dedup_jaccard:
                dup=True; break
        if not dup:
            keep[lbl]=s
    if not keep:
        return dict(flows=[], strata=[], cluster_map={}, group_sets={})
    # Cluster
    cluster_map = _cluster_group_sets(keep, jaccard_thresh=cluster_jaccard)
    # Build flows (chunk->group, group->cluster)
    flows = []
    for lbl, s in keep.items():
        chunk = lbl.split(':',1)[0]
        flows.append(dict(src=('chunk',chunk), dst=('group',lbl), size=len(s)))
        flows.append(dict(src=('group',lbl), dst=('cluster',cluster_map[lbl]), size=len(s)))
    # Strata sizes
    strata = {}
    # chunks
    for ch in chunks:
        size_union = 0
        # union all sets from that chunk (avoid double count duplicates we already deduped)
        union = set()
        for lbl, s in keep.items():
            if lbl.startswith(ch + ":"):
                union |= s
        size_union = len(union)
        strata[('chunk',ch)] = size_union
    # groups
    for lbl, s in keep.items():
        strata[('group',lbl)] = len(s)
    # clusters: union of member sets
    cl_union = {}
    for lbl, cid in cluster_map.items():
        cl_union.setdefault(cid, set()).update(keep[lbl])
    for cid, s in cl_union.items():
        strata[('cluster',cid)] = len(s)
    return dict(flows=flows, strata=strata, cluster_map=cluster_map, group_sets=keep)


def plot_alluvial(alluvial_spec: Dict[str,Any],
                  stage_order: Tuple[str,str,str] = ('chunk','group','cluster'),
                  colormap: str = 'tab20',
                  flow_alpha: float = 0.55,
                  min_flow: int = 1,
                  figsize: Tuple[float,float] = (14,6),
                  gap: float = 4.0,
                  palette_override: Optional[Dict[str,str]] = None) -> Optional[plt.Figure]:
    """
    Draw alluvial with three vertical stages. Width encodes count (ROIs).
    """
    if not alluvial_spec.get('flows'):
        print("[alluvial] nothing to plot.")
        return None
    import numpy as np
    flows = [f for f in alluvial_spec['flows'] if f['size'] >= min_flow]
    strata = alluvial_spec['strata']
    # Collect nodes per stage
    stage_nodes = {}
    for (stage,name), size in strata.items():
        stage_nodes.setdefault(stage, []).append((name,size))
    # Consistent order (largest first)
    for st in stage_nodes:
        stage_nodes[st].sort(key=lambda x: -x[1])
    # y positions (stacked)
    node_pos = {}
    for i, stage in enumerate(stage_order):
        y = 0.0
        for name,size in stage_nodes.get(stage, []):
            node_pos[(stage,name)] = (i, y, size)   # (x, y, size)
            y += size + gap
    # Colors (cluster-based). Handle case of zero clusters gracefully.
    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap, 20)
    cluster_ids = [name for (stage,name) in node_pos if stage == 'cluster']
    cluster_ids_sorted = sorted(cluster_ids)
    cluster_colors = {}
    for idx, cid in enumerate(cluster_ids_sorted):
        cluster_colors[cid] = (palette_override[cid] if (palette_override and cid in palette_override)
                               else cmap(idx % cmap.N))
    # Map group -> cluster color; chunk color = mean of its outgoing groups (else light gray)
    group_to_cluster = {}
    for lbl, cid in alluvial_spec['cluster_map'].items():
        group_to_cluster[lbl] = cluster_colors.get(cid, (0.6,0.6,0.6,1))
    chunk_colors = {}
    for (stage,name) in node_pos:
        if stage != 'chunk':
            continue
        ccols = []
        for g,cid in alluvial_spec['cluster_map'].items():
            if g.startswith(name + ":"):
                ccols.append(cluster_colors.get(cid, (0.6,0.6,0.6,1)))
        if ccols:
            chunk_colors[name] = np.mean(np.array(ccols), axis=0)
        else:
            chunk_colors[name] = (0.85,0.85,0.85,1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    def draw_node(stage,name):
        x,y,size = node_pos[(stage,name)]
        face = (cluster_colors.get(name) if stage=='cluster'
                else group_to_cluster.get(name, chunk_colors.get(name,(0.85,0.85,0.85,1))))
        ax.add_patch(plt.Rectangle((x-0.4, y), 0.8, size,
                                   facecolor=face,
                                   edgecolor='k', linewidth=0.4))
        ax.text(x, y+size/2, name, ha='center', va='center',
                fontsize=7, rotation=90 if stage=='group' else 0)

    flow_stack_out = {}
    flow_stack_in  = {}
    def alloc(slot, size, which):
        stack = flow_stack_out if which=='out' else flow_stack_in
        cur = stack.get(slot, 0.0)
        stack[slot] = cur + size
        return cur

    # Draw nodes
    for st in stage_order:
        for name,_size in stage_nodes.get(st, []):
            draw_node(st, name)

    # Draw flows
    for f in flows:
        (s_stage,s_name), (d_stage,d_name) = f['src'], f['dst']
        if (s_stage,s_name) not in node_pos or (d_stage,d_name) not in node_pos:
            continue
        sx, sy0, _ssize = node_pos[(s_stage,s_name)]
        dx, dy0, _dsize = node_pos[(d_stage,d_name)]
        size = f['size']
        sy_off = alloc((s_stage,s_name), size, 'out')
        dy_off = alloc((d_stage,d_name), size, 'in')
        sy1 = sy0 + sy_off
        dy1 = dy0 + dy_off
        x0 = sx + 0.4
        x1 = dx - 0.4
        verts = [
            (x0, sy1), (x0+0.2, sy1), (x1-0.2, dy1), (x1, dy1),
            (x1, dy1+size), (x1-0.2, dy1+size), (x0+0.2, sy1+size), (x0, sy1+size),
            (x0, sy1)
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.CLOSEPOLY]
        flow_color = (group_to_cluster.get(s_name) if s_stage!='cluster'
                      else group_to_cluster.get(d_name, (0.5,0.5,0.5,1)))
        ax.add_patch(PathPatch(Path(verts, codes),
                               facecolor=flow_color,
                               alpha=flow_alpha,
                               edgecolor='none'))

    # Compute vertical extent (BUG FIX: simpler & safe)
    if node_pos:
        y_max = max(y + size for (_, y, size) in node_pos.values())
    else:
        y_max = 0.0

    ax.set_xlim(-0.8, len(stage_order)-0.2)
    ax.set_ylim(-gap*0.5, y_max + gap*0.5)

    for i,st in enumerate(stage_order):
        ax.text(i, y_max + gap*0.25, st.upper(), ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_title("Alluvial: chunk → group(+/-) → cluster", fontsize=11)
    return fig


# Example usage (after you have 'signed_pruned'):
# spec = build_alluvial_spec(signed_pruned,
#                             chunks=['isi','choice_start','lick_start'],
#                             include_negative=True,
#                             min_size=6,
#                             dedup_jaccard=0.97,
#                             cluster_jaccard=0.6)
# fig = plot_alluvial(spec, flow_alpha=0.55)
# fig.savefig("alluvial_static.png", dpi=150)

















































# --- ROI MEMBERSHIP OVERLAP VISUALIZATION UTILITIES ------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def roi_membership_summary(catalog: Dict[str, Any]) -> pd.DataFrame:
    """
    Summarize per-ROI component membership.
    Returns DataFrame with:
      roi        : ROI index
      n_groups   : number of components containing ROI
      groups     : semicolon list of labels
    """
    inc = catalog['incidence']      # (n_rois, n_comp) bool
    labels = catalog['comp_labels']
    n_rois = inc.shape[0]
    rows = []
    for i in range(n_rois):
        cols = np.where(inc[i])[0]
        rows.append(dict(
            roi=i,
            n_groups=cols.size,
            groups=";".join(labels[c] for c in cols)
        ))
    return pd.DataFrame(rows)

def plot_roi_membership_distribution(catalog: Dict[str, Any]):
    df = roi_membership_summary(catalog)
    plt.figure(figsize=(4.2,3.2))
    vc = df['n_groups'].value_counts().sort_index()
    plt.bar(vc.index, vc.values, width=0.6, color='0.4')
    plt.xlabel("# component memberships")
    plt.ylabel("# ROIs")
    plt.title("Per-ROI membership count")
    for x,v in zip(vc.index, vc.values):
        plt.text(x, v, str(v), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    return df

def plot_roi_membership_matrix(catalog: Dict[str, Any],
                               max_rois: int = 400,
                               order: str = 'hier',   # 'hier'|'degree'
                               cmap: str = 'Greys',
                               figsize: Tuple[float,float] = (10,6),
                               show_colorbar: bool = False):
    """
    Binary incidence heatmap (ROI vs component).
    order:
      'degree' : sort ROIs descending by n_groups
      'hier'   : hierarchical clustering on Jaccard / (binary) incidence
    If n_rois > max_rois, keeps top (by degree) + random sample of singles.
    """
    inc = catalog['incidence']
    labels = catalog['comp_labels']
    n_rois = inc.shape[0]
    n_comp = inc.shape[1]
    deg = inc.sum(axis=1)
    idx = np.arange(n_rois)
    if n_rois > max_rois:
        multi = idx[deg>1]
        single = idx[deg<=1]
        keep_multi = multi
        n_rem = max(0, max_rois - keep_multi.size)
        if n_rem > 0 and single.size:
            rng = np.random.default_rng(0)
            keep_single = rng.choice(single, size=min(n_rem, single.size), replace=False)
            keep = np.concatenate([keep_multi, keep_single])
        else:
            keep = keep_multi
        inc_sub = inc[keep]
        deg_sub = deg[keep]
        roi_map = keep
    else:
        inc_sub = inc
        deg_sub = deg
        roi_map = idx
    if order == 'degree':
        row_order = np.argsort(-deg_sub)
    else:
        # hierarchical clustering (Seaborn clustermap)
        cg = sns.clustermap(inc_sub, metric='jaccard', method='average',
                            cmap=cmap, col_cluster=False, yticklabels=False,
                            figsize=figsize)
        cg.ax_heatmap.set_xlabel("Components")
        cg.ax_heatmap.set_ylabel("ROIs")
        cg.ax_heatmap.set_title("ROI membership (clustered)")
        return dict(row_index=roi_map[cg.dendrogram_row.reordered_ind],
                    col_labels=labels)
    inc_plot = inc_sub[row_order]
    plt.figure(figsize=figsize)
    plt.imshow(inc_plot, aspect='auto', interpolation='none', cmap=cmap)
    plt.title("ROI membership (sorted)" if order=='degree' else "ROI membership")
    plt.xlabel("Components")
    plt.ylabel("ROIs")
    if not show_colorbar:
        plt.colorbar().remove() if plt.gca().images else None
    plt.tight_layout()
    return dict(row_index=roi_map[row_order], col_labels=labels)

def roi_co_membership_matrix(catalog: Dict[str, Any],
                             top_k_multi: int = 120,
                             metric: str = 'shared',  # 'shared'|'jaccard'
                             min_groups: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (matrix, roi_indices) where matrix is ROI x ROI co-membership.
    Only uses ROIs with n_groups >= min_groups, limited to top_k_multi by degree.
    metric:
      'shared'  : # of shared component memberships
      'jaccard' : Jaccard similarity on binary membership vectors
    """
    inc = catalog['incidence']
    deg = inc.sum(axis=1)
    sel = np.where(deg >= min_groups)[0]
    sel = sel[np.argsort(-deg[sel])]
    if sel.size > top_k_multi:
        sel = sel[:top_k_multi]
    sub = inc[sel]
    if metric == 'shared':
        mat = sub @ sub.T
    else:
        # Jaccard
        a = sub.astype(bool)
        inter = a @ a.T
        sums = a.sum(axis=1)
        union = sums[:,None] + sums[None,:] - inter
        mat = inter / np.maximum(union, 1)
    return mat, sel


def plot_roi_co_membership(catalog: Dict[str, Any],
                           top_k_multi: int = 120,
                           metric: str = 'shared',
                           figsize: Tuple[float,float] = (7,6),
                           vmax: Optional[float] = None):
    """
    Heatmap of ROI co-membership (selected multi-role ROIs).

    FIX:
      Cast matrix to float before np.quantile (bool subtraction in newer NumPy raises TypeError).
      Handle degenerate all-zero matrix gracefully.
    """
    mat, sel = roi_co_membership_matrix(catalog, top_k_multi=top_k_multi, metric=metric)
    if mat.size == 0:
        print("No multi-role ROIs for co-membership heatmap.")
        return None
    # Ensure float dtype to avoid boolean subtraction error inside np.quantile
    mat = mat.astype(float, copy=False)
    if vmax is None:
        try:
            vmax = float(np.quantile(mat, 0.98))
        except Exception:
            vmax = float(np.nanmax(mat))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
    if vmax <= 0:
        vmax = 1.0
    plt.figure(figsize=figsize)
    sns.heatmap(mat,
                cmap='magma',
                vmin=0,
                vmax=vmax,
                cbar_kws=dict(label='Shared comps' if metric=='shared' else 'Jaccard'))
    plt.title(f"ROI co-membership ({metric})  n={len(sel)}")
    plt.xlabel("ROIs (subset)")
    plt.ylabel("ROIs (subset)")
    plt.tight_layout()
    return dict(roi_indices=sel, matrix=mat, vmax=vmax)

def embed_rois_by_membership(catalog: Dict[str, Any],
                             method: str = 'tsne',  # 'tsne'|'umap' (umap optional)
                             perplexity: float = 30,
                             min_groups: int = 1,
                             random_state: int = 0,
                             figsize: Tuple[float,float] = (5.2,4.6)):
    """
    2D embedding of ROIs based on binary component membership vector.
    Color = membership count; size emphasizes multi-role.
    """
    inc = catalog['incidence'].astype(float)
    deg = inc.sum(axis=1)
    sel = np.where(deg >= min_groups)[0]
    X = inc[sel]
    if X.shape[0] < 5:
        print("Too few ROIs for embedding.")
        return None
    if method == 'umap':
        try:
            import umap
            emb = umap.UMAP(random_state=random_state, n_neighbors=15, min_dist=0.2).fit_transform(X)
        except ImportError:
            print("umap-learn not installed; falling back to t-SNE.")
            from sklearn.manifold import TSNE
            emb = TSNE(perplexity=min(perplexity, max(5, X.shape[0]//4)),
                       random_state=random_state).fit_transform(X)
    else:
        from sklearn.manifold import TSNE
        emb = TSNE(perplexity=min(perplexity, max(5, X.shape[0]//4)),
                   random_state=random_state).fit_transform(X)
    plt.figure(figsize=figsize)
    sc = plt.scatter(emb[:,0], emb[:,1],
                     c=deg[sel], s=15 + 10*(deg[sel]-deg[sel].min()),
                     cmap='viridis', alpha=0.85, linewidths=0)
    plt.colorbar(sc, label="# memberships")
    plt.title("ROI embedding by component membership")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    return dict(roi_indices=sel, embedding=emb, degree=deg[sel])

def annotate_independent_vs_overlap(df_roles: pd.DataFrame,
                                    overlap_thresh: int = 1) -> pd.DataFrame:
    """
    Adds column 'role_type': 'independent' if n_groups<=overlap_thresh else 'multi'.
    """
    df = df_roles.copy()
    df['role_type'] = np.where(df['n_groups'] <= overlap_thresh, 'independent', 'multi')
    return df

# High-level convenience wrapper
def visualize_roi_overlap_suite(catalog: Dict[str, Any],
                                co_metric: str = 'shared',
                                top_k_multi: int = 120):
    """
    Runs:
      1. Membership distribution
      2. ROI membership matrix (hier clustering)
      3. Co-membership heatmap
      4. 2D embedding
    """
    df = plot_roi_membership_distribution(catalog)
    plot_roi_membership_matrix(catalog, order='hier')
    plot_roi_co_membership(catalog, metric=co_metric, top_k_multi=top_k_multi)
    embed_rois_by_membership(catalog)
    return df
# ----------------------------------------------------------------------------------------
# Example usage after building catalog (with signed groups integrated):
# df_roles = roi_membership_summary(catalog)
# visualize_roi_overlap_suite(catalog, co_metric='jaccard', top_k_multi=100)
# co = plot_roi_co_membership(catalog, metric='shared', top_k_multi=80)
# emb = embed_rois_by_membership(catalog, method='tsne')
# ----------------------------------------------------------------------------------------














































# --- Functional component descriptor + category overlap utilities -----------------------
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

def _mean_trace_for_component(M, roi_indices, chunk: str, smooth_sigma: float = 0.6):
    t = M['chunks'][chunk]['time']
    X = M['chunks'][chunk]['data']  # trials x rois x time
    mean_roi = np.nanmean(X[:, roi_indices, :], axis=1)       # trials x time
    mean_trace_all = np.nanmean(mean_roi, axis=0)
    # Short vs Long (if isi classification present)
    short_mask = M.get('is_short')
    mean_short = mean_long = None
    if short_mask is not None and short_mask.size == mean_roi.shape[0]:
        mean_short = np.nanmean(mean_roi[short_mask], axis=0)
        mean_long  = np.nanmean(mean_roi[~short_mask], axis=0)
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        mean_trace_all = gaussian_filter1d(mean_trace_all, smooth_sigma)
        if mean_short is not None:
            mean_short = gaussian_filter1d(mean_short, smooth_sigma)
            mean_long  = gaussian_filter1d(mean_long, smooth_sigma)
    return dict(t=t, all=mean_trace_all, short=mean_short, long=mean_long)

def compute_component_functional_descriptors(M: Dict[str,Any],
                                             catalog: Dict[str,Any],
                                             *,
                                             chunks_filter: Optional[List[str]] = None,
                                             onset_window: Tuple[float,float] = (-0.2, 0.6),
                                             peak_window: Optional[Tuple[float,float]] = None,
                                             smoothing_sigma: float = 0.6,
                                             min_z: float = 0.5) -> pd.DataFrame:
    """
    Derive simple temporal descriptors for each catalog component (unsigned).
    Returns DataFrame with: label, chunk, size, onset_time, peak_time, peak_amp,
      short_pref (Δ mean), is_ramp (bool), early/late flags, duration_50 (s),
      time_to_peak, category_hint (blank initially).
    """
    inc = catalog['incidence']
    labels = catalog['comp_labels']
    out = []
    t_trial = M['chunks']['trial_start']['time']
    # Helper to compute onset
    def _onset(t, y, w, zthr):
        mask = (t >= w[0]) & (t <= w[1])
        if not mask.any(): return np.nan
        y_win = y[mask]
        base = np.nanmean(y[t < w[0]])
        sd = np.nanstd(y[t < w[0]]) + 1e-9
        z = (y_win - base)/sd
        idx = np.where(z > zthr)[0]
        return (t[mask][idx[0]] if idx.size else np.nan)
    for j, label in enumerate(labels):
        if chunks_filter and not any(label.startswith(cf) for cf in chunks_filter):
            continue
        roi_idx = np.where(inc[:, j])[0]
        if roi_idx.size == 0: continue
        chunk = label.split(':',1)[0]
        data = _mean_trace_for_component(M, roi_idx, chunk, smoothing_sigma)
        t = data['t']; y = data['all']
        if peak_window is None:
            pw = (t.min(), t.max())
        else:
            pw = peak_window
        p_mask = (t>=pw[0]) & (t<=pw[1])
        if p_mask.any():
            pk_idx = p_mask.nonzero()[0][np.argmax(y[p_mask])]
            peak_time = t[pk_idx]; peak_amp = y[pk_idx]
        else:
            peak_time = np.nan; peak_amp = np.nan
        onset_time = _onset(t, y, onset_window, min_z)
        # Ramp metric: correlation with linear time in window
        ramp_mask = (t>=onset_window[0]) & (t<=onset_window[1])
        ramp_r = np.nan
        if ramp_mask.sum() > 5:
            from scipy.stats import pearsonr
            try:
                ramp_r = pearsonr(t[ramp_mask], y[ramp_mask])[0]
            except Exception:
                ramp_r = np.nan
        # Duration 50% of peak
        if np.isfinite(peak_amp):
            half = 0.5 * peak_amp
            above = y >= half
            # contiguous region containing peak
            duration_50 = np.nan
            if above.any():
                # find indices around peak
                pk = np.argmax(y)
                left = pk
                while left>0 and above[left-1]: left -= 1
                right = pk
                while right < len(y)-1 and above[right+1]: right += 1
                duration_50 = t[right]-t[left]
        else:
            duration_50 = np.nan
        short_pref = np.nan
        if data['short'] is not None:
            short_pref = float(np.nanmean(data['short']) - np.nanmean(data['long']))
        out.append(dict(
            label=label,
            chunk=chunk,
            size=int(roi_idx.size),
            onset_time=onset_time,
            peak_time=peak_time,
            peak_amp=peak_amp,
            duration_50=duration_50,
            time_to_peak=(peak_time - onset_time) if (np.isfinite(onset_time) and np.isfinite(peak_time)) else np.nan,
            ramp_r=ramp_r,
            short_pref=short_pref,
            early_flag = onset_time < 0.05 if np.isfinite(onset_time) else False,
            late_flag  = peak_time > 0.3 if np.isfinite(peak_time) else False,
            category_hint=""
        ))
    return pd.DataFrame(out)

def categorize_components(descriptor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based assignment of functional category from descriptors.
    Modify rules to refine categories.
    """
    df = descriptor_df.copy()
    cats=[]
    for r in df.itertuples():
        c = r.chunk
        # base by chunk
        if c.startswith('start_flash'):
            cat = 'stim_on'
        elif c.startswith('end_flash'):
            cat = 'stim_off'
        elif c == 'isi':
            if r.ramp_r > 0.35 and (r.peak_time > 0.2):
                cat = 'isi_ramp'
            else:
                cat = 'isi_transient'
        elif c == 'choice_start':
            cat = 'pre_choice' if (r.onset_time < 0) else 'choice_aligned'
        elif c == 'lick_start':
            cat = 'lick'
        else:
            cat = c
        # refine with short/long preference
        if np.isfinite(r.short_pref):
            if r.short_pref > 0.4:
                cat += '_shortPref'
            elif r.short_pref < -0.4:
                cat += '_longPref'
        cats.append(cat)
    df['category'] = cats
    return df

def build_functional_category_sets(catalog: Dict[str,Any],
                                   descriptor_df: pd.DataFrame,
                                   *,
                                   min_size: int = 5,
                                   min_component_size: int = 5) -> Dict[str,set]:
    """
    Aggregate ROI sets per functional category (union of components in that category).
    """
    inc = catalog['incidence']
    labels = catalog['comp_labels']
    comp_index = {lab:i for i,lab in enumerate(labels)}
    cat_to_indices = defaultdict(list)
    for row in descriptor_df.itertuples():
        if row.size < min_component_size: continue
        cat_to_indices[row.category].append(comp_index[row.label])
    cat_sets = {}
    for cat, comp_idx_list in cat_to_indices.items():
        if not comp_idx_list: continue
        roi_mask = inc[:, comp_idx_list].any(axis=1)
        roi_set = set(np.where(roi_mask)[0])
        if len(roi_set) >= min_size:
            cat_sets[cat] = roi_set
    return cat_sets

def category_overlap_table(category_sets: Dict[str,set]) -> pd.DataFrame:
    rows=[]
    cats=list(category_sets.keys())
    for i,a in enumerate(cats):
        for j,b in enumerate(cats):
            if j<=i: continue
            A,B = category_sets[a], category_sets[b]
            inter = len(A & B)
            union = len(A | B)
            jac = inter/union if union else 0
            if inter>0:
                rows.append(dict(cat_a=a, cat_b=b, inter=inter, union=union, jaccard=jac))
    return pd.DataFrame(rows).sort_values('jaccard', ascending=False)

def plot_category_jaccard_heatmap(category_sets: Dict[str,set],
                                  min_j: float = 0.0,
                                  figsize: Tuple[float,float]=(6,5)):
    cats=list(category_sets.keys())
    n=len(cats)
    J=np.zeros((n,n))
    for i in range(n):
        A=category_sets[cats[i]]
        for j in range(n):
            if i==j: J[i,j]=1; continue
            B=category_sets[cats[j]]
            inter=len(A & B); union=len(A | B)
            J[i,j]= inter/union if union else 0
    if min_j>0:
        M=J.copy()
        M[M<min_j]=0
    else:
        M=J
    plt.figure(figsize=figsize)
    sns.heatmap(M, annot=False, cmap='viridis', xticklabels=cats, yticklabels=cats)
    plt.xticks(rotation=60, ha='right')
    plt.title("Functional category Jaccard")
    plt.tight_layout()
    return J, cats

def plot_category_overlap_network(category_sets: Dict[str,set],
                                  min_jaccard: float = 0.1,
                                  scale: float = 600.0):
    try:
        import networkx as nx
    except ImportError:
        print("Install networkx for network plot: pip install networkx")
        return None
    cats=list(category_sets.keys())
    G=nx.Graph()
    for c in cats:
        G.add_node(c, size=len(category_sets[c]))
    for i in range(len(cats)):
        for j in range(i+1,len(cats)):
            A,B = category_sets[cats[i]], category_sets[cats[j]]
            inter=len(A & B); union=len(A | B)
            if union==0: continue
            jacc=inter/union
            if jacc >= min_jaccard:
                G.add_edge(cats[i], cats[j], weight=jacc, inter=inter)
    pos = nx.spring_layout(G, k=0.9/np.sqrt(max(1,len(G.nodes()))), seed=0)
    plt.figure(figsize=(6,5))
    node_sizes=[scale*(G.nodes[n]['size']/max(1,max(nx.get_node_attributes(G,'size').values()))) for n in G.nodes()]
    edges=G.edges(data=True)
    widths=[2+4*e[2]['weight'] for e in edges]
    nx.draw_networkx_nodes(G,pos,node_size=node_sizes,node_color='steelblue',alpha=0.8)
    nx.draw_networkx_edges(G,pos,width=widths,alpha=0.5)
    nx.draw_networkx_labels(G,pos,font_size=8)
    # edge labels (optional)
    edge_labels={(u,v):f"{d['weight']:.2f}" for u,v,d in edges}
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_size=7)
    plt.title("Functional category overlap network")
    plt.axis('off')
    plt.tight_layout()
    return G

def plot_category_upset(category_sets: Dict[str,set],
                        top_k: int = 20,
                        max_depth: int = 3,
                        min_inter: int = 5):
    import itertools
    cats=list(category_sets.keys())
    sets=[category_sets[c] for c in cats]
    comb_rows=[]
    for r in range(2, min(max_depth, len(cats))+1):
        for comb_idx in itertools.combinations(range(len(cats)), r):
            inter=set.intersection(*(sets[i] for i in comb_idx))
            if len(inter) >= min_inter:
                comb_rows.append((comb_idx,len(inter),inter))
    if not comb_rows:
        print("No intersections for UpSet.")
        return
    comb_rows.sort(key=lambda x: x[1], reverse=True)
    comb_rows=comb_rows[:top_k]
    fig=plt.figure(figsize=(8,4+0.25*len(comb_rows)))
    gs=fig.add_gridspec(2,1,height_ratios=[2,1])
    ax_bar=fig.add_subplot(gs[0,0]); ax_mat=fig.add_subplot(gs[1,0])
    sizes=[r[1] for r in comb_rows]
    ax_bar.bar(range(len(sizes)), sizes, color='#444')
    ax_bar.set_ylabel("Intersection ROIs"); ax_bar.set_xticks([])
    ax_bar.set_title("Functional category intersections (UpSet)")
    for x,v in enumerate(sizes):
        ax_bar.text(x, v, v, ha='center', va='bottom', fontsize=7)
    for x,(comb,sz,_) in enumerate(comb_rows):
        ys=list(comb)
        for y in ys:
            ax_mat.scatter(x,y, s=70, color='tab:blue')
        ax_mat.plot([x,x],[min(ys),max(ys)], color='tab:blue', lw=2)
    ax_mat.set_yticks(range(len(cats))); ax_mat.set_yticklabels(cats, fontsize=8)
    ax_mat.invert_yaxis()
    ax_mat.set_xlabel("Intersection rank")
    plt.tight_layout()
    return fig

def functional_category_workflow(M, catalog,
                                 *,
                                 chunks_filter=None,
                                 min_component_size=5,
                                 min_cat_size=8,
                                 min_jaccard_edge=0.15):
    descriptors = compute_component_functional_descriptors(
        M, catalog,
        chunks_filter=chunks_filter,
        onset_window=(-0.2,0.4),
        smoothing_sigma=0.6
    )
    descriptors = categorize_components(descriptors)
    cat_sets = build_functional_category_sets(
        catalog, descriptors,
        min_size=min_cat_size,
        min_component_size=min_component_size
    )
    print("Categories:", {k:len(v) for k,v in cat_sets.items()})
    df_overlap = category_overlap_table(cat_sets)
    print(df_overlap.head(12))
    plot_category_jaccard_heatmap(cat_sets)
    plot_category_overlap_network(cat_sets, min_jaccard=min_jaccard_edge)
    plot_category_upset(cat_sets)
    return dict(descriptors=descriptors, category_sets=cat_sets, overlap=df_overlap)

# --- Example usage (after catalog with signed groups integrated) -------------
# fc = functional_category_workflow(M, catalog, chunks_filter=None,
#                                   min_component_size=6,
#                                   min_cat_size=10,
#                                   min_jaccard_edge=0.18)
# fc['descriptors'].head()



































# --- Venn diagram helpers (ONLY useful for 2–3 sets; else prefer UpSet) -----------------
# Requires: pip install matplotlib-venn
def _collect_signed_sets_for_venn(signed_groups_by_chunk,
                                  *,
                                  chunks=None,
                                  mode='pos',          # 'pos'|'neg'|'both'
                                  min_size=5,
                                  dedup_jaccard=0.97):
    """
    Return dict label->set limited to selected chunks (positive / negative / union).
    Greedy Jaccard dedup to avoid near-identical duplicates.
    """
    if chunks is None:
        chunks = list(signed_groups_by_chunk.keys())
    raw = {}
    for ch in chunks:
        for g in signed_groups_by_chunk[ch]:
            base = f"{ch}:c{g['comp']}"
            roi_idx = g['roi_idx']
            if mode == 'pos':
                S = set(roi_idx[g['positive_mask']])
                suffix = "_pos"
            elif mode == 'neg':
                S = set(roi_idx[g['negative_mask']])
                suffix = "_neg"
            else:
                S = set(roi_idx[g['positive_mask']]) | set(roi_idx[g['negative_mask']])
                suffix = "_both"
            if len(S) >= min_size:
                raw[base + suffix] = S
    # Deduplicate
    keep = {}
    for lbl, s in raw.items():
        dup = False
        for k2, s2 in keep.items():
            inter = len(s & s2)
            if inter == 0:
                continue
            jac = inter / len(s | s2)
            if jac >= dedup_jaccard:
                dup = True
                break
        if not dup:
            keep[lbl] = s
    return keep

def plot_signed_group_venn(signed_groups_by_chunk,
                           *,
                           select_labels=None,  # explicit list of labels (must match collected keys)
                           select_contains=None,# list of substrings; first 2–3 matches used
                           chunks=None,
                           mode='pos',
                           min_size=5,
                           dedup_jaccard=0.97,
                           ax=None,
                           title=None):
    """
    Plot Venn for exactly 2 or 3 ROI sets.
    select_labels: explicit full labels (e.g. ['isi:c2_pos','choice_start:c3_pos'])
    select_contains: alternative – substrings used to pick first matching labels
    """
    try:
        from matplotlib_venn import venn2, venn3
    except ImportError:
        print("Install matplotlib-venn: pip install matplotlib-venn")
        return None
    sets_dict = _collect_signed_sets_for_venn(signed_groups_by_chunk,
                                              chunks=chunks,
                                              mode=mode,
                                              min_size=min_size,
                                              dedup_jaccard=dedup_jaccard)
    if not sets_dict:
        print("[venn] no sets collected.")
        return None
    labels = list(sets_dict.keys())
    # Selection
    chosen = []
    if select_labels:
        for lbl in select_labels:
            if lbl in sets_dict:
                chosen.append(lbl)
    elif select_contains:
        for pattern in select_contains:
            hit = next((l for l in labels if pattern in l and l not in chosen), None)
            if hit:
                chosen.append(hit)
    else:
        # fallback: pick largest 2–3 sets
        sizes = sorted(((len(sets_dict[l]), l) for l in labels), reverse=True)
        chosen = [l for _, l in sizes[:3]]
    if len(chosen) < 2:
        print("[venn] need at least 2 sets after selection.")
        return None
    if len(chosen) > 3:
        chosen = chosen[:3]
    data = [sets_dict[l] for l in chosen]
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
    else:
        fig = ax.figure
    if len(data) == 2:
        venn2(subsets=(data[0], data[1]),
              set_labels=[chosen[0], chosen[1]], ax=ax)
    else:
        venn3(subsets=(data[0], data[1], data[2]),
              set_labels=[chosen[0], chosen[1], chosen[2]], ax=ax)
    ax.set_title(title or f"Venn ({mode})")
    # Print numeric overlaps for clarity
    print("[venn] sizes:")
    for l, s in zip(chosen, data):
        print(f"  {l}: {len(s)}")
    if len(data) == 3:
        a,b,c = data
        print("  pair overlaps:",
              f"{len(a & b)} (1∩2), {len(a & c)} (1∩3), {len(b & c)} (2∩3)")
        print("  triple overlap:", len(a & b & c))
    else:
        a,b = data
        print("  overlap:", len(a & b))
    return fig

# Example usage (after signed_pruned ready):
# plot_signed_group_venn(signed_pruned,
#                        chunks=['isi','choice_start'],
#                        mode='pos',
#                        select_contains=['isi:c0','choice_start:c1'])
# plot_signed_group_venn(signed_pruned,
#                        chunks=['isi','choice_start','lick_start'],
#                        mode='both',
#                        select_labels=




























# --- POS / NEG OVERLAP DIAGNOSTICS -------------------------------------------------------
def compute_pos_neg_overlap_matrices(signed_groups: List[Dict[str, Any]],
                                     metric: str = 'jaccard') -> Dict[str, Any]:
    """
    Build overlap matrices across components:
      P_vs_N[i,j] = overlap( pos(i), neg(j) )
      P_vs_P[i,j] = overlap( pos(i), pos(j) )
      N_vs_N[i,j] = overlap( neg(i), neg(j) )

    metric: 'jaccard' (|∩|/|∪|) or 'count' (|∩|).

    Note: pos & neg subsets within the SAME component are disjoint by construction
          (so diagonal of P_vs_N will be 0). Use this to see cross‑sign reuse.
    """
    comps = sorted({g['comp'] for g in signed_groups})
    idx_map = {c:i for i,c in enumerate(comps)}
    n = len(comps)
    P_vs_N = np.zeros((n,n), float)
    P_vs_P = np.zeros((n,n), float)
    N_vs_N = np.zeros((n,n), float)

    pos_sets = {}
    neg_sets = {}
    for g in signed_groups:
        c = g['comp']
        ri = g['roi_idx']
        pos_sets[c] = set(ri[g['positive_mask']])
        neg_sets[c] = set(ri[g['negative_mask']])

    def _ov(a: set, b: set):
        if metric == 'count':
            return len(a & b)
        # jaccard
        if not a and not b: return 1.0
        if not a or not b: return 0.0
        inter = len(a & b)
        if inter == 0: return 0.0
        return inter / len(a | b)

    for i,c_i in enumerate(comps):
        for j,c_j in enumerate(comps):
            P_vs_N[i,j] = _ov(pos_sets[c_i], neg_sets[c_j])
            P_vs_P[i,j] = _ov(pos_sets[c_i], pos_sets[c_j])
            N_vs_N[i,j] = _ov(neg_sets[c_i], neg_sets[c_j])

    return dict(comps=comps,
                P_vs_N=P_vs_N,
                P_vs_P=P_vs_P,
                N_vs_N=N_vs_N,
                metric=metric)

def plot_pos_neg_overlap_heatmaps(overlaps: Dict[str, Any],
                                  vmin: float = 0.0,
                                  vmax: Optional[float] = None,
                                  figsize: Tuple[float,float] = (10,3.2),
                                  cmap: str = 'magma'):
    """
    Quick side‑by‑side heatmaps (PvsN, PvsP, NvsN).
    """
    comps = overlaps['comps']
    P_vs_N = overlaps['P_vs_N']
    P_vs_P = overlaps['P_vs_P']
    N_vs_N = overlaps['N_vs_N']
    if vmax is None:
        vmax = np.nanmax([P_vs_N, P_vs_P, N_vs_N])
        if vmax <= 0: vmax = 1.0
    fig, axes = plt.subplots(1,3, figsize=figsize, sharey=True)
    for ax, mat, title in zip(axes,
                              [P_vs_N, P_vs_P, N_vs_N],
                              ['Pos vs Neg', 'Pos vs Pos', 'Neg vs Neg']):
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(comps))); ax.set_yticks(range(len(comps)))
        ax.set_xticklabels(comps, rotation=90, fontsize=7)
        ax.set_yticklabels(comps, fontsize=7)
    axes[0].set_ylabel("Component (rows)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label=overlaps['metric'])
    fig.suptitle(f"Component overlap matrices ({overlaps['metric']})", fontsize=10)
    plt.tight_layout(rect=[0,0,1,0.95])
    return fig

def summarize_cross_sign_reuse(signed_groups: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    For each component quantify how many of its positive (negative) ROIs
    appear in negative (positive) subsets of OTHER components.

    Columns:
      comp, n_pos, n_neg, pos_in_other_neg, pos_in_other_neg_frac,
      neg_in_other_pos, neg_in_other_pos_frac
    """
    comps = sorted({g['comp'] for g in signed_groups})
    pos_sets = {}
    neg_sets = {}
    for g in signed_groups:
        c = g['comp']
        ri = g['roi_idx']
        pos_sets[c] = set(ri[g['positive_mask']])
        neg_sets[c] = set(ri[g['negative_mask']])

    rows = []
    for c in comps:
        p = pos_sets[c]; n = neg_sets[c]
        # unions of opposite sign in OTHER comps
        other_neg_union = set().union(*(neg_sets[o] for o in comps if o != c))
        other_pos_union = set().union(*(pos_sets[o] for o in comps if o != c))
        pos_in_o_neg = len(p & other_neg_union)
        neg_in_o_pos = len(n & other_pos_union)
        rows.append(dict(
            comp=c,
            n_pos=len(p),
            n_neg=len(n),
            pos_in_other_neg=pos_in_o_neg,
            pos_in_other_neg_frac=(pos_in_o_neg/len(p) if p else 0.0),
            neg_in_other_pos=neg_in_o_pos,
            neg_in_other_pos_frac=(neg_in_o_pos/len(n) if n else 0.0)
        ))
    return pd.DataFrame(rows)

# Convenience wrapper
def analyze_pos_neg_overlap_for_chunk(signed_groups_by_chunk: Dict[str,List[Dict[str,Any]]],
                                      chunk: str,
                                      metric: str = 'jaccard'):
    """
    One call:
      overlaps dict (matrices)
      heatmap figure
      cross‑sign reuse DataFrame
    """
    if chunk not in signed_groups_by_chunk:
        raise KeyError(f"Chunk '{chunk}' not found.")
    sg = signed_groups_by_chunk[chunk]
    overlaps = compute_pos_neg_overlap_matrices(sg, metric=metric)
    fig = plot_pos_neg_overlap_heatmaps(overlaps)
    reuse_df = summarize_cross_sign_reuse(sg)
    print(reuse_df)
    return dict(overlaps=overlaps, figure=fig, reuse=reuse_df)

# Example:
# ov = analyze_pos_neg_overlap_for_chunk(signed_pruned, 'choice_start', metric='jaccard')
# ov_cnt = analyze_pos_neg_overlap_for_chunk(signed_pruned, 'choice_start', metric='count')



















def find_cross_sign_rois(signed_groups: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Return DataFrame of ROIs that appear positive in one component and negative in another.
    Columns: roi, pos_comp, neg_comp
    """
    pos_map = {}
    neg_map = {}
    for g in signed_groups:
        c = g['comp']
        idx = g['roi_idx']
        pos = idx[g['positive_mask']]
        neg = idx[g['negative_mask']]
        for r in pos: pos_map.setdefault(r, set()).add(c)
        for r in neg: neg_map.setdefault(r, set()).add(c)
    rows = []
    for roi, pos_cs in pos_map.items():
        if roi in neg_map:
            for pc in sorted(pos_cs):
                for nc in sorted(neg_map[roi]):
                    rows.append(dict(roi=int(roi), pos_comp=pc, neg_comp=nc))
    return pd.DataFrame(rows)

# Example:
# flip_df = find_cross_sign_rois(signed_pruned['choice_start'])
# print(flip_df)
























def summarize_sign_flips(signed_groups: List[Dict[str, Any]],
                         A: np.ndarray,
                         loading_eps: float = 0.0) -> pd.DataFrame:
    """
    Build table of ROIs that appear with both signs across components.
    loading_eps > 0 will zero out (ignore) loadings with abs < eps before sign evaluation.
    Returns columns:
      roi, pos_comps, neg_comps, n_pos, n_neg, max_pos_loading, max_neg_loading
    """
    # Map ROI -> sets of comps by sign
    pos_map = {}
    neg_map = {}
    for g in signed_groups:
        c = g['comp']
        roi_idx = g['roi_idx']
        signs = g['signs']
        # Optionally mask near-zero for robustness
        if loading_eps > 0:
            keep_mask = np.abs(A[roi_idx, c]) >= loading_eps
            roi_idx = roi_idx[keep_mask]
            signs = signs[keep_mask]
        for r, s in zip(roi_idx, signs):
            if s > 0:
                pos_map.setdefault(r, set()).add(c)
            elif s < 0:
                neg_map.setdefault(r, set()).add(c)
    rows = []
    mixed = sorted(set(pos_map.keys()) & set(neg_map.keys()))
    for roi in mixed:
        p = sorted(pos_map[roi])
        n = sorted(neg_map[roi])
        # Peak absolute loading for context
        max_p = float(np.max(np.abs([A[roi, c] for c in p]))) if p else 0.0
        max_n = float(np.max(np.abs([A[roi, c] for c in n]))) if n else 0.0
        rows.append(dict(
            roi=int(roi),
            pos_comps=p,
            neg_comps=n,
            n_pos=len(p),
            n_neg=len(n),
            max_pos_loading=max_p,
            max_neg_loading=max_n
        ))
    return pd.DataFrame(rows)

def plot_sign_flip_roi(M: Dict[str, Any],
                       chunk: str,
                       signed_groups: List[Dict[str, Any]],
                       A: np.ndarray,
                       roi: int,
                       smooth_sigma: float = 0.6):
    """
    For a single ROI that flips sign, overlay its trial-averaged trace for each
    component where it is positive vs negative (different colors).
    """
    if chunk not in M['chunks']:
        print(f"[plot_sign_flip_roi] chunk '{chunk}' missing.")
        return
    X = M['chunks'][chunk]['data']      # (trials, rois, time)
    t = M['chunks'][chunk]['time']
    is_short = M['is_short'].astype(bool)
    # Determine membership
    pos_comps = []
    neg_comps = []
    for g in signed_groups:
        c = g['comp']
        idx = g['roi_idx']
        signs = g['signs']
        where = np.where(idx == roi)[0]
        if where.size:
            if signs[where[0]] > 0:
                pos_comps.append(c)
            elif signs[where[0]] < 0:
                neg_comps.append(c)
    if not pos_comps and not neg_comps:
        print(f"[plot_sign_flip_roi] ROI {roi} not in signed groups.")
        return
    roi_trace_trials = X[:, roi, :]  # (trials, time)
    mean_all = safe_nanmean(roi_trace_trials, axis=0)
    mean_short = safe_nanmean(roi_trace_trials[is_short], axis=0)
    mean_long = safe_nanmean(roi_trace_trials[~is_short], axis=0)
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        for arr_name in ['mean_all','mean_short','mean_long']:
            arr = locals()[arr_name]
            if arr is not None and np.any(np.isfinite(arr)):
                locals()[arr_name][:] = gaussian_filter1d(arr, smooth_sigma)
    plt.figure(figsize=(5.0,3.2))
    plt.plot(t, mean_all, color='k', lw=1.4, label='All')
    if np.any(np.isfinite(mean_short)):
        plt.plot(t, mean_short, color='tab:blue', lw=1.0, label='Short')
    if np.any(np.isfinite(mean_long)):
        plt.plot(t, mean_long, color='tab:orange', lw=1.0, label='Long')
    # Annotate comps & loadings
    text = []
    for c in pos_comps:
        text.append(f"+c{c}")
    for c in neg_comps:
        text.append(f"-c{c}")
    plt.title(f"{chunk} ROI {roi} sign flip ({', '.join(text)})", fontsize=10)
    plt.xlabel("Time (s)")
    plt.ylabel("z-dF/F (mean)")
    plt.axhline(0, color='k', lw=0.5)
    plt.legend(fontsize=7, frameon=False, ncol=3)
    plt.tight_layout()

# Example quick usage after detecting flips:
# flips_choice = summarize_sign_flips(signed_pruned['choice_start'], time_results['choice_start']['final_A'])
# print(flips_choice)
# plot_sign_flip_roi(M, 'choice_start', signed_pruned['choice_start'],
#                    time_results['choice_start']['final_A'], roi=1932)

















# ===================== INTERSECTION PARTITIONING (UNIQUE / WITHIN / ACROSS) =====================

def build_intersection_partitions(catalog: Dict[str, Any],
                                  *,
                                  group_order: Optional[List[str]] = None,
                                  abbrev: Optional[Dict[str, str]] = None,
                                  min_size_within: Optional[int] = None,
                                  min_size_across: Optional[int] = None,
                                  drop_small: bool = True,
                                  report_dropped_max: int = 15,
                                  return_dropped: bool = True
                                  ) -> Dict[str, Any]:
    """
    Partition ROIs into three non-overlapping catalogs derived from a signed catalog:
      A. Non-intersectional (unique): ROI in exactly one (group, component) (ignoring sign).
      B. Within-group intersectional: ROI in ≥2 components of same group only.
      C. Across-group intersectional: ROI spanning components from ≥2 groups.

    Input catalog:
      catalog['incidence']: (n_rois, n_components) bool
      catalog['comp_labels']: e.g. "isi:comp3_pos" / "choice_start:comp2_neg"
      Components may have separate pos / neg columns (treated as same base component for partition logic).

    Output (dict):
      {
        'non_intersectional': {'incidence','comp_labels','meta'},
        'within_group_intersectional': {...},
        'across_group_intersectional': {...},
        'assignment': DataFrame per ROI,
        'within_meta': DataFrame,
        'across_meta': DataFrame,
        'dropped': {'within': DataFrame, 'across': DataFrame} (if return_dropped)
      }

    Label formats:
      Within-group fused:  <group>:c_2_5_7_pos / _neg / _ambig
      Across-group fused:  <groupA+groupB+...>:c_groupA-2_5|groupB-1|groupC-4_9_pos  (order = group_order or sorted)
      Sign subsets are exclusive (_pos, _neg, _ambig). _ambig only emitted if non-empty.

    Ambiguity rule:
      For a fused entity, an ROI is:
        pos   if it appears ONLY in positive source columns (no negative) among contributing components
        neg   if ONLY negative
        ambig if appears in both pos and neg source columns

    Size thresholding:
      * Thresholds applied on UNION (all signs) size per fused entity.
      * If drop_small and min_size_within / min_size_across set, fused entities below threshold removed
        from their catalogs; affected ROIs still accounted in assignment (category unchanged).
      * Dropped metadata reported & returned.

    Parameters
    ----------
    group_order : list[str] | None
        Trial progression order for stable across-group naming. If None -> alphabetical of involved groups.
    abbrev : dict[str,str] | None
        Optional mapping to shorten group tokens in across-group component segment labels.
    min_size_within / min_size_across : int | None
        Minimum fused UNION size to keep (within / across). None disables filter.
    drop_small : bool
        Apply thresholds if True.
    report_dropped_max : int
        Max lines printed listing dropped fused labels per class.
    return_dropped : bool
        Include dropped fused meta tables in result.

    Notes
    -----
    * No ROI discarded; classification partition is exhaustive.
    * If an ROI appears in both pos & neg of SAME component, that counts as 1 component for classification
      but sign sets record both signs for ambiguity resolution.
    """
    inc = catalog['incidence'].astype(bool)
    labels = catalog['comp_labels']
    n_rois, n_cols = inc.shape

    # -------- Parse component columns into base units --------
    # label pattern assumed: "<group>:comp<idx>_<sign>" OR "<group>:comp<idx>" fallback
    import re
    parsed = []
    base_key_to_col_sign = {}  # (group, comp_idx, sign) -> column index
    base_membership_cols = {}  # (group, comp_idx) -> list of column indices (pos/neg)
    comp_re = re.compile(r'^(?P<group>[^:]+):comp(?P<comp>\d+)(?:_(?P<sign>pos|neg))?$')
    for j, lbl in enumerate(labels):
        m = comp_re.match(lbl)
        if not m:
            # Skip non-standard labels silently
            continue
        group = m.group('group')
        comp_idx = int(m.group('comp'))
        sign = m.group('sign') or 'unspecified'
        parsed.append((j, group, comp_idx, sign))
        base_key_to_col_sign[(group, comp_idx, sign)] = j
        base_membership_cols.setdefault((group, comp_idx), []).append(j)

    if not parsed:
        raise ValueError("No parsable component labels found (expected pattern '<group>:comp<k>[_pos|_neg]').")

    groups_all = sorted({g for _, g, _, _ in parsed})
    if group_order:
        # keep provided order, append any missing groups at end
        seen_order = [g for g in group_order if g in groups_all]
        tail = [g for g in groups_all if g not in seen_order]
        group_order_use = seen_order + tail
    else:
        group_order_use = groups_all

    group_rank = {g: i for i, g in enumerate(group_order_use)}

    # -------- Per-ROI base membership & sign collection --------
    # For classification we treat (group, comp_idx) as base component ignoring sign.
    roi_base_members = [set() for _ in range(n_rois)]
    roi_sign_map = [{} for _ in range(n_rois)]  # (group, comp_idx) -> set({'pos','neg'})
    for (group, comp_idx), cols in base_membership_cols.items():
        col_block = inc[:, cols]  # (n_rois, n_sign_cols)
        any_member = col_block.any(axis=1)
        # Determine which sign each ROI had
        sign_cols = {}
        for c in cols:
            # determine sign
            lbl = labels[c]
            if lbl.endswith('_pos'):
                sign_cols.setdefault('pos', []).append(c)
            elif lbl.endswith('_neg'):
                sign_cols.setdefault('neg', []).append(c)
            else:
                sign_cols.setdefault('unspecified', []).append(c)
        # For each ROI with membership add base component and signs
        for roi in np.where(any_member)[0]:
            roi_base_members[roi].add((group, comp_idx))
            sign_set = roi_sign_map[roi].setdefault((group, comp_idx), set())
            for sgn, scols in sign_cols.items():
                if sgn == 'unspecified':
                    continue
                # If ROI present in any column with that sign -> record
                if inc[roi, scols].any():
                    sign_set.add(sgn)

    # -------- Classify ROIs --------
    categories = []  # 'unique' | 'within' | 'across'
    for roi in range(n_rois):
        bases = roi_base_members[roi]
        if not bases:
            categories.append('none')
            continue
        groups_roi = {g for g, _ in bases}
        if len(bases) == 1:
            categories.append('unique')
        else:
            if len(groups_roi) == 1:
                categories.append('within')
            else:
                categories.append('across')

    # -------- Containers for fused sets --------
    # Non-intersectional: keep original signed columns but filter to ROIs with category 'unique'
    unique_mask = np.array([cat == 'unique' for cat in categories])
    non_cols = []
    non_labels = []
    for j, lbl in enumerate(labels):
        col = inc[:, j] & unique_mask
        if col.any():
            non_cols.append(col)
            non_labels.append(lbl)
    if non_cols:
        non_inc = np.column_stack(non_cols)
    else:
        non_inc = np.zeros((n_rois, 0), bool)

    # Within-group fused
    within_fused = {}  # key: (group, frozenset(comp_indices)) -> dict with roi indices & sign sets
    for roi in range(n_rois):
        if categories[roi] != 'within':
            continue
        bases = sorted(roi_base_members[roi], key=lambda x: x[1])
        group = bases[0][0]
        comp_indices = tuple(sorted(ci for _, ci in bases))
        key = (group, frozenset(comp_indices))
        entry = within_fused.setdefault(key, {'rois': [], 'signs': []})
        # Determine aggregated sign sets over these components for ROI
        pos_flag = False
        neg_flag = False
        for bc in bases:
            sset = roi_sign_map[roi].get(bc, set())
            if 'pos' in sset:
                pos_flag = True
            if 'neg' in sset:
                neg_flag = True
        if pos_flag and neg_flag:
            sign_cat = 'ambig'
        elif pos_flag:
            sign_cat = 'pos'
        elif neg_flag:
            sign_cat = 'neg'
        else:
            # fallback: treat as pos if unspecified
            sign_cat = 'pos'
        entry['rois'].append(roi)
        entry['signs'].append(sign_cat)

    # Across-group fused
    across_fused = {}  # key: (tuple(groups_ordered), tuple(sorted per-group component sets)) -> entry
    for roi in range(n_rois):
        if categories[roi] != 'across':
            continue
        bases = roi_base_members[roi]
        groups_roi = sorted({g for g, _ in bases}, key=lambda g: group_rank.get(g, 1e9))
        # Map group -> component indices for this ROI
        per_group = {}
        for g, cidx in bases:
            per_group.setdefault(g, set()).add(cidx)
        # Build key components (frozensets)
        group_comp_tuple = tuple((g, frozenset(sorted(per_group[g]))) for g in groups_roi)
        key = group_comp_tuple  # already ordered structure
        entry = across_fused.setdefault(key, {'rois': [], 'signs': []})
        # Determine sign ambiguity across all contributing components
        pos_flag = False
        neg_flag = False
        for g, comp_set in per_group.items():
            for cidx in comp_set:
                sset = roi_sign_map[roi].get((g, cidx), set())
                if 'pos' in sset:
                    pos_flag = True
                if 'neg' in sset:
                    neg_flag = True
        if pos_flag and neg_flag:
            sign_cat = 'ambig'
        elif pos_flag:
            sign_cat = 'pos'
        elif neg_flag:
            sign_cat = 'neg'
        else:
            sign_cat = 'pos'
        entry['rois'].append(roi)
        entry['signs'].append(sign_cat)

    def _apply_threshold_and_build(fused_dict, within: bool):
        """
        Returns (incidence, labels, meta_df, dropped_df)
        """
        out_cols = []
        out_labels = []
        meta_rows = []
        dropped_rows = []
        for key, entry in fused_dict.items():
            rois = np.array(entry['rois'], int)
            signs = entry['signs']
            union_size = rois.size
            # Determine threshold
            thresh = min_size_within if within else min_size_across
            if drop_small and thresh is not None and union_size < thresh:
                dropped_rows.append(dict(
                    label_key=str(key),
                    union_size=union_size,
                    n_pos=signs.count('pos'),
                    n_neg=signs.count('neg'),
                    n_ambig=signs.count('ambig')
                ))
                continue
            # Partition ROI indices by sign category
            idx_pos = rois[[s == 'pos' for s in signs]]
            idx_neg = rois[[s == 'neg' for s in signs]]
            idx_amb = rois[[s == 'ambig' for s in signs]]
            # Build base name
            if within:
                group, comp_set_fs = key
                comp_list = sorted(comp_set_fs)
                comp_token = "c_" + "_".join(str(c) for c in comp_list)
                base_label = f"{group}:{comp_token}"
            else:
                # Across: key = tuple( (group, frozenset(comp_idx_set)), ... ) already ordered
                group_names = [g for g, _ in key]
                group_concat = "+".join(group_names)
                segs = []
                for g, comps_fs in key:
                    comp_list = sorted(comps_fs)
                    gtok = abbrev.get(g, g) if abbrev else g
                    segs.append(f"{gtok}-" + "_".join(str(c) for c in comp_list))
                comp_token = "c_" + "|".join(segs)
                base_label = f"{group_concat}:{comp_token}"
            # Emit columns (_pos/_neg/_ambig if non-empty)
            def add_subset(arr, suffix):
                if arr.size == 0:
                    return
                col = np.zeros(n_rois, bool)
                col[arr] = True
                out_cols.append(col)
                out_labels.append(base_label + f"_{suffix}")
            add_subset(idx_pos, 'pos')
            add_subset(idx_neg, 'neg')
            add_subset(idx_amb, 'ambig')
            meta_rows.append(dict(
                base_label=base_label,
                class_type='within' if within else 'across',
                union_size=union_size,
                n_pos=idx_pos.size,
                n_neg=idx_neg.size,
                n_ambig=idx_amb.size,
                frac_ambig=(idx_amb.size / union_size if union_size else 0.0),
                groups=(group if within else "+".join([g for g, _ in key])),
                component_sets=(sorted(comp_set_fs) if within else {g: sorted(list(cs)) for g, cs in key})
            ))
        if out_cols:
            inc_mat = np.column_stack(out_cols)
        else:
            inc_mat = np.zeros((n_rois, 0), bool)
        meta_df = pd.DataFrame(meta_rows)
        dropped_df = pd.DataFrame(dropped_rows)
        return inc_mat, out_labels, meta_df, dropped_df

    within_inc, within_labels, within_meta, within_dropped = _apply_threshold_and_build(within_fused, within=True)
    across_inc, across_labels, across_meta, across_dropped = _apply_threshold_and_build(across_fused, within=False)

    # -------- ROI assignment table --------
    assign_rows = []
    for roi in range(n_rois):
        cat = categories[roi]
        if cat == 'unique':
            # fetch original label(s) where ROI present (only one base component but may have 1-2 sign cols)
            cols = np.where(inc[roi])[0]
            # filter to base components that appear only once ignoring sign
            # Use first matching original label for primary_label
            role_labels = [labels[c] for c in cols]
            primary_label = role_labels[0] if role_labels else ""
            fused_label = primary_label
        elif cat == 'within':
            fused_label = None
            # identify fused key
            bases = sorted(roi_base_members[roi], key=lambda x: x[1])
            group = bases[0][0]
            comp_list = sorted(ci for _, ci in bases)
            comp_token = "c_" + "_".join(str(c) for c in comp_list)
            base_label = f"{group}:{comp_token}"
            # Determine sign category for ROI
            pos_flag = any(('pos' in roi_sign_map[roi].get((group, ci), set())) for ci in comp_list)
            neg_flag = any(('neg' in roi_sign_map[roi].get((group, ci), set())) for ci in comp_list)
            if pos_flag and neg_flag:
                suffix = "_ambig"
            elif pos_flag:
                suffix = "_pos"
            elif neg_flag:
                suffix = "_neg"
            else:
                suffix = "_pos"
            fused_label = base_label + suffix
        elif cat == 'across':
            per_group = {}
            for g, ci in roi_base_members[roi]:
                per_group.setdefault(g, set()).add(ci)
            ordered_groups = sorted(per_group.keys(), key=lambda g: group_rank.get(g, 1e9))
            segs = []
            for g in ordered_groups:
                comp_list = sorted(per_group[g])
                gtok = abbrev.get(g, g) if abbrev else g
                segs.append(f"{gtok}-" + "_".join(str(c) for c in comp_list))
            comp_token = "c_" + "|".join(segs)
            group_concat = "+".join(ordered_groups)
            base_label = f"{group_concat}:{comp_token}"
            pos_flag = any('pos' in roi_sign_map[roi].get((g, ci), set()) for g in per_group for ci in per_group[g])
            neg_flag = any('neg' in roi_sign_map[roi].get((g, ci), set()) for g in per_group for ci in per_group[g])
            if pos_flag and neg_flag:
                suffix = "_ambig"
            elif pos_flag:
                suffix = "_pos"
            elif neg_flag:
                suffix = "_neg"
            else:
                suffix = "_pos"
            fused_label = base_label + suffix
        else:
            fused_label = ""
        assign_rows.append(dict(
            roi=roi,
            category=cat,
            fused_label=fused_label
        ))
    assignment_df = pd.DataFrame(assign_rows)

    # -------- Construct catalog dicts --------
    non_catalog = dict(
        incidence=non_inc,
        comp_labels=non_labels,
        meta=pd.DataFrame(dict(label=non_labels,
                               size=[non_inc[:, i].sum() for i in range(non_inc.shape[1])],
                               class_type='non_intersectional'))
    )
    within_catalog = dict(
        incidence=within_inc,
        comp_labels=within_labels,
        meta=within_meta
    )
    across_catalog = dict(
        incidence=across_inc,
        comp_labels=across_labels,
        meta=across_meta
    )

    # -------- Reporting --------
    if drop_small:
        if min_size_within is not None and not within_dropped.empty:
            print(f"[intersection] Dropped {len(within_dropped)} within-group fused (<{min_size_within}).")
            for _, r in within_dropped.head(report_dropped_max).iterrows():
                print(f"  within drop {r.label_key} union={r.union_size} pos={r.n_pos} neg={r.n_neg} ambig={r.n_ambig}")
        if min_size_across is not None and not across_dropped.empty:
            print(f"[intersection] Dropped {len(across_dropped)} across-group fused (<{min_size_across}).")
            for _, r in across_dropped.head(report_dropped_max).iterrows():
                print(f"  across drop {r.label_key} union={r.union_size} pos={r.n_pos} neg={r.n_neg} ambig={r.n_ambig}")

    out = dict(
        non_intersectional=non_catalog,
        within_group_intersectional=within_catalog,
        across_group_intersectional=across_catalog,
        assignment=assignment_df,
        within_meta=within_meta,
        across_meta=across_meta
    )
    if return_dropped:
        out['dropped'] = dict(within=within_dropped, across=across_dropped)
    return out

# Convenience wrapper to quickly get all three catalogs (returns only catalogs)
def get_intersection_catalogs(catalog: Dict[str, Any],
                              **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    parts = build_intersection_partitions(catalog, **kwargs)
    return (parts['non_intersectional'],
            parts['within_group_intersectional'],
            parts['across_group_intersectional'])

# Example usage (after you have 'catalog' with signed *_pos/_neg columns integrated):
# parts = build_intersection_partitions(
#     catalog,
#     group_order=['trial_start','isi','start_flash_1','end_flash_1','start_flash_2','end_flash_2','choice_start','lick_start'],
#     min_size_within=5,
#     min_size_across=5
# )
# non_cat = parts['non_intersectional']
# within_cat = parts['within_group_intersectional']
# across_cat = parts['across_group_intersectional']
# print(non_cat['comp_labels'][:5])
# print(within_cat['comp_labels'][:5])
# print(across_cat['comp_labels'][:5])
# Raster plotting can now use e.g. within_cat['incidence'] with labels within_cat['comp_labels'].




























if __name__ == '__main__':
    print('Starting ROI motif labeling analysis...\n')
    
    
    
    




# %%


# path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_simplex_20250529_2afc-379/sid_imaging_segmented_data.pkl'
path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/sid_imaging_segmented_data.pkl'



path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/trial_start.pkl'
import pickle
with open(path, 'rb') as f:
    trial_data_trial_start = pickle.load(f)   # one object back (e.g., a dict)  
print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_trial_start.keys())}")

path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/trial_isi.pkl'
import pickle
with open(path, 'rb') as f:
    trial_data_isi = pickle.load(f)   # one object back (e.g., a dict)  
print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_isi.keys())}")

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



# trial_data_trial_start
# trial_data_isi              -0.2 to max isi
# trial_data_start_flash_1    -0.25 to 0.5
# trial_data_end_flash_1      -0.25 to 0.6
# trial_data_start_flash_2    -0.25 to 0.6
# trial_data_end_flash_2      -0.25 to 0.6
# trial_data_choice_start     -0.6 to 0.6
# trial_data_lick_start       -0.4 to 0.4


# %%


# %%
# # =====================================================================
# # MASTER EXPLORATORY WORKFLOW (interactive – no monolithic runner)
# # From raw per‑event pickles -> M -> CP components -> signed ROI
# # groups -> diagnostics -> plots -> annotations.
# #
# # Replace ad‑hoc code blocks below with this ordered, documented recipe.
# # Execute sequentially in a notebook / REPL; comment out sections as needed.
# # =====================================================================

# # --------------------------- CONFIG ----------------------------------
# CHUNK_KEYS        = ['isi','start_flash_1','end_flash_1','start_flash_2',
#                      'end_flash_2','choice_start','lick_start']
# RANKS             = [6,8,10]          # candidate CP ranks (time-domain)
# MIN_ABS_CORR      = 0.65              # split-half loading stability threshold
# SIGNED_Q          = 0.10              # quantile for |loading| selection (pos+neg)
# PHASE_BINS        = 80                # phase-resample bins for ISI (0..1)
# MIN_NEG_FRAC      = 0.12              # pruning: min fraction of ROIs for negative subset
# MIN_NEG_REL_MAG   = 0.30              # pruning: min (neg_mag / pos_mag)
# OVERLAY_CHUNKS    = CHUNK_KEYS        # which chunks to include in overlay plots
# OVERLAY_SMOOTH    = 0.6               # smoothing σ (s) for overlay traces
# OVERLAY_ENERGY    = 'combined'        # 'combined' or 'pos_all'
# DEDUP_PHASE_CORR  = 0.97              # optional correlation dedup for phase matrix (None to disable)
# EVENTS_FULL       = ('F1_on','F1_off','F2_on','F2_off','choice_start','lick_start')
# EXPORT_ROI_ROLES  = "roi_roles.csv"   # set None to skip export
# DEBUG_PRINTS      = True              # verbose diagnostics

# # --------------------------- INPUT -----------------------------------
# # Assumes the per-chunk pickles already loaded as:
# #   trial_data_trial_start, trial_data_isi, trial_data_start_flash_1, ...
# # If not: load them before running the steps below.

# chunks_raw = {
#     'trial_start':   trial_data_trial_start,
#     'isi':           trial_data_isi,
#     'start_flash_1': trial_data_start_flash_1,
#     'end_flash_1':   trial_data_end_flash_1,
#     'start_flash_2': trial_data_start_flash_2,
#     'end_flash_2':   trial_data_end_flash_2,
#     'choice_start':  trial_data_choice_start,
#     'lick_start':    trial_data_lick_start
# }

# # =====================================================================
# # STEP 1. Build session structure M (set mask_after_isi=False to keep post‑ISI data)
# # =====================================================================
# M = build_M_from_chunks(chunks_raw, cfg={'isi': {'mask_after_isi': True}},
#                         primary_chunk='trial_start')
# summarize_M(M)

# # =====================================================================
# # STEP 2. Time‑domain CP decomposition per chunk (rank sweep + pruning + stability)
# # Produces: time_results[ch] with 'final_A','final_B','final_roi_sets','stability_vec',...
# # =====================================================================
# time_results = run_time_domain_analysis(M, CHUNK_KEYS, RANKS, MIN_ABS_CORR)

# if DEBUG_PRINTS:
#     for ch in CHUNK_KEYS:
#         print(f"[time CP] {ch} final components={time_results[ch]['final_A'].shape[1]} "
#               f"stability={np.round(time_results[ch]['stability_vec'],3)}")

# # Optional quick temporal factor plots
# for ch in CHUNK_KEYS:
#     plot_component_temporals(time_results[ch], f"{ch} temporal factors")

# # =====================================================================
# # STEP 3. Build component catalog (time-domain only) + overlap diagnostics
# # catalog: {'incidence': (n_rois,n_comp), 'comp_labels': [...], 'catalog': list[dict]}
# # =====================================================================
# catalog = build_component_catalog(M, time_results)
# summarize_component_overlaps(catalog)

# # =====================================================================
# # STEP 4. Phase-normalized ISI CP
# # Adds 'isi_phase' chunk, runs CP, extracts + orients signed groups twice (phase then full trace)
# # Outputs res_phase (with res_phase['factors']['A']), signed_phase (list of signed group dicts)
# # =====================================================================
# M['chunks'].pop('isi_phase', None)
# PHASE_BINS_OLD = PHASE_BINS
# PHASE_BINS = PHASE_BINS          # (kept for clarity; global constant used in run_phase_analysis)
# res_phase, signed_phase = run_phase_analysis(M)

# # =====================================================================
# # STEP 5. Build signed (pos/neg) groups for ALL chunks (time + phase)
# # Orientation: trace_mean (time chunks) + preserved dual orientation for phase.
# # signed_all[ch] -> list of signed group dicts with fields:
# #   roi_idx, signs, positive_mask, negative_mask, comp
# # =====================================================================
# combined_results = dict(time_results)
# combined_results['isi_phase'] = {'final_A': res_phase['factors']['A']}
# signed_all = build_signed_groups_for_chunks(M, combined_results,
#                                             q=SIGNED_Q,
#                                             use_final=True,
#                                             orient=True,
#                                             orient_method='trace_mean')
# # Replace automatic phase with original (dual-oriented) version
# signed_all['isi_phase'] = signed_phase

# # =====================================================================
# # STEP 6. Redundancy summary across signed groups (pairwise Jaccard)
# # mode='pos'|'neg'|'both'
# # =====================================================================
# summarize_signed_group_overlap(signed_all, mode='pos', min_jaccard=0.7)

# # =====================================================================
# # STEP 7. Integrate signed groups into catalog (adds *_pos / *_neg columns)
# # Updates catalog['incidence'] and catalog['comp_labels'] in place
# # =====================================================================
# catalog = integrate_signed_groups_into_catalog(catalog, M, signed_all,
#                                                label_prefix='', skip_existing=True)

# # =====================================================================
# # STEP 8. Assess negative subset strength (loading + trace metrics)
# # assess_signed_groups returns DataFrame per chunk
# # =====================================================================
# neg_assessment = {}
# for ch in CHUNK_KEYS + ['isi_phase']:
#     A_mat = res_phase['factors']['A'] if ch == 'isi_phase' else time_results[ch]['final_A']
#     sg    = signed_all[ch]
#     df_load  = assess_signed_groups(A_mat, sg, metric='loading', print_table=False)
#     df_trace = assess_signed_groups(A_mat, sg, metric='trace_mean', M=M,
#                                     chunk=('isi_phase' if ch=='isi_phase' else ch),
#                                     print_table=False)
#     neg_assessment[ch] = dict(loading=df_load, trace=df_trace)
#     if DEBUG_PRINTS:
#         weak = df_load['weak'].sum()
#         print(f"[neg assess] {ch}: weak (loading metric) {weak}/{len(df_load)}")

# # =====================================================================
# # STEP 9. Prune weak negative subsets (convert to single-sided if below thresholds)
# # Produces signed_pruned[ch]
# # =====================================================================
# signed_pruned = {}
# for ch in CHUNK_KEYS + ['isi_phase']:
#     A_mat = res_phase['factors']['A'] if ch == 'isi_phase' else time_results[ch]['final_A']
#     sg_copy = [dict(**g) for g in signed_all[ch]]
#     prune_weak_negatives(A_mat, sg_copy,
#                          min_neg_frac=MIN_NEG_FRAC,
#                          min_neg_rel_mag=MIN_NEG_REL_MAG,
#                          drop_entire=False)
#     signed_pruned[ch] = sg_copy

# # =====================================================================
# # STEP 10. Multi-component overlay plots (All/Short/Long) with energy-based alpha
# # Use signed_pruned to hide pruned neg sides. Set debug_energies=True to print energies.
# # =====================================================================
# overlay_figs = {}
# for ch in OVERLAY_CHUNKS:
#     overlay_figs[ch] = plot_signed_components_overlay(
#         M, signed_pruned[ch],
#         trial_chunk='trial_start',
#         chunk_name=ch,
#         smooth_sigma=OVERLAY_SMOOTH,
#         energy_mode=OVERLAY_ENERGY,
#         sort_by_energy=True,
#         debug_energies=False
#     )
# # Phase overlay on full trial timeline
# overlay_figs['isi_phase'] = plot_signed_components_overlay(
#     M, signed_pruned['isi_phase'],
#     trial_chunk='trial_start',
#     chunk_name='isi_phase',
#     smooth_sigma=OVERLAY_SMOOTH,
#     energy_mode=OVERLAY_ENERGY
# )

# # =====================================================================
# # STEP 11. Per-component stacked signed traces (time chunk OR full trial)
# #   a) chunk time axis: plot_signed_components_bigstack
# #   b) full trial axis: plot_signed_component_mean_traces_stack / plot_phase_components_on_full_trial
# # =====================================================================
# # Example: time chunk 'choice_start'
# plot_signed_components_bigstack(M, time_results['choice_start']['base'],
#                                 chunk='choice_start',
#                                 signed_groups=signed_pruned['choice_start'],
#                                 smooth_sigma=0.6)
# # Example: all components over full trial timeline
# plot_signed_component_mean_traces_stack(M, signed_pruned['choice_start'],
#                                         trial_chunk='trial_start',
#                                         smooth_sigma=0.6)
# # Example: phase groups over full trial
# plot_phase_components_on_full_trial(M, signed_pruned['isi_phase'],
#                                     trial_chunk='trial_start',
#                                     smooth_sigma=0.6)

# # =====================================================================
# # STEP 12. Phase component grid (Positive vs Negative per condition)
# # Optionally deduplicate highly correlated Positive traces.
# # =====================================================================
# plot_isi_phase_components_matrix(
#     M, res_phase, signed_pruned['isi_phase'],
#     comps=None,
#     events=('F2_on','F2_off'),
#     layout='grid',
#     smooth_sigma=0.6,
#     dedup_corr_thresh=DEDUP_PHASE_CORR
# )

# # Optional: cluster phase components & replot only representatives
# phase_clust = cluster_isi_phase_components(signed_pruned['isi_phase'],
#                                            jaccard_thresh=0.85, mode='pos')
# phase_reps  = build_representative_phase_groups(signed_pruned['isi_phase'], phase_clust)
# plot_isi_phase_components_matrix(M, res_phase, phase_reps,
#                                  comps=phase_clust['keep_components'],
#                                  events=('F2_on','F2_off'),
#                                  layout='grid', smooth_sigma=0.6)

# # =====================================================================
# # STEP 13. Catalog-based group visualizations (raster + traces)
# # plot_all_groups_v2 uses catalog component ROI sets (NOT signed pos/neg split).
# # =====================================================================
# # Global value-range normalization across all groups
# plot_all_groups_v2(M, catalog, groups=None, cfg=dict(
#     sort_mode='onset',
#     onset_window=(0.5, 2.8),
#     onset_z=1.0,
#     smoothing_sigma=0.8,
#     events=list(EVENTS_FULL),
#     figsize=(7.2,8.0)
# ), compute_global_vlim=True)

# # Example subset (e.g., choice & lick related)
# subset_labels = [l for l in catalog['comp_labels']
#                  if l.startswith('choice_start:') or l.startswith('lick_start:')]
# plot_all_groups_v2(M, catalog, groups=subset_labels,
#                    cfg=dict(sort_mode='onset',
#                             onset_window=(-0.3,0.6),
#                             onset_z=1.0,
#                             smoothing_sigma=0.8,
#                             events=list(EVENTS_FULL)),
#                    compute_global_vlim=True)

# # =====================================================================
# # STEP 14. Component uniqueness metrics & ROI annotation table
# # =====================================================================
# uni_metrics = component_uniqueness_metrics(catalog)
# roi_roles_df = build_roi_annotation_dataframe(M, catalog, uni_metrics)
# if EXPORT_ROI_ROLES:
#     export_roi_annotations(roi_roles_df, uni_metrics, EXPORT_ROI_ROLES)

# # =====================================================================
# # STEP 15. Example: inspect a specific phase component’s ROI sets & overlaps
# # =====================================================================
# df_phase_sets = list_isi_phase_component_rois(
#     M, res_phase, signed_pruned['isi_phase'],
#     overlap_mode='pos',
#     min_jaccard_report=0.4
# )
# # Retrieve positive ROIs for component 0 (if exists)
# if not df_phase_sets.empty:
#     try:
#         comp0_pos = df_phase_sets[(df_phase_sets.comp==0) & (df_phase_sets.group=='pos')].iloc[0].roi_indices
#         print("Phase comp 0 (pos) example ROI indices:", comp0_pos[:10])
#     except Exception:
#         pass

# # =====================================================================
# # (OPTIONAL) STEP 16. Adaptive re-run of a single chunk with relaxed pruning
# # =====================================================================
# # relaxed_res = rerun_single_chunk_with_relaxed_pruning(M, 'start_flash_1',
# #                                                       ranks=[4,6,8],
# #                                                       min_abs_corr=0.55)
# # print("Relaxed start_flash_1 components:",
# #       relaxed_res['relaxed']['A_relaxed'].shape[1])

# # =====================================================================
# # END OF MASTER WORKFLOW
# # =====================================================================








# %%

"""
============================ SID EXPLORATORY PIPELINE (ORDERED RECIPE) ============================
This block documents the canonical, linear sequence of analysis calls from raw chunk pickles
to component interpretation. Copy & execute sequentially; comment out optional branches.

Legend:
  CORE = recommended sequential step
  OPT  = optional / diagnostic / refinement

CONFIG (edit before running)
----------------------------
CHUNK_KEYS       = [...]       # time-domain windows to CP
RANKS            = [6,8,10]    # candidate CP ranks
MIN_ABS_CORR     = 0.65        # split-half abs loading corr threshold (stability prune)
SIGNED_Q         = 0.10        # quantile for top-|loading| selection (pos/neg sets)
PHASE_BINS       = 80          # ISI phase resample bins
MIN_NEG_FRAC     = 0.12        # prune weak neg: min fraction of ROIs that are negative
MIN_NEG_REL_MAG  = 0.30        # prune weak neg: min (neg_mag / pos_mag)
OVERLAY_SMOOTH   = 0.6         # σ (samples) for overlay smoothing
OVERLAY_ENERGY   = 'combined'  # 'combined' | 'pos_all'
DEDUP_PHASE_CORR = 0.97        # phase grid: drop later comps corr>thr (None to skip)
EXPORT_ROI_ROLES = "roi_roles.csv" or None
DEBUG            = True

--------------------------------------------------------------------------------
CORE Step 0. Prepare raw chunk dicts (one per alignment)
--------------------------------------------------------------------------------
chunks_raw = {
  'trial_start': trial_data_trial_start,
  'isi':          trial_data_isi,
  'start_flash_1':trial_data_start_flash_1,
  'end_flash_1':  trial_data_end_flash_1,
  'start_flash_2':trial_data_start_flash_2,
  'end_flash_2':  trial_data_end_flash_2,
  'choice_start': trial_data_choice_start,
  'lick_start':   trial_data_lick_start
}

--------------------------------------------------------------------------------
CORE Step 1. Build session structure
--------------------------------------------------------------------------------
M = build_M_from_chunks(
      chunks_raw,
      cfg={'isi': {'mask_after_isi': True}},   # set False to keep post-ISI samples
      primary_chunk='trial_start'
)
summarize_M(M)

Key outputs:
  M['chunks'][chunk]['data']  shape=(trials, rois, time)
  M['time'] / M['roi_traces'] baseline global axis (trial_start)

--------------------------------------------------------------------------------
CORE Step 2. Time-domain CP across chunks
--------------------------------------------------------------------------------
time_results = run_time_domain_analysis(
    M,
    chunk_keys=CHUNK_KEYS,
    ranks=RANKS,
    min_abs_corr=MIN_ABS_CORR
)

Each time_results[ch]:
  final_A, final_B, final_roi_sets, stability_vec, base, stability, rank

Inspect (diagnostics):
  time_results[ch]['stability_vec']  # abs corr pre stability prune

--------------------------------------------------------------------------------
CORE Step 3. Component catalog + overlaps
--------------------------------------------------------------------------------
catalog = build_component_catalog(M, time_results)
summarize_component_overlaps(catalog)   # top Jaccard pairs (redundancy hint)

Catalog fields:
  catalog['incidence'] (n_rois, n_components)
  catalog['comp_labels'] ["chunk:idx", ...]

--------------------------------------------------------------------------------
CORE Step 4. Phase-normalized ISI CP (adds 'isi_phase')
--------------------------------------------------------------------------------
M['chunks'].pop('isi_phase', None)  # reset if re-running
res_phase, signed_phase = run_phase_analysis(M)
# res_phase['factors']['A'] shape (rois, R_phase)

--------------------------------------------------------------------------------
CORE Step 5. Signed (pos/neg) groups for ALL chunks
--------------------------------------------------------------------------------
combined_results = dict(time_results)
combined_results['isi_phase'] = {'final_A': res_phase['factors']['A']}

signed_all = build_signed_groups_for_chunks(
    M,
    combined_results,
    q=SIGNED_Q,
    use_final=True,
    orient=True,             # uses orient_method below
    orient_method='trace_mean'
)
# Keep the dual-oriented phase result (phase + trace) from run_phase_analysis:
signed_all['isi_phase'] = signed_phase

Each signed_all[ch] = list of dicts:
  {'comp','roi_idx','signs','positive_mask','negative_mask'}

--------------------------------------------------------------------------------
OPT Step 6. Redundancy across signed groups
--------------------------------------------------------------------------------
summarize_signed_group_overlap(
    signed_all,
    mode='pos',          # 'pos' | 'neg' | 'both'
    min_jaccard=0.7
)

--------------------------------------------------------------------------------
CORE Step 7. Integrate signed groups into catalog
--------------------------------------------------------------------------------
catalog = integrate_signed_groups_into_catalog(
    catalog, M, signed_all,
    label_prefix='',       # optional e.g. 'signed:'
    skip_existing=True
)
# New columns: <chunk>:compK_pos / _neg

--------------------------------------------------------------------------------
CORE Step 8. Assess negative subset strength
--------------------------------------------------------------------------------
neg_assessment = {}
for ch in CHUNK_KEYS + ['isi_phase']:
    A_mat = res_phase['factors']['A'] if ch=='isi_phase' else time_results[ch]['final_A']
    sg    = signed_all[ch]
    neg_assessment[ch] = dict(
        loading = assess_signed_groups(A_mat, sg, metric='loading',     print_table=False),
        trace   = assess_signed_groups(A_mat, sg, metric='trace_mean',  M=M, chunk=('isi_phase' if ch=='isi_phase' else ch), print_table=False)
    )

Metrics:
  n_pos, n_neg, frac_neg, pos_mag, neg_mag, rel_mag, weak flag.

--------------------------------------------------------------------------------
CORE Step 9. Prune weak negatives (convert to single-sided)
--------------------------------------------------------------------------------
signed_pruned = {}
for ch in CHUNK_KEYS + ['isi_phase']:
    A_mat = res_phase['factors']['A'] if ch=='isi_phase' else time_results[ch]['final_A']
    sg_copy = [dict(**g) for g in signed_all[ch]]
    prune_weak_negatives(
        A_mat, sg_copy,
        min_neg_frac=MIN_NEG_FRAC,
        min_neg_rel_mag=MIN_NEG_REL_MAG,
        drop_entire=False    # set True to fully drop weak components
    )
    signed_pruned[ch] = sg_copy

--------------------------------------------------------------------------------
CORE Step 10. Overlay multi-component traces (energy → alpha)
--------------------------------------------------------------------------------
overlay_figs = {}
for ch in CHUNK_KEYS:
    overlay_figs[ch] = plot_signed_components_overlay(
        M, signed_pruned[ch],
        trial_chunk='trial_start',
        chunk_name=ch,
        smooth_sigma=OVERLAY_SMOOTH,
        energy_mode=OVERLAY_ENERGY,    # 'combined' or 'pos_all'
        sort_by_energy=True,
        debug_energies=False
    )
overlay_figs['isi_phase'] = plot_signed_components_overlay(
    M, signed_pruned['isi_phase'],
    trial_chunk='trial_start',
    chunk_name='isi_phase',
    smooth_sigma=OVERLAY_SMOOTH,
    energy_mode=OVERLAY_ENERGY
)

--------------------------------------------------------------------------------
OPT Step 11. Detailed per-component stacks
--------------------------------------------------------------------------------
# Time-axis native window:
plot_signed_components_bigstack(
    M, time_results['choice_start']['base'],
    chunk='choice_start',
    signed_groups=signed_pruned['choice_start'],
    smooth_sigma=0.6
)
# Full-trial axis:
plot_signed_component_mean_traces_stack(M, signed_pruned['choice_start'], trial_chunk='trial_start', smooth_sigma=0.6)
# Phase groups projected onto full trial:
plot_phase_components_on_full_trial(M, signed_pruned['isi_phase'], trial_chunk='trial_start', smooth_sigma=0.6)

--------------------------------------------------------------------------------
CORE Step 12. Phase component matrix (All / Short / Long)
--------------------------------------------------------------------------------
plot_isi_phase_components_matrix(
    M, res_phase, signed_pruned['isi_phase'],
    comps=None,                    # or subset list
    events=('F2_on','F2_off'),
    layout='grid',                 # 'grid' | 'per_component'
    smooth_sigma=0.6,
    dedup_corr_thresh=DEDUP_PHASE_CORR,  # None to disable
    corr_mode='all'
)

--------------------------------------------------------------------------------
OPT Phase clustering / representatives
--------------------------------------------------------------------------------
phase_clust = cluster_isi_phase_components(
    signed_pruned['isi_phase'],
    jaccard_thresh=0.85,
    mode='pos'          # 'pos'|'neg'|'both'
)
phase_reps = build_representative_phase_groups(signed_pruned['isi_phase'], phase_clust)
plot_isi_phase_components_matrix(
    M, res_phase, phase_reps,
    comps=phase_clust['keep_components'],
    events=('F2_on','F2_off'),
    layout='grid',
    smooth_sigma=0.6
)

--------------------------------------------------------------------------------
CORE Step 13. Catalog visualizations (ROI rasters + mean traces)
--------------------------------------------------------------------------------
# Global vlim & all groups
plot_all_groups_v2(
    M, catalog,
    groups=None,
    cfg=dict(
        sort_mode='onset',          # 'index'|'onset'
        onset_window=(0.5,2.8),     # onset search window
        onset_z=1.0,                # z-threshold
        smoothing_sigma=0.8,
        events=['F1_on','F1_off','F2_on','F2_off','choice_start','lick_start'],
        figsize=(7.2,8.0)
    ),
    compute_global_vlim=True
)

# Subset example (choice + lick)
subset_labels = [l for l in catalog['comp_labels'] if l.startswith('choice_start:') or l.startswith('lick_start:')]
plot_all_groups_v2(M, catalog, groups=subset_labels, cfg=dict(sort_mode='onset', onset_window=(-0.3,0.6), onset_z=1.0, smoothing_sigma=0.8, events=['F1_on','F1_off','F2_on','F2_off','choice_start','lick_start']), compute_global_vlim=True)

--------------------------------------------------------------------------------
CORE Step 14. Component uniqueness + per-ROI roles
--------------------------------------------------------------------------------
uni_metrics = component_uniqueness_metrics(catalog)
roi_roles_df = build_roi_annotation_dataframe(M, catalog, uni_metrics)
if EXPORT_ROI_ROLES:
    export_roi_annotations(roi_roles_df, uni_metrics, EXPORT_ROI_ROLES)

--------------------------------------------------------------------------------
OPT Additional utilities
--------------------------------------------------------------------------------
# Relaxed re-run for a single short window:
relaxed = rerun_single_chunk_with_relaxed_pruning(M, 'start_flash_1', ranks=[4,6,8], min_abs_corr=0.55)
# Merge highly overlapping catalog components (union):
merged_catalog = merge_high_overlap_components(catalog, overlap_thresh=0.6)
# Keep only selected alignment families:
canonical_catalog = filter_catalog_by_label_prefix(catalog, keep_prefixes=['isi:', 'choice_start:', 'start_flash_2:', 'lick_start:'])

================================================================================
Parameter quick reference (most-used)
================================================================================
run_time_domain_analysis:
  ranks (list[int]) | min_abs_corr (float)

run_phase_analysis:
  uses global PHASE_BINS; internally sets pre_zscored=True for isi_phase.

build_signed_groups_for_chunks:
  q (quantile of |loadings|) | orient (bool) | orient_method {'trace_mean','loading','trace_phase'}

assess_signed_groups:
  metric {'loading','trace_mean','trace_energy'}
  min_neg_frac / min_neg_rel_mag thresholds → weak flag

plot_signed_components_overlay:
  energy_mode {'combined','pos_all'} | min_alpha/max_alpha | annotate_mode {'end','peak'}

plot_isi_phase_components_matrix:
  layout {'grid','per_component'} | dedup_corr_thresh (float|None) | corr_mode {'all','pos','neg'}

cluster_isi_phase_components:
  jaccard_thresh | mode {'pos','neg','both'} | representative {'largest_pos','largest_total','first'}

rerun_single_chunk_with_relaxed_pruning:
  ranks | min_abs_corr (lower for exploratory) → adds 'relaxed' section

================================================================================
END OF ORDERED RECIPE
================================================================================
"""






































# %%
# Build M with / without ISI masking
# 1. Load raw chunk pickles
chunks_raw = {
        'trial_start': trial_data_trial_start,
        'isi': trial_data_isi,
        'start_flash_1': trial_data_start_flash_1,
        'end_flash_1': trial_data_end_flash_1,
        'start_flash_2': trial_data_start_flash_2,
        'end_flash_2': trial_data_end_flash_2,
        'choice_start': trial_data_choice_start,
        'lick_start': trial_data_lick_start
    }



# 2. Build M (with ISI truncation on by default; set mask_after_isi=False to disable)
M = build_M_from_chunks(chunks_raw, cfg={'isi': {'mask_after_isi': True}})
summarize_M(M)

# 3. Time-domain CP across selected chunks
chunk_keys = ['isi','start_flash_1','end_flash_1','start_flash_2','end_flash_2','choice_start','lick_start']
time_results = run_time_domain_analysis(M, chunk_keys, RANKS, MIN_ABS_CORR)


# 4. Catalog + overlaps
catalog = build_component_catalog(M, time_results)
summarize_component_overlaps(catalog)


M['chunks'].pop('isi_phase', None)
res_phase, signed_groups = run_phase_analysis(M)
# After you have: time_results (dict from run_time_domain_analysis) and res_phase / signed_groups
# 1. Combine results dict (add phase under its chunk key)
combined_results = dict(time_results)
combined_results['isi_phase'] = {'final_A': res_phase['factors']['A']}  # minimal wrapper for build function
# 2. Build signed groups for each chunk
signed_all = build_signed_groups_for_chunks(M, combined_results, q=0.10, use_final=True, orient=True)
# 3. (Optional) redundancy check across all pos subsets
summarize_signed_group_overlap(signed_all, mode='pos', min_jaccard=0.7)
# 4. Integrate into catalog (adds *_pos / *_neg columns for every component)
catalog = integrate_signed_groups_into_catalog(catalog, M, signed_all, label_prefix='', skip_existing=True)



for chunk in chunk_keys:
    # then choose a chunk’s signed group list, e.g. signed_all['choice_start']
    plot_signed_components_overlay(M, signed_all[chunk], 
                                   trial_chunk='trial_start', 
                                   chunk_name=chunk, 
                                   debug_energies=True, 
                                   energy_flat_tol=1e-5, 
                                   flat_energy_alpha='max')



plot_signed_components_overlay(
    M, signed_all['choice_start'],
    trial_chunk='trial_start',
    chunk_name='choice_start',
    debug_energies=True,
    energy_flat_tol=1e-5,
    flat_energy_alpha='max'
)


plot_signed_components_overlay(..., energy_mode='pos_all')


plot_signed_components_overlay(M, signed_all['choice_start'], trial_chunk='trial_start')

# 5. Plot temporal factors for each chunk
for ch in chunk_keys:
    plot_component_temporals(time_results[ch], f"{ch} final temporals")
plt.show()


# %%

# 6. Phase-normalized ISI analysis (kept separate)
M['chunks'].pop('isi_phase', None)
res_phase, signed_groups = run_phase_analysis(M)
plt.show()



# ...existing code...
res_phase, signed_groups = run_phase_analysis(M)
catalog = integrate_phase_components_into_catalog(catalog, res_phase, signed_groups)



# 7. Component uniqueness + ROI annotation export
metrics = component_uniqueness_metrics(catalog)
roi_df = build_roi_annotation_dataframe(M, catalog, metrics)
export_roi_annotations(roi_df, metrics, ANNOT_CSV)

print("Done.")

# %%

# ---- Advanced group plots (label-specific onset windows) ----
# Define per-prefix onset search windows (relative to trial_start timeline)
prefix_onset_windows = {
    'isi':          (0.5, 2.8),
    'start_flash_1':(0.0, 0.45),
    'end_flash_1':  (0.0, 0.55),
    'start_flash_2':(0.0, 0.55),
    'end_flash_2':  (0.0, 0.55),
    'choice_start': (-0.3, 0.3),
    'lick_start':   (-0.25, 0.25)
}

base_cfg = dict(
    sort_mode='onset',
    onset_z=1.0,
    smoothing_sigma=0.8,
    events=['F1_on','F1_off','F2_on','F2_off','choice_start','lick_start'],
    figsize=(7.2,8.0)
)

# Precompute global vlim once
plot_all_groups_v2(M, catalog, groups=[], cfg=base_cfg, compute_global_vlim=True)
global_vlim = base_cfg.get('vlim')  # after the pre-scan base_cfg is updated

labels = catalog['comp_labels']
inc = catalog['incidence']

for lbl in labels:
    prefix = lbl.split(':', 1)[0]
    roi_idx = np.flatnonzero(inc[:, labels.index(lbl)])
    cfg = dict(base_cfg)
    cfg['vlim'] = global_vlim
    cfg['onset_window'] = prefix_onset_windows.get(prefix, (0.5, 2.8))
    plot_group_overview_v2(M, roi_idx, lbl, cfg=cfg)

plt.show()


'''
matplotlib.pyplot.close()
'''


# %%



# After res_phase, signed_groups = run_phase_analysis(M)
# Plot all components in a grid with F2_on & F2_off markers, deduplicate near-identical (>0.97 corr)
plot_isi_phase_components_matrix(
    M, res_phase, signed_groups,
    comps=None,
    events=('F2_on','F2_off'),
    show_negative=True,
    smooth_sigma=0.6,
    layout='grid',
    dedup_corr_thresh=1.0
)

# Plot only selected components 0,2,5 without dedup, per-component vertical figures
plot_isi_phase_components_matrix(
    M, res_phase, signed_groups,
    comps=None,
    events=('F2_on',),
    layout='per_component'
)


# # Plot only selected components 0,2,5 without dedup, per-component vertical figures
# plot_isi_phase_components_matrix(
#     M, res_phase, signed_groups,
#     comps=[0,2,5],
#     events=('F2_on',),
#     layout='per_component'
# )



# %%


roi_labels = M.get('roi_labels', None)  # if you have them
df_phase_sets = list_isi_phase_component_rois(
    M, res_phase, signed_groups,
    roi_labels=roi_labels,
    overlap_mode='pos',          # or 'both'
    min_jaccard_report=0.4
)
# Inspect full lists for a specific component:
comp3_pos = df_phase_sets[(df_phase_sets.comp==3) & (df_phase_sets.group=='pos')].iloc[0].roi_indices




# %%




# After: res_phase, signed_groups = run_phase_analysis(M)
clust = cluster_isi_phase_components(signed_groups, jaccard_thresh=0.85, mode='pos')
signed_groups_reduced = build_representative_phase_groups(signed_groups, clust)

# Plot only representatives
plot_isi_phase_components_matrix(
    M, res_phase, signed_groups_reduced,
    comps=clust['keep_components'],
    events=('F2_on','F2_off'),
    layout='grid'
)









# %%



# After time_results['isi'] (or any chunk) is ready:
res_isi = time_results['isi']['base']  # or time_results['isi'] if factors at top-level differ
A_isi = res_isi['factors']['A']
signed_isi = extract_signed_roi_groups(A_isi, q=0.10)
orient_signed_groups(A_isi, signed_isi, M=M, chunk='isi', method='trace_mean')
fig = plot_signed_components_bigstack(M, res_isi, chunk='isi',
                                      signed_groups=signed_isi,
                                      comps=None,  # all
                                      events=('F1_on','F1_off','F2_on','F2_off'))
plt.show()





# %%


# After res_phase, signed_groups = run_phase_analysis(M)
fig = plot_phase_components_on_full_trial(
    M,
    signed_groups,
    comps=[0,1,2,3],              # or None for all
    trial_chunk='trial_start',
    events=('F1_on','F1_off','F2_on','F2_off','choice_start','lick_start'),
    smooth_sigma=0.6
)
plt.show()



res_isi = time_results['isi']['base']
sg_isi = extract_signed_roi_groups(res_isi['factors']['A'], q=0.10)
orient_signed_groups(res_isi['factors']['A'], sg_isi, M=M, chunk='isi', method='trace_mean')
fig = plot_phase_components_on_full_trial(M, sg_isi, comps=None)
plt.show()






# %%

# After: res_phase, signed_groups = run_phase_analysis(M)
fig = plot_signed_component_mean_traces_stack(
    M, signed_groups,
    trial_chunk='trial_start',
    comps=[0,1,2,3],
    events=('F1_on','F1_off','F2_on','F2_off','choice_start','lick_start'),
    smooth_sigma=0.6
)
plt.show()



res_isi = time_results['isi']['base']
sg_isi = extract_signed_roi_groups(res_isi['factors']['A'], q=0.10)
orient_signed_groups(res_isi['factors']['A'], sg_isi, M=M, chunk='isi', method='trace_mean')
fig = plot_signed_component_mean_traces_stack(M, sg_isi, trial_chunk='trial_start')
plt.show()




# %%

for chunk in chunk_keys:
    print(chunk)
    A_choice = time_results[chunk]['final_A']
    sg_choice = signed_all[chunk]
    assess_signed_groups(A_choice, sg_choice)
    
    
A_phase = res_phase['factors']['A']
sg_phase = signed_groups  # from run_phase_analysis
assess_signed_groups(A_phase, sg_phase, metric='trace_mean', M=M, chunk='isi_phase') 
    
    
assess_signed_groups(A_phase, sg_phase, metric='trace_energy', M=M, chunk='isi_phase',
                     min_neg_frac=0.08, min_neg_rel_mag=0.3)
    
    
    
A_choice = time_results['choice_start']['final_A']
sg_choice = signed_all['choice_start']
assess_signed_groups(A_choice, sg_choice)
sg_choice_pruned = prune_weak_negatives(A_choice, sg_choice, min_neg_frac=0.12, min_neg_rel_mag=0.3)
















# %%





# After res_phase, signed_groups = run_phase_analysis(M)
fig = plot_signed_components_overlay(
    M, signed_groups,
    trial_chunk='trial_start',
    comps=[0,1,2,3,4,5,6,7],
    energy_mode='combined',
    smooth_sigma=0.6,
    sort_by_energy=True,
    annotate=True,
    annotate_mode='end'
)
plt.show()






























# %%

subset = [l for l in catalog['comp_labels'] if l.startswith('choice_start:') or l.startswith('lick_start:')]
cfg = dict(sort_mode='onset', onset_window=(-0.3,0.6), onset_z=1.0, smoothing_sigma=0.8)
plot_all_groups_v2(M, catalog, groups=subset, cfg=cfg, compute_global_vlim=True)
plt.show()







# %%
# try:
#     M = build_M_from_chunks({
#         'trial_start': trial_data_trial_start,
#         'isi': trial_data_isi,
#         'start_flash_1': trial_data_start_flash_1,
#         'end_flash_1': trial_data_end_flash_1,
#         'start_flash_2': trial_data_start_flash_2,
#         'end_flash_2': trial_data_end_flash_2,
#         'choice_start': trial_data_choice_start,
#         'lick_start': trial_data_lick_start
#     })
#     summarize_M(M)
# except Exception as e:
#     print(f"ERROR: {e}")
#     import traceback
#     traceback.print_exc()





# %%


try:
    chunks_to_run = ['isi','choice_start','start_flash_1','end_flash_1','start_flash_2','end_flash_2','lick_start']
    all_res = run_pipeline_for_chunks(M, chunks_to_run, ranks=[6,8,10], min_abs_corr=0.65)

    catalog = build_component_catalog(M, all_res)
    summarize_component_overlaps(catalog)

    for ch in chunks_to_run:
        plot_component_temporals(all_res[ch], f"{ch} final temporals")
    plt.show()
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()   
    
    
    
    
# %%
    
    
    
# ---- Advanced group plots (global config) ----
cfg = dict(
    sort_mode='onset',          # 'index' or 'onset'
    onset_window=(0.5, 2.8),    # time range (s) to search for first rise
    onset_z=1.0,                # SD threshold above baseline
    smoothing_sigma=0.8,        # temporal smoothing (frames units)
    events=['F1_on','F1_off','F2_on','F2_off','choice_start','lick_start'],
    figsize=(7.2, 8.0)
)

# Plot every catalog component
plot_all_groups_v2(M, catalog, cfg=cfg, compute_global_vlim=True)
plt.show()





    
    

















    
    
# %%



try:
    
     # After building 'catalog'
    plot_all_groups(M, catalog, smoothing_sigma=0.8, condition_split=True)
    plt.show()   
    
    
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()    
    
    
    
    
    
    
# %%    
    
try:    
    roi_roles = build_roi_role_table(M, catalog)
    metrics = component_uniqueness_metrics(catalog)
    print("Component uniqueness (label, size, max_jacc_other, exclusivity):")
    for i, lbl in enumerate(metrics['labels']):
        print(f"{lbl}: size={metrics['size'][i]}, maxJ={metrics['max_jacc_other'][i]:.3f}, excl={metrics['exclusivity'][i]:.3f}")
    print("Multi-role ROI count:", roi_roles['multi_role_mask'].sum())
    
    
    merged_catalog = merge_high_overlap_components(catalog, overlap_thresh=0.6)
    print("Merged components:", merged_catalog['comp_labels'])
    
    
    
    sf1_relaxed = rerun_single_chunk_with_relaxed_pruning(M, 'start_flash_1', ranks=[4,6,8], min_abs_corr=0.55)
    print("Relaxed start_flash_1 components:", sf1_relaxed['relaxed']['A_relaxed'].shape[1])
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()


# %%


try:            
    metrics = component_uniqueness_metrics(catalog)
    roi_df = build_roi_annotation_dataframe(M, catalog, metrics)
    export_roi_annotations(roi_df, metrics, "roi_roles.csv")

    # Optionally keep only one F2 alignment (e.g., start_flash_2 and merged forms)
    canonical_catalog = filter_catalog_by_label_prefix(catalog, keep_prefixes=["isi:", "choice_start:", "start_flash_2:", "lick_start:", "end_flash_1:"])
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()    








# %%


# ...existing code...

# %% CP decomposition examples
try:
    # 1. ISI chunk (timing / anticipation)
    cp_isi = run_chunk_cp_pipeline(
        M,
        chunk='isi',
        ranks=[8, 10, 12, 14],
        B=50,
        sample_frac=0.7,
        stratify_isi=True,
        random_state=0
    )
    print(f"[isi] best rank={cp_isi['factors']['rank']} loss={cp_isi['factors']['loss']:.4g}")
    for k, s in enumerate(cp_isi['roi_sets']):
        print(f"  comp {k}: {len(s)} ROIs")
except Exception as e:
    print("ERROR (ISI CP):", e)

try:
    # 2. Choice-aligned chunk (premotor)
    cp_choice = run_chunk_cp_pipeline(
        M,
        chunk='choice_start',
        ranks=[6, 8, 10],
        B=40,
        sample_frac=0.7,
        stratify_isi=False,
        random_state=1
    )
    print(f"[choice] best rank={cp_choice['factors']['rank']} loss={cp_choice['factors']['loss']:.4g}")
    for k, s in enumerate(cp_choice['roi_sets']):
        print(f"  comp {k}: {len(s)} ROIs")
except Exception as e:
    print("ERROR (choice CP):", e)

# Simple visualization of temporal factors for a result
def plot_temporal_factors(res, title):
    Bmat = res['factors']['B']  # (time, components)
    t = res['time']
    plt.figure(figsize=(6, 4))
    for i in range(Bmat.shape[1]):
        plt.plot(t, Bmat[:, i], lw=1, label=f"c{i}")
    plt.axvline(0, color='k', ls='--', lw=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Component (arb)")
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

try:
    plot_temporal_factors(cp_isi, "ISI temporal factors")
    plot_temporal_factors(cp_choice, "Choice temporal factors")
    plt.show()
except Exception as e:
    print("ERROR (plot):", e)

# Extract ROI indices for further rasters
if 'cp_isi' in locals():
    comp0_rois = cp_isi['roi_sets'][0]
    print("Example ISI comp0 ROIs:", comp0_rois[:15])



# %%



# After building M

try:
    cp_isi = run_cp_with_adaptive_sets(
        M,
        chunk='isi',
        ranks=[8,10,12],
        B=50,
        sample_frac=0.7,
        stratify_isi=True,
        random_state=0
    )

 
    # After cp_isi = run_cp_with_adaptive_sets(...)
    stab = split_half_stability(M, 'isi', rank=cp_isi['factors']['rank'],
                                B=60, sample_frac=0.7, random_state=42)
    # Map stability to pruned components (need mapping keep_mask)
    # First get indices of original components kept in pruning:
    kept_idx = np.where(cp_isi['keep_mask'])[0]
    # Build vector of abs stability for those indices (matched pairs):
    # For simplicity pick matched_abs_corr aligned to original order via mapping
    stability_vec = np.zeros(len(kept_idx))
    for k, orig_idx in enumerate(kept_idx):
        # find which matched pair corresponds to orig_idx
        try:
            pos = np.where(stab['row_idx'] == orig_idx)[0]
            if pos.size:
                stability_vec[k] = np.abs(stab['matched_corr'][pos[0]])
        except Exception:
            pass

    A_final, B_final, roi_sets_final, stable_mask = filter_components_by_stability(
        cp_isi['A_pruned'], cp_isi['B_pruned'], cp_isi['pruned_roi_sets'],
        stability_vec, cp_isi['keep_mask'][cp_isi['keep_mask']],
        min_abs_corr=0.65
    )
    # print("[ISI] rank chosen:", cp_isi['factors']['rank'])
    # print("Kept components:", cp_isi['A_pruned'].shape[1])
    # for i, s in enumerate(cp_isi['pruned_roi_sets']):
    #     print(f"  comp {i}: {len(s)} ROIs")    
    print("Final kept components:", A_final.shape[1])
        
    print(f"[choice] best rank={cp_choice['factors']['rank']} loss={cp_choice['factors']['loss']:.4g}")
    for k, s in enumerate(cp_choice['roi_sets']):
        print(f"  comp {k}: {len(s)} ROIs")


    # Simple visualization of temporal factors for a result
    def plot_temporal_factors(res, title):
        Bmat = res['factors']['B']  # (time, components)
        t = res['time']
        plt.figure(figsize=(6, 4))
        for i in range(Bmat.shape[1]):
            plt.plot(t, Bmat[:, i], lw=1, label=f"c{i}")
        plt.axvline(0, color='k', ls='--', lw=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Component (arb)")
        plt.title(title)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
    
    try:
        plot_temporal_factors(cp_isi, "ISI temporal factors")
        plot_temporal_factors(cp_choice, "Choice temporal factors")
        plt.show()
    except Exception as e:
        print("ERROR (plot):", e)         
        
        
except Exception as e:
    print("ERROR adaptive ISI:", e)

try:
    stab = split_half_stability(M, 'isi', rank=cp_isi['factors']['rank'],
                                B=60, sample_frac=0.7, random_state=42)
    print("Split-half matched loading correlations:",
          np.round(stab['matched_corr'], 3))
except Exception as e:
    print("ERROR stability:", e)




# %%


try:

    cp_full_isi = run_cp_full_pipeline(M, 'isi', ranks=[8,10,12], min_abs_corr=0.65)
    print("Final ISI components:", cp_full_isi['final_A'].shape[1])
    print("Per-component abs stability (kept before stability prune):",
        np.round(cp_full_isi['stability_vec'],3))
    for i,s in enumerate(cp_full_isi['final_roi_sets']):
        print(f"  final comp {i}: {len(s)} ROIs")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()




# %%
    
# Run the analysis pipeline

   
   

    
# %%


try:
    print('holder')
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    
    
    
# %%
# Run the analysis with your loaded data
# ...existing code...

# Run the analysis with your loaded data
try:
    print("=== RUNNING EVENT-ALIGNED ROI ANALYSIS ===")
    
    # Prepare event data list in the order specified by cfg
    event_data_list = [
        trial_data_trial_start,     # For full raster/mean/etc plots to inspect resultant groups/behavior
        trial_data_isi,             # For isi-sensitive activity search
        trial_data_start_flash_1,   # F1 onset
        trial_data_end_flash_1,     # F1 offset
        trial_data_start_flash_2,   # F2 onset
        trial_data_end_flash_2,     # F2 offset
        trial_data_choice_start,    # Choice start
        trial_data_lick_start       # Lick start
    ]
    

    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()





# %%
"""
============================ SID EXPLORATORY PIPELINE (ORDERED RECIPE) ============================
This block documents the canonical, linear sequence of analysis calls from raw chunk pickles
to component interpretation. Copy & execute sequentially; comment out optional branches.

Legend:
  CORE = recommended sequential step
  OPT  = optional / diagnostic / refinement

CONFIG (edit before running)
----------------------------
CHUNK_KEYS       = [...]       # time-domain windows to CP
RANKS            = [6,8,10]    # candidate CP ranks
MIN_ABS_CORR     = 0.65        # split-half abs loading corr threshold (stability prune)
SIGNED_Q         = 0.10        # quantile for top-|loading| selection (pos/neg sets)
PHASE_BINS       = 80          # ISI phase resample bins
MIN_NEG_FRAC     = 0.12        # prune weak neg: min fraction of ROIs that are negative
MIN_NEG_REL_MAG  = 0.30        # prune weak neg: min (neg_mag / pos_mag)
OVERLAY_SMOOTH   = 0.6         # σ (samples) for overlay smoothing
OVERLAY_ENERGY   = 'combined'  # 'combined' | 'pos_all'
DEDUP_PHASE_CORR = 0.97        # phase grid: drop later comps corr>thr (None to skip)
EXPORT_ROI_ROLES = "roi_roles.csv" or None
DEBUG            = True

--------------------------------------------------------------------------------
CORE Step 0. Prepare raw chunk dicts (one per alignment)
--------------------------------------------------------------------------------
chunks_raw = {
  'trial_start': trial_data_trial_start,
  'isi':          trial_data_isi,
  'start_flash_1':trial_data_start_flash_1,
  'end_flash_1':  trial_data_end_flash_1,
  'start_flash_2':trial_data_start_flash_2,
  'end_flash_2':  trial_data_end_flash_2,
  'choice_start': trial_data_choice_start,
  'lick_start':   trial_data_lick_start
}

--------------------------------------------------------------------------------
CORE Step 1. Build session structure
--------------------------------------------------------------------------------
M = build_M_from_chunks(
      chunks_raw,
      cfg={'isi': {'mask_after_isi': True}},   # set False to keep post-ISI samples
      primary_chunk='trial_start'
)
summarize_M(M)

Key outputs:
  M['chunks'][chunk]['data']  shape=(trials, rois, time)
  M['time'] / M['roi_traces'] baseline global axis (trial_start)

--------------------------------------------------------------------------------
CORE Step 2. Time-domain CP across chunks
--------------------------------------------------------------------------------
time_results = run_time_domain_analysis(
    M,
    chunk_keys=CHUNK_KEYS,
    ranks=RANKS,
    min_abs_corr=MIN_ABS_CORR
)

Each time_results[ch]:
  final_A, final_B, final_roi_sets, stability_vec, base, stability, rank

Inspect (diagnostics):
  time_results[ch]['stability_vec']  # abs corr pre stability prune

--------------------------------------------------------------------------------
CORE Step 3. Component catalog + overlaps
--------------------------------------------------------------------------------
catalog = build_component_catalog(M, time_results)
summarize_component_overlaps(catalog)   # top Jaccard pairs (redundancy hint)

Catalog fields:
  catalog['incidence'] (n_rois, n_components)
  catalog['comp_labels'] ["chunk:idx", ...]

--------------------------------------------------------------------------------
CORE Step 4. Phase-normalized ISI CP (adds 'isi_phase')
--------------------------------------------------------------------------------
M['chunks'].pop('isi_phase', None)  # reset if re-running
res_phase, signed_phase = run_phase_analysis(M)
# res_phase['factors']['A'] shape (rois, R_phase)

--------------------------------------------------------------------------------
CORE Step 5. Signed (pos/neg) groups for ALL chunks
--------------------------------------------------------------------------------
combined_results = dict(time_results)
combined_results['isi_phase'] = {'final_A': res_phase['factors']['A']}

signed_all = build_signed_groups_for_chunks(
    M,
    combined_results,
    q=SIGNED_Q,
    use_final=True,
    orient=True,             # uses orient_method below
    orient_method='trace_mean'
)
# Keep the dual-oriented phase result (phase + trace) from run_phase_analysis:
signed_all['isi_phase'] = signed_phase

Each signed_all[ch] = list of dicts:
  {'comp','roi_idx','signs','positive_mask','negative_mask'}

--------------------------------------------------------------------------------
OPT Step 6. Redundancy across signed groups
--------------------------------------------------------------------------------
summarize_signed_group_overlap(
    signed_all,
    mode='pos',          # 'pos' | 'neg' | 'both'
    min_jaccard=0.7
)

--------------------------------------------------------------------------------
CORE Step 7. Integrate signed groups into catalog
--------------------------------------------------------------------------------
catalog = integrate_signed_groups_into_catalog(
    catalog, M, signed_all,
    label_prefix='',       # optional e.g. 'signed:'
    skip_existing=True
)
# New columns: <chunk>:compK_pos / _neg

--------------------------------------------------------------------------------
CORE Step 8. Assess negative subset strength
--------------------------------------------------------------------------------
neg_assessment = {}
for ch in CHUNK_KEYS + ['isi_phase']:
    A_mat = res_phase['factors']['A'] if ch=='isi_phase' else time_results[ch]['final_A']
    sg    = signed_all[ch]
    neg_assessment[ch] = dict(
        loading = assess_signed_groups(A_mat, sg, metric='loading',     print_table=False),
        trace   = assess_signed_groups(A_mat, sg, metric='trace_mean',  M=M, chunk=('isi_phase' if ch=='isi_phase' else ch), print_table=False)
    )

Metrics:
  n_pos, n_neg, frac_neg, pos_mag, neg_mag, rel_mag, weak flag.

--------------------------------------------------------------------------------
CORE Step 9. Prune weak negatives (convert to single-sided)
--------------------------------------------------------------------------------
signed_pruned = {}
for ch in CHUNK_KEYS + ['isi_phase']:
    A_mat = res_phase['factors']['A'] if ch=='isi_phase' else time_results[ch]['final_A']
    sg_copy = [dict(**g) for g in signed_all[ch]]
    prune_weak_negatives(
        A_mat, sg_copy,
        min_neg_frac=MIN_NEG_FRAC,
        min_neg_rel_mag=MIN_NEG_REL_MAG,
        drop_entire=False    # set True to fully drop weak components
    )
    signed_pruned[ch] = sg_copy

--------------------------------------------------------------------------------
CORE Step 10. Overlay multi-component traces (energy → alpha)
--------------------------------------------------------------------------------
overlay_figs = {}
for ch in CHUNK_KEYS:
    overlay_figs[ch] = plot_signed_components_overlay(
        M, signed_pruned[ch],
        trial_chunk='trial_start',
        chunk_name=ch,
        smooth_sigma=OVERLAY_SMOOTH,
        energy_mode=OVERLAY_ENERGY,    # 'combined' or 'pos_all'
        sort_by_energy=True,
        debug_energies=False
    )
overlay_figs['isi_phase'] = plot_signed_components_overlay(
    M, signed_pruned['isi_phase'],
    trial_chunk='trial_start',
    chunk_name='isi_phase',
    smooth_sigma=OVERLAY_SMOOTH,
    energy_mode=OVERLAY_ENERGY
)

--------------------------------------------------------------------------------
OPT Step 11. Detailed per-component stacks
--------------------------------------------------------------------------------
# Time-axis native window:
plot_signed_components_bigstack(
    M, time_results['choice_start']['base'],
    chunk='choice_start',
    signed_groups=signed_pruned['choice_start'],
    smooth_sigma=0.6
)
# Full-trial axis:
plot_signed_component_mean_traces_stack(M, signed_pruned['choice_start'], trial_chunk='trial_start', smooth_sigma=0.6)
# Phase groups projected onto full trial:
plot_phase_components_on_full_trial(M, signed_pruned['isi_phase'], trial_chunk='trial_start', smooth_sigma=0.6)

--------------------------------------------------------------------------------
CORE Step 12. Phase component matrix (All / Short / Long)
--------------------------------------------------------------------------------
plot_isi_phase_components_matrix(
    M, res_phase, signed_pruned['isi_phase'],
    comps=None,                    # or subset list
    events=('F2_on','F2_off'),
    layout='grid',                 # 'grid' | 'per_component'
    smooth_sigma=0.6,
    dedup_corr_thresh=DEDUP_PHASE_CORR,  # None to disable
    corr_mode='all'
)

--------------------------------------------------------------------------------
OPT Phase clustering / representatives
--------------------------------------------------------------------------------
phase_clust = cluster_isi_phase_components(
    signed_pruned['isi_phase'],
    jaccard_thresh=0.85,
    mode='pos'          # 'pos'|'neg'|'both'
)
phase_reps = build_representative_phase_groups(signed_pruned['isi_phase'], phase_clust)
plot_isi_phase_components_matrix(
    M, res_phase, phase_reps,
    comps=phase_clust['keep_components'],
    events=('F2_on','F2_off'),
    layout='grid',
    smooth_sigma=0.6
)

--------------------------------------------------------------------------------
CORE Step 13. Catalog visualizations (ROI rasters + mean traces)
--------------------------------------------------------------------------------
# Global vlim & all groups
plot_all_groups_v2(
    M, catalog,
    groups=None,
    cfg=dict(
        sort_mode='onset',          # 'index'|'onset'
        onset_window=(0.5,2.8),     # onset search window
        onset_z=1.0,                # z-threshold
        smoothing_sigma=0.8,
        events=['F1_on','F1_off','F2_on','F2_off','choice_start','lick_start'],
        figsize=(7.2,8.0)
    ),
    compute_global_vlim=True
)

# Subset example (choice + lick)
subset_labels = [l for l in catalog['comp_labels'] if l.startswith('choice_start:') or l.startswith('lick_start:')]
plot_all_groups_v2(M, catalog, groups=subset_labels, cfg=dict(sort_mode='onset', onset_window=(-0.3,0.6), onset_z=1.0, smoothing_sigma=0.8, events=['F1_on','F1_off','F2_on','F2_off','choice_start','lick_start']), compute_global_vlim=True)

--------------------------------------------------------------------------------
CORE Step 14. Component uniqueness + per-ROI roles
--------------------------------------------------------------------------------
uni_metrics = component_uniqueness_metrics(catalog)
roi_roles_df = build_roi_annotation_dataframe(M, catalog, uni_metrics)
if EXPORT_ROI_ROLES:
    export_roi_annotations(roi_roles_df, uni_metrics, EXPORT_ROI_ROLES)

--------------------------------------------------------------------------------
OPT Additional utilities
--------------------------------------------------------------------------------
# Relaxed re-run for a single short window:
relaxed = rerun_single_chunk_with_relaxed_pruning(M, 'start_flash_1', ranks=[4,6,8], min_abs_corr=0.55)
# Merge highly overlapping catalog components (union):
merged_catalog = merge_high_overlap_components(catalog, overlap_thresh=0.6)
# Keep only selected alignment families:
canonical_catalog = filter_catalog_by_label_prefix(catalog, keep_prefixes=['isi:', 'choice_start:', 'start_flash_2:', 'lick_start:'])

================================================================================
Parameter quick reference (most-used)
================================================================================
run_time_domain_analysis:
  ranks (list[int]) | min_abs_corr (float)

run_phase_analysis:
  uses global PHASE_BINS; internally sets pre_zscored=True for isi_phase.

build_signed_groups_for_chunks:
  q (quantile of |loadings|) | orient (bool) | orient_method {'trace_mean','loading','trace_phase'}

assess_signed_groups:
  metric {'loading','trace_mean','trace_energy'}
  min_neg_frac / min_neg_rel_mag thresholds → weak flag

plot_signed_components_overlay:
  energy_mode {'combined','pos_all'} | min_alpha/max_alpha | annotate_mode {'end','peak'}

plot_isi_phase_components_matrix:
  layout {'grid','per_component'} | dedup_corr_thresh (float|None) | corr_mode {'all','pos','neg'}

cluster_isi_phase_components:
  jaccard_thresh | mode {'pos','neg','both'} | representative {'largest_pos','largest_total','first'}

rerun_single_chunk_with_relaxed_pruning:
  ranks | min_abs_corr (lower for exploratory) → adds 'relaxed' section

================================================================================
END OF ORDERED RECIPE
================================================================================
"""


# %%
# ...existing code above...

# =====================================================================
# MASTER EXPLORATORY WORKFLOW (INTERACTIVE)
# ---------------------------------------------------------------------
# Purpose: Step‑by‑step, fully annotated execution order from raw chunk
# pickles → session structure M → CP components → signed ROI groups →
# diagnostics → plots → component / ROI annotations.
#
# Philosophy:
#   * Keep everything explicit (no opaque runner).
#   * Allow you to pause, inspect intermediate artifacts, adjust params.
#   * Comments below expand WHAT each step does, WHY it exists, HOW to
#     tune key parameters, and OPTIONAL variations.
#
# IMPORTANT: Only edit parameter values in the CONFIG block; the rest
#            of the code is a reference sequence.
# =====================================================================

# --------------------------- CONFIG ----------------------------------
# CHUNK_KEYS: Which event‑aligned windows (chunks) receive time‑domain CP.
#   Order influences visual ordering only (analysis independent).
CHUNK_KEYS        = ['isi','start_flash_1','end_flash_1','start_flash_2',
                     'end_flash_2','choice_start','lick_start']

# RANKS: Candidate CP ranks tried per chunk; best (loss) retained then pruned.
#   Broader list = longer runtime, chance of overfit before pruning.
#   small, spaced probe set: 6 (minimum to separate obvious epochs: baseline, F1, ISI, F2, pre‑choice, lick), 
#   8 (adds finer sub‑phases or condition splits), 10 (tests for marginal extra motifs / overfit). Using an ascending trio lets you:
#   Pick “best” by loss before pruning, then remove unstable components (stability threshold).
#   Detect overfit: if rank 10 adds only low‑stability or duplicate (high Jaccard) sets, effective intrinsic rank is lower.
#   Keep runtime modest (compared to 4–12 sweep).
#   Guidelines if adjusting:
#   If many residual patterns remain or stability still high at 10 → extend (add 12,14).
#   If >30–40% of components pruned for instability at all ranks → include lower (4) or tighten MIN_ABS_CORR.
#   If high redundancy (Jaccard >0.8) appears early → narrow (e.g. 5,7,9).
#   Automate: run ranks=[4,6,8,10,12], compute (recon_loss, #stable, redundancy), pick first rank where (Δloss% < ε AND added stable comps < k).
#   Quick heuristic code (replace RANKS): RANKS = [4,6,8,10,12] # then post-hoc select via stability + redundancy elbow.
# So 6,8,10 is a pragmatic midpoint: wide enough to see diminishing returns, narrow enough to stay fast. 
# Adjust based on stability_vec and overlap metrics.
RANKS             = [6,8,10]

# MIN_ABS_CORR: Split‑half absolute loading correlation threshold for
#   stability pruning (filtering weak / non‑reproducible components).
#   Typical: 0.6–0.7 (raise to be stricter, lower to keep borderline motifs).
MIN_ABS_CORR      = 0.65

# SIGNED_Q: Quantile for selecting top-|loading| ROIs per component when
#   forming signed groups. 0.10 = top 10% (≥ min_rois inside function).
#   Larger → broader groups (risk: heterogeneous); smaller → tighter (risk: too few).
SIGNED_Q          = 0.10

# PHASE_BINS: Number of phase bins for ISI phase resampling (0..1).
#   More bins = finer temporal resolution but more NaNs in sparse areas.
PHASE_BINS        = 80

# MIN_NEG_FRAC / MIN_NEG_REL_MAG: Criteria to treat negative subset as
#   meaningful. If it fails either → negative side pruned (depending on prune call).
#   frac = n_neg/(n_pos+n_neg); rel_mag = neg_mag/pos_mag (metric‑dependent).
MIN_NEG_FRAC      = 0.12
MIN_NEG_REL_MAG   = 0.30

# OVERLAY_CHUNKS: Which chunks to summarize with alpha‑encoded overlays.
OVERLAY_CHUNKS    = CHUNK_KEYS

# OVERLAY_SMOOTH: Gaussian σ (samples) for overlay traces (light smoothing of mean traces).
OVERLAY_SMOOTH    = 0.6

# OVERLAY_ENERGY: Energy metric for transparency scaling.
#   'combined' = Σ (pos - neg)^2   (emphasizes signed divergence)
#   'pos_all'  = Σ pos^2           (ignores negative subset)
OVERLAY_ENERGY    = 'combined'

# DEDUP_PHASE_CORR: Correlation threshold for optional phase grid deduplication.
#   If None → keep all. Typical 0.95–0.98 to remove near‑duplicates.
DEDUP_PHASE_CORR  = 0.97

# EVENTS_FULL: Standard list of trial events for multi‑panel visualizations.
EVENTS_FULL       = ('F1_on','F1_off','F2_on','F2_off','choice_start','lick_start')

# EXPORT_ROI_ROLES: CSV path for exporting ROI role table & component uniqueness.
EXPORT_ROI_ROLES  = "roi_roles.csv"   # set to None to skip writing

# DEBUG_PRINTS: If True print per‑chunk counts, weak component summaries, etc.
DEBUG_PRINTS      = True

# --------------------------- INPUT -----------------------------------
# Assumes you have already loaded each trial_data_* dict (with
# df_trials_with_segments) into the variables below. If not, load first.
chunks_raw = {
    'trial_start':   trial_data_trial_start,
    'isi':           trial_data_isi,
    'start_flash_1': trial_data_start_flash_1,
    'end_flash_1':   trial_data_end_flash_1,
    'start_flash_2': trial_data_start_flash_2,
    'end_flash_2':   trial_data_end_flash_2,
    'choice_start':  trial_data_choice_start,
    'lick_start':    trial_data_lick_start
}

# =====================================================================
# STEP 1. BUILD SESSION STRUCTURE (build_M_from_chunks)
# ---------------------------------------------------------------------
# WHAT: Harmonizes all chunk DataFrames into unified dict M:
#   - Interpolates per‑row time vectors to a canonical chunk time base.
#   - Adds trial behavioral scalars (ISI, choice_start, etc.) from primary chunk.
#   - Optionally truncates 'isi' chunk at actual ISI (mask_after_isi flag).
# WHY: Downstream CP & grouping require consistent (trials, rois, time) arrays.
# PARAM TUNING:
#   cfg={'isi': {'mask_after_isi': True}} → set False if you need post‑ISI activity.
# OUTPUT: M['chunks'][chunk]['data'] (trials, rois, time), M['time'], M['isi'], M['is_short'].
# =====================================================================
M = build_M_from_chunks(chunks_raw, cfg={'isi': {'mask_after_isi': True}},
                        primary_chunk='trial_start')
summarize_M(M)



# %%
# =====================================================================
# STEP 2. TIME-DOMAIN CP DECOMPOSITION (run_time_domain_analysis)
# ---------------------------------------------------------------------
# WHAT: For each chunk in CHUNK_KEYS:
#   - Bagged trial means tensor → CP rank sweep (RANKS).
#   - Adaptive ROI subset extraction + variance/ptp redundancy pruning.
#   - Split-half stability → stability pruning with MIN_ABS_CORR.
# WHY: Extract temporally distinct ROI co‑activation motifs per alignment.
# PARAM TUNING:
#   RANKS: add a lower rank if overfitting (many weak comps); add higher if residual patterns.
#   MIN_ABS_CORR: raise to enforce stricter reproducibility.
# OUTPUT (time_results[ch]):
#   final_A, final_B, final_roi_sets, stability_vec (+ base & stability diagnostics).
# =====================================================================
time_results = run_time_domain_analysis(M, CHUNK_KEYS, RANKS, MIN_ABS_CORR)

if DEBUG_PRINTS:
    for ch in CHUNK_KEYS:
        print(f"[time CP] {ch} final components={time_results[ch]['final_A'].shape[1]} "
              f"stability={np.round(time_results[ch]['stability_vec'],3)}")

# (Optional) Inspect normalized temporal factor shapes
for ch in CHUNK_KEYS:
    plot_component_temporals(time_results[ch], f"{ch} temporal factors")



# %%
# =====================================================================
# STEP 3. BUILD COMPONENT CATALOG (build_component_catalog)
# ---------------------------------------------------------------------
# WHAT: Collapses all time-domain components into a single incidence matrix:
#   rows = ROIs, columns = components (boolean membership).
# WHY: Provides a unified handle for overlap & uniqueness analyses.
# DIAGNOSTIC: summarize_component_overlaps → top Jaccard pairs (redundancy).
# =====================================================================
catalog = build_component_catalog(M, time_results)
summarize_component_overlaps(catalog)



# %%
# =====================================================================
# STEP 4. PHASE-NORMALIZED ISI ANALYSIS (run_phase_analysis)
# ---------------------------------------------------------------------
# WHAT: Resamples each ISI segment to PHASE_BINS between 0..1 (phase bins),
#       runs CP on phase domain, orients signed groups twice:
#         trace_phase (center window) then trace_mean.
# WHY: Detect motifs anchored to relative ISI progress rather than absolute time.
# PARAM: PHASE_BINS (more = finer phase resolution).
# OUTPUT: res_phase (CP factors), signed_phase (phase component signed groups).
# =====================================================================
M['chunks'].pop('isi_phase', None)  # ensure clean re-run
res_phase, signed_phase = run_phase_analysis(M)



# %%
# =====================================================================
# STEP 5. BUILD SIGNED GROUPS ACROSS ALL CHUNKS (build_signed_groups_for_chunks)
# ---------------------------------------------------------------------
# WHAT: For each chunk (time + phase) select top SIGNED_Q fraction of |loadings|
#       and partition by sign into positive/negative ROI subsets.
# ORIENTATION: orient_method='trace_mean' ensures positive subset has larger mean activity.
# SPECIAL: Replace auto phase groups with dual-oriented signed_phase from step 4.
# WHY: Enables polarity-aware plotting and energy contrast metrics.
# =====================================================================
combined_results = dict(time_results)
combined_results['isi_phase'] = {'final_A': res_phase['factors']['A']}
signed_all = build_signed_groups_for_chunks(M, combined_results,
                                            q=SIGNED_Q,
                                            use_final=True,
                                            orient=True,
                                            orient_method='trace_mean')
signed_all['isi_phase'] = signed_phase  # preserve dual orientation




# %%
# =====================================================================
# STEP 6 (OPTIONAL). REDUNDANCY ACROSS SIGNED GROUPS (summarize_signed_group_overlap)
# ---------------------------------------------------------------------
# WHAT: Prints pairwise Jaccard overlaps ≥ threshold across positive (or chosen) sets.
# WHY: Spot near-duplicate ROI assemblies across different alignments.
# PARAMS:
#   mode: 'pos' | 'neg' | 'both'
#   min_jaccard: report cutoff (e.g. 0.7).
# =====================================================================
summarize_signed_group_overlap(signed_all, mode='both', min_jaccard=0.7)
summarize_signed_group_overlap(signed_all, mode='pos', min_jaccard=0.7)
summarize_signed_group_overlap(signed_all, mode='neg', min_jaccard=0.7)



# %%
# =====================================================================
# STEP 7. INTEGRATE SIGNED GROUPS INTO CATALOG (integrate_signed_groups_into_catalog)
# ---------------------------------------------------------------------
# WHAT: Appends columns for each signed component subset:
#   <chunk>:compK_pos and (if non-empty) <chunk>:compK_neg
# WHY: Allow downstream analytics (uniqueness, clustering) to consider polarity-specific sets.
# NOTE: Sets are appended; original time-domain components retained.
# =====================================================================
catalog = integrate_signed_groups_into_catalog(catalog, M, signed_all,
                                               label_prefix='', skip_existing=True)




# %%
# =====================================================================
# STEP 8. ASSESS NEGATIVE SUBSET STRENGTH (assess_signed_groups)
# ---------------------------------------------------------------------
# WHAT: Quantifies whether negative side is meaningful:
#   metrics: 'loading' (mean|loading|), 'trace_mean' (mean abs time trace), 'trace_energy' (Σ x^2).
# RULE: weak if frac_neg < MIN_NEG_FRAC OR rel_mag < MIN_NEG_REL_MAG.
# WHY: Avoid interpreting trivial “negative” sides (few ROIs or negligible amplitude).
# USE: Store both loading & trace_mean metrics for comparison.
# =====================================================================
neg_assessment = {}
for ch in CHUNK_KEYS + ['isi_phase']:
    A_mat = res_phase['factors']['A'] if ch == 'isi_phase' else time_results[ch]['final_A']
    sg    = signed_all[ch]
    df_load  = assess_signed_groups(A_mat, sg, metric='loading', print_table=False)
    df_trace = assess_signed_groups(A_mat, sg, metric='trace_mean', M=M,
                                    chunk=('isi_phase' if ch=='isi_phase' else ch),
                                    print_table=False)
    neg_assessment[ch] = dict(loading=df_load, trace=df_trace)
    if DEBUG_PRINTS:
        print(f"[neg assess] {ch}: weak(load) {df_load['weak'].sum()}/{len(df_load)}")




# %%
# =====================================================================
# STEP 9. PRUNE WEAK NEGATIVES (prune_weak_negatives)
# ---------------------------------------------------------------------
# WHAT: Convert weak negative subsets to single-sided (positive only) OR drop entire component.
# WHY: Cleaner visualizations & energy metrics.
# TUNING:
#   Increase MIN_NEG_FRAC to demand broader negative involvement.
#   Increase MIN_NEG_REL_MAG to demand stronger amplitude balance.
# =====================================================================
signed_pruned = {}
for ch in CHUNK_KEYS + ['isi_phase']:
    A_mat = res_phase['factors']['A'] if ch == 'isi_phase' else time_results[ch]['final_A']
    sg_copy = [dict(**g) for g in signed_all[ch]]
    prune_weak_negatives(A_mat, sg_copy,
                         min_neg_frac=MIN_NEG_FRAC,
                         min_neg_rel_mag=MIN_NEG_REL_MAG,
                         drop_entire=False)
    signed_pruned[ch] = sg_copy




# %%
# =====================================================================
# STEP 10. OVERLAY MULTI-COMPONENT PLOTS (plot_signed_components_overlay)
# ---------------------------------------------------------------------
# WHAT: 3 panels (All / Short / Long) overlaying positive (solid) & negative (dashed)
#       mean traces for all components with alpha scaled by energy.
# WHY: Fast holistic inspection; energy sorting highlights dominant motifs.
# KEY PARAMS:
#   energy_mode: 'combined' | 'pos_all'
#   smooth_sigma: minor smoothing of mean trace
#   annotate / annotate_mode / annotate_top_n: labeling control
#   energy_flat_tol + flat_energy_alpha: uniform alpha fallback for near-constant energies
# =====================================================================
overlay_figs = {}
for ch in OVERLAY_CHUNKS:
    overlay_figs[ch] = plot_signed_components_overlay(
        M, signed_pruned[ch],
        trial_chunk='trial_start',
        chunk_name=ch,
        smooth_sigma=OVERLAY_SMOOTH,
        energy_mode=OVERLAY_ENERGY,
        sort_by_energy=True,
        debug_energies=False
    )
overlay_figs['isi_phase'] = plot_signed_components_overlay(
    M, signed_pruned['isi_phase'],
    trial_chunk='trial_start',
    chunk_name='isi_phase',
    smooth_sigma=OVERLAY_SMOOTH,
    energy_mode=OVERLAY_ENERGY
)





# %%
# =====================================================================
# STEP 11 (OPTIONAL). PER-COMPONENT STACKED PLOTS
# ---------------------------------------------------------------------
# FUNCTIONS:
#   plot_signed_components_bigstack  (chunk-native time axis)
#   plot_signed_component_mean_traces_stack (global trial axis)
#   plot_phase_components_on_full_trial (phase groups on full trial axis)
# WHY: Inspect each component individually without alpha compression.
# WHEN: Use after identifying interesting overlays needing granular view.
# =====================================================================
plot_signed_components_bigstack(
    M, time_results['choice_start']['base'],
    chunk='choice_start',
    signed_groups=signed_pruned['choice_start'],
    smooth_sigma=0.6
)
plot_signed_component_mean_traces_stack(
    M, signed_pruned['choice_start'],
    trial_chunk='trial_start',
    smooth_sigma=0.6
)
plot_phase_components_on_full_trial(
    M, signed_pruned['isi_phase'],
    trial_chunk='trial_start',
    smooth_sigma=0.6
)






# %%
# =====================================================================
# STEP 12. PHASE COMPONENT MATRIX (plot_isi_phase_components_matrix)
# ---------------------------------------------------------------------
# WHAT: Grid/column of phase-resampled component POS/NEG traces (All/Short/Long).
# DEDUP: dedup_corr_thresh removes highly correlated motifs (basis trace selected by corr_mode).
# PARAMS:
#   layout: 'grid' (compact) or 'per_component'
#   corr_mode: 'all'|'pos'|'neg' (trace used for similarity)
# =====================================================================
plot_isi_phase_components_matrix(
    M, res_phase, signed_pruned['isi_phase'],
    comps=None,
    events=('F2_on','F2_off'),
    layout='grid',
    smooth_sigma=0.6,
    dedup_corr_thresh=DEDUP_PHASE_CORR
)

# OPTIONAL: Cluster & replot only representative phase components
phase_clust = cluster_isi_phase_components(
    signed_pruned['isi_phase'],
    jaccard_thresh=0.85,
    mode='pos'          # 'pos'|'neg'|'both'
)
phase_reps = build_representative_phase_groups(signed_pruned['isi_phase'], phase_clust)
plot_isi_phase_components_matrix(
    M, res_phase, phase_reps,
    comps=phase_clust['keep_components'],
    events=('F2_on','F2_off'),
    layout='grid',
    smooth_sigma=0.6
)





# %%
# =====================================================================
# STEP 13. CATALOG VISUALIZATIONS (plot_all_groups_v2)
# ---------------------------------------------------------------------
# WHAT: For each catalog component (union of time-domain & signed additions),
#        produce multi-panel raster + mean traces (Short / Long / All).
# CONFIG KEYS (cfg):
#   sort_mode: 'index' or 'onset'
#   onset_window: (t0,t1) window for first-rise detection
#   onset_z: z-threshold above baseline
#   smoothing_sigma: temporal smoothing (frames) before rasters
#   events: list of event names for vertical lines
# GLOBAL SCALING: compute_global_vlim=True derives symmetric robust vlim.
# SUBSETS: Provide groups list for thematic subsets (e.g., choice-related).
# =====================================================================

cfg = dict(
    sort_mode='onset',
    onset_window=(0.5,2.8),
    onset_z=1.0,
    smoothing_sigma=0.8,
    events=list(EVENTS_FULL),
    show_onset_markers=True,
    onset_marker_kwargs=dict(marker='o', s=8, color='k', alpha=0.7),
    zero_line=True,
    zero_line_kwargs=dict(color='k', lw=0.7, ls='--', alpha=0.9)
)
plot_all_groups_v2(M, catalog, cfg=cfg, compute_global_vlim=True)




# plot_all_groups_v2(
#     M, catalog,
#     groups=None,
#     cfg=dict(
#         sort_mode='onset',
#         onset_window=(0.5, 2.8),
#         onset_z=1.0,
#         smoothing_sigma=0.8,
#         events=list(EVENTS_FULL),
#         figsize=(7.2,8.0)
#     ),
#     compute_global_vlim=True
# )




# %%
subset_labels = [l for l in catalog['comp_labels']
                 if l.startswith('choice_start:') or l.startswith('lick_start:')]
plot_all_groups_v2(
    M, catalog,
    groups=subset_labels,
    cfg=dict(
        sort_mode='onset',
        onset_window=(-0.3,0.6),
        onset_z=1.0,
        smoothing_sigma=0.8,
        events=list(EVENTS_FULL)
    ),
    compute_global_vlim=True
)





# %%
# =====================================================================
# STEP 14. UNIQUENESS & ROI ROLE ANNOTATIONS
# ---------------------------------------------------------------------
# WHAT:
#   component_uniqueness_metrics → size, max_jacc_other, exclusivity matrix.
#   build_roi_annotation_dataframe → per-ROI role counts & primary role label.
# WHY: Identify broad vs specific motifs; quantify ROI participation diversity.
# EXPORT: export_roi_annotations writes ROI table + component summary CSVs.
# =====================================================================
uni_metrics = component_uniqueness_metrics(catalog)
roi_roles_df = build_roi_annotation_dataframe(M, catalog, uni_metrics)
if EXPORT_ROI_ROLES:
    export_roi_annotations(roi_roles_df, uni_metrics, EXPORT_ROI_ROLES)





# %%
# =====================================================================
# STEP 15. PHASE COMPONENT ROI LISTS (list_isi_phase_component_rois)
# ---------------------------------------------------------------------
# WHAT: Tabular listing of positive / negative ROI indices per phase component;
#       optional overlap reporting using chosen subset (pos/neg/both).
# WHY: Manual inspection / external anatomical mapping / custom statistics.
# PARAMS:
#   overlap_mode: 'pos'|'neg'|'both'
#   min_jaccard_report: print only overlaps ≥ value.
# =====================================================================
df_phase_sets = list_isi_phase_component_rois(
    M, res_phase, signed_pruned['isi_phase'],
    overlap_mode='pos',
    min_jaccard_report=0.4
)
if not df_phase_sets.empty:
    try:
        comp0_pos = df_phase_sets[(df_phase_sets.comp==0) & (df_phase_sets.group=='pos')].iloc[0].roi_indices
        print("Phase comp 0 example positive ROI indices:", comp0_pos[:10])
    except Exception:
        pass





# %%
# =====================================================================
# STEP 16 (OPTIONAL). RELAXED RE-RUN OF A CHUNK (rerun_single_chunk_with_relaxed_pruning)
# ---------------------------------------------------------------------
# WHY: Explore motifs suppressed by initial thresholds (short windows or low amplitude).
# TUNING: Provide lower min_abs_corr or fewer ranks for quick scans.
# =====================================================================
# relaxed_res = rerun_single_chunk_with_relaxed_pruning(M, 'start_flash_1',
#                                                       ranks=[4,6,8],
#                                                       min_abs_corr=0.55)
# print("Relaxed start_flash_1 components:",
#       relaxed_res['relaxed']['A_relaxed'].shape[1])

# =====================================================================
# END OF MASTER WORKFLOW
# =====================================================================

# ...existing

# %%
# Ideas (pick 1–2 to implement; combine later):

# UpSet plot (best for many set intersections, avoids Venn clutter).
# Jaccard heatmap + hierarchical clustering (quick overview; reorder to expose blocks).
# Sankey / alluvial (good for flow chunk -> comp(+/-) -> merged cluster, not for dense all-to-all).
# Chord diagram (visually appealing but quickly messy >25 groups).
# Bipartite compression: left = ROI clusters (e.g. anatomical or k‑means), right = component(+/-), edges weighted.
# Edge‑bundled overlap graph (networkx + force layout) if you want a network feel.
# Matrix “barcode” (ROI vs group incidence) reordered by seriation (gives fine-grained structure).
# Recommended sequence: A) Collapse / filter groups (drop very small and near duplicates). B) UpSet for top intersection combos. C) Clustered Jaccard heatmap. D) Optional Sankey (chunk -> group -> (merged cluster)).

# Below: utility + three plot options (heatmap, UpSet, Sankey). Replace placeholders where noted.

# %%
res = upset_roi_group_overlaps(
    signed_pruned,
    chunks=['isi','choice_start','lick_start'],  # or None for all
    mode='both',          # 'pos' | 'neg' | 'both'
    include_negative=True,
    min_size=6,
    dedup_jaccard=0.97,
    max_combo_size=3,
    top_k=25,
    min_intersection=4,
    coverage_priority=True,
    sort_by='size'
)
res['intersections_df'].head()




# %%

# res = upset_plot_enhanced_workflow(
#     signed_pruned,
#     chunks=['isi','choice_start','lick_start'],
#     mode='both',
#     min_size=6,
#     max_combo_size=3,
#     top_k=30,
#     wrap_width=18,
#     truncate_labels=None,
#     annotate_fmt="{size} ({pct:.1f}%)",
#     palette=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02'],
#     figsize_scale=1.2
# )

# res = upset_plot_enhanced_workflow(
#     signed_pruned,
#     chunks=['isi','choice_start','lick_start'],
#     mode='both',
#     min_size=6,
#     max_combo_size=3,
#     top_k=30,
#     wrap_width=26,
#     figsize_scale=1.4,
#     matrix_use_ids=True,
#     full_label_legend='right',
#     legend_cols=3,              # (kept for future extension if you move to real Legend)
#     label_fontsize=9,
#     legend_fontsize=8,
#     annotate_fmt="{size}",
# )
# fig = res['figure']
# fig.savefig("upset_readable.png", dpi=150)


# subset_labels = [l for l in catalog['comp_labels']]


res = upset_roi_group_overlaps(
    signed_pruned,
    chunks=None,  # or None for all
    mode='both',          # 'pos' | 'neg' | 'both'
    include_negative=True,
    min_size=6,
    dedup_jaccard=0.97,
    max_combo_size=3,
    top_k=25,
    min_intersection=4,
    coverage_priority=True,
    sort_by='size'
)
res['intersections_df'].head()


labels = catalog['comp_labels']
fig, mapping = plot_upset_enhanced(
    res['intersections_df'], labels,
    matrix_use_ids=True,
    inline_ids_with_labels=True,   # <- key
    full_label_legend=None,        # ensure no side legend
    wrap_width=28,
    figsize_scale=1.4
)


# %%
# UpSet (correct label usage)
res = upset_roi_group_overlaps(
    signed_pruned,
    chunks=None,
    mode='both',
    include_negative=True,
    min_size=6,
    dedup_jaccard=0.97,
    max_combo_size=3,
    top_k=25,
    min_intersection=4,
    coverage_priority=True,
    sort_by='size'
)

# The wrapper did NOT use catalog labels; use only the sets actually in the UpSet computation.
group_sets = res['group_sets']          # dict label -> set
inter_df   = res['intersections_df']    # intersection table

# Optional: if wrapper already stored 'labels', prefer that ordering; else derive a stable order.
ordered_labels = res.get('labels')
if ordered_labels is None:
    # keep insertion order (Python 3.7+) which reflects build_group_sets_for_upset order
    ordered_labels = list(group_sets.keys())

# Diagnostics: show mismatches vs catalog labels (just to verify)
catalog_labels = set(catalog['comp_labels'])
gs_labels = set(ordered_labels)
extra_catalog = sorted(catalog_labels - gs_labels)
extra_upset   = sorted(gs_labels - catalog_labels)
if extra_catalog:
    print(f"[UpSet warn] {len(extra_catalog)} catalog labels not in group_sets (expected if catalog has raw comps).")
if extra_upset:
    print(f"[UpSet info] {len(extra_upset)} UpSet group labels not in catalog (e.g. pruned pos/neg).")

# Plot with only the relevant labels so matrix rows match intersections
fig, mapping_df = plot_upset_enhanced(
    inter_df,
    ordered_labels,
    matrix_use_ids=True,
    inline_ids_with_labels=True,
    set_size_position='right',
    left_pad=0.22,  # may adjust further if labels still long    
    full_label_legend=None,
    figsize_scale=1.6,
    explain=True,
    show_side_pct=True,
    show_bar_pct=False,
    annotate_bars=True,
    annotate_fmt="{size}",
    label_fontsize=9,
    show_group_separators=True,
    group_separator_color='k',
    intersection_count_labels=True,
    intersection_count_fmt="n={size}",
    intersection_count_fontsize=8,
    intersection_count_min_sets=2,
    wrap_width=38,    
)
# fig.savefig("upset_signed_groups.png", dpi=140)
print("Saved: upset_signed_groups.png")
print(mapping_df.head())






# %%
fig1 = plot_sankey_overlap(signed_all, chunks=['isi','choice_start','lick_start'], jaccard_thresh=0.3)
# fig1.show()
fig1.write_html("plot1.html")   # open in browser
# fig1.write_image("plot.png")   # static image
fig2 = plot_sankey_hierarchy(signed_all, chunks=['isi','choice_start','lick_start'], cluster_jaccard=0.6)
# fig2.show()
fig2.write_html("plot2.html")   # open in browser
# fig2.write_image("plot.png")   # static image




# %%



fig1 = plot_sankey_overlap(signed_pruned, chunks=CHUNK_KEYS, jaccard_thresh=0.3)
# fig1.show()
fig1.write_html("plot3.html")   # open in browser
fig2 = plot_sankey_hierarchy(signed_pruned, chunks=CHUNK_KEYS, cluster_jaccard=0.6)
# fig2.show()
fig2.write_html("plot4.html")   # open in browser

# %%

fig1 = plot_sankey_overlap(signed_pruned, chunks=['isi','choice_start','lick_start'], jaccard_thresh=0.3)
# fig1.show()
fig1.write_html("plot3.html")   # open in browser
fig2 = plot_sankey_hierarchy(signed_pruned, chunks=['isi','choice_start','lick_start'], cluster_jaccard=0.6)
# fig2.show()
fig2.write_html("plot4.html")   # open in browser


# %%


# Example usage (after you have 'signed_pruned'):
spec = build_alluvial_spec(signed_pruned,
                            chunks=['isi','choice_start','lick_start'],
                            include_negative=True,
                            min_size=6,
                            dedup_jaccard=0.97,
                            cluster_jaccard=0.6)
fig = plot_alluvial(spec, flow_alpha=0.55)
fig.savefig("alluvial_static.png", dpi=150)




# %%


df_roles = roi_membership_summary(catalog)
visualize_roi_overlap_suite(catalog, co_metric='jaccard', top_k_multi=100)
co = plot_roi_co_membership(catalog, metric='shared', top_k_multi=80)
emb = embed_rois_by_membership(catalog, method='tsne')




# %%
fc = functional_category_workflow(M, catalog, chunks_filter=None,
                                  min_component_size=6,
                                  min_cat_size=10,
                                  min_jaccard_edge=0.18)
fc['descriptors'].head()


# %%



# Inspect cross-sign reuse directly
sg = signed_pruned['choice_start']  # or signed_all
pos_sets = {g['comp']: set(g['roi_idx'][g['positive_mask']]) for g in sg}
neg_sets = {g['comp']: set(g['roi_idx'][g['negative_mask']]) for g in sg}

cross = []
for ci, p in pos_sets.items():
    for cj, n in neg_sets.items():
        if p & n:
            cross.append((ci, cj, len(p & n)))
print("Cross-sign overlaps:", cross)  # Expect [] for your current result

# Jaccard view
ov_choice = analyze_pos_neg_overlap_for_chunk(signed_pruned, 'choice_start', metric='jaccard')
# Raw counts
ov_choice_counts = analyze_pos_neg_overlap_for_chunk(signed_pruned, 'choice_start', metric='count')


# %%
# Example usage (after signed_pruned ready):
plot_signed_group_venn(signed_pruned,
                       chunks=['isi','choice_start'],
                       mode='pos',
                       select_contains=['isi:c0','choice_start:c1'])
plot_signed_group_venn(signed_pruned,
                       chunks=['isi','choice_start','lick_start'],
                       mode='both',
                       select_labels=['isi:c2_both','choice_start:c3_both','lick_start:c1_both'])

# %%




for chunk in CHUNK_KEYS:
    # Inspect cross-sign reuse directly
    sg = signed_pruned[chunk]  # or signed_all
    pos_sets = {g['comp']: set(g['roi_idx'][g['positive_mask']]) for g in sg}
    neg_sets = {g['comp']: set(g['roi_idx'][g['negative_mask']]) for g in sg}
    
    cross = []
    for ci, p in pos_sets.items():
        for cj, n in neg_sets.items():
            if p & n:
                cross.append((ci, cj, len(p & n)))
    print("Cross-sign overlaps:", cross)  # Expect [] for your current result
    
    # Jaccard view
    ov_choice = analyze_pos_neg_overlap_for_chunk(signed_pruned, chunk, metric='jaccard')
    # Raw counts
    ov_choice_counts = analyze_pos_neg_overlap_for_chunk(signed_pruned, chunk, metric='count')

# %%

flip_df = find_cross_sign_rois(signed_pruned['choice_start'])
print(flip_df)



# %%
for chunk in CHUNK_KEYS:
    print(chunk)
    flip_df = find_cross_sign_rois(signed_pruned[chunk])
    print(flip_df)
    
# %%# Example quick usage after detecting flips:
flips_choice = summarize_sign_flips(signed_pruned['choice_start'], time_results['choice_start']['final_A'])
print(flips_choice)
plot_sign_flip_roi(M, 'choice_start', signed_pruned['choice_start'],
                   time_results['choice_start']['final_A'], roi=1932)








# %%
parts = build_intersection_partitions(
    catalog,
    group_order=['trial_start','isi','start_flash_1','end_flash_1','start_flash_2','end_flash_2','choice_start','lick_start'],
    min_size_within=5,
    min_size_across=5
)
non_cat = parts['non_intersectional']
within_cat = parts['within_group_intersectional']
across_cat = parts['across_group_intersectional']
print(non_cat['comp_labels'][:5])
print(within_cat['comp_labels'][:5])
print(across_cat['comp_labels'][:5])
