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
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, roc_auc_score

from matplotlib.gridspec import GridSpec

from copy import deepcopy



# ---------------------------
# helpers (local, self-contained)
# ---------------------------

def _to_seconds(arr_like):
    """Convert array-like to seconds. If max>10, assume input was ms."""
    arr = np.asarray(arr_like, dtype=float)
    if arr.size == 0:
        return arr.astype(float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    print(f"[to_seconds] input range: {mn:.3f} .. {mx:.3f}")
    if np.nanmax(arr) > 10.0:  # looks like ms
        arr = arr / 1000.0
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        print(f"[to_seconds] converted to seconds: {mn:.3f} .. {mx:.3f}")
    return arr

def _check_session_labels(si: Dict[str, Any], tol: float, require_sets: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate session label sets: unique_isis, short_isis, long_isis (all seconds).
       Ensures:
         - provided and non-empty (if require_sets)
         - short & long disjoint
         - union matches unique (within tol)
    """
    if 'unique_isis' not in si:
        raise ValueError("session_info must contain 'unique_isis'.")

    unique = _to_seconds(si['unique_isis'])
    unique = np.sort(np.unique(unique))
    print(f"[labels] unique_isis (s): {unique}")

    short = _to_seconds(si.get('short_isis', []))
    long  = _to_seconds(si.get('long_isis',  []))
    short = np.sort(np.unique(short))
    long  = np.sort(np.unique(long))
    print(f"[labels] short_isis (s): {short}")
    print(f"[labels] long_isis  (s): {long}")

    if require_sets:
        if short.size == 0 or long.size == 0:
            raise ValueError("require_sets=True but 'short_isis' or 'long_isis' missing/empty in session_info.")

    # Disjointness check
    if short.size and long.size:
        # consider two values equal if within tol
        for s in short:
            if np.any(np.abs(long - s) <= tol):
                raise ValueError(f"short_isis and long_isis overlap at ~{s:.3f}s (within tol={tol}).")

    # Coverage check: union ≈ unique
    if short.size or long.size:
        union = np.sort(np.unique(np.concatenate([short, long])))
        # For each unique, must be matched by some union element within tol, and vice versa
        def _covered(A, B):
            if A.size == 0 and B.size == 0: return True
            flags = []
            for a in A:
                flags.append(np.any(np.abs(B - a) <= tol))
            return np.all(flags)

        ok1 = _covered(unique, union)
        ok2 = _covered(union, unique)
        if not (ok1 and ok2):
            raise ValueError("short_isis ∪ long_isis does not match unique_isis within tolerance.")

    return unique, short, long

def _interp_to_grid(t_vec: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """NaN-safe 1D interpolation without extrapolation: outside support -> NaN."""
    vals = np.interp(grid, t_vec, y, left=np.nan, right=np.nan)
    tmin, tmax = np.nanmin(t_vec), np.nanmax(t_vec)
    out = (grid < tmin) | (grid > tmax)
    vals[out] = np.nan
    return vals

def _require_monotonic_time(t_vec: np.ndarray) -> np.ndarray:
    """Ensure strictly increasing time; if not, sort (and warn once per call)."""
    if not np.all(np.diff(t_vec) > 0):
        print("[warn] dff_time_vector not strictly increasing; sorting time and corresponding dff samples.")
        order = np.argsort(t_vec)
        return order
    return None

def _member_mask(vals: np.ndarray, levels: np.ndarray, tol: float) -> np.ndarray:
    """Return boolean mask where each val is within tol of any level."""
    if levels.size == 0:
        return np.zeros_like(vals, dtype=bool)
    return (np.abs(vals[:, None] - levels[None, :]) <= tol).any(axis=1)

# ---------------------------
# main builder
# ---------------------------


def build_M_from_trials(trial_data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonical session builder. Uses ONLY trial_data + cfg.
    Grids built strictly per cfg['grids'] without fallbacks/aliasing.
    Labels (short/long/unique) must be present/consistent if cfg['labels']['require_sets'] is True.
    """
    print("=== build_M_from_trials: START ===")

    # --- pull QA + label config
    qa = cfg.get("qa", {})
    verbose = bool(qa.get("verbose", True))
    err_on_miss = bool(qa.get("error_on_missing_alignment", True))
    tol = float(cfg.get("labels", {}).get("tolerance_sec", 1e-3))
    require_sets = bool(cfg.get("labels", {}).get("require_sets", True))

    # --- dataframe + basic counts
    df = trial_data['df_trials_with_segments']
    si = trial_data.get('session_info', {})
    n_trials = len(df)
    print(f"[info] trials: {n_trials}")

    # --- labels / ISIs (strict)
    unique_isis, short_isis, long_isis = _check_session_labels(si, tol, require_sets=require_sets)
    isi_vec = _to_seconds(df['isi'].to_numpy()).astype(float)
    print(f"[isi] per-trial ISI range: {np.nanmin(isi_vec):.3f} .. {np.nanmax(isi_vec):.3f} (s)")

    # Verify every isi maps to some allowed unique_isis within tol
    unmapped = ~_member_mask(isi_vec, unique_isis, tol)
    if np.any(unmapped):
        bad_vals = np.unique(np.round(isi_vec[unmapped], 6))
        raise ValueError(f"Found ISIs not in session 'unique_isis' within tol={tol}: {bad_vals}")

    # Strict short/long per trial (no heuristics)
    is_short = _member_mask(isi_vec, short_isis, tol)
    is_long  = _member_mask(isi_vec, long_isis,  tol)
    both = is_short & is_long
    if np.any(both):
        idx = np.where(both)[0][:10]
        raise ValueError(f"{both.sum()} trials labeled BOTH short and long (tol={tol}). Examples: {idx.tolist()}")
    if require_sets and not np.all(is_short | is_long):
        cnt = int((~(is_short | is_long)).sum())
        idx = np.where(~(is_short | is_long))[0][:10]
        raise ValueError(f"{cnt} trials not labeled short OR long by session sets. Examples: {idx.tolist()}")

    # --- grid config
    grids_cfg = cfg.get("grids", {})

    # F1OFF options (now supports pre_seconds and dt_sec)
    f1_cfg = grids_cfg.get("f1off", {})
    want_f1        = bool(f1_cfg.get("build", True))
    f1_bins        = int(f1_cfg.get("bins", 240))                  # used only if dt_sec not provided
    f1_tmax_spec   = f1_cfg.get("tmax", "max_isi")                 # "max_isi" or float seconds
    f1_mask_preF2  = bool(f1_cfg.get("mask_preF2", True))
    f1_pre_seconds = float(f1_cfg.get("pre_seconds", 0.0))         # NEW
    f1_dt_sec      = f1_cfg.get("dt_sec", None)                    # NEW (None or float>0)

    # --- probe 1st row for ROI shape
    first = df.iloc[0]
    dff0 = np.asarray(first['dff_segment'], dtype=float)
    if dff0.ndim != 2:
        raise ValueError("dff_segment must be 2D (n_rois x n_samples) per trial")
    n_rois = dff0.shape[0]
    print(f"[info] ROIs: {n_rois}, first trial dff shape: {dff0.shape}")

    # --- containers for behavior
    is_right_trial  = np.zeros(n_trials, dtype=bool)
    is_right_choice = np.zeros(n_trials, dtype=bool)
    rewarded        = np.zeros(n_trials, dtype=bool)
    punished        = np.zeros(n_trials, dtype=bool)
    did_not_choose  = np.zeros(n_trials, dtype=bool)
    time_dnc        = np.full(n_trials, np.nan, float)
    choice_start    = np.full(n_trials, np.nan, float)
    choice_stop     = np.full(n_trials, np.nan, float)
    servo_in        = np.full(n_trials, np.nan, float)
    servo_out       = np.full(n_trials, np.nan, float)
    lick            = np.zeros(n_trials, dtype=bool)
    lick_start      = np.full(n_trials, np.nan, float)
    RT              = np.full(n_trials, np.nan, float)
    F1_on           = np.full(n_trials, np.nan, float)
    F1_off          = np.full(n_trials, np.nan, float)
    F2_on           = np.full(n_trials, np.nan, float)
    F2_off          = np.full(n_trials, np.nan, float)

    # --- prep F1-OFF grid if needed
    roi_traces = None
    time_f1 = None
    nans_f1 = 0

    if want_f1:
        # determine tmax (end of grid)
        if f1_tmax_spec == "max_isi":
            t_max = float(np.nanmax(isi_vec))
        else:
            t_max = float(f1_tmax_spec)

        if f1_dt_sec is not None:
            f1_dt_sec = float(f1_dt_sec)
            if not (f1_dt_sec > 0):
                raise ValueError("f1off.dt_sec must be > 0 when provided.")
            # uniform step grid from -pre to +t_max
            time_f1 = np.arange(-f1_pre_seconds, t_max + 1e-12, f1_dt_sec).astype(np.float32)
        else:
            # fallback: use bins (kept for backward compatibility)
            time_f1 = np.linspace(-f1_pre_seconds, t_max, int(f1_bins)).astype(np.float32)

        roi_traces = np.full((n_trials, n_rois, time_f1.size), np.nan, dtype=np.float32)
        dt_print = (time_f1[1] - time_f1[0]) if time_f1.size > 1 else np.nan
        print(f"[grid:F1OFF] pre={f1_pre_seconds:.3f}s, bins={time_f1.size}, "
              f"span=({time_f1[0]:.3f},{time_f1[-1]:.3f})s, dt={dt_print:.4f}s, tmax_mode={f1_tmax_spec}")

    # --- which other grids to build (unchanged)
    def _win_and_bins(name):
        g = grids_cfg.get(name, {})
        return tuple(map(float, g.get("window", (-0.2, 0.6)))), int(g.get("bins", 160))

    build_f2 = bool(grids_cfg.get("f2", {}).get("build", False))
    build_choice = bool(grids_cfg.get("choice", {}).get("build", False))
    build_servo_in = bool(grids_cfg.get("servo_in", {}).get("build", False))
    build_servo_out = bool(grids_cfg.get("servo_out", {}).get("build", False))

    tau_f2 = tau_choice = tau_servo_in = tau_servo_out = None
    X_f2 = X_choice = X_servo_in = X_servo_out = None

    if build_f2:
        win, nb = _win_and_bins("f2")
        tau_f2 = np.linspace(win[0], win[1], nb).astype(np.float32)
        X_f2 = np.full((n_trials, n_rois, tau_f2.size), np.nan, dtype=np.float32)
        print(f"[grid:F2] window={win}, bins={nb}, dt={(tau_f2[1]-tau_f2[0]):.4f}s")

    if build_choice:
        win, nb = _win_and_bins("choice")
        tau_choice = np.linspace(win[0], win[1], nb).astype(np.float32)
        X_choice = np.full((n_trials, n_rois, tau_choice.size), np.nan, dtype=np.float32)
        print(f"[grid:CHOICE] window={win}, bins={nb}, dt={(tau_choice[1]-tau_choice[0]):.4f}s")

    if build_servo_in:
        win, nb = _win_and_bins("servo_in")
        tau_servo_in = np.linspace(win[0], win[1], nb).astype(np.float32)
        X_servo_in = np.full((n_trials, n_rois, tau_servo_in.size), np.nan, dtype=np.float32)
        print(f"[grid:SERVO_IN] window={win}, bins={nb}, dt={(tau_servo_in[1]-tau_servo_in[0]):.4f}s")

    if build_servo_out:
        win, nb = _win_and_bins("servo_out")
        tau_servo_out = np.linspace(win[0], win[1], nb).astype(np.float32)
        X_servo_out = np.full((n_trials, n_rois, tau_servo_out.size), np.nan, dtype=np.float32)
        print(f"[grid:SERVO_OUT] window={win}, bins={nb}, dt={(tau_servo_out[1]-tau_servo_out[0]):.4f}s")

    # --- per-trial processing
    missing_f2 = []
    missing_choice = []
    missing_servo_in = []
    missing_servo_out = []

    print("[loop] processing trials …")
    for i, (_, row) in enumerate(df.iterrows()):
        # event times
        F1_on[i]  = float(row.get('start_flash_1', np.nan))
        F1_off[i] = float(row.get('end_flash_1',   np.nan))
        F2_on[i]  = float(row.get('start_flash_2', np.nan))
        F2_off[i] = float(row.get('end_flash_2',   np.nan))
        choice_start[i] = float(row.get('choice_start', np.nan))
        choice_stop[i]  = float(row.get('choice_stop',  np.nan))
        servo_in[i]     = float(row.get('servo_in',     np.nan))
        servo_out[i]    = float(row.get('servo_out',    np.nan))
        lick_start[i]   = float(row.get('lick_start',   np.nan))

        # behavior flags
        is_right_trial[i]  = bool(row.get('is_right', False))
        is_right_choice[i] = bool(row.get('is_right_choice', False))
        rewarded[i]        = bool(row.get('rewarded', False))
        punished[i]        = bool(row.get('punished', False))
        did_not_choose[i]  = bool(row.get('did_not_choose', False))
        time_dnc[i]        = float(row.get('time_did_not_choose', np.nan))
        lick[i]            = bool(row.get('lick', False))
        RT[i]              = float(row.get('RT', np.nan))

        # dff arrays
        t_vec = np.asarray(row['dff_time_vector'], dtype=float)
        dff   = np.asarray(row['dff_segment'],     dtype=float)
        if dff.ndim != 2 or dff.shape[0] != n_rois:
            raise ValueError(f"Trial {i}: dff_segment shape mismatch; expected (n_rois, n_samples) with n_rois={n_rois}, got {dff.shape}.")

        # enforce strictly increasing time vector
        order = _require_monotonic_time(t_vec)
        if order is not None:
            t_vec = t_vec[order]
            dff   = dff[:, order]

        # --- F1-OFF aligned (pre-F2 masked)
        if want_f1:
            if not np.isfinite(F1_off[i]):
                msg = f"Trial {i}: missing end_flash_1 for F1-OFF alignment."
                if err_on_miss: raise ValueError(msg)
                else: print("[warn]", msg)
            else:
                t_rel = t_vec - F1_off[i]
                tmin, tmax = float(np.nanmin(t_rel)), float(np.nanmax(t_rel))
                if verbose and i == 0:
                    print(f"[F1OFF] trial0 native aligned span: {tmin:.3f}..{tmax:.3f}s")
                for r in range(n_rois):
                    vals = _interp_to_grid(t_rel, dff[r], time_f1)
                    roi_traces[i, r] = vals
                    nans_f1 += np.isnan(vals).sum()
                # mask t >= ISI to ensure strictly pre-F2 content
                if f1_mask_preF2:
                    isi = float(isi_vec[i])
                    roi_traces[i, :, time_f1 >= isi] = np.nan

        # --- F2-locked grid
        if build_f2:
            if not np.isfinite(F2_on[i]):
                missing_f2.append(i)
            else:
                t_rel = t_vec - F2_on[i]
                for r in range(n_rois):
                    X_f2[i, r] = _interp_to_grid(t_rel, dff[r], tau_f2)

        # --- choice-locked grid
        if build_choice:
            if not np.isfinite(choice_start[i]):
                missing_choice.append(i)
            else:
                t_rel = t_vec - choice_start[i]
                for r in range(n_rois):
                    X_choice[i, r] = _interp_to_grid(t_rel, dff[r], tau_choice)

        # --- servo_in locked grid
        if build_servo_in:
            if not np.isfinite(servo_in[i]):
                missing_servo_in.append(i)
            else:
                t_rel = t_vec - servo_in[i]
                for r in range(n_rois):
                    X_servo_in[i, r] = _interp_to_grid(t_rel, dff[r], tau_servo_in)

        # --- servo_out locked grid
        if build_servo_out:
            if not np.isfinite(servo_out[i]):
                missing_servo_out.append(i)
            else:
                t_rel = t_vec - servo_out[i]
                for r in range(n_rois):
                    X_servo_out[i, r] = _interp_to_grid(t_rel, dff[r], tau_servo_out)

        if verbose and (i % 50 == 0 or i == n_trials - 1):
            print(f"[loop] trial {i+1}/{n_trials} processed")

    # --- post-loop QA summaries
    qc = {}
    if want_f1:
        total = roi_traces.size
        nans = int(np.isnan(roi_traces).sum())
        qc['f1off_nan_frac'] = nans / max(1, total)
        print(f"[QA:F1OFF] NaNs: {nans}/{total} ({100*qc['f1off_nan_frac']:.1f}%)")

    if build_f2 and missing_f2:
        msg = f"[QA:F2] missing F2_on in {len(missing_f2)}/{n_trials} trials; idx[:10]={missing_f2[:10]}"
        if err_on_miss: raise ValueError(msg)
        else: print("[warn]", msg)

    if build_choice and missing_choice:
        msg = f"[QA:CHOICE] missing choice_start in {len(missing_choice)}/{n_trials} trials; idx[:10]={missing_choice[:10]}"
        if err_on_miss: raise ValueError(msg)
        else: print("[warn]", msg)

    if build_servo_in and missing_servo_in:
        msg = f"[QA:SERVO_IN] missing servo_in in {len(missing_servo_in)}/{n_trials} trials; idx[:10]={missing_servo_in[:10]}"
        if err_on_miss: raise ValueError(msg)
        else: print("[warn]", msg)

    if build_servo_out and missing_servo_out:
        msg = f"[QA:SERVO_OUT] missing servo_out in {len(missing_servo_out)}/{n_trials} trials; idx[:10]={missing_servo_out[:10]}"
        if err_on_miss: raise ValueError(msg)
        else: print("[warn]", msg)

    # --- build M dict
    M = dict(
        # labels & timing
        isi=isi_vec.astype(np.float32),
        isi_allowed=unique_isis.astype(np.float32),
        short_levels=short_isis.astype(np.float32),
        long_levels=long_isis.astype(np.float32),
        is_short=is_short.astype(bool),

        # behavior arrays
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

        # meta
        n_trials=n_trials, n_rois=n_rois,
        cfg_used=cfg,
        qc=qc,
    )

    # primary F1-OFF grid (if built)
    if want_f1:
        M['roi_traces'] = roi_traces
        M['time'] = time_f1
        M['F2_time'] = M['isi']  # alias kept for downstream code that expects this name

    # optional grids
    if build_f2:
        M['roi_traces_F2locked'] = X_f2
        M['time_F2locked'] = tau_f2
        M['servo_in_relF2'] = (servo_in - F2_on).astype(np.float32)
        M['lick_start_relF2'] = (lick_start - F2_on).astype(np.float32)

    if build_choice:
        M['roi_traces_choice'] = X_choice
        M['time_choice'] = tau_choice
        M['lick_start_relChoice'] = (lick_start - choice_start).astype(np.float32)

    if build_servo_in:
        M['roi_traces_servo'] = X_servo_in
        M['time_servo'] = tau_servo_in
        M['lick_start_relServo'] = (lick_start - servo_in).astype(np.float32)

    if build_servo_out:
        M['roi_traces_servo_out'] = X_servo_out
        M['time_servo_out'] = tau_servo_out
        M['lick_start_relServoOut'] = (lick_start - servo_out).astype(np.float32)

    # --- wrap-up summary
    if qa.get("print_summary", True):
        print("----- build_M_from_trials SUMMARY -----")
        print(f"Trials: {n_trials}, ROIs: {n_rois}")
        print(f"ISI allowed (s): {unique_isis}")
        print(f"Short levels: {short_isis}  |  Long levels: {long_isis}")
        if want_f1:
            print(f"F1-OFF grid: shape {M['roi_traces'].shape}, time {M['time'][0]:.3f}..{M['time'][-1]:.3f}s")
        if build_f2:
            print(f"F2 grid: shape {M['roi_traces_F2locked'].shape}, window {tau_f2[0]:.3f}..{tau_f2[-1]:.3f}s")
        if build_choice:
            print(f"Choice grid: shape {M['roi_traces_choice'].shape}, window {tau_choice[0]:.3f}..{tau_choice[-1]:.3f}s")
        if build_servo_in:
            print(f"Servo_in grid: shape {M['roi_traces_servo'].shape}, window {tau_servo_in[0]:.3f}..{tau_servo_in[-1]:.3f}s")
        if build_servo_out:
            print(f"Servo_out grid: shape {M['roi_traces_servo_out'].shape}, window {tau_servo_out[0]:.3f}..{tau_servo_out[-1]:.3f}s")
        print("---------------------------------------")

    print("=== build_M_from_trials: DONE ===\n")
    return M



from copy import deepcopy
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge

def scaling_vs_clock(M: dict, cfg: dict) -> dict:
    """
    Compare 'clock' (absolute time) vs 'scaling' (phase = t / ISI) models
    for each ROI using sliding windows on the F1-OFF grid (pre-F2 region).

    Adds M['svclock'] with:
        win_centers_s, win_bounds_s, r2_time, r2_phase, delta_r2, labels, per_window_stats
    and (extra) corr_time, corr_phase, delta_corr, trials_used_per_window
    """
    sv = deepcopy(cfg.get('svclock', {}))
    verbose        = bool(sv.get('verbose', True))
    preF2_window   = tuple(map(float, sv.get('preF2_window', (0.15, 0.70))))
    win_sec        = float(sv.get('win_sec', 0.12))
    stride_frac    = float(sv.get('stride_frac', 0.25))
    cv_folds       = int(sv.get('cv_folds', 5))
    knots_time     = int(sv.get('knots_time', 8))
    knots_phase    = int(sv.get('knots_phase', 8))
    alpha          = float(sv.get('alpha', 1.0))
    min_trials     = int(sv.get('min_trials', 10))
    min_bins_per_trial = int(sv.get('min_bins_per_trial', 3))
    delta_r2_eps   = float(sv.get('delta_r2_eps', 0.02))
    # New stability controls
    target_centering = str(sv.get('target_centering', 'per_trial')).lower()   # 'none'|'per_trial'
    var_floor_frac_iqr = float(sv.get('r2_var_floor_frac_iqr', 0.05))          # variance floor as (frac * IQR)^2
    report_corr    = bool(sv.get('report_corr', True))

    # --- Preconditions
    if 'roi_traces' not in M or 'time' not in M:
        raise KeyError("M must contain F1-OFF grid: 'roi_traces' and 'time'.")
    X = np.asarray(M['roi_traces'], float)         # (n_trials, n_rois, Nt)
    T = np.asarray(M['time'], float)               # (Nt,)
    isi = np.asarray(M['isi'], float)              # (n_trials,)
    n_trials, n_rois, Nt = X.shape
    dt = float(np.nanmedian(np.diff(T))) if Nt > 1 else np.nan

    if verbose:
        print("=== scaling_vs_clock: START ===")
        print(f"[svclock] trials={n_trials}, rois={n_rois}, Nt={Nt}, dt={dt:.4f}s")
        print(f"[svclock] preF2_window=({preF2_window[0]:.3f},{preF2_window[1]:.3f})s | win={win_sec:.3f}s stride={stride_frac:.2f}×win")
        print(f"[svclock] CV folds={cv_folds}, knots(time/phase)={knots_time}/{knots_phase}, alpha={alpha}")
        print(f"[svclock] min_trials={min_trials}, min_bins_per_trial={min_bins_per_trial}, ΔR² eps={delta_r2_eps:.3f}")
        print(f"[svclock] centering={target_centering}, var_floor=( {var_floor_frac_iqr:.2f} * IQR )², report_corr={report_corr}")

    # --- Build sliding windows within requested analysis window
    w0, w1 = preF2_window
    if w1 - w0 < win_sec - 1e-12:
        raise ValueError("preF2_window shorter than win_sec.")
    step = win_sec * stride_frac
    starts = []
    s = w0
    while s + win_sec <= w1 + 1e-12:
        starts.append(s)
        s += step
    starts = np.array(starts, float)
    stops  = starts + win_sec
    centers = (starts + stops) / 2.0
    idx_list = [(T >= a) & (T <= b) for (a, b) in zip(starts, stops)]
    nW = len(starts)
    if verbose:
        print(f"[svclock] windows={nW} (width={win_sec:.3f}s, step={step:.3f}s)")

    # --- Pre-make spline bases (fitted on the whole domain of predictors)
    # Note: fitting SplineTransformer does not use y, so using all rows is fine.
    # We'll transform per-window with the appropriate t/phi slices.
    spl_time  = SplineTransformer(n_knots=knots_time, degree=3, include_bias=False)
    spl_phase = SplineTransformer(n_knots=knots_phase, degree=3, include_bias=False)

    # Seed with a representative vector so the transformers learn a reasonable range
    # (They will be re-applied to the per-window predictors.)
    # Use the full T for time, and phase in [T/w_min_ISI, T/w_max_ISI] for a rough range:
    _ = spl_time.fit(T.reshape(-1, 1))
    # avoid division by zero (all ISIs are > 0 in real data)
    minISI, maxISI = np.nanmin(isi[np.isfinite(isi)]), np.nanmax(isi[np.isfinite(isi)])
    _phi_probe = (T / max(minISI, 1e-9)).reshape(-1, 1)
    _ = spl_phase.fit(_phi_probe)

    # --- Outputs
    r2_time  = np.full((n_rois, nW), np.nan, float)
    r2_phase = np.full((n_rois, nW), np.nan, float)
    corr_time  = np.full((n_rois, nW), np.nan, float)
    corr_phase = np.full((n_rois, nW), np.nan, float)
    trials_used_per_window = np.zeros(nW, int)
    perwin_stats = []

    ridge = Ridge(alpha=alpha, fit_intercept=True)

    # --- Utility
    def _fold_scores(y, yhat):
        # classic R^2 on held-out set
        y = np.asarray(y, float)
        yhat = np.asarray(yhat, float)
        mu = np.nanmean(y)
        ss_tot = np.nansum((y - mu) ** 2)
        ss_res = np.nansum((y - yhat) ** 2)
        r2 = 1.0 - (ss_res / np.maximum(ss_tot, 1e-12))
        if report_corr:
            # Pearson r (bounded, more stable for tiny variances)
            ym = y - mu
            xm = yhat - np.nanmean(yhat)
            num = np.nansum(ym * xm)
            den = np.sqrt(np.nansum(ym ** 2) * np.nansum(xm ** 2))
            r = num / np.maximum(den, 1e-12)
            return r2, np.clip(r, -1.0, 1.0)
        return r2, np.nan

    # --- Loop windows → per-ROI CV
    for w, idx_w in enumerate(idx_list):
        t_slice = T[idx_w]                                  # (Nb,)
        if t_slice.size < min_bins_per_trial:
            perwin_stats.append(dict(n_trials_used=0, nbins=0, median_bins_per_trial=0,
                                     heldout_var_median=np.nan, heldout_var_p10=np.nan, heldout_var_p90=np.nan))
            continue

        # Gather per-trial finite masks (independent of ROI) to know which trials have enough support
        # We’ll still re-check per ROI inside, but this gives us a quick count.
        # (Use the first ROI as a probe for availability; robust check is done later per-ROI.)
        probe = np.isfinite(X[:, 0, :][:, idx_w])
        bins_per_trial = probe.sum(axis=1)
        candidate_trials = np.where(bins_per_trial >= min_bins_per_trial)[0]
        trials_used_per_window[w] = candidate_trials.size

        # Print ISI counts summary for a couple of windows
        if verbose and (w in (0, nW-1)):
            # ISI distribution among trials that have enough data in this window
            vals, cnts = np.unique(np.round(isi[candidate_trials], 9), return_counts=True)
            print(f"[svclock:w{w+1}/{nW}] trials used={candidate_trials.size} | "
                  f"ISI counts p10/p50/p90 = "
                  f"{np.percentile(cnts, 10) if cnts.size else 0:.1f}/"
                  f"{np.percentile(cnts, 50) if cnts.size else 0:.1f}/"
                  f"{np.percentile(cnts, 90) if cnts.size else 0:.1f}")
            if cnts.size:
                d = {float(k): int(v) for k, v in zip(vals, cnts)}
                print(f"[svclock:w{w+1}] per-ISI trial counts: {d}")

        # Precompute a variance floor for this window (based on all ROIs/trials bins)
        all_bins_window = X[:, :, :][:, :, idx_w]                        # (n_trials, n_rois, Nb)
        # robust IQR of *values* in this window (over all trials/rois)
        q10, q90 = np.nanpercentile(all_bins_window, [10, 90])
        iqr_est = max(q90 - q10, 1e-9)
        var_floor = (var_floor_frac_iqr * iqr_est) ** 2

        # Now per ROI
        for r in range(n_rois):
            # Collect rows per trial for this ROI
            y_rows   = []
            t_rows   = []
            phi_rows = []
            g_rows   = []

            for tr in candidate_trials:
                y = X[tr, r, idx_w]
                m = np.isfinite(y)
                if m.sum() < min_bins_per_trial:
                    continue
                yy = y[m]
                tt = t_slice[m]
                pp = tt / max(isi[tr], 1e-9)
                if target_centering == 'per_trial':
                    yy = yy - np.nanmean(yy)
                y_rows.append(yy)
                t_rows.append(tt)
                phi_rows.append(pp)
                g_rows.append(np.full(yy.size, tr, int))

            if not y_rows:
                continue

            y_all   = np.concatenate(y_rows)
            t_all   = np.concatenate(t_rows).reshape(-1, 1)
            phi_all = np.concatenate(phi_rows).reshape(-1, 1)
            groups  = np.concatenate(g_rows)

            uniq_trials = np.unique(groups)
            if uniq_trials.size < max(2, min_trials):
                # Not enough trials → leave NaNs
                continue

            # Design matrices
            Xt = spl_time.transform(t_all)          # absolute time features
            Xp = spl_phase.transform(phi_all)       # phase features

            # GroupKFold by trial: no leakage of the same trial across folds
            n_splits = min(cv_folds, uniq_trials.size)
            gkf = GroupKFold(n_splits=n_splits)

            r2_t_folds, r2_p_folds = [], []
            r_t_folds, r_p_folds = [], []

            for tr_idx, te_idx in gkf.split(Xt, y_all, groups):
                y_tr, y_te = y_all[tr_idx], y_all[te_idx]
                Xt_tr, Xt_te = Xt[tr_idx], Xt[te_idx]
                Xp_tr, Xp_te = Xp[tr_idx], Xp[te_idx]

                # Skip fold if held-out variance is too small (unstable R²)
                if np.nanvar(y_te) < var_floor:
                    continue

                # Fit + predict (Ridge with intercept)
                ridge.fit(Xt_tr, y_tr)
                yhat_t = ridge.predict(Xt_te)

                ridge.fit(Xp_tr, y_tr)
                yhat_p = ridge.predict(Xp_te)

                r2_t, rc_t = _fold_scores(y_te, yhat_t)
                r2_p, rc_p = _fold_scores(y_te, yhat_p)

                r2_t_folds.append(r2_t)
                r2_p_folds.append(r2_p)
                r_t_folds.append(rc_t)
                r_p_folds.append(rc_p)

            if len(r2_t_folds) == 0:
                # all folds skipped for stability → leave NaNs
                continue

            r2_time[r,  w] = float(np.nanmean(r2_t_folds))
            r2_phase[r, w] = float(np.nanmean(r2_p_folds))
            if report_corr:
                corr_time[r,  w] = float(np.nanmean(r_t_folds))
                corr_phase[r, w] = float(np.nanmean(r_p_folds))

        # Per-window QA stats
        # Median held-out variance across folds can’t be gathered cheaply here without rerunning;
        # approximate by variance across all per-trial centered bins:
        # heldout_var_med = float(np.nanmedian([np.nanvar(x) for x in X[:, :, idx_w].reshape(n_trials, -1)]))
        # perwin_stats.append(dict(
        #     n_trials_used=int(trials_used_per_window[w]),
        #     nbins=int(t_slice.size),
        #     median_bins_per_trial=int(np.median(bins_per_trial[candidate_trials]) if candidate_trials.size else 0),
        #     heldout_var_median=heldout_var_med,
        #     heldout_var_p10=float(q10),
        #     heldout_var_p90=float(q90),
        #     var_floor=var_floor,
        # ))
        
        # --- Per-window QA stats (safe, no warnings)
        # Compute a robust median per-trial variance using only finite values
        vals_window = X[:, :, idx_w]                    # (n_trials, n_rois, Nb)
        trial_vars = []
        for tr in range(n_trials):
            v = vals_window[tr].ravel()
            v = v[np.isfinite(v)]
            trial_vars.append(np.nan if v.size < 2 else np.nanvar(v))

        heldout_var_med = float(np.nanmedian(trial_vars))

        perwin_stats.append(dict(
            n_trials_used=int(trials_used_per_window[w]),
            nbins=int(t_slice.size),
            median_bins_per_trial=int(
                np.median(bins_per_trial[candidate_trials]) if candidate_trials.size else 0
            ),
            median_trial_var=heldout_var_med,      # renamed (it’s a per-trial var summary)
            window_value_p10=float(q10),           # value percentiles in this window (not variances)
            window_value_p90=float(q90),
            var_floor=var_floor,
        ))
        
        

        if verbose and (w in (0, nW-1)):
            print(f"[svclock:w{w+1}] per-window trials used (approx)={trials_used_per_window[w]} | "
                  f"nbins={t_slice.size} | var_floor={var_floor:.6g}")

        if verbose and n_rois >= 50 and (w == 0 or (w+1) % 5 == 0 or w == nW-1):
            print(f"[svclock] ROI {min(1, n_rois)}/{n_rois} … {min((w+1)*n_rois, n_rois):d}/{n_rois} done for window {w+1}/{nW}")

    # ΔR² and labels
    delta = r2_phase - r2_time
    labels = np.full(n_rois, 'ambiguous', dtype=object)
    med_delta = np.nanmedian(delta, axis=1)
    labels[med_delta >  +delta_r2_eps] = 'scaling_like'
    labels[med_delta <  -delta_r2_eps] = 'clock_like'

    if verbose:
        valid = np.isfinite(delta)
        frac_valid = 100.0 * valid.sum() / max(1, delta.size)
        print(f"[svclock] valid ΔR² points: {valid.sum()}/{delta.size} ({frac_valid:.1f}%)")
        used = np.array(trials_used_per_window, int)
        if used.size:
            p10, p50, p90 = np.percentile(used, [10, 50, 90])
            print(f"[svclock] per-window trials used (median) p10/p50/p90: {p10:.1f} / {p50:.1f} / {p90:.1f}")
        _, cnts = np.unique(labels, return_counts=True)
        lab_counts = {k: int(v) for k, v in zip(np.unique(labels), cnts)}
        print(f"[svclock] ROI labels → {lab_counts}")
        # quick leaderboard
        mean_delta = np.nanmean(delta, axis=1)
        top = np.argsort(-mean_delta)[:10]
        txt = ", ".join([f"r{int(i)}:{mean_delta[i]:+.3f}" for i in top])
        print(f"[svclock] top by mean ΔR² (phase−time): {txt}")
        print("=== scaling_vs_clock: DONE ===\n")

    # Pack outputs (required)
    M['svclock'] = dict(
        win_centers_s=centers.astype(float),
        win_bounds_s=np.c_[starts, stops].astype(float),
        r2_time=r2_time,
        r2_phase=r2_phase,
        delta_r2=delta,
        labels=labels,
        per_window_stats=perwin_stats,
    )
    # Extras that are often useful for QA/plots
    M['svclock']['corr_time']  = corr_time
    M['svclock']['corr_phase'] = corr_phase
    M['svclock']['delta_corr'] = corr_phase - corr_time
    M['svclock']['trials_used_per_window'] = trials_used_per_window.astype(int)
    M['svclock']['params_used'] = dict(
        preF2_window=preF2_window, win_sec=win_sec, stride_frac=stride_frac,
        cv_folds=cv_folds, knots_time=knots_time, knots_phase=knots_phase, alpha=alpha,
        min_trials=min_trials, min_bins_per_trial=min_bins_per_trial, delta_r2_eps=delta_r2_eps,
        target_centering=target_centering, r2_var_floor_frac_iqr=var_floor_frac_iqr,
        report_corr=report_corr
    )
    return M



def plot_svclock_overlays(M, cfg, roi_idx, window_idx=None,
                          phase_bins=60, show_sem=True,
                          equal_ylim=True, save_path=None):
    """
    Overlay per-ISI trial-means for a given ROI in:
      (A) absolute time from F1-OFF, and
      (B) phase φ = t / ISI (0..1), regridded to a common φ axis.
    """
    if 'roi_traces' not in M or 'time' not in M:
        raise KeyError("F1-OFF grid not present in M ('roi_traces'/'time'). Build it first.")

    T = np.asarray(M['time'], float)             # (Nt,)
    X = np.asarray(M['roi_traces'], float)       # (n_trials, n_rois, Nt)
    isi = np.asarray(M['isi'], float)            # (n_trials,)
    n_trials, n_rois, Nt = X.shape

    if not (0 <= roi_idx < n_rois):
        raise IndexError(f"roi_idx {roi_idx} out of range [0,{n_rois-1}]")

    tol = float(cfg.get('labels', {}).get('tolerance_sec', 1e-3))
    levels = np.asarray(M.get('isi_allowed', np.unique(np.round(isi,6))), float)
    groups = []
    for lvl in levels:
        idx = np.where(np.abs(isi - lvl) <= tol)[0]
        if idx.size:
            groups.append((float(lvl), idx))
    if not groups:
        raise ValueError("No ISI groups found; check M['isi'] and cfg['labels']['tolerance_sec'].")

    print("=== svclock overlays ===")
    print(f"ROI={roi_idx}, Nt={Nt}, dt≈{(T[1]-T[0]) if Nt>1 else np.nan:.4f}s")
    print("per-ISI trial counts:")
    print({np.round(lvl,3): int(idx.size) for lvl, idx in groups})

    win_bounds = None
    delta_here = None
    if window_idx is not None and 'svclock' in M:
        wb = np.asarray(M['svclock']['win_bounds_s'], float)  # (Nw, 2)
        if 0 <= window_idx < wb.shape[0]:
            win_bounds = tuple(map(float, wb[window_idx]))
            delta = np.asarray(M['svclock']['delta_r2'], float)  # (n_rois, Nw)
            if delta.ndim == 2 and roi_idx < delta.shape[0]:
                delta_here = float(delta[roi_idx, window_idx])
                print(f"[info] window #{window_idx} bounds={win_bounds}, ΔR²(phase−time) for ROI={delta_here:+.4f}")
        else:
            print(f"[warn] window_idx {window_idx} out of range; skipping shading.")

    # --- absolute time means/SEM ---
    means_time, sems_time = [], []
    for lvl, idx in groups:
        Y = X[idx, roi_idx, :]                          # (n_trials_lvl, Nt)
        mu = np.nanmean(Y, axis=0)                      # (Nt,)
        means_time.append(mu)
        if show_sem:
            n = np.sum(np.isfinite(Y), axis=0)          # (Nt,)
            den = np.sqrt(np.maximum(1, n))             # element-wise protection
            sem = np.nanstd(Y, axis=0) / den
        else:
            sem = None
        sems_time.append(sem)

    # --- phase means/SEM on common grid ---
    phi_grid = np.linspace(0.0, 1.0, int(phase_bins))
    means_phase, sems_phase = [], []
    for lvl, idx in groups:
        Ys = []
        for tr in idx:
            y = X[tr, roi_idx, :]
            mask = np.isfinite(y) & (T <= isi[tr]) & np.isfinite(isi[tr])
            if mask.sum() < 2:
                continue
            phi = T[mask] / isi[tr]
            yv  = y[mask]
            y_interp = np.full_like(phi_grid, np.nan, dtype=float)
            mseg = (phi_grid >= phi.min()) & (phi_grid <= phi.max())
            if mseg.any():
                y_interp[mseg] = np.interp(phi_grid[mseg], phi, yv)
            Ys.append(y_interp)
        if len(Ys) == 0:
            mu_phi = np.full_like(phi_grid, np.nan, dtype=float)
            se_phi = None if not show_sem else np.full_like(phi_grid, np.nan, dtype=float)
        else:
            Yst = np.vstack(Ys)                          # (n_trials_lvl, phase_bins)
            mu_phi = np.nanmean(Yst, axis=0)
            if show_sem:
                n = np.sum(np.isfinite(Yst), axis=0)     # (phase_bins,)
                den = np.sqrt(np.maximum(1, n))
                se_phi = np.nanstd(Yst, axis=0) / den
            else:
                se_phi = None
        means_phase.append(mu_phi); sems_phase.append(se_phi)

    # --- plotting ---
    fig, (ax_time, ax_phase) = plt.subplots(2, 1, figsize=(10, 6), sharex=False,
                                            gridspec_kw={'height_ratios':[1,1]})
    for k, (lvl, _) in enumerate(groups):
        ax_time.plot(T, means_time[k], label=f"ISI={lvl:.3f}s", lw=1.8)
        if show_sem and sems_time[k] is not None:
            m = np.isfinite(means_time[k]) & np.isfinite(sems_time[k])
            if m.any():
                ax_time.fill_between(T[m],
                                     means_time[k][m] - sems_time[k][m],
                                     means_time[k][m] + sems_time[k][m],
                                     alpha=0.15)
    ax_time.axvline(0.0, ls='--', lw=1.0, color='k', alpha=0.6)
    for lvl, _ in groups:
        ax_time.axvline(lvl, ls=':', lw=0.8, alpha=0.4)
    if win_bounds is not None:
        ax_time.axvspan(win_bounds[0], win_bounds[1], color='0.9', alpha=0.5, label='window')
        if delta_here is not None:
            ax_time.text(win_bounds[0], ax_time.get_ylim()[1], f" ΔR²={delta_here:+.3f}",
                         va='top', ha='left', fontsize=9, color='C3')
    ax_time.set_xlim(T[0], T[-1])
    ax_time.set_ylabel('ΔF/F (time)')
    ax_time.set_title(f'ROI {roi_idx}: per-ISI means (F1-OFF time)')

    for k, (lvl, _) in enumerate(groups):
        ax_phase.plot(phi_grid, means_phase[k], label=f"ISI={lvl:.3f}s", lw=1.8)
        if show_sem and sems_phase[k] is not None:
            m = np.isfinite(means_phase[k]) & np.isfinite(sems_phase[k])
            if m.any():
                ax_phase.fill_between(phi_grid[m],
                                      means_phase[k][m] - sems_phase[k][m],
                                      means_phase[k][m] + sems_phase[k][m],
                                      alpha=0.15)
    ax_phase.axvline(1.0, ls=':', lw=0.8, alpha=0.6)
    ax_phase.set_xlim(0, 1.0)
    ax_phase.set_xlabel('Phase (t / ISI)')
    ax_phase.set_ylabel('ΔF/F (phase)')
    ax_phase.set_title('per-ISI means (phase)')

    ax_time.legend(ncol=min(4, len(groups)), fontsize=9, frameon=False)
    ax_phase.legend(ncol=min(4, len(groups)), fontsize=9, frameon=False)
    # Equal y-limits (robust)
    if equal_ylim:
        arrs = []

        # collect Y arrays from both axes, coercing to np.ndarray
        for line in ax_time.get_lines():
            y = np.asarray(line.get_ydata(), dtype=float)
            if y.size:
                arrs.append(y)
        for line in ax_phase.get_lines():
            y = np.asarray(line.get_ydata(), dtype=float)
            if y.size:
                arrs.append(y)

        if arrs:
            # concatenate only finite values
            finite_chunks = [y[np.isfinite(y)] for y in arrs if np.isfinite(y).any()]
            if finite_chunks:
                cat = np.concatenate(finite_chunks)
                if cat.size:
                    lo, hi = np.nanpercentile(cat, (2, 98))
                    pad = (hi - lo) * 0.05 if np.isfinite(hi - lo) else 0.0
                    ylo, yhi = lo - pad, hi + pad
                    ax_time.set_ylim(ylo, yhi)
                    ax_phase.set_ylim(ylo, yhi)
                    print(f"[y] shared limits set to [{ylo:.3f}, {yhi:.3f}]")


    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig





























from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def cluster_roi_profiles(M, cfg, profiles=None):
    cl = dict(cfg.get('cluster', {}))
    if profiles is None:
        profiles = build_roi_profiles(M, cfg)

    TP = np.asarray(profiles['time_profiles'], float)   # (R, nb_t)
    PP = np.asarray(profiles['phase_profiles'], float)  # (R, nb_ph)
    R = TP.shape[0]
    # concatenate features; z-norm per feature (across ROIs) to equalize scales
    feats = np.c_[TP, PP]                               # (R, nb_t+nb_ph)
    mu = np.nanmean(feats, axis=0)
    sd = np.nanstd(feats, axis=0); sd[sd < 1e-9] = 1.0
    feats = (feats - mu[None, :]) / sd[None, :]

    # optionally append median ΔR² from svclock as a 1D cue
    if bool(cl.get('include_svclock_delta', True)) and ('svclock' in M):
        delta = np.asarray(M['svclock']['delta_r2'], float)  # (R, nW)
        if delta.ndim == 2 and delta.shape[0] == R:
            med = np.nanmedian(delta, axis=1).reshape(-1, 1)
            m2  = np.nanmean(med); s2 = np.nanstd(med) if np.nanstd(med)>0 else 1.0
            feats = np.c_[feats, (med - m2)/s2]

    # drop ROIs with all-NaN features
    good = np.isfinite(feats).any(axis=1)
    idx_map = np.where(good)[0]
    Xf = np.where(np.isfinite(feats[good]), feats[good], 0.0)

    # PCA compress
    pca_dim = int(cl.get('pca_dim', 10))
    pca = PCA(n_components=min(pca_dim, Xf.shape[1]), random_state=int(cl.get('random_state', 0)))
    Z = pca.fit_transform(Xf)

    labels = np.full(R, -1, int)
    method = str(cl.get('method', 'gmm')).lower()
    rng = int(cl.get('random_state', 0))

    if method == 'gmm':
        kmin, kmax = map(int, cl.get('k_range', (2, 8)))
        best_bic, best = np.inf, None
        for k in range(kmin, kmax+1):
            gm = GaussianMixture(n_components=k, covariance_type='full',
                                 random_state=rng, n_init=3)
            gm.fit(Z)
            bic = gm.bic(Z)
            if bic < best_bic:
                best_bic, best = bic, gm
        gmm = best
        lab_good = gmm.predict(Z)
        k_used = int(gmm.n_components)
    else:
        # kmeans: pick k by silhouette-like heuristic on inertia elbow
        kmin, kmax = map(int, cl.get('k_range', (3, 8)))
        inertias = []
        models = []
        for k in range(kmin, kmax+1):
            km = KMeans(n_clusters=k, n_init='auto', random_state=rng)
            km.fit(Z)
            inertias.append(km.inertia_); models.append(km)
        # simple elbow: argmin of second derivative
        d1 = np.diff(inertias); d2 = np.diff(d1)
        if d2.size > 0:
            k_idx = np.argmin(d2) + 1  # +1 to map to k indexing after two diffs
            km = models[k_idx]
        else:
            km = models[0]
        lab_good = km.labels_; k_used = int(km.n_clusters)

    labels[idx_map] = lab_good

    # cluster summaries in original profile spaces
    K = int(np.max(labels) + 1) if labels.max() >= 0 else 0
    cluster_time = np.full((K, TP.shape[1]), np.nan, float)
    cluster_phase = np.full((K, PP.shape[1]), np.nan, float)
    counts = np.zeros(K, int)
    for k in range(K):
        m = labels == k
        counts[k] = int(m.sum())
        if counts[k] > 0:
            cluster_time[k]  = np.nanmean(TP[m], axis=0)
            cluster_phase[k] = np.nanmean(PP[m], axis=0)

    # cross-tab with svclock labels if present
    sv_tab = None
    if 'svclock' in M and 'labels' in M['svclock'] and K > 0:
        sv_lab = np.asarray(M['svclock']['labels'], object)
        keys = ['scaling_like','clock_like','ambiguous']
        sv_tab = {k: np.array([np.sum((labels==c) & (sv_lab==k)) for c in range(K)], int) for k in keys}

    M['cluster'] = dict(
        params_used=cl,
        labels=labels.astype(int),
        k_used=k_used,
        pca_explained=pca.explained_variance_ratio_.tolist(),
        idx_good=idx_map.astype(int),
        centroids=dict(time=cluster_time, phase=cluster_phase),
        counts=counts,
        svclock_crosstab=sv_tab,
        profiles=dict(t_grid=profiles['t_grid'], ph_grid=profiles['ph_grid'])
    )
    if bool(cl.get('verbose', True)):
        print(f"[cluster] method={method}, k={k_used}, assigned={int(good.sum())}/{R} ROIs")
        if K>0:
            print("[cluster] cluster sizes:", counts.tolist())
    return M


import matplotlib.pyplot as plt

def plot_cluster_profiles(M, cfg, ylim='shared'):
    C = M.get('cluster', {})
    if not C:
        raise KeyError("Run cluster_roi_profiles(M,cfg) first.")
    timeC = np.asarray(C['centroids']['time'], float)
    phaseC = np.asarray(C['centroids']['phase'], float)
    counts = np.asarray(C['counts'], int)
    t = np.asarray(C['profiles']['t_grid'], float)
    ph = np.asarray(C['profiles']['ph_grid'], float)
    K = timeC.shape[0]
    if K == 0:
        print("[plot] no clusters to display."); return None

    fig, axes = plt.subplots(K, 2, figsize=(10, 3.0*K), sharex=False, sharey=(ylim=='shared'))
    axes = np.atleast_2d(axes)
    for k in range(K):
        ax1, ax2 = axes[k, 0], axes[k, 1]
        ax1.plot(t, timeC[k], lw=2)
        ax1.axvline(0, ls='--', alpha=0.6, lw=1)
        ax1.set_ylabel(f'ΔF/F\n(k={counts[k]})' if k==0 else f'(k={counts[k]})')
        ax1.set_title(f'Cluster {k}: time')
        ax2.plot(ph, phaseC[k], lw=2)
        ax2.axvline(1.0, ls=':', alpha=0.6, lw=1)
        ax2.set_title('phase')
        if k == K-1:
            ax1.set_xlabel('Time from F1-OFF (s)')
            ax2.set_xlabel('Phase (t / ISI)')
    plt.tight_layout()
    return fig





def _phase_regrid_one(y, T, isi, phase_grid):
    """NaN-safe regrid of a single ROI's trial trace to a common phase grid."""
    if not (np.isfinite(isi) and isi > 0): 
        return np.full_like(phase_grid, np.nan, float)
    m = np.isfinite(y) & (T >= 0) & (T <= isi)
    if m.sum() < 2:
        return np.full_like(phase_grid, np.nan, float)
    phi = T[m] / isi
    yy  = y[m]
    out = np.full_like(phase_grid, np.nan, float)
    seg = (phase_grid >= phi.min()) & (phase_grid <= phi.max())
    if seg.any():
        out[seg] = np.interp(phase_grid[seg], phi, yy)
    return out

def build_roi_profiles(M, cfg):
    """Return per-ROI mean profiles in absolute time and phase (pre-F2)."""
    cl = dict(cfg.get('cluster', {}))
    pre = tuple(map(float, cl.get('preF2_window', (0.15, 0.70))))
    nb_t  = int(cl.get('time_bins', 30))
    nb_ph = int(cl.get('phase_bins', 30))

    X = np.asarray(M['roi_traces'], float)   # (n_trials, n_rois, Nt) pre-F2 masked already
    T = np.asarray(M['time'], float)         # (Nt,)
    isi = np.asarray(M['isi'], float)
    n_trials, n_rois, Nt = X.shape

    # absolute time grid and mask
    t_grid = np.linspace(pre[0], pre[1], nb_t).astype(float) if nb_t>1 else np.array([(pre[0]+pre[1])/2.0])
    # phase grid
    ph_grid = np.linspace(0.0, 1.0, nb_ph).astype(float) if nb_ph>1 else np.array([0.5])

    # --- accumulate trial means per ROI in time ---
    time_profiles = np.full((n_rois, t_grid.size), np.nan, float)
    for r in range(n_rois):
        Ys = []
        for tr in range(n_trials):
            y = X[tr, r]
            if not np.isfinite(y).any(): 
                continue
            # restrict to window, then interp to t_grid
            m = np.isfinite(y) & (T >= pre[0]) & (T <= pre[1])
            if m.sum() < 2: 
                continue
            yv = y[m]; tv = T[m]
            yy = np.interp(t_grid, tv, yv, left=np.nan, right=np.nan)
            Ys.append(yy)
        if Ys:
            Y = np.vstack(Ys)
            time_profiles[r] = np.nanmean(Y, axis=0)

    # --- accumulate trial means per ROI in phase ---
    phase_profiles = np.full((n_rois, ph_grid.size), np.nan, float)
    for r in range(n_rois):
        Ys = []
        for tr in range(n_trials):
            y = X[tr, r]
            yy = _phase_regrid_one(y, T, isi[tr], ph_grid)
            if np.isfinite(yy).any():
                Ys.append(yy)
        if Ys:
            Y = np.vstack(Ys)
            phase_profiles[r] = np.nanmean(Y, axis=0)

    return dict(
        t_grid=t_grid, ph_grid=ph_grid,
        time_profiles=time_profiles,     # shape (n_rois, nb_t)
        phase_profiles=phase_profiles    # shape (n_rois, nb_ph)
    )



import numpy as np
import matplotlib.pyplot as plt

def _profiles_from_M(M, cfg):
    # reuse your existing builder so plots don’t depend on what cluster stored
    prof = build_roi_profiles(M, cfg)
    TP = np.asarray(prof['time_profiles'], float)   # (R, nb_t)
    PP = np.asarray(prof['phase_profiles'], float)  # (R, nb_ph)
    t  = np.asarray(prof['t_grid'], float)
    ph = np.asarray(prof['ph_grid'], float)
    return TP, PP, t, ph

def plot_cluster_spaghetti(M, cfg, which='both'):
    """Thin per-ROI curves + bold centroid for each cluster."""
    if 'cluster' not in M or 'labels' not in M['cluster']:
        raise KeyError("Run cluster_roi_profiles(M, cfg) first.")
    labels = np.asarray(M['cluster']['labels'], int)
    counts = np.asarray(M['cluster']['counts'], int)
    cent_t = np.asarray(M['cluster']['centroids']['time'], float)
    cent_p = np.asarray(M['cluster']['centroids']['phase'], float)
    TP, PP, t, ph = _profiles_from_M(M, cfg)

    K = int(labels.max()+1) if labels.size else 0
    ps = cfg.get('plots', {}).get('cluster', {})
    Kshow = K if K>0 else 0
    show_time = which in ('both','time')
    show_phase= which in ('both','phase')

    figs = []
    if show_time:
        fig_t, axes_t = plt.subplots(Kshow, 1, figsize=(8, 2.2*Kshow), squeeze=False)
        for k in range(Kshow):
            ax = axes_t[k,0]
            m = (labels == k)
            idx = np.where(m)[0]
            if idx.size:
                # limit spaghetti for legibility
                nmax = int(ps.get('max_spaghetti', 60))
                take = idx[:min(nmax, idx.size)]
                ax.plot(t, TP[take].T, lw=0.6, alpha=0.25)
                ax.plot(t, cent_t[k], lw=2.2)
            ax.axvline(0, ls='--', lw=1, alpha=0.6)
            ax.set_ylabel(f'ΔF/F (k={int(counts[k])})')
            if k==Kshow-1: ax.set_xlabel('Time from F1-OFF (s)')
            ax.set_title(f'Cluster {k} — time')
        # robust shared y
        vals = np.concatenate([cent_t[np.isfinite(cent_t)]]) if np.isfinite(cent_t).any() else np.array([])
        if vals.size:
            p0,p1 = ps.get('ylim_percentiles',(2,98))
            lo,hi = np.nanpercentile(vals,(p0,p1)); pad=(hi-lo)*ps.get('ylim_pad_frac',0.05)
            for k in range(Kshow): axes_t[k,0].set_ylim(lo-pad, hi+pad)
        plt.tight_layout()
        figs.append(fig_t)

    if show_phase:
        fig_p, axes_p = plt.subplots(Kshow, 1, figsize=(8, 2.2*Kshow), squeeze=False)
        for k in range(Kshow):
            ax = axes_p[k,0]
            m = (labels == k)
            idx = np.where(m)[0]
            if idx.size:
                nmax = int(ps.get('max_spaghetti', 60))
                take = idx[:min(nmax, idx.size)]
                ax.plot(ph, PP[take].T, lw=0.6, alpha=0.25)
                ax.plot(ph, cent_p[k], lw=2.2)
            ax.axvline(1.0, ls=':', lw=1, alpha=0.6)
            ax.set_ylabel(f'ΔF/F (k={int(counts[k])})')
            if k==Kshow-1: ax.set_xlabel('Phase (t/ISI)')
            ax.set_title(f'Cluster {k} — phase')
        vals = np.concatenate([cent_p[np.isfinite(cent_p)]]) if np.isfinite(cent_p).any() else np.array([])
        if vals.size:
            p0,p1 = ps.get('ylim_percentiles',(2,98))
            lo,hi = np.nanpercentile(vals,(p0,p1)); pad=(hi-lo)*ps.get('ylim_pad_frac',0.05)
            for k in range(Kshow): axes_p[k,0].set_ylim(lo-pad, hi+pad)
        plt.tight_layout()
        figs.append(fig_p)
    return figs



def plot_cluster_heatmaps(M, cfg, which='both', order='slope'):
    """Heatmaps of ROI profiles within each cluster, ordered for visibility."""
    if 'cluster' not in M or 'labels' not in M['cluster']:
        raise KeyError("Run cluster_roi_profiles(M, cfg) first.")
    labels = np.asarray(M['cluster']['labels'], int)
    cent_t = np.asarray(M['cluster']['centroids']['time'], float)
    cent_p = np.asarray(M['cluster']['centroids']['phase'], float)
    TP, PP, t, ph = _profiles_from_M(M, cfg)
    K = int(labels.max()+1)

    figs = []
    def _sort_idx(Mat):
        # order by slope over the window (or mean)
        if order == 'slope':
            x = np.linspace(0,1,Mat.shape[1])
            num = (Mat * (x - x.mean())).sum(1)
            den = ((x - x.mean())**2).sum()
            s = num / np.maximum(den, 1e-12)
            return np.argsort(s)  # low→high
        else:
            m = np.nanmean(Mat, axis=1)
            return np.argsort(m)

    if which in ('both','time'):
        fig, axes = plt.subplots(1, K, figsize=(3.2*K, 3), squeeze=False)
        for k in range(K):
            ax = axes[0,k]
            idx = np.where(labels==k)[0]
            if idx.size:
                Mat = TP[idx]
                order_idx = _sort_idx(Mat)
                ax.imshow(Mat[order_idx, :], aspect='auto', interpolation='nearest')
            ax.set_title(f'k={k}')
            ax.set_xlabel('time bins')
            if k==0: ax.set_ylabel('ROIs')
        plt.tight_layout(); figs.append(fig)

    if which in ('both','phase'):
        fig, axes = plt.subplots(1, K, figsize=(3.2*K, 3), squeeze=False)
        for k in range(K):
            ax = axes[0,k]
            idx = np.where(labels==k)[0]
            if idx.size:
                Mat = PP[idx]
                order_idx = _sort_idx(Mat)
                ax.imshow(Mat[order_idx, :], aspect='auto', interpolation='nearest')
            ax.set_title(f'k={k}')
            ax.set_xlabel('phase bins')
            if k==0: ax.set_ylabel('ROIs')
        plt.tight_layout(); figs.append(fig)
    return figs




def plot_cluster_perISI_overlays(M, cfg, which='time'):
    """
    For each cluster, overlay per-ISI *cluster means* (avg over trials & ROIs in cluster).
    Great for seeing if responses are aligned in time vs phase without fitting models.
    """
    labels = np.asarray(M['cluster']['labels'], int)
    K = int(labels.max()+1)
    T = np.asarray(M['time'], float)
    X = np.asarray(M['roi_traces'], float)      # (trials, rois, Nt)
    isi = np.asarray(M['isi'], float)
    levels = np.asarray(M.get('isi_allowed', np.unique(np.round(isi,6))), float)
    tol = float(cfg.get('labels', {}).get('tolerance_sec', 1e-3))
    pre = tuple(map(float, cfg.get('cluster', {}).get('preF2_window', (0.15, 0.70))))
    mwin = (T >= pre[0]) & (T <= pre[1])

    figs = []
    for k in range(K):
        idx_roi = np.where(labels==k)[0]
        if idx_roi.size == 0: 
            continue
        fig, ax = plt.subplots(1,1, figsize=(6,3))
        for lvl in levels:
            tr_idx = np.where(np.abs(isi - lvl) <= tol)[0]
            if tr_idx.size == 0: 
                continue
            # cluster mean over trials×ROIs, in absolute time
            Y = X[np.ix_(tr_idx, idx_roi, mwin)]       # (ntr, nroi, nb)
            mu = np.nanmean(Y, axis=(0,1))
            ax.plot(T[mwin], mu, label=f'ISI={lvl:.3f}s', lw=1.8)
        ax.axvline(0, ls='--', lw=1, alpha=0.6)
        for lvl in levels: ax.axvline(lvl, ls=':', lw=0.8, alpha=0.4)
        ax.set_title(f'Cluster {k} — per-ISI overlays (time)')
        ax.set_xlabel('Time from F1-OFF (s)'); ax.set_ylabel('ΔF/F')
        ax.legend(frameon=False, ncol=min(3, levels.size))
        plt.tight_layout(); figs.append(fig)
    return figs




from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist

def compute_cluster_metrics(M, cfg):
    """Return a small dict with intuitive cluster quality numbers."""
    if 'cluster' not in M or 'labels' not in M['cluster']:
        raise KeyError("Run cluster_roi_profiles(M, cfg) first.")
    labels = np.asarray(M['cluster']['labels'], int)
    counts = np.asarray(M['cluster']['counts'], int)
    TP, PP, t, ph = _profiles_from_M(M, cfg)
    # feature space = concatenated time+phase profiles (z-norm per feature)
    feats = np.c_[TP, PP]
    mu = np.nanmean(feats, axis=0)
    sd = np.nanstd(feats, axis=0); sd[sd<1e-9]=1.0
    Xf = np.where(np.isfinite(feats), (feats-mu)/sd, 0.0)

    good = labels >= 0
    Xg, lg = Xf[good], labels[good]
    if Xg.shape[0] < 5 or np.unique(lg).size < 2:
        return {'sizes': counts.tolist(), 'note': 'not enough data for silhouettes'}

    # silhouettes in PCA space (keeps runtime tiny)
    pca = PCA(n_components=min(10, Xg.shape[1]), random_state=int(cfg.get('qa',{}).get('random_state',0)))
    Z = pca.fit_transform(Xg)
    sil_overall = float(silhouette_score(Z, lg))
    sil_each = []
    for k in range(int(lg.max()+1)):
        mk = (lg==k)
        sil_each.append(float(np.nanmean(silhouette_samples(Z, lg)[mk])) if mk.any() else np.nan)

    # compactness/separation in original (time+phase) mean-centered space
    K = int(labels.max()+1)
    centroids = np.vstack([np.nanmean(Xf[labels==k], axis=0) for k in range(K)])
    within = []
    for k in range(K):
        idx = np.where(labels==k)[0]
        if idx.size:
            d = cdist(Xf[idx], centroids[k:k+1], metric='euclidean')**2
            within.append(float(np.nanmean(d)))
        else:
            within.append(np.nan)
    between = cdist(centroids, centroids, metric='euclidean')
    np.fill_diagonal(between, np.nan)
    sep_min = float(np.nanmin(between)) if np.isfinite(between).any() else np.nan

    out = dict(
        sizes=counts.tolist(),
        silhouette_overall=sil_overall,
        silhouette_per_cluster=sil_each,
        within_mse_per_cluster=within,
        min_centroid_separation=sep_min,
        pca_explained=np.asarray(M['cluster'].get('pca_explained', [])).tolist()
    )
    print("[cluster metrics]")
    print(" sizes:", out['sizes'])
    print(f" silhouette overall: {out['silhouette_overall']:.3f}")
    print(" silhouette per cluster:", ["{:.3f}".format(x) if np.isfinite(x) else "nan" for x in out['silhouette_per_cluster']])
    print(" within-cluster MSE:", ["{:.4f}".format(x) if np.isfinite(x) else "nan" for x in out['within_mse_per_cluster']])
    print(f" min centroid separation: {out['min_centroid_separation']:.3f}")
    return out



# --- quick cluster metrics + plots (drop-in) -------------------------------
import numpy as np
import matplotlib.pyplot as plt

def _linear_fit_stats(x, y):
    """Return slope and R^2 of LS line (NaN-safe)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return np.nan, np.nan
    x0 = x[m] - x[m].mean(); y0 = y[m] - y[m].mean()
    s = float(np.dot(x0, y0) / (np.dot(x0, x0) + 1e-12))
    yhat = s * x0
    r2 = 1.0 - float(np.sum((y0 - yhat)**2) / (np.sum(y0**2) + 1e-12))
    return s, max(min(r2, 1.0), -1.0)

def cluster_quickstats(M, cfg):
    """Compute per-ROI slopes, dynamic range, ramp R^2 (time & phase) and summarize per cluster."""
    assert 'cluster' in M and 'labels' in M['cluster'], "Run cluster_roi_profiles(M,cfg) first."
    labels = np.asarray(M['cluster']['labels'], int)
    TP = np.asarray(build_roi_profiles(M, cfg)['time_profiles'],  float)  # (R, nb_t)
    PP = np.asarray(build_roi_profiles(M, cfg)['phase_profiles'], float)  # (R, nb_ph)
    t  = np.asarray(build_roi_profiles(M, cfg)['t_grid'],  float)
    ph = np.asarray(build_roi_profiles(M, cfg)['ph_grid'], float)

    R = TP.shape[0]
    slope_t = np.full(R, np.nan); slope_p = np.full(R, np.nan)
    r2_t    = np.full(R, np.nan); r2_p    = np.full(R, np.nan)
    dr_t    = np.full(R, np.nan); dr_p    = np.full(R, np.nan)

    # ramp templates: decreasing line over the window
    tmpl_t = (t - t.mean());      tmpl_t /= np.sqrt((tmpl_t**2).sum() + 1e-12)
    tmpl_p = (ph - ph.mean());    tmpl_p /= np.sqrt((tmpl_p**2).sum() + 1e-12)

    for r in range(R):
        yT = TP[r]; yP = PP[r]
        # slopes
        slope_t[r], r2_t[r] = _linear_fit_stats(t,  yT)
        slope_p[r], r2_p[r] = _linear_fit_stats(ph, yP)
        # dynamic ranges
        if np.isfinite(yT).any(): dr_t[r] = np.nanpercentile(yT, 95) - np.nanpercentile(yT, 5)
        if np.isfinite(yP).any(): dr_p[r] = np.nanpercentile(yP, 95) - np.nanpercentile(yP, 5)
        # rampiness R^2 vs template (just correlation^2 with a linear ramp)
        # (equivalent to _linear_fit_stats with x = template)
        # already covered by r2_t / r2_p; keeping as-is.

    M['cluster_quickstats'] = dict(
        labels=labels, slope_time=slope_t, slope_phase=slope_p,
        r2_time=r2_t, r2_phase=r2_p, dynrange_time=dr_t, dynrange_phase=dr_p
    )
    # concise per-cluster summary
    K = int(labels.max()+1)
    summary = []
    for k in range(K):
        m = labels == k
        summary.append(dict(
            cluster=int(k),
            n=int(m.sum()),
            slope_time_med=float(np.nanmedian(slope_t[m])),
            slope_phase_med=float(np.nanmedian(slope_p[m])),
            dynrange_time_med=float(np.nanmedian(dr_t[m])),
            dynrange_phase_med=float(np.nanmedian(dr_p[m])),
            rampR2_time_med=float(np.nanmedian(r2_t[m])),
            rampR2_phase_med=float(np.nanmedian(r2_p[m]))
        ))
    M['cluster_quicksummary'] = summary
    print("[cluster quicksummary]")
    for s in summary:
        print(
          f"k={s['cluster']} n={s['n']} | "
          f"slope_t={s['slope_time_med']:+.4f}, slope_p={s['slope_phase_med']:+.4f} | "
          f"Δrange_t={s['dynrange_time_med']:.3f}, Δrange_p={s['dynrange_phase_med']:.3f} | "
          f"R²_t={s['rampR2_time_med']:.2f}, R²_p={s['rampR2_phase_med']:.2f}"
        )
    return M

def plot_cluster_quickstats(M):
    """Box/violin style plots for slopes and dynamic range by cluster; and a slope scatter."""
    Q = M.get('cluster_quickstats', {})
    if not Q: raise KeyError("Run cluster_quickstats(M,cfg) first.")
    lab = np.asarray(Q['labels'], int)
    st, sp = np.asarray(Q['slope_time'], float),  np.asarray(Q['slope_phase'], float)
    dt, dp = np.asarray(Q['dynrange_time'], float), np.asarray(Q['dynrange_phase'], float)

    K = int(lab.max()+1)
    # 1) slopes
    fig1, ax1 = plt.subplots(1,2, figsize=(8,3))
    data_t = [st[lab==k] for k in range(K)]
    data_p = [sp[lab==k] for k in range(K)]
    ax1[0].boxplot([d[np.isfinite(d)] for d in data_t], labels=[f'k{k}' for k in range(K)], showfliers=False)
    ax1[0].axhline(0, ls='--', lw=1, alpha=0.6); ax1[0].set_title('slope (time)')
    ax1[1].boxplot([d[np.isfinite(d)] for d in data_p], labels=[f'k{k}' for k in range(K)], showfliers=False)
    ax1[1].axhline(0, ls='--', lw=1, alpha=0.6); ax1[1].set_title('slope (phase)')
    fig1.tight_layout()

    # 2) dynamic range
    fig2, ax2 = plt.subplots(1,2, figsize=(8,3))
    ax2[0].boxplot([dt[lab==k] for k in range(K)], labels=[f'k{k}' for k in range(K)], showfliers=False)
    ax2[0].set_title('dynamic range (time)')
    ax2[1].boxplot([dp[lab==k] for k in range(K)], labels=[f'k{k}' for k in range(K)], showfliers=False)
    ax2[1].set_title('dynamic range (phase)')
    fig2.tight_layout()

    # 3) slope scatter
    fig3, ax3 = plt.subplots(1,1, figsize=(4.2,3.6))
    for k in range(K):
        m = lab==k
        ax3.scatter(st[m], sp[m], s=8, alpha=0.7, label=f'k{k}')
    ax3.axvline(0, ls='--', lw=1, alpha=0.6); ax3.axhline(0, ls='--', lw=1, alpha=0.6)
    ax3.set_xlabel('slope (time)'); ax3.set_ylabel('slope (phase)'); ax3.set_title('per-ROI slopes')
    ax3.legend(frameon=False)
    fig3.tight_layout()
    return (fig1, fig2, fig3)
# ---------------------------------------------------------------------------


def cluster_phase_preference(M):
    Q = M['cluster_quickstats']  # from cluster_quickstats(M,cfg)
    lab = np.asarray(Q['labels'], int)
    st = np.abs(np.asarray(Q['slope_time'],  float))
    sp = np.abs(np.asarray(Q['slope_phase'], float))
    pi = (sp - st) / (sp + st + 1e-12)  # [-1,1]
    # compact summary per cluster
    K = int(lab.max() + 1)
    summ = []
    for k in range(K):
        m = lab == k
        summ.append(dict(
            cluster=int(k), n=int(m.sum()),
            PI_median=float(np.nanmedian(pi[m])),
            PI_iqr=tuple(np.nanpercentile(pi[m], [25, 75]).tolist())
        ))
    M['cluster_PI'] = dict(values=pi, summary=summ)
    print("[cluster PI]", summ)
    return M

def plot_cluster_PI(M):
    import matplotlib.pyplot as plt, numpy as np
    PI = np.asarray(M['cluster_PI']['values'], float)
    lab = np.asarray(M['cluster_quickstats']['labels'], int)
    K = int(lab.max()+1)
    fig, ax = plt.subplots(1,1, figsize=(4.2,3.2))
    for k in range(K):
        vals = PI[lab==k]
        ax.hist(vals[np.isfinite(vals)], bins=20, alpha=0.6, label=f'k{k}', density=True)
    ax.axvline(0, ls='--', lw=1); ax.set_xlabel('Phase Preference Index (π)')
    ax.set_ylabel('density'); ax.legend(frameon=False); ax.set_title('phase vs time preference')
    fig.tight_layout(); return fig


import numpy as np

def cluster_alignment_scores(M, cfg):
    prof = build_roi_profiles(M, cfg)
    TP = np.asarray(prof['time_profiles'],  float)  # (R, nb_t)
    PP = np.asarray(prof['phase_profiles'], float)  # (R, nb_ph)
    lab = np.asarray(M['cluster']['labels'], int)
    K = int(lab.max()+1)

    def _med_corr_to_centroid(Mat, idx):
        if idx.size == 0: return np.nan
        C = np.nanmean(Mat[idx], axis=0)
        # corr of each ROI profile to centroid
        corrs = []
        for i in idx:
            a, b = Mat[i], C
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= 3:
                a0 = a[m] - a[m].mean(); b0 = b[m] - b[m].mean()
                den = np.sqrt((a0*a0).sum() * (b0*b0).sum()) + 1e-12
                corrs.append(float((a0*b0).sum() / den))
        return float(np.nanmedian(corrs)) if corrs else np.nan

    rows = []
    for k in range(K):
        idx = np.where(lab==k)[0]
        rows.append(dict(
            cluster=int(k),
            r_time_med=_med_corr_to_centroid(TP, idx),
            r_phase_med=_med_corr_to_centroid(PP, idx),
            phase_minus_time=lambda rt, rp: None  # placeholder
        ))
    for r in rows:
        r['phase_minus_time'] = float((r['r_phase_med'] or np.nan) - (r['r_time_med'] or np.nan))
    M['cluster_align'] = rows
    print("[cluster align]", rows)
    return M


import numpy as np
import matplotlib.pyplot as plt

def compute_phase_collapse_index(M, cfg, phase_bins=40):
    """Per-ROI PCI = (Var_time − Var_phase) / (Var_time + Var_phase). Positive ⇒ better collapse in phase."""
    T   = np.asarray(M['time'], float)                       # (Nt,)
    X   = np.asarray(M['roi_traces'], float)                 # (trials, rois, Nt)
    isi = np.asarray(M['isi'], float)
    levels = np.asarray(M.get('isi_allowed', np.unique(np.round(isi, 6))), float)
    tol = float(cfg.get('labels', {}).get('tolerance_sec', 1e-3))
    pre = tuple(map(float, cfg.get('svclock', {}).get('preF2_window', (0.15, 0.70))))
    mwin = (T >= pre[0]) & (T <= pre[1])
    T_w  = T[mwin]
    phi_grid = np.linspace(0, 1, int(phase_bins))

    R = X.shape[1]
    Vt = np.full(R, np.nan); Vp = np.full(R, np.nan)

    # gather per-ISI mean profiles (time & phase) for each ROI
    for r in range(R):
        means_t = []   # list of (nb_t,)
        means_p = []   # list of (phase_bins,)
        for lvl in levels:
            tr_idx = np.where(np.abs(isi - lvl) <= tol)[0]
            if tr_idx.size == 0: 
                continue
            Y = X[np.ix_(tr_idx, [r], mwin)].squeeze(1)      # (ntr, nb_t)
            mu_t = np.nanmean(Y, axis=0)                     # (nb_t,)
            if not np.isfinite(mu_t).any(): 
                continue
            means_t.append(mu_t)

            # time→phase for this ISI
            mask = np.isfinite(mu_t) & np.isfinite(lvl) & (T_w <= lvl)
            if mask.sum() >= 3:
                phi = T_w[mask] / lvl
                yi  = mu_t[mask]
                yi_phase = np.full_like(phi_grid, np.nan, float)
                seg = (phi_grid >= phi.min()) & (phi_grid <= phi.max())
                yi_phase[seg] = np.interp(phi_grid[seg], phi, yi)
                means_p.append(yi_phase)

        if len(means_t) >= 3:
            M_t = np.vstack(means_t)                         # (n_isi, nb_t)
            # variance across ISIs at each time bin, then mean across bins (finite only)
            var_t = np.nanvar(M_t, axis=0)
            Vt[r] = float(np.nanmean(var_t))

        if len(means_p) >= 3:
            # stack with matching support (finite per-bin)
            M_p = np.vstack(means_p)                         # (n_isi, phase_bins)
            var_p = np.nanvar(M_p, axis=0)
            Vp[r] = float(np.nanmean(var_p))

    PCI = (Vt - Vp) / (Vt + Vp + 1e-12)
    M['phase_collapse'] = dict(V_time=Vt, V_phase=Vp, PCI=PCI)
    return M




# Do a ROI’s per-ISI means collapse better in phase than in absolute time?
def plot_PCI_by_cluster(M):
    PCI = np.asarray(M['phase_collapse']['PCI'], float)
    lab = np.asarray(M['cluster_quickstats']['labels'], int)
    K = int(lab.max()+1)
    fig, ax = plt.subplots(1,1, figsize=(4.8,3.2))
    for k in range(K):
        vals = PCI[(lab==k) & np.isfinite(PCI)]
        ax.hist(vals, bins=24, alpha=0.65, density=True, label=f'k{k}')
    ax.axvline(0, ls='--', lw=1); ax.set_xlabel('Phase-Collapse Index (PCI)')
    ax.set_ylabel('density'); ax.legend(frameon=False); ax.set_title('collapse: phase vs time')
    fig.tight_layout()
    # quick printout
    for k in range(K):
        vals = PCI[(lab==k) & np.isfinite(PCI)]
        if vals.size:
            q = np.nanpercentile(vals, [25,50,75])
            print(f"k={k}: PCI median={q[1]:+.3f}  IQR=[{q[0]:+.3f},{q[2]:+.3f}]  n={vals.size}")
    return fig



def amplitude_weighted_alignment(M):
    Q   = M['cluster_quickstats']
    lab = np.asarray(Q['labels'], int)
    r_t, r_p = np.asarray(M['cluster_align'][0]['r_time_med']), np.asarray(M['cluster_align'][0]['r_phase_med'])  # not per-ROI
    # per-ROI corr to centroid:
    prof = build_roi_profiles(M, cfg)
    TP, PP = np.asarray(prof['time_profiles'], float), np.asarray(prof['phase_profiles'], float)
    def corr_to_centroid(Mat, idx):
        C = np.nanmean(Mat[idx], axis=0)
        corrs = []
        for i in idx:
            a = Mat[i]; m = np.isfinite(a) & np.isfinite(C)
            if m.sum() < 3: 
                corrs.append(np.nan); continue
            a0 = a[m]-a[m].mean(); c0=C[m]-C[m].mean()
            den = np.sqrt((a0*a0).sum()*(c0*c0).sum())+1e-12
            corrs.append(float((a0*c0).sum()/den))
        return np.array(corrs, float)

    K = int(lab.max()+1)
    out = []
    for k in range(K):
        idx = np.where(lab==k)[0]
        if not idx.size: 
            out.append(dict(cluster=k)); continue
        ct = corr_to_centroid(TP, idx); cp = corr_to_centroid(PP, idx)
        dr_t = np.asarray(Q['dynrange_time'])[idx]; dr_p = np.asarray(Q['dynrange_phase'])[idx]
        wt, wp = ct * dr_t, cp * dr_p
        out.append(dict(
            cluster=k,
            med_wcorr_time=float(np.nanmedian(wt)),
            med_wcorr_phase=float(np.nanmedian(wp)),
            advantage=float(np.nanmedian(wp - wt)),
            n=int(idx.size)
        ))
    M['cluster_align_weighted'] = out
    print("[amplitude-weighted align]", out)
    return M


import numpy as np
import matplotlib.pyplot as plt

def phase_of_min_per_roi(M, cfg, roi_mask=None, phase_bins=60, phase_range=(0.05, 0.98)):
    T   = np.asarray(M['time'], float)
    X   = np.asarray(M['roi_traces'], float)         # (trials, rois, Nt)
    isi = np.asarray(M['isi'], float)
    R   = X.shape[1]
    if roi_mask is None:
        roi_mask = np.ones(R, bool)

    # build per-ROI mean-in-phase across ISIs (trial means first, then ISI mean)
    phi_grid = np.linspace(0, 1, int(phase_bins))
    mins_phi = np.full(R, np.nan)

    for r in np.where(roi_mask)[0]:
        curves = []
        for tr in range(X.shape[0]):
            y = X[tr, r]
            if not np.isfinite(isi[tr]): continue
            m = np.isfinite(y) & (T <= isi[tr])
            if m.sum() < 3: continue
            phi = T[m] / isi[tr]
            yy  = y[m]
            yi  = np.full_like(phi_grid, np.nan, float)
            seg = (phi_grid >= phi.min()) & (phi_grid <= phi.max())
            yi[seg] = np.interp(phi_grid[seg], phi, yy)
            curves.append(yi)
        if len(curves) < 3: 
            continue
        mu = np.nanmean(np.vstack(curves), axis=0)
        # restrict to valid phase window
        vr = (phi_grid >= phase_range[0]) & (phi_grid <= phase_range[1]) & np.isfinite(mu)
        if vr.any():
            mins_phi[r] = float(phi_grid[vr][np.nanargmin(mu[vr])])

    # plot + summary
    fig, ax = plt.subplots(figsize=(4.2,3.2))
    lab = np.asarray(M['cluster']['labels'], int)
    for k in range(int(lab.max()+1)):
        vals = mins_phi[(lab==k) & np.isfinite(mins_phi)]
        if vals.size:
            ax.hist(vals, bins=24, alpha=0.6, density=True, label=f'k{k}')
            q = np.nanpercentile(vals, [25,50,75])
            print(f"k={k}: φ* median={q[1]:.3f}  IQR=[{q[0]:.3f},{q[2]:.3f}]  n={vals.size}")
    ax.axvline(1.0, ls=':', lw=1); ax.set_xlim(0,1)
    ax.set_xlabel('phase of minimum (φ*)'); ax.set_ylabel('density'); ax.set_title('where ramps end')
    ax.legend(frameon=False)
    return mins_phi, fig


def compute_zPCI(M, cfg, phase_bins=40):
    T   = np.asarray(M['time'], float)
    X   = np.asarray(M['roi_traces'], float)
    isi = np.asarray(M['isi'], float)
    levels = np.asarray(M.get('isi_allowed', np.unique(np.round(isi,6))), float)
    tol = float(cfg.get('labels', {}).get('tolerance_sec', 1e-3))
    pre = tuple(map(float, cfg.get('svclock', {}).get('preF2_window', (0.15, 0.70))))
    mwin = (T >= pre[0]) & (T <= pre[1])
    T_w  = T[mwin]
    phi_grid = np.linspace(0,1,int(phase_bins))
    R = X.shape[1]
    Vt, Vp = np.full(R, np.nan), np.full(R, np.nan)

    for r in range(R):
        Ms_t, Ms_p = [], []
        for lvl in levels:
            tr_idx = np.where(np.abs(isi - lvl) <= tol)[0]
            if tr_idx.size == 0: continue
            Y = X[np.ix_(tr_idx, [r], mwin)].squeeze(1)   # (ntr, nb)
            mu_t = np.nanmean(Y, axis=0)
            if not np.isfinite(mu_t).any(): continue
            # z across bins (shape-only)
            mt, st = np.nanmean(mu_t), np.nanstd(mu_t)
            mu_tz = (mu_t - mt) / (st if st>1e-9 else 1.0)
            Ms_t.append(mu_tz)

            mask = np.isfinite(mu_t) & (T_w <= lvl)
            if mask.sum() >= 3:
                phi = T_w[mask] / lvl
                yi  = mu_tz[mask]
                yi_phase = np.full_like(phi_grid, np.nan, float)
                seg = (phi_grid >= phi.min()) & (phi_grid <= phi.max())
                yi_phase[seg] = np.interp(phi_grid[seg], phi, yi)
                Ms_p.append(yi_phase)

        if len(Ms_t) >= 3:
            var_t = np.nanvar(np.vstack(Ms_t), axis=0)
            Vt[r] = float(np.nanmean(var_t))
        if len(Ms_p) >= 3:
            var_p = np.nanvar(np.vstack(Ms_p), axis=0)
            Vp[r] = float(np.nanmean(var_p))

    zPCI = (Vt - Vp) / (Vt + Vp + 1e-12)
    M['phase_collapse_z'] = dict(PCI=zPCI, V_time=Vt, V_phase=Vp)
    # quick print
    lab = np.asarray(M['cluster']['labels'], int)
    for k in range(int(lab.max()+1)):
        vals = zPCI[(lab==k) & np.isfinite(zPCI)]
        if vals.size:
            q = np.nanpercentile(vals, [25,50,75])
            print(f"k={k}: zPCI median={q[1]:+.3f}  IQR=[{q[0]:+.3f},{q[2]:+.3f}]  n={vals.size}")
    return M



import numpy as np

def summarize_phase_minima(M, phi_star, late_thr=0.80, early_thr=0.10):
    lab = np.asarray(M['cluster']['labels'], int)
    K = int(lab.max()+1)
    rows = []
    for k in range(K):
        m = (lab==k) & np.isfinite(phi_star)
        vals = phi_star[m]
        if vals.size:
            q25, q50, q75 = np.nanpercentile(vals, [25,50,75])
            frac_late  = float(np.mean(vals > late_thr))
            frac_early = float(np.mean(vals < early_thr))
            rows.append(dict(cluster=int(k), n=int(vals.size),
                             phi_med=float(q50), phi_iqr=(float(q25), float(q75)),
                             frac_phi_gt_0p8=frac_late, frac_phi_lt_0p1=frac_early))
    M['cluster_phi_min'] = rows
    print("[φ* summary]", rows)
    return M

def ramp_mask_from_phi(M, phi_star, min_slope=-0.05, late_thr=0.80):
    """Conservative ramp mask: ends near F2 AND has sufficiently negative time-slope."""
    Q = M['cluster_quickstats']
    slope_t = np.asarray(Q['slope_time'], float)
    good = np.isfinite(phi_star) & (phi_star >= late_thr) & (slope_t <= min_slope)
    M['ramp_mask_phi'] = good
    print(f"[ramp mask] {int(good.sum())}/{good.size} ROIs kept (φ*>={late_thr}, slope_t≤{min_slope})")
    return M



import numpy as np

def summarize_overlap_cluster_rampmask(M, ramp_mask):
    lab = np.asarray(M['cluster']['labels'], int)
    K = int(lab.max()+1)
    rows = []
    for k in range(K):
        in_k = lab==k
        rows.append(dict(
            cluster=int(k),
            n_cluster=int(in_k.sum()),
            n_k_and_mask=int(np.sum(in_k & ramp_mask)),
            n_k_not_mask=int(np.sum(in_k & ~ramp_mask))
        ))
    rows.append(dict(
        cluster='not_k0',
        n_cluster=int((lab!=0).sum()),
        n_k_and_mask=int(np.sum((lab!=0) & ramp_mask)),
        n_k_not_mask=int(np.sum((lab!=0) & ~ramp_mask))
    ))
    print("[overlap cluster×ramp]", rows)
    M['ramp_overlap'] = rows
    return M


import numpy as np
import matplotlib.pyplot as plt

def _profile_matrix(M, roi_mask, mode='time', phase_bins=60, preF2_window=(0.15,0.70)):
    T  = np.asarray(M['time'], float)
    X  = np.asarray(M['roi_traces'], float)     # (trials, rois, Nt)
    isi= np.asarray(M['isi'], float)
    tr, R, Nt = X.shape
    idx_r = np.where(roi_mask)[0]
    if mode=='time':
        w = (T >= preF2_window[0]) & (T <= preF2_window[1])
        return np.nanmean(X[:, idx_r][:, :, w], axis=0), T[w]      # (Rk, nb), t
    else:
        # mean in phase per ROI across trials/ISIs
        phi_grid = np.linspace(0,1,int(phase_bins))
        Mat = np.full((idx_r.size, phi_grid.size), np.nan, float)
        for j, r in enumerate(idx_r):
            curves = []
            for tr_i in range(tr):
                y = X[tr_i, r]
                m = np.isfinite(y) & (T <= isi[tr_i]) & np.isfinite(isi[tr_i])
                if m.sum() < 3: continue
                phi = T[m]/isi[tr_i]; yy = y[m]
                yi = np.full_like(phi_grid, np.nan, float)
                seg=(phi_grid>=phi.min())&(phi_grid<=phi.max())
                yi[seg] = np.interp(phi_grid[seg], phi, yy)
                curves.append(yi)
            if curves: Mat[j] = np.nanmean(np.vstack(curves), axis=0)
        return Mat, phi_grid

def plot_rampmask_centroids(M, cfg, ramp_mask):
    pre = tuple(map(float, cfg.get('svclock', {}).get('preF2_window', (0.15,0.70))))
    TP_in, t  = _profile_matrix(M, ramp_mask,     'time',  preF2_window=pre)
    TP_out, _ = _profile_matrix(M, ~ramp_mask,    'time',  preF2_window=pre)
    PP_in, ph = _profile_matrix(M, ramp_mask,     'phase')
    PP_out, _ = _profile_matrix(M, ~ramp_mask,    'phase')

    fig, ax = plt.subplots(1,2, figsize=(8,3))
    for mat, a, ttl in [(TP_in, ax[0], 'ramp mask — time'), (TP_out, ax[0], 'non-mask — time')]:
        if mat.size:
            a.plot(t, np.nanmean(mat, axis=0), lw=2)
    ax[0].axvline(0, ls='--', lw=1); ax[0].set_xlabel('Time from F1-OFF (s)'); ax[0].set_ylabel('ΔF/F'); ax[0].set_title('centroids (time)')
    for mat, a, ttl in [(PP_in, ax[1], 'ramp mask — phase'), (PP_out, ax[1], 'non-mask — phase')]:
        if mat.size:
            a.plot(ph, np.nanmean(mat, axis=0), lw=2)
    ax[1].axvline(1, ls=':', lw=1); ax[1].set_xlabel('Phase (t/ISI)'); ax[1].set_title('centroids (phase)')
    fig.tight_layout()
    return fig



import numpy as np
import matplotlib.pyplot as plt

def ramp_dprime_timecourse(M, cfg, ramp_mask, preF2_window=(0.15,0.70), min_rois_per_trial=5):
    T  = np.asarray(M['time'], float)
    X  = np.asarray(M['roi_traces'], float)     # (trials, rois, Nt)
    yS = np.asarray(M['is_short'], bool)
    slopes_t = np.asarray(M['cluster_quickstats']['slope_time'], float)
    pre = tuple(map(float, cfg.get('svclock', {}).get('preF2_window', preF2_window)))
    w = (T >= pre[0]) & (T <= pre[1])

    idx_r = np.where(ramp_mask)[0]
    if idx_r.size == 0:
        raise ValueError("ramp_mask selects 0 ROIs.")
    # align ramp direction: decreasing (negative slope) → multiply by -1 so “more ramp” is up
    sign = np.where(slopes_t[idx_r] < 0, -1.0, +1.0)  # your mask already uses slope<=-0.05, so sign ≈ -1
    Z = X[:, idx_r][:, :, w] * sign[None, :, None]    # (trials, Rk, nb)

    # per-trial ramp score = mean across ramp ROIs (require enough finite)
    nb = Z.shape[-1]
    score = np.full((X.shape[0], nb), np.nan)
    for tr in range(X.shape[0]):
        vals = Z[tr]                                  # (Rk, nb)
        cnt  = np.isfinite(vals).sum(axis=0)
        mu   = np.nanmean(vals, axis=0)
        mu[cnt < min_rois_per_trial] = np.nan
        score[tr] = mu

    # Cohen’s d′(t) between short / long
    def _cohen_d(a, b):
        ma, mb = np.nanmean(a), np.nanmean(b)
        va, vb = np.nanvar(a), np.nanvar(b)
        sp = np.sqrt(0.5*(va+vb) + 1e-12)
        return (ma - mb)/sp

    dprime = np.full(nb, np.nan)
    for j in range(nb):
        s = score[yS, j]; l = score[~yS, j]
        if np.isfinite(s).sum() >= 10 and np.isfinite(l).sum() >= 10:
            dprime[j] = _cohen_d(s, l)

    # simple “divergence” time using |d′| threshold (no classifier)
    thr = float(cfg.get('analyses',{}).get('decode_time_resolved',{}).get('divergence_auc_threshold', 0.6))  # reuse knob
    # heuristically map AUC thr to d′: AUC≈Φ(d′/√2). For AUC=0.6 → d′≈0.358.
    from math import sqrt
    d_thr = 0.358 if thr==0.6 else 0.358  # keep fixed unless you want to map exactly
    sustain = int(cfg.get('decode',{}).get('sustain_bins', 3))
    div_bin, run = -1, 0
    for i, v in enumerate(np.abs(dprime)):
        run = run+1 if (np.isfinite(v) and v>=d_thr) else 0
        if run >= sustain: div_bin = i - sustain + 1; break
    div_time = float(T[w][div_bin]) if div_bin >= 0 else np.nan

    # plot
    fig, ax = plt.subplots(1,1, figsize=(6,3))
    ax.plot(T[w], dprime, lw=2)
    ax.axhline(0, ls='--', lw=1); ax.set_xlabel('Time from F1-OFF (s)'); ax.set_ylabel("d′ (short − long)")
    if np.isfinite(div_time): ax.axvline(div_time, ls=':', lw=1); ax.set_title(f'd′(t), divergence ≈ {div_time:.3f}s')
    else: ax.set_title("d′(t)")
    fig.tight_layout()

    # prints
    good = np.isfinite(dprime)
    if good.any():
        p10,p50,p90 = np.nanpercentile(np.abs(dprime[good]), [10,50,90])
        print(f"[ramp d′] |d′| p10/p50/p90: {p10:.3f}/{p50:.3f}/{p90:.3f}  |  divergence t≈{div_time if np.isfinite(div_time) else 'nan'}")
    else:
        print("[ramp d′] no valid bins.")
    return dict(time=T[w], dprime=dprime, divergence_time=div_time), fig




import numpy as np

def ramp_dprime_timecourse_balanced(M, cfg, ramp_mask, min_rois_per_trial=5,
                                    preF2_window=None, min_trials_per_isi=8):
    T   = np.asarray(M['time'], float)
    X   = np.asarray(M['roi_traces'], float)     # (trials, rois, Nt)
    yS  = np.asarray(M['is_short'], bool)
    isi = np.asarray(M['isi'], float)
    short_lv = np.asarray(M['short_levels'], float)
    long_lv  = np.asarray(M['long_levels'],  float)

    pre = preF2_window or tuple(map(float, cfg.get('svclock', {}).get('preF2_window', (0.15, 0.70))))
    w = (T >= pre[0]) & (T <= pre[1]); tw = T[w]; nb = int(w.sum())

    idx_r = np.where(ramp_mask)[0]
    if idx_r.size == 0: raise ValueError("ramp_mask selects 0 ROIs.")

    # flip negative-slope rampers so “more ramp” is up
    slopes_t = np.asarray(M['cluster_quickstats']['slope_time'], float)[idx_r]
    sign = np.where(slopes_t < 0, -1.0, +1.0)
    Z = X[:, idx_r][:, :, w] * sign[None, :, None]      # (trials, Rk, nb)

    # per-trial ramp score (mean across ramp ROIs; require enough finite per bin)
    score = np.full((X.shape[0], nb), np.nan, float)
    for tr in range(X.shape[0]):
        vals = Z[tr]                     # (Rk, nb)
        cnt  = np.isfinite(vals).sum(axis=0)
        mu   = np.nanmean(vals, axis=0)
        mu[cnt < min_rois_per_trial] = np.nan
        score[tr] = mu

    rng = np.random.default_rng(int(cfg.get('qa', {}).get('random_state', 0)))

    def _by_levels(mask_class, levels):
        # indices of trials in this class grouped by exact ISI level
        idxs = []
        for lvl in levels:
            ii = np.where(mask_class & (np.abs(isi - lvl) <= float(cfg.get('labels',{}).get('tolerance_sec',1e-3))))[0]
            idxs.append(ii)
        return idxs

    dprime = np.full(nb, np.nan)
    perbin_counts = []

    for j in range(nb):
        fin = np.isfinite(score[:, j])
        S_idx = _by_levels(fin &  yS, short_lv)
        L_idx = _by_levels(fin & ~yS, long_lv)
        # minimum per-ISI count we can take from both classes
        k_per_isi = []
        for a, b in zip(S_idx, L_idx):
            k_per_isi.append(min(len(a), len(b)))
        k = min([c for c in k_per_isi if c is not None] + [0])
        if k < min_trials_per_isi:
            perbin_counts.append((0,0)); continue

        take_S = np.concatenate([rng.choice(a, k, replace=False) for a in S_idx if len(a)>=k])
        take_L = np.concatenate([rng.choice(b, k, replace=False) for b in L_idx if len(b)>=k])

        s = score[take_S, j]; l = score[take_L, j]
        ma, mb = np.nanmean(s), np.nanmean(l)
        va, vb = np.nanvar(s), np.nanvar(l)
        sp = np.sqrt(0.5*(va + vb) + 1e-12)
        dprime[j] = (ma - mb) / sp
        perbin_counts.append((len(take_S), len(take_L)))

    # divergence in d′ units (map from AUC if desired)
    auc_thr = float(cfg.get('decode',{}).get('auc_threshold', 0.60))
    # AUC ≈ Φ(d′/√2) ⇒ d′ ≈ √2 Φ^{-1}(AUC); for 0.60 → ≈0.358
    d_thr = 0.358 if abs(auc_thr-0.60)<1e-9 else 0.358
    sustain = int(cfg.get('decode',{}).get('sustain_bins', 3))
    div_bin = -1; run = 0
    for i, v in enumerate(np.abs(dprime)):
        run = run+1 if (np.isfinite(v) and v>=d_thr) else 0
        if run >= sustain: div_bin = i - sustain + 1; break
    div_time = float(tw[div_bin]) if div_bin >= 0 else np.nan

    # prints
    good = np.isfinite(dprime)
    if good.any():
        p10,p50,p90 = np.nanpercentile(np.abs(dprime[good]), [10,50,90])
        print(f"[ramp d′ balanced] |d′| p10/p50/p90: {p10:.3f}/{p50:.3f}/{p90:.3f}  |  divergence t≈{div_time if np.isfinite(div_time) else 'nan'}")
    return dict(time=tw, dprime=dprime, divergence_time=div_time, perbin_counts=perbin_counts)


def ramp_dprime_phase(M, cfg, ramp_mask, phase_bins=60, min_rois_per_trial=5, min_trials_per_isi=8):
    T   = np.asarray(M['time'], float)
    X   = np.asarray(M['roi_traces'], float)
    yS  = np.asarray(M['is_short'], bool)
    isi = np.asarray(M['isi'], float)
    short_lv = np.asarray(M['short_levels'], float)
    long_lv  = np.asarray(M['long_levels'],  float)

    idx_r = np.where(ramp_mask)[0]
    slopes_t = np.asarray(M['cluster_quickstats']['slope_time'], float)[idx_r]
    sign = np.where(slopes_t < 0, -1.0, +1.0)

    phi_grid = np.linspace(0, 1, int(phase_bins))
    # per-trial ramp score in phase
    S = np.full((X.shape[0], phi_grid.size), np.nan, float)
    for tr in range(X.shape[0]):
        curves = []
        for j, r in enumerate(idx_r):
            y = X[tr, r] * sign[j]
            m = np.isfinite(y) & np.isfinite(isi[tr]) & (T <= isi[tr])
            if m.sum() < 3: continue
            phi = T[m] / isi[tr]; yy = y[m]
            yi = np.full_like(phi_grid, np.nan, float)
            seg = (phi_grid >= phi.min()) & (phi_grid <= phi.max())
            yi[seg] = np.interp(phi_grid[seg], phi, yy)
            curves.append(yi)
        if len(curves) >= min_rois_per_trial:
            Y = np.vstack(curves)
            # require enough finite ROIs per bin
            cnt = np.isfinite(Y).sum(axis=0)
            mu  = np.nanmean(Y, axis=0)
            mu[cnt < min_rois_per_trial] = np.nan
            S[tr] = mu

    rng = np.random.default_rng(int(cfg.get('qa', {}).get('random_state', 0)))
    tol = float(cfg.get('labels',{}).get('tolerance_sec', 1e-3))

    def _idx_levels(mask_class, levels):
        groups = []
        for lvl in levels:
            ii = np.where(mask_class & (np.abs(isi - lvl) <= tol))[0]
            groups.append(ii)
        return groups

    dprime = np.full(phi_grid.size, np.nan)
    for j in range(phi_grid.size):
        fin = np.isfinite(S[:, j])
        S_groups = _idx_levels(fin &  yS, short_lv)
        L_groups = _idx_levels(fin & ~yS, long_lv)
        k = min([min(len(a), len(b)) for a, b in zip(S_groups, L_groups)] + [0])
        if k < min_trials_per_isi: continue
        take_S = np.concatenate([rng.choice(a, k, replace=False) for a in S_groups if len(a)>=k])
        take_L = np.concatenate([rng.choice(b, k, replace=False) for b in L_groups if len(b)>=k])

        s = S[take_S, j]; l = S[take_L, j]
        ma, mb = np.nanmean(s), np.nanmean(l)
        va, vb = np.nanvar(s), np.nanvar(l)
        sp = np.sqrt(0.5*(va + vb) + 1e-12)
        dprime[j] = (ma - mb) / sp

    # divergence phase
    auc_thr = float(cfg.get('decode',{}).get('auc_threshold', 0.60))
    d_thr = 0.358 if abs(auc_thr-0.60)<1e-9 else 0.358
    sustain = int(cfg.get('decode',{}).get('sustain_bins', 3))
    div_bin = -1; run = 0
    for i, v in enumerate(np.abs(dprime)):
        run = run+1 if (np.isfinite(v) and v>=d_thr) else 0
        if run >= sustain: div_bin = i - sustain + 1; break
    div_phase = float(phi_grid[div_bin]) if div_bin >= 0 else np.nan

    print(f"[ramp d′ phase] divergence φ≈{div_phase if np.isfinite(div_phase) else 'nan'}")
    return dict(phase=phi_grid, dprime=dprime, divergence_phase=div_phase)


import numpy as np
import matplotlib.pyplot as plt

def _nan_gauss_smooth(y, sigma_bins):
    if sigma_bins is None or sigma_bins <= 0: return y
    y = np.asarray(y, float)
    if y.size == 0: return y
    rad = int(np.ceil(4.0 * float(sigma_bins)))
    if rad < 1: return y
    x = np.arange(-rad, rad+1, dtype=float)
    k = np.exp(-0.5*(x/float(sigma_bins))**2); k /= np.nansum(k)
    m = np.isfinite(y).astype(float)
    y0 = np.where(np.isfinite(y), y, 0.0)
    num = np.convolve(y0, k, mode='same')
    den = np.convolve(m,  k, mode='same')
    out = num / np.maximum(den, 1e-12)
    out[den < 1e-12] = np.nan
    return out

def _earliest_onset_time(trace, t, baseline_win, polarity='neg',
                         thresh_mode='z', z=1.0, abs_amp=0.05,
                         min_bins=3, smooth_bins=1.0, search_win=None):
    """
    Earliest sustained deviation from baseline.
    polarity: 'neg' (default for rampers) or 'pos'
    thresh_mode: 'z' (baseline mean ± z*sd) or 'abs' (baseline mean ± abs_amp)
    """
    y = np.asarray(trace, float)
    t = np.asarray(t, float)
    if y.size < 5 or t.size != y.size: return np.inf

    y = _nan_gauss_smooth(y, smooth_bins)

    # baseline stats
    bmask = (t >= baseline_win[0]) & (t <= baseline_win[1]) & np.isfinite(y)
    if bmask.sum() < 5: return np.inf
    mu = float(np.nanmean(y[bmask]))
    sd = float(np.nanstd(y[bmask]))

    if thresh_mode == 'z':
        thr = mu - z*sd if polarity == 'neg' else mu + z*sd
    else:
        thr = mu - abs_amp  if polarity == 'neg' else mu + abs_amp

    # search region (e.g., preF2 window)
    smask = np.isfinite(y)
    if search_win is not None:
        smask &= (t >= search_win[0]) & (t <= search_win[1])

    if not smask.any(): return np.inf
    yy = y.copy()

    if polarity == 'neg':
        cond = (yy <= thr) & smask
    else:
        cond = (yy >= thr) & smask

    # look for first run of >= min_bins consecutive True
    run = 0
    for i, ok in enumerate(cond):
        run = run + 1 if ok else 0
        if run >= int(min_bins):
            return float(t[i - min_bins + 1])
    return np.inf

def _mean_trace_matrix(M, preF2_window):
    """Per-ROI mean across trials on the F1-OFF grid, restricted to preF2_window."""
    T = np.asarray(M['time'], float)
    X = np.asarray(M['roi_traces'], float)  # (trials, rois, Nt)
    w = (T >= preF2_window[0]) & (T <= preF2_window[1])
    Tw = T[w]
    # mean across trials (NaN-aware)
    Mat = np.nanmean(X[:, :, w], axis=0)   # (rois, nbins)
    return Mat, Tw

def plot_rasters_by_rampmask_sorted(M, cfg, ramp_mask, *,
                                    polarity_ramp='neg', polarity_non='neg',
                                    onset={'thresh_mode':'z','z':1.0,'abs_amp':0.05,
                                           'min_bins':3,'smooth_bins':2.0},
                                    vmin_vmax_mode='percentile'):
    """
    Two rasters (ramp vs non-ramp), rows = ROIs sorted by earliest onset time.
    Uses F1-OFF grid already stored in M (strictly pre-F2).
    """
    assert 'roi_traces' in M and 'time' in M, "Build F1-OFF grid first."

    pre = tuple(map(float, cfg.get('svclock', {}).get('preF2_window', (0.15, 0.70))))
    # baseline: use the pre-seconds you built into the grid if available
    pre_sec = float(cfg.get('grids', {}).get('f1off', {}).get('pre_seconds', 0.30))
    baseline_win = (-pre_sec, 0.0)

    # per-ROI mean traces
    MAT, Tw = _mean_trace_matrix(M, pre)

    # onset per ROI in each group
    ramp_mask = np.asarray(ramp_mask, bool)
    R = MAT.shape[0]
    on_ramp    = np.full(R, np.inf, float)
    on_nonramp = np.full(R, np.inf, float)

    # search window = preF2_window
    for r in np.where(ramp_mask)[0]:
        on_ramp[r] = _earliest_onset_time(
            MAT[r], Tw, baseline_win, polarity=polarity_ramp,
            thresh_mode=onset.get('thresh_mode','z'),
            z=float(onset.get('z',1.0)),
            abs_amp=float(onset.get('abs_amp',0.05)),
            min_bins=int(onset.get('min_bins',3)),
            smooth_bins=float(onset.get('smooth_bins',2.0)),
            search_win=pre
        )
    for r in np.where(~ramp_mask)[0]:
        on_nonramp[r] = _earliest_onset_time(
            MAT[r], Tw, baseline_win, polarity=polarity_non,
            thresh_mode=onset.get('thresh_mode','z'),
            z=float(onset.get('z',1.0)),
            abs_amp=float(onset.get('abs_amp',0.05)),
            min_bins=int(onset.get('min_bins',3)),
            smooth_bins=float(onset.get('smooth_bins',2.0)),
            search_win=pre
        )

    # sort indices (finite first, then inf), tiebreak by slope_time if available
    def _sort_idx(mask, onset_vec, tiebreak=None):
        idx = np.where(mask)[0]
        if idx.size == 0: return idx
        o  = onset_vec[idx]
        finite = np.isfinite(o)
        if tiebreak is None:
            key = np.lexsort((idx, o))      # onset primary
            return idx[key]
        else:
            tb = np.asarray(tiebreak, float)[idx]
            # sort by: (isfinite onset asc), (onset asc), (tiebreak asc)
            key = np.lexsort((tb, o, ~finite))
            return idx[key]

    tbreak = None
    if 'cluster_quickstats' in M and 'slope_time' in M['cluster_quickstats']:
        tbreak = M['cluster_quickstats']['slope_time']

    ord_ramp    = _sort_idx(ramp_mask, on_ramp,    tiebreak=tbreak)
    ord_nonramp = _sort_idx(~ramp_mask, on_nonramp, tiebreak=tbreak)

    # matrices in sorted order
    A = MAT[ord_ramp]
    B = MAT[ord_nonramp]

    # color limits
    pcts = cfg.get('plots', {}).get('raster', {}).get('vmin_vmax_percentiles', [5, 95])
    def _lims(MAT_):
        if MAT_.size == 0: return (-0.1, 0.1)
        v = MAT_[np.isfinite(MAT_)]
        if v.size < 10: return (np.nanmin(v), np.nanmax(v))
        lo, hi = np.nanpercentile(v, pcts)
        if vmin_vmax_mode == 'symmetric':
            m = max(abs(lo), abs(hi)); return (-m, m)
        return (lo, hi)

    vA = _lims(A); vB = _lims(B)

    # figure sizing
    h_ramp    = max(1.8, 0.018 * max(10, A.shape[0]))
    h_nonramp = max(1.8, 0.018 * max(10, B.shape[0]))
    fig, axes = plt.subplots(2, 1, figsize=(8, h_ramp + h_nonramp + 1.0), sharex=True)

    def _imshow(ax, MAT_, vlims, title):
        if MAT_.size == 0:
            ax.set_title(title + " (n=0)")
            ax.axis('off'); return
        im = ax.imshow(MAT_, aspect='auto', origin='lower',
                       extent=[Tw[0], Tw[-1], 0, MAT_.shape[0]],
                       vmin=vlims[0], vmax=vlims[1], cmap='viridis', interpolation='nearest')
        ax.axvline(0.0, color='w', lw=0.9, ls='--', alpha=0.8)
        ax.axvline(pre[1], color='w', lw=0.6, ls=':', alpha=0.6)
        ax.set_ylabel('ROIs')
        ax.set_title(f"{title} (n={MAT_.shape[0]})")
        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('ΔF/F')

    _imshow(axes[0], A, vA, "ramp mask — sorted by earliest onset")
    _imshow(axes[1], B, vB, "non-ramp — sorted by earliest onset")
    axes[1].set_xlabel('Time from F1-OFF (s)')
    fig.tight_layout()

    # save bookkeeping
    M.setdefault('rasters', {})
    M['rasters']['rampmask_sorted'] = dict(
        time=Tw, ord_ramp=ord_ramp, ord_nonramp=ord_nonramp,
        onset_ramp=on_ramp[ord_ramp], onset_nonramp=on_nonramp[ord_nonramp],
        params=dict(preF2_window=pre, baseline_window=baseline_win,
                    onset=onset, vmin_vmax_percentiles=pcts)
    )
    return fig




















def filter_trials_for_inspection(trial_data, cfg):
    """
    Filter trial_data based on cfg['group_inspect'] criteria.
    
    Returns filtered copy of trial_data with only selected trial rows.
    """
    print("=== Starting filter_trials_for_inspection ===")
    
    gi = cfg.get('group_inspect', {})
    df = trial_data['df_trials_with_segments'].copy()
    n_trials_orig = len(df)
    
    print(f"Original trials: {n_trials_orig}")
    print(f"Alignment point: {trial_data.get('alignment_point', 'unknown')}")
    
    # Convert ISI values to seconds for comparison
    isi_vec = _to_seconds(df['isi'].to_numpy())
    
    # Trial type filter
    trial_type = str(gi.get('trial_type', 'all')).lower()
    if trial_type == 'rewarded':
        mask = df['rewarded'].astype(bool)
        print(f"Filtering to rewarded trials: {mask.sum()}/{n_trials_orig}")
    elif trial_type == 'punished':
        mask = df['punished'].astype(bool)
        print(f"Filtering to punished trials: {mask.sum()}/{n_trials_orig}")
    elif trial_type == 'correct':
        is_right_trial = df['is_right'].astype(bool)
        is_right_choice = df['is_right_choice'].astype(bool)
        did_not_choose = df['did_not_choose'].astype(bool)
        mask = (is_right_trial == is_right_choice) & (~did_not_choose)
        print(f"Filtering to correct trials: {mask.sum()}/{n_trials_orig}")
    elif trial_type == 'incorrect':
        is_right_trial = df['is_right'].astype(bool)
        is_right_choice = df['is_right_choice'].astype(bool)
        did_not_choose = df['did_not_choose'].astype(bool)
        correct = (is_right_trial == is_right_choice) & (~did_not_choose)
        mask = (~correct) & (~did_not_choose)
        print(f"Filtering to incorrect trials: {mask.sum()}/{n_trials_orig}")
    elif trial_type == 'did_not_choose':
        mask = df['did_not_choose'].astype(bool)
        print(f"Filtering to did_not_choose trials: {mask.sum()}/{n_trials_orig}")
    else:  # 'all'
        mask = np.ones(n_trials_orig, dtype=bool)
        print(f"Keeping all trials: {n_trials_orig}")
    
    # ISI constraint filter
    isi_constraint = gi.get('isi_constraint', 'all')
    if isinstance(isi_constraint, str):
        if isi_constraint.lower() == 'short':
            # Use session short_isis levels
            short_levels = _to_seconds(trial_data['session_info']['short_isis'])
            tol = float(cfg.get('labels', {}).get('tolerance_sec', 1e-3))
            isi_mask = _member_mask(isi_vec, short_levels, tol)
            print(f"ISI filter (short): {isi_mask.sum()}/{n_trials_orig} trials")
        elif isi_constraint.lower() == 'long':
            # Use session long_isis levels
            long_levels = _to_seconds(trial_data['session_info']['long_isis'])
            tol = float(cfg.get('labels', {}).get('tolerance_sec', 1e-3))
            isi_mask = _member_mask(isi_vec, long_levels, tol)
            print(f"ISI filter (long): {isi_mask.sum()}/{n_trials_orig} trials")
        else:  # 'all'
            isi_mask = np.ones(n_trials_orig, dtype=bool)
            print(f"ISI filter (all): {n_trials_orig}")
    elif isinstance(isi_constraint, (list, tuple, np.ndarray)):
        # Specific ISI values provided
        target_isis = _to_seconds(isi_constraint)
        tol = float(cfg.get('labels', {}).get('tolerance_sec', 1e-3))
        isi_mask = _member_mask(isi_vec, target_isis, tol)
        print(f"ISI filter (specific {target_isis}): {isi_mask.sum()}/{n_trials_orig} trials")
    else:
        isi_mask = np.ones(n_trials_orig, dtype=bool)
        print(f"ISI filter (default all): {n_trials_orig}")
    
    # Combine filters
    final_mask = mask & isi_mask
    df_filtered = df[final_mask].copy()
    n_trials_final = len(df_filtered)
    
    print(f"Final filtered trials: {n_trials_final}/{n_trials_orig}")
    
    # Create filtered trial_data copy
    trial_data_filtered = trial_data.copy()
    trial_data_filtered['df_trials_with_segments'] = df_filtered
    
    # Print summary of filtered data
    if n_trials_final > 0:
        print("Filtered trial summary:")
        if 'rewarded' in df_filtered.columns:
            print(f"  Rewarded: {df_filtered['rewarded'].sum()}")
        if 'punished' in df_filtered.columns:
            print(f"  Punished: {df_filtered['punished'].sum()}")
        if 'is_right' in df_filtered.columns:
            print(f"  Right trials: {df_filtered['is_right'].sum()}")
        isi_filtered = _to_seconds(df_filtered['isi'].to_numpy())
        unique_isis, counts = np.unique(np.round(isi_filtered, 3), return_counts=True)
        print(f"  ISI distribution: {dict(zip(unique_isis, counts))}")
    
    print("=== Completed filter_trials_for_inspection ===\n")
    return trial_data_filtered


def _sort_rois_by_method(traces, time_vec, method='max_peak_time', tiebreak_values=None, baseline_win=(-0.3, 0.0)):
    """
    Sort ROI indices based on specified method.
    
    Parameters:
    - traces: (n_rois, n_timepoints) array of mean traces
    - time_vec: (n_timepoints,) time vector
    - method: sorting method string
    - tiebreak_values: optional array for tiebreaking (e.g., slopes)
    - baseline_win: tuple for baseline window (for onset methods)
    
    Returns:
    - sort_indices: array of ROI indices in sort order
    - sort_values: array of values used for sorting
    """
    n_rois, n_timepoints = traces.shape
    sort_values = np.full(n_rois, np.inf)
    
    print(f"Sorting {n_rois} ROIs by method: {method}")
    
    # Helper for baseline stats
    def _baseline_stats(trace):
        bmask = (time_vec >= baseline_win[0]) & (time_vec <= baseline_win[1]) & np.isfinite(trace)
        if bmask.sum() < 5:
            return np.nan, np.nan
        mu = np.nanmean(trace[bmask])
        sd = np.nanstd(trace[bmask])
        return mu, sd
    
    if method == 'max_peak_time':
        # Time of maximum value
        for i, trace in enumerate(traces):
            if np.isfinite(trace).any():
                sort_values[i] = time_vec[np.nanargmax(trace)]
        print(f"Max peak times range: {np.nanmin(sort_values):.3f} to {np.nanmax(sort_values):.3f}s")
                
    elif method == 'min_peak_time':
        # Time of minimum value (good for ramps)
        for i, trace in enumerate(traces):
            if np.isfinite(trace).any():
                sort_values[i] = time_vec[np.nanargmin(trace)]
        print(f"Min peak times range: {np.nanmin(sort_values):.3f} to {np.nanmax(sort_values):.3f}s")
                
    elif method == 'abs_peak_time':
        # Time of largest absolute deviation from baseline
        for i, trace in enumerate(traces):
            mu, _ = _baseline_stats(trace)
            if np.isfinite(mu) and np.isfinite(trace).any():
                abs_dev = np.abs(trace - mu)
                sort_values[i] = time_vec[np.nanargmax(abs_dev)]
        print(f"Abs peak times range: {np.nanmin(sort_values):.3f} to {np.nanmax(sort_values):.3f}s")
                
    elif method == 'first_threshold_cross':
        # First crossing of 1 SD threshold (more permissive than sustained)
        for i, trace in enumerate(traces):
            mu, sd = _baseline_stats(trace)
            if np.isfinite(mu) and np.isfinite(sd) and sd > 1e-9:
                thr = mu - 1.0 * sd  # Look for negative deflections
                below = (trace <= thr) & np.isfinite(trace) & (time_vec >= 0)
                if below.any():
                    sort_values[i] = time_vec[np.where(below)[0][0]]
        finite_count = np.isfinite(sort_values).sum()
        print(f"First threshold crossings: {finite_count}/{n_rois} ROIs, range: {np.nanmin(sort_values):.3f} to {np.nanmax(sort_values):.3f}s")
                    
    elif method == 'slope_onset':
        # Time when derivative first becomes strongly negative
        for i, trace in enumerate(traces):
            if np.isfinite(trace).sum() < 5:
                continue
            # Smooth derivative
            dt = np.median(np.diff(time_vec)) if len(time_vec) > 1 else 1.0
            deriv = np.gradient(trace) / dt
            # Look for first strong negative slope after t=0
            search_mask = (time_vec >= 0) & np.isfinite(deriv)
            if not search_mask.any():
                continue
            
            mu, sd = _baseline_stats(deriv)
            if np.isfinite(sd) and sd > 1e-9:
                thr = mu - 2.0 * sd  # Strong negative slope threshold
                strong_neg = (deriv <= thr) & search_mask
                if strong_neg.any():
                    sort_values[i] = time_vec[np.where(strong_neg)[0][0]]
        finite_count = np.isfinite(sort_values).sum()
        print(f"Slope onsets: {finite_count}/{n_rois} ROIs, range: {np.nanmin(sort_values):.3f} to {np.nanmax(sort_values):.3f}s")
                    
    elif method == 'centroid_time':
        # Center of mass of absolute response
        for i, trace in enumerate(traces):
            mu, _ = _baseline_stats(trace)
            if np.isfinite(mu) and np.isfinite(trace).any():
                weights = np.abs(trace - mu)
                weights = weights / (np.nansum(weights) + 1e-12)
                sort_values[i] = np.nansum(weights * time_vec)
        print(f"Centroid times range: {np.nanmin(sort_values):.3f} to {np.nanmax(sort_values):.3f}s")
                
    elif method == 'earliest_onset':
        # Sustained threshold crossing (original method)
        thresh_z = 1.0
        min_bins = 3
        for i, trace in enumerate(traces):
            mu, sd = _baseline_stats(trace)
            if np.isfinite(mu) and np.isfinite(sd) and sd > 1e-9:
                thr = mu - thresh_z * sd
                below = (trace <= thr) & np.isfinite(trace) & (time_vec >= 0)
                if not below.any():
                    continue
                
                # Find first run of min_bins consecutive points
                run = 0
                for j, flag in enumerate(below):
                    run = run + 1 if flag else 0
                    if run >= min_bins:
                        sort_values[i] = time_vec[j - min_bins + 1]
                        break
        finite_count = np.isfinite(sort_values).sum()
        print(f"Earliest onsets: {finite_count}/{n_rois} ROIs, range: {np.nanmin(sort_values):.3f} to {np.nanmax(sort_values):.3f}s")
                        
    elif method == 'ramp_endpoint':
        # For ramp ROIs, time when trace reaches most negative value in second half
        search_start = len(time_vec) // 2  # Second half only
        for i, trace in enumerate(traces):
            if np.isfinite(trace[search_start:]).any():
                min_idx = search_start + np.nanargmin(trace[search_start:])
                sort_values[i] = time_vec[min_idx]
        print(f"Ramp endpoints range: {np.nanmin(sort_values):.3f} to {np.nanmax(sort_values):.3f}s")
                
    elif method == 'transient_peak':
        # Time of sharpest peak/trough (max absolute second derivative)
        for i, trace in enumerate(traces):
            if np.isfinite(trace).sum() < 5:
                continue
            # Second derivative
            d2 = np.gradient(np.gradient(trace))
            search_mask = (time_vec >= 0) & np.isfinite(d2)
            if search_mask.any():
                max_d2_idx = np.nanargmax(np.abs(d2[search_mask]))
                sort_values[i] = time_vec[search_mask][max_d2_idx]
        print(f"Transient peaks range: {np.nanmin(sort_values):.3f} to {np.nanmax(sort_values):.3f}s")
                
    else:
        print(f"Unknown sorting method: {method}, using max_peak_time")
        for i, trace in enumerate(traces):
            if np.isfinite(trace).any():
                sort_values[i] = time_vec[np.nanargmax(trace)]
    
    # Sort: finite values first, then by sort value, then by tiebreak
    finite_mask = np.isfinite(sort_values)
    if tiebreak_values is not None:
        tiebreak = np.asarray(tiebreak_values)
        sort_key = np.lexsort((tiebreak, sort_values, ~finite_mask))
    else:
        sort_key = np.lexsort((sort_values, ~finite_mask))
    
    finite_count = finite_mask.sum()
    print(f"Sorting result: {finite_count}/{n_rois} ROIs with finite sort values")
    
    return sort_key, sort_values




# def plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg, 
#                                         use_stored_order=False, store_order=True):
#     """
#     Plot session-averaged rasters for ramp vs non-ramp ROIs using original dF/F traces.
#     ROIs within each group are sorted by configurable methods.
    
#     Parameters:
#     - use_stored_order: if True, use previously stored sorting order from M['raster_analysis']['canonical_order']
#     - store_order: if True, save the computed sorting order to M for future use
    
#     Uses the native dff_time_vector timeline from trial_data (already aligned to alignment_point).
#     """
#     print("=== Starting plot_ramp_vs_nonramp_session_rasters ===")
    
#     if 'ramp_mask_phi' not in M:
#         raise KeyError("M['ramp_mask_phi'] not found. Run ramp mask analysis first.")
    
#     ramp_mask = np.asarray(M['ramp_mask_phi'], bool)
#     df = trial_data_filtered['df_trials_with_segments']
#     n_trials = len(df)
#     n_rois = int(M['n_rois'])
    
#     print(f"Processing {n_trials} filtered trials")
#     print(f"Ramp mask: {ramp_mask.sum()}/{len(ramp_mask)} ROIs")
#     print(f"Alignment point: {trial_data_filtered.get('alignment_point', 'unknown')}")
    
#     # Get sorting methods from config
#     gi = cfg.get('group_inspect', {})
#     ramp_sort_method = gi.get('ramp_sort', 'min_peak_time')
#     nonramp_sort_method = gi.get('non_ramp_sort', 'abs_peak_time')
    
#     print(f"Sorting methods: ramp={ramp_sort_method}, non-ramp={nonramp_sort_method}")
#     print(f"Use stored order: {use_stored_order}, Store order: {store_order}")
    
#     # Check if we should use stored order
#     stored_order_available = False
#     if use_stored_order and 'raster_analysis' in M and 'canonical_order' in M['raster_analysis']:
#         stored = M['raster_analysis']['canonical_order']
#         if ('ramp_roi_sorted' in stored and 'nonramp_roi_sorted' in stored and
#             'ramp_sort_method' in stored and 'nonramp_sort_method' in stored):
#             stored_order_available = True
#             print(f"Found stored order: ramp method={stored['ramp_sort_method']}, "
#                   f"non-ramp method={stored['nonramp_sort_method']}")
    
#     # Find common time axis across all trials
#     all_time_vecs = []
#     for idx, (_, row) in enumerate(df.iterrows()):
#         t_vec = np.asarray(row['dff_time_vector'], dtype=float)
#         all_time_vecs.append(t_vec)
#         if idx < 3:  # Print first few for verification
#             print(f"Trial {idx} time range: {t_vec.min():.3f} to {t_vec.max():.3f}s")
    
#     # Create common time grid spanning all trials
#     t_min = min(t.min() for t in all_time_vecs)
#     t_max = max(t.max() for t in all_time_vecs)
    
#     # Use finest resolution from any trial, but cap at reasonable limit
#     dt_all = [np.median(np.diff(t)) for t in all_time_vecs if len(t) > 1]
#     dt_common = min(dt_all) if dt_all else 0.01
#     dt_common = max(dt_common, 0.005)  # Don't go below 5ms
    
#     n_bins = int(np.ceil((t_max - t_min) / dt_common))
#     n_bins = min(n_bins, 5000)  # Cap total bins for memory
    
#     T_common = np.linspace(t_min, t_max, n_bins)
#     print(f"Common time grid: {T_common[0]:.3f} to {T_common[-1]:.3f}s, {n_bins} bins, dt={dt_common:.4f}s")
    
#     # Initialize arrays for session averages
#     ramp_traces = []  # List of (n_rois_ramp, n_bins) arrays per trial
#     nonramp_traces = []  # List of (n_rois_nonramp, n_bins) arrays per trial
    
#     ramp_roi_idx = np.where(ramp_mask)[0]
#     nonramp_roi_idx = np.where(~ramp_mask)[0]
    
#     print(f"Ramp ROIs: {len(ramp_roi_idx)}, Non-ramp ROIs: {len(nonramp_roi_idx)}")
    
#     # Process each trial
#     trials_used = 0
#     for trial_idx, (_, row) in enumerate(df.iterrows()):
#         if trial_idx % 20 == 0:
#             print(f"Processing trial {trial_idx}/{n_trials}")
        
#         t_vec = np.asarray(row['dff_time_vector'], dtype=float)
#         dff = np.asarray(row['dff_segment'], dtype=float)  # (n_rois, n_samples)
        
#         if dff.shape[0] != n_rois:
#             print(f"Trial {trial_idx}: dF/F shape mismatch, skipping")
#             continue
        
#         # Ensure monotonic time
#         if not np.all(np.diff(t_vec) > 0):
#             order = np.argsort(t_vec)
#             t_vec = t_vec[order]
#             dff = dff[:, order]
        
#         # Interpolate each ROI to common grid
#         dff_interp = np.full((n_rois, n_bins), np.nan, dtype=float)
#         for r in range(n_rois):
#             # Only interpolate within the native support
#             valid = np.isfinite(dff[r])
#             if valid.sum() < 3:
#                 continue
            
#             y_vals = dff[r][valid]
#             t_vals = t_vec[valid]
            
#             # Interpolate only within native range
#             mask_interp = (T_common >= t_vals.min()) & (T_common <= t_vals.max())
#             if mask_interp.sum() > 0:
#                 dff_interp[r, mask_interp] = np.interp(
#                     T_common[mask_interp], t_vals, y_vals
#                 )
        
#         # Separate into ramp vs non-ramp groups
#         if len(ramp_roi_idx) > 0:
#             ramp_traces.append(dff_interp[ramp_roi_idx])
#         if len(nonramp_roi_idx) > 0:
#             nonramp_traces.append(dff_interp[nonramp_roi_idx])
        
#         trials_used += 1
    
#     print(f"Successfully processed {trials_used}/{n_trials} trials")
    
#     # Compute session averages
#     if ramp_traces:
#         ramp_stack = np.stack(ramp_traces, axis=0)  # (trials, rois, bins)
#         ramp_mean = np.nanmean(ramp_stack, axis=0)  # (rois, bins)
#         print(f"Ramp group session average shape: {ramp_mean.shape}")
#     else:
#         ramp_mean = np.empty((0, n_bins))
    
#     if nonramp_traces:
#         nonramp_stack = np.stack(nonramp_traces, axis=0)  # (trials, rois, bins)
#         nonramp_mean = np.nanmean(nonramp_stack, axis=0)  # (rois, bins)
#         print(f"Non-ramp group session average shape: {nonramp_mean.shape}")
#     else:
#         nonramp_mean = np.empty((0, n_bins))
    
#     # Determine sorting order
#     if stored_order_available and use_stored_order:
#         # Use stored canonical order
#         stored = M['raster_analysis']['canonical_order']
        
#         # Map stored indices back to current group indices
#         stored_ramp_global = stored['ramp_roi_sorted']
#         stored_nonramp_global = stored['nonramp_roi_sorted']
        
#         # Find intersection with current groups (in case ramp mask changed)
#         current_ramp_set = set(ramp_roi_idx)
#         current_nonramp_set = set(nonramp_roi_idx)
        
#         # Keep stored order but only for ROIs still in the current groups
#         ramp_roi_sorted = [idx for idx in stored_ramp_global if idx in current_ramp_set]
#         nonramp_roi_sorted = [idx for idx in stored_nonramp_global if idx in current_nonramp_set]
        
#         # Convert to arrays and get local indices
#         ramp_roi_sorted = np.array(ramp_roi_sorted)
#         nonramp_roi_sorted = np.array(nonramp_roi_sorted)
        
#         # Get local indices within the group for matrix indexing
#         if len(ramp_roi_sorted) > 0:
#             # Map global ROI indices to local group indices
#             ramp_local_idx = np.array([np.where(ramp_roi_idx == global_idx)[0][0] 
#                                      for global_idx in ramp_roi_sorted])
#             ramp_mean_sorted = ramp_mean[ramp_local_idx]
#         else:
#             ramp_mean_sorted = ramp_mean
            
#         if len(nonramp_roi_sorted) > 0:
#             nonramp_local_idx = np.array([np.where(nonramp_roi_idx == global_idx)[0][0] 
#                                         for global_idx in nonramp_roi_sorted])
#             nonramp_mean_sorted = nonramp_mean[nonramp_local_idx]
#         else:
#             nonramp_mean_sorted = nonramp_mean
        
#         print(f"Using stored order: ramp ROIs={len(ramp_roi_sorted)}, non-ramp ROIs={len(nonramp_roi_sorted)}")
        
#         # Update the used methods to reflect what was stored
#         ramp_sort_method_used = stored['ramp_sort_method']
#         nonramp_sort_method_used = stored['nonramp_sort_method']
        
#     else:
#         # Compute new sorting order
#         print("Computing new sorting order...")
        
#         # Get tiebreak values (slopes) if available
#         tiebreak_values = None
#         if 'cluster_quickstats' in M and 'slope_time' in M['cluster_quickstats']:
#             tiebreak_values = M['cluster_quickstats']['slope_time']
        
#         # Sort ramp group
#         if ramp_mean.shape[0] > 0:
#             sort_key_ramp, sort_vals_ramp = _sort_rois_by_method(
#                 ramp_mean, T_common, method=ramp_sort_method,
#                 tiebreak_values=tiebreak_values[ramp_roi_idx] if tiebreak_values is not None else None
#             )
#             ramp_mean_sorted = ramp_mean[sort_key_ramp]
#             ramp_roi_sorted = ramp_roi_idx[sort_key_ramp]
            
#             finite_count = np.isfinite(sort_vals_ramp).sum()
#             print(f"Ramp group: {finite_count}/{len(sort_vals_ramp)} ROIs with detectable {ramp_sort_method}")
#             if finite_count > 0:
#                 finite_vals = sort_vals_ramp[np.isfinite(sort_vals_ramp)]
#                 print(f"Ramp {ramp_sort_method} range: {finite_vals.min():.3f} to {finite_vals.max():.3f}s")
#         else:
#             ramp_mean_sorted = ramp_mean
#             ramp_roi_sorted = ramp_roi_idx
        
#         # Sort non-ramp group
#         if nonramp_mean.shape[0] > 0:
#             sort_key_nonramp, sort_vals_nonramp = _sort_rois_by_method(
#                 nonramp_mean, T_common, method=nonramp_sort_method,
#                 tiebreak_values=tiebreak_values[nonramp_roi_idx] if tiebreak_values is not None else None
#             )
#             nonramp_mean_sorted = nonramp_mean[sort_key_nonramp]
#             nonramp_roi_sorted = nonramp_roi_idx[sort_key_nonramp]
            
#             finite_count = np.isfinite(sort_vals_nonramp).sum()
#             print(f"Non-ramp group: {finite_count}/{len(sort_vals_nonramp)} ROIs with detectable {nonramp_sort_method}")
#             if finite_count > 0:
#                 finite_vals = sort_vals_nonramp[np.isfinite(sort_vals_nonramp)]
#                 print(f"Non-ramp {nonramp_sort_method} range: {finite_vals.min():.3f} to {finite_vals.max():.3f}s")
#         else:
#             nonramp_mean_sorted = nonramp_mean
#             nonramp_roi_sorted = nonramp_roi_idx
        
#         ramp_sort_method_used = ramp_sort_method
#         nonramp_sort_method_used = nonramp_sort_method
        
#         # Store canonical order if requested
#         if store_order:
#             M.setdefault('raster_analysis', {})
#             M['raster_analysis']['canonical_order'] = {
#                 'ramp_roi_sorted': ramp_roi_sorted.astype(int),
#                 'nonramp_roi_sorted': nonramp_roi_sorted.astype(int),
#                 'ramp_sort_method': ramp_sort_method_used,
#                 'nonramp_sort_method': nonramp_sort_method_used,
#                 'computed_on': {
#                     'trial_type': gi.get('trial_type', 'all'),
#                     'isi_constraint': gi.get('isi_constraint', 'all'),
#                     'alignment_point': trial_data_filtered.get('alignment_point', 'unknown'),
#                     'n_trials_used': trials_used
#                 }
#             }
#             print("Stored canonical sorting order in M['raster_analysis']['canonical_order']")
    
#     # *** Compute SHARED color limits across both groups ***
#     pcts = cfg.get('plots', {}).get('raster', {}).get('vmin_vmax_percentiles', [5, 95])
    
#     def _get_shared_color_limits(data1, data2):
#         """Compute color limits from combined data of both groups."""
#         all_finite_vals = []
        
#         # Collect finite values from both datasets
#         if data1.size > 0:
#             finite_vals1 = data1[np.isfinite(data1)]
#             if finite_vals1.size > 0:
#                 all_finite_vals.append(finite_vals1)
        
#         if data2.size > 0:
#             finite_vals2 = data2[np.isfinite(data2)]
#             if finite_vals2.size > 0:
#                 all_finite_vals.append(finite_vals2)
        
#         if not all_finite_vals:
#             return (-0.1, 0.1)
        
#         combined_vals = np.concatenate(all_finite_vals)
#         if combined_vals.size < 10:
#             return (np.nanmin(combined_vals), np.nanmax(combined_vals))
        
#         vlims = tuple(np.percentile(combined_vals, pcts))
#         print(f"Shared color limits: {vlims[0]:.4f} to {vlims[1]:.4f} (from {combined_vals.size} finite values)")
#         return vlims
    
#     # Use shared color limits for both groups
#     vlims_shared = _get_shared_color_limits(ramp_mean_sorted, nonramp_mean_sorted)
    
#     # Set up figure
#     n_ramp = ramp_mean_sorted.shape[0]
#     n_nonramp = nonramp_mean_sorted.shape[0]
    
#     height_ramp = max(2.0, 0.02 * n_ramp) if n_ramp > 0 else 0.5
#     height_nonramp = max(2.0, 0.02 * n_nonramp) if n_nonramp > 0 else 0.5
    
#     fig, axes = plt.subplots(2, 1, figsize=(12, height_ramp + height_nonramp), 
#                             sharex=True, gridspec_kw={'height_ratios': [height_ramp, height_nonramp]})
    
#     # Plot ramp group (using shared color limits)
#     if n_ramp > 0:
#         im1 = axes[0].imshow(ramp_mean_sorted, aspect='auto', origin='lower',
#                             extent=[T_common[0], T_common[-1], 0, n_ramp],
#                             vmin=vlims_shared[0], vmax=vlims_shared[1], 
#                             cmap='viridis', interpolation='nearest')
#         axes[0].set_title(f'Ramp ROIs (n={n_ramp}) - Session Average, Sorted by {ramp_sort_method_used}')
#         axes[0].set_ylabel('ROI Index')
        
#         # Add colorbar
#         cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.02, pad=0.02)
#         cbar1.set_label('ΔF/F')
#     else:
#         axes[0].text(0.5, 0.5, 'No ramp ROIs', ha='center', va='center', transform=axes[0].transAxes)
#         axes[0].set_title('Ramp ROIs (n=0)')
    
#     # Plot non-ramp group (using same shared color limits)
#     if n_nonramp > 0:
#         im2 = axes[1].imshow(nonramp_mean_sorted, aspect='auto', origin='lower',
#                             extent=[T_common[0], T_common[-1], 0, n_nonramp],
#                             vmin=vlims_shared[0], vmax=vlims_shared[1], 
#                             cmap='viridis', interpolation='nearest')
#         axes[1].set_title(f'Non-Ramp ROIs (n={n_nonramp}) - Session Average, Sorted by {nonramp_sort_method_used}')
#         axes[1].set_ylabel('ROI Index')
        
#         # Add colorbar
#         cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.02, pad=0.02)
#         cbar2.set_label('ΔF/F')
#     else:
#         axes[1].text(0.5, 0.5, 'No non-ramp ROIs', ha='center', va='center', transform=axes[1].transAxes)
#         axes[1].set_title('Non-Ramp ROIs (n=0)')
    
#     # Common formatting
#     for ax in axes:
#         ax.axvline(0.0, color='white', linestyle='--', linewidth=1, alpha=0.8)
#         ax.grid(False)
    
#     axes[1].set_xlabel(f'Time from {trial_data_filtered.get("alignment_point", "alignment")} (s)')
    
#     plt.tight_layout()
    
#     # Store results in M for reference (update the existing analysis)
#     M.setdefault('raster_analysis', {})
#     M['raster_analysis']['session_rasters'] = {
#         'time_common': T_common,
#         'ramp_mean_sorted': ramp_mean_sorted,
#         'nonramp_mean_sorted': nonramp_mean_sorted,
#         'ramp_roi_indices': ramp_roi_sorted,
#         'nonramp_roi_indices': nonramp_roi_sorted,
#         'ramp_sort_method': ramp_sort_method_used,
#         'nonramp_sort_method': nonramp_sort_method_used,
#         'trials_used': trials_used,
#         'alignment_point': trial_data_filtered.get('alignment_point', 'unknown'),
#         'filter_params': cfg.get('group_inspect', {}),
#         'shared_color_limits': vlims_shared,
#         'used_stored_order': use_stored_order and stored_order_available
#     }
    
#     print("=== Completed plot_ramp_vs_nonramp_session_rasters ===\n")
#     return fig


# def plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg, 
#                                         use_stored_order=False, store_order=True):
#     """
#     Plot session-averaged rasters for ramp vs non-ramp ROIs using original dF/F traces.
#     ROIs within each group are sorted by configurable methods.
    
#     When filtering by specific ISI values, adds event markers for aligned stimulus/choice events.
    
#     Parameters:
#     - use_stored_order: if True, use previously stored sorting order from M['raster_analysis']['canonical_order']
#     - store_order: if True, save the computed sorting order to M for future use
    
#     Uses the native dff_time_vector timeline from trial_data (already aligned to alignment_point).
#     """
#     print("=== Starting plot_ramp_vs_nonramp_session_rasters ===")
    
#     if 'ramp_mask_phi' not in M:
#         raise KeyError("M['ramp_mask_phi'] not found. Run ramp mask analysis first.")
    
#     ramp_mask = np.asarray(M['ramp_mask_phi'], bool)
#     df = trial_data_filtered['df_trials_with_segments']
#     n_trials = len(df)
#     n_rois = int(M['n_rois'])
    
#     print(f"Processing {n_trials} filtered trials")
#     print(f"Ramp mask: {ramp_mask.sum()}/{len(ramp_mask)} ROIs")
#     print(f"Alignment point: {trial_data_filtered.get('alignment_point', 'unknown')}")
    
#     # Get sorting methods from config
#     gi = cfg.get('group_inspect', {})
#     ramp_sort_method = gi.get('ramp_sort', 'min_peak_time')
#     nonramp_sort_method = gi.get('non_ramp_sort', 'abs_peak_time')
    
#     print(f"Sorting methods: ramp={ramp_sort_method}, non-ramp={nonramp_sort_method}")
#     print(f"Use stored order: {use_stored_order}, Store order: {store_order}")
    
#     # Check if we should use stored order
#     stored_order_available = False
#     if use_stored_order and 'raster_analysis' in M and 'canonical_order' in M['raster_analysis']:
#         stored = M['raster_analysis']['canonical_order']
#         if ('ramp_roi_sorted' in stored and 'nonramp_roi_sorted' in stored and
#             'ramp_sort_method' in stored and 'nonramp_sort_method' in stored):
#             stored_order_available = True
#             print(f"Found stored order: ramp method={stored['ramp_sort_method']}, "
#                   f"non-ramp method={stored['nonramp_sort_method']}")
    
#     # *** NEW: Check for aligned events when filtering by specific ISI(s) ***
#     isi_constraint = gi.get('isi_constraint', 'all')
#     show_event_markers = False
#     aligned_events = {}
    
#     if (isinstance(isi_constraint, (list, tuple, np.ndarray)) and len(isi_constraint) == 1) or \
#        (isinstance(isi_constraint, str) and isi_constraint in ['short', 'long']):
#         # Single ISI or homogeneous ISI group - events should be aligned
#         show_event_markers = True
        
#         # Extract event times relative to alignment point
#         alignment_point = trial_data_filtered.get('alignment_point', 'trial_start')
        
#         # Get event columns that exist in the data
#         event_columns = {
#             'start_flash_1': 'F1 ON',
#             'end_flash_1': 'F1 OFF', 
#             'start_flash_2': 'F2 ON',
#             'end_flash_2': 'F2 OFF',
#             'choice_start': 'Choice'
#         }
        
#         # Calculate mean event times (should be very consistent for single ISI)
#         for col, label in event_columns.items():
#             if col in df.columns:
#                 event_times = df[col].to_numpy()
#                 finite_times = event_times[np.isfinite(event_times)]
#                 if len(finite_times) > 0:
#                     mean_time = np.mean(finite_times)
#                     std_time = np.std(finite_times)
                    
#                     # Only add marker if events are well-aligned (std < 10ms for single ISI)
#                     max_std = 0.01 if isinstance(isi_constraint, (list, tuple, np.ndarray)) else 0.05
#                     if std_time < max_std:
#                         aligned_events[label] = mean_time
#                         print(f"Event marker: {label} at {mean_time:.3f}s ± {std_time*1000:.1f}ms")
#                     else:
#                         print(f"Event {label} not well-aligned (std={std_time*1000:.1f}ms), skipping marker")
        
#         if aligned_events:
#             print(f"Will show {len(aligned_events)} event markers on rasters")
#         else:
#             show_event_markers = False
#             print("No well-aligned events found for markers")
    
#     # Find common time axis across all trials
#     all_time_vecs = []
#     for idx, (_, row) in enumerate(df.iterrows()):
#         t_vec = np.asarray(row['dff_time_vector'], dtype=float)
#         all_time_vecs.append(t_vec)
#         if idx < 3:  # Print first few for verification
#             print(f"Trial {idx} time range: {t_vec.min():.3f} to {t_vec.max():.3f}s")
    
#     # Create common time grid spanning all trials
#     t_min = min(t.min() for t in all_time_vecs)
#     t_max = max(t.max() for t in all_time_vecs)
    
#     # Use finest resolution from any trial, but cap at reasonable limit
#     dt_all = [np.median(np.diff(t)) for t in all_time_vecs if len(t) > 1]
#     dt_common = min(dt_all) if dt_all else 0.01
#     dt_common = max(dt_common, 0.005)  # Don't go below 5ms
    
#     n_bins = int(np.ceil((t_max - t_min) / dt_common))
#     n_bins = min(n_bins, 5000)  # Cap total bins for memory
    
#     T_common = np.linspace(t_min, t_max, n_bins)
#     print(f"Common time grid: {T_common[0]:.3f} to {T_common[-1]:.3f}s, {n_bins} bins, dt={dt_common:.4f}s")
    
#     # Initialize arrays for session averages
#     ramp_traces = []  # List of (n_rois_ramp, n_bins) arrays per trial
#     nonramp_traces = []  # List of (n_rois_nonramp, n_bins) arrays per trial
    
#     ramp_roi_idx = np.where(ramp_mask)[0]
#     nonramp_roi_idx = np.where(~ramp_mask)[0]
    
#     print(f"Ramp ROIs: {len(ramp_roi_idx)}, Non-ramp ROIs: {len(nonramp_roi_idx)}")
    
#     # Process each trial
#     trials_used = 0
#     for trial_idx, (_, row) in enumerate(df.iterrows()):
#         if trial_idx % 20 == 0:
#             print(f"Processing trial {trial_idx}/{n_trials}")
        
#         t_vec = np.asarray(row['dff_time_vector'], dtype=float)
#         dff = np.asarray(row['dff_segment'], dtype=float)  # (n_rois, n_samples)
        
#         if dff.shape[0] != n_rois:
#             print(f"Trial {trial_idx}: dF/F shape mismatch, skipping")
#             continue
        
#         # Ensure monotonic time
#         if not np.all(np.diff(t_vec) > 0):
#             order = np.argsort(t_vec)
#             t_vec = t_vec[order]
#             dff = dff[:, order]
        
#         # Interpolate each ROI to common grid
#         dff_interp = np.full((n_rois, n_bins), np.nan, dtype=float)
#         for r in range(n_rois):
#             # Only interpolate within the native support
#             valid = np.isfinite(dff[r])
#             if valid.sum() < 3:
#                 continue
            
#             y_vals = dff[r][valid]
#             t_vals = t_vec[valid]
            
#             # Interpolate only within native range
#             mask_interp = (T_common >= t_vals.min()) & (T_common <= t_vals.max())
#             if mask_interp.sum() > 0:
#                 dff_interp[r, mask_interp] = np.interp(
#                     T_common[mask_interp], t_vals, y_vals
#                 )
        
#         # Separate into ramp vs non-ramp groups
#         if len(ramp_roi_idx) > 0:
#             ramp_traces.append(dff_interp[ramp_roi_idx])
#         if len(nonramp_roi_idx) > 0:
#             nonramp_traces.append(dff_interp[nonramp_roi_idx])
        
#         trials_used += 1
    
#     print(f"Successfully processed {trials_used}/{n_trials} trials")
    
#     # Compute session averages
#     if ramp_traces:
#         ramp_stack = np.stack(ramp_traces, axis=0)  # (trials, rois, bins)
#         ramp_mean = np.nanmean(ramp_stack, axis=0)  # (rois, bins)
#         print(f"Ramp group session average shape: {ramp_mean.shape}")
#     else:
#         ramp_mean = np.empty((0, n_bins))
    
#     if nonramp_traces:
#         nonramp_stack = np.stack(nonramp_traces, axis=0)  # (trials, rois, bins)
#         nonramp_mean = np.nanmean(nonramp_stack, axis=0)  # (rois, bins)
#         print(f"Non-ramp group session average shape: {nonramp_mean.shape}")
#     else:
#         nonramp_mean = np.empty((0, n_bins))
    
#     # [Sorting logic remains the same as before - not repeated for brevity]
#     # Determine sorting order
#     if stored_order_available and use_stored_order:
#         # Use stored canonical order
#         stored = M['raster_analysis']['canonical_order']
        
#         # Map stored indices back to current group indices
#         stored_ramp_global = stored['ramp_roi_sorted']
#         stored_nonramp_global = stored['nonramp_roi_sorted']
        
#         # Find intersection with current groups (in case ramp mask changed)
#         current_ramp_set = set(ramp_roi_idx)
#         current_nonramp_set = set(nonramp_roi_idx)
        
#         # Keep stored order but only for ROIs still in the current groups
#         ramp_roi_sorted = [idx for idx in stored_ramp_global if idx in current_ramp_set]
#         nonramp_roi_sorted = [idx for idx in stored_nonramp_global if idx in current_nonramp_set]
        
#         # Convert to arrays and get local indices
#         ramp_roi_sorted = np.array(ramp_roi_sorted)
#         nonramp_roi_sorted = np.array(nonramp_roi_sorted)
        
#         # Get local indices within the group for matrix indexing
#         if len(ramp_roi_sorted) > 0:
#             # Map global ROI indices to local group indices
#             ramp_local_idx = np.array([np.where(ramp_roi_idx == global_idx)[0][0] 
#                                      for global_idx in ramp_roi_sorted])
#             ramp_mean_sorted = ramp_mean[ramp_local_idx]
#         else:
#             ramp_mean_sorted = ramp_mean
            
#         if len(nonramp_roi_sorted) > 0:
#             nonramp_local_idx = np.array([np.where(nonramp_roi_idx == global_idx)[0][0] 
#                                         for global_idx in nonramp_roi_sorted])
#             nonramp_mean_sorted = nonramp_mean[nonramp_local_idx]
#         else:
#             nonramp_mean_sorted = nonramp_mean
        
#         print(f"Using stored order: ramp ROIs={len(ramp_roi_sorted)}, non-ramp ROIs={len(nonramp_roi_sorted)}")
        
#         # Update the used methods to reflect what was stored
#         ramp_sort_method_used = stored['ramp_sort_method']
#         nonramp_sort_method_used = stored['nonramp_sort_method']
        
#     else:
#         # Compute new sorting order
#         print("Computing new sorting order...")
        
#         # Get tiebreak values (slopes) if available
#         tiebreak_values = None
#         if 'cluster_quickstats' in M and 'slope_time' in M['cluster_quickstats']:
#             tiebreak_values = M['cluster_quickstats']['slope_time']
        
#         # Sort ramp group
#         if ramp_mean.shape[0] > 0:
#             sort_key_ramp, sort_vals_ramp = _sort_rois_by_method(
#                 ramp_mean, T_common, method=ramp_sort_method,
#                 tiebreak_values=tiebreak_values[ramp_roi_idx] if tiebreak_values is not None else None
#             )
#             ramp_mean_sorted = ramp_mean[sort_key_ramp]
#             ramp_roi_sorted = ramp_roi_idx[sort_key_ramp]
            
#             finite_count = np.isfinite(sort_vals_ramp).sum()
#             print(f"Ramp group: {finite_count}/{len(sort_vals_ramp)} ROIs with detectable {ramp_sort_method}")
#             if finite_count > 0:
#                 finite_vals = sort_vals_ramp[np.isfinite(sort_vals_ramp)]
#                 print(f"Ramp {ramp_sort_method} range: {finite_vals.min():.3f} to {finite_vals.max():.3f}s")
#         else:
#             ramp_mean_sorted = ramp_mean
#             ramp_roi_sorted = ramp_roi_idx
        
#         # Sort non-ramp group
#         if nonramp_mean.shape[0] > 0:
#             sort_key_nonramp, sort_vals_nonramp = _sort_rois_by_method(
#                 nonramp_mean, T_common, method=nonramp_sort_method,
#                 tiebreak_values=tiebreak_values[nonramp_roi_idx] if tiebreak_values is not None else None
#             )
#             nonramp_mean_sorted = nonramp_mean[sort_key_nonramp]
#             nonramp_roi_sorted = nonramp_roi_idx[sort_key_nonramp]
            
#             finite_count = np.isfinite(sort_vals_nonramp).sum()
#             print(f"Non-ramp group: {finite_count}/{len(sort_vals_nonramp)} ROIs with detectable {nonramp_sort_method}")
#             if finite_count > 0:
#                 finite_vals = sort_vals_nonramp[np.isfinite(sort_vals_nonramp)]
#                 print(f"Non-ramp {nonramp_sort_method} range: {finite_vals.min():.3f} to {finite_vals.max():.3f}s")
#         else:
#             nonramp_mean_sorted = nonramp_mean
#             nonramp_roi_sorted = nonramp_roi_idx
        
#         ramp_sort_method_used = ramp_sort_method
#         nonramp_sort_method_used = nonramp_sort_method
        
#         # Store canonical order if requested
#         if store_order:
#             M.setdefault('raster_analysis', {})
#             M['raster_analysis']['canonical_order'] = {
#                 'ramp_roi_sorted': ramp_roi_sorted.astype(int),
#                 'nonramp_roi_sorted': nonramp_roi_sorted.astype(int),
#                 'ramp_sort_method': ramp_sort_method_used,
#                 'nonramp_sort_method': nonramp_sort_method_used,
#                 'computed_on': {
#                     'trial_type': gi.get('trial_type', 'all'),
#                     'isi_constraint': gi.get('isi_constraint', 'all'),
#                     'alignment_point': trial_data_filtered.get('alignment_point', 'unknown'),
#                     'n_trials_used': trials_used
#                 }
#             }
#             print("Stored canonical sorting order in M['raster_analysis']['canonical_order']")
    
#     # *** Compute SHARED color limits across both groups ***
#     pcts = cfg.get('plots', {}).get('raster', {}).get('vmin_vmax_percentiles', [5, 95])
    
#     def _get_shared_color_limits(data1, data2):
#         """Compute color limits from combined data of both groups."""
#         all_finite_vals = []
        
#         # Collect finite values from both datasets
#         if data1.size > 0:
#             finite_vals1 = data1[np.isfinite(data1)]
#             if finite_vals1.size > 0:
#                 all_finite_vals.append(finite_vals1)
        
#         if data2.size > 0:
#             finite_vals2 = data2[np.isfinite(data2)]
#             if finite_vals2.size > 0:
#                 all_finite_vals.append(finite_vals2)
        
#         if not all_finite_vals:
#             return (-0.1, 0.1)
        
#         combined_vals = np.concatenate(all_finite_vals)
#         if combined_vals.size < 10:
#             return (np.nanmin(combined_vals), np.nanmax(combined_vals))
        
#         vlims = tuple(np.percentile(combined_vals, pcts))
#         print(f"Shared color limits: {vlims[0]:.4f} to {vlims[1]:.4f} (from {combined_vals.size} finite values)")
#         return vlims
    
#     # Use shared color limits for both groups
#     vlims_shared = _get_shared_color_limits(ramp_mean_sorted, nonramp_mean_sorted)
    
#     # Set up figure
#     n_ramp = ramp_mean_sorted.shape[0]
#     n_nonramp = nonramp_mean_sorted.shape[0]
    
#     height_ramp = max(2.0, 0.02 * n_ramp) if n_ramp > 0 else 0.5
#     height_nonramp = max(2.0, 0.02 * n_nonramp) if n_nonramp > 0 else 0.5
    
#     fig, axes = plt.subplots(2, 1, figsize=(12, height_ramp + height_nonramp), 
#                             sharex=True, gridspec_kw={'height_ratios': [height_ramp, height_nonramp]})
    
#     # Plot ramp group (using shared color limits)
#     if n_ramp > 0:
#         im1 = axes[0].imshow(ramp_mean_sorted, aspect='auto', origin='lower',
#                             extent=[T_common[0], T_common[-1], 0, n_ramp],
#                             vmin=vlims_shared[0], vmax=vlims_shared[1], 
#                             cmap='viridis', interpolation='nearest')
#         axes[0].set_title(f'Ramp ROIs (n={n_ramp}) - Session Average, Sorted by {ramp_sort_method_used}')
#         axes[0].set_ylabel('ROI Index')
        
#         # Add colorbar
#         cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.02, pad=0.02)
#         cbar1.set_label('ΔF/F')
#     else:
#         axes[0].text(0.5, 0.5, 'No ramp ROIs', ha='center', va='center', transform=axes[0].transAxes)
#         axes[0].set_title('Ramp ROIs (n=0)')
    
#     # Plot non-ramp group (using same shared color limits)
#     if n_nonramp > 0:
#         im2 = axes[1].imshow(nonramp_mean_sorted, aspect='auto', origin='lower',
#                             extent=[T_common[0], T_common[-1], 0, n_nonramp],
#                             vmin=vlims_shared[0], vmax=vlims_shared[1], 
#                             cmap='viridis', interpolation='nearest')
#         axes[1].set_title(f'Non-Ramp ROIs (n={n_nonramp}) - Session Average, Sorted by {nonramp_sort_method_used}')
#         axes[1].set_ylabel('ROI Index')
        
#         # Add colorbar
#         cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.02, pad=0.02)
#         cbar2.set_label('ΔF/F')
#     else:
#         axes[1].text(0.5, 0.5, 'No non-ramp ROIs', ha='center', va='center', transform=axes[1].transAxes)
#         axes[1].set_title('Non-Ramp ROIs (n=0)')
    
#     # *** NEW: Add event markers for aligned events ***
#     if show_event_markers and aligned_events:
#         # Define colors and line styles for different events
#         event_styles = {
#             'F1 ON': {'color': 'red', 'linestyle': '-', 'alpha': 0.8, 'linewidth': 1.5},
#             'F1 OFF': {'color': 'red', 'linestyle': '--', 'alpha': 0.8, 'linewidth': 1.5},
#             'F2 ON': {'color': 'orange', 'linestyle': '-', 'alpha': 0.8, 'linewidth': 1.5},
#             'F2 OFF': {'color': 'orange', 'linestyle': '--', 'alpha': 0.8, 'linewidth': 1.5},
#             'Choice': {'color': 'cyan', 'linestyle': '-', 'alpha': 0.8, 'linewidth': 1.5}
#         }
        
#         for ax in axes:
#             # Standard alignment marker (should be at 0 for most alignments)
#             ax.axvline(0.0, color='white', linestyle='--', linewidth=1, alpha=0.8)
            
#             # Event markers
#             for event_label, event_time in aligned_events.items():
#                 if event_label in event_styles:
#                     style = event_styles[event_label]
#                     ax.axvline(event_time, **style)
            
#         # Add legend to top subplot only
#         if aligned_events:
#             legend_lines = []
#             legend_labels = []
            
#             # Add alignment marker to legend
#             legend_lines.append(plt.Line2D([0], [0], color='white', linestyle='--', linewidth=1, alpha=0.8))
#             legend_labels.append(f"Alignment ({trial_data_filtered.get('alignment_point', 'unknown')})")
            
#             # Add event markers to legend
#             for event_label, event_time in aligned_events.items():
#                 if event_label in event_styles:
#                     style = event_styles[event_label]
#                     legend_lines.append(plt.Line2D([0], [0], **style))
#                     legend_labels.append(f"{event_label} ({event_time:.3f}s)")
            
#             axes[0].legend(legend_lines, legend_labels, loc='upper right', 
#                           bbox_to_anchor=(1.15, 1.0), fontsize=9, frameon=True, 
#                           fancybox=True, shadow=True)
#     else:
#         # Standard alignment marker only
#         for ax in axes:
#             ax.axvline(0.0, color='white', linestyle='--', linewidth=1, alpha=0.8)
    
#     axes[0].grid(False)
#     axes[1].grid(False)
#     axes[1].set_xlabel(f'Time from {trial_data_filtered.get("alignment_point", "alignment")} (s)')
    
#     plt.tight_layout()
    
#     # Store results in M for reference (update the existing analysis)
#     M.setdefault('raster_analysis', {})
#     M['raster_analysis']['session_rasters'] = {
#         'time_common': T_common,
#         'ramp_mean_sorted': ramp_mean_sorted,
#         'nonramp_mean_sorted': nonramp_mean_sorted,
#         'ramp_roi_indices': ramp_roi_sorted,
#         'nonramp_roi_indices': nonramp_roi_sorted,
#         'ramp_sort_method': ramp_sort_method_used,
#         'nonramp_sort_method': nonramp_sort_method_used,
#         'trials_used': trials_used,
#         'alignment_point': trial_data_filtered.get('alignment_point', 'unknown'),
#         'filter_params': cfg.get('group_inspect', {}),
#         'shared_color_limits': vlims_shared,
#         'used_stored_order': use_stored_order and stored_order_available,
#         'event_markers': aligned_events if show_event_markers else {}
#     }
    
#     print("=== Completed plot_ramp_vs_nonramp_session_rasters ===\n")
#     return fig






def plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg, 
                                        use_stored_order=False, store_order=True):
    """
    Plot session-averaged rasters for ramp vs non-ramp ROIs using original dF/F traces.
    ROIs within each group are sorted by configurable methods.
    
    When filtering by specific ISI values, adds event markers for aligned stimulus/choice events.
    
    Parameters:
    - use_stored_order: if True, use previously stored sorting order from M['raster_analysis']['canonical_order']
    - store_order: if True, save the computed sorting order to M for future use
    
    Uses the native dff_time_vector timeline from trial_data (already aligned to alignment_point).
    """
    print("=== Starting plot_ramp_vs_nonramp_session_rasters ===")
    
    if 'ramp_mask_phi' not in M:
        raise KeyError("M['ramp_mask_phi'] not found. Run ramp mask analysis first.")
    
    ramp_mask = np.asarray(M['ramp_mask_phi'], bool)
    df = trial_data_filtered['df_trials_with_segments']
    n_trials = len(df)
    n_rois = int(M['n_rois'])
    
    print(f"Processing {n_trials} filtered trials")
    print(f"Ramp mask: {ramp_mask.sum()}/{len(ramp_mask)} ROIs")
    print(f"Alignment point: {trial_data_filtered.get('alignment_point', 'unknown')}")
    
    # Get sorting methods from config
    gi = cfg.get('group_inspect', {})
    ramp_sort_method = gi.get('ramp_sort', 'min_peak_time')
    nonramp_sort_method = gi.get('non_ramp_sort', 'abs_peak_time')
    
    print(f"Sorting methods: ramp={ramp_sort_method}, non-ramp={nonramp_sort_method}")
    print(f"Use stored order: {use_stored_order}, Store order: {store_order}")
    
    # Check if we should use stored order
    stored_order_available = False
    if use_stored_order and 'raster_analysis' in M and 'canonical_order' in M['raster_analysis']:
        stored = M['raster_analysis']['canonical_order']
        if ('ramp_roi_sorted' in stored and 'nonramp_roi_sorted' in stored and
            'ramp_sort_method' in stored and 'nonramp_sort_method' in stored):
            stored_order_available = True
            print(f"Found stored order: ramp method={stored['ramp_sort_method']}, "
                  f"non-ramp method={stored['nonramp_sort_method']}")
    
    # *** MODIFIED: More permissive event marker detection ***
    isi_constraint = gi.get('isi_constraint', 'all')
    show_event_markers = False
    aligned_events = {}
    
    if (isinstance(isi_constraint, (list, tuple, np.ndarray)) and len(isi_constraint) == 1) or \
       (isinstance(isi_constraint, str) and isi_constraint in ['short', 'long']):
        # Single ISI or homogeneous ISI group - show mean event times even if variable
        show_event_markers = True
        
        # Extract event times relative to alignment point
        alignment_point = trial_data_filtered.get('alignment_point', 'trial_start')
        
        # Get event columns that exist in the data
        event_columns = {
            'start_flash_1': 'F1 ON',
            'end_flash_1': 'F1 OFF', 
            'start_flash_2': 'F2 ON',
            'end_flash_2': 'F2 OFF',
            'choice_start': 'Choice'
        }
        
        # Calculate mean event times (show regardless of variability, but warn)
        for col, label in event_columns.items():
            if col in df.columns:
                event_times = df[col].to_numpy()
                finite_times = event_times[np.isfinite(event_times)]
                if len(finite_times) > 0:
                    mean_time = np.mean(finite_times)
                    std_time = np.std(finite_times)
                    
                    # Always add the marker, but indicate if it's variable
                    aligned_events[label] = mean_time
                    if std_time < 0.01:  # < 10ms
                        print(f"Event marker: {label} at {mean_time:.3f}s ± {std_time*1000:.1f}ms (well-aligned)")
                    elif std_time < 0.05:  # < 50ms  
                        print(f"Event marker: {label} at {mean_time:.3f}s ± {std_time*1000:.1f}ms (moderately aligned)")
                    else:  # >= 50ms
                        print(f"Event marker: {label} at {mean_time:.3f}s ± {std_time*1000:.1f}ms (VARIABLE - mean shown)")
        
        if aligned_events:
            print(f"Will show {len(aligned_events)} event markers on rasters")
        else:
            show_event_markers = False
            print("No events found for markers")
    
    # Find common time axis across all trials
    all_time_vecs = []
    for idx, (_, row) in enumerate(df.iterrows()):
        t_vec = np.asarray(row['dff_time_vector'], dtype=float)
        all_time_vecs.append(t_vec)
        if idx < 3:  # Print first few for verification
            print(f"Trial {idx} time range: {t_vec.min():.3f} to {t_vec.max():.3f}s")
    
    # Create common time grid spanning all trials
    t_min = min(t.min() for t in all_time_vecs)
    t_max = max(t.max() for t in all_time_vecs)
    
    # Use finest resolution from any trial, but cap at reasonable limit
    dt_all = [np.median(np.diff(t)) for t in all_time_vecs if len(t) > 1]
    dt_common = min(dt_all) if dt_all else 0.01
    dt_common = max(dt_common, 0.005)  # Don't go below 5ms
    
    n_bins = int(np.ceil((t_max - t_min) / dt_common))
    n_bins = min(n_bins, 5000)  # Cap total bins for memory
    
    T_common = np.linspace(t_min, t_max, n_bins)
    print(f"Common time grid: {T_common[0]:.3f} to {T_common[-1]:.3f}s, {n_bins} bins, dt={dt_common:.4f}s")
    
    # Initialize arrays for session averages
    ramp_traces = []  # List of (n_rois_ramp, n_bins) arrays per trial
    nonramp_traces = []  # List of (n_rois_nonramp, n_bins) arrays per trial
    
    ramp_roi_idx = np.where(ramp_mask)[0]
    nonramp_roi_idx = np.where(~ramp_mask)[0]
    
    print(f"Ramp ROIs: {len(ramp_roi_idx)}, Non-ramp ROIs: {len(nonramp_roi_idx)}")
    
    # Process each trial
    trials_used = 0
    for trial_idx, (_, row) in enumerate(df.iterrows()):
        if trial_idx % 20 == 0:
            print(f"Processing trial {trial_idx}/{n_trials}")
        
        t_vec = np.asarray(row['dff_time_vector'], dtype=float)
        dff = np.asarray(row['dff_segment'], dtype=float)  # (n_rois, n_samples)
        
        if dff.shape[0] != n_rois:
            print(f"Trial {trial_idx}: dF/F shape mismatch, skipping")
            continue
        
        # Ensure monotonic time
        if not np.all(np.diff(t_vec) > 0):
            order = np.argsort(t_vec)
            t_vec = t_vec[order]
            dff = dff[:, order]
        
        # Interpolate each ROI to common grid
        dff_interp = np.full((n_rois, n_bins), np.nan, dtype=float)
        for r in range(n_rois):
            # Only interpolate within the native support
            valid = np.isfinite(dff[r])
            if valid.sum() < 3:
                continue
            
            y_vals = dff[r][valid]
            t_vals = t_vec[valid]
            
            # Interpolate only within native range
            mask_interp = (T_common >= t_vals.min()) & (T_common <= t_vals.max())
            if mask_interp.sum() > 0:
                dff_interp[r, mask_interp] = np.interp(
                    T_common[mask_interp], t_vals, y_vals
                )
        
        # Separate into ramp vs non-ramp groups
        if len(ramp_roi_idx) > 0:
            ramp_traces.append(dff_interp[ramp_roi_idx])
        if len(nonramp_roi_idx) > 0:
            nonramp_traces.append(dff_interp[nonramp_roi_idx])
        
        trials_used += 1
    
    print(f"Successfully processed {trials_used}/{n_trials} trials")
    
    # Compute session averages
    if ramp_traces:
        ramp_stack = np.stack(ramp_traces, axis=0)  # (trials, rois, bins)
        ramp_mean = np.nanmean(ramp_stack, axis=0)  # (rois, bins)
        print(f"Ramp group session average shape: {ramp_mean.shape}")
    else:
        ramp_mean = np.empty((0, n_bins))
    
    if nonramp_traces:
        nonramp_stack = np.stack(nonramp_traces, axis=0)  # (trials, rois, bins)
        nonramp_mean = np.nanmean(nonramp_stack, axis=0)  # (rois, bins)
        print(f"Non-ramp group session average shape: {nonramp_mean.shape}")
    else:
        nonramp_mean = np.empty((0, n_bins))
    
    # [Sorting logic remains the same as before - not repeated for brevity]
    # Determine sorting order
    if stored_order_available and use_stored_order:
        # Use stored canonical order
        stored = M['raster_analysis']['canonical_order']
        
        # Map stored indices back to current group indices
        stored_ramp_global = stored['ramp_roi_sorted']
        stored_nonramp_global = stored['nonramp_roi_sorted']
        
        # Find intersection with current groups (in case ramp mask changed)
        current_ramp_set = set(ramp_roi_idx)
        current_nonramp_set = set(nonramp_roi_idx)
        
        # Keep stored order but only for ROIs still in the current groups
        ramp_roi_sorted = [idx for idx in stored_ramp_global if idx in current_ramp_set]
        nonramp_roi_sorted = [idx for idx in stored_nonramp_global if idx in current_nonramp_set]
        
        # Convert to arrays and get local indices
        ramp_roi_sorted = np.array(ramp_roi_sorted)
        nonramp_roi_sorted = np.array(nonramp_roi_sorted)
        
        # Get local indices within the group for matrix indexing
        if len(ramp_roi_sorted) > 0:
            # Map global ROI indices to local group indices
            ramp_local_idx = np.array([np.where(ramp_roi_idx == global_idx)[0][0] 
                                     for global_idx in ramp_roi_sorted])
            ramp_mean_sorted = ramp_mean[ramp_local_idx]
        else:
            ramp_mean_sorted = ramp_mean
            
        if len(nonramp_roi_sorted) > 0:
            nonramp_local_idx = np.array([np.where(nonramp_roi_idx == global_idx)[0][0] 
                                        for global_idx in nonramp_roi_sorted])
            nonramp_mean_sorted = nonramp_mean[nonramp_local_idx]
        else:
            nonramp_mean_sorted = nonramp_mean
        
        print(f"Using stored order: ramp ROIs={len(ramp_roi_sorted)}, non-ramp ROIs={len(nonramp_roi_sorted)}")
        
        # Update the used methods to reflect what was stored
        ramp_sort_method_used = stored['ramp_sort_method']
        nonramp_sort_method_used = stored['nonramp_sort_method']
        
    else:
        # Compute new sorting order
        print("Computing new sorting order...")
        
        # Get tiebreak values (slopes) if available
        tiebreak_values = None
        if 'cluster_quickstats' in M and 'slope_time' in M['cluster_quickstats']:
            tiebreak_values = M['cluster_quickstats']['slope_time']
        
        # Sort ramp group
        if ramp_mean.shape[0] > 0:
            sort_key_ramp, sort_vals_ramp = _sort_rois_by_method(
                ramp_mean, T_common, method=ramp_sort_method,
                tiebreak_values=tiebreak_values[ramp_roi_idx] if tiebreak_values is not None else None
            )
            ramp_mean_sorted = ramp_mean[sort_key_ramp]
            ramp_roi_sorted = ramp_roi_idx[sort_key_ramp]
            
            finite_count = np.isfinite(sort_vals_ramp).sum()
            print(f"Ramp group: {finite_count}/{len(sort_vals_ramp)} ROIs with detectable {ramp_sort_method}")
            if finite_count > 0:
                finite_vals = sort_vals_ramp[np.isfinite(sort_vals_ramp)]
                print(f"Ramp {ramp_sort_method} range: {finite_vals.min():.3f} to {finite_vals.max():.3f}s")
        else:
            ramp_mean_sorted = ramp_mean
            ramp_roi_sorted = ramp_roi_idx
        
        # Sort non-ramp group
        if nonramp_mean.shape[0] > 0:
            sort_key_nonramp, sort_vals_nonramp = _sort_rois_by_method(
                nonramp_mean, T_common, method=nonramp_sort_method,
                tiebreak_values=tiebreak_values[nonramp_roi_idx] if tiebreak_values is not None else None
            )
            nonramp_mean_sorted = nonramp_mean[sort_key_nonramp]
            nonramp_roi_sorted = nonramp_roi_idx[sort_key_nonramp]
            
            finite_count = np.isfinite(sort_vals_nonramp).sum()
            print(f"Non-ramp group: {finite_count}/{len(sort_vals_nonramp)} ROIs with detectable {nonramp_sort_method}")
            if finite_count > 0:
                finite_vals = sort_vals_nonramp[np.isfinite(sort_vals_nonramp)]
                print(f"Non-ramp {nonramp_sort_method} range: {finite_vals.min():.3f} to {finite_vals.max():.3f}s")
        else:
            nonramp_mean_sorted = nonramp_mean
            nonramp_roi_sorted = nonramp_roi_idx
        
        ramp_sort_method_used = ramp_sort_method
        nonramp_sort_method_used = nonramp_sort_method
        
        # Store canonical order if requested
        if store_order:
            M.setdefault('raster_analysis', {})
            M['raster_analysis']['canonical_order'] = {
                'ramp_roi_sorted': ramp_roi_sorted.astype(int),
                'nonramp_roi_sorted': nonramp_roi_sorted.astype(int),
                'ramp_sort_method': ramp_sort_method_used,
                'nonramp_sort_method': nonramp_sort_method_used,
                'computed_on': {
                    'trial_type': gi.get('trial_type', 'all'),
                    'isi_constraint': gi.get('isi_constraint', 'all'),
                    'alignment_point': trial_data_filtered.get('alignment_point', 'unknown'),
                    'n_trials_used': trials_used
                }
            }
            print("Stored canonical sorting order in M['raster_analysis']['canonical_order']")
    
    # *** Compute SHARED color limits across both groups ***
    pcts = cfg.get('plots', {}).get('raster', {}).get('vmin_vmax_percentiles', [5, 95])
    
    def _get_shared_color_limits(data1, data2):
        """Compute color limits from combined data of both groups."""
        all_finite_vals = []
        
        # Collect finite values from both datasets
        if data1.size > 0:
            finite_vals1 = data1[np.isfinite(data1)]
            if finite_vals1.size > 0:
                all_finite_vals.append(finite_vals1)
        
        if data2.size > 0:
            finite_vals2 = data2[np.isfinite(data2)]
            if finite_vals2.size > 0:
                all_finite_vals.append(finite_vals2)
        
        if not all_finite_vals:
            return (-0.1, 0.1)
        
        combined_vals = np.concatenate(all_finite_vals)
        if combined_vals.size < 10:
            return (np.nanmin(combined_vals), np.nanmax(combined_vals))
        
        vlims = tuple(np.percentile(combined_vals, pcts))
        print(f"Shared color limits: {vlims[0]:.4f} to {vlims[1]:.4f} (from {combined_vals.size} finite values)")
        return vlims
    
    # Use shared color limits for both groups
    vlims_shared = _get_shared_color_limits(ramp_mean_sorted, nonramp_mean_sorted)
    
    # Set up figure
    n_ramp = ramp_mean_sorted.shape[0]
    n_nonramp = nonramp_mean_sorted.shape[0]
    
    height_ramp = max(2.0, 0.02 * n_ramp) if n_ramp > 0 else 0.5
    height_nonramp = max(2.0, 0.02 * n_nonramp) if n_nonramp > 0 else 0.5
    
    fig, axes = plt.subplots(2, 1, figsize=(12, height_ramp + height_nonramp), 
                            sharex=True, gridspec_kw={'height_ratios': [height_ramp, height_nonramp]})
    
    # Plot ramp group (using shared color limits)
    if n_ramp > 0:
        im1 = axes[0].imshow(ramp_mean_sorted, aspect='auto', origin='lower',
                            extent=[T_common[0], T_common[-1], 0, n_ramp],
                            vmin=vlims_shared[0], vmax=vlims_shared[1], 
                            cmap='viridis', interpolation='nearest')
        axes[0].set_title(f'Ramp ROIs (n={n_ramp}) - Session Average, Sorted by {ramp_sort_method_used}')
        axes[0].set_ylabel('ROI Index')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.02, pad=0.02)
        cbar1.set_label('ΔF/F')
    else:
        axes[0].text(0.5, 0.5, 'No ramp ROIs', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Ramp ROIs (n=0)')
    
    # Plot non-ramp group (using same shared color limits)
    if n_nonramp > 0:
        im2 = axes[1].imshow(nonramp_mean_sorted, aspect='auto', origin='lower',
                            extent=[T_common[0], T_common[-1], 0, n_nonramp],
                            vmin=vlims_shared[0], vmax=vlims_shared[1], 
                            cmap='viridis', interpolation='nearest')
        axes[1].set_title(f'Non-Ramp ROIs (n={n_nonramp}) - Session Average, Sorted by {nonramp_sort_method_used}')
        axes[1].set_ylabel('ROI Index')
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.02, pad=0.02)
        cbar2.set_label('ΔF/F')
    else:
        axes[1].text(0.5, 0.5, 'No non-ramp ROIs', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Non-Ramp ROIs (n=0)')
    
    # *** MODIFIED: Add event markers (always show mean times, warn about variability) ***
    if show_event_markers and aligned_events:
        # Define colors and line styles for different events
        event_styles = {
            'F1 ON': {'color': 'red', 'linestyle': '-', 'alpha': 0.8, 'linewidth': 1.5},
            'F1 OFF': {'color': 'red', 'linestyle': '--', 'alpha': 0.8, 'linewidth': 1.5},
            'F2 ON': {'color': 'orange', 'linestyle': '-', 'alpha': 0.8, 'linewidth': 1.5},
            'F2 OFF': {'color': 'orange', 'linestyle': '--', 'alpha': 0.8, 'linewidth': 1.5},
            'Choice': {'color': 'cyan', 'linestyle': '-', 'alpha': 0.8, 'linewidth': 1.5}
        }
        
        for ax in axes:
            # Standard alignment marker (should be at 0 for most alignments)
            ax.axvline(0.0, color='white', linestyle='--', linewidth=1, alpha=0.8)
            
            # Event markers
            for event_label, event_time in aligned_events.items():
                if event_label in event_styles:
                    style = event_styles[event_label]
                    ax.axvline(event_time, **style)
            
        # Add legend to top subplot only
        if aligned_events:
            legend_lines = []
            legend_labels = []
            
            # Add alignment marker to legend
            legend_lines.append(plt.Line2D([0], [0], color='white', linestyle='--', linewidth=1, alpha=0.8))
            legend_labels.append(f"Alignment ({trial_data_filtered.get('alignment_point', 'unknown')})")
            
            # Add event markers to legend
            for event_label, event_time in aligned_events.items():
                if event_label in event_styles:
                    style = event_styles[event_label]
                    legend_lines.append(plt.Line2D([0], [0], **style))
                    legend_labels.append(f"{event_label} ({event_time:.3f}s)")
            
            axes[0].legend(legend_lines, legend_labels, loc='upper right', 
                          bbox_to_anchor=(1.15, 1.0), fontsize=9, frameon=True, 
                          fancybox=True, shadow=True)
    else:
        # Standard alignment marker only
        for ax in axes:
            ax.axvline(0.0, color='white', linestyle='--', linewidth=1, alpha=0.8)
    
    axes[0].grid(False)
    axes[1].grid(False)
    axes[1].set_xlabel(f'Time from {trial_data_filtered.get("alignment_point", "alignment")} (s)')
    
    plt.tight_layout()
    
    # Store results in M for reference (update the existing analysis)
    M.setdefault('raster_analysis', {})
    M['raster_analysis']['session_rasters'] = {
        'time_common': T_common,
        'ramp_mean_sorted': ramp_mean_sorted,
        'nonramp_mean_sorted': nonramp_mean_sorted,
        'ramp_roi_indices': ramp_roi_sorted,
        'nonramp_roi_indices': nonramp_roi_sorted,
        'ramp_sort_method': ramp_sort_method_used,
        'nonramp_sort_method': nonramp_sort_method_used,
        'trials_used': trials_used,
        'alignment_point': trial_data_filtered.get('alignment_point', 'unknown'),
        'filter_params': cfg.get('group_inspect', {}),
        'shared_color_limits': vlims_shared,
        'used_stored_order': use_stored_order and stored_order_available,
        'event_markers': aligned_events if show_event_markers else {}
    }
    
    print("=== Completed plot_ramp_vs_nonramp_session_rasters ===\n")
    return fig





def reset_canonical_sorting_order(M):
    """Helper function to clear stored sorting order when you want to recompute."""
    if 'raster_analysis' in M and 'canonical_order' in M['raster_analysis']:
        del M['raster_analysis']['canonical_order']
        print("Cleared canonical sorting order. Next plot will recompute order.")
    else:
        print("No canonical sorting order found to clear.")


def get_canonical_sorting_info(M):
    """Helper function to inspect the stored sorting order."""
    if 'raster_analysis' in M and 'canonical_order' in M['raster_analysis']:
        stored = M['raster_analysis']['canonical_order']
        print("Canonical sorting order info:")
        print(f"  Ramp ROIs: {len(stored['ramp_roi_sorted'])} ROIs, method: {stored['ramp_sort_method']}")
        print(f"  Non-ramp ROIs: {len(stored['nonramp_roi_sorted'])} ROIs, method: {stored['nonramp_sort_method']}")
        print(f"  Computed on: {stored['computed_on']}")
        print(f"  Ramp ROI indices: {stored['ramp_roi_sorted'][:10]}..." if len(stored['ramp_roi_sorted']) > 10 else f"  Ramp ROI indices: {stored['ramp_roi_sorted']}")
        print(f"  Non-ramp ROI indices: {stored['nonramp_roi_sorted'][:10]}..." if len(stored['nonramp_roi_sorted']) > 10 else f"  Non-ramp ROI indices: {stored['nonramp_roi_sorted']}")
    else:
        print("No canonical sorting order stored.")



# def plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg):
#     """
#     Plot session-averaged rasters for ramp vs non-ramp ROIs using original dF/F traces.
#     ROIs within each group are sorted by configurable methods.
    
#     Uses the native dff_time_vector timeline from trial_data (already aligned to alignment_point).
#     """
#     print("=== Starting plot_ramp_vs_nonramp_session_rasters ===")
    
#     if 'ramp_mask_phi' not in M:
#         raise KeyError("M['ramp_mask_phi'] not found. Run ramp mask analysis first.")
    
#     ramp_mask = np.asarray(M['ramp_mask_phi'], bool)
#     df = trial_data_filtered['df_trials_with_segments']
#     n_trials = len(df)
#     n_rois = int(M['n_rois'])
    
#     print(f"Processing {n_trials} filtered trials")
#     print(f"Ramp mask: {ramp_mask.sum()}/{len(ramp_mask)} ROIs")
#     print(f"Alignment point: {trial_data_filtered.get('alignment_point', 'unknown')}")
    
#     # Get sorting methods from config
#     gi = cfg.get('group_inspect', {})
#     ramp_sort_method = gi.get('ramp_sort', 'min_peak_time')
#     nonramp_sort_method = gi.get('non_ramp_sort', 'abs_peak_time')
    
#     print(f"Sorting methods: ramp={ramp_sort_method}, non-ramp={nonramp_sort_method}")
    
#     # Find common time axis across all trials
#     all_time_vecs = []
#     for idx, (_, row) in enumerate(df.iterrows()):
#         t_vec = np.asarray(row['dff_time_vector'], dtype=float)
#         all_time_vecs.append(t_vec)
#         if idx < 3:  # Print first few for verification
#             print(f"Trial {idx} time range: {t_vec.min():.3f} to {t_vec.max():.3f}s")
    
#     # Create common time grid spanning all trials
#     t_min = min(t.min() for t in all_time_vecs)
#     t_max = max(t.max() for t in all_time_vecs)
    
#     # Use finest resolution from any trial, but cap at reasonable limit
#     dt_all = [np.median(np.diff(t)) for t in all_time_vecs if len(t) > 1]
#     dt_common = min(dt_all) if dt_all else 0.01
#     dt_common = max(dt_common, 0.005)  # Don't go below 5ms
    
#     n_bins = int(np.ceil((t_max - t_min) / dt_common))
#     n_bins = min(n_bins, 5000)  # Cap total bins for memory
    
#     T_common = np.linspace(t_min, t_max, n_bins)
#     print(f"Common time grid: {T_common[0]:.3f} to {T_common[-1]:.3f}s, {n_bins} bins, dt={dt_common:.4f}s")
    
#     # Initialize arrays for session averages
#     ramp_traces = []  # List of (n_rois_ramp, n_bins) arrays per trial
#     nonramp_traces = []  # List of (n_rois_nonramp, n_bins) arrays per trial
    
#     ramp_roi_idx = np.where(ramp_mask)[0]
#     nonramp_roi_idx = np.where(~ramp_mask)[0]
    
#     print(f"Ramp ROIs: {len(ramp_roi_idx)}, Non-ramp ROIs: {len(nonramp_roi_idx)}")
    
#     # Process each trial
#     trials_used = 0
#     for trial_idx, (_, row) in enumerate(df.iterrows()):
#         if trial_idx % 20 == 0:
#             print(f"Processing trial {trial_idx}/{n_trials}")
        
#         t_vec = np.asarray(row['dff_time_vector'], dtype=float)
#         dff = np.asarray(row['dff_segment'], dtype=float)  # (n_rois, n_samples)
        
#         if dff.shape[0] != n_rois:
#             print(f"Trial {trial_idx}: dF/F shape mismatch, skipping")
#             continue
        
#         # Ensure monotonic time
#         if not np.all(np.diff(t_vec) > 0):
#             order = np.argsort(t_vec)
#             t_vec = t_vec[order]
#             dff = dff[:, order]
        
#         # Interpolate each ROI to common grid
#         dff_interp = np.full((n_rois, n_bins), np.nan, dtype=float)
#         for r in range(n_rois):
#             # Only interpolate within the native support
#             valid = np.isfinite(dff[r])
#             if valid.sum() < 3:
#                 continue
            
#             y_vals = dff[r][valid]
#             t_vals = t_vec[valid]
            
#             # Interpolate only within native range
#             mask_interp = (T_common >= t_vals.min()) & (T_common <= t_vals.max())
#             if mask_interp.sum() > 0:
#                 dff_interp[r, mask_interp] = np.interp(
#                     T_common[mask_interp], t_vals, y_vals
#                 )
        
#         # Separate into ramp vs non-ramp groups
#         if len(ramp_roi_idx) > 0:
#             ramp_traces.append(dff_interp[ramp_roi_idx])
#         if len(nonramp_roi_idx) > 0:
#             nonramp_traces.append(dff_interp[nonramp_roi_idx])
        
#         trials_used += 1
    
#     print(f"Successfully processed {trials_used}/{n_trials} trials")
    
#     # Compute session averages
#     if ramp_traces:
#         ramp_stack = np.stack(ramp_traces, axis=0)  # (trials, rois, bins)
#         ramp_mean = np.nanmean(ramp_stack, axis=0)  # (rois, bins)
#         print(f"Ramp group session average shape: {ramp_mean.shape}")
#     else:
#         ramp_mean = np.empty((0, n_bins))
    
#     if nonramp_traces:
#         nonramp_stack = np.stack(nonramp_traces, axis=0)  # (trials, rois, bins)
#         nonramp_mean = np.nanmean(nonramp_stack, axis=0)  # (rois, bins)
#         print(f"Non-ramp group session average shape: {nonramp_mean.shape}")
#     else:
#         nonramp_mean = np.empty((0, n_bins))
    
#     # Get tiebreak values (slopes) if available
#     tiebreak_values = None
#     if 'cluster_quickstats' in M and 'slope_time' in M['cluster_quickstats']:
#         tiebreak_values = M['cluster_quickstats']['slope_time']
    
#     # Sort ramp group
#     if ramp_mean.shape[0] > 0:
#         sort_key_ramp, sort_vals_ramp = _sort_rois_by_method(
#             ramp_mean, T_common, method=ramp_sort_method,
#             tiebreak_values=tiebreak_values[ramp_roi_idx] if tiebreak_values is not None else None
#         )
#         ramp_mean_sorted = ramp_mean[sort_key_ramp]
#         ramp_roi_sorted = ramp_roi_idx[sort_key_ramp]
        
#         finite_count = np.isfinite(sort_vals_ramp).sum()
#         print(f"Ramp group: {finite_count}/{len(sort_vals_ramp)} ROIs with detectable {ramp_sort_method}")
#         if finite_count > 0:
#             finite_vals = sort_vals_ramp[np.isfinite(sort_vals_ramp)]
#             print(f"Ramp {ramp_sort_method} range: {finite_vals.min():.3f} to {finite_vals.max():.3f}s")
#     else:
#         ramp_mean_sorted = ramp_mean
#         ramp_roi_sorted = ramp_roi_idx
    
#     # Sort non-ramp group
#     if nonramp_mean.shape[0] > 0:
#         sort_key_nonramp, sort_vals_nonramp = _sort_rois_by_method(
#             nonramp_mean, T_common, method=nonramp_sort_method,
#             tiebreak_values=tiebreak_values[nonramp_roi_idx] if tiebreak_values is not None else None
#         )
#         nonramp_mean_sorted = nonramp_mean[sort_key_nonramp]
#         nonramp_roi_sorted = nonramp_roi_idx[sort_key_nonramp]
        
#         finite_count = np.isfinite(sort_vals_nonramp).sum()
#         print(f"Non-ramp group: {finite_count}/{len(sort_vals_nonramp)} ROIs with detectable {nonramp_sort_method}")
#         if finite_count > 0:
#             finite_vals = sort_vals_nonramp[np.isfinite(sort_vals_nonramp)]
#             print(f"Non-ramp {nonramp_sort_method} range: {finite_vals.min():.3f} to {finite_vals.max():.3f}s")
#     else:
#         nonramp_mean_sorted = nonramp_mean
#         nonramp_roi_sorted = nonramp_roi_idx
    
#     # *** FIXED: Compute SHARED color limits across both groups ***
#     pcts = cfg.get('plots', {}).get('raster', {}).get('vmin_vmax_percentiles', [5, 95])
    
#     def _get_shared_color_limits(data1, data2):
#         """Compute color limits from combined data of both groups."""
#         all_finite_vals = []
        
#         # Collect finite values from both datasets
#         if data1.size > 0:
#             finite_vals1 = data1[np.isfinite(data1)]
#             if finite_vals1.size > 0:
#                 all_finite_vals.append(finite_vals1)
        
#         if data2.size > 0:
#             finite_vals2 = data2[np.isfinite(data2)]
#             if finite_vals2.size > 0:
#                 all_finite_vals.append(finite_vals2)
        
#         if not all_finite_vals:
#             return (-0.1, 0.1)
        
#         combined_vals = np.concatenate(all_finite_vals)
#         if combined_vals.size < 10:
#             return (np.nanmin(combined_vals), np.nanmax(combined_vals))
        
#         vlims = tuple(np.percentile(combined_vals, pcts))
#         print(f"Shared color limits: {vlims[0]:.4f} to {vlims[1]:.4f} (from {combined_vals.size} finite values)")
#         return vlims
    
#     # Use shared color limits for both groups
#     vlims_shared = _get_shared_color_limits(ramp_mean_sorted, nonramp_mean_sorted)
    
#     # Set up figure
#     n_ramp = ramp_mean_sorted.shape[0]
#     n_nonramp = nonramp_mean_sorted.shape[0]
    
#     height_ramp = max(2.0, 0.02 * n_ramp) if n_ramp > 0 else 0.5
#     height_nonramp = max(2.0, 0.02 * n_nonramp) if n_nonramp > 0 else 0.5
    
#     fig, axes = plt.subplots(2, 1, figsize=(12, height_ramp + height_nonramp), 
#                             sharex=True, gridspec_kw={'height_ratios': [height_ramp, height_nonramp]})
    
#     # Plot ramp group (using shared color limits)
#     if n_ramp > 0:
#         im1 = axes[0].imshow(ramp_mean_sorted, aspect='auto', origin='lower',
#                             extent=[T_common[0], T_common[-1], 0, n_ramp],
#                             vmin=vlims_shared[0], vmax=vlims_shared[1], 
#                             cmap='viridis', interpolation='nearest')
#         axes[0].set_title(f'Ramp ROIs (n={n_ramp}) - Session Average, Sorted by {ramp_sort_method}')
#         axes[0].set_ylabel('ROI Index')
        
#         # Add colorbar
#         cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.02, pad=0.02)
#         cbar1.set_label('ΔF/F')
#     else:
#         axes[0].text(0.5, 0.5, 'No ramp ROIs', ha='center', va='center', transform=axes[0].transAxes)
#         axes[0].set_title('Ramp ROIs (n=0)')
    
#     # Plot non-ramp group (using same shared color limits)
#     if n_nonramp > 0:
#         im2 = axes[1].imshow(nonramp_mean_sorted, aspect='auto', origin='lower',
#                             extent=[T_common[0], T_common[-1], 0, n_nonramp],
#                             vmin=vlims_shared[0], vmax=vlims_shared[1], 
#                             cmap='viridis', interpolation='nearest')
#         axes[1].set_title(f'Non-Ramp ROIs (n={n_nonramp}) - Session Average, Sorted by {nonramp_sort_method}')
#         axes[1].set_ylabel('ROI Index')
        
#         # Add colorbar
#         cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.02, pad=0.02)
#         cbar2.set_label('ΔF/F')
#     else:
#         axes[1].text(0.5, 0.5, 'No non-ramp ROIs', ha='center', va='center', transform=axes[1].transAxes)
#         axes[1].set_title('Non-Ramp ROIs (n=0)')
    
#     # Common formatting
#     for ax in axes:
#         ax.axvline(0.0, color='white', linestyle='--', linewidth=1, alpha=0.8)
#         ax.grid(False)
    
#     axes[1].set_xlabel(f'Time from {trial_data_filtered.get("alignment_point", "alignment")} (s)')
    
#     plt.tight_layout()
    
#     # Store results in M for reference
#     M.setdefault('raster_analysis', {})
#     M['raster_analysis']['session_rasters'] = {
#         'time_common': T_common,
#         'ramp_mean_sorted': ramp_mean_sorted,
#         'nonramp_mean_sorted': nonramp_mean_sorted,
#         'ramp_roi_indices': ramp_roi_sorted,
#         'nonramp_roi_indices': nonramp_roi_sorted,
#         'ramp_sort_method': ramp_sort_method,
#         'nonramp_sort_method': nonramp_sort_method,
#         'trials_used': trials_used,
#         'alignment_point': trial_data_filtered.get('alignment_point', 'unknown'),
#         'filter_params': cfg.get('group_inspect', {}),
#         'shared_color_limits': vlims_shared  # Store for reference
#     }
    
#     print("=== Completed plot_ramp_vs_nonramp_session_rasters ===\n")
#     return fig


# def plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg):
#     """
#     Plot session-averaged rasters for ramp vs non-ramp ROIs using original dF/F traces.
#     ROIs within each group are sorted by configurable methods.
    
#     Uses the native dff_time_vector timeline from trial_data (already aligned to alignment_point).
#     """
#     print("=== Starting plot_ramp_vs_nonramp_session_rasters ===")
    
#     if 'ramp_mask_phi' not in M:
#         raise KeyError("M['ramp_mask_phi'] not found. Run ramp mask analysis first.")
    
#     ramp_mask = np.asarray(M['ramp_mask_phi'], bool)
#     df = trial_data_filtered['df_trials_with_segments']
#     n_trials = len(df)
#     n_rois = int(M['n_rois'])
    
#     print(f"Processing {n_trials} filtered trials")
#     print(f"Ramp mask: {ramp_mask.sum()}/{len(ramp_mask)} ROIs")
#     print(f"Alignment point: {trial_data_filtered.get('alignment_point', 'unknown')}")
    
#     # Get sorting methods from config
#     gi = cfg.get('group_inspect', {})
#     ramp_sort_method = gi.get('ramp_sort', 'min_peak_time')
#     nonramp_sort_method = gi.get('non_ramp_sort', 'abs_peak_time')
    
#     print(f"Sorting methods: ramp={ramp_sort_method}, non-ramp={nonramp_sort_method}")
    
#     # Find common time axis across all trials
#     all_time_vecs = []
#     for idx, (_, row) in enumerate(df.iterrows()):
#         t_vec = np.asarray(row['dff_time_vector'], dtype=float)
#         all_time_vecs.append(t_vec)
#         if idx < 3:  # Print first few for verification
#             print(f"Trial {idx} time range: {t_vec.min():.3f} to {t_vec.max():.3f}s")
    
#     # Create common time grid spanning all trials
#     t_min = min(t.min() for t in all_time_vecs)
#     t_max = max(t.max() for t in all_time_vecs)
    
#     # Use finest resolution from any trial, but cap at reasonable limit
#     dt_all = [np.median(np.diff(t)) for t in all_time_vecs if len(t) > 1]
#     dt_common = min(dt_all) if dt_all else 0.01
#     dt_common = max(dt_common, 0.005)  # Don't go below 5ms
    
#     n_bins = int(np.ceil((t_max - t_min) / dt_common))
#     n_bins = min(n_bins, 5000)  # Cap total bins for memory
    
#     T_common = np.linspace(t_min, t_max, n_bins)
#     print(f"Common time grid: {T_common[0]:.3f} to {T_common[-1]:.3f}s, {n_bins} bins, dt={dt_common:.4f}s")
    
#     # Initialize arrays for session averages
#     ramp_traces = []  # List of (n_rois_ramp, n_bins) arrays per trial
#     nonramp_traces = []  # List of (n_rois_nonramp, n_bins) arrays per trial
    
#     ramp_roi_idx = np.where(ramp_mask)[0]
#     nonramp_roi_idx = np.where(~ramp_mask)[0]
    
#     print(f"Ramp ROIs: {len(ramp_roi_idx)}, Non-ramp ROIs: {len(nonramp_roi_idx)}")
    
#     # Process each trial
#     trials_used = 0
#     for trial_idx, (_, row) in enumerate(df.iterrows()):
#         if trial_idx % 20 == 0:
#             print(f"Processing trial {trial_idx}/{n_trials}")
        
#         t_vec = np.asarray(row['dff_time_vector'], dtype=float)
#         dff = np.asarray(row['dff_segment'], dtype=float)  # (n_rois, n_samples)
        
#         if dff.shape[0] != n_rois:
#             print(f"Trial {trial_idx}: dF/F shape mismatch, skipping")
#             continue
        
#         # Ensure monotonic time
#         if not np.all(np.diff(t_vec) > 0):
#             order = np.argsort(t_vec)
#             t_vec = t_vec[order]
#             dff = dff[:, order]
        
#         # Interpolate each ROI to common grid
#         dff_interp = np.full((n_rois, n_bins), np.nan, dtype=float)
#         for r in range(n_rois):
#             # Only interpolate within the native support
#             valid = np.isfinite(dff[r])
#             if valid.sum() < 3:
#                 continue
            
#             y_vals = dff[r][valid]
#             t_vals = t_vec[valid]
            
#             # Interpolate only within native range
#             mask_interp = (T_common >= t_vals.min()) & (T_common <= t_vals.max())
#             if mask_interp.sum() > 0:
#                 dff_interp[r, mask_interp] = np.interp(
#                     T_common[mask_interp], t_vals, y_vals
#                 )
        
#         # Separate into ramp vs non-ramp groups
#         if len(ramp_roi_idx) > 0:
#             ramp_traces.append(dff_interp[ramp_roi_idx])
#         if len(nonramp_roi_idx) > 0:
#             nonramp_traces.append(dff_interp[nonramp_roi_idx])
        
#         trials_used += 1
    
#     print(f"Successfully processed {trials_used}/{n_trials} trials")
    
#     # Compute session averages
#     if ramp_traces:
#         ramp_stack = np.stack(ramp_traces, axis=0)  # (trials, rois, bins)
#         ramp_mean = np.nanmean(ramp_stack, axis=0)  # (rois, bins)
#         print(f"Ramp group session average shape: {ramp_mean.shape}")
#     else:
#         ramp_mean = np.empty((0, n_bins))
    
#     if nonramp_traces:
#         nonramp_stack = np.stack(nonramp_traces, axis=0)  # (trials, rois, bins)
#         nonramp_mean = np.nanmean(nonramp_stack, axis=0)  # (rois, bins)
#         print(f"Non-ramp group session average shape: {nonramp_mean.shape}")
#     else:
#         nonramp_mean = np.empty((0, n_bins))
    
#     # Get tiebreak values (slopes) if available
#     tiebreak_values = None
#     if 'cluster_quickstats' in M and 'slope_time' in M['cluster_quickstats']:
#         tiebreak_values = M['cluster_quickstats']['slope_time']
    
#     # Sort ramp group
#     if ramp_mean.shape[0] > 0:
#         sort_key_ramp, sort_vals_ramp = _sort_rois_by_method(
#             ramp_mean, T_common, method=ramp_sort_method,
#             tiebreak_values=tiebreak_values[ramp_roi_idx] if tiebreak_values is not None else None
#         )
#         ramp_mean_sorted = ramp_mean[sort_key_ramp]
#         ramp_roi_sorted = ramp_roi_idx[sort_key_ramp]
        
#         finite_count = np.isfinite(sort_vals_ramp).sum()
#         print(f"Ramp group: {finite_count}/{len(sort_vals_ramp)} ROIs with detectable {ramp_sort_method}")
#         if finite_count > 0:
#             finite_vals = sort_vals_ramp[np.isfinite(sort_vals_ramp)]
#             print(f"Ramp {ramp_sort_method} range: {finite_vals.min():.3f} to {finite_vals.max():.3f}s")
#     else:
#         ramp_mean_sorted = ramp_mean
#         ramp_roi_sorted = ramp_roi_idx
    
#     # Sort non-ramp group
#     if nonramp_mean.shape[0] > 0:
#         sort_key_nonramp, sort_vals_nonramp = _sort_rois_by_method(
#             nonramp_mean, T_common, method=nonramp_sort_method,
#             tiebreak_values=tiebreak_values[nonramp_roi_idx] if tiebreak_values is not None else None
#         )
#         nonramp_mean_sorted = nonramp_mean[sort_key_nonramp]
#         nonramp_roi_sorted = nonramp_roi_idx[sort_key_nonramp]
        
#         finite_count = np.isfinite(sort_vals_nonramp).sum()
#         print(f"Non-ramp group: {finite_count}/{len(sort_vals_nonramp)} ROIs with detectable {nonramp_sort_method}")
#         if finite_count > 0:
#             finite_vals = sort_vals_nonramp[np.isfinite(sort_vals_nonramp)]
#             print(f"Non-ramp {nonramp_sort_method} range: {finite_vals.min():.3f} to {finite_vals.max():.3f}s")
#     else:
#         nonramp_mean_sorted = nonramp_mean
#         nonramp_roi_sorted = nonramp_roi_idx
    
#     # Plotting
#     pcts = cfg.get('plots', {}).get('raster', {}).get('vmin_vmax_percentiles', [5, 95])
    
#     def _get_color_limits(data):
#         if data.size == 0:
#             return (-0.1, 0.1)
#         finite_vals = data[np.isfinite(data)]
#         if finite_vals.size < 10:
#             return (np.nanmin(data), np.nanmax(data))
#         return tuple(np.percentile(finite_vals, pcts))
    
#     # Set up figure
#     n_ramp = ramp_mean_sorted.shape[0]
#     n_nonramp = nonramp_mean_sorted.shape[0]
    
#     height_ramp = max(2.0, 0.02 * n_ramp) if n_ramp > 0 else 0.5
#     height_nonramp = max(2.0, 0.02 * n_nonramp) if n_nonramp > 0 else 0.5
    
#     fig, axes = plt.subplots(2, 1, figsize=(12, height_ramp + height_nonramp), 
#                             sharex=True, gridspec_kw={'height_ratios': [height_ramp, height_nonramp]})
    
#     # Plot ramp group
#     if n_ramp > 0:
#         vlims_ramp = _get_color_limits(ramp_mean_sorted)
#         im1 = axes[0].imshow(ramp_mean_sorted, aspect='auto', origin='lower',
#                             extent=[T_common[0], T_common[-1], 0, n_ramp],
#                             vmin=vlims_ramp[0], vmax=vlims_ramp[1], 
#                             cmap='viridis', interpolation='nearest')
#         axes[0].set_title(f'Ramp ROIs (n={n_ramp}) - Session Average, Sorted by {ramp_sort_method}')
#         axes[0].set_ylabel('ROI Index')
        
#         # Add colorbar
#         cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.02, pad=0.02)
#         cbar1.set_label('ΔF/F')
#     else:
#         axes[0].text(0.5, 0.5, 'No ramp ROIs', ha='center', va='center', transform=axes[0].transAxes)
#         axes[0].set_title('Ramp ROIs (n=0)')
    
#     # Plot non-ramp group
#     if n_nonramp > 0:
#         vlims_nonramp = _get_color_limits(nonramp_mean_sorted)
#         im2 = axes[1].imshow(nonramp_mean_sorted, aspect='auto', origin='lower',
#                             extent=[T_common[0], T_common[-1], 0, n_nonramp],
#                             vmin=vlims_nonramp[0], vmax=vlims_nonramp[1], 
#                             cmap='viridis', interpolation='nearest')
#         axes[1].set_title(f'Non-Ramp ROIs (n={n_nonramp}) - Session Average, Sorted by {nonramp_sort_method}')
#         axes[1].set_ylabel('ROI Index')
        
#         # Add colorbar
#         cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.02, pad=0.02)
#         cbar2.set_label('ΔF/F')
#     else:
#         axes[1].text(0.5, 0.5, 'No non-ramp ROIs', ha='center', va='center', transform=axes[1].transAxes)
#         axes[1].set_title('Non-Ramp ROIs (n=0)')
    
#     # Common formatting
#     for ax in axes:
#         ax.axvline(0.0, color='white', linestyle='--', linewidth=1, alpha=0.8)
#         ax.grid(False)
    
#     axes[1].set_xlabel(f'Time from {trial_data_filtered.get("alignment_point", "alignment")} (s)')
    
#     plt.tight_layout()
    
#     # Store results in M for reference
#     M.setdefault('raster_analysis', {})
#     M['raster_analysis']['session_rasters'] = {
#         'time_common': T_common,
#         'ramp_mean_sorted': ramp_mean_sorted,
#         'nonramp_mean_sorted': nonramp_mean_sorted,
#         'ramp_roi_indices': ramp_roi_sorted,
#         'nonramp_roi_indices': nonramp_roi_sorted,
#         'ramp_sort_method': ramp_sort_method,
#         'nonramp_sort_method': nonramp_sort_method,
#         'trials_used': trials_used,
#         'alignment_point': trial_data_filtered.get('alignment_point', 'unknown'),
#         'filter_params': cfg.get('group_inspect', {})
#     }
    
#     print("=== Completed plot_ramp_vs_nonramp_session_rasters ===\n")
#     return fig





# def filter_trials_for_inspection(trial_data, cfg):
#     """
#     Filter trial_data based on cfg['group_inspect'] criteria.
    
#     Returns filtered copy of trial_data with only selected trial rows.
#     """
#     print("=== Starting filter_trials_for_inspection ===")
    
#     gi = cfg.get('group_inspect', {})
#     df = trial_data['df_trials_with_segments'].copy()
#     n_trials_orig = len(df)
    
#     print(f"Original trials: {n_trials_orig}")
#     print(f"Alignment point: {trial_data.get('alignment_point', 'unknown')}")
    
#     # Convert ISI values to seconds for comparison
#     isi_vec = _to_seconds(df['isi'].to_numpy())
    
#     # Trial type filter
#     trial_type = str(gi.get('trial_type', 'all')).lower()
#     if trial_type == 'rewarded':
#         mask = df['rewarded'].astype(bool)
#         print(f"Filtering to rewarded trials: {mask.sum()}/{n_trials_orig}")
#     elif trial_type == 'punished':
#         mask = df['punished'].astype(bool)
#         print(f"Filtering to punished trials: {mask.sum()}/{n_trials_orig}")
#     elif trial_type == 'correct':
#         is_right_trial = df['is_right'].astype(bool)
#         is_right_choice = df['is_right_choice'].astype(bool)
#         did_not_choose = df['did_not_choose'].astype(bool)
#         mask = (is_right_trial == is_right_choice) & (~did_not_choose)
#         print(f"Filtering to correct trials: {mask.sum()}/{n_trials_orig}")
#     elif trial_type == 'incorrect':
#         is_right_trial = df['is_right'].astype(bool)
#         is_right_choice = df['is_right_choice'].astype(bool)
#         did_not_choose = df['did_not_choose'].astype(bool)
#         correct = (is_right_trial == is_right_choice) & (~did_not_choose)
#         mask = (~correct) & (~did_not_choose)
#         print(f"Filtering to incorrect trials: {mask.sum()}/{n_trials_orig}")
#     elif trial_type == 'did_not_choose':
#         mask = df['did_not_choose'].astype(bool)
#         print(f"Filtering to did_not_choose trials: {mask.sum()}/{n_trials_orig}")
#     else:  # 'all'
#         mask = np.ones(n_trials_orig, dtype=bool)
#         print(f"Keeping all trials: {n_trials_orig}")
    
#     # ISI constraint filter
#     isi_constraint = gi.get('isi_constraint', 'all')
#     if isinstance(isi_constraint, str):
#         if isi_constraint.lower() == 'short':
#             # Use session short_isis levels
#             short_levels = _to_seconds(trial_data['session_info']['short_isis'])
#             tol = float(cfg.get('labels', {}).get('tolerance_sec', 1e-3))
#             isi_mask = _member_mask(isi_vec, short_levels, tol)
#             print(f"ISI filter (short): {isi_mask.sum()}/{n_trials_orig} trials")
#         elif isi_constraint.lower() == 'long':
#             # Use session long_isis levels
#             long_levels = _to_seconds(trial_data['session_info']['long_isis'])
#             tol = float(cfg.get('labels', {}).get('tolerance_sec', 1e-3))
#             isi_mask = _member_mask(isi_vec, long_levels, tol)
#             print(f"ISI filter (long): {isi_mask.sum()}/{n_trials_orig} trials")
#         else:  # 'all'
#             isi_mask = np.ones(n_trials_orig, dtype=bool)
#             print(f"ISI filter (all): {n_trials_orig}")
#     elif isinstance(isi_constraint, (list, tuple, np.ndarray)):
#         # Specific ISI values provided
#         target_isis = _to_seconds(isi_constraint)
#         tol = float(cfg.get('labels', {}).get('tolerance_sec', 1e-3))
#         isi_mask = _member_mask(isi_vec, target_isis, tol)
#         print(f"ISI filter (specific {target_isis}): {isi_mask.sum()}/{n_trials_orig} trials")
#     else:
#         isi_mask = np.ones(n_trials_orig, dtype=bool)
#         print(f"ISI filter (default all): {n_trials_orig}")
    
#     # Combine filters
#     final_mask = mask & isi_mask
#     df_filtered = df[final_mask].copy()
#     n_trials_final = len(df_filtered)
    
#     print(f"Final filtered trials: {n_trials_final}/{n_trials_orig}")
    
#     # Create filtered trial_data copy
#     trial_data_filtered = trial_data.copy()
#     trial_data_filtered['df_trials_with_segments'] = df_filtered
    
#     # Print summary of filtered data
#     if n_trials_final > 0:
#         print("Filtered trial summary:")
#         if 'rewarded' in df_filtered.columns:
#             print(f"  Rewarded: {df_filtered['rewarded'].sum()}")
#         if 'punished' in df_filtered.columns:
#             print(f"  Punished: {df_filtered['punished'].sum()}")
#         if 'is_right' in df_filtered.columns:
#             print(f"  Right trials: {df_filtered['is_right'].sum()}")
#         isi_filtered = _to_seconds(df_filtered['isi'].to_numpy())
#         unique_isis, counts = np.unique(np.round(isi_filtered, 3), return_counts=True)
#         print(f"  ISI distribution: {dict(zip(unique_isis, counts))}")
    
#     print("=== Completed filter_trials_for_inspection ===\n")
#     return trial_data_filtered


# def plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg):
#     """
#     Plot session-averaged rasters for ramp vs non-ramp ROIs using original dF/F traces.
#     ROIs within each group are sorted by earliest onset time.
    
#     Uses the native dff_time_vector timeline from trial_data (already aligned to alignment_point).
#     """
#     print("=== Starting plot_ramp_vs_nonramp_session_rasters ===")
    
#     if 'ramp_mask_phi' not in M:
#         raise KeyError("M['ramp_mask_phi'] not found. Run ramp mask analysis first.")
    
#     ramp_mask = np.asarray(M['ramp_mask_phi'], bool)
#     df = trial_data_filtered['df_trials_with_segments']
#     n_trials = len(df)
#     n_rois = int(M['n_rois'])
    
#     print(f"Processing {n_trials} filtered trials")
#     print(f"Ramp mask: {ramp_mask.sum()}/{len(ramp_mask)} ROIs")
#     print(f"Alignment point: {trial_data_filtered.get('alignment_point', 'unknown')}")
    
#     # Find common time axis across all trials
#     all_time_vecs = []
#     for idx, (_, row) in enumerate(df.iterrows()):
#         t_vec = np.asarray(row['dff_time_vector'], dtype=float)
#         all_time_vecs.append(t_vec)
#         if idx < 3:  # Print first few for verification
#             print(f"Trial {idx} time range: {t_vec.min():.3f} to {t_vec.max():.3f}s")
    
#     # Create common time grid spanning all trials
#     t_min = min(t.min() for t in all_time_vecs)
#     t_max = max(t.max() for t in all_time_vecs)
    
#     # Use finest resolution from any trial, but cap at reasonable limit
#     dt_all = [np.median(np.diff(t)) for t in all_time_vecs if len(t) > 1]
#     dt_common = min(dt_all) if dt_all else 0.01
#     dt_common = max(dt_common, 0.005)  # Don't go below 5ms
    
#     n_bins = int(np.ceil((t_max - t_min) / dt_common))
#     n_bins = min(n_bins, 5000)  # Cap total bins for memory
    
#     T_common = np.linspace(t_min, t_max, n_bins)
#     print(f"Common time grid: {T_common[0]:.3f} to {T_common[-1]:.3f}s, {n_bins} bins, dt={dt_common:.4f}s")
    
#     # Initialize arrays for session averages
#     ramp_traces = []  # List of (n_rois_ramp, n_bins) arrays per trial
#     nonramp_traces = []  # List of (n_rois_nonramp, n_bins) arrays per trial
    
#     ramp_roi_idx = np.where(ramp_mask)[0]
#     nonramp_roi_idx = np.where(~ramp_mask)[0]
    
#     print(f"Ramp ROIs: {len(ramp_roi_idx)}, Non-ramp ROIs: {len(nonramp_roi_idx)}")
    
#     # Process each trial
#     trials_used = 0
#     for trial_idx, (_, row) in enumerate(df.iterrows()):
#         if trial_idx % 20 == 0:
#             print(f"Processing trial {trial_idx}/{n_trials}")
        
#         t_vec = np.asarray(row['dff_time_vector'], dtype=float)
#         dff = np.asarray(row['dff_segment'], dtype=float)  # (n_rois, n_samples)
        
#         if dff.shape[0] != n_rois:
#             print(f"Trial {trial_idx}: dF/F shape mismatch, skipping")
#             continue
        
#         # Ensure monotonic time
#         if not np.all(np.diff(t_vec) > 0):
#             order = np.argsort(t_vec)
#             t_vec = t_vec[order]
#             dff = dff[:, order]
        
#         # Interpolate each ROI to common grid
#         dff_interp = np.full((n_rois, n_bins), np.nan, dtype=float)
#         for r in range(n_rois):
#             # Only interpolate within the native support
#             valid = np.isfinite(dff[r])
#             if valid.sum() < 3:
#                 continue
            
#             y_vals = dff[r][valid]
#             t_vals = t_vec[valid]
            
#             # Interpolate only within native range
#             mask_interp = (T_common >= t_vals.min()) & (T_common <= t_vals.max())
#             if mask_interp.sum() > 0:
#                 dff_interp[r, mask_interp] = np.interp(
#                     T_common[mask_interp], t_vals, y_vals
#                 )
        
#         # Separate into ramp vs non-ramp groups
#         if len(ramp_roi_idx) > 0:
#             ramp_traces.append(dff_interp[ramp_roi_idx])
#         if len(nonramp_roi_idx) > 0:
#             nonramp_traces.append(dff_interp[nonramp_roi_idx])
        
#         trials_used += 1
    
#     print(f"Successfully processed {trials_used}/{n_trials} trials")
    
#     # Compute session averages
#     if ramp_traces:
#         ramp_stack = np.stack(ramp_traces, axis=0)  # (trials, rois, bins)
#         ramp_mean = np.nanmean(ramp_stack, axis=0)  # (rois, bins)
#         print(f"Ramp group session average shape: {ramp_mean.shape}")
#     else:
#         ramp_mean = np.empty((0, n_bins))
    
#     if nonramp_traces:
#         nonramp_stack = np.stack(nonramp_traces, axis=0)  # (trials, rois, bins)
#         nonramp_mean = np.nanmean(nonramp_stack, axis=0)  # (rois, bins)
#         print(f"Non-ramp group session average shape: {nonramp_mean.shape}")
#     else:
#         nonramp_mean = np.empty((0, n_bins))
    
#     # Sort ROIs by onset within each group (reuse onset detection from earlier)
#     baseline_win = (-0.3, 0.0)  # Adjust based on your alignment
    
#     def _earliest_onset_simple(trace, t, baseline_win, thresh_z=1.0, min_bins=3):
#         """Simplified onset detection for sorting."""
#         # Baseline stats
#         bmask = (t >= baseline_win[0]) & (t <= baseline_win[1]) & np.isfinite(trace)
#         if bmask.sum() < 5:
#             return np.inf
        
#         mu = np.nanmean(trace[bmask])
#         sd = np.nanstd(trace[bmask])
#         if sd < 1e-9:
#             return np.inf
        
#         thr = mu - thresh_z * sd  # Look for negative deflections
        
#         # Find first sustained crossing
#         below = (trace <= thr) & np.isfinite(trace) & (t >= 0)  # Search after alignment point
#         if not below.any():
#             return np.inf
        
#         # Find first run of min_bins consecutive points
#         run = 0
#         for i, flag in enumerate(below):
#             run = run + 1 if flag else 0
#             if run >= min_bins:
#                 return t[i - min_bins + 1]
#         return np.inf
    
#     # Sort ramp group by onset
#     if ramp_mean.shape[0] > 0:
#         onsets_ramp = np.full(ramp_mean.shape[0], np.inf)
#         for i, trace in enumerate(ramp_mean):
#             onsets_ramp[i] = _earliest_onset_simple(trace, T_common, baseline_win)
        
#         # Sort: finite onsets first, then by onset time
#         finite_mask = np.isfinite(onsets_ramp)
#         sort_key = np.lexsort((onsets_ramp, ~finite_mask))
#         ramp_mean_sorted = ramp_mean[sort_key]
#         ramp_roi_sorted = ramp_roi_idx[sort_key]
#         onsets_ramp_sorted = onsets_ramp[sort_key]
        
#         n_with_onset = finite_mask.sum()
#         print(f"Ramp group: {n_with_onset}/{len(onsets_ramp)} ROIs with detectable onset")
#         if n_with_onset > 0:
#             print(f"Ramp onset range: {onsets_ramp_sorted[0]:.3f} to {onsets_ramp_sorted[n_with_onset-1]:.3f}s")
#     else:
#         ramp_mean_sorted = ramp_mean
#         ramp_roi_sorted = ramp_roi_idx
    
#     # Sort non-ramp group by onset
#     if nonramp_mean.shape[0] > 0:
#         onsets_nonramp = np.full(nonramp_mean.shape[0], np.inf)
#         for i, trace in enumerate(nonramp_mean):
#             onsets_nonramp[i] = _earliest_onset_simple(trace, T_common, baseline_win)
        
#         finite_mask = np.isfinite(onsets_nonramp)
#         sort_key = np.lexsort((onsets_nonramp, ~finite_mask))
#         nonramp_mean_sorted = nonramp_mean[sort_key]
#         nonramp_roi_sorted = nonramp_roi_idx[sort_key]
#         onsets_nonramp_sorted = onsets_nonramp[sort_key]
        
#         n_with_onset = finite_mask.sum()
#         print(f"Non-ramp group: {n_with_onset}/{len(onsets_nonramp)} ROIs with detectable onset")
#         if n_with_onset > 0:
#             print(f"Non-ramp onset range: {onsets_nonramp_sorted[0]:.3f} to {onsets_nonramp_sorted[n_with_onset-1]:.3f}s")
#     else:
#         nonramp_mean_sorted = nonramp_mean
#         nonramp_roi_sorted = nonramp_roi_idx
    
#     # Plotting
#     pcts = cfg.get('plots', {}).get('raster', {}).get('vmin_vmax_percentiles', [5, 95])
    
#     def _get_color_limits(data):
#         if data.size == 0:
#             return (-0.1, 0.1)
#         finite_vals = data[np.isfinite(data)]
#         if finite_vals.size < 10:
#             return (np.nanmin(data), np.nanmax(data))
#         return tuple(np.percentile(finite_vals, pcts))
    
#     # Set up figure
#     n_ramp = ramp_mean_sorted.shape[0]
#     n_nonramp = nonramp_mean_sorted.shape[0]
    
#     height_ramp = max(2.0, 0.02 * n_ramp) if n_ramp > 0 else 0.5
#     height_nonramp = max(2.0, 0.02 * n_nonramp) if n_nonramp > 0 else 0.5
    
#     fig, axes = plt.subplots(2, 1, figsize=(12, height_ramp + height_nonramp), 
#                             sharex=True, gridspec_kw={'height_ratios': [height_ramp, height_nonramp]})
    
#     # Plot ramp group
#     if n_ramp > 0:
#         vlims_ramp = _get_color_limits(ramp_mean_sorted)
#         im1 = axes[0].imshow(ramp_mean_sorted, aspect='auto', origin='lower',
#                             extent=[T_common[0], T_common[-1], 0, n_ramp],
#                             vmin=vlims_ramp[0], vmax=vlims_ramp[1], 
#                             cmap='viridis', interpolation='nearest')
#         axes[0].set_title(f'Ramp ROIs (n={n_ramp}) - Session Average, Sorted by Onset')
#         axes[0].set_ylabel('ROI Index')
        
#         # Add colorbar
#         cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.02, pad=0.02)
#         cbar1.set_label('ΔF/F')
#     else:
#         axes[0].text(0.5, 0.5, 'No ramp ROIs', ha='center', va='center', transform=axes[0].transAxes)
#         axes[0].set_title('Ramp ROIs (n=0)')
    
#     # Plot non-ramp group
#     if n_nonramp > 0:
#         vlims_nonramp = _get_color_limits(nonramp_mean_sorted)
#         im2 = axes[1].imshow(nonramp_mean_sorted, aspect='auto', origin='lower',
#                             extent=[T_common[0], T_common[-1], 0, n_nonramp],
#                             vmin=vlims_nonramp[0], vmax=vlims_nonramp[1], 
#                             cmap='viridis', interpolation='nearest')
#         axes[1].set_title(f'Non-Ramp ROIs (n={n_nonramp}) - Session Average, Sorted by Onset')
#         axes[1].set_ylabel('ROI Index')
        
#         # Add colorbar
#         cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.02, pad=0.02)
#         cbar2.set_label('ΔF/F')
#     else:
#         axes[1].text(0.5, 0.5, 'No non-ramp ROIs', ha='center', va='center', transform=axes[1].transAxes)
#         axes[1].set_title('Non-Ramp ROIs (n=0)')
    
#     # Common formatting
#     for ax in axes:
#         ax.axvline(0.0, color='white', linestyle='--', linewidth=1, alpha=0.8)
#         ax.grid(False)
    
#     axes[1].set_xlabel(f'Time from {trial_data_filtered.get("alignment_point", "alignment")} (s)')
    
#     plt.tight_layout()
    
#     # Store results in M for reference
#     M.setdefault('raster_analysis', {})
#     M['raster_analysis']['session_rasters'] = {
#         'time_common': T_common,
#         'ramp_mean_sorted': ramp_mean_sorted,
#         'nonramp_mean_sorted': nonramp_mean_sorted,
#         'ramp_roi_indices': ramp_roi_sorted,
#         'nonramp_roi_indices': nonramp_roi_sorted,
#         'trials_used': trials_used,
#         'alignment_point': trial_data_filtered.get('alignment_point', 'unknown'),
#         'filter_params': cfg.get('group_inspect', {})
#     }
    
#     print("=== Completed plot_ramp_vs_nonramp_session_rasters ===\n")
#     return fig














































def time_resolved_decoding(M: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sliding-window, pre-F2, time-resolved decoding of SHORT vs LONG using the F1-OFF grid in M.
    Adds results to M['decode'].

    Backward-compatible with your previous outputs; adds per-window ROI counts and (optionally)
    parallel decoding for feature='both'.
    """
    dec = dict(cfg.get('decode', {}))
    verbose       = bool(dec.get('verbose', True))
    pre_win       = tuple(map(float, dec.get('preF2_window', cfg.get('svclock', {}).get('preF2_window', (0.15, 0.70)))))
    win_sec       = float(dec.get('win_sec', 0.12))
    stride_frac   = float(dec.get('stride_frac', 0.25))
    cv_folds      = int(dec.get('cv_folds', 5))
    feature       = str(dec.get('feature', 'mean')).lower()   # 'mean' | 'slope' | 'both'
    z_in_fold     = bool(dec.get('zscore_within_fold', True))
    balance_lab   = bool(dec.get('balance_labels', True))
    min_trials    = int(dec.get('min_trials_per_window', 40))
    min_bins_roi  = int(dec.get('min_bins_per_roi', 3))
    auc_thr       = float(dec.get('auc_threshold', 0.60))
    sustain_k     = int(dec.get('sustain_bins', 3))
    restrict_to   = str(dec.get('restrict_to', 'all')).lower()
    min_per_class = dec.get('min_trials_per_class', None)
    if min_per_class is None:
        min_per_class = int(np.ceil(0.5 * max(1, min_trials)))  # conservative default

    # --- pull arrays
    X = np.asarray(M['roi_traces'], float)    # (n_trials, n_rois, Nt)
    T = np.asarray(M['time'], float)
    n_trials, n_rois, Nt = X.shape
    y_short = np.asarray(M['is_short'], bool)

    # trial selection by correctness
    is_right_trial  = np.asarray(M['is_right'], bool)
    is_right_choice = np.asarray(M['is_right_choice'], bool)
    did_not_choose  = np.asarray(M['did_not_choose'], bool)
    correct = (is_right_trial == is_right_choice) & (~did_not_choose)
    if restrict_to == 'correct':
        trial_mask = correct
    elif restrict_to == 'incorrect':
        trial_mask = (~correct) & (~did_not_choose)
    else:
        trial_mask = ~did_not_choose

    X = X[trial_mask]
    y = y_short[trial_mask]
    n_trials_kept = int(trial_mask.sum())


    # --- (NEW) optional ROI mask / weights from per-ROI screen ---
    scr = M.get('roi_screen', {})
    rule = dec.get('roi_mask_rule', {})  # e.g., {'type': 'qval', 'thresh': 0.1} or {'type': 'auc', 'thresh': 0.58}
    use_mask    = bool(dec.get('use_roi_mask', False))
    use_weights = bool(dec.get('use_roi_weights', False))

    roi_keep = None
    roi_w    = None

    if use_mask and ('auc' in scr or 'qval' in scr):
        if rule.get('type') == 'qval' and (scr.get('qval') is not None):
            q = np.asarray(scr['qval'], float)
            roi_keep = np.isfinite(q) & (q <= float(rule.get('thresh', 0.1)))
        elif rule.get('type') == 'auc' and ('auc' in scr):
            a = np.asarray(scr['auc'], float)
            roi_keep = np.isfinite(a) & (a >= float(rule.get('thresh', 0.58)))

        # apply mask globally (same ROIs for all windows)
        if roi_keep is not None and np.any(roi_keep):
            X = X[:, roi_keep, :]             # shrink feature set
            kept_roi_idx = np.where(roi_keep)[0]
        else:
            kept_roi_idx = np.arange(n_rois)

        # build weights (after masking so they align with X's ROI axis)
        if use_weights and ('auc' in scr):
            a = np.asarray(scr['auc'], float)
            a = a[kept_roi_idx]
            # simple nonnegative weight: max(0, auc-0.5); z-normalize for scale stability
            w = np.clip(a - 0.5, 0, None)
            roi_w = w / (np.nanstd(w) + 1e-9)
        else:
            roi_w = None

        # update n_rois to reflect any masking (important for later shapes/logs)
        n_rois = X.shape[1]



    # windowing inside preF2_window
    dt = float(np.nanmedian(np.diff(T))) if Nt > 1 else np.nan
    nb = max(1, int(round(win_sec / max(dt, 1e-9))))
    step = max(1, int(round(stride_frac * nb)))
    idx_pre = np.where((T >= pre_win[0]) & (T <= pre_win[1]))[0]
    if idx_pre.size == 0:
        raise ValueError("preF2_window has no coverage on M['time'].")
    starts = np.arange(idx_pre[0], idx_pre[-1] - nb + 2, step, dtype=int)
    stops  = starts + nb - 1
    centers = 0.5 * (T[starts] + T[stops])
    nW = len(starts)

    if verbose:
        print("=== time_resolved_decoding: START ===")
        print(f"[decode] trials={n_trials_kept}/{n_trials} kept (restrict_to='{restrict_to}') | rois={n_rois}, Nt={Nt}, dt={dt:.4f}s")
        print(f"[decode] preF2_window={pre_win} | win={win_sec:.3f}s stride={stride_frac:.2f}×win → {nW} windows")
        nS, nL = int(y.sum()), int((~y).sum())
        print(f"[decode] overall class counts: short={nS}, long={nL} (short frac={nS/max(1,nS+nL):.3f})")
        print(f"[decode] feature={feature}, zscore_within_fold={z_in_fold}, balance_labels={balance_lab}, "
              f"min_bins_per_roi={min_bins_roi}, min_trials_per_window={min_trials}, min_trials_per_class={min_per_class}")

    # --- helpers
    def _means_safely(vals):
        # vals: (R, nb), may contain NaNs. Compute per-ROI mean without triggering warnings.
        finite = np.isfinite(vals)
        counts = finite.sum(axis=1)                     # (R,)
        sums = np.where(finite, vals, 0.0).sum(axis=1) # (R,)
        out = np.full(vals.shape[0], np.nan, float)
        good = counts >= min_bins_roi
        out[good] = sums[good] / np.maximum(1, counts[good])
        return out

    def _slope_per_roi(vals, t_win):
        # vals: (R, nb)
        finite = np.isfinite(vals)
        out = np.full(vals.shape[0], np.nan, float)
        for r in range(vals.shape[0]):
            m = finite[r]
            if m.sum() >= min_bins_roi:
                t = t_win[m]
                yv = vals[r, m]
                t = t - t.mean()
                yv = yv - np.nanmean(yv)
                denom = float(np.dot(t, t)) + 1e-12
                out[r] = float(np.dot(t, yv) / denom)
        return out

    def _standardize_train_test(Xtr, Xte):
        mu = np.nanmean(Xtr, axis=0)
        sd = np.nanstd(Xtr, axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        # impute train/test NaNs with train means (no leakage)
        Xtr = np.where(np.isfinite(Xtr), Xtr, mu)
        Xte = np.where(np.isfinite(Xte), Xte, mu)
        return (Xtr - mu) / sd, (Xte - mu) / sd

    def _balance_indices(y_idx, rng):
        cls0 = np.where(~y_idx)[0]
        cls1 = np.where( y_idx)[0]
        if cls0.size == 0 or cls1.size == 0:
            return np.arange(y_idx.size)
        k = min(cls0.size, cls1.size)
        take0 = rng.choice(cls0, k, replace=False)
        take1 = rng.choice(cls1, k, replace=False)
        return np.sort(np.r_[take0, take1])

    def _cv_auc(F, yw, seed):
        # Drop all-NaN features (ROIs) *before* CV
        feat_mask = ~np.isnan(F).all(axis=0)
        F = F[:, feat_mask]
        n_roi_kept = int(feat_mask.sum())
        if n_roi_kept == 0:
            return np.nan, 0

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        rng = np.random.default_rng(seed)
        aucs = []
        for tr_idx, te_idx in skf.split(F, yw):
            Xtr, Xte = F[tr_idx], F[te_idx]
            ytr, yte = yw[tr_idx], yw[te_idx]

            # optional downsample on train
            if balance_lab and (np.any(ytr) and np.any(~ytr)):
                keep = _balance_indices(ytr, rng)
                Xtr, ytr = Xtr[keep], ytr[keep]

            # impute from train means and (optionally) standardize
            if z_in_fold:
                Xtr, Xte = _standardize_train_test(Xtr, Xte)
            else:
                mu = np.nanmean(Xtr, axis=0)
                Xtr = np.where(np.isfinite(Xtr), Xtr, mu)
                Xte = np.where(np.isfinite(Xte), Xte, mu)

            # drop zero-variance columns on train
            sd = np.nanstd(Xtr, axis=0)
            keep_f = sd > 1e-9
            if not np.any(keep_f):
                continue
            Xtr, Xte = Xtr[:, keep_f], Xte[:, keep_f]

            clf = LogisticRegression(solver='liblinear', penalty='l2', C=1.0, max_iter=200)
            clf.fit(Xtr, ytr)
            pr = clf.predict_proba(Xte)[:, 1]
            try:
                aucs.append(roc_auc_score(yte, pr))
            except ValueError:
                pass

        return (float(np.mean(aucs)) if aucs else np.nan), n_roi_kept

    # --- which features to run
    feats = ['mean', 'slope'] if feature == 'both' else [feature]

    # containers
    per_feat = {}
    seed = int(cfg.get('qa', {}).get('random_state', 0))

    for feat in feats:
        auc = np.full(nW, np.nan, float)
        n_roi_kept_list = np.zeros(nW, int)
        perwin_stats = dict(n_trials=[], short_counts=[], long_counts=[], frac_nan_features=[])

        for wi, (a, b) in enumerate(zip(starts, stops), start=1):
            idx = slice(a, b + 1)
            t_win = T[idx]

            # build feature matrix (n_trials_kept × n_rois), no warnings
            F = np.full((n_trials_kept, n_rois), np.nan, float)
            if feat == 'mean':
                for tr in range(n_trials_kept):
                    vals = X[tr, :, idx]  # (R, nb)
                    F[tr] = _means_safely(vals)
                    # (NEW) apply ROI weights if provided
                    if roi_w is not None:
                        F = F * roi_w[None, :]
            else:  # 'slope'
                for tr in range(n_trials_kept):
                    vals = X[tr, :, idx]
                    F[tr] = _slope_per_roi(vals, t_win)
                    # (NEW) apply ROI weights if provided
                    if roi_w is not None:
                        F = F * roi_w[None, :]

            # drop trials with all-NaN features
            good_trial = np.any(np.isfinite(F), axis=1)
            if good_trial.sum() < min_trials:
                if verbose and (wi == 1 or wi == nW or wi % 5 == 0):
                    print(f"[w{wi}/{nW}] too few trials after NaN screen: {int(good_trial.sum())} < {min_trials} → skip")
                perwin_stats['n_trials'].append(int(good_trial.sum()))
                perwin_stats['short_counts'].append(np.nan)
                perwin_stats['long_counts'].append(np.nan)
                perwin_stats['frac_nan_features'].append(1.0)
                continue

            F = F[good_trial]
            yw = y[good_trial]
            nS, nL = int(yw.sum()), int((~yw).sum())

            # per-class sufficiency guard
            if (nS < min_per_class) or (nL < min_per_class):
                if verbose and (wi == 1 or wi == nW or wi % 5 == 0):
                    print(f"[w{wi}/{nW}] SKIP (class counts) short={nS} long={nL} (need ≥{min_per_class} each)")
                perwin_stats['n_trials'].append(int(good_trial.sum()))
                perwin_stats['short_counts'].append(nS)
                perwin_stats['long_counts'].append(nL)
                perwin_stats['frac_nan_features'].append(float(np.isnan(F).sum()/F.size))
                continue

            frac_nan_feat = float(np.isnan(F).sum() / F.size)
            if verbose and (wi == 1 or wi == nW or wi % 5 == 0):
                print(f"[w{wi}/{nW}] trials={F.shape[0]} (short={nS}, long={nL}) | fracNaNfeat={frac_nan_feat:.3f}")

            # CV AUC with proper ROI handling
            auc_w, n_roi_kept = _cv_auc(F, yw, seed + wi)
            auc[wi - 1] = auc_w
            n_roi_kept_list[wi - 1] = n_roi_kept

            perwin_stats['n_trials'].append(int(F.shape[0]))
            perwin_stats['short_counts'].append(nS)
            perwin_stats['long_counts'].append(nL)
            perwin_stats['frac_nan_features'].append(frac_nan_feat)

        # divergence detection
        div_bin = -1
        if np.any(np.isfinite(auc)):
            m = np.isfinite(auc) & (auc >= auc_thr)
            run = 0
            for i, flag in enumerate(m):
                run = run + 1 if flag else 0
                if run >= sustain_k:
                    div_bin = i - sustain_k + 1
                    break
        div_time = float(centers[div_bin]) if div_bin >= 0 and centers.size else np.nan

        if verbose:
            good = np.isfinite(auc)
            if np.any(good):
                p10, p50, p90 = np.nanpercentile(auc[good], [10, 50, 90])
                print(f"[decode:{feat}] AUC p10/p50/p90: {p10:.3f}/{p50:.3f}/{p90:.3f}")
            else:
                print(f"[decode:{feat}] No valid AUC windows.")
            if div_bin >= 0:
                print(f"[decode:{feat}] Divergence: bin={div_bin} at t={div_time:.3f}s (thr={auc_thr}, sustain={sustain_k})")
            else:
                print(f"[decode:{feat}] No divergence ≥ {auc_thr} sustained {sustain_k} bins.")

        per_feat[feat] = dict(
            auc=auc.astype(float),
            n_roi_kept=n_roi_kept_list.astype(int),
            divergence_bin=int(div_bin),
            divergence_time_s=float(div_time),
            per_window_stats=perwin_stats
        )

    print("=== time_resolved_decoding: DONE ===\n")

    # pack outputs (preserve original keys; extend when feature='both')
    out = dict(
        win_centers_s=centers.astype(float),
        win_bounds_s=np.c_[T[starts], T[stops]].astype(float),
        params_used=dict(
            preF2_window=pre_win, win_sec=win_sec, stride_frac=stride_frac,
            cv_folds=cv_folds, feature=feature, zscore_within_fold=z_in_fold,
            balance_labels=balance_lab, min_trials_per_window=min_trials,
            min_trials_per_class=min_per_class, min_bins_per_roi=min_bins_roi,
            auc_threshold=auc_thr, sustain_bins=sustain_k, restrict_to=restrict_to
        )
    )

    if feature == 'both':
        out['auc_mean']  = per_feat['mean']['auc']
        out['auc_slope'] = per_feat['slope']['auc']
        out['divergence_time_s'] = dict(mean=per_feat['mean']['divergence_time_s'],
                                        slope=per_feat['slope']['divergence_time_s'])
        out['divergence_bin'] = dict(mean=per_feat['mean']['divergence_bin'],
                                     slope=per_feat['slope']['divergence_bin'])
        out['n_roi_kept'] = dict(mean=per_feat['mean']['n_roi_kept'],
                                 slope=per_feat['slope']['n_roi_kept'])
        out['per_window_stats'] = dict(mean=per_feat['mean']['per_window_stats'],
                                       slope=per_feat['slope']['per_window_stats'])
        # for backward compatibility, expose 'auc' as mean-feature by default
        out['auc'] = per_feat['mean']['auc']
    else:
        f = feats[0]
        out['auc'] = per_feat[f]['auc']
        out['divergence_bin'] = per_feat[f]['divergence_bin']
        out['divergence_time_s'] = per_feat[f]['divergence_time_s']
        out['n_roi_kept'] = per_feat[f]['n_roi_kept']
        out['per_window_stats'] = per_feat[f]['per_window_stats']

    M['decode'] = out
    M['decode']['roi_idx_used']   = kept_roi_idx.astype(int)
    M['decode']['roi_mask_rule']  = rule
    M['decode']['used_roi_mask']  = bool(roi_keep is not None and np.any(roi_keep))
    M['decode']['used_roi_weights'] = bool(roi_w is not None)
    
    return M



def roi_shortlong_screen(M, cfg):
    dec = dict(cfg.get('decode', {}))
    pre_win = tuple(map(float, dec.get('preF2_window', (0.15, 0.70))))
    feature = str(dec.get('feature', 'mean')).lower()  # 'mean'|'slope'
    min_bins = int(dec.get('min_bins_per_roi', 3))
    n_perm   = int(dec.get('n_perm', 0))               # 0 = off
    rng      = np.random.default_rng(int(cfg.get('qa', {}).get('random_state', 0)))

    X = np.asarray(M['roi_traces'], float)   # (trials, rois, Nt)
    T = np.asarray(M['time'], float)
    y = np.asarray(M['is_short'], bool)
    isi = np.asarray(M['isi'], float)

    # restrict trials (like your decoder)
    did_not_choose = np.asarray(M['did_not_choose'], bool)
    is_right_trial  = np.asarray(M['is_right'], bool)
    is_right_choice = np.asarray(M['is_right_choice'], bool)
    restrict = str(dec.get('restrict_to', 'all')).lower()
    correct = (is_right_trial == is_right_choice) & (~did_not_choose)
    if restrict == 'correct':
        keep_tr = correct
    elif restrict == 'incorrect':
        keep_tr = (~correct) & (~did_not_choose)
    else:
        keep_tr = ~did_not_choose

    X, y, isi = X[keep_tr], y[keep_tr], isi[keep_tr]

    # window indices
    mwin = (T >= pre_win[0]) & (T <= pre_win[1])
    t_win = T[mwin]
    R = X.shape[1]

    def feat_mean(vals):  # vals: (nb,)
        m = np.isfinite(vals); 
        return np.nan if m.sum()<min_bins else float(np.nanmean(vals[m]))
    def feat_slope(vals):
        m = np.isfinite(vals)
        if m.sum()<min_bins: return np.nan
        tt = t_win[m] - np.nanmean(t_win[m]); vv = vals[m] - np.nanmean(vals[m])
        return float(np.dot(tt, vv) / (np.dot(tt, tt) + 1e-12))

    F = np.full((X.shape[0], R), np.nan, float)
    if feature == 'slope':
        for tr in range(X.shape[0]):
            vals = _roi_time(X[tr, :, mwin], R)
            for r in range(R):
                F[tr, r] = feat_slope(X[tr, r, mwin])
    else:
        for tr in range(X.shape[0]):
            # vals = X[tr, :, mwin]                         # (R, nb)
            vals = _roi_time(X[tr, :, mwin], R)
            finite = np.isfinite(vals)
            cnt = finite.sum(axis=1)
            mu = np.where(finite, vals, 0.0).sum(axis=1) / np.maximum(1, cnt)
            mu[cnt < min_bins] = np.nan
            F[tr] = mu

    # per‑ROI univariate AUC + Cohen's d
    auc = np.full(R, np.nan); d = np.full(R, np.nan)
    for r in range(R):
        col = F[:, r]; m = np.isfinite(col)
        if m.sum()<20 or len(np.unique(y[m]))<2: continue
        x = col[m]; yy = y[m].astype(int)
        # AUC via rank-sum
        order = np.argsort(x); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, m.sum()+1)
        n1, n0 = int(yy.sum()), int((~yy.astype(bool)).sum())
        if n1==0 or n0==0: continue
        U1 = ranks[yy==1].sum() - n1*(n1+1)/2
        auc[r] = U1 / (n1*n0)
        # Cohen's d
        d[r] = (np.nanmean(x[yy==1]) - np.nanmean(x[yy==0])) / (np.nanstd(x) + 1e-12)

    # optional within‑ISI permutation p‑values
    pval = np.full(R, np.nan)
    if n_perm > 0:
        levels = np.asarray(M.get('isi_allowed', np.unique(np.round(isi,6))), float)
        lvl_idx = np.argmin(np.abs(isi[:, None] - levels[None, :]), axis=1)
        for r in range(R):
            if not np.isfinite(auc[r]): continue
            col = F[:, r]; m = np.isfinite(col)
            if m.sum()<20: continue
            x = col[m]; yy = y[m].astype(int); li = lvl_idx[m]
            # real auc
            order = np.argsort(x); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, x.size+1)
            n1 = int(yy.sum()); n0 = int((~yy.astype(bool)).sum())
            U1 = ranks[yy==1].sum() - n1*(n1+1)/2
            auc_real = U1 / max(1, n1*n0)
            # null
            nulls = []
            for _ in range(n_perm):
                yy_p = yy.copy()
                for liu in np.unique(li):
                    ix = np.where(li==liu)[0]
                    if ix.size>=2: yy_p[ix] = yy_p[ix][rng.permutation(ix.size)]
                order = np.argsort(x); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, x.size+1)
                n1 = int(yy_p.sum()); n0 = int((~yy_p.astype(bool)).sum())
                if n1==0 or n0==0: continue
                U1 = ranks[yy_p==1].sum() - n1*(n1+1)/2
                nulls.append(U1 / (n1*n0))
            if nulls:
                nulls = np.array(nulls, float)
                pval[r] = (1 + (nulls >= auc_real).sum()) / (len(nulls) + 1)

    # simple BH-FDR if pvals exist
    qval = None
    if np.isfinite(pval).any():
        pv = pval.copy(); order = np.argsort(np.where(np.isfinite(pv), pv, 1.0))
        qval = np.full_like(pv, np.nan)
        m = np.isfinite(pv).sum(); best = 1.0
        for rank, idx in enumerate(order, start=1):
            if not np.isfinite(pv[idx]): continue
            q = pv[idx] * m / rank
            best = min(best, q)
            qval[idx] = best

    M['roi_screen'] = dict(
        feature=feature, window=pre_win,
        auc=auc, cohend=d, pval=pval, qval=qval
    )
    return M

def _roi_time(vals, R):
    vals = np.asarray(vals)
    if vals.shape[0] == R:          # (R, nb)
        return vals
    if vals.shape[1] == R:          # (nb, R) -> transpose
        return vals.T
    raise ValueError(f"Need ROI axis length {R}, got {vals.shape}")






























def tag_cs_likeness(M: dict, cfg: dict) -> dict:
    """
    Detect CF-like 'complex spike' (CS) events and tag ROIs accordingly.
    Uses only grids already present in M (no aliasing). Adds results to M['cs'].

    Key changes vs prior version:
      - Operates on already-preprocessed traces (your dF/F have been baseline-corrected,
        per-ROI z-normalized, and SavGol smoothed).
      - Default detection is in ABSOLUTE units after control-median centering
        (mode='absolute'), i.e., *no additional z-score*.
      - Optional mode='local_z' computes a control-only robust z (median/MAD) per trial×ROI.
      - Polarity control: direction in {'positive','negative','both'} (default 'positive').
      - Wider default max width to accommodate dendritic Ca2+ (default 0.6 s).
      - Rich QA prints: control MAD percentiles, peri peak amplitude percentiles, etc.

    Outputs (unchanged structure):
      M['cs'] = {
        'roi_is_cs_like': (n_rois,) bool,
        'metrics': { '<ALIGN>': {...} },
        'event_flags': { '<ALIGN>': (n_trials, n_rois) bool }   # if enabled
        'params_used': {...}
      }
    """
    cs_cfg = deepcopy(cfg.get('cs', {}))

    # ------- defaults tailored to your preprocessing -------
    aligns = list(cs_cfg.get('aligns', ['F1OFF','F2','CHOICE']))  # must be built in M
    peri_window = cs_cfg.get('peri_window', dict(F2=(0.00,0.25), CHOICE=(0.00,0.25), F1OFF=(0.00,0.25)))
    ctrl_window = cs_cfg.get('ctrl_window', dict(F2=(-0.25,0.00), CHOICE=(-0.25,0.00), F1OFF=(-0.25,0.00)))

    # Preprocessing (kept minimal; you already SavGol'd)
    detrend = str(cs_cfg.get('detrend', 'median')).lower()          # 'none'|'median' (centers using union if used)
    smooth_sigma_bins = float(cs_cfg.get('smooth_sigma_bins', 0.0)) # 0 → OFF

    # Detection mode / thresholds
    mode = str(cs_cfg.get('mode', 'absolute')).lower()              # 'absolute'|'local_z'
    direction = str(cs_cfg.get('direction', 'positive')).lower()    # 'positive'|'negative'|'both'

    amp_thresh_abs = float(cs_cfg.get('amp_thresh_abs', 1.8))       # used when mode='absolute'
    amp_thresh_z   = float(cs_cfg.get('amp_thresh_z', 2.5))         # used when mode='local_z'

    min_width_bins = int(cs_cfg.get('min_width_bins', 1))
    max_width_bins = int(cs_cfg.get('max_width_bins', 999999))      # we gate by seconds primarily
    max_width_s    = float(cs_cfg.get('max_width_s', 0.60))
    refractory_s   = float(cs_cfg.get('refractory_s', 0.08))

    # Optional slope gate (OFF by default)
    use_slope_gate   = bool(cs_cfg.get('use_slope_gate', False))
    slope_window_s   = float(cs_cfg.get('slope_window_s', 0.12))
    slope_min_per_s  = float(cs_cfg.get('slope_min_per_s', 0.0))

    # Tag rule (session-level)
    min_event_rate        = float(cs_cfg.get('min_event_rate', 0.08))  # fraction of trials with a peri event
    min_rate_ratio        = float(cs_cfg.get('min_rate_ratio', 1.5))   # peri / ctrl
    min_trials_with_event = int(cs_cfg.get('min_trials_with_event', 3))
    merge_rule            = str(cs_cfg.get('merge_rule', 'any_align')).lower()  # 'any_align'|'all_aligns'

    # QA / prints
    save_event_flags = bool(cs_cfg.get('save_event_flags', False))
    print_top_k      = int(cs_cfg.get('print_top_k', 10))
    verbose          = bool(cs_cfg.get('verbose', True))

    # ------- map alignment names to M keys (no aliasing) -------
    REG = {
        'F1OFF': dict(X='roi_traces',           T='time',          label='F1OFF'),
        'F2':    dict(X='roi_traces_F2locked',  T='time_F2locked', label='F2'),
        'CHOICE':dict(X='roi_traces_choice',    T='time_choice',   label='CHOICE'),
    }

    n_trials = int(M['n_trials'])
    n_rois   = int(M['n_rois'])

    # ------- helpers -------
    def _get_mask(T, win):
        w0, w1 = float(win[0]), float(win[1])
        if w1 < w0: w0, w1 = w1, w0
        return (T >= w0) & (T <= w1)

    def _nan_gauss_smooth(y, sigma_bins):
        if not np.isfinite(sigma_bins) or sigma_bins <= 0:
            return y
        y = np.asarray(y, float)
        rad = int(np.ceil(4.0 * sigma_bins))
        if rad < 1: return y
        x = np.arange(-rad, rad+1, dtype=float)
        k = np.exp(-0.5*(x/sigma_bins)**2); k /= k.sum()
        mask = np.isfinite(y).astype(float)
        y0   = np.where(np.isfinite(y), y, 0.0)
        num  = np.convolve(y0, k, mode='same')
        den  = np.convolve(mask, k, mode='same')
        out  = num / np.maximum(den, 1e-12)
        out[(den < 1e-12)] = np.nan
        return out

    def _mad(x):
        med = np.nanmedian(x)
        return np.nanmedian(np.abs(x - med))

    def _find_events(S, mask_win, dt, thr, minw, maxw, refr_s, polarity='pos'):
        """
        Generic threshold crossing on S (already centered & optionally normalized).
        polarity: 'pos' => S>=thr; 'neg' => (-S)>=thr (i.e., S<=-thr).
        """
        if S is None: return []
        S = np.asarray(S, float)
        ok = np.isfinite(S) & mask_win
        if not ok.any(): return []
        if polarity == 'pos':
            above = (S >= thr) & ok
        else:
            above = ((-S) >= thr) & ok

        if not above.any(): return []

        # contiguous runs
        starts = np.where(np.diff(np.r_[False, above]) == 1)[0]
        stops  = np.where(np.diff(np.r_[above, False]) == -1)[0] - 1

        events = []
        maxw_bins_from_s = int(np.floor(maxw / max(dt, 1e-12))) if np.isfinite(dt) else maxw
        maxw_bins_final  = min(max_width_bins, maxw_bins_from_s) if max_width_s > 0 else max_width_bins

        for s, e in zip(starts, stops):
            w = e - s + 1
            if w < minw or w > maxw_bins_final:
                continue
            seg = S[s:e+1]
            pk_rel = int(np.nanargmax(seg if polarity=='pos' else -seg))
            pk = s + pk_rel
            amp = float(seg[pk_rel]) if polarity=='pos' else float(-seg[pk_rel])
            events.append({'start': s, 'end': e, 'peak': pk, 'amp': amp, 'width_bins': w, 'polarity': (1 if polarity=='pos' else -1)})

        if not events: return []

        # refractory (greedy keep largest amp)
        events.sort(key=lambda d: d['amp'], reverse=True)
        keep = []
        used = np.zeros(len(S), dtype=bool)
        refr_bins = int(np.round(refr_s / max(dt, 1e-12))) if np.isfinite(dt) else 0
        for ev in events:
            p = ev['peak']
            a = max(0, p - refr_bins)
            b = min(len(S)-1, p + refr_bins)
            if not used[a:b+1].any():
                keep.append(ev)
                used[a:b+1] = True
        keep.sort(key=lambda d: d['peak'])
        return keep

    # ------- resolve requested aligns and sanity-check presence -------
    align_list = []
    for name in aligns:
        u = name.upper()
        if u not in REG:
            raise ValueError(f"[cs] Unknown align '{name}'. Allowed: {list(REG.keys())}.")
        Xk, Tk = REG[u]['X'], REG[u]['T']
        if Xk not in M or Tk not in M:
            raise KeyError(f"[cs:{u}] Grid '{Xk}'/'{Tk}' not present in M. Build it first.")
        align_list.append(u)

    if verbose:
        print("=== tag_cs_likeness: START ===")
        print(f"[cs] aligns to analyze: {align_list}")
        print(f"[cs] smoothing sigma (bins): {smooth_sigma_bins} (0 means OFF)")
        print(f"[cs] mode={mode}, direction={direction}")
        if mode == 'absolute':
            print(f"[cs] abs-threshold={amp_thresh_abs:.3f} (after control-median centering)")
        else:
            print(f"[cs] local-z threshold={amp_thresh_z:.3f} (control-only robust z)")
        print(f"[cs] min/max width (bins): {min_width_bins}/{max_width_bins}, max_width_s={max_width_s:.3f}, refractory={refractory_s:.3f}s")

    eps = 1e-9
    metrics_by_align = {}
    flags_by_align = {}
    roi_votes = np.zeros(n_rois, dtype=int)

    for u in align_list:
        Xk, Tk = REG[u]['X'], REG[u]['T']
        X = np.asarray(M[Xk], float)     # (n_trials, n_rois, Nt)
        T = np.asarray(M[Tk], float)     # (Nt,)
        Nt = T.size
        dt = float(np.nanmedian(np.diff(T))) if Nt > 1 else np.nan

        if u not in peri_window or u not in ctrl_window:
            raise KeyError(f"[cs:{u}] peri_window/ctrl_window missing in cfg['cs'].")

        peri_mask = _get_mask(T, peri_window[u])
        ctrl_mask = _get_mask(T, ctrl_window[u])
        union_mask = peri_mask | ctrl_mask

        # metrics
        event_rate_peri = np.zeros(n_rois, float)
        event_rate_ctrl = np.zeros(n_rois, float)
        rate_ratio      = np.zeros(n_rois, float)
        mean_amp        = np.full(n_rois, np.nan, float)   # in chosen units (abs or local_z)
        mean_width_s    = np.full(n_rois, np.nan, float)
        mean_iei_s      = np.full(n_rois, np.nan, float)
        n_trials_ev     = np.zeros(n_rois, int)
        n_events_total  = np.zeros(n_rois, int)

        if save_event_flags:
            flags = np.zeros((n_trials, n_rois), dtype=bool)
        else:
            flags = None

        # QA accumulators
        ctrl_mads = []      # per trial×roi
        peri_peaks = []     # max amplitude in peri window per trial×roi (chosen units)
        missing_ctrl_count = 0

        amps_all = [[] for _ in range(n_rois)]
        widths_all = [[] for _ in range(n_rois)]
        ieis_all = [[] for _ in range(n_rois)]

        for tr in range(n_trials):
            for r in range(n_rois):
                y = X[tr, r]            # length Nt
                if not np.isfinite(y[union_mask]).any():
                    continue

                y_proc = y.copy()
                if detrend == 'median':
                    base_union = y_proc[union_mask & np.isfinite(y_proc)]
                    if base_union.size >= 3:
                        y_proc = y_proc - np.nanmedian(base_union)

                if smooth_sigma_bins > 0:
                    y_proc = _nan_gauss_smooth(y_proc, smooth_sigma_bins)

                # Control-only baseline + (optional) local-z
                ctrl_vals = y_proc[ctrl_mask & np.isfinite(y_proc)]
                if ctrl_vals.size >= 3:
                    med_ctrl = np.nanmedian(ctrl_vals)
                    mad_ctrl = _mad(ctrl_vals)
                    sd_ctrl  = 1.4826*mad_ctrl if mad_ctrl > 0 else np.nanstd(ctrl_vals)
                    centered = y_proc - med_ctrl
                    ctrl_mads.append(1.4826*mad_ctrl if mad_ctrl > 0 else (np.nanstd(ctrl_vals) + eps))
                else:
                    # No control available — fall back to no centering; count for QA
                    centered = y_proc
                    sd_ctrl  = np.nan
                    missing_ctrl_count += 1

                if mode == 'absolute':
                    S = centered
                    thr = amp_thresh_abs
                else:  # 'local_z'
                    if not np.isfinite(sd_ctrl) or sd_ctrl < eps:
                        # Cannot compute local z reliably; skip detection for this trial×roi
                        S = None
                    else:
                        S = centered / sd_ctrl
                    thr = amp_thresh_z

                # record peri peak amplitude (for QA percentiles)
                if S is not None and np.isfinite(S[peri_mask]).any():
                    sp = S[peri_mask & np.isfinite(S)]
                    if sp.size:
                        peri_peaks.append(np.nanmax(np.abs(sp)) if direction == 'both'
                                          else (np.nanmax(sp) if direction=='positive' else np.nanmax(-sp)))

                # detect events peri/ctrl
                def detect_in_window(which_mask):
                    evs = []
                    if S is None: 
                        return evs
                    if direction in ('positive','both'):
                        evs += _find_events(S, which_mask, dt, thr, min_width_bins, max_width_bins, refractory_s, polarity='pos')
                    if direction in ('negative','both'):
                        evs += _find_events(S, which_mask, dt, thr, min_width_bins, max_width_bins, refractory_s, polarity='neg')
                    # sort by peak time (they are already refractory-pruned per polarity)
                    evs.sort(key=lambda d: d['peak'])
                    return evs

                ev_peri = detect_in_window(peri_mask)
                ev_ctrl = detect_in_window(ctrl_mask)

                if ev_peri:
                    n_trials_ev[r] += 1
                    if flags is not None:
                        flags[tr, r] = True
                    for ev in ev_peri:
                        amps_all[r].append(ev['amp'])
                        widths_all[r].append(ev['width_bins'] * dt if np.isfinite(dt) else np.nan)
                    if len(ev_peri) >= 2 and np.isfinite(dt):
                        peaks = np.array([ev['peak'] for ev in ev_peri], int)
                        iei = np.diff(peaks) * dt
                        ieis_all[r].extend(iei.tolist())

                n_events_total[r] += (len(ev_peri) + len(ev_ctrl))
                if len(ev_peri) > 0:
                    event_rate_peri[r] += 1.0
                if len(ev_ctrl) > 0:
                    event_rate_ctrl[r] += 1.0

        # finalize metrics per ROI
        event_rate_peri /= max(1, n_trials)
        event_rate_ctrl /= max(1, n_trials)
        rate_ratio = (event_rate_peri + eps) / (event_rate_ctrl + eps)

        for r in range(n_rois):
            if amps_all[r]:
                mean_amp[r] = float(np.nanmean(amps_all[r]))
            if widths_all[r]:
                mean_width_s[r] = float(np.nanmean(widths_all[r]))
            if ieis_all[r]:
                mean_iei_s[r] = float(np.nanmean(ieis_all[r]))

        metrics_by_align[u] = dict(
            event_rate_peri=event_rate_peri,
            event_rate_ctrl=event_rate_ctrl,
            rate_ratio=rate_ratio,
            mean_amp=mean_amp,
            mean_width_s=mean_width_s,
            mean_iei_s=mean_iei_s,
            n_trials_with_event=n_trials_ev.astype(int),
            n_events=n_events_total.astype(int),
        )
        if save_event_flags:
            flags_by_align[u] = flags

        # session-level pass/fail per ROI for this alignment
        passes = (
            (event_rate_peri >= min_event_rate) &
            (rate_ratio      >= min_rate_ratio) &
            (np.nan_to_num(mean_width_s, nan=np.inf) <= (max_width_s if max_width_s > 0 else np.inf)) &
            (np.nan_to_num(mean_iei_s,  nan=np.inf) >= refractory_s) &
            (n_trials_ev >= min_trials_with_event)
        )
        roi_votes += passes.astype(int)

        # ------- QA prints for this alignment -------
        if verbose:
            # Ctrl MAD percentiles (noise scale)
            if ctrl_mads:
                cm = np.array(ctrl_mads, float)
                p10, p50, p90 = np.nanpercentile(cm, [10,50,90])
                print(f"[cs:{u}] ctrl MAD (robust SD per trial×ROI) p10/p50/p90: {p10:.3f}/{p50:.3f}/{p90:.3f}")
            else:
                print(f"[cs:{u}] ctrl MAD: no control samples available.")

            # Peri peak amplitude percentiles (chosen units)
            if peri_peaks:
                pp = np.array(peri_peaks, float)
                p50, p90, p95, p99 = np.nanpercentile(pp, [50,90,95,99])
                unit = "abs-units" if mode=='absolute' else "local-z"
                print(f"[cs:{u}] peri peak ({unit}) p50/p90/p95/p99: {p50:.3f}/{p90:.3f}/{p95:.3f}/{p99:.3f}")
            else:
                print(f"[cs:{u}] peri peak: no measurable peri samples.")

            if missing_ctrl_count:
                print(f"[cs:{u}] trials without control baseline: {missing_ctrl_count} (of {n_trials*n_rois})")

            print(f"[cs:{u}] mean(event_rate_peri)={np.nanmean(event_rate_peri):.3f}, "
                  f"mean(ctrl)={np.nanmean(event_rate_ctrl):.3f}, mean(ratio)={np.nanmean(rate_ratio):.2f}")

            # top-k by peri rate
            top = np.argsort(-event_rate_peri)[:max(1, min(print_top_k, n_rois))]
            txt = ", ".join([f"r{int(r)}:{event_rate_peri[r]:.2f}" for r in top])
            print(f"[cs:{u}] top by peri rate: {txt}")
            # top-k by ratio
            top2 = np.argsort(-rate_ratio)[:max(1, min(print_top_k, n_rois))]
            txt2 = ", ".join([f"r{int(r)}:{rate_ratio[r]:.2f}" for r in top2])
            print(f"[cs:{u}] top by rate ratio: {txt2}")

    # Merge across alignments
    if merge_rule == 'all_aligns':
        roi_is_cs_like = (roi_votes == len(align_list))
    else:
        roi_is_cs_like = (roi_votes >= 1)

    if verbose:
        n_cs = int(np.sum(roi_is_cs_like))
        print(f"[cs] merge_rule={merge_rule} → CS-like ROIs: {n_cs}/{n_rois} ({100.0*n_cs/max(1,n_rois):.1f}%)")
        print("=== tag_cs_likeness: DONE ===\n")

    # Pack outputs
    out = dict(
        roi_is_cs_like=roi_is_cs_like.astype(bool),
        metrics=metrics_by_align,
        params_used=dict(
            aligns=align_list,
            peri_window=peri_window,
            ctrl_window=ctrl_window,
            detrend=detrend,
            smooth_sigma_bins=smooth_sigma_bins,
            mode=mode,
            direction=direction,
            amp_thresh_abs=amp_thresh_abs,
            amp_thresh_z=amp_thresh_z,
            min_width_bins=min_width_bins,
            max_width_bins=max_width_bins,
            max_width_s=max_width_s,
            refractory_s=refractory_s,
            use_slope_gate=use_slope_gate,
            slope_window_s=slope_window_s,
            slope_min_per_s=slope_min_per_s,
            min_event_rate=min_event_rate,
            min_rate_ratio=min_rate_ratio,
            min_trials_with_event=min_trials_with_event,
            merge_rule=merge_rule
        )
    )
    if save_event_flags and flags_by_align:
        out['event_flags'] = flags_by_align

    M['cs'] = out
    return M



def plot_interpolation_check(
    M, trial_data, cfg,
    align='F1OFF',
    trial_idx=0,
    roi_idx=0,
    show_mask_events=True,
    save_path=None,
    equal_ylim=True, 
    ylim_mode='percentile',
    ylim_percentiles=(2, 98), ylim_pad_frac=0.05
):
    """
    QA plot: raw dF/F trace (aligned) vs interpolated grid output for a given
    trial/ROI. Supports align in {'F1OFF','F2','CHOICE'}.

    * No aliasing: CHOICE uses 'choice_start' only.
    * F1OFF additionally checks that t >= ISI are NaN.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    align = str(align).upper()
    df = trial_data['df_trials_with_segments']

    # ---- guardrails
    n_trials = len(df)
    if not (0 <= trial_idx < n_trials):
        raise IndexError(f"trial_idx {trial_idx} out of range [0,{n_trials-1}]")
    n_rois = int(M['n_rois'])
    if not (0 <= roi_idx < n_rois):
        raise IndexError(f"roi_idx {roi_idx} out of range [0,{n_rois-1}]")

    row = df.iloc[trial_idx]

    # ---- select alignment
    if align == 'F1OFF':
        # grid from M
        T = np.asarray(M['time'], float)
        y_grid = np.asarray(M['roi_traces'], float)[trial_idx, roi_idx]
        # raw aligned to end of F1
        ref = float(row['end_flash_1'])
        xlab = 'Time from F1-OFF (s)'
        title = f'Raw vs Interpolated | trial={trial_idx}, ROI={roi_idx} (F1-OFF aligned)'
        # for masking check
        isi_this = float(M['isi'][trial_idx]) if 'isi' in M else np.nan
        event_label = 'ISI'
        event_x = isi_this
    elif align == 'F2':
        if ('roi_traces_F2locked' not in M) or ('time_F2locked' not in M):
            raise KeyError("F2-locked grid not present in M. Build it first.")
        T = np.asarray(M['time_F2locked'], float)
        y_grid = np.asarray(M['roi_traces_F2locked'], float)[trial_idx, roi_idx]
        ref = float(row['start_flash_2'])
        xlab = 'Time from F2 (s)'
        title = f'Raw vs Interpolated | trial={trial_idx}, ROI={roi_idx} (F2-locked)'
        isi_this = np.nan
        event_label = 'F2'
        event_x = 0.0
    elif align == 'CHOICE':
        if ('roi_traces_choice' not in M) or ('time_choice' not in M):
            raise KeyError("Choice-locked grid not present in M. Build it first.")
        T = np.asarray(M['time_choice'], float)
        y_grid = np.asarray(M['roi_traces_choice'], float)[trial_idx, roi_idx]
        # strictly use choice_start (no aliasing to servo)
        if row.get('choice_start') is None or not np.isfinite(row.get('choice_start')):
            print("[QA] choice_start is NaN for this trial; plot will still render.")
        ref = float(row.get('choice_start', np.nan))
        xlab = 'Time from choice start (s)'
        title = f'Raw vs Interpolated | trial={trial_idx}, ROI={roi_idx} (choice-locked)'
        isi_this = np.nan
        event_label = 'choice'
        event_x = 0.0
    else:
        raise ValueError("align must be one of {'F1OFF','F2','CHOICE'}")

    # ---- raw trace aligned to the selected reference
    t_raw_all = np.asarray(row['dff_time_vector'], float)
    y_raw_all = np.asarray(row['dff_segment'], float)[roi_idx]
    if not np.all(np.diff(t_raw_all) > 0):
        order = np.argsort(t_raw_all)
        t_raw_all = t_raw_all[order]
        y_raw_all = y_raw_all[order]
        print("[QA] Raw time for this trial was non-monotonic → sorted for display.")

    t_raw = t_raw_all - ref
    y_raw = y_raw_all

    # ---- prints
    print(f"=== Interpolation QA ({align}) ===")
    print(f" trial={trial_idx} / ROI={roi_idx}")
    print(f" raw  time range: {np.nanmin(t_raw):.4f} .. {np.nanmax(t_raw):.4f} s  (N={t_raw.size})")
    print(f" grid time range: {T[0]:.4f} .. {T[-1]:.4f} s  (Nt={T.size})")
    if np.isfinite(isi_this):
        print(f" trial ISI (s):  {isi_this:.4f}")

    # ---- overlap & RMSE (robust to NaNs in grid)
    finite_grid = np.isfinite(y_grid)
    y_fin = y_grid[finite_grid]                         # finite grid values (may be empty)
    T_fin = T[finite_grid]
    # raw points inside the plotted grid domain
    m_raw_plot = (t_raw >= T[0]) & (t_raw <= T[-1]) & np.isfinite(y_raw)

    rmse_msg = "[QA] No overlap between raw & grid (or insufficient finite points) → skip RMSE."
    if finite_grid.sum() >= 2 and m_raw_plot.any():
        y_pred = np.interp(t_raw[m_raw_plot], T_fin, y_fin, left=np.nan, right=np.nan)
        y_ref  = y_raw[m_raw_plot]
        m_cmp  = np.isfinite(y_pred) & np.isfinite(y_ref)
        if m_cmp.sum() >= 3:
            diff = y_pred[m_cmp] - y_ref[m_cmp]
            rmse = float(np.sqrt(np.mean(diff**2)))
            nrmse = float(rmse / (np.std(y_ref[m_cmp]) + 1e-12))
            rmse_msg = f"[QA] RMSE={rmse:.5f}, NRMSE={nrmse:.5f} (N={int(m_cmp.sum())} pts)"
    print(rmse_msg)

    # ---- F1OFF-only masking check (t >= ISI must be NaN)
    if align == 'F1OFF' and np.isfinite(isi_this):
        mask_after = T >= isi_this
        n_after = int(mask_after.sum())
        n_after_nan = int(np.isnan(y_grid[mask_after]).sum())
        frac_nan = (n_after_nan / max(n_after, 1)) if n_after else np.nan
        print(f"[QA] Post-ISI bins: {n_after} | NaNs after ISI: {n_after_nan} ({frac_nan*100:.1f}%)")
        if n_after and (n_after_nan < n_after):
            print(">>> [WARN] Some bins at/after ISI are not NaN for this trial.")

    # ---- plotting
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                         gridspec_kw={'height_ratios':[1.2, 1.0]})
    fig.suptitle(title)

    # top: raw (restricted to grid domain for visual alignment)
    ax_top.plot(t_raw, y_raw, lw=1.5, marker='.', ms=3, alpha=0.9)
    ax_top.set_ylabel('ΔF/F (raw)')

    # bottom: grid
    ax_bot.plot(T, y_grid, lw=1.8)
    ax_bot.set_xlabel(xlab)
    ax_bot.set_ylabel('ΔF/F (grid)')

    # Consistent x-limits = grid domain
    ax_top.set_xlim(T[0], T[-1])

    # Equal y-limits (robust)
    if equal_ylim:
        vals = []
        if np.any(np.isfinite(y_raw[m_raw_plot])): vals.append(y_raw[m_raw_plot][np.isfinite(y_raw[m_raw_plot])])
        if np.any(np.isfinite(y_grid)):            vals.append(y_grid[np.isfinite(y_grid)])
        if vals:
            v = np.concatenate(vals)
            if ylim_mode == 'percentile':
                lo, hi = np.nanpercentile(v, ylim_percentiles)
            else:
                lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
            pad = (hi - lo) * float(ylim_pad_frac)
            ylo, yhi = lo - pad, hi + pad
            ax_top.set_ylim(ylo, yhi); ax_bot.set_ylim(ylo, yhi)
            print(f"[QA] shared y-limits set to [{ylo:.3f}, {yhi:.3f}] (mode={ylim_mode})")

    # Visual aids
    if show_mask_events:
        for ax in (ax_top, ax_bot):
            ax.axvline(0.0, color='k', lw=1.0, ls='--', alpha=0.6)
        # label the event line
        if np.isfinite(event_x):
            for ax in (ax_top, ax_bot):
                ax.axvline(event_x, color='C3', lw=1.0, ls=':', alpha=0.8)
                ax.text(event_x, ax.get_ylim()[1], f' {event_label}', color='C3',
                        va='top', ha='left', fontsize=8)
        # Shade NaN zones in the grid (bottom)
        nan_mask = ~np.isfinite(y_grid)
        if np.any(nan_mask):
            # spans from mask
            starts = np.where(np.diff(np.r_[0, nan_mask.astype(int)]) == 1)[0]
            stops  = np.where(np.diff(np.r_[nan_mask.astype(int), 0]) == -1)[0] - 1
            for s, e in zip(starts, stops):
                ax_bot.axvspan(T[s], T[e], color='0.9', alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_interpolation_check_for_all(M, trial_data, cfg, trial_idx=0, roi_idx=0):
    """
    Convenience wrapper: render QA plots for all alignments present in M.
    Returns list of figures (order: F1OFF, F2, CHOICE when available).
    """
    figs = []
    if 'roi_traces' in M and 'time' in M:
        figs.append(plot_interpolation_check(M, trial_data, cfg, 'F1OFF', trial_idx, roi_idx))
    if 'roi_traces_F2locked' in M and 'time_F2locked' in M:
        figs.append(plot_interpolation_check(M, trial_data, cfg, 'F2', trial_idx, roi_idx))
    if 'roi_traces_choice' in M and 'time_choice' in M:
        figs.append(plot_interpolation_check(M, trial_data, cfg, 'CHOICE', trial_idx, roi_idx))
    if not figs:
        print("[QA] No interpolation grids found in M.")
    return figs

if __name__ == '__main__':
    print('Import and call run_page1_reports(...) or individual functions.')



        
        
        
# %%



cfg = {
  # --- labels (must come from trial_data.session_info; no guessing) ---
  "labels": {
    "require_sets": True,           # require short_isis & long_isis + unique_isis
    "tolerance_sec": 1e-3           # match tolerance when mapping observed ISIs to session sets
  },

  # --- aligned grids to build (only those with build=True will be created) ---
  "grids": {
    "f1off": {                      # F1-OFF locked, strictly pre-F2
      "build": True,
      "bins": 240,
      "tmax": "max_isi",            # "max_isi" or a numeric (seconds)
      "pre_seconds": 0.30,          # NEW: give us baseline
      "dt_sec": 0.005,              # NEW: keep 5 ms resolution
      "mask_preF2": True            # must be True for this grid
    },
    "f2": {                         # F2-onset locked
      "build": True,
      "window": (-0.2, 0.6),
      "bins": 160
    },
    "choice": {                     # choice_start locked (NOT aliased to servo)
      "build": True,
      "window": (-0.6, 0.6),
      "bins": 240
    },
    "servo_in": {                   # servo_in locked (independent of choice)
      "build": False,
      "window": (-0.6, 0.6),
      "bins": 240
    },
    "servo_out": {                  # servo_out locked
      "build": False,
      "window": (-0.6, 0.6),
      "bins": 240
    }
  },

  # --- analyses (math knobs only; no plotting here) ---
  "analyses": {
    "svclock": {                    # scaling vs clock
      "cv_folds": 5,
      "knots_phase": 8,
      "knots_time": 8,
      "delta_r2_eps": 0.02,
      "windowed": {                 # transient-free pre-F2 window (applies to “windowed” ΔR²)
        "use": True,
        "window": (0.15, 0.70)
      }
    },

    "decode_time_resolved": {       # pre-F2 time-resolved decode from F1-OFF grid
      "bins": 20,
      "divergence_auc_threshold": 0.6,
      "divergence_sustain_bins": 3
    },

    "fair_window": {                # fair-window pre-F2 decoders (amp/slope)
      "tmax": 0.7,
      "bins": 12,
      "use_slope": True
    },

    "rsa": {
      "phase_bins": 20,
      "abs_time_window": (0.15, 0.70)  # absolute-time RSA uses this pre-F2 subwindow
    },

    "hazard": {
      "kernel_s": 0.12,             # width of discrete hazard bumps (s)
      "roi_positive_thresh": 0.0    # threshold to call per-ROI partial R² “positive”
    },

    "trough": {
      "smooth_w": 7
    },

    # choice decoders (kept separate; you can enable one or both)
    "choice_decode_choice_locked": {
      "enabled": True,
      "bins": 24,
      "baseline_window": (-0.6, -0.2),
      "restrict_pre": True,
      "zscore_within_fold": True,
      "stratify_by_isi": True
    },

    "choice_decode_f2_locked": {
      "enabled": False,
      "bins": 20,
      "baseline_window": (-0.2, -0.05),
      "restrict_preF2": False,
      "zscore_within_fold": True
    }
  },

  # --- plotting defaults (used by report & group-trace helpers) ---
  "plots": {
    "groups": {
      "show_sem": True,
      "max_spaghetti": 10,
      "ylim_mode": "auto_percentile",        # "auto_percentile" or "tight"
      "ylim_percentiles": [2, 98],
      "ylim_pad_frac": 0.10
    },
    "raster": {
      "vmin_vmax_percentiles": [5, 95],
      "show_isi_guides": True
    }
  },

  # --- QA / logging ---
  "qa": {
    "verbose": True,
    "random_state": 0,
    "print_per_trial_snapshots": True,
    "print_summary": True,
    "error_on_missing_alignment": True  # if True, raise when any trial lacks the required align time
  }
}



cfg['svclock'] = dict(
    # analysis window on the F1-OFF grid (strictly pre-F2 content in M)
    # preF2_window=(0.15, 0.70),
    preF2_window=(0.15, 0.70),

    # sliding-window geometry
    # win_sec=0.12,          # window width (s)
    win_sec=0.08,          # window width (s)
    stride_frac=0.25,      # step as fraction of window width (0.25 => 75% overlap)

    # model / basis
    cv_folds=5,            # CV folds
    knots_time=8,          # #knots for absolute-time spline
    knots_phase=8,         # #knots for phase spline (t / ISI)
    alpha=1.0,             # Ridge(alpha)
    standardize_within_fold=True,  # z-score X inside each CV fold

    # data sufficiency
    min_trials=10,         # min trials contributing to a window
    min_bins_per_trial=3,  # min finite bins from a trial within the window

    # ISI fairness & reporting
    stratify_by_isi=True,        # stratify CV folds by ISI
    isi_binning='levels',        # 'levels' (unique ISIs) or 'quantiles'
    min_trials_per_isi=8,        # per window; required for inclusion
    balance_mode='subsample',    # 'subsample' | 'weights' (class balance within folds)
    report_isi_tables=True,      # print ISI counts per window/fold

    # cross-ISI generalization diagnostics
    do_cross_isi_generalization=True,  # train on ISI set, test on held-out ISI

    # uncertainty on ΔR²
    n_bootstrap=200,         # bootstrap resamples per ROI×window for CI
    permutation_tests=False, # optional label shuffles (slower; off by default)

    # labeling thresholds
    delta_r2_eps=0.02,       # main threshold to call scaling vs clock
    effect_eps=0.01,         # reporting threshold for “small but positive” effects

    # misc
    random_state=0,
    verbose=True
)



cfg['decode'] = dict(
    preF2_window=(0.15, 0.70),   # analysis span on F1-OFF grid
    win_sec=0.12,                # window width (s)
    stride_frac=0.25,            # step = stride_frac * win_sec
    cv_folds=5,
    feature='mean',              # 'mean' or 'slope' or 'both'
    zscore_within_fold=True,
    balance_labels=True,         # downsample majority within each train fold
    min_trials_per_window=40,
    min_trials_per_class=40,
    min_bins_per_roi=3,          # per trial/ROI, bins needed inside a window
    auc_threshold=0.90,          # divergence threshold
    sustain_bins=3,              # bins above threshold to call divergence
    restrict_to='all',           # 'all' | 'correct' | 'incorrect'
    verbose=True
)


cfg['cluster'] = dict(
    preF2_window=(0.15, 0.70),     # same span you use elsewhere
    time_bins=30,                  # coarse summary in absolute time
    phase_bins=30,                 # coarse summary in phase (0..1)
    include_svclock_delta=True,    # append median ΔR² as an extra feature
    pca_dim=10,                    # compress before clustering
    method='gmm',                  # 'gmm' or 'kmeans'
    k_range=(2, 8),                # model selection range for GMM
    random_state=0,
    verbose=True
)



cfg['cs'] = dict(
    # Which alignments to analyze (must be built in M). Options: 'F1OFF','F2','CHOICE'
    aligns=['F1OFF', 'F2', 'CHOICE'],

    # Windows (seconds) relative to each alignment
    # peri_window=dict(F2=(0.00, 0.25), CHOICE=(0.00, 0.25), F1OFF=(0.00, 0.25)),
    # ctrl_window=dict(F2=(-0.25, 0.00), CHOICE=(-0.25, 0.00), F1OFF=(-0.15, -0.02)),

    peri_window=dict(F1OFF=(0.00, 0.25), F2=(0.00, 0.25), CHOICE=(0.00, 0.25)),
    ctrl_window=dict(F1OFF=(-0.15, -0.00), F2=(-0.25, 0.00), CHOICE=(-0.25, 0.00)),

    # Preprocessing
    detrend='median',          # 'none' | 'median'
    smooth_sigma_bins=0.0,     # 0 → no smoothing, min_rate_ratio=1.5,

    
    # Detection thresholds
    mode='absolute',           # or 'local_z'
    direction='positive',      # try 'both' for QA passes
    amp_thresh_abs=1.2,        # try 1.5–2.0 if nothing fires
    amp_thresh_z=2.0,          # if using local_z
    zscore_mode='robust',      # 'robust' (median/MAD) | 'standard'
    min_width_bins=1,          # minimum continuous bins above threshold
    max_width_bins=60,    
    refractory_s=0.08,         # minimum inter-event interval

    # Tag rule
    min_event_rate=0.08,       # fraction of trials with peri event
    min_rate_ratio=1.5,        # peri / ctrl
    max_width_s=0.60,
    min_trials_with_event=3,
    merge_rule='any_align',    # 'any_align' | 'all_aligns'

    # QA / output
    save_event_flags=False,
    print_top_k=10,            # show top ROIs by peri rate and by ratio
    verbose=True
)





# %%

    path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_simplex_20250529_2afc-379/sid_imaging_segmented_data.pkl'
    # path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/sid_imaging_segmented_data.pkl'

    import pickle

    with open(path, 'rb') as f:
        trial_data = pickle.load(f)   # one object back (e.g., a dict)  

# %%

# 1) Build canonical M from trial_data (+ all requested grids)
M = build_M_from_trials(trial_data, cfg)
print(f"[ok] M built. Trials={M['n_trials']}, ROIs={M['n_rois']}")
print(f"[F1OFF grid] span = {M['time'][0]:.3f} .. {M['time'][-1]:.3f} s | bins = {M['time'].size}")
have_f1   = ('roi_traces' in M) and ('time' in M)
have_f2   = ('roi_traces_F2locked' in M) and ('time_F2locked' in M)
have_choice = ('roi_traces_choice' in M) and ('time_choice' in M)
print(f"[grids] F1-OFF={have_f1} | F2-locked={have_f2} | CHOICE-locked={have_choice}")



# %%



for trial_idx in range(0,5):
    for roi_idx in range(0,3):
        fig = plot_interpolation_check(M, trial_data, cfg, align='F1OFF', trial_idx=trial_idx, roi_idx=roi_idx)




# %%
for trial_idx in range(30,33):
    for roi_idx in range(0, 10):
        plot_interpolation_check_for_all(M, trial_data, cfg, trial_idx=trial_idx, roi_idx=roi_idx)




# %%

# Run the analysis
M = scaling_vs_clock(M, cfg)

# Quick checks
sv = M['svclock']

print("ΔR² matrix shape:", sv['delta_r2'].shape)
print("Window centers (s):", sv['win_centers_s'])
unique, counts = np.unique(sv['labels'], return_counts=True)
print("Label counts:", dict(zip(unique, counts)))
print("Mean ΔR² (p10/50/90):", np.percentile(np.nanmean(sv['delta_r2'], axis=1), [10,50,90]).round(3))

# %%


for roi_idx in range(0, 10):
    plot_svclock_overlays(M, cfg, roi_idx=roi_idx, window_idx=4)



# %%


profiles = build_roi_profiles(M, cfg)           # per-ROI time/phase means in pre-F2
M = cluster_roi_profiles(M, cfg, profiles)      # labels + centroids + counts (+ svclock cross-tab)
_ = plot_cluster_profiles(M, cfg)               # sanity check the archetypes

M = cluster_quickstats(M, cfg)
_ = plot_cluster_quickstats(M)

M = cluster_phase_preference(M)
_ = plot_cluster_PI(M)

M = cluster_alignment_scores(M, cfg)

M = compute_phase_collapse_index(M, cfg)
_ = plot_PCI_by_cluster(M)

# optional amplitude-weighted check
M = amplitude_weighted_alignment(M)

mins_phi, _ = phase_of_min_per_roi(M, cfg)

compute_zPCI(M, cfg)


# you already computed: mins_phi, _ = phase_of_min_per_roi(M, cfg)
M = summarize_phase_minima(M, mins_phi)
M = ramp_mask_from_phi(M, mins_phi, min_slope=-0.05, late_thr=0.80)
# %%
# after your clustering step
M = cluster_quickstats(M, cfg)
_ = plot_cluster_quickstats(M)


M = cluster_phase_preference(M)
_ = plot_cluster_PI(M)

M = cluster_alignment_scores(M, cfg)



# %%

M = compute_phase_collapse_index(M, cfg)
_ = plot_PCI_by_cluster(M)

# optional amplitude-weighted check
M = amplitude_weighted_alignment(M)


# %%


mins_phi, _ = phase_of_min_per_roi(M, cfg)

compute_zPCI(M, cfg)



# %%

# you already computed: mins_phi, _ = phase_of_min_per_roi(M, cfg)
M = summarize_phase_minima(M, mins_phi)
M = ramp_mask_from_phi(M, mins_phi, min_slope=-0.05, late_thr=0.80)


# %%

# Set up filtering and sorting configuration
cfg['group_inspect'] = {
    'trial_type': 'all',
    'isi_constraint': 'all',
    'ramp_sort': 'min_peak_time',        # Good for ramps
    'non_ramp_sort': 'abs_peak_time'     # Good for mixed responses
}

# Available sorting methods:
# - 'max_peak_time': Time of maximum value
# - 'min_peak_time': Time of minimum value (good for ramps)
# - 'abs_peak_time': Time of largest absolute deviation from baseline
# - 'first_threshold_cross': First crossing of 1 SD threshold
# - 'slope_onset': Time when derivative becomes strongly negative
# - 'centroid_time': Center of mass of response
# - 'earliest_onset': Sustained threshold crossing (original method)
# - 'ramp_endpoint': Time of most negative value in second half
# - 'transient_peak': Time of sharpest peak (max second derivative)

# Examples:
cfg['group_inspect']['ramp_sort'] = 'ramp_endpoint'      # For ramping cells
cfg['group_inspect']['non_ramp_sort'] = 'transient_peak'  # For transient responses

# Or try onset-based sorting:
cfg['group_inspect']['ramp_sort'] = 'slope_onset'
cfg['group_inspect']['non_ramp_sort'] = 'first_threshold_cross'








# %%
# Set up filtering configuration
cfg['group_inspect'] = {
    'trial_type': 'all',        # 'all', 'rewarded', 'punished', 'correct', 'incorrect', 'did_not_choose'
    'isi_constraint': 'all'     # 'all', 'short', 'long', or list of specific ISI values
}

# Filter trials
trial_data_filtered = filter_trials_for_inspection(trial_data, cfg)

# Plot session-averaged rasters
fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg)


# %%

# Available sorting methods:
# - 'max_peak_time': Time of maximum value
# - 'min_peak_time': Time of minimum value (good for ramps)
# - 'abs_peak_time': Time of largest absolute deviation from baseline
# - 'first_threshold_cross': First crossing of 1 SD threshold
# - 'slope_onset': Time when derivative becomes strongly negative
# - 'centroid_time': Center of mass of response
# - 'earliest_onset': Sustained threshold crossing (original method)
# - 'ramp_endpoint': Time of most negative value in second half
# - 'transient_peak': Time of sharpest peak (max second derivative)

# Only rewarded trials
cfg['group_inspect'] = {'trial_type': 'rewarded', 'isi_constraint': 'all'}
cfg['group_inspect']['ramp_sort'] = 'min_peak_time'      # For ramping cells
cfg['group_inspect']['non_ramp_sort'] = 'min_peak_time'  # For transient responses

# Filter trials
trial_data_filtered = filter_trials_for_inspection(trial_data, cfg)

# Plot session-averaged rasters
fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg)

# %%

# Only rewarded trials
cfg['group_inspect'] = {'trial_type': 'rewarded', 'isi_constraint': 'short'}
cfg['group_inspect']['ramp_sort'] = 'min_peak_time'      # For ramping cells
cfg['group_inspect']['non_ramp_sort'] = 'min_peak_time'  # For transient responses

# Filter trials
trial_data_filtered = filter_trials_for_inspection(trial_data, cfg)

# Plot session-averaged rasters
fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg)

# %%
# Available sorting methods:
# - 'max_peak_time': Time of maximum value
# - 'min_peak_time': Time of minimum value (good for ramps)
# - 'abs_peak_time': Time of largest absolute deviation from baseline
# - 'first_threshold_cross': First crossing of 1 SD threshold
# - 'slope_onset': Time when derivative becomes strongly negative
# - 'centroid_time': Center of mass of response
# - 'earliest_onset': Sustained threshold crossing (original method)
# - 'ramp_endpoint': Time of most negative value in second half
# - 'transient_peak': Time of sharpest peak (max second derivative)


short_isis = M['short_levels']
for isi in short_isis:
    # Only short ISI trials  
    cfg['group_inspect'] = {'trial_type': 'rewarded', 'isi_constraint': [isi]}
    cfg['group_inspect']['ramp_sort'] = 'min_peak_time'      # For ramping cells
    cfg['group_inspect']['non_ramp_sort'] = 'min_peak_time'  # For transient responses
    
    # Filter trials
    trial_data_filtered = filter_trials_for_inspection(trial_data, cfg)
    
    # Plot session-averaged rasters
    fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg)



# %%


short_isis = M['short_levels']
for idx, isi in enumerate(short_isis):
    # Only short ISI trials  
    cfg['group_inspect'] = {'trial_type': 'rewarded', 'isi_constraint': [isi]}
    cfg['group_inspect']['ramp_sort'] = 'min_peak_time'      # For ramping cells
    cfg['group_inspect']['non_ramp_sort'] = 'min_peak_time'  # For transient responses
    print(idx)
    print(isi)
    # Filter trials
    trial_data_filtered = filter_trials_for_inspection(trial_data, cfg)
    if idx == 0:
        fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg, 
                                          use_stored_order=False, store_order=True)
    else:
        # Plot session-averaged rasters
        fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg, 
                                                use_stored_order=True, store_order=False)


long_isis = M['long_levels']
for idx, isi in enumerate(long_isis):
    # Only short ISI trials  
    cfg['group_inspect'] = {'trial_type': 'rewarded', 'isi_constraint': [isi]}
    print(idx)
    print(isi)
    # Filter trials
    trial_data_filtered = filter_trials_for_inspection(trial_data, cfg)
    # Plot session-averaged rasters
    fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg, 
                                                use_stored_order=True, store_order=False)



# %%


short_isis = M['short_levels']
for idx, isi in enumerate(short_isis):
    # Only short ISI trials  
    cfg['group_inspect'] = {'trial_type': 'punished', 'isi_constraint': [isi]}
    cfg['group_inspect']['ramp_sort'] = 'min_peak_time'      # For ramping cells
    cfg['group_inspect']['non_ramp_sort'] = 'min_peak_time'  # For transient responses
    print(idx)
    print(isi)
    # Filter trials
    trial_data_filtered = filter_trials_for_inspection(trial_data, cfg)
    if idx == 0:
        fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg, 
                                          use_stored_order=False, store_order=True)
    else:
        # Plot session-averaged rasters
        fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg, 
                                                use_stored_order=True, store_order=False)


# %%

# Only rewarded trials
cfg['group_inspect'] = {'trial_type': 'rewarded', 'isi_constraint': 'long'}
cfg['group_inspect']['ramp_sort'] = 'min_peak_time'      # For ramping cells
cfg['group_inspect']['non_ramp_sort'] = 'min_peak_time'  # For transient responses

# Filter trials
trial_data_filtered = filter_trials_for_inspection(trial_data, cfg)

# Plot session-averaged rasters
fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg)

# %%


long_isis = M['long_levels']
for idx, isi in enumerate(long_isis):
    # Only short ISI trials  
    cfg['group_inspect'] = {'trial_type': 'rewarded', 'isi_constraint': [isi]}
    cfg['group_inspect']['ramp_sort'] = 'min_peak_time'      # For ramping cells
    cfg['group_inspect']['non_ramp_sort'] = 'min_peak_time'  # For transient responses
    print(idx)
    print(isi)
    # Filter trials
    trial_data_filtered = filter_trials_for_inspection(trial_data, cfg)
    if idx == 0:
        fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg, 
                                          use_stored_order=False, store_order=True)
    else:
        # Plot session-averaged rasters
        fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg, 
                                                use_stored_order=True, store_order=False)


short_isis = M['short_levels']
for idx, isi in enumerate(short_isis):
    # Only short ISI trials  
    cfg['group_inspect'] = {'trial_type': 'rewarded', 'isi_constraint': [isi]}
    print(idx)
    print(isi)
    # Filter trials
    trial_data_filtered = filter_trials_for_inspection(trial_data, cfg)
    # Plot session-averaged rasters
    fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg, 
                                                use_stored_order=True, store_order=False)


# %%


long_isis = M['long_levels']
for isi in long_isis:
    # Only short ISI trials  
    cfg['group_inspect'] = {'trial_type': 'punished', 'isi_constraint': [isi]}
    cfg['group_inspect']['ramp_sort'] = 'min_peak_time'      # For ramping cells
    cfg['group_inspect']['non_ramp_sort'] = 'min_peak_time'  # For transient responses
    
    # Filter trials
    trial_data_filtered = filter_trials_for_inspection(trial_data, cfg)
    
    # Plot session-averaged rasters
    fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg)


# %%

# Specific ISI values
cfg['group_inspect'] = {'trial_type': 'all', 'isi_constraint': [0.45, 0.70]}


# Filter trials
trial_data_filtered = filter_trials_for_inspection(trial_data, cfg)

# Plot session-averaged rasters
fig = plot_ramp_vs_nonramp_session_rasters(trial_data_filtered, M, cfg)



# %%

# you already have: M['ramp_mask_phi']
fig = plot_rasters_by_rampmask_sorted(M, cfg, M['ramp_mask_phi'],
                                      onset={'thresh_mode':'z','z':1.0,'min_bins':3,'smooth_bins':2.0})



# %%

# You already computed: mins_phi and made ramp_mask_phi
M = summarize_overlap_cluster_rampmask(M, M['ramp_mask_phi'])
_ = plot_rampmask_centroids(M, cfg, M['ramp_mask_phi'])
dprime_out, _ = ramp_dprime_timecourse(M, cfg, M['ramp_mask_phi'])



# %%

dprime_out = ramp_dprime_timecourse_balanced(M, cfg, M['ramp_mask_phi'])
dprime_out = ramp_dprime_phase(M, cfg, M['ramp_mask_phi'])


# %%


cfg.setdefault('plots', {})
cfg['plots'].setdefault('cluster', {
    'max_spaghetti': 60,      # per cluster
    'ylim_percentiles': (2, 98),
    'ylim_pad_frac': 0.05
})


# you already ran:
# profiles = build_roi_profiles(M, cfg)
# M = cluster_roi_profiles(M, cfg, profiles)     # produced the Cluster 0/1 plot you posted

_ = plot_cluster_spaghetti(M, cfg)               # thin per-ROI + bold centroid (time & phase)
_ = plot_cluster_heatmaps(M, cfg)                # ROI×time / ROI×phase heatmaps, ordered
_ = plot_cluster_perISI_overlays(M, cfg)         # per-ISI overlays per cluster (time axis)
metrics = compute_cluster_metrics(M, cfg)        # small numbers to sanity-check clustering






# %%










# %%

M = roi_shortlong_screen(M, cfg)
cfg['decode']['use_roi_mask'] = True
cfg['decode']['roi_mask_rule'] = {'type': 'qval', 'thresh': 0.1}  # or {'type':'auc', 'thresh':0.58}



# %%


cfg['decode']['min_trials_per_class'] = 40  # or match your session’s class counts
cfg['decode']['feature'] = 'both'           # quick comparison of mean vs slope
cfg['decode']['preF2_window'] = (0.15, 0.60)  # or 0.55
cfg['decode']['min_trials_per_class'] = 80
cfg['decode']['auc_threshold'] = 0.6
cfg['decode']['sustain_bins'] = 3

# hard subset by q ≤ 0.1 from the screen
cfg['decode'].update({
    'use_roi_mask': True,
    'roi_mask_rule': {'type': 'qval', 'thresh': 0.10},
    # or: 'roi_mask_rule': {'type': 'auc', 'thresh': 0.58},
    'use_roi_weights': False,   # set True to down-weight weak ROIs instead of masking
})


M = time_resolved_decoding(M, cfg)
print("AUC shape:", M['decode']['auc'].shape)
print("Divergence @", M['decode']['divergence_time_s'], "s")




# %%


# assumes you already built M with F1OFF+F2+CHOICE grids per cfg['grids']
M = tag_cs_likeness(M, cfg)

# quick look at how many tagged
print("CS-like count:", int(M['cs']['roi_is_cs_like'].sum()))
for a, met in M['cs']['metrics'].items():
    print(a, "mean peri rate:", np.nanmean(met['event_rate_peri']),
              "max z (union) p95:", np.nanpercentile(met['max_z_union'], 95))
    print(a, "finite_trial_frac p50/p10:", 
          np.percentile(met['finite_trial_frac'], 50),
          np.percentile(met['finite_trial_frac'], 10))

