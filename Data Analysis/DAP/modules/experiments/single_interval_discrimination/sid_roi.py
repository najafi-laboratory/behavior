import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def run_sid_insight_roi_analysis(
    traces: np.ndarray,
    time: np.ndarray,
    f2_times: np.ndarray,
    is_short: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
    phase_bins: int = 60,
    pe_window: Tuple[float, float] = (0.0, 0.5),
    dprime_threshold: float = 1.0,
    sustain_ms: float = 100.0,
) -> Dict[str, Any]:
    """
    Minimal insight analysis (scaling vs clock, decode, hazard unique var, PE@F2).

    Inputs
      traces: float array [n_trials, n_rois, n_time], aligned to F1 (stim1) at t=0
      time:   float array [n_time], seconds relative to F1
      f2_times: float array [n_trials], seconds from F1 to F2 on each trial
      is_short: optional bool array [n_trials], True for "short" category (defaults to median split)
      output_dir: optional directory to save a one-page figure (PNG)
      phase_bins: number of bins for phase warping (scaling model)
      pe_window: window relative to F2 for PE measurement (seconds)
      dprime_threshold: threshold for divergence detection (|d'| > threshold)
      sustain_ms: required sustained time above threshold (ms)

    Returns
      dict with:
        delta_r2: [n_rois] ΔR² (scaling – clock)
        r2_scaling, r2_clock: [n_rois]
        alpha_phase_pref: [n_rois] preferred phase (0–1) if scaling wins, else NaN
        latency_ms_pref: [n_rois] preferred latency (ms) if clock wins, else NaN
        sort_index: ROI order to visualize winners by their preferred phase/latency
        frac_scaling, frac_clock, frac_tie: population fractions
        decode:
          dprime_time: [n_time_common] time vector (s)
          dprime_curve: [n_time_common] population d′(t)
          divergence_time_s: float or None
        hazard_unique_variance: float
        pe:
          slope: float (population mean)
          p_value: float (population mean)
        figure_path: str or None
    """
    assert traces.ndim == 3, "traces must be [n_trials, n_rois, n_time]"
    n_trials, n_rois, n_time = traces.shape
    time = np.asarray(time)
    f2_times = np.asarray(f2_times)

    # Common pre-F2 window: up to the earliest F2 across trials
    T_abs = float(np.nanmin(f2_times))
    common_mask = (time >= 0) & (time <= T_abs)
    t_common = time[common_mask]
    if t_common.size < 5:
        return {"success": False, "error": "Common pre-F2 window too short"}

    # Prepare short/long labels if not provided
    if is_short is None:
        median_isi = np.nanmedian(f2_times)
        is_short = f2_times <= median_isi
    is_short = is_short.astype(bool)

    # 1) Scaling vs Clock CV comparison (leave-one-trial-out)
    r2_scaling = np.full(n_rois, np.nan, float)
    r2_clock = np.full(n_rois, np.nan, float)
    pref_phase = np.full(n_rois, np.nan, float)
    pref_latency = np.full(n_rois, np.nan, float)

    # Precompute per-trial pre-F2 segments for speed
    # Keep only the common window for R² fairness
    Y = traces[:, :, :][:, :, common_mask]  # [n_trials, n_rois, n_common]
    # For scaling, we also need full pre-F2 segment per trial for phase template; we’ll interpolate to phase then back to t_common

    for r in range(n_rois):
        y = Y[:, r, :]  # [n_trials, n_common]
        if not np.isfinite(y).any():
            continue

        # Clock model template (absolute time): mean across training trials at t_common
        r2s_clock = []
        r2s_scal = []

        # Also estimate "preferred" features for sorting
        # - Scaling: preferred phase = argmax of phase-template
        # - Clock: preferred latency = argmax of absolute-time template
        scal_templates = []
        clock_templates = []

        for loo in range(n_trials):
            train_idx = np.arange(n_trials) != loo
            test_idx = loo

            # CLOCK template (absolute time)
            clock_template = np.nanmean(y[train_idx, :], axis=0)  # [n_common]
            clock_templates.append(clock_template)

            # Predict test trial with clock template
            y_true = y[test_idx, :]
            y_pred_c = clock_template
            r2_c = _safe_r2(y_true, y_pred_c)
            r2s_clock.append(r2_c)

            # SCALING template:
            # Build phase-averaged template from training trials:
            #  - For each training trial, take its pre-F2 trace on [0, f2_i], interpolate onto phase grid [0..1]
            #  - Average across trials to get a phase template
            phase_grid = np.linspace(0, 1, phase_bins)
            phase_stack = []
            for tr in np.where(train_idx)[0]:
                # This trial’s full pre-F2 time and signal
                mask_tr = (time >= 0) & (time <= f2_times[tr])
                t_tr = time[mask_tr]
                if t_tr.size < 5:
                    continue
                y_tr_full = traces[tr, r, mask_tr]
                # Map to phase
                if (t_tr[-1] - t_tr[0]) <= 0:
                    continue
                phase_tr = (t_tr - t_tr[0]) / (t_tr[-1] - t_tr[0])
                phase_tr = np.clip(phase_tr, 0, 1)
                y_phase = np.interp(phase_grid, phase_tr, y_tr_full, left=np.nan, right=np.nan)
                phase_stack.append(y_phase)
            if len(phase_stack) == 0:
                r2s_scal.append(np.nan)
                continue
            phase_template = np.nanmean(np.vstack(phase_stack), axis=0)  # [phase_bins]
            scal_templates.append(phase_template)

            # Predict left-out trial by sampling phase_template at phases that correspond to t_common on this trial
            # Left-out trial’s phase at t_common is phi(t) = t / f2_loo
            f2_loo = f2_times[test_idx]
            if f2_loo <= 0:
                r2s_scal.append(np.nan)
                continue
            phi_test = np.clip(t_common / f2_loo, 0, 1)
            y_pred_s = np.interp(phi_test, np.linspace(0, 1, phase_bins), phase_template)
            r2_s = _safe_r2(y_true, y_pred_s)
            r2s_scal.append(r2_s)

        # Aggregate CV scores
        r2_clock[r] = np.nanmean(r2s_clock)
        r2_scaling[r] = np.nanmean(r2s_scal)

        # Preferred features for sorting (use the mean template across folds)
        if len(clock_templates) > 0:
            mean_clock = np.nanmean(np.vstack(clock_templates), axis=0)  # [n_common]
            pref_latency[r] = float(t_common[np.nanargmax(mean_clock)]) * 1000.0  # ms
        if len(scal_templates) > 0:
            mean_scal = np.nanmean(np.vstack(scal_templates), axis=0)  # [phase_bins]
            pref_phase[r] = float(np.nanargmax(mean_scal) / (phase_bins - 1))

    delta_r2 = r2_scaling - r2_clock
    winners_scaling = delta_r2 > 0
    winners_clock = delta_r2 < 0
    frac_scaling = np.nanmean(winners_scaling.astype(float))
    frac_clock = np.nanmean(winners_clock.astype(float))
    frac_tie = 1.0 - (frac_scaling + frac_clock)

    # Build sort index: scale-winners by phase asc, then clock-winners by latency asc
    idx_scal = np.where(winners_scaling)[0]
    idx_clk = np.where(winners_clock)[0]
    idx_scal_sorted = idx_scal[np.argsort(np.nan_to_num(pref_phase[idx_scal], nan=1e9))]
    idx_clk_sorted = idx_clk[np.argsort(np.nan_to_num(pref_latency[idx_clk], nan=1e9))]
    sort_index = np.concatenate([idx_scal_sorted, idx_clk_sorted])

    # 2) Time-resolved decode (population d′ using ROI-averaged signal)
    # Simple and dependency-free: average across ROIs to a population trace per trial, then d′(t) over time
    pop_traces = np.nanmean(traces[:, :, :], axis=1)  # [n_trials, n_time]
    pop_common = pop_traces[:, common_mask]          # [n_trials, n_common]
    dprime_curve = _time_resolved_dprime(pop_common, is_short)
    # Divergence time: when |d'| > threshold for sustain_ms
    sustain_samples = max(1, int(sustain_ms / (np.median(np.diff(t_common)) * 1000.0)))
    divergence_time_s = _first_sustained_crossing(np.abs(dprime_curve), dprime_threshold, sustain_samples, t_common)

    # 3) Hazard unique variance (trial-level)
    # Discrete hazard for allowed F2 times (assume schedule is the set of unique F2 times)
    haz = _discrete_hazard_from_schedule(f2_times)
    # Trial-level pre-F2 activity summary (mean over common window)
    trial_mean_pre = np.nanmean(pop_common, axis=1)
    # Regress trial_mean_pre ~ zscore(ISI) + zscore(hazard), report unique variance (semi-partial R²) for hazard
    hazard_unique = _unique_variance_two_regressors(trial_mean_pre, f2_times, haz)

    # 4) Prediction error at F2: correlate F2-locked response with |ISI - mean(ISI)|
    pe_mag = _pe_response_magnitude(traces, time, f2_times, window=pe_window)  # [n_trials, n_rois]
    surprise = np.abs(f2_times - np.nanmean(f2_times))
    # Population-level: average ROI-wise slopes (each ROI regresses pe_mag[:, r] ~ surprise)
    slopes = []
    pvals = []
    for r in range(n_rois):
        y = pe_mag[:, r]
        if np.isfinite(y).sum() > 3:
            slope, intercept, r_val, p_val, stderr = stats.linregress(surprise, y)
            slopes.append(slope)
            pvals.append(p_val)
    pe_slope = float(np.nanmean(slopes)) if len(slopes) else np.nan
    pe_p = float(np.nanmean(pvals)) if len(pvals) else np.nan

    # One-page figure (optional)
    figure_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        figure_path = os.path.join(output_dir, "sid_insight_summary.png")
        _plot_summary_figure(
            delta_r2, r2_scaling, r2_clock, winners_scaling, winners_clock,
            pref_phase, pref_latency, sort_index, traces, time, t_common,
            dprime_curve, divergence_time_s, hazard_unique, pe_mag, surprise, figure_path
        )

    return {
        "success": True,
        "delta_r2": delta_r2,
        "r2_scaling": r2_scaling,
        "r2_clock": r2_clock,
        "alpha_phase_pref": pref_phase,
        "latency_ms_pref": pref_latency,
        "sort_index": sort_index,
        "frac_scaling": float(frac_scaling),
        "frac_clock": float(frac_clock),
        "frac_tie": float(frac_tie),
        "decode": {
            "dprime_time": t_common,
            "dprime_curve": dprime_curve,
            "divergence_time_s": float(divergence_time_s) if divergence_time_s is not None else None,
        },
        "hazard_unique_variance": float(hazard_unique),
        "pe": {"slope": float(pe_slope), "p_value": float(pe_p)},
        "figure_path": figure_path,
    }


# ----------------- helpers -----------------

def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return np.nan
    ss_res = np.sum((y_true[ok] - y_pred[ok]) ** 2)
    ss_tot = np.sum((y_true[ok] - np.mean(y_true[ok])) ** 2)
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _time_resolved_dprime(pop_traces_common: np.ndarray, is_short: np.ndarray) -> np.ndarray:
    # d′(t) = (μ1-μ2)/sqrt(0.5*(σ1^2+σ2^2))
    short = pop_traces_common[is_short, :]
    longg = pop_traces_common[~is_short, :]
    mu1 = np.nanmean(short, axis=0)
    mu2 = np.nanmean(longg, axis=0)
    s1 = np.nanvar(short, axis=0, ddof=1)
    s2 = np.nanvar(longg, axis=0, ddof=1)
    denom = np.sqrt(0.5 * (s1 + s2) + 1e-12)
    dprime = (mu1 - mu2) / denom
    # small smoothing
    if dprime.size >= 5:
        kernel = np.ones(5) / 5.0
        dprime = np.convolve(dprime, kernel, mode="same")
    return dprime


def _first_sustained_crossing(arr_abs: np.ndarray, thr: float, sustain_samples: int, t: np.ndarray):
    above = arr_abs > thr
    # Find first index where a run of True of length >= sustain_samples begins
    if sustain_samples <= 1:
        idx = np.argmax(above)
        return float(t[idx]) if above.any() else None
    count = 0
    for i, a in enumerate(above):
        count = count + 1 if a else 0
        if count >= sustain_samples:
            start_idx = i - sustain_samples + 1
            return float(t[start_idx])
    return None


def _discrete_hazard_from_schedule(f2_times: np.ndarray) -> np.ndarray:
    # Build discrete hazard from set of unique allowed F2 times (assume quasi-uniform schedule)
    uniq = np.sort(np.unique(np.round(f2_times, 6)))
    n = len(uniq)
    if n < 2:
        return np.zeros_like(f2_times)
    # hazard at k-th time: h_k = p(t_k | not occurred) ≈ 1 / (n - k + 1) for uniform schedule
    # Map each trial’s F2 to its index in uniq
    idx = np.searchsorted(uniq, np.round(f2_times, 6))
    idx = np.clip(idx, 0, n - 1)
    haz_levels = np.array([1.0 / (n - k) if (n - k) > 0 else 1.0 for k in range(n)])  # simple discrete hazard
    haz = haz_levels[idx]
    # z-score
    return (haz - np.nanmean(haz)) / (np.nanstd(haz) + 1e-12)


def _unique_variance_two_regressors(y: np.ndarray, x_time: np.ndarray, x_hazard: np.ndarray) -> float:
    # semi-partial R² for hazard after controlling time (ISI)
    y = np.asarray(y).ravel()
    X1 = np.column_stack([np.ones_like(x_time), _z(x_time)])
    X2 = np.column_stack([np.ones_like(x_time), _z(x_time), _z(x_hazard)])
    r2_1 = _ols_r2(y, X1)
    r2_2 = _ols_r2(y, X2)
    # unique variance of hazard
    uv = max(0.0, r2_2 - r2_1)
    return uv


def _ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if ok.sum() < X.shape[1] + 1:
        return np.nan
    Xok = X[ok, :]
    yok = y[ok]
    # OLS via normal equations with ridge epsilon
    ridge = 1e-8
    beta = np.linalg.pinv(Xok.T @ Xok + ridge * np.eye(Xok.shape[1])) @ (Xok.T @ yok)
    yhat = Xok @ beta
    return _safe_r2(yok, yhat)


def _z(x):
    x = np.asarray(x).ravel()
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def _pe_response_magnitude(traces: np.ndarray, time: np.ndarray, f2_times: np.ndarray, window=(0.0, 0.5)) -> np.ndarray:
    # For each trial, ROI: mean activity in [F2+win[0], F2+win[1]]
    n_trials, n_rois, _ = traces.shape
    pe = np.full((n_trials, n_rois), np.nan, float)
    dt = np.median(np.diff(time))
    for tr in range(n_trials):
        f2 = f2_times[tr]
        w0 = f2 + window[0]
        w1 = f2 + window[1]
        mask = (time >= w0) & (time <= w1)
        if mask.sum() < max(3, int(0.05 / max(dt, 1e-6))):  # need a few points
            continue
        seg = traces[tr, :, :][:, mask]  # [n_rois, n_win]
        pe[tr, :] = np.nanmean(seg, axis=1)
    return pe


def _plot_summary_figure(
    delta_r2, r2_scaling, r2_clock, winners_scaling, winners_clock,
    pref_phase, pref_latency, sort_index, traces, time, t_common,
    dprime_curve, divergence_time_s, hazard_unique, pe_mag, surprise, out_path
):
    import matplotlib.gridspec as gridspec

    n_rois = traces.shape[1]
    fig = plt.figure(figsize=(12, 9), dpi=150)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1.2], width_ratios=[1, 1, 1], wspace=0.35, hspace=0.45)

    # ΔR² histogram
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(delta_r2[np.isfinite(delta_r2)], bins=40, color='gray', alpha=0.9)
    ax.axvline(0, color='k', linestyle='--', lw=1)
    ax.set_title(f"ΔR² (scaling - clock)\nfrac S={np.nanmean(winners_scaling):.2f}, C={np.nanmean(winners_clock):.2f}")
    ax.set_xlabel("ΔR²"); ax.set_ylabel("Count")

    # Decode curve
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t_common, dprime_curve, color='C3', lw=2)
    ax.axhline(0, color='k', lw=1, ls=':')
    if divergence_time_s is not None:
        ax.axvline(divergence_time_s, color='C3', lw=1.5, ls='--', label=f"div={divergence_time_s*1000:.0f} ms")
        ax.legend(frameon=False)
    ax.set_title("Population d′(t) short vs long")
    ax.set_xlabel("Time from F1 (s)"); ax.set_ylabel("d′")

    # Hazard unique variance
    ax = fig.add_subplot(gs[0, 2])
    ax.bar([0], [hazard_unique], color='C0', width=0.6)
    ax.set_xticks([0]); ax.set_xticklabels(["Hazard (unique)"])
    ax.set_ylim(0, max(0.25, hazard_unique * 1.5))
    ax.set_title("Hazard unique variance")

    # PE at F2 (population)
    ax = fig.add_subplot(gs[1, 0])
    pe_pop = np.nanmean(pe_mag, axis=1)  # per-trial population PE
    slope, intercept, r, p, _ = stats.linregress(surprise, pe_pop)
    ax.scatter(surprise, pe_pop, s=14, alpha=0.6, color='C2')
    xs = np.linspace(np.nanmin(surprise), np.nanmax(surprise), 50)
    ax.plot(xs, intercept + slope * xs, color='C2', lw=2, label=f"slope={slope:.3g}, p={p:.3g}")
    ax.legend(frameon=False)
    ax.set_xlabel("|ISI - mean(ISI)| (s)"); ax.set_ylabel("F2 response (pop)")
    ax.set_title("Prediction error @F2")

    # Preferred phase (scaling winners)
    ax = fig.add_subplot(gs[1, 1])
    ph = pref_phase[winners_scaling]
    ax.hist(ph[np.isfinite(ph)], bins=20, color='C1', alpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Preferred phase"); ax.set_ylabel("Count")
    ax.set_title("Scaling winners: phase pref")

    # Preferred latency (clock winners)
    ax = fig.add_subplot(gs[1, 2])
    lat = pref_latency[winners_clock]
    ax.hist(lat[np.isfinite(lat)], bins=20, color='C4', alpha=0.9)
    ax.set_xlabel("Preferred latency (ms)"); ax.set_ylabel("Count")
    ax.set_title("Clock winners: latency pref")

    # Raster sorted by winning story
    ax = fig.add_subplot(gs[2, :])
    # Build pre-F2 population raster on common window, sorted
    # Use mean across trials to visualize response per ROI
    mean_pre = np.nanmean(traces[:, :, :][:, :, (time >= 0) & (time <= t_common[-1])], axis=0)  # [n_rois, n_common]
    im = ax.imshow(mean_pre[sort_index, :], aspect='auto', extent=[t_common[0], t_common[-1], 0, len(sort_index)],
                   cmap='RdBu_r', vmin=-np.nanstd(mean_pre), vmax=np.nanstd(mean_pre))
    ax.set_xlabel("Time from F1 (s)"); ax.set_ylabel("ROIs (sorted)")
    ax.set_title("Raster sorted by winning story")
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("ΔF/F")

    fig.suptitle("SID Insight Summary", y=0.995, fontsize=14)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)