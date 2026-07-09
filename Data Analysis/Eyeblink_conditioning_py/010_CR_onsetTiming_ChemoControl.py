from __future__ import annotations
import glob
import os
import re
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.use14corefonts"] = False

CTRL_COLOR  = "black"
CHEMO_COLOR = "blue"
CTRL_POOR   = "dimgray"
CHEMO_POOR  = "cornflowerblue"
CTRL_NOCR   = "lightgray"
CHEMO_NOCR  = "lightblue"


def scalar_or_nan(x):
    arr = np.asarray(x).squeeze()
    if arr.size == 0:
        return np.nan
    if arr.ndim == 0:
        try:
            return float(arr)
        except Exception:
            return np.nan
    try:
        return float(arr.flat[0])
    except Exception:
        return np.nan


def get_field(obj, field_name, default=None):
    if obj is None:
        return default
    if hasattr(obj, field_name):
        return getattr(obj, field_name)
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return default


def has_field(obj, field_name):
    if obj is None:
        return False
    if hasattr(obj, field_name):
        return True
    if isinstance(obj, dict):
        return field_name in obj
    return False


def safe_array(x):
    if x is None:
        return np.array([], dtype=float)
    arr = np.asarray(x).squeeze()
    try:
        return arr.astype(float)
    except Exception:
        return arr


def ensure_trial_list(raw_trials):
    if raw_trials is None:
        return []
    if isinstance(raw_trials, np.ndarray):
        return list(raw_trials.flat)
    if isinstance(raw_trials, (list, tuple)):
        return list(raw_trials)
    return [raw_trials]


def smooth_trace(x, window=5):
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    return uniform_filter1d(x, size=window, mode="nearest")


def is_chemogenetics_session(session_data):
    trial_settings = get_field(session_data, "TrialSettings", None)
    if trial_settings is None:
        return False

    for trial_setting in np.asarray(trial_settings).flat:
        gui = get_field(trial_setting, "GUI", None)
        chemo_enabled = safe_array(get_field(gui, "ChemogeneticsEnabled", 0))
        if np.any(chemo_enabled.astype(float) != 0):
            return True
    return False


def interp_to_grid(x, y, xq):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return np.full_like(xq, np.nan, dtype=float)
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]
    if len(x_unique) < 2:
        return np.full_like(xq, np.nan, dtype=float)
    f = interp1d(x_unique, y_unique, kind="linear", bounds_error=False, fill_value=np.nan)
    return f(xq)


def collect_overall_max(trials):
    total_vals = []
    eye_vals = []
    for tr in trials:
        data = get_field(tr, "Data", None)
        if data is None:
            continue
        total_ellipse = get_field(data, "totalEllipsePixels", None)
        if total_ellipse is None:
            for cand in ["totalEllipseP", "totalEllipsePix", "totalEllipsePixel"]:
                total_ellipse = get_field(data, cand, None)
                if total_ellipse is not None:
                    break
        if total_ellipse is not None:
            arr = np.asarray(total_ellipse).squeeze()
            if arr.size > 0:
                if arr.ndim == 0:
                    total_vals.append(float(arr))
                else:
                    total_vals.extend(np.asarray(arr, dtype=float).ravel().tolist())
        eye_area = get_field(data, "eyeAreaPixels", None)
        if eye_area is not None:
            arr_eye = np.asarray(eye_area).squeeze()
            if arr_eye.size > 0:
                if arr_eye.ndim == 0:
                    eye_vals.append(float(arr_eye))
                else:
                    eye_vals.extend(np.asarray(arr_eye, dtype=float).ravel().tolist())
    if len(eye_vals) > 0:
        return np.nanmax(np.asarray(eye_vals, dtype=float))
    if len(total_vals) > 0:
        return np.nanmax(np.asarray(total_vals, dtype=float))
    return np.nan


def get_cr_window_from_block(
    t_puff,
    block_label,
    short_pre_ms=25,
    short_post_ms=15,
    long_pre_ms=50,
    long_post_ms=15,
):
    block_label = str(block_label).strip().lower()
    short_pre = short_pre_ms / 1000.0
    short_post = short_post_ms / 1000.0
    long_pre = long_pre_ms / 1000.0
    long_post = long_post_ms / 1000.0
    if block_label == "short":
        cr_start = t_puff - short_pre
        cr_end = t_puff + short_post
    elif block_label == "long":
        cr_start = t_puff - long_pre
        cr_end = t_puff + long_post
    else:
        return np.nan, np.nan
    return cr_start, cr_end


def classify_trial(
    time,
    signal,
    t_led,
    t_puff,
    block_label,
    good_cr_threshold,
    poor_cr_threshold,
    short_pre_ms=25,
    short_post_ms=15,
    long_pre_ms=50,
    long_post_ms=15,
    poor_drop_threshold=0.05,
):
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time) != len(signal) or len(time) == 0:
        return "No CR", np.nan, np.nan
    baseline_idx = (time >= (t_led - 0.2)) & (time <= t_led)
    if not np.any(baseline_idx):
        return "No CR", np.nan, np.nan
    baseline_amp = np.nanmean(signal[baseline_idx])
    if not np.isfinite(baseline_amp):
        return "No CR", np.nan, np.nan

    cr_start, cr_end = get_cr_window_from_block(
        t_puff=t_puff,
        block_label=block_label,
        short_pre_ms=short_pre_ms,
        short_post_ms=short_post_ms,
        long_pre_ms=long_pre_ms,
        long_post_ms=long_post_ms,
    )
    if not (np.isfinite(cr_start) and np.isfinite(cr_end)):
        return "No CR", np.nan, baseline_amp

    timing_idx = (time >= t_led) & (time < t_puff)
    if not np.any(timing_idx):
        return "No CR", np.nan, baseline_amp

    cr_indices = np.flatnonzero(timing_idx & np.isfinite(signal))
    before_candidates = np.flatnonzero((time < t_puff) & np.isfinite(signal))
    if cr_indices.size == 0 or before_candidates.size == 0:
        return "No CR", np.nan, baseline_amp

    max_idx = cr_indices[np.nanargmax(signal[cr_indices])]
    before_us_idx = before_candidates[-1]
    cr_amp = signal[max_idx]
    cr_peak_bs = cr_amp - baseline_amp

    if not np.isfinite(cr_peak_bs) or cr_peak_bs <= poor_cr_threshold:
        return "No CR", cr_peak_bs, baseline_amp

    if (
        np.isfinite(signal[before_us_idx])
        and (cr_amp - signal[before_us_idx]) > poor_drop_threshold
        and time[max_idx] < time[before_us_idx]
    ):
        return "Poor CR", cr_peak_bs, baseline_amp

    if cr_peak_bs > good_cr_threshold:
        return "Good CR", cr_peak_bs, baseline_amp
    return "Poor CR", cr_peak_bs, baseline_amp


def get_onset_search_end(block_label, short_end_s=0.212, long_end_s=0.412):
    block_label = str(block_label).strip().lower()
    if block_label == "short":
        return short_end_s
    elif block_label == "long":
        return long_end_s
    return np.nan


def find_detected_onset_velocity_based(
    time,
    signal,
    block_label,
    t_led=0.0,
    min_peak_height=0.01,
    baseline_window=(-0.2, 0.0),
    short_end_s=0.212,
    long_end_s=0.412,
    velocity_smooth_window=5,
    velocity_std_factor=2.0,
    velocity_floor=0.5,
    min_sustain_samples=2,
    min_onset_latency_s=0.050,
):
    """Detect CR onset on individual trials via velocity threshold (forward scan).

    Smooths the baseline-corrected FEC, computes the velocity (gradient),
    then scans forward from min_onset_latency_s to find the first run of
    min_sustain_samples consecutive samples where velocity exceeds
    baseline_mean + velocity_std_factor * baseline_std.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time) != len(signal) or len(time) == 0:
        return np.nan, np.nan, np.nan, np.nan
    valid = np.isfinite(time) & np.isfinite(signal)
    time = time[valid]
    signal = signal[valid]
    if len(time) < 2:
        return np.nan, np.nan, np.nan, np.nan
    baseline_idx = (time >= (t_led + baseline_window[0])) & (time <= (t_led + baseline_window[1]))
    if not np.any(baseline_idx):
        return np.nan, np.nan, np.nan, np.nan
    baseline_amp = np.nanmean(signal[baseline_idx])
    signal_bs = signal - baseline_amp

    search_end = get_onset_search_end(block_label=block_label, short_end_s=short_end_s, long_end_s=long_end_s)
    if not np.isfinite(search_end):
        return np.nan, np.nan, baseline_amp, np.nan

    # Verify a meaningful CR peak exists in the search window
    search_full_idx = (time >= t_led) & (time <= search_end)
    if not np.any(search_full_idx):
        return np.nan, np.nan, baseline_amp, np.nan
    peak_val = float(np.nanmax(signal_bs[search_full_idx]))
    if not np.isfinite(peak_val) or peak_val < min_peak_height:
        return np.nan, np.nan, baseline_amp, peak_val

    # Smooth and compute velocity on the full trace
    sig_smooth = smooth_trace(signal_bs, window=velocity_smooth_window)
    velocity = np.gradient(sig_smooth, time)

    # Velocity threshold from baseline period
    vel_baseline = velocity[baseline_idx]
    vel_baseline = vel_baseline[np.isfinite(vel_baseline)]
    if vel_baseline.size > 1:
        vel_threshold = np.nanmean(vel_baseline) + velocity_std_factor * np.nanstd(vel_baseline)
    else:
        vel_threshold = 0.0
    vel_threshold = max(vel_threshold, velocity_floor)

    # Forward scan from min_onset_latency: find first sustained rise
    onset_search_idx = (time >= (t_led + min_onset_latency_s)) & (time <= search_end)
    if not np.any(onset_search_idx):
        return np.nan, np.nan, baseline_amp, peak_val

    time_s = time[onset_search_idx]
    vel_s = velocity[onset_search_idx]
    sig_s = sig_smooth[onset_search_idx]

    onset_idx = None
    run_start = None
    for i in range(len(vel_s)):
        if vel_s[i] >= vel_threshold and sig_s[i] >= 0:
            if run_start is None:
                run_start = i
            if (i - run_start + 1) >= min_sustain_samples:
                onset_idx = run_start
                break
        else:
            run_start = None

    if onset_idx is None:
        return np.nan, np.nan, baseline_amp, peak_val

    onset_time = time_s[onset_idx]
    onset_latency = onset_time - t_led
    return onset_time, onset_latency, baseline_amp, peak_val


def detect_onset_on_mean_trace(t_grid, mean_trace_bs, block_label,
                                smooth_window=3, vel_std_factor=2.0,
                                vel_floor=0.01, min_sustain=2,
                                min_onset_latency_s=0.050,
                                short_end_s=0.212, long_end_s=0.412):
    """Velocity-based onset detection on the noise-averaged mean trace.

    Averaging hundreds of trials reduces noise by ~sqrt(N), so a very low
    velocity floor is sufficient to catch the true first rise without false
    positives. This gives an onset that corresponds to where the mean trace
    visually begins to deviate from baseline.
    """
    t = np.asarray(t_grid, dtype=float)
    sig = np.asarray(mean_trace_bs, dtype=float)
    bl_idx = (t >= -0.2) & (t <= 0.0)
    sig_smooth = smooth_trace(sig, window=smooth_window)
    velocity = np.gradient(sig_smooth, t)
    vel_baseline = velocity[bl_idx]
    vel_baseline = vel_baseline[np.isfinite(vel_baseline)]
    if vel_baseline.size > 1:
        vel_threshold = np.nanmean(vel_baseline) + vel_std_factor * np.nanstd(vel_baseline)
    else:
        vel_threshold = 0.0
    vel_threshold = max(vel_threshold, vel_floor)
    search_end = short_end_s if str(block_label).strip().lower() == "short" else long_end_s
    onset_search_idx = (t >= min_onset_latency_s) & (t <= search_end)
    if not np.any(onset_search_idx):
        return np.nan
    t_s = t[onset_search_idx]
    vel_s = velocity[onset_search_idx]
    sig_s = sig_smooth[onset_search_idx]
    if np.nanmax(sig_s) < 0.01:
        return np.nan
    onset_idx = None
    run_start = None
    for i in range(len(vel_s)):
        if vel_s[i] >= vel_threshold and sig_s[i] >= 0:
            if run_start is None:
                run_start = i
            if (i - run_start + 1) >= min_sustain:
                onset_idx = run_start
                break
        else:
            run_start = None
    return t_s[onset_idx] if onset_idx is not None else np.nan


def moving_average_same(x, window=5):
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) == 0:
        return x.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def hist_fraction(values, edges, smooth_window=1):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    nbins = len(edges) - 1
    if len(values) == 0:
        return np.zeros(nbins, dtype=float)
    counts, _ = np.histogram(values, bins=edges)
    total = np.sum(counts)
    if total <= 0:
        return np.zeros(nbins, dtype=float)
    frac = counts.astype(float) / total
    if smooth_window > 1:
        frac = moving_average_same(frac, window=smooth_window)
        frac = np.clip(frac, 0, None)
        s = np.sum(frac)
        if s > 0:
            frac = frac / s
    return frac


def hist_cumulative(values, edges):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    nbins = len(edges) - 1
    if len(values) == 0:
        return np.zeros(nbins, dtype=float)
    counts, _ = np.histogram(values, bins=edges)
    total = np.sum(counts)
    if total <= 0:
        return np.zeros(nbins, dtype=float)
    return np.cumsum(counts.astype(float) / total)


def style_axis(ax, title, ylabel, xlabel):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(False)


def load_mat_trials(filepath):
    print(f"Loading file: {os.path.basename(filepath)}  010_CR_onsetTiming_Distribution_SanityCheck.py:422 - 010_CR_onsetTiming_ChemoControl.py:418")
    try:
        S = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        print(f"Skipping {os.path.basename(filepath)}: loadmat failed ({exc})  010_CR_onsetTiming_Distribution_SanityCheck.py:426 - 010_CR_onsetTiming_ChemoControl.py:422")
        return [], 0
    if "SessionData" not in S:
        return [], 0
    SD = S["SessionData"]
    raw_events = get_field(SD, "RawEvents", None)
    trials = ensure_trial_list(get_field(raw_events, "Trial", None))
    return trials, is_chemogenetics_session(SD)


def prepare_trial_records(data_files):
    records = []
    t_pre = 0.2
    t_post = 2.5
    dt = 1.0 / 250.0
    t_grid = np.arange(-t_pre, t_post + dt / 2, dt)
    short_isi_max = 0.30
    smooth_win = 1  # no FEC pre-smoothing — large windows shift apparent onset timing
    good_cr_threshold = 0.05
    poor_cr_threshold = 0.02
    short_cr_pre_ms = 25
    short_cr_post_ms = 10
    long_cr_pre_ms = 50
    long_cr_post_ms = 10
    short_onset_end_s = 0.212
    long_onset_end_s = 0.412
    min_peak_height_good = 0.03
    min_peak_height_poor = 0.02
    min_peak_height_no = 0.005
    for filepath in data_files:
        trials, is_chemo = load_mat_trials(filepath)
        if len(trials) == 0:
            continue

        overall_max = collect_overall_max(trials)
        if not np.isfinite(overall_max):
            continue
        for trial_index, tr in enumerate(trials, start=1):
            states = get_field(tr, "States", None)
            events = get_field(tr, "Events", None)
            data = get_field(tr, "Data", None)
            if states is None or events is None or data is None:
                continue
            if has_field(states, "CheckEyeOpenTimeout"):
                timeout_val = safe_array(get_field(states, "CheckEyeOpenTimeout", np.nan))
                if np.any(np.isfinite(timeout_val)):
                    continue
            if has_field(data, "IsProbeTrial"):
                is_probe = np.asarray(get_field(data, "IsProbeTrial", 0)).squeeze()
                if np.any(is_probe == 1):
                    continue
            FECTimes = safe_array(get_field(data, "FECTimes", None))
            fec_from_file = safe_array(get_field(data, "FEC", None))
            eyeAreaPixels = safe_array(get_field(data, "eyeAreaPixels", None))
            GT1_start = get_field(events, "GlobalTimer1_Start", None)
            GT2_start = get_field(events, "GlobalTimer2_Start", None)
            LED_Puff_ISI = safe_array(get_field(states, "LED_Puff_ISI", None))
            if FECTimes.size == 0 or (fec_from_file.size == 0 and eyeAreaPixels.size == 0):
                continue
            if fec_from_file.size > 0 and FECTimes.size != fec_from_file.size:
                continue
            if fec_from_file.size == 0 and FECTimes.size != eyeAreaPixels.size:
                continue
            if LED_Puff_ISI.size < 2:
                continue
            if GT1_start is None or GT2_start is None:
                continue
            t_led_abs = scalar_or_nan(GT1_start)
            t_puff_abs = scalar_or_nan(GT2_start)
            if not (np.isfinite(t_led_abs) and np.isfinite(t_puff_abs)):
                continue
            if fec_from_file.size > 0:
                FEC = fec_from_file.astype(float)
            else:
                FEC = 1.0 - eyeAreaPixels / overall_max
            t_rel = FECTimes - t_led_abs
            if smooth_win > 1:
                FEC = smooth_trace(FEC, smooth_win)
            Fq = interp_to_grid(t_rel, FEC, t_grid)
            isi_dur = float(LED_Puff_ISI[1] - LED_Puff_ISI[0])
            block_label = "short" if isi_dur <= short_isi_max else "long"
            category, cr_peak_bs, baseline_amp = classify_trial(
                time=t_grid,
                signal=Fq,
                t_led=0.0,
                t_puff=t_puff_abs - t_led_abs,
                block_label=block_label,
                good_cr_threshold=good_cr_threshold,
                poor_cr_threshold=poor_cr_threshold,
                short_pre_ms=short_cr_pre_ms,
                short_post_ms=short_cr_post_ms,
                long_pre_ms=long_cr_pre_ms,
                long_post_ms=long_cr_post_ms,
            )
            if category == "Good CR":
                min_peak_height = min_peak_height_good
            elif category == "Poor CR":
                min_peak_height = min_peak_height_poor
            else:
                min_peak_height = min_peak_height_no
            onset_time, onset_latency, _, peak_val = find_detected_onset_velocity_based(
                time=t_grid,
                signal=Fq,
                block_label=block_label,
                t_led=0.0,
                min_peak_height=min_peak_height,
                baseline_window=(-0.2, 0.0),
                short_end_s=short_onset_end_s,
                long_end_s=long_onset_end_s,
            )
            records.append({
                "filepath": filepath,
                "session": os.path.basename(filepath),
                "trial_index": trial_index,
                "block_label": block_label,
                "category": category,
                "t_grid": t_grid,
                "Fq": Fq,
                "t_puff": t_puff_abs - t_led_abs,
                "onset_time": onset_time,
                "onset_latency": onset_latency,
                "peak_val": peak_val,
                "baseline_amp": baseline_amp,
                "good_cr_threshold": good_cr_threshold,
                "poor_cr_threshold": poor_cr_threshold,
                "is_chemo": is_chemo,
            })
    return records


def build_session_range(data_files):
    dates = []
    for path in data_files:
        m = re.search(r"\d{8}", os.path.basename(path))
        if m:
            try:
                dates.append(datetime.strptime(m.group(), "%Y%m%d"))
            except Exception:
                pass
    if not dates:
        return "Unknown", "Unknown", "Unknown", "Unknown"
    first = min(dates)
    last = max(dates)
    return first.strftime("%Y%m%d"), last.strftime("%Y%m%d"), first.strftime("%m/%d/%Y"), last.strftime("%m/%d/%Y")


def _median_onset(records, block, category):
    vals = [r["onset_latency"] for r in records if r["block_label"] == block and r["category"] == category and np.isfinite(r.get("onset_latency", np.nan))]
    return np.nanmedian(vals) if vals else np.nan, len(vals)


def plot_onset_distributions(records, first_date, last_date, first_tok, last_tok):
    onset_edges = np.arange(0.0, 0.4501, 0.01)
    onset_centers = onset_edges[:-1] + np.diff(onset_edges) / 2
    hist_smooth_window = 3

    ctrl_recs  = [r for r in records if not r.get("is_chemo", False)]
    chemo_recs = [r for r in records if r.get("is_chemo", False)]

    # ctrl: black family; chemo: blue family; linestyle solid=ctrl, dashed=chemo
    cat_colors = {
        "Good CR": (CTRL_COLOR,  CHEMO_COLOR),
        "Poor CR": (CTRL_POOR,   CHEMO_POOR),
        "No CR":   (CTRL_NOCR,   CHEMO_NOCR),
    }

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, block in zip(axs, ["short", "long"]):
        for category in ["Good CR", "Poor CR", "No CR"]:
            ctrl_col, chemo_col = cat_colors[category]
            for recs, col, ls, tag in [
                (ctrl_recs,  ctrl_col,  "-",  "Ctrl"),
                (chemo_recs, chemo_col, "--", "Chemo"),
            ]:
                vals = [r["onset_latency"] for r in recs if r["block_label"] == block and r["category"] == category]
                n = len(vals)
                frac = hist_fraction(vals, onset_edges, smooth_window=hist_smooth_window)
                ax.plot(onset_centers, frac, color=col, linestyle=ls, linewidth=2.0,
                        label=f"{category} {tag} (n={n})")
        title = "Short Block" if block == "short" else "Long Block"
        style_axis(ax, f"{title} onset distribution\nGood/Poor/No CR — ctrl (solid) vs chemo (dashed)", "Fraction of detected onsets", "Detected onset (s)")
        ax.set_xlim(0, 0.22 if block == "short" else 0.42)
        ax.set_ylim(bottom=0)
        ax.axvspan(0.0, 0.05, color="gray", alpha=0.12)
        puff_x = 0.200 if block == "short" else 0.400
        ax.axvline(puff_x, linestyle="--", color="black", alpha=0.8)
        ax.text(0.025, ax.get_ylim()[1] * 0.92, "LED", fontsize=9)
        ax.text(puff_x + 0.01, ax.get_ylim()[1] * 0.92, "Airpuff", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"CR onset timing distributions (ctrl=solid/black  chemo=dashed/blue)\nSessions: {first_date} -- {last_date}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_name = f"CR_onset_distribution_sanity_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  010_CR_onsetTiming_Distribution_SanityCheck.py:621 - 010_CR_onsetTiming_ChemoControl.py:629")
    return fig


def plot_good_cr_onset_distribution(records, first_date, last_date, first_tok, last_tok):
    onset_edges = np.arange(0.0, 0.4501, 0.01)
    onset_centers = onset_edges[:-1] + np.diff(onset_edges) / 2

    ctrl_recs  = [r for r in records if not r.get("is_chemo", False)]
    chemo_recs = [r for r in records if r.get("is_chemo", False)]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex="col")
    for col, block in enumerate(["short", "long"]):
        puff_x = 0.200 if block == "short" else 0.400

        # ── Row 0: fraction distribution ──────────────────────────────
        ax = axs[0, col]
        for recs, col_c, label in [
            (ctrl_recs,  CTRL_COLOR,  "Control"),
            (chemo_recs, CHEMO_COLOR, "Chemo"),
        ]:
            vals = [r["onset_latency"] for r in recs if r["block_label"] == block and r["category"] == "Good CR"]
            n = len(vals)
            frac = hist_fraction(vals, onset_edges, smooth_window=3)
            med, _ = _median_onset(recs, block, "Good CR")
            ax.plot(onset_centers, frac, color=col_c, linewidth=2.2, label=f"{label} n={n}")
            if np.isfinite(med):
                ax.axvline(med, color=col_c, linestyle="--", linewidth=1.5,
                           label=f"{label} median {med:.3f}s")
        style_axis(ax, f"{block.title()} block — Good CR  [ctrl=black | chemo=blue]",
                   "Fraction of detected onsets", "Detected onset (s)")
        ax.set_xlim(0, 0.42)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.18)
        ax.axvline(puff_x, linestyle="--", color="black", alpha=0.8)
        ax.text(puff_x + 0.01, ax.get_ylim()[1] * 0.90, "Puff", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)

        # ── Row 1: cumulative ─────────────────────────────────────────
        ax = axs[1, col]
        for recs, col_c, label in [
            (ctrl_recs,  CTRL_COLOR,  "Control"),
            (chemo_recs, CHEMO_COLOR, "Chemo"),
        ]:
            vals = [r["onset_latency"] for r in recs if r["block_label"] == block and r["category"] == "Good CR"]
            n = len(vals)
            cum = hist_cumulative(vals, onset_edges)
            med, _ = _median_onset(recs, block, "Good CR")
            ax.plot(onset_centers, cum, color=col_c, linewidth=2.2, label=f"{label} n={n}")
            if np.isfinite(med):
                ax.axvline(med, color=col_c, linestyle="--", linewidth=1.5)
        style_axis(ax, f"{block.title()} block cumulative",
                   "Cumulative fraction", "Detected onset (s)")
        ax.set_xlim(0, 0.42)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.set_ylim(0, 1.0)
        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.18)
        ax.axvline(puff_x, linestyle="--", color="black", alpha=0.8)
        ax.text(puff_x + 0.01, 0.85, "Puff", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Good CR onset distributions — Control (black) vs Chemo (blue)\nSessions: {first_date} -- {last_date}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_name = f"CR_onset_goodCR_distribution_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  010_CR_onsetTiming_Distribution_SanityCheck.py:690 - 010_CR_onsetTiming_ChemoControl.py:698")
    return fig


def plot_poor_cr_onset_distribution(records, first_date, last_date, first_tok, last_tok):
    onset_edges = np.arange(0.0, 0.4501, 0.01)
    onset_centers = onset_edges[:-1] + np.diff(onset_edges) / 2

    ctrl_recs  = [r for r in records if not r.get("is_chemo", False)]
    chemo_recs = [r for r in records if r.get("is_chemo", False)]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, block in zip(axs, ["short", "long"]):
        puff_x = 0.200 if block == "short" else 0.400
        for recs, col_c, label in [
            (ctrl_recs,  CTRL_COLOR,  "Control"),
            (chemo_recs, CHEMO_COLOR, "Chemo"),
        ]:
            vals = [r["onset_latency"] for r in recs if r["block_label"] == block and r["category"] == "Poor CR"]
            n = len(vals)
            frac = hist_fraction(vals, onset_edges, smooth_window=3)
            med = np.nanmedian(vals) if n else np.nan
            ax.plot(onset_centers, frac, color=col_c, linewidth=2.2, label=f"{label} n={n}")
            if np.isfinite(med):
                ax.axvline(med, color=col_c, linestyle="--", linewidth=1.5,
                           label=f"{label} median {med:.3f}s")
        style_axis(ax, f"{block.title()} block — Poor CR  [ctrl=black | chemo=blue]",
                   "Fraction of detected onsets", "Detected onset (s)")
        ax.set_xlim(0, 0.42)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.18)
        ax.axvline(puff_x, linestyle="--", color="black", alpha=0.8)
        ax.text(puff_x + 0.01, ax.get_ylim()[1] * 0.90, "Puff", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Poor CR onset distributions — Control (black) vs Chemo (blue)\nSessions: {first_date} -- {last_date}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_name = f"CR_onset_poorCR_distribution_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  010_CR_onsetTiming_Distribution_SanityCheck.py:732 - 010_CR_onsetTiming_ChemoControl.py:740")
    return fig


def plot_superimposed_onset_distribution(records, first_date, last_date, first_tok, last_tok):
    onset_edges = np.arange(0.0, 0.4501, 0.01)
    onset_centers = onset_edges[:-1] + np.diff(onset_edges) / 2

    ctrl_recs  = [r for r in records if not r.get("is_chemo", False)]
    chemo_recs = [r for r in records if r.get("is_chemo", False)]

    # ctrl-short=black solid, ctrl-long=black dashed, chemo-short=blue solid, chemo-long=blue dashed
    combos = [
        (ctrl_recs,  "short", CTRL_COLOR,  "-",  "Ctrl Short"),
        (ctrl_recs,  "long",  CTRL_COLOR,  "--", "Ctrl Long"),
        (chemo_recs, "short", CHEMO_COLOR, "-",  "Chemo Short"),
        (chemo_recs, "long",  CHEMO_COLOR, "--", "Chemo Long"),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, category in zip(axs, ["Good CR", "Poor CR"]):
        for recs, block, col_c, ls, tag in combos:
            vals = [r["onset_latency"] for r in recs if r["block_label"] == block and r["category"] == category]
            n = len(vals)
            frac = hist_fraction(vals, onset_edges, smooth_window=3)
            med = np.nanmedian(vals) if n else np.nan
            lbl = f"{tag} n={n}"
            if np.isfinite(med):
                lbl += f" med={med:.3f}s"
            ax.plot(onset_centers, frac, color=col_c, linestyle=ls, linewidth=2.0, label=lbl)
        style_axis(ax, f"{category}: ctrl (black) vs chemo (blue), short (solid) vs long (dashed)",
                   "Fraction of detected onsets", "Detected onset (s)")
        ax.set_xlim(0, 0.42)
        ax.set_ylim(bottom=0)
        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.18)
        ax.axvline(0.200, linestyle=":", color="royalblue", alpha=0.9, label="Short puff")
        ax.axvline(0.400, linestyle=":", color="seagreen", alpha=0.9, label="Long puff")
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"Superimposed CR onset — Control (black) vs Chemo (blue)\nSessions: {first_date} -- {last_date}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_name = f"CR_onset_superimposed_short_long_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  010_CR_onsetTiming_Distribution_SanityCheck.py:777 - 010_CR_onsetTiming_ChemoControl.py:785")
    return fig


def plot_good_cr_cumulative_comparison(records, first_date, last_date, first_tok, last_tok):
    onset_edges = np.arange(0.0, 0.4501, 0.005)
    onset_centers = onset_edges[:-1] + np.diff(onset_edges) / 2

    ctrl_recs  = [r for r in records if not r.get("is_chemo", False)]
    chemo_recs = [r for r in records if r.get("is_chemo", False)]

    block_styles = {
        "short": {"puff_x": 0.200, "puff_color": "steelblue"},
        "long":  {"puff_x": 0.400, "puff_color": "darkgreen"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for block in ["short", "long"]:
        s = block_styles[block]
        for recs, col_c, ls, tag in [
            (ctrl_recs,  CTRL_COLOR,  "-",  "Ctrl"),
            (chemo_recs, CHEMO_COLOR, "--", "Chemo"),
        ]:
            vals = [r["onset_latency"] for r in recs if r["block_label"] == block and r["category"] == "Good CR"]
            n = len(vals)
            cum = hist_cumulative(vals, onset_edges)
            ax.plot(onset_centers, cum, color=col_c, linestyle=ls, linewidth=2.2,
                    label=f"{block.title()} {tag} Good CR (n={n})")
        ax.axvspan(s["puff_x"], s["puff_x"] + 0.020, color=s["puff_color"], alpha=0.15)
        ax.axvline(s["puff_x"], linestyle="--", color=s["puff_color"], alpha=0.85)
        ax.text(s["puff_x"] + 0.005, 0.96, "AirPuff", color=s["puff_color"], fontsize=9, va="top")

    ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.35)
    ax.axvline(0.0, linestyle=":", color="gray", alpha=0.7)
    ax.text(0.002, 0.03, "LED", fontsize=9, color="gray")
    ax.set_xlabel("Detected onset (s)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(0.0, 0.45)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.legend(loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    fig.suptitle(
        f"Good CR cumulative onset — Control (black) vs Chemo (blue)\nSessions: {first_date} -- {last_date}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_name = f"CR_onset_goodCR_cumulative_comparison_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  010_CR_onsetTiming_Distribution_SanityCheck.py:829 - 010_CR_onsetTiming_ChemoControl.py:837")
    return fig


def plot_good_cr_average_traces(records, first_tok, last_tok):
    ctrl_recs  = [r for r in records if not r.get("is_chemo", False)]
    chemo_recs = [r for r in records if r.get("is_chemo", False)]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for col, block_label in enumerate(["short", "long"]):
        ax = axs[col]
        puff_x = 0.200 if block_label == "short" else 0.400
        shade_color = "blue" if block_label == "short" else "green"
        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.35)
        ax.axvspan(puff_x, puff_x + 0.020, color=shade_color, alpha=0.20)
        ax.axvline(puff_x, linestyle="--", color="black", alpha=0.8)

        for recs, col_c, label in [
            (ctrl_recs,  CTRL_COLOR,  "Control"),
            (chemo_recs, CHEMO_COLOR, "Chemo"),
        ]:
            subset = [r for r in recs if r["block_label"] == block_label and r["category"] == "Good CR"]
            if not subset:
                continue
            all_traces = np.vstack([r["Fq"] for r in subset])
            mean_trace = np.nanmean(all_traces, axis=0)
            onset_vals = [r["onset_time"] for r in subset if np.isfinite(r["onset_time"])]
            med_onset = np.nanmedian(onset_vals) if onset_vals else np.nan
            n = len(subset)
            ax.plot(subset[0]["t_grid"], mean_trace, color=col_c, linewidth=2,
                    label=f"{label} (n={n})")
            if np.isfinite(med_onset):
                ax.axvline(med_onset, color=col_c, linestyle="--", linewidth=1.5,
                           label=f"{label} onset {med_onset:.3f}s")

        ax.set_title(f"{block_label.title()} block — Good CR  [ctrl=black | chemo=blue]")
        ax.set_xlim(-0.1, 2.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time from LED onset (s)")
        ax.set_ylabel("FEC")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Average Good CR FEC traces — Control (black) vs Chemo (blue)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_name = f"CR_onset_goodCR_average_traces_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  010_CR_onsetTiming_Distribution_SanityCheck.py:875 - 010_CR_onsetTiming_ChemoControl.py:883")
    return fig


def plot_good_cr_average_traces_baselined(records, first_tok, last_tok):
    """Baseline-subtracted Good CR mean traces.

    Layout (2 rows × 2 cols):
      Row 0 — full view  : −0.1 to 2.5 s
      Row 1 — zoomed ISI : −0.05 to just past the airpuff marker

    Onset dashed lines are computed from the mean trace itself (not per-trial
    median) so they accurately reflect where the averaged signal first rises.
    """
    ctrl_recs  = [r for r in records if not r.get("is_chemo", False)]
    chemo_recs = [r for r in records if r.get("is_chemo", False)]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    for col, block_label in enumerate(["short", "long"]):
        puff_x   = 0.200 if block_label == "short" else 0.400
        shade_col = "cornflowerblue" if block_label == "short" else "mediumseagreen"
        zoom_end  = 0.26 if block_label == "short" else 0.46

        ax_full = axs[0, col]
        ax_zoom = axs[1, col]

        for ax in (ax_full, ax_zoom):
            ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.35)
            ax.axvspan(puff_x, puff_x + 0.020, color=shade_col, alpha=0.22)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)

        for recs, col_c, label in [
            (ctrl_recs,  CTRL_COLOR,  "Control"),
            (chemo_recs, CHEMO_COLOR, "Chemo"),
        ]:
            subset = [r for r in recs if r["block_label"] == block_label and r["category"] == "Good CR"]
            if not subset:
                continue
            t_grid = subset[0]["t_grid"]
            bl_idx = (t_grid >= -0.2) & (t_grid <= 0.0)
            corrected = []
            for r in subset:
                bl = np.nanmean(r["Fq"][bl_idx])
                corrected.append(r["Fq"] - (bl if np.isfinite(bl) else 0.0))
            mean_trace = np.nanmean(np.vstack(corrected), axis=0)
            n = len(subset)

            # Detect onset directly on the mean trace — far lower noise than
            # individual trials, so velocity threshold catches the true first rise
            onset_time = detect_onset_on_mean_trace(
                t_grid, mean_trace, block_label,
                smooth_window=3, vel_std_factor=2.0, vel_floor=0.01,
                min_sustain=2, min_onset_latency_s=0.050,
            )

            for ax in (ax_full, ax_zoom):
                ax.plot(t_grid, mean_trace, color=col_c, linewidth=2,
                        label=f"{label} (n={n})")
                if np.isfinite(onset_time):
                    ax.axvline(onset_time, color=col_c, linestyle="--", linewidth=1.5,
                               label=f"{label} onset {onset_time:.3f}s")

        # ── full view ────────────────────────────────────────────────
        ax_full.set_title(
            f"{block_label.title()} block — Good CR  baseline corrected  [ctrl=black | chemo=blue]"
        )
        ax_full.set_xlim(-0.1, 2.5)
        ax_full.set_xlabel("Time from LED onset (s)")
        ax_full.set_ylabel("ΔFEC (baseline subtracted)")
        ax_full.legend(loc="upper right", fontsize=8)

        # ── zoomed ISI view ──────────────────────────────────────────
        ax_zoom.set_title(
            f"{block_label.title()} block — zoomed  [0 → {zoom_end:.2f} s]"
        )
        ax_zoom.set_xlim(-0.05, zoom_end)
        ax_zoom.set_xlabel("Time from LED onset (s)")
        ax_zoom.set_ylabel("ΔFEC (baseline subtracted)")
        ax_zoom.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Average Good CR FEC traces (baseline subtracted) — Control (black) vs Chemo (blue)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_name = f"CR_onset_goodCR_average_traces_baselined_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}")
    return fig


def plot_poor_cr_average_traces_baselined(records, first_tok, last_tok):
    """Baseline-subtracted Poor CR mean traces — same 2×2 layout as Good CR version.

    Row 0: full view (−0.1 to 2.5 s)
    Row 1: zoomed ISI view
    Onset detected from the mean trace via velocity threshold.
    """
    ctrl_recs  = [r for r in records if not r.get("is_chemo", False)]
    chemo_recs = [r for r in records if r.get("is_chemo", False)]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    for col, block_label in enumerate(["short", "long"]):
        puff_x    = 0.200 if block_label == "short" else 0.400
        shade_col = "cornflowerblue" if block_label == "short" else "mediumseagreen"
        zoom_end  = 0.26 if block_label == "short" else 0.46

        ax_full = axs[0, col]
        ax_zoom = axs[1, col]

        for ax in (ax_full, ax_zoom):
            ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.35)
            ax.axvspan(puff_x, puff_x + 0.020, color=shade_col, alpha=0.22)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)

        for recs, col_c, label in [
            (ctrl_recs,  CTRL_COLOR,  "Control"),
            (chemo_recs, CHEMO_COLOR, "Chemo"),
        ]:
            subset = [r for r in recs if r["block_label"] == block_label and r["category"] == "Poor CR"]
            if not subset:
                continue
            t_grid = subset[0]["t_grid"]
            bl_idx = (t_grid >= -0.2) & (t_grid <= 0.0)
            corrected = []
            for r in subset:
                bl = np.nanmean(r["Fq"][bl_idx])
                corrected.append(r["Fq"] - (bl if np.isfinite(bl) else 0.0))
            mean_trace = np.nanmean(np.vstack(corrected), axis=0)
            n = len(subset)

            onset_time = detect_onset_on_mean_trace(
                t_grid, mean_trace, block_label,
                smooth_window=3, vel_std_factor=2.0, vel_floor=0.01,
                min_sustain=2, min_onset_latency_s=0.050,
            )

            for ax in (ax_full, ax_zoom):
                ax.plot(t_grid, mean_trace, color=col_c, linewidth=2,
                        label=f"{label} (n={n})")
                if np.isfinite(onset_time):
                    ax.axvline(onset_time, color=col_c, linestyle="--", linewidth=1.5,
                               label=f"{label} onset {onset_time:.3f}s")

        ax_full.set_title(
            f"{block_label.title()} block — Poor CR  baseline corrected  [ctrl=black | chemo=blue]"
        )
        ax_full.set_xlim(-0.1, 2.5)
        ax_full.set_xlabel("Time from LED onset (s)")
        ax_full.set_ylabel("ΔFEC (baseline subtracted)")
        ax_full.legend(loc="upper right", fontsize=8)

        ax_zoom.set_title(
            f"{block_label.title()} block — zoomed  [0 → {zoom_end:.2f} s]"
        )
        ax_zoom.set_xlim(-0.05, zoom_end)
        ax_zoom.set_xlabel("Time from LED onset (s)")
        ax_zoom.set_ylabel("ΔFEC (baseline subtracted)")
        ax_zoom.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Average Poor CR FEC traces (baseline subtracted) — Control (black) vs Chemo (blue)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_name = f"CR_onset_poorCR_average_traces_baselined_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}")
    return fig


def plot_poor_cr_average_traces(records, first_tok, last_tok):
    ctrl_recs  = [r for r in records if not r.get("is_chemo", False)]
    chemo_recs = [r for r in records if r.get("is_chemo", False)]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for col, block_label in enumerate(["short", "long"]):
        ax = axs[col]
        puff_x = 0.200 if block_label == "short" else 0.400
        shade_color = "blue" if block_label == "short" else "green"
        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.35)
        ax.axvspan(puff_x, puff_x + 0.020, color=shade_color, alpha=0.20)
        ax.axvline(puff_x, linestyle="--", color="black", alpha=0.8)

        for recs, col_c, label in [
            (ctrl_recs,  CTRL_COLOR,  "Control"),
            (chemo_recs, CHEMO_COLOR, "Chemo"),
        ]:
            subset = [r for r in recs if r["block_label"] == block_label and r["category"] == "Poor CR"]
            if not subset:
                continue
            all_traces = np.vstack([r["Fq"] for r in subset])
            mean_trace = np.nanmean(all_traces, axis=0)
            onset_vals = [r["onset_time"] for r in subset if np.isfinite(r["onset_time"])]
            med_onset = np.nanmedian(onset_vals) if onset_vals else np.nan
            n = len(subset)
            ax.plot(subset[0]["t_grid"], mean_trace, color=col_c, linewidth=2,
                    label=f"{label} (n={n})")
            if np.isfinite(med_onset):
                ax.axvline(med_onset, color=col_c, linestyle="--", linewidth=1.5,
                           label=f"{label} onset {med_onset:.3f}s")

        ax.set_title(f"{block_label.title()} block — Poor CR  [ctrl=black | chemo=blue]")
        ax.set_xlim(-0.1, 2.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time from LED onset (s)")
        ax.set_ylabel("FEC")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Average Poor CR FEC traces — Control (black) vs Chemo (blue)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_name = f"CR_onset_poorCR_average_traces_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  010_CR_onsetTiming_Distribution_SanityCheck.py:921 - 010_CR_onsetTiming_ChemoControl.py:929")
    return fig


def plot_latency_summary_table(records, first_date, last_date, first_tok, last_tok):
    ctrl_recs  = [r for r in records if not r.get("is_chemo", False)]
    chemo_recs = [r for r in records if r.get("is_chemo", False)]

    def summarize(recs, block_label):
        values = [
            r["onset_latency"]
            for r in recs
            if r["block_label"] == block_label and r["category"] == "Good CR" and np.isfinite(r["onset_latency"])
        ]
        n = len(values)
        median = np.nanmedian(values) if n else np.nan
        p25 = np.nanpercentile(values, 25) if n else np.nan
        p75 = np.nanpercentile(values, 75) if n else np.nan
        return n, median, p25, p75

    rows = []
    for block_label in ["short", "long"]:
        for recs, tag in [(ctrl_recs, "Control"), (chemo_recs, "Chemo")]:
            n, median, p25, p75 = summarize(recs, block_label)
            rows.append([
                f"{block_label.title()} [{tag}]",
                str(n),
                f"{median:.3f}" if np.isfinite(median) else "NaN",
                f"{p25:.3f}" if np.isfinite(p25) else "NaN",
                f"{p75:.3f}" if np.isfinite(p75) else "NaN",
            ])

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Block [Group]", "Good CR n", "Median onset (s)", "25th pctile", "75th pctile"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    fig.suptitle(
        f"Good CR onset latency summary — Control vs Chemo\nSessions: {first_date} -- {last_date}",
        fontsize=14,
    )
    out_name = f"CR_onset_latency_summary_table_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  010_CR_onsetTiming_Distribution_SanityCheck.py:970 - 010_CR_onsetTiming_ChemoControl.py:978")
    return fig


def plot_sanity_examples(records, first_tok, last_tok):
    categories = ["Good CR", "Poor CR", "No CR"]
    fig, axs = plt.subplots(len(categories), 2, figsize=(12, 10), squeeze=False)
    for i, category in enumerate(categories):
        subset = [r for r in records if r["category"] == category and np.isfinite(r["onset_time"])]
        for j in range(2):
            ax = axs[i, j]
            if j < len(subset):
                r = subset[j]
                group_tag = "Chemo" if r.get("is_chemo", False) else "Control"
                ax.plot(r["t_grid"], r["Fq"], color="black", linewidth=1.5)
                ax.axvline(r["onset_time"], color="red", linestyle="--", label="Detected onset")
                ax.set_title(
                    f"{category} example {j + 1} [{group_tag}]\n{os.path.basename(r['filepath'])}"
                )
                ax.set_xlim(-0.1, 2.5)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlabel("Time from LED onset (s)")
                ax.set_ylabel("FEC")
                ax.legend(loc="best")
            else:
                ax.axis("off")
    fig.suptitle("Sanity-check example trials with detected CR onset", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_name = f"CR_onset_sanity_examples_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  010_CR_onsetTiming_Distribution_SanityCheck.py:1000 - 010_CR_onsetTiming_ChemoControl.py:1008")
    return fig


def plot_average_traces(records, first_tok, last_tok):
    ctrl_recs  = [r for r in records if not r.get("is_chemo", False)]
    chemo_recs = [r for r in records if r.get("is_chemo", False)]

    categories = ["Good CR", "Poor CR", "No CR"]
    fig, axs = plt.subplots(3, 2, figsize=(16, 9), sharex=True, sharey=True)
    for col, block_label in enumerate(["short", "long"]):
        for row, category in enumerate(categories):
            ax = axs[row, col]
            for recs, col_c, label in [
                (ctrl_recs,  CTRL_COLOR,  "Control"),
                (chemo_recs, CHEMO_COLOR, "Chemo"),
            ]:
                subset = [r for r in recs if r["block_label"] == block_label and r["category"] == category]
                if not subset:
                    continue
                all_traces = np.vstack([r["Fq"] for r in subset])
                mean_trace = np.nanmean(all_traces, axis=0)
                onset_vals = [r["onset_time"] for r in subset if np.isfinite(r["onset_time"])]
                med_onset = np.nanmedian(onset_vals) if onset_vals else np.nan
                ax.plot(subset[0]["t_grid"], mean_trace, color=col_c, linewidth=2,
                        label=f"{label} (n={len(subset)})")
                if np.isfinite(med_onset):
                    ax.axvline(med_onset, color=col_c, linestyle="--", linewidth=1.3)
            ax.set_title(f"{block_label.title()} / {category}")
            ax.set_xlim(-0.1, 2.5)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Time from LED onset (s)")
            ax.set_ylabel("FEC")
            ax.legend(loc="upper left", fontsize=7)
    fig.suptitle("Average FEC traces — Control (black) vs Chemo (blue)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_name = f"CR_onset_sanity_average_traces_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  010_CR_onsetTiming_Distribution_SanityCheck.py:1038 - 010_CR_onsetTiming_ChemoControl.py:1046")
    return fig


def main():
    data_files = sorted(glob.glob("*_EBC_*.mat"))
    if len(data_files) == 0:
        print("No *_EBC_*.mat files found in current folder.")
        return
    first_tok, last_tok, first_date, last_date = build_session_range(data_files)
    records = prepare_trial_records(data_files)
    if len(records) == 0:
        print("No usable trials found.")
        return
    plot_good_cr_average_traces_baselined(records, first_tok, last_tok)
    plot_poor_cr_average_traces_baselined(records, first_tok, last_tok)
    plt.close("all")


if __name__ == "__main__":
    main()
