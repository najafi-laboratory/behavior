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
    """
    Classify trials using the current Good/Poor timing definition.

    CR+ if max pre-US FEC minus baseline > 0.02.
    Poor CR only if the pre-US max drops by > 0.05 before AirPuff.
    Otherwise CR+ is Good CR.
    """
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
    velocity_smooth_window=3,  # centered: 1 sample before + 1 after
    velocity_std_factor=1.5,
    velocity_floor=0.03,
    min_sustain_s=0.008,
    backtrack_velocity_floor=0.006,
    min_signal_rise=0.006,
    min_onset_latency_s=0.050,
    long_block_onset_fraction=0.15,
):
    """
    Detect CR onset from the FEC velocity trace.

    The detector first finds a sustained positive velocity before the CR peak,
    then walks backward to the earliest local start of the upward curve. This
    places onset near the visible bend in the FEC trace instead of after the
    trace has already risen substantially.
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
    search_idx = (time >= t_led) & (time <= search_end)
    if not np.any(search_idx):
        return np.nan, np.nan, baseline_amp, np.nan
    time_search = time[search_idx]
    sig_search = signal_bs[search_idx]
    if len(time_search) < 2 or not np.any(np.isfinite(sig_search)):
        return np.nan, np.nan, baseline_amp, np.nan
    local_peaks, _ = find_peaks(sig_search, height=min_peak_height)
    if len(local_peaks) > 0:
        peak_idx_local = int(local_peaks[0])
    else:
        peak_idx_local = int(np.nanargmax(sig_search))
    peak_val = sig_search[peak_idx_local]
    if (not np.isfinite(peak_val)) or (peak_val < min_peak_height):
        return np.nan, np.nan, baseline_amp, peak_val

    sig_smooth = smooth_trace(signal_bs, window=velocity_smooth_window)
    velocity = np.gradient(sig_smooth, time)
    baseline_vel_idx = (time >= (t_led + baseline_window[0])) & (time <= (t_led + baseline_window[1]))
    baseline_velocity = velocity[baseline_vel_idx]
    baseline_velocity = baseline_velocity[np.isfinite(baseline_velocity)]
    if baseline_velocity.size:
        med = np.nanmedian(baseline_velocity)
        mad = np.nanmedian(np.abs(baseline_velocity - med))
        robust_std = 1.4826 * mad if np.isfinite(mad) else np.nan
        if not np.isfinite(robust_std) or robust_std == 0:
            robust_std = np.nanstd(baseline_velocity)
        velocity_threshold = med + velocity_std_factor * robust_std
        velocity_threshold = max(velocity_threshold, velocity_floor)
    else:
        velocity_threshold = velocity_floor

    pre_peak_time = time_search[:peak_idx_local + 1]
    pre_peak_velocity = velocity[search_idx][:peak_idx_local + 1]
    pre_peak_signal = sig_smooth[search_idx][:peak_idx_local + 1]

    # Long-block onset: anchor to the peak near AirPuff then walk backward.
    # The forward velocity-run method fails for long blocks because the FEC
    # rises monotonically over ~400 ms, producing one unbroken velocity run
    # from LED onset to peak, driving detected onset to the LED floor.
    if str(block_label).strip().lower() == "long":
        threshold_signal = long_block_onset_fraction * peak_val
        above_thresh = pre_peak_signal >= threshold_signal
        if not np.any(above_thresh):
            return np.nan, np.nan, baseline_amp, peak_val
        first_above = int(np.argmax(above_thresh))
        onset_idx_back = first_above
        while onset_idx_back > 0:
            v_prev = pre_peak_velocity[onset_idx_back - 1]
            s_prev = pre_peak_signal[onset_idx_back - 1]
            if not (np.isfinite(v_prev) and np.isfinite(s_prev)):
                break
            if v_prev < backtrack_velocity_floor:
                break
            onset_idx_back -= 1
        onset_time = pre_peak_time[onset_idx_back]
        onset_latency = onset_time - t_led
        if onset_latency < min_onset_latency_s:
            return np.nan, np.nan, baseline_amp, peak_val
        return onset_time, onset_latency, baseline_amp, peak_val

    above = (
        np.isfinite(pre_peak_velocity)
        & np.isfinite(pre_peak_signal)
        & (pre_peak_velocity >= velocity_threshold)
        & (pre_peak_signal >= min_signal_rise)
    )

    if len(pre_peak_time) > 1:
        dt = float(np.nanmedian(np.diff(pre_peak_time)))
    else:
        dt = np.nan
    min_sustain_samples = max(8, int(np.ceil(min_sustain_s / dt))) if np.isfinite(dt) and dt > 0 else 8

    idx_candidates = []
    run_start = None
    for idx, is_above in enumerate(above):
        if is_above and run_start is None:
            run_start = idx
        elif (not is_above) and run_start is not None:
            if idx - run_start >= min_sustain_samples:
                idx_candidates.append(run_start)
            run_start = None
    if run_start is not None and len(above) - run_start >= min_sustain_samples:
        idx_candidates.append(run_start)

    if len(idx_candidates) == 0:
        return np.nan, np.nan, baseline_amp, peak_val
    onset_idx = int(idx_candidates[0])
    onset_time = pre_peak_time[onset_idx]
    onset_latency = onset_time - t_led
    if onset_latency < min_onset_latency_s:
        return np.nan, np.nan, baseline_amp, peak_val
    return onset_time, onset_latency, baseline_amp, peak_val


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
    print(f"Loading file: {os.path.basename(filepath)}  CR_onsetTiming_Distribution_SanityCheck.py:378 - 010_CR_onsetTiming_Distribution_SanityCheck.py:434")
    try:
        S = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        print(f"Skipping {os.path.basename(filepath)}: loadmat failed ({exc})  CR_onsetTiming_Distribution_SanityCheck.py:382 - 010_CR_onsetTiming_Distribution_SanityCheck.py:438")
        return []
    if "SessionData" not in S:
        return []
    SD = S["SessionData"]
    raw_events = get_field(SD, "RawEvents", None)
    trials = ensure_trial_list(get_field(raw_events, "Trial", None))
    return trials


def prepare_trial_records(data_files):
    records = []
    t_pre = 0.2
    t_post = 0.6
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
        trials = load_mat_trials(filepath)
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
            eyeAreaPixels = safe_array(get_field(data, "eyeAreaPixels", None))
            GT1_start = get_field(events, "GlobalTimer1_Start", None)
            GT2_start = get_field(events, "GlobalTimer2_Start", None)
            LED_Puff_ISI = safe_array(get_field(states, "LED_Puff_ISI", None))
            if FECTimes.size == 0 or eyeAreaPixels.size == 0:
                continue
            if FECTimes.size != eyeAreaPixels.size:
                continue
            if LED_Puff_ISI.size < 2:
                continue
            if GT1_start is None or GT2_start is None:
                continue
            t_led_abs = scalar_or_nan(GT1_start)
            t_puff_abs = scalar_or_nan(GT2_start)
            if not (np.isfinite(t_led_abs) and np.isfinite(t_puff_abs)):
                continue
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


def plot_onset_distributions(records, first_date, last_date, first_tok, last_tok):
    onset_edges = np.arange(0.0, 0.4501, 0.01)
    onset_centers = onset_edges[:-1] + np.diff(onset_edges) / 2
    hist_smooth_window = 3
    colors = {"Good CR": "black", "Poor CR": "gray", "No CR": "lightgray"}
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, block in zip(axs, ["short", "long"]):
        counts = {category: len([r for r in records if r["block_label"] == block and r["category"] == category]) for category in ["Good CR", "Poor CR", "No CR"]}
        for category in ["Good CR", "Poor CR", "No CR"]:
            values = [r["onset_latency"] for r in records if r["block_label"] == block and r["category"] == category]
            frac = hist_fraction(values, onset_edges, smooth_window=hist_smooth_window)
            ax.plot(onset_centers, frac, label=f"{category} (n={counts[category]})", color=colors[category], linewidth=2.2)
        title = "Short Block" if block == "short" else "Long Block"
        style_axis(ax, f"{title} onset distribution\nGood/Poor/No CR (sanity check)", "Fraction of detected onsets", "Detected onset (s)")
        ax.set_xlim(0, 0.22 if block == "short" else 0.42)
        ax.set_ylim(bottom=0)
        ax.axvspan(0.0, 0.05, color="gray", alpha=0.12)
        puff_x = 0.200 if block == "short" else 0.400
        ax.axvline(puff_x, linestyle="--", color="black", alpha=0.8)
        ax.text(0.025, ax.get_ylim()[1] * 0.92, "LED", fontsize=9)
        ax.text(puff_x + 0.01, ax.get_ylim()[1] * 0.92, "Airpuff", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"CR onset timing distributions (all categories)\nSessions: {first_date} -- {last_date}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_name = f"CR_onset_distribution_sanity_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  CR_onsetTiming_Distribution_SanityCheck.py:546 - 010_CR_onsetTiming_Distribution_SanityCheck.py:605")
    return fig


def plot_good_cr_onset_distribution(records, first_date, last_date, first_tok, last_tok):
    onset_edges = np.arange(0.0, 0.4501, 0.01)
    onset_centers = onset_edges[:-1] + np.diff(onset_edges) / 2
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex='col')
    for col, block in enumerate(["short", "long"]):
        values = [r["onset_latency"] for r in records if r["block_label"] == block and r["category"] == "Good CR"]
        count_good = len(values)
        frac = hist_fraction(values, onset_edges, smooth_window=3)
        median_onset = np.nanmedian(values) if count_good else np.nan
        ax = axs[0, col]
        ax.plot(onset_centers, frac, color="black", linewidth=2.2)
        if np.isfinite(median_onset):
            ax.axvline(median_onset, color="red", linestyle="--", label=f"Median onset {median_onset:.3f}s")
        style_axis(ax, f"{block.title()} block - Good CR only", "Fraction of detected onsets", "Detected onset (s)")
        ax.set_xlim(0, 0.42)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.text(0.02, 0.92, f"n={count_good}", transform=ax.transAxes, fontsize=10)
        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.18)
        puff_x = 0.200 if block == "short" else 0.400
        ax.axvline(puff_x, linestyle="--", color="black", alpha=0.8)
        ax.text(puff_x + 0.01, ax.get_ylim()[1] * 0.90, "Puff", fontsize=9)
        if np.isfinite(median_onset):
            ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(bottom=0)

        cum = hist_cumulative(values, onset_edges)
        ax = axs[1, col]
        ax.plot(onset_centers, cum, color="black", linewidth=2.2)
        style_axis(ax, f"{block.title()} block cumulative", "Cumulative fraction", "Detected onset (s)")
        ax.set_xlim(0, 0.42)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.set_ylim(0, 1.0)
        ax.text(0.02, 0.92, f"n={count_good}", transform=ax.transAxes, fontsize=10)
        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.18)
        ax.axvline(puff_x, linestyle="--", color="black", alpha=0.8)
        ax.text(puff_x + 0.01, 0.85, "Puff", fontsize=9)
    fig.suptitle(
        f"Good CR onset timing distributions and cumulative curves\nSessions: {first_date} -- {last_date}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_name = f"CR_onset_goodCR_distribution_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  CR_onsetTiming_Distribution_SanityCheck.py:594 - 010_CR_onsetTiming_Distribution_SanityCheck.py:653")
    return fig


def plot_poor_cr_onset_distribution(records, first_date, last_date, first_tok, last_tok):
    onset_edges = np.arange(0.0, 0.4501, 0.01)
    onset_centers = onset_edges[:-1] + np.diff(onset_edges) / 2
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, block in zip(axs, ["short", "long"]):
        values = [
            r["onset_latency"]
            for r in records
            if r["block_label"] == block and r["category"] == "Poor CR"
        ]
        count_poor = len(values)
        frac = hist_fraction(values, onset_edges, smooth_window=3)
        median_onset = np.nanmedian(values) if count_poor else np.nan
        ax.plot(onset_centers, frac, color="dimgray", linewidth=2.2)
        if np.isfinite(median_onset):
            ax.axvline(median_onset, color="red", linestyle="--", label=f"Median onset {median_onset:.3f}s")
        style_axis(ax, f"{block.title()} block - Poor CR only", "Fraction of detected onsets", "Detected onset (s)")
        ax.set_xlim(0, 0.42)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.text(0.02, 0.92, f"n={count_poor}", transform=ax.transAxes, fontsize=10)
        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.18)
        puff_x = 0.200 if block == "short" else 0.400
        ax.axvline(puff_x, linestyle="--", color="black", alpha=0.8)
        ax.text(puff_x + 0.01, ax.get_ylim()[1] * 0.90, "Puff", fontsize=9)
        if np.isfinite(median_onset):
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Poor CR onset timing distributions\nSessions: {first_date} -- {last_date}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_name = f"CR_onset_poorCR_distribution_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  CR_onsetTiming_Distribution_SanityCheck.py - 010_CR_onsetTiming_Distribution_SanityCheck.py:691")
    return fig


def plot_superimposed_onset_distribution(records, first_date, last_date, first_tok, last_tok):
    onset_edges = np.arange(0.0, 0.4501, 0.01)
    onset_centers = onset_edges[:-1] + np.diff(onset_edges) / 2
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    categories = [("Good CR", "black"), ("Poor CR", "dimgray")]
    block_styles = {"short": "-", "long": "--"}
    for ax, (category, color) in zip(axs, categories):
        for block in ["short", "long"]:
            values = [
                r["onset_latency"]
                for r in records
                if r["block_label"] == block and r["category"] == category
            ]
            frac = hist_fraction(values, onset_edges, smooth_window=3)
            median_onset = np.nanmedian(values) if len(values) else np.nan
            label = f"{block.title()} n={len(values)}"
            if np.isfinite(median_onset):
                label += f", med={median_onset:.3f}s"
            ax.plot(
                onset_centers,
                frac,
                color=color,
                linestyle=block_styles[block],
                linewidth=2.2,
                label=label,
            )
        style_axis(ax, f"{category}: Short/Long superimposed", "Fraction of detected onsets", "Detected onset (s)")
        ax.set_xlim(0, 0.42)
        ax.set_ylim(bottom=0)
        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.18)
        ax.axvline(0.200, linestyle=":", color="royalblue", alpha=0.9, label="Short puff")
        ax.axvline(0.400, linestyle=":", color="seagreen", alpha=0.9, label="Long puff")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Superimposed short/long CR onset distributions\nSessions: {first_date} -- {last_date}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_name = f"CR_onset_superimposed_short_long_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  CR_onsetTiming_Distribution_SanityCheck.py - 010_CR_onsetTiming_Distribution_SanityCheck.py:735")
    return fig


def plot_good_cr_cumulative_comparison(records, first_date, last_date, first_tok, last_tok):
    """Superimpose Short and Long Good CR cumulative onset distributions on one panel."""
    onset_edges = np.arange(0.0, 0.4501, 0.005)
    onset_centers = onset_edges[:-1] + np.diff(onset_edges) / 2

    block_styles = {
        "short": {"color": "royalblue", "puff_x": 0.200, "puff_color": "steelblue"},
        "long":  {"color": "seagreen",  "puff_x": 0.400, "puff_color": "darkgreen"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for block in ["short", "long"]:
        values = [
            r["onset_latency"]
            for r in records
            if r["block_label"] == block and r["category"] == "Good CR"
        ]
        n = len(values)
        cum = hist_cumulative(values, onset_edges)
        s = block_styles[block]
        ax.plot(onset_centers, cum, color=s["color"], linewidth=2.5,
                label=f"{block.title()} Good CR (n={n})")
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
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    fig.suptitle(
        f"Good CR cumulative velocity-onset comparison\nSessions: {first_date} -- {last_date}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_name = f"CR_onset_goodCR_cumulative_comparison_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name} - 010_CR_onsetTiming_Distribution_SanityCheck.py:787")
    return fig


def plot_good_cr_average_traces(records, first_tok, last_tok):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for col, block_label in enumerate(["short", "long"]):
        ax = axs[col]
        subset = [r for r in records if r["block_label"] == block_label and r["category"] == "Good CR"]
        if len(subset) == 0:
            ax.text(0.5, 0.5, "No Good CR data", ha="center", va="center", fontsize=12)
            ax.set_title(f"{block_label.title()} block - Good CR")
            ax.set_xlabel("Time from LED onset (s)")
            ax.set_ylabel("FEC")
            continue
        all_traces = np.vstack([r["Fq"] for r in subset])
        mean_trace = np.nanmean(all_traces, axis=0)
        onset_times = [r["onset_time"] for r in subset if np.isfinite(r["onset_time"])]
        median_onset = np.nanmedian(onset_times) if len(onset_times) else np.nan
        shade_color = "blue" if block_label == "short" else "green"
        puff_x = 0.200 if block_label == "short" else 0.400
        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.35)
        ax.axvspan(puff_x, puff_x + 0.020, color=shade_color, alpha=0.35)
        ax.plot(subset[0]["t_grid"], mean_trace, color="black", linewidth=2)
        if np.isfinite(median_onset):
            ax.axvline(median_onset, color="red", linestyle="--", label=f"Median onset {median_onset:.3f}s")
        ax.axvline(puff_x, linestyle="--", color="black", alpha=0.8)
        ax.text(0.01, 0.92 * ax.get_ylim()[1], "LED", fontsize=9)
        ax.text(puff_x + 0.01, 0.92 * ax.get_ylim()[1], "AirPuff", color=shade_color, fontsize=9)
        ax.set_title(f"{block_label.title()} block - Good CR")
        ax.set_xlim(-0.1, 0.45)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time from LED onset (s)")
        ax.set_ylabel("FEC")
        if np.isfinite(median_onset):
            ax.legend(loc="best", fontsize=9)
    fig.suptitle("Average Good CR FEC traces with median onset", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_name = f"CR_onset_goodCR_average_traces_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  CR_onsetTiming_Distribution_SanityCheck.py:634 - 010_CR_onsetTiming_Distribution_SanityCheck.py:827")
    return fig


def plot_poor_cr_average_traces(records, first_tok, last_tok):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for col, block_label in enumerate(["short", "long"]):
        ax = axs[col]
        subset = [r for r in records if r["block_label"] == block_label and r["category"] == "Poor CR"]
        if len(subset) == 0:
            ax.text(0.5, 0.5, "No Poor CR data", ha="center", va="center", fontsize=12)
            ax.set_title(f"{block_label.title()} block - Poor CR")
            ax.set_xlabel("Time from LED onset (s)")
            ax.set_ylabel("FEC")
            continue
        all_traces = np.vstack([r["Fq"] for r in subset])
        mean_trace = np.nanmean(all_traces, axis=0)
        onset_times = [r["onset_time"] for r in subset if np.isfinite(r["onset_time"])]
        median_onset = np.nanmedian(onset_times) if len(onset_times) else np.nan
        shade_color = "blue" if block_label == "short" else "green"
        puff_x = 0.200 if block_label == "short" else 0.400
        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.35)
        ax.axvspan(puff_x, puff_x + 0.020, color=shade_color, alpha=0.35)
        ax.plot(subset[0]["t_grid"], mean_trace, color="dimgray", linewidth=2)
        if np.isfinite(median_onset):
            ax.axvline(median_onset, color="red", linestyle="--", label=f"Median onset {median_onset:.3f}s")
        ax.axvline(puff_x, linestyle="--", color="black", alpha=0.8)
        ax.text(0.01, 0.92 * ax.get_ylim()[1], "LED", fontsize=9)
        ax.text(puff_x + 0.01, 0.92 * ax.get_ylim()[1], "AirPuff", color=shade_color, fontsize=9)
        ax.set_title(f"{block_label.title()} block - Poor CR (n={len(subset)})")
        ax.set_xlim(-0.1, 0.45)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time from LED onset (s)")
        ax.set_ylabel("FEC")
        if np.isfinite(median_onset):
            ax.legend(loc="best", fontsize=9)
    fig.suptitle("Average Poor CR FEC traces with median onset", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_name = f"CR_onset_poorCR_average_traces_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name} - 010_CR_onsetTiming_Distribution_SanityCheck.py:867")
    return fig


def plot_latency_summary_table(records, first_date, last_date, first_tok, last_tok):
    def summarize(block_label):
        values = [
            r["onset_latency"]
            for r in records
            if r["block_label"] == block_label and r["category"] == "Good CR" and np.isfinite(r["onset_latency"])
        ]
        n = len(values)
        median = np.nanmedian(values) if n else np.nan
        p25 = np.nanpercentile(values, 25) if n else np.nan
        p75 = np.nanpercentile(values, 75) if n else np.nan
        return n, median, p25, p75

    rows = []
    for block_label in ["short", "long"]:
        n, median, p25, p75 = summarize(block_label)
        rows.append([
            block_label.title(),
            str(n),
            f"{median:.3f}" if np.isfinite(median) else "NaN",
            f"{p25:.3f}" if np.isfinite(p25) else "NaN",
            f"{p75:.3f}" if np.isfinite(p75) else "NaN",
        ])

    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Block", "Good CR n", "Median onset (s)", "25th pctile", "75th pctile"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    fig.suptitle(
        f"Good CR onset latency summary\nSessions: {first_date} -- {last_date}",
        fontsize=14,
    )
    out_name = f"CR_onset_latency_summary_table_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  CR_onsetTiming_Distribution_SanityCheck.py:679 - 010_CR_onsetTiming_Distribution_SanityCheck.py:912")
    print("Latency summary:  CR_onsetTiming_Distribution_SanityCheck.py:680 - 010_CR_onsetTiming_Distribution_SanityCheck.py:913")
    for row in rows:
        print("|  CR_onsetTiming_Distribution_SanityCheck.py:682 - 010_CR_onsetTiming_Distribution_SanityCheck.py:915".join(map(str, row)))
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
                ax.plot(r["t_grid"], r["Fq"], color="black", linewidth=1.5)
                ax.axvline(r["onset_time"], color="red", linestyle="--", label="Detected onset")
                ax.set_title(f"{category} example {j+1}\n{os.path.basename(r['filepath'])}")
                ax.set_xlim(-0.1, 0.45)
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
    print(f"Saved {out_name}  CR_onsetTiming_Distribution_SanityCheck.py:709 - 010_CR_onsetTiming_Distribution_SanityCheck.py:942")
    return fig


def plot_average_traces(records, first_tok, last_tok):
    categories = ["Good CR", "Poor CR", "No CR"]
    fig, axs = plt.subplots(3, 2, figsize=(16, 9), sharex=True, sharey=True)
    for col, block_label in enumerate(["short", "long"]):
        for row, category in enumerate(categories):
            ax = axs[row, col]
            subset = [r for r in records if r["block_label"] == block_label and r["category"] == category]
            if len(subset) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue
            all_traces = np.vstack([r["Fq"] for r in subset])
            mean_trace = np.nanmean(all_traces, axis=0)
            onset_times = [r["onset_time"] for r in subset if np.isfinite(r["onset_time"])]
            median_onset = np.nanmedian(onset_times) if len(onset_times) else np.nan
            ax.plot(subset[0]["t_grid"], mean_trace, color="black", linewidth=2)
            if np.isfinite(median_onset):
                ax.axvline(median_onset, color="red", linestyle="--", label=f"Median onset {median_onset:.3f}s")
            ax.set_title(f"{block_label.title()} / {category}")
            ax.set_xlim(-0.1, 0.45)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Time from LED onset (s)")
            ax.set_ylabel("FEC")
            if np.isfinite(median_onset):
                ax.legend(loc="best")
    fig.suptitle("Average FEC traces by category with detected onset", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_name = f"CR_onset_sanity_average_traces_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}  CR_onsetTiming_Distribution_SanityCheck.py:741 - 010_CR_onsetTiming_Distribution_SanityCheck.py:974")
    return fig


def main():
    data_files = sorted(glob.glob("*_EBC_*.mat"))
    if len(data_files) == 0:
        print("No *_EBC_*.mat files found in current folder.  CR_onsetTiming_Distribution_SanityCheck.py:748 - 010_CR_onsetTiming_Distribution_SanityCheck.py:981")
        return
    first_tok, last_tok, first_date, last_date = build_session_range(data_files)
    records = prepare_trial_records(data_files)
    if len(records) == 0:
        print("No usable trials found.  CR_onsetTiming_Distribution_SanityCheck.py:753 - 010_CR_onsetTiming_Distribution_SanityCheck.py:986")
        return
    # Main published outputs:
    plot_onset_distributions(records, first_date, last_date, first_tok, last_tok)
    plot_good_cr_onset_distribution(records, first_date, last_date, first_tok, last_tok)
    plot_poor_cr_onset_distribution(records, first_date, last_date, first_tok, last_tok)
    plot_superimposed_onset_distribution(records, first_date, last_date, first_tok, last_tok)
    plot_good_cr_cumulative_comparison(records, first_date, last_date, first_tok, last_tok)
    plot_good_cr_average_traces(records, first_tok, last_tok)
    plot_poor_cr_average_traces(records, first_tok, last_tok)
    plot_latency_summary_table(records, first_date, last_date, first_tok, last_tok)
    plt.close("all")


if __name__ == "__main__":
    main()
