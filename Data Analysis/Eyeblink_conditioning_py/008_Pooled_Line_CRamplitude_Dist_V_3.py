from __future__ import annotations
import os
import glob
import re
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

# ============================================================
# Matplotlib config
# ============================================================
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.use14corefonts"] = False

# ============================================================
# Helpers
# ============================================================
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


def interp_to_grid(x, y, xq, method="linear"):
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

    f = interp1d(
        x_unique,
        y_unique,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
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

    if len(total_vals) > 0:
        return np.nanmax(np.asarray(total_vals, dtype=float))
    if len(eye_vals) > 0:
        return np.nanmax(np.asarray(eye_vals, dtype=float))
    return np.nan


# ============================================================
# CR window / amplitude / classification
# ============================================================
def get_cr_window_from_block(
    t_puff,
    block_label,
    short_pre_ms=25,
    short_post_ms=10,
    long_pre_ms=50,
    long_post_ms=10,
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


def classify_and_measure_cr(
    time,
    signal,
    t_led,
    t_puff,
    block_label,
    good_cr_threshold,
    poor_cr_threshold,
    short_pre_ms=25,
    short_post_ms=10,
    long_pre_ms=50,
    long_post_ms=10,
):
    """
    Returns
    -------
    category : str
    plot_mag_bs : float
        Amplitude used for plotting (category-matched).
        Good CR -> CR-window amplitude (baseline-subtracted)
        Poor CR -> poor-window amplitude (baseline-subtracted)
        No CR   -> CR-window amplitude (baseline-subtracted)
    plot_mag_raw : float
        Same idea, raw version.
    baseline_amp : float
    cr_start : float
    cr_end : float
    cr_mag_bs : float
    cr_mag_raw : float
    poor_mag_bs : float
    poor_mag_raw : float
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if len(time) != len(signal) or len(time) == 0:
        return "No CR", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    baseline_idx = (time >= (t_led - 0.2)) & (time <= t_led)
    if not np.any(baseline_idx):
        return "No CR", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    baseline_amp = np.nanmean(signal[baseline_idx])

    cr_start, cr_end = get_cr_window_from_block(
        t_puff=t_puff,
        block_label=block_label,
        short_pre_ms=short_pre_ms,
        short_post_ms=short_post_ms,
        long_pre_ms=long_pre_ms,
        long_post_ms=long_post_ms,
    )

    if not (np.isfinite(cr_start) and np.isfinite(cr_end)):
        return "No CR", np.nan, np.nan, baseline_amp, cr_start, cr_end, np.nan, np.nan, np.nan, np.nan

    # CR-window amplitude
    cr_idx = (time >= cr_start) & (time <= cr_end)
    cr_mag_raw = np.nan
    cr_mag_bs = np.nan
    if np.any(cr_idx):
        cr_signal = signal[cr_idx]
        if np.any(np.isfinite(cr_signal)):
            cr_mag_raw = np.nanmax(cr_signal)
            cr_mag_bs = cr_mag_raw - baseline_amp

    # Poor-window amplitude
    poor_idx = (time >= t_led) & (time < cr_start)
    poor_mag_raw = np.nan
    poor_mag_bs = np.nan
    if np.any(poor_idx):
        poor_signal = signal[poor_idx]
        if np.any(np.isfinite(poor_signal)):
            poor_mag_raw = np.nanmax(poor_signal)
            poor_mag_bs = poor_mag_raw - baseline_amp

    block_label = str(block_label).strip().lower()
    if block_label == "short":
        good_thr = 0.03
    else:
        good_thr = good_cr_threshold

    # Classification
    if np.isfinite(cr_mag_bs) and cr_mag_bs >= good_thr:
        category = "Good CR"
    elif np.isfinite(poor_mag_bs) and poor_mag_bs >= poor_cr_threshold:
        category = "Poor CR"
    else:
        category = "No CR"

    # Category-matched plotted amplitude
    if category == "Good CR":
        plot_mag_bs = cr_mag_bs
        plot_mag_raw = cr_mag_raw
    elif category == "Poor CR":
        plot_mag_bs = poor_mag_bs
        plot_mag_raw = poor_mag_raw
    else:
        # No CR stays with the CR-window max, which should cluster near zero
        plot_mag_bs = cr_mag_bs
        plot_mag_raw = cr_mag_raw

    return (
        category,
        plot_mag_bs,
        plot_mag_raw,
        baseline_amp,
        cr_start,
        cr_end,
        cr_mag_bs,
        cr_mag_raw,
        poor_mag_bs,
        poor_mag_raw,
    )


# ============================================================
# Histogram / plotting helpers
# ============================================================
def moving_average_same(x, window=5):
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) == 0:
        return x.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def hist_fraction(values, edges, smooth_window=1):
    """
    Histogram-based fraction curve that sums to 1.
    Smoothing is followed by renormalization.
    """
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

    frac = counts.astype(float) / total
    return np.cumsum(frac)


def get_onset_search_end(block_label, short_end_s=0.212, long_end_s=0.412):
    block_label = str(block_label).strip().lower()
    if block_label == "short":
        return short_end_s
    if block_label == "long":
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
    """Same trial-level velocity onset detector used by the CR-onset QC code."""
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

    search_end = get_onset_search_end(block_label, short_end_s=short_end_s, long_end_s=long_end_s)
    if not np.isfinite(search_end):
        return np.nan, np.nan, baseline_amp, np.nan

    search_full_idx = (time >= t_led) & (time <= search_end)
    if not np.any(search_full_idx):
        return np.nan, np.nan, baseline_amp, np.nan

    peak_val = float(np.nanmax(signal_bs[search_full_idx]))
    if not np.isfinite(peak_val) or peak_val < min_peak_height:
        return np.nan, np.nan, baseline_amp, peak_val

    sig_smooth = smooth_trace(signal_bs, window=velocity_smooth_window)
    velocity = np.gradient(sig_smooth, time)

    vel_baseline = velocity[baseline_idx]
    vel_baseline = vel_baseline[np.isfinite(vel_baseline)]
    if vel_baseline.size > 1:
        vel_threshold = np.nanmean(vel_baseline) + velocity_std_factor * np.nanstd(vel_baseline)
    else:
        vel_threshold = 0.0
    vel_threshold = max(vel_threshold, velocity_floor)

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
    return onset_time, onset_time - t_led, baseline_amp, peak_val


def match_records_by_baseline(records, categories=("All", "No CR", "Poor CR", "Good CR"), n_bins=10, rng_seed=42):
    """Baseline-match control/chemo trial records independently for each plotted row."""
    rng = np.random.default_rng(rng_seed)
    matched_records = []

    for block in ("short", "long"):
        for category in categories:
            if category == "All":
                subset = [r for r in records if r["block_label"] == block]
            else:
                subset = [r for r in records if r["block_label"] == block and r["category"] == category]

            ctrl_idx = [i for i, r in enumerate(subset) if r["group"] == "ctrl" and np.isfinite(r["baseline_amp"])]
            chemo_idx = [i for i, r in enumerate(subset) if r["group"] == "chemo" and np.isfinite(r["baseline_amp"])]
            if not ctrl_idx or not chemo_idx:
                for r in subset:
                    rr = r.copy()
                    rr["plot_category"] = category
                    matched_records.append(rr)
                continue

            b_ctrl = np.asarray([subset[i]["baseline_amp"] for i in ctrl_idx], dtype=float)
            b_chemo = np.asarray([subset[i]["baseline_amp"] for i in chemo_idx], dtype=float)
            lo = max(np.nanmin(b_ctrl), np.nanmin(b_chemo))
            hi = min(np.nanmax(b_ctrl), np.nanmax(b_chemo))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                for r in subset:
                    rr = r.copy()
                    rr["plot_category"] = category
                    matched_records.append(rr)
                continue

            edges = np.linspace(lo, hi, n_bins + 1)
            for bin_i in range(n_bins):
                if bin_i == n_bins - 1:
                    c_mask = (b_ctrl >= edges[bin_i]) & (b_ctrl <= edges[bin_i + 1])
                    h_mask = (b_chemo >= edges[bin_i]) & (b_chemo <= edges[bin_i + 1])
                else:
                    c_mask = (b_ctrl >= edges[bin_i]) & (b_ctrl < edges[bin_i + 1])
                    h_mask = (b_chemo >= edges[bin_i]) & (b_chemo < edges[bin_i + 1])

                c_bin = np.asarray(ctrl_idx, dtype=int)[c_mask]
                h_bin = np.asarray(chemo_idx, dtype=int)[h_mask]
                n = min(len(c_bin), len(h_bin))
                if n == 0:
                    continue
                for idx in rng.choice(c_bin, n, replace=False):
                    rr = subset[int(idx)].copy()
                    rr["plot_category"] = category
                    matched_records.append(rr)
                for idx in rng.choice(h_bin, n, replace=False):
                    rr = subset[int(idx)].copy()
                    rr["plot_category"] = category
                    matched_records.append(rr)

    return matched_records


def _record_matches_category(record, category):
    if "plot_category" in record:
        return record["plot_category"] == category
    return category == "All" or record["category"] == category


def _values(records, group, category, field):
    vals = []
    for r in records:
        if r["group"] != group:
            continue
        if not _record_matches_category(r, category):
            continue
        v = r.get(field, np.nan)
        if np.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=float)


def _block_values(records, group, category, block, field):
    vals = []
    for r in records:
        if r["group"] != group or r["block_label"] != block:
            continue
        if not _record_matches_category(r, category):
            continue
        v = r.get(field, np.nan)
        if np.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=float)


def plot_baseline_onset_distribution_grid(records, title, outname, first_date, last_date):
    categories = ["All", "No CR", "Poor CR", "Good CR"]
    blocks = [("short", "Short", 0.200), ("long", "Long", 0.400)]
    time_edges = np.arange(-0.200, 0.4501, 0.01)
    time_centers = time_edges[:-1] + np.diff(time_edges) / 2
    led_window_info = "LED baseline window: [-200, 0] ms relative to LED"
    cr_window_info = (
        "CR windows: Short [-25, +10] ms relative to AirPuff at 200 ms; "
        "Long [-50, +10] ms relative to AirPuff at 400 ms"
    )

    fig, axs = plt.subplots(len(categories), 2, figsize=(11, 12), sharey=False)
    group_meta = {
        "ctrl": {"label": "Control", "color": "black"},
        "chemo": {"label": "Chemo", "color": "#D14949"},
    }

    for row, category in enumerate(categories):
        for col, (block, block_title, puff_x) in enumerate(blocks):
            ax = axs[row, col]

            for group in ("ctrl", "chemo"):
                meta = group_meta[group]

                baseline_vals = _block_values(records, group, category, block, "baseline_peak_time")
                baseline_frac = hist_fraction(baseline_vals, time_edges, smooth_window=3)
                ax.plot(
                    time_centers,
                    baseline_frac,
                    linestyle=":",
                    color=meta["color"],
                    linewidth=2,
                    label=f"{meta['label']} BL n={len(baseline_vals)}",
                )

                onset_vals = _block_values(records, group, category, block, "onset_time")
                onset_frac = hist_fraction(onset_vals, time_edges, smooth_window=3)
                ax.plot(
                    time_centers,
                    onset_frac,
                    linestyle="-",
                    color=meta["color"],
                    linewidth=2,
                    label=f"{meta['label']} CR n={len(onset_vals)}",
                )

            ax.axvspan(-0.2, 0.0, color="0.92", alpha=0.75, linewidth=0)
            ax.axvline(0.0, color="0.35", linestyle="-", linewidth=0.9)
            ax.axvline(puff_x, color="0.35", linestyle="--", linewidth=1.0)
            ax.text(0.0, 0.98, "LED", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8)
            ax.text(puff_x, 0.98, "AirPuff", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="0.25")
            ax.set_title(f"{category} | {block_title}", fontsize=10)
            ax.set_xlabel("Event time from LED onset (s)")
            ax.set_ylabel("Fraction")
            ax.set_xlim(-0.2, 0.45)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out")
            ax.grid(False)
            ax.legend(frameon=False, fontsize=7, loc="best")

    fig.suptitle(
        f"{title}\nSessions: {first_date} -- {last_date}",
        fontsize=14,
        y=0.995,
    )
    fig.text(0.5, 0.952, led_window_info, ha="center", va="center", fontsize=9)
    fig.text(0.5, 0.935, cr_window_info, ha="center", va="center", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(outname, bbox_inches="tight")
    print(f"Saved: {outname}")


def make_edges_from_data(values, step=0.02, fallback_min=-0.4, fallback_max=1.0):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        lo = fallback_min
        hi = fallback_max
    else:
        lo = min(fallback_min, np.floor(np.min(vals) / step) * step)
        hi = max(fallback_max, np.ceil(np.max(vals) / step) * step)

    edges = np.arange(lo, hi + step * 1.01, step)
    if len(edges) < 2:
        edges = np.array([lo, lo + step], dtype=float)
    return edges


def get_plot_limits(values, pad_frac=0.06, min_span=0.12, hard_min=None, hard_max=None):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        lo, hi = -0.1, 1.0
    else:
        lo = np.min(vals)
        hi = np.max(vals)
        span = hi - lo
        if span < min_span:
            mid = 0.5 * (lo + hi)
            lo = mid - min_span / 2
            hi = mid + min_span / 2
        else:
            pad = span * pad_frac
            lo -= pad
            hi += pad

    if hard_min is not None:
        lo = min(lo, hard_min)
    if hard_max is not None:
        hi = max(hi, hard_max)

    return lo, hi


def style_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("CR Magnitude")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(False)


def values_for_plot(dist, block, category, group, amp_type):
    return dist[block][category][group][amp_type]


def combined_values(dist, blocks, categories, groups, amp_type):
    vals = []
    for block in blocks:
        for category in categories:
            for group in groups:
                arr = values_for_plot(dist, block, category, group, amp_type)
                if len(arr) > 0:
                    vals.append(arr)
    if not vals:
        return np.array([], dtype=float)
    return np.concatenate(vals)


# ============================================================
# Main
# ============================================================
def main():
    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_files = sorted(glob.glob(os.path.join(script_dir, "*_EBC_*.mat")))
    if len(data_files) == 0:
        print(f"No *_EBC_*.mat files found in {script_dir}.  Pooled_Line_CRamplitude_Dist_V_3.py:420 - 008_Pooled_Line_CRamplitude_Dist_V_3.py:451")
        return

    # --------------------------------------------------------
    # Settings
    # --------------------------------------------------------
    good_cr_threshold = 0.05
    poor_cr_threshold = 0.02

    exclude_probe = True
    exclude_timeout = True

    smooth_win = 5
    interp_method = "linear"

    t_pre = 0.2
    t_post = 0.6
    dt = 1 / 250
    t_grid = np.arange(-t_pre, t_post + dt / 2, dt)

    short_isi_max = 0.30

    # Modified CR windows
    short_cr_pre_ms = 25
    short_cr_post_ms = 10
    long_cr_pre_ms = 50
    long_cr_post_ms = 10

    # Histogram settings
    raw_step = 0.02
    bs_step = 0.02
    hist_smooth_window = 5

    raw_edges = np.arange(0, 1.0001 + raw_step, raw_step)
    raw_bin_centers = raw_edges[:-1] + np.diff(raw_edges) / 2

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------
    dist = {
        "short": {
            "Good CR": {"raw": [], "bs": []},
            "Poor CR": {"raw": [], "bs": []},
            "No CR": {"raw": [], "bs": []},
        },
        "long": {
            "Good CR": {"raw": [], "bs": []},
            "Poor CR": {"raw": [], "bs": []},
            "No CR": {"raw": [], "bs": []},
        },
    }
    distribution_records = []
    record_id = 0

    # --------------------------------------------------------
    # Date range for title
    # --------------------------------------------------------
    session_dates = []
    for f in data_files:
        m = re.search(r"\d{8}", os.path.basename(f))
        if m:
            try:
                session_dates.append(datetime.strptime(m.group(), "%Y%m%d"))
            except Exception:
                pass

    if session_dates:
        first_tok = min(session_dates).strftime("%Y%m%d")
        last_tok = max(session_dates).strftime("%Y%m%d")
        first_date = min(session_dates).strftime("%m/%d/%Y")
        last_date = max(session_dates).strftime("%m/%d/%Y")
    else:
        first_tok = "Unknown"
        last_tok = "Unknown"
        first_date = "Unknown"
        last_date = "Unknown"

    n_sessions_used = 0

    # --------------------------------------------------------
    # Loop over sessions
    # --------------------------------------------------------
    for filepath in data_files:
        print(f"Processing: {filepath}  Pooled_Line_CRamplitude_Dist_V_3.py:501 - 008_Pooled_Line_CRamplitude_Dist_V_3.py:532")
        S = loadmat(filepath, squeeze_me=True, struct_as_record=False)

        if "SessionData" not in S:
            print(f"Skipping {filepath}: SessionData not found  Pooled_Line_CRamplitude_Dist_V_3.py:505 - 008_Pooled_Line_CRamplitude_Dist_V_3.py:536")
            continue

        SD = S["SessionData"]
        is_chemo_session = is_chemogenetics_session(SD)
        group_label = "chemo" if is_chemo_session else "ctrl"
        raw_events = get_field(SD, "RawEvents", None)
        trials = ensure_trial_list(get_field(raw_events, "Trial", None))

        overall_max = collect_overall_max(trials)
        if not np.isfinite(overall_max):
            print(f"Skipping {filepath}: could not compute overall_max  Pooled_Line_CRamplitude_Dist_V_3.py:514 - 008_Pooled_Line_CRamplitude_Dist_V_3.py:545")
            continue

        n_sessions_used += 1

        for tr in trials:
            states = get_field(tr, "States", None)
            events = get_field(tr, "Events", None)
            data = get_field(tr, "Data", None)

            if states is None or events is None or data is None:
                continue

            # Exclude timeout
            if exclude_timeout and has_field(states, "CheckEyeOpenTimeout"):
                v = safe_array(get_field(states, "CheckEyeOpenTimeout", np.nan))
                if np.any(np.isfinite(v)):
                    continue

            # Exclude probe
            if exclude_probe and has_field(data, "IsProbeTrial"):
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

            FEC = 1.0 - eyeAreaPixels / overall_max

            t_led_abs = scalar_or_nan(GT1_start)
            t_puff_abs = scalar_or_nan(GT2_start)

            if not (np.isfinite(t_led_abs) and np.isfinite(t_puff_abs)):
                continue

            t_rel = FECTimes - t_led_abs
            puff_rel = t_puff_abs - t_led_abs

            if smooth_win > 1:
                FEC = smooth_trace(FEC, smooth_win)

            Fq = interp_to_grid(t_rel, FEC, t_grid, method=interp_method)

            isi_dur = float(LED_Puff_ISI[1] - LED_Puff_ISI[0])
            is_short = isi_dur <= short_isi_max
            block_label = "short" if is_short else "long"

            (
                category,
                plot_mag_bs,
                plot_mag_raw,
                baseline_amp,
                cr_start,
                cr_end,
                cr_mag_bs,
                cr_mag_raw,
                poor_mag_bs,
                poor_mag_raw,
            ) = classify_and_measure_cr(
                time=t_grid,
                signal=Fq,
                t_led=0.0,
                t_puff=puff_rel,
                block_label=block_label,
                good_cr_threshold=good_cr_threshold,
                poor_cr_threshold=poor_cr_threshold,
                short_pre_ms=short_cr_pre_ms,
                short_post_ms=short_cr_post_ms,
                long_pre_ms=long_cr_pre_ms,
                long_post_ms=long_cr_post_ms,
            )

            if np.isfinite(plot_mag_raw):
                dist[block_label][category]["raw"].append(plot_mag_raw)
            if np.isfinite(plot_mag_bs):
                dist[block_label][category]["bs"].append(plot_mag_bs)

            if category == "Good CR":
                min_peak_height = 0.03
            elif category == "Poor CR":
                min_peak_height = 0.02
            else:
                min_peak_height = 0.005

            onset_time, onset_latency, _, onset_peak_val = find_detected_onset_velocity_based(
                time=t_grid,
                signal=Fq,
                block_label=block_label,
                t_led=0.0,
                min_peak_height=min_peak_height,
                baseline_window=(-0.2, 0.0),
                short_end_s=0.212,
                long_end_s=0.412,
            )

            if np.isfinite(baseline_amp):
                baseline_idx = (t_grid >= -0.2) & (t_grid <= 0.0) & np.isfinite(Fq)
                if np.any(baseline_idx):
                    baseline_times = t_grid[baseline_idx]
                    baseline_vals = Fq[baseline_idx]
                    baseline_peak_time = baseline_times[int(np.nanargmax(baseline_vals))]
                else:
                    baseline_peak_time = np.nan

                distribution_records.append({
                    "record_id": record_id,
                    "group": group_label,
                    "block_label": block_label,
                    "category": category,
                    "baseline_amp": baseline_amp,
                    "baseline_peak_time": baseline_peak_time,
                    "onset_time": onset_time,
                    "onset_latency": onset_latency,
                    "onset_peak_val": onset_peak_val,
                })
                record_id += 1

    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------
    for block in ["short", "long"]:
        for cat in ["Good CR", "Poor CR", "No CR"]:
            dist[block][cat]["raw"] = np.asarray(dist[block][cat]["raw"], dtype=float)
            dist[block][cat]["bs"] = np.asarray(dist[block][cat]["bs"], dtype=float)

    # --------------------------------------------------------
    # Baseline and detected CR-onset distributions
    # --------------------------------------------------------
    matched_distribution_records = match_records_by_baseline(distribution_records)
    plot_baseline_onset_distribution_grid(
        distribution_records,
        "Raw Distribution of FEC Magnitude",
        f"Baseline_CRonset_Distributions_Raw_{first_tok}_to_{last_tok}.pdf",
        first_date,
        last_date,
    )
    plot_baseline_onset_distribution_grid(
        matched_distribution_records,
        "Baseline-matched Distribution of FEC Magnitude",
        f"Baseline_CRonset_Distributions_BaselineMatched_{first_tok}_to_{last_tok}.pdf",
        first_date,
        last_date,
    )

    # --------------------------------------------------------
    # Build baseline-subtracted edges from actual data
    # --------------------------------------------------------
    all_bs = []
    for block in ["short", "long"]:
        for cat in ["Good CR", "Poor CR", "No CR"]:
            arr = dist[block][cat]["bs"]
            if len(arr) > 0:
                all_bs.append(arr)

    if len(all_bs) > 0:
        all_bs_concat = np.concatenate(all_bs)
    else:
        all_bs_concat = np.array([], dtype=float)

    bs_edges = make_edges_from_data(
        all_bs_concat,
        step=bs_step,
        fallback_min=-0.4,
        fallback_max=1.0,
    )
    bs_bin_centers = bs_edges[:-1] + np.diff(bs_edges) / 2

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------
    color_good = "black"
    color_poor = "gray"
    color_no = "lightgray"

    # --------------------------------------------------------
    # Plot line distributions
    # --------------------------------------------------------
    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 10))
    axs1 = axs1.ravel()

    configs = [
        ("short", "raw", "Short Block --- Raw"),
        ("short", "bs", "Short Block --- Baseline Subtracted"),
        ("long", "raw", "Long Block --- Raw"),
        ("long", "bs", "Long Block --- Baseline Subtracted"),
    ]

    for ax, (block, amp_type, title) in zip(axs1, configs):
        if amp_type == "raw":
            edges_use = raw_edges
            centers_use = raw_bin_centers
        else:
            edges_use = bs_edges
            centers_use = bs_bin_centers

        frac_good = hist_fraction(
            dist[block]["Good CR"][amp_type],
            edges=edges_use,
            smooth_window=hist_smooth_window,
        )
        frac_poor = hist_fraction(
            dist[block]["Poor CR"][amp_type],
            edges=edges_use,
            smooth_window=hist_smooth_window,
        )
        frac_no = hist_fraction(
            dist[block]["No CR"][amp_type],
            edges=edges_use,
            smooth_window=hist_smooth_window,
        )

        ax.plot(centers_use, frac_good, "-", color=color_good, linewidth=2, label="Good")
        ax.plot(centers_use, frac_poor, "-", color=color_poor, linewidth=2, label="Poor")
        ax.plot(centers_use, frac_no, "-", color=color_no, linewidth=2, label="No")

        style_axis(ax, title, ylabel="Fraction")
        ax.set_ylim(bottom=0)

        if amp_type == "raw":
            ax.set_xlim(0, 1)
        else:
            vals_bs = np.concatenate([
                dist[block]["Good CR"]["bs"],
                dist[block]["Poor CR"]["bs"],
                dist[block]["No CR"]["bs"],
            ])
            xlo, xhi = get_plot_limits(
                vals_bs,
                pad_frac=0.08,
                min_span=0.20,
                hard_min=-0.4,
                hard_max=0.5
            )
            ax.set_xlim(xlo, xhi)

    fig1.suptitle(
        f"Line Distribution of Pooled CRs\nSessions: {first_date} -- {last_date} | sessions={n_sessions_used}",
        fontsize=14
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    out_line = f"Pooled_Line_CR_Distributions_ModifiedWindow_{first_tok}_to_{last_tok}.pdf"
    fig1.savefig(out_line, bbox_inches="tight")
    print(f"Saved: {out_line}  Pooled_Line_CRamplitude_Dist_V_3.py:713 - 008_Pooled_Line_CRamplitude_Dist_V_3.py:744")

    # --------------------------------------------------------
    # Plot cumulative distributions
    # --------------------------------------------------------
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10))
    axs2 = axs2.ravel()

    for ax, (block, amp_type, title) in zip(axs2, configs):
        if amp_type == "raw":
            edges_use = raw_edges
            centers_use = raw_bin_centers
        else:
            edges_use = bs_edges
            centers_use = bs_bin_centers

        cum_good = hist_cumulative(dist[block]["Good CR"][amp_type], edges_use)
        cum_poor = hist_cumulative(dist[block]["Poor CR"][amp_type], edges_use)
        cum_no = hist_cumulative(dist[block]["No CR"][amp_type], edges_use)

        ax.plot(centers_use, cum_good, "-", color=color_good, linewidth=2, label="Good")
        ax.plot(centers_use, cum_poor, "-", color=color_poor, linewidth=2, label="Poor")
        ax.plot(centers_use, cum_no, "-", color=color_no, linewidth=2, label="No")

        style_axis(ax, title, ylabel="Cumulative Fraction")
        ax.set_ylim(0, 1.02)

        if amp_type == "raw":
            ax.set_xlim(0, 1)
        else:
            vals_bs = np.concatenate([
                dist[block]["Good CR"]["bs"],
                dist[block]["Poor CR"]["bs"],
                dist[block]["No CR"]["bs"],
            ])
            xlo, xhi = get_plot_limits(
                vals_bs,
                pad_frac=0.08,
                min_span=0.20,
                hard_min=-0.4,
                hard_max=0.5
            )
            ax.set_xlim(xlo, xhi)

    fig2.suptitle(
        f"Cumulative CR Distributions Using Modified CR Window\nSessions: {first_date} -- {last_date} | sessions={n_sessions_used}",
        fontsize=14
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    out_cum = f"Pooled_Cumulative_CR_Distributions_ModifiedWindow_{first_tok}_to_{last_tok}.pdf"
    fig2.savefig(out_cum, bbox_inches="tight")
    print(f"Saved: {out_cum}  Pooled_Line_CRamplitude_Dist_V_3.py:765 - 008_Pooled_Line_CRamplitude_Dist_V_3.py:796")

    # --------------------------------------------------------
    # Summary print
    # --------------------------------------------------------
    print("\nCounts:  Pooled_Line_CRamplitude_Dist_V_3.py:770 - 008_Pooled_Line_CRamplitude_Dist_V_3.py:801")
    for block in ["short", "long"]:
        for cat in ["Good CR", "Poor CR", "No CR"]:
            n_raw = len(dist[block][cat]["raw"])
            n_bs = len(dist[block][cat]["bs"])
            print(f"{block:>5s} | {cat:>7s} | raw={n_raw:4d} | bs={n_bs:4d}  Pooled_Line_CRamplitude_Dist_V_3.py:775 - 008_Pooled_Line_CRamplitude_Dist_V_3.py:806")

    plt.show()


if __name__ == "__main__":
    main()
