import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from datetime import datetime


# ============================================================
# Helpers
# ============================================================

def scalar_or_nan(x):
    arr = np.asarray(x).squeeze()
    if arr.size == 0:
        return np.nan
    if arr.ndim == 0:
        return float(arr)
    return float(arr.flat[0])

def style_axes(ax):
    # keep only left and bottom spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # keep x-y axes visible
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # ticks only on left and bottom
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.tick_params(direction="out")
    
def get_trial_timing(tr, short_isi_max=0.30):
    """
    Version-robust timing extractor.

    Returns
    -------
    t_led_abs : float
    t_led_end_abs : float
    t_puff_abs : float
    t_puff_end_abs : float
    isi_dur : float
    is_short : bool
    block_label : str
    """
    states = get_field(tr, "States", None)
    events = get_field(tr, "Events", None)
    data = get_field(tr, "Data", None)

    # ----- LED start / end -----
    t_led_abs = scalar_or_nan(get_field(events, "GlobalTimer1_Start", None))
    t_led_end_abs = scalar_or_nan(get_field(events, "GlobalTimer1_End", None))

    # ----- Puff start / end -----
    t_puff_abs = scalar_or_nan(get_field(events, "GlobalTimer2_Start", None))
    t_puff_end_abs = scalar_or_nan(get_field(events, "GlobalTimer2_End", None))

    # Fallback for newer versions: AirPuff onset may be stored in Data
    if not np.isfinite(t_puff_abs):
        for cand in ["AirPuff_Onset", "AirPuff_Ons", "AirPuff_On", "AirPuff_OnsetTime"]:
            val = get_field(data, cand, None)
            t_puff_abs = scalar_or_nan(val)
            if np.isfinite(t_puff_abs):
                break

    # ----- ISI duration -----
    led_puff_isi = safe_array(get_field(states, "LED_Puff_ISI", None))
    if led_puff_isi.size >= 2:
        isi_dur = float(led_puff_isi[1] - led_puff_isi[0])
    else:
        # fallback to Data.ISI in newer versions
        isi_dur = scalar_or_nan(get_field(data, "ISI", None))

    # ----- block label -----
    block_type = get_field(data, "BlockType", None)
    if block_type is None:
        block_label = "short" if np.isfinite(isi_dur) and isi_dur <= short_isi_max else "long"
    else:
        block_type = str(block_type).strip().lower()
        if "short" in block_type:
            block_label = "short"
        elif "long" in block_type:
            block_label = "long"
        else:
            # warm_up / single / unknown → infer from ISI
            block_label = "short" if np.isfinite(isi_dur) and isi_dur <= short_isi_max else "long"

    is_short = (block_label == "short")

    return t_led_abs, t_led_end_abs, t_puff_abs, t_puff_end_abs, isi_dur, is_short, block_label

def get_field(obj, field_name, default=None):
    if obj is None:
        return default
    if hasattr(obj, field_name):
        return getattr(obj, field_name)
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return default

def compute_baseline(trace, t, baseline_window=(-0.2, 0.0)):
    trace = np.asarray(trace, dtype=float)
    t = np.asarray(t, dtype=float)

    idx = (t >= baseline_window[0]) & (t <= baseline_window[1])
    if not np.any(idx):
        return np.nan

    vals = trace[idx]
    if np.all(~np.isfinite(vals)):
        return np.nan

    return np.nanmean(vals)

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


def matlab_load_sessiondata(filepath):
    data = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    if "SessionData" not in data:
        raise KeyError(f"SessionData not found in {filepath}")
    return data["SessionData"]


def smooth_trace(x, window=5):
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    return uniform_filter1d(x, size=window, mode="nearest")


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

    kind = "linear" if method == "linear" else "linear"

    f = interp1d(
        x_unique,
        y_unique,
        kind=kind,
        bounds_error=False,
        fill_value=np.nan
    )
    return f(xq)

def collect_overall_max(trials):
    """
    Version-robust session-wide normalization factor.

    Prefer totalEllipsePixels if available:
      - if vector: use all values
      - if scalar: use the scalar
    fallback:
      - use eyeAreaPixels values
    """
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
# CR classifier
# ============================================================

def classify_cr_by_block(
    time,
    signal,
    t_led,
    t_puff,
    good_cr_threshold,
    poor_cr_threshold,
    block_label,
    short_pre_ms=25,
    short_post_ms=12,
    long_pre_ms=50,
    long_post_ms=12
):
    """
    Block-specific CR classifier using mean + peak amplitude in a configurable CR window.

    Parameters
    ----------
    time, signal : 1D arrays
        LED-aligned time vector and normalized FEC trace.
    t_led : float
        LED onset time in the aligned coordinate system. Usually 0.
    t_puff : float
        Airpuff onset time in the aligned coordinate system.
    block_label : str
        'short' or 'long'
    short_pre_ms, short_post_ms, long_pre_ms, long_post_ms : float
        CR window edges relative to puff onset, in ms.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if len(time) != len(signal):
        return "No CR", np.nan, np.nan, np.nan, np.nan

    baseline_idx = (time >= (t_led - 0.2)) & (time <= t_led)
    if not np.any(baseline_idx):
        return "No CR", np.nan, np.nan, np.nan, np.nan

    block_label = str(block_label).strip().lower()

    short_pre  = short_pre_ms / 1000.0
    short_post = short_post_ms / 1000.0
    long_pre   = long_pre_ms / 1000.0
    long_post  = long_post_ms / 1000.0

    if block_label == "short":
        good_start = t_puff - short_pre
        good_end   = t_puff + short_post
        good_thr   = 0.03
        peak_thr   = 0.035

    elif block_label == "long":
        good_start = t_puff - long_pre
        good_end   = t_puff + long_post
        good_thr   = good_cr_threshold
        peak_thr   = good_cr_threshold

    else:
        return "No CR", np.nan, np.nan, np.nan, np.nan

    poor_start = t_led
    poor_end   = good_start

    good_idx = (time >= good_start) & (time <= good_end)
    poor_idx = (time >= poor_start) & (time < poor_end)

    if not np.any(good_idx):
        return "No CR", np.nan, np.nan, good_start, good_end

    baseline_signal = signal[baseline_idx]
    good_signal = signal[good_idx]

    baseline_amp = np.nanmean(baseline_signal)
    cr_amp = np.nanmean(good_signal)

    good_mean_above = cr_amp - baseline_amp
    good_peak_above = np.nanmax(good_signal) - baseline_amp

    if np.any(poor_idx):
        poor_signal = signal[poor_idx]
        poor_amp = np.nanmean(poor_signal)
        poor_mean_above = poor_amp - baseline_amp
    else:
        poor_mean_above = -np.inf

    if (good_mean_above >= good_thr) or (good_peak_above >= peak_thr):
        category = "Good CR"
    elif poor_mean_above >= poor_cr_threshold:
        category = "Poor CR"
    else:
        category = "No CR"

    return category, cr_amp, baseline_amp, good_start, good_end

# ============================================================
# Stats packer
# ============================================================

def pack_stats(t, fmat, puff_starts, puff_ends, cr_starts, cr_ends, led_starts, led_ends):
    out = {}
    out["t"] = np.asarray(t, dtype=float)

    # convert list -> 2D numpy array if needed
    if isinstance(fmat, list):
        if len(fmat) == 0:
            fmat = np.empty((0, len(t)))
        else:
            fmat = np.vstack(fmat)
    else:
        fmat = np.asarray(fmat, dtype=float)
        if fmat.size == 0:
            fmat = np.empty((0, len(t)))

    out["n"] = fmat.shape[0]

    if out["n"] == 0:
        out["mean"] = np.full_like(out["t"], np.nan, dtype=float)
        out["sem"] = np.full_like(out["t"], np.nan, dtype=float)
        out["puff_start"] = np.nan
        out["puff_end"] = np.nan
        out["cr_start"] = np.nan
        out["cr_end"] = np.nan
        out["led_start"] = np.nan
        out["led_end"] = np.nan
    else:
        out["mean"] = np.nanmean(fmat, axis=0)
        out["sem"] = np.nanstd(fmat, axis=0) / np.sqrt(out["n"])

        out["puff_start"] = np.nanmedian(puff_starts) if len(puff_starts) else np.nan
        out["puff_end"]   = np.nanmedian(puff_ends)   if len(puff_ends) else np.nan

        out["cr_start"] = np.nanmedian(cr_starts) if len(cr_starts) else np.nan
        out["cr_end"]   = np.nanmedian(cr_ends)   if len(cr_ends) else np.nan

        out["led_start"] = np.nanmedian(led_starts) if len(led_starts) else 0.0
        out["led_end"]   = np.nanmedian(led_ends)   if len(led_ends) else np.nan

    return out


# ============================================================
# Plot helper
# ============================================================

def plot_panel(ax, S, cr_label, block_label):
    ax.plot(S["t"], S["mean"], linewidth=1.8, color="k")

    if np.any(np.isfinite(S["mean"])):
        ax.fill_between(
            S["t"],
            S["mean"] + S["sem"],
            S["mean"] - S["sem"],
            color="k",
            alpha=0.15
        )

    # LED shaded region from actual event timing
    if np.isfinite(S["led_start"]) and np.isfinite(S["led_end"]):
        ax.axvspan(S["led_start"], S["led_end"], color="gray", alpha=0.18)

    # AirPuff shaded region from actual event timing
    if block_label.lower() == "long":
        puff_color = "limegreen"
        puff_text_color = "green"
    else:
        puff_color = "dodgerblue"
        puff_text_color = "dodgerblue"

    if np.isfinite(S["puff_start"]) and np.isfinite(S["puff_end"]):
        ax.axvspan(S["puff_start"], S["puff_end"], color=puff_color, alpha=0.35)

    # CR window from actual classification window
    if np.isfinite(S["cr_start"]) and np.isfinite(S["cr_end"]):
        ax.axvspan(S["cr_start"], S["cr_end"], color="gold", alpha=0.18)
        ax.axvline(S["cr_start"], color="darkorange", linestyle=":", linewidth=1.5)
        ax.axvline(S["cr_end"], color="darkorange", linestyle=":", linewidth=1.5)

    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.03 * (ymax - ymin)

    if np.isfinite(S["led_start"]) and np.isfinite(S["led_end"]):
        ax.text(S["led_start"] + 0.002, y_text, "LED",
                color="black", va="bottom", ha="left", fontsize=10)

    if np.isfinite(S["puff_start"]) and np.isfinite(S["puff_end"]):
        ax.text(S["puff_start"] + 0.002, y_text, "AirPuff",
                color=puff_text_color, va="bottom", ha="left", fontsize=10)

    if np.isfinite(S["cr_start"]) and np.isfinite(S["cr_end"]):
        ax.text(S["cr_start"] + 0.002, y_text - 0.06 * (ymax - ymin), "CR window",
                color="darkorange", va="bottom", ha="left", fontsize=9)

    ax.set_title(
        f"{cr_label} -- {block_label}\n"
        f"Control n={S_control['n']}, Chemo n={S_chemo['n']}",
        fontsize=13
    )
    ax.set_xlabel("Time from LED onset (s)")
    ax.set_ylabel("FEC")
    ax.grid(False)
    ax.tick_params(direction="out")
    style_axes(ax)

def plot_panel_two_groups(ax, S_control, S_chemo, cr_label, block_label):
    # Control = black
    if np.any(np.isfinite(S_control["mean"])):
        ax.plot(S_control["t"], S_control["mean"], linewidth=1.8, color="k", label="Control")
        ax.fill_between(
            S_control["t"],
            S_control["mean"] + S_control["sem"],
            S_control["mean"] - S_control["sem"],
            color="k",
            alpha=0.15
        )

    # Chemo = blue
    if np.any(np.isfinite(S_chemo["mean"])):
        ax.plot(S_chemo["t"], S_chemo["mean"], linewidth=1.8, color="blue", label="Chemo")
        ax.fill_between(
            S_chemo["t"],
            S_chemo["mean"] + S_chemo["sem"],
            S_chemo["mean"] - S_chemo["sem"],
            color="blue",
            alpha=0.15
        )

    # Use control timing if available, otherwise chemo
    S_ref = S_control if np.any(np.isfinite(S_control["mean"])) else S_chemo

    if np.isfinite(S_ref["led_start"]) and np.isfinite(S_ref["led_end"]):
        ax.axvspan(S_ref["led_start"], S_ref["led_end"], color="gray", alpha=0.18)

    if block_label.lower() == "long":
        puff_color = "limegreen"
        puff_text_color = "green"
    else:
        puff_color = "dodgerblue"
        puff_text_color = "dodgerblue"

    if np.isfinite(S_ref["puff_start"]) and np.isfinite(S_ref["puff_end"]):
        ax.axvspan(S_ref["puff_start"], S_ref["puff_end"], color=puff_color, alpha=0.35)

    if np.isfinite(S_ref["cr_start"]) and np.isfinite(S_ref["cr_end"]):
        ax.axvspan(S_ref["cr_start"], S_ref["cr_end"], color="gold", alpha=0.18)
        ax.axvline(S_ref["cr_start"], color="darkorange", linestyle=":", linewidth=1.5)
        ax.axvline(S_ref["cr_end"], color="darkorange", linestyle=":", linewidth=1.5)

    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.03 * (ymax - ymin)

    if np.isfinite(S_ref["led_start"]) and np.isfinite(S_ref["led_end"]):
        ax.text(S_ref["led_start"] + 0.002, y_text, "LED",
                color="black", va="bottom", ha="left", fontsize=10)

    if np.isfinite(S_ref["puff_start"]) and np.isfinite(S_ref["puff_end"]):
        ax.text(S_ref["puff_start"] + 0.002, y_text, "AirPuff",
                color=puff_text_color, va="bottom", ha="left", fontsize=10)

    if np.isfinite(S_ref["cr_start"]) and np.isfinite(S_ref["cr_end"]):
        ax.text(S_ref["cr_start"] + 0.002, y_text - 0.06 * (ymax - ymin), "CR window",
                color="darkorange", va="bottom", ha="left", fontsize=9)

    
    ax.set_title(
        f"{cr_label} -- {block_label}\n"
        f"Control n={S_control['n']}, Chemo n={S_chemo['n']}",
        fontsize=13
    )
    ax.set_xlabel("Time from LED onset (s)")
    ax.set_ylabel("FEC")
    ax.grid(False)
    ax.tick_params(direction="out")
    ax.legend(frameon=False, loc="best")
    style_axes(ax)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
# ============================================================
# Main
# ============================================================

def main():
    # ------------ sessions ------------
    data_files = sorted(glob.glob("*_EBC_*.mat"))
    if len(data_files) == 0:
        print("No *_EBC_*.mat files found.  FEC_AvgClassified_CR.py:532 - 05_FEC_AvgClassified_CR.py:532")
        return

    # ------------ settings ------------
    
    baseline_window = (-0.2, 0.0)

    # define separately for each group if you want
    baseline_max_control = 0.48
    baseline_max_chemo = 0.40
    
    t_pre = 0.2
    t_post = 0.6
    dt = 1 / 250
    smooth_win = 5

    exclude_probe = True
    exclude_timeout = True

    good_cr_threshold = 0.05
    poor_cr_threshold = 0.02

    interp_method = "linear"
    short_isi_max = 0.30

    short_cr_pre_ms  = 25
    short_cr_post_ms = 12
    long_cr_pre_ms   = 50
    long_cr_post_ms  = 12
    # ------------ date range ------------
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

    # ------------ collect + classify ------------
    t_grid = np.arange(-t_pre, t_post + dt/2, dt)

    # =========================
    # CONTROL accumulators
    # =========================

    # Short
    F_short_good_control = []
    F_short_poor_control = []
    F_short_no_control = []

    puff_short_good_start_control, puff_short_good_end_control = [], []
    puff_short_poor_start_control, puff_short_poor_end_control = [], []
    puff_short_no_start_control,   puff_short_no_end_control   = [], []

    cr_short_good_start_control, cr_short_good_end_control = [], []
    cr_short_poor_start_control, cr_short_poor_end_control = [], []
    cr_short_no_start_control,   cr_short_no_end_control   = [], []

    led_short_good_start_control, led_short_good_end_control = [], []
    led_short_poor_start_control, led_short_poor_end_control = [], []
    led_short_no_start_control,   led_short_no_end_control   = [], []

    # Long
    F_long_good_control = []
    F_long_poor_control = []
    F_long_no_control = []

    puff_long_good_start_control, puff_long_good_end_control = [], []
    puff_long_poor_start_control, puff_long_poor_end_control = [], []
    puff_long_no_start_control,   puff_long_no_end_control   = [], []

    cr_long_good_start_control, cr_long_good_end_control = [], []
    cr_long_poor_start_control, cr_long_poor_end_control = [], []
    cr_long_no_start_control,   cr_long_no_end_control   = [], []

    led_long_good_start_control, led_long_good_end_control = [], []
    led_long_poor_start_control, led_long_poor_end_control = [], []
    led_long_no_start_control,   led_long_no_end_control   = [], []

    # =========================
    # CHEMO accumulators
    # =========================

    # Short
    F_short_good_chemo = []
    F_short_poor_chemo = []
    F_short_no_chemo = []

    puff_short_good_start_chemo, puff_short_good_end_chemo = [], []
    puff_short_poor_start_chemo, puff_short_poor_end_chemo = [], []
    puff_short_no_start_chemo,   puff_short_no_end_chemo   = [], []

    cr_short_good_start_chemo, cr_short_good_end_chemo = [], []
    cr_short_poor_start_chemo, cr_short_poor_end_chemo = [], []
    cr_short_no_start_chemo,   cr_short_no_end_chemo   = [], []

    led_short_good_start_chemo, led_short_good_end_chemo = [], []
    led_short_poor_start_chemo, led_short_poor_end_chemo = [], []
    led_short_no_start_chemo,   led_short_no_end_chemo   = [], []

    # Long
    F_long_good_chemo = []
    F_long_poor_chemo = []
    F_long_no_chemo = []

    puff_long_good_start_chemo, puff_long_good_end_chemo = [], []
    puff_long_poor_start_chemo, puff_long_poor_end_chemo = [], []
    puff_long_no_start_chemo,   puff_long_no_end_chemo   = [], []

    cr_long_good_start_chemo, cr_long_good_end_chemo = [], []
    cr_long_poor_start_chemo, cr_long_poor_end_chemo = [], []
    cr_long_no_start_chemo,   cr_long_no_end_chemo   = [], []

    led_long_good_start_chemo, led_long_good_end_chemo = [], []
    led_long_poor_start_chemo, led_long_poor_end_chemo = [], []
    led_long_no_start_chemo,   led_long_no_end_chemo   = [], []

    n_sessions_used = 0
    n_control_sessions_used = 0
    n_chemo_sessions_used = 0
    led_duration_all = []
    
    excluded_short_control_baseline = 0
    excluded_long_control_baseline = 0
    excluded_short_chemo_baseline = 0
    excluded_long_chemo_baseline = 0
    
    for filepath in data_files:
        S = loadmat(filepath, squeeze_me=True, struct_as_record=False)

        if "SessionData" not in S:
            continue

        SD = S["SessionData"]

        chemo_flag = get_field(SD, "Chemogenetics", 0)
        try:
            chemo_flag = int(np.asarray(chemo_flag).squeeze())
        except Exception:
            chemo_flag = 0

        n_sessions_used += 1
        if chemo_flag == 1:
            n_chemo_sessions_used += 1
        else:
            n_control_sessions_used += 1

        raw_events = get_field(SD, "RawEvents", None)
        trials = ensure_trial_list(get_field(raw_events, "Trial", None))
        overall_max = collect_overall_max(trials)
        if not np.isfinite(overall_max):
            print(f"Skipping {os.path.basename(filepath)}: could not compute overall_max  FEC_AvgClassified_CR.py:695 - 05_FEC_AvgClassified_CR.py:695")
            continue
        
        for tr in trials:
            states = get_field(tr, "States", None)
            events = get_field(tr, "Events", None)
            data = get_field(tr, "Data", None)

            if states is None or events is None or data is None:
                continue

            # exclude timeout
            if exclude_timeout and has_field(states, "CheckEyeOpenTimeout"):
                v = safe_array(get_field(states, "CheckEyeOpenTimeout", np.nan))
                if np.any(np.isfinite(v)):
                    continue

            # exclude probe
            if exclude_probe and has_field(data, "IsProbeTrial"):
                is_probe = np.asarray(get_field(data, "IsProbeTrial", 0)).squeeze()
                if np.any(is_probe == 1):
                    continue

            # required fields for FEC reconstruction
            FECTimes = safe_array(get_field(data, "FECTimes", None))
            eyeAreaPixels = safe_array(get_field(data, "eyeAreaPixels", None))

            GT1_start = get_field(events, "GlobalTimer1_Start", None)
            GT1_end   = get_field(events, "GlobalTimer1_End", None)
            GT2_start = get_field(events, "GlobalTimer2_Start", None)
            GT2_end   = get_field(events, "GlobalTimer2_End", None)

            LED_Puff_ISI = safe_array(get_field(states, "LED_Puff_ISI", None))

            if FECTimes.size == 0 or eyeAreaPixels.size == 0:
                continue
            if FECTimes.size != eyeAreaPixels.size:
                continue

            # reconstruct normalized FEC like your other code
            FEC = 1.0 - eyeAreaPixels / overall_max
            
            t_led_abs = float(np.asarray(GT1_start).squeeze())
            t_puff_abs = float(np.asarray(GT2_start).squeeze())

            # LED-aligned time
            t_rel = FECTimes - t_led_abs
            puff_rel = t_puff_abs - t_led_abs

            # LED window in LED-aligned coordinates
            led_start_rel = 0.0

            if GT1_end is not None and np.asarray(GT1_end).size > 0:
                t_led_end_abs = float(np.asarray(GT1_end).squeeze())
                led_end_rel = t_led_end_abs - t_led_abs
            else:
                led_end_rel = np.nan

            puff_start_rel = t_puff_abs - t_led_abs

            if GT2_end is not None and np.asarray(GT2_end).size > 0:
                t_puff_end_abs = float(np.asarray(GT2_end).squeeze())
                puff_end_rel = t_puff_end_abs - t_led_abs
            else:
                puff_end_rel = np.nan

            # smooth before interpolation
            if smooth_win > 1:
                FEC = smooth_trace(FEC, smooth_win)

            # interpolate onto common grid
            Fq = interp_to_grid(t_rel, FEC, t_grid, method=interp_method)

            # determine block type
            isi_dur = LED_Puff_ISI[1] - LED_Puff_ISI[0]
            is_short = isi_dur <= short_isi_max
            block_label = "short" if is_short else "long"

            # ---------------------------------
            # baseline filter
            # ---------------------------------
            baseline_val = compute_baseline(Fq, t_grid, baseline_window=baseline_window)

            if np.isfinite(baseline_val):
                if chemo_flag == 1:
                    if baseline_val > baseline_max_chemo:
                        if is_short:
                            excluded_short_chemo_baseline += 1
                        else:
                            excluded_long_chemo_baseline += 1
                        continue
                else:
                    if baseline_val > baseline_max_control:
                        if is_short:
                            excluded_short_control_baseline += 1
                        else:
                            excluded_long_control_baseline += 1
                        continue

            

            CR_category, _, _, cr_start_rel, cr_end_rel = classify_cr_by_block(
                t_grid,
                Fq,
                0,
                puff_rel,
                good_cr_threshold,
                poor_cr_threshold,
                block_label,
                short_pre_ms=short_cr_pre_ms,
                short_post_ms=short_cr_post_ms,
                long_pre_ms=long_cr_pre_ms,
                long_post_ms=long_cr_post_ms
            )

            if chemo_flag == 1:
                    # =========================
                    # CHEMO
                    # =========================
                if is_short:
                    if CR_category == "Good CR":
                        F_short_good_chemo.append(Fq)
                        puff_short_good_start_chemo.append(puff_start_rel)
                        puff_short_good_end_chemo.append(puff_end_rel)
                        cr_short_good_start_chemo.append(cr_start_rel)
                        cr_short_good_end_chemo.append(cr_end_rel)
                        led_short_good_start_chemo.append(led_start_rel)
                        led_short_good_end_chemo.append(led_end_rel)

                    elif CR_category == "Poor CR":
                        F_short_poor_chemo.append(Fq)
                        puff_short_poor_start_chemo.append(puff_start_rel)
                        puff_short_poor_end_chemo.append(puff_end_rel)
                        cr_short_poor_start_chemo.append(cr_start_rel)
                        cr_short_poor_end_chemo.append(cr_end_rel)
                        led_short_poor_start_chemo.append(led_start_rel)
                        led_short_poor_end_chemo.append(led_end_rel)

                    else:
                        F_short_no_chemo.append(Fq)
                        puff_short_no_start_chemo.append(puff_start_rel)
                        puff_short_no_end_chemo.append(puff_end_rel)
                        cr_short_no_start_chemo.append(cr_start_rel)
                        cr_short_no_end_chemo.append(cr_end_rel)
                        led_short_no_start_chemo.append(led_start_rel)
                        led_short_no_end_chemo.append(led_end_rel)

                else:
                    if CR_category == "Good CR":
                        F_long_good_chemo.append(Fq)
                        puff_long_good_start_chemo.append(puff_start_rel)
                        puff_long_good_end_chemo.append(puff_end_rel)
                        cr_long_good_start_chemo.append(cr_start_rel)
                        cr_long_good_end_chemo.append(cr_end_rel)
                        led_long_good_start_chemo.append(led_start_rel)
                        led_long_good_end_chemo.append(led_end_rel)

                    elif CR_category == "Poor CR":
                        F_long_poor_chemo.append(Fq)
                        puff_long_poor_start_chemo.append(puff_start_rel)
                        puff_long_poor_end_chemo.append(puff_end_rel)
                        cr_long_poor_start_chemo.append(cr_start_rel)
                        cr_long_poor_end_chemo.append(cr_end_rel)
                        led_long_poor_start_chemo.append(led_start_rel)
                        led_long_poor_end_chemo.append(led_end_rel)

                    else:
                        F_long_no_chemo.append(Fq)
                        puff_long_no_start_chemo.append(puff_start_rel)
                        puff_long_no_end_chemo.append(puff_end_rel)
                        cr_long_no_start_chemo.append(cr_start_rel)
                        cr_long_no_end_chemo.append(cr_end_rel)
                        led_long_no_start_chemo.append(led_start_rel)
                        led_long_no_end_chemo.append(led_end_rel)

            else:

                if is_short:
                    if CR_category == "Good CR":
                        F_short_good_control.append(Fq)
                        puff_short_good_start_control.append(puff_start_rel)
                        puff_short_good_end_control.append(puff_end_rel)
                        cr_short_good_start_control.append(cr_start_rel)
                        cr_short_good_end_control.append(cr_end_rel)
                        led_short_good_start_control.append(led_start_rel)
                        led_short_good_end_control.append(led_end_rel)

                    elif CR_category == "Poor CR":
                        F_short_poor_control.append(Fq)
                        puff_short_poor_start_control.append(puff_start_rel)
                        puff_short_poor_end_control.append(puff_end_rel)
                        cr_short_poor_start_control.append(cr_start_rel)
                        cr_short_poor_end_control.append(cr_end_rel)
                        led_short_poor_start_control.append(led_start_rel)
                        led_short_poor_end_control.append(led_end_rel)

                    else:
                        F_short_no_control.append(Fq)
                        puff_short_no_start_control.append(puff_start_rel)
                        puff_short_no_end_control.append(puff_end_rel)
                        cr_short_no_start_control.append(cr_start_rel)
                        cr_short_no_end_control.append(cr_end_rel)
                        led_short_no_start_control.append(led_start_rel)
                        led_short_no_end_control.append(led_end_rel)

                else:
                    if CR_category == "Good CR":
                        F_long_good_control.append(Fq)
                        puff_long_good_start_control.append(puff_start_rel)
                        puff_long_good_end_control.append(puff_end_rel)
                        cr_long_good_start_control.append(cr_start_rel)
                        cr_long_good_end_control.append(cr_end_rel)
                        led_long_good_start_control.append(led_start_rel)
                        led_long_good_end_control.append(led_end_rel)

                    elif CR_category == "Poor CR":
                        F_long_poor_control.append(Fq)
                        puff_long_poor_start_control.append(puff_start_rel)
                        puff_long_poor_end_control.append(puff_end_rel)
                        cr_long_poor_start_control.append(cr_start_rel)
                        cr_long_poor_end_control.append(cr_end_rel)
                        led_long_poor_start_control.append(led_start_rel)
                        led_long_poor_end_control.append(led_end_rel)

                    else:
                        F_long_no_control.append(Fq)
                        puff_long_no_start_control.append(puff_start_rel)
                        puff_long_no_end_control.append(puff_end_rel)
                        cr_long_no_start_control.append(cr_start_rel)
                        cr_long_no_end_control.append(cr_end_rel)
                        led_long_no_start_control.append(led_start_rel)
                        led_long_no_end_control.append(led_end_rel)
    # =========================
    # CONTROL
    # =========================
    S_short_good_control = pack_stats(
        t_grid, F_short_good_control,
        puff_short_good_start_control, puff_short_good_end_control,
        cr_short_good_start_control, cr_short_good_end_control,
        led_short_good_start_control, led_short_good_end_control
    )

    S_short_poor_control = pack_stats(
        t_grid, F_short_poor_control,
        puff_short_poor_start_control, puff_short_poor_end_control,
        cr_short_poor_start_control, cr_short_poor_end_control,
        led_short_poor_start_control, led_short_poor_end_control
    )

    S_short_no_control = pack_stats(
        t_grid, F_short_no_control,
        puff_short_no_start_control, puff_short_no_end_control,
        cr_short_no_start_control, cr_short_no_end_control,
        led_short_no_start_control, led_short_no_end_control
    )

    S_long_good_control = pack_stats(
        t_grid, F_long_good_control,
        puff_long_good_start_control, puff_long_good_end_control,
        cr_long_good_start_control, cr_long_good_end_control,
        led_long_good_start_control, led_long_good_end_control
    )

    S_long_poor_control = pack_stats(
        t_grid, F_long_poor_control,
        puff_long_poor_start_control, puff_long_poor_end_control,
        cr_long_poor_start_control, cr_long_poor_end_control,
        led_long_poor_start_control, led_long_poor_end_control
    )

    S_long_no_control = pack_stats(
        t_grid, F_long_no_control,
        puff_long_no_start_control, puff_long_no_end_control,
        cr_long_no_start_control, cr_long_no_end_control,
        led_long_no_start_control, led_long_no_end_control
    )

    # =========================
    # CHEMO
    # =========================
    S_short_good_chemo = pack_stats(
        t_grid, F_short_good_chemo,
        puff_short_good_start_chemo, puff_short_good_end_chemo,
        cr_short_good_start_chemo, cr_short_good_end_chemo,
        led_short_good_start_chemo, led_short_good_end_chemo
    )

    S_short_poor_chemo = pack_stats(
        t_grid, F_short_poor_chemo,
        puff_short_poor_start_chemo, puff_short_poor_end_chemo,
        cr_short_poor_start_chemo, cr_short_poor_end_chemo,
        led_short_poor_start_chemo, led_short_poor_end_chemo
    )

    S_short_no_chemo = pack_stats(
        t_grid, F_short_no_chemo,
        puff_short_no_start_chemo, puff_short_no_end_chemo,
        cr_short_no_start_chemo, cr_short_no_end_chemo,
        led_short_no_start_chemo, led_short_no_end_chemo
    )

    S_long_good_chemo = pack_stats(
        t_grid, F_long_good_chemo,
        puff_long_good_start_chemo, puff_long_good_end_chemo,
        cr_long_good_start_chemo, cr_long_good_end_chemo,
        led_long_good_start_chemo, led_long_good_end_chemo
    )

    S_long_poor_chemo = pack_stats(
        t_grid, F_long_poor_chemo,
        puff_long_poor_start_chemo, puff_long_poor_end_chemo,
        cr_long_poor_start_chemo, cr_long_poor_end_chemo,
        led_long_poor_start_chemo, led_long_poor_end_chemo
    )

    S_long_no_chemo = pack_stats(
        t_grid, F_long_no_chemo,
        puff_long_no_start_chemo, puff_long_no_end_chemo,
        cr_long_no_start_chemo, cr_long_no_end_chemo,
        led_long_no_start_chemo, led_long_no_end_chemo
    )
    # ------------ plot 3x2 ------------
    fig, axs = plt.subplots(3, 2, figsize=(11.5, 8.5))
    axs = axs.ravel()

    plot_panel_two_groups(axs[0], S_short_good_control, S_short_good_chemo, "Good CR", "Short")
    plot_panel_two_groups(axs[1], S_long_good_control,  S_long_good_chemo,  "Good CR", "Long")

    plot_panel_two_groups(axs[2], S_short_poor_control, S_short_poor_chemo, "Poor CR", "Short")
    plot_panel_two_groups(axs[3], S_long_poor_control,  S_long_poor_chemo,  "Poor CR", "Long")

    plot_panel_two_groups(axs[4], S_short_no_control,   S_short_no_chemo,   "No CR",   "Short")
    plot_panel_two_groups(axs[5], S_long_no_control,    S_long_no_chemo,    "No CR",   "Long")

    fig.suptitle(
        f"Pooled Avg FEC CR-classified Short/Long\n"
        f"Sessions: {first_date} -- {last_date} | "
        f"Control sessions={n_control_sessions_used}, "
        f"Chemo sessions={n_chemo_sessions_used}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    outname = f"PooledAvgFEC_CRclassified_ShortLong_{first_tok}_to_{last_tok}.pdf"
    plt.savefig(outname, bbox_inches="tight")
    print(f"Saved figure to: {outname}  FEC_AvgClassified_CR.py:1041 - 05_FEC_AvgClassified_CR.py:1041")

    print("\nHighbaseline trial exclusion summary:  FEC_AvgClassified_CR.py:1043 - 05_FEC_AvgClassified_CR.py:1043")
    print(f"Baseline window: {baseline_window}  FEC_AvgClassified_CR.py:1044 - 05_FEC_AvgClassified_CR.py:1044")
    print(f"Control baseline threshold: > {baseline_max_control}  FEC_AvgClassified_CR.py:1045 - 05_FEC_AvgClassified_CR.py:1045")
    print(f"Chemo baseline threshold:   > {baseline_max_chemo}  FEC_AvgClassified_CR.py:1046 - 05_FEC_AvgClassified_CR.py:1046")

    print(f"Excluded control shortblock trials: {excluded_short_control_baseline}  FEC_AvgClassified_CR.py:1048 - 05_FEC_AvgClassified_CR.py:1048")
    print(f"Excluded control longblock trials:  {excluded_long_control_baseline}  FEC_AvgClassified_CR.py:1049 - 05_FEC_AvgClassified_CR.py:1049")
    print(f"Excluded chemo shortblock trials:   {excluded_short_chemo_baseline}  FEC_AvgClassified_CR.py:1050 - 05_FEC_AvgClassified_CR.py:1050")
    print(f"Excluded chemo longblock trials:    {excluded_long_chemo_baseline}  FEC_AvgClassified_CR.py:1051 - 05_FEC_AvgClassified_CR.py:1051")

    plt.show()


if __name__ == "__main__":
    main()