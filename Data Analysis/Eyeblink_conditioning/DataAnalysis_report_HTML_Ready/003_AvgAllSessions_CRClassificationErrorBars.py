import os
import glob
import re
import math
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MPLCONFIGDIR = os.path.join(SCRIPT_DIR, ".matplotlib")
XDG_CACHE_HOME = os.path.join(SCRIPT_DIR, ".cache")
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.makedirs(XDG_CACHE_HOME, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
os.environ.setdefault("XDG_CACHE_HOME", XDG_CACHE_HOME)
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from datetime import datetime


# ============================================================
# Helpers
# ============================================================

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


def scalar_or_nan(x):
    if x is None:
        return np.nan
    arr = np.asarray(x).squeeze()
    if arr.size == 0:
        return np.nan
    if arr.ndim == 0:
        if arr is None:
            return np.nan
        return float(arr)
    return float(arr.flat[0])


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


def collect_overall_max(trials):
    """
    Version-robust normalization factor.
    Prefer totalEllipsePixels if available; fallback to eyeAreaPixels.
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


def get_trial_timing(tr, short_isi_max=0.30):
    """
    Version-robust timing extractor.

    Returns
    -------
    t_led_abs, t_led_end_abs, t_puff_abs, t_puff_end_abs, isi_dur, is_short, block_label
    """
    states = get_field(tr, "States", None)
    events = get_field(tr, "Events", None)
    data = get_field(tr, "Data", None)

    t_led_abs = scalar_or_nan(get_field(events, "GlobalTimer1_Start", None))
    t_led_end_abs = scalar_or_nan(get_field(events, "GlobalTimer1_End", None))

    t_puff_abs = scalar_or_nan(get_field(events, "GlobalTimer2_Start", None))
    t_puff_end_abs = scalar_or_nan(get_field(events, "GlobalTimer2_End", None))

    if not np.isfinite(t_puff_abs):
        for cand in ["AirPuff_Onset", "AirPuff_Ons", "AirPuff_On", "AirPuff_OnsetTime"]:
            val = get_field(data, cand, None)
            t_puff_abs = scalar_or_nan(val)
            if np.isfinite(t_puff_abs):
                break

    led_puff_isi = safe_array(get_field(states, "LED_Puff_ISI", None))
    if led_puff_isi.size >= 2:
        isi_dur = float(led_puff_isi[1] - led_puff_isi[0])
    else:
        isi_dur = scalar_or_nan(get_field(data, "ISI", None))

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
            block_label = "short" if np.isfinite(isi_dur) and isi_dur <= short_isi_max else "long"

    is_short = (block_label == "short")
    return t_led_abs, t_led_end_abs, t_puff_abs, t_puff_end_abs, isi_dur, is_short, block_label


# ============================================================
# Classifier
# ============================================================

def classify_cr_by_block(time, signal, t_led, t_puff, good_cr_threshold, poor_cr_threshold, block_label):
    """
    More permissive CR classifier.

    Short:
        Good window = [t_puff - 0.025, t_puff + 0.020]
        Poor window = [t_led, t_puff - 0.025]

    Long:
        Good window = [t_puff - 0.050, t_puff]
        Poor window = [t_led, t_puff - 0.050]

    Good CR if either:
      1) CR-window mean exceeds baseline by threshold
      2) CR-window peak exceeds baseline by threshold
      3) CR-window mean is slightly above baseline
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if len(time) != len(signal):
        return "No CR", np.nan, np.nan

    baseline_idx = (time >= (t_led - 0.2)) & (time <= t_led)
    if not np.any(baseline_idx):
        return "No CR", np.nan, np.nan

    block_label = str(block_label).strip().lower()

    if block_label == "short":
        good_start = t_puff - 0.025
        good_end   = t_puff + 0.020
        good_thr   = 0.03
        peak_thr   = 0.035
        small_mean_thr = 0.01
    elif block_label == "long":
        good_start = t_puff - 0.050
        good_end   = t_puff
        good_thr   = good_cr_threshold
        peak_thr   = good_cr_threshold
        small_mean_thr = 0.01
    else:
        return "No CR", np.nan, np.nan

    poor_start = t_led
    poor_end   = good_start

    good_idx = (time >= good_start) & (time <= good_end)
    poor_idx = (time >= poor_start) & (time < poor_end)

    if not np.any(good_idx):
        return "No CR", np.nan, np.nan

    baseline_amp = np.nanmean(signal[baseline_idx])
    good_signal = signal[good_idx]
    good_amp = np.nanmean(good_signal)
    good_peak = np.nanmax(good_signal)

    good_mean_above = good_amp - baseline_amp
    good_peak_above = good_peak - baseline_amp
    good_mean_above_baseline = good_mean_above > small_mean_thr

    if np.any(poor_idx):
        poor_amp = np.nanmean(signal[poor_idx])
        poor_mean_above = poor_amp - baseline_amp
    else:
        poor_mean_above = -np.inf

    if (good_mean_above >= good_thr) or (good_peak_above >= peak_thr) or good_mean_above_baseline:
        category = "Good CR"
    elif poor_mean_above >= poor_cr_threshold:
        category = "Poor CR"
    else:
        category = "No CR"

    return category, good_amp, baseline_amp


# ============================================================
# Main
# ============================================================

def main():
    script_dir = SCRIPT_DIR
    data_files = sorted(glob.glob(os.path.join(script_dir, "*.mat")))
    if len(data_files) == 0:
        print(f"No .mat files found in {script_dir}.  AvgAllSessions_CRClassificationErrorBars.py:265 - 003_AvgAllSessions_CRClassificationErrorBars.py:274")
        return
    print(f"Found {len(data_files)} .mat files in {script_dir} - 003_AvgAllSessions_CRClassificationErrorBars.py:276")

    fps = 250
    seconds_before = 0.5
    seconds_after = 2.0
    frames_before = int(fps * seconds_before)
    frames_after = int(fps * seconds_after)
    smooth_win = 5
    short_isi_max = 0.30
    good_cr_threshold = 0.05
    poor_cr_threshold = 0.02

    exclude_probe = True
    exclude_timeout = True

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

    # per-session fractions
    good_short_frac = []
    poor_short_frac = []
    no_short_frac = []

    good_long_frac = []
    poor_long_frac = []
    no_long_frac = []

    used_sessions = 0

    # pooled counts — control sessions
    ctrl_total_good_short = 0
    ctrl_total_poor_short = 0
    ctrl_total_no_short = 0
    ctrl_total_short_trials = 0
    ctrl_total_good_long = 0
    ctrl_total_poor_long = 0
    ctrl_total_no_long = 0
    ctrl_total_long_trials = 0

    # pooled counts — chemo sessions
    chemo_total_good_short = 0
    chemo_total_poor_short = 0
    chemo_total_no_short = 0
    chemo_total_short_trials = 0
    chemo_total_good_long = 0
    chemo_total_poor_long = 0
    chemo_total_no_long = 0
    chemo_total_long_trials = 0

    n_ctrl_sessions = 0
    n_chemo_sessions = 0

    for filepath in data_files:
        print(f"Processing: {filepath}  AvgAllSessions_CRClassificationErrorBars.py:323 - 003_AvgAllSessions_CRClassificationErrorBars.py:346")
        SD = matlab_load_sessiondata(filepath)
        is_chemo = is_chemogenetics_session(SD)

        raw_events = get_field(SD, "RawEvents", None)
        trials = ensure_trial_list(get_field(raw_events, "Trial", None))

        overall_max = collect_overall_max(trials)
        if not np.isfinite(overall_max):
            print(f"Skipping {os.path.basename(filepath)}: could not compute overall_max  AvgAllSessions_CRClassificationErrorBars.py:334 - 003_AvgAllSessions_CRClassificationErrorBars.py:372")
            continue

        num_good_short = 0
        num_poor_short = 0
        num_no_short = 0
        num_short_trials = 0

        num_good_long = 0
        num_poor_long = 0
        num_no_long = 0
        num_long_trials = 0
        

        for tr in trials:
            states = get_field(tr, "States", None)
            events = get_field(tr, "Events", None)
            data = get_field(tr, "Data", None)

            if states is None or events is None or data is None:
                continue

            if exclude_timeout and has_field(states, "CheckEyeOpenTimeout"):
                v = safe_array(get_field(states, "CheckEyeOpenTimeout", np.nan))
                if np.any(np.isfinite(v)):
                    continue

            if exclude_probe and has_field(data, "IsProbeTrial"):
                is_probe = np.asarray(get_field(data, "IsProbeTrial", 0)).squeeze()
                if np.any(is_probe == 1):
                    continue

            FECTimes = safe_array(get_field(data, "FECTimes", None))
            eyeAreaPixels = safe_array(get_field(data, "eyeAreaPixels", None))

            if FECTimes.size == 0 or eyeAreaPixels.size == 0:
                continue
            if FECTimes.size != eyeAreaPixels.size:
                continue

            FEC = 1.0 - eyeAreaPixels / overall_max

            t_led_abs, t_led_end_abs, t_puff_abs, t_puff_end_abs, isi_dur, is_short, block_label = get_trial_timing(
                tr, short_isi_max=short_isi_max
            )

            if not np.isfinite(t_led_abs) or not np.isfinite(t_puff_abs) or not np.isfinite(isi_dur):
                continue

            t_rel = FECTimes - t_led_abs

            closest_idx = int(np.argmin(np.abs(t_rel)))
            start_idx = max(0, closest_idx - frames_before)
            stop_idx = min(len(t_rel) - 1, closest_idx + frames_after)

            t_trim = t_rel[start_idx:stop_idx + 1]
            fec_trim = FEC[start_idx:stop_idx + 1]

            if len(t_trim) < 5 or len(fec_trim) < 5:
                continue

            if smooth_win > 1:
                t_trim = smooth_trace(t_trim, smooth_win)
                fec_trim = smooth_trace(fec_trim, smooth_win)

            t_led = t_led_end_abs - t_led_abs if np.isfinite(t_led_end_abs) else 0.0
            t_puff = t_puff_abs - t_led_abs

            cr_category, _, _ = classify_cr_by_block(
                t_trim, fec_trim, t_led, t_puff, good_cr_threshold, poor_cr_threshold, block_label
            )

            if is_short:
                num_short_trials += 1
                if cr_category == "Good CR":
                    num_good_short += 1
                elif cr_category == "Poor CR":
                    num_poor_short += 1
                else:
                    num_no_short += 1
            else:
                num_long_trials += 1
                if cr_category == "Good CR":
                    num_good_long += 1
                elif cr_category == "Poor CR":
                    num_poor_long += 1
                else:
                    num_no_long += 1

        total_trials = num_short_trials + num_long_trials
        if total_trials == 0:
            continue

        total_cr = num_good_short + num_poor_short + num_good_long + num_poor_long
        total_cr_fraction = total_cr / float(total_trials)
        if total_cr_fraction <= 0.20:
            print(
                f"Skipping {os.path.basename(filepath)}: CR fraction {total_cr_fraction:.2f} <= 0.20 "
                "- AvgAllSessions_CRClassificationErrorBars.py"
            )
            continue

        good_short_frac.append(num_good_short / max(num_short_trials, 1))
        poor_short_frac.append(num_poor_short / max(num_short_trials, 1))
        no_short_frac.append(num_no_short / max(num_short_trials, 1))

        good_long_frac.append(num_good_long / max(num_long_trials, 1))
        poor_long_frac.append(num_poor_long / max(num_long_trials, 1))
        no_long_frac.append(num_no_long / max(num_long_trials, 1))

        if is_chemo:
            chemo_total_good_short   += num_good_short
            chemo_total_poor_short   += num_poor_short
            chemo_total_no_short     += num_no_short
            chemo_total_short_trials += num_short_trials
            chemo_total_good_long    += num_good_long
            chemo_total_poor_long    += num_poor_long
            chemo_total_no_long      += num_no_long
            chemo_total_long_trials  += num_long_trials
            n_chemo_sessions += 1
        else:
            ctrl_total_good_short   += num_good_short
            ctrl_total_poor_short   += num_poor_short
            ctrl_total_no_short     += num_no_short
            ctrl_total_short_trials += num_short_trials
            ctrl_total_good_long    += num_good_long
            ctrl_total_poor_long    += num_poor_long
            ctrl_total_no_long      += num_no_long
            ctrl_total_long_trials  += num_long_trials
            n_ctrl_sessions += 1

        used_sessions += 1

    if used_sessions == 0:
        print("No sessions with usable trials found.  AvgAllSessions_CRClassificationErrorBars.py:457 - 003_AvgAllSessions_CRClassificationErrorBars.py:506")
        return

    # control pooled fractions
    ctrl_short_good_mean = ctrl_total_good_short / max(ctrl_total_short_trials, 1)
    ctrl_short_poor_mean = ctrl_total_poor_short / max(ctrl_total_short_trials, 1)
    ctrl_short_no_mean   = ctrl_total_no_short   / max(ctrl_total_short_trials, 1)
    ctrl_long_good_mean  = ctrl_total_good_long  / max(ctrl_total_long_trials, 1)
    ctrl_long_poor_mean  = ctrl_total_poor_long  / max(ctrl_total_long_trials, 1)
    ctrl_long_no_mean    = ctrl_total_no_long    / max(ctrl_total_long_trials, 1)

    ctrl_short_good_sem = np.sqrt(ctrl_short_good_mean * (1 - ctrl_short_good_mean) / max(ctrl_total_short_trials, 1))
    ctrl_short_poor_sem = np.sqrt(ctrl_short_poor_mean * (1 - ctrl_short_poor_mean) / max(ctrl_total_short_trials, 1))
    ctrl_short_no_sem   = np.sqrt(ctrl_short_no_mean   * (1 - ctrl_short_no_mean)   / max(ctrl_total_short_trials, 1))
    ctrl_long_good_sem  = np.sqrt(ctrl_long_good_mean  * (1 - ctrl_long_good_mean)  / max(ctrl_total_long_trials, 1))
    ctrl_long_poor_sem  = np.sqrt(ctrl_long_poor_mean  * (1 - ctrl_long_poor_mean)  / max(ctrl_total_long_trials, 1))
    ctrl_long_no_sem    = np.sqrt(ctrl_long_no_mean    * (1 - ctrl_long_no_mean)    / max(ctrl_total_long_trials, 1))

    # chemo pooled fractions
    chemo_short_good_mean = chemo_total_good_short / max(chemo_total_short_trials, 1)
    chemo_short_poor_mean = chemo_total_poor_short / max(chemo_total_short_trials, 1)
    chemo_short_no_mean   = chemo_total_no_short   / max(chemo_total_short_trials, 1)
    chemo_long_good_mean  = chemo_total_good_long  / max(chemo_total_long_trials, 1)
    chemo_long_poor_mean  = chemo_total_poor_long  / max(chemo_total_long_trials, 1)
    chemo_long_no_mean    = chemo_total_no_long    / max(chemo_total_long_trials, 1)

    chemo_short_good_sem = np.sqrt(chemo_short_good_mean * (1 - chemo_short_good_mean) / max(chemo_total_short_trials, 1))
    chemo_short_poor_sem = np.sqrt(chemo_short_poor_mean * (1 - chemo_short_poor_mean) / max(chemo_total_short_trials, 1))
    chemo_short_no_sem   = np.sqrt(chemo_short_no_mean   * (1 - chemo_short_no_mean)   / max(chemo_total_short_trials, 1))
    chemo_long_good_sem  = np.sqrt(chemo_long_good_mean  * (1 - chemo_long_good_mean)  / max(chemo_total_long_trials, 1))
    chemo_long_poor_sem  = np.sqrt(chemo_long_poor_mean  * (1 - chemo_long_poor_mean)  / max(chemo_total_long_trials, 1))
    chemo_long_no_sem    = np.sqrt(chemo_long_no_mean    * (1 - chemo_long_no_mean)    / max(chemo_total_long_trials, 1))

    plt.rcParams.update({
        "font.size": 8,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    fig.subplots_adjust(left=0.18, right=0.94, bottom=0.28, top=0.62)

    # Compare categories first, with Short/Long nested inside each category.
    # Within each Short/Long pair, control is left and chemo is right.
    cat_centers = np.array([0.0, 0.68, 1.36])
    block_off = 0.15
    group_off = 0.045

    x_no_short = cat_centers[0] - block_off
    x_no_long = cat_centers[0] + block_off
    x_poor_short = cat_centers[1] - block_off
    x_poor_long = cat_centers[1] + block_off
    x_good_short = cat_centers[2] - block_off
    x_good_long = cat_centers[2] + block_off

    ctrl_x_no = np.array([x_no_short - group_off, x_no_long - group_off])
    chemo_x_no = np.array([x_no_short + group_off, x_no_long + group_off])
    ctrl_x_poor = np.array([x_poor_short - group_off, x_poor_long - group_off])
    chemo_x_poor = np.array([x_poor_short + group_off, x_poor_long + group_off])
    ctrl_x_good = np.array([x_good_short - group_off, x_good_long - group_off])
    chemo_x_good = np.array([x_good_short + group_off, x_good_long + group_off])

    ctrl_vals_no   = np.array([ctrl_short_no_mean,   ctrl_long_no_mean])
    ctrl_vals_poor = np.array([ctrl_short_poor_mean, ctrl_long_poor_mean])
    ctrl_vals_good = np.array([ctrl_short_good_mean, ctrl_long_good_mean])
    ctrl_sem_no    = np.array([ctrl_short_no_sem,    ctrl_long_no_sem])
    ctrl_sem_poor  = np.array([ctrl_short_poor_sem,  ctrl_long_poor_sem])
    ctrl_sem_good  = np.array([ctrl_short_good_sem,  ctrl_long_good_sem])

    chemo_vals_no   = np.array([chemo_short_no_mean,   chemo_long_no_mean])
    chemo_vals_poor = np.array([chemo_short_poor_mean, chemo_long_poor_mean])
    chemo_vals_good = np.array([chemo_short_good_mean, chemo_long_good_mean])
    chemo_sem_no    = np.array([chemo_short_no_sem,    chemo_long_no_sem])
    chemo_sem_poor  = np.array([chemo_short_poor_sem,  chemo_long_poor_sem])
    chemo_sem_good  = np.array([chemo_short_good_sem,  chemo_long_good_sem])

    ctrl_colors = {
        "No CR": "#9E9E9E",
        "Poor CR": "#555555",
        "Good CR": "#000000",
    }
    chemo_colors = {
        "No CR": "#F2A3A3",
        "Poor CR": "#D14949",
        "Good CR": "#8B0000",
    }

    def plot_group(x, vals, sem, color):
        ax.errorbar(
            x,
            vals,
            yerr=sem,
            fmt="none",
            ecolor=color,
            capsize=3.0,
            capthick=0.8,
            elinewidth=0.8,
            zorder=4,
        )
        ax.scatter(
            x,
            vals,
            s=9,
            c=color,
            edgecolors="white",
            linewidths=0.25,
            zorder=5,
        )

    plot_group(ctrl_x_no, ctrl_vals_no, ctrl_sem_no, ctrl_colors["No CR"])
    plot_group(ctrl_x_poor, ctrl_vals_poor, ctrl_sem_poor, ctrl_colors["Poor CR"])
    plot_group(ctrl_x_good, ctrl_vals_good, ctrl_sem_good, ctrl_colors["Good CR"])

    plot_group(chemo_x_no, chemo_vals_no, chemo_sem_no, chemo_colors["No CR"])
    plot_group(chemo_x_poor, chemo_vals_poor, chemo_sem_poor, chemo_colors["Poor CR"])
    plot_group(chemo_x_good, chemo_vals_good, chemo_sem_good, chemo_colors["Good CR"])

    minor_ticks = [
        x_no_short, x_no_long,
        x_poor_short, x_poor_long,
        x_good_short, x_good_long,
    ]
    ax.set_xticks(minor_ticks)
    ax.set_xticklabels(["S", "L", "S", "L", "S", "L"], fontsize=8)
    ax.set_xlim(cat_centers[0] - 0.38, cat_centers[-1] + 0.38)
    ax.set_ylim(-0.04, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_ylabel("Pooled fraction across trials", fontsize=8.5, labelpad=7)
    ax.set_xlabel("CR category and block type", fontsize=9.5, labelpad=11)
    ax.xaxis.set_label_coords(0.5, -0.17)
    ax.tick_params(axis="x", direction="out", labelsize=8, width=0.65, length=3, pad=3)
    ax.tick_params(axis="y", direction="out", labelsize=8, width=0.65, length=3, pad=4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    for xc, label in zip(cat_centers, ["No CR", "Poor CR", "Good CR"]):
        ax.text(
            xc,
            -0.39,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
        )
    for boundary in (0.34, 1.02):
        ax.axvline(boundary, color="0.82", linestyle=":", linewidth=0.8, zorder=0)

    fig.suptitle(
        f"Average Short/Long Classified CR Fractions Across Sessions\n"
        f"Sessions: {first_date} -- {last_date} | Control n={n_ctrl_sessions}, Chemo n={n_chemo_sessions}",
        fontsize=12,
        y=0.96,
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', label='No CR (Control)',
               markerfacecolor=ctrl_colors["No CR"], markeredgecolor='none', markersize=7, linestyle='None'),
        Line2D([0], [0], marker='o', color='none', label='No CR (Chemo)',
               markerfacecolor=chemo_colors["No CR"], markeredgecolor='none', markersize=7, linestyle='None'),
        Line2D([0], [0], marker='o', color='none', label='Poor CR (Control)',
               markerfacecolor=ctrl_colors["Poor CR"], markeredgecolor='none', markersize=7, linestyle='None'),
        Line2D([0], [0], marker='o', color='none', label='Poor CR (Chemo)',
               markerfacecolor=chemo_colors["Poor CR"], markeredgecolor='none', markersize=7, linestyle='None'),
        Line2D([0], [0], marker='o', color='none', label='Good CR (Control)',
               markerfacecolor=ctrl_colors["Good CR"], markeredgecolor='none', markersize=7, linestyle='None'),
        Line2D([0], [0], marker='o', color='none', label='Good CR (Chemo)',
               markerfacecolor=chemo_colors["Good CR"], markeredgecolor='none', markersize=7, linestyle='None'),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.78),
        bbox_transform=fig.transFigure,
        ncol=3,
        borderaxespad=0,
        frameon=False,
        fontsize=7.5,
        handletextpad=0.4,
        columnspacing=1.0
    )

    outname = os.path.join(script_dir, f"AvgCircles_ClassifiedCRFractions_{first_tok}_to_{last_tok}.pdf")
    plt.savefig(outname, bbox_inches="tight")
    print(f"Saved figure to: {outname}  AvgAllSessions_CRClassificationErrorBars.py:555 - 003_AvgAllSessions_CRClassificationErrorBars.py:638")

    plt.show()


if __name__ == "__main__":
    main()
