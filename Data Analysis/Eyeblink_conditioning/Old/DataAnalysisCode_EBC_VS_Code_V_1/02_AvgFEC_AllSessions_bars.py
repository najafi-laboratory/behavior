import os
import glob
import re
import numpy as np
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
    arr = np.asarray(x).squeeze()
    if arr.size == 0:
        return np.nan
    if arr.ndim == 0:
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

    if (good_mean_above >= good_thr) or (good_peak_above >= peak_thr):
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
    data_files = sorted(glob.glob("*_EBC_*.mat"))
    if len(data_files) == 0:
        print("No *_EBC_*.mat files found. - AvgFEC_AllSessions_bars.py:260")
        return

    # settings
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

    # session date range
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

    
    # per-session fractions: Control
    good_short_frac_control = []
    poor_short_frac_control = []
    no_short_frac_control = []

    good_long_frac_control = []
    poor_long_frac_control = []
    no_long_frac_control = []

    # per-session fractions: Chemo
    good_short_frac_chemo = []
    poor_short_frac_chemo = []
    no_short_frac_chemo = []

    good_long_frac_chemo = []
    poor_long_frac_chemo = []
    no_long_frac_chemo = []

    used_session_labels = []

    for filepath in data_files:
        print(f"Processing: {filepath} - AvgFEC_AllSessions_bars.py:320")
        SD = matlab_load_sessiondata(filepath)
        chemo_flag = get_field(SD, "Chemogenetics", 0)
        try:
            chemo_flag = int(np.asarray(chemo_flag).squeeze())
        except Exception:
            chemo_flag = 0

        raw_events = get_field(SD, "RawEvents", None)
        trials = ensure_trial_list(get_field(raw_events, "Trial", None))

        overall_max = collect_overall_max(trials)
        if not np.isfinite(overall_max):
            print(f"Skipping {os.path.basename(filepath)}: could not compute overall_max - AvgFEC_AllSessions_bars.py:333")
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

            # trim around LED onset
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

        # skip sessions with no usable trials at all
        if (num_short_trials + num_long_trials) == 0:
            continue

        gs = num_good_short / max(num_short_trials, 1)
        ps = num_poor_short / max(num_short_trials, 1)
        ns = num_no_short / max(num_short_trials, 1)

        gl = num_good_long / max(num_long_trials, 1)
        pl = num_poor_long / max(num_long_trials, 1)
        nl = num_no_long / max(num_long_trials, 1)

        if chemo_flag == 1:
            good_short_frac_chemo.append(gs)
            poor_short_frac_chemo.append(ps)
            no_short_frac_chemo.append(ns)

            good_long_frac_chemo.append(gl)
            poor_long_frac_chemo.append(pl)
            no_long_frac_chemo.append(nl)
        else:
            good_short_frac_control.append(gs)
            poor_short_frac_control.append(ps)
            no_short_frac_control.append(ns)

            good_long_frac_control.append(gl)
            poor_long_frac_control.append(pl)
            no_long_frac_control.append(nl)

        d = re.search(r"(\d{8})", os.path.basename(filepath))
        if d:
            used_session_labels.append(d.group(1))
        else:
            used_session_labels.append(os.path.basename(filepath))

    n_control = len(good_short_frac_control)
    n_chemo = len(good_short_frac_chemo)

    if n_control == 0 and n_chemo == 0:
        print("No sessions with usable trials found. - AvgFEC_AllSessions_bars.py:463")
        return

    def mean_sem(x):
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return np.nan, np.nan
        return np.nanmean(x), np.nanstd(x, ddof=0) / np.sqrt(len(x))

    # Control
    short_good_mean_control, short_good_sem_control = mean_sem(good_short_frac_control)
    short_poor_mean_control, short_poor_sem_control = mean_sem(poor_short_frac_control)
    short_no_mean_control,   short_no_sem_control   = mean_sem(no_short_frac_control)

    long_good_mean_control, long_good_sem_control = mean_sem(good_long_frac_control)
    long_poor_mean_control, long_poor_sem_control = mean_sem(poor_long_frac_control)
    long_no_mean_control,   long_no_sem_control   = mean_sem(no_long_frac_control)

    # Chemo
    short_good_mean_chemo, short_good_sem_chemo = mean_sem(good_short_frac_chemo)
    short_poor_mean_chemo, short_poor_sem_chemo = mean_sem(poor_short_frac_chemo)
    short_no_mean_chemo,   short_no_sem_chemo   = mean_sem(no_short_frac_chemo)

    long_good_mean_chemo, long_good_sem_chemo = mean_sem(good_long_frac_chemo)
    long_poor_mean_chemo, long_poor_sem_chemo = mean_sem(poor_long_frac_chemo)
    long_no_mean_chemo,   long_no_sem_chemo   = mean_sem(no_long_frac_chemo)
    
    #  Control colors
    color_good = (0.0, 0.0, 0.0)         # black
    color_poor = (0.5, 0.5, 0.5)         # gray
    color_no = (0.83, 0.83, 0.83)        # light gray

    # Chemo color palette
    COLOR_NO_CHEMO   = "#ADD8E6"   # light blue
    COLOR_POOR_CHEMO = "#6C8EBF"   # grey-blue
    COLOR_GOOD_CHEMO = "#003366"   # dark blue
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    x_centers = np.array([0, 1])
    labels = ["Short", "Long"]

    offset = 0.18
    group_shift = 0.05
    width = 0.08

    x_no   = x_centers - offset
    x_poor = x_centers
    x_good = x_centers + offset

    x_no_control   = x_no - group_shift
    x_poor_control = x_poor - group_shift
    x_good_control = x_good - group_shift

    x_no_chemo   = x_no + group_shift
    x_poor_chemo = x_poor + group_shift
    x_good_chemo = x_good + group_shift

    vals_no_control   = np.array([short_no_mean_control,   long_no_mean_control])
    vals_poor_control = np.array([short_poor_mean_control, long_poor_mean_control])
    vals_good_control = np.array([short_good_mean_control, long_good_mean_control])

    vals_no_chemo   = np.array([short_no_mean_chemo,   long_no_mean_chemo])
    vals_poor_chemo = np.array([short_poor_mean_chemo, long_poor_mean_chemo])
    vals_good_chemo = np.array([short_good_mean_chemo, long_good_mean_chemo])

    sem_no_control   = np.array([short_no_sem_control,   long_no_sem_control])
    sem_poor_control = np.array([short_poor_sem_control, long_poor_sem_control])
    sem_good_control = np.array([short_good_sem_control, long_good_sem_control])

    sem_no_chemo   = np.array([short_no_sem_chemo,   long_no_sem_chemo])
    sem_poor_chemo = np.array([short_poor_sem_chemo, long_poor_sem_chemo])
    sem_good_chemo = np.array([short_good_sem_chemo, long_good_sem_chemo])

    # Control bars
    ax.bar(x_no_control, vals_no_control, width,
        color='white', edgecolor='black', linewidth=1.5, label="No CR Control")
    ax.bar(x_poor_control, vals_poor_control, width,
        color='gray', edgecolor='black', linewidth=1.5, label="Poor CR Control")
    ax.bar(x_good_control, vals_good_control, width,
        color='black', edgecolor='black', linewidth=1.5, label="Good CR Control")

    # Chemo bars
    ax.bar(x_no_chemo, vals_no_chemo, width,
        color=COLOR_NO_CHEMO, edgecolor='blue', linewidth=1.5, label="No CR Chemo")
    ax.bar(x_poor_chemo, vals_poor_chemo, width,
        color=COLOR_POOR_CHEMO, edgecolor='blue', linewidth=1.5, label="Poor CR Chemo")
    ax.bar(x_good_chemo, vals_good_chemo, width,
        color=COLOR_GOOD_CHEMO, edgecolor='blue', linewidth=1.5, label="Good CR Chemo")

    # Error bars: control
    ax.errorbar(x_no_control, vals_no_control, yerr=sem_no_control,
                fmt='none', ecolor='black', elinewidth=1, capsize=3)
    ax.errorbar(x_poor_control, vals_poor_control, yerr=sem_poor_control,
                fmt='none', ecolor='black', elinewidth=1, capsize=3)
    ax.errorbar(x_good_control, vals_good_control, yerr=sem_good_control,
                fmt='none', ecolor='black', elinewidth=1, capsize=3)

    # Error bars: chemo
    ax.errorbar(x_no_chemo, vals_no_chemo, yerr=sem_no_chemo,
                fmt='none', ecolor='blue', elinewidth=1, capsize=3)
    ax.errorbar(x_poor_chemo, vals_poor_chemo, yerr=sem_poor_chemo,
                fmt='none', ecolor='blue', elinewidth=1, capsize=3)
    ax.errorbar(x_good_chemo, vals_good_chemo, yerr=sem_good_chemo,
                fmt='none', ecolor='blue', elinewidth=1, capsize=3)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean fraction across sessions")
    ax.set_xlabel("Block type")
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(
        f"Average Classified CR Fractions Across Sessions\n"
        f"Sessions: {first_date} -- {last_date} | Control n={n_control}, Chemo n={n_chemo}",
        fontsize=13
    )

    ax.legend(frameon=False, loc="best")

    plt.tight_layout()

    outname = f"AvgBarPlot_ClassifiedCRFractions_{first_tok}_to_{last_tok}.pdf"
    plt.savefig(outname, bbox_inches="tight")
    print(f"Saved figure to: {outname} - AvgFEC_AllSessions_bars.py:591")

    plt.show()


if __name__ == "__main__":
    main()