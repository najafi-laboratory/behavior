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


def smooth_trace(x, window=5):
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    return uniform_filter1d(x, size=window, mode="nearest")


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


def parse_session_date(filepath):
    base = os.path.basename(filepath)
    m = re.search(r"(\d{8})", base)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d")
        except Exception:
            return None
    return None


# ============================================================
# Classifier
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
    short_post_ms=8,
    long_pre_ms=50,
    long_post_ms=8
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
    good_cr_threshold : float
        Threshold for Good CR classification.
    poor_cr_threshold : float
        Threshold for Poor CR classification.
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
        small_mean_thr = 0.01

    elif block_label == "long":
        good_start = t_puff - long_pre
        good_end   = t_puff + long_post
        good_thr   = good_cr_threshold
        peak_thr   = good_cr_threshold
        small_mean_thr = 0.01

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
    good_mean_above_baseline = good_mean_above > small_mean_thr

    if np.any(poor_idx):
        poor_signal = signal[poor_idx]
        poor_amp = np.nanmean(poor_signal)
        poor_mean_above = poor_amp - baseline_amp
    else:
        poor_mean_above = -np.inf

    if (good_mean_above >= good_thr) or (good_peak_above >= peak_thr) or good_mean_above_baseline:
        category = "Good CR"
    elif poor_mean_above >= poor_cr_threshold:
        category = "Poor CR"
    else:
        category = "No CR"

    return category, cr_amp, baseline_amp, good_start, good_end

# ============================================================
# Main
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_files = sorted(glob.glob(os.path.join(script_dir, "*_EBC_*.mat")))
    if len(data_files) == 0:
        print(f"No *_EBC_*.mat files found in {script_dir}.  BarPlotCR_Classification.py:202 - 001_BarPlotCR_Classification.py:203")
        return
    print(f"Found {len(data_files)} .mat files in {script_dir} - 001_BarPlotCR_Classification.py:205")

    # thresholds
    
    good_cr_threshold = 0.05
    poor_cr_threshold = 0.02

    # CR window settings aligned with AvgAllSessions_CRClassificationErrorBars.py
    short_cr_pre_ms  = 25
    short_cr_post_ms = 20

    long_cr_pre_ms   = 50
    long_cr_post_ms  = 0
    
    # timing
    fps = 250
    seconds_before = 0.5
    seconds_after = 2.0
    frames_before = int(fps * seconds_before)
    frames_after = int(fps * seconds_after)

    # block split
    short_isi_max = 0.30

    # colors: Control only
    control_short_colors = {
        "Good CR": (0.0, 0.0, 0.0),      # black
        "Poor CR": (0.5, 0.5, 0.5),      # gray
        "No CR":   (0.83, 0.83, 0.83)    # light gray
    }
    control_long_colors = {
        "Good CR": (0.0, 0.0, 0.0),      # black
        "Poor CR": (0.5, 0.5, 0.5),      # gray
        "No CR":   (0.83, 0.83, 0.83)    # light gray
    }

    chemo_colors = {
        "Good CR": (0.12, 0.47, 0.71),   # blue
        "Poor CR": (0.50, 0.70, 0.90),   # medium blue
        "No CR":   (0.80, 0.90, 1.00)    # light blue
    }

    session_labels = []
    session_dates = []
    selected_sessions = []

    good_short = []
    poor_short = []
    no_short = []

    good_long = []
    poor_long = []
    no_long = []

    n_control_sessions = 0

    for filepath in data_files:
        print(f"Processing: {filepath}  BarPlotCR_Classification.py:253 - 001_BarPlotCR_Classification.py:262")

        SD = matlab_load_sessiondata(filepath)

        raw_events = get_field(SD, "RawEvents", None)
        trials = ensure_trial_list(get_field(raw_events, "Trial", None))

        num_good_short = 0
        num_poor_short = 0
        num_no_short = 0

        num_good_long = 0
        num_poor_long = 0
        num_no_long = 0

        num_short_trials = 0
        num_long_trials = 0

        
        # overall max from eyeAreaPixels across trials
        all_pixels = []
        for tr in trials:
            data = get_field(tr, "Data", None)
            if data is None:
                continue
            eye_area_pixels = safe_array(get_field(data, "eyeAreaPixels", None))
            if eye_area_pixels.size > 0:
                all_pixels.append(eye_area_pixels)

        if len(all_pixels) == 0:
            print(f"Skipping {os.path.basename(filepath)}: no eyeAreaPixels found  BarPlotCR_Classification.py:289 - 001_BarPlotCR_Classification.py:292")
            continue

        overall_max = np.nanmax(np.concatenate(all_pixels))

        for ctr_trial, tr in enumerate(trials, start=1):
            states = get_field(tr, "States", None)
            events = get_field(tr, "Events", None)
            data = get_field(tr, "Data", None)

            if states is None or events is None or data is None:
                continue

            # skip timeout
            if has_field(states, "CheckEyeOpenTimeout"):
                timeout_val = safe_array(get_field(states, "CheckEyeOpenTimeout", np.nan))
                if np.any(np.isfinite(timeout_val)):
                    continue

            # skip if no ISI
            led_puff_isi = safe_array(get_field(states, "LED_Puff_ISI", None))
            if led_puff_isi.size < 2:
                continue

            # required fields
            fec_times = safe_array(get_field(data, "FECTimes", None))
            eye_area_pixels = safe_array(get_field(data, "eyeAreaPixels", None))

            gt1_start = get_field(events, "GlobalTimer1_Start", None)
            gt1_end = get_field(events, "GlobalTimer1_End", None)
            gt2_start = get_field(events, "GlobalTimer2_Start", None)

            if fec_times.size == 0 or eye_area_pixels.size == 0:
                continue
            if fec_times.size != eye_area_pixels.size:
                continue
            if gt1_start is None or gt2_start is None or gt1_end is None:
                continue

            led_onset = scalar_or_nan(gt1_start)
            led_end = scalar_or_nan(gt1_end)
            airpuff_start = scalar_or_nan(gt2_start)

            if not np.isfinite(led_onset) or not np.isfinite(airpuff_start):
                continue

            fec_led_aligned = fec_times - led_onset
            fec_norm = 1 - eye_area_pixels / overall_max

            # trim around LED onset
            closest_idx = int(np.argmin(np.abs(fec_led_aligned)))
            start_idx = max(0, closest_idx - frames_before)
            stop_idx = min(len(fec_led_aligned) - 1, closest_idx + frames_after)

            fec_times_trimmed = fec_led_aligned[start_idx:stop_idx + 1]
            fec_trimmed = fec_norm[start_idx:stop_idx + 1]

            if len(fec_times_trimmed) < 5 or len(fec_trimmed) < 5:
                continue

            # smooth
            fec_times_smooth = smooth_trace(fec_times_trimmed, 5)
            fec_smooth = smooth_trace(fec_trimmed, 5)

            t_led = t_led = (led_end - led_onset) if np.isfinite(led_end) else 0.0
            t_puff = airpuff_start - led_onset

            isi_dur = led_puff_isi[1] - led_puff_isi[0]
            is_long_block = isi_dur > short_isi_max
            block_label = "long" if is_long_block else "short"

            cr_category = classify_cr_by_block(
                fec_times_smooth,
                fec_smooth,
                t_led,
                t_puff,
                good_cr_threshold,
                poor_cr_threshold,
                block_label,
                short_pre_ms=short_cr_pre_ms,
                short_post_ms=short_cr_post_ms,
                long_pre_ms=long_cr_pre_ms,
                long_post_ms=long_cr_post_ms
            )[0]
            
            if is_long_block:
                num_long_trials += 1
                if cr_category == "Good CR":
                    num_good_long += 1
                elif cr_category == "Poor CR":
                    num_poor_long += 1
                else:
                    num_no_long += 1
            else:
                num_short_trials += 1
                if cr_category == "Good CR":
                    num_good_short += 1
                elif cr_category == "Poor CR":
                    num_poor_short += 1
                else:
                    num_no_short += 1

        total_trials = num_short_trials + num_long_trials
        total_cr = num_good_short + num_poor_short + num_good_long + num_poor_long
        total_cr_fraction = total_cr / max(total_trials, 1)

        if total_trials == 0:
            print(f"Skipping {os.path.basename(filepath)}: no valid trials  BarPlotCR_Classification.py:396 - 001_BarPlotCR_Classification.py:399")
            continue

        short_good_fraction = (
            num_good_short / max(num_short_trials, 1)
            if num_short_trials > 0 else np.nan
        )
        long_good_fraction = (
            num_good_long / max(num_long_trials, 1)
            if num_long_trials > 0 else np.nan
        )

        if (
            (num_short_trials > 0 and short_good_fraction < 0.20)
            or (num_long_trials > 0 and long_good_fraction < 0.20)
        ):
            print(
                f"Warning {os.path.basename(filepath)}: short good CR {short_good_fraction:.2f}, "
                f"long good CR {long_good_fraction:.2f}; one or both < 0.20"
            )

        selected_sessions.append(os.path.basename(filepath))

        # store fractions for this session
        good_short.append(num_good_short / num_short_trials if num_short_trials > 0 else 0.0)
        poor_short.append(num_poor_short / num_short_trials if num_short_trials > 0 else 0.0)
        no_short.append(num_no_short / num_short_trials if num_short_trials > 0 else 0.0)

        good_long.append(num_good_long / num_long_trials if num_long_trials > 0 else 0.0)
        poor_long.append(num_poor_long / num_long_trials if num_long_trials > 0 else 0.0)
        no_long.append(num_no_long / num_long_trials if num_long_trials > 0 else 0.0)

        d = parse_session_date(filepath)
        if d is not None:
            session_dates.append(d)
            session_labels.append(d.strftime("%m/%d/%Y"))
        else:
            session_labels.append(os.path.basename(filepath))
            session_dates.append(None)

    if len(session_labels) == 0:
        print("No sessions found.  BarPlotCR_Classification.py:421 - 001_BarPlotCR_Classification.py:440")
        return

    # sort all session-level arrays by actual session date
    sort_idx = sorted(
        range(len(session_labels)),
        key=lambda i: session_dates[i] if session_dates[i] is not None else datetime.max
    )

    session_labels = [session_labels[i] for i in sort_idx]
    session_dates  = [session_dates[i] for i in sort_idx]
    selected_sessions = [selected_sessions[i] for i in sort_idx]

    good_short = [good_short[i] for i in sort_idx]
    poor_short = [poor_short[i] for i in sort_idx]
    no_short   = [no_short[i] for i in sort_idx]

    good_long = [good_long[i] for i in sort_idx]
    poor_long = [poor_long[i] for i in sort_idx]
    no_long   = [no_long[i] for i in sort_idx]

    selected_list_file = "selected_barplot_sessions.txt"
    if selected_sessions:
        with open(selected_list_file, "w") as fh:
            fh.write("\n".join(selected_sessions) + "\n")
        print(f"Saved selected barplot sessions to: {selected_list_file} - 001_BarPlotCR_Classification.py:465")

    x = np.arange(len(session_labels))

    x_offset = 0.11
    bar_width = 0.20

    x_short = x - x_offset
    x_long = x + x_offset

    chemo_dates = {"05/21", "05/27", "05/29", "06/02", "06/04"}  # MM/DD
    is_chemo = [lbl[:5] in chemo_dates for lbl in session_labels]

    # append "Chemo" to tick labels for chemo sessions
    session_labels = [
        (lbl + "\nChemo" if lbl[:5] in chemo_dates else lbl)
        for lbl in session_labels
    ]

    def bar_colors(ctrl_c, chemo_c):
        return [chemo_c if ch else ctrl_c for ch in is_chemo]

    fig, ax = plt.subplots(figsize=(14, 6))

    # -------- Short stacked bars --------
    b1 = ax.bar(x_short, good_short, width=bar_width,
                color=bar_colors(control_short_colors["Good CR"], chemo_colors["Good CR"]), edgecolor="none")
    b2 = ax.bar(x_short, poor_short, width=bar_width,
                bottom=np.array(good_short),
                color=bar_colors(control_short_colors["Poor CR"], chemo_colors["Poor CR"]), edgecolor="none")
    b3 = ax.bar(x_short, no_short, width=bar_width,
                bottom=np.array(good_short) + np.array(poor_short),
                color=bar_colors(control_short_colors["No CR"], chemo_colors["No CR"]), edgecolor="none")

    # -------- Long stacked bars --------
    b4 = ax.bar(x_long, good_long, width=bar_width,
                color=bar_colors(control_long_colors["Good CR"], chemo_colors["Good CR"]), edgecolor="none")
    b5 = ax.bar(x_long, poor_long, width=bar_width,
                bottom=np.array(good_long),
                color=bar_colors(control_long_colors["Poor CR"], chemo_colors["Poor CR"]), edgecolor="none")
    b6 = ax.bar(x_long, no_long, width=bar_width,
                bottom=np.array(good_long) + np.array(poor_long),
                color=bar_colors(control_long_colors["No CR"], chemo_colors["No CR"]), edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(session_labels, rotation=45, ha="right")
    for tick, label in zip(ax.get_xticklabels(), session_labels):
        if label[:5] in chemo_dates:
            tick.set_color("blue")

    ax.set_ylabel("CR Fraction")
    ax.set_xlabel("Sessions")
    ax.set_ylim([0, 1])
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # title with date range
    valid_dates = [d for d in session_dates if d is not None]
    session_count = len(session_labels)
    if valid_dates:
        first_date = min(valid_dates).strftime("%m/%d/%Y")
        last_date = max(valid_dates).strftime("%m/%d/%Y")
        title_str = (
            f"FEC Fraction of Classified CRs (All Sessions, n={session_count})\n"
            f"Sessions from {first_date} to {last_date}"
        )
    else:
        title_str = f"FEC Fraction of Classified CRs (All Sessions, n={session_count})"

    ax.set_title(title_str, fontsize=14)

    # simple legend like your MATLAB control legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=control_short_colors["Good CR"]),
        plt.Rectangle((0, 0), 1, 1, color=control_short_colors["Poor CR"]),
        plt.Rectangle((0, 0), 1, 1, color=control_short_colors["No CR"]),
        plt.Rectangle((0, 0), 1, 1, color=chemo_colors["Good CR"]),
        plt.Rectangle((0, 0), 1, 1, color=chemo_colors["Poor CR"]),
        plt.Rectangle((0, 0), 1, 1, color=chemo_colors["No CR"]),
    ]
    ax.legend(
        legend_handles,
        ["Good CR Control", "Poor CR Control", "No CR Control",
         "Good CR Chemo", "Poor CR Chemo", "No CR Chemo"],
        loc="best",
        frameon=False
    )

    plt.tight_layout()

    output_name = "BarPlt_FECFraction_ClassifiedCRs_AllSessions.pdf"
    plt.savefig(output_name, bbox_inches="tight")
    print(f"Saved figure to: {output_name}  BarPlotCR_Classification.py:512 - 001_BarPlotCR_Classification.py:558")

    plt.show()


if __name__ == "__main__":
    main()
