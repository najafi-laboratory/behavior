import os
import glob
import re
import math
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

def classify_cr_by_block(
    time,
    signal,
    t_led,
    t_puff,
    good_cr_threshold,
    poor_cr_threshold,
    block_label,
    poor_drop_threshold=0.05,
):
    """
    Good/Poor CR timing classifier.

    CR+ if max pre-US FEC minus baseline > poor_cr_threshold.
    Poor CR only if that pre-US maximum drops by more than
    poor_drop_threshold by the sample right before AirPuff.
    Otherwise the CR+ trial is Good CR.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if len(time) != len(signal):
        return "No CR", np.nan, np.nan

    baseline_idx = (time >= (t_led - 0.2)) & (time <= t_led)
    if not np.any(baseline_idx):
        return "No CR", np.nan, np.nan

    block_label = str(block_label).strip().lower()

    if block_label not in {"short", "long"} or not np.isfinite(t_puff):
        return "No CR", np.nan, np.nan

    timing_idx = (time >= t_led) & (time < t_puff)
    if not np.any(timing_idx):
        return "No CR", np.nan, np.nan

    baseline_amp = np.nanmean(signal[baseline_idx])
    if not np.isfinite(baseline_amp):
        return "No CR", np.nan, np.nan

    cr_indices = np.flatnonzero(timing_idx & np.isfinite(signal))
    before_candidates = np.flatnonzero((time < t_puff) & np.isfinite(signal))
    if cr_indices.size == 0 or before_candidates.size == 0:
        return "No CR", np.nan, baseline_amp

    max_idx = cr_indices[np.nanargmax(signal[cr_indices])]
    before_us_idx = before_candidates[-1]
    cr_amp = signal[max_idx]
    peak_above_baseline = cr_amp - baseline_amp

    if not np.isfinite(peak_above_baseline) or peak_above_baseline <= poor_cr_threshold:
        category = "No CR"
    elif (
        np.isfinite(signal[before_us_idx])
        and (cr_amp - signal[before_us_idx]) > poor_drop_threshold
        and time[max_idx] < time[before_us_idx]
    ):
        category = "Poor CR"
    else:
        category = "Good CR"

    return category, cr_amp, baseline_amp


# ============================================================
# Stats helpers
# ============================================================

def frac_and_sem(count, total):
    if total <= 0:
        return np.nan, np.nan
    p = count / total
    sem = math.sqrt(p * (1 - p) / total)
    return p, sem


def parse_session_label(filepath):
    m = re.search(r"(\d{8})", os.path.basename(filepath))
    if m:
        try:
            d = datetime.strptime(m.group(1), "%Y%m%d")
            return d.strftime("%Y-%m-%d")
        except Exception:
            return m.group(1)
    return os.path.basename(filepath)


# ============================================================
# Main
# ============================================================

def main():
    data_files = sorted(glob.glob("*_EBC_*.mat"))
    if len(data_files) == 0:
        print("No *_EBC_*.mat files found. - CRClassificationErrorBars_Sessions.py:284")
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

    session_results = []

    for filepath in data_files:
        print(f"Processing: {filepath} - CRClassificationErrorBars_Sessions.py:304")
        SD = matlab_load_sessiondata(filepath)

        raw_events = get_field(SD, "RawEvents", None)
        trials = ensure_trial_list(get_field(raw_events, "Trial", None))

        overall_max = collect_overall_max(trials)
        if not np.isfinite(overall_max):
            print(f"Skipping {os.path.basename(filepath)}: could not compute overall_max - CRClassificationErrorBars_Sessions.py:312")
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

        if (num_short_trials + num_long_trials) == 0:
            continue

        gs, gs_sem = frac_and_sem(num_good_short, num_short_trials)
        ps, ps_sem = frac_and_sem(num_poor_short, num_short_trials)
        ns, ns_sem = frac_and_sem(num_no_short, num_short_trials)

        gl, gl_sem = frac_and_sem(num_good_long, num_long_trials)
        pl, pl_sem = frac_and_sem(num_poor_long, num_long_trials)
        nl, nl_sem = frac_and_sem(num_no_long, num_long_trials)

        session_results.append({
            "label": parse_session_label(filepath),
            "short": {"Good": gs, "Poor": ps, "No": ns,
                      "Good_sem": gs_sem, "Poor_sem": ps_sem, "No_sem": ns_sem,
                      "N": num_short_trials},
            "long":  {"Good": gl, "Poor": pl, "No": nl,
                      "Good_sem": gl_sem, "Poor_sem": pl_sem, "No_sem": nl_sem,
                      "N": num_long_trials}
        })

    if len(session_results) == 0:
        print("No sessions with usable trials found. - CRClassificationErrorBars_Sessions.py:422")
        return

    # ============================================================
    # Plot
    # ============================================================

    n_sessions = len(session_results)
    fig_h = max(2.2 * n_sessions, 6)

    fig, axs = plt.subplots(n_sessions, 1, figsize=(9, fig_h), squeeze=False)
    axs = axs.ravel()

    for ax, res in zip(axs, session_results):

        x_centers = np.array([0, 1])   # Short, Long
        offset = 0.18

        x_no   = x_centers - offset
        x_poor = x_centers
        x_good = x_centers + offset

        vals_no   = np.array([res["short"]["No"],   res["long"]["No"]])
        vals_poor = np.array([res["short"]["Poor"], res["long"]["Poor"]])
        vals_good = np.array([res["short"]["Good"], res["long"]["Good"]])

        sem_no   = np.array([res["short"]["No_sem"],   res["long"]["No_sem"]])
        sem_poor = np.array([res["short"]["Poor_sem"], res["long"]["Poor_sem"]])
        sem_good = np.array([res["short"]["Good_sem"], res["long"]["Good_sem"]])

        # No CR
        ax.errorbar(
            x_no, vals_no,
            yerr=sem_no,
            fmt='o',
            mfc='white',
            mec='black',
            markersize=5,
            ecolor='black',
            capsize=4,
            linewidth=1.4
        )

        # Poor CR
        ax.errorbar(
            x_poor, vals_poor,
            yerr=sem_poor,
            fmt='o',
            mfc='0.6',
            mec='black',
            markersize=5,
            ecolor='black',
            capsize=4,
            linewidth=1.4
        )

        # Good CR
        ax.errorbar(
            x_good, vals_good,
            yerr=sem_good,
            fmt='o',
            mfc='black',
            mec='black',
            markersize=5,
            ecolor='black',
            capsize=4,
            linewidth=1.4
        )

        ax.set_xticks(x_centers)
        ax.set_xticklabels(["Short", "Long"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Fraction")

        ax.set_title(
            f"Session {res['label']}  |  Short n={res['short']['N']}, Long n={res['long']['N']}",
            fontsize=12
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


    # legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color='black', label='No CR',
            markerfacecolor='white', markersize=9, linestyle='None'),
        Line2D([0], [0], marker='o', color='black', label='Poor CR',
            markerfacecolor='0.6', markersize=9, linestyle='None'),
        Line2D([0], [0], marker='o', color='black', label='Good CR',
            markerfacecolor='black', markersize=9, linestyle='None')
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=False,
        fontsize=12
    )

    plt.tight_layout(rect=[0,0,0.92,1])

    plt.savefig(
        "PerSession_ShortLong_ClassifiedCR_Circles.pdf",
        bbox_inches="tight"
    )

    plt.show() 

if __name__ == "__main__":
    main()
