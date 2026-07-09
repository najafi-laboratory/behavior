import os
import glob
import re
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from datetime import datetime


# ============================================================
# USER SETTINGS
# ============================================================

DATA_DIR = "/Users/zahra/Desktop/NajafiLab/EBC_Analysis/SM09_Summary/SM09_Summary_all "
FILE_PATTERN = "*_EBC_*.mat"
SAVE_DIR = os.path.join(DATA_DIR, "Blockwise_CR_Percent_Plots")
os.makedirs(SAVE_DIR, exist_ok=True)

# CR thresholds
GOOD_CR_THRESHOLD = 0.05
POOR_CR_THRESHOLD = 0.02

# baseline exclusion
BASELINE_MAX = 0.40
MIN_TRIALS_PER_BLOCK = 20

# CR window relative to airpuff onset
SHORT_CR_WINDOW = (-0.050, 0.000)   # 50 ms before puff to puff onset
LONG_CR_WINDOW  = (-0.100, 0.000)   # 100 ms before puff to puff onset

# plot colors
COLOR_GOOD = "#000000"
COLOR_POOR = "#696969"
COLOR_NO   = "#d9d9d9"

CHEMO_COLOR_GOOD = "#003366"
CHEMO_COLOR_POOR = "#5f7fa7"
CHEMO_COLOR_NO   = "#b7d0eb"

# ============================================================
# HELPERS
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
    try:
        arr = np.atleast_1d(np.asarray(x).squeeze())
        return arr.astype(float)
    except Exception:
        try:
            return np.atleast_1d(np.array(x, dtype=float).squeeze())
        except Exception:
            return np.array([], dtype=float)


def ensure_trial_list(raw_trials):
    if raw_trials is None:
        return []
    if isinstance(raw_trials, list):
        return raw_trials
    if isinstance(raw_trials, np.ndarray):
        if raw_trials.dtype == object:
            return [x for x in raw_trials.flat]
        return [x for x in raw_trials]
    return [raw_trials]


def scalar_or_nan(x):
    arr = np.asarray(x).squeeze()
    if arr.size == 0:
        return np.nan
    try:
        return float(arr.flat[0])
    except Exception:
        return np.nan


def extract_date_from_filename(fname):
    base = os.path.basename(fname)
    m = re.search(r'(\d{8})', base)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d")
        except Exception:
            pass
    return None


def session_label_from_filename(fname):
    base = os.path.basename(fname).replace(".mat", "")
    date_obj = extract_date_from_filename(base)
    if date_obj is not None:
        return date_obj.strftime("%Y%m%d")
    return base


def load_session_mat(filepath):
    return loadmat(filepath, squeeze_me=True, struct_as_record=False)


def smooth_trace(x, window=5):
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    return uniform_filter1d(x, size=window, mode="nearest")


# ============================================================
# TIMING / BLOCK LABEL HELPERS
# ============================================================

def get_trial_timing(tr, short_isi_max=0.30):
    """
    Returns
    -------
    t_led_abs, t_puff_abs, isi_dur, is_short, block_label
    """
    states = get_field(tr, "States", None)
    events = get_field(tr, "Events", None)

    t_led_abs = np.nan
    t_puff_abs = np.nan
    isi_dur = np.nan
    block_label = None

    # LED onset
    if has_field(events, "GlobalTimer1_Start"):
        led = safe_array(get_field(events, "GlobalTimer1_Start"))
        if led.size > 0:
            t_led_abs = led[0]

    # Puff onset
    if has_field(events, "GlobalTimer2_Start"):
        puff = safe_array(get_field(events, "GlobalTimer2_Start"))
        if puff.size > 0:
            t_puff_abs = puff[0]

    # fallback from states
    if np.isnan(t_led_abs) and has_field(states, "LED"):
        arr = safe_array(get_field(states, "LED"))
        if arr.size >= 1:
            t_led_abs = arr[0]

    if np.isnan(t_puff_abs) and has_field(states, "AirPuff"):
        arr = safe_array(get_field(states, "AirPuff"))
        if arr.size >= 1:
            t_puff_abs = arr[0]

    if np.isnan(t_led_abs) and has_field(states, "LED_Puff_ISI"):
        arr = safe_array(get_field(states, "LED_Puff_ISI"))
        if arr.size >= 1:
            t_led_abs = arr[0]

    if np.isnan(t_puff_abs) and has_field(states, "LED_Puff_ISI"):
        arr = safe_array(get_field(states, "LED_Puff_ISI"))
        if arr.size >= 2:
            t_puff_abs = arr[1]

    if not np.isnan(t_led_abs) and not np.isnan(t_puff_abs):
        isi_dur = t_puff_abs - t_led_abs

    is_short = False
    if not np.isnan(isi_dur):
        is_short = (isi_dur <= short_isi_max)

    block_label = "short" if is_short else "long"
    return t_led_abs, t_puff_abs, isi_dur, is_short, block_label


def get_block_numbers(trials):
    """
    Assign block numbers separately for short and long trial streams.
    Example:
        short trials blocks: 1,2,3,...
        long  trials blocks: 1,2,3,...
    """
    short_block_num = 0
    long_block_num = 0
    prev_label = None

    out = []
    for tr in trials:
        _, _, _, _, label = get_trial_timing(tr)

        if label == "short":
            if prev_label != "short":
                short_block_num += 1
            out.append(("short", short_block_num))
        else:
            if prev_label != "long":
                long_block_num += 1
            out.append(("long", long_block_num))

        prev_label = label

    return out


# ============================================================
# FEC / CR HELPERS
# ============================================================

def get_trial_fec(tr):
    """
    Tries several common field names.
    Returns fec trace and time vector.
    """
    data = get_field(tr, "Data", None)

    fec = None
    t = None

    # direct fec
    for name in ["FEC", "fec", "FEC_norm", "FECTrace"]:
        if has_field(data, name):
            fec = safe_array(get_field(data, name))
            break
        if has_field(tr, name):
            fec = safe_array(get_field(tr, name))
            break

    # time
    for name in ["FECTimes", "fec_time", "time", "t"]:
        if has_field(data, name):
            t = safe_array(get_field(data, name))
            break
        if has_field(tr, name):
            t = safe_array(get_field(tr, name))
            break

    # fallback from eye area if FEC not present
    if fec is None or fec.size == 0:
        eye = None
        total = None

        if has_field(data, "eyeAreaPixels"):
            eye = safe_array(get_field(data, "eyeAreaPixels"))
        elif has_field(tr, "eyeAreaPixels"):
            eye = safe_array(get_field(tr, "eyeAreaPixels"))

        if has_field(data, "totalEllipsePixels"):
            total = safe_array(get_field(data, "totalEllipsePixels"))
        elif has_field(tr, "totalEllipsePixels"):
            total = safe_array(get_field(tr, "totalEllipsePixels"))

        if eye is not None and eye.size > 0:
            if total is not None and total.size > 0:
                overall_max = np.nanmax(total)
            else:
                overall_max = np.nanmax(eye)

            if overall_max > 0:
                fec = 1.0 - (eye / overall_max)

    # fallback time from frame rate
    if (t is None or t.size == 0) and fec is not None and fec.size > 0:
        fps = 250.0
        t = np.arange(len(fec)) / fps

    if fec is None:
        fec = np.array([])
    if t is None:
        t = np.array([])

    return safe_array(fec), safe_array(t)


def get_trial_eye_and_time(tr):
    data = get_field(tr, "Data", None)
    if data is None:
        return np.array([]), np.array([])
    eye = safe_array(get_field(data, "eyeAreaPixels", None))
    t = safe_array(get_field(data, "FECTimes", None))
    return eye, t


def session_overall_max_eye(trials):
    all_pixels = []
    for tr in trials:
        eye, _ = get_trial_eye_and_time(tr)
        if eye.size > 0:
            all_pixels.append(eye)

    if len(all_pixels) == 0:
        return np.nan

    return float(np.nanmax(np.concatenate(all_pixels)))


def trial_is_probe(tr):
    data = get_field(tr, "Data", None)
    for name in ["IsProbeTrial", "ProbeTrial"]:
        if has_field(data, name):
            val = scalar_or_nan(get_field(data, name))
            return bool(val == 1)
        if has_field(tr, name):
            val = scalar_or_nan(get_field(tr, name))
            return bool(val == 1)
    return False


def trial_is_timeout(tr):
    states = get_field(tr, "States", None)
    if has_field(states, "CheckEyeOpenTimeout"):
        arr = safe_array(get_field(states, "CheckEyeOpenTimeout"))
        return arr.size > 0 and np.any(np.isfinite(arr))
    return False


def classify_cr_by_block(
    time,
    signal,
    t_led,
    t_puff,
    block_label,
    short_pre_ms=25,
    short_post_ms=8,
    long_pre_ms=50,
    long_post_ms=8
):
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if len(time) != len(signal):
        return None, np.nan

    baseline_idx = (time >= (t_led - 0.2)) & (time <= t_led)
    if not np.any(baseline_idx):
        return None, np.nan

    baseline_amp = np.nanmean(signal[baseline_idx])
    if not np.isfinite(baseline_amp) or baseline_amp > BASELINE_MAX:
        return None, baseline_amp

    block_label = str(block_label).strip().lower()
    short_pre = short_pre_ms / 1000.0
    short_post = short_post_ms / 1000.0
    long_pre = long_pre_ms / 1000.0
    long_post = long_post_ms / 1000.0

    if block_label == "short":
        good_start = t_puff - short_pre
        good_end = t_puff + short_post
        good_thr = 0.03
        peak_thr = 0.035
    elif block_label == "long":
        good_start = t_puff - long_pre
        good_end = t_puff + long_post
        good_thr = GOOD_CR_THRESHOLD
        peak_thr = GOOD_CR_THRESHOLD
    else:
        return None, baseline_amp

    poor_start = t_led
    poor_end = good_start
    good_idx = (time >= good_start) & (time <= good_end)
    poor_idx = (time >= poor_start) & (time < poor_end)

    if not np.any(good_idx):
        return None, baseline_amp

    good_signal = signal[good_idx]
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
        return "Good CR", baseline_amp
    if poor_mean_above >= POOR_CR_THRESHOLD:
        return "Poor CR", baseline_amp
    return "No CR", baseline_amp


def classify_cr(tr, overall_max):
    states = get_field(tr, "States", None)
    events = get_field(tr, "Events", None)
    data = get_field(tr, "Data", None)

    if states is None or events is None or data is None:
        return None, np.nan
    if trial_is_probe(tr) or trial_is_timeout(tr):
        return None, np.nan
    if not np.isfinite(overall_max) or overall_max <= 0:
        return None, np.nan

    fec_times = safe_array(get_field(data, "FECTimes", None))
    eye_area_pixels = safe_array(get_field(data, "eyeAreaPixels", None))

    if fec_times.size == 0 or eye_area_pixels.size == 0:
        return None, np.nan
    if fec_times.size != eye_area_pixels.size:
        return None, np.nan

    led_onset = scalar_or_nan(get_field(events, "GlobalTimer1_Start", None))
    airpuff_start = scalar_or_nan(get_field(events, "GlobalTimer2_Start", None))
    if not np.isfinite(led_onset) or not np.isfinite(airpuff_start):
        return None, np.nan

    _, _, _, _, block_label = get_trial_timing(tr)
    if block_label not in ("short", "long"):
        return None, np.nan

    fec_led_aligned = fec_times - led_onset
    fec_norm = 1.0 - (eye_area_pixels / overall_max)
    fec_smooth = smooth_trace(fec_norm, window=5)

    t_led = 0.0
    t_puff = airpuff_start - led_onset

    return classify_cr_by_block(
        fec_led_aligned,
        fec_smooth,
        t_led,
        t_puff,
        block_label
    )


# ============================================================
# SESSION ANALYSIS
# ============================================================

def analyze_session(filepath):
    mat = load_session_mat(filepath)
    SD = None

    for key in ["SessionData", "session_data", "SD"]:
        if key in mat:
            SD = mat[key]
            break

    if SD is None:
        raise ValueError(f"Could not find SessionData in {os.path.basename(filepath)}")

    raw_trials = get_field(get_field(SD, "RawEvents", None), "Trial", None)
    trials = ensure_trial_list(raw_trials)

    if len(trials) == 0:
        raise ValueError(f"No trials found in {os.path.basename(filepath)}")

    overall_max = session_overall_max_eye(trials)
    if not np.isfinite(overall_max) or overall_max <= 0:
        raise ValueError(f"Could not compute session-wide eyeArea max in {os.path.basename(filepath)}")

    block_info = get_block_numbers(trials)

    results = {
        "short": {},
        "long": {}
    }

    for tr, (label, block_num) in zip(trials, block_info):
        cat, baseline_val = classify_cr(tr, overall_max)
        if cat is None:
            continue

        if block_num not in results[label]:
            results[label][block_num] = {
                "Good CR": 0,
                "Poor CR": 0,
                "No CR": 0,
                "n": 0
            }

        results[label][block_num][cat] += 1
        results[label][block_num]["n"] += 1

    chemo_flag = get_field(SD, "Chemogenetics", 0)
    try:
        chemo_flag = int(np.asarray(chemo_flag).squeeze())
    except Exception:
        chemo_flag = 0

    return results, bool(chemo_flag)


# ============================================================
# PLOTTING
# ============================================================

def block_percent_dict(block_dict):
    out = {}
    for blk, vals in block_dict.items():
        n = vals["n"]
        if n < MIN_TRIALS_PER_BLOCK:
            continue
        out[blk] = {
            "Good CR": 100 * vals["Good CR"] / n,
            "Poor CR": 100 * vals["Poor CR"] / n,
            "No CR":   100 * vals["No CR"] / n,
            "n": n
        }
    return out


def plot_all_sessions(session_results):
    valid_sessions = []
    max_block = 0

    for item in session_results:
        short_res = block_percent_dict(item["results"]["short"])
        long_res = block_percent_dict(item["results"]["long"])
        item["short_percent"] = short_res
        item["long_percent"] = long_res
        item["total_valid_trials"] = (
            sum(vals["n"] for vals in item["results"]["short"].values()) +
            sum(vals["n"] for vals in item["results"]["long"].values())
        )

        max_block = max(
            max_block,
            max(short_res.keys()) if len(short_res) > 0 else 0,
            max(long_res.keys()) if len(long_res) > 0 else 0
        )

        if len(short_res) == 0 and len(long_res) == 0:
            print(f"Skipping {os.path.basename(item['filepath'])} (no valid classified trials).  CrFraction_BlockSessions.py:538 - 06_CrFraction_BlockSessions.py:548")
            continue

        valid_sessions.append(item)

    block_nums = sorted({
        block_num
        for item in valid_sessions
        for block_num in list(item["short_percent"].keys()) + list(item["long_percent"].keys())
    })

    if len(valid_sessions) == 0 or len(block_nums) == 0:
        print("No sessions with valid classified trials to plot.  CrFraction_BlockSessions.py:544 - 06_CrFraction_BlockSessions.py:560")
        return

    n_sessions = len(valid_sessions)
    fig_w = max(14, 1.25 * n_sessions + 4)
    fig_h = max(3.1 * len(block_nums) + 1.5, 6)
    fig, axes = plt.subplots(len(block_nums), 1, figsize=(fig_w, fig_h), sharex=True, sharey=True)

    if len(block_nums) == 1:
        axes = [axes]

    x_base = np.arange(n_sessions)
    short_offset = -0.16
    long_offset = 0.16
    bar_w = 0.26

    def get_block_vals(res, block_num, key):
        if block_num in res:
            return res[block_num][key]
        return 0.0

    def get_block_n(res, block_num):
        if block_num in res:
            return int(res[block_num]["n"])
        return 0

    for block_num, ax in zip(block_nums, axes):
        short_good = np.array([get_block_vals(item["short_percent"], block_num, "Good CR") for item in valid_sessions])
        short_poor = np.array([get_block_vals(item["short_percent"], block_num, "Poor CR") for item in valid_sessions])
        short_no = np.array([get_block_vals(item["short_percent"], block_num, "No CR") for item in valid_sessions])

        long_good = np.array([get_block_vals(item["long_percent"], block_num, "Good CR") for item in valid_sessions])
        long_poor = np.array([get_block_vals(item["long_percent"], block_num, "Poor CR") for item in valid_sessions])
        long_no = np.array([get_block_vals(item["long_percent"], block_num, "No CR") for item in valid_sessions])
        short_n = np.array([get_block_n(item["short_percent"], block_num) for item in valid_sessions])
        long_n = np.array([get_block_n(item["long_percent"], block_num) for item in valid_sessions])

        short_x = x_base + short_offset
        long_x = x_base + long_offset

        short_good_colors = [CHEMO_COLOR_GOOD if item['chemo'] else COLOR_GOOD for item in valid_sessions]
        short_poor_colors = [CHEMO_COLOR_POOR if item['chemo'] else COLOR_POOR for item in valid_sessions]
        short_no_colors = [CHEMO_COLOR_NO if item['chemo'] else COLOR_NO for item in valid_sessions]
        long_good_colors = [CHEMO_COLOR_GOOD if item['chemo'] else COLOR_GOOD for item in valid_sessions]
        long_poor_colors = [CHEMO_COLOR_POOR if item['chemo'] else COLOR_POOR for item in valid_sessions]
        long_no_colors = [CHEMO_COLOR_NO if item['chemo'] else COLOR_NO for item in valid_sessions]

        ax.bar(short_x, short_good, width=bar_w, color=short_good_colors, edgecolor="none")
        ax.bar(short_x, short_poor, width=bar_w, bottom=short_good, color=short_poor_colors, edgecolor="none")
        ax.bar(short_x, short_no, width=bar_w, bottom=short_good + short_poor, color=short_no_colors, edgecolor="none")

        ax.bar(long_x, long_good, width=bar_w, color=long_good_colors, edgecolor="none")
        ax.bar(long_x, long_poor, width=bar_w, bottom=long_good, color=long_poor_colors, edgecolor="none")
        ax.bar(long_x, long_no, width=bar_w, bottom=long_good + long_poor, color=long_no_colors, edgecolor="none")

        for x, n in zip(short_x, short_n):
            if n > 0:
                ax.text(x, 101.5, f"n={n}", ha="center", va="bottom", fontsize=7, clip_on=False)

        for x, n in zip(long_x, long_n):
            if n > 0:
                ax.text(x, 101.5, f"n={n}", ha="center", va="bottom", fontsize=7, clip_on=False)

        ax.set_ylim(0, 106)
        ax.set_ylabel(f"Block {block_num}\nCR %", fontsize=11)

    axes[0].set_title(
        "Percents of Good/Poor/No CR in each block\nfor both short and long blocks across sessions",
        fontsize=13,
        pad=18
    )
    axes[-1].set_xlabel("Session", fontsize=12)
    axes[-1].set_xticks(x_base)
    axes[-1].set_xticklabels(
        [f"{session_label_from_filename(item['filepath'])}\nS   L" for item in valid_sessions],
        rotation=0,
        ha="center",
        fontsize=10
    )

    for i, item in enumerate(valid_sessions):
        axes[0].text(
            x_base[i],
            106.8,
            f"N={item['total_valid_trials']}",
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False
        )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLOR_GOOD, label="Good CR"),
        Patch(facecolor=COLOR_POOR, label="Poor CR"),
        Patch(facecolor=COLOR_NO, edgecolor="0.7", label="No CR"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.16, top=0.92)

    out_png = os.path.join(SAVE_DIR, "AllSessions_Blockwise_CRPercent.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}  CrFraction_BlockSessions.py:643 - 06_CrFraction_BlockSessions.py:666")


def save_summary_csv(session_results):
    out_csv = os.path.join(SAVE_DIR, "AllSessions_Blockwise_CRPercent.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "session_file", "session_label", "block_type", "block_num",
            "n_valid_trials", "good_count", "poor_count", "no_count",
            "good_percent", "poor_percent", "no_percent"
        ])

        for item in session_results:
            for block_type in ["short", "long"]:
                for block_num, vals in sorted(item["results"][block_type].items()):
                    n = vals["n"]
                    if n == 0:
                        continue
                    writer.writerow([
                        os.path.basename(item["filepath"]),
                        session_label_from_filename(item["filepath"]),
                        block_type,
                        block_num,
                        n,
                        vals["Good CR"],
                        vals["Poor CR"],
                        vals["No CR"],
                        100.0 * vals["Good CR"] / n,
                        100.0 * vals["Poor CR"] / n,
                        100.0 * vals["No CR"] / n,
                    ])

    print(f"Saved: {out_csv}  CrFraction_BlockSessions.py:676 - 06_CrFraction_BlockSessions.py:699")


# ============================================================
# MAIN
# ============================================================

def main():
    files = glob.glob(os.path.join(DATA_DIR, FILE_PATTERN))

    if len(files) == 0:
        print("No .mat files found.  CrFraction_BlockSessions.py:687 - 06_CrFraction_BlockSessions.py:710")
        return

    files = sorted(
        files,
        key=lambda f: (
            extract_date_from_filename(f) is None,
            extract_date_from_filename(f) or datetime.max,
            os.path.basename(f).lower()
        )
    )

    session_results = []

    for fp in files:
        print(f"Processing: {os.path.basename(fp)}  CrFraction_BlockSessions.py:702 - 06_CrFraction_BlockSessions.py:725")
        try:
            results, chemo_flag = analyze_session(fp)
            session_results.append({
                "filepath": fp,
                "results": results,
                "chemo": chemo_flag,
            })
        except Exception as e:
            print(f"Failed on {os.path.basename(fp)}: {e}  CrFraction_BlockSessions.py:710 - 06_CrFraction_BlockSessions.py:734")

    save_summary_csv(session_results)
    plot_all_sessions(session_results)


if __name__ == "__main__":
    main()
