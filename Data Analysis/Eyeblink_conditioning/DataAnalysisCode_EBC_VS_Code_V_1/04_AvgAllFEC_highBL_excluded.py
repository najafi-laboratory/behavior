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
        fill_value=np.nan
    )
    return f(xq)


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


def pack_stats(t, fmat, led_starts, led_ends, puff_starts, puff_ends):
    out = {}
    out["t"] = np.asarray(t, dtype=float)

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
        out["led_start"] = np.nan
        out["led_end"] = np.nan
        out["puff_start"] = np.nan
        out["puff_end"] = np.nan
    else:
        out["mean"] = np.nanmean(fmat, axis=0)
        out["sem"] = np.nanstd(fmat, axis=0) / np.sqrt(out["n"])
        out["led_start"] = np.nanmedian(led_starts) if len(led_starts) else np.nan
        out["led_end"] = np.nanmedian(led_ends) if len(led_ends) else np.nan
        out["puff_start"] = np.nanmedian(puff_starts) if len(puff_starts) else np.nan
        out["puff_end"] = np.nanmedian(puff_ends) if len(puff_ends) else np.nan

    return out

def style_axes(ax):
    # Keep only left and bottom axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optional: make left/bottom slightly thicker
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Ticks only on left/bottom
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    # Remove grid if any
    ax.grid(False)
    
def plot_panel(ax, S, block_label):
    ax.plot(S["t"], S["mean"], linewidth=2.0, color="k")

    if np.any(np.isfinite(S["mean"])):
        ax.fill_between(
            S["t"],
            S["mean"] + S["sem"],
            S["mean"] - S["sem"],
            color="k",
            alpha=0.15
        )

    if np.isfinite(S["led_start"]) and np.isfinite(S["led_end"]):
        ax.axvspan(S["led_start"], S["led_end"], color="gray", alpha=0.18)

    if block_label.lower() == "long":
        puff_color = "limegreen"
        puff_text_color = "green"
    else:
        puff_color = "dodgerblue"
        puff_text_color = "dodgerblue"

    if np.isfinite(S["puff_start"]) and np.isfinite(S["puff_end"]):
        ax.axvspan(S["puff_start"], S["puff_end"], color=puff_color, alpha=0.35)

    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.03 * (ymax - ymin)

    if np.isfinite(S["led_start"]) and np.isfinite(S["led_end"]):
        ax.text(S["led_start"] + 0.002, y_text, "LED",
                color="black", va="bottom", ha="left", fontsize=10)

    if np.isfinite(S["puff_start"]) and np.isfinite(S["puff_end"]):
        ax.text(S["puff_start"] + 0.002, y_text, "AirPuff",
                color=puff_text_color, va="bottom", ha="left", fontsize=10)

    ax.set_title(f"{block_label} blocks (n={S['n']})", fontsize=14)
    ax.set_xlabel("Time from LED onset (s)")
    ax.set_ylabel("FEC")
    ax.grid(False)
    #ax.tick_params(direction="out")
    ax.tick_params(direction="out", length=4, width=1)
    style_axes(ax)


def plot_panel_two_groups(ax, S_control, S_chemo, block_label):
    if np.any(np.isfinite(S_control["mean"])):
        ax.plot(S_control["t"], S_control["mean"], linewidth=2.0, color="k", label="Control")
        ax.fill_between(
            S_control["t"],
            S_control["mean"] + S_control["sem"],
            S_control["mean"] - S_control["sem"],
            color="k",
            alpha=0.15
        )

    if np.any(np.isfinite(S_chemo["mean"])):
        ax.plot(S_chemo["t"], S_chemo["mean"], linewidth=2.0, color="blue", label="Chemo")
        ax.fill_between(
            S_chemo["t"],
            S_chemo["mean"] + S_chemo["sem"],
            S_chemo["mean"] - S_chemo["sem"],
            color="blue",
            alpha=0.15
        )

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

    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.03 * (ymax - ymin)

    if np.isfinite(S_ref["led_start"]) and np.isfinite(S_ref["led_end"]):
        ax.text(S_ref["led_start"] + 0.002, y_text, "LED",
                color="black", va="bottom", ha="left", fontsize=10)

    if np.isfinite(S_ref["puff_start"]) and np.isfinite(S_ref["puff_end"]):
        ax.text(S_ref["puff_start"] + 0.002, y_text, "AirPuff",
                color=puff_text_color, va="bottom", ha="left", fontsize=10)

    ax.set_title(
        f"{block_label} blocks\n"
        f"Control n={S_control['n']}, Chemo n={S_chemo['n']}",
        fontsize=14
    )
    ax.set_xlabel("Time from LED onset (s)")
    ax.set_ylabel("FEC")
    ax.grid(False)
    #ax.tick_params(direction="out")
    ax.tick_params(direction="out", length=4, width=1)
    ax.legend(frameon=False, loc="best")
    style_axes(ax)

# ============================================================
# Main
# ============================================================

def main():
    data_files = sorted(glob.glob("*_EBC_*.mat"))
    if len(data_files) == 0:
        print("No *_EBC_*.mat files found.  AvgAllFEC.py:347  AvgAllFEC_highBL_excluded.py:375 - 04_AvgAllFEC_highBL_excluded.py:375")
        return
    excluded_short_control_baseline = 0
    excluded_long_control_baseline = 0
    excluded_short_chemo_baseline = 0
    excluded_long_chemo_baseline = 0
    # settings
    t_pre = 0.2
    t_post = 0.6
    dt = 1 / 250
    smooth_win = 5
    exclude_probe = True
    exclude_timeout = True
    short_isi_max = 0.30
    interp_method = "linear"

    # NEW: baseline filtering settings
    baseline_window = (-0.2, 0.0)
    baseline_max_control = 0.4
    baseline_max_chemo = 0.4
    #filter_chemo_high_baseline_only = True

    # date range
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

    t_grid = np.arange(-t_pre, t_post + dt/2, dt)

    # =========================
    # CONTROL accumulators
    # =========================
    F_short_control = []
    led_short_start_control, led_short_end_control = [], []
    puff_short_start_control, puff_short_end_control = [], []

    F_long_control = []
    led_long_start_control, led_long_end_control = [], []
    puff_long_start_control, puff_long_end_control = [], []

    # =========================
    # CHEMO accumulators
    # =========================
    F_short_chemo = []
    led_short_start_chemo, led_short_end_chemo = [], []
    puff_short_start_chemo, puff_short_end_chemo = [], []

    F_long_chemo = []
    led_long_start_chemo, led_long_end_chemo = [], []
    puff_long_start_chemo, puff_long_end_chemo = [], []

    n_sessions_used = 0
    n_control_sessions_used = 0
    n_chemo_sessions_used = 0

    # NEW: counters for excluded chemo trials
    excluded_short_chemo_baseline = 0
    excluded_long_chemo_baseline = 0

    for filepath in data_files:
        print(f"Processing: {filepath}  AvgAllFEC.py:412  AvgAllFEC_highBL_excluded.py:451 - 04_AvgAllFEC_highBL_excluded.py:451")
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
            print(f"Skipping {os.path.basename(filepath)}: could not compute overall_max  AvgAllFEC.py:426  AvgAllFEC_highBL_excluded.py:465 - 04_AvgAllFEC_highBL_excluded.py:465")
            continue

        n_sessions_used += 1
        if chemo_flag == 1:
            n_chemo_sessions_used += 1
        else:
            n_control_sessions_used += 1

        for tr in trials:
            states = get_field(tr, "States", None)
            events = get_field(tr, "Events", None)
            data = get_field(tr, "Data", None)

            if states is None or events is None or data is None:
                continue

            # timeout exclusion
            if exclude_timeout and has_field(states, "CheckEyeOpenTimeout"):
                v = safe_array(get_field(states, "CheckEyeOpenTimeout", np.nan))
                if np.any(np.isfinite(v)):
                    continue

            # probe exclusion
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

            led_start_rel = 0.0
            led_end_rel = t_led_end_abs - t_led_abs if np.isfinite(t_led_end_abs) else np.nan

            puff_start_rel = t_puff_abs - t_led_abs
            puff_end_rel = t_puff_end_abs - t_led_abs if np.isfinite(t_puff_end_abs) else np.nan

            if smooth_win > 1:
                FEC = smooth_trace(FEC, smooth_win)

            Fq = interp_to_grid(t_rel, FEC, t_grid, method=interp_method)

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

            if chemo_flag == 1:
                if is_short:
                    F_short_chemo.append(Fq)
                    led_short_start_chemo.append(led_start_rel)
                    led_short_end_chemo.append(led_end_rel)
                    puff_short_start_chemo.append(puff_start_rel)
                    puff_short_end_chemo.append(puff_end_rel)
                else:
                    F_long_chemo.append(Fq)
                    led_long_start_chemo.append(led_start_rel)
                    led_long_end_chemo.append(led_end_rel)
                    puff_long_start_chemo.append(puff_start_rel)
                    puff_long_end_chemo.append(puff_end_rel)
            else:
                if is_short:
                    F_short_control.append(Fq)
                    led_short_start_control.append(led_start_rel)
                    led_short_end_control.append(led_end_rel)
                    puff_short_start_control.append(puff_start_rel)
                    puff_short_end_control.append(puff_end_rel)
                else:
                    F_long_control.append(Fq)
                    led_long_start_control.append(led_start_rel)
                    led_long_end_control.append(led_end_rel)
                    puff_long_start_control.append(puff_start_rel)
                    puff_long_end_control.append(puff_end_rel)

    S_short_control = pack_stats(
        t_grid, F_short_control,
        led_short_start_control, led_short_end_control,
        puff_short_start_control, puff_short_end_control
    )
    S_long_control = pack_stats(
        t_grid, F_long_control,
        led_long_start_control, led_long_end_control,
        puff_long_start_control, puff_long_end_control
    )

    S_short_chemo = pack_stats(
        t_grid, F_short_chemo,
        led_short_start_chemo, led_short_end_chemo,
        puff_short_start_chemo, puff_short_end_chemo
    )
    S_long_chemo = pack_stats(
        t_grid, F_long_chemo,
        led_long_start_chemo, led_long_end_chemo,
        puff_long_start_chemo, puff_long_end_chemo
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_panel_two_groups(axs[0], S_short_control, S_short_chemo, "Short")
    plot_panel_two_groups(axs[1], S_long_control, S_long_chemo, "Long")

    fig.suptitle(
        f"Pooled Avg FEC Without Classification\n"
        f"Sessions: {first_date} -- {last_date} | "
        f"Control sessions={n_control_sessions_used}, "
        f"Chemo sessions={n_chemo_sessions_used}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    outname = f"PooledAvgFEC_AllTrials_ShortLong_{first_tok}_to_{last_tok}.pdf"
    plt.savefig(outname, bbox_inches="tight")
    print(f"Saved figure to: {outname}  AvgAllFEC.py:558  AvgAllFEC_highBL_excluded.py:607 - 04_AvgAllFEC_highBL_excluded.py:607")

    print("\nHighbaseline trial exclusion summary:  AvgAllFEC_highBL_excluded.py:609 - 04_AvgAllFEC_highBL_excluded.py:609")
    print(f"Baseline window: {baseline_window}  AvgAllFEC_highBL_excluded.py:610 - 04_AvgAllFEC_highBL_excluded.py:610")
    print(f"Control baseline threshold: > {baseline_max_control}  AvgAllFEC_highBL_excluded.py:611 - 04_AvgAllFEC_highBL_excluded.py:611")
    print(f"Chemo baseline threshold:   > {baseline_max_chemo}  AvgAllFEC_highBL_excluded.py:612 - 04_AvgAllFEC_highBL_excluded.py:612")

    print(f"Excluded control shortblock trials: {excluded_short_control_baseline}  AvgAllFEC_highBL_excluded.py:614 - 04_AvgAllFEC_highBL_excluded.py:614")
    print(f"Excluded control longblock trials:  {excluded_long_control_baseline}  AvgAllFEC_highBL_excluded.py:615 - 04_AvgAllFEC_highBL_excluded.py:615")
    print(f"Excluded chemo shortblock trials:   {excluded_short_chemo_baseline}  AvgAllFEC_highBL_excluded.py:616 - 04_AvgAllFEC_highBL_excluded.py:616")
    print(f"Excluded chemo longblock trials:    {excluded_long_chemo_baseline}  AvgAllFEC_highBL_excluded.py:617 - 04_AvgAllFEC_highBL_excluded.py:617")

    print(f"Remaining control shortblock trials: {S_short_control['n']}  AvgAllFEC_highBL_excluded.py:619 - 04_AvgAllFEC_highBL_excluded.py:619")
    print(f"Remaining control longblock trials:  {S_long_control['n']}  AvgAllFEC_highBL_excluded.py:620 - 04_AvgAllFEC_highBL_excluded.py:620")
    print(f"Remaining chemo shortblock trials:   {S_short_chemo['n']}  AvgAllFEC_highBL_excluded.py:621 - 04_AvgAllFEC_highBL_excluded.py:621")
    print(f"Remaining chemo longblock trials:    {S_long_chemo['n']}  AvgAllFEC_highBL_excluded.py:622 - 04_AvgAllFEC_highBL_excluded.py:622")
    plt.show()


if __name__ == "__main__":
    main()