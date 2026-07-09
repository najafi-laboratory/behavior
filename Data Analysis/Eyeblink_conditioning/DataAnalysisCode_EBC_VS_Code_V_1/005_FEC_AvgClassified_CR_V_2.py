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
    if x is None:
        return np.nan

    arr = np.asarray(x).squeeze()
    if arr is None:
        return np.nan
    if arr.size == 0:
        return np.nan
    if arr.ndim == 0:
        try:
            return float(arr)
        except (TypeError, ValueError):
            return np.nan
    try:
        return float(arr.flat[0])
    except (TypeError, ValueError):
        return np.nan


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


def estimate_session_open_eye_area(trials, short_isi_max=0.30, exclude_probe=True, exclude_timeout=True,
                                   open_eye_percentile=99):
    """
    Estimate one open-eye reference per session.

    Uses the 99th percentile of eyeAreaPixels across ALL frames of all valid
    trials in the session.  This matches the MATLAB protocol convention of
    normalising by the maximum observed eye area (totalEllipsePixels - minFur),
    and avoids the bias introduced by restricting sampling to the 200 ms
    pre-LED window, which can contain anticipatory partial closures in learned
    animals.  Eye-closure events (blinks, CRs) produce lower eye areas and
    therefore fall well below the 99th percentile and do not contaminate the
    reference.
    """
    all_eye_vals = []

    for tr in trials:
        states = get_field(tr, "States", None)
        data = get_field(tr, "Data", None)

        if states is None or data is None:
            continue

        if exclude_timeout and has_field(states, "CheckEyeOpenTimeout"):
            v = safe_array(get_field(states, "CheckEyeOpenTimeout", np.nan))
            if np.any(np.isfinite(v)):
                continue

        if exclude_probe and has_field(data, "IsProbeTrial"):
            is_probe = np.asarray(get_field(data, "IsProbeTrial", 0)).squeeze()
            if np.any(is_probe == 1):
                continue

        eyeAreaPixels = safe_array(get_field(data, "eyeAreaPixels", None))

        if eyeAreaPixels.size == 0:
            continue

        valid = eyeAreaPixels[np.isfinite(eyeAreaPixels)]
        if valid.size > 0:
            all_eye_vals.extend(valid.tolist())

    if not all_eye_vals:
        return np.nan

    all_eye_vals = np.asarray(all_eye_vals, dtype=float)
    return np.nanpercentile(all_eye_vals, open_eye_percentile)


def normalize_fec_to_open_eye_reference(eye_area_pixels, open_eye_area):
    """FEC = 1 - eyeArea / session open-eye reference."""
    if not np.isfinite(open_eye_area) or open_eye_area <= 0:
        return None
    return 1.0 - np.asarray(eye_area_pixels, dtype=float) / open_eye_area


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

    if len(eye_vals) > 0:
        return np.nanmax(np.asarray(eye_vals, dtype=float))

    if len(total_vals) > 0:
        return np.nanmax(np.asarray(total_vals, dtype=float))

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
    short_post_ms=10,
    long_pre_ms=50,
    long_post_ms=10
):
    """
    Block-specific CR classifier using a configurable CR window.

    Parameters
    ----------
    short_pre_ms : float
        How many ms before puff onset to start CR window in short block.
    short_post_ms : float
        How many ms after puff onset to end CR window in short block.
    long_pre_ms : float
        How many ms before puff onset to start CR window in long block.
    long_post_ms : float
        How many ms after puff onset to end CR window in long block.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if len(time) != len(signal):
        return "No CR", np.nan, np.nan, np.nan, np.nan

    baseline_idx = (time >= (t_led - 0.2)) & (time <= t_led)
    if not np.any(baseline_idx):
        return "No CR", np.nan, np.nan, np.nan, np.nan

    block_label = str(block_label).strip().lower()

    # convert ms to sec
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


def match_baselines(F_ctrl, F_chemo, t_grid, n_bins=10, rng_seed=42):
    """
    Subsample ctrl and chemo trial lists so their pre-LED baseline (mean FEC
    in [-0.2, 0) s) distributions match via equal-bin subsampling.
    Returns (idx_ctrl, idx_chemo) integer index arrays.
    """
    if len(F_ctrl) == 0 or len(F_chemo) == 0:
        return np.arange(len(F_ctrl)), np.arange(len(F_chemo))
    bl = (t_grid >= -0.2) & (t_grid < 0.0)
    b_c = np.nanmean(np.vstack(F_ctrl)[:,  bl], axis=1)
    b_h = np.nanmean(np.vstack(F_chemo)[:, bl], axis=1)
    lo = max(np.nanmin(b_c), np.nanmin(b_h))
    hi = min(np.nanmax(b_c), np.nanmax(b_h))
    if lo >= hi:
        return np.arange(len(F_ctrl)), np.arange(len(F_chemo))
    edges = np.linspace(lo, hi, n_bins + 1)
    rng = np.random.default_rng(rng_seed)
    sel_c, sel_h = [], []
    for i in range(n_bins):
        ic = np.where((b_c >= edges[i]) & (b_c < edges[i + 1]))[0]
        ih = np.where((b_h >= edges[i]) & (b_h < edges[i + 1]))[0]
        n = min(len(ic), len(ih))
        if n == 0:
            continue
        sel_c.extend(rng.choice(ic, n, replace=False).tolist())
        sel_h.extend(rng.choice(ih, n, replace=False).tolist())
    if not sel_c or not sel_h:
        return np.arange(len(F_ctrl)), np.arange(len(F_chemo))
    return np.array(sel_c, dtype=int), np.array(sel_h, dtype=int)


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

    ax.set_title(f"{cr_label} -- {block_label} (n={S['n']})", fontsize=13)
    ax.set_xlabel("Time from LED onset (s)")
    ax.set_ylabel("FEC")
    ax.grid(False)
    ax.tick_params(direction="out")


def plot_panel_two_groups(ax, S_ctrl, S_chemo, cr_label, block_label):
    # Control = black
    if np.any(np.isfinite(S_ctrl["mean"])):
        ax.plot(S_ctrl["t"], S_ctrl["mean"], linewidth=1.8, color="k", label="Control")
        ax.fill_between(
            S_ctrl["t"],
            S_ctrl["mean"] + S_ctrl["sem"],
            S_ctrl["mean"] - S_ctrl["sem"],
            color="k", alpha=0.15
        )

    # Chemo = steelblue
    if np.any(np.isfinite(S_chemo["mean"])):
        ax.plot(S_chemo["t"], S_chemo["mean"], linewidth=1.8, color="steelblue", label="Chemo")
        ax.fill_between(
            S_chemo["t"],
            S_chemo["mean"] + S_chemo["sem"],
            S_chemo["mean"] - S_chemo["sem"],
            color="steelblue", alpha=0.15
        )

    S_ref = S_ctrl if np.any(np.isfinite(S_ctrl["mean"])) else S_chemo

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
        ax.axvline(S_ref["cr_end"],   color="darkorange", linestyle=":", linewidth=1.5)

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
        f"{cr_label} -- {block_label}\nControl n={S_ctrl['n']}, Chemo n={S_chemo['n']}",
        fontsize=12
    )
    ax.set_xlabel("Time from LED onset (s)")
    ax.set_ylabel("FEC")
    ax.grid(False)
    ax.tick_params(direction="out")
    ax.legend(frameon=False, loc="best")


# ============================================================
# Main
# ============================================================

def main():
    # ------------ sessions ------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_files = sorted(glob.glob(os.path.join(script_dir, "*_EBC_*.mat")))
    if len(data_files) == 0:
        print(f"No *_EBC_*.mat files found in {script_dir}.  FEC_AvgClassified_CR.py:425  FEC_AvgClassified_CR_V_2.py:436 - 005_FEC_AvgClassified_CR_V_2.py:603")
        return

    # ------------ settings ------------
    t_pre = 0.2
    t_post = 0.6
    dt = 1 / 250
    smooth_win = 5

    exclude_probe = True
    exclude_timeout = True

    good_cr_threshold = 0.05
    poor_cr_threshold = 0.02

    # ------------ CR window settings ------------
    short_cr_pre_ms  = 25   # start 25 ms before puff
    short_cr_post_ms = 12    # end 12 ms after puff

    long_cr_pre_ms   = 50   # start 50 ms before puff
    long_cr_post_ms  = 12    # end 12 ms after puff

    interp_method = "linear"
    short_isi_max = 0.30

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

    F_acc = {}
    puff_s_acc = {}; puff_e_acc = {}
    cr_s_acc   = {}; cr_e_acc   = {}
    led_s_acc  = {}; led_e_acc  = {}
    for _bl in ("short", "long"):
        for _cat in ("good", "poor", "no"):
            for _grp in ("ctrl", "chemo"):
                _k = (_bl, _cat, _grp)
                F_acc[_k] = []; puff_s_acc[_k] = []; puff_e_acc[_k] = []
                cr_s_acc[_k] = []; cr_e_acc[_k] = []
                led_s_acc[_k] = []; led_e_acc[_k] = []

    n_sessions_used = 0
    n_ctrl_sessions = 0
    n_chemo_sessions = 0
    led_duration_all = []

    for filepath in data_files:
        print(f"Processing: {filepath}  FEC_AvgClassified_CR.py:512  FEC_AvgClassified_CR_V_2.py:523 - 005_FEC_AvgClassified_CR_V_2.py:670")
        S = loadmat(filepath, squeeze_me=True, struct_as_record=False)

        if "SessionData" not in S:
            continue

        SD = S["SessionData"]
        n_sessions_used += 1

        chemo_flag = int(get_field(SD, "Chemogenetics", 0))
        is_chemo = (chemo_flag == 1)

        # Date-based chemo override for sessions missing the Chemogenetics flag
        _chemo_dates = {"05/21", "05/27", "05/29", "06/02", "06/04"}
        _dm = re.search(r"(\d{8})", os.path.basename(filepath))
        if _dm:
            try:
                _sd = datetime.strptime(_dm.group(1), "%Y%m%d")
                if _sd.strftime("%m/%d") in _chemo_dates:
                    is_chemo = True
            except Exception:
                pass

        if is_chemo:
            n_chemo_sessions += 1
        else:
            n_ctrl_sessions += 1

        raw_events = get_field(SD, "RawEvents", None)
        trials = ensure_trial_list(get_field(raw_events, "Trial", None))
        open_eye_area = estimate_session_open_eye_area(
            trials,
            short_isi_max=short_isi_max,
            exclude_probe=exclude_probe,
            exclude_timeout=exclude_timeout
        )
        if not np.isfinite(open_eye_area):
            print(f"Skipping {os.path.basename(filepath)}: could not estimate session openeye area  FEC_AvgClassified_CR_V_2.py - 005_FEC_AvgClassified_CR_V_2.py:707")
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

            t_led_abs = scalar_or_nan(GT1_start)
            t_puff_abs = scalar_or_nan(GT2_start)
            if not np.isfinite(t_led_abs) or not np.isfinite(t_puff_abs):
                continue

            # LED-aligned time
            t_rel = FECTimes - t_led_abs
            FEC = normalize_fec_to_open_eye_reference(eyeAreaPixels, open_eye_area)
            if FEC is None:
                continue

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

            isi_dur = LED_Puff_ISI[1] - LED_Puff_ISI[0]
            is_short = isi_dur <= short_isi_max

            block_label = "short" if is_short else "long"

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

            bl  = "short" if is_short else "long"
            cat = "good" if CR_category == "Good CR" else ("poor" if CR_category == "Poor CR" else "no")
            grp = "chemo" if is_chemo else "ctrl"
            key = (bl, cat, grp)

            F_acc[key].append(Fq)
            puff_s_acc[key].append(puff_start_rel)
            puff_e_acc[key].append(puff_end_rel)
            cr_s_acc[key].append(cr_start_rel)
            cr_e_acc[key].append(cr_end_rel)
            led_s_acc[key].append(led_start_rel)
            led_e_acc[key].append(led_end_rel)

    # ---- baseline matching: equalize pre-LED baseline distributions per (block, CR category) ----
    for _bl in ("short", "long"):
        for _cat in ("good", "poor", "no"):
            _kc = (_bl, _cat, "ctrl")
            _kh = (_bl, _cat, "chemo")
            _Fc = F_acc[_kc]
            _Fh = F_acc[_kh]
            if not _Fc or not _Fh:
                continue
            _ic, _ih = match_baselines(_Fc, _Fh, t_grid)
            F_acc[_kc] = [_Fc[i] for i in _ic]
            F_acc[_kh] = [_Fh[i] for i in _ih]
            for _d in (puff_s_acc, puff_e_acc, cr_s_acc, cr_e_acc, led_s_acc, led_e_acc):
                _d[_kc] = [_d[_kc][i] for i in _ic]
                _d[_kh] = [_d[_kh][i] for i in _ih]
    print("Baselinematched (ctrl > chemo trials per category): - 005_FEC_AvgClassified_CR_V_2.py:830")
    for _bl in ("short", "long"):
        for _cat in ("good", "poor", "no"):
            _nc = len(F_acc[(_bl, _cat, "ctrl")])
            _nh = len(F_acc[(_bl, _cat, "chemo")])
            print(f"{_bl:5s} {_cat:4s}: ctrl={_nc}, chemo={_nh} - 005_FEC_AvgClassified_CR_V_2.py:835")

    # ------------ pack stats for all 12 combinations ------------
    def ps(bl, cat, grp):
        k = (bl, cat, grp)
        return pack_stats(
            t_grid, F_acc[k],
            puff_s_acc[k], puff_e_acc[k],
            cr_s_acc[k], cr_e_acc[k],
            led_s_acc[k], led_e_acc[k]
        )

    # ------------ plot 3x2 ------------
    fig, axs = plt.subplots(3, 2, figsize=(11.5, 8.5))
    axs = axs.ravel()

    plot_panel_two_groups(axs[0], ps("short", "good", "ctrl"), ps("short", "good", "chemo"), "Good CR", "Short")
    plot_panel_two_groups(axs[1], ps("long",  "good", "ctrl"), ps("long",  "good", "chemo"), "Good CR", "Long")

    plot_panel_two_groups(axs[2], ps("short", "poor", "ctrl"), ps("short", "poor", "chemo"), "Poor CR", "Short")
    plot_panel_two_groups(axs[3], ps("long",  "poor", "ctrl"), ps("long",  "poor", "chemo"), "Poor CR", "Long")

    plot_panel_two_groups(axs[4], ps("short", "no", "ctrl"), ps("short", "no", "chemo"), "No CR", "Short")
    plot_panel_two_groups(axs[5], ps("long",  "no", "ctrl"), ps("long",  "no", "chemo"), "No CR", "Long")

    fig.suptitle(
        f"Pooled Avg FEC CR-classified Short/Long\nSessions: {first_date} -- {last_date} | Control={n_ctrl_sessions}, Chemo={n_chemo_sessions}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    outname = f"PooledAvgFEC_CRclassified_ShortLong_{first_tok}_to_{last_tok}.pdf"
    plt.savefig(outname, bbox_inches="tight")
    print(f"Saved figure to: {outname}  FEC_AvgClassified_CR.py:738  FEC_AvgClassified_CR_V_2.py:751 - 005_FEC_AvgClassified_CR_V_2.py:869")

    plt.show()


if __name__ == "__main__":
    main()
