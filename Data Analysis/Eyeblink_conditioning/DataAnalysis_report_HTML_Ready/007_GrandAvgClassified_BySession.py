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
        kind=method,
        bounds_error=False,
        fill_value=np.nan
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


def get_trial_timing(tr, short_isi_max=0.30):
    states = get_field(tr, "States", None)
    events = get_field(tr, "Events", None)
    data = get_field(tr, "Data", None)

    t_led_abs = scalar_or_nan(get_field(events, "GlobalTimer1_Start", None))
    t_led_end_abs = scalar_or_nan(get_field(events, "GlobalTimer1_End", None))

    t_puff_abs = scalar_or_nan(get_field(events, "GlobalTimer2_Start", None))
    t_puff_end_abs = scalar_or_nan(get_field(events, "GlobalTimer2_End", None))

    led_puff_isi = safe_array(get_field(states, "LED_Puff_ISI", None))
    if led_puff_isi.size >= 2:
        isi_dur = float(led_puff_isi[1] - led_puff_isi[0])
    else:
        isi_dur = np.nan

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


def estimate_session_open_eye_area(trials, short_isi_max=0.30, exclude_probe=True,
                                   exclude_timeout=True, open_eye_percentile=99):
    """
    Uses the 99th percentile of eyeAreaPixels across ALL frames of all valid
    trials.  Matches the MATLAB convention (normalise by max observed eye area),
    avoiding bias from restricting to the 200 ms pre-LED window which can
    contain anticipatory partial closures in learned animals.
    """
    all_eye_vals = []
    for tr in trials:
        states = get_field(tr, "States", None)
        data   = get_field(tr, "Data", None)
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
    if not np.isfinite(open_eye_area) or open_eye_area <= 0:
        return None
    return 1.0 - np.asarray(eye_area_pixels, dtype=float) / open_eye_area


# ============================================================
# CR classification
# ============================================================

def classify_cr_by_block(
    t_grid,
    fec_trace,
    puff_rel,
    block_label,
    good_cr_threshold=0.05,
    poor_cr_threshold=0.02,
    short_pre_ms=25,
    short_post_ms=12,
    long_pre_ms=50,
    long_post_ms=12
):
    """
    Returns:
        category: "good_cr", "poor_cr", "no_cr"
        peak_amp
        baseline_mean
    """

    if not np.isfinite(puff_rel):
        return "no_cr", np.nan, np.nan

    # baseline
    mb = (t_grid >= -0.2) & (t_grid < 0.0)
    if not np.any(mb):
        return "no_cr", np.nan, np.nan
    baseline_mean = np.nanmean(fec_trace[mb])

    short_pre = short_pre_ms / 1000.0
    short_post = short_post_ms / 1000.0
    long_pre = long_pre_ms / 1000.0
    long_post = long_post_ms / 1000.0

    # CR window depends on short/long
    if block_label == "short":
        cr_start = puff_rel - short_pre
        cr_end = puff_rel + short_post
    else:
        cr_start = puff_rel - long_pre
        cr_end = puff_rel + long_post

    mw = (t_grid >= cr_start) & (t_grid <= cr_end)

    if not np.any(mw):
        return "no_cr", np.nan, baseline_mean

    peak_amp = np.nanmax(fec_trace[mw]) - baseline_mean

    if not np.isfinite(peak_amp):
        return "no_cr", np.nan, baseline_mean

    if peak_amp >= good_cr_threshold:
        return "good_cr", peak_amp, baseline_mean
    elif peak_amp >= poor_cr_threshold:
        return "poor_cr", peak_amp, baseline_mean
    else:
        return "no_cr", peak_amp, baseline_mean


def match_baselines(F_ctrl, F_chemo, t_grid, n_bins=10, rng_seed=42):
    """
    Subsample ctrl and chemo observation lists (session means or trials) so their
    pre-LED baseline (mean FEC in [-0.2, 0) s) distributions match via equal-bin
    subsampling.  Returns (idx_ctrl, idx_chemo) integer index arrays.
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


def align_session_mean_baselines(F_ctrl, F_chemo, t_grid, baseline_window=(-0.2, 0.0)):
    """
    Baseline-align session-average traces without dropping sessions.

    Each session mean trace is shifted so its pre-LED baseline mean matches the
    pooled control+chemo baseline target for that panel. This is more stable
    than bin-wise subsampling for grand averages, where each group can have only
    a handful of session means.
    """
    bl = (t_grid >= baseline_window[0]) & (t_grid < baseline_window[1])
    if not np.any(bl):
        return list(F_ctrl), list(F_chemo)

    def _baselines(traces):
        vals = []
        for tr in traces:
            tr = np.asarray(tr, dtype=float)
            vals.append(np.nanmean(tr[bl]) if tr.size else np.nan)
        return np.asarray(vals, dtype=float)

    b_ctrl = _baselines(F_ctrl)
    b_chemo = _baselines(F_chemo)
    target = np.nanmean(np.concatenate([b_ctrl, b_chemo]))
    if not np.isfinite(target):
        return list(F_ctrl), list(F_chemo)

    def _align(traces, baselines):
        out = []
        for tr, b in zip(traces, baselines):
            tr = np.asarray(tr, dtype=float)
            if np.isfinite(b):
                out.append(tr - b + target)
            else:
                out.append(tr)
        return out

    return _align(F_ctrl, b_ctrl), _align(F_chemo, b_chemo)


# ============================================================
# Stats packing
# ============================================================

def pack_stats(t, session_mean_list, led_starts, led_ends, puff_starts, puff_ends, cr_starts, cr_ends):
    out = {}
    out["t"] = np.asarray(t, dtype=float)

    if len(session_mean_list) == 0:
        session_mean_mat = np.empty((0, len(t)))
    else:
        session_mean_mat = np.vstack(session_mean_list)

    out["n_sessions"] = session_mean_mat.shape[0]
    out["n"] = out["n_sessions"]

    if out["n_sessions"] == 0:
        out["mean"] = np.full_like(out["t"], np.nan, dtype=float)
        out["sem"] = np.full_like(out["t"], np.nan, dtype=float)
        out["led_start"] = np.nan
        out["led_end"] = np.nan
        out["puff_start"] = np.nan
        out["puff_end"] = np.nan
        out["cr_start"] = np.nan
        out["cr_end"] = np.nan
    else:
        out["mean"] = np.nanmean(session_mean_mat, axis=0)
        out["sem"] = np.nanstd(session_mean_mat, axis=0) / np.sqrt(out["n_sessions"])
        out["led_start"] = np.nanmedian(led_starts) if len(led_starts) else np.nan
        out["led_end"] = np.nanmedian(led_ends) if len(led_ends) else np.nan
        out["puff_start"] = np.nanmedian(puff_starts) if len(puff_starts) else np.nan
        out["puff_end"] = np.nanmedian(puff_ends) if len(puff_ends) else np.nan
        out["cr_start"] = np.nanmedian(cr_starts) if len(cr_starts) else np.nan
        out["cr_end"] = np.nanmedian(cr_ends) if len(cr_ends) else np.nan

    return out


def plot_panel_two_groups(ax, S_ctrl, S_chemo, cr_label, block_label):
    for S, color, label in [(S_ctrl, "k", "Control"), (S_chemo, "#D14949", "Chemo")]:
        if np.any(np.isfinite(S["mean"])):
            ax.plot(S["t"], S["mean"], linewidth=1.8, color=color, label=label)
            ax.fill_between(S["t"], S["mean"] + S["sem"], S["mean"] - S["sem"],
                            color=color, alpha=0.15)

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

    if cr_label != "All" and np.isfinite(S_ref["cr_start"]) and np.isfinite(S_ref["cr_end"]):
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

    if cr_label != "All" and np.isfinite(S_ref["cr_start"]) and np.isfinite(S_ref["cr_end"]):
        ax.text(S_ref["cr_start"] + 0.002, y_text - 0.06 * (ymax - ymin), "CR window",
                color="darkorange", va="bottom", ha="left", fontsize=9)

    ax.set_title(
        f"{cr_label}\nControl n={S_ctrl['n']}, Chemo n={S_chemo['n']}",
        fontsize=13
    )
    ax.set_xlabel("Time from LED onset (s)")
    ax.set_ylabel("FEC")
    ax.grid(False)
    ax.tick_params(direction="out")
    ax.legend(frameon=False, loc="best")


def plot_four_row_grand_figure(stats, title, outname, first_date, last_date, n_ctrl_sessions, n_chemo_sessions):
    rows = [
        ("all", "All"),
        ("none", "No CR"),
        ("poor", "Poor CR"),
        ("good", "Good CR"),
    ]
    fig, axs = plt.subplots(4, 2, figsize=(11.5, 10.5), sharex=True, sharey=False)

    for r, (cat, label) in enumerate(rows):
        plot_panel_two_groups(
            axs[r, 0],
            stats[f"short_{cat}_ctrl"],
            stats[f"short_{cat}_chemo"],
            label,
            "short",
        )
        plot_panel_two_groups(
            axs[r, 1],
            stats[f"long_{cat}_ctrl"],
            stats[f"long_{cat}_chemo"],
            label,
            "long",
        )

    fig.suptitle(
        f"{title}\nSessions: {first_date} -- {last_date} | Control={n_ctrl_sessions}, Chemo={n_chemo_sessions}",
        fontsize=15
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(outname, bbox_inches="tight")
    print(f"Saved figure to: {outname}")
    plt.show()


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


# ============================================================
# Main
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_files = sorted(glob.glob(os.path.join(script_dir, "*_EBC_*.mat")))
    if len(data_files) == 0:
        print(f"No *_EBC_*.mat files found in {script_dir}.  GrandAvgClassified_BySession.py:301 - 007_GrandAvgClassified_BySession.py:449")
        return

    # settings
    t_pre = 0.2
    t_post = 0.6
    dt = 1 / 250
    smooth_win = 5
    exclude_probe = True
    exclude_timeout = True
    short_isi_max = 0.30
    interp_method = "linear"

    # CR thresholds
    good_cr_threshold = 0.05
    poor_cr_threshold = 0.02

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

    # across-session accumulators split by ctrl/chemo
    _base_keys = [
        "short_all", "short_good", "short_poor", "short_none",
        "long_all", "long_good", "long_poor", "long_none",
    ]
    groups = {f"{k}_{g}": [] for k in _base_keys for g in ("ctrl", "chemo")}
    timing = {
        f"{k}_{g}": {"led_s": [], "led_e": [], "puff_s": [], "puff_e": [], "cr_s": [], "cr_e": []}
        for k in _base_keys for g in ("ctrl", "chemo")
    }

    n_ctrl_sessions = 0
    n_chemo_sessions = 0

    for filepath in data_files:
        print(f"Processing: {filepath}  GrandAvgClassified_BySession.py:361 - 007_GrandAvgClassified_BySession.py:501")
        SD = matlab_load_sessiondata(filepath)
        is_chemo = is_chemogenetics_session(SD)

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
            print(f"Skipping {os.path.basename(filepath)}: could not estimate session openeye area - 007_GrandAvgClassified_BySession.py:533")
            continue

        # per-session trace collections
        sess = {
            "short_all": [],
            "short_good": [],
            "short_poor": [],
            "short_none": [],
            "long_all": [],
            "long_good": [],
            "long_poor": [],
            "long_none": [],
        }

        sess_time = {
            "short_all": {"led_s": [], "led_e": [], "puff_s": [], "puff_e": [], "cr_s": [], "cr_e": []},
            "short_good": {"led_s": [], "led_e": [], "puff_s": [], "puff_e": [], "cr_s": [], "cr_e": []},
            "short_poor": {"led_s": [], "led_e": [], "puff_s": [], "puff_e": [], "cr_s": [], "cr_e": []},
            "short_none": {"led_s": [], "led_e": [], "puff_s": [], "puff_e": [], "cr_s": [], "cr_e": []},
            "long_all":   {"led_s": [], "led_e": [], "puff_s": [], "puff_e": [], "cr_s": [], "cr_e": []},
            "long_good":  {"led_s": [], "led_e": [], "puff_s": [], "puff_e": [], "cr_s": [], "cr_e": []},
            "long_poor":  {"led_s": [], "led_e": [], "puff_s": [], "puff_e": [], "cr_s": [], "cr_e": []},
            "long_none":  {"led_s": [], "led_e": [], "puff_s": [], "puff_e": [], "cr_s": [], "cr_e": []},
        }

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

            t_led_abs, t_led_end_abs, t_puff_abs, t_puff_end_abs, isi_dur, is_short, block_label = get_trial_timing(
                tr, short_isi_max=short_isi_max
            )

            if not np.isfinite(t_led_abs) or not np.isfinite(t_puff_abs):
                continue

            t_rel = FECTimes - t_led_abs
            FEC = normalize_fec_to_open_eye_reference(eyeAreaPixels, open_eye_area)
            if FEC is None:
                continue
            led_start_rel = 0.0
            led_end_rel = t_led_end_abs - t_led_abs if np.isfinite(t_led_end_abs) else np.nan
            puff_start_rel = t_puff_abs - t_led_abs
            puff_end_rel = t_puff_end_abs - t_led_abs if np.isfinite(t_puff_end_abs) else np.nan

            if smooth_win > 1:
                FEC = smooth_trace(FEC, smooth_win)

            Fq = interp_to_grid(t_rel, FEC, t_grid, method=interp_method)

            category, peak_amp, baseline_mean = classify_cr_by_block(
                t_grid, Fq, puff_start_rel, block_label,
                good_cr_threshold=good_cr_threshold,
                poor_cr_threshold=poor_cr_threshold
            )

            if block_label == "short":
                if category == "good_cr":
                    key = "short_good"
                elif category == "poor_cr":
                    key = "short_poor"
                else:
                    key = "short_none"
            else:
                if category == "good_cr":
                    key = "long_good"
                elif category == "poor_cr":
                    key = "long_poor"
                else:
                    key = "long_none"

            if block_label == "short":
                cr_start_rel = puff_start_rel - 0.025
                cr_end_rel   = puff_start_rel + 0.012
            else:
                cr_start_rel = puff_start_rel - 0.050
                cr_end_rel   = puff_start_rel + 0.012

            sess[key].append(Fq)
            sess_time[key]["led_s"].append(led_start_rel)
            sess_time[key]["led_e"].append(led_end_rel)
            sess_time[key]["puff_s"].append(puff_start_rel)
            sess_time[key]["puff_e"].append(puff_end_rel)
            sess_time[key]["cr_s"].append(cr_start_rel)
            sess_time[key]["cr_e"].append(cr_end_rel)

            all_key = "short_all" if block_label == "short" else "long_all"
            sess[all_key].append(Fq)
            sess_time[all_key]["led_s"].append(led_start_rel)
            sess_time[all_key]["led_e"].append(led_end_rel)
            sess_time[all_key]["puff_s"].append(puff_start_rel)
            sess_time[all_key]["puff_e"].append(puff_end_rel)
            sess_time[all_key]["cr_s"].append(np.nan)
            sess_time[all_key]["cr_e"].append(np.nan)

        # average each session separately, then append session mean
        grp = "chemo" if is_chemo else "ctrl"
        for base_key in sess.keys():
            if len(sess[base_key]) > 0:
                mat = np.vstack(sess[base_key])
                gkey = f"{base_key}_{grp}"
                groups[gkey].append(np.nanmean(mat, axis=0))
                timing[gkey]["led_s"].append(np.nanmedian(sess_time[base_key]["led_s"]))
                timing[gkey]["led_e"].append(np.nanmedian(sess_time[base_key]["led_e"]))
                timing[gkey]["puff_s"].append(np.nanmedian(sess_time[base_key]["puff_s"]))
                timing[gkey]["puff_e"].append(np.nanmedian(sess_time[base_key]["puff_e"]))
                timing[gkey]["cr_s"].append(np.nanmedian(sess_time[base_key]["cr_s"]))
                timing[gkey]["cr_e"].append(np.nanmedian(sess_time[base_key]["cr_e"]))

    raw_groups = {k: list(v) for k, v in groups.items()}
    raw_timing = {
        k: {tk: list(vals) for tk, vals in timing[k].items()}
        for k in timing
    }

    # ---- baseline alignment at session level: equalize pre-LED baseline means without dropping sessions ----
    for _bk in _base_keys:
        _kc = f"{_bk}_ctrl"
        _kh = f"{_bk}_chemo"
        if not groups[_kc] or not groups[_kh]:
            continue
        groups[_kc], groups[_kh] = align_session_mean_baselines(groups[_kc], groups[_kh], t_grid)
    print("Baseline-aligned (session means per CR category):")
    for _bk in _base_keys:
        _nc = len(groups[f"{_bk}_ctrl"])
        _nh = len(groups[f"{_bk}_chemo"])
        print(f"  {_bk:12s}: ctrl={_nc}, chemo={_nh}")

    def pack_all(src_groups, src_timing):
        return {
            key: pack_stats(
                t_grid,
                src_groups[key],
                src_timing[key]["led_s"],
                src_timing[key]["led_e"],
                src_timing[key]["puff_s"],
                src_timing[key]["puff_e"],
                src_timing[key]["cr_s"],
                src_timing[key]["cr_e"],
            )
            for key in src_groups.keys()
        }

    raw_stats = pack_all(raw_groups, raw_timing)
    stats = pack_all(groups, timing)

    raw_outname = f"GrandAvgFEC_BySession_Raw_AllAndCRclassified_ShortLong_{first_tok}_to_{last_tok}.pdf"
    plot_four_row_grand_figure(
        raw_stats,
        "Raw Grand Avg FEC (Session-wise Means): All Trials and CR-classified Short/Long",
        raw_outname,
        first_date,
        last_date,
        n_ctrl_sessions,
        n_chemo_sessions,
    )

    matched_outname = f"GrandAvgFEC_BySession_BaselineMatched_AllAndCRclassified_ShortLong_{first_tok}_to_{last_tok}.pdf"
    plot_four_row_grand_figure(
        stats,
        "Baseline-matched Grand Avg FEC (Session-wise Means): All Trials and CR-classified Short/Long",
        matched_outname,
        first_date,
        last_date,
        n_ctrl_sessions,
        n_chemo_sessions,
    )

    legacy_stats = {}
    for key in groups.keys():
        legacy_stats[key] = pack_stats(
            t_grid,
            groups[key],
            timing[key]["led_s"],
            timing[key]["led_e"],
            timing[key]["puff_s"],
            timing[key]["puff_e"],
            timing[key]["cr_s"],
            timing[key]["cr_e"]
        )

    fig, axs = plt.subplots(3, 2, figsize=(11.5, 12), sharex=True, sharey=False)

    plot_panel_two_groups(axs[0, 0], legacy_stats["short_good_ctrl"], legacy_stats["short_good_chemo"], "Good CR", "short")
    plot_panel_two_groups(axs[0, 1], legacy_stats["long_good_ctrl"],  legacy_stats["long_good_chemo"],  "Good CR", "long")

    plot_panel_two_groups(axs[1, 0], legacy_stats["short_poor_ctrl"], legacy_stats["short_poor_chemo"], "Poor CR", "short")
    plot_panel_two_groups(axs[1, 1], legacy_stats["long_poor_ctrl"],  legacy_stats["long_poor_chemo"],  "Poor CR", "long")

    plot_panel_two_groups(axs[2, 0], legacy_stats["short_none_ctrl"], legacy_stats["short_none_chemo"], "No CR", "short")
    plot_panel_two_groups(axs[2, 1], legacy_stats["long_none_ctrl"],  legacy_stats["long_none_chemo"],  "No CR", "long")

    fig.suptitle(
        f"Grand Avg Classified FEC (Session-wise Means)\n"
        f"Sessions: {first_date} -- {last_date} | Control={n_ctrl_sessions}, Chemo={n_chemo_sessions}",
        fontsize=15
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    outname = f"GrandAvgClassifiedFEC_BySession_{first_tok}_to_{last_tok}.pdf"
    plt.savefig(outname, bbox_inches="tight")
    print(f"Saved figure to: {outname}  GrandAvgClassified_BySession.py:505 - 007_GrandAvgClassified_BySession.py:686")

    plt.show()


if __name__ == "__main__":
    main()
