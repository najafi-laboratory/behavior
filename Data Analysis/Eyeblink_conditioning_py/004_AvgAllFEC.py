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
    if arr.size == 0:
        return np.nan
    if arr.ndim == 0:
        if arr is None:
            return np.nan
        try:
            return float(arr)
        except Exception:
            return np.nan
    try:
        return float(arr.flat[0])
    except Exception:
        return np.nan


def matlab_load_sessiondata(filepath):
    data = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    if "SessionData" not in data:
        raise KeyError(f"SessionData not found in {filepath}")
    return data["SessionData"]


def load_selected_session_names(filename):
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as fh:
        names = [line.strip() for line in fh if line.strip()]
    return set(names) if names else None


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
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if len(time) != len(signal):
        return "No CR"

    baseline_idx = (time >= (t_led - 0.2)) & (time <= t_led)
    if not np.any(baseline_idx):
        return "No CR"

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
        small_mean_thr = 0.01
    elif block_label == "long":
        good_start = t_puff - long_pre
        good_end = t_puff + long_post
        good_thr = good_cr_threshold
        peak_thr = good_cr_threshold
        small_mean_thr = 0.01
    else:
        return "No CR"

    poor_start = t_led
    poor_end = good_start

    good_idx = (time >= good_start) & (time <= good_end)
    poor_idx = (time >= poor_start) & (time < poor_end)

    if not np.any(good_idx):
        return "No CR"

    baseline_amp = np.nanmean(signal[baseline_idx])
    cr_amp = np.nanmean(signal[good_idx])

    good_mean_above = cr_amp - baseline_amp
    good_peak_above = np.nanmax(signal[good_idx]) - baseline_amp
    good_mean_above_baseline = good_mean_above > small_mean_thr

    if np.any(poor_idx):
        poor_amp = np.nanmean(signal[poor_idx])
        poor_mean_above = poor_amp - baseline_amp
    else:
        poor_mean_above = -np.inf

    if (good_mean_above >= good_thr) or (good_peak_above >= peak_thr) or good_mean_above_baseline:
        return "Good CR"
    elif poor_mean_above >= poor_cr_threshold:
        return "Poor CR"
    else:
        return "No CR"


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


def collect_overall_max(trials):
    """
    Version-robust normalization factor.
    Prefer eyeAreaPixels if available; fallback to totalEllipsePixels.
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

    # LED
    if np.isfinite(S["led_start"]) and np.isfinite(S["led_end"]):
        ax.axvspan(S["led_start"], S["led_end"], color="gray", alpha=0.18)

    # AirPuff
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
    ax.tick_params(direction="out")


def plot_panel_two_groups(ax, S_ctrl, S_chemo, block_label):
    # Control = black
    if np.any(np.isfinite(S_ctrl["mean"])):
        ax.plot(S_ctrl["t"], S_ctrl["mean"], linewidth=2.0, color="k", label="Control")
        ax.fill_between(
            S_ctrl["t"],
            S_ctrl["mean"] + S_ctrl["sem"],
            S_ctrl["mean"] - S_ctrl["sem"],
            color="k", alpha=0.15
        )

    # Chemo = red
    if np.any(np.isfinite(S_chemo["mean"])):
        ax.plot(S_chemo["t"], S_chemo["mean"], linewidth=2.0, color="#D14949", label="Chemo")
        ax.fill_between(
            S_chemo["t"],
            S_chemo["mean"] + S_chemo["sem"],
            S_chemo["mean"] - S_chemo["sem"],
            color="#D14949", alpha=0.15
        )

    # Use control timing if available, otherwise chemo
    S_ref = S_ctrl if np.any(np.isfinite(S_ctrl["mean"])) else S_chemo

    # LED
    if np.isfinite(S_ref["led_start"]) and np.isfinite(S_ref["led_end"]):
        ax.axvspan(S_ref["led_start"], S_ref["led_end"], color="gray", alpha=0.18)

    # AirPuff
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
        f"{block_label} blocks\nControl n={S_ctrl['n']}, Chemo n={S_chemo['n']}",
        fontsize=14
    )
    ax.set_xlabel("Time from LED onset (s)")
    ax.set_ylabel("FEC")
    ax.grid(False)
    ax.tick_params(direction="out")
    ax.legend(frameon=False, loc="best")


def plot_per_session_superimposed(
    session_avgs_ctrl, session_avgs_chemo,
    t_grid, first_date, last_date,
    n_ctrl_sessions, n_chemo_sessions,
    first_tok, last_tok,
    script_dir,
):
    """Superimpose per-session average FEC traces.

    Control sessions: black → grey gradient (darker = earlier sessions).
    Chemo sessions:   dark red → light red gradient.
    Grand mean per group shown as a thick solid line on top.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    for ax, block in zip(axs, ["short", "long"]):
        puff_x    = 0.200 if block == "short" else 0.400
        puff_col  = "dodgerblue" if block == "short" else "limegreen"

        ctrl_sessions  = session_avgs_ctrl[block]
        chemo_sessions = session_avgs_chemo[block]
        n_ctrl  = len(ctrl_sessions)
        n_chemo = len(chemo_sessions)

        # ── Control: black → medium grey ──────────────────────────────
        for i, (name, trace) in enumerate(ctrl_sessions):
            frac = i / max(n_ctrl - 1, 1)
            g = frac * 0.65          # 0 = black, 0.65 = medium grey
            ax.plot(t_grid, trace, color=(g, g, g), linewidth=1.2, alpha=0.85,
                    label=name if i == 0 else "_nolegend_")

        if ctrl_sessions:
            grand_ctrl = np.nanmean(
                np.vstack([tr for _, tr in ctrl_sessions]), axis=0
            )
            ax.plot(t_grid, grand_ctrl, color="black", linewidth=2.5, zorder=5,
                    label=f"Control grand mean ({n_ctrl} sessions)")

        # ── Chemo: dark red → light red ──────────────────────────────
        for i, (name, trace) in enumerate(chemo_sessions):
            frac = i / max(n_chemo - 1, 1)
            r = 0.55 + frac * 0.40
            g = frac * 0.35
            b = frac * 0.35
            ax.plot(t_grid, trace, color=(r, g, b), linewidth=1.2, alpha=0.85,
                    label=name if i == 0 else "_nolegend_")

        if chemo_sessions:
            grand_chemo = np.nanmean(
                np.vstack([tr for _, tr in chemo_sessions]), axis=0
            )
            ax.plot(t_grid, grand_chemo, color="#D14949", linewidth=2.5, zorder=5,
                    label=f"Chemo grand mean ({n_chemo} sessions)")

        # Annotations
        ax.axvspan(0.0, 0.05, color="gray", alpha=0.18)
        ax.axvspan(puff_x, puff_x + 0.02, color=puff_col, alpha=0.22)
        ax.axvline(puff_x, linestyle="--", color="gray", linewidth=0.8, alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)

        ax.set_title(
            f"{block.title()} block — per-session averages\n"
            f"ctrl: black → grey  |  chemo: dark → light red"
        )
        ax.set_xlabel("Time from LED onset (s)")
        ax.set_ylabel("FEC")
        ax.set_xlim(t_grid[0], t_grid[-1])
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    fig.suptitle(
        f"Per-session average FEC — Control (black→grey) vs Chemo (dark→light red)\n"
        f"Sessions: {first_date} — {last_date}  |  "
        f"Control n={n_ctrl_sessions}, Chemo n={n_chemo_sessions}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    outname = os.path.join(
        script_dir,
        f"PerSession_AvgFEC_ShortLong_{first_tok}_to_{last_tok}.pdf",
    )
    fig.savefig(outname, bbox_inches="tight")
    print(f"Saved per-session figure to: {outname}")
    return fig


# ============================================================
# Main
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_files = sorted(glob.glob(os.path.join(script_dir, "*_EBC_*.mat")))
    if len(data_files) == 0:
        print(f"No *_EBC_*.mat files found in {script_dir}.  AvgAllFEC.py:295 - 004_AvgAllFEC.py:527")
        return

    selected_sessions_file = os.path.join(script_dir, "selected_barplot_sessions.txt")
    selected_sessions = load_selected_session_names(selected_sessions_file)
    use_barplot_session_list = selected_sessions is not None
    if use_barplot_session_list:
        data_files = [f for f in data_files if os.path.basename(f) in selected_sessions]
        print(f"Using {len(data_files)} barplotselected sessions from {selected_sessions_file} - 004_AvgAllFEC.py:535")
        if len(data_files) == 0:
            print(f"No selected sessions found in {selected_sessions_file}. - 004_AvgAllFEC.py:537")
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

    # control accumulators
    F_short_ctrl = []
    led_short_start_ctrl, led_short_end_ctrl = [], []
    puff_short_start_ctrl, puff_short_end_ctrl = [], []

    F_long_ctrl = []
    led_long_start_ctrl, led_long_end_ctrl = [], []
    puff_long_start_ctrl, puff_long_end_ctrl = [], []

    # chemo accumulators
    F_short_chemo = []
    led_short_start_chemo, led_short_end_chemo = [], []
    puff_short_start_chemo, puff_short_end_chemo = [], []

    F_long_chemo = []
    led_long_start_chemo, led_long_end_chemo = [], []
    puff_long_start_chemo, puff_long_end_chemo = [], []

    n_sessions_used = 0
    n_ctrl_sessions_used = 0
    n_chemo_sessions_used = 0
    selected_session_dates = []

    # Per-session mean traces for the superimposed session plot
    session_avgs_ctrl  = {"short": [], "long": []}
    session_avgs_chemo = {"short": [], "long": []}

    # CR classification settings for session filtering
    good_cr_threshold = 0.05
    poor_cr_threshold = 0.02
    short_cr_pre_ms = 25
    short_cr_post_ms = 20
    long_cr_pre_ms = 50
    long_cr_post_ms = 0

    for filepath in data_files:
        print(f"Processing: {filepath}  AvgAllFEC.py:344 - 004_AvgAllFEC.py:605")
        SD = matlab_load_sessiondata(filepath)
        is_chemo = is_chemogenetics_session(SD)

        raw_events = get_field(SD, "RawEvents", None)
        trials = ensure_trial_list(get_field(raw_events, "Trial", None))
        open_eye_area = estimate_session_open_eye_area(
            trials,
            short_isi_max=short_isi_max,
            exclude_probe=exclude_probe,
            exclude_timeout=exclude_timeout
        )
        if not np.isfinite(open_eye_area):
            print(f"Skipping {os.path.basename(filepath)}: could not estimate session openeye area  AvgAllFEC.py - 004_AvgAllFEC.py:635")
            continue

        session_F_short = []
        session_led_short_start, session_led_short_end = [], []
        session_puff_short_start, session_puff_short_end = [], []
        session_F_long = []
        session_led_long_start, session_led_long_end = [], []
        session_puff_long_start, session_puff_long_end = [], []

        num_good_short = 0
        num_good_long = 0
        num_short_trials = 0
        num_long_trials = 0

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

            # reconstruct FEC from eyeAreaPixels
            FECTimes = safe_array(get_field(data, "FECTimes", None))
            eyeAreaPixels = safe_array(get_field(data, "eyeAreaPixels", None))

            if FECTimes.size == 0 or eyeAreaPixels.size == 0:
                continue
            if FECTimes.size != eyeAreaPixels.size:
                continue

            # version-robust timing
            t_led_abs, t_led_end_abs, t_puff_abs, t_puff_end_abs, isi_dur, is_short, block_label = get_trial_timing(
                tr, short_isi_max=short_isi_max
            )

            if not np.isfinite(t_led_abs) or not np.isfinite(t_puff_abs) or not np.isfinite(isi_dur):
                continue

            # LED-aligned time
            t_rel = FECTimes - t_led_abs
            FEC = normalize_fec_to_open_eye_reference(eyeAreaPixels, open_eye_area)
            if FEC is None:
                continue

            led_start_rel = 0.0
            led_end_rel = t_led_end_abs - t_led_abs if np.isfinite(t_led_end_abs) else np.nan

            puff_start_rel = t_puff_abs - t_led_abs
            puff_end_rel = t_puff_end_abs - t_led_abs if np.isfinite(t_puff_end_abs) else np.nan

            # smooth
            if smooth_win > 1:
                FEC = smooth_trace(FEC, smooth_win)

            # interpolate
            Fq = interp_to_grid(t_rel, FEC, t_grid, method=interp_method)

            # classify session CR
            t_led_rel = led_start_rel
            cr_category = classify_cr_by_block(
                t_rel,
                FEC,
                t_led_rel,
                puff_start_rel,
                good_cr_threshold,
                poor_cr_threshold,
                block_label,
                short_pre_ms=short_cr_pre_ms,
                short_post_ms=short_cr_post_ms,
                long_pre_ms=long_cr_pre_ms,
                long_post_ms=long_cr_post_ms
            )

            if is_short:
                num_short_trials += 1
                if cr_category == "Good CR":
                    num_good_short += 1
                session_F_short.append(Fq)
                session_led_short_start.append(led_start_rel)
                session_led_short_end.append(led_end_rel)
                session_puff_short_start.append(puff_start_rel)
                session_puff_short_end.append(puff_end_rel)
            else:
                num_long_trials += 1
                if cr_category == "Good CR":
                    num_good_long += 1
                session_F_long.append(Fq)
                session_led_long_start.append(led_start_rel)
                session_led_long_end.append(led_end_rel)
                session_puff_long_start.append(puff_start_rel)
                session_puff_long_end.append(puff_end_rel)

        if not use_barplot_session_list:
            if num_short_trials == 0 or num_long_trials == 0:
                print(
                    f"Skipping {os.path.basename(filepath)}: single-block session (short_trials={num_short_trials}, "
                    f"long_trials={num_long_trials})  AvgAllFEC.py:390"
                )
                continue

            short_good_fraction = num_good_short / max(num_short_trials, 1)
            long_good_fraction = num_good_long / max(num_long_trials, 1)

            if short_good_fraction < 0.20 or long_good_fraction < 0.20:
                print(
                    f"Skipping {os.path.basename(filepath)}: short good CR {short_good_fraction:.2f}, "
                    f"long good CR {long_good_fraction:.2f}; one or both < 0.20  AvgAllFEC.py:395"
                )
                continue

        # keep session
        n_sessions_used += 1
        if is_chemo:
            n_chemo_sessions_used += 1
        else:
            n_ctrl_sessions_used += 1

        # only accumulate accepted session dates for the final title
        m = re.search(r"\d{8}", os.path.basename(filepath))
        if m:
            try:
                selected_session_dates.append(datetime.strptime(m.group(), "%Y%m%d"))
            except Exception:
                pass

        # Store per-session means for the superimposed session plot
        target_sess = session_avgs_chemo if is_chemo else session_avgs_ctrl
        sess_name = os.path.basename(filepath)
        if session_F_short:
            target_sess["short"].append(
                (sess_name, np.nanmean(np.vstack(session_F_short), axis=0))
            )
        if session_F_long:
            target_sess["long"].append(
                (sess_name, np.nanmean(np.vstack(session_F_long), axis=0))
            )

        if is_chemo:
            F_short_chemo.extend(session_F_short)
            led_short_start_chemo.extend(session_led_short_start)
            led_short_end_chemo.extend(session_led_short_end)
            puff_short_start_chemo.extend(session_puff_short_start)
            puff_short_end_chemo.extend(session_puff_short_end)

            F_long_chemo.extend(session_F_long)
            led_long_start_chemo.extend(session_led_long_start)
            led_long_end_chemo.extend(session_led_long_end)
            puff_long_start_chemo.extend(session_puff_long_start)
            puff_long_end_chemo.extend(session_puff_long_end)
        else:
            F_short_ctrl.extend(session_F_short)
            led_short_start_ctrl.extend(session_led_short_start)
            led_short_end_ctrl.extend(session_led_short_end)
            puff_short_start_ctrl.extend(session_puff_short_start)
            puff_short_end_ctrl.extend(session_puff_short_end)

            F_long_ctrl.extend(session_F_long)
            led_long_start_ctrl.extend(session_led_long_start)
            led_long_end_ctrl.extend(session_led_long_end)
            puff_long_start_ctrl.extend(session_puff_long_start)
            puff_long_end_ctrl.extend(session_puff_long_end)

    # ---- raw (unmatched) stats – ctrl and chemo kept separate ----
    S_short_ctrl_raw = pack_stats(t_grid, list(F_short_ctrl),
                                  list(led_short_start_ctrl), list(led_short_end_ctrl),
                                  list(puff_short_start_ctrl), list(puff_short_end_ctrl))
    S_long_ctrl_raw  = pack_stats(t_grid, list(F_long_ctrl),
                                  list(led_long_start_ctrl), list(led_long_end_ctrl),
                                  list(puff_long_start_ctrl), list(puff_long_end_ctrl))
    S_short_chemo_raw = pack_stats(t_grid, list(F_short_chemo),
                                   list(led_short_start_chemo), list(led_short_end_chemo),
                                   list(puff_short_start_chemo), list(puff_short_end_chemo))
    S_long_chemo_raw  = pack_stats(t_grid, list(F_long_chemo),
                                   list(led_long_start_chemo), list(led_long_end_chemo),
                                   list(puff_long_start_chemo), list(puff_long_end_chemo))

    # ---- baseline matching: subsample so ctrl/chemo have equal pre-LED baseline distributions ----
    def _bm(F_c, F_h, *pairs):
        ic, ih = match_baselines(F_c, F_h, t_grid)
        out = [[F_c[i] for i in ic], [F_h[i] for i in ih]]
        for a_c, a_h in pairs:
            out.append([a_c[i] for i in ic])
            out.append([a_h[i] for i in ih])
        return out

    (F_short_ctrl, F_short_chemo,
     led_short_start_ctrl,  led_short_start_chemo,
     led_short_end_ctrl,    led_short_end_chemo,
     puff_short_start_ctrl, puff_short_start_chemo,
     puff_short_end_ctrl,   puff_short_end_chemo) = _bm(
        F_short_ctrl, F_short_chemo,
        (led_short_start_ctrl,  led_short_start_chemo),
        (led_short_end_ctrl,    led_short_end_chemo),
        (puff_short_start_ctrl, puff_short_start_chemo),
        (puff_short_end_ctrl,   puff_short_end_chemo),
    )
    (F_long_ctrl, F_long_chemo,
     led_long_start_ctrl,  led_long_start_chemo,
     led_long_end_ctrl,    led_long_end_chemo,
     puff_long_start_ctrl, puff_long_start_chemo,
     puff_long_end_ctrl,   puff_long_end_chemo) = _bm(
        F_long_ctrl, F_long_chemo,
        (led_long_start_ctrl,  led_long_start_chemo),
        (led_long_end_ctrl,    led_long_end_chemo),
        (puff_long_start_ctrl, puff_long_start_chemo),
        (puff_long_end_ctrl,   puff_long_end_chemo),
    )
    print(f"Baselinematched: short ctrl={len(F_short_ctrl)}, chemo={len(F_short_chemo)} - 004_AvgAllFEC.py:844"
          f" | long ctrl={len(F_long_ctrl)}, chemo={len(F_long_chemo)}")

    # pack stats
    S_short_ctrl = pack_stats(t_grid, F_short_ctrl,
                              led_short_start_ctrl, led_short_end_ctrl,
                              puff_short_start_ctrl, puff_short_end_ctrl)
    S_long_ctrl  = pack_stats(t_grid, F_long_ctrl,
                              led_long_start_ctrl, led_long_end_ctrl,
                              puff_long_start_ctrl, puff_long_end_ctrl)

    S_short_chemo = pack_stats(t_grid, F_short_chemo,
                               led_short_start_chemo, led_short_end_chemo,
                               puff_short_start_chemo, puff_short_end_chemo)
    S_long_chemo  = pack_stats(t_grid, F_long_chemo,
                               led_long_start_chemo, led_long_end_chemo,
                               puff_long_start_chemo, puff_long_end_chemo)

    print(f"Accepted {n_sessions_used}/{len(data_files)} sessions - 004_AvgAllFEC.py:862"
          f"(Control={n_ctrl_sessions_used}, Chemo={n_chemo_sessions_used}) - 004_AvgAllFEC.py")

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_panel_two_groups(axs[0], S_short_ctrl, S_short_chemo, "Short")
    plot_panel_two_groups(axs[1], S_long_ctrl,  S_long_chemo,  "Long")

    if selected_session_dates:
        first_date = min(selected_session_dates).strftime("%m/%d/%Y")
        last_date = max(selected_session_dates).strftime("%m/%d/%Y")
        first_tok = min(selected_session_dates).strftime("%Y%m%d")
        last_tok = max(selected_session_dates).strftime("%Y%m%d")

    fig.suptitle(
        f"Pooled Avg FEC Without Classification\n"
        f"Sessions: {first_date} -- {last_date} | "
        f"Control={n_ctrl_sessions_used}, Chemo={n_chemo_sessions_used}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    outname = os.path.join(script_dir, f"PooledAvgFEC_AllTrials_ShortLong_{first_tok}_to_{last_tok}.pdf")
    plt.savefig(outname, bbox_inches="tight")
    print(f"Saved figure to: {outname}  AvgAllFEC.py:443 - 004_AvgAllFEC.py:887")

    plt.show()

    # ---- Figure 2: raw average — ctrl vs chemo, no baseline matching ----
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_panel_two_groups(axs2[0], S_short_ctrl_raw, S_short_chemo_raw, "Short")
    plot_panel_two_groups(axs2[1], S_long_ctrl_raw,  S_long_chemo_raw,  "Long")

    fig2.suptitle(
        f"Pooled Avg FEC — Raw (No Baseline Matching)\n"
        f"Sessions: {first_date} -- {last_date} | "
        f"Control={n_ctrl_sessions_used}, Chemo={n_chemo_sessions_used}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    outname_raw = os.path.join(script_dir, f"PooledAvgFEC_Raw_ShortLong_{first_tok}_to_{last_tok}.pdf")
    plt.savefig(outname_raw, bbox_inches="tight")
    print(f"Saved raw figure to: {outname_raw} - 004_AvgAllFEC.py:907")

    plt.show()

    # ---- Figure 3: per-session averages superimposed ----
    plot_per_session_superimposed(
        session_avgs_ctrl, session_avgs_chemo,
        t_grid,
        first_date, last_date,
        n_ctrl_sessions_used, n_chemo_sessions_used,
        first_tok, last_tok,
        script_dir,
    )
    plt.show()


if __name__ == "__main__":
    main()
