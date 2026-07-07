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


def normalize_fec_to_baseline(t_rel, eye_area_pixels, baseline_window=(-0.2, 0.0)):
    t_rel = np.asarray(t_rel, dtype=float)
    eye_area_pixels = np.asarray(eye_area_pixels, dtype=float)

    baseline_idx = (
        np.isfinite(t_rel)
        & np.isfinite(eye_area_pixels)
        & (t_rel >= baseline_window[0])
        & (t_rel < baseline_window[1])
    )
    if not np.any(baseline_idx):
        return None

    open_eye_area = np.nanmedian(eye_area_pixels[baseline_idx])
    if not np.isfinite(open_eye_area) or open_eye_area <= 0:
        return None

    return 1.0 - eye_area_pixels / open_eye_area


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


def parse_session_label(filepath):
    m = re.search(r"(\d{8})", os.path.basename(filepath))
    if not m:
        return os.path.basename(filepath)
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").strftime("%m/%d/%Y")
    except Exception:
        return os.path.basename(filepath)


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


def pack_grand_stats(t, session_mean_mat, led_starts, led_ends, puff_starts, puff_ends):
    out = {}
    out["t"] = np.asarray(t, dtype=float)

    if isinstance(session_mean_mat, list):
        if len(session_mean_mat) == 0:
            session_mean_mat = np.empty((0, len(t)))
        else:
            session_mean_mat = np.vstack(session_mean_mat)
    else:
        session_mean_mat = np.asarray(session_mean_mat, dtype=float)
        if session_mean_mat.size == 0:
            session_mean_mat = np.empty((0, len(t)))

    out["n_sessions"] = session_mean_mat.shape[0]

    if out["n_sessions"] == 0:
        out["mean"] = np.full_like(out["t"], np.nan, dtype=float)
        out["sem"] = np.full_like(out["t"], np.nan, dtype=float)
        out["led_start"] = np.nan
        out["led_end"] = np.nan
        out["puff_start"] = np.nan
        out["puff_end"] = np.nan
    else:
        out["mean"] = np.nanmean(session_mean_mat, axis=0)
        out["sem"] = np.nanstd(session_mean_mat, axis=0) / np.sqrt(out["n_sessions"])
        out["led_start"] = np.nanmedian(led_starts) if len(led_starts) else np.nan
        out["led_end"] = np.nanmedian(led_ends) if len(led_ends) else np.nan
        out["puff_start"] = np.nanmedian(puff_starts) if len(puff_starts) else np.nan
        out["puff_end"] = np.nanmedian(puff_ends) if len(puff_ends) else np.nan

    return out


def plot_panel_two_groups(ax, S_ctrl, S_chemo, block_label):
    for S, color, label in [(S_ctrl, "k", "Control"), (S_chemo, "#D14949", "Chemo")]:
        if np.any(np.isfinite(S["mean"])):
            ax.plot(S["t"], S["mean"], linewidth=2.0, color=color, label=label)
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

    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.03 * (ymax - ymin)

    if np.isfinite(S_ref["led_start"]) and np.isfinite(S_ref["led_end"]):
        ax.text(S_ref["led_start"] + 0.002, y_text, "LED",
                color="black", va="bottom", ha="left", fontsize=10)

    if np.isfinite(S_ref["puff_start"]) and np.isfinite(S_ref["puff_end"]):
        ax.text(S_ref["puff_start"] + 0.002, y_text, "AirPuff",
                color=puff_text_color, va="bottom", ha="left", fontsize=10)

    ax.set_title(
        f"{block_label} blocks\nControl n={S_ctrl['n_sessions']}, Chemo n={S_chemo['n_sessions']}",
        fontsize=14
    )
    ax.set_xlabel("Time from LED onset (s)")
    ax.set_ylabel("FEC")
    ax.grid(False)
    ax.tick_params(direction="out")
    ax.legend(frameon=False, loc="best")


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

    ax.set_title(f"{block_label} blocks (sessions={S['n_sessions']})", fontsize=14)
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
        print(f"No *_EBC_*.mat files found in {script_dir}.  GranAvgAll_FEC.py:283 - 006_GranAvgAll_FEC.py:402")
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
    apply_baseline_matching = False

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
    session_means = {k: [] for k in ("short_ctrl", "short_chemo", "long_ctrl", "long_chemo")}
    timing_acc = {
        k: {"led_s": [], "led_e": [], "puff_s": [], "puff_e": []}
        for k in ("short_ctrl", "short_chemo", "long_ctrl", "long_chemo")
    }

    n_sessions_used = 0
    n_ctrl_sessions = 0
    n_chemo_sessions = 0
    used_ctrl_labels = []
    used_chemo_labels = []

    for filepath in data_files:
        print(f"Processing: {filepath}  GranAvgAll_FEC.py:335 - 006_GranAvgAll_FEC.py:453")
        SD = matlab_load_sessiondata(filepath)
        is_chemo = is_chemogenetics_session(SD)
        session_label = parse_session_label(filepath)

        raw_events = get_field(SD, "RawEvents", None)
        trials = ensure_trial_list(get_field(raw_events, "Trial", None))

        # per-session accumulators
        F_short_sess = []
        led_short_start_sess, led_short_end_sess = [], []
        puff_short_start_sess, puff_short_end_sess = [], []

        F_long_sess = []
        led_long_start_sess, led_long_end_sess = [], []
        puff_long_start_sess, puff_long_end_sess = [], []

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

            if not np.isfinite(t_led_abs) or not np.isfinite(t_puff_abs) or not np.isfinite(isi_dur):
                continue

            t_rel = FECTimes - t_led_abs
            FEC = normalize_fec_to_baseline(t_rel, eyeAreaPixels)
            if FEC is None:
                continue

            led_start_rel = 0.0
            led_end_rel = t_led_end_abs - t_led_abs if np.isfinite(t_led_end_abs) else np.nan
            puff_start_rel = t_puff_abs - t_led_abs
            puff_end_rel = t_puff_end_abs - t_led_abs if np.isfinite(t_puff_end_abs) else np.nan

            if smooth_win > 1:
                FEC = smooth_trace(FEC, smooth_win)

            Fq = interp_to_grid(t_rel, FEC, t_grid, method=interp_method)

            if is_short:
                F_short_sess.append(Fq)
                led_short_start_sess.append(led_start_rel)
                led_short_end_sess.append(led_end_rel)
                puff_short_start_sess.append(puff_start_rel)
                puff_short_end_sess.append(puff_end_rel)
            else:
                F_long_sess.append(Fq)
                led_long_start_sess.append(led_start_rel)
                led_long_end_sess.append(led_end_rel)
                puff_long_start_sess.append(puff_start_rel)
                puff_long_end_sess.append(puff_end_rel)

        grp = "chemo" if is_chemo else "ctrl"
        used_this_session = False

        if len(F_short_sess) > 0:
            key = f"short_{grp}"
            session_means[key].append(np.nanmean(np.vstack(F_short_sess), axis=0))
            timing_acc[key]["led_s"].append(np.nanmedian(led_short_start_sess))
            timing_acc[key]["led_e"].append(np.nanmedian(led_short_end_sess))
            timing_acc[key]["puff_s"].append(np.nanmedian(puff_short_start_sess))
            timing_acc[key]["puff_e"].append(np.nanmedian(puff_short_end_sess))
            used_this_session = True

        if len(F_long_sess) > 0:
            key = f"long_{grp}"
            session_means[key].append(np.nanmean(np.vstack(F_long_sess), axis=0))
            timing_acc[key]["led_s"].append(np.nanmedian(led_long_start_sess))
            timing_acc[key]["led_e"].append(np.nanmedian(led_long_end_sess))
            timing_acc[key]["puff_s"].append(np.nanmedian(puff_long_start_sess))
            timing_acc[key]["puff_e"].append(np.nanmedian(puff_long_end_sess))
            used_this_session = True

        if used_this_session:
            n_sessions_used += 1
            if is_chemo:
                n_chemo_sessions += 1
                used_chemo_labels.append(session_label)
            else:
                n_ctrl_sessions += 1
                used_ctrl_labels.append(session_label)

    if apply_baseline_matching:
        # Optional: equalize pre-LED baseline distributions by dropping unmatched sessions.
        for _bl in ("short", "long"):
            _kc = f"{_bl}_ctrl"
            _kh = f"{_bl}_chemo"
            if not session_means[_kc] or not session_means[_kh]:
                continue
            _ic, _ih = match_baselines(session_means[_kc], session_means[_kh], t_grid)
            session_means[_kc] = [session_means[_kc][i] for i in _ic]
            session_means[_kh] = [session_means[_kh][i] for i in _ih]
            for _side, _idx in (("ctrl", _ic), ("chemo", _ih)):
                _k = f"{_bl}_{_side}"
                for _tk in ("led_s", "led_e", "puff_s", "puff_e"):
                    timing_acc[_k][_tk] = [timing_acc[_k][_tk][i] for i in _idx]
        print(f"Baselinematched (session means): - 006_GranAvgAll_FEC.py:575"
              f"short ctrl={len(session_means['short_ctrl'])}, chemo={len(session_means['short_chemo'])} | "
              f"long ctrl={len(session_means['long_ctrl'])}, chemo={len(session_means['long_chemo'])}")
    else:
        print(f"Averaging all usable session means: - 006_GranAvgAll_FEC.py:579"
              f"short ctrl={len(session_means['short_ctrl'])}, chemo={len(session_means['short_chemo'])} | "
              f"long ctrl={len(session_means['long_ctrl'])}, chemo={len(session_means['long_chemo'])}")
    print(f"Control sessions used ({n_ctrl_sessions}): {', '.join(used_ctrl_labels) if used_ctrl_labels else 'none'} - 006_GranAvgAll_FEC.py:582")
    print(f"Chemo sessions used ({n_chemo_sessions}): {', '.join(used_chemo_labels) if used_chemo_labels else 'none'} - 006_GranAvgAll_FEC.py:583")

    def pgs(bl, grp):
        k = f"{bl}_{grp}"
        return pack_grand_stats(
            t_grid, session_means[k],
            timing_acc[k]["led_s"], timing_acc[k]["led_e"],
            timing_acc[k]["puff_s"], timing_acc[k]["puff_e"]
        )

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_panel_two_groups(axs[0], pgs("short", "ctrl"), pgs("short", "chemo"), "Short")
    plot_panel_two_groups(axs[1], pgs("long", "ctrl"), pgs("long", "chemo"), "Long")

    fig.suptitle(
        f"Grand Avg FEC (Session-wise Means)\n"
        f"Sessions: {first_date} -- {last_date} | used={n_sessions_used} | Control={n_ctrl_sessions}, Chemo={n_chemo_sessions}\n"
        f"Chemo: {', '.join(used_chemo_labels) if used_chemo_labels else 'none'}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    outname = f"GrandAvgFEC_BySession_ShortLong_{first_tok}_to_{last_tok}.pdf"
    plt.savefig(outname, bbox_inches="tight")
    print(f"Saved figure to: {outname}  GranAvgAll_FEC.py:472 - 006_GranAvgAll_FEC.py:608")

    plt.show()


if __name__ == "__main__":
    main()
