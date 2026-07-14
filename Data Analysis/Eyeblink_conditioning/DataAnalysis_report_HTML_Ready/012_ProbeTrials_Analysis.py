"""
012_ProbeTrials_Analysis.py

Three analyses of probe trials (CS-only, IsProbeTrial==1) across EBC sessions:

  Figure 1  — Fraction of probe trials per session (bar chart)
  Figure 2  — Probe trial fraction aligned on block transitions (trial-by-trial)
              Separate lines for S→L and L→S transitions; ctrl vs chemo
  Figure 3  — Average FEC: probe−1, probe trial, probe+1
              2×2 panels: (short / long) × (control / chemo)
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from datetime import datetime
from collections import defaultdict


# ============================================================
# Helpers
# ============================================================

def get_field(obj, name, default=None):
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def has_field(obj, name):
    if obj is None:
        return False
    return hasattr(obj, name) or (isinstance(obj, dict) and name in obj)


def safe_array(x):
    if x is None:
        return np.array([], dtype=float)
    arr = np.asarray(x).squeeze()
    try:
        return arr.astype(float)
    except Exception:
        return arr


def scalar_or_nan(x):
    if x is None:
        return np.nan
    arr = np.asarray(x).squeeze()
    if arr.size == 0:
        return np.nan
    if arr.ndim == 0:
        try:
            return float(arr)
        except Exception:
            return np.nan
    try:
        return float(arr.flat[0])
    except Exception:
        return np.nan


def ensure_trial_list(raw):
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        return list(raw.flat)
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


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


def interp_to_grid(x, y, xq):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return np.full_like(xq, np.nan, dtype=float)
    xu, idx = np.unique(x, return_index=True)
    if len(xu) < 2:
        return np.full_like(xq, np.nan, dtype=float)
    return interp1d(xu, y[idx], kind="linear",
                    bounds_error=False, fill_value=np.nan)(xq)


def get_trial_timing(tr, short_isi_max=0.30):
    states = get_field(tr, "States")
    events = get_field(tr, "Events")
    data   = get_field(tr, "Data")

    t_led     = scalar_or_nan(get_field(events, "GlobalTimer1_Start"))
    t_led_end = scalar_or_nan(get_field(events, "GlobalTimer1_End"))
    t_puff    = scalar_or_nan(get_field(events, "GlobalTimer2_Start"))
    t_puff_end= scalar_or_nan(get_field(events, "GlobalTimer2_End"))

    if not np.isfinite(t_puff):
        for cand in ("AirPuff_Onset", "AirPuff_Ons", "AirPuff_On", "AirPuff_OnsetTime"):
            t_puff = scalar_or_nan(get_field(data, cand))
            if np.isfinite(t_puff):
                break

    isi = safe_array(get_field(states, "LED_Puff_ISI"))
    isi_dur = float(isi[1] - isi[0]) if isi.size >= 2 else scalar_or_nan(get_field(data, "ISI"))

    bt = get_field(data, "BlockType")
    if bt is None:
        bl = "short" if np.isfinite(isi_dur) and isi_dur <= short_isi_max else "long"
    else:
        bt = str(bt).strip().lower()
        if "short" in bt:
            bl = "short"
        elif "long" in bt:
            bl = "long"
        else:
            bl = "short" if np.isfinite(isi_dur) and isi_dur <= short_isi_max else "long"

    return t_led, t_led_end, t_puff, t_puff_end, isi_dur, bl == "short", bl


def estimate_open_eye_area(trials, short_isi_max=0.30):
    """95th-percentile of pre-LED eyeAreaPixels across non-probe, non-timeout trials."""
    vals = []
    for tr in trials:
        states = get_field(tr, "States")
        events = get_field(tr, "Events")
        data   = get_field(tr, "Data")
        if states is None or events is None or data is None:
            continue
        if has_field(states, "CheckEyeOpenTimeout"):
            v = safe_array(get_field(states, "CheckEyeOpenTimeout", np.nan))
            if np.any(np.isfinite(v)):
                continue
        if has_field(data, "IsProbeTrial"):
            p = np.asarray(get_field(data, "IsProbeTrial", 0)).squeeze()
            if np.any(p == 1):
                continue
        FT  = safe_array(get_field(data, "FECTimes"))
        eye = safe_array(get_field(data, "eyeAreaPixels"))
        if FT.size == 0 or eye.size == 0 or FT.size != eye.size:
            continue
        t_led = scalar_or_nan(get_field(events, "GlobalTimer1_Start"))
        if not np.isfinite(t_led):
            continue
        t_rel = FT - t_led
        idx = np.isfinite(t_rel) & np.isfinite(eye) & (t_rel >= -0.2) & (t_rel < 0.0)
        if np.any(idx):
            vals.extend(eye[idx].tolist())
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return np.nanpercentile(vals, 95) if vals.size > 0 else np.nan


# ============================================================
# Session loader
# ============================================================

def collect_session(filepath, t_grid, short_isi_max=0.30, smooth_win=5):
    """
    Load one .mat file and return a dict with per-trial records.

    Each record contains:
        is_probe     : bool
        block        : 'short' | 'long'
        fec          : 1-D array interpolated onto t_grid
        baseline_mean: mean FEC in [-0.2, 0) s  (for sanity checks)
        puff_rel     : puff onset relative to LED (NaN for probe trials)
        led_end_rel  : LED-off relative to LED onset
    """
    try:
        raw = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"  Could not load {os.path.basename(filepath)}: {e}")
        return None
    if "SessionData" not in raw:
        return None
    SD = raw["SessionData"]

    is_chemo = is_chemogenetics_session(SD)
    dm = re.search(r"(\d{8})", os.path.basename(filepath))

    trials = ensure_trial_list(get_field(get_field(SD, "RawEvents"), "Trial"))
    open_eye = estimate_open_eye_area(trials, short_isi_max)
    if not np.isfinite(open_eye):
        return None

    date_str = None
    if dm:
        try:
            date_str = datetime.strptime(dm.group(1), "%Y%m%d").strftime("%m/%d/%Y")
        except Exception:
            pass
    date_str = date_str or os.path.basename(filepath)

    bl_mask = (t_grid >= -0.2) & (t_grid < 0.0)
    records = []
    protocol_records = []

    for trial_index, tr in enumerate(trials):
        states = get_field(tr, "States")
        events = get_field(tr, "Events")
        data   = get_field(tr, "Data")
        if states is None or events is None or data is None:
            continue

        is_probe = False
        if has_field(data, "IsProbeTrial"):
            p = np.asarray(get_field(data, "IsProbeTrial", 0)).squeeze()
            is_probe = bool(np.any(p == 1))

        # Preserve the original protocol trial sequence for probe scheduling
        # analyses, even when a trial has no usable FEC/video samples.
        t_led, t_led_end, t_puff, t_puff_end, isi_dur, is_short, bl = get_trial_timing(
            tr, short_isi_max)
        if bl in ("short", "long"):
            protocol_records.append({
                "trial_index": trial_index,
                "is_probe": is_probe,
                "block": bl,
            })

        # FEC analyses continue to exclude timeout and missing-video trials.
        if has_field(states, "CheckEyeOpenTimeout"):
            v = safe_array(get_field(states, "CheckEyeOpenTimeout", np.nan))
            if np.any(np.isfinite(v)):
                continue

        FT  = safe_array(get_field(data, "FECTimes"))
        eye = safe_array(get_field(data, "eyeAreaPixels"))
        if FT.size == 0 or eye.size == 0 or FT.size != eye.size:
            continue

        if not np.isfinite(t_led):
            continue

        fec = 1.0 - eye / open_eye
        if smooth_win > 1:
            fec = smooth_trace(fec, smooth_win)
        fec_q = interp_to_grid(FT - t_led, fec, t_grid)

        records.append({
            "is_probe":      is_probe,
            "block":         bl,
            "fec":           fec_q,
            "baseline_mean": float(np.nanmean(fec_q[bl_mask])) if np.any(bl_mask) else np.nan,
            "puff_rel":      t_puff - t_led if np.isfinite(t_puff) else np.nan,
            "puff_end_rel":  t_puff_end - t_led if np.isfinite(t_puff_end) else np.nan,
            "led_end_rel":   t_led_end - t_led if np.isfinite(t_led_end) else np.nan,
        })

    return {
        "name":     date_str,
        "filepath": filepath,
        "is_chemo": is_chemo,
        "records":  records,
        "protocol_records": protocol_records,
    }


# ============================================================
# Figure 1 — probe fraction per session
# ============================================================

def plot_probe_fraction(sessions, out_dir):
    names, short_fracs, long_fracs, is_chemo_session = [], [], [], []
    for s in sessions:
        recs = s["protocol_records"]
        if not recs:
            continue
        block_fractions = {}
        for block in ("short", "long"):
            block_records = [r for r in recs if r["block"] == block]
            if block_records:
                block_fractions[block] = (
                    sum(1 for r in block_records if r["is_probe"])
                    / len(block_records)
                )
            else:
                block_fractions[block] = np.nan
        names.append(s["name"])
        short_fracs.append(block_fractions["short"])
        long_fracs.append(block_fractions["long"])
        is_chemo_session.append(s["is_chemo"])

    if not names:
        print("Fig 1: no sessions with trials.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9 + 1), 4.8))
    x = np.arange(len(names))
    width = 0.36
    short_colors = ["#B65A5A" if chemo else "#888888" for chemo in is_chemo_session]
    long_colors = ["#8B0000" if chemo else "#000000" for chemo in is_chemo_session]
    short_bars = ax.bar(
        x - width / 2, short_fracs, width,
        color=short_colors, edgecolor="none",
    )
    long_bars = ax.bar(
        x + width / 2, long_fracs, width,
        color=long_colors, edgecolor="none",
    )
    for bars, values in ((short_bars, short_fracs), (long_bars, long_fracs)):
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.003,
                    f"{value:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    for tick, chemo in zip(ax.get_xticklabels(), is_chemo_session):
        tick.set_color("#8B0000" if chemo else "black")
    ax.set_ylabel("Probe trial fraction")
    ax.set_xlabel("Session")
    finite_fracs = np.asarray(short_fracs + long_fracs, dtype=float)
    finite_fracs = finite_fracs[np.isfinite(finite_fracs)]
    ax.set_ylim(0, min(1.0, np.nanmax(finite_fracs) * 1.30))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.set_title(
        "Fraction of Probe Trials per Session and Block",
        fontsize=12,
    )
    ax.legend(
        handles=[
            Patch(facecolor="#888888", edgecolor="none", label="Short — Control"),
            Patch(facecolor="#000000", edgecolor="none", label="Long — Control"),
            Patch(facecolor="#B65A5A", edgecolor="none", label="Short — Chemo"),
            Patch(facecolor="#8B0000", edgecolor="none", label="Long — Chemo"),
        ],
        frameon=False, ncol=2, fontsize=8, loc="best",
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "Probe_01_FractionPerSession.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


# ============================================================
# Figure 2 — probe fraction aligned on block transitions
# ============================================================

def plot_probe_at_transitions(sessions, out_dir, half_win=40):
    """
    For every block transition (S->L or L->S), collect ±half_win trials.
    Position 0 = first trial of the new block.
    Plot mean ± SEM probe fraction at each relative position.
    Separate panels for S->L vs L->S; ctrl (black) and chemo (red) overlaid.
    """
    # counts[direction][group][rel_pos] = list of 0/1
    directions = ("S->L", "L->S")
    groups     = ("ctrl", "chemo")
    counts = {d: {g: defaultdict(list) for g in groups} for d in directions}
    protected_window_violations = []

    for s in sessions:
        grp  = "chemo" if s["is_chemo"] else "ctrl"
        recs = s["protocol_records"]
        n    = len(recs)
        for i in range(1, n):
            prev_bl = recs[i - 1]["block"]
            curr_bl = recs[i]["block"]
            if prev_bl == curr_bl:
                continue
            direction = "S->L" if prev_bl == "short" else "L->S"
            for j in range(max(0, i - half_win), min(n, i + half_win + 1)):
                relative_trial = j - i
                recorded_probe = int(recs[j]["is_probe"])
                if 0 <= relative_trial <= 6 and recorded_probe:
                    protected_window_violations.append(
                        (s["name"], direction, relative_trial, recs[j]["trial_index"])
                    )
                # Protocol-enforced analysis: recorded probe flags in the first
                # seven trials are QC violations and are not counted as probes.
                valid_probe = recorded_probe if not (0 <= relative_trial <= 6) else 0
                counts[direction][grp][relative_trial].append(valid_probe)

    # Keep this appended row compact so it matches one summary row visually.
    fig, axs = plt.subplots(1, 2, figsize=(13, 2.25), sharey=True)
    style = {"ctrl": ("black", "-"), "chemo": ("#D14949", "-")}

    for ax, direction in zip(axs, directions):
        ax.axvspan(-0.5, 6.5, color="#F2E8B6", alpha=0.45,
                   label="No-probe protocol window (trials 0–6)")
        for grp in groups:
            c = counts[direction][grp]
            if not c:
                continue
            positions = sorted(c.keys())
            means = [np.mean(c[p])  for p in positions]
            sems  = [np.std(c[p]) / np.sqrt(len(c[p])) for p in positions]
            col, ls = style[grp]
            ax.fill_between(positions,
                            np.array(means) - np.array(sems),
                            np.array(means) + np.array(sems),
                            color=col, alpha=0.15)
            ax.plot(positions, means, color=col, lw=2, ls=ls,
                    label=grp.capitalize())
        ax.axvline(-40, color="#666666", ls="--", lw=1.0)
        ax.axvline(0, color="k", ls="--", lw=1.2)
        ax.axvline(40, color="#666666", ls="--", lw=1.0)
        ax.set_xlim(-40.5, 40.5)
        ax.set_xticks(np.arange(-40, 41, 5))
        ax.set_xlabel("Aligned trial # (0 = first trial of new block)", fontsize=8)
        ax.set_ylabel("Probe trial fraction", fontsize=8)
        ax.set_title(f"Block transition: {direction}", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", labelsize=8)
        ax.legend(frameon=False, fontsize=7)

    plt.tight_layout()
    out = os.path.join(out_dir, "Probe_02_FractionAtTransitions.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    if protected_window_violations:
        print("WARNING: recorded probe flags excluded within protected trials 0–6:")
        for session, direction, relative_trial, trial_index in protected_window_violations:
            print(
                f"  {session} {direction}: relative trial {relative_trial}, "
                f"raw trial index {trial_index}"
            )
    else:
        print("Verified: no recorded probes in the first 7 trials (0–6) after transitions.")
    plt.show()


# ============================================================
# Figure 3 — average FEC: probe−1, probe, probe+1
# ============================================================

def _probe_triplet_baseline(triplet, t_grid):
    """Use the central probe trial baseline as the matching value for a triplet."""
    bl = (t_grid >= -0.2) & (t_grid < 0.0)
    if not np.any(bl):
        return np.nan
    vals = np.asarray(triplet["probe"], dtype=float)[bl]
    return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan


def match_probe_triplet_baselines(ctrl_triplets, chemo_triplets, t_grid, n_bins=10, rng_seed=42):
    """Subsample whole probe triplets so control/chemo probe baselines match."""
    if len(ctrl_triplets) == 0 or len(chemo_triplets) == 0:
        return list(ctrl_triplets), list(chemo_triplets)

    b_c = np.asarray([_probe_triplet_baseline(tr, t_grid) for tr in ctrl_triplets], dtype=float)
    b_h = np.asarray([_probe_triplet_baseline(tr, t_grid) for tr in chemo_triplets], dtype=float)

    valid_c = np.where(np.isfinite(b_c))[0]
    valid_h = np.where(np.isfinite(b_h))[0]
    if len(valid_c) == 0 or len(valid_h) == 0:
        return list(ctrl_triplets), list(chemo_triplets)

    lo = max(np.nanmin(b_c[valid_c]), np.nanmin(b_h[valid_h]))
    hi = min(np.nanmax(b_c[valid_c]), np.nanmax(b_h[valid_h]))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return list(ctrl_triplets), list(chemo_triplets)

    edges = np.linspace(lo, hi, n_bins + 1)
    rng = np.random.default_rng(rng_seed)
    sel_c, sel_h = [], []
    for i in range(n_bins):
        if i == n_bins - 1:
            ic = np.where((b_c >= edges[i]) & (b_c <= edges[i + 1]))[0]
            ih = np.where((b_h >= edges[i]) & (b_h <= edges[i + 1]))[0]
        else:
            ic = np.where((b_c >= edges[i]) & (b_c < edges[i + 1]))[0]
            ih = np.where((b_h >= edges[i]) & (b_h < edges[i + 1]))[0]
        n = min(len(ic), len(ih))
        if n == 0:
            continue
        sel_c.extend(rng.choice(ic, n, replace=False).tolist())
        sel_h.extend(rng.choice(ih, n, replace=False).tolist())

    if len(sel_c) == 0 or len(sel_h) == 0:
        return list(ctrl_triplets), list(chemo_triplets)

    return [ctrl_triplets[i] for i in sel_c], [chemo_triplets[i] for i in sel_h]


def align_triplet_trace_baselines(triplets, t_grid):
    """Offset each trace so all Probe-1/Probe/Probe+1 baseline means coincide."""
    if not triplets:
        return []
    bl = (t_grid >= -0.2) & (t_grid < 0.0)
    baseline_means = []
    for triplet in triplets:
        for pos in ("prev", "probe", "next"):
            value = np.nanmean(np.asarray(triplet[pos], dtype=float)[bl])
            if np.isfinite(value):
                baseline_means.append(value)
    if not baseline_means:
        return list(triplets)

    common_baseline = float(np.nanmean(baseline_means))
    aligned = []
    for triplet in triplets:
        aligned_triplet = {}
        for pos in ("prev", "probe", "next"):
            trace = np.asarray(triplet[pos], dtype=float).copy()
            trace_baseline = np.nanmean(trace[bl])
            if np.isfinite(trace_baseline):
                trace += common_baseline - trace_baseline
            aligned_triplet[pos] = trace
        aligned.append(aligned_triplet)
    return aligned


def add_probe_event_windows(ax, block):
    led_start, led_end = 0.0, 0.05
    puff_start = 0.2 if block == "short" else 0.4
    puff_end = puff_start + 0.02
    puff_color = "dodgerblue" if block == "short" else "limegreen"
    puff_text_color = "dodgerblue" if block == "short" else "green"

    ax.axvspan(led_start, led_end, color="gray", alpha=0.18, linewidth=0)
    ax.axvspan(puff_start, puff_end, color=puff_color, alpha=0.25, linewidth=0)

    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.04 * (ymax - ymin)
    ax.text(led_start + 0.002, y_text, "LED",
            color="black", va="top", ha="left", fontsize=9)
    ax.text(puff_start + 0.002, y_text, "AirPuff",
            color=puff_text_color, va="top", ha="left", fontsize=9)


def plot_probe_fec_triplets(sessions, t_grid, out_dir, baseline_matched=False):
    """
    For each probe trial at position i, collect FEC of trials i-1, i, i+1.
    Average across all triplets; plot mean ± SEM.
    2×2 panels: (short / long) × (control / chemo).
    """
    keys = [("short", "ctrl"), ("short", "chemo"),
            ("long",  "ctrl"), ("long",  "chemo")]
    triplets = {k: [] for k in keys}

    for s in sessions:
        grp  = "chemo" if s["is_chemo"] else "ctrl"
        recs = s["records"]
        n    = len(recs)
        for i in range(1, n - 1):
            if not recs[i]["is_probe"]:
                continue
            bl  = recs[i]["block"]
            key = (bl, grp)
            if key not in triplets:
                continue
            fp = recs[i - 1]["fec"]
            fc = recs[i    ]["fec"]
            fn = recs[i + 1]["fec"]
            if (np.any(np.isfinite(fp)) and
                    np.any(np.isfinite(fc)) and
                    np.any(np.isfinite(fn))):
                triplets[key].append({"prev": fp, "probe": fc, "next": fn})

    if baseline_matched:
        for block in ("short", "long"):
            ctrl_key = (block, "ctrl")
            chemo_key = (block, "chemo")
            ctrl_matched, chemo_matched = match_probe_triplet_baselines(
                triplets[ctrl_key],
                triplets[chemo_key],
                t_grid,
                rng_seed=42 + (0 if block == "short" else 10),
            )
            triplets[ctrl_key] = ctrl_matched
            triplets[chemo_key] = chemo_matched

        # After matching Control/Chemo baseline distributions, align the three
        # epoch traces within every panel to one common pre-LED baseline level.
        # Post-LED differences are therefore changes relative to the same start.
        for key in keys:
            triplets[key] = align_triplet_trace_baselines(triplets[key], t_grid)

    def _avg(triplet_list, pos):
        if not triplet_list:
            return np.full_like(t_grid, np.nan), np.full_like(t_grid, np.nan), 0
        lst = [tr[pos] for tr in triplet_list]
        M = np.vstack(lst)
        m = np.nanmean(M, axis=0)
        s = np.nanstd(M,  axis=0) / np.sqrt(M.shape[0])
        return m, s, M.shape[0]

    ctrl_styles = {
        "prev":  ("0.55",       "--", "Probe-1"),
        "probe": ("black",      "-",  "Probe"),
        "next":  ("#3B7DBA",    "--", "Probe+1"),
    }
    chemo_styles = {
        "prev":  ("0.55",       "--", "Probe-1"),
        "probe": ("#D14949",    "-",  "Probe"),
        "next":  ("#3B7DBA",    "--", "Probe+1"),
    }
    panel_titles = {
        ("short", "ctrl"):  "Short block – Control",
        ("short", "chemo"): "Short block – Chemo",
        ("long",  "ctrl"):  "Long block – Control",
        ("long",  "chemo"): "Long block – Chemo",
    }

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    panel_map = {
        ("short", "ctrl"):  axs[0, 0],
        ("long",  "ctrl"):  axs[0, 1],
        ("short", "chemo"): axs[1, 0],
        ("long",  "chemo"): axs[1, 1],
    }

    for key, ax in panel_map.items():
        any_data = False
        block, grp = key
        styles = chemo_styles if grp == "chemo" else ctrl_styles
        for pos in ("prev", "probe", "next"):
            m, se, n = _avg(triplets[key], pos)
            if n == 0:
                continue
            any_data = True
            col, ls, lbl = styles[pos]
            ax.plot(t_grid, m, color=col, lw=1.8, ls=ls, label=f"{lbl} (n={n})")
            ax.fill_between(t_grid, m - se, m + se, color=col, alpha=0.15)
        add_probe_event_windows(ax, block)
        ax.axvline(0, color="k", ls=":", lw=1)
        ax.set_title(panel_titles[key], fontsize=12)
        ax.set_xlabel("Time from LED onset (s)")
        ax.set_ylabel("FEC")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")
        if any_data:
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(0.5, 0.5, "No probe triplets", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=10)

    label = "Baseline-matched" if baseline_matched else "Raw"
    fig.suptitle(f"{label} Average FEC: Probe-1, Probe, Probe+1", fontsize=14)
    plt.tight_layout()
    suffix = "BaselineMatched" if baseline_matched else "Raw"
    out = os.path.join(out_dir, f"Probe_03_FEC_Triplets_{suffix}.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    if not baseline_matched:
        legacy_out = os.path.join(out_dir, "Probe_03_FEC_Triplets.pdf")
        plt.savefig(legacy_out, bbox_inches="tight")
        print(f"Saved: {legacy_out}")
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_files = sorted(glob.glob(os.path.join(script_dir, "*_EBC_*.mat")))
    if not data_files:
        print(f"No *_EBC_*.mat files found in {script_dir}")
        return

    t_pre, t_post, dt = 0.2, 0.6, 1 / 250
    t_grid = np.arange(-t_pre, t_post + dt / 2, dt)

    sessions = []
    n_no_probes = 0
    for fp in data_files:
        print(f"Loading: {os.path.basename(fp)}")
        s = collect_session(fp, t_grid)
        if s is None:
            print("  Skipped (no open-eye estimate or load error)")
            continue
        n_total = len(s["records"])
        n_probe = sum(1 for r in s["records"] if r["is_probe"])
        print(f"  {n_total} trials, {n_probe} probe trials"
              f"  ({'Chemo' if s['is_chemo'] else 'Control'})")
        if n_probe == 0:
            n_no_probes += 1
        sessions.append(s)

    if not sessions:
        print("No sessions loaded.")
        return

    n_ctrl  = sum(1 for s in sessions if not s["is_chemo"])
    n_chemo = sum(1 for s in sessions if s["is_chemo"])
    print(f"\nLoaded {len(sessions)} sessions "
          f"(Control={n_ctrl}, Chemo={n_chemo}); "
          f"{n_no_probes} session(s) had 0 probe trials.\n")

    plot_probe_fraction(sessions, script_dir)
    plot_probe_at_transitions(sessions, script_dir)
    plot_probe_fec_triplets(sessions, t_grid, script_dir, baseline_matched=False)
    plot_probe_fec_triplets(sessions, t_grid, script_dir, baseline_matched=True)


if __name__ == "__main__":
    main()
