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

CHEMO_DATES = {"05/21", "05/27", "05/29", "06/02", "06/04"}


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

    # chemo flag
    is_chemo = False
    try:
        is_chemo = int(np.asarray(get_field(SD, "Chemogenetics", 0)).squeeze()) == 1
    except Exception:
        pass
    dm = re.search(r"(\d{8})", os.path.basename(filepath))
    if dm:
        try:
            if datetime.strptime(dm.group(1), "%Y%m%d").strftime("%m/%d") in CHEMO_DATES:
                is_chemo = True
        except Exception:
            pass

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

    for tr in trials:
        states = get_field(tr, "States")
        events = get_field(tr, "Events")
        data   = get_field(tr, "Data")
        if states is None or events is None or data is None:
            continue

        # skip timeouts
        if has_field(states, "CheckEyeOpenTimeout"):
            v = safe_array(get_field(states, "CheckEyeOpenTimeout", np.nan))
            if np.any(np.isfinite(v)):
                continue

        is_probe = False
        if has_field(data, "IsProbeTrial"):
            p = np.asarray(get_field(data, "IsProbeTrial", 0)).squeeze()
            is_probe = bool(np.any(p == 1))

        FT  = safe_array(get_field(data, "FECTimes"))
        eye = safe_array(get_field(data, "eyeAreaPixels"))
        if FT.size == 0 or eye.size == 0 or FT.size != eye.size:
            continue

        t_led, t_led_end, t_puff, t_puff_end, isi_dur, is_short, bl = get_trial_timing(
            tr, short_isi_max)
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
            "led_end_rel":   t_led_end - t_led if np.isfinite(t_led_end) else np.nan,
        })

    return {
        "name":     date_str,
        "filepath": filepath,
        "is_chemo": is_chemo,
        "records":  records,
    }


# ============================================================
# Figure 1 — probe fraction per session
# ============================================================

def plot_probe_fraction(sessions, out_dir):
    names, fracs, colors = [], [], []
    for s in sessions:
        recs = s["records"]
        n_total = len(recs)
        if n_total == 0:
            continue
        n_probe = sum(1 for r in recs if r["is_probe"])
        names.append(s["name"])
        fracs.append(n_probe / n_total)
        colors.append("steelblue" if s["is_chemo"] else "black")

    if not names:
        print("Fig 1: no sessions with trials.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.7 + 1), 4))
    x = np.arange(len(names))
    ax.bar(x, fracs, color=colors, edgecolor="none")
    for xi, fi in zip(x, fracs):
        ax.text(xi, fi + 0.003, f"{fi:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    for tick, col in zip(ax.get_xticklabels(), colors):
        tick.set_color(col)
    ax.set_ylabel("Probe trial fraction")
    ax.set_xlabel("Session")
    ax.set_ylim(0, min(1.0, max(fracs) * 1.25))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.set_title("Fraction of Probe Trials per Session\n(black = Control, blue = Chemo)",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "Probe_01_FractionPerSession.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


# ============================================================
# Figure 2 — probe fraction aligned on block transitions
# ============================================================

def plot_probe_at_transitions(sessions, out_dir, half_win=15):
    """
    For every block transition (S->L or L->S), collect ±half_win trials.
    Position 0 = first trial of the new block.
    Plot mean ± SEM probe fraction at each relative position.
    Separate panels for S->L vs L->S; ctrl (black) and chemo (blue) overlaid.
    """
    # counts[direction][group][rel_pos] = list of 0/1
    directions = ("S->L", "L->S")
    groups     = ("ctrl", "chemo")
    counts = {d: {g: defaultdict(list) for g in groups} for d in directions}

    for s in sessions:
        grp  = "chemo" if s["is_chemo"] else "ctrl"
        recs = s["records"]
        n    = len(recs)
        for i in range(1, n):
            prev_bl = recs[i - 1]["block"]
            curr_bl = recs[i]["block"]
            if prev_bl == curr_bl:
                continue
            direction = "S->L" if prev_bl == "short" else "L->S"
            for j in range(max(0, i - half_win), min(n, i + half_win)):
                counts[direction][grp][j - i].append(int(recs[j]["is_probe"]))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    style = {"ctrl": ("black", "-"), "chemo": ("steelblue", "-")}

    for ax, direction in zip(axs, directions):
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
        ax.axvline(0, color="k", ls="--", lw=1.2)
        ax.set_xlabel("Trials relative to block transition\n(0 = first trial of new block)")
        ax.set_ylabel("Probe trial fraction")
        ax.set_title(f"Block transition: {direction}", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Probe Trial Fraction Around Block Transitions", fontsize=13)
    plt.tight_layout()
    out = os.path.join(out_dir, "Probe_02_FractionAtTransitions.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


# ============================================================
# Figure 3 — average FEC: probe−1, probe, probe+1
# ============================================================

def plot_probe_fec_triplets(sessions, t_grid, out_dir):
    """
    For each probe trial at position i, collect FEC of trials i-1, i, i+1.
    Average across all triplets; plot mean ± SEM.
    2×2 panels: (short / long) × (control / chemo).
    """
    keys = [("short", "ctrl"), ("short", "chemo"),
            ("long",  "ctrl"), ("long",  "chemo")]
    triplets = {k: {"prev": [], "probe": [], "next": []} for k in keys}

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
                triplets[key]["prev"].append(fp)
                triplets[key]["probe"].append(fc)
                triplets[key]["next"].append(fn)

    def _avg(lst):
        if not lst:
            return np.full_like(t_grid, np.nan), np.full_like(t_grid, np.nan), 0
        M = np.vstack(lst)
        m = np.nanmean(M, axis=0)
        s = np.nanstd(M,  axis=0) / np.sqrt(M.shape[0])
        return m, s, M.shape[0]

    pos_styles = {
        "prev":  ("gray",       "--", "Probe−1"),
        "probe": ("black",      "-",  "Probe"),
        "next":  ("steelblue",  "--", "Probe+1"),
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
        ("short", "chemo"): axs[0, 1],
        ("long",  "ctrl"):  axs[1, 0],
        ("long",  "chemo"): axs[1, 1],
    }

    for key, ax in panel_map.items():
        any_data = False
        for pos in ("prev", "probe", "next"):
            m, se, n = _avg(triplets[key][pos])
            if n == 0:
                continue
            any_data = True
            col, ls, lbl = pos_styles[pos]
            ax.plot(t_grid, m, color=col, lw=1.8, ls=ls, label=f"{lbl} (n={n})")
            ax.fill_between(t_grid, m - se, m + se, color=col, alpha=0.15)
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

    fig.suptitle("Average FEC: Probe−1, Probe, Probe+1", fontsize=14)
    plt.tight_layout()
    out = os.path.join(out_dir, "Probe_03_FEC_Triplets.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
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
    plot_probe_fec_triplets(sessions, t_grid, script_dir)


if __name__ == "__main__":
    main()
