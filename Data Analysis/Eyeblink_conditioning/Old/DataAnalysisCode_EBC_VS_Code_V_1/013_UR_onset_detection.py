"""
013_UR_onset_detection.py

Detects Unconditioned Response (UR) onset in EBC FEC traces.

Algorithm (per trial):
  1. Search the post-airpuff window (t_puff → t_puff + 150 ms).
  2. Find the first local FEC peak above a minimum height.
  3. Find the first sustained 8-sample velocity run above the baseline
     velocity threshold before that peak  →  UR onset.
  4. CR window end  =  UR onset − 50 ms  (buffer to exclude any UR).

Outputs (saved to current folder):
  • UR_onset_distribution_<date>.pdf   – latency histograms by block
  • UR_onset_average_traces_<date>.pdf – mean FEC ± median UR onset
  • UR_onset_summary_table_<date>.pdf  – median / IQR table
  • UR_onset_QC_<YYYYMMDD>.pdf        – per-session trial-by-trial QC

Usage: cd to the folder containing *_EBC_*.mat files, then run.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.use14corefonts"] = False

ONSET_SCRIPT = "010_CR_onsetTiming_Distribution_SanityCheck.py"
CR_BUFFER_S = 0.050   # 50 ms safety buffer before UR onset


# ── load shared utilities from 010 ───────────────────────────────────────────

def load_onset_module():
    spec = importlib.util.spec_from_file_location("cr_onset_010", ONSET_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {ONSET_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── UR onset detector ─────────────────────────────────────────────────────────

def find_ur_onset_velocity_based(
    time,
    signal,
    t_puff,
    t_led=0.0,
    baseline_window=(-0.2, 0.0),
    velocity_smooth_window=3,
    velocity_std_factor=1.5,
    velocity_floor=0.02,
    min_sustain_samples=8,
    min_signal_rise=0.004,
    search_post_puff_s=0.150,
    min_ur_latency_s=0.005,
    min_peak_height=0.02,
):
    """
    Detect UR onset from the FEC velocity trace after the airpuff.

    Mirrors the CR onset algorithm but applied to the post-puff window.
    The onset is the first timepoint of the first 8-sample velocity run
    above the baseline velocity threshold before the first post-puff peak.

    Returns
    -------
    ur_onset_time       : float  – time from LED (s), NaN if not detected
    ur_onset_latency    : float  – latency from puff onset (s)
    baseline_amp        : float  – mean FEC in baseline window
    ur_peak_val         : float  – baseline-subtracted FEC at UR peak
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    valid = np.isfinite(time) & np.isfinite(signal)
    time = time[valid]
    signal = signal[valid]
    if len(time) < 2:
        return np.nan, np.nan, np.nan, np.nan

    # Baseline amplitude and velocity threshold from the pre-LED window
    baseline_idx = (time >= (t_led + baseline_window[0])) & (time <= (t_led + baseline_window[1]))
    if not np.any(baseline_idx):
        return np.nan, np.nan, np.nan, np.nan

    baseline_amp = np.nanmean(signal[baseline_idx])
    signal_bs = signal - baseline_amp

    sig_smooth = uniform_filter1d(signal_bs, size=velocity_smooth_window, mode="nearest")
    velocity = np.gradient(sig_smooth, time)

    baseline_vel = velocity[baseline_idx]
    baseline_vel = baseline_vel[np.isfinite(baseline_vel)]
    if baseline_vel.size > 0:
        med = np.nanmedian(baseline_vel)
        mad = np.nanmedian(np.abs(baseline_vel - med))
        robust_std = 1.4826 * mad if np.isfinite(mad) else np.nan
        if not np.isfinite(robust_std) or robust_std == 0:
            robust_std = np.nanstd(baseline_vel)
        velocity_threshold = max(med + velocity_std_factor * robust_std, velocity_floor)
    else:
        velocity_threshold = velocity_floor

    # Post-puff search window
    search_idx = (time >= t_puff) & (time <= t_puff + search_post_puff_s)
    if not np.any(search_idx):
        return np.nan, np.nan, baseline_amp, np.nan

    time_search = time[search_idx]
    sig_search = signal_bs[search_idx]
    sig_smooth_search = sig_smooth[search_idx]
    vel_search = velocity[search_idx]

    if len(time_search) < 2 or not np.any(np.isfinite(sig_search)):
        return np.nan, np.nan, baseline_amp, np.nan

    # First local peak above min_peak_height (handles double-peak UR correctly)
    local_peaks, _ = find_peaks(sig_search, height=min_peak_height)
    if len(local_peaks) > 0:
        peak_idx_local = int(local_peaks[0])
    else:
        peak_idx_local = int(np.nanargmax(sig_search))
    ur_peak_val = sig_search[peak_idx_local]

    if not np.isfinite(ur_peak_val) or ur_peak_val < min_peak_height:
        return np.nan, np.nan, baseline_amp, ur_peak_val

    # Velocity run scan up to the peak
    pre_peak_time = time_search[:peak_idx_local + 1]
    pre_peak_vel = vel_search[:peak_idx_local + 1]
    pre_peak_sig = sig_smooth_search[:peak_idx_local + 1]

    above = (
        np.isfinite(pre_peak_vel)
        & np.isfinite(pre_peak_sig)
        & (pre_peak_vel >= velocity_threshold)
        & (pre_peak_sig >= min_signal_rise)
    )

    idx_candidates = []
    run_start = None
    for idx, is_above in enumerate(above):
        if is_above and run_start is None:
            run_start = idx
        elif (not is_above) and run_start is not None:
            if idx - run_start >= min_sustain_samples:
                idx_candidates.append(run_start)
            run_start = None
    if run_start is not None and len(above) - run_start >= min_sustain_samples:
        idx_candidates.append(run_start)

    if len(idx_candidates) == 0:
        return np.nan, np.nan, baseline_amp, ur_peak_val

    onset_idx = int(idx_candidates[0])
    ur_onset_time = pre_peak_time[onset_idx]
    ur_onset_latency = ur_onset_time - t_puff

    if ur_onset_latency < min_ur_latency_s:
        return np.nan, np.nan, baseline_amp, ur_peak_val

    return ur_onset_time, ur_onset_latency, baseline_amp, ur_peak_val


# ── record preparation ────────────────────────────────────────────────────────

def prepare_ur_records(data_files, onset_module):
    """
    Build per-trial records with both CR and UR onset information.
    Reuses prepare_trial_records from the 010 module for CR classification.
    """
    records = onset_module.prepare_trial_records(data_files)
    for r in records:
        ur_onset_time, ur_onset_latency, _, ur_peak_val = find_ur_onset_velocity_based(
            time=r["t_grid"],
            signal=r["Fq"],
            t_puff=r["t_puff"],
            t_led=0.0,
        )
        r["ur_onset_time"] = ur_onset_time
        r["ur_onset_latency"] = ur_onset_latency
        r["ur_peak_val"] = ur_peak_val
        # Safe CR window end: 50 ms before detected UR onset
        r["cr_window_end"] = (ur_onset_time - CR_BUFFER_S) if np.isfinite(ur_onset_time) else np.nan
    return records


def group_by_session(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[r["filepath"]].append(r)
    for recs in grouped.values():
        recs.sort(key=lambda r: r.get("trial_index", 0))
    return dict(sorted(grouped.items()))


def session_token(filepath):
    m = re.search(r"\d{8}", os.path.basename(filepath))
    return m.group() if m else os.path.splitext(os.path.basename(filepath))[0]


def build_session_range(data_files):
    dates = []
    for path in data_files:
        m = re.search(r"\d{8}", os.path.basename(path))
        if m:
            try:
                dates.append(datetime.strptime(m.group(), "%Y%m%d"))
            except Exception:
                pass
    if not dates:
        return "Unknown", "Unknown", "Unknown", "Unknown"
    first, last = min(dates), max(dates)
    return (first.strftime("%Y%m%d"), last.strftime("%Y%m%d"),
            first.strftime("%m/%d/%Y"), last.strftime("%m/%d/%Y"))


# ── pool-level plots ──────────────────────────────────────────────────────────

def plot_ur_onset_distribution(records, first_date, last_date, first_tok, last_tok):
    edges = np.arange(0.0, 0.1551, 0.005)
    centers = edges[:-1] + np.diff(edges) / 2
    bar_w = np.diff(edges)[0] * 0.85

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, block in zip(axs, ["short", "long"]):
        vals = [
            r["ur_onset_latency"] for r in records
            if r["block_label"] == block and np.isfinite(r.get("ur_onset_latency", np.nan))
        ]
        n = len(vals)
        if n > 0:
            counts, _ = np.histogram(vals, bins=edges)
            frac = counts.astype(float) / n
            med = float(np.nanmedian(vals))
            ax.bar(centers, frac, width=bar_w, color="steelblue", alpha=0.75, label=f"n = {n}")
            ax.axvline(med, color="red", linestyle="--", linewidth=1.8,
                       label=f"Median  {med * 1000:.0f} ms")
            ax.legend(fontsize=9)
        ax.set_title(f"{block.title()} block — UR onset latency from puff", fontsize=11)
        ax.set_xlabel("UR onset latency from puff (s)")
        ax.set_ylabel("Fraction of trials")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"UR onset latency distributions\nSessions: {first_date} — {last_date}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_name = f"UR_onset_distribution_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}")
    plt.close(fig)


def plot_ur_average_traces(records, first_tok, last_tok):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    cat_colors = {"Good CR": "black", "Poor CR": "dimgray", "No CR": "lightgray"}

    for col, block_label in enumerate(["short", "long"]):
        ax = axs[col]
        puff_x = 0.200 if block_label == "short" else 0.400

        # Shading
        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.28, zorder=0)
        ax.axvspan(puff_x, puff_x + 0.020, color="orange", alpha=0.28, zorder=0, label="Airpuff (20 ms)")
        ax.axvline(puff_x, color="orange", linewidth=1.2, zorder=1)

        # Average trace per CR category
        for category, color in cat_colors.items():
            subset = [r for r in records if r["block_label"] == block_label and r["category"] == category]
            if not subset:
                continue
            all_traces = np.vstack([r["Fq"] for r in subset])
            mean_trace = np.nanmean(all_traces, axis=0)
            ax.plot(subset[0]["t_grid"], mean_trace, color=color, linewidth=1.6,
                    label=f"{category} (n={len(subset)})", zorder=2)

        # Median UR onset (all trials)
        ur_lats = [r["ur_onset_latency"] for r in records
                   if r["block_label"] == block_label and np.isfinite(r.get("ur_onset_latency", np.nan))]
        if ur_lats:
            med_ur = float(np.nanmedian(ur_lats))
            ax.axvline(puff_x + med_ur, color="red", linestyle="--", linewidth=1.8,
                       label=f"Median UR onset  {med_ur * 1000:.0f} ms post-puff", zorder=3)
            ax.axvline(puff_x + med_ur - CR_BUFFER_S, color="green", linestyle=":",
                       linewidth=1.5, label=f"CR window end  (−{CR_BUFFER_S*1000:.0f} ms buffer)",
                       zorder=3)

        ax.set_title(f"{block_label.title()} block — avg FEC by category")
        ax.set_xlim(-0.1, 0.65)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time from LED onset (s)")
        ax.set_ylabel("FEC")
        ax.legend(loc="upper left", fontsize=7, ncol=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Average FEC traces with detected UR onset and safe CR window end", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_name = f"UR_onset_average_traces_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}")
    plt.close(fig)


def plot_ur_summary_table(records, first_date, last_date, first_tok, last_tok):
    rows = []
    for block_label in ["short", "long"]:
        puff_x = 0.200 if block_label == "short" else 0.400
        ur_vals = [
            r["ur_onset_latency"] for r in records
            if r["block_label"] == block_label and np.isfinite(r.get("ur_onset_latency", np.nan))
        ]
        cr_end_vals = [
            r["cr_window_end"] for r in records
            if r["block_label"] == block_label and np.isfinite(r.get("cr_window_end", np.nan))
        ]
        n = len(ur_vals)
        med = np.nanmedian(ur_vals) if n else np.nan
        p25 = np.nanpercentile(ur_vals, 25) if n else np.nan
        p75 = np.nanpercentile(ur_vals, 75) if n else np.nan
        med_cr_end = np.nanmedian(cr_end_vals) if cr_end_vals else np.nan

        def ms(v):
            return f"{v * 1000:.1f} ms" if np.isfinite(v) else "NaN"

        rows.append([
            block_label.title(),
            str(n),
            ms(med),
            ms(p25),
            ms(p75),
            f"{med_cr_end:.3f} s" if np.isfinite(med_cr_end) else "NaN",
        ])

    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Block", "Trials w/ UR", "Median UR latency",
                   "25th pctile", "75th pctile", f"Median CR win end (UR − {CR_BUFFER_S*1000:.0f} ms)"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    fig.suptitle(
        f"UR onset latency summary (from puff)\nSessions: {first_date} — {last_date}",
        fontsize=13,
    )
    out_name = f"UR_onset_summary_table_{first_tok}_to_{last_tok}.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"Saved {out_name}")
    plt.close(fig)


# ── per-session QC PDFs ───────────────────────────────────────────────────────

def _add_session_summary_page(pdf, session_name, records):
    fig, axs = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    colors = {"Good CR": "black", "Poor CR": "tab:orange", "No CR": "lightgray"}
    markers = {"short": "o", "long": "s"}
    trial_indices = [r["trial_index"] for r in records]
    block_labels = [r["block_label"] for r in records]

    # Top panel: UR onset latency per trial
    ax = axs[0]
    for category, color in colors.items():
        for block, marker in markers.items():
            subset = [
                r for r in records
                if r["category"] == category and r["block_label"] == block
                and np.isfinite(r.get("ur_onset_latency", np.nan))
            ]
            if not subset:
                continue
            ax.scatter(
                [r["trial_index"] for r in subset],
                [r["ur_onset_latency"] for r in subset],
                s=22, color=color, marker=marker,
                label=f"{block.title()} {category}", alpha=0.85,
            )
    for i in range(1, len(records)):
        if block_labels[i] != block_labels[i - 1]:
            x = 0.5 * (trial_indices[i] + trial_indices[i - 1])
            ax.axvline(x, color="red", linestyle="--", linewidth=1.0)
            ax.text(x, 0.145, "block\ntransition", rotation=90,
                    va="top", ha="right", fontsize=7, color="red")
    ax.set_ylabel("UR onset latency from puff (s)")
    ax.set_ylim(-0.005, 0.155)
    ax.set_title(f"{session_name}\nTrial-by-trial UR onset latency from airpuff")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bottom panel: CR classification
    ax = axs[1]
    y_map = {"No CR": 0, "Poor CR": 1, "Good CR": 2}
    for record in records:
        ax.scatter(
            record["trial_index"],
            y_map.get(record["category"], np.nan),
            color=colors.get(record["category"], "gray"),
            marker=markers.get(record["block_label"], "o"),
            s=24,
        )
    for i in range(1, len(records)):
        if block_labels[i] != block_labels[i - 1]:
            ax.axvline(0.5 * (trial_indices[i] + trial_indices[i - 1]),
                       color="red", linestyle="--", linewidth=1.0)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["No CR", "Poor CR", "Good CR"])
    ax.set_xlabel("Trial number")
    ax.set_ylabel("CR classification")
    ax.set_ylim(-0.5, 2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_trial_page(pdf, session_name, records, page_index):
    fig, axs = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    axs = axs.ravel()

    for ax, record in zip(axs, records):
        time = record["t_grid"]
        signal = record["Fq"]
        puff_time = record["t_puff"]
        cr_onset = record.get("onset_time", np.nan)
        ur_onset = record.get("ur_onset_time", np.nan)
        cr_win_end = record.get("cr_window_end", np.nan)

        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.22, zorder=0)
        ax.axvspan(puff_time, puff_time + 0.020, color="orange", alpha=0.25, zorder=0)
        ax.axvline(puff_time, color="orange", linewidth=1.0, zorder=1)
        ax.plot(time, signal, color="black", linewidth=1.1, zorder=2, label="FEC")

        if np.isfinite(cr_onset):
            ax.axvline(cr_onset, color="royalblue", linestyle="--", linewidth=1.2,
                       zorder=3, label="CR onset")
        if np.isfinite(ur_onset):
            ax.axvline(ur_onset, color="red", linestyle="--", linewidth=1.3,
                       zorder=4, label="UR onset")
        if np.isfinite(cr_win_end):
            ax.axvline(cr_win_end, color="green", linestyle=":", linewidth=1.1,
                       zorder=3, label=f"CR win end")

        ur_lat = record.get("ur_onset_latency", np.nan)
        ur_label = f"UR {ur_lat * 1000:.0f} ms" if np.isfinite(ur_lat) else "UR not det."
        ax.set_title(
            f"Trial {record['trial_index']} | {record['block_label']} | "
            f"{record['category']} | {ur_label}",
            fontsize=8,
        )
        ax.set_xlim(-0.05, 0.65)
        ax.set_ylim(-0.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axs[len(records):]:
        ax.axis("off")
    for ax in axs[-2:]:
        ax.set_xlabel("Time from LED onset (s)")
    for ax in axs[::2]:
        ax.set_ylabel("FEC")

    if len(records) > 0:
        axs[0].legend(loc="upper left", fontsize=6, ncol=2)

    fig.suptitle(f"{session_name} | UR onset QC | page {page_index}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_session_pdf(filepath, records):
    token = session_token(filepath)
    session_name = os.path.basename(filepath)
    out_name = f"UR_onset_QC_{token}.pdf"
    with PdfPages(out_name) as pdf:
        _add_session_summary_page(pdf, session_name, records)
        for page_index, start in enumerate(range(0, len(records), 6), start=2):
            _plot_trial_page(pdf, session_name, records[start:start + 6], page_index)
    n_ur = sum(1 for r in records if np.isfinite(r.get("ur_onset_latency", np.nan)))
    print(f"Saved {out_name}  ({len(records)} trials, UR detected in {n_ur})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    onset_module = load_onset_module()
    data_files = sorted(glob.glob("*_EBC_*.mat"))
    if not data_files:
        print("No *_EBC_*.mat files found in current folder.")
        return

    first_tok, last_tok, first_date, last_date = build_session_range(data_files)
    print(f"Processing {len(data_files)} session(s)  [{first_date} — {last_date}]")

    records = prepare_ur_records(data_files, onset_module)
    if not records:
        print("No usable trials found.")
        return

    n_total = len(records)
    n_ur = sum(1 for r in records if np.isfinite(r.get("ur_onset_latency", np.nan)))
    print(f"Trials: {n_total}  |  UR detected: {n_ur}  ({100 * n_ur / n_total:.1f}%)")

    # Pool-level outputs
    plot_ur_onset_distribution(records, first_date, last_date, first_tok, last_tok)
    plot_ur_average_traces(records, first_tok, last_tok)
    plot_ur_summary_table(records, first_date, last_date, first_tok, last_tok)

    # Per-session QC PDFs
    for filepath, session_records in group_by_session(records).items():
        write_session_pdf(filepath, session_records)

    plt.close("all")


if __name__ == "__main__":
    main()
