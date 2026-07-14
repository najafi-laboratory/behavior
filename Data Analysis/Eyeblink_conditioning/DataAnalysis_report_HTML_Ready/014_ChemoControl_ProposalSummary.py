#!/usr/bin/env python3
"""Create a proposal-ready summary of control versus chemogenetic EBC results.

The figure deliberately uses the session as the statistical unit. Trial traces are
first averaged within each session, and uncertainty/statistical tests are then
computed across sessions. This avoids treating trials from the same session as
independent observations.

Outputs
-------
ChemoControl_ProposalSummary_<date>_to_<date>.pdf
ChemoControl_ProposalSummary_<date>_to_<date>.png
ChemoControl_ProposalSummary_<date>_to_<date>.svg
ChemoControl_ProposalSummary_SessionMetrics_<date>_to_<date>.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.stats import mannwhitneyu


CONTROL_COLOR = "#000000"
CHEMO_COLOR = "#D14949"
CONTROL_CATEGORY_COLORS = {
    "No CR": "#9E9E9E",
    "Poor CR": "#555555",
    "Good CR": "#000000",
}
CHEMO_CATEGORY_COLORS = {
    "No CR": "#F2A3A3",
    "Poor CR": "#D14949",
    "Good CR": "#8B0000",
}
BLOCKS = ("short", "long")
CATEGORIES = ("Good CR", "Poor CR", "No CR")

matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def get_field(obj, name, default=None):
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def safe_array(value) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(value).squeeze().astype(float)
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def scalar(value) -> float:
    arr = safe_array(value)
    return float(arr.flat[0]) if arr.size else np.nan


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.flat)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def is_chemo_session(session_data) -> bool:
    for setting in as_list(get_field(session_data, "TrialSettings")):
        gui = get_field(setting, "GUI")
        enabled = safe_array(get_field(gui, "ChemogeneticsEnabled", 0))
        if enabled.size and np.any(enabled != 0):
            return True
    return False


def is_excluded_trial(trial) -> bool:
    states = get_field(trial, "States")
    data = get_field(trial, "Data")
    if states is None or data is None:
        return True
    timeout = safe_array(get_field(states, "CheckEyeOpenTimeout"))
    if timeout.size and np.any(np.isfinite(timeout)):
        return True
    probe = safe_array(get_field(data, "IsProbeTrial", 0))
    return bool(probe.size and np.any(probe == 1))


def session_open_eye_area(trials: list) -> float:
    values = []
    for trial in trials:
        if is_excluded_trial(trial):
            continue
        eye = safe_array(get_field(get_field(trial, "Data"), "eyeAreaPixels"))
        if eye.size:
            values.append(eye[np.isfinite(eye)])
    values = [x for x in values if x.size]
    return float(np.nanpercentile(np.concatenate(values), 99)) if values else np.nan


def interpolate_trace(time: np.ndarray, signal: np.ndarray, grid: np.ndarray) -> np.ndarray:
    valid = np.isfinite(time) & np.isfinite(signal)
    time, signal = time[valid], signal[valid]
    if time.size < 2:
        return np.full(grid.shape, np.nan)
    time, idx = np.unique(time, return_index=True)
    if time.size < 2:
        return np.full(grid.shape, np.nan)
    return interp1d(
        time, signal[idx], bounds_error=False, fill_value=np.nan
    )(grid)


def trial_timing(trial, short_isi_max: float) -> tuple[float, float, float, str]:
    states = get_field(trial, "States")
    events = get_field(trial, "Events")
    data = get_field(trial, "Data")
    led = scalar(get_field(events, "GlobalTimer1_Start"))
    led_end = scalar(get_field(events, "GlobalTimer1_End"))
    puff = scalar(get_field(events, "GlobalTimer2_Start"))
    if not np.isfinite(puff):
        for candidate in ("AirPuff_Onset", "AirPuff_Ons", "AirPuff_On", "AirPuff_OnsetTime"):
            puff = scalar(get_field(data, candidate))
            if np.isfinite(puff):
                break
    isi_state = safe_array(get_field(states, "LED_Puff_ISI"))
    isi = float(isi_state[1] - isi_state[0]) if isi_state.size >= 2 else scalar(get_field(data, "ISI"))
    block_type = str(get_field(data, "BlockType", "")).lower()
    if "short" in block_type:
        block = "short"
    elif "long" in block_type:
        block = "long"
    else:
        block = "short" if np.isfinite(isi) and isi <= short_isi_max else "long"
    return led, led_end, puff, block


def classify_trial(
    grid: np.ndarray,
    trace: np.ndarray,
    puff: float,
    block: str,
    good_long: float = 0.05,
    poor_threshold: float = 0.02,
) -> tuple[str, float, float]:
    """Return category, baseline-subtracted CR amplitude, and baseline."""
    baseline_mask = (grid >= -0.2) & (grid <= 0)
    baseline = float(np.nanmean(trace[baseline_mask]))
    pre_ms = 25 if block == "short" else 50
    start, end = puff - pre_ms / 1000, puff + 0.012
    good_mask = (grid >= start) & (grid <= end)
    poor_mask = (grid >= 0) & (grid < start)
    if not np.any(good_mask) or not np.isfinite(baseline):
        return "No CR", np.nan, baseline
    good_mean = float(np.nanmean(trace[good_mask]) - baseline)
    good_peak = float(np.nanmax(trace[good_mask]) - baseline)
    poor_mean = float(np.nanmean(trace[poor_mask]) - baseline) if np.any(poor_mask) else -np.inf
    good_threshold = 0.03 if block == "short" else good_long
    peak_threshold = 0.035 if block == "short" else good_long
    if good_mean >= good_threshold or good_peak >= peak_threshold:
        category = "Good CR"
    elif poor_mean >= poor_threshold:
        category = "Poor CR"
    else:
        category = "No CR"
    return category, good_mean, baseline


def detect_onset(grid: np.ndarray, trace: np.ndarray, block: str) -> float:
    """Velocity-threshold CR onset in seconds from LED; NaN if unreliable."""
    valid = np.isfinite(grid) & np.isfinite(trace)
    baseline_mask = (grid >= -0.2) & (grid <= 0) & valid
    search_end = 0.212 if block == "short" else 0.412
    search_mask = (grid >= 0.05) & (grid <= search_end) & valid
    if np.count_nonzero(baseline_mask) < 2 or np.count_nonzero(search_mask) < 2:
        return np.nan
    corrected = trace - np.nanmean(trace[baseline_mask])
    smoothed = uniform_filter1d(corrected, size=5, mode="nearest")
    velocity = np.gradient(smoothed, grid)
    base_velocity = velocity[baseline_mask]
    threshold = max(float(np.nanmean(base_velocity) + 2 * np.nanstd(base_velocity)), 0.5)
    indices = np.flatnonzero(search_mask)
    if np.nanmax(corrected[search_mask]) < 0.03:
        return np.nan
    for i in range(len(indices) - 1):
        pair = indices[i : i + 2]
        if np.all(velocity[pair] >= threshold) and corrected[pair[0]] >= 0:
            return float(grid[pair[0]])
    return np.nan


@dataclass
class SessionResult:
    session: str
    date: str
    group: str
    traces: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {block: [] for block in BLOCKS}
    )
    categories: dict[str, list[str]] = field(
        default_factory=lambda: {block: [] for block in BLOCKS}
    )
    amplitudes: dict[str, list[float]] = field(
        default_factory=lambda: {block: [] for block in BLOCKS}
    )
    baselines: dict[str, list[float]] = field(
        default_factory=lambda: {block: [] for block in BLOCKS}
    )
    good_onsets: dict[str, list[float]] = field(
        default_factory=lambda: {block: [] for block in BLOCKS}
    )
    puff_times: dict[str, list[float]] = field(
        default_factory=lambda: {block: [] for block in BLOCKS}
    )
    led_durations: list[float] = field(default_factory=list)

    def mean_trace(self, block: str, grid: np.ndarray) -> np.ndarray:
        values = self.traces[block]
        return np.nanmean(np.vstack(values), axis=0) if values else np.full(grid.shape, np.nan)

    def fraction(self, block: str, category: str) -> float:
        values = self.categories[block]
        return values.count(category) / len(values) if values else np.nan

    def mean_amplitude(self, block: str) -> float:
        values = np.asarray(self.amplitudes[block], dtype=float)
        return float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan

    def median_onset_ms(self, block: str) -> float:
        values = np.asarray(self.good_onsets[block], dtype=float)
        return float(np.nanmedian(values) * 1000) if np.any(np.isfinite(values)) else np.nan


def load_sessions(paths: list[str], grid: np.ndarray, short_isi_max: float) -> list[SessionResult]:
    sessions = []
    for path in paths:
        loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
        session_data = loaded.get("SessionData")
        if session_data is None:
            print(f"Skipping {os.path.basename(path)}: SessionData not found")
            continue
        trials = as_list(get_field(get_field(session_data, "RawEvents"), "Trial"))
        open_eye = session_open_eye_area(trials)
        if not np.isfinite(open_eye) or open_eye <= 0:
            print(f"Skipping {os.path.basename(path)}: no open-eye reference")
            continue
        match = re.search(r"\d{8}", os.path.basename(path))
        date = match.group() if match else "Unknown"
        result = SessionResult(
            session=os.path.basename(path),
            date=date,
            group="Chemo" if is_chemo_session(session_data) else "Control",
        )
        for trial in trials:
            if is_excluded_trial(trial):
                continue
            data = get_field(trial, "Data")
            times = safe_array(get_field(data, "FECTimes"))
            eye = safe_array(get_field(data, "eyeAreaPixels"))
            if not times.size or times.size != eye.size:
                continue
            led, led_end, puff_abs, block = trial_timing(trial, short_isi_max)
            if not np.isfinite(led) or not np.isfinite(puff_abs):
                continue
            trace = 1 - eye / open_eye
            trace = uniform_filter1d(trace, size=5, mode="nearest")
            aligned = interpolate_trace(times - led, trace, grid)
            puff = puff_abs - led
            category, amplitude, baseline = classify_trial(grid, aligned, puff, block)
            result.traces[block].append(aligned)
            result.categories[block].append(category)
            result.amplitudes[block].append(amplitude)
            result.baselines[block].append(baseline)
            result.puff_times[block].append(puff)
            if category == "Good CR":
                result.good_onsets[block].append(detect_onset(grid, aligned, block))
            if np.isfinite(led_end):
                result.led_durations.append(led_end - led)
        if sum(len(result.traces[b]) for b in BLOCKS):
            sessions.append(result)
            counts = ", ".join(f"{b}={len(result.traces[b])}" for b in BLOCKS)
            print(f"Loaded {result.date} | {result.group:7s} | {counts}")
    return sessions


def sem(values: np.ndarray, axis=0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = np.sum(np.isfinite(values), axis=axis)
    sd = np.nanstd(values, axis=axis, ddof=1)
    return np.divide(sd, np.sqrt(n), out=np.full_like(sd, np.nan), where=n > 1)


def group_values(sessions: list[SessionResult], group: str, getter) -> np.ndarray:
    return np.asarray([getter(s) for s in sessions if s.group == group], dtype=float)


def p_value(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def p_label(p: float) -> str:
    if not np.isfinite(p):
        return "p = n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.15, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def add_stimulus(ax, puff: float, led_duration: float) -> None:
    if np.isfinite(led_duration):
        ax.axvspan(0, led_duration, color="#777777", alpha=0.16, linewidth=0)
    if np.isfinite(puff):
        ax.axvline(puff, color="#168AAD", linewidth=1.2, linestyle="--")


def baseline_matched_session_traces(
    sessions: list[SessionResult], grid: np.ndarray, block: str, n_bins: int = 10, seed: int = 42
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Match pooled trial baselines, then average retained trials within session."""
    pooled = {}
    for group in ("Control", "Chemo"):
        records = []
        for session_index, session in enumerate(sessions):
            if session.group != group:
                continue
            for trial_index, (trace, baseline) in enumerate(
                zip(session.traces[block], session.baselines[block])
            ):
                if np.isfinite(baseline):
                    records.append((session_index, trial_index, float(baseline), trace))
        pooled[group] = records

    ctrl_baseline = np.asarray([r[2] for r in pooled["Control"]])
    chemo_baseline = np.asarray([r[2] for r in pooled["Chemo"]])
    selected = {"Control": [], "Chemo": []}
    if not ctrl_baseline.size or not chemo_baseline.size:
        return {group: np.empty((0, len(grid))) for group in selected}, {group: 0 for group in selected}

    low = max(np.min(ctrl_baseline), np.min(chemo_baseline))
    high = min(np.max(ctrl_baseline), np.max(chemo_baseline))
    rng = np.random.default_rng(seed)
    if low < high:
        edges = np.linspace(low, high, n_bins + 1)
        for bin_index in range(n_bins):
            right_closed = bin_index == n_bins - 1
            ctrl_idx = np.flatnonzero(
                (ctrl_baseline >= edges[bin_index])
                & ((ctrl_baseline <= edges[bin_index + 1]) if right_closed else (ctrl_baseline < edges[bin_index + 1]))
            )
            chemo_idx = np.flatnonzero(
                (chemo_baseline >= edges[bin_index])
                & ((chemo_baseline <= edges[bin_index + 1]) if right_closed else (chemo_baseline < edges[bin_index + 1]))
            )
            count = min(len(ctrl_idx), len(chemo_idx))
            if count:
                selected["Control"].extend(rng.choice(ctrl_idx, count, replace=False))
                selected["Chemo"].extend(rng.choice(chemo_idx, count, replace=False))
    if not selected["Control"] or not selected["Chemo"]:
        count = min(len(ctrl_baseline), len(chemo_baseline))
        selected["Control"] = rng.choice(len(ctrl_baseline), count, replace=False).tolist()
        selected["Chemo"] = rng.choice(len(chemo_baseline), count, replace=False).tolist()

    matrices = {}
    trial_counts = {}
    for group in ("Control", "Chemo"):
        by_session: dict[int, list[np.ndarray]] = {}
        for pooled_index in selected[group]:
            session_index, _, _, trace = pooled[group][int(pooled_index)]
            by_session.setdefault(session_index, []).append(trace)
        session_means = [np.nanmean(np.vstack(values), axis=0) for values in by_session.values() if values]
        matrices[group] = np.vstack(session_means) if session_means else np.empty((0, len(grid)))
        trial_counts[group] = len(selected[group])
    return matrices, trial_counts


def plot_trace_panel(ax, sessions: list[SessionResult], grid: np.ndarray, block: str, label: str) -> None:
    matched, trial_counts = baseline_matched_session_traces(sessions, grid, block)
    for group, color in (("Control", CONTROL_COLOR), ("Chemo", CHEMO_COLOR)):
        traces = matched[group]
        if not traces.size:
            continue
        mean = np.nanmean(traces, axis=0)
        error = sem(traces, axis=0)
        ax.plot(
            grid * 1000, mean, color=color, linewidth=1.8,
            label=f"{group} ({len(traces)} sess.; {trial_counts[group]} trials)",
        )
        ax.fill_between(grid * 1000, mean - error, mean + error, color=color, alpha=0.18, linewidth=0)
    puff = np.nanmedian([x for s in sessions for x in s.puff_times[block]])
    led_duration = np.nanmedian([x for s in sessions for x in s.led_durations])
    add_stimulus(ax, puff * 1000, led_duration * 1000)
    ax.set(
        xlim=(-200, 600), xlabel="Time from LED onset (ms)", ylabel="Fraction eyelid closure",
        title=f"Baseline-matched — {block.capitalize()} block",
    )
    ax.set_ylim(bottom=min(-0.04, ax.get_ylim()[0]))
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    add_panel_label(ax, label)


def plot_composition(ax, sessions: list[SessionResult]) -> None:
    category_centers = np.arange(3, dtype=float)
    block_offset = 0.18
    group_offset = 0.055
    width = 0.10
    for category_index, category in enumerate(("No CR", "Poor CR", "Good CR")):
        for block_index, block in enumerate(BLOCKS):
            block_x = category_centers[category_index] + (-block_offset if block == "short" else block_offset)
            for group, direction, palette in (
                ("Control", -1, CONTROL_CATEGORY_COLORS),
                ("Chemo", 1, CHEMO_CATEGORY_COLORS),
            ):
                values = group_values(sessions, group, lambda s, b=block, c=category: s.fraction(b, c))
                finite = values[np.isfinite(values)]
                mean = np.mean(finite) if len(finite) else np.nan
                error = sem(finite) if len(finite) else np.nan
                x = block_x + direction * group_offset
                ax.bar(x, mean, width=width, color=palette[category], edgecolor="white", linewidth=0.4)
                ax.errorbar(x, mean, yerr=error, fmt="none", color=palette[category], capsize=2, linewidth=0.8)
    minor_positions = np.ravel(
        [[center - block_offset, center + block_offset] for center in category_centers]
    )
    ax.set_xticks(minor_positions, ["S", "L"] * 3)
    for center, category in zip(category_centers, ("No CR", "Poor CR", "Good CR")):
        ax.text(center, -0.17, category, transform=ax.get_xaxis_transform(), ha="center", fontweight="bold")
    for boundary in (0.5, 1.5):
        ax.axvline(boundary, color="#D0D0D0", linestyle=":", linewidth=0.8)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=CONTROL_CATEGORY_COLORS["Good CR"], label="Control"),
        Patch(facecolor=CHEMO_CATEGORY_COLORS["Good CR"], label="Chemo"),
    ]
    ax.set(ylabel="Mean fraction across sessions", ylim=(0, 1.08), title="CR classification by group")
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left", ncol=2)
    add_panel_label(ax, "C")


def dot_summary(ax, sessions: list[SessionResult], getter, ylabel: str, title: str, label: str, ylim=None) -> None:
    rng = np.random.default_rng(7)
    xticks, xticklabels = [], []
    for block_index, block in enumerate(BLOCKS):
        for group_index, (group, color) in enumerate((("Control", CONTROL_COLOR), ("Chemo", CHEMO_COLOR))):
            x = block_index * 3 + group_index
            values = group_values(sessions, group, lambda s, b=block: getter(s, b))
            finite = values[np.isfinite(values)]
            jitter = rng.uniform(-0.09, 0.09, size=len(finite))
            ax.scatter(np.full(len(finite), x) + jitter, finite, s=22, facecolor="white", edgecolor=color, linewidth=1, zorder=3)
            if len(finite):
                mean = np.mean(finite)
                error = sem(finite)
                ax.errorbar(x, mean, yerr=error, fmt="o", color=color, markerfacecolor=color, markersize=4.5, capsize=3, linewidth=1.2, zorder=4)
            xticks.append(x)
            xticklabels.append(group)
        ctrl = group_values(sessions, "Control", lambda s, b=block: getter(s, b))
        chemo = group_values(sessions, "Chemo", lambda s, b=block: getter(s, b))
        p = p_value(ctrl, chemo)
        vals = np.concatenate((ctrl[np.isfinite(ctrl)], chemo[np.isfinite(chemo)]))
        if len(vals):
            top = np.max(vals)
            bottom = np.min(vals)
            scale = max(abs(top), abs(bottom), 1.0)
            pad = max((top - bottom) * 0.12, 0.025 * scale)
            ax.plot([block_index * 3, block_index * 3 + 1], [top + pad, top + pad], color="#555555", linewidth=0.8)
            ax.text(block_index * 3 + 0.5, top + pad * 1.15, p_label(p), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xticks, xticklabels, rotation=18)
    ax.text(0.5, -0.25, "Short", ha="center", transform=ax.get_xaxis_transform(), fontweight="bold")
    ax.text(3.5, -0.25, "Long", ha="center", transform=ax.get_xaxis_transform(), fontweight="bold")
    ax.set(ylabel=ylabel, title=title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    add_panel_label(ax, label)


def session_rows(sessions: list[SessionResult]) -> list[dict[str, object]]:
    rows = []
    for session in sessions:
        for block in BLOCKS:
            row = {
                "session": session.session,
                "date": session.date,
                "group": session.group,
                "block": block,
                "n_trials": len(session.categories[block]),
                "mean_cr_amplitude_baseline_subtracted": session.mean_amplitude(block),
                "median_good_cr_onset_ms": session.median_onset_ms(block),
            }
            for category in CATEGORIES:
                row["fraction_" + category.lower().replace(" ", "_")] = session.fraction(block, category)
            rows.append(row)
    return rows


def save_csv(rows: list[dict[str, object]], path: str) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(sessions: list[SessionResult], grid: np.ndarray, output_stem: str, dpi: int) -> None:
    n_control = sum(s.group == "Control" for s in sessions)
    n_chemo = sum(s.group == "Chemo" for s in sessions)
    dates = [s.date for s in sessions if s.date != "Unknown"]
    date_text = f"{min(dates)}–{max(dates)}" if dates else "date unavailable"
    fig = plt.figure(figsize=(11.5, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 1.05))
    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[0, 2]),
        "d": fig.add_subplot(gs[1, 0]),
        "e": fig.add_subplot(gs[1, 1]),
        "f": fig.add_subplot(gs[1, 2]),
    }
    plot_trace_panel(axes["a"], sessions, grid, "short", "A")
    plot_trace_panel(axes["b"], sessions, grid, "long", "B")
    plot_composition(axes["c"], sessions)
    dot_summary(
        axes["d"], sessions,
        lambda s, b: s.fraction(b, "Good CR"),
        "Good CR fraction", "Well-timed responses", "D", ylim=(0, 1.12),
    )
    dot_summary(
        axes["e"], sessions,
        lambda s, b: s.mean_amplitude(b),
        "Baseline-subtracted FEC", "Anticipatory response amplitude", "E",
    )
    dot_summary(
        axes["f"], sessions,
        lambda s, b: s.median_onset_ms(b),
        "Good CR onset from LED (ms)", "Response onset timing", "F",
    )
    fig.suptitle(
        "Conditioned eyelid responses during chemogenetic inhibition\n"
        f"Control n={n_control} sessions; Chemo n={n_chemo} sessions | {date_text}",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5, -0.015,
        "A–B: baseline-matched trials, averaged within session; lines and shading show mean ± SEM across sessions. "
        "Dots: individual sessions; filled points: mean ± SEM. "
        "P values: two-sided Mann–Whitney U tests.",
        ha="center", fontsize=7, color="#444444",
    )
    for extension in ("pdf", "svg", "png"):
        fig.savefig(f"{output_stem}.{extension}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.dirname(os.path.abspath(__file__)), help="Folder containing *_EBC_*.mat files")
    parser.add_argument("--output-dir", default=None, help="Output folder (default: data folder)")
    parser.add_argument("--dpi", type=int, default=400, help="PNG resolution")
    parser.add_argument("--short-isi-max", type=float, default=0.30, help="Maximum short-block ISI in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir or data_dir)
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(data_dir, "*_EBC_*.mat")))
    if not paths:
        raise SystemExit(f"No *_EBC_*.mat files found in {data_dir}")
    grid = np.arange(-0.2, 0.6001, 1 / 250)
    sessions = load_sessions(paths, grid, args.short_isi_max)
    if not sessions:
        raise SystemExit("No usable sessions were found")
    groups = {s.group for s in sessions}
    if groups != {"Control", "Chemo"}:
        raise SystemExit(f"Both Control and Chemo sessions are required; found {sorted(groups)}")
    dates = [s.date for s in sessions if s.date != "Unknown"]
    first, last = (min(dates), max(dates)) if dates else ("Unknown", "Unknown")
    stem = os.path.join(output_dir, f"ChemoControl_ProposalSummary_{first}_to_{last}")
    make_figure(sessions, grid, stem, args.dpi)
    rows = session_rows(sessions)
    csv_path = os.path.join(output_dir, f"ChemoControl_ProposalSummary_SessionMetrics_{first}_to_{last}.csv")
    save_csv(rows, csv_path)
    print(f"\nSaved figure: {stem}.pdf/.png/.svg")
    print(f"Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()
