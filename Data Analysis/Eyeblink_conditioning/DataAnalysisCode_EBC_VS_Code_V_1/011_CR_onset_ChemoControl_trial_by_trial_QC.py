from __future__ import annotations

import glob
import importlib.util
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ONSET_SCRIPT = "010_CR_onsetTiming_ChemoControl.py"


def load_onset_module():
    spec = importlib.util.spec_from_file_location("cr_onset_010", ONSET_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {ONSET_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session_token(filepath):
    match = re.search(r"\d{8}", os.path.basename(filepath))
    if match:
        return match.group()
    return os.path.splitext(os.path.basename(filepath))[0]


def session_group_label(records):
    """Return 'Chemo' or 'Control' based on the is_chemo flag in the records."""
    if not records:
        return "Unknown"
    return "Chemo" if records[0].get("is_chemo", False) else "Control"


def compute_velocity_diagnostics(
    onset_module,
    time,
    signal,
    t_led=0.0,
    baseline_window=(-0.2, 0.0),
    velocity_smooth_window=9,
    velocity_std_factor=1.5,
    velocity_floor=0.03,
):
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    valid = np.isfinite(time) & np.isfinite(signal)
    out = {
        "time": time,
        "velocity": np.full_like(time, np.nan, dtype=float),
        "threshold": np.nan,
        "baseline_amp": np.nan,
    }
    if np.sum(valid) < 2:
        return out

    baseline_idx = (time >= (t_led + baseline_window[0])) & (time <= (t_led + baseline_window[1]))
    if not np.any(baseline_idx):
        return out

    baseline_amp = np.nanmean(signal[baseline_idx])
    signal_bs = signal - baseline_amp
    sig_smooth = onset_module.smooth_trace(signal_bs, window=velocity_smooth_window)
    velocity = np.gradient(sig_smooth, time)

    baseline_velocity = velocity[baseline_idx]
    baseline_velocity = baseline_velocity[np.isfinite(baseline_velocity)]
    if baseline_velocity.size:
        med = np.nanmedian(baseline_velocity)
        mad = np.nanmedian(np.abs(baseline_velocity - med))
        robust_std = 1.4826 * mad if np.isfinite(mad) else np.nan
        if not np.isfinite(robust_std) or robust_std == 0:
            robust_std = np.nanstd(baseline_velocity)
        threshold = max(med + velocity_std_factor * robust_std, velocity_floor)
    else:
        threshold = velocity_floor

    out["velocity"] = velocity
    out["threshold"] = threshold
    out["baseline_amp"] = baseline_amp
    return out


def group_by_session(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["filepath"]].append(record)
    for records_for_session in grouped.values():
        records_for_session.sort(key=lambda r: r.get("trial_index", 0))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def add_session_summary_page(pdf, session_name, records, group_label):
    fig, axs = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    colors = {"Good CR": "black", "Poor CR": "tab:orange", "No CR": "lightgray"}
    markers = {"short": "o", "long": "s"}
    trial_indices = [r["trial_index"] for r in records]
    block_labels = [r["block_label"] for r in records]

    ax = axs[0]
    for category, color in colors.items():
        for block, marker in markers.items():
            subset = [
                r for r in records
                if r["category"] == category and r["block_label"] == block and np.isfinite(r["onset_latency"])
            ]
            if not subset:
                continue
            ax.scatter(
                [r["trial_index"] for r in subset],
                [r["onset_latency"] for r in subset],
                s=22,
                color=color,
                marker=marker,
                label=f"{block.title()} {category}",
                alpha=0.85,
            )
    for i in range(1, len(records)):
        if block_labels[i] != block_labels[i - 1]:
            x = 0.5 * (trial_indices[i] + trial_indices[i - 1])
            ax.axvline(x, color="red", linestyle="--", linewidth=1.2)
            ax.text(x, 0.43, "block transition", rotation=90, va="top", ha="right", fontsize=8, color="red")
    ax.axhline(0.200, color="royalblue", linestyle=":", linewidth=1, label="Short puff")
    ax.axhline(0.400, color="seagreen", linestyle=":", linewidth=1, label="Long puff")
    ax.set_ylabel("Detected onset from LED (s)")
    ax.set_ylim(-0.02, 0.45)
    ax.set_title(
        f"{session_name}  [{group_label}]\nTrial-by-trial CR onset aligned to session trial order"
    )
    ax.legend(loc="upper right", fontsize=7, ncol=2)

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
            ax.axvline(0.5 * (trial_indices[i] + trial_indices[i - 1]), color="red", linestyle="--", linewidth=1.2)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["No CR", "Poor CR", "Good CR"])
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Classification")
    ax.set_ylim(-0.5, 2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_trial_page(pdf, onset_module, session_name, records, page_index, group_label):
    fig, axs = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    axs = axs.ravel()
    for ax, record in zip(axs, records):
        time = record["t_grid"]
        signal = record["Fq"]
        onset_time = record["onset_time"]
        puff_time = record.get("t_puff", 0.200 if record["block_label"] == "short" else 0.400)
        diag = compute_velocity_diagnostics(onset_module, time, signal)

        ax.axvspan(0.0, 0.05, color="lightgray", alpha=0.25)
        ax.axvline(puff_time, color="tab:blue", linestyle="--", linewidth=1.1)
        ax.plot(time, signal, color="black", linewidth=1.2, label="FEC")
        if np.isfinite(onset_time):
            ax.axvline(onset_time, color="red", linestyle="--", linewidth=1.3, label="onset")

        ax_vel = ax.twinx()
        ax_vel.plot(diag["time"], diag["velocity"], color="tab:purple", alpha=0.45, linewidth=0.8, label="velocity")
        if np.isfinite(diag["threshold"]):
            ax_vel.axhline(diag["threshold"], color="tab:purple", linestyle=":", alpha=0.65, linewidth=0.8)
        ax_vel.set_yticks([])
        ax_vel.spines["right"].set_visible(False)

        onset_label = f"{onset_time:.3f}s" if np.isfinite(onset_time) else "not detected"
        ax.set_title(
            f"Trial {record['trial_index']} | {record['block_label']} | {record['category']} | onset {onset_label}",
            fontsize=8,
        )
        ax.set_xlim(-0.05, 0.45)
        ax.set_ylim(-0.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axs[len(records):]:
        ax.axis("off")
    for ax in axs[-2:]:
        ax.set_xlabel("Time from LED onset (s)")
    for ax in axs[::2]:
        ax.set_ylabel("FEC")

    fig.suptitle(
        f"{session_name}  [{group_label}] | Trial traces and velocity onset QC | page {page_index}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_session_pdf(onset_module, filepath, records):
    token = session_token(filepath)
    session_name = os.path.basename(filepath)
    group_label = session_group_label(records)
    out_name = f"CR_onset_trial_by_trial_QC_{token}.pdf"
    with PdfPages(out_name) as pdf:
        add_session_summary_page(pdf, session_name, records, group_label)
        page_index = 1
        for start in range(0, len(records), 6):
            page_index += 1
            plot_trial_page(pdf, onset_module, session_name, records[start:start + 6], page_index, group_label)
    print(f"Saved {out_name} ({len(records)} trials)  [{group_label}]")


def main():
    onset_module = load_onset_module()
    data_files = sorted(glob.glob("*_EBC_*.mat"))
    if not data_files:
        print("No *_EBC_*.mat files found in current folder.")
        return
    records = onset_module.prepare_trial_records(data_files)
    if not records:
        print("No usable trials found.")
        return
    for filepath, records_for_session in group_by_session(records).items():
        write_session_pdf(onset_module, filepath, records_for_session)


if __name__ == "__main__":
    main()
