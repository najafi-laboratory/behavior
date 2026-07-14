from __future__ import annotations
import pathlib
import re
import csv
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgb
from scipy.optimize import curve_fit


# ============================================================
# Matplotlib config
# ============================================================
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.use14corefonts"] = False


# ============================================================
# PATHS
# ============================================================
MOUSE_ROOTS: Dict[str, pathlib.Path] = {
    "SM09": pathlib.Path("/Users/zahra/Desktop/NajafiLab/EBC_Analysis/SM09_Summary/SM09_Summary_all"),
}

SAVE_DIR = pathlib.Path(
    "/Users/zahra/Desktop/NajafiLab/EBC_Analysis/SM09_Summary/SM09_Transition_Trial_By_Trial_Adaptation_FIXED"
)

DATE_MIN: Optional[int] = None


# ============================================================
# Analysis params
# ============================================================
FIXED_BLOCK_LEN = 40
MIN_BLOCK_LEN = 20
BASELINE_THRESHOLD = 0.35

REMOVE_FIRST_LONG_WARMUP = True
FIRST_LONG_WARMUP_TRIALS = 15

SHORT_DELAY = (150, 300)
LONG_DELAY = (350, 550)

RESP_WIN = (170, 210)
BASELINE_WIN = (-100, 0)
TRACE_WIN = (-100, 600)
TRACE_STEP = 1

MIN_FRAC = 0.40
SMOOTH_WINDOW = 5
ALPHA_FILL = 0.25

COLOR_SHORT = "blue"
COLOR_LONG = "lime"
COLOR_BASELINE = "tab:orange"
COLOR_BLOCKNUM_AVG = "tab:purple"

BASE_NEG = "tab:green"
BASE_POS = "tab:blue"

EXPECTED_SHORT = (200, 220)
EXPECTED_LONG = (400, 420)


# ============================================================
# Color helpers
# ============================================================
def _blend(c1, c2, t: float):
    a = np.array(to_rgb(c1))
    b = np.array(to_rgb(c2))
    return tuple((1 - t) * a + t * b)

def lighten(color, t=0.35):
    return _blend(color, "white", t)

def darken(color, t=0.20):
    return _blend(color, "black", t)

COLOR_TRACE_NEG_1 = lighten(BASE_NEG, 0.35)
COLOR_TRACE_NEG_2 = darken(BASE_NEG, 0.18)
COLOR_TRACE_1 = lighten(BASE_POS, 0.35)
COLOR_TRACE_2 = darken(BASE_POS, 0.18)


# ============================================================
# Utility helpers
# ============================================================
def smooth_nan(y: np.ndarray, w: int = 5) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if w <= 1:
        return y.copy()
    if w % 2 == 0:
        w += 1

    out = np.full_like(y, np.nan)
    hw = w // 2
    for i in range(len(y)):
        lo = max(0, i - hw)
        hi = min(len(y), i + hw + 1)
        yy = y[lo:hi]
        good = np.isfinite(yy)
        if np.any(good):
            out[i] = np.nanmean(yy[good])
    return out

def is_chemo_session(p):
    return "Opto" in p.name


# ============================================================
# MAT loading helpers
# ============================================================
def to_py(obj: Any) -> Any:
    try:
        from scipy.io.matlab import mat_struct
    except Exception:
        from scipy.io.matlab.mio5_params import mat_struct

    if isinstance(obj, mat_struct):
        out = {}
        for k in getattr(obj, "_fieldnames", []):
            out[k] = to_py(getattr(obj, k))
        return out
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return [to_py(x) for x in obj.ravel().tolist()]
    if isinstance(obj, list):
        return [to_py(x) for x in obj]
    return obj

def get_ci(d: Any, *names, default=None):
    if not isinstance(d, dict):
        return default

    keys = list(d.keys())
    for n in names:
        if n in d:
            return d[n]
        if isinstance(n, str):
            nl = n.lower()
            for k in keys:
                if isinstance(k, str) and k.lower() == nl:
                    return d[k]
    return default

def first_array(*candidates) -> np.ndarray:
    for c in candidates:
        if c is None:
            continue
        try:
            a = np.asarray(c).ravel()
            if a.size > 0:
                return a
        except Exception:
            pass
    return np.array([])

def array_ms(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a).ravel()
    return a * 1000.0 if a.size else a


# ============================================================
# Trial extraction
# ============================================================
def load_trials_from_mat(p: pathlib.Path) -> List[dict]:
    raw = sio.loadmat(p, struct_as_record=False, squeeze_me=True)
    raw = {k: to_py(v) for k, v in raw.items() if not k.startswith("__")}
    SD = to_py(get_ci(raw, "SessionData"))
    if not isinstance(SD, dict):
        raise RuntimeError("No SessionData")

    RE = get_ci(SD, "RawEvents", "rawevents")
    Trials = get_ci(RE, "Trial") if RE else get_ci(SD, "Trial")
    if Trials is None:
        raise RuntimeError("No Trials found")

    Trials = to_py(Trials)
    if isinstance(Trials, dict):
        keys = sorted(Trials.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
        tlist = [to_py(Trials[k]) for k in keys]
    elif isinstance(Trials, (list, np.ndarray)):
        tlist = [to_py(x) for x in (Trials if isinstance(Trials, list) else Trials.ravel())]
    else:
        tlist = [Trials]

    out = []
    for tr in tlist:
        ev = to_py(get_ci(tr, "Events", "events")) or {}
        dat = to_py(get_ci(tr, "Data", "data")) or {}

        led_on = first_array(get_ci(ev, "GlobalTimer1_Start"))
        ap_on = first_array(get_ci(ev, "GlobalTimer2_Start"))
        ap_off = first_array(get_ci(ev, "GlobalTimer2_End"))
        fec_t = first_array(get_ci(dat, "FECTimes", "FEC_time", "fec_time"))
        eye = first_array(get_ci(dat, "eyeAreaPixels", "EyeAreaPixels", "eye_area"))

        out.append(
            dict(
                LED_on=float(led_on[0]) * 1000.0 if led_on.size else np.nan,
                AP_on=float(ap_on[0]) * 1000.0 if ap_on.size else np.nan,
                AP_off=float(ap_off[0]) * 1000.0 if ap_off.size else np.nan,
                FECTimes=array_ms(fec_t) if fec_t.size else np.array([]),
                eye=eye,
            )
        )
    return out

def session_overall_max_eye(trials: List[dict]) -> float:
    vals = []
    per_trial_max = []

    for tr in trials:
        eye = np.asarray(tr.get("eye", []), dtype=float)
        eye = eye[np.isfinite(eye)]
        if eye.size:
            vals.append(eye)
            per_trial_max.append(np.nanmax(eye))

    if not vals:
        return np.nan

    allv = np.concatenate(vals)
    mx, mn = np.nanmax(allv), np.nanmin(allv)

    if mx <= 1.2 and mn >= -0.2:
        return 1.0

    p99 = np.nanpercentile(allv, 99)
    if p99 > 0:
        return p99

    if per_trial_max:
        return np.nanmax(per_trial_max)

    return np.nan

def window_mean(t, y, win):
    m = (t >= win[0]) & (t <= win[1])
    return np.nanmean(y[m]) if np.any(m) else np.nan

def classify_block(delay_ms: float):
    if not np.isfinite(delay_ms):
        return None
    if SHORT_DELAY[0] <= delay_ms <= SHORT_DELAY[1]:
        return "short"
    if LONG_DELAY[0] <= delay_ms <= LONG_DELAY[1]:
        return "long"
    return None

def make_block_count_lines_pdf(save_path, meta_all):
    rows = []
    for m in meta_all:
        rows.extend(m.get("kept_rows", []))

    s2l_rows = [r for r in rows if r["direction"] == "s2l"]
    l2s_rows = [r for r in rows if r["direction"] == "l2s"]

    def build_counts(rows_dir):
        if not rows_dir:
            return None, None, None

        x = np.arange(-FIXED_BLOCK_LEN, FIXED_BLOCK_LEN)

        max_block = 0
        for r in rows_dir:
            max_block = max(max_block, int(r["pre_block_num"]), int(r["post_block_num"]))

        counts = {b: np.zeros(len(x), dtype=float) for b in range(1, max_block + 1)}

        for r in rows_dir:
            pre_block = int(r["pre_block_num"])
            post_block = int(r["post_block_num"])
            pre_n = int(r["pre_take_n"])
            post_n = int(r["post_take_n"])

            for i, xv in enumerate(x):
                if -pre_n <= xv < 0:
                    counts[pre_block][i] += 1
                elif 0 <= xv < post_n:
                    counts[post_block][i] += 1

        return counts, x, max_block

    s2l_counts, x, _ = build_counts(s2l_rows)
    l2s_counts, _, _ = build_counts(l2s_rows)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    def plot_one(ax, counts, title):
        ax.axvline(0, color="k", ls="--", lw=1.5)
        ax.set_title(title)

        if counts is None:
            ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
            return

        for b in sorted(counts.keys()):
            ax.plot(x, counts[b], lw=2, label=f"Block {b}")

        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=4)

    plot_one(axs[0], s2l_counts, "S→L: block counts at each aligned trial")
    plot_one(axs[1], l2s_counts, "L→S: block counts at each aligned trial")

    axs[1].set_xlabel("Aligned trial # around transition")

    plt.tight_layout()
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved: {save_path}  trans6_8RowSummary_V_3.py:323 - 08_trans6_8RowSummary_V_3.py:322")

def per_trial_metrics(tr: dict, overall_max: float):
    led, t, eye = tr["LED_on"], tr["FECTimes"], tr["eye"]
    if not (np.isfinite(led) and t.size >= 2 and eye.size >= 2):
        return np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([])

    fec = 1.0 - (eye / overall_max) if (np.isfinite(overall_max) and overall_max > 0 and overall_max != 1.0) else eye
    fec = np.clip(fec, -0.5, 1.5)

    m = min(fec.size, t.size)
    fec, t = fec[:m], t[:m]

    order = np.argsort(t)
    t, fec = t[order], fec[order]
    rel_t = t - led

    base = window_mean(rel_t, fec, BASELINE_WIN)
    resp = window_mean(rel_t, fec, RESP_WIN)
    amp_raw = resp
    amp_bs = resp - base if (np.isfinite(resp) and np.isfinite(base)) else np.nan

    vel_window = np.nan
    if rel_t.size >= 3:
        dfdt = np.gradient(fec, rel_t)
        vel_window = window_mean(rel_t, dfdt, RESP_WIN)

    return amp_raw, amp_bs, vel_window, base, rel_t, fec


# ============================================================
# Block helpers
# ============================================================
def build_blocks(blks: List[Optional[str]]) -> List[dict]:
    blocks = []
    if not blks:
        return blocks

    start = 0
    cur = blks[0]
    for i in range(1, len(blks)):
        if blks[i] != cur:
            blocks.append({"label": cur, "start": start, "end": i - 1, "length": i - start})
            start = i
            cur = blks[i]
    blocks.append({"label": cur, "start": start, "end": len(blks) - 1, "length": len(blks) - start})
    return blocks

def apply_first_long_warmup_removal(blocks: List[dict]) -> List[dict]:
    out = [dict(b) for b in blocks]
    if not REMOVE_FIRST_LONG_WARMUP or not out:
        return out

    first = out[0]
    if first["label"] == "long" and first["length"] > FIRST_LONG_WARMUP_TRIALS:
        first["start"] += FIRST_LONG_WARMUP_TRIALS
        first["length"] -= FIRST_LONG_WARMUP_TRIALS

    return out

def infer_session_family(eff_blocks: List[dict]) -> Optional[int]:
    lens = [b["length"] for b in eff_blocks if b["label"] in ("short", "long")]
    if not lens:
        return None
    med = float(np.median(lens))
    return 40 if med < 45 else 50

def family_bounds(family: int):
    if family == 40:
        return 35, 45
    if family == 50:
        return 45, 55
    return None, None


# ============================================================
# Fitting
# ============================================================
def _exp_rise(x, y0, A, tau):
    xeff = np.maximum(0.0, x)
    return y0 + A * (1.0 - np.exp(-xeff / np.maximum(1e-6, tau)))

def _exp_decay(x, y0, A, tau):
    xeff = np.maximum(0.0, x)
    return y0 + A * np.exp(-xeff / np.maximum(1e-6, tau))

def _fit_stack_post0(x, stack, kind):
    cnt = np.sum(np.isfinite(stack), axis=0)
    mu = np.nanmean(stack, axis=0)
    se = np.nanstd(stack, axis=0, ddof=0) / np.sqrt(np.maximum(1, cnt))

    frac = cnt / max(1, stack.shape[0]) if stack.shape[0] else np.zeros_like(cnt, dtype=float)
    mask = frac >= MIN_FRAC
    mu_plot = np.where(mask, mu, np.nan)
    se_plot = np.where(mask, se, np.nan)

    valid_x = (x >= 0) & mask
    xx = x[valid_x]
    yy = mu[valid_x]
    ss = se[valid_x]

    info = {"ok": False, "msg": "no data"}
    yfit = np.full_like(x, np.nan, dtype=float)

    if xx.size >= 4:
        func = _exp_rise if kind == "rise" else _exp_decay
        y0 = np.nanmedian(yy)
        amp = (yy[-1] - yy[0]) if kind == "rise" else (yy[0] - yy[-1])
        p0 = [y0, amp, 5.0]
        bounds = ([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, 1e3])

        try:
            sigma = ss if np.all(ss > 1e-9) else None
            popt, _ = curve_fit(func, xx, yy, p0=p0, bounds=bounds, sigma=sigma, maxfev=10000)

            yfit = func(x, *popt)
            tau = popt[2]
            plat_x = tau * np.log(10)
            plat_y = func(plat_x, *popt)

            info = {
                "ok": True,
                "tau": tau,
                "plat_x": plat_x,
                "plat_y": plat_y,
                "params": popt,
            }
        except Exception as e:
            info["msg"] = str(e)

    return mu_plot, se_plot, yfit, info


# ============================================================
# Plot helpers
# ============================================================
def mean_sem_trace(traces, tgrid, bl_sub=False):
    if not traces:
        return np.full_like(tgrid, np.nan), np.full_like(tgrid, np.nan)

    M = np.vstack(traces)
    if bl_sub:
        mask_bl = (tgrid >= BASELINE_WIN[0]) & (tgrid <= BASELINE_WIN[1])
        bls = np.nanmean(M[:, mask_bl], axis=1)
        M = M - bls[:, None]

    mu = np.nanmean(M, axis=0)
    cnt = np.sum(np.isfinite(M), axis=0)
    se = np.nanstd(M, axis=0, ddof=0) / np.sqrt(np.maximum(1, cnt))
    return mu, se

def mean_sem_counts(stack):
    cnt = np.sum(np.isfinite(stack), axis=0)
    mu = np.nanmean(stack, axis=0)
    se = np.nanstd(stack, axis=0, ddof=0) / np.sqrt(np.maximum(1, cnt))
    frac = cnt / max(1, stack.shape[0]) if stack.shape[0] else np.zeros_like(cnt, dtype=float)
    return mu, se, cnt, frac

def _shade_background(ax, mode):
    ax.axvspan(RESP_WIN[0], RESP_WIN[1], color="gray", alpha=0.12, lw=0)
    ax.axvspan(0, 50, color="gray", alpha=0.18, lw=0)
    if mode == "s2l":
        ax.axvspan(EXPECTED_LONG[0], EXPECTED_LONG[1], color=COLOR_LONG, alpha=0.25, lw=0)
    else:
        ax.axvspan(EXPECTED_SHORT[0], EXPECTED_SHORT[1], color=COLOR_SHORT, alpha=0.25, lw=0)

def format_summary_title(summary_text: str) -> str:
    parts = [p.strip() for p in summary_text.split("|")]
    if len(parts) <= 3:
        return summary_text
    return " | ".join(parts[:3]) + "\n" + " | ".join(parts[3:])

def format_trial_position_label(block_label: str, positions: List[int]) -> str:
    if not positions:
        return block_label
    vals = [int(v) for v in positions if np.isfinite(v)]
    if not vals:
        return block_label
    return f"{block_label}: {min(vals)}:{max(vals)}"

def style_axes_keep_xy(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def draw_time_panel(ax, d, title, mode, tgrid, bl_sub=False):
    def p(trace_keys, pos_keys, col, block_label):
        tr = []
        pos = []
        for k in trace_keys:
            tr.extend(d.get(k, []))
        for k in pos_keys:
            pos.extend(d.get(k, []))
        if tr:
            m, s = mean_sem_trace(tr, tgrid, bl_sub)
            ax.fill_between(tgrid, m - s, m + s, color=col, alpha=ALPHA_FILL, lw=0)
            ax.plot(tgrid, m, color=col, lw=1.4, label=format_trial_position_label(block_label, pos))

    p(["traces_neg1"], ["pos_neg1"], COLOR_TRACE_NEG_1, "Prev block")
    p(["traces_neg2"], ["pos_neg2"], COLOR_TRACE_NEG_2, "Prev block")
    p(["traces1"], ["pos1"], COLOR_TRACE_1, "New block")
    p(["traces2"], ["pos2"], COLOR_TRACE_2, "New block")

    _shade_background(ax, mode)
    ax.axvline(0, color="k", ls="--")
    ax.set_title(title)
    ax.set_xlim(TRACE_WIN)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc="upper left")
    style_axes_keep_xy(ax)

def draw_trial_panel(ax, x, stack, title, mode, do_fit=False, color_override=None):
    ax.axvline(0, color="k", ls="--")

    mu, se, cnt, frac = mean_sem_counts(stack)
    mask = frac >= MIN_FRAC
    mu_p = smooth_nan(np.where(mask, mu, np.nan), SMOOTH_WINDOW)
    se_p = smooth_nan(np.where(mask, se, np.nan), SMOOTH_WINDOW)

    col = color_override if color_override is not None else (COLOR_SHORT if mode == "s2l" else COLOR_LONG)

    good = np.isfinite(mu_p) & np.isfinite(se_p)
    if np.any(good):
        ax.fill_between(x[good], mu_p[good] - se_p[good], mu_p[good] + se_p[good], color=col, alpha=0.20, lw=0)
        ax.plot(x[good], mu_p[good], color=col, alpha=0.9, lw=1.8)

    if do_fit and stack.size > 0:
        kind = "rise" if mode == "s2l" else "decay"
        _, _, yfit, info = _fit_stack_post0(x, stack, kind)
        if info["ok"]:
            fit_col = darken(col, 0.3)
            post = x >= 0
            ax.plot(x[post], yfit[post], color=fit_col, lw=2.5)
            px, py = info["plat_x"], info["plat_y"]
            if x.min() <= px <= x.max():
                ax.plot(px, py, "o", color="red", markersize=6, zorder=10)
                ax.text(px, py, f" {px:.1f}", color="red", fontsize=9, fontweight="bold", va="bottom")
            ax.text(0.05, 0.90, f"τ={info['tau']:.1f}", transform=ax.transAxes, fontsize=9)

    ax.set_title(title)
    ax.set_xlim(x.min(), x.max())
    ax.set_xticks(np.arange(x.min(), x.max() + 1, 5))
    style_axes_keep_xy(ax)

def draw_transition_type_average_blocknum_panel(ax, meta_all, direction, title, color=COLOR_BLOCKNUM_AVG):
    rows = []
    for m in meta_all:
        rows.extend(m.get("kept_rows", []))
    rows = [r for r in rows if r["direction"] == direction]

    ax.axvline(0, color="k", ls="--")
    ax.set_title(title)

    if not rows:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
        return

    grouped = {}
    for r in rows:
        key = f"B{r['pre_block_num']}→B{r['post_block_num']}"
        grouped.setdefault(key, []).append(r)

    x = np.arange(-FIXED_BLOCK_LEN, FIXED_BLOCK_LEN)
    group_means = []

    for _, group_rows in sorted(grouped.items()):
        pre_block = float(group_rows[0]["pre_block_num"])
        post_block = float(group_rows[0]["post_block_num"])
        y = np.where(x < 0, pre_block, post_block).astype(float)
        group_means.append(y)

    if not group_means:
        ax.text(0.5, 0.5, "No grouped means", ha="center", va="center", transform=ax.transAxes)
        return

    G = np.vstack(group_means)
    mu = np.nanmean(G, axis=0)
    se = np.nanstd(G, axis=0, ddof=0) / np.sqrt(max(1, G.shape[0]))

    ax.fill_between(x, mu - se, mu + se, color=color, alpha=0.22, lw=0)
    ax.plot(x, mu, color=color, lw=2.5)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel("Mean Block #")
    style_axes_keep_xy(ax)

def draw_valid_transition_count_panel(ax, meta_all, direction, title):
    rows = []
    for m in meta_all:
        rows.extend(m.get("kept_rows", []))
    rows = [r for r in rows if r["direction"] == direction]

    ax.axvline(0, color="k", ls="--", lw=1.5)
    ax.set_title(title)

    if not rows:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(-FIXED_BLOCK_LEN, FIXED_BLOCK_LEN)
    counts = np.zeros(len(x), dtype=float)

    for i, xv in enumerate(x):
        n_here = 0
        for r in rows:
            pre_n = int(r["pre_take_n"])
            post_n = int(r["post_take_n"])

            if -pre_n <= xv < 0:
                n_here += 1
            elif 0 <= xv < post_n:
                n_here += 1

        counts[i] = n_here

    ax.plot(x, counts, lw=2.5, color="black")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel("Valid transitions")
    ax.set_xlabel("Aligned trial #")
    style_axes_keep_xy(ax)

def make_block_count_lines_pdf(save_path, meta_all, zoom_only=True, zoom_window=3):
    rows = []
    for m in meta_all:
        rows.extend(m.get("kept_rows", []))

    s2l_rows = [r for r in rows if r["direction"] == "s2l"]
    l2s_rows = [r for r in rows if r["direction"] == "l2s"]

    def build_counts(rows_dir):
        if not rows_dir:
            return None, None, None

        x = np.arange(-FIXED_BLOCK_LEN, FIXED_BLOCK_LEN)

        max_block = 0
        for r in rows_dir:
            max_block = max(max_block, int(r["pre_block_num"]), int(r["post_block_num"]))

        counts = {b: np.zeros(len(x), dtype=float) for b in range(1, max_block + 1)}

        for r in rows_dir:
            pre_block = int(r["pre_block_num"])
            post_block = int(r["post_block_num"])
            pre_n = int(r["pre_take_n"])
            post_n = int(r["post_take_n"])

            for i, xv in enumerate(x):
                if -pre_n <= xv < 0:
                    counts[pre_block][i] += 1
                elif 0 <= xv < post_n:
                    counts[post_block][i] += 1

        return counts, x, max_block

    s2l_counts, x, _ = build_counts(s2l_rows)
    l2s_counts, _, _ = build_counts(l2s_rows)

    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    def plot_one(ax, counts, title):
        ax.axvline(0, color="k", ls="--", lw=1.5)
        ax.set_title(title)

        if counts is None:
            ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
            return

        # only keep the changing area around transition
        if zoom_only:
            mask = (x >= -zoom_window) & (x <= zoom_window)
            x_plot = x[mask]
        else:
            mask = np.ones_like(x, dtype=bool)
            x_plot = x

        for b in sorted(counts.keys()):
            y = counts[b][mask]

            # skip completely flat zero lines in zoomed plot
            if np.allclose(y, 0):
                continue

            ax.plot(
                x_plot,
                y,
                lw=2.5,
                marker="o",
                markersize=5,
                label=f"Block {b}"
            )

        ax.set_ylabel("Count")
        ax.legend(fontsize=8, ncol=3)
        style_axes_keep_xy(ax)

        if zoom_only:
            ax.set_xlim(-zoom_window, zoom_window)

    plot_one(axs[0], s2l_counts, "S→L: block counts near transition")
    plot_one(axs[1], l2s_counts, "L→S: block counts near transition")

    axs[1].set_xlabel("Aligned trial # around transition")

    plt.tight_layout()
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    print(f"Saved: {save_path}  trans6_8RowSummary_V_3.py:732 - 08_trans6_8RowSummary_V_3.py:731")

def make_8row_pdf(path, x, chemo_s2l, chemo_l2s, control_s2l, control_l2s, tgrid, summary_text, meta_all):
    fig, axs = plt.subplots(9, 2, figsize=(13, 27), sharex=False)
    fig.suptitle(format_summary_title(summary_text), fontsize=12, y=0.992)

    # Define colors
    CHEMO_COLOR = "blue"
    CONTROL_COLOR = "black"

    def draw_time_panel_dual(ax, chemo_d, control_d, title, mode, tgrid, bl_sub=False):
        def p(d, col, label):
            tr = []
            pos = []
            for k in ["traces_neg1", "traces_neg2", "traces1", "traces2"]:
                tr.extend(d.get(k, []))
            for k in ["pos_neg1", "pos_neg2", "pos1", "pos2"]:
                pos.extend(d.get(k, []))
            if tr:
                m, s = mean_sem_trace(tr, tgrid, bl_sub)
                ax.fill_between(tgrid, m - s, m + s, color=col, alpha=ALPHA_FILL, lw=0)
                ax.plot(tgrid, m, color=col, lw=1.4, label=label)

        p(chemo_d, CHEMO_COLOR, "Chemo")
        p(control_d, CONTROL_COLOR, "Control")

        _shade_background(ax, mode)
        ax.axvline(0, color="k", ls="--")
        ax.set_title(title)
        ax.set_xlim(TRACE_WIN)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8, loc="upper left")
        style_axes_keep_xy(ax)

    draw_time_panel_dual(axs[0, 0], chemo_s2l, control_s2l, "S->L FEC (Raw)", "s2l", tgrid, False)
    draw_time_panel_dual(axs[0, 1], chemo_l2s, control_l2s, "L->S FEC (Raw)", "l2s", tgrid, False)

    draw_time_panel_dual(axs[1, 0], chemo_s2l, control_s2l, "S->L FEC (BL-sub)", "s2l", tgrid, True)
    draw_time_panel_dual(axs[1, 1], chemo_l2s, control_l2s, "L->S FEC (BL-sub)", "l2s", tgrid, True)

    def wrap_vel(d):
        return {
            "traces_neg1": d["vtr_neg1"],
            "traces_neg2": d["vtr_neg2"],
            "traces1": d["vtr1"],
            "traces2": d["vtr2"],
            "pos_neg1": d["pos_neg1"],
            "pos_neg2": d["pos_neg2"],
            "pos1": d["pos1"],
            "pos2": d["pos2"],
        }

    draw_time_panel_dual(axs[2, 0], wrap_vel(chemo_s2l), wrap_vel(control_s2l), "S->L Velocity", "s2l", tgrid, False)
    draw_time_panel_dual(axs[2, 1], wrap_vel(chemo_l2s), wrap_vel(control_l2s), "L->S Velocity", "l2s", tgrid, False)

    def draw_trial_panel_dual(ax, x, chemo_stack, control_stack, title, mode, do_fit=False):
        ax.axvline(0, color="k", ls="--")

        def plot_stack(stack, col, label):
            if stack.size == 0:
                return
            mu, se, cnt, frac = mean_sem_counts(stack)
            mask = frac >= MIN_FRAC
            mu_p = smooth_nan(np.where(mask, mu, np.nan), SMOOTH_WINDOW)
            se_p = smooth_nan(np.where(mask, se, np.nan), SMOOTH_WINDOW)

            good = np.isfinite(mu_p) & np.isfinite(se_p)
            if np.any(good):
                ax.fill_between(x[good], mu_p[good] - se_p[good], mu_p[good] + se_p[good], color=col, alpha=0.20, lw=0)
                ax.plot(x[good], mu_p[good], color=col, alpha=0.9, lw=1.8, label=label)

            if do_fit and stack.size > 0:
                kind = "rise" if mode == "s2l" else "decay"
                _, _, yfit, info = _fit_stack_post0(x, stack, kind)
                if info["ok"]:
                    fit_col = darken(col, 0.3)
                    post = x >= 0
                    ax.plot(x[post], yfit[post], color=fit_col, lw=2.5)
                    px, py = info["plat_x"], info["plat_y"]
                    if x.min() <= px <= x.max():
                        ax.plot(px, py, "o", color="red", markersize=6, zorder=10)
                        ax.text(px, py, f" {px:.1f}", color="red", fontsize=9, fontweight="bold", va="bottom")
                    ax.text(0.05, 0.90, f"τ={info['tau']:.1f}", transform=ax.transAxes, fontsize=9)

        plot_stack(chemo_stack, CHEMO_COLOR, "Chemo")
        plot_stack(control_stack, CONTROL_COLOR, "Control")

        ax.set_title(title)
        ax.set_xlim(x.min(), x.max())
        ax.set_xticks(np.arange(x.min(), x.max() + 1, 5))
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8, loc="upper left")
        style_axes_keep_xy(ax)

    draw_trial_panel_dual(axs[3, 0], x, chemo_s2l["raw"], control_s2l["raw"], "S->L FEC @ 200 (Raw)", "s2l", False)
    draw_trial_panel_dual(axs[3, 1], x, chemo_l2s["raw"], control_l2s["raw"], "L->S FEC @ 200 (Raw)", "l2s", False)

    draw_trial_panel_dual(axs[4, 0], x, chemo_s2l["bs"], control_s2l["bs"], "S->L BL-Sub (Fit)", "s2l", True)
    draw_trial_panel_dual(axs[4, 1], x, chemo_l2s["bs"], control_l2s["bs"], "L->S BL-Sub (Fit)", "l2s", True)

    draw_trial_panel_dual(axs[5, 0], x, chemo_s2l["vel"], control_s2l["vel"], "S->L Velocity (Fit)", "s2l", True)
    draw_trial_panel_dual(axs[5, 1], x, chemo_l2s["vel"], control_l2s["vel"], "L->S Velocity (Fit)", "l2s", True)

    def draw_baseline_panel_dual(ax, x, chemo_stack, control_stack, title, mode):
        ax.axvline(0, color="k", ls="--")

        def plot_stack(stack, col, label):
            if stack.size == 0:
                return
            mu, se, cnt, frac = mean_sem_counts(stack)
            mask = frac >= MIN_FRAC
            mu_p = smooth_nan(np.where(mask, mu, np.nan), SMOOTH_WINDOW)
            se_p = smooth_nan(np.where(mask, se, np.nan), SMOOTH_WINDOW)

            good = np.isfinite(mu_p) & np.isfinite(se_p)
            if np.any(good):
                ax.fill_between(x[good], mu_p[good] - se_p[good], mu_p[good] + se_p[good], color=col, alpha=0.20, lw=0)
                ax.plot(x[good], mu_p[good], color=col, alpha=0.9, lw=1.8, label=label)

        plot_stack(chemo_stack, CHEMO_COLOR, "Chemo")
        plot_stack(control_stack, CONTROL_COLOR, "Control")

        ax.set_title(title)
        ax.set_xlim(x.min(), x.max())
        ax.set_xticks(np.arange(x.min(), x.max() + 1, 5))
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8, loc="upper left")
        style_axes_keep_xy(ax)

    draw_baseline_panel_dual(axs[6, 0], x, chemo_s2l["base"], control_s2l["base"], "S->L Baseline amplitude", "s2l")
    draw_baseline_panel_dual(axs[6, 1], x, chemo_l2s["base"], control_l2s["base"], "L->S Baseline amplitude", "l2s")

    draw_real_block_number_panel_dual(
        axs[7, 0],
        meta_all,
        direction="s2l",
        title="S→L block number across aligned trials",
    )

    draw_real_block_number_panel_dual(
        axs[7, 1],
        meta_all,
        direction="l2s",
        title="L→S block number across aligned trials",
    )

    draw_valid_transition_count_panel_dual(
        axs[8, 0],
        meta_all,
        direction="s2l",
        title="S→L valid transition count across aligned trials",
    )

    draw_valid_transition_count_panel_dual(
        axs[8, 1],
        meta_all,
        direction="l2s",
        title="L→S valid transition count across aligned trials",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.965])
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved: {path}  trans6_8RowSummary_V_3.py:803 - 08_trans6_8RowSummary_V_3.py:895")

def draw_valid_transition_count_panel(ax, meta_all, direction, title, count_window=60):
    rows = []
    for m in meta_all:
        rows.extend(m.get("kept_rows", []))
    rows = [r for r in rows if r["direction"] == direction]

    ax.axvline(0, color="k", ls="--", lw=1.5)
    ax.set_title(title)

    if not rows:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
        return

    # IMPORTANT:
    # use a larger range than FIXED_BLOCK_LEN so real block-length differences appear
    x = np.arange(-count_window, count_window + 1)
    counts = np.zeros(len(x), dtype=float)

    for i, xv in enumerate(x):
        n_here = 0
        for r in rows:
            pre_len = int(r["pre_len"])
            post_len = int(r["post_len"])

            # pre side: trial positions -pre_len ... -1 exist
            if -pre_len <= xv < 0:
                n_here += 1

            # post side: trial positions 0 ... post_len-1 exist
            elif 0 <= xv < post_len:
                n_here += 1

        counts[i] = n_here

    ax.plot(x, counts, lw=2.5, color="black")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel("N transitions")
    ax.set_xlabel("Aligned trial #")
    ymax = int(np.nanmax(counts)) if np.any(np.isfinite(counts)) else 0
    ax.set_yticks(np.arange(0, ymax + 1, 1))
    style_axes_keep_xy(ax)
    
def draw_block_length_panel(ax, meta_all, direction, title):
    rows = []
    for m in meta_all:
        rows.extend(m.get("kept_rows", []))
    rows = [r for r in rows if r["direction"] == direction]

    ax.set_title(title)

    if not rows:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
        return

    pre_lengths = np.array([float(r["pre_len"]) for r in rows], dtype=float)
    post_lengths = np.array([float(r["post_len"]) for r in rows], dtype=float)

    y_pre = np.zeros_like(pre_lengths, dtype=float)
    y_post = np.ones_like(post_lengths, dtype=float)

    # deterministic small jitter
    rng = np.random.default_rng(0)
    y_pre_j = y_pre + rng.uniform(-0.06, 0.06, size=len(pre_lengths))
    y_post_j = y_post + rng.uniform(-0.06, 0.06, size=len(post_lengths))

    # paired light lines
    for i in range(len(rows)):
        ax.plot([pre_lengths[i], post_lengths[i]], [y_pre_j[i], y_post_j[i]],
                color="0.75", lw=1, alpha=0.7, zorder=1)

    # points
    ax.scatter(pre_lengths, y_pre_j, s=28, color="tab:blue", alpha=0.85, zorder=2, label="Pre block")
    ax.scatter(post_lengths, y_post_j, s=28, color="tab:orange", alpha=0.85, zorder=2, label="Post block")

    # mean ± SEM
    pre_mu = np.nanmean(pre_lengths)
    post_mu = np.nanmean(post_lengths)
    pre_se = np.nanstd(pre_lengths, ddof=0) / np.sqrt(max(1, len(pre_lengths)))
    post_se = np.nanstd(post_lengths, ddof=0) / np.sqrt(max(1, len(post_lengths)))

    ax.errorbar([pre_mu, post_mu], [0, 1], xerr=[pre_se, post_se],
                fmt="o-", color="black", lw=2.2, capsize=4, zorder=3)

    ax.axvline(MIN_BLOCK_LEN, color="crimson", ls="--", lw=1.5, alpha=0.9, label=f"Minimum = {MIN_BLOCK_LEN} trials")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Pre block", "Post block"])
    ax.set_xlabel("# trials in block")
    ax.legend(fontsize=8, loc="best")

def make_block_length_pairs_pdf(save_path, meta_all, summary_text=None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), sharex=False, sharey=False)

    if summary_text:
        fig.suptitle(summary_text, fontsize=12, y=0.98)

    draw_block_length_panel(
        axs[0],
        meta_all,
        direction="s2l",
        title="S→L number of trials in pre/post block",
    )
    draw_block_length_panel(
        axs[1],
        meta_all,
        direction="l2s",
        title="L→S number of trials in pre/post block",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94] if summary_text else None)
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved: {save_path}  trans6_8RowSummary_V_3.py:915 - 08_trans6_8RowSummary_V_3.py:1010")


def draw_real_block_number_panel(ax, meta_all, direction, title):
    rows = []
    for m in meta_all:
        rows.extend(m.get("kept_rows", []))
    rows = [r for r in rows if r["direction"] == direction]

    ax.axvline(0, color="k", ls="--", lw=1.5)
    ax.set_title(title)

    if not rows:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(-FIXED_BLOCK_LEN, FIXED_BLOCK_LEN)
    stack = []

    for r in rows:
        pre_block = float(r["pre_block_num"])
        post_block = float(r["post_block_num"])
        pre_n = int(r["pre_take_n"])
        post_n = int(r["post_take_n"])

        y = np.full(len(x), np.nan)

        for i, xv in enumerate(x):
            if -pre_n <= xv < 0:
                y[i] = pre_block
            elif 0 <= xv < post_n:
                y[i] = post_block

        stack.append(y)

    M = np.vstack(stack)
    cnt = np.sum(np.isfinite(M), axis=0)
    mu = np.nanmean(M, axis=0)
    se = np.nanstd(M, axis=0, ddof=0) / np.sqrt(np.maximum(1, cnt))

    good = np.isfinite(mu) & np.isfinite(se)
    if np.any(good):
        ax.fill_between(x[good], mu[good] - se[good], mu[good] + se[good],
                        color="tab:purple", alpha=0.22, lw=0)
        ax.plot(x[good], mu[good], color="tab:purple", lw=2.5)

    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel("Aligned trial #")
    ax.set_ylabel("Mean Block #")
    style_axes_keep_xy(ax)


def draw_real_block_number_panel_dual(ax, meta_all, direction, title):
    rows = []
    for m in meta_all:
        rows.extend(m.get("kept_rows", []))
    rows = [r for r in rows if r["direction"] == direction]

    chemo_rows = [r for r in rows if "Opto" in r.get("session_name", "")]
    control_rows = [r for r in rows if "Opto" not in r.get("session_name", "")]

    ax.axvline(0, color="k", ls="--", lw=1.5)
    ax.set_title(title)

    if not chemo_rows and not control_rows:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(-FIXED_BLOCK_LEN, FIXED_BLOCK_LEN)

    def plot_rows(rows, color, label):
        if not rows:
            return
        stack = []
        for r in rows:
            pre_block = float(r["pre_block_num"])
            post_block = float(r["post_block_num"])
            pre_n = int(r["pre_take_n"])
            post_n = int(r["post_take_n"])
            y = np.full(len(x), np.nan)
            for i, xv in enumerate(x):
                if -pre_n <= xv < 0:
                    y[i] = pre_block
                elif 0 <= xv < post_n:
                    y[i] = post_block
            stack.append(y)

        M = np.vstack(stack)
        cnt = np.sum(np.isfinite(M), axis=0)
        mu = np.nanmean(M, axis=0)
        se = np.nanstd(M, axis=0, ddof=0) / np.sqrt(np.maximum(1, cnt))
        good = np.isfinite(mu) & np.isfinite(se)
        if np.any(good):
            ax.fill_between(x[good], mu[good] - se[good], mu[good] + se[good], color=color, alpha=0.18, lw=0)
            ax.plot(x[good], mu[good], color=color, lw=2.2, label=label)

    plot_rows(control_rows, "black", "Control")
    plot_rows(chemo_rows, "blue", "Chemo")

    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel("Aligned trial #")
    ax.set_ylabel("Mean Block #")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc="upper left")
    style_axes_keep_xy(ax)


def draw_valid_transition_count_panel_dual(ax, meta_all, direction, title, count_window=60):
    rows = []
    for m in meta_all:
        rows.extend(m.get("kept_rows", []))
    rows = [r for r in rows if r["direction"] == direction]

    chemo_rows = [r for r in rows if "Opto" in r.get("session_name", "")]
    control_rows = [r for r in rows if "Opto" not in r.get("session_name", "")]

    ax.axvline(0, color="k", ls="--", lw=1.5)
    ax.set_title(title)

    x = np.arange(-count_window, count_window + 1)

    def count_rows(rows):
        counts = np.zeros(len(x), dtype=float)
        for i, xv in enumerate(x):
            n_here = 0
            for r in rows:
                pre_len = int(r["pre_len"])
                post_len = int(r["post_len"])
                if -pre_len <= xv < 0:
                    n_here += 1
                elif 0 <= xv < post_len:
                    n_here += 1
            counts[i] = n_here
        return counts

    if not chemo_rows and not control_rows:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
        return

    if control_rows:
        control_counts = count_rows(control_rows)
        ax.plot(x, control_counts, color="black", lw=2.2, label="Control")
    if chemo_rows:
        chemo_counts = count_rows(chemo_rows)
        ax.plot(x, chemo_counts, color="blue", lw=2.2, label="Chemo")

    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel("N transitions")
    ax.set_xlabel("Aligned trial #")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc="upper left")
    style_axes_keep_xy(ax)


def draw_block_length_aligned_panel(ax, meta_all, direction, title):
    rows = []
    for m in meta_all:
        rows.extend(m.get("kept_rows", []))
    rows = [r for r in rows if r["direction"] == direction]

    ax.axvline(0, color="k", ls="--", lw=1.5)
    ax.set_title(title)

    if not rows:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(-FIXED_BLOCK_LEN, FIXED_BLOCK_LEN)
    stack = []

    for r in rows:
        pre_len = float(r["pre_len"])
        post_len = float(r["post_len"])
        y = np.full(len(x), np.nan)

        for i, xv in enumerate(x):
            if -pre_len <= xv < 0:
                y[i] = pre_len
            elif 0 <= xv < post_len:
                y[i] = post_len

        stack.append(y)

    M = np.vstack(stack)
    cnt = np.sum(np.isfinite(M), axis=0)
    mu = np.nanmean(M, axis=0)
    se = np.nanstd(M, axis=0, ddof=0) / np.sqrt(np.maximum(1, cnt))

    good = np.isfinite(mu) & np.isfinite(se)
    if np.any(good):
        ax.fill_between(
            x[good],
            mu[good] - se[good],
            mu[good] + se[good],
            color="tab:orange",
            alpha=0.22,
            lw=0,
        )
        ax.plot(x[good], mu[good], color="tab:orange", lw=2.5)

    ax.axhline(MIN_BLOCK_LEN, color="crimson", ls="--", lw=1.5, alpha=0.9)
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel("Aligned trial #")
    ax.set_ylabel("Number of trials for each block")
                
# ============================================================
# Metadata / summary writers
# ============================================================
def write_transition_metadata(save_path, meta_all):
    with open(save_path, "w") as f:
        for m in meta_all:
            f.write(f"Session: {m['session_name']}\n")
            f.write(f"  Inferred family: {m['family']}\n")
            f.write("  S->L kept transitions:\n")
            for i, ((pre_len, post_len), ((ps, pe), (ns, ne))) in enumerate(
                zip(m["s2l_lengths"], m["s2l_trial_ranges"]), start=1
            ):
                f.write(f"    #{i}: pre={pre_len} [{ps}-{pe}] | post={post_len} [{ns}-{ne}]\n")
            f.write("  L->S kept transitions:\n")
            for i, ((pre_len, post_len), ((ps, pe), (ns, ne))) in enumerate(
                zip(m["l2s_lengths"], m["l2s_trial_ranges"]), start=1
            ):
                f.write(f"    #{i}: pre={pre_len} [{ps}-{pe}] | post={post_len} [{ns}-{ne}]\n")
            if m["skipped"]:
                f.write("  Skipped transitions:\n")
                for s in m["skipped"]:
                    f.write(f"    {s}\n")
            f.write("\n")

def write_transition_rows_csv(save_path, meta_all):
    rows = []
    for m in meta_all:
        rows.extend(m.get("kept_rows", []))

    if not rows:
        with open(save_path, "w", newline="") as f:
            f.write("no rows\n")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def make_session_block_timeline_pdf(save_path, meta_all):
    rows = []
    for m in meta_all:
        session_name = m["session_name"]
        blocks = m.get("all_blocks", [])
        if not blocks:
            continue
        rows.append((session_name, blocks))

    if not rows:
        print(f"No session block data for {save_path}  trans6_8RowSummary_V_3.py:1067 - 08_trans6_8RowSummary_V_3.py:1163")
        return

    fig_h = max(6, 0.7 * len(rows) + 2)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    y_base = np.arange(len(rows), 0, -1)

    for y0, (session_name, blocks) in zip(y_base, rows):
        for blk in blocks:
            bnum = blk["block_num"]
            x0 = blk["start"] + 1
            x1 = blk["end"] + 1
            ax.plot([x0, x1], [y0, y0], lw=4, solid_capstyle="butt")
            xm = 0.5 * (x0 + x1)
            ax.text(xm, y0 + 0.12, f"B{bnum}", ha="center", va="bottom", fontsize=8)
            ax.plot([x0, x0], [y0 - 0.18, y0 + 0.18], lw=1.5, color="k", alpha=0.45)

        last_end = blocks[-1]["end"] + 1
        ax.plot([last_end, last_end], [y0 - 0.18, y0 + 0.18], lw=1.5, color="k", alpha=0.45)

    ax.set_yticks(y_base)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Session")
    ax.set_title("Block number across full session trials")
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved: {save_path}  trans6_8RowSummary_V_3.py:1098 - 08_trans6_8RowSummary_V_3.py:1194")

def make_block_vs_trial_mean_pdf(save_path, meta_all):
    all_traces = []
    max_len = 0

    for m in meta_all:
        blocks = m.get("all_blocks", [])
        if not blocks:
            continue

        last_trial = max(b["end"] for b in blocks) + 1
        trace = np.full(last_trial, np.nan)
        for blk in blocks:
            bnum = blk["block_num"]
            trace[blk["start"]:blk["end"] + 1] = bnum

        all_traces.append(trace)
        max_len = max(max_len, len(trace))

    if not all_traces:
        print(f"No data for {save_path}  trans6_8RowSummary_V_3.py:1119 - 08_trans6_8RowSummary_V_3.py:1215")
        return

    padded = []
    for tr in all_traces:
        arr = np.full(max_len, np.nan)
        arr[:len(tr)] = tr
        padded.append(arr)

    M = np.vstack(padded)
    mu = np.nanmean(M, axis=0)
    cnt = np.sum(np.isfinite(M), axis=0)
    se = np.nanstd(M, axis=0) / np.sqrt(np.maximum(cnt, 1))

    x = np.arange(max_len)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(x, mu - se, mu + se, alpha=0.25)
    ax.plot(x, mu, lw=2)
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Mean Block #")
    ax.set_title("Mean Block Number vs Trial (Pooled Sessions)")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved: {save_path}  trans6_8RowSummary_V_3.py:1146 - 08_trans6_8RowSummary_V_3.py:1242")


# ============================================================
# Main per-session processing
# ============================================================
def collect_session(p):
    trials = load_trials_from_mat(p)
    overall = session_overall_max_eye(trials)
    if not (np.isfinite(overall) and overall > 0):
        return None

    amps_raw, amps_bs, vels, baselines, rel_ts, fecs = [], [], [], [], [], []
    blks = []

    for tr in trials:
        a, b, v, base, rt, y = per_trial_metrics(tr, overall)

        if np.isfinite(base) and base > BASELINE_THRESHOLD:
            a = np.nan
            b = np.nan
            v = np.nan
            base = np.nan
            rt = np.array([])
            y = np.array([])

        amps_raw.append(a)
        amps_bs.append(b)
        vels.append(v)
        baselines.append(base)
        rel_ts.append(rt)
        fecs.append(y)

        d = tr["AP_on"] - tr["LED_on"]
        blks.append(classify_block(d))

    amps_raw = np.asarray(amps_raw, dtype=float)
    amps_bs = np.asarray(amps_bs, dtype=float)
    vels = np.asarray(vels, dtype=float)
    baselines = np.asarray(baselines, dtype=float)

    raw_blocks = build_blocks(blks)
    eff_blocks = apply_first_long_warmup_removal(raw_blocks)

    filtered_blocks = [dict(b) for b in eff_blocks if b["label"] in ("short", "long")]
    for block_num, blk in enumerate(filtered_blocks, start=1):
        blk["block_num"] = block_num

    family = infer_session_family(filtered_blocks)
    low, high = family_bounds(family)
    if low is None:
        return None

    L = 2 * FIXED_BLOCK_LEN
    tgrid = np.arange(TRACE_WIN[0], TRACE_WIN[1] + TRACE_STEP, TRACE_STEP)

    res = {
        "s2l": {"raw": [], "bs": [], "vel": [], "base": [], "traces_neg1": [], "traces_neg2": [], "traces1": [], "traces2": [], "vtr_neg1": [], "vtr_neg2": [], "vtr1": [], "vtr2": [], "pos_neg1": [], "pos_neg2": [], "pos1": [], "pos2": []},
        "l2s": {"raw": [], "bs": [], "vel": [], "base": [], "traces_neg1": [], "traces_neg2": [], "traces1": [], "traces2": [], "vtr_neg1": [], "vtr_neg2": [], "vtr1": [], "vtr2": [], "pos_neg1": [], "pos_neg2": [], "pos1": [], "pos2": []},
    }

    meta = {
        "session_name": p.name,
        "family": family,
        "s2l_lengths": [],
        "l2s_lengths": [],
        "s2l_trial_ranges": [],
        "l2s_trial_ranges": [],
        "skipped": [],
        "kept_rows": [],
        "all_blocks": [dict(blk) for blk in filtered_blocks],
    }

    for bi in range(len(filtered_blocks) - 1):
        pre_blk = filtered_blocks[bi]
        post_blk = filtered_blocks[bi + 1]
        b0, b1 = pre_blk["label"], post_blk["label"]

        if b0 == b1:
            continue

        pre_len = pre_blk["length"]
        post_len = post_blk["length"]

        if not (low <= pre_len <= high and low <= post_len <= high):
            meta["skipped"].append(
                f"{b0}->{b1}: pre={pre_len} [{pre_blk['start']}-{pre_blk['end']}] | "
                f"post={post_len} [{post_blk['start']}-{post_blk['end']}]"
            )
            continue

        info_tuple = (pre_len, post_len)
        range_tuple = ((pre_blk["start"], pre_blk["end"]), (post_blk["start"], post_blk["end"]))

        if b0 == "short" and b1 == "long":
            meta["s2l_lengths"].append(info_tuple)
            meta["s2l_trial_ranges"].append(range_tuple)
            d = res["s2l"]
            direction = "s2l"
        elif b0 == "long" and b1 == "short":
            meta["l2s_lengths"].append(info_tuple)
            meta["l2s_trial_ranges"].append(range_tuple)
            d = res["l2s"]
            direction = "l2s"
        else:
            continue

        pre_trials = list(range(pre_blk["start"], pre_blk["end"] + 1))
        post_trials = list(range(post_blk["start"], post_blk["end"] + 1))

        pre_take = pre_trials[-FIXED_BLOCK_LEN:]
        post_take = post_trials[:FIXED_BLOCK_LEN]

        r_row = np.full(L, np.nan)
        b_row = np.full(L, np.nan)
        v_row = np.full(L, np.nan)
        base_row = np.full(L, np.nan)

        pre_dst_start = FIXED_BLOCK_LEN - len(pre_take)
        for j, idx in enumerate(pre_take):
            dst = pre_dst_start + j
            r_row[dst] = amps_raw[idx]
            b_row[dst] = amps_bs[idx]
            v_row[dst] = vels[idx]
            base_row[dst] = baselines[idx]

        for j, idx in enumerate(post_take):
            dst = FIXED_BLOCK_LEN + j
            r_row[dst] = amps_raw[idx]
            b_row[dst] = amps_bs[idx]
            v_row[dst] = vels[idx]
            base_row[dst] = baselines[idx]

        def get_aligned_value(arr, aligned_trial):
            idx = aligned_trial + FIXED_BLOCK_LEN
            if 0 <= idx < len(arr):
                return arr[idx]
            return np.nan

        meta["kept_rows"].append({
            "session_name": p.name,
            "family": family,
            "direction": direction,
            "pre_block_num": pre_blk["block_num"],
            "post_block_num": post_blk["block_num"],
            "pre_block_label": b0,
            "post_block_label": b1,
            "pre_start": pre_blk["start"],
            "pre_end": pre_blk["end"],
            "post_start": post_blk["start"],
            "post_end": post_blk["end"],
            "pre_len": pre_len,
            "post_len": post_len,
            "pre_take_start": pre_take[0] if len(pre_take) else np.nan,
            "pre_take_end": pre_take[-1] if len(pre_take) else np.nan,
            "post_take_start": post_take[0] if len(post_take) else np.nan,
            "post_take_end": post_take[-1] if len(post_take) else np.nan,
            "pre_take_n": len(pre_take),
            "post_take_n": len(post_take),
            "raw_at_10": get_aligned_value(r_row, 10),
            "raw_at_15": get_aligned_value(r_row, 15),
            "raw_at_20": get_aligned_value(r_row, 20),
            "bs_at_10": get_aligned_value(b_row, 10),
            "bs_at_15": get_aligned_value(b_row, 15),
            "bs_at_20": get_aligned_value(b_row, 20),
            "base_at_10": get_aligned_value(base_row, 10),
            "base_at_15": get_aligned_value(base_row, 15),
            "base_at_20": get_aligned_value(base_row, 20),
        })

        def get_trace(k_idx):
            if 0 <= k_idx < len(rel_ts):
                rt, y = rel_ts[k_idx], fecs[k_idx]
                if rt.size > 1:
                    yi = np.interp(tgrid, rt, y, left=np.nan, right=np.nan)
                    vi = np.full_like(yi, np.nan)
                    ok = np.isfinite(yi)
                    if ok.sum() > 3:
                        vi[ok] = np.gradient(yi[ok], tgrid[ok])
                    return yi, vi
            return None, None

        def add_traces_from_trials(target, idx_list, pos_target, pos_values):
            for idx, pos in zip(idx_list, pos_values):
                f, v = get_trace(idx)
                if f is not None:
                    target[0].append(f)
                    target[1].append(v)
                    pos_target.append(pos)

        pre_mid = len(pre_take) // 2
        post_mid = len(post_take) // 2

        d["raw"].append(r_row)
        d["bs"].append(b_row)
        d["vel"].append(v_row)
        d["base"].append(base_row)

        pre_pos = list(range(-len(pre_take), 0))
        post_pos = list(range(1, len(post_take) + 1))

        add_traces_from_trials((d["traces_neg1"], d["vtr_neg1"]), pre_take[:pre_mid], d["pos_neg1"], pre_pos[:pre_mid])
        add_traces_from_trials((d["traces_neg2"], d["vtr_neg2"]), pre_take[pre_mid:], d["pos_neg2"], pre_pos[pre_mid:])
        add_traces_from_trials((d["traces1"], d["vtr1"]), post_take[:post_mid], d["pos1"], post_pos[:post_mid])
        add_traces_from_trials((d["traces2"], d["vtr2"]), post_take[post_mid:], d["pos2"], post_pos[post_mid:])

    return res, tgrid, meta


# ============================================================
# Run
# ============================================================
def run():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    def init_pool():
        return {
            "s2l": {"raw": [], "bs": [], "vel": [], "base": [], "traces_neg1": [], "traces_neg2": [], "traces1": [], "traces2": [], "vtr_neg1": [], "vtr_neg2": [], "vtr1": [], "vtr2": [], "pos_neg1": [], "pos_neg2": [], "pos1": [], "pos2": []},
            "l2s": {"raw": [], "bs": [], "vel": [], "base": [], "traces_neg1": [], "traces_neg2": [], "traces1": [], "traces2": [], "vtr_neg1": [], "vtr_neg2": [], "vtr1": [], "vtr2": [], "pos_neg1": [], "pos_neg2": [], "pos1": [], "pos2": []},
        }

    chemo_pool = init_pool()
    control_pool = init_pool()

    x = np.arange(-FIXED_BLOCK_LEN, FIXED_BLOCK_LEN)

    def finalize(d):
        out = {}
        for k in ["raw", "bs", "vel", "base"]:
            out[k] = np.vstack(d[k]) if d[k] else np.empty((0, len(x)))
        for k in ["traces_neg1", "traces_neg2", "traces1", "traces2", "vtr_neg1", "vtr_neg2", "vtr1", "vtr2", "pos_neg1", "pos_neg2", "pos1", "pos2"]:
            out[k] = d[k]
        return out

    all_meta = []

    for mouse, root in MOUSE_ROOTS.items():
        print(f"Processing {mouse}...  trans6_8RowSummary_V_3.py:1379 - 08_trans6_8RowSummary_V_3.py:1479")

        m_chemo_acc = init_pool()
        m_control_acc = init_pool()

        files = sorted(root.glob("*_EBC_*.mat"))
        tgrid_ref = None
        meta_all = []

        for p in files:
            if DATE_MIN is not None:
                dt = parse_date_from_filename(p.name)
                if dt is None or dt < DATE_MIN:
                    continue

            try:
                out = collect_session(p)
                if out is None:
                    continue

                data, tg, meta = out
                if tgrid_ref is None:
                    tgrid_ref = tg

                meta_all.append(meta)
                all_meta.append(meta)

                target_acc = m_chemo_acc if is_chemo_session(p) else m_control_acc
                for direction in ["s2l", "l2s"]:
                    for key in target_acc[direction]:
                        target_acc[direction][key].extend(data[direction][key])

            except Exception as e:
                print(f"Skipped {p.name}: {e}  trans6_8RowSummary_V_3.py:1413 - 08_trans6_8RowSummary_V_3.py:1509")

        if tgrid_ref is None:
            continue

        write_transition_metadata(SAVE_DIR / f"{mouse}_transition_metadata.txt", meta_all)
        write_transition_rows_csv(SAVE_DIR / f"{mouse}_transition_rows.csv", meta_all)
        make_session_block_timeline_pdf(SAVE_DIR / f"{mouse}_Session_BlockTimeline.pdf", meta_all)

        m_chemo_s2l = finalize(m_chemo_acc["s2l"])
        m_chemo_l2s = finalize(m_chemo_acc["l2s"])
        m_control_s2l = finalize(m_control_acc["s2l"])
        m_control_l2s = finalize(m_control_acc["l2s"])

        # Add to global pools
        for direction, src in [("s2l", m_chemo_s2l), ("l2s", m_chemo_l2s)]:
            for k in ["raw", "bs", "vel", "base"]:
                if src[k].size:
                    chemo_pool[direction][k].append(src[k])
            for k in ["traces_neg1", "traces_neg2", "traces1", "traces2", "vtr_neg1", "vtr_neg2", "vtr1", "vtr2", "pos_neg1", "pos_neg2", "pos1", "pos2"]:
                chemo_pool[direction][k].extend(src[k])

        for direction, src in [("s2l", m_control_s2l), ("l2s", m_control_l2s)]:
            for k in ["raw", "bs", "vel", "base"]:
                if src[k].size:
                    control_pool[direction][k].append(src[k])
            for k in ["traces_neg1", "traces_neg2", "traces1", "traces2", "vtr_neg1", "vtr_neg2", "vtr1", "vtr2", "pos_neg1", "pos_neg2", "pos1", "pos2"]:
                control_pool[direction][k].extend(src[k])

        n_sessions = len(meta_all)
        n_s2l = sum(len(m["s2l_lengths"]) for m in meta_all)
        n_l2s = sum(len(m["l2s_lengths"]) for m in meta_all)
        fams = sorted(set(m["family"] for m in meta_all if m["family"] is not None))

        summary = (
            f"Within-block transition summary | Sessions={n_sessions} | "
            f"S->L kept={n_s2l} | L->S kept={n_l2s} | "
            f"Fixed aligned length={FIXED_BLOCK_LEN} | Families={fams} | "
            f"Baseline>{BASELINE_THRESHOLD} excluded"
        )

        make_8row_pdf(
            SAVE_DIR / f"{mouse}_Transitions_Summary_FIXED.pdf",
            x, m_chemo_s2l, m_chemo_l2s, m_control_s2l, m_control_l2s, tgrid_ref, summary, meta_all
        )

        make_block_count_lines_pdf(
            SAVE_DIR / f"{mouse}_BlockCount_Lines_Zoom.pdf",
            meta_all,
            zoom_only=True,
            zoom_window=3
        )

    print("Processing pooled...  trans6_8RowSummary_V_3.py:1456 - 08_trans6_8RowSummary_V_3.py:1570")

    p_chemo_s2l = finalize(chemo_pool["s2l"])
    p_chemo_l2s = finalize(chemo_pool["l2s"])
    p_control_s2l = finalize(control_pool["s2l"])
    p_control_l2s = finalize(control_pool["l2s"])
    tg = np.arange(TRACE_WIN[0], TRACE_WIN[1] + TRACE_STEP, TRACE_STEP)

    write_transition_metadata(SAVE_DIR / "POOLED_transition_metadata.txt", all_meta)
    write_transition_rows_csv(SAVE_DIR / "POOLED_transition_rows.csv", all_meta)
    make_session_block_timeline_pdf(SAVE_DIR / "POOLED_Session_BlockTimeline.pdf", all_meta)
    make_block_vs_trial_mean_pdf(SAVE_DIR / "POOLED_BlockVsTrial_MEAN.pdf", all_meta)

    pooled_summary = (
        f"POOLED within-block transition summary | Sessions={len(all_meta)} | "
        f"S->L kept={sum(len(m['s2l_lengths']) for m in all_meta)} | "
        f"L->S kept={sum(len(m['l2s_lengths']) for m in all_meta)} | "
        f"Fixed aligned length={FIXED_BLOCK_LEN} | "
        f"Baseline>{BASELINE_THRESHOLD} excluded"
    )

    make_8row_pdf(
        SAVE_DIR / "POOLED_Transitions_Summary_FIXED.pdf",
        x, p_chemo_s2l, p_chemo_l2s, p_control_s2l, p_control_l2s, tg, pooled_summary, all_meta
    )


if __name__ == "__main__":
    run()
