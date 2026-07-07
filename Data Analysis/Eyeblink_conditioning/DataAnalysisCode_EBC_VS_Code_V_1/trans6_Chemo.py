# Transitions: 6-Row Summary (Time + Trials with Fits on BL-sub/Vel)
from __future__ import annotations
import pathlib, re, warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.signal import savgol_filter

# ─────────────────────────────────────────────────────────────
# Matplotlib config
# ─────────────────────────────────────────────────────────────
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["pdf.use14corefonts"] = False

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
MOUSE_ROOTS: Dict[str, pathlib.Path] = {
    "SM09": pathlib.Path("/Users/zahra/Desktop/NajafiLab/EBC_Analysis/SM09_Summary"),
    
}
SAVE_DIR = pathlib.Path("/Users/zahra/Desktop/NajafiLab/EBC_Analysis/SM09_Summary/SM09_Transition_Trial_By_Trial_Adaptation")

# Optional date filter (YYYYMMDD) or None
DATE_MIN: Optional[int] = None

# ─────────────────────────────────────────────────────────────
# Analysis params
# ─────────────────────────────────────────────────────────────
N_PRE  = 50
N_POST = 50

# Gates
MIN_POST_TRIALS_TRACES  = 40  # lenient gate

# Windows (ms)
RESP_WIN      = (170, 210)
BASELINE_WIN  = (-100, 0)

# Delay classification
SHORT_DELAY = (150, 300)
LONG_DELAY  = (350, 550)

# Min fraction of trials for valid mean
MIN_FRAC = 0.4

# Time-domain resampling
TRACE_WIN  = (-100, 600)
TRACE_STEP = 1

# Colors
COLOR_SHORT = "blue"
COLOR_LONG  = "lime"
BASE_NEG = "tab:green"
BASE_POS = "tab:blue"

def _blend(c1, c2, t: float):
    a = np.array(to_rgb(c1)); b = np.array(to_rgb(c2))
    return tuple((1 - t) * a + t * b)

def lighten(color, t=0.35):  return _blend(color, "white", t)
def darken(color, t=0.20):   return _blend(color, "black", t)

# Time-domain hues
COLOR_TRACE_NEG_1 = lighten(BASE_NEG, 0.35)
COLOR_TRACE_NEG_2 = darken(BASE_NEG, 0.18)
COLOR_TRACE_1     = lighten(BASE_POS, 0.35)
COLOR_TRACE_2     = darken(BASE_POS, 0.18)

LINEWIDTH   = 1.5
ALPHA_FILL  = 0.25
FIGSIZE     = (16, 20)  # Taller for 6 rows
PDF_DPI     = 300
OSCILLATION_SMOOTH_WIN = 15
OSCILLATION_SMOOTH_POLY = 3
OSCILLATION_BASELINE_STD_THR = 0.02

POS1_STOP   = 7
EXPECTED_SHORT = (200, 220)
EXPECTED_LONG  = (400, 420)
TRACE_SMOOTH_WIN = 21
TRACE_SMOOTH_POLY = 3
TRACE_BASELINE_STD_MAX = 0.04
TRACE_RESIDUAL_RATIO_MAX = 0.18

CONTROL_EPOCH_COLORS = {
    "traces_neg1": "0.75",
    "traces_neg2": "0.45",
    "traces1": "k",
    "traces2": "0.25",
}

CHEMO_EPOCH_COLORS = {
    "traces_neg1": "#9ecae1",
    "traces_neg2": "#6baed6",
    "traces1": "#08519c",
    "traces2": "#3182bd",
}

EPOCH_LABELS = {
    "traces_neg1": "Prev block avg: -50..-26",
    "traces_neg2": "Prev block avg: -25..-1",
    "traces1": "New block avg: 1..7",
    "traces2": "New block avg: 8..50",
}

def epoch_labels_for_mode(mode: str) -> Dict[str, str]:
    prev_block = "Short" if mode == "s2l" else "Long"
    new_block = "Long" if mode == "s2l" else "Short"
    return {
        "traces_neg1": f"{prev_block} block avg: -50..-26",
        "traces_neg2": f"{prev_block} block avg: -25..-1",
        "traces1": f"{new_block} block avg: 1..7",
        "traces2": f"{new_block} block avg: 8..50",
    }

def smooth_and_filter_trace(y: np.ndarray, tgrid: np.ndarray) -> Optional[np.ndarray]:
    """
    Smooth interpolated traces and reject strongly oscillatory traces so the
    epoch averages reflect the block dynamics rather than frame-to-frame noise.
    """
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 11:
        return None

    seg = y[ok]
    if seg.size >= 7:
        win = min(TRACE_SMOOTH_WIN, seg.size if seg.size % 2 == 1 else seg.size - 1)
        win = max(5, win)
        if win > TRACE_SMOOTH_POLY:
            seg_smooth = savgol_filter(seg, win, TRACE_SMOOTH_POLY, mode="interp")
        else:
            seg_smooth = seg.copy()
    else:
        seg_smooth = seg.copy()

    resid_std = float(np.nanstd(seg - seg_smooth))
    signal_span = float(np.nanpercentile(seg_smooth, 95) - np.nanpercentile(seg_smooth, 5))
    signal_span = max(signal_span, 1e-6)

    baseline_mask = ok & (tgrid >= BASELINE_WIN[0]) & (tgrid <= BASELINE_WIN[1])
    baseline_std = np.nan
    if np.any(baseline_mask):
        baseline_std = float(np.nanstd(y[baseline_mask]))

    if np.isfinite(baseline_std) and baseline_std > TRACE_BASELINE_STD_MAX:
        return None
    if resid_std / signal_span > TRACE_RESIDUAL_RATIO_MAX:
        return None

    out = np.full_like(y, np.nan, dtype=float)
    out[ok] = seg_smooth
    return out

# ─────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ─────────────────────────────────────────────────────────────
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
    if not isinstance(d, dict): return default
    keys = list(d.keys())
    for n in names:
        if n in d: return d[n]
        if isinstance(n, str):
            nl = n.lower()
            for k in keys:
                if isinstance(k, str) and k.lower() == nl:
                    return d[k]
    return default

def first_array(*candidates) -> np.ndarray:
    for c in candidates:
        if c is None: continue
        try:
            a = np.asarray(c).ravel()
            if a.size > 0:
                return a
        except Exception:
            pass
    return np.array([])

def array_ms(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a).ravel()
    return (a * 1000.0) if a.size else a

def parse_date_from_filename(name: str) -> Optional[int]:
    m = re.search(r"_(20\d{6})_", name)
    return int(m.group(1)) if m else None

# ─────────────────────────────────────────────────────────────
# TRIALS & METRICS
# ─────────────────────────────────────────────────────────────
def load_trials_from_mat(p: pathlib.Path) -> List[dict]:
    raw = sio.loadmat(p, struct_as_record=False, squeeze_me=True)
    raw = {k: to_py(v) for k, v in raw.items() if not k.startswith("__")}
    SD = to_py(get_ci(raw, "SessionData"))
    if not isinstance(SD, dict): raise RuntimeError("No SessionData")
    chemo_flag = int(get_ci(SD, "Chemogenetics", default=0) or 0)    
    
    RE = get_ci(SD, "RawEvents", "rawevents")
    Trials = get_ci(RE, "Trial") if RE else get_ci(SD, "Trial")
    if Trials is None: raise RuntimeError("No Trials found")

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
        ap_on  = first_array(get_ci(ev, "GlobalTimer2_Start"))
        ap_off = first_array(get_ci(ev, "GlobalTimer2_End"))
        fec_t = first_array(get_ci(dat, "FECTimes", "FEC_time", "fec_time"))
        eye   = first_array(get_ci(dat, "eyeAreaPixels", "EyeAreaPixels", "eye_area"))
        is_probe = bool(np.asarray(get_ci(dat, "IsProbeTrial", default=0)).squeeze())
        
        isi = get_ci(dat, "ISI")
        block_type = get_ci(dat, "BlockType")

        out.append(dict(
            LED_on=float(led_on[0])*1000.0 if led_on.size else np.nan,
            AP_on=float(ap_on[0])*1000.0 if ap_on.size else np.nan,
            AP_off=float(ap_off[0])*1000.0 if ap_off.size else np.nan,
            FECTimes=array_ms(fec_t) if fec_t.size else np.array([]),
            eye=eye,
            is_probe=is_probe,
            ISI=float(isi) * 1000.0 if isi is not None and np.isfinite(isi) else np.nan,
            BlockType=str(block_type).strip().lower() if block_type is not None else None
        ))
    return out, chemo_flag

def session_overall_max_eye(trials: List[dict]) -> float:
    vals = []
    per_trial_max = []
    for tr in trials:
        eye = np.asarray(tr.get("eye", []), dtype=float)
        eye = eye[np.isfinite(eye)]
        if eye.size:
            vals.append(eye)
            per_trial_max.append(np.nanmax(eye))
    if not vals: return np.nan
    allv = np.concatenate(vals)
    mx, mn = np.nanmax(allv), np.nanmin(allv)
    if mx <= 1.2 and mn >= -0.2: return 1.0
    p99 = np.nanpercentile(allv, 99)
    if p99 > 0: return p99
    if per_trial_max: return np.nanmax(per_trial_max)
    return np.nan

def per_trial_metrics(tr: dict, overall_max: float):
    led, t, eye = tr["LED_on"], tr["FECTimes"], tr["eye"]
    if not (np.isfinite(led) and t.size >= 2 and eye.size >= 2):
        return np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([])
    
    # Normalize FEC
    fec = 1.0 - (eye / overall_max) if (np.isfinite(overall_max) and overall_max > 0 and overall_max != 1.0) else eye
    fec = np.clip(fec, -0.5, 1.5)
    
    m = min(fec.size, t.size)
    fec, t = fec[:m], t[:m]
    
    # Sort time
    order = np.argsort(t)
    t, fec = t[order], fec[order]
    rel_t = t - led

    base = window_mean(rel_t, fec, BASELINE_WIN)
    resp = window_mean(rel_t, fec, RESP_WIN)
    amp_raw = resp
    amp_bs  = resp - base if (np.isfinite(resp) and np.isfinite(base)) else np.nan

    vel_window = np.nan
    if rel_t.size >= 3:
        dfdt = np.gradient(fec, rel_t)
        vel_window = window_mean(rel_t, dfdt, RESP_WIN)

    return amp_raw, amp_bs, vel_window, base, rel_t, fec

def window_mean(t, y, win):
    m = (t >= win[0]) & (t <= win[1])
    return np.nanmean(y[m]) if np.any(m) else np.nan

def classify_block(delay):
    if not np.isfinite(delay): return None
    if SHORT_DELAY[0] <= delay <= SHORT_DELAY[1]: return "short"
    if LONG_DELAY[0]  <= delay <= LONG_DELAY[1]:  return "long"
    return None

# ─────────────────────────────────────────────────────────────
# CURVE FITTING
# ─────────────────────────────────────────────────────────────
def _exp_rise(x, y0, A, tau):
    xeff = np.maximum(0.0, x)
    return y0 + A * (1.0 - np.exp(-xeff / np.maximum(1e-6, tau)))

def _exp_decay(x, y0, A, tau):
    xeff = np.maximum(0.0, x)
    return y0 + A * np.exp(-xeff / np.maximum(1e-6, tau))

def _half_life(tau): return np.log(2.0) * tau

def _fit_stack_post0(x, stack, kind):
    # Mean/SEM calc
    L = stack.shape[1]
    cnt = np.sum(np.isfinite(stack), axis=0)
    mu  = np.nanmean(stack, axis=0)
    se  = np.nanstd(stack, axis=0, ddof=0) / np.sqrt(np.maximum(1, cnt))
    
    # Filter for fit: x >= 0 and enough data
    frac = cnt / max(1, stack.shape[0])
    mask = frac >= MIN_FRAC
    mu_plot = np.where(mask, mu, np.nan)
    se_plot = np.where(mask, se, np.nan)
    
    # Data for curve_fit
    valid_x = (x >= 0) & mask
    xx = x[valid_x]
    yy = mu[valid_x]
    ss = se[valid_x]
    
    info = {"ok": False, "msg": "no data"}
    yfit = np.full_like(x, np.nan, dtype=float)

    if xx.size >= 4:
        func = _exp_rise if kind == "rise" else _exp_decay
        # Guess
        y0 = np.median(yy)
        amp = (yy[-1] - yy[0]) if kind=="rise" else (yy[0] - yy[-1])
        p0 = [y0, amp, 5.0]
        bounds = ([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, 1e3])
        
        try:
            # Weight by 1/sem^2 (approx sigma=sem)
            sigma = ss if np.all(ss > 1e-9) else None
            popt, _ = curve_fit(func, xx, yy, p0=p0, bounds=bounds, sigma=sigma, maxfev=10000)
            
            yfit = func(x, *popt)
            # 90% Plateau logic: x = tau * ln(10)
            tau = popt[2]
            plat_x = tau * np.log(10) # ~ 2.3 * tau
            plat_y = func(plat_x, *popt)
            
            info = {
                "ok": True, 
                "tau": tau, 
                "plat_x": plat_x, 
                "plat_y": plat_y,
                "params": popt
            }
        except Exception as e:
            info["msg"] = str(e)

    return mu_plot, se_plot, yfit, info

# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────
def mean_sem_trace(traces, tgrid, bl_sub=False):
    if not traces: return np.full_like(tgrid, np.nan), np.full_like(tgrid, np.nan)
    M = np.vstack(traces)
    if bl_sub:
        # Subtract baseline per trial
        mask_bl = (tgrid >= BASELINE_WIN[0]) & (tgrid <= BASELINE_WIN[1])
        bls = np.nanmean(M[:, mask_bl], axis=1)
        M = M - bls[:, None]
        
    mu = np.nanmean(M, axis=0)
    cnt = np.sum(np.isfinite(M), axis=0)
    se = np.nanstd(M, axis=0, ddof=0) / np.sqrt(np.maximum(1, cnt))
    return mu, se

def _shade_background(ax, mode):
    ax.axvspan(RESP_WIN[0], RESP_WIN[1], color="gray", alpha=0.12, lw=0)
    ax.axvspan(0, 50, color="gray", alpha=0.18, lw=0)
    if mode == "s2l":
        ax.axvspan(EXPECTED_LONG[0], EXPECTED_LONG[1], color=COLOR_LONG, alpha=0.25, lw=0)
    else:
        ax.axvspan(EXPECTED_SHORT[0], EXPECTED_SHORT[1], color=COLOR_SHORT, alpha=0.25, lw=0)

def draw_time_panel(ax, d, title, mode, tgrid, bl_sub=False):
    # Helper to plot one group
    def p(keys, col, lbl):
        tr = []
        for k in keys: tr.extend(d.get(k, []))
        if tr:
            m, s = mean_sem_trace(tr, tgrid, bl_sub)
            ax.fill_between(tgrid, m-s, m+s, color=col, alpha=ALPHA_FILL, lw=0)
            ax.plot(tgrid, m, color=col, lw=2, label=lbl)

    p(["traces_neg1"], COLOR_TRACE_NEG_1, "-50..-26")
    p(["traces_neg2"], COLOR_TRACE_NEG_2, "-25..-1")
    p(["traces1"],     COLOR_TRACE_1,     "1..7")
    p(["traces2"],     COLOR_TRACE_2,     "8..50")

    _shade_background(ax, mode)
    ax.axvline(0, color="k", ls="--")
    ax.set_title(title)
    ax.set_xlim(TRACE_WIN)
    if ax.get_legend_handles_labels()[1]: ax.legend(fontsize=8, loc="upper left")


def contiguous_lengths_around_transition(blks, i):
    """
    Transition is between trial i and i+1.
    Returns:
        pre_len  = number of contiguous trials ending at i with block blks[i]
        post_len = number of contiguous trials starting at i+1 with block blks[i+1]
    """
    n = len(blks)
    b0 = blks[i]
    b1 = blks[i + 1]

    pre_len = 0
    j = i
    while j >= 0 and blks[j] == b0:
        pre_len += 1
        j -= 1

    post_len = 0
    j = i + 1
    while j < n and blks[j] == b1:
        post_len += 1
        j += 1

    return pre_len, post_len

def draw_time_panel_two_groups(ax, d_control, d_chemo, title, mode, tgrid, bl_sub=False):
    def p(d, keys, col, lbl):
        tr = []
        for k in keys:
            tr.extend(d.get(k, []))
        if tr:
            m, s = mean_sem_trace(tr, tgrid, bl_sub)
            ax.fill_between(tgrid, m - s, m + s, color=col, alpha=0.18, lw=0)
            ax.plot(tgrid, m, color=col, lw=2, label=lbl)

    # Control in black-ish shades
    p(d_control, ["traces_neg1"], "0.6", "Control -50..-26")
    p(d_control, ["traces_neg2"], "0.35", "Control -25..-1")
    p(d_control, ["traces1"],     "k",    "Control 1..7")
    p(d_control, ["traces2"],     "k",    "Control 8..50")

    # Chemo in blue shades
    p(d_chemo, ["traces_neg1"], "#9ecae1", "Chemo -50..-26")
    p(d_chemo, ["traces_neg2"], "#6baed6", "Chemo -25..-1")
    p(d_chemo, ["traces1"],     "#3182bd", "Chemo 1..7")
    p(d_chemo, ["traces2"],     "#08519c", "Chemo 8..50")

    _shade_background(ax, mode)
    ax.axvline(0, color="k", ls="--")
    ax.set_title(title)
    ax.set_xlim(TRACE_WIN)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc="upper left")

def draw_time_panel_superimposed(ax, d_control, d_chemo, title, mode, tgrid, bl_sub=False):
    """Plot control and chemo traces superimposed with distinct colors"""
    def p(d, keys, col, lbl, alpha=0.25):
        tr = []
        for k in keys:
            tr.extend(d.get(k, []))
        if tr:
            m, s = mean_sem_trace(tr, tgrid, bl_sub)
            ax.fill_between(tgrid, m - s, m + s, color=col, alpha=alpha, lw=0)
            ax.plot(tgrid, m, color=col, lw=1.8, label=lbl)

    # Control in dark colors (black/dark gray)
    p(d_control, ["traces_neg1"], "0.5", "Control", alpha=0.15)
    p(d_control, ["traces_neg2"], "0.4", None, alpha=0.15)
    p(d_control, ["traces1"],     "k",   None, alpha=0.15)
    p(d_control, ["traces2"],     "0.2", None, alpha=0.15)

    # Chemo in blue colors (distinct from control)
    p(d_chemo, ["traces_neg1"], "#08519c", "Chemo", alpha=0.15)
    p(d_chemo, ["traces_neg2"], "#3182bd", None, alpha=0.15)
    p(d_chemo, ["traces1"],     "#6baed6", None, alpha=0.15)
    p(d_chemo, ["traces2"],     "#9ecae1", None, alpha=0.15)

    _shade_background(ax, mode)
    ax.axvline(0, color="k", ls="--")
    ax.set_title(title)
    ax.set_xlim(TRACE_WIN)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        # Keep only first occurrence of each label
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="upper left")

def draw_trial_panel(ax, x, stack, title, mode, do_fit=False):
    ax.axvline(0, color="k", ls="--")
    
    # Basic Mean/SEM (always plotted)
    mu, se, cnt, frac, _ = mean_sem_counts(stack) # helper reused
    mask = frac >= MIN_FRAC
    mu_p = np.where(mask, mu, np.nan)
    se_p = np.where(mask, se, np.nan)
    
    col = COLOR_SHORT if mode=="s2l" else COLOR_LONG
    
    # Plot Mean/SEM
    # Left side (pre)
    pre = (x < 0) & np.isfinite(mu_p) & np.isfinite(se_p)
    if np.any(pre):
        ax.fill_between(
            x[pre],
            mu_p[pre] - se_p[pre],
            mu_p[pre] + se_p[pre],
            color=col,
            alpha=0.20,
            lw=0
        )
        ax.plot(x[pre], mu_p[pre], color=col, alpha=0.85, lw=1.3)

    # Right side (post)
    post = (x >= 0) & np.isfinite(mu_p) & np.isfinite(se_p)
    if np.any(post):
        ax.fill_between(
            x[post],
            mu_p[post] - se_p[post],
            mu_p[post] + se_p[post],
            color=col,
            alpha=0.20,
            lw=0
        )
        ax.plot(x[post], mu_p[post], color=col, alpha=0.85, lw=1.3, label="Mean ± SEM")

    # Fit Logic (only for rows 5 and 6)
    if do_fit and stack.size > 0:
        kind = "rise" if mode=="s2l" else "decay"
        _, _, yfit, info = _fit_stack_post0(x, stack, kind)
        
        if info["ok"]:
            # Plot Fit Line
            fit_col = darken(col, 0.3)
            ax.plot(x[x>=0], yfit[x>=0], color=fit_col, lw=1.8, label="Exp Fit")
            
            # Plot Marker at Plateau
            px, py = info["plat_x"], info["plat_y"]
            if -N_PRE <= px <= N_POST:
                ax.plot(px, py, 'o', color='red', markersize=6, zorder=10)
                ax.text(px, py, f" {px:.1f}", color='red', fontsize=9, fontweight='bold', va='bottom')
            
            # Text stats
            txt = f"τ={info['tau']:.1f}"
            ax.text(0.05, 0.9, txt, transform=ax.transAxes, fontsize=9)

    ax.set_title(title)
    ax.set_xlim(-N_PRE, N_POST)

def draw_trial_panel_two_groups(ax, x, stack_control, stack_chemo, title, mode, do_fit=False):
    ax.axvline(0, color="k", ls="--")

    def plot_one(stack, color, label):
        if stack.size == 0:
            return
        mu, se, cnt, frac, _ = mean_sem_counts(stack)
        mask = frac >= MIN_FRAC
        mu_p = np.where(mask, mu, np.nan)
        se_p = np.where(mask, se, np.nan)

        valid = np.isfinite(mu_p) & np.isfinite(se_p)
        if np.any(valid):
            ax.fill_between(
                x[valid],
                mu_p[valid] - se_p[valid],
                mu_p[valid] + se_p[valid],
                color=color,
                alpha=0.18,
                lw=0
            )
            ax.plot(x[valid], mu_p[valid], color=color, lw=1.5, label=label)

    plot_one(stack_control, "k", "Control")
    plot_one(stack_chemo, "blue", "Chemo")

    ax.set_title(title)
    ax.set_xlim(-N_PRE, N_POST)
    ax.legend(frameon=False, fontsize=9, loc="best")


def draw_baseline_panel_dual(ax, x, control_stack, chemo_stack, title, mode):
    ax.axvline(0, color="k", ls="--")

    def plot_stack(stack, color, label):
        if stack.size == 0:
            return
        mu, se, cnt, frac, _ = mean_sem_counts(stack)
        mask = frac >= MIN_FRAC
        mu_p = np.where(mask, mu, np.nan)
        se_p = np.where(mask, se, np.nan)
        valid = np.isfinite(mu_p) & np.isfinite(se_p)
        if np.any(valid):
            ax.fill_between(x[valid], mu_p[valid] - se_p[valid], mu_p[valid] + se_p[valid], color=color, alpha=0.18, lw=0)
            ax.plot(x[valid], mu_p[valid], color=color, lw=1.5, label=label)

    plot_stack(control_stack, "k", "Control")
    plot_stack(chemo_stack, "blue", "Chemo")

    ax.set_title(title)
    ax.set_xlim(-N_PRE, N_POST)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=9, loc="best")


def draw_real_block_number_panel_dual(ax, meta_all, direction, title):
    rows = []
    for m in meta_all:
        rows.extend(m.get("kept_rows", []))
    rows = [r for r in rows if r.get("direction") == direction]

    control_rows = [r for r in rows if r.get("chemo_flag") == 0]
    chemo_rows = [r for r in rows if r.get("chemo_flag") == 1]

    ax.axvline(0, color="k", ls="--", lw=1.5)
    ax.set_title(title)

    if not control_rows and not chemo_rows:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(-N_PRE, N_POST + 1)

    def plot_rows(target_rows, color, label):
        stack = []
        for r in target_rows:
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
        valid = np.isfinite(mu) & np.isfinite(se)
        if np.any(valid):
            ax.fill_between(x[valid], mu[valid] - se[valid], mu[valid] + se[valid], color=color, alpha=0.18, lw=0)
            ax.plot(x[valid], mu[valid], color=color, lw=1.7, label=label)

    if control_rows:
        plot_rows(control_rows, "black", "Control")
    if chemo_rows:
        plot_rows(chemo_rows, "blue", "Chemo")

    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel("Aligned trial #")
    ax.set_ylabel("Mean Block #")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=9, loc="upper left")


def draw_valid_transition_count_panel_dual(ax, meta_all, direction, title, count_window=None):
    rows = []
    for m in meta_all:
        rows.extend(m.get("kept_rows", []))
    rows = [r for r in rows if r.get("direction") == direction]

    control_rows = [r for r in rows if r.get("chemo_flag") == 0]
    chemo_rows = [r for r in rows if r.get("chemo_flag") == 1]

    ax.axvline(0, color="k", ls="--", lw=1.5)
    ax.set_title(title)

    if not control_rows and not chemo_rows:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
        return

    if count_window is None:
        count_window = max(N_PRE, N_POST)
    x = np.arange(-count_window, count_window + 1)

    def count_rows(target_rows):
        counts = np.zeros(len(x), dtype=float)
        for i, xv in enumerate(x):
            n_here = 0
            for r in target_rows:
                pre_len = int(r["pre_len"])
                post_len = int(r["post_len"])
                if -pre_len <= xv < 0:
                    n_here += 1
                elif 0 <= xv < post_len:
                    n_here += 1
            counts[i] = n_here
        return counts

    if control_rows:
        ax.plot(x, count_rows(control_rows), color="black", lw=1.7, label="Control")
    if chemo_rows:
        ax.plot(x, count_rows(chemo_rows), color="blue", lw=1.7, label="Chemo")

    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel("N transitions")
    ax.set_xlabel("Aligned trial #")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=9, loc="upper left")


def make_6row_pdf(path, x,
                  s2l_control, l2s_control,
                  s2l_chemo, l2s_chemo,
                  tgrid,
                  n_control_sessions=0,
                  n_chemo_sessions=0):

    fig, axs = plt.subplots(6, 2, figsize=(12, 18), sharex=False)

    def wrap_vel(d):
        return {
            "traces_neg1": d["vtr_neg1"], "traces_neg2": d["vtr_neg2"],
            "traces1": d["vtr1"],         "traces2": d["vtr2"]
        }

    # -------------------------------------------------
    # Row 1: S->L raw epochs | control-only / chemo-only
    # -------------------------------------------------
    draw_time_panel(
        axs[0, 0], s2l_control,
        f"S->L FEC (Raw) | Control sessions={n_control_sessions}",
        "s2l", tgrid, False
    )
    draw_time_panel(
        axs[0, 1], s2l_chemo,
        f"S->L FEC (Raw) | Chemo sessions={n_chemo_sessions}",
        "s2l", tgrid, False
    )

    # -------------------------------------------------
    # Row 2: L->S raw epochs | control-only / chemo-only
    # -------------------------------------------------
    draw_time_panel(
        axs[1, 0], l2s_control,
        f"L->S FEC (Raw) | Control sessions={n_control_sessions}",
        "l2s", tgrid, False
    )
    draw_time_panel(
        axs[1, 1], l2s_chemo,
        f"L->S FEC (Raw) | Chemo sessions={n_chemo_sessions}",
        "l2s", tgrid, False
    )

    # -------------------------------------------------
    # Row 3: S->L BL-sub epochs | control-only / chemo-only
    # -------------------------------------------------
    draw_time_panel(
        axs[2, 0], s2l_control,
        f"S->L FEC (BL-sub) | Control sessions={n_control_sessions}",
        "s2l", tgrid, True
    )
    draw_time_panel(
        axs[2, 1], s2l_chemo,
        f"S->L FEC (BL-sub) | Chemo sessions={n_chemo_sessions}",
        "s2l", tgrid, True
    )

    # -------------------------------------------------
    # Row 4: L->S BL-sub epochs | control-only / chemo-only
    # -------------------------------------------------
    draw_time_panel(
        axs[3, 0], l2s_control,
        f"L->S FEC (BL-sub) | Control sessions={n_control_sessions}",
        "l2s", tgrid, True
    )
    draw_time_panel(
        axs[3, 1], l2s_chemo,
        f"L->S FEC (BL-sub) | Chemo sessions={n_chemo_sessions}",
        "l2s", tgrid, True
    )

    # -------------------------------------------------
    # Row 5: short/long trial adaptation | superimposed
    # -------------------------------------------------
    draw_trial_panel_two_groups(
        axs[4, 0], x,
        s2l_control["raw"], s2l_chemo["raw"],
        "S->L FEC @ 200 (Raw) | Control + Chemo",
        "s2l", do_fit=False
    )
    draw_trial_panel_two_groups(
        axs[4, 1], x,
        l2s_control["raw"], l2s_chemo["raw"],
        "L->S FEC @ 200 (Raw) | Control + Chemo",
        "l2s", do_fit=False
    )

    # -------------------------------------------------
    # Row 6: BL-sub adaptation | superimposed
    # -------------------------------------------------
    draw_trial_panel_two_groups(
        axs[5, 0], x,
        s2l_control["bs"], s2l_chemo["bs"],
        "S->L BL-Sub (Fit) | Control + Chemo",
        "s2l", do_fit=True
    )
    draw_trial_panel_two_groups(
        axs[5, 1], x,
        l2s_control["bs"], l2s_chemo["bs"],
        "L->S BL-Sub (Fit) | Control + Chemo",
        "l2s", do_fit=True
    )

    plt.tight_layout()
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved: {path}   grouped version  trans6.py:607 - trans6_Chemo.py:845")
def make_6row_pdf_superimposed(path, x,
                               s2l_control, l2s_control,
                               s2l_chemo, l2s_chemo,
                               tgrid,
                               meta_all,
                               n_control_sessions=0,
                               n_chemo_sessions=0):
    """Create 9-row PDF with control and chemo SUPERIMPOSED"""
    fig, axs = plt.subplots(9, 2, figsize=(12, 27), sharex=False)

    def trace_count(d):
        return sum(len(d.get(k, [])) for k in ["traces_neg1", "traces_neg2", "traces1", "traces2"])

    # Generate accurate metadata counts from the retained transition rows
    def count_session_groups(meta_list):
        control_names = set()
        chemo_names = set()
        control_trans = 0
        chemo_trans = 0
        n_s2l = 0
        n_l2s = 0
        for m in meta_list:
            for r in m.get("kept_rows", []):
                if r.get("chemo_flag") == 0:
                    control_names.add(m.get("session_name"))
                    control_trans += 1
                elif r.get("chemo_flag") == 1:
                    chemo_names.add(m.get("session_name"))
                    chemo_trans += 1
                if r.get("direction") == "s2l":
                    n_s2l += 1
                elif r.get("direction") == "l2s":
                    n_l2s += 1
        return (len(control_names), len(chemo_names), control_trans, chemo_trans, n_s2l, n_l2s)

    if meta_all:
        n_control_sessions, n_chemo_sessions, n_control_trans, n_chemo_trans, n_s2l, n_l2s = count_session_groups(meta_all)
    else:
        n_control_sessions = n_chemo_sessions = n_control_trans = n_chemo_trans = n_s2l = n_l2s = 0

    summary = (
        "Within-block transition summary\n"
        f"Sessions={len(meta_all)}  |  Control sessions={n_control_sessions}  |  Chemo sessions={n_chemo_sessions}\n"
        f"S->L kept={n_s2l}  |  L->S kept={n_l2s}  |  Control transitions={n_control_trans}  |  Chemo transitions={n_chemo_trans}\n"
        f"Aligned length={len(x)}"
    )
    fig.suptitle(summary, fontsize=10)
    fig.subplots_adjust(top=0.92)

    s2l_control_traces = trace_count(s2l_control)
    s2l_chemo_traces = trace_count(s2l_chemo)
    l2s_control_traces = trace_count(l2s_control)
    l2s_chemo_traces = trace_count(l2s_chemo)

    # -------------------------------------------------
    # Row 1: S->L raw epochs | SUPERIMPOSED
    # -------------------------------------------------
    draw_time_panel_superimposed(
        axs[0, 0], s2l_control, s2l_chemo,
        f"S->L FEC (Raw) | Control trials={s2l_control_traces} / Chemo trials={s2l_chemo_traces}",
        "s2l", tgrid, False
    )
    axs[0, 1].axis('off')

    # -------------------------------------------------
    # Row 2: L->S raw epochs | SUPERIMPOSED
    # -------------------------------------------------
    draw_time_panel_superimposed(
        axs[1, 0], l2s_control, l2s_chemo,
        f"L->S FEC (Raw) | Control trials={l2s_control_traces} / Chemo trials={l2s_chemo_traces}",
        "l2s", tgrid, False
    )
    axs[1, 1].axis('off')

    # -------------------------------------------------
    # Row 3: S->L BL-sub epochs | SUPERIMPOSED
    # -------------------------------------------------
    draw_time_panel_superimposed(
        axs[2, 0], s2l_control, s2l_chemo,
        f"S->L FEC (BL-sub) | Control trials={s2l_control_traces} / Chemo trials={s2l_chemo_traces}",
        "s2l", tgrid, True
    )
    axs[2, 1].axis('off')

    # -------------------------------------------------
    # Row 4: L->S BL-sub epochs | SUPERIMPOSED
    # -------------------------------------------------
    draw_time_panel_superimposed(
        axs[3, 0], l2s_control, l2s_chemo,
        f"L->S FEC (BL-sub) | Control trials={l2s_control_traces} / Chemo trials={l2s_chemo_traces}",
        "l2s", tgrid, True
    )
    axs[3, 1].axis('off')

    # -------------------------------------------------
    # Row 5: short/long trial adaptation | control vs chemo
    # -------------------------------------------------
    draw_trial_panel_two_groups(
        axs[4, 0], x,
        s2l_control["raw"], s2l_chemo["raw"],
        "S->L FEC @ 200 (Raw) | Control + Chemo",
        "s2l", do_fit=False
    )
    draw_trial_panel_two_groups(
        axs[4, 1], x,
        l2s_control["raw"], l2s_chemo["raw"],
        "L->S FEC @ 200 (Raw) | Control + Chemo",
        "l2s", do_fit=False
    )

    # -------------------------------------------------
    # Row 6: BL-sub adaptation | control vs chemo with fit
    # -------------------------------------------------
    draw_trial_panel_two_groups(
        axs[5, 0], x,
        s2l_control["bs"], s2l_chemo["bs"],
        "S->L BL-Sub (Fit) | Control + Chemo",
        "s2l", do_fit=True
    )
    draw_trial_panel_two_groups(
        axs[5, 1], x,
        l2s_control["bs"], l2s_chemo["bs"],
        "L->S BL-Sub (Fit) | Control + Chemo",
        "l2s", do_fit=True
    )

    # -------------------------------------------------
    # Row 7: Baseline amplitude | control vs chemo
    # -------------------------------------------------
    draw_baseline_panel_dual(
        axs[6, 0], x,
        s2l_control["base"], s2l_chemo["base"],
        "S->L Baseline amplitude | Control + Chemo",
        "s2l"
    )
    draw_baseline_panel_dual(
        axs[6, 1], x,
        l2s_control["base"], l2s_chemo["base"],
        "L->S Baseline amplitude | Control + Chemo",
        "l2s"
    )

    # -------------------------------------------------
    # Row 8: Block numbers across aligned trials | control vs chemo
    # -------------------------------------------------
    draw_real_block_number_panel_dual(
        axs[7, 0], meta_all,
        "s2l",
        "S->L block number across aligned trials"
    )
    draw_real_block_number_panel_dual(
        axs[7, 1], meta_all,
        "l2s",
        "L->S block number across aligned trials"
    )

    # -------------------------------------------------
    # Row 9: Valid transition count across aligned trials
    # -------------------------------------------------
    draw_valid_transition_count_panel_dual(
        axs[8, 0], meta_all,
        "s2l",
        "S->L valid transition count across aligned trials"
    )
    draw_valid_transition_count_panel_dual(
        axs[8, 1], meta_all,
        "l2s",
        "L->S valid transition count across aligned trials"
    )

    plt.tight_layout()
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved: {path}   SUPERIMPOSED version - trans6_Chemo.py:1020")
def mean_sem_counts(stack):
    # Simple helper for trial arrays
    cnt = np.sum(np.isfinite(stack), axis=0)
    mu = np.nanmean(stack, axis=0)
    se = np.nanstd(stack, axis=0, ddof=0) / np.sqrt(np.maximum(1, cnt))
    frac = cnt / max(1, stack.shape[0])
    return mu, se, cnt, frac, stack.shape[0]

def empty_direction_accumulator():
    return {
        "raw": [], "bs": [], "vel": [], "base": [],
        "traces_neg1": [], "traces_neg2": [], "traces1": [], "traces2": [],
        "vtr_neg1": [], "vtr_neg2": [], "vtr1": [], "vtr2": []
    }

def empty_group_accumulator():
    return {"s2l": empty_direction_accumulator(), "l2s": empty_direction_accumulator()}

def finalize_accumulator(d, x):
    out = {}
    for k in ["raw", "bs", "vel", "base"]:
        out[k] = np.vstack(d[k]) if d[k] else np.empty((0, len(x)))
    for k in ["traces_neg1", "traces_neg2", "traces1", "traces2", "vtr_neg1", "vtr_neg2", "vtr1", "vtr2"]:
        out[k] = d[k]
    return out

def probe_policy_label(exclude_probe_plus1: bool) -> str:
    return "exclude_probe_plus1" if exclude_probe_plus1 else "include_all_probe_related"

def count_session_groups(meta_list):
    control_names = set()
    chemo_names = set()
    control_trans = 0
    chemo_trans = 0
    n_s2l = 0
    n_l2s = 0
    for m in meta_list:
        for r in m.get("kept_rows", []):
            if r.get("chemo_flag") == 0:
                control_names.add(m.get("session_name"))
                control_trans += 1
            elif r.get("chemo_flag") == 1:
                chemo_names.add(m.get("session_name"))
                chemo_trans += 1
            if r.get("direction") == "s2l":
                n_s2l += 1
            elif r.get("direction") == "l2s":
                n_l2s += 1
    return (len(control_names), len(chemo_names), control_trans, chemo_trans, n_s2l, n_l2s)

def detect_session_oscillation(rel_ts, fecs):
    baseline_stds = []
    for rt, y in zip(rel_ts, fecs):
        if rt.size < 5 or y.size < 5:
            continue
        mask = (rt >= BASELINE_WIN[0]) & (rt <= BASELINE_WIN[1])
        if np.sum(mask) >= 5:
            baseline_stds.append(float(np.nanstd(y[mask])))
    if not baseline_stds:
        return False
    return float(np.nanmedian(baseline_stds)) > OSCILLATION_BASELINE_STD_THR

def smooth_trace_if_needed(y):
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 7:
        return y
    seg = y[ok]
    win = min(OSCILLATION_SMOOTH_WIN, seg.size if seg.size % 2 == 1 else seg.size - 1)
    win = max(5, win)
    if win <= OSCILLATION_SMOOTH_POLY:
        return y
    y2 = y.copy()
    y2[ok] = savgol_filter(seg, window_length=win, polyorder=OSCILLATION_SMOOTH_POLY, mode="interp")
    return y2


# ─────────────────────────────────────────────────────────────
# MAIN PROCESSING
# ─────────────────────────────────────────────────────────────
def collect_session(p, exclude_probe_plus1=False):
    trials, chemo_flag = load_trials_from_mat(p)
    overall = session_overall_max_eye(trials)
    print(f"Overall max eye: {overall} for {p.name} - trans6_Chemo.py:1104")
    if not (np.isfinite(overall) and overall > 0): 
        print(f"Skipping {p.name} due to invalid overall - trans6_Chemo.py:1106")
        return None
    
    # Extract metrics
    amps_raw, amps_bs, vels, bases, rel_ts, fecs = [], [], [], [], [], []
    blks = []
    post_probe_excluded = 0
    
    for j, tr in enumerate(trials):
        a, b, v, base, rt, y = per_trial_metrics(tr, overall)
        amps_raw.append(a); amps_bs.append(b); vels.append(v); bases.append(base)
        rel_ts.append(rt); fecs.append(y)

        # Keep all probe-related trials by default, or exclude only probe+1.
        is_post_probe = exclude_probe_plus1 and (j > 0 and bool(trials[j - 1].get("is_probe", False)))
        if is_post_probe:
            blks.append(None)
            post_probe_excluded += 1
            continue

        bt = tr.get("BlockType", None)
        isi = tr.get("ISI", np.nan)

        if isinstance(bt, str):
            if "short" in bt:
                blks.append("short")
            elif "long" in bt:
                blks.append("long")
            else:
                blks.append(classify_block(isi if np.isfinite(isi) else (tr["AP_on"] - tr["LED_on"])))
        else:
            blks.append(classify_block(isi if np.isfinite(isi) else (tr["AP_on"] - tr["LED_on"])))

    amps_raw = np.array(amps_raw)
    amps_bs  = np.array(amps_bs)
    vels     = np.array(vels)
    bases    = np.array(bases)
    n = len(blks)
    oscillatory_session = detect_session_oscillation(rel_ts, fecs)
    if oscillatory_session:
        print(f"Applying trace smoothing to oscillatory session {p.name} - trans6_Chemo.py:1146")
    if exclude_probe_plus1:
        print(f"Excluded {post_probe_excluded} probe+1 trials for {p.name} - trans6_Chemo.py:1148")
    else:
        print(f"Kept all proberelated trials for {p.name} - trans6_Chemo.py:1150")

    # Add block ids for transitions
    block_ids = []
    current_block = 1
    for j in range(n):
        if j > 0 and blks[j] != blks[j-1]:
            current_block += 1
        block_ids.append(current_block)

    meta = {
        "session_name": p.name,
        "chemo_flag": chemo_flag,
        "probe_policy": probe_policy_label(exclude_probe_plus1),
        "kept_rows": []
    }

    # Containers
    L = N_PRE + N_POST + 1
    tgrid = np.arange(TRACE_WIN[0], TRACE_WIN[1]+TRACE_STEP, TRACE_STEP)
    
    res = {
        "s2l": {"raw":[], "bs":[], "vel":[], "base":[], "traces_neg1":[], "traces_neg2":[], "traces1":[], "traces2":[], "vtr_neg1":[], "vtr_neg2":[], "vtr1":[], "vtr2":[]},
        "l2s": {"raw":[], "bs":[], "vel":[], "base":[], "traces_neg1":[], "traces_neg2":[], "traces1":[], "traces2":[], "vtr_neg1":[], "vtr_neg2":[], "vtr1":[], "vtr2":[]}
    }

    # Identify transitions
        # Identify transitions
    for i in range(n - 1):
        b0, b1 = blks[i], blks[i + 1]
        if b0 not in ("short", "long") or b1 not in ("short", "long") or b0 == b1:
            continue

        t0 = i + 1  # first trial of the new block

        # choose destination container
        if b0 == "short" and b1 == "long":
            dest = res["s2l"]
        elif b0 == "long" and b1 == "short":
            dest = res["l2s"]
        else:
            continue

        # contiguous block lengths around the transition
        pre_len, post_len = contiguous_lengths_around_transition(blks, i)

        neg_keep = min(N_PRE, pre_len)
        pos_keep = min(N_POST, post_len)

        # Trial-domain rows
        r_row = np.full(L, np.nan)
        b_row = np.full(L, np.nan)
        v_row = np.full(L, np.nan)
        base_row = np.full(L, np.nan)

        # pre side: offsets -neg_keep ... -1
        for off in range(-neg_keep, 0):
            src_idx = t0 + off
            dst_idx = off + N_PRE
            if 0 <= src_idx < n and 0 <= dst_idx < L:
                r_row[dst_idx] = amps_raw[src_idx]
                b_row[dst_idx] = amps_bs[src_idx]
                v_row[dst_idx] = vels[src_idx]
                base_row[dst_idx] = bases[src_idx]

        # post side: offsets 0 ... pos_keep-1
        for off in range(0, pos_keep):
            src_idx = t0 + off
            dst_idx = off + N_PRE
            if 0 <= src_idx < n and 0 <= dst_idx < L:
                r_row[dst_idx] = amps_raw[src_idx]
                b_row[dst_idx] = amps_bs[src_idx]
                v_row[dst_idx] = vels[src_idx]
                base_row[dst_idx] = bases[src_idx]

        # IMPORTANT: append rows
        dest["raw"].append(r_row)
        dest["bs"].append(b_row)
        dest["vel"].append(v_row)
        dest["base"].append(base_row)

        meta["kept_rows"].append({
            "direction": "s2l" if b0 == "short" and b1 == "long" else "l2s",
            "pre_block_num": block_ids[i],
            "post_block_num": block_ids[i + 1],
            "pre_len": pre_len,
            "post_len": post_len,
            "pre_take_n": neg_keep,
            "post_take_n": pos_keep,
            "chemo_flag": chemo_flag,
            "session_name": p.name,
        })

        # trace helper
        def get_trace(k_idx):
            if 0 <= k_idx < n:
                rt, y = rel_ts[k_idx], fecs[k_idx]
                if rt.size > 1:
                    yi = np.interp(tgrid, rt, y, left=np.nan, right=np.nan)
                    vi = np.full_like(yi, np.nan)
                    ok = np.isfinite(yi)
                    if ok.sum() > 3:
                        vi[ok] = np.gradient(yi[ok], tgrid[ok])
                    return yi, vi
            return None, None

        def add_traces(target, offset_range):
            for k in offset_range:
                # keep only offsets within the contiguous block
                if k < 0 and abs(k) > pre_len:
                    continue
                if k >= 0 and k >= post_len:
                    continue

                f, v = get_trace(t0 + k)
                if f is not None:
                    target[0].append(f)
                    target[1].append(v)

        # Trace collection windows
        add_traces((dest["traces_neg1"], dest["vtr_neg1"]), range(-50, -25))
        add_traces((dest["traces_neg2"], dest["vtr_neg2"]), range(-25, 0))
        add_traces((dest["traces1"],     dest["vtr1"]),     range(0, 7))
        add_traces((dest["traces2"],     dest["vtr2"]),     range(7, 50))
    
    print(f"Collected {len(res['s2l']['raw'])} s2l and {len(res['l2s']['raw'])} l2s transitions for {p.name} - trans6_Chemo.py:1275")
    return res, tgrid, chemo_flag, meta

def run():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Pools
    pool_control = {
        "s2l": {"raw":[], "bs":[], "vel":[], "base":[], "traces_neg1":[], "traces_neg2":[], "traces1":[], "traces2":[], "vtr_neg1":[], "vtr_neg2":[], "vtr1":[], "vtr2":[]},
        "l2s": {"raw":[], "bs":[], "vel":[], "base":[], "traces_neg1":[], "traces_neg2":[], "traces1":[], "traces2":[], "vtr_neg1":[], "vtr_neg2":[], "vtr1":[], "vtr2":[]}
    }

    pool_chemo = {
        "s2l": {"raw":[], "bs":[], "vel":[], "base":[], "traces_neg1":[], "traces_neg2":[], "traces1":[], "traces2":[], "vtr_neg1":[], "vtr_neg2":[], "vtr1":[], "vtr2":[]},
        "l2s": {"raw":[], "bs":[], "vel":[], "base":[], "traces_neg1":[], "traces_neg2":[], "traces1":[], "traces2":[], "vtr_neg1":[], "vtr_neg2":[], "vtr1":[], "vtr2":[]}
    }

    meta_all = []
    
    x = np.arange(-N_PRE, N_POST+1)
    
    for mouse, root in MOUSE_ROOTS.items():
        print(f"Processing {mouse}...  trans6.py:763 - trans6_Chemo.py:1297")
        
        # Mouse Accumulators
        m_acc_control = {
            "s2l": {"raw":[], "bs":[], "vel":[], "base":[], "traces_neg1":[], "traces_neg2":[], "traces1":[], "traces2":[], "vtr_neg1":[], "vtr_neg2":[], "vtr1":[], "vtr2":[]},
            "l2s": {"raw":[], "bs":[], "vel":[], "base":[], "traces_neg1":[], "traces_neg2":[], "traces1":[], "traces2":[], "vtr_neg1":[], "vtr_neg2":[], "vtr1":[], "vtr2":[]}
        }

        m_acc_chemo = {
            "s2l": {"raw":[], "bs":[], "vel":[], "base":[], "traces_neg1":[], "traces_neg2":[], "traces1":[], "traces2":[], "vtr_neg1":[], "vtr_neg2":[], "vtr1":[], "vtr2":[]},
            "l2s": {"raw":[], "bs":[], "vel":[], "base":[], "traces_neg1":[], "traces_neg2":[], "traces1":[], "traces2":[], "vtr_neg1":[], "vtr_neg2":[], "vtr1":[], "vtr2":[]}
        }

        n_control_sessions = 0
        n_chemo_sessions = 0
        
        files = sorted(root.glob("*_EBC_*.mat"))
        print(f"Found {len(files)} files for {mouse} - trans6_Chemo.py:1314")
        tgrid_ref = None
        
        mouse_meta_all = []
        for p in files:
            if DATE_MIN:
                dt = parse_date_from_filename(p.name)
                if not dt or dt < DATE_MIN: continue
                
            try:
                data, tg, chemo_flag, meta = collect_session(p)
                if not data: continue
                if tgrid_ref is None: tgrid_ref = tg
                
                target = m_acc_chemo if chemo_flag == 1 else m_acc_control
                if chemo_flag == 1:
                    n_chemo_sessions += 1
                else:
                    n_control_sessions += 1

                for direction in ["s2l", "l2s"]:
                    for key in target[direction]:
                        target[direction][key].extend(data[direction][key])

                mouse_meta_all.append(meta)
                meta_all.append(meta)
                        
            except Exception as e:
                print(f"Skipped {p.name}: {e}  trans6.py:803 - trans6_Chemo.py:1342")

        if tgrid_ref is None: continue

        # Convert mouse lists to arrays for plotting
        def finalize(d):
            out = {}
            for k in ["raw", "bs", "vel", "base"]:
                out[k] = np.vstack(d[k]) if d[k] else np.empty((0, len(x)))
            # keep traces as lists
            for k in ["traces_neg1", "traces_neg2", "traces1", "traces2", "vtr_neg1", "vtr_neg2", "vtr1", "vtr2"]:
                out[k] = d[k]
            return out
        
        m_s2l_control = finalize(m_acc_control["s2l"])
        m_l2s_control = finalize(m_acc_control["l2s"])

        m_s2l_chemo = finalize(m_acc_chemo["s2l"])
        m_l2s_chemo = finalize(m_acc_chemo["l2s"])
        
        print(f"Mouse {mouse}: control s2l raw shape {m_s2l_control['raw'].shape}, chemo s2l raw shape {m_s2l_chemo['raw'].shape} - trans6_Chemo.py:1362")
        
        # Plot Mouse PDF (side-by-side)
        make_6row_pdf(
            SAVE_DIR / f"{mouse}_Transitions_Summary.pdf",
            x,
            m_s2l_control, m_l2s_control,
            m_s2l_chemo, m_l2s_chemo,
            tgrid_ref,
            n_control_sessions=n_control_sessions,
            n_chemo_sessions=n_chemo_sessions
        )
        
        # Plot Mouse PDF (superimposed)
        make_6row_pdf_superimposed(
            SAVE_DIR / f"{mouse}_Transitions_Summary_SUPERIMPOSED.pdf",
            x,
            m_s2l_control, m_l2s_control,
            m_s2l_chemo, m_l2s_chemo,
            tgrid_ref,
            mouse_meta_all,
            n_control_sessions=n_control_sessions,
            n_chemo_sessions=n_chemo_sessions
        )
        
        # Add to pool
        for direction in ["s2l", "l2s"]:
            src_control = m_s2l_control if direction == "s2l" else m_l2s_control
            src_chemo = m_s2l_chemo if direction == "s2l" else m_l2s_chemo
            for k in ["raw", "bs", "vel", "base"]:
                if src_control[k].size: pool_control[direction][k].append(src_control[k])
                if src_chemo[k].size: pool_chemo[direction][k].append(src_chemo[k])
            for k in ["traces_neg1", "traces_neg2", "traces1", "traces2", "vtr_neg1", "vtr_neg2", "vtr1", "vtr2"]:
                pool_control[direction][k].extend(src_control[k])
                pool_chemo[direction][k].extend(src_chemo[k])

    # Plot Pooled PDF
    print("Processing Pooled...  trans6.py:834 - trans6_Chemo.py:1399")
    def finalize_pool(d):
        out = {}
        for k in ["raw", "bs", "vel", "base"]:
            out[k] = np.vstack(d[k]) if d[k] else np.empty((0, len(x)))
        for k in ["traces_neg1", "traces_neg2", "traces1", "traces2", "vtr_neg1", "vtr_neg2", "vtr1", "vtr2"]:
            out[k] = d[k]
        return out
        
    p_s2l_control = finalize_pool(pool_control["s2l"])
    p_l2s_control = finalize_pool(pool_control["l2s"])
    p_s2l_chemo = finalize_pool(pool_chemo["s2l"])
    p_l2s_chemo = finalize_pool(pool_chemo["l2s"])
    
    print(f"Pooled: s2l control raw shape {p_s2l_control['raw'].shape}, s2l chemo raw shape {p_s2l_chemo['raw'].shape} - trans6_Chemo.py:1413")
    
    # Use standard tgrid if none found (shouldn't happen if sessions exist)
    tg = np.arange(TRACE_WIN[0], TRACE_WIN[1]+TRACE_STEP, TRACE_STEP)
    make_6row_pdf(
        SAVE_DIR / "POOLED_Transitions_Summary.pdf",
        x,
        p_s2l_control, p_l2s_control,
        p_s2l_chemo, p_l2s_chemo,
        tg
    )
    make_6row_pdf_superimposed(
        SAVE_DIR / "POOLED_Transitions_Summary_SUPERIMPOSED.pdf",
        x,
        p_s2l_control, p_l2s_control,
        p_s2l_chemo, p_l2s_chemo,
        tg,
        meta_all
    )

if __name__ == "__main__":
    run()
