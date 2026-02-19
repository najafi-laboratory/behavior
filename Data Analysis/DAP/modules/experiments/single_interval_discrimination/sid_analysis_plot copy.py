"""
Plotting utilities for the Lobule V MVP analysis.

Main entry:
    plot_session_onepager(M, R, session_title=None, save_path=None)

Panels rendered (auto-skip if data missing):
1) ΔR² histogram with fractions (scaling/clock/mixed) + median ΔR².
2) Time-resolved decoders: ISI R²(t) and short/long AUC(t) with divergence marker.
3) Hazard unique R² bar.
4) Raster (ROIs × time) sorted by the winning story; vertical lines at allowed ISIs.

Inputs:
- M: canonical session dict produced by the adapter (build_M_from_trials)
- R: results dict from run_all_from_raw or equivalent

Notes:
- Designed for smooth calcium traces (ΔF/F), NaN-safe.
- Uses matplotlib only.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _safe_percentile(a, q):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return np.percentile(a, q)


def plot_session_onepager(M: dict, R: dict, session_title: str | None = None, save_path: str | None = None):
    """Create a one-page summary figure for a single session.

    Parameters
    ----------
    M : dict
        Canonical session object (must contain 'roi_traces', 'time', 'isi_allowed').
    R : dict
        Results object with keys like 'delta_r2', 'model_pref', 'decode', 'hazard_unique_r2', 'sort_index'.
    session_title : str, optional
        Title for the page (mouse/session/date), by default None.
    save_path : str, optional
        If provided, path to save the figure (PNG/PDF) with dpi=300, by default None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Figure canvas
    fig = plt.figure(figsize=(12, 7.5))
    gs = GridSpec(2, 3, height_ratios=[1.0, 1.6], width_ratios=[1, 1.2, 0.8], wspace=0.35, hspace=0.35)

    # ---------------------
    # Panel 1: ΔR² histogram + fractions
    # ---------------------
    ax1 = fig.add_subplot(gs[0, 0])
    delta = np.asarray(R.get('delta_r2'))
    bins = np.linspace(np.nanmin(delta) if np.isfinite(np.nanmin(delta)) else -0.5,
                       np.nanmax(delta) if np.isfinite(np.nanmax(delta)) else 0.5, 31)
    ax1.hist(delta[np.isfinite(delta)], bins=bins, edgecolor='none')
    ax1.axvline(0, color='k', lw=1, ls='--')
    med = np.nanmedian(delta)
    ax1.axvline(med, color='k', lw=1.5)
    ax1.set_xlabel('ΔR² (phase − ms)')
    ax1.set_ylabel('ROIs')
    ax1.set_title('Scaling vs Clock')
    # Fractions text box
    pref = R.get('model_pref', {})
    n_total = np.isfinite(delta).sum()
    n_scal = len(pref.get('scaling', []))
    n_clock = len(pref.get('clock', []))
    n_mixed = len(pref.get('mixed', []))
    txt = (f"median ΔR² = {med:.03f}\n"
           f"scaling: {n_scal}/{n_total} ({(100*n_scal/max(n_total,1)):.1f}%)\n"
           f"clock:   {n_clock}/{n_total} ({(100*n_clock/max(n_total,1)):.1f}%)\n"
           f"mixed:   {n_mixed}/{n_total} ({(100*n_mixed/max(n_total,1)):.1f}%)")
    ax1.text(0.98, 0.98, txt, transform=ax1.transAxes, va='top', ha='right', fontsize=9,
             bbox=dict(boxstyle='round', fc='white', ec='0.8', alpha=0.9))

    # ---------------------
    # Panel 2: Time-resolved decoders
    # ---------------------
    ax2 = fig.add_subplot(gs[0, 1])
    dec = R.get('decode', {})
    tpts = np.asarray(dec.get('time_points'))
    r2_t = np.asarray(dec.get('isi_r2_t'))
    auc_t = np.asarray(dec.get('shortlong_auc_t'))
    # Left axis: R2(t)
    ax2.plot(tpts, r2_t, lw=2, label='ISI $R^2$(t)')
    ax2.set_xlabel('Time from F1-OFF (s)')
    ax2.set_ylabel('ISI $R^2$(t)')
    # Right axis: AUC(t)
    ax2b = ax2.twinx()
    ax2b.plot(tpts, auc_t, lw=2, ls='--', label='Short/Long AUC(t)')
    ax2b.set_ylabel('AUC(t)')
    # Divergence marker if present
    div_t = dec.get('divergence_time')
    if div_t is not None and np.isfinite(div_t):
        ax2.axvline(div_t, color='k', lw=1.5, ls=':')
        ax2.text(div_t, ax2.get_ylim()[1], 'divergence', rotation=90, va='top', ha='right', fontsize=8)
    # Legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, frameon=True)
    ax2.set_title('Time-resolved decode')

    # ---------------------
    # Panel 3: Hazard unique R² bar
    # ---------------------
    ax3 = fig.add_subplot(gs[0, 2])
    huv = R.get('hazard_unique_r2')
    if huv is None or not np.isfinite(huv):
        ax3.text(0.5, 0.5, 'Hazard UV N/A', va='center', ha='center')
        ax3.set_axis_off()
    else:
        ax3.bar(["Hazard unique $R^2$"], [huv])
        ax3.set_ylim(0, max(0.01, 1.05*huv))
        ax3.set_title('Anticipation beyond ramp')
        for i, v in enumerate([huv]):
            ax3.text(i, v, f" {v:.3f}", va='bottom', ha='left')

    # ---------------------
    # Panel 4: Raster (ROI × time) sorted by winner
    # ---------------------
    ax4 = fig.add_subplot(gs[1, :])
    X = M['roi_traces']  # (Ntr, Nroi, Nt)
    T = M['time']
    order = np.asarray(R.get('sort_index')) if 'sort_index' in R else np.arange(X.shape[1])
    # Average across trials (NaN-safe)
    mean_tr = np.nanmean(X, axis=0)  # (Nroi, Nt)
    mean_tr = mean_tr[order]
    # Color scale from robust percentiles
    vmin = _safe_percentile(mean_tr, 5)
    vmax = _safe_percentile(mean_tr, 95)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(mean_tr), np.nanmax(mean_tr)
    im = ax4.imshow(mean_tr, aspect='auto', interpolation='nearest',
                    extent=[T[0], T[-1], 0, mean_tr.shape[0]], origin='lower',
                    vmin=vmin, vmax=vmax)
    ax4.set_xlabel('Time from F1-OFF (s)')
    ax4.set_ylabel('ROIs (sorted)')
    ax4.set_title('Population raster (avg across trials)')
    # Vertical lines at allowed ISIs
    if 'isi_allowed' in M:
        for isi in np.asarray(M['isi_allowed']):
            ax4.axvline(isi, color='w', lw=0.8, ls=':', alpha=0.8)
    cbar = plt.colorbar(im, ax=ax4, fraction=0.02, pad=0.02)
    cbar.set_label('ΔF/F (a.u.)')

    # Page title
    if session_title:
        fig.suptitle(session_title, y=0.99, fontsize=13)

    # Tight layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
