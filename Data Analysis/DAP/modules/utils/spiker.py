import os
from typing import Tuple, Optional
import numpy as np
from scipy import signal, ndimage
from scipy.ndimage import gaussian_filter

try:
    from suite2p.extraction.dcnv import oasis, preprocess
    SUITE2P_AVAILABLE = True
    print("[spiker] Suite2p OASIS available")
except ImportError:
    SUITE2P_AVAILABLE = False
    print("[spiker] Suite2p not available, using fallback")

# ---- Hard-coded settings ----
PLANE_DIR = r"D:\behavior\2p_imaging\processed\2afc\YH24LG\YH24LG_CRBL_lobulev_20250620_2afc-494\suite2p\plane0"
FS_HZ = 29.760441593958042        # frame rate (Hz)
NPIL_COEFF = 0.7      # neuropil subtraction coefficient
TAU_SEC = 0.25         # calcium decay time constant (s) - more realistic
ONLY_CELLS = True     # use only ROIs marked as cells
N_SHOW = 8            # number of example cells to plot

# Pipeline-specific parameters
SIG_BASELINE = 600  # Gaussian filter sigma for baseline (like your pipeline)
NORMALIZE = True    # Z-score normalization
SAVGOL_WINDOW = 9   # Savitzky-Golay window
SAVGOL_POLY = 3     # Savitzky-Golay polynomial order



def suite2p_oasis_spike_inference(Fc: np.ndarray, fs_hz: float = FS_HZ, tau_sec: float = TAU_SEC) -> np.ndarray:
    """
    Use Suite2p's preprocess + OASIS implementation for spike inference.
    Suite2p signature: 
    - preprocess(F, baseline, win_baseline, sig_baseline, fs, prctile_baseline=8) -> F_processed
    - oasis(F, batch_size, tau, fs) -> S (deconvolved spikes)
    """
    if not SUITE2P_AVAILABLE:
        print("[spiker] Suite2p not available, using fallback method")
        return batch_spike_inference_fast(Fc, fs_hz, tau_sec)

    n_cells, n_frames = Fc.shape

    print(f"[spiker] Running Suite2p preprocess + OASIS for {n_cells} cells...")
    
    try:
        # Step 1: Suite2p preprocessing (baseline correction)
        print("[spiker] Suite2p preprocessing...")
        F_processed = preprocess(
            F=Fc.astype(np.float64),      # (n_cells, n_frames) - neuropil-corrected fluorescence
            baseline='maximin',             # baseline method: 'maximin', 'constant', 'constant_prctile'
            win_baseline=60.0,             # baseline window in seconds
            sig_baseline=10.0,             # Gaussian filter width in frames
            fs=fs_hz,                      # sampling rate
            prctile_baseline=8.0           # percentile for baseline if using 'constant_prctile'
        )
        
        print(f"[spiker] Preprocessing completed, shape: {F_processed.shape}")
        
        # Step 2: Suite2p OASIS deconvolution
        print("[spiker] Suite2p OASIS deconvolution...")
        spikes = oasis(
            F=F_processed,                 # preprocessed fluorescence
            batch_size=8000,               # frames per batch
            tau=tau_sec,                   # decay time constant
            fs=fs_hz                       # sampling rate
        )
        
        print(f"[spiker] OASIS completed successfully for all {n_cells} cells")
        return spikes
        
    except Exception as e:
        print(f"[spiker] Suite2p preprocess+OASIS failed: {e}")
        # Fallback to custom implementation


# def suite2p_oasis_spike_inference(dff: np.ndarray, fs_hz: float = FS_HZ, tau_sec: float = TAU_SEC) -> np.ndarray:
#     """
#     Use Suite2p's OASIS implementation for spike inference.
#     Suite2p signature: oasis(F, batch_size, tau, fs) -> S (deconvolved spikes)
#     """
#     if not SUITE2P_AVAILABLE:
#         print("[spiker] Suite2p not available, using fallback method")
#         return batch_spike_inference_fast(dff, fs_hz, tau_sec)
    
#     n_cells, n_frames = dff.shape
    
#     print(f"[spiker] Running Suite2p OASIS for {n_cells} cells...")
    
#     try:
#         # Suite2p OASIS expects (n_cells, n_frames) and returns (n_cells, n_frames)
#         spikes = oasis(
#             F=dff.astype(np.float64),  # (n_cells, n_frames)
#             batch_size=8000,           # frames per batch
#             tau=tau_sec,               # decay time constant
#             fs=fs_hz                   # sampling rate
#         )
        
#         print(f"[spiker] OASIS completed successfully for all {n_cells} cells")
#         return spikes
        
#     except Exception as e:
#         print(f"[spiker] OASIS failed: {e}, using fallback method")
#         # Fallback to custom implementation


# def suite2p_oasis_spike_inference(dff: np.ndarray, fs_hz: float = FS_HZ, tau_sec: float = TAU_SEC) -> np.ndarray:
#     """
#     Use Suite2p's OASIS implementation for spike inference.
#     """
#     if not SUITE2P_AVAILABLE:
#         print("[spiker] Suite2p not available, using fallback method")
#         return batch_spike_inference_fast(dff, fs_hz, tau_sec)
    
#     n_cells, n_frames = dff.shape
#     spikes = np.zeros_like(dff)
    
#     print(f"[spiker] Running Suite2p OASIS for {n_cells} cells...")
    
#     # Suite2p OASIS parameters
#     batch_size = 8000  # Suite2p default
#     tau = tau_sec  # decay time constant
    
#     for i in range(n_cells):
#         if i % 100 == 0:
#             print(f"[spiker] OASIS cell {i}/{n_cells}")
        
#         try:
#             # Suite2p OASIS expects (n_frames,) input
#             trace = dff[i, :].astype(np.float64)
            
#             # Call Suite2p OASIS function
#             # Returns: (deconvolved_trace, spikes, baseline, neucoeff, lam)
#             c, s, b, neu, lam = oasis(
#                 F=trace.reshape(1, -1),  # Suite2p expects (1, n_frames) for single cell
#                 Fneu=np.zeros((1, n_frames)),  # Not needed since we already did neuropil correction
#                 ops={
#                     'tau': tau,
#                     'fs': fs_hz,
#                     'neucoeff': 0.0,  # Already corrected
#                     'baseline': 'maximin',  # Suite2p baseline method
#                     'win_baseline': 60.0,  # baseline window in seconds
#                     'sig_baseline': 10.0,   # baseline smoothing
#                     'prctile_baseline': 8.0,  # baseline percentile
#                     'batch_size': batch_size
#                 }
#             )
            
#             # s contains the spike inference
#             spikes[i, :] = s.flatten()
            
#         except Exception as e:
#             print(f"[spiker] OASIS failed for cell {i}: {e}, using fallback")
#             # Fallback to simple method
#             spikes[i, :] = oasis_deconvolve_fast(dff[i, :], fs_hz, tau_sec)
    
#     return spikes


def load_suite2p_plane(plane_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    F = np.load(os.path.join(plane_dir, "F.npy"), allow_pickle=False)
    Fneu = np.load(os.path.join(plane_dir, "Fneu.npy"), allow_pickle=False)
    iscell = np.load(os.path.join(plane_dir, "iscell.npy"), allow_pickle=True)
    iscell_mask = np.asarray(iscell[:, 0]).astype(bool)
    spks_path = os.path.join(plane_dir, "spks.npy")
    spks_suite2p = np.load(spks_path, allow_pickle=False) if os.path.isfile(spks_path) else None
    return F, Fneu, iscell_mask, spks_suite2p

def neuropil_subtract(F: np.ndarray, Fneu: np.ndarray, npil_coeff: float = NPIL_COEFF) -> np.ndarray:
    """Neuropil subtraction matching your pipeline."""
    return F - npil_coeff * Fneu

def pipeline_dff_computation(Fc: np.ndarray, sig_baseline: float = SIG_BASELINE, 
                           normalize: bool = NORMALIZE, outlier_threshold: float = 5.0) -> np.ndarray:
    """
    Compute dF/F exactly like your pipeline:
    1. Neuropil-corrected F -> baseline via Gaussian filter
    2. Baseline subtraction: (F - F0) / F0
    3. Z-score normalization
    4. Outlier detection and replacement
    5. Savitzky-Golay smoothing
    """
    print("[spiker] Computing dF/F using pipeline method...")
    
    # Step 1: Copy corrected fluorescence
    dff = Fc.copy()
    
    # Step 2: Baseline estimation using Gaussian filter (exactly like your pipeline)
    print(f"[spiker] Computing baseline with Gaussian filter (sigma={sig_baseline})...")
    f0 = gaussian_filter(dff, [0., sig_baseline])
    
    # Step 3: Baseline subtraction: dF/F = (F - F0) / F0
    print("[spiker] Applying baseline subtraction and normalization...")
    total_outliers = 0
    total_points = 0
    
    for j in range(dff.shape[0]):
        # Baseline subtraction
        dff_cell = (dff[j, :] - f0[j, :]) / (f0[j, :] + 1e-10)
        
        # Outlier detection (simplified z-score method like your pipeline)
        outliers_mask = detect_outliers_zscore(dff_cell, threshold=outlier_threshold)
        n_outliers = np.sum(outliers_mask)
        total_outliers += n_outliers
        total_points += len(dff_cell)
        
        if n_outliers > 0:
            # Replace outliers with median
            median_val = np.nanmedian(dff_cell[~outliers_mask])
            dff_cell[outliers_mask] = median_val
        
        # Z-score normalization
        if normalize:
            mean_val = np.nanmean(dff_cell)
            std_val = np.nanstd(dff_cell)
            if std_val > 1e-10:
                dff_cell = (dff_cell - mean_val) / std_val
            else:
                dff_cell = dff_cell - mean_val
        
        dff[j, :] = dff_cell
    
    outlier_percent = (total_outliers / total_points) * 100 if total_points > 0 else 0
    print(f"[spiker] Found {total_outliers} outliers ({outlier_percent:.2f}%)")
    
    # Step 4: Apply Savitzky-Golay filter (like your pipeline)
    print(f"[spiker] Applying Savitzky-Golay filter (window={SAVGOL_WINDOW}, poly={SAVGOL_POLY})...")
    dff_filtered = apply_savgol_filter(dff, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY)
    
    return dff_filtered

def detect_outliers_zscore(data: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Simplified z-score outlier detection matching your pipeline."""
    valid_mask = ~np.isnan(data)
    if np.sum(valid_mask) < 3:
        return np.zeros_like(data, dtype=bool)
    
    valid_data = data[valid_mask]
    outliers_mask = np.zeros_like(data, dtype=bool)
    
    # Modified z-score using median absolute deviation (robust)
    median_val = np.median(valid_data)
    mad_val = np.median(np.abs(valid_data - median_val))
    if mad_val > 0:
        modified_z_scores = 0.6745 * (data - median_val) / mad_val
        outliers_mask = np.abs(modified_z_scores) > threshold
    
    return outliers_mask

def apply_savgol_filter(dff: np.ndarray, window_length: int = SAVGOL_WINDOW, 
                       polyorder: int = SAVGOL_POLY) -> np.ndarray:
    """Apply Savitzky-Golay filter exactly like your pipeline."""
    from scipy.signal import savgol_filter
    
    # Ensure window_length is odd and valid
    if window_length % 2 == 0:
        window_length += 1
    
    n_timepoints = dff.shape[1]
    if window_length > n_timepoints:
        window_length = n_timepoints if n_timepoints % 2 == 1 else n_timepoints - 1
    
    if polyorder >= window_length:
        polyorder = window_length - 1
    
    # Apply filter to each cell
    dff_filtered = np.apply_along_axis(
        savgol_filter, 1, dff,
        window_length=window_length,
        polyorder=polyorder
    )
    
    return dff_filtered

def robust_df_over_f(Fc: np.ndarray, window_sec: float = 300.0, fs_hz: float = FS_HZ) -> np.ndarray:
    """
    Robust dF/F calculation using a sliding window approach.
    For each timepoint, F0 is the 10th percentile in a window around that timepoint.
    """
    window_frames = int(window_sec * fs_hz)
    n_cells, n_frames = Fc.shape
    dff = np.zeros_like(Fc)
    
    for i in range(n_cells):
        trace = Fc[i, :]
        f0_trace = np.zeros(n_frames)
        
        for t in range(n_frames):
            # Define window around current timepoint
            start_idx = max(0, t - window_frames // 2)
            end_idx = min(n_frames, t + window_frames // 2)
            window_data = trace[start_idx:end_idx]
            
            # Use 10th percentile as baseline
            f0_trace[t] = np.percentile(window_data, 10)
        
        # Smooth the baseline to avoid rapid fluctuations
        f0_smooth = ndimage.uniform_filter1d(f0_trace, size=int(fs_hz * 10))  # 10s smoothing
        f0_smooth = np.maximum(f0_smooth, np.percentile(trace, 5))  # Lower bound
        
        # Calculate dF/F
        dff[i, :] = (trace - f0_smooth) / f0_smooth
    
    return dff

def oasis_deconvolve(dff_trace: np.ndarray, fs_hz: float, tau_sec: float = TAU_SEC) -> np.ndarray:
    """
    Simple OASIS-like deconvolution for spike inference.
    Uses constrained non-negative deconvolution.
    """
    dt = 1.0 / fs_hz
    gamma = np.exp(-dt / tau_sec)
    
    # Simple constrained deconvolution
    n = len(dff_trace)
    spikes = np.zeros(n)
    c = np.zeros(n)
    
    # Initialize
    c[0] = max(0, dff_trace[0])
    spikes[0] = c[0]
    
    for t in range(1, n):
        # Predict calcium based on previous timepoint
        c_pred = gamma * c[t-1]
        
        # If observed > predicted, infer spike
        if dff_trace[t] > c_pred:
            spikes[t] = dff_trace[t] - c_pred
            c[t] = dff_trace[t]
        else:
            spikes[t] = 0
            c[t] = c_pred
    
    # Apply mild smoothing to reduce noise
    spikes = ndimage.gaussian_filter1d(spikes, sigma=0.5)
    spikes = np.maximum(spikes, 0)  # Ensure non-negative
    
    return spikes

def batch_spike_inference(dff: np.ndarray, fs_hz: float = FS_HZ, tau_sec: float = TAU_SEC) -> np.ndarray:
    """Apply spike inference to all cells."""
    n_cells, n_frames = dff.shape
    spikes = np.zeros_like(dff)
    
    for i in range(n_cells):
        spikes[i, :] = oasis_deconvolve(dff[i, :], fs_hz, tau_sec)
    
    return spikes


def robust_df_over_f_fast(Fc: np.ndarray, window_sec: float = 300.0, fs_hz: float = FS_HZ) -> np.ndarray:
    """
    Fast robust dF/F using vectorized rolling percentile.
    """
    from scipy.ndimage import uniform_filter1d
    
    window_frames = int(window_sec * fs_hz)
    n_cells, n_frames = Fc.shape
    
    # Pre-allocate
    dff = np.zeros_like(Fc)
    
    print(f"[spiker] Processing {n_cells} cells with {window_frames}-frame window...")
    
    for i in range(n_cells):
        if i % 50 == 0:
            print(f"[spiker] Cell {i}/{n_cells}")
            
        trace = Fc[i, :]
        
        # Use pandas rolling for fast percentile (if available)
        try:
            import pandas as pd
            df = pd.Series(trace)
            f0_trace = df.rolling(window=window_frames, center=True, min_periods=1).quantile(0.1).values
        except ImportError:
            # Fallback: faster numpy approach with stride_tricks
            f0_trace = np.zeros(n_frames)
            half_win = window_frames // 2
            for t in range(n_frames):
                start_idx = max(0, t - half_win)
                end_idx = min(n_frames, t + half_win)
                f0_trace[t] = np.percentile(trace[start_idx:end_idx], 10)
        
        # Smooth baseline (vectorized)
        f0_smooth = uniform_filter1d(f0_trace, size=int(fs_hz * 10))
        f0_smooth = np.maximum(f0_smooth, np.percentile(trace, 5))
        
        # Calculate dF/F
        dff[i, :] = (trace - f0_smooth) / f0_smooth
    
    return dff

def oasis_deconvolve_fast(dff_trace: np.ndarray, fs_hz: float, tau_sec: float = TAU_SEC) -> np.ndarray:
    """
    Faster vectorized OASIS-like deconvolution.
    """
    dt = 1.0 / fs_hz
    gamma = np.exp(-dt / tau_sec)
    
    n = len(dff_trace)
    spikes = np.zeros(n)
    c = np.zeros(n)
    
    # Vectorized where possible
    c[0] = max(0, dff_trace[0])
    spikes[0] = c[0]
    
    for t in range(1, n):
        c_pred = gamma * c[t-1]
        if dff_trace[t] > c_pred:
            spikes[t] = dff_trace[t] - c_pred
            c[t] = dff_trace[t]
        else:
            spikes[t] = 0
            c[t] = c_pred
    
    # Light smoothing
    spikes = ndimage.gaussian_filter1d(spikes, sigma=0.3)
    return np.maximum(spikes, 0)

def batch_spike_inference_fast(dff: np.ndarray, fs_hz: float = FS_HZ, tau_sec: float = TAU_SEC) -> np.ndarray:
    """Fast batch spike inference with progress."""
    n_cells, n_frames = dff.shape
    spikes = np.zeros_like(dff)
    
    print(f"[spiker] Spike inference for {n_cells} cells...")
    for i in range(n_cells):
        if i % 100 == 0:
            print(f"[spiker] Spike cell {i}/{n_cells}")
        spikes[i, :] = oasis_deconvolve_fast(dff[i, :], fs_hz, tau_sec)
    
    return spikes



def plot_raster_segments(
    dff: np.ndarray,
    spk: np.ndarray,
    fs_hz: float,
    out_dir: str,
    seg_sec: float = 6.0,
    n_segments: int = 30,
    cmap: str = "viridis"
) -> list[str]:
    """
    Save raster figures of ROIs (rows) vs time (columns) for sequential segments.
    Each figure: top = DFF raster, bottom = spike raster. Time axes are aligned.
    """
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    n_roi, T = dff.shape
    seg_len = int(round(seg_sec * fs_hz))
    if seg_len <= 1:
        raise ValueError("seg_len too small; check fs_hz/seg_sec")

    # Robust color scaling
    dff_vmin, dff_vmax = np.percentile(dff, [1, 99])
    spk_vmin, spk_vmax = 0, np.percentile(spk, 99.5)

    saved = []
    max_segments = min(n_segments, max(0, T // seg_len))
    for i in range(max_segments):
        s0 = i * seg_len
        s1 = s0 + seg_len
        if s0 >= T:
            break
        s1 = min(s1, T)

        t0 = s0 / fs_hz
        t1 = s1 / fs_hz

        dff_seg = dff[:, s0:s1]
        spk_seg = spk[:, s0:s1]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"hspace": 0.2})

        # DFF raster (top)
        im1 = axes[0].imshow(
            dff_seg,
            aspect="auto",
            cmap=cmap,
            extent=[t0, t1, 0, n_roi],
            vmin=dff_vmin,
            vmax=dff_vmax,
            interpolation="nearest",
        )
        axes[0].set_ylabel("ROI")
        axes[0].set_title(f"dF/F raster (seg {i+1}/{max_segments})")
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.02, pad=0.02)
        cbar1.set_label("dF/F")

        # Spike raster (bottom)
        im2 = axes[1].imshow(
            spk_seg,
            aspect="auto",
            cmap="hot",  # Better for sparse spike data
            extent=[t0, t1, 0, n_roi],
            vmin=spk_vmin,
            vmax=spk_vmax,
            interpolation="nearest",
        )
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("ROI")
        axes[1].set_title("Inferred spikes")
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.02, pad=0.02)
        cbar2.set_label("Spike rate")

        fig.suptitle(f"Recording segment [{t0:.1f}, {t1:.1f}] s", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        fname = os.path.join(out_dir, f"raster_seg_{i:02d}.png")
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        # plt.close(fig)
        saved.append(fname)

    return saved

def main():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    plane = PLANE_DIR
    if not os.path.isdir(plane):
        raise FileNotFoundError(f"Plane folder not found: {plane}")

    F, Fneu, iscell_mask, spks_suite2p = load_suite2p_plane(plane)
    print(f"[spiker] Loaded from: {plane}")
    print(f"[spiker] Shapes: F={F.shape}, Fneu={Fneu.shape}, iscell={iscell_mask.shape}, fs={FS_HZ:.2f} Hz")
    if spks_suite2p is not None:
        print(f"[spiker] Found Suite2p spks.npy: {spks_suite2p.shape}")

    # Select ROIs
    if ONLY_CELLS:
        sel = iscell_mask
        print(f"[spiker] Using only cells: {sel.sum()} / {sel.size}")
    else:
        sel = np.ones(F.shape[0], dtype=bool)
        print(f"[spiker] Using all ROIs: {sel.sum()}")

    F = F[sel, :]
    Fneu = Fneu[sel, :]
    if spks_suite2p is not None:
        spks_suite2p = spks_suite2p[sel, :]

    # Pipeline-matching processing
    print("[spiker] Neuropil subtraction...")
    Fc = neuropil_subtract(F, Fneu, NPIL_COEFF)
    
    print("[spiker] Computing dF/F using pipeline method...")
    dff = pipeline_dff_computation(Fc, SIG_BASELINE, NORMALIZE)
    
    # Use Suite2p OASIS for spike inference
    print("[spiker] Running Suite2p OASIS spike inference...")
    spks_est = suite2p_oasis_spike_inference(Fc, FS_HZ, TAU_SEC)
    
    # Also compare with Suite2p's pre-computed spikes if available
    spks_suite2p_orig = spks_suite2p if spks_suite2p is not None else None


    # Save outputs
    out_path_dff = os.path.join(plane, "dff_pipeline_method.npy")
    out_path_spks = os.path.join(plane, "spks_oasis_custom.npy")
    np.save(out_path_dff, dff)
    np.save(out_path_spks, spks_est)
    print(f"[spiker] Saved pipeline-method dF/F: {out_path_dff}")
    print(f"[spiker] Saved OASIS spike estimate: {out_path_spks}")

    # Save raster segments
    ras_out = os.path.join(plane, "spiker_rasters_oasis")
    saved = plot_raster_segments(dff, spks_est, FS_HZ, ras_out, seg_sec=6.0, n_segments=30)
    print(f"[spiker] Saved {len(saved)} raster segment(s) to: {ras_out}")

    # Example traces - compare all methods
    if plt is None:
        print("[spiker] matplotlib not available; skipping example plots.")
        return

    T = F.shape[1]
    n = min(N_SHOW, F.shape[0])
    t = np.arange(T) / FS_HZ

    fig, axes = plt.subplots(n, 1, figsize=(16, 2*n), sharex=True)
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        ax2 = ax.twinx()

        # dF/F
        ax.plot(t, dff[i], color="C0", lw=0.8, alpha=0.9, label="dF/F (pipeline)")
        ax.set_ylabel("dF/F", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")

        # Spikes comparison
        ax2.plot(t, spks_est[i], color="C3", lw=0.8, alpha=0.9, label="OASIS (custom)")
        if spks_suite2p_orig is not None:
            ax2.plot(t, spks_suite2p_orig[i], color="0.5", lw=0.6, alpha=0.7, label="Suite2p (original)")
        ax2.set_ylabel("Spike rate", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")

        ax.set_title(f"Cell {i} — OASIS comparison, τ={TAU_SEC}s")
        
        if i == 0:
            lines, labels = [], []
            for a in (ax, ax2):
                lns, labs = a.get_legend_handles_labels()
                lines += lns; labels += labs
            ax.legend(lines, labels, frameon=False, fontsize=9, loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Suite2p OASIS spike inference — {plane}", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    main()