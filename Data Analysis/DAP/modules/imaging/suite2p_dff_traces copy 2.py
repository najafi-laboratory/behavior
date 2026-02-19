"""
Suite2p dF/F Trace Extraction Module

dF/F trace extraction for Suite2p output data.
Follows the same pattern as other pipeline components.

This handles:
1. Load QC-filtered Suite2p data (from qc_results/)
2. Compute Fneu-corrected Frescence
3. Apply baseline subtraction and normalization
4. Save dF/F traces
"""

import os
import numpy as np
import h5py
import logging
from typing import Dict, Any, List, Tuple, Optional
from scipy.ndimage import gaussian_filter
from suite2p.extraction.dcnv import oasis, preprocess


try:
    from numba import njit
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False

if _NUMBA_OK:
    @njit
    def _rq_hist_1d_causal(x, win, q, lo, hi, nbins):
        n = x.size
        out = np.empty(n, np.float32)
        h = np.zeros(nbins, np.int64)
        scale = (nbins - 1) / (hi - lo + 1e-9)
        def to_bin(v):
            if np.isnan(v): return -1
            b = int((v - lo)*scale)
            if b < 0: b = 0
            if b >= nbins: b = nbins - 1
            return b
        cnt = 0
        # prime first window
        for t in range(min(win, n)):
            b = to_bin(x[t])
            if b >= 0:
                h[b]+=1; cnt+=1
        def q_val():
            if cnt == 0: return np.nan
            target = int(q*(cnt-1))
            c = 0
            for b in range(nbins):
                c += h[b]
                if c > target:
                    return lo + (b + 0.5)/scale
            return lo + (nbins - 0.5)/scale
        for t in range(n):
            if t == 0:
                out[t] = q_val(); continue
            l = t - win
            r = t
            if l >= 0:
                b = to_bin(x[l])
                if b >= 0:
                    h[b]-=1; cnt-=1
            if r < n:
                b = to_bin(x[r])
                if b >= 0:
                    h[b]+=1; cnt+=1
            out[t] = q_val()
        return out
   
   

from modules.utils.spiker import NPIL_COEFF

class Suite2pDffTraces:
    """
    Suite2p dF/F trace extraction processor following pipeline component pattern.
    
    Handles dF/F computation from QC-filtered Suite2p data.
    """
    
    def __init__(self, config_manager, subject_list, logger=None):
        """
        Initialize the Suite2p dF/F processor.
        
        Args:
            config_manager: ConfigManager instance
            subject_list: List of subject IDs to process
            logger: Logger instance
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        self.subject_list = subject_list
        self.logger = logger or logging.getLogger(__name__)
        
        # Get imaging paths from config
        self.imaging_data_base = self.config.get('paths', {}).get('imaging_data_base', '')
        
        self.logger.info("S2P_DFF: Suite2pDffTraces initialized")

    def _ensure_check_trace_dir(self, output_path: str) -> str:
        path = os.path.join(output_path, 'qc_results', 'check_trace')
        os.makedirs(path, exist_ok=True)
        return path

    # ================== NEW HELPERS (Step 1) ==================
    def _neuropil_regress(self, F, Fneu, alpha_init=0.7, bounds=(0.3, 1.2)):
        """
        Per-ROI linear regression: F ≈ a*Fneu + b -> Fc = F - a*Fneu - b
        Clamps 'a' to bounds. Returns Fc, a_hat, b_hat.
        """
        n, T = F.shape
        a = np.full(n, alpha_init, float)
        b = np.zeros(n, float)
        lam = 1e-6
        ones = np.ones((T,), float)
        for i in range(n):
            x = Fneu[i]
            X = np.stack([x, ones], axis=1)          # (T,2)
            XtX = X.T @ X + lam * np.eye(2)
            beta = np.linalg.solve(XtX, X.T @ F[i])
            a[i] = float(np.clip(beta[0], bounds[0], bounds[1]))
            b[i] = float(beta[1])
        Fc = F - (a[:, None] * Fneu) - b[:, None]
        return Fc, a, b

    # ================= POST dFF QC HELPERS =================
    def _rowwise_corr(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y = np.nan_to_num(y)
        y = (y - y.mean()) / (y.std() + 1e-9)
        Xc = X - np.nanmean(X, axis=1, keepdims=True)
        Xc /= (np.nanstd(Xc, axis=1, keepdims=True) + 1e-9)
        return np.nanmean(Xc * y[None, :], axis=1)

    def _bandpower_ratio(self, x: np.ndarray, fs: float, band) -> Tuple[float, float]:
        from scipy.signal import welch
        f, Pxx = welch(np.nan_to_num(x), fs=fs, nperseg=min(1024, x.size))
        tot = np.trapz(Pxx, f)
        sel = (f >= band[0]) & (f <= band[1])
        bp = np.trapz(Pxx[sel], f[sel]) if sel.any() else 0.0
        return bp, tot

    def _robust_slope_per_min(self, t_min: np.ndarray, y: np.ndarray) -> float:
        y0 = np.nan_to_num(y)
        A = np.c_[t_min, np.ones_like(t_min)]
        beta, *_ = np.linalg.lstsq(A, y0, rcond=None)
        return float(beta[0])

    def _metrics_on_dff(self, dff: np.ndarray, baseline: np.ndarray,
                        motion: Optional[np.ndarray], neuropil_dff: Optional[np.ndarray],
                        fs: float, band: Tuple[float, float]) -> Dict[str, np.ndarray]:
        n, T = dff.shape
        med = np.nanmedian(dff, axis=1)[:, None]
        mad = 1.4826 * np.nanmedian(np.abs(dff - med), axis=1)[:, None] + 1e-9
        z = (dff - med) / mad
        events = z > 3.0

        # SNR proxy (median event amp / noise)
        peak = np.where(events, dff, np.nan)
        snr = np.nanmedian(peak, axis=1) / (mad[:, 0] + 1e-9)
        snr = np.nan_to_num(snr)

        transientness = np.zeros(n, float)
        for i in range(n):
            bp, tp = self._bandpower_ratio(dff[i], fs, band)
            transientness[i] = bp / (tp + 1e-12)

        tmin = np.arange(T) / fs / 60.0
        drift_abs_slope = np.array([abs(self._robust_slope_per_min(tmin, baseline[i])) for i in range(n)])

        motion_corr = np.zeros(n)
        if motion is not None:
            motion_corr = np.abs(self._rowwise_corr(dff, motion))

        # Neuropil coupling (if stored in intermediate Fc and Fneu — here only ratio vs mean neuropil proxy)
        neuropil_ratio = np.full(n, np.nan)
        neuropil_corr = np.full(n, np.nan)
        if neuropil_dff is not None:
            neuropil_corr = np.abs(self._rowwise_corr(dff, np.nanmean(neuropil_dff, axis=0)))
            amp_roi = np.nanmedian(np.abs(dff), axis=1)
            amp_np = np.nanmedian(np.abs(neuropil_dff), axis=1) + 1e-9
            neuropil_ratio = 1.0 / (amp_roi / amp_np + 1e-9)

        artifact_frac = (np.abs(z) > 6.0).mean(axis=1)
        event_rate_hz = events.sum(axis=1) / (T / fs)

        return dict(
            snr=snr.astype(np.float32),
            transientness=transientness.astype(np.float32),
            drift_abs_slope=drift_abs_slope.astype(np.float32),
            motion_corr=motion_corr.astype(np.float32),
            neuropil_corr=neuropil_corr.astype(np.float32),
            neuropil_ratio=neuropil_ratio.astype(np.float32),
            artifact_frac=artifact_frac.astype(np.float32),
            event_rate_hz=event_rate_hz.astype(np.float32)
        )

    def _decide_keep(self, metrics: Dict[str, np.ndarray], labels: List[str], qc_cfg: Dict[str, Any]):
        n = len(labels)
        keep = np.ones(n, bool)
        tests_by_roi = []
        common = qc_cfg.get('common', {})
        per_label = qc_cfg.get('per_label', {})
        artifact_z_max = float(common.get('artifact_z_max', 6.0))
        rate_max = float(common.get('event_rate_max_hz', 5.0))

        for i, lab in enumerate(labels):
            L = lab if lab in per_label else 'uncertain'
            rules = per_label.get(L, {})
            tests = dict(
                snr = metrics['snr'][i] >= rules.get('snr_min', -1),
                motion = metrics['motion_corr'][i] <= rules.get('motion_corr_max', 1e9),
                neuropil = (np.isnan(metrics['neuropil_ratio'][i]) or
                            metrics['neuropil_ratio'][i] <= rules.get('neuropil_ratio_max', 1e9)),
                drift = metrics['drift_abs_slope'][i] <= rules.get('drift_abs_slope_max_per_min', 1e9),
                transient = metrics['transientness'][i] >= rules.get('transientness_min', -1),
                artifact = metrics['artifact_frac'][i] <= (artifact_z_max / 100.0),  # simple scaled gate
                rate = metrics['event_rate_hz'][i] <= rate_max
            )
            tests_by_roi.append(tests)
            keep[i] = all(tests.values())
        return keep, tests_by_roi

    def _run_post_dff_qc(self, output_path: str, ops: Dict, dff: np.ndarray):
        exp_cfg = self.config_manager.get_experiment_config()
        ip_cfg = exp_cfg.get('imaging_preprocessing', {})
        qc_cfg = ip_cfg.get('post_dff_qc', {})
        if not qc_cfg.get('enabled', False):
            self.logger.info("S2P_DFF: post_dff_qc disabled.")
            return None

        fs = float(ops.get('fs', ops.get('fs_hz', 30.0)))
        band = tuple(qc_cfg.get('event_band_hz', [0.05, 1.5]))

        # Load baseline from dff.h5 if present
        baseline = None
        dff_file = os.path.join(output_path, 'dff.h5')
        try:
            with h5py.File(dff_file, 'r') as f:
                if 'baseline' in f:
                    baseline = f['baseline'][()]
        except Exception:
            pass
        if baseline is None:
            self.logger.info("S2P_DFF: QC baseline missing; recomputing quick fallback (15s, q=0.2).")
            baseline = self._rolling_quantile_2d(dff, fs, win_s=15, min_win_s=8, q=0.2)

        # Motion trace (optional)
        motion = None
        mname = qc_cfg.get('motion_trace', 'rigid_corr')
        mpath = os.path.join(output_path, 'qc_results', f'{mname}.npy')
        if os.path.exists(mpath):
            try:
                motion = np.load(mpath)
                if motion.ndim > 1:
                    motion = motion[0]
            except Exception:
                motion = None
        if motion is not None and motion.size != dff.shape[1]:
            self.logger.warning("S2P_DFF: Motion trace length mismatch; ignoring.")
            motion = None

        # Labels
        labels_csv = os.path.join(output_path, 'qc_results', 'roi_labels.csv')
        import pandas as pd
        if os.path.exists(labels_csv):
            lab_df = pd.read_csv(labels_csv)
            lab_df = lab_df.sort_values('roi_id')
            labels = lab_df['final_label'].fillna('uncertain').tolist()
        else:
            self.logger.warning("S2P_DFF: roi_labels.csv missing; assuming all uncertain.")
            labels = ['uncertain'] * dff.shape[0]

        metrics = self._metrics_on_dff(dff, baseline, motion, None, fs, band)
        keep, tests = self._decide_keep(metrics, labels, qc_cfg)

        # Write CSV summary
        rows = []
        for i in range(dff.shape[0]):
            r = dict(roi_id=i+1, final_label=labels[i], keep_dff=int(keep[i]))
            for k, v in metrics.items():
                r[k] = float(v[i])
            for k, passed in tests[i].items():
                r[f'pass_{k}'] = int(passed)
            rows.append(r)
        out_csv = os.path.join(output_path, 'qc_results', 'post_dff_qc.csv')
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        self.logger.info(f"S2P_DFF: post_dff_qc wrote {out_csv}")

        # Merge keep flag into roi_labels.csv
        if os.path.exists(labels_csv):
            lab_df['keep_dff'] = keep.astype(int)
            lab_df.to_csv(labels_csv, index=False)
            self.logger.info("S2P_DFF: Added keep_dff to roi_labels.csv")

        # Store in dff.h5
        try:
            with h5py.File(dff_file, 'a') as f:
                if 'qc' in f:
                    del f['qc']
                grp = f.create_group('qc')
                for k, v in metrics.items():
                    grp.create_dataset(k, data=v, compression='gzip')
                grp.create_dataset('keep_dff', data=keep.astype(np.uint8))
        except Exception as e:
            self.logger.warning(f"S2P_DFF: Could not append QC metrics to dff.h5 ({e})")

        kept_frac = keep.mean()
        self.logger.info(f"S2P_DFF: keep_dff fraction={kept_frac:.2%}")
        return dict(keep=keep, metrics=metrics)

    def _neuropil_regress_fast(self, F: np.ndarray, Fneu: np.ndarray,
                                alpha_init: float = 0.7,
                                bounds: Tuple[float, float] = (0.3, 1.2),
                                fit_cfg: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Vectorized per‑ROI regression:
                F ≈ a*Fneu + b  ->  Fc = F - a*Fneu - b
            Optional:
            - subsample_k: stride frames
            - low_activity_quantile: restrict fit to lowest activity frames (robust)
            """
            if fit_cfg is None:
                fit_cfg = {}
            subsample_k = int(fit_cfg.get('subsample_k', 1))
            qa = float(fit_cfg.get('low_activity_quantile', 0.0))
            min_frames = int(fit_cfg.get('min_frames', 500))
            eps = 1e-9

            F = F.astype(np.float32, copy=False)
            N = Fneu.astype(np.float32, copy=False)
            n, T = F.shape

            # Frame selection mask
            use_idx = np.arange(T)
            if subsample_k > 1:
                use_idx = use_idx[::subsample_k]

            if 0.0 < qa < 0.9:
                # crude activity proxy: residual under initial alpha_init
                resid = F - alpha_init * N
                # percentile per ROI across time
                thresh = np.quantile(resid, qa, axis=1)[:, None]
                low_mask = resid <= thresh
                # combine with subsample stride
                stride_mask = np.zeros((T,), bool)
                stride_mask[use_idx] = True
                mask = low_mask & stride_mask[None, :]
                # ensure minimum frames
                valid_counts = mask.sum(axis=1)
                too_few = valid_counts < min_frames
                if np.any(too_few):
                    # fallback: just stride for those
                    mask[too_few, :] = stride_mask
                # For sums we need per‑ROI masked frames; implement via boolean multiply
                M = mask.astype(np.float32)
                denom = M.sum(axis=1) + eps
                sumF = (F * M).sum(axis=1)
                sumN = (N * M).sum(axis=1)
                sumNN = (N * N * M).sum(axis=1)
                sumNF = (N * F * M).sum(axis=1)
                T_eff = denom
                varN = sumNN - (sumN * sumN) / T_eff
                cov = sumNF - (sumN * sumF) / T_eff
            else:
                # simple stride only
                Fsub = F[:, use_idx]
                Nsub = N[:, use_idx]
                T_eff = Fsub.shape[1]
                sumF = Fsub.sum(axis=1)
                sumN = Nsub.sum(axis=1)
                sumNN = (Nsub * Nsub).sum(axis=1)
                sumNF = (Nsub * Fsub).sum(axis=1)
                varN = sumNN - (sumN * sumN) / T_eff
                cov = sumNF - (sumN * sumF) / T_eff

            a = cov / (varN + eps)
            a = np.clip(a, bounds[0], bounds[1])
            b = (sumF - a * sumN) / (T_eff + eps)

            # Fitted correction on full time axis
            Fc = F - a[:, None] * N - b[:, None]

            # Handle degenerate ROIs
            bad = varN < 1e-6
            if np.any(bad):
                a[bad] = alpha_init
                b[bad] = 0.0
                Fc[bad] = F[bad] - a[bad, None] * N[bad]

            self.logger.info(
                f"S2P_DFF: Neuropil regress fast (subsample_k={subsample_k}, "
                f"low_q={qa}, median a={np.median(a):.3f}, range=({a.min():.3f},{a.max():.3f}))"
            )
            return Fc, a, b

    def _baseline_percentile_fast(self, Fc: np.ndarray, fs: float, cfg: Dict[str, Any]) -> np.ndarray:
        """
        Fast rolling percentile baseline (centered or causal).
        Centered mode approximated by causal then shifting back by win//2 (minimal smoothing side-effects).
        """
        method = cfg.get('method', 'rolling_percentile_fast')
        if method not in ('rolling_percentile_fast', 'rolling_percentile'):
            raise ValueError("baseline method mismatch in _baseline_percentile_fast call")
        win_s = float(cfg.get('win_s', 15))
        q = float(cfg.get('quantile', 0.20))
        min_win_s = float(cfg.get('min_win_s', 8))
        nbins = int(cfg.get('nbins', 256))
        mode = cfg.get('mode', 'centered')
        win = max(int(win_s*fs), int(min_win_s*fs), 5)
        n, T = Fc.shape
        out = np.empty((n, T), np.float32)

        if not _NUMBA_OK:
            # fallback to slower python centered exact
            self.logger.info("S2P_DFF: Numba unavailable; using slower Python percentile baseline.")
            return self._rolling_quantile_2d(Fc, fs, win_s=win_s, min_win_s=min_win_s, q=q)

        for i in range(n):
            x = Fc[i].astype(np.float32)
            finite = x[np.isfinite(x)]
            if finite.size == 0:
                out[i] = 0; continue
            lo, hi = np.quantile(finite, [0.01, 0.995])
            if hi <= lo: hi = lo + 1.0
            causal = _rq_hist_1d_causal(x, win, q, lo, hi, nbins)
            if mode == 'causal':
                out[i] = causal
            else:
                shift = win // 2
                # shift back (centered)
                out[i, :-shift] = causal[shift:]
                out[i, -shift:] = causal[-1]
        self.logger.info(f"S2P_DFF: Fast rolling percentile baseline win={win} q={q} mode={mode}")
        return out

    def _rolling_quantile_1d(self, x, q=0.2, win=451):
        win = max(5, int(win) | 1)   # ensure odd
        n = x.size
        if n == 0:
            return np.zeros_like(x)
        if n < win:
            # use global quantile if shorter than window
            return np.full_like(x, np.nanquantile(x, q), dtype=float)
        half = win // 2
        out = np.empty_like(x, dtype=float)
        for t in range(n):
            lo = max(0, t - half)
            hi = min(n, t + half + 1)
            out[t] = np.nanquantile(x[lo:hi], q)
        return out

    def _rolling_quantile_2d(self, X, fs, win_s=15, min_win_s=8, q=0.2):
        win = max(int(win_s * fs), int(min_win_s * fs))
        win = max(5, win | 1)
        return np.vstack([self._rolling_quantile_1d(row, q=q, win=win) for row in X]).astype(np.float32)

    
    def load_dff_data(self, subject_id: str, suite2p_path: str, output_path: str) -> Optional[Dict[str, Any]]:
        """
        Load data needed for dF/F computation.
        
        Args:
            subject_id: Subject identifier
            suite2p_path: Path to Suite2p output directory (plane0)
            output_path: Path to QC output
            
        Returns:
            Dictionary containing loaded data or None if failed
        """
        try:
            data = {}
            
            # Load ops file for metadata (need neucoeff for Fneu correction)
            ops_path = os.path.join(output_path, 'ops.npy')
            if os.path.exists(ops_path):
                data['ops'] = np.load(ops_path, allow_pickle=True).item()
                self.logger.info(f"S2P_DFF: Loaded ops for {subject_id}")
            else:
                self.logger.error(f"S2P_DFF: Missing ops.npy at {ops_path}")
                return None
            
            # Load QC-filtered Frescence data
            F_path = os.path.join(output_path, 'qc_results', 'F.npy')
            if os.path.exists(F_path):
                data['F'] = np.load(F_path, allow_pickle=True)
                self.logger.info(f"S2P_DFF: Loaded QC-filtered Frescence for {subject_id}")
            else:
                self.logger.error(f"S2P_DFF: Missing QC Frescence at {F_path}")
                return None
            
            # Load QC-filtered Fneu data
            Fneu_path = os.path.join(output_path, 'qc_results', 'Fneu.npy')
            if os.path.exists(Fneu_path):
                data['Fneu'] = np.load(Fneu_path, allow_pickle=True)
                self.logger.info(f"S2P_DFF: Loaded QC-filtered Fneu for {subject_id}")
            else:
                self.logger.error(f"S2P_DFF: Missing QC Fneu at {Fneu_path}")
                return None
            
            # TODO update suite2p processing pipeline to get spks when running pace
            
            # # Load QC-filtered spike deconvolution data
            # spike_deconvolution_path = os.path.join(output_path, 'qc_results', 'spks.npy')
            # if os.path.exists(spike_deconvolution_path):
            #     data['spks'] = np.load(spike_deconvolution_path, allow_pickle=True)
            #     self.logger.info(f"S2P_DFF: Loaded QC-filtered spike deconvolution for {subject_id}")
            # else:
            #     self.logger.error(f"S2P_DFF: Missing QC spike deconvolution at {spike_deconvolution_path}")
            #     return None         
            
            # Load QC-filtered stat data
            stat_path = os.path.join(output_path, 'qc_results', 'stat.npy')
            if os.path.exists(stat_path):
                data['stat'] = np.load(stat_path, allow_pickle=True)
                self.logger.info(f"S2P_DFF: Loaded QC-filtered stat data for {subject_id}")
            else:
                self.logger.error(f"S2P_DFF: Missing QC stat data at {stat_path}")
                return None

            self.logger.info(f"S2P_DFF: Successfully loaded dF/F data for {subject_id}")
            return data
            
        except Exception as e:
            self.logger.error(f"S2P_DFF: Failed to load dF/F data for {subject_id}: {e}")
            return None
    
   

    def suite2p_oasis_spike_inference(self, Fc: np.ndarray, fs_hz: float = 30, tau_sec: float = 0.25) -> np.ndarray:
        """
        Use Suite2p's preprocess + OASIS implementation for spike inference.
        Suite2p signature: 
        - preprocess(F, baseline, win_baseline, sig_baseline, fs, prctile_baseline=8) -> F_processed
        - oasis(F, batch_size, tau, fs) -> S (deconvolved spikes)
        """
                
        n_cells, n_frames = Fc.shape
        self.logger.info(f"S2P_DFF: Running Suite2p OASIS spike inference for {n_cells} cells, {n_frames} frames")

        
        try:
            # Step 1: Suite2p preprocessing (baseline correction)
            self.logger.info("[spiker] Suite2p preprocessing...")
            F_processed = preprocess(
                F=Fc.astype(np.float64),      # (n_cells, n_frames) - Fneu-corrected Frescence
                baseline='maximin',             # baseline method: 'maximin', 'constant', 'constant_prctile'
                win_baseline=60.0,             # baseline window in seconds
                sig_baseline=10.0,             # Gaussian filter width in frames
                fs=fs_hz,                      # sampling rate
                prctile_baseline=8.0           # percentile for baseline if using 'constant_prctile'
            )
            self.logger.info(f"[spiker] Preprocessing completed, shape: {F_processed.shape}")
            
            # Step 2: Suite2p OASIS deconvolution
            self.logger.info("[spiker] Suite2p OASIS deconvolution...")
            spikes = oasis(
                F=F_processed,                 # preprocessed Frescence
                batch_size=8000,               # frames per batch
                tau=tau_sec,                   # decay time constant
                fs=fs_hz                       # sampling rate
            )

            self.logger.info(f"[spiker] OASIS completed successfully for all {n_cells} cells")
            return spikes

        except Exception as e:
            self.logger.error(f"[spiker] Suite2p preprocess+OASIS failed: {e}")
            return np.zeros_like(Fc)

    def Fneu_subtract(self, F: np.ndarray, Fneu: np.ndarray, npil_coeff: float = 0.7) -> np.ndarray:
        """Fneu subtraction matching your pipeline."""
        return F - npil_coeff * Fneu


    def compute_spks(self, F: np.ndarray, Fneu: np.ndarray, ops: Dict) -> np.ndarray:
        """
        Compute spike traces from Frescence and Fneu data.
        
        Args:
            F: Frescence traces (n_cells x n_timepoints)
            Fneu: Fneu traces (n_cells x n_timepoints)
            ops: Suite2p ops dictionary (contains neucoeff)
            
        Returns:
            Spike traces array
        """
        try:
            self.logger.info("S2P_SPKS: Computing spike traces...")
            neucoeff = ops.get('neucoeff', 0.7)  # Default Suite2p Fneu coefficient
            Fc = self.Fneu_subtract(F.copy(), Fneu, neucoeff)            
            fs_hz = 29.760441593958042        # frame rate (Hz)
            tau_sec = 0.25
            spks = self.suite2p_oasis_spike_inference(Fc, fs_hz=fs_hz, tau_sec=tau_sec)
            self.logger.info(f"S2P_SPKS: Computed spike traces for {spks.shape[0]} cells")
            return spks

        except Exception as e:
            self.logger.error(f"S2P_SPKS: Failed to compute spike traces: {e}")
            return np.zeros_like(F)

    def compute_dff(self, F: np.ndarray, Fneu: np.ndarray, ops: Dict,
                    normalize: bool = False,
                    sig_baseline: float = 600,
                    outlier_threshold: float = 5.0,
                    outlier_method: str = 'zscore') -> np.ndarray:
        """
        High-fidelity dFF:
          - Vectorized neuropil regression (optional low-activity subset)
          - Rolling percentile baseline (fast) or Gaussian fallback
          - Optional controlled post filtering
          - Stores intermediate (handled in caller save)
        """
        try:
            exp_cfg = self.config_manager.get_experiment_config()
            ip_cfg = exp_cfg.get('imaging_preprocessing', {})
            tr_cfg = ip_cfg.get('trace_extraction', {})
            ol_cfg = tr_cfg.get('outliers', {})
            npil_cfg = tr_cfg.get('neuropil', {})
            base_cfg = tr_cfg.get('baseline', {})
            filt_cfg = tr_cfg.get('post_filter', {})
            fs_hz = float(tr_cfg.get('fs_hz', ops.get('fs', ops.get('fs_hz', 30.0))))

            # Neuropil
            npil_method = npil_cfg.get('method', 'regress')
            if npil_method == 'regress':
                alpha_init = float(npil_cfg.get('alpha_init', 0.7))
                a_lo, a_hi = npil_cfg.get('alpha_bounds', [0.3, 1.2])
                fit_cfg = npil_cfg.get('fit', {})
                Fc, a_hat, b_hat = self._neuropil_regress_fast(
                    F, Fneu, alpha_init=alpha_init,
                    bounds=(a_lo, a_hi), fit_cfg=fit_cfg
                )
                self._last_np_a = a_hat
                self._last_np_b = b_hat
            elif npil_method == 'scalar':
                alpha = float(npil_cfg.get('alpha_init', 0.7))
                Fc = F - alpha * Fneu
                self._last_np_a = np.full(F.shape[0], alpha, float)
                self._last_np_b = np.zeros(F.shape[0], float)
            else:
                Fc = F
                self._last_np_a = np.zeros(F.shape[0])
                self._last_np_b = np.zeros(F.shape[0])

            # Baseline
            b_method = base_cfg.get('method', 'rolling_percentile_fast')
            if b_method in ('rolling_percentile_fast', 'rolling_percentile'):
                f0 = self._baseline_percentile_fast(Fc, fs_hz, base_cfg)
            else:
                sig_baseline = float(base_cfg.get('sig_baseline', 600))
                f0 = gaussian_filter(Fc, [0., sig_baseline])
                self.logger.info(f"S2P_DFF: Gaussian baseline sigma={sig_baseline}")
            self._last_baseline = f0.astype(np.float32)

            # dFF
            dff = (Fc - f0) / (f0 + 1e-10)

            # Outliers
            outlier_method = ol_cfg.get('method', outlier_method)
            outlier_threshold = float(ol_cfg.get('threshold', outlier_threshold))
            total_out = 0
            total_pts = dff.size
            if outlier_threshold > 0:
                for i in range(dff.shape[0]):
                    mask = self._detect_outliers(dff[i], method=outlier_method, threshold=outlier_threshold)
                    if mask.any():
                        total_out += int(mask.sum())
                        med = np.nanmedian(dff[i][~mask]) if (~mask).any() else 0.0
                        dff[i][mask] = med
                if total_pts:
                    self.logger.info(f"S2P_DFF: Outliers replaced {total_out/total_pts*100:.2f}% ({outlier_method}, thr={outlier_threshold})")

            # Optional post filter (light touch)
            method_pf = filt_cfg.get('method', 'none')
            if method_pf == 'savgol':
                from scipy.signal import savgol_filter
                win_s = float(filt_cfg.get('savgol_window_s', 0.3))
                win = max(5, int(win_s * fs_hz) | 1)
                poly = int(filt_cfg.get('savgol_poly', 3))
                dff = np.apply_along_axis(savgol_filter, 1, dff, window_length=win, polyorder=poly)
                self.logger.info(f"S2P_DFF: SavGol smoothing (win={win} frames, poly={poly})")
            elif method_pf == 'lowpass':
                from scipy.signal import butter, filtfilt
                lp = float(filt_cfg.get('lowpass_hz', 8.0))
                ny = fs_hz / 2.0
                if lp < ny:
                    b, a = butter(2, lp / ny, btype='low')
                    for i in range(dff.shape[0]):
                        dff[i] = filtfilt(b, a, dff[i])
                    self.logger.info(f"S2P_DFF: Lowpass Butterworth {lp}Hz")

            if tr_cfg.get('normalize', normalize):
                mu = np.nanmean(dff, axis=1, keepdims=True)
                sd = np.nanstd(dff, axis=1, keepdims=True) + 1e-9
                dff = (dff - mu) / sd
                self.logger.info("S2P_DFF: z-normalized (config normalize=true)")

            self._last_Fc = Fc.astype(np.float32)
            self.logger.info(f"S2P_DFF: dFF shape={dff.shape} fs={fs_hz:.2f}Hz")
            return dff.astype(np.float32)

        except Exception as e:
            self.logger.error(f"S2P_DFF: Failed compute_dff ({e})")
            return F.copy()


    # def compute_dff(self, F: np.ndarray, Fneu: np.ndarray, ops: Dict,
    #                normalize: bool = True, sig_baseline: float = 600,
    #                outlier_threshold: float = 5.0, outlier_method: str = 'zscore') -> np.ndarray:
    #     """
    #     Compute dF/F traces from Frescence and Fneu data.
        
    #     Args:
    #         F: Frescence traces (n_cells x n_timepoints)
    #         Fneu: Fneu traces (n_cells x n_timepoints)
    #         ops: Suite2p ops dictionary (contains neucoeff)
    #         normalize: Whether to apply z-score normalization
    #         sig_baseline: Gaussian filter sigma for baseline estimation
    #         outlier_threshold: Threshold for outlier detection (z-score or IQR multiplier)
    #         outlier_method: Method for outlier detection ('zscore', 'iqr', 'percentile')
            
    #     Returns:
    #         dF/F traces array
    #     """
    #     try:
    #         self.logger.info("S2P_DFF: Computing Fneu-corrected Frescence...")
            
    #         dff = np.zeros_like(F)  # Initialize dF/F array
            
    #         # Fneu correction: F_corrected = F - neucoeff * Fneu
    #         neucoeff = ops.get('neucoeff', 0.7)  # Default Suite2p Fneu coefficient
    #         # dff = F.copy() - neucoeff * Fneu
    #         Fc = self.Fneu_subtract(F.copy(), Fneu, neucoeff)

    #         self.logger.info(f"S2P_DFF: Applied Fneu correction with coefficient {neucoeff}")
    #         self.logger.info("S2P_DFF: Computing baseline subtraction and normalization...")
            
    #         # Baseline estimation using Gaussian filter
    #         f0 = gaussian_filter(Fc, [0., sig_baseline])
            
    #         # Track outlier statistics
    #         total_outliers = 0
    #         total_points = 0
            
    #         # Compute dF/F for each cell
    #         for j in range(Fc.shape[0]):
    #             # Baseline subtraction: dF/F = (F - F0) / F0
    #             dff_cell = (Fc[j, :] - f0[j, :]) / (f0[j, :] + 1e-10)  # Add small epsilon to avoid division by zero

    #             # Check dF/F before outlier handling
    #             dff_mean_before = np.nanmean(dff_cell)
    #             dff_std_before = np.nanstd(dff_cell)
    #             dff_min_before = np.nanmin(dff_cell)
    #             dff_max_before = np.nanmax(dff_cell)
                
    #             # Detect and handle outliers
    #             outliers_mask = self._detect_outliers(dff_cell, method=outlier_method, threshold=outlier_threshold)
    #             n_outliers = np.sum(outliers_mask)
    #             total_outliers += n_outliers
    #             total_points += len(dff_cell)
                
    #             if n_outliers > 0:
    #                 # Replace outliers with median value
    #                 median_val = np.nanmedian(dff_cell[~outliers_mask])
    #                 dff_cell[outliers_mask] = median_val
                    
    #                 if j < 5:  # Log details for first few cells
    #                     self.logger.info(f"S2P_DFF: Cell {j}: Found {n_outliers} outliers ({n_outliers/len(dff_cell)*100:.1f}%)")
                
    #             # Check dF/F after outlier handling
    #             dff_mean_after_outliers = np.nanmean(dff_cell)
    #             dff_std_after_outliers = np.nanstd(dff_cell)
    #             dff_min_after_outliers = np.nanmin(dff_cell)
    #             dff_max_after_outliers = np.nanmax(dff_cell)
                
    #             if j < 3:  # Log stats for first few cells
    #                 self.logger.info(f"S2P_DFF: Cell {j} before outlier removal - Mean: {dff_mean_before:.4f}, Std: {dff_std_before:.4f}, Range: [{dff_min_before:.4f}, {dff_max_before:.4f}]")
    #                 self.logger.info(f"S2P_DFF: Cell {j} after outlier removal - Mean: {dff_mean_after_outliers:.4f}, Std: {dff_std_after_outliers:.4f}, Range: [{dff_min_after_outliers:.4f}, {dff_max_after_outliers:.4f}]")
                
    #             if normalize:
    #                 # Z-score normalization using robust statistics
    #                 mean_val = np.nanmean(dff_cell)
    #                 std_val = np.nanstd(dff_cell)
                    
    #                 if std_val > 1e-10:  # Avoid division by very small numbers
    #                     dff_cell = (dff_cell - mean_val) / std_val
    #                 else:
    #                     self.logger.warning(f"S2P_DFF: Cell {j} has very small std ({std_val:.2e}), skipping normalization")
    #                     dff_cell = dff_cell - mean_val  # Just center around zero
                
    #             dff[j, :] = dff_cell
            
    #         # Overall outlier statistics
    #         outlier_percent = (total_outliers / total_points) * 100 if total_points > 0 else 0
    #         self.logger.info(f"S2P_DFF: Outlier detection - Method: {outlier_method}, Threshold: {outlier_threshold}")
    #         self.logger.info(f"S2P_DFF: Found {total_outliers} outliers out of {total_points} points ({outlier_percent:.2f}%)")
            
    #         # Check final dF/F statistics
    #         dff_mean_final = np.nanmean(dff)
    #         dff_std_final = np.nanstd(dff)
    #         dff_min_final = np.nanmin(dff)
    #         dff_max_final = np.nanmax(dff)
            
    #         self.logger.info(f"S2P_DFF: Final dF/F - Mean: {dff_mean_final:.4f}, Std: {dff_std_final:.4f}")
    #         self.logger.info(f"S2P_DFF: Final dF/F - Range: [{dff_min_final:.4f}, {dff_max_final:.4f}]")
            
    #         # Sanity checks
    #         if normalize and abs(dff_mean_final) > 0.1:
    #             self.logger.warning(f"S2P_DFF: Normalized dF/F mean ({dff_mean_final:.4f}) is not near zero!")
    #         if normalize and abs(dff_std_final - 1.0) > 0.2:
    #             self.logger.warning(f"S2P_DFF: Normalized dF/F std ({dff_std_final:.4f}) is not near 1.0!")
            
    #         # Check for problematic values
    #         n_nan = np.sum(np.isnan(dff))
    #         n_inf = np.sum(np.isinf(dff))
            
    #         if n_nan > 0:
    #             self.logger.warning(f"S2P_DFF: Found {n_nan} NaN values in final dF/F traces")
    #         if n_inf > 0:
    #             self.logger.warning(f"S2P_DFF: Found {n_inf} infinite values in final dF/F traces")
            
    #         n_cells, n_timepoints = dff.shape
    #         self.logger.info(f"S2P_DFF: Computed dF/F for {n_cells} cells, {n_timepoints} timepoints")
    #         self.logger.info(f"S2P_DFF: Baseline sigma: {sig_baseline}, Normalization: {normalize}")
            
    #         return dff
            
    #     except Exception as e:
    #         self.logger.error(f"S2P_DFF: Failed to compute dF/F: {e}")
    #         return F.copy()  # Return original if failed
    
    def _detect_outliers(self, data: np.ndarray, method: str = 'zscore', threshold: float = 5.0) -> np.ndarray:
        """
        Detect outliers in 1D data array.
        
        Args:
            data: 1D array of data points
            method: Method for outlier detection ('zscore', 'iqr', 'percentile')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean mask where True indicates outliers
        """
        try:
            # Remove NaN values for outlier detection
            valid_mask = ~np.isnan(data)
            if np.sum(valid_mask) < 3:  # Need at least 3 points
                return np.zeros_like(data, dtype=bool)
            
            valid_data = data[valid_mask]
            outliers_mask = np.zeros_like(data, dtype=bool)
            
            if method == 'zscore':
                # Z-score method using robust statistics
                median_val = np.median(valid_data)
                mad_val = np.median(np.abs(valid_data - median_val))  # Median Absolute Deviation
                if mad_val > 0:
                    modified_z_scores = 0.6745 * (data - median_val) / mad_val
                    outliers_mask = np.abs(modified_z_scores) > threshold
                
            elif method == 'iqr':
                # Interquartile Range method
                q25, q75 = np.percentile(valid_data, [25, 75])
                iqr = q75 - q25
                if iqr > 0:
                    lower_bound = q25 - threshold * iqr
                    upper_bound = q75 + threshold * iqr
                    outliers_mask = (data < lower_bound) | (data > upper_bound)
                    
            elif method == 'percentile':
                # Percentile-based method
                lower_percentile = threshold
                upper_percentile = 100 - threshold
                lower_bound, upper_bound = np.percentile(valid_data, [lower_percentile, upper_percentile])
                outliers_mask = (data < lower_bound) | (data > upper_bound)
                
            else:
                self.logger.warning(f"S2P_DFF: Unknown outlier method '{method}', using zscore")
                return self._detect_outliers(data, method='zscore', threshold=threshold)
            
            return outliers_mask
            
        except Exception as e:
            self.logger.error(f"S2P_DFF: Failed to detect outliers: {e}")
            return np.zeros_like(data, dtype=bool)

    def save_spks_results(self, output_path: str, spks: np.ndarray) -> bool:
        """
        Save spike results to HDF5 file.
        """
        try:
            spks_file = os.path.join(output_path, 'spks.h5')

            with h5py.File(spks_file, 'w') as f:
                f['spks'] = spks

            self.logger.info(f"S2P_DFF: Saved spike results to {spks_file}")
            return True

        except Exception as e:
            self.logger.error(f"S2P_DFF: Failed to save spike results: {e}")
            return False

    def save_dff_results(self, output_path: str, dff: np.ndarray) -> bool:
        try:
            dff_file = os.path.join(output_path, 'dff.h5')
            with h5py.File(dff_file, 'w') as f:
                f.create_dataset('dff', data=dff, compression='gzip')
                # Store intermediates for reproducibility
                if hasattr(self, '_last_Fc'):
                    f.create_dataset('Fc', data=self._last_Fc, compression='gzip')
                if hasattr(self, '_last_baseline'):
                    f.create_dataset('baseline', data=self._last_baseline, compression='gzip')
                if hasattr(self, '_last_np_a'):
                    f.create_dataset('neuropil_a', data=self._last_np_a.astype(np.float32))
                if hasattr(self, '_last_np_b'):
                    f.create_dataset('neuropil_b', data=self._last_np_b.astype(np.float32))
            self.logger.info(f"S2P_DFF: Saved dF/F + intermediates to {dff_file}")
            return True
        except Exception as e:
            self.logger.error(f"S2P_DFF: Failed to save dF/F ({e})")
            return False

    # def save_dff_results(self, output_path: str, dff: np.ndarray) -> bool:
    #     """
    #     Save dF/F results to HDF5 file.
        
    #     Args:
    #         output_path: Output directory path
    #         dff: dF/F traces array
    #         F: Original Frescence traces array
            
    #     Returns:
    #         True if successful, False otherwise
    #     """
    #     try:
    #         dff_file = os.path.join(output_path, 'dff.h5')
            
    #         with h5py.File(dff_file, 'w') as f:
    #             f['dff'] = dff

    #         self.logger.info(f"S2P_DFF: Saved dF/F results to {dff_file}")
    #         return True
            
    #     except Exception as e:
    #         self.logger.error(f"S2P_DFF: Failed to save dF/F results: {e}")
    #         return False


    def apply_temporal_filtering(self, dff_data: np.ndarray, fs: float, 
                            filter_type: str = 'butterworth', 
                            lowpass_freq: float = 10.0,
                            highpass_freq: float = 0.1,
                            filter_order: int = 2) -> np.ndarray:
        """
        Apply temporal filtering to DFF traces to reduce noise.
        
        Args:
            dff_data: DFF traces array (n_rois, n_timepoints)
            fs: Sampling frequency in Hz
            filter_type: 'butterworth', 'gaussian', or 'savgol'
            lowpass_freq: Low-pass cutoff frequency in Hz
            highpass_freq: High-pass cutoff frequency in Hz (optional)
            filter_order: Filter order
            
        Returns:
            Filtered DFF traces
        """
        try:
            from scipy import signal
            from scipy.ndimage import gaussian_filter1d
            
            self.logger.info(f"SID_IMG: Applying {filter_type} filtering to DFF traces")
            self.logger.info(f"SID_IMG: Sampling rate: {fs:.1f} Hz, Lowpass: {lowpass_freq} Hz")
            
            filtered_dff = dff_data.copy()
            
            if filter_type == 'butterworth':
                # Design Butterworth filter
                nyquist = fs / 2.0
                
                # Low-pass filter
                if lowpass_freq < nyquist:
                    low_critical = lowpass_freq / nyquist
                    b_low, a_low = signal.butter(filter_order, low_critical, btype='low')
                    
                    # Apply to each ROI
                    for roi_idx in range(filtered_dff.shape[0]):
                        filtered_dff[roi_idx, :] = signal.filtfilt(b_low, a_low, filtered_dff[roi_idx, :])
                
                # Optional high-pass filter to remove slow drift
                if highpass_freq > 0 and highpass_freq < nyquist:
                    high_critical = highpass_freq / nyquist
                    b_high, a_high = signal.butter(filter_order, high_critical, btype='high')
                    
                    for roi_idx in range(filtered_dff.shape[0]):
                        filtered_dff[roi_idx, :] = signal.filtfilt(b_high, a_high, filtered_dff[roi_idx, :])
            
            elif filter_type == 'gaussian':
                # Gaussian smoothing (simple low-pass)
                sigma = fs / (2 * np.pi * lowpass_freq)  # Convert frequency to sigma
                
                for roi_idx in range(filtered_dff.shape[0]):
                    filtered_dff[roi_idx, :] = gaussian_filter1d(filtered_dff[roi_idx, :], sigma=sigma)
            
            elif filter_type == 'savgol':
                # Savitzky-Golay filter
                window_length = int(fs / lowpass_freq)  # Window based on cutoff frequency
                if window_length % 2 == 0:  # Must be odd
                    window_length += 1
                window_length = max(window_length, 5)  # Minimum window size
                
                for roi_idx in range(filtered_dff.shape[0]):
                    filtered_dff[roi_idx, :] = signal.savgol_filter(filtered_dff[roi_idx, :], 
                                                                window_length, filter_order)
            
            # Calculate noise reduction metrics
            original_noise = np.std(np.diff(dff_data, axis=1), axis=1)  # High-frequency noise
            filtered_noise = np.std(np.diff(filtered_dff, axis=1), axis=1)
            noise_reduction = np.mean(original_noise) / np.mean(filtered_noise)
            
            self.logger.info(f"SID_IMG: Filtering complete - noise reduction factor: {noise_reduction:.2f}x")
            
            return filtered_dff
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to apply temporal filtering: {e}")
            return dff_data  # Return original data if filtering fails


    def plot_filtering_comparison(self, original_dff: np.ndarray, filtered_dff: np.ndarray,
                                  subject_id: str, roi_indices: List[int] = None,
                                  fs: float = 30.0, start_time: float = 0.0,
                                  window_length: float = None,
                                  save_dir: Optional[str] = None,
                                  show: bool = True,
                                  tag: str = "") -> Optional[str]:
        """
        Plot comparison of original vs filtered DFF traces with windowing capability.
        
        Args:
            original_dff: Original DFF traces (n_rois, n_timepoints)
            filtered_dff: Filtered DFF traces (n_rois, n_timepoints)
            subject_id: Subject ID for plot title
            roi_indices: ROI indices to plot (if None, plot first 5)
            fs: Sampling frequency for time axis
            start_time: Start time for viewing window in seconds
            window_length: Length of viewing window in seconds (if None, show all data)
        """
        try:
            import matplotlib.pyplot as plt
            
            if roi_indices is None:
                roi_indices = list(range(min(5, original_dff.shape[0])))
            
            # Create full time vector
            n_timepoints = original_dff.shape[1]
            full_time_vector = np.arange(n_timepoints) / fs
            
            # Determine viewing window
            if window_length is None:
                window_length = full_time_vector[-1] - start_time
            
            end_time = start_time + window_length
            
            # Find indices for the window
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(n_timepoints, end_idx)
            
            if start_idx >= end_idx:
                self.logger.error(f"S2P_DFF: Invalid time window [{start_time}, {end_time}] for data length {full_time_vector[-1]:.1f}s")
                return
            
            # Extract windowed data
            time_vector = full_time_vector[start_idx:end_idx]
            original_window = original_dff[:, start_idx:end_idx]
            filtered_window = filtered_dff[:, start_idx:end_idx]
            
            # Create figure
            fig, axes = plt.subplots(len(roi_indices) + 1, 1, figsize=(14, 2 * len(roi_indices) + 3))
            
            # Update title to show window info
            window_info = f" [Time: {start_time:.1f}-{end_time:.1f}s]" if window_length < full_time_vector[-1] else ""
            fig.suptitle(f'DFF Temporal Filtering Comparison - {subject_id}{window_info}', fontsize=14)
            
            if len(roi_indices) == 1:
                axes = [axes]
            elif len(roi_indices) == 0:
                axes = [axes]
                roi_indices = [0]  # At least plot one ROI
            
            # Color scheme for better differentiation
            original_color = '#808080'  # Gray
            filtered_color = '#1f77b4'   # Blue
            mean_original_color = '#d62728'  # Red
            mean_filtered_color = '#2ca02c'  # Green
            
            # Plot individual ROIs
            for i, roi_idx in enumerate(roi_indices):
                if roi_idx < original_dff.shape[0]:
                    axes[i].plot(time_vector, original_window[roi_idx, :], 
                            color=original_color, alpha=0.8, linewidth=1.5, 
                            label='Original', linestyle='-')
                    axes[i].plot(time_vector, filtered_window[roi_idx, :], 
                            color=filtered_color, linewidth=2.0, 
                            label='Filtered', linestyle='-')
                    
                    axes[i].set_ylabel(f'ROI {roi_idx}\nDFF')
                    axes[i].legend(loc='upper right')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Calculate and display noise metrics for the window
                    orig_noise = np.std(np.diff(original_window[roi_idx, :]))
                    filt_noise = np.std(np.diff(filtered_window[roi_idx, :]))
                    noise_reduction = orig_noise / (filt_noise + 1e-10)
                    
                    # Calculate signal preservation (correlation) for the window
                    correlation = np.corrcoef(original_window[roi_idx, :], filtered_window[roi_idx, :])[0, 1]
                    
                    # Calculate dynamic range for this window
                    orig_range = np.ptp(original_window[roi_idx, :])
                    filt_range = np.ptp(filtered_window[roi_idx, :])
                    
                    # Info box with metrics
                    info_text = f'Noise ↓: {noise_reduction:.1f}x\nCorr: {correlation:.3f}\nRange: {orig_range:.3f}→{filt_range:.3f}'
                    axes[i].text(0.02, 0.95, info_text, 
                            transform=axes[i].transAxes, fontsize=9, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                            verticalalignment='top')
                    
                    # Set y-axis limits with some padding
                    y_min = min(np.min(original_window[roi_idx, :]), np.min(filtered_window[roi_idx, :]))
                    y_max = max(np.max(original_window[roi_idx, :]), np.max(filtered_window[roi_idx, :]))
                    y_padding = 0.1 * (y_max - y_min)
                    axes[i].set_ylim(y_min - y_padding, y_max + y_padding)
            
            # Plot mean across all ROIs
            mean_original = np.mean(original_window, axis=0)
            mean_filtered = np.mean(filtered_window, axis=0)
            
            axes[-1].plot(time_vector, mean_original, 
                        color=mean_original_color, alpha=0.8, linewidth=2.5, 
                        label='Original (mean)', linestyle='-')
            axes[-1].plot(time_vector, mean_filtered, 
                        color=mean_filtered_color, linewidth=3.0, 
                        label='Filtered (mean)', linestyle='-')
            
            axes[-1].set_xlabel('Time (s)')
            axes[-1].set_ylabel('Mean DFF')
            axes[-1].legend(loc='upper right')
            axes[-1].grid(True, alpha=0.3)
            
            # Overall noise reduction and signal preservation for the window
            overall_orig_noise = np.std(np.diff(mean_original))
            overall_filt_noise = np.std(np.diff(mean_filtered))
            overall_noise_reduction = overall_orig_noise / (overall_filt_noise + 1e-10)
            overall_correlation = np.corrcoef(mean_original, mean_filtered)[0, 1]
            
            # Calculate signal-to-noise improvement
            orig_signal = np.std(mean_original)
            filt_signal = np.std(mean_filtered)
            snr_improvement = (filt_signal / overall_filt_noise) / (orig_signal / overall_orig_noise) if overall_orig_noise > 0 else 1
            
            # Enhanced info box for mean plot
            overall_info = (f'Overall noise ↓: {overall_noise_reduction:.1f}x\n'
                        f'Correlation: {overall_correlation:.3f}\n'
                        f'SNR improvement: {snr_improvement:.1f}x')
            
            axes[-1].text(0.02, 0.95, overall_info, 
                        transform=axes[-1].transAxes, fontsize=11, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                        verticalalignment='top')
            
            # Set consistent x-axis limits for all subplots
            for ax in axes:
                ax.set_xlim(time_vector[0], time_vector[-1])
            
            plt.tight_layout()            
            out_file = None
            if save_dir:
                fname = f"{subject_id}_filter_comp_{int(start_time)}s_{int(start_time+window_length if window_length else 0)}s{('_'+tag) if tag else ''}.png"
                out_file = os.path.join(save_dir, fname)
                plt.savefig(out_file, dpi=180, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
            return out_file            
            
            # Enhanced logging with window info
            self.logger.info(f"S2P_DFF: Filtering comparison for {subject_id} (window: {start_time:.1f}-{end_time:.1f}s):")
            self.logger.info(f"S2P_DFF: - Overall noise reduction: {overall_noise_reduction:.2f}x")
            self.logger.info(f"S2P_DFF: - Signal correlation: {overall_correlation:.3f}")
            self.logger.info(f"S2P_DFF: - SNR improvement: {snr_improvement:.2f}x")
            self.logger.info(f"S2P_DFF: - Original DFF range: [{np.min(original_window):.3f}, {np.max(original_window):.3f}]")
            self.logger.info(f"S2P_DFF: - Filtered DFF range: [{np.min(filtered_window):.3f}, {np.max(filtered_window):.3f}]")
            
        except Exception as e:
            self.logger.error(f"S2P_DFF: Failed to create filtering comparison plot: {e}")


    def plot_filtering_comparison_multi_window(self, original_dff: np.ndarray, filtered_dff: np.ndarray,
                                               subject_id: str, roi_indices: List[int] = None,
                                               fs: float = 30.0, windows: List[Tuple[float, float]] = None,
                                               save_dir: Optional[str] = None,
                                               show: bool = True,
                                               tag: str = "") -> Optional[str]:
        """
        Plot filtering comparison across multiple time windows.
        
        Args:
            original_dff: Original DFF traces (n_rois, n_timepoints)
            filtered_dff: Filtered DFF traces (n_rois, n_timepoints)
            subject_id: Subject ID for plot title
            roi_indices: ROI indices to plot (if None, plot first 3)
            fs: Sampling frequency
            windows: List of (start_time, window_length) tuples. If None, use 4 evenly spaced windows
        """
        try:
            import matplotlib.pyplot as plt
            
            if roi_indices is None:
                roi_indices = list(range(min(3, original_dff.shape[0])))
            
            # Create full time vector
            n_timepoints = original_dff.shape[1]
            full_time_vector = np.arange(n_timepoints) / fs
            total_duration = full_time_vector[-1]
            
            # Define default windows if not provided
            if windows is None:
                window_length = total_duration / 4  # 4 windows spanning the entire recording
                windows = [(i * window_length, window_length) for i in range(4)]
            
            n_windows = len(windows)
            n_rois = len(roi_indices)
            
            # Create subplots: rows for ROIs, columns for time windows
            fig, axes = plt.subplots(n_rois, n_windows, figsize=(4 * n_windows, 2 * n_rois))
            fig.suptitle(f'DFF Filtering Comparison - Multiple Windows - {subject_id}', fontsize=14)
            
            if n_rois == 1:
                axes = axes.reshape(1, -1)
            if n_windows == 1:
                axes = axes.reshape(-1, 1)
            
            # Color scheme
            original_color = '#808080'  # Gray
            filtered_color = '#1f77b4'   # Blue
            
            for roi_i, roi_idx in enumerate(roi_indices):
                for win_i, (start_time, window_length) in enumerate(windows):
                    end_time = start_time + window_length
                    
                    # Find indices for this window
                    start_idx = int(start_time * fs)
                    end_idx = int(end_time * fs)
                    start_idx = max(0, start_idx)
                    end_idx = min(n_timepoints, end_idx)
                    
                    if start_idx >= end_idx:
                        continue
                    
                    # Extract windowed data
                    time_vector = full_time_vector[start_idx:end_idx]
                    original_window = original_dff[roi_idx, start_idx:end_idx]
                    filtered_window = filtered_dff[roi_idx, start_idx:end_idx]
                    
                    # Plot
                    ax = axes[roi_i, win_i]
                    ax.plot(time_vector, original_window, 
                        color=original_color, alpha=0.8, linewidth=1.5, label='Original')
                    ax.plot(time_vector, filtered_window, 
                        color=filtered_color, linewidth=2.0, label='Filtered')
                    
                    # Calculate metrics
                    orig_noise = np.std(np.diff(original_window))
                    filt_noise = np.std(np.diff(filtered_window))
                    noise_reduction = orig_noise / (filt_noise + 1e-10)
                    
                    # Formatting
                    ax.set_title(f'ROI {roi_idx} | {start_time:.1f}-{end_time:.1f}s', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    # Add noise reduction info
                    ax.text(0.02, 0.98, f'↓{noise_reduction:.1f}x', 
                        transform=ax.transAxes, fontsize=9, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        verticalalignment='top')
                    
                    # Only add legend to first subplot
                    if roi_i == 0 and win_i == 0:
                        ax.legend(fontsize=9)
                    
                    # Only add y-label to first column
                    if win_i == 0:
                        ax.set_ylabel('DFF')
                    
                    # Only add x-label to bottom row
                    if roi_i == n_rois - 1:
                        ax.set_xlabel('Time (s)')
            
            plt.tight_layout()
            out_file = None
            if save_dir:
                fname = f"{subject_id}_filter_comp_multi{('_'+tag) if tag else ''}.png"
                out_file = os.path.join(save_dir, fname)
                plt.savefig(out_file, dpi=180, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
            return out_file
            
            
            
            self.logger.info(f"S2P_DFF: Multi-window filtering comparison completed for {subject_id}")
            
        except Exception as e:
            self.logger.error(f"S2P_DFF: Failed to create multi-window filtering comparison: {e}")


    # def plot_filtering_comparison(self, original_dff: np.ndarray, filtered_dff: np.ndarray, 
    #                             subject_id: str, roi_indices: List[int] = None, 
    #                             fs: float = 30.0) -> None:
    #     """
    #     Plot comparison of original vs filtered DFF traces.
        
    #     Args:
    #         original_dff: Original DFF traces (n_rois, n_timepoints)
    #         filtered_dff: Filtered DFF traces (n_rois, n_timepoints)
    #         subject_id: Subject ID for plot title
    #         roi_indices: ROI indices to plot (if None, plot first 5)
    #         fs: Sampling frequency for time axis
    #     """
    #     try:
    #         import matplotlib.pyplot as plt
            
    #         if roi_indices is None:
    #             roi_indices = list(range(min(5, original_dff.shape[0])))
            
    #         # Create time vector
    #         n_timepoints = original_dff.shape[1]
    #         time_vector = np.arange(n_timepoints) / fs
            
    #         fig, axes = plt.subplots(len(roi_indices) + 1, 1, figsize=(14, 2 * len(roi_indices) + 3))
    #         fig.suptitle(f'DFF Temporal Filtering Comparison - {subject_id}', fontsize=14)
            
    #         if len(roi_indices) == 1:
    #             axes = [axes]
    #         elif len(roi_indices) == 0:
    #             axes = [axes]
    #             roi_indices = [0]  # At least plot one ROI
            
    #         # Plot individual ROIs
    #         for i, roi_idx in enumerate(roi_indices):
    #             if roi_idx < original_dff.shape[0]:
    #                 axes[i].plot(time_vector, original_dff[roi_idx, :], 'gray', alpha=0.7, 
    #                         linewidth=1, label='Original')
    #                 axes[i].plot(time_vector, filtered_dff[roi_idx, :], 'blue', 
    #                         linewidth=1.5, label='Filtered')
    #                 axes[i].set_ylabel(f'ROI {roi_idx}\nDFF')
    #                 axes[i].legend()
    #                 axes[i].grid(True, alpha=0.3)
                    
    #                 # Calculate and display noise metrics
    #                 orig_noise = np.std(np.diff(original_dff[roi_idx, :]))
    #                 filt_noise = np.std(np.diff(filtered_dff[roi_idx, :]))
    #                 noise_reduction = orig_noise / (filt_noise + 1e-10)
                    
    #                 # Calculate signal preservation (correlation)
    #                 correlation = np.corrcoef(original_dff[roi_idx, :], filtered_dff[roi_idx, :])[0, 1]
                    
    #                 axes[i].text(0.02, 0.95, f'Noise ↓: {noise_reduction:.1f}x\nCorr: {correlation:.3f}', 
    #                         transform=axes[i].transAxes, fontsize=10, 
    #                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    #         # Plot mean across all ROIs
    #         mean_original = np.mean(original_dff, axis=0)
    #         mean_filtered = np.mean(filtered_dff, axis=0)
            
    #         axes[-1].plot(time_vector, mean_original, 'gray', alpha=0.7, 
    #                     linewidth=1.5, label='Original (mean)')
    #         axes[-1].plot(time_vector, mean_filtered, 'red', 
    #                     linewidth=2, label='Filtered (mean)')
    #         axes[-1].set_xlabel('Time (s)')
    #         axes[-1].set_ylabel('Mean DFF')
    #         axes[-1].legend()
    #         axes[-1].grid(True, alpha=0.3)
            
    #         # Overall noise reduction and signal preservation
    #         overall_orig_noise = np.std(np.diff(mean_original))
    #         overall_filt_noise = np.std(np.diff(mean_filtered))
    #         overall_noise_reduction = overall_orig_noise / (overall_filt_noise + 1e-10)
    #         overall_correlation = np.corrcoef(mean_original, mean_filtered)[0, 1]
            
    #         axes[-1].text(0.02, 0.95, f'Overall noise ↓: {overall_noise_reduction:.1f}x\nCorrelation: {overall_correlation:.3f}', 
    #                     transform=axes[-1].transAxes, fontsize=12, weight='bold',
    #                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
    #         plt.tight_layout()
    #         plt.show()
            
    #         # Log summary statistics
    #         self.logger.info(f"S2P_DFF: Filtering comparison for {subject_id}:")
    #         self.logger.info(f"S2P_DFF: - Overall noise reduction: {overall_noise_reduction:.2f}x")
    #         self.logger.info(f"S2P_DFF: - Signal correlation: {overall_correlation:.3f}")
    #         self.logger.info(f"S2P_DFF: - Original DFF range: [{np.min(original_dff):.3f}, {np.max(original_dff):.3f}]")
    #         self.logger.info(f"S2P_DFF: - Filtered DFF range: [{np.min(filtered_dff):.3f}, {np.max(filtered_dff):.3f}]")
            
    #     except Exception as e:
    #         self.logger.error(f"S2P_DFF: Failed to create filtering comparison plot: {e}")


    def plot_power_spectrum_comparison(self, original_dff: np.ndarray, filtered_dff: np.ndarray,
                                       subject_id: str, fs: float = 30.0, roi_idx: int = 0,
                                       save_dir: Optional[str] = None,
                                       show: bool = True,
                                       tag: str = "") -> Optional[str]:
        """
        Plot power spectrum comparison to show filtering effects in frequency domain.
        
        Args:
            original_dff: Original DFF traces
            filtered_dff: Filtered DFF traces
            subject_id: Subject ID for plot title
            fs: Sampling frequency
            roi_idx: ROI index to analyze (default: 0)
        """
        try:
            import matplotlib.pyplot as plt
            from scipy import signal
            
            if roi_idx >= original_dff.shape[0]:
                roi_idx = 0
            
            # Compute power spectral density
            freqs_orig, psd_orig = signal.welch(original_dff[roi_idx, :], fs=fs, nperseg=min(1024, original_dff.shape[1]//4))
            freqs_filt, psd_filt = signal.welch(filtered_dff[roi_idx, :], fs=fs, nperseg=min(1024, filtered_dff.shape[1]//4))
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f'Power Spectrum Analysis - {subject_id} (ROI {roi_idx})', fontsize=14)
            
            # Plot power spectra
            ax1.loglog(freqs_orig, psd_orig, 'gray', alpha=0.7, label='Original')
            ax1.loglog(freqs_filt, psd_filt, 'blue', linewidth=2, label='Filtered')
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Power Spectral Density')
            ax1.set_title('Power Spectrum Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot attenuation (dB reduction)
            # Interpolate to common frequency grid
            common_freqs = freqs_orig
            psd_filt_interp = np.interp(common_freqs, freqs_filt, psd_filt)
            
            attenuation_db = 10 * np.log10(psd_orig / (psd_filt_interp + 1e-12))
            
            ax2.semilogx(common_freqs, attenuation_db, 'red', linewidth=2)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Attenuation (dB)')
            ax2.set_title('Frequency Attenuation')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            out_file = None
            if save_dir:
                fname = f"{subject_id}_power_ROI{roi_idx}{('_'+tag) if tag else ''}.png"
                out_file = os.path.join(save_dir, fname)
                plt.savefig(out_file, dpi=180, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
            return out_file
        except Exception as e:
            self.logger.error(f"S2P_DFF: Failed to create power spectrum comparison: {e}")



    def apply_savgol_filter(self, dff: np.ndarray, window_length: int = 9, polyorder: int = 3) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to DFF traces for smoothing.
        
        Args:
            dff: DFF traces array (n_cells, n_timepoints)
            window_length: Length of the filter window (must be odd)
            polyorder: Order of the polynomial used for filtering
            
        Returns:
            Filtered DFF traces
        """
        try:
            from scipy.signal import savgol_filter
            
            # Ensure window_length is odd
            if window_length % 2 == 0:
                window_length += 1
                self.logger.warning(f"S2P_DFF: Adjusted window_length to {window_length} (must be odd)")
            
            # Ensure window_length is not larger than the data
            n_timepoints = dff.shape[1]
            if window_length > n_timepoints:
                window_length = n_timepoints if n_timepoints % 2 == 1 else n_timepoints - 1
                self.logger.warning(f"S2P_DFF: Adjusted window_length to {window_length} (data too short)")
            
            # Ensure polyorder is less than window_length
            if polyorder >= window_length:
                polyorder = window_length - 1
                self.logger.warning(f"S2P_DFF: Adjusted polyorder to {polyorder} (must be < window_length)")
            
            self.logger.info(f"S2P_DFF: Applying Savitzky-Golay filter (window={window_length}, polyorder={polyorder})")
            
            # Apply filter to each cell
            dff_filtered = np.apply_along_axis(
                savgol_filter, 1, dff,
                window_length=window_length,
                polyorder=polyorder
            )
            
            # Calculate smoothing effect
            original_noise = np.std(np.diff(dff, axis=1), axis=1)
            filtered_noise = np.std(np.diff(dff_filtered, axis=1), axis=1)
            noise_reduction = np.mean(original_noise) / (np.mean(filtered_noise) + 1e-10)
            
            self.logger.info(f"S2P_DFF: Savitzky-Golay filtering complete - noise reduction: {noise_reduction:.2f}x")
            
            return dff_filtered
            
        except Exception as e:
            self.logger.error(f"S2P_DFF: Failed to apply Savitzky-Golay filter: {e}")
            return dff  # Return original if filtering fails


    def plot_dff_windowed_rasters(self, dff_data, time_data, subject_id, window_length_sec=60,
                                  overlap_sec=10, sort_by_peak=True, save_path=None,
                                  fs=30.0, colormap='viridis',
                                  show: bool = True) -> list:
        """
        Plot windowed raster heatmaps of DFF data with optional ROI sorting by peak response.
        Creates separate plots for each window, not subplots.
        
        Args:
            dff_data (numpy.ndarray): DFF traces array (n_rois, n_timepoints)
            time_data (numpy.ndarray): Time vector corresponding to DFF data
            subject_id (str): Subject identifier for plot titles
            window_length_sec (float): Length of each window in seconds
            overlap_sec (float): Overlap between windows in seconds
            sort_by_peak (bool): Whether to sort ROIs by peak response magnitude
            save_path (str, optional): Path to save plots. If None, only display
            fs (float): Sampling frequency in Hz
            colormap (str): Colormap for heatmaps ('viridis', 'plasma', 'RdBu_r', etc.)
        
        Returns:
            list: List of dictionaries containing window info and ROI sorting order
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        try:
            n_rois, n_timepoints = dff_data.shape
            total_duration = time_data[-1] - time_data[0]
            
            # Calculate window parameters
            window_length_frames = int(window_length_sec * fs)
            overlap_frames = int(overlap_sec * fs)
            step_size_frames = window_length_frames - overlap_frames
            
            # Generate window start times to cover entire DFF recording
            window_starts = []
            start_frame = 0
            while start_frame < n_timepoints:
                window_starts.append(start_frame)
                start_frame += step_size_frames
                # Stop if we've covered the entire recording
                if start_frame >= n_timepoints:
                    break
            
            n_windows = len(window_starts)
            print(f"Processing {n_windows} windows of {window_length_sec}s each for {subject_id}")
            print(f"Total recording duration: {total_duration:.1f}s")
            
            # Sort ROIs by peak response if requested (calculated across entire recording)
            if sort_by_peak:
                roi_peak_magnitudes = np.max(np.abs(dff_data), axis=1)
                roi_sort_order = np.argsort(roi_peak_magnitudes)[::-1]  # Highest to lowest
                print(f"ROIs sorted by peak magnitude (range: {np.min(roi_peak_magnitudes):.3f} - {np.max(roi_peak_magnitudes):.3f})")
            else:
                roi_sort_order = np.arange(n_rois)
                roi_peak_magnitudes = None
            
            window_info = []
            
            # Loop through entire DFF, creating separate plot for each window
            for i, start_frame in enumerate(window_starts):
                # Extract window data
                end_frame = min(start_frame + window_length_frames, n_timepoints)
                window_dff = dff_data[roi_sort_order, start_frame:end_frame]
                window_time = time_data[start_frame:end_frame]
                
                # Skip if window is too small
                if end_frame - start_frame < 10:  # Skip very small windows
                    continue
                
                # Create individual figure for this window
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                # Calculate colormap range for this window
                vmin = np.percentile(window_dff, 5)
                vmax = np.percentile(window_dff, 95)
                
                # Create heatmap
                im = ax.imshow(window_dff, aspect='auto', cmap=colormap,
                            extent=[window_time[0], window_time[-1], 0, n_rois],
                            vmin=vmin, vmax=vmax, interpolation='nearest')
                
                # Formatting
                start_time = window_time[0]
                end_time = window_time[-1]
                window_duration = end_time - start_time
                
                ax.set_title(f'DFF Raster - {subject_id} - Window {i+1}/{n_windows}\n'
                            f'Time: {start_time:.1f} - {end_time:.1f}s (Duration: {window_duration:.1f}s)', 
                            fontsize=14)
                ax.set_xlabel('Time (s)', fontsize=12)
                ax.set_ylabel('ROI # (sorted by peak)' if sort_by_peak else 'ROI #', fontsize=12)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('DFF', fontsize=12)
                
                # Add ROI information text box
                if sort_by_peak and i == 0:  # Only on first window
                    top_5_rois = roi_sort_order[:5]
                    top_5_peaks = roi_peak_magnitudes[top_5_rois]
                    info_text = 'Top 5 Most Responsive ROIs:\n' + '\n'.join([
                        f'ROI {roi}: {peak:.3f}' 
                        for roi, peak in zip(top_5_rois, top_5_peaks)
                    ])
                    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                        fontsize=10, verticalalignment='top')
                
                # Add window statistics
                stats_text = (f'ROIs: {n_rois}\n'
                            f'Data range: [{vmin:.3f}, {vmax:.3f}]\n'
                            f'Window {i+1}/{n_windows}')
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=9, verticalalignment='top', horizontalalignment='right')
                
                plt.tight_layout()
                
                # Save individual window plot if path provided
                if save_path:
                    filename = f'{subject_id}_dff_raster_window_{i+1:03d}_{start_time:.1f}s-{end_time:.1f}s.png'
                    full_path = os.path.join(save_path, filename)
                    plt.savefig(full_path, dpi=300, bbox_inches='tight')
                    print(f"Saved window {i+1} raster to: {filename}")
                if show:
                    plt.show()
                else:
                    plt.close(fig)                                    
                
                # Store window information
                window_info.append({
                    'window_index': i,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time': start_time,
                    'end_time': end_time,
                    'window_duration': window_duration,
                    'roi_sort_order': roi_sort_order.copy(),
                    'peak_magnitudes': roi_peak_magnitudes[roi_sort_order] if sort_by_peak else None,
                    'data_range': (vmin, vmax),
                    'n_rois_plotted': n_rois,
                    'filename': f'{subject_id}_dff_raster_window_{i+1:03d}_{start_time:.1f}s-{end_time:.1f}s.png' if save_path else None
                })
                
                # Optional: Close figure to save memory if many windows
                plt.close(fig)
            
            # Print final summary
            total_windows_created = len(window_info)
            print(f"\nDFF Windowed Raster Summary for {subject_id}:")
            print(f"  Total ROIs: {n_rois}")
            print(f"  Total recording duration: {total_duration:.1f}s")
            print(f"  Windows created: {total_windows_created}")
            print(f"  Window length: {window_length_sec}s")
            print(f"  Window overlap: {overlap_sec}s")
            print(f"  ROI sorting: {'By peak magnitude' if sort_by_peak else 'Original order'}")
            if sort_by_peak and roi_peak_magnitudes is not None:
                print(f"  Peak magnitude range: {np.min(roi_peak_magnitudes):.3f} - {np.max(roi_peak_magnitudes):.3f}")
            if save_path:
                print(f"  Plots saved to: {save_path}")
            
            return window_info
        
        except Exception as e:
            print(f"Error plotting windowed rasters for {subject_id}: {e}")
            return []








    def process_subject(self, subject_id: str, suite2p_path: str, output_path: str, force: bool = False) -> Dict[str, Any]:
        """
        Process dF/F extraction for a single subject.
        
        Args:
            subject_id: Subject identifier
            suite2p_path: Path to Suite2p output directory (plane0)
            output_path: Path for dF/F output
            force: Force reprocessing even if output exists
            
        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f"S2P_DFF: ========== Starting dF/F processing for {subject_id} ==========")
            self.logger.info(f"S2P_DFF: Suite2p path: {suite2p_path}")
            self.logger.info(f"S2P_DFF: Output path: {output_path}")
            
            # Check if already processed
            dff_file = os.path.join(output_path, 'dff.h5')
            spks_file = os.path.join(output_path, 'spks.h5')
            
            have_dff = os.path.exists(dff_file)
            have_spks = os.path.exists(spks_file)

            force = True

            if have_dff and have_spks and not force:
                self.logger.info(f"S2P_DFF: dF/F and spike data already exists for {subject_id}, skipping (use force=True to reprocess)")
                return {
                    'success': True,
                    'sessions_processed': 1,
                    'message': 'Already processed (skipped)'
                }


            # Load required data
            data = self.load_dff_data(subject_id, suite2p_path, output_path)
            if data is None:
                return {
                    'success': False,
                    'sessions_processed': 0,
                    'error_message': 'Failed to load dF/F data'
                }

            if not have_dff or force:

                

                # Get trace extraction parameters from config
                experiment_config = self.config_manager.get_experiment_config()
                imaging_config = experiment_config.get('imaging_preprocessing', {})
                trace_config = imaging_config.get('trace_extraction', {})
                
                normalize = trace_config.get('normalize', True)
                baseline_sigma = trace_config.get('baseline_sigma', 600)
                outlier_threshold = trace_config.get('outlier_threshold', 5.0)
                outlier_method = trace_config.get('outlier_method', 'zscore')
                
                self.logger.info(f"S2P_DFF: Using normalize={normalize}, baseline_sigma={baseline_sigma}")
                self.logger.info(f"S2P_DFF: Outlier detection - method={outlier_method}, threshold={outlier_threshold}")
                
                # # todo add separate spike loading later
                # spks = data['spks']
            
            if not have_spks:
                # Compute spks with oasis deconvolution
                spks = self.compute_spks(
                    data['F'], 
                    data['Fneu'], 
                    data['ops'],
                )
                
                # Save spks results
                success = self.save_spks_results(output_path, spks)

                if success:
                    self.logger.info(f"S2P_DFF: ========== Successfully completed spks processing for {subject_id} ==========")
                else:
                    self.logger.error(f"S2P_DFF: ========== Failed spks processing for {subject_id} ==========")

            if not have_dff or force:
                # Compute dF/F traces
                dff = self.compute_dff(
                    data['F'], 
                    data['Fneu'], 
                    data['ops'],
                    normalize=normalize,
                    sig_baseline=baseline_sigma,
                    outlier_threshold=outlier_threshold,
                    outlier_method=outlier_method
                )


                # Store original DFF for comparison
                dff_original = dff.copy()
                
                # Apply Savitzky-Golay filter for additional smoothing
                # dff_filtered = self.apply_savgol_filter(dff, window_length=9, polyorder=3)
                dff_filtered = dff  # filtering handled inside compute_dff via post_filter config
                
                # # todo add separate spike loading later
                # dff_filtered['spks'] = spks            

                # Plot filtering comparison (optional - you can control this with a parameter)
                # after computing dff_original / dff_filtered
                check_dir = self._ensure_check_trace_dir(output_path)                
                show_filtering_plots = True  # Set to False to disable plots
                if show_filtering_plots:
                    # Show full recording
                    self.plot_filtering_comparison(dff_original, dff_filtered, subject_id,
                                                   fs=30.0, start_time=120.0, window_length=10.0,
                                                   save_dir=check_dir, show=False, tag='w1')
                    # Show specific 60-second window starting at 120 seconds
                    self.plot_filtering_comparison(dff_original, dff_filtered, subject_id,
                                                   fs=30.0, start_time=20.0, window_length=10.0,
                                                   save_dir=check_dir, show=False, tag='w2')
                    # Show first 30 seconds
                    self.plot_filtering_comparison(dff_original, dff_filtered, subject_id,
                                                   fs=30.0, start_time=120.0, window_length=10.0,
                                                   save_dir=check_dir, show=False, tag='w3_dup')
                    # Show multiple windows across the recording
                    self.plot_filtering_comparison_multi_window(dff_original, dff_filtered, subject_id,
                                                                fs=30.0, save_dir=check_dir, show=False)
                    # Also show power spectrum for first ROI
                    self.plot_power_spectrum_comparison(dff_original, dff_filtered, subject_id,
                                                        fs=30.0, roi_idx=0, save_dir=check_dir, show=False)
                # optional: windowed rasters
                # time vector for rasters
                exp_cfg = self.config_manager.get_experiment_config()
                ip_cfg = exp_cfg.get('imaging_preprocessing', {})
                tr_cfg = ip_cfg.get('trace_extraction', {})
                fs = float(tr_cfg.get('fs_hz', 29.760441593958042))
                time_vec = np.arange(dff_filtered.shape[1]) / fs
                self.plot_dff_windowed_rasters(dff_filtered, time_vec, subject_id,
                                               window_length_sec=60, overlap_sec=30,
                                               save_path=check_dir, show=False)                

                # return
                
                

                # Save results
                # TODO need to move loading/deriving spks into initial suite2p modules
                success = self.save_dff_results(output_path, dff_filtered)

                # Run post-dFF QC (optional)
                qc_out = self._run_post_dff_qc(output_path, data['ops'], dff_filtered)

                if success:
                    self.logger.info(f"S2P_DFF: ========== Successfully completed dF/F processing for {subject_id} ==========")
                else:
                    self.logger.error(f"S2P_DFF: ========== Failed dF/F processing for {subject_id} ==========")
            
            return {
                'success': success,
                'sessions_processed': 1 if success else 0,
                'dff_stats': {
                    'n_cells': dff.shape[0],
                    'n_timepoints': dff.shape[1],
                    'normalize': normalize,
                    'baseline_sigma': baseline_sigma,
                    'neucoeff': data['ops'].get('neucoeff', 0.7)
                },
                'error_message': None if success else 'Failed to save dF/F results'
            }
            
        except Exception as e:
            self.logger.error(f"S2P_DFF: Processing failed for {subject_id}: {e}")
            return {
                'success': False,
                'sessions_processed': 0,
                'error_message': str(e)
            }
    

