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
import matplotlib.pyplot as plt

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
        
        self._checktrace_enabled = False
        
        self.logger.info("S2P_DFF: Suite2pDffTraces initialized")

    def _ensure_check_trace_dir(self, output_path: str) -> str:
        path = os.path.join(output_path, 'qc_results', 'check_trace')
        os.makedirs(path, exist_ok=True)
        return path

    # ========== STEP A: CHECKTRACE CAPTURE & SINGLE-ROI PIPELINE PLOT ==========

    def _init_checktrace(self, output_path: str):
        """
        Initialize checktrace capture based on config.
        """
        try:
            exp_cfg = self.config_manager.get_experiment_config()
            tcfg = (exp_cfg.get('imaging_preprocessing', {})
                    .get('trace_extraction', {})
                    .get('outputs', {}))
        except Exception:
            tcfg = {}        
        self._checktrace_cfg = tcfg.get('checktrace', {})
        self._checktrace_enabled = bool(tcfg.get('write_checktrace', False))
        self._single_folder = bool(self._checktrace_cfg.get('storage', {}).get('single_folder', False))
        self._checktrace_root = os.path.join(output_path, 'qc_results', 'check_trace')
        if self._checktrace_enabled:
            os.makedirs(self._checktrace_root, exist_ok=True)
        self._stage_capture = {}  # name -> ndarray reference
        self._checktrace_meta = {}  # per ROI metadata accumulation

    def _capture_stage(self, name: str, arr: np.ndarray, persist: str = 'ref'):
        """
        Store reference to processing stage arrays. Only if enabled.
        """
        if not getattr(self, '_checktrace_enabled', False):
            return
        if arr is None:
            return
        if persist == 'copy':
            self._stage_capture[name] = np.array(arr, copy=True)
        else:
            self._stage_capture[name] = arr  # assume not mutated later destructively

    def _select_checktrace_rois(self, dff: np.ndarray, metrics_df=None):
        """
        Decide which ROI IDs to plot (1-based) based on config.
        """
        cfg = self._checktrace_cfg.get('roi_selection', {})
        mode = cfg.get('mode', 'top_snr')
        n = int(cfg.get('n', 12))
        rng = np.random.default_rng(cfg.get('seed', 17))
        total = dff.shape[0]
        all_ids = np.arange(1, total+1)
        labels = None
        keep_mask = None
        if metrics_df is not None:
            if 'roi_id' in metrics_df.columns:
                metrics_df = metrics_df.sort_values('roi_id')
            labels = metrics_df.get('final_label', None)
            if 'keep_dff' in metrics_df.columns:
                keep_mask = metrics_df['keep_dff'].values.astype(bool)

        if cfg.get('require_keep', False) and keep_mask is not None:
            eligible = all_ids[keep_mask]
        else:
            eligible = all_ids

        # if cfg.get('label_filter') and labels is not None:
        #     lf = set(cfg.get('label_filter'))
        #     label_arr = np.array(labels)
        #     mask = np.array([lab in lf for lab in label_arr])
        #     eligible = eligible[mask[:eligible.size]]        
        if cfg.get('label_filter') and labels is not None:
            lf = set(cfg['label_filter'])
            label_arr = np.asarray(labels)
            ok_ids = np.where(np.isin(label_arr, list(lf)))[0] + 1  # 1-based ROI ids
            eligible = eligible[np.isin(eligible, ok_ids)]        
        if mode == 'ids':
            ids = [i for i in cfg.get('ids', []) if 1 <= i <= total]
            return ids[:n]
        elif mode == 'random':
            if eligible.size <= n:
                return list(eligible)
            return list(rng.choice(eligible, size=n, replace=False))
        elif mode == 'all':
            return list(eligible)
        else:
            # Need SNR; derive quick proxy if metrics_df missing
            # if metrics_df is not None and 'snr' in metrics_df.columns:
            #     snr = metrics_df['snr'].values
            if metrics_df is not None and 'snr' in metrics_df.columns:
                snr = metrics_df['snr'].values
            else:
                med = np.nanmedian(dff, axis=1)
                mad = 1.4826 * np.nanmedian(np.abs(dff - med[:, None]), axis=1) + 1e-6            
            # else:
            #     # quick MAD-based SNR proxy
            #     med = np.nanmedian(dff, axis=1)
            #     mad = 1.4826 * np.nanmedian(np.abs(dff - med[:, None]), axis=1) + 1e-9
            #     # approximate "event" amplitude = p95
            #     p95 = np.nanpercentile(dff, 95, axis=1)
            #     snr = (p95 - med) / mad
            order = np.argsort(-snr)  # descending
            ranked = all_ids[order]
            ranked = np.array([r for r in ranked if r in eligible])
            return list(ranked[:n])


    def _robust_std(self, x):
        x = np.asarray(x)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med)) + 1e-9
        return 1.4826 * mad

    def _make_template_kernel(self, fs, rise_s, decay_s, len_s):
        t = np.arange(int(len_s * fs)) / fs
        # simple bi-exponential (normalized)
        k = (1 - np.exp(-t / max(rise_s, 1e-3))) * np.exp(-t / max(decay_s, 1e-3))
        k /= np.max(k) + 1e-9
        return k.astype(np.float32)

    def _change_point_zscore(self, x, min_gap):
        # simple max absolute diff over min_gap frames normalized by robust std
        if x.size < 5:
            return 0.0
        diffs = np.abs(np.diff(x))
        if min_gap > 1:
            # aggregate over window
            import numpy as _np
            k = int(min_gap)
            if k < 1: k = 1
            agg = _np.convolve(diffs, np.ones(k), 'valid') / k
            d = agg
        else:
            d = diffs
        rsd = self._robust_std(x) + 1e-9
        return float(np.max(d) / rsd)

    def _extract_events(self, z, thr_z, refractory_frames):
        # returns list of (onset, peak)
        events = []
        above = z > thr_z
        i = 0
        N = z.size
        while i < N:
            if above[i]:
                onset = i
                # hunt peak until drop or end
                peak = i
                while i+1 < N and z[i+1] >= z[i]:
                    i += 1
                    peak = i
                events.append((onset, peak))
                i = peak + refractory_frames
            else:
                i += 1
        return events


    def _choose_window_for_roi(self, T: int, fs: float):
        """
        Return list of (start_frame, end_frame) windows for plotting.
        Supports explicit segment pairs in seconds via config:
          windows.explicit_segments_s:
            - [start_s, end_s]
            - [ ... ]
        """
        cfg = getattr(self, '_checktrace_cfg', {}).get('windows', {})
        seg_len_s = float(cfg.get('segment_length_s', 8))
        seg_len = max(1, int(round(seg_len_s * fs)))
        n_segments = int(cfg.get('n_segments', 4))

        # ----- NEW: explicit segments override -----
        explicit = cfg.get('explicit_segments_s', [])
        use_explicit_only = cfg.get('use_explicit_only', True)
        windows = []
        if explicit:
            for pair in explicit:
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    continue
                s_s, e_s = pair
                try:
                    s_f = int(round(float(s_s) * fs))
                    e_f = int(round(float(e_s) * fs))
                except Exception:
                    continue
                if e_f <= s_f:
                    continue
                # Clamp
                s_f = max(0, min(T - 1, s_f))
                e_f = max(s_f + 1, min(T, e_f))
                windows.append((s_f, e_f))
                if len(windows) >= n_segments:
                    break
            if use_explicit_only:
                return windows[:n_segments]
            # else fall through and (optionally) add more below if fewer than n
        # -------------------------------------------

        if not windows and (seg_len >= T):
            return [(0, T)]

        # Fallback to existing strategy (keep your prior logic here) ...
        # SIMPLE fallback: evenly spaced
        if not windows:
            import numpy as _np
            starts = _np.linspace(0, max(0, T - seg_len), n_segments, dtype=int)
            windows = [(s, s + seg_len) for s in starts]

        return windows[:n_segments]

    def _qc_metric_overlay(self, metrics_row: dict) -> str:
        """
        Compact overlay string including unified core + guardrails summary.
        """
        if not metrics_row:
            return ""
        import numpy as np
        try:
            exp_cfg = self.config_manager.get_experiment_config()
            qc_cfg = exp_cfg.get('imaging_preprocessing', {}).get('post_dff_qc', {})
        except Exception:
            qc_cfg = {}
        common = qc_cfg.get('common', {})
        use_rel = bool(common.get('use_relative_drift', True))

        thr = dict(
            snr_min = common.get('snr_min'),
            tpl_min = common.get('template_corr_p95_min'),
            art_max = common.get('artifact_frac_max'),
            drift_rel = common.get('drift_rel_slope_max_per_min'),
            drift_abs = common.get('drift_abs_slope_max_per_min'),
            mot_max = common.get('motion_corr_max'),
            trans_min = common.get('transientness_min'),
            min_dff_min = common.get('min_dff_min'),
            dff_abs_max = common.get('dff_abs_max'),          # renamed key
            neg_min = common.get('negativity_frac_min'),
            shape_lo = common.get('ca_shape_frac_lo', common.get('ca_shape_frac_min')),
            shape_hi = common.get('ca_shape_frac_hi'),
            gr_need = int(common.get('guardrails_require_pass', 2))
        )
        if thr['art_max'] is None:
            az = common.get('artifact_z_max', common.get('artifact_z'))
            thr['art_max'] = (az / 100.0) if az is not None else None

        drift_key = 'drift_rel_slope' if (use_rel and 'drift_rel_slope' in metrics_row) else 'drift_abs_slope'
        drift_thr = thr['drift_rel'] if 'rel' in drift_key else thr['drift_abs']

        def fmt(v):
            if v is None or (isinstance(v,float) and np.isnan(v)): return "NA"
            if isinstance(v,(int,float)):
                return f"{v:.2f}" if abs(v) < 100 else f"{v:.2e}"
            return str(v)

        segs = []
        def add(val, name, sym, limit):
            if val is None: return
            ok = True
            if limit is not None and isinstance(val,(int,float)) and not np.isnan(val):
                ok = (val >= limit) if sym == '>=' else (val <= limit) if sym in ('<=','≤') else True
            crit = f"({sym}{fmt(limit)})" if limit is not None else ""
            segs.append(f"{name}={fmt(val)}{crit}{'✓' if ok else '✗'}")

        add(metrics_row.get('snr'), 'SNR', '>=', thr['snr_min'])
        add(metrics_row.get('template_corr_p95'), 'Tpl95', '>=', thr['tpl_min'])
        add(metrics_row.get('artifact_frac'), 'ArtFrac', '<=', thr['art_max'])
        add(metrics_row.get(drift_key), 'RelDrift' if 'rel' in drift_key else 'AbsDrift', '<=', drift_thr)
        add(metrics_row.get('motion_corr'), 'Mot', '<=', thr['mot_max'])
        add(metrics_row.get('transientness'), 'Trans', '>=', thr['trans_min'])
        add(metrics_row.get('dff_max_abs'), 'dFFabs', '<=', thr['dff_abs_max'])

        # Guardrails summary (already computed in decide_keep)
        gr_ct = metrics_row.get('guardrails_count')
        if gr_ct is not None:
            segs.append(f"Guardrails={gr_ct}/{thr['gr_need']}{'✓' if gr_ct >= thr['gr_need'] else '✗'}")

        if 'keep_dff' in metrics_row:
            segs.append(f"KEEP={'Y' if int(metrics_row['keep_dff'])==1 else 'N'}")
        return "  ".join(segs)



    def _qc_metric_table_rows(self, metrics_row: dict) -> List[Tuple[str, str, str]]:
        """
        Detailed table rows for unified QC (core + guardrails).
        """
        import numpy as np
        rows: List[Tuple[str,str,str]] = []
        if not metrics_row:
            return rows
        try:
            exp_cfg = self.config_manager.get_experiment_config()
            qc_cfg = exp_cfg.get('imaging_preprocessing', {}).get('post_dff_qc', {})
        except Exception:
            qc_cfg = {}
        common = qc_cfg.get('common', {})

        thr = dict(
            snr_min = common.get('snr_min'),
            tpl_min = common.get('template_corr_p95_min'),
            art_max = common.get('artifact_frac_max'),
            drift_rel = common.get('drift_rel_slope_max_per_min'),
            drift_abs = common.get('drift_abs_slope_max_per_min'),
            mot_max = common.get('motion_corr_max'),
            trans_min = common.get('transientness_min'),
            min_dff_min = common.get('min_dff_min'),
            dff_abs_max = common.get('dff_abs_max'),
            neg_min = common.get('negativity_frac_min'),
            shape_lo = common.get('ca_shape_frac_lo', common.get('ca_shape_frac_min')),
            shape_hi = common.get('ca_shape_frac_hi'),
            gr_need = int(common.get('guardrails_require_pass', 2))
        )
        if thr['art_max'] is None:
            az = common.get('artifact_z_max', common.get('artifact_z'))
            thr['art_max'] = (az / 100.0) if az is not None else None
        use_rel = bool(common.get('use_relative_drift', True))
        drift_key = 'drift_rel_slope' if (use_rel and 'drift_rel_slope' in metrics_row) else 'drift_abs_slope'
        drift_thr = thr['drift_rel'] if 'rel' in drift_key else thr['drift_abs']

        def fmtv(v):
            if v is None or (isinstance(v,float) and np.isnan(v)): return "NA"
            return f"{v:.3f}"

        def line(val, name, mode, limit):
            if limit is None:
                return (f"{name}={fmtv(val)}", "-", "PASS")
            ok = False
            if val is not None and not (isinstance(val,float) and np.isnan(val)):
                if mode == 'ge': ok = val >= limit
                elif mode == 'le': ok = val <= limit
            return (f"{name}={fmtv(val)}", f"{'>=' if mode=='ge' else '<='} {limit}", "PASS" if ok else "FAIL")

        rows.append(line(metrics_row.get('snr'), 'SNR', 'ge', thr['snr_min']))
        rows.append(line(metrics_row.get('template_corr_p95'), 'TemplateCorrP95', 'ge', thr['tpl_min']))
        rows.append(line(metrics_row.get('artifact_frac'), 'ArtifactFrac', 'le', thr['art_max']))
        rows.append(line(metrics_row.get(drift_key), 'RelDrift' if 'rel' in drift_key else 'AbsDrift', 'le', drift_thr))
        rows.append(line(metrics_row.get('motion_corr'), 'MotionCorr', 'le', thr['mot_max']))
        rows.append(line(metrics_row.get('transientness'), 'Transientness', 'ge', thr['trans_min']))

        if 'dff_max_abs' in metrics_row:
            rows.append(line(metrics_row.get('dff_max_abs'), 'dFFmaxAbs', 'le', thr['dff_abs_max']))

        # Guardrails
        if thr['min_dff_min'] is not None and 'min_dff' in metrics_row:
            rows.append(line(metrics_row['min_dff'], 'MinDFF', 'ge', thr['min_dff_min']))
        neg_key = 'negativity_frac' if 'negativity_frac' in metrics_row else ('neg_frac' if 'neg_frac' in metrics_row else None)
        if neg_key and thr['neg_min'] is not None:
            rows.append(line(metrics_row[neg_key], 'NegFrac', 'ge', thr['neg_min']))
        if 'ca_shape_frac' in metrics_row and (thr['shape_lo'] is not None or thr['shape_hi'] is not None):
            v = metrics_row['ca_shape_frac']
            ok = True
            crit_parts = []
            if thr['shape_lo'] is not None:
                crit_parts.append(f">={thr['shape_lo']}")
                ok &= (v is not None and not np.isnan(v) and v >= thr['shape_lo'])
            if thr['shape_hi'] is not None:
                crit_parts.append(f"<={thr['shape_hi']}")
                ok &= (v is not None and not np.isnan(v) and v <= thr['shape_hi'])
            rows.append((f"CaShapeFrac={fmtv(v)}", " & ".join(crit_parts), "PASS" if ok else "FAIL"))

        gr_ct = metrics_row.get('guardrails_count')
        if gr_ct is not None:
            rows.append((f"Guardrails={gr_ct}", f"need ≥{thr['gr_need']}", "PASS" if gr_ct >= thr['gr_need'] else "FAIL"))

        overall = "PASS" if int(metrics_row.get('keep_dff', 0)) == 1 else "FAIL"
        rows.append(("", "OVERALL", overall))
        return rows


   

  

    def plot_roi_pipeline_single(self,
                                 roi_id: int,
                                 fs: float,
                                 windows: List[Tuple[int, int]],
                                 output_path: str,
                                 metrics_row: Optional[dict] = None):
        if not getattr(self, '_checktrace_enabled', False):
            return []
        keep_flag = 0
        if metrics_row is not None:
            keep_flag = int(metrics_row.get('keep_dff', metrics_row.get('keep', 0) or 0))

        # Must load panel config BEFORE building overlay
        panels_cfg = self._checktrace_cfg.get('panels', {})
        layout_cfg = self._checktrace_cfg.get('layout', {})
        windows_cfg = self._checktrace_cfg.get('windows', {})
        show_dff_outlier = bool(panels_cfg.get('show_dff_outlier', True))
        show_metrics_table = bool(panels_cfg.get('show_metrics_table', True))  # new

        # Build overlay AFTER configs
        show_metrics_flag = bool(panels_cfg.get('show_metrics', True))
        qc_class = (metrics_row or {}).get('qc_class', None)
        final_label = (metrics_row or {}).get('final_label', (metrics_row or {}).get('label', '')) if metrics_row else ""
        qc_overlay_line = self._qc_metric_overlay(metrics_row) if (metrics_row and show_metrics_flag) else ""

        if show_metrics_flag and qc_overlay_line and panels_cfg.get('qc_overlay_max_lines', 3) == 0:
            self.logger.debug(f"S2P_DFF: Full QC overlay shown in table+overlay for ROI {roi_id} (no truncation).")

        if self._single_folder:
            roi_dir = os.path.join(self._checktrace_root, "roi_all")
        else:
            suffix = "_keep" if keep_flag == 1 else ""
            roi_dir = os.path.join(self._checktrace_root, f"roi_{roi_id:04d}{suffix}")
        os.makedirs(roi_dir, exist_ok=True)

        panels_cfg = self._checktrace_cfg.get('panels', {})
        layout_cfg = self._checktrace_cfg.get('layout', {})
        windows_cfg = self._checktrace_cfg.get('windows', {})

        # NEW config
        time_mode = windows_cfg.get('time_mode', panels_cfg.get('window_time_mode', 'absolute'))  # absolute|relative
        show_full_overview = bool(panels_cfg.get('show_full_overview', False))
        full_overview_mode = panels_cfg.get('full_overview_mode', 'full')  # full|window_pad
        full_overview_pad_s = float(panels_cfg.get('full_overview_pad_s', 0.0))

        order = layout_cfg.get('stage_order',
                               ["raw", "spks", "Fc", "baseline", "deltaF", "dff", "dff_filtered"])
        show_motion = bool(panels_cfg.get('show_motion', True))
        show_spks = bool(panels_cfg.get('show_spks', False))
        spks_time_shift = int(panels_cfg.get('spks_time_shift_frames', 0))

        per_row_h = float(panels_cfg.get('fig_height_per_row', 1.45))   # was ~1.1
        min_fig_h = float(panels_cfg.get('min_fig_height', 5.0))
        max_fig_h = float(panels_cfg.get('max_fig_height', 30.0))       # safety cap

        if show_spks and 'spks' not in order:
            if 'raw' in order:
                idxr = order.index('raw')
                order = order[:idxr+1] + ['spks'] + order[idxr+1:]
            else:
                order = ['spks'] + order

        # Ensure deltaF present
        if 'deltaF' not in order and panels_cfg.get('show_deltaF', True):
            if 'baseline' in order:
                bi = order.index('baseline')
                order = order[:bi+1] + ['deltaF'] + order[bi+1:]
            else:
                order.append('deltaF')

        if show_dff_outlier and 'dff' in order and 'dff_outlier' not in order:
            di = order.index('dff')
            order = order[:di+1] + ['dff_outlier'] + order[di+1:]

        enabled_map = {
            "raw": panels_cfg.get('show_raw', True),
            "Fc": panels_cfg.get('show_Fc', True),
            "baseline": panels_cfg.get('show_baseline', True),
            "deltaF": panels_cfg.get('show_deltaF', True),
            "dff": panels_cfg.get('show_dff', True),
            "dff_outlier": show_dff_outlier,
            "dff_filtered": panels_cfg.get('show_dff_filtered', True),
            "spks": show_spks,
        }
        stages_plot = [s for s in order if enabled_map.get(s, False)]
        self.logger.debug(f"S2P_DFF: ROI {roi_id} stages_plot={stages_plot}")



        # Fetch captured arrays
        F        = self._stage_capture.get('F')
        Fneu     = self._stage_capture.get('Fneu')
        Fc       = self._stage_capture.get('Fc')
        f0       = self._stage_capture.get('baseline')
        deltaF   = self._stage_capture.get('deltaF')
        dff_denom= self._stage_capture.get('dff_denom')
        dff      = self._stage_capture.get('dff')
        dff_clean = self._stage_capture.get('dff_outlier')
        dff_filt = self._stage_capture.get('dff_filtered')
        spks     = self._stage_capture.get('spks')



        def get_trace(arr):
            if arr is None:
                return None
            if arr.ndim == 2:
                if 0 <= (roi_id - 1) < arr.shape[0]:
                    return arr[roi_id - 1]
                return None
            return arr

        F_r        = get_trace(F)
        Fneu_r     = get_trace(Fneu)
        Fc_r       = get_trace(Fc)
        f0_r       = get_trace(f0)
        deltaF_r   = get_trace(deltaF)
        dff_denom_r= get_trace(dff_denom)
        dff_r      = get_trace(dff)
        dff_clean_r= get_trace(dff_clean)
        dff_f_r    = get_trace(dff_filt)
        spks_r     = get_trace(spks)

        # Auto-drop stages whose data is missing (except we always keep 'dff' if requested)
        stage_data_map = {
            'raw': F_r,
            'spks': spks_r,
            'Fc': Fc_r,
            'baseline': f0_r,
            'deltaF': deltaF_r,
            'dff': dff_r,
            'dff_outlier': dff_clean_r,
            'dff_filtered': dff_f_r
        }
        stages_plot = [s for s in stages_plot if (stage_data_map.get(s) is not None or s == 'dff')]


        if spks_r is not None and spks_time_shift != 0:
            shift = spks_time_shift
            if abs(shift) < spks_r.size:
                if shift > 0:
                    spks_r = np.concatenate([np.zeros(shift, spks_r.dtype), spks_r[:-shift]])
                else:
                    k = -shift
                    spks_r = np.concatenate([spks_r[k:], np.zeros(k, spks_r.dtype)])
            else:
                spks_r = np.zeros_like(spks_r)

        # Per‑ROI length (drop global cache)
        if dff_r is not None:
            T_full = dff_r.size
        elif F_r is not None:
            T_full = F_r.size
        else:
            T_full = 0

        annotate_cfg = self._checktrace_cfg.get('annotate', {})
        show_metrics = annotate_cfg.get('show_metrics', True)
        paths = []

        # Quick function to avoid empty look: add tiny dots if single x-sample after slicing
        def _safety_scatter(ax, tvec, xvec):
            if tvec.size and np.allclose(xvec, xvec[0]):
                ax.plot(tvec[::max(1, tvec.size//20)], xvec[::max(1, tvec.size//20)],
                        '.', color='0.5', ms=2)

        for w_i, (start, end) in enumerate(windows):
            if end > T_full:
                end = T_full
            if end - start < 2:
                continue
            seg_len = end - start

            n_core_rows = len(stages_plot)
            overview_rows = 2 if (show_full_overview and T_full > 0) else 0
            extra_rows = 0
            want_overlay_row = (show_metrics and metrics_row is not None and qc_overlay_line)
            want_table_row = (show_metrics and metrics_row is not None and show_metrics_table)
            if want_overlay_row: extra_rows += 1
            if want_table_row:   extra_rows += 1

            fig_h = (n_core_rows + overview_rows + extra_rows) * per_row_h
            if fig_h < min_fig_h: fig_h = min_fig_h
            if fig_h > max_fig_h: fig_h = max_fig_h
            fig_w = float(panels_cfg.get('fixed_fig_width', 9.5))
            fig, axes = plt.subplots(n_core_rows + overview_rows + extra_rows, 1,
                                     figsize=(fig_w, fig_h),
                                     sharex=False)
            if (n_core_rows + overview_rows + extra_rows) == 1:
                axes = [axes]

            ax_idx = 0

            # ---- Overviews ----
            if overview_rows:
                # Range
                if full_overview_mode == 'window_pad':
                    pad_f = int(round(full_overview_pad_s * fs))
                    o_start = max(0, start - pad_f)
                    o_end = min(T_full, end + pad_f)
                else:
                    o_start, o_end = 0, T_full
                o_len = o_end - o_start
                t_over = np.arange(o_start, o_end) / fs
                max_pts = 6000
                if o_len > max_pts:
                    stride = int(np.ceil(o_len / max_pts))
                    sel_over = slice(0, o_len, stride)
                else:
                    sel_over = slice(None)

                # Overview 1 raw
                ax_full = axes[0]
                if F_r is not None:
                    ax_full.plot(t_over[sel_over], F_r[o_start:o_end][sel_over], color='k', lw=0.4, label='F')
                if Fneu_r is not None:
                    ax_full.plot(t_over[sel_over], Fneu_r[o_start:o_end][sel_over], color='orange', lw=0.4, label='Fneu')
                if f0_r is not None:
                    ax_full.plot(t_over[sel_over], f0_r[o_start:o_end][sel_over], color='crimson', lw=0.5, label='f0')
                ax_full.axvspan(start / fs, end / fs, color='yellow', alpha=0.15, lw=0)
                if show_motion:
                    motion = self._stage_capture.get('motion')
                    if motion is not None and motion.size >= o_end:
                        axm = ax_full.twinx()
                        mot_seg = motion[o_start:o_end]
                        axm.plot(t_over[sel_over], mot_seg[sel_over], color='magenta', lw=0.5, label='motion')
                        axm.set_ylabel('Motion(px)', fontsize=7, color='magenta')
                        axm.tick_params(axis='y', labelsize=7, colors='magenta')
                        h1,l1 = ax_full.get_legend_handles_labels()
                        h2,l2 = axm.get_legend_handles_labels()
                        ax_full.legend(h1+h2, l1+l2, loc='upper right', fontsize=6, frameon=False, ncol=5)
                    else:
                        ax_full.legend(loc='upper right', fontsize=6, frameon=False, ncol=4)
                else:
                    ax_full.legend(loc='upper right', fontsize=6, frameon=False)
                ax_full.set_ylabel('Full\nF', fontsize=8)
                ax_full.set_title(f"ROI {roi_id} full ({'pad' if full_overview_mode!='full' else 'session'})", fontsize=9)
                if panels_cfg.get('show_zero_line', True):
                    ax_full.axhline(0, color='0.85', lw=0.5)
                ax_idx = 1

                # Overview 2 Fc/delta
                ax_full2 = axes[1]
                if Fc_r is not None:
                    Fc_seg = Fc_r[o_start:o_end]
                    ax_full2.plot(t_over[sel_over], Fc_seg[sel_over], color='steelblue', lw=0.4, label='Fc')
                if f0_r is not None:
                    f0_seg = f0_r[o_start:o_end]
                    ax_full2.plot(t_over[sel_over], f0_seg[sel_over], color='crimson', lw=0.4, label='f0')
                if Fc_r is not None and f0_r is not None:
                    delta_full = (Fc_r - f0_r)[o_start:o_end]
                    ax_full2.plot(t_over[sel_over], delta_full[sel_over], color='darkgreen', lw=0.4, label='Fc-f0')
                if show_spks and spks_r is not None:
                    sp_slice = spks_r[o_start:o_end]
                    sp_down = sp_slice[sel_over]
                    thr = float(panels_cfg.get('spks_threshold', 0.0))
                    mask = sp_down > thr
                    if mask.any():
                        t_lines = t_over[sel_over][mask]
                        v_lines = sp_down[mask]
                        scale = np.percentile(v_lines, 99) or 1.0
                        v_plot = v_lines / (scale + 1e-9)
                        ax_s = ax_full2.twinx()
                        ax_s.vlines(t_lines, 0, v_plot, color='black', lw=0.4, label='spks')
                        ax_s.set_ylim(0, 1.05)
                        ax_s.set_ylabel('Spks', fontsize=7)
                        h1,l1 = ax_full2.get_legend_handles_labels()
                        h2,l2 = ax_s.get_legend_handles_labels()
                        ax_full2.legend(h1+h2, l1+l2, loc='upper right', fontsize=6, frameon=False, ncol=5)
                    else:
                        ax_full2.legend(loc='upper right', fontsize=6, frameon=False)
                else:
                    ax_full2.legend(loc='upper right', fontsize=6, frameon=False)
                ax_full2.axvspan(start / fs, end / fs, color='yellow', alpha=0.12, lw=0)
                if panels_cfg.get('show_zero_line', True):
                    ax_full2.axhline(0, color='0.85', lw=0.5)
                ax_full2.set_ylabel('Full\nFc', fontsize=8)
                ax_idx = 2

            # ---- Window stage plots ----
            for stage in stages_plot:
                if ax_idx >= len(axes):
                    break
                ax = axes[ax_idx]
                if time_mode == 'relative':
                    t_stage = np.arange(seg_len) / fs
                else:
                    t_stage = np.arange(start, end) / fs
                sl = slice(start, end)

                if stage == 'raw' and F_r is not None:
                    ax.plot(t_stage, F_r[sl], color='k', lw=0.6, label='F')
                    if Fneu_r is not None:
                        ax.plot(t_stage, Fneu_r[sl], color='orange', lw=0.6, label='Fneu')
                    ax.set_ylabel('F', fontsize=8)
                    ax.legend(fontsize=6, loc='upper right', frameon=False)
                elif stage == 'spks' and spks_r is not None:
                    thr = float(panels_cfg.get('spks_threshold', 0.0))
                    seg_spk = spks_r[sl]
                    mask = seg_spk > thr
                    ts = t_stage[mask]
                    vs = seg_spk[mask]
                    if vs.size:
                        p99 = np.percentile(vs, 99) or 1.0
                        vplt = np.clip(vs / (p99 + 1e-9), 0, 1.0)
                        ax.vlines(ts, 0, vplt, color='black', lw=0.7)
                        ax.set_ylim(0, 1.05)
                    else:
                        ax.text(0.5, 0.5, "(no spikes)", ha='center', va='center', fontsize=8)
                    ax.set_ylabel('Spks', fontsize=8)
                elif stage == 'Fc' and Fc_r is not None:
                    ax.plot(t_stage, Fc_r[sl], color='steelblue', lw=0.6)
                    ax.set_ylabel('Fc', fontsize=8)
                elif stage == 'baseline' and Fc_r is not None and f0_r is not None:
                    ax.plot(t_stage, Fc_r[sl], color='grey', lw=0.5, label='Fc')
                    ax.plot(t_stage, f0_r[sl], color='crimson', lw=0.7, label='f0')
                    ax.set_ylabel('Base', fontsize=8)
                    ax.legend(fontsize=6, loc='upper right', frameon=False)
                elif stage == 'deltaF' and deltaF_r is not None:
                    ax.plot(t_stage, deltaF_r[sl], color='teal', lw=0.6, label='dF')
                    if panels_cfg.get('show_dff_denom', True) and dff_denom_r is not None:
                        ax.plot(t_stage, dff_denom_r[sl], color='goldenrod', lw=0.5, label='denom')
                    ax.set_ylabel('dF', fontsize=8)
                    ax.legend(fontsize=6, loc='upper right', frameon=False)
                elif stage == 'dff_outlier' and dff_clean_r is not None:
                    ax.plot(t_stage, dff_clean_r[sl], color='purple', lw=0.6)
                    ax.set_ylabel('dFF*', fontsize=8)
                elif stage == 'dff' and dff_r is not None:
                    ax.plot(t_stage, dff_r[sl], color='k', lw=0.6)
                    ax.set_ylabel('dFF', fontsize=8)
                elif stage == 'dff_filtered':
                    if dff_f_r is not None and dff_r is not None:
                        ax.plot(t_stage, dff_r[sl], color='0.75', lw=0.5, label='dFF')
                        ax.plot(t_stage, dff_f_r[sl], color='green', lw=0.7, label='filt')
                        ax.set_ylabel('dFFf', fontsize=8)
                        ax.legend(fontsize=6, loc='upper right', frameon=False)
                    else:
                        # (Stage should usually have been auto-dropped; keep silent fallback)
                        ax.text(0.5, 0.5, "no filtered trace", ha='center', va='center', fontsize=7, alpha=0.6)
                elif stage == 'dff_outlier' and dff_clean_r is not None:
                    ax.plot(t_stage, dff_clean_r[sl], color='purple', lw=0.6)
                    ax.set_ylabel('dFF*', fontsize=8)
                    if panels_cfg.get('annotate_dff_outlier', True):
                        ax.text(0.01, 0.92, "* post outlier fill", transform=ax.transAxes,
                                fontsize=6, color='purple', ha='left', va='top',
                                alpha=0.8)
                
                
                # --------------
                # Fallback overlay if no dedicated dFF stage was plotted
                # if qc_overlay_line and 'dff' not in stages_plot and panels_cfg.get('show_metrics', True):
                #     ax_last = axes[-1]
                #     wrap_len = int(panels_cfg.get('qc_overlay_wrap', 110))
                #     import textwrap
                #     wrapped = textwrap.wrap(qc_overlay_line, width=wrap_len)
                #     txt = "\n".join(wrapped[:int(panels_cfg.get('qc_overlay_max_lines', 3))])
                #     ax_last.text(0.01, 0.98, txt, transform=ax_last.transAxes,
                #                 va='top', ha='left', fontsize=6, color='navy',
                #                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.65, lw=0.3))
                    
                # ----------------                    
                if panels_cfg.get('show_zero_line', True):
                    ax.axhline(0, color='0.85', lw=0.4)
                _safety_scatter(ax, t_stage, (deltaF_r[sl] if deltaF_r is not None else np.zeros(seg_len)))
                ax_idx += 1

            # X limits for relative mode
            if time_mode == 'relative':
                for j in range(overview_rows, len(axes)):
                    axes[j].set_xlim(0, seg_len / fs)


            # --- Append QC overlay & table rows at bottom ---
            tail_axes = axes[overview_rows + n_core_rows : overview_rows + n_core_rows + extra_rows]
            tail_idx = 0
            if want_overlay_row:
                ax_ov = tail_axes[tail_idx]
                tail_idx += 1
                ax_ov.axis('off')
                wrap_len = int(panels_cfg.get('qc_overlay_wrap', 110))
                max_lines_cfg = int(panels_cfg.get('qc_overlay_max_lines', 0))
                import textwrap
                wrapped = textwrap.wrap(qc_overlay_line, width=wrap_len) if wrap_len > 0 else [qc_overlay_line]
                if max_lines_cfg > 0:
                    shown = wrapped[:max_lines_cfg]
                    truncated = len(wrapped) > max_lines_cfg
                else:
                    shown = wrapped
                    truncated = False
                txt = "\n".join(shown)
                ax_ov.text(0.0, 1.0, txt,
                           va='top', ha='left', fontsize=7, color='navy',
                           transform=ax_ov.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, lw=0.3))
                if truncated:
                    ax_ov.text(0.99, 0.02,
                               f"+{len(wrapped)-len(shown)} more",
                               va='bottom', ha='right', fontsize=6, color='crimson',
                               transform=ax_ov.transAxes)
            if want_table_row:
                ax_tab = tail_axes[tail_idx]
                tail_idx += 1
                rows_tbl = self._qc_metric_table_rows(metrics_row)
                ax_tab.axis('off')
                if rows_tbl:
                    col_labels = ["Metric (value)", "Criterion", "Result"]
                    cell_text = [list(r) for r in rows_tbl]
                    tbl = ax_tab.table(cellText=cell_text,
                                       colLabels=col_labels,
                                       loc='center',
                                       cellLoc='left',
                                       colLoc='left')
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(6)
                    tbl.scale(1, 1.05)
                    for r_i, row in enumerate(rows_tbl):
                        result = row[2]
                        face = "#d9f7d9" if result == "PASS" else "#f9d0d0" if result == "FAIL" else "white"
                        tbl[(r_i+1, 2)].set_facecolor(face)
                    for c in range(3):
                        tbl[(0, c)].set_facecolor("#e0e0e0")
                        tbl[(0, c)].set_fontsize(6)


            axes[-1].set_xlabel('Time (s)', fontsize=9)
            # Colored title background based on qc_class
            class_color = {
                'BURST': '#e0f7e9',
                'PLATEAU': '#e0e9f7',
                'BURST_ON_PLATEAU': '#e6e0f7',
                'AMBIG': '#fff7e0',
                'FAIL_ARTIFACT': '#fbe4e4'
            }.get(qc_class, '#f2f2f2')
            title_keep = " [KEEP]" if keep_flag else ""
            fig.suptitle(
                f"ROI {roi_id}{title_keep}  label={final_label or 'NA'}  qc_class={qc_class or 'NA'}  "
                f"win {w_i} [{start/fs:.1f}-{end/fs:.1f}s] len={seg_len}f",
                fontsize=10,
                backgroundcolor=class_color
            )

            # Avoid tight_layout failures with many axes
            try:
                fig.tight_layout(rect=[0, 0, 1, 0.96])
            except Exception:
                fig.subplots_adjust(top=0.92, hspace=0.25)

           # ---- QC summary table (below plots) ----
            # if metrics_row is not None and panels_cfg.get('show_metrics', True):
            #     rows = self._qc_metric_table_rows(metrics_row)
            #     if rows:
            #         # Reserve bottom margin
            #         fig.subplots_adjust(bottom=0.12)
            #         ax_tab = fig.add_axes([0.02, 0.005, 0.96, 0.10])
            #         ax_tab.axis('off')
            #         col_labels = ["Metric (value)", "Criterion", "Result"]
            #         cell_text = [list(r) for r in rows]
            #         # Build table
            #         tbl = ax_tab.table(cellText=cell_text,
            #                            colLabels=col_labels,
            #                            loc='center',
            #                            cellLoc='left',
            #                            colLoc='left')
            #         tbl.auto_set_font_size(False)
            #         tbl.set_fontsize(7)
            #         tbl.scale(1, 1.1)
            #         # Color PASS/FAIL column
            #         n_rows = len(rows)
            #         for r_i in range(n_rows):
            #             result = rows[r_i][2]
            #             face = "#d9f7d9" if result == "PASS" else "#f9d0d0" if result == "FAIL" else "white"
            #             tbl[(r_i+1, 2)].set_facecolor(face)
            #         # Header formatting
            #         for c in range(3):
            #             tbl[(0, c)].set_facecolor("#e0e0e0")
            #             tbl[(0, c)].set_fontsize(7)
            # ----------------------------------------

            keep_token = "_keep" if keep_flag else ""
            fname = f"pipeline_roi{roi_id:04d}{keep_token}_w{w_i:02d}.png"
            out_path = os.path.join(roi_dir, fname)
            fig.savefig(out_path, dpi=panels_cfg.get('dpi', 150), bbox_inches='tight')
            plt.close(fig)
            paths.append(out_path)

        return paths



    def _build_ring_traces(self, ops: dict, neuropil_pixels: np.ndarray,
                           agg: str = 'median', trim_frac: float = 0.1,
                           chunk: int = 1000) -> np.ndarray:
        Ly, Lx = int(ops['Ly']), int(ops['Lx'])
        n_rois = neuropil_pixels.size
        n_frames = int(ops.get('nframes', ops.get('nframes_plane', 0)))
        if n_frames <= 0:
            raise ValueError("Ops missing nframes for ring trace build.")
        reg_file = ops.get('reg_file', None)
        if reg_file is None or not os.path.exists(reg_file):
            # fallback to data.bin in same folder
            reg_candidate = os.path.join(os.path.dirname(self._current_output_path), 'data.bin')
            if os.path.exists(reg_candidate):
                reg_file = reg_candidate
        if reg_file is None or not os.path.exists(reg_file):
            raise FileNotFoundError("Registered movie file not found for ring traces.")
        self.logger.info(f"S2P_DFF: Building ring traces from {os.path.basename(reg_file)} frames={n_frames}")

        # Precompute linear index lists
        lin_indices = []
        for i in range(n_rois):
            coords = neuropil_pixels[i]
            if coords is None or coords.size == 0:
                lin_indices.append(np.empty(0, np.int64))
                continue
            y, x = coords
            lin_indices.append((y.astype(np.int64) * Lx + x.astype(np.int64)))

        mm = np.memmap(reg_file, dtype=np.int16, mode='r')
        expected_size = n_frames * Ly * Lx
        if mm.size < expected_size:
            raise ValueError("Reg file smaller than expected for given frames/Ly/Lx.")
        ring_tr = np.zeros((n_rois, n_frames), np.float32)

        for start in range(0, n_frames, chunk):
            end = min(n_frames, start + chunk)
            block_len = end - start
            buf = np.asarray(mm[start*Ly*Lx : end*Ly*Lx], dtype=np.int16).reshape(block_len, Ly*Lx)
            # Convert to float once
            buf = buf.astype(np.float32)
            for r in range(n_rois):
                idx = lin_indices[r]
                if idx.size == 0:
                    continue
                vals = buf[:, idx]  # shape (block_len, n_pix_r)
                if agg == 'trimmed_mean' and idx.size > 4:
                    k = int(idx.size * trim_frac)
                    if k > 0:
                        sorted_vals = np.sort(vals, axis=1)
                        vals_trim = sorted_vals[:, k:idx.size - k]
                        ring_tr[r, start:end] = np.nanmean(vals_trim, axis=1)
                    else:
                        ring_tr[r, start:end] = np.nanmean(vals, axis=1)
                else:
                    ring_tr[r, start:end] = np.nanmedian(vals, axis=1)
        return ring_tr

    def _get_ring_neuropil_traces(self, n_rois: int, T_expected: int, ops: dict) -> Optional[np.ndarray]:
        """
        Standard loader: read neuropil_rings.npz -> neuropil_pixels (object array of coordinate arrays),
        build ring_traces if cache missing or shape mismatch. Returns (n_rois, T) float32.
        """
        try:
            out_path = getattr(self, '_current_output_path', None)
            if out_path is None:
                return None
            ring_file = os.path.join(out_path, 'qc_results', 'neuropil_rings.npz')
            if not os.path.exists(ring_file):
                self.logger.warning("S2P_DFF: neuropil_rings.npz missing.")
                return None
            with np.load(ring_file, allow_pickle=True) as nf:
                if 'neuropil_pixels' not in nf:
                    self.logger.warning("S2P_DFF: neuropil_pixels key missing.")
                    return None
                neuropil_pixels = nf['neuropil_pixels']

            if neuropil_pixels.size != n_rois:
                self.logger.warning("S2P_DFF: neuropil_pixels ROI count mismatch.")
                return None

            cache_path = os.path.join(out_path, 'qc_results', 'ring_traces.npy')
            if os.path.exists(cache_path):
                ring_tr = np.load(cache_path, mmap_mode='r')
                if ring_tr.shape == (n_rois, T_expected):
                    self.logger.info("S2P_DFF: Loaded cached ring_traces.npy")
                    return ring_tr.astype(np.float32)
                else:
                    self.logger.info("S2P_DFF: Cached ring_traces shape mismatch; rebuilding.")

            exp_cfg = self.config_manager.get_experiment_config()
            npil_cfg = exp_cfg.get('imaging_preprocessing', {}) \
                               .get('trace_extraction', {}) \
                               .get('neuropil', {})
            agg = npil_cfg.get('ring_aggregate', 'median')
            trim_frac = float(npil_cfg.get('trimmed_mean_frac', 0.1))
            chunk = int(npil_cfg.get('ring_chunk_frames', 1000))
            do_cache = bool(npil_cfg.get('ring_cache', True))

            ring_traces = self._build_ring_traces(
                ops, neuropil_pixels, agg=agg, trim_frac=trim_frac, chunk=chunk
            )

            if ring_traces.shape[1] != T_expected:
                self.logger.warning("S2P_DFF: Built ring traces length mismatch.")
                return None

            if do_cache:
                np.save(cache_path, ring_traces.astype(np.float32))
                self.logger.info("S2P_DFF: Cached ring_traces.npy")

            return ring_traces.astype(np.float32)

        except Exception as e:
            self.logger.warning(f"S2P_DFF: Ring trace build failed ({e})")
            return None

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
        """
        Extended QC metrics (see spec).
        Config (post_dff_qc.common) optional keys:
          event_band_hz
          event_refractory_s (default 0.3)
          plateau_high_prct (80)
          plateau_deriv_thr (0.25 * robust_std)
          change_point_min_gap_s (0.5)
          template_kernel_rise_s (0.05)
          template_kernel_decay_s (0.4)
          template_kernel_len_s (2.0)
          shape_ratio_min (0.2)
          shape_ratio_max (1.2)
        """
        exp_cfg = self.config_manager.get_experiment_config()
        qc_cfg = (exp_cfg.get('imaging_preprocessing', {})
                           .get('post_dff_qc', {}))
        common = qc_cfg.get('common', {})

        n, T = dff.shape
        # Prepare outputs
        snr = np.full(n, np.nan, np.float32)
        transientness = np.full(n, np.nan, np.float32)
        drift_abs_slope = np.full(n, np.nan, np.float32)
        drift_rel_slope = np.full(n, np.nan, np.float32)
        motion_corr = np.full(n, np.nan, np.float32)
        artifact_frac = np.full(n, np.nan, np.float32)
        event_rate_hz = np.full(n, np.nan, np.float32)
        neuropil_corr = np.full(n, np.nan, np.float32)
        neuropil_ratio = np.full(n, np.nan, np.float32)
        plateau_frac = np.full(n, np.nan, np.float32)
        change_point_z = np.full(n, np.nan, np.float32)
        template_corr_p95 = np.full(n, np.nan, np.float32)
        ca_shape_frac = np.full(n, np.nan, np.float32)
        negativity_frac = np.full(n, np.nan, np.float32)
        min_dff = np.full(n, np.nan, np.float32)

        # Config params
        refractory_s = float(common.get('event_refractory_s', 0.3))
        refractory_frames = max(1, int(round(refractory_s * fs)))
        plateau_prct = float(common.get('plateau_high_prct', 80.0))
        plateau_deriv_rel = float(common.get('plateau_deriv_thr', 0.25))  # multiplier of robust std per frame
        cp_gap_s = float(common.get('change_point_min_gap_s', 0.5))
        cp_gap_frames = max(1, int(round(cp_gap_s * fs)))
        k_rise = float(common.get('template_kernel_rise_s', 0.05))
        k_decay = float(common.get('template_kernel_decay_s', 0.4))
        k_len = float(common.get('template_kernel_len_s', 2.0))
        shape_ratio_min = float(common.get('shape_ratio_min', 0.2))
        shape_ratio_max = float(common.get('shape_ratio_max', 1.2))

        # Template kernel once
        kernel = self._make_template_kernel(fs, k_rise, k_decay, k_len)
        kN = kernel.size
        k_norm = (kernel - kernel.mean()) / (kernel.std() + 1e-9)

        # Pre-normalize motion
        if motion is not None and motion.size == T:
            mv = np.nan_to_num(motion)
            mv = (mv - mv.mean()) / (mv.std() + 1e-9)
        else:
            mv = None

        # Times for drift
        tmin = (np.arange(T) / fs) / 60.0

        # Neuro proxy
        if neuropil_dff is not None and neuropil_dff.shape == dff.shape:
            np_proxy_all = np.nan_to_num(neuropil_dff)
        else:
            np_proxy_all = None

        # Main loop
        for i in range(n):
            x = np.nan_to_num(dff[i])
            min_dff[i] = np.min(x)
            negativity_frac[i] = float(np.mean(x < 0.0))

            # Robust scaling
            med = np.median(x)
            rsd = self._robust_std(x)
            if rsd < 1e-9:
                rsd = 1.0
            z = (x - med) / rsd

            # Event detection (z>3)
            events = self._extract_events(z, 3.0, refractory_frames)
            if events:
                # SNR: median peak amplitude (raw dFF amplitude above baseline) / robust noise
                peaks = [x[p] - med for (_, p) in events]
                if peaks:
                    snr[i] = np.median(peaks) / (rsd + 1e-9)
            else:
                snr[i] = 0.0

            event_rate_hz[i] = len(events) / (T / fs + 1e-9)

            # Transientness
            try:
                bp, tp = self._bandpower_ratio(x, fs, band)
                transientness[i] = bp / (tp + 1e-12)
            except Exception:
                transientness[i] = np.nan

            # Drift (baseline row available)
            if baseline is not None and baseline.shape == dff.shape:
                brow = np.nan_to_num(baseline[i])
                try:
                    A = np.c_[tmin, np.ones_like(tmin)]
                    beta, *_ = np.linalg.lstsq(A, brow, rcond=None)
                    slope = float(beta[0])  # units / min
                except Exception:
                    slope = 0.0
                drift_abs_slope[i] = abs(slope)
                med_b = np.median(brow)
                drift_rel_slope[i] = slope / (abs(med_b) + 1e-9)
            else:
                drift_abs_slope[i] = 0.0
                drift_rel_slope[i] = 0.0

            # Motion corr
            if mv is not None:
                xc = (x - x.mean()) / (x.std() + 1e-9)
                motion_corr[i] = abs(float(np.mean(xc * mv)))

            # Neuropil coupling
            if np_proxy_all is not None:
                np_row = np.nan_to_num(np_proxy_all[i])
                nr = (np_row - np_row.mean()) / (np_row.std() + 1e-9)
                xc = (x - x.mean()) / (x.std() + 1e-9)
                neuropil_corr[i] = abs(float(np.mean(xc * nr)))
                amp_roi = np.median(np.abs(x))
                amp_np = np.median(np.abs(np_row)) + 1e-9
                neuropil_ratio[i] = 1.0 / (amp_roi / amp_np + 1e-9)

            # Artifact fraction (|z|>6)
            artifact_frac[i] = float(np.mean(np.abs(z) > 6.0))

            # Plateau fraction: frames above high percentile & low derivative
            hi_thr = np.percentile(x, plateau_prct)
            dx = np.abs(np.diff(x, prepend=x[0]))
            deriv_thr = plateau_deriv_rel * rsd
            plateau_mask = (x >= hi_thr) & (dx <= deriv_thr)
            plateau_frac[i] = float(np.mean(plateau_mask))

            # Change-point z-score
            change_point_z[i] = self._change_point_zscore(x, cp_gap_frames)

            # Template correlation (sliding normalized window, keep 95th)
            if T >= kN + 2:
                win_corr = []
                k_mu = k_norm.mean(); k_sd = k_norm.std() + 1e-9
                for s in range(0, T - kN, max(1, kN//4)):
                    seg = x[s:s+kN]
                    segn = (seg - seg.mean()) / (seg.std() + 1e-9)
                    win_corr.append(float(np.mean(segn * k_norm)))
                if win_corr:
                    template_corr_p95[i] = np.percentile(win_corr, 95)
                else:
                    template_corr_p95[i] = 0.0
            else:
                template_corr_p95[i] = 0.0

            # Ca-shape fraction: ratio rise/decay within bounds
            shape_hits = 0
            shape_tot = 0
            for (on, pk) in events:
                shape_tot += 1
                peak_val = x[pk]
                onset_val = x[on]
                # rise time
                rise_t = (pk - on) / fs
                # decay: time to 1/e of peak baseline
                target = med + (peak_val - med) / np.e
                # search forward
                decay_t = None
                for j in range(pk+1, min(pk + int(fs * k_len), T)):
                    if x[j] <= target:
                        decay_t = (j - pk) / fs
                        break
                if decay_t is None:
                    decay_t = (min(pk + int(fs * k_len), T) - pk) / fs
                if decay_t > 0:
                    ratio = rise_t / (decay_t + 1e-9)
                    if shape_ratio_min <= ratio <= shape_ratio_max:
                        shape_hits += 1
            ca_shape_frac[i] = (shape_hits / shape_tot) if shape_tot else 0.0

        dff_max_abs = np.nanmax(np.abs(dff), axis=1).astype(np.float32)

        return dict(
            snr=snr,
            transientness=transientness,
            drift_abs_slope=drift_abs_slope,
            drift_rel_slope=drift_rel_slope,
            motion_corr=motion_corr,
            neuropil_corr=neuropil_corr,
            neuropil_ratio=neuropil_ratio,
            artifact_frac=artifact_frac,
            event_rate_hz=event_rate_hz,
            plateau_frac=plateau_frac,
            change_point_z=change_point_z,
            template_corr_p95=template_corr_p95,
            ca_shape_frac=ca_shape_frac,
            negativity_frac=negativity_frac,
            min_dff=min_dff,
            dff_max_abs=dff_max_abs          # <-- NEW
        )


    def _decide_keep(self, metrics: Dict[str, np.ndarray], labels: List[str], qc_cfg: Dict[str, Any]):
        """
        Final keep decision (unified):
          Core (must pass all):
            snr >= snr_min
            template_corr_p95 >= template_corr_p95_min
            artifact_frac <= artifact_frac_max (else artifact_z[_max] percent fallback)
            drift_rel_slope (preferred) or drift_abs_slope <= drift_*_slope_max_per_min
            motion_corr <= motion_corr_max
            transientness >= transientness_min
          Guardrails (need >= guardrails_require_pass of):
            min_dff >= min_dff_min
            negativity_frac >= negativity_frac_min   (fallback to >= -inf if not set)
            ca_shape_frac in [ca_shape_frac_lo, ca_shape_frac_hi]
        """
        import numpy as np
        common = qc_cfg.get('common', {})
        n = len(labels)

        # Thresholds (all optional; if missing, criterion auto-passes)
        dff_abs_max = common.get('dff_abs_max')
        snr_min   = common.get('snr_min')
        tpl_min   = common.get('template_corr_p95_min')
        art_max   = common.get('artifact_frac_max')
        if art_max is None:
            az = common.get('artifact_z_max', common.get('artifact_z'))
            art_max = (az / 100.0) if az is not None else None
        drift_rel_max = common.get('drift_rel_slope_max_per_min')
        drift_abs_max = common.get('drift_abs_slope_max_per_min')
        mot_max   = common.get('motion_corr_max')
        trans_min = common.get('transientness_min')

        min_dff_min = common.get('min_dff_min')
        dff_abs_max = common.get('dff_abs_max')
        neg_min     = common.get('negativity_frac_min')  # NEW (spec: NegFrac >= -0.25)
        # backward compatibility: if only max present previously, ignore it for new rule
        shape_lo    = common.get('ca_shape_frac_lo', common.get('ca_shape_frac_min'))
        shape_hi    = common.get('ca_shape_frac_hi')
        gr_need     = int(common.get('guardrails_require_pass', 2))
        use_rel     = bool(common.get('use_relative_drift', True))

        drift_rel = metrics.get('drift_rel_slope')
        drift_abs = metrics.get('drift_abs_slope')

        keep = np.zeros(n, bool)
        tests_by_roi = []

        def ge(val, thr):
            return True if thr is None else (val >= thr) if (val is not None and not np.isnan(val)) else False
        def le(val, thr):
            return True if thr is None else (val <= thr) if (val is not None and not np.isnan(val)) else False
        def in_range(val, lo, hi):
            if val is None or np.isnan(val): return False
            if lo is not None and val < lo: return False
            if hi is not None and val > hi: return False
            return True

        for i in range(n):
            snr_v = metrics.get('snr', np.full(n, np.nan))[i]
            tpl_v = metrics.get('template_corr_p95', np.full(n, np.nan))[i]
            art_v = metrics.get('artifact_frac', np.full(n, np.nan))[i]
            mot_v = metrics.get('motion_corr', np.full(n, np.nan))[i]
            trans_v = metrics.get('transientness', np.full(n, np.nan))[i]
            mdff_v = metrics.get('min_dff', np.full(n, np.nan))[i]
            neg_key = 'negativity_frac' if 'negativity_frac' in metrics else ('neg_frac' if 'neg_frac' in metrics else None)
            neg_v = metrics.get(neg_key, np.full(n, np.nan))[i] if neg_key else np.nan
            shape_v = metrics.get('ca_shape_frac', np.full(n, np.nan))[i]

            if use_rel and drift_rel is not None:
                drift_v = drift_rel[i]
                drift_ok = le(drift_v, drift_rel_max)
            else:
                drift_v = drift_abs[i] if drift_abs is not None else np.nan
                drift_ok = le(drift_v, drift_abs_max)


            dff_max_abs_v = metrics.get('dff_max_abs', np.full(n, np.nan))[i]
            core_tests = dict(
                snr = ge(snr_v, snr_min),
                template = ge(tpl_v, tpl_min),
                artifact = le(art_v, art_max),
                drift = drift_ok,
                motion = le(mot_v, mot_max),
                transient = ge(trans_v, trans_min),
                dff_abs = le(dff_max_abs_v, dff_abs_max)  # NEW
            )

            gr_flags = []
            if min_dff_min is not None:
                gr_flags.append(ge(mdff_v, min_dff_min))
            if neg_min is not None and not np.isnan(neg_v):
                gr_flags.append(ge(neg_v, neg_min))
            if shape_lo is not None or shape_hi is not None:
                gr_flags.append(in_range(shape_v, shape_lo, shape_hi))

            gr_count = int(sum(gr_flags))
            guardrails_ok = gr_count >= gr_need

            core_ok = all(core_tests.values())
            keep[i] = core_ok and guardrails_ok

            rec = core_tests.copy()
            rec.update(
                pass_core=core_ok,
                guardrails_count=gr_count,
                pass_guardrails=guardrails_ok,
                keep_overall=keep[i]
            )
            tests_by_roi.append(rec)

        return keep, tests_by_roi

    # def _decide_keep(self, metrics: Dict[str, np.ndarray], labels: List[str], qc_cfg: Dict[str, Any]):
    #     n = len(labels)
    #     keep = np.ones(n, bool)
    #     tests_by_roi = []
    #     common = qc_cfg.get('common', {})
    #     per_label = qc_cfg.get('per_label', {})
    #     artifact_z_max = float(common.get('artifact_z_max', 6.0))
    #     rate_max = float(common.get('event_rate_max_hz', 5.0))

    #     use_rel = bool(common.get('use_relative_drift', True))

    #     art_frac_thr = common.get('artifact_frac_max', None)
    #     if art_frac_thr is None:
    #         art_frac_thr = float(common.get('artifact_z_max', common.get('artifact_z', 6.0))) / 100.0

    #     def ge_or_pass(val, thr):
    #         if np.isnan(val):  # treat missing as pass
    #             return True
    #         return val >= thr

    #     def le_or_pass(val, thr):
    #         if np.isnan(val):
    #             return True
    #         return val <= thr

    #     for i, lab in enumerate(labels):
    #         L = lab if lab in per_label else 'uncertain'
    #         rules = per_label.get(L, {})
    #         # drift_val = metrics['drift_rel_slope' if use_rel and 'drift_rel_slope' in metrics else 'drift_abs_slope'][i]
    #         # tests = dict(
    #         #     snr = ge_or_pass(metrics['snr'][i], rules.get('snr_min', -1)),
    #         #     motion = le_or_pass(metrics['motion_corr'][i], rules.get('motion_corr_max', 1e9)),
    #         #     neuropil = le_or_pass(metrics['neuropil_ratio'][i], rules.get('neuropil_ratio_max', 1e9)),
    #         #     drift = le_or_pass(drift_val, rules.get('drift_abs_slope_max_per_min', 1e9)),
    #         #     transient = ge_or_pass(metrics['transientness'][i], rules.get('transientness_min', -1)),
    #         #     artifact = le_or_pass(metrics['artifact_frac'][i], (artifact_z_max / 100.0)),
    #         #     rate = le_or_pass(metrics['event_rate_hz'][i], rate_max)
    #         # )
    #         drift_key = 'drift_rel_slope' if (use_rel and 'drift_rel_slope' in metrics) else 'drift_abs_slope'
    #         thr_key = 'drift_rel_slope_max_per_min' if 'rel' in drift_key else 'drift_abs_slope_max_per_min'
    #         drift_val = metrics[drift_key][i]
    #         drift_thr = rules.get(thr_key, 1e9)
    #         tests = dict(
    #             snr = ge_or_pass(metrics['snr'][i], rules.get('snr_min', -1)),
    #             motion = le_or_pass(metrics['motion_corr'][i], rules.get('motion_corr_max', 1e9)),
    #             neuropil = le_or_pass(metrics['neuropil_ratio'][i], rules.get('neuropil_ratio_max', 1e9)),
    #             drift = le_or_pass(drift_val, drift_thr),
    #             transient = ge_or_pass(metrics['transientness'][i], rules.get('transientness_min', -1)),
    #             artifact = le_or_pass(metrics['artifact_frac'][i], art_frac_thr),
    #             rate = le_or_pass(metrics['event_rate_hz'][i], rate_max)
    #         )            
    #         tests_by_roi.append(tests)
    #         keep[i] = all(tests.values())
    #     return keep, tests_by_roi


    def _run_post_dff_qc(self, output_path: str, ops: Dict, dff: np.ndarray):
        exp_cfg = self.config_manager.get_experiment_config()
        ip_cfg = exp_cfg.get('imaging_preprocessing', {})
        tr_cfg = ip_cfg.get('trace_extraction', {})
        qc_cfg = ip_cfg.get('post_dff_qc', {})
        if not qc_cfg.get('enabled', False):
            self.logger.info("S2P_DFF: post_dff_qc disabled.")
            return None

        # fs = float(ops.get('fs', ops.get('fs_hz', 30.0)))
        fs = float(tr_cfg.get('fs_hz', 29.760441593958042))
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
        # if baseline is None:
        #     self.logger.info("S2P_DFF: QC baseline missing; recomputing quick fallback (15s, q=0.2).")
        #     baseline = self._rolling_quantile_2d(dff, fs, win_s=15, min_win_s=8, q=0.2)

        # Motion trace (optional)
        motion = None

        # 1) Use already captured motion trace if available
        if hasattr(self, '_motion_trace'):
            mt = getattr(self, '_motion_trace')
            if isinstance(mt, np.ndarray):
                motion = mt

        # 2) Try dff.h5 stored motion
        if motion is None:
            try:
                with h5py.File(dff_file, 'r') as f:
                    if 'motion' in f:
                        motion = f['motion'][()]
            except Exception:
                pass

        # 3) Try qc_results file
        if motion is None:
            mname = qc_cfg.get('motion_trace', 'rigid_corr')
            mpath = os.path.join(output_path, 'qc_results', f'{mname}.npy')
            if os.path.exists(mpath):
                try:
                    m = np.load(mpath)
                    if m.ndim > 1:
                        m = m[0]
                    motion = m
                except Exception:
                    motion = None

        # 4) Derive from ops xoff/yoff if still None
        if motion is None:
            try:
                xoff = ops.get('xoff')
                yoff = ops.get('yoff')
                if isinstance(xoff, (list, np.ndarray)) and isinstance(yoff, (list, np.ndarray)):
                    xoff = np.asarray(xoff)
                    yoff = np.asarray(yoff)
                    if xoff.ndim == 1 and xoff.size == dff.shape[1] and yoff.shape == xoff.shape:
                        motion = np.sqrt(xoff.astype(np.float32)**2 + yoff.astype(np.float32)**2)
                        self.logger.info("S2P_DFF: Derived motion from ops xoff/yoff for QC.")
            except Exception:
                motion = None

        if motion is not None and motion.size != dff.shape[1]:
            self.logger.warning("S2P_DFF: Motion trace length mismatch; ignoring motion for QC.")
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
       
        # ================= Functional phenotype classification (updated normalized scoring) =================
        m = metrics
        N = dff.shape[0]



        common_cfg = qc_cfg.get('common', {})
        phen_cfg   = qc_cfg.get('phenotype', {})  # optional new subsection

        # --- Hard fail thresholds (percent → fraction for artifact) ---
        cp_thr_cfg  = float(common_cfg.get('change_point_z_max', 1e9))
        mot_thr_cfg = float(qc_cfg.get('per_label', {}).get('uncertain', {}).get(
            'motion_corr_max', common_cfg.get('motion_corr_max', 1e9)))
        # art_thr_cfg = float(common_cfg.get('artifact_z_max', common_cfg.get('artifact_z', 6.0))) / 100.0
        art_frac_thr_cfg = common_cfg.get('artifact_frac_max', None)
        if art_frac_thr_cfg is None:
            art_frac_thr_cfg = float(common_cfg.get('artifact_z_max', common_cfg.get('artifact_z', 6.0))) / 100.0        

        dff_abs_max_cfg = common_cfg.get('dff_abs_max', None)
        dff_max_abs_vec = m.get('dff_max_abs')

        cp   = np.nan_to_num(m.get('change_point_z', np.zeros(N, np.float32)))
        mot  = np.nan_to_num(m.get('motion_corr', np.zeros(N, np.float32)))
        art  = np.nan_to_num(m.get('artifact_frac', np.zeros(N, np.float32)))
        trans_raw = np.nan_to_num(m.get('transientness', np.zeros(N, np.float32)))
        trans_clip = np.clip(trans_raw, 0, 1)

        hard_fail = np.zeros(N, bool)
        # hard_fail |= (art > art_thr_cfg)
        hard_fail |= (art > art_frac_thr_cfg)
        hard_fail |= (mot > mot_thr_cfg)
        # Only hard-fail big change-points if low transientness (likely artifact / bleach edge)
        hard_fail |= ((cp > cp_thr_cfg) & (trans_clip < 0.15))
        # Add amplitude auto-fail (if threshold configured and metric present)
        if dff_abs_max_cfg is not None and dff_max_abs_vec is not None:
            hard_fail |= (dff_max_abs_vec > float(dff_abs_max_cfg))

        amp_fail_mask = (dff_abs_max_cfg is not None and dff_max_abs_vec is not None and
                         (dff_max_abs_vec > float(dff_abs_max_cfg)))

        # --- Robust quantile normalization helper ---
        def _qnorm(x, lo=0.10, hi=0.90):
            x = np.nan_to_num(x.astype(np.float32))
            if x.size == 0:
                return x
            ql, qh = np.quantile(x, [lo, hi])
            if not np.isfinite(ql): ql = 0.0
            if not np.isfinite(qh): qh = ql + 1.0
            if qh <= ql:
                return np.zeros_like(x)
            return np.clip((x - ql) / (qh - ql + 1e-9), 0, 1)

        # --- Feature normalizations ---
        SNRn    = _qnorm(m.get('snr', np.zeros(N)))
        TRANSn  = trans_clip
        ERn     = _qnorm(m.get('event_rate_hz', np.zeros(N)), 0.10, 0.95)
        TPLn    = np.clip(np.nan_to_num(m.get('plateau_frac', np.zeros(N))), 0, 1)
        CPn     = 1.0 / (1.0 + np.exp(-(cp - 2.0) / 1.0))  # smooth sigmoid
        CPn     = np.clip(CPn, 0, 1)
        TEMPLn  = _qnorm(m.get('template_corr_p95', np.zeros(N)), 0.10, 0.95)
        SHAPEn  = _qnorm(m.get('ca_shape_frac',     np.zeros(N)), 0.10, 0.95)

        NEGsrc  = m.get('negativity_frac', m.get('neg_frac', np.zeros(N)))
        NEGp    = _qnorm(np.nan_to_num(NEGsrc))
        MOTp    = _qnorm(np.abs(mot))
        ARTp    = _qnorm(art)
        MINp    = _qnorm(-np.nan_to_num(m.get('min_dff', np.zeros(N))), 0.50, 0.98)

        # --- Weights (override via phenotype.weights) ---
        default_w = {
            'snr':1.3, 'trans':1.2, 'templ':1.0, 'er':0.8, 'shape':0.6,
            'tpl':1.3, 'cp':1.0,
            'mot':0.7, 'art':0.7, 'neg':0.5, 'min':0.3
        }
        w = default_w.copy()
        w.update(phen_cfg.get('weights', {}))

        # --- Scores with penalties ---
        burst_score = (
            w['snr']*SNRn + w['trans']*TRANSn + w['templ']*TEMPLn +
            w['er']*ERn  + w['shape']*SHAPEn
            - (w['mot']*MOTp + w['art']*ARTp + w['neg']*NEGp + w['min']*MINp)
        )
        plateau_score = (
            w['tpl']*TPLn + w['cp']*CPn + 0.4*SNRn
            - (0.5*TRANSn + w['mot']*MOTp + w['art']*ARTp)
        )

        # --- Thresholds & overlap margin (configurable) ---
        thr_burst = float(phen_cfg.get('thr_burst', 0.55))
        thr_plat  = float(phen_cfg.get('thr_plateau', 0.50))
        margin    = float(phen_cfg.get('overlap_margin', 0.15))

        cand_burst = burst_score > thr_burst
        cand_plat  = plateau_score > thr_plat
        both_mask  = cand_burst & cand_plat & (np.abs(burst_score - plateau_score) < margin)

        qc_class = np.full(N, 'AMBIG', object)
        qc_class[cand_burst & ~cand_plat] = 'BURST'
        qc_class[cand_plat  & ~cand_burst] = 'PLATEAU'
        qc_class[both_mask] = 'BURST_ON_PLATEAU'
        qc_class[amp_fail_mask] = 'FAIL_AMPLITUDE'

        # Confidence = margin between top and second of (burst, plateau, 0 baseline)
        tri = np.vstack([burst_score, plateau_score, np.zeros(N)]).T
        best = tri.max(axis=1)
        second = np.sort(tri, axis=1)[:, -2]
        qc_confidence = np.clip(best - second, 0, None)

        # Apply hard-fails last
        qc_class[hard_fail] = 'FAIL_ARTIFACT'
       
        keep, tests = self._decide_keep(metrics, labels, qc_cfg)

        # # Debug summary of failure rates
        # fail_counts = {k: 0 for k in ['snr','motion','neuropil','drift','transient','artifact','rate']}
        # for t in tests:
        #     for k, v in t.items():
        #         if not v:
        #             fail_counts[k] += 1
        # total = dff.shape[0]
        # self.logger.info("S2P_DFF: QC fail counts " +
        #                  ", ".join([f"{k}={fail_counts[k]}/{total}" for k in fail_counts]))
        # if motion is None:
        #     self.logger.info("S2P_DFF: QC ran with NO motion trace; motion test auto-passed (NaN-safe).")

        # Dynamic QC fail count summary (avoids KeyError for new tests like 'template', 'dff_abs')
        import numpy as _np
        fail_counts = {}
        for t in tests:
            for k, v in t.items():
                if k in ('pass_core', 'pass_guardrails', 'guardrails_count', 'keep_overall'):
                    continue
                if isinstance(v, (bool, _np.bool_)) and not v:
                    fail_counts[k] = fail_counts.get(k, 0) + 1
        total = dff.shape[0]
        if fail_counts:
            self.logger.info(
                "S2P_DFF: QC fail counts " +
                ", ".join(f"{k}={fail_counts[k]}/{total}" for k in sorted(fail_counts))
            )
        else:
            self.logger.info("S2P_DFF: QC fail counts (none)")

        # Write CSV summary
        class_map = {
            'FAIL_ARTIFACT': 0,
            'FAIL_AMPLITUDE': -1,   # or another positive code if you prefer
            'BURST': 1,
            'PLATEAU': 2,
            'BURST_ON_PLATEAU': 3,
            'AMBIG': 4
        }
        qc_class_code = np.array([class_map.get(c, -1) for c in qc_class], np.int16)
        
        # not sure, check
        keep[qc_class == 'FAIL_ARTIFACT'] = False
        if phen_cfg.get('require_positive_class', False):
            pos_mask = np.isin(qc_class, ['BURST', 'PLATEAU', 'BURST_ON_PLATEAU'])
            keep &= pos_mask
        
        rows = []
        for i in range(dff.shape[0]):
            r = dict(
                roi_id=i+1,
                final_label=labels[i],
                keep_dff=int(keep[i]),
                qc_class=qc_class[i],
                qc_class_code=int(qc_class_code[i]),
                burst_score=float(burst_score[i]),
                plateau_score=float(plateau_score[i]),
                qc_confidence=float(qc_confidence[i])
            )
            # Scalar per-metric values
            for k, v in metrics.items():
                r[k] = float(v[i])

            # Unified QC aggregate flags (explicit)
            r['pass_core'] = int(tests[i].get('pass_core', 0))
            r['pass_guardrails'] = int(tests[i].get('pass_guardrails', 0))
            r['guardrails_count'] = int(tests[i].get('guardrails_count', -1))

            # Add individual core metric pass/fail flags (snr, template, artifact, drift, motion, transient)
            for k, passed in tests[i].items():
                if k in ('pass_core', 'pass_guardrails', 'guardrails_count', 'keep_overall'):
                    continue
                # k are the base test names (snr, template, artifact, drift, motion, transient)
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
                if motion is not None:
                    grp.create_dataset('motion_trace_used', data=motion, compression='gzip')
                grp.create_dataset('qc_class_code', data=qc_class_code, compression='gzip')
                grp.create_dataset('burst_score', data=burst_score.astype(np.float32), compression='gzip')
                grp.create_dataset('plateau_score', data=plateau_score.astype(np.float32), compression='gzip')
                grp.create_dataset('qc_confidence', data=qc_confidence.astype(np.float32), compression='gzip')
                # one-hot
                one_hot = np.zeros((dff.shape[0], 5), np.uint8)
                for idx, c in enumerate(qc_class):
                    if c == 'FAIL_ARTIFACT':        one_hot[idx, 0] = 1
                    elif c == 'BURST':              one_hot[idx, 1] = 1
                    elif c == 'PLATEAU':            one_hot[idx, 2] = 1
                    elif c == 'BURST_ON_PLATEAU':   one_hot[idx, 3] = 1
                    elif c == 'AMBIG':              one_hot[idx, 4] = 1
                grp.create_dataset('qc_class_onehot', data=one_hot, compression='gzip')                    
        except Exception as e:
            self.logger.warning(f"S2P_DFF: Could not append QC metrics to dff.h5 ({e})")

        kept_frac = keep.mean()
        unique, counts = np.unique(qc_class, return_counts=True)
        self.logger.info("S2P_DFF: qc_class counts: " + ", ".join(f"{u}:{c}" for u, c in zip(unique, counts)))
        self.logger.info("S2P_DFF: median burst_score=%.3f plateau_score=%.3f qc_conf=%.3f" %
                         (np.nanmedian(burst_score), np.nanmedian(plateau_score), np.nanmedian(qc_confidence)))
        self.logger.info(f"S2P_DFF: keep_dff fraction={kept_frac:.2%}")
        return dict(keep=keep, metrics=metrics)



    def _neuropil_regress_fast(self, F: np.ndarray, Fneu: np.ndarray,
                               alpha_init: float = 0.7,
                               bounds: Tuple[float, float] = (0.3, 1.2),
                               fit_cfg: Optional[Dict[str, Any]] = None,
                               use_intercept: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized per‑ROI neuropil slope regression:
            F ≈ a * Fneu (+ b if use_intercept True)
            Fc = F - a * Fneu   (NOTE: no intercept subtraction in default pipeline)

        Rationale (slope only):
          - We perform a robust rolling / percentile baseline later; any constant (additive) offset
            between F and a*Fneu is handled by the baseline stage.
          - Including an intercept can double-subtract low-frequency / DC components and bias
            the percentile baseline downward, producing artificial negative dF/F.
          - Slope-only keeps neuropil contamination scaling while leaving offset correction
            to the dedicated baseline step (clean separation of concerns).

        If use_intercept=True is explicitly requested (e.g., for diagnostics), b is estimated
        but by default we return b=0 so downstream baseline logic remains unchanged.
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

        use_idx = np.arange(T)
        if subsample_k > 1:
            use_idx = use_idx[::subsample_k]

        if 0.0 < qa < 0.9:
            resid = F - alpha_init * N
            thresh = np.quantile(resid, qa, axis=1)[:, None]
            low_mask = resid <= thresh
            stride_mask = np.zeros((T,), bool)
            stride_mask[use_idx] = True
            mask = low_mask & stride_mask[None, :]
            valid_counts = mask.sum(axis=1)
            too_few = valid_counts < min_frames
            if np.any(too_few):
                mask[too_few, :] = stride_mask
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

        if use_intercept:
            # Optional intercept (diagnostics only)
            b = (sumF - a * sumN) / (T_eff + eps)
        else:
            b = np.zeros_like(a)

        Fc = F - a[:, None] * N  # slope-only correction (no -b)

        bad = varN < 1e-6
        if np.any(bad):
            a[bad] = alpha_init
            b[bad] = 0.0
            Fc[bad] = F[bad] - a[bad, None] * N[bad]

        self.logger.info(
            f"S2P_DFF: Neuropil regress fast (slope only, subsample_k={subsample_k}, "
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
            # if mode == 'causal':
            #     out[i] = causal
            # else:
            #     shift = win // 2
            #     # shift back (centered)
            #     out[i, :-shift] = causal[shift:]
            #     out[i, -shift:] = causal[-1]
            if mode == 'causal':
                out[i] = causal
            else:
                shift = win // 2
                if shift >= T:
                    out[i] = causal
                else:
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

    def compute_dff(self, F: np.ndarray, Fneu: np.ndarray, ops: Dict) -> np.ndarray:
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

            # ---- NEW: derive & capture motion trace from ops ----
            motion = None
            try:
                self.logger.info("S2P_DFF: Attempting to derive motion trace from ops...")
                xoff = ops.get('xoff')
                yoff = ops.get('yoff')
                if isinstance(xoff, (list, np.ndarray)) and isinstance(yoff, (list, np.ndarray)):
                    xoff = np.asarray(xoff)
                    yoff = np.asarray(yoff)
                    if xoff.ndim == 1 and yoff.ndim == 1 and xoff.size == F.shape[1]:
                        motion = np.sqrt(xoff.astype(np.float32)**2 + yoff.astype(np.float32)**2)
                if motion is None:
                    self.logger.warning("S2P_DFF: Motion trace not found in ops; attempting fallback.")
                    # fallback to non‑rigid blocks
                    xoff1 = ops.get('xoff1')
                    yoff1 = ops.get('yoff1')
                    if isinstance(xoff1, np.ndarray) and isinstance(yoff1, np.ndarray) and xoff1.ndim == 2 and yoff1.shape == xoff1.shape:
                        if xoff1.shape[1] == F.shape[1]:
                            motion = np.mean(np.sqrt(xoff1.astype(np.float32)**2 + yoff1.astype(np.float32)**2), axis=0)
                if motion is not None and motion.size == F.shape[1]:
                    self.logger.info(f"S2P_DFF: Derived motion trace (len={motion.size})")
                    self._motion_trace = motion.astype(np.float32)
                    self._capture_stage('motion', self._motion_trace, persist='copy')
                    self.logger.info(f"S2P_DFF: Captured motion trace (len={motion.size})")
                else:
                    self.logger.info("S2P_DFF: Motion trace unavailable or length mismatch; skipping capture.")
            except Exception as me:
                self.logger.warning(f"S2P_DFF: Motion derivation failed ({me})")
            # ------------------------------------------------------


            # Capture raw
            self._capture_stage('F', F, persist='copy')
            self._capture_stage('Fneu', Fneu, persist='copy')

            # Optionally load ring neuropil traces for plotting even if not used
            want_ring_plot = tr_cfg.get('outputs', {}).get('checktrace', {}).get('panels', {}) \
                .get('show_neuropil_ring', False)
            ring_tr = None
            if want_ring_plot:
                ring_tr = self._get_ring_neuropil_traces(F.shape[0], F.shape[1], ops=ops)
                if ring_tr is not None:
                    self._capture_stage('Fneu_ring', ring_tr, persist='copy')

            # Neuropil method selection
            npil_method = npil_cfg.get('method', 'regress')
            npil_used = npil_method  # track which actually applied
            
            if npil_method == 'regress_ring':
                self.logger.info("S2P_DFF: Neuropil ring regression enabled.")
                # Prefer previously loaded ring_tr if present
                if ring_tr is None:
                    ring_tr = self._get_ring_neuropil_traces(F.shape[0], F.shape[1], ops=ops)
                    if ring_tr is not None:
                        self._capture_stage('Fneu_ring', ring_tr, persist='copy')
                if ring_tr is not None:
                    alpha_init = float(npil_cfg.get('alpha_init', 0.7))
                    a_lo, a_hi = npil_cfg.get('alpha_bounds', [0.3, 1.2])
                    fit_cfg = npil_cfg.get('fit', {})
                    Fc, a_hat, b_hat = self._neuropil_regress_fast(
                        F, ring_tr, alpha_init=alpha_init,
                        bounds=(a_lo, a_hi), fit_cfg=fit_cfg,
                        use_intercept=False  # slope only: baseline handles offsets
                    )
                    self._last_np_a = a_hat
                    self._last_np_b = b_hat
                    self.logger.info(f"S2P_DFF: Neuropil ring regression applied (alpha_init={alpha_init}, bounds=({a_lo},{a_hi}))")
                else:
                    self.logger.warning("S2P_DFF: Ring traces unavailable; falling back to standard Fneu regression.")
                    npil_method = 'regress'
                    npil_used = 'regress'

            if npil_method == 'regress':
                if npil_used != 'regress':
                    npil_used = 'regress'
                self.logger.info("S2P_DFF: Neuropil regression enabled.")
                alpha_init = float(npil_cfg.get('alpha_init', 0.7))
                a_lo, a_hi = npil_cfg.get('alpha_bounds', [0.3, 1.2])
                fit_cfg = npil_cfg.get('fit', {})
                Fc, a_hat, b_hat = self._neuropil_regress_fast(
                    F, Fneu, alpha_init=alpha_init,
                    bounds=(a_lo, a_hi), fit_cfg=fit_cfg,
                    use_intercept=False  # slope only: baseline handles offsets
                )
                self._last_np_a = a_hat
                self._last_np_b = b_hat
                self.logger.info(f"S2P_DFF: Neuropil regression applied (alpha_init={alpha_init}, bounds=({a_lo},{a_hi}))")
            elif npil_method == 'scalar':
                npil_used = 'scalar'
                self.logger.info("S2P_DFF: Neuropil scalar subtraction enabled.")
                alpha = float(npil_cfg.get('alpha_init', 0.7))
                Fc = F - alpha * Fneu
                self._last_np_a = np.full(F.shape[0], alpha, float)
                self._last_np_b = np.zeros(F.shape[0], float)
                self.logger.info(f"S2P_DFF: Neuropil scalar subtraction applied (alpha={alpha})")
            elif npil_method == 'none':
                npil_used = 'none'
                Fc = F
                self._last_np_a = np.zeros(F.shape[0])
                self._last_np_b = np.zeros(F.shape[0])
                self.logger.warning("S2P_DFF: Not neuropil corrected. Fc = F")

            # Record which neuropil source actually used (for plotting annotation)
            self._npil_used_method = npil_used
            
            
            # After Fc computed:
            self._capture_stage('Fc', Fc, persist='copy')
            
            # -------- Positive shift guard BEFORE baseline --------
            # Per-ROI constant c_i = max(0, -P1%(Fc_i)) so shifted Fc is >= ~P1%(Fc)>=0
            pct1 = np.percentile(Fc, 1, axis=1)          # shape (n_rois,)
            c_shift = np.maximum(0.0, -pct1).astype(np.float32)
            # X = Fc + c_shift[:, None]                    # shifted signal used for baseline & dF/F
            X = Fc
            self._last_shift_c = c_shift
            self._capture_stage('Fc_shifted', X, persist='copy')
            # -------------------------------------------------------

            # Baseline (computed on shifted X, NOT raw Fc)
            b_method = base_cfg.get('method', 'rolling_percentile_fast')
            if b_method in ('rolling_percentile_fast', 'rolling_percentile'):
                f0 = self._baseline_percentile_fast(X, fs_hz, base_cfg)
            else:
                sig_baseline = float(base_cfg.get('sig_baseline', 600))
                f0 = gaussian_filter(X, [0., sig_baseline])
                self.logger.info(f"S2P_DFF: Gaussian baseline sigma={sig_baseline}")

            # -------- Baseline floor guard --------
            floor_eps = float(base_cfg.get('floor_eps', 1e-3))
            # Optional: if very small, scale by median raw F (comment out if not desired)
            floor_eps = max(floor_eps, 1e-3 * np.median(F))
            np.maximum(f0, floor_eps, out=f0)   # in-place floor
            # -------------------------------------------------------

            self._last_baseline = f0.astype(np.float32)
            self._capture_stage('baseline', f0, persist='copy')

            # Numerator (shifted Fc minus baseline)
            deltaF = (X - f0).astype(np.float32)
            self._capture_stage('deltaF', deltaF, persist='copy')
            self.logger.info(f"S2P_DFF: Captured deltaF numerator shape={deltaF.shape}")

            # dF/F denominator (baseline + epsilon) for diagnostics
            eps = float(base_cfg.get('floor_eps', 1e-3))  # same as floor_eps used below
            self._last_floor_eps = eps
            dff_denom = (f0 + eps).astype(np.float32)
            self._capture_stage('dff_denom', dff_denom, persist='copy')

            # dF/F using SAME shifted signal X
            dff = deltaF / dff_denom
            self._capture_stage('dff', dff, persist='copy')         
                        
            n_timepoints = F.shape[1]
            full_time_vector = np.arange(n_timepoints) / fs_hz
            t_center = 3
            t_start = t_center - 3
            t_stop = t_center + 5
            start_idx = np.abs(full_time_vector - t_start).argmin()
            stop_idx = np.abs(full_time_vector - t_stop).argmin()
            
            fig, ax = plt.subplots(figsize=(10, 4))
            roi = 172
            t = full_time_vector[start_idx:stop_idx]
            df = deltaF[roi][start_idx:stop_idx]
            denom = dff_denom[roi][start_idx:stop_idx]
            dff_ = df/denom
            dff_curr = dff[roi][start_idx:stop_idx]
            ax.plot(t, df)
            ax.plot(t, denom)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t, dff_)
            ax.plot(t, dff_curr)
            # # Baseline
            # b_method = base_cfg.get('method', 'rolling_percentile_fast')
            # if b_method in ('rolling_percentile_fast', 'rolling_percentile'):
            #     f0 = self._baseline_percentile_fast(Fc, fs_hz, base_cfg)
            # else:
            #     sig_baseline = float(base_cfg.get('sig_baseline', 600))
            #     f0 = gaussian_filter(Fc, [0., sig_baseline])
            #     self.logger.info(f"S2P_DFF: Gaussian baseline sigma={sig_baseline}")
            # self._last_baseline = f0.astype(np.float32)
            # # After baseline f0 computed:
            # self._capture_stage('baseline', f0, persist='ref')

            # # dFF
            # dff = (Fc - f0) / (f0 + 1e-10)
            # self._capture_stage('dff', dff, persist='ref')

            # Outliers
            outlier_enable = bool(ol_cfg.get('enable', False))
            if outlier_enable:
                self.logger.info("S2P_DFF: Detecting and replacing outliers in dFF traces...")
                outlier_method = ol_cfg.get('method', 'zscore')
                outlier_threshold = float(ol_cfg.get('threshold', 5.0))
                total_out = 0
                total_pts = dff.size
                if outlier_threshold > 0:
                    T = dff.shape[1]
                    for i in range(dff.shape[0]):
                        x = dff[i]
                        mask = self._detect_outliers(x, method=outlier_method, threshold=outlier_threshold)
                        if not mask.any():
                            continue
                        total_out += int(mask.sum())
                        # Find contiguous outlier segments
                        idx = np.where(mask)[0]
                        # segment boundaries
                        cuts = np.where(np.diff(idx) > 1)[0] + 1
                        segments = np.split(idx, cuts)
                        for seg in segments:
                            if seg.size == 0:
                                continue
                            s = seg[0]
                            e = seg[-1]
                            left_val = None
                            right_val = None
                            if s - 1 >= 0 and not mask[s - 1]:
                                left_val = x[s - 1]
                            if e + 1 < T and not mask[e + 1]:
                                right_val = x[e + 1]
                            if left_val is not None and right_val is not None:
                                repl = 0.5 * (left_val + right_val)
                            elif left_val is not None:
                                repl = left_val
                            elif right_val is not None:
                                repl = right_val
                            else:
                                # Entire trace (or segment bounded by NaNs/all outliers)
                                repl = 0.0
                            x[s:e + 1] = repl
                        dff[i] = x
                if total_pts:
                    self.logger.info(
                        f"S2P_DFF: Outliers replaced {total_out/total_pts*100:.2f}% "
                        f"({outlier_method}, thr={outlier_threshold}, method=neighbor-mean)"
                    )
                # capture cleaned version
                self._capture_stage('dff_outlier', dff, persist='copy')
            else:
                self.logger.info("S2P_DFF: Outlier detection disabled (config enable=false)")
                self._capture_stage('dff_outlier', dff, persist='copy')


            # Optional post filter (light touch)
            enable_pf = bool(filt_cfg.get('enable', False))
            if enable_pf:
                self.logger.info("S2P_DFF: Applying post-filtering to dFF traces...")
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
                # Capture post-filter
                self._capture_stage('dff_filtered', dff, persist='copy')
            else:
                self.logger.info("S2P_DFF: Post-filtering disabled (config enable=false)")


            if tr_cfg.get('normalize', False):
                self.logger.info("S2P_DFF: Normalizing dFF traces (z-score)...")
                mu = np.nanmean(dff, axis=1, keepdims=True)
                sd = np.nanstd(dff, axis=1, keepdims=True) + 1e-9
                dff = (dff - mu) / sd
                self.logger.info("S2P_DFF: z-normalized (config normalize=true)")
                # Capture post-normalize
                self._capture_stage('dff_normalized', dff, persist='copy')
            else:
                self.logger.info("S2P_DFF: Normalization disabled (config normalize=false)")


            self._last_Fc = Fc.astype(np.float32)
            self.logger.info(f"S2P_DFF: dFF shape={dff.shape} fs={fs_hz:.2f}Hz")
            return dff.astype(np.float32)

        except Exception as e:
            self.logger.error(f"S2P_DFF: Failed compute_dff ({e})")
            return F.copy()


    
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
                if hasattr(self, '_last_shift_c'):
                    f.create_dataset('shift_c', data=self._last_shift_c.astype(np.float32))                    
                if hasattr(self, '_last_baseline'):
                    f.create_dataset('baseline', data=self._last_baseline, compression='gzip')
                if hasattr(self, '_last_np_a'):
                    f.create_dataset('neuropil_a', data=self._last_np_a.astype(np.float32))
                if hasattr(self, '_last_np_b'):
                    f.create_dataset('neuropil_b', data=self._last_np_b.astype(np.float32))
                if hasattr(self, '_motion_trace'):
                    f.create_dataset('motion', data=self._motion_trace.astype(np.float32))                    
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
                                  fs=29.760441593958042, colormap='viridis',
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

            # TODO fix later - need consistent path access rather than params and properties
            self._current_output_path = output_path

            self._init_checktrace(output_path)

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
                
                # normalize = trace_config.get('normalize', True)
                # baseline_sigma = trace_config.get('baseline_sigma', 600)
                # outlier_threshold = trace_config.get('outlier_threshold', 5.0)
                # outlier_method = trace_config.get('outlier_method', 'zscore')
                
                # self.logger.info(f"S2P_DFF: Using normalize={normalize}, baseline_sigma={baseline_sigma}")
                # self.logger.info(f"S2P_DFF: Outlier detection - method={outlier_method}, threshold={outlier_threshold}")
                
                # # todo add separate spike loading later
                # spks = data['spks']
            
            if not have_spks or force:
                # Compute spks with oasis deconvolution
                spks = self.compute_spks(
                    data['F'], 
                    data['Fneu'], 
                    data['ops'],
                )
                
                
                # --- ALIGN SPIKES LENGTH WITH F ---
                try:
                    if spks is not None:
                        # Expect shape (n_rois, T)
                        T_F = data['F'].shape[1]
                        if spks.shape[1] != T_F:
                            self.logger.warning(f"S2P_DFF: spks length {spks.shape[1]} != F length {T_F}; reconciling.")
                            T_min = min(T_F, spks.shape[1])
                            spks = spks[:, :T_min]
                            # Also truncate F/Fneu/etc. if longer (rare)
                            if data['F'].shape[1] != T_min:
                                data['F'] = data['F'][:, :T_min]
                            if data['Fneu'].shape[1] != T_min:
                                data['Fneu'] = data['Fneu'][:, :T_min]
                        else:
                            self.logger.info(f"S2P_DFF: spks length {spks.shape[1]} matches F length {T_F}, no reconciliation needed.")
                except Exception as _e:
                    self.logger.warning(f"S2P_DFF: Spike/F length reconcile failed ({_e})")                
                
                
                # NEW: capture spikes for plotting
                try:
                    self._capture_stage('spks', spks, persist='copy')
                except Exception as _e:
                    self.logger.warning(f"S2P_DFF: Could not capture spks for plotting ({_e})")                
                

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
                    data['ops']
                )

                # return
                # Store original DFF for comparison
                dff_original = dff.copy()
                
                # Apply Savitzky-Golay filter for additional smoothing
                # dff_filtered = self.apply_savgol_filter(dff, window_length=9, polyorder=3)
                dff_filtered = dff  # filtering handled inside compute_dff via post_filter config
                
                # # todo add separate spike loading later
                # dff_filtered['spks'] = spks            

                # === Run post-dFF QC first so metrics are available for plotting ===
                self.logger.info(f"S2P_DFF: Running post-dF/F QC for {subject_id}")
                try:
                    qc_out = self._run_post_dff_qc(output_path, data['ops'], dff_filtered)
                    self.logger.info(f"S2P_DFF: Post-dF/F QC completed for {subject_id}")
                except Exception as e:
                    self.logger.warning(f"S2P_DFF: Post-dF/F QC failed for {subject_id}: {e}")
                    qc_out = None

                # Now load metrics CSV for plotting
                metrics_csv = os.path.join(output_path, 'qc_results', 'post_dff_qc.csv')
                metrics_df = None
                if os.path.exists(metrics_csv):
                    import pandas as pd
                    metrics_df = pd.read_csv(metrics_csv)

                # === Generate Step A single-window pipeline plots ===
                try:
                    if self._checktrace_enabled:
                        roi_ids = self._select_checktrace_rois(self._stage_capture.get('dff', dff_filtered),
                                                               metrics_df=metrics_df)
                        fs = float(self.config_manager.get_experiment_config()
                                   .get('imaging_preprocessing', {})
                                   .get('trace_extraction', {})
                                   .get('fs_hz', 29.760441593958042))
                        dff_arr = self._stage_capture.get('dff')
                        T = dff_arr.shape[1] if dff_arr is not None else 0
                        for rid in roi_ids:
                            windows = self._choose_window_for_roi(T, fs)
                            row_dict = None
                            if metrics_df is not None:
                                rsel = metrics_df[metrics_df['roi_id'] == rid]
                                if not rsel.empty:
                                    row_dict = rsel.iloc[0].to_dict()
                            self.logger.info(f"S2P_DFF: Step A plotting ROI {rid} in {subject_id}")
                            self.plot_roi_pipeline_single(rid, fs, windows, output_path, metrics_row=row_dict)
                    self.logger.info(f"S2P_DFF: Step A plotting completed for {subject_id}")
                except Exception as e:
                    self.logger.warning(f"S2P_DFF: checktrace plotting failed (Step A): {e}")






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
                # self.plot_dff_windowed_rasters(dff_filtered, time_vec, subject_id,
                #                                window_length_sec=60, overlap_sec=30,
                #                                save_path=check_dir, fs=fs, show=False)                

                # return
                
                

                # Save results
                # TODO need to move loading/deriving spks into initial suite2p modules
                success = self.save_dff_results(output_path, dff_filtered)


                # After saving dff:
                # === Generate Step A single-window pipeline plots ===
                # try:
                #     if self._checktrace_enabled:
                #         # Load metrics if available
                #         metrics_csv = os.path.join(output_path, 'qc_results', 'post_dff_qc.csv')
                #         metrics_df = None
                #         if os.path.exists(metrics_csv):
                #             import pandas as pd
                #             metrics_df = pd.read_csv(metrics_csv)
                #         roi_ids = self._select_checktrace_rois(self._stage_capture.get('dff', dff),
                #                                             metrics_df=metrics_df)                        
                #         # fs = fs
                #         dff_arr = self._stage_capture.get('dff')
                #         if dff_arr is not None:
                #             T = dff_arr.shape[1]
                #         else:
                #             T = 0
                #         for rid in roi_ids:
                #             windows = self._choose_window_for_roi(T, fs)
                #             # optional: metrics row
                #             row_dict = None
                #             if metrics_df is not None:
                #                 rsel = metrics_df[metrics_df['roi_id'] == rid]
                #                 if not rsel.empty:
                #                     row_dict = rsel.iloc[0].to_dict()
                #             self.logger.info(f"S2P_DFF: Step A plotting ROI {rid} in {subject_id}")
                #             self.plot_roi_pipeline_single(rid, fs, windows, output_path, metrics_row=row_dict)
                #     self.logger.info(f"S2P_DFF: Step A plotting completed for {subject_id}")
                # except Exception as e:
                #     self.logger.warning(f"S2P_DFF: checktrace plotting failed (Step A): {e}")


                # # Run post-dFF QC (optional)
                # self.logger.info(f"S2P_DFF: Running post-dF/F QC for {subject_id}")
                # try:
                #     qc_out = self._run_post_dff_qc(output_path, data['ops'], dff_filtered)
                #     self.logger.info(f"S2P_DFF: Post-dF/F QC completed for {subject_id}")
                # except Exception as e:
                #     self.logger.warning(f"S2P_DFF: Post-dF/F QC failed for {subject_id}: {e}")

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
    

