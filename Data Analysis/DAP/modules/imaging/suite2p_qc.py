"""
Suite2p Quality Control Module

Quality control filtering for Suite2p output data.
Follows the same pattern as other pipeline components.

This handles:
1. Load Suite2p raw data (F.npy, iscell.npy, ops.npy, etc.)
2. Apply quality control filters
3. Remove bad cells, artifacts, etc.
4. Save QC-filtered data
"""

import os
import numpy as np
import h5py
import logging
from typing import Dict, Any, List, Tuple, Optional
from skimage.measure import label
import math
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull

class Suite2pQC:
    """
    Suite2p Quality Control processor following pipeline component pattern.
    
    Handles QC filtering of Suite2p data with configurable parameters.
    """
    
    def __init__(self, config_manager, subject_list, logger=None):
        """
        Initialize the Suite2p QC processor.
        
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
        self.suite2p_output_path = self.config.get('paths', {}).get('suite2p_output', '')
        
        # Get QC parameters from config
        # self.qc_params = self.config.get('suite2p_qc', {})
        
        
        # Get QC method and parameters from config
        self.experiment_config = self.config_manager.get_experiment_config()
        self.imaging_config = self.experiment_config.get('imaging_preprocessing', {})
        self.qc_config = self.imaging_config.get('quality_control', {})                                
        
        self.logger.info("S2P_QC: Suite2pQC initialized")
    
    
    # ---------------- NEW QC HELPER FUNCTIONS ----------------

    def __get_qc_cfg(self) -> Dict[str, Any]:
        """
        Pull & normalize quality_control config block.
        Provides defaults if keys missing.
        """
        exp_cfg = self.config_manager.get_experiment_config()
        im_cfg = exp_cfg.get('imaging_preprocessing', {})
        qc = im_cfg.get('quality_control', {})
        qc = self.qc_config
        # Defaults
        defaults = {
            'use_iscell': True,
            'min_probcell': 0.0,
            'hard_filters': {
                'area_px': [20, 8000],
                'n_components_max': 1,
                'border_touch': False,
                'border_margin_px': 0,
            },
            'flag_filters': {
                'neuropil_ratio_max': 1.5,
                'snr_proxy_min': 1.0,
                'snr_quantile': 0.95,
                'motion_corr_max': 0.4,
            },
            'legacy_metrics': {
                'use': False,
                'range_skew': [0, 2],
                'range_aspect_ratio': [1.2, 5],
                'range_compact': [1.06, 5],
                'range_footprint': [1, 2],
                'max_connect': 2,
            },
            'soma_tag_rules': {
                'circularity_min': 0.60,
                'solidity_min': 0.85,
                'aspect_max': 1.8,
            },
            'update_suite2p_iscell': True,
        }
        # Deep-merge qc over defaults (shallow is fine for our structure)
        cfg = defaults
        for k, v in qc.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
        # Ensure legacy_metrics has 'use'
        if 'legacy_metrics' not in cfg or not isinstance(cfg['legacy_metrics'], dict):
            cfg['legacy_metrics'] = dict(use=False)
        cfg['legacy_metrics'].setdefault('use', False)
        return cfg        
        
        
        
        
        # # Defaults
        # defaults = dict(
        #     area_px_min=20,
        #     area_px_max=5000,
        #     n_components_max=3,
        #     neuropil_ratio_max=0.9,
        #     snr_proxy_min=2.0,
        #     motion_corr_max=20.0,
        #     circularity_min=0.25,
        #     solidity_min=0.75,
        #     aspect_max=4.0,
        #     legacy_metrics=dict(use=False)
        # )
        # out = {**defaults, **qc}
        # # Ensure legacy_metrics has 'use'
        # if 'legacy_metrics' not in out or not isinstance(out['legacy_metrics'], dict):
        #     out['legacy_metrics'] = dict(use=False)
        # out['legacy_metrics'].setdefault('use', False)
        # return out

    def __compute_geometry_metrics(self, ops: Dict[str, Any], stat: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Geometry per ROI (original ordering).
        """
        Ly = int(ops.get('Ly', 0)); Lx = int(ops.get('Lx', 0))
        n = len(stat)
        area_px = np.zeros(n, int)
        n_components = np.zeros(n, int)
        aspect = np.zeros(n, float)
        circularity = np.zeros(n, float)
        solidity = np.zeros(n, float)
        border_touch = np.zeros(n, bool)

        for i, s in enumerate(stat):
            y = np.asarray(s.get('ypix', []), int)
            x = np.asarray(s.get('xpix', []), int)
            area_px[i] = y.size
            if y.size == 0:
                n_components[i] = 0
                aspect[i] = np.nan
                circularity[i] = np.nan
                solidity[i] = np.nan
                continue
            # Build tight mask
            ymin, ymax = y.min(), y.max()
            xmin, xmax = x.min(), x.max()
            h = ymax - ymin + 1
            w = xmax - xmin + 1
            local_mask = np.zeros((h, w), bool)
            local_mask[y - ymin, x - xmin] = True
            # Connected components
            n_components[i] = np.max(label(local_mask, connectivity=1))
            # Aspect via PCA
            pts = np.column_stack([x, y]).astype(float)
            if pts.shape[0] >= 3:
                pts_center = pts - pts.mean(0)
                cov = np.cov(pts_center.T)
                ev, _ = np.linalg.eigh(cov)
                ev = np.sort(ev)
                if ev[0] <= 0:
                    aspect[i] = np.inf
                else:
                    aspect[i] = math.sqrt(ev[-1] / ev[0])
            else:
                aspect[i] = 1.0
            # Perimeter (4-neigh approx)
            eroded = binary_erosion(local_mask, border_value=0)
            perim_mask = local_mask & (~eroded)
            P = perim_mask.sum()
            A = area_px[i]
            if P > 0:
                circularity[i] = 4 * math.pi * A / (P * P)
            else:
                circularity[i] = np.nan
            # Solidity via convex hull
            if ConvexHull is not None and pts.shape[0] >= 3:
                try:
                    # hull = ConvexHull(pts)
                    # hull_area = hull.area if hasattr(hull, 'area') else hull.volume
                    # if hull_area <= 0:
                    #     solidity[i] = np.nan
                    # else:
                    #     solidity[i] = A / hull_area
                                                
                    hull = ConvexHull(pts)
                    hull_area = getattr(hull, 'volume', None)  # in 2D, 'volume' is area
                    if hull_area is None:  # fallback if SciPy behavior changes
                        hull_area = hull.volume if hasattr(hull, 'volume') else hull.area
                    solidity[i] = A / hull_area if hull_area and hull_area > 0 else np.nan                                                
                        
                except Exception:
                    solidity[i] = np.nan
            else:
                solidity[i] = np.nan
            # Border touch
            # if (ymin == 0) or (xmin == 0) or (ymax == Ly - 1) or (xmax == Lx - 1):
            #     border_touch[i] = True
            margin = int(self.__get_qc_cfg()['hard_filters'].get('border_margin_px', 0))
            if (ymin <= 0 + margin) or (xmin <= 0 + margin) or (ymax >= Ly - 1 - margin) or (xmax >= Lx - 1 - margin):
                border_touch[i] = True
                

        return dict(
            area_px=area_px,
            n_components=n_components,
            aspect=aspect,
            circularity=circularity,
            solidity=solidity,
            border_touch=border_touch
        )

    def __compute_signal_flags(self, F: np.ndarray, Fneu: np.ndarray,
                                ops: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Compute signal-related metrics (arrays length n_rois).
        """
        n = F.shape[0]
        mean_F = F.mean(1)
        mean_Fneu = Fneu.mean(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            neuropil_ratio = np.where(mean_F > 0, mean_Fneu / mean_F, np.nan)

        dF = F - Fneu
        # Robust stats per ROI
        
        q = float(cfg.get('flag_filters', {}).get('snr_quantile', 0.95))
        p = int(round(q * 100))
        p_hi = np.percentile(dF, p, axis=1)
        # p_hi = np.percentile(dF, 95, axis=1)
        med = np.median(dF, axis=1)
        mad = np.median(np.abs(dF - med[:, None]), axis=1)
        mad_safe = np.where(mad == 0, np.nan, mad)
        snr_proxy = (p_hi - med) / mad_safe

        # Motion proxy (global -> broadcast)
        # xoff = np.asarray(ops.get('xoff', []))
        # yoff = np.asarray(ops.get('yoff', []))
        # if xoff.size and yoff.size and xoff.shape == yoff.shape:
        #     motion_series = np.abs(np.diff(xoff)) + np.abs(np.diff(yoff))
        #     motion_proxy_val = np.max(motion_series) if motion_series.size else 0.0
        # else:
        #     motion_proxy_val = np.nan
        # motion_proxy = np.full(n, motion_proxy_val, float)
        
        xoff = np.asarray(ops.get('xoff', []))
        yoff = np.asarray(ops.get('yoff', []))
        if xoff.size and yoff.size and xoff.shape == yoff.shape:
            motion_proxy_val = np.max(np.abs(xoff) + np.abs(yoff))
        else:
            motion_proxy_val = np.nan
        motion_proxy = np.full(n, motion_proxy_val, float)

        return dict(
            neuropil_ratio=neuropil_ratio,
            snr_proxy=snr_proxy,
            motion_proxy=motion_proxy
        )

    def __apply_hard_filters(self, base_mask: np.ndarray,
                                geom: Dict[str, np.ndarray],
                                cfg: Dict[str, Any]) -> np.ndarray:
        """
        Hard gates: area range + n_components.
        """
        hf = cfg['hard_filters']
        area_min, area_max = hf['area_px']
        area_ok = (geom['area_px'] >= area_min) & (geom['area_px'] <= area_max)
        comp_ok = (geom['n_components'] <= hf['n_components_max'])
        good = base_mask & area_ok & comp_ok
        if hf.get('border_touch', False):
            good &= ~geom['border_touch']
        return good        
        

        # area_ok = (geom['area_px'] >= cfg['area_px_min']) & (geom['area_px'] <= cfg['area_px_max'])
        # comp_ok = geom['n_components'] <= cfg['n_components_max']
        # good = base_mask & area_ok & comp_ok
        # return good

    def __make_flag_dict(self, geom: Dict[str, np.ndarray],
                            sig: Dict[str, np.ndarray],
                            cfg: Dict[str, Any],
                            legacy: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Build annotation flags (no dropping).
        """
        flag_cfg = cfg.get('flag_filters', {})
        flags = dict(
            flag_neuropil=(sig['neuropil_ratio'] > flag_cfg['neuropil_ratio_max']),
            flag_low_snr=(sig['snr_proxy'] < flag_cfg['snr_proxy_min']),
            flag_motion=(sig['motion_proxy'] > flag_cfg['motion_corr_max'])
        )
        if legacy and legacy.get('use', False):
            # Placeholder: combine out-of-range on some legacy metric
            flags['flag_legacy_outside'] = np.zeros_like(flags['flag_motion'])
        return flags

    def __compute_tags(self, geom: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Semantic tags (soma-like vs dendrite-like).
        """
        tag_cfg = cfg.get('soma_tag_rules', {})
        soma = (
            (geom['circularity'] >= tag_cfg['circularity_min']) &
            (geom['solidity'] >= tag_cfg['solidity_min']) &
            (geom['aspect'] <= tag_cfg['aspect_max'])
        )
        tags = dict(
            tag_soma_like=soma,
            tag_dendrite_like=~soma
        )
        return tags

    def __summarize_qc(self,
                        base_mask: np.ndarray,
                        good_mask: np.ndarray,
                        flags: Dict[str, np.ndarray],
                        tags: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Summary counts for logging & persistence.
        """
        stats = dict(
            n_total=int(base_mask.size),
            n_base_iscell=int(base_mask.sum()),
            n_after_hard_filters=int(good_mask.sum())
        )
        for k, arr in flags.items():
            stats[f"{k}_total"] = int(arr.sum())
            stats[f"{k}_kept"] = int(arr[good_mask].sum())
        for k, arr in tags.items():
            stats[f"{k}_kept"] = int(arr[good_mask].sum())
        return stats


    def load_suite2p_data(self, subject_id: str, session_path: str) -> Optional[Dict[str, Any]]:
        """
        Load Suite2p data files for a session.
        
        Args:
            subject_id: Subject identifier
            session_path: Path to Suite2p output directory
            
        Returns:
            Dictionary containing loaded Suite2p data or None if failed
        """
        try:
            data = {}
            
            # Define required files
            required_files = {
                'F': 'F.npy',
                'Fneu': 'Fneu.npy', 
                'iscell': 'iscell.npy',
                'ops': 'ops.npy',
                'stat': 'stat.npy',
                'spks': 'spks.npy'
            }
            
            # Load each required file
            for key, filename in required_files.items():
                filepath = os.path.join(session_path, filename)
                if os.path.exists(filepath):
                    data[key] = np.load(filepath, allow_pickle=True)
                    self.logger.debug(f"S2P_QC: Loaded {filename} for {subject_id}")
                else:
                    self.logger.warning(f"S2P_QC: Missing {filename} for {subject_id}")
                    return None
            
            self.logger.info(f"S2P_QC: Successfully loaded Suite2p data for {subject_id}")
            return data
            
        except Exception as e:
            self.logger.error(f"S2P_QC: Failed to load Suite2p data for {subject_id}: {e}")
            return None
    
    def get_suite2p_metrics(self, ops: Dict, stat: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Extract Suite2p quality metrics from stat array.
        
        Args:
            ops: Suite2p ops dictionary
            stat: Suite2p stat array
            
        Returns:
            Tuple of metric arrays (skew, connect, aspect, compact, footprint)
        """
        # Extract existing statistics for masks
        footprint = np.array([stat[i]['footprint'] for i in range(len(stat))])
        skew = np.array([stat[i]['skew'] for i in range(len(stat))])
        aspect = np.array([stat[i]['aspect_ratio'] for i in range(len(stat))])
        compact = np.array([stat[i]['compact'] for i in range(len(stat))])
        
        # Compute connectivity of ROIs
        masks = self.stat_to_masks(ops, stat)
        connect = []
        for i in np.unique(masks)[1:]:
            # Find a mask with one roi
            m = masks.copy() * (masks == i)
            # Find component number
            connect.append(np.max(label(m, connectivity=1)))
        connect = np.array(connect)
        
        return skew, connect, aspect, compact, footprint

    def stat_to_masks(self, ops: Dict, stat: np.ndarray) -> np.ndarray:
        """
        Convert stat.npy results to ROI masks matrix.
        Uses FULL image dimensions (Ly, Lx) to match original QC behavior.
        
        Args:
            ops: Suite2p ops dictionary
            stat: Suite2p stat array
            
        Returns:
            2D array with ROI masks at full image size
        """
        # Use full image dimensions, not cropped ones
        masks = np.zeros((ops['Ly'], ops['Lx']), dtype=np.int32)
        for n in range(len(stat)):
            ypix = stat[n]['ypix']
            xpix = stat[n]['xpix']
            masks[ypix, xpix] = n + 1
        return masks

    def _export_roi_geometry_npz(self, ops, stat_final, out_path):
        from scipy.ndimage import binary_erosion
    
        rows = []
        core_rows = []
        edge_rows = []
    
        for s in stat_final:
            y = np.asarray(s.get('ypix', []), dtype=np.int32)
            x = np.asarray(s.get('xpix', []), dtype=np.int32)
    
            # ROI pixels (2, n_i)
            rows.append(np.vstack((y, x)))
    
            # Core/edge via tight local mask
            if y.size:
                ymin, ymax = y.min(), y.max()
                xmin, xmax = x.min(), x.max()
                local = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=bool)
                local[y - ymin, x - xmin] = True
                core = binary_erosion(local, iterations=1, border_value=0)
                edge = local & (~core)
    
                y_core, x_core = np.nonzero(core)
                y_edge, x_edge = np.nonzero(edge)
    
                core_rows.append(np.vstack((y_core + ymin, x_core + xmin)).astype(np.int32))
                edge_rows.append(np.vstack((y_edge + ymin, x_edge + xmin)).astype(np.int32))
            else:
                core_rows.append(np.empty((2, 0), dtype=np.int32))
                edge_rows.append(np.empty((2, 0), dtype=np.int32))
    
        # Build object arrays explicitly to avoid broadcasting issues
        roi_pixels = np.empty(len(rows), dtype=object)
        roi_core_pixels = np.empty(len(core_rows), dtype=object)
        roi_edge_pixels = np.empty(len(edge_rows), dtype=object)
        for i in range(len(rows)):
            roi_pixels[i] = rows[i]
            roi_core_pixels[i] = core_rows[i]
            roi_edge_pixels[i] = edge_rows[i]
    
        np.savez_compressed(
            out_path,
            roi_pixels=roi_pixels,
            roi_core_pixels=roi_core_pixels,
            roi_edge_pixels=roi_edge_pixels,
            Ly=int(ops['Ly']),
            Lx=int(ops['Lx']),
        )



    
    
    def _compute_neuropil_rings(self, masks: np.ndarray, inner: int = 3, outer: int = 25) -> np.ndarray:
        """
        Build neuropil rings for each kept ROI label in `masks`.
        Returns an object array where each element is a (2, N_i) int32 array of [y; x] ring coords.
        """
        from scipy.ndimage import distance_transform_edt, binary_dilation
        assert inner >= 0 and outer > inner, "outer must be > inner"
        roi_ids = np.unique(masks)
        roi_ids = roi_ids[roi_ids > 0]  # drop background
        occupied = masks > 0
    
        rings_obj = np.empty(roi_ids.size, dtype=object)
    
        for idx, rid in enumerate(roi_ids):
            roi = (masks == rid)
    
            # Distance from the ROI boundary for all pixels (0 inside ROI)
            dist = distance_transform_edt(~roi)
    
            # Ring between [inner, outer], excluding any ROI pixels from any label
            ring = (dist >= inner) & (dist <= outer)
            ring &= ~occupied
    
            y, x = np.nonzero(ring)
            rings_obj[idx] = np.vstack((y.astype(np.int32), x.astype(np.int32)))
    
        return rings_obj


    def _save_neuropil_rings(self, rings_obj: np.ndarray, out_path: str):
        # rings_obj is already dtype=object with per-ROI (2, N_i) arrays
        np.savez_compressed(out_path, neuropil_pixels=rings_obj)




    def apply_qc_filters(self, data: Dict[str, Any], subject_id: str, output_path: str) -> Dict[str, Any]:
        """
        Apply quality control filters to Suite2p data using configured QC method.
        
        Args:
            data: Raw Suite2p data dictionary
            subject_id: Subject identifier
            
        Returns:
            QC-filtered data dictionary
        """
        try:
            self.logger.info(f"S2P_QC: === Starting QC filtering for {subject_id} ===")
            
            # Extract data arrays
            F = data['F']
            Fneu = data['Fneu']
            stat = data['stat']
            iscell = data['iscell'].copy()  # Make a copy so we can modify it
            ops = data['ops'].item()  # ops is saved as 0-d array
            spks = data['spks']

            self.logger.info(f"S2P_QC: Loaded data - {F.shape[0]} ROIs, {F.shape[1]} time points")
            self.logger.info(f"S2P_QC: Initial manual selections: {np.sum(iscell[:, 0] == 1)} selected, {np.sum(iscell[:, 0] == 0)} rejected")
            
            # Get QC method and parameters from config
            # experiment_config = self.config_manager.get_experiment_config()
            # self.imaging_config = experiment_config.get('imaging_preprocessing', {})
            qc_method = self.qc_config.get('qc_method', 'manual')          
            
            self.logger.info(f"S2P_QC: Using QC method '{qc_method}' for {subject_id}")
            
            # Apply appropriate QC method
            if qc_method == 'manual':
                self.logger.info("S2P_QC: Applying manual QC (using existing iscell selections)...")
                good_roi_mask = self._apply_manual_qc(iscell, subject_id)
            elif qc_method == 'threshold':
                self.logger.info("S2P_QC: Applying threshold-based QC (using config parameters)...")
                good_roi_mask = self._apply_threshold_qc(ops, stat, subject_id)
            elif qc_method == 'threshold_learn':
                self.logger.info("S2P_QC: Applying threshold learning QC (learn from manual, apply to all)...")
                good_roi_mask = self._apply_threshold_learn_qc(ops, stat, iscell, subject_id)
            elif qc_method == 'qc_new':
                self.logger.info("S2P_QC: Applying new QC method (two-stage)...")
                
                # Stage 1: base + geometry
                base_mask = self._apply_qc_new(ops, stat, iscell, subject_id)
                cfg = self._qc_new_cfg
                geom = self._qc_new_temp_geom
                
                # Stage 2: signal metrics on full arrays
                self.logger.info("S2P_QC: qc_new: computing signal metrics")
                sig = self.__compute_signal_flags(F, Fneu, ops, cfg)
                
                # Final hard filters (geometry-only + optional border-touch)
                good_roi_mask = self.__apply_hard_filters(base_mask, geom, cfg)
                
                # Build flags/tags (annotation only)
                flags = self.__make_flag_dict(geom, sig, cfg, cfg.get('legacy_metrics'))
                tags = self.__compute_tags(geom, cfg)
                
                # Log percent flagged (base vs kept) using counts
                base_count = int(base_mask.sum()) if qc_method == 'qc_new' else F.shape[0]
                for k, arr in flags.items():
                    pct_all = 100.0 * arr[base_mask].sum() / base_count if base_count else 0.0
                    kept_count = good_roi_mask.sum()
                    pct_kept = 100.0 * arr[good_roi_mask].sum() / kept_count if kept_count else 0.0
                    self.logger.info(f"S2P_QC: {k}: {pct_all:.1f}% of base flagged; {pct_kept:.1f}% of kept flagged")
                
                
                # Optionally write back iscell for Suite2p GUI                          
                if bool(cfg.get('update_suite2p_iscell', True)):
                    iscell[:, 0] = good_roi_mask.astype(int)
                # qc_data['iscell_updated'] = iscell
                
                
                # Persist extended results for caller
                stats_ext = self.__summarize_qc(base_mask, good_roi_mask, flags, tags)
                self._qc_new_payload = dict(metrics={**geom, **sig},
                                            flags=flags,
                                            tags=tags,
                                            stats_ext=stats_ext,
                                            cfg=cfg)
                
                self.logger.info(f"S2P_QC: qc_new: base {base_mask.sum()} -> kept {good_roi_mask.sum()}")
            else:
                self.logger.warning(f"S2P_QC: Unknown QC method '{qc_method}', using threshold")
                good_roi_mask = self._apply_threshold_qc(ops, stat, subject_id)
            
            
            kept_idx_orig = np.flatnonzero(good_roi_mask).astype(np.int32)
            kept_idx_seq = np.arange(kept_idx_orig.size, dtype=np.int32)
            self.logger.info(f"S2P_QC: Keeping {kept_idx_orig.size} ROIs (original indices: {kept_idx_orig})")

            
            
            
            # Initial cell count
            n_initial_cells = F.shape[0]
            
            self.logger.info(f"S2P_QC: Applying QC mask to extract final data...")
            
            # Apply the mask to get final data
            F = F[good_roi_mask, :]
            Fneu = Fneu[good_roi_mask, :]
            stat_final = stat[good_roi_mask]
            spks = spks[good_roi_mask, :]
            
            # iscell[good_roi_mask, 0] = 1
            
            self.logger.info(f"S2P_QC: Generating spatial masks for {len(stat_final)} final ROIs...")
            
            # Generate masks for final ROIs
            masks = self.stat_to_masks(ops, stat_final)

            # Export geometry information
            geom_npz = os.path.join(output_path, 'qc_results', 'roi_geometry.npz')
            self._export_roi_geometry_npz(ops, stat_final, geom_npz)

            # Compute neuropil rings
            rings = self._compute_neuropil_rings(masks, inner=3, outer=25)
            rings_npz = os.path.join(output_path, 'qc_results', 'neuropil_rings.npz')
            self._save_neuropil_rings(rings, rings_npz)
            
            
            # np.savez_compressed(os.path.join(output_path, 'qc_results', 'neuropil_rings.npz'),
            #                     neuropil_pixels=rings)

            qc_data = {
                'F': F,
                'Fneu': Fneu,
                'stat': stat_final,
                'masks': masks,
                'ops': ops,
                'spks': spks,
                'iscell_updated': iscell,  # Include updated iscell array
                'kept_idx_orig': kept_idx_orig,
                'kept_idx_seq': kept_idx_seq,
                'qc_stats': {
                    'n_initial_cells': n_initial_cells,
                    'n_final_cells': np.sum(good_roi_mask),
                    'n_rejected_cells': n_initial_cells - np.sum(good_roi_mask),
                    'rejection_rate': (n_initial_cells - np.sum(good_roi_mask)) / n_initial_cells,
                    'qc_method': qc_method,
                    'good_roi_mask': good_roi_mask
                }
            }
            
            
            # Attach extended qc_new artifacts
            if qc_method == 'qc_new' and hasattr(self, '_qc_new_payload'):
                p = self._qc_new_payload
                qc_data['qc_metrics_full'] = p['metrics']          # dict of arrays (original order)
                qc_data['qc_flags'] = p['flags']
                qc_data['qc_tags'] = p['tags']
                qc_data['qc_stats'].update(p['stats_ext'])            
            
            


            
            
            self.logger.info(f"S2P_QC: === QC filtering complete for {subject_id} ===")
            self.logger.info(f"S2P_QC: Final result: {n_initial_cells} -> {np.sum(good_roi_mask)} cells "
                           f"({n_initial_cells - np.sum(good_roi_mask)} rejected, "
                           f"{(n_initial_cells - np.sum(good_roi_mask))/n_initial_cells:.1%} rejection rate)")
            
            return qc_data
            
        except Exception as e:
            self.logger.error(f"S2P_QC: QC filtering failed for {subject_id}: {e}")
            return data




    # ---------------- UPDATED qc_new METHOD ----------------

    def _apply_qc_new(self, ops: Dict, stat: np.ndarray, iscell: np.ndarray, subject_id: str) -> np.ndarray:
        """
        New QC pipeline: start from manual iscell, apply geometry hard filters only.
        Annotates metrics/flags/tags (saved later).
        """
        self.logger.info("S2P_QC: qc_new: loading config & base mask")
        cfg = self.__get_qc_cfg()
        # base = (iscell[:, 0] == 1)
        base = np.ones(len(stat), dtype=bool)
        if cfg.get('use_iscell', True):
            base &= (iscell[:, 0] == 1)
            min_prob = float(cfg.get('min_probcell', 0.0))
            if iscell.shape[1] >= 2 and min_prob > 0.0:
                base &= (iscell[:, 1] >= min_prob)
        if base.sum() == 0:
            self.logger.warning("S2P_QC: qc_new: no base iscell selections; using ALL as base")
            base = np.ones_like(base, bool)
        self.logger.info("S2P_QC: qc_new: computing geometry metrics")
        geom = self.__compute_geometry_metrics(ops, stat)
        self.logger.info("S2P_QC: qc_new: geometry stats "
                            f"area_px median={np.median(geom['area_px']):.1f} "
                            f"aspect median={np.nanmedian(geom['aspect']):.2f} "
                            f"circ median={np.nanmedian(geom['circularity']):.2f} "
                            f"sol median={np.nanmedian(geom['solidity']):.2f}")

        # Signal metrics
        # (F/Fneu not available here; will be provided by caller apply_qc_filters)
        # Defer: In apply_qc_filters we call this AFTER loading F/Fneu.
        # For convenience compute minimal placeholders here; replaced later if needed.
        self._qc_new_temp_geom = geom  # cache for later extension
        self._qc_new_cfg = cfg  # cache config
        # Return placeholder; final good mask computed in apply_qc_filters (after signal metrics)
        # We still need to compute signal metrics here -> move logic into this method; pass F arrays.

        # This method signature lacks F, Fneu; adjust by computing in apply_qc_filters.
        # We'll store a marker for apply_qc_filters to finish the pipeline.
        iscell[:, 0] = base.astype(int)
        return base  # provisional; final gating happens later




    def _apply_manual_qc(self, iscell: np.ndarray, subject_id: str) -> np.ndarray:
        """
        Apply manual QC using iscell selections from Suite2p GUI.
        
        Args:
            iscell: Suite2p iscell array
            subject_id: Subject identifier
            
        Returns:
            Boolean mask for good ROIs
        """
        # Simply use manual selections
        good_roi_mask = iscell[:, 0] == 1
        
        self.logger.info(f"S2P_QC: Manual QC selected {np.sum(good_roi_mask)} cells for {subject_id}")
        return good_roi_mask
    
    def _apply_threshold_qc(self, ops: Dict, stat: np.ndarray, subject_id: str) -> np.ndarray:
        """
        Apply threshold-based QC using config parameters.
        
        Args:
            ops: Suite2p ops dictionary
            stat: Suite2p stat array
            subject_id: Subject identifier
            
        Returns:
            Boolean mask for good ROIs
        """
        self.logger.info("S2P_QC: Loading QC parameters from config...")
        
        # Get QC parameters from config
        experiment_config = self.config_manager.get_experiment_config()
        imaging_config = experiment_config.get('imaging_preprocessing', {})
        qc_params = imaging_config.get('quality_control', {})
        
        range_skew = qc_params.get('range_skew', [-5, 5])
        max_connect = qc_params.get('max_connect', 1)
        range_aspect_ratio = qc_params.get('range_aspect_ratio', [0, 5])
        range_compact = qc_params.get('range_compact', [0, 1.06])
        range_footprint = qc_params.get('range_footprint', [1, 2])
        
        self.logger.info(f"S2P_QC: Threshold parameters:")
        self.logger.info(f"S2P_QC:   range_skew: {range_skew}")
        self.logger.info(f"S2P_QC:   max_connect: {max_connect}")
        self.logger.info(f"S2P_QC:   range_aspect_ratio: {range_aspect_ratio}")
        self.logger.info(f"S2P_QC:   range_compact: {range_compact}")
        self.logger.info(f"S2P_QC:   range_footprint: {range_footprint}")
        
        self.logger.info("S2P_QC: Computing Suite2p quality metrics...")
        
        # Get Suite2p quality metrics
        skew, connect, aspect, compact, footprint = self.get_suite2p_metrics(ops, stat)
        
        self.logger.info("S2P_QC: Applying threshold filters...")
        
        # Apply thresholds and count each filter's effect
        skew_mask = (skew >= range_skew[0]) & (skew <= range_skew[1])
        connect_mask = connect <= max_connect
        aspect_mask = (aspect >= range_aspect_ratio[0]) & (aspect <= range_aspect_ratio[1])
        compact_mask = (compact >= range_compact[0]) & (compact <= range_compact[1])
        footprint_mask = (footprint >= range_footprint[0]) & (footprint <= range_footprint[1])
        
        # Log individual filter results
        self.logger.info(f"S2P_QC: Filter results:")
        self.logger.info(f"S2P_QC:   skew: {np.sum(skew_mask)}/{len(skew_mask)} pass")
        self.logger.info(f"S2P_QC:   connect: {np.sum(connect_mask)}/{len(connect_mask)} pass")
        self.logger.info(f"S2P_QC:   aspect: {np.sum(aspect_mask)}/{len(aspect_mask)} pass")
        self.logger.info(f"S2P_QC:   compact: {np.sum(compact_mask)}/{len(compact_mask)} pass")
        self.logger.info(f"S2P_QC:   footprint: {np.sum(footprint_mask)}/{len(footprint_mask)} pass")
        
        # Combine all filters
        good_roi_mask = skew_mask & connect_mask & aspect_mask & compact_mask & footprint_mask
        
        self.logger.info(f"S2P_QC: Threshold QC selected {np.sum(good_roi_mask)}/{len(good_roi_mask)} cells for {subject_id}")
        return good_roi_mask
    
    def _apply_threshold_learn_qc(self, ops: Dict, stat: np.ndarray, iscell: np.ndarray, 
                                  subject_id: str) -> np.ndarray:
        """
        Apply threshold learning QC: learn thresholds from manual selection, then apply to all cells.
        Updates iscell array to promote cells that pass learned thresholds.
        
        Args:
            ops: Suite2p ops dictionary
            stat: Suite2p stat array
            iscell: Suite2p iscell array (modified in-place)
            subject_id: Subject identifier
            
        Returns:
            Boolean mask for good ROIs
        """
        # Get manually selected cells
        manual_mask = iscell[:, 0] == 1
        n_manual = np.sum(manual_mask)
        
        self.logger.info(f"S2P_QC: Found {n_manual} manually selected cells to learn from")
        
        if n_manual == 0:
            self.logger.warning(f"S2P_QC: No manual selections found for {subject_id}, using threshold QC")
            return self._apply_threshold_qc(ops, stat, subject_id)
        
        self.logger.info("S2P_QC: Learning thresholds from manually selected cells...")
        
        # Learn thresholds from manually selected cells
        learned_params = self._learn_thresholds_from_manual(ops, stat, manual_mask, subject_id)
        
        self.logger.info("S2P_QC: Computing Suite2p quality metrics for all cells...")
        
        # Apply learned thresholds to ALL cells (not just manual selections)
        skew, connect, aspect, compact, footprint = self.get_suite2p_metrics(ops, stat)
        
        self.logger.info("S2P_QC: Applying learned thresholds to all cells...")
        
        # Apply learned thresholds and count each filter's effect
        skew_mask = (skew >= learned_params['range_skew'][0]) & (skew <= learned_params['range_skew'][1])
        connect_mask = connect <= learned_params['max_connect']
        aspect_mask = (aspect >= learned_params['range_aspect_ratio'][0]) & (aspect <= learned_params['range_aspect_ratio'][1])
        compact_mask = (compact >= learned_params['range_compact'][0]) & (compact <= learned_params['range_compact'][1])
        footprint_mask = (footprint >= learned_params['range_footprint'][0]) & (footprint <= learned_params['range_footprint'][1])
        
        # Log individual learned filter results
        self.logger.info(f"S2P_QC: Learned filter results:")
        self.logger.info(f"S2P_QC:   skew: {np.sum(skew_mask)}/{len(skew_mask)} pass")
        self.logger.info(f"S2P_QC:   connect: {np.sum(connect_mask)}/{len(connect_mask)} pass")
        self.logger.info(f"S2P_QC:   aspect: {np.sum(aspect_mask)}/{len(aspect_mask)} pass")
        self.logger.info(f"S2P_QC:   compact: {np.sum(compact_mask)}/{len(compact_mask)} pass")
        self.logger.info(f"S2P_QC:   footprint: {np.sum(footprint_mask)}/{len(footprint_mask)} pass")
        
        final_mask = skew_mask & connect_mask & aspect_mask & compact_mask & footprint_mask
        
        # Count how many cells were promoted from manual rejection
        manual_rejected = iscell[:, 0] == 0
        promoted_cells = final_mask & manual_rejected
        n_promoted = np.sum(promoted_cells)
        
        # Count how many manual selections were retained
        manual_retained = final_mask & manual_mask
        n_retained = np.sum(manual_retained)
        
        self.logger.info(f"S2P_QC: Threshold learning results:")
        self.logger.info(f"S2P_QC:   Manual cells retained: {n_retained}/{n_manual}")
        self.logger.info(f"S2P_QC:   Cells promoted from rejected: {n_promoted}")
        
        # Update iscell array to promote cells that pass learned thresholds
        if n_promoted > 0:
            iscell[promoted_cells, 0] = 1  # Set promoted cells to selected
            self.logger.info(f"S2P_QC: Updated iscell array - promoted {n_promoted} cells from rejected to selected")
        
        self.logger.info(f"S2P_QC: Threshold learning QC for {subject_id}: "
                        f"learned from {n_manual} manual selections, "
                        f"final={np.sum(final_mask)} cells "
                        f"(promoted {n_promoted} from rejected)")
        
        return final_mask
    
    def _learn_thresholds_from_manual(self, ops: Dict, stat: np.ndarray, 
                                     manual_mask: np.ndarray, subject_id: str) -> Dict[str, Any]:
        """
        Learn quality control thresholds from manually selected cells.
        
        Args:
            ops: Suite2p ops dictionary
            stat: Suite2p stat array
            manual_mask: Boolean mask for manually selected cells
            subject_id: Subject identifier
            
        Returns:
            Dictionary with learned threshold parameters
        """
        # Get metrics for manually selected cells
        skew, connect, aspect, compact, footprint = self.get_suite2p_metrics(ops, stat)
        
        # Extract values for manual selections only
        manual_skew = skew[manual_mask]
        manual_connect = connect[manual_mask]
        manual_aspect = aspect[manual_mask]
        manual_compact = compact[manual_mask]
        manual_footprint = footprint[manual_mask]
        
        # Learn thresholds based on parameter type
        learned_params = {
            # Range parameters: use min/max of manual selections
            'range_skew': [np.min(manual_skew), np.max(manual_skew)],
            'range_aspect_ratio': [np.min(manual_aspect), np.max(manual_aspect)],
            'range_compact': [np.min(manual_compact), np.max(manual_compact)],
            'range_footprint': [np.min(manual_footprint), np.max(manual_footprint)],
            
            # Threshold parameters: use max of manual selections (to be inclusive)
            'max_connect': np.max(manual_connect),  # Use max instead of mean to be more inclusive
        }
        
        # Add tolerance to ranges to avoid being too restrictive
        tolerance_factor = 0.1  # 10% tolerance
        min_tolerance = 1e-6   # Minimum absolute tolerance for near-zero ranges
        
        # Expand ranges with proper handling of zero-width ranges
        for param_name in ['range_skew', 'range_aspect_ratio', 'range_compact', 'range_footprint']:
            param_range = learned_params[param_name]
            range_width = param_range[1] - param_range[0]
            
            if range_width < min_tolerance:
                # If range is very small, expand symmetrically around the value
                center = (param_range[0] + param_range[1]) / 2
                expansion = max(abs(center) * tolerance_factor, min_tolerance)
                learned_params[param_name] = [center - expansion, center + expansion]
            else:
                # Normal range expansion
                expansion = range_width * tolerance_factor
                learned_params[param_name][0] -= expansion
                learned_params[param_name][1] += expansion
        
        # Add tolerance to max_connect (increase threshold to be more inclusive)
        learned_params['max_connect'] += max(learned_params['max_connect'] * tolerance_factor, min_tolerance)
        
        self.logger.info(f"S2P_QC: Learned thresholds for {subject_id} from {np.sum(manual_mask)} manual selections:")
        self.logger.info(f"S2P_QC:   range_skew: [{learned_params['range_skew'][0]:.3f}, {learned_params['range_skew'][1]:.3f}]")
        self.logger.info(f"S2P_QC:   range_aspect_ratio: [{learned_params['range_aspect_ratio'][0]:.3f}, {learned_params['range_aspect_ratio'][1]:.3f}]")
        self.logger.info(f"S2P_QC:   range_compact: [{learned_params['range_compact'][0]:.3f}, {learned_params['range_compact'][1]:.3f}]")
        self.logger.info(f"S2P_QC:   range_footprint: [{learned_params['range_footprint'][0]:.3f}, {learned_params['range_footprint'][1]:.3f}]")
        self.logger.info(f"S2P_QC:   max_connect: {learned_params['max_connect']:.3f}")
        
        # Verify that all manual selections pass the learned thresholds
        self._verify_learned_thresholds(ops, stat, manual_mask, learned_params, subject_id)
        
        return learned_params

    def _verify_learned_thresholds(self, ops: Dict, stat: np.ndarray, manual_mask: np.ndarray, 
                                  learned_params: Dict[str, Any], subject_id: str):
        """
        Verify that all manually selected cells pass the learned thresholds.
        
        Args:
            ops: Suite2p ops dictionary
            stat: Suite2p stat array
            manual_mask: Boolean mask for manually selected cells
            learned_params: Learned threshold parameters
            subject_id: Subject identifier
        """
        # Get metrics for verification
        skew, connect, aspect, compact, footprint = self.get_suite2p_metrics(ops, stat)
        
        # Apply learned thresholds to manual selections only
        manual_indices = np.where(manual_mask)[0]
        
        # Check each filter on manual selections
        skew_pass = (skew[manual_mask] >= learned_params['range_skew'][0]) & (skew[manual_mask] <= learned_params['range_skew'][1])
        connect_pass = connect[manual_mask] <= learned_params['max_connect']
        aspect_pass = (aspect[manual_mask] >= learned_params['range_aspect_ratio'][0]) & (aspect[manual_mask] <= learned_params['range_aspect_ratio'][1])
        compact_pass = (compact[manual_mask] >= learned_params['range_compact'][0]) & (compact[manual_mask] <= learned_params['range_compact'][1])
        footprint_pass = (footprint[manual_mask] >= learned_params['range_footprint'][0]) & (footprint[manual_mask] <= learned_params['range_footprint'][1])
        
        # Log verification results
        n_manual = np.sum(manual_mask)
        self.logger.info(f"S2P_QC: Verification - manual cells passing learned thresholds:")
        self.logger.info(f"S2P_QC:   skew: {np.sum(skew_pass)}/{n_manual}")
        self.logger.info(f"S2P_QC:   connect: {np.sum(connect_pass)}/{n_manual}")
        self.logger.info(f"S2P_QC:   aspect: {np.sum(aspect_pass)}/{n_manual}")
        self.logger.info(f"S2P_QC:   compact: {np.sum(compact_pass)}/{n_manual}")
        self.logger.info(f"S2P_QC:   footprint: {np.sum(footprint_pass)}/{n_manual}")
        
        # Check if all manual selections pass all filters
        all_pass = skew_pass & connect_pass & aspect_pass & compact_pass & footprint_pass
        n_all_pass = np.sum(all_pass)
        
        if n_all_pass != n_manual:
            self.logger.warning(f"S2P_QC: WARNING - Only {n_all_pass}/{n_manual} manual selections pass learned thresholds!")
            # Log which manual cells failed and why
            failed_indices = manual_indices[~all_pass]
            for idx in failed_indices[:5]:  # Show first 5 failures
                self.logger.warning(f"S2P_QC:   Manual cell {idx} failed: "
                                  f"skew={skew[idx]:.3f} ({'pass' if skew_pass[idx-manual_indices[0]] else 'fail'}), "
                                  f"connect={connect[idx]:.3f} ({'pass' if connect_pass[idx-manual_indices[0]] else 'fail'}), "
                                  f"aspect={aspect[idx]:.3f} ({'pass' if aspect_pass[idx-manual_indices[0]] else 'fail'}), "
                                  f"compact={compact[idx]:.3f} ({'pass' if compact_pass[idx-manual_indices[0]] else 'fail'}), "
                                  f"footprint={footprint[idx]:.3f} ({'pass' if footprint_pass[idx-manual_indices[0]] else 'fail'})")
        else:
            self.logger.info(f"S2P_QC: All {n_manual} manual selections pass learned thresholds")
    
    def save_motion_correction(self, ops: Dict, output_path: str, subject_id: str) -> bool:
        """
        Save motion correction offsets to match original workflow.
        
        Args:
            ops: Suite2p ops dictionary containing motion correction data
            output_path: Output directory path
            subject_id: Subject identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            xoff = ops.get('xoff', [])
            yoff = ops.get('yoff', [])
            
            if len(xoff) == 0 and len(yoff) == 0:
                self.logger.warning(f"S2P_QC: No motion correction data found for {subject_id}")
                return True  # Not an error, just no data
            
            motion_file = os.path.join(output_path, 'move_offset.h5')
            with h5py.File(motion_file, 'w') as f:
                f['xoff'] = xoff
                f['yoff'] = yoff
            
            self.logger.info(f"S2P_QC: Saved motion correction offsets for {subject_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"S2P_QC: Failed to save motion correction for {subject_id}: {e}")
            return False

    def save_qc_data(self, qc_data: Dict[str, Any], subject_id: str, output_path: str, suite2p_path: str = None) -> bool:
        """
        Save QC-filtered data to output directory matching existing workflow.
        
        Args:
            qc_data: QC-filtered data dictionary
            subject_id: Subject identifier
            output_path: Output directory path
            suite2p_path: Path to original Suite2p plane0 directory (for saving updated iscell)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"S2P_QC: === Saving QC results for {subject_id} ===")
            
            # Create qc_results subdirectory to match existing workflow
            qc_results_path = os.path.join(output_path, 'qc_results')
            os.makedirs(qc_results_path, exist_ok=True)
            self.logger.info(f"S2P_QC: Created output directory: {qc_results_path}")
            
            self.logger.info("S2P_QC: Saving QC-filtered arrays...")
            
            # Save QC-filtered arrays using existing naming convention
            np.save(os.path.join(qc_results_path, 'F.npy'), qc_data['F'])
            np.save(os.path.join(qc_results_path, 'Fneu.npy'), qc_data['Fneu'])
            np.save(os.path.join(qc_results_path, 'stat.npy'), qc_data['stat'])
            np.save(os.path.join(qc_results_path, 'masks.npy'), qc_data['masks'])
            np.save(os.path.join(qc_results_path, 'spks.npy'), qc_data['spks'])
            np.save(os.path.join(qc_results_path, 'kept_idx_orig.npy'), qc_data['kept_idx_orig'])
            np.save(os.path.join(qc_results_path, 'kept_idx_seq.npy'), qc_data['kept_idx_seq'])


            self.logger.info("S2P_QC: Saving ops file...")
            
            # Save ops to main output directory
            np.save(os.path.join(output_path, 'ops.npy'), qc_data['ops'])
            
            # Save updated iscell array back to original Suite2p plane0 directory
            if 'iscell_updated' in qc_data and suite2p_path:
                original_iscell_path = os.path.join(suite2p_path, 'iscell.npy')
                np.save(original_iscell_path, qc_data['iscell_updated'])
                self.logger.info(f"S2P_QC: Updated original iscell.npy at {original_iscell_path}")
            
            self.logger.info("S2P_QC: Saving motion correction data...")
            
            # Save motion correction offsets to match original workflow
            self.save_motion_correction(qc_data['ops'], output_path, subject_id)
            
            self.logger.info("S2P_QC: Saving QC statistics...")
            
            # Save QC statistics for reference
            qc_stats_path = os.path.join(qc_results_path, 'qc_stats.npy')
            np.save(qc_stats_path, qc_data['qc_stats'])
            
            # Extended qc_new artifacts
            if 'qc_metrics_full' in qc_data:
                np.save(os.path.join(qc_results_path, 'qc_metrics.npy'), qc_data['qc_metrics_full'])
            if 'qc_flags' in qc_data:
                np.save(os.path.join(qc_results_path, 'qc_flags.npy'), qc_data['qc_flags'])
            if 'qc_tags' in qc_data:
                np.save(os.path.join(qc_results_path, 'qc_tags.npy'), qc_data['qc_tags'])            
            
            
            
            # Write flags.csv (one row per ROI in original indexing)
            flags = qc_data.get('qc_flags')
            if flags:
                import csv
                csv_path = os.path.join(qc_results_path, 'flags.csv')
                keys = sorted(flags.keys())
                n = len(next(iter(flags.values())))
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['roi_idx'] + keys)
                    for i in range(n):
                        writer.writerow([i] + [int(bool(flags[k][i])) for k in keys])
                self.logger.info(f"S2P_QC: Saved flags CSV -> {csv_path}")
            
            
            
            self.logger.info(f"S2P_QC: === Successfully saved all QC results for {subject_id} ===")
            self.logger.info(f"S2P_QC: Output location: {qc_results_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"S2P_QC: Failed to save QC data for {subject_id}: {e}")
            return False
    
    def process_subject(self, subject_id: str, suite2p_path: str, output_path: str, force: bool = False) -> Dict[str, Any]:
        """
        Process QC filtering for a single subject.
        
        Args:
            subject_id: Subject identifier
            suite2p_path: Path to Suite2p output directory (plane0)
            output_path: Path for QC filtered output
            force: Force reprocessing even if output exists
            
        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f"S2P_QC: ========== Starting QC processing for {subject_id} ==========")
            self.logger.info(f"S2P_QC: Suite2p path: {suite2p_path}")
            self.logger.info(f"S2P_QC: Output path: {output_path}")
            
            # Check if already processed
            qc_stats_file = os.path.join(output_path, 'qc_results', 'qc_stats.npy')
            if os.path.exists(qc_stats_file) and not force:
                self.logger.info(f"S2P_QC: QC data already exists for {subject_id}, skipping (use force=True to reprocess)")
                return {
                    'success': True,
                    'sessions_processed': 1,
                    'message': 'Already processed (skipped)'
                }
            
            self.logger.info("S2P_QC: Loading Suite2p data...")
            
            # Load Suite2p data
            raw_data = self.load_suite2p_data(subject_id, suite2p_path)
            if raw_data is None:
                self.logger.error(f"S2P_QC: Failed to load Suite2p data for {subject_id}")
                return {
                    'success': False,
                    'sessions_processed': 0,
                    'error_message': 'Failed to load Suite2p data'
                }
            
            # Apply QC filters
            qc_data = self.apply_qc_filters(raw_data, subject_id, output_path)
            
            # Save QC-filtered data (pass suite2p_path for iscell update)
            success = self.save_qc_data(qc_data, subject_id, output_path, suite2p_path)
            
            if success:
                self.logger.info(f"S2P_QC: ========== Successfully completed QC processing for {subject_id} ==========")
            else:
                self.logger.error(f"S2P_QC: ========== Failed QC processing for {subject_id} ==========")
            
            return {
                'success': success,
                'sessions_processed': 1 if success else 0,
                'qc_stats': qc_data.get('qc_stats', {}),
                'error_message': None if success else 'Failed to save QC data'
            }
            
        except Exception as e:
            self.logger.error(f"S2P_QC: Processing failed for {subject_id}: {e}")
            return {
                'success': False,
                'sessions_processed': 0,
                'error_message': str(e)
            }
    
    def batch_process(self, force: bool = False) -> Dict[str, bool]:
        """
        Process QC filtering for all subjects in the list.
        
        Args:
            force: Force reprocessing even if output exists
            
        Returns:
            Dictionary mapping subject_id to success status
        """
        results = {}
        
        self.logger.info(f"S2P_QC: Starting batch QC processing for {len(self.subject_list)} subjects")
        
        for subject_id in self.subject_list:
            self.logger.info(f"S2P_QC: Processing {subject_id}...")
            results[subject_id] = self.process_subject(subject_id, force)
        
        # Log summary
        successful = sum(results.values())
        self.logger.info(f"S2P_QC: Batch processing complete: {successful}/{len(self.subject_list)} subjects successful")
        
        return results

