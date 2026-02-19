"""
Suite2p ROI Labeling Module

ROI labeling for Suite2p output data.
Follows the same pattern as other pipeline components.

This handles:
1. Load QC-filtered Suite2p data (from qc_results/)
2. Run cellpose on anatomical channel
3. Compute overlap-based excitatory/inhibitory labeling
4. Save labeling results
"""

import os
import numpy as np
import h5py
import tifffile
import logging
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import csv
from pathlib import Path
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter
from dataclasses import dataclass

# ======================= NEW DATA CLASS (optional clarity) =======================
@dataclass
class MorphEvidence:
    roi_id: int
    geom_tag_soma_like: bool
    overlap_label: str
    overlap_metric: float
    surrogate_label: str
    area_core: int
    area_anat_hit: int
    soma_score: float
    process_score: float
    final_label: str
    final_source: str
    final_confidence: float
    notes: str

class Suite2pLabeling:
    """
    Suite2p ROI Labeling processor following pipeline component pattern.
    
    Handles ROI labeling of Suite2p data with configurable parameters.
    """
    
    def __init__(self, config_manager, subject_list, logger=None):
        """
        Initialize the Suite2p labeling processor.
        
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
        
        # Get configs
        self.experiment_config = self.config_manager.get_experiment_config()
        self.img_cfg = self.experiment_config.get('imaging_preprocessing', {})
        self.lab_cfg = self.img_cfg.get('labeling', {})
        
        self.logger.info("S2P_LABEL: Suite2pLabeling initialized")
    
    
    
    
    
    
    def _load_qc_tags(self, output_path: str) -> Optional[np.ndarray]:
        """Load geometry QC soma tag if present."""
        qc_tags_path = Path(output_path) / 'qc_results' / 'qc_tags.npy'
        if not qc_tags_path.exists():
            self.logger.warning("S2P_LABEL: qc_tags.npy not found (geometry soma tag missing).")
            return None
        try:
            d = np.load(qc_tags_path, allow_pickle=True).item()
            tag = d.get('tag_soma_like', None)
            if tag is None:
                self.logger.warning("S2P_LABEL: 'tag_soma_like' missing in qc_tags.npy.")
            return tag
        except Exception as e:
            self.logger.error(f"S2P_LABEL: Failed to load qc_tags.npy ({e})")
            return None

    def _compute_geometry_score(self, geom_tag_soma_like: np.ndarray) -> np.ndarray:
        """Simple geometry score: 1.0 for soma_like, 0.3 otherwise (placeholder)."""
        return np.where(geom_tag_soma_like, 1.0, 0.3)

    def _build_surrogate_mask(self, mean_func: np.ndarray) -> np.ndarray:
        """
        Lightweight surrogate mask for single channel (no Cellpose):
        threshold robustly -> binary blob map. Not a real soma detector; weak evidence.
        """
        finite = mean_func[np.isfinite(mean_func)]
        if finite.size == 0:
            return np.zeros_like(mean_func, bool)
        lo, hi = np.quantile(finite, [0.20, 0.995])
        if hi <= lo:
            hi = lo + 1
        norm = (mean_func - lo) / (hi - lo)
        surrogate = norm > 0.65  # empirical high‑intensity blob threshold
        return surrogate

    def _surrogate_overlap_labels(self, masks_func: np.ndarray, surrogate_mask: np.ndarray,
                                  core_erosion_px: int = 1, min_overlap_frac: float = 0.25) -> List[str]:
        """
        Produce surrogate 'soma' / 'uncertain' labels based on overlap with heuristic surrogate mask.
        Never labels 'process'.
        """
        n_rois = int(masks_func.max())
        out = ['uncertain'] * n_rois
        if surrogate_mask is None or not surrogate_mask.any():
            return out
        from scipy.ndimage import binary_erosion
        for rid in range(1, n_rois + 1):
            roi_mask = (masks_func == rid)
            if core_erosion_px > 0:
                core = binary_erosion(roi_mask, iterations=core_erosion_px, border_value=0)
                if not core.any():
                    core = roi_mask
            else:
                core = roi_mask
            core_area = core.sum()
            if core_area == 0:
                continue
            hit = core & surrogate_mask
            frac = hit.sum() / core_area
            if frac >= min_overlap_frac:
                out[rid - 1] = 'soma'
        return out

    def _score_and_finalize(self,
                            geom_tag_soma_like: np.ndarray,
                            overlap_dict: Dict[str, Any],
                            surrogate_labels: List[str],
                            cfg_scoring: Dict[str, Any]) -> List[MorphEvidence]:
        """
        Combine evidence into final labels + confidence.
        Strategy:
          - Build component scores
          - Decide final_label with safeguards (confidence & margin thresholds)
          - Source precedence: anatomical > surrogate > geometry
          - If conflict low margin => uncertain
        """
        w_geom = float(cfg_scoring.get('w_geom', 0.6))
        w_anat = float(cfg_scoring.get('w_anat', 1.0))
        w_sur = float(cfg_scoring.get('w_surrogate', 0.4))
        conf_min = float(cfg_scoring.get('decision_conf_min', 0.55))
        margin_min = float(cfg_scoring.get('decision_margin_min', 0.15))
        cfg_anat_guard = bool(cfg_scoring.get('anat_guardrails', False))

        n = len(geom_tag_soma_like)
        overlap_label = overlap_dict['label_morph']               # soma / process / uncertain
        overlap_metric = overlap_dict['overlap_metric']
        area_core = overlap_dict['area_core']
        area_hit = overlap_dict['area_anat_hit']

        geom_score = self._compute_geometry_score(geom_tag_soma_like)

        # Convert overlap label to scores (soft encoding)
        overlap_soma_score = np.where(np.array(overlap_label) == 'soma', 1.0,
                               np.where(np.array(overlap_label) == 'process', 0.0, 0.5))
        overlap_process_score = np.where(np.array(overlap_label) == 'process', 1.0,
                                  np.where(np.array(overlap_label) == 'soma', 0.0, 0.5))

        surrogate_soma = np.array([1.0 if l == 'soma' else 0.0 for l in surrogate_labels])
        surrogate_proc = np.zeros_like(surrogate_soma)  # surrogate never asserts process

        soma_score = w_geom * geom_score + w_anat * overlap_soma_score + w_sur * surrogate_soma
        process_score = w_geom * (1 - geom_score) + w_anat * overlap_process_score + w_sur * surrogate_proc

        evidences: List[MorphEvidence] = []
        for i in range(n):
            ss = soma_score[i]
            ps = process_score[i]
            total = ss + ps + 1e-9
            conf = max(ss, ps) / total
            margin = abs(ss - ps)
            # Preliminary label
            prelim = 'soma' if ss > ps else 'process'
            source = 'anat' if overlap_label[i] in ('soma', 'process') else (
                     'surrogate' if surrogate_labels[i] == 'soma' else (
                     'geometry'))
            notes_parts = []
            if overlap_label[i] == 'uncertain':
                notes_parts.append('anat=uncertain')
            if surrogate_labels[i] == 'soma':
                notes_parts.append('surrogate_soma')
            if geom_tag_soma_like[i]:
                notes_parts.append('geom_soma_like')
            
            # --- HARD GUARDRAILS ---
            if cfg_anat_guard:
                # If anatomy is confident "process", do not let other evidence overrule it.
                if overlap_label[i] == 'process':
                    final_label = 'process'
                    final_source = 'anat'
                    notes_parts.append('hard_guardrail:anat_process')
                    evidences.append(MorphEvidence(
                        roi_id=i+1, geom_tag_soma_like=bool(geom_tag_soma_like[i]),
                        overlap_label=overlap_label[i], overlap_metric=float(overlap_metric[i]),
                        surrogate_label=surrogate_labels[i], area_core=int(area_core[i]), area_anat_hit=int(area_hit[i]),
                        soma_score=float(ss), process_score=float(ps),
                        final_label=final_label, final_source=final_source,
                        final_confidence=1.0, notes=";".join(notes_parts)
                    ))
                    continue

                # If anatomy is uncertain, do not let geometry/surrogate upgrade to soma.
                if overlap_label[i] == 'uncertain':
                    # geometry/surrogate can’t promote to soma; keep uncertain unless process dominates
                    # (optional: keep 'process' only if ps >> ss, else 'uncertain')
                    if ps > ss and (ps - ss) >= margin_min:
                        final_label = 'process'
                        final_source = 'low_conf'  # or 'anat_uncertain'
                        notes_parts.append('hard_guardrail:anat_uncertain->process_if_strong')
                    else:
                        final_label = 'uncertain'
                        final_source = 'anat_uncertain'
                        notes_parts.append('hard_guardrail:anat_uncertain_hold')
                    evidences.append(MorphEvidence(
                        roi_id=i+1, geom_tag_soma_like=bool(geom_tag_soma_like[i]),
                        overlap_label=overlap_label[i], overlap_metric=float(overlap_metric[i]),
                        surrogate_label=surrogate_labels[i], area_core=int(area_core[i]), area_anat_hit=int(area_hit[i]),
                        soma_score=float(ss), process_score=float(ps),
                        final_label=final_label, final_source=final_source,
                        final_confidence=float(max(ss, ps) / (ss + ps + 1e-9)), notes=";".join(notes_parts)
                    ))
                    continue
                # --- END GUARDRAILS ---

            # Confidence gating
            if (conf < conf_min) or (margin < margin_min):
                final_label = 'uncertain'
                final_source = 'low_conf'
                notes_parts.append(f"low_conf(conf={conf:.2f},margin={margin:.2f})")
            else:
                final_label = prelim
                final_source = source
            evidences.append(MorphEvidence(
                roi_id=i + 1,
                geom_tag_soma_like=bool(geom_tag_soma_like[i]),
                overlap_label=overlap_label[i],
                overlap_metric=float(overlap_metric[i]),
                surrogate_label=surrogate_labels[i],
                area_core=int(area_core[i]),
                area_anat_hit=int(area_hit[i]),
                soma_score=float(ss),
                process_score=float(ps),
                final_label=final_label,
                final_source=final_source,
                final_confidence=float(conf),
                notes=";".join(notes_parts)
            ))
        # After evidence build add quick stats
        soma_geom = int(np.sum(geom_tag_soma_like))
        soma_final = int(np.sum([e.final_label == 'soma' for e in evidences]))
        self.logger.info(f"S2P_LABEL: Geometry soma_like={soma_geom} -> Final soma={soma_final} "
                         f"(Delta={soma_final - soma_geom})")
        low_conf_frac = np.mean([e.final_source == 'low_conf' for e in evidences])
        if low_conf_frac > 0.30:
            self.logger.warning(f"S2P_LABEL: High low_conf fraction {low_conf_frac:.2%}; consider tuning scoring thresholds.")            
        return evidences

    def _write_roi_labels_csv_multi(self, output_path: str, evidences: List[MorphEvidence]):
        """
        Overwrite roi_labels.csv with multi-evidence columns (provenance & scores).
        """
        csv_path = Path(output_path) / 'qc_results' / 'roi_labels.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        rows = []
        for ev in evidences:
            rows.append(dict(
                roi_id=ev.roi_id,
                geom_tag_soma_like=int(ev.geom_tag_soma_like),
                overlap_label=ev.overlap_label,
                overlap_metric=ev.overlap_metric,
                surrogate_label=ev.surrogate_label,
                area_core=ev.area_core,
                area_anat_hit=ev.area_anat_hit,
                soma_score=ev.soma_score,
                process_score=ev.process_score,
                final_label=ev.final_label,
                final_source=ev.final_source,
                final_confidence=ev.final_confidence,
                evidence_notes=ev.notes
            ))
        df_new = pd.DataFrame(rows)

        # Merge if existing (preserve any unrelated older columns like idx_orig, qc flags)
        if csv_path.exists():
            try:
                old = pd.read_csv(csv_path)
                if 'roi_id' in old.columns:
                    drop_cols = [c for c in df_new.columns if c in old.columns and c != 'roi_id']
                    old = old.drop(columns=drop_cols)
                    out = pd.merge(old, df_new, on='roi_id', how='outer').sort_values('roi_id')
                else:
                    out = df_new
            except Exception as e:
                self.logger.warning(f"S2P_LABEL: CSV merge failed ({e}); overwriting.")
                out = df_new
        else:
            out = df_new
        out.to_csv(csv_path, index=False)
        self.logger.info(f"S2P_LABEL: Wrote multi-evidence roi_labels.csv (n={len(rows)})")

        # Quick sanity log
        counts = out['final_label'].value_counts().to_dict()
        self.logger.info(f"S2P_LABEL: Final label counts {counts}")
    
    
    
    
    
    
    
    
    
    
    
    
    def load_labeling_data(self, subject_id: str, suite2p_path: str, output_path: str) -> Optional[Dict[str, Any]]:
        """
        Load data needed for ROI labeling.
        
        Args:
            subject_id: Subject identifier
            suite2p_path: Path to Suite2p output directory (plane0)
            output_path: Path to QC output
            
        Returns:
            Dictionary containing loaded data or None if failed
        """
        try:
            data = {}
            
            # Load ops file for metadata
            ops_path = os.path.join(output_path, 'ops.npy')
            if os.path.exists(ops_path):
                data['ops'] = np.load(ops_path, allow_pickle=True).item()
                self.logger.info(f"S2P_LABEL: Loaded ops for {subject_id}")
            else:
                self.logger.error(f"S2P_LABEL: Missing ops.npy at {ops_path}")
                return None
            
            # Load QC-filtered masks from QC results (these are already filtered but need to be full-size)
            # TODO: PIPELINE REVIEW NEEDED
            # Current implementation assumes QC saves masks to qc_results/masks.npy
            # But original standalone may load from suite2p/plane0/ directly
            # Need to verify what original QC actually saves and where labeling loads from
            masks_path = os.path.join(output_path, 'qc_results', 'masks.npy')
            if os.path.exists(masks_path):
                data['masks_qc_filtered'] = np.load(masks_path, allow_pickle=True)
                self.logger.info(f"S2P_LABEL: Loaded QC-filtered masks for {subject_id}")
            else:
                self.logger.error(f"S2P_LABEL: Missing QC masks at {masks_path}")
                return None
            
            # Load fluorescence traces if dual channel
            if data['ops'].get('nchannels', 1) == 2:
                f_ch1_path = os.path.join(suite2p_path, 'F.npy')
                f_ch2_path = os.path.join(suite2p_path, 'F_chan2.npy')
                
                if os.path.exists(f_ch1_path) and os.path.exists(f_ch2_path):
                    data['fluo_ch1'] = np.load(f_ch1_path, allow_pickle=True)
                    data['fluo_ch2'] = np.load(f_ch2_path, allow_pickle=True)
                    self.logger.info(f"S2P_LABEL: Loaded dual-channel traces for {subject_id}")
                else:
                    self.logger.warning(f"S2P_LABEL: Missing dual-channel traces for {subject_id}")
            
            self.logger.info(f"S2P_LABEL: Successfully loaded labeling data for {subject_id}")
            return data
            
        except Exception as e:
            self.logger.error(f"S2P_LABEL: Failed to load labeling data for {subject_id}: {e}")
            return None
    
    # def extract_mean_images(self, ops: Dict) -> Tuple[np.ndarray, ...]:
    #     """
    #     Extract and crop mean images from ops to match original standalone behavior.
        
    #     Args:
    #         ops: Suite2p ops dictionary
            
    #     Returns:
    #         Tuple of (masks_func_cropped, mean_func, max_func, mean_anat)
    #     """
    #     # Get crop coordinates
    #     x1, x2 = ops['xrange'][0], ops['xrange'][1]
    #     y1, y2 = ops['yrange'][0], ops['yrange'][1]
        
    #     # Extract functional channel images (these should be cropped)
    #     mean_func = ops['meanImg'][y1:y2, x1:x2]
    #     max_func = ops['max_proj']  # This should already be cropped size
        
    #     # Extract anatomical channel image if available
    #     if ops.get('nchannels', 1) == 2 and 'meanImg_chan2' in ops:
    #         mean_anat = ops['meanImg_chan2'][y1:y2, x1:x2]
    #     else:
    #         mean_anat = None
        
    #     return mean_func, max_func, mean_anat
    
     # ------------------------------------------------------------------
    # CONFIG-AWARE IMAGE EXTRACTION
    # ------------------------------------------------------------------
    def extract_mean_images(self, ops: Dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Pick functional (mean_func), max_func, anatomical (mean_anat) per config.
        Functional background: prefer meanImgE then meanImg.
        Anatomical selection controlled by labeling.anat_image (default 'auto'):
          'auto'  : if dual channel & meanImg_chan2 present use it else fallback to functional
          'chan2' : require meanImg_chan2
          'meanImgE' / 'meanImg' : explicit ops key
          'path:<path>' : load npy / tif external file
        Optional preprocessing labeling.anat_preproc:
          bleed_alpha (float) subtract alpha * mean_func
          normalize (bool) scale 0..1 robust (1%-99.5%)
          gaussian_sigma (float or None) blur
        """
        cfg_img = self.lab_cfg.get('anat_image', 'auto')
        preproc = self.lab_cfg.get('anat_preproc', {})
        bleed_alpha = float(preproc.get('bleed_alpha', 0.0))
        do_norm = bool(preproc.get('normalize', True))
        gauss_sigma = preproc.get('gaussian_sigma', None)

        # Crop coords
        x1, x2 = ops['xrange'][0], ops['xrange'][1]
        y1, y2 = ops['yrange'][0], ops['yrange'][1]

        # Functional base
        if 'meanImgE' in ops and ops['meanImgE'] is not None:
            mean_func_full = ops['meanImgE']
        else:
            mean_func_full = ops['meanImg']
        mean_func = mean_func_full[y1:y2, x1:x2]
        max_func = ops.get('max_proj', mean_func)

        def _robust_norm(img):
            finite = img[np.isfinite(img)]
            if finite.size:
                lo, hi = np.quantile(finite, [0.01, 0.995])
                if hi <= lo:
                    hi = lo + 1
                img = np.clip((img - lo) / (hi - lo), 0, 1)
            return img

        mean_anat = None
        def _load_external(path_str: str):
            p = Path(path_str)
            if not p.exists():
                self.logger.warning(f"S2P_LABEL: External anat image missing: {p}")
                return None
            if p.suffix.lower() in ('.tif', '.tiff'):
                import tifffile
                arr = tifffile.imread(str(p))
            elif p.suffix.lower() == '.npy':
                arr = np.load(p)
            else:
                self.logger.warning(f"S2P_LABEL: Unsupported anat external format: {p.suffix}")
                return None
            if arr.ndim > 2:
                arr = arr[0]
            return arr

        if cfg_img == 'auto':
            if ops.get('nchannels', 1) == 2 and 'meanImg_chan2' in ops:
                mean_anat_full = ops['meanImg_chan2']
            else:
                mean_anat_full = mean_func_full
        elif cfg_img == 'chan2':
            if 'meanImg_chan2' not in ops:
                self.logger.warning("S2P_LABEL: anat_image 'chan2' requested but not found; falling back to functional.")
                mean_anat_full = mean_func_full
            else:
                mean_anat_full = ops['meanImg_chan2']
        elif cfg_img in ('meanImgE', 'meanImg'):
            mean_anat_full = ops.get(cfg_img, mean_func_full)
        elif cfg_img.startswith('path:'):
            ext_path = cfg_img.split(':', 1)[1]
            mean_anat_full = _load_external(ext_path) or mean_func_full
        else:
            self.logger.warning(f"S2P_LABEL: Unknown anat_image '{cfg_img}', using functional.")
            mean_anat_full = mean_func_full

        mean_anat = mean_anat_full[y1:y2, x1:x2].astype(float)

        # Preproc
        if bleed_alpha > 0:
            mean_anat = mean_anat - bleed_alpha * mean_func
        if do_norm:
            mean_anat = _robust_norm(mean_anat)
            mean_func = _robust_norm(mean_func)
        if gauss_sigma not in (None, 0):
            try:
                mean_anat = gaussian_filter(mean_anat, gauss_sigma)
            except Exception:
                pass

        return mean_func.astype(np.float32), max_func.astype(np.float32), mean_anat.astype(np.float32)
   
    # ------------------------------------------------------------------
    # CLASSIFY ROIs BY OVERLAP WITH ANATOMICAL MASKS (soma/process/uncertain)
    # ------------------------------------------------------------------
    def classify_by_overlap(self,
                            masks_func: np.ndarray,
                            masks_anat: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Overlap classification.
        Added:
          - per-object IoU (best matching Cellpose object) if overlap.mode == 'best_object'
          - logs erosion fallbacks
          - returns overlap_object_id (0 if none) + label_morph_codes
        """
        n_rois = int(masks_func.max())
        if n_rois == 0:
            return dict(label_morph=[], label_morph_codes=np.array([], int),
                        overlap_metric=np.array([], float),
                        area_core=np.array([], int),
                        area_anat_hit=np.array([], int),
                        overlap_object_id=np.array([], int),
                        erosion_fallbacks=0)

        cfg = self.lab_cfg.get('overlap', {})
        metric_type = cfg.get('metric', 'iou')  # 'iou', 'frac_core', 'frac_object'
        mode = cfg.get('mode', 'global')  # 'global' or 'best_object'
        soma_thr = float(cfg.get('soma_iou_min', 0.35))
        proc_thr = float(cfg.get('process_iou_max', 0.15))
        uncertain_between = bool(cfg.get('uncertain_between', True))
        core_erosion_px = int(cfg.get('core_erosion_px', 1))
        anat_dilate_px = int(cfg.get('anat_dilate_px', 0))

        erosion_fallbacks = 0
        
        if mode == 'best_object' and anat_dilate_px > 0:
            self.logger.info("S2P_LABEL: Forcing anat_dilate_px=0 in best_object mode (avoid ID conflicts).")
            anat_dilate_px = 0        

        if masks_anat is None or masks_anat.max() == 0:
            self.logger.warning("S2P_LABEL: No anatomical mask available; all ROIs uncertain.")
            morph = ['uncertain'] * n_rois
            return dict(label_morph=morph,
                        label_morph_codes=np.zeros(n_rois, int),
                        overlap_metric=np.zeros(n_rois, float),
                        area_core=np.zeros(n_rois, int),
                        area_anat_hit=np.zeros(n_rois, int),
                        overlap_object_id=np.zeros(n_rois, int),
                        erosion_fallbacks=0)

        anat_lbl = masks_anat.astype(int)
        if anat_dilate_px > 0:
            se = np.ones((2 * anat_dilate_px + 1, 2 * anat_dilate_px + 1), bool)
            # dilate each object separately to avoid merging IDs
            dilated = np.zeros_like(anat_lbl)
            for oid in np.unique(anat_lbl):
                if oid == 0:
                    continue
                mask_o = (anat_lbl == oid)
                mask_o = binary_dilation(mask_o, structure=se)
                dilated[mask_o] = oid
            anat_lbl = dilated

        # Precompute object areas (per-object IoU)
        if mode == 'best_object':
            # obj_ids = np.unique(anat_lbl)
            # obj_ids = obj_ids[obj_ids > 0]
            # obj_area = {oid: int((anat_lbl == oid).sum()) for oid in obj_ids}
            # Vectorized object area via bincount
            flat = anat_lbl.ravel()
            obj_area_arr = np.bincount(flat)            
        else:
            anat_bin = (anat_lbl > 0)
            anat_area_global = anat_bin.sum()
            if anat_area_global / anat_bin.size > 0.6:
                self.logger.info(f"S2P_LABEL: Global anat coverage {anat_area_global/anat_bin.size:.2f} may compress IoU range.")

        overlap_metric = np.zeros(n_rois, float)
        area_core = np.zeros(n_rois, int)
        area_hit = np.zeros(n_rois, int)
        labels_txt = []
        codes = np.zeros(n_rois, int)
        best_obj = np.zeros(n_rois, int)
        
        # Prepare extra per-object metrics
        matched_object_area = np.zeros(n_rois, int)
        inter_area = np.zeros(n_rois, int)
        frac_core_metric = np.zeros(n_rois, float)
        frac_object_metric = np.zeros(n_rois, float)        

        for rid in range(1, n_rois + 1):
            roi_mask = (masks_func == rid)
            if core_erosion_px > 0:
                core = binary_erosion(roi_mask, iterations=core_erosion_px, border_value=0)
                if not core.any():
                    core = roi_mask
                    erosion_fallbacks += 1
            else:
                core = roi_mask
            core_area = core.sum()
            area_core[rid - 1] = core_area
            if core_area == 0:
                labels_txt.append('uncertain')
                continue

            if mode == 'best_object':
                # Evaluate each overlapping object
                candidate_ids = np.unique(anat_lbl[core])
                candidate_ids = candidate_ids[candidate_ids > 0]
                if candidate_ids.size == 0:
                    labels_txt.append('uncertain')
                    continue
                best_val = 0.0
                best_id = 0
                best_inter = 0
                best_obj_area = 0
                for oid in candidate_ids:
                    inter = np.logical_and(core, anat_lbl == oid).sum()
                    if inter == 0:
                        continue
                    obj_area = obj_area_arr[oid] if oid < obj_area_arr.size else int((anat_lbl == oid).sum())
                    union = core_area + obj_area - inter
                    iou = inter / (union + 1e-9)
                    if iou > best_val:
                        best_val = iou
                        best_id = oid
                        best_inter = inter
                        best_obj_area = obj_area
                overlap_metric[rid - 1] = best_val
                area_hit[rid - 1] = best_inter
                best_obj[rid - 1] = best_id
                matched_object_area[rid - 1] = best_obj_area
                inter_area[rid - 1] = best_inter
                if best_obj_area > 0:
                    frac_core_metric[rid - 1] = best_inter / (core_area + 1e-9)
                    frac_object_metric[rid - 1] = best_inter / (best_obj_area + 1e-9)                
                # candidate_ids = np.unique(anat_lbl[core])
                # candidate_ids = candidate_ids[candidate_ids > 0]
                # if candidate_ids.size == 0:
                #     labels_txt.append('uncertain')
                #     continue
                # best_iou = 0.0
                # best_id = 0
                # hit_area_best = 0
                # for oid in candidate_ids:
                #     obj_mask = (anat_lbl == oid)
                #     hit = core & obj_mask
                #     inter = hit.sum()
                #     if inter == 0:
                #         continue
                #     union = core_area + obj_area[oid] - inter
                #     iou = inter / union
                #     if iou > best_iou:
                #         best_iou = iou
                #         best_id = oid
                #         hit_area_best = inter
                # overlap_metric[rid - 1] = best_iou
                # area_hit[rid - 1] = hit_area_best
                # best_obj[rid - 1] = best_id
            else:  # global union
                hit = core & anat_bin
                inter = hit.sum()
                area_hit[rid - 1] = inter
                union = core_area + anat_area_global - inter
                overlap_metric[rid - 1] = inter / (union + 1e-9)

            mval = overlap_metric[rid - 1]
            if mval >= soma_thr:
                labels_txt.append('soma'); codes[rid - 1] = 1
            elif mval <= proc_thr:
                labels_txt.append('process'); codes[rid - 1] = 2
            else:
                if uncertain_between:
                    labels_txt.append('uncertain'); codes[rid - 1] = 0
                else:
                    # force to nearer threshold
                    if (mval - proc_thr) < (soma_thr - mval):
                        labels_txt.append('process'); codes[rid - 1] = 2
                    else:
                        labels_txt.append('soma'); codes[rid - 1] = 1

        if mode == 'best_object':
            no_cand = np.sum((best_obj == 0) & (area_core > 0))
            self.logger.info(f"S2P_LABEL: best_object: no-candidate ROIs={no_cand} / {n_rois}")
            q = np.quantile(overlap_metric[overlap_metric>0], [0.1,0.5,0.9]) if np.any(overlap_metric>0) else [0,0,0]
            self.logger.info(f"S2P_LABEL: overlap_metric quantiles (10/50/90%) {q}")

        if erosion_fallbacks:
            self.logger.info(f"S2P_LABEL: Erosion fallback (core collapsed) for {erosion_fallbacks} / {n_rois} ROIs.")
       
        return dict(
            label_morph=labels_txt,
            label_morph_codes=codes,
            overlap_metric=overlap_metric,
            area_core=area_core,
            area_anat_hit=area_hit,
            overlap_object_id=best_obj,
            matched_object_area=matched_object_area,
            inter_area=inter_area,
            frac_core=frac_core_metric,
            frac_object=frac_object_metric,
            erosion_fallbacks=erosion_fallbacks
        )


        # return dict(label_morph=labels_txt,
        #             label_morph_codes=codes,
        #             overlap_metric=overlap_metric,
        #             area_core=area_core,
        #             area_anat_hit=area_hit,
        #             overlap_object_id=best_obj,
        #             erosion_fallbacks=erosion_fallbacks)    
    # def classify_by_overlap(self,
    #                         masks_func: np.ndarray,
    #                         masks_anat: Optional[np.ndarray]) -> Dict[str, Any]:
    #     """
    #     Compute morphological labels by overlap metric between eroded functional ROI core
    #     and (optionally dilated) anatomical mask.

    #     Returns dict with:
    #       label_morph (str list)
    #       label_morph_codes (int array: 1=soma,2=process,0=uncertain)
    #       overlap_metric (float array)
    #       area_core, area_anat_hit (int arrays)
    #     """
    #     n_rois = int(masks_func.max())
    #     if n_rois == 0:
    #         return dict(label_morph=[], label_morph_codes=np.array([], int),
    #                     overlap_metric=np.array([], float),
    #                     area_core=np.array([], int),
    #                     area_anat_hit=np.array([], int))

    #     cfg = self.lab_cfg.get('overlap', {})
    #     method = cfg.get('metric', 'iou')  # 'iou' or 'frac_core'
    #     soma_thr = float(cfg.get('soma_iou_min', 0.35))
    #     proc_thr = float(cfg.get('process_iou_max', 0.15))
    #     uncertain_between = bool(cfg.get('uncertain_between', True))
    #     core_erosion_px = int(cfg.get('core_erosion_px', 1))
    #     anat_dilate_px = int(cfg.get('anat_dilate_px', 0))

    #     # Prepare anatomical binary
    #     if masks_anat is None or masks_anat.max() == 0:
    #         self.logger.warning("S2P_LABEL: No anatomical mask available; all ROIs uncertain.")
    #         morph = ['uncertain'] * n_rois
    #         return dict(label_morph=morph,
    #                     label_morph_codes=np.zeros(n_rois, int),
    #                     overlap_metric=np.zeros(n_rois, float),
    #                     area_core=np.zeros(n_rois, int),
    #                     area_anat_hit=np.zeros(n_rois, int))

    #     anat_bin = (masks_anat > 0)
    #     if anat_dilate_px > 0:
    #         se = np.ones((2 * anat_dilate_px + 1, 2 * anat_dilate_px + 1), bool)
    #         anat_bin = binary_dilation(anat_bin, structure=se)

    #     overlap_metric = np.zeros(n_rois, float)
    #     area_core = np.zeros(n_rois, int)
    #     area_hit = np.zeros(n_rois, int)
    #     labels_txt = []
    #     codes = np.zeros(n_rois, int)  # 1=soma 2=process 0=uncertain

    #     for rid in range(1, n_rois + 1):
    #         roi_mask = (masks_func == rid)
    #         if core_erosion_px > 0:
    #             core = binary_erosion(roi_mask, iterations=core_erosion_px, border_value=0)
    #             if not core.any():  # fallback to original if fully eroded
    #                 core = roi_mask
    #         else:
    #             core = roi_mask
    #         core_area = core.sum()
    #         area_core[rid - 1] = core_area
    #         if core_area == 0:
    #             labels_txt.append('uncertain')
    #             continue
    #         hit = core & anat_bin
    #         hit_area = hit.sum()
    #         area_hit[rid - 1] = hit_area
    #         if method == 'frac_core':
    #             metric = hit_area / core_area
    #         else:  # IoU
    #             union = core | anat_bin
    #             metric = hit_area / (union.sum() + 1e-9)
    #         overlap_metric[rid - 1] = metric

    #         if metric >= soma_thr:
    #             labels_txt.append('soma')
    #             codes[rid - 1] = 1
    #         elif metric <= proc_thr:
    #             labels_txt.append('process')
    #             codes[rid - 1] = 2
    #         else:
    #             labels_txt.append('uncertain')
    #             codes[rid - 1] = 0
    #             if not uncertain_between:
    #                 # Force binary (assign to closer threshold side)
    #                 if (metric - proc_thr) < (soma_thr - metric):
    #                     labels_txt[-1] = 'process'; codes[rid - 1] = 2
    #                 else:
    #                     labels_txt[-1] = 'soma'; codes[rid - 1] = 1

    #     return dict(label_morph=labels_txt,
    #                 label_morph_codes=codes,
    #                 overlap_metric=overlap_metric,
    #                 area_core=area_core,
    #                 area_anat_hit=area_hit)

    # ------------------------------------------------------------------
    # OPTIONAL DUPLICATE ANNOTATION (stub / lightweight)
    # ------------------------------------------------------------------
    def find_duplicates(self, F: np.ndarray, max_dist: float = 5.0, corr_min: float = 0.92) -> np.ndarray:
        """
        Placeholder: returns -1 for all (no duplicates).
        Hook point for future duplicate detection (spatial + correlation).
        """
        return -1 * np.ones(F.shape[0], dtype=int)
    
    def prepare_masks_for_labeling(self, masks_qc_filtered: np.ndarray, ops: Dict) -> np.ndarray:
        """
        Prepare masks for labeling by ensuring they match the original full-size format.
        This matches the standalone get_mask() function behavior.
        
        Args:
            masks_qc_filtered: QC-filtered masks from qc_results
            ops: Suite2p ops dictionary
            
        Returns:
            Full-size masks array matching original dimensions
        """
        # If masks are already full size, crop them like the original
        if masks_qc_filtered.shape == (ops['Ly'], ops['Lx']):
            # Get crop coordinates
            x1, x2 = ops['xrange'][0], ops['xrange'][1]
            y1, y2 = ops['yrange'][0], ops['yrange'][1]
            
            # Crop to functional area (matching original get_mask behavior)
            masks_func = masks_qc_filtered[y1:y2, x1:x2]
            
            self.logger.info(f"S2P_LABEL: Cropped masks from {masks_qc_filtered.shape} to {masks_func.shape}")
            return masks_func
        else:
            # Masks are already cropped size - use as is
            self.logger.info(f"S2P_LABEL: Using pre-cropped masks of shape {masks_qc_filtered.shape}")
            return masks_qc_filtered

    # ------------------------------------------------------------------
    # UPDATED run_cellpose WITH AREA FILTERING
    # ------------------------------------------------------------------
    def run_cellpose(self, mean_anat: np.ndarray, output_path: str, diameter: float = 6, 
                    flow_threshold: float = 0.5) -> np.ndarray:
        """
        Run cellpose on anatomical channel image.
        
        Args:
            mean_anat: Mean anatomical channel image
            output_path: Output directory path
            diameter: Cellpose diameter parameter
            flow_threshold: Cellpose flow threshold
            
        Returns:
            Cellpose masks
        """
        try:
            # Import cellpose modules
            from cellpose import models, io
            
            # Create cellpose output directory
            cellpose_dir = os.path.join(output_path, 'cellpose')
            os.makedirs(cellpose_dir, exist_ok=True)
            
            # Save mean anatomical image
            tifffile.imwrite(os.path.join(cellpose_dir, 'mean_anat.tif'), mean_anat)
            
            self.logger.info(f"S2P_LABEL: Running cellpose with diameter={diameter}")
            
            # Run cellpose
            model = models.Cellpose(model_type="cyto3")
            masks_anat, flows, styles, diams = model.eval(
                mean_anat,
                diameter=diameter,
                flow_threshold=flow_threshold
            )
            
            # Save cellpose results
            io.masks_flows_to_seg(
                images=mean_anat,
                masks=masks_anat,
                flows=flows,
                file_names=os.path.join(cellpose_dir, 'mean_anat'),
                diams=diameter
            )
            
            self.logger.info(f"S2P_LABEL: Cellpose completed, found {np.max(masks_anat)} ROIs")
            # Area filtering
            morph_cfg = self.lab_cfg.get('morphology', {})
            area_low, area_high = morph_cfg.get('soma_area_px', [20, 8000])
            if masks_anat.max() > 0:
                ids = np.unique(masks_anat); ids = ids[ids > 0]
                for rid in ids:
                    sz = (masks_anat == rid).sum()
                    if sz < area_low or sz > area_high:
                        masks_anat[masks_anat == rid] = 0
            return masks_anat.astype(np.int32)
            
        except ImportError:
            self.logger.error("S2P_LABEL: Cellpose not available - please install cellpose")
            return np.zeros_like(mean_anat)
        except Exception as e:
            self.logger.error(f"S2P_LABEL: Cellpose failed: {e}")
            return np.zeros_like(mean_anat)
    
    
    # ------------------------------------------------------------------
    # CSV AUGMENT / WRITE
    # ------------------------------------------------------------------
    # def _write_roi_labels_csv(self, output_path: str, morph: Dict[str, Any], dup_with: np.ndarray):
    #     """
    #     Append / create roi_labels.csv with new columns.
    #     """
    #     csv_path = Path(output_path) / 'qc_results' / 'roi_labels.csv'
    #     csv_path.parent.mkdir(parents=True, exist_ok=True)

    #     n = morph['label_morph_codes'].size
    #     morph_cols = ['label_morph', 'label_morph_code',
    #                   'label_overlap_metric', 'area_core',
    #                   'area_anat_hit', 'dup_with']

    #     # Build new morphology DataFrame
    #     try:
    #         import pandas as pd
    #         new_df = pd.DataFrame({
    #             'roi_id': np.arange(1, n + 1, dtype=int),
    #             'label_morph': morph['label_morph'],
    #             'label_morph_code': morph['label_morph_codes'],
    #             'label_overlap_metric': morph['overlap_metric'],
    #             'area_core': morph['area_core'],
    #             'area_anat_hit': morph['area_anat_hit'],
    #             'dup_with': dup_with.astype(int)
    #         })

    #         if csv_path.exists():
    #             old = pd.read_csv(csv_path)
    #             if 'roi_id' not in old.columns:
    #                 self.logger.warning("S2P_LABEL: Existing roi_labels.csv lacks roi_id; overwriting fully.")
    #                 out_df = new_df
    #             else:
    #                 # Drop any previous morphology columns
    #                 drop_cols = [c for c in old.columns if c in morph_cols]
    #                 if drop_cols:
    #                     old = old.drop(columns=drop_cols)
    #                 # Outer merge to keep any legacy rows (unlikely mismatch)
    #                 out_df = pd.merge(old, new_df, on='roi_id', how='outer')
    #                 out_df = out_df.sort_values('roi_id')
    #             out_df.to_csv(csv_path, index=False)
    #             self.logger.info("S2P_LABEL: roi_labels.csv overwritten with updated morphology labels.")
    #         else:
    #             new_df.to_csv(csv_path, index=False)
    #             self.logger.info("S2P_LABEL: roi_labels.csv created with morphology labels.")
    #     except Exception as e:
    #         # Fallback minimal writer (no merge preservation)
    #         self.logger.warning(f"S2P_LABEL: Pandas write failed ({e}); writing basic CSV.")
    #         header = ['roi_id'] + morph_cols
    #         with open(csv_path, 'w', newline='') as f:
    #             w = csv.writer(f)
    #             w.writerow(header)
    #             for i in range(n):
    #                 w.writerow([
    #                     i + 1,
    #                     morph['label_morph'][i],
    #                     int(morph['label_morph_codes'][i]),
    #                     float(morph['overlap_metric'][i]),
    #                     int(morph['area_core'][i]),
    #                     int(morph['area_anat_hit'][i]),
    #                     int(dup_with[i])
    #                 ])
    #     self.logger.info(f"S2P_LABEL: ROI labels CSV written to {csv_path}")
        
        
    # ------------------------------------------------------------------
    # QUICKCHECK FIGURES
    # ------------------------------------------------------------------    
    def _quickcheck_outputs(self, output_path: str,
                            mean_func: np.ndarray,
                            masks_func: np.ndarray,
                            overlap_morph: Dict[str, Any],
                            final_labels: Optional[List[str]] = None):
        """
        Write quickcheck figures.
        Always writes overlap-based overlay/hist.
        If final_labels provided (multi-evidence), also writes final_overlay.png using those labels.
        """       
        qc_cfg = self.lab_cfg.get('outputs', {})
        if not qc_cfg.get('write_quickcheck', False):
            return
        import matplotlib.pyplot as plt
        from scipy.ndimage import binary_erosion, binary_dilation

        ov_cfg = qc_cfg.get('overlay', {})
        plow = float(ov_cfg.get('plow', 0.01))
        phigh = float(ov_cfg.get('phigh', 0.995))
        gamma = float(ov_cfg.get('gamma', 1.0))
        dim = float(ov_cfg.get('dim', 0.6))
        edge_w = int(ov_cfg.get('edge_width', 2))
        fill_alpha = float(ov_cfg.get('fill_alpha', 0.0))
        show_legend = bool(ov_cfg.get('show_legend', True))

        out_dir = os.path.join(output_path, 'qc_results', 'quickcheck')
        os.makedirs(out_dir, exist_ok=True)

        def _prep_bg(img):
            finite = img[np.isfinite(img)]
            if finite.size:
                lo, hi = np.quantile(finite, [plow, phigh])
                if hi <= lo:
                    hi = lo + 1
                bg = np.clip((img - lo) / (hi - lo), 0, 1)
            else:
                bg = img.copy()
            if gamma != 1.0:
                bg = np.power(bg, gamma)
            return bg * dim

        def _render(labels: List[str], fname: str, title: str):
            bg = _prep_bg(mean_func)
            rgb = np.dstack([bg]*3)
            cmap_map = {
                'soma': (0.05, 0.75, 1.00),
                'process': (1.00, 0.30, 0.30),
                'uncertain': (0.75, 0.75, 0.75)
            }
            for rid in range(1, int(masks_func.max()) + 1):
                roi_mask = (masks_func == rid)
                if not roi_mask.any():
                    continue
                label = labels[rid - 1]
                col = np.array(cmap_map.get(label, (1, 1, 1)))
                if fill_alpha > 0:
                    for c in range(3):
                        rgb[..., c][roi_mask] = (1 - fill_alpha) * rgb[..., c][roi_mask] + fill_alpha * col[c]
                core = binary_erosion(roi_mask, iterations=1) if edge_w > 1 else binary_erosion(roi_mask, iterations=0)
                edge = roi_mask ^ core
                if edge_w > 1:
                    edge = binary_dilation(edge, iterations=edge_w - 1)
                halo = binary_dilation(edge, iterations=1) & (~edge)
                rgb[halo] = np.clip(rgb[halo] * 0.2, 0, 1)
                for c in range(3):
                    rgb[..., c][edge] = col[c]
            plt.figure(figsize=(5, 5))
            plt.imshow(rgb, interpolation='nearest')
            plt.axis('off')
            plt.title(title)
            if show_legend:
                from matplotlib.patches import Patch
                patches = [Patch(color=cmap_map[k],
                                 label=f"{k} ({(np.array(labels)==k).sum()})")
                           for k in ['soma','process','uncertain']]
                leg = plt.legend(handles=patches, fontsize=7, loc='lower left', frameon=False)
                                # Force legend text white
                for txt in leg.get_texts():
                    txt.set_color('white')
            plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
            plt.close()

        # Overlap (initial)
        _render(overlap_morph['label_morph'], 'overlap_overlay.png', 'Morph overlap classification')

        # Histogram (overlap metric)
        metric = overlap_morph['overlap_metric']
        soma_thr = self.lab_cfg.get('overlap', {}).get('soma_iou_min', 0.35)
        proc_thr = self.lab_cfg.get('overlap', {}).get('process_iou_max', 0.15)
        plt.figure(figsize=(5, 3))
        plt.hist(metric, bins=40, color='0.65', edgecolor='0.3')
        plt.axvline(soma_thr, color='tab:blue', linestyle='--', label='soma_thr')
        plt.axvline(proc_thr, color='tab:red', linestyle='--', label='process_thr')
        plt.xlabel('Overlap metric')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'overlap_hist.png'), dpi=150)
        plt.close()

        # Final (multi-evidence) labels if provided
        if final_labels is not None:
            _render(final_labels, 'final_overlay.png', 'Final multi-evidence classification')
            self.logger.info("S2P_LABEL: Quickcheck final_overlay.png written.")

        self.logger.info("S2P_LABEL: Quickcheck outputs written (overlap + hist{})."
                         .format(" + final" if final_labels is not None else ""))
   
    # def _quickcheck_outputs(self, output_path: str,
    #                         mean_func: np.ndarray,
    #                         masks_func: np.ndarray,
    #                         morph: Dict[str, Any],
    #                         final_labels: Optional[List[str]] = None):
    #     qc_cfg = self.lab_cfg.get('outputs', {})
    #     if not qc_cfg.get('write_quickcheck', False):
    #         return
    #     import matplotlib.pyplot as plt
    #     from scipy.ndimage import binary_erosion, binary_dilation

    #     ov_cfg = qc_cfg.get('overlay', {})
    #     plow = float(ov_cfg.get('plow', 0.01))
    #     phigh = float(ov_cfg.get('phigh', 0.995))
    #     gamma = float(ov_cfg.get('gamma', 1.0))
    #     dim = float(ov_cfg.get('dim', 0.6))            # multiply background
    #     edge_w = int(ov_cfg.get('edge_width', 2))
    #     fill_alpha = float(ov_cfg.get('fill_alpha', 0.0))
    #     show_legend = bool(ov_cfg.get('show_legend', True))

    #     out_dir = os.path.join(output_path, 'qc_results', 'quickcheck')
    #     os.makedirs(out_dir, exist_ok=True)

    #     # Robust rescale background
    #     finite = mean_func[np.isfinite(mean_func)]
    #     if finite.size:
    #         lo, hi = np.quantile(finite, [plow, phigh])
    #         if hi <= lo:
    #             hi = lo + 1
    #         bg = np.clip((mean_func - lo) / (hi - lo), 0, 1)
    #     else:
    #         bg = mean_func.copy()
    #     if gamma != 1.0:
    #         bg = np.power(bg, gamma)
    #     bg = (bg * dim)

    #     rgb = np.dstack([bg]*3)

    #     # Colors
    #     cmap_map = {
    #         'soma': (0.05, 0.75, 1.00),
    #         'process': (1.00, 0.30, 0.30),
    #         'uncertain': (0.75, 0.75, 0.75)
    #     }

    #     # Optional fill (light alpha), then colored outline with dark border for contrast
    #     for rid in range(1, int(masks_func.max()) + 1):
    #         roi_mask = (masks_func == rid)
    #         if not roi_mask.any():
    #             continue
    #         label = morph['label_morph'][rid - 1]
    #         col = np.array(cmap_map.get(label, (1, 1, 1)))

    #         if fill_alpha > 0:
    #             for c in range(3):
    #                 rgb[..., c][roi_mask] = (1 - fill_alpha) * rgb[..., c][roi_mask] + fill_alpha * col[c]

    #         core = binary_erosion(roi_mask, iterations=1) if edge_w > 1 else binary_erosion(roi_mask, iterations=0)
    #         edge = roi_mask ^ core
    #         # Thicken edge if requested
    #         if edge_w > 1:
    #             edge = binary_dilation(edge, iterations=edge_w - 1)
    #         # Dark halo (draw black first)
    #         halo = binary_dilation(edge, iterations=1) & (~edge)
    #         rgb[halo] = np.clip(rgb[halo] * 0.2, 0, 1)
    #         # Color edge
    #         for c in range(3):
    #             rgb[..., c][edge] = col[c]

    #     plt.figure(figsize=(5, 5))
    #     plt.imshow(rgb, interpolation='nearest')
    #     plt.axis('off')
    #     plt.title("Morph overlap classification")
    #     if show_legend:
    #         from matplotlib.patches import Patch
    #         patches = [Patch(color=cmap_map[k], label=f"{k} ({(np.array(morph['label_morph'])==k).sum()})")
    #                     for k in ['soma','process','uncertain']]
    #         plt.legend(handles=patches, fontsize=7, loc='lower left', frameon=False)
    #     plt.savefig(os.path.join(out_dir, 'overlap_overlay.png'), dpi=150, bbox_inches='tight')
    #     plt.close()

    #     # Histogram (unchanged except thresholds lines)
    #     metric = morph['overlap_metric']
    #     soma_thr = self.lab_cfg.get('overlap', {}).get('soma_iou_min', 0.35)
    #     proc_thr = self.lab_cfg.get('overlap', {}).get('process_iou_max', 0.15)
    #     plt.figure(figsize=(5, 3))
    #     plt.hist(metric, bins=40, color='0.65', edgecolor='0.3')
    #     plt.axvline(soma_thr, color=cmap_map['soma'], linestyle='--', label='soma_thr')
    #     plt.axvline(proc_thr, color=cmap_map['process'], linestyle='--', label='process_thr')
    #     plt.xlabel('Overlap metric')
    #     plt.ylabel('Count')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(out_dir, 'overlap_hist.png'), dpi=150)
    #     plt.close()
    #     self.logger.info("S2P_LABEL: Quickcheck outputs written (enhanced contrast).")

    # def _quickcheck_outputs(self, output_path: str,
    #                         mean_func: np.ndarray,
    #                         masks_func: np.ndarray,
    #                         morph: Dict[str, Any]):
    #     qc_cfg = self.lab_cfg.get('outputs', {})
    #     if not qc_cfg.get('write_quickcheck', False):
    #         return
    #     import matplotlib.pyplot as plt
    #     from matplotlib import colors

    #     # out_dir = Path(output_path) / 'quickcheck'
    #     out_dir = os.path.join(output_path, 'qc_results', 'quickcheck')
    #     os.makedirs(out_dir, exist_ok=True)

    #     # Overlay
    #     cmap_map = {'soma': (0, 0.8, 1), 'process': (1, 0.2, 0.2), 'uncertain': (0.9, 0.9, 0.9)}
    #     rgb = np.dstack([mean_func] * 3)
    #     rgb = (rgb / (rgb.max() + 1e-9)) ** 0.8
    #     for rid in range(1, int(masks_func.max()) + 1):
    #         roi_mask = (masks_func == rid)
    #         if not roi_mask.any():
    #             continue
    #         # outline
    #         er = binary_erosion(roi_mask)
    #         edge = roi_mask ^ er
    #         label = morph['label_morph'][rid - 1]
    #         color = cmap_map.get(label, (1, 1, 1))
    #         for c in range(3):
    #             rgb[..., c][edge] = color[c]
    #     plt.figure(figsize=(5, 5))
    #     plt.imshow(rgb)
    #     plt.axis('off')
    #     plt.title("Morph overlap classification")
    #     # plt.savefig(out_dir / 'overlap_overlay.png', dpi=150, bbox_inches='tight')
    #     plt.savefig(os.path.join(out_dir, 'overlap_overlay.png'), dpi=150, bbox_inches='tight')
    #     plt.close()

    #     # Histogram
    #     metric = morph['overlap_metric']
    #     soma_thr = self.lab_cfg.get('overlap', {}).get('soma_iou_min', 0.35)
    #     proc_thr = self.lab_cfg.get('overlap', {}).get('process_iou_max', 0.15)
    #     plt.figure(figsize=(5, 3))
    #     plt.hist(metric, bins=40, color='gray', alpha=0.8)
    #     plt.axvline(soma_thr, color='blue', linestyle='--', label='soma_thr')
    #     plt.axvline(proc_thr, color='red', linestyle='--', label='process_thr')
    #     plt.xlabel('Overlap metric')
    #     plt.ylabel('Count')
    #     plt.legend()
    #     plt.tight_layout()
    #     # plt.savefig(out_dir / 'overlap_hist.png', dpi=150)
    #     plt.savefig(os.path.join(out_dir, 'overlap_hist.png'), dpi=150)
    #     plt.close()
    #     self.logger.info("S2P_LABEL: Quickcheck outputs written.")            
    
    # def compute_overlap_labels(self, masks_func: np.ndarray, masks_anat: np.ndarray, 
    #                          thres1: float = 0.2, thres2: float = 0.9) -> np.ndarray:
    #     """
    #     Compute excitatory/inhibitory labels based on overlap with anatomical masks.
        
    #     Args:
    #         masks_func: Functional channel masks (QC-filtered)
    #         masks_anat: Anatomical channel masks (from cellpose)
    #         thres1: Lower threshold for excitatory classification
    #         thres2: Upper threshold for inhibitory classification
            
    #     Returns:
    #         Array of labels: -1 (excitatory), 0 (unlabeled), 1 (inhibitory)
    #     """
    #     try:
    #         # Get unique ROI IDs
    #         anat_roi_ids = np.unique(masks_anat)[1:]  # Exclude background (0)
    #         func_roi_ids = np.unique(masks_func)[1:]   # Exclude background (0)
            
    #         if len(anat_roi_ids) == 0:
    #             self.logger.warning("S2P_LABEL: No anatomical ROIs found")
    #             return -1 * np.ones(len(func_roi_ids), dtype=np.int32)
            
    #         # Create 3D array of anatomical masks
    #         masks_3d = np.zeros((len(anat_roi_ids), masks_anat.shape[0], masks_anat.shape[1]))
    #         for i, roi_id in enumerate(anat_roi_ids):
    #             masks_3d[i] = (masks_anat == roi_id).astype(int)
            
    #         self.logger.info(f"S2P_LABEL: Computing overlaps for {len(func_roi_ids)} functional ROIs")
            
    #         # Compute overlap probabilities
    #         prob = []
    #         for func_roi_id in tqdm(func_roi_ids, desc="Computing overlaps"):
    #             # Extract functional ROI mask
    #             roi_mask_func = (masks_func == func_roi_id).astype(np.int32)
                
    #             # Tile functional mask to match anatomical ROIs
    #             roi_masks_tile = np.tile(
    #                 np.expand_dims(roi_mask_func, 0),
    #                 (len(anat_roi_ids), 1, 1)
    #             )
                
    #             # Compute overlaps with all anatomical ROIs
    #             overlap = (roi_masks_tile * masks_3d).reshape(len(anat_roi_ids), -1)
    #             overlap = np.sum(overlap, axis=1)
                
    #             # Find best matching anatomical ROI
    #             best_anat_idx = np.argmax(overlap)
    #             roi_mask_anat = (masks_anat == anat_roi_ids[best_anat_idx]).astype(np.int32)
                
    #             # Compute overlap probability (relative to both ROIs)
    #             overlap_prob = np.max([
    #                 np.max(overlap) / (np.sum(roi_mask_func) + 1e-10),
    #                 np.max(overlap) / (np.sum(roi_mask_anat) + 1e-10)
    #             ])
    #             prob.append(overlap_prob)
            
    #         # Apply thresholds to classify ROIs
    #         prob = np.array(prob)
    #         labels = np.zeros_like(prob, dtype=np.int32)
            
    #         # Excitatory (low overlap)
    #         labels[prob < thres1] = -1
    #         # Inhibitory (high overlap)  
    #         labels[prob > thres2] = 1
    #         # Unlabeled (medium overlap) remains 0
            
    #         n_excitatory = np.sum(labels == -1)
    #         n_inhibitory = np.sum(labels == 1)
    #         n_unlabeled = np.sum(labels == 0)
            
    #         self.logger.info(f"S2P_LABEL: Labeling results: {n_excitatory} excitatory, "
    #                        f"{n_inhibitory} inhibitory, {n_unlabeled} unlabeled")
            
    #         return labels
            
    #     except Exception as e:
    #         self.logger.error(f"S2P_LABEL: Failed to compute overlap labels: {e}")
    #         return -1 * np.ones(len(np.unique(masks_func)[1:]), dtype=np.int32)
    
    def save_labeling_results(self, output_path: str, masks_func: np.ndarray, masks_anat: Optional[np.ndarray],
                              mean_func: np.ndarray, max_func: np.ndarray, mean_anat: Optional[np.ndarray],
                              final_codes: np.ndarray, ops: Dict[str, Any],
                              overlap_codes: Optional[np.ndarray] = None,
                              geom_tag_snapshot: Optional[np.ndarray] = None,
                              config_subset: Optional[Dict[str, Any]] = None,
                              overlap_dict: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save labeling results to HDF5 file.
        
        Args:
            output_path: Output directory path
            masks_func: Functional channel masks
            masks_anat: Anatomical channel masks (None for single channel)
            mean_func: Mean functional image
            max_func: Max projection functional image
            mean_anat: Mean anatomical image (None for single channel)
            labels: ROI labels array
            ops: Suite2p ops dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import h5py, json, os
            h5_path = Path(output_path) / 'qc_results' / 'masks.h5'
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(h5_path, 'w') as h5:
                h5.create_dataset('masks_func', data=masks_func.astype(np.int32), compression='gzip')
                if masks_anat is not None:
                    h5.create_dataset('masks_anat', data=masks_anat.astype(np.int32), compression='gzip')
                h5.create_dataset('label_morph_final_codes', data=final_codes.astype(np.int16))
                
                # Overlap (initial) codes + per-object metrics
                if overlap_codes is not None:
                    h5.create_dataset('label_morph_overlap_codes', data=overlap_codes.astype(np.int16))
                if overlap_dict is not None:
                                    # Core fields (only write if lengths match)
                                    n_rois = final_codes.size
                                    def _safe_arr(key, default_val=0):
                                        arr = overlap_dict.get(key, None)
                                        if arr is None or len(arr) != n_rois:
                                            return np.full(n_rois, default_val, dtype=np.int32)
                                        return arr
                                    h5.create_dataset('overlap_object_id', data=_safe_arr('overlap_object_id'))
                                    for k in ('matched_object_area','inter_area'):
                                        h5.create_dataset(f'overlap_{k}', data=_safe_arr(k))
                                    # float metrics (frac_core / frac_object / overlap_metric)
                                    for k in ('frac_core','frac_object','overlap_metric'):
                                        arr = overlap_dict.get(k, None)
                                        if arr is not None and len(arr) == n_rois:
                                            h5.create_dataset(f'overlap_{k}', data=arr.astype(np.float32))
                                            
                if geom_tag_snapshot is not None:
                    h5.create_dataset('geom_tag_soma_like', data=geom_tag_snapshot.astype(np.uint8))
                h5.create_dataset('mean_func', data=mean_func.astype(np.float32), compression='gzip')
                h5.create_dataset('max_func', data=max_func.astype(np.float32), compression='gzip')
                if mean_anat is not None:
                    h5.create_dataset('mean_anat', data=mean_anat.astype(np.float32), compression='gzip')
                # Config snapshot as attributes
                if config_subset:
                    cfg_json = json.dumps(config_subset, indent=2)
                    h5.attrs['labeling_config_snapshot'] = cfg_json
            # Also write plain text snapshot for quick diff
            if config_subset:
                snap_txt = Path(output_path) / 'qc_results' / 'labeling_config_snapshot.txt'
                with open(snap_txt, 'w', encoding='utf-8') as f:
                    f.write(cfg_json)
            self.logger.info(f"S2P_LABEL: Saved masks.h5 with final + overlap + geometry snapshots.")
            return True
        except Exception as e:
            self.logger.error(f"S2P_LABEL: Failed to save labeling HDF5 ({e})")
            return False
    
    
    # def save_labeling_results(self, output_path: str, masks_func: np.ndarray, masks_anat: np.ndarray,
    #                         mean_func: np.ndarray, max_func: np.ndarray, mean_anat: np.ndarray,
    #                         labels: np.ndarray, ops: Dict) -> bool:
    #     """
    #     Save labeling results to HDF5 file.
        
    #     Args:
    #         output_path: Output directory path
    #         masks_func: Functional channel masks
    #         masks_anat: Anatomical channel masks (None for single channel)
    #         mean_func: Mean functional image
    #         max_func: Max projection functional image
    #         mean_anat: Mean anatomical image (None for single channel)
    #         labels: ROI labels array
    #         ops: Suite2p ops dictionary
            
    #     Returns:
    #         True if successful, False otherwise
    #     """
    #     try:
    #         masks_file = os.path.join(output_path, 'qc_results', 'masks.h5')
            
    #         with h5py.File(masks_file, 'w') as f:
    #             f['labels'] = labels
    #             f['masks_func'] = masks_func
    #             f['mean_func'] = mean_func
    #             f['max_func'] = max_func
                
    #             if ops.get('nchannels', 1) == 2 and mean_anat is not None:
    #                 f['mean_anat'] = mean_anat
    #                 f['masks_anat'] = masks_anat
            
    #         self.logger.info(f"S2P_LABEL: Saved labeling results to {masks_file}")
    #         return True
            
    #     except Exception as e:
    #         self.logger.error(f"S2P_LABEL: Failed to save labeling results: {e}")
    #         return False

    def process_subject(self, subject_id: str, suite2p_path: str, output_path: str, force: bool = False) -> Dict[str, Any]:
        """
        Process ROI labeling for a single subject.
        
        Args:
            subject_id: Subject identifier
            suite2p_path: Path to Suite2p output directory (plane0)
            output_path: Path for labeling output
            force: Force reprocessing even if output exists
            
        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f"S2P_LABEL: ========== Starting labeling processing for {subject_id} ==========")
            self.logger.info(f"S2P_LABEL: Suite2p path: {suite2p_path}")
            self.logger.info(f"S2P_LABEL: Output path: {output_path}")
            
            force = True
            
            # Check if already processed
            labels_file = os.path.join(output_path, 'qc_results', 'masks.h5')
            if os.path.exists(labels_file) and not force:
                self.logger.info(f"S2P_LABEL: Labeling data already exists for {subject_id}, skipping (use force=True to reprocess)")
                return {
                    'success': True,
                    'sessions_processed': 1,
                    'message': 'Already processed (skipped)'
                }
            
            # Load required data
            data = self.load_labeling_data(subject_id, suite2p_path, output_path)
            if data is None:
                return {
                    'success': False,
                    'sessions_processed': 0,
                    'error_message': 'Failed to load labeling data'
                }
            
            # Extract images and prepare masks
            mean_func, max_func, mean_anat = self.extract_mean_images(data['ops'])
            masks_func = self.prepare_masks_for_labeling(data['masks_qc_filtered'], data['ops'])
            
            # Check if masks exist
            if np.max(masks_func) == 0:
                self.logger.error(f"S2P_LABEL: No functional masks found for {subject_id}")
                return {
                    'success': False,
                    'sessions_processed': 0,
                    'error_message': 'No functional masks found'
                }
            
            # === Decide anatomical availability & run Cellpose early (BEFORE overlap) ===
            if data['ops'].get('nchannels', 1) == 2 and mean_anat is not None:
                if self.lab_cfg.get('run_cellpose', True):
                    diameter = self.lab_cfg.get('cellpose', {}).get('diameter', 6)
                    flow_threshold = self.lab_cfg.get('cellpose', {}).get('flow_threshold', 0.5)
                    self.logger.info("S2P_LABEL: Dual-channel -> running Cellpose.")
                    masks_anat = self.run_cellpose(mean_anat, output_path, diameter, flow_threshold)
                else:
                    self.logger.info("S2P_LABEL: Dual-channel but run_cellpose disabled; no anatomical mask.")
                    masks_anat = np.zeros_like(mean_func, dtype=np.int32)
            else:
                self.logger.info("S2P_LABEL: Single-channel (no anatomical mask).")
                masks_anat = None            
            
            # === LOAD GEOMETRY TAG FROM QC ===
            geom_tag = self._load_qc_tags(output_path)
            if geom_tag is None:
                geom_tag = np.zeros(int(masks_func.max()), bool)
                self.logger.warning("S2P_LABEL: Geometry soma tag missing -> default all False.")

            # === ANATOMICAL / CELLPOSE PATH ===
            overlap_dict = self.classify_by_overlap(masks_func, masks_anat)

            # === SINGLE-CHANNEL SURROGATE (optional) ===
            ov_cfg = self.lab_cfg.get('overlap', {})
            surrogate_cfg = self.lab_cfg.get('surrogate', {})
            use_qc_tags_single = bool(ov_cfg.get('use_qc_tags_when_single_channel', True))
            surrogate_enabled = bool(surrogate_cfg.get('run_cellpose_on_single_channel', False))
           
            if masks_anat is None:
                if surrogate_enabled:
                    self.logger.info("S2P_LABEL: Building surrogate mask (single-channel mode).")
                    surrogate_mask = self._build_surrogate_mask(mean_func)
                    surrogate_labels = self._surrogate_overlap_labels(
                        masks_func,
                        surrogate_mask,
                        core_erosion_px=int(ov_cfg.get('core_erosion_px', 1)),
                        min_overlap_frac=float(surrogate_cfg.get('min_overlap_frac', 0.25))
                    )
                elif use_qc_tags_single:
                    # Promote geometry tag to surrogate 'soma'
                    surrogate_labels = ['soma' if g else 'uncertain' for g in geom_tag]
                    self.logger.info("S2P_LABEL: Using geometry tag as surrogate labels (single-channel).")
                else:
                    surrogate_labels = ['uncertain'] * int(masks_func.max())
            else:
                surrogate_labels = ['uncertain'] * int(masks_func.max())  # not used in dual channel

            if masks_anat is None and any(l == 'soma' for l in surrogate_labels):
                self.logger.info(f"S2P_LABEL: Surrogate soma count (single-channel) = "
                                 f"{sum(l=='soma' for l in surrogate_labels)}")

            # Contiguous label assertion
            uniq = np.unique(masks_func)
            if masks_func.max() > 0:
                expected = np.arange(1, masks_func.max() + 1)
                if not np.array_equal(uniq[uniq > 0], expected):
                    self.logger.warning("S2P_LABEL: Non-contiguous ROI labels detected; downstream indexing may misalign.")


            # === COMBINE & SCORE ===
            scoring_cfg = self.lab_cfg.get('scoring', {})
            if masks_anat is None and use_qc_tags_single and not surrogate_enabled:
                if 'w_surrogate' in scoring_cfg and scoring_cfg.get('w_surrogate',0) != 0:
                    self.logger.info("S2P_LABEL: Zeroing w_surrogate to avoid geometry double count.")
                scoring_cfg['w_surrogate'] = 0.0            
            evidences = self._score_and_finalize(
                geom_tag_soma_like=geom_tag[:int(masks_func.max())],
                overlap_dict=overlap_dict,
                surrogate_labels=surrogate_labels,
                cfg_scoring=scoring_cfg
            )
            
            # Evidence source table
            src_counts = {}
            for e in evidences:
                src_counts[e.final_source] = src_counts.get(e.final_source, 0) + 1
            self.logger.info(f"S2P_LABEL: Evidence sources counts: {src_counts}")            

            # === Duplicate annotation (optional) ===
            dup_with = self.find_duplicates(
                F=np.load(Path(output_path) / 'qc_results' / 'F.npy')
                  if (Path(output_path)/'qc_results'/'F.npy').exists()
                  else np.zeros((int(masks_func.max()), 10))
            )
            # For now we only log duplicates count (stub returns -1).
            n_dups = np.sum(dup_with >= 0)
            if n_dups:
                self.logger.info(f"S2P_LABEL: Detected {n_dups} duplicate ROIs (annotation only).")


            # === BUILD label_morph_codes FOR HDF5 (retain original overlap codes or final?) ===
            final_codes_map = {'uncertain': 0, 'soma': 1, 'process': 2}
            final_codes = np.array([final_codes_map[e.final_label] for e in evidences], dtype=int)

            cfg_subset = {
                'overlap': {k: v for k, v in self.lab_cfg.get('overlap', {}).items()
                            if k in ('metric','mode','soma_iou_min','process_iou_max','core_erosion_px','anat_dilate_px','uncertain_between')},
                'scoring': self.lab_cfg.get('scoring', {}),
                'surrogate': self.lab_cfg.get('surrogate', {})
            }
            success = self.save_labeling_results(
                output_path, masks_func, masks_anat,
                mean_func, max_func, mean_anat,
                final_codes, data['ops'],
                overlap_codes=overlap_dict['label_morph_codes'],
                geom_tag_snapshot=geom_tag[:int(masks_func.max())],
                config_subset=cfg_subset,
                overlap_dict=overlap_dict
            )

            # # === SAVE HDF5 (store final codes) ===
            # success = self.save_labeling_results(
            #     output_path, masks_func, masks_anat,
            #     mean_func, max_func, mean_anat,
            #     final_codes, data['ops']
            # )

            # === WRITE CSV (multi evidence) ===
            self._write_roi_labels_csv_multi(output_path, evidences)

            # === QUICKCHECK (reuse overlap for now) ===
            # Replace morph dict to reuse existing visualization (still based on overlap stage)
            # === QUICKCHECK (now pass final labels) ===
            overlap_for_plot = dict(
                label_morph=overlap_dict['label_morph'],
                label_morph_codes=overlap_dict['label_morph_codes'],
                overlap_metric=overlap_dict['overlap_metric'],
                area_core=overlap_dict['area_core'],
                area_anat_hit=overlap_dict['area_anat_hit']
            )
            self._quickcheck_outputs(
                output_path,
                mean_func,
                masks_func,
                overlap_for_plot,
                final_labels=[e.final_label for e in evidences]
            )

            # === LOG SUMMARY ===
            final_labels = [e.final_label for e in evidences]
            n_soma = final_labels.count('soma')
            n_proc = final_labels.count('process')
            n_unc = final_labels.count('uncertain')
            self.logger.info(f"S2P_LABEL: Final multi-evidence stats: soma={n_soma} process={n_proc} uncertain={n_unc}")
            

            
            # # Decide anatomical availability
            # if data['ops'].get('nchannels', 1) == 2 and mean_anat is not None:
            #     # Run cellpose unless config disables
            #     if self.lab_cfg.get('run_cellpose', True):
            #         diameter = self.lab_cfg.get('cellpose', {}).get('diameter', 6)
            #         flow_threshold = self.lab_cfg.get('cellpose', {}).get('flow_threshold', 0.5)
            #         masks_anat = self.run_cellpose(mean_anat, output_path, diameter, flow_threshold)
            #     else:
            #         masks_anat = np.zeros_like(mean_func, dtype=np.int32)
            # else:
            #     masks_anat = None            
            
            
            # # Morph classification
            # morph = self.classify_by_overlap(masks_func, masks_anat)

            # # Duplicate annotation (optional)
            # dup_with = self.find_duplicates(
            #     F=np.load(Path(output_path) / 'qc_results' / 'F.npy') if (Path(output_path)/'qc_results'/'F.npy').exists() else
            #       np.zeros((int(masks_func.max()), 10)),
            # )

            # # Save main HDF5 (include codes + metrics)
            # success = self.save_labeling_results(
            #     output_path, masks_func, masks_anat,
            #     mean_func, max_func, mean_anat,
            #     morph['label_morph_codes'], data['ops']
            # )
            # # Write CSV
            # self._write_roi_labels_csv(output_path, morph, dup_with)

            # # Quickcheck
            # self._quickcheck_outputs(output_path, mean_func, masks_func, morph)            
            
            # # Handle single vs dual channel
            # if data['ops'].get('nchannels', 1) == 1:
            #     self.logger.info("S2P_LABEL: Single channel recording - labeling all ROIs as excitatory")
            #     labels = -1 * np.ones(int(np.max(masks_func)), dtype=np.int32)
            #     masks_anat = None
                
            # else:
            #     # Get labeling parameters from config
            #     experiment_config = self.config_manager.get_experiment_config()
            #     imaging_config = experiment_config.get('imaging_preprocessing', {})
            #     qc_params = imaging_config.get('quality_control', {})
            #     diameter = qc_params.get('diameter', 6)
                
            #     self.logger.info("S2P_LABEL: Dual channel recording - running cellpose and overlap analysis")
                
            #     # Run cellpose on anatomical channel
            #     masks_anat = self.run_cellpose(mean_anat, output_path, diameter)
                
            #     # Compute overlap-based labels
            #     labels = self.compute_overlap_labels(masks_func, masks_anat)
            
            # # Save results (use the processed masks_func, not the original)
            # success = self.save_labeling_results(
            #     output_path, masks_func, masks_anat,
            #     mean_func, max_func, mean_anat, labels, data['ops']
            # )
            
            if success:
                self.logger.info(f"S2P_LABEL: ========== Successfully completed labeling for {subject_id} ==========")
            else:
                self.logger.error(f"S2P_LABEL: ========== Failed labeling for {subject_id} ==========")
            
            # return {
            #     'success': success,
            #     'sessions_processed': 1 if success else 0,
            #     'labeling_stats': {
            #         'n_rois': len(labels),
            #         'n_excitatory': np.sum(labels == -1),
            #         'n_inhibitory': np.sum(labels == 1),
            #         'n_unlabeled': np.sum(labels == 0),
            #         'channel_mode': 'dual' if data['ops'].get('nchannels', 1) == 2 else 'single'
            #     },
            #     'error_message': None if success else 'Failed to save labeling results'
            # }
            # return {
            #     'success': success,
            #     'sessions_processed': 1 if success else 0,
            #     'labeling_stats': {
            #         'n_rois': morph['label_morph_codes'].size,
            #         'n_soma': int((morph['label_morph_codes'] == 1).sum()),
            #         'n_process': int((morph['label_morph_codes'] == 2).sum()),
            #         'n_uncertain': int((morph['label_morph_codes'] == 0).sum()),
            #         'channel_mode': 'dual' if data['ops'].get('nchannels', 1) == 2 else 'single'
            #     },
            #     'error_message': None if success else 'Failed to save labeling results'
            
            return {
                'success': success,
                'sessions_processed': 1 if success else 0,
                'labeling_stats': {
                    'n_rois': len(evidences),
                    'n_soma': n_soma,
                    'n_process': n_proc,
                    'n_uncertain': n_unc,
                    'channel_mode': 'dual' if data['ops'].get('nchannels', 1) == 2 else 'single'
                },
                'error_message': None if success else 'Failed to save labeling results'                        
            }            
            
        except Exception as e:
            self.logger.error(f"S2P_LABEL: Processing failed for {subject_id}: {e}")
            return {
                'success': False,
                'sessions_processed': 0,
                'error_message': str(e)
            }
