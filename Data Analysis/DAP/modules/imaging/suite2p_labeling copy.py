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
        
        self.logger.info("S2P_LABEL: Suite2pLabeling initialized")
    
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
    
    def extract_mean_images(self, ops: Dict) -> Tuple[np.ndarray, ...]:
        """
        Extract and crop mean images from ops to match original standalone behavior.
        
        Args:
            ops: Suite2p ops dictionary
            
        Returns:
            Tuple of (masks_func_cropped, mean_func, max_func, mean_anat)
        """
        # Get crop coordinates
        x1, x2 = ops['xrange'][0], ops['xrange'][1]
        y1, y2 = ops['yrange'][0], ops['yrange'][1]
        
        # Extract functional channel images (these should be cropped)
        mean_func = ops['meanImg'][y1:y2, x1:x2]
        max_func = ops['max_proj']  # This should already be cropped size
        
        # Extract anatomical channel image if available
        if ops.get('nchannels', 1) == 2 and 'meanImg_chan2' in ops:
            mean_anat = ops['meanImg_chan2'][y1:y2, x1:x2]
        else:
            mean_anat = None
        
        return mean_func, max_func, mean_anat
    
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
            return masks_anat
            
        except ImportError:
            self.logger.error("S2P_LABEL: Cellpose not available - please install cellpose")
            return np.zeros_like(mean_anat)
        except Exception as e:
            self.logger.error(f"S2P_LABEL: Cellpose failed: {e}")
            return np.zeros_like(mean_anat)
    
    def compute_overlap_labels(self, masks_func: np.ndarray, masks_anat: np.ndarray, 
                             thres1: float = 0.2, thres2: float = 0.9) -> np.ndarray:
        """
        Compute excitatory/inhibitory labels based on overlap with anatomical masks.
        
        Args:
            masks_func: Functional channel masks (QC-filtered)
            masks_anat: Anatomical channel masks (from cellpose)
            thres1: Lower threshold for excitatory classification
            thres2: Upper threshold for inhibitory classification
            
        Returns:
            Array of labels: -1 (excitatory), 0 (unlabeled), 1 (inhibitory)
        """
        try:
            # Get unique ROI IDs
            anat_roi_ids = np.unique(masks_anat)[1:]  # Exclude background (0)
            func_roi_ids = np.unique(masks_func)[1:]   # Exclude background (0)
            
            if len(anat_roi_ids) == 0:
                self.logger.warning("S2P_LABEL: No anatomical ROIs found")
                return -1 * np.ones(len(func_roi_ids), dtype=np.int32)
            
            # Create 3D array of anatomical masks
            masks_3d = np.zeros((len(anat_roi_ids), masks_anat.shape[0], masks_anat.shape[1]))
            for i, roi_id in enumerate(anat_roi_ids):
                masks_3d[i] = (masks_anat == roi_id).astype(int)
            
            self.logger.info(f"S2P_LABEL: Computing overlaps for {len(func_roi_ids)} functional ROIs")
            
            # Compute overlap probabilities
            prob = []
            for func_roi_id in tqdm(func_roi_ids, desc="Computing overlaps"):
                # Extract functional ROI mask
                roi_mask_func = (masks_func == func_roi_id).astype(np.int32)
                
                # Tile functional mask to match anatomical ROIs
                roi_masks_tile = np.tile(
                    np.expand_dims(roi_mask_func, 0),
                    (len(anat_roi_ids), 1, 1)
                )
                
                # Compute overlaps with all anatomical ROIs
                overlap = (roi_masks_tile * masks_3d).reshape(len(anat_roi_ids), -1)
                overlap = np.sum(overlap, axis=1)
                
                # Find best matching anatomical ROI
                best_anat_idx = np.argmax(overlap)
                roi_mask_anat = (masks_anat == anat_roi_ids[best_anat_idx]).astype(np.int32)
                
                # Compute overlap probability (relative to both ROIs)
                overlap_prob = np.max([
                    np.max(overlap) / (np.sum(roi_mask_func) + 1e-10),
                    np.max(overlap) / (np.sum(roi_mask_anat) + 1e-10)
                ])
                prob.append(overlap_prob)
            
            # Apply thresholds to classify ROIs
            prob = np.array(prob)
            labels = np.zeros_like(prob, dtype=np.int32)
            
            # Excitatory (low overlap)
            labels[prob < thres1] = -1
            # Inhibitory (high overlap)  
            labels[prob > thres2] = 1
            # Unlabeled (medium overlap) remains 0
            
            n_excitatory = np.sum(labels == -1)
            n_inhibitory = np.sum(labels == 1)
            n_unlabeled = np.sum(labels == 0)
            
            self.logger.info(f"S2P_LABEL: Labeling results: {n_excitatory} excitatory, "
                           f"{n_inhibitory} inhibitory, {n_unlabeled} unlabeled")
            
            return labels
            
        except Exception as e:
            self.logger.error(f"S2P_LABEL: Failed to compute overlap labels: {e}")
            return -1 * np.ones(len(np.unique(masks_func)[1:]), dtype=np.int32)
    
    def save_labeling_results(self, output_path: str, masks_func: np.ndarray, masks_anat: np.ndarray,
                            mean_func: np.ndarray, max_func: np.ndarray, mean_anat: np.ndarray,
                            labels: np.ndarray, ops: Dict) -> bool:
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
            masks_file = os.path.join(output_path, 'masks.h5')
            
            with h5py.File(masks_file, 'w') as f:
                f['labels'] = labels
                f['masks_func'] = masks_func
                f['mean_func'] = mean_func
                f['max_func'] = max_func
                
                if ops.get('nchannels', 1) == 2 and mean_anat is not None:
                    f['mean_anat'] = mean_anat
                    f['masks_anat'] = masks_anat
            
            self.logger.info(f"S2P_LABEL: Saved labeling results to {masks_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"S2P_LABEL: Failed to save labeling results: {e}")
            return False

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
            
            # Check if already processed
            labels_file = os.path.join(output_path, 'masks.h5')
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
            
            # Handle single vs dual channel
            if data['ops'].get('nchannels', 1) == 1:
                self.logger.info("S2P_LABEL: Single channel recording - labeling all ROIs as excitatory")
                labels = -1 * np.ones(int(np.max(masks_func)), dtype=np.int32)
                masks_anat = None
                
            else:
                # Get labeling parameters from config
                experiment_config = self.config_manager.get_experiment_config()
                imaging_config = experiment_config.get('imaging_preprocessing', {})
                qc_params = imaging_config.get('quality_control', {})
                diameter = qc_params.get('diameter', 6)
                
                self.logger.info("S2P_LABEL: Dual channel recording - running cellpose and overlap analysis")
                
                # Run cellpose on anatomical channel
                masks_anat = self.run_cellpose(mean_anat, output_path, diameter)
                
                # Compute overlap-based labels
                labels = self.compute_overlap_labels(masks_func, masks_anat)
            
            # Save results (use the processed masks_func, not the original)
            success = self.save_labeling_results(
                output_path, masks_func, masks_anat,
                mean_func, max_func, mean_anat, labels, data['ops']
            )
            
            if success:
                self.logger.info(f"S2P_LABEL: ========== Successfully completed labeling for {subject_id} ==========")
            else:
                self.logger.error(f"S2P_LABEL: ========== Failed labeling for {subject_id} ==========")
            
            return {
                'success': success,
                'sessions_processed': 1 if success else 0,
                'labeling_stats': {
                    'n_rois': len(labels),
                    'n_excitatory': np.sum(labels == -1),
                    'n_inhibitory': np.sum(labels == 1),
                    'n_unlabeled': np.sum(labels == 0),
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
