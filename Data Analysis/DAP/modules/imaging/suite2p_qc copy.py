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
        self.qc_params = self.config.get('suite2p_qc', {})
        
        self.logger.info("S2P_QC: Suite2pQC initialized")
    
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
        masks = np.zeros((ops['Ly'], ops['Lx']))
        for n in range(len(stat)):
            ypix = stat[n]['ypix']
            xpix = stat[n]['xpix']
            masks[ypix, xpix] = n + 1
        return masks

    def apply_qc_filters(self, data: Dict[str, Any], subject_id: str) -> Dict[str, Any]:
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
            experiment_config = self.config_manager.get_experiment_config()
            imaging_config = experiment_config.get('imaging_preprocessing', {})
            qc_method = imaging_config.get('qc_method', 'threshold')          
            
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
            else:
                self.logger.warning(f"S2P_QC: Unknown QC method '{qc_method}', using threshold")
                good_roi_mask = self._apply_threshold_qc(ops, stat, subject_id)
            
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
            
            qc_data = {
                'F': F,
                'Fneu': Fneu,
                'stat': stat_final,
                'masks': masks,
                'ops': ops,
                'spks': spks,
                'iscell_updated': iscell,  # Include updated iscell array
                'qc_stats': {
                    'n_initial_cells': n_initial_cells,
                    'n_final_cells': np.sum(good_roi_mask),
                    'n_rejected_cells': n_initial_cells - np.sum(good_roi_mask),
                    'rejection_rate': (n_initial_cells - np.sum(good_roi_mask)) / n_initial_cells,
                    'qc_method': qc_method,
                    'good_roi_mask': good_roi_mask
                }
            }
            
            self.logger.info(f"S2P_QC: === QC filtering complete for {subject_id} ===")
            self.logger.info(f"S2P_QC: Final result: {n_initial_cells} -> {np.sum(good_roi_mask)} cells "
                           f"({n_initial_cells - np.sum(good_roi_mask)} rejected, "
                           f"{(n_initial_cells - np.sum(good_roi_mask))/n_initial_cells:.1%} rejection rate)")
            
            return qc_data
            
        except Exception as e:
            self.logger.error(f"S2P_QC: QC filtering failed for {subject_id}: {e}")
            return data
    
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
        range_aspect = qc_params.get('range_aspect', [0, 5])
        range_compact = qc_params.get('range_compact', [0, 1.06])
        range_footprint = qc_params.get('range_footprint', [1, 2])
        
        self.logger.info(f"S2P_QC: Threshold parameters:")
        self.logger.info(f"S2P_QC:   range_skew: {range_skew}")
        self.logger.info(f"S2P_QC:   max_connect: {max_connect}")
        self.logger.info(f"S2P_QC:   range_aspect: {range_aspect}")
        self.logger.info(f"S2P_QC:   range_compact: {range_compact}")
        self.logger.info(f"S2P_QC:   range_footprint: {range_footprint}")
        
        self.logger.info("S2P_QC: Computing Suite2p quality metrics...")
        
        # Get Suite2p quality metrics
        skew, connect, aspect, compact, footprint = self.get_suite2p_metrics(ops, stat)
        
        self.logger.info("S2P_QC: Applying threshold filters...")
        
        # Apply thresholds and count each filter's effect
        skew_mask = (skew >= range_skew[0]) & (skew <= range_skew[1])
        connect_mask = connect <= max_connect
        aspect_mask = (aspect >= range_aspect[0]) & (aspect <= range_aspect[1])
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
        aspect_mask = (aspect >= learned_params['range_aspect'][0]) & (aspect <= learned_params['range_aspect'][1])
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
            'range_aspect': [np.min(manual_aspect), np.max(manual_aspect)],
            'range_compact': [np.min(manual_compact), np.max(manual_compact)],
            'range_footprint': [np.min(manual_footprint), np.max(manual_footprint)],
            
            # Threshold parameters: use max of manual selections (to be inclusive)
            'max_connect': np.max(manual_connect),  # Use max instead of mean to be more inclusive
        }
        
        # Add tolerance to ranges to avoid being too restrictive
        tolerance_factor = 0.1  # 10% tolerance
        min_tolerance = 1e-6   # Minimum absolute tolerance for near-zero ranges
        
        # Expand ranges with proper handling of zero-width ranges
        for param_name in ['range_skew', 'range_aspect', 'range_compact', 'range_footprint']:
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
        self.logger.info(f"S2P_QC:   range_aspect: [{learned_params['range_aspect'][0]:.3f}, {learned_params['range_aspect'][1]:.3f}]")
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
        aspect_pass = (aspect[manual_mask] >= learned_params['range_aspect'][0]) & (aspect[manual_mask] <= learned_params['range_aspect'][1])
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
            qc_data = self.apply_qc_filters(raw_data, subject_id)
            
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

