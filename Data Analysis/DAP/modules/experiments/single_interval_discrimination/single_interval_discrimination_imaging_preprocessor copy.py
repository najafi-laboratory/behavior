"""
Single Interval Discrimination Imaging Preprocessor

Experiment-specific imaging preprocessing for single interval discrimination experiments.
Follows the same pattern as other pipeline components.

This handles:
1. Trial-based imaging data alignment
2. Stimulus-response analysis
3. Behavioral-imaging synchronization
4. Experiment-specific quality metrics
"""

import os
import numpy as np
import pickle
import h5py
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from modules.utils.utils import create_memmap, get_memmap_path, clean_memmap_path


class SingleIntervalDiscriminationImagingPreprocessor:
    """
    Single interval discrimination imaging preprocessor following pipeline component pattern.
    
    Handles experiment-specific imaging preprocessing for SID experiments.
    """
    
    def __init__(self, config_manager, subject_list, logger=None):
        """
        Initialize the SID imaging preprocessor.
        
        Args:
            config_manager: ConfigManager instance
            subject_list: List of subject IDs to process
            logger: Logger instance
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        self.subject_list = subject_list
        self.logger = logger or logging.getLogger(__name__)
        
        # Get experiment config
        self.experiment_config = self.config_manager.get_experiment_config()

        paths = self.config.get('paths', {})
        self.preprocessed_root_dir = paths.get('preprocessed_data', '')

        self.logger.info("SID_IMG: SingleIntervalDiscriminationImagingPreprocessor initialized")

    def load_behavior_session(self, subject_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a single preprocessed session.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None if loading fails
        """
        # Get the matching behavioral session from subject config
        subject_config = self.config.get('subjects', {}).get(subject_id, {})
        imaging_sessions = subject_config.get('imaging_sessions', [])
        
        if not imaging_sessions:
            self.logger.error(f"SID_IMG: No imaging sessions found for subject {subject_id}")
            return None
        
        # For now, use the first imaging session (will enhance for multiple sessions later)
        imaging_session = imaging_sessions[0]
        session_id = imaging_session.get('behavior_session')
        
        if not session_id:
            self.logger.error(f"SID_IMG: No behavior_session specified for subject {subject_id}")
            return None        
        
        # Get paths from ConfigManager's config
        preprocessed_dir = os.path.join(self.preprocessed_root_dir, subject_id)
        session_path = os.path.join(preprocessed_dir, f"{session_id}_preprocessed.pkl")

        if not os.path.isfile(session_path):
            self.logger.warning(f"SID_IMG: Preprocessed session not found: {session_path}")
            return None
        
        try:
            with open(session_path, 'rb') as f:
                session_data = pickle.load(f)
            return session_data
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to load session {session_id} for subject {subject_id}: {e}")
            return None

    def load_roi_labels(self, output_path: str, subject_id: str, use_memmap: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load ROI labels from labeling module output.
        
        Args:
            output_path: Path to processed output directory
            subject_id: Subject identifier for logging
            use_memmap: If True, create memory-mapped arrays instead of loading into RAM
            
        Returns:
            Dictionary containing ROI label data or None if failed
        """
        try:
            labels_data = {}
            
            labels_file = os.path.join(output_path, 'masks.h5')
            if not os.path.exists(labels_file):
                self.logger.error(f"SID_IMG: Missing labels file at {labels_file}")
                return None
            
            with h5py.File(labels_file, 'r') as f:
                if use_memmap:
                    # Create memmap for ROI labels
                    memmap_dir = get_memmap_path(output_path, 'sid_imaging', self.logger)
                    labels_data['roi_labels'] = create_memmap(
                        f['labels'][:], 'int8',
                        os.path.join(memmap_dir, 'roi_labels.mmap'), self.logger)
                else:
                    # Load into RAM
                    labels_data['roi_labels'] = f['labels'][:]
                    
                self.logger.info(f"SID_IMG: Loaded ROI labels - shape: {labels_data['roi_labels'].shape}")
                
                # Count excitatory/inhibitory (works with both memmap and regular arrays)
                n_exc = np.sum(labels_data['roi_labels'] == -1)
                n_inh = np.sum(labels_data['roi_labels'] == 1) 
                n_unlabeled = np.sum(labels_data['roi_labels'] == 0)
                
                labels_data['n_excitatory'] = n_exc
                labels_data['n_inhibitory'] = n_inh
                labels_data['n_unlabeled'] = n_unlabeled
                labels_data['n_total'] = len(labels_data['roi_labels'])
                
                self.logger.info(f"SID_IMG: ROI distribution - {n_exc} excitatory, {n_inh} inhibitory, {n_unlabeled} unlabeled")
            
            return labels_data
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to load ROI labels for {subject_id}: {e}")
            return None

    def load_dff_traces(self, output_path: str, subject_id: str, use_memmap: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load DFF traces and imaging timestamps, optionally as memory-mapped arrays.
        
        Args:
            output_path: Path to processed output directory
            subject_id: Subject identifier for logging
            use_memmap: If True, create memory-mapped arrays instead of loading into RAM
            
        Returns:
            Dictionary containing DFF trace data and timing or None if failed
        """
        try:
            dff_data = {}
            
            # Load DFF traces from DFF module
            dff_file = os.path.join(output_path, 'dff.h5')
            if not os.path.exists(dff_file):
                self.logger.error(f"SID_IMG: Missing DFF file at {dff_file}")
                return None
            
            with h5py.File(dff_file, 'r') as f:
                if use_memmap:
                    # Create memmaps directly
                    memmap_dir = get_memmap_path(output_path, 'sid_imaging', self.logger)
                    dff_data['dff_traces'] = create_memmap(
                        f['dff'][:], 'float32',
                        os.path.join(memmap_dir, 'dff_traces.mmap'), self.logger)
                    dff_data['fluo_traces'] = create_memmap(
                        f['fluo'][:], 'float32',
                        os.path.join(memmap_dir, 'fluo_traces.mmap'), self.logger)
                else:
                    # Load into RAM
                    dff_data['dff_traces'] = f['dff'][:]
                    dff_data['fluo_traces'] = f['fluo'][:]
                
                self.logger.info(f"SID_IMG: Loaded DFF traces - shape: {dff_data['dff_traces'].shape}")
                self.logger.info(f"SID_IMG: Loaded fluo traces - shape: {dff_data['fluo_traces'].shape}")
            
            # Load imaging timestamps from ops
            ops_file = os.path.join(output_path, 'ops.npy')
            if not os.path.exists(ops_file):
                self.logger.error(f"SID_IMG: Missing ops file at {ops_file}")
                return None
            
            ops = np.load(ops_file, allow_pickle=True).item()
            
            # Extract timing information
            dff_data['fs'] = ops.get('fs', 30.0)  # Imaging sampling rate
            n_frames = dff_data['dff_traces'].shape[1]
            imaging_time_array = np.arange(n_frames) / dff_data['fs']
            
            # Adjust for photon integration cycle delay
            imaging_time_array = self.adjust_imaging_timing_for_integration_delay(imaging_time_array)
            
            dff_data['n_frames'] = n_frames
            
            if use_memmap:
                # Create memmap for imaging time
                dff_data['imaging_time'] = create_memmap(
                    imaging_time_array, 'float32',
                    os.path.join(memmap_dir, 'imaging_time.mmap'), self.logger)
            else:
                dff_data['imaging_time'] = imaging_time_array
                    
            self.logger.info(f"SID_IMG: Imaging sampling rate: {dff_data['fs']} Hz")
            self.logger.info(f"SID_IMG: Imaging duration: {dff_data['imaging_time'][-1]:.1f} seconds")
            self.logger.info(f"SID_IMG: Number of frames: {n_frames}")
            
            return dff_data
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to load DFF traces for {subject_id}: {e}")
            return None

    def load_voltage_data(self, output_path: str, subject_id: str, use_memmap: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load voltage recordings from raw_voltages.h5 file.
        
        Args:
            output_path: Path to processed output directory
            subject_id: Subject identifier for logging
            use_memmap: If True, create memory-mapped arrays instead of loading into RAM
            
        Returns:
            Dictionary containing voltage data or None if failed
        """
        try:
            voltage_data = {}
            
            voltage_file = os.path.join(output_path, 'raw_voltages.h5')
            
            if not os.path.exists(voltage_file):
                self.logger.warning(f"SID_IMG: No voltage file found at {voltage_file}")
                return self._get_empty_voltage_structure()
            
            self.logger.info(f"SID_IMG: Found voltage file at {voltage_file}")
            
            with h5py.File(voltage_file, 'r') as f:
                # Log available datasets
                self.logger.info(f"SID_IMG: Voltage file contains datasets: {list(f.keys())}")
                
                # Load from 'raw' group
                if 'raw' not in f:
                    self.logger.error("SID_IMG: 'raw' group not found in voltage file")
                    return None
                
                raw_group = f['raw']
                self.logger.info(f"SID_IMG: 'raw' group contains: {list(raw_group.keys())}")
                
                # Load timing information first
                if 'vol_time' not in raw_group:
                    self.logger.error("SID_IMG: 'vol_time' not found in voltage file")
                    return None
                
                voltage_data['voltage_fs'] = 1000
                
                if use_memmap:
                    memmap_dir = get_memmap_path(output_path, 'sid_imaging', self.logger)
                    voltage_data['vol_time'] = create_memmap(
                        raw_group['vol_time'][:] / voltage_data['voltage_fs'], 'float32',
                        os.path.join(memmap_dir, 'vol_time.mmap'), self.logger)
                else:
                    voltage_data['vol_time'] = raw_group['vol_time'][:] / voltage_data['voltage_fs']


                # voltage_data['vol_time'] = voltage_data['vol_time'] / voltage_data['voltage_fs']
                self.logger.info(f"SID_IMG: Loaded voltage time - {len(voltage_data['vol_time'])} samples at {voltage_data['voltage_fs']:.1f} Hz")
                
                
                # Load voltage channels directly to named variables
                if use_memmap:
                    # Load each channel with appropriate dtype
                    voltage_data['vol_start'] = create_memmap(raw_group['vol_start'][:], 'int8', 
                                                            os.path.join(memmap_dir, 'vol_start.mmap'), self.logger) if 'vol_start' in raw_group else None
                    voltage_data['vol_stim_vis'] = create_memmap(raw_group['vol_stim_vis'][:], 'int8',
                                                               os.path.join(memmap_dir, 'vol_stim_vis.mmap'), self.logger) if 'vol_stim_vis' in raw_group else None
                    voltage_data['vol_hifi'] = create_memmap(raw_group['vol_hifi'][:], 'int8',
                                                           os.path.join(memmap_dir, 'vol_hifi.mmap'), self.logger) if 'vol_hifi' in raw_group else None
                    voltage_data['vol_img'] = create_memmap(raw_group['vol_img'][:], 'int8',
                                                          os.path.join(memmap_dir, 'vol_img.mmap'), self.logger) if 'vol_img' in raw_group else None
                    voltage_data['vol_stim_aud'] = create_memmap(raw_group['vol_stim_aud'][:], 'float32',
                                                               os.path.join(memmap_dir, 'vol_stim_aud.mmap'), self.logger) if 'vol_stim_aud' in raw_group else None
                    voltage_data['vol_flir'] = create_memmap(raw_group['vol_flir'][:], 'int8',
                                                           os.path.join(memmap_dir, 'vol_flir.mmap'), self.logger) if 'vol_flir' in raw_group else None
                    voltage_data['vol_pmt'] = create_memmap(raw_group['vol_pmt'][:], 'int8',
                                                          os.path.join(memmap_dir, 'vol_pmt.mmap'), self.logger) if 'vol_pmt' in raw_group else None
                    voltage_data['vol_led'] = create_memmap(raw_group['vol_led'][:], 'int8',
                                                          os.path.join(memmap_dir, 'vol_led.mmap'), self.logger) if 'vol_led' in raw_group else None
                else:
                    # Load directly to RAM
                    voltage_data['vol_start'] = raw_group['vol_start'][:] if 'vol_start' in raw_group else None
                    voltage_data['vol_stim_vis'] = raw_group['vol_stim_vis'][:] if 'vol_stim_vis' in raw_group else None
                    voltage_data['vol_hifi'] = raw_group['vol_hifi'][:] if 'vol_hifi' in raw_group else None
                    voltage_data['vol_img'] = raw_group['vol_img'][:] if 'vol_img' in raw_group else None
                    voltage_data['vol_stim_aud'] = raw_group['vol_stim_aud'][:] if 'vol_stim_aud' in raw_group else None
                    voltage_data['vol_flir'] = raw_group['vol_flir'][:] if 'vol_flir' in raw_group else None
                    voltage_data['vol_pmt'] = raw_group['vol_pmt'][:] if 'vol_pmt' in raw_group else None
                    voltage_data['vol_led'] = raw_group['vol_led'][:] if 'vol_led' in raw_group else None
                
                # Count loaded channels
                loaded_channels = [ch for ch in ['vol_start', 'vol_stim_vis', 'vol_hifi', 'vol_img', 
                                               'vol_stim_aud', 'vol_flir', 'vol_pmt', 'vol_led'] 
                                 if voltage_data.get(ch) is not None]
                
                self.logger.info(f"SID_IMG: Loaded {len(loaded_channels)} voltage channels: {loaded_channels}")
                self.logger.info(f"SID_IMG: Voltage duration: {voltage_data['vol_time'][-1]:.1f} seconds")
            
            return voltage_data
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to load voltage data for {subject_id}: {e}")
            return None
    
    def _get_empty_voltage_structure(self) -> Dict[str, Any]:
        """Return empty voltage data structure for consistency when no voltage file exists."""
        return {
            'vol_time': None,
            'voltage_fs': None,
            'vol_start': None,
            'vol_stim_vis': None,
            'vol_hifi': None,
            'vol_img': None,
            'vol_stim_aud': None,
            'vol_flir': None,
            'vol_pmt': None,
            'vol_led': None
        }

    def load_imaging_data(self, subject_id: str, suite2p_path: str, output_path: str, behavioral_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load all imaging-related data for SID preprocessing.
        
        Args:
            subject_id: Subject identifier
            suite2p_path: Path to Suite2p output directory (plane0)
            output_path: Path to processed output
            
        Returns:
            Dictionary containing loaded imaging data or None if failed
        """
        try:
            data = {}
            
            self.logger.info(f"SID_IMG: Loading imaging data for {subject_id}")
            
            use_memmap = False
            
            # Load ROI labels
            labels_data = self.load_roi_labels(output_path, subject_id, use_memmap=use_memmap)
            if labels_data is None:
                return None
            data.update(labels_data)

            # Load DFF traces and timestamps
            dff_data = self.load_dff_traces(output_path, subject_id, use_memmap=use_memmap)
            if dff_data is None:
                return None
            data.update(dff_data)

            # Load voltage recordings
            voltage_data = self.load_voltage_data(output_path, subject_id, use_memmap=use_memmap)
            if voltage_data is None:
                return None
            data.update(voltage_data)
            
            # Comprehensive alignment check BEFORE time alignment
            # self.plot_comprehensive_alignment_check(data, behavioral_data, subject_id, 
            #                                        alignment_stage='before_time_alignment')
            
            # Test plot showing 120 seconds of voltage traces and DFF
            if voltage_data.get('vol_time') is not None:
                import matplotlib.pyplot as plt
                
                plot_window = 15.0
                
                # Find indices for first 120 seconds in voltage data
                vol_time_mask = data['vol_time'] <= plot_window
                vol_plot_time = data['vol_time'][vol_time_mask]
                
                # Find corresponding imaging frames for first 120 seconds
                img_time_mask = data['imaging_time'] <= plot_window
                img_plot_time = data['imaging_time'][img_time_mask]
                
                fig, axes = plt.subplots(5, 2, figsize=(16, 12))
                fig.suptitle(f'Voltage Traces and DFF - First 120s ({subject_id})', fontsize=14)
                
                # Plot voltage channels (first 4 rows)
                voltage_channels_to_plot = ['vol_start', 'vol_stim_vis', 'vol_hifi', 'vol_img', 
                                          'vol_stim_aud', 'vol_flir', 'vol_pmt', 'vol_led']
                
                for idx, ch_name in enumerate(voltage_channels_to_plot):
                    row = idx // 2
                    col = idx % 2
                    
                    if ch_name in data and data[ch_name] is not None:
                        plot_data = data[ch_name][vol_time_mask]
                        axes[row, col].plot(vol_plot_time, plot_data, linewidth=0.5)
                        axes[row, col].set_title(f'{ch_name}')
                        axes[row, col].set_xlabel('Time (s)')
                        axes[row, col].grid(True, alpha=0.3)
                        
                        # Highlight peaks for vol_start
                        if ch_name == 'vol_start':
                            from scipy.signal import find_peaks
                            peaks, _ = find_peaks(plot_data, height=0.5, distance=10)
                            if len(peaks) > 0:
                                axes[row, col].plot(vol_plot_time[peaks], plot_data[peaks], 'ro', markersize=3)
                                axes[row, col].set_title(f'{ch_name} ({len(peaks)} peaks)')
                    else:
                        axes[row, col].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                                          transform=axes[row, col].transAxes)
                        axes[row, col].set_title(f'{ch_name} (missing)')
                
                # Plot DFF traces (bottom row) - show first few ROIs
                if 'dff_traces' in data and len(img_plot_time) > 0:
                    n_rois_to_plot = min(6, data['dff_traces'].shape[0])  # Plot up to 6 ROIs
                    dff_subset = data['dff_traces'][:n_rois_to_plot, img_time_mask]
                    
                    # Left subplot: individual DFF traces
                    for roi_idx in range(n_rois_to_plot):
                        axes[4, 0].plot(img_plot_time, dff_subset[roi_idx, :] + roi_idx*2, 
                                      linewidth=0.7, alpha=0.8, label=f'ROI {roi_idx}')
                    axes[4, 0].set_title(f'DFF Traces (first {n_rois_to_plot} ROIs)')
                    axes[4, 0].set_xlabel('Time (s)')
                    axes[4, 0].set_ylabel('DFF (offset)')
                    axes[4, 0].grid(True, alpha=0.3)
                    axes[4, 0].legend(fontsize=8)
                    
                    # Right subplot: mean DFF
                    mean_dff = np.mean(dff_subset, axis=0)
                    axes[4, 1].plot(img_plot_time, mean_dff, linewidth=1, color='blue')
                    axes[4, 1].set_title(f'Mean DFF (first {n_rois_to_plot} ROIs)')
                    axes[4, 1].set_xlabel('Time (s)')
                    axes[4, 1].set_ylabel('DFF')
                    axes[4, 1].grid(True, alpha=0.3)
                else:
                    for col in [0, 1]:
                        axes[4, col].text(0.5, 0.5, 'No DFF Data', ha='center', va='center', 
                                        transform=axes[4, col].transAxes)
                        axes[4, col].set_title('DFF (missing)')
                
                plt.tight_layout()
                plt.show()
                
                self.logger.info(f"SID_IMG: Generated voltage and DFF traces plot for first 120 seconds")
            
            # print min/max of each voltage channel
            for ch_name in ['vol_start', 'vol_stim_vis', 'vol_hifi', 'vol_img', 
                            'vol_stim_aud', 'vol_flir', 'vol_pmt', 'vol_led']:
                ch_data = voltage_data.get(ch_name)
                if ch_data is not None:
                    self.logger.info(f"SID_IMG: {ch_name} - min: {np.min(ch_data)}, max: {np.max(ch_data)}")
            
            # print number of vol_start peaks
            if voltage_data.get('vol_start') is not None:
                # Find first index where vol_start goes high (threshold crossing)
                threshold = 0.5
                high_indices = np.where(voltage_data['vol_start'] > threshold)[0]
                
                if len(high_indices) > 0:
                    # Find transitions from low to high by looking for gaps in indices
                    pulse_starts = [high_indices[0]]  # First high point
                    for i in range(1, len(high_indices)):
                        if high_indices[i] - high_indices[i-1] > 1:  # Gap indicates new pulse
                            pulse_starts.append(high_indices[i])
                    
                    self.logger.info(f"SID_IMG: vol_start - detected {len(pulse_starts)} start pulse rising edges")
                    
                    # Align time vectors to first vol_start pulse rising edge (t=0)
                    if len(pulse_starts) > 0:
                        first_pulse_idx = pulse_starts[0]
                        first_pulse_time = data['vol_time'][first_pulse_idx]
                        
                        # Shift voltage time to zero at first pulse
                        data['vol_time'] = data['vol_time'] - first_pulse_time
                        
                        # Shift imaging time to same reference point
                        data['imaging_time'] = data['imaging_time'] - first_pulse_time
                        
                        self.logger.info(f"SID_IMG: Aligned time vectors to first vol_start rising edge at original time {first_pulse_time:.3f}s")
                        self.logger.info(f"SID_IMG: New imaging duration: {data['imaging_time'][-1]:.1f}s")
                        self.logger.info(f"SID_IMG: New voltage duration: {data['vol_time'][-1]:.1f}s")
                        
                        # # Comprehensive alignment check AFTER time alignment
                        # self.plot_comprehensive_alignment_check(data, behavioral_data, subject_id,
                        #                                        alignment_stage='after_time_alignment')
                    else:
                        self.logger.warning(f"SID_IMG: No vol_start pulse rising edges found - cannot align time vectors")
                else:
                    self.logger.warning(f"SID_IMG: No vol_start values above threshold - cannot align time vectors")
                    
            
            
            self.logger.info(f"SID_IMG: Successfully loaded imaging data for {subject_id}")
            return data
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to load imaging data for {subject_id}: {e}")
            return None


    def validate_imaging_behavioral_data(self, imaging_data: Dict[str, Any], behavioral_data: Dict[str, Any], 
                                       subject_id: str) -> Dict[str, Any]:
        """
        Validate that imaging and behavioral data are compatible for alignment.
        
        Args:
            imaging_data: Dictionary containing imaging data
            behavioral_data: Dictionary containing behavioral data
            subject_id: Subject identifier for logging
            
        Returns:
            Dictionary with validation results and alignment info
        """
        try:
            validation_results = {
                'valid': False,
                'n_voltage_trials': 0,
                'n_behavioral_sessions': 0,
                'n_behavioral_trials': 0,
                'missing_stimuli': [],
                'duration_mismatch': False,
                'recommendations': []
            }
            
            self.logger.info(f"SID_IMG: Validating imaging-behavioral data alignment for {subject_id}")
            
            # Count voltage trial start pulses
            if imaging_data.get('vol_start') is not None:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(imaging_data['vol_start'], height=0.5, distance=10)
                num_start_pulses = len(peaks)
                validation_results['n_voltage_trials'] = num_start_pulses
                self.logger.info(f"SID_IMG: Found {num_start_pulses} voltage trial start pulses")
            else:
                self.logger.warning(f"SID_IMG: No vol_start channel available for trial detection")
            
            # Count behavioral trials from loaded session
            n_trials = behavioral_data.get('nTrials', 0)
            session_id = behavioral_data.get('session_id', 'unknown')
            
            self.logger.info(f"SID_IMG: Behavioral session '{session_id}' has {n_trials} trials")
            
            if n_trials != num_start_pulses:
                self.logger.warning(f"SID_IMG: Behavioral trial count ({n_trials}) does not match voltage trials ({num_start_pulses})") 
            
            validation_results['n_behavioral_trials'] = n_trials
            validation_results['n_behavioral_sessions'] = 1  # Single session loaded
            
            
            
            
            # Check trial count matching
            if validation_results['n_voltage_trials'] == validation_results['n_behavioral_trials']:
                self.logger.info(f"SID_IMG: Trial counts match: {validation_results['n_voltage_trials']} trials")
                validation_results['valid'] = True
            else:
                self.logger.warning(f"SID_IMG: Trial count mismatch: {validation_results['n_voltage_trials']} voltage vs {validation_results['n_behavioral_trials']} behavioral")
                validation_results['recommendations'].append("Check for missing trials or extra pulses")
            
            # Check for missing stimulus channels
            required_channels = ['vol_stim_vis', 'vol_stim_aud']
            for ch_name in required_channels:
                if imaging_data.get(ch_name) is None:
                    validation_results['missing_stimuli'].append(ch_name)
                    self.logger.warning(f"SID_IMG: Missing stimulus channel: {ch_name}")
            
            if validation_results['missing_stimuli']:
                validation_results['recommendations'].append("Some stimulus channels are missing - alignment may be incomplete")
            
            # Check duration compatibility
            imaging_duration = imaging_data.get('imaging_time', [0])[-1] if len(imaging_data.get('imaging_time', [])) > 0 else 0
            voltage_duration = imaging_data.get('vol_time', [0])[-1] if imaging_data.get('vol_time') is not None else 0
            
            if abs(imaging_duration - voltage_duration) > 1.0:  # Allow 1 second tolerance
                validation_results['duration_mismatch'] = True
                validation_results['recommendations'].append("Imaging and voltage durations don't match - check synchronization")
                self.logger.warning(f"SID_IMG: Duration mismatch - imaging: {imaging_duration:.1f}s, voltage: {voltage_duration:.1f}s")
            
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to validate imaging-behavioral data for {subject_id}: {e}")
            return {'valid': False, 'error': str(e)}

    def process_subject(self, subject_id: str, suite2p_path: str, output_path: str, force: bool = False) -> Dict[str, Any]:
        """
        Process experiment-specific imaging preprocessing for a single subject.
        
        Args:
            subject_id: Subject identifier
            suite2p_path: Path to Suite2p output directory (plane0)
            output_path: Path for preprocessing output
            force: Force reprocessing even if output exists
            
        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f"SID_IMG: ========== Starting SID imaging preprocessing for {subject_id} ==========")
            self.logger.info(f"SID_IMG: Suite2p path: {suite2p_path}")
            self.logger.info(f"SID_IMG: Output path: {output_path}")
            
            # Check if already processed
            results_file = os.path.join(output_path, 'sid_imaging_results.h5')
            if os.path.exists(results_file) and not force:
                self.logger.info(f"SID_IMG: SID imaging data already exists for {subject_id}, skipping (use force=True to reprocess)")
                return {
                    'success': True,
                    'sessions_processed': 1,
                    'message': 'Already processed (skipped)'
                }
            
 # Get the matching behavioral session from subject config
            subject_config = self.config.get('subjects', {}).get(subject_id, {})
            imaging_sessions = subject_config.get('imaging_sessions', [])
            
            if not imaging_sessions:
                self.logger.error(f"SID_IMG: No imaging sessions found for subject {subject_id}")
                return None
            
            # For now, use the first imaging session (will enhance for multiple sessions later)
            imaging_session = imaging_sessions[0]
            behavior_session_id = imaging_session.get('behavior_session')
            
            if not behavior_session_id:
                self.logger.error(f"SID_IMG: No behavior_session specified for subject {subject_id}")
                return None            
            
            # Load behavioral data
            behavioral_data = self.load_behavior_session(subject_id, behavior_session_id)
            if behavioral_data is None:
                return {
                    'success': False,
                    'sessions_processed': 0,
                    'error_message': 'Failed to load behavioral data'
                }            
            
            # Load imaging data
            imaging_data = self.load_imaging_data(subject_id, suite2p_path, output_path, behavioral_data)
            if imaging_data is None:
                return {
                    'success': False,
                    'sessions_processed': 0,
                    'error_message': 'Failed to load imaging data'
                }
            
            # Comprehensive alignment check AFTER time alignment
            self.plot_comprehensive_alignment_check(imaging_data, behavioral_data, subject_id,
                                                               alignment_stage='after_time_alignment')
           
           # After initial alignment, correct trial timestamps
            df_trials = behavioral_data['df_trials'].copy()
            for trial_idx, row in df_trials.iterrows():
                expected_time = row['trial_start_timestamp']
                actual_pulse_time = self.find_closest_vol_pulse(imaging_data, expected_time, search_window=0.2)
                df_trials.loc[trial_idx, 'trial_start_timestamp'] = actual_pulse_time

            # Update behavioral data with corrected timestamps
            behavioral_data['df_trials'] = df_trials 

            self.plot_comprehensive_alignment_check(imaging_data, behavioral_data, subject_id, 
                                                   alignment_stage='after_trial_time_alignment')

            # Extract trial segments for analysis
            trial_data = self.extract_trial_segments_simple(
                imaging_data, behavioral_data['df_trials'], 
                alignment='trial_start_timestamp', pre_sec=2.0, post_sec=8.0
            )
            
            self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments'].shape[0]} trials")
            self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments'].shape}")
            
            
            
            # Plot trial-segmented data to check alignment
            self.plot_trial_segments_check(trial_data, subject_id)
            
            # Plot individual trials using segmented data
            self._plot_individual_trials_from_segments(trial_data, subject_id)

            # Quick alignment check - plot DFF raster for first 3 trials
            # self.plot_trial_alignment_check(imaging_data, behavioral_data, subject_id)
            
            # Validate data compatibility
            # validation_results = self.validate_imaging_behavioral_data(imaging_data, behavioral_data, subject_id)
            
            # if not validation_results.get('valid', False):
            #     self.logger.error(f"SID_IMG: Data validation failed for {subject_id}")
            #     return {
            #         'success': False,
            #         'sessions_processed': 0,
            #         'error_message': 'Data validation failed',
            #         'validation_results': validation_results
            # }  

            # Data is ready for alignment
            self.logger.info("SID_IMG: Data loaded and validated - ready for trial alignment")
            
            
            return {
                'success': True,
                'sessions_processed': 1,
                'imaging_data_summary': {
                    'n_rois': len(imaging_data['roi_labels']),
                    'n_frames': imaging_data['dff_traces'].shape[1],
                    'imaging_duration': imaging_data['imaging_time'][-1],
                    'has_voltage': imaging_data.get('vol_time') is not None,
                    'voltage_duration': imaging_data['vol_time'][-1] if imaging_data.get('vol_time') is not None else None
                },
                'trial_data': trial_data
            }
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Processing failed for {subject_id}: {e}")
            return {
                'success': False,
                'sessions_processed': 0,
                'error_message': str(e)
            }

    def find_closest_vol_pulse(self, imaging_data: Dict[str, Any], expected_time: float, search_window: float = 0.2) -> float:
        """
        Find the closest vol_start pulse to an expected time within a search window.
        
        Args:
            imaging_data: Dictionary containing voltage data
            expected_time: Expected time of the pulse
            search_window: Search window in seconds (±search_window around expected_time)
            
        Returns:
            Actual time of the closest vol_start pulse
        """
        try:
            # Define search window
            search_start = expected_time - search_window
            search_end = expected_time + search_window
            
            # Find indices within search window
            start_idx = np.searchsorted(imaging_data['vol_time'], search_start)
            end_idx = np.searchsorted(imaging_data['vol_time'], search_end)
            
            if start_idx >= end_idx:
                self.logger.warning(f"SID_IMG: No voltage data in search window around {expected_time:.3f}s")
                return expected_time  # Return original if no data found
            
            # Extract voltage segment and time segment
            vol_segment = imaging_data['vol_start'][start_idx:end_idx]
            time_segment = imaging_data['vol_time'][start_idx:end_idx]
            
            # Find pulse rising edges in this segment
            threshold = 0.5
            high_indices = np.where(vol_segment > threshold)[0]
            
            if len(high_indices) == 0:
                self.logger.warning(f"SID_IMG: No vol_start pulse found in search window around {expected_time:.3f}s")
                return expected_time
            
            # Find transitions from low to high (pulse starts)
            pulse_starts_local = [high_indices[0]]  # First high point
            for i in range(1, len(high_indices)):
                if high_indices[i] - high_indices[i-1] > 1:  # Gap indicates new pulse
                    pulse_starts_local.append(high_indices[i])
            
            # Convert local indices to actual times
            pulse_times = [time_segment[idx] for idx in pulse_starts_local]
            
            # Find closest pulse to expected time
            distances = [abs(t - expected_time) for t in pulse_times]
            closest_idx = np.argmin(distances)
            closest_time = pulse_times[closest_idx]
            
            # Log the correction
            drift = closest_time - expected_time
            if abs(drift) > 0.01:  # Log if drift > 10ms
                self.logger.debug(f"SID_IMG: Trial pulse drift: {drift*1000:.1f}ms at t={expected_time:.3f}s")
            
            return closest_time
            
        except Exception as e:
            self.logger.warning(f"SID_IMG: Error finding closest vol pulse at {expected_time:.3f}s: {e}")
            return expected_time    
    
    def clean_output(self, subject_id: str, output_path: str) -> None:
        """
        Clean up output directory for a subject by removing existing memmap files.
        """
        self.logger.info(f"SID_IMG: Cleaning output for {subject_id}...")
        clean_memmap_path(output_path, 'sid_imaging', self.logger)
        self.logger.info(f"SID_IMG: Output cleaned for {subject_id}")
        return None        
        
    def batch_process(self, force: bool = False) -> Dict[str, bool]:
        """
        Process SID imaging preprocessing for all subjects in the list.
        
        Args:
            force: Force reprocessing even if output exists
            
        Returns:
            Dictionary mapping subject_id to success status
        """
        results = {}
        
        self.logger.info(f"SID_IMG: Starting batch SID imaging processing for {len(self.subject_list)} subjects")
        
        for subject_id in self.subject_list:
            self.logger.info(f"SID_IMG: Processing {subject_id}...")
            results[subject_id] = self.process_subject(subject_id, force)
        
        
        
        # Log summary
        successful = sum(results.values())
        self.logger.info(f"SID_IMG: Batch processing complete: {successful}/{len(self.subject_list)} subjects successful")
        
        return results

    def create_memmap_data(self, imaging_data: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """
        Convert imaging data to memory-mapped arrays for efficient access and Spyder inspection.
        
        Args:
            imaging_data: Dictionary containing loaded imaging data
            output_path: Base output path for memmap files
            
        Returns:
            Dictionary with memory-mapped versions of the data
        """
        try:
            self.logger.info("SID_IMG: Creating memory-mapped arrays for development...")
            
            # Get memmap directory with logger
            memmap_dir = get_memmap_path(output_path, 'sid_imaging', self.logger)
            
            memmap_data = imaging_data.copy()
            
            # Convert DFF traces to memmap
            if 'dff_traces' in imaging_data:
                memmap_data['dff_traces'] = create_memmap(
                    imaging_data['dff_traces'], 'float32',
                    os.path.join(memmap_dir, 'dff_traces.mmap'), self.logger)
                
            # Convert fluorescence traces to memmap  
            if 'fluo_traces' in imaging_data:
                memmap_data['fluo_traces'] = create_memmap(
                    imaging_data['fluo_traces'], 'float32',
                    os.path.join(memmap_dir, 'fluo_traces.mmap'), self.logger)
            
            # Convert voltage channels to memmap
            if 'voltage_channels' in imaging_data and imaging_data['voltage_channels']:
                memmap_voltage_channels = {}
                for ch_name, ch_data in imaging_data['voltage_channels'].items():
                    # Determine appropriate dtype based on channel
                    if 'time' in ch_name or 'stim_aud' in ch_name:
                        dtype = 'float32'
                    else:
                        dtype = 'int8'
                    
                    memmap_voltage_channels[ch_name] = create_memmap(
                        ch_data, dtype,
                        os.path.join(memmap_dir, f'{ch_name}.mmap'), self.logger)
                
                memmap_data['voltage_channels'] = memmap_voltage_channels
            
            # Convert time vectors to memmap
            if 'imaging_time' in imaging_data:
                memmap_data['imaging_time'] = create_memmap(
                    imaging_data['imaging_time'], 'float32',
                    os.path.join(memmap_dir, 'imaging_time.mmap'), self.logger)
                
            if 'voltage_time' in imaging_data and imaging_data['voltage_time'] is not None:
                memmap_data['voltage_time'] = create_memmap(
                    imaging_data['voltage_time'], 'float32',
                    os.path.join(memmap_dir, 'voltage_time.mmap'), self.logger)
            
            self.logger.info(f"SID_IMG: Created memory-mapped arrays in {memmap_dir}")
            self.logger.info("SID_IMG: Arrays now available for Spyder variable inspector")
            
            return memmap_data
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create memory-mapped data: {e}")
            return imaging_data  # Return original data if memmap creation fails

    def adjust_imaging_timing_for_integration_delay(self, imaging_time_array: np.ndarray) -> np.ndarray:
        """
        Adjust imaging time vector to account for photon integration cycle delay.
        
        This shifts the timing to represent the center of the photon integration cycle
        rather than the start of each frame acquisition.
        
        Args:
            imaging_time_array: Original imaging time vector (frame start times)
            
        Returns:
            Adjusted imaging time vector (center of integration cycles)
        """
        # Calculate the mean frame interval (photon integration period)
        mean_frame_interval = np.mean(np.diff(imaging_time_array))
        
        # Shift by half the integration period to center timing
        adjusted_timing = imaging_time_array + (mean_frame_interval / 2.0)
        
        self.logger.debug(f"SID_IMG: Adjusted imaging timing by {mean_frame_interval/2.0:.6f}s for integration delay")
        
        return adjusted_timing


    def trim_sequences_to_common_length(self, sequences: List[np.ndarray], data_type: str) -> np.ndarray:
        """
        Trim sequences to common length (similar to trim_seq from Alignment.py).
        
        Args:
            sequences: List of arrays to trim
            data_type: Type of data for logging
            
        Returns:
            Array with trimmed sequences of common length
        """
        if not sequences:
            return np.array([])
        
        # Find minimum length
        min_length = min(seq.shape[-1] for seq in sequences)
        
        # Trim all sequences to minimum length
        if sequences[0].ndim == 1:
            # 1D sequences (time vectors)
            trimmed = [seq[:min_length] for seq in sequences]
        else:
            # 2D sequences (DFF traces)
            trimmed = [seq[:, :min_length] for seq in sequences]
        
        self.logger.debug(f"SID_IMG: Trimmed {len(sequences)} {data_type} sequences to length {min_length}")
        
        return np.array(trimmed)
    
    def plot_trial_alignment_check(self, imaging_data: Dict[str, Any], behavioral_data: Dict[str, Any], subject_id: str) -> None:
        """
        Quick check of trial alignment with mean DFF per trial, split by rewarded left vs right trials.
        
        Args:
            imaging_data: Dictionary containing imaging data
            behavioral_data: Dictionary containing behavioral data  
            subject_id: Subject identifier for plot title
        """
        try:
            import matplotlib.pyplot as plt
            
            df_trials = behavioral_data['df_trials']
            
            # Filter trials by reward and side
            left_rewarded = df_trials[(df_trials['trial_side'] == 'left') & (df_trials['rewarded'] == 1)]
            right_rewarded = df_trials[(df_trials['trial_side'] == 'right') & (df_trials['rewarded'] == 1)]
            
            n_left = min(150, len(left_rewarded))
            n_right = min(150, len(right_rewarded))
            
            if n_left == 0 and n_right == 0:
                self.logger.warning(f"SID_IMG: No rewarded trials available for alignment check")
                return
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f'Trial Alignment Check - Left vs Right Rewarded Trials ({subject_id})', fontsize=14)
            
            # Extract segments for trials, aligned to choice_start
            pre_choice_sec = 3.0
            post_choice_sec = 5.0
            
            trial_types = [
                ('Left Rewarded', left_rewarded.iloc[:n_left], axes[0]),
                ('Right Rewarded', right_rewarded.iloc[:n_right], axes[1])
            ]
            
            for trial_type_name, trial_subset, ax in trial_types:
                if len(trial_subset) == 0:
                    ax.text(0.5, 0.5, f'No {trial_type_name} trials', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{trial_type_name} (n=0)')
                    continue
                
                trial_dff_segments = []
                trial_time_vectors = []
                
                for _, trial_row in trial_subset.iterrows():
                    choice_start_time = trial_row.get('reward_start', np.nan)
                    
                    if pd.isna(choice_start_time):
                        continue
                    
                    # Define trial window relative to choice_start
                    window_start = choice_start_time - pre_choice_sec
                    window_end = choice_start_time + post_choice_sec
                    
                    # Extract imaging segment
                    img_start_idx = np.searchsorted(imaging_data['imaging_time'], window_start)
                    img_end_idx = np.searchsorted(imaging_data['imaging_time'], window_end)
                    
                    if img_start_idx < img_end_idx and img_end_idx <= len(imaging_data['imaging_time']):
                        # DFF segment and mean across ROIs
                        dff_segment = imaging_data['dff_traces'][:, img_start_idx:img_end_idx]
                        mean_dff = np.mean(dff_segment, axis=0)
                        trial_time = imaging_data['imaging_time'][img_start_idx:img_end_idx] - choice_start_time
                        
                        trial_dff_segments.append(mean_dff)
                        trial_time_vectors.append(trial_time)
                
                if not trial_dff_segments:
                    ax.text(0.5, 0.5, f'No valid {trial_type_name} segments', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{trial_type_name} (n=0)')
                    continue
                
                # Trim to common length
                # trial_dff_segments = self.trim_sequences_to_common_length(trial_dff_segments, 'trial_dff')
                # trial_time_vectors = self.trim_sequences_to_common_length(trial_time_vectors, 'trial_time')
                
                # Create raster plot - each row is one trial
                time_vector = trial_time_vectors[0]
                
                im = ax.imshow(trial_dff_segments, aspect='auto', cmap='RdBu_r',
                              extent=[time_vector[0], time_vector[-1], 0, len(trial_dff_segments)],
                              vmin=-1, vmax=2, interpolation='nearest')
                
                # Mark choice_start (should be at x=0)
                ax.axvline(x=0, color='white', linestyle='--', linewidth=3, alpha=0.9)
                
                # Mark other events relative to choice_start
                event_columns = ['start_flash1', 'end_flash1', 'start_flash2', 'end_flash2', 'reward_start']
                event_colors = ['red', 'orange', 'blue', 'cyan', 'green']
                
                # Calculate average event times for this trial type
                for event_name, color in zip(event_columns, event_colors):
                    event_times_relative = []
                    for _, trial_row in trial_subset.iterrows():
                        choice_start = trial_row.get('choice_start', np.nan)
                        event_time = trial_row.get(event_name, np.nan)
                        
                        if not pd.isna(choice_start) and not pd.isna(event_time):
                            event_times_relative.append(event_time - choice_start)
                    
                    if event_times_relative:
                        avg_event_time = np.mean(event_times_relative)
                        if time_vector[0] <= avg_event_time <= time_vector[-1]:
                            ax.axvline(x=avg_event_time, color='white', linestyle='-', 
                                      linewidth=2, alpha=0.7)
                            ax.text(avg_event_time, len(trial_dff_segments)*0.95, event_name,
                                   rotation=90, fontsize=8, color=color,
                                   ha='center', va='bottom', weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                ax.set_xlabel('Time relative to choice start (s)')
                ax.set_ylabel('Trial #')
                ax.set_title(f'{trial_type_name} (n={len(trial_dff_segments)})')
                
                # Add colorbar for the right plot only
                if trial_type_name == 'Right Rewarded':
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Mean DFF')
            
            plt.tight_layout()
            plt.show()
            
            self.logger.info(f"SID_IMG: Generated split trial alignment raster plot - Left: {n_left}, Right: {n_right} trials")
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create trial alignment check plot: {e}")

    def plot_comprehensive_alignment_check(self, imaging_data: Dict[str, Any], behavioral_data: Dict[str, Any], 
                                         subject_id: str, alignment_stage: str) -> None:
        """
        Comprehensive alignment check with DFF/voltage vs behavioral event validation.
        
        Args:
            imaging_data: Dictionary containing imaging data
            behavioral_data: Dictionary containing behavioral data
            subject_id: Subject identifier for plot title
            alignment_stage: 'before_time_alignment' or 'after_time_alignment'
        """
        try:
            import matplotlib.pyplot as plt
            
            df_trials = behavioral_data['df_trials']
            
            self.logger.info(f"SID_IMG: Creating comprehensive alignment check ({alignment_stage})")
            
            # Plot 1: Overview showing first 3 trials with DFF/voltage/behavioral events
            self._plot_session_overview_with_trials(imaging_data, df_trials, subject_id, alignment_stage)
            
            # Plot 2: Individual trial plots for first 10 trials
            self._plot_individual_trials(imaging_data, df_trials, subject_id, alignment_stage)
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create comprehensive alignment check: {e}")

    def _plot_session_overview_with_trials(self, imaging_data, df_trials, subject_id, alignment_stage):
        """Plot session overview showing DFF, voltage channels, and behavioral events for first 3 trials"""
        try:
            import matplotlib.pyplot as plt
            
            # Determine time window to show first 3 trials plus some buffer
            if len(df_trials) >= 3:
                trial_3_start = df_trials.iloc[2]['trial_start_timestamp']
                time_window = trial_3_start + 30.0  # Show 30s past 3rd trial start
            else:
                time_window = 60.0  # Default to 60s
            
            # Create time masks
            img_time_mask = imaging_data['imaging_time'] <= time_window
            vol_time_mask = imaging_data['vol_time'] <= time_window
            
            img_time_subset = imaging_data['imaging_time'][img_time_mask]
            vol_time_subset = imaging_data['vol_time'][vol_time_mask]
            
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            fig.suptitle(f'Session Overview - {alignment_stage.replace("_", " ").title()} ({subject_id})', fontsize=14)
            
            # Plot 1: Mean DFF
            mean_dff = np.mean(imaging_data['dff_traces'][:, img_time_mask], axis=0)
            axes[0].plot(img_time_subset, mean_dff, 'b-', linewidth=1.5, label='Mean DFF')
            axes[0].set_ylabel('Mean DFF')
            axes[0].set_title('Mean DFF Activity')
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Voltage start channel
            if imaging_data.get('vol_start') is not None:
                axes[1].plot(vol_time_subset, imaging_data['vol_start'][vol_time_mask], 'r-', linewidth=1, label='vol_start')
                axes[1].set_ylabel('Voltage')
                axes[1].set_title('Trial Start Pulses')
                axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Voltage stimulus channels
            y_offset = 0
            for ch_name, color in [('vol_stim_vis', 'green'), ('vol_stim_aud', 'purple')]:
                if imaging_data.get(ch_name) is not None:
                    ch_data = imaging_data[ch_name][vol_time_mask]
                    axes[2].plot(vol_time_subset, ch_data + y_offset, color=color, 
                               linewidth=1, label=ch_name, alpha=0.8)
                    y_offset += 2
            axes[2].set_ylabel('Voltage + Offset')
            axes[2].set_title('Stimulus Channels')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            # Plot 4: Behavioral events timeline
            axes[3].set_xlim(0, time_window)
            axes[3].set_ylim(-1, len(df_trials[:3]))
            
            # Mark behavioral events for first 3 trials
            event_colors = {'start_flash_1': 'red', 'end_flash_1': 'orange', 'start_flash_2': 'blue', 
                           'end_flash_2': 'cyan', 'choice_start': 'green', 'reward_start': 'purple',
                           'punish_start': 'black'}
            
            for trial_idx in range(min(3, len(df_trials))):
                trial_row = df_trials.iloc[trial_idx]
                
                # Plot trial start as vertical line
                trial_start = trial_row.get('trial_start_timestamp', np.nan)
                if not pd.isna(trial_start) and 0 <= trial_start <= time_window:
                    axes[3].axvline(x=trial_start, color='gray', linestyle='--', alpha=0.5)
                    axes[3].text(trial_start, trial_idx + 0.1, 'Trial Start', rotation=90, fontsize=8)
                
                # Plot other events as points
                for event_name, color in event_colors.items():
                    if event_name in df_trials.columns and not pd.isna(trial_row[event_name]):
                        event_time = trial_row[event_name]
                        if 0 <= event_time <= time_window:
                            axes[3].scatter(event_time, trial_idx, color=color, s=50, alpha=0.8, label=event_name if trial_idx == 0 else "")
            
            axes[3].set_ylabel('Trial #')
            axes[3].set_xlabel('Time (s)')
            axes[3].set_title('Behavioral Events (First 3 Trials)')
            axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[3].grid(True, alpha=0.3)
            
            # Add behavioral events as vertical lines on all plots
            for trial_idx in range(min(3, len(df_trials))):
                trial_row = df_trials.iloc[trial_idx]
                for event_name, color in event_colors.items():
                    if event_name in df_trials.columns and not pd.isna(trial_row[event_name]):
                        event_time = trial_row[event_name]
                        if 0 <= event_time <= time_window:
                            for ax in axes[:3]:
                                ax.axvline(x=event_time, color=color, alpha=0.3, linestyle='-', linewidth=1)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create session overview plot: {e}")

    def _plot_individual_trials(self, imaging_data, df_trials, subject_id, alignment_stage):
        """Plot individual trial alignment for first 10 trials"""
        try:
            import matplotlib.pyplot as plt
            
            # n_trials_to_plot = min(10, len(df_trials))
            start_trial = 400
            trial_incr = 3
            stop_trial = start_trial + trial_incr
            n_trials_to_plot = len(df_trials)
            
            for trial_idx in range(start_trial, stop_trial):
                trial_row = df_trials.iloc[trial_idx]
                trial_start = trial_row.get('trial_start_timestamp', np.nan)
                
                if pd.isna(trial_start):
                    continue
                
                # Define trial window
                pre_trial_sec = 3.0
                post_trial_sec = 10.0
                window_start = trial_start - pre_trial_sec
                window_end = trial_start + post_trial_sec
                
                # Create time masks
                img_start_idx = np.searchsorted(imaging_data['imaging_time'], window_start)
                img_end_idx = np.searchsorted(imaging_data['imaging_time'], window_end)
                vol_start_idx = np.searchsorted(imaging_data['vol_time'], window_start)
                vol_end_idx = np.searchsorted(imaging_data['vol_time'], window_end)
                
                if img_start_idx >= img_end_idx or vol_start_idx >= vol_end_idx:
                    continue
                
                # Extract data segments
                img_time_segment = imaging_data['imaging_time'][img_start_idx:img_end_idx] - trial_start
                vol_time_segment = imaging_data['vol_time'][vol_start_idx:vol_end_idx] - trial_start
                mean_dff_segment = np.mean(imaging_data['dff_traces'][:, img_start_idx:img_end_idx], axis=0)
                
                # Create plot for this trial
                fig, axes = plt.subplots(3, 1, figsize=(12, 8))
                fig.suptitle(f'Trial {trial_idx} Alignment - {alignment_stage.replace("_", " ").title()} ({subject_id})', fontsize=12)
                
                # Plot 1: Mean DFF
                axes[0].plot(img_time_segment, mean_dff_segment, 'b-', linewidth=2)
                axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Trial Start')
                axes[0].set_ylabel('Mean DFF')
                axes[0].set_title('Mean DFF Activity')
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: Voltage start
                if imaging_data.get('vol_start') is not None:
                    vol_start_segment = imaging_data['vol_start'][vol_start_idx:vol_end_idx]
                    axes[1].plot(vol_time_segment, vol_start_segment, 'r-', linewidth=1.5)
                    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                    axes[1].set_ylabel('Voltage')
                    axes[1].set_title('Trial Start Pulse')
                    axes[1].grid(True, alpha=0.3)
                
                # Plot 3: Voltage stimulus channels
                y_offset = 0
                for ch_name, color in [('vol_stim_vis', 'green'), ('vol_stim_aud', 'purple')]:
                    if imaging_data.get(ch_name) is not None:
                        ch_segment = imaging_data[ch_name][vol_start_idx:vol_end_idx]
                        axes[2].plot(vol_time_segment, ch_segment + y_offset, color=color,
                                   linewidth=1.5, label=ch_name, alpha=0.8)
                        y_offset += 2
                
                axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                axes[2].set_ylabel('Voltage + Offset')
                axes[2].set_xlabel('Time relative to trial start (s)')
                axes[2].set_title('Stimulus Channels')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                # Add behavioral events as vertical lines to ALL subplots
                event_colors = {
                    'start_flash_1': 'red', 'end_flash_1': 'orange', 'start_flash_2': 'blue',
                    'end_flash_2': 'cyan', 'choice_start': 'green', 'reward_start': 'purple',
                    'punish_start': 'black'
                }

                trial_start = 0
                for event_name, color in event_colors.items():
                    if event_name in df_trials.columns and not pd.isna(trial_row[event_name]):
                        event_time_rel = trial_row[event_name] - trial_start
                        if -pre_trial_sec <= event_time_rel <= post_trial_sec:
                            for ax_idx, ax in enumerate(axes):
                                ax.axvline(x=event_time_rel, color=color, alpha=0.6, 
                                         linestyle='-', linewidth=2)
                                # Add event labels on each subplot
                                ax.text(event_time_rel, ax.get_ylim()[1]*0.9, event_name,
                                       rotation=90, fontsize=8, color=color, 
                                       ha='center', va='top', weight='bold')
                
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create individual trial plots: {e}")

    def _plot_mean_dff_trial_aligned(self, ax, imaging_data, df_trials):
        """Plot mean DFF aligned to trial starts"""
        try:
            # Extract trial segments aligned to trial start
            pre_sec, post_sec = 2.0, 8.0
            trial_segments = []
            
            for _, trial_row in df_trials.iterrows():
                trial_start = trial_row.get('trial_start_timestamp', np.nan)
                if pd.isna(trial_start):
                    continue
                    
                window_start = trial_start - pre_sec
                window_end = trial_start + post_sec
                
                start_idx = np.searchsorted(imaging_data['imaging_time'], window_start)
                end_idx = np.searchsorted(imaging_data['imaging_time'], window_end)
                
                if start_idx < end_idx and end_idx <= len(imaging_data['imaging_time']):
                    segment = np.mean(imaging_data['dff_traces'][:, start_idx:end_idx], axis=0)
                    trial_segments.append(segment)
            
            if trial_segments:
                # Trim to common length and average
                min_len = min(len(seg) for seg in trial_segments)
                trimmed_segments = [seg[:min_len] for seg in trial_segments]
                mean_segment = np.mean(trimmed_segments, axis=0)
                
                time_vector = np.linspace(-pre_sec, post_sec, len(mean_segment))
                ax.plot(time_vector, mean_segment, linewidth=2, color='blue')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Trial Start')
                
                ax.set_title(f'Mean DFF aligned to trial start (n={len(trial_segments)})')
                ax.set_xlabel('Time relative to trial start (s)')
                ax.set_ylabel('Mean DFF')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No valid trial segments', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_alignment_metrics(self, ax, imaging_data, df_trials, alignment_stage):
        """Plot alignment quality metrics"""
        try:
            from scipy.signal import find_peaks
            
            metrics = {}
            
            # Voltage-behavioral sync quality
            if imaging_data.get('vol_start') is not None:
                peaks, _ = find_peaks(imaging_data['vol_start'], height=0.5, distance=10)
                vol_times = imaging_data['vol_time'][peaks]
                beh_times = df_trials['trial_start_timestamp'].values
                
                if len(vol_times) == len(behaves):
                    time_diffs = np.abs(vol_times - behaves)
                    metrics['Max Time Diff (s)'] = np.max(time_diffs)
                    metrics['Mean Time Diff (s)'] = np.mean(time_diffs)
                    metrics['Sync Quality'] = 1.0 if np.max(time_diffs) < 0.05 else 0.5
                else:
                    metrics['Count Mismatch'] = abs(len(vol_times) - len(behaves))
                    metrics['Sync Quality'] = 0.0
            
            # Data completeness
            event_columns = ['start_flash_1', 'end_flash_1', 'start_flash_2', 'end_flash_2', 'choice_start']
            total_events = len(df_trials) * len([col for col in event_columns if col in df_trials.columns])
            missing_events = sum(df_trials[col].isna().sum() for col in event_columns if col in df_trials.columns)
            metrics['Data Completeness'] = 1.0 - (missing_events / total_events) if total_events > 0 else 0.0
            
            # DFF signal quality (mean absolute signal)
            metrics['Mean DFF Signal'] = np.mean(np.abs(imaging_data['dff_traces']))
            
            # Plot metrics as horizontal bar chart
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = ax.barh(range(len(metric_names)), metric_values, alpha=0.7)
            ax.set_yticks(range(len(metric_names)))
            ax.set_yticklabels(metric_names)
            ax.set_xlabel('Value')
            ax.set_title(f'Alignment Quality Metrics\n({alignment_stage})')
            
            # Color bars based on quality
            for bar, value in zip(bars, metric_values):
                if 'Quality' in metric_names[bars.index(bar)] or 'Completeness' in metric_names[bars.index(bar)]:
                    if value > 0.8:
                        bar.set_color('green')
                    elif value > 0.5:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
            
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def extract_trial_segments_simple(self, imaging_data: Dict[str, Any], df_trials: pd.DataFrame,
                                    alignment: str = 'trial_start_timestamp', 
                                    pre_sec: float = 2.0, post_sec: float = 8.0) -> Dict[str, Any]:
        """
        Simple trial segmentation function for flexible analysis.
        
        Args:
            imaging_data: Dictionary containing continuous imaging data
            df_trials: Behavioral trial metadata
            alignment: Column name in df_trials to align to
            pre_sec: Seconds before alignment point
            post_sec: Seconds after alignment point
            
        Returns:
            Dictionary with trial-segmented arrays ready for analysis
        """
        try:
            self.logger.info(f"SID_IMG: Extracting trial segments aligned to '{alignment}'")
            
            dff_segments = []
            vol_segments = {}
            valid_trials = []
            
            # Initialize voltage segment containers
            vol_channels = ['vol_start', 'vol_stim_vis', 'vol_stim_aud']
            for ch in vol_channels:
                if imaging_data.get(ch) is not None:
                    vol_segments[ch] = []
            
            # Extract segments for each trial
            for trial_idx, row in df_trials.iterrows():
                align_time = row.get(alignment, np.nan)
                
                if pd.isna(align_time):
                    continue
                
                # Define window
                window_start = align_time - pre_sec
                window_end = align_time + post_sec
                
                # Extract DFF segment
                img_start_idx = np.searchsorted(imaging_data['imaging_time'], window_start)
                img_end_idx = np.searchsorted(imaging_data['imaging_time'], window_end)

                if img_start_idx < img_end_idx and img_end_idx <= len(imaging_data['imaging_time']):
                    dff_segment = imaging_data['dff_traces'][:, img_start_idx:img_end_idx]
                    dff_segments.append(dff_segment)
                    valid_trials.append(trial_idx)
                    
                    # Extract voltage segments
                    vol_start_idx = np.searchsorted(imaging_data['vol_time'], window_start)
                    vol_end_idx = np.searchsorted(imaging_data['vol_time'], window_end)
                    
                    for ch in vol_channels:
                        if ch in vol_segments and imaging_data.get(ch) is not None:
                            vol_segment = imaging_data[ch][vol_start_idx:vol_end_idx]
                            vol_segments[ch].append(vol_segment)
            
            # Trim to common length
            if dff_segments:
                # dff_segments = self.trim_sequences_to_common_length(dff_segments, 'dff')
                
                # for ch in vol_segments:
                #     if vol_segments[ch]:
                #         vol_segments[ch] = self.trim_sequences_to_common_length(vol_segments[ch], ch)
                
                # Create time vector
                n_timepoints = dff_segments.shape[2]
                time_vector = np.linspace(-pre_sec, post_sec, n_timepoints)
                
                return {
                    'dff_segments': dff_segments,  # Shape: (n_trials, n_rois, n_timepoints)
                    'vol_segments': vol_segments,  # Each channel: (n_trials, n_timepoints)
                    'time_vector': time_vector,
                    'df_trials': df_trials.iloc[valid_trials].copy(),
                    'alignment_point': alignment,
                    'window': {'pre_sec': pre_sec, 'post_sec': post_sec},
                    'n_trials': len(valid_trials)
                }
            else:
                return {'error': 'No valid trial segments extracted'}
                
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to extract trial segments: {e}")
            return {'error': str(e)}

    def plot_trial_segments_check(self, trial_data: Dict[str, Any], subject_id: str) -> None:
        """
        Plot trial-segmented data to check alignment quality.
        
        Args:
            trial_data: Dictionary containing segmented trial data
            subject_id: Subject identifier for plot title
        """
        try:
            import matplotlib.pyplot as plt
            
            if 'error' in trial_data:
                self.logger.error(f"SID_IMG: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            dff_segments = trial_data['dff_segments']
            time_vector = trial_data['time_vector']
            df_trials = trial_data['df_trials']
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f'Trial Segments Check - {subject_id}', fontsize=14)
            
            # Plot 1: Mean DFF across all trials and ROIs
            mean_dff_all = np.mean(dff_segments, axis=(0, 1))  # Average across trials and ROIs
            axes[0, 0].plot(time_vector, mean_dff_all, 'b-', linewidth=2)
            axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Alignment Point')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Mean DFF')
            axes[0, 0].set_title(f'Grand Average DFF (n={dff_segments.shape[0]} trials)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Plot 2: Trial raster - mean DFF per trial
            trial_means = np.mean(dff_segments, axis=1)  # Average across ROIs for each trial
            im = axes[0, 1].imshow(trial_means, aspect='auto', cmap='RdBu_r',
                                  extent=[time_vector[0], time_vector[-1], 0, len(trial_means)],
                                  vmin=-1, vmax=2, interpolation='nearest')
            axes[0, 1].axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.9)
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Trial #')
            axes[0, 1].set_title('Trial Raster (Mean DFF per trial)')
            plt.colorbar(im, ax=axes[0, 1], label='Mean DFF')
            
            # Plot 3: Left vs Right trials (if available)
            if 'trial_side' in df_trials.columns:
                left_trials = df_trials['trial_side'] == 'left'
                right_trials = df_trials['trial_side'] == 'right'
                
                if np.any(left_trials):
                    left_mean = np.mean(dff_segments[left_trials.values], axis=(0, 1))
                    axes[1, 0].plot(time_vector, left_mean, 'g-', linewidth=2, label=f'Left (n={np.sum(left_trials)})')
                
                if np.any(right_trials):
                    right_mean = np.mean(dff_segments[right_trials.values], axis=(0, 1))
                    axes[1, 0].plot(time_vector, right_mean, 'r-', linewidth=2, label=f'Right (n={np.sum(right_trials)})')
                
                axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.7)
                axes[1, 0].set_xlabel('Time (s)')
                axes[1, 0].set_ylabel('Mean DFF')
                axes[1, 0].set_title('Left vs Right Trials')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, 'No trial_side data', ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # Plot 4: Voltage segments (if available)
            if 'vol_start' in trial_data['vol_segments']:
                vol_start_segments = trial_data['vol_segments']['vol_start']
                vol_time_vector = time_vector  # Assuming same time vector for voltage
                
                # Plot first few trials
                n_trials_to_show = min(5, len(vol_start_segments))
                for i in range(n_trials_to_show):
                    axes[1, 1].plot(vol_time_vector, vol_start_segments[i] + i*2, 
                                   linewidth=1, alpha=0.7, label=f'Trial {i}')
                
                axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                axes[1, 1].set_xlabel('Time (s)')
                axes[1, 1].set_ylabel('Voltage + Offset')
                axes[1, 1].set_title(f'vol_start Segments (first {n_trials_to_show} trials)')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
            else:
                axes[1, 1].text(0.5, 0.5, 'No voltage data', ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout()
            plt.show()
            
            self.logger.info(f"SID_IMG: Generated trial segments check plot")
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create trial segments check plot: {e}")

    def _plot_individual_trials_from_segments(self, trial_data: Dict[str, Any], subject_id: str) -> None:
        """
        Plot individual trials using pre-segmented trial data.
        
        Args:
            trial_data: Dictionary containing segmented trial data
            subject_id: Subject identifier for plot title
        """
        try:
            import matplotlib.pyplot as plt
            
            if 'error' in trial_data:
                self.logger.error(f"SID_IMG: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            dff_segments = trial_data['dff_segments']
            vol_segments = trial_data['vol_segments']
            time_vector = trial_data['time_vector']
            df_trials = trial_data['df_trials']
            
            # Plot first few trials
            start_trial = 0
            n_trials_to_plot = min(3, len(dff_segments))
            
            for trial_idx in range(start_trial, start_trial + n_trials_to_plot):
                if trial_idx >= len(dff_segments):
                    break
                    
                trial_row = df_trials.iloc[trial_idx]
                
                # Create plot for this trial
                fig, axes = plt.subplots(3, 1, figsize=(12, 8))
                fig.suptitle(f'Trial {trial_idx} Segments - {subject_id}', fontsize=12)
                
                # Plot 1: Mean DFF for this trial
                mean_dff_trial = np.mean(dff_segments[trial_idx], axis=0)
                axes[0].plot(time_vector, mean_dff_trial, 'b-', linewidth=2)
                axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Alignment Point')
                axes[0].set_ylabel('Mean DFF')
                axes[0].set_title('Mean DFF Activity')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
                
                # Plot 2: Voltage start for this trial (if available)
                if 'vol_start' in vol_segments and len(vol_segments['vol_start']) > trial_idx:
                    vol_start_trial = vol_segments['vol_start'][trial_idx]
                    axes[1].plot(time_vector, vol_start_trial, 'r-', linewidth=1.5)
                    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                    axes[1].set_ylabel('Voltage')
                    axes[1].set_title('Trial Start Pulse')
                    axes[1].grid(True, alpha=0.3)
                else:
                    axes[1].text(0.5, 0.5, 'No vol_start data', ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title('Trial Start Pulse (no data)')
                
                # Plot 3: Voltage stimulus channels for this trial
                y_offset = 0
                for ch_name, color in [('vol_stim_vis', 'green'), ('vol_stim_aud', 'purple')]:
                    if ch_name in vol_segments and len(vol_segments[ch_name]) > trial_idx:
                        ch_trial = vol_segments[ch_name][trial_idx]
                        axes[2].plot(time_vector, ch_trial + y_offset, color=color,
                                   linewidth=1.5, label=ch_name, alpha=0.8)
                        y_offset += 2
                
                axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                axes[2].set_ylabel('Voltage + Offset')
                axes[2].set_xlabel('Time relative to alignment (s)')
                axes[2].set_title('Stimulus Channels')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                # Add behavioral events as vertical lines (relative to alignment point)
                event_colors = {
                    'start_flash_1': 'red', 'end_flash_1': 'orange', 'start_flash_2': 'blue',
                    'end_flash_2': 'cyan', 'choice_start': 'green', 'reward_start': 'purple',
                    'punish_start': 'black'
                }
                
                alignment_time = trial_row.get(trial_data['alignment_point'], 0)
                for event_name, color in event_colors.items():
                    if event_name in trial_row and not pd.isna(trial_row[event_name]):
                        event_time_rel = trial_row[event_name] - alignment_time
                        if time_vector[0] <= event_time_rel <= time_vector[-1]:
                            for ax in axes:
                                ax.axvline(x=event_time_rel, color=color, alpha=0.6, 
                                         linestyle='-', linewidth=2)
                                # Add event labels
                                ax.text(event_time_rel, ax.get_ylim()[1]*0.9, event_name,
                                       rotation=90, fontsize=8, color=color, 
                                       ha='center', va='top', weight='bold')
                
                plt.tight_layout()
                plt.show()
                
            self.logger.info(f"SID_IMG: Generated individual trial plots from segments for {n_trials_to_plot} trials")
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create individual trial plots from segments: {e}")


