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
import matplotlib.pyplot as plt

from scipy import stats
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

        self.logger.info("SID_IMG_PRE: SingleIntervalDiscriminationImagingPreprocessor initialized")

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
            self.logger.error(f"SID_IMG_PRE: No imaging sessions found for subject {subject_id}")
            return None
        
        # For now, use the first imaging session (will enhance for multiple sessions later)
        imaging_session = imaging_sessions[0]
        session_id = imaging_session.get('behavior_session')
        
        if not session_id:
            self.logger.error(f"SID_IMG_PRE: No behavior_session specified for subject {subject_id}")
            return None        
        
        # Get paths from ConfigManager's config
        preprocessed_dir = os.path.join(self.preprocessed_root_dir, subject_id)
        session_path = os.path.join(preprocessed_dir, f"{session_id}_preprocessed.pkl")

        if not os.path.isfile(session_path):
            self.logger.warning(f"SID_IMG_PRE: Preprocessed session not found: {session_path}")
            return None
        
        try:
            with open(session_path, 'rb') as f:
                session_data = pickle.load(f)
            return session_data
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to load session {session_id} for subject {subject_id}: {e}")
            return None

    # def load_roi_labels(self, output_path: str, subject_id: str, use_memmap: bool = True) -> Optional[Dict[str, Any]]:
    def load_roi_labels(self, output_path: str, subject_id: str) -> Optional[Dict[str, Any]]:
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
            
            labels_file = os.path.join(output_path, 'qc_results', 'masks.h5')
            if not os.path.exists(labels_file):
                self.logger.error(f"SID_IMG_PRE: Missing labels file at {labels_file}")
                return None
            
            with h5py.File(labels_file, 'r') as f:
                # Load into RAM
                labels_data['roi_labels'] = f['labels'][:]
                    
                self.logger.info(f"SID_IMG_PRE: Loaded ROI labels - shape: {labels_data['roi_labels'].shape}")
                
                # Count excitatory/inhibitory (works with both memmap and regular arrays)
                n_exc = np.sum(labels_data['roi_labels'] == -1)
                n_inh = np.sum(labels_data['roi_labels'] == 1) 
                n_unlabeled = np.sum(labels_data['roi_labels'] == 0)
                
                labels_data['n_excitatory'] = n_exc
                labels_data['n_inhibitory'] = n_inh
                labels_data['n_unlabeled'] = n_unlabeled
                labels_data['n_total'] = len(labels_data['roi_labels'])
                
                self.logger.info(f"SID_IMG_PRE: ROI distribution - {n_exc} excitatory, {n_inh} inhibitory, {n_unlabeled} unlabeled")
            
            return labels_data
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to load ROI labels for {subject_id}: {e}")
            return None

    # detect the rising edge and falling edge of binary series.
    def get_trigger_time(self,
            vol_time,
            vol_bin
            ):
        # find the edge with np.diff and correct it by preappend one 0.
        diff_vol = np.diff(vol_bin, prepend=0)
        idx_up = np.where(diff_vol == 1)[0]
        idx_down = np.where(diff_vol == -1)[0]
        # select the indice for risging and falling.
        # give the edges in ms.
        time_up   = vol_time[idx_up]
        time_down = vol_time[idx_down]
        return time_up, time_down

    # TODO move spike derivation to suite2p loading
    # def derive_spikes_from_imaging(self, imaging_data, tau=0.25):
    #     """
    #     Derive spikes (Suite2p-style OASIS) from in-memory imaging data.

    #     Parameters
    #     ----------
    #     imaging_data : dict with keys
    #         'fluo_traces' : np.ndarray, shape (n_roi, n_time)
    #             Raw fluorescence F (after registration; not dF/F).
    #         'neuropil'    : np.ndarray, shape (n_roi, n_time)
    #             Neuropil fluorescence Fneu.
    #         'neucoeff'    : float or np.ndarray
    #             Neuropil coefficient(s). Scalar or per-ROI vector (n_roi,).
    #         'imaging_fs'  : float
    #             Imaging frame rate in Hz.

    #     tau : float
    #         Calcium decay constant in seconds (e.g., 0.7 for GCaMP6s, ~0.25 for 6f).

    #     Returns
    #     -------
    #     spks : np.ndarray, shape (n_roi, n_time)
    #         Deconvolved spike events (nonnegative). Same format as Suite2p spks.npy.
    #     params : dict
    #         {'tau': float, 'fs': float, 'neucoeff_shape': shape, 'method': str, 'g': float}
    #     """

    #     # ---- unpack & sanity checks ----
    #     F = np.asarray(imaging_data['fluo_traces'])
    #     Fneu = np.asarray(imaging_data['neuropil'])
    #     neucoeff = imaging_data['neucoeff']
    #     fs = float(imaging_data['imaging_fs'])

    #     if F.shape != Fneu.shape:
    #         raise ValueError(f"F and Fneu must match shape; got {F.shape} vs {Fneu.shape}")

    #     n_roi, n_time = F.shape

    #     # ---- neuropil subtraction (scalar or per-ROI coeff) ----
    #     if np.ndim(neucoeff) == 0:
    #         neucoeff = float(neucoeff)
    #         Fcorr = F - neucoeff * Fneu
    #     else:
    #         neucoeff = np.asarray(neucoeff).reshape(-1)
    #         if neucoeff.shape[0] != n_roi:
    #             raise ValueError(f"neucoeff length {neucoeff.shape[0]} != n_roi {n_roi}")
    #         Fcorr = F - neucoeff[:, None] * Fneu

    #     # ---- convert tau -> AR(1) decay g ----
    #     # AR(1): c_t = g*c_{t-1} + s_t ; g = exp(-1 / (tau * fs))
    #     g = float(np.exp(-1.0 / (tau * fs)))

    #     # ---- try deconvolution backends ----
    #     spks = np.zeros_like(Fcorr, dtype=np.float64)
    #     method = None

    #     # Helper: minimal AR(1) deconv in case neither backend is available
    #     def _ar1_deconv_fallback(y, g_val, smin=0.0):
    #         y = np.asarray(y, dtype=np.float64).copy()
    #         # Robust baseline subtraction (similar spirit to OASIS)
    #         base = np.nanpercentile(y, 8)
    #         y -= base
    #         T = y.size
    #         c = 0.0
    #         out = np.zeros(T, dtype=np.float64)
    #         for t in range(T):
    #             c *= g_val
    #             resid = y[t] - c
    #             st = resid if resid > 0 else 0.0
    #             if st > smin:
    #                 out[t] = st - smin
    #                 c += out[t]
    #         return out

    #     # Backend 1: Suite2p's OASIS wrapper (fast, if available)
    #     try:
    #         from suite2p.dcnv import oasisc  # type: ignore
    #         self.logger.info(f"S2P_DFF: Using Suite2p's OASIS for deconvolution")
    #         method = 'suite2p'
    #         for i in range(n_roi):
    #             y = Fcorr[i].astype(np.float64)
    #             # smin=0: nonnegative L0 deconvolution; matches Suite2p defaults closely
    #             _, _, s, _, _ = oasisc(y, g=[g], smin=0)
    #             spks[i] = s
    #     except Exception:
    #         # Backend 2: oasis package
    #         try:
    #             from oasis.functions import deconvolve  # type: ignore
    #             self.logger.info(f"S2P_DFF: Using oasis for deconvolution")
    #             method = 'oasis'
    #             for i in range(n_roi):
    #                 y = Fcorr[i].astype(np.float64)
    #                 # penalty=0 for L0-like; smin=0 nonnegative constraint
    #                 _, s, _, _, _ = deconvolve(y, g=[g], smin=0, penalty=0)
    #                 spks[i] = s
    #         except Exception:
    #             # Backend 3: minimal AR(1) fallback
    #             self.logger.warning(f"S2P_DFF: Failed to deconvolve using {method}, falling back to AR(1)")
    #             method = 'ar1_fallback'
    #             for i in range(n_roi):
    #                 spks[i] = _ar1_deconv_fallback(Fcorr[i], g)

    #     params = {
    #         'tau': float(tau),
    #         'fs': fs,
    #         'neucoeff_shape': (np.asarray(neucoeff).shape if np.ndim(neucoeff) else ()),
    #         'method': method,
    #         'g': g,
    #     }
    #     return spks, params

    # -----------------------
    # Example usage:
    # -----------------------
    # imaging_data = {
    #     'fluo_traces': F,        # (n_roi, n_time)
    #     'neuropil': Fneu,        # (n_roi, n_time)
    #     'neucoeff': 0.7,         # or np.array of shape (n_roi,)
    #     'imaging_fs': 30.0       # Hz
    # }
    # spks, params = derive_spikes_from_imaging(imaging_data, tau=0.7)
    # print(params['method'], params)

    def load_spike_traces(self, output_path: str, subject_id: str) -> Optional[Dict[str, Any]]:
        """
        Load spike traces from the output directory.

        Args:
            output_path: Path to processed output directory
            subject_id: Subject identifier for logging

        Returns:
            Dictionary containing spike trace data or None if failed
        """
        try:
            spike_data = {}

            # Load spike traces from DFF module
            spike_file = os.path.join(output_path, 'spks.h5')
            if not os.path.exists(spike_file):
                self.logger.error(f"SID_IMG_PRE: Missing spike file at {spike_file}")
                return None

            with h5py.File(spike_file, 'r') as f:
                spike_data['spks'] = f['spks'][:]
                self.logger.info(f"SID_IMG_PRE: Loaded spike traces - shape: {spike_data['spks'].shape}")

            return spike_data

        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to load spike traces for {subject_id}: {e}")
            return None

    # def load_dff_traces(self, output_path: str, subject_id: str, voltage_data: Dict[str, Any], use_memmap: bool = True) -> Optional[Dict[str, Any]]:
    def load_dff_traces(self, output_path: str, subject_id: str) -> Optional[Dict[str, Any]]:
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
                self.logger.error(f"SID_IMG_PRE: Missing DFF file at {dff_file}")
                return None
            
            with h5py.File(dff_file, 'r') as f:
                # Load into RAM
                dff_data['dff_traces'] = f['dff'][:]
                self.logger.info(f"SID_IMG_PRE: Loaded DFF traces - shape: {dff_data['dff_traces'].shape}")

            # Load imaging timestamps from ops
            ops_file = os.path.join(output_path, 'ops.npy')
            if not os.path.exists(ops_file):
                self.logger.error(f"SID_IMG_PRE: Missing ops file at {ops_file}")
                return None
            
            # ops = np.load(ops_file, allow_pickle=True).item()
            
            # Extract timing information
            # dff_data['imaging_fs'] = ops.get('imaging_fs', 30.0)  # Imaging sampling rate, ops fs inaccurate
            # looks like dff and vol have matching duration, however use scope voltage trace to get imaging time vector
            # dff_data['imaging_fs'] = 29.760441593958042
            n_frames = dff_data['dff_traces'].shape[1]
            dff_data['n_frames'] = n_frames
            # imaging_time_array = np.arange(n_frames) / dff_data['imaging_fs']
            
            # # get vol recording time and scope imaging vector
            # vol_time = voltage_data.get('vol_time', None)
            # vol_img = voltage_data.get('vol_img', None)
            
            # # get scope imgaging signal trigger time stamps.
            # imaging_time_array, _   = self.get_trigger_time(vol_time, vol_img)            
                        
            # # Adjust for photon integration cycle delay
            # imaging_time_array = self.adjust_imaging_timing_for_integration_delay(imaging_time_array)
            # dff_data['imaging_fs'] = n_frames / (imaging_time_array[-1] - imaging_time_array[0])  # Recalculate fs based on timing

            # if use_memmap:
            #     # Create memmap for imaging time
            #     dff_data['imaging_time'] = create_memmap(
            #         imaging_time_array, 'float32',
            #         os.path.join(memmap_dir, 'imaging_time.mmap'), self.logger)
            # else:
            #     dff_data['imaging_time'] = imaging_time_array
                    
            # self.logger.info(f"SID_IMG_PRE: Imaging sampling rate: {dff_data['imaging_fs']} Hz")
            # self.logger.info(f"SID_IMG_PRE: Imaging duration: {dff_data['imaging_time'][-1]:.1f} seconds")
            self.logger.info(f"SID_IMG_PRE: Number of frames: {n_frames}")
            
            return dff_data
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to load DFF traces for {subject_id}: {e}")
            return None

    # def load_voltage_data(self, output_path: str, subject_id: str, use_memmap: bool = True) -> Optional[Dict[str, Any]]:
    def load_voltage_data(self, output_path: str, subject_id: str) -> Optional[Dict[str, Any]]:
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
                self.logger.warning(f"SID_IMG_PRE: No voltage file found at {voltage_file}")
                return self._get_empty_voltage_structure()
            
            self.logger.info(f"SID_IMG_PRE: Found voltage file at {voltage_file}")
            
            with h5py.File(voltage_file, 'r') as f:
                # Log available datasets
                self.logger.info(f"SID_IMG_PRE: Voltage file contains datasets: {list(f.keys())}")
                
                # Load from 'raw' group
                if 'raw' not in f:
                    self.logger.error("SID_IMG_PRE: 'raw' group not found in voltage file")
                    return None
                
                raw_group = f['raw']
                self.logger.info(f"SID_IMG_PRE: 'raw' group contains: {list(raw_group.keys())}")
                
                # Load timing information first
                if 'vol_time' not in raw_group:
                    self.logger.error("SID_IMG_PRE: 'vol_time' not found in voltage file")
                    return None
                
                # vol recording reports time in ms, convert to s
                ms_per_s = 1000
                voltage_data['vol_time'] = raw_group['vol_time'][:] / ms_per_s
                # calculate duration and sampling rate
                voltage_data['vol_n_samples'] = len(voltage_data['vol_time'])
                voltage_data['vol_duration'] = voltage_data['vol_time'][-1]                
                voltage_data['voltage_fs'] = voltage_data['vol_n_samples'] / voltage_data['vol_duration']


                # voltage_data['vol_time'] = voltage_data['vol_time'] / voltage_data['voltage_fs']
                self.logger.info(f"SID_IMG_PRE: Loaded voltage time - {voltage_data['vol_n_samples']} samples at {voltage_data['voltage_fs']:.1f} Hz")
                self.logger.info(f"SID_IMG_PRE: Voltage duration: {voltage_data['vol_duration']:.1f} seconds")
                
                # Load voltage channels directly to named variables
                # Load directly to RAM
                voltage_data['vol_start'] = raw_group['vol_start'][:] if 'vol_start' in raw_group else None
                voltage_data['vol_stim_vis'] = raw_group['vol_stim_vis'][:] if 'vol_stim_vis' in raw_group else None
                # voltage_data['vol_hifi'] = raw_group['vol_hifi'][:] if 'vol_hifi' in raw_group else None
                voltage_data['vol_img'] = raw_group['vol_img'][:] if 'vol_img' in raw_group else None
                voltage_data['vol_stim_aud'] = raw_group['vol_stim_aud'][:] if 'vol_stim_aud' in raw_group else None
                # voltage_data['vol_flir'] = raw_group['vol_flir'][:] if 'vol_flir' in raw_group else None
                # voltage_data['vol_pmt'] = raw_group['vol_pmt'][:] if 'vol_pmt' in raw_group else None
                # voltage_data['vol_led'] = raw_group['vol_led'][:] if 'vol_led' in raw_group else None
                
                # Count loaded channels
                loaded_channels = [ch for ch in ['vol_start', 'vol_stim_vis', 'vol_hifi', 'vol_img', 
                                               'vol_stim_aud', 'vol_flir', 'vol_pmt', 'vol_led'] 
                                 if voltage_data.get(ch) is not None]
                
                self.logger.info(f"SID_IMG_PRE: Loaded {len(loaded_channels)} voltage channels: {loaded_channels}")            
            
            return voltage_data
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to load voltage data for {subject_id}: {e}")
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

    # def load_imaging_data(self, subject_id: str, suite2p_path: str, output_path: str, behavioral_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    def load_imaging_data(self, subject_id: str, suite2p_path: str, output_path: str) -> Optional[Dict[str, Any]]:
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
            
            self.logger.info(f"SID_IMG_PRE: Loading imaging data for {subject_id}")
            
            # use_memmap = False
            
            # Load ROI labels
            # labels_data = self.load_roi_labels(output_path, subject_id, use_memmap=use_memmap)
            # labels_data = self.load_roi_labels(output_path, subject_id)
            # if labels_data is None:
            #     return None
            # data.update(labels_data)

            # Load voltage recordings
            # voltage_data = self.load_voltage_data(output_path, subject_id, use_memmap=use_memmap)
            voltage_data = self.load_voltage_data(output_path, subject_id)
            if voltage_data is None:
                return None
            data.update(voltage_data)

            # Load DFF traces and timestamps, use voltage scope channel for timing
            # dff_data = self.load_dff_traces(output_path, subject_id, voltage_data, use_memmap=use_memmap)
            dff_data = self.load_dff_traces(output_path, subject_id)
            if dff_data is None:
                return None
            data.update(dff_data)

            spike_data = self.load_spike_traces(output_path, subject_id)
            if spike_data is None:
                return None
            data.update(spike_data)

            # # print min/max of each voltage channel
            # for ch_name in ['vol_start', 'vol_stim_vis', 'vol_hifi', 'vol_img', 
            #                 'vol_stim_aud', 'vol_flir', 'vol_pmt', 'vol_led']:
            #     ch_data = voltage_data.get(ch_name)
            #     if ch_data is not None:
            #         self.logger.info(f"SID_IMG_PRE: {ch_name} - min: {np.min(ch_data)}, max: {np.max(ch_data)}")
            
            # # get vol recording time and scope imaging vector
            # vol_time = voltage_data.get('vol_time', None)
            # vol_img = voltage_data.get('vol_img', None)
            
            # # get scope imgaging signal trigger time stamps.
            # imaging_time_array, _   = self.get_trigger_time(vol_time, vol_img)            
                        
            # # Adjust for photon integration cycle delay
            # imaging_time_array = self.adjust_imaging_timing_for_integration_delay(imaging_time_array)
            # n_frames = dff_data['n_frames']
            # dff_data['imaging_fs'] = n_frames / (imaging_time_array[-1] - imaging_time_array[0])  # Recalculate fs based on timing
            # self.logger.info(f"SID_IMG_PRE: Imaging sampling rate: {dff_data['imaging_fs']} Hz")

            # dff_data['imaging_time'] = imaging_time_array  
            # self.logger.info(f"SID_IMG_PRE: Imaging duration: {dff_data['imaging_time'][-1]:.1f} seconds")          
            # data.update(dff_data)
            
            # # TODO Add method to align traces to first vol_start pulse
            # # print number of vol_start peaks
            # if voltage_data.get('vol_start') is not None:
            #     # Find first index where vol_start goes high (threshold crossing)
            #     threshold = 0.5
            #     high_indices = np.where(voltage_data['vol_start'] > threshold)[0]
                
            #     if len(high_indices) > 0:
            #         # Find transitions from low to high by looking for gaps in indices
            #         pulse_starts = [high_indices[0]]  # First high point
            #         # for i in range(1, len(high_indices)):
            #         #     if high_indices[i] - high_indices[i-1] > 1:  # Gap indicates new pulse
            #         #         pulse_starts.append(high_indices[i])
                    
            #         # self.logger.info(f"SID_IMG_PRE: vol_start - detected {len(pulse_starts)} start pulse rising edges")
                    
            #         # Align time vectors to first vol_start pulse rising edge (t=0)
            #         if len(pulse_starts) > 0:
            #             first_pulse_idx = pulse_starts[0]
            #             first_pulse_time = data['vol_time'][first_pulse_idx]
                        
            #             # Shift voltage time to zero at first pulse
            #             data['vol_time'] = data['vol_time'] - first_pulse_time
                        
            #             # Shift imaging time to same reference point
            #             data['imaging_time'] = data['imaging_time'] - first_pulse_time
                        
            #             self.logger.info(f"SID_IMG_PRE: Aligned time vectors to first vol_start rising edge at original time {first_pulse_time:.3f}s")
            #             self.logger.info(f"SID_IMG_PRE: New imaging duration: {data['imaging_time'][-1]:.1f}s")
            #             self.logger.info(f"SID_IMG_PRE: New voltage duration: {data['vol_time'][-1]:.1f}s")
                        
            #         else:
            #             self.logger.warning(f"SID_IMG_PRE: No vol_start pulse rising edges found - cannot align time vectors")
            #     else:
            #         self.logger.warning(f"SID_IMG_PRE: No vol_start values above threshold - cannot align time vectors")
                    
            
            
            self.logger.info(f"SID_IMG_PRE: Successfully loaded imaging data for {subject_id}")
            return data
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to load imaging data for {subject_id}: {e}")
            return None


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
                self.logger.warning(f"SID_IMG_PRE: No voltage data in search window around {expected_time:.3f}s")
                return expected_time  # Return original if no data found
            
            # Extract voltage segment and time segment
            vol_segment = imaging_data['vol_start'][start_idx:end_idx]
            time_segment = imaging_data['vol_time'][start_idx:end_idx]
            
            # Find pulse rising edges in this segment
            threshold = 0.5
            high_indices = np.where(vol_segment > threshold)[0]
            
            if len(high_indices) == 0:
                self.logger.warning(f"SID_IMG_PRE: No vol_start pulse found in search window around {expected_time:.3f}s")
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
                self.logger.debug(f"SID_IMG_PRE: Trial pulse drift: {drift*1000:.1f}ms at t={expected_time:.3f}s")
            
            return closest_time
            
        except Exception as e:
            self.logger.warning(f"SID_IMG_PRE: Error finding closest vol pulse at {expected_time:.3f}s: {e}")
            return expected_time    
    
    def clean_output(self, subject_id: str, output_path: str) -> None:
        """
        Clean up output directory for a subject by removing existing memmap files.
        """
        self.logger.info(f"SID_IMG_PRE: Cleaning output for {subject_id}...")
        clean_memmap_path(output_path, 'sid_imaging', self.logger)
        self.logger.info(f"SID_IMG_PRE: Output cleaned for {subject_id}")
        return None        
        

    # def create_memmap_data(self, imaging_data: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    #     """
    #     Convert imaging data to memory-mapped arrays for efficient access and Spyder inspection.
        
    #     Args:
    #         imaging_data: Dictionary containing loaded imaging data
    #         output_path: Base output path for memmap files
            
    #     Returns:
    #         Dictionary with memory-mapped versions of the data
    #     """
    #     try:
    #         self.logger.info("SID_IMG_PRE: Creating memory-mapped arrays for development...")
            
    #         # Get memmap directory with logger
    #         memmap_dir = get_memmap_path(output_path, 'sid_imaging', self.logger)
            
    #         memmap_data = imaging_data.copy()
            
    #         # Convert DFF traces to memmap
    #         if 'dff_traces' in imaging_data:
    #             memmap_data['dff_traces'] = create_memmap(
    #                 imaging_data['dff_traces'], 'float32',
    #                 os.path.join(memmap_dir, 'dff_traces.mmap'), self.logger)
                
    #         # Convert fluorescence traces to memmap  
    #         if 'fluo_traces' in imaging_data:
    #             memmap_data['fluo_traces'] = create_memmap(
    #                 imaging_data['fluo_traces'], 'float32',
    #                 os.path.join(memmap_dir, 'fluo_traces.mmap'), self.logger)
            
    #         # Convert voltage channels to memmap
    #         if 'voltage_channels' in imaging_data and imaging_data['voltage_channels']:
    #             memmap_voltage_channels = {}
    #             for ch_name, ch_data in imaging_data['voltage_channels'].items():
    #                 # Determine appropriate dtype based on channel
    #                 if 'time' in ch_name or 'stim_aud' in ch_name:
    #                     dtype = 'float32'
    #                 else:
    #                     dtype = 'int8'
                    
    #                 memmap_voltage_channels[ch_name] = create_memmap(
    #                     ch_data, dtype,
    #                     os.path.join(memmap_dir, f'{ch_name}.mmap'), self.logger)
                
    #             memmap_data['voltage_channels'] = memmap_voltage_channels
            
    #         # Convert time vectors to memmap
    #         if 'imaging_time' in imaging_data:
    #             memmap_data['imaging_time'] = create_memmap(
    #                 imaging_data['imaging_time'], 'float32',
    #                 os.path.join(memmap_dir, 'imaging_time.mmap'), self.logger)
                
    #         if 'voltage_time' in imaging_data and imaging_data['voltage_time'] is not None:
    #             memmap_data['voltage_time'] = create_memmap(
    #                 imaging_data['voltage_time'], 'float32',
    #                 os.path.join(memmap_dir, 'voltage_time.mmap'), self.logger)
            
    #         self.logger.info(f"SID_IMG_PRE: Created memory-mapped arrays in {memmap_dir}")
    #         self.logger.info("SID_IMG_PRE: Arrays now available for Spyder variable inspector")
            
    #         return memmap_data
            
    #     except Exception as e:
    #         self.logger.error(f"SID_IMG_PRE: Failed to create memory-mapped data: {e}")
    #         return imaging_data  # Return original data if memmap creation fails

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
        
        self.logger.debug(f"SID_IMG_PRE: Adjusted imaging timing by {mean_frame_interval/2.0:.6f}s for integration delay")
        
        return adjusted_timing


    # def trim_sequences_to_common_length(self, sequences: List[np.ndarray], data_type: str) -> np.ndarray:
    #     """
    #     Trim sequences to common length (similar to trim_seq from Alignment.py).
        
    #     Args:
    #         sequences: List of arrays to trim
    #         data_type: Type of data for logging
            
    #     Returns:
    #         Array with trimmed sequences of common length
    #     """
    #     if not sequences:
    #         return np.array([])
        
    #     # Find minimum length
    #     min_length = min(seq.shape[-1] for seq in sequences)
        
    #     # Trim all sequences to minimum length
    #     if sequences[0].ndim == 1:
    #         # 1D sequences (time vectors)
    #         trimmed = [seq[:min_length] for seq in sequences]
    #     else:
    #         # 2D sequences (DFF traces)
    #         trimmed = [seq[:, :min_length] for seq in sequences]
        
    #     self.logger.debug(f"SID_IMG_PRE: Trimmed {len(sequences)} {data_type} sequences to length {min_length}")
        
    #     return np.array(trimmed)
    

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
            
            self.logger.info(f"SID_IMG_PRE: Creating comprehensive alignment check ({alignment_stage})")
            
            # Plot 1: Overview showing first 3 trials with DFF/voltage/behavioral events
            self._plot_session_overview_with_trials(imaging_data, df_trials, subject_id, alignment_stage)
            
            # Plot 2: Individual trial plots for first 10 trials
            self._plot_individual_trials(imaging_data, df_trials, subject_id, alignment_stage)
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create comprehensive alignment check: {e}")

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
            self.logger.error(f"SID_IMG_PRE: Failed to create session overview plot: {e}")

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
            self.logger.error(f"SID_IMG_PRE: Failed to create individual trial plots: {e}")

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
                
                if len(vol_times) == len(beh_times):
                    time_diffs = np.abs(vol_times - beh_times)
                    metrics['Max Time Diff (s)'] = np.max(time_diffs)
                    metrics['Mean Time Diff (s)'] = np.mean(time_diffs)
                    metrics['Sync Quality'] = 1.0 if np.max(time_diffs) < 0.05 else 0.5
                else:
                    metrics['Count Mismatch'] = abs(len(vol_times) - len(beh_times))
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
                                        alignment: str = 'trial_start', 
                                        pre_sec: float = 2.0, post_sec: float = 8.0) -> Dict[str, Any]:
        """
        Simple trial segmentation function that stores segments directly in DataFrame rows.
        
        Args:
            imaging_data: Dictionary containing continuous imaging data
            df_trials: Behavioral trial metadata
            alignment: Event to align to relative to trial start
            pre_sec: Seconds before alignment point
            post_sec: Seconds after alignment point
            
        Returns:
            Dictionary with enhanced DataFrame and metadata
        """
        try:
            self.logger.info(f"SID_IMG: Extracting trial segments aligned to '{alignment}'")
            
            # Work with a copy to avoid modifying original
            df_enhanced = df_trials.copy()
            
            # Initialize new columns for segment data
            df_enhanced['dff_segment'] = None
            df_enhanced['spike_segment'] = None
            df_enhanced['dff_time_vector'] = None
            df_enhanced['vol_segments'] = None
            df_enhanced['vol_time_vectors'] = None
            df_enhanced['segment_valid'] = False
            df_enhanced['segment_duration'] = np.nan
            df_enhanced['drop_reason'] = None
            
            dropped_trials = []
            vol_channels = ['vol_start', 'vol_stim_vis', 'vol_stim_aud']
            
            # First pass: extract all segments without trimming
            temp_dff_segments = []
            temp_spike_segments = []
            temp_dff_time_vectors = []
            temp_vol_segments = {ch: [] for ch in vol_channels}
            temp_vol_time_vectors = {ch: [] for ch in vol_channels}
            valid_trial_indices = []
            
            # Extract segments for each trial
            for idx, (trial_idx, row) in enumerate(df_enhanced.iterrows()):
                # Always use trial_start_timestamp as the fundamental reference
                trial_start_time = row.get('trial_start_timestamp', np.nan)
                
                if pd.isna(trial_start_time):
                    df_enhanced.loc[trial_idx, 'drop_reason'] = 'missing_trial_start_timestamp'
                    dropped_trials.append((trial_idx, 'missing_trial_start_timestamp'))
                    continue
                
                # Calculate actual alignment time
                alignment_event_time = row.get(alignment, np.nan)
                if pd.isna(alignment_event_time):
                    df_enhanced.loc[trial_idx, 'drop_reason'] = f'missing_{alignment}_time'
                    dropped_trials.append((trial_idx, f'missing_{alignment}_time'))
                    continue
                
                align_time = trial_start_time + alignment_event_time  # Event times are already absolute timestamps
                
                # Define window around alignment point
                window_start = align_time - pre_sec
                window_end = align_time + post_sec
                required_duration = pre_sec + post_sec
                
                # Extract DFF segment
                img_start_idx = np.searchsorted(imaging_data['imaging_time'], window_start)
                img_end_idx = np.searchsorted(imaging_data['imaging_time'], window_end)

                # Validate segment has sufficient data coverage
                if img_start_idx < img_end_idx and img_end_idx <= len(imaging_data['imaging_time']):
                    # Check actual time coverage
                    actual_start_time = imaging_data['imaging_time'][img_start_idx]
                    actual_end_time = imaging_data['imaging_time'][img_end_idx - 1]
                    actual_duration = actual_end_time - actual_start_time
                    
                    # Ensure we have at least 85% of the requested window duration
                    if actual_duration >= 0.75 * required_duration:
                        # Store temporary segments
                        dff_segment = imaging_data['dff_traces'][:, img_start_idx:img_end_idx]
                        spike_segment = imaging_data['spks'][:, img_start_idx:img_end_idx]
                        dff_time_vector = imaging_data['imaging_time'][img_start_idx:img_end_idx] - align_time
                        
                        temp_dff_segments.append(dff_segment)
                        temp_dff_time_vectors.append(dff_time_vector)
                        temp_spike_segments.append(spike_segment)
                        valid_trial_indices.append(trial_idx)
                        
                        df_enhanced.at[trial_idx, 'segment_duration'] = actual_duration
                        df_enhanced.at[trial_idx, 'segment_valid'] = True
                        
                        # Extract voltage segments
                        vol_start_idx = np.searchsorted(imaging_data['vol_time'], window_start)
                        vol_end_idx = np.searchsorted(imaging_data['vol_time'], window_end)
                        
                        if vol_start_idx < vol_end_idx and vol_end_idx <= len(imaging_data['vol_time']):
                            vol_time_vector = imaging_data['vol_time'][vol_start_idx:vol_end_idx] - align_time
                            
                            for ch in vol_channels:
                                if imaging_data.get(ch) is not None:
                                    vol_segment = imaging_data[ch][vol_start_idx:vol_end_idx]
                                    temp_vol_segments[ch].append(vol_segment)
                                    temp_vol_time_vectors[ch].append(vol_time_vector)
                                else:
                                    temp_vol_segments[ch].append(None)
                                    temp_vol_time_vectors[ch].append(None)
                        else:
                            # No voltage data for this trial
                            for ch in vol_channels:
                                temp_vol_segments[ch].append(None)
                                temp_vol_time_vectors[ch].append(None)
                        # align df event times
                        if alignment != 'trial_start':
                            df_enhanced.at[trial_idx, 'trial_start'] = df_enhanced.at[trial_idx, 'trial_start'] - align_time
                            df_enhanced.at[trial_idx, 'start_flash_1'] = df_enhanced.at[trial_idx, 'start_flash_1'] - align_time
                            df_enhanced.at[trial_idx, 'end_flash_1'] = df_enhanced.at[trial_idx, 'end_flash_1'] - align_time
                            df_enhanced.at[trial_idx, 'start_flash_2'] = df_enhanced.at[trial_idx, 'start_flash_2'] - align_time
                            df_enhanced.at[trial_idx, 'end_flash_2'] = df_enhanced.at[trial_idx, 'end_flash_2'] - align_time
                            df_enhanced.at[trial_idx, 'servo_in'] = df_enhanced.at[trial_idx, 'servo_in'] - align_time
                            df_enhanced.at[trial_idx, 'choice_start'] = df_enhanced.at[trial_idx, 'choice_start'] - align_time
                            df_enhanced.at[trial_idx, 'choice_stop'] = df_enhanced.at[trial_idx, 'choice_stop'] - align_time
                            df_enhanced.at[trial_idx, 'lick_start'] = df_enhanced.at[trial_idx, 'lick_start'] - align_time
                            df_enhanced.at[trial_idx, 'reward_start'] = df_enhanced.at[trial_idx, 'reward_start'] - align_time
                            df_enhanced.at[trial_idx, 'reward_stop'] = df_enhanced.at[trial_idx, 'reward_stop'] - align_time
                            df_enhanced.at[trial_idx, 'punish_start'] = df_enhanced.at[trial_idx, 'punish_start'] - align_time
                            df_enhanced.at[trial_idx, 'punish_stop'] = df_enhanced.at[trial_idx, 'punish_stop'] - align_time
                            df_enhanced.at[trial_idx, 'servo_out'] = df_enhanced.at[trial_idx, 'servo_out'] - align_time
                            df_enhanced.at[trial_idx, 'trial_stop'] = df_enhanced.at[trial_idx, 'trial_stop'] - align_time

                    else:
                        df_enhanced.loc[trial_idx, 'drop_reason'] = f'insufficient_coverage_{actual_duration:.2f}s'
                        dropped_trials.append((trial_idx, f'insufficient_coverage_{actual_duration:.2f}s'))
                else:
                    df_enhanced.loc[trial_idx, 'drop_reason'] = 'outside_data_bounds'
                    dropped_trials.append((trial_idx, 'outside_data_bounds'))
            
            # Log dropped trials
            if dropped_trials:
                self.logger.warning(f"SID_IMG: Dropped {len(dropped_trials)} trials due to insufficient data:")
                for trial_idx, reason in dropped_trials:
                    self.logger.warning(f"  Trial {trial_idx}: {reason}")
            
            # Second pass: trim all segments to common length
            if temp_dff_segments:
                # Trim DFF segments to common length
                trimmed_dff_segments = self.trim_sequences_to_common_length(temp_dff_segments, 'dff')
                trimmed_spike_segments = self.trim_sequences_to_common_length(temp_spike_segments, 'spks')
                trimmed_dff_time_vectors = self.trim_sequences_to_common_length(temp_dff_time_vectors, 'dff_time')
                
                # Get common time vector (all should be identical after trimming)
                common_dff_time_vector = trimmed_dff_time_vectors[0]
                
                # Trim voltage segments to common length
                trimmed_vol_segments = {}
                trimmed_vol_time_vectors = {}
                common_vol_time_vectors = {}
                
                for ch in vol_channels:
                    # Filter out None values for trimming
                    valid_vol_segments = [seg for seg in temp_vol_segments[ch] if seg is not None]
                    valid_vol_time_vectors = [tvec for tvec in temp_vol_time_vectors[ch] if tvec is not None]
                    
                    if valid_vol_segments:
                        trimmed_vol_segments[ch] = self.trim_sequences_to_common_length(valid_vol_segments, f'vol_{ch}')
                        trimmed_vol_time_vectors[ch] = self.trim_sequences_to_common_length(valid_vol_time_vectors, f'vol_{ch}_time')
                        common_vol_time_vectors[ch] = trimmed_vol_time_vectors[ch][0]
                    else:
                        trimmed_vol_segments[ch] = []
                        trimmed_vol_time_vectors[ch] = []
                        common_vol_time_vectors[ch] = common_dff_time_vector  # Use DFF time as fallback
                

                # Third pass: store trimmed segments back in DataFrame
                for i, trial_idx in enumerate(valid_trial_indices):
                    # Store trimmed DFF data
                    df_enhanced.at[trial_idx, 'dff_segment'] = trimmed_dff_segments[i]
                    df_enhanced.at[trial_idx, 'spike_segment'] = trimmed_spike_segments[i]
                    df_enhanced.at[trial_idx, 'dff_time_vector'] = common_dff_time_vector
                    
                    # Store trimmed voltage data
                    vol_segments_dict = {}
                    vol_time_vectors_dict = {}
                    
                    for ch in vol_channels:
                        if len(trimmed_vol_segments[ch]) > 0:
                            # Simple indexing: trial i gets voltage segment i (if it exists)
                            # Count how many non-None voltage segments exist up to position i
                            valid_vol_count = sum(1 for seg in temp_vol_segments[ch][:i+1] if seg is not None)
                            vol_idx = valid_vol_count - 1  # Convert to 0-based index
                            
                            if 0 <= vol_idx < len(trimmed_vol_segments[ch]):
                                vol_segments_dict[ch] = trimmed_vol_segments[ch][vol_idx]
                                vol_time_vectors_dict[ch] = common_vol_time_vectors[ch]
                            else:
                                vol_segments_dict[ch] = None
                                vol_time_vectors_dict[ch] = common_dff_time_vector
                        else:
                            vol_segments_dict[ch] = None
                            vol_time_vectors_dict[ch] = common_dff_time_vector
                    
                    df_enhanced.at[trial_idx, 'vol_segments'] = vol_segments_dict
                    df_enhanced.at[trial_idx, 'vol_time_vectors'] = vol_time_vectors_dict

                
                # Get valid trials mask
                valid_trials = df_enhanced['segment_valid']
                n_valid = np.sum(valid_trials)
                
                self.logger.info(f"SID_IMG: Successfully extracted and trimmed segments for {n_valid}/{len(df_enhanced)} trials")
                self.logger.info(f"SID_IMG: Common DFF segment shape: {trimmed_dff_segments[0].shape}")
                self.logger.info(f"SID_IMG: Common time vector length: {len(common_dff_time_vector)}")
                
                return {
                    'df_trials_with_segments': df_enhanced,
                    'valid_trials_mask': valid_trials,
                    'n_valid_trials': n_valid,
                    'alignment_point': alignment,
                    'window': {'pre_sec': pre_sec, 'post_sec': post_sec},
                    'dropped_trials': dropped_trials,
                    
                    # Convenience arrays for analysis (now guaranteed to be homogeneous)
                    'dff_segments_array': trimmed_dff_segments,
                    'spike_segments_array': trimmed_spike_segments,
                    'common_time_vector': common_dff_time_vector,
                    'common_vol_time_vectors': common_vol_time_vectors
                }
            else:
                return {'error': 'No valid trial segments extracted'}
                
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to extract trial segments: {e}")
            return {'error': str(e)}



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


    # def extract_trial_segments_simple(self, imaging_data: Dict[str, Any], df_trials: pd.DataFrame,
    #                                 alignment: str = 'trial_start', 
    #                                 pre_sec: float = 2.0, post_sec: float = 8.0) -> Dict[str, Any]:
    #     """
    #     Simple trial segmentation function that stores segments directly in DataFrame rows.
        
    #     Args:
    #         imaging_data: Dictionary containing continuous imaging data
    #         df_trials: Behavioral trial metadata
    #         alignment: Event to align to relative to trial start
    #         pre_sec: Seconds before alignment point
    #         post_sec: Seconds after alignment point
            
    #     Returns:
    #         Dictionary with enhanced DataFrame and metadata
    #     """
    #     try:
    #         self.logger.info(f"SID_IMG_PRE: Extracting trial segments aligned to '{alignment}'")
            
    #         # Work with a copy to avoid modifying original
    #         df_enhanced = df_trials.copy()
            
    #         # Initialize new columns for segment data
    #         df_enhanced['dff_segment'] = None
    #         df_enhanced['dff_time_vector'] = None
    #         df_enhanced['vol_segments'] = None
    #         df_enhanced['vol_time_vectors'] = None
    #         df_enhanced['segment_valid'] = False
    #         df_enhanced['segment_duration'] = np.nan
    #         df_enhanced['drop_reason'] = None
            
    #         dropped_trials = []
    #         vol_channels = ['vol_start', 'vol_stim_vis', 'vol_stim_aud']
            
    #         # First pass: extract all segments without trimming
    #         temp_dff_segments = []
    #         temp_dff_time_vectors = []
    #         temp_vol_segments = {ch: [] for ch in vol_channels}
    #         temp_vol_time_vectors = {ch: [] for ch in vol_channels}
    #         valid_trial_indices = []
            
    #         # Extract segments for each trial
    #         for idx, (trial_idx, row) in enumerate(df_enhanced.iterrows()):
    #             # Always use trial_start_timestamp as the fundamental reference
    #             trial_start_time = row.get('trial_start_timestamp', np.nan)
                
    #             if pd.isna(trial_start_time):
    #                 df_enhanced.loc[trial_idx, 'drop_reason'] = 'missing_trial_start_timestamp'
    #                 dropped_trials.append((trial_idx, 'missing_trial_start_timestamp'))
    #                 continue
                
    #             # Calculate actual alignment time
    #             alignment_event_time = row.get(alignment, np.nan)
    #             if pd.isna(alignment_event_time):
    #                 df_enhanced.loc[trial_idx, 'drop_reason'] = f'missing_{alignment}_time'
    #                 dropped_trials.append((trial_idx, f'missing_{alignment}_time'))
    #                 continue
                
    #             align_time = trial_start_time + alignment_event_time  # Event times are already absolute timestamps
                
    #             # Define window around alignment point
    #             window_start = align_time - pre_sec
    #             window_end = align_time + post_sec
    #             required_duration = pre_sec + post_sec
                
    #             # Extract DFF segment
    #             img_start_idx = np.searchsorted(imaging_data['imaging_time'], window_start)
    #             img_end_idx = np.searchsorted(imaging_data['imaging_time'], window_end)

    #             # Validate segment has sufficient data coverage
    #             if img_start_idx < img_end_idx and img_end_idx <= len(imaging_data['imaging_time']):
    #                 # Check actual time coverage
    #                 actual_start_time = imaging_data['imaging_time'][img_start_idx]
    #                 actual_end_time = imaging_data['imaging_time'][img_end_idx - 1]
    #                 actual_duration = actual_end_time - actual_start_time
                    
    #                 # Ensure we have at least 90% of the requested window duration
    #                 if actual_duration >= 0.9 * required_duration:
    #                     # Store temporary segments
    #                     dff_segment = imaging_data['dff_traces'][:, img_start_idx:img_end_idx]
    #                     dff_time_vector = imaging_data['imaging_time'][img_start_idx:img_end_idx] - align_time
                        
    #                     temp_dff_segments.append(dff_segment)
    #                     temp_dff_time_vectors.append(dff_time_vector)
    #                     valid_trial_indices.append(trial_idx)
                        
    #                     df_enhanced.at[trial_idx, 'segment_duration'] = actual_duration
    #                     df_enhanced.at[trial_idx, 'segment_valid'] = True
                        
    #                     # Extract voltage segments
    #                     vol_start_idx = np.searchsorted(imaging_data['vol_time'], window_start)
    #                     vol_end_idx = np.searchsorted(imaging_data['vol_time'], window_end)
                        
    #                     if vol_start_idx < vol_end_idx and vol_end_idx <= len(imaging_data['vol_time']):
    #                         vol_time_vector = imaging_data['vol_time'][vol_start_idx:vol_end_idx] - align_time
                            
    #                         for ch in vol_channels:
    #                             if imaging_data.get(ch) is not None:
    #                                 vol_segment = imaging_data[ch][vol_start_idx:vol_end_idx]
    #                                 temp_vol_segments[ch].append(vol_segment)
    #                                 temp_vol_time_vectors[ch].append(vol_time_vector)
    #                             else:
    #                                 temp_vol_segments[ch].append(None)
    #                                 temp_vol_time_vectors[ch].append(None)
    #                     else:
    #                         # No voltage data for this trial
    #                         for ch in vol_channels:
    #                             temp_vol_segments[ch].append(None)
    #                             temp_vol_time_vectors[ch].append(None)
                        
    #                 else:
    #                     df_enhanced.loc[trial_idx, 'drop_reason'] = f'insufficient_coverage_{actual_duration:.2f}s'
    #                     dropped_trials.append((trial_idx, f'insufficient_coverage_{actual_duration:.2f}s'))
    #             else:
    #                 df_enhanced.loc[trial_idx, 'drop_reason'] = 'outside_data_bounds'
    #                 dropped_trials.append((trial_idx, 'outside_data_bounds'))
            
    #         # Log dropped trials
    #         if dropped_trials:
    #             self.logger.warning(f"SID_IMG_PRE: Dropped {len(dropped_trials)} trials due to insufficient data:")
    #             for trial_idx, reason in dropped_trials:
    #                 self.logger.warning(f"  Trial {trial_idx}: {reason}")
            
    #         # Second pass: trim all segments to common length
    #         if temp_dff_segments:
    #             # Trim DFF segments to common length
    #             trimmed_dff_segments = self.trim_sequences_to_common_length(temp_dff_segments, 'dff')
    #             trimmed_dff_time_vectors = self.trim_sequences_to_common_length(temp_dff_time_vectors, 'dff_time')
                
    #             # Get common time vector (all should be identical after trimming)
    #             common_dff_time_vector = trimmed_dff_time_vectors[0]
                
    #             # Trim voltage segments to common length
    #             trimmed_vol_segments = {}
    #             trimmed_vol_time_vectors = {}
    #             common_vol_time_vectors = {}
                
    #             for ch in vol_channels:
    #                 # Filter out None values for trimming
    #                 valid_vol_segments = [seg for seg in temp_vol_segments[ch] if seg is not None]
    #                 valid_vol_time_vectors = [tvec for tvec in temp_vol_time_vectors[ch] if tvec is not None]
                    
    #                 if valid_vol_segments:
    #                     trimmed_vol_segments[ch] = self.trim_sequences_to_common_length(valid_vol_segments, f'vol_{ch}')
    #                     trimmed_vol_time_vectors[ch] = self.trim_sequences_to_common_length(valid_vol_time_vectors, f'vol_{ch}_time')
    #                     common_vol_time_vectors[ch] = trimmed_vol_time_vectors[ch][0]
    #                 else:
    #                     trimmed_vol_segments[ch] = []
    #                     trimmed_vol_time_vectors[ch] = []
    #                     common_vol_time_vectors[ch] = common_dff_time_vector  # Use DFF time as fallback
                

    #             # Third pass: store trimmed segments back in DataFrame
    #             for i, trial_idx in enumerate(valid_trial_indices):
    #                 # Store trimmed DFF data
    #                 df_enhanced.at[trial_idx, 'dff_segment'] = trimmed_dff_segments[i]
    #                 df_enhanced.at[trial_idx, 'dff_time_vector'] = common_dff_time_vector
                    
    #                 # Store trimmed voltage data
    #                 vol_segments_dict = {}
    #                 vol_time_vectors_dict = {}
                    
    #                 for ch in vol_channels:
    #                     if len(trimmed_vol_segments[ch]) > 0:
    #                         # Simple indexing: trial i gets voltage segment i (if it exists)
    #                         # Count how many non-None voltage segments exist up to position i
    #                         valid_vol_count = sum(1 for seg in temp_vol_segments[ch][:i+1] if seg is not None)
    #                         vol_idx = valid_vol_count - 1  # Convert to 0-based index
                            
    #                         if 0 <= vol_idx < len(trimmed_vol_segments[ch]):
    #                             vol_segments_dict[ch] = trimmed_vol_segments[ch][vol_idx]
    #                             vol_time_vectors_dict[ch] = common_vol_time_vectors[ch]
    #                         else:
    #                             vol_segments_dict[ch] = None
    #                             vol_time_vectors_dict[ch] = common_dff_time_vector
    #                     else:
    #                         vol_segments_dict[ch] = None
    #                         vol_time_vectors_dict[ch] = common_dff_time_vector
                    
    #                 df_enhanced.at[trial_idx, 'vol_segments'] = vol_segments_dict
    #                 df_enhanced.at[trial_idx, 'vol_time_vectors'] = vol_time_vectors_dict

                
    #             # Get valid trials mask
    #             valid_trials = df_enhanced['segment_valid']
    #             n_valid = np.sum(valid_trials)
                
    #             self.logger.info(f"SID_IMG_PRE: Successfully extracted and trimmed segments for {n_valid}/{len(df_enhanced)} trials")
    #             self.logger.info(f"SID_IMG_PRE: Common DFF segment shape: {trimmed_dff_segments[0].shape}")
    #             self.logger.info(f"SID_IMG_PRE: Common time vector length: {len(common_dff_time_vector)}")
                
    #             return {
    #                 'df_trials_with_segments': df_enhanced,
    #                 'valid_trials_mask': valid_trials,
    #                 'n_valid_trials': n_valid,
    #                 'alignment_point': alignment,
    #                 'window': {'pre_sec': pre_sec, 'post_sec': post_sec},
    #                 'dropped_trials': dropped_trials,
                    
    #                 # Convenience arrays for analysis (now guaranteed to be homogeneous)
    #                 'dff_segments_array': trimmed_dff_segments,
    #                 'common_time_vector': common_dff_time_vector,
    #                 'common_vol_time_vectors': common_vol_time_vectors
    #             }
    #         else:
    #             return {'error': 'No valid trial segments extracted'}
                
    #     except Exception as e:
    #         self.logger.error(f"SID_IMG_PRE: Failed to extract trial segments: {e}")
    #         return {'error': str(e)}






    def segments_to_arrays(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert DataFrame-stored segments back to array format for compatibility.
        
        Args:
            trial_data: Dictionary from extract_trial_segments_simple
            
        Returns:
            Dictionary with array-format data for backward compatibility
        """
        try:
            df_with_segments = trial_data['df_trials_with_segments']
            valid_mask = trial_data['valid_trials_mask']
            df_valid = df_with_segments[valid_mask]
            
            if len(df_valid) == 0:
                return {'error': 'No valid trials'}
            
            # Extract segments into arrays
            dff_segments = np.array([row['dff_segment'] for _, row in df_valid.iterrows()])
            dff_time_vector = df_valid.iloc[0]['dff_time_vector']  # Should be identical for all
            
            # Extract voltage segments
            vol_segments = {}
            vol_time_vectors = {}
            
            first_trial_vol = df_valid.iloc[0]['vol_segments']
            if first_trial_vol:
                for ch_name in first_trial_vol.keys():
                    vol_segments[ch_name] = np.array([row['vol_segments'].get(ch_name, []) 
                                                    for _, row in df_valid.iterrows() 
                                                    if row['vol_segments'] and ch_name in row['vol_segments']])
                    vol_time_vectors[ch_name] = df_valid.iloc[0]['vol_time_vectors'].get(ch_name, dff_time_vector)
            
            return {
                'dff_segments': dff_segments,
                'vol_segments': vol_segments,
                'dff_time_vector': dff_time_vector,
                'vol_time_vectors': vol_time_vectors,
                'df_trials': df_valid.drop(columns=['dff_segment', 'dff_time_vector', 'vol_segments', 'vol_time_vectors']),
                'alignment_point': trial_data['alignment_point'],
                'window': trial_data['window'],
                'n_trials': len(df_valid),
                'dropped_trials': trial_data['dropped_trials']
            }
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to convert segments to arrays: {e}")
            return {'error': str(e)}



    def ensure_array_format(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure trial data is in array format for functions that require it.
        
        Args:
            trial_data: Trial data in either DataFrame or array format
            
        Returns:
            Trial data in array format
        """
        if 'df_trials_with_segments' in trial_data:
            # Convert from DataFrame format to array format
            return self.segments_to_arrays(trial_data)
        else:
            # Already in array format
            return trial_data

    def ensure_dataframe_format(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure trial data is in DataFrame format for functions that require it.
        
        Args:
            trial_data: Trial data in either DataFrame or array format
            
        Returns:
            Trial data in DataFrame format (or error if conversion not possible)
        """
        if 'df_trials_with_segments' in trial_data:
            # Already in DataFrame format
            return trial_data
        else:
            # Cannot convert from array format back to DataFrame format
            # (would need to re-extract segments)
            self.logger.error("SID_IMG_PRE: Cannot convert array format back to DataFrame format")
            return {'error': 'Cannot convert from array to DataFrame format'}





    def plot_trial_segments_check(self, trial_data: Dict[str, Any], subject_id: str) -> None:
        """
        Plot trial-segmented data to check alignment quality.
        
        Args:
            trial_data: Dictionary containing segmented trial data (DataFrame format)
            subject_id: Subject identifier for plot title
        """
        try:
            import matplotlib.pyplot as plt
            
            if 'error' in trial_data:
                self.logger.error(f"SID_IMG_PRE: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            # Handle both old array format and new DataFrame format
            if 'df_trials_with_segments' in trial_data:
                # New DataFrame format
                df_with_segments = trial_data['df_trials_with_segments']
                valid_mask = trial_data['valid_trials_mask']
                df_valid = df_with_segments[valid_mask]
                
                if len(df_valid) == 0:
                    self.logger.error("SID_IMG_PRE: No valid trials with segments")
                    return
                
                # Extract arrays from DataFrame for plotting
                dff_segments = np.array([row['dff_segment'] for _, row in df_valid.iterrows()])
                dff_time_vector = df_valid.iloc[0]['dff_time_vector']
                
                # Extract voltage segments if available
                vol_segments = {}
                vol_time_vectors = {}
                first_trial_vol = df_valid.iloc[0]['vol_segments']
                if first_trial_vol:
                    for ch_name in first_trial_vol.keys():
                        vol_segments[ch_name] = np.array([row['vol_segments'].get(ch_name, []) 
                                                        for _, row in df_valid.iterrows() 
                                                        if row['vol_segments'] and ch_name in row['vol_segments']])
                        vol_time_vectors[ch_name] = df_valid.iloc[0]['vol_time_vectors'].get(ch_name, dff_time_vector)
                
                df_trials = df_valid
            else:
                # Old array format (backward compatibility)
                dff_segments = trial_data['dff_segments']
                dff_time_vector = trial_data['dff_time_vector']
                vol_segments = trial_data.get('vol_segments', {})
                vol_time_vectors = trial_data.get('vol_time_vectors', {})
                df_trials = trial_data['df_trials']
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            alignment = trial_data['alignment_point']            
            fig.suptitle(f'Trial Segments Check - {subject_id} - {alignment}', fontsize=14)
            
            # Plot 1: Mean DFF across all trials and ROIs
            mean_dff_all = np.mean(dff_segments, axis=(0, 1))  # Average across trials and ROIs
            axes[0, 0].plot(dff_time_vector, mean_dff_all, 'b-', linewidth=2)
            axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Alignment Point')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Mean DFF')
            axes[0, 0].set_title(f'Grand Average DFF (n={dff_segments.shape[0]} trials)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            

            # Plot 2: Trial raster - mean DFF per trial
            trial_means = np.mean(dff_segments, axis=1)  # Average across ROIs for each trial
            
            # Calculate actual data range for colormap
            vmin_actual = np.percentile(trial_means, 5)  # 5th percentile
            vmax_actual = np.percentile(trial_means, 95)  # 95th percentile
            
            # RdBu_r
            im = axes[0, 1].imshow(trial_means, aspect='auto', cmap='viridis',
                                extent=[dff_time_vector[0], dff_time_vector[-1], 0, len(trial_means)],
                                vmin=vmin_actual, vmax=vmax_actual, interpolation='nearest')
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
                    axes[1, 0].plot(dff_time_vector, left_mean, 'g-', linewidth=2, label=f'Left (n={np.sum(left_trials)})')

                if np.any(right_trials):
                    right_mean = np.mean(dff_segments[right_trials.values], axis=(0, 1))
                    axes[1, 0].plot(dff_time_vector, right_mean, 'r-', linewidth=2, label=f'Right (n={np.sum(right_trials)})')
                
                axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.7)
                axes[1, 0].set_xlabel('Time (s)')
                axes[1, 0].set_ylabel('Mean DFF')
                axes[1, 0].set_title('Left vs Right Trials')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, 'No trial_side data', ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # Plot 4: Voltage segments (if available)
            if 'vol_start' in vol_segments and len(vol_segments['vol_start']) > 0:
                vol_start_segments = vol_segments['vol_start']
                vol_start_time_vector = vol_time_vectors.get('vol_start', dff_time_vector)
                
                # Plot first few trials
                n_trials_to_show = min(5, len(vol_start_segments))
                for i in range(n_trials_to_show):
                    axes[1, 1].plot(vol_start_time_vector, vol_start_segments[i] + i*2, 
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
            
            self.logger.info(f"SID_IMG_PRE: Generated trial segments check plot")
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create trial segments check plot: {e}")


    def _plot_individual_trials_from_segments(self, trial_data: Dict[str, Any], subject_id: str) -> None:
        """
        Plot individual trials using pre-segmented trial data.
        
        Args:
            trial_data: Dictionary containing segmented trial data (DataFrame format)
            subject_id: Subject identifier for plot title
        """
        try:
            import matplotlib.pyplot as plt
            
            if 'error' in trial_data:
                self.logger.error(f"SID_IMG_PRE: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            # Handle both old array format and new DataFrame format
            if 'df_trials_with_segments' in trial_data:
                # New DataFrame format
                df_with_segments = trial_data['df_trials_with_segments']
                valid_mask = trial_data['valid_trials_mask']
                df_valid = df_with_segments[valid_mask]
                
                if len(df_valid) == 0:
                    self.logger.error("SID_IMG_PRE: No valid trials with segments")
                    return
                
                # Get alignment point
                alignment_point = trial_data['alignment_point']
                
            else:
                # Old array format (backward compatibility)
                dff_segments = trial_data['dff_segments']
                vol_segments = trial_data['vol_segments']
                dff_time_vector = trial_data['dff_time_vector']
                vol_time_vectors = trial_data['vol_time_vectors']
                df_trials = trial_data['df_trials']
                alignment_point = trial_data['alignment_point']
            
            # Plot first few trials
            start_trial = 0
            n_trials_to_plot = min(3, len(df_valid) if 'df_trials_with_segments' in trial_data else len(dff_segments))
            
            for trial_idx in range(start_trial, start_trial + n_trials_to_plot):
                if 'df_trials_with_segments' in trial_data:
                    # New DataFrame format
                    if trial_idx >= len(df_valid):
                        break
                    
                    trial_row = df_valid.iloc[trial_idx]
                    dff_segment = trial_row['dff_segment']
                    dff_time_vector = trial_row['dff_time_vector']
                    vol_segments_dict = trial_row['vol_segments']
                    vol_time_vectors_dict = trial_row['vol_time_vectors']
                    
                else:
                    # Old array format
                    if trial_idx >= len(dff_segments):
                        break
                    
                    trial_row = df_trials.iloc[trial_idx]
                    dff_segment = dff_segments[trial_idx]
                    # dff_time_vector already defined above
                    # vol_segments and vol_time_vectors already defined above
                
                # Create plot for this trial
                fig, axes = plt.subplots(3, 1, figsize=(12, 8))
                actual_trial_index = trial_row['trial_index']
                fig.suptitle(f'Trial {actual_trial_index} Segments - {subject_id}', fontsize=12)
                
                # Plot 1: Mean DFF for this trial
                mean_dff_trial = np.mean(dff_segment, axis=0)
                axes[0].plot(dff_time_vector, mean_dff_trial, 'b-', linewidth=2)
                axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Alignment Point')
                axes[0].set_ylabel('Mean DFF')
                axes[0].set_title('Mean DFF Activity')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
                
                # Plot 2: Voltage start for this trial (if available)
                vol_start_available = False
                if 'df_trials_with_segments' in trial_data:
                    # New format - check if this trial has vol_start data
                    if vol_segments_dict and 'vol_start' in vol_segments_dict:
                        vol_start_trial = vol_segments_dict['vol_start']
                        vol_start_time_vector = vol_time_vectors_dict.get('vol_start', dff_time_vector)
                        vol_start_available = True
                else:
                    # Old format
                    if 'vol_start' in vol_segments and len(vol_segments['vol_start']) > trial_idx:
                        vol_start_trial = vol_segments['vol_start'][trial_idx]
                        vol_start_time_vector = vol_time_vectors.get('vol_start', dff_time_vector)
                        vol_start_available = True
                
                if vol_start_available:
                    axes[1].plot(vol_start_time_vector, vol_start_trial, 'r-', linewidth=1.5)
                    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                    axes[1].set_ylabel('Voltage')
                    axes[1].set_title('Trial Start Pulse')
                    axes[1].grid(True, alpha=0.3)
                else:
                    axes[1].text(0.5, 0.5, 'No vol_start data', ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title('Trial Start Pulse (no data)')
                
                # Plot 3: Voltage stimulus channels for this trial
                y_offset = 0
                legend_added = False
                for ch_name, color in [('vol_stim_vis', 'green'), ('vol_stim_aud', 'purple')]:
                    ch_available = False
                    
                    if 'df_trials_with_segments' in trial_data:
                        # New format
                        if vol_segments_dict and ch_name in vol_segments_dict:
                            ch_trial = vol_segments_dict[ch_name]
                            ch_time_vector = vol_time_vectors_dict.get(ch_name, dff_time_vector)
                            ch_available = True
                    else:
                        # Old format
                        if ch_name in vol_segments and len(vol_segments[ch_name]) > trial_idx:
                            ch_trial = vol_segments[ch_name][trial_idx]
                            ch_time_vector = vol_time_vectors.get(ch_name, dff_time_vector)
                            ch_available = True
                    
                    if ch_available:
                        axes[2].plot(ch_time_vector, ch_trial + y_offset, color=color,
                                linewidth=1.5, label=ch_name, alpha=0.8)
                        y_offset += 2
                        legend_added = True
                
                axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                axes[2].set_ylabel('Voltage + Offset')
                axes[2].set_xlabel('Time relative to alignment (s)')
                axes[2].set_title('Stimulus Channels')
                if legend_added:
                    axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                # Add behavioral events as vertical lines (relative to alignment point)
                event_colors = {
                    'start_flash_1': 'red', 'end_flash_1': 'orange', 'start_flash_2': 'blue',
                    'end_flash_2': 'cyan', 'choice_start': 'green', 'reward_start': 'purple',
                    'punish_start': 'black'
                }
                
                # Get the alignment timestamp for this trial
                alignment_timestamp = trial_row.get(alignment_point, np.nan)
                
                if not pd.isna(alignment_timestamp):
                    for event_name, color in event_colors.items():
                        if event_name in trial_row and not pd.isna(trial_row[event_name]):
                            # Calculate relative time to alignment point
                            event_time_rel = trial_row[event_name] - alignment_timestamp
                            
                            # Check if event falls within the plotted time window
                            if dff_time_vector[0] <= event_time_rel <= dff_time_vector[-1]:
                                # Add vertical lines to ALL axes
                                for ax_idx, ax in enumerate(axes):
                                    ax.axvline(x=event_time_rel, color=color, alpha=0.6, 
                                            linestyle='-', linewidth=2)
                                    
                                    # Add event labels (only on top plot to avoid clutter)
                                    if ax_idx == 0:
                                        ax.text(event_time_rel, ax.get_ylim()[1]*0.9, event_name,
                                            rotation=90, fontsize=8, color=color, 
                                            ha='center', va='top', weight='bold')
                
                plt.tight_layout()
                plt.show()
                
            self.logger.info(f"SID_IMG_PRE: Generated individual trial plots from segments for {n_trials_to_plot} trials")
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create individual trial plots from segments: {e}")


 

    def plot_responsive_rois_by_condition(self, trial_data: Dict[str, Any], subject_id: str) -> None:
        """
        Plot ROIs that are responsive to alignment, filtered by trial conditions.
        """
        try:
            
            
            if 'error' in trial_data:
                self.logger.error(f"SID_IMG_PRE: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            df_with_segments = trial_data['df_trials_with_segments']
            valid_mask = trial_data['valid_trials_mask']
            
            # Filter to valid trials only
            df_valid = df_with_segments[valid_mask].copy()
            
            if len(df_valid) == 0:
                self.logger.error("SID_IMG_PRE: No valid trials with segments")
                return
            
            # Get common time vector (all should be identical)
            common_time_vector = df_valid.iloc[0]['dff_time_vector']
            
            # Define analysis windows
            baseline_window = (-2.0, -0.5)
            response_window = (0.0, 2.0)
            
            # Find time indices for windows
            baseline_mask = (common_time_vector >= baseline_window[0]) & (common_time_vector <= baseline_window[1])
            response_mask = (common_time_vector >= response_window[0]) & (common_time_vector <= response_window[1])
            
            # Define trial conditions
            if 'trial_side' not in df_valid.columns or 'rewarded' not in df_valid.columns:
                self.logger.warning("SID_IMG_PRE: Missing trial_side or rewarded columns")
                return
            
            left_rewarded = (df_valid['trial_side'] == 'left') & (df_valid['rewarded'] == 1)
            right_rewarded = (df_valid['trial_side'] == 'right') & (df_valid['rewarded'] == 1)
            
            if not np.any(left_rewarded) or not np.any(right_rewarded):
                self.logger.warning("SID_IMG_PRE: Insufficient trials for left+rewarded or right+rewarded conditions")
                return
            
            # Get number of ROIs from first trial
            n_rois = df_valid.iloc[0]['dff_segment'].shape[0]
            
            # Detect responsive ROIs for each condition
            left_responsive_rois = []
            right_responsive_rois = []
            left_p_values = []
            right_p_values = []
            left_effect_sizes = []
            right_effect_sizes = []
            
            for roi_idx in range(n_rois):
                # Left+rewarded condition
                left_trials = df_valid[left_rewarded]
                if len(left_trials) > 2:
                    # Extract baseline and response values for this ROI across left trials
                    left_baseline_vals = []
                    left_response_vals = []
                    
                    for _, trial_row in left_trials.iterrows():
                        dff_segment = trial_row['dff_segment']
                        baseline_val = np.mean(dff_segment[roi_idx, baseline_mask])
                        response_val = np.mean(dff_segment[roi_idx, response_mask])
                        left_baseline_vals.append(baseline_val)
                        left_response_vals.append(response_val)
                    
                    # Statistical test
                    _, left_p = stats.ttest_rel(left_response_vals, left_baseline_vals)
                    
                    # Effect size
                    pooled_std = np.sqrt((np.var(left_baseline_vals) + np.var(left_response_vals)) / 2)
                    left_effect = (np.mean(left_response_vals) - np.mean(left_baseline_vals)) / (pooled_std + 1e-10)
                    
                    left_p_values.append(left_p)
                    left_effect_sizes.append(left_effect)
                    
                    if left_p < 0.01:
                        left_responsive_rois.append(roi_idx)
                else:
                    left_p_values.append(1.0)
                    left_effect_sizes.append(0.0)
                
                # Right+rewarded condition
                right_trials = df_valid[right_rewarded]
                if len(right_trials) > 2:
                    # Extract baseline and response values for this ROI across right trials
                    right_baseline_vals = []
                    right_response_vals = []
                    
                    for _, trial_row in right_trials.iterrows():
                        dff_segment = trial_row['dff_segment']
                        baseline_val = np.mean(dff_segment[roi_idx, baseline_mask])
                        response_val = np.mean(dff_segment[roi_idx, response_mask])
                        right_baseline_vals.append(baseline_val)
                        right_response_vals.append(response_val)
                    
                    # Statistical test
                    _, right_p = stats.ttest_rel(right_response_vals, right_baseline_vals)
                    
                    # Effect size
                    pooled_std = np.sqrt((np.var(right_baseline_vals) + np.var(right_response_vals)) / 2)
                    right_effect = (np.mean(right_response_vals) - np.mean(right_baseline_vals)) / (pooled_std + 1e-10)
                    
                    right_p_values.append(right_p)
                    right_effect_sizes.append(right_effect)
                    
                    if right_p < 0.01:
                        right_responsive_rois.append(roi_idx)
                else:
                    right_p_values.append(1.0)
                    right_effect_sizes.append(0.0)
            
            # Rest of plotting code remains similar, but now you can easily access individual trial data:
            # For example, to plot mean traces:
            if left_responsive_rois:
                left_mean_traces = []
                for _, trial_row in df_valid[left_rewarded].iterrows():
                    trial_mean = np.mean(trial_row['dff_segment'][left_responsive_rois, :], axis=0)
                    left_mean_traces.append(trial_mean)
                left_grand_mean = np.mean(left_mean_traces, axis=0)
                # Plot left_grand_mean...
            
            # ... rest of plotting code ...
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to plot responsive ROIs by condition: {e}")
            return None





    def _get_roi_sort_order(self, left_traces: np.ndarray, right_traces: np.ndarray, 
                        time_vector: np.ndarray, sort_method: str) -> np.ndarray:
        """
        Get ROI sort order based on response characteristics.
        
        Args:
            left_traces: Left condition traces (n_rois, n_timepoints)
            right_traces: Right condition traces (n_rois, n_timepoints)
            time_vector: Time vector
            sort_method: Sorting method
            
        Returns:
            Array of ROI indices in sorted order
        """
        n_rois = left_traces.shape[0]
        
        if sort_method == 'peak_time':
            # Sort by time of peak response (combined conditions)
            combined_traces = (left_traces + right_traces) / 2
            response_mask = time_vector >= 0  # Only consider post-stimulus
            
            peak_times = []
            for roi_idx in range(n_rois):
                response_trace = combined_traces[roi_idx, response_mask]
                if len(response_trace) > 0:
                    peak_idx = np.argmax(np.abs(response_trace))
                    peak_time = time_vector[response_mask][peak_idx]
                    peak_times.append(peak_time)
                else:
                    peak_times.append(np.inf)  # No response data
            
            # Sort by peak time (earliest to latest)
            sort_indices = np.argsort(peak_times)
            
        elif sort_method == 'peak_magnitude':
            # Sort by magnitude of peak response
            combined_traces = (left_traces + right_traces) / 2
            response_mask = time_vector >= 0
            
            peak_magnitudes = []
            for roi_idx in range(n_rois):
                response_trace = combined_traces[roi_idx, response_mask]
                if len(response_trace) > 0:
                    peak_mag = np.max(np.abs(response_trace))
                    peak_magnitudes.append(peak_mag)
                else:
                    peak_magnitudes.append(0)
            
            # Sort by peak magnitude (largest to smallest)
            sort_indices = np.argsort(peak_magnitudes)[::-1]
            
        elif sort_method == 'response_onset':
            # Sort by response onset time (threshold crossing)
            combined_traces = (left_traces + right_traces) / 2
            baseline_mask = time_vector < 0
            response_mask = time_vector >= 0
            
            onset_times = []
            for roi_idx in range(n_rois):
                baseline_std = np.std(combined_traces[roi_idx, baseline_mask])
                threshold = 2 * baseline_std  # 2 standard deviations above baseline
                
                response_trace = combined_traces[roi_idx, response_mask]
                onset_idx = np.where(np.abs(response_trace) > threshold)[0]
                
                if len(onset_idx) > 0:
                    onset_time = time_vector[response_mask][onset_idx[0]]
                    onset_times.append(onset_time)
                else:
                    onset_times.append(np.inf)  # No significant response
            
            # Sort by onset time (earliest to latest)
            sort_indices = np.argsort(onset_times)
            
        else:
            # No sorting - original order
            sort_indices = np.arange(n_rois)
        
        return sort_indices


    def plot_condition_comparison_heatmap(self, trial_data: Dict[str, Any], subject_id: str, 
                                        sort_rois: bool = True, sort_method: str = 'peak_time') -> None:
        """
        Plot heatmap comparing ROI responses across left+rewarded vs right+rewarded conditions.
        
        Args:
            sort_rois: Whether to sort ROIs by response characteristics
            sort_method: 'peak_time', 'peak_magnitude', 'response_onset', or 'none'
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if 'error' in trial_data:
                self.logger.error(f"SID_IMG_PRE: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            # Handle both old array format and new DataFrame format
            if 'df_trials_with_segments' in trial_data:
                # New DataFrame format
                df_with_segments = trial_data['df_trials_with_segments']
                valid_mask = trial_data['valid_trials_mask']
                df_valid = df_with_segments[valid_mask]
                
                if len(df_valid) == 0:
                    self.logger.error("SID_IMG_PRE: No valid trials with segments")
                    return
                
                # Get common time vector
                dff_time_vector = df_valid.iloc[0]['dff_time_vector']
                
                # Define conditions
                left_rewarded = (df_valid['trial_side'] == 'left') & (df_valid['rewarded'] == 1)
                right_rewarded = (df_valid['trial_side'] == 'right') & (df_valid['rewarded'] == 1)
                
                if not np.any(left_rewarded) or not np.any(right_rewarded):
                    self.logger.warning("SID_IMG_PRE: Insufficient trials for condition comparison")
                    return
                
                # Calculate mean response for each ROI in each condition using DataFrame format
                left_trials = df_valid[left_rewarded]
                right_trials = df_valid[right_rewarded]
                
                # Get number of ROIs and timepoints from first trial
                n_rois, n_timepoints = left_trials.iloc[0]['dff_segment'].shape
                
                # Initialize arrays to collect data
                left_mean_traces = np.zeros((n_rois, n_timepoints))
                right_mean_traces = np.zeros((n_rois, n_timepoints))
                
                # Calculate mean for left trials
                all_left_segments = np.array([row['dff_segment'] for _, row in left_trials.iterrows()])
                left_mean_traces = np.mean(all_left_segments, axis=0)  # (n_rois, n_timepoints)
                
                # Calculate mean for right trials
                all_right_segments = np.array([row['dff_segment'] for _, row in right_trials.iterrows()])
                right_mean_traces = np.mean(all_right_segments, axis=0)  # (n_rois, n_timepoints)
                
            
            # Sort ROIs if requested
            if sort_rois and sort_method != 'none':
                roi_order = self._get_roi_sort_order(left_mean_traces, right_mean_traces, 
                                                    dff_time_vector, sort_method)
                
                # Reorder the traces
                left_mean_traces = left_mean_traces[roi_order, :]
                right_mean_traces = right_mean_traces[roi_order, :]
                
                # Create ROI labels for y-axis
                roi_labels = [f'ROI {roi_order[i]}' for i in range(len(roi_order))]
            else:
                roi_order = np.arange(left_mean_traces.shape[0])
                roi_labels = [f'ROI {i}' for i in range(left_mean_traces.shape[0])]
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            alignment = trial_data['alignment_point']
            fig.suptitle(f'ROI Response Comparison: Left+Rewarded vs Right+Rewarded - {subject_id} - Aligned: {alignment}', fontsize=14)
            
            # Calculate actual data range for consistent colormaps
            all_data = np.concatenate([left_mean_traces.flatten(), right_mean_traces.flatten()])
            vmin_actual = np.percentile(all_data, 5)
            vmax_actual = np.percentile(all_data, 95)
            
            # For difference plot, use symmetric range around zero
            diff_traces = right_mean_traces - left_mean_traces
            diff_max = np.max(np.abs(np.percentile(diff_traces, [5, 95])))
            
            # Plot 1: Left+rewarded heatmap
            # RdBu_r
            im1 = axes[0].imshow(left_mean_traces, aspect='auto', cmap='viridis',
                                extent=[dff_time_vector[0], dff_time_vector[-1], 0, left_mean_traces.shape[0]],
                                vmin=vmin_actual, vmax=vmax_actual, interpolation='nearest')
            axes[0].axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.9)
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('ROI #')
            # Update y-axis to show original ROI indices
            axes[0].set_yticks(np.arange(0, len(roi_labels), max(1, len(roi_labels)//10)))
            axes[0].set_yticklabels([roi_labels[i] for i in axes[0].get_yticks().astype(int)])
                    
            
            if 'df_trials_with_segments' in trial_data:
                axes[0].set_title(f'Left+Rewarded (n={len(left_trials)} trials)')
            else:
                axes[0].set_title(f'Left+Rewarded (n={np.sum(left_rewarded)} trials)')
            plt.colorbar(im1, ax=axes[0], label='Mean DFF')
            
            # Plot 2: Right+rewarded heatmap
            # RdBu_r
            im2 = axes[1].imshow(right_mean_traces, aspect='auto', cmap='viridis',
                                extent=[dff_time_vector[0], dff_time_vector[-1], 0, right_mean_traces.shape[0]],
                                vmin=vmin_actual, vmax=vmax_actual, interpolation='nearest')
            axes[1].axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.9)
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('ROI #')
            # Update y-axis to show original ROI indices
            axes[1].set_yticks(np.arange(0, len(roi_labels), max(1, len(roi_labels)//10)))
            axes[1].set_yticklabels([roi_labels[i] for i in axes[0].get_yticks().astype(int)])
                    
            
            if 'df_trials_with_segments' in trial_data:
                axes[1].set_title(f'Right+Rewarded (n={len(right_trials)} trials)')
            else:
                axes[1].set_title(f'Right+Rewarded (n={np.sum(right_rewarded)} trials)')
            plt.colorbar(im2, ax=axes[1], label='Mean DFF')
            
            # Plot 3: Difference (Right - Left) with symmetric range
            # RdBu_r
            im3 = axes[2].imshow(diff_traces, aspect='auto', cmap='viridis',
                                extent=[dff_time_vector[0], dff_time_vector[-1], 0, diff_traces.shape[0]],
                                vmin=-diff_max, vmax=diff_max, interpolation='nearest')
            axes[2].axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.9)
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('ROI #')
            axes[2].set_title('Difference (Right - Left)')
            # Update y-axis to show original ROI indices
            axes[2].set_yticks(np.arange(0, len(roi_labels), max(1, len(roi_labels)//10)))
            axes[2].set_yticklabels([roi_labels[i] for i in axes[0].get_yticks().astype(int)])
                    
            plt.colorbar(im3, ax=axes[2], label='ΔDF/F')
        
            
            plt.tight_layout()
            plt.show()
            
            # Log summary
            if 'df_trials_with_segments' in trial_data:
                self.logger.info(f"SID_IMG_PRE: Condition comparison heatmap - Left: {len(left_trials)} trials, Right: {len(right_trials)} trials")
            else:
                self.logger.info(f"SID_IMG_PRE: Condition comparison heatmap - Left: {np.sum(left_rewarded)} trials, Right: {np.sum(right_rewarded)} trials")
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create condition comparison heatmap: {e}")


 

    def plot_all_rois_heatmap(self, trial_data: Dict[str, Any], subject_id: str, 
                            sort_rois: bool = True, sort_method: str = 'peak_time',
                            trial_type: str = 'all') -> None:
        """
        Plot heatmap showing all ROI responses across trials.
        
        Args:
            trial_data: Dictionary from extract_trial_segments_simple
            subject_id: Subject ID for plot title
            sort_rois: Whether to sort ROIs by response characteristics
            sort_method: 'peak_time', 'peak_magnitude', 'response_onset', or 'none'
            trial_type: 'all', 'left', 'right', 'rewarded', 'unrewarded'
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if 'error' in trial_data:
                self.logger.error(f"SID_IMG_PRE: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            # Handle both old array format and new DataFrame format
            if 'df_trials_with_segments' in trial_data:
                # New DataFrame format
                df_with_segments = trial_data['df_trials_with_segments']
                valid_mask = trial_data['valid_trials_mask']
                df_valid = df_with_segments[valid_mask]
                
                if len(df_valid) == 0:
                    self.logger.error("SID_IMG_PRE: No valid trials with segments")
                    return
                
                # Get common time vector
                dff_time_vector = df_valid.iloc[0]['dff_time_vector']
                
                # Filter trials based on trial_type
                if trial_type == 'all':
                    selected_trials = df_valid
                    title_suffix = 'All Trials'
                elif trial_type == 'left':
                    selected_trials = df_valid[df_valid['trial_side'] == 'left']
                    title_suffix = 'Left Trials'
                elif trial_type == 'right':
                    selected_trials = df_valid[df_valid['trial_side'] == 'right']
                    title_suffix = 'Right Trials'
                elif trial_type == 'rewarded':
                    selected_trials = df_valid[df_valid['rewarded'] == 1]
                    title_suffix = 'Rewarded Trials'
                elif trial_type == 'unrewarded':
                    selected_trials = df_valid[df_valid['rewarded'] == 0]
                    title_suffix = 'Unrewarded Trials'
                else:
                    selected_trials = df_valid
                    title_suffix = f'{trial_type.title()} Trials'
                
                if len(selected_trials) == 0:
                    self.logger.warning(f"SID_IMG_PRE: No trials found for trial_type '{trial_type}'")
                    return
                
                # Get number of ROIs and timepoints from first trial
                n_rois, n_timepoints = selected_trials.iloc[0]['dff_segment'].shape
                
                # Calculate mean response across selected trials
                all_segments = np.array([row['dff_segment'] for _, row in selected_trials.iterrows()])
                mean_traces = np.mean(all_segments, axis=0)  # (n_rois, n_timepoints)
                
            else:
                # Old array format (backward compatibility)
                dff_segments = trial_data['dff_segments']
                dff_time_vector = trial_data['dff_time_vector']
                df_trials = trial_data['df_trials']
                
                # Filter trials based on trial_type
                if trial_type == 'all':
                    trial_mask = np.ones(len(df_trials), dtype=bool)
                    title_suffix = 'All Trials'
                elif trial_type == 'left':
                    trial_mask = (df_trials['trial_side'] == 'left').values
                    title_suffix = 'Left Trials'
                elif trial_type == 'right':
                    trial_mask = (df_trials['trial_side'] == 'right').values
                    title_suffix = 'Right Trials'
                elif trial_type == 'rewarded':
                    trial_mask = (df_trials['rewarded'] == 1).values
                    title_suffix = 'Rewarded Trials'
                elif trial_type == 'unrewarded':
                    trial_mask = (df_trials['rewarded'] == 0).values
                    title_suffix = 'Unrewarded Trials'
                else:
                    trial_mask = np.ones(len(df_trials), dtype=bool)
                    title_suffix = f'{trial_type.title()} Trials'
                
                if not np.any(trial_mask):
                    self.logger.warning(f"SID_IMG_PRE: No trials found for trial_type '{trial_type}'")
                    return
                
                # Calculate mean response across selected trials
                mean_traces = np.mean(dff_segments[trial_mask, :, :], axis=0)  # (n_rois, n_timepoints)
                n_rois = mean_traces.shape[0]
            
            # Sort ROIs if requested
            if sort_rois and sort_method != 'none':
                roi_order = self._get_roi_sort_order_single(mean_traces, dff_time_vector, sort_method)
                
                # Reorder the traces
                mean_traces = mean_traces[roi_order, :]
                
                # Create ROI labels for y-axis
                roi_labels = [f'ROI {roi_order[i]}' for i in range(len(roi_order))]
            else:
                roi_order = np.arange(mean_traces.shape[0])
                roi_labels = [f'ROI {i}' for i in range(mean_traces.shape[0])]
            
            # Create figure with multiple views
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            alignment = trial_data['alignment_point']
            fig.suptitle(f'All ROI Responses - {title_suffix} - {subject_id} - Alignment: {alignment}', fontsize=14)
            
            # Calculate actual data range for consistent colormaps
            vmin_actual = np.percentile(mean_traces, 5)
            vmax_actual = np.percentile(mean_traces, 95)
            
            # Plot 1: Full heatmap
            im1 = axes[0].imshow(mean_traces, aspect='auto', cmap='viridis',
                                extent=[dff_time_vector[0], dff_time_vector[-1], 0, mean_traces.shape[0]],
                                vmin=vmin_actual, vmax=vmax_actual, interpolation='nearest')
            axes[0].axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.9)
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('ROI #')
            
            axes[0].set_title(f'All ROIs Heatmap\n(sorted by {sort_method})')
            
            # Update y-axis to show original ROI indices
            y_tick_spacing = max(1, len(roi_labels)//10)
            y_ticks = np.arange(0, len(roi_labels), y_tick_spacing)
            axes[0].set_yticks(y_ticks)
            axes[0].set_yticklabels([roi_labels[i] for i in y_ticks])
            
            plt.colorbar(im1, ax=axes[0], label='Mean DFF')
            
            # Plot 2: Mean trace across all ROIs
            grand_mean = np.mean(mean_traces, axis=0)
            grand_sem = np.std(mean_traces, axis=0) / np.sqrt(mean_traces.shape[0])
            
            axes[1].plot(dff_time_vector, grand_mean, 'blue', linewidth=3, label='Grand Mean')
            axes[1].fill_between(dff_time_vector, grand_mean - grand_sem, grand_mean + grand_sem,
                            alpha=0.3, color='blue', label='±SEM')
            axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Alignment Point')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Mean DFF')
            axes[1].set_title(f'Grand Average\n(n={n_rois} ROIs)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Top responsive ROIs
            # Find ROIs with highest response magnitude
            response_mask = dff_time_vector >= 0  # Post-alignment period
            roi_response_magnitudes = np.max(np.abs(mean_traces[:, response_mask]), axis=1)
            top_roi_indices = np.argsort(roi_response_magnitudes)[-15:]  # Top 5 most responsive
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(top_roi_indices)))
            for i, roi_idx in enumerate(top_roi_indices):
                original_roi_idx = roi_order[roi_idx] if sort_rois else roi_idx
                axes[2].plot(dff_time_vector, mean_traces[roi_idx, :], 
                            color=colors[i], linewidth=2, alpha=0.8,
                            label=f'ROI {original_roi_idx}')
            
            axes[2].axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Alignment')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('DFF')
            axes[2].set_title('Top 5 Responsive ROIs')
            axes[2].legend(fontsize=9)
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Log summary
            if 'df_trials_with_segments' in trial_data:
                n_trials = len(selected_trials)
            else:
                n_trials = np.sum(trial_mask)
            
            self.logger.info(f"SID_IMG_PRE: All ROIs heatmap - {n_trials} trials, {n_rois} ROIs")
            self.logger.info(f"SID_IMG_PRE: ROI sorting: {sort_method}, Trial type: {trial_type}")
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create all ROIs heatmap: {e}")


    def _get_roi_sort_order_single(self, traces: np.ndarray, time_vector: np.ndarray, 
                                sort_method: str) -> np.ndarray:
        """
        Get ROI sort order based on response characteristics for a single condition.
        
        Args:
            traces: ROI traces (n_rois, n_timepoints)
            time_vector: Time vector
            sort_method: Sorting method
            
        Returns:
            Array of ROI indices in sorted order
        """
        n_rois = traces.shape[0]
        
        if sort_method == 'peak_time':
            # Sort by time of peak response
            response_mask = time_vector >= 0  # Only consider post-stimulus
            
            peak_times = []
            for roi_idx in range(n_rois):
                response_trace = traces[roi_idx, response_mask]
                if len(response_trace) > 0:
                    peak_idx = np.argmax(np.abs(response_trace))
                    peak_time = time_vector[response_mask][peak_idx]
                    peak_times.append(peak_time)
                else:
                    peak_times.append(np.inf)  # No response data
            
            # Sort by peak time (earliest to latest)
            sort_indices = np.argsort(peak_times)
            
        elif sort_method == 'peak_magnitude':
            # Sort by magnitude of peak response
            response_mask = time_vector >= 0
            
            peak_magnitudes = []
            for roi_idx in range(n_rois):
                response_trace = traces[roi_idx, response_mask]
                if len(response_trace) > 0:
                    peak_mag = np.max(np.abs(response_trace))
                    peak_magnitudes.append(peak_mag)
                else:
                    peak_magnitudes.append(0)
            
            # Sort by peak magnitude (largest to smallest)
            sort_indices = np.argsort(peak_magnitudes)[::-1]
            
        elif sort_method == 'response_onset':
            # Sort by response onset time (threshold crossing)
            baseline_mask = time_vector < 0
            response_mask = time_vector >= 0
            
            onset_times = []
            for roi_idx in range(n_rois):
                baseline_std = np.std(traces[roi_idx, baseline_mask])
                threshold = 2 * baseline_std  # 2 standard deviations above baseline
                
                response_trace = traces[roi_idx, response_mask]
                onset_idx = np.where(np.abs(response_trace) > threshold)[0]
                
                if len(onset_idx) > 0:
                    onset_time = time_vector[response_mask][onset_idx[0]]
                    onset_times.append(onset_time)
                else:
                    onset_times.append(np.inf)  # No significant response
            
            # Sort by onset time (earliest to latest)
            sort_indices = np.argsort(onset_times)
            
        elif sort_method == 'baseline_activity':
            # Sort by baseline activity level
            baseline_mask = time_vector < 0
            
            baseline_activities = []
            for roi_idx in range(n_rois):
                baseline_activity = np.mean(np.abs(traces[roi_idx, baseline_mask]))
                baseline_activities.append(baseline_activity)
            
            # Sort by baseline activity (lowest to highest)
            sort_indices = np.argsort(baseline_activities)
            
        else:
            # No sorting - original order
            sort_indices = np.arange(n_rois)
        
        return sort_indices


    def plot_single_trial_raster(self, trial_data: Dict[str, Any], subject_id: str, 
                                trial_index: int, sort_method: str = 'none') -> None:
        """
        Plot raster heatmap for a single trial showing all ROI responses.
        
        Args:
            trial_data: Dictionary from extract_trial_segments_simple
            subject_id: Subject ID for plot title
            trial_index: Index of trial to plot (within valid trials)
            sort_method: 'none', 'max_peak', or 'min_peak'
        """
        try:
            import matplotlib.pyplot as plt
            
            if 'error' in trial_data:
                self.logger.error(f"SID_IMG_PRE: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            # Handle both old array format and new DataFrame format
            if 'df_trials_with_segments' in trial_data:
                # New DataFrame format
                df_with_segments = trial_data['df_trials_with_segments']
                valid_mask = trial_data['valid_trials_mask']
                df_valid = df_with_segments[valid_mask]
                
                if len(df_valid) == 0:
                    self.logger.error("SID_IMG_PRE: No valid trials with segments")
                    return
                
                if trial_index >= len(df_valid) or trial_index < 0:
                    self.logger.error(f"SID_IMG_PRE: Trial index {trial_index} out of range (0-{len(df_valid)-1})")
                    return
                
                # Get trial data
                trial_row = df_valid.iloc[trial_index]
                dff_segment = trial_row['dff_segment']  # (n_rois, n_timepoints)
                dff_time_vector = trial_row['dff_time_vector']
                original_trial_index = trial_row.name  # Original DataFrame index
                
            else:
                # Old array format (backward compatibility)
                dff_segments = trial_data['dff_segments']
                dff_time_vector = trial_data['dff_time_vector']
                df_trials = trial_data['df_trials']
                
                if trial_index >= len(dff_segments) or trial_index < 0:
                    self.logger.error(f"SID_IMG_PRE: Trial index {trial_index} out of range (0-{len(dff_segments)-1})")
                    return
                
                # Get trial data
                dff_segment = dff_segments[trial_index]  # (n_rois, n_timepoints)
                trial_row = df_trials.iloc[trial_index]
                original_trial_index = trial_row.name
            
            n_rois, n_timepoints = dff_segment.shape
            
            # Sort ROIs if requested
            if sort_method == 'max_peak':
                # Sort by maximum peak response (highest to lowest)
                peak_values = np.max(dff_segment, axis=1)
                roi_order = np.argsort(peak_values)[::-1]  # Descending order
                sorted_dff = dff_segment[roi_order, :]
                roi_labels = [f'ROI {roi_order[i]}' for i in range(n_rois)]
                sort_title = 'Sorted by Max Peak (High→Low)'
                
            elif sort_method == 'min_peak':
                # Sort by minimum peak response (lowest to highest)
                peak_values = np.min(dff_segment, axis=1)
                roi_order = np.argsort(peak_values)  # Ascending order
                sorted_dff = dff_segment[roi_order, :]
                roi_labels = [f'ROI {roi_order[i]}' for i in range(n_rois)]
                sort_title = 'Sorted by Min Peak (Low→High)'
                
            else:  # sort_method == 'none'
                roi_order = np.arange(n_rois)
                sorted_dff = dff_segment
                roi_labels = [f'ROI {i}' for i in range(n_rois)]
                sort_title = 'Original Order'
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Get trial information for title
            trial_info = []
            if 'trial_side' in trial_row:
                trial_info.append(f"Side: {trial_row['trial_side']}")
            if 'rewarded' in trial_row:
                reward_status = "Rewarded" if trial_row['rewarded'] == 1 else "Unrewarded"
                trial_info.append(f"Outcome: {reward_status}")
            if 'trial_index' in trial_row:
                trial_info.append(f"Trial #{trial_row['trial_index']}")
            
            trial_info_str = " | ".join(trial_info) if trial_info else f"Trial #{original_trial_index}"
            
            # Calculate colormap range for this trial
            vmin = np.percentile(sorted_dff, 5)
            vmax = np.percentile(sorted_dff, 95)
            
            # Create heatmap
            im = ax.imshow(sorted_dff, aspect='auto', cmap='viridis',
                        extent=[dff_time_vector[0], dff_time_vector[-1], 0, n_rois],
                        vmin=vmin, vmax=vmax, interpolation='nearest')
            
            # Add alignment line
            ax.axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.9, label='Alignment Point')
            
            # Formatting
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('ROI # (sorted)' if sort_method != 'none' else 'ROI #', fontsize=12)
            ax.set_title(f'Single Trial Raster - {subject_id}\n{trial_info_str} | {sort_title}', fontsize=14)
            
            # Set y-axis ticks to show ROI labels
            y_tick_spacing = max(1, n_rois // 15)  # Show ~15 labels max
            y_ticks = np.arange(0, n_rois, y_tick_spacing)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([roi_labels[i] for i in y_ticks], fontsize=8)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('DFF', fontsize=12)
            
            # Add trial statistics text box
            trial_stats = {
                'ROIs': n_rois,
                'Duration': f'{dff_time_vector[-1] - dff_time_vector[0]:.1f}s',
                'Data range': f'[{vmin:.3f}, {vmax:.3f}]',
                'Peak activity': f'{np.max(sorted_dff):.3f}',
                'Min activity': f'{np.min(sorted_dff):.3f}'
            }
            
            stats_text = '\n'.join([f'{k}: {v}' for k, v in trial_stats.items()])
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=10, verticalalignment='top')
            
            # Add behavioral events as vertical lines (if available)
            event_colors = {
                'start_flash_1': 'red', 'end_flash_1': 'orange', 'start_flash_2': 'blue',
                'end_flash_2': 'cyan', 'choice_start': 'green', 'reward_start': 'purple',
                'punish_start': 'black'
            }
            
            # Get alignment point for relative timing
            if 'df_trials_with_segments' in trial_data:
                alignment_point = trial_data['alignment_point']
                alignment_timestamp = trial_row.get(alignment_point, np.nan)
            else:
                alignment_point = trial_data['alignment_point']
                alignment_timestamp = trial_row.get(alignment_point, np.nan)
            
            if not pd.isna(alignment_timestamp):
                for event_name, color in event_colors.items():
                    if event_name in trial_row and not pd.isna(trial_row[event_name]):
                        # Calculate relative time to alignment point
                        event_time_rel = trial_row[event_name] - alignment_timestamp
                        
                        # Check if event falls within the plotted time window
                        if dff_time_vector[0] <= event_time_rel <= dff_time_vector[-1]:
                            ax.axvline(x=event_time_rel, color=color, alpha=0.7, 
                                    linestyle='-', linewidth=2)
                            
                            # Add event label at top of plot
                            ax.text(event_time_rel, n_rois * 0.95, event_name,
                                rotation=90, fontsize=8, color=color, 
                                ha='center', va='top', weight='bold')
            
            # Add sorting information
            if sort_method != 'none':
                peak_values_sorted = np.max(sorted_dff, axis=1) if sort_method == 'max_peak' else np.min(sorted_dff, axis=1)
                sort_info = f'Peak range: {np.min(peak_values_sorted):.3f} to {np.max(peak_values_sorted):.3f}'
                ax.text(0.98, 0.02, sort_info, transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=9, verticalalignment='bottom', horizontalalignment='right')
            
            plt.tight_layout()
            plt.show()
            
            # Log information
            self.logger.info(f"SID_IMG_PRE: Single trial raster plot - Trial {trial_index}, {n_rois} ROIs, Sort: {sort_method}")
            if sort_method != 'none':
                if sort_method == 'max_peak':
                    top_roi = roi_order[0]
                    self.logger.info(f"SID_IMG_PRE: Most active ROI: {top_roi} (peak: {np.max(dff_segment[top_roi, :]):.3f})")
                else:
                    bottom_roi = roi_order[0]
                    self.logger.info(f"SID_IMG_PRE: Least active ROI: {bottom_roi} (min: {np.min(dff_segment[bottom_roi, :]):.3f})")
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create single trial raster plot: {e}")



    def plot_trial_alignment_diagnostic(self, imaging_data: Dict[str, Any], behavioral_data: Dict[str, Any], 
                                    subject_id: str, trial_indices: List[int] = None, 
                                    window_sec: float = 10.0) -> None:
        """
        Plot diagnostic to check alignment between behavioral events, voltage traces, and neural activity
        BEFORE trial segmentation. Shows continuous data with event overlays.
        
        Args:
            imaging_data: Dictionary containing continuous imaging data
            behavioral_data: Dictionary containing behavioral data
            subject_id: Subject identifier for plot title
            trial_indices: List of trial indices to plot (default: first 5 trials)
            window_sec: Window around each trial to show (seconds)
        """
        try:
            import matplotlib.pyplot as plt
            
            df_trials = behavioral_data['df_trials']
            
            if trial_indices is None:
                trial_indices = list(range(min(300, len(df_trials))))
            
            self.logger.info(f"SID_IMG_PRE: Creating trial alignment diagnostic for {len(trial_indices)} trials")
            
            # Define event colors for consistency
            event_colors = {
                'trial_start_timestamp': 'black',
                'start_flash_1': 'red', 
                'end_flash_1': 'orange', 
                'start_flash_2': 'blue',
                'end_flash_2': 'cyan', 
                'choice_start': 'green', 
                'reward_start': 'purple',
                'punish_start': 'maroon'
            }
            
            # Plot each trial separately
            for trial_idx in trial_indices:
                if trial_idx >= len(df_trials):
                    self.logger.warning(f"SID_IMG_PRE: Trial index {trial_idx} out of range, skipping")
                    continue
                    
                # FIX: Use iloc to get the correct trial by position, not by index
                trial_row = df_trials.iloc[trial_idx]
                trial_start = trial_row.get('trial_start_timestamp', np.nan)
                
                if pd.isna(trial_start):
                    self.logger.warning(f"SID_IMG_PRE: Trial {trial_idx} has no trial_start_timestamp, skipping")
                    continue
                
                # Define window around trial
                window_start = trial_start - window_sec/2
                window_end = trial_start + window_sec/2
                
                # Extract imaging data for this window
                img_start_idx = np.searchsorted(imaging_data['imaging_time'], window_start)
                img_end_idx = np.searchsorted(imaging_data['imaging_time'], window_end)
                
                if img_start_idx >= img_end_idx:
                    self.logger.warning(f"SID_IMG_PRE: No imaging data for trial {trial_idx} window, skipping")
                    continue
                
                # Extract voltage data for this window
                vol_start_idx = np.searchsorted(imaging_data['vol_time'], window_start)
                vol_end_idx = np.searchsorted(imaging_data['vol_time'], window_end)
                
                # Get data segments
                img_time_segment = imaging_data['imaging_time'][img_start_idx:img_end_idx]
                dff_segment = imaging_data['dff_traces'][:, img_start_idx:img_end_idx]
                
                vol_time_segment = imaging_data['vol_time'][vol_start_idx:vol_end_idx] if vol_start_idx < vol_end_idx else np.array([])
                vol_start_segment = imaging_data['vol_start'][vol_start_idx:vol_end_idx] if vol_start_idx < vol_end_idx and imaging_data.get('vol_start') is not None else np.array([])
                vol_stim_vis_segment = imaging_data['vol_stim_vis'][vol_start_idx:vol_end_idx] if vol_start_idx < vol_end_idx and imaging_data.get('vol_stim_vis') is not None else np.array([])
                
                # FIX: Create figure with GridSpec for better control over subplot sizes and alignment
                from matplotlib.gridspec import GridSpec
                fig = plt.figure(figsize=(14, 10))
                gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[20, 1])
                
                # Create axes
                ax_raster = fig.add_subplot(gs[0, 0])
                ax_cbar = fig.add_subplot(gs[0, 1])
                ax_voltage = fig.add_subplot(gs[1, 0])
                
                # Get trial info for title
                trial_info = []
                if 'trial_side' in trial_row:
                    trial_info.append(f"Side: {trial_row['trial_side']}")
                if 'rewarded' in trial_row:
                    reward_status = "Rewarded" if trial_row['rewarded'] == 1 else "Unrewarded"
                    trial_info.append(f"Outcome: {reward_status}")
                if 'trial_index' in trial_row:
                    trial_info.append(f"Trial #{trial_row['trial_index']}")
                
                trial_info_str = " | ".join(trial_info) if trial_info else f"Trial #{trial_idx}"
                
                fig.suptitle(f'Trial Alignment Diagnostic - {subject_id}\n{trial_info_str}', fontsize=14)
                
                # Plot 1: ROI raster heatmap
                n_rois = dff_segment.shape[0]
                
                # Calculate colormap range for this window
                vmin = np.percentile(dff_segment, 5)
                vmax = np.percentile(dff_segment, 95)
                
                # Create heatmap
                im = ax_raster.imshow(dff_segment, aspect='auto', cmap='viridis',
                                    extent=[img_time_segment[0], img_time_segment[-1], 0, n_rois],
                                    vmin=vmin, vmax=vmax, interpolation='nearest')
                
                # FIX: Plot events from THIS SPECIFIC TRIAL only
                for event_name, color in event_colors.items():
                    if event_name in trial_row and not pd.isna(trial_row[event_name]):
                        event_time = trial_row[event_name] + trial_start
                        
                        # Check if event falls within the plotted time window
                        if img_time_segment[0] <= event_time <= img_time_segment[-1]:
                            ax_raster.axvline(x=event_time, color=color, alpha=0.8, 
                                            linestyle='-', linewidth=2)
                            
                            # Add event label at top of plot
                            ax_raster.text(event_time, n_rois * 0.95, event_name.replace('_', '\n'),
                                        rotation=0, fontsize=8, color=color, 
                                        ha='center', va='top', weight='bold')
                
                # Format raster plot
                ax_raster.set_ylabel('ROI #', fontsize=12)
                ax_raster.set_title('All ROI Activity (Continuous DFF)', fontsize=12)
                
                # Set y-axis ticks
                y_tick_spacing = max(1, n_rois // 10)
                y_ticks = np.arange(0, n_rois, y_tick_spacing)
                ax_raster.set_yticks(y_ticks)
                ax_raster.set_yticklabels([f'ROI {i}' for i in y_ticks])
                
                # Add colorbar in separate axis for better alignment
                cbar = plt.colorbar(im, cax=ax_cbar)
                cbar.set_label('DFF', fontsize=10)
                
                # Add trial statistics
                stats_text = f'ROIs: {n_rois} | Duration: {img_time_segment[-1] - img_time_segment[0]:.1f}s | Range: [{vmin:.3f}, {vmax:.3f}]'
                ax_raster.text(0.02, 0.02, stats_text, transform=ax_raster.transAxes,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                            fontsize=9, verticalalignment='bottom')
                
                # Plot 2: Voltage traces (vol_start and vol_stim_vis)
                # FIX: Ensure voltage plot x-axis exactly matches raster x-axis
                ax_voltage.set_xlim(ax_raster.get_xlim())  # Match x-limits exactly
                
                if len(vol_time_segment) > 0:
                    # Plot vol_start
                    if len(vol_start_segment) > 0:
                        ax_voltage.plot(vol_time_segment, vol_start_segment, 'r-', 
                                    linewidth=2, alpha=0.8, label='vol_start')
                    
                    # Plot vol_stim_vis (offset for visibility)
                    if len(vol_stim_vis_segment) > 0:
                        offset = 1.5
                        ax_voltage.plot(vol_time_segment, vol_stim_vis_segment + offset, 'g-', 
                                    linewidth=2, alpha=0.8, label='vol_stim_vis')
                    
                    # FIX: Add behavioral events from THIS SPECIFIC TRIAL to voltage plot
                    for event_name, color in event_colors.items():
                        if event_name in trial_row and not pd.isna(trial_row[event_name]):
                            event_time = trial_row[event_name]
                            
                            # Check if event falls within voltage time window AND overall window
                            if (len(vol_time_segment) > 0 and 
                                vol_time_segment[0] <= event_time <= vol_time_segment[-1] and
                                img_time_segment[0] <= event_time <= img_time_segment[-1]):
                                
                                ax_voltage.axvline(x=event_time, color=color, alpha=0.8, 
                                                linestyle='-', linewidth=2)
                                
                                # Add event label
                                y_pos = ax_voltage.get_ylim()[1] * 0.9
                                ax_voltage.text(event_time, y_pos, event_name.replace('_', '\n'),
                                            rotation=0, fontsize=8, color=color, 
                                            ha='center', va='top', weight='bold')
                    
                    ax_voltage.set_xlabel('Time (s)', fontsize=12)
                    ax_voltage.set_ylabel('Voltage + Offset', fontsize=12)
                    ax_voltage.set_title('Voltage Traces (vol_start and vol_stim_vis)', fontsize=12)
                    ax_voltage.legend()
                    ax_voltage.grid(True, alpha=0.3)
                    
                    # Add voltage statistics
                    vol_stats = []
                    if len(vol_start_segment) > 0:
                        vol_stats.append(f'vol_start: [{np.min(vol_start_segment):.2f}, {np.max(vol_start_segment):.2f}]')
                    if len(vol_stim_vis_segment) > 0:
                        vol_stats.append(f'vol_stim_vis: [{np.min(vol_stim_vis_segment):.2f}, {np.max(vol_stim_vis_segment):.2f}]')
                    
                    if vol_stats:
                        vol_stats_text = ' | '.join(vol_stats)
                        ax_voltage.text(0.02, 0.02, vol_stats_text, transform=ax_voltage.transAxes,
                                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
                                    fontsize=9, verticalalignment='bottom')
                else:
                    ax_voltage.text(0.5, 0.5, 'No voltage data available', ha='center', va='center', 
                                transform=ax_voltage.transAxes, fontsize=12)
                    ax_voltage.set_xlabel('Time (s)', fontsize=12)
                    ax_voltage.set_ylabel('Voltage', fontsize=12)
                    ax_voltage.set_title('Voltage Traces (no data)', fontsize=12)
                
                # Add event legend at bottom
                legend_elements = []
                for event_name, color in event_colors.items():
                    if event_name in trial_row and not pd.isna(trial_row[event_name]):
                        from matplotlib.lines import Line2D
                        legend_elements.append(Line2D([0], [0], color=color, lw=2, label=event_name))
                
                if legend_elements:
                    fig.legend(handles=legend_elements, loc='lower center', 
                            bbox_to_anchor=(0.5, -0.02), ncol=min(4, len(legend_elements)),
                            fontsize=8, title='Behavioral Events')
                
                plt.tight_layout()
                plt.show()
                
                # Log trial-specific information
                actual_trial_index = trial_row.get('trial_index', trial_idx)
                self.logger.info(f"SID_IMG_PRE: Trial {trial_idx} (actual trial #{actual_trial_index}) diagnostic - {n_rois} ROIs, "
                            f"Window: {window_start:.1f}-{window_end:.1f}s")
                
                # Check for potential alignment issues
                alignment_issues = []
                
                # Check if trial_start pulse aligns with vol_start
                if len(vol_start_segment) > 0 and len(vol_time_segment) > 0:
                    trial_start_time = trial_row['trial_start_timestamp']
                    
                    # Find vol_start peaks in the window
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(vol_start_segment, height=0.5)
                    if len(peaks) > 0:
                        peak_times = vol_time_segment[peaks]
                        closest_peak = peak_times[np.argmin(np.abs(peak_times - trial_start_time))]
                        drift = abs(closest_peak - trial_start_time)
                        
                        if drift > 0.05:  # 50ms threshold
                            alignment_issues.append(f"Trial start drift: {drift*1000:.1f}ms")
                    else:
                        alignment_issues.append("No vol_start pulse found")
                
                # Check if visual stimulus events align with vol_stim_vis
                if len(vol_stim_vis_segment) > 0 and len(vol_time_segment) > 0:
                    vis_events = ['start_flash_1', 'end_flash_1', 'start_flash_2', 'end_flash_2']
                    for event in vis_events:
                        if event in trial_row and not pd.isna(trial_row[event]):                            
                            event_time = trial_row[event] + trial_start
                            # Find corresponding voltage activity
                            time_idx = np.searchsorted(vol_time_segment, event_time)
                            if 0 < time_idx < len(vol_stim_vis_segment) - 1:
                                vol_value = vol_stim_vis_segment[time_idx]
                                expected_high = 'start' in event
                                actual_high = vol_value > 0.5                                
                                if expected_high != actual_high:
                                    fig
                                    plt.plot(vol_time_segment,vol_start_segment)
                                    plt.plot(vol_time_segment,vol_stim_vis_segment)                                    
                                    alignment_issues.append(f"{event}: voltage mismatch")
                
                if alignment_issues:
                    self.logger.warning(f"SID_IMG_PRE: Trial {trial_idx} potential alignment issues: {', '.join(alignment_issues)}")
                else:
                    self.logger.info(f"SID_IMG_PRE: Trial {trial_idx} alignment looks good")
            
            self.logger.info(f"SID_IMG_PRE: Trial alignment diagnostic completed for {len(trial_indices)} trials")
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create trial alignment diagnostic: {e}")



    def plot_session_alignment_overview(self, imaging_data: Dict[str, Any], behavioral_data: Dict[str, Any], 
                                    subject_id: str, time_window: float = 300.0) -> None:
        """
        Plot session overview showing alignment between imaging, voltage, and behavioral events
        for the first few minutes of the session.
        
        Args:
            imaging_data: Dictionary containing continuous imaging data
            behavioral_data: Dictionary containing behavioral data
            subject_id: Subject identifier for plot title
            time_window: Time window to show in seconds (default: 5 minutes)
        """
        try:
            import matplotlib.pyplot as plt
            
            df_trials = behavioral_data['df_trials']
            
            self.logger.info(f"SID_IMG_PRE: Creating session alignment overview for first {time_window}s")
            
            # Find time window
            window_start = imaging_data['imaging_time'][0]
            window_end = min(window_start + time_window, imaging_data['imaging_time'][-1])
            
            # Extract imaging data for window
            img_start_idx = np.searchsorted(imaging_data['imaging_time'], window_start)
            img_end_idx = np.searchsorted(imaging_data['imaging_time'], window_end)
            
            img_time_segment = imaging_data['imaging_time'][img_start_idx:img_end_idx]
            dff_segment = imaging_data['dff_traces'][:, img_start_idx:img_end_idx]
            
            # Extract voltage data for window
            vol_start_idx = np.searchsorted(imaging_data['vol_time'], window_start)
            vol_end_idx = np.searchsorted(imaging_data['vol_time'], window_end)
            
            vol_time_segment = imaging_data['vol_time'][vol_start_idx:vol_end_idx]
            vol_start_segment = imaging_data['vol_start'][vol_start_idx:vol_end_idx] if imaging_data.get('vol_start') is not None else np.array([])
            vol_stim_vis_segment = imaging_data['vol_stim_vis'][vol_start_idx:vol_end_idx] if imaging_data.get('vol_stim_vis') is not None else np.array([])
            
            # Create figure
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            fig.suptitle(f'Session Alignment Overview - {subject_id}\nFirst {time_window}s', fontsize=14)
            
            # Plot 1: Population DFF activity
            population_dff = np.mean(dff_segment, axis=0)
            axes[0].plot(img_time_segment, population_dff, 'b-', linewidth=1.5, alpha=0.8)
            axes[0].set_ylabel('Mean DFF')
            axes[0].set_title('Population Neural Activity')
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Individual ROI traces (first 10 ROIs)
            n_rois_to_show = min(10, dff_segment.shape[0])
            colors = plt.cm.tab10(np.linspace(0, 1, n_rois_to_show))
            for roi_idx in range(n_rois_to_show):
                offset = roi_idx * 1.0
                axes[1].plot(img_time_segment, dff_segment[roi_idx, :] + offset, 
                            color=colors[roi_idx], linewidth=0.8, alpha=0.8, 
                            label=f'ROI {roi_idx}')
            axes[1].set_ylabel('DFF + Offset')
            axes[1].set_title(f'Individual ROI Traces (first {n_rois_to_show})')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Voltage start channel
            if len(vol_start_segment) > 0:
                axes[2].plot(vol_time_segment, vol_start_segment, 'r-', linewidth=1.5, alpha=0.8)
                
                # Mark detected peaks
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(vol_start_segment, height=0.5, distance=30)  # 30 samples ~ 30ms
                if len(peaks) > 0:
                    peak_times = vol_time_segment[peaks]
                    axes[2].scatter(peak_times, vol_start_segment[peaks], 
                                color='red', s=30, zorder=5, label=f'{len(peaks)} pulses')
                    axes[2].legend()
            else:
                axes[2].text(0.5, 0.5, 'No vol_start data', ha='center', va='center', transform=axes[2].transAxes)
            
            axes[2].set_ylabel('vol_start')
            axes[2].set_title('Trial Start Pulses')
            axes[2].grid(True, alpha=0.3)
            
            # Plot 4: Voltage stimulus channel
            if len(vol_stim_vis_segment) > 0:
                axes[3].plot(vol_time_segment, vol_stim_vis_segment, 'g-', linewidth=1.5, alpha=0.8)
            else:
                axes[3].text(0.5, 0.5, 'No vol_stim_vis data', ha='center', va='center', transform=axes[3].transAxes)
            
            axes[3].set_ylabel('vol_stim_vis')
            axes[3].set_xlabel('Time (s)')
            axes[3].set_title('Visual Stimulus Channel')
            axes[3].grid(True, alpha=0.3)
            
            # Add behavioral events as vertical lines to ALL plots
            event_colors = {
                'trial_start_timestamp': 'black',
                'start_flash_1': 'red', 
                'end_flash_1': 'orange', 
                'start_flash_2': 'blue',
                'end_flash_2': 'cyan', 
                'choice_start': 'green', 
                'reward_start': 'purple',
                'punish_start': 'maroon'
            }
            
            # Get trials within the window
            trials_in_window = df_trials[
                (df_trials['trial_start_timestamp'] >= window_start) & 
                (df_trials['trial_start_timestamp'] <= window_end)
            ]
            
            for _, trial_row in trials_in_window.iterrows():
                for event_name, color in event_colors.items():
                    if event_name in trial_row and not pd.isna(trial_row[event_name]):
                        event_time = trial_row[event_name]
                        
                        if window_start <= event_time <= window_end:
                            for ax in axes:
                                ax.axvline(x=event_time, color=color, alpha=0.6, 
                                        linestyle='-', linewidth=1)
            
            # Add trial numbers
            for trial_idx, trial_row in trials_in_window.iterrows():
                trial_start = trial_row.get('trial_start_timestamp', np.nan)
                if not pd.isna(trial_start) and window_start <= trial_start <= window_end:
                    # Add trial number at top of first plot
                    axes[0].text(trial_start, axes[0].get_ylim()[1] * 0.9, 
                            f'T{trial_row.get("trial_index", trial_idx)}',
                            rotation=90, fontsize=8, ha='center', va='top')
            
            plt.tight_layout()
            plt.show()
            
            self.logger.info(f"SID_IMG_PRE: Session overview shows {len(trials_in_window)} trials in {time_window}s window")
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create session alignment overview: {e}")




    def analyze_roi_responsiveness(self, imaging_data: Dict[str, Any], trial_data: Dict[str, Any], subject_id: str) -> Dict[str, Any]:
        """
        Comprehensive ROI responsiveness analysis across multiple dimensions.
        
        Returns:
            Dictionary with hierarchical responsiveness data structure
        """
        try:
            # Initialize responsiveness structure
            responsiveness_data = {
                'subject_id': subject_id,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'roi_responsiveness': {},  # Main ROI data
                'summary_stats': {},       # Cross-ROI summaries
                'analysis_parameters': {}   # Analysis settings
            }
            
            # Define analysis parameters
            analysis_params = {
                # 'alignment_events': ['trial_start', 'start_flash_1', 'start_flash_2', 'choice_start', 'reward_start', 'servo_in'],
                'alignment_events': ['start_flash_1','choice_start'],
                # 'trial_filters': {
                #     'all': lambda df: np.ones(len(df), dtype=bool),
                #     'left': lambda df: df['trial_side'] == 'left',
                #     'right': lambda df: df['trial_side'] == 'right',
                #     'rewarded': lambda df: df['rewarded'] == 1,
                #     'punished': lambda df: df['punished'] == 0,
                #     'left_rewarded': lambda df: (df['trial_side'] == 'left') & (df['rewarded'] == 1),
                #     'right_rewarded': lambda df: (df['trial_side'] == 'right') & (df['rewarded'] == 1),
                #     'left_punished': lambda df: (df['trial_side'] == 'left') & (df['punished'] == 1),
                #     'right_punished': lambda df: (df['trial_side'] == 'right') & (df['punished'] == 1)
                # },
                'trial_filters': {
                    'all': lambda df: np.ones(len(df), dtype=bool),
                    'left': lambda df: df['trial_side'] == 'left',
                    'right': lambda df: df['trial_side'] == 'right',
                },                
                'responsiveness_types': {
                    'activation': {'direction': 'positive', 'window': (0.0, 2.0)},
                    'suppression': {'direction': 'negative', 'window': (0.0, 2.0)},
                    'predictive': {'direction': 'either', 'window': (-1.0, 0.0)},
                    'delayed': {'direction': 'either', 'window': (1.0, 4.0)},
                    'ramping_up': {'direction': 'ramp_positive', 'window': (-1.0, 2.0)},
                    'ramping_down': {'direction': 'ramp_negative', 'window': (-1.0, 2.0)},
                    'transient': {'direction': 'transient', 'window': (0.0, 1.0)},
                    'sustained': {'direction': 'sustained', 'window': (0.0, 4.0)}
                },
                # 'statistical_thresholds': {
                #     'p_value': 0.01,
                #     'effect_size_min': 0.2,
                #     'min_trials': 5,
                #     'baseline_window': (-2.0, -0.5)
                # },
                'statistical_thresholds': {
                    'p_value': 0.05,           # <- More lenient
                    'effect_size_min': 0.1,    # <- More lenient  
                    'min_trials': 5,
                    'baseline_window': (-2.0, -0.5)
                }
            }
            
            # Add after defining analysis_params:
            self.logger.info(f"SID_IMG_PRE: Statistical thresholds - p_value: {analysis_params['statistical_thresholds']['p_value']}, "
                            f"effect_size_min: {analysis_params['statistical_thresholds']['effect_size_min']}")            
            
            # Get ROI count and basic info
            responsiveness_data['analysis_parameters'] = analysis_params
            n_rois = imaging_data['dff_traces'].shape[0]
                    
            
            # Initialize ROI responsiveness structure
            for roi_idx in range(n_rois):
                responsiveness_data['roi_responsiveness'][roi_idx] = {
                    'roi_id': roi_idx,
                    'basic_properties': {},
                    'event_responses': {},
                    'cross_event_patterns': {},
                    'reliability_metrics': {}
                }
            
            # Analyze each alignment event
            for event_name in analysis_params['alignment_events']:
                self.logger.info(f"SID_IMG_PRE: Analyzing responsiveness to {event_name}")
                
                # Extract trial segments for this event
                event_trial_data = self.extract_trial_segments_simple(
                    # Need to pass the full imaging data and df_trials here
                    # This is a limitation - we'd need to store them or restructure
                    # For now, let's assume we have access to them
                    imaging_data, trial_data, 
                    alignment=event_name, pre_sec=3.0, post_sec=5.0
                )
                
                if 'error' in event_trial_data:
                    self.logger.warning(f"SID_IMG_PRE: Skipping {event_name} - {event_trial_data['error']}")
                    continue
                
                if 'df_trials_with_segments' in event_trial_data:
                    df_valid = event_trial_data['df_trials_with_segments'][event_trial_data['valid_trials_mask']]
                    common_time_vector = df_valid.iloc[0]['dff_time_vector']
                else:
                    return {'error': 'Need DataFrame format for responsiveness analysis'}                
                
                
                # Analyze each trial filter condition
                for filter_name, filter_func in analysis_params['trial_filters'].items():
                    self._analyze_event_filter_responsiveness(
                        event_trial_data, event_name, filter_name, filter_func,
                        analysis_params, responsiveness_data, common_time_vector
                    )
            
            # Compute cross-event patterns and summaries
            self._compute_cross_event_patterns(responsiveness_data, analysis_params)
            
            # NEW: Compute basic properties and reliability metrics
            self._compute_roi_basic_properties(responsiveness_data, imaging_data)
            self._compute_roi_reliability_metrics(responsiveness_data)                        
            
            self._compute_summary_statistics(responsiveness_data, n_rois)
            
            return responsiveness_data
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to analyze ROI responsiveness: {e}")
            return {'error': str(e)}


    def _analyze_event_filter_responsiveness(self, event_trial_data: Dict[str, Any], 
                                        event_name: str, filter_name: str, filter_func,
                                        analysis_params: Dict, responsiveness_data: Dict,
                                        common_time_vector: np.ndarray) -> None:
        """Analyze responsiveness for a specific event and trial filter combination."""
        try:
            from scipy import stats
            
            df_event_valid = event_trial_data['df_trials_with_segments'][event_trial_data['valid_trials_mask']]
            
            # Apply trial filter
            filter_mask = filter_func(df_event_valid)
            if not np.any(filter_mask) or np.sum(filter_mask) < analysis_params['statistical_thresholds']['min_trials']:
                return
            
            filtered_trials = df_event_valid[filter_mask]
            n_rois = filtered_trials.iloc[0]['dff_segment'].shape[0]
            
            # Define time windows
            baseline_window = analysis_params['statistical_thresholds']['baseline_window']
            baseline_mask = (common_time_vector >= baseline_window[0]) & (common_time_vector <= baseline_window[1])
            
            # Analyze each responsiveness type
            for response_type, type_params in analysis_params['responsiveness_types'].items():
                response_window = type_params['window']
                response_mask = (common_time_vector >= response_window[0]) & (common_time_vector <= response_window[1])
                
                # Analyze each ROI
                for roi_idx in range(n_rois):
                    responsiveness_result = self._test_roi_responsiveness(
                        filtered_trials, roi_idx, baseline_mask, response_mask,
                        common_time_vector, response_type, type_params, analysis_params
                    )
                    
                    # Store result in hierarchical structure
                    if event_name not in responsiveness_data['roi_responsiveness'][roi_idx]['event_responses']:
                        responsiveness_data['roi_responsiveness'][roi_idx]['event_responses'][event_name] = {}
                    
                    if filter_name not in responsiveness_data['roi_responsiveness'][roi_idx]['event_responses'][event_name]:
                        responsiveness_data['roi_responsiveness'][roi_idx]['event_responses'][event_name][filter_name] = {}
                    
                    responsiveness_data['roi_responsiveness'][roi_idx]['event_responses'][event_name][filter_name][response_type] = responsiveness_result
                    
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Error in _analyze_event_filter_responsiveness: {e}")




    def _test_roi_responsiveness(self, filtered_trials: pd.DataFrame, roi_idx: int,
                            baseline_mask: np.ndarray, response_mask: np.ndarray,
                            time_vector: np.ndarray, response_type: str, 
                            type_params: Dict, analysis_params: Dict) -> Dict[str, Any]:
        """Test a specific type of responsiveness for one ROI."""
        try:
            from scipy import stats
            from scipy.signal import find_peaks
            
            # Extract baseline and response values across trials
            baseline_values = []
            response_values = []
            trial_traces = []
            
            # filtered_trials = filtered_trials.iloc[0:6]
            for _, trial_row in filtered_trials.iterrows():
                dff_segment = trial_row['dff_segment']
                roi_trace = dff_segment[roi_idx, :]
                trial_traces.append(roi_trace)
                
                baseline_val = np.mean(roi_trace[baseline_mask])
                baseline_values.append(baseline_val)
                
                # Calculate response value based on response type
                response_trace = roi_trace[response_mask]
                response_time = time_vector[response_mask]
                
                if type_params['direction'] == 'positive':
                    response_val = np.max(response_trace)
                elif type_params['direction'] == 'negative':
                    response_val = np.min(response_trace)
                elif type_params['direction'] == 'either':
                    max_pos = np.max(response_trace)
                    min_neg = np.min(response_trace)
                    response_val = max_pos if abs(max_pos) > abs(min_neg) else min_neg
                elif type_params['direction'] == 'ramp_positive':
                    # Linear trend (slope)
                    slope, _, _, _, _ = stats.linregress(response_time, response_trace)
                    response_val = slope
                elif type_params['direction'] == 'ramp_negative':
                    slope, _, _, _, _ = stats.linregress(response_time, response_trace)
                    response_val = -slope  # Negative slope for ramping down
                elif type_params['direction'] == 'transient':
                    # Peak magnitude in early part of response window
                    early_response = response_trace[:len(response_trace)//2]
                    response_val = np.max(np.abs(early_response))
                elif type_params['direction'] == 'sustained':
                    # Mean activity in later part of response window
                    late_response = response_trace[len(response_trace)//2:]
                    response_val = np.mean(late_response)
                else:
                    response_val = np.mean(response_trace)
                
                response_values.append(response_val)
            
            # FIX: Add debugging for p-value calculation
            self.logger.debug(f"ROI {roi_idx}: baseline_values = {baseline_values[:3]}... (n={len(baseline_values)})")
            self.logger.debug(f"ROI {roi_idx}: response_values = {response_values[:3]}... (n={len(response_values)})")
            
            # Statistical tests
            if len(baseline_values) >= analysis_params['statistical_thresholds']['min_trials']:
                # FIX: Check for identical values which would cause invalid t-test
                baseline_var = np.var(baseline_values)
                response_var = np.var(response_values)
                
                if baseline_var < 1e-15 and response_var < 1e-15:
                    # Both baseline and response are essentially constant
                    self.logger.debug(f"ROI {roi_idx}: Both baseline and response are constant, setting p=1.0")
                    p_value = 1.0
                    t_stat = 0.0
                elif baseline_var < 1e-15 or response_var < 1e-15:
                    # One is constant - use one-sample test against the constant
                    if baseline_var < 1e-15:
                        # Baseline is constant, test if response differs from baseline mean
                        baseline_mean = np.mean(baseline_values)
                        t_stat, p_value = stats.ttest_1samp(response_values, baseline_mean)
                    else:
                        # Response is constant, test if baseline differs from response mean
                        response_mean = np.mean(response_values)
                        t_stat, p_value = stats.ttest_1samp(baseline_values, response_mean)
                else:
                    # Normal case - paired t-test
                    t_stat, p_value = stats.ttest_rel(response_values, baseline_values)
                
                # FIX: Additional validation
                if np.isnan(p_value) or np.isinf(p_value):
                    self.logger.warning(f"ROI {roi_idx}: Invalid p-value, setting to 1.0")
                    p_value = 1.0
                    t_stat = 0.0
                
                # Ensure p-value is in valid range [0, 1]
                p_value = np.clip(p_value, 1e-15, 1.0)
                
                self.logger.debug(f"ROI {roi_idx}: t_stat = {t_stat:.6f}, p_value = {p_value:.2e}")
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(baseline_values) + np.var(response_values)) / 2)
                if pooled_std > 1e-15:
                    effect_size = (np.mean(response_values) - np.mean(baseline_values)) / pooled_std
                else:
                    effect_size = 0.0
                
                # Reliability metrics
                trial_to_trial_corr = self._calculate_trial_reliability_fast(trial_traces, response_mask)
                
                # Determine significance
                is_significant = (p_value < analysis_params['statistical_thresholds']['p_value'] and 
                                abs(effect_size) >= analysis_params['statistical_thresholds']['effect_size_min'])
                
                # Response magnitude and timing
                mean_trace = np.mean(trial_traces, axis=0)
                response_magnitude = np.max(np.abs(mean_trace[response_mask]))
                response_timing = time_vector[response_mask][np.argmax(np.abs(mean_trace[response_mask]))]
                
                return {
                    'is_responsive': is_significant,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    't_statistic': t_stat,
                    'response_magnitude': response_magnitude,
                    'response_timing': response_timing,
                    'baseline_mean': np.mean(baseline_values),
                    'baseline_std': np.std(baseline_values),
                    'response_mean': np.mean(response_values),
                    'response_std': np.std(response_values),
                    'n_trials': len(baseline_values),
                    'trial_reliability': trial_to_trial_corr,
                    'response_direction': 'positive' if effect_size > 0 else 'negative',
                    'response_type': response_type,
                    'mean_trace': mean_trace.tolist(),
                    'sem_trace': (np.std(trial_traces, axis=0) / np.sqrt(len(trial_traces))).tolist()
                }
            else:
                return {
                    'is_responsive': False,
                    'p_value': 1.0,
                    'effect_size': 0.0,
                    'n_trials': len(baseline_values),
                    'insufficient_trials': True
                }
                
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Error in _test_roi_responsiveness: {e}")
            return {'error': str(e)}







    def _calculate_trial_reliability(self, trial_traces: List[np.ndarray], response_mask: np.ndarray) -> float:
        """Calculate trial-to-trial reliability using optimized correlation."""
        try:
            if len(trial_traces) < 3:
                return np.nan
            
            # Extract response traces
            response_traces = [trace[response_mask] for trace in trial_traces]
            
            # Filter out constant traces before correlation analysis
            variable_traces = []
            for trace in response_traces:
                if np.var(trace) > 1e-10 and not np.any(np.isnan(trace)):
                    variable_traces.append(trace)
            
            # Need at least 2 variable traces for correlation
            if len(variable_traces) < 2:
                return np.nan
            
            # OPTIMIZATION 1: Limit number of trials for correlation computation
            max_trials_for_correlation = 50  # Adjustable parameter
            if len(variable_traces) > max_trials_for_correlation:
                # Randomly sample trials to reduce computation
                import random
                sampled_indices = random.sample(range(len(variable_traces)), max_trials_for_correlation)
                variable_traces = [variable_traces[i] for i in sampled_indices]
            
            # OPTIMIZATION 2: Use vectorized correlation computation instead of loops
            if len(variable_traces) <= 10:
                # For small numbers of trials, use pairwise correlation
                correlations = []
                for i in range(len(variable_traces)):
                    for j in range(i+1, len(variable_traces)):
                        try:
                            corr, _ = stats.pearsonr(variable_traces[i], variable_traces[j])
                            if not np.isnan(corr):
                                correlations.append(corr)
                        except:
                            continue
                
                return np.mean(correlations) if correlations else np.nan
            
            else:
                # OPTIMIZATION 3: For larger numbers, use matrix correlation approach
                # Stack traces into matrix (n_trials, n_timepoints)
                trace_matrix = np.array(variable_traces)
                
                # Compute correlation matrix using numpy's corrcoef (much faster)
                corr_matrix = np.corrcoef(trace_matrix)
                
                # Extract upper triangle (excluding diagonal)
                upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                
                # Remove NaN values
                valid_correlations = upper_triangle[~np.isnan(upper_triangle)]
                
                return np.mean(valid_correlations) if len(valid_correlations) > 0 else np.nan
                
        except Exception as e:
            self.logger.debug(f"SID_IMG_PRE: Error in trial reliability calculation: {e}")
            return np.nan



    def _calculate_trial_reliability_fast(self, trial_traces: List[np.ndarray], response_mask: np.ndarray) -> float:
        """Fast approximation of trial reliability using coefficient of variation."""
        try:
            if len(trial_traces) < 3:
                return np.nan
            
            # Extract response traces
            response_traces = [trace[response_mask] for trace in trial_traces]
            
            # Filter constant traces
            variable_traces = []
            for trace in response_traces:
                if np.var(trace) > 1e-10 and not np.any(np.isnan(trace)):
                    variable_traces.append(trace)
            
            if len(variable_traces) < 2:
                return np.nan
            
            # Stack into matrix and compute mean response per trial
            trace_matrix = np.array(variable_traces)
            trial_means = np.mean(trace_matrix, axis=1)  # Mean response per trial
            
            # Reliability = 1 - coefficient of variation
            # (lower variability across trials = higher reliability)
            if np.mean(trial_means) != 0:
                cv = np.std(trial_means) / abs(np.mean(trial_means))
                reliability = max(0, 1 - cv)  # Bound between 0 and 1
            else:
                reliability = 0
            
            return reliability
            
        except Exception as e:
            self.logger.debug(f"SID_IMG_PRE: Error in fast trial reliability calculation: {e}")
            return np.nan






    def _compute_roi_basic_properties(self, responsiveness_data: Dict, imaging_data: Dict[str, Any]) -> None:
        """Compute basic ROI properties."""
        try:
            n_rois = imaging_data['dff_traces'].shape[0]
            
            for roi_idx in range(n_rois):
                # Basic signal properties
                roi_trace = imaging_data['dff_traces'][roi_idx, :]
                
                basic_props = {
                    'mean_activity': float(np.mean(roi_trace)),
                    'activity_variance': float(np.var(roi_trace)),
                    'baseline_activity': float(np.percentile(roi_trace, 10)),  # 10th percentile as baseline
                    'peak_activity': float(np.percentile(roi_trace, 90)),      # 90th percentile as peak
                    'dynamic_range': float(np.percentile(roi_trace, 90) - np.percentile(roi_trace, 10)),
                    'activity_skewness': float(stats.skew(roi_trace)),
                    'activity_kurtosis': float(stats.kurtosis(roi_trace)),
                    'roi_type': 'excitatory' if imaging_data['roi_labels'][roi_idx] == -1 else 
                            'inhibitory' if imaging_data['roi_labels'][roi_idx] == 1 else 'unlabeled'
                }
                
                responsiveness_data['roi_responsiveness'][roi_idx]['basic_properties'] = basic_props
                
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Error computing basic properties: {e}")

    def _compute_roi_reliability_metrics(self, responsiveness_data: Dict) -> None:
        """Compute ROI reliability metrics across all conditions."""
        try:
            for roi_idx, roi_data in responsiveness_data['roi_responsiveness'].items():
                reliability_metrics = {
                    'mean_trial_reliability': np.nan,
                    'response_consistency': 0,
                    'significant_response_rate': 0,
                    'effect_size_consistency': np.nan
                }
                
                # Collect reliability scores and effect sizes across all conditions
                trial_reliabilities = []
                effect_sizes = []
                significant_responses = 0
                total_tests = 0
                
                for event_name, event_data in roi_data.get('event_responses', {}).items():
                    for filter_name, filter_data in event_data.items():
                        for response_type, result in filter_data.items():
                            if not result.get('insufficient_trials', False):
                                total_tests += 1
                                
                                if result.get('is_responsive', False):
                                    significant_responses += 1
                                    effect_sizes.append(result.get('effect_size', 0))
                                
                                reliability = result.get('trial_reliability', np.nan)
                                if not np.isnan(reliability):
                                    trial_reliabilities.append(reliability)
                
                # Compute summary metrics
                if trial_reliabilities:
                    reliability_metrics['mean_trial_reliability'] = np.mean(trial_reliabilities)
                
                if total_tests > 0:
                    reliability_metrics['significant_response_rate'] = significant_responses / total_tests
                
                if len(effect_sizes) > 1:
                    reliability_metrics['effect_size_consistency'] = 1.0 - (np.std(effect_sizes) / (np.mean(np.abs(effect_sizes)) + 1e-10))
                    reliability_metrics['response_consistency'] = len(effect_sizes) / total_tests
                
                roi_data['reliability_metrics'] = reliability_metrics
                
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Error computing reliability metrics: {e}")




    def _compute_cross_event_patterns(self, responsiveness_data: Dict, analysis_params: Dict) -> None:
        """Compute patterns across different events for each ROI."""
        try:
            events = analysis_params['alignment_events']
            filters = list(analysis_params['trial_filters'].keys())

            for roi_idx, roi_data in responsiveness_data['roi_responsiveness'].items():
                cross_patterns = {
                    'responsive_events': [],
                    'preferred_response_types': {},
                    'event_selectivity': {},
                    'temporal_patterns': {}
                }
                
                # Find events this ROI responds to
                for event_name, event_data in roi_data.get('event_responses', {}).items():
                    for filter_name, filter_data in event_data.items():
                        for response_type, response_result in filter_data.items():
                            if response_result.get('is_responsive', False):
                                cross_patterns['responsive_events'].append({
                                    'event': event_name,
                                    'filter': filter_name,
                                    'response_type': response_type,
                                    'effect_size': response_result['effect_size'],
                                    'p_value': response_result['p_value']
                                })
                
                # Determine preferred response types
                response_type_counts = {}
                for resp in cross_patterns['responsive_events']:
                    resp_type = resp['response_type']
                    if resp_type not in response_type_counts:
                        response_type_counts[resp_type] = []
                    response_type_counts[resp_type].append(abs(resp['effect_size']))
                
                # Rank response types by frequency and magnitude
                for resp_type, effect_sizes in response_type_counts.items():
                    cross_patterns['preferred_response_types'][resp_type] = {
                        'frequency': len(effect_sizes),
                        'mean_magnitude': np.mean(effect_sizes),
                        'max_magnitude': np.max(effect_sizes)
                    }
               
               
                # Compute temporal patterns
                temporal_patterns = {}
                if cross_patterns['responsive_events']:
                    # Calculate response timing consistency across events
                    response_timings = [resp['response_timing'] for resp in cross_patterns['responsive_events'] 
                                      if 'response_timing' in resp]
                    if response_timings:
                        temporal_patterns['mean_response_time'] = np.mean(response_timings)
                        temporal_patterns['timing_variability'] = np.std(response_timings)
                        temporal_patterns['early_responder'] = np.mean(response_timings) < 1.0  # Responds within 1s
                        temporal_patterns['late_responder'] = np.mean(response_timings) > 2.0   # Responds after 2s
                
                # Compute event selectivity metrics
                event_selectivity = {}
                if cross_patterns['responsive_events']:
                    events_responded = set([resp['event'] for resp in cross_patterns['responsive_events']])
                    filters_responded = set([resp['filter'] for resp in cross_patterns['responsive_events']])
                    
                    event_selectivity['n_events_responsive'] = len(events_responded)
                    event_selectivity['n_filters_responsive'] = len(filters_responded)
                    event_selectivity['selectivity_index'] = len(events_responded) / len(events)  # 0-1 scale
                    event_selectivity['is_event_selective'] = len(events_responded) == 1  # Only responds to one event
                    event_selectivity['is_broadly_responsive'] = len(events_responded) >= len(events) * 0.8
                
                cross_patterns['temporal_patterns'] = temporal_patterns
                cross_patterns['event_selectivity'] = event_selectivity               
               
               
                
                roi_data['cross_event_patterns'] = cross_patterns
                
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Error in _compute_cross_event_patterns: {e}")


    def _compute_summary_statistics(self, responsiveness_data: Dict, n_rois: int) -> None:
        """Compute summary statistics across all ROIs."""
        try:
            summary = {
                'total_rois': n_rois,
                'responsive_roi_counts': {},
                'response_type_distributions': {},
                'event_response_rates': {},
                'cross_event_statistics': {}
            }
            
            # Count responsive ROIs by event and filter
            for event_name in responsiveness_data['analysis_parameters']['alignment_events']:
                summary['event_response_rates'][event_name] = {}
                
                for filter_name in responsiveness_data['analysis_parameters']['trial_filters'].keys():
                    responsive_count = 0
                    total_tested = 0
                    
                    for roi_idx in range(n_rois):
                        roi_data = responsiveness_data['roi_responsiveness'][roi_idx]
                        event_data = roi_data.get('event_responses', {}).get(event_name, {})
                        filter_data = event_data.get(filter_name, {})
                        
                        for response_type, result in filter_data.items():
                            if not result.get('insufficient_trials', False):
                                total_tested += 1
                                if result.get('is_responsive', False):
                                    responsive_count += 1
                    
                    if total_tested > 0:
                        summary['event_response_rates'][event_name][filter_name] = {
                            'responsive_count': responsive_count,
                            'total_tested': total_tested,
                            'response_rate': responsive_count / total_tested
                        }
            
            
            # Responsive ROI counts by category
            summary['responsive_roi_counts'] = {
                'highly_responsive': 0,  # Responsive to >50% of conditions
                'moderately_responsive': 0,  # Responsive to 20-50% of conditions
                'selective_responsive': 0,  # Responsive to <20% of conditions
                'non_responsive': 0
            }
            
            # Response type distributions
            summary['response_type_distributions'] = {}
            for response_type in responsiveness_data['analysis_parameters']['responsiveness_types'].keys():
                summary['response_type_distributions'][response_type] = {
                    'total_responses': 0,
                    'roi_count': 0,
                    'mean_effect_size': 0,
                    'strongest_response': 0
                }
            
            # Cross-event statistics
            summary['cross_event_statistics'] = {
                'multi_event_responsive_rois': 0,
                'event_selective_rois': 0,
                'broadly_responsive_rois': 0,
                'temporal_consistency_score': 0,
                'cross_event_correlation': {}
            }
            
            # Populate the new fields
            n_events = len(responsiveness_data['analysis_parameters']['alignment_events'])
            n_filters = len(responsiveness_data['analysis_parameters']['trial_filters'])
            n_response_types = len(responsiveness_data['analysis_parameters']['responsiveness_types'])
            max_possible_responses = n_events * n_filters * n_response_types
            
            effect_sizes_by_type = {resp_type: [] for resp_type in responsiveness_data['analysis_parameters']['responsiveness_types'].keys()}
            roi_response_counts = []
            
            for roi_idx in range(n_rois):
                roi_data = responsiveness_data['roi_responsiveness'][roi_idx]
                roi_response_count = 0
                roi_events = set()
                
                for event_name, event_data in roi_data.get('event_responses', {}).items():
                    for filter_name, filter_data in event_data.items():
                        for response_type, result in filter_data.items():
                            if result.get('is_responsive', False):
                                roi_response_count += 1
                                roi_events.add(event_name)
                                
                                effect_size = result.get('effect_size', 0)
                                effect_sizes_by_type[response_type].append(abs(effect_size))
                
                roi_response_counts.append(roi_response_count)
                
                # Categorize ROI
                response_rate = roi_response_count / max_possible_responses if max_possible_responses > 0 else 0
                
                if response_rate > 0.5:
                    summary['responsive_roi_counts']['highly_responsive'] += 1
                elif response_rate > 0.2:
                    summary['responsive_roi_counts']['moderately_responsive'] += 1
                elif response_rate > 0:
                    summary['responsive_roi_counts']['selective_responsive'] += 1
                else:
                    summary['responsive_roi_counts']['non_responsive'] += 1
                
                # Cross-event statistics
                if len(roi_events) > 1:
                    summary['cross_event_statistics']['multi_event_responsive_rois'] += 1
                elif len(roi_events) == 1:
                    summary['cross_event_statistics']['event_selective_rois'] += 1
                
                if len(roi_events) >= n_events * 0.8:  # Responds to ≥80% of events
                    summary['cross_event_statistics']['broadly_responsive_rois'] += 1
            
            # Response type distributions
            for response_type, effect_sizes in effect_sizes_by_type.items():
                if effect_sizes:
                    summary['response_type_distributions'][response_type] = {
                        'total_responses': len(effect_sizes),
                        'roi_count': len(set(range(len(effect_sizes)))),  # Approximate unique ROIs
                        'mean_effect_size': np.mean(effect_sizes),
                        'strongest_response': np.max(effect_sizes)
                    }
            
            # Temporal consistency
            if roi_response_counts:
                summary['cross_event_statistics']['temporal_consistency_score'] = np.std(roi_response_counts) / (np.mean(roi_response_counts) + 1e-10)            
            
            responsiveness_data['summary_stats'] = summary
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Error in _compute_summary_statistics: {e}")




    def save_responsiveness_data(self, responsiveness_data: Dict[str, Any], output_path: str) -> None:
        """Save ROI responsiveness data to file."""
        try:
            import json
            
            responsiveness_file = os.path.join(output_path, 'roi_responsiveness.json')
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):  # FIX: Handle numpy boolean
                    return bool(obj)
                elif isinstance(obj, (bool, int, float, str)):  # Handle native Python types
                    return obj
                elif obj is None:
                    return None
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, tuple):
                    return [convert_numpy(item) for item in obj]
                else:
                    # For any other type, try to convert to string as fallback
                    try:
                        return str(obj)
                    except:
                        return None
            
            json_data = convert_numpy(responsiveness_data)
            
            with open(responsiveness_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            self.logger.info(f"SID_IMG_PRE: Saved ROI responsiveness data to {responsiveness_file}")
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to save responsiveness data: {e}")



    def load_responsiveness_data(self, output_path: str) -> Optional[Dict[str, Any]]:
        """Load ROI responsiveness data from file."""
        try:
            import json
            
            responsiveness_file = os.path.join(output_path, 'roi_responsiveness.json')
            
            if not os.path.exists(responsiveness_file):
                return None
            
            with open(responsiveness_file, 'r') as f:
                responsiveness_data = json.load(f)
            
            # FIX: Convert string keys back to integers for roi_responsiveness
            if 'roi_responsiveness' in responsiveness_data:
                # Convert string keys to integer keys
                roi_responsiveness_fixed = {}
                for roi_key, roi_data in responsiveness_data['roi_responsiveness'].items():
                    roi_idx = int(roi_key)  # Convert string key back to int
                    roi_responsiveness_fixed[roi_idx] = roi_data
                
                responsiveness_data['roi_responsiveness'] = roi_responsiveness_fixed
            
            self.logger.info(f"SID_IMG_PRE: Loaded ROI responsiveness data from {responsiveness_file}")
            return responsiveness_data
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to load responsiveness data: {e}")
            return None




    def _plot_responsiveness_heatmap(self, responsiveness_data: Dict[str, Any], subject_id: str) -> None:
        """Plot responsiveness heatmap by event and filter."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Extract data for heatmap
            events = responsiveness_data['analysis_parameters']['alignment_events']
            filters = list(responsiveness_data['analysis_parameters']['trial_filters'].keys())
            response_types = list(responsiveness_data['analysis_parameters']['responsiveness_types'].keys())
            
            n_rois = responsiveness_data['summary_stats']['total_rois']
            
            # Create matrix for each response type
            for response_type in response_types:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create matrix: rows = events, columns = filters
                response_matrix = np.zeros((len(events), len(filters)))
                
                for i, event in enumerate(events):
                    for j, filter_name in enumerate(filters):
                        # Count responsive ROIs for this event-filter-response_type combination
                        responsive_count = 0
                        total_count = 0
                        
                        for roi_idx in range(n_rois):
                            roi_data = responsiveness_data['roi_responsiveness'][roi_idx]
                            event_data = roi_data.get('event_responses', {}).get(event, {})
                            filter_data = event_data.get(filter_name, {})
                            result = filter_data.get(response_type, {})
                            
                            if not result.get('insufficient_trials', False):
                                total_count += 1
                                if result.get('is_responsive', False):
                                    responsive_count += 1
                        
                        # Calculate response rate
                        if total_count > 0:
                            response_matrix[i, j] = responsive_count / total_count
                        else:
                            response_matrix[i, j] = 0
                
                # Create heatmap
                sns.heatmap(response_matrix, 
                        xticklabels=filters, 
                        yticklabels=events,
                        annot=True, 
                        fmt='.2f',
                        cmap='viridis',
                        vmin=0, vmax=1,
                        ax=ax)
                
                ax.set_title(f'ROI Responsiveness Rate - {response_type.title()}\n{subject_id}')
                ax.set_xlabel('Trial Filter')
                ax.set_ylabel('Alignment Event')
                
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create responsiveness heatmap: {e}")


    def _plot_response_type_distributions(self, responsiveness_data: Dict[str, Any], subject_id: str) -> None:
        """Plot response type distributions across ROIs."""
        try:
            import matplotlib.pyplot as plt
            
            response_types = list(responsiveness_data['analysis_parameters']['responsiveness_types'].keys())
            events = responsiveness_data['analysis_parameters']['alignment_events']
            filters = list(responsiveness_data['analysis_parameters']['trial_filters'].keys())
            n_rois = responsiveness_data['summary_stats']['total_rois']
            
            # Count responsive ROIs by response type
            for event in events:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'Response Type Distributions - {event} - {subject_id}', fontsize=14)
                axes = axes.flatten()
                
                for filter_idx, filter_name in enumerate(filters[:4]):  # Limit to 4 filters for display
                    if filter_idx >= len(axes):
                        break
                        
                    response_counts = {resp_type: 0 for resp_type in response_types}
                    
                    for roi_idx in range(n_rois):
                        roi_data = responsiveness_data['roi_responsiveness'][roi_idx]
                        event_data = roi_data.get('event_responses', {}).get(event, {})
                        filter_data = event_data.get(filter_name, {})
                        
                        for response_type in response_types:
                            result = filter_data.get(response_type, {})
                            if result.get('is_responsive', False):
                                response_counts[response_type] += 1
                    
                    # Create bar plot
                    types = list(response_counts.keys())
                    counts = list(response_counts.values())
                    
                    bars = axes[filter_idx].bar(types, counts, alpha=0.7)
                    axes[filter_idx].set_title(f'{filter_name.title()} Trials (n={sum(counts)} responsive)')
                    axes[filter_idx].set_ylabel('Number of ROIs')
                    axes[filter_idx].tick_params(axis='x', rotation=45)
                    axes[filter_idx].grid(True, alpha=0.3)
                    
                    # Add count labels on bars
                    for bar, count in zip(bars, counts):
                        if count > 0:
                            axes[filter_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                                str(count), ha='center', va='bottom', fontsize=9)
                
                # Hide unused subplots
                for i in range(len(filters), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create response type distributions: {e}")





    def _plot_roi_classification(self, responsiveness_data: Dict[str, Any], subject_id: str) -> None:
        """Plot ROI classification dendrogram based on response patterns."""
        try:
            import matplotlib.pyplot as plt
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import pdist
            
            events = responsiveness_data['analysis_parameters']['alignment_events']
            filters = list(responsiveness_data['analysis_parameters']['trial_filters'].keys())
            response_types = list(responsiveness_data['analysis_parameters']['responsiveness_types'].keys())
            n_rois = responsiveness_data['summary_stats']['total_rois']
            
            # Create feature matrix for each ROI
            n_features = len(events) * len(filters) * len(response_types)
            feature_matrix = np.zeros((n_rois, n_features))
            
            feature_idx = 0
            for event in events:
                for filter_name in filters:
                    for response_type in response_types:
                        for roi_idx in range(n_rois):
                            roi_data = responsiveness_data['roi_responsiveness'][roi_idx]
                            event_data = roi_data.get('event_responses', {}).get(event, {})
                            filter_data = event_data.get(filter_name, {})
                            result = filter_data.get(response_type, {})
                            
                            if result.get('is_responsive', False):
                                feature_matrix[roi_idx, feature_idx] = result.get('effect_size', 0)
                            else:
                                feature_matrix[roi_idx, feature_idx] = 0
                        feature_idx += 1
            
            # Filter to responsive ROIs only
            responsive_rois = np.any(feature_matrix != 0, axis=1)
            responsive_roi_indices = np.where(responsive_rois)[0]
            
            if len(responsive_roi_indices) < 2:
                self.logger.warning("SID_IMG_PRE: Too few responsive ROIs for clustering analysis")
                return
            
            # FIX: Limit to top N most responsive ROIs for cleaner visualization
            max_rois_for_clustering = 50  # Adjustable parameter
            
            if len(responsive_roi_indices) > max_rois_for_clustering:
                # Calculate total responsiveness per ROI (sum of absolute effect sizes)
                roi_responsiveness_scores = np.sum(np.abs(feature_matrix[responsive_roi_indices, :]), axis=1)
                
                # Get indices of top N most responsive ROIs
                top_roi_local_indices = np.argsort(roi_responsiveness_scores)[-max_rois_for_clustering:]
                top_roi_global_indices = responsive_roi_indices[top_roi_local_indices]
                
                self.logger.info(f"SID_IMG_PRE: Clustering top {max_rois_for_clustering} most responsive ROIs out of {len(responsive_roi_indices)} total")
            else:
                top_roi_global_indices = responsive_roi_indices
                self.logger.info(f"SID_IMG_PRE: Clustering all {len(responsive_roi_indices)} responsive ROIs")
            
            # Use only top responsive ROIs
            top_responsive_features = feature_matrix[top_roi_global_indices, :]
            
            # Remove features with no variance
            feature_variance = np.var(top_responsive_features, axis=0)
            informative_features = feature_variance > 1e-10
            
            if np.any(informative_features):
                filtered_features = top_responsive_features[:, informative_features]
                
                # Compute hierarchical clustering
                distances = pdist(filtered_features, metric='euclidean')
                linkage_matrix = linkage(distances, method='ward')
                
                # Create cleaner dendrogram
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Plot dendrogram with better formatting
                dendrogram(linkage_matrix, 
                        labels=[f'ROI{i}' for i in top_roi_global_indices],  # Shorter labels
                        ax=ax1,
                        orientation='top',
                        leaf_rotation=90,
                        leaf_font_size=8)  # Smaller font
                
                ax1.set_title(f'ROI Classification Dendrogram\n{subject_id} (top {len(top_roi_global_indices)} responsive ROIs)', fontsize=12)
                ax1.set_ylabel('Distance')
                
                # Plot feature heatmap for selected ROIs
                im = ax2.imshow(filtered_features, aspect='auto', cmap='RdBu_r', 
                            interpolation='nearest')
                ax2.set_xlabel('Feature Index')
                ax2.set_ylabel('ROI Index')
                ax2.set_title('Response Feature Matrix')
                
                # Better y-axis labels
                n_labels = min(10, len(top_roi_global_indices))  # Max 10 labels
                step = len(top_roi_global_indices) // n_labels
                tick_positions = np.arange(0, len(top_roi_global_indices), step)[:n_labels]
                tick_labels = [f'ROI{top_roi_global_indices[i]}' for i in tick_positions]
                
                ax2.set_yticks(tick_positions)
                ax2.set_yticklabels(tick_labels, fontsize=8)
                
                plt.colorbar(im, ax=ax2, label='Effect Size')
                plt.tight_layout()
                plt.show()
                
            else:
                self.logger.warning("SID_IMG_PRE: No informative features for clustering")
                
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create ROI classification: {e}")




    def _plot_roi_classification_summary(self, responsiveness_data: Dict[str, Any], subject_id: str) -> None:
        """Plot ROI classification summary without cluttered dendrogram."""
        try:
            import matplotlib.pyplot as plt
            
            events = responsiveness_data['analysis_parameters']['alignment_events']
            filters = list(responsiveness_data['analysis_parameters']['trial_filters'].keys())
            response_types = list(responsiveness_data['analysis_parameters']['responsiveness_types'].keys())
            n_rois = responsiveness_data['summary_stats']['total_rois']
            
            # Count responsive ROIs by category
            roi_categories = {
                'highly_responsive': [],  # Responds to multiple events/conditions
                'selective': [],          # Responds to specific events only
                'broad': [],             # Responds to many conditions
                'minimal': []            # Few responses
            }
            
            for roi_idx in range(n_rois):
                roi_data = responsiveness_data['roi_responsiveness'][roi_idx]
                
                # Count total responses
                total_responses = 0
                response_events = set()
                
                for event in events:
                    event_data = roi_data.get('event_responses', {}).get(event, {})
                    for filter_name in filters:
                        filter_data = event_data.get(filter_name, {})
                        for response_type in response_types:
                            result = filter_data.get(response_type, {})
                            if result.get('is_responsive', False):
                                total_responses += 1
                                response_events.add(event)
                
                # Categorize ROI
                if total_responses == 0:
                    continue  # Skip non-responsive ROIs
                elif total_responses >= 8:  # Highly responsive
                    roi_categories['highly_responsive'].append(roi_idx)
                elif len(response_events) == 1:  # Event-selective
                    roi_categories['selective'].append(roi_idx)
                elif len(response_events) >= len(events) * 0.8:  # Broad
                    roi_categories['broad'].append(roi_idx)
                else:
                    roi_categories['minimal'].append(roi_idx)
            
            # Create summary plots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'ROI Response Classification Summary - {subject_id}', fontsize=14)
            
            # Plot 1: Category counts
            categories = list(roi_categories.keys())
            counts = [len(roi_categories[cat]) for cat in categories]
            
            colors = ['red', 'orange', 'green', 'blue']
            bars = axes[0, 0].bar(categories, counts, color=colors, alpha=0.7)
            axes[0, 0].set_ylabel('Number of ROIs')
            axes[0, 0].set_title('ROI Classification by Response Pattern')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Add count labels
            for bar, count in zip(bars, counts):
                if count > 0:
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                str(count), ha='center', va='bottom')
            
            # Plot 2: Response distribution by event
            event_responsive_counts = {}
            for event in events:
                count = 0
                for roi_idx in range(n_rois):
                    roi_data = responsiveness_data['roi_responsiveness'][roi_idx]
                    event_data = roi_data.get('event_responses', {}).get(event, {})
                    
                    has_response = False
                    for filter_name in filters:
                        filter_data = event_data.get(filter_name, {})
                        for response_type in response_types:
                            result = filter_data.get(response_type, {})
                            if result.get('is_responsive', False):
                                has_response = True
                                break
                        if has_response:
                            break
                    
                    if has_response:
                        count += 1
                
                event_responsive_counts[event] = count
            
            bars2 = axes[0, 1].bar(event_responsive_counts.keys(), event_responsive_counts.values(), alpha=0.7)
            axes[0, 1].set_ylabel('Number of Responsive ROIs')
            axes[0, 1].set_title('Responsive ROIs by Event')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Response type distribution
            response_type_counts = {resp_type: 0 for resp_type in response_types}
            
            for roi_idx in range(n_rois):
                roi_data = responsiveness_data['roi_responsiveness'][roi_idx]
                for event in events:
                    event_data = roi_data.get('event_responses', {}).get(event, {})
                    for filter_name in filters:
                        filter_data = event_data.get(filter_name, {})
                        for response_type in response_types:
                            result = filter_data.get(response_type, {})
                            if result.get('is_responsive', False):
                                response_type_counts[response_type] += 1
            
            bars3 = axes[1, 0].bar(response_type_counts.keys(), response_type_counts.values(), alpha=0.7)
            axes[1, 0].set_ylabel('Number of Responses')
            axes[1, 0].set_title('Response Type Distribution')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Example ROIs from each category
            axes[1, 1].axis('off')
            y_pos = 0.9
            
            for category, color in zip(categories, colors):
                roi_list = roi_categories[category]
                if roi_list:
                    example_rois = roi_list[:5]  # Show first 5 as examples
                    roi_text = f"{category.replace('_', ' ').title()}: {len(roi_list)} ROIs\n"
                    roi_text += f"Examples: {', '.join([f'ROI{i}' for i in example_rois])}"
                    if len(roi_list) > 5:
                        roi_text += f"... (+{len(roi_list)-5} more)"
                else:
                    roi_text = f"{category.replace('_', ' ').title()}: 0 ROIs"
                
                axes[1, 1].text(0.05, y_pos, roi_text, transform=axes[1, 1].transAxes,
                            fontsize=10, verticalalignment='top', color=color, weight='bold')
                y_pos -= 0.2
            
            axes[1, 1].set_title('ROI Examples by Category')
            
            plt.tight_layout()
            plt.show()
            
            # Log summary
            total_responsive = sum(counts)
            self.logger.info(f"SID_IMG_PRE: ROI Classification Summary:")
            for category, count in zip(categories, counts):
                percentage = (count / total_responsive * 100) if total_responsive > 0 else 0
                self.logger.info(f"  {category.replace('_', ' ').title()}: {count} ROIs ({percentage:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create ROI classification summary: {e}")



    def _plot_event_selectivity(self, responsiveness_data: Dict[str, Any], subject_id: str) -> None:
        """Plot event selectivity patterns across ROIs."""
        try:
            import matplotlib.pyplot as plt
            
            events = responsiveness_data['analysis_parameters']['alignment_events']
            filters = list(responsiveness_data['analysis_parameters']['trial_filters'].keys())
            response_types = list(responsiveness_data['analysis_parameters']['responsiveness_types'].keys())
            n_rois = responsiveness_data['summary_stats']['total_rois']
            
            # Analyze selectivity patterns
            roi_selectivity = []
            
            for roi_idx in range(n_rois):
                roi_data = responsiveness_data['roi_responsiveness'][roi_idx]
                
                # Count responses across events and filters
                event_responses = {event: 0 for event in events}
                filter_responses = {filter_name: 0 for filter_name in filters}
                total_responses = 0
                
                for event in events:
                    event_data = roi_data.get('event_responses', {}).get(event, {})
                    
                    for filter_name in filters:
                        filter_data = event_data.get(filter_name, {})
                        
                        # Count significant responses of any type
                        responsive_types = 0
                        for response_type in response_types:
                            result = filter_data.get(response_type, {})
                            if result.get('is_responsive', False):
                                responsive_types += 1
                        
                        if responsive_types > 0:
                            event_responses[event] += 1
                            filter_responses[filter_name] += 1
                            total_responses += 1
                
                # Calculate selectivity indices
                if total_responses > 0:
                    # Event selectivity: how selective is this ROI across events?
                    event_entropy = 0
                    for event in events:
                        if event_responses[event] > 0:
                            p = event_responses[event] / total_responses
                            event_entropy -= p * np.log2(p)
                    
                    # Filter selectivity: how selective across trial types?
                    filter_entropy = 0
                    for filter_name in filters:
                        if filter_responses[filter_name] > 0:
                            p = filter_responses[filter_name] / total_responses
                            filter_entropy -= p * np.log2(p)
                    
                    roi_selectivity.append({
                        'roi_idx': roi_idx,
                        'total_responses': total_responses,
                        'event_entropy': event_entropy,
                        'filter_entropy': filter_entropy,
                        'event_responses': event_responses,
                        'filter_responses': filter_responses
                    })
            
            if not roi_selectivity:
                self.logger.warning("SID_IMG_PRE: No responsive ROIs found for selectivity analysis")
                return
            
            # Create selectivity plots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Event Selectivity Analysis - {subject_id}', fontsize=14)
            
            # Plot 1: Event entropy vs total responses
            total_resp = [roi['total_responses'] for roi in roi_selectivity]
            event_ent = [roi['event_entropy'] for roi in roi_selectivity]
            roi_indices = [roi['roi_idx'] for roi in roi_selectivity]
            
            scatter = axes[0, 0].scatter(total_resp, event_ent, c=roi_indices, 
                                    cmap='viridis', alpha=0.7, s=50)
            axes[0, 0].set_xlabel('Total Responses')
            axes[0, 0].set_ylabel('Event Entropy (bits)')
            axes[0, 0].set_title('Event Selectivity vs Responsiveness')
            axes[0, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 0], label='ROI Index')
            
            # Plot 2: Filter entropy vs total responses
            filter_ent = [roi['filter_entropy'] for roi in roi_selectivity]
            scatter2 = axes[0, 1].scatter(total_resp, filter_ent, c=roi_indices, 
                                        cmap='viridis', alpha=0.7, s=50)
            axes[0, 1].set_xlabel('Total Responses')
            axes[0, 1].set_ylabel('Filter Entropy (bits)')
            axes[0, 1].set_title('Trial Type Selectivity vs Responsiveness')
            axes[0, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=axes[0, 1], label='ROI Index')
            
            # Plot 3: Event response distribution
            event_totals = {event: sum(roi['event_responses'][event] for roi in roi_selectivity) 
                        for event in events}
            
            bars1 = axes[1, 0].bar(event_totals.keys(), event_totals.values(), alpha=0.7)
            axes[1, 0].set_ylabel('Number of Responsive ROIs')
            axes[1, 0].set_title('ROI Responses by Event')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars1, event_totals.values()):
                if count > 0:
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                str(count), ha='center', va='bottom')
            
            # Plot 4: Filter response distribution
            filter_totals = {filter_name: sum(roi['filter_responses'][filter_name] for roi in roi_selectivity) 
                            for filter_name in filters}
            
            bars2 = axes[1, 1].bar(filter_totals.keys(), filter_totals.values(), alpha=0.7)
            axes[1, 1].set_ylabel('Number of Responsive ROIs')
            axes[1, 1].set_title('ROI Responses by Trial Type')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars2, filter_totals.values()):
                if count > 0:
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
            # Log selectivity summary
            high_selectivity_rois = [roi for roi in roi_selectivity if roi['event_entropy'] < 1.0 and roi['total_responses'] > 0]
            broad_response_rois = [roi for roi in roi_selectivity if roi['total_responses'] >= len(events) * len(filters) * 0.5]
            
            self.logger.info(f"SID_IMG_PRE: Selectivity analysis - {len(roi_selectivity)} responsive ROIs")
            self.logger.info(f"SID_IMG_PRE: High selectivity ROIs (entropy < 1.0): {len(high_selectivity_rois)}")
            self.logger.info(f"SID_IMG_PRE: Broadly responsive ROIs: {len(broad_response_rois)}")
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create event selectivity plots: {e}")







    def plot_responsiveness_summary(self, responsiveness_data: Dict[str, Any], 
                                subject_id: str, output_path: str) -> None:
        """Create comprehensive responsiveness summary plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Plot 1: Responsiveness heatmap by event and filter
            self._plot_responsiveness_heatmap(responsiveness_data, subject_id)
            
            # Plot 2: Response type distributions
            self._plot_response_type_distributions(responsiveness_data, subject_id)
            
            # Plot 3: ROI classification dendrogram
            self._plot_roi_classification(responsiveness_data, subject_id)
            
            # Plot 4: ROI classification summary (cleaner alternative)
            self._plot_roi_classification_summary(responsiveness_data, subject_id)
            
            # Plot 5: Event selectivity patterns
            self._plot_event_selectivity(responsiveness_data, subject_id)
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create responsiveness summary plots: {e}")





    def plot_all_rois_heatmap_by_responsiveness(self, trial_data: Dict[str, Any], 
                                            responsiveness_data: Dict[str, Any], 
                                            subject_id: str, 
                                            event_filter: str = 'start_flash_1',
                                            trial_filter: str = 'left',
                                            response_type_filter: str = 'activation',
                                            sort_method: str = 'effect_size',
                                            trial_type: str = 'all') -> None:
        """
        Plot heatmap showing all ROI responses sorted by responsiveness characteristics.
        
        Args:
            trial_data: Dictionary from extract_trial_segments_simple
            responsiveness_data: Dictionary from analyze_roi_responsiveness
            subject_id: Subject ID for plot title
            event_filter: Which event responsiveness to use for sorting ('start_flash_1', 'choice_start', etc.)
            trial_filter: Which trial condition to use ('left', 'right', etc.)
            response_type_filter: Which response type to use ('activation', 'suppression', etc.)
            sort_method: 'effect_size', 'p_value', 'response_magnitude', 'response_timing', 'reliability'
            trial_type: 'all', 'left', 'right', 'rewarded', 'unrewarded'
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if 'error' in trial_data:
                self.logger.error(f"SID_IMG_PRE: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            if 'error' in responsiveness_data:
                self.logger.error(f"SID_IMG_PRE: Cannot plot - responsiveness analysis failed: {responsiveness_data['error']}")
                return
            
            # Handle both old array format and new DataFrame format
            if 'df_trials_with_segments' in trial_data:
                # New DataFrame format
                df_with_segments = trial_data['df_trials_with_segments']
                valid_mask = trial_data['valid_trials_mask']
                df_valid = df_with_segments[valid_mask]
                
                if len(df_valid) == 0:
                    self.logger.error("SID_IMG_PRE: No valid trials with segments")
                    return
                
                # Get common time vector
                dff_time_vector = df_valid.iloc[0]['dff_time_vector']
                
                # Filter trials based on trial_type
                if trial_type == 'all':
                    selected_trials = df_valid
                    title_suffix = 'All Trials'
                elif trial_type == 'left':
                    selected_trials = df_valid[df_valid['trial_side'] == 'left']
                    title_suffix = 'Left Trials'
                elif trial_type == 'right':
                    selected_trials = df_valid[df_valid['trial_side'] == 'right']
                    title_suffix = 'Right Trials'
                elif trial_type == 'rewarded':
                    selected_trials = df_valid[df_valid['rewarded'] == 1]
                    title_suffix = 'Rewarded Trials'
                elif trial_type == 'unrewarded':
                    selected_trials = df_valid[df_valid['rewarded'] == 0]
                    title_suffix = 'Unrewarded Trials'
                else:
                    selected_trials = df_valid
                    title_suffix = f'{trial_type.title()} Trials'
                
                if len(selected_trials) == 0:
                    self.logger.warning(f"SID_IMG_PRE: No trials found for trial_type '{trial_type}'")
                    return
                
                # Calculate mean response across selected trials
                all_segments = np.array([row['dff_segment'] for _, row in selected_trials.iterrows()])
                mean_traces = np.mean(all_segments, axis=0)  # (n_rois, n_timepoints)
                
            else:
                self.logger.error("SID_IMG_PRE: This function requires DataFrame format trial data")
                return
            
            n_rois = mean_traces.shape[0]
            
            # Extract responsiveness metrics for sorting
            roi_metrics = []
            responsive_rois = []
            non_responsive_rois = []
            
            for roi_idx in range(n_rois):
                roi_data = responsiveness_data['roi_responsiveness'].get(roi_idx, {})
                event_data = roi_data.get('event_responses', {}).get(event_filter, {})
                filter_data = event_data.get(trial_filter, {})
                result = filter_data.get(response_type_filter, {})
                
                if result.get('is_responsive', False):
                    responsive_rois.append(roi_idx)
                    
                    # Extract sorting metric
                    if sort_method == 'effect_size':
                        metric = abs(result.get('effect_size', 0))
                    elif sort_method == 'p_value':
                        metric = -np.log10(result.get('p_value', 1) + 1e-10)  # Negative log for better sorting
                    elif sort_method == 'response_magnitude':
                        metric = result.get('response_magnitude', 0)
                    elif sort_method == 'response_timing':
                        metric = result.get('response_timing', 0)
                    elif sort_method == 'reliability':
                        metric = result.get('trial_reliability', 0)
                        if np.isnan(metric):
                            metric = 0
                    else:
                        metric = abs(result.get('effect_size', 0))
                    
                    roi_metrics.append((roi_idx, metric, result.get('effect_size', 0)))
                else:
                    non_responsive_rois.append(roi_idx)
                    roi_metrics.append((roi_idx, 0, 0))  # Non-responsive gets 0 metric
            
            # Sort ROIs by responsiveness metrics
            if sort_method == 'response_timing':
                # For timing, sort by value (earliest first)
                sorted_metrics = sorted(roi_metrics, key=lambda x: x[1])
            else:
                # For other metrics, sort by magnitude (highest first)
                sorted_metrics = sorted(roi_metrics, key=lambda x: x[1], reverse=True)
            
            roi_order = [item[0] for item in sorted_metrics]
            roi_effect_sizes = [item[2] for item in sorted_metrics]
            
            # Reorder the traces
            mean_traces_sorted = mean_traces[roi_order, :]
            
            # Create ROI labels with responsiveness info
            roi_labels = []
            for i, (roi_idx, metric, effect_size) in enumerate(sorted_metrics):
                is_responsive = roi_idx in responsive_rois
                if is_responsive:
                    direction = '+' if effect_size > 0 else '-'
                    roi_labels.append(f'ROI{roi_idx}{direction}')
                else:
                    roi_labels.append(f'ROI{roi_idx}')
            
            # Create figure with multiple views
            fig, axes = plt.subplots(1, 3, figsize=(20, 8))
            alignment = trial_data['alignment_point']
            fig.suptitle(f'ROI Responses (Sorted by {sort_method.title()}) - {title_suffix}\n'
                        f'{subject_id} - Alignment: {alignment} - Filter: {event_filter}|{trial_filter}|{response_type_filter}', 
                        fontsize=14)
            
            # Calculate actual data range for consistent colormaps
            vmin_actual = np.percentile(mean_traces_sorted, 5)
            vmax_actual = np.percentile(mean_traces_sorted, 95)
            
            # Plot 1: Full heatmap
            im1 = axes[0].imshow(mean_traces_sorted, aspect='auto', cmap='viridis',
                                extent=[dff_time_vector[0], dff_time_vector[-1], 0, mean_traces_sorted.shape[0]],
                                vmin=vmin_actual, vmax=vmax_actual, interpolation='nearest')
            axes[0].axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.9)
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('ROI # (sorted by responsiveness)')
            axes[0].set_title(f'All ROIs Heatmap\n(n={len(responsive_rois)} responsive)')
            
            # Add horizontal line separating responsive from non-responsive
            if responsive_rois and non_responsive_rois:
                separation_line = len(responsive_rois)
                axes[0].axhline(y=separation_line, color='red', linestyle='-', linewidth=2, alpha=0.8)
                axes[0].text(dff_time_vector[-1]*0.98, separation_line + 1, 'Non-responsive', 
                            ha='right', va='bottom', color='red', fontweight='bold')
            
            # Update y-axis to show ROI labels (sparse to avoid clutter)
            y_tick_spacing = max(1, len(roi_labels)//15)  # Show ~15 labels max
            y_ticks = np.arange(0, len(roi_labels), y_tick_spacing)
            axes[0].set_yticks(y_ticks)
            axes[0].set_yticklabels([roi_labels[i] for i in y_ticks], fontsize=8)
            
            plt.colorbar(im1, ax=axes[0], label='Mean DFF')
            
            # Plot 2: Mean trace comparison (responsive vs non-responsive)
            if responsive_rois:
                responsive_traces = mean_traces_sorted[:len(responsive_rois), :]
                responsive_mean = np.mean(responsive_traces, axis=0)
                responsive_sem = np.std(responsive_traces, axis=0) / np.sqrt(responsive_traces.shape[0])
                
                axes[1].plot(dff_time_vector, responsive_mean, 'red', linewidth=3, 
                            label=f'Responsive (n={len(responsive_rois)})')
                axes[1].fill_between(dff_time_vector, responsive_mean - responsive_sem, 
                                    responsive_mean + responsive_sem, alpha=0.3, color='red')
            
            if non_responsive_rois:
                non_responsive_traces = mean_traces_sorted[len(responsive_rois):, :]
                non_responsive_mean = np.mean(non_responsive_traces, axis=0)
                non_responsive_sem = np.std(non_responsive_traces, axis=0) / np.sqrt(non_responsive_traces.shape[0])
                
                axes[1].plot(dff_time_vector, non_responsive_mean, 'gray', linewidth=3, 
                            label=f'Non-responsive (n={len(non_responsive_rois)})')
                axes[1].fill_between(dff_time_vector, non_responsive_mean - non_responsive_sem, 
                                    non_responsive_mean + non_responsive_sem, alpha=0.3, color='gray')
            
            axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Alignment Point')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Mean DFF')
            axes[1].set_title('Responsive vs Non-responsive')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Responsiveness metrics scatter
            if responsive_rois:
                # Plot sorting metric for responsive ROIs
                responsive_indices = list(range(len(responsive_rois)))
                responsive_metrics = [sorted_metrics[i][1] for i in responsive_indices]
                responsive_effect_sizes = [sorted_metrics[i][2] for i in responsive_indices]
                
                # Color by effect size direction
                colors = ['red' if es > 0 else 'blue' for es in responsive_effect_sizes]
                
                scatter = axes[2].scatter(responsive_indices, responsive_metrics, 
                                        c=colors, s=50, alpha=0.7)
                
                axes[2].set_xlabel('ROI Rank (by responsiveness)')
                axes[2].set_ylabel(f'{sort_method.replace("_", " ").title()}')
                axes[2].set_title(f'Responsiveness Ranking\n(Red=activation, Blue=suppression)')
                axes[2].grid(True, alpha=0.3)
                
                # Add top 5 ROI labels
                for i in range(min(5, len(responsive_rois))):
                    roi_idx = sorted_metrics[i][0]
                    metric_val = sorted_metrics[i][1]
                    axes[2].annotate(f'ROI{roi_idx}', 
                                (i, metric_val), 
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, ha='left')
            else:
                axes[2].text(0.5, 0.5, 'No responsive ROIs found', ha='center', va='center', 
                            transform=axes[2].transAxes, fontsize=12)
                axes[2].set_title('No Responsive ROIs')
            
            plt.tight_layout()
            plt.show()
            
            # Log summary
            n_trials = len(selected_trials)
            self.logger.info(f"SID_IMG_PRE: Responsiveness-sorted heatmap - {n_trials} trials, {n_rois} ROIs")
            self.logger.info(f"SID_IMG_PRE: {len(responsive_rois)} responsive, {len(non_responsive_rois)} non-responsive")
            self.logger.info(f"SID_IMG_PRE: Sort criteria: {event_filter}|{trial_filter}|{response_type_filter}|{sort_method}")
            
            if responsive_rois:
                top_roi_idx = sorted_metrics[0][0]
                top_metric = sorted_metrics[0][1]
                self.logger.info(f"SID_IMG_PRE: Most responsive ROI: {top_roi_idx} ({sort_method}: {top_metric:.3f})")
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to create responsiveness-sorted ROI heatmap: {e}")


    def plot_roi_functional_connectivity(self, responsiveness_data: Dict[str, Any], 
                                    imaging_data: Dict[str, Any], subject_id: str) -> None:
        """Plot functional connectivity between responsive ROIs."""
        try:
            import matplotlib.pyplot as plt
            from scipy.spatial.distance import pdist, squareform
            from scipy.cluster.hierarchy import linkage, dendrogram
            
            # Get highly responsive ROIs
            responsive_rois = []
            for roi_idx, roi_data in responsiveness_data['roi_responsiveness'].items():
                total_responses = len(roi_data.get('cross_event_patterns', {}).get('responsive_events', []))
                if total_responses >= 3:  # Threshold for "highly responsive"
                    responsive_rois.append(roi_idx)
            
            if len(responsive_rois) < 10:
                self.logger.warning("Too few responsive ROIs for connectivity analysis")
                return
            
            # Calculate correlation matrix between responsive ROIs
            roi_traces = imaging_data['dff_traces'][responsive_rois, :]
            correlation_matrix = np.corrcoef(roi_traces)
            
            # Create connectivity plot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'ROI Functional Connectivity - {subject_id}', fontsize=14)
            
            # Plot 1: Correlation matrix heatmap
            im = axes[0, 0].imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0, 0].set_title(f'ROI Correlation Matrix (n={len(responsive_rois)})')
            axes[0, 0].set_xlabel('ROI Index')
            axes[0, 0].set_ylabel('ROI Index')
            plt.colorbar(im, ax=axes[0, 0])
            
            # Plot 2: Hierarchical clustering of ROIs
            distances = pdist(roi_traces, metric='correlation')
            linkage_matrix = linkage(distances, method='ward')
            dendrogram(linkage_matrix, ax=axes[0, 1], 
                    labels=[f'ROI{i}' for i in responsive_rois], 
                    leaf_rotation=90, leaf_font_size=8)
            axes[0, 1].set_title('ROI Clustering (by activity similarity)')
            
            # Plot 3: Network graph of high correlations
            import networkx as nx
            
            # Create network from high correlations
            G = nx.Graph()
            correlation_threshold = 0.5
            
            for i, roi_i in enumerate(responsive_rois):
                G.add_node(roi_i)
                for j, roi_j in enumerate(responsive_rois):
                    if i != j and abs(correlation_matrix[i, j]) > correlation_threshold:
                        G.add_edge(roi_i, roi_j, weight=correlation_matrix[i, j])
            
            # Draw network
            pos = nx.spring_layout(G)
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            
            nx.draw(G, pos, ax=axes[1, 0], 
                node_color=[len(responsiveness_data['roi_responsiveness'][roi]['cross_event_patterns']['responsive_events']) 
                            for roi in G.nodes()],
                node_size=200, cmap='viridis', 
                with_labels=True, font_size=8,
                edge_color=weights, edge_cmap=plt.cm.RdBu_r)
            axes[1, 0].set_title(f'Functional Network (|r| > {correlation_threshold})')
            
            # Plot 4: Correlation distribution
            upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            axes[1, 1].hist(upper_triangle, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(correlation_threshold, color='red', linestyle='--', 
                            label=f'Threshold = {correlation_threshold}')
            axes[1, 1].axvline(-correlation_threshold, color='red', linestyle='--')
            axes[1, 1].set_xlabel('Correlation Coefficient')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('ROI-ROI Correlation Distribution')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to create connectivity plot: {e}")


    def plot_roi_response_profiles(self, responsiveness_data: Dict[str, Any], subject_id: str) -> None:
        """Create detailed response profiles for different ROI types."""
        try:
            import matplotlib.pyplot as plt
            
            # Extract ROI types and properties
            roi_profiles = {'excitatory': [], 'inhibitory': [], 'unlabeled': []}
            
            for roi_idx, roi_data in responsiveness_data['roi_responsiveness'].items():
                roi_type = roi_data['basic_properties']['roi_type']
                
                # Calculate response profile metrics
                profile = {
                    'roi_idx': roi_idx,
                    'dynamic_range': roi_data['basic_properties']['dynamic_range'],
                    'mean_activity': roi_data['basic_properties']['mean_activity'],
                    'n_responsive_events': len(roi_data['cross_event_patterns']['responsive_events']),
                    'preferred_response_types': list(roi_data['cross_event_patterns']['preferred_response_types'].keys()),
                    'event_selectivity': roi_data['cross_event_patterns']['event_selectivity'].get('selectivity_index', 0),
                    'temporal_consistency': roi_data['cross_event_patterns']['temporal_patterns'].get('timing_variability', np.nan),
                    'reliability': roi_data['reliability_metrics']['mean_trial_reliability']
                }
                
                if roi_type in roi_profiles:
                    roi_profiles[roi_type].append(profile)
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'ROI Response Profiles by Cell Type - {subject_id}', fontsize=16)
            
            colors = {'excitatory': 'red', 'inhibitory': 'blue', 'unlabeled': 'gray'}
            
            # Plot 1: Dynamic range comparison
            for cell_type, profiles in roi_profiles.items():
                if profiles:
                    values = [p['dynamic_range'] for p in profiles]
                    axes[0, 0].hist(values, alpha=0.6, label=f'{cell_type} (n={len(profiles)})', 
                                color=colors[cell_type], bins=20)
            axes[0, 0].set_xlabel('Dynamic Range')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Activity Dynamic Range by Cell Type')
            axes[0, 0].legend()
            
            # Plot 2: Responsiveness comparison
            responsiveness_data_plot = []
            labels_plot = []
            colors_plot = []
            
            for cell_type, profiles in roi_profiles.items():
                if profiles:
                    values = [p['n_responsive_events'] for p in profiles]
                    responsiveness_data_plot.append(values)
                    labels_plot.append(f'{cell_type}\n(n={len(profiles)})')
                    colors_plot.append(colors[cell_type])
            
            bp = axes[0, 1].boxplot(responsiveness_data_plot, labels=labels_plot, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_plot):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            axes[0, 1].set_ylabel('Number of Responsive Events')
            axes[0, 1].set_title('Event Responsiveness by Cell Type')
            
            # Plot 3: Event selectivity comparison
            selectivity_data_plot = []
            for cell_type, profiles in roi_profiles.items():
                if profiles:
                    values = [p['event_selectivity'] for p in profiles if not np.isnan(p['event_selectivity'])]
                    selectivity_data_plot.append(values)
            
            bp2 = axes[0, 2].boxplot(selectivity_data_plot, labels=labels_plot, patch_artist=True)
            for patch, color in zip(bp2['boxes'], colors_plot):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            axes[0, 2].set_ylabel('Event Selectivity Index')
            axes[0, 2].set_title('Event Selectivity by Cell Type')
            
            # Plot 4: Response type preferences
            response_type_counts = {cell_type: {} for cell_type in roi_profiles.keys()}
            all_response_types = set()
            
            for cell_type, profiles in roi_profiles.items():
                for profile in profiles:
                    for resp_type in profile['preferred_response_types']:
                        all_response_types.add(resp_type)
                        if resp_type not in response_type_counts[cell_type]:
                            response_type_counts[cell_type][resp_type] = 0
                        response_type_counts[cell_type][resp_type] += 1
            
            # Normalize by cell type count
            for cell_type in response_type_counts:
                total_cells = len(roi_profiles[cell_type])
                if total_cells > 0:
                    for resp_type in response_type_counts[cell_type]:
                        response_type_counts[cell_type][resp_type] /= total_cells
            
            # Create stacked bar plot
            bottom_exc = np.zeros(len(all_response_types))
            bottom_inh = np.zeros(len(all_response_types))
            bottom_unl = np.zeros(len(all_response_types))
            
            x_pos = np.arange(len(all_response_types))
            response_types_list = list(all_response_types)
            
            exc_values = [response_type_counts['excitatory'].get(rt, 0) for rt in response_types_list]
            inh_values = [response_type_counts['inhibitory'].get(rt, 0) for rt in response_types_list]
            unl_values = [response_type_counts['unlabeled'].get(rt, 0) for rt in response_types_list]
            
            axes[1, 0].bar(x_pos, exc_values, color='red', alpha=0.7, label='Excitatory')
            axes[1, 0].bar(x_pos, inh_values, bottom=exc_values, color='blue', alpha=0.7, label='Inhibitory')
            axes[1, 0].bar(x_pos, unl_values, bottom=np.array(exc_values) + np.array(inh_values), 
                        color='gray', alpha=0.7, label='Unlabeled')
            
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(response_types_list, rotation=45)
            axes[1, 0].set_ylabel('Fraction of Cells')
            axes[1, 0].set_title('Response Type Preferences by Cell Type')
            axes[1, 0].legend()
            
            # Plot 5: Reliability vs Responsiveness scatter
            for cell_type, profiles in roi_profiles.items():
                if profiles:
                    x_vals = [p['n_responsive_events'] for p in profiles]
                    y_vals = [p['reliability'] for p in profiles if not np.isnan(p['reliability'])]
                    x_vals = x_vals[:len(y_vals)]  # Match lengths
                    
                    axes[1, 1].scatter(x_vals, y_vals, color=colors[cell_type], 
                                    alpha=0.6, s=50, label=f'{cell_type}')
            
            axes[1, 1].set_xlabel('Number of Responsive Events')
            axes[1, 1].set_ylabel('Mean Trial Reliability')
            axes[1, 1].set_title('Reliability vs Responsiveness')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: Summary statistics table
            axes[1, 2].axis('off')
            
            # Create summary table
            summary_text = "Summary Statistics by Cell Type\n\n"
            for cell_type, profiles in roi_profiles.items():
                if profiles:
                    n_cells = len(profiles)
                    mean_responsiveness = np.mean([p['n_responsive_events'] for p in profiles])
                    mean_selectivity = np.mean([p['event_selectivity'] for p in profiles 
                                            if not np.isnan(p['event_selectivity'])])
                    
                    summary_text += f"{cell_type.title()}:\n"
                    summary_text += f"  N = {n_cells}\n"
                    summary_text += f"  Mean responsiveness = {mean_responsiveness:.1f}\n"
                    summary_text += f"  Mean selectivity = {mean_selectivity:.2f}\n\n"
            
            axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to create ROI response profiles: {e}")


    def plot_population_dynamics(self, responsiveness_data: Dict[str, Any], 
                            trial_data: Dict[str, Any], subject_id: str) -> None:
        """Analyze population-level dynamics across events."""
        try:
            import matplotlib.pyplot as plt
            
            # Get responsive ROI populations for different events
            event_populations = {}
            events = responsiveness_data['analysis_parameters']['alignment_events']
            
            for event in events:
                event_populations[event] = {
                    'activation_rois': [],
                    'suppression_rois': [],
                    'timing_data': []
                }
                
                for roi_idx, roi_data in responsiveness_data['roi_responsiveness'].items():
                    event_data = roi_data.get('event_responses', {}).get(event, {})
                    
                    for filter_name, filter_data in event_data.items():
                        # Check for activation
                        activation_result = filter_data.get('activation', {})
                        if activation_result.get('is_responsive', False):
                            event_populations[event]['activation_rois'].append(roi_idx)
                            event_populations[event]['timing_data'].append({
                                'roi': roi_idx,
                                'type': 'activation',
                                'timing': activation_result.get('response_timing', 0),
                                'magnitude': activation_result.get('effect_size', 0)
                            })
                        
                        # Check for suppression
                        suppression_result = filter_data.get('suppression', {})
                        if suppression_result.get('is_responsive', False):
                            event_populations[event]['suppression_rois'].append(roi_idx)
                            event_populations[event]['timing_data'].append({
                                'roi': roi_idx,
                                'type': 'suppression',
                                'timing': suppression_result.get('response_timing', 0),
                                'magnitude': abs(suppression_result.get('effect_size', 0))
                            })
            
            # Create population dynamics plot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Population Response Dynamics - {subject_id}', fontsize=14)
            
            # Plot 1: Population size by event
            activation_counts = [len(event_populations[event]['activation_rois']) for event in events]
            suppression_counts = [len(event_populations[event]['suppression_rois']) for event in events]
            
            x_pos = np.arange(len(events))
            width = 0.35
            
            axes[0, 0].bar(x_pos - width/2, activation_counts, width, label='Activation', 
                        color='red', alpha=0.7)
            axes[0, 0].bar(x_pos + width/2, suppression_counts, width, label='Suppression', 
                        color='blue', alpha=0.7)
            
            axes[0, 0].set_xlabel('Event')
            axes[0, 0].set_ylabel('Number of Responsive ROIs')
            axes[0, 0].set_title('Population Response Size by Event')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(events, rotation=45)
            axes[0, 0].legend()
            
            # Plot 2: Response timing distributions
            for i, event in enumerate(events):
                timing_data = event_populations[event]['timing_data']
                if timing_data:
                    timings = [td['timing'] for td in timing_data]
                    axes[0, 1].hist(timings, alpha=0.6, label=f'{event}', bins=20)
            
            axes[0, 1].set_xlabel('Response Timing (s)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Response Timing Distribution by Event')
            axes[0, 1].legend()
            axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Plot 3: Population overlap analysis
            # Create Venn-like analysis of ROI overlap between events
            if len(events) >= 2:
                event1_rois = set(event_populations[events[0]]['activation_rois'] + 
                                event_populations[events[0]]['suppression_rois'])
                event2_rois = set(event_populations[events[1]]['activation_rois'] + 
                                event_populations[events[1]]['suppression_rois'])
                
                overlap = len(event1_rois & event2_rois)
                event1_only = len(event1_rois - event2_rois)
                event2_only = len(event2_rois - event1_rois)
                
                categories = [f'{events[0]} only', 'Overlap', f'{events[1]} only']
                counts = [event1_only, overlap, event2_only]
                
                axes[1, 0].pie(counts, labels=categories, autopct='%1.1f%%', 
                            colors=['lightcoral', 'yellow', 'lightblue'])
                axes[1, 0].set_title(f'ROI Population Overlap\n{events[0]} vs {events[1]}')
            
            # Plot 4: Magnitude vs Timing scatter
            colors_map = {'activation': 'red', 'suppression': 'blue'}
            
            for event in events:
                timing_data = event_populations[event]['timing_data']
                for td in timing_data:
                    axes[1, 1].scatter(td['timing'], td['magnitude'], 
                                    color=colors_map[td['type']], alpha=0.6, s=30)
            
            axes[1, 1].set_xlabel('Response Timing (s)')
            axes[1, 1].set_ylabel('Response Magnitude')
            axes[1, 1].set_title('Response Magnitude vs Timing')
            axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add legend
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', label='Activation')
            blue_patch = mpatches.Patch(color='blue', label='Suppression')
            axes[1, 1].legend(handles=[red_patch, blue_patch])
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to create population dynamics plot: {e}")


    def plot_population_vector_analysis(self, trial_data: Dict[str, Any], 
                                    responsiveness_data: Dict[str, Any], subject_id: str) -> None:
        """Analyze population vector trajectories over time."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            
            # Extract population activity for responsive ROIs
            responsive_rois = []
            for roi_idx, roi_data in responsiveness_data['roi_responsiveness'].items():
                if len(roi_data['cross_event_patterns']['responsive_events']) > 0:
                    responsive_rois.append(roi_idx)
            
            if len(responsive_rois) < 10:
                self.logger.warning("Too few responsive ROIs for PCA analysis")
                return
            
            # Get trial data in DataFrame format
            if 'df_trials_with_segments' not in trial_data:
                self.logger.error("Need DataFrame format for population analysis")
                return
            
            df_valid = trial_data['df_trials_with_segments'][trial_data['valid_trials_mask']]
            
            # Separate left and right trials
            left_trials = df_valid[df_valid['trial_side'] == 'left']
            right_trials = df_valid[df_valid['trial_side'] == 'right']
            
            if len(left_trials) < 5 or len(right_trials) < 5:
                self.logger.warning("Too few trials for population analysis")
                return
            
            # Extract population vectors
            left_population = np.array([trial['dff_segment'][responsive_rois, :] 
                                    for _, trial in left_trials.iterrows()])  # (n_left_trials, n_rois, n_time)
            right_population = np.array([trial['dff_segment'][responsive_rois, :] 
                                        for _, trial in right_trials.iterrows()]) # (n_right_trials, n_rois, n_time)
            
            time_vector = left_trials.iloc[0]['dff_time_vector']
            
            # Reshape for PCA: (n_trials * n_timepoints, n_rois)
            left_reshaped = left_population.reshape(-1, len(responsive_rois))
            right_reshaped = right_population.reshape(-1, len(responsive_rois))
            all_data = np.vstack([left_reshaped, right_reshaped])
            
            # Fit PCA on all data
            pca = PCA(n_components=3)
            pca.fit(all_data)
            
            # Transform data
            left_pca = pca.transform(left_reshaped).reshape(len(left_trials), len(time_vector), 3)
            right_pca = pca.transform(right_reshaped).reshape(len(right_trials), len(time_vector), 3)
            
            # Create population vector plots
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle(f'Population Vector Analysis - {subject_id}', fontsize=16)
            
            # Plot 1: PCA trajectories over time
            ax1 = plt.subplot(2, 3, 1, projection='3d')
            
            # Average trajectories
            left_mean_traj = np.mean(left_pca, axis=0)
            right_mean_traj = np.mean(right_pca, axis=0)
            
            # Plot trajectories
            ax1.plot(left_mean_traj[:, 0], left_mean_traj[:, 1], left_mean_traj[:, 2], 
                    'r-', linewidth=3, label=f'Left (n={len(left_trials)})')
            ax1.plot(right_mean_traj[:, 0], right_mean_traj[:, 1], right_mean_traj[:, 2], 
                    'b-', linewidth=3, label=f'Right (n={len(right_trials)})')
            
            # Mark time points
            time_points = [0, len(time_vector)//4, len(time_vector)//2, 3*len(time_vector)//4, -1]
            for tp in time_points:
                ax1.scatter(left_mean_traj[tp, 0], left_mean_traj[tp, 1], left_mean_traj[tp, 2], 
                        c='red', s=100, alpha=0.8)
                ax1.scatter(right_mean_traj[tp, 0], right_mean_traj[tp, 1], right_mean_traj[tp, 2], 
                        c='blue', s=100, alpha=0.8)
            
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            ax1.set_title('Population Trajectory (3D)')
            ax1.legend()
            
            # Plot 2: PC1 vs PC2 projection
            ax2 = plt.subplot(2, 3, 2)
            ax2.plot(left_mean_traj[:, 0], left_mean_traj[:, 1], 'r-', linewidth=2, label='Left')
            ax2.plot(right_mean_traj[:, 0], right_mean_traj[:, 1], 'b-', linewidth=2, label='Right')
            
            # Mark start and end points
            ax2.scatter(left_mean_traj[0, 0], left_mean_traj[0, 1], c='red', s=100, marker='o', label='Start')
            ax2.scatter(left_mean_traj[-1, 0], left_mean_traj[-1, 1], c='red', s=100, marker='s', label='End')
            ax2.scatter(right_mean_traj[0, 0], right_mean_traj[0, 1], c='blue', s=100, marker='o')
            ax2.scatter(right_mean_traj[-1, 0], right_mean_traj[-1, 1], c='blue', s=100, marker='s')
            
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax2.set_title('Population Trajectory (2D)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Time courses of PCs
            ax3 = plt.subplot(2, 3, 3)
            for pc in range(3):
                ax3.plot(time_vector, left_mean_traj[:, pc], 
                        color=f'C{pc}', linestyle='-', linewidth=2, 
                        label=f'Left PC{pc+1}')
                ax3.plot(time_vector, right_mean_traj[:, pc], 
                        color=f'C{pc}', linestyle='--', linewidth=2, 
                        label=f'Right PC{pc+1}')
            
            ax3.axvline(x=0, color='black', linestyle=':', alpha=0.7)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('PC Score')
            ax3.set_title('Principal Component Time Courses')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Population distance over time
            ax4 = plt.subplot(2, 3, 4)
            population_distance = np.sqrt(np.sum((left_mean_traj - right_mean_traj)**2, axis=1))
            ax4.plot(time_vector, population_distance, 'purple', linewidth=3)
            ax4.axvline(x=0, color='black', linestyle=':', alpha=0.7)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Population Distance')
            ax4.set_title('Left vs Right Population Distance')
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Variance explained
            ax5 = plt.subplot(2, 3, 5)
            ax5.bar(range(1, len(pca.explained_variance_ratio_)+1), 
                pca.explained_variance_ratio_ * 100, alpha=0.7)
            ax5.set_xlabel('Principal Component')
            ax5.set_ylabel('Variance Explained (%)')
            ax5.set_title('PCA Variance Explained')
            ax5.grid(True, alpha=0.3)
            
            # Plot 6: Individual trial trajectories (sample)
            ax6 = plt.subplot(2, 3, 6)
            
            # Plot a few individual trials
            n_sample_trials = min(5, len(left_trials), len(right_trials))
            for i in range(n_sample_trials):
                ax6.plot(left_pca[i, :, 0], left_pca[i, :, 1], 
                        'r-', alpha=0.3, linewidth=1)
                ax6.plot(right_pca[i, :, 0], right_pca[i, :, 1], 
                        'b-', alpha=0.3, linewidth=1)
            
            # Overlay mean trajectories
            ax6.plot(left_mean_traj[:, 0], left_mean_traj[:, 1], 'r-', linewidth=3, label='Left mean')
            ax6.plot(right_mean_traj[:, 0], right_mean_traj[:, 1], 'b-', linewidth=3, label='Right mean')
            
            ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax6.set_title('Individual Trial Variability')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Log analysis results
            max_distance_time = time_vector[np.argmax(population_distance)]
            max_distance_value = np.max(population_distance)
            
            self.logger.info(f"Population Vector Analysis Results:")
            self.logger.info(f"  Responsive ROIs analyzed: {len(responsive_rois)}")
            self.logger.info(f"  Variance explained by PC1-3: {pca.explained_variance_ratio_[:3].sum():.1%}")
            self.logger.info(f"  Maximum population distance: {max_distance_value:.3f} at {max_distance_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to create population vector analysis: {e}")



    def correct_trial_timestamps_with_voltage_pulses(self, df_trials: pd.DataFrame, imaging_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Correct trial timestamps by finding the closest voltage pulse to each expected trial start.
        
        Args:
            df_trials: DataFrame containing behavioral trial data with trial_start_timestamp column
            imaging_data: Dictionary containing voltage data including vol_start and vol_time
            
        Returns:
            DataFrame with corrected trial_start_timestamp values
        """
        try:
            # Work with a copy to avoid modifying original
            df_corrected = df_trials.copy()
            
            for trial_idx, row in df_corrected.iterrows():
                expected_time = row['trial_start_timestamp']
                actual_pulse_time = self.find_closest_vol_pulse(imaging_data, expected_time, search_window=0.2)
                df_corrected.loc[trial_idx, 'trial_start_timestamp'] = actual_pulse_time

            return df_corrected
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to correct trial timestamps: {e}")
            return df_trials  # Return original if correction fails



    def explore_roi_responsiveness_patterns(self, trial_data: Dict[str, Any], 
                                        responsiveness_data: Dict[str, Any], 
                                        alignment: str,
                                        subject_id: str) -> None:
        """
        Comprehensive exploration of ROI responsiveness patterns across different parameter combinations.
        Shows sort methods contiguously for each combination to compare their effects.
        
        Args:
            trial_data: Dictionary from extract_trial_segments_simple
            responsiveness_data: Dictionary from analyze_roi_responsiveness
            alignment: Alignment event string
            subject_id: Subject ID for plot titles
        """
        try:
            if 'error' in trial_data or 'error' in responsiveness_data:
                self.logger.error("SID_IMG_PRE: Cannot explore patterns - data extraction failed")
                return
            
            # Get available parameters from responsiveness data
            events = [alignment]
            trial_filters = list(responsiveness_data['analysis_parameters']['trial_filters'].keys())
            response_types = list(responsiveness_data['analysis_parameters']['responsiveness_types'].keys())
            
            # Define parameter combinations to test
            sort_methods = ['effect_size', 'p_value', 'response_magnitude', 'response_timing']
            trial_types = ['all', 'left', 'right', 'rewarded', 'punished']
            
            self.logger.info(f"SID_IMG_PRE: Starting comprehensive responsiveness pattern exploration for {subject_id}")
            self.logger.info(f"SID_IMG_PRE: Testing {len(events)} events × {len(trial_filters)} filters × {len(response_types)} response types × {len(trial_types)} trial types × {len(sort_methods)} sort methods")
            
            # FIX: Create base combinations in the order you want: iterate through response_types first, then trial_types
            base_combinations = []
            
            for event in events:
                for trial_filter in trial_filters:
                    # NEW: First iterate through ALL response types with trial_type='all'
                    for response_type in response_types:
                        # Check if this combination has any responsive ROIs
                        responsive_count = 0
                        for roi_idx, roi_data in responsiveness_data['roi_responsiveness'].items():
                            event_data = roi_data.get('event_responses', {}).get(event, {})
                            filter_data = event_data.get(trial_filter, {})
                            result = filter_data.get(response_type, {})
                            if result.get('is_responsive', False):
                                responsive_count += 1
                        
                        if responsive_count >= 5:  # Only test combinations with at least 5 responsive ROIs
                            base_combinations.append({
                                'event': event,
                                'trial_filter': trial_filter,
                                'response_type': response_type,
                                'trial_type': 'all',  # Start with 'all' for each response type
                                'responsive_count': responsive_count
                            })
                    
                    # NEW: Then iterate through ALL response types again with other trial_types
                    for trial_type in trial_types[1:]:  # Skip 'all' since we already did it
                        for response_type in response_types:
                            # Check if this combination has any responsive ROIs
                            responsive_count = 0
                            for roi_idx, roi_data in responsiveness_data['roi_responsiveness'].items():
                                event_data = roi_data.get('event_responses', {}).get(event, {})
                                filter_data = event_data.get(trial_filter, {})
                                result = filter_data.get(response_type, {})
                                if result.get('is_responsive', False):
                                    responsive_count += 1
                            
                            if responsive_count >= 5:  # Only test combinations with at least 5 responsive ROIs
                                base_combinations.append({
                                    'event': event,
                                    'trial_filter': trial_filter,
                                    'response_type': response_type,
                                    'trial_type': trial_type,
                                    'responsive_count': responsive_count
                                })
            
            # Sort base combinations by responsive count (most interesting first)
            base_combinations.sort(key=lambda x: x['responsive_count'], reverse=True)
            
            self.logger.info(f"SID_IMG_PRE: Found {len(base_combinations)} valid base combinations (≥5 responsive ROIs)")
            
            # Create summary table of top base combinations
            self.logger.info("SID_IMG_PRE: Top 10 most responsive base combinations:")
            for i, combo in enumerate(base_combinations[:10]):
                self.logger.info(f"  {i+1:2d}. {combo['event']:15s} | {combo['trial_filter']:8s} | {combo['response_type']:12s} | {combo['trial_type']:10s} | {combo['responsive_count']:3d} ROIs")
            
            # Group plots by base combination, showing all sort methods contiguously
            max_combinations_to_show = 12  # Adjust this to control how many combinations to plot
            combinations_to_plot = base_combinations[:max_combinations_to_show]
            
            self.logger.info(f"SID_IMG_PRE: Plotting top {len(combinations_to_plot)} combinations with all sort methods")
            
            plot_count = 0
            for combo_idx, base_combo in enumerate(combinations_to_plot):
                combo_number = combo_idx + 1
                self.logger.info(f"SID_IMG_PRE: Combination {combo_number}: {base_combo['event']} | {base_combo['trial_filter']} | {base_combo['response_type']} | {base_combo['trial_type']} ({base_combo['responsive_count']} ROIs)")
                
                # Plot this combination with each sort method
                for sort_idx, sort_method in enumerate(sort_methods):
                    try:
                        plot_count += 1
                        
                        self.logger.info(f"    Sort method {sort_idx + 1}/{len(sort_methods)}: {sort_method}")
                        
                        # Generate the plot
                        self.plot_all_rois_heatmap_by_responsiveness(
                            trial_data, responsiveness_data, subject_id,
                            event_filter=base_combo['event'],
                            trial_filter=base_combo['trial_filter'],
                            response_type_filter=base_combo['response_type'],
                            sort_method=sort_method,
                            trial_type=base_combo['trial_type']
                        )
                        
                    except Exception as e:
                        self.logger.warning(f"    Failed to plot {sort_method}: {e}")
                        continue
            
            # Generate exploration summary report
            self._generate_exploration_summary_by_combination(base_combinations, sort_methods, subject_id)
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to explore responsiveness patterns: {e}")



    def _get_response_type_survey(self, combinations: List[Dict]) -> List[Dict]:
        """Get representative combinations for each response type."""
        response_type_examples = {}
        
        for combo in combinations:
            resp_type = combo['response_type']
            if resp_type not in response_type_examples:
                response_type_examples[resp_type] = []
            
            # Take the top 2 combinations for each response type
            if len(response_type_examples[resp_type]) < 2:
                response_type_examples[resp_type].append(combo)
        
        # Flatten the results
        result = []
        for resp_type, examples in response_type_examples.items():
            result.extend(examples)
        
        return result


    def _get_event_comparison(self, combinations: List[Dict]) -> List[Dict]:
        """Get combinations that allow comparison across events."""
        # Find the most responsive trial_filter + response_type combination
        combo_counts = {}
        for combo in combinations:
            key = (combo['trial_filter'], combo['response_type'])
            if key not in combo_counts:
                combo_counts[key] = []
            combo_counts[key].append(combo)
        
        # Get the top combination and show it for each event
        if combo_counts:
            top_combo_key = max(combo_counts.keys(), key=lambda k: sum(c['responsive_count'] for c in combo_counts[k]))
            
            # Return one example for each event using this combination
            result = []
            seen_events = set()
            for combo in combo_counts[top_combo_key]:
                if combo['event'] not in seen_events:
                    result.append(combo)
                    seen_events.add(combo['event'])
            
            return result
        
        return []


    def _get_sort_method_demo(self, combinations: List[Dict]) -> List[Dict]:
        """Get combinations that demonstrate different sort methods."""
        if not combinations:
            return []
        
        # Take the most responsive combination and show it with different sort methods
        top_combo = combinations[0]
        
        result = []
        sort_methods = ['effect_size', 'p_value', 'response_magnitude', 'response_timing']
        
        for sort_method in sort_methods:
            combo_copy = top_combo.copy()
            combo_copy['sort_method'] = sort_method
            result.append(combo_copy)
        
        return result


    def _generate_exploration_summary(self, combinations: List[Dict], subject_id: str) -> None:
        """Generate a summary report of the exploration results."""
        try:
            import matplotlib.pyplot as plt
            
            # Analyze patterns in the combinations
            event_responsiveness = {}
            response_type_responsiveness = {}
            trial_filter_responsiveness = {}
            
            for combo in combinations:
                # Count by event
                event = combo['event']
                if event not in event_responsiveness:
                    event_responsiveness[event] = []
                event_responsiveness[event].append(combo['responsive_count'])
                
                # Count by response type
                resp_type = combo['response_type']
                if resp_type not in response_type_responsiveness:
                    response_type_responsiveness[resp_type] = []
                response_type_responsiveness[resp_type].append(combo['responsive_count'])
                
                # Count by trial filter
                trial_filter = combo['trial_filter']
                if trial_filter not in trial_filter_responsiveness:
                    trial_filter_responsiveness[trial_filter] = []
                trial_filter_responsiveness[trial_filter].append(combo['responsive_count'])
            
            # Create summary plot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'ROI Responsiveness Exploration Summary - {subject_id}', fontsize=14)
            
            # Plot 1: Responsiveness by event
            events = list(event_responsiveness.keys())
            event_means = [np.mean(event_responsiveness[event]) for event in events]
            event_maxes = [np.max(event_responsiveness[event]) for event in events]
            
            x_pos = np.arange(len(events))
            axes[0, 0].bar(x_pos, event_means, alpha=0.7, label='Mean')
            axes[0, 0].bar(x_pos, event_maxes, alpha=0.4, label='Max')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(events, rotation=45)
            axes[0, 0].set_ylabel('Responsive ROI Count')
            axes[0, 0].set_title('Responsiveness by Event')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Responsiveness by response type
            resp_types = list(response_type_responsiveness.keys())
            resp_type_means = [np.mean(response_type_responsiveness[rt]) for rt in resp_types]
            resp_type_maxes = [np.max(response_type_responsiveness[rt]) for rt in resp_types]
            
            x_pos = np.arange(len(resp_types))
            axes[0, 1].bar(x_pos, resp_type_means, alpha=0.7, label='Mean')
            axes[0, 1].bar(x_pos, resp_type_maxes, alpha=0.4, label='Max')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(resp_types, rotation=45)
            axes[0, 1].set_ylabel('Responsive ROI Count')
            axes[0, 1].set_title('Responsiveness by Response Type')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Responsiveness by trial filter
            filters = list(trial_filter_responsiveness.keys())
            filter_means = [np.mean(trial_filter_responsiveness[f]) for f in filters]
            filter_maxes = [np.max(trial_filter_responsiveness[f]) for f in filters]
            
            x_pos = np.arange(len(filters))
            axes[1, 0].bar(x_pos, filter_means, alpha=0.7, label='Mean')
            axes[1, 0].bar(x_pos, filter_maxes, alpha=0.4, label='Max')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(filters, rotation=45)
            axes[1, 0].set_ylabel('Responsive ROI Count')
            axes[1, 0].set_title('Responsiveness by Trial Filter')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Top combinations table
            axes[1, 1].axis('off')
            
            # Create table of top combinations
            top_10 = combinations[:10]
            table_text = "Top 10 Most Responsive Combinations:\n\n"
            table_text += f"{'#':<2} {'Event':<15} {'Filter':<8} {'Type':<12} {'ROIs':<4}\n"
            table_text += "-" * 50 + "\n"
            
            for i, combo in enumerate(top_10):
                table_text += f"{i+1:<2} {combo['event']:<15} {combo['trial_filter']:<8} {combo['response_type']:<12} {combo['responsive_count']:<4}\n"
            
            axes[1, 1].text(0.05, 0.95, table_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            # Log summary statistics
            total_combinations = len(combinations)
            total_responsive_rois = sum(combo['responsive_count'] for combo in combinations)
            mean_responsive = total_responsive_rois / total_combinations if total_combinations > 0 else 0
            
            self.logger.info(f"SID_IMG_PRE: Exploration Summary for {subject_id}:")
            self.logger.info(f"  Total valid combinations: {total_combinations}")
            self.logger.info(f"  Mean responsive ROIs per combination: {mean_responsive:.1f}")
            self.logger.info(f"  Most responsive combination: {combinations[0]['responsive_count']} ROIs")
            self.logger.info(f"  Best event: {max(events, key=lambda e: np.mean(event_responsiveness[e]))}")
            self.logger.info(f"  Best response type: {max(resp_types, key=lambda rt: np.mean(response_type_responsiveness[rt]))}")
            self.logger.info(f"  Best trial filter: {max(filters, key=lambda f: np.mean(trial_filter_responsiveness[f]))}")
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to generate exploration summary: {e}")


    def align_imaging_behavioral_timestamps(self, imaging_data: Dict[str, Any], behavioral_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Align imaging and behavioral timestamps to session start.
        
        Args:
            imaging_data: Dictionary containing imaging and voltage data
            behavioral_data: Dictionary containing behavioral trial data
            
        Returns:
            Tuple of (aligned_imaging_data, aligned_behavioral_data)
        """
        try:
            self.logger.info("SID_IMG_PRE: Aligning imaging and behavioral timestamps")
            
            # Get voltage data
            vol_time = imaging_data.get('vol_time', None)
            vol_img = imaging_data.get('vol_img', None)
            vol_start = imaging_data.get('vol_start', None)
            
            if vol_time is None or vol_img is None or vol_start is None:
                self.logger.error("SID_IMG_PRE: Missing voltage data for alignment")
                return imaging_data, behavioral_data
            
            # Get scope imaging signal trigger timestamps
            imaging_time_array, _ = self.get_trigger_time(vol_time, vol_img)
            
            # Adjust for photon integration cycle delay
            imaging_time_array = self.adjust_imaging_timing_for_integration_delay(imaging_time_array)
            
            # Calculate imaging parameters
            n_frames = imaging_data['n_frames']
            imaging_duration = imaging_time_array[-1] - imaging_time_array[0]
            imaging_fs = n_frames / imaging_duration
            
            # Update imaging data with timing information
            imaging_data['imaging_time'] = imaging_time_array
            imaging_data['imaging_duration'] = imaging_duration
            imaging_data['imaging_fs'] = imaging_fs
            
            # imaging_data['fluo_traces']
            # imaging_data['neuropil']
            # imaging_data['neucoeff']
            # imaging_data['imaging_fs']
            
            
            # imaging_data['spks'], params = self.derive_spikes_from_imaging(imaging_data, tau=0.7)
            
            
            self.logger.info(f"SID_IMG_PRE: Imaging sampling rate: {imaging_fs:.2f} Hz")
            self.logger.info(f"SID_IMG_PRE: Imaging duration: {imaging_duration:.1f} seconds")
            
            # Find first session start pulse for alignment reference
            first_start_pulse_time, _ = self.get_trigger_time(vol_time, vol_start)
            first_start_pulse_time = first_start_pulse_time[0]
            
            # Align voltage time to zero at first pulse
            imaging_data['vol_time'] = vol_time - first_start_pulse_time
            imaging_data['vol_duration'] = imaging_data['vol_time'][-1] - imaging_data['vol_time'][0]
            imaging_data['vol_duration_session'] = imaging_data['vol_time'][-1]
            
            # Align imaging time to same reference point
            imaging_data['imaging_time'] = imaging_data['imaging_time'] - first_start_pulse_time
            imaging_data['imaging_duration_session'] = imaging_data['imaging_time'][-1]
            
            self.logger.info(f"SID_IMG_PRE: Aligned time vectors to first vol_start rising edge at original time {first_start_pulse_time:.3f}s")
            self.logger.info(f"SID_IMG_PRE: Total imaging duration: {imaging_duration:.1f}s")
            self.logger.info(f"SID_IMG_PRE: Total voltage duration: {imaging_data['vol_duration']:.1f}s")
            self.logger.info(f"SID_IMG_PRE: Aligned imaging duration: {imaging_data['imaging_duration_session']:.1f}s")
            self.logger.info(f"SID_IMG_PRE: Aligned voltage duration: {imaging_data['vol_duration_session']:.1f}s")
            
            # Correct trial timestamps using voltage pulses
            self.logger.info("SID_IMG_PRE: Correcting trial timestamps with voltage pulses")
            behavioral_data['df_trials'] = self.correct_trial_timestamps_with_voltage_pulses(
                behavioral_data['df_trials'], imaging_data
            )
            
            return imaging_data, behavioral_data
            
        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Failed to align timestamps: {e}")
            return imaging_data, behavioral_data



    def process_subject(self, subject_id: str, suite2p_path: str, output_path: str, force: bool = False) -> Dict[str, Any]:
        """
        Minimal preprocessing:
        - If both cache files exist and not force -> do NOT load, just report success.
        - Else load -> align -> save, and report success.
        """
        try:
            self.logger.info(f"SID_IMG_PRE: Start SID imaging preprocessing for {subject_id}")
            os.makedirs(output_path, exist_ok=True)

            imaging_data_file = os.path.join(output_path, 'sid_imaging_preprocess_data.pkl')
            behavioral_data_file = os.path.join(output_path, 'sid_behavioral_preprocess_data.pkl')

            # If both files exist and not forcing, just report success (do not load)
            if (os.path.exists(imaging_data_file) and os.path.exists(behavioral_data_file)) and not force:
                self.logger.info("SID_IMG_PRE: Found existing preprocessed files; skipping load/process.")
                return {
                    'success': True,
                    'sessions_processed': 0,
                    'used_cache': True,
                    'imaging_data_file': imaging_data_file,
                    'behavioral_data_file': behavioral_data_file,
                    'message': 'Existing preprocessed files present'
                }

            # Otherwise, load/process/save
            subject_config = self.config.get('subjects', {}).get(subject_id, {})
            imaging_sessions = subject_config.get('imaging_sessions', [])
            if not imaging_sessions:
                return {'success': False, 'sessions_processed': 0, 'error_message': f'No imaging sessions for {subject_id}'}

            imaging_session = imaging_sessions[0]
            behavior_session_id = imaging_session.get('behavior_session')
            if not behavior_session_id:
                return {'success': False, 'sessions_processed': 0, 'error_message': 'Missing behavior_session in config'}

            # Load behavior
            behavioral_data = self.load_behavior_session(subject_id, behavior_session_id)
            if behavioral_data is None:
                return {'success': False, 'sessions_processed': 0, 'error_message': 'Failed to load behavioral data'}

            # Load imaging
            imaging_data = self.load_imaging_data(subject_id, suite2p_path, output_path)
            if imaging_data is None:
                return {'success': False, 'sessions_processed': 0, 'error_message': 'Failed to load imaging data'}

            # Align
            imaging_data, behavioral_data = self.align_imaging_behavioral_timestamps(imaging_data, behavioral_data)

            # Save
            try:
                with open(imaging_data_file, 'wb') as f:
                    pickle.dump(imaging_data, f)
                with open(behavioral_data_file, 'wb') as f:
                    pickle.dump(behavioral_data, f)
                self.logger.info("SID_IMG_PRE: Saved preprocessed imaging/behavioral data.")
            except Exception as e:
                self.logger.warning(f"SID_IMG_PRE: Failed saving preprocessed data: {e}")



            # TODO update integration of trial segmentation save/load/proc after we get preproc/analysis stable

            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='trial_start', pre_sec=0.0, post_sec=8.0
            # )            
            
            # trial_data['session_info'] = {}
            # trial_data['session_info']['unique_isis'] = behavioral_data['session_info']['unique_isis']
            # trial_data['session_info']['mean_isi'] = behavioral_data['session_info']['mean_isi']
            # trial_data['session_info']['long_isis'] = behavioral_data['session_info']['long_isis']
            # trial_data['session_info']['short_isis'] = behavioral_data['session_info']['short_isis']
            # trial_data['session_id'] = behavioral_data['session_id']
            # trial_data['subject_name'] = behavioral_data['subject_name']


            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")


            # segmented_imaging_data_file = os.path.join(output_path, 'sid_imaging_segmented_data.pkl')

            # trial_data = {}
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='trial_start', pre_sec=0.0, post_sec=8.0
            # )            
            
            # trial_data['session_info'] = {}
            # trial_data['session_info']['unique_isis'] = behavioral_data['session_info']['unique_isis']
            # trial_data['session_info']['mean_isi'] = behavioral_data['session_info']['mean_isi']
            # trial_data['session_info']['long_isis'] = behavioral_data['session_info']['long_isis']
            # trial_data['session_info']['short_isis'] = behavioral_data['session_info']['short_isis']
            # trial_data['session_id'] = behavioral_data['session_id']
            # trial_data['subject_name'] = behavioral_data['subject_name']


            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")


            # segmented_imaging_data_file = os.path.join(output_path, 'trial_start.pkl')
            # # Save
            # try:
            #     with open(segmented_imaging_data_file, 'wb') as f:
            #         pickle.dump(trial_data, f)
            #     self.logger.info("SID_IMG_PRE: Saved segmented imaging/behavioral data.")
            # except Exception as e:
            #     self.logger.warning(f"SID_IMG_PRE: Failed segmented preprocessed data: {e}") 

            
            # trial_data = {}
            # max_isi = np.max(behavioral_data['df_trials']['isi']) / 1000
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='end_flash_1', pre_sec=0.2, post_sec=max_isi
            # )            
            
            # trial_data['session_info'] = {}
            # trial_data['session_info']['unique_isis'] = behavioral_data['session_info']['unique_isis']
            # trial_data['session_info']['mean_isi'] = behavioral_data['session_info']['mean_isi']
            # trial_data['session_info']['long_isis'] = behavioral_data['session_info']['long_isis']
            # trial_data['session_info']['short_isis'] = behavioral_data['session_info']['short_isis']
            # trial_data['session_id'] = behavioral_data['session_id']
            # trial_data['subject_name'] = behavioral_data['subject_name']


            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")


            # segmented_imaging_data_file = os.path.join(output_path, 'trial_isi.pkl')
            # # Save
            # try:
            #     with open(segmented_imaging_data_file, 'wb') as f:
            #         pickle.dump(trial_data, f)
            #     self.logger.info("SID_IMG_PRE: Saved segmented imaging/behavioral data.")
            # except Exception as e:
            #     self.logger.warning(f"SID_IMG_PRE: Failed segmented preprocessed data: {e}") 



            # trial_data = {}   
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='start_flash_1', pre_sec=0.25, post_sec=0.5
            # )            
            
            # trial_data['session_info'] = {}
            # trial_data['session_info']['unique_isis'] = behavioral_data['session_info']['unique_isis']
            # trial_data['session_info']['mean_isi'] = behavioral_data['session_info']['mean_isi']
            # trial_data['session_info']['long_isis'] = behavioral_data['session_info']['long_isis']
            # trial_data['session_info']['short_isis'] = behavioral_data['session_info']['short_isis']
            # trial_data['session_id'] = behavioral_data['session_id']
            # trial_data['subject_name'] = behavioral_data['subject_name']


            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")


            # segmented_imaging_data_file = os.path.join(output_path, 'start_flash_1.pkl')
            # # Save
            # try:
            #     with open(segmented_imaging_data_file, 'wb') as f:
            #         pickle.dump(trial_data, f)
            #     self.logger.info("SID_IMG_PRE: Saved segmented imaging/behavioral data.")
            # except Exception as e:
            #     self.logger.warning(f"SID_IMG_PRE: Failed segmented preprocessed data: {e}") 


            # trial_data = {}   
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='end_flash_1', pre_sec=0.25, post_sec=0.6
            # )            
            
            # trial_data['session_info'] = {}
            # trial_data['session_info']['unique_isis'] = behavioral_data['session_info']['unique_isis']
            # trial_data['session_info']['mean_isi'] = behavioral_data['session_info']['mean_isi']
            # trial_data['session_info']['long_isis'] = behavioral_data['session_info']['long_isis']
            # trial_data['session_info']['short_isis'] = behavioral_data['session_info']['short_isis']
            # trial_data['session_id'] = behavioral_data['session_id']
            # trial_data['subject_name'] = behavioral_data['subject_name']


            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")


            # segmented_imaging_data_file = os.path.join(output_path, 'end_flash_1.pkl')
            # # Save
            # try:
            #     with open(segmented_imaging_data_file, 'wb') as f:
            #         pickle.dump(trial_data, f)
            #     self.logger.info("SID_IMG_PRE: Saved segmented imaging/behavioral data.")
            # except Exception as e:
            #     self.logger.warning(f"SID_IMG_PRE: Failed segmented preprocessed data: {e}") 




            # trial_data = {}   
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='start_flash_2', pre_sec=0.25, post_sec=0.6
            # )            
            
            # trial_data['session_info'] = {}
            # trial_data['session_info']['unique_isis'] = behavioral_data['session_info']['unique_isis']
            # trial_data['session_info']['mean_isi'] = behavioral_data['session_info']['mean_isi']
            # trial_data['session_info']['long_isis'] = behavioral_data['session_info']['long_isis']
            # trial_data['session_info']['short_isis'] = behavioral_data['session_info']['short_isis']
            # trial_data['session_id'] = behavioral_data['session_id']
            # trial_data['subject_name'] = behavioral_data['subject_name']


            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")


            # segmented_imaging_data_file = os.path.join(output_path, 'start_flash_2.pkl')
            # # Save
            # try:
            #     with open(segmented_imaging_data_file, 'wb') as f:
            #         pickle.dump(trial_data, f)
            #     self.logger.info("SID_IMG_PRE: Saved segmented imaging/behavioral data.")
            # except Exception as e:
            #     self.logger.warning(f"SID_IMG_PRE: Failed segmented preprocessed data: {e}") 



            # trial_data = {}   
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='end_flash_2', pre_sec=0.25, post_sec=0.6
            # )            
            
            # trial_data['session_info'] = {}
            # trial_data['session_info']['unique_isis'] = behavioral_data['session_info']['unique_isis']
            # trial_data['session_info']['mean_isi'] = behavioral_data['session_info']['mean_isi']
            # trial_data['session_info']['long_isis'] = behavioral_data['session_info']['long_isis']
            # trial_data['session_info']['short_isis'] = behavioral_data['session_info']['short_isis']
            # trial_data['session_id'] = behavioral_data['session_id']
            # trial_data['subject_name'] = behavioral_data['subject_name']


            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")


            # segmented_imaging_data_file = os.path.join(output_path, 'end_flash_2.pkl')
            # # Save
            # try:
            #     with open(segmented_imaging_data_file, 'wb') as f:
            #         pickle.dump(trial_data, f)
            #     self.logger.info("SID_IMG_PRE: Saved segmented imaging/behavioral data.")
            # except Exception as e:
            #     self.logger.warning(f"SID_IMG_PRE: Failed segmented preprocessed data: {e}")             
            
            
            
            
            # trial_data = {}            
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='choice_start', pre_sec=0.6, post_sec=0.6
            # )            
            
            # trial_data['session_info'] = {}
            # trial_data['session_info']['unique_isis'] = behavioral_data['session_info']['unique_isis']
            # trial_data['session_info']['mean_isi'] = behavioral_data['session_info']['mean_isi']
            # trial_data['session_info']['long_isis'] = behavioral_data['session_info']['long_isis']
            # trial_data['session_info']['short_isis'] = behavioral_data['session_info']['short_isis']
            # trial_data['session_id'] = behavioral_data['session_id']
            # trial_data['subject_name'] = behavioral_data['subject_name']


            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")


            # segmented_imaging_data_file = os.path.join(output_path, 'choice_start.pkl')     
            # # Save
            # try:
            #     with open(segmented_imaging_data_file, 'wb') as f:
            #         pickle.dump(trial_data, f)
            #     self.logger.info("SID_IMG_PRE: Saved segmented imaging/behavioral data.")
            # except Exception as e:
            #     self.logger.warning(f"SID_IMG_PRE: Failed segmented preprocessed data: {e}")             
            
            
            
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='lick_start', pre_sec=0.4, post_sec=0.4
            # )            
            
            # trial_data['session_info'] = {}
            # trial_data['session_info']['unique_isis'] = behavioral_data['session_info']['unique_isis']
            # trial_data['session_info']['mean_isi'] = behavioral_data['session_info']['mean_isi']
            # trial_data['session_info']['long_isis'] = behavioral_data['session_info']['long_isis']
            # trial_data['session_info']['short_isis'] = behavioral_data['session_info']['short_isis']
            # trial_data['session_id'] = behavioral_data['session_id']
            # trial_data['subject_name'] = behavioral_data['subject_name']


            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")


            # segmented_imaging_data_file = os.path.join(output_path, 'lick_start.pkl')                   



            # # Save
            # try:
            #     with open(segmented_imaging_data_file, 'wb') as f:
            #         pickle.dump(trial_data, f)
            #     self.logger.info("SID_IMG_PRE: Saved segmented imaging/behavioral data.")
            # except Exception as e:
            #     self.logger.warning(f"SID_IMG_PRE: Failed segmented preprocessed data: {e}")            


            return {
                'success': True,
                'sessions_processed': 1,
                'used_cache': False,
                'imaging_data_file': imaging_data_file,
                'behavioral_data_file': behavioral_data_file,
                'message': 'Preprocessing completed'
            }

        # def process_subject(self, subject_id: str, suite2p_path: str, output_path: str, force: bool = False) -> Dict[str, Any]:
        #     """
        #     Process experiment-specific imaging preprocessing for a single subject.
        #     """
        #     try:
        #         self.logger.info(f"SID_IMG_PRE: ========== Starting SID imaging preprocessing for {subject_id} ==========")
        #         self.logger.info(f"SID_IMG_PRE: Suite2p path: {suite2p_path}")
        #         self.logger.info(f"SID_IMG_PRE: Output path: {output_path}")

        #         # Expected cache files
        #         results_file = os.path.join(output_path, 'sid_imaging_preprocessing_results.h5')
        #         imaging_data_file = os.path.join(output_path, 'sid_imaging_preprocess_data.pkl')
        #         behavioral_data_file = os.path.join(output_path, 'sid_behavioral_preprocess_data.pkl')

        #         # Get the matching behavioral session from subject config
        #         subject_config = self.config.get('subjects', {}).get(subject_id, {})
        #         imaging_sessions = subject_config.get('imaging_sessions', [])
        #         if not imaging_sessions:
        #             self.logger.error(f"SID_IMG_PRE: No imaging sessions found for subject {subject_id}")
        #             return {
        #                 'success': False,
        #                 'sessions_processed': 0,
        #                 'error_message': f'No imaging sessions found for subject {subject_id}'
        #             }

        #         imaging_session = imaging_sessions[0]
        #         behavior_session_id = imaging_session.get('behavior_session')
        #         if not behavior_session_id:
        #             self.logger.error(f"SID_IMG_PRE: No behavior_session specified for subject {subject_id} session {imaging_session.get('imaging_folder', '?')}")
        #             return {
        #                 'success': False,
        #                 'sessions_processed': 0,
        #                 'error_message': 'Missing behavior_session in config'
        #             }

        #         # If cache exists and not forcing, load and return
        #         if (os.path.exists(imaging_data_file) and os.path.exists(behavioral_data_file)) and not force:
        #             try:
        #                 with open(imaging_data_file, 'rb') as f:
        #                     imaging_data = pickle.load(f)
        #                 with open(behavioral_data_file, 'rb') as f:
        #                     behavioral_data = pickle.load(f)

        #                 self.logger.info(f"SID_IMG_PRE: Loaded cached imaging/behavioral data for {subject_id} (session {imaging_session.get('imaging_folder', '?')})")

        #                 return {
        #                     'success': True,
        #                     'sessions_processed': 0,  # no reprocessing
        #                     'imaging_data_summary': {
        #                         'n_rois': int(len(imaging_data.get('roi_labels', []))) if imaging_data.get('roi_labels') is not None else int(imaging_data['dff_traces'].shape[0]),
        #                         'n_frames': int(imaging_data['dff_traces'].shape[1]),
        #                         'imaging_duration': float(imaging_data.get('imaging_duration_session', imaging_data.get('imaging_duration', np.nan))),
        #                         'has_voltage': bool(imaging_data.get('vol_time') is not None),
        #                         'vol_duration': float(imaging_data.get('vol_duration_session', imaging_data.get('vol_duration', np.nan)))
        #                     },
        #                     'behavioral_data_summary': {
        #                         'n_trials': int(len(behavioral_data['df_trials'])),
        #                         'session_duration': float(behavioral_data['df_trials']['trial_start_timestamp'].max() - behavioral_data['df_trials']['trial_start_timestamp'].min()),
        #                         'behavior_session_id': behavior_session_id
        #                     },
        #                     'imaging_data': imaging_data,
        #                     'behavioral_data': behavioral_data
        #                 }
        #             except Exception as e:
        #                 self.logger.warning(f"SID_IMG_PRE: Failed loading cached data, reprocessing. Reason: {e}")

        #         # Missing either cache file or forced → load/process/save starting here
        #         self.logger.info("SID_IMG_PRE: Preparing new aligned imaging/behavioral data (force or cache missing)")

        #         # Load behavioral data
        #         behavioral_data = self.load_behavior_session(subject_id, behavior_session_id)
        #         if behavioral_data is None:
        #             return {
        #                 'success': False,
        #                 'sessions_processed': 0,
        #                 'error_message': 'Failed to load behavioral data'
        #             }

        #         # Load imaging data
        #         imaging_data = self.load_imaging_data(subject_id, suite2p_path, output_path)
        #         if imaging_data is None:
        #             return {
        #                 'success': False,
        #                 'sessions_processed': 0,
        #                 'error_message': 'Failed to load imaging data'
        #             }

        #         # Align timestamps (imaging_time, vol_time) and correct behavioral timestamps
        #         imaging_data, behavioral_data = self.align_imaging_behavioral_timestamps(imaging_data, behavioral_data)

        #         # Log voltage channel ranges
        #         for ch_name in ['vol_start', 'vol_stim_vis', 'vol_hifi', 'vol_img', 'vol_stim_aud', 'vol_flir', 'vol_pmt', 'vol_led']:
        #             ch_data = imaging_data.get(ch_name)
        #             if ch_data is not None:
        #                 try:
        #                     self.logger.info(f"SID_IMG_PRE: {ch_name} - min: {np.min(ch_data)}, max: {np.max(ch_data)}")
        #                 except Exception:
        #                     pass

        #         # Save aligned data
        #         try:
        #             with open(imaging_data_file, 'wb') as f:
        #                 pickle.dump(imaging_data, f)
        #             with open(behavioral_data_file, 'wb') as f:
        #                 pickle.dump(behavioral_data, f)
        #             self.logger.info(f"SID_IMG_PRE: Saved imaging/behavioral preprocess data to {output_path}")
        #         except Exception as e:
        #             self.logger.warning(f"SID_IMG_PRE: Failed to save preprocessed data: {e}")

        #         return {
        #             'success': True,
        #             'sessions_processed': 1,
        #             'imaging_data_summary': {
        #                 'n_rois': int(len(imaging_data.get('roi_labels', []))) if imaging_data.get('roi_labels') is not None else int(imaging_data['dff_traces'].shape[0]),
        #                 'n_frames': int(imaging_data['dff_traces'].shape[1]),
        #                 'imaging_duration': float(imaging_data.get('imaging_duration_session', imaging_data.get('imaging_duration', np.nan))),
        #                 'has_voltage': bool(imaging_data.get('vol_time') is not None),
        #                 'vol_duration': float(imaging_data.get('vol_duration_session', imaging_data.get('vol_duration', np.nan)))
        #             },
        #             'behavioral_data_summary': {
        #                 'n_trials': int(len(behavioral_data['df_trials'])),
        #                 'session_duration': float(behavioral_data['df_trials']['trial_start_timestamp'].max() - behavioral_data['df_trials']['trial_start_timestamp'].min()),
        #                 'behavior_session_id': behavior_session_id
        #             },
        #             'imaging_data': imaging_data,
        #             'behavioral_data': behavioral_data
        #         }

        #     except Exception as e:
        #         self.logger.error(f"SID_IMG_PRE: Processing failed for {subject_id}: {e}")
        #         return {
        #             'success': False,
        #             'sessions_processed': 0,
        #             'error_message': str(e)
        #         }



        # def process_subject(self, subject_id: str, suite2p_path: str, output_path: str, force: bool = False) -> Dict[str, Any]:
        #         """
        #         Process experiment-specific imaging preprocessing for a single subject.
                
        #         Args:
        #             subject_id: Subject identifier
        #             suite2p_path: Path to Suite2p output directory (plane0)
        #             output_path: Path for preprocessing output
        #             force: Force reprocessing even if output exists
                    
        #         Returns:
        #             Dictionary with processing results
        #         """
        #         try:                
        #             self.logger.info(f"SID_IMG_PRE: ========== Starting SID imaging preprocessing for {subject_id} ==========")
        #             self.logger.info(f"SID_IMG_PRE: Suite2p path: {suite2p_path}")
        #             self.logger.info(f"SID_IMG_PRE: Output path: {output_path}")
                    
        #             # Check if already processed
        #             results_file = os.path.join(output_path, 'sid_imaging_preprocessing_results.h5')
        #             imaging_data_file = os.path.join(output_path, 'sid_imaging_preprocess_data.pkl')
        #             behavioral_data_file = os.path.join(output_path, 'sid_behavioral_preprocess_data.pkl')                
                    
        #             imaging_behavioral_force = False                
                    
        #             # Get the matching behavioral session from subject config
        #             subject_config = self.config.get('subjects', {}).get(subject_id, {})
        #             imaging_sessions = subject_config.get('imaging_sessions', [])
                    
        #             if not imaging_sessions:
        #                 self.logger.error(f"SID_IMG_PRE: No imaging sessions found for subject {subject_id}")
        #                 return None
                    
        #             # For now, use the first imaging session
        #             imaging_session = imaging_sessions[0]
        #             behavior_session_id = imaging_session.get('behavior_session')
                    
        #             if not behavior_session_id:
        #                 self.logger.error(f"SID_IMG_PRE: No behavior_session specified for subject {subject_id} session {imaging_session['imaging_folder']}")
        #                 return None                     
                    
        #             # if (os.path.exists(results_file) and 
        #             #     os.path.exists(imaging_data_file) and 
        #             #     os.path.exists(behavioral_data_file) and not imaging_behavioral_force):
                    
        #             if ((not os.path.exists(imaging_data_file) or not os.path.exists(behavioral_data_file)) and not imaging_behavioral_force):                    
        #                     self.logger.warning(f"SID_IMG_PRE: No existing aligned imaging data, reprocessing.")
        #                     load_data = True                   

        #                 # # Load existing data
        #                 # try:
        #                 #     with open(imaging_data_file, 'rb') as f:
        #                 #         imaging_data = pickle.load(f)
                            
        #                 #     with open(behavioral_data_file, 'rb') as f:
        #                 #         behavioral_data = pickle.load(f)
                            
        #                 #     self.logger.info(f"SID_IMG_PRE: Successfully loaded existing data for {subject_id} session {imaging_session['imaging_folder']}")
        #             else:
        #                 self.logger.info(f"SID_IMG_PRE: SID imaging data already exists for {subject_id} session {imaging_session['imaging_folder']}, use existing data")
        #                 imaging_data = None
        #                 behavioral_data = None          
                    
        #             # Only do load/align if we don't have existing aligned data
        #             if imaging_data is None or behavioral_data is None:
        #                 self.logger.info(f"SID_IMG_PRE: Loading new data for {subject_id} session {imaging_session['imaging_folder']}")
        #                 # Load behavioral data
        #                 behavioral_data = self.load_behavior_session(subject_id, behavior_session_id)
        #                 if behavioral_data is None:
        #                     return {
        #                         'success': False,
        #                         'sessions_processed': 0,
        #                         'error_message': 'Failed to load behavioral data'
        #                     }                                
        #                 # Load imaging data
        #                 # imaging_data = self.load_imaging_data(subject_id, suite2p_path, output_path, behavioral_data)
        #                 imaging_data = self.load_imaging_data(subject_id, suite2p_path, output_path)
        #                 if imaging_data is None:
        #                     return {
        #                         'success': False,
        #                         'sessions_processed': 0,
        #                         'error_message': 'Failed to load imaging data'
        #                     }                    
                        
        #                  # NEW: Use alignment function
        #                 imaging_data, behavioral_data = self.align_imaging_behavioral_timestamps(imaging_data, behavioral_data)
                                
        #                 # voltage_data = imaging_data['v']
                        
        #                 # print min/max of each voltage channel
        #                 for ch_name in ['vol_start', 'vol_stim_vis', 'vol_hifi', 'vol_img', 
        #                                 'vol_stim_aud', 'vol_flir', 'vol_pmt', 'vol_led']:
        #                     ch_data = imaging_data.get(ch_name)
        #                     if ch_data is not None:
        #                         self.logger.info(f"SID_IMG_PRE: {ch_name} - min: {np.min(ch_data)}, max: {np.max(ch_data)}")
                        
        #                 # # get vol recording time and scope imaging vector
        #                 # vol_time = imaging_data.get('vol_time', None)
        #                 # vol_img = imaging_data.get('vol_img', None)
        #                 # vol_start = imaging_data.get('vol_start', None)

        #                 # # get scope imgaging signal trigger time stamps.
        #                 # imaging_time_array, _   = self.get_trigger_time(vol_time, vol_img)            
                                    
        #                 # # Adjust for photon integration cycle delay
        #                 # imaging_time_array = self.adjust_imaging_timing_for_integration_delay(imaging_time_array)
        #                 # n_frames = imaging_data['n_frames']
        #                 # imaging_data['imaging_duration'] = (imaging_time_array[-1] - imaging_time_array[0])
        #                 # imaging_data['imaging_fs'] = n_frames / imaging_data['imaging_duration']  # Recalculate fs based on timing
        #                 # self.logger.info(f"SID_IMG_PRE: Imaging sampling rate: {imaging_data['imaging_fs']} Hz")

        #                 # imaging_data['imaging_time'] = imaging_time_array
        #                 # self.logger.info(f"SID_IMG_PRE: Imaging duration: {imaging_data['imaging_duration']:.1f} seconds")
                                
        #                 # # Shift imaging and voltage time arrays to align to rising edge of session start pulse
        #                 # # get rising edge time of first start pulse
        #                 # first_start_pulse_time, _ = self.get_trigger_time(vol_time, vol_start)
        #                 # first_start_pulse_time = first_start_pulse_time[0]                    
                        
        #                 # # Shift voltage time to zero at first pulse
        #                 # imaging_data['vol_time'] = vol_time - first_start_pulse_time
        #                 # imaging_data['vol_duration'] = imaging_data['vol_time'][-1] - imaging_data['vol_time'][0]
        #                 # imaging_data['vol_duration_session'] = imaging_data['vol_time'][-1]

        #                 # # Shift imaging time to same reference point
        #                 # imaging_data['imaging_time'] = imaging_data['imaging_time'] - first_start_pulse_time                    
        #                 # imaging_data['imaging_duration_session'] = imaging_data['imaging_time'][-1]
                        

        #                 # self.logger.info(f"SID_IMG_PRE: Aligned time vectors to first vol_start rising edge at original time {first_start_pulse_time:.3f}s")
        #                 # self.logger.info(f"SID_IMG_PRE: Total imaging duration: {imaging_data['imaging_duration']:.1f}s")
        #                 # self.logger.info(f"SID_IMG_PRE: Total voltage duration: {imaging_data['vol_duration']:.1f}s")                    
        #                 # self.logger.info(f"SID_IMG_PRE: Aligned imaging duration: {imaging_data['imaging_duration_session']:.1f}s")
        #                 # self.logger.info(f"SID_IMG_PRE: Aligned voltage duration: {imaging_data['vol_duration_session']:.1f}s")
                        
        #                 # # After initial alignment, correct trial timestamps
        #                 # self.logger.info(f"SID_IMG_PRE: Correcting trial timestamps for {subject_id} session {imaging_session['imaging_folder']}")                    
        #                 # behavioral_data['df_trials'] = self.correct_trial_timestamps_with_voltage_pulses(
        #                 #     behavioral_data['df_trials'], imaging_data
        #                 # )
                        
                        
        #                 # Save the data for future use
        #                 try:
        #                     with open(imaging_data_file, 'wb') as f:
        #                         pickle.dump(imaging_data, f)
                            
        #                     with open(behavioral_data_file, 'wb') as f:
        #                         pickle.dump(behavioral_data, f)

        #                     self.logger.info(f"SID_IMG_PRE: Saved imaging and behavioral data for {subject_id} session {imaging_session['imaging_folder']}")

        #                 except Exception as e:
        #                     self.logger.warning(f"SID_IMG_PRE: Failed to save data files: {e}")

        #             return {
        #                 'success': True,
        #                 'sessions_processed': 1,
        #                 'imaging_data_summary': {
        #                     'n_rois': len(imaging_data['roi_labels']),
        #                     'n_frames': imaging_data['dff_traces'].shape[1],
        #                     'imaging_duration': imaging_data['imaging_duration_session'],
        #                     'has_voltage': imaging_data.get('vol_time') is not None,
        #                     'vol_duration': imaging_data['vol_duration_session']
        #                 },
        #                 'behavioral_data_summary': {
        #                     'n_trials': len(behavioral_data['df_trials']),
        #                     'session_duration': behavioral_data['df_trials']['trial_start_timestamp'].max() - behavioral_data['df_trials']['trial_start_timestamp'].min(),
        #                     'behavior_session_id': behavior_session_id
        #                 },
        #                 'imaging_data': imaging_data,  # Available for further analysis
        #                 'behavioral_data': behavioral_data  # Available for further analysis
        #             }
                    
        #         except Exception as e:
        #             self.logger.error(f"SID_IMG_PRE: Processing failed for {subject_id}: {e}")
        #             return {
        #                 'success': False,
        #                 'sessions_processed': 0,
        #                 'error_message': str(e)
        #             }



        except Exception as e:
            self.logger.error(f"SID_IMG_PRE: Processing failed for {subject_id}: {e}")
            return {'success': False, 'sessions_processed': 0, 'error_message': str(e)}






    def batch_process(self, force: bool = False) -> Dict[str, bool]:
        """
        Process SID imaging preprocessing for all subjects in the list.
        
        Args:
            force: Force reprocessing even if output exists
            
        Returns:
            Dictionary mapping subject_id to success status
        """
        results = {}
        
        self.logger.info(f"SID_IMG_PRE: Starting batch SID imaging processing for {len(self.subject_list)} subjects")
        
        for subject_id in self.subject_list:
            self.logger.info(f"SID_IMG_PRE: Processing {subject_id}...")
            results[subject_id] = self.process_subject(subject_id, force)
        
        
        
        # Log summary
        successful = sum(results.values())
        self.logger.info(f"SID_IMG_PRE: Batch processing complete: {successful}/{len(self.subject_list)} subjects successful")
        
        return results