"""
Single Interval Discrimination Imaging Analyzer

Experiment-specific imaging analysis for single interval discrimination experiments.
Minimal implementation that loads preprocessed data.
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

import modules.utils.utils as utils

# from modules.experiments.single_interval_discrimination.sid_roi import run_sid_insight_roi_analysis


from modules.experiments.single_interval_discrimination.roi_timing_analysis import (
    zscore_across_trials, scaling_vs_clock, time_resolved_decoding,
    hazard_unique_variance, prediction_error_at_F2, raster_sort_index,
    split_half_reliability, primary_timing_label, thin_contract
)

import warnings; warnings.filterwarnings("ignore", message=r".*groups.*ignored.*KFold.*", category=UserWarning, module=r"sklearn\.model_selection\._split")

class SingleIntervalDiscriminationImagingAnalyzer:
    """
    Imaging analyzer for single interval discrimination experiments.
    """
    
    def __init__(self, config_manager, subject_list, logger=None):
        """
        Initialize the SID imaging analyzer.
        
        Args:
            config_manager: Configuration manager instance
            subject_list: List of subject IDs to analyze
            logger: Logger instance (optional)
        """
        self.config_manager = config_manager
        self.subject_list = subject_list
        self.config = config_manager.config
        self.logger = logger or logging.getLogger(__name__)
        
        
        self.logger.info("SID_IMG_ANALYZER: Single interval discrimination imaging analyzer initialized")
    

    def _load_preprocessed_data(self, subject_id: str, suite2p_path: str, output_path: str) -> Dict[str, Any]:
        """
        Load preprocessed imaging and behavioral data for a subject.
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            Dictionary with loaded data or error information
        """
        try:
            self.logger.info(f"SID_IMG_ANALYZER: Loading preprocessed data for {subject_id}")
            
            # Get subject configuration
            subject_config = self.config.get('subjects', {}).get(subject_id, {})
            imaging_sessions = subject_config.get('imaging_sessions', [])
            
            if not imaging_sessions:
                return {
                    'success': False,
                    'error': f'No imaging sessions found for subject {subject_id}'
                }
            
            # For now, use first imaging session
            session = imaging_sessions[0]
            # output_path = session.get('output_path')
            
            if not output_path:
                return {
                    'success': False,
                    'error': f'Missing output_path for subject {subject_id}'
                }
            
            # Check if preprocessed data exists
            imaging_data_file = os.path.join(output_path, 'sid_imaging_preprocess_data.pkl')
            behavioral_data_file = os.path.join(output_path, 'sid_behavioral_preprocess_data.pkl')
            
            if not (os.path.exists(imaging_data_file) and os.path.exists(behavioral_data_file)):
                return {
                    'success': False,
                    'error': 'Preprocessed data not found. Run preprocessing first.',
                    'missing_files': {
                        'imaging_data': not os.path.exists(imaging_data_file),
                        'behavioral_data': not os.path.exists(behavioral_data_file)
                    }
                }
            
            # Load preprocessed data
            with open(imaging_data_file, 'rb') as f:
                imaging_data = pickle.load(f)
            
            with open(behavioral_data_file, 'rb') as f:
                behavioral_data = pickle.load(f)
            
            self.logger.info(f"SID_IMG_ANALYZER: Successfully loaded preprocessed data for {subject_id}")
            
            return {
                'success': True,
                'subject_id': subject_id,
                'imaging_data': imaging_data,
                'behavioral_data': behavioral_data,
                'output_path': output_path,
                'data_summary': {
                    'n_rois': imaging_data['dff_traces'].shape[0],
                    'n_frames': imaging_data['dff_traces'].shape[1],
                    'n_trials': len(behavioral_data['df_trials']),
                    'imaging_duration': imaging_data['imaging_time'][-1] - imaging_data['imaging_time'][0],
                    'has_voltage': 'vol_time' in imaging_data and imaging_data['vol_time'] is not None
                }
            }
            
        except Exception as e:
            self.logger.error(f"SID_IMG_ANALYZER: Failed to load preprocessed data for {subject_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            

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
                        
                        # Ensure we have at least 90% of the requested window duration
                        if actual_duration >= 0.9 * required_duration:
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
                self.logger.error(f"SID_IMG: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            # Handle both old array format and new DataFrame format
            if 'df_trials_with_segments' in trial_data:
                # New DataFrame format
                df_with_segments = trial_data['df_trials_with_segments']
                valid_mask = trial_data['valid_trials_mask']
                df_valid = df_with_segments[valid_mask]
                
                if len(df_valid) == 0:
                    self.logger.error("SID_IMG: No valid trials with segments")
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
            
            self.logger.info(f"SID_IMG: Generated trial segments check plot")
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create trial segments check plot: {e}")


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
                self.logger.error(f"SID_IMG: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            # Handle both old array format and new DataFrame format
            if 'df_trials_with_segments' in trial_data:
                # New DataFrame format
                df_with_segments = trial_data['df_trials_with_segments']
                valid_mask = trial_data['valid_trials_mask']
                df_valid = df_with_segments[valid_mask]
                
                if len(df_valid) == 0:
                    self.logger.error("SID_IMG: No valid trials with segments")
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
                
            self.logger.info(f"SID_IMG: Generated individual trial plots from segments for {n_trials_to_plot} trials")
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create individual trial plots from segments: {e}")


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
                self.logger.error(f"SID_IMG: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            # Handle both old array format and new DataFrame format
            if 'df_trials_with_segments' in trial_data:
                # New DataFrame format
                df_with_segments = trial_data['df_trials_with_segments']
                valid_mask = trial_data['valid_trials_mask']
                df_valid = df_with_segments[valid_mask]
                
                if len(df_valid) == 0:
                    self.logger.error("SID_IMG: No valid trials with segments")
                    return
                
                # Get common time vector
                dff_time_vector = df_valid.iloc[0]['dff_time_vector']
                
                # Define conditions
                left_rewarded = (df_valid['trial_side'] == 'left') & (df_valid['rewarded'] == 1)
                right_rewarded = (df_valid['trial_side'] == 'right') & (df_valid['rewarded'] == 1)
                
                if not np.any(left_rewarded) or not np.any(right_rewarded):
                    self.logger.warning("SID_IMG: Insufficient trials for condition comparison")
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
                self.logger.info(f"SID_IMG: Condition comparison heatmap - Left: {len(left_trials)} trials, Right: {len(right_trials)} trials")
            else:
                self.logger.info(f"SID_IMG: Condition comparison heatmap - Left: {np.sum(left_rewarded)} trials, Right: {np.sum(right_rewarded)} trials")
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create condition comparison heatmap: {e}")

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
                self.logger.error(f"SID_IMG: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            # Handle both old array format and new DataFrame format
            if 'df_trials_with_segments' in trial_data:
                # New DataFrame format
                df_with_segments = trial_data['df_trials_with_segments']
                valid_mask = trial_data['valid_trials_mask']
                df_valid = df_with_segments[valid_mask]
                
                if len(df_valid) == 0:
                    self.logger.error("SID_IMG: No valid trials with segments")
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
                
            self.logger.info(f"SID_IMG: Generated individual trial plots from segments for {n_trials_to_plot} trials")
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create individual trial plots from segments: {e}")


 

    def plot_responsive_rois_by_condition(self, trial_data: Dict[str, Any], subject_id: str) -> None:
        """
        Plot ROIs that are responsive to alignment, filtered by trial conditions.
        """
        try:
            
            
            if 'error' in trial_data:
                self.logger.error(f"SID_IMG: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            df_with_segments = trial_data['df_trials_with_segments']
            valid_mask = trial_data['valid_trials_mask']
            
            # Filter to valid trials only
            df_valid = df_with_segments[valid_mask].copy()
            
            if len(df_valid) == 0:
                self.logger.error("SID_IMG: No valid trials with segments")
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
                self.logger.warning("SID_IMG: Missing trial_side or rewarded columns")
                return
            
            left_rewarded = (df_valid['trial_side'] == 'left') & (df_valid['rewarded'] == 1)
            right_rewarded = (df_valid['trial_side'] == 'right') & (df_valid['rewarded'] == 1)
            
            if not np.any(left_rewarded) or not np.any(right_rewarded):
                self.logger.warning("SID_IMG: Insufficient trials for left+rewarded or right+rewarded conditions")
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
            self.logger.error(f"SID_IMG: Failed to plot responsive ROIs by condition: {e}")
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
                self.logger.error(f"SID_IMG: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            # Handle both old array format and new DataFrame format
            if 'df_trials_with_segments' in trial_data:
                # New DataFrame format
                df_with_segments = trial_data['df_trials_with_segments']
                valid_mask = trial_data['valid_trials_mask']
                df_valid = df_with_segments[valid_mask]
                
                if len(df_valid) == 0:
                    self.logger.error("SID_IMG: No valid trials with segments")
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
                    self.logger.warning(f"SID_IMG: No trials found for trial_type '{trial_type}'")
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
                    self.logger.warning(f"SID_IMG: No trials found for trial_type '{trial_type}'")
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
            
            self.logger.info(f"SID_IMG: All ROIs heatmap - {n_trials} trials, {n_rois} ROIs")
            self.logger.info(f"SID_IMG: ROI sorting: {sort_method}, Trial type: {trial_type}")
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create all ROIs heatmap: {e}")


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
                self.logger.error(f"SID_IMG: Cannot plot - trial extraction failed: {trial_data['error']}")
                return
            
            # Handle both old array format and new DataFrame format
            if 'df_trials_with_segments' in trial_data:
                # New DataFrame format
                df_with_segments = trial_data['df_trials_with_segments']
                valid_mask = trial_data['valid_trials_mask']
                df_valid = df_with_segments[valid_mask]
                
                if len(df_valid) == 0:
                    self.logger.error("SID_IMG: No valid trials with segments")
                    return
                
                if trial_index >= len(df_valid) or trial_index < 0:
                    self.logger.error(f"SID_IMG: Trial index {trial_index} out of range (0-{len(df_valid)-1})")
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
                    self.logger.error(f"SID_IMG: Trial index {trial_index} out of range (0-{len(dff_segments)-1})")
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
            self.logger.info(f"SID_IMG: Single trial raster plot - Trial {trial_index}, {n_rois} ROIs, Sort: {sort_method}")
            if sort_method != 'none':
                if sort_method == 'max_peak':
                    top_roi = roi_order[0]
                    self.logger.info(f"SID_IMG: Most active ROI: {top_roi} (peak: {np.max(dff_segment[top_roi, :]):.3f})")
                else:
                    bottom_roi = roi_order[0]
                    self.logger.info(f"SID_IMG: Least active ROI: {bottom_roi} (min: {np.min(dff_segment[bottom_roi, :]):.3f})")
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create single trial raster plot: {e}")



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
            
            self.logger.info(f"SID_IMG: Creating trial alignment diagnostic for {len(trial_indices)} trials")
            
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
                    self.logger.warning(f"SID_IMG: Trial index {trial_idx} out of range, skipping")
                    continue
                    
                # FIX: Use iloc to get the correct trial by position, not by index
                trial_row = df_trials.iloc[trial_idx]
                trial_start = trial_row.get('trial_start_timestamp', np.nan)
                
                if pd.isna(trial_start):
                    self.logger.warning(f"SID_IMG: Trial {trial_idx} has no trial_start_timestamp, skipping")
                    continue
                
                # Define window around trial
                window_start = trial_start - window_sec/2
                window_end = trial_start + window_sec/2
                
                # Extract imaging data for this window
                img_start_idx = np.searchsorted(imaging_data['imaging_time'], window_start)
                img_end_idx = np.searchsorted(imaging_data['imaging_time'], window_end)
                
                if img_start_idx >= img_end_idx:
                    self.logger.warning(f"SID_IMG: No imaging data for trial {trial_idx} window, skipping")
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
                self.logger.info(f"SID_IMG: Trial {trial_idx} (actual trial #{actual_trial_index}) diagnostic - {n_rois} ROIs, "
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
                    self.logger.warning(f"SID_IMG: Trial {trial_idx} potential alignment issues: {', '.join(alignment_issues)}")
                else:
                    self.logger.info(f"SID_IMG: Trial {trial_idx} alignment looks good")
            
            self.logger.info(f"SID_IMG: Trial alignment diagnostic completed for {len(trial_indices)} trials")
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create trial alignment diagnostic: {e}")



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
            
            self.logger.info(f"SID_IMG: Creating session alignment overview for first {time_window}s")
            
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
            
            self.logger.info(f"SID_IMG: Session overview shows {len(trials_in_window)} trials in {time_window}s window")
            
        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create session alignment overview: {e}")


    # NEW: adapter to the ROI timing contract
    def _segments_to_contract(self, trial_data: Dict[str, Any], alignment: str) -> Dict[str, Any]:
        """
        Convert extract_trial_segments_simple output into:
          X: (n_roi, n_time, n_trial), t: (n_time,), ISI: (n_trial,), F2_idx: (n_trial,)
        """
        try:
            if 'error' in trial_data:
                return {'success': False, 'error': trial_data['error']}

            if 'df_trials_with_segments' not in trial_data:
                return {'success': False, 'error': 'df_trials_with_segments missing'}

            dfw = trial_data['df_trials_with_segments']
            valid = np.asarray(trial_data.get('valid_trials_mask', np.ones(len(dfw), dtype=bool)))
            dfv = dfw[valid]
            if len(dfv) == 0:
                return {'success': False, 'error': 'no valid trials after segmentation'}

            # time vector (assumed identical after trimming)
            t = np.asarray(dfv.iloc[0]['dff_time_vector'], dtype=float)

            # stack segments → (n_trials, n_roi, n_time) → (n_roi, n_time, n_trials)
            segs = [np.asarray(row['dff_segment'], dtype=float) for _, row in dfv.iterrows()]
            segs = np.stack(segs, axis=0)
            X = np.transpose(segs, (1, 2, 0))

            # ISI (seconds) and F2 index
            s1 = np.asarray(dfv['start_flash_1'], dtype=float)
            s2 = np.asarray(dfv['start_flash_2'], dtype=float)
            ISI = s2 - s1

            if alignment == 'start_flash_1':
                f2_rel = ISI.copy()
            elif alignment == 'start_flash_2':
                f2_rel = np.zeros(len(dfv), dtype=float)
            else:
                a = np.asarray(dfv[alignment], dtype=float)
                f2_rel = s2 - a

            # convert relative F2 time to index in t
            F2_idx = np.clip(np.searchsorted(t, f2_rel, side='left'), 0, len(t) - 1).astype(int)
            # Also carry trial indices to align F1/F2 contracts later
            trial_indices = np.asarray(dfv['trial_index'].values, dtype=int)

            return {
                'success': True,
                'X': X, 't': t, 'ISI': ISI, 'F2_idx': F2_idx,
                'n_roi': X.shape[0], 'n_time': X.shape[1], 'n_trial': X.shape[2],
                'trial_indices': trial_indices,
            }
        except Exception as e:
            return {'success': False, 'error': f'contract adapter failed: {e}'}



    def _median_lick_time_rel_F1(self, df_trials: pd.DataFrame) -> Optional[float]:
        """
        Compute median lick or choice onset relative to F1 (seconds), fallback to choice_start if lick missing.
        """
        try:
            # Prefer lick_start if available; else choice_start
            if 'lick_start' in df_trials.columns:
                rel = df_trials['lick_start'] - df_trials['start_flash_1']
            elif 'choice_start' in df_trials.columns:
                rel = df_trials['choice_start'] - df_trials['start_flash_1']
            else:
                return None
            rel = rel.replace([np.inf, -np.inf], np.nan).astype(float)
            if rel.notna().sum() < 3:
                return None
            return float(rel.median())
        except Exception:
            return None

    def _compute_pe_points(self, X_f2: np.ndarray, t_f2: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
        """
        Per-trial population PE magnitude in [F2+win0, F2+win1] assuming F2 at 0 in F2-alignment.
        Returns y: shape (n_trial,)
        """
        n_roi, n_time, n_trial = X_f2.shape
        dt = np.median(np.diff(t_f2)) if len(t_f2) > 1 else 0.01
        w0 = max(0, int(np.floor(window[0] / max(dt, 1e-9))))
        w1 = min(n_time, int(np.ceil(window[1] / max(dt, 1e-9))))
        if w1 <= w0 + 1:
            return np.full(n_trial, np.nan)
        # population mean in window
        win = X_f2[:, w0:w1, :].mean(axis=0)      # (n_time_win, n_trial)
        return win.mean(axis=0)                    # (n_trial,)

    def plot_roi_timing_report(
        self,
        subject_id: str,
        output_dir: str,
        # F1-aligned contract + z-scored data for panels A/B/C/E
        c1: Dict[str, Any],
        Xz_f1: np.ndarray,
        sc_res: Any,
        neu: Any,
        haz: Any,
        order: np.ndarray,
        # F2-aligned contract + z-scored data for panel D
        c2: Optional[Dict[str, Any]] = None,
        Xz_f2: Optional[np.ndarray] = None,
        behavioral_df: Optional[pd.DataFrame] = None,
        pe_res: Optional[Any] = None,
        pe_window: Tuple[float, float] = (0.0, 0.25),
    ) -> str:
        """
        Create the one-page figure per session summarizing ROI timing analyses.
        Returns the saved figure path.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, f"{subject_id}_sid_roi_timing_report.png")

        # Panel A: ΔR² histogram and fractions
        delta = getattr(sc_res, 'delta_R2', None)
        alpha_opt = getattr(sc_res, 'alpha_opt', None)
        if isinstance(alpha_opt, (list, np.ndarray)) and np.ndim(alpha_opt) == 1:
            alpha_med = float(np.nanmedian(alpha_opt))
            # simple bootstrap CI
            boot = []
            rng = np.random.RandomState(0)
            a_clean = np.asarray(alpha_opt, float)
            a_clean = a_clean[np.isfinite(a_clean)]
            for _ in range(200):
                if a_clean.size == 0:
                    break
                boot.append(np.nanmedian(rng.choice(a_clean, size=a_clean.size, replace=True)))
            if boot:
                lo, hi = np.percentile(boot, [2.5, 97.5])
            else:
                lo = hi = np.nan
        else:
            try:
                alpha_val = float(alpha_opt)
                alpha_med = alpha_val if np.isfinite(alpha_val) else np.nan
            except Exception:
                alpha_med = np.nan
            lo, hi = np.nan, np.nan

        # Fractions
        if delta is not None:
            delta = np.asarray(delta, float)
            frac_scal = float(np.nanmean(delta > 0))
            frac_clk = float(np.nanmean(delta < 0))
            frac_mix = max(0.0, 1.0 - (frac_scal + frac_clk))
        else:
            frac_scal = frac_clk = frac_mix = np.nan

        # Panel B: Neurometric decoding
        times = getattr(neu, 'times', None)
        auc = getattr(neu, 'auc_binary', None)
        div_t = getattr(neu, 'divergence_time_s', None)

        # Median lick time relative to F1
        lick_t = self._median_lick_time_rel_F1(behavioral_df) if behavioral_df is not None else None

        # # Panel C: Hazard unique variance
        # pR2_haz = getattr(haz, 'partial_r2_hazard', None)
        # r2_full = getattr(haz, 'r2_full', None)
        # r2_red = getattr(haz, 'r2_reduced_time', None)
        # # time unique variance if available
        # pR2_time = None
        # if r2_full is not None and r2_red is not None and np.all(np.isfinite(r2_full)) and np.all(np.isfinite(r2_red)):
        #     pR2_time = float(r2_red)  # reduced model with time only; display alongside hazard

        # Panel C: Hazard unique variance
        # Coerce outputs to scalars (median if arrays) to avoid ambiguous truth/shape issues
        raw_pR2_haz = getattr(haz, 'partial_r2_hazard', None)
        raw_r2_full = getattr(haz, 'r2_full', None)
        raw_r2_time_only = getattr(haz, 'r2_reduced_time', None)

        def _median_scalar(x):
            if x is None:
                return None
            try:
                arr = np.asarray(x, dtype=float)
                if arr.ndim == 0:
                    return float(arr)
                if arr.size == 0:
                    return None
                return float(np.nanmedian(arr))
            except Exception:
                try:
                    return float(x)
                except Exception:
                    return None

        pR2_haz = _median_scalar(raw_pR2_haz)              # unique variance for hazard (already partial R²)
        pR2_time = _median_scalar(raw_r2_time_only)        # time-only R² (display alongside hazard)


        # Panel D: PE scatter
        pe_points = None
        surprise = None
        pe_slope = getattr(pe_res, 'slope', None) if pe_res is not None else None
        pe_pval = getattr(pe_res, 'pval', None) if pe_res is not None else None

        if (c2 is not None) and (Xz_f2 is not None):
            # Align trials by trial_index between F1 and F2 contracts
            idx_f1 = np.asarray(c1.get('trial_indices', []), dtype=int)
            idx_f2 = np.asarray(c2.get('trial_indices', []), dtype=int)
            if idx_f1.size and idx_f2.size:
                common, f1_pos, f2_pos = np.intersect1d(idx_f1, idx_f2, assume_unique=False, return_indices=True)
                if common.size >= 5:
                    # Subset tensors to the common trials
                    Xz_f2_aligned = Xz_f2[:, :, f2_pos]
                    # population PE per trial vs |ISI - mean(ISI)| (from F1 contract, aligned)
                    pe_points = self._compute_pe_points(Xz_f2_aligned, c2['t'], pe_window)
                    ISI_aligned = c1['ISI'][f1_pos]
                    surprise = np.abs(ISI_aligned - np.nanmean(ISI_aligned))
            # Fallback (no index columns): use original behavior (may be slightly misaligned)
            if pe_points is None:
                pe_points = self._compute_pe_points(Xz_f2, c2['t'], pe_window)
                surprise = np.abs(c1['ISI'] - np.nanmean(c1['ISI']))

        # Panel E: Raster sorted by winning story (pre-F2 common window)
        kmax = int(np.nanmin(c1['F2_idx']))
        t_pre = c1['t'][:kmax]
        mean_pre = np.nanmean(Xz_f1[:, :kmax, :], axis=2)  # (n_roi, kmax)
        if order is None or len(order) == 0:
            order = np.arange(mean_pre.shape[0])
        mean_pre_sorted = mean_pre[order, :]

        # Build figure
        fig = plt.figure(figsize=(12, 10), dpi=150)
        gs = gridspec.GridSpec(3, 3, height_ratios=[1.1, 1.1, 1.4], width_ratios=[1, 1, 1], wspace=0.35, hspace=0.5)
        fig.suptitle(f"SID ROI Timing Report — {subject_id}", y=0.995, fontsize=14)

        # A: ΔR² histogram + fractions + alpha
        ax = fig.add_subplot(gs[0, 0])
        if delta is not None:
            dclean = delta[np.isfinite(delta)]
            ax.hist(dclean, bins=40, color='gray', alpha=0.9)
            ax.axvline(0, color='k', ls='--', lw=1)
        ax.set_title(
            f"Scaling vs Clock (ΔR²)\n"
            f"frac S={frac_scal:.2f}, C={frac_clk:.2f}, mixed={frac_mix:.2f} | N={dclean.size}"
        )
        ax.set_xlabel("ΔR² (scale − clock)"); ax.set_ylabel("Count")
        # alpha annotation
        if np.isfinite(alpha_med):
            txt = f"median α={alpha_med:.2f}"
            if np.isfinite(lo) and np.isfinite(hi):
                txt += f" [{lo:.2f},{hi:.2f}]"
            ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

        # B: Neurometric timecourse (AUC) + divergence + lick marker
        ax = fig.add_subplot(gs[0, 1])
        if (times is not None) and (auc is not None):
            ax.plot(times, auc, color='C3', lw=2, label='AUC(short vs long)')
            ax.axhline(0.5, color='k', ls=':', lw=1)
            if div_t is not None:
                ax.axvline(div_t, color='C3', ls='--', lw=1.5, label=f"div {div_t*1000:.0f} ms")
            if lick_t is not None and np.isfinite(lick_t):
                ax.axvline(lick_t, color='C0', ls='-.', lw=1.2, label=f"median lick {lick_t*1000:.0f} ms")
            ax.legend(frameon=False, fontsize=8)
        ax.set_title("Neurometric (binary) over time")
        ax.set_xlabel("Time from F1 (s)"); ax.set_ylabel("AUC")

        # # C: Hazard unique variance bar(s)
        # ax = fig.add_subplot(gs[0, 2])
        # bars = []
        # labels = []
        # colors = []
        # if pR2_time is not None:
        #     bars.append(pR2_time); labels.append("Time-only R²"); colors.append('C1')
        # if pR2_haz is not None:
        #     bars.append(pR2_haz); labels.append("Hazard (unique)"); colors.append('C0')
        # if bars:
        #     ax.bar(np.arange(len(bars)), bars, color=colors, width=0.6)
        #     ax.set_xticks(np.arange(len(bars))); ax.set_xticklabels(labels, rotation=20)
        #     ymax = max(0.25, np.nanmax(bars) * 1.4)
        #     ax.set_ylim(0, ymax)
        # ax.set_title("Pre-F2 variance components")

       # Panel C: bars
        ax = fig.add_subplot(gs[0, 2])
        bars = []
        xticklabels = []
        colors = []
        if pR2_time is not None and np.isfinite(pR2_time):
            bars.append(pR2_time); xticklabels.append("Time-only R²"); colors.append('C1')
        if pR2_haz is not None and np.isfinite(pR2_haz):
            bars.append(pR2_haz); xticklabels.append("Hazard (unique)"); colors.append('C0')

        if bars:
            ax.bar(np.arange(len(bars)), bars, color=colors, width=0.6)
            ax.set_xticks(np.arange(len(bars))); ax.set_xticklabels(xticklabels, rotation=20)
            ymax = max(0.25, float(np.nanmax(bars)) * 1.4)
            ax.set_ylim(0, ymax)
        ax.set_title("Pre-F2 variance components")

        # D: Prediction error at F2
        ax = fig.add_subplot(gs[1, 0])
        if (pe_points is not None) and (surprise is not None):
            ok = np.isfinite(pe_points) & np.isfinite(surprise)
            ax.scatter(surprise[ok], pe_points[ok], s=16, alpha=0.6, color='C2')
            # fit line for display only
            if ok.sum() > 3:
                m = np.polyfit(surprise[ok], pe_points[ok], 1)
                xs = np.linspace(np.nanmin(surprise[ok]), np.nanmax(surprise[ok]), 50)
                ax.plot(xs, m[0]*xs + m[1], color='C2', lw=2)
                
                
        # Annotate stats
        def _scalar_or_stat(x, reducer=np.nanmedian):
            if x is None:
                return None
            try:
                arr = np.asarray(x, dtype=float)
                if arr.ndim == 0:
                    val = float(arr)
                else:
                    if arr.size == 0:
                        return None
                    val = float(reducer(arr))
                return val if np.isfinite(val) else None
            except Exception:
                try:
                    val = float(x)
                    return val if np.isfinite(val) else None
                except Exception:
                    return None

        ps = _scalar_or_stat(pe_slope)
        pp = _scalar_or_stat(pe_pval)                
                
            
        lab = []
        if pe_slope is not None: lab.append(f"slope={ps:.3g}")
        if pe_pval is not None: lab.append(f"p={pp:.3g}")
        if lab:
            ax.text(0.98, 0.95, ", ".join(lab), transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
        ax.set_xlabel("|ISI − mean(ISI)| (s)"); ax.set_ylabel("F2 response (pop)")
        ax.set_title("Prediction error @F2")

        # E: Raster sorted by winning story (pre-F2)
        ax = fig.add_subplot(gs[1:, 1:])
        im = ax.imshow(
            mean_pre_sorted, aspect='auto',
            extent=[t_pre[0], t_pre[-1], 0, mean_pre_sorted.shape[0]],
            cmap='RdBu_r',
            vmin=-np.nanstd(mean_pre), vmax=np.nanstd(mean_pre)
        )
        ax.set_xlabel("Time from F1 (s)"); ax.set_ylabel("ROIs (sorted)")
        ax.set_title("Pre-F2 raster sorted by model verdict")
        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label("z(ΔF/F)")

        fig.savefig(fig_path, bbox_inches='tight')
        # plt.close(fig)
        return fig_path
 

    def filter_trial_data_by_isi(
        self,
        trial_data: Dict[str, Any],
        drop_isi_values: List[float],
        atol: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Filter segmented trial_data by removing trials whose ISI equals any value in drop_isi_values.
        ISI is computed per trial as (start_flash_2 - start_flash_1), which is robust to alignment.

        Args:
            trial_data: dict returned by extract_trial_segments_simple
            drop_isi_values: list of ISI values (in seconds) to drop
            atol: absolute tolerance for matching ISI (np.isclose)

        Returns:
            New trial_data dict with rows removed and arrays/masks updated.
        """
        try:
            if not drop_isi_values:
                return trial_data

            if 'df_trials_with_segments' not in trial_data or 'valid_trials_mask' not in trial_data:
                return {'error': 'filter_trial_data_by_isi: incompatible trial_data (missing df or mask)'}

            dfw = trial_data['df_trials_with_segments']
            valid_mask = np.asarray(trial_data['valid_trials_mask'], dtype=bool)

            # Work within the valid subset (this matches the order of dff_segments_array)
            df_valid = dfw[valid_mask].copy()
            if df_valid.empty:
                return {'error': 'filter_trial_data_by_isi: no valid trials to filter'}

            # Compute ISI per valid trial (prefer DataFrame 'isi' column)
            if 'isi' in df_valid.columns:
                isi = np.asarray(df_valid['isi'], dtype=float)
            else:
                s1 = np.asarray(df_valid['end_flash_1'], dtype=float)
                s2 = np.asarray(df_valid['start_flash_2'], dtype=float)
                isi = s2 - s1  # seconds
                print('ISI column not found')

            # Build drop mask within valid trials
            drop_vals = np.asarray(drop_isi_values, dtype=float).reshape(1, -1)
            is_drop = np.any(np.isclose(isi.reshape(-1, 1), drop_vals, atol=atol, rtol=0.0), axis=1)
            keep_valid_mask = ~is_drop

            # If nothing to drop, return original
            if not np.any(is_drop):
                return trial_data

            # Map kept valid trials back to full DataFrame index
            orig_valid_indices = df_valid.index.to_numpy()
            keep_indices_full = orig_valid_indices[keep_valid_mask]

            # Build new valid mask across all trials
            new_valid_mask = np.zeros(len(dfw), dtype=bool)
            new_valid_mask[keep_indices_full] = True

            # Slice dff_segments_array to keep only the same kept valid trials (order preserved)
            dff_arr = trial_data.get('dff_segments_array', None)
            if dff_arr is not None:
                dff_arr_np = np.asarray(dff_arr)
                if dff_arr_np.shape[0] == df_valid.shape[0]:
                    dff_arr_np = dff_arr_np[keep_valid_mask, ...]
                else:
                    # Shape mismatch; fall back to reconstructing from DataFrame rows
                    dff_arr_np = np.array([row['dff_segment'] for _, row in dfw[new_valid_mask].iterrows()])
            else:
                # Rebuild from DataFrame if needed
                dff_arr_np = np.array([row['dff_segment'] for _, row in dfw[new_valid_mask].iterrows()])

            # Update dropped_trials log
            dropped_existing = list(trial_data.get('dropped_trials', []))
            dropped_new = []
            for idx_in_valid, drop_flag in enumerate(is_drop):
                if drop_flag:
                    orig_idx = int(orig_valid_indices[idx_in_valid])
                    dropped_new.append((orig_idx, f'filtered_by_isi_{isi[idx_in_valid]:.6f}s'))
            dropped_all = dropped_existing + dropped_new

            # Return a new trial_data dict with updates
            new_trial_data = dict(trial_data)  # shallow copy
            new_trial_data['valid_trials_mask'] = new_valid_mask
            new_trial_data['n_valid_trials'] = int(np.sum(new_valid_mask))
            new_trial_data['dff_segments_array'] = dff_arr_np
            new_trial_data['dropped_trials'] = dropped_all

            return new_trial_data

        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"SID_IMG: filter_trial_data_by_isi failed: {e}")
            return {'error': f'filter_trial_data_by_isi failed: {e}'}



    def plot_trial_segments_spike_trial_check(self, trial_data: Dict[str, Any], subject_id: str,
                                sort_method: str = 'peak_time') -> None:
        """
        Plot segmented data for a quick alignment check with DFF and spike comparisons.
        Shows trial-by-time rasters instead of ROI-by-time.

        Layout:
        - Top-left: mean DFF (left vs right)
        - Bottom-left: mean spikes (left vs right)
        - Top-right: DFF trial raster (trial x time), mean across ROIs per trial
        - Bottom-right: Spike trial raster (trial x time), mean across ROIs per trial

        Notes:
        - Time axes are aligned across all panels.
        - If 'trial_side' is missing, plots show 'All trials' as a single curve.
        - If spikes are missing, the spike panels will display a message.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            if 'error' in trial_data:
                self.logger.error(f"SID_IMG: Cannot plot - trial extraction failed: {trial_data['error']}")
                return

            if 'df_trials_with_segments' not in trial_data:
                self.logger.error("SID_IMG: Expected DataFrame-based segmented trial_data")
                return

            df_with_segments = trial_data['df_trials_with_segments']
            valid_mask = np.asarray(trial_data.get('valid_trials_mask', np.ones(len(df_with_segments), dtype=bool)))
            df_valid = df_with_segments[valid_mask]

            if len(df_valid) == 0:
                self.logger.error("SID_IMG: No valid trials with segments")
                return

            # Extract arrays (n_trials, n_rois, n_time)
            dff_segments = np.array([np.asarray(row['dff_segment'], dtype=float) for _, row in df_valid.iterrows()])
            spike_available = 'spike_segment' in df_valid.columns and isinstance(df_valid.iloc[0]['spike_segment'], np.ndarray)
            if spike_available:
                spk_segments = np.array([np.asarray(row['spike_segment'], dtype=float) for _, row in df_valid.iterrows()])
            else:
                spk_segments = None

            # Common time vector
            t = np.asarray(df_valid.iloc[0]['dff_time_vector'], dtype=float)

            # Trial-side masks
            has_side = 'trial_side' in df_valid.columns
            if has_side:
                left_trials = (df_valid['trial_side'] == 'left').values
                right_trials = (df_valid['trial_side'] == 'right').values
                any_left = np.any(left_trials)
                any_right = np.any(right_trials)
            else:
                left_trials = np.ones(len(df_valid), dtype=bool)
                right_trials = np.zeros(len(df_valid), dtype=bool)
                any_left = True
                any_right = False

            # Compute mean DFF/spikes across trials and ROIs for left/right curves
            def _mean_pop_trace(seg, mask):
                if seg is None or not np.any(mask):
                    return None
                # seg: (n_trials, n_rois, n_time) -> mean over trials+ROIs -> (n_time,)
                return np.mean(seg[mask, :, :], axis=(0, 1))

            mean_dff_left = _mean_pop_trace(dff_segments, left_trials)
            mean_dff_right = _mean_pop_trace(dff_segments, right_trials)
            mean_spk_left = _mean_pop_trace(spk_segments, left_trials) if spike_available else None
            mean_spk_right = _mean_pop_trace(spk_segments, right_trials) if spike_available else None

            # Trial x time mean across ROIs (for rasters)
            dff_trial_mean = np.mean(dff_segments, axis=1)  # (n_trials, n_time)
            spk_trial_mean = np.mean(spk_segments, axis=1) if spike_available else None  # (n_trials, n_time)

            # Trial sorting based on DFF response characteristics
            def _get_trial_sort_order(traces: np.ndarray, time_vector: np.ndarray, method: str) -> np.ndarray:
                n_trials = traces.shape[0]
                if method == 'peak_time':
                    response_mask = time_vector >= 0
                    peak_times = []
                    for tr in range(n_trials):
                        resp = traces[tr, response_mask]
                        if resp.size > 0:
                            idx = np.argmax(np.abs(resp))
                            peak_times.append(time_vector[response_mask][idx])
                        else:
                            peak_times.append(np.inf)
                    return np.argsort(peak_times)
                elif method == 'peak_magnitude':
                    response_mask = time_vector >= 0
                    mags = []
                    for tr in range(n_trials):
                        resp = traces[tr, response_mask]
                        mags.append(np.max(np.abs(resp)) if resp.size > 0 else 0.0)
                    return np.argsort(mags)[::-1]
                elif method == 'response_onset':
                    baseline_mask = time_vector < 0
                    response_mask = time_vector >= 0
                    onsets = []
                    for tr in range(n_trials):
                        base_std = np.std(traces[tr, baseline_mask]) if np.any(baseline_mask) else 0.0
                        thr = 2.0 * base_std
                        resp = traces[tr, response_mask]
                        idxs = np.where(np.abs(resp) > thr)[0]
                        if idxs.size > 0:
                            onsets.append(time_vector[response_mask][idxs[0]])
                        else:
                            onsets.append(np.inf)
                    return np.argsort(onsets)
                elif method == 'trial_side':
                    # Sort by trial side (left first, then right)
                    if has_side:
                        sort_key = []
                        for tr in range(n_trials):
                            if left_trials[tr]:
                                sort_key.append(0)  # Left trials first
                            elif right_trials[tr]:
                                sort_key.append(1)  # Right trials second
                            else:
                                sort_key.append(2)  # Other trials last
                        return np.argsort(sort_key)
                    else:
                        return np.arange(n_trials)
                else:
                    return np.arange(n_trials)

            trial_order = _get_trial_sort_order(dff_trial_mean, t, sort_method)
            dff_sorted = dff_trial_mean[trial_order, :]
            spk_sorted = spk_trial_mean[trial_order, :] if spike_available else None

            # Figure: 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            alignment = trial_data.get('alignment_point', 'alignment')
            fig.suptitle(f'Trial Segments Check (Trial Rasters) — {subject_id} — {alignment}', fontsize=14)

            # Top-left: mean DFF left/right
            ax = axes[0, 0]
            if mean_dff_left is not None:
                ax.plot(t, mean_dff_left, color='g', lw=2, label=f'Left (n={int(np.sum(left_trials))})')
            if mean_dff_right is not None and any_right:
                ax.plot(t, mean_dff_right, color='r', lw=2, label=f'Right (n={int(np.sum(right_trials))})')
            if mean_dff_left is None and mean_dff_right is None:
                ax.text(0.5, 0.5, 'No trial_side data', ha='center', va='center', transform=ax.transAxes)
            ax.axvline(0, color='k', ls='--', alpha=0.7)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Mean DFF')
            ax.set_title('Population DFF (left vs right)')
            if (any_left or any_right):
                ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)

            # Bottom-left: mean spikes left/right
            ax = axes[1, 0]
            if spike_available:
                if mean_spk_left is not None:
                    ax.plot(t, mean_spk_left, color='g', lw=2, label=f'Left (n={int(np.sum(left_trials))})')
                if mean_spk_right is not None and any_right:
                    ax.plot(t, mean_spk_right, color='r', lw=2, label=f'Right (n={int(np.sum(right_trials))})')
                ax.axvline(0, color='k', ls='--', alpha=0.7)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Mean spikes')
                ax.set_title('Population spikes (left vs right)')
                if (any_left or any_right):
                    ax.legend(frameon=False)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No spike data', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()

            # Top-right: DFF trial raster (sorted)
            ax = axes[0, 1]
            vmin = np.percentile(dff_sorted, 5)
            vmax = np.percentile(dff_sorted, 95)
            im = ax.imshow(dff_sorted, aspect='auto', cmap='viridis',
                        extent=[t[0], t[-1], 0, dff_sorted.shape[0]],
                        vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.axvline(0, color='white', ls='--', lw=2, alpha=0.9)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Trials (sorted)')
            ax.set_title(f'DFF trial raster (sorted by {sort_method})')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Mean DF/F')

            # Add trial side color coding on y-axis if available
            if has_side and sort_method != 'trial_side':
                # Add colored markers to indicate trial side
                for i, trial_idx in enumerate(trial_order):
                    if left_trials[trial_idx]:
                        ax.scatter(t[0], i, marker='|', s=50, c='green', alpha=0.7)
                    elif right_trials[trial_idx]:
                        ax.scatter(t[0], i, marker='|', s=50, c='red', alpha=0.7)

            # Bottom-right: Spike trial raster (same order)
            ax = axes[1, 1]
            if spike_available and spk_sorted is not None:
                vmin_s = np.percentile(spk_sorted, 5)
                vmax_s = np.percentile(spk_sorted, 95)
                im2 = ax.imshow(spk_sorted, aspect='auto', cmap='viridis',
                                extent=[t[0], t[-1], 0, spk_sorted.shape[0]],
                                vmin=vmin_s, vmax=vmax_s, interpolation='nearest')
                ax.axvline(0, color='white', ls='--', lw=2, alpha=0.9)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Trials (same order)')
                ax.set_title('Spike trial raster (matched order)')
                plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='Mean Spikes')
                
                # Add trial side color coding on y-axis if available
                if has_side and sort_method != 'trial_side':
                    for i, trial_idx in enumerate(trial_order):
                        if left_trials[trial_idx]:
                            ax.scatter(t[0], i, marker='|', s=50, c='green', alpha=0.7)
                        elif right_trials[trial_idx]:
                            ax.scatter(t[0], i, marker='|', s=50, c='red', alpha=0.7)
            else:
                ax.text(0.5, 0.5, 'No spike data', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()

            # Ensure all x-axes share same limits for alignment
            xlim = (t[0], t[-1])
            for ax in axes.ravel():
                if hasattr(ax, 'set_xlim'):
                    ax.set_xlim(xlim)

            # Add legend for trial side markers if present
            if has_side and sort_method != 'trial_side':
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='|', color='w', markerfacecolor='green', markersize=10, label='Left trials'),
                    Line2D([0], [0], marker='|', color='w', markerfacecolor='red', markersize=10, label='Right trials')
                ]
                fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

            plt.tight_layout()
            plt.show()
            self.logger.info("SID_IMG: Generated DFF+spike trial raster check plot")

        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create trial raster check plot: {e}")

 

    def plot_trial_segments_spike_check(self, trial_data: Dict[str, Any], subject_id: str,
                                  sort_method: str = 'peak_time') -> None:
        """
        Plot segmented data for a quick alignment check with DFF and spike comparisons.

        Layout:
          - Top-left: mean DFF (left vs right)
          - Bottom-left: mean spikes (left vs right)
          - Top-right: DFF raster (ROI x time), sorted by DFF criterion
          - Bottom-right: Spike raster (ROI x time), same ROI order as DFF

        Notes:
          - Time axes are aligned across all panels.
          - If 'trial_side' is missing, plots show 'All trials' as a single curve.
          - If spikes are missing, the spike panels will display a message.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            if 'error' in trial_data:
                self.logger.error(f"SID_IMG: Cannot plot - trial extraction failed: {trial_data['error']}")
                return

            if 'df_trials_with_segments' not in trial_data:
                self.logger.error("SID_IMG: Expected DataFrame-based segmented trial_data")
                return

            df_with_segments = trial_data['df_trials_with_segments']
            valid_mask = np.asarray(trial_data.get('valid_trials_mask', np.ones(len(df_with_segments), dtype=bool)))
            df_valid = df_with_segments[valid_mask]

            if len(df_valid) == 0:
                self.logger.error("SID_IMG: No valid trials with segments")
                return

            # Extract arrays (n_trials, n_rois, n_time)
            dff_segments = np.array([np.asarray(row['dff_segment'], dtype=float) for _, row in df_valid.iterrows()])
            spike_available = 'spike_segment' in df_valid.columns and isinstance(df_valid.iloc[0]['spike_segment'], np.ndarray)
            if spike_available:
                spk_segments = np.array([np.asarray(row['spike_segment'], dtype=float) for _, row in df_valid.iterrows()])
            else:
                spk_segments = None

            # Common time vector
            t = np.asarray(df_valid.iloc[0]['dff_time_vector'], dtype=float)

            # Trial-side masks
            has_side = 'trial_side' in df_valid.columns
            if has_side:
                left_trials = (df_valid['trial_side'] == 'left').values
                right_trials = (df_valid['trial_side'] == 'right').values
                any_left = np.any(left_trials)
                any_right = np.any(right_trials)
            else:
                left_trials = np.ones(len(df_valid), dtype=bool)
                right_trials = np.zeros(len(df_valid), dtype=bool)
                any_left = True
                any_right = False

            # Compute mean DFF/spikes across trials and ROIs for left/right curves
            def _mean_pop_trace(seg, mask):
                if seg is None or not np.any(mask):
                    return None
                # seg: (n_trials, n_rois, n_time) -> mean over trials+ROIs -> (n_time,)
                return np.mean(seg[mask, :, :], axis=(0, 1))

            mean_dff_left = _mean_pop_trace(dff_segments, left_trials)
            mean_dff_right = _mean_pop_trace(dff_segments, right_trials)
            mean_spk_left = _mean_pop_trace(spk_segments, left_trials) if spike_available else None
            mean_spk_right = _mean_pop_trace(spk_segments, right_trials) if spike_available else None

            # ROI x time mean across trials (for rasters)
            dff_roi_mean = np.mean(dff_segments, axis=0)  # (n_rois, n_time)
            spk_roi_mean = np.mean(spk_segments, axis=0) if spike_available else None

            # ROI sorting from DFF only, then apply to spikes
            def _get_roi_sort_order_single(traces: np.ndarray, time_vector: np.ndarray, method: str) -> np.ndarray:
                n_rois = traces.shape[0]
                if method == 'peak_time':
                    response_mask = time_vector >= 0
                    peak_times = []
                    for r in range(n_rois):
                        resp = traces[r, response_mask]
                        if resp.size > 0:
                            idx = np.argmax(np.abs(resp))
                            peak_times.append(time_vector[response_mask][idx])
                        else:
                            peak_times.append(np.inf)
                    return np.argsort(peak_times)
                elif method == 'peak_magnitude':
                    response_mask = time_vector >= 0
                    mags = []
                    for r in range(n_rois):
                        resp = traces[r, response_mask]
                        mags.append(np.max(np.abs(resp)) if resp.size > 0 else 0.0)
                    return np.argsort(mags)[::-1]
                elif method == 'response_onset':
                    baseline_mask = time_vector < 0
                    response_mask = time_vector >= 0
                    onsets = []
                    for r in range(n_rois):
                        base_std = np.std(traces[r, baseline_mask]) if np.any(baseline_mask) else 0.0
                        thr = 2.0 * base_std
                        resp = traces[r, response_mask]
                        idxs = np.where(np.abs(resp) > thr)[0]
                        if idxs.size > 0:
                            onsets.append(time_vector[response_mask][idxs[0]])
                        else:
                            onsets.append(np.inf)
                    return np.argsort(onsets)
                else:
                    return np.arange(traces.shape[0])

            roi_order = _get_roi_sort_order_single(dff_roi_mean, t, sort_method)
            dff_sorted = dff_roi_mean[roi_order, :]
            spk_sorted = spk_roi_mean[roi_order, :] if spike_available else None

            # Figure: 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            alignment = trial_data.get('alignment_point', 'alignment')
            fig.suptitle(f'Trial Segments Check — {subject_id} — {alignment}', fontsize=14)

            # Top-left: mean DFF left/right
            ax = axes[0, 0]
            if mean_dff_left is not None:
                ax.plot(t, mean_dff_left, color='g', lw=2, label=f'Left (n={int(np.sum(left_trials))})')
            if mean_dff_right is not None and any_right:
                ax.plot(t, mean_dff_right, color='r', lw=2, label=f'Right (n={int(np.sum(right_trials))})')
            if mean_dff_left is None and mean_dff_right is None:
                ax.text(0.5, 0.5, 'No trial_side data', ha='center', va='center', transform=ax.transAxes)
            ax.axvline(0, color='k', ls='--', alpha=0.7)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Mean DFF')
            ax.set_title('Population DFF (left vs right)')
            if (any_left or any_right):
                ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)

            # Bottom-left: mean spikes left/right
            ax = axes[1, 0]
            if spike_available:
                if mean_spk_left is not None:
                    ax.plot(t, mean_spk_left, color='g', lw=2, label=f'Left (n={int(np.sum(left_trials))})')
                if mean_spk_right is not None and any_right:
                    ax.plot(t, mean_spk_right, color='r', lw=2, label=f'Right (n={int(np.sum(right_trials))})')
                ax.axvline(0, color='k', ls='--', alpha=0.7)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Mean spikes')
                ax.set_title('Population spikes (left vs right)')
                if (any_left or any_right):
                    ax.legend(frameon=False)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No spike data', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()

            # Top-right: DFF ROI raster (sorted)
            ax = axes[0, 1]
            vmin = np.percentile(dff_sorted, 5)
            vmax = np.percentile(dff_sorted, 95)
            im = ax.imshow(dff_sorted, aspect='auto', cmap='viridis',
                           extent=[t[0], t[-1], 0, dff_sorted.shape[0]],
                           vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.axvline(0, color='white', ls='--', lw=2, alpha=0.9)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('ROIs (sorted)')
            ax.set_title(f'DFF ROI raster (sorted by {sort_method})')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='DF/F')

            # Bottom-right: Spike ROI raster (same order)
            ax = axes[1, 1]
            if spike_available and spk_sorted is not None:
                vmin_s = np.percentile(spk_sorted, 5)
                vmax_s = np.percentile(spk_sorted, 95)
                im2 = ax.imshow(spk_sorted, aspect='auto', cmap='viridis',
                                extent=[t[0], t[-1], 0, spk_sorted.shape[0]],
                                vmin=vmin_s, vmax=vmax_s, interpolation='nearest')
                ax.axvline(0, color='white', ls='--', lw=2, alpha=0.9)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('ROIs (same order)')
                ax.set_title('Spike ROI raster (matched order)')
                plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='Spikes')
            else:
                ax.text(0.5, 0.5, 'No spike data', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()

            # Ensure all x-axes share same limits for alignment
            xlim = (t[0], t[-1])
            for ax in axes.ravel():
                if hasattr(ax, 'set_xlim'):
                    ax.set_xlim(xlim)

            plt.tight_layout()
            plt.show()
            self.logger.info("SID_IMG: Generated DFF+spike trial segments check plot")

        except Exception as e:
            self.logger.error(f"SID_IMG: Failed to create trial segments check plot: {e}")

 
 
    def _load_qc_table(self, output_path: str) -> Optional[pd.DataFrame]:
        """
        Load QC metrics table written by dF/F QC step.
        Looks for (in order):
          - dff_qc_metrics.csv
          - *dff*qc*.csv (first match)
        Returns DataFrame or None.
        """
        try:
            cand = os.path.join(output_path, 'qc_results', 'post_dff_qc.csv')
            if os.path.exists(cand):
                return pd.read_csv(cand)
            # glob fallback
            # matches = glob.glob(os.path.join(output_path, '*dff*qc*.csv'))
            # if matches:
            #     return pd.read_csv(matches[0])
        except Exception as e:
            self.logger.warning(f"SID_IMG: Failed loading QC table: {e}")
        return None

    def _apply_qc_roi_filter(
            self,
            imaging_data: Dict[str, Any],
            output_path: str,
            subject_id: str
        ) -> Dict[str, Any]:
        """
        Apply ROI QC filter in-place (returns same dict for chaining).

        Uses columns:
          roi_id (1-based), keep_dff (1/0), optionally pass_core / pass_guardrails.
        Keeps rows where keep_dff == 1 (and any optional extra criteria configured).

        Stores:
          imaging_data['roi_keep_mask']
          imaging_data['roi_keep_indices']
          imaging_data['roi_excluded_indices']
          imaging_data['n_rois_qc_kept']
          imaging_data['n_rois_qc_total']
        """
        qc_df = self._load_qc_table(output_path)
        n_rois = imaging_data['dff_traces'].shape[0]
        if qc_df is None:
            self.logger.warning(f"SID_IMG: QC table not found for {subject_id}; using all {n_rois} ROIs.")
            mask = np.ones(n_rois, bool)
        else:
            # Configurable extra constraints (optional)
            cfg_qc = self.config.get('subjects', {}).get(subject_id, {}).get('imaging_qc', {})
            require_core = bool(cfg_qc.get('require_pass_core', False))
            min_snr = cfg_qc.get('min_snr', None)

            # Basic keep set
            if 'roi_id' not in qc_df.columns or 'keep_dff' not in qc_df.columns:
                self.logger.warning("SID_IMG: QC table missing roi_id / keep_dff; skipping QC filter.")
                mask = np.ones(n_rois, bool)
            else:
                df = qc_df.copy()
                # Build boolean condition
                cond = df['keep_dff'] == 1
                if require_core and 'pass_core' in df.columns:
                    cond &= (df['pass_core'] == 1)
                if min_snr is not None and 'snr' in df.columns:
                    cond &= (df['snr'] >= float(min_snr))
                keep_ids = df.loc[cond, 'roi_id'].astype(int).values - 1  # 0-based
                mask = np.zeros(n_rois, bool)
                # Guard against ids outside range
                keep_ids = keep_ids[(keep_ids >= 0) & (keep_ids < n_rois)]
                mask[keep_ids] = True

                lost = np.sum(~mask)
                self.logger.info(
                    f"SID_IMG: QC filter kept {mask.sum()}/{n_rois} ROIs "
                    f"(lost {lost}; require_core={require_core}, min_snr={min_snr})"
                )

        # Apply mask to primary arrays if not already filtered
        for key in ('dff_traces', 'spks', 'F', 'Fneu'):
            if key in imaging_data and isinstance(imaging_data[key], np.ndarray):
                if imaging_data[key].shape[0] == n_rois:
                    imaging_data[key] = imaging_data[key][mask, ...]

        # Record metadata
        imaging_data['roi_keep_mask'] = mask
        imaging_data['roi_keep_indices'] = np.nonzero(mask)[0].astype(int)
        imaging_data['roi_excluded_indices'] = np.nonzero(~mask)[0].astype(int)
        imaging_data['n_rois_qc_total'] = int(n_rois)
        imaging_data['n_rois_qc_kept'] = int(mask.sum())
        return imaging_data
 

    def analyze_subject(self, subject_id: str, suite2p_path: str, output_path: str, force: bool = False) -> Dict[str, Any]:
        """
        Analyze imaging sessions for a single subject.
        
        Args:
            subject_id: Subject identifier
            force: Force reanalysis even if results exist
            
        Returns:
            Analysis results dictionary
        """
        try:
            self.logger.info(f"SID_IMG_ANALYZER: Analyzing imaging sessions for subject {subject_id}")
            
            
            
            # Load preprocessed data
            loaded_data = self._load_preprocessed_data(subject_id, suite2p_path, output_path)

            if not loaded_data['success']:
                return loaded_data  # Return the error from loading
            
            behavioral_data = loaded_data['behavioral_data']
            imaging_data = loaded_data['imaging_data']
            
            # === APPLY QC ROI FILTER HERE ===
            imaging_data = self._apply_qc_roi_filter(imaging_data, output_path, subject_id)
            
            trial_data = self.extract_trial_segments_simple(
                imaging_data, behavioral_data['df_trials'], 
                alignment='trial_start', pre_sec=3.0, post_sec=3.0
            )            
            


            self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")
            
            # return

            # self.plot_trial_segments_spike_check(trial_data, subject_id)
            # self.plot_trial_segments_spike_trial_check(trial_data, subject_id, sort_method=None)
            # self.plot_trial_segments_spike_trial_check(trial_data, subject_id, sort_method='peak_time')
            # # Plot trial-segmented data to check alignment
            # self.plot_trial_segments_check(trial_data, subject_id)            
            
            
            
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='start_flash_1', pre_sec=3.0, post_sec=3.0
            # )            
            
            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")
            
            

            # self.plot_trial_segments_spike_check(trial_data, subject_id)
            # self.plot_trial_segments_spike_trial_check(trial_data, subject_id, sort_method=None)
            # self.plot_trial_segments_spike_trial_check(trial_data, subject_id)
            # # Plot trial-segmented data to check alignment
            # self.plot_trial_segments_check(trial_data, subject_id)            
            
            
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='start_flash_2', pre_sec=3.0, post_sec=3.0
            # )            
            
            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")
            
            

            # self.plot_trial_segments_spike_check(trial_data, subject_id)
            # self.plot_trial_segments_spike_trial_check(trial_data, subject_id, sort_method=None)
            # self.plot_trial_segments_spike_trial_check(trial_data, subject_id)
            # # Plot trial-segmented data to check alignment
            # self.plot_trial_segments_check(trial_data, subject_id)
            
            
            
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='choice_start', pre_sec=3.0, post_sec=3.0
            # )            
            
            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")
            
            

            # self.plot_trial_segments_spike_check(trial_data, subject_id)
            # self.plot_trial_segments_spike_trial_check(trial_data, subject_id, sort_method=None)
            # self.plot_trial_segments_spike_trial_check(trial_data, subject_id)
            # # Plot trial-segmented data to check alignment
            # self.plot_trial_segments_check(trial_data, subject_id)            
            
            
            
            
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='lick_start', pre_sec=3.0, post_sec=3.0
            # )            
            
            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")
            
            

            # self.plot_trial_segments_spike_check(trial_data, subject_id)
            # self.plot_trial_segments_spike_trial_check(trial_data, subject_id, sort_method=None)
            # self.plot_trial_segments_spike_trial_check(trial_data, subject_id)
            # # Plot trial-segmented data to check alignment
            # self.plot_trial_segments_check(trial_data, subject_id)            
            
            
            

            # # Plot trial 0 with no sorting
            # for sort_method in ['none', 'max_peak', 'min_peak']:
            #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=0, sort_method=sort_method)

            # # Plot trial 5 sorted by maximum peak response (most active ROIs at top)
            # for sort_method in ['none', 'max_peak', 'min_peak']:
            #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=5, sort_method=sort_method)

            # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # for sort_method in ['none', 'max_peak', 'min_peak']:
            #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=10, sort_method=sort_method)

            # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # for sort_method in ['none', 'max_peak', 'min_peak']:
            #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=20, sort_method=sort_method)
                
            # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # for sort_method in ['none', 'max_peak', 'min_peak']:
            #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=50, sort_method=sort_method)

            # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # for sort_method in ['none', 'max_peak', 'min_peak']:
            #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=80, sort_method=sort_method)
                
            # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # for sort_method in ['none', 'max_peak', 'min_peak']:
            #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=150, sort_method=sort_method)

            # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # for sort_method in ['none', 'max_peak', 'min_peak']:
            #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=200, sort_method=sort_method)                    

            # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # for sort_method in ['none', 'max_peak', 'min_peak']:
            #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=300, sort_method=sort_method)                    


            # Loop through first 5 trials with different sorting
            # for trial_idx in range(5):
            #     for sort_method in ['none', 'max_peak', 'min_peak']:
            #         self.plot_single_trial_raster(trial_data, subject_id, trial_idx, sort_method)
            
            # # Extract trial segments for analysis
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='trial_start', pre_sec=1.0, post_sec=8.0
            # )
                     
            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")
                        
            
            
            # Plot trial-segmented data to check alignment
            self.plot_trial_segments_check(trial_data, subject_id)
            
            # # Plot individual trials using segmented data
            # # self._plot_individual_trials_from_segments(trial_data, subject_id)

            # # NEW: Test condition-specific responsive ROI analysis
            # responsive_results = self.plot_responsive_rois_by_condition(trial_data, subject_id)
            
            # Sort by peak timing
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')
            
            # Sort by response magnitude  
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_magnitude')
            
            # Sort by response onset
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset')                
            
            # No sorting (current behavior)
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=False)


            # Plot all ROIs, all trials, sorted by peak timing
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='all')

            # Plot all ROIs, only rewarded trials, sorted by response magnitude
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='rewarded')

            # Plot all ROIs, left trials only, no sorting
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='left')

            # Plot all ROIs, right trials, sorted by response onset
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='right')









            trial_data = self.extract_trial_segments_simple(
                imaging_data, behavioral_data['df_trials'], 
                alignment='start_flash_1', pre_sec=2.0, post_sec=4.0
            )            
            
            self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")

            # # Plot trial-segmented data to check alignment
            # self.plot_trial_segments_check(trial_data, subject_id)
            
            
            drop_isi_values = [1700.0, 1850.0, 2000.0, 2150.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)
            drop_isi_values = [1700.0, 1850.0, 2000.0, 2300.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)            
            drop_isi_values = [1700.0, 1850.0, 2150.0, 2300.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)                        
            drop_isi_values = [1700.0, 2000.0, 2150.0, 2300.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)  
            drop_isi_values = [1850.0, 2000.0, 2150.0, 2300.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)             
            
            
            # Plot individual trials using segmented data
            # self._plot_individual_trials_from_segments(trial_data, subject_id)
            
            # NEW: Test condition-specific responsive ROI analysis
            # responsive_results = self.plot_responsive_rois_by_condition(trial_data, subject_id)
            
            # Sort by peak timing
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')
            
            # Sort by response magnitude  
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_magnitude')
            
            # Sort by response onset
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset')                                
            
            # No sorting (current behavior)
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=False)


            # Plot all ROIs, all trials, sorted by peak timing
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='all')

            # Plot all ROIs, only rewarded trials, sorted by response magnitude
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='rewarded')

            # Plot all ROIs, left trials only, no sorting
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='left')

            # Plot all ROIs, right trials, sorted by response onset
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='right')



       
            trial_data = self.extract_trial_segments_simple(
                imaging_data, behavioral_data['df_trials'], 
                alignment='end_flash_1', pre_sec=3.0, post_sec=3.0
            )            
            
            self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")
            
       
       
            drop_isi_values = [325.0,  450.0,  575.0,  700.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)
            drop_isi_values = [200.0,  450.0,  575.0,  700.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)            
            drop_isi_values = [200.0,  325.0,  575.0,  700.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)                        
            drop_isi_values = [200.0,  325.0,  450.0,  700.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)  
            drop_isi_values = [200.0,  325.0,  450.0,  575.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)
       
       
            # return
       
            # # long 1700., 1850., 2000., 2150., 2300
            
            drop_isi_values = [1700.0, 1850.0, 2000.0, 2150.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)
            drop_isi_values = [1700.0, 1850.0, 2000.0, 2300.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)            
            drop_isi_values = [1700.0, 1850.0, 2150.0, 2300.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)                        
            drop_isi_values = [1700.0, 2000.0, 2150.0, 2300.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)  
            drop_isi_values = [1850.0, 2000.0, 2150.0, 2300.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)              
            
            
            # Plot individual trials using segmented data
            # self._plot_individual_trials_from_segments(trial_data, subject_id)
            
       
            # Sort by peak timing
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')
            
            # Sort by response magnitude  
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_magnitude')
            
            # Sort by response onset
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset')                                
            
            # No sorting (current behavior)
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=False)
       
       
       
       
       
            # Plot all ROIs, all trials, sorted by peak timing
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='all')
       
            # Plot all ROIs, only rewarded trials, sorted by response magnitude
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='rewarded')
       
            # Plot all ROIs, left trials only, no sorting
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='left')
       
            # Plot all ROIs, right trials, sorted by response onset
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='right')
       





            trial_data = self.extract_trial_segments_simple(
                imaging_data, behavioral_data['df_trials'], 
                alignment='start_flash_2', pre_sec=3.0, post_sec=3.0
            )            
            
            self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")
            
            

            # self.plot_trial_segments_spike_check(trial_data, subject_id)
            # Plot trial-segmented data to check alignment
            self.plot_trial_segments_check(trial_data, subject_id)

            # # # Plot trial 0 with no sorting
            # # for sort_method in ['none', 'max_peak', 'min_peak']:
            # #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=0, sort_method=sort_method)

            # # # Plot trial 5 sorted by maximum peak response (most active ROIs at top)
            # # for sort_method in ['none', 'max_peak', 'min_peak']:
            # #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=5, sort_method=sort_method)

            # # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # # for sort_method in ['none', 'max_peak', 'min_peak']:
            # #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=10, sort_method=sort_method)

            # # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # # for sort_method in ['none', 'max_peak', 'min_peak']:
            # #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=20, sort_method=sort_method)
                
            # # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # # for sort_method in ['none', 'max_peak', 'min_peak']:
            # #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=50, sort_method=sort_method)

            # # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # # for sort_method in ['none', 'max_peak', 'min_peak']:
            # #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=80, sort_method=sort_method)
                
            # # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # # for sort_method in ['none', 'max_peak', 'min_peak']:
            # #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=150, sort_method=sort_method)

            # # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # # for sort_method in ['none', 'max_peak', 'min_peak']:
            # #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=200, sort_method=sort_method)                    

            # # # Plot trial 10 sorted by minimum peak response (least active ROIs at top)
            # # for sort_method in ['none', 'max_peak', 'min_peak']:
            # #     self.plot_single_trial_raster(trial_data, subject_id, trial_index=300, sort_method=sort_method)                    





            # self.plot_trial_segments_spike_check(trial_data, subject_id)
            # # Plot trial-segmented data to check alignment
            # self.plot_trial_segments_check(trial_data, subject_id)
            
            
            # # short  200.0,  325.0,  450.0,  575.0,  700.0

            drop_isi_values = [325.0,  450.0,  575.0,  700.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)
            drop_isi_values = [200.0,  450.0,  575.0,  700.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)            
            drop_isi_values = [200.0,  325.0,  575.0,  700.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)                        
            drop_isi_values = [200.0,  325.0,  450.0,  700.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)  
            drop_isi_values = [200.0,  325.0,  450.0,  575.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)    



            drop_isi_values = [1700.0, 1850.0, 2000.0, 2150.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)
            drop_isi_values = [1700.0, 1850.0, 2000.0, 2300.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)            
            drop_isi_values = [1700.0, 1850.0, 2150.0, 2300.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)                        
            drop_isi_values = [1700.0, 2000.0, 2150.0, 2300.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)  
            drop_isi_values = [1850.0, 2000.0, 2150.0, 2300.0]
            # After you call extract_trial_segments_simple(...)
            trial_data_filtered = self.filter_trial_data_by_isi(trial_data, drop_isi_values=drop_isi_values, atol=1e-3)                        
            self.plot_trial_segments_check(trial_data_filtered, subject_id)         


    
            # Sort by peak timing
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')
            
            # Sort by response magnitude  
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_magnitude')
            
            # Sort by response onset
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset')                                
            
            # No sorting (current behavior)
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=False)
  
  
            # Plot all ROIs, all trials, sorted by peak timing
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='all')
  
            # Plot all ROIs, only rewarded trials, sorted by response magnitude
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='rewarded')
  
            # Plot all ROIs, left trials only, no sorting
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='left')
  
            # Plot all ROIs, right trials, sorted by response onset
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='right')





     



       
            trial_data = self.extract_trial_segments_simple(
                imaging_data, behavioral_data['df_trials'], 
                alignment='servo_in', pre_sec=3.0, post_sec=3.0
            )            
            
            self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")

            # Plot trial-segmented data to check alignment
            self.plot_trial_segments_check(trial_data, subject_id)
            
            # Plot individual trials using segmented data
            # self._plot_individual_trials_from_segments(trial_data, subject_id)

            # Sort by peak timing
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')
            
            # Sort by response magnitude  
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_magnitude')
            
            # Sort by response onset
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset')                                
            
            # No sorting (current behavior)
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=False)





            # Plot all ROIs, all trials, sorted by peak timing
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='all')

            # Plot all ROIs, only rewarded trials, sorted by response magnitude
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='rewarded')

            # Plot all ROIs, left trials only, no sorting
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='left')

            # Plot all ROIs, right trials, sorted by response onset
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='right')





            trial_data = self.extract_trial_segments_simple(
                imaging_data, behavioral_data['df_trials'], 
                alignment='choice_start', pre_sec=4.0, post_sec=3.0
            ) 


            self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")

            # Plot trial-segmented data to check alignment
            self.plot_trial_segments_check(trial_data, subject_id)
            
            # Plot individual trials using segmented data
            # self._plot_individual_trials_from_segments(trial_data, subject_id)

            # Sort by peak timing
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')
            
            # Sort by response magnitude  
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_magnitude')
            
            # Sort by response onset
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset')                                
            
            # No sorting (current behavior)
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=False)





            # Plot all ROIs, all trials, sorted by peak timing
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='all')

            # Plot all ROIs, only rewarded trials, sorted by response magnitude
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='rewarded')

            # Plot all ROIs, left trials only, no sorting
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='left')

            # Plot all ROIs, right trials, sorted by response onset
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='right')




            trial_data = self.extract_trial_segments_simple(
                imaging_data, behavioral_data['df_trials'], 
                alignment='lick_start', pre_sec=4.0, post_sec=3.0
            ) 


            self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")

            # Plot trial-segmented data to check alignment
            self.plot_trial_segments_check(trial_data, subject_id)
            
            # Plot individual trials using segmented data
            # self._plot_individual_trials_from_segments(trial_data, subject_id)

            # Sort by peak timing
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')
            
            # Sort by response magnitude  
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_magnitude')
            
            # Sort by response onset
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset')                                
            
            # No sorting (current behavior)
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=False)





            # Plot all ROIs, all trials, sorted by peak timing
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='all')

            # Plot all ROIs, only rewarded trials, sorted by response magnitude
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='rewarded')

            # Plot all ROIs, left trials only, no sorting
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='left')

            # Plot all ROIs, right trials, sorted by response onset
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='right')





            trial_data = self.extract_trial_segments_simple(
                imaging_data, behavioral_data['df_trials'], 
                alignment='reward_start', pre_sec=4.0, post_sec=3.0
            )            
            
            self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")

            # Plot trial-segmented data to check alignment
            self.plot_trial_segments_check(trial_data, subject_id)
            
            # Plot individual trials using segmented data
            # self._plot_individual_trials_from_segments(trial_data, subject_id)

            # Sort by peak timing
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')
            
            # Sort by response magnitude  
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_magnitude')
            
            # Sort by response onset
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset')                                
            
            # No sorting (current behavior)
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=False)


            # Plot all ROIs, all trials, sorted by peak timing
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='all')

            # Plot all ROIs, only rewarded trials, sorted by response magnitude
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='rewarded')

            # Plot all ROIs, left trials only, no sorting
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='left')

            # Plot all ROIs, right trials, sorted by response onset
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='right')


            
            trial_data = self.extract_trial_segments_simple(
                imaging_data, behavioral_data['df_trials'], 
                alignment='punish_start', pre_sec=4.0, post_sec=4.0
            )            
            
            self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")
      
            # Plot trial-segmented data to check alignment
            self.plot_trial_segments_check(trial_data, subject_id)
            
            # Plot individual trials using segmented data
            # self._plot_individual_trials_from_segments(trial_data, subject_id)
      
            # Sort by peak timing
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')
            
            # Sort by response magnitude  
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_magnitude')
            
            # Sort by response onset
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset')                                
            
            # No sorting (current behavior)
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=False)
      
      
            # Plot all ROIs, all trials, sorted by peak timing
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='all')
      
            # Plot all ROIs, only rewarded trials, sorted by response magnitude
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='rewarded')
      
            # Plot all ROIs, left trials only, no sorting
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='left')
      
            # Plot all ROIs, right trials, sorted by response onset
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='right')
     


            trial_data = self.extract_trial_segments_simple(
                imaging_data, behavioral_data['df_trials'], 
                alignment='servo_out', pre_sec=3.0, post_sec=3.0
            )            
            
            self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")

            # Plot trial-segmented data to check alignment
            self.plot_trial_segments_check(trial_data, subject_id)
            
            # Plot individual trials using segmented data
            # self._plot_individual_trials_from_segments(trial_data, subject_id)
            

            # Sort by peak timing
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')
            
            # Sort by response magnitude  
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_magnitude')
            
            # Sort by response onset
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset')                                
            
            # No sorting (current behavior)
            self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=False)





            # Plot all ROIs, all trials, sorted by peak timing
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='all')

            # Plot all ROIs, only rewarded trials, sorted by response magnitude
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='rewarded')

            # Plot all ROIs, left trials only, no sorting
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='left')

            # Plot all ROIs, right trials, sorted by response onset
            self.plot_all_rois_heatmap(trial_data, subject_id, sort_rois=True, sort_method='response_onset', trial_type='right')
            
            
            
            
            
            
            
            
            
            
            
        except Exception as e:
            self.logger.error(f"SID_IMG_ANALYZER: Failed to analyze subject {subject_id}: {e}")
            return {'success': False, 'error': str(e)}
        
        
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='start_flash_1', pre_sec=2.0, post_sec=4.0
            # )            
            
            # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")            
            
            
         #    # # Plot trial-segmented data to check alignment
         #    # self.plot_trial_segments_check(trial_data, subject_id)
                        
         #    # # Sort by peak timing
         #    # self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')            
            
            
         #    # -----------------------------
         #    # Build F1-aligned segments → contract
         #    # -----------------------------
         #    td_f1 = self.extract_trial_segments_simple(
         #        imaging_data, behavioral_data['df_trials'],
         #        alignment='start_flash_1', pre_sec=2.0, post_sec=4.0
         #    )
         #    c1 = self._segments_to_contract(td_f1, alignment='start_flash_1')
         #    if not c1.get('success'):
         #        return {'success': False, 'error': f"F1 contract failed: {c1.get('error')}"}

         #    # QUICK TEST: thin data to speed up (adjust as needed)
         #    quick = False
         #    if quick:
         #        Xq, tq, ISIq, F2q, roi_idx, trial_idx = thin_contract(
         #            c1['X'], c1['t'], c1['ISI'], c1['F2_idx'],
         #            roi_max=32,            # keep first 32 ROIs
         #            trials_per_level=8,    # keep up to 8 trials per ISI level
         #            time_stride=2,         # keep every 2nd frame
         #            random_state=0
         #        )
         #    else:
         #        Xq, tq, ISIq, F2q = c1['X'], c1['t'], c1['ISI'], c1['F2_idx']

         # # Z-score and run analyses with fewer CV folds for speed
         #    Xz = zscore_across_trials(Xq, baseline_idx=(tq < 0))

         #    sc_res = scaling_vs_clock(Xz, tq, ISIq, F2q, n_centers=8, n_splits=3, seed=0)
         #    labels = primary_timing_label(sc_res.delta_R2, getattr(sc_res, 'alpha_opt', None))

         #    # Boundary for short/long
         #    uniq = np.sort(np.unique(np.round(ISIq, 6)))
         #    if len(uniq) >= 2:
         #        mid = len(uniq) // 2
         #        boundary = 0.5 * (uniq[mid - 1] + uniq[mid])
         #    else:
         #        boundary = float(np.nanmedian(ISIq))

         #    neu = time_resolved_decoding(Xz, tq, ISIq, F2q, boundary=boundary, preF2_only=True, n_splits=3, seed=0)
         #    haz = hazard_unique_variance(Xz, tq, ISIq, F2q, allowed_ISI=np.unique(ISIq), n_splits=3, seed=0)

         #    # F2-aligned for PE (also thin similarly if desired)
         #    td_f2 = self.extract_trial_segments_simple(imaging_data, behavioral_data['df_trials'],
         #                                               alignment='start_flash_2', pre_sec=2.0, post_sec=4.0)
         #    c2 = self._segments_to_contract(td_f2, alignment='start_flash_2')
         #    pe_res = None
         #    if c2.get('success'):
         #        X2q, t2q, _, F22q, _, trial_idx2 = thin_contract(
         #            c2['X'], c2['t'], ISIq, c2['F2_idx'],
         #            roi_max=32, trials_per_level=8, time_stride=2, random_state=0
         #        )
         #        Xz_f2 = zscore_across_trials(X2q, baseline_idx=(t2q < 0))
         #        pe_res = prediction_error_at_F2(Xz_f2, t2q, ISIq[trial_idx2], window=(0.0, 0.25), surprise='abs_deviation')

         #    order = raster_sort_index(sc_res.delta_R2, getattr(sc_res, 'alpha_opt', None), Xz, tq, F2q, mode='auto')


         #    # return {
         #    #     'success': True,
         #    #     'subject_id': subject_id,
         #    #     'roi_timing_quick': True,
         #    #     'contract_sizes': {
         #    #         'orig': {'n_roi': c1['n_roi'], 'n_time': c1['n_time'], 'n_trial': c1['n_trial']},
         #    #         'thin': {'n_roi': int(Xz.shape[0]), 'n_time': int(Xz.shape[1]), 'n_trial': int(Xz.shape[2])}
         #    #     },
         #    #     'scaling_clock': {
         #    #         'delta_R2': getattr(sc_res, 'delta_R2', None),
         #    #         'alpha_opt': getattr(sc_res, 'alpha_opt', None),
         #    #         'labels': labels
         #    #     },
         #    #     'neurometric': {
         #    #         'time': getattr(neu, 'times', None),
         #    #         'auc_binary': getattr(neu, 'auc_binary', None),
         #    #         'r2_continuous': getattr(neu, 'r2_continuous', None)
         #    #     },
         #    #     'hazard': {
         #    #         'partial_r2_hazard': getattr(haz, 'partial_r2_hazard', None)
         #    #     },
         #    #     'prediction_error': None if pe_res is None else {
         #    #         'slope': getattr(pe_res, 'slope', None),
         #    #         'pval': getattr(pe_res, 'pval', None)
         #    #     },
         #    #     'raster_order': order.tolist() if hasattr(order, 'tolist') else order
         #    # }

         #    # Build z-scored F2-aligned tensor if available
         #    Xz_f2 = None
         #    if c2.get('success'):
         #        Xz_f2 = zscore_across_trials(c2['X'], baseline_idx=(c2['t'] < 0))

         #    # One-page report
         #    report_path = self.plot_roi_timing_report(
         #        subject_id=subject_id,
         #        output_dir=output_path,
         #        c1=c1, Xz_f1=Xz,
         #        sc_res=sc_res, neu=neu, haz=haz, order=order,
         #        c2=c2 if c2.get('success') else None, Xz_f2=Xz_f2,
         #        behavioral_df=behavioral_data['df_trials'],
         #        pe_res=pe_res, pe_window=(0.0, 0.25)
         #    )

         #    return {
         #        'success': True,
         #        'subject_id': subject_id,
         #        'roi_timing_quick': True,
         #        'report_path': report_path,
         #        'contract_sizes': {
         #            'orig': {'n_roi': c1['n_roi'], 'n_time': c1['n_time'], 'n_trial': c1['n_trial']},
         #            'thin': {'n_roi': int(Xz.shape[0]), 'n_time': int(Xz.shape[1]), 'n_trial': int(Xz.shape[2])}
         #        },
         #        'scaling_clock': {
         #            'delta_R2': getattr(sc_res, 'delta_R2', None),
         #            'alpha_opt': getattr(sc_res, 'alpha_opt', None),
         #            'labels': labels
         #        },
         #        'neurometric': {
         #            'time': getattr(neu, 'times', None),
         #            'auc_binary': getattr(neu, 'auc_binary', None),
         #            'r2_continuous': getattr(neu, 'r2_continuous', None),
         #            'divergence_time_s': getattr(neu, 'divergence_time_s', None)
         #        },
         #        'hazard': {
         #            'partial_r2_hazard': getattr(haz, 'partial_r2_hazard', None)
         #        },
         #        'prediction_error': None if pe_res is None else {
         #            'slope': getattr(pe_res, 'slope', None),
         #            'pval': getattr(pe_res, 'pval', None)
         #        },
         #        'raster_order': order.tolist() if hasattr(order, 'tolist') else order
         #    }


            # # Z-score across trials using baseline (t<0)
            # baseline_mask = c1['t'] < 0
            # Xz = zscore_across_trials(c1['X'], baseline_idx=baseline_mask)

            # # 1) Scaling vs Clock (ΔR²) + labels
            # sc_res = scaling_vs_clock(Xz, c1['t'], c1['ISI'], c1['F2_idx'])
            # labels = primary_timing_label(sc_res.delta_R2, getattr(sc_res, 'alpha_opt', None))

            # # 2) Neurometric decoding (binary short/long). Boundary = midpoint of two middle ISIs.
            # uniq = np.sort(np.unique(np.round(c1['ISI'], 6)))
            # if len(uniq) >= 2:
            #     mid = len(uniq) // 2
            #     boundary = 0.5 * (uniq[mid - 1] + uniq[mid])
            # else:
            #     boundary = float(np.nanmedian(c1['ISI']))

            # neu = time_resolved_decoding(
            #     Xz, c1['t'], c1['ISI'], c1['F2_idx'],
            #     boundary=boundary, preF2_only=True
            # )

            # # 3) Hazard unique variance (pre-F2)
            # haz = hazard_unique_variance(
            #     Xz, c1['t'], c1['ISI'], c1['F2_idx'], allowed_ISI=np.unique(c1['ISI'])
            # )

            # # -----------------------------
            # # Build F2-aligned segments → contract (for PE)
            # # -----------------------------
            # td_f2 = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'],
            #     alignment='start_flash_2', pre_sec=2.0, post_sec=4.0
            # )
            # c2 = self._segments_to_contract(td_f2, alignment='start_flash_2')
            # pe_res = None
            # if c2.get('success'):
            #     Xz_f2 = zscore_across_trials(c2['X'], baseline_idx=(c2['t'] < 0))
            #     pe_res = prediction_error_at_F2(
            #         Xz_f2, c2['t'], c1['ISI'], window=(0.0, 0.25), surprise='abs_deviation'
            #     )

            # # 5) Raster sort index (auto from ΔR² and templates)
            # order = raster_sort_index(
            #     sc_res.delta_R2, getattr(sc_res, 'alpha_opt', None),
            #     Xz, c1['t'], c1['F2_idx'], mode='auto'
            # )

            # # 6) Split-half reliability
            # sh = split_half_reliability(Xz, c1['t'], c1['ISI'], c1['F2_idx'])

            # # Package results (lightweight)
            # return {
            #     'success': True,
            #     'subject_id': subject_id,
            #     'roi_timing': {
            #         'contract_f1': {'n_roi': c1['n_roi'], 'n_time': c1['n_time'], 'n_trial': c1['n_trial']},
            #         'scaling_clock': {
            #             'R2_clock': getattr(sc_res, 'R2_clock', None),
            #             'R2_scale': getattr(sc_res, 'R2_scale', None),
            #             'delta_R2': getattr(sc_res, 'delta_R2', None),
            #             'alpha_opt': getattr(sc_res, 'alpha_opt', None),
            #             'labels': labels,
            #         },
            #         'neurometric': {
            #             'time': getattr(neu, 'times', None),
            #             'auc_binary': getattr(neu, 'auc_binary', None),
            #             'r2_continuous': getattr(neu, 'r2_continuous', None),
            #             'divergence_time_s': getattr(neu, 'divergence_time_s', None),
            #             'boundary': boundary,
            #         },
            #         'hazard': {
            #             'partial_r2_hazard': getattr(haz, 'partial_r2_hazard', None),
            #             'r2_full': getattr(haz, 'r2_full', None),
            #             'r2_reduced_time': getattr(haz, 'r2_reduced_time', None),
            #         },
            #         'prediction_error': None if pe_res is None else {
            #             'slope': getattr(pe_res, 'slope', None),
            #             'pval': getattr(pe_res, 'pval', None),
            #             'name': getattr(pe_res, 'surprise_name', 'abs_deviation'),
            #         },
            #         'raster_order': order,
            #         'split_half': {
            #             'deltaR2_agreement': getattr(sh, 'deltaR2_agreement', None),
            #             'order_reliability': getattr(sh, 'order_reliability', None),
            #         },
            #     },
            # }            
            
            
            
            # trial_data = self.extract_trial_segments_simple(
            #     imaging_data, behavioral_data['df_trials'], 
            #     alignment='start_flash_2', pre_sec=2.0, post_sec=4.0
            # )              
            
            
            
            
            
            
            # # trial_data = self.extract_trial_segments_simple(
            # #     imaging_data, behavioral_data['df_trials'], 
            # #     alignment='start_flash_1', pre_sec=2.0, post_sec=4.0
            # # )            
            
            # # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")            
            
            
            # # # Plot trial-segmented data to check alignment
            # # self.plot_trial_segments_check(trial_data, subject_id)
                        
            # # # Sort by peak timing
            # # self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')
            
            
            
            
            
            # # trial_data = self.extract_trial_segments_simple(
            # #     imaging_data, behavioral_data['df_trials'], 
            # #     alignment='start_flash_2', pre_sec=2.0, post_sec=4.0
            # # )            
            
            # # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")            
            
            
            # # # Plot trial-segmented data to check alignment
            # # self.plot_trial_segments_check(trial_data, subject_id)
                        
            # # # Sort by peak timing
            # # self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')            
            
            
            
            
            
            # # trial_data = self.extract_trial_segments_simple(
            # #     imaging_data, behavioral_data['df_trials'], 
            # #     alignment='choice_start', pre_sec=2.0, post_sec=4.0
            # # )            
            
            # # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")            
            
            
            # # # Plot trial-segmented data to check alignment
            # # self.plot_trial_segments_check(trial_data, subject_id)
                        
            # # # Sort by peak timing
            # # self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')            
            
            
            
            
            # # trial_data = self.extract_trial_segments_simple(
            # #     imaging_data, behavioral_data['df_trials'], 
            # #     alignment='reward_start', pre_sec=2.0, post_sec=4.0
            # # )            
            
            # # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")            
            
            
            # # # Plot trial-segmented data to check alignment
            # # self.plot_trial_segments_check(trial_data, subject_id)
                        
            # # # Sort by peak timing
            # # self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')            
            
            
            
            
            # # trial_data = self.extract_trial_segments_simple(
            # #     imaging_data, behavioral_data['df_trials'], 
            # #     alignment='servo_out', pre_sec=2.0, post_sec=4.0
            # # )            
            
            # # self.logger.info(f"SID_IMG: Extracted {trial_data['dff_segments_array'].shape[0]} trials")
            # # self.logger.info(f"SID_IMG: Segment shape: {trial_data['dff_segments_array'].shape}")            
            
            
            # # # Plot trial-segmented data to check alignment
            # # self.plot_trial_segments_check(trial_data, subject_id)
                        
            # # # Sort by peak timing
            # # self.plot_condition_comparison_heatmap(trial_data, subject_id, sort_rois=True, sort_method='peak_time')            
            
            
            
            
            # # For now, just return the loaded data
            # return {
            #     'success': True,
            #     'subject_id': subject_id,
            #     'loaded_data': loaded_data
            # }        