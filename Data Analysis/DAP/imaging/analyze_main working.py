from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional, Union, Callable
from dataclasses import dataclass
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.decomposition import NMF
from scipy.optimize import minimize
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
import warnings
from matplotlib.gridspec import GridSpec
from copy import deepcopy
import pandas as pd
from scipy import stats
from matplotlib.gridspec import GridSpec
import pickle
import os
import sys
from pathlib import Path
import yaml
from dPCA.dPCA import dPCA




def load_sid_data(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load SID imaging data from pickle file using config path"""
    
    data_path = os.path.join(cfg['paths']['output_dir'], 'sid_imaging_data.pkl')
    print(f"Loading SID data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"SID imaging data file not found: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data loaded successfully:")
    print(f"  ROIs: {data['F'].shape[0]}")
    print(f"  Timepoints: {data['F'].shape[1]} ({data['F'].shape[1]/data['imaging_fs']:.1f}s)")
    print(f"  Trials: {len(data['df_trials'])}")
    print(f"  Sampling rate: {data['imaging_fs']:.1f} Hz")
    
    return data



def load_cfg_yaml(path: str) -> Dict[str, Any]:
    print(f"Loading config from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"Config loaded successfully with {len(cfg)} sections")
    return cfg


# def trim_session_trials(data: Dict[str, Any], n_trim_start: int = 15, n_trim_end: int = 15) -> Dict[str, Any]:
#     """Trim trials from start/end of session for all analyses"""
#     df_trials = data['df_trials']
    
#     if len(df_trials) <= (n_trim_start + n_trim_end):
#         print(f"WARNING: Only {len(df_trials)} trials, cannot trim {n_trim_start}+{n_trim_end}")
#         trimmed_trials = df_trials
#     else:
#         trimmed_trials = df_trials.iloc[n_trim_start:-n_trim_end].reset_index(drop=True)
#         print(f"Trimmed trials: {len(df_trials)} -> {len(trimmed_trials)} (removed {n_trim_start} start, {n_trim_end} end)")
    
#     # Update data dict
#     data_trimmed = data.copy()
#     data_trimmed['df_trials'] = trimmed_trials
#     data_trimmed['n_trials_original'] = len(df_trials)
#     data_trimmed['n_trials_trimmed'] = len(trimmed_trials)
#     data_trimmed['trim_params'] = {'n_trim_start': n_trim_start, 'n_trim_end': n_trim_end}
    
#     return data_trimmed



def trim_session_trials(data: Dict[str, Any], n_trim_start: int = 15, n_trim_end: int = 15) -> Dict[str, Any]:
    """Trim trials from start/end of session for all analyses"""
    df_trials = data['df_trials']
    
    if len(df_trials) <= (n_trim_start + n_trim_end):
        print(f"WARNING: Only {len(df_trials)} trials, cannot trim {n_trim_start}+{n_trim_end}")
        trimmed_trials = df_trials.copy()
    else:
        # Keep original trial indices and timestamps - just select subset
        trimmed_trials = df_trials.iloc[n_trim_start:-n_trim_end].copy()
        # DON'T reset index - keep original trial numbers
        print(f"Trimmed trials: {len(df_trials)} -> {len(trimmed_trials)} (removed {n_trim_start} start, {n_trim_end} end)")
        print(f"Trial index range: {trimmed_trials.index.min()} to {trimmed_trials.index.max()}")
    
    # Update data dict - keep all imaging data
    data_trimmed = data.copy()
    data_trimmed['df_trials'] = trimmed_trials
    data_trimmed['n_trials_original'] = len(df_trials)
    data_trimmed['n_trials_trimmed'] = len(trimmed_trials)
    data_trimmed['trim_params'] = {'n_trim_start': n_trim_start, 'n_trim_end': n_trim_end}
    
    # Keep full session imaging data
    # data_trimmed['dFF_clean'] = data['dFF_clean']  # unchanged
    # data_trimmed['imaging_time'] = data['imaging_time']  # unchanged
    
    return data_trimmed






















def create_event_aligned_stacks(data: Dict[str, Any], 
                               cfg: Dict[str, Any],
                               event_name: str) -> Dict[str, Any]:
    """Create event-aligned stacks using IMAGING time vectors and proper interpolation"""
    
    print(f"\n=== CREATING {event_name.upper()} ALIGNED STACKS ===")
    
    # Get window parameters from config
    window_cfg = cfg['event_analysis']['windows'][event_name]
    pre_event_s = window_cfg['pre_event_s']
    post_event_s = window_cfg['post_event_s']
    
    df_trials = data['df_trials']
    dff_clean = data['dFF_clean']  # (n_rois, n_timepoints)
    imaging_time = data['imaging_time']  # CORRECT: use imaging_time, not vol_time
    imaging_fs = data['imaging_fs']      # CORRECT: use imaging_fs (~30Hz)
    
    print(f"  Event: {event_name}")
    print(f"  Window: -{pre_event_s:.3f}s to +{post_event_s:.3f}s")
    print(f"  Imaging sampling: {imaging_fs:.1f} Hz")
    print(f"  Imaging time range: {imaging_time[0]:.1f} to {imaging_time[-1]:.1f}s")
    
    # For 30Hz sampling, we get very few points per window
    window_duration = pre_event_s + post_event_s
    expected_samples = int(window_duration * imaging_fs)
    print(f"  Expected samples per window: {expected_samples} at {imaging_fs:.1f}Hz")
    
    if expected_samples < 10:
        print(f"  WARNING: Only {expected_samples} samples per window - consider longer windows or interpolation")
    
    return _extract_trial_segments_imaging(df_trials, dff_clean, imaging_time, imaging_fs, 
                                          event_name, pre_event_s, post_event_s)



def _extract_trial_segments_imaging(df_trials: pd.DataFrame, 
                                   dff_clean: np.ndarray,
                                   imaging_time: np.ndarray,
                                   imaging_fs: float,
                                   event_name: str,
                                   pre_event_s: float,
                                   post_event_s: float) -> Dict[str, Any]:
    """Extract event-aligned segments using IMAGING time indexing with fixed slice length"""
    
    n_rois, _ = dff_clean.shape
    n_trials = len(df_trials)
    
    # Remove edge trials
    edge_buffer = 15
    if n_trials <= 2 * edge_buffer:
        print(f"  WARNING: Only {n_trials} trials, cannot remove edge buffer")
        df_trials_trimmed = df_trials
    else:
        df_trials_trimmed = df_trials.iloc[edge_buffer:-edge_buffer].reset_index(drop=True)
        print(f"  Removed {edge_buffer} trials from start/end: {n_trials} -> {len(df_trials_trimmed)}")
    
    # Z-score normalize dF/F per ROI across entire session
    dff_zscore = zscore(dff_clean, axis=1)  # normalize across time dimension
    print(f"  Applied z-score normalization per ROI")
    
    # Calculate fixed slice length based on window duration and sampling rate
    window_duration = pre_event_s + post_event_s
    target_n_samples = int(window_duration * imaging_fs) + 1  # +1 to ensure we have enough
    pre_samples = int(pre_event_s * imaging_fs)
    post_samples = target_n_samples - pre_samples
    
    print(f"  Fixed slice length: {target_n_samples} samples")
    print(f"    Pre-event: {pre_samples} samples ({pre_samples/imaging_fs:.3f}s)")
    print(f"    Post-event: {post_samples} samples ({post_samples/imaging_fs:.3f}s)")
    
    trial_stacks = []
    trial_metadata = []
    
    for trial_idx in range(len(df_trials_trimmed)):
        trial = df_trials_trimmed.iloc[trial_idx]
        
        # Skip if event time is NaN
        if pd.isna(trial[event_name]):
            continue
            
        # Calculate absolute event time (already in seconds)
        event_abs_time = trial['trial_start_timestamp'] + trial[event_name]
        
        # Find closest imaging sample to event time
        event_idx = np.argmin(np.abs(imaging_time - event_abs_time))
        
        # Calculate slice indices with fixed length
        start_idx = event_idx - pre_samples
        end_idx = start_idx + target_n_samples
        
        # Check bounds
        if start_idx < 0 or end_idx >= len(imaging_time):
            print(f"    Trial {trial_idx}: out of bounds (indices {start_idx}:{end_idx})")
            continue
            
        # Extract segment for all ROIs with FIXED length
        trial_stack = dff_zscore[:, start_idx:end_idx]  # (n_rois, target_n_samples)
        
        # Verify we got the expected shape
        if trial_stack.shape[1] != target_n_samples:
            print(f"    Trial {trial_idx}: unexpected shape {trial_stack.shape}, skipping")
            continue
            
        trial_stacks.append(trial_stack)
        
        # Create time vector relative to event using FIXED sampling
        # Use the target window duration and number of samples
        segment_time = np.linspace(-pre_event_s, post_event_s, target_n_samples)
        
        # Store trial metadata
        trial_meta = _create_trial_metadata_fixed(trial, trial_idx, event_abs_time, 
                                                 segment_time, event_idx, start_idx, end_idx)
        trial_metadata.append(trial_meta)
    
    if len(trial_stacks) == 0:
        raise ValueError(f"No valid trials found for event {event_name}")
    
    # Convert to array (trials, rois, time) - all should have same shape now
    trial_stacks = np.stack(trial_stacks, axis=0)
    
    # Use the fixed time vector
    time_vector = segment_time
    
    print(f"  Valid trials: {len(trial_stacks)}/{len(df_trials_trimmed)}")
    print(f"  Stack shape: {trial_stacks.shape}")
    print(f"  Time vector: {len(time_vector)} samples")
    print(f"  Time range: {time_vector[0]:.3f} to {time_vector[-1]:.3f}s")
    print(f"  Effective sampling rate: {len(time_vector)/(time_vector[-1]-time_vector[0]):.1f} Hz")
    
    return {
        'event_name': event_name,
        'stacks': trial_stacks,  # z-scored, (trials, rois, time)
        'time_vector': time_vector,
        'trial_metadata': trial_metadata,
        'n_trials': len(trial_stacks),
        'n_rois': trial_stacks.shape[1],
        'normalization': 'zscore_per_roi_session',
        'original_fs': imaging_fs,
        'effective_fs': len(time_vector)/(time_vector[-1]-time_vector[0]) if len(time_vector) > 1 else imaging_fs,
        'fixed_slice_length': target_n_samples
    }

def _create_trial_metadata_fixed(trial: pd.Series, 
                               trial_idx: int, 
                               event_abs_time: float, 
                               segment_time: np.ndarray,
                               event_idx: int,
                               start_idx: int,
                               end_idx: int) -> Dict[str, Any]:
    """Create metadata dict for a single trial with fixed slicing info"""
    return {
        'trial_idx': trial_idx,
        'event_abs_time': event_abs_time,
        'event_idx': event_idx,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'segment_time': segment_time,
        'isi': trial['isi'],
        'isi_category': 'short' if trial['isi'] <= trial['mean_isi'] else 'long',
        'is_short': trial['isi'] <= trial['mean_isi'],
        'mouse_correct': trial.get('mouse_correct', np.nan),
        'mouse_choice': trial.get('mouse_choice', np.nan),
        'rewarded': trial.get('rewarded', False),
        'punished': trial.get('punished', False),
        'lick': trial.get('lick', False),
        'lick_start': trial.get('lick_start', False),
        'did_not_choose': trial.get('did_not_choose', False),
        'time_did_not_choose': trial.get('time_did_not_choose', False),
        'RT': trial.get('RT', np.nan)
    }





def interpolate_event_stacks(stack_data: Dict[str, Any], target_fs: float = 100.0) -> Dict[str, Any]:
    """Interpolate event stacks to higher sampling rate for better temporal resolution"""
    print(f"\n=== INTERPOLATING STACKS TO {target_fs:.1f} Hz ===")

    stacks = stack_data['stacks']  # (trials, rois, time)
    time_vector = stack_data['time_vector']
    original_fs = stack_data.get('effective_fs', stack_data.get('original_fs', 30.0))

    print(f"  Original: {stacks.shape[2]} samples at {original_fs:.1f} Hz")

    # Create new time vector
    time_start = time_vector[0]
    time_end = time_vector[-1]
    duration = time_end - time_start
    n_new_samples = int(duration * target_fs) + 1

    new_time_vector = np.linspace(time_start, time_end, n_new_samples)

    print(f"  Target: {n_new_samples} samples at {target_fs:.1f} Hz")
    print(f"  Interpolation factor: {n_new_samples / stacks.shape[2]:.2f}x")

    # Interpolate each ROI trace for each trial
    from scipy.interpolate import interp1d

    n_trials, n_rois, _ = stacks.shape
    new_stacks = np.zeros((n_trials, n_rois, n_new_samples), dtype=np.float32)

    for trial_idx in range(n_trials):
        for roi_idx in range(n_rois):
            # Skip if all NaN
            trace = stacks[trial_idx, roi_idx, :]
            if np.all(np.isnan(trace)):
                new_stacks[trial_idx, roi_idx, :] = np.nan
                continue
            
            # Handle NaN values by interpolation
            valid_mask = np.isfinite(trace)
            if np.sum(valid_mask) < 2:
                new_stacks[trial_idx, roi_idx, :] = np.nan
                continue
            
            # Interpolate
            interp_func = interp1d(time_vector[valid_mask], trace[valid_mask], 
                                    kind='linear', bounds_error=False, fill_value='extrapolate')
            new_stacks[trial_idx, roi_idx, :] = interp_func(new_time_vector)

    # Update stack_data
    stack_data_interp = stack_data.copy()
    stack_data_interp.update({
        'stacks': new_stacks,
        'time_vector': new_time_vector,
        'effective_fs': target_fs,
        'interpolated': True,
        'original_n_samples': stacks.shape[2],
        'interpolated_n_samples': n_new_samples
    })

    print(f"  Interpolation complete: {new_stacks.shape}")

    return stack_data_interp






def visualize_stack_data(stack_data: Dict[str, Any], n_rois_show: int = 10):
    """Quick visualization of stack data"""
    stacks = stack_data['stacks']  # (trials, rois, time)
    time_vector = stack_data['time_vector']
    event_name = stack_data['event_name']
    
    # Calculate trial-averaged response
    mean_response = np.mean(stacks, axis=0)  # (rois, time)
    
    # Show first n_rois_show ROIs
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: Heatmap of ROI responses
    roi_subset = slice(0, min(n_rois_show, mean_response.shape[0]))
    im = axes[0].imshow(mean_response[roi_subset], aspect='auto', cmap='RdBu_r',
                       extent=[time_vector[0], time_vector[-1], 0, n_rois_show])
    axes[0].axvline(0, color='white', linestyle='--', alpha=0.8, linewidth=2)
    axes[0].set_ylabel(f'ROI Index (first {n_rois_show})')
    axes[0].set_title(f'{event_name}: Trial-Averaged Responses (z-scored dF/F)')
    plt.colorbar(im, ax=axes[0], label='z-scored dF/F')
    
    # Bottom: Population average
    pop_mean = np.mean(mean_response, axis=0)
    pop_sem = np.std(mean_response, axis=0) / np.sqrt(mean_response.shape[0])
    
    axes[1].plot(time_vector, pop_mean, 'k-', linewidth=2, label='Population Mean')
    axes[1].fill_between(time_vector, pop_mean - pop_sem, pop_mean + pop_sem, 
                        alpha=0.3, color='gray', label='±SEM')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Event')
    axes[1].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('z-scored dF/F')
    axes[1].set_title('Population Response')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some stats
    print(f"\n=== {event_name.upper()} STACK SUMMARY ===")
    print(f"Shape: {stacks.shape} (trials, rois, time)")
    print(f"Time resolution: {1000/stack_data['effective_fs']:.1f}ms per sample")
    print(f"Z-score range: {np.nanmin(stacks):.2f} to {np.nanmax(stacks):.2f}")
    print(f"Population response peak: {np.max(np.abs(pop_mean)):.3f} at {time_vector[np.argmax(np.abs(pop_mean))]:.3f}s")




















































def run_isi_phase_analysis_from_data(data: Dict[str, Any], 
                                    n_phase_bins: int = 80,
                                    n_components: int = 12,
                                    apply_zscore: bool = False) -> Dict[str, Any]:
    """
    Replicate the successful phase analysis approach using your data structure
    This creates phase-normalized ISI segments and runs CP decomposition
    """
    print(f"\n=== ISI PHASE ANALYSIS FROM DATA STRUCTURE ===")
    
    df_trials = data['df_trials']
    dff_clean = data['dFF_clean']
    imaging_time = data['imaging_time']
    
    # 1. Extract ISI segments and resample to phase bins (like add_phase_resampled_chunk)
    isi_phase_array, trial_metadata = _extract_isi_phase_segments(
        df_trials, dff_clean, imaging_time, n_phase_bins, apply_zscore
    )
    
    print(f"Extracted ISI phase array: {isi_phase_array.shape}")
    
    # 2. Run CP decomposition on phase data (like run_chunk_cp_pipeline)
    cp_results = _run_cp_on_phase_data(isi_phase_array, n_components)
    
    # 3. Extract signed groups from CP loadings (like extract_signed_roi_groups)
    A_matrix = cp_results['A']  # ROI loadings
    signed_groups = _extract_signed_groups_from_cp(A_matrix, q=0.10)
    
    # 4. Orient the signed groups (like orient_signed_groups)
    _orient_signed_groups_phase(signed_groups, cp_results, isi_phase_array, trial_metadata)
    
    return {
        'cp_results': cp_results,
        'signed_groups': signed_groups,
        'isi_phase_array': isi_phase_array,
        'trial_metadata': trial_metadata,
        'phase_bins': np.linspace(0, 1, n_phase_bins)
    }



def _extract_isi_phase_segments(df_trials: pd.DataFrame, 
                               dff_clean: np.ndarray,
                               imaging_time: np.ndarray,
                               n_phase_bins: int,
                               apply_zscore: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """Extract ISI segments and resample to phase bins (0-1)"""
    
    from scipy.interpolate import interp1d
    
    n_rois = dff_clean.shape[0]
    
    # OPTIONAL z-scoring per ROI
    if apply_zscore:
        dff_processed = zscore(dff_clean, axis=1)
        print("Applied z-score normalization per ROI")
    else:
        dff_processed = dff_clean
        print("Using original dF/F (no z-scoring)")
    
    phase_segments = []
    trial_metadata = []
    
    for trial_idx, trial in df_trials.iterrows():
        if pd.isna(trial['end_flash_1']) or pd.isna(trial['start_flash_2']):
            continue
            
        # Get ISI period
        isi_start_abs = trial['trial_start_timestamp'] + trial['end_flash_1']
        isi_end_abs = trial['trial_start_timestamp'] + trial['start_flash_2']
        
        # Find imaging indices
        isi_start_idx = np.argmin(np.abs(imaging_time - isi_start_abs))
        isi_end_idx = np.argmin(np.abs(imaging_time - isi_end_abs))
        
        if isi_end_idx - isi_start_idx < 3:
            continue
            
        # Extract ISI segment
        isi_segment = dff_processed[:, isi_start_idx:isi_end_idx+1]
        isi_times = imaging_time[isi_start_idx:isi_end_idx+1]
        
        # Convert to phase (0 to 1)
        isi_duration = isi_end_abs - isi_start_abs
        isi_phases = (isi_times - isi_start_abs) / isi_duration
        
        # Resample to fixed phase grid
        phase_grid = np.linspace(0, 1, n_phase_bins)
        phase_segment = np.zeros((n_rois, n_phase_bins))
        
        for roi_idx in range(n_rois):
            roi_trace = isi_segment[roi_idx]
            if not np.all(np.isnan(roi_trace)):
                valid_mask = np.isfinite(roi_trace)
                if np.sum(valid_mask) >= 2:
                    try:
                        interp_func = interp1d(isi_phases[valid_mask], roi_trace[valid_mask],
                                             kind='linear', bounds_error=False, fill_value='extrapolate')
                        phase_segment[roi_idx] = interp_func(phase_grid)
                    except:
                        phase_segment[roi_idx] = np.nan
                else:
                    phase_segment[roi_idx] = np.nan
            else:
                phase_segment[roi_idx] = np.nan
        
        phase_segments.append(phase_segment)
        trial_metadata.append({
            'trial_idx': trial_idx,
            'isi_ms': trial['isi'],
            'is_short': trial['isi'] < np.mean(df_trials['isi'].dropna()),
            'rewarded': trial.get('rewarded', False)
        })
    
    # Stack into array: (trials, rois, phase_bins)
    isi_phase_array = np.stack(phase_segments, axis=0)
    
    return isi_phase_array, trial_metadata



# def _extract_isi_phase_segments(df_trials: pd.DataFrame, 
#                                dff_clean: np.ndarray,
#                                imaging_time: np.ndarray,
#                                n_phase_bins: int) -> Tuple[np.ndarray, List[Dict]]:
#     """Extract ISI segments and resample to phase bins (0-1)"""
    
#     from scipy.interpolate import interp1d
    
#     n_rois = dff_clean.shape[0]
    
#     # Apply z-scoring per ROI
#     dff_zscore = zscore(dff_clean, axis=1)
    
#     phase_segments = []
#     trial_metadata = []
    
#     for trial_idx, trial in df_trials.iterrows():
#         if pd.isna(trial['end_flash_1']) or pd.isna(trial['start_flash_2']):
#             continue
            
#         # Get ISI period
#         isi_start_abs = trial['trial_start_timestamp'] + trial['end_flash_1']
#         isi_end_abs = trial['trial_start_timestamp'] + trial['start_flash_2']
        
#         # Find imaging indices
#         isi_start_idx = np.argmin(np.abs(imaging_time - isi_start_abs))
#         isi_end_idx = np.argmin(np.abs(imaging_time - isi_end_abs))
        
#         if isi_end_idx - isi_start_idx < 3:
#             continue
            
#         # Extract ISI segment
#         isi_segment = dff_zscore[:, isi_start_idx:isi_end_idx+1]
#         isi_times = imaging_time[isi_start_idx:isi_end_idx+1]
        
#         # Convert to phase (0 to 1)
#         isi_duration = isi_end_abs - isi_start_abs
#         isi_phases = (isi_times - isi_start_abs) / isi_duration
        
#         # Resample to fixed phase grid
#         phase_grid = np.linspace(0, 1, n_phase_bins)
#         phase_segment = np.zeros((n_rois, n_phase_bins))
        
#         for roi_idx in range(n_rois):
#             roi_trace = isi_segment[roi_idx]
#             if not np.all(np.isnan(roi_trace)):
#                 valid_mask = np.isfinite(roi_trace)
#                 if np.sum(valid_mask) >= 2:
#                     try:
#                         interp_func = interp1d(isi_phases[valid_mask], roi_trace[valid_mask],
#                                              kind='linear', bounds_error=False, fill_value='extrapolate')
#                         phase_segment[roi_idx] = interp_func(phase_grid)
#                     except:
#                         phase_segment[roi_idx] = np.nan
#                 else:
#                     phase_segment[roi_idx] = np.nan
#             else:
#                 phase_segment[roi_idx] = np.nan
        
#         phase_segments.append(phase_segment)
#         trial_metadata.append({
#             'trial_idx': trial_idx,
#             'isi_ms': trial['isi'],
#             'is_short': trial['isi'] < np.mean(df_trials['isi'].dropna()),
#             'rewarded': trial.get('rewarded', False)
#         })
    
#     # Stack into array: (trials, rois, phase_bins)
#     isi_phase_array = np.stack(phase_segments, axis=0)
    
#     return isi_phase_array, trial_metadata





def _run_cp_on_phase_data(isi_phase_array: np.ndarray, n_components: int) -> Dict[str, Any]:
    """Run CP decomposition on phase data"""
    
    try:
        import tensorly as tl
        from tensorly.decomposition import parafac
        
        print(f"Running CP decomposition with {n_components} components")
        
        # Handle NaN values by masking
        tensor = isi_phase_array.copy()
        nan_mask = np.isnan(tensor)
        tensor[nan_mask] = 0  # Replace NaN with 0 for CP
        
        # Run CP decomposition
        factors = parafac(tensor, rank=n_components, init='random', random_state=42)
        
        # Access factors correctly based on TensorLy version
        if hasattr(factors, 'factors'):
            # Newer TensorLy versions return a CPTensor object
            A = factors.factors[1]  # ROI factors
            B = factors.factors[2]  # Phase factors  
            C = factors.factors[0]  # Trial factors
        else:
            # Older versions return a tuple
            A = factors[1][1]  # ROI factors
            B = factors[1][2]  # Phase factors  
            C = factors[1][0]  # Trial factors
        
        # Calculate reconstruction using correct API
        try:
            # Try newer API first
            from tensorly.cp_tensor import cp_to_tensor
            reconstructed = cp_to_tensor(factors)
        except (ImportError, AttributeError):
            try:
                # Try alternative newer API
                from tensorly.kruskal_tensor import kruskal_to_tensor
                reconstructed = kruskal_to_tensor(factors)
            except (ImportError, AttributeError):
                try:
                    # Try legacy API
                    reconstructed = tl.kruskal_to_tensor(factors)
                except AttributeError:
                    # Manual reconstruction as fallback
                    print("Using manual reconstruction (TensorLy API compatibility)")
                    if hasattr(factors, 'factors'):
                        factor_list = factors.factors
                    else:
                        factor_list = factors[1] if isinstance(factors, tuple) else factors
                    
                    # Manual tensor reconstruction
                    reconstructed = tl.zeros_like(tensor)
                    for r in range(n_components):
                        component_tensor = tl.outer([factor_list[i][:, r] for i in range(len(factor_list))])
                        reconstructed = reconstructed + component_tensor
        
        # Calculate loss
        diff = tensor - reconstructed
        loss = tl.norm(diff) / tl.norm(tensor) if tl.norm(tensor) > 0 else 0
        
        print(f"CP decomposition completed. Loss: {loss:.6f}")
        
        return {
            'A': A,  # (n_rois, n_components)
            'B': B,  # (n_phase_bins, n_components) 
            'C': C,  # (n_trials, n_components)
            'factors': factors,
            'loss': loss,
            'rank': n_components
        }
        
    except ImportError:
        print("Tensorly not available, using NMF as fallback")
        return _run_nmf_fallback(isi_phase_array, n_components)
    except Exception as e:
        print(f"CP decomposition failed: {e}")
        print("Falling back to NMF")
        return _run_nmf_fallback(isi_phase_array, n_components)





def _run_nmf_fallback(isi_phase_array: np.ndarray, n_components: int) -> Dict[str, Any]:
    """Fallback to NMF if tensorly not available"""
    
    from sklearn.decomposition import NMF
    
    # Reshape to (trials*rois, phase_bins)
    n_trials, n_rois, n_phase_bins = isi_phase_array.shape
    data_matrix = isi_phase_array.reshape(n_trials * n_rois, n_phase_bins)
    
    # Remove NaN patterns
    valid_mask = ~np.any(np.isnan(data_matrix), axis=1)
    clean_data = data_matrix[valid_mask]
    
    # Make non-negative for NMF
    data_shifted = clean_data - np.min(clean_data) + 1e-6
    
    # Run NMF
    nmf = NMF(n_components=n_components, random_state=42)
    W = nmf.fit_transform(data_shifted)  # (patterns, components)
    H = nmf.components_  # (components, phase_bins)
    
    # Map back to ROI space (approximate)
    A = np.zeros((n_rois, n_components))
    pattern_idx = 0
    for trial_idx in range(n_trials):
        for roi_idx in range(n_rois):
            if pattern_idx < len(valid_mask) and valid_mask[pattern_idx]:
                A[roi_idx] += W[pattern_idx] / n_trials  # Average across trials
            pattern_idx += 1
    
    return {
        'A': A,
        'B': H.T,  # Transpose to match (phase_bins, components) format
        'nmf_model': nmf,
        'loss': nmf.reconstruction_err_,
        'rank': n_components
    }

def _extract_signed_groups_from_cp(A: np.ndarray, q: float = 0.10) -> List[Dict]:
    """Extract signed ROI groups from CP A matrix"""
    
    n_rois, n_components = A.shape
    signed_groups = []
    
    for comp_idx in range(n_components):
        loadings = A[:, comp_idx]
        
        # Find ROIs with strong positive/negative loadings
        pos_threshold = np.percentile(loadings, (1-q) * 100)
        neg_threshold = np.percentile(loadings, q * 100)
        
        pos_rois = np.where(loadings >= pos_threshold)[0]
        neg_rois = np.where(loadings <= neg_threshold)[0]
        
        signed_groups.append({
            'component_idx': comp_idx,
            'positive_rois': pos_rois,
            'negative_rois': neg_rois,
            'positive_weights': loadings[pos_rois],
            'negative_weights': loadings[neg_rois],
            'all_loadings': loadings
        })
    
    return signed_groups

def _orient_signed_groups_phase(signed_groups: List[Dict], 
                               cp_results: Dict[str, Any],
                               isi_phase_array: np.ndarray,
                               trial_metadata: List[Dict]) -> None:
    """Orient signed groups for consistent interpretation"""
    
    B = cp_results['B']  # Phase factors
    
    for group in signed_groups:
        comp_idx = group['component_idx']
        
        # Use phase factor to determine orientation
        phase_factor = B[:, comp_idx]
        
        # If phase factor is mostly negative, flip the sign
        if np.mean(phase_factor) < 0:
            group['positive_rois'], group['negative_rois'] = group['negative_rois'], group['positive_rois']
            group['positive_weights'], group['negative_weights'] = -group['negative_weights'], -group['positive_weights']
            group['flipped'] = True
        else:
            group['flipped'] = False

def visualize_phase_cp_results(phase_results: Dict[str, Any]):
    """Visualize the phase CP results"""
    
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    phase_bins = phase_results['phase_bins']
    
    A = cp_results['A']  # ROI loadings
    B = cp_results['B']  # Phase factors
    
    n_components = A.shape[1]
    n_show = min(6, n_components)
    
    fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 8))
    if n_show == 1:
        axes = axes.reshape(-1, 1)
    
    for comp_idx in range(n_show):
        # Top: Phase temporal factor
        ax_top = axes[0, comp_idx]
        phase_factor = B[:, comp_idx]
        
        ax_top.plot(phase_bins * 100, phase_factor, 'b-', linewidth=2)
        ax_top.set_title(f'Component {comp_idx}\nPhase Factor')
        ax_top.set_xlabel('ISI Phase (%)')
        ax_top.set_ylabel('Factor Weight')
        ax_top.grid(True, alpha=0.3)
        ax_top.axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        # Bottom: ROI signed groups
        ax_bottom = axes[1, comp_idx]
        group = signed_groups[comp_idx]
        
        # Show ROI loadings
        all_loadings = group['all_loadings']
        pos_rois = group['positive_rois']
        neg_rois = group['negative_rois']
        
        ax_bottom.scatter(range(len(all_loadings)), all_loadings, alpha=0.5, s=2, color='gray')
        
        if len(pos_rois) > 0:
            ax_bottom.scatter(pos_rois, all_loadings[pos_rois], color='red', s=10, 
                            label=f'Pos ROIs (n={len(pos_rois)})')
        
        if len(neg_rois) > 0:
            ax_bottom.scatter(neg_rois, all_loadings[neg_rois], color='blue', s=10,
                            label=f'Neg ROIs (n={len(neg_rois)})')
        
        ax_bottom.set_title(f'Component {comp_idx}\nROI Loadings')
        ax_bottom.set_xlabel('ROI Index')
        ax_bottom.set_ylabel('Loading Weight')
        ax_bottom.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax_bottom.legend()
        ax_bottom.grid(True, alpha=0.3)
    
    plt.suptitle('ISI Phase CP Decomposition Results', fontsize=16)
    plt.tight_layout()
    plt.show()
















def analyze_cp_roi_assignments(phase_results: Dict[str, Any], 
                              show_loading_distributions: bool = True,
                              show_roi_overlap: bool = True) -> None:
    """Analyze the quality and exclusivity of ROI assignments from CP"""
    
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    
    A = cp_results['A']  # (n_rois, n_components)
    n_rois, n_components = A.shape
    
    print(f"\n=== CP ROI ASSIGNMENT ANALYSIS ===")
    print(f"Total ROIs: {n_rois}")
    print(f"Components: {n_components}")
    
    # 1. Analyze loading distributions
    if show_loading_distributions:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for comp_idx in range(min(6, n_components)):
            ax = axes[comp_idx]
            loadings = A[:, comp_idx]
            
            # Histogram of loadings
            ax.hist(loadings, bins=50, alpha=0.7, density=True)
            ax.axvline(np.percentile(loadings, 10), color='blue', linestyle='--', 
                      label=f'10th %ile: {np.percentile(loadings, 10):.3f}')
            ax.axvline(np.percentile(loadings, 90), color='red', linestyle='--',
                      label=f'90th %ile: {np.percentile(loadings, 90):.3f}')
            ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
            
            ax.set_title(f'Component {comp_idx}\nLoading Distribution')
            ax.set_xlabel('Loading Weight')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 2. Check ROI overlap between components
    if show_roi_overlap:
        print(f"\n=== ROI OVERLAP ANALYSIS ===")
        
        # Count how many components each ROI belongs to
        roi_component_counts = np.zeros(n_rois)
        roi_assignments = {}  # ROI -> list of components
        
        for comp_idx, group in enumerate(signed_groups):
            pos_rois = group['positive_rois']
            neg_rois = group['negative_rois']
            all_comp_rois = np.concatenate([pos_rois, neg_rois])
            
            for roi in all_comp_rois:
                roi_component_counts[roi] += 1
                if roi not in roi_assignments:
                    roi_assignments[roi] = []
                roi_assignments[roi].append(comp_idx)
            
            print(f"Component {comp_idx}: {len(pos_rois)} pos + {len(neg_rois)} neg = {len(all_comp_rois)} total")
        
        # Show overlap statistics
        unique_counts, count_frequencies = np.unique(roi_component_counts, return_counts=True)
        print(f"\nROI Assignment Counts:")
        for count, freq in zip(unique_counts, count_frequencies):
            if count > 0:
                print(f"  {freq} ROIs belong to {count:.0f} component(s)")
        
        # Show highly overlapping ROIs
        multi_component_rois = np.where(roi_component_counts > 1)[0]
        if len(multi_component_rois) > 0:
            print(f"\nROIs belonging to multiple components (first 10):")
            for roi in multi_component_rois[:10]:
                components = roi_assignments[roi]
                loadings = [A[roi, comp] for comp in components]
                print(f"  ROI {roi}: components {components}, loadings {loadings}")

def validate_component_temporal_patterns(phase_results: Dict[str, Any],
                                       component_idx: int = 0) -> None:
    """Validate that assigned ROIs actually show the component's temporal pattern"""
    
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    isi_phase_array = phase_results['isi_phase_array']
    phase_bins = phase_results['phase_bins']
    
    A = cp_results['A']
    B = cp_results['B']
    group = signed_groups[component_idx]
    
    print(f"\n=== VALIDATING COMPONENT {component_idx} TEMPORAL PATTERN ===")
    
    # Get component's temporal pattern
    component_phase_pattern = B[:, component_idx]
    
    # Get assigned ROIs
    pos_rois = group['positive_rois']
    neg_rois = group['negative_rois']
    
    print(f"Positive ROIs: {len(pos_rois)}")
    print(f"Negative ROIs: {len(neg_rois)}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Top left: Component temporal pattern
    axes[0,0].plot(phase_bins * 100, component_phase_pattern, 'k-', linewidth=2)
    axes[0,0].set_title(f'Component {component_idx} Phase Pattern')
    axes[0,0].set_xlabel('ISI Phase (%)')
    axes[0,0].set_ylabel('Component Weight')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    # Top right: Positive ROI average
    if len(pos_rois) > 0:
        pos_roi_traces = isi_phase_array[:, pos_rois, :]  # (trials, pos_rois, phase_bins)
        pos_roi_mean = np.nanmean(pos_roi_traces, axis=(0,1))  # Average across trials and ROIs
        
        axes[0,1].plot(phase_bins * 100, pos_roi_mean, 'r-', linewidth=2, label='Actual Data')
        axes[0,1].plot(phase_bins * 100, component_phase_pattern, 'k--', alpha=0.7, label='Component Pattern')
        axes[0,1].set_title(f'Positive ROIs Average (n={len(pos_rois)})')
        axes[0,1].set_xlabel('ISI Phase (%)')
        axes[0,1].set_ylabel('z-scored dF/F')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        # Calculate correlation
        valid_mask = np.isfinite(pos_roi_mean) & np.isfinite(component_phase_pattern)
        if np.sum(valid_mask) > 3:
            corr = np.corrcoef(pos_roi_mean[valid_mask], component_phase_pattern[valid_mask])[0,1]
            axes[0,1].text(0.05, 0.95, f'Corr: {corr:.3f}', transform=axes[0,1].transAxes, 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Bottom right: Negative ROI average
    if len(neg_rois) > 0:
        neg_roi_traces = isi_phase_array[:, neg_rois, :]
        neg_roi_mean = np.nanmean(neg_roi_traces, axis=(0,1))
        
        axes[1,1].plot(phase_bins * 100, neg_roi_mean, 'b-', linewidth=2, label='Actual Data')
        axes[1,1].plot(phase_bins * 100, -component_phase_pattern, 'k--', alpha=0.7, label='Inverted Component')
        axes[1,1].set_title(f'Negative ROIs Average (n={len(neg_rois)})')
        axes[1,1].set_xlabel('ISI Phase (%)')
        axes[1,1].set_ylabel('z-scored dF/F')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        # Calculate correlation with inverted pattern
        valid_mask = np.isfinite(neg_roi_mean) & np.isfinite(component_phase_pattern)
        if np.sum(valid_mask) > 3:
            corr = np.corrcoef(neg_roi_mean[valid_mask], -component_phase_pattern[valid_mask])[0,1]
            axes[1,1].text(0.05, 0.95, f'Corr: {corr:.3f}', transform=axes[1,1].transAxes,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Bottom left: Loading distribution for this component
    axes[1,0].scatter(range(len(A[:, component_idx])), A[:, component_idx], alpha=0.5, s=2, color='gray')
    if len(pos_rois) > 0:
        axes[1,0].scatter(pos_rois, A[pos_rois, component_idx], color='red', s=10, 
                         label=f'Positive (n={len(pos_rois)})')
    if len(neg_rois) > 0:
        axes[1,0].scatter(neg_rois, A[neg_rois, component_idx], color='blue', s=10,
                         label=f'Negative (n={len(neg_rois)})')
    
    axes[1,0].set_title(f'Component {component_idx} ROI Loadings')
    axes[1,0].set_xlabel('ROI Index')
    axes[1,0].set_ylabel('Loading Weight')
    axes[1,0].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Add this to your analysis pipeline





























def analyze_ALL_cp_roi_assignments(phase_results: Dict[str, Any]) -> None:
    """Analyze ALL components, not just a few"""
    
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    
    A = cp_results['A']  # (n_rois, n_components)
    n_rois, n_components = A.shape
    
    print(f"\n=== COMPREHENSIVE CP ROI ASSIGNMENT ANALYSIS ===")
    print(f"Total ROIs: {n_rois}")
    print(f"Components: {n_components}")
    print(f"Analyzing ALL {n_components} components...")
    
    # 1. Loading distributions for ALL components
    n_cols = 5
    n_rows = int(np.ceil(n_components / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten() if n_components > 1 else [axes]
    
    for comp_idx in range(n_components):
        ax = axes[comp_idx]
        loadings = A[:, comp_idx]
        
        # Histogram of loadings
        ax.hist(loadings, bins=50, alpha=0.7, density=True)
        ax.axvline(np.percentile(loadings, 10), color='blue', linestyle='--', 
                  label=f'10th: {np.percentile(loadings, 10):.3f}')
        ax.axvline(np.percentile(loadings, 90), color='red', linestyle='--',
                  label=f'90th: {np.percentile(loadings, 90):.3f}')
        ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
        
        ax.set_title(f'Comp {comp_idx}')
        ax.set_xlabel('Loading')
        ax.set_ylabel('Density')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_components, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'ALL {n_components} Component Loading Distributions', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 2. Comprehensive overlap analysis
    print(f"\n=== COMPREHENSIVE ROI OVERLAP ANALYSIS ===")
    
    roi_component_counts = np.zeros(n_rois)
    roi_assignments = {}  # ROI -> list of components
    component_sizes = []
    
    for comp_idx, group in enumerate(signed_groups):
        pos_rois = group['positive_rois']
        neg_rois = group['negative_rois']
        all_comp_rois = np.concatenate([pos_rois, neg_rois])
        
        component_sizes.append(len(all_comp_rois))
        
        for roi in all_comp_rois:
            roi_component_counts[roi] += 1
            if roi not in roi_assignments:
                roi_assignments[roi] = []
            roi_assignments[roi].append(comp_idx)
        
        print(f"Component {comp_idx:2d}: {len(pos_rois):3d} pos + {len(neg_rois):3d} neg = {len(all_comp_rois):3d} total")
    
    # Component size distribution
    print(f"\nComponent size statistics:")
    print(f"  Mean: {np.mean(component_sizes):.1f} ROIs")
    print(f"  Std:  {np.std(component_sizes):.1f} ROIs")
    print(f"  Min:  {np.min(component_sizes)} ROIs")
    print(f"  Max:  {np.max(component_sizes)} ROIs")
    
    # ROI overlap statistics
    unique_counts, count_frequencies = np.unique(roi_component_counts, return_counts=True)
    print(f"\nROI Assignment Distribution:")
    for count, freq in zip(unique_counts, count_frequencies):
        pct = 100 * freq / n_rois
        if count == 0:
            print(f"  {freq:4d} ROIs ({pct:5.1f}%) belong to NO components")
        elif count > 0:
            print(f"  {freq:4d} ROIs ({pct:5.1f}%) belong to {count:.0f} component(s)")
    
    # Show most promiscuous ROIs
    multi_component_rois = np.where(roi_component_counts > 1)[0]
    if len(multi_component_rois) > 0:
        print(f"\nMost promiscuous ROIs (belong to multiple components):")
        # Sort by number of components
        roi_comp_counts_multi = roi_component_counts[multi_component_rois]
        sorted_indices = np.argsort(roi_comp_counts_multi)[::-1]  # Descending
        
        for i in sorted_indices[:20]:  # Show top 20
            roi = multi_component_rois[i]
            components = roi_assignments[roi]
            loadings = [f"{A[roi, comp]:.3f}" for comp in components]
            print(f"  ROI {roi:3d}: {len(components)} comps {components}, loadings {loadings}")

def validate_ALL_component_temporal_patterns(phase_results: Dict[str, Any]) -> None:
    """Validate ALL components, not just a few"""
    
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    isi_phase_array = phase_results['isi_phase_array']
    phase_bins = phase_results['phase_bins']
    
    A = cp_results['A']
    B = cp_results['B']
    n_components = len(signed_groups)
    
    print(f"\n=== VALIDATING ALL {n_components} COMPONENT TEMPORAL PATTERNS ===")
    
    # Create correlation summary
    correlations_pos = []
    correlations_neg = []
    
    n_cols = 4
    n_rows = int(np.ceil(n_components / n_cols))
    
    # Phase patterns
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes1 = axes1.flatten() if n_components > 1 else [axes1]
    
    # ROI validations (separate figure due to complexity)
    for comp_idx in range(n_components):
        group = signed_groups[comp_idx]
        component_phase_pattern = B[:, comp_idx]
        pos_rois = group['positive_rois']
        neg_rois = group['negative_rois']
        
        # Phase pattern plot
        ax = axes1[comp_idx]
        ax.plot(phase_bins * 100, component_phase_pattern, 'k-', linewidth=2)
        ax.set_title(f'Comp {comp_idx}\n{len(pos_rois)}+/{len(neg_rois)}-')
        ax.set_xlabel('ISI Phase (%)')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        # Calculate correlations
        pos_corr = np.nan
        neg_corr = np.nan
        
        if len(pos_rois) > 0:
            pos_roi_traces = isi_phase_array[:, pos_rois, :]
            pos_roi_mean = np.nanmean(pos_roi_traces, axis=(0,1))
            valid_mask = np.isfinite(pos_roi_mean) & np.isfinite(component_phase_pattern)
            if np.sum(valid_mask) > 3:
                pos_corr = np.corrcoef(pos_roi_mean[valid_mask], component_phase_pattern[valid_mask])[0,1]
        
        if len(neg_rois) > 0:
            neg_roi_traces = isi_phase_array[:, neg_rois, :]
            neg_roi_mean = np.nanmean(neg_roi_traces, axis=(0,1))
            valid_mask = np.isfinite(neg_roi_mean) & np.isfinite(component_phase_pattern)
            if np.sum(valid_mask) > 3:
                neg_corr = np.corrcoef(neg_roi_mean[valid_mask], -component_phase_pattern[valid_mask])[0,1]
        
        correlations_pos.append(pos_corr)
        correlations_neg.append(neg_corr)
        
        # Add correlation to plot
        ax.text(0.02, 0.98, f'pos:{pos_corr:.3f}\nneg:{neg_corr:.3f}', 
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Hide empty subplots
    for i in range(n_components, len(axes1)):
        axes1[i].set_visible(False)
    
    plt.suptitle(f'ALL {n_components} Component Phase Patterns with Correlations', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Correlation summary
    correlations_pos = np.array(correlations_pos)
    correlations_neg = np.array(correlations_neg)
    
    print(f"\n=== CORRELATION SUMMARY ===")
    print(f"Positive ROI correlations:")
    print(f"  Valid: {np.sum(np.isfinite(correlations_pos))}/{n_components}")
    if np.sum(np.isfinite(correlations_pos)) > 0:
        print(f"  Mean: {np.nanmean(correlations_pos):.3f}")
        print(f"  Std:  {np.nanstd(correlations_pos):.3f}")
        print(f"  Range: {np.nanmin(correlations_pos):.3f} to {np.nanmax(correlations_pos):.3f}")
    
    print(f"Negative ROI correlations:")
    print(f"  Valid: {np.sum(np.isfinite(correlations_neg))}/{n_components}")
    if np.sum(np.isfinite(correlations_neg)) > 0:
        print(f"  Mean: {np.nanmean(correlations_neg):.3f}")
        print(f"  Std:  {np.nanstd(correlations_neg):.3f}")
        print(f"  Range: {np.nanmin(correlations_neg):.3f} to {np.nanmax(correlations_neg):.3f}")
    
    # Find best components
    valid_pos = np.isfinite(correlations_pos)
    valid_neg = np.isfinite(correlations_neg)
    
    if np.any(valid_pos):
        best_pos_idx = np.nanargmax(correlations_pos)
        print(f"\nBest positive correlation: Component {best_pos_idx} (r={correlations_pos[best_pos_idx]:.3f})")
    
    if np.any(valid_neg):
        best_neg_idx = np.nanargmax(correlations_neg)
        print(f"Best negative correlation: Component {best_neg_idx} (r={correlations_neg[best_neg_idx]:.3f})")

def visualize_ALL_component_isi_traces(phase_results: Dict[str, Any], 
                                     data: Dict[str, Any]) -> None:
    """
    Visualize ISI traces for ALL components, with ISIs sorted properly
    """
    print(f"\n=== ALL COMPONENT ISI TRACE VISUALIZATION ===")
    
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    
    # Get ISI data
    df_trials = data['df_trials']
    dff_clean = data['dFF_clean']
    imaging_time = data['imaging_time']
    
    # Extract ISI segments
    isi_traces, isi_time_vector, trial_info = _extract_isi_absolute_segments(
        df_trials, dff_clean, imaging_time
    )
    
    # Get unique ISI values and SORT them properly
    all_isis = [info['isi_ms'] for info in trial_info]
    unique_isis = sorted(list(set(all_isis)))  # SORTED!
    
    print(f"Found {len(unique_isis)} unique ISI values: {unique_isis}")
    
    n_components = len(signed_groups)
    n_isis = len(unique_isis)
    
    # Create one big plot with ALL components and ALL ISIs
    fig, axes = plt.subplots(n_components, n_isis + 1, figsize=(2*(n_isis+1), 2*n_components))
    
    if n_components == 1:
        axes = axes.reshape(1, -1)
    if n_isis == 0:
        print("No ISI data found!")
        return
    
    for comp_idx in range(n_components):
        group = signed_groups[comp_idx]
        component_rois = np.concatenate([group['positive_rois'], group['negative_rois']])
        
        if len(component_rois) == 0:
            continue
        
        # Calculate component activity
        component_traces = _calculate_component_activity(
            isi_traces, component_rois, group['all_loadings']
        )
        
        # Column 0: All trials
        ax = axes[comp_idx, 0]
        _plot_trial_traces(ax, isi_time_vector, component_traces, 
                          title=f'C{comp_idx} All',
                          color='black')
        
        # Columns 1+: Each ISI value (SORTED)
        for isi_idx, target_isi in enumerate(unique_isis):
            ax = axes[comp_idx, isi_idx + 1]
            
            # Find trials with this ISI
            isi_mask = np.array([info['isi_ms'] == target_isi for info in trial_info])
            
            if np.sum(isi_mask) > 0:
                isi_traces_subset = component_traces[isi_mask]
                _plot_trial_traces(ax, isi_time_vector, isi_traces_subset,
                                  title=f'{target_isi:.0f}ms\n(n={np.sum(isi_mask)})',
                                  color='blue')
                
                # Add ISI duration marker
                if len(trial_info) > 0:
                    # Get actual duration for this ISI
                    isi_durations = [info['isi_duration'] for i, info in enumerate(trial_info) if isi_mask[i]]
                    if isi_durations:
                        mean_duration = np.mean(isi_durations)
                        ax.axvline(mean_duration, color='orange', linestyle='--', alpha=0.7, linewidth=1)
            else:
                ax.set_title(f'{target_isi:.0f}ms\n(n=0)')
        
        # Add component info to leftmost plot
        axes[comp_idx, 0].text(0.02, 0.98, 
                              f'{len(group["positive_rois"])}+/{len(group["negative_rois"])}-', 
                              transform=axes[comp_idx, 0].transAxes, va='top', fontsize=8,
                              bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    plt.suptitle(f'ALL {n_components} Components × ALL {n_isis} ISI Values (SORTED)', fontsize=16)
    plt.tight_layout()
    plt.show()

def comprehensive_component_validation(phase_results: Dict[str, Any], 
                                     data: Dict[str, Any]) -> None:
    """Run ALL validation checks in sequence"""
    
    print("=" * 60)
    print("COMPREHENSIVE COMPONENT VALIDATION")
    print("=" * 60)
    
    # 1. ROI assignment analysis
    analyze_ALL_cp_roi_assignments(phase_results)
    
    # 2. Temporal pattern validation
    validate_ALL_component_temporal_patterns(phase_results)
    
    # 3. ISI trace visualization
    visualize_ALL_component_isi_traces(phase_results, data)
    
    # 4. Summary statistics
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    n_components = len(signed_groups)
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total components analyzed: {n_components}")
    
    # Count components with good ROI assignments
    good_components = 0
    for group in signed_groups:
        if len(group['positive_rois']) > 0 or len(group['negative_rois']) > 0:
            good_components += 1
    
    print(f"Components with ROI assignments: {good_components}/{n_components}")
    print(f"CP decomposition loss: {cp_results.get('loss', 'N/A')}")
    print(f"Analysis complete!")








def _extract_isi_absolute_segments_sorted(df_trials: pd.DataFrame, 
                                        dff_clean: np.ndarray,
                                        imaging_time: np.ndarray,
                                        max_isi_duration: float = 8.0) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Extract ISI segments with proper ISI sorting"""
    
    from scipy.interpolate import interp1d
    
    # Apply z-scoring per ROI
    dff_zscore = zscore(dff_clean, axis=1)
    
    # Create fixed time vector for ISI period
    dt = 1.0 / 30.0  # 30Hz sampling  
    isi_time_vector = np.arange(0, max_isi_duration + dt, dt)
    
    n_rois = dff_clean.shape[0]
    isi_segments = []
    trial_info = []
    
    # First pass: collect all valid trials
    valid_trials = []
    for trial_idx, trial in df_trials.iterrows():
        if pd.isna(trial['end_flash_1']) or pd.isna(trial['start_flash_2']):
            continue
            
        isi_start_abs = trial['trial_start_timestamp'] + trial['end_flash_1']
        isi_end_abs = trial['trial_start_timestamp'] + trial['start_flash_2']
        isi_duration = isi_end_abs - isi_start_abs
        
        if isi_duration > max_isi_duration:
            continue
            
        isi_start_idx = np.argmin(np.abs(imaging_time - isi_start_abs))
        isi_end_idx = np.argmin(np.abs(imaging_time - isi_end_abs))
        
        if isi_end_idx - isi_start_idx < 3:
            continue
            
        valid_trials.append({
            'trial_idx': trial_idx,
            'trial': trial,
            'isi_ms': trial['isi'],
            'isi_duration': isi_duration,
            'isi_start_abs': isi_start_abs,
            'isi_end_abs': isi_end_abs,
            'isi_start_idx': isi_start_idx,
            'isi_end_idx': isi_end_idx
        })
    
    # Sort by ISI duration (PROPER SORTING!)
    valid_trials.sort(key=lambda x: x['isi_ms'])
    
    print(f"Processing {len(valid_trials)} valid trials, sorted by ISI")
    
    # Second pass: extract in sorted order
    for trial_data in valid_trials:
        trial = trial_data['trial']
        isi_start_idx = trial_data['isi_start_idx']
        isi_end_idx = trial_data['isi_end_idx']
        
        # Extract ISI segment
        isi_segment = dff_zscore[:, isi_start_idx:isi_end_idx+1]
        segment_times = imaging_time[isi_start_idx:isi_end_idx+1]
        relative_times = segment_times - trial_data['isi_start_abs']
        
        # Interpolate to fixed time grid
        interpolated_segment = np.full((n_rois, len(isi_time_vector)), np.nan)
        
        for roi_idx in range(n_rois):
            roi_trace = isi_segment[roi_idx]
            if not np.all(np.isnan(roi_trace)):
                valid_mask = np.isfinite(roi_trace)
                if np.sum(valid_mask) >= 2:
                    try:
                        valid_time_mask = isi_time_vector <= trial_data['isi_duration']
                        interp_func = interp1d(relative_times[valid_mask], roi_trace[valid_mask],
                                             kind='linear', bounds_error=False, fill_value=np.nan)
                        interpolated_segment[roi_idx, valid_time_mask] = interp_func(isi_time_vector[valid_time_mask])
                    except:
                        pass
        
        isi_segments.append(interpolated_segment)
        trial_info.append({
            'trial_idx': trial_data['trial_idx'],
            'isi_ms': trial_data['isi_ms'],
            'isi_duration': trial_data['isi_duration'],
            'is_short': trial['isi'] < np.mean(df_trials['isi'].dropna())
        })
    
    # Stack into array: (trials, rois, time) - NOW SORTED BY ISI
    isi_traces = np.stack(isi_segments, axis=0)
    
    isis_sorted = [info['isi_ms'] for info in trial_info]
    print(f"ISI order check: {isis_sorted[:10]}... (first 10)")
    
    return isi_traces, isi_time_vector, trial_info




















def visualize_ALL_component_isi_traces_proper(phase_results: Dict[str, Any], 
                                            data: Dict[str, Any]) -> None:
    """
    Visualize ALL components with proper full trial traces aligned to F1 start
    Uses multiple figures for better visibility
    """
    print(f"\n=== ALL COMPONENT FULL TRIAL VISUALIZATION ===")
    
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    
    # Get data
    df_trials = data['df_trials']
    dff_clean = data['dFF_clean']
    imaging_time = data['imaging_time']
    
    # Get unique ISI values and SORT them properly
    all_isis = df_trials['isi'].dropna().values
    unique_isis = sorted(list(set(all_isis)))  # SORTED!
    
    print(f"Found {len(unique_isis)} unique ISI values: {unique_isis}")
    
    n_components = len(signed_groups)
    n_isis = len(unique_isis)
    
    # Use multiple figures - 6 components per figure
    components_per_figure = 6
    n_figures = int(np.ceil(n_components / components_per_figure))
    
    for fig_idx in range(n_figures):
        start_comp = fig_idx * components_per_figure
        end_comp = min(start_comp + components_per_figure, n_components)
        n_comps_this_fig = end_comp - start_comp
        
        # Create figure with proper size
        fig, axes = plt.subplots(n_comps_this_fig, n_isis + 1, 
                                figsize=(3*(n_isis+1), 4*n_comps_this_fig))
        
        if n_comps_this_fig == 1:
            axes = axes.reshape(1, -1)
        if n_isis == 0:
            print("No ISI data found!")
            continue
        
        for comp_offset, comp_idx in enumerate(range(start_comp, end_comp)):
            group = signed_groups[comp_idx]
            component_rois = np.concatenate([group['positive_rois'], group['negative_rois']])
            
            if len(component_rois) == 0:
                continue
            
            # Extract FULL trial traces aligned to F1 start
            full_trial_traces, full_time_vector, full_trial_info = _extract_full_trial_traces_f1_aligned(
                df_trials, dff_clean, imaging_time, component_rois, group['all_loadings']
            )
            
            # Column 0: All trials
            ax = axes[comp_offset, 0]
            _plot_trial_traces_with_events(ax, full_time_vector, full_trial_traces, 
                                         full_trial_info, None,
                                         title=f'C{comp_idx} All (n={len(full_trial_traces)})',
                                         color='black')
            
            # Columns 1+: Each ISI value (SORTED)
            for isi_idx, target_isi in enumerate(unique_isis):
                ax = axes[comp_offset, isi_idx + 1]
                
                # Find trials with this ISI
                isi_mask = np.array([info['isi_ms'] == target_isi for info in full_trial_info])
                
                if np.sum(isi_mask) > 0:
                    isi_traces_subset = full_trial_traces[isi_mask]
                    isi_info_subset = [info for i, info in enumerate(full_trial_info) if isi_mask[i]]
                    
                    _plot_trial_traces_with_events(ax, full_time_vector, isi_traces_subset,
                                                 isi_info_subset, isi_mask,
                                                 title=f'{target_isi:.0f}ms (n={np.sum(isi_mask)})',
                                                 color='blue')
                else:
                    ax.set_title(f'{target_isi:.0f}ms (n=0)')
                    ax.set_xlabel('Time from F1 Start (s)')
                    ax.set_ylabel('Component Activity')
            
            # Add component info to leftmost plot
            axes[comp_offset, 0].text(0.02, 0.98, 
                                    f'{len(group["positive_rois"])}+/{len(group["negative_rois"])}-', 
                                    transform=axes[comp_offset, 0].transAxes, va='top', fontsize=8,
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        plt.suptitle(f'Components {start_comp}-{end_comp-1}: Full Trial Traces Aligned to F1 Start', fontsize=16)
        plt.tight_layout()
        plt.show()

# def _extract_full_trial_traces_f1_aligned(df_trials: pd.DataFrame,
#                                         dff_clean: np.ndarray,
#                                         imaging_time: np.ndarray,
#                                         component_rois: np.ndarray,
#                                         roi_loadings: np.ndarray,
#                                         pre_f1_s: float = 1.0,
#                                         post_f1_s: float = 8.0) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
#     """Extract full trial traces aligned to F1 start"""
    
#     # Apply z-scoring per ROI
#     dff_zscore = zscore(dff_clean, axis=1)
    
#     # Create fixed time vector relative to F1 start
#     dt = 1.0 / 30.0  # 30Hz sampling
#     time_vector = np.arange(-pre_f1_s, post_f1_s + dt, dt)
    
#     # Calculate component weights
#     weights = np.abs(roi_loadings[component_rois])
#     weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights)
    
#     trial_traces = []
#     trial_info = []
    
#     for trial_idx, trial in df_trials.iterrows():
#         if pd.isna(trial['start_flash_1']):
#             continue
        
#         # Get F1 start time
#         f1_start_abs = trial['trial_start_timestamp'] + trial['start_flash_1']
        
#         # Define extraction window
#         extract_start_abs = f1_start_abs - pre_f1_s
#         extract_end_abs = f1_start_abs + post_f1_s
        
#         # Find imaging indices
#         start_idx = np.argmin(np.abs(imaging_time - extract_start_abs))
#         end_idx = np.argmin(np.abs(imaging_time - extract_end_abs))
        
#         if end_idx - start_idx < 10:  # Need at least 10 samples
#             continue
        
#         # Extract component traces
#         component_traces = dff_zscore[component_rois, start_idx:end_idx+1]
#         segment_times = imaging_time[start_idx:end_idx+1]
#         relative_times = segment_times - f1_start_abs  # Relative to F1 start
        
#         # Calculate weighted average across component ROIs
#         if len(component_rois) > 0:
#             weighted_trace = np.average(component_traces, axis=0, weights=weights)
#         else:
#             weighted_trace = np.zeros(end_idx - start_idx + 1)
        
#         # Interpolate to fixed time grid
#         from scipy.interpolate import interp1d
        
#         try:
#             valid_mask = np.isfinite(weighted_trace) & np.isfinite(relative_times)
#             if np.sum(valid_mask) >= 2:
#                 interp_func = interp1d(relative_times[valid_mask], weighted_trace[valid_mask],
#                                      kind='linear', bounds_error=False, fill_value=np.nan)
#                 interpolated_trace = interp_func(time_vector)
#             else:
#                 interpolated_trace = np.full_like(time_vector, np.nan)
#         except:
#             interpolated_trace = np.full_like(time_vector, np.nan)
        
#         trial_traces.append(interpolated_trace)
        
#         # Store trial info with event times relative to F1 start
#         trial_info.append({
#             'trial_idx': trial_idx,
#             'isi_ms': trial['isi'],
#             'is_short': trial['isi'] < np.mean(df_trials['isi'].dropna()),
#             'f1_start': 0.0,  # F1 start is our reference (t=0)
#             'f1_end': trial.get('end_flash_1', np.nan) - trial.get('start_flash_1', np.nan) if not pd.isna(trial.get('end_flash_1')) else np.nan,
#             'f2_start': trial.get('start_flash_2', np.nan) - trial.get('start_flash_1', np.nan) if not pd.isna(trial.get('start_flash_2')) else np.nan,
#             'f2_end': trial.get('end_flash_2', np.nan) - trial.get('start_flash_1', np.nan) if not pd.isna(trial.get('end_flash_2')) else np.nan,
#             'choice_start': trial.get('choice_start', np.nan) - trial.get('start_flash_1', np.nan) if not pd.isna(trial.get('choice_start')) else np.nan,
#             'lick_start': trial.get('lick_start', np.nan) - trial.get('start_flash_1', np.nan) if not pd.isna(trial.get('lick_start')) else np.nan,
#         })
    
#     if len(trial_traces) == 0:
#         return np.array([]), np.array([]), []
    
#     # Stack into array: (trials, time)
#     trial_traces = np.stack(trial_traces, axis=0)
    
#     print(f"Extracted {len(trial_traces)} full trial traces aligned to F1 start")
#     print(f"Time vector: {len(time_vector)} samples, {time_vector[0]:.1f} to {time_vector[-1]:.1f}s")
    
#     return trial_traces, time_vector, trial_info



def _extract_full_trial_traces_f1_aligned(df_trials: pd.DataFrame,
                                        dff_clean: np.ndarray,
                                        imaging_time: np.ndarray,
                                        component_rois: np.ndarray,
                                        roi_loadings: np.ndarray,
                                        pre_f1_s: float = 1.0,
                                        post_f1_s: float = 8.0,
                                        apply_zscore: bool = False) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Extract full trial traces aligned to F1 start"""
    
    # Apply z-scoring per ROI
    # dff_zscore = zscore(dff_clean, axis=1)
    
    # OPTIONAL z-scoring per ROI
    if apply_zscore:
        dff_processed = zscore(dff_clean, axis=1)
        print("Applied z-score normalization per ROI")
    else:
        dff_processed = dff_clean
        print("Using original dF/F (no z-scoring)")    
    
    # Create fixed time vector relative to F1 start
    dt = 1.0 / 30.0  # 30Hz sampling
    time_vector = np.arange(-pre_f1_s, post_f1_s + dt, dt)
    
    # FIX: Calculate component weights properly
    # roi_loadings should be the full loadings array for the component
    # component_rois are the indices of ROIs in this component
    
    print(f"DEBUG: component_rois shape: {component_rois.shape}, roi_loadings shape: {roi_loadings.shape}")
    print(f"DEBUG: component_rois range: {component_rois.min() if len(component_rois) > 0 else 'empty'} to {component_rois.max() if len(component_rois) > 0 else 'empty'}")
    print(f"DEBUG: roi_loadings is full component loadings, need to index properly")
    
    # Check if roi_loadings is the full component vector or just the subset
    if len(roi_loadings) == len(component_rois):
        # roi_loadings is already the subset for component_rois
        weights = np.abs(roi_loadings)
    else:
        # roi_loadings is the full component vector, need to index
        weights = np.abs(roi_loadings[component_rois])
    
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights)
    
    trial_traces = []
    trial_info = []
    
    for trial_idx, trial in df_trials.iterrows():
        if pd.isna(trial['start_flash_1']):
            continue
        
        # Get F1 start time
        f1_start_abs = trial['trial_start_timestamp'] + trial['start_flash_1']
        
        # Define extraction window
        extract_start_abs = f1_start_abs - pre_f1_s
        extract_end_abs = f1_start_abs + post_f1_s
        
        # Find imaging indices
        start_idx = np.argmin(np.abs(imaging_time - extract_start_abs))
        end_idx = np.argmin(np.abs(imaging_time - extract_end_abs))
        
        if end_idx - start_idx < 10:  # Need at least 10 samples
            continue
        
        # Extract component traces
        component_traces = dff_processed[component_rois, start_idx:end_idx+1]
        segment_times = imaging_time[start_idx:end_idx+1]
        relative_times = segment_times - f1_start_abs  # Relative to F1 start
        
        # Calculate weighted average across component ROIs
        if len(component_rois) > 0:
            weighted_trace = np.average(component_traces, axis=0, weights=weights)
        else:
            weighted_trace = np.zeros(end_idx - start_idx + 1)
        
        # Interpolate to fixed time grid
        from scipy.interpolate import interp1d
        
        try:
            valid_mask = np.isfinite(weighted_trace) & np.isfinite(relative_times)
            if np.sum(valid_mask) >= 2:
                interp_func = interp1d(relative_times[valid_mask], weighted_trace[valid_mask],
                                     kind='linear', bounds_error=False, fill_value=np.nan)
                interpolated_trace = interp_func(time_vector)
            else:
                interpolated_trace = np.full_like(time_vector, np.nan)
        except:
            interpolated_trace = np.full_like(time_vector, np.nan)
        
        trial_traces.append(interpolated_trace)
        
        # Store trial info with event times relative to F1 start
        trial_info.append({
            'trial_idx': trial_idx,
            'isi_ms': trial['isi'],
            'is_short': trial['isi'] < np.mean(df_trials['isi'].dropna()),
            'f1_start': 0.0,  # F1 start is our reference (t=0)
            'f1_end': trial.get('end_flash_1', np.nan) - trial.get('start_flash_1', np.nan) if not pd.isna(trial.get('end_flash_1')) else np.nan,
            'f2_start': trial.get('start_flash_2', np.nan) - trial.get('start_flash_1', np.nan) if not pd.isna(trial.get('start_flash_2')) else np.nan,
            'f2_end': trial.get('end_flash_2', np.nan) - trial.get('start_flash_1', np.nan) if not pd.isna(trial.get('end_flash_2')) else np.nan,
            'choice_start': trial.get('choice_start', np.nan) - trial.get('start_flash_1', np.nan) if not pd.isna(trial.get('choice_start')) else np.nan,
            'lick_start': trial.get('lick_start', np.nan) - trial.get('start_flash_1', np.nan) if not pd.isna(trial.get('lick_start')) else np.nan,
        })
    
    if len(trial_traces) == 0:
        return np.array([]), np.array([]), []
    
    # Stack into array: (trials, time)
    trial_traces = np.stack(trial_traces, axis=0)
    
    print(f"Extracted {len(trial_traces)} full trial traces aligned to F1 start")
    print(f"Time vector: {len(time_vector)} samples, {time_vector[0]:.1f} to {time_vector[-1]:.1f}s")
    
    return trial_traces, time_vector, trial_info




def _plot_trial_traces_with_events(ax, time_vector: np.ndarray, traces: np.ndarray,
                                 trial_info: List[Dict], trial_mask: Optional[np.ndarray],
                                 title: str, color: str = 'blue'):
    """Plot trial traces with event markers"""
    
    if traces.size == 0:
        ax.set_title(title)
        ax.set_xlabel('Time from F1 Start (s)')
        ax.set_ylabel('Component Activity')
        return
    
    # Plot individual trials with transparency
    for trial_idx in range(traces.shape[0]):
        trace = traces[trial_idx]
        valid_mask = np.isfinite(trace)
        if np.sum(valid_mask) > 0:
            ax.plot(time_vector[valid_mask], trace[valid_mask], 
                   color=color, alpha=0.15, linewidth=0.5)
    
    # Plot mean trace
    mean_trace = np.nanmean(traces, axis=0)
    valid_mask = np.isfinite(mean_trace)
    if np.sum(valid_mask) > 0:
        ax.plot(time_vector[valid_mask], mean_trace[valid_mask], 
               color=color, linewidth=2.5, alpha=1.0, label='Mean')
    
    # Add event markers (using first trial as representative)
    if len(trial_info) > 0:
        trial = trial_info[0]
        
        # F1 start (reference point)
        ax.axvline(0, color='green', linestyle='-', alpha=0.8, linewidth=2, label='F1 Start')
        
        # F1 end (ISI start)
        if not pd.isna(trial['f1_end']):
            ax.axvline(trial['f1_end'], color='orange', linestyle='--', alpha=0.7, label='F1 End')
        
        # F2 start (ISI end)
        if not pd.isna(trial['f2_start']):
            ax.axvline(trial['f2_start'], color='red', linestyle='--', alpha=0.7, label='F2 Start')
        
        # Choice
        if not pd.isna(trial['choice_start']):
            ax.axvline(trial['choice_start'], color='purple', linestyle=':', alpha=0.6, label='Choice')
        
        # Lick
        if not pd.isna(trial['lick_start']):
            ax.axvline(trial['lick_start'], color='brown', linestyle=':', alpha=0.6, label='Lick')
        
        # Highlight ISI period
        if not pd.isna(trial['f1_end']) and not pd.isna(trial['f2_start']):
            ax.axvspan(trial['f1_end'], trial['f2_start'], 
                     alpha=0.15, color='yellow', label='ISI Period')
    
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel('Time from F1 Start (s)')
    ax.set_ylabel('Component Activity (z-score)')
    ax.grid(True, alpha=0.3)
    
    # Add legend only to first plot
    if 'C0 All' in title or 'C6 All' in title or 'C12 All' in title:  # First plot of each figure
        ax.legend(fontsize=6, loc='upper right')

def comprehensive_component_validation_improved(phase_results: Dict[str, Any], 
                                              data: Dict[str, Any]) -> None:
    """Run ALL validation checks with improved visualization"""
    
    print("=" * 60)
    print("COMPREHENSIVE COMPONENT VALIDATION - IMPROVED")
    print("=" * 60)
    
    # 1. ROI assignment analysis
    analyze_ALL_cp_roi_assignments(phase_results)
    
    # 2. Temporal pattern validation
    validate_ALL_component_temporal_patterns(phase_results)
    
    # 3. IMPROVED: Full trial ISI trace visualization
    visualize_ALL_component_isi_traces_proper(phase_results, data)
    
    # 4. Summary statistics
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    n_components = len(signed_groups)
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total components analyzed: {n_components}")
    
    # Count components with good ROI assignments
    good_components = 0
    for group in signed_groups:
        if len(group['positive_rois']) > 0 or len(group['negative_rois']) > 0:
            good_components += 1
    
    print(f"Components with ROI assignments: {good_components}/{n_components}")
    print(f"CP decomposition loss: {cp_results.get('loss', 'N/A')}")
    print(f"Analysis complete!")













# def visualize_ALL_component_isi_traces_pos_neg_separated(phase_results: Dict[str, Any], 
#                                                        data: Dict[str, Any]) -> None:
#     """
#     Visualize ALL components with pos/neg ROIs separated
#     Each component gets 2 rows: positive ROIs, then negative ROIs
#     """
#     print(f"\n=== ALL COMPONENT FULL TRIAL VISUALIZATION: POS/NEG SEPARATED ===")
    
#     cp_results = phase_results['cp_results']
#     signed_groups = phase_results['signed_groups']
    
#     # Get data
#     df_trials = data['df_trials']
#     dff_clean = data['dFF_clean']
#     imaging_time = data['imaging_time']
    
#     # Get unique ISI values and SORT them properly
#     all_isis = df_trials['isi'].dropna().values
#     unique_isis = sorted(list(set(all_isis)))  # SORTED!
    
#     print(f"Found {len(unique_isis)} unique ISI values: {unique_isis}")
    
#     n_components = len(signed_groups)
#     n_isis = len(unique_isis)
    
#     # Use multiple figures - 3 components per figure (6 rows: 3 pos + 3 neg)
#     components_per_figure = 3
#     n_figures = int(np.ceil(n_components / components_per_figure))
    
#     for fig_idx in range(n_figures):
#         start_comp = fig_idx * components_per_figure
#         end_comp = min(start_comp + components_per_figure, n_components)
#         n_comps_this_fig = end_comp - start_comp
        
#         # Each component gets 2 rows (pos + neg)
#         n_rows = n_comps_this_fig * 2
        
#         # Create figure with proper size
#         fig, axes = plt.subplots(n_rows, n_isis + 1, 
#                                 figsize=(3*(n_isis+1), 2.5*n_rows))
        
#         if n_rows == 1:
#             axes = axes.reshape(1, -1)
#         if n_isis == 0:
#             print("No ISI data found!")
#             continue
        
#         row_idx = 0
#         for comp_idx in range(start_comp, end_comp):
#             group = signed_groups[comp_idx]
            
#             # Process positive and negative ROIs separately
#             for roi_sign, (roi_list, roi_weights, color, sign_name) in enumerate([
#                 (group['positive_rois'], group['positive_weights'], 'red', 'POS'),
#                 (group['negative_rois'], group['negative_weights'], 'blue', 'NEG')
#             ]):
                
#                 if len(roi_list) == 0:
#                     # Skip empty groups but increment row
#                     for col in range(n_isis + 1):
#                         axes[row_idx, col].text(0.5, 0.5, 'No ROIs', 
#                                                ha='center', va='center', transform=axes[row_idx, col].transAxes)
#                         axes[row_idx, col].set_title(f'C{comp_idx} {sign_name} (0 ROIs)')
#                     row_idx += 1
#                     continue
                
#                 # Extract FULL trial traces aligned to F1 start for this ROI group
#                 full_trial_traces, full_time_vector, full_trial_info = _extract_full_trial_traces_f1_aligned(
#                     df_trials, dff_clean, imaging_time, roi_list, 
#                     np.abs(roi_weights)  # Use absolute weights for averaging
#                 )
                
#                 # Column 0: All trials (fix the blur issue)
#                 ax = axes[row_idx, 0]
#                 _plot_trial_traces_with_events_improved(ax, full_time_vector, full_trial_traces, 
#                                                       full_trial_info, None,
#                                                       title=f'C{comp_idx} {sign_name} All (n={len(full_trial_traces)})',
#                                                       color=color, roi_count=len(roi_list))
                
#                 # Columns 1+: Each ISI value (SORTED)
#                 for isi_idx, target_isi in enumerate(unique_isis):
#                     ax = axes[row_idx, isi_idx + 1]
                    
#                     # Find trials with this ISI
#                     isi_mask = np.array([info['isi_ms'] == target_isi for info in full_trial_info])
                    
#                     if np.sum(isi_mask) > 0:
#                         isi_traces_subset = full_trial_traces[isi_mask]
#                         isi_info_subset = [info for i, info in enumerate(full_trial_info) if isi_mask[i]]
                        
#                         _plot_trial_traces_with_events_improved(ax, full_time_vector, isi_traces_subset,
#                                                               isi_info_subset, isi_mask,
#                                                               title=f'{target_isi:.0f}ms (n={np.sum(isi_mask)})',
#                                                               color=color, roi_count=len(roi_list))
#                     else:
#                         ax.set_title(f'{target_isi:.0f}ms (n=0)')
#                         ax.set_xlabel('Time from F1 Start (s)')
#                         ax.set_ylabel('Component Activity')
                
#                 row_idx += 1
        
#         plt.suptitle(f'Components {start_comp}-{end_comp-1}: POS/NEG ROIs Separated, Full Trial Traces', fontsize=16)
#         plt.tight_layout()
#         plt.show()





def visualize_ALL_component_isi_traces_pos_neg_separated(phase_results: Dict[str, Any], 
                                                       data: Dict[str, Any]) -> None:
    """
    Visualize ALL components with pos/neg ROIs separated
    Each component gets 2 rows: positive ROIs, then negative ROIs
    """
    print(f"\n=== ALL COMPONENT FULL TRIAL VISUALIZATION: POS/NEG SEPARATED ===")
    
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    
    # Get data
    df_trials = data['df_trials']
    dff_clean = data['dFF_clean']
    imaging_time = data['imaging_time']
    
    # Get unique ISI values and SORT them properly
    all_isis = df_trials['isi'].dropna().values
    unique_isis = sorted(list(set(all_isis)))  # SORTED!
    
    print(f"Found {len(unique_isis)} unique ISI values: {unique_isis}")
    
    n_components = len(signed_groups)
    n_isis = len(unique_isis)
    
    # Use multiple figures - 3 components per figure (6 rows: 3 pos + 3 neg)
    components_per_figure = 3
    n_figures = int(np.ceil(n_components / components_per_figure))
    
    for fig_idx in range(n_figures):
        start_comp = fig_idx * components_per_figure
        end_comp = min(start_comp + components_per_figure, n_components)
        n_comps_this_fig = end_comp - start_comp
        
        # Each component gets 2 rows (pos + neg)
        n_rows = n_comps_this_fig * 2
        
        # Create figure with proper size
        fig, axes = plt.subplots(n_rows, n_isis + 1, 
                                figsize=(3*(n_isis+1), 2.5*n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_isis == 0:
            print("No ISI data found!")
            continue
        
        row_idx = 0
        for comp_idx in range(start_comp, end_comp):
            group = signed_groups[comp_idx]
            
            # Process positive and negative ROIs separately
            for roi_sign, (roi_list, color, sign_name) in enumerate([
                (group['positive_rois'], 'red', 'POS'),
                (group['negative_rois'], 'blue', 'NEG')
            ]):
                
                if len(roi_list) == 0:
                    # Skip empty groups but increment row
                    for col in range(n_isis + 1):
                        axes[row_idx, col].text(0.5, 0.5, 'No ROIs', 
                                               ha='center', va='center', transform=axes[row_idx, col].transAxes)
                        axes[row_idx, col].set_title(f'C{comp_idx} {sign_name} (0 ROIs)')
                    row_idx += 1
                    continue
                
                # FIX: Get the weights properly for this specific ROI subset
                all_loadings = group['all_loadings']  # Full component loadings
                roi_weights = all_loadings[roi_list]  # Extract weights for this ROI subset
                
                # Extract FULL trial traces aligned to F1 start for this ROI group
                full_trial_traces, full_time_vector, full_trial_info = _extract_full_trial_traces_f1_aligned(
                    df_trials, dff_clean, imaging_time, roi_list, roi_weights
                )
                
                # Column 0: All trials (fix the blur issue)
                ax = axes[row_idx, 0]
                _plot_trial_traces_with_events_improved(ax, full_time_vector, full_trial_traces, 
                                                      full_trial_info, None,
                                                      title=f'C{comp_idx} {sign_name} All (n={len(full_trial_traces)})',
                                                      color=color, roi_count=len(roi_list))
                
                # Columns 1+: Each ISI value (SORTED)
                for isi_idx, target_isi in enumerate(unique_isis):
                    ax = axes[row_idx, isi_idx + 1]
                    
                    # Find trials with this ISI
                    isi_mask = np.array([info['isi_ms'] == target_isi for info in full_trial_info])
                    
                    if np.sum(isi_mask) > 0:
                        isi_traces_subset = full_trial_traces[isi_mask]
                        isi_info_subset = [info for i, info in enumerate(full_trial_info) if isi_mask[i]]
                        
                        _plot_trial_traces_with_events_improved(ax, full_time_vector, isi_traces_subset,
                                                              isi_info_subset, isi_mask,
                                                              title=f'{target_isi:.0f}ms (n={np.sum(isi_mask)})',
                                                              color=color, roi_count=len(roi_list))
                    else:
                        ax.set_title(f'{target_isi:.0f}ms (n=0)')
                        ax.set_xlabel('Time from F1 Start (s)')
                        ax.set_ylabel('Component Activity')
                
                row_idx += 1
        
        plt.suptitle(f'Components {start_comp}-{end_comp-1}: POS/NEG ROIs Separated, Full Trial Traces', fontsize=16)
        plt.tight_layout()
        plt.show()







def _plot_trial_traces_with_events_improved(ax, time_vector: np.ndarray, traces: np.ndarray,
                                          trial_info: List[Dict], trial_mask: Optional[np.ndarray],
                                          title: str, color: str = 'blue', roi_count: int = 0):
    """Improved plotting with better scaling and visibility"""
    
    if traces.size == 0:
        ax.set_title(title)
        ax.set_xlabel('Time from F1 Start (s)')
        ax.set_ylabel('Component Activity')
        return
    
    # Calculate robust statistics for better scaling
    finite_traces = traces[np.isfinite(traces)]
    if len(finite_traces) > 0:
        trace_5th = np.percentile(finite_traces, 5)
        trace_95th = np.percentile(finite_traces, 95)
        trace_range = trace_95th - trace_5th
        y_margin = trace_range * 0.1
        y_min = trace_5th - y_margin
        y_max = trace_95th + y_margin
    else:
        y_min, y_max = -0.5, 0.5
    
    # Plot individual trials with MORE transparency for first column
    alpha_individual = 0.05 if 'All' in title else 0.1
    for trial_idx in range(traces.shape[0]):
        trace = traces[trial_idx]
        valid_mask = np.isfinite(trace)
        if np.sum(valid_mask) > 0:
            ax.plot(time_vector[valid_mask], trace[valid_mask], 
                   color=color, alpha=alpha_individual, linewidth=0.5)
    
    # Plot mean trace with THICKER line
    mean_trace = np.nanmean(traces, axis=0)
    valid_mask = np.isfinite(mean_trace)
    if np.sum(valid_mask) > 0:
        ax.plot(time_vector[valid_mask], mean_trace[valid_mask], 
               color=color, linewidth=3, alpha=1.0, label='Mean')
    
    # Add event markers (using first trial as representative)
    if len(trial_info) > 0:
        trial = trial_info[0]
        
        # F1 start (reference point) - THICKER
        ax.axvline(0, color='green', linestyle='-', alpha=0.9, linewidth=3, label='F1 Start')
        
        # F1 end (ISI start)
        if not pd.isna(trial['f1_end']):
            ax.axvline(trial['f1_end'], color='orange', linestyle='--', alpha=0.8, linewidth=2, label='F1 End')
        
        # F2 start (ISI end)
        if not pd.isna(trial['f2_start']):
            ax.axvline(trial['f2_start'], color='red', linestyle='--', alpha=0.8, linewidth=2, label='F2 Start')
        
        # Choice - THICKER if significant
        if not pd.isna(trial['choice_start']):
            ax.axvline(trial['choice_start'], color='purple', linestyle=':', alpha=0.8, linewidth=2, label='Choice')
        
        # Lick - THICKER if significant
        if not pd.isna(trial['lick_start']):
            ax.axvline(trial['lick_start'], color='brown', linestyle=':', alpha=0.8, linewidth=2, label='Lick')
        
        # Highlight ISI period with STRONGER color
        if not pd.isna(trial['f1_end']) and not pd.isna(trial['f2_start']):
            ax.axvspan(trial['f1_end'], trial['f2_start'], 
                     alpha=0.25, color='yellow', label='ISI Period')
    
    # Better formatting
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    ax.set_title(title + f' ({roi_count} ROIs)', fontsize=10)
    ax.set_xlabel('Time from F1 Start (s)')
    ax.set_ylabel('Activity (z-score)')
    ax.grid(True, alpha=0.2)
    
    # Set consistent y-limits for better comparison
    ax.set_ylim(y_min, y_max)
    
    # Add legend only to first plot of each figure
    if 'C0 POS All' in title or 'C3 POS All' in title or 'C6 POS All' in title:  # First plot of each figure
        ax.legend(fontsize=6, loc='upper right')

def show_component_roi_statistics(phase_results: Dict[str, Any]) -> None:
    """Show statistics about ROI assignments across components"""
    
    signed_groups = phase_results['signed_groups']
    cp_results = phase_results['cp_results']
    A = cp_results['A']
    
    print(f"\n=== ROI ASSIGNMENT STATISTICS ===")
    print(f"{'Comp':<4} {'Pos ROIs':<8} {'Neg ROIs':<8} {'Total':<6} {'Pos %ile':<8} {'Neg %ile':<8}")
    print("-" * 50)
    
    for comp_idx, group in enumerate(signed_groups):
        pos_count = len(group['positive_rois'])
        neg_count = len(group['negative_rois'])
        total_count = pos_count + neg_count
        
        # Calculate what percentile the thresholds represent
        loadings = A[:, comp_idx]
        pos_threshold = np.percentile(loadings, 90)  # 10% = top 10% = 90th percentile
        neg_threshold = np.percentile(loadings, 10)  # 10% = bottom 10% = 10th percentile
        
        print(f"{comp_idx:<4} {pos_count:<8} {neg_count:<8} {total_count:<6} {pos_threshold:<8.3f} {neg_threshold:<8.3f}")

def comprehensive_component_validation_pos_neg(phase_results: Dict[str, Any], 
                                              data: Dict[str, Any]) -> None:
    """Run validation with pos/neg separation"""
    
    print("=" * 60)
    print("COMPREHENSIVE COMPONENT VALIDATION - POS/NEG SEPARATED")
    print("=" * 60)
    
    # 1. Show ROI statistics
    show_component_roi_statistics(phase_results)
    
    # 2. ROI assignment analysis
    analyze_ALL_cp_roi_assignments(phase_results)
    
    # 3. Temporal pattern validation
    validate_ALL_component_temporal_patterns(phase_results)
    
    # 4. NEW: Full trial ISI trace visualization with pos/neg separation
    visualize_ALL_component_isi_traces_pos_neg_separated(phase_results, data)
    
    # 5. Summary statistics
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    n_components = len(signed_groups)
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total components analyzed: {n_components}")
    
    # Count components with good ROI assignments
    good_components = 0
    for group in signed_groups:
        if len(group['positive_rois']) > 0 or len(group['negative_rois']) > 0:
            good_components += 1
    
    print(f"Components with ROI assignments: {good_components}/{n_components}")
    print(f"CP decomposition loss: {cp_results.get('loss', 'N/A')}")
    print(f"Analysis complete!")
































































def diagnose_signal_vs_noise(data: Dict[str, Any]) -> None:
    """Diagnose whether we have real signals or just noise"""
    
    print("=== SIGNAL VS NOISE DIAGNOSIS ===")
    
    dff_clean = data['dFF_clean']
    df_trials = data['df_trials']
    imaging_time = data['imaging_time']
    
    # 1. Check raw signal amplitudes
    print(f"dF/F statistics:")
    print(f"  Mean: {np.mean(dff_clean):.4f}")
    print(f"  Std: {np.std(dff_clean):.4f}")
    print(f"  Range: {np.min(dff_clean):.4f} to {np.max(dff_clean):.4f}")
    
    # 2. Compare ISI vs non-ISI periods
    isi_signals = []
    non_isi_signals = []
    
    for trial_idx, trial in df_trials.iterrows():
        if pd.isna(trial['end_flash_1']) or pd.isna(trial['start_flash_2']):
            continue
            
        # ISI period
        isi_start_abs = trial['trial_start_timestamp'] + trial['end_flash_1']
        isi_end_abs = trial['trial_start_timestamp'] + trial['start_flash_2']
        
        isi_start_idx = np.argmin(np.abs(imaging_time - isi_start_abs))
        isi_end_idx = np.argmin(np.abs(imaging_time - isi_end_abs))
        
        if isi_end_idx > isi_start_idx:
            isi_segment = dff_clean[:, isi_start_idx:isi_end_idx]
            isi_signals.append(np.std(isi_segment, axis=1))  # Std per ROI
            
        # Non-ISI period (same duration before trial)
        pre_trial_start = isi_start_abs - (isi_end_abs - isi_start_abs)
        pre_trial_end = isi_start_abs
        
        pre_start_idx = np.argmin(np.abs(imaging_time - pre_trial_start))
        pre_end_idx = np.argmin(np.abs(imaging_time - pre_trial_end))
        
        if pre_end_idx > pre_start_idx and pre_start_idx >= 0:
            pre_segment = dff_clean[:, pre_start_idx:pre_end_idx]
            non_isi_signals.append(np.std(pre_segment, axis=1))
    
    if len(isi_signals) > 0 and len(non_isi_signals) > 0:
        isi_signals = np.array(isi_signals)
        non_isi_signals = np.array(non_isi_signals)
        
        print(f"\nISI vs Non-ISI signal comparison:")
        print(f"  ISI signal std: {np.mean(isi_signals):.4f} ± {np.std(isi_signals):.4f}")
        print(f"  Non-ISI signal std: {np.mean(non_isi_signals):.4f} ± {np.std(non_isi_signals):.4f}")
        print(f"  Signal-to-noise ratio: {np.mean(isi_signals) / np.mean(non_isi_signals):.3f}")
        
        # Statistical test
        from scipy.stats import ttest_rel
        if isi_signals.shape == non_isi_signals.shape:
            t_stat, p_val = ttest_rel(isi_signals.flatten(), non_isi_signals.flatten())
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.6f}")

def compare_baseline_methods(data: Dict[str, Any]) -> None:
    """Compare different baseline methods to see which preserves real signals"""
    
    print("\n=== COMPARING BASELINE METHODS ===")
    
    F = data['F']  # Raw fluorescence
    dff_clean = data['dFF_clean']  # Your current method
    
    # Method 1: Your current method (already computed)
    print(f"Current dF/F method:")
    print(f"  Range: {np.min(dff_clean):.4f} to {np.max(dff_clean):.4f}")
    print(f"  Std: {np.std(dff_clean):.4f}")
    
    # Method 2: Simple session-wide F0
    F0_session = np.percentile(F, 10, axis=1, keepdims=True)
    dff_session = (F - F0_session) / F0_session
    print(f"Session-wide F0 (10th percentile):")
    print(f"  Range: {np.min(dff_session):.4f} to {np.max(dff_session):.4f}")
    print(f"  Std: {np.std(dff_session):.4f}")
    
    # Method 3: Rolling percentile baseline (more conservative)
    dff_rolling = _compute_percentile_baseline(data, window_s=60.0)  # 60s window
    print(f"Rolling percentile baseline (60s window):")
    print(f"  Range: {np.min(dff_rolling):.4f} to {np.max(dff_rolling):.4f}")
    print(f"  Std: {np.std(dff_rolling):.4f}")
    
    # Method 4: Z-scored version
    dff_zscore = zscore(dff_clean, axis=1)
    print(f"Z-scored current method:")
    print(f"  Range: {np.min(dff_zscore):.4f} to {np.max(dff_zscore):.4f}")
    print(f"  Std: {np.std(dff_zscore):.4f}")

def test_isi_modulation_strength(data: Dict[str, Any]) -> None:
    """Test how strong ISI modulation really is with different methods"""
    
    print("\n=== ISI MODULATION STRENGTH TEST ===")
    
    # Test with different baseline methods
    methods = {
        'original': data['dFF_clean'],
        'session_f0': None,  # Will compute
        'rolling_baseline': None,  # Will compute
        'zscore_original': zscore(data['dFF_clean'], axis=1)
    }
    
    # Compute additional methods
    F = data['F']
    F0_session = np.percentile(F, 10, axis=1, keepdims=True)
    methods['session_f0'] = (F - F0_session) / F0_session
    methods['rolling_baseline'] = _compute_percentile_baseline(data, window_s=30.0)
    
    for method_name, dff_data in methods.items():
        if dff_data is None:
            continue
            
        print(f"\n{method_name.upper()}:")
        stats = _compute_isi_statistics(dff_data, data)
        print(f"  Mean ISI modulation: {stats['mean_modulation']:.4f}")
        print(f"  Max ISI modulation: {stats['max_modulation']:.4f}")
        print(f"  Std ISI modulation: {stats['std_modulation']:.4f}")
        print(f"  ROIs analyzed: {stats['n_rois']}")
        
        
        
        
def conservative_isi_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """More conservative approach to find real ISI signals"""
    
    print("=== CONSERVATIVE ISI ANALYSIS ===")
    
    # Use minimal preprocessing
    F = data['F']
    dff_conservative = (F - np.percentile(F, 20, axis=1, keepdims=True)) / np.percentile(F, 20, axis=1, keepdims=True)
    
    # Look for ROIs with consistent ISI modulation across trials
    df_trials = data['df_trials']
    imaging_time = data['imaging_time']
    
    roi_isi_modulations = []
    roi_consistency_scores = []
    
    for roi_idx in range(dff_conservative.shape[0]):
        roi_trace = dff_conservative[roi_idx, :]
        trial_modulations = []
        
        for trial_idx, trial in df_trials.iterrows():
            if pd.isna(trial['end_flash_1']) or pd.isna(trial['start_flash_2']):
                continue
                
            # Extract ISI segment
            isi_start_abs = trial['trial_start_timestamp'] + trial['end_flash_1']
            isi_end_abs = trial['trial_start_timestamp'] + trial['start_flash_2']
            
            isi_start_idx = np.argmin(np.abs(imaging_time - isi_start_abs))
            isi_end_idx = np.argmin(np.abs(imaging_time - isi_end_abs))
            
            if isi_end_idx > isi_start_idx + 2:
                isi_segment = roi_trace[isi_start_idx:isi_end_idx]
                if len(isi_segment) > 0:
                    # Simple modulation metric: range during ISI
                    modulation = np.max(isi_segment) - np.min(isi_segment)
                    trial_modulations.append(modulation)
        
        if len(trial_modulations) >= 5:
            mean_mod = np.mean(trial_modulations)
            consistency = 1.0 / (1.0 + np.std(trial_modulations))  # Higher = more consistent
            
            roi_isi_modulations.append(mean_mod)
            roi_consistency_scores.append(consistency)
        else:
            roi_isi_modulations.append(0.0)
            roi_consistency_scores.append(0.0)
    
    roi_isi_modulations = np.array(roi_isi_modulations)
    roi_consistency_scores = np.array(roi_consistency_scores)
    
    # Find ROIs with both high modulation AND consistency
    modulation_threshold = np.percentile(roi_isi_modulations, 95)  # Top 5%
    consistency_threshold = np.percentile(roi_consistency_scores, 90)  # Top 10%
    
    candidate_rois = np.where(
        (roi_isi_modulations >= modulation_threshold) & 
        (roi_consistency_scores >= consistency_threshold)
    )[0]
    
    print(f"Found {len(candidate_rois)} candidate ISI-modulated ROIs")
    print(f"Modulation threshold: {modulation_threshold:.4f}")
    print(f"Consistency threshold: {consistency_threshold:.4f}")
    
    return {
        'candidate_rois': candidate_rois,
        'roi_modulations': roi_isi_modulations,
        'roi_consistency': roi_consistency_scores,
        'dff_conservative': dff_conservative
    }

























































def visualize_ALL_component_isi_traces_pos_neg_rewarded(phase_results: Dict[str, Any], 
                                                       data: Dict[str, Any]) -> None:
    """
    Visualize ALL components with pos/neg ROIs separated AND reward 0/1 separated
    Each component gets 4 rows: pos_rewarded, pos_unrewarded, neg_rewarded, neg_unrewarded
    """
    print(f"\n=== ALL COMPONENT FULL TRIAL VISUALIZATION: POS/NEG + REWARD SEPARATED ===")
    
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    
    # Get data
    df_trials = data['df_trials']
    dff_clean = data['dFF_clean']
    imaging_time = data['imaging_time']
    
    # Get unique ISI values and SORT them properly
    all_isis = df_trials['isi'].dropna().values
    unique_isis = sorted(list(set(all_isis)))  # SORTED!
    
    print(f"Found {len(unique_isis)} unique ISI values: {unique_isis}")
    
    n_components = len(signed_groups)
    n_isis = len(unique_isis)
    
    # Use multiple figures - 2 components per figure (8 rows: 2 × (pos_rew + pos_unrew + neg_rew + neg_unrew))
    components_per_figure = 2
    n_figures = int(np.ceil(n_components / components_per_figure))
    
    for fig_idx in range(n_figures):
        start_comp = fig_idx * components_per_figure
        end_comp = min(start_comp + components_per_figure, n_components)
        n_comps_this_fig = end_comp - start_comp
        
        # Each component gets 4 rows (pos_rew + pos_unrew + neg_rew + neg_unrew)
        n_rows = n_comps_this_fig * 4
        
        # Create figure with proper size
        fig, axes = plt.subplots(n_rows, n_isis + 1, 
                                figsize=(3*(n_isis+1), 2*n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_isis == 0:
            print("No ISI data found!")
            continue
        
        row_idx = 0
        for comp_idx in range(start_comp, end_comp):
            group = signed_groups[comp_idx]
            
            # Process positive and negative ROIs, then rewarded/unrewarded
            for roi_sign, (roi_list, color_base, sign_name) in enumerate([
                (group['positive_rois'], 'red', 'POS'),
                (group['negative_rois'], 'blue', 'NEG')
            ]):
                
                # For each ROI type, split by reward
                for reward_status, (reward_val, color_modifier, reward_name) in enumerate([
                    (1, '', 'REWARDED'),
                    (0, 'dark', 'UNREWARDED')
                ]):
                    
                    # Adjust color for reward status
                    if color_modifier == 'dark':
                        color = f'dark{color_base}' if color_base in ['red', 'blue'] else color_base
                    else:
                        color = color_base
                    
                    condition_name = f'{sign_name}_{reward_name}'
                    
                    if len(roi_list) == 0:
                        # Skip empty ROI groups but increment row
                        for col in range(n_isis + 1):
                            axes[row_idx, col].text(0.5, 0.5, 'No ROIs', 
                                                   ha='center', va='center', 
                                                   transform=axes[row_idx, col].transAxes)
                            axes[row_idx, col].set_title(f'C{comp_idx} {condition_name} (0 ROIs)')
                        row_idx += 1
                        continue
                    
                    # Get the weights properly for this specific ROI subset
                    all_loadings = group['all_loadings']  # Full component loadings
                    roi_weights = all_loadings[roi_list]  # Extract weights for this ROI subset
                    
                    # Extract FULL trial traces aligned to F1 start for this ROI group
                    full_trial_traces, full_time_vector, full_trial_info = _extract_full_trial_traces_f1_aligned(
                        df_trials, dff_clean, imaging_time, roi_list, roi_weights
                    )
                    
                    # Filter by reward status
                    if 'rewarded' in df_trials.columns:
                        reward_mask = np.array([df_trials.loc[info['trial_idx'], 'rewarded'] == reward_val 
                                              for info in full_trial_info])
                        
                        if np.sum(reward_mask) > 0:
                            reward_traces = full_trial_traces[reward_mask]
                            reward_info = [info for i, info in enumerate(full_trial_info) if reward_mask[i]]
                        else:
                            reward_traces = np.array([])
                            reward_info = []
                    else:
                        # If no reward column, show all trials
                        reward_traces = full_trial_traces
                        reward_info = full_trial_info
                    
                    # Column 0: All trials for this condition
                    ax = axes[row_idx, 0]
                    if len(reward_traces) > 0:
                        _plot_trial_traces_with_events_improved(
                            ax, full_time_vector, reward_traces, reward_info, None,
                            title=f'C{comp_idx} {condition_name} All (n={len(reward_traces)})',
                            color=color, roi_count=len(roi_list)
                        )
                    else:
                        ax.text(0.5, 0.5, f'No {reward_name.lower()} trials', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'C{comp_idx} {condition_name} All (n=0)')
                    
                    # Columns 1+: Each ISI value (SORTED)
                    for isi_idx, target_isi in enumerate(unique_isis):
                        ax = axes[row_idx, isi_idx + 1]
                        
                        if len(reward_info) > 0:
                            # Find trials with this ISI
                            isi_mask = np.array([info['isi_ms'] == target_isi for info in reward_info])
                            
                            if np.sum(isi_mask) > 0:
                                isi_traces_subset = reward_traces[isi_mask]
                                isi_info_subset = [info for i, info in enumerate(reward_info) if isi_mask[i]]
                                
                                _plot_trial_traces_with_events_improved(
                                    ax, full_time_vector, isi_traces_subset, isi_info_subset, isi_mask,
                                    title=f'{target_isi:.0f}ms (n={np.sum(isi_mask)})',
                                    color=color, roi_count=len(roi_list)
                                )
                            else:
                                ax.set_title(f'{target_isi:.0f}ms (n=0)')
                                ax.set_xlabel('Time from F1 Start (s)')
                                ax.set_ylabel('Component Activity')
                        else:
                            ax.set_title(f'{target_isi:.0f}ms (n=0)')
                            ax.set_xlabel('Time from F1 Start (s)')
                            ax.set_ylabel('Component Activity')
                    
                    row_idx += 1
        
        plt.suptitle(f'Components {start_comp}-{end_comp-1}: POS/NEG × REWARD/UNREWARD Separated', fontsize=16)
        plt.tight_layout()
        plt.show()

def comprehensive_component_validation_pos_neg_reward(phase_results: Dict[str, Any], 
                                                     data: Dict[str, Any]) -> None:
    """Run validation with pos/neg + reward separation"""
    
    print("=" * 60)
    print("COMPREHENSIVE COMPONENT VALIDATION - POS/NEG × REWARD SEPARATED")
    print("=" * 60)
    
    # Check if reward column exists
    df_trials = data['df_trials']
    if 'rewarded' not in df_trials.columns:
        print("⚠️  No 'rewarded' column found in df_trials")
        print("Available columns:", list(df_trials.columns))
        print("Falling back to pos/neg only separation...")
        comprehensive_component_validation_pos_neg(phase_results, data)
        return
    
    # Show reward statistics
    reward_counts = df_trials['rewarded'].value_counts()
    print(f"\n=== REWARD STATISTICS ===")
    print(f"Rewarded trials: {reward_counts.get(1, 0)}")
    print(f"Unrewarded trials: {reward_counts.get(0, 0)}")
    print(f"Total trials: {len(df_trials)}")
    
    # 1. Show ROI statistics
    show_component_roi_statistics(phase_results)
    
    # 2. ROI assignment analysis
    analyze_ALL_cp_roi_assignments(phase_results)
    
    # 3. Temporal pattern validation
    validate_ALL_component_temporal_patterns(phase_results)
    
    # 4. NEW: Full trial ISI trace visualization with pos/neg + reward separation
    visualize_ALL_component_isi_traces_pos_neg_rewarded(phase_results, data)
    
    # 5. Summary statistics by reward
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    n_components = len(signed_groups)
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total components analyzed: {n_components}")
    
    # Count components with good ROI assignments
    good_components = 0
    for group in signed_groups:
        if len(group['positive_rois']) > 0 or len(group['negative_rois']) > 0:
            good_components += 1
    
    print(f"Components with ROI assignments: {good_components}/{n_components}")
    print(f"CP decomposition loss: {cp_results.get('loss', 'N/A')}")
    print(f"Analysis complete with reward separation!")


































































def analyze_lick_direction_components(phase_results: Dict[str, Any], 
                                    data: Dict[str, Any]) -> None:
    """Analyze components that show lick-direction specific activity"""
    
    print("=== ANALYZING LICK DIRECTION COMPONENTS ===")
    
    df_trials = data['df_trials']
    signed_groups = phase_results['signed_groups']
    
    # Check if we have lick direction info
    if 'mouse_choice' not in df_trials.columns:
        print("No mouse_choice column found")
        return
    
    # Analyze choice patterns
    choice_counts = df_trials['mouse_choice'].value_counts()
    print(f"Choice distribution: {choice_counts}")
    
    # Look at ISI vs choice patterns
    isi_choice_cross = pd.crosstab(df_trials['isi'], df_trials['mouse_choice'], 
                                  normalize='index')
    print(f"\nISI vs Choice patterns:")
    print(isi_choice_cross)
    
    # Identify components with choice-related activity
    print(f"\nComponents showing choice/lick activity:")
    
    for comp_idx, group in enumerate(signed_groups[:10]):  # Check first 10
        # Check if this component shows the spike/dip pattern
        # (You can identify these from your visualization)
        
        # These would be components that show:
        # 1. Activity during choice/lick period
        # 2. Different patterns for rewarded vs unrewarded
        # 3. Consistent across short rewarded + long unrewarded
        
        print(f"Component {comp_idx}: "
              f"{len(group['positive_rois'])} pos, "
              f"{len(group['negative_rois'])} neg ROIs")

def interpret_behavioral_components(phase_results: Dict[str, Any], 
                                  data: Dict[str, Any]) -> None:
    """Interpret what we're seeing behaviorally"""
    
    print("=== BEHAVIORAL INTERPRETATION ===")
    
    df_trials = data['df_trials']
    
    # ISI discrimination strategy analysis
    print("ISI discrimination patterns:")
    
    # Short ISI trials (should be "different" → right spout for reward)
    short_trials = df_trials[df_trials['isi'] <= df_trials['isi'].median()]
    long_trials = df_trials[df_trials['isi'] > df_trials['isi'].median()]
    
    print(f"\nShort ISI trials ({len(short_trials)}):")
    if 'mouse_choice' in df_trials.columns:
        short_choices = short_trials['mouse_choice'].value_counts()
        print(f"  Choices: {short_choices}")
        if 'rewarded' in df_trials.columns:
            short_rewarded = short_trials['rewarded'].value_counts()
            print(f"  Rewarded: {short_rewarded}")
    
    print(f"\nLong ISI trials ({len(long_trials)}):")
    if 'mouse_choice' in df_trials.columns:
        long_choices = long_trials['mouse_choice'].value_counts()
        print(f"  Choices: {long_choices}")
        if 'rewarded' in df_trials.columns:
            long_rewarded = long_trials['rewarded'].value_counts()
            print(f"  Rewarded: {long_rewarded}")
    
    # The pattern you observed suggests:
    print(f"\n=== KEY FINDING ===")
    print("Components showing spike/dip at choice/lick:")
    print("- Rewarded SHORT trials (mouse correctly chose 'different')")
    print("- Unrewarded LONG trials (mouse incorrectly chose 'different')")
    print("- Both involve LEFT SPOUT licking")
    print("- This indicates MOTOR-SPECIFIC activity, not ISI discrimination")
    
    print(f"\nThis suggests these components encode:")
    print("1. Left spout licking motor activity")
    print("2. Choice execution (regardless of correctness)")
    print("3. Possibly choice confidence or vigor")
    print("4. NOT ISI temporal discrimination per se")
















# Look for TRUE ISI discrimination components
def find_isi_timing_components(phase_results: Dict[str, Any], 
                              data: Dict[str, Any]) -> None:
    """Find components that truly encode ISI timing, not motor execution"""
    
    print("=== LOOKING FOR TRUE ISI TIMING COMPONENTS ===")
    
    signed_groups = phase_results['signed_groups']
    
    # Look for components that:
    # 1. Show activity during ISI period (not just at choice)
    # 2. Scale with ISI duration
    # 3. Are similar regardless of choice direction
    
    isi_components = []
    motor_components = []
    
    for comp_idx, group in enumerate(signed_groups):
        # Criteria for ISI timing vs motor components:
        # - ISI components: sustained activity during ISI
        # - Motor components: brief activity at choice/lick
        
        # You can categorize based on your visualization:
        # Components 0, 1, 5, 7, 8, 10, 13 look like ISI-related
        # Components with choice spikes are motor-related
        
        if comp_idx in [0, 1, 5, 7, 8, 10, 13]:  # Update based on your visual inspection
            isi_components.append(comp_idx)
        else:
            motor_components.append(comp_idx)
    
    print(f"Potential ISI timing components: {isi_components}")
    print(f"Potential motor execution components: {motor_components}")
    
    return isi_components, motor_components















def _calculate_rise_time(phase_pattern: np.ndarray, phase_bins: np.ndarray) -> Optional[float]:
    """Calculate rise time as percentage of ISI phase"""
    
    if len(phase_pattern) < 3:
        return None
    
    # Find peak
    peak_idx = np.argmax(phase_pattern)
    if peak_idx == 0:
        return None
    
    # Find where pattern starts rising (crosses baseline)
    baseline = np.mean(phase_pattern[:5])  # Use first 5% as baseline
    threshold = baseline + 0.1 * (phase_pattern[peak_idx] - baseline)
    
    rise_start_idx = None
    for i in range(peak_idx):
        if phase_pattern[i] >= threshold:
            rise_start_idx = i
            break
    
    if rise_start_idx is None:
        return None
    
    # Convert to percentage of ISI
    rise_time_pct = (phase_bins[peak_idx] - phase_bins[rise_start_idx]) * 100
    return rise_time_pct

def _calculate_decay_time(phase_pattern: np.ndarray, phase_bins: np.ndarray) -> Optional[float]:
    """Calculate decay time as percentage of ISI phase"""
    
    if len(phase_pattern) < 3:
        return None
    
    # Find peak
    peak_idx = np.argmax(phase_pattern)
    if peak_idx >= len(phase_pattern) - 1:
        return None
    
    # Find where pattern decays to threshold after peak
    peak_val = phase_pattern[peak_idx]
    baseline = np.mean(phase_pattern[-5:])  # Use last 5% as baseline
    threshold = peak_val - 0.63 * (peak_val - baseline)  # 63% decay (1/e)
    
    decay_end_idx = None
    for i in range(peak_idx + 1, len(phase_pattern)):
        if phase_pattern[i] <= threshold:
            decay_end_idx = i
            break
    
    if decay_end_idx is None:
        return None
    
    # Convert to percentage of ISI
    decay_time_pct = (phase_bins[decay_end_idx] - phase_bins[peak_idx]) * 100
    return decay_time_pct

def _compute_percentile_baseline(data: Dict[str, Any], window_s: float = 30.0) -> np.ndarray:
    """Compute percentile baseline for dF/F calculation"""
    
    F = data['F']  # Raw fluorescence
    imaging_fs = data['imaging_fs']
    
    # Convert window to samples
    window_samples = int(window_s * imaging_fs)
    
    n_rois, n_timepoints = F.shape
    dff_percentile = np.zeros_like(F)
    
    print(f"Computing {window_s}s percentile baseline (window: {window_samples} samples)")
    
    for roi_idx in range(n_rois):
        roi_trace = F[roi_idx, :]
        
        # Use rolling percentile (20th percentile as baseline)
        baseline = np.zeros_like(roi_trace)
        
        for t in range(n_timepoints):
            # Define window around timepoint
            start_idx = max(0, t - window_samples // 2)
            end_idx = min(n_timepoints, t + window_samples // 2)
            
            # Calculate 20th percentile in window
            window_data = roi_trace[start_idx:end_idx]
            baseline[t] = np.percentile(window_data, 20)
        
        # Calculate dF/F
        dff_percentile[roi_idx, :] = (roi_trace - baseline) / baseline
        
        # Handle any inf/nan values
        dff_percentile[roi_idx, :] = np.nan_to_num(dff_percentile[roi_idx, :], 
                                                   nan=0.0, posinf=10.0, neginf=-1.0)
    
    return dff_percentile

def _compute_isi_statistics(dff_data: np.ndarray, data: Dict[str, Any]) -> Dict[str, float]:
    """Compute ISI modulation statistics for different baseline methods"""
    
    df_trials = data['df_trials']
    imaging_time = data['imaging_time']
    
    # Extract ISI segments
    isi_segments = []
    
    for trial_idx, trial in df_trials.iterrows():
        if pd.isna(trial['end_flash_1']) or pd.isna(trial['start_flash_2']):
            continue
            
        # Get ISI period
        isi_start_abs = trial['trial_start_timestamp'] + trial['end_flash_1']
        isi_end_abs = trial['trial_start_timestamp'] + trial['start_flash_2']
        
        # Find imaging indices
        isi_start_idx = np.argmin(np.abs(imaging_time - isi_start_abs))
        isi_end_idx = np.argmin(np.abs(imaging_time - isi_end_abs))
        
        if isi_end_idx - isi_start_idx < 3:
            continue
            
        # Extract ISI segment for all ROIs
        isi_segment = dff_data[:, isi_start_idx:isi_end_idx+1]
        isi_segments.append(isi_segment)
    
    if len(isi_segments) == 0:
        return {'mean_modulation': 0.0, 'max_modulation': 0.0, 'std_modulation': 0.0}
    
    # Stack segments and calculate statistics
    isi_stack = np.stack(isi_segments, axis=0)  # (trials, rois, time)
    
    # Calculate modulation per ROI (range across ISI)
    roi_modulations = []
    for roi_idx in range(isi_stack.shape[1]):
        roi_isi_traces = isi_stack[:, roi_idx, :]  # (trials, time)
        
        # Calculate range (max - min) for each trial, then average
        trial_ranges = []
        for trial_idx in range(roi_isi_traces.shape[0]):
            trial_trace = roi_isi_traces[trial_idx, :]
            if not np.all(np.isnan(trial_trace)):
                trial_range = np.nanmax(trial_trace) - np.nanmin(trial_trace)
                trial_ranges.append(trial_range)
        
        if len(trial_ranges) > 0:
            roi_modulations.append(np.mean(trial_ranges))
    
    if len(roi_modulations) == 0:
        return {'mean_modulation': 0.0, 'max_modulation': 0.0, 'std_modulation': 0.0}
    
    roi_modulations = np.array(roi_modulations)
    
    return {
        'mean_modulation': np.mean(roi_modulations),
        'max_modulation': np.max(roi_modulations),
        'std_modulation': np.std(roi_modulations),
        'n_rois': len(roi_modulations)
    }

def analyze_isi_timing_components_specifically(phase_results: Dict[str, Any], 
                                             data: Dict[str, Any]) -> None:
    """Focus analysis on the ISI timing components (not motor)"""
    
    print("=== ANALYZING ISI TIMING COMPONENTS SPECIFICALLY ===")
    
    # Your identified ISI timing components
    isi_timing_components = [0, 1, 5, 7, 8, 10, 13]
    
    signed_groups = phase_results['signed_groups']
    cp_results = phase_results['cp_results']
    B = cp_results['B']  # Phase factors
    phase_bins = phase_results['phase_bins']
    
    print(f"Focusing on {len(isi_timing_components)} ISI timing components: {isi_timing_components}")
    
    # Create focused visualization
    fig, axes = plt.subplots(len(isi_timing_components), 2, figsize=(12, 3*len(isi_timing_components)))
    if len(isi_timing_components) == 1:
        axes = axes.reshape(1, -1)
    
    for plot_idx, comp_idx in enumerate(isi_timing_components):
        group = signed_groups[comp_idx]
        phase_pattern = B[:, comp_idx]
        
        # Left: Phase pattern
        ax_phase = axes[plot_idx, 0]
        ax_phase.plot(phase_bins * 100, phase_pattern, 'b-', linewidth=2)
        ax_phase.set_title(f'Component {comp_idx}: ISI Phase Pattern')
        ax_phase.set_xlabel('ISI Phase (%)')
        ax_phase.set_ylabel('Component Weight')
        ax_phase.grid(True, alpha=0.3)
        ax_phase.axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        # Calculate and show temporal characteristics
        rise_time = _calculate_rise_time(phase_pattern, phase_bins)
        decay_time = _calculate_decay_time(phase_pattern, phase_bins)
        peak_phase = phase_bins[np.argmax(phase_pattern)] * 100
        
        text_info = f'Peak: {peak_phase:.1f}%\n'
        if rise_time is not None:
            text_info += f'Rise: {rise_time:.1f}%\n'
        if decay_time is not None:
            text_info += f'Decay: {decay_time:.1f}%'
        
        ax_phase.text(0.02, 0.98, text_info, transform=ax_phase.transAxes, 
                     va='top', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # Right: ROI assignments
        ax_roi = axes[plot_idx, 1]
        all_loadings = group['all_loadings']
        pos_rois = group['positive_rois']
        neg_rois = group['negative_rois']
        
        ax_roi.scatter(range(len(all_loadings)), all_loadings, alpha=0.3, s=1, color='gray')
        
        if len(pos_rois) > 0:
            ax_roi.scatter(pos_rois, all_loadings[pos_rois], color='red', s=8, 
                          label=f'Pos ROIs (n={len(pos_rois)})')
        
        if len(neg_rois) > 0:
            ax_roi.scatter(neg_rois, all_loadings[neg_rois], color='blue', s=8,
                          label=f'Neg ROIs (n={len(neg_rois)})')
        
        ax_roi.set_title(f'Component {comp_idx}: ROI Loadings')
        ax_roi.set_xlabel('ROI Index')
        ax_roi.set_ylabel('Loading Weight')
        ax_roi.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax_roi.legend(fontsize=8)
        ax_roi.grid(True, alpha=0.3)
    
    plt.suptitle('ISI Timing Components (Excluding Motor Components)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Summary of ISI timing characteristics
    print(f"\n=== ISI TIMING COMPONENT SUMMARY ===")
    print(f"{'Comp':<4} {'Peak Phase':<10} {'Rise Time':<10} {'Decay Time':<10} {'ROIs':<8}")
    print("-" * 50)
    
    for comp_idx in isi_timing_components:
        group = signed_groups[comp_idx]
        phase_pattern = B[:, comp_idx]
        
        peak_phase = phase_bins[np.argmax(phase_pattern)] * 100
        rise_time = _calculate_rise_time(phase_pattern, phase_bins)
        decay_time = _calculate_decay_time(phase_pattern, phase_bins)
        n_rois = len(group['positive_rois']) + len(group['negative_rois'])
        
        rise_str = f"{rise_time:.1f}%" if rise_time is not None else "N/A"
        decay_str = f"{decay_time:.1f}%" if decay_time is not None else "N/A"
        
        print(f"{comp_idx:<4} {peak_phase:<10.1f} {rise_str:<10} {decay_str:<10} {n_rois:<8}")

















































































def visualize_component_isi_traces(phase_results: Dict[str, Any], 
                                  data: Dict[str, Any],
                                  max_components: int = 6) -> None:
    """
    Replicate the component ISI trace visualization showing:
    - Component temporal patterns
    - All trials, Short trials, Long trials
    - Filtered to ROIs belonging to each component
    """
    print(f"\n=== COMPONENT ISI TRACE VISUALIZATION ===")
    
    cp_results = phase_results['cp_results']
    signed_groups = phase_results['signed_groups']
    trial_metadata = phase_results['trial_metadata']
    
    # Get the actual ISI data from your data structure
    df_trials = data['df_trials']
    dff_clean = data['dFF_clean']
    imaging_time = data['imaging_time']
    
    # Extract ISI segments in absolute time (not phase-normalized)
    isi_traces, isi_time_vector, trial_info = _extract_isi_absolute_segments(
        df_trials, dff_clean, imaging_time
    )
    
    n_components = len(signed_groups)
    n_show = min(max_components, n_components)
    
    # Create the subplot layout like your image
    fig, axes = plt.subplots(n_show, 3, figsize=(15, 4*n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    
    for comp_idx in range(n_show):
        group = signed_groups[comp_idx]
        
        # Get ROIs for this component (combine positive and negative)
        component_rois = np.concatenate([group['positive_rois'], group['negative_rois']])
        pos_rois = group['positive_rois']
        neg_rois = group['negative_rois']
        
        print(f"Component {comp_idx}: {len(pos_rois)} pos + {len(neg_rois)} neg = {len(component_rois)} total ROIs")
        
        # Calculate component activity by taking weighted average of ROI traces
        component_traces = _calculate_component_activity(
            isi_traces, component_rois, group['all_loadings']
        )
        
        # Separate short vs long trials
        short_mask = np.array([info['is_short'] for info in trial_info])
        long_mask = ~short_mask
        
        # Column 0: All trials
        ax = axes[comp_idx, 0]
        _plot_trial_traces(ax, isi_time_vector, component_traces, 
                          title=f'comp {comp_idx} | All',
                          color='blue')
        
        # Column 1: Short trials
        ax = axes[comp_idx, 1]
        if np.sum(short_mask) > 0:
            _plot_trial_traces(ax, isi_time_vector, component_traces[short_mask], 
                              title=f'Short',
                              color='blue')
        
        # Column 2: Long trials  
        ax = axes[comp_idx, 2]
        if np.sum(long_mask) > 0:
            _plot_trial_traces(ax, isi_time_vector, component_traces[long_mask], 
                              title=f'long',
                              color='blue')
        
        # Add ISI duration markers
        for col in range(3):
            ax = axes[comp_idx, col]
            _add_isi_markers(ax, trial_info, short_mask if col == 1 else long_mask if col == 2 else None)
    
    plt.suptitle('Component ISI Traces (Filtered by ROI Groups)', fontsize=16)
    plt.tight_layout()
    plt.show()

def _extract_isi_absolute_segments(df_trials: pd.DataFrame, 
                                  dff_clean: np.ndarray,
                                  imaging_time: np.ndarray,
                                  max_isi_duration: float = 8.0) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Extract ISI segments in absolute time (not phase-normalized)"""
    
    from scipy.interpolate import interp1d
    
    # Apply z-scoring per ROI
    dff_zscore = zscore(dff_clean, axis=1)
    
    # Create fixed time vector for ISI period
    dt = 1.0 / 30.0  # 30Hz sampling  
    isi_time_vector = np.arange(0, max_isi_duration + dt, dt)
    
    n_rois = dff_clean.shape[0]
    isi_segments = []
    trial_info = []
    
    for trial_idx, trial in df_trials.iterrows():
        if pd.isna(trial['end_flash_1']) or pd.isna(trial['start_flash_2']):
            continue
            
        # Get ISI period
        isi_start_abs = trial['trial_start_timestamp'] + trial['end_flash_1']
        isi_end_abs = trial['trial_start_timestamp'] + trial['start_flash_2']
        isi_duration = isi_end_abs - isi_start_abs
        
        if isi_duration > max_isi_duration:
            continue
            
        # Find imaging indices
        isi_start_idx = np.argmin(np.abs(imaging_time - isi_start_abs))
        isi_end_idx = np.argmin(np.abs(imaging_time - isi_end_abs))
        
        if isi_end_idx - isi_start_idx < 3:
            continue
            
        # Extract ISI segment
        isi_segment = dff_zscore[:, isi_start_idx:isi_end_idx+1]
        segment_times = imaging_time[isi_start_idx:isi_end_idx+1]
        relative_times = segment_times - isi_start_abs
        
        # Interpolate to fixed time grid
        interpolated_segment = np.full((n_rois, len(isi_time_vector)), np.nan)
        
        for roi_idx in range(n_rois):
            roi_trace = isi_segment[roi_idx]
            if not np.all(np.isnan(roi_trace)):
                valid_mask = np.isfinite(roi_trace)
                if np.sum(valid_mask) >= 2:
                    try:
                        # Only interpolate up to the actual ISI duration
                        valid_time_mask = isi_time_vector <= isi_duration
                        interp_func = interp1d(relative_times[valid_mask], roi_trace[valid_mask],
                                             kind='linear', bounds_error=False, fill_value=np.nan)
                        interpolated_segment[roi_idx, valid_time_mask] = interp_func(isi_time_vector[valid_time_mask])
                    except:
                        pass
        
        isi_segments.append(interpolated_segment)
        trial_info.append({
            'trial_idx': trial_idx,
            'isi_ms': trial['isi'],
            'isi_duration': isi_duration,
            'is_short': trial['isi'] < np.mean(df_trials['isi'].dropna())
        })
    
    # Stack into array: (trials, rois, time)
    isi_traces = np.stack(isi_segments, axis=0)
    
    print(f"Extracted {len(isi_segments)} ISI segments")
    print(f"ISI traces shape: {isi_traces.shape}")
    print(f"Time vector: {len(isi_time_vector)} samples, 0 to {isi_time_vector[-1]:.1f}s")
    
    return isi_traces, isi_time_vector, trial_info

def _calculate_component_activity(isi_traces: np.ndarray, 
                                 component_rois: np.ndarray,
                                 roi_loadings: np.ndarray) -> np.ndarray:
    """Calculate component activity as weighted average of component ROIs"""
    
    if len(component_rois) == 0:
        return np.full((isi_traces.shape[0], isi_traces.shape[2]), np.nan)
    
    # Get traces for component ROIs
    component_traces = isi_traces[:, component_rois, :]  # (trials, component_rois, time)
    
    # Weight by loadings (take absolute value to combine pos/neg ROIs)
    weights = np.abs(roi_loadings[component_rois])
    weights = weights / np.sum(weights)  # normalize
    
    # Calculate weighted average across ROIs
    weighted_traces = np.nansum(component_traces * weights[None, :, None], axis=1)
    
    return weighted_traces  # (trials, time)

def _plot_trial_traces(ax, time_vector: np.ndarray, traces: np.ndarray, 
                      title: str, color: str = 'blue'):
    """Plot individual trial traces with mean"""
    
    if traces.size == 0:
        ax.set_title(title)
        return
    
    # Plot individual trials with transparency
    for trial_idx in range(traces.shape[0]):
        trace = traces[trial_idx]
        valid_mask = np.isfinite(trace)
        if np.sum(valid_mask) > 0:
            ax.plot(time_vector[valid_mask], trace[valid_mask], 
                   color=color, alpha=0.1, linewidth=0.5)
    
    # Plot mean trace
    mean_trace = np.nanmean(traces, axis=0)
    valid_mask = np.isfinite(mean_trace)
    if np.sum(valid_mask) > 0:
        ax.plot(time_vector[valid_mask], mean_trace[valid_mask], 
               color=color, linewidth=2, alpha=0.8)
    
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_title(title)
    ax.set_ylabel('Mean')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

def _add_isi_markers(ax, trial_info: List[Dict], trial_mask: Optional[np.ndarray] = None):
    """Add ISI duration markers to plot"""
    
    if trial_mask is not None:
        relevant_trials = [info for i, info in enumerate(trial_info) if trial_mask[i]]
    else:
        relevant_trials = trial_info
    
    if len(relevant_trials) == 0:
        return
    
    # Get unique ISI durations for this subset
    isi_durations = [info['isi_duration'] for info in relevant_trials]
    unique_isis = np.unique(np.round(isi_durations, 1))
    
    # Add vertical lines for ISI boundaries
    for isi_dur in unique_isis[:5]:  # Show first 5 to avoid clutter
        ax.axvline(isi_dur, color='orange', linestyle='--', alpha=0.6, linewidth=1)

# Run

























































def debug_isi_extraction(data: Dict[str, Any], max_trials: int = 5):
    """Debug ISI extraction to see what's happening with the zero activity"""
    
    df_trials = data['df_trials']
    dff_clean = data['dFF_clean']
    imaging_time = data['imaging_time']
    
    print("=== DEBUGGING ISI EXTRACTION ===")
    
    # Check a few specific trials
    for i in range(min(max_trials, len(df_trials))):
        trial = df_trials.iloc[i]
        
        if pd.isna(trial['end_flash_1']) or pd.isna(trial['start_flash_2']):
            continue
            
        # Get ISI period
        isi_start_abs = trial['trial_start_timestamp'] + trial['end_flash_1']
        isi_end_abs = trial['trial_start_timestamp'] + trial['start_flash_2']
        isi_duration = isi_end_abs - isi_start_abs
        
        print(f"\nTrial {i}:")
        print(f"  ISI metadata: {trial['isi']:.0f}ms")
        print(f"  Actual ISI duration: {isi_duration:.3f}s = {isi_duration*1000:.0f}ms")
        print(f"  end_flash_1: {trial['end_flash_1']:.3f}s")
        print(f"  start_flash_2: {trial['start_flash_2']:.3f}s")
        
        # Find imaging indices
        isi_start_idx = np.argmin(np.abs(imaging_time - isi_start_abs))
        isi_end_idx = np.argmin(np.abs(imaging_time - isi_end_abs))
        
        print(f"  Imaging indices: {isi_start_idx} to {isi_end_idx}")
        print(f"  Imaging samples in ISI: {isi_end_idx - isi_start_idx + 1}")
        
        # Check our interpolation grid
        dt = 1.0 / 30.0
        max_isi_duration = 8.0
        isi_time_vector = np.arange(0, max_isi_duration + dt, dt)
        
        valid_time_mask = isi_time_vector <= isi_duration
        n_valid_samples = np.sum(valid_time_mask)
        n_nan_samples = len(isi_time_vector) - n_valid_samples
        
        print(f"  Time vector total samples: {len(isi_time_vector)}")
        print(f"  Valid samples (≤ ISI duration): {n_valid_samples}")
        print(f"  NaN samples (> ISI duration): {n_nan_samples}")
        print(f"  Transition at: {isi_duration:.3f}s (sample {n_valid_samples})")
















def verify_component_traces_full_trial(phase_results: Dict[str, Any], 
                                     data: Dict[str, Any],
                                     component_idx: int = 0,
                                     n_trials_show: int = 5) -> None:
    """
    Plot full trial traces for component ROIs to verify ISI patterns are real
    This should show activity during ISI but also at choice/lick events
    """
    print(f"\n=== VERIFYING COMPONENT {component_idx} FULL TRIAL TRACES ===")
    
    signed_groups = phase_results['signed_groups']
    df_trials = data['df_trials']
    # dff_clean = data['dFF_clean']  # Original dF/F, not z-scored
    dff_clean = data['dFF_smoothed']
    imaging_time = data['imaging_time']
    
    if component_idx >= len(signed_groups):
        print(f"Component {component_idx} not available. Max: {len(signed_groups)-1}")
        return
    
    group = signed_groups[component_idx]
    pos_rois = group['positive_rois']
    neg_rois = group['negative_rois']
    
    print(f"Component {component_idx}: {len(pos_rois)} positive + {len(neg_rois)} negative ROIs")
    
    # Select a few trials of different ISI durations
    trial_isis = df_trials['isi'].dropna()
    unique_isis = sorted(trial_isis.unique())
    
    # Pick representative ISIs
    if len(unique_isis) >= 3:
        target_isis = [unique_isis[0], unique_isis[len(unique_isis)//2], unique_isis[-1]]
    else:
        target_isis = unique_isis
    
    print(f"Showing trials with ISIs: {target_isis}")
    
    fig, axes = plt.subplots(len(target_isis), 2, figsize=(16, 4*len(target_isis)))
    if len(target_isis) == 1:
        axes = axes.reshape(1, -1)
    
    for isi_idx, target_isi in enumerate(target_isis):
        # Find trials with this ISI
        isi_trials = df_trials[df_trials['isi'] == target_isi]
        
        if len(isi_trials) == 0:
            continue
            
        # Pick first few trials
        trials_to_show = isi_trials.iloc[:min(n_trials_show, len(isi_trials))]
        
        for col, (roi_type, roi_list) in enumerate([('Positive', pos_rois), ('Negative', neg_rois)]):
            ax = axes[isi_idx, col]
            
            if len(roi_list) == 0:
                ax.set_title(f'ISI {target_isi}ms | {roi_type} ROIs (none)')
                continue
                
            # Take average across ROI type (weighted by loadings)
            if roi_type == 'Positive':
                weights = np.abs(group['positive_weights'])
            else:
                weights = np.abs(group['negative_weights'])
            
            weights = weights / np.sum(weights)  # normalize
            
            for trial_idx, (_, trial) in enumerate(trials_to_show.iterrows()):
                # Extract full trial
                trial_start_abs = trial['trial_start_timestamp']
                trial_end_abs = trial_start_abs + 8.0  # Show ~8s of trial
                
                # Find imaging indices
                start_idx = np.argmin(np.abs(imaging_time - trial_start_abs))
                end_idx = np.argmin(np.abs(imaging_time - trial_end_abs))
                
                if end_idx - start_idx < 10:
                    continue
                
                # Get ROI traces for this trial
                roi_traces = dff_clean[roi_list, start_idx:end_idx]  # (rois, time)
                trial_time = imaging_time[start_idx:end_idx] - trial_start_abs  # Relative to trial start
                
                # Calculate weighted average
                weighted_trace = np.average(roi_traces, axis=0, weights=weights)
                
                # Plot with transparency
                color = f'C{trial_idx}'
                ax.plot(trial_time, weighted_trace, color=color, alpha=0.7, linewidth=1,
                       label=f'Trial {trial_idx+1}' if trial_idx < 3 else '')
                
                # Mark key events
                if not pd.isna(trial['start_flash_1']):
                    ax.axvline(trial['start_flash_1'], color='green', linestyle=':', alpha=0.5)
                if not pd.isna(trial['end_flash_1']):
                    ax.axvline(trial['end_flash_1'], color='orange', linestyle=':', alpha=0.5)
                if not pd.isna(trial['start_flash_2']):
                    ax.axvline(trial['start_flash_2'], color='red', linestyle=':', alpha=0.5)
                if not pd.isna(trial['choice_start']):
                    ax.axvline(trial['choice_start'], color='purple', linestyle=':', alpha=0.5)
                if not pd.isna(trial['lick_start']):
                    ax.axvline(trial['lick_start'], color='brown', linestyle=':', alpha=0.5)
            
            # Highlight ISI period
            if len(trials_to_show) > 0:
                trial = trials_to_show.iloc[0]
                if not pd.isna(trial['end_flash_1']) and not pd.isna(trial['start_flash_2']):
                    ax.axvspan(trial['end_flash_1'], trial['start_flash_2'], 
                             alpha=0.2, color='yellow', label='ISI Period')
            
            ax.set_title(f'ISI {target_isi}ms | {roi_type} ROIs (n={len(roi_list)})')
            ax.set_xlabel('Time from Trial Start (s)')
            ax.set_ylabel('dF/F (original)')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
            
            if trial_idx < 3:
                ax.legend(fontsize=8)
    
    # Add event legend
    event_lines = [
        plt.Line2D([0], [0], color='green', linestyle=':', label='F1 Start'),
        plt.Line2D([0], [0], color='orange', linestyle=':', label='F1 End (ISI Start)'),
        plt.Line2D([0], [0], color='red', linestyle=':', label='F2 Start (ISI End)'),
        plt.Line2D([0], [0], color='purple', linestyle=':', label='Choice'),
        plt.Line2D([0], [0], color='brown', linestyle=':', label='Lick'),
        plt.Rectangle((0,0),1,1, facecolor='yellow', alpha=0.2, label='ISI Period')
    ]
    
    fig.legend(handles=event_lines, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=6, fontsize=10)
    plt.suptitle(f'Component {component_idx}: Full Trial Traces (Original dF/F)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def compare_isi_vs_full_trial_activity(phase_results: Dict[str, Any], 
                                     data: Dict[str, Any],
                                     component_idx: int = 0) -> None:
    """
    Direct comparison of ISI-only traces vs full trial traces for verification
    """
    print(f"\n=== ISI VS FULL TRIAL COMPARISON: COMPONENT {component_idx} ===")
    
    signed_groups = phase_results['signed_groups']
    df_trials = data['df_trials']
    dff_clean = data['dFF_clean']
    imaging_time = data['imaging_time']
    
    group = signed_groups[component_idx]
    component_rois = np.concatenate([group['positive_rois'], group['negative_rois']])
    roi_loadings = group['all_loadings']
    
    if len(component_rois) == 0:
        print("No ROIs in this component")
        return
    
    # Get a few representative trials
    sample_trials = df_trials.iloc[:10]  # First 10 trials
    
    fig, axes = plt.subplots(len(sample_trials), 2, figsize=(16, 3*len(sample_trials)))
    if len(sample_trials) == 1:
        axes = axes.reshape(1, -1)
    
    for trial_idx, (_, trial) in enumerate(sample_trials.iterrows()):
        if pd.isna(trial['end_flash_1']) or pd.isna(trial['start_flash_2']):
            continue
        
        # === LEFT: Full trial trace ===
        ax_full = axes[trial_idx, 0]
        
        trial_start_abs = trial['trial_start_timestamp']
        trial_end_abs = trial_start_abs + 8.0
        
        start_idx = np.argmin(np.abs(imaging_time - trial_start_abs))
        end_idx = np.argmin(np.abs(imaging_time - trial_end_abs))
        
        if end_idx - start_idx < 10:
            continue
        
        # Component activity (weighted average)
        component_traces = dff_clean[component_rois, start_idx:end_idx]
        weights = np.abs(roi_loadings[component_rois])
        weights = weights / np.sum(weights)
        
        weighted_trace = np.average(component_traces, axis=0, weights=weights)
        trial_time = imaging_time[start_idx:end_idx] - trial_start_abs
        
        ax_full.plot(trial_time, weighted_trace, 'b-', linewidth=1.5)
        ax_full.set_title(f'Trial {trial_idx}: Full Trace (ISI={trial["isi"]:.0f}ms)')
        ax_full.set_ylabel('dF/F')
        ax_full.grid(True, alpha=0.3)
        
        # Mark events
        if not pd.isna(trial['end_flash_1']):
            ax_full.axvline(trial['end_flash_1'], color='orange', linestyle='--', label='ISI Start')
        if not pd.isna(trial['start_flash_2']):
            ax_full.axvline(trial['start_flash_2'], color='red', linestyle='--', label='ISI End')
        if not pd.isna(trial['choice_start']):
            ax_full.axvline(trial['choice_start'], color='purple', linestyle='--', label='Choice')
        
        # Highlight ISI
        if not pd.isna(trial['end_flash_1']) and not pd.isna(trial['start_flash_2']):
            ax_full.axvspan(trial['end_flash_1'], trial['start_flash_2'], 
                           alpha=0.2, color='yellow')
        
        ax_full.axhline(0, color='gray', linestyle='-', alpha=0.3)
        if trial_idx == 0:
            ax_full.legend(fontsize=8)
        
        # === RIGHT: ISI-only trace ===
        ax_isi = axes[trial_idx, 1]
        
        isi_start_abs = trial_start_abs + trial['end_flash_1']
        isi_end_abs = trial_start_abs + trial['start_flash_2']
        isi_duration = isi_end_abs - isi_start_abs
        
        isi_start_idx = np.argmin(np.abs(imaging_time - isi_start_abs))
        isi_end_idx = np.argmin(np.abs(imaging_time - isi_end_abs))
        
        if isi_end_idx - isi_start_idx >= 3:
            isi_traces = dff_clean[component_rois, isi_start_idx:isi_end_idx+1]
            isi_weighted = np.average(isi_traces, axis=0, weights=weights)
            isi_time = imaging_time[isi_start_idx:isi_end_idx+1] - isi_start_abs
            
            ax_isi.plot(isi_time, isi_weighted, 'r-', linewidth=2, marker='o', markersize=3)
            ax_isi.set_title(f'ISI Only (Duration: {isi_duration:.3f}s)')
            ax_isi.set_ylabel('dF/F')
            ax_isi.set_xlabel('Time from ISI Start (s)')
            ax_isi.grid(True, alpha=0.3)
            ax_isi.axhline(0, color='gray', linestyle='-', alpha=0.3)
        else:
            ax_isi.set_title('ISI too short')
    
    plt.suptitle(f'Component {component_idx}: ISI vs Full Trial Activity', fontsize=16)
    plt.tight_layout()
    plt.show()






















if __name__ == "__main__":
    print("=== SUITE2P PROCESSING PIPELINE ===")
    
    

    
# # Configuration
cfg_path = r"D:/PHD/GIT/data_analysis/DAP/imaging/config.yaml"
# cfg = load_cfg_yaml(cfg_path)





# %%
cfg = load_cfg_yaml(cfg_path)
data = load_sid_data(cfg)
print("\n=== DATA STRUCTURE CHECK ===")
print(f"Available keys: {list(data.keys())}")
print(f"df_trials columns: {list(data['df_trials'].columns)}")
print(f"First few trial events:")
print(data['df_trials'][['trial_start_timestamp', 'start_flash_1', 'isi']].head())

# %%

n_trim = 15
data = trim_session_trials(data, n_trim_start=n_trim, n_trim_end=n_trim)


# %%
cfg['event_analysis'] = {
    'windows': {
        'trial_start': {'pre_event_s': 1.0, 'post_event_s': 0.2},
        'start_flash_1': {'pre_event_s': 0.3, 'post_event_s': 0.2},
        'end_flash_1': {'pre_event_s': 0.1, 'post_event_s': 0.2},
        'start_flash_2': {'pre_event_s': 0.2, 'post_event_s': 0.1},
        'end_flash_2': {'pre_event_s': 0.1, 'post_event_s': 0.1},
        'choice_start': {'pre_event_s': 0.2, 'post_event_s': 0.2},
        'lick_start': {'pre_event_s': 0.1, 'post_event_s': 0.7},
        'choice_stop': {'pre_event_s': 0.1, 'post_event_s': 0.2}
    },
}

# Test improved stack creation
# Test the corrected stack creation using IMAGING time vectors
stack_data = create_event_aligned_stacks(data, cfg, 'start_flash_2')
print(f"Stack data keys: {list(stack_data.keys())}")
print(f"Data is z-scored: {stack_data['normalization']}")
print(f"Stack shape: {stack_data['stacks'].shape}")
print(f"Original sampling: {stack_data['original_fs']:.1f} Hz")
print(f"Effective sampling: {stack_data['effective_fs']:.1f} Hz")


# %%
# Visualize the original stack data
print("=== ORIGINAL 30Hz DATA ===")
visualize_stack_data(stack_data, n_rois_show=700)

# %%
# Test and visualize interpolated data
if stack_data['stacks'].shape[2] < 20:  # If we have fewer than 20 samples
    print("\n=== INTERPOLATED 100Hz DATA ===")
    stack_data_interp = interpolate_event_stacks(stack_data, target_fs=100.0)
    print(f"Interpolated shape: {stack_data_interp['stacks'].shape}")
    print(f"New sampling rate: {stack_data_interp['effective_fs']:.1f} Hz")
    print(f"Time vector length: {len(stack_data_interp['time_vector'])}")
    print(f"Time range: {stack_data_interp['time_vector'][0]:.3f} to {stack_data_interp['time_vector'][-1]:.3f}s")
    visualize_stack_data(stack_data_interp, n_rois_show=700)

# %%


# %%
# Now you can run this with your data structure:
print("=== RUNNING PHASE ANALYSIS WITH YOUR DATA STRUCTURE ===")
apply_zscore = False
phase_results = run_isi_phase_analysis_from_data(
    data,
    n_phase_bins=120,
    n_components=20,
    apply_zscore=apply_zscore
)

# Visualize the results
visualize_phase_cp_results(phase_results)


# %%


# After running your phase analysis:
analyze_cp_roi_assignments(phase_results)

# Then validate a few components:
for comp_idx in range(min(3, len(phase_results['signed_groups']))):
    validate_component_temporal_patterns(phase_results, comp_idx)


# %%
# %%




# Replace the previous analysis with comprehensive version
print("=== RUNNING COMPREHENSIVE ANALYSIS ===")

# Run the full comprehensive validation
comprehensive_component_validation(phase_results, data)



# %%

# Replace the old comprehensive analysis with the improved version
print("=== RUNNING IMPROVED COMPREHENSIVE ANALYSIS ===")

# Run the improved validation with proper full trial traces
comprehensive_component_validation_improved(phase_results, data)



# %%

# Add to your main analysis file
# filepath: d:\PHD\GIT\data_analysis\DAP\imaging\analyze_main.py

# After your existing phase analysis, run the improved pos/neg separated version:
print("=== RUNNING POS/NEG SEPARATED COMPREHENSIVE ANALYSIS ===")

# Run the pos/neg separated validation
comprehensive_component_validation_pos_neg(phase_results, data)







# %%
# Replace your previous comprehensive analysis with reward separation
print("=== RUNNING POS/NEG × REWARD SEPARATED COMPREHENSIVE ANALYSIS ===")

# Run the pos/neg + reward separated validation
comprehensive_component_validation_pos_neg_reward(phase_results, data)























# %%




# Run this analysis
print("=== ANALYZING YOUR BEHAVIORAL FINDINGS ===")
analyze_lick_direction_components(phase_results, data)
interpret_behavioral_components(phase_results, data)

# Focus your analysis on ISI-specific components
isi_comps, motor_comps = find_isi_timing_components(phase_results, data)






# %%
# Run the focused ISI timing analysis
analyze_isi_timing_components_specifically(phase_results, data)
















# %%
max_components=10
# Run the visualization
visualize_component_isi_traces(phase_results, data, max_components=max_components)

# %%


# Run the debug
debug_isi_extraction(data, max_trials=5)







# %%
# Let's verify the component traces are real
print("=== VERIFYING COMPONENT TRACES ARE REAL ===")

# Check component 0 (or whatever looked most promising)
verify_component_traces_full_trial(phase_results, data, component_idx=0, n_trials_show=5)

# Direct comparison


# %%



for component_idx in range(0,max_components):
    # Check another component
    compare_isi_vs_full_trial_activity(phase_results, data, component_idx=component_idx)
    verify_component_traces_full_trial(phase_results, data, component_idx=component_idx, n_trials_show=3)















# %%



# Add this to your analysis pipeline:

# First, diagnose the signal quality
diagnose_signal_vs_noise(data)
compare_baseline_methods(data)
test_isi_modulation_strength(data)

# Then try conservative analysis
conservative_results = conservative_isi_analysis(data)















# %%

def find_exploded_rois(data: Dict[str, Any]) -> None:
    """Find ROIs with exploded dF/F values"""
    
    print("=== FINDING EXPLODED dF/F ROIs ===")
    
    dff_clean = data['dFF_clean']
    n_rois = dff_clean.shape[0]
    
    # Check each ROI for extreme values
    exploded_rois = []
    extreme_rois = []
    
    for roi_idx in range(n_rois):
        roi_trace = dff_clean[roi_idx, :]
        
        roi_min = np.min(roi_trace)
        roi_max = np.max(roi_trace)
        roi_std = np.std(roi_trace)
        roi_range = roi_max - roi_min
        
        # Flag ROIs with extreme values
        if roi_max > 50 or roi_min < -50:
            exploded_rois.append({
                'roi': roi_idx,
                'min': roi_min,
                'max': roi_max,
                'std': roi_std,
                'range': roi_range
            })
        elif roi_max > 5 or roi_min < -5 or roi_std > 5:
            extreme_rois.append({
                'roi': roi_idx,
                'min': roi_min,
                'max': roi_max,
                'std': roi_std,
                'range': roi_range
            })
    
    print(f"Total ROIs: {n_rois}")
    print(f"Exploded ROIs (>50 or <-50): {len(exploded_rois)}")
    print(f"Extreme ROIs (>10, <-5, std>5): {len(extreme_rois)}")
    
    if len(exploded_rois) > 0:
        print(f"\nWorst exploded ROIs:")
        exploded_sorted = sorted(exploded_rois, key=lambda x: x['range'], reverse=True)
        for roi_info in exploded_sorted[:]:
            print(f"  ROI {roi_info['roi']:3d}: range {roi_info['range']:8.1f} "
                  f"({roi_info['min']:6.1f} to {roi_info['max']:6.1f}), std {roi_info['std']:6.1f}")
    
    if len(extreme_rois) > 0:
        print(f"\nWorst extreme ROIs:")
        extreme_sorted = sorted(extreme_rois, key=lambda x: x['range'], reverse=True)
        for roi_info in extreme_sorted[:]:
            print(f"  ROI {roi_info['roi']:3d}: range {roi_info['range']:8.1f} "
                  f"({roi_info['min']:6.1f} to {roi_info['max']:6.1f}), std {roi_info['std']:6.1f}")
    
    return exploded_rois, extreme_rois

# Add this to your analysis right after loading data:
exploded_rois, extreme_rois = find_exploded_rois(data)