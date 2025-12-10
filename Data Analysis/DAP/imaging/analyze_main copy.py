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















































def discover_temporal_patterns(stack_data: Dict[str, Any], 
                              cfg: Dict[str, Any],
                              n_patterns: int = 5) -> Dict[str, Any]:
    """
    Discover temporal patterns using PCA on trial-averaged responses
    Simpler alternative to CP decomposition for initial pattern discovery
    """
    print(f"\n=== DISCOVERING TEMPORAL PATTERNS ===")
    
    stacks = stack_data['stacks']  # (trials, rois, time)
    time_vector = stack_data['time_vector']
    event_name = stack_data['event_name']
    
    # Compute trial-averaged response for each ROI
    mean_response = np.mean(stacks, axis=0)  # (rois, time)
    
    print(f"  Data shape: {mean_response.shape} (rois, time)")
    print(f"  Looking for {n_patterns} temporal patterns")
    
    # Apply PCA to find dominant temporal patterns
    from sklearn.decomposition import PCA
    
    # PCA on ROI responses (each ROI is a sample, time points are features)
    pca = PCA(n_components=n_patterns)
    roi_loadings = pca.fit_transform(mean_response)  # (rois, n_patterns)
    temporal_patterns = pca.components_  # (n_patterns, time)
    
    print(f"  Explained variance: {pca.explained_variance_ratio_}")
    print(f"  Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Find ROIs that strongly load on each pattern
    pattern_rois = {}
    for pattern_idx in range(n_patterns):
        # Find ROIs with strong positive/negative loadings
        loadings = roi_loadings[:, pattern_idx]
        
        # Define strong loading threshold (e.g., top/bottom 10%)
        threshold_pos = np.percentile(loadings, 90)
        threshold_neg = np.percentile(loadings, 10)
        
        pos_rois = np.where(loadings >= threshold_pos)[0]
        neg_rois = np.where(loadings <= threshold_neg)[0]
        
        pattern_rois[pattern_idx] = {
            'positive_rois': pos_rois,
            'negative_rois': neg_rois,
            'positive_loadings': loadings[pos_rois],
            'negative_loadings': loadings[neg_rois],
            'temporal_pattern': temporal_patterns[pattern_idx],
            'explained_variance': pca.explained_variance_ratio_[pattern_idx]
        }
        
        print(f"  Pattern {pattern_idx}: {len(pos_rois)} pos ROIs, {len(neg_rois)} neg ROIs")
        print(f"    Variance explained: {pca.explained_variance_ratio_[pattern_idx]:.3f}")
    
    return {
        'event_name': event_name,
        'temporal_patterns': temporal_patterns,  # (n_patterns, time)
        'roi_loadings': roi_loadings,           # (rois, n_patterns) 
        'pattern_rois': pattern_rois,
        'time_vector': time_vector,
        'pca_model': pca,
        'mean_response': mean_response,
        'explained_variance_ratio': pca.explained_variance_ratio_
    }




def discover_temporal_patterns_improved(stack_data: Dict[str, Any], 
                                       cfg: Dict[str, Any],
                                       n_patterns: int = 5,
                                       loading_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Improved pattern discovery with better ROI grouping
    """
    print(f"\n=== DISCOVERING TEMPORAL PATTERNS (IMPROVED) ===")
    
    stacks = stack_data['stacks']  # (trials, rois, time)
    time_vector = stack_data['time_vector']
    event_name = stack_data['event_name']
    
    # Compute trial-averaged response for each ROI
    mean_response = np.mean(stacks, axis=0)  # (rois, time)
    
    print(f"  Data shape: {mean_response.shape} (rois, time)")
    print(f"  Looking for {n_patterns} temporal patterns")
    print(f"  Loading threshold: {loading_threshold} (absolute value)")
    
    # Apply PCA to find dominant temporal patterns
    from sklearn.decomposition import PCA
    
    # PCA on ROI responses (each ROI is a sample, time points are features)
    pca = PCA(n_components=n_patterns)
    roi_loadings = pca.fit_transform(mean_response)  # (rois, n_patterns)
    temporal_patterns = pca.components_  # (n_patterns, time)
    
    print(f"  Explained variance: {pca.explained_variance_ratio_}")
    print(f"  Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Find ROIs that strongly load on each pattern using absolute threshold
    pattern_rois = {}
    for pattern_idx in range(n_patterns):
        loadings = roi_loadings[:, pattern_idx]
        
        # Normalize loadings to unit variance for consistent thresholding
        loadings_std = np.std(loadings)
        normalized_loadings = loadings / loadings_std
        
        # Find ROIs with strong positive/negative loadings (absolute threshold)
        pos_rois = np.where(normalized_loadings >= loading_threshold)[0]
        neg_rois = np.where(normalized_loadings <= -loading_threshold)[0]
        
        # Also store the actual loading values
        pos_loadings = loadings[pos_rois] if len(pos_rois) > 0 else np.array([])
        neg_loadings = loadings[neg_rois] if len(neg_rois) > 0 else np.array([])
        
        pattern_rois[pattern_idx] = {
            'positive_rois': pos_rois,
            'negative_rois': neg_rois,
            'positive_loadings': pos_loadings,
            'negative_loadings': neg_loadings,
            'all_loadings': loadings,
            'temporal_pattern': temporal_patterns[pattern_idx],
            'explained_variance': pca.explained_variance_ratio_[pattern_idx],
            'loading_stats': {
                'mean': np.mean(loadings),
                'std': np.std(loadings),
                'min': np.min(loadings),
                'max': np.max(loadings)
            }
        }
        
        print(f"  Pattern {pattern_idx}: {len(pos_rois)} pos ROIs, {len(neg_rois)} neg ROIs")
        print(f"    Variance explained: {pca.explained_variance_ratio_[pattern_idx]:.3f}")
        print(f"    Loading range: {np.min(loadings):.3f} to {np.max(loadings):.3f}")
        
        if len(pos_rois) > 0:
            print(f"    Pos loading range: {np.min(pos_loadings):.3f} to {np.max(pos_loadings):.3f}")
        if len(neg_rois) > 0:
            print(f"    Neg loading range: {np.min(neg_loadings):.3f} to {np.max(neg_loadings):.3f}")
    
    return {
        'event_name': event_name,
        'temporal_patterns': temporal_patterns,  # (n_patterns, time)
        'roi_loadings': roi_loadings,           # (rois, n_patterns) 
        'pattern_rois': pattern_rois,
        'time_vector': time_vector,
        'pca_model': pca,
        'mean_response': mean_response,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'parameters': {
            'n_patterns': n_patterns,
            'loading_threshold': loading_threshold
        }
    }

def visualize_pattern_loadings(pattern_results: Dict[str, Any], pattern_idx: int = 0):
    """Visualize the distribution of ROI loadings for a specific pattern"""
    
    pattern_info = pattern_results['pattern_rois'][pattern_idx]
    all_loadings = pattern_info['all_loadings']
    pos_rois = pattern_info['positive_rois'] 
    neg_rois = pattern_info['negative_rois']
    threshold = pattern_results['parameters']['loading_threshold']
    
    # Create histogram of loadings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Histogram of all loadings
    ax1.hist(all_loadings, bins=50, alpha=0.7, color='gray', edgecolor='black')
    ax1.axvline(0, color='black', linestyle='-', alpha=0.5)
    
    # Mark thresholds
    loadings_std = np.std(all_loadings)
    pos_threshold = threshold * loadings_std
    neg_threshold = -threshold * loadings_std
    
    ax1.axvline(pos_threshold, color='blue', linestyle='--', linewidth=2, 
               label=f'Pos threshold ({threshold}σ)')
    ax1.axvline(neg_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Neg threshold (-{threshold}σ)')
    
    # Highlight selected ROIs
    if len(pos_rois) > 0:
        ax1.hist(all_loadings[pos_rois], bins=50, alpha=0.8, color='blue', 
                label=f'Positive ROIs (n={len(pos_rois)})')
    if len(neg_rois) > 0:
        ax1.hist(all_loadings[neg_rois], bins=50, alpha=0.8, color='red',
                label=f'Negative ROIs (n={len(neg_rois)})')
    
    ax1.set_xlabel('PCA Loading')
    ax1.set_ylabel('Number of ROIs')
    ax1.set_title(f'Pattern {pattern_idx}: ROI Loading Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Scatter plot of loadings vs ROI index
    ax2.scatter(range(len(all_loadings)), all_loadings, alpha=0.5, color='gray', s=1)
    
    if len(pos_rois) > 0:
        ax2.scatter(pos_rois, all_loadings[pos_rois], color='blue', s=3, alpha=0.8,
                   label=f'Positive ROIs (n={len(pos_rois)})')
    if len(neg_rois) > 0:
        ax2.scatter(neg_rois, all_loadings[neg_rois], color='red', s=3, alpha=0.8,
                   label=f'Negative ROIs (n={len(neg_rois)})')
    
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(pos_threshold, color='blue', linestyle='--', alpha=0.7)
    ax2.axhline(neg_threshold, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('ROI Index')
    ax2.set_ylabel('PCA Loading')
    ax2.set_title(f'Pattern {pattern_idx}: ROI Loadings by Index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary stats
    print(f"\n=== PATTERN {pattern_idx} LOADING SUMMARY ===")
    print(f"Total ROIs: {len(all_loadings)}")
    print(f"Positive ROIs: {len(pos_rois)} ({100*len(pos_rois)/len(all_loadings):.1f}%)")
    print(f"Negative ROIs: {len(neg_rois)} ({100*len(neg_rois)/len(all_loadings):.1f}%)")
    print(f"Neutral ROIs: {len(all_loadings) - len(pos_rois) - len(neg_rois)}")
    print(f"Loading stats: mean={np.mean(all_loadings):.3f}, std={np.std(all_loadings):.3f}")
    print(f"Thresholds: +{pos_threshold:.3f}, -{neg_threshold:.3f}")





def visualize_discovered_patterns(pattern_results: Dict[str, Any]):
    """Visualize the discovered temporal patterns and associated ROIs"""
    
    temporal_patterns = pattern_results['temporal_patterns']
    pattern_rois = pattern_results['pattern_rois']
    time_vector = pattern_results['time_vector']
    mean_response = pattern_results['mean_response']
    event_name = pattern_results['event_name']
    
    n_patterns = temporal_patterns.shape[0]
    
    # Create subplot grid
    fig, axes = plt.subplots(n_patterns, 2, figsize=(15, 3*n_patterns))
    if n_patterns == 1:
        axes = axes.reshape(1, -1)
    
    for pattern_idx in range(n_patterns):
        pattern_info = pattern_rois[pattern_idx]
        temporal_pattern = pattern_info['temporal_pattern']
        pos_rois = pattern_info['positive_rois']
        neg_rois = pattern_info['negative_rois']
        
        # Left: Temporal pattern
        axes[pattern_idx, 0].plot(time_vector, temporal_pattern, 'k-', linewidth=2)
        axes[pattern_idx, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[pattern_idx, 0].axhline(0, color='gray', linestyle='-', alpha=0.5)
        axes[pattern_idx, 0].set_title(f'Pattern {pattern_idx}: Temporal Component\n'
                                      f'(Variance: {pattern_info["explained_variance"]:.3f})')
        axes[pattern_idx, 0].set_xlabel('Time (s)')
        axes[pattern_idx, 0].set_ylabel('Pattern Weight')
        axes[pattern_idx, 0].grid(True, alpha=0.3)
        
        # Right: ROI responses for this pattern
        if len(pos_rois) > 0:
            pos_mean = np.mean(mean_response[pos_rois], axis=0)
            axes[pattern_idx, 1].plot(time_vector, pos_mean, 'b-', linewidth=2, 
                                     label=f'Positive ROIs (n={len(pos_rois)})')
        
        if len(neg_rois) > 0:
            neg_mean = np.mean(mean_response[neg_rois], axis=0)
            axes[pattern_idx, 1].plot(time_vector, neg_mean, 'r--', linewidth=2,
                                     label=f'Negative ROIs (n={len(neg_rois)})')
        
        axes[pattern_idx, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[pattern_idx, 1].axhline(0, color='gray', linestyle='-', alpha=0.5)
        axes[pattern_idx, 1].set_title(f'Pattern {pattern_idx}: ROI Group Responses')
        axes[pattern_idx, 1].set_xlabel('Time (s)')
        axes[pattern_idx, 1].set_ylabel('z-scored dF/F')
        axes[pattern_idx, 1].legend()
        axes[pattern_idx, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{event_name}: Discovered Temporal Patterns', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    
    
    
















































def discover_event_specific_patterns(stack_data: Dict[str, Any], 
                                   cfg: Dict[str, Any],
                                   baseline_subtract: bool = True) -> Dict[str, Any]:
    """
    Discover event-specific patterns by removing baseline differences
    More similar to NMRS approach
    """
    print(f"\n=== DISCOVERING EVENT-SPECIFIC PATTERNS ===")
    
    stacks = stack_data['stacks']  # (trials, rois, time)
    time_vector = stack_data['time_vector']
    
    # Find baseline period (pre-event)
    baseline_mask = time_vector < -0.05  # 50ms before event
    event_mask = time_vector >= 0       # post-event period
    
    if baseline_subtract:
        # Subtract baseline for each trial/ROI to focus on event-evoked changes
        baseline_mean = np.mean(stacks[:, :, baseline_mask], axis=2, keepdims=True)
        stacks_corrected = stacks - baseline_mean
        print(f"  Applied baseline subtraction (pre-event mean)")
    else:
        stacks_corrected = stacks
    
    # Use post-event period only for pattern discovery
    post_event_stacks = stacks_corrected[:, :, event_mask]
    post_event_time = time_vector[event_mask]
    
    # Trial-average the baseline-corrected responses
    mean_response = np.mean(post_event_stacks, axis=0)  # (rois, time_post_event)
    
    print(f"  Post-event shape: {mean_response.shape}")
    print(f"  Post-event time: {post_event_time[0]:.3f} to {post_event_time[-1]:.3f}s")
    
    # Apply PCA to baseline-corrected, post-event responses
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=5)
    roi_loadings = pca.fit_transform(mean_response)
    temporal_patterns = pca.components_
    
    print(f"  Explained variance (event-specific): {pca.explained_variance_ratio_}")
    print(f"  Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    return {
        'event_name': stack_data['event_name'],
        'temporal_patterns': temporal_patterns,
        'roi_loadings': roi_loadings,
        'time_vector': post_event_time,
        'mean_response': mean_response,
        'baseline_corrected': baseline_subtract,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'pca_model': pca
    }




def discover_anticipatory_vs_response_patterns(stack_data: Dict[str, Any], 
                                             cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Discover anticipatory vs response patterns by analyzing pre vs post event periods separately
    """
    print(f"\n=== DISCOVERING ANTICIPATORY vs RESPONSE PATTERNS ===")
    
    stacks = stack_data['stacks']  # (trials, rois, time)
    time_vector = stack_data['time_vector']
    event_name = stack_data['event_name']
    
    # Define time periods
    pre_event_mask = time_vector < -0.05   # Anticipatory period (exclude 50ms before event)
    post_event_mask = time_vector >= 0.02  # Response period (start 20ms after event)
    baseline_mask = time_vector < -0.15    # Earlier baseline for subtraction
    
    print(f"  Anticipatory period: {np.sum(pre_event_mask)} samples")
    print(f"  Response period: {np.sum(post_event_mask)} samples")
    print(f"  Baseline period: {np.sum(baseline_mask)} samples")
    
    # Baseline subtract each trial
    baseline_mean = np.mean(stacks[:, :, baseline_mask], axis=2, keepdims=True)
    stacks_corrected = stacks - baseline_mean
    
    # Extract periods
    anticipatory_stacks = stacks_corrected[:, :, pre_event_mask]
    response_stacks = stacks_corrected[:, :, post_event_mask]
    
    anticipatory_time = time_vector[pre_event_mask]
    response_time = time_vector[post_event_mask]
    
    # Trial-average each period
    anticipatory_mean = np.mean(anticipatory_stacks, axis=0)  # (rois, time_pre)
    response_mean = np.mean(response_stacks, axis=0)         # (rois, time_post)
    
    print(f"  Anticipatory shape: {anticipatory_mean.shape}")
    print(f"  Response shape: {response_mean.shape}")
    
    # Run PCA on each period separately
    from sklearn.decomposition import PCA
    
    # Anticipatory patterns
    pca_antic = PCA(n_components=3)
    if anticipatory_mean.shape[1] > 0:
        roi_loadings_antic = pca_antic.fit_transform(anticipatory_mean)
        temporal_patterns_antic = pca_antic.components_
        print(f"  Anticipatory explained variance: {pca_antic.explained_variance_ratio_}")
    else:
        roi_loadings_antic = np.zeros((anticipatory_mean.shape[0], 3))
        temporal_patterns_antic = np.zeros((3, 0))
        print(f"  No anticipatory period available")
    
    # Response patterns  
    pca_resp = PCA(n_components=3)
    if response_mean.shape[1] > 0:
        roi_loadings_resp = pca_resp.fit_transform(response_mean)
        temporal_patterns_resp = pca_resp.components_
        print(f"  Response explained variance: {pca_resp.explained_variance_ratio_}")
    else:
        roi_loadings_resp = np.zeros((response_mean.shape[0], 3))
        temporal_patterns_resp = np.zeros((3, 0))
        print(f"  No response period available")
    
    return {
        'event_name': event_name,
        'anticipatory': {
            'temporal_patterns': temporal_patterns_antic,
            'roi_loadings': roi_loadings_antic,
            'time_vector': anticipatory_time,
            'mean_response': anticipatory_mean,
            'explained_variance': pca_antic.explained_variance_ratio_ if anticipatory_mean.shape[1] > 0 else [],
            'pca_model': pca_antic
        },
        'response': {
            'temporal_patterns': temporal_patterns_resp,
            'roi_loadings': roi_loadings_resp,
            'time_vector': response_time,
            'mean_response': response_mean,
            'explained_variance': pca_resp.explained_variance_ratio_ if response_mean.shape[1] > 0 else [],
            'pca_model': pca_resp
        },
        'baseline_corrected': True
    }

def visualize_anticipatory_vs_response(pattern_results: Dict[str, Any], 
                                     loading_threshold: float = 0.5):
    """Visualize anticipatory vs response patterns side by side"""
    
    event_name = pattern_results['event_name']
    antic_data = pattern_results['anticipatory']
    resp_data = pattern_results['response']
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    
    for pattern_idx in range(3):
        row = pattern_idx
        
        # === ANTICIPATORY PATTERNS ===
        if antic_data['temporal_patterns'].shape[1] > 0:
            # Temporal pattern
            axes[row, 0].plot(antic_data['time_vector'], 
                             antic_data['temporal_patterns'][pattern_idx], 
                             'purple', linewidth=2)
            axes[row, 0].axhline(0, color='gray', linestyle='-', alpha=0.5)
            axes[row, 0].set_title(f'Anticipatory Pattern {pattern_idx}\n'
                                  f'Var: {antic_data["explained_variance"][pattern_idx]:.3f}')
            axes[row, 0].set_xlabel('Time (s)')
            axes[row, 0].set_ylabel('Pattern Weight')
            axes[row, 0].grid(True, alpha=0.3)
            
            # ROI loadings
            loadings = antic_data['roi_loadings'][:, pattern_idx]
            loadings_std = np.std(loadings)
            
            pos_rois = np.where(loadings >= loading_threshold * loadings_std)[0]
            neg_rois = np.where(loadings <= -loading_threshold * loadings_std)[0]
            
            if len(pos_rois) > 0 or len(neg_rois) > 0:
                if len(pos_rois) > 0:
                    pos_mean = np.mean(antic_data['mean_response'][pos_rois], axis=0)
                    axes[row, 1].plot(antic_data['time_vector'], pos_mean, 'b-', 
                                     linewidth=2, label=f'Pos (n={len(pos_rois)})')
                
                if len(neg_rois) > 0:
                    neg_mean = np.mean(antic_data['mean_response'][neg_rois], axis=0)
                    axes[row, 1].plot(antic_data['time_vector'], neg_mean, 'r--', 
                                     linewidth=2, label=f'Neg (n={len(neg_rois)})')
                
                axes[row, 1].axhline(0, color='gray', linestyle='-', alpha=0.5)
                axes[row, 1].set_title(f'Anticipatory ROI Groups {pattern_idx}')
                axes[row, 1].set_xlabel('Time (s)')
                axes[row, 1].set_ylabel('Baseline-corrected dF/F')
                axes[row, 1].legend()
                axes[row, 1].grid(True, alpha=0.3)
            else:
                axes[row, 1].text(0.5, 0.5, 'No significant ROIs', 
                                 ha='center', va='center', transform=axes[row, 1].transAxes)
        else:
            axes[row, 0].text(0.5, 0.5, 'No anticipatory data', 
                             ha='center', va='center', transform=axes[row, 0].transAxes)
            axes[row, 1].text(0.5, 0.5, 'No anticipatory data', 
                             ha='center', va='center', transform=axes[row, 1].transAxes)
        
        # === RESPONSE PATTERNS ===
        if resp_data['temporal_patterns'].shape[1] > 0:
            # Temporal pattern
            axes[row, 2].plot(resp_data['time_vector'], 
                             resp_data['temporal_patterns'][pattern_idx], 
                             'orange', linewidth=2)
            axes[row, 2].axhline(0, color='gray', linestyle='-', alpha=0.5)
            axes[row, 2].set_title(f'Response Pattern {pattern_idx}\n'
                                  f'Var: {resp_data["explained_variance"][pattern_idx]:.3f}')
            axes[row, 2].set_xlabel('Time (s)')
            axes[row, 2].set_ylabel('Pattern Weight')
            axes[row, 2].grid(True, alpha=0.3)
            
            # ROI loadings
            loadings = resp_data['roi_loadings'][:, pattern_idx]
            loadings_std = np.std(loadings)
            
            pos_rois = np.where(loadings >= loading_threshold * loadings_std)[0]
            neg_rois = np.where(loadings <= -loading_threshold * loadings_std)[0]
            
            if len(pos_rois) > 0 or len(neg_rois) > 0:
                if len(pos_rois) > 0:
                    pos_mean = np.mean(resp_data['mean_response'][pos_rois], axis=0)
                    axes[row, 3].plot(resp_data['time_vector'], pos_mean, 'b-', 
                                     linewidth=2, label=f'Pos (n={len(pos_rois)})')
                
                if len(neg_rois) > 0:
                    neg_mean = np.mean(resp_data['mean_response'][neg_rois], axis=0)
                    axes[row, 3].plot(resp_data['time_vector'], neg_mean, 'r--', 
                                     linewidth=2, label=f'Neg (n={len(neg_rois)})')
                
                axes[row, 3].axhline(0, color='gray', linestyle='-', alpha=0.5)
                axes[row, 3].set_title(f'Response ROI Groups {pattern_idx}')
                axes[row, 3].set_xlabel('Time (s)')
                axes[row, 3].set_ylabel('Baseline-corrected dF/F')
                axes[row, 3].legend()
                axes[row, 3].grid(True, alpha=0.3)
            else:
                axes[row, 3].text(0.5, 0.5, 'No significant ROIs', 
                                 ha='center', va='center', transform=axes[row, 3].transAxes)
        else:
            axes[row, 2].text(0.5, 0.5, 'No response data', 
                             ha='center', va='center', transform=axes[row, 2].transAxes)
            axes[row, 3].text(0.5, 0.5, 'No response data', 
                             ha='center', va='center', transform=axes[row, 3].transAxes)
    
    plt.suptitle(f'{event_name}: Anticipatory vs Response Patterns', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n=== ANTICIPATORY vs RESPONSE SUMMARY ===")
    if len(antic_data['explained_variance']) > 0:
        print(f"Anticipatory patterns explain: {np.sum(antic_data['explained_variance'][:3]):.3f} of variance")
    if len(resp_data['explained_variance']) > 0:
        print(f"Response patterns explain: {np.sum(resp_data['explained_variance'][:3]):.3f} of variance")




def analyze_roi_overlap(pattern_results: Dict[str, Any]):
    """Analyze how many ROIs are selected for multiple patterns"""
    
    antic_data = pattern_results['anticipatory']
    resp_data = pattern_results['response']
    
    print("=== ROI OVERLAP ANALYSIS ===")
    
    # Check anticipatory patterns
    print("\nANTICIPATORY PATTERNS:")
    antic_all_selected = set()
    antic_pattern_rois = {}
    
    for i in range(3):
        loadings = antic_data['roi_loadings'][:, i]
        loadings_std = np.std(loadings)
        
        pos_rois = set(np.where(loadings >= 0.5 * loadings_std)[0])
        neg_rois = set(np.where(loadings <= -0.5 * loadings_std)[0])
        pattern_rois = pos_rois | neg_rois
        
        antic_pattern_rois[i] = pattern_rois
        antic_all_selected |= pattern_rois
        
        print(f"  Pattern {i}: {len(pattern_rois)} ROIs")
    
    print(f"  Total unique ROIs across all anticipatory patterns: {len(antic_all_selected)}")
    
    # Check response patterns
    print("\nRESPONSE PATTERNS:")
    resp_all_selected = set()
    resp_pattern_rois = {}
    
    for i in range(3):
        loadings = resp_data['roi_loadings'][:, i]
        loadings_std = np.std(loadings)
        
        pos_rois = set(np.where(loadings >= 0.5 * loadings_std)[0])
        neg_rois = set(np.where(loadings <= -0.5 * loadings_std)[0])
        pattern_rois = pos_rois | neg_rois
        
        resp_pattern_rois[i] = pattern_rois
        resp_all_selected |= pattern_rois
        
        print(f"  Pattern {i}: {len(pattern_rois)} ROIs")
    
    print(f"  Total unique ROIs across all response patterns: {len(resp_all_selected)}")
    
    # Check overlap between anticipatory and response
    antic_resp_overlap = antic_all_selected & resp_all_selected
    print(f"\nROIs active in BOTH anticipatory and response: {len(antic_resp_overlap)}")
    print(f"Only anticipatory: {len(antic_all_selected - resp_all_selected)}")
    print(f"Only response: {len(resp_all_selected - antic_all_selected)}")
    print(f"Total active ROIs: {len(antic_all_selected | resp_all_selected)}")
    
    # Create overlap matrix
    print("\nPATTERN OVERLAP MATRIX (Anticipatory):")
    for i in range(3):
        for j in range(i+1, 3):
            overlap = len(antic_pattern_rois[i] & antic_pattern_rois[j])
            print(f"  Pattern {i} ∩ Pattern {j}: {overlap} ROIs")
    
    print("\nPATTERN OVERLAP MATRIX (Response):")
    for i in range(3):
        for j in range(i+1, 3):
            overlap = len(resp_pattern_rois[i] & resp_pattern_rois[j])
            print(f"  Pattern {i} ∩ Pattern {j}: {overlap} ROIs")





























def filter_trials_by_component_activity(stack_data: Dict[str, Any],
                                       pattern_results: Dict[str, Any],
                                       component_idx: int,
                                       activity_threshold: float = 1.0,
                                       min_roi_fraction: float = 0.3) -> Dict[str, Any]:
    """
    Filter trials based on whether ROIs show activity consistent with a specific component
    
    Args:
        stack_data: Event-aligned stack data
        pattern_results: Results from pattern discovery
        component_idx: Which component to filter by
        activity_threshold: How many std above baseline to consider "active"
        min_roi_fraction: Minimum fraction of component ROIs that must be active
    
    Returns:
        Dict with trial indices and activity measures
    """
    print(f"\n=== FILTERING TRIALS BY COMPONENT {component_idx} ACTIVITY ===")
    
    stacks = stack_data['stacks']  # (trials, rois, time)
    time_vector = stack_data['time_vector']
    
    # Get ROIs for this component
    if 'anticipatory' in pattern_results:
        # Handle anticipatory vs response results
        component_data = pattern_results['response']  # or 'anticipatory'
    else:
        # Handle regular pattern results
        component_data = pattern_results
    
    pattern_info = component_data['pattern_rois'][component_idx] if 'pattern_rois' in component_data else None
    
    if pattern_info is None:
        # Use loadings directly
        loadings = component_data['roi_loadings'][:, component_idx]
        loadings_std = np.std(loadings)
        pos_rois = np.where(loadings >= 0.5 * loadings_std)[0]
        neg_rois = np.where(loadings <= -0.5 * loadings_std)[0]
    else:
        pos_rois = pattern_info['positive_rois']
        neg_rois = pattern_info['negative_rois']
    
    component_rois = np.concatenate([pos_rois, neg_rois])
    
    print(f"  Component {component_idx}: {len(pos_rois)} pos, {len(neg_rois)} neg ROIs")
    print(f"  Total component ROIs: {len(component_rois)}")
    
    # Define time windows for activity measurement
    baseline_mask = time_vector < -0.05  # Pre-event baseline
    response_mask = time_vector >= 0.02   # Post-event response
    
    n_trials = stacks.shape[0]
    trial_activities = []
    trial_scores = []
    
    for trial_idx in range(n_trials):
        trial_stack = stacks[trial_idx]  # (rois, time)
        
        # Calculate baseline and response activity for component ROIs
        baseline_activity = np.mean(trial_stack[component_rois][:, baseline_mask], axis=1)
        response_activity = np.mean(trial_stack[component_rois][:, response_mask], axis=1)
        
        # Calculate z-scored activity (response relative to baseline)
        baseline_std = np.std(trial_stack[component_rois][:, baseline_mask], axis=1)
        baseline_std[baseline_std == 0] = 1  # Avoid division by zero
        
        activity_zscore = (response_activity - baseline_activity) / baseline_std
        
        # For positive ROIs, look for positive activity; for negative ROIs, look for negative activity
        pos_activity = activity_zscore[:len(pos_rois)] if len(pos_rois) > 0 else np.array([])
        neg_activity = -activity_zscore[len(pos_rois):] if len(neg_rois) > 0 else np.array([])  # Flip sign
        
        # Count how many ROIs are "active" (above threshold)
        pos_active = np.sum(pos_activity >= activity_threshold) if len(pos_activity) > 0 else 0
        neg_active = np.sum(neg_activity >= activity_threshold) if len(neg_activity) > 0 else 0
        total_active = pos_active + neg_active
        
        # Calculate activity fraction
        activity_fraction = total_active / len(component_rois) if len(component_rois) > 0 else 0
        
        # Overall component score (mean activity of all component ROIs)
        component_score = np.mean(np.concatenate([pos_activity, neg_activity])) if len(component_rois) > 0 else 0
        
        trial_activities.append({
            'trial_idx': trial_idx,
            'pos_active': pos_active,
            'neg_active': neg_active,
            'total_active': total_active,
            'activity_fraction': activity_fraction,
            'component_score': component_score,
            'pos_activity': pos_activity,
            'neg_activity': neg_activity
        })
        
        trial_scores.append(component_score)
    
    # Filter trials based on activity criteria
    active_trials = [t['trial_idx'] for t in trial_activities 
                    if t['activity_fraction'] >= min_roi_fraction]
    
    inactive_trials = [t['trial_idx'] for t in trial_activities 
                      if t['activity_fraction'] < min_roi_fraction]
    
    # Sort by component score
    sorted_trials = sorted(range(n_trials), key=lambda i: trial_scores[i], reverse=True)
    top_trials = sorted_trials[:int(0.3 * n_trials)]  # Top 30%
    bottom_trials = sorted_trials[-int(0.3 * n_trials):]  # Bottom 30%
    
    print(f"  Active trials (≥{min_roi_fraction:.1%} ROIs active): {len(active_trials)}")
    print(f"  Inactive trials: {len(inactive_trials)}")
    print(f"  Top 30% by component score: {len(top_trials)}")
    print(f"  Bottom 30% by component score: {len(bottom_trials)}")
    
    return {
        'component_idx': component_idx,
        'component_rois': component_rois,
        'pos_rois': pos_rois,
        'neg_rois': neg_rois,
        'trial_activities': trial_activities,
        'trial_scores': np.array(trial_scores),
        'active_trials': active_trials,
        'inactive_trials': inactive_trials,
        'top_trials': top_trials,
        'bottom_trials': bottom_trials,
        'parameters': {
            'activity_threshold': activity_threshold,
            'min_roi_fraction': min_roi_fraction
        }
    }



def visualize_component_trial_filtering(stack_data: Dict[str, Any],
                                       filter_results: Dict[str, Any],
                                       data: Optional[Dict[str, Any]] = None,
                                       use_zscore: bool = True):
    """
    Visualize the results of component-based trial filtering
    
    Args:
        stack_data: Event-aligned stack data (z-scored)
        filter_results: Results from component filtering
        data: Original SID data (for non-z-scored plotting)
        use_zscore: If True, plot z-scored dF/F; if False, plot original dF/F
    """
    
    if not use_zscore and data is None:
        raise ValueError("Original data must be provided when use_zscore=False")
    
    # Choose which data to plot
    if use_zscore:
        stacks = stack_data['stacks']  # z-scored
        ylabel = 'z-scored dF/F'
        cmap = 'RdBu_r'
        title_suffix = "(z-scored)"
    else:
        # Create original dF/F stacks from the raw data
        print("  Creating original dF/F stacks for visualization...")
        stacks_orig = _create_original_stacks(stack_data, data)
        stacks = stacks_orig
        ylabel = 'dF/F'
        cmap = 'viridis'  # Better for positive-only values
        title_suffix = "(original dF/F)"
    
    time_vector = stack_data['time_vector']
    event_name = stack_data['event_name']
    
    component_idx = filter_results['component_idx']
    component_rois = filter_results['component_rois']
    active_trials = filter_results['active_trials']
    inactive_trials = filter_results['inactive_trials']
    top_trials = filter_results['top_trials']
    bottom_trials = filter_results['bottom_trials']
    trial_scores = filter_results['trial_scores']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Top row: Trial score distribution and selection
    
    # Left: Histogram of component scores
    axes[0, 0].hist(trial_scores, bins=30, alpha=0.7, color='gray', edgecolor='black')
    axes[0, 0].axvline(np.mean(trial_scores), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(trial_scores):.2f}')
    axes[0, 0].set_xlabel('Component Score')
    axes[0, 0].set_ylabel('Number of Trials')
    axes[0, 0].set_title(f'Component {component_idx}: Trial Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Middle: Active vs inactive trial responses
    if len(active_trials) > 0 and len(inactive_trials) > 0:
        active_mean = np.mean(stacks[active_trials][:, component_rois, :], axis=(0, 1))
        inactive_mean = np.mean(stacks[inactive_trials][:, component_rois, :], axis=(0, 1))
        
        axes[0, 1].plot(time_vector, active_mean, 'b-', linewidth=2, 
                       label=f'Active trials (n={len(active_trials)})')
        axes[0, 1].plot(time_vector, inactive_mean, 'r-', linewidth=2,
                       label=f'Inactive trials (n={len(inactive_trials)})')
        axes[0, 1].axvline(0, color='black', linestyle='--', alpha=0.7)
        
        # Only add zero line for z-scored data
        if use_zscore:
            axes[0, 1].axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel(ylabel)
        axes[0, 1].set_title(f'Component ROIs: Active vs Inactive Trials {title_suffix}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Insufficient trials for comparison', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # Right: Top vs bottom trial responses
    if len(top_trials) > 0 and len(bottom_trials) > 0:
        top_mean = np.mean(stacks[top_trials][:, component_rois, :], axis=(0, 1))
        bottom_mean = np.mean(stacks[bottom_trials][:, component_rois, :], axis=(0, 1))
        
        axes[0, 2].plot(time_vector, top_mean, 'g-', linewidth=2,
                       label=f'Top 30% (n={len(top_trials)})')
        axes[0, 2].plot(time_vector, bottom_mean, 'orange', linewidth=2,
                       label=f'Bottom 30% (n={len(bottom_trials)})')
        axes[0, 2].axvline(0, color='black', linestyle='--', alpha=0.7)
        
        # Only add zero line for z-scored data
        if use_zscore:
            axes[0, 2].axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel(ylabel)
        axes[0, 2].set_title(f'Component ROIs: Top vs Bottom Trials {title_suffix}')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'Insufficient trials for comparison', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
    
    # Bottom row: Heatmaps
    
    # Left: All trials, component ROIs only
    component_data = stacks[:, component_rois, :]
    trial_avg = np.mean(component_data, axis=0)  # Average across trials
    
    # Set colormap limits appropriately
    if use_zscore:
        vmin, vmax = -3, 3  # Standard z-score range
    else:
        vmin, vmax = np.percentile(trial_avg, [5, 95])  # 5th to 95th percentile
    
    im1 = axes[1, 0].imshow(trial_avg, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=[time_vector[0], time_vector[-1], 0, len(component_rois)])
    axes[1, 0].axvline(0, color='white', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Component ROI Index')
    axes[1, 0].set_title(f'Component {component_idx} ROIs: All Trials {title_suffix}')
    plt.colorbar(im1, ax=axes[1, 0], label=ylabel)
    
    # Middle: Active trials only
    if len(active_trials) > 0:
        active_avg = np.mean(stacks[active_trials][:, component_rois, :], axis=0)
        im2 = axes[1, 1].imshow(active_avg, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                               extent=[time_vector[0], time_vector[-1], 0, len(component_rois)])
        axes[1, 1].axvline(0, color='white', linestyle='--', alpha=0.8)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Component ROI Index')
        axes[1, 1].set_title(f'Active Trials Only (n={len(active_trials)}) {title_suffix}')
        plt.colorbar(im2, ax=axes[1, 1], label=ylabel)
    else:
        axes[1, 1].text(0.5, 0.5, 'No active trials', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # Right: Top trials only
    if len(top_trials) > 0:
        top_avg = np.mean(stacks[top_trials][:, component_rois, :], axis=0)
        im3 = axes[1, 2].imshow(top_avg, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                               extent=[time_vector[0], time_vector[-1], 0, len(component_rois)])
        axes[1, 2].axvline(0, color='white', linestyle='--', alpha=0.8)
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Component ROI Index')
        axes[1, 2].set_title(f'Top 30% Trials (n={len(top_trials)}) {title_suffix}')
        plt.colorbar(im3, ax=axes[1, 2], label=ylabel)
    else:
        axes[1, 2].text(0.5, 0.5, 'No top trials', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
    
    plt.suptitle(f'{event_name}: Component {component_idx} Trial Filtering Results {title_suffix}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n=== COMPONENT {component_idx} TRIAL FILTERING SUMMARY {title_suffix} ===")
    print(f"Component ROIs: {len(component_rois)}")
    print(f"Active trials: {len(active_trials)} ({100*len(active_trials)/len(trial_scores):.1f}%)")
    print(f"Inactive trials: {len(inactive_trials)} ({100*len(inactive_trials)/len(trial_scores):.1f}%)")
    print(f"Score range: {np.min(trial_scores):.2f} to {np.max(trial_scores):.2f}")
    print(f"Score mean ± std: {np.mean(trial_scores):.2f} ± {np.std(trial_scores):.2f}")

def _create_original_stacks(stack_data: Dict[str, Any], data: Dict[str, Any]) -> np.ndarray:
    """
    Create original dF/F stacks (non-z-scored) from the raw data
    Uses the same trial indices and time windows as the z-scored stacks
    """
    # Get original dF/F data
    dff_clean = data['dFF_clean']  # (n_rois, n_timepoints)
    imaging_time = data['imaging_time']
    
    # Get trial metadata to reconstruct the same trials
    trial_metadata = stack_data['trial_metadata']
    time_vector = stack_data['time_vector']
    
    # Parameters for reconstruction
    target_n_samples = len(time_vector)
    pre_event_s = abs(time_vector[0])
    post_event_s = time_vector[-1]
    
    n_rois = dff_clean.shape[0]
    n_trials = len(trial_metadata)
    
    original_stacks = np.zeros((n_trials, n_rois, target_n_samples))
    
    for trial_idx, trial_meta in enumerate(trial_metadata):
        # Get the same time indices that were used for this trial
        start_idx = trial_meta['start_idx']
        end_idx = trial_meta['end_idx']
        
        # Extract the original dF/F data
        trial_stack = dff_clean[:, start_idx:end_idx]
        
        # Handle potential size mismatches due to rounding
        if trial_stack.shape[1] == target_n_samples:
            original_stacks[trial_idx] = trial_stack
        elif trial_stack.shape[1] < target_n_samples:
            # Pad with the last value
            original_stacks[trial_idx, :, :trial_stack.shape[1]] = trial_stack
            original_stacks[trial_idx, :, trial_stack.shape[1]:] = trial_stack[:, -1:]
        else:
            # Truncate
            original_stacks[trial_idx] = trial_stack[:, :target_n_samples]
    
    return original_stacks



































































































def check_dpca_condition_distribution(data: Dict[str, Any]) -> Dict[str, Any]:
    """Check how many trials we have for each ISI×choice×outcome combination"""
    print(f"\n=== CHECKING dPCA CONDITION DISTRIBUTION ===")
    
    unique_isis = data['unique_isis']
    df_trials = data['df_trials']
    
    print(f"Unique ISIs: {unique_isis}")
    print(f"Total trials: {len(df_trials)}")
    
    # Create condition matrix
    condition_counts = {}
    total_combinations = 0
    valid_combinations = 0
    
    for isi in unique_isis:
        for choice in [0, 1]:  # left, right (is_right_choice)
            for outcome in [0, 1]:  # not rewarded, rewarded
                # Find trials matching this combination
                condition_mask = (
                    (df_trials['isi'] == isi) & 
                    (df_trials['is_right_choice'] == choice) & 
                    (df_trials['rewarded'] == outcome)
                )
                
                trial_count = np.sum(condition_mask)
                condition_name = f'isi_{isi:.0f}_choice_{choice}_outcome_{outcome}'
                condition_counts[condition_name] = trial_count
                
                total_combinations += 1
                if trial_count > 0:
                    valid_combinations += 1
    
    print(f"\nCondition breakdown:")
    print(f"  Total possible combinations: {total_combinations}")
    print(f"  Combinations with trials: {valid_combinations}")
    
    # Print detailed counts
    print(f"\nDetailed trial counts:")
    for condition, count in sorted(condition_counts.items()):
        isi_val = float(condition.split('_')[1])
        choice_val = int(condition.split('_')[3])
        outcome_val = int(condition.split('_')[5])
        choice_str = 'Right' if choice_val else 'Left'
        outcome_str = 'Rewarded' if outcome_val else 'Not Rewarded'
        print(f"  ISI {isi_val:.0f}ms, {choice_str}, {outcome_str}: {count} trials")
    
    # Summary statistics
    counts_array = np.array(list(condition_counts.values()))
    non_zero_counts = counts_array[counts_array > 0]
    
    print(f"\nSummary statistics:")
    print(f"  Min trials (non-zero): {np.min(non_zero_counts) if len(non_zero_counts) > 0 else 0}")
    print(f"  Max trials: {np.max(counts_array)}")
    print(f"  Mean trials (non-zero): {np.mean(non_zero_counts):.1f}")
    print(f"  Median trials (non-zero): {np.median(non_zero_counts):.1f}")
    
    return {
        'condition_counts': condition_counts,
        'total_combinations': total_combinations,
        'valid_combinations': valid_combinations,
        'min_trials': np.min(non_zero_counts) if len(non_zero_counts) > 0 else 0,
        'max_trials': np.max(counts_array),
        'mean_trials': np.mean(non_zero_counts) if len(non_zero_counts) > 0 else 0
    }

def extract_isi_phase_normalized(data: Dict[str, Any], 
                                cfg: Dict[str, Any],
                                n_phase_samples: int = 100,
                                use_zscore: bool = False) -> Dict[str, Any]:
    """
    Extract ISI periods and normalize to 0-1 phase across all trials
    
    Args:
        data: SID data dictionary  
        cfg: Configuration
        n_phase_samples: Number of samples across ISI phase (0 to 1)
        use_zscore: If True, use z-scored dF/F; if False, use raw dF/F
    """
    print(f"\n=== EXTRACTING ISI PHASE-NORMALIZED DATA ===")
    print(f"  Phase samples: {n_phase_samples}")
    print(f"  Data type: {'z-scored dF/F' if use_zscore else 'raw dF/F'}")
    
    df_trials = data['df_trials']
    
    # Choose data type based on config
    if use_zscore:
        dff_data = zscore(data['dFF_clean'], axis=1)  # z-score per ROI
        print(f"  Applied z-scoring per ROI")
    else:
        dff_data = data['dFF_clean']  # raw dF/F
        print(f"  Using raw dF/F data")
    
    imaging_time = data['imaging_time']
    imaging_fs = data['imaging_fs']
    n_rois = dff_data.shape[0]
    
    print(f"  ROIs: {n_rois}")
    print(f"  Imaging sampling: {imaging_fs:.1f} Hz")
    
    # Phase grid from 0 to 1
    target_phases = np.linspace(0, 1, n_phase_samples)
    
    # Extract ISI segments for each trial
    isi_phase_data = []
    trial_metadata = []
    failed_extractions = 0
    
    for trial_idx, trial in df_trials.iterrows():
        # Skip if critical ISI events are missing
        if pd.isna(trial['end_flash_1']) or pd.isna(trial['start_flash_2']):
            failed_extractions += 1
            continue
        
        # Calculate ISI period times (relative to trial_start_timestamp)
        isi_start_abs = trial['trial_start_timestamp'] + trial['end_flash_1']
        isi_end_abs = trial['trial_start_timestamp'] + trial['start_flash_2']
        isi_duration = isi_end_abs - isi_start_abs
        
        # Verify ISI duration matches expected
        expected_isi_duration = trial['isi'] / 1000.0  # Convert ms to s
        if abs(isi_duration - expected_isi_duration) > 0.1:  # 100ms tolerance
            print(f"    Trial {trial_idx}: ISI duration mismatch ({isi_duration:.3f}s vs {expected_isi_duration:.3f}s)")
            failed_extractions += 1
            continue
        
        # Find imaging indices for ISI period
        isi_start_idx = np.argmin(np.abs(imaging_time - isi_start_abs))
        isi_end_idx = np.argmin(np.abs(imaging_time - isi_end_abs))
        
        # Check if we have enough samples in ISI
        isi_n_samples = isi_end_idx - isi_start_idx
        if isi_n_samples < 3:  # Need at least 3 samples for interpolation
            failed_extractions += 1
            continue
        
        # Extract ISI segment
        isi_segment = dff_data[:, isi_start_idx:isi_end_idx]  # (rois, isi_samples)
        
        # Create phase vector for this ISI
        isi_time_points = imaging_time[isi_start_idx:isi_end_idx]
        isi_phases = (isi_time_points - isi_start_abs) / isi_duration  # 0 to 1
        
        # Interpolate each ROI to common phase grid
        from scipy.interpolate import interp1d
        isi_phase_interpolated = np.zeros((n_rois, n_phase_samples))
        
        for roi_idx in range(n_rois):
            roi_trace = isi_segment[roi_idx]
            
            # Skip if all NaN
            if np.all(np.isnan(roi_trace)):
                isi_phase_interpolated[roi_idx] = np.nan
                continue
            
            # Handle NaN values
            valid_mask = np.isfinite(roi_trace)
            if np.sum(valid_mask) < 2:
                isi_phase_interpolated[roi_idx] = np.nan
                continue
            
            # Interpolate to phase grid
            try:
                interp_func = interp1d(isi_phases[valid_mask], roi_trace[valid_mask], 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                isi_phase_interpolated[roi_idx] = interp_func(target_phases)
            except:
                isi_phase_interpolated[roi_idx] = np.nan
                continue
        
        isi_phase_data.append(isi_phase_interpolated)
        
        # Store trial metadata
        trial_meta = {
            'trial_idx': trial_idx,
            'isi_ms': trial['isi'],
            'isi_duration_s': isi_duration,
            'is_right_choice': trial['is_right_choice'], 
            'rewarded': trial['rewarded'],
            'isi_start_abs': isi_start_abs,
            'isi_end_abs': isi_end_abs,
            'isi_start_idx': isi_start_idx,
            'isi_end_idx': isi_end_idx,
            'original_isi_samples': isi_n_samples
        }
        trial_metadata.append(trial_meta)
    
    if len(isi_phase_data) == 0:
        raise ValueError("No valid ISI segments extracted")
    
    # Convert to array
    isi_phase_array = np.stack(isi_phase_data, axis=0)  # (trials, rois, phase_samples)
    
    print(f"  Successfully extracted: {len(isi_phase_data)} trials")
    print(f"  Failed extractions: {failed_extractions}")
    print(f"  ISI phase array shape: {isi_phase_array.shape}")
    print(f"  Phase resolution: {1/n_phase_samples:.3f} per sample")
    
    return {
        'isi_phase_data': isi_phase_array,  # (trials, rois, phase_samples)
        'trial_metadata': trial_metadata,
        'target_phases': target_phases,  # 0 to 1
        'n_phase_samples': n_phase_samples,
        'use_zscore': use_zscore,
        'data_type': 'z-scored dF/F' if use_zscore else 'raw dF/F',
        'n_successful_trials': len(isi_phase_data),
        'n_failed_trials': failed_extractions
    }

def run_dpca_on_isi_phases(isi_phase_data: Dict[str, Any], 
                          min_trials_per_condition: int = 0) -> Dict[str, Any]:
    """
    Run dPCA analysis on ISI phase-normalized data
    """
    print(f"\n=== RUNNING dPCA ON ISI PHASES ===")
    print(f"  Minimum trials per condition: {min_trials_per_condition}")
    
    # try:
    #     from dpca import dPCA
    # except ImportError:
    #     print("ERROR: dPCA package not found. Install with: pip install dpca")
    #     return None
    
    isi_array = isi_phase_data['isi_phase_data']  # (trials, rois, phase_samples)
    trial_metadata = isi_phase_data['trial_metadata']
    target_phases = isi_phase_data['target_phases']
    
    print(f"  Input shape: {isi_array.shape}")
    
    # Group trials by conditions
    unique_isis = sorted(list(set([meta['isi_ms'] for meta in trial_metadata])))
    
    conditions = {}
    condition_counts = {}
    condition_labels = []
    
    for isi in unique_isis:
        for choice in [0, 1]:  # left, right
            for outcome in [0, 1]:  # not rewarded, rewarded
                # Find trials matching this combination
                matching_trials = []
                for i, meta in enumerate(trial_metadata):
                    if (meta['isi_ms'] == isi and 
                        meta['is_right_choice'] == choice and 
                        meta['rewarded'] == outcome):
                        matching_trials.append(i)
                
                if len(matching_trials) >= min_trials_per_condition:
                    condition_name = f'isi_{isi:.0f}_choice_{choice}_outcome_{outcome}'
                    conditions[condition_name] = matching_trials
                    condition_counts[condition_name] = len(matching_trials)
                    condition_labels.append(condition_name)
    
    print(f"  Valid conditions: {len(conditions)}")
    for condition, count in condition_counts.items():
        print(f"    {condition}: {count} trials")
    
    if len(conditions) == 0:
        raise ValueError("No conditions meet minimum trial requirements")
    
    # Balance trial counts across conditions
    min_trials = min(condition_counts.values())
    print(f"  Balancing to {min_trials} trials per condition")
    
    # Create balanced condition tensor
    balanced_data = []
    final_condition_labels = []
    
    for condition_name, trial_indices in conditions.items():
        # Randomly subsample to minimum count
        if len(trial_indices) > min_trials:
            selected_trials = np.random.choice(trial_indices, min_trials, replace=False)
        else:
            selected_trials = trial_indices
        
        condition_data = isi_array[selected_trials]  # (min_trials, rois, phase_samples)
        balanced_data.append(condition_data)
        final_condition_labels.append(condition_name)
    
    # Stack into dPCA format: (conditions, trials, rois, phase_samples)
    dpca_tensor = np.stack(balanced_data, axis=0)
    print(f"  Balanced tensor shape: {dpca_tensor.shape}")
    
    # Average across trials for each condition (required for dPCA)
    dpca_avg = np.mean(dpca_tensor, axis=1)  # (conditions, rois, phase_samples)
    print(f"  Condition-averaged shape: {dpca_avg.shape}")
    
    # Parse condition labels for dPCA structure
    isi_values = []
    choice_values = []
    outcome_values = []
    
    for label in final_condition_labels:
        parts = label.split('_')
        isi_val = float(parts[1])
        choice_val = int(parts[3])
        outcome_val = int(parts[5])
        
        isi_values.append(isi_val)
        choice_values.append(choice_val)
        outcome_values.append(outcome_val)
    
    # Map ISI values to indices
    unique_isis_final = sorted(list(set(isi_values)))
    isi_indices = [unique_isis_final.index(isi) for isi in isi_values]
    
    # Create full tensor for dPCA: (isi, choice, outcome, rois, phase_samples)
    n_isis = len(unique_isis_final)
    n_choices = 2
    n_outcomes = 2
    n_rois, n_phases = dpca_avg.shape[1:]
    
    dpca_full_tensor = np.full((n_isis, n_choices, n_outcomes, n_rois, n_phases), np.nan)
    
    # Fill tensor with available conditions
    for i, label in enumerate(final_condition_labels):
        isi_idx = isi_indices[i]
        choice_idx = choice_values[i]
        outcome_idx = outcome_values[i]
        
        dpca_full_tensor[isi_idx, choice_idx, outcome_idx] = dpca_avg[i]
    
    # Handle missing conditions
    missing_conditions = np.isnan(dpca_full_tensor).any(axis=(3, 4))
    n_missing = np.sum(missing_conditions)
    
    if n_missing > 0:
        print(f"  WARNING: {n_missing} condition combinations are missing - filling with zeros")
        dpca_full_tensor[np.isnan(dpca_full_tensor)] = 0
    
    print(f"  Final dPCA tensor shape: {dpca_full_tensor.shape}")
    
    # Fit dPCA model
    print("  Fitting dPCA model...")
    dpca_model = dPCA(labels='icot', regularizer='auto', n_components=8)
    
    try:
        Z = dpca_model.fit_transform(dpca_full_tensor)
        print("  dPCA fitting successful!")
        
        # Print explained variance
        print(f"  Explained variance by marginalization:")
        for margin, var_exp in dpca_model.explained_variance_ratio_.items():
            total_var = np.sum(var_exp[:3])  # First 3 components
            print(f"    {margin}: {total_var:.3f} ({100*total_var:.1f}%)")
    
    except Exception as e:
        print(f"  ERROR during dPCA fitting: {e}")
        return None
    
    return {
        'dpca_model': dpca_model,
        'transformed_data': Z,
        'original_tensor': dpca_full_tensor,
        'condition_labels': final_condition_labels,
        'unique_isis': unique_isis_final,
        'target_phases': target_phases,
        'balanced_tensor': dpca_tensor,
        'isi_indices': isi_indices,
        'choice_values': choice_values,
        'outcome_values': outcome_values,
        'explained_variance': dpca_model.explained_variance_ratio_,
        'min_trials_used': min_trials,
        'n_conditions': len(final_condition_labels)
    }

def visualize_isi_dpca_results(dpca_results: Dict[str, Any], 
                              isi_phase_data: Dict[str, Any],
                              max_components: int = 3):
    """
    Visualize dPCA results for ISI phase analysis
    """
    if dpca_results is None:
        print("No dPCA results to visualize")
        return
    
    print(f"\n=== VISUALIZING ISI dPCA RESULTS ===")
    
    Z = dpca_results['transformed_data']
    unique_isis = dpca_results['unique_isis']
    target_phases = dpca_results['target_phases']
    data_type = isi_phase_data['data_type']
    
    # Convert phase to percentage for display
    phase_percent = target_phases * 100
    
    # Available marginalizations
    available_margins = [m for m in ['i', 'c', 'o', 'ic', 'io', 'co', 'ico'] if m in Z]
    n_margins = len(available_margins)
    
    if n_margins == 0:
        print("No marginalization data available")
        return
    
    # Create subplot grid
    fig, axes = plt.subplots(n_margins, max_components, figsize=(5*max_components, 4*n_margins))
    
    if n_margins == 1:
        axes = axes.reshape(1, -1)
    elif max_components == 1:
        axes = axes.reshape(-1, 1)
    
    for margin_idx, margin in enumerate(available_margins):
        margin_data = Z[margin]
        
        for comp_idx in range(min(max_components, margin_data.shape[0])):
            ax = axes[margin_idx, comp_idx] if n_margins > 1 and max_components > 1 else \
                (axes[margin_idx] if n_margins > 1 else axes[comp_idx])
            
            if margin == 'i':  # ISI marginalization
                for isi_idx, isi_val in enumerate(unique_isis):
                    if isi_idx < margin_data.shape[1]:
                        ax.plot(phase_percent, margin_data[comp_idx, isi_idx], 
                               label=f'ISI {isi_val:.0f}ms', linewidth=2)
                ax.set_title(f'ISI Component {comp_idx+1}')
                ax.set_xlabel('ISI Phase (%)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
            elif margin == 'c':  # Choice marginalization
                choice_labels = ['Left', 'Right']
                colors = ['blue', 'red']
                for choice_idx in range(min(2, margin_data.shape[1])):
                    ax.plot(phase_percent, margin_data[comp_idx, choice_idx], 
                           label=choice_labels[choice_idx], color=colors[choice_idx], linewidth=2)
                ax.set_title(f'Choice Component {comp_idx+1}')
                ax.set_xlabel('ISI Phase (%)')
                ax.legend()
                
            elif margin == 'o':  # Outcome marginalization  
                outcome_labels = ['Not Rewarded', 'Rewarded']
                colors = ['orange', 'green']
                for outcome_idx in range(min(2, margin_data.shape[1])):
                    ax.plot(phase_percent, margin_data[comp_idx, outcome_idx], 
                           label=outcome_labels[outcome_idx], color=colors[outcome_idx], linewidth=2)
                ax.set_title(f'Outcome Component {comp_idx+1}')
                ax.set_xlabel('ISI Phase (%)')
                ax.legend()
                
            else:  # Interaction terms - just plot mean
                if len(margin_data.shape) > 2:
                    # Take mean across condition dimensions
                    mean_trace = np.mean(margin_data[comp_idx], axis=tuple(range(margin_data[comp_idx].ndim-1)))
                else:
                    mean_trace = margin_data[comp_idx]
                
                if len(mean_trace) == len(phase_percent):
                    ax.plot(phase_percent, mean_trace, 'k-', linewidth=2)
                ax.set_title(f'{margin.upper()} Interaction {comp_idx+1}')
                ax.set_xlabel('ISI Phase (%)')
            
            ax.set_ylabel(f'Component Weight\n({data_type})')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
            
            # Add phase markers
            ax.axvline(0, color='purple', linestyle='--', alpha=0.7, label='End Flash 1')
            ax.axvline(100, color='purple', linestyle='--', alpha=0.7, label='Start Flash 2')
    
    plt.suptitle(f'ISI Phase dPCA Results ({data_type})', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print variance summary
    print(f"\nVariance Explained Summary:")
    total_var = 0
    for margin, var_exp in dpca_results['explained_variance'].items():
        margin_total = np.sum(var_exp[:max_components])
        total_var += margin_total
        print(f"  {margin}: {margin_total:.3f} ({100*margin_total:.1f}%)")
    print(f"  Total (first {max_components} components): {total_var:.3f} ({100*total_var:.1f}%)")





















# Try without regularization
def run_isi_only_dpca_no_reg(isi_phase_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run dPCA focusing only on ISI effects - no regularization"""
    print(f"\n=== ISI-ONLY dPCA ANALYSIS (NO REGULARIZATION) ===")
    
    isi_array = isi_phase_data['isi_phase_data']  # (trials, rois, phase_samples)
    trial_metadata = isi_phase_data['trial_metadata']
    target_phases = isi_phase_data['target_phases']
    
    # Group trials by ISI only
    unique_isis = sorted(list(set([meta['isi_ms'] for meta in trial_metadata])))
    print(f"  Analyzing {len(unique_isis)} ISI conditions: {unique_isis}")
    
    isi_conditions = {}
    for isi in unique_isis:
        matching_trials = [i for i, meta in enumerate(trial_metadata) if meta['isi_ms'] == isi]
        isi_conditions[isi] = matching_trials
        print(f"    ISI {isi}ms: {len(matching_trials)} trials")
    
    # Balance trial counts
    min_trials = min(len(trials) for trials in isi_conditions.values())
    print(f"  Balancing to {min_trials} trials per ISI")
    
    balanced_data = []
    isi_labels = []
    
    for isi in sorted(unique_isis):
        trial_indices = isi_conditions[isi]
        selected_trials = np.random.choice(trial_indices, min_trials, replace=False)
        isi_data = isi_array[selected_trials]  # (min_trials, rois, phase_samples)
        balanced_data.append(isi_data)
        isi_labels.append(isi)
    
    # Create ISI tensor: (isis, trials, rois, phase_samples)
    isi_tensor = np.stack(balanced_data, axis=0)
    print(f"  ISI tensor shape: {isi_tensor.shape}")
    
    # Average across trials: (isis, rois, phase_samples)
    isi_avg = np.mean(isi_tensor, axis=1)
    
    # Try different dPCA settings
    try:
        # Option 1: No regularization
        dpca_model = dPCA(labels='it', regularizer=None, n_components=5)
        Z = dpca_model.fit_transform(isi_avg)
        print("  ISI-only dPCA (no reg) successful!")
        
        return {
            'method': 'no_regularization',
            'dpca_model': dpca_model,
            'transformed_data': Z,
            'isi_tensor': isi_tensor,
            'isi_avg': isi_avg,
            'isi_labels': isi_labels,
            'target_phases': target_phases,
            'explained_variance': dpca_model.explained_variance_ratio_,
            'min_trials_used': min_trials
        }
    
    except Exception as e:
        print(f"  dPCA (no reg) failed: {e}")
        
        # Option 2: Try with trial-averaged data directly
        try:
            # Provide trial-level data for regularization
            dpca_model = dPCA(labels='it', regularizer='auto', n_components=5)
            Z = dpca_model.fit_transform(isi_avg, trialX=isi_tensor)
            print("  ISI-only dPCA (auto reg with trials) successful!")
            
            return {
                'method': 'auto_regularization_with_trials',
                'dpca_model': dpca_model,
                'transformed_data': Z,
                'isi_tensor': isi_tensor,
                'isi_avg': isi_avg,
                'isi_labels': isi_labels,
                'target_phases': target_phases,
                'explained_variance': dpca_model.explained_variance_ratio_,
                'min_trials_used': min_trials
            }
        
        except Exception as e2:
            print(f"  dPCA (auto reg) also failed: {e2}")
            return None
































































# Run simplified ISI-only dPCA
def run_isi_only_dpca(isi_phase_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run dPCA focusing only on ISI effects"""
    print(f"\n=== ISI-ONLY dPCA ANALYSIS ===")
    
    isi_array = isi_phase_data['isi_phase_data']  # (trials, rois, phase_samples)
    trial_metadata = isi_phase_data['trial_metadata']
    target_phases = isi_phase_data['target_phases']
    
    # Group trials by ISI only
    unique_isis = sorted(list(set([meta['isi_ms'] for meta in trial_metadata])))
    print(f"  Analyzing {len(unique_isis)} ISI conditions: {unique_isis}")
    
    isi_conditions = {}
    for isi in unique_isis:
        matching_trials = [i for i, meta in enumerate(trial_metadata) if meta['isi_ms'] == isi]
        isi_conditions[isi] = matching_trials
        print(f"    ISI {isi}ms: {len(matching_trials)} trials")
    
    # Balance trial counts
    min_trials = min(len(trials) for trials in isi_conditions.values())
    print(f"  Balancing to {min_trials} trials per ISI")
    
    balanced_data = []
    isi_labels = []
    
    for isi in sorted(unique_isis):
        trial_indices = isi_conditions[isi]
        selected_trials = np.random.choice(trial_indices, min_trials, replace=False)
        isi_data = isi_array[selected_trials]  # (min_trials, rois, phase_samples)
        balanced_data.append(isi_data)
        isi_labels.append(isi)
    
    # Create ISI tensor: (isis, trials, rois, phase_samples)
    isi_tensor = np.stack(balanced_data, axis=0)
    print(f"  ISI tensor shape: {isi_tensor.shape}")
    
    # Average across trials: (isis, rois, phase_samples)
    isi_avg = np.mean(isi_tensor, axis=1)
    
    # Fit dPCA with ISI and time only
    dpca_model = dPCA(labels='it', regularizer=0.01, n_components=5)
    
    try:
        Z = dpca_model.fit_transform(isi_avg)
        print("  ISI-only dPCA successful!")
        
        return {
            'dpca_model': dpca_model,
            'transformed_data': Z,
            'isi_tensor': isi_tensor,
            'isi_avg': isi_avg,
            'isi_labels': isi_labels,
            'target_phases': target_phases,
            'explained_variance': dpca_model.explained_variance_ratio_,
            'min_trials_used': min_trials
        }
    
    except Exception as e:
        print(f"  dPCA failed: {e}")
        return None


# Visualize ISI-only results
def visualize_isi_only_dpca(results: Dict[str, Any], 
                           isi_phase_data: Dict[str, Any],
                           max_components: int = 3):
    """Visualize ISI-only dPCA results"""
    if results is None:
        return
    
    Z = results['transformed_data']
    isi_labels = results['isi_labels']
    target_phases = results['target_phases']
    explained_var = results['explained_variance']
    data_type = isi_phase_data['data_type']
    
    phase_percent = target_phases * 100
    
    fig, axes = plt.subplots(2, max_components, figsize=(6*max_components, 10))
    
    # Top row: ISI marginalization
    for comp_idx in range(max_components):
        ax = axes[0, comp_idx]
        
        if 'i' in Z:
            isi_data = Z['i'][comp_idx]  # (isis, phase_samples)
            
            for isi_idx, isi_val in enumerate(isi_labels):
                color = 'blue' if isi_val <= 700 else 'red'
                alpha = 0.7 + 0.3 * (isi_idx / len(isi_labels))
                
                ax.plot(phase_percent, isi_data[isi_idx], 
                       color=color, alpha=alpha, linewidth=2,
                       label=f'ISI {isi_val:.0f}ms')
            
            ax.set_title(f'ISI Component {comp_idx+1}\nVar: {explained_var["i"][comp_idx]:.3f}')
            ax.set_xlabel('ISI Phase (%)')
            ax.set_ylabel(f'Component Weight ({data_type})')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax.axvline(0, color='purple', linestyle='--', alpha=0.7)
            ax.axvline(100, color='purple', linestyle='--', alpha=0.7)
    
    # Bottom row: Short vs Long comparison
    short_isis = [isi for isi in isi_labels if isi <= 700]
    long_isis = [isi for isi in isi_labels if isi > 700]
    
    for comp_idx in range(max_components):
        ax = axes[1, comp_idx]
        
        if 'i' in Z and len(short_isis) > 0 and len(long_isis) > 0:
            isi_data = Z['i'][comp_idx]
            
            # Average short and long ISIs
            short_indices = [i for i, isi in enumerate(isi_labels) if isi <= 700]
            long_indices = [i for i, isi in enumerate(isi_labels) if isi > 700]
            
            short_mean = np.mean(isi_data[short_indices], axis=0)
            long_mean = np.mean(isi_data[long_indices], axis=0)
            
            ax.plot(phase_percent, short_mean, 'b-', linewidth=3, 
                   label=f'Short ISIs (n={len(short_isis)})')
            ax.plot(phase_percent, long_mean, 'r-', linewidth=3,
                   label=f'Long ISIs (n={len(long_isis)})')
            
            ax.set_title(f'Short vs Long ISIs - Component {comp_idx+1}')
            ax.set_xlabel('ISI Phase (%)')
            ax.set_ylabel(f'Component Weight ({data_type})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax.axvline(0, color='purple', linestyle='--', alpha=0.7)
            ax.axvline(100, color='purple', linestyle='--', alpha=0.7)
    
    plt.suptitle(f'ISI-Only dPCA Results ({data_type})', fontsize=16)
    plt.tight_layout()
    plt.show()















































def plot_isi_components_in_time_domain(dpca_results, isi_phase_data, component_idx=1):
    """Plot dPCA components back in real time domain for each ISI duration"""
    
    Z = dpca_results['transformed_data']
    isi_labels = dpca_results['isi_labels']
    target_phases = dpca_results['target_phases']
    
    if 'i' not in Z:
        return
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, isi_ms in enumerate(isi_labels):
        if i >= 10:  # Only plot first 10 ISIs
            break
            
        # Convert phase back to real time for this ISI
        isi_duration_s = isi_ms / 1000.0
        time_vector = target_phases * isi_duration_s  # Real time within ISI
        
        # Plot component pattern
        component_pattern = Z['i'][component_idx, i]  # Component pattern for this ISI
        
        axes[i].plot(time_vector * 1000, component_pattern, 'o-', linewidth=2)  # Convert to ms
        axes[i].set_title(f'ISI {isi_ms}ms\n(Duration: {isi_duration_s:.2f}s)')
        axes[i].set_xlabel('Time within ISI (ms)')
        axes[i].set_ylabel(f'Component {component_idx} Weight')
        axes[i].grid(True, alpha=0.3)
        axes[i].axvline(0, color='purple', linestyle='--', alpha=0.7, label='End Flash 1')
        axes[i].axvline(isi_ms, color='purple', linestyle='--', alpha=0.7, label='Start Flash 2')
    
    plt.suptitle(f'Component {component_idx}: Real Time Domain Patterns', fontsize=16)
    plt.tight_layout()
    plt.show()

























def run_tsne_on_isi_data(isi_phase_data: Dict[str, Any], 
                        cfg: Dict[str, Any],
                        use_full_phase: bool = True,
                        tsne_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run t-SNE analysis on ISI phase-normalized data
    
    Args:
        isi_phase_data: ISI phase data from extract_isi_phase_normalized
        cfg: Configuration
        use_full_phase: If True, use entire ISI phase; if False, use summary features
        tsne_params: t-SNE parameters (perplexity, learning_rate, etc.)
    """
    print(f"\n=== RUNNING t-SNE ON ISI DATA ===")
    
    if tsne_params is None:
        tsne_params = {
            'perplexity': 30,
            'learning_rate': 200,
            'n_iter': 1000,
            'random_state': 42
        }
    
    isi_array = isi_phase_data['isi_phase_data']  # (trials, rois, phase_samples)
    trial_metadata = isi_phase_data['trial_metadata']
    target_phases = isi_phase_data['target_phases']
    data_type = isi_phase_data['data_type']
    
    print(f"  Input shape: {isi_array.shape}")
    print(f"  Data type: {data_type}")
    print(f"  Use full phase: {use_full_phase}")
    
    # Prepare data for t-SNE
    if use_full_phase:
        # Flatten each trial: (trials, rois * phase_samples)
        n_trials, n_rois, n_phases = isi_array.shape
        tsne_input = isi_array.reshape(n_trials, n_rois * n_phases)
        feature_description = f"full phase data ({n_rois} ROIs × {n_phases} phases = {n_rois * n_phases} features)"
    else:
        # Use summary statistics as features
        tsne_input = _extract_isi_summary_features(isi_array, target_phases)
        feature_description = f"summary features ({tsne_input.shape[1]} features per trial)"
    
    print(f"  t-SNE input: {tsne_input.shape} - {feature_description}")
    
    # Remove any NaN trials
    valid_trials = ~np.any(np.isnan(tsne_input), axis=1)
    tsne_input_clean = tsne_input[valid_trials]
    metadata_clean = [trial_metadata[i] for i in range(len(trial_metadata)) if valid_trials[i]]
    
    print(f"  Valid trials: {len(metadata_clean)}/{len(trial_metadata)}")
    
    if len(metadata_clean) < 10:
        print("  ERROR: Too few valid trials for t-SNE")
        return None
    
    # Extract trial labels for coloring
    isi_values = [meta['isi_ms'] for meta in metadata_clean]
    choice_values = [meta['is_right_choice'] for meta in metadata_clean]
    outcome_values = [meta['rewarded'] for meta in metadata_clean]
    
    # Run t-SNE
    from sklearn.manifold import TSNE
    
    print(f"  Running t-SNE with parameters:")
    for key, value in tsne_params.items():
        print(f"    {key}: {value}")
    
    tsne = TSNE(**tsne_params)
    
    try:
        tsne_embedding = tsne.fit_transform(tsne_input_clean)
        print(f"  t-SNE successful! Embedding shape: {tsne_embedding.shape}")
    except Exception as e:
        print(f"  t-SNE failed: {e}")
        return None
    
    return {
        'embedding': tsne_embedding,  # (n_valid_trials, 2)
        'trial_metadata': metadata_clean,
        'isi_values': np.array(isi_values),
        'choice_values': np.array(choice_values),
        'outcome_values': np.array(outcome_values),
        'valid_trials_mask': valid_trials,
        'tsne_model': tsne,
        'tsne_params': tsne_params,
        'input_features': tsne_input_clean,
        'feature_description': feature_description,
        'data_type': data_type,
        'use_full_phase': use_full_phase
    }






def run_tsne_on_isi_data_fixed(isi_phase_data: Dict[str, Any], 
                               cfg: Dict[str, Any],
                               use_full_phase: bool = True,
                               tsne_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run t-SNE analysis on ISI phase-normalized data with fixed parameters for scikit-learn 1.7+
    """
    print(f"\n=== RUNNING t-SNE ON ISI DATA (FIXED) ===")
    
    if tsne_params is None:
        # Use correct parameter names for scikit-learn 1.7+
        tsne_params = {
            'perplexity': 30,
            'learning_rate': 200,
            'max_iter': 1000,  # Changed from 'n_iter'
            'random_state': 42
        }
    
    # Fix any old parameter names if passed in
    if 'n_iter' in tsne_params:
        tsne_params['max_iter'] = tsne_params.pop('n_iter')
        print("  Fixed parameter: n_iter -> max_iter")
    
    isi_array = isi_phase_data['isi_phase_data']  # (trials, rois, phase_samples)
    trial_metadata = isi_phase_data['trial_metadata']
    target_phases = isi_phase_data['target_phases']
    data_type = isi_phase_data['data_type']
    
    print(f"  Input shape: {isi_array.shape}")
    print(f"  Data type: {data_type}")
    print(f"  Use full phase: {use_full_phase}")
    
    # Prepare data for t-SNE
    if use_full_phase:
        # Flatten each trial: (trials, rois * phase_samples)
        n_trials, n_rois, n_phases = isi_array.shape
        tsne_input = isi_array.reshape(n_trials, n_rois * n_phases)
        feature_description = f"full phase data ({n_rois} ROIs × {n_phases} phases = {n_rois * n_phases} features)"
    else:
        # Use summary statistics as features
        tsne_input = _extract_isi_summary_features(isi_array, target_phases)
        feature_description = f"summary features ({tsne_input.shape[1]} features per trial)"
    
    print(f"  t-SNE input: {tsne_input.shape} - {feature_description}")
    
    # Remove any NaN trials
    valid_trials = ~np.any(np.isnan(tsne_input), axis=1)
    tsne_input_clean = tsne_input[valid_trials]
    metadata_clean = [trial_metadata[i] for i in range(len(trial_metadata)) if valid_trials[i]]
    
    print(f"  Valid trials: {len(metadata_clean)}/{len(trial_metadata)}")
    
    if len(metadata_clean) < 10:
        print("  ERROR: Too few valid trials for t-SNE")
        return None
    
    # Check perplexity vs number of samples
    perplexity = tsne_params.get('perplexity', 30)
    if len(metadata_clean) <= 3 * perplexity:
        new_perplexity = max(5, len(metadata_clean) // 4)
        print(f"  WARNING: Reducing perplexity from {perplexity} to {new_perplexity} (too few samples)")
        tsne_params['perplexity'] = new_perplexity
    
    # Handle high dimensionality with PCA preprocessing
    n_features = tsne_input_clean.shape[1]
    if n_features > 1000:
        print(f"  High dimensionality ({n_features}), applying PCA preprocessing...")
        from sklearn.decomposition import PCA
        
        n_components = min(50, len(metadata_clean) - 1)
        pca = PCA(n_components=n_components, random_state=42)
        tsne_input_clean = pca.fit_transform(tsne_input_clean)
        print(f"  PCA reduced to {tsne_input_clean.shape[1]} components")
        print(f"  PCA explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")
    
    # Extract trial labels for coloring
    isi_values = [meta['isi_ms'] for meta in metadata_clean]
    choice_values = [meta['is_right_choice'] for meta in metadata_clean]
    outcome_values = [meta['rewarded'] for meta in metadata_clean]
    
    # Run t-SNE
    from sklearn.manifold import TSNE
    
    print(f"  Running t-SNE with parameters:")
    for key, value in tsne_params.items():
        print(f"    {key}: {value}")
    
    try:
        tsne = TSNE(**tsne_params)
        tsne_embedding = tsne.fit_transform(tsne_input_clean)
        print(f"  t-SNE successful! Embedding shape: {tsne_embedding.shape}")
    except Exception as e:
        print(f"  t-SNE failed: {e}")
        return None
    
    return {
        'embedding': tsne_embedding,  # (n_valid_trials, 2)
        'trial_metadata': metadata_clean,
        'isi_values': np.array(isi_values),
        'choice_values': np.array(choice_values),
        'outcome_values': np.array(outcome_values),
        'valid_trials_mask': valid_trials,
        'tsne_model': tsne,
        'tsne_params': tsne_params,
        'input_features': tsne_input_clean,
        'feature_description': feature_description,
        'data_type': data_type,
        'use_full_phase': use_full_phase
    }








def _extract_isi_summary_features(isi_array: np.ndarray, target_phases: np.ndarray) -> np.ndarray:
    """Extract summary features from ISI phase data instead of using full time series"""
    
    n_trials, n_rois, n_phases = isi_array.shape
    
    # Define phase windows
    early_phase = target_phases <= 0.33   # First third
    mid_phase = (target_phases > 0.33) & (target_phases <= 0.67)  # Middle third  
    late_phase = target_phases > 0.67     # Last third
    
    features_list = []
    
    for trial_idx in range(n_trials):
        trial_data = isi_array[trial_idx]  # (rois, phases)
        
        trial_features = []
        
        # For each ROI, extract summary statistics
        for roi_idx in range(n_rois):
            roi_trace = trial_data[roi_idx]
            
            if np.all(np.isnan(roi_trace)):
                # If ROI has no data, fill with zeros
                trial_features.extend([0] * 9)  # 9 features per ROI
                continue
            
            # Overall statistics
            mean_activity = np.nanmean(roi_trace)
            std_activity = np.nanstd(roi_trace)
            
            # Phase-specific means
            early_mean = np.nanmean(roi_trace[early_phase])
            mid_mean = np.nanmean(roi_trace[mid_phase])  
            late_mean = np.nanmean(roi_trace[late_phase])
            
            # Temporal dynamics
            linear_trend = np.corrcoef(target_phases, roi_trace)[0, 1] if not np.all(np.isnan(roi_trace)) else 0
            max_activity = np.nanmax(roi_trace)
            min_activity = np.nanmin(roi_trace)
            range_activity = max_activity - min_activity
            
            trial_features.extend([
                mean_activity, std_activity,
                early_mean, mid_mean, late_mean,
                linear_trend, max_activity, min_activity, range_activity
            ])
        
        features_list.append(trial_features)
    
    return np.array(features_list)

def visualize_tsne_results(tsne_results: Dict[str, Any], 
                          max_isi_show: int = 10):
    """Visualize t-SNE results with different colorings"""
    
    if tsne_results is None:
        print("No t-SNE results to visualize")
        return
    
    embedding = tsne_results['embedding']
    isi_values = tsne_results['isi_values']
    choice_values = tsne_results['choice_values']
    outcome_values = tsne_results['outcome_values']
    data_type = tsne_results['data_type']
    feature_desc = tsne_results['feature_description']
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Color by ISI duration
    scatter1 = axes[0, 0].scatter(embedding[:, 0], embedding[:, 1], 
                                 c=isi_values, cmap='viridis', s=30, alpha=0.7)
    axes[0, 0].set_title('t-SNE: Colored by ISI Duration')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0, 0], label='ISI (ms)')
    
    # Plot 2: Color by choice
    choice_colors = ['blue' if c == 0 else 'red' for c in choice_values]
    choice_labels = ['Left' if c == 0 else 'Right' for c in choice_values]
    
    for choice, color, label in [(0, 'blue', 'Left Choice'), (1, 'red', 'Right Choice')]:
        mask = choice_values == choice
        if np.any(mask):
            axes[0, 1].scatter(embedding[mask, 0], embedding[mask, 1], 
                             c=color, s=30, alpha=0.7, label=label)
    
    axes[0, 1].set_title('t-SNE: Colored by Choice')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    axes[0, 1].legend()
    
    # Plot 3: Color by outcome
    for outcome, color, label in [(0, 'orange', 'Not Rewarded'), (1, 'green', 'Rewarded')]:
        mask = outcome_values == outcome
        if np.any(mask):
            axes[0, 2].scatter(embedding[mask, 0], embedding[mask, 1],
                             c=color, s=30, alpha=0.7, label=label)
    
    axes[0, 2].set_title('t-SNE: Colored by Outcome')
    axes[0, 2].set_xlabel('t-SNE 1')
    axes[0, 2].set_ylabel('t-SNE 2')
    axes[0, 2].legend()
    
    # Plot 4: Short vs Long ISIs
    short_mask = isi_values <= 700
    long_mask = isi_values > 700
    
    if np.any(short_mask):
        axes[1, 0].scatter(embedding[short_mask, 0], embedding[short_mask, 1],
                          c='blue', s=30, alpha=0.7, label=f'Short ISIs (≤700ms, n={np.sum(short_mask)})')
    if np.any(long_mask):
        axes[1, 0].scatter(embedding[long_mask, 0], embedding[long_mask, 1],
                          c='red', s=30, alpha=0.7, label=f'Long ISIs (>700ms, n={np.sum(long_mask)})')
    
    axes[1, 0].set_title('t-SNE: Short vs Long ISIs')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    axes[1, 0].legend()
    
    # Plot 5: Individual ISI values (if not too many)
    unique_isis = np.unique(isi_values)
    if len(unique_isis) <= max_isi_show:
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_isis)))
        
        for i, isi in enumerate(unique_isis):
            mask = isi_values == isi
            axes[1, 1].scatter(embedding[mask, 0], embedding[mask, 1],
                             c=[colors[i]], s=30, alpha=0.7, 
                             label=f'ISI {isi:.0f}ms (n={np.sum(mask)})')
        
        axes[1, 1].set_title('t-SNE: Individual ISI Conditions')
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[1, 1].text(0.5, 0.5, f'Too many ISI conditions to plot\n({len(unique_isis)} unique ISIs)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # Plot 6: Choice × Outcome interaction
    for choice in [0, 1]:
        for outcome in [0, 1]:
            mask = (choice_values == choice) & (outcome_values == outcome)
            if np.any(mask):
                choice_str = 'Left' if choice == 0 else 'Right'
                outcome_str = 'Rewarded' if outcome == 1 else 'Not Rewarded'
                color = 'green' if outcome == 1 else 'red'
                marker = 'o' if choice == 0 else 's'
                
                axes[1, 2].scatter(embedding[mask, 0], embedding[mask, 1],
                                 c=color, marker=marker, s=30, alpha=0.7,
                                 label=f'{choice_str} {outcome_str} (n={np.sum(mask)})')
    
    axes[1, 2].set_title('t-SNE: Choice × Outcome')
    axes[1, 2].set_xlabel('t-SNE 1')
    axes[1, 2].set_ylabel('t-SNE 2')
    axes[1, 2].legend()
    
    plt.suptitle(f't-SNE Analysis: {data_type}\n{feature_desc}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== t-SNE RESULTS SUMMARY ===")
    print(f"Data type: {data_type}")
    print(f"Features: {feature_desc}")
    print(f"Embedding range:")
    print(f"  t-SNE 1: {np.min(embedding[:, 0]):.2f} to {np.max(embedding[:, 0]):.2f}")
    print(f"  t-SNE 2: {np.min(embedding[:, 1]):.2f} to {np.max(embedding[:, 1]):.2f}")
    print(f"ISI distribution:")
    for isi in unique_isis:
        count = np.sum(isi_values == isi)
        print(f"  ISI {isi:.0f}ms: {count} trials ({100*count/len(isi_values):.1f}%)")

def compare_tsne_parameters(isi_phase_data: Dict[str, Any],
                           perplexity_values: List[float] = [5, 15, 30, 50],
                           use_full_phase: bool = True) -> Dict[str, Any]:
    """Compare t-SNE results with different perplexity values"""
    
    print(f"\n=== COMPARING t-SNE PARAMETERS ===")
    
    results = {}
    
    fig, axes = plt.subplots(2, len(perplexity_values), figsize=(5*len(perplexity_values), 10))
    
    for i, perplexity in enumerate(perplexity_values):
        print(f"\n--- Testing perplexity = {perplexity} ---")
        
        tsne_params = {
            'perplexity': perplexity,
            'learning_rate': 200,
            'n_iter': 1000,
            'random_state': 42
        }
        
        tsne_result = run_tsne_on_isi_data(isi_phase_data, {}, 
                                          use_full_phase=use_full_phase,
                                          tsne_params=tsne_params)
        
        if tsne_result is not None:
            results[perplexity] = tsne_result
            
            embedding = tsne_result['embedding']
            isi_values = tsne_result['isi_values']
            choice_values = tsne_result['choice_values']
            
            # Top row: Color by ISI
            scatter1 = axes[0, i].scatter(embedding[:, 0], embedding[:, 1],
                                        c=isi_values, cmap='viridis', s=20, alpha=0.7)
            axes[0, i].set_title(f'Perplexity = {perplexity}\n(ISI Duration)')
            axes[0, i].set_xlabel('t-SNE 1')
            axes[0, i].set_ylabel('t-SNE 2')
            
            # Bottom row: Color by choice
            for choice, color, label in [(0, 'blue', 'Left'), (1, 'red', 'Right')]:
                mask = choice_values == choice
                if np.any(mask):
                    axes[1, i].scatter(embedding[mask, 0], embedding[mask, 1],
                                     c=color, s=20, alpha=0.7, label=label)
            
            axes[1, i].set_title(f'Perplexity = {perplexity}\n(Choice)')
            axes[1, i].set_xlabel('t-SNE 1')
            axes[1, i].set_ylabel('t-SNE 2')
            axes[1, i].legend()
        else:
            axes[0, i].text(0.5, 0.5, 'Failed', ha='center', va='center', 
                           transform=axes[0, i].transAxes)
            axes[1, i].text(0.5, 0.5, 'Failed', ha='center', va='center',
                           transform=axes[1, i].transAxes)
    
    plt.tight_layout()
    plt.show()
    
    return results






































































































def run_tsne_on_isi_responsive_rois(isi_phase_data: Dict[str, Any],
                                   roi_selection_method: str = 'isi_responsive',
                                   cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run t-SNE only on ROIs that show ISI-dependent responses"""
    
    print(f"\n=== t-SNE ON ISI-RESPONSIVE ROIs ===")
    
    isi_array = isi_phase_data['isi_phase_data']  # (trials, rois, phase_samples)
    trial_metadata = isi_phase_data['trial_metadata']
    
    # Method 1: Find ROIs with ISI-dependent variance
    if roi_selection_method == 'isi_responsive':
        isi_responsive_rois = find_isi_responsive_rois(isi_array, trial_metadata, cfg['variance_threshold'])
    
    # Method 2: Use your temporal pattern ROIs
    elif roi_selection_method == 'temporal_patterns':
        # Use ROIs from your pattern discovery analysis
        if 'pattern_results' in cfg:
            isi_responsive_rois = get_rois_from_patterns(cfg['pattern_results'])
        else:
            print("No pattern results available, falling back to variance method")
            isi_responsive_rois = find_isi_responsive_rois(isi_array, trial_metadata)
    
    print(f"  Selected {len(isi_responsive_rois)} ISI-responsive ROIs out of {isi_array.shape[1]}")
    
    # Subset data to only ISI-responsive ROIs
    isi_array_subset = isi_array[:, isi_responsive_rois, :]
    
    # Run t-SNE on subset
    return run_tsne_on_isi_data_fixed(
        {'isi_phase_data': isi_array_subset, 
         'trial_metadata': trial_metadata,
         'target_phases': isi_phase_data['target_phases'],
         'data_type': f"{isi_phase_data['data_type']} (ISI-responsive ROIs only)"},
        cfg, use_full_phase=True
    )

def find_isi_responsive_rois(isi_array: np.ndarray, 
                            trial_metadata: List[Dict],
                            variance_threshold: float = 0.5) -> np.ndarray:
    """Find ROIs that show significant variance across ISI conditions"""
    
    # Group trials by ISI
    unique_isis = sorted(list(set([meta['isi_ms'] for meta in trial_metadata])))
    
    n_trials, n_rois, n_phases = isi_array.shape
    isi_responsive_rois = []
    
    for roi_idx in range(n_rois):
        roi_data = isi_array[:, roi_idx, :]  # (trials, phases)
        
        # Calculate mean response for each ISI condition
        isi_means = []
        for isi in unique_isis:
            isi_trials = [i for i, meta in enumerate(trial_metadata) if meta['isi_ms'] == isi]
            if len(isi_trials) > 0:
                isi_mean = np.mean(roi_data[isi_trials], axis=0)  # Mean across trials
                isi_means.append(np.mean(isi_mean))  # Mean across time
        
        # Calculate variance across ISI conditions
        if len(isi_means) > 1:
            isi_variance = np.var(isi_means)
            if isi_variance > variance_threshold:
                isi_responsive_rois.append(roi_idx)
    
    return np.array(isi_responsive_rois)

def run_tsne_choice_matched(isi_phase_data: Dict[str, Any],
                           cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run t-SNE with choice/outcome matched across ISIs"""
    
    print(f"\n=== t-SNE WITH CHOICE/OUTCOME MATCHING ===")
    
    isi_array = isi_phase_data['isi_phase_data']
    trial_metadata = isi_phase_data['trial_metadata']
    
    # Find trials that break the ISI→choice determinism
    # Look for short ISIs with right choices, long ISIs with left choices
    
    alternative_trials = []
    for i, meta in enumerate(trial_metadata):
        isi = meta['isi_ms']
        choice = meta['is_right_choice']
        
        # Keep trials that break the expected pattern
        if (isi <= 700 and choice == 1) or (isi > 700 and choice == 0):
            alternative_trials.append(i)
    
    print(f"  Found {len(alternative_trials)} trials with alternative choice patterns")
    
    if len(alternative_trials) > 20:  # Need sufficient trials
        # Run t-SNE on these trials only
        subset_array = isi_array[alternative_trials]
        subset_metadata = [trial_metadata[i] for i in alternative_trials]
        
        return run_tsne_on_isi_data_fixed({
            'isi_phase_data': subset_array,
            'trial_metadata': subset_metadata, 
            'target_phases': isi_phase_data['target_phases'],
            'data_type': f"{isi_phase_data['data_type']} (choice-matched)"
        }, cfg, use_full_phase=True)
    else:
        print("  Insufficient alternative trials for analysis")
        return None
    
    
    
    
    
def extract_roi_specific_isi_features(isi_array: np.ndarray,
                                        trial_metadata: List[Dict],
                                        roi_subset: np.ndarray = None) -> np.ndarray:
    """Extract ISI-specific features for individual ROIs rather than population"""
    
    n_trials, n_rois, n_phases = isi_array.shape
    
    if roi_subset is not None:
        roi_indices = roi_subset
        isi_array = isi_array[:, roi_subset, :]
        n_rois = len(roi_subset)
    else:
        roi_indices = np.arange(n_rois)
    
    # For each trial, create features that capture ROI-specific ISI responses
    trial_features = []
    
    for trial_idx in range(n_trials):
        trial_data = isi_array[trial_idx]  # (rois, phases)
        isi_duration = trial_metadata[trial_idx]['isi_ms']
        
        roi_features = []
        for roi_idx in range(n_rois):
            roi_trace = trial_data[roi_idx]
            
            # ISI-specific features for this ROI
            features = [
                np.mean(roi_trace),  # Overall activity
                np.std(roi_trace),   # Variability
                np.max(roi_trace) - np.min(roi_trace),  # Dynamic range
                
                # Early vs late ISI activity
                np.mean(roi_trace[:n_phases//3]),     # Early third
                np.mean(roi_trace[n_phases//3:2*n_phases//3]),  # Middle third  
                np.mean(roi_trace[2*n_phases//3:]),   # Late third
                
                # ISI duration context (normalized)
                isi_duration / 2000.0,  # Normalize to 0-1 range
            ]
            
            roi_features.extend(features)
        
        trial_features.append(roi_features)
    
    return np.array(trial_features)    




























































# Try UMAP analysis on ISI data
try:
    import umap
    print("✓ UMAP available")
    
    def run_umap_on_isi_data(isi_phase_data: Dict[str, Any], 
                            n_neighbors: int = 15, 
                            min_dist: float = 0.1,
                            n_components: int = 2,
                            use_full_phase: bool = True,
                            random_state: int = 42) -> Dict[str, Any]:
        """Run UMAP analysis on ISI phase-normalized data"""
        print(f"\n=== RUNNING UMAP ON ISI DATA ===")
        
        isi_array = isi_phase_data['isi_phase_data']  # (trials, rois, phase_samples)
        trial_metadata = isi_phase_data['trial_metadata']
        target_phases = isi_phase_data['target_phases']
        data_type = isi_phase_data['data_type']
        
        print(f"  Input shape: {isi_array.shape}")
        print(f"  Data type: {data_type}")
        print(f"  Use full phase: {use_full_phase}")
        
        # Prepare data for UMAP
        if use_full_phase:
            # Flatten each trial: (trials, rois * phase_samples)
            n_trials, n_rois, n_phases = isi_array.shape
            umap_input = isi_array.reshape(n_trials, n_rois * n_phases)
            feature_description = f"full phase data ({n_rois} ROIs × {n_phases} phases = {n_rois * n_phases} features)"
        else:
            # Use summary statistics as features
            umap_input = _extract_isi_summary_features(isi_array, target_phases)
            feature_description = f"summary features ({umap_input.shape[1]} features per trial)"
        
        print(f"  UMAP input: {umap_input.shape} - {feature_description}")
        
        # Remove any NaN trials
        valid_trials = ~np.any(np.isnan(umap_input), axis=1)
        umap_input_clean = umap_input[valid_trials]
        metadata_clean = [trial_metadata[i] for i in range(len(trial_metadata)) if valid_trials[i]]
        
        print(f"  Valid trials: {len(metadata_clean)}/{len(trial_metadata)}")
        
        if len(metadata_clean) < 10:
            print("  ERROR: Too few valid trials for UMAP")
            return None
        
        # Handle high dimensionality with PCA preprocessing if needed
        n_features = umap_input_clean.shape[1]
        if n_features > 1000:
            print(f"  High dimensionality ({n_features}), applying PCA preprocessing...")
            from sklearn.decomposition import PCA
            
            n_components_pca = min(100, len(metadata_clean) - 1)
            pca = PCA(n_components=n_components_pca, random_state=random_state)
            umap_input_clean = pca.fit_transform(umap_input_clean)
            print(f"  PCA reduced to {umap_input_clean.shape[1]} components")
            print(f"  PCA explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")
        
        # Extract trial labels for coloring
        isi_values = [meta['isi_ms'] for meta in metadata_clean]
        choice_values = [meta['is_right_choice'] for meta in metadata_clean]
        outcome_values = [meta['rewarded'] for meta in metadata_clean]
        
        # Run UMAP
        print(f"  Running UMAP with parameters:")
        print(f"    n_neighbors: {n_neighbors}")
        print(f"    min_dist: {min_dist}")
        print(f"    n_components: {n_components}")
        print(f"    random_state: {random_state}")
        
        try:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors, 
                min_dist=min_dist, 
                n_components=n_components,
                random_state=random_state,
                verbose=True
            )
            umap_embedding = reducer.fit_transform(umap_input_clean)
            print(f"  UMAP successful! Embedding shape: {umap_embedding.shape}")
        except Exception as e:
            print(f"  UMAP failed: {e}")
            return None
        
        return {
            'embedding': umap_embedding,  # (n_valid_trials, n_components)
            'trial_metadata': metadata_clean,
            'isi_values': np.array(isi_values),
            'choice_values': np.array(choice_values),
            'outcome_values': np.array(outcome_values),
            'valid_trials_mask': valid_trials,
            'umap_model': reducer,
            'input_features': umap_input_clean,
            'feature_description': feature_description,
            'data_type': data_type,
            'use_full_phase': use_full_phase,
            'method': 'UMAP'
        }
    
    def visualize_umap_results(umap_results: Dict[str, Any], 
                              max_isi_show: int = 10):
        """Visualize UMAP results with different colorings"""
        
        if umap_results is None:
            print("No UMAP results to visualize")
            return
        
        embedding = umap_results['embedding']
        isi_values = umap_results['isi_values']
        choice_values = umap_results['choice_values']
        outcome_values = umap_results['outcome_values']
        data_type = umap_results['data_type']
        feature_desc = umap_results['feature_description']
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Color by ISI duration
        scatter1 = axes[0, 0].scatter(embedding[:, 0], embedding[:, 1], 
                                     c=isi_values, cmap='viridis', s=30, alpha=0.7)
        axes[0, 0].set_title('UMAP: Colored by ISI Duration')
        axes[0, 0].set_xlabel('UMAP 1')
        axes[0, 0].set_ylabel('UMAP 2')
        plt.colorbar(scatter1, ax=axes[0, 0], label='ISI (ms)')
        
        # Plot 2: Color by choice
        for choice, color, label in [(0, 'blue', 'Left Choice'), (1, 'red', 'Right Choice')]:
            mask = choice_values == choice
            if np.any(mask):
                axes[0, 1].scatter(embedding[mask, 0], embedding[mask, 1], 
                                 c=color, s=30, alpha=0.7, label=label)
        
        axes[0, 1].set_title('UMAP: Colored by Choice')
        axes[0, 1].set_xlabel('UMAP 1')
        axes[0, 1].set_ylabel('UMAP 2')
        axes[0, 1].legend()
        
        # Plot 3: Color by outcome
        for outcome, color, label in [(0, 'orange', 'Not Rewarded'), (1, 'green', 'Rewarded')]:
            mask = outcome_values == outcome
            if np.any(mask):
                axes[0, 2].scatter(embedding[mask, 0], embedding[mask, 1],
                                 c=color, s=30, alpha=0.7, label=label)
        
        axes[0, 2].set_title('UMAP: Colored by Outcome')
        axes[0, 2].set_xlabel('UMAP 1')
        axes[0, 2].set_ylabel('UMAP 2')
        axes[0, 2].legend()
        
        # Plot 4: Short vs Long ISIs
        short_mask = isi_values <= 700
        long_mask = isi_values > 700
        
        if np.any(short_mask):
            axes[1, 0].scatter(embedding[short_mask, 0], embedding[short_mask, 1],
                              c='blue', s=30, alpha=0.7, label=f'Short ISIs (≤700ms, n={np.sum(short_mask)})')
        if np.any(long_mask):
            axes[1, 0].scatter(embedding[long_mask, 0], embedding[long_mask, 1],
                              c='red', s=30, alpha=0.7, label=f'Long ISIs (>700ms, n={np.sum(long_mask)})')
        
        axes[1, 0].set_title('UMAP: Short vs Long ISIs')
        axes[1, 0].set_xlabel('UMAP 1')
        axes[1, 0].set_ylabel('UMAP 2')
        axes[1, 0].legend()
        
        # Plot 5: Individual ISI values (if not too many)
        unique_isis = np.unique(isi_values)
        if len(unique_isis) <= max_isi_show:
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_isis)))
            
            for i, isi in enumerate(unique_isis):
                mask = isi_values == isi
                axes[1, 1].scatter(embedding[mask, 0], embedding[mask, 1],
                                 c=[colors[i]], s=30, alpha=0.7, 
                                 label=f'ISI {isi:.0f}ms (n={np.sum(mask)})')
            
            axes[1, 1].set_title('UMAP: Individual ISI Conditions')
            axes[1, 1].set_xlabel('UMAP 1')
            axes[1, 1].set_ylabel('UMAP 2')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes[1, 1].text(0.5, 0.5, f'Too many ISI conditions to plot\n({len(unique_isis)} unique ISIs)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Plot 6: Choice × Outcome interaction
        for choice in [0, 1]:
            for outcome in [0, 1]:
                mask = (choice_values == choice) & (outcome_values == outcome)
                if np.any(mask):
                    choice_str = 'Left' if choice == 0 else 'Right'
                    outcome_str = 'Rewarded' if outcome == 1 else 'Not Rewarded'
                    color = 'green' if outcome == 1 else 'red'
                    marker = 'o' if choice == 0 else 's'
                    
                    axes[1, 2].scatter(embedding[mask, 0], embedding[mask, 1],
                                     c=color, marker=marker, s=30, alpha=0.7,
                                     label=f'{choice_str} {outcome_str} (n={np.sum(mask)})')
        
        axes[1, 2].set_title('UMAP: Choice × Outcome')
        axes[1, 2].set_xlabel('UMAP 1')
        axes[1, 2].set_ylabel('UMAP 2')
        axes[1, 2].legend()
        
        plt.suptitle(f'UMAP Analysis: {data_type}\n{feature_desc}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n=== UMAP RESULTS SUMMARY ===")
        print(f"Data type: {data_type}")
        print(f"Features: {feature_desc}")
        print(f"Embedding range:")
        print(f"  UMAP 1: {np.min(embedding[:, 0]):.2f} to {np.max(embedding[:, 0]):.2f}")
        print(f"  UMAP 2: {np.min(embedding[:, 1]):.2f} to {np.max(embedding[:, 1]):.2f}")
        print(f"ISI distribution:")
        for isi in unique_isis:
            count = np.sum(isi_values == isi)
            print(f"  ISI {isi:.0f}ms: {count} trials ({100*count/len(isi_values):.1f}%)")

    print("✓ UMAP functions defined successfully")
    
except ImportError:
    print("✗ UMAP not available. Install with: pip install umap-learn")

















































# Try UMAP on ISI-responsive ROIs only
def run_umap_on_isi_responsive_rois(isi_phase_data: Dict[str, Any],
                                   isi_responsive_rois: np.ndarray,
                                   n_neighbors: int = 15,
                                   min_dist: float = 0.1) -> Dict[str, Any]:
    """Run UMAP only on ISI-responsive ROIs"""
    
    isi_array_subset = isi_phase_data['isi_phase_data'][:, isi_responsive_rois, :]
    
    umap_data_subset = {
        'isi_phase_data': isi_array_subset,
        'trial_metadata': isi_phase_data['trial_metadata'],
        'target_phases': isi_phase_data['target_phases'],
        'data_type': f"{isi_phase_data['data_type']} (ISI-responsive ROIs, n={len(isi_responsive_rois)})"
    }
    
    return run_umap_on_isi_data(
        umap_data_subset,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        use_full_phase=True
    )







































































def analyze_isi_consistency_per_roi(isi_phase_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze how consistently each ROI encodes specific ISI durations
    Look for ROIs that reliably encode 200ms as 200ms, 1700ms as 1700ms, etc.
    """
    
    isi_array = isi_phase_data['isi_phase_data']  # (trials, rois, phases)
    trial_metadata = isi_phase_data['trial_metadata']
    
    # Group trials by exact ISI duration
    isi_groups = {}
    for trial_idx, meta in enumerate(trial_metadata):
        isi_ms = meta['isi_ms']
        if isi_ms not in isi_groups:
            isi_groups[isi_ms] = []
        isi_groups[isi_ms].append(trial_idx)
    
    print(f"Found {len(isi_groups)} unique ISI durations:")
    for isi_ms, trial_indices in isi_groups.items():
        print(f"  {isi_ms}ms: {len(trial_indices)} trials")
    
    # Calculate consistency metrics for each ROI
    roi_consistency_metrics = []
    
    for roi_idx in range(isi_array.shape[1]):
        roi_metrics = {
            'roi_idx': roi_idx,
            'isi_responses': {},
            'consistency_score': 0,
            'reliability_score': 0
        }
        
        # Calculate mean response for each ISI duration
        for isi_ms, trial_indices in isi_groups.items():
            if len(trial_indices) >= 3:  # Need minimum trials
                roi_responses = isi_array[trial_indices, roi_idx, :]  # (trials, phases)
                mean_response = np.mean(roi_responses, axis=0)
                std_response = np.std(roi_responses, axis=0)
                
                roi_metrics['isi_responses'][isi_ms] = {
                    'mean_response': mean_response,
                    'std_response': std_response,
                    'reliability': 1 / (1 + np.mean(std_response))  # Higher when less variable
                }
        
        # Calculate how consistently this ROI differentiates between ISI durations
        if len(roi_metrics['isi_responses']) >= 2:
            # Compare all pairs of ISI responses
            isi_list = list(roi_metrics['isi_responses'].keys())
            pairwise_distances = []
            
            for i in range(len(isi_list)):
                for j in range(i+1, len(isi_list)):
                    isi1, isi2 = isi_list[i], isi_list[j]
                    resp1 = roi_metrics['isi_responses'][isi1]['mean_response']
                    resp2 = roi_metrics['isi_responses'][isi2]['mean_response']
                    
                    # Calculate Euclidean distance between responses
                    distance = np.linalg.norm(resp1 - resp2)
                    pairwise_distances.append(distance)
            
            roi_metrics['consistency_score'] = np.mean(pairwise_distances)
            roi_metrics['reliability_score'] = np.mean([
                resp_data['reliability'] 
                for resp_data in roi_metrics['isi_responses'].values()
            ])
        
        roi_consistency_metrics.append(roi_metrics)
    
    return {
        'roi_metrics': roi_consistency_metrics,
        'isi_groups': isi_groups,
        'analysis_type': 'isi_consistency_per_roi'
    }

def find_reliable_isi_encoders(consistency_results: Dict[str, Any], 
                              consistency_threshold: float = 1.0,
                              reliability_threshold: float = 0.7) -> Dict[str, Any]:
    """Find ROIs that reliably and consistently encode ISI information"""
    
    roi_metrics = consistency_results['roi_metrics']
    
    reliable_rois = []
    for metrics in roi_metrics:
        if (metrics['consistency_score'] > consistency_threshold and 
            metrics['reliability_score'] > reliability_threshold and
            len(metrics['isi_responses']) >= 3):  # Must respond to at least 3 ISIs
            reliable_rois.append(metrics)
    
    print(f"Found {len(reliable_rois)} reliable ISI encoders")
    print(f"Criteria: consistency > {consistency_threshold}, reliability > {reliability_threshold}")
    
    return {
        'reliable_rois': reliable_rois,
        'n_reliable': len(reliable_rois),
        'thresholds': {
            'consistency': consistency_threshold,
            'reliability': reliability_threshold
        }
    }

def find_differential_pairs(consistency_results: Dict[str, Any],
                           spatial_threshold: float = 50.0) -> Dict[str, Any]:
    """
    Find spatially proximal ROI pairs that show complementary ISI encoding
    (the pos/neg shielding pairs you described)
    """
    # This would need ROI spatial coordinates
    # For now, let's identify potential pairs based on response patterns
    
    roi_metrics = consistency_results['roi_metrics']
    
    # Calculate correlation matrix between ROI ISI responses
    reliable_rois = [m for m in roi_metrics if len(m['isi_responses']) >= 3]
    
    if len(reliable_rois) < 2:
        return {'pairs': [], 'message': 'Not enough reliable ROIs for pair analysis'}
    
    correlation_matrix = np.zeros((len(reliable_rois), len(reliable_rois)))
    
    for i, roi1 in enumerate(reliable_rois):
        for j, roi2 in enumerate(reliable_rois):
            if i != j:
                # Find common ISIs
                common_isis = set(roi1['isi_responses'].keys()) & set(roi2['isi_responses'].keys())
                
                if len(common_isis) >= 2:
                    roi1_responses = []
                    roi2_responses = []
                    
                    for isi in common_isis:
                        roi1_responses.append(np.mean(roi1['isi_responses'][isi]['mean_response']))
                        roi2_responses.append(np.mean(roi2['isi_responses'][isi]['mean_response']))
                    
                    correlation = np.corrcoef(roi1_responses, roi2_responses)[0, 1]
                    correlation_matrix[i, j] = correlation
    
    # Find strongly anti-correlated pairs (potential differential encoders)
    differential_pairs = []
    for i in range(len(reliable_rois)):
        for j in range(i+1, len(reliable_rois)):
            correlation = correlation_matrix[i, j]
            if correlation < -0.5:  # Strong anti-correlation
                differential_pairs.append({
                    'roi1_idx': reliable_rois[i]['roi_idx'],
                    'roi2_idx': reliable_rois[j]['roi_idx'],
                    'correlation': correlation,
                    'pair_type': 'differential'
                })
    
    print(f"Found {len(differential_pairs)} potential differential encoding pairs")
    
    return {
        'pairs': differential_pairs,
        'correlation_matrix': correlation_matrix,
        'reliable_rois': reliable_rois
    }







































def classify_temporal_pattern(response: np.ndarray) -> str:
    """Classify the temporal pattern of a response"""
    
    # Normalize response
    if np.std(response) == 0:
        return 'flat'
    
    response_norm = (response - np.mean(response)) / np.std(response)
    n_samples = len(response)  # Use actual length instead of assuming 100
    
    # Define pattern templates with correct length
    early_peak = np.concatenate([np.ones(n_samples//3), np.zeros(n_samples - n_samples//3)])
    mid_peak = np.concatenate([np.zeros(n_samples//3), np.ones(n_samples//3), np.zeros(n_samples - 2*(n_samples//3))])
    late_peak = np.concatenate([np.zeros(2*(n_samples//3)), np.ones(n_samples - 2*(n_samples//3))])
    sustained = np.ones(n_samples)
    ramp_up = np.linspace(0, 1, n_samples)
    ramp_down = np.linspace(1, 0, n_samples)
    
    # Calculate correlations with error handling
    patterns = {}
    
    try:
        patterns['early_peak'] = np.corrcoef(response_norm, early_peak)[0,1]
    except:
        patterns['early_peak'] = 0
        
    try:
        patterns['mid_peak'] = np.corrcoef(response_norm, mid_peak)[0,1]
    except:
        patterns['mid_peak'] = 0
        
    try:
        patterns['late_peak'] = np.corrcoef(response_norm, late_peak)[0,1]
    except:
        patterns['late_peak'] = 0
        
    try:
        patterns['sustained'] = np.corrcoef(response_norm, sustained)[0,1]
    except:
        patterns['sustained'] = 0
        
    try:
        patterns['ramp_up'] = np.corrcoef(response_norm, ramp_up)[0,1]
    except:
        patterns['ramp_up'] = 0
        
    try:
        patterns['ramp_down'] = np.corrcoef(response_norm, ramp_down)[0,1]
    except:
        patterns['ramp_down'] = 0
    
    # Find best match, handling NaN values
    valid_patterns = {k: v for k, v in patterns.items() if not np.isnan(v)}
    
    if not valid_patterns:
        return 'unknown'
    
    best_pattern = max(valid_patterns.items(), key=lambda x: abs(x[1]))[0]
    return best_pattern




def analyze_roi_isi_specialization(encoding_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze whether ROIs specialize for specific ISIs or are generalists
    """
    print(f"\n=== ANALYZING ROI ISI SPECIALIZATION ===")
    
    roi_encoding_map = encoding_map['roi_encoding_map']
    analyzed_isis = encoding_map['analyzed_isis']
    
    specialization_analysis = {}
    
    for roi_idx, roi_data in roi_encoding_map.items():
        patterns = roi_data['patterns']
        
        # Calculate response strength for each ISI
        isi_strengths = {}
        for isi_ms, pattern in patterns.items():
            isi_strengths[isi_ms] = pattern['peak_amplitude']
        
        # Determine specialization
        if len(isi_strengths) == 0:
            continue
            
        max_isi = max(isi_strengths.items(), key=lambda x: x[1])
        min_isi = min(isi_strengths.items(), key=lambda x: x[1])
        
        specialization_ratio = max_isi[1] / (min_isi[1] + 1e-6)  # Avoid division by zero
        
        # Classify ROI type
        if specialization_ratio > 3.0:
            roi_type = 'specialist'
            primary_isi = max_isi[0]
        elif specialization_ratio > 1.5:
            roi_type = 'moderate_specialist'
            primary_isi = max_isi[0]
        else:
            roi_type = 'generalist'
            primary_isi = None
        
        specialization_analysis[roi_idx] = {
            'roi_type': roi_type,
            'primary_isi': primary_isi,
            'specialization_ratio': specialization_ratio,
            'isi_strengths': isi_strengths,
            'preferred_pattern': patterns[max_isi[0]]['pattern_type'] if max_isi[0] in patterns else None
        }
    
    # Summary statistics
    type_counts = {}
    for analysis in specialization_analysis.values():
        roi_type = analysis['roi_type']
        type_counts[roi_type] = type_counts.get(roi_type, 0) + 1
    
    print(f"ROI Specialization Summary:")
    for roi_type, count in type_counts.items():
        print(f"  {roi_type}: {count} ROIs")
    
    return {
        'specialization_analysis': specialization_analysis,
        'type_counts': type_counts,
        'summary': type_counts
    }

def visualize_roi_isi_encoding_matrix(encoding_map: Dict[str, Any],
                                     specialization_analysis: Dict[str, Any],
                                     max_rois_show: int = 20):
    """
    Create a matrix visualization showing how each ROI responds to each ISI
    """
    
    roi_encoding_map = encoding_map['roi_encoding_map']
    analyzed_isis = sorted(encoding_map['analyzed_isis'])
    specialization_data = specialization_analysis['specialization_analysis']
    
    # Limit to most interesting ROIs
    roi_indices = list(roi_encoding_map.keys())[:max_rois_show]
    
    # Create response matrix
    response_matrix = np.zeros((len(roi_indices), len(analyzed_isis)))
    pattern_matrix = np.empty((len(roi_indices), len(analyzed_isis)), dtype=object)
    
    for i, roi_idx in enumerate(roi_indices):
        roi_data = roi_encoding_map[roi_idx]
        for j, isi_ms in enumerate(analyzed_isis):
            if isi_ms in roi_data['patterns']:
                response_matrix[i, j] = roi_data['patterns'][isi_ms]['peak_amplitude']
                pattern_matrix[i, j] = roi_data['patterns'][isi_ms]['pattern_type']
            else:
                response_matrix[i, j] = 0
                pattern_matrix[i, j] = 'none'
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Left: Response amplitude heatmap
    im1 = axes[0].imshow(response_matrix, aspect='auto', cmap='RdBu_r', 
                        extent=[0, len(analyzed_isis), len(roi_indices), 0])
    axes[0].set_title('ROI Response Amplitudes by ISI')
    axes[0].set_xlabel('ISI Condition')
    axes[0].set_ylabel('ROI Index')
    axes[0].set_xticks(range(len(analyzed_isis)))
    axes[0].set_xticklabels([f'{isi:.0f}ms' for isi in analyzed_isis], rotation=45)
    axes[0].set_yticks(range(len(roi_indices)))
    axes[0].set_yticklabels([f'ROI {roi}' for roi in roi_indices])
    plt.colorbar(im1, ax=axes[0], label='Peak Amplitude')
    
    # Middle: ROI specialization
    roi_types = [specialization_data.get(roi_idx, {}).get('roi_type', 'unknown') 
                for roi_idx in roi_indices]
    type_colors = {'specialist': 'red', 'moderate_specialist': 'orange', 
                  'generalist': 'blue', 'unknown': 'gray'}
    
    for i, (roi_idx, roi_type) in enumerate(zip(roi_indices, roi_types)):
        color = type_colors.get(roi_type, 'gray')
        axes[1].barh(i, 1, color=color, alpha=0.7)
        
        # Add primary ISI if specialist
        if roi_type in ['specialist', 'moderate_specialist']:
            primary_isi = specialization_data.get(roi_idx, {}).get('primary_isi')
            if primary_isi:
                axes[1].text(0.5, i, f'{primary_isi:.0f}ms', 
                           ha='center', va='center', fontweight='bold')
    
    axes[1].set_title('ROI Specialization Types')
    axes[1].set_xlabel('Specialization')
    axes[1].set_ylabel('ROI Index')
    axes[1].set_yticks(range(len(roi_indices)))
    axes[1].set_yticklabels([f'ROI {roi}' for roi in roi_indices])
    
    # Create legend
    for roi_type, color in type_colors.items():
        if roi_type != 'unknown':
            axes[1].barh(-1, 0, color=color, alpha=0.7, label=roi_type)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Right: Short vs Long ISI preference
    short_response = np.mean([response_matrix[:, j] for j, isi in enumerate(analyzed_isis) 
                             if isi <= 700], axis=0) if any(isi <= 700 for isi in analyzed_isis) else np.zeros(len(roi_indices))
    long_response = np.mean([response_matrix[:, j] for j, isi in enumerate(analyzed_isis) 
                            if isi > 700], axis=0) if any(isi > 700 for isi in analyzed_isis) else np.zeros(len(roi_indices))
    
    preference_ratio = (long_response - short_response) / (long_response + short_response + 1e-6)
    
    colors = ['blue' if ratio < -0.2 else 'red' if ratio > 0.2 else 'gray' for ratio in preference_ratio]
    
    axes[2].barh(range(len(roi_indices)), preference_ratio, color=colors, alpha=0.7)
    axes[2].set_title('Short vs Long ISI Preference')
    axes[2].set_xlabel('Preference Ratio\n(Long-Short)/(Long+Short)')
    axes[2].set_ylabel('ROI Index')
    axes[2].set_yticks(range(len(roi_indices)))
    axes[2].set_yticklabels([f'ROI {roi}' for roi in roi_indices])
    axes[2].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[2].text(0.7, len(roi_indices)*0.9, 'Long ISI\nPreference', ha='center', color='red')
    axes[2].text(-0.7, len(roi_indices)*0.9, 'Short ISI\nPreference', ha='center', color='blue')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'response_matrix': response_matrix,
        'pattern_matrix': pattern_matrix,
        'roi_indices': roi_indices,
        'analyzed_isis': analyzed_isis
    }

def find_roi_component_switching(encoding_map: Dict[str, Any],
                                isi_phase_data: Dict[str, Any],
                                time_window_size: int = 50) -> Dict[str, Any]:
    """
    Look for ROIs that switch their ISI encoding patterns across the session
    (your "dancing pairs/groups" hypothesis)
    """
    print(f"\n=== ANALYZING ROI COMPONENT SWITCHING ===")
    
    isi_array = isi_phase_data['isi_phase_data']  # (trials, rois, phases)
    trial_metadata = isi_phase_data['trial_metadata']
    roi_encoding_map = encoding_map['roi_encoding_map']
    
    # Split session into time windows
    n_trials = len(trial_metadata)
    n_windows = max(2, n_trials // time_window_size)
    
    print(f"Analyzing {n_trials} trials in {n_windows} time windows of ~{time_window_size} trials each")
    
    switching_analysis = {}
    
    for roi_idx in roi_encoding_map.keys():
        if roi_idx >= isi_array.shape[1]:  # Safety check
            continue
            
        # Analyze this ROI's responses in different time windows
        window_patterns = {}
        
        for window_idx in range(n_windows):
            start_trial = window_idx * (n_trials // n_windows)
            end_trial = (window_idx + 1) * (n_trials // n_windows) if window_idx < n_windows - 1 else n_trials
            
            window_trials = range(start_trial, end_trial)
            window_metadata = [trial_metadata[i] for i in window_trials]
            
            # Group by ISI within this time window
            window_isi_responses = {}
            for trial_local_idx, trial_global_idx in enumerate(window_trials):
                isi_ms = trial_metadata[trial_global_idx]['isi_ms']
                
                if isi_ms not in window_isi_responses:
                    window_isi_responses[isi_ms] = []
                
                window_isi_responses[isi_ms].append(isi_array[trial_global_idx, roi_idx, :])
            
            # Calculate mean responses for each ISI in this window
            window_isi_means = {}
            for isi_ms, responses in window_isi_responses.items():
                if len(responses) >= 2:  # Need minimum trials
                    window_isi_means[isi_ms] = np.mean(responses, axis=0)
            
            window_patterns[window_idx] = {
                'isi_means': window_isi_means,
                'trial_range': (start_trial, end_trial),
                'n_trials': len(window_trials)
            }
        
        # Analyze pattern consistency across windows
        consistency_scores = {}
        if len(window_patterns) >= 2:
            # Compare each ISI response across windows
            common_isis = set.intersection(*[set(wp['isi_means'].keys()) for wp in window_patterns.values()])
            
            for isi_ms in common_isis:
                correlations = []
                for i in range(len(window_patterns)):
                    for j in range(i+1, len(window_patterns)):
                        if (isi_ms in window_patterns[i]['isi_means'] and 
                            isi_ms in window_patterns[j]['isi_means']):
                            
                            resp1 = window_patterns[i]['isi_means'][isi_ms]
                            resp2 = window_patterns[j]['isi_means'][isi_ms]
                            
                            if np.std(resp1) > 0 and np.std(resp2) > 0:
                                corr = np.corrcoef(resp1, resp2)[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(corr)
                
                consistency_scores[isi_ms] = np.mean(correlations) if correlations else np.nan
        
        # Determine if this ROI shows switching behavior
        mean_consistency = np.nanmean(list(consistency_scores.values())) if consistency_scores else np.nan
        
        switching_analysis[roi_idx] = {
            'window_patterns': window_patterns,
            'consistency_scores': consistency_scores,
            'mean_consistency': mean_consistency,
            'shows_switching': mean_consistency < 0.5 if not np.isnan(mean_consistency) else False,
            'n_windows_analyzed': len(window_patterns)
        }
    
    # Summary
    switching_rois = [roi_idx for roi_idx, analysis in switching_analysis.items() 
                     if analysis['shows_switching']]
    stable_rois = [roi_idx for roi_idx, analysis in switching_analysis.items() 
                  if not analysis['shows_switching'] and not np.isnan(analysis['mean_consistency'])]
    
    print(f"Found {len(switching_rois)} ROIs showing potential switching behavior")
    print(f"Found {len(stable_rois)} ROIs with stable patterns")
    
    return {
        'switching_analysis': switching_analysis,
        'switching_rois': switching_rois,
        'stable_rois': stable_rois,
        'time_window_size': time_window_size,
        'n_windows': n_windows
    }




































def analyze_reliable_roi_response_patterns(encoding_map: Dict[str, Any],
                                         isi_phase_data: Dict[str, Any],
                                         use_pca: bool = True,
                                         use_kmeans: bool = True,
                                         n_components: int = 3,
                                         n_clusters: int = 4) -> Dict[str, Any]:
    """
    Analyze response patterns for reliable ROIs using PCA and/or k-means
    Let the data tell us what patterns exist rather than forcing categories
    """
    print(f"\n=== ANALYZING RELIABLE ROI RESPONSE PATTERNS ===")
    
    roi_encoding_map = encoding_map['roi_encoding_map']
    isi_array = isi_phase_data['isi_phase_data']
    trial_metadata = isi_phase_data['trial_metadata']
    target_phases = isi_phase_data['target_phases']
    
    reliable_roi_indices = list(roi_encoding_map.keys())
    print(f"Analyzing {len(reliable_roi_indices)} reliable ROIs")
    
    # Extract all ISI response patterns for reliable ROIs
    unique_isis = sorted(list(set([meta['isi_ms'] for meta in trial_metadata])))
    print(f"ISI conditions: {unique_isis}")
    
    # Create pattern matrix: (roi_idx, isi_idx, phase_samples)
    pattern_data = []
    roi_isi_labels = []
    roi_labels = []
    isi_labels = []
    
    for roi_idx in reliable_roi_indices:
        roi_patterns = roi_encoding_map[roi_idx]['patterns']
        
        for isi_ms in unique_isis:
            if isi_ms in roi_patterns:
                # Fix: Access the correct key structure
                if 'mean_response' in roi_patterns[isi_ms]:
                    pattern = roi_patterns[isi_ms]['mean_response']
                elif 'overall_mean' in roi_patterns[isi_ms]:
                    # If mean_response doesn't exist, we need to get it from the original data
                    # Get the actual response pattern from the ISI array
                    isi_trials = [i for i, meta in enumerate(trial_metadata) if meta['isi_ms'] == isi_ms]
                    if len(isi_trials) >= 3:
                        roi_responses = isi_array[isi_trials, roi_idx, :]  # (trials, phases)
                        pattern = np.mean(roi_responses, axis=0)
                    else:
                        continue
                else:
                    # Skip this pattern if we can't find the response data
                    print(f"  Warning: No response data found for ROI {roi_idx}, ISI {isi_ms}")
                    continue
                
                pattern_data.append(pattern)
                roi_isi_labels.append(f'ROI{roi_idx}_ISI{isi_ms:.0f}')
                roi_labels.append(roi_idx)
                isi_labels.append(isi_ms)
    
    if len(pattern_data) == 0:
        print("  ERROR: No valid patterns found!")
        return None
    
    pattern_matrix = np.array(pattern_data)  # (n_patterns, n_phase_samples)
    print(f"Pattern matrix shape: {pattern_matrix.shape}")
    
    results = {
        'pattern_matrix': pattern_matrix,
        'roi_isi_labels': roi_isi_labels,
        'roi_labels': np.array(roi_labels),
        'isi_labels': np.array(isi_labels),
        'reliable_roi_indices': reliable_roi_indices,
        'unique_isis': unique_isis,
        'target_phases': target_phases
    }
    
    # Run PCA analysis
    if use_pca and len(pattern_data) > 1:
        print(f"\n--- PCA Analysis (n_components={n_components}) ---")
        from sklearn.decomposition import PCA
        
        # Adjust n_components based on available data
        max_components = min(n_components, len(pattern_data) - 1, pattern_matrix.shape[1] - 1)
        
        pca = PCA(n_components=max_components)
        pca_transformed = pca.fit_transform(pattern_matrix)
        
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
        
        results['pca'] = {
            'model': pca,
            'transformed': pca_transformed,
            'components': pca.components_,
            'explained_variance_ratio': pca.explained_variance_ratio_
        }
    
    # Run k-means clustering
    if use_kmeans and len(pattern_data) > n_clusters:
        print(f"\n--- K-means Clustering (n_clusters={n_clusters}) ---")
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pattern_matrix)
        
        # Analyze cluster composition
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_rois = np.unique(np.array(roi_labels)[cluster_mask])
            cluster_isis = np.array(isi_labels)[cluster_mask]
            
            print(f"  Cluster {cluster_id}: {np.sum(cluster_mask)} patterns")
            print(f"    ROIs: {cluster_rois}")
            if len(cluster_isis) > 0:
                unique_cluster_isis, counts = np.unique(cluster_isis, return_counts=True)
                print(f"    ISI distribution: {dict(zip(unique_cluster_isis.astype(int), counts))}")
        
        results['kmeans'] = {
            'model': kmeans,
            'labels': cluster_labels,
            'centers': kmeans.cluster_centers_,
            'n_clusters': n_clusters
        }
    elif use_kmeans:
        print(f"  Skipping k-means: insufficient patterns ({len(pattern_data)} < {n_clusters})")
    
    return results

def visualize_roi_pattern_analysis(analysis_results: Dict[str, Any],
                                  max_patterns_show: int = 50):
    """Visualize PCA and clustering results for reliable ROI patterns"""
    
    pattern_matrix = analysis_results['pattern_matrix']
    roi_labels = analysis_results['roi_labels']
    isi_labels = analysis_results['isi_labels']
    target_phases = analysis_results['target_phases']
    unique_isis = analysis_results['unique_isis']
    
    # Create comprehensive visualization
    if 'pca' in analysis_results and 'kmeans' in analysis_results:
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    elif 'pca' in analysis_results:
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    plot_idx = 0
    
    # PCA Analysis
    if 'pca' in analysis_results:
        pca_data = analysis_results['pca']
        pca_transformed = pca_data['transformed']
        components = pca_data['components']
        
        # Plot 1: PCA scatter colored by ISI
        scatter1 = axes[plot_idx].scatter(pca_transformed[:, 0], pca_transformed[:, 1],
                                         c=isi_labels, cmap='viridis', s=50, alpha=0.7)
        axes[plot_idx].set_xlabel(f'PC1 ({pca_data["explained_variance_ratio"][0]:.3f})')
        axes[plot_idx].set_ylabel(f'PC2 ({pca_data["explained_variance_ratio"][1]:.3f})')
        axes[plot_idx].set_title('PCA: Colored by ISI Duration')
        plt.colorbar(scatter1, ax=axes[plot_idx], label='ISI (ms)')
        plot_idx += 1
        
        # Plot 2: PCA scatter colored by ROI
        scatter2 = axes[plot_idx].scatter(pca_transformed[:, 0], pca_transformed[:, 1],
                                         c=roi_labels, cmap='tab20', s=50, alpha=0.7)
        axes[plot_idx].set_xlabel(f'PC1 ({pca_data["explained_variance_ratio"][0]:.3f})')
        axes[plot_idx].set_ylabel(f'PC2 ({pca_data["explained_variance_ratio"][1]:.3f})')
        axes[plot_idx].set_title('PCA: Colored by ROI Index')
        plt.colorbar(scatter2, ax=axes[plot_idx], label='ROI Index')
        plot_idx += 1
        
        # Plot 3: Principal components as temporal patterns
        for i in range(min(3, components.shape[0])):
            axes[plot_idx].plot(target_phases * 100, components[i], 
                               linewidth=2, label=f'PC{i+1} ({pca_data["explained_variance_ratio"][i]:.3f})')
        
        axes[plot_idx].set_xlabel('ISI Phase (%)')
        axes[plot_idx].set_ylabel('Component Weight')
        axes[plot_idx].set_title('Principal Components (Temporal Patterns)')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].axhline(0, color='gray', linestyle='-', alpha=0.5)
        plot_idx += 1
    
    # K-means Analysis
    if 'kmeans' in analysis_results:
        kmeans_data = analysis_results['kmeans']
        cluster_labels = kmeans_data['labels']
        cluster_centers = kmeans_data['centers']
        n_clusters = kmeans_data['n_clusters']
        
        # Plot 4: Cluster assignments (if we have PCA)
        if 'pca' in analysis_results:
            colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
            for cluster_id in range(n_clusters):
                mask = cluster_labels == cluster_id
                axes[plot_idx].scatter(pca_transformed[mask, 0], pca_transformed[mask, 1],
                                      c=[colors[cluster_id]], s=50, alpha=0.7,
                                      label=f'Cluster {cluster_id} (n={np.sum(mask)})')
            
            axes[plot_idx].set_xlabel(f'PC1 ({pca_data["explained_variance_ratio"][0]:.3f})')
            axes[plot_idx].set_ylabel(f'PC2 ({pca_data["explained_variance_ratio"][1]:.3f})')
            axes[plot_idx].set_title('K-means Clusters in PCA Space')
            axes[plot_idx].legend()
            plot_idx += 1
        
        # Plot 5: Cluster centers as temporal patterns
        colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
        for cluster_id in range(n_clusters):
            center = cluster_centers[cluster_id]
            n_in_cluster = np.sum(cluster_labels == cluster_id)
            axes[plot_idx].plot(target_phases * 100, center, color=colors[cluster_id],
                               linewidth=3, label=f'Cluster {cluster_id} (n={n_in_cluster})')
        
        axes[plot_idx].set_xlabel('ISI Phase (%)')
        axes[plot_idx].set_ylabel('z-scored dF/F')
        axes[plot_idx].set_title('K-means Cluster Centers (Temporal Patterns)')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].axhline(0, color='gray', linestyle='-', alpha=0.5)
        plot_idx += 1
        
        # Plot 6: Cluster composition by ISI
        cluster_isi_matrix = np.zeros((n_clusters, len(unique_isis)))
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_isis = isi_labels[mask]
            for isi_idx, isi in enumerate(unique_isis):
                cluster_isi_matrix[cluster_id, isi_idx] = np.sum(cluster_isis == isi)
        
        # Normalize by cluster size for percentage
        cluster_sizes = np.sum(cluster_isi_matrix, axis=1, keepdims=True)
        cluster_isi_percent = cluster_isi_matrix / (cluster_sizes + 1e-6) * 100
        
        im = axes[plot_idx].imshow(cluster_isi_percent, aspect='auto', cmap='Blues',
                                  extent=[0, len(unique_isis), n_clusters, 0])
        axes[plot_idx].set_xticks(range(len(unique_isis)))
        axes[plot_idx].set_xticklabels([f'{isi:.0f}ms' for isi in unique_isis], rotation=45)
        axes[plot_idx].set_yticks(range(n_clusters))
        axes[plot_idx].set_yticklabels([f'Cluster {i}' for i in range(n_clusters)])
        axes[plot_idx].set_title('Cluster Composition by ISI (%)')
        plt.colorbar(im, ax=axes[plot_idx], label='Percentage of cluster')
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def visualize_individual_roi_analysis(roi_analysis: Dict[str, Any]):
    """Visualize analysis results for a single ROI"""
    
    if roi_analysis is None:
        return
    
    roi_idx = roi_analysis['roi_idx']
    response_matrix = roi_analysis['response_matrix']
    isi_values = roi_analysis['isi_values']
    target_phases = roi_analysis['target_phases']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: All ISI responses
    colors = plt.cm.viridis(np.linspace(0, 1, len(isi_values)))
    for i, (isi, response) in enumerate(zip(isi_values, response_matrix)):
        color = 'blue' if isi <= 700 else 'red'
        axes[0, 0].plot(target_phases * 100, response, color=color, alpha=0.7,
                       linewidth=2, label=f'ISI {isi:.0f}ms')
    
    axes[0, 0].set_xlabel('ISI Phase (%)')
    axes[0, 0].set_ylabel('z-scored dF/F')
    axes[0, 0].set_title(f'ROI {roi_idx}: All ISI Responses')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    # Plot 2: Response heatmap
    im = axes[0, 1].imshow(response_matrix, aspect='auto', cmap='RdBu_r',
                          extent=[0, 100, len(isi_values), 0])
    axes[0, 1].set_xlabel('ISI Phase (%)')
    axes[0, 1].set_ylabel('ISI Condition')
    axes[0, 1].set_yticks(range(len(isi_values)))
    axes[0, 1].set_yticklabels([f'{isi:.0f}ms' for isi in isi_values])
    axes[0, 1].set_title(f'ROI {roi_idx}: Response Heatmap')
    plt.colorbar(im, ax=axes[0, 1], label='z-scored dF/F')
    
    # PCA analysis if available
    if 'pca' in roi_analysis:
        pca_data = roi_analysis['pca']
        components = pca_data['components']
        explained_var = pca_data['explained_variance_ratio']
        
        # Plot 3: Principal components
        for i in range(components.shape[0]):
            axes[1, 0].plot(target_phases * 100, components[i], linewidth=2,
                           label=f'PC{i+1} ({explained_var[i]:.3f})')
        
        axes[1, 0].set_xlabel('ISI Phase (%)')
        axes[1, 0].set_ylabel('Component Weight')
        axes[1, 0].set_title(f'ROI {roi_idx}: Principal Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        # Plot 4: ISI projections in PC space
        if components.shape[0] >= 2:
            pca_transformed = pca_data['transformed']
            scatter = axes[1, 1].scatter(pca_transformed[:, 0], pca_transformed[:, 1],
                                        c=isi_values, cmap='viridis', s=100, alpha=0.7)
            
            # Annotate points with ISI values
            for i, isi in enumerate(isi_values):
                axes[1, 1].annotate(f'{isi:.0f}', 
                                   (pca_transformed[i, 0], pca_transformed[i, 1]),
                                   xytext=(5, 5), textcoords='offset points')
            
            axes[1, 1].set_xlabel(f'PC1 ({explained_var[0]:.3f})')
            axes[1, 1].set_ylabel(f'PC2 ({explained_var[1]:.3f})')
            axes[1, 1].set_title(f'ROI {roi_idx}: ISI Responses in PC Space')
            plt.colorbar(scatter, ax=axes[1, 1], label='ISI (ms)')
        else:
            axes[1, 1].text(0.5, 0.5, 'Need ≥2 PCs for scatter plot', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
    else:
        axes[1, 0].text(0.5, 0.5, 'No PCA data', ha='center', va='center', 
                       transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, 'No PCA data', ha='center', va='center',
                       transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.show()






























def analyze_individual_roi_patterns(encoding_map: Dict[str, Any],
                                   isi_phase_data: Dict[str, Any],
                                   roi_idx: int,
                                   use_pca: bool = True,
                                   n_components: int = 3) -> Dict[str, Any]:
    """
    Focus on a single ROI - analyze its response patterns across ISIs
    """
    print(f"\n=== ANALYZING ROI {roi_idx} PATTERNS ===")
    
    roi_encoding_map = encoding_map['roi_encoding_map']
    target_phases = isi_phase_data['target_phases']
    isi_array = isi_phase_data['isi_phase_data']
    trial_metadata = isi_phase_data['trial_metadata']
    
    if roi_idx not in roi_encoding_map:
        print(f"ROI {roi_idx} not found in reliable encoders")
        return None
    
    roi_data = roi_encoding_map[roi_idx]
    roi_patterns = roi_data['patterns']
    
    print(f"ROI {roi_idx} responds to {len(roi_patterns)} ISI conditions")
    print(f"Consistency score: {roi_data['consistency_score']:.3f}")
    print(f"Reliability score: {roi_data['reliability_score']:.3f}")
    
    # Extract response patterns for this ROI by reconstructing from original data
    isi_responses = []
    isi_values = []
    
    for isi_ms in roi_patterns.keys():
        # Get trials for this ISI
        isi_trials = [i for i, meta in enumerate(trial_metadata) if meta['isi_ms'] == isi_ms]
        
        if len(isi_trials) >= 3:  # Need minimum trials
            # Extract actual response pattern from original data
            roi_responses = isi_array[isi_trials, roi_idx, :]  # (trials, phases)
            mean_response = np.mean(roi_responses, axis=0)
            
            isi_responses.append(mean_response)
            isi_values.append(isi_ms)
    
    if len(isi_responses) == 0:
        print(f"No valid responses found for ROI {roi_idx}")
        return None
    
    response_matrix = np.array(isi_responses)  # (n_isis, n_phase_samples)
    
    # Run PCA on this ROI's ISI responses
    results = {
        'roi_idx': roi_idx,
        'response_matrix': response_matrix,
        'isi_values': np.array(isi_values),
        'target_phases': target_phases,
        'roi_data': roi_data
    }
    
    if use_pca and len(isi_responses) > 1:
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=min(n_components, len(isi_responses)))
        pca_transformed = pca.fit_transform(response_matrix)
        
        print(f"PCA explained variance: {pca.explained_variance_ratio_}")
        
        results['pca'] = {
            'model': pca,
            'transformed': pca_transformed,
            'components': pca.components_,
            'explained_variance_ratio': pca.explained_variance_ratio_
        }
    
    return results





def map_roi_isi_encoding_patterns(consistency_results: Dict[str, Any], 
                                 reliable_encoders: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map primary response patterns for reliable encoder ROIs across different ISIs
    """
    print(f"\n=== MAPPING ROI ISI ENCODING PATTERNS ===")
    
    reliable_rois = reliable_encoders['reliable_rois']
    isi_groups = consistency_results['isi_groups']
    
    print(f"Analyzing {len(reliable_rois)} reliable encoder ROIs")
    
    # For each reliable ROI, characterize its response pattern to each ISI
    roi_encoding_map = {}
    
    for roi_data in reliable_rois:
        roi_idx = roi_data['roi_idx']
        isi_responses = roi_data['isi_responses']
        
        roi_patterns = {}
        
        for isi_ms, response_data in isi_responses.items():
            mean_response = response_data['mean_response']
            
            # Characterize the temporal pattern AND store the mean response
            pattern_features = {
                'mean_response': mean_response,  # Store the actual response array
                'overall_mean': np.mean(mean_response),
                'overall_std': np.std(mean_response),
                'peak_amplitude': np.max(np.abs(mean_response)),
                'peak_time_percent': np.argmax(np.abs(mean_response)) / len(mean_response) * 100,
                'early_activity': np.mean(mean_response[:len(mean_response)//3]),
                'mid_activity': np.mean(mean_response[len(mean_response)//3:2*len(mean_response)//3]),
                'late_activity': np.mean(mean_response[2*len(mean_response)//3:]),
                'direction': 'positive' if np.mean(mean_response) > 0 else 'negative',
                'pattern_type': classify_temporal_pattern(mean_response)
            }
            
            roi_patterns[isi_ms] = pattern_features
        
        roi_encoding_map[roi_idx] = {
            'patterns': roi_patterns,
            'consistency_score': roi_data['consistency_score'],
            'reliability_score': roi_data['reliability_score']
        }
    
    return {
        'roi_encoding_map': roi_encoding_map,
        'n_reliable_rois': len(reliable_rois),
        'analyzed_isis': list(isi_groups.keys())
    }

























def analyze_trial_level_roi_components(isi_phase_data: Dict[str, Any],
                                     n_components: int = 5,
                                     use_trial_groups: bool = True) -> Dict[str, Any]:
    """
    Find temporal components while preserving ROI-specific and trial-level variation
    This allows us to track which ROIs use which components on individual trials
    """
    print(f"\n=== TRIAL-LEVEL ROI COMPONENT ANALYSIS ===")
    
    isi_array = isi_phase_data['isi_phase_data']  # (trials, rois, phases)
    trial_metadata = isi_phase_data['trial_metadata']
    target_phases = isi_phase_data['target_phases']
    
    n_trials, n_rois, n_phases = isi_array.shape
    print(f"Analyzing {n_trials} trials × {n_rois} ROIs × {n_phases} phase samples")
    
    # Reshape to (trials*rois, phases) to find components across all trial-ROI combinations
    trial_roi_matrix = isi_array.reshape(n_trials * n_rois, n_phases)
    
    # Remove any all-NaN trial-ROI combinations
    valid_mask = ~np.all(np.isnan(trial_roi_matrix), axis=1)
    trial_roi_matrix_clean = trial_roi_matrix[valid_mask]
    
    print(f"Valid trial-ROI combinations: {np.sum(valid_mask)}/{len(valid_mask)}")
    
    # Find temporal components using PCA
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=n_components)
    trial_roi_loadings = pca.fit_transform(trial_roi_matrix_clean)  # (valid_trial_rois, components)
    temporal_components = pca.components_  # (components, phases)
    
    print(f"Component explained variance: {pca.explained_variance_ratio_}")
    
    # Reshape loadings back to (trials, rois, components) structure
    trial_roi_component_loadings = np.full((n_trials, n_rois, n_components), np.nan)
    
    valid_idx = 0
    for trial_idx in range(n_trials):
        for roi_idx in range(n_rois):
            flat_idx = trial_idx * n_rois + roi_idx
            if valid_mask[flat_idx]:
                trial_roi_component_loadings[trial_idx, roi_idx, :] = trial_roi_loadings[valid_idx]
                valid_idx += 1
    
    # For each trial and ROI, determine which component is dominant
    dominant_components = np.zeros((n_trials, n_rois), dtype=int)
    component_strengths = np.zeros((n_trials, n_rois))
    
    for trial_idx in range(n_trials):
        for roi_idx in range(n_rois):
            loadings = trial_roi_component_loadings[trial_idx, roi_idx, :]
            if not np.all(np.isnan(loadings)):
                # Find component with highest absolute loading
                abs_loadings = np.abs(loadings)
                dominant_comp = np.argmax(abs_loadings)
                dominant_components[trial_idx, roi_idx] = dominant_comp
                component_strengths[trial_idx, roi_idx] = abs_loadings[dominant_comp]
    
    return {
        'temporal_components': temporal_components,  # (components, phases)
        'trial_roi_component_loadings': trial_roi_component_loadings,  # (trials, rois, components)
        'dominant_components': dominant_components,  # (trials, rois) - which component is strongest
        'component_strengths': component_strengths,  # (trials, rois) - strength of dominant component
        'pca_model': pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'target_phases': target_phases,
        'trial_metadata': trial_metadata,
        'n_components': n_components
    }

def analyze_roi_component_switching(component_results: Dict[str, Any],
                                  min_trials_per_window: int = 20) -> Dict[str, Any]:
    """
    Analyze which ROIs switch between components across trials/time
    This directly tests the "dancing pairs/groups" hypothesis
    """
    print(f"\n=== ANALYZING ROI COMPONENT SWITCHING ===")
    
    dominant_components = component_results['dominant_components']  # (trials, rois)
    component_strengths = component_results['component_strengths']
    trial_metadata = component_results['trial_metadata']
    n_components = component_results['n_components']
    
    n_trials, n_rois = dominant_components.shape
    
    # Split trials into temporal windows
    n_windows = max(2, n_trials // min_trials_per_window)
    window_size = n_trials // n_windows
    
    print(f"Analyzing {n_windows} time windows of ~{window_size} trials each")
    
    roi_switching_analysis = {}
    
    for roi_idx in range(n_rois):
        # Track this ROI's component usage across time windows
        window_component_usage = {}
        
        for window_idx in range(n_windows):
            start_trial = window_idx * window_size
            end_trial = (window_idx + 1) * window_size if window_idx < n_windows - 1 else n_trials
            
            window_trials = range(start_trial, end_trial)
            
            # Get component usage in this window
            window_components = dominant_components[start_trial:end_trial, roi_idx]
            window_strengths = component_strengths[start_trial:end_trial, roi_idx]
            
            # Only consider trials with sufficient component strength
            strong_trials = window_strengths > np.percentile(window_strengths, 50)  # Above median
            
            if np.sum(strong_trials) > 0:
                strong_components = window_components[strong_trials]
                
                # Calculate component usage percentages
                component_counts = np.bincount(strong_components, minlength=n_components)
                component_percentages = component_counts / np.sum(component_counts) * 100
                
                window_component_usage[window_idx] = {
                    'component_percentages': component_percentages,
                    'primary_component': np.argmax(component_percentages),
                    'primary_percentage': np.max(component_percentages),
                    'n_strong_trials': np.sum(strong_trials),
                    'trial_range': (start_trial, end_trial)
                }
        
        # Analyze switching behavior
        primary_components = [usage['primary_component'] for usage in window_component_usage.values()]
        primary_percentages = [usage['primary_percentage'] for usage in window_component_usage.values()]
        
        # Calculate switching metrics
        n_switches = np.sum(np.diff(primary_components) != 0)
        switching_rate = n_switches / (len(primary_components) - 1) if len(primary_components) > 1 else 0
        
        # Component diversity (entropy-like measure)
        all_percentages = np.array([usage['component_percentages'] for usage in window_component_usage.values()])
        mean_percentages = np.mean(all_percentages, axis=0)
        component_diversity = -np.sum(mean_percentages * np.log(mean_percentages + 1e-6)) / np.log(n_components)
        
        roi_switching_analysis[roi_idx] = {
            'window_component_usage': window_component_usage,
            'primary_components': primary_components,
            'n_switches': n_switches,
            'switching_rate': switching_rate,
            'component_diversity': component_diversity,
            'shows_switching': switching_rate > 0.3 and component_diversity > 0.3,  # Threshold for "switcher"
            'n_windows_analyzed': len(window_component_usage)
        }
    
    # Identify different ROI types
    switcher_rois = [roi_idx for roi_idx, analysis in roi_switching_analysis.items() 
                    if analysis['shows_switching']]
    
    stable_rois = [roi_idx for roi_idx, analysis in roi_switching_analysis.items() 
                  if not analysis['shows_switching'] and analysis['component_diversity'] < 0.3]
    
    diverse_rois = [roi_idx for roi_idx, analysis in roi_switching_analysis.items() 
                   if not analysis['shows_switching'] and analysis['component_diversity'] >= 0.3]
    
    print(f"ROI classification:")
    print(f"  Switcher ROIs: {len(switcher_rois)} (change primary component across time)")
    print(f"  Stable ROIs: {len(stable_rois)} (consistent primary component)")
    print(f"  Diverse ROIs: {len(diverse_rois)} (use multiple components consistently)")
    
    return {
        'roi_switching_analysis': roi_switching_analysis,
        'switcher_rois': switcher_rois,
        'stable_rois': stable_rois,
        'diverse_rois': diverse_rois,
        'n_windows': n_windows,
        'window_size': window_size
    }

def find_roi_component_pairs(component_results: Dict[str, Any],
                           switching_analysis: Dict[str, Any],
                           spatial_threshold: float = None) -> Dict[str, Any]:
    """
    Find ROI pairs that show complementary component switching
    (the "dancing pairs" - when one ROI uses component A, the other uses component B)
    """
    print(f"\n=== FINDING ROI COMPONENT PAIRS ===")
    
    dominant_components = component_results['dominant_components']  # (trials, rois)
    switcher_rois = switching_analysis['switcher_rois']
    
    if len(switcher_rois) < 2:
        print("Need at least 2 switcher ROIs for pair analysis")
        return {'pairs': []}
    
    print(f"Analyzing {len(switcher_rois)} switcher ROIs for complementary patterns")
    
    # Calculate component correlation between all switcher ROI pairs
    complementary_pairs = []
    
    for i, roi1 in enumerate(switcher_rois):
        for j, roi2 in enumerate(switcher_rois[i+1:], i+1):
            
            # Get component usage time series for both ROIs
            roi1_components = dominant_components[:, roi1]
            roi2_components = dominant_components[:, roi2]
            
            # Calculate trial-by-trial component correlation
            # Negative correlation suggests complementary switching
            try:
                correlation = np.corrcoef(roi1_components, roi2_components)[0, 1]
                
                # Look for anti-correlated component usage (complementary switching)
                if correlation < -0.3:  # Threshold for complementary behavior
                    
                    # Calculate more detailed switching metrics
                    switching_events = []
                    for trial_idx in range(len(roi1_components)-1):
                        roi1_switch = roi1_components[trial_idx] != roi1_components[trial_idx+1]
                        roi2_switch = roi2_components[trial_idx] != roi2_components[trial_idx+1]
                        
                        # Both switch at the same time (coordinated switching)
                        if roi1_switch and roi2_switch:
                            switching_events.append({
                                'trial': trial_idx,
                                'roi1_from': roi1_components[trial_idx],
                                'roi1_to': roi1_components[trial_idx+1],
                                'roi2_from': roi2_components[trial_idx],
                                'roi2_to': roi2_components[trial_idx+1]
                            })
                    
                    complementary_pairs.append({
                        'roi1': roi1,
                        'roi2': roi2,
                        'correlation': correlation,
                        'n_coordinated_switches': len(switching_events),
                        'switching_events': switching_events,
                        'pair_type': 'complementary'
                    })
                
                # Also look for highly correlated pairs (coordinated switching to same components)
                elif correlation > 0.6:
                    complementary_pairs.append({
                        'roi1': roi1,
                        'roi2': roi2,
                        'correlation': correlation,
                        'pair_type': 'coordinated'
                    })
                    
            except:
                continue  # Skip if correlation calculation fails
    
    print(f"Found {len(complementary_pairs)} ROI pairs with component switching relationships")
    
    # Group by pair type
    complementary_type = [p for p in complementary_pairs if p['pair_type'] == 'complementary']
    coordinated_type = [p for p in complementary_pairs if p['pair_type'] == 'coordinated']
    
    print(f"  Complementary pairs: {len(complementary_type)} (anti-correlated switching)")
    print(f"  Coordinated pairs: {len(coordinated_type)} (synchronized switching)")
    
    return {
        'pairs': complementary_pairs,
        'complementary_pairs': complementary_type,
        'coordinated_pairs': coordinated_type,
        'switcher_rois': switcher_rois
    }









def visualize_roi_component_dynamics(component_results: Dict[str, Any],
                                   switching_analysis: Dict[str, Any],
                                   pairs_analysis: Dict[str, Any],
                                   max_rois_show: int = 6):
    """
    Visualize the temporal component dynamics and ROI switching behavior
    """
    
    temporal_components = component_results['temporal_components']
    dominant_components = component_results['dominant_components']
    target_phases = component_results['target_phases']
    switcher_rois = switching_analysis['switcher_rois']
    
    # Handle case where no pairs were found
    if pairs_analysis and 'complementary_pairs' in pairs_analysis:
        complementary_pairs = pairs_analysis['complementary_pairs']
    else:
        complementary_pairs = []
        print("No complementary pairs found")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(4, 3, figure=fig)
    
    # Top row: Temporal components
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.Set1(np.linspace(0, 1, len(temporal_components)))
    
    for comp_idx, component in enumerate(temporal_components):
        ax1.plot(target_phases * 100, component, color=colors[comp_idx], 
                linewidth=3, label=f'Component {comp_idx} ({component_results["explained_variance_ratio"][comp_idx]:.3f})')
    
    ax1.set_xlabel('ISI Phase (%)')
    ax1.set_ylabel('Component Weight')
    ax1.set_title('Discovered Temporal Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    # Second row: ROI component usage over trials
    n_show = min(max_rois_show, len(switcher_rois))
    
    for plot_idx in range(min(3, n_show)):
        ax = fig.add_subplot(gs[1, plot_idx])
        
        if plot_idx < len(switcher_rois):
            roi_idx = switcher_rois[plot_idx]
            roi_components = dominant_components[:, roi_idx]
            
            # Create color-coded trial sequence
            trial_colors = [colors[comp] for comp in roi_components]
            ax.scatter(range(len(roi_components)), roi_components, 
                      c=trial_colors, s=30, alpha=0.7)
            
            ax.set_xlabel('Trial Number')
            ax.set_ylabel('Dominant Component')
            ax.set_title(f'ROI {roi_idx}: Component Switching')
            ax.set_ylim(-0.5, len(temporal_components)-0.5)
            ax.grid(True, alpha=0.3)
    
    # Third row: Component usage heatmaps for switcher ROIs
    if len(switcher_rois) > 0:
        ax_heatmap = fig.add_subplot(gs[2, :])
        
        # Create heatmap showing component usage for switcher ROIs
        switcher_component_matrix = dominant_components[:, switcher_rois[:20]].T  # Transpose for ROIs on y-axis
        
        im = ax_heatmap.imshow(switcher_component_matrix, aspect='auto', cmap='tab10', 
                              vmin=0, vmax=len(temporal_components)-1,
                              extent=[0, len(dominant_components), len(switcher_rois[:20]), 0])
        
        ax_heatmap.set_xlabel('Trial Number')
        ax_heatmap.set_ylabel('ROI Index (Switchers only)')
        ax_heatmap.set_title('Component Usage Across Trials (Switcher ROIs)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_heatmap, label='Dominant Component')
        cbar.set_ticks(range(len(temporal_components)))
    
    # Bottom row: Complementary pairs (if any found)
    if len(complementary_pairs) > 0:
        for pair_idx in range(min(3, len(complementary_pairs))):
            ax = fig.add_subplot(gs[3, pair_idx])
            
            pair = complementary_pairs[pair_idx]
            roi1, roi2 = pair['roi1'], pair['roi2']
            
            roi1_components = dominant_components[:, roi1]
            roi2_components = dominant_components[:, roi2]
            
            ax.plot(roi1_components, 'o-', alpha=0.7, label=f'ROI {roi1}')
            ax.plot(roi2_components, 's-', alpha=0.7, label=f'ROI {roi2}')
            
            ax.set_xlabel('Trial Number')
            ax.set_ylabel('Dominant Component')
            ax.set_title(f'Complementary Pair: ROI {roi1} & {roi2}\n(r={pair["correlation"]:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        # Fill bottom row with message if no pairs found
        ax = fig.add_subplot(gs[3, :])
        ax.text(0.5, 0.5, 'No complementary pairs found\n(insufficient switcher ROIs or no anti-correlated patterns)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== COMPONENT DYNAMICS SUMMARY ===")
    print(f"Total components found: {len(temporal_components)}")
    print(f"Switcher ROIs: {len(switcher_rois)}")
    print(f"Complementary pairs: {len(complementary_pairs)}")
    
    if len(complementary_pairs) > 0:
        print(f"\nComplementary pair details:")
        for pair in complementary_pairs[:5]:  # Show first 5
            print(f"  ROI {pair['roi1']} & {pair['roi2']}: r={pair['correlation']:.3f}, "
                  f"{pair.get('n_coordinated_switches', 0)} coordinated switches")
    else:
        print(f"\nNo complementary pairs found. This could indicate:")
        print(f"  - Insufficient switcher ROIs ({len(switcher_rois)} found)")
        print(f"  - No anti-correlated component switching patterns")
        print(f"  - ROIs may switch independently rather than in coordinated pairs")





































def find_all_isi_components_comprehensive(isi_phase_data: Dict[str, Any],
                                        n_components: int = 30,
                                        include_all_rois: bool = True) -> Dict[str, Any]:
    """
    Find ALL temporal components that occur across ISIs without losing sparse ones
    This is the first step - comprehensive component discovery
    """
    print(f"\n=== COMPREHENSIVE COMPONENT DISCOVERY ===")
    
    isi_array = isi_phase_data['isi_phase_data']  # (trials, rois, phases)
    trial_metadata = isi_phase_data['trial_metadata']
    target_phases = isi_phase_data['target_phases']
    
    n_trials, n_rois, n_phases = isi_array.shape
    print(f"Analyzing {n_trials} trials × {n_rois} ROIs × {n_phases} phase samples")
    
    # Group trials by ISI to ensure we capture components from all ISI conditions
    unique_isis = sorted(list(set([meta['isi_ms'] for meta in trial_metadata])))
    print(f"ISI conditions: {unique_isis}")
    
    # Collect ALL trial-ROI-ISI combinations for comprehensive component discovery
    all_patterns = []
    all_metadata = []
    
    for isi_ms in unique_isis:
        isi_trials = [i for i, meta in enumerate(trial_metadata) if meta['isi_ms'] == isi_ms]
        print(f"  ISI {isi_ms}ms: {len(isi_trials)} trials")
        
        for trial_idx in isi_trials:
            for roi_idx in range(n_rois):
                pattern = isi_array[trial_idx, roi_idx, :]
                
                # Only include if not all NaN
                if not np.all(np.isnan(pattern)):
                    all_patterns.append(pattern)
                    all_metadata.append({
                        'trial_idx': trial_idx,
                        'roi_idx': roi_idx,
                        'isi_ms': isi_ms,
                        'pattern_id': len(all_patterns) - 1
                    })
    
    print(f"Total patterns collected: {len(all_patterns)}")
    
    # Convert to matrix for PCA
    pattern_matrix = np.array(all_patterns)  # (n_patterns, n_phases)
    
    # Use higher number of components to capture sparse patterns
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=min(n_components, len(all_patterns) - 1))
    pattern_loadings = pca.fit_transform(pattern_matrix)  # (n_patterns, n_components)
    temporal_components = pca.components_  # (n_components, n_phases)
    
    print(f"Found {len(temporal_components)} components")
    print(f"Explained variance (first 10): {pca.explained_variance_ratio_[:10]}")
    print(f"Cumulative variance (first 10): {np.cumsum(pca.explained_variance_ratio_[:10])}")
    
    return {
        'temporal_components': temporal_components,  # (n_components, n_phases)
        'pattern_loadings': pattern_loadings,       # (n_patterns, n_components)
        'pattern_metadata': all_metadata,           # Metadata for each pattern
        'pattern_matrix': pattern_matrix,           # Original patterns
        'pca_model': pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'target_phases': target_phases,
        'unique_isis': unique_isis,
        'n_total_patterns': len(all_patterns)
    }

def map_roi_isi_to_components(component_results: Dict[str, Any],
                             loading_threshold: float = 0.1) -> Dict[str, Any]:
    """
    Map which components each ROI uses for each ISI condition
    This preserves multiple component usage per ROI-ISI combination
    """
    print(f"\n=== MAPPING ROI-ISI TO COMPONENTS ===")
    
    pattern_loadings = component_results['pattern_loadings']
    pattern_metadata = component_results['pattern_metadata']
    temporal_components = component_results['temporal_components']
    unique_isis = component_results['unique_isis']
    
    n_components = temporal_components.shape[0]
    
    # Create comprehensive mapping: roi -> isi -> components
    roi_isi_component_map = {}
    
    # Get all unique ROIs
    all_rois = sorted(list(set([meta['roi_idx'] for meta in pattern_metadata])))
    
    for roi_idx in all_rois:
        roi_isi_component_map[roi_idx] = {}
        
        for isi_ms in unique_isis:
            # Find all patterns for this ROI-ISI combination
            roi_isi_patterns = [
                i for i, meta in enumerate(pattern_metadata)
                if meta['roi_idx'] == roi_idx and meta['isi_ms'] == isi_ms
            ]
            
            if len(roi_isi_patterns) == 0:
                continue
            
            # Get loadings for these patterns
            roi_isi_loadings = pattern_loadings[roi_isi_patterns]  # (n_patterns_for_roi_isi, n_components)
            
            # Find components that are consistently used (across trials for this ROI-ISI)
            mean_loadings = np.mean(np.abs(roi_isi_loadings), axis=0)  # Mean absolute loading per component
            
            # Find components above threshold
            active_components = np.where(mean_loadings > loading_threshold)[0]
            
            if len(active_components) > 0:
                component_info = {}
                for comp_idx in active_components:
                    # Calculate statistics for this component
                    comp_loadings = roi_isi_loadings[:, comp_idx]
                    
                    component_info[comp_idx] = {
                        'mean_loading': mean_loadings[comp_idx],
                        'loading_std': np.std(comp_loadings),
                        'loading_consistency': 1.0 / (1.0 + np.std(comp_loadings)),  # Higher = more consistent
                        'n_trials': len(roi_isi_patterns),
                        'raw_loadings': comp_loadings
                    }
                
                roi_isi_component_map[roi_idx][isi_ms] = component_info
    
    # Create summary statistics
    total_mappings = sum(len(isi_dict) for isi_dict in roi_isi_component_map.values())
    print(f"Created mappings for {len(roi_isi_component_map)} ROIs across {len(unique_isis)} ISIs")
    print(f"Total ROI-ISI combinations mapped: {total_mappings}")
    
    # Analyze component usage statistics
    component_usage_count = np.zeros(n_components)
    component_usage_by_isi = {isi: np.zeros(n_components) for isi in unique_isis}
    
    for roi_idx, isi_dict in roi_isi_component_map.items():
        for isi_ms, comp_dict in isi_dict.items():
            for comp_idx in comp_dict.keys():
                component_usage_count[comp_idx] += 1
                component_usage_by_isi[isi_ms][comp_idx] += 1
    
    print(f"\nComponent usage statistics:")
    for comp_idx in range(min(15, n_components)):  # Show first 15
        usage_count = component_usage_count[comp_idx]
        usage_percent = 100 * usage_count / total_mappings
        print(f"  Component {comp_idx}: used in {usage_count} ROI-ISI combinations ({usage_percent:.1f}%)")
    
    return {
        'roi_isi_component_map': roi_isi_component_map,
        'component_usage_count': component_usage_count,
        'component_usage_by_isi': component_usage_by_isi,
        'total_mappings': total_mappings,
        'n_components': n_components,
        'unique_isis': unique_isis,
        'loading_threshold': loading_threshold
    }

def analyze_component_isi_specificity(component_results: Dict[str, Any],
                                    mapping_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze which components are ISI-specific vs general
    This helps identify the deeper, more specific components
    """
    print(f"\n=== ANALYZING COMPONENT ISI SPECIFICITY ===")
    
    temporal_components = component_results['temporal_components']
    component_usage_by_isi = mapping_results['component_usage_by_isi']
    unique_isis = mapping_results['unique_isis']
    n_components = mapping_results['n_components']
    
    component_specificity = {}
    
    for comp_idx in range(n_components):
        # Calculate usage across ISIs
        isi_usage = [component_usage_by_isi[isi][comp_idx] for isi in unique_isis]
        total_usage = sum(isi_usage)
        
        if total_usage == 0:
            continue
        
        # Calculate specificity metrics
        isi_proportions = np.array(isi_usage) / total_usage
        
        # Shannon entropy (lower = more specific)
        entropy = -np.sum(isi_proportions * np.log(isi_proportions + 1e-10))
        max_entropy = np.log(len(unique_isis))
        specificity_score = 1.0 - (entropy / max_entropy)  # 1 = completely specific, 0 = completely general
        
        # Find primary ISI
        primary_isi_idx = np.argmax(isi_usage)
        primary_isi = unique_isis[primary_isi_idx]
        primary_proportion = isi_proportions[primary_isi_idx]
        
        component_specificity[comp_idx] = {
            'total_usage': total_usage,
            'isi_usage': dict(zip(unique_isis, isi_usage)),
            'isi_proportions': dict(zip(unique_isis, isi_proportions)),
            'specificity_score': specificity_score,
            'primary_isi': primary_isi,
            'primary_proportion': primary_proportion,
            'component_type': 'specific' if specificity_score > 0.7 else 'moderate' if specificity_score > 0.3 else 'general'
        }
    
    # Categorize components
    specific_components = [c for c, info in component_specificity.items() if info['component_type'] == 'specific']
    moderate_components = [c for c, info in component_specificity.items() if info['component_type'] == 'moderate']
    general_components = [c for c, info in component_specificity.items() if info['component_type'] == 'general']
    
    print(f"Component categorization:")
    print(f"  Specific components (ISI-specific): {len(specific_components)}")
    print(f"  Moderate components (partially specific): {len(moderate_components)}")
    print(f"  General components (used across ISIs): {len(general_components)}")
    
    # Show details for specific components
    print(f"\nISI-specific components:")
    for comp_idx in specific_components[:10]:  # Show first 10
        info = component_specificity[comp_idx]
        print(f"  Component {comp_idx}: {info['primary_proportion']:.1%} usage in ISI {info['primary_isi']}ms")
    
    return {
        'component_specificity': component_specificity,
        'specific_components': specific_components,
        'moderate_components': moderate_components,
        'general_components': general_components,
        'categorization_summary': {
            'specific': len(specific_components),
            'moderate': len(moderate_components),
            'general': len(general_components)
        }
    }

def visualize_comprehensive_components(component_results: Dict[str, Any],
                                     mapping_results: Dict[str, Any],
                                     specificity_results: Dict[str, Any],
                                     max_components_show: int = 12):
    """
    Visualize the comprehensive component analysis results
    """
    
    temporal_components = component_results['temporal_components']
    target_phases = component_results['target_phases']
    explained_variance = component_results['explained_variance_ratio']
    
    specific_components = specificity_results['specific_components']
    moderate_components = specificity_results['moderate_components']
    general_components = specificity_results['general_components']
    component_specificity = specificity_results['component_specificity']
    
    # Create visualization
    n_show = min(max_components_show, len(temporal_components))
    n_cols = 4
    n_rows = (n_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i in range(n_show):
        ax = axes[i]
        
        # Plot temporal component
        component = temporal_components[i]
        ax.plot(target_phases * 100, component, linewidth=2)
        
        # Color and label based on specificity
        if i in specific_components:
            color = 'red'
            comp_type = 'Specific'
            primary_isi = component_specificity[i]['primary_isi']
            ax.set_facecolor((1.0, 0.9, 0.9))  # Light red background
            title_suffix = f"\nPrimary ISI: {primary_isi}ms"
        elif i in moderate_components:
            color = 'orange'
            comp_type = 'Moderate'
            ax.set_facecolor((1.0, 0.95, 0.9))  # Light orange background
            title_suffix = ""
        elif i in general_components:
            color = 'blue'
            comp_type = 'General'
            ax.set_facecolor((0.9, 0.9, 1.0))  # Light blue background
            title_suffix = ""
        else:
            color = 'gray'
            comp_type = 'Unused'
            title_suffix = ""
        
        usage_count = component_specificity.get(i, {}).get('total_usage', 0)
        
        ax.set_title(f'Component {i} ({comp_type})\nVar: {explained_variance[i]:.3f}, Usage: {usage_count}{title_suffix}',
                    color=color, fontweight='bold')
        ax.set_xlabel('ISI Phase (%)')
        ax.set_ylabel('Component Weight')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.axvline(0, color='purple', linestyle='--', alpha=0.7)
        ax.axvline(100, color='purple', linestyle='--', alpha=0.7)
    
    # Hide unused subplots
    for i in range(n_show, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Comprehensive Component Analysis\n'
                f'Total: {len(temporal_components)} components | '
                f'Specific: {len(specific_components)} | '
                f'Moderate: {len(moderate_components)} | '
                f'General: {len(general_components)}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Also create a summary heatmap
    fig2, ax2 = plt.subplots(1, 1, figsize=(15, 8))
    
    # Create ISI × Component usage matrix
    unique_isis = mapping_results['unique_isis']
    usage_matrix = np.zeros((len(unique_isis), min(20, len(temporal_components))))
    
    for isi_idx, isi in enumerate(unique_isis):
        for comp_idx in range(min(20, len(temporal_components))):
            if comp_idx in component_specificity:
                usage_matrix[isi_idx, comp_idx] = component_specificity[comp_idx]['isi_usage'].get(isi, 0)
    
    im = ax2.imshow(usage_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_xlabel('Component Index')
    ax2.set_ylabel('ISI Condition')
    ax2.set_title('Component Usage by ISI Condition')
    ax2.set_xticks(range(min(20, len(temporal_components))))
    ax2.set_yticks(range(len(unique_isis)))
    ax2.set_yticklabels([f'{isi:.0f}ms' for isi in unique_isis])
    plt.colorbar(im, ax=ax2, label='Usage Count')
    
    plt.tight_layout()
    plt.show()













































def find_isi_temporal_components_advanced(isi_phase_data: Dict[str, Any],
                                        n_components: int = 20,
                                        use_ica: bool = True,
                                        use_nmf: bool = True,
                                        frequency_analysis: bool = True) -> Dict[str, Any]:
    """
    Find ISI temporal components using multiple decomposition methods
    ICA and NMF are better for sparse, clock-like temporal patterns
    """
    print(f"\n=== ADVANCED TEMPORAL COMPONENT DISCOVERY ===")
    
    isi_array = isi_phase_data['isi_phase_data']  # (trials, rois, phases)
    trial_metadata = isi_phase_data['trial_metadata']
    target_phases = isi_phase_data['target_phases']
    
    n_trials, n_rois, n_phases = isi_array.shape
    print(f"Analyzing {n_trials} trials × {n_rois} ROIs × {n_phases} phase samples")
    
    # Group trials by ISI
    unique_isis = sorted(list(set([meta['isi_ms'] for meta in trial_metadata])))
    print(f"ISI conditions: {unique_isis}")
    
    # Collect patterns separated by ISI
    isi_patterns = {}
    for isi_ms in unique_isis:
        isi_trials = [i for i, meta in enumerate(trial_metadata) if meta['isi_ms'] == isi_ms]
        isi_data = isi_array[isi_trials]  # (isi_trials, rois, phases)
        
        # Reshape to (isi_trials * rois, phases)
        isi_patterns[isi_ms] = isi_data.reshape(-1, n_phases)
        print(f"  ISI {isi_ms}ms: {len(isi_trials)} trials → {isi_patterns[isi_ms].shape[0]} patterns")
    
    # Combine all patterns for decomposition
    all_patterns = np.vstack([patterns for patterns in isi_patterns.values()])
    valid_mask = ~np.any(np.isnan(all_patterns), axis=1)
    all_patterns_clean = all_patterns[valid_mask]
    
    print(f"Total valid patterns: {all_patterns_clean.shape[0]}")
    
    results = {
        'target_phases': target_phases,
        'unique_isis': unique_isis,
        'isi_patterns': isi_patterns,
        'all_patterns': all_patterns_clean,
        'n_components': n_components
    }
    
    # Method 1: ICA - Good for finding independent temporal sources
    if use_ica:
        print("\n--- Running ICA ---")
        try:
            from sklearn.decomposition import FastICA
            
            ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
            ica_sources = ica.fit_transform(all_patterns_clean)  # (patterns, components)
            ica_components = ica.components_  # (components, phases)
            
            print(f"ICA converged in {ica.n_iter_} iterations")
            
            results['ica'] = {
                'model': ica,
                'sources': ica_sources,
                'components': ica_components,
                'mixing_matrix': ica.mixing_
            }
            
        except Exception as e:
            print(f"ICA failed: {e}")
    
    # Method 2: NMF - Good for sparse, non-negative patterns  
    if use_nmf:
        print("\n--- Running NMF ---")
        try:
            from sklearn.decomposition import NMF as sklearn_NMF
            
            # NMF requires non-negative data, so shift and scale
            patterns_shifted = all_patterns_clean - np.min(all_patterns_clean) + 1e-6
            
            nmf = sklearn_NMF(n_components=n_components, random_state=42, max_iter=1000)
            nmf_sources = nmf.fit_transform(patterns_shifted)
            nmf_components = nmf.components_
            
            print(f"NMF reconstruction error: {nmf.reconstruction_err_:.6f}")
            
            results['nmf'] = {
                'model': nmf,
                'sources': nmf_sources,
                'components': nmf_components,
                'patterns_shifted': patterns_shifted
            }
            
        except Exception as e:
            print(f"NMF failed: {e}")
    
    # Method 3: Frequency domain analysis
    if frequency_analysis:
        print("\n--- Running Frequency Analysis ---")
        
        # Calculate dominant frequencies for each ISI condition
        isi_frequencies = {}
        for isi_ms, patterns in isi_patterns.items():
            # Average pattern for this ISI
            mean_pattern = np.nanmean(patterns, axis=0)
            
            # Expected frequency based on ISI duration
            expected_freq = 1000.0 / isi_ms  # Hz (since ISI is in ms)
            phase_duration = isi_ms / 1000.0  # seconds
            sample_rate = n_phases / phase_duration  # samples per second
            
            # FFT analysis
            fft_vals = np.fft.fft(mean_pattern - np.mean(mean_pattern))
            freqs = np.fft.fftfreq(n_phases, 1/sample_rate)
            
            # Find dominant frequencies
            power = np.abs(fft_vals[:n_phases//2])
            freq_pos = freqs[:n_phases//2]
            
            isi_frequencies[isi_ms] = {
                'expected_freq': expected_freq,
                'freqs': freq_pos,
                'power': power,
                'dominant_freq': freq_pos[np.argmax(power[1:]) + 1],  # Skip DC component
                'mean_pattern': mean_pattern
            }
            
            print(f"  ISI {isi_ms}ms: expected {expected_freq:.3f}Hz, dominant {isi_frequencies[isi_ms]['dominant_freq']:.3f}Hz")
        
        results['frequency_analysis'] = isi_frequencies
    
    return results

def analyze_isi_component_specificity_advanced(advanced_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze which components are specific to particular ISI frequencies
    """
    print(f"\n=== ANALYZING ADVANCED COMPONENT SPECIFICITY ===")
    
    unique_isis = advanced_results['unique_isis']
    target_phases = advanced_results['target_phases']
    isi_patterns = advanced_results['isi_patterns']
    
    component_analysis = {}
    
    # Analyze ICA components if available
    if 'ica' in advanced_results:
        print("\n--- ICA Component Analysis ---")
        ica_components = advanced_results['ica']['components']
        ica_sources = advanced_results['ica']['sources']
        
        # Map sources back to ISI conditions
        ica_isi_mapping = {}
        source_idx = 0
        
        for isi_ms in unique_isis:
            n_patterns = isi_patterns[isi_ms].shape[0]
            ica_isi_mapping[isi_ms] = ica_sources[source_idx:source_idx + n_patterns]
            source_idx += n_patterns
        
        # Find components that show ISI-specific activation
        ica_specificity = []
        for comp_idx in range(ica_components.shape[0]):
            comp_specificity = {}
            
            # Calculate activation strength for each ISI
            for isi_ms in unique_isis:
                isi_activations = ica_isi_mapping[isi_ms][:, comp_idx]
                comp_specificity[isi_ms] = {
                    'mean_activation': np.mean(np.abs(isi_activations)),
                    'std_activation': np.std(isi_activations),
                    'max_activation': np.max(np.abs(isi_activations))
                }
            
            # Calculate specificity score (higher = more ISI-specific)
            activations = [comp_specificity[isi]['mean_activation'] for isi in unique_isis]
            specificity_score = np.max(activations) / (np.mean(activations) + 1e-6)
            primary_isi = unique_isis[np.argmax(activations)]
            
            ica_specificity.append({
                'component_idx': comp_idx,
                'specificity_score': specificity_score,
                'primary_isi': primary_isi,
                'isi_activations': comp_specificity,
                'temporal_pattern': ica_components[comp_idx]
            })
        
        # Sort by specificity
        ica_specificity.sort(key=lambda x: x['specificity_score'], reverse=True)
        component_analysis['ica'] = ica_specificity
        
        print(f"Top 5 most ISI-specific ICA components:")
        for i, comp in enumerate(ica_specificity[:5]):
            print(f"  Component {comp['component_idx']}: specificity={comp['specificity_score']:.2f}, primary ISI={comp['primary_isi']}ms")
    
    # Analyze NMF components if available
    if 'nmf' in advanced_results:
        print("\n--- NMF Component Analysis ---")
        nmf_components = advanced_results['nmf']['components']
        nmf_sources = advanced_results['nmf']['sources']
        
        # Similar analysis for NMF
        nmf_isi_mapping = {}
        source_idx = 0
        
        for isi_ms in unique_isis:
            n_patterns = isi_patterns[isi_ms].shape[0]
            nmf_isi_mapping[isi_ms] = nmf_sources[source_idx:source_idx + n_patterns]
            source_idx += n_patterns
        
        nmf_specificity = []
        for comp_idx in range(nmf_components.shape[0]):
            comp_specificity = {}
            
            for isi_ms in unique_isis:
                isi_activations = nmf_isi_mapping[isi_ms][:, comp_idx]
                comp_specificity[isi_ms] = {
                    'mean_activation': np.mean(isi_activations),
                    'std_activation': np.std(isi_activations),
                    'max_activation': np.max(isi_activations)
                }
            
            activations = [comp_specificity[isi]['mean_activation'] for isi in unique_isis]
            specificity_score = np.max(activations) / (np.mean(activations) + 1e-6)
            primary_isi = unique_isis[np.argmax(activations)]
            
            nmf_specificity.append({
                'component_idx': comp_idx,
                'specificity_score': specificity_score,
                'primary_isi': primary_isi,
                'isi_activations': comp_specificity,
                'temporal_pattern': nmf_components[comp_idx]
            })
        
        nmf_specificity.sort(key=lambda x: x['specificity_score'], reverse=True)
        component_analysis['nmf'] = nmf_specificity
        
        print(f"Top 5 most ISI-specific NMF components:")
        for i, comp in enumerate(nmf_specificity[:5]):
            print(f"  Component {comp['component_idx']}: specificity={comp['specificity_score']:.2f}, primary ISI={comp['primary_isi']}ms")
    
    return component_analysis

def visualize_advanced_components(advanced_results: Dict[str, Any],
                                component_analysis: Dict[str, Any],
                                show_top_n: int = 6) -> None:
    """
    Visualize the most ISI-specific components from ICA and NMF
    """
    target_phases = advanced_results['target_phases']
    unique_isis = advanced_results['unique_isis']
    
    # Determine number of methods
    methods = []
    if 'ica' in component_analysis:
        methods.append('ica')
    if 'nmf' in component_analysis:
        methods.append('nmf')
    if 'frequency_analysis' in advanced_results:
        methods.append('frequency')
    
    n_methods = len(methods)
    if n_methods == 0:
        print("No decomposition methods succeeded")
        return
    
    fig, axes = plt.subplots(n_methods, show_top_n, figsize=(4*show_top_n, 4*n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    if show_top_n == 1:
        axes = axes.reshape(-1, 1)
    
    row_idx = 0
    
    # Plot ICA components
    if 'ica' in component_analysis:
        ica_components = component_analysis['ica']
        for col_idx in range(min(show_top_n, len(ica_components))):
            comp = ica_components[col_idx]
            ax = axes[row_idx, col_idx]
            
            ax.plot(target_phases * 100, comp['temporal_pattern'], 'b-', linewidth=2)
            ax.set_title(f"ICA Component {comp['component_idx']}\n"
                        f"Primary: {comp['primary_isi']}ms\n"
                        f"Specificity: {comp['specificity_score']:.2f}")
            ax.set_xlabel('ISI Phase (%)')
            ax.set_ylabel('ICA Weight')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        # Fill remaining subplots
        for col_idx in range(len(ica_components), show_top_n):
            axes[row_idx, col_idx].set_visible(False)
        
        row_idx += 1
    
    # Plot NMF components
    if 'nmf' in component_analysis:
        nmf_components = component_analysis['nmf']
        for col_idx in range(min(show_top_n, len(nmf_components))):
            comp = nmf_components[col_idx]
            ax = axes[row_idx, col_idx]
            
            ax.plot(target_phases * 100, comp['temporal_pattern'], 'r-', linewidth=2)
            ax.set_title(f"NMF Component {comp['component_idx']}\n"
                        f"Primary: {comp['primary_isi']}ms\n"
                        f"Specificity: {comp['specificity_score']:.2f}")
            ax.set_xlabel('ISI Phase (%)')
            ax.set_ylabel('NMF Weight')
            ax.grid(True, alpha=0.3)
        
        for col_idx in range(len(nmf_components), show_top_n):
            axes[row_idx, col_idx].set_visible(False)
        
        row_idx += 1
    
    # Plot frequency analysis
    if 'frequency_analysis' in advanced_results:
        freq_analysis = advanced_results['frequency_analysis']
        isis_to_show = sorted(unique_isis)[:show_top_n]
        
        for col_idx, isi_ms in enumerate(isis_to_show):
            if col_idx >= show_top_n:
                break
                
            ax = axes[row_idx, col_idx]
            freq_data = freq_analysis[isi_ms]
            
            # Plot the mean pattern and its FFT
            ax.plot(target_phases * 100, freq_data['mean_pattern'], 'g-', linewidth=2)
            ax.set_title(f"ISI {isi_ms}ms\n"
                        f"Expected: {freq_data['expected_freq']:.2f}Hz\n"
                        f"Dominant: {freq_data['dominant_freq']:.2f}Hz")
            ax.set_xlabel('ISI Phase (%)')
            ax.set_ylabel('Response')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        for col_idx in range(len(isis_to_show), show_top_n):
            axes[row_idx, col_idx].set_visible(False)
    
    plt.suptitle('Advanced Temporal Component Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()





def create_isi_absolute_time_grid_from_pkl(data: Dict[str, Any],
                                          max_isi_ms: float = 2300.0,
                                          dt_ms: float = 10.0) -> Dict[str, Any]:
    """
    Create ISI data aligned to absolute time grid using pkl data structure
    instead of phase normalization. This preserves temporal structure.
    
    Parameters
    ----------
    data : dict
        Data loaded from pkl with 'df_trials', 'dFF_clean', 'imaging_time', etc.
    max_isi_ms : float
        Maximum ISI duration to cover (e.g., 2300ms)
    dt_ms : float  
        Time bin size in milliseconds
        
    Returns
    -------
    isi_absolute_data : dict
        'isi_array': (trials, rois, time_bins) - NaN where t > trial's ISI
        'time_grid': (time_bins,) - absolute time from F1_off
        'trial_metadata': list of trial info dicts
        'valid_mask': (trials, time_bins) - True where data is valid
    """
    print(f"\n=== CREATING ISI ABSOLUTE TIME GRID FROM PKL DATA ===")
    
    # Get data from pkl structure
    df_trials = data['df_trials']
    dff_clean = data['dFF_clean']  # (n_rois, n_timepoints)
    imaging_time = data['imaging_time']  # Absolute imaging timestamps
    imaging_fs = data['imaging_fs']
    
    n_rois, n_timepoints = dff_clean.shape
    n_trials = len(df_trials)
    
    print(f"Input data: {n_rois} ROIs, {n_timepoints} timepoints, {n_trials} trials")
    print(f"Imaging time range: {imaging_time[0]:.1f} to {imaging_time[-1]:.1f}s")
    print(f"Imaging sampling: {imaging_fs:.1f} Hz")
    print(f"df_trials index range: {df_trials.index.min()} to {df_trials.index.max()}")
    
    # Calculate mean ISI for short/long classification
    all_isis = df_trials['isi'].values
    valid_isis = all_isis[~pd.isna(all_isis)]
    mean_isi = np.mean(valid_isis)
    print(f"ISI statistics: mean={mean_isi:.1f}ms, range={valid_isis.min():.1f}-{valid_isis.max():.1f}ms")
    
    # Create absolute time grid
    time_grid = np.arange(0, max_isi_ms + dt_ms, dt_ms)  # 0 to max_isi_ms
    n_time_bins = len(time_grid)
    
    print(f"Time grid: 0 to {max_isi_ms}ms in {dt_ms}ms steps = {n_time_bins} bins")
    
    # Initialize output array
    isi_array = np.full((n_trials, n_rois, n_time_bins), np.nan, dtype=np.float32)
    valid_mask = np.zeros((n_trials, n_time_bins), dtype=bool)
    trial_metadata = []
    
    # Apply z-scoring per ROI
    dff_zscore = zscore(dff_clean, axis=1)  # z-score across time for each ROI
    print(f"Applied z-score normalization per ROI")
    
    # Track successful trial processing
    processed_trials = 0
    
    # Use enumerate to get consecutive indices for our arrays
    for array_idx, (original_trial_idx, trial) in enumerate(df_trials.iterrows()):
        # Skip if critical ISI events are missing
        if pd.isna(trial['end_flash_1']) or pd.isna(trial['start_flash_2']) or pd.isna(trial['isi']):
            continue
            
        # Calculate ISI period times (absolute timestamps)
        isi_start_abs = trial['trial_start_timestamp'] + trial['end_flash_1']
        isi_end_abs = trial['trial_start_timestamp'] + trial['start_flash_2']
        isi_duration_s = isi_end_abs - isi_start_abs
        isi_duration_ms = isi_duration_s * 1000
        
        # Verify ISI duration matches expected
        expected_isi_ms = trial['isi']  # Already in ms
        if abs(isi_duration_ms - expected_isi_ms) > 100:  # 100ms tolerance
            print(f"    Trial {original_trial_idx}: ISI duration mismatch ({isi_duration_ms:.1f}ms vs {expected_isi_ms:.1f}ms)")
            continue
        
        # Find imaging indices for ISI period
        isi_start_idx = np.argmin(np.abs(imaging_time - isi_start_abs))
        isi_end_idx = np.argmin(np.abs(imaging_time - isi_end_abs))
        
        # Check if we have enough samples in ISI
        isi_n_samples = isi_end_idx - isi_start_idx + 1
        if isi_n_samples < 3:  # Need at least 3 samples for interpolation
            continue
        
        # Extract ISI segment
        isi_segment = dff_zscore[:, isi_start_idx:isi_end_idx+1]  # (rois, isi_samples)
        
        # Create time vector for this ISI (relative to ISI start)
        isi_imaging_times = imaging_time[isi_start_idx:isi_end_idx+1]
        isi_relative_times = (isi_imaging_times - isi_start_abs) * 1000  # Convert to ms
        
        # Find valid time range in our grid
        valid_time_mask = time_grid <= isi_duration_ms
        valid_indices = np.where(valid_time_mask)[0]
        
        if len(valid_indices) == 0:
            continue
        
        # Interpolate each ROI's data onto the grid
        from scipy.interpolate import interp1d
        
        for roi_idx in range(n_rois):
            roi_trace = isi_segment[roi_idx]
            
            # Skip if all NaN
            if np.all(np.isnan(roi_trace)):
                continue
            
            # Handle NaN values
            valid_mask_roi = np.isfinite(roi_trace)
            if np.sum(valid_mask_roi) < 2:
                continue
            
            # Interpolate onto grid (only for valid time range)
            try:
                interp_func = interp1d(isi_relative_times[valid_mask_roi], 
                                     roi_trace[valid_mask_roi], 
                                     kind='linear', bounds_error=False, 
                                     fill_value='extrapolate')
                isi_array[array_idx, roi_idx, valid_indices] = interp_func(time_grid[valid_indices])
            except Exception as e:
                print(f"      Interpolation failed for trial {original_trial_idx}, ROI {roi_idx}: {e}")
                continue
        
        # Mark valid time bins for this trial (using array_idx, not original_trial_idx)
        valid_mask[array_idx, valid_indices] = True
        
        # Derive is_short from ISI value
        is_short = expected_isi_ms < mean_isi
        
        # Store trial metadata
        trial_metadata.append({
            'original_trial_idx': original_trial_idx,  # Keep track of original index
            'array_idx': array_idx,  # Index in our arrays
            'isi_ms': expected_isi_ms,
            'is_short': is_short,  # Derived from ISI < mean_ISI
            'isi_duration_s': isi_duration_s,
            'is_right_choice': trial.get('is_right_choice', np.nan),
            'is_right': trial.get('is_right', np.nan),
            'rewarded': trial.get('rewarded', False),
            'trial_side': trial.get('trial_side', np.nan),
            'isi_start_abs': isi_start_abs,
            'isi_end_abs': isi_end_abs,
            'isi_start_idx': isi_start_idx,
            'isi_end_idx': isi_end_idx,
            'original_isi_samples': isi_n_samples
        })
        
        processed_trials += 1
    
    # Trim arrays to only include processed trials
    isi_array = isi_array[:processed_trials]
    valid_mask = valid_mask[:processed_trials]
    
    # Calculate statistics
    total_samples = processed_trials * n_rois * n_time_bins
    valid_samples = np.sum(np.isfinite(isi_array))
    nan_fraction = 1.0 - (valid_samples / total_samples) if total_samples > 0 else 1.0
    
    print(f"Successfully processed: {processed_trials} trials out of {n_trials} total")
    print(f"Valid samples: {valid_samples}/{total_samples} ({100*(1-nan_fraction):.1f}%)")
    print(f"NaN fraction: {nan_fraction:.3f}")
    
    # Show ISI distribution
    isi_values = [meta['isi_ms'] for meta in trial_metadata]
    unique_isis = np.unique(np.round(isi_values, 0))
    short_isis = [isi for isi in unique_isis if isi < mean_isi]
    long_isis = [isi for isi in unique_isis if isi >= mean_isi]
    
    print(f"Unique ISIs: {unique_isis}")
    print(f"Short ISIs (<{mean_isi:.1f}ms): {short_isis}")
    print(f"Long ISIs (>={mean_isi:.1f}ms): {long_isis}")
    
    return {
        'isi_array': isi_array,
        'time_grid': time_grid,  # in milliseconds
        'trial_metadata': trial_metadata,
        'valid_mask': valid_mask,
        'dt_ms': dt_ms,
        'max_isi_ms': max_isi_ms,
        'nan_fraction': nan_fraction,
        'n_trials': processed_trials,  # Use actual processed count
        'n_rois': n_rois,
        'n_time_bins': n_time_bins,
        'mean_isi': mean_isi,  # Store for reference
        'short_isis': short_isis,
        'long_isis': long_isis
    }












def run_isi_absolute_time_analysis(isi_absolute_data: Dict[str, Any],
                                  n_components: int = 20,
                                  use_ica: bool = True,
                                  use_nmf: bool = True,
                                  min_trial_coverage: float = 0.3) -> Dict[str, Any]:
    """
    Run component analysis on absolute time ISI grid.
    
    Parameters
    ----------
    isi_absolute_data : dict
        Output from create_isi_absolute_time_grid
    n_components : int
        Number of components to extract
    min_trial_coverage : float
        Minimum fraction of trials that must have data at each time point
    """
    print(f"\n=== ISI ABSOLUTE TIME COMPONENT ANALYSIS ===")
    
    isi_array = isi_absolute_data['isi_array']
    time_grid = isi_absolute_data['time_grid']
    valid_mask = isi_absolute_data['valid_mask']
    trial_metadata = isi_absolute_data['trial_metadata']
    
    n_trials, n_rois, n_time_bins = isi_array.shape
    
    # Filter time bins by coverage
    trial_coverage = np.sum(valid_mask, axis=0) / n_trials
    good_time_mask = trial_coverage >= min_trial_coverage
    good_time_indices = np.where(good_time_mask)[0]
    
    print(f"Time bins with >{min_trial_coverage*100:.0f}% trial coverage: {len(good_time_indices)}/{n_time_bins}")
    
    if len(good_time_indices) < 10:
        print("Too few time bins with sufficient coverage!")
        return {}
    
    # Extract data for good time bins
    data_subset = isi_array[:, :, good_time_indices]  # (trials, rois, good_times)
    time_subset = time_grid[good_time_indices]
    valid_subset = valid_mask[:, good_time_indices]
    
    # Reshape for component analysis: (samples, time)
    # Each sample is one trial-ROI combination
    patterns = []
    pattern_metadata = []
    
    for trial_idx in range(n_trials):
        trial_info = trial_metadata[trial_idx]
        for roi_idx in range(n_rois):
            pattern = data_subset[trial_idx, roi_idx, :]
            valid_pattern = valid_subset[trial_idx, :]
            
            # Only include if enough valid time points
            if np.sum(valid_pattern) >= len(good_time_indices) * 0.5:
                # Mask out invalid time points with NaN
                masked_pattern = pattern.copy()
                masked_pattern[~valid_pattern[good_time_indices]] = np.nan
                
                patterns.append(masked_pattern)
                pattern_metadata.append({
                    'trial_idx': trial_idx,
                    'roi_idx': roi_idx,
                    'isi_ms': trial_info['isi_ms'],
                    'is_short': trial_info['is_short'],
                    'pattern_id': len(patterns) - 1
                })
    
    if len(patterns) == 0:
        print("No valid patterns found!")
        return {}
        
    pattern_matrix = np.array(patterns)  # (n_patterns, n_good_times)
    print(f"Pattern matrix: {pattern_matrix.shape}")
    
    results = {
        'time_subset': time_subset,
        'pattern_matrix': pattern_matrix,
        'pattern_metadata': pattern_metadata,
        'good_time_indices': good_time_indices,
        'trial_coverage': trial_coverage
    }
    
    # ICA Analysis
    if use_ica:
        print("\n--- Running ICA on absolute time data ---")
        try:
            from sklearn.decomposition import FastICA
            
            # Remove NaN patterns for ICA
            finite_mask = ~np.any(np.isnan(pattern_matrix), axis=1)
            clean_patterns = pattern_matrix[finite_mask]
            
            if len(clean_patterns) < n_components:
                print(f"Warning: Only {len(clean_patterns)} clean patterns, reducing components")
                n_components_ica = min(n_components, len(clean_patterns) - 1)
            else:
                n_components_ica = n_components
            
            ica = FastICA(n_components=n_components_ica, random_state=42, max_iter=1000)
            ica_sources = ica.fit_transform(clean_patterns)
            ica_components = ica.components_  # (components, time)
            
            print(f"ICA converged in {ica.n_iter_} iterations")
            
            results['ica'] = {
                'model': ica,
                'sources': ica_sources,
                'components': ica_components,
                'clean_mask': finite_mask
            }
            
        except Exception as e:
            print(f"ICA failed: {e}")
    
    # NMF Analysis  
    if use_nmf:
        print("\n--- Running NMF on absolute time data ---")
        try:
            from sklearn.decomposition import NMF as sklearn_NMF
            
            # NMF requires non-negative data
            finite_mask = ~np.any(np.isnan(pattern_matrix), axis=1)
            clean_patterns = pattern_matrix[finite_mask]
            
            # Shift to make non-negative
            patterns_shifted = clean_patterns - np.min(clean_patterns) + 1e-6
            
            if len(patterns_shifted) < n_components:
                n_components_nmf = min(n_components, len(patterns_shifted) - 1)
            else:
                n_components_nmf = n_components
            
            nmf = sklearn_NMF(n_components=n_components_nmf, random_state=42, max_iter=1000)
            nmf_sources = nmf.fit_transform(patterns_shifted)
            nmf_components = nmf.components_
            
            print(f"NMF reconstruction error: {nmf.reconstruction_err_:.6f}")
            
            results['nmf'] = {
                'model': nmf,
                'sources': nmf_sources,
                'components': nmf_components,
                'patterns_shifted': patterns_shifted,
                'clean_mask': finite_mask
            }
            
        except Exception as e:
            print(f"NMF failed: {e}")
    
    return results

def visualize_isi_absolute_components(isi_absolute_data: Dict[str, Any],
                                    analysis_results: Dict[str, Any],
                                    max_components_show: int = 12) -> None:
    """
    Visualize components from absolute time ISI analysis
    """
    time_subset = analysis_results['time_subset']
    trial_coverage = analysis_results['trial_coverage']
    good_time_indices = analysis_results['good_time_indices']
    
    methods = []
    if 'ica' in analysis_results:
        methods.append('ica')
    if 'nmf' in analysis_results:
        methods.append('nmf')
    
    if len(methods) == 0:
        print("No analysis results to visualize")
        return
    
    # First show trial coverage
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(isi_absolute_data['time_grid'], trial_coverage, 'k-', linewidth=2)
    plt.axhline(0.3, color='r', linestyle='--', alpha=0.7, label='30% threshold')
    plt.xlabel('Time from F1_off (ms)')
    plt.ylabel('Fraction of trials with data')
    plt.title('Trial Coverage vs Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Show which time bins were used
    plt.subplot(1, 2, 2)
    used_mask = np.zeros_like(trial_coverage, dtype=bool)
    used_mask[good_time_indices] = True
    plt.plot(isi_absolute_data['time_grid'], used_mask.astype(float), 'g-', linewidth=2)
    plt.xlabel('Time from F1_off (ms)')
    plt.ylabel('Used in analysis')
    plt.title('Time Bins Used in Analysis')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot components
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, max_components_show, 
                           figsize=(3*max_components_show, 4*n_methods))
    
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    if max_components_show == 1:
        axes = axes.reshape(-1, 1)
    
    for method_idx, method in enumerate(methods):
        components = analysis_results[method]['components']
        n_show = min(max_components_show, components.shape[0])
        
        for comp_idx in range(n_show):
            ax = axes[method_idx, comp_idx]
            
            component = components[comp_idx]
            
            if method == 'ica':
                color = 'blue'
                method_name = 'ICA'
            else:
                color = 'red' 
                method_name = 'NMF'
            
            ax.plot(time_subset, component, color=color, linewidth=2)
            ax.set_title(f'{method_name} Component {comp_idx}')
            ax.set_xlabel('Time from F1_off (ms)')
            ax.set_ylabel('Component Weight')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
            
            # Mark ISI durations
            for isi_ms in [200, 325, 450, 575, 700, 1700, 1850, 2000, 2150, 2300]:
                if isi_ms <= time_subset[-1]:
                    ax.axvline(isi_ms, color='purple', linestyle='--', alpha=0.3)
        
        # Hide unused subplots
        for comp_idx in range(n_show, max_components_show):
            axes[method_idx, comp_idx].set_visible(False)
    
    plt.suptitle(f'ISI Absolute Time Component Analysis\nTime grid: 0-{isi_absolute_data["max_isi_ms"]}ms', 
                fontsize=16)
    plt.tight_layout()
    plt.show()























































def create_isi_masked_grid_with_phase(data: Dict[str, Any],
                                     max_isi_ms: float = 2300.0,
                                     dt_ms: float = 10.0,
                                     n_phase_samples: int = 100) -> Dict[str, Any]:
    """
    Create ISI data using the approach that worked before:
    1. Masked grid from F1_off to max_isi_ms with NaN padding for shorter ISIs
    2. Phase-normalized version for comparison
    3. Prevents long trials from overweighting the analysis
    
    This combines absolute time structure with phase normalization
    """
    print(f"\n=== CREATING ISI MASKED GRID WITH PHASE COMPARISON ===")
    
    df_trials = data['df_trials']
    dff_clean = data['dFF_clean']
    imaging_time = data['imaging_time']
    imaging_fs = data['imaging_fs']
    
    n_rois, n_timepoints = dff_clean.shape
    print(f"Input data: {n_rois} ROIs, {n_timepoints} timepoints")
    
    # Apply z-scoring per ROI
    dff_zscore = zscore(dff_clean, axis=1)
    
    # Create absolute time grid (F1_off to max_isi_ms)
    time_grid = np.arange(0, max_isi_ms + dt_ms, dt_ms)
    n_time_bins = len(time_grid)
    print(f"Absolute time grid: 0 to {max_isi_ms}ms in {dt_ms}ms steps = {n_time_bins} bins")
    
    # Create phase grid (0 to 100% of each trial's ISI)
    phase_grid = np.linspace(0, 1, n_phase_samples)
    print(f"Phase grid: 0 to 100% in {n_phase_samples} samples")
    
    # Initialize arrays
    absolute_array = np.full((len(df_trials), n_rois, n_time_bins), np.nan, dtype=np.float32)
    phase_array = np.full((len(df_trials), n_rois, n_phase_samples), np.nan, dtype=np.float32)
    valid_mask_abs = np.zeros((len(df_trials), n_time_bins), dtype=bool)
    valid_mask_phase = np.zeros((len(df_trials), n_phase_samples), dtype=bool)
    
    trial_metadata = []
    processed_trials = 0
    
    # Calculate ISI statistics
    all_isis = df_trials['isi'].values
    valid_isis = all_isis[~pd.isna(all_isis)]
    mean_isi = np.mean(valid_isis)
    unique_isis = np.unique(np.round(valid_isis, 0))
    
    print(f"ISI statistics: mean={mean_isi:.1f}ms, range={valid_isis.min():.1f}-{valid_isis.max():.1f}ms")
    print(f"Unique ISIs: {unique_isis}")
    
    for trial_idx, (_, trial) in enumerate(df_trials.iterrows()):
        if pd.isna(trial['end_flash_1']) or pd.isna(trial['start_flash_2']) or pd.isna(trial['isi']):
            continue
        
        # Calculate ISI period
        isi_start_abs = trial['trial_start_timestamp'] + trial['end_flash_1']
        isi_end_abs = trial['trial_start_timestamp'] + trial['start_flash_2']
        isi_duration_s = isi_end_abs - isi_start_abs
        isi_duration_ms = isi_duration_s * 1000
        expected_isi_ms = trial['isi']
        
        if abs(isi_duration_ms - expected_isi_ms) > 100:  # 100ms tolerance
            continue
        
        # Find imaging indices for ISI period
        isi_start_idx = np.argmin(np.abs(imaging_time - isi_start_abs))
        isi_end_idx = np.argmin(np.abs(imaging_time - isi_end_abs))
        
        if isi_end_idx - isi_start_idx < 3:
            continue
        
        # Extract ISI segment
        isi_segment = dff_zscore[:, isi_start_idx:isi_end_idx+1]
        isi_imaging_times = imaging_time[isi_start_idx:isi_end_idx+1]
        isi_relative_times = (isi_imaging_times - isi_start_abs) * 1000  # Convert to ms
        
        from scipy.interpolate import interp1d
        
        # === ABSOLUTE TIME GRID (with NaN masking for shorter ISIs) ===
        # Only fill up to this trial's ISI duration, rest stays NaN
        valid_time_mask = time_grid <= isi_duration_ms
        valid_time_indices = np.where(valid_time_mask)[0]
        
        if len(valid_time_indices) > 0:
            for roi_idx in range(n_rois):
                roi_trace = isi_segment[roi_idx]
                valid_mask_roi = np.isfinite(roi_trace)
                
                if np.sum(valid_mask_roi) >= 2:
                    try:
                        interp_func = interp1d(isi_relative_times[valid_mask_roi], 
                                             roi_trace[valid_mask_roi], 
                                             kind='linear', bounds_error=False, 
                                             fill_value='extrapolate')
                        absolute_array[trial_idx, roi_idx, valid_time_indices] = interp_func(time_grid[valid_time_indices])
                    except:
                        continue
            
            valid_mask_abs[trial_idx, valid_time_indices] = True
        
        # === PHASE GRID (normalized to 0-100% of ISI) ===
        # This prevents long trials from dominating
        phase_times = phase_grid * isi_duration_ms  # Convert phase to absolute time for this trial
        
        for roi_idx in range(n_rois):
            roi_trace = isi_segment[roi_idx]
            valid_mask_roi = np.isfinite(roi_trace)
            
            if np.sum(valid_mask_roi) >= 2:
                try:
                    interp_func = interp1d(isi_relative_times[valid_mask_roi], 
                                         roi_trace[valid_mask_roi], 
                                         kind='linear', bounds_error=False, 
                                         fill_value='extrapolate')
                    phase_array[trial_idx, roi_idx, :] = interp_func(phase_times)
                except:
                    continue
        
        valid_mask_phase[trial_idx, :] = True
        
        # Store metadata
        trial_metadata.append({
            'trial_idx': trial_idx,
            'isi_ms': expected_isi_ms,
            'is_short': expected_isi_ms < mean_isi,
            'isi_duration_s': isi_duration_s,
            'is_right_choice': trial.get('is_right_choice', np.nan),
            'rewarded': trial.get('rewarded', False)
        })
        
        processed_trials += 1
    
    # Trim arrays
    absolute_array = absolute_array[:processed_trials]
    phase_array = phase_array[:processed_trials]
    valid_mask_abs = valid_mask_abs[:processed_trials]
    valid_mask_phase = valid_mask_phase[:processed_trials]
    
    print(f"Successfully processed: {processed_trials} trials")
    
    # Calculate coverage statistics
    abs_coverage = np.sum(valid_mask_abs, axis=0) / processed_trials
    phase_coverage = np.sum(valid_mask_phase, axis=0) / processed_trials
    
    print(f"Absolute grid coverage: {np.mean(abs_coverage):.3f} ± {np.std(abs_coverage):.3f}")
    print(f"Phase grid coverage: {np.mean(phase_coverage):.3f} ± {np.std(phase_coverage):.3f}")
    
    return {
        'absolute_array': absolute_array,      # (trials, rois, time_bins) - NaN masked
        'phase_array': phase_array,            # (trials, rois, phase_samples) - phase normalized
        'time_grid': time_grid,                # Absolute time in ms
        'phase_grid': phase_grid,              # Phase from 0 to 1
        'trial_metadata': trial_metadata,
        'valid_mask_abs': valid_mask_abs,
        'valid_mask_phase': valid_mask_phase,
        'abs_coverage': abs_coverage,
        'phase_coverage': phase_coverage,
        'n_trials': processed_trials,
        'n_rois': n_rois,
        'dt_ms': dt_ms,
        'max_isi_ms': max_isi_ms,
        'unique_isis': unique_isis,
        'mean_isi': mean_isi
    }

def run_nmf_on_masked_grid(masked_grid_data: Dict[str, Any],
                          n_components: int = 15,
                          use_absolute: bool = True,
                          use_phase: bool = True,
                          min_coverage: float = 0.3) -> Dict[str, Any]:
    """
    Run NMF on both absolute and phase grids like your previous successful approach
    """
    print(f"\n=== RUNNING NMF ON MASKED GRIDS ===")
    
    results = {}
    
    if use_absolute:
        print("\n--- NMF on Absolute Time Grid ---")
        absolute_array = masked_grid_data['absolute_array']
        time_grid = masked_grid_data['time_grid']
        abs_coverage = masked_grid_data['abs_coverage']
        
        # Filter time bins by coverage
        good_time_mask = abs_coverage >= min_coverage
        good_time_indices = np.where(good_time_mask)[0]
        print(f"Using {len(good_time_indices)}/{len(time_grid)} time bins with >{min_coverage*100:.0f}% coverage")
        
        if len(good_time_indices) > 10:
            # Prepare data for NMF
            data_subset = absolute_array[:, :, good_time_indices]
            
            # Reshape: (trials*rois, time) and remove NaN patterns
            n_trials, n_rois, n_times = data_subset.shape
            patterns = data_subset.reshape(n_trials * n_rois, n_times)
            
            # Remove patterns that are all NaN
            valid_pattern_mask = ~np.all(np.isnan(patterns), axis=1)
            clean_patterns = patterns[valid_pattern_mask]
            
            print(f"Clean patterns: {clean_patterns.shape[0]}/{patterns.shape[0]}")
            
            if clean_patterns.shape[0] > n_components:
                # NMF requires non-negative data
                patterns_shifted = clean_patterns - np.nanmin(clean_patterns) + 1e-6
                
                # Replace remaining NaN with 0 for NMF
                patterns_shifted = np.nan_to_num(patterns_shifted, nan=0.0)
                
                try:
                    from sklearn.decomposition import NMF as sklearn_NMF
                    
                    nmf = sklearn_NMF(n_components=n_components, random_state=42, max_iter=1000)
                    nmf_weights = nmf.fit_transform(patterns_shifted)
                    nmf_components = nmf.components_
                    
                    print(f"NMF converged with reconstruction error: {nmf.reconstruction_err_:.6f}")
                    
                    results['absolute'] = {
                        'nmf_model': nmf,
                        'components': nmf_components,     # (n_components, n_times)
                        'weights': nmf_weights,           # (n_patterns, n_components)
                        'time_subset': time_grid[good_time_indices],
                        'good_time_indices': good_time_indices,
                        'valid_pattern_mask': valid_pattern_mask,
                        'patterns_shifted': patterns_shifted,
                        'reconstruction_error': nmf.reconstruction_err_
                    }
                    
                except Exception as e:
                    print(f"Absolute NMF failed: {e}")
    
    if use_phase:
        print("\n--- NMF on Phase Grid ---")
        phase_array = masked_grid_data['phase_array']
        phase_grid = masked_grid_data['phase_grid']
        
        # Phase data should have good coverage everywhere
        n_trials, n_rois, n_phases = phase_array.shape
        patterns = phase_array.reshape(n_trials * n_rois, n_phases)
        
        # Remove patterns that are all NaN
        valid_pattern_mask = ~np.all(np.isnan(patterns), axis=1)
        clean_patterns = patterns[valid_pattern_mask]
        
        print(f"Clean phase patterns: {clean_patterns.shape[0]}/{patterns.shape[0]}")
        
        if clean_patterns.shape[0] > n_components:
            # NMF requires non-negative data
            patterns_shifted = clean_patterns - np.nanmin(clean_patterns) + 1e-6
            patterns_shifted = np.nan_to_num(patterns_shifted, nan=0.0)
            
            try:
                nmf = sklearn_NMF(n_components=n_components, random_state=42, max_iter=1000)
                nmf_weights = nmf.fit_transform(patterns_shifted)
                nmf_components = nmf.components_
                
                print(f"Phase NMF converged with reconstruction error: {nmf.reconstruction_err_:.6f}")
                
                results['phase'] = {
                    'nmf_model': nmf,
                    'components': nmf_components,
                    'weights': nmf_weights,
                    'phase_grid': phase_grid,
                    'valid_pattern_mask': valid_pattern_mask,
                    'patterns_shifted': patterns_shifted,
                    'reconstruction_error': nmf.reconstruction_err_
                }
                
            except Exception as e:
                print(f"Phase NMF failed: {e}")
    
    return results

def visualize_masked_grid_nmf_results(masked_grid_data: Dict[str, Any],
                                     nmf_results: Dict[str, Any],
                                     max_components: int = 12) -> None:
    """
    Visualize NMF results from both absolute and phase grids
    """
    
    methods = []
    if 'absolute' in nmf_results:
        methods.append('absolute')
    if 'phase' in nmf_results:
        methods.append('phase')
    
    if len(methods) == 0:
        print("No NMF results to visualize")
        return
    
    fig, axes = plt.subplots(len(methods), max_components, 
                           figsize=(3*max_components, 4*len(methods)))
    
    if len(methods) == 1:
        axes = axes.reshape(1, -1)
    if max_components == 1:
        axes = axes.reshape(-1, 1)
    
    for method_idx, method in enumerate(methods):
        result = nmf_results[method]
        components = result['components']
        n_show = min(max_components, components.shape[0])
        
        if method == 'absolute':
            x_data = result['time_subset']
            x_label = 'Time from F1_off (ms)'
            title_prefix = 'Abs'
            color = 'blue'
            
            # Mark ISI boundaries
            unique_isis = masked_grid_data['unique_isis']
            isi_marks = [isi for isi in unique_isis if isi <= x_data[-1]]
        else:
            x_data = result['phase_grid'] * 100  # Convert to percentage
            x_label = 'ISI Phase (%)'
            title_prefix = 'Phase'
            color = 'red'
            isi_marks = []
        
        for comp_idx in range(n_show):
            ax = axes[method_idx, comp_idx]
            
            component = components[comp_idx]
            ax.plot(x_data, component, color=color, linewidth=2)
            ax.set_title(f'{title_prefix} Component {comp_idx}')
            ax.set_xlabel(x_label)
            ax.set_ylabel('NMF Weight')
            ax.grid(True, alpha=0.3)
            
            # Mark ISI boundaries for absolute time
            if method == 'absolute':
                for isi_ms in isi_marks:
                    ax.axvline(isi_ms, color='purple', linestyle='--', alpha=0.4, linewidth=1)
        
        # Hide unused subplots
        for comp_idx in range(n_show, max_components):
            axes[method_idx, comp_idx].set_visible(False)
    
    plt.suptitle('NMF Components: Absolute Time vs Phase Normalized', fontsize=16)
    plt.tight_layout()
    plt.show()







































def analyze_nmf_component_isi_specificity(nmf_results: Dict[str, Any],
                                        masked_grid_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze which ISI durations each NMF component is specifically encoding
    This will help us see if we're finding the meaningful patterns
    """
    print(f"\n=== ANALYZING NMF COMPONENT ISI SPECIFICITY ===")
    
    if 'absolute' not in nmf_results:
        print("No absolute time NMF results to analyze")
        return {}
    
    absolute_data = nmf_results['absolute']
    nmf_components = absolute_data['components']  # (n_components, n_times)
    nmf_weights = absolute_data['weights']        # (n_patterns, n_components)
    time_subset = absolute_data['time_subset']
    valid_pattern_mask = absolute_data['valid_pattern_mask']
    
    # Get original trial metadata
    trial_metadata = masked_grid_data['trial_metadata']
    unique_isis = masked_grid_data['unique_isis']
    
    print(f"Analyzing {nmf_components.shape[0]} components across {len(unique_isis)} ISI conditions")
    
    # Map weights back to trial-ROI combinations
    # The patterns were created as (trials * rois, time), so we need to reconstruct
    n_trials = len(trial_metadata)
    n_rois = masked_grid_data['n_rois']
    
    # Create mapping from pattern index to (trial, roi, isi)
    pattern_to_trial_roi = []
    pattern_idx = 0
    
    for trial_idx in range(n_trials):
        trial_meta = trial_metadata[trial_idx]
        isi_ms = trial_meta['isi_ms']
        
        for roi_idx in range(n_rois):
            if pattern_idx < len(valid_pattern_mask) and valid_pattern_mask[pattern_idx]:
                pattern_to_trial_roi.append({
                    'trial_idx': trial_idx,
                    'roi_idx': roi_idx,
                    'isi_ms': isi_ms,
                    'pattern_idx': pattern_idx
                })
            pattern_idx += 1
    
    print(f"Mapped {len(pattern_to_trial_roi)} valid patterns")
    
    # For each component, calculate ISI-specific activation
    component_isi_specificity = {}
    
    for comp_idx in range(nmf_components.shape[0]):
        comp_weights = nmf_weights[:, comp_idx]  # Weights for this component
        
        # Group by ISI
        isi_activations = {}
        for isi_ms in unique_isis:
            isi_pattern_indices = [i for i, p in enumerate(pattern_to_trial_roi) 
                                 if p['isi_ms'] == isi_ms]
            
            if len(isi_pattern_indices) > 0:
                isi_weights = comp_weights[isi_pattern_indices]
                isi_activations[isi_ms] = {
                    'mean_weight': np.mean(isi_weights),
                    'std_weight': np.std(isi_weights),
                    'max_weight': np.max(isi_weights),
                    'n_patterns': len(isi_pattern_indices)
                }
        
        # Calculate specificity metrics
        mean_activations = [isi_activations[isi]['mean_weight'] for isi in unique_isis 
                          if isi in isi_activations]
        
        if len(mean_activations) > 1:
            specificity_ratio = np.max(mean_activations) / (np.mean(mean_activations) + 1e-6)
            primary_isi = unique_isis[np.argmax(mean_activations)]
            
            component_isi_specificity[comp_idx] = {
                'specificity_ratio': specificity_ratio,
                'primary_isi': primary_isi,
                'isi_activations': isi_activations,
                'temporal_pattern': nmf_components[comp_idx],
                'component_type': 'specific' if specificity_ratio > 2.0 else 'moderate' if specificity_ratio > 1.5 else 'general'
            }
    
    # Sort by specificity
    sorted_components = sorted(component_isi_specificity.items(), 
                             key=lambda x: x[1]['specificity_ratio'], reverse=True)
    
    print(f"\nTop 10 most ISI-specific components:")
    for comp_idx, comp_info in sorted_components[:10]:
        primary_isi = comp_info['primary_isi']
        specificity = comp_info['specificity_ratio']
        comp_type = comp_info['component_type']
        
        print(f"  Component {comp_idx}: {comp_type} for ISI {primary_isi}ms (ratio={specificity:.2f})")
    
    return {
        'component_isi_specificity': component_isi_specificity,
        'sorted_components': sorted_components,
        'time_subset': time_subset,
        'unique_isis': unique_isis
    }





def visualize_top_isi_specific_components(specificity_results: Dict[str, Any],
                                        max_components: int = 6) -> None:
    """
    Visualize the most ISI-specific components found
    """
    if not specificity_results:
        print("No specificity results to visualize")
        return
    
    sorted_components = specificity_results['sorted_components']
    time_subset = specificity_results['time_subset']
    unique_isis = specificity_results['unique_isis']
    
    n_show = min(max_components, len(sorted_components))
    
    fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 8))
    if n_show == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_show):
        comp_idx, comp_info = sorted_components[i]
        
        # Top row: Temporal pattern
        ax_top = axes[0, i]
        temporal_pattern = comp_info['temporal_pattern']
        
        ax_top.plot(time_subset, temporal_pattern, 'r-', linewidth=2)
        ax_top.set_title(f'Component {comp_idx}\nPrimary: {comp_info["primary_isi"]}ms\n'
                        f'Ratio: {comp_info["specificity_ratio"]:.2f}')
        ax_top.set_xlabel('Time from F1_off (ms)')
        ax_top.set_ylabel('NMF Weight')
        ax_top.grid(True, alpha=0.3)
        
        # Mark the primary ISI duration
        primary_isi = float(comp_info['primary_isi'])  # Convert to float
        if primary_isi <= time_subset[-1]:
            ax_top.axvline(primary_isi, color='blue', linestyle='--', linewidth=2, 
                          label=f'Primary ISI: {primary_isi}ms')
            ax_top.legend()
        
        # Mark other ISI durations
        for isi_ms in unique_isis:
            isi_float = float(isi_ms)  # Convert to float for comparison
            if isi_float != primary_isi and isi_float <= time_subset[-1]:
                ax_top.axvline(isi_float, color='gray', linestyle='--', alpha=0.3)
        
        # Bottom row: ISI activation strength
        ax_bottom = axes[1, i]
        isi_activations = comp_info['isi_activations']
        
        # Convert keys to float for consistent comparison
        isis = [float(isi) for isi in isi_activations.keys()]
        mean_weights = [isi_activations[isi]['mean_weight'] for isi in isi_activations.keys()]
        std_weights = [isi_activations[isi]['std_weight'] for isi in isi_activations.keys()]
        
        # Color bars by ISI type
        colors = ['blue' if isi <= 700 else 'red' for isi in isis]
        
        bars = ax_bottom.bar(range(len(isis)), mean_weights, yerr=std_weights, 
                           color=colors, alpha=0.7, capsize=3)
        
        # Highlight the primary ISI - find index by value comparison
        try:
            primary_idx = None
            for idx, isi in enumerate(isis):
                if abs(isi - primary_isi) < 0.1:  # Use small tolerance for float comparison
                    primary_idx = idx
                    break
            
            if primary_idx is not None:
                bars[primary_idx].set_edgecolor('black')
                bars[primary_idx].set_linewidth(3)
        except Exception as e:
            print(f"Warning: Could not highlight primary ISI {primary_isi}: {e}")
        
        ax_bottom.set_title(f'Activation by ISI')
        ax_bottom.set_xlabel('ISI Condition')
        ax_bottom.set_ylabel('Mean NMF Weight')
        ax_bottom.set_xticks(range(len(isis)))
        ax_bottom.set_xticklabels([f'{isi:.0f}ms' for isi in isis], rotation=45)
        ax_bottom.grid(True, alpha=0.3)
    
    plt.suptitle('Most ISI-Specific NMF Components', fontsize=16)
    plt.tight_layout()
    plt.show()





















def run_isi_phase_analysis_from_data(data: Dict[str, Any], 
                                    n_phase_bins: int = 80,
                                    n_components: int = 12) -> Dict[str, Any]:
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
        df_trials, dff_clean, imaging_time, n_phase_bins
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
                               n_phase_bins: int) -> Tuple[np.ndarray, List[Dict]]:
    """Extract ISI segments and resample to phase bins (0-1)"""
    
    from scipy.interpolate import interp1d
    
    n_rois = dff_clean.shape[0]
    
    # Apply z-scoring per ROI
    dff_zscore = zscore(dff_clean, axis=1)
    
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
        isi_segment = dff_zscore[:, isi_start_idx:isi_end_idx+1]
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

n_trim = 15
data_t = trim_session_trials(data, n_trim_start=n_trim, n_trim_end=n_trim)


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
# Discover temporal patterns using PCA
pattern_results = discover_temporal_patterns(stack_data_interp, cfg, n_patterns=5)

# Visualize the discovered patterns
visualize_discovered_patterns(pattern_results)


# %%
# Test improved pattern discovery with adjustable thresholds
n_patterns = 5
pattern_results_improved = discover_temporal_patterns_improved(
    stack_data_interp, cfg, 
    n_patterns=n_patterns, 
    loading_threshold=0.5  # Try different values: 0.3, 0.5, 1.0
)

# Visualize the discovered patterns
visualize_discovered_patterns(pattern_results_improved)

# Look at loading distribution for the main pattern
visualize_pattern_loadings(pattern_results_improved, pattern_idx=0)


# %%
for pattern_idx in range(n_patterns):
    # Look at loading distribution for the main pattern
    visualize_pattern_loadings(pattern_results_improved, pattern_idx=pattern_idx)
    
 
# %%
# Test anticipatory vs response pattern discovery
anticipatory_response_results = discover_anticipatory_vs_response_patterns(stack_data_interp, cfg)

# Visualize the comparison
visualize_anticipatory_vs_response(anticipatory_response_results, loading_threshold=0.5)

# %%
# Analyze ROI overlap
analyze_roi_overlap(anticipatory_response_results)










# %%
# Filter trials by component 1 activity (the one with temporal dynamics)
component_filter_results = filter_trials_by_component_activity(
    stack_data_interp, 
    pattern_results_improved, 
    component_idx=0,  # Component 1 showed the most temporal dynamics
    activity_threshold=1.0,  # 1 std above baseline
    min_roi_fraction=0.3     # 30% of component ROIs must be active
)

# Visualize the filtering results
visualize_component_trial_filtering(stack_data_interp, component_filter_results)



# %%
# Visualize with z-scored data (default)
visualize_component_trial_filtering(stack_data_interp, component_filter_results, use_zscore=True)

# %%
# Visualize with original dF/F data
visualize_component_trial_filtering(stack_data_interp, component_filter_results, 
                                   data=data, use_zscore=False)





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

# Add dPCA config option
cfg['dpca'] = {
    'use_zscore': False,  # Set to True for z-scored dF/F, False for raw dF/F
    'n_phase_samples': 100,
    'min_trials_per_condition': 0  # Start with 0 to see all counts
}

# Check condition distribution first
condition_dist = check_dpca_condition_distribution(data)

# %%

# Add dPCA config option
cfg['dpca'] = {
    'use_zscore': False,  # Set to True for z-scored dF/F, False for raw dF/F
    'n_phase_samples': 100,
    'min_trials_per_condition': 0  # Start with 0 to see all counts
}
# Extract ISI phase-normalized data
isi_phase_data = extract_isi_phase_normalized(
    data, cfg, 
    n_phase_samples=cfg['dpca']['n_phase_samples'],
    use_zscore=cfg['dpca']['use_zscore']
)

# %%
# Run dPCA on ISI phases
dpca_results = run_dpca_on_isi_phases(
    isi_phase_data, 
    min_trials_per_condition=cfg['dpca']['min_trials_per_condition']
)

# %%
# Visualize results
if dpca_results is not None:
    visualize_isi_dpca_results(dpca_results, isi_phase_data, max_components=3)
    
    
    
    
    
    
# %%
# Update config for simplified ISI-only analysis
cfg['dpca'] = {
    'use_zscore': False,  # Start with raw dF/F
    'n_phase_samples': 100,
    'min_trials_per_condition': 5  # Require at least 5 trials per ISI condition
}

# Run ISI-only analysis
isi_dpca_results = run_isi_only_dpca(isi_phase_data)


# Visualize ISI-only results
if isi_dpca_results is not None:
    visualize_isi_only_dpca(isi_dpca_results, isi_phase_data, max_components=3)






# %%

# Test the no-regularization version
isi_dpca_results_no_reg = run_isi_only_dpca_no_reg(isi_phase_data)

if isi_dpca_results_no_reg is not None:
    print(f"Method used: {isi_dpca_results_no_reg['method']}")
    visualize_isi_only_dpca(isi_dpca_results_no_reg, isi_phase_data, max_components=3)






# %%
# Update config for simplified ISI-only analysis
cfg['dpca'] = {
    'use_zscore': False,  # Start with raw dF/F
    'n_phase_samples': 100,
    'min_trials_per_condition': 5  # Require at least 5 trials per ISI condition
}
# Try with z-scored data instead of raw dF/F
cfg['dpca']['use_zscore'] = True

# Extract ISI phase-normalized data with z-scoring
isi_phase_data_zscore = extract_isi_phase_normalized(
    data, cfg, 
    n_phase_samples=cfg['dpca']['n_phase_samples'],
    use_zscore=cfg['dpca']['use_zscore']
)

# Run ISI-only analysis with z-scored data
isi_dpca_results_zscore = run_isi_only_dpca(isi_phase_data_zscore)

if isi_dpca_results_zscore is not None:
    visualize_isi_only_dpca(isi_dpca_results_zscore, isi_phase_data_zscore, max_components=3)
    
    
# %%

# Test the no-regularization version
isi_dpca_results_zscore_no_reg = run_isi_only_dpca_no_reg(isi_phase_data_zscore)

if isi_dpca_results_no_reg is not None:
    print(f"Method used: {isi_dpca_results_zscore_no_reg['method']}")
    visualize_isi_only_dpca(isi_dpca_results_zscore_no_reg, isi_phase_data_zscore, max_components=3)



# %%
# Plot components 1 & 2 in time domain
plot_isi_components_in_time_domain(isi_dpca_results_zscore_no_reg, isi_phase_data_zscore, component_idx=0)
plot_isi_components_in_time_domain(isi_dpca_results_zscore_no_reg, isi_phase_data_zscore, component_idx=1)
plot_isi_components_in_time_domain(isi_dpca_results_zscore_no_reg, isi_phase_data_zscore, component_idx=2)


# %%
# Plot components 1 & 2 in time domain
plot_isi_components_in_time_domain(isi_dpca_results_no_reg, isi_phase_data, component_idx=0)
plot_isi_components_in_time_domain(isi_dpca_results_no_reg, isi_phase_data, component_idx=1)
plot_isi_components_in_time_domain(isi_dpca_results_no_reg, isi_phase_data, component_idx=2)












# %%
# Run fixed t-SNE on ISI phase data
tsne_results_fixed = run_tsne_on_isi_data_fixed(
    isi_phase_data, cfg, 
    use_full_phase=True,
    tsne_params={
        'perplexity': 30,
        'learning_rate': 200, 
        'max_iter': 1000,  # Correct parameter name
        'random_state': 42
    }
)

# Visualize results
if tsne_results_fixed is not None:
    visualize_tsne_results(tsne_results_fixed)









# %%
# Try with summary features instead of full phase data
tsne_results_summary_fixed = run_tsne_on_isi_data_fixed(
    isi_phase_data, cfg,
    use_full_phase=False,  # Use summary features
    tsne_params={
        'perplexity': 30,
        'learning_rate': 200,
        'max_iter': 1000,  # Correct parameter name
        'random_state': 42
    }
)

if tsne_results_summary_fixed is not None:
    visualize_tsne_results(tsne_results_summary_fixed)
    
    
    
    
# %%
# Try with z-scored data
tsne_results_zscore_fixed = run_tsne_on_isi_data_fixed(
    isi_phase_data_zscore, cfg,
    use_full_phase=True,
    tsne_params={
        'perplexity': 30,
        'learning_rate': 200,
        'max_iter': 1000,  # Correct parameter name
        'random_state': 42
    }
)

if tsne_results_zscore_fixed is not None:
    visualize_tsne_results(tsne_results_zscore_fixed)







# %%
# Try with z-scored data
tsne_results_zscore_summary_fixed = run_tsne_on_isi_data_fixed(
    isi_phase_data_zscore, cfg,
    use_full_phase=False,
    tsne_params={
        'perplexity': 30,
        'learning_rate': 200,
        'max_iter': 1000,  # Correct parameter name
        'random_state': 42
    }
)

if tsne_results_zscore_summary_fixed is not None:
    visualize_tsne_results(tsne_results_zscore_summary_fixed)
























# %%
# First, let's identify ISI-responsive ROIs using variance-based method
print("=== FINDING ISI-RESPONSIVE ROIs ===")

# Test with different variance thresholds to see what we get
variance_thresholds = [0.1, 0.5, 1.0, 2.0]

for threshold in variance_thresholds:
    isi_responsive_rois = find_isi_responsive_rois(
        isi_phase_data_zscore['isi_phase_data'], 
        isi_phase_data_zscore['trial_metadata'],
        variance_threshold=threshold
    )
    print(f"  Threshold {threshold}: {len(isi_responsive_rois)} ROIs ({100*len(isi_responsive_rois)/757:.1f}%)")

# Choose a reasonable threshold (e.g., 0.5) for analysis
selected_threshold = 0.1
isi_responsive_rois = find_isi_responsive_rois(
    isi_phase_data_zscore['isi_phase_data'], 
    isi_phase_data_zscore['trial_metadata'],
    variance_threshold=selected_threshold
)

print(f"\nSelected {len(isi_responsive_rois)} ISI-responsive ROIs with threshold {selected_threshold}")



# %%
# First, let's identify ISI-responsive ROIs using variance-based method
print("=== FINDING ISI-RESPONSIVE ROIs ===")

# Test with different variance thresholds to see what we get
variance_thresholds = [0.1, 0.5, 1.0, 2.0]

for threshold in variance_thresholds:
    isi_responsive_rois = find_isi_responsive_rois(
        isi_phase_data['isi_phase_data'], 
        isi_phase_data['trial_metadata'],
        variance_threshold=threshold
    )
    print(f"  Threshold {threshold}: {len(isi_responsive_rois)} ROIs ({100*len(isi_responsive_rois)/757:.1f}%)")

# Choose a reasonable threshold (e.g., 0.5) for analysis
selected_threshold = 2.0
isi_responsive_rois = find_isi_responsive_rois(
    isi_phase_data['isi_phase_data'], 
    isi_phase_data['trial_metadata'],
    variance_threshold=selected_threshold
)

print(f"\nSelected {len(isi_responsive_rois)} ISI-responsive ROIs with threshold {selected_threshold}")




# %%
# Run t-SNE on ISI-responsive ROIs only
tsne_results_isi_rois = run_tsne_on_isi_responsive_rois(
    isi_phase_data,
    roi_selection_method='isi_responsive',
    cfg={'variance_threshold': selected_threshold}
)

if tsne_results_isi_rois is not None:
    print("t-SNE on ISI-responsive ROIs successful!")
    visualize_tsne_results(tsne_results_isi_rois)
else:
    print("t-SNE on ISI-responsive ROIs failed")
    
  

# %%
# Also try with a more restrictive threshold to get fewer, more selective ROIs
restrictive_threshold = 2.0
isi_responsive_rois_restrictive = find_isi_responsive_rois(
    isi_phase_data['isi_phase_data'], 
    isi_phase_data['trial_metadata'],
    variance_threshold=restrictive_threshold
)

print(f"Restrictive selection: {len(isi_responsive_rois_restrictive)} ROIs")

if len(isi_responsive_rois_restrictive) > 10:  # Need sufficient ROIs
    # Create subset data manually
    isi_array_restrictive = isi_phase_data['isi_phase_data'][:, isi_responsive_rois_restrictive, :]
    
    tsne_results_restrictive = run_tsne_on_isi_data_fixed(
        {
            'isi_phase_data': isi_array_restrictive,
            'trial_metadata': isi_phase_data['trial_metadata'],
            'target_phases': isi_phase_data['target_phases'],
            'data_type': f"z-scored dF/F (highly ISI-responsive ROIs, n={len(isi_responsive_rois_restrictive)})"
        },
        cfg, use_full_phase=True
    )
    
    if tsne_results_restrictive is not None:
        visualize_tsne_results(tsne_results_restrictive)
else:
    print("Too few highly responsive ROIs for analysis")



# %%
# Try choice-matched analysis to control for behavioral confounds
tsne_results_choice_matched = run_tsne_choice_matched(isi_phase_data, cfg)

if tsne_results_choice_matched is not None:
    print("Choice-matched t-SNE successful!")
    visualize_tsne_results(tsne_results_choice_matched)
else:
    print("Insufficient choice-matched trials")
    
    
    
    
# %%
# Let's also examine the ISI-responsive ROIs more closely
print("=== ANALYZING ISI-RESPONSIVE ROI PATTERNS ===")

isi_array = isi_phase_data['isi_phase_data']
trial_metadata = isi_phase_data['trial_metadata']
target_phases = isi_phase_data['target_phases']

# Group trials by ISI
unique_isis = sorted(list(set([meta['isi_ms'] for meta in trial_metadata])))
print(f"Unique ISIs: {unique_isis}")

# Calculate mean response for each ISI condition for responsive ROIs
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for plot_idx, roi_idx in enumerate(isi_responsive_rois[:6]):  # Show first 6 responsive ROIs
    if plot_idx >= 6:
        break
        
    ax = axes[plot_idx]
    
    for isi in unique_isis:
        isi_trials = [i for i, meta in enumerate(trial_metadata) if meta['isi_ms'] == isi]
        if len(isi_trials) > 0:
            roi_responses = isi_array[isi_trials, roi_idx, :]  # (trials, phases)
            mean_response = np.mean(roi_responses, axis=0)
            
            color = 'blue' if isi <= 700 else 'red'
            ax.plot(target_phases * 100, mean_response, color=color, alpha=0.7, 
                   linewidth=2, label=f'ISI {isi:.0f}ms')
    
    ax.set_title(f'ISI-Responsive ROI {roi_idx}')
    ax.set_xlabel('ISI Phase (%)')
    ax.set_ylabel('z-scored dF/F')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)

plt.suptitle('ISI-Responsive ROI Patterns', fontsize=16)
plt.tight_layout()
plt.show()



# %%
# Compare population averages: all ROIs vs ISI-responsive ROIs
print("=== COMPARING POPULATION RESPONSES ===")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: All ROIs
for isi in unique_isis:
    isi_trials = [i for i, meta in enumerate(trial_metadata) if meta['isi_ms'] == isi]
    if len(isi_trials) > 0:
        # Average across all ROIs and trials for this ISI
        isi_responses = isi_array[isi_trials, :, :]  # (trials, all_rois, phases)
        pop_mean = np.mean(isi_responses, axis=(0, 1))  # Mean across trials and ROIs
        
        color = 'blue' if isi <= 700 else 'red'
        axes[0].plot(target_phases * 100, pop_mean, color=color, alpha=0.7,
                    linewidth=2, label=f'ISI {isi:.0f}ms')

axes[0].set_title('Population Response: All ROIs')
axes[0].set_xlabel('ISI Phase (%)')
axes[0].set_ylabel('z-scored dF/F')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axhline(0, color='gray', linestyle='-', alpha=0.5)

# Right: ISI-responsive ROIs only
for isi in unique_isis:
    isi_trials = [i for i, meta in enumerate(trial_metadata) if meta['isi_ms'] == isi]
    if len(isi_trials) > 0:
        # Average across ISI-responsive ROIs and trials for this ISI
        isi_responses = isi_array[isi_trials][:, isi_responsive_rois, :]  # (trials, responsive_rois, phases)
        pop_mean = np.mean(isi_responses, axis=(0, 1))  # Mean across trials and ROIs
        
        color = 'blue' if isi <= 700 else 'red'
        axes[1].plot(target_phases * 100, pop_mean, color=color, alpha=0.7,
                    linewidth=2, label=f'ISI {isi:.0f}ms')

axes[1].set_title(f'Population Response: ISI-Responsive ROIs (n={len(isi_responsive_rois)})')
axes[1].set_xlabel('ISI Phase (%)')
axes[1].set_ylabel('z-scored dF/F')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axhline(0, color='gray', linestyle='-', alpha=0.5)

plt.tight_layout()
plt.show()

































# %%
# Run t-SNE on ISI-responsive ROIs only
tsne_results_isi_rois = run_tsne_on_isi_responsive_rois(
    isi_phase_data_zscore,
    roi_selection_method='isi_responsive',
    cfg={'variance_threshold': selected_threshold}
)

if tsne_results_isi_rois is not None:
    print("t-SNE on ISI-responsive ROIs successful!")
    visualize_tsne_results(tsne_results_isi_rois)
else:
    print("t-SNE on ISI-responsive ROIs failed")
    
    
    
# %%
# Also try with a more restrictive threshold to get fewer, more selective ROIs
restrictive_threshold = 1.0
isi_responsive_rois_restrictive = find_isi_responsive_rois(
    isi_phase_data_zscore['isi_phase_data'], 
    isi_phase_data_zscore['trial_metadata'],
    variance_threshold=restrictive_threshold
)

print(f"Restrictive selection: {len(isi_responsive_rois_restrictive)} ROIs")

if len(isi_responsive_rois_restrictive) > 10:  # Need sufficient ROIs
    # Create subset data manually
    isi_array_restrictive = isi_phase_data_zscore['isi_phase_data'][:, isi_responsive_rois_restrictive, :]
    
    tsne_results_restrictive = run_tsne_on_isi_data_fixed(
        {
            'isi_phase_data': isi_array_restrictive,
            'trial_metadata': isi_phase_data_zscore['trial_metadata'],
            'target_phases': isi_phase_data_zscore['target_phases'],
            'data_type': f"z-scored dF/F (highly ISI-responsive ROIs, n={len(isi_responsive_rois_restrictive)})"
        },
        cfg, use_full_phase=True
    )
    
    if tsne_results_restrictive is not None:
        visualize_tsne_results(tsne_results_restrictive)
else:
    print("Too few highly responsive ROIs for analysis")
    

# %%
# Try choice-matched analysis to control for behavioral confounds
tsne_results_choice_matched = run_tsne_choice_matched(isi_phase_data_zscore, cfg)

if tsne_results_choice_matched is not None:
    print("Choice-matched t-SNE successful!")
    visualize_tsne_results(tsne_results_choice_matched)
else:
    print("Insufficient choice-matched trials")
    
    
    
    
# %%
# Let's also examine the ISI-responsive ROIs more closely
print("=== ANALYZING ISI-RESPONSIVE ROI PATTERNS ===")

isi_array = isi_phase_data_zscore['isi_phase_data']
trial_metadata = isi_phase_data_zscore['trial_metadata']
target_phases = isi_phase_data_zscore['target_phases']

# Group trials by ISI
unique_isis = sorted(list(set([meta['isi_ms'] for meta in trial_metadata])))
print(f"Unique ISIs: {unique_isis}")

# Calculate mean response for each ISI condition for responsive ROIs
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for plot_idx, roi_idx in enumerate(isi_responsive_rois[:6]):  # Show first 6 responsive ROIs
    if plot_idx >= 6:
        break
        
    ax = axes[plot_idx]
    
    for isi in unique_isis:
        isi_trials = [i for i, meta in enumerate(trial_metadata) if meta['isi_ms'] == isi]
        if len(isi_trials) > 0:
            roi_responses = isi_array[isi_trials, roi_idx, :]  # (trials, phases)
            mean_response = np.mean(roi_responses, axis=0)
            
            color = 'blue' if isi <= 700 else 'red'
            ax.plot(target_phases * 100, mean_response, color=color, alpha=0.7, 
                   linewidth=2, label=f'ISI {isi:.0f}ms')
    
    ax.set_title(f'ISI-Responsive ROI {roi_idx}')
    ax.set_xlabel('ISI Phase (%)')
    ax.set_ylabel('z-scored dF/F')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)

plt.suptitle('ISI-Responsive ROI Patterns', fontsize=16)
plt.tight_layout()
plt.show()



# %%
# Compare population averages: all ROIs vs ISI-responsive ROIs
print("=== COMPARING POPULATION RESPONSES ===")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: All ROIs
for isi in unique_isis:
    isi_trials = [i for i, meta in enumerate(trial_metadata) if meta['isi_ms'] == isi]
    if len(isi_trials) > 0:
        # Average across all ROIs and trials for this ISI
        isi_responses = isi_array[isi_trials, :, :]  # (trials, all_rois, phases)
        pop_mean = np.mean(isi_responses, axis=(0, 1))  # Mean across trials and ROIs
        
        color = 'blue' if isi <= 700 else 'red'
        axes[0].plot(target_phases * 100, pop_mean, color=color, alpha=0.7,
                    linewidth=2, label=f'ISI {isi:.0f}ms')

axes[0].set_title('Population Response: All ROIs')
axes[0].set_xlabel('ISI Phase (%)')
axes[0].set_ylabel('z-scored dF/F')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axhline(0, color='gray', linestyle='-', alpha=0.5)

# Right: ISI-responsive ROIs only
for isi in unique_isis:
    isi_trials = [i for i, meta in enumerate(trial_metadata) if meta['isi_ms'] == isi]
    if len(isi_trials) > 0:
        # Average across ISI-responsive ROIs and trials for this ISI
        isi_responses = isi_array[isi_trials][:, isi_responsive_rois, :]  # (trials, responsive_rois, phases)
        pop_mean = np.mean(isi_responses, axis=(0, 1))  # Mean across trials and ROIs
        
        color = 'blue' if isi <= 700 else 'red'
        axes[1].plot(target_phases * 100, pop_mean, color=color, alpha=0.7,
                    linewidth=2, label=f'ISI {isi:.0f}ms')

axes[1].set_title(f'Population Response: ISI-Responsive ROIs (n={len(isi_responsive_rois)})')
axes[1].set_xlabel('ISI Phase (%)')
axes[1].set_ylabel('z-scored dF/F')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axhline(0, color='gray', linestyle='-', alpha=0.5)

plt.tight_layout()
plt.show()

































# %%
# Run UMAP on full phase data (raw)
umap_results_full = run_umap_on_isi_data(
    isi_phase_data,
    n_neighbors=15,
    min_dist=0.1,
    use_full_phase=True
)

if umap_results_full is not None:
    visualize_umap_results(umap_results_full)
    


# %%
# Run UMAP on summary features (raw)
umap_results_summary = run_umap_on_isi_data(
    isi_phase_data,
    n_neighbors=15,
    min_dist=0.1,
    use_full_phase=False  # Use summary features
)

if umap_results_summary is not None:
    visualize_umap_results(umap_results_summary)
    
    
    
    
    
    
# %%
# Run UMAP on full phase data (z-scored)
umap_results_zscore_full = run_umap_on_isi_data(
    isi_phase_data_zscore,
    n_neighbors=15,
    min_dist=0.1,
    use_full_phase=True
)

if umap_results_zscore_full is not None:
    visualize_umap_results(umap_results_zscore_full)
    




# %%
# Run UMAP on summary features (z-scored)
umap_results_zscore_summary = run_umap_on_isi_data(
    isi_phase_data_zscore,
    n_neighbors=15,
    min_dist=0.1,
    use_full_phase=False
)

if umap_results_zscore_summary is not None:
    visualize_umap_results(umap_results_zscore_summary)







# %%
# Run UMAP on ISI-responsive ROIs (using the ones we identified earlier)
umap_results_isi_rois = run_umap_on_isi_responsive_rois(
    isi_phase_data_zscore,
    isi_responsive_rois,
    n_neighbors=10,  # Smaller since fewer features
    min_dist=0.1
)

if umap_results_isi_rois is not None:
    visualize_umap_results(umap_results_isi_rois)
















# %%
# Try different UMAP parameters to see if we get better separation
print("=== TESTING DIFFERENT UMAP PARAMETERS ===")

parameter_sets = [
    {'n_neighbors': 5, 'min_dist': 0.0},    # Tight clusters
    {'n_neighbors': 15, 'min_dist': 0.1},   # Default
    {'n_neighbors': 30, 'min_dist': 0.3},   # Looser clusters
    {'n_neighbors': 50, 'min_dist': 0.5},   # Very loose
]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, params in enumerate(parameter_sets):
    print(f"\nTesting n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}")
    
    umap_result = run_umap_on_isi_data(
        isi_phase_data_zscore,
        n_neighbors=params['n_neighbors'],
        min_dist=params['min_dist'],
        use_full_phase=True
    )
    
    if umap_result is not None:
        embedding = umap_result['embedding']
        isi_values = umap_result['isi_values']
        choice_values = umap_result['choice_values']
        
        # Top row: Color by ISI
        scatter1 = axes[i].scatter(embedding[:, 0], embedding[:, 1],
                                  c=isi_values, cmap='viridis', s=20, alpha=0.7)
        axes[i].set_title(f'n_neighbors={params["n_neighbors"]}, min_dist={params["min_dist"]}\n(ISI Duration)')
        axes[i].set_xlabel('UMAP 1')
        axes[i].set_ylabel('UMAP 2')
        
        # Bottom row: Color by choice
        for choice, color, label in [(0, 'blue', 'Left'), (1, 'red', 'Right')]:
            mask = choice_values == choice
            if np.any(mask):
                axes[i+4].scatter(embedding[mask, 0], embedding[mask, 1],
                                 c=color, s=20, alpha=0.7, label=label)
        
        axes[i+4].set_title(f'n_neighbors={params["n_neighbors"]}, min_dist={params["min_dist"]}\n(Choice)')
        axes[i+4].set_xlabel('UMAP 1')
        axes[i+4].set_ylabel('UMAP 2')
        axes[i+4].legend()
    else:
        axes[i].text(0.5, 0.5, 'Failed', ha='center', va='center', transform=axes[i].transAxes)
        axes[i+4].text(0.5, 0.5, 'Failed', ha='center', va='center', transform=axes[i+4].transAxes)

plt.suptitle('UMAP Parameter Comparison (z-scored data)', fontsize=16)
plt.tight_layout()
plt.show()





















# %%

# Focus on individual ROI reliability
consistency_results = analyze_isi_consistency_per_roi(isi_phase_data_zscore)
reliable_encoders = find_reliable_isi_encoders(consistency_results, 
                                              consistency_threshold=0.5,
                                              reliability_threshold=0.6)
differential_pairs = find_differential_pairs(consistency_results)


# %%


# Now let's map the encoding patterns for our reliable ROIs
encoding_map = map_roi_isi_encoding_patterns(consistency_results, reliable_encoders)
specialization_analysis = analyze_roi_isi_specialization(encoding_map)


# Visualize the encoding matrix
visualization_results = visualize_roi_isi_encoding_matrix(encoding_map, specialization_analysis, max_rois_show=20)

# Look for temporal switching behavior (the "dancing pairs" hypothesis)
switching_analysis = find_roi_component_switching(encoding_map, isi_phase_data_zscore, time_window_size=50)











# %%
# Run comprehensive pattern analysis on all reliable ROIs
pattern_analysis = analyze_reliable_roi_response_patterns(
    encoding_map, 
    isi_phase_data_zscore,
    use_pca=True,
    use_kmeans=True,
    n_components=4,
    n_clusters=5
)

# Visualize the results
visualize_roi_pattern_analysis(pattern_analysis)

# %%
# Now analyze individual ROIs
reliable_rois = list(encoding_map['roi_encoding_map'].keys())
print(f"Available reliable ROIs: {reliable_rois}")

# Analyze first few reliable ROIs individually
for roi_idx in reliable_rois[:]:  # Look at first 3 ROIs
    roi_analysis = analyze_individual_roi_patterns(
        encoding_map, 
        isi_phase_data_zscore, 
        roi_idx,
        use_pca=True,
        n_components=3
    )
    if roi_analysis is not None:
        visualize_individual_roi_analysis(roi_analysis)
























# %%

# Re-run the analysis with original dF/F data
consistency_results_orig = analyze_isi_consistency_per_roi(isi_phase_data)  # Not isi_phase_data_zscore
reliable_encoders_orig = find_reliable_isi_encoders(consistency_results_orig, 
                                                   consistency_threshold=0.5,
                                                   reliability_threshold=0.6)

# Map patterns with original data
encoding_map = map_roi_isi_encoding_patterns(consistency_results_orig, reliable_encoders_orig)

differential_pairs = find_differential_pairs(consistency_results)


specialization_analysis = analyze_roi_isi_specialization(encoding_map)


# Visualize the encoding matrix
visualization_results = visualize_roi_isi_encoding_matrix(encoding_map, specialization_analysis, max_rois_show=20)

# Look for temporal switching behavior (the "dancing pairs" hypothesis)
switching_analysis = find_roi_component_switching(encoding_map, isi_phase_data, time_window_size=50)


# %%
# Run comprehensive pattern analysis on all reliable ROIs
pattern_analysis = analyze_reliable_roi_response_patterns(
    encoding_map, 
    isi_phase_data,
    use_pca=True,
    use_kmeans=True,
    n_components=4,
    n_clusters=5
)

# Visualize the results
visualize_roi_pattern_analysis(pattern_analysis)

# %%
# Now analyze individual ROIs
reliable_rois = list(encoding_map['roi_encoding_map'].keys())
print(f"Available reliable ROIs: {reliable_rois}")

# Analyze first few reliable ROIs individually
for roi_idx in reliable_rois[:]:  # Look at first 3 ROIs
    roi_analysis = analyze_individual_roi_patterns(
        encoding_map, 
        isi_phase_data, 
        roi_idx,
        use_pca=True,
        n_components=3
    )
    if roi_analysis is not None:
        visualize_individual_roi_analysis(roi_analysis)
























# %%
# Run the trial-level component analysis
component_results = analyze_trial_level_roi_components(
    isi_phase_data, 
    n_components=16,  # Try more components to capture different patterns
    use_trial_groups=True
)

# Analyze ROI switching behavior
switching_analysis = analyze_roi_component_switching(component_results, min_trials_per_window=25)

# Find complementary ROI pairs
pairs_analysis = find_roi_component_pairs(component_results, switching_analysis)

# Visualize the dynamics
visualize_roi_component_dynamics(component_results, switching_analysis, pairs_analysis)




















# %%

# Run the comprehensive analysis
print("=== RUNNING COMPREHENSIVE COMPONENT ANALYSIS ===")

# Step 1: Find ALL components comprehensively
component_results = find_all_isi_components_comprehensive(
    isi_phase_data_zscore,  # Using z-scored data for better component discovery
    n_components=30,        # Higher number to capture sparse patterns
    include_all_rois=True
)

# Step 2: Map ROI-ISI combinations to components
mapping_results = map_roi_isi_to_components(
    component_results,
    loading_threshold=0.1  # Lower threshold to capture more subtle patterns
)

# Step 3: Analyze component specificity
specificity_results = analyze_component_isi_specificity(
    component_results,
    mapping_results
)




# %%
# Step 4: Visualize comprehensive results
visualize_comprehensive_components(
    component_results,
    mapping_results,
    specificity_results,
    max_components_show=30
)































# %%
# Run the advanced analysis
print("=== RUNNING ADVANCED TEMPORAL COMPONENT ANALYSIS ===")

advanced_results = find_isi_temporal_components_advanced(
    isi_phase_data_zscore,
    n_components=20,
    use_ica=True,
    use_nmf=True,
    frequency_analysis=True
)

component_analysis = analyze_isi_component_specificity_advanced(advanced_results)

visualize_advanced_components(advanced_results, component_analysis, show_top_n=6)









# %%
# Run the absolute time analysis using pkl data
print("=== RUNNING ISI ABSOLUTE TIME ANALYSIS (PKL DATA) ===")

# Create absolute time grid using pkl data
isi_absolute_data = create_isi_absolute_time_grid_from_pkl(
    data,
    max_isi_ms=2300.0,
    dt_ms=10.0
)

# Run component analysis on absolute time grid
absolute_results = run_isi_absolute_time_analysis(
    isi_absolute_data,
    n_components=20,
    use_ica=True,
    use_nmf=True,
    min_trial_coverage=0.3
)

# Visualize results
if absolute_results:
    visualize_isi_absolute_components(isi_absolute_data, absolute_results, max_components_show=12)
else:
    print("Absolute time analysis failed - no results to visualize")


# %%
# Run the absolute time analysis using pkl data
print("=== RUNNING ISI ABSOLUTE TIME ANALYSIS (PKL DATA) ===")

# Create absolute time grid using pkl data
isi_absolute_data = create_isi_absolute_time_grid_from_pkl(
    data,
    max_isi_ms=2300.0,
    dt_ms=10.0
)

# Run component analysis on absolute time grid
absolute_results = run_isi_absolute_time_analysis(
    isi_absolute_data,
    n_components=20,
    use_ica=True,
    use_nmf=True,
    min_trial_coverage=0.3
)

# Visualize results
if absolute_results:
    visualize_isi_absolute_components(isi_absolute_data, absolute_results, max_components_show=12)
else:
    print("Absolute time analysis failed - no results to visualize")




















# %%

# Run the masked grid approach
print("=== RUNNING MASKED GRID WITH PHASE COMPARISON ===")

# Create the masked grids (both absolute and phase)
masked_grid_data = create_isi_masked_grid_with_phase(
    data,
    max_isi_ms=2300.0,
    dt_ms=10.0,
    n_phase_samples=100
)

# Run NMF on both grids
nmf_results = run_nmf_on_masked_grid(
    masked_grid_data,
    n_components=15,
    use_absolute=True,
    use_phase=True,
    min_coverage=0.3
)

# Visualize the results
visualize_masked_grid_nmf_results(masked_grid_data, nmf_results, max_components=12)











# %%
# Run the specificity analysis
specificity_results = analyze_nmf_component_isi_specificity(nmf_results, masked_grid_data)

# Visualize the top ISI-specific components
visualize_top_isi_specific_components(specificity_results, max_components=6)




# %%
# Now you can run this with your data structure:
print("=== RUNNING PHASE ANALYSIS WITH YOUR DATA STRUCTURE ===")

phase_results = run_isi_phase_analysis_from_data(
    data,
    n_phase_bins=80,
    n_components=12
)

# Visualize the results
visualize_phase_cp_results(phase_results)