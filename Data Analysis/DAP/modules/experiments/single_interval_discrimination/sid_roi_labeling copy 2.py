from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional

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


def load_event_aligned_data(event_data_dict: Dict[str, Any], event_name: str, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract event-aligned trial data from provided data dictionary.
    
    Args:
        event_data_dict: Dictionary containing event-aligned data
        event_name: Name of event for logging
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with trial data or None if failed
    """
    try:
        print(f"Processing {event_name} data: {event_data_dict['dff_segments_array'].shape} (trials, ROIs, time)")
        return event_data_dict
        
    except Exception as e:
        print(f"Failed to process {event_name} data: {e}")
        return None

def detect_event_responsive_rois(event_data: Dict[str, Any], event_name: str, 
                                cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect ROIs responsive to a specific event using event-aligned data.
    
    Args:
        event_data: Dictionary containing event-aligned trial segments
        event_name: Name of the event for logging
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with detection results
    """
    try:
        print(f"=== Detecting {event_name} responsive ROIs ===")
        
        # Extract data
        dff_segments = event_data['dff_segments_array']  # (n_trials, n_rois, n_time)
        time_vector = event_data['common_time_vector']
        df_trials = event_data['df_trials_with_segments'].copy() if 'df_trials_with_segments' in event_data else None
        
        n_trials, n_rois, n_timepoints = dff_segments.shape
        print(f"Analyzing {n_trials} trials, {n_rois} ROIs, {n_timepoints} time points")
        
        # Define baseline and response windows
        baseline_pct = cfg['detection_thresholds']['baseline_window_pct']
        response_pct = cfg['detection_thresholds']['response_window_pct']
        
        # Event should be at time=0, so baseline is negative times, response is positive
        baseline_start_idx = 0
        baseline_end_idx = int(baseline_pct * n_timepoints)
        
        # Find event time (should be around time=0)
        event_idx = np.argmin(np.abs(time_vector))
        response_start_idx = max(event_idx + 1, baseline_end_idx + 1)  # After baseline + small gap
        response_end_idx = min(response_start_idx + int(response_pct * n_timepoints), n_timepoints)
        
        print(f"Baseline window: indices {baseline_start_idx}-{baseline_end_idx} "
              f"(t={time_vector[baseline_start_idx]:.3f} to {time_vector[baseline_end_idx-1]:.3f}s)")
        print(f"Response window: indices {response_start_idx}-{response_end_idx} "
              f"(t={time_vector[response_start_idx]:.3f} to {time_vector[response_end_idx-1]:.3f}s)")
        
        # Analyze each ROI
        responsive_rois = []
        roi_results = {}
        
        for roi_idx in range(n_rois):
            # Extract baseline and response values across all trials
            baseline_values = np.mean(dff_segments[:, roi_idx, baseline_start_idx:baseline_end_idx], axis=1)
            response_values = np.mean(dff_segments[:, roi_idx, response_start_idx:response_end_idx], axis=1)
            
            # Remove trials with NaN values
            valid_mask = np.isfinite(baseline_values) & np.isfinite(response_values)
            baseline_clean = baseline_values[valid_mask]
            response_clean = response_values[valid_mask]
            
            if len(baseline_clean) < cfg['detection_thresholds']['min_trials']:
                roi_results[roi_idx] = {
                    'is_responsive': False,
                    'p_value': 1.0,
                    'effect_size': 0.0,
                    'n_trials': len(baseline_clean),
                    'insufficient_trials': True
                }
                continue
            
            # Statistical test (paired t-test)
            try:
                t_stat, p_value = stats.ttest_rel(response_clean, baseline_clean)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(baseline_clean) + np.var(response_clean)) / 2)
                if pooled_std > 1e-15:
                    effect_size = (np.mean(response_clean) - np.mean(baseline_clean)) / pooled_std
                else:
                    effect_size = 0.0
                
                # Response magnitude and timing
                mean_trace = np.mean(dff_segments[valid_mask, roi_idx, :], axis=0)
                response_magnitude = np.max(np.abs(mean_trace[response_start_idx:response_end_idx]))
                response_timing_idx = response_start_idx + np.argmax(np.abs(mean_trace[response_start_idx:response_end_idx]))
                response_timing = time_vector[response_timing_idx]
                
                # Determine if responsive
                is_responsive = (p_value < cfg['detection_thresholds']['response_threshold'] and 
                               abs(effect_size) >= cfg['detection_thresholds']['min_effect_size'])
                
                if is_responsive:
                    responsive_rois.append(roi_idx)
                
                roi_results[roi_idx] = {
                    'is_responsive': is_responsive,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    't_statistic': t_stat,
                    'response_magnitude': response_magnitude,
                    'response_timing': response_timing,
                    'baseline_mean': np.mean(baseline_clean),
                    'baseline_std': np.std(baseline_clean),
                    'response_mean': np.mean(response_clean),
                    'response_std': np.std(response_clean),
                    'n_trials': len(baseline_clean),
                    'response_direction': 'positive' if effect_size > 0 else 'negative'
                }
                
            except Exception as e:
                print(f"Error analyzing ROI {roi_idx}: {e}")
                roi_results[roi_idx] = {
                    'is_responsive': False,
                    'p_value': 1.0,
                    'effect_size': 0.0,
                    'error': str(e)
                }
        
        print(f"Found {len(responsive_rois)} {event_name}-responsive ROIs")
        
        return {
            'event_name': event_name,
            'responsive_rois': responsive_rois,
            'roi_results': roi_results,
            'analysis_params': {
                'baseline_window': (time_vector[baseline_start_idx], time_vector[baseline_end_idx-1]),
                'response_window': (time_vector[response_start_idx], time_vector[response_end_idx-1]),
                'n_trials_analyzed': n_trials,
                'thresholds': cfg['detection_thresholds']
            }
        }
        
    except Exception as e:
        print(f"Failed to detect {event_name} responsive ROIs: {e}")
        return {'error': str(e)}

def detect_isi_sensitive_rois(trial_start_data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect ROIs sensitive to ISI duration using trial_start aligned data.
    Focus only on F1 offset to F2 onset period for long ISI trials.
    
    Args:
        trial_start_data: Dictionary containing trial_start aligned data
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with ISI sensitivity detection results
    """
    try:
        print("=== Detecting ISI-sensitive ROIs ===")
        
        # Extract data
        dff_segments = trial_start_data['dff_segments_array']  # (n_trials, n_rois, n_time)
        time_vector = trial_start_data['common_time_vector']
        df_trials = trial_start_data['df_trials_with_segments'].copy()
        
        n_trials, n_rois, n_timepoints = dff_segments.shape
        print(f"Analyzing {n_trials} trials, {n_rois} ROIs")
        
        # Filter to long ISI trials using is_right==1 as proxy
        if 'is_right' in df_trials.columns:
            long_isi_mask = df_trials['is_right'] == 1
        else:
            print("Warning: 'is_right' column not found, using ISI threshold")
            isi_values = df_trials['isi'] if 'isi' in df_trials.columns else np.ones(len(df_trials))
            long_isi_threshold = np.percentile(isi_values, 75)
            long_isi_mask = isi_values >= long_isi_threshold
        
        long_isi_trials = np.where(long_isi_mask)[0]
        print(f"Using {len(long_isi_trials)} long ISI trials")
        
        if len(long_isi_trials) < cfg['detection_thresholds']['min_trials']:
            print("Insufficient long ISI trials for analysis")
            return {'error': 'insufficient_long_isi_trials'}
        
        # Define time windows for ISI analysis
        # Estimate timing from typical trial structure
        f1_offset_time = cfg['isi_analysis']['f1_offset_time']
        
        # Get F2 onset time from trial data if available
        if 'start_flash_2' in df_trials.columns:
            f2_onset_time = df_trials['start_flash_2'].mean()
        else:
            f2_onset_time = cfg['isi_analysis']['f2_onset_time_fallback']
        
        print(f"F1 offset time: {f1_offset_time:.2f}s")
        print(f"F2 onset time: {f2_onset_time:.2f}s")
        
        # Find corresponding indices
        f1_offset_idx = np.argmin(np.abs(time_vector - f1_offset_time))
        f2_onset_idx = np.argmin(np.abs(time_vector - f2_onset_time))
        
        # Define baseline (before F1 offset) and ISI period (F1 offset to F2 onset)
        baseline_duration = cfg['isi_analysis']['baseline_duration']
        baseline_start_idx = max(0, f1_offset_idx - int(baseline_duration * 30))  # 30Hz approx
        baseline_end_idx = f1_offset_idx
        
        isi_buffer = cfg['isi_analysis']['isi_buffer']
        isi_start_idx = f1_offset_idx + int(isi_buffer * 30)  # Skip immediate F1 offset response
        isi_end_idx = f2_onset_idx - int(isi_buffer * 30)     # Stop before F2 onset response
        
        if isi_end_idx <= isi_start_idx:
            print("Error: ISI window too small")
            return {'error': 'isi_window_too_small'}
        
        print(f"Baseline window: {time_vector[baseline_start_idx]:.3f} to {time_vector[baseline_end_idx-1]:.3f}s")
        print(f"ISI window: {time_vector[isi_start_idx]:.3f} to {time_vector[isi_end_idx-1]:.3f}s")
        
        # Analyze each ROI for ISI sensitivity
        isi_sensitive_rois = []
        roi_results = {}
        
        for roi_idx in range(n_rois):
            # Extract baseline and ISI values for long ISI trials only
            baseline_values = np.mean(dff_segments[long_isi_trials, roi_idx, baseline_start_idx:baseline_end_idx], axis=1)
            isi_values = np.mean(dff_segments[long_isi_trials, roi_idx, isi_start_idx:isi_end_idx], axis=1)
            
            # Remove trials with NaN values
            valid_mask = np.isfinite(baseline_values) & np.isfinite(isi_values)
            baseline_clean = baseline_values[valid_mask]
            isi_clean = isi_values[valid_mask]
            
            if len(baseline_clean) < cfg['detection_thresholds']['min_isi_trials']:
                roi_results[roi_idx] = {
                    'is_isi_sensitive': False,
                    'p_value': 1.0,
                    'effect_size': 0.0,
                    'insufficient_trials': True
                }
                continue
            
            # Statistical test for sustained modulation
            try:
                t_stat, p_value = stats.ttest_rel(isi_clean, baseline_clean)
                
                # Effect size
                pooled_std = np.sqrt((np.var(baseline_clean) + np.var(isi_clean)) / 2)
                if pooled_std > 1e-15:
                    effect_size = (np.mean(isi_clean) - np.mean(baseline_clean)) / pooled_std
                else:
                    effect_size = 0.0
                
                # Determine if ISI sensitive
                is_isi_sensitive = (p_value < cfg['detection_thresholds']['isi_sensitivity_threshold'] and 
                                  abs(effect_size) >= cfg['detection_thresholds']['min_effect_size'])
                
                if is_isi_sensitive:
                    isi_sensitive_rois.append(roi_idx)
                
                roi_results[roi_idx] = {
                    'is_isi_sensitive': is_isi_sensitive,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    't_statistic': t_stat,
                    'baseline_mean': np.mean(baseline_clean),
                    'isi_mean': np.mean(isi_clean),
                    'n_trials': len(baseline_clean),
                    'modulation_direction': 'sustained_suppression' if effect_size < 0 else 'sustained_activation'
                }
                
            except Exception as e:
                roi_results[roi_idx] = {
                    'is_isi_sensitive': False,
                    'p_value': 1.0,
                    'effect_size': 0.0,
                    'error': str(e)
                }
        
        print(f"Found {len(isi_sensitive_rois)} ISI-sensitive ROIs")
        
        return {
            'isi_sensitive_rois': isi_sensitive_rois,
            'roi_results': roi_results,
            'analysis_params': {
                'baseline_window': (time_vector[baseline_start_idx], time_vector[baseline_end_idx-1]),
                'isi_window': (time_vector[isi_start_idx], time_vector[isi_end_idx-1]),
                'n_long_isi_trials': len(long_isi_trials),
                'thresholds': cfg['detection_thresholds']
            }
        }
        
    except Exception as e:
        print(f"Failed to detect ISI-sensitive ROIs: {e}")
        return {'error': str(e)}

def filter_lick_trials(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter lick_start data to only include trials with actual licks.
    
    Args:
        event_data: Dictionary containing lick_start event data
        
    Returns:
        Filtered event data with only trials where lick == 1
    """
    try:
        if 'df_trials_with_segments' not in event_data:
            print("Warning: No trial metadata found for lick filtering")
            return event_data
        
        df_trials = event_data['df_trials_with_segments']
        
        if 'lick' not in df_trials.columns:
            print("Warning: 'lick' column not found, using all trials")
            return event_data
        
        # Filter to trials with lick == 1
        lick_mask = df_trials['lick'] == 1
        n_lick_trials = np.sum(lick_mask)
        
        print(f"Filtering to {n_lick_trials} trials with licks (from {len(df_trials)} total)")
        
        if n_lick_trials < 10:
            print("Warning: Very few lick trials available")
            return event_data
        
        # Filter all relevant arrays
        filtered_data = event_data.copy()
        filtered_data['dff_segments_array'] = event_data['dff_segments_array'][lick_mask]
        filtered_data['df_trials_with_segments'] = df_trials[lick_mask].copy()
        
        # Update trial count info
        if 'n_trials' in filtered_data:
            filtered_data['n_trials'] = n_lick_trials
        
        return filtered_data
        
    except Exception as e:
        print(f"Error filtering lick trials: {e}")
        return event_data

def compute_roi_group_intersections(roi_groups: Dict[str, List[int]], cfg: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Compute intersections between ROI groups.
    
    Args:
        roi_groups: Dictionary mapping group names to lists of ROI indices
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with intersection groups added
    """
    try:
        intersections = {}
        group_names = list(roi_groups.keys())
        
        # Pairwise intersections
        for i, name1 in enumerate(group_names):
            for name2 in group_names[i+1:]:
                set1 = set(roi_groups[name1])
                set2 = set(roi_groups[name2])
                
                intersection = set1.intersection(set2)
                
                if len(intersection) >= cfg['roi_grouping']['intersection_threshold']:
                    intersections[f'{name1}∩{name2}'] = list(intersection)
        
        # Triple intersections (if any pairwise intersections are large enough)
        if len(group_names) >= 3:
            for i, name1 in enumerate(group_names):
                for j, name2 in enumerate(group_names[i+1:], i+1):
                    for name3 in group_names[j+1:]:
                        set1 = set(roi_groups[name1])
                        set2 = set(roi_groups[name2])
                        set3 = set(roi_groups[name3])
                        
                        intersection = set1.intersection(set2).intersection(set3)
                        
                        if len(intersection) >= cfg['roi_grouping']['intersection_threshold']:
                            intersections[f'{name1}∩{name2}∩{name3}'] = list(intersection)
        
        return intersections
        
    except Exception as e:
        print(f"Error computing intersections: {e}")
        return {}





def analyze_event_aligned_roi_groups(event_data_list: List[Dict[str, Any]], 
                                    cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete event-aligned ROI group analysis pipeline.
    
    Args:
        event_data_list: List of dictionaries containing event-aligned data
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with all ROI group analysis results
    """
    try:
        print("=" * 80)
        print("ANALYZING EVENT-ALIGNED ROI GROUPS")
        print("=" * 80)
        
        results = {
            'roi_groups': {},
            'group_intersections': {},
            'analysis_summary': {},
            'config': cfg
        }
        
        # Load and analyze each event
        event_analyses = {}
        event_names = list(cfg['event_files'].keys())
        
        for i, event_name in enumerate(event_names):
            if i >= len(event_data_list):
                print(f"Skipping {event_name} - data not provided")
                continue
            
            print(f"\n--- Analyzing {event_name} ---")
            
            # Get event data
            event_data = event_data_list[i]
            if event_data is None:
                print(f"Skipping {event_name} - data is None")
                continue
            
            # Skip trial_start for standard event detection - only use for ISI analysis
            if event_name == 'trial_start':
                print("Skipping trial_start for event detection (used only for ISI analysis)")
                continue
            
            # Special handling for lick_start
            if event_name == 'lick_start':
                event_data = filter_lick_trials(event_data)
            
            # Detect responsive ROIs for meaningful events only
            analysis_result = detect_event_responsive_rois(event_data, event_name, cfg)
            if 'error' not in analysis_result:
                event_analyses[event_name] = analysis_result
                results['roi_groups'][event_name] = analysis_result['responsive_rois']
            else:
                print(f"Failed to analyze {event_name}: {analysis_result['error']}")
        
        # Run ISI sensitivity analysis using trial_start data
        if len(event_data_list) > 0 and event_data_list[0] is not None:
            print(f"\n--- Analyzing ISI sensitivity ---")
            isi_analysis_result = detect_isi_sensitive_rois(event_data_list[0], cfg)
            if 'error' not in isi_analysis_result:
                event_analyses['isi_sensitivity'] = isi_analysis_result
                results['roi_groups']['isi_sensitivity'] = isi_analysis_result['isi_sensitive_rois']
            else:
                print(f"Failed ISI sensitivity analysis: {isi_analysis_result['error']}")
        
        # Compute intersections
        print(f"\n--- Computing ROI group intersections ---")
        intersections = compute_roi_group_intersections(results['roi_groups'], cfg)
        results['group_intersections'] = intersections
        
        # Create analysis summary
        summary = {
            'total_events_analyzed': len(event_analyses),
            'total_roi_groups': len(results['roi_groups']),
            'total_intersections': len(intersections),
            'group_sizes': {name: len(rois) for name, rois in results['roi_groups'].items()},
            'intersection_sizes': {name: len(rois) for name, rois in intersections.items()}
        }
        
        results['analysis_summary'] = summary
        results['event_analyses'] = event_analyses  # Detailed results
        
        # Print summary
        print(f"\n" + "=" * 80)
        print("EVENT-ALIGNED ROI GROUP ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Events analyzed: {summary['total_events_analyzed']}")
        print(f"ROI groups found: {summary['total_roi_groups']}")
        print(f"Intersections found: {summary['total_intersections']}")
        
        print(f"\nROI Group Sizes:")
        for name, size in summary['group_sizes'].items():
            print(f"  {name}: {size} ROIs")
        
        if intersections:
            print(f"\nIntersection Sizes:")
            for name, size in summary['intersection_sizes'].items():
                print(f"  {name}: {size} ROIs")
        
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"Failed to analyze event-aligned ROI groups: {e}")
        return {'error': str(e)}






def create_simplified_raster_plot(roi_groups: Dict[str, List[int]], 
                                 trial_start_data: Dict[str, Any],
                                 target_isi: float, cfg: Dict[str, Any]) -> plt.Figure:
    """
    Create simplified raster plot for specified ISI using discovered ROI groups.
    
    Args:
        roi_groups: Dictionary of ROI groups from analysis
        trial_start_data: Trial-start aligned data for raster creation
        target_isi: Target ISI value for trial filtering
        cfg: Configuration dictionary
        
    Returns:
        Figure object or None if failed
    """
    try:
        print(f"=== Creating raster for ISI {target_isi}ms ===")
        
        # Extract data
        dff_segments = trial_start_data['dff_segments_array']  # (n_trials, n_rois, n_time)
        time_vector = trial_start_data['common_time_vector']
        df_trials = trial_start_data['df_trials_with_segments']
        
        # Filter trials by ISI
        if 'isi' in df_trials.columns:
            isi_tolerance = cfg['plotting']['isi_tolerance']
            isi_mask = np.abs(df_trials['isi'] - target_isi) < isi_tolerance
            trial_indices = np.where(isi_mask)[0]
            print(f"Using {len(trial_indices)} trials with ISI ≈ {target_isi}ms")
        else:
            trial_indices = np.arange(len(df_trials))
        
        if len(trial_indices) < cfg['plotting']['min_trials_for_plot']:
            print("Too few trials for meaningful raster plot")
            return None
        
        # Filter data to selected trials
        filtered_segments = dff_segments[trial_indices]
        filtered_trials = df_trials.iloc[trial_indices]
        
        # Get main ROI groups (exclude intersections for cleaner display)
        main_groups = {name: indices for name, indices in roi_groups.items() if '∩' not in name}
        
        if len(main_groups) == 0:
            print("No main ROI groups found")
            return None
        
        # Create figure
        n_groups = len(main_groups)
        fig = plt.figure(figsize=(14, 3 * n_groups + 2))
        
        # Add behavioral context at top
        gs = GridSpec(n_groups + 1, 2, height_ratios=[0.3] + [1] * n_groups,
                      width_ratios=[3, 1], hspace=0.3, wspace=0.2, figure=fig)
        
        # Plot behavioral context
        ax_behav = fig.add_subplot(gs[0, :])
        plot_behavioral_context(ax_behav, filtered_trials, target_isi)
        
        # Process each ROI group
        group_averages = {}
        
        for group_idx, (group_name, roi_indices) in enumerate(main_groups.items()):
            print(f"Processing {group_name}: {len(roi_indices)} ROIs")
            
            # Select top ROIs (limit for clarity)
            max_rois = cfg['plotting']['max_rois_per_group']
            selected_rois = roi_indices[:max_rois]
            
            # Get data for selected ROIs
            group_data = filtered_segments[:, selected_rois, :]  # (trials, selected_rois, time)
            
            # Create raster subplot
            ax_raster = fig.add_subplot(gs[group_idx + 1, 0])
            is_bottom = (group_idx == len(main_groups) - 1)
            plot_group_raster(ax_raster, group_data, time_vector, filtered_trials, 
                             group_name, target_isi, is_bottom)
            
            # Create average trace subplot
            ax_avg = fig.add_subplot(gs[group_idx + 1, 1])
            avg_trace = plot_group_average(ax_avg, group_data, time_vector, 
                                         group_name, target_isi)
            group_averages[group_name] = avg_trace
        
        # Title
        title = f'Functional ROI Groups - ISI {target_isi}ms'
        fig.suptitle(title, fontsize=14, y=0.95)
        
        return fig
        
    except Exception as e:
        print(f"Failed to create raster plot: {e}")
        return None

def plot_behavioral_context(ax: plt.Axes, trial_data: pd.DataFrame, target_isi: float):
    """Plot behavioral context strip at top of raster."""
    try:
        n_trials = len(trial_data)
        
        # Sort trials by choice for cleaner visualization
        if 'is_right_choice' in trial_data.columns:
            sort_order = trial_data['is_right_choice'].argsort()
        else:
            sort_order = np.arange(n_trials)
        
        # Create behavioral strips
        y_positions = [0, 1, 2]
        labels = ['Choice', 'Stimulus', 'Outcome']
        
        for i, trial_idx in enumerate(sort_order):
            row = trial_data.iloc[trial_idx]
            x_pos = i
            
            # Choice (red=right, blue=left, gray=no choice)
            if 'is_right_choice' in trial_data.columns and 'did_not_choose' in trial_data.columns:
                if row.get('did_not_choose', False):
                    color = 'gray'
                elif row.get('is_right_choice', False):
                    color = 'red'
                else:
                    color = 'blue'
                ax.barh(y_positions[0], 1, left=x_pos, height=0.8, color=color, alpha=0.8)
            
            # Stimulus type
            if 'is_right' in trial_data.columns:
                color = 'lightcoral' if row.get('is_right', False) else 'lightblue'
                ax.barh(y_positions[1], 1, left=x_pos, height=0.8, color=color, alpha=0.8)
            
            # Outcome
            if 'rewarded' in trial_data.columns:
                color = 'gold' if row.get('rewarded', False) else 'lightgray'
                ax.barh(y_positions[2], 1, left=x_pos, height=0.8, color=color, alpha=0.8)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, n_trials)
        ax.set_ylim(-0.5, 2.5)
        ax.set_title(f'Trial Context (n={n_trials}, ISI={target_isi}ms)', fontsize=12)
        ax.set_xlabel('Trial (sorted by choice)')
        
    except Exception as e:
        print(f"Error plotting behavioral context: {e}")




def plot_group_raster(ax: plt.Axes, group_data: np.ndarray, time_vector: np.ndarray,
                     trial_data: pd.DataFrame, group_name: str, target_isi: float, 
                     is_bottom: bool, event_name: str = None):
    """Plot raster for a single ROI group with correct event markers."""
    try:
        n_trials, n_rois, n_timepoints = group_data.shape
        
        # Sort trials by choice
        if 'is_right_choice' in trial_data.columns:
            sort_order = trial_data['is_right_choice'].argsort()
        else:
            sort_order = np.arange(n_trials)
        
        # Transpose for plotting: (n_rois, n_timepoints, n_trials)
        plot_data_transposed = group_data.transpose(1, 2, 0)
        
        # Smooth and normalize
        smoothed_data = gaussian_filter1d(plot_data_transposed, sigma=1.5, axis=1)
        
        # Z-score normalize each ROI
        for roi_idx in range(n_rois):
            roi_data = smoothed_data[roi_idx, :, :].flatten()
            valid_mask = np.isfinite(roi_data)
            if valid_mask.sum() > 10:
                mean_val = np.mean(roi_data[valid_mask])
                std_val = np.std(roi_data[valid_mask])
                if std_val > 0:
                    smoothed_data[roi_idx, :, :] = (smoothed_data[roi_idx, :, :] - mean_val) / std_val
        
        # Create raster matrix: (n_rois * n_trials, n_timepoints)
        raster_matrix = np.zeros((n_rois * n_trials, n_timepoints))
        
        for i, trial_idx in enumerate(sort_order):
            start_row = i * n_rois
            end_row = start_row + n_rois
            raster_matrix[start_row:end_row, :] = smoothed_data[:, :, trial_idx]
        
        # Plot raster
        im = ax.imshow(raster_matrix, aspect='auto', cmap='RdBu_r',
                       extent=[time_vector[0], time_vector[-1], 0, n_rois * n_trials],
                       vmin=-3, vmax=3, interpolation='bilinear')
        
        # Add trial separators
        for i in range(1, n_trials):
            y_pos = i * n_rois
            ax.axhline(y_pos, color='white', linewidth=1.5, alpha=0.9)
        
        # Add correct event markers based on alignment
        if event_name:
            # Event at time=0 for event-aligned data
            ax.axvline(0, color='red', linestyle='-', alpha=0.8, linewidth=2, 
                      label=event_name.replace('_', ' ').title())
        else:
            # For trial_start aligned data, show F1 and F2
            ax.axvline(0.5, color='blue', linestyle='-', alpha=0.8, linewidth=2, label='F1')
            if target_isi:
                f2_time = 0.5 + target_isi / 1000
                ax.axvline(f2_time, color='blue', linestyle='-', alpha=0.8, linewidth=2, label='F2')
        
        # Formatting
        clean_name = group_name.replace('_', ' ').title()
        ax.set_ylabel(f'{clean_name}\n({n_rois} ROIs)', fontsize=10)
        ax.set_xlim(time_vector[0], time_vector[-1])
        ax.legend(loc='upper right', fontsize=8)
        
        if is_bottom:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xticks([])
            
    except Exception as e:
        print(f"Error plotting group raster: {e}")

def plot_group_average(ax: plt.Axes, group_data: np.ndarray, time_vector: np.ndarray,
                      group_name: str, target_isi: float, event_name: str = None) -> np.ndarray:
    """Plot average trace for ROI group with correct event markers."""
    try:
        # Average across ROIs and trials: (n_trials, n_rois, n_timepoints) -> (n_timepoints,)
        avg_trace = np.nanmean(group_data, axis=(0, 1))
        std_trace = np.nanstd(group_data, axis=(0, 1)) / np.sqrt(group_data.shape[0] * group_data.shape[1])
        
        # Smooth
        avg_smooth = gaussian_filter1d(avg_trace, sigma=1.0)
        std_smooth = gaussian_filter1d(std_trace, sigma=1.0)
        
        # Plot
        ax.plot(time_vector, avg_smooth, linewidth=2, label=group_name.split('_')[0].title())
        ax.fill_between(time_vector, avg_smooth - std_smooth, avg_smooth + std_smooth, alpha=0.3)
        
        # Add correct event markers
        if event_name:
            ax.axvline(0, color='red', linestyle='--', alpha=0.7, label=event_name.replace('_', ' '))
        else:
            ax.axvline(0.5, color='blue', linestyle='--', alpha=0.5)
            if target_isi:
                f2_time = 0.5 + target_isi / 1000
                ax.axvline(f2_time, color='blue', linestyle='--', alpha=0.5)
        
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylabel('ΔF/F (z-score)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        return avg_smooth
        
    except Exception as e:
        print(f"Error plotting group average: {e}")
        return np.zeros_like(time_vector)

def create_group_comparison_plot(roi_groups: Dict[str, List[int]], 
                               event_data_list: List[Dict[str, Any]],
                               cfg: Dict[str, Any]) -> plt.Figure:
    """
    Create comprehensive group comparison plot showing all functional groups
    aligned to their respective events.
    """
    try:
        print("=== Creating functional group comparison plot ===")
        
        # Get main groups (exclude intersections)
        main_groups = {name: indices for name, indices in roi_groups.items() if '∩' not in name}
        event_names = list(cfg['event_files'].keys())
        
        # Create figure with subplots for each group
        n_groups = len(main_groups)
        fig, axes = plt.subplots(n_groups, 1, figsize=(12, 2.5 * n_groups), 
                                sharex=True, constrained_layout=True)
        
        if n_groups == 1:
            axes = [axes]
        
        # Color map for groups
        colors = plt.cm.Set1(np.linspace(0, 1, n_groups))
        
        group_idx = 0
        for group_name, roi_indices in main_groups.items():
            if len(roi_indices) == 0:
                continue
                
            ax = axes[group_idx]
            
            # Find corresponding event data
            event_data = None
            event_name = None
            
            if group_name == 'isi_sensitivity':
                # Use trial_start data
                event_data = event_data_list[0] if len(event_data_list) > 0 else None
                event_name = 'trial_start'
            else:
                # Find matching event
                for i, ename in enumerate(event_names):
                    if ename in group_name or group_name in ename:
                        if i < len(event_data_list):
                            event_data = event_data_list[i]
                            event_name = ename
                        break
            
            if event_data is None:
                print(f"No event data found for {group_name}")
                continue
            
            # Get data for this group
            dff_segments = event_data['dff_segments_array']
            time_vector = event_data['common_time_vector']
            
            # Select ROIs (limit for performance)
            max_rois = min(len(roi_indices), 50)
            selected_rois = roi_indices[:max_rois]
            group_data = dff_segments[:, selected_rois, :]
            
            # Calculate group average
            avg_trace = np.nanmean(group_data, axis=(0, 1))
            std_trace = np.nanstd(group_data, axis=(0, 1)) / np.sqrt(group_data.shape[0] * group_data.shape[1])
            
            # Smooth
            avg_smooth = gaussian_filter1d(avg_trace, sigma=1.0)
            std_smooth = gaussian_filter1d(std_trace, sigma=1.0)
            
            # Plot
            color = colors[group_idx]
            clean_name = group_name.replace('_', ' ').title()
            
            ax.plot(time_vector, avg_smooth, linewidth=2.5, color=color, 
                   label=f'{clean_name} (n={len(selected_rois)})')
            ax.fill_between(time_vector, avg_smooth - std_smooth, avg_smooth + std_smooth, 
                           alpha=0.3, color=color)
            
            # Add event marker
            if event_name == 'trial_start':
                # Show F1 and F2 for trial_start aligned
                ax.axvline(0.5, color='blue', linestyle='--', alpha=0.7, label='F1 onset')
                ax.axvline(1.2, color='blue', linestyle=':', alpha=0.7, label='F2 onset (avg)')
            else:
                # Show event at time=0
                ax.axvline(0, color='red', linestyle='--', alpha=0.8, 
                          label=event_name.replace('_', ' ').title())
            
            ax.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax.set_ylabel('ΔF/F (z-score)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_title(f'{clean_name} - Aligned to {event_name.replace("_", " ").title()}', 
                        fontsize=11)
            
            group_idx += 1
        
        # Set common x-label
        axes[-1].set_xlabel('Time relative to event (s)', fontsize=10)
        
        fig.suptitle('Functional ROI Groups - Event-Aligned Responses', fontsize=14)
        
        return fig
        
    except Exception as e:
        print(f"Failed to create group comparison plot: {e}")
        return None

def create_trial_start_overview_plot(roi_groups: Dict[str, List[int]], 
                                   trial_start_data: Dict[str, Any],
                                   cfg: Dict[str, Any]) -> plt.Figure:
    """
    Create overview plot showing all groups on trial_start timeline for comparison.
    """
    try:
        print("=== Creating trial-start overview plot ===")
        
        # Get main groups
        main_groups = {name: indices for name, indices in roi_groups.items() if '∩' not in name}
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Extract trial_start data
        dff_segments = trial_start_data['dff_segments_array']
        time_vector = trial_start_data['common_time_vector']
        
        # Color map
        colors = plt.cm.Set1(np.linspace(0, 1, len(main_groups)))
        
        # Plot each group
        for i, (group_name, roi_indices) in enumerate(main_groups.items()):
            if len(roi_indices) == 0:
                continue
            
            # Select ROIs
            max_rois = min(len(roi_indices), 30)
            selected_rois = roi_indices[:max_rois]
            group_data = dff_segments[:, selected_rois, :]
            
            # Calculate average
            avg_trace = np.nanmean(group_data, axis=(0, 1))
            std_trace = np.nanstd(group_data, axis=(0, 1)) / np.sqrt(group_data.shape[0] * group_data.shape[1])
            
            # Smooth
            avg_smooth = gaussian_filter1d(avg_trace, sigma=1.5)
            std_smooth = gaussian_filter1d(std_trace, sigma=1.5)
            
            # Plot
            color = colors[i]
            clean_name = group_name.replace('_', ' ').title()
            
            ax.plot(time_vector, avg_smooth, linewidth=2.5, color=color, 
                   label=f'{clean_name} (n={len(selected_rois)})')
            ax.fill_between(time_vector, avg_smooth - std_smooth, avg_smooth + std_smooth, 
                           alpha=0.2, color=color)
        
        # Add trial events
        ax.axvline(0.5, color='blue', linestyle='-', alpha=0.8, linewidth=2, label='F1 onset')
        ax.axvline(1.0, color='blue', linestyle='--', alpha=0.6, label='F1 offset')
        ax.axvline(1.7, color='green', linestyle='-', alpha=0.8, linewidth=2, label='F2 onset (1700ms)')
        ax.axvline(2.2, color='green', linestyle='--', alpha=0.6, label='F2 offset')
        ax.axvline(3.0, color='red', linestyle='-', alpha=0.8, linewidth=2, label='Choice start')
        
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Time from trial start (s)', fontsize=12)
        ax.set_ylabel('ΔF/F (z-score)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.set_title('Functional ROI Groups - Trial Timeline Overview', fontsize=14)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Failed to create trial overview plot: {e}")
        return None



def run_event_aligned_analysis(event_data_list: List[Dict[str, Any]], 
                              cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the complete event-aligned ROI analysis.
    
    Args:
        event_data_list: List of event-aligned data dictionaries in order specified by cfg
        cfg: Configuration dictionary
        
    Returns:
        Complete analysis results
    """
    try:
        # Run main analysis
        results = analyze_event_aligned_roi_groups(event_data_list, cfg)
        
        if 'error' in results:
            return results
        
        # Create comprehensive visualization
        roi_groups = results['roi_groups']
        
        # Create group comparison plot (each group aligned to its event)
        group_comparison_fig = create_group_comparison_plot(roi_groups, event_data_list, cfg)
        
        # Create trial timeline overview (all groups on trial_start timeline)
        trial_start_data = event_data_list[0] if len(event_data_list) > 0 else None
        trial_overview_fig = None
        if trial_start_data is not None:
            trial_overview_fig = create_trial_start_overview_plot(roi_groups, trial_start_data, cfg)
        
        # Create raster plots for target ISIs
        raster_figures = {}
        if trial_start_data is not None:
            for target_isi in cfg['plotting']['target_isis']:
                print(f"Creating raster for ISI {target_isi}ms...")
                fig = create_simplified_raster_plot(roi_groups, trial_start_data, target_isi, cfg)
                if fig is not None:
                    raster_figures[f'isi_{target_isi}'] = fig
        
        results['figures'] = {
            'group_comparison': group_comparison_fig,
            'trial_overview': trial_overview_fig,
            'rasters': raster_figures
        }
        
        return results
        
    except Exception as e:
        print(f"Failed to run event-aligned analysis: {e}")
        return {'error': str(e)}







if __name__ == '__main__':
    print('Starting ROI motif labeling analysis...\n')
    
    
    
    
# %%




# %%


# path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_simplex_20250529_2afc-379/sid_imaging_segmented_data.pkl'
path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/sid_imaging_segmented_data.pkl'



path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/trial_start.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_trial_start = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_trial_start.keys())}")




path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/start_flash_1.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_start_flash_1 = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_start_flash_1.keys())}")




path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/end_flash_1.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_end_flash_1 = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_end_flash_1.keys())}")




path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/start_flash_2.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_start_flash_2 = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_start_flash_2.keys())}")






path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/end_flash_2.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_end_flash_2 = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_end_flash_2.keys())}")




path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/choice_start.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_choice_start = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_choice_start.keys())}")





path = 'D:/behavior/2p_imaging/processed/2afc/YH24LG/YH24LG_CRBL_lobulev_20250620_2afc-494/lick_start.pkl'

import pickle

with open(path, 'rb') as f:
    trial_data_lick_start = pickle.load(f)   # one object back (e.g., a dict)  

print(f"Loaded trial data from: {path}")
print(f"Data keys: {list(trial_data_lick_start.keys())}")



# trial_data_trial_start
# trial_data_start_flash_1
# trial_data_end_flash_1
# trial_data_start_flash_2
# trial_data_end_flash_2
# trial_data_choice_start
# trial_data_lick_start

# %%


# Configuration for event-aligned analysis
cfg_event_aligned = {
    'event_files': {
        'trial_start': 'trial_start.pkl',        # For ISI analysis and raster plots
        'f1_onset': 'start_flash_1.pkl',
        'f1_offset': 'end_flash_1.pkl', 
        'f2_onset': 'start_flash_2.pkl',
        'f2_offset': 'end_flash_2.pkl',
        'choice_start': 'choice_start.pkl',
        'lick_start': 'lick_start.pkl'
    },
    'detection_thresholds': {
        'response_threshold': 0.05,      # p-value threshold for significance
        'min_effect_size': 0.2,          # Minimum Cohen's d for response
        'baseline_window_pct': 0.4,      # % of window for baseline (before event)
        'response_window_pct': 0.4,      # % of window for response (after event)
        'min_trials': 10,                # Minimum trials for detection
        'min_isi_trials': 5,             # Minimum trials for ISI analysis
        'isi_sensitivity_threshold': 0.05,  # p-value for ISI sensitivity
    },
    'isi_analysis': {
        'f1_offset_time': 0.5,           # Estimated F1 offset time in trial_start data
        'f2_onset_time_fallback': 2.0,   # Fallback F2 onset time if not in data
        'baseline_duration': 0.3,        # Baseline duration before F1 offset (seconds)
        'isi_buffer': 0.1,               # Buffer to avoid event responses (seconds)
    },
    'roi_grouping': {
        'intersection_threshold': 3,      # Min ROIs for intersections
    },
    'plotting': {
        'target_isis': [200, 700, 1700], # ISI values for raster plots
        'isi_tolerance': 50,             # ISI tolerance for trial filtering (ms)
        'min_trials_for_plot': 5,       # Min trials needed for raster plot
        'max_rois_per_group': 10,       # Max ROIs to show per group in raster
    }
}






# %%
# Run the analysis with your loaded data
# ...existing code...

# Run the analysis with your loaded data
try:
    print("=== RUNNING EVENT-ALIGNED ROI ANALYSIS ===")
    
    # Prepare event data list in the order specified by cfg
    event_data_list = [
        trial_data_trial_start,     # For ISI analysis and raster plots
        trial_data_start_flash_1,   # F1 onset
        trial_data_end_flash_1,     # F1 offset
        trial_data_start_flash_2,   # F2 onset
        trial_data_end_flash_2,     # F2 offset
        trial_data_choice_start,    # Choice start
        trial_data_lick_start       # Lick start
    ]
    
    # Run complete analysis
    event_results = run_event_aligned_analysis(event_data_list, cfg_event_aligned)
    
    if 'error' in event_results:
        print(f"Analysis failed: {event_results['error']}")
    else:
        # Display results
        print("\n=== ANALYSIS RESULTS ===")
        roi_groups = event_results['roi_groups']
        
        for group_name, roi_indices in roi_groups.items():
            print(f"{group_name}: {len(roi_indices)} ROIs")
            if len(roi_indices) > 0:
                # Fix: roi_indices is already a list, no need for .tolist()
                sample_rois = roi_indices[:5]
                print(f"  Sample ROIs: {sample_rois}{'...' if len(roi_indices) > 5 else ''}")
        
        # Display raster plots
        if 'raster_figures' in event_results:
            for fig_name, fig in event_results['raster_figures'].items():
                fig.show()
                print(f"Displaying {fig_name}")
        
        plt.show()
        
        print("=== ANALYSIS COMPLETE ===")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()







# %%
    
# Run the analysis pipeline

   
   

    
# %%


try:
    print('holder')
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()