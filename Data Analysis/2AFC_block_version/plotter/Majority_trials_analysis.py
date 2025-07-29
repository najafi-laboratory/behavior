import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import matplotlib.gridspec as gridspec
import os

def identify_majority_trials_by_epoch(trial_types, block_types):
    """
    Identify majority trials in early vs late epochs within each block.
    
    Parameters:
    trial_types: list/array where 1=short, 2=long
    block_types: list/array where 0=neutral, 1=short block, 2=long block
    
    Returns:
    dict: Contains 'early_majority_mask' and 'late_majority_mask' boolean arrays
    """
    trial_types = np.array(trial_types)
    block_types = np.array(block_types)
    
    early_majority_mask = np.zeros(len(trial_types), dtype=bool)
    late_majority_mask = np.zeros(len(trial_types), dtype=bool)
    
    # Process each block separately
    unique_blocks = np.unique(block_types)
    
    for block_type in unique_blocks:
        if block_type == 0:  # Skip neutral blocks
            continue
            
        # Find all trials in this block type
        block_mask = block_types == block_type
        block_indices = np.where(block_mask)[0]
        
        if len(block_indices) == 0:
            continue
            
        # Find continuous segments of this block type
        block_segments = []
        current_segment = [block_indices[0]]
        
        for i in range(1, len(block_indices)):
            if block_indices[i] == block_indices[i-1] + 1:
                current_segment.append(block_indices[i])
            else:
                block_segments.append(current_segment)
                current_segment = [block_indices[i]]
        block_segments.append(current_segment)
        
        # For each continuous block segment, split into early/late halves
        for segment in block_segments:
            if len(segment) < 2:  # Need at least 2 trials to split
                continue
                
            segment = np.array(segment)
            mid_point = len(segment) // 2
            
            early_indices = segment[:mid_point]
            late_indices = segment[mid_point:]
            
            # Define majority trials based on block type
            if block_type == 1:  # Short block - majority are short trials (trial_type=1)
                majority_trial_type = 1
            elif block_type == 2:  # Long block - majority are long trials (trial_type=2)
                majority_trial_type = 2
            else:
                continue
            
            # Mark early majority trials
            for idx in early_indices:
                if trial_types[idx] == majority_trial_type:
                    early_majority_mask[idx] = True
            
            # Mark late majority trials  
            for idx in late_indices:
                if trial_types[idx] == majority_trial_type:
                    late_majority_mask[idx] = True
    
    return {
        'early_majority_mask': early_majority_mask,
        'late_majority_mask': late_majority_mask
    }

def compute_majority_trial_fractions_from_sessions(sessions_data, pool_blocks=True):
    """
    Compute fraction correct for majority trials in early vs late epochs for each session.
    
    Parameters:
    sessions_data: dict with keys 'outcomes', 'trial_types', 'block_type', 'dates'
    pool_blocks: if True, pool majority trials from short and long blocks
    
    Returns:
    DataFrame with columns: session, epoch, fraction_correct, n_trials
    """
    
    results = []
    n_sessions = len(sessions_data['outcomes'])
    
    for session_idx in range(n_sessions):
        outcomes = sessions_data['outcomes'][session_idx]
        trial_types = sessions_data['trial_types'][session_idx]
        block_types = sessions_data['block_type'][session_idx]
        date = sessions_data['dates'][session_idx]
        
        # Convert outcomes to binary (1 for correct, 0 for incorrect)
        outcomes_binary = np.array([1 if outcome == 'Reward' else 0 for outcome in outcomes])
        
        # Skip sessions with insufficient trials
        if len(outcomes) < 4:
            continue
            
        # Identify majority trials by epoch
        epoch_masks = identify_majority_trials_by_epoch(trial_types, block_types)
        
        # Process early and late epochs
        for epoch_name, mask in epoch_masks.items():
            if not mask.any():  # Skip if no trials found
                continue
                
            epoch_label = 'early' if 'early' in epoch_name else 'late'
            epoch_indices = np.where(mask)[0]
            
            if len(epoch_indices) > 0:
                epoch_outcomes = outcomes_binary[epoch_indices]
                
                # Only include completed trials (not NaN)
                valid_mask = ~np.isnan(epoch_outcomes)
                completed_outcomes = epoch_outcomes[valid_mask]
                
                if len(completed_outcomes) > 0:
                    fraction_correct = completed_outcomes.mean()
                    n_trials = len(completed_outcomes)
                    
                    results.append({
                        'session': session_idx,
                        'date': date,
                        'epoch': epoch_label,
                        'fraction_correct': fraction_correct,
                        'n_trials': n_trials
                    })
    
    return pd.DataFrame(results)

def compute_majority_trial_fractions_by_block_type(sessions_data, block_analysis='short'):
    """
    Compute fraction correct for majority trials in early vs late epochs, separated by block type.
    
    Parameters:
    sessions_data: dict with keys 'outcomes', 'trial_types', 'block_type', 'dates'
    block_analysis: 'short' for short blocks only, 'long' for long blocks only
    
    Returns:
    DataFrame with columns: session, epoch, fraction_correct, n_trials, block_type
    """
    
    results = []
    n_sessions = len(sessions_data['outcomes'])
    
    for session_idx in range(n_sessions):
        outcomes = sessions_data['outcomes'][session_idx]
        trial_types = sessions_data['trial_types'][session_idx]
        block_types = sessions_data['block_type'][session_idx]
        date = sessions_data['dates'][session_idx]
        
        # Convert outcomes to binary
        outcomes_binary = np.array([1 if outcome == 'Reward' else 0 for outcome in outcomes])
        
        if len(outcomes) < 4:
            continue
        
        # Filter for specific block type
        if block_analysis == 'short':
            target_block_type = 1
            majority_trial_type = 1  # Short trials are majority in short blocks
        else:  # 'long'
            target_block_type = 2
            majority_trial_type = 2  # Long trials are majority in long blocks
        
        # Create filtered data for this block type only
        block_mask = np.array(block_types) == target_block_type
        
        if not block_mask.any():
            continue
        
        # Get epoch masks for the specific block type
        epoch_masks = identify_majority_trials_by_epoch(trial_types, block_types)
        
        # Process early and late epochs for this block type
        for epoch_name, mask in epoch_masks.items():
            epoch_label = 'early' if 'early' in epoch_name else 'late'
            
            # Combine epoch mask with block type mask and majority trial type
            combined_mask = mask & block_mask & (np.array(trial_types) == majority_trial_type)
            epoch_indices = np.where(combined_mask)[0]
            
            if len(epoch_indices) > 0:
                epoch_outcomes = outcomes_binary[epoch_indices]
                
                # Only include completed trials
                valid_mask = ~np.isnan(epoch_outcomes)
                completed_outcomes = epoch_outcomes[valid_mask]
                
                if len(completed_outcomes) > 0:
                    fraction_correct = completed_outcomes.mean()
                    n_trials = len(completed_outcomes)
                    
                    results.append({
                        'session': session_idx,
                        'date': date,
                        'epoch': epoch_label,
                        'fraction_correct': fraction_correct,
                        'n_trials': n_trials,
                        'block_type': block_analysis
                    })
    
    return pd.DataFrame(results)

def plot_all_majority_trial_analysis(sessions_data, subject, data_paths, save_path=None):
    """
    Plot all three analyses (pooled, short blocks, long blocks) in a 3x2 grid with summary stats.
    Shows early vs late epoch comparison for majority trials.
    
    Parameters:
    sessions_data: dict from prepare_session_data function
    save_path: optional path to save the figure
    """
    
    # Compute all three analyses
    pooled_results = compute_majority_trial_fractions_from_sessions(sessions_data, pool_blocks=True)
    short_results = compute_majority_trial_fractions_by_block_type(sessions_data, block_analysis='short')
    long_results = compute_majority_trial_fractions_by_block_type(sessions_data, block_analysis='long')
    
    # Debug: Print results info
    print(f"Pooled results: {len(pooled_results)} rows, columns: {list(pooled_results.columns) if len(pooled_results) > 0 else 'Empty'}")
    print(f"Short results: {len(short_results)} rows, columns: {list(short_results.columns) if len(short_results) > 0 else 'Empty'}")
    print(f"Long results: {len(long_results)} rows, columns: {list(long_results.columns) if len(long_results) > 0 else 'Empty'}")
    
    # Create empty DataFrame with correct structure if results are empty
    empty_df = pd.DataFrame(columns=['session', 'date', 'epoch', 'fraction_correct', 'n_trials'])
    
    if len(pooled_results) == 0:
        pooled_results = empty_df.copy()
    if len(short_results) == 0:
        short_results = empty_df.copy()
    if len(long_results) == 0:
        long_results = empty_df.copy()
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.4], hspace=0.3, wspace=0.3)
    
    # Define epochs and colors
    epochs = ['early', 'late']
    colors = ['purple', 'orange']
    epoch_labels = ['Early Epoch (1st half)', 'Late Epoch (2nd half)']
    
    # Analysis results and titles
    analyses = [
        (pooled_results, "Pooled: Short & Long Blocks"),
        (short_results, "Short Blocks Only"),
        (long_results, "Long Blocks Only")
    ]
    
    for col, (fraction_data, title) in enumerate(analyses):
        
        # Top row: Probability density
        ax_density = fig.add_subplot(gs[0, col])
        
        for epoch, color, label in zip(epochs, colors, epoch_labels):
            # Check if DataFrame has the epoch column and data
            if 'epoch' in fraction_data.columns and len(fraction_data) > 0:
                epoch_data = fraction_data[fraction_data['epoch'] == epoch]['fraction_correct']
            else:
                epoch_data = pd.Series(dtype=float)  # Empty series
            
            if len(epoch_data) > 1:  # Need at least 2 points for histogram
                try:
                    # Check if data has sufficient variance
                    if epoch_data.std() > 1e-10:  # Has meaningful variance
                        # Create normalized histogram (frequencies sum to 1)
                        ax_density.hist(epoch_data, bins=15, alpha=0.6, color=color, 
                                    label=f"{label} (n={len(epoch_data)})", 
                                    density=True, histtype='stepfilled', edgecolor='black', linewidth=0.5)
                                                
                        # Add mean line
                        mean_val = epoch_data.mean()
                        ax_density.axvline(mean_val, color=color, linestyle='--', alpha=0.7)
                    else:
                        # All values are nearly identical - show as single vertical line
                        mean_val = epoch_data.mean()
                        ax_density.axvline(mean_val, color=color, label=f"{label} (n={len(epoch_data)})", 
                                         linewidth=3, alpha=0.7)
                        
                except Exception:
                    # Fallback for any issues - use histogram
                    hist_values, bin_edges = np.histogram(epoch_data, bins=min(10, len(epoch_data)), density=True)
                    normalized_hist = hist_values / hist_values.max() if hist_values.max() > 0 else hist_values
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ax_density.plot(bin_centers, normalized_hist, color=color, label=f"{label} (histogram)", 
                                linewidth=2, drawstyle='steps-mid')
                    
                    # Add mean line
                    mean_val = epoch_data.mean()
                    ax_density.axvline(mean_val, color=color, linestyle='--', alpha=0.7)
                    
            elif len(epoch_data) == 1:
                # Single point - show as vertical line
                ax_density.axvline(epoch_data.iloc[0], color=color, label=label, linewidth=3, alpha=0.7)
        
        ax_density.set_xlabel('Fraction Correct')
        ax_density.set_ylabel('Probability')
        ax_density.set_title(f'{title}\nDistribution Across Sessions')
        ax_density.legend(fontsize=8)
        ax_density.set_xlim(0, 1)
        ax_density.spines['top'].set_visible(False)
        ax_density.spines['right'].set_visible(False)
        
        # Middle row: Cumulative distributions
        ax_cumulative = fig.add_subplot(gs[1, col])
        
        for epoch, color, label in zip(epochs, colors, epoch_labels):
            # Check if DataFrame has the epoch column and data
            if 'epoch' in fraction_data.columns and len(fraction_data) > 0:
                epoch_data = fraction_data[fraction_data['epoch'] == epoch]['fraction_correct']
            else:
                epoch_data = pd.Series(dtype=float)  # Empty series
            
            if len(epoch_data) > 0:
                # Sort data for cumulative plot
                sorted_data = np.sort(epoch_data)
                n = len(sorted_data)
                cumulative = np.arange(1, n + 1) / n
                
                ax_cumulative.plot(sorted_data, cumulative, color=color, label=label, 
                                 linewidth=2, marker='o', markersize=4)
        
        ax_cumulative.set_xlabel('Fraction Correct')
        ax_cumulative.set_ylabel('Cumulative Probability')
        ax_cumulative.set_title('Cumulative Distribution')
        ax_cumulative.legend(fontsize=8)
        ax_cumulative.set_xlim(0, 1)
        ax_cumulative.set_ylim(-0.1, 1.1)
        ax_cumulative.spines['top'].set_visible(False)
        ax_cumulative.spines['right'].set_visible(False)
        
        # Bottom row: Summary statistics text
        ax_text = fig.add_subplot(gs[2, col])
        ax_text.axis('off')
        
        # Create summary statistics text
        summary_text = []
        summary_text.append(f"Summary Statistics - {title}")
        summary_text.append("=" * 40)
        
        for epoch in epochs:
            if 'epoch' in fraction_data.columns and len(fraction_data) > 0:
                epoch_data = fraction_data[fraction_data['epoch'] == epoch]['fraction_correct']
            else:
                epoch_data = pd.Series(dtype=float)  # Empty series
                
            if len(epoch_data) > 0:
                summary_text.append(f"\n{epoch.upper()} EPOCH:")
                summary_text.append(f"  N sessions: {len(epoch_data)}")
                summary_text.append(f"  Mean: {epoch_data.mean():.4f}")
                summary_text.append(f"  Std: {epoch_data.std():.4f}")
                summary_text.append(f"  Median: {epoch_data.median():.4f}")
                summary_text.append(f"  Range: {epoch_data.min():.4f} - {epoch_data.max():.4f}")
            else:
                summary_text.append(f"\n{epoch.upper()} EPOCH: No data")
        
        # Add statistical comparison if both epochs have data
        if 'epoch' in fraction_data.columns and len(fraction_data) > 0:
            early_data = fraction_data[fraction_data['epoch'] == 'early']['fraction_correct']
            late_data = fraction_data[fraction_data['epoch'] == 'late']['fraction_correct']
            
            if len(early_data) > 1 and len(late_data) > 1:
                try:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(early_data, late_data)
                    summary_text.append(f"\nSTATISTICAL COMPARISON:")
                    summary_text.append(f"  t-statistic: {t_stat:.4f}")
                    summary_text.append(f"  p-value: {p_value:.4f}")
                    summary_text.append(f"  Effect size (Cohen's d): {(early_data.mean() - late_data.mean()) / np.sqrt(((len(early_data)-1)*early_data.var() + (len(late_data)-1)*late_data.var()) / (len(early_data) + len(late_data) - 2)):.4f}")
                except Exception:
                    summary_text.append(f"\nSTATISTICAL COMPARISON: Could not compute")
        
        # Add text to subplot
        text_content = '\n'.join(summary_text)
        ax_text.text(0.05, 0.95, text_content, transform=ax_text.transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Majority Trial Analysis: Early vs Late Epoch Fraction Correct', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        output_path = os.path.join(save_path, f'Majority_Trial_Analysis_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return pooled_results, short_results, long_results