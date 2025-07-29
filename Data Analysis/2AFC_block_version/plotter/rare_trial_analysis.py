import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import matplotlib.gridspec as gridspec
import os

def identify_rare_trials(trial_types, block_types):
    """
    Identify rare trials based on block type and trial type.
    
    Parameters:
    trial_types: list/array where 1=short, 2=long
    block_types: list/array where 0=neutral, 1=short block, 2=long block
    
    Returns:
    numpy array: Boolean mask indicating rare trials
    """
    trial_types = np.array(trial_types)
    block_types = np.array(block_types)
    
    # Short rare trials: block_type=2 (long block) AND trial_type=1 (short trial)
    # Long rare trials: block_type=1 (short block) AND trial_type=2 (long trial)
    rare_mask = ((block_types == 2) & (trial_types == 1)) | \
                ((block_types == 1) & (trial_types == 2))
    
    return rare_mask

def compute_rare_trial_fractions_from_sessions(sessions_data, pool_blocks=True):
    """
    Compute fraction correct for rare-1, rare, and rare+1 trials for each session.
    
    Parameters:
    sessions_data: dict with keys 'outcomes', 'trial_types', 'block_type', 'dates'
    pool_blocks: if True, pool rare trials from short and long blocks
    
    Returns:
    DataFrame with columns: session, condition, fraction_correct, n_trials
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
        if len(outcomes) < 3:
            continue
            
        # Identify rare trials
        rare_mask = identify_rare_trials(trial_types, block_types)
        rare_indices = np.where(rare_mask)[0]
        
        if len(rare_indices) == 0:
            continue
            
        # Get rare-1, rare, and rare+1 trials
        conditions = {
            'rare-1': [],
            'rare': [],
            'rare+1': []
        }
        
        for rare_idx in rare_indices:
            # rare-1 (trial before rare trial)
            if rare_idx > 0:
                conditions['rare-1'].append(rare_idx - 1)
            
            # rare (the rare trial itself)
            conditions['rare'].append(rare_idx)
            
            # rare+1 (trial after rare trial)
            if rare_idx < len(outcomes) - 1:
                conditions['rare+1'].append(rare_idx + 1)
        
        # Compute fraction correct for each condition
        for condition, indices in conditions.items():
            if len(indices) > 0:
                condition_outcomes = outcomes_binary[indices]
                
                # Only include completed trials (not NaN)
                valid_mask = ~np.isnan(condition_outcomes)
                completed_outcomes = condition_outcomes[valid_mask]
                
                if len(completed_outcomes) > 0:
                    fraction_correct = completed_outcomes.mean()
                    n_trials = len(completed_outcomes)
                    
                    results.append({
                        'session': session_idx,
                        'date': date,
                        'condition': condition,
                        'fraction_correct': fraction_correct,
                        'n_trials': n_trials
                    })
    
    return pd.DataFrame(results)

def compute_rare_trial_fractions_by_block_type(sessions_data, block_analysis='short'):
    """
    Compute fraction correct for rare trials separated by block type.
    
    Parameters:
    sessions_data: dict with keys 'outcomes', 'trial_types', 'block_type', 'dates'
    block_analysis: 'short' for short blocks only, 'long' for long blocks only
    
    Returns:
    DataFrame with columns: session, condition, fraction_correct, n_trials
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
        
        if len(outcomes) < 3:
            continue
        
        # Filter for specific block type and identify rare trials
        if block_analysis == 'short':
            # Short blocks (block_type=1) with long rare trials (trial_type=2)
            block_mask = np.array(block_types) == 1
            rare_mask = (np.array(block_types) == 1) & (np.array(trial_types) == 2)
        else:  # 'long'
            # Long blocks (block_type=2) with short rare trials (trial_type=1)
            block_mask = np.array(block_types) == 2
            rare_mask = (np.array(block_types) == 2) & (np.array(trial_types) == 1)
        
        rare_indices = np.where(rare_mask)[0]
        
        if len(rare_indices) == 0:
            continue
        
        # Get rare-1, rare, and rare+1 trials
        conditions = {
            'rare-1': [],
            'rare': [],
            'rare+1': []
        }
        
        for rare_idx in rare_indices:
            # rare-1 (trial before rare trial) - must also be in same block type
            if rare_idx > 0 and block_mask[rare_idx - 1]:
                conditions['rare-1'].append(rare_idx - 1)
            
            # rare (the rare trial itself)
            conditions['rare'].append(rare_idx)
            
            # rare+1 (trial after rare trial) - must also be in same block type
            if rare_idx < len(outcomes) - 1 and block_mask[rare_idx + 1]:
                conditions['rare+1'].append(rare_idx + 1)
        
        # Compute fraction correct for each condition
        for condition, indices in conditions.items():
            if len(indices) > 0:
                condition_outcomes = outcomes_binary[indices]
                
                # Only include completed trials
                valid_mask = ~np.isnan(condition_outcomes)
                completed_outcomes = condition_outcomes[valid_mask]
                
                if len(completed_outcomes) > 0:
                    fraction_correct = completed_outcomes.mean()
                    n_trials = len(completed_outcomes)
                    
                    results.append({
                        'session': session_idx,
                        'date': date,
                        'condition': condition,
                        'fraction_correct': fraction_correct,
                        'n_trials': n_trials,
                        'block_type': block_analysis
                    })
    
    return pd.DataFrame(results)

def plot_all_rare_trial_analyses(sessions_data, subject, data_paths, save_path=None):
    """
     Plot all three analyses (pooled, short blocks, long blocks) in a 3x2 grid with summary stats.
    
    Parameters:
    sessions_data: dict from prepare_session_data function
    save_path: optional path to save the figure
    """
    
    # Compute all three analyses
    pooled_results = compute_rare_trial_fractions_from_sessions(sessions_data, pool_blocks=True)
    short_results = compute_rare_trial_fractions_by_block_type(sessions_data, block_analysis='short')
    long_results = compute_rare_trial_fractions_by_block_type(sessions_data, block_analysis='long')
    
    # Debug: Print results info
    print(f"Pooled results: {len(pooled_results)} rows, columns: {list(pooled_results.columns) if len(pooled_results) > 0 else 'Empty'}")
    print(f"Short results: {len(short_results)} rows, columns: {list(short_results.columns) if len(short_results) > 0 else 'Empty'}")
    print(f"Long results: {len(long_results)} rows, columns: {list(long_results.columns) if len(long_results) > 0 else 'Empty'}")
    
    # Create empty DataFrame with correct structure if results are empty
    empty_df = pd.DataFrame(columns=['session', 'date', 'condition', 'fraction_correct', 'n_trials'])
    
    if len(pooled_results) == 0:
        pooled_results = empty_df.copy()
    if len(short_results) == 0:
        short_results = empty_df.copy()
    if len(long_results) == 0:
        long_results = empty_df.copy()
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.4], hspace=0.3, wspace=0.3)
    
    # Define conditions and colors
    conditions = ['rare-1', 'rare', 'rare+1']
    colors = ['blue', 'red', 'green']
    condition_labels = ['Control (rare-1)', 'Perturbation (rare)', 'Adaptation (rare+1)']
    
    # Analysis results and titles
    analyses = [
        (pooled_results, "Pooled: Short & Long Blocks"),
        (short_results, "Short Blocks Only"),
        (long_results, "Long Blocks Only")
    ]
    
    for col, (fraction_data, title) in enumerate(analyses):
        
        # Top row: Probability density
        ax_density = fig.add_subplot(gs[0, col])
        
        for condition, color, label in zip(conditions, colors, condition_labels):
            # Check if DataFrame has the condition column and data
            if 'condition' in fraction_data.columns and len(fraction_data) > 0:
                condition_data = fraction_data[fraction_data['condition'] == condition]['fraction_correct']
            else:
                condition_data = pd.Series(dtype=float)  # Empty series
            
            if len(condition_data) > 1:  # Need at least 2 points for KDE
                try:
                    # Check if data has sufficient variance for KDE
                    if condition_data.std() > 1e-10:  # Has meaningful variance
                        # # Kernel density estimation
                        # density = stats.gaussian_kde(condition_data)
                        # x_range = np.linspace(0, 1, 100)
                        # density_values = density(x_range)
                        # normalized_density = density_values / density_values.max()  # Normalize to [0,1]
                        # ax_density.plot(x_range, normalized_density, color=color, label=label, linewidth=2)
                        # Create normalized histogram (frequencies sum to 1)
                        ax_density.hist(condition_data, bins=15, alpha=0.6, color=color, 
                                    label=f"{label} (n={len(condition_data)})", 
                                    density=True, histtype='stepfilled', edgecolor='black', linewidth=0.5)
                                                
                        # Add mean line
                        mean_val = condition_data.mean()
                        ax_density.axvline(mean_val, color=color, linestyle='--', alpha=0.7)
                    else:
                        # All values are nearly identical - show as single vertical line
                        mean_val = condition_data.mean()
                        ax_density.axvline(mean_val, color=color, label=f"{label} (n={len(condition_data)})", 
                                         linewidth=3, alpha=0.7)
                        
                except np.linalg.LinAlgError:
                    # Fallback for singular covariance matrix - use histogram instead
                    hist_values, bin_edges = np.histogram(condition_data, bins=min(10, len(condition_data)), density=True)
                    normalized_hist = hist_values / hist_values.max() if hist_values.max() > 0 else hist_values
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ax_density.plot(bin_centers, normalized_hist, color=color, label=f"{label} (histogram)", 
                                linewidth=2, drawstyle='steps-mid')
                    
                    # Add mean line
                    mean_val = condition_data.mean()
                    ax_density.axvline(mean_val, color=color, linestyle='--', alpha=0.7)
                    
            elif len(condition_data) == 1:
                # Single point - show as vertical line
                ax_density.axvline(condition_data.iloc[0], color=color, label=label, linewidth=3, alpha=0.7)
        
        ax_density.set_xlabel('Fraction Correct')
        ax_density.set_ylabel('Probability')
        ax_density.set_title(f'{title}\nDistribution Across Sessions')
        ax_density.legend(fontsize=8)
        # ax_density.grid(True, alpha=0.3)
        ax_density.set_xlim(0, 1)
        ax_density.spines['top'].set_visible(False)
        ax_density.spines['right'].set_visible(False)
        
        # Middle row: Cumulative distributions
        ax_cumulative = fig.add_subplot(gs[1, col])
        
        for condition, color, label in zip(conditions, colors, condition_labels):
            # Check if DataFrame has the condition column and data
            if 'condition' in fraction_data.columns and len(fraction_data) > 0:
                condition_data = fraction_data[fraction_data['condition'] == condition]['fraction_correct']
            else:
                condition_data = pd.Series(dtype=float)  # Empty series
            
            if len(condition_data) > 0:
                # Sort data for cumulative plot
                sorted_data = np.sort(condition_data)
                n = len(sorted_data)
                cumulative = np.arange(1, n + 1) / n
                
                ax_cumulative.plot(sorted_data, cumulative, color=color, label=label, 
                                 linewidth=2, marker='o', markersize=4)
        
        ax_cumulative.set_xlabel('Fraction Correct')
        ax_cumulative.set_ylabel('Cumulative Probability')
        ax_cumulative.set_title('Cumulative Distribution')
        ax_cumulative.legend(fontsize=8)
        # ax_cumulative.grid(True, alpha=0.3)
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
        
        for condition in conditions:
            if 'condition' in fraction_data.columns and len(fraction_data) > 0:
                condition_data = fraction_data[fraction_data['condition'] == condition]['fraction_correct']
            else:
                condition_data = pd.Series(dtype=float)  # Empty series
                
            if len(condition_data) > 0:
                summary_text.append(f"\n{condition.upper()}:")
                summary_text.append(f"  N sessions: {len(condition_data)}")
                summary_text.append(f"  Mean: {condition_data.mean():.4f}")
                summary_text.append(f"  Std: {condition_data.std():.4f}")
                summary_text.append(f"  Median: {condition_data.median():.4f}")
                summary_text.append(f"  Range: {condition_data.min():.4f} - {condition_data.max():.4f}")
            else:
                summary_text.append(f"\n{condition.upper()}: No data")
        
        # Add text to subplot
        text_content = '\n'.join(summary_text)
        ax_text.text(0.05, 0.95, text_content, transform=ax_text.transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Rare Trial Analysis: Fraction Correct Across All Conditions', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        output_path = os.path.join(save_path, f'Rare_Trial_Analysis_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return pooled_results, short_results, long_results

