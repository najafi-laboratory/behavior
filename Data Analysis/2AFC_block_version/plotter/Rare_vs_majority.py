import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import matplotlib.gridspec as gridspec
import os

def identify_rare_vs_majority_trials(trial_types, block_types):
    """
    Identify rare vs majority trials within each block type.
    
    Parameters:
    trial_types: list/array where 1=short, 2=long
    block_types: list/array where 0=neutral, 1=short block, 2=long block
    
    Returns:
    dict: Contains 'rare_mask' and 'majority_mask' boolean arrays
    """
    trial_types = np.array(trial_types)
    block_types = np.array(block_types)
    
    rare_mask = np.zeros(len(trial_types), dtype=bool)
    majority_mask = np.zeros(len(trial_types), dtype=bool)
    
    # Short blocks: rare=long trials (2), majority=short trials (1)
    short_block_mask = block_types == 1
    rare_mask |= short_block_mask & (trial_types == 2)  # Rare: long trials in short blocks
    majority_mask |= short_block_mask & (trial_types == 1)  # Majority: short trials in short blocks
    
    # Long blocks: rare=short trials (1), majority=long trials (2)
    long_block_mask = block_types == 2
    rare_mask |= long_block_mask & (trial_types == 1)  # Rare: short trials in long blocks
    majority_mask |= long_block_mask & (trial_types == 2)  # Majority: long trials in long blocks
    
    return {
        'rare_mask': rare_mask,
        'majority_mask': majority_mask
    }

def compute_rare_vs_majority_fractions_from_sessions(sessions_data, pool_blocks=True):
    """
    Compute fraction correct for rare vs majority trials for each session (pooled across block types).
    
    Parameters:
    sessions_data: dict with keys 'outcomes', 'trial_types', 'block_type', 'dates'
    pool_blocks: if True, pool rare and majority trials from short and long blocks
    
    Returns:
    DataFrame with columns: session, trial_category, fraction_correct, n_trials
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
        if len(outcomes) < 2:
            continue
            
        # Identify rare vs majority trials
        trial_masks = identify_rare_vs_majority_trials(trial_types, block_types)
        
        # Process rare and majority trials
        for trial_category, mask in trial_masks.items():
            if not mask.any():  # Skip if no trials found
                continue
                
            category_label = 'rare' if 'rare' in trial_category else 'majority'
            category_indices = np.where(mask)[0]
            
            if len(category_indices) > 0:
                category_outcomes = outcomes_binary[category_indices]
                
                # Only include completed trials (not NaN)
                valid_mask = ~np.isnan(category_outcomes)
                completed_outcomes = category_outcomes[valid_mask]
                
                if len(completed_outcomes) > 0:
                    fraction_correct = completed_outcomes.mean()
                    n_trials = len(completed_outcomes)
                    
                    results.append({
                        'session': session_idx,
                        'date': date,
                        'trial_category': category_label,
                        'fraction_correct': fraction_correct,
                        'n_trials': n_trials
                    })
    
    return pd.DataFrame(results)

def compute_rare_vs_majority_fractions_by_block_type(sessions_data, block_analysis='short'):
    """
    Compute fraction correct for rare vs majority trials, separated by block type.
    
    Parameters:
    sessions_data: dict with keys 'outcomes', 'trial_types', 'block_type', 'dates'
    block_analysis: 'short' for short blocks only, 'long' for long blocks only
    
    Returns:
    DataFrame with columns: session, trial_category, fraction_correct, n_trials, block_type
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
        
        if len(outcomes) < 2:
            continue
        
        # Filter for specific block type
        if block_analysis == 'short':
            block_mask = np.array(block_types) == 1
            rare_trial_type = 2  # Long trials are rare in short blocks
            majority_trial_type = 1  # Short trials are majority in short blocks
        else:  # 'long'
            block_mask = np.array(block_types) == 2
            rare_trial_type = 1  # Short trials are rare in long blocks
            majority_trial_type = 2  # Long trials are majority in long blocks
        
        if not block_mask.any():
            continue
        
        # Create masks for rare and majority trials in this block type
        rare_mask = block_mask & (np.array(trial_types) == rare_trial_type)
        majority_mask = block_mask & (np.array(trial_types) == majority_trial_type)
        
        # Process rare and majority trials
        trial_categories = {
            'rare': rare_mask,
            'majority': majority_mask
        }
        
        for category_label, mask in trial_categories.items():
            category_indices = np.where(mask)[0]
            
            if len(category_indices) > 0:
                category_outcomes = outcomes_binary[category_indices]
                
                # Only include completed trials
                valid_mask = ~np.isnan(category_outcomes)
                completed_outcomes = category_outcomes[valid_mask]
                
                if len(completed_outcomes) > 0:
                    fraction_correct = completed_outcomes.mean()
                    n_trials = len(completed_outcomes)
                    
                    results.append({
                        'session': session_idx,
                        'date': date,
                        'trial_category': category_label,
                        'fraction_correct': fraction_correct,
                        'n_trials': n_trials,
                        'block_type': block_analysis
                    })
    
    return pd.DataFrame(results)

def plot_all_rare_vs_majority_analysis(sessions_data, subject, data_paths, save_path=None):
    """
    Plot all three analyses (pooled, short blocks, long blocks) in a 3x2 grid with summary stats.
    Shows rare vs majority trial comparison using KDE line plots for distributions.
    
    Parameters:
    sessions_data: dict from prepare_session_data function
    subject: subject identifier
    data_paths: list of data file paths
    save_path: optional path to save the figure
    """
    
    # Compute all three analyses
    pooled_results = compute_rare_vs_majority_fractions_from_sessions(sessions_data, pool_blocks=True)
    short_results = compute_rare_vs_majority_fractions_by_block_type(sessions_data, block_analysis='short')
    long_results = compute_rare_vs_majority_fractions_by_block_type(sessions_data, block_analysis='long')
    
    # Debug: Print results info
    print(f"Pooled results: {len(pooled_results)} rows, columns: {list(pooled_results.columns) if len(pooled_results) > 0 else 'Empty'}")
    print(f"Short results: {len(short_results)} rows, columns: {list(short_results.columns) if len(short_results) > 0 else 'Empty'}")
    print(f"Long results: {len(long_results)} rows, columns: {list(long_results.columns) if len(long_results) > 0 else 'Empty'}")
    
    # Create empty DataFrame with correct structure if results are empty
    empty_df = pd.DataFrame(columns=['session', 'date', 'trial_category', 'fraction_correct', 'n_trials'])
    
    if len(pooled_results) == 0:
        pooled_results = empty_df.copy()
    if len(short_results) == 0:
        short_results = empty_df.copy()
    if len(long_results) == 0:
        long_results = empty_df.copy()
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.4], hspace=0.3, wspace=0.3)
    
    # Define trial categories and colors
    categories = ['rare', 'majority']
    colors = ['red', 'blue']
    category_labels = ['Rare Trials', 'Standard Trials']
    
    # Analysis results and titles
    analyses = [
        (pooled_results, "Pooled: Short & Long Blocks"),
        (short_results, "Short Blocks Only\n(Rare=Long, Standard=Short)"),
        (long_results, "Long Blocks Only\n(Rare=Short, Standard=Long)")
    ]
    
    for col, (fraction_data, title) in enumerate(analyses):
        
        # Top row: KDE line plot for distribution
        ax_density = fig.add_subplot(gs[0, col])
        
        for category, color, label in zip(categories, colors, category_labels):
            # Check if DataFrame has the trial_category column and data
            if 'trial_category' in fraction_data.columns and len(fraction_data) > 0:
                category_data = fraction_data[fraction_data['trial_category'] == category]['fraction_correct']
            else:
                category_data = pd.Series(dtype=float)  # Empty series
            
            if len(category_data) > 1:  # Need at least 2 points for KDE
                try:
                    # Check if data has sufficient variance
                    if category_data.std() > 1e-10:  # Has meaningful variance
                        # Plot KDE
                        sns.kdeplot(data=category_data, ax=ax_density, color=color, 
                                    label=f"{label} (n={len(category_data)})", 
                                    linewidth=2, fill=False, alpha = 1)
                        # Add mean line
                        mean_val = category_data.mean()
                        ax_density.axvline(mean_val, color=color, linestyle='--', alpha=0.7)
                    else:
                        # All values are nearly identical - show as single vertical line
                        mean_val = category_data.mean()
                        ax_density.axvline(mean_val, color=color, label=f"{label} (n={len(category_data)})", 
                                         linewidth=3, alpha=0.7)
                except Exception:
                    # Fallback for any issues - show as vertical line
                    mean_val = category_data.mean()
                    ax_density.axvline(mean_val, color=color, label=f"{label} (n={len(category_data)})", 
                                      linewidth=3, alpha=0.7)
            elif len(category_data) == 1:
                # Single point - show as vertical line
                ax_density.axvline(category_data.iloc[0], color=color, label=label, 
                                  linewidth=3, alpha=0.7)
        
        ax_density.set_xlabel('Fraction Correct')
        ax_density.set_ylabel('Density')
        ax_density.set_title(f'{title}\nDistribution Across Sessions')
        ax_density.legend(fontsize=8)
        ax_density.set_xlim(0, 1)
        ax_density.spines['top'].set_visible(False)
        ax_density.spines['right'].set_visible(False)
        
        # Middle row: Cumulative distributions
        ax_cumulative = fig.add_subplot(gs[1, col])
        
        for category, color, label in zip(categories, colors, category_labels):
            if 'trial_category' in fraction_data.columns and len(fraction_data) > 0:
                category_data = fraction_data[fraction_data['trial_category'] == category]['fraction_correct']
            else:
                category_data = pd.Series(dtype=float)  # Empty series
            
            if len(category_data) > 0:
                # Sort data for cumulative plot
                sorted_data = np.sort(category_data)
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
        summary_text.append(f"Summary Statistics - {title.split(chr(10))[0]}")
        summary_text.append("=" * 40)
        
        for category in categories:
            if 'trial_category' in fraction_data.columns and len(fraction_data) > 0:
                category_data = fraction_data[fraction_data['trial_category'] == category]['fraction_correct']
            else:
                category_data = pd.Series(dtype=float)  # Empty series
                
            if len(category_data) > 0:
                summary_text.append(f"\n{category.upper()} TRIALS:")
                summary_text.append(f"  N sessions: {len(category_data)}")
                summary_text.append(f"  Mean: {category_data.mean():.4f}")
                summary_text.append(f"  Std: {category_data.std():.4f}")
                summary_text.append(f"  Median: {category_data.median():.4f}")
                summary_text.append(f"  Range: {category_data.min():.4f} - {category_data.max():.4f}")
            else:
                summary_text.append(f"\n{category.upper()} TRIALS: No data")
        
        # Add statistical comparison if both categories have data
        if 'trial_category' in fraction_data.columns and len(fraction_data) > 0:
            rare_data = fraction_data[fraction_data['trial_category'] == 'rare']['fraction_correct']
            majority_data = fraction_data[fraction_data['trial_category'] == 'majority']['fraction_correct']
            
            if len(rare_data) > 1 and len(majority_data) > 1:
                try:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(rare_data, majority_data)
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(rare_data)-1)*rare_data.var() + (len(majority_data)-1)*majority_data.var()) / (len(rare_data) + len(majority_data) - 2))
                    cohens_d = (rare_data.mean() - majority_data.mean()) / pooled_std
                    
                    summary_text.append(f"\nSTATISTICAL COMPARISON:")
                    summary_text.append(f"  t-statistic: {t_stat:.4f}")
                    summary_text.append(f"  p-value: {p_value:.4f}")
                    summary_text.append(f"  Effect size (Cohen's d): {cohens_d:.4f}")
                    summary_text.append(f"  Mean difference: {rare_data.mean() - majority_data.mean():.4f}")
                except Exception:
                    summary_text.append(f"\nSTATISTICAL COMPARISON: Could not compute")
        
        # Add text to subplot
        text_content = '\n'.join(summary_text)
        ax_text.text(0.05, 0.95, text_content, transform=ax_text.transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Rare vs Standard Trial Analysis: Fraction Correct Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        output_path = os.path.join(save_path, f'Rare_vs_Standard_Analysis_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return pooled_results, short_results, long_results