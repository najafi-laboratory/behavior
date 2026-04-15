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
    """
    trial_types = np.array(trial_types)
    block_types = np.array(block_types)
    
    # Short rare trials: block_type=2 (long block) AND trial_type=1 (short trial)
    # Long rare trials: block_type=1 (short block) AND trial_type=2 (long trial)
    rare_mask = ((block_types == 2) & (trial_types == 1)) | \
                ((block_types == 1) & (trial_types == 2))
    
    return rare_mask

def compute_rare_trial_fractions(sessions_data, block_analysis='pooled', filter_mode='all'):
    """
    Compute fraction correct for rare-1, rare, and rare+1 trials.
    
    Parameters:
    sessions_data: dict with session data
    block_analysis: 'pooled', 'short', 'long'
    filter_mode: 
        'control': ALL trials (rare-1, rare, rare+1) must be non-opto.
        'opto': The RARE trial must be opto (neighbors included regardless).
        'all': No opto filtering.
    """
    results = []
    n_sessions = len(sessions_data['outcomes'])
    opto_list = sessions_data.get('opto_tags', [[]] * n_sessions)
    
    for session_idx in range(n_sessions):
        outcomes = sessions_data['outcomes'][session_idx]
        trial_types = sessions_data['trial_types'][session_idx]
        block_types = sessions_data['block_type'][session_idx]
        date = sessions_data['dates'][session_idx]
        opto_raw = opto_list[session_idx]
        
        n_trials = len(outcomes)
        if n_trials < 3:
            continue

        # 1. Clean Opto Tags
        if opto_raw is None or len(opto_raw) == 0:
            opto_clean = np.zeros(n_trials)
        else:
            temp = pd.to_numeric(opto_raw, errors='coerce')
            opto_clean = np.nan_to_num(temp, nan=0.0)
            if len(opto_clean) < n_trials:
                opto_clean = np.pad(opto_clean, (0, n_trials - len(opto_clean)), constant_values=0)
            elif len(opto_clean) > n_trials:
                opto_clean = opto_clean[:n_trials]

        # 2. Identify Rare Trials
        rare_mask = identify_rare_trials(trial_types, block_types)
        
        # 3. Apply Block Filtering
        if block_analysis == 'short':
            block_mask = (np.array(block_types) == 1)
            rare_mask = rare_mask & block_mask
        elif block_analysis == 'long':
            block_mask = (np.array(block_types) == 2)
            rare_mask = rare_mask & block_mask
        
        rare_indices = np.where(rare_mask)[0]
        if len(rare_indices) == 0:
            continue
            
        outcomes_binary = np.array([1 if outcome == 'Reward' else 0 for outcome in outcomes])
        
        # 4. Process Conditions (rare-1, rare, rare+1)
        conditions = {'rare-1': [], 'rare': [], 'rare+1': []}
        
        for rare_idx in rare_indices:
            # Check Indices Validity
            if rare_idx == 0 or rare_idx >= n_trials - 1:
                continue
                
            idx_prev = rare_idx - 1
            idx_curr = rare_idx
            idx_next = rare_idx + 1

            # --- FILTERING LOGIC ---
            if filter_mode == 'control':
                # STRICT: All 3 trials must be non-opto
                if (opto_clean[idx_prev] == 0 and 
                    opto_clean[idx_curr] == 0 and 
                    opto_clean[idx_next] == 0):
                    
                    conditions['rare-1'].append(idx_prev)
                    conditions['rare'].append(idx_curr)
                    conditions['rare+1'].append(idx_next)
                    
            elif filter_mode == 'opto':
                # TARGET: The Rare trial (idx_curr) must be opto. 
                # We don't filter neighbors based on opto.
                if opto_clean[idx_curr] != 0:
                    conditions['rare-1'].append(idx_prev)
                    conditions['rare'].append(idx_curr)
                    conditions['rare+1'].append(idx_next)
                    
            else: # 'all'
                # No filter
                conditions['rare-1'].append(idx_prev)
                conditions['rare'].append(idx_curr)
                conditions['rare+1'].append(idx_next)

        # 5. Compute Fractions
        for condition, indices in conditions.items():
            if len(indices) > 0:
                condition_outcomes = outcomes_binary[indices]
                valid_mask = ~np.isnan(condition_outcomes)
                completed_outcomes = condition_outcomes[valid_mask]
                
                if len(completed_outcomes) > 0:
                    fraction_correct = completed_outcomes.mean()
                    n_count = len(completed_outcomes)
                    
                    results.append({
                        'session': session_idx,
                        'date': date,
                        'condition': condition,
                        'fraction_correct': fraction_correct,
                        'n_trials': n_count
                    })
    
    return pd.DataFrame(results)

def plot_all_rare_trial_analyses(sessions_data, subject, data_paths, save_path=None):
    """
    Plot rare trial analysis in a 9-column grid (Control, Opto, All) x (Pooled, Short, Long).
    """
    
    # Configuration: (Filter Mode, Block Analysis, Title)
    configs = [
        ('control', 'pooled', 'Control: Pooled'),
        ('control', 'short',  'Control: Short Blocks'),
        ('control', 'long',   'Control: Long Blocks'),
        ('opto',    'pooled', 'Opto: Pooled'),
        ('opto',    'short',  'Opto: Short Blocks'),
        ('opto',    'long',   'Opto: Long Blocks'),
        ('all',     'pooled', 'All: Pooled'),
        ('all',     'short',  'All: Short Blocks'),
        ('all',     'long',   'All: Long Blocks')
    ]
    
    # Create figure with GridSpec (9 columns)
    fig = plt.figure(figsize=(40, 12))
    gs = gridspec.GridSpec(3, 9, height_ratios=[1, 1, 0.6], hspace=0.3, wspace=0.3)
    
    conditions = ['rare-1', 'rare', 'rare+1']
    colors = ['blue', 'red', 'green']
    condition_labels = ['Control (rare-1)', 'Perturbation (rare)', 'Adaptation (rare+1)']
    
    for col_idx, (filt_mode, block_cond, title) in enumerate(configs):
        
        # Compute data for this column
        fraction_data = compute_rare_trial_fractions(sessions_data, block_analysis=block_cond, filter_mode=filt_mode)
        
        # --- Row 0: Density Plot (KDE) ---
        ax_density = fig.add_subplot(gs[0, col_idx])
        
        for condition, color, label in zip(conditions, colors, condition_labels):
            if 'condition' in fraction_data.columns and len(fraction_data) > 0:
                condition_data = fraction_data[fraction_data['condition'] == condition]['fraction_correct']
            else:
                condition_data = pd.Series(dtype=float)
            
            if len(condition_data) > 1:
                try:
                    if condition_data.std() > 1e-10:
                        sns.kdeplot(data=condition_data, ax=ax_density, color=color, 
                                    label=f"{label} (n={len(condition_data)})", 
                                    linewidth=2, fill=False)
                        ax_density.axvline(condition_data.mean(), color=color, linestyle='--', alpha=0.7)
                    else:
                        ax_density.axvline(condition_data.mean(), color=color, label=f"{label} (n={len(condition_data)})", 
                                           linewidth=3, alpha=0.7)
                except Exception:
                    ax_density.axvline(condition_data.mean(), color=color, label=f"{label} (n={len(condition_data)})", 
                                       linewidth=3, alpha=0.7)
            elif len(condition_data) == 1:
                ax_density.axvline(condition_data.iloc[0], color=color, label=label, linewidth=3, alpha=0.7)
        
        ax_density.set_title(title, fontsize=10, fontweight='bold')
        if col_idx == 0: ax_density.set_ylabel('Density')
        else: ax_density.set_ylabel('')
        ax_density.set_xlabel('')
        ax_density.set_xlim(0, 1)
        if col_idx == 0: ax_density.legend(fontsize=6, loc='upper left')
        ax_density.spines['top'].set_visible(False)
        ax_density.spines['right'].set_visible(False)
        
        # --- Row 1: Cumulative distributions ---
        ax_cumulative = fig.add_subplot(gs[1, col_idx])
        
        for condition, color, label in zip(conditions, colors, condition_labels):
            if 'condition' in fraction_data.columns and len(fraction_data) > 0:
                condition_data = fraction_data[fraction_data['condition'] == condition]['fraction_correct']
            else:
                condition_data = pd.Series(dtype=float)
            
            if len(condition_data) > 0:
                sorted_data = np.sort(condition_data)
                n = len(sorted_data)
                cumulative = np.arange(1, n + 1) / n
                ax_cumulative.plot(sorted_data, cumulative, color=color, label=label, 
                                   linewidth=2, marker='o', markersize=4)
        
        if col_idx == 0: ax_cumulative.set_ylabel('Cumulative Prob.')
        else: ax_cumulative.set_ylabel('')
        ax_cumulative.set_xlabel('Fraction Correct')
        ax_cumulative.set_xlim(0, 1)
        ax_cumulative.set_ylim(-0.1, 1.1)
        ax_cumulative.spines['top'].set_visible(False)
        ax_cumulative.spines['right'].set_visible(False)
        
        # --- Row 2: Summary statistics text ---
        ax_text = fig.add_subplot(gs[2, col_idx])
        ax_text.axis('off')
        
        summary_text = []
        summary_text.append(f"Summary Statistics - {title}")
        summary_text.append("=" * 40)
        
        for condition in conditions:
            if 'condition' in fraction_data.columns and len(fraction_data) > 0:
                condition_data = fraction_data[fraction_data['condition'] == condition]['fraction_correct']
            else:
                condition_data = pd.Series(dtype=float)
                
            if len(condition_data) > 0:
                summary_text.append(f"\n{condition.upper()}:")
                summary_text.append(f"  N sessions: {len(condition_data)}")
                summary_text.append(f"  Mean: {condition_data.mean():.4f}")
                summary_text.append(f"  Std: {condition_data.std():.4f}")
                summary_text.append(f"  Median: {condition_data.median():.4f}")
                summary_text.append(f"  Range: {condition_data.min():.4f} - {condition_data.max():.4f}")
            else:
                summary_text.append(f"\n{condition.upper()}: No data")
        
        text_content = '\n'.join(summary_text)
        ax_text.text(0.05, 0.95, text_content, transform=ax_text.transAxes, 
                     fontsize=9, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.suptitle(f'Rare Trial Analysis: Split by Opto - {subject}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        s_str = data_paths[-1].split("_")[-2] if len(data_paths) > 0 else "end"
        e_str = data_paths[0].split("_")[-2] if len(data_paths) > 0 else "start"
        output_path = os.path.join(save_path, f'Rare_Trial_Analysis_SplitOpto_{subject}_{s_str}_{e_str}.pdf')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return