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
    """
    trial_types = np.array(trial_types)
    block_types = np.array(block_types)
    
    early_majority_mask = np.zeros(len(trial_types), dtype=bool)
    late_majority_mask = np.zeros(len(trial_types), dtype=bool)
    
    unique_blocks = np.unique(block_types)
    
    for block_type in unique_blocks:
        if block_type == 0:  # Skip neutral blocks
            continue
            
        block_mask = block_types == block_type
        block_indices = np.where(block_mask)[0]
        
        if len(block_indices) == 0:
            continue
            
        block_segments = []
        current_segment = [block_indices[0]]
        
        for i in range(1, len(block_indices)):
            if block_indices[i] == block_indices[i-1] + 1:
                current_segment.append(block_indices[i])
            else:
                block_segments.append(current_segment)
                current_segment = [block_indices[i]]
        block_segments.append(current_segment)
        
        for segment in block_segments:
            if len(segment) < 2:
                continue
                
            segment = np.array(segment)
            mid_point = len(segment) // 2
            
            early_indices = segment[:mid_point]
            late_indices = segment[mid_point:]
            
            if block_type == 1:  # Short block - majority are short trials
                majority_trial_type = 1
            elif block_type == 2:  # Long block - majority are long trials
                majority_trial_type = 2
            else:
                continue
            
            for idx in early_indices:
                if trial_types[idx] == majority_trial_type:
                    early_majority_mask[idx] = True
            
            for idx in late_indices:
                if trial_types[idx] == majority_trial_type:
                    late_majority_mask[idx] = True
    
    return {
        'early_majority_mask': early_majority_mask,
        'late_majority_mask': late_majority_mask
    }

def compute_fractions(sessions_data, block_analysis='pooled', opto_condition='all'):
    """
    Compute fraction correct for majority trials in early vs late epochs.
    Filters by block type and opto condition.
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
        if n_trials < 4:
            continue
            
        # 1. Prepare Opto Mask
        if opto_raw is None or len(opto_raw) == 0:
            opto_clean = np.zeros(n_trials)
        else:
            temp = pd.to_numeric(opto_raw, errors='coerce')
            opto_clean = np.nan_to_num(temp, nan=0.0)
            if len(opto_clean) < n_trials:
                opto_clean = np.pad(opto_clean, (0, n_trials - len(opto_clean)), constant_values=0)
            elif len(opto_clean) > n_trials:
                opto_clean = opto_clean[:n_trials]
                
        if opto_condition == 'control':
            opto_mask = (opto_clean == 0)
        elif opto_condition == 'opto':
            opto_mask = (opto_clean != 0)
        else: # 'all'
            opto_mask = np.ones(n_trials, dtype=bool)

        # 2. Prepare Block Mask
        if block_analysis == 'short':
            block_mask = (np.array(block_types) == 1)
        elif block_analysis == 'long':
            block_mask = (np.array(block_types) == 2)
        else: # 'pooled'
            block_mask = (np.array(block_types) == 1) | (np.array(block_types) == 2)

        # 3. Get Epoch Masks (Indices of majority trials)
        epoch_masks = identify_majority_trials_by_epoch(trial_types, block_types)
        
        outcomes_binary = np.array([1 if outcome == 'Reward' else 0 for outcome in outcomes])
        
        # 4. Process Epochs
        for epoch_name, base_mask in epoch_masks.items():
            # Combine masks: Is MajorityEpoch AND Is TargetBlock AND Is TargetOpto
            final_mask = base_mask & block_mask & opto_mask
            
            if not final_mask.any():
                continue
                
            epoch_label = 'early' if 'early' in epoch_name else 'late'
            indices = np.where(final_mask)[0]
            
            if len(indices) > 0:
                epoch_outcomes = outcomes_binary[indices]
                valid_outcomes = epoch_outcomes[~np.isnan(epoch_outcomes)]
                
                if len(valid_outcomes) > 0:
                    fraction = valid_outcomes.mean()
                    results.append({
                        'session': session_idx,
                        'date': date,
                        'epoch': epoch_label,
                        'fraction_correct': fraction,
                        'n_trials': len(valid_outcomes)
                    })
                    
    return pd.DataFrame(results)

def plot_all_majority_trial_analysis(sessions_data, subject, data_paths, save_path=None):
    """
    Plot majority trial analysis in a 9-column grid (Control, Opto, All) x (Pooled, Short, Long).
    """
    
    # Configuration: (Opto Condition, Block Analysis, Title)
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
    fig = plt.figure(figsize=(32, 12))
    gs = gridspec.GridSpec(3, 9, height_ratios=[1, 1, 0.6], hspace=0.3, wspace=0.3)
    
    epochs = ['early', 'late']
    colors = ['purple', 'orange']
    epoch_labels = ['Early Epoch (1st half)', 'Late Epoch (2nd half)']
    
    for col_idx, (opto_cond, block_cond, title) in enumerate(configs):
        
        # Compute data for this column
        fraction_data = compute_fractions(sessions_data, block_analysis=block_cond, opto_condition=opto_cond)
        
        # --- Top row: Density Plot ---
        ax_density = fig.add_subplot(gs[0, col_idx])
        
        for epoch, color, label in zip(epochs, colors, epoch_labels):
            if 'epoch' in fraction_data.columns and len(fraction_data) > 0:
                epoch_data = fraction_data[fraction_data['epoch'] == epoch]['fraction_correct']
            else:
                epoch_data = pd.Series(dtype=float)
            
            if len(epoch_data) > 1:
                try:
                    if epoch_data.std() > 1e-10:
                        sns.kdeplot(data=epoch_data, ax=ax_density, color=color, 
                                    label=f"{label} (n={len(epoch_data)})", 
                                    linewidth=2, fill=False)
                        ax_density.axvline(epoch_data.mean(), color=color, linestyle='--', alpha=0.7)
                    else:
                        ax_density.axvline(epoch_data.mean(), color=color, label=f"{label} (n={len(epoch_data)})", 
                                           linewidth=3, alpha=0.7)
                except Exception:
                    ax_density.axvline(epoch_data.mean(), color=color, label=f"{label} (n={len(epoch_data)})", 
                                       linewidth=3, alpha=0.7)
            elif len(epoch_data) == 1:
                ax_density.axvline(epoch_data.iloc[0], color=color, label=label, linewidth=3, alpha=0.7)
        
        ax_density.set_title(title, fontsize=10, fontweight='bold')
        if col_idx == 0: ax_density.set_ylabel('Density')
        else: ax_density.set_ylabel('')
        ax_density.set_xlabel('')
        ax_density.set_xlim(0, 1)
        if col_idx == 0: ax_density.legend(fontsize=6, loc='upper left')
        ax_density.spines['top'].set_visible(False)
        ax_density.spines['right'].set_visible(False)
        
        # --- Middle row: Cumulative distributions ---
        ax_cumulative = fig.add_subplot(gs[1, col_idx])
        
        for epoch, color, label in zip(epochs, colors, epoch_labels):
            if 'epoch' in fraction_data.columns and len(fraction_data) > 0:
                epoch_data = fraction_data[fraction_data['epoch'] == epoch]['fraction_correct']
            else:
                epoch_data = pd.Series(dtype=float)
            
            if len(epoch_data) > 0:
                sorted_data = np.sort(epoch_data)
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
        
        # --- Bottom row: Summary statistics text ---
        ax_text = fig.add_subplot(gs[2, col_idx])
        ax_text.axis('off')
        
        summary_text = []
        summary_text.append(f"Summary Statistics")
        summary_text.append("=" * 20)
        
        for epoch in epochs:
            if 'epoch' in fraction_data.columns and len(fraction_data) > 0:
                epoch_data = fraction_data[fraction_data['epoch'] == epoch]['fraction_correct']
            else:
                epoch_data = pd.Series(dtype=float)
                
            if len(epoch_data) > 0:
                summary_text.append(f"\n{epoch.upper()} EPOCH:")
                summary_text.append(f" N sessions: {len(epoch_data)}")
                summary_text.append(f" Mean: {epoch_data.mean():.4f}")
                summary_text.append(f" Std: {epoch_data.std():.4f}")
                summary_text.append(f" Median: {epoch_data.median():.4f}")
                summary_text.append(f" Range: {epoch_data.min():.4f} - {epoch_data.max():.4f}")
            else:
                summary_text.append(f"\n{epoch.upper()} EPOCH: No data")
        
        if 'epoch' in fraction_data.columns and len(fraction_data) > 0:
            early_data = fraction_data[fraction_data['epoch'] == 'early']['fraction_correct']
            late_data = fraction_data[fraction_data['epoch'] == 'late']['fraction_correct']
            
            if len(early_data) > 1 and len(late_data) > 1:
                try:
                    t_stat, p_value = stats.ttest_ind(early_data, late_data)
                    pooled_std = np.sqrt(((len(early_data)-1)*early_data.var() + (len(late_data)-1)*late_data.var()) / (len(early_data) + len(late_data) - 2))
                    cohens_d = (early_data.mean() - late_data.mean()) / pooled_std if pooled_std != 0 else 0
                    
                    summary_text.append(f"\nSTATISTICAL COMPARISON:")
                    summary_text.append(f" t-statistic: {t_stat:.4f}")
                    summary_text.append(f" p-value: {p_value:.4f}")
                    summary_text.append(f" Effect size: {cohens_d:.4f}")
                except Exception:
                    summary_text.append(f"\nSTATISTICAL COMPARISON: Error")
        
        text_content = '\n'.join(summary_text)
        ax_text.text(0.05, 1.0, text_content, transform=ax_text.transAxes, 
                     fontsize=8, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

    plt.suptitle(f'Standard Trial Analysis: Early vs Late Epoch - Split by Opto - {subject}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        s_str = data_paths[-1].split("_")[-2] if len(data_paths) > 0 else "end"
        e_str = data_paths[0].split("_")[-2] if len(data_paths) > 0 else "start"
        output_path = os.path.join(save_path, f'Standard_Trial_Analysis_SplitOpto_{subject}_{s_str}_{e_str}.pdf')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return