import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.gridspec as gridspec
import os

def identify_rare_vs_majority_trials(trial_types, block_types):
    """
    Identify rare vs majority trials within each block type.
    """
    trial_types = np.array(trial_types)
    block_types = np.array(block_types)
    
    rare_mask = np.zeros(len(trial_types), dtype=bool)
    majority_mask = np.zeros(len(trial_types), dtype=bool)
    
    # Short blocks: rare=long(2), majority=short(1)
    short_block_mask = block_types == 1
    rare_mask |= short_block_mask & (trial_types == 2)
    majority_mask |= short_block_mask & (trial_types == 1)
    
    # Long blocks: rare=short(1), majority=long(2)
    long_block_mask = block_types == 2
    rare_mask |= long_block_mask & (trial_types == 1)
    majority_mask |= long_block_mask & (trial_types == 2)
    
    return {'rare_mask': rare_mask, 'majority_mask': majority_mask}

def compute_fractions(sessions_data, block_analysis='pooled', opto_condition='all'):
    """
    Compute fraction correct for rare vs majority trials, filtered by block type and opto condition.
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
        if n_trials < 2:
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
                
        # 2. Define Opto Mask
        if opto_condition == 'control':
            opto_mask = (opto_clean == 0)
        elif opto_condition == 'opto':
            opto_mask = (opto_clean != 0)
        else: # 'all'
            opto_mask = np.ones(n_trials, dtype=bool)
            
        # 3. Define Block Mask
        if block_analysis == 'short':
            block_mask = (np.array(block_types) == 1)
        elif block_analysis == 'long':
            block_mask = (np.array(block_types) == 2)
        else: # 'pooled'
            valid_blocks = (np.array(block_types) == 1) | (np.array(block_types) == 2)
            block_mask = valid_blocks

        # 4. Identify Trial Categories
        type_masks = identify_rare_vs_majority_trials(trial_types, block_types)
        
        # 5. Combine Masks and Compute
        outcomes_binary = np.array([1 if o == 'Reward' else 0 for o in outcomes])
        
        for category_label, base_type_mask in type_masks.items():
            final_mask = base_type_mask & block_mask & opto_mask
            clean_label = 'rare' if 'rare' in category_label else 'majority'
            
            indices = np.where(final_mask)[0]
            
            if len(indices) > 0:
                epoch_outcomes = outcomes_binary[indices]
                valid_outcomes = epoch_outcomes[~np.isnan(epoch_outcomes)]
                
                if len(valid_outcomes) > 0:
                    fraction = valid_outcomes.mean()
                    results.append({
                        'session': session_idx,
                        'date': date,
                        'trial_category': clean_label,
                        'fraction_correct': fraction,
                        'n_trials': len(valid_outcomes)
                    })
                    
    return pd.DataFrame(results)

def plot_all_rare_vs_majority_analysis(sessions_data, subject, data_paths, save_path=None):
    """
    Plot rare vs majority performance in a 9-column grid with detailed statistics.
    """
    
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
    
    # Increased height ratio for the stats row to accommodate detailed text
    fig = plt.figure(figsize=(32, 14))
    gs = gridspec.GridSpec(3, 9, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.4)
    
    categories = ['rare', 'majority']
    colors = ['red', 'blue']
    category_labels = ['Rare', 'Standard']

    for col_idx, (opto_cond, block_cond, title) in enumerate(configs):
        
        # 1. Compute Data
        df_results = compute_fractions(sessions_data, block_analysis=block_cond, opto_condition=opto_cond)
        
        # --- Row 0: Density Plot ---
        ax_density = fig.add_subplot(gs[0, col_idx])
        
        has_data = False
        if not df_results.empty:
            for cat, color, label in zip(categories, colors, category_labels):
                cat_data = df_results[df_results['trial_category'] == cat]['fraction_correct']
                
                if len(cat_data) > 1:
                    has_data = True
                    try:
                        if cat_data.std() > 1e-10:
                            sns.kdeplot(data=cat_data, ax=ax_density, color=color, 
                                        label=f"{label}", linewidth=2, fill=False)
                            ax_density.axvline(cat_data.mean(), color=color, linestyle='--', alpha=0.5)
                        else:
                            ax_density.axvline(cat_data.mean(), color=color, label=f"{label}", linewidth=2)
                    except:
                         ax_density.axvline(cat_data.mean(), color=color, label=f"{label}", linewidth=2)
                elif len(cat_data) == 1:
                    has_data = True
                    ax_density.axvline(cat_data.iloc[0], color=color, label=f"{label}", linewidth=2)

        ax_density.set_title(title, fontsize=10, fontweight='bold')
        if col_idx == 0: ax_density.set_ylabel('Density')
        else: ax_density.set_ylabel('')
        ax_density.set_xlabel('')
        ax_density.set_xlim(0, 1)
        if col_idx == 0: ax_density.legend(fontsize=6, loc='upper left')
        ax_density.spines['top'].set_visible(False)
        ax_density.spines['right'].set_visible(False)

        # --- Row 1: Cumulative Distribution ---
        ax_cum = fig.add_subplot(gs[1, col_idx])
        
        if not df_results.empty:
            for cat, color, label in zip(categories, colors, category_labels):
                cat_data = df_results[df_results['trial_category'] == cat]['fraction_correct']
                if len(cat_data) > 0:
                    sorted_data = np.sort(cat_data)
                    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    ax_cum.plot(sorted_data, yvals, color=color, marker='.', linestyle='-', linewidth=1, alpha=0.7)

        if col_idx == 0: ax_cum.set_ylabel('Cumulative Prob.')
        else: ax_cum.set_ylabel('')
        ax_cum.set_xlabel('Fraction Correct')
        ax_cum.set_xlim(0, 1)
        ax_cum.set_ylim(-0.05, 1.05)
        ax_cum.spines['top'].set_visible(False)
        ax_cum.spines['right'].set_visible(False)

        # --- Row 2: DETAILED Statistics Text ---
        ax_text = fig.add_subplot(gs[2, col_idx])
        ax_text.axis('off')
        
        summary_text = []
        summary_text.append(f"Stats: {title}")
        summary_text.append("-" * 25)

        for cat, label in zip(categories, category_labels):
            if not df_results.empty:
                cat_data = df_results[df_results['trial_category'] == cat]['fraction_correct']
                
                if len(cat_data) > 0:
                    summary_text.append(f"{label.upper()} (n={len(cat_data)}):")
                    summary_text.append(f" Mean: {cat_data.mean():.3f}")
                    summary_text.append(f" Std:  {cat_data.std():.3f}")
                    summary_text.append(f" Med:  {cat_data.median():.3f}")
                    summary_text.append(f" Rng:  {cat_data.min():.2f}-{cat_data.max():.2f}")
                else:
                    summary_text.append(f"{label.upper()}: No Data")
            else:
                summary_text.append(f"{label.upper()}: No Data")
            summary_text.append("") # Spacer

        # Statistical Comparison
        if not df_results.empty:
            rare_data = df_results[df_results['trial_category'] == 'rare']['fraction_correct']
            maj_data = df_results[df_results['trial_category'] == 'majority']['fraction_correct']
            
            if len(rare_data) > 1 and len(maj_data) > 1:
                summary_text.append("COMPARISON:")
                try:
                    t_stat, p_value = stats.ttest_ind(rare_data, maj_data, nan_policy='omit')
                    
                    # Cohen's d
                    pooled_std = np.sqrt(((len(rare_data)-1)*rare_data.var() + (len(maj_data)-1)*maj_data.var()) / (len(rare_data) + len(maj_data) - 2))
                    cohens_d = (rare_data.mean() - maj_data.mean()) / pooled_std if pooled_std != 0 else 0
                    
                    summary_text.append(f" t: {t_stat:.3f}, p: {p_value:.4f}")
                    summary_text.append(f" d: {cohens_d:.3f}")
                    summary_text.append(f" Diff: {rare_data.mean() - maj_data.mean():.3f}")
                except:
                    summary_text.append(" Stats Error")
            else:
                summary_text.append(" Comp: N/A (low n)")

        text_content = '\n'.join(summary_text)
        
        # Use a slightly smaller font to ensure the detailed stats fit in the column width
        ax_text.text(0.05, 1.0, text_content, transform=ax_text.transAxes, 
                     fontsize=7, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', alpha=0.5))

    plt.suptitle(f'Rare vs Standard Trial Analysis: Split by Opto - {subject}', 
                 fontsize=16, fontweight='bold', y=0.99)
    
    if save_path:
        s_str = data_paths[-1].split("_")[-2] if len(data_paths) > 0 else "end"
        e_str = data_paths[0].split("_")[-2] if len(data_paths) > 0 else "start"
        output_path = os.path.join(save_path, f'Rare_vs_Standard_SplitOpto_{subject}_{s_str}_{e_str}.pdf')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()