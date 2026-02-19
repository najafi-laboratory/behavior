import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from matplotlib import gridspec

# Set up logging
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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
    
    short_block_mask = block_types == 1
    rare_mask |= short_block_mask & (trial_types == 2)  # Long trials in short blocks
    majority_mask |= short_block_mask & (trial_types == 1)  # Short trials in short blocks
    
    long_block_mask = block_types == 2
    rare_mask |= long_block_mask & (trial_types == 1)  # Short trials in long blocks
    majority_mask |= long_block_mask & (trial_types == 2)  # Long trials in long blocks
    
    logging.debug(f"Rare trials: {rare_mask.sum()}, Majority trials: {majority_mask.sum()}")
    return {'rare_mask': rare_mask, 'majority_mask': majority_mask}

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
    
    for block_type in [1, 2]:
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
            mid_point = len(segment) // 2
            early_indices = segment[:mid_point]
            late_indices = segment[mid_point:]
            majority_type = 1 if block_type == 1 else 2
            
            for idx in early_indices:
                if trial_types[idx] == majority_type:
                    early_majority_mask[idx] = True
            for idx in late_indices:
                if trial_types[idx] == majority_type:
                    late_majority_mask[idx] = True
    
    logging.debug(f"Early majority: {early_majority_mask.sum()}, Late majority: {late_majority_mask.sum()}")
    return {'early_majority_mask': early_majority_mask, 'late_majority_mask': late_majority_mask}

def identify_rare_and_rare_plus_one_trials(trial_types, block_types):
    """
    Identify rare and rare + 1 trials in early/late epochs for short/long blocks.
    
    Parameters:
    trial_types: list/array where 1=short, 2=long
    block_types: list/array where 0=neutral, 1=short block, 2=long block
    
    Returns:
    dict: Boolean masks for rare and rare + 1 trials in early/late epochs
    """
    trial_types = np.array(trial_types)
    block_types = np.array(block_types)
    
    masks = {
        'short_early_rare': np.zeros(len(trial_types), dtype=bool),
        'short_early_rare_plus_one': np.zeros(len(trial_types), dtype=bool),
        'short_late_rare': np.zeros(len(trial_types), dtype=bool),
        'short_late_rare_plus_one': np.zeros(len(trial_types), dtype=bool),
        'long_early_rare': np.zeros(len(trial_types), dtype=bool),
        'long_early_rare_plus_one': np.zeros(len(trial_types), dtype=bool),
        'long_late_rare': np.zeros(len(trial_types), dtype=bool),
        'long_late_rare_plus_one': np.zeros(len(trial_types), dtype=bool),
        'early_rare': np.zeros(len(trial_types), dtype=bool),
        'early_rare_plus_one': np.zeros(len(trial_types), dtype=bool),
        'late_rare': np.zeros(len(trial_types), dtype=bool),
        'late_rare_plus_one': np.zeros(len(trial_types), dtype=bool)
    }
    
    rare_majority = identify_rare_vs_majority_trials(trial_types, block_types)
    epoch_masks = identify_majority_trials_by_epoch(trial_types, block_types)
    rare_mask = rare_majority['rare_mask']
    early_mask = epoch_masks['early_majority_mask']
    late_mask = epoch_masks['late_majority_mask']
    
    short_block_mask = block_types == 1
    long_block_mask = block_types == 2
    
    # Identify rare trials
    masks['short_early_rare'] = short_block_mask & rare_mask & early_mask
    masks['short_late_rare'] = short_block_mask & rare_mask & late_mask
    masks['long_early_rare'] = long_block_mask & rare_mask & early_mask
    masks['long_late_rare'] = long_block_mask & rare_mask & late_mask
    
    # Identify rare + 1 trials
    for block_type, block_mask in [(1, short_block_mask), (2, long_block_mask)]:
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
            segment_rare_mask = rare_mask[segment]
            for i in range(len(segment) - 1):
                if segment_rare_mask[i]:
                    next_idx = segment[i + 1]
                    if block_type == 1:
                        if early_mask[next_idx]:
                            masks['short_early_rare_plus_one'][next_idx] = True
                        elif late_mask[next_idx]:
                            masks['short_late_rare_plus_one'][next_idx] = True
                    elif block_type == 2:
                        if early_mask[next_idx]:
                            masks['long_early_rare_plus_one'][next_idx] = True
                        elif late_mask[next_idx]:
                            masks['long_late_rare_plus_one'][next_idx] = True
    
    # Pooled masks for early and late epochs
    masks['early_rare'] = masks['short_early_rare'] | masks['long_early_rare']
    masks['early_rare_plus_one'] = masks['short_early_rare_plus_one'] | masks['long_early_rare_plus_one']
    masks['late_rare'] = masks['short_late_rare'] | masks['long_late_rare']
    masks['late_rare_plus_one'] = masks['short_late_rare_plus_one'] | masks['long_late_rare_plus_one']
    
    for key, mask in masks.items():
        logging.debug(f"{key}: {mask.sum()} trials")
    return masks

def compute_rare_trial_fractions(sessions_data):
    """
    Compute fraction of 'Reward' outcomes for rare and rare + 1 trials.
    
    Parameters:
    sessions_data: dict with keys 'outcomes', 'trial_types', 'block_type', 'dates'
    
    Returns:
    tuple: (pooled_results, short_results, long_results) as DataFrames
    """
    pooled_results = []
    short_results = []
    long_results = []
    
    if not all(key in sessions_data for key in ['outcomes', 'trial_types', 'block_type', 'dates']):
        logging.error("Missing required keys in sessions_data")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    n_sessions = len(sessions_data['outcomes'])
    for session_idx in range(n_sessions):
        outcomes = sessions_data['outcomes'][session_idx]
        trial_types = sessions_data['trial_types'][session_idx]
        block_types = sessions_data['block_type'][session_idx]
        date = sessions_data['dates'][session_idx]
        
        if not (len(outcomes) == len(trial_types) == len(block_types)):
            logging.warning(f"Session {session_idx}: Data length mismatch, skipping")
            continue
        if len(outcomes) < 4:
            logging.info(f"Session {session_idx}: Too few trials ({len(outcomes)}), skipping")
            continue
        
        outcomes_binary = np.array([1 if o == 'Reward' else 0 for o in outcomes], dtype=float)
        masks = identify_rare_and_rare_plus_one_trials(trial_types, block_types)
        
        conditions = [
            ('short_early_rare', short_results, 'short'),
            ('short_early_rare_plus_one', short_results, 'short'),
            ('short_late_rare', short_results, 'short'),
            ('short_late_rare_plus_one', short_results, 'short'),
            ('long_early_rare', long_results, 'long'),
            ('long_early_rare_plus_one', long_results, 'long'),
            ('long_late_rare', long_results, 'long'),
            ('long_late_rare_plus_one', long_results, 'long'),
            ('early_rare', pooled_results, 'pooled'),
            ('early_rare_plus_one', pooled_results, 'pooled'),
            ('late_rare', pooled_results, 'pooled'),
            ('late_rare_plus_one', pooled_results, 'pooled')
        ]
        
        for mask_name, results_list, block_type in conditions:
            mask = masks[mask_name]
            if mask.any():
                epoch_indices = np.where(mask)[0]
                epoch_outcomes = outcomes_binary[epoch_indices]
                valid_mask = ~np.isnan(epoch_outcomes)
                completed_outcomes = epoch_outcomes[valid_mask]
                
                if len(completed_outcomes) > 0:
                    fraction_correct = completed_outcomes.mean()
                    results_list.append({
                        'session': session_idx,
                        'date': date,
                        'condition': mask_name,
                        'fraction_correct': fraction_correct,
                        'n_trials': len(completed_outcomes),
                        'block_type': block_type
                    })
                else:
                    logging.debug(f"Session {session_idx}, {mask_name}: No valid outcomes")
            else:
                logging.debug(f"Session {session_idx}, {mask_name}: No trials identified")
    
    return (pd.DataFrame(pooled_results), pd.DataFrame(short_results), pd.DataFrame(long_results))

def plot_post_rare_trial_analysis(sessions_data, subject, data_paths, save_path=None):
    """
    Plot fraction of 'Reward' for rare and rare + 1 trials in a (n+1) x 3 grid.
    
    Parameters:
    sessions_data: dict from prepare_session_data function
    subject: subject identifier
    data_paths: list of data file paths
    save_path: optional path to save the figure
    """
    pooled_results, short_results, long_results = compute_rare_trial_fractions(sessions_data)
    
    logging.info(f"Pooled results: {len(pooled_results)} rows")
    logging.info(f"Short results: {len(short_results)} rows")
    logging.info(f"Long results: {len(long_results)} rows")
    
    n_sessions = len(sessions_data['outcomes'])
    fig = plt.figure(figsize=(18, 4 * (n_sessions + 1)))
    gs = gridspec.GridSpec(n_sessions + 1, 3, hspace=0.4, wspace=0.3)
    
    conditions = [
        ('early_rare', 'purple', 'Early Rare'),
        ('early_rare_plus_one', 'blue', 'Early Rare + 1'),
        ('late_rare', 'orange', 'Late Rare'),
        ('late_rare_plus_one', 'red', 'Late Rare + 1')
    ]
    
    analyses = [
        (short_results, "Short Blocks", 'short'),
        (long_results, "Long Blocks", 'long'),
        (pooled_results, "Pooled Blocks", 'pooled')
    ]
    
    for col, (fraction_data, title, block_type) in enumerate(analyses):
        # Pooled results
        ax = fig.add_subplot(gs[0, col])
        for cond, color, label in conditions:
            cond_name = cond if block_type == 'pooled' else f"{block_type}_{cond}"
            cond_data = fraction_data[fraction_data['condition'] == cond_name]['fraction_correct']
            if len(cond_data) > 1 and cond_data.std() > 1e-10:
                sns.kdeplot(data=cond_data, ax=ax, color=color, label=f"{label} (n={len(cond_data)})", 
                            linewidth=2, fill=False)
                ax.axvline(cond_data.mean(), color=color, linestyle='--', alpha=0.7)
            elif len(cond_data) == 1:
                ax.axvline(cond_data.iloc[0], color=color, label=f"{label} (n=1)", 
                           linewidth=3, alpha=0.7)
        ax.set_xlabel('Fraction Correct')
        ax.set_ylabel('Density')
        ax.set_title(f'{title}\nPooled Across Sessions')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Individual sessions
        for session_idx in range(n_sessions):
            ax = fig.add_subplot(gs[session_idx + 1, col])
            session_data = fraction_data[fraction_data['session'] == session_idx]
            for cond, color, label in conditions:
                cond_name = cond if block_type == 'pooled' else f"{block_type}_{cond}"
                cond_data = session_data[session_data['condition'] == cond_name]['fraction_correct']
                if len(cond_data) > 1 and cond_data.std() > 1e-10:
                    sns.kdeplot(data=cond_data, ax=ax, color=color, label=f"{label} (n={len(cond_data)})", 
                                linewidth=2, fill=False)
                    ax.axvline(cond_data.mean(), color=color, linestyle='--', alpha=0.7)
                elif len(cond_data) == 1:
                    ax.axvline(cond_data.iloc[0], color=color, label=f"{label} (n=1)", 
                               linewidth=3, alpha=0.7)
            ax.set_xlabel('Fraction Correct')
            ax.set_ylabel('Density')
            ax.set_title(f'Session {session_idx + 1} ({sessions_data["dates"][session_idx]})')
            ax.legend(fontsize=8)
            ax.set_xlim(0, 1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.suptitle(f'Rare and Rare + 1 Trial Analysis: Fraction Correct for {subject}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        output_path = os.path.join(save_path, f'Post_rare_prior_update_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return pooled_results, short_results, long_results
