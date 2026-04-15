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
    
    return {'rare_mask': rare_mask, 'majority_mask': majority_mask}

def identify_majority_trials_by_epoch(trial_types, block_types):
    """
    Identify majority trials in early vs late epochs within each block.
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
    
    return {'early_majority_mask': early_majority_mask, 'late_majority_mask': late_majority_mask}

def identify_rare_and_rare_plus_one_trials(trial_types, block_types, opto_tags):
    """
    Identify rare and rare + 1 trials, split by whether the RARE trial was Opto or Control.
    """
    trial_types = np.array(trial_types)
    block_types = np.array(block_types)
    
    # Clean opto_tags: Treat NaN, None as 0. Ensure numeric comparison.
    # Convert to numeric, forcing errors to NaN, then filling with 0
    opto_numeric = pd.to_numeric(opto_tags, errors='coerce')
    opto_clean = np.nan_to_num(opto_numeric, nan=0.0)
    
    # Base masks
    rare_majority = identify_rare_vs_majority_trials(trial_types, block_types)
    epoch_masks = identify_majority_trials_by_epoch(trial_types, block_types)
    base_rare_mask = rare_majority['rare_mask']
    early_mask = epoch_masks['early_majority_mask']
    late_mask = epoch_masks['late_majority_mask']
    
    short_block_mask = block_types == 1
    long_block_mask = block_types == 2
    
    masks = {}
    
    # Define conditions to iterate
    conditions = ['control', 'opto', 'all']
    blocks = ['short', 'long']
    epochs = ['early', 'late']
    types = ['rare_plus_one'] # We focus on +1 for this request
    
    # Pre-initialize dictionary to avoid KeyErrors
    for c in conditions:
        for b in blocks:
            for e in epochs:
                masks[f"{c}_{b}_{e}_rare_plus_one"] = np.zeros(len(trial_types), dtype=bool)

    # 1. Create Base Rare Masks for each condition
    # Control Rare: Rare trial AND opto is 0
    rare_control_mask = base_rare_mask & (opto_clean == 0)
    # Opto Rare: Rare trial AND opto is NOT 0
    rare_opto_mask = base_rare_mask & (opto_clean != 0)
    # All Rare: Just the base mask
    rare_all_mask = base_rare_mask

    # 2. Identify Rare + 1 Trials
    for block_type, block_mask, block_str in [(1, short_block_mask, 'short'), (2, long_block_mask, 'long')]:
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
            for i in range(len(segment) - 1):
                curr_idx = segment[i]
                next_idx = segment[i+1]
                
                # Check epoch of the NEXT trial
                is_early_next = early_mask[next_idx]
                is_late_next = late_mask[next_idx]
                
                epoch_str = None
                if is_early_next: epoch_str = 'early'
                elif is_late_next: epoch_str = 'late'
                
                if epoch_str:
                    # Fill masks based on the CURRENT (Rare) trial's type
                    if rare_control_mask[curr_idx]:
                        masks[f"control_{block_str}_{epoch_str}_rare_plus_one"][next_idx] = True
                    if rare_opto_mask[curr_idx]:
                        masks[f"opto_{block_str}_{epoch_str}_rare_plus_one"][next_idx] = True
                    if rare_all_mask[curr_idx]:
                        masks[f"all_{block_str}_{epoch_str}_rare_plus_one"][next_idx] = True

    # 3. Create Pooled Masks (combining Short and Long)
    for c in conditions:
        for e in epochs:
            # Union of short and long
            short_key = f"{c}_short_{e}_rare_plus_one"
            long_key = f"{c}_long_{e}_rare_plus_one"
            pooled_key = f"{c}_pooled_{e}_rare_plus_one" # Naming explicitly 'pooled' for easier parsing
            
            masks[pooled_key] = masks[short_key] | masks[long_key]

    return masks

def compute_rare_trial_fractions(sessions_data):
    """
    Compute fraction of 'Reward' outcomes for rare + 1 trials.
    """
    all_results = []
    
    # Check keys
    required_keys = ['outcomes', 'trial_types', 'block_type', 'dates', 'opto_tags']
    if not all(key in sessions_data for key in required_keys):
        logging.error(f"Missing keys. Have: {list(sessions_data.keys())}")
        return pd.DataFrame()
    
    n_sessions = len(sessions_data['outcomes'])
    
    for session_idx in range(n_sessions):
        outcomes = sessions_data['outcomes'][session_idx]
        trial_types = sessions_data['trial_types'][session_idx]
        block_types = sessions_data['block_type'][session_idx]
        opto_tags = sessions_data['opto_tags'][session_idx]
        date = sessions_data['dates'][session_idx]
        
        if len(outcomes) < 4: 
            continue
            
        outcomes_binary = np.array([1 if o == 'Reward' else 0 for o in outcomes], dtype=float)
        
        masks = identify_rare_and_rare_plus_one_trials(trial_types, block_types, opto_tags)
        
        for key, mask in masks.items():
            if not key.endswith('_rare_plus_one'):
                continue
                
            # ROBUST PARSING logic
            # Remove the suffix first
            base_name = key.replace('_rare_plus_one', '')
            parts = base_name.split('_')
            
            # Expected patterns:
            # 1. {opto}_{block}_{epoch} -> e.g., control_short_early
            # 2. {opto}_{pooled}_{epoch} -> e.g., control_pooled_early (since we renamed it above)
            
            if len(parts) != 3:
                # Fallback or skip unexpected keys
                continue
                
            opto_cond = parts[0]   # control, opto, all
            block_cond = parts[1]  # short, long, pooled
            epoch_cond = parts[2]  # early, late
            
            if mask.any():
                epoch_indices = np.where(mask)[0]
                epoch_outcomes = outcomes_binary[epoch_indices]
                valid_mask = ~np.isnan(epoch_outcomes)
                completed_outcomes = epoch_outcomes[valid_mask]
                
                if len(completed_outcomes) > 0:
                    fraction_correct = completed_outcomes.mean()
                    all_results.append({
                        'session': session_idx,
                        'date': date,
                        'opto_condition': opto_cond,
                        'block_condition': block_cond,
                        'epoch_condition': epoch_cond,
                        'fraction_correct': fraction_correct,
                        'n_trials': len(completed_outcomes)
                    })

    return pd.DataFrame(all_results)

def plot_post_rare_trial_analysis(sessions_data, subject, data_paths, save_path=None):
    """
    Plot fraction of 'Reward' for rare + 1 trials in a (n+1) x 9 grid.
    """
    results_df = compute_rare_trial_fractions(sessions_data)
    
    if results_df.empty:
        logging.warning("No data found for analysis.")
        return results_df

    n_sessions = len(sessions_data['outcomes'])
    
    # Setup Figure
    # 9 columns: Control(S,L,P), Opto(S,L,P), All(S,L,P)
    fig = plt.figure(figsize=(24, 4 * (n_sessions + 1)))
    gs = gridspec.GridSpec(n_sessions + 1, 9, hspace=0.4, wspace=0.3)
    
    column_defs = [
        (0, 'control', 'short', 'Control: Short Blocks'),
        (1, 'control', 'long',  'Control: Long Blocks'),
        (2, 'control', 'pooled', 'Control: Pooled'),
        (3, 'opto', 'short', 'Opto: Short Blocks'),
        (4, 'opto', 'long',  'Opto: Long Blocks'),
        (5, 'opto', 'pooled', 'Opto: Pooled'),
        (6, 'all', 'short', 'All: Short Blocks'),
        (7, 'all', 'long',  'All: Long Blocks'),
        (8, 'all', 'pooled', 'All: Pooled'),
    ]
    
    epoch_styles = [
        ('early', 'blue', 'Early Rare+1'),
        ('late', 'red', 'Late Rare+1')
    ]

    # Plot Columns
    for col_idx, opto_c, block_c, title in column_defs:
        
        # Filter Data
        col_data = results_df[
            (results_df['opto_condition'] == opto_c) & 
            (results_df['block_condition'] == block_c)
        ]
        
        # --- TOP ROW: Aggregated Distribution ---
        ax = fig.add_subplot(gs[0, col_idx])
        
        if not col_data.empty:
            for epoch, color, label in epoch_styles:
                epoch_data = col_data[col_data['epoch_condition'] == epoch]['fraction_correct']
                # Drop NaNs just in case
                epoch_data = epoch_data.dropna()
                
                if len(epoch_data) > 1 and epoch_data.std() > 1e-10:
                    sns.kdeplot(data=epoch_data, ax=ax, color=color, 
                                label=f"{label}", linewidth=2, fill=False)
                    ax.axvline(epoch_data.mean(), color=color, linestyle='--', alpha=0.7)
                elif len(epoch_data) >= 1:
                    ax.axvline(epoch_data.iloc[0], color=color, label=f"{label}", 
                               linewidth=3, alpha=0.7)
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        if col_idx == 0: ax.set_ylabel('Density (All Sessions)')
        else: ax.set_ylabel('')
        
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # --- SESSION ROWS ---
        for session_idx in range(n_sessions):
            ax = fig.add_subplot(gs[session_idx + 1, col_idx])
            
            sess_data = col_data[col_data['session'] == session_idx]
            
            has_data = False
            if not sess_data.empty:
                for epoch, color, label in epoch_styles:
                    val = sess_data[sess_data['epoch_condition'] == epoch]['fraction_correct']
                    if not val.empty:
                        ax.axvline(val.iloc[0], color=color, label=label, linewidth=3, alpha=0.7)
                        has_data = True
            
            if session_idx == 0:
                ax.set_title(f"Sess {session_idx+1}", fontsize=8)
            
            if col_idx == 0:
                ax.set_ylabel(f'Sess {session_idx+1}', fontsize=8)
            else:
                ax.set_ylabel('')
                
            ax.set_xlim(0, 1)
            ax.set_yticks([]) # Hide y ticks for cleaner look on single value plots
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

    plt.suptitle(f'Rare + 1 Analysis by Opto Type: {subject}', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    if save_path:
        s_str = data_paths[-1].split("_")[-2] if len(data_paths) > 0 else "end"
        e_str = data_paths[0].split("_")[-2] if len(data_paths) > 0 else "start"
        output_path = os.path.join(save_path, f'Post_rare_opto_split_{subject}_{s_str}_{e_str}.pdf')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
    plt.close()
    return results_df