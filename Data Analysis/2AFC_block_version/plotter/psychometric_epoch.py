import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import sem
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import os

def plot_psychometric_epochs(sessions_data, subject, data_paths, output_pdf='psychometric_epochs_opto.pdf', bin_width=0.01, fit_logistic=True, save_path=None):
    """
    Plot psychometric curves separating Block (Short/Long), Epoch (Early/Late), and Trial Type (Opto/Control).
    Generates a 12-column figure.
    """
    
    # --- Configuration ---
    # Colors
    colors = {
        'early_opto': 'deepskyblue',  # Light Blue
        'early_control': 'gray',      # Gray
        'late_opto': 'blue',          # Blue
        'late_control': 'black'       # Black
    }
    
    # Comparisons config: (Title, [Condition 1 key, Condition 2 key])
    comparisons = [
        ('Opto: Early vs Late', ['early_opto', 'late_opto']),
        ('Ctrl: Early vs Late', ['early_control', 'late_control']),
        ('Early: Opto vs Ctrl', ['early_opto', 'early_control']),
        ('Late: Opto vs Ctrl', ['late_opto', 'late_control'])
    ]
    
    block_order = ['short', 'long', 'all']

    # --- Helper Functions ---

    def identify_blocks(block_types):
        """Identify start and end indices of short (1) and long (2) blocks, excluding neutral (0)."""
        blocks = []
        current_block_type = None
        start_idx = 0
        for i, bt in enumerate(block_types):
            if bt == 0:  # Skip neutral blocks
                if current_block_type is not None:
                    blocks.append((current_block_type, start_idx, i))
                    current_block_type = None
                continue
            if current_block_type is None:
                current_block_type = bt
                start_idx = i
            elif current_block_type != bt:
                blocks.append((current_block_type, start_idx, i))
                current_block_type = bt
                start_idx = i
        if current_block_type is not None and start_idx < len(block_types):
            blocks.append((current_block_type, start_idx, len(block_types)))
        return blocks

    def initialize_storage():
        """Creates the nested dictionary structure for data storage."""
        # Structure: data[block][epoch][type] = {'isi': [], 'choices': []}
        data = {}
        for b in ['short', 'long', 'all']:
            data[b] = {}
            for e in ['early', 'late']:
                data[b][e] = {
                    'opto': {'isi': [], 'choices': []},
                    'control': {'isi': [], 'choices': []}
                }
        return data

    def split_block_epochs(lick_properties):
        """Split blocks into early/late and opto/control."""
        session_data = initialize_storage()
        
        # 1. Reconstruct Trial Order
        all_block_types = []
        trial_indices = []
        keys = ['short_ISI_reward_left_correct_lick', 'short_ISI_reward_right_incorrect_lick',
                'short_ISI_punish_right_incorrect_lick', 'short_ISI_punish_left_correct_lick',
                'long_ISI_reward_right_correct_lick', 'long_ISI_reward_left_incorrect_lick',
                'long_ISI_punish_left_incorrect_lick', 'long_ISI_punish_right_correct_lick']
                
        for key in keys:
            if key in lick_properties:
                block_types = lick_properties[key].get('block_type', [])
                for i in range(len(block_types)):
                    all_block_types.append(block_types[i])
                    trial_indices.append(i)
        
        if not all_block_types:
            return session_data

        sorted_indices = np.argsort(trial_indices)
        all_block_types = np.array(all_block_types)[sorted_indices]
        
        # 2. Identify Blocks (Neutral blocks handled here)
        # We don't strictly need block start/stop indices for the logic below because 
        # specific keys categorize data, but we filter neutral blocks via block_type check
        
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
        
        for key in left_keys + right_keys:
            if key not in lick_properties: continue
            
            trial_isi = lick_properties[key].get('Trial_ISI', [])
            opto_tags = lick_properties[key].get('opto_tag', [])
            block_types = lick_properties[key].get('block_type', [])
            epochs = lick_properties[key].get('epoch', [])
            
            for i in range(len(trial_isi)):
                # Skip invalid ISIs or Neutral Blocks (0)
                if trial_isi[i] is None or block_types[i] == 0:
                    continue
                
                # Determine Categories
                block_key = 'short' if block_types[i] == 1 else 'long' if block_types[i] == 2 else None
                if block_key is None: continue
                
                epoch_key = 'early' if epochs[i] == 1 else 'late'
                
                # Check Opto (Handle NaNs as Control)
                is_opto = (opto_tags[i] == 1)
                type_key = 'opto' if is_opto else 'control'
                
                choice_val = 1 if key in right_keys else 0
                
                # Store in Specific Block
                session_data[block_key][epoch_key][type_key]['isi'].append(trial_isi[i])
                session_data[block_key][epoch_key][type_key]['choices'].append(choice_val)
                
                # Store in 'All'
                session_data['all'][epoch_key][type_key]['isi'].append(trial_isi[i])
                session_data['all'][epoch_key][type_key]['choices'].append(choice_val)
                
        return session_data

    def calculate_psychometric(isi, choices, bin_width, single_isi_case):
        """Calculate choice probabilities and SEM."""
        if len(isi) == 0: return None, None, None
        
        if single_isi_case:
            unique_isi = np.unique(isi)
            if len(unique_isi) != 2: return None, None, None
            right_prob = np.zeros(2)
            sem_values = np.zeros(2)
            counts = np.zeros(2)
            for i, isi_val in enumerate(unique_isi):
                mask = isi == isi_val
                bin_choices = choices[mask]
                if len(bin_choices) > 0:
                    right_prob[i] = np.mean(bin_choices)
                    sem_values[i] = sem(bin_choices, nan_policy='omit')
                    counts[i] = len(bin_choices)
            valid_mask = counts > 0
            return unique_isi[valid_mask], right_prob[valid_mask], sem_values[valid_mask]
        else:
            min_isi = np.floor(np.min(isi) / bin_width) * bin_width
            max_isi = np.ceil(np.max(isi) / bin_width) * bin_width
            bins = np.arange(min_isi, max_isi + bin_width, bin_width)
            bin_centers = bins[:-1] + bin_width / 2
            right_prob = np.zeros(len(bins) - 1)
            sem_values = np.zeros(len(bins) - 1)
            counts = np.zeros(len(bins) - 1)
            for i in range(len(bins) - 1):
                mask = (isi >= bins[i]) & (isi < bins[i + 1])
                bin_choices = choices[mask]
                if len(bin_choices) > 0:
                    right_prob[i] = np.mean(bin_choices)
                    sem_values[i] = sem(bin_choices, nan_policy='omit')
                    counts[i] = len(bin_choices)
            valid_mask = counts > 0
            return bin_centers[valid_mask], right_prob[valid_mask], sem_values[valid_mask]

    def logistic_function(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def plot_psychometric(ax, isi, choices, label, color, isi_divider, single_isi_case, fit_logistic):
        x_data, y_data, y_sem = calculate_psychometric(isi, choices, bin_width, single_isi_case)
        if x_data is None: return False
        
        n_trials = len(choices)
        label_with_count = f'{label} (n={n_trials})'
        
        ax.errorbar(x_data, y_data, yerr=y_sem, fmt='o', color=color, capsize=3, alpha=0.7, label=label_with_count)
        
        if single_isi_case and len(x_data) == 2:
            ax.plot(x_data, y_data, '-', color=color, linewidth=2, alpha=0.7)
            # Add inflection line if crossing 0.5
            x1, x2 = x_data; y1, y2 = y_data
            if y2 != y1:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                inflection_point = (0.5 - b) / m
                if min(x1, x2) <= inflection_point <= max(x1, x2):
                    ax.axvline(x=inflection_point, color=color, linestyle='--', alpha=0.5)
        elif fit_logistic and len(x_data) > 3:
            try:
                popt, _ = curve_fit(logistic_function, x_data, y_data, p0=[1.0, 1.0, np.median(x_data)],
                                    bounds=([0.5, -10, np.min(x_data)], [1, 10, np.max(x_data)]))
                x_fit = np.linspace(np.min(x_data), np.max(x_data), 100)
                y_fit = logistic_function(x_fit, *popt)
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2, alpha=0.7)
                ip_index = np.argmin(np.abs(y_fit - 0.5))
                inflection_point = x_fit[ip_index]
                ax.axvline(x=inflection_point, color=color, linestyle='--', alpha=0.5)
            except:
                pass # Fit failed
        return True

    def plot_row(ax_list, data_dict, isi_divider, single_isi, row_title_prefix=''):
        """Plots a full row of 12 subplots given a data dictionary for that row (session or pooled)."""
        
        for b_idx, block in enumerate(block_order): # 0: Short, 1: Long, 2: All
            for c_idx, (comp_title, keys) in enumerate(comparisons): # 0..3 comparisons
                
                # Calculate absolute column index (0 to 11)
                col_idx = (b_idx * 4) + c_idx
                ax = ax_list[col_idx]
                
                has_data = False
                
                # Plot the two conditions in the comparison
                for k in keys:
                    # Parse key (e.g., 'early_opto' -> epoch='early', type='opto')
                    epoch, trial_type = k.split('_')
                    
                    isi_data = np.array(data_dict[block][epoch][trial_type]['isi'])
                    choice_data = np.array(data_dict[block][epoch][trial_type]['choices'])
                    
                    # Convert key to display label (e.g. 'early_opto' -> 'Early Opto')
                    display_label = k.replace('_', ' ').title()
                    
                    plotted = plot_psychometric(ax, isi_data, choice_data, display_label, colors[k], 
                                              isi_divider, single_isi, fit_logistic)
                    if plotted: has_data = True

                # Styling
                if has_data:
                    ax.axvline(x=isi_divider, color='red', linestyle='--', alpha=0.3)
                    ax.axhline(y=0.5, color='black', linestyle='-', alpha=0.2)
                    ax.set_ylim(-0.05, 1.05)
                    ax.legend(fontsize=6)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=8)
                
                # Titles and Labels
                full_title = f"{block.title()} - {comp_title}"
                if row_title_prefix:
                    full_title = f"{row_title_prefix}\n{full_title}"
                ax.set_title(full_title, fontsize=8)
                
                ax.grid(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
                # Only label left-most axes to reduce clutter
                if col_idx == 0:
                    ax.set_ylabel('Prob. Right')
                else:
                    ax.set_yticklabels([])

    # --- Main Processing ---

    dates = sessions_data['dates']
    lick_properties_list = sessions_data['lick_properties']
    n_sessions = len(dates)
    isi_divider = lick_properties_list[0]['ISI_devider']

    # 1. Process Pooled Data
    pooled_data = initialize_storage()
    session_datasets = []

    all_isi_check = [] # For checking single ISI case globally

    for lick_props in lick_properties_list:
        sess_data = split_block_epochs(lick_props)
        session_datasets.append(sess_data)
        
        # Merge into pooled
        for b in ['short', 'long', 'all']:
            for e in ['early', 'late']:
                for t in ['opto', 'control']:
                    isi_vals = sess_data[b][e][t]['isi']
                    choice_vals = sess_data[b][e][t]['choices']
                    
                    pooled_data[b][e][t]['isi'].extend(isi_vals)
                    pooled_data[b][e][t]['choices'].extend(choice_vals)
                    all_isi_check.extend(isi_vals)

    # Check for single ISI case
    single_isi_case = len(np.unique(np.round(all_isi_check, 3))) == 2

    # 2. Plotting
    with PdfPages(output_pdf) as pdf:
        # Create figure: Very wide to accommodate 12 columns
        # (Width, Height)
        fig = plt.figure(figsize=(40, 4 * (n_sessions + 1)))
        gs = gridspec.GridSpec(n_sessions + 1, 12, figure=fig)
        
        # Plot Pooled (Row 0)
        axes_pool = [fig.add_subplot(gs[0, i]) for i in range(12)]
        plot_row(axes_pool, pooled_data, isi_divider, single_isi_case, row_title_prefix='Pooled')

        # Plot Sessions (Rows 1..N)
        for i, (date, sess_data) in enumerate(zip(dates, session_datasets)):
            axes_sess = [fig.add_subplot(gs[i + 1, c]) for c in range(12)]
            plot_row(axes_sess, sess_data, isi_divider, single_isi_case, row_title_prefix=f'{date}')

        plt.tight_layout()
        
        if save_path:
            filename = f'Psychometric_Epochs_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf'
            output_path = os.path.join(save_path, filename)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            
        plt.close(fig)