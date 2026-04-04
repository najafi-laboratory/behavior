def plot_psychometric_control_opto_epochs(sessions_data, subject, data_paths, output_pdf='psychometric_opto_control_comparisons.pdf', bin_width=0.1, fit_logistic=True, save_path=None):
    """
    Modified to plot 6 specific comparisons per block type (12 columns total).
    Includes logic sanity checks to ensure strict separation of control vs non-opto trials.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.gridspec as gridspec
    from scipy.stats import sem
    from scipy.optimize import curve_fit
    import os
    
    # Extract session data
    dates = sessions_data['dates']
    lick_properties_list = sessions_data['lick_properties']
    n_sessions = len(dates)
    
    # --- 1. ROBUST DATA PARSING ---
    def identify_blocks_with_opto_status(lick_properties):
        """
        Scans trials to identify contiguous blocks and flags if the block
        contains ANY opto trials.
        """
        all_trials = []
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
        
        # Aggregation of all trials from the sparse lists
        for key in left_keys + right_keys:
            trial_isi = lick_properties[key].get('Trial_ISI', [])
            opto_tags = lick_properties[key].get('opto_tag', [])
            block_types = lick_properties[key].get('block_type', [])
            epochs = lick_properties[key].get('epoch', [])
            is_right = 1 if key in right_keys else 0
            
            for i in range(len(trial_isi)):
                # Only consider valid trials (not None) that are not in neutral blocks (0)
                if trial_isi[i] is not None and block_types[i] != 0:
                    all_trials.append({
                        'index': i, 'block_type': block_types[i], 'opto_tag': opto_tags[i],
                        'epoch': epochs[i], 'isi': trial_isi[i], 'choice': is_right
                    })
        
        if not all_trials: return [], []
        
        # Sort by trial index to restore temporal order
        all_trials.sort(key=lambda x: x['index'])
        
        blocks = []
        current_block_type = None
        start_idx = 0
        has_opto_in_block = False
        
        for i, trial in enumerate(all_trials):
            bt = trial['block_type']
            
            if current_block_type is None:
                current_block_type = bt
                start_idx = i
                has_opto_in_block = (trial['opto_tag'] == 1)
            
            elif current_block_type != bt:
                # Block transition detected: Save previous block
                blocks.append((current_block_type, start_idx, i, has_opto_in_block))
                
                # Start new block
                current_block_type = bt
                start_idx = i
                # Reset opto flag for the new block
                has_opto_in_block = (trial['opto_tag'] == 1)
            
            else:
                # Within the same block, check if this trial triggers the opto flag
                if trial['opto_tag'] == 1: 
                    has_opto_in_block = True
        
        # Append the final block
        if current_block_type is not None:
            blocks.append((current_block_type, start_idx, len(all_trials), has_opto_in_block))
            
        return blocks, all_trials

    def split_trials_by_block_type(lick_properties):
        """
        Splits trials into Control, Opto, and Non-Opto (within Opto block) categories.
        """
        blocks, all_trials = identify_blocks_with_opto_status(lick_properties)
        
        # Factory for creating fresh data structures
        data_struct = lambda: {'short': {'early': {'isi': [], 'choices': []}, 'late': {'isi': [], 'choices': []}},
                               'long': {'early': {'isi': [], 'choices': []}, 'late': {'isi': [], 'choices': []}}}
        
        control_data = data_struct()
        opto_data = data_struct()
        non_opto_data = data_struct()
        
        if not blocks: return control_data, opto_data, non_opto_data
        
        for block_type, start_idx, end_idx, has_opto in blocks:
            # Map block type ID to string key
            block_key = 'short' if block_type == 1 else 'long' if block_type == 2 else None
            if block_key is None: continue
            
            # Slice trials belonging to this specific block
            block_trials = all_trials[start_idx:end_idx]
            
            for trial in block_trials:
                epoch_key = 'early' if trial['epoch'] == 1 else 'late'
                
                if not has_opto:
                    # CASE 1: CONTROL BLOCK
                    # Block contains ZERO opto trials. All trials here are Control.
                    control_data[block_key][epoch_key]['isi'].append(trial['isi'])
                    control_data[block_key][epoch_key]['choices'].append(trial['choice'])
                else:
                    # CASE 2: OPTO BLOCK
                    # Block contains AT LEAST ONE opto trial.
                    if trial['opto_tag'] == 1:
                        # This is the actual Opto trial
                        opto_data[block_key][epoch_key]['isi'].append(trial['isi'])
                        opto_data[block_key][epoch_key]['choices'].append(trial['choice'])
                    else:
                        # This is a Non-Opto trial interacting with Opto context
                        non_opto_data[block_key][epoch_key]['isi'].append(trial['isi'])
                        non_opto_data[block_key][epoch_key]['choices'].append(trial['choice'])
                        
        return control_data, opto_data, non_opto_data

    # --- 2. MATH & FITTING HELPER FUNCTIONS ---
    def calculate_psychometric(isi, choices, bin_width, single_isi_case):
        if len(isi) == 0: return None, None, None
        isi = np.array(isi)
        choices = np.array(choices)
        
        if single_isi_case:
            # For cases with only 2 ISI values (no binning needed)
            unique_isi = np.unique(isi)
            if len(unique_isi) < 1: return None, None, None
            right_prob = np.zeros(len(unique_isi))
            sem_values = np.zeros(len(unique_isi))
            counts = np.zeros(len(unique_isi))
            
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
            # For continuous ISI (binning required)
            min_isi = np.floor(np.min(isi) / bin_width) * bin_width
            max_isi = np.ceil(np.max(isi) / bin_width) * bin_width
            bins = np.arange(min_isi, max_isi + bin_width, bin_width)
            if len(bins) < 2: return None, None, None 
            
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
        if x_data is None or len(x_data) == 0: return False
        
        n_trials = len(choices)
        label_with_count = f'{label}\n(n={n_trials})'
        
        # Plot data points with error bars
        ax.errorbar(x_data, y_data, yerr=y_sem, fmt='o', color=color, capsize=3, alpha=0.7, label=label_with_count, markersize=4)
        
        # Curve Fitting Logic
        if single_isi_case and len(x_data) == 2:
            # Simple line for 2-point data
            ax.plot(x_data, y_data, '-', color=color, linewidth=2, alpha=0.7)
            x1, x2 = x_data
            y1, y2 = y_data
            if y2 != y1:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                inflection_point = (0.5 - b) / m
                # Only plot inflection line if it falls between the two points
                if min(x1, x2) <= inflection_point <= max(x1, x2):
                    ax.axvline(x=inflection_point, color=color, linestyle='--', alpha=0.5)
                    
        elif fit_logistic and len(x_data) > 3:
            # Logistic regression for >3 points
            try:
                popt, _ = curve_fit(logistic_function, x_data, y_data, p0=[1.0, 1.0, np.median(x_data)],
                                    bounds=([0.5, -10, np.min(x_data)], [1, 10, np.max(x_data)]))
                x_fit = np.linspace(np.min(x_data), np.max(x_data), 100)
                y_fit = logistic_function(x_fit, *popt)
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2, alpha=0.7)
                
                # Calculate inflection point (x where y=0.5)
                ip_index = np.argmin(np.abs(y_fit - 0.5))
                inflection_point = x_fit[ip_index]
                # ax.axvline(x=inflection_point, color=color, linestyle='--', alpha=0.5)
            except: 
                pass # Fail silently if fit doesn't converge
                
        return True

    # --- 3. CONFIGURATION & POOLING ---
    
    colors = {
        'control_early': '#808080',      # Gray
        'control_late': '#000000',       # Black
        'opto_early': '#87CEEB',         # Sky Blue
        'opto_late': '#0000FF',          # Blue
        'non_opto_early': '#FFB6C6',     # Light Pink
        'non_opto_late': '#DC143C'       # Crimson
    }
    
    # Updated Legend names
    display_names = {
        'control_early': 'Control Early',
        'control_late': 'Control Late',
        'opto_early': 'Opto Early',
        'opto_late': 'Opto Late',
        'non_opto_early': 'Non-Opto (in Opto Block) Early',
        'non_opto_late': 'Non-Opto (in Opto Block) Late'
    }

    # Define the 6 columns of comparisons as requested
    # Format: (Title, [(data_type, epoch), (data_type, epoch)])
    comparisons = [
        ('Ctrl Early vs Opto Early',        [('control', 'early'), ('opto', 'early')]),
        ('Ctrl Late vs Opto Late',          [('control', 'late'), ('opto', 'late')]),
        ('Ctrl Early vs Ctrl Late',         [('control', 'early'), ('control', 'late')]),
        ('Ctrl Early vs Non-Opto Early',    [('control', 'early'), ('non_opto', 'early')]),
        ('Ctrl Late vs Non-Opto Late',      [('control', 'late'), ('non_opto', 'late')]),
        ('Non-Opto Early vs Non-Opto Late', [('non_opto', 'early'), ('non_opto', 'late')])
    ]

    # Initialize pooling containers
    pooled = {
        'control': {'short': {'early': {'isi': [], 'choices': []}, 'late': {'isi': [], 'choices': []}},
                    'long': {'early': {'isi': [], 'choices': []}, 'late': {'isi': [], 'choices': []}}},
        'opto':    {'short': {'early': {'isi': [], 'choices': []}, 'late': {'isi': [], 'choices': []}},
                    'long': {'early': {'isi': [], 'choices': []}, 'late': {'isi': [], 'choices': []}}},
        'non_opto':{'short': {'early': {'isi': [], 'choices': []}, 'late': {'isi': [], 'choices': []}},
                    'long': {'early': {'isi': [], 'choices': []}, 'late': {'isi': [], 'choices': []}}}
    }

    session_data_store = [] 

    # --- 4. DATA PROCESSING LOOP ---
    for lick_props in lick_properties_list:
        c_dat, o_dat, no_dat = split_trials_by_block_type(lick_props)
        session_data_store.append({'control': c_dat, 'opto': o_dat, 'non_opto': no_dat})
        
        # Accumulate into pooled data (Extend lists)
        for b_type in ['short', 'long']:
            for ep in ['early', 'late']:
                pooled['control'][b_type][ep]['isi'].extend(c_dat[b_type][ep]['isi'])
                pooled['control'][b_type][ep]['choices'].extend(c_dat[b_type][ep]['choices'])
                pooled['opto'][b_type][ep]['isi'].extend(o_dat[b_type][ep]['isi'])
                pooled['opto'][b_type][ep]['choices'].extend(o_dat[b_type][ep]['choices'])
                pooled['non_opto'][b_type][ep]['isi'].extend(no_dat[b_type][ep]['isi'])
                pooled['non_opto'][b_type][ep]['choices'].extend(no_dat[b_type][ep]['choices'])

    # Determine Global Single ISI Case
    all_isi = []
    for b_type in ['short', 'long']:
        for ep in ['early', 'late']:
            all_isi.extend(pooled['control'][b_type][ep]['isi'])
    
    # If we have data, check unique ISIs (rounded to 3 decimals to avoid float jitter)
    single_isi_case = False
    if len(all_isi) > 0:
        unique_rounded = np.unique(np.round(all_isi, 3))
        single_isi_case = (len(unique_rounded) == 2)

    isi_divider = lick_properties_list[0]['ISI_devider']

    # --- 5. GENERATE PDF PLOTS ---
    with PdfPages(output_pdf) as pdf:
        # Increase figure width to accommodate 12 columns
        fig = plt.figure(figsize=(40, 4 * (n_sessions + 1))) 
        
        # Add Super Title
        fig.suptitle(f"{subject} - Analysis of {n_sessions} Sessions", fontsize=20, y=0.99)
        
        # Grid: Rows = (Pooled + N Sessions), Columns = 12 (6 Short comparisons + 6 Long comparisons)
        gs = gridspec.GridSpec(n_sessions + 1, 12, figure=fig)
        
        # Prepare list of rows to iterate: first pooled, then individual sessions
        rows_to_plot = [('Pooled', pooled)] + list(zip(dates, session_data_store))

        for row_idx, (row_label, data_source) in enumerate(rows_to_plot):
            
            # --- PLOT SHORT BLOCK COLUMNS (0-5) ---
            for col_idx, (comp_name, pairs) in enumerate(comparisons):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                has_data = False
                
                for dtype, epoch in pairs:
                    isi_vals = data_source[dtype]['short'][epoch]['isi']
                    choice_vals = data_source[dtype]['short'][epoch]['choices']
                    
                    key = f"{dtype}_{epoch}"
                    # Check if plot was successful (returns True if data existed)
                    if plot_psychometric(ax, isi_vals, choice_vals, display_names[key], colors[key], 
                                       isi_divider, single_isi_case, fit_logistic):
                        has_data = True
                
                # Axis Styling
                if row_idx == 0:
                    ax.set_title(f"SHORT: {comp_name}", fontsize=10, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f"{row_label}\nProb. Right", fontsize=9)
                else:
                    ax.set_yticklabels([]) # Clean look: Remove Y-axis labels for inner columns
                
                if has_data:
                    # ax.axvline(x=isi_divider, color='red', linestyle='--', alpha=0.3)
                    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)
                    ax.set_ylim(-0.05, 1.05)
                    ax.legend(fontsize=6, loc='best')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', transform=ax.transAxes, fontsize=8)

            # --- PLOT LONG BLOCK COLUMNS (6-11) ---
            for col_idx, (comp_name, pairs) in enumerate(comparisons):
                # Offset grid column index by 6
                ax = fig.add_subplot(gs[row_idx, col_idx + 6])
                has_data = False
                
                for dtype, epoch in pairs:
                    isi_vals = data_source[dtype]['long'][epoch]['isi']
                    choice_vals = data_source[dtype]['long'][epoch]['choices']
                    
                    key = f"{dtype}_{epoch}"
                    if plot_psychometric(ax, isi_vals, choice_vals, display_names[key], colors[key], 
                                       isi_divider, single_isi_case, fit_logistic):
                        has_data = True

                # Axis Styling
                if row_idx == 0:
                    ax.set_title(f"LONG: {comp_name}", fontsize=10, fontweight='bold')
                
                ax.set_yticklabels([]) # Hide Y ticks for all Long columns
                
                if has_data:
                    # ax.axvline(x=isi_divider, color='red', linestyle='--', alpha=0.3)
                    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)
                    ax.set_ylim(-0.05, 1.05)
                    ax.legend(fontsize=6, loc='best')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', transform=ax.transAxes, fontsize=8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Leave room for suptitle
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
        # Optionally save individual PDF for this run
        if save_path:
            output_path = os.path.join(save_path, f'Psychometric_Comparison_optoblock_vs_controlblock_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)
    
    print(f"Psychometric comparison analysis complete. PDF saved to {output_pdf}")