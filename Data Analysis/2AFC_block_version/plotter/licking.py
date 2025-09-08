import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Single sessions plotter ------------------------------------------------------------

# Plotting function for psychometric curve
def plot_psychometric_curve(lick_properties, filter_outcomes='all', ax=None, bin_width=0.01, fit_logistic=True, opto_split=False):
    """
    Plot psychometric curves showing probability of right port choice as a function of ISI with SEM.
    Superimposes curves for block_type 1 (Short) and block_type 2 (Long).
    Handles both single ISI per condition (short/long) and multiple ISI values.
    
    Parameters:
    -----------
    lick_properties : dict
        Dictionary containing lick properties from extract_lick_properties function
    filter_outcomes : str, optional
        'all' for both rewarded and punished trials, 'rewarded' for rewarded trials only, 'punished' for punished trials only
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    bin_width : float, optional
        Width of ISI bins for calculating choice probability (used when multiple ISIs)
    fit_logistic : bool, optional
        Whether to fit and plot a logistic function to the data (used when multiple ISIs)
    opto_split : bool, optional
        Whether to split data by opto_tag (control vs. opto trials)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict or None
        Parameters of the logistic fit (for multiple ISIs) or linear fit (for single ISIs), if applicable
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Define colors based on filter_outcomes and block_type
    colors = {
        'all': {'block0': '#999999', 'block1': '#EC882B', 'block2': '#ff00ff'},
        'rewarded': {'block0': '#a1e0a1', 'block1': '#63f250', 'block2': '#00cc00'},
        'punished': {'block0': '#f7b0a9', 'block1': '#e74c3c', 'block2': '#cc0000'},
        'opto': {'block0': '#005eff', 'block1': '#9999ff', 'block2': '#0033cc'},
        'control': {'block0': '#999999', 'block1': "#EC882B", 'block2': '#ff00ff'}
    }
    
    # Extract data efficiently
    def extract_lick_data(keys):
        isi = []
        opto = []
        block = []
        for key in keys:
            isi.extend(lick_properties[key]['Trial_ISI'])
            opto.extend(lick_properties[key]['opto_tag'])
            block.extend(lick_properties[key]['block_type'])
        return np.array(isi), np.array(opto), np.array(block)
    
    # Get ISI, opto tags, and block types for left and right choices
    if filter_outcomes == 'all':
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'rewarded':
        left_keys = ['short_ISI_reward_left_correct_lick']
        right_keys = ['long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'punished':
        left_keys = ['long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick']
    
    left_isi, left_opto, left_block = extract_lick_data(left_keys)
    right_isi, right_opto, right_block = extract_lick_data(right_keys)
    
    # Combine data
    isi_values = np.concatenate([left_isi, right_isi])
    choices = np.concatenate([np.zeros(len(left_isi)), np.ones(len(right_isi))])
    opto_tags = np.concatenate([left_opto, right_opto])
    block_types = np.concatenate([left_block, right_block])
    
    # Check for single ISI per condition
    unique_left_isi = np.unique(left_isi)
    unique_right_isi = np.unique(right_isi)
    single_isi_case = len(unique_left_isi) <= 1 and len(unique_right_isi) <= 1 and len(np.unique(isi_values)) == 2
    
    # Define logistic function for multiple ISI case
    def logistic_function(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    
    # Process data for each block type
    def plot_group(isi, choices, block_types, label, color, opto_value=None, block_type=None):
        # Ensure all arrays are numpy arrays
        isi = np.array(isi)
        choices = np.array(choices)
        block_types = np.array(block_types)
        local_opto_tags = np.array(opto_tags)  # use the global opto_tags variable
    
        # Build a combined mask so all arrays are filtered together
        mask = np.ones(len(isi), dtype=bool)
        if block_type is not None:
            mask &= (block_types == block_type)
        if opto_value is not None:
            mask &= (local_opto_tags == opto_value)
        isi, choices = isi[mask], choices[mask]
        
        if len(isi) == 0:
            return None
        
        # Map block_type to label
        if block_type == 0:
            block_label = 'Neutral'
        elif block_type == 1:
            block_label = 'Short'
        elif block_type == 2:
            block_label = 'Long'
        else:
            block_label = f'Block {block_type}'
        
        # Handle single ISI case
        if single_isi_case:
            unique_isi = np.unique(isi)
            if len(unique_isi) != 2:
                return None  # Need exactly two ISI values
            
            right_prob = np.zeros(2)
            sem = np.zeros(2)
            counts = np.zeros(2)
            
            for i, isi_val in enumerate(unique_isi):
                mask = isi == isi_val
                bin_choices = choices[mask]
                if len(bin_choices) > 0:
                    right_prob[i] = np.mean(bin_choices)
                    sem[i] = np.std(bin_choices) / np.sqrt(len(bin_choices))
                    counts[i] = len(bin_choices)
            
            valid_mask = counts > 0
            valid_isi = unique_isi[valid_mask]
            valid_prob = right_prob[valid_mask]
            valid_sem = sem[valid_mask]
            
            # Plot points with error bars
            ax.errorbar(valid_isi, valid_prob, yerr=valid_sem, fmt='o', color=color,
                       label=f"{label} data ({block_label})", capsize=3, alpha=0.7)
            
            # Connect points with a line
            ax.plot(valid_isi, valid_prob, '-', color=color, linewidth=2,
                   label=f"{label} fit ({block_label})")
            
            # Calculate inflection point (where prob = 0.5)
            x1, x2 = valid_isi
            y1, y2 = valid_prob
            fit_params = None
            if y2 != y1:  # Avoid division by zero
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                inflection_point = (0.5 - b) / m
                if min(x1, x2) <= inflection_point <= max(x1, x2):
                    ax.axvline(x=inflection_point, color=color, linestyle='--', alpha=0.5)
                    ax.text(inflection_point, 0.1, f'{label} IP ({block_label}): {inflection_point:.2f}s',
                           color=color, ha='center', bbox=dict(facecolor='white', alpha=0.7))
                    fit_params = {'x0': inflection_point, 'slope': m, 'intercept': b}
            
            return fit_params
        
        # Original multiple ISI case
        else:
            # Create bins
            min_isi = np.floor(isi.min() / bin_width) * bin_width
            max_isi = np.ceil(isi.max() / bin_width) * bin_width
            bins = np.arange(min_isi, max_isi + bin_width, bin_width)
            bin_centers = bins[:-1] + bin_width/2
            
            # Calculate statistics
            right_prob = np.zeros(len(bins) - 1)
            sem = np.zeros(len(bins) - 1)
            counts = np.zeros(len(bins) - 1)
            
            for i in range(len(bins) - 1):
                mask = (isi >= bins[i]) & (isi < bins[i + 1])
                bin_choices = choices[mask]
                if len(bin_choices) > 0:
                    right_prob[i] = np.mean(bin_choices)
                    sem[i] = np.std(bin_choices) / np.sqrt(len(bin_choices))
                    counts[i] = len(bin_choices)
            
            valid_mask = counts > 0
            valid_centers = bin_centers[valid_mask]
            valid_prob = right_prob[valid_mask]
            valid_sem = sem[valid_mask]
            
            # Plot with error bars
            ax.errorbar(valid_centers, valid_prob, yerr=valid_sem, fmt='o', color=color,
                       label=f"{label} data ({block_label})", capsize=3, alpha=0.7)
            
            # Fit logistic curve
            fit_params = None
            if fit_logistic and len(valid_centers) > 3:
                try:
                    modified_bounds = ([0.5, -10, min_isi], [1, 10, max_isi])
                    popt, _ = curve_fit(logistic_function, valid_centers, valid_prob,
                                      p0=[1.0, 1.0, np.median(valid_centers)],
                                      bounds=modified_bounds)
                    
                    x_fit = np.linspace(min_isi, max_isi, 100)
                    y_fit = logistic_function(x_fit, *popt)
                    ax.plot(x_fit, y_fit, '-', color=color, linewidth=2,
                           label=f"{label} fit ({block_label})")
                    ip_index = np.argmin(np.abs(y_fit - 0.5))
                    ip_value = x_fit[ip_index]
                    inflection_point = ip_value
                    ax.axvline(x=inflection_point, color=color, linestyle='--', alpha=0.5)
                    # ax.text(inflection_point, 0.1, f'{label} IP ({block_label}): {inflection_point:.2f}s',
                    #        color=color, ha='center', bbox=dict(facecolor='white', alpha=0.7))
                    
                    fit_params = {'L': popt[0], 'k': popt[1], 'x0': popt[2]}
                except Exception as e:
                    print(f"Could not fit logistic function for {label} ({block_label}): {e}")
            
            return fit_params
    
    # Plot data for each block type
    fit_params = {}
    title_suffix = filter_outcomes.capitalize() + ' Trials'
    
    if opto_split:
        # Plot control trials for both block types
        fit_params['control'] = {}
        for block_type in [0, 1, 2]:
            if np.any(block_types == block_type):
                fit_params['control'][f'block{block_type}'] = plot_group(
                    isi_values, choices, block_types, 'Control',
                    colors['control'][f'block{block_type}'], opto_value=0, block_type=block_type
                )
        
        # Plot opto trials for both block types if they exist
        if np.any(opto_tags == 1):
            fit_params['opto'] = {}
            for block_type in [0, 1, 2]:
                if np.any(block_types == block_type):
                    fit_params['opto'][f'block{block_type}'] = plot_group(
                        isi_values, choices, block_types, 'Opto',
                        colors['opto'][f'block{block_type}'], opto_value=1, block_type=block_type
                    )
    else:
        # Plot for both block types
        for block_type in [0, 1, 2]:
            if np.any(block_types == block_type):
                fit_params[f'block{block_type}'] = plot_group(
                    isi_values, choices, block_types, filter_outcomes.capitalize(),
                    colors[filter_outcomes][f'block{block_type}'], block_type=block_type
                )
    
    # Add ISI divider if available
    if 'ISI_devider' in lick_properties:
        isi_divider = lick_properties['ISI_devider']
        ax.axvline(x=isi_divider, color='r', linestyle='--', alpha=0.3)
    
    # Customize plot
    ax.set_xlabel('Inter-Stimulus Interval (s)', fontsize=12)
    ax.set_ylabel('Probability of Right Choice', fontsize=12)
    ax.set_title(f'Psychometric Curve - {title_suffix} - ' + lick_properties['session_date'], fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at y=0.5
    ax.axhline(y=0.5, color='k', linestyle='-', alpha=0.2)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    return fig, ax, fit_params


# Plotting function for lick reaction time curve
def plot_reaction_time_curve(lick_properties, filter_outcomes='all', ax=None, bin_width=0.05, fit_quadratic=True, opto_split=False):
    """
    Plot lick reaction time as a function of ISI with SEM error bars and quadratic fit.
    Superimposes curves for block_type 1 (Short) and block_type 2 (Long).
    
    Parameters:
    -----------
    lick_properties : dict
        Dictionary containing lick properties with structure:
        {'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
         'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': []}
    filter_outcomes : str, optional
        'all' for both rewarded and punished trials, 'rewarded' for rewarded trials only, 'punished' for punished trials only
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    bin_width : float, optional
        Width of ISI bins for calculating reaction time statistics
    fit_quadratic : bool, optional
        Whether to fit and plot a quadratic function to the data
    opto_split : bool, optional
        Whether to split data by opto_tag (control vs. opto trials)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict or None
        Parameters of the quadratic fit, if fit_quadratic=True
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Define colors based on filter_outcomes and block_type
    colors = {
        'all': {'block0': '#999999', 'block1': '#EC882B', 'block2': '#ff00ff'},
        'rewarded': {'block0': '#a1e0a1', 'block1': '#63f250', 'block2': '#00cc00'},
        'punished': {'block0': '#f7b0a9', 'block1': '#e74c3c', 'block2': '#cc0000'},
        'opto': {'block0': '#005eff', 'block1': '#9999ff', 'block2': '#0033cc'},
        'control': {'block0': '#999999', 'block1': "#EC882B", 'block2': '#ff00ff'}
    }
    
    # Extract data efficiently with validation
    def extract_lick_data(keys):
        isi = []
        reaction_times = []
        opto = []
        block = []
        for key in keys:
            if key not in lick_properties:
                continue
            # Ensure all required fields exist and are aligned
            trial_isi = np.array(lick_properties[key]['Trial_ISI'])
            trial_rt = np.array(lick_properties[key]['Lick_reaction_time'])
            trial_opto = np.array(lick_properties[key]['opto_tag'])
            trial_block = np.array(lick_properties[key]['block_type'])
            
            # Validate lengths
            min_length = min(len(trial_isi), len(trial_rt), len(trial_opto), len(trial_block))
            if min_length == 0:
                continue
                
            # Truncate to shortest length to ensure alignment
            valid_mask = (
                ~np.isnan(trial_isi[:min_length]) & 
                ~np.isnan(trial_rt[:min_length]) & 
                ~np.isnan(trial_opto[:min_length]) &
                ~np.isnan(trial_block[:min_length])
            )
            
            isi.extend(trial_isi[:min_length][valid_mask])
            reaction_times.extend(trial_rt[:min_length][valid_mask])
            opto.extend(trial_opto[:min_length][valid_mask])
            block.extend(trial_block[:min_length][valid_mask])
        
        return np.array(isi), np.array(reaction_times), np.array(opto), np.array(block)
    
    # Get ISI, reaction times, opto tags, and block types for left and right choices
    if filter_outcomes == 'all':
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'rewarded':
        left_keys = ['short_ISI_reward_left_correct_lick']
        right_keys = ['long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'punished':
        left_keys = ['long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick']
    
    left_isi, left_rt, left_opto, left_block = extract_lick_data(left_keys)
    right_isi, right_rt, right_opto, right_block = extract_lick_data(right_keys)
    
    # Combine data
    isi_values = np.concatenate([left_isi, right_isi])
    reaction_times = np.concatenate([left_rt, right_rt])
    opto_tags = np.concatenate([left_opto, right_opto])
    block_types = np.concatenate([left_block, right_block])
    
    # Verify array lengths
    if not (len(isi_values) == len(reaction_times) == len(opto_tags) == len(block_types)):
        raise ValueError(f"Array length mismatch: ISI={len(isi_values)}, "
                        f"Reaction Times={len(reaction_times)}, Opto Tags={len(opto_tags)}, "
                        f"Block Types={len(block_types)}")
    
    if len(isi_values) == 0:
        print("No valid data to plot after filtering.")
        return fig, ax, None
    
    # Create bins
    min_isi = np.floor(isi_values.min() / bin_width) * bin_width
    max_isi = np.ceil(isi_values.max() / bin_width) * bin_width
    bins = np.arange(min_isi, max_isi + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width/2
    
    # Define quadratic function for fitting
    def quadratic_function(x, a, b, c):
        return a * x**2 + b * x + c
    
    # Process data
    def plot_group(isi, rt, block_types, label, color, opto_value=None, block_type=None):
        # Validate array lengths first
        if not (len(isi) == len(rt) == len(block_types) == len(opto_tags)):
            print(f"Warning: Array length mismatch — ISI: {len(isi)}, RT: {len(rt)}, Block: {len(block_types)}, Opto: {len(opto_tags)}")
            return None

        # Build full boolean mask
        mask = np.ones(len(isi), dtype=bool)
        if block_type is not None:
            mask &= (block_types == block_type)
        if opto_value is not None:
            mask &= (opto_tags == opto_value)

        # Apply mask to all relevant arrays
        isi = isi[mask]
        rt = rt[mask]
        
        if len(isi) == 0:
            return None
        
        # Map block_type to label
        if block_type == 0:
            block_label = 'Neutral'
        elif block_type == 1:
            block_label = 'Short'
        elif block_type == 2:
            block_label = 'Long'
        else:
            block_label = f'Block {block_type}'
                
        # Calculate statistics
        mean_rt = np.zeros(len(bins) - 1)
        sem = np.zeros(len(bins) - 1)
        counts = np.zeros(len(bins) - 1)
        
        for i in range(len(bins) - 1):
            mask = (isi >= bins[i]) & (isi < bins[i + 1])
            bin_rt = rt[mask]
            if len(bin_rt) > 0:
                mean_rt[i] = np.mean(bin_rt)
                sem[i] = np.std(bin_rt) / np.sqrt(len(bin_rt))
                counts[i] = len(bin_rt)
        
        valid_mask = counts > 0
        valid_centers = bin_centers[valid_mask]
        valid_rt = mean_rt[valid_mask]
        valid_sem = sem[valid_mask]
        
        # Plot with error bars using same color as curve
        ax.errorbar(valid_centers, valid_rt, yerr=valid_sem, fmt='o', color=color,
                   label=f"{label} data ({block_label})", capsize=3, alpha=0.7)
        
        # Fit quadratic curve
        fit_params = None
        if fit_quadratic and len(valid_centers) > 3:
            try:
                popt, _ = curve_fit(quadratic_function, valid_centers, valid_rt,
                                  p0=[0, 0, np.mean(valid_rt)])
                
                x_fit = np.linspace(min_isi, max_isi, 100)
                y_fit = quadratic_function(x_fit, *popt)
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2,
                       label=f"{label} fit ({block_label})")
                
                fit_params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
            except Exception as e:
                print(f"Could not fit quadratic function for {label} ({block_label}): {e}")
        
        return fit_params
    
    # Plot data
    fit_params = {}
    title_suffix = filter_outcomes.capitalize() + ' Trials'
    if opto_split:
        # Plot control trials for both block types
        fit_params['control'] = {}
        for block_type in [0, 1, 2]:
            if np.any(block_types == block_type):
                fit_params['control'][f'block{block_type}'] = plot_group(
                    isi_values, reaction_times, block_types, 'Control',
                    colors['control'][f'block{block_type}'], opto_value=0, block_type=block_type
                )
        
        # Plot opto trials for both block types if they exist
        if np.any(opto_tags == 1):
            fit_params['opto'] = {}
            for block_type in [0, 1, 2]:
                if np.any(block_types == block_type):
                    fit_params['opto'][f'block{block_type}'] = plot_group(
                        isi_values, reaction_times, block_types, 'Opto',
                        colors['opto'][f'block{block_type}'], opto_value=1, block_type=block_type
                    )
    else:
        # Plot for both block types
        for block_type in [0, 1, 2]:
            if np.any(block_types == block_type):
                fit_params[f'block{block_type}'] = plot_group(
                    isi_values, reaction_times, block_types, filter_outcomes.capitalize(),
                    colors[filter_outcomes][f'block{block_type}'], block_type=block_type
                )
    
    # Add ISI divider if available
    if 'ISI_devider' in lick_properties:
        isi_divider = lick_properties['ISI_devider']
        ax.axvline(x=isi_divider, color='r', linestyle='--', alpha=0.3)
    
    # Customize plot
    ax.set_xlabel('Inter-Stimulus Interval (s)', fontsize=12)
    ax.set_ylabel('Lick Reaction Time (s)', fontsize=12)
    ax.set_title(f'Reaction Time Curve - {title_suffix} - ' + lick_properties['session_date'], fontsize=14, fontweight='bold')
    ax.set_xlim(min_isi, max_isi)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    return fig, ax, fit_params

# error bars for lick reaction time
def plot_isi_reaction_time(lick_properties, filter_outcomes='all', ax=None, opto_split=False):
    """
    Plot mean lick reaction time ± SEM for short and long ISI trials.
    
    Parameters:
    -----------
    lick_properties : dict
        Dictionary containing lick properties with structure:
        {'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
         'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': []}
    filter_outcomes : str, optional
        'all' for both rewarded and punished trials, 'rewarded' for rewarded trials only, 'punished' for punished trials only
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    opto_split : bool, optional
        Whether to split data by opto_tag (control vs. opto trials)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    stats : dict or None
        Dictionary containing mean and SEM for each ISI group
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    
    # Define colors based on filter_outcomes
    colors = {
        'all': '#808080',      # Gray for all trials
        'rewarded': '#63f250', # Green for rewarded trials
        'punished': '#e74c3c', # Red for punished trials
        'opto': '#005eff',     # Blue for opto trials
        'control': '#808080'   # Gray for control trials when opto_split
    }
    
    # Extract data efficiently with validation
    def extract_lick_data(keys):
        isi = []
        reaction_times = []
        opto = []
        for key in keys:
            if key not in lick_properties:
                continue
            # Ensure all required fields exist and are aligned
            trial_isi = np.array(lick_properties[key]['Trial_ISI'])
            trial_rt = np.array(lick_properties[key]['Lick_reaction_time'])
            trial_opto = np.array(lick_properties[key]['opto_tag'])
            
            # Validate lengths
            min_length = min(len(trial_isi), len(trial_rt), len(trial_opto))
            if min_length == 0:
                continue
                
            # Truncate to shortest length and filter NaNs
            valid_mask = (
                ~np.isnan(trial_isi[:min_length]) & 
                ~np.isnan(trial_rt[:min_length]) & 
                ~np.isnan(trial_opto[:min_length])
            )
            
            isi.extend(trial_isi[:min_length][valid_mask])
            reaction_times.extend(trial_rt[:min_length][valid_mask])
            opto.extend(trial_opto[:min_length][valid_mask])
        
        return np.array(isi), np.array(reaction_times), np.array(opto)
    
    # Get ISI, reaction times, and opto tags for left and right choices
    if filter_outcomes == 'all':
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'rewarded':
        left_keys = ['short_ISI_reward_left_correct_lick']
        right_keys = ['long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'punished':
        left_keys = ['long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick']
    
    left_isi, left_rt, left_opto = extract_lick_data(left_keys)
    right_isi, right_rt, right_opto = extract_lick_data(right_keys)
    
    # Combine data
    isi_values = np.concatenate([left_isi, right_isi])
    reaction_times = np.concatenate([left_rt, right_rt])
    opto_tags = np.concatenate([left_opto, right_opto])
    
    # Verify array lengths
    if not (len(isi_values) == len(reaction_times) == len(opto_tags)):
        raise ValueError(f"Array length mismatch: ISI={len(isi_values)}, "
                        f"Reaction Times={len(reaction_times)}, Opto Tags={len(opto_tags)}")
    
    if len(isi_values) == 0:
        print("No valid data to plot after filtering.")
        return fig, ax, None
    
    # Determine ISI divider
    isi_divider = lick_properties.get('ISI_devider', np.median(isi_values))
    
    # Process data
    def plot_group(isi, rt, label, color, opto_value=None):
        if opto_value is not None:
            if len(opto_tags) != len(isi):
                print(f"Warning: Opto tags length ({len(opto_tags)}) does not match ISI length ({len(isi)})")
                return None
            mask = opto_tags == opto_value
            isi, rt = isi[mask], rt[mask]
        
        if len(isi) == 0:
            return None
        
        # Split into short and long ISI
        short_mask = isi <= isi_divider
        long_mask = isi > isi_divider
        
        # Calculate mean and SEM
        short_rt = rt[short_mask]
        long_rt = rt[long_mask]
        
        stats = {
            'short': {'mean': np.nan, 'sem': np.nan, 'count': 0},
            'long': {'mean': np.nan, 'sem': np.nan, 'count': 0}
        }
        
        if len(short_rt) > 0:
            stats['short']['mean'] = np.mean(short_rt)
            stats['short']['sem'] = np.std(short_rt) / np.sqrt(len(short_rt))
            stats['short']['count'] = len(short_rt)
        
        if len(long_rt) > 0:
            stats['long']['mean'] = np.mean(long_rt)
            stats['long']['sem'] = np.std(long_rt) / np.sqrt(len(long_rt))
            stats['long']['count'] = len(long_rt)
        
        # Plot error bars
        if stats['short']['count'] > 0 or stats['long']['count'] > 0:
            x = [0, 1]
            means = [stats['short']['mean'], stats['long']['mean']]
            sems = [stats['short']['sem'], stats['long']['sem']]
            valid_mask = ~np.isnan(means)
            
            ax.errorbar(np.array(x)[valid_mask], np.array(means)[valid_mask], 
                       yerr=np.array(sems)[valid_mask], fmt='o', color=color,
                       label=label, capsize=5, alpha=0.7)
        
        return stats
    
    # Plot data
    stats = None
    title_suffix = filter_outcomes.capitalize() + ' Trials'
    if opto_split:
        # Plot control trials
        stats = plot_group(isi_values, reaction_times, 'Control', colors['control'],
                          opto_value=0)
        
        # Plot opto trials if they exist
        if np.any(opto_tags == 1):
            opto_stats = plot_group(isi_values, reaction_times, 'Opto', colors['opto'],
                                   opto_value=1)
            stats = {'control': stats, 'opto': opto_stats}
    else:
        stats = plot_group(isi_values, reaction_times, filter_outcomes.capitalize(),
                          colors[filter_outcomes])
    
    # Customize plot
    ax.set_xlabel('ISI Category', fontsize=12)
    ax.set_ylabel('Lick Reaction Time (s)', fontsize=12)
    ax.set_title(f'Reaction Time vs ISI - {title_suffix}', fontsize=14, fontweight='bold')
    
    # Set x-axis
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['short_ISI', 'long_ISI'])
    
    # Set y-axis limits with padding
    # valid_rt = reaction_times[np.isfinite(reaction_times)]
    # if len(valid_rt) > 0:
    #     rt_min = max(0, valid_rt.min() - 0.1)
    #     rt_max = valid_rt.max() + 0.1
    #     ax.set_ylim(rt_min, rt_max)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Add task explanation
    # ax.text(0.02, 0.98, 'Reaction time from stimulus onset to lick',
    #         transform=ax.transAxes, fontsize=7, va='top',
    #         bbox=dict(facecolor='white', alpha=0.7))
    
    # Add ISI divider annotation
    # ax.text(0.5, 0.02, f'ISI Divider: {isi_divider:.2f}s',
    #         transform=ax.transAxes, fontsize=7, ha='center', va='bottom',
    #         bbox=dict(facecolor='white', alpha=0.7))
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    return fig, ax, stats

# Plotting functions for pooling sessions ------------------------------------------------------
# Plotting function for pooled psychometric curve
def plot_pooled_psychometric_curve(lick_properties_list, filter_outcomes='all', ax=None, bin_width=0.05, fit_logistic=True, opto_split=False):
    """
    Plot psychometric curve for pooled data from multiple sessions, showing probability of right port choice as a function of ISI with SEM.
    Superimposes curves for block_type 1 (Short) and block_type 2 (Long).
    Handles both continuous ISI values and fixed single ISI values for short and long trials.
    
    Parameters:
    -----------
    lick_properties_list : list of dict
        List of dictionaries, each containing lick properties from a session with structure:
        {'short_ISI_reward_left_correct_lick': {...}, 'long_ISI_punish_left_incorrect_lick': {...}, ...}
    filter_outcomes : str, optional
        'all' for both rewarded and punished trials, 'rewarded' for rewarded trials only, 'punished' for punished trials only
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    bin_width : float, optional
        Width of ISI bins for calculating choice probability (used for continuous ISI data)
    fit_logistic : bool, optional
        Whether to fit and plot a logistic function to the data (only for continuous ISI data)
    opto_split : bool, optional
        Whether to split data by opto_tag (control vs. opto trials)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict or None
        Parameters of the logistic fit, if fit_logistic=True and continuous ISI data
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Define colors based on filter_outcomes and block_type
    colors = {
        'all': {'block0': '#999999', 'block1': '#EC882B', 'block2': '#ff00ff'},
        'rewarded': {'block0': '#a1e0a1', 'block1': '#63f250', 'block2': '#00cc00'},
        'punished': {'block0': '#f7b0a9', 'block1': '#e74c3c', 'block2': '#cc0000'},
        'opto': {'block0': '#005eff', 'block1': '#9999ff', 'block2': '#0033cc'},
        'control': {'block0': '#999999', 'block1': "#EC882B", 'block2': '#ff00ff'}
    }
    
    # Extract data efficiently from all sessions
    def extract_lick_data(keys, lick_properties_list):
        isi = []
        opto = []
        block = []
        for lick_properties in lick_properties_list:
            for key in keys:
                if key not in lick_properties:
                    continue
                trial_isi = np.array(lick_properties[key]['Trial_ISI'])
                trial_opto = np.array(lick_properties[key]['opto_tag'])
                trial_block = np.array(lick_properties[key]['block_type'])
                
                # Validate lengths and filter NaNs
                min_length = min(len(trial_isi), len(trial_opto), len(trial_block))
                if min_length == 0:
                    continue
                
                valid_mask = (
                    ~np.isnan(trial_isi[:min_length]) & 
                    ~np.isnan(trial_opto[:min_length]) &
                    ~np.isnan(trial_block[:min_length])
                )
                
                isi.extend(trial_isi[:min_length][valid_mask])
                opto.extend(trial_opto[:min_length][valid_mask])
                block.extend(trial_block[:min_length][valid_mask])
        
        return np.array(isi), np.array(opto), np.array(block)
    
    # Get ISI, opto tags, and block types for left and right choices
    if filter_outcomes == 'all':
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'rewarded':
        left_keys = ['short_ISI_reward_left_correct_lick']
        right_keys = ['long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'punished':
        left_keys = ['long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick']
    
    left_isi, left_opto, left_block = extract_lick_data(left_keys, lick_properties_list)
    right_isi, right_opto, right_block = extract_lick_data(right_keys, lick_properties_list)
    
    # Combine data
    isi_values = np.concatenate([left_isi, right_isi])
    choices = np.concatenate([np.zeros(len(left_isi)), np.ones(len(right_isi))])
    opto_tags = np.concatenate([left_opto, right_opto])
    block_types = np.concatenate([left_block, right_block])
    
    # Verify array lengths
    if not (len(isi_values) == len(choices) == len(opto_tags) == len(block_types)):
        raise ValueError(f"Array length mismatch: ISI={len(isi_values)}, "
                        f"Choices={len(choices)}, Opto Tags={len(opto_tags)}, Block Types={len(block_types)}")
    
    if len(isi_values) == 0:
        print("No valid data to plot after pooling sessions.")
        return fig, ax, None
    
    # Check if ISI values are discrete (fixed short and long ISI)
    unique_isi = np.unique(isi_values)
    is_discrete = len(unique_isi) <= 2  # Assume discrete if 1 or 2 unique ISI values
    
    # Define logistic function
    def logistic_function(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    
    # Process data
    def plot_group(isi, choices, block_types, label, color, opto_value=None, block_type=None):
        # Check array lengths match
        if not (len(isi) == len(choices) == len(block_types) == len(opto_tags)):
            print(f"Warning: Length mismatch - ISI={len(isi)}, Choices={len(choices)}, Block={len(block_types)}, Opto={len(opto_tags)}")
            return None

        # Apply joint filtering
        mask = np.ones(len(isi), dtype=bool)
        if block_type is not None:
            mask &= (block_types == block_type)
        if opto_value is not None:
            mask &= (opto_tags == opto_value)

        isi = isi[mask]
        choices = choices[mask]
        
        if len(isi) == 0:
            return None
        
        # Map block_type to label
        if block_type == 0:
            block_label = 'Neutral'
        elif block_type == 1:
            block_label = 'Short'
        elif block_type == 2:
            block_label = 'Long'
        else:
            block_label = f'Block {block_type}'
        
        if is_discrete:
            # Handle discrete ISI values (fixed short and long ISI)
            isi_centers = np.sort(np.unique(isi))
            right_prob = np.zeros(len(isi_centers))
            sem = np.zeros(len(isi_centers))
            counts = np.zeros(len(isi_centers))
            
            for i, isi_val in enumerate(isi_centers):
                mask = isi == isi_val
                bin_choices = choices[mask]
                if len(bin_choices) > 0:
                    right_prob[i] = np.mean(bin_choices)
                    sem[i] = np.std(bin_choices) / np.sqrt(len(bin_choices))
                    counts[i] = len(bin_choices)
            
            valid_mask = counts > 0
            valid_centers = isi_centers[valid_mask]
            valid_prob = right_prob[valid_mask]
            valid_sem = sem[valid_mask]
            
            # Plot discrete points with error bars
            ax.errorbar(valid_centers, valid_prob, yerr=valid_sem, fmt='o', color=color,
                       label=f"{label} data ({block_label})", capsize=3, alpha=0.7)
            
            # Connect points with a line
            if len(valid_centers) > 1:
                ax.plot(valid_centers, valid_prob, '-', color=color, linewidth=2,
                       label=f"{label} line ({block_label})", alpha=0.7)
            
            fit_params = None  # No logistic fit for discrete data
            
        else:
            # Handle continuous ISI values (original binning logic)
            min_isi = np.floor(isi.min() / bin_width) * bin_width
            max_isi = np.ceil(isi.max() / bin_width) * bin_width
            bins = np.arange(min_isi, max_isi + bin_width, bin_width)
            bin_centers = bins[:-1] + bin_width/2
            
            # Calculate statistics
            right_prob = np.zeros(len(bins) - 1)
            sem = np.zeros(len(bins) - 1)
            counts = np.zeros(len(bins) - 1)
            
            for i in range(len(bins) - 1):
                mask = (isi >= bins[i]) & (isi < bins[i + 1])
                bin_choices = choices[mask]
                if len(bin_choices) > 0:
                    right_prob[i] = np.mean(bin_choices)
                    sem[i] = np.std(bin_choices) / np.sqrt(len(bin_choices))
                    counts[i] = len(bin_choices)
            
            valid_mask = counts > 0
            valid_centers = bin_centers[valid_mask]
            valid_prob = right_prob[valid_mask]
            valid_sem = sem[valid_mask]
            
            # Plot with error bars
            ax.errorbar(valid_centers, valid_prob, yerr=valid_sem, fmt='o', color=color,
                       label=f"{label} data ({block_label})", capsize=3, alpha=0.7)
            
            # Fit logistic curve
            fit_params = None
            if fit_logistic and len(valid_centers) > 3:
                try:
                    modified_bounds = ([0.5, -10, min_isi], [1, 10, max_isi])
                    popt, _ = curve_fit(logistic_function, valid_centers, valid_prob,
                                      p0=[1.0, 1.0, np.median(valid_centers)],
                                      bounds=modified_bounds)
                    
                    x_fit = np.linspace(min_isi, max_isi, 100)
                    y_fit = logistic_function(x_fit, *popt)
                    ax.plot(x_fit, y_fit, '-', color=color, linewidth=2,
                           label=f"{label} fit ({block_label})")
                    ip_index = np.argmin(np.abs(y_fit - 0.5))
                    ip_value = x_fit[ip_index]
                    inflection_point = ip_value
                    ax.axvline(x=inflection_point, color=color, linestyle='--', alpha=0.5)
                    # ax.text(inflection_point, 0.1, f'{label} IP ({block_label}): {inflection_point:.2f}s',
                    #        color=color, ha='center', bbox=dict(facecolor='white', alpha=0.7))
                    
                    fit_params = {'L': popt[0], 'k': popt[1], 'x0': popt[2]}
                except Exception as e:
                    print(f"Could not fit logistic function for {label} ({block_label}): {e}")
        
        return fit_params
    
    # Plot data
    fit_params = {}
    title_suffix = filter_outcomes.capitalize() + ' Trials'
    if opto_split:
        # Plot control trials for both block types
        fit_params['control'] = {}
        for block_type in [0, 1, 2]:
            if np.any(block_types == block_type):
                fit_params['control'][f'block{block_type}'] = plot_group(
                    isi_values, choices, block_types, 'Control',
                    colors['control'][f'block{block_type}'], opto_value=0, block_type=block_type
                )
        
        # Plot opto trials for both block types if they exist
        if np.any(opto_tags == 1):
            fit_params['opto'] = {}
            for block_type in [0, 1, 2]:
                if np.any(block_types == block_type):
                    fit_params['opto'][f'block{block_type}'] = plot_group(
                        isi_values, choices, block_types, 'Opto',
                        colors['opto'][f'block{block_type}'], opto_value=1, block_type=block_type
                    )
    else:
        # Plot for both block types
        for block_type in [0, 1, 2]:
            if np.any(block_types == block_type):
                fit_params[f'block{block_type}'] = plot_group(
                    isi_values, choices, block_types, filter_outcomes.capitalize(),
                    colors[filter_outcomes][f'block{block_type}'], block_type=block_type
                )
    
    # Check for ISI divider consistency and use first session's divider
    isi_divider = None
    for lick_properties in lick_properties_list:
        if 'ISI_devider' in lick_properties:
            if isi_divider is None:
                isi_divider = lick_properties['ISI_devider']
            elif isi_divider != lick_properties['ISI_devider']:
                print(f"Warning: Inconsistent ISI_devider values across sessions. Using {isi_divider}.")
    
    if isi_divider is not None:
        ax.axvline(x=isi_divider, color='r', linestyle='--', alpha=0.3)
    
    # Customize plot
    ax.set_xlabel('Inter-Stimulus Interval (s)', fontsize=12)
    ax.set_ylabel('Probability of Right Choice', fontsize=12)
    ax.set_title(f'Pooled Psychometric Curve - {title_suffix}', fontsize=14, fontweight='bold')
    
    # Set x-axis limits
    if is_discrete:
        min_isi = min(unique_isi) - 0.1 * abs(min(unique_isi))
        max_isi = max(unique_isi) + 0.1 * abs(max(unique_isi))
    else:
        min_isi = np.floor(isi_values.min() / bin_width) * bin_width
        max_isi = np.ceil(isi_values.max() / bin_width) * bin_width
    ax.set_xlim(min_isi, max_isi)
    ax.set_ylim(-0.05, 1.05)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at y=0.5
    ax.axhline(y=0.5, color='k', linestyle='-', alpha=0.2)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    return fig, ax, fit_params

# reaction time error bars for pooled sessions

def plot_pooled_isi_reaction_time(lick_properties_list, filter_outcomes='all', ax=None, opto_split=False):
    """
    Plot mean lick reaction time ± SEM for short and long ISI trials, pooled from multiple sessions.
    
    Parameters:
    -----------
    lick_properties_list : list of dict
        List of dictionaries, each containing lick properties from a session with structure:
        {'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
         'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': []}
    filter_outcomes : str, optional
        'all' for both rewarded and punished trials, 'rewarded' for rewarded trials only, 'punished' for punished trials only
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    opto_split : bool, optional
        Whether to split data by opto_tag (control vs. opto trials)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    stats : dict or None
        Dictionary containing mean and SEM for each ISI group
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    
    # Define colors based on filter_outcomes
    colors = {
        'all': '#808080',      # Gray for all trials
        'rewarded': '#63f250', # Green for rewarded trials
        'punished': '#e74c3c', # Red for punished trials
        'opto': '#005eff',     # Blue for opto trials
        'control': '#808080'   # Gray for control trials when opto_split
    }
    
    # Extract data efficiently from all sessions
    def extract_lick_data(keys, lick_properties_list):
        isi = []
        reaction_times = []
        opto = []
        for lick_properties in lick_properties_list:
            for key in keys:
                if key not in lick_properties:
                    continue
                # Ensure all required fields exist and are aligned
                trial_isi = np.array(lick_properties[key]['Trial_ISI'])
                trial_rt = np.array(lick_properties[key]['Lick_reaction_time'])
                trial_opto = np.array(lick_properties[key]['opto_tag'])
                
                # Validate lengths
                min_length = min(len(trial_isi), len(trial_rt), len(trial_opto))
                if min_length == 0:
                    continue
                
                # Truncate to shortest length and filter NaNs
                valid_mask = (
                    ~np.isnan(trial_isi[:min_length]) & 
                    ~np.isnan(trial_rt[:min_length]) & 
                    ~np.isnan(trial_opto[:min_length])
                )
                
                isi.extend(trial_isi[:min_length][valid_mask])
                reaction_times.extend(trial_rt[:min_length][valid_mask])
                opto.extend(trial_opto[:min_length][valid_mask])
        
        return np.array(isi), np.array(reaction_times), np.array(opto)
    
    # Get ISI, reaction times, and opto tags for left and right choices
    if filter_outcomes == 'all':
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'rewarded':
        left_keys = ['short_ISI_reward_left_correct_lick']
        right_keys = ['long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'punished':
        left_keys = ['long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick']
    
    left_isi, left_rt, left_opto = extract_lick_data(left_keys, lick_properties_list)
    right_isi, right_rt, right_opto = extract_lick_data(right_keys, lick_properties_list)
    
    # Combine data
    isi_values = np.concatenate([left_isi, right_isi])
    reaction_times = np.concatenate([left_rt, right_rt])
    opto_tags = np.concatenate([left_opto, right_opto])
    
    # Verify array lengths
    if not (len(isi_values) == len(reaction_times) == len(opto_tags)):
        raise ValueError(f"Array length mismatch: ISI={len(isi_values)}, "
                        f"Reaction Times={len(reaction_times)}, Opto Tags={len(opto_tags)}")
    
    if len(isi_values) == 0:
        print("No valid data to plot after pooling sessions.")
        return fig, ax, None
    
    # Determine ISI divider
    isi_divider = None
    for lick_properties in lick_properties_list:
        if 'ISI_devider' in lick_properties:
            if isi_divider is None:
                isi_divider = lick_properties['ISI_devider']
            elif isi_divider != lick_properties['ISI_devider']:
                print(f"Warning: Inconsistent ISI_devider values across sessions. Using {isi_divider}.")
    
    # Default to median ISI if no divider is found
    if isi_divider is None:
        isi_divider = np.median(isi_values)
        print(f"No ISI_devider found. Using median ISI: {isi_divider:.2f}s")
    
    # Process data
    def plot_group(isi, rt, label, color, opto_value=None):
        if opto_value is not None:
            if len(opto_tags) != len(isi):
                print(f"Warning: Opto tags length ({len(opto_tags)}) does not match ISI length ({len(isi)})")
                return None
            mask = opto_tags == opto_value
            isi, rt = isi[mask], rt[mask]
        
        if len(isi) == 0:
            return None
        
        # Split into short and long ISI
        short_mask = isi <= isi_divider
        long_mask = isi > isi_divider
        
        # Calculate mean and SEM
        short_rt = rt[short_mask]
        long_rt = rt[long_mask]
        
        stats = {
            'short': {'mean': np.nan, 'sem': np.nan, 'count': 0},
            'long': {'mean': np.nan, 'sem': np.nan, 'count': 0}
        }
        
        if len(short_rt) > 0:
            stats['short']['mean'] = np.mean(short_rt)
            stats['short']['sem'] = np.std(short_rt) / np.sqrt(len(short_rt))
            stats['short']['count'] = len(short_rt)
        
        if len(long_rt) > 0:
            stats['long']['mean'] = np.mean(long_rt)
            stats['long']['sem'] = np.std(long_rt) / np.sqrt(len(long_rt))
            stats['long']['count'] = len(long_rt)
        
        # Plot error bars
        if stats['short']['count'] > 0 or stats['long']['count'] > 0:
            x = [0, 1]
            means = [stats['short']['mean'], stats['long']['mean']]
            sems = [stats['short']['sem'], stats['long']['sem']]
            valid_mask = ~np.isnan(means)
            
            ax.errorbar(np.array(x)[valid_mask], np.array(means)[valid_mask], 
                       yerr=np.array(sems)[valid_mask], fmt='o', color=color,
                       label=label, capsize=5, alpha=0.7)
        
        return stats
    
    # Plot data
    stats = None
    title_suffix = filter_outcomes.capitalize() + ' Trials (Pooled Sessions)'
    if opto_split:
        # Plot control trials
        stats = plot_group(isi_values, reaction_times, 'Control', colors['control'],
                          opto_value=0)
        
        # Plot opto trials if they exist
        if np.any(opto_tags == 1):
            opto_stats = plot_group(isi_values, reaction_times, 'Opto', colors['opto'],
                                   opto_value=1)
            stats = {'control': stats, 'opto': opto_stats}
    else:
        stats = plot_group(isi_values, reaction_times, filter_outcomes.capitalize(),
                          colors[filter_outcomes])
    
    # Customize plot
    ax.set_xlabel('ISI Category', fontsize=12)
    ax.set_ylabel('Lick Reaction Time (s)', fontsize=12)
    ax.set_title(f'Pooled Reaction Time vs ISI - {title_suffix}', fontsize=14, fontweight='bold')
    
    # Set x-axis
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['short_ISI', 'long_ISI'])
    
    # Set y-axis limits with padding
    # valid_rt = reaction_times[np.isfinite(reaction_times)]
    # if len(valid_rt) > 0:
    #     rt_min = max(0, valid_rt.min() - 0.1)
    #     rt_max = valid_rt.max() + 0.1
    #     ax.set_ylim(rt_min, rt_max)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Add task explanation
    # ax.text(0.02, 0.98, 'Reaction time from stimulus onset to lick',
    #         transform=ax.transAxes, fontsize=7, va='top',
    #         bbox=dict(facecolor='white', alpha=0.7))
    
    # Add ISI divider annotation
    # ax.text(0.5, 0.02, f'ISI Divider: {isi_divider:.2f}s',
    #         transform=ax.transAxes, fontsize=7, ha='center', va='bottom',
    #         bbox=dict(facecolor='white', alpha=0.7))
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    return fig, ax, stats

# Fuction for reaction time curve for pooled sessions
def plot_pooled_reaction_time_curve(lick_properties_list, filter_outcomes='all', ax=None, bin_width=0.05, fit_quadratic=True, opto_split=False):
    """
    Plot lick reaction time as a function of ISI with SEM error bars and quadratic fit, pooled from multiple sessions.
    Superimposes curves for block_type 1 (Short) and block_type 2 (Long).
    
    Parameters:
    -----------
    lick_properties_list : list of dict
        List of dictionaries, each containing lick properties from a session with structure:
        {'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
         'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': []}
    filter_outcomes : str, optional
        'all' for both rewarded and punished trials, 'rewarded' for rewarded trials only, 'punished' for punished trials only
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    bin_width : float, optional
        Width of ISI bins for calculating reaction time statistics
    fit_quadratic : bool, optional
        Whether to fit and plot a quadratic function to the data
    opto_split : bool, optional
        Whether to split data by opto_tag (control vs. opto trials)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict or None
        Parameters of the quadratic fit, if fit_quadratic=True
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Define colors based on filter_outcomes and block_type
    colors = {
        'all': {'block0': '#999999', 'block1': '#EC882B', 'block2': '#ff00ff'},
        'rewarded': {'block0': '#a1e0a1', 'block1': '#63f250', 'block2': '#00cc00'},
        'punished': {'block0': '#f7b0a9', 'block1': '#e74c3c', 'block2': '#cc0000'},
        'opto': {'block0': '#005eff', 'block1': '#9999ff', 'block2': '#0033cc'},
        'control': {'block0': '#999999', 'block1': "#EC882B", 'block2': '#ff00ff'}
    }
    
    # Extract data efficiently from all sessions
    def extract_lick_data(keys, lick_properties_list):
        isi = []
        reaction_times = []
        opto = []
        block = []
        for lick_properties in lick_properties_list:
            for key in keys:
                if key not in lick_properties:
                    continue
                # Ensure all required fields exist and are aligned
                trial_isi = np.array(lick_properties[key]['Trial_ISI'])
                trial_rt = np.array(lick_properties[key]['Lick_reaction_time'])
                trial_opto = np.array(lick_properties[key]['opto_tag'])
                trial_block = np.array(lick_properties[key]['block_type'])
                
                # Validate lengths
                min_length = min(len(trial_isi), len(trial_rt), len(trial_opto), len(trial_block))
                if min_length == 0:
                    continue
                
                # Truncate to shortest length and filter NaNs
                valid_mask = (
                    ~np.isnan(trial_isi[:min_length]) & 
                    ~np.isnan(trial_rt[:min_length]) & 
                    ~np.isnan(trial_opto[:min_length]) &
                    ~np.isnan(trial_block[:min_length])
                )
                
                isi.extend(trial_isi[:min_length][valid_mask])
                reaction_times.extend(trial_rt[:min_length][valid_mask])
                opto.extend(trial_opto[:min_length][valid_mask])
                block.extend(trial_block[:min_length][valid_mask])
        
        return np.array(isi), np.array(reaction_times), np.array(opto), np.array(block)
    
    # Get ISI, reaction times, opto tags, and block types for left and right choices
    if filter_outcomes == 'all':
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'rewarded':
        left_keys = ['short_ISI_reward_left_correct_lick']
        right_keys = ['long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'punished':
        left_keys = ['long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick']
    
    left_isi, left_rt, left_opto, left_block = extract_lick_data(left_keys, lick_properties_list)
    right_isi, right_rt, right_opto, right_block = extract_lick_data(right_keys, lick_properties_list)
    
    # Combine data
    isi_values = np.concatenate([left_isi, right_isi])
    reaction_times = np.concatenate([left_rt, right_rt])
    opto_tags = np.concatenate([left_opto, right_opto])
    block_types = np.concatenate([left_block, right_block])
    
    # Verify array lengths
    if not (len(isi_values) == len(reaction_times) == len(opto_tags) == len(block_types)):
        raise ValueError(f"Array length mismatch: ISI={len(isi_values)}, "
                        f"Reaction Times={len(reaction_times)}, Opto Tags={len(opto_tags)}, "
                        f"Block Types={len(block_types)}")
    
    if len(isi_values) == 0:
        print("No valid data to plot after pooling sessions.")
        return fig, ax, None
    
    # Create bins
    min_isi = np.floor(isi_values.min() / bin_width) * bin_width
    max_isi = np.ceil(isi_values.max() / bin_width) * bin_width
    bins = np.arange(min_isi, max_isi + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width/2
    
    # Define quadratic function for fitting
    def quadratic_function(x, a, b, c):
        return a * x**2 + b * x + c
    
    # Process data
    def plot_group(isi, rt, block_types, label, color, opto_value=None, block_type=None):
        # Validate array lengths first
        if not (len(isi) == len(rt) == len(block_types) == len(opto_tags)):
            print(f"Warning: Array length mismatch — ISI: {len(isi)}, RT: {len(rt)}, Block: {len(block_types)}, Opto: {len(opto_tags)}")
            return None

        # Build full boolean mask
        mask = np.ones(len(isi), dtype=bool)
        if block_type is not None:
            mask &= (block_types == block_type)
        if opto_value is not None:
            mask &= (opto_tags == opto_value)

        # Apply mask to all relevant arrays
        isi = isi[mask]
        rt = rt[mask]
        
        if len(isi) == 0:
            return None
        
        # Map block_type to label
        block_label = ['Neutral', 'Short', 'Long'][block_type] if block_type in [0, 1, 2] else f"Block {block_type}"
        
        # Calculate statistics
        mean_rt = np.zeros(len(bins) - 1)
        sem = np.zeros(len(bins) - 1)
        counts = np.zeros(len(bins) - 1)
        
        for i in range(len(bins) - 1):
            mask = (isi >= bins[i]) & (isi < bins[i + 1])
            bin_rt = rt[mask]
            if len(bin_rt) > 0:
                mean_rt[i] = np.mean(bin_rt)
                sem[i] = np.std(bin_rt) / np.sqrt(len(bin_rt))
                counts[i] = len(bin_rt)
        
        valid_mask = counts > 0
        valid_centers = bin_centers[valid_mask]
        valid_rt = mean_rt[valid_mask]
        valid_sem = sem[valid_mask]
        
        # Plot with error bars using same color as curve
        ax.errorbar(valid_centers, valid_rt, yerr=valid_sem, fmt='o', color=color,
                   label=f"{label} data ({block_label})", capsize=3, alpha=0.7)
        
        # Fit quadratic curve
        fit_params = None
        if fit_quadratic and len(valid_centers) > 3:
            try:
                popt, _ = curve_fit(quadratic_function, valid_centers, valid_rt,
                                  p0=[0, 0, np.mean(valid_rt)])
                
                x_fit = np.linspace(min_isi, max_isi, 100)
                y_fit = quadratic_function(x_fit, *popt)
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2,
                       label=f"{label} fit ({block_label})")
                
                fit_params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
            except Exception as e:
                print(f"Could not fit quadratic function for {label} ({block_label}): {e}")
        
        return fit_params
    
    # Plot data
    fit_params = {}
    title_suffix = filter_outcomes.capitalize() + ' Trials (Pooled Sessions)'
    if opto_split:
        # Plot control trials for both block types
        fit_params['control'] = {}
        for block_type in [0, 1, 2]:
            if np.any(block_types == block_type):
                fit_params['control'][f'block{block_type}'] = plot_group(
                    isi_values, reaction_times, block_types, 'Control',
                    colors['control'][f'block{block_type}'], opto_value=0, block_type=block_type
                )
        
        # Plot opto trials for both block types if they exist
        if np.any(opto_tags == 1):
            fit_params['opto'] = {}
            for block_type in [0, 1, 2]:
                if np.any(block_types == block_type):
                    fit_params['opto'][f'block{block_type}'] = plot_group(
                        isi_values, reaction_times, block_types, 'Opto',
                        colors['opto'][f'block{block_type}'], opto_value=1, block_type=block_type
                    )
    else:
        # Plot for both block types
        for block_type in [0, 1, 2]:
            if np.any(block_types == block_type):
                fit_params[f'block{block_type}'] = plot_group(
                    isi_values, reaction_times, block_types, filter_outcomes.capitalize(),
                    colors[filter_outcomes][f'block{block_type}'], block_type=block_type
                )
    
    # Check for ISI divider consistency and use first session's divider
    isi_divider = None
    for lick_properties in lick_properties_list:
        if 'ISI_devider' in lick_properties:
            if isi_divider is None:
                isi_divider = lick_properties['ISI_devider']
            elif isi_divider != lick_properties['ISI_devider']:
                print(f"Warning: Inconsistent ISI_devider values across sessions. Using {isi_divider}.")
    
    if isi_divider is not None:
        ax.axvline(x=isi_divider, color='r', linestyle='--', alpha=0.3)
    
    # Customize plot
    ax.set_xlabel('Inter-Stimulus Interval (s)', fontsize=12)
    ax.set_ylabel('Lick Reaction Time (s)', fontsize=12)
    ax.set_title(f'Pooled Reaction Time Curve - {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlim(min_isi, max_isi)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    return fig, ax, fit_params

# Fucntions for Grand Average plots ------------------------------------------------------
def plot_grand_average_psychometric_curve(lick_properties_list, filter_outcomes='all', ax=None, bin_width=0.05, fit_logistic=True, opto_split=False):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    colors = {
        'all': {'block0': '#999999', 'block1': '#EC882B', 'block2': '#ff00ff'},
        'rewarded': {'block0': '#a1e0a1', 'block1': '#63f250', 'block2': '#00cc00'},
        'punished': {'block0': '#f7b0a9', 'block1': '#e74c3c', 'block2': '#cc0000'},
        'opto': {'block0': '#005eff', 'block1': '#9999ff', 'block2': '#0033cc'},
        'control': {'block0': '#999999', 'block1': "#EC882B", 'block2': '#ff00ff'}
    }

    def extract_session_data(keys, lick_properties):
        isi, opto, block = [], [], []
        for key in keys:
            if key not in lick_properties:
                continue
            trial_isi = np.array(lick_properties[key]['Trial_ISI'])
            trial_opto = np.array(lick_properties[key]['opto_tag'])
            trial_block = np.array(lick_properties[key]['block_type'])

            min_length = min(len(trial_isi), len(trial_opto), len(trial_block))
            if min_length == 0:
                continue

            valid_mask = (
                ~np.isnan(trial_isi[:min_length]) &
                ~np.isnan(trial_opto[:min_length]) &
                ~np.isnan(trial_block[:min_length])
            )
            isi.extend(trial_isi[:min_length][valid_mask])
            opto.extend(trial_opto[:min_length][valid_mask])
            block.extend(trial_block[:min_length][valid_mask])
        return np.array(isi), np.array(opto), np.array(block)

    if filter_outcomes == 'all':
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'rewarded':
        left_keys = ['short_ISI_reward_left_correct_lick']
        right_keys = ['long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'punished':
        left_keys = ['long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick']

    all_isi, all_block = [], []
    for lick_properties in lick_properties_list:
        l_isi, _, l_block = extract_session_data(left_keys, lick_properties)
        r_isi, _, r_block = extract_session_data(right_keys, lick_properties)
        all_isi.extend(l_isi)
        all_isi.extend(r_isi)
        all_block.extend(l_block)
        all_block.extend(r_block)

    all_isi = np.array(all_isi)
    all_block = np.array(all_block)
    if len(all_isi) == 0:
        print("No valid ISI data to plot across sessions.")
        return fig, ax, None

    unique_isi = np.unique(all_isi)
    is_discrete = len(unique_isi) <= 2

    if is_discrete:
        isi_centers = np.sort(unique_isi)
    else:
        min_isi = np.floor(all_isi.min() / bin_width) * bin_width
        max_isi = np.ceil(all_isi.max() / bin_width) * bin_width
        bins = np.arange(min_isi, max_isi + bin_width, bin_width)
        bin_centers = bins[:-1] + bin_width / 2

    def logistic_function(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def plot_group(centers, bins, label, color, opto_value=None, block_type=None):
        session_means, session_counts = [], []
        for lick_properties in lick_properties_list:
            l_isi, l_opto, l_block = extract_session_data(left_keys, lick_properties)
            r_isi, r_opto, r_block = extract_session_data(right_keys, lick_properties)
            isi = np.concatenate([l_isi, r_isi])
            opto = np.concatenate([l_opto, r_opto])
            block = np.concatenate([l_block, r_block])
            choices = np.concatenate([np.zeros(len(l_isi)), np.ones(len(r_isi))])

            if not (len(isi) == len(choices) == len(opto) == len(block)):
                print("Skipping session due to array length mismatch")
                continue

            mask = np.ones(len(isi), dtype=bool)
            if block_type is not None:
                mask &= (block == block_type)
            if opto_value is not None:
                mask &= (opto == opto_value)
            isi = isi[mask]
            choices = choices[mask]

            if len(isi) == 0:
                continue

            if is_discrete:
                right_prob = np.zeros(len(isi_centers))
                counts = np.zeros(len(isi_centers))
                for i, val in enumerate(isi_centers):
                    bin_mask = isi == val
                    bin_choices = choices[bin_mask]
                    if len(bin_choices) > 0:
                        right_prob[i] = np.mean(bin_choices)
                        counts[i] = len(bin_choices)
            else:
                right_prob = np.zeros(len(bins) - 1)
                counts = np.zeros(len(bins) - 1)
                for i in range(len(bins) - 1):
                    bin_mask = (isi >= bins[i]) & (isi < bins[i + 1])
                    bin_choices = choices[bin_mask]
                    if len(bin_choices) > 0:
                        right_prob[i] = np.mean(bin_choices)
                        counts[i] = len(bin_choices)

            session_means.append(right_prob)
            session_counts.append(counts)

        if not session_means:
            print(f"No valid data for {label} (Block {block_type})")
            return None

        session_means = np.array(session_means)
        session_counts = np.array(session_counts)
        grand_mean = np.nanmean(session_means, axis=0)
        grand_sem = np.nanstd(session_means, axis=0) / np.sqrt(np.sum(session_counts > 0, axis=0))
        valid_mask = np.sum(session_counts, axis=0) > 0
        valid_centers = isi_centers[valid_mask] if is_discrete else bin_centers[valid_mask]
        valid_mean = grand_mean[valid_mask]
        valid_sem = grand_sem[valid_mask]

        block_label = ['Neutral', 'Short', 'Long'][block_type] if block_type in [0, 1, 2] else f"Block {block_type}"

        ax.errorbar(valid_centers, valid_mean, yerr=valid_sem, fmt='o', color=color,
                    label=f"{label} data ({block_label})", capsize=3, alpha=0.7)

        if is_discrete and len(valid_centers) > 1:
            ax.plot(valid_centers, valid_mean, '-', color=color, linewidth=2,
                    label=f"{label} line ({block_label})", alpha=0.7)

        fit_params = None
        if fit_logistic and not is_discrete and len(valid_centers) > 3:
            try:
                popt, _ = curve_fit(logistic_function, valid_centers, valid_mean,
                                    p0=[1.0, 1.0, np.median(valid_centers)],
                                    bounds=([0.5, -10, min_isi], [1, 10, max_isi]))
                x_fit = np.linspace(min_isi, max_isi, 100)
                y_fit = logistic_function(x_fit, *popt)
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2,
                        label=f"{label} fit ({block_label})")
                ip_value = x_fit[np.argmin(np.abs(y_fit - 0.5))]
                ax.axvline(x=ip_value, color=color, linestyle='--', alpha=0.5)
                # ax.text(ip_value, 0.1, f'{label} IP ({block_label}): {ip_value:.2f}s',
                #         color=color, ha='center', bbox=dict(facecolor='white', alpha=0.7))
                fit_params = {'L': popt[0], 'k': popt[1], 'x0': popt[2]}
            except Exception as e:
                print(f"Fit error for {label} ({block_label}): {e}")

        return fit_params

    fit_params = {}
    title_suffix = filter_outcomes.capitalize() + ' Trials'
    for mode in ['control', 'opto'] if opto_split else ['main']:
        fit_params[mode] = {}
        for block_type in [0, 1, 2]:
            if np.any(all_block == block_type):
                args = (isi_centers, None) if is_discrete else (bin_centers, bins)
                fit_params[mode][f'block{block_type}'] = plot_group(*args,
                    label='Opto' if mode == 'opto' else filter_outcomes.capitalize(),
                    color=colors[mode if mode != 'main' else filter_outcomes][f'block{block_type}'],
                    opto_value=1 if mode == 'opto' else None,
                    block_type=block_type)

    isi_divider = next((lp['ISI_devider'] for lp in lick_properties_list if 'ISI_devider' in lp), None)
    if isi_divider is not None:
        ax.axvline(x=isi_divider, color='r', linestyle='--', alpha=0.3)

    ax.set_xlabel('Inter-Stimulus Interval (s)', fontsize=12)
    ax.set_ylabel('Probability of Right Choice', fontsize=12)
    ax.set_title(f'Grand Average Psychometric Curve - {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlim(min_isi, max_isi)
    ax.set_ylim(-0.05, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0.5, color='k', linestyle='-', alpha=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    return fig, ax, fit_params

# Fucn

def plot_grand_average_isi_reaction_time(lick_properties_list, filter_outcomes='all', ax=None, opto_split=False):
    """
    Plot grand average mean lick reaction time ± SEM for short and long ISI trials across sessions.
    
    Parameters:
    -----------
    lick_properties_list : list of dict
        List of dictionaries, each containing lick properties from a session with structure:
        {'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
         'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': []}
    filter_outcomes : str, optional
        'all' for both rewarded and punished trials, 'rewarded' for rewarded trials only, 'punished' for punished trials only
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    opto_split : bool, optional
        Whether to split data by opto_tag (control vs. opto trials)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    stats : dict or None
        Dictionary containing grand average mean and SEM for each ISI group
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    
    # Define colors based on filter_outcomes
    colors = {
        'all': '#808080',      # Gray for all trials
        'rewarded': '#63f250', # Green for rewarded trials
        'punished': '#e74c3c', # Red for punished trials
        'opto': '#005eff',     # Blue for opto trials
        'control': '#808080'   # Gray for control trials when opto_split
    }
    
    # Extract data for a single session
    def extract_session_data(keys, lick_properties):
        isi = []
        reaction_times = []
        opto = []
        for key in keys:
            if key not in lick_properties:
                continue
            # Ensure all required fields exist and are aligned
            trial_isi = np.array(lick_properties[key]['Trial_ISI'])
            trial_rt = np.array(lick_properties[key]['Lick_reaction_time'])
            trial_opto = np.array(lick_properties[key]['opto_tag'])
            
            # Validate lengths
            min_length = min(len(trial_isi), len(trial_rt), len(trial_opto))
            if min_length == 0:
                continue
            
            # Truncate to shortest length and filter NaNs
            valid_mask = (
                ~np.isnan(trial_isi[:min_length]) & 
                ~np.isnan(trial_rt[:min_length]) & 
                ~np.isnan(trial_opto[:min_length])
            )
            
            isi.extend(trial_isi[:min_length][valid_mask])
            reaction_times.extend(trial_rt[:min_length][valid_mask])
            opto.extend(trial_opto[:min_length][valid_mask])
        
        return np.array(isi), np.array(reaction_times), np.array(opto)
    
    # Get keys based on filter_outcomes
    if filter_outcomes == 'all':
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'rewarded':
        left_keys = ['short_ISI_reward_left_correct_lick']
        right_keys = ['long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'punished':
        left_keys = ['long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick']
    
    # Determine ISI divider
    isi_divider = None
    all_isi = []
    for lick_properties in lick_properties_list:
        if 'ISI_devider' in lick_properties:
            if isi_divider is None:
                isi_divider = lick_properties['ISI_devider']
            elif isi_divider != lick_properties['ISI_devider']:
                print(f"Warning: Inconsistent ISI_devider values across sessions. Using {isi_divider}.")
        # Collect ISI for median fallback
        left_isi, _, _ = extract_session_data(left_keys, lick_properties)
        right_isi, _, _ = extract_session_data(right_keys, lick_properties)
        all_isi.extend(left_isi)
        all_isi.extend(right_isi)
    
    all_isi = np.array(all_isi)
    if len(all_isi) == 0:
        print("No valid ISI data to plot across sessions.")
        return fig, ax, None
    
    # Default to median ISI if no divider is found
    if isi_divider is None:
        isi_divider = np.median(all_isi)
        print(f"No ISI_devider found. Using median ISI: {isi_divider:.2f}s")
    
    # Process data for grand average
    def plot_group(label, color, opto_value=None):
        session_short_means = []
        session_long_means = []
        session_short_counts = []
        session_long_counts = []
        
        for lick_properties in lick_properties_list:
            # Extract session data
            left_isi, left_rt, left_opto = extract_session_data(left_keys, lick_properties)
            right_isi, right_rt, right_opto = extract_session_data(right_keys, lick_properties)
            
            isi_values = np.concatenate([left_isi, right_isi])
            reaction_times = np.concatenate([left_rt, right_rt])
            opto_tags = np.concatenate([left_opto, right_opto])
            
            # Verify array lengths
            if not (len(isi_values) == len(reaction_times) == len(opto_tags)):
                print(f"Skipping session due to array length mismatch: ISI={len(isi_values)}, "
                      f"Reaction Times={len(reaction_times)}, Opto Tags={len(opto_tags)}")
                continue
            
            if len(isi_values) == 0:
                continue
            
            # Apply opto filter if needed
            if opto_value is not None:
                mask = opto_tags == opto_value
                isi_values = isi_values[mask]
                reaction_times = reaction_times[mask]
            
            if len(isi_values) == 0:
                continue
            
            # Split into short and long ISI
            short_mask = isi_values <= isi_divider
            long_mask = isi_values > isi_divider
            
            short_rt = reaction_times[short_mask]
            long_rt = reaction_times[long_mask]
            
            # Calculate mean and count
            short_mean = np.mean(short_rt) if len(short_rt) > 0 else np.nan
            long_mean = np.mean(long_rt) if len(long_rt) > 0 else np.nan
            short_count = len(short_rt)
            long_count = len(long_rt)
            
            session_short_means.append(short_mean)
            session_long_means.append(long_mean)
            session_short_counts.append(short_count)
            session_long_counts.append(long_count)
        
        if not session_short_means and not session_long_means:
            print(f"No valid data for {label} after processing sessions.")
            return None
        
        # Convert to arrays
        session_short_means = np.array(session_short_means)
        session_long_means = np.array(session_long_means)
        session_short_counts = np.array(session_short_counts)
        session_long_counts = np.array(session_long_counts)
        
        # Calculate grand average mean and SEM
        grand_short_mean = np.nanmean(session_short_means)
        grand_long_mean = np.nanmean(session_long_means)
        grand_short_sem = np.nanstd(session_short_means) / np.sqrt(np.sum(session_short_counts > 0))
        grand_long_sem = np.nanstd(session_long_means) / np.sqrt(np.sum(session_long_counts > 0))
        
        # Prepare stats
        stats = {
            'short': {
                'mean': grand_short_mean,
                'sem': grand_short_sem,
                'count': np.sum(session_short_counts)
            },
            'long': {
                'mean': grand_long_mean,
                'sem': grand_long_sem,
                'count': np.sum(session_long_counts)
            }
        }
        
        # Plot error bars
        x = [0, 1]
        means = [grand_short_mean, grand_long_mean]
        sems = [grand_short_sem, grand_long_sem]
        valid_mask = ~np.isnan(means)
        
        if np.any(valid_mask):
            ax.errorbar(np.array(x)[valid_mask], np.array(means)[valid_mask], 
                       yerr=np.array(sems)[valid_mask], fmt='o', color=color,
                       label=label, capsize=5, alpha=0.7)
        
        return stats
    
    # Plot data
    stats = None
    title_suffix = filter_outcomes.capitalize() + ' Trials'
    if opto_split:
        # Plot control trials
        stats = plot_group('Control', colors['control'], opto_value=0)
        
        # Plot opto trials if they exist
        has_opto = any(np.any(np.array(lick_properties[key]['opto_tag']) == 1)
                      for lick_properties in lick_properties_list
                      for key in left_keys + right_keys)
        if has_opto:
            opto_stats = plot_group('Opto', colors['opto'], opto_value=1)
            stats = {'control': stats, 'opto': opto_stats}
    else:
        stats = plot_group(filter_outcomes.capitalize(), colors[filter_outcomes])
    
    # Customize plot
    ax.set_xlabel('ISI Category', fontsize=12)
    ax.set_ylabel('Lick Reaction Time (s)', fontsize=12)
    ax.set_title(f'Grand Average Reaction Time vs ISI - {title_suffix}', fontsize=14, fontweight='bold')
    
    # Set x-axis
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['short_ISI', 'long_ISI'])
    
    # Set y-axis limits with padding
    valid_rt = all_rt = []
    for lick_properties in lick_properties_list:
        left_isi, left_rt, _ = extract_session_data(left_keys, lick_properties)
        right_isi, right_rt, _ = extract_session_data(right_keys, lick_properties)
        all_rt.extend(left_rt)
        all_rt.extend(right_rt)
    
    # all_rt = np.array(all_rt)
    # valid_rt = all_rt[np.isfinite(all_rt)]
    # if len(valid_rt) > 0:
    #     rt_min = max(0, valid_rt.min() - 0.1)
    #     rt_max = valid_rt.max() + 0.1
    #     ax.set_ylim(rt_min, rt_max)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Add task explanation
    # ax.text(0.02, 0.98, 'Reaction time from stimulus onset to lick',
    #         transform=ax.transAxes, fontsize=7, va='top',
    #         bbox=dict(facecolor='white', alpha=0.7))
    
    # Add ISI divider annotation
    # ax.text(0.5, 0.02, f'ISI Divider: {isi_divider:.2f}s',
    #         transform=ax.transAxes, fontsize=7, ha='center', va='bottom',
    #         bbox=dict(facecolor='white', alpha=0.7))
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    return fig, ax, stats


def plot_grand_average_reaction_time_curve(lick_properties_list, filter_outcomes='all', ax=None, bin_width=0.05, fit_quadratic=True, opto_split=False):
    """
    Plot grand average lick reaction time ± SEM as a function of ISI with quadratic fit across sessions.
    Superimposes curves for block_type 1 (Short) and block_type 2 (Long).
    
    Parameters:
    -----------
    lick_properties_list : list of dict
        List of dictionaries, each containing lick properties from a session with structure:
        {'Lick_reaction_time': [], 'Lick_start_time': [], 'Lick_end_time': [], 
         'Lick_duration': [], 'Trial_ISI': [], 'opto_tag': [], 'block_type': []}
    filter_outcomes : str, optional
        'all' for both rewarded and punished trials, 'rewarded' for rewarded trials only, 'punished' for punished trials only
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    bin_width : float, optional
        Width of ISI bins for calculating reaction time statistics
    fit_quadratic : bool, optional
        Whether to fit and plot a quadratic function to the data
    opto_split : bool, optional
        Whether to split data by opto_tag (control vs. opto trials)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict or None
        Parameters of the quadratic fit, if fit_quadratic=True
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Define colors based on filter_outcomes and block_type
    colors = {
        'all': {'block0': '#999999', 'block1': '#EC882B', 'block2': '#ff00ff'},
        'rewarded': {'block0': '#a1e0a1', 'block1': '#63f250', 'block2': '#00cc00'},
        'punished': {'block0': '#f7b0a9', 'block1': '#e74c3c', 'block2': '#cc0000'},
        'opto': {'block0': '#005eff', 'block1': '#9999ff', 'block2': '#0033cc'},
        'control': {'block0': '#999999', 'block1': "#EC882B", 'block2': '#ff00ff'}
    }
    
    # Extract data for a single session
    def extract_session_data(keys, lick_properties):
        isi = []
        reaction_times = []
        opto = []
        block = []
        for key in keys:
            if key not in lick_properties:
                continue
            # Ensure all required fields exist and are aligned
            trial_isi = np.array(lick_properties[key]['Trial_ISI'])
            trial_rt = np.array(lick_properties[key]['Lick_reaction_time'])
            trial_opto = np.array(lick_properties[key]['opto_tag'])
            trial_block = np.array(lick_properties[key]['block_type'])
            
            # Validate lengths
            min_length = min(len(trial_isi), len(trial_rt), len(trial_opto), len(trial_block))
            if min_length == 0:
                continue
            
            # Truncate to shortest length and filter NaNs
            valid_mask = (
                ~np.isnan(trial_isi[:min_length]) & 
                ~np.isnan(trial_rt[:min_length]) & 
                ~np.isnan(trial_opto[:min_length]) &
                ~np.isnan(trial_block[:min_length])
            )
            
            isi.extend(trial_isi[:min_length][valid_mask])
            reaction_times.extend(trial_rt[:min_length][valid_mask])
            opto.extend(trial_opto[:min_length][valid_mask])
            block.extend(trial_block[:min_length][valid_mask])
        
        return np.array(isi), np.array(reaction_times), np.array(opto), np.array(block)
    
    # Get keys based on filter_outcomes
    if filter_outcomes == 'all':
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'rewarded':
        left_keys = ['short_ISI_reward_left_correct_lick']
        right_keys = ['long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'punished':
        left_keys = ['long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick']
    
    # Determine global ISI range
    all_isi = []
    all_rt = []
    all_block = []
    for lick_properties in lick_properties_list:
        left_isi, left_rt, _, left_block = extract_session_data(left_keys, lick_properties)
        right_isi, right_rt, _, right_block = extract_session_data(right_keys, lick_properties)
        all_isi.extend(left_isi)
        all_isi.extend(right_isi)
        all_rt.extend(left_rt)
        all_rt.extend(right_rt)
        all_block.extend(left_block)
        all_block.extend(right_block)
    
    all_isi = np.array(all_isi)
    all_block = np.array(all_block)
    if len(all_isi) == 0:
        print("No valid ISI data to plot across sessions.")
        return fig, ax, None
    
    min_isi = np.floor(all_isi.min() / bin_width) * bin_width
    max_isi = np.ceil(all_isi.max() / bin_width) * bin_width
    bins = np.arange(min_isi, max_isi + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width/2
    
    # Define quadratic function for fitting
    def quadratic_function(x, a, b, c):
        return a * x**2 + b * x + c
    
    # Process data for grand average
    def plot_group(bin_centers, bins, block_types, label, color, opto_value=None, block_type=None):
        session_means = []
        session_counts = []
        
        for lick_properties in lick_properties_list:
            # Extract session data
            left_isi, left_rt, left_opto, left_block = extract_session_data(left_keys, lick_properties)
            right_isi, right_rt, right_opto, right_block = extract_session_data(right_keys, lick_properties)
            
            isi_values = np.concatenate([left_isi, right_isi])
            reaction_times = np.concatenate([left_rt, right_rt])
            opto_tags = np.concatenate([left_opto, right_opto])
            block_types_session = np.concatenate([left_block, right_block])
            
            # Verify array lengths
            if not (len(isi_values) == len(reaction_times) == len(opto_tags) == len(block_types_session)):
                print(f"Skipping session due to array length mismatch: ISI={len(isi_values)}, "
                      f"Reaction Times={len(reaction_times)}, Opto Tags={len(opto_tags)}, "
                      f"Block Types={len(block_types_session)}")
                continue
            
            if len(isi_values) == 0:
                continue
            
            # Apply block_type filter
            if block_type is not None:
                mask = block_types_session == block_type
                isi_values = isi_values[mask]
                reaction_times = reaction_times[mask]
                opto_tags = opto_tags[mask]
            
            # Apply opto filter if needed
            if opto_value is not None:
                mask = opto_tags == opto_value
                isi_values = isi_values[mask]
                reaction_times = reaction_times[mask]
            
            if len(isi_values) == 0:
                continue
            
            # Calculate mean reaction time for each bin
            mean_rt = np.zeros(len(bins) - 1)
            counts = np.zeros(len(bins) - 1)
            
            for i in range(len(bins) - 1):
                bin_mask = (isi_values >= bins[i]) & (isi_values < bins[i + 1])
                bin_rt = reaction_times[bin_mask]
                if len(bin_rt) > 0:
                    mean_rt[i] = np.mean(bin_rt)
                    counts[i] = len(bin_rt)
            
            session_means.append(mean_rt)
            session_counts.append(counts)
        
        if not session_means:
            print(f"No valid data for {label} (Block {block_type}) after processing sessions.")
            return None
        
        # Convert to arrays
        session_means = np.array(session_means)
        session_counts = np.array(session_counts)
        
        # Calculate grand average mean and SEM
        grand_mean = np.nanmean(session_means, axis=0)
        grand_sem = np.nanstd(session_means, axis=0) / np.sqrt(np.sum(session_counts > 0, axis=0))
        
        # Filter valid bins
        valid_mask = np.sum(session_counts, axis=0) > 0
        valid_centers = bin_centers[valid_mask]
        valid_mean = grand_mean[valid_mask]
        valid_sem = grand_sem[valid_mask]
        
        # Map block_type to label
        block_label = ['Neutral', 'Short', 'Long'][block_type] if block_type in [0, 1, 2] else f"Block {block_type}"
        
        # Plot with error bars
        ax.errorbar(valid_centers, valid_mean, yerr=valid_sem, fmt='o', color=color,
                   label=f"{label} data ({block_label})", capsize=3, alpha=0.7)
        
        # Fit quadratic curve
        fit_params = None
        if fit_quadratic and len(valid_centers) > 3:
            try:
                popt, _ = curve_fit(quadratic_function, valid_centers, valid_mean,
                                  p0=[0, 0, np.mean(valid_mean)])
                
                x_fit = np.linspace(min_isi, max_isi, 100)
                y_fit = quadratic_function(x_fit, *popt)
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2,
                       label=f"{label} fit ({block_label})")
                
                fit_params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
            except Exception as e:
                print(f"Could not fit quadratic function for {label} ({block_label}): {e}")
        
        return fit_params
    
    # Plot data
    fit_params = {}
    title_suffix = filter_outcomes.capitalize() + ' Trials'
    if opto_split:
        # Plot control trials for both block types
        fit_params['control'] = {}
        for block_type in [0, 1, 2]:
            if np.any(all_block == block_type):
                fit_params['control'][f'block{block_type}'] = plot_group(
                    bin_centers, bins, all_block, 'Control',
                    colors['control'][f'block{block_type}'], opto_value=0, block_type=block_type
                )
        
        # Plot opto trials for both block types if they exist
        has_opto = any(np.any(np.array(lick_properties[key]['opto_tag']) == 1)
                      for lick_properties in lick_properties_list
                      for key in left_keys + right_keys)
        if has_opto:
            fit_params['opto'] = {}
            for block_type in [0, 1, 2]:
                if np.any(all_block == block_type):
                    fit_params['opto'][f'block{block_type}'] = plot_group(
                        bin_centers, bins, all_block, 'Opto',
                        colors['opto'][f'block{block_type}'], opto_value=1, block_type=block_type
                    )
    else:
        # Plot for both block types
        for block_type in [0, 1, 2]:
            if np.any(all_block == block_type):
                fit_params[f'block{block_type}'] = plot_group(
                    bin_centers, bins, all_block, filter_outcomes.capitalize(),
                    colors[filter_outcomes][f'block{block_type}'], block_type=block_type
                )
    
    # Check for ISI divider consistency and use first session's divider
    isi_divider = None
    for lick_properties in lick_properties_list:
        if 'ISI_devider' in lick_properties:
            if isi_divider is None:
                isi_divider = lick_properties['ISI_devider']
            elif isi_divider != lick_properties['ISI_devider']:
                print(f"Warning: Inconsistent ISI_devider values across sessions. Using {isi_divider}.")
    
    if isi_divider is not None:
        ax.axvline(x=isi_divider, color='r', linestyle='--', alpha=0.3)
    
    # Customize plot
    ax.set_xlabel('Inter-Stimulus Interval (s)', fontsize=12)
    ax.set_ylabel('Lick Reaction Time (s)', fontsize=12)
    ax.set_title(f'Grand Average Reaction Time Curve - {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlim(min_isi, max_isi)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    return fig, ax, fit_params