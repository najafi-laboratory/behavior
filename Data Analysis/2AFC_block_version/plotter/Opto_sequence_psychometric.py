import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_opto_comparison_curve(lick_properties, block_mode='all', ax=None, bin_width=0.01, fit_logistic=True):
    """
    Plot psychometric curves comparing opto-1, opto, and opto+1 trials, showing probability
    of right port choice as a function of ISI with SEM. Excludes neutral block (block_type 0).
    
    Parameters:
    -----------
    lick_properties : dict
        Dictionary containing lick properties from extract_lick_properties function
    block_mode : str, optional
        'short' for opto trials in short block (block_type 1),
        'long' for opto trials in long block (block_type 2),
        'all' for all opto trials regardless of block type
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    bin_width : float, optional
        Width of ISI bins for calculating choice probability (used when multiple ISIs)
    fit_logistic : bool, optional
        Whether to fit and plot a logistic function to the data (used when multiple ISIs)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict
        Parameters of the logistic fit (for multiple ISIs) or linear fit (for single ISIs), if applicable
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Define colors for opto-1, opto, and opto+1
    colors = {
        'opto-1': '#808080',  # Gray
        'opto': '#005eff',    # Blue
        'opto+1': '#000000'   # Black
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
    
    # Get data for all trials
    all_keys = [
        'short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick',
        'short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick'
    ]
    isi_values, opto_tags, block_types = extract_lick_data(all_keys)
    choices = np.concatenate([
        np.zeros(len(lick_properties['short_ISI_reward_left_correct_lick']['Trial_ISI']) +
                 len(lick_properties['long_ISI_punish_left_incorrect_lick']['Trial_ISI'])),
        np.ones(len(lick_properties['short_ISI_punish_right_incorrect_lick']['Trial_ISI']) +
                len(lick_properties['long_ISI_reward_right_correct_lick']['Trial_ISI']))
    ])
    
    # Identify opto trial indices and their neighbors
    opto_indices = np.where(opto_tags == 1)[0]
    opto_minus_one = opto_indices - 1
    opto_plus_one = opto_indices + 1
    
    # Filter out invalid indices (e.g., opto-1 or opto+1 out of bounds or in neutral block)
    valid_mask = (opto_minus_one >= 0) & (opto_plus_one < len(isi_values))
    valid_mask &= (block_types[opto_minus_one] != 0) & (block_types[opto_indices] != 0) & (block_types[opto_plus_one] != 0)
    
    # Apply block mode filter
    if block_mode == 'short':
        valid_mask &= block_types[opto_indices] == 1
    elif block_mode == 'long':
        valid_mask &= block_types[opto_indices] == 2
    
    opto_indices = opto_indices[valid_mask]
    opto_minus_one = opto_minus_one[valid_mask]
    opto_plus_one = opto_plus_one[valid_mask]
    
    # Combine data for opto-1, opto, and opto+1
    trial_groups = {
        'opto-1': (opto_minus_one, 'Opto-1'),
        'opto': (opto_indices, 'Opto'),
        'opto+1': (opto_plus_one, 'Opto+1')
    }
    
    # Check for single ISI case
    unique_isi = np.unique(isi_values)
    single_isi_case = len(unique_isi) == 2
    
    # Define logistic function for multiple ISI case
    def logistic_function(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    
    # Process data for each trial group
    def plot_group(indices, label, color):
        if len(indices) == 0:
            return None
        
        isi = isi_values[indices]
        group_choices = choices[indices]
        
        # Handle single ISI case
        if single_isi_case:
            unique_isi = np.unique(isi)
            if len(unique_isi) != 2:
                return None
            
            right_prob = np.zeros(2)
            sem = np.zeros(2)
            counts = np.zeros(2)
            
            for i, isi_val in enumerate(unique_isi):
                mask = isi == isi_val
                bin_choices = group_choices[mask]
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
                       label=f"{label} data", capsize=3, alpha=0.7)
            
            # Connect points with a line
            ax.plot(valid_isi, valid_prob, '-', color=color, linewidth=2,
                   label=f"{label} fit")
            
            # Calculate inflection point
            fit_params = None
            if len(valid_isi) == 2:
                x1, x2 = valid_isi
                y1, y2 = valid_prob
                if y2 != y1:
                    m = (y2 - y1) / (x2 - x1)
                    b = y1 - m * x1
                    inflection_point = (0.5 - b) / m
                    if min(x1, x2) <= inflection_point <= max(x1, x2):
                        ax.axvline(x=inflection_point, color=color, linestyle='--', alpha=0.5)
                        ax.text(inflection_point, 0.1, f'{label} IP: {inflection_point:.2f}s',
                               color=color, ha='center', bbox=dict(facecolor='white', alpha=0.7))
                        fit_params = {'x0': inflection_point, 'slope': m, 'intercept': b}
            
            return fit_params
        
        # Multiple ISI case
        else:
            min_isi = np.floor(isi.min() / bin_width) * bin_width
            max_isi = np.ceil(isi.max() / bin_width) * bin_width
            bins = np.arange(min_isi, max_isi + bin_width, bin_width)
            bin_centers = bins[:-1] + bin_width/2
            
            right_prob = np.zeros(len(bins) - 1)
            sem = np.zeros(len(bins) - 1)
            counts = np.zeros(len(bins) - 1)
            
            for i in range(len(bins) - 1):
                mask = (isi >= bins[i]) & (isi < bins[i + 1])
                bin_choices = group_choices[mask]
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
                       label=f"{label} data", capsize=3, alpha=0.7)
            
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
                           label=f"{label} fit")
                    ip_index = np.argmin(np.abs(y_fit - 0.5))
                    inflection_point = x_fit[ip_index]
                    ax.axvline(x=inflection_point, color=color, linestyle='--', alpha=0.5)
                    ax.text(inflection_point, 0.1, f'{label} IP: {inflection_point:.2f}s',
                           color=color, ha='center', bbox=dict(facecolor='white', alpha=0.7))
                    fit_params = {'L': popt[0], 'k': popt[1], 'x0': popt[2]}
                except Exception as e:
                    print(f"Could not fit logistic function for {label}: {e}")
            
            return fit_params
    
    # Plot data for each trial group
    fit_params = {}
    title_suffix = f"Opto Comparison ({block_mode.capitalize()})"
    
    for group, (indices, label) in trial_groups.items():
        fit_params[group] = plot_group(indices, label, colors[group])
    
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

#############################################

def plot_pooled_opto_comparison_curve(lick_properties_list, block_mode='all', ax=None, bin_width=0.05, fit_logistic=True):
    """
    Plot psychometric curves for pooled data from multiple sessions, comparing opto-1, opto, and opto+1 trials.
    Shows probability of right port choice as a function of ISI with SEM. Excludes neutral block (block_type 0).
    
    Parameters:
    -----------
    lick_properties_list : list of dict
        List of dictionaries, each containing lick properties from a session with structure:
        {'short_ISI_reward_left_correct_lick': {...}, 'long_ISI_punish_left_incorrect_lick': {...}, ...}
    block_mode : str, optional
        'short' for opto trials in short block (block_type 1),
        'long' for opto trials in long block (block_type 2),
        'all' for all opto trials regardless of block type
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    bin_width : float, optional
        Width of ISI bins for calculating choice probability (used for continuous ISI data)
    fit_logistic : bool, optional
        Whether to fit and plot a logistic function to the data (only for continuous ISI data)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict
        Parameters of the logistic fit (for continuous ISI) or linear fit (for discrete ISI), if applicable
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Define colors for opto-1, opto, and opto+1
    colors = {
        'opto-1': '#808080',  # Gray
        'opto': '#005eff',    # Blue
        'opto+1': '#000000'   # Black
    }
    
    # Extract data efficiently from all sessions
    def extract_lick_data(keys, lick_properties_list):
        isi = []
        opto = []
        block = []
        choices = []
        session_indices = []
        for session_idx, lick_properties in enumerate(lick_properties_list):
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
                session_indices.extend([session_idx] * np.sum(valid_mask))
                
                # Assign choices based on key
                if 'left' in key:
                    choices.extend([0] * np.sum(valid_mask))
                else:
                    choices.extend([1] * np.sum(valid_mask))
        
        return np.array(isi), np.array(opto), np.array(block), np.array(choices), np.array(session_indices)
    
    # Get data for all trials
    all_keys = [
        'short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick',
        'short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick'
    ]
    isi_values, opto_tags, block_types, choices, session_indices = extract_lick_data(all_keys, lick_properties_list)
    
    if len(isi_values) == 0:
        print("No valid data to plot after pooling sessions.")
        return fig, ax, {}
    
    # Identify opto trial indices and their neighbors
    opto_indices = np.where(opto_tags == 1)[0]
    opto_minus_one = opto_indices - 1
    opto_plus_one = opto_indices + 1
    
    # Filter out invalid indices (out of bounds, neutral block, or across sessions)
    valid_mask = (opto_minus_one >= 0) & (opto_plus_one < len(isi_values))
    valid_mask &= (block_types[opto_minus_one] != 0) & (block_types[opto_indices] != 0) & (block_types[opto_plus_one] != 0)
    valid_mask &= (session_indices[opto_minus_one] == session_indices[opto_indices]) & (session_indices[opto_indices] == session_indices[opto_plus_one])
    
    # Apply block mode filter
    if block_mode == 'short':
        valid_mask &= (block_types[opto_indices] == 1)
    elif block_mode == 'long':
        valid_mask &= (block_types[opto_indices] == 2)
    
    opto_indices = opto_indices[valid_mask]
    opto_minus_one = opto_minus_one[valid_mask]
    opto_plus_one = opto_plus_one[valid_mask]
    
    if len(opto_indices) == 0:
        print("No valid opto trials after filtering.")
        return fig, ax, {}
    
    # Combine data for opto-1, opto, and opto+1
    trial_groups = {
        'opto-1': (opto_minus_one, 'Opto-1'),
        'opto': (opto_indices, 'Opto'),
        'opto+1': (opto_plus_one, 'Opto+1')
    }
    
    # Check for discrete ISI case
    unique_isi = np.unique(isi_values)
    is_discrete = len(unique_isi) <= 2
    
    # Define logistic function
    def logistic_function(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    
    # Process data for each trial group
    def plot_group(indices, label, color):
        if len(indices) == 0:
            return None
        
        isi = isi_values[indices]
        group_choices = choices[indices]
        
        if is_discrete:
            # Handle discrete ISI case
            isi_centers = np.sort(np.unique(isi))
            right_prob = np.zeros(len(isi_centers))
            sem = np.zeros(len(isi_centers))
            counts = np.zeros(len(isi_centers))
            
            for i, isi_val in enumerate(isi_centers):
                mask = isi == isi_val
                bin_choices = group_choices[mask]
                if len(bin_choices) > 0:
                    right_prob[i] = np.mean(bin_choices)
                    sem[i] = np.std(bin_choices) / np.sqrt(len(bin_choices))
                    counts[i] = len(bin_choices)
            
            valid_mask = counts > 0
            valid_centers = isi_centers[valid_mask]
            valid_prob = right_prob[valid_mask]
            valid_sem = sem[valid_mask]
            
            # Plot points with error bars
            ax.errorbar(valid_centers, valid_prob, yerr=valid_sem, fmt='o', color=color,
                       label=f"{label} data", capsize=3, alpha=0.7)
            
            # Connect points with a line
            if len(valid_centers) > 1:
                ax.plot(valid_centers, valid_prob, '-', color=color, linewidth=2,
                       label=f"{label} fit")
            
            # Calculate inflection point for linear fit
            fit_params = None
            if len(valid_centers) == 2:
                x1, x2 = valid_centers
                y1, y2 = valid_prob
                if y2 != y1:
                    m = (y2 - y1) / (x2 - x1)
                    b = y1 - m * x1
                    inflection_point = (0.5 - b) / m
                    if min(x1, x2) <= inflection_point <= max(x1, x2):
                        ax.axvline(x=inflection_point, color=color, linestyle='--', alpha=0.5)
                        ax.text(inflection_point, 0.1, f'{label} IP: {inflection_point:.2f}s',
                               color=color, ha='center', bbox=dict(facecolor='white', alpha=0.7))
                        fit_params = {'x0': inflection_point, 'slope': m, 'intercept': b}
            
            return fit_params
        
        else:
            # Handle continuous ISI case
            min_isi = np.floor(isi.min() / bin_width) * bin_width
            max_isi = np.ceil(isi.max() / bin_width) * bin_width
            bins = np.arange(min_isi, max_isi + bin_width, bin_width)
            bin_centers = bins[:-1] + bin_width/2
            
            right_prob = np.zeros(len(bins) - 1)
            sem = np.zeros(len(bins) - 1)
            counts = np.zeros(len(bins) - 1)
            
            for i in range(len(bins) - 1):
                mask = (isi >= bins[i]) & (isi < bins[i + 1])
                bin_choices = group_choices[mask]
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
                       label=f"{label} data", capsize=3, alpha=0.7)
            
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
                           label=f"{label} fit")
                    ip_index = np.argmin(np.abs(y_fit - 0.5))
                    inflection_point = x_fit[ip_index]
                    ax.axvline(x=inflection_point, color=color, linestyle='--', alpha=0.5)
                    ax.text(inflection_point, 0.1, f'{label} IP: {inflection_point:.2f}s',
                           color=color, ha='center', bbox=dict(facecolor='white', alpha=0.7))
                    fit_params = {'L': popt[0], 'k': popt[1], 'x0': popt[2]}
                except Exception as e:
                    print(f"Could not fit logistic function for {label}: {e}")
            
            return fit_params
    
    # Plot data for each trial group
    fit_params = {}
    title_suffix = f"Pooled Opto Comparison ({block_mode.capitalize()})"
    
    for group, (indices, label) in trial_groups.items():
        fit_params[group] = plot_group(indices, label, colors[group])
    
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

def plot_grand_average_opto_comparison_curve(lick_properties_list, block_mode='all', ax=None, bin_width=0.05, fit_logistic=True):
    """
    Plot grand average psychometric curves across sessions, comparing opto-1, opto, and opto+1 trials.
    Shows mean probability of right port choice as a function of ISI with SEM across sessions.
    Excludes neutral block (block_type 0).
    
    Parameters:
    -----------
    lick_properties_list : list of dict
        List of dictionaries, each containing lick properties from a session with structure:
        {'short_ISI_reward_left_correct_lick': {...}, 'long_ISI_punish_left_incorrect_lick': {...}, ...}
    block_mode : str, optional
        'short' for opto trials in short block (block_type 1),
        'long' for opto trials in long block (block_type 2),
        'all' for all opto trials regardless of block type
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    bin_width : float, optional
        Width of ISI bins for calculating choice probability (used for continuous ISI data)
    fit_logistic : bool, optional
        Whether to fit and plot a logistic function to the data (only for continuous ISI data)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict
        Parameters of the logistic fit (for continuous ISI) or linear fit (for discrete ISI), if applicable
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Define colors for opto-1, opto, and opto+1
    colors = {
        'opto-1': '#808080',  # Gray
        'opto': '#005eff',    # Blue
        'opto+1': '#000000'   # Black
    }
    
    # Extract data from a single session
    def extract_session_data(keys, lick_properties):
        isi, opto, block, choices = [], [], [], []
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
            choices.extend([0 if 'left' in key else 1] * np.sum(valid_mask))
        return np.array(isi), np.array(opto), np.array(block), np.array(choices)
    
    # Get all trial data across sessions
    all_keys = [
        'short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick',
        'short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick'
    ]
    
    # Collect all ISI values to determine discrete/continuous case
    all_isi = []
    for lick_properties in lick_properties_list:
        isi, _, _, _ = extract_session_data(all_keys, lick_properties)
        all_isi.extend(isi)
    
    all_isi = np.array(all_isi)
    if len(all_isi) == 0:
        print("No valid ISI data to plot across sessions.")
        return fig, ax, {}
    
    unique_isi = np.unique(all_isi)
    is_discrete = len(unique_isi) <= 2
    
    # Set up bins or centers
    if is_discrete:
        isi_centers = np.sort(unique_isi)
    else:
        min_isi = np.floor(all_isi.min() / bin_width) * bin_width
        max_isi = np.ceil(all_isi.max() / bin_width) * bin_width
        bins = np.arange(min_isi, max_isi + bin_width, bin_width)
        bin_centers = bins[:-1] + bin_width / 2
    
    # Define logistic function
    def logistic_function(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    
    # Process data for each trial group
    def plot_group(centers, bins, group_indices, label, color):
        session_means, session_counts = [], []
        for lick_properties in lick_properties_list:
            isi, opto_tags, block_types, choices = extract_session_data(all_keys, lick_properties)
            if len(isi) == 0:
                continue
            
            # Identify opto trial indices and neighbors
            opto_indices = np.where(opto_tags == 1)[0]
            opto_minus_one = opto_indices - 1
            opto_plus_one = opto_indices + 1
            
            # Filter valid indices
            valid_mask = (opto_minus_one >= 0) & (opto_plus_one < len(isi))
            valid_mask &= (block_types[opto_minus_one] != 0) & (block_types[opto_indices] != 0) & (block_types[opto_plus_one] != 0)
            if block_mode == 'short':
                valid_mask &= (block_types[opto_indices] == 1)
            elif block_mode == 'long':
                valid_mask &= (block_types[opto_indices] == 2)
            
            opto_indices = opto_indices[valid_mask]
            if group_indices == 'opto-1':
                indices = opto_minus_one[valid_mask]
            elif group_indices == 'opto':
                indices = opto_indices
            else:  # opto+1
                indices = opto_plus_one[valid_mask]
            
            if len(indices) == 0:
                continue
            
            group_isi = isi[indices]
            group_choices = choices[indices]
            
            # Calculate probabilities
            if is_discrete:
                right_prob = np.zeros(len(isi_centers))
                counts = np.zeros(len(isi_centers))
                for i, val in enumerate(isi_centers):
                    bin_mask = group_isi == val
                    bin_choices = group_choices[bin_mask]
                    if len(bin_choices) > 0:
                        right_prob[i] = np.mean(bin_choices)
                        counts[i] = len(bin_choices)
            else:
                right_prob = np.zeros(len(bins) - 1)
                counts = np.zeros(len(bins) - 1)
                for i in range(len(bins) - 1):
                    bin_mask = (group_isi >= bins[i]) & (group_isi < bins[i + 1])
                    bin_choices = group_choices[bin_mask]
                    if len(bin_choices) > 0:
                        right_prob[i] = np.mean(bin_choices)
                        counts[i] = len(bin_choices)
            
            session_means.append(right_prob)
            session_counts.append(counts)
        
        if not session_means:
            print(f"No valid data for {label}")
            return None
        
        # Compute grand average
        session_means = np.array(session_means)
        session_counts = np.array(session_counts)
        grand_mean = np.nanmean(session_means, axis=0)
        grand_sem = np.nanstd(session_means, axis=0) / np.sqrt(np.sum(session_counts > 0, axis=0))
        valid_mask = np.sum(session_counts, axis=0) > 0
        valid_centers = isi_centers[valid_mask] if is_discrete else bin_centers[valid_mask]
        valid_mean = grand_mean[valid_mask]
        valid_sem = grand_sem[valid_mask]
        
        # Plot data
        ax.errorbar(valid_centers, valid_mean, yerr=valid_sem, fmt='o', color=color,
                    label=f"{label} data", capsize=3, alpha=0.7)
        
        # Plot line for discrete case
        if is_discrete and len(valid_centers) > 1:
            ax.plot(valid_centers, valid_mean, '-', color=color, linewidth=2,
                    label=f"{label} line")
        
        # Fit logistic curve for continuous case
        fit_params = None
        if fit_logistic and not is_discrete and len(valid_centers) > 3:
            try:
                popt, _ = curve_fit(logistic_function, valid_centers, valid_mean,
                                    p0=[1.0, 1.0, np.median(valid_centers)],
                                    bounds=([0.5, -10, min_isi], [1, 10, max_isi]))
                x_fit = np.linspace(min_isi, max_isi, 100)
                y_fit = logistic_function(x_fit, *popt)
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2,
                        label=f"{label} fit")
                ip_value = x_fit[np.argmin(np.abs(y_fit - 0.5))]
                ax.axvline(x=ip_value, color=color, linestyle='--', alpha=0.5)
                ax.text(ip_value, 0.1, f'{label} IP: {ip_value:.2f}s',
                        color=color, ha='center', bbox=dict(facecolor='white', alpha=0.7))
                fit_params = {'L': popt[0], 'k': popt[1], 'x0': popt[2]}
            except Exception as e:
                print(f"Fit error for {label}: {e}")
        
        # Linear fit for discrete case
        elif is_discrete and len(valid_centers) == 2:
            x1, x2 = valid_centers
            y1, y2 = valid_mean
            if y2 != y1:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                inflection_point = (0.5 - b) / m
                if min(x1, x2) <= inflection_point <= max(x1, x2):
                    ax.axvline(x=inflection_point, color=color, linestyle='--', alpha=0.5)
                    ax.text(inflection_point, 0.1, f'{label} IP: {inflection_point:.2f}s',
                            color=color, ha='center', bbox=dict(facecolor='white', alpha=0.7))
                    fit_params = {'x0': inflection_point, 'slope': m, 'intercept': b}
        
        return fit_params
    
    # Plot data for each trial group
    fit_params = {}
    trial_groups = {
        'opto-1': ('opto-1', 'Opto-1'),
        'opto': ('opto', 'Opto'),
        'opto+1': ('opto+1', 'Opto+1')
    }
    title_suffix = f"Grand Average Opto Comparison ({block_mode.capitalize()})"
    
    for group, (indices, label) in trial_groups.items():
        args = (isi_centers, None) if is_discrete else (bin_centers, bins)
        fit_params[group] = plot_group(*args, indices, label, colors[group])
    
    # Add ISI divider
    isi_divider = next((lp['ISI_devider'] for lp in lick_properties_list if 'ISI_devider' in lp), None)
    if isi_divider is not None:
        ax.axvline(x=isi_divider, color='r', linestyle='--', alpha=0.3)
    
    # Customize plot
    ax.set_xlabel('Inter-Stimulus Interval (s)', fontsize=12)
    ax.set_ylabel('Probability of Right Choice', fontsize=12)
    ax.set_title(f'Grand Average Psychometric Curve - {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlim(min_isi - 0.1 * abs(min_isi), max_isi + 0.1 * abs(max_isi) if is_discrete else max_isi)
    ax.set_ylim(-0.05, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0.5, color='k', linestyle='-', alpha=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    return fig, ax, fit_params

def plot_opto_comparison_reaction_time_curve(lick_properties, block_mode='all', ax=None, bin_width=0.05, fit_quadratic=True):
    """
    Plot lick reaction time curves comparing opto-1, opto, and opto+1 trials as a function of ISI with SEM error bars.
    Excludes neutral block (block_type 0). Supports quadratic fitting for continuous ISI data.
    
    Parameters:
    -----------
    lick_properties : dict
        Dictionary containing lick properties with structure:
        {'short_ISI_reward_left_correct_lick': {...}, 'long_ISI_punish_left_incorrect_lick': {...}, ...}
    block_mode : str, optional
        'short' for opto trials in short block (block_type 1),
        'long' for opto trials in long block (block_type 2),
        'all' for all opto trials regardless of block type
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    bin_width : float, optional
        Width of ISI bins for calculating reaction time statistics
    fit_quadratic : bool, optional
        Whether to fit and plot a quadratic function to the data (only for continuous ISI data)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict
        Parameters of the quadratic fit, if fit_quadratic=True and continuous ISI data
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Define colors for opto-1, opto, and opto+1
    colors = {
        'opto-1': '#808080',  # Gray
        'opto': '#005eff',    # Blue
        'opto+1': '#000000'   # Black
    }
    
    # Extract data efficiently
    def extract_lick_data(keys):
        isi = []
        reaction_times = []
        opto = []
        block = []
        for key in keys:
            if key not in lick_properties:
                continue
            trial_isi = np.array(lick_properties[key]['Trial_ISI'])
            trial_rt = np.array(lick_properties[key]['Lick_reaction_time'])
            trial_opto = np.array(lick_properties[key]['opto_tag'])
            trial_block = np.array(lick_properties[key]['block_type'])
            
            min_length = min(len(trial_isi), len(trial_rt), len(trial_opto), len(trial_block))
            if min_length == 0:
                continue
                
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
    
    # Get data for all trials
    all_keys = [
        'short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick',
        'short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick'
    ]
    isi_values, reaction_times, opto_tags, block_types = extract_lick_data(all_keys)
    
    if len(isi_values) == 0:
        print("No valid data to plot after filtering.")
        return fig, ax, {}
    
    # Identify opto trial indices and their neighbors
    opto_indices = np.where(opto_tags == 1)[0]
    opto_minus_one = opto_indices - 1
    opto_plus_one = opto_indices + 1
    
    # Filter out invalid indices (out of bounds or in neutral block)
    valid_mask = (opto_minus_one >= 0) & (opto_plus_one < len(isi_values))
    valid_mask &= (block_types[opto_minus_one] != 0) & (block_types[opto_indices] != 0) & (block_types[opto_plus_one] != 0)
    
    # Apply block mode filter
    if block_mode == 'short':
        valid_mask &= (block_types[opto_indices] == 1)
    elif block_mode == 'long':
        valid_mask &= (block_types[opto_indices] == 2)
    
    opto_indices = opto_indices[valid_mask]
    opto_minus_one = opto_minus_one[valid_mask]
    opto_plus_one = opto_plus_one[valid_mask]
    
    if len(opto_indices) == 0:
        print("No valid opto trials after filtering.")
        return fig, ax, {}
    
    # Combine data for opto-1, opto, and opto+1
    trial_groups = {
        'opto-1': (opto_minus_one, 'Opto-1'),
        'opto': (opto_indices, 'Opto'),
        'opto+1': (opto_plus_one, 'Opto+1')
    }
    
    # Check for discrete ISI case
    unique_isi = np.unique(isi_values)
    is_discrete = len(unique_isi) <= 2
    
    # Set up bins or centers
    if is_discrete:
        bin_centers = np.sort(unique_isi)
        bins = None
    else:
        min_isi = np.floor(isi_values.min() / bin_width) * bin_width
        max_isi = np.ceil(isi_values.max() / bin_width) * bin_width
        bins = np.arange(min_isi, max_isi + bin_width, bin_width)
        bin_centers = bins[:-1] + bin_width / 2
    
    # Define quadratic function
    def quadratic_function(x, a, b, c):
        return a * x**2 + b * x + c
    
    # Process data for each trial group
    def plot_group(indices, label, color):
        if len(indices) == 0:
            return None
        
        isi = isi_values[indices]
        rt = reaction_times[indices]
        
        if is_discrete:
            centers = bin_centers
            mean_rt = np.zeros(len(centers))
            sem = np.zeros(len(centers))
            counts = np.zeros(len(centers))
            
            for i, val in enumerate(centers):
                mask = isi == val
                bin_rt = rt[mask]
                if len(bin_rt) > 0:
                    mean_rt[i] = np.mean(bin_rt)
                    sem[i] = np.std(bin_rt) / np.sqrt(len(bin_rt))
                    counts[i] = len(bin_rt)
        else:
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
        
        # Plot with error bars
        ax.errorbar(valid_centers, valid_rt, yerr=valid_sem, fmt='o', color=color,
                    label=f"{label} data", capsize=3, alpha=0.7)
        
        # Plot line for discrete case
        if is_discrete and len(valid_centers) > 1:
            ax.plot(valid_centers, valid_rt, '-', color=color, linewidth=2,
                    label=f"{label} line")
        
        # Fit quadratic curve for continuous case
        fit_params = None
        if fit_quadratic and not is_discrete and len(valid_centers) > 3:
            try:
                popt, _ = curve_fit(quadratic_function, valid_centers, valid_rt,
                                    p0=[0, 0, np.mean(valid_rt)])
                x_fit = np.linspace(min_isi, max_isi, 100)
                y_fit = quadratic_function(x_fit, *popt)
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2,
                        label=f"{label} fit")
                fit_params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
            except Exception as e:
                print(f"Could not fit quadratic function for {label}: {e}")
        
        return fit_params
    
    # Plot data for each trial group
    fit_params = {}
    title_suffix = f"Opto Comparison ({block_mode.capitalize()})"
    
    for group, (indices, label) in trial_groups.items():
        fit_params[group] = plot_group(indices, label, colors[group])
    
    # Add ISI divider if available
    if 'ISI_devider' in lick_properties:
        isi_divider = lick_properties['ISI_devider']
        ax.axvline(x=isi_divider, color='r', linestyle='--', alpha=0.3)
    
    # Customize plot
    ax.set_xlabel('Inter-Stimulus Interval (s)', fontsize=12)
    ax.set_ylabel('Lick Reaction Time (s)', fontsize=12)
    ax.set_title(f'Reaction Time Curve - {title_suffix} - ' + lick_properties['session_date'], fontsize=14, fontweight='bold')
    ax.set_xlim(min_isi - 0.1 * abs(min_isi) if is_discrete else min_isi, max_isi + 0.1 * abs(max_isi) if is_discrete else max_isi)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    return fig, ax, fit_params


def plot_pooled_opto_comparison_reaction_time_curve(lick_properties_list, block_mode='all', ax=None, bin_width=0.05, fit_quadratic=True):
    """
    Plot reaction time curves for pooled data from multiple sessions, comparing opto-1, opto, and opto+1 trials.
    Shows mean lick reaction time as a function of ISI with SEM across sessions. Excludes neutral block (block_type 0).
    
    Parameters:
    -----------
    lick_properties_list : list of dict
        List of dictionaries, each containing lick properties from a session with structure:
        {'short_ISI_reward_left_correct_lick': {...}, 'long_ISI_punish_left_incorrect_lick': {...}, ...}
    block_mode : str, optional
        'short' for opto trials in short block (block_type 1),
        'long' for opto trials in long block (block_type 2),
        'all' for all opto trials regardless of block type
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    bin_width : float, optional
        Width of ISI bins for calculating reaction time statistics
    fit_quadratic : bool, optional
        Whether to fit and plot a quadratic function to the data (only for continuous ISI data)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict
        Parameters of the quadratic fit, if fit_quadratic=True and continuous ISI data
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Define colors for opto-1, opto, and opto+1
    colors = {
        'opto-1': '#808080',  # Gray
        'opto': '#005eff',    # Blue
        'opto+1': '#000000'   # Black
    }
    
    # Extract data from all sessions
    def extract_lick_data(keys, lick_properties_list):
        isi = []
        reaction_times = []
        opto = []
        block = []
        session_indices = []
        for session_idx, lick_properties in enumerate(lick_properties_list):
            for key in keys:
                if key not in lick_properties:
                    continue
                trial_isi = np.array(lick_properties[key]['Trial_ISI'])
                trial_rt = np.array(lick_properties[key]['Lick_reaction_time'])
                trial_opto = np.array(lick_properties[key]['opto_tag'])
                trial_block = np.array(lick_properties[key]['block_type'])
                
                min_length = min(len(trial_isi), len(trial_rt), len(trial_opto), len(trial_block))
                if min_length == 0:
                    continue
                
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
                session_indices.extend([session_idx] * np.sum(valid_mask))
        
        return np.array(isi), np.array(reaction_times), np.array(opto), np.array(block), np.array(session_indices)
    
    # Get data for all trials
    all_keys = [
        'short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick',
        'short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick'
    ]
    isi_values, reaction_times, opto_tags, block_types, session_indices = extract_lick_data(all_keys, lick_properties_list)
    
    if len(isi_values) == 0:
        print("No valid data to plot after pooling sessions.")
        return fig, ax, {}
    
    # Identify opto trial indices and their neighbors
    opto_indices = np.where(opto_tags == 1)[0]
    opto_minus_one = opto_indices - 1
    opto_plus_one = opto_indices + 1
    
    # Filter out invalid indices (out of bounds, neutral block, or across sessions)
    valid_mask = (opto_minus_one >= 0) & (opto_plus_one < len(isi_values))
    valid_mask &= (block_types[opto_minus_one] != 0) & (block_types[opto_indices] != 0) & (block_types[opto_plus_one] != 0)
    valid_mask &= (session_indices[opto_minus_one] == session_indices[opto_indices]) & (session_indices[opto_indices] == session_indices[opto_plus_one])
    
    # Apply block mode filter
    if block_mode == 'short':
        valid_mask &= (block_types[opto_indices] == 1)
    elif block_mode == 'long':
        valid_mask &= (block_types[opto_indices] == 2)
    
    opto_indices = opto_indices[valid_mask]
    opto_minus_one = opto_minus_one[valid_mask]
    opto_plus_one = opto_plus_one[valid_mask]
    
    if len(opto_indices) == 0:
        print("No valid opto trials after filtering.")
        return fig, ax, {}
    
    # Combine data for opto-1, opto, and opto+1
    trial_groups = {
        'opto-1': (opto_minus_one, 'Opto-1'),
        'opto': (opto_indices, 'Opto'),
        'opto+1': (opto_plus_one, 'Opto+1')
    }
    
    # Check for discrete ISI case
    unique_isi = np.unique(isi_values)
    is_discrete = len(unique_isi) <= 2
    
    # Set up bins or centers
    if is_discrete:
        bin_centers = np.sort(unique_isi)
        bins = None
    else:
        min_isi = np.floor(isi_values.min() / bin_width) * bin_width
        max_isi = np.ceil(isi_values.max() / bin_width) * bin_width
        bins = np.arange(min_isi, max_isi + bin_width, bin_width)
        bin_centers = bins[:-1] + bin_width / 2
    
    # Define quadratic function
    def quadratic_function(x, a, b, c):
        return a * x**2 + b * x + c
    
    # Process data for each trial group
    def plot_group(indices, label, color):
        if len(indices) == 0:
            return None
        
        isi = isi_values[indices]
        rt = reaction_times[indices]
        
        if is_discrete:
            centers = bin_centers
            mean_rt = np.zeros(len(centers))
            sem = np.zeros(len(centers))
            counts = np.zeros(len(centers))
            
            for i, val in enumerate(centers):
                mask = isi == val
                bin_rt = rt[mask]
                if len(bin_rt) > 0:
                    mean_rt[i] = np.mean(bin_rt)
                    sem[i] = np.std(bin_rt) / np.sqrt(len(bin_rt))
                    counts[i] = len(bin_rt)
        else:
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
        
        # Plot with error bars
        ax.errorbar(valid_centers, valid_rt, yerr=valid_sem, fmt='o', color=color,
                    label=f"{label} data", capsize=3, alpha=0.7)
        
        # Plot line for discrete case
        if is_discrete and len(valid_centers) > 1:
            ax.plot(valid_centers, valid_rt, '-', color=color, linewidth=2,
                    label=f"{label} line")
        
        # Fit quadratic curve for continuous case
        fit_params = None
        if fit_quadratic and not is_discrete and len(valid_centers) > 3:
            try:
                popt, _ = curve_fit(quadratic_function, valid_centers, valid_rt,
                                    p0=[0, 0, np.mean(valid_rt)])
                x_fit = np.linspace(min_isi, max_isi, 100)
                y_fit = quadratic_function(x_fit, *popt)
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2,
                        label=f"{label} fit")
                fit_params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
            except Exception as e:
                print(f"Could not fit quadratic function for {label}: {e}")
        
        return fit_params
    
    # Plot data for each trial group
    fit_params = {}
    title_suffix = f"Pooled Opto Comparison ({block_mode.capitalize()})"
    
    for group, (indices, label) in trial_groups.items():
        fit_params[group] = plot_group(indices, label, colors[group])
    
    # Add ISI divider
    isi_divider = next((lp['ISI_devider'] for lp in lick_properties_list if 'ISI_devider' in lp), None)
    if isi_divider is not None:
        ax.axvline(x=isi_divider, color='r', linestyle='--', alpha=0.3)
    
    # Customize plot
    ax.set_xlabel('Inter-Stimulus Interval (s)', fontsize=12)
    ax.set_ylabel('Lick Reaction Time (s)', fontsize=12)
    ax.set_title(f'Pooled Reaction Time Curve - {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlim(min_isi - 0.1 * abs(min_isi) if is_discrete else min_isi, max_isi + 0.1 * abs(max_isi) if is_discrete else max_isi)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    return fig, ax, fit_params

def plot_grand_average_opto_comparison_reaction_time_curve(lick_properties_list, block_mode='all', ax=None, bin_width=0.05, fit_quadratic=True):
    """
    Plot grand average reaction time curves across sessions, comparing opto-1, opto, and opto+1 trials.
    Shows mean lick reaction time as a function of ISI with SEM across sessions. Excludes neutral block (block_type 0).
    
    Parameters:
    -----------
    lick_properties_list : list of dict
        List of dictionaries, each containing lick properties from a session with structure:
        {'short_ISI_reward_left_correct_lick': {...}, 'long_ISI_punish_left_incorrect_lick': {...}, ...}
    block_mode : str, optional
        'short' for opto trials in short block (block_type 1),
        'long' for opto trials in long block (block_type 2),
        'all' for all opto trials regardless of block type
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes
    bin_width : float, optional
        Width of ISI bins for calculating reaction time statistics
    fit_quadratic : bool, optional
        Whether to fit and plot a quadratic function to the data (only for continuous ISI data)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict
        Parameters of the quadratic fit, if fit_quadratic=True and continuous ISI data
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Define colors for opto-1, opto, and opto+1
    colors = {
        'opto-1': '#808080',  # Gray
        'opto': '#005eff',    # Blue
        'opto+1': '#000000'   # Black
    }
    
    # Extract data from a single session
    def extract_session_data(keys, lick_properties):
        isi, reaction_times, opto, block = [], [], [], []
        for key in keys:
            if key not in lick_properties:
                continue
            trial_isi = np.array(lick_properties[key]['Trial_ISI'])
            trial_rt = np.array(lick_properties[key]['Lick_reaction_time'])
            trial_opto = np.array(lick_properties[key]['opto_tag'])
            trial_block = np.array(lick_properties[key]['block_type'])
            
            min_length = min(len(trial_isi), len(trial_rt), len(trial_opto), len(trial_block))
            if min_length == 0:
                continue
                
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
    
    # Get all trial data across sessions
    all_keys = [
        'short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick',
        'short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick'
    ]

    # Collect all ISI values to determine discrete/continuous case
    all_isi = []
    for lick_properties in lick_properties_list:
        isi, _, _, _ = extract_session_data(all_keys, lick_properties)
        all_isi.extend(isi)
    
    all_isi = np.array(all_isi)
    if len(all_isi) == 0:
        print("No valid ISI data to plot across sessions.")
        return fig, ax, {}
    
    unique_isi = np.unique(all_isi)
    is_discrete = len(unique_isi) <= 2
    
    # Set up bins or centers
    if is_discrete:
        isi_centers = np.sort(unique_isi)
        bins = None
    else:
        min_isi = np.floor(all_isi.min() / bin_width) * bin_width
        max_isi = np.ceil(all_isi.max() / bin_width) * bin_width
        bins = np.arange(min_isi, max_isi + bin_width, bin_width)
        isi_centers = bins[:-1] + bin_width / 2
    
    # Define quadratic function
    def quadratic_function(x, a, b, c):
        return a * x**2 + b * x + c
    
    # Process data for each trial group
    def plot_group(centers, bins, group_indices, label, color):
        session_means, session_counts = [], []
        for lick_properties in lick_properties_list:
            isi, reaction_times, opto_tags, block_types = extract_session_data(all_keys, lick_properties)
            if len(isi) == 0:
                continue
            
            # Identify opto trial indices and neighbors
            opto_indices = np.where(opto_tags == 1)[0]
            opto_minus_one = opto_indices - 1
            opto_plus_one = opto_indices + 1
            
            # Filter valid indices
            valid_mask = (opto_minus_one >= 0) & (opto_plus_one < len(isi))
            valid_mask &= (block_types[opto_minus_one] != 0) & (block_types[opto_indices] != 0) & (block_types[opto_plus_one] != 0)
            if block_mode == 'short':
                valid_mask &= (block_types[opto_indices] == 1)
            elif block_mode == 'long':
                valid_mask &= (block_types[opto_indices] == 2)
            
            opto_indices = opto_indices[valid_mask]
            if group_indices == 'opto-1':
                indices = opto_minus_one[valid_mask]
            elif group_indices == 'opto':
                indices = opto_indices
            else:  # opto+1
                indices = opto_plus_one[valid_mask]
            
            if len(indices) == 0:
                continue
            
            group_isi = isi[indices]
            group_rt = reaction_times[indices]
            
            # Calculate reaction time statistics
            if is_discrete:
                mean_rt = np.zeros(len(centers))
                counts = np.zeros(len(centers))
                for i, val in enumerate(centers):
                    bin_mask = group_isi == val
                    bin_rt = group_rt[bin_mask]
                    if len(bin_rt) > 0:
                        mean_rt[i] = np.mean(bin_rt)
                        counts[i] = len(bin_rt)
            else:
                mean_rt = np.zeros(len(bins) - 1)
                counts = np.zeros(len(bins) - 1)
                for i in range(len(bins) - 1):
                    bin_mask = (group_isi >= bins[i]) & (group_isi < bins[i + 1])
                    bin_rt = group_rt[bin_mask]
                    if len(bin_rt) > 0:
                        mean_rt[i] = np.mean(bin_rt)
                        counts[i] = len(bin_rt)
            
            session_means.append(mean_rt)
            session_counts.append(counts)
        
        if not session_means:
            print(f"No valid data for {label}")
            return None
        
        # Compute grand average
        session_means = np.array(session_means)
        session_counts = np.array(session_counts)
        grand_mean = np.nanmean(session_means, axis=0)
        grand_sem = np.nanstd(session_means, axis=0) / np.sqrt(np.sum(session_counts > 0, axis=0))
        valid_mask = np.sum(session_counts, axis=0) > 0
        valid_centers = centers[valid_mask]
        valid_mean = grand_mean[valid_mask]
        valid_sem = grand_sem[valid_mask]
        
        # Plot data
        ax.errorbar(valid_centers, valid_mean, yerr=valid_sem, fmt='o', color=color,
                    label=f"{label} data", capsize=3, alpha=0.7)
        
        # Plot line for discrete case
        if is_discrete and len(valid_centers) > 1:
            ax.plot(valid_centers, valid_mean, '-', color=color, linewidth=2,
                    label=f"{label} line")
        
        # Fit quadratic curve for continuous case
        fit_params = None
        if fit_quadratic and not is_discrete and len(valid_centers) > 3:
            try:
                popt, _ = curve_fit(quadratic_function, valid_centers, valid_mean,
                                    p0=[0, 0, np.mean(valid_mean)])
                x_fit = np.linspace(min_isi, max_isi, 100)
                y_fit = quadratic_function(x_fit, *popt)
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2,
                        label=f"{label} fit")
                fit_params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
            except Exception as e:
                print(f"Fit error for {label}: {e}")
        
        return fit_params
    
    # Get all trial data across sessions
    all_keys = [
        'short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick',
        'short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick'
    ]
    
    # Plot data for each trial group
    fit_params = {}
    trial_groups = {
        'opto-1': ('opto-1', 'Opto-1'),
        'opto': ('opto', 'Opto'),
        'opto+1': ('opto+1', 'Opto+1')
    }
    title_suffix = f"Grand Average Opto Comparison ({block_mode.capitalize()})"
    
    for group, (indices, label) in trial_groups.items():
        args = (isi_centers, None) if is_discrete else (isi_centers, bins)
        fit_params[group] = plot_group(*args, indices, label, colors[group])
    
    # Add ISI divider
    isi_divider = next((lp['ISI_devider'] for lp in lick_properties_list if 'ISI_devider' in lp), None)
    if isi_divider is not None:
        ax.axvline(x=isi_divider, color='r', linestyle='--', alpha=0.3)
    
    # Customize plot
    ax.set_xlabel('Inter-Stimulus Interval (s)', fontsize=12)
    ax.set_ylabel('Lick Reaction Time (s)', fontsize=12)
    ax.set_title(f'Grand Average Reaction Time Curve - {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlim(min_isi - 0.1 * abs(min_isi) if is_discrete else min_isi, max_isi + 0.1 * abs(max_isi) if is_discrete else max_isi)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    return fig, ax, fit_params