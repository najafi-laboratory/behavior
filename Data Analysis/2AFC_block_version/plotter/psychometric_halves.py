import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_psychometric_neutral_and_halves(lick_properties, filter_outcomes='all', ax=None, bin_width=0.01, fit_logistic=True, opto_split=False, block_selection='neutral'):
    """
    Plot psychometric curve for a single session, either for neutral blocks (block_type 0), first half, or second half of non-neutral trials.
    Shows probability of right port choice as a function of ISI with SEM. Handles both single ISI per condition and multiple ISI values.

    Parameters:
    -----------
    lick_properties : dict
        Dictionary containing lick properties from extract_lick_properties function (assumes 'trial_number' key for halves)
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
    block_selection : str, optional
        'neutral' for block_type 0, 'first_half' for first half of non-neutral trials, 'second_half' for second half

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict or None
        Parameters of the logistic fit (for multiple ISIs) or linear fit (for single ISIs), if applicable
    """
    # Validate block_selection parameter
    valid_blocks = ['neutral', 'first_half', 'second_half']
    if block_selection not in valid_blocks:
        raise ValueError(f"block_selection must be one of {valid_blocks}")

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Define colors
    colors = {
        'all': {'neutral': '#787573', 'first_half': '#1f77b4', 'second_half': '#d62728'},
        'rewarded': {'neutral': "#8bfa7c", 'first_half': '#1f77b4', 'second_half': '#d62728'},
        'punished': {'neutral': "#d77267", 'first_half': '#1f77b4', 'second_half': '#d62728'},
        'opto': {'neutral': "#8c8ce2", 'first_half': '#8c8ce2', 'second_half': '#0133c7'},
        'control': {'neutral': "#787573", 'first_half': '#787573', 'second_half': '#000000'}
    }

    # Extract data efficiently
    def extract_lick_data(keys):
        isi = []
        opto = []
        block = []
        trial_num = []
        for key in keys:
            isi.extend(lick_properties[key]['Trial_ISI'])
            opto.extend(lick_properties[key]['opto_tag'])
            block.extend(lick_properties[key]['block_type'])
            trial_num.extend(lick_properties[key].get('trial_number', [None] * len(lick_properties[key]['Trial_ISI'])))
        return np.array(isi), np.array(opto), np.array(block), np.array(trial_num)

    # Get ISI, opto tags, block types, and trial numbers
    if filter_outcomes == 'all':
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'rewarded':
        left_keys = ['short_ISI_reward_left_correct_lick']
        right_keys = ['long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'punished':
        left_keys = ['long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick']

    left_isi, left_opto, left_block, left_trial = extract_lick_data(left_keys)
    right_isi, right_opto, right_block, right_trial = extract_lick_data(right_keys)

    # Combine data
    isi_values = np.concatenate([left_isi, right_isi])
    choices = np.concatenate([np.zeros(len(left_isi)), np.ones(len(right_isi))])
    opto_tags = np.concatenate([left_opto, right_opto])
    block_types = np.concatenate([left_block, right_block])
    trial_numbers = np.concatenate([left_trial, right_trial])

    # Filter data based on block_selection
    if block_selection == 'neutral':
        mask = block_types == 0
        isi_values = isi_values[mask]
        choices = choices[mask]
        opto_tags = opto_tags[mask]
        if len(isi_values) == 0:
            ax.text(0.5, 0.5, 'No neutral block trials found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Psychometric Curve - Neutral ({filter_outcomes.capitalize()}) - {lick_properties["session_date"]}')
            return fig, ax, {}
    else:  # first_half or second_half
        mask = block_types != 0
        isi_values = isi_values[mask]
        choices = choices[mask]
        opto_tags = opto_tags[mask]
        trial_numbers = trial_numbers[mask]
        if len(isi_values) == 0:
            ax.text(0.5, 0.5, 'No non-neutral trials found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Psychometric Curve - {block_selection.capitalize()} ({filter_outcomes.capitalize()}) - {lick_properties["session_date"]}')
            return fig, ax, {}

        # Sort by trial number for chronological order
        valid_mask = ~np.isnan(trial_numbers)
        if np.sum(valid_mask) == 0:
            sorted_indices = np.arange(len(isi_values))
        else:
            isi_values = isi_values[valid_mask]
            choices = choices[valid_mask]
            opto_tags = opto_tags[valid_mask]
            trial_numbers = trial_numbers[valid_mask]
            sorted_indices = np.argsort(trial_numbers)
            isi_values = isi_values[sorted_indices]
            choices = choices[sorted_indices]
            opto_tags = opto_tags[sorted_indices]

        # Split into halves
        n_trials = len(isi_values)
        mid_point = n_trials // 2
        if block_selection == 'first_half':
            isi_values = isi_values[:mid_point]
            choices = choices[:mid_point]
            opto_tags = opto_tags[:mid_point]
        else:  # second_half
            isi_values = isi_values[mid_point:]
            choices = choices[mid_point:]
            opto_tags = opto_tags[mid_point:]

    # Check for single ISI case
    unique_isi = np.unique(isi_values)
    single_isi_case = len(unique_isi) == 2

    # Define logistic function
    def logistic_function(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    # Plotting function for a group
    def plot_group(isi, choices, opto_tags, label, color, opto_value=None):
        isi = np.array(isi)
        choices = np.array(choices)
        opto_tags = np.array(opto_tags)

        mask = np.ones(len(isi), dtype=bool)
        if opto_value is not None:
            mask &= (opto_tags == opto_value)
        isi, choices = isi[mask], choices[mask]

        if len(isi) == 0:
            return None

        if single_isi_case:
            unique_isi_group = np.unique(isi)
            if len(unique_isi_group) != 2:
                return None

            right_prob = np.zeros(2)
            sem = np.zeros(2)
            counts = np.zeros(2)

            for i, isi_val in enumerate(unique_isi_group):
                mask = isi == isi_val
                bin_choices = choices[mask]
                if len(bin_choices) > 0:
                    right_prob[i] = np.mean(bin_choices)
                    sem[i] = np.std(bin_choices) / np.sqrt(len(bin_choices))
                    counts[i] = len(bin_choices)

            valid_mask = counts > 0
            valid_isi = unique_isi_group[valid_mask]
            valid_prob = right_prob[valid_mask]
            valid_sem = sem[valid_mask]

            ax.errorbar(valid_isi, valid_prob, yerr=valid_sem, fmt='o', color=color,
                       label=f"{label} data", capsize=3, alpha=0.7)
            ax.plot(valid_isi, valid_prob, '-', color=color, linewidth=2,
                   label=f"{label} fit")

            fit_params = None
            if len(valid_isi) == 2:
                x1, x2 = valid_isi
                y1, y2 = valid_prob
                if y2 != y1:
                    m = (y2 - y1) / (x2 - x1)
                    b = y1 - m * x1
                    inflection_point = (0.5 - b) / m
                    if min(x1, x2) <= inflection_point <= max(x1, x2):
                        ax.text(inflection_point, 0.1, f'{label} IP: {inflection_point:.2f}s',
                               color=color, ha='center', bbox=dict(facecolor='white', alpha=0.7))
                        fit_params = {'x0': inflection_point, 'slope': m, 'intercept': b}

            return fit_params

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
                bin_choices = choices[mask]
                if len(bin_choices) > 0:
                    right_prob[i] = np.mean(bin_choices)
                    sem[i] = np.std(bin_choices) / np.sqrt(len(bin_choices))
                    counts[i] = len(bin_choices)

            valid_mask = counts > 0
            valid_centers = bin_centers[valid_mask]
            valid_prob = right_prob[valid_mask]
            valid_sem = sem[valid_mask]

            ax.errorbar(valid_centers, valid_prob, yerr=valid_sem, fmt='o', color=color,
                       label=f"{label} data", capsize=3, alpha=0.7)

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
                    ip_value = x_fit[ip_index]
                    fit_params = {'L': popt[0], 'k': popt[1], 'x0': popt[2]}
                except Exception as e:
                    print(f"Could not fit logistic function for {label}: {e}")

            return fit_params

    # Plot data based on block_selection
    fit_params = {}
    title_suffix = filter_outcomes.capitalize()
    if block_selection == 'neutral':
        title_suffix += ' Trials - Neutral Block'
        if opto_split:
            fit_params['control'] = plot_group(
                isi_values, choices, opto_tags, 'Control',
                colors['control']['neutral'], opto_value=0
            )
            if np.any(opto_tags == 1):
                fit_params['opto'] = plot_group(
                    isi_values, choices, opto_tags, 'Opto',
                    colors['opto']['neutral'], opto_value=1
                )
        else:
            fit_params['neutral'] = plot_group(
                isi_values, choices, opto_tags, 'Neutral',
                colors[filter_outcomes]['neutral']
            )
    elif block_selection == 'first_half':
        title_suffix += ' Trials - First Half (Excl. Neutral)'
        if opto_split:
            fit_params['control'] = plot_group(
                isi_values, choices, opto_tags, 'First Half Control',
                colors['control']['first_half'], opto_value=0
            )
            if np.any(opto_tags == 1):
                fit_params['opto'] = plot_group(
                    isi_values, choices, opto_tags, 'First Half Opto',
                    colors['opto']['first_half'], opto_value=1
                )
        else:
            fit_params['first_half'] = plot_group(
                isi_values, choices, opto_tags, 'First Half',
                colors[filter_outcomes]['first_half']
            )
    else:  # second_half
        title_suffix += ' Trials - Second Half (Excl. Neutral)'
        if opto_split:
            fit_params['control'] = plot_group(
                isi_values, choices, opto_tags, 'Second Half Control',
                colors['control']['second_half'], opto_value=0
            )
            if np.any(opto_tags == 1):
                fit_params['opto'] = plot_group(
                    isi_values, choices, opto_tags, 'Second Half Opto',
                    colors['opto']['second_half'], opto_value=1
                )
        else:
            fit_params['second_half'] = plot_group(
                isi_values, choices, opto_tags, 'Second Half',
                colors[filter_outcomes]['second_half']
            )

    # Customize plot
    ax.set_xlabel('Inter-Stimulus Interval (s)', fontsize=12)
    ax.set_ylabel('Probability of Right Choice', fontsize=12)
    ax.set_title(f'Psychometric Curve - {title_suffix} - {lick_properties["session_date"]}', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0.5, color='k', linestyle='-', alpha=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    return fig, ax, fit_params

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_pooled_psychometric_neutral_and_halves(lick_properties_list, filter_outcomes='all', ax=None, bin_width=0.05, fit_logistic=True, opto_split=False, block_selection='neutral'):
    """
    Plot psychometric curve for pooled data from multiple sessions, either for neutral blocks (block_type 0), first half, or second half of non-neutral trials.
    Shows probability of right port choice as a function of ISI with SEM. Handles both continuous and discrete ISI values.

    Parameters:
    -----------
    lick_properties_list : list of dict
        List of dictionaries, each containing lick properties from a session
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
    block_selection : str, optional
        'neutral' for block_type 0, 'first_half' for first half of non-neutral trials, 'second_half' for second half

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict or None
        Parameters of the logistic fit (for continuous ISIs) or linear fit (for discrete ISIs), if applicable
    """
    # Validate block_selection parameter
    valid_blocks = ['neutral', 'first_half', 'second_half']
    if block_selection not in valid_blocks:
        raise ValueError(f"block_selection must be one of {valid_blocks}")

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Define colors
    colors = {
        'all': {'neutral': '#787573', 'first_half': '#1f77b4', 'second_half': '#d62728'},
        'rewarded': {'neutral': "#8bfa7c", 'first_half': '#1f77b4', 'second_half': '#d62728'},
        'punished': {'neutral': "#d77267", 'first_half': '#1f77b4', 'second_half': '#d62728'},
        'opto': {'neutral': "#8c8ce2", 'first_half': '#8c8ce2', 'second_half': '#0133c7'},
        'control': {'neutral': "#787573", 'first_half': '#787573', 'second_half': '#000000'}
    }

    # Extract data from all sessions
    def extract_lick_data(keys, lick_properties_list):
        isi, opto, block, trial_num = [], [], [], []
        for lick_properties in lick_properties_list:
            for key in keys:
                if key not in lick_properties:
                    continue
                trial_isi = np.array(lick_properties[key]['Trial_ISI'])
                trial_opto = np.array(lick_properties[key]['opto_tag'])
                trial_block = np.array(lick_properties[key]['block_type'])
                # Safely handle trial_number, replacing None or invalid values with np.nan
                trial_num_key = lick_properties[key].get('trial_number', [np.nan] * len(trial_isi))
                # Convert to numeric array, replacing None with np.nan
                trial_num_key = [x if x is not None and not np.isnan(x) else np.nan for x in trial_num_key]
                trial_num_key = np.array(trial_num_key, dtype=float)

                min_length = min(len(trial_isi), len(trial_opto), len(trial_block), len(trial_num_key))
                if min_length == 0:
                    continue

                valid_mask = (
                    ~np.isnan(trial_isi[:min_length]) &
                    ~np.isnan(trial_opto[:min_length]) &
                    ~np.isnan(trial_block[:min_length]) &
                    ~np.isnan(trial_num_key[:min_length])
                )

                isi.extend(trial_isi[:min_length][valid_mask])
                opto.extend(trial_opto[:min_length][valid_mask])
                block.extend(trial_block[:min_length][valid_mask])
                trial_num.extend(trial_num_key[:min_length][valid_mask])

        return np.array(isi), np.array(opto), np.array(block), np.array(trial_num)

    if filter_outcomes == 'all':
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'rewarded':
        left_keys = ['short_ISI_reward_left_correct_lick']
        right_keys = ['long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'punished':
        left_keys = ['long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick']

    left_isi, left_opto, left_block, left_trial = extract_lick_data(left_keys, lick_properties_list)
    right_isi, right_opto, right_block, right_trial = extract_lick_data(right_keys, lick_properties_list)

    # Combine data
    isi_values = np.concatenate([left_isi, right_isi])
    choices = np.concatenate([np.zeros(len(left_isi)), np.ones(len(right_isi))])
    opto_tags = np.concatenate([left_opto, right_opto])
    block_types = np.concatenate([left_block, right_block])
    trial_numbers = np.concatenate([left_trial, right_trial])

    # Verify array lengths
    if not (len(isi_values) == len(choices) == len(opto_tags) == len(block_types) == len(trial_numbers)):
        raise ValueError(f"Array length mismatch: ISI={len(isi_values)}, "
                        f"Choices={len(choices)}, Opto={len(opto_tags)}, Block={len(block_types)}, Trial={len(trial_numbers)}")

    if len(isi_values) == 0:
        ax.text(0.5, 0.5, 'No valid data to plot after pooling sessions', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Pooled Psychometric Curve - {filter_outcomes.capitalize()}')
        return fig, ax, {}

    # Filter data based on block_selection
    if block_selection == 'neutral':
        mask = block_types == 0
        isi_values = isi_values[mask]
        choices = choices[mask]
        opto_tags = opto_tags[mask]
        trial_numbers = trial_numbers[mask]
        if len(isi_values) == 0:
            ax.text(0.5, 0.5, 'No neutral block trials found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Pooled Psychometric Curve - Neutral ({filter_outcomes.capitalize()})')
            return fig, ax, {}
    else:  # first_half or second_half
        mask = block_types != 0
        isi_values = isi_values[mask]
        choices = choices[mask]
        opto_tags = opto_tags[mask]
        trial_numbers = trial_numbers[mask]
        if len(isi_values) == 0:
            ax.text(0.5, 0.5, 'No non-neutral trials found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Pooled Psychometric Curve - {block_selection.capitalize()} ({filter_outcomes.capitalize()})')
            return fig, ax, {}

        # Sort by trial number for chronological order
        valid_mask = ~np.isnan(trial_numbers)
        if np.sum(valid_mask) == 0:
            sorted_indices = np.arange(len(isi_values))
        else:
            isi_values = isi_values[valid_mask]
            choices = choices[valid_mask]
            opto_tags = opto_tags[valid_mask]
            trial_numbers = trial_numbers[valid_mask]
            sorted_indices = np.argsort(trial_numbers)
            isi_values = isi_values[sorted_indices]
            choices = choices[sorted_indices]
            opto_tags = opto_tags[sorted_indices]

        # Split into halves
        n_trials = len(isi_values)
        mid_point = n_trials // 2
        if block_selection == 'first_half':
            isi_values = isi_values[:mid_point]
            choices = choices[:mid_point]
            opto_tags = opto_tags[:mid_point]
        else:  # second_half
            isi_values = isi_values[mid_point:]
            choices = choices[mid_point:]
            opto_tags = opto_tags[mid_point:]

    # Check for discrete ISI case
    unique_isi = np.unique(isi_values)
    is_discrete = len(unique_isi) <= 2

    # Define logistic function
    def logistic_function(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    # Plotting function for a group
    def plot_group(isi, choices, opto_tags, label, color, opto_value=None):
        isi = np.array(isi)
        choices = np.array(choices)
        opto_tags = np.array(opto_tags)

        mask = np.ones(len(isi), dtype=bool)
        if opto_value is not None:
            mask &= (opto_tags == opto_value)
        isi = isi[mask]
        choices = choices[mask]

        if len(isi) == 0:
            return None

        if is_discrete:
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

            ax.errorbar(valid_centers, valid_prob, yerr=valid_sem, fmt='o', color=color,
                       label=f"{label} data", capsize=3, alpha=0.7)
            if len(valid_centers) > 1:
                ax.plot(valid_centers, valid_prob, '-', color=color, linewidth=2,
                       label=f"{label} line", alpha=0.7)

            fit_params = None
            if len(valid_centers) == 2:
                x1, x2 = valid_centers
                y1, y2 = valid_prob
                if y2 != y1:
                    m = (y2 - y1) / (x2 - x1)
                    b = y1 - m * x1
                    inflection_point = (0.5 - b) / m
                    if min(x1, x2) <= inflection_point <= max(x1, x2):
                        ax.text(inflection_point, 0.1, f'{label} IP: {inflection_point:.2f}s',
                               color=color, ha='center', bbox=dict(facecolor='white', alpha=0.7))
                        fit_params = {'x0': inflection_point, 'slope': m, 'intercept': b}

            return fit_params

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
                bin_choices = choices[mask]
                if len(bin_choices) > 0:
                    right_prob[i] = np.mean(bin_choices)
                    sem[i] = np.std(bin_choices) / np.sqrt(len(bin_choices))
                    counts[i] = len(bin_choices)

            valid_mask = counts > 0
            valid_centers = bin_centers[valid_mask]
            valid_prob = right_prob[valid_mask]
            valid_sem = sem[valid_mask]

            ax.errorbar(valid_centers, valid_prob, yerr=valid_sem, fmt='o', color=color,
                       label=f"{label} data", capsize=3, alpha=0.7)

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
                    ip_value = x_fit[ip_index]
                    fit_params = {'L': popt[0], 'k': popt[1], 'x0': popt[2]}
                except Exception as e:
                    print(f"Could not fit logistic function for {label}: {e}")

            return fit_params

    # Plot data based on block_selection
    fit_params = {}
    title_suffix = filter_outcomes.capitalize()
    if block_selection == 'neutral':
        title_suffix += ' Trials - Neutral Block'
        if opto_split:
            fit_params['control'] = plot_group(
                isi_values, choices, opto_tags, 'Control',
                colors['control']['neutral'], opto_value=0
            )
            if np.any(opto_tags == 1):
                fit_params['opto'] = plot_group(
                    isi_values, choices, opto_tags, 'Opto',
                    colors['opto']['neutral'], opto_value=1
                )
        else:
            fit_params['neutral'] = plot_group(
                isi_values, choices, opto_tags, 'Neutral',
                colors[filter_outcomes]['neutral']
            )
    elif block_selection == 'first_half':
        title_suffix += ' Trials - First Half (Excl. Neutral)'
        if opto_split:
            fit_params['control'] = plot_group(
                isi_values, choices, opto_tags, 'First Half Control',
                colors['control']['first_half'], opto_value=0
            )
            if np.any(opto_tags == 1):
                fit_params['opto'] = plot_group(
                    isi_values, choices, opto_tags, 'First Half Opto',
                    colors['opto']['first_half'], opto_value=1
                )
        else:
            fit_params['first_half'] = plot_group(
                isi_values, choices, opto_tags, 'First Half',
                colors[filter_outcomes]['first_half']
            )
    else:  # second_half
        title_suffix += ' Trials - Second Half (Excl. Neutral)'
        if opto_split:
            fit_params['control'] = plot_group(
                isi_values, choices, opto_tags, 'Second Half Control',
                colors['control']['second_half'], opto_value=0
            )
            if np.any(opto_tags == 1):
                fit_params['opto'] = plot_group(
                    isi_values, choices, opto_tags, 'Second Half Opto',
                    colors['opto']['second_half'], opto_value=1
                )
        else:
            fit_params['second_half'] = plot_group(
                isi_values, choices, opto_tags, 'Second Half',
                colors[filter_outcomes]['second_half']
            )

    # Set x-axis limits
    if is_discrete:
        min_isi = min(unique_isi) - 0.1 * abs(min(unique_isi))
        max_isi = max(unique_isi) + 0.1 * abs(max(unique_isi))
    else:
        min_isi = np.floor(isi_values.min() / bin_width) * bin_width
        max_isi = np.ceil(isi_values.max() / bin_width) * bin_width
    ax.set_xlim(min_isi, max_isi)
    ax.set_ylim(-0.05, 1.05)

    # Customize plot
    ax.set_xlabel('Inter-Stimulus Interval (s)', fontsize=12)
    ax.set_ylabel('Probability of Right Choice', fontsize=12)
    ax.set_title(f'Pooled Psychometric Curve - {title_suffix}', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0.5, color='k', linestyle='-', alpha=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    return fig, ax, fit_params

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_grand_average_psychometric_neutral_and_halves(lick_properties_list, filter_outcomes='all', ax=None, bin_width=0.05, fit_logistic=True, opto_split=False, block_selection='neutral'):
    """
    Plot grand average psychometric curve across multiple sessions for neutral blocks (block_type 0), first half, or second half of non-neutral trials.
    Shows probability of right port choice as a function of ISI with SEM across sessions.
    Handles both continuous ISI values and fixed single ISI values.

    Parameters:
    -----------
    lick_properties_list : list of dict
        List of dictionaries, each containing lick properties from a session
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
    block_selection : str, optional
        'neutral' for block_type 0, 'first_half' for first half of non-neutral trials, 'second_half' for second half

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object with the plot
    fit_params : dict or None
        Parameters of the logistic fit (for continuous ISIs) or linear fit (for discrete ISIs), if applicable
    """
    # Validate block_selection parameter
    valid_blocks = ['neutral', 'first_half', 'second_half']
    if block_selection not in valid_blocks:
        raise ValueError(f"block_selection must be one of {valid_blocks}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Define colors
    colors = {
        'all': {'neutral': '#787573', 'first_half': '#1f77b4', 'second_half': '#d62728'},
        'rewarded': {'neutral': "#8bfa7c", 'first_half': '#1f77b4', 'second_half': '#d62728'},
        'punished': {'neutral': "#d77267", 'first_half': '#1f77b4', 'second_half': '#d62728'},
        'opto': {'neutral': "#8c8ce2", 'first_half': '#8c8ce2', 'second_half': '#0133c7'},
        'control': {'neutral': "#787573", 'first_half': '#787573', 'second_half': '#000000'}
    }

    def extract_session_data(keys, lick_properties):
        isi, opto, block, trial_num = [], [], [], []
        for key in keys:
            if key not in lick_properties:
                continue
            trial_isi = np.array(lick_properties[key]['Trial_ISI'])
            trial_opto = np.array(lick_properties[key]['opto_tag'])
            trial_block = np.array(lick_properties[key]['block_type'])
            # Safely handle trial_number, replacing None or invalid values with np.nan
            trial_num_key = lick_properties[key].get('trial_number', [np.nan] * len(trial_isi))
            # Convert to numeric array, replacing None with np.nan
            trial_num_key = [x if x is not None and not np.isnan(x) else np.nan for x in trial_num_key]
            trial_num_key = np.array(trial_num_key, dtype=float)

            min_length = min(len(trial_isi), len(trial_opto), len(trial_block), len(trial_num_key))
            if min_length == 0:
                continue

            valid_mask = (
                ~np.isnan(trial_isi[:min_length]) &
                ~np.isnan(trial_opto[:min_length]) &
                ~np.isnan(trial_block[:min_length]) &
                ~np.isnan(trial_num_key[:min_length])
            )

            isi.extend(trial_isi[:min_length][valid_mask])
            opto.extend(trial_opto[:min_length][valid_mask])
            block.extend(trial_block[:min_length][valid_mask])
            trial_num.extend(trial_num_key[:min_length][valid_mask])

        return np.array(isi), np.array(opto), np.array(block), np.array(trial_num)

    if filter_outcomes == 'all':
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'rewarded':
        left_keys = ['short_ISI_reward_left_correct_lick']
        right_keys = ['long_ISI_reward_right_correct_lick']
    elif filter_outcomes == 'punished':
        left_keys = ['long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick']

    # Collect all ISI and block for discrete check
    all_isi, all_block = [], []
    for lick_properties in lick_properties_list:
        l_isi, _, l_block, _ = extract_session_data(left_keys, lick_properties)
        r_isi, _, r_block, _ = extract_session_data(right_keys, lick_properties)
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
        min_isi = min(isi_centers) - 0.1 * abs(min(isi_centers)) if len(isi_centers) > 0 else 0
        max_isi = max(isi_centers) + 0.1 * abs(max(isi_centers)) if len(isi_centers) > 0 else 1
        bins = None
    else:
        min_isi = np.floor(all_isi.min() / bin_width) * bin_width
        max_isi = np.ceil(all_isi.max() / bin_width) * bin_width
        bins = np.arange(min_isi, max_isi + bin_width, bin_width)
        bin_centers = bins[:-1] + bin_width / 2
        isi_centers = bin_centers

    def logistic_function(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def plot_group(label, color, opto_value=None):
        session_means, session_counts = [], []
        for lick_properties in lick_properties_list:
            l_isi, l_opto, l_block, l_trial = extract_session_data(left_keys, lick_properties)
            r_isi, r_opto, r_block, r_trial = extract_session_data(right_keys, lick_properties)
            isi = np.concatenate([l_isi, r_isi])
            opto = np.concatenate([l_opto, r_opto])
            block = np.concatenate([l_block, r_block])
            trial = np.concatenate([l_trial, r_trial])
            choices = np.concatenate([np.zeros(len(l_isi)), np.ones(len(r_isi))])

            if len(isi) == 0:
                continue

            # Apply block_selection filtering
            initial_mask = np.ones(len(isi), dtype=bool)
            if block_selection == 'neutral':
                initial_mask &= (block == 0)
            else:  # first_half or second_half
                initial_mask &= (block != 0)

                # Sort by trial number for chronological order
                valid_trial_mask = ~np.isnan(trial)
                if np.sum(valid_trial_mask) == 0:
                    # Fallback: use original order
                    sorted_idx = np.arange(len(isi))
                else:
                    # Use only valid trials for sorting
                    sub_isi = isi[valid_trial_mask]
                    sub_choices = choices[valid_trial_mask]
                    sub_opto = opto[valid_trial_mask]
                    sub_block = block[valid_trial_mask]
                    sub_trial = trial[valid_trial_mask]
                    sorted_idx = np.argsort(sub_trial)
                    isi = sub_isi[sorted_idx]
                    choices = sub_choices[sorted_idx]
                    opto = sub_opto[sorted_idx]
                    block = sub_block[sorted_idx]
                    initial_mask = np.ones(len(isi), dtype=bool)

                # Split into halves
                n = len(isi)
                mid = n // 2
                if block_selection == 'first_half':
                    half_mask = np.arange(mid)
                else:  # second_half
                    half_mask = np.arange(mid, n)
                initial_mask &= np.isin(np.arange(len(isi)), half_mask)

            # Apply opto mask
            opto_mask = np.ones(len(isi), dtype=bool)
            if opto_value is not None:
                opto_mask &= (opto == opto_value)

            final_mask = initial_mask & opto_mask
            isi = isi[final_mask]
            choices = choices[final_mask]

            if len(isi) == 0:
                continue

            # Compute per-bin statistics for this session
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
            print(f"No valid data for {label}")
            return None

        session_means = np.array(session_means)
        session_counts = np.array(session_counts)
        grand_mean = np.nanmean(session_means, axis=0)
        grand_sem = np.nanstd(session_means, axis=0) / np.sqrt(np.sum(session_counts > 0, axis=0))
        valid_mask = np.sum(session_counts, axis=0) > 0
        valid_centers = isi_centers[valid_mask]
        valid_mean = grand_mean[valid_mask]
        valid_sem = grand_sem[valid_mask]

        block_label = {
            'neutral': 'Neutral Block',
            'first_half': 'First Half (Excl. Neutral)',
            'second_half': 'Second Half (Excl. Neutral)'
        }[block_selection]

        ax.errorbar(valid_centers, valid_mean, yerr=valid_sem, fmt='o', color=color,
                    label=f"{label} data ({block_label})", capsize=3, alpha=0.7)

        if is_discrete and len(valid_centers) > 1:
            ax.plot(valid_centers, valid_mean, '-', color=color, linewidth=2,
                    label=f"{label} line ({block_label})", alpha=0.7)

        fit_params = None
        if fit_logistic and not is_discrete and len(valid_centers) > 3:
            try:
                modified_bounds = ([0.5, -10, min_isi], [1, 10, max_isi])
                popt, _ = curve_fit(logistic_function, valid_centers, valid_mean,
                                    p0=[1.0, 1.0, np.median(valid_centers)],
                                    bounds=modified_bounds)
                x_fit = np.linspace(min_isi, max_isi, 100)
                y_fit = logistic_function(x_fit, *popt)
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2,
                        label=f"{label} fit ({block_label})")
                ip_index = np.argmin(np.abs(y_fit - 0.5))
                ip_value = x_fit[ip_index]
                fit_params = {'L': popt[0], 'k': popt[1], 'x0': popt[2]}
            except Exception as e:
                print(f"Could not fit logistic function for {label} ({block_label}): {e}")

        # For discrete case, calculate inflection point
        if is_discrete and len(valid_centers) == 2:
            x1, x2 = valid_centers
            y1, y2 = valid_mean
            if y2 != y1:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                inflection_point = (0.5 - b) / m
                if min(x1, x2) <= inflection_point <= max(x1, x2):
                    ax.text(inflection_point, 0.1, f'{label} IP ({block_label}): {inflection_point:.2f}s',
                            color=color, ha='center', bbox=dict(facecolor='white', alpha=0.7))
                    fit_params = {'x0': inflection_point, 'slope': m, 'intercept': b}

        return fit_params

    # Plot data
    fit_params = {}
    title_suffix = filter_outcomes.capitalize() + ' Trials'
    if block_selection == 'neutral':
        title_suffix += ' - Neutral Block'
    elif block_selection == 'first_half':
        title_suffix += ' - First Half (Excl. Neutral)'
    else:
        title_suffix += ' - Second Half (Excl. Neutral)'

    if opto_split:
        # Plot control
        fit_params['control'] = plot_group(
            'Control',
            colors['control'][block_selection],
            opto_value=0
        )
        # Plot opto if exists
        if any(any(lp.get('opto_tag', [])) for lp in lick_properties_list):
            fit_params['opto'] = plot_group(
                'Opto',
                colors['opto'][block_selection],
                opto_value=1
            )
    else:
        fit_params['main'] = plot_group(
            filter_outcomes.capitalize(),
            colors[filter_outcomes][block_selection]
        )

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