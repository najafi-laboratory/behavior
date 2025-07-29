import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import sem
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import os

def plot_psychometric_epochs(sessions_data, subject, data_paths, output_pdf='psychometric_epochs.pdf', bin_width=0.01, fit_logistic=True, save_path=None):
    """
    Plot psychometric curves for early and late epochs of short and long blocks, including pooled data
    across sessions and individual session plots, excluding opto trials and neutral blocks. Uses GridSpec
    with 6 subplots per row (early/late for short, long, and all blocks) and saves to a PDF.

    Args:
        sessions_data (dict): Dictionary from prepare_session_data containing outcomes, lick_properties, opto_tags, block_types, and dates.
        output_pdf (str): Path to save the output PDF file.
        bin_width (float): Width of ISI bins for multiple ISI cases.
        fit_logistic (bool): Whether to fit a logistic curve for multiple ISI cases.

    Returns:
        None: Generates and saves plots to a PDF file.
    """
    # Extract session data
    dates = sessions_data['dates']
    outcomes_list = sessions_data['outcomes']
    lick_properties_list = sessions_data['lick_properties']
    opto_tags_list = sessions_data['opto_tags']
    block_types_list = sessions_data['block_type']
    n_sessions = len(dates)

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

    def split_block_epochs(block_types, outcomes, opto_tags, lick_properties):
        """Split blocks into early and late epochs, excluding opto trials."""
        blocks = identify_blocks(block_types)
        early_data = {'short': {'isi': [], 'choices': []}, 'long': {'isi': [], 'choices': []}, 'all': {'isi': [], 'choices': []}}
        late_data = {'short': {'isi': [], 'choices': []}, 'long': {'isi': [], 'choices': []}, 'all': {'isi': [], 'choices': []}}
        
        left_keys = ['short_ISI_reward_left_correct_lick', 'long_ISI_punish_left_incorrect_lick']
        right_keys = ['short_ISI_punish_right_incorrect_lick', 'long_ISI_reward_right_correct_lick']
        
        for block_type, start, end in blocks:
            if block_type not in [1, 2]:
                continue
            block_length = end - start
            midpoint = start + block_length // 2
            epoch = 'early' if block_length > 1 else 'late'  # Single-trial blocks go to late
            
            for i in range(start, end):
                if opto_tags[i] == 1:  # Exclude opto trials
                    continue
                block_key = 'short' if block_type == 1 else 'long'
                target = early_data if i < midpoint else late_data
                
                # Extract ISI and choices from lick_properties for this trial
                for key in left_keys + right_keys:
                    if i < len(lick_properties[key]['Trial_ISI']) and lick_properties[key]['Trial_ISI'][i] is not None:
                        target[block_key]['isi'].append(lick_properties[key]['Trial_ISI'][i])
                        target[block_key]['choices'].append(1 if key in right_keys else 0)
                        target['all']['isi'].append(lick_properties[key]['Trial_ISI'][i])
                        target['all']['choices'].append(1 if key in right_keys else 0)
        
        return early_data, late_data

    def calculate_psychometric(isi, choices, bin_width, single_isi_case):
        """Calculate choice probabilities and SEM for psychometric curve."""
        if len(isi) == 0:
            return None, None, None
        
        if single_isi_case:
            unique_isi = np.unique(isi)
            if len(unique_isi) != 2:
                return None, None, None
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
        """Logistic function for fitting psychometric curves."""
        return L / (1 + np.exp(-k * (x - x0)))

    def plot_psychometric(ax, isi, choices, title, color, isi_divider, single_isi_case, fit_logistic):
        """Plot psychometric curve with SEM on given axis."""
        x_data, y_data, y_sem = calculate_psychometric(isi, choices, bin_width, single_isi_case)
        if x_data is None:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(title)
            ax.grid(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            return None
        
        ax.errorbar(x_data, y_data, yerr=y_sem, fmt='o', color=color, capsize=3, alpha=0.7, label='Data')
        if single_isi_case and len(x_data) == 2:
            ax.plot(x_data, y_data, '-', color=color, linewidth=2, label='Fit')
            x1, x2 = x_data
            y1, y2 = y_data
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
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=2, label='Fit')
                ip_index = np.argmin(np.abs(y_fit - 0.5))
                inflection_point = x_fit[ip_index]
                ax.axvline(x=inflection_point, color=color, linestyle='--', alpha=0.5)
            except Exception as e:
                print(f"Could not fit logistic function for {title}: {e}")
        
        ax.axvline(x=isi_divider, color='red', linestyle='--', alpha=0.3)
        ax.axhline(y=0.5, color='black', linestyle='-', alpha=0.2)
        ax.set_xlabel('Inter-Stimulus Interval (s)')
        ax.set_ylabel('Probability of Right Choice')
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend()

    # Colors for different conditions
    colors = {
        'early_short': '#1f77b4',  # Blue
        'late_short': '#ff7f0e',   # Orange
        'early_long': '#2ca02c',   # Green
        'late_long': '#d62728',    # Red
        'pooled_early_all': '#9467bd', # Purple
        'pooled_late_all': '#8c564b'   # Brown
    }

    # Process pooled data
    pooled_early = {'short': {'isi': [], 'choices': []}, 'long': {'isi': [], 'choices': []}, 'all': {'isi': [], 'choices': []}}
    pooled_late = {'short': {'isi': [], 'choices': []}, 'long': {'isi': [], 'choices': []}, 'all': {'isi': [], 'choices': []}}
    for outcomes, lick_props, opto_tags, block_types in zip(outcomes_list, lick_properties_list, opto_tags_list, block_types_list):
        early_data, late_data = split_block_epochs(block_types, outcomes, opto_tags, lick_props)
        for block in ['short', 'long', 'all']:
            pooled_early[block]['isi'].extend(early_data[block]['isi'])
            pooled_early[block]['choices'].extend(early_data[block]['choices'])
            pooled_late[block]['isi'].extend(late_data[block]['isi'])
            pooled_late[block]['choices'].extend(late_data[block]['choices'])
    
    # Check for single ISI case
    all_isi = []
    for block in ['short', 'long', 'all']:
        all_isi.extend(pooled_early[block]['isi'])
        all_isi.extend(pooled_late[block]['isi'])
    single_isi_case = len(np.unique(np.round(all_isi, 3))) == 2

    # Initialize PDF
    with PdfPages(output_pdf) as pdf:
        fig = plt.figure(figsize=(18, 4 * (n_sessions + 1)))
        gs = gridspec.GridSpec(n_sessions + 1, 6, figure=fig)

        # Plot pooled data (6 subplots)
        ax1 = fig.add_subplot(gs[0, 0])
        plot_psychometric(ax1, np.array(pooled_early['short']['isi']), np.array(pooled_early['short']['choices']),
                          'Pooled Early Short Blocks', colors['early_short'], lick_properties_list[0]['ISI_devider'],
                          single_isi_case, fit_logistic)
        
        ax2 = fig.add_subplot(gs[0, 1])
        plot_psychometric(ax2, np.array(pooled_late['short']['isi']), np.array(pooled_late['short']['choices']),
                          'Pooled Late Short Blocks', colors['late_short'], lick_properties_list[0]['ISI_devider'],
                          single_isi_case, fit_logistic)
        
        ax3 = fig.add_subplot(gs[0, 2])
        plot_psychometric(ax3, np.array(pooled_early['long']['isi']), np.array(pooled_early['long']['choices']),
                          'Pooled Early Long Blocks', colors['early_long'], lick_properties_list[0]['ISI_devider'],
                          single_isi_case, fit_logistic)
        
        ax4 = fig.add_subplot(gs[0, 3])
        plot_psychometric(ax4, np.array(pooled_late['long']['isi']), np.array(pooled_late['long']['choices']),
                          'Pooled Late Long Blocks', colors['late_long'], lick_properties_list[0]['ISI_devider'],
                          single_isi_case, fit_logistic)
        
        ax5 = fig.add_subplot(gs[0, 4])
        plot_psychometric(ax5, np.array(pooled_early['all']['isi']), np.array(pooled_early['all']['choices']),
                          'Pooled Early All Blocks', colors['pooled_early_all'], lick_properties_list[0]['ISI_devider'],
                          single_isi_case, fit_logistic)
        
        ax6 = fig.add_subplot(gs[0, 5])
        plot_psychometric(ax6, np.array(pooled_late['all']['isi']), np.array(pooled_late['all']['choices']),
                          'Pooled Late All Blocks', colors['pooled_late_all'], lick_properties_list[0]['ISI_devider'],
                          single_isi_case, fit_logistic)

        # Plot individual sessions (6 subplots per session)
        for i, (date, outcomes, lick_props, opto_tags, block_types) in enumerate(zip(dates, outcomes_list, lick_properties_list, opto_tags_list, block_types_list)):
            early_data, late_data = split_block_epochs(block_types, outcomes, opto_tags, lick_props)
            
            ax1 = fig.add_subplot(gs[i + 1, 0])
            plot_psychometric(ax1, np.array(early_data['short']['isi']), np.array(early_data['short']['choices']),
                              f'Early Short Blocks - {date}', colors['early_short'], lick_props['ISI_devider'],
                              single_isi_case, fit_logistic)
            
            ax2 = fig.add_subplot(gs[i + 1, 1])
            plot_psychometric(ax2, np.array(late_data['short']['isi']), np.array(late_data['short']['choices']),
                              f'Late Short Blocks - {date}', colors['late_short'], lick_props['ISI_devider'],
                              single_isi_case, fit_logistic)
            
            ax3 = fig.add_subplot(gs[i + 1, 2])
            plot_psychometric(ax3, np.array(early_data['long']['isi']), np.array(early_data['long']['choices']),
                              f'Early Long Blocks - {date}', colors['early_long'], lick_props['ISI_devider'],
                              single_isi_case, fit_logistic)
            
            ax4 = fig.add_subplot(gs[i + 1, 3])
            plot_psychometric(ax4, np.array(late_data['long']['isi']), np.array(late_data['long']['choices']),
                              f'Late Long Blocks - {date}', colors['late_long'], lick_props['ISI_devider'],
                              single_isi_case, fit_logistic)
            
            ax5 = fig.add_subplot(gs[i + 1, 4])
            plot_psychometric(ax5, np.array(early_data['all']['isi']), np.array(early_data['all']['choices']),
                              f'Early All Blocks - {date}', colors['pooled_early_all'], lick_props['ISI_devider'],
                              single_isi_case, fit_logistic)
            
            ax6 = fig.add_subplot(gs[i + 1, 5])
            plot_psychometric(ax6, np.array(late_data['all']['isi']), np.array(late_data['all']['choices']),
                              f'Late All Blocks - {date}', colors['pooled_late_all'], lick_props['ISI_devider'],
                              single_isi_case, fit_logistic)
        
        plt.tight_layout()
        if save_path:
            output_path = os.path.join(save_path, f'Psychometric_epochs_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)