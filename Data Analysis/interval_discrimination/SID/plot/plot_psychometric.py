# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 17:03:25 2025

@author: timst
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.special import expit  # logistic function
from plot.color_map import get_plot_color, get_residual_colormap

def plot_psychometric(
    data,
    session_info,
    config,
    ax=None,
    show_ci=True,
    show_fit=False,
    fit_params=None,
    legend=True,
    title=None,
    colors=None,
    discrete_alpha=0.5,
    ci_alpha=0.08,
    show_plot=True
):
    """
    Plot psychometric data (and optional logistic fits) by condition.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    if colors is None:
        colors = {}

    # If grouped by condition
    grouped = data.groupby('condition')
    
    # Create label map from condition to condition_label (assumes it's consistent)
    # label_map = data.drop_duplicates('condition').set_index('condition')['condition_label'].to_dict()
    # region_map = data.drop_duplicates('condition').set_index('condition')['opto_region'].to_dict()
    label_map = (
        data.drop_duplicates('condition')
            .set_index('condition')['condition_label']
            .to_dict()
    )
    
    region_map = (
        data.drop_duplicates('condition')
            .set_index('condition')['opto_region']
            .to_dict()
    )
    
    # Add fallback for 'all' (no condition grouping)
    label_map['all'] = 'All Trials'
    region_map['all'] = 'Control'    

    for i, (condition, group) in enumerate(grouped):
        color = colors.get(condition, f"C{i}")
        x = group['stim_value']
        y = group['p_right']
        err = group['stderr']

    

        # Plot points
        # ax.plot(x, y, 'o-', label=str(condition), color=color)
        # ax.plot(x, y, 'o-', color=color)
        label = label_map.get(condition, str(condition))
        region = region_map.get(condition, str(condition))
        color = get_plot_color(label)
        ax.plot(x, y, 'o-', label=label, color=color, alpha=discrete_alpha)        

        # labels
        # for label, group in data.groupby('condition_label'):
        #     ax.plot(group['stim_value'], group['p_right'], label=label)

        # Add CI fill
        if show_ci:
            ax.fill_between(x, y - 1.96 * err, y + 1.96 * err, alpha=ci_alpha, color=color)

        for xi, yi, n in zip(group['stim_value'], group['p_right'], group['n_trials']):
            ax.annotate(f"n={n}", (xi, yi), textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=7, color=color, alpha=discrete_alpha)

        # Add logistic fit
        if show_fit and fit_params and condition in fit_params:
            # fit_x = np.linspace(min(x), max(x), 200)
            fit_x = fit_params[condition]['fit_x']
            beta_0, beta_1 = fit_params[condition]['params']
            # fit_y = expit(beta_0 + beta_1 * fit_x)
            fit_y = fit_params[condition]['fit_y']
            ax.plot(fit_x, fit_y, linestyle='--', color=color, alpha=1, label='Logistic Fit')

            # Optional: threshold line
            threshold = fit_params[condition]['threshold']
            ax.axvline(threshold, linestyle=':', color=color, alpha=0.5)

            # Annotate slope/threshold
            ax.annotate(f"Thresh: {threshold:.2f}", (threshold, 0.5),
                        xytext=(5, -5), textcoords='offset points',
                        fontsize=8, color=color)

        # annotate if sigmoid fit failed
        if show_fit and fit_params:
            fit = fit_params.get(condition)
            if fit and fit['fit_method'] == 'failed':
                ax.annotate(
                    "Fit skipped: insufficient data",
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=8, color=color,
                    verticalalignment='top'
                )

    # 50% line
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # effective category boundary
    # data['isi_mean']
    if 'isi_mean' in data.columns:
        isi_mean = data['isi_mean'].iloc[0]
        ax.axvline(isi_mean, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.annotate(f"Mean ISI\n{isi_mean:.0f} ms", 
                    xy=(isi_mean, 0.02),
                    xycoords='data',
                    textcoords='offset points',
                    xytext=(5, 5),
                    fontsize=8,
                    color='black',
                    rotation=90,
                    va='bottom')    


    # residual
    # region = session_info.get("OptoRegionShortText", "Unknown")
    # cmap = get_residual_colormap(region)

    # # labels
    # for label, group in data.groupby('condition_label'):
    #     ax.plot(group['stim_value'], group['p_right'], label=label)

    # set title
    subject_name = data['subject_name'].iloc[0] if 'subject_name' in data.columns else 'Unknown'
    session_date = data['SessionDate'].iloc[0] if 'SessionDate' in data.columns else 'Unknown'
    title = f"{subject_name}  {session_date}  ISI Psychometric"

    # Axes formatting
    ax.set_ylim(0, 1.05)
    ax.set_xlim(data['stim_value'].min()-200, data['stim_value'].max()+200)
    ax.set_xlabel("ISI [ms]")
    ax.set_ylabel("P(Right)")    
    if title:
        ax.set_title(title)
    if legend:
        # ax.legend(frameon=False)
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),  # Pushes legend outside the top-right corner
            borderaxespad=0,
            frameon=False,
            fontsize=9
        )     
    plt.tight_layout(pad=1.0)
    
    if show_plot:
        plt.show()
    
    subject = session_info['subject_name']
    date = session_info['date']
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject + '\\' + date)
    output_dir = os.path.join('plots\\' + subject + '\\' + date)
    figure_id = f"{subject}_{date}_psychometric"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)
    # fig.savefig(out_path, dpi=300)
    plt.close(fig)       
    
    
    
    return out_path
