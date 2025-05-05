# -*- coding: utf-8 -*-
"""
Created on Thu May  1 14:54:00 2025

@author: timst
"""
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_isi_pdf(pdf_dict, session_info, title="ISI Probability Density", is_cover=False, ax=None, show_mean_lines=True, show_plot=True):
    """
    Plot ISI PDF curves with shaded areas from compute_isi_pdf() output.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    color_map = {
        'left': {'line': '#1f77b4', 'fill': '#aec7e8'},   # blue tones
        'left_opto':     {'line': '#5fa2d3', 'fill': '#d0e5f5'},
        'right': {'line': '#d62728', 'fill': '#f7b6b2'},   # red tones
        'right_opto':     {'line': '#e36a6a', 'fill': '#fcdede'},
    }
    
    
    for group, data in pdf_dict.items():
        if isinstance(group, tuple):
            isi = group[0]
            opto = group[1]
            label_isi = f"{'Short' if isi == 'left' else 'Long'}"
            label_opto = f"{'Control' if opto == 0 else 'Opto'}"
            label = f"{label_isi} {label_opto} ISI (n={data['count']})"
            group_color = f"{isi}{'_opto' if opto == 1 else ''}"
            colors = color_map.get(group_color, {'line': 'gray', 'fill': 'lightgray'})
        else:
            colors = color_map.get(group, {'line': 'gray', 'fill': 'lightgray'})
            label = f"{'Short' if group == 'left' else 'Long'} ISI (n={data['count']})"

        # Line
        ax.plot(data['x'], data['y'], label=label, color=colors['line'], linewidth=2)

        # Fill under curve
        ax.fill_between(data['x'], data['y'], color=colors['fill'], alpha=0.5)

        # Optional mean line
        if show_mean_lines:
            ax.axvline(data['mean'], color=colors['line'], linestyle='--', alpha=0.7)

    # Add a fake, invisible tick label to reserve space
    # ax.text(
    #     0, 1, "0.00000", 
    #     transform=ax.transAxes,
    #     fontsize=8,
    #     ha='left', va='top',
    #     alpha=1  # invisible, but reserves space
    # )
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    
    # ax.ticklabel_format(axis='y', style='plain')
    # ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.5f"))    
    
    # fig.subplots_adjust(left=0.2)  # Increase if needed
    
    # ax.set_xlim(ax.get_xlim())
    # ax.set_ylim(ax.get_ylim())
    # ax.set_position(ax.get_position())
    # ax.set_autoscale_on(False)

    ax.set_xlabel("ISI Duration (ms)")
    ax.set_ylabel("Probability Density")
    
    subject = session_info['subject_name']
    date = session_info['date']
    
    # title = subject + ' ' + session_info['SessionDate'] + ' ' + title
    
    # ax.set_title(title)

    if not is_cover:
        title = subject + ' ' + session_info['SessionDate'] + ' ' + title
            
    ax.set_title(title, fontsize=10)

    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),  # Pushes legend outside the top-right corner
        borderaxespad=0,
        frameon=False,
        fontsize=9
    )     
    
    ax.grid(True, alpha=0.3)
   
    if show_plot:
        plt.show()
    
    
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject + '\\' + date)
    output_dir = os.path.join('plots\\' + subject + '\\' + date)
    figure_id = f"{subject}_{date}_isi_pdf"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)       

    return out_path
