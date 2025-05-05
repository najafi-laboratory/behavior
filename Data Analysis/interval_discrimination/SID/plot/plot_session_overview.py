# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 21:52:26 2025

@author: timst
"""
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


def plot_session_overview_trial_type(summary_df, session_info, ax=None, title=None, trial_spacing=2, show_mcs=True, show_plot=True):
    """
    Plot trial type (Left/Right) across trials.
    Colors indicate outcome (green = correct, red = incorrect),
    markers indicate naive (×) vs trained (o).
    Naive block is shaded automatically.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(24, 3.5))
    else:
        fig = ax.figure

    # Trial type Y values
    y_map = {0: 1, 1: 0}  # new: left on top, right on bottom    

    # Trial styling
    for _, row in summary_df.iterrows():
        
        outcome = row['outcome']
        if outcome == 'DidNotChoose':
            marker = 'o'
            facecolor = 'none'
            edgecolor = 'black'
        else:
            facecolor = 'green' if row['correct'] == 1 else 'red'
            marker = 'x' if row['naive'] == 1 else 'o'
            edgecolor = 'black'

        y = y_map[row['is_right']]
        # set left as top row, right as bottom row
        
        x = (row['trial_index'] + 1) * trial_spacing

        ax.scatter(x, y, facecolor=facecolor, marker=marker, s=20, alpha=0.85, edgecolors=edgecolor, zorder=3)
        
        # opto
        # Optional: draw purple triangle if MoveCorrectSpout is set
        if row.get('is_opto', 0) == 1:
            ax.scatter(
                x, y,
                marker='^',         # triangle
                color='purple',
                s=100,
                alpha=0.6,
                zorder=2            # behind outcome marker
            )
        
        # Optional: draw purple cross if MoveCorrectSpout is set
        if show_mcs:
            if row.get('MoveCorrectSpout', 0) == 1:
                ax.scatter(x, y, marker='+', color='purple', s=160, alpha=0.7, zorder=2)            

    # Shade naive trial range
    naive_trials = (summary_df[summary_df['naive'] == 1]['trial_index'] + 1) * trial_spacing
    naive_trials = naive_trials - trial_spacing/2
    if not naive_trials.empty:
        ax.axvspan(naive_trials.min(), naive_trials.max() + 1 * trial_spacing, color='lightblue', alpha=0.3, zorder=1)

    # Axes
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Right', 'Left'])
    ax.set_ylim(-0.5, 1.5)
    x_max = summary_df['trial_index'].max() * trial_spacing
    ax.set_xlim(0, x_max + trial_spacing*2)
    
    # Tick marks every 5 trials (in display, not spacing)
    tick_spacing = 5
    xticks_scaled = np.arange(0, x_max + 1, trial_spacing * tick_spacing)
    ax.set_xticks(xticks_scaled)
    ax.set_xticklabels([int(x / trial_spacing) for x in xticks_scaled])    
    
    ax.set_xlabel("Trial Index")
    ax.set_ylabel("Trial Type")
    if title:
        ax.set_title(title)

    # Legend (minimal)
    legend_elements = [
        Line2D([0], [0], marker='o', linestyle='None', color='green', label='Rewarded', markersize=7),
        Line2D([0], [0], marker='x', linestyle='None', color='red', label='Punished', markersize=7),
        Line2D([0], [0], marker='o', linestyle='None', color='green', label='Rewarded Naive', markersize=7),
        Line2D([0], [0], marker='x', linestyle='None', color='red', label='Punished Naive', markersize=7),
        Line2D([0], [0], marker='s', linestyle='None', color='lightblue', alpha=0.4, label='Naive Block', markersize=10)        
        
    ]
    
    
    # opto
    if 'is_opto' in summary_df.columns and summary_df['is_opto'].any():
        legend_elements.append(
            Line2D([0], [0], marker='^', linestyle='None',
                   color='purple', label='Opto Trial', markersize=7)
        )
    
    # move correct spout
    if show_mcs:
        if 'MoveCorrectSpout' in summary_df.columns and summary_df['MoveCorrectSpout'].any():
            legend_elements.append(
                Line2D([0], [0], marker='+', linestyle='None',
                       color='purple', label='MoveCorrectSpout', markersize=10)
            )    
    
    ax.legend(handles=legend_elements,
              loc='upper left',
              bbox_to_anchor=(1.005, 1.0),  # Pushes legend outside the top-right corner
              borderaxespad=0,
              frameon=False,
              fontsize=9)



    # set title
    subject_name = summary_df['subject_name'].iloc[0] if 'subject_name' in summary_df.columns else 'Unknown'
    session_date = summary_df['SessionDate'].iloc[0] if 'SessionDate' in summary_df.columns else 'Unknown'
    title = f"{subject_name}  {session_date}  Session Summary by Trial Type"

    if title:
        ax.set_title(title)

    plt.tight_layout(pad=1.0)
    
    if show_plot:
        plt.show()  
        
    subject = session_info['subject_name']
    date = session_info['date']
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject + '\\' + date)
    output_dir = os.path.join('plots\\' + subject + '\\' + date)
    figure_id = f"{subject}_{date}_session_overview_tt"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)           

    return out_path


def plot_session_overview_rt(summary_df, session_info, ax=None, title=None, trial_spacing=2, show_mcs=True, show_plot=True):
    """
    Plot reaction time across trials.
    Colors indicate outcome (green = correct, red = incorrect),
    markers indicate naive (×) vs trained (o),
    DidNotChoose trials shown as hollow black circles.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(24, 3.5))
    else:
        fig = ax.figure

    # Trial scatter
    for _, row in summary_df.iterrows():
        outcome = row['outcome']
        if outcome == 'DidNotChoose':
            marker = 'o'
            facecolor = 'none'
            edgecolor = 'black'
        else:
            facecolor = 'green' if row['correct'] == 1 else 'red'
            marker = 'x' if row['naive'] == 1 else 'o'
            edgecolor = 'black'

        x = (row['trial_index'] + 1) * trial_spacing
        y = row['reaction_time']

        ax.scatter(x, y, facecolor=facecolor, marker=marker, s=20,
                   alpha=0.85, edgecolors=edgecolor, zorder=3)
        
        # Optional: draw purple cross if MoveCorrectSpout is set
        if show_mcs:
            if row.get('MoveCorrectSpout', 0) == 1:
                ax.scatter(x, y, marker='+', color='purple', s=160, alpha=0.7, zorder=2)        

    # Shade naive trial range
    naive_trials = (summary_df[summary_df['naive'] == 1]['trial_index'] + 1) * trial_spacing
    naive_trials = naive_trials - trial_spacing / 2
    if not naive_trials.empty:
        ax.axvspan(naive_trials.min(), naive_trials.max() + trial_spacing,
                   color='lightblue', alpha=0.3, zorder=1)

    # Shade DidNotChoose trials
    dn_trials = (summary_df[summary_df['outcome'] == 'DidNotChoose']['trial_index'] + 1) * trial_spacing
    dn_trials = dn_trials - trial_spacing / 2
    
    for x in dn_trials:
        ax.axvspan(x, x + trial_spacing, color='lightgray', alpha=0.3, zorder=0)

    # Axis settings
    x_max = summary_df['trial_index'].max() * trial_spacing
    ax.set_xlim(0, x_max + trial_spacing * 2)
    
    # X-axis ticks every 5 trials
    tick_spacing = 5
    xticks_scaled = np.arange(0, x_max + 1, trial_spacing * tick_spacing)
    ax.set_xticks(xticks_scaled)
    ax.set_xticklabels([int(x / trial_spacing) for x in xticks_scaled])
    
    ax.set_xlabel("Trial Index")
    ax.set_ylabel("Reaction Time (s)")

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', linestyle='None', color='green', label='Rewarded', markersize=7),
        Line2D([0], [0], marker='x', linestyle='None', color='red', label='Punished', markersize=7),
        Line2D([0], [0], marker='o', linestyle='None', color='green', label='Rewarded Naive', markersize=7),
        Line2D([0], [0], marker='x', linestyle='None', color='red', label='Punished Naive', markersize=7),
        Line2D([0], [0], marker='o', linestyle='None', color='black', markerfacecolor='none', label='Did Not Choose', markersize=7),
        Line2D([0], [0], marker='s', linestyle='None', color='lightblue', alpha=0.4, label='Naive Block', markersize=10)
    ]
    
    # opto
    if 'is_opto' in summary_df.columns and summary_df['is_opto'].any():
        legend_elements.append(
            Line2D([0], [0], marker='^', linestyle='None',
                   color='purple', label='Opto Trial', markersize=7)
        )    
    
    # move correct spout
    if show_mcs:
        if 'MoveCorrectSpout' in summary_df.columns and summary_df['MoveCorrectSpout'].any():
            legend_elements.append(
                Line2D([0], [0], marker='+', linestyle='None',
                       color='purple', label='MoveCorrectSpout', markersize=10)
            )     
    
    ax.legend(handles=legend_elements,
              loc='upper left',
              bbox_to_anchor=(1.005, 0.97),
              borderaxespad=0,
              frameon=False,
              fontsize=9)

    # Auto-title
    subject_name = summary_df['subject_name'].iloc[0] if 'subject_name' in summary_df.columns else 'Unknown'
    session_date = summary_df['SessionDate'].iloc[0] if 'SessionDate' in summary_df.columns else 'Unknown'
    title = f"{subject_name}  {session_date}  Session Summary by Reaction Time"
    ax.set_title(title)

    plt.tight_layout(pad=1.0)
    
    if show_plot:
        plt.show()
    
    subject = session_info['subject_name']
    date = session_info['date']
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject + '\\' + date)
    output_dir = os.path.join('plots\\' + subject + '\\' + date)
    figure_id = f"{subject}_{date}_session_overview_rt"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)       

    return out_path