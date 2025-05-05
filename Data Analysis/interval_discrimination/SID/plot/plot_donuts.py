# -*- coding: utf-8 -*-
"""
Created on Thu May  1 13:17:27 2025

@author: timst
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

# from adjustText import adjust_text

def plot_donut_chart(value_counts, title="", ax=None, colors=None, is_outcome_dist=False):
    """
    Draw a donut plot on the given axes with optional label or legend fallback.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure    
    
    # total = sum(values)
    autopct = lambda pct: f"{pct:.1f}%" if pct > 0 else ""
    
    total = value_counts.sum()
    labels = [f"{cat}\n(n={val})" for cat, val in zip(value_counts.index, value_counts.values)]
    values = value_counts.values    

    max_labels = 4

    if len(labels) > max_labels or is_outcome_dist:
        # Use legend instead of direct labels
        wedges, _ = ax.pie(
            values,
            startangle=90,
            counterclock=False,
            radius=0.9,  # slightly smaller than full circle
            wedgeprops=dict(width=0.1),
            colors=colors,
            labels=None
        )
        # ax.legend(
        #     wedges,
        #     [f"{l} ({v} | {v/total:.0%})" for l, v in zip(labels, values)],
        #     loc="center left",
        #     bbox_to_anchor=(1, 0.5),
        #     fontsize=8,
        # )
        # ax.legend(
        #     wedges,
        #     [f"{l} ({v} | {v/total:.0%})" for l, v in zip(labels, values)],
        #     loc="lower right",         # inside the axes
        #     fontsize=8,
        #     frameon=False,
        # )    
        ax.legend(
            wedges,
            [f"{l} ({v} | {v/total:.0%})" for l, v in zip(labels, values)],
            loc="center",
            fontsize=10,
            frameon=False,
            handlelength=1.0,
            handletextpad=0.5,
            borderaxespad=0.0
        )        
    else:
        # Use labels directly on slices
        # wedges, texts, autotexts = ax.pie(
        #     values,
        #     labels=labels,
        #     startangle=90,
        #     counterclock=False,
        #     wedgeprops=dict(width=0.1),
        #     colors=colors,
        #     autopct=autopct,
        #     textprops=dict(color="black", fontsize=8),
        #     labeldistance=0.5,  # ← smaller = closer to center
        # )
        wedges, texts = ax.pie(
            values,
            labels=labels,
            startangle=90,
            counterclock=False,
            radius=0.9,  # slightly smaller than full circle
            wedgeprops=dict(width=0.1),
            colors=colors,
            textprops={'fontsize': 10},
            labeldistance=1.1  # move labels further out
        )         

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # ax.set_title(title, fontsize=10)
    # ax.set_aspect("equal")   
    full_title = f"{title} (n={total})" if total > 0 else title
    ax.set_title(full_title, fontsize=13, pad=35)
    ax.axis('equal')  # Keep the circle    
    
    return ax
   # """
    # Plot a donut chart showing value_counts with n= labels and total in title.
    # """
    # if ax is None:
    #     fig, ax = plt.subplots(figsize=(6, 6))
    # else:
    #     fig = ax.figure

    # total = value_counts.sum()
    # labels = [f"{cat}\n(n={val})" for cat, val in zip(value_counts.index, value_counts.values)]
    # sizes = value_counts.values

    # wedges, texts = ax.pie(
    #     sizes,
    #     labels=labels,
    #     startangle=90,
    #     counterclock=False,
    #     radius=0.9,  # slightly smaller than full circle
    #     wedgeprops=dict(width=0.1),
    #     colors=colors,
    #     textprops={'fontsize': 10},
    #     labeldistance=1.2  # move labels further out
    # )    

    # # if len(labels) > 5:
    #     # adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle='->'))
    # # adjust_text(
    # #     texts,
    # #     only_move={'points': 'y', 'texts': 'y'},
    # #     arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
    # #     force_text=0.75, force_points=0.5,
    # #     expand_points=(1.2, 1.4),  # optional tweak to spacing behavior
    # # )        

    # full_title = f"{title} (n={total})" if total > 0 else title
    # ax.set_title(full_title, fontsize=13, pad=35)
    # ax.axis('equal')  # Keep the circle
    # return ax


def make_outcome_by_trial_side_non_naive_donut(df, session_info, ax=None, show_plot=True):
    """
    Donut chart split *vertically*:
    - Left trials drawn in left half (90° to 270°)
    - Right trials drawn in right half (-90° to +90°)
    - Outcome colors: Reward=green, Punish=red
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Wedge

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # Filter to non-naive reward/punish
    subset = df[
        (df['naive'] == 0) &
        (df['outcome'].isin(['Reward', 'Punish']))
    ]

    # Get side counts
    left = subset[subset['trial_side'] == 'left']['outcome'].value_counts()
    right = subset[subset['trial_side'] == 'right']['outcome'].value_counts()

    for grp in [left, right]:
        for k in ['Reward', 'Punish']:
            if k not in grp:
                grp[k] = 0

    left = left[['Reward', 'Punish']]
    right = right[['Reward', 'Punish']]

    total_left = left.sum()
    total_right = right.sum()
    total = total_left + total_right

    if total == 0:
        ax.text(0.5, 0.5, "No non-naive Reward/Punish trials", ha='center', va='center')
        ax.axis('off')
        return ax

    def add_half_donut(ax, counts, center_angle, side_label):
        wedges = []
        angle = center_angle
        direction = 1  # clockwise
        outcome_order = ['Reward', 'Punish']
        colors = ['green', 'red']
        start = angle - 90
        labels = []

        # Normalize this half to 180°
        if counts.sum() > 0:
            fracs = (counts / counts.sum()) * 180
        else:
            fracs = [0, 0]

        for outcome, size, color in zip(outcome_order, fracs, colors):
            wedge = Wedge(
                center=(0, 0),
                r=1.1,
                theta1=start,
                theta2=start + size,
                width=0.1,
                facecolor=color,
                edgecolor='white'
            )
            ax.add_patch(wedge)

            # Label angle and placement
            label_angle = (start + start + size) / 2
            rad = np.deg2rad(label_angle)
            # x = 0.7 * np.cos(rad)
            # y = 0.7 * np.sin(rad)
            x = 1.2 * np.cos(rad)
            y = 1.2 * np.sin(rad)       
            ha = 'left' if x >= 0 else 'right'
            count = counts[outcome]
            if count > 0:
                ax.text(
                    x, y, f"{side_label}\n{outcome}\n(n={count})",
                    ha=ha, va='center', fontsize=10
                )
            start += size

    # Plot left (90° center), right (-90° center)
    add_half_donut(ax, left, center_angle=180, side_label="Left")   # left side: 90°–270°
    add_half_donut(ax, right, center_angle=0, side_label="Right")   # right side: −90°–+90°

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(f"Outcome by Trial Side (n={total})", fontsize=13, pad=35)
    ax.axis('off')

    if show_plot:
        plt.show()
    
    subject = session_info['subject_name']
    date = session_info['date']
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject + '\\' + date)
    output_dir = os.path.join('plots\\' + subject + '\\' + date)
    figure_id = f"{subject}_{date}_outcome_by_trial_side_non_naive_donut"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches=None, pad_inches=0.1, dpi=300)
    plt.close(fig)       

    return out_path    


def make_opto_control_donut(df, session_info, ax=None, show_plot=True):
    """
    Create a donut chart showing opto vs control trial count.
    """
    # Filter to non-naive reward/punish
    subset = df[
        (df['naive'] == 0) &
        (df['outcome'].isin(['Reward', 'Punish']))
    ]
    
    
    mapping = {0: 'Control', 1: 'Opto'}
    value_counts = subset['is_opto'].map(mapping).value_counts().sort_index()
      
    # Fixed outcome color map
    condition_colors = {
        'Control': '#000000',         # green
        'Opto': '#1f77b4',         # red
    }    
    
    # Assign colors based on value_counts index order
    labels = value_counts.index.tolist()
    colors = [condition_colors.get(label, '#cccccc') for label in labels]      
    
    
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure   
    
    title = f"Opto vs Control"
    # return plot_donut_chart(value_counts, title=title, ax=ax, colors=colors)
    plot_donut_chart(value_counts, title=title, ax=ax, colors=colors)

    if show_plot:
        plt.show()

    subject = session_info['subject_name']
    date = session_info['date']
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject + '\\' + date)
    output_dir = os.path.join('plots\\' + subject + '\\' + date)
    figure_id = f"{subject}_{date}_opto_control_donut"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches=None, pad_inches=0.1, dpi=300)
    plt.close(fig)       

    return out_path     



def make_trial_isi_donut(df, session_info, ax=None, show_plot=True):
    """
    Donut plot showing distribution of trial types mapped to ISI:
    - Left = Short ISI
    - Right = Long ISI
    """
    isi_map = {0: 'Short ISI', 1: 'Long ISI'}
    
    
    # Filter to non-naive reward/punish
    subset = df[
        (df['naive'] == 0) &
        (df['outcome'].isin(['Reward', 'Punish']))
    ]
        
    value_counts = subset['is_right'].map(isi_map).value_counts()
 
    # Fixed outcome color map
    condition_colors = {
        'Short ISI': '#d9d9d9',         # light grey
        'Long ISI': '#b0b0b0',         # medium grey
    }
        
    # Assign colors based on value_counts index order
    labels = value_counts.index.tolist()
    colors = [condition_colors.get(label, '#cccccc') for label in labels]   


    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure   
        
    
    # return plot_donut_chart(value_counts, title="Trial Type (ISI)", ax=ax, colors=colors)
    plot_donut_chart(value_counts, title="Trial Type (ISI)", ax=ax, colors=colors)

    if show_plot:
        plt.show()

    subject = session_info['subject_name']
    date = session_info['date']
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject + '\\' + date)
    output_dir = os.path.join('plots\\' + subject + '\\' + date)
    figure_id = f"{subject}_{date}_trial_isi_donut"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    # fig.savefig(out_path, bbox_inches=None, pad_inches=0.1, dpi=300)
    fig.savefig(out_path, pad_inches=0.1, dpi=300)
    plt.close(fig)       

    return out_path    
    

def make_outcome_all_trials_donut(df, session_info, ax=None, show_plot=True):
    """
    Donut chart showing outcome distribution across all trials,
    including naive, did-not-choose, etc.
    """
    value_counts = df['outcome'].value_counts()
    
    # Fixed outcome color map
    outcome_colors = {
        'Reward': '#2ca02c',         # green
        'Punish': '#d62728',         # red
        'DidNotChoose': '#e0e0e0',   # very light grey
        'RewardNaive': '#98df8a',    # light green
        'PunishNaive': '#ff9896'     # light red
    }
    
    # Assign colors based on value_counts index order
    labels = value_counts.index.tolist()
    colors = [outcome_colors.get(label, '#cccccc') for label in labels]    
    
    # return plot_donut_chart(value_counts, title="Outcome Distribution (All Trials)", ax=ax, colors=colors)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure    
    
    plot_donut_chart(value_counts, ax=ax, title="Outcome Distribution (All Trials)", colors=colors, is_outcome_dist=True)

    if show_plot:
        plt.show()
    
    subject = session_info['subject_name']
    date = session_info['date']
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject + '\\' + date)
    output_dir = os.path.join('plots\\' + subject + '\\' + date)
    figure_id = f"{subject}_{date}_outcome_all_trials_donut"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    # fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1, dpi=300)  #  Use bbox_inches='tight' (but carefully)
    fig.savefig(out_path, bbox_inches=None, pad_inches=0.1, dpi=300)  #  Use bbox_inches='tight' (but carefully)
    plt.close(fig)       

    return out_path
    
