# -*- coding: utf-8 -*-
"""
Created on Fri May  2 13:53:36 2025

@author: timst
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from scipy.interpolate import interp1d
from statsmodels.stats.proportion import proportion_confint

def plot_rt_histogram(df, session_info, ax=None, side_col='trial_side', opto_col='is_opto', rt_col='RT', bins=50, color_map=None, label_map=None, show_plot=True):
    """
    Stacked response time histograms by (stim_side, opto) group.
    """
 
    if color_map is None:
        color_map = {
            ('left', 0): '#1f78b4',   # Dark Blue
            ('left', 1): '#00bfff',   # Neon Blue
            ('right', 0): '#e31a1c',  # Dark Red
            ('right', 1): '#ff6666',  # Neon Red
        }

    if label_map is None:
        label_map = {
            ('left', 0): 'Left Control',
            ('left', 1): 'Left Opto',
            ('right', 0): 'Right Control',
            ('right', 1): 'Right Opto',
        }  

    # filter out no lick
    df = df[df['lick'] != 0]
    # filter out naive
    df = df[df['naive'] == 0]
    # filter out move single spout
    df = df[df['MoveCorrectSpout'] == 0]  

    group_keys = [('left', 0), ('left', 1), ('right', 0), ('right', 1)]
    n_rows = len(group_keys)

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(6, 1.8 * n_rows),
        sharex=True,
        gridspec_kw={'hspace': 0.15},  # 🔧 Less vertical space
        # constrained_layout=True
    )

    # Optional: Set global x-limits across all groups
    global_min = df[rt_col].min()
    global_max = df[rt_col].max()
    
    global_min = 0
    global_max = 1000
    xlim = (global_min, global_max)
    
    # Shared x-axis binning
    global_rt = df[rt_col].dropna()
    bin_edges = np.linspace(global_rt.min(), global_rt.max(), bins + 1)    

    for ax, (stim, opto) in zip(axes, group_keys):
        group_df = df[(df[side_col] == stim) & (df[opto_col] == opto)]
        rt_vals = group_df[rt_col].dropna()

        color = color_map.get((stim, opto), 'gray')
        label = f"{stim.capitalize()} {'Opto' if opto else 'Control'} (n={len(rt_vals)})"

        # ax.hist(rt_vals, bins=bins, color=color, edgecolor='black', alpha=0.6)
        ax.hist(rt_vals, bins=bin_edges, color=color, edgecolor='black', alpha=0.6)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title(label, fontsize=9, pad=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xlim)  # Consistent x-axis for comparison
        ax.tick_params(axis='both', labelsize=7)

    axes[-1].set_xlabel("Response Time (s)", fontsize=9)
    # fig.tight_layout(pad=0.5)
   
    if show_plot:
        plt.show()
    
    # fig.subplots_adjust(top=0.75)
    # fig.tight_layout(rect=[0, 0, 1, 0.95])  # reserve top 5% for title, like tiger
    subject = session_info['subject_name']
    session_date = session_info['SessionDate']    
    title = f"{subject}  {session_date}  Response Time Distribution"
    fig.suptitle(title, fontsize=12, y=0.98)
    # fig.suptitle(title, fontsize=12)

    # Reserve only the top 5% for the title
    # fig.tight_layout(rect=[0, 0, 1, 0.99])
    
    
    subject = session_info['subject_name']
    date = session_info['date']
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject + '\\' + date)
    output_dir = os.path.join('plots\\' + subject + '\\' + date)
    figure_id = f"{subject}_{date}_response_time_histogram"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches=None, pad_inches=0.1, dpi=300)
    plt.close(fig)       

    return out_path    
 
    

def interpolate_curve(x, y, num_points=1000):
    f = interp1d(x, y, kind='linear')  # or 'cubic' for smoothness
    x_new = np.linspace(x.min(), x.max(), num_points)
    y_new = f(x_new)
    return x_new, y_new

# designer line collection
def plot_kde_with_variable_alpha(ax, x, y, alphas, color='blue', linewidth=2):
    """
    Plots a line with segment-wise alpha blending using LineCollection.
    """
    # segments = [
    #     [[x[i], y[i]], [x[i+1], y[i+1]]]
    #     for i in range(len(x) - 1)
    # ]
    
    # interpolate to get smoother line
    rt_interp, p_correct_interp = interpolate_curve(x, y, num_points=1000)

    segments = [
        [[rt_interp[i], p_correct_interp[i]], [rt_interp[i+1], p_correct_interp[i+1]]]
        for i in range(len(rt_interp) - 1)
    ]

    f_alpha = interp1d(x, alphas, kind='linear')
    alphas_interp = f_alpha(rt_interp)
    
    # lc = LineCollection(segments, colors=[(*plt.colors.to_rgb(color), a) for a in alphas], linewidths=linewidth)
       
    lc = LineCollection(
        segments,
        colors=[(*to_rgb(color), float(np.clip(a, 0, 1))) for a in alphas_interp],
        linewidths=linewidth
    )
    ax.add_collection(lc)
 
 
 
def normalize_density_to_alpha(trial_density, alpha_range=(0.2, 1.0), power=6):
    """
    Maps trial density to alpha using custom power-law scaling.
    """
    max_td = np.max(trial_density)
    if max_td == 0:
        return np.full_like(trial_density, alpha_range[0])

    td_scaled = (trial_density / max_td) ** power
    alphas = alpha_range[0] + td_scaled * (alpha_range[1] - alpha_range[0])
    return np.clip(alphas, *alpha_range)
    
def plot_pcorrect_kde_by_group(    
    df,
    session_info,    
    ax=None,
    rt_col='RT',
    correct_col='mouse_correct',
    side_col='trial_side',
    opto_col='is_opto',
    grouping=['trial_side', 'is_opto'],
    bandwidth=0.1,
    color_map=None,
    label_map=None,
    resolution=200,
    alpha_range=(0.2, 1.0),
    return_density=True,
    show_plot=True,
    alpha_scaling='nonlinear-kompressor', # alpha_scaling='log' | 'sqrt' | 'linear' | 'nonlinear-kompressor'
    fade_factor=2
):
    """
    Plot P(correct) vs RT using weighted KDEs.
    Alpha of each curve is scaled by trial density at each point.
    
    Optionally returns density curves for reuse.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure        
    
    # filter out no lick
    df = df[df['lick'] != 0]
    # filter out naive
    df = df[df['naive'] == 0]
    # filter out move single spout
    df = df[df['MoveCorrectSpout'] == 0]        
    
    if grouping == 'all' or grouping is None:
        group_iter = [(('all',), df)]
    else:
        group_iter = df.groupby(grouping)

    if color_map is None:
        color_map = {
            ('all',):    '#000000',
            (0,): '#000000',
            (1,): '#1f77b4',            # green,  #7CFC00' - slime green
            ('left',): '#1f78b4',   # Dark Blue
            ('right', ): '#e31a1c',  # Dark Red            
            ('left', 0): '#1f78b4',   # Dark Blue
            ('left', 1): '#00bfff',   # Neon Blue
            ('right', 0): '#e31a1c',  # Dark Red
            ('right', 1): '#ff6666',  # Neon Red
        }

    if label_map is None:
        label_map = {
            ('all',): 'All Trials',
            (0,): 'Control',
            (1,): 'Opto',
            ('left',): 'Left Control',
            ('right',): 'Right Control',            
            ('left', 0): 'Left Control',
            ('left', 1): 'Left Opto',
            ('right', 0): 'Right Control',
            ('right', 1): 'Right Opto',
        } 
        
    # legend_label_map = None
    # if legend_label_map is None:
    #     legend_label_map = {
    #         ('all',): 'All Trials 1/Trials',
    #         ('left',): 'Left Control 1/Trials',
    #         ('right',): 'Right Control 1/Trials',            
    #         ('left', 0): 'Left Control 1/Trials',
    #         ('left', 1): 'Left Opto 1/Trials',
    #         ('right', 0): 'Right Control 1/Trials',
    #         ('right', 1): 'Right Opto 1/Trials',
    #     }         

    global_min = df[rt_col].min()
    global_max = df[rt_col].max()
    rt_range = np.linspace(global_min, global_max, resolution)

    all_density_curves = {} if return_density else None

    for group_key, group_df in group_iter:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        # label = " ".join(str(g).capitalize() for g in group_key if g != 'all')
        label = label_map.get(group_key, None)
        color = color_map.get(group_key, None)

        rts = group_df[rt_col].values
        corrects = group_df[correct_col].values

        if len(rts) < 10 or np.std(rts) == 0:
            continue  # Not enough variation to KDE

        try:
            # Filter out NaNs or infs from both RTs and correctness
            # dj valid
            valid_mask = (
                np.isfinite(rts) &
                np.isfinite(corrects)
            )
            
            rts = rts[valid_mask]
            corrects = corrects[valid_mask]
            
            if len(rts) < 10 or np.std(rts) == 0:
                continue  # Not enough clean data, or 
            
            # corrects are all zero -> kde divide by zero
            if np.all(corrects == 0):
                p_correct = np.zeros_like(rt_range)
            else:            
                kde_all = gaussian_kde(rts, bw_method=bandwidth)
                # kde_correct = gaussian_kde(rts, weights=corrects, bw_method=bandwidth)
                
                epsilon = 1e-8
                # p_correct = kde_correct(rt_range) / (kde_all(rt_range) + epsilon)
                # p_correct = np.clip(p_correct, 0, 1)                
            
        except np.linalg.LinAlgError:
            continue  # Degenerate case
        # except Exception as e:
        #     print("Exception:", e)        


        trial_density = kde_all(rt_range)

        # mask low density regions         
        mask = trial_density > 1e-4  # skip unstable low-density areas
        # p_correct[~mask] = np.nan

        if alpha_scaling == 'linear':
            # Normalize trial density to [0.2, 1.0] range for alpha scaling
            td_norm = (trial_density - trial_density.min()) / (trial_density.max() - trial_density.min())
            alphas = alpha_range[0] + td_norm * (alpha_range[1] - alpha_range[0])
        elif alpha_scaling == 'log':
            # Avoid log(0) by adding small constant
            log_td = np.log10(trial_density + 1e-6)        
            # Normalize log-scaled density to [0, 1]
            log_td_norm = (log_td - log_td.min()) / (log_td.max() - log_td.min())        
            # Scale to alpha range
            alphas = alpha_range[0] + log_td_norm * (alpha_range[1] - alpha_range[0])
        elif alpha_scaling == 'sqrt':
            td_scaled = trial_density ** 3.8  # or try 0.5 for sqrt
            td_norm = (td_scaled - td_scaled.min()) / (td_scaled.max() - td_scaled.min())
            alphas = alpha_range[0] + td_norm * (alpha_range[1] - alpha_range[0])
        elif alpha_scaling == 'nonlinear-kompressor':
            alphas = normalize_density_to_alpha(trial_density, alpha_range=(0.0, 1.0), power=fade_factor)
        else:
            for i in range(len(rt_range) - 1):
                ax.plot(
                    rt_range[i:i+2],
                    p_correct[i:i+2],
                    color=color,
                    alpha=alphas[i],
                    linewidth=2
                )
        # plot_kde_with_variable_alpha(ax, rt_range, p_correct, alphas, color=color, linewidth=3)

        # ax.plot([], [], color=color, label=f"{label} (n={len(rts)})")  # for legend

        if return_density:
            all_density_curves[group_key] = trial_density
            
        # Overlay binned P(correct) for validation
        n_bins = 20
        bin_counts, bin_edges = np.histogram(rts, bins=n_bins)
        bin_indices = np.digitize(rts, bin_edges) - 1  # index of each trial's bin
        
        bin_pcorrect = np.zeros(n_bins)
        for i in range(n_bins):
            in_bin = (bin_indices == i)
            if in_bin.sum() >= 5:  # Skip low-count bins
                bin_pcorrect[i] = corrects[in_bin].mean()
            else:
                bin_pcorrect[i] = np.nan
        
        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot binned P(correct)
        ax.plot(bin_centers, bin_pcorrect, marker='o', linestyle='-',
                label=f"{label} (n={len(rts)})", color=color, alpha=0.7, markersize=4)
        
        
        
        ci_lower = np.zeros(n_bins)
        ci_upper = np.zeros(n_bins)
        
        for i in range(n_bins):
            in_bin = (bin_indices == i)
            count = in_bin.sum()
            if count >= 5:
                correct_sum = corrects[in_bin].sum()
                ci_low, ci_upp = proportion_confint(correct_sum, count, alpha=0.05, method='wilson')
                ci_lower[i] = ci_low
                ci_upper[i] = ci_upp
            else:
                ci_lower[i] = np.nan
                ci_upper[i] = np.nan        
           
        # Compute errors
        yerr_lower = np.clip(bin_pcorrect - ci_lower, 0, None)
        yerr_upper = np.clip(ci_upper - bin_pcorrect, 0, None)                
           
        # Optional: mask bins with NaN
        valid_bins = ~np.isnan(bin_pcorrect) & ~np.isnan(yerr_lower) & ~np.isnan(yerr_upper)
        
        # ax.errorbar(
        #     bin_centers, bin_pcorrect,
        #     yerr=[yerr_lower, yerr_upper],
        #     fmt='o', color=color, alpha=0.7, capsize=3, label=f"{label} (binned)"
        # )     

        # ax.errorbar(
        #     bin_centers[valid_bins], bin_pcorrect[valid_bins],
        #     yerr=[yerr_lower[valid_bins], yerr_upper[valid_bins]],
        #     fmt='o', color=color, alpha=0.7, capsize=3, label=f"{label} (binned)"
        # )   

        ax.errorbar(
            bin_centers[valid_bins], bin_pcorrect[valid_bins],
            yerr=[yerr_lower[valid_bins], yerr_upper[valid_bins]],
            fmt='o', color=color, alpha=0.3, capsize=3,
        ) 


    ax2 = ax.twinx()
    ax2.set_ylabel('Trial Density (PDF)')
    for group_key, density in all_density_curves.items():
        label = label_map.get(group_key, None) + f' 1/Trials'
        color = color_map.get(group_key, None)
        ax2.plot(rt_range, density, label=label, alpha=0.3, linestyle='--', color=color)
        # ax2.plot(rt_range, density, label=" ".join(str(g) for g in group_key))

    ax.set_xlabel("Response Time (s)")
    ax.set_ylabel("P(Correct)")
    ax.set_ylim(0, 1.05)
    global_min = 0
    global_max = 1000
    ax.set_xlim(global_min, global_max)
    ax.set_title("P(Correct) vs RT (KDE)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Get handles and labels from ax2
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    # Add to ax1's legend
    ax.legend(handles=handles2, labels=labels2, loc='best')

    # Existing legend in ax
    handles1, labels1 = ax.get_legend_handles_labels()
    
    # Combine
    handles = handles1 + handles2
    labels = labels1 + labels2
    
    # Set combined legend
    ax.legend(handles, labels, loc='upper right')


    # if return_density:
    #     return all_density_curves


    subject = session_info['subject_name']
    date = session_info['SessionDate']
    title = f"{subject}  {date}  P(Correct) by Response Time"
    ax.set_title(title)

    if show_plot:
        plt.show()

    subject = session_info['subject_name']
    date = session_info['date']
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject + '\\' + date)
    output_dir = os.path.join('plots\\' + subject + '\\' + date)
    figure_id = f"{subject}_{date}_response_time_pcorrect_kde_by_group"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches=None, pad_inches=0.1, dpi=300)
    plt.close(fig)       

    return out_path    


def plot_rt_density_by_group(    
    df,
    session_info,    
    ax=None,
    rt_col='RT',
    correct_col='mouse_correct',
    side_col='trial_side',
    opto_col='is_opto',
    grouping=['trial_side', 'is_opto'],
    bandwidth=0.1,
    color_map=None,
    label_map=None,
    resolution=200,
    alpha_range=(0.2, 1.0),
    return_density=True,
    show_plot=True,
    alpha_scaling='nonlinear-kompressor', # alpha_scaling='log' | 'sqrt' | 'linear' | 'nonlinear-kompressor'
):
    """
    Plot P(correct) vs RT using weighted KDEs.
    Alpha of each curve is scaled by trial density at each point.
    
    Optionally returns density curves for reuse.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure        
    
    # filter out no lick
    df = df[df['lick'] != 0]
    # filter out naive
    df = df[df['naive'] == 0]
    # filter out move single spout
    df = df[df['MoveCorrectSpout'] == 0]        
    
    if grouping == 'all' or grouping is None:
        group_iter = [(('all',), df)]
    else:
        group_iter = df.groupby(grouping)

    if color_map is None:
        color_map = {
            ('all',):    '#000000',
            (0,): '#000000',
            (1,): '#1f77b4',            # green,  #7CFC00' - slime green
            ('left',): '#1f78b4',   # Dark Blue
            ('right', ): '#e31a1c',  # Dark Red            
            ('left', 0): '#1f78b4',   # Dark Blue
            ('left', 1): '#00bfff',   # Neon Blue
            ('right', 0): '#e31a1c',  # Dark Red
            ('right', 1): '#ff6666',  # Neon Red
        }

    if label_map is None:
        label_map = {
            ('all',): 'All Trials',
            (0,): 'Control',
            (1,): 'Opto',
            ('left',): 'Left Control',
            ('right',): 'Right Control',            
            ('left', 0): 'Left Control',
            ('left', 1): 'Left Opto',
            ('right', 0): 'Right Control',
            ('right', 1): 'Right Opto',
        } 
         
        
    # legend_label_map = None
    # if legend_label_map is None:
    #     legend_label_map = {
    #         ('all',): 'All Trials 1/Trials',
    #         ('left',): 'Left Control 1/Trials',
    #         ('right',): 'Right Control 1/Trials',            
    #         ('left', 0): 'Left Control 1/Trials',
    #         ('left', 1): 'Left Opto 1/Trials',
    #         ('right', 0): 'Right Control 1/Trials',
    #         ('right', 1): 'Right Opto 1/Trials',
    #     }         

    global_min = df[rt_col].min()
    global_max = df[rt_col].max()
    rt_range = np.linspace(global_min, global_max, resolution)

    all_density_curves = {} if return_density else None

    rts_list = []
    
    for group_key, group_df in group_iter:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        # label = " ".join(str(g).capitalize() for g in group_key if g != 'all')
        label = label_map.get(group_key, None)
        color = color_map.get(group_key, None)

        rts = group_df[rt_col].values
        corrects = group_df[correct_col].values

        if len(rts) < 10 or np.std(rts) == 0:
            continue  # Not enough variation to KDE

        try:
            # Filter out NaNs or infs from both RTs and correctness
            # dj valid
            valid_mask = (
                np.isfinite(rts) &
                np.isfinite(corrects)
            )
            
            rts = rts[valid_mask]
            rts_list.append(rts)
            corrects = corrects[valid_mask]
            
            if len(rts) < 10 or np.std(rts) == 0:
                continue  # Not enough clean data, or 
            
            # corrects are all zero -> kde divide by zero
            if np.all(corrects == 0):
                p_correct = np.zeros_like(rt_range)
                
                
                # for group_key, group_df in group_iter:
                #     if not isinstance(group_key, tuple):
                #         group_key = (group_key,)

                #     # label = " ".join(str(g).capitalize() for g in group_key if g != 'all')
                #     label = label_map.get(group_key, None)
                #     color = color_map.get(group_key, None)

                #     rts = group_df[rt_col].values
                #     corrects = group_df[correct_col].values
                
                kde_all = gaussian_kde(rts, bw_method=bandwidth)
                
            else:            
                kde_all = gaussian_kde(rts, bw_method=bandwidth)
                kde_correct = gaussian_kde(rts, weights=corrects, bw_method=bandwidth)
                
                epsilon = 1e-8
                p_correct = kde_correct(rt_range) / (kde_all(rt_range) + epsilon)
                p_correct = np.clip(p_correct, 0, 1)                
            
        except np.linalg.LinAlgError:
            continue  # Degenerate case

        trial_density = kde_all(rt_range)

        # mask low density regions         
        mask = trial_density > 1e-4  # skip unstable low-density areas
        p_correct[~mask] = np.nan

        if alpha_scaling == 'linear':
            # Normalize trial density to [0.2, 1.0] range for alpha scaling
            td_norm = (trial_density - trial_density.min()) / (trial_density.max() - trial_density.min())
            alphas = alpha_range[0] + td_norm * (alpha_range[1] - alpha_range[0])
        elif alpha_scaling == 'log':
            # Avoid log(0) by adding small constant
            log_td = np.log10(trial_density + 1e-6)        
            # Normalize log-scaled density to [0, 1]
            log_td_norm = (log_td - log_td.min()) / (log_td.max() - log_td.min())        
            # Scale to alpha range
            alphas = alpha_range[0] + log_td_norm * (alpha_range[1] - alpha_range[0])
        elif alpha_scaling == 'sqrt':
            td_scaled = trial_density ** 3.8  # or try 0.5 for sqrt
            td_norm = (td_scaled - td_scaled.min()) / (td_scaled.max() - td_scaled.min())
            alphas = alpha_range[0] + td_norm * (alpha_range[1] - alpha_range[0])
        elif alpha_scaling == 'nonlinear-kompressor':
            alphas = normalize_density_to_alpha(trial_density, alpha_range=(0.0, 1.0), power=6)
        else:
            for i in range(len(rt_range) - 1):
                ax.plot(
                    rt_range[i:i+2],
                    p_correct[i:i+2],
                    color=color,
                    alpha=alphas[i],
                    linewidth=2
                )
        # plot_kde_with_variable_alpha(ax, rt_range, p_correct, alphas, color=color, linewidth=3)

        # ax.plot([], [], color=color, label=f"{label} (n={len(rts)})")  # for legend

        if return_density:
            all_density_curves[group_key] = trial_density

        label = label_map.get(group_key, None) + f' 1/Trials (n={len(rts)})'
        color = color_map.get(group_key, None)
        # if group_key == ('left', np.int64(0)):
        #     ax.plot(rt_range, trial_density, label=label, alpha=1, linestyle='-', color=color, linewidth=20)
        # else:
        ax.plot(rt_range, trial_density, label=label, alpha=1, linestyle='-', color=color)

    # ax2 = ax.twinx()
    # ax2.set_ylabel('Trial Density (PDF)')
    # for group_key, density in all_density_curves.items():
        
    #     label = label_map.get(group_key, None) + f' 1/Trials (n={len(rts)})'
    #     color = color_map.get(group_key, None)
    #     if group_key == ('left', np.int64(0)):
    #         ax.plot(rt_range, density, label=label, alpha=1, linestyle='-', color=color, linewidth=20)
    #     else:
    #         ax.plot(rt_range, density, label=label, alpha=1, linestyle='-', color=color)
        # ax2.plot(rt_range, density, label=" ".join(str(g) for g in group_key))

    ax.set_xlabel("Response Time (s)")
    ax.set_ylabel("PDF")
    # ax.set_ylim(0, 1.05)
    global_min = 0
    global_max = 1000
    ax.set_xlim(global_min, global_max)
    ax.set_title("P(Correct) vs RT (KDE)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Get handles and labels from ax2
    # handles2, labels2 = ax2.get_legend_handles_labels()
    
    # # Add to ax1's legend
    # ax.legend(handles=handles2, labels=labels2, loc='best')

    # # Existing legend in ax
    # handles1, labels1 = ax.get_legend_handles_labels()
    
    # # Combine
    # handles = handles1 + handles2
    # labels = labels1 + labels2
    
    # # Set combined legend
    # ax.legend(handles, labels, loc='best')


    # if return_density:
    #     return all_density_curves


    subject = session_info['subject_name']
    date = session_info['SessionDate']
    title = f"{subject}  {date}  Response Time Trial Density"
    ax.set_title(title)

    if show_plot:
        plt.show()

    subject = session_info['subject_name']
    date = session_info['date']
    # output_dir = os.path.join(config['paths']['figure_dir_local'] + subject + '\\' + date)
    output_dir = os.path.join('plots\\' + subject + '\\' + date)
    figure_id = f"{subject}_{date}_response_time_pcorrect_kde_by_group"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches=None, pad_inches=0.1, dpi=300)
    plt.close(fig)       

    return out_path    