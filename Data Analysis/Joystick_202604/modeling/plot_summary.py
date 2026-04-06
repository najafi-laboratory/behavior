# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 19:57:49 2025

@author: saminnaji3
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 09:44:21 2025

@author: saminnaji3
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fitz
def lay_out(axs):
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)

def _rolling(x, w=25):
    x = np.asarray(x, float)
    mask = ~np.isnan(x)
    y = np.full_like(x, np.nan, float)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        m = mask[lo:i+1]
        if m.any():
            y[i] = np.nanmean(x[lo:i+1][m])
    return y

def plot_rolling_accuracy(axs, res, window=25, title="Rolling accuracy"):
    """Rolling mean of rewards (0/1) across trials."""
    acc = _rolling(res, w=window)
    axs.plot(acc)
    axs.set_xlabel("Trial")
    axs.set_ylabel(f"Accuracy (rolling {window})")
    axs.set_title(title)
    lay_out(axs)

def plot_interval_trace(axs, res, blocks, windows, title="Intervals vs tolerance"):
    """Intervals (actions) with block identity; show tolerance bands."""
    a = res
    axs.plot(a, linewidth=1)
    # overlay bands per block
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        L, U = windows[int(b)]
        axs.fill_between(idx, L, U, alpha=0.2, step=None)
    axs.set_xlabel("Trial")
    axs.set_ylabel("Interval (s)")
    axs.set_title(title)
    lay_out(axs)
    
def plot_predicted_success(axs, res, window=25, title="Predicted success (p̂)"):
    """Rolling mean of model’s p̂(y=1 | φ,a)."""
    p_roll = _rolling(res, w=window)
    axs.plot(p_roll)
    axs.set_xlabel("Trial")
    axs.set_ylabel(f"Predicted p(success) (rolling {window})")
    axs.set_title(title)
    lay_out(axs)

def plot_calibration(axs, probs, rewards, n_bins=10, title="Calibration: predicted vs observed"):
    """Reliability curve: bin p̂ and compare to empirical success."""
    p = probs.copy()
    y = rewards.copy()
    m = ~np.isnan(p) & ~np.isnan(y)
    p, y = p[m], y[m]
    if len(p) == 0:
        print("No valid points for calibration."); return
    bins = np.quantile(p, np.linspace(0,1,n_bins+1))
    bins[0], bins[-1] = 0.0, 1.0
    digit = np.digitize(p, bins) - 1
    bin_p, bin_y = [], []
    for b in range(n_bins):
        sel = digit == b
        if np.any(sel):
            bin_p.append(np.mean(p[sel]))
            bin_y.append(np.mean(y[sel]))
    axs.plot([0,1], [0,1], linestyle="--")
    axs.plot(bin_p, bin_y, marker="o")
    axs.set_xlabel("Mean predicted p̂ in bin")
    axs.set_ylabel("Observed success")
    axs.set_title(title)
    lay_out(axs)

def plot_psychometric(axs, actions, rewards, blocks, n_bins=12, title="Psychometric: success vs interval"):
    """Empirical P(success) as function of chosen interval, per block."""
    a = actions
    y = rewards
    mask = ~np.isnan(a) & ~np.isnan(y)
    a, y, blocks = a[mask], y[mask], blocks[mask]
    if len(a) == 0:
        print("No valid points for psychometric."); return
    a_min, a_max = np.nanmin(a), np.nanmax(a)
    edges = np.linspace(a_min, a_max, n_bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    for b in np.unique(blocks):
        sel_b = blocks == b
        p_bin = []
        for i in range(n_bins):
            sel = sel_b & (a >= edges[i]) & (a < edges[i+1])
            if np.any(sel):
                p_bin.append(np.mean(y[sel]))
            else:
                p_bin.append(np.nan)
        axs.plot(centers, p_bin, marker="o")
    axs.set_xlabel("Interval (s)")
    axs.set_ylabel("P(success)")
    axs.set_title(title)
    lay_out(axs)

def plot_blockwise_accuracy(axs, res, blocks, title="Block-wise accuracy"):
    """Bar plot of mean reward per block."""
    means = []
    labels = []
    for b in np.unique(blocks):
        sel = (blocks == b) & ~np.isnan(res)
        means.append(np.mean(res[sel]) if np.any(sel) else np.nan)
        labels.append(f"Block {int(b)}")
    axs.bar(labels, means)
    axs.set_ylabel("Mean accuracy")
    axs.set_title(title)
    lay_out(axs)

def save_temp_fig(fig, report):
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    temp_fig = fitz.open(fname)
    report.insert_pdf(temp_fig)
    temp_fig.close()
    os.remove(fname)
    

def plot_weight_norm(axs, res, title="Weight norm over trials"):
    """L2 norm of parameter vector (learning dynamics / stability)."""
    wnorm = np.linalg.norm(res, axis=1)
    axs.plot(wnorm, linewidth = 2, c = 'k')
    for i in range(res.shape[1]):
        axs.plot(res[:,i])
    axs.set_xlabel("Trial")
    axs.set_ylabel("||w||₂")
    axs.set_title(title)
    lay_out(axs)

def run_plot(axs, res, blocks, windows):
    # suppose you already ran:
    # res = results[0]  # a TrainResult for one session
    # blocks = your_blocks_array  # shape (T,)
    # windows = {0:(L0,U0), 1:(L1,U1)}
    
    plot_rolling_accuracy(axs[0], res, window=10)
    plot_interval_trace(axs[1], res, blocks, windows)
    plot_predicted_success(axs[2], res, window=10)
    plot_calibration(axs[3], res, n_bins=10)
    plot_psychometric(axs[4], res, blocks, n_bins=12)
    plot_blockwise_accuracy(axs[5], res, blocks)
    plot_weight_norm(axs[6], res)
def run_plot_summary(axs, res, blocks, windows):
    # suppose you already ran:
    # res = results[0]  # a TrainResult for one session
    # blocks = your_blocks_array  # shape (T,)
    # windows = {0:(L0,U0), 1:(L1,U1)}
    
    plot_rolling_accuracy(axs[0], res, window=25)
    plot_interval_trace(axs[1], res, blocks, windows)
    plot_predicted_success(axs[2], res, window=25)
    plot_calibration(axs[3], res, n_bins=10)
    plot_psychometric(axs[4], res, blocks, n_bins=12)
    #plot_blockwise_accuracy(axs[5], res, blocks)
    plot_weight_norm(axs[6], res)
    
def run(sessions_input, windows, results, output_dir_onedrive):
    report = fitz.open()
    
    # performance
    print('Plotting General Performance')
    #fig = plt.figure(layout='constrained', figsize=(30, 15))
    #gs = GridSpec(2, 4, figure=fig)
    for i in range(len(results)):
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(2, 4, figure=fig)
        blocks = sessions_input[i]['blocks']
        axs_all = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), plt.subplot(gs[0, 3]),
                   plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])]
        run_plot(axs_all, results[i], blocks, windows)
        plt.suptitle('modelling results')
        save_temp_fig(fig, report)
    
    report.save(output_dir_onedrive+ 'modeling_test_indivi.pdf')
    report.close()
    
def concat(results):
    actions = []
    probs = []
    rewards = []
    weights = []
    num_trials = []
    for i in range(len(results)):
        sess = results[i]
        actions.append(sess.actions)
        probs.append(sess.probs)
        rewards.append(sess.rewards)
        weights.append(sess.weights)
        num_trials.append(len(actions[0]))
    actions_all = np.concatenate(actions)
    probs_all = np.concatenate(probs)
    rewards_all = np.concatenate(rewards)
    weights_all = np.concatenate(weights, axis = 0)
                   
    return [actions_all, probs_all, rewards_all, weights_all, num_trials]
def run_all(sessions_input, windows, results, output_dir_onedrive):
    report = fitz.open()
    
    # performance
    print('Plotting General Performance')
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(2, 4, figure=fig)
    for i in range(len(results)):
        blocks = sessions_input[i]['blocks']
        axs_all = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), plt.subplot(gs[0, 3]),
                   plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])]
        run_plot_summary(axs_all, results[i], blocks, windows)
    plt.suptitle('modelling results')
    save_temp_fig(fig, report)
    
    report.save(output_dir_onedrive+ 'modeling_test_summary1.pdf')
    report.close()
    
def run_summary(sessions_input, windows, results, output_dir_onedrive):
    report = fitz.open()
    
    # performance
    print('Plotting General Performance')
    #fig = plt.figure(layout='constrained', figsize=(30, 15))
    #gs = GridSpec(2, 4, figure=fig)
    [actions_all, probs_all, rewards_all, weights_all, num_trials] = concat(results)
    
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(2, 3, figure=fig)
    blocks = sessions_input[0]['blocks']
    axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), plt.subplot(gs[1,0]),
               plt.subplot(gs[1, 1])]
    
    plot_rolling_accuracy(axs[0], rewards_all, window=25)
    plot_interval_trace(axs[1], actions_all, blocks, windows)
    plot_predicted_success(axs[2], probs_all, window=25)
    #plot_calibration(axs[3], res, n_bins=10)
    #plot_psychometric(axs[4], res, blocks, n_bins=12)
    #plot_blockwise_accuracy(axs[5], res, blocks)
    plot_weight_norm(axs[3], weights_all)
    num_trials = np.cumsum(num_trials)
    for i in num_trials:
        axs[0].axvline(i,color='r', linestyle='--', alpha=0.7)
    
    plt.suptitle('modelling results')
    save_temp_fig(fig, report)
    
    report.save(output_dir_onedrive+ 'modeling_test_summary_new.pdf')
    report.close()
