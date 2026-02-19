import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def logistic_function(x, L, k, x0):
    """
    Logistic function for psychometric curve fitting.
    Args:
        x (np.ndarray): Input values (e.g., ISI).
        L (float): Curve's maximum value.
        k (float): Steepness of the curve.
        x0 (float): Midpoint of the curve.
    Returns:
        np.ndarray: Output values of the logistic function.
    """
    return L / (1 + np.exp(-k * (x - x0)))

def plot_psych_data_on_ax(ax, df, choice_col, label, color, bins):
    """
    Plot psychometric data and fit on a given axis.
    Args:
        ax (matplotlib.axes.Axes): Axis to plot on.
        df (pd.DataFrame): DataFrame containing trial data.
        choice_col (str): Column name for mouse/model choices.
        label (str): Label for the data (e.g., 'Mouse' or 'Model').
        color (str): Color for the plot.
        bins (np.ndarray): Bins for ISI values.
    Returns:
        None
    """
    if df.empty:
        return
    centers = (bins[:-1] + bins[1:]) / 2
    probs, sems = [], []

    for i in range(len(bins) - 1):
        sub = df[(df['isi'] >= bins[i]) & (df['isi'] < bins[i + 1])]
        if sub.empty:
            probs.append(np.nan)
            sems.append(np.nan)
            continue
        p = (sub[choice_col] == 'right').mean()
        n = len(sub)
        probs.append(p)
        sems.append(np.sqrt(p * (1 - p) / n) if n else 0)

    probs = np.array(probs)
    sems = np.array(sems)
    ok = ~np.isnan(probs)

    ax.errorbar(centers[ok], probs[ok], yerr=sems[ok],
                fmt='o', color=color, label=f'{label} Data',
                capsize=5, markersize=8, alpha=0.7)

    try:
        p0 = [1.0, 1.0, 1.25]
        bounds_fit = ([0.5, -10, 0.0], [1.0, 10, 2.5])
        popt, _ = curve_fit(logistic_function, centers[ok], probs[ok],
                            p0=p0, bounds=bounds_fit, maxfev=5000)
        xfit = np.linspace(bins.min(), bins.max(), 200)
        ax.plot(xfit, logistic_function(xfit, *popt), '-', color=color,
                linewidth=2, label=f'{label} Fit')
    except Exception as e:
        print(f'Fit failed for {label}: {e}')


def plot_psychometric_comparison(df_exp, df_sim, boundary,save_path):
    """
    Plot psychometric curves comparing experimental data and model simulation.
    Args:
        df_exp (pd.DataFrame): DataFrame with experimental trial data.
        df_sim (pd.DataFrame): DataFrame with simulated trial data.
        boundary (float): Reference boundary value to plot.
    Returns:
        None
    """
    fig, axs = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    fig.suptitle('Model vs. Mouse Psychometric Curves (Optimized Beliefs)', fontsize=18)
    bins = np.linspace(0, 2.5, 21)

    plot_psych_data_on_ax(axs[0], df_exp, 'mouse_choice', 'Mouse', 'black', bins)
    plot_psych_data_on_ax(axs[0], df_sim, 'model_choice', 'Model', 'red', bins)
    axs[0].set_title('All Blocks')

    plot_psych_data_on_ax(axs[1], df_exp[df_exp['block_type'] == 'short_block'],
                          'mouse_choice', 'Mouse', 'black', bins)
    plot_psych_data_on_ax(axs[1], df_sim[df_sim['block_type'] == 'short_block'],
                          'model_choice', 'Model', 'red', bins)
    axs[1].set_title('Short Block Context')

    plot_psych_data_on_ax(axs[2], df_exp[df_exp['block_type'] == 'long_block'],
                          'mouse_choice', 'Mouse', 'black', bins)
    plot_psych_data_on_ax(axs[2], df_sim[df_sim['block_type'] == 'long_block'],
                          'model_choice', 'Model', 'red', bins)
    axs[2].set_title('Long Block Context')

    for ax in axs:
        ax.set_xlabel('Inter-Stimulus Interval (s)')
        ax.axvline(boundary, color='gray', linestyle=':', label=f'Ref Boundary {boundary:.2f}s')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, ls='--', alpha=0.6)
        ax.set_ylim(-0.05, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0].set_ylabel('P(choose right)')
    handles, labels = axs[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc='upper right', bbox_to_anchor=(0.98, 0.85))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        saving = os.path.join(save_path, "psychometric_comparison.pdf")
        plt.savefig(saving, dpi=300)

    plt.close()