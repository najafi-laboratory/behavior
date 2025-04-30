# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 21:11:12 2025

@author: timst
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_weight_spikes_sorted(weights, feature_cols, sort=True, state_id=None, session_date=None, figsize=(14, 4)):
    """
    Plot GLM weights as neural-style asymmetric spikes, sorted by magnitude if desired.
    
    Args:
        weights (array): The 1D weight vector (e.g., from model.observations.params[state_id, 0, :])
        feature_cols (list): Corresponding feature names
        sort (bool): Whether to sort by absolute weight magnitude
        state_id (int or None): For labeling the state
        session_date (str or None): For labeling the session
        figsize (tuple): Size of the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm

    weights = np.array(weights)
    if sort:
        sort_idx = np.argsort(np.abs(weights))[::-1]
        weights = weights[sort_idx]
        feature_cols = [feature_cols[i] for i in sort_idx]

    n = len(weights)
    spike_length = 40
    x_vals = []
    y_vals = []
    color_vals = []

    for i, w in enumerate(weights):
        x = np.linspace(0, 1, spike_length)
        spike = np.zeros_like(x)

        peak_idx = int(spike_length * 0.25)
        spike[:peak_idx] = np.linspace(0, 1, peak_idx)
        spike[peak_idx:] = np.exp(-5 * (x[peak_idx:] - x[peak_idx]))
        spike *= np.sign(w) * np.abs(w)

        x_vals.extend(x + i)
        y_vals.extend(spike)

        norm = (w - weights.min()) / (weights.max() - weights.min() + 1e-6)
        color_vals.append(cm.coolwarm(norm))

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n):
        start = i * spike_length
        end = (i + 1) * spike_length
        ax.plot(
            x_vals[start:end],
            y_vals[start:end],
            color=color_vals[i],
            linewidth=2
        )

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(feature_cols, rotation=45, ha='right')
    ax.set_xlim(0, n)
    ax.set_ylim(-1.2 * np.max(np.abs(weights)), 1.2 * np.max(np.abs(weights)))
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel("Weight Magnitude (Signed)")

    # Dynamic title with context
    title_parts = ["Feature Weights"]
    if state_id is not None:
        title_parts.append(f"State {state_id}")
    if session_date is not None:
        title_parts.append(f"Session {session_date}")
    if sort:
        title_parts.append("(Sorted)")
    ax.set_title(" — ".join(title_parts), fontsize=12)

    plt.tight_layout()
    plt.show()
    
def plot_all_state_weight_spikes(glm_hmm, feature_cols, session_date=None, sort=False):
    """
    Plot spike-style feature weights for all states as vertically stacked subplots.

    Args:
        glm_hmm: Trained GLM-HMM model.
        feature_cols: List of feature names.
        session_date: Optional session date for figure title.
        sort: Whether to sort features by weight magnitude.
    """
    num_states = glm_hmm.K
    D = len(feature_cols)

    if sort:
        # Sort by average absolute weight across all states
        avg_weights = np.mean(np.abs([glm_hmm.observations.params[k][0] for k in range(num_states)]), axis=0)
        sorted_indices = np.argsort(avg_weights)[::-1]
        sorted_features = [feature_cols[i] for i in sorted_indices]
    else:
        sorted_indices = np.arange(D)
        sorted_features = feature_cols

    fig, axs = plt.subplots(num_states, 1, figsize=(12, 2.2 * num_states), sharex=True)

    if num_states == 1:
        axs = [axs]  # ensure list-like

    x = np.arange(len(sorted_features))

    for k, ax in enumerate(axs):
        weights = glm_hmm.observations.params[k][0][sorted_indices]

        # for i, w in enumerate(weights):
        #     spike = np.zeros(7)
        #     spike[3] = w * 0.9  # main spike height
        #     shape = np.convolve(spike, [0.2, 0.6, 1.0, 0.6, 0.2], mode='same')

        #     color = "crimson" if w > 0 else "royalblue"
        #     ax.plot(np.linspace(i - 0.4, i + 0.4, len(shape)), shape, color=color)
        
        for i, w in enumerate(weights):
            # Generate an action-potential-like shape: sharp up and smooth decay
            amp = np.sign(w) * np.sqrt(abs(w)) * 1.0  # emphasize polarity, compress extreme values
            t = np.linspace(-1, 1, 21)
            shape = amp * np.exp(-5 * np.abs(t)) * (t >= 0)  # unipolar spike, rightward decay
            x_vals = np.linspace(i - 0.4, i + 0.6, len(t))  # adjust horizontal alignment

            color = "crimson" if w > 0 else "royalblue"
            ax.plot(x_vals, shape, color=color)
        

        ax.axhline(0, linestyle='--', color='gray', linewidth=0.5)
        ax.set_ylabel(f"State {k}", rotation=0, labelpad=30, fontsize=10, va="center")

    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(sorted_features, rotation=45, ha="right")
    axs[-1].set_xlabel("Features")

    fig.suptitle(f"GLM-HMM Feature Weights Across States", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()    
    
def plot_state_weights(model_output):
    glm_hmm = model_output['glm_hmm']
    feature_cols = model_output['feature_cols']
    
    plot_all_state_weight_spikes(glm_hmm, feature_cols, sort=False)    
    
    
    
    return
