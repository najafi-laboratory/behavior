# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 22:41:02 2025

@author: timst
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_state_transition_matrix(model, model_summary, mouse_id=None, session_date=None, figsize=(7, 6)):
    """
    Plots the state transition matrix using the trained GLM-HMM and descriptive model summary.

    Args:
        glm_hmm: Trained GLM-HMM model object.
        model_summary (dict): Output from summarize_glm_hmm_model().
        mouse_id (str): Optional mouse identifier for annotation.
        session_date (str): Optional session identifier for annotation.
        figsize (tuple): Size of the output figure.
    """
    print("📊 Plotting state transition matrix from model...")

    num_states = model_summary["num_states"]
    glm_hmm = model['glm_hmm']
    trans_matrix = glm_hmm.transitions.transition_matrix

    # Label states with IDs and summary labels
    state_labels = [f"{s['state_id']} – {s['label']}" for s in model_summary["states"]]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        trans_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=state_labels,
        yticklabels=state_labels,
        square=True,
        cbar_kws={"label": "Transition Probability"},
        linewidths=0.5,
        ax=ax
    )

    title = "State Transition Matrix"
    subtitle = " ".join(filter(None, [mouse_id, session_date]))
    ax.set_title(f"{title}\n{subtitle}", fontsize=14, pad=20)

    ax.set_xlabel("To State")
    ax.set_ylabel("From State")

    plt.tight_layout()
    plt.show()