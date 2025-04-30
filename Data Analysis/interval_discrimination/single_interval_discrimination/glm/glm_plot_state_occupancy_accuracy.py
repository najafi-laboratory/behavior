# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 22:18:48 2025

@author: timst
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_state_occupancy_and_accuracy(model_summary, session_date=None, mouse_id=None):
    """
    Bar plot for state occupancy and per-state accuracy.
    
    Args:
        model_summary (dict): Output from glm_hmm_summary.
        session_date (str): Optional session date.
        mouse_id (str): Optional mouse name.
    """
    print("📊 Plotting state occupancy and accuracy...")

    states = model_summary["states"]
    num_states = len(states)

    occupancies = [s["occupancy"] for s in states]
    accuracies = [s["accuracy"] for s in states]
    labels = [f"State {s['state_id']}" for s in states]

    x = np.arange(num_states)
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left axis: Occupancy
    bar1 = ax1.bar(x - width/2, occupancies, width, label="Occupancy (%)", color="#1f77b4", alpha=0.8)
    ax1.set_ylabel("Occupancy (%)", color="#1f77b4")
    ax1.tick_params(axis='y', labelcolor="#1f77b4")
    ax1.set_ylim(0, max(occupancies) * 1.2)

    # Right axis: Accuracy
    ax2 = ax1.twinx()
    bar2 = ax2.bar(x + width/2, accuracies, width, label="Accuracy", color="#d62728", alpha=0.6)
    ax2.set_ylabel("Prediction Accuracy", color="#d62728")
    ax2.tick_params(axis='y', labelcolor="#d62728")
    ax2.set_ylim(0, 1.0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    title = f"State Occupancy and Accuracy\n{mouse_id or ''} {session_date or ''}"
    ax1.set_title(title.strip(), fontsize=14)

    # Legend
    fig.legend([bar1, bar2], ["Occupancy", "Accuracy"], loc="upper right", bbox_to_anchor=(0.9, 0.9))

    plt.tight_layout()
    plt.show()

