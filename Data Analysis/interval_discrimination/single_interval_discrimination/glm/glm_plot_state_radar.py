# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 21:54:58 2025

@author: timst
"""
import matplotlib.pyplot as plt
import numpy as np

# def plot_combined_behavioral_radar(profiles, state_labels, session_date=None, mouse_id=None):
#     """
#     Plots radar chart of behavioral profiles for all states overlaid on one plot.

#     Args:
#         profiles (list of dict): List of behavioral profiles for each state.
#         state_labels (list of str): Descriptive labels for each state.
#         session_date (str): Optional date for figure title.
#         mouse_id (str): Optional mouse identifier.
#     """
#     categories = list(profiles[0].keys())
#     N = len(categories)
#     angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
#     angles += angles[:1]

#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

#     colors = ['#1f77b4', '#2ca02c', '#d62728']  # blue, green, red for states 0, 1, 2

#     for i, profile in enumerate(profiles):
#         values = list(profile.values())
#         values += values[:1]
#         ax.plot(angles, values, color=colors[i], linewidth=2, label=f"State {i} - {state_labels[i]}")
#         ax.fill(angles, values, color=colors[i], alpha=0.15)

#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(categories, fontsize=10)
#     ax.set_yticklabels([])
#     ax.set_title(f"Behavioral Profiles Radar\n{mouse_id or ''} {session_date or ''}", fontsize=14, pad=20)
#     ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

#     plt.tight_layout()
#     plt.show()

def plot_combined_behavioral_radar(model_summary, session_date=None, mouse_id=None):
    """
    Plots a combined radar chart using the output from model_summary['states'].

    Args:
        model_summary (dict): Output from glm_hmm_summary, must contain 'states' key.
        session_date (str): Optional date to display in title.
        mouse_id (str): Optional mouse identifier.
    """
    print("📈 Generating radar plot for behavioral state profiles...")

    states = model_summary["states"]
    profiles = [s["profile"] for s in states]
    state_labels = [s["label"] for s in states]

    categories = list(profiles[0].keys())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ['#1f77b4', '#2ca02c', '#d62728']  # Customize as needed

    for i, profile in enumerate(profiles):
        values = list(profile.values())
        values += values[:1]  # Loop back to start
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=f"State {i} - {state_labels[i]}")
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title(f"Behavioral Profiles Radar\n{mouse_id or ''} {session_date or ''}", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.show()
