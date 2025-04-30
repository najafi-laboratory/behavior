# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 20:13:18 2025

@author: timst
"""
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_weight_spikes_sorted(weights, feature_names, sort=True, state_id=0, session_date=None):
#     """
#     Plot GLM-HMM weights as unipolar action potential-like spikes.

#     Args:
#         weights (np.ndarray): Shape (D,), weights for the given state.
#         feature_names (list): Corresponding feature names.
#         sort (bool): If True, sort by magnitude.
#         state_id (int): Which state to label.
#         session_date (str): Optional date to include in title.
#     """
#     weights = weights.flatten()
#     abs_weights = np.abs(weights)

#     if sort:
#         sorted_indices = np.argsort(-abs_weights)
#         weights = weights[sorted_indices]
#         feature_names = [feature_names[i] for i in sorted_indices]

#     fig, ax = plt.subplots(figsize=(12, 4))
#     x = np.arange(len(weights))
#     height_scale = 1.5 * np.max(abs_weights)

#     for i, (w, name) in enumerate(zip(weights, feature_names)):
#         spike = np.array([0, 0.4 * w, w, 0.4 * w, 0]) / height_scale
#         spike_color = plt.cm.coolwarm(0.5 + 0.5 * np.sign(w))
#         ax.plot(x[i] + np.linspace(-0.4, 0.4, len(spike)), spike, lw=2, color=spike_color)

#     ax.axhline(0, color='black', lw=0.5, linestyle='--')
#     ax.set_xticks(x)
#     ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
#     ax.set_ylabel("Weight (normalized)", fontsize=12)
    
#     title = f"State {state_id} Weight Profile"
#     if session_date:
#         title += f" | Session {session_date}"
#     ax.set_title(title, fontsize=14)

#     plt.tight_layout()
#     plt.show()


def glm_interpret(M, config, subjectIdx, sessionIdx=-1, glm_hmm=None, model_results=None, session_data_by_date=None, all_sessions_df=None):

    



    return