# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 19:57:57 2025

@author: saminnaji3
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def cost_function(params, behavior, target_low, target_high):
    alpha_f, alpha_s, gain_f, gain_s, reward_rate = params
    x_f, x_s = 0, 0
    #reward_rate = 0.01
    pred = []
    for t in range(len(target_low)):
        if (gain_f * x_f + gain_s * x_s)>target_high[t]:
            err = target_high[t] - (gain_f * x_f + gain_s * x_s)
        elif (gain_f * x_f + gain_s * x_s)<target_low[t]:
            err = target_low[t] - (gain_f * x_f + gain_s * x_s)
        elif (gain_f * x_f + gain_s * x_s)>target_low[t] and (gain_f * x_f + gain_s * x_s)<target_high[t]:
            err = reward_rate*(target_low[t] - (gain_f * x_f + gain_s * x_s))
        x_f = (1 - alpha_f) * x_f + alpha_f * err
        x_s = (1 - alpha_s) * x_s + alpha_s * err
        pred.append(gain_f * x_f + gain_s * x_s)
    return np.nansum((behavior - np.array(pred))**2)
def simulate_model(params, target_low, target_high):
    alpha_f, alpha_s, gain_f, gain_s, reward_rate = params
    x_f, x_s = 0, 0
    #reward_rate = 0.01
    model_output = []
    
    for t in range(len(target_low)):
        y_pred = gain_f * x_f + gain_s * x_s
        if (gain_f * x_f + gain_s * x_s)>target_high[t]:
            err = target_high[t] - (gain_f * x_f + gain_s * x_s)
        elif (gain_f * x_f + gain_s * x_s)<target_low[t]:
            err = target_low[t] - (gain_f * x_f + gain_s * x_s)
        elif (gain_f * x_f + gain_s * x_s)>target_low[t] and (gain_f * x_f + gain_s * x_s)<target_high[t]:
            err = reward_rate*(target_low[t] - (gain_f * x_f + gain_s * x_s))
        x_f = (1 - alpha_f) * x_f + alpha_f * err
        x_s = (1 - alpha_s) * x_s + alpha_s * err
        model_output.append(y_pred)
    
    return np.array(model_output)

def plot_modeling(axs, y_model, y_data, i, num_sessions):
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=0, vmax=num_sessions+1)
    color_tag = cmap(norm(num_sessions+1-i))
    axs.plot(y_data, label='Behavior', linewidth=1, color = color_tag)
    axs.plot(y_model, label='Model Prediction', linewidth=1, linestyle='--', color = color_tag)
    axs.set_xlabel('Trial')
    axs.set_ylim([0, 2])
    axs.set_ylabel('Output')
    axs.set_title('Model Fit to Behavior')
