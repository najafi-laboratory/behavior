# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 20:40:40 2025

@author: timst
"""
import matplotlib.pyplot as plt

REGION_BASE_COLOR = {
    'Control': '#000000',     # Always black
    'RLat':    '#d62728',     # red
    'LLat':    '#1f77b4',     # blue
    'RIntA':   '#d62728',     # red
    'LIntA':   '#1f77b4',     # blue
    'LPPC':    '#1f77b4',     # blue
    'RPPC':    '#d62728',     # red
    'mPFC':    '#2ca02c',     # green
    'LPost':   '#1f77b4',     # blue
    'RPost':   '#d62728',     # red
}

REGION_RESIDUAL_CMAP = {
    'RLat':  plt.cm.Reds,
    'LLat':  plt.cm.Blues,
    'RIntA': plt.cm.Reds,
    'LIntA': plt.cm.Blues,
    'LPPC':  plt.cm.Blues,
    'RPPC':  plt.cm.Reds,
    'mPFC':  plt.cm.Greens,
    'LPost': plt.cm.Blues,
    'RPost': plt.cm.Reds,
}

def get_plot_color(condition_label):
    """
    Returns a fixed color for psychometric curves based on condition label (e.g., 'Control', 'Opto LPPC').
    """
    if "Control" in condition_label:
        return REGION_BASE_COLOR['Control']
    for region, base_color in REGION_BASE_COLOR.items():
        if region in condition_label:
            return base_color
    return '#000000'  # fallback

def get_residual_colormap(region_abbrev):
    """
    Returns the colormap for residual effect plotting based on opto target region abbreviation.
    """
    return REGION_RESIDUAL_CMAP.get(region_abbrev, plt.cm.Greys)