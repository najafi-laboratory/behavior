# -*- coding: utf-8 -*-
"""
Created on Thu May  1 14:43:53 2025

@author: timst
"""
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

def compute_isi_pdf(df, isi_col='isi', group_col='trial_side', bandwidth='scott', n_points=200):
    """
    Compute ISI probability density for each group (e.g., 'left' vs 'right').
    
    Returns a dict: {group_name: {'x': x_vals, 'y': pdf_vals, 'mean': μ, 'count': N}}
    """
    result = {}

    
    # filter out no lick
    df = df[df['lick'] != 0]
    # filter out naive
    df = df[df['naive'] == 0]
    # filter out move single spout
    df = df[df['MoveCorrectSpout'] == 0]    
    
    groups = df[group_col].dropna().unique()    

    for group in groups:
        group_data = df[df[group_col] == group][isi_col].dropna().values
    
        if len(group_data) < 2 or np.std(group_data) == 0 or np.allclose(group_data, group_data[0]):
            # Degenerate case: single value or no variance
            result[group] = {
                'x': np.array([group_data[0]]) if len(group_data) > 0 else np.array([np.nan]),
                'y': np.array([1.0]),
                'mean': group_data[0] if len(group_data) > 0 else np.nan,
                'count': len(group_data),
                'fit_failed': True
            }
            continue

        try:
            kde = gaussian_kde(group_data, bw_method=bandwidth)
            x_vals = np.linspace(group_data.min(), group_data.max(), n_points)
            y_vals = kde(x_vals)

            result[group] = {
                'x': x_vals,
                'y': y_vals,
                'mean': group_data.mean(),
                'count': len(group_data),
                'fit_failed': False
            }

        except Exception as e:
            result[group] = {
                'x': np.array([group_data[0]]) if len(group_data) > 0 else np.array([np.nan]),
                'y': np.array([1.0]),
                'mean': group_data[0] if len(group_data) > 0 else np.nan,
                'count': len(group_data),
                'fit_failed': True
            }

    return result        
    #     group_data = df[df[group_col] == group][isi_col].dropna().values
    #     if len(group_data) < 2:
    #         continue  # Not enough data to compute KDE
        
    #     kde = gaussian_kde(group_data, bw_method=bandwidth)
    #     x_vals = np.linspace(group_data.min(), group_data.max(), n_points)
    #     y_vals = kde(x_vals)

    #     result[group] = {
    #         'x': x_vals,
    #         'y': y_vals,
    #         'mean': group_data.mean(),
    #         'count': len(group_data)
    #     }

    # return result

def compute_isi_pdf_grouped(df, isi_col='isi', 
                            group_cols=['trial_side', 'is_opto'], 
                            bandwidth='scott', n_points=200):
    """
    Compute ISI PDF for each combination of group_cols (e.g., trial_side + condition).
    Returns dict of {(group1, group2): {'x': ..., 'y': ..., ...}} entries.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    result = {}
    
    # filter out no lick
    df = df[df['lick'] != 0]
    # filter out naive
    df = df[df['naive'] == 0]
    # filter out move single spout
    df = df[df['MoveCorrectSpout'] == 0]        
    
    grouped = df.dropna(subset=group_cols + [isi_col]).groupby(group_cols)
    
    

    for group_vals, group_df in grouped:
        key = group_vals if isinstance(group_vals, tuple) else (group_vals,)
        x_data = group_df[isi_col].dropna().values

        # if len(x_data) < 2:
        #     continue  # not enough to compute PDF

        std = np.std(x_data)
        if len(x_data) < 2 or std == 0 or np.allclose(x_data, x_data[0]):
            # Not enough variance to compute KDE
            result[key] = {
                'x': np.array([x_data[0]]),
                'y': np.array([1.0]),
                'mean': x_data[0],
                'count': len(x_data),
                'fit_failed': True
            }
            continue
            
            

        kde = gaussian_kde(x_data, bw_method=bandwidth)
        x_vals = np.linspace(x_data.min(), x_data.max(), n_points)
        y_vals = kde(x_vals)

        result[key] = {
            'x': x_vals,
            'y': y_vals,
            'mean': x_data.mean(),
            'count': len(x_data),
        }

    return result
