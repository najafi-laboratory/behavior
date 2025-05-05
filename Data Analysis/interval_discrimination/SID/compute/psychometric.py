# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 16:22:09 2025

@author: timst
"""
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit, OptimizeWarning
from scipy.special import expit  # logistic function
import warnings

def logistic(x, beta0, beta1):
    return expit(beta0 + beta1 * x)

def compute_psychometric_fit(
    psychometric_df,
    x_col='stim_value',
    y_col='p_right',
    n_points=200,
    clip_eps=1e-3,
    bounds=([-10, -20], [10, 20]),
    fallback=True,
    verbose=False, 
    extend_fit_x=False, 
    fit_margin=0.8
):
    """
    Fit a logistic sigmoid to psychometric data.

    Returns dict with:
        - params: [β0, β1]
        - threshold: -β0/β1
        - slope: β1
        - fit_x, fit_y: smooth sigmoid
        - fit_method: 'logistic' or 'linear2pt'
        - fit_quality: 'good', 'unstable', or 'failed'
    """
    # x = psychometric_df[x_col].values
    # y = np.clip(psychometric_df[y_col].values, clip_eps, 1 - clip_eps)  # avoid exact 0/1

    
    # Center x
    x_raw = psychometric_df[x_col].values
    x_mean = np.mean(x_raw)
    x_centered = x_raw - x_mean  # center around 0 for better β0/β1 convergence    
    x = x_centered
    fit_x = x_centered
    
    # clip y
    y = np.clip(psychometric_df[y_col].values, clip_eps, 1 - clip_eps)
    
    # Good initial guess
    beta1_init = 10 / (x_centered.max() - x_centered.min())
    beta0_init = 0  # centered around threshold at x=0
    p0 = [beta0_init, beta1_init]    
    
    bounds = ([-np.inf, -np.inf], [np.inf, np.inf])  # for testing

    if len(np.unique(x)) < 2:
        if verbose:
            print("⚠️ Not enough unique x-values to fit a curve.")
        return _fit_failed_output(fit_x)

    try:
        beta1_init = 10 / (fit_x.max() - fit_x.min())  # gentle slope
        
        p0 = [beta0_init, beta1_init]
        
        
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=OptimizeWarning)
            popt, _ = curve_fit(
                logistic, x_centered, y, p0=p0,
                bounds=bounds, maxfev=5000
            )
        beta0, beta1 = popt
        slope = beta1
        threshold = -beta0 / beta1 + x_mean  # shift back to original domain
        
        x_min, x_max = x_centered.min(), x_centered.max()
        x_range = x_max - x_min
        
        if extend_fit_x:
            fit_x_min = x_min - fit_margin * x_range
            fit_x_max = x_max + fit_margin * x_range
        else:
            fit_x_min = x_min
            fit_x_max = x_max        
        
        fit_x_centered = np.linspace(fit_x_min, fit_x_max, 200)
        fit_y = logistic(fit_x_centered, beta0, beta1)
        fit_x = fit_x_centered + x_mean  # un-center for plotting            

        if verbose:
            print(np.unique(y, return_counts=True))        
            print(f"Fit range: {fit_y.min():.2f}–{fit_y.max():.2f} | Δ={np.ptp(fit_y):.2f}")
            print(f"β0 = {beta0:.3f}, β1 = {beta1:.3f}, threshold = {-beta0 / beta1:.3f}")
            print(f"x range: {x.min():.2f}–{x.max():.2f}")

        # check fit stability
        if np.ptp(fit_y) < 0.05:  # if sigmoid rises < 5% total range
            raise RuntimeError("Fit output too flat to be useful")

        return {
            'params': (beta0, beta1),
            'threshold': threshold,
            'slope': slope,
            'fit_x': fit_x,
            'fit_y': fit_y,
            'fit_method': 'logistic',
            'fit_quality': 'good'
        }

    except Exception as e:
        if verbose:
            print(f"⚠️ Logistic fit failed: {e}")
        if fallback and len(x) == 2:
            # fallback to 2-point interpolated sigmoid
            x1, x2 = x
            y1, y2 = y
            try:
                threshold = np.interp(0.5, [y1, y2], [x1, x2])
                slope = (y2 - y1) / (x2 - x1)
                beta1 = max(min(slope * 10, 20), -20)  # moderate steepness
                beta0 = -beta1 * threshold
                fit_y = logistic(fit_x, beta0, beta1)
                return {
                    'params': (beta0, beta1),
                    'threshold': threshold,
                    'slope': beta1,
                    'fit_x': fit_x,
                    'fit_y': fit_y,
                    'fit_method': 'linear2pt',
                    'fit_quality': 'unstable'
                }
            except Exception as interp_error:
                if verbose:
                    print(f"⚠️ Fallback 2-point interpolation failed: {interp_error}")
                return _fit_failed_output(fit_x)
        else:
            return _fit_failed_output(fit_x)

def _fit_failed_output(fit_x):
    return {
        'params': (np.nan, np.nan),
        'threshold': np.nan,
        'slope': np.nan,
        'fit_x': fit_x,
        'fit_y': np.full_like(fit_x, np.nan),
        'fit_method': 'failed',
        'fit_quality': 'failed'
    }


def compute_psychometric(
    df,
    stim_col='isi',
    choice_col='mouse_choice',
    condition_col='is_opto',
    side_of_interest='right',
    binning='quantile',          #'discrete' 'or 'quantile'
    bins=8,                      # number of bins if binning='quantile'
    round_stim=2,                # decimal rounding for bin keys    
    dropna=True
):
    """
    Compute psychometric data: P(right) vs stimulus value, optionally split by condition.

    Returns:
        pd.DataFrame with columns:
        - stim_value (bin center)
        - p_right
        - stderr
        - n_trials
        - condition (if condition_col provided)
    """
    # if dropna:
    #     df = df.dropna(subset=[stim_col, choice_col])
    df = df.copy()
    
    # filter out no lick
    df = df[df['lick'] != 0]
    # filter out naive
    df = df[df['naive'] == 0]
    # filter out move single spout
    df = df[df['MoveCorrectSpout'] == 0]

    # Prepare choice indicator
    df['is_right_choice'] = (df[choice_col] == side_of_interest).astype(int)

    # Binning strategy
    if binning == 'discrete':
        df['stim_value'] = df[stim_col].round(round_stim)
        df['stim_bin'] = pd.Categorical(df['stim_value'], categories=sorted(df['stim_value'].unique()), ordered=True)
    # elif binning == 'quantile':
    #     bin_edges = np.unique(np.quantile(df[stim_col], np.linspace(0, 1, bins)))
    #     if len(bin_edges) < 2:
    #         raise ValueError("Not enough unique stimulus values for quantile binning.")
    #     df['stim_bin'] = pd.cut(df[stim_col], bins=bin_edges, include_lowest=True)
    #     df['stim_value'] = df['stim_bin'].apply(lambda b: b.mid)
    
    elif binning == 'quantile':
        raw_values = df[stim_col].values
        quantiles = np.linspace(0, 1, bins)
    
        # Use quantiles, but ensure unique bin edges
        bin_edges = np.unique(np.quantile(raw_values, quantiles))
    
        # If too few edges (e.g. 2 unique values give only 2 edges), force 3-point edge around them
        if len(bin_edges) < 3 and len(np.unique(raw_values)) == 2:
            v1, v2 = sorted(np.unique(raw_values))
            # Create midpoint to split them
            mid = (v1 + v2) / 2.0
            bin_edges = [v1 - 1, mid, v2 + 1]  # Pad slightly outside to ensure inclusion
        elif len(bin_edges) < 2:
            raise ValueError("Not enough unique stimulus values to perform quantile binning.")
    
        # Apply binning
        df['stim_bin'] = pd.cut(df[stim_col], bins=bin_edges, include_lowest=True)
    
        # Assign a representative value per bin
        df['stim_value'] = df['stim_bin'].apply(lambda b: b.mid if pd.notnull(b) else np.nan)    
    else:
        raise ValueError(f"Unsupported binning mode: {binning}")

    # Grouping
    group_cols = ['stim_value']
    if condition_col:
        group_cols.append(condition_col)

    grouped = df.groupby(group_cols, observed=True)

    # get isi mean as effective category boundary
    isi_mean = df[stim_col].mean()

    # Compute summary
    results = []
    for group_vals, subdf in grouped:
        if condition_col and not isinstance(group_vals, tuple):
            raise ValueError("Expected tuple (stim, condition) but got single value.")        
        
        if condition_col:
            stim_val, condition = group_vals
        else:
            stim_val = group_vals
            condition = 'all'

        n = len(subdf)
        p = subdf['is_right_choice'].mean()
        stderr = np.sqrt(p * (1 - p) / n) if n > 0 else np.nan

        if isinstance(stim_val, tuple) and len(stim_val) == 1:
            stim_val = stim_val[0]

        results.append({
            'stim_value': stim_val,
            'p_right': p,
            'stderr': stderr,
            'n_trials': n,
            'condition': condition if condition_col else 'all',
            'isi_mean': isi_mean,
        })

    return pd.DataFrame(results)

