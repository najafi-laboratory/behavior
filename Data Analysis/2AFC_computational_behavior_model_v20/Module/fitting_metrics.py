"""
Module/fitting_metrics.py

Contains objective functions, soft barrier penalties, bias extraction,
and post-hoc asymmetry loss logic used to guide the optimization process.
"""

import torch
import numpy as np
import pandas as pd
from constants import BARRIER_K, BARRIER_MARGIN, N_POST_SWITCH, TRANSITION_LOSS_WEIGHT

def compute_soft_barrier_penalty(
    params_vec: torch.Tensor,
    bounds: list,
    k: float = BARRIER_K,
    margin: float = BARRIER_MARGIN
) -> torch.Tensor:
    """
    Returns a differentiable penalty that spikes as parameters approach bounds.
    Anchored to params_vec.sum() * 0.0 to maintain the computational graph.
    """
    penalty = params_vec.sum() * 0.0
    eps = 1e-6
 
    for i, (lo, hi) in enumerate(bounds):
        p        = params_vec[i]
        rng      = hi - lo
        lo_inner = lo + margin * rng
        hi_inner = hi - margin * rng
 
        dist_lo = p - lo + eps
        dist_hi = hi - p + eps
 
        in_lo = (p < lo_inner).float()
        in_hi = (p > hi_inner).float()
 
        penalty = penalty + in_lo * k / dist_lo + in_hi * k / dist_hi
 
    return penalty

def compute_transition_asymmetry_loss(
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
    n_post_switch: int = N_POST_SWITCH
) -> float:
    """Computes MSE between model & mouse post-switch accuracy curves."""
    def _transition_curves(df, correct_col):
        blocks = df['block_type'].values
        correct = df[correct_col].values.astype(float)
 
        curves = {
            'short_to_long': np.full(n_post_switch, np.nan),
            'long_to_short': np.full(n_post_switch, np.nan),
        }
        accum = {
            'short_to_long': [[] for _ in range(n_post_switch)],
            'long_to_short': [[] for _ in range(n_post_switch)],
        }
 
        n = len(blocks)
        for t in range(1, n):
            prev, curr = blocks[t - 1], blocks[t]
            if prev == 'short_block' and curr == 'long_block':
                key = 'short_to_long'
            elif prev == 'long_block' and curr == 'short_block':
                key = 'long_to_short'
            else:
                continue 
 
            for offset in range(n_post_switch):
                idx = t + offset
                if idx >= n:
                    break
                if idx > t and blocks[idx] != curr:
                    break
                accum[key][offset].append(correct[idx])
 
        for key in curves:
            for pos in range(n_post_switch):
                vals = accum[key][pos]
                if len(vals) >= 2:            
                    curves[key][pos] = np.mean(vals)
 
        return curves
 
    sim_curves  = _transition_curves(sim_df,  'correct_model')
    real_curves = _transition_curves(real_df, 'rewarded')
 
    mse_total = 0.0
    n_valid   = 0
 
    for key in ['short_to_long', 'long_to_short']:
        s = sim_curves[key]
        r = real_curves[key]
        mask = ~np.isnan(s) & ~np.isnan(r)
        if mask.sum() > 0:
            mse_total += np.mean((s[mask] - r[mask]) ** 2)
            n_valid   += 1
 
    if n_valid == 0:
        return 0.0          
    return mse_total / n_valid

def select_best_restart(
    restart_results: list,
    real_df: pd.DataFrame,
    w_transition: float = TRANSITION_LOSS_WEIGHT
) -> dict:
    """Picks the best fitting restart using composite NLL + Asymmetry Loss."""
    n_trials = len(real_df)
    normaliser = n_trials / (2.0 * N_POST_SWITCH + 1e-6)
 
    best_score  = float('inf')
    best_result = restart_results[0]
 
    for res in restart_results:
        if not res['success']:
            continue
        t_loss = compute_transition_asymmetry_loss(res['sim_df'], real_df)
        composite = res['nll'] + w_transition * t_loss * normaliser
        res['transition_loss']  = t_loss
        res['composite_score']  = composite
        if composite < best_score:
            best_score  = composite
            best_result = res
 
    return best_result

def calculate_neutral_bias(df_session):
    """Calculates inherent baseline bias from the neutral blocks."""
    neutral_df = df_session[df_session['block_type'] == 'neutral']
    if len(neutral_df) < 10: return 0.0
    p_left = (neutral_df['mouse_choice'] == 'left').mean()
    p_left = np.clip(p_left, 0.01, 0.99)
    return float(-np.log(p_left / (1.0 - p_left)))

def get_session_tensors(df_session):
    """Converts a session DataFrame into PyTorch tensors for JIT speed."""
    isi = torch.tensor(df_session['isi'].values, dtype=torch.float32)
    is_short = torch.tensor((df_session['trial_type'] == 'short').values, dtype=torch.bool)
    mouse_ch_left = torch.tensor((df_session['mouse_choice'] == 'left').values, dtype=torch.bool)
    return isi, is_short, mouse_ch_left