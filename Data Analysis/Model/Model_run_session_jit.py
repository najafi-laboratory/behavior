import pandas as pd
import numpy as np
import torch

# --- JIT COMPILED LOOP (With Reparameterization & MC Averaging) ---
@torch.jit.script
def run_session_jit(decay, noise, a_rew, a_pun, a_unc, a_chr, a_chp, beta, lapse, 
                    p_switch, p_rare, 
                    alpha_bound,  
                    fixed_bias, start_phys_bound,
                    isi_ten, is_short_ten, ch_left_ten, n_samples: int):
    """
    Run a single session of the 2AFC task with MC-averaged decision variable.
    
    Computes negative log-likelihood while updating decision boundary, weights,
    and context beliefs based on trial outcomes.
    
    Args:
        decay: Temporal decay parameter for CDF transformation
        noise: Sensory noise scaling factor (inverse of precision)
        a_rew: Learning rate for correct trials
        a_pun: Learning rate for incorrect trials
        a_unc: Uncertainty modulation factor for learning rate
        a_chr: Bias update rate for correct trials
        a_chp: Bias update rate for incorrect trials
        beta: Inverse temperature (decision softmax parameter)
        lapse: Lapse rate (probability of random choice)
        p_switch: Probability of context switching per trial
        p_rare: Base rate of rare context state
        alpha_bound: Learning rate for decision boundary adjustment
        fixed_bias: Constant bias term added to decision variable
        start_phys_bound: Initial decision boundary value
        isi_ten: Inter-stimulus interval times [N]
        is_short_ten: Ground truth context (1=short, 0=long) [N]
        ch_left_ten: Choice made (1=left, 0=right) [N]
        n_samples: Number of MC samples for noise integration
        
    Returns:
        nll_total: Negative log-likelihood for the session
    """
    normalizer = 1.0
    p_common = 1.0 - p_rare
    
    # Initialize state
    w_time = torch.tensor(0.5)
    w_ctx = torch.tensor(0.5)
    p_short = torch.tensor(0.5)
    last_bias = torch.tensor(0.0)
    
    # Initialize Dynamic Boundary
    curr_phys_bound = start_phys_bound 
    
    nll_total = torch.tensor(0.0)
    eps_log = 1e-9
    
    N = isi_ten.size(0)
    
    for i in range(N):
        isi_t = isi_ten[i]
        is_short_t = is_short_ten[i]
        ch_left_t = ch_left_ten[i]
        
        # --- 1. Calculate Internal Boundary for THIS trial ---
        cdf_bound = 1.0 - (1.0 + decay * curr_phys_bound) * torch.exp(-decay * curr_phys_bound)
        internal_boundary = normalizer * cdf_bound
        
        # --- 2. Sensory Noise Reparameterization & Sampling ---
        # Expand inputs to size [n_samples]
        isi_exp = isi_t.expand(n_samples)
        isi_safe = torch.clamp(isi_exp, min=0.0)
        
        # Deterministic Mean & Std
        cdf_val = 1.0 - (1.0 + decay * isi_safe) * torch.exp(-decay * isi_safe)
        mean_perc = normalizer * cdf_val
        std_perc = mean_perc / noise
        
        # Sample Noise (epsilon ~ N(0,1))
        # This is where the reparameterization trick happens
        noise_vec = torch.randn_like(mean_perc)
        
        # Calculate Perceived Times (Vectorized)
        perceived_time_vec = mean_perc + noise_vec * torch.clamp(std_perc, min=1e-6)
        
        # --- 3. Decision Variable (Vectorized across samples) ---
        dist = perceived_time_vec - internal_boundary
        exp_arg = torch.clamp(-dist, -50.0, 50.0)
        sigmoid_val = 1.0 / (1.0 + torch.exp(exp_arg))
        time_evidence = (2.0 * sigmoid_val - 1.0)
        
        # Context is constant for all samples in this trial
        context_evidence = 0.5 - p_short
        
        # DV for each sample
        dv_time = w_time * time_evidence
        dv_ctx = w_ctx * context_evidence
        dv = dv_time + dv_ctx + last_bias + fixed_bias
        
        # Probability for each sample
        dv_scaled = torch.clamp(beta * dv, -50.0, 50.0)
        p_left_vec = 1.0 / (1.0 + torch.exp(dv_scaled))
        
        # --- 4. Monte Carlo Average for Probability ---
        # Average the PROBABILITIES, not the times/DVs, to get the expected trial probability
        avg_p = torch.mean(p_left_vec)
        p_left = avg_p * (1.0 - lapse) + 0.5 * lapse
        
        # --- 5. Accumulate NLL ---
        if ch_left_t:
            nll_total = nll_total - torch.log(p_left + eps_log)
        else:
            nll_total = nll_total - torch.log(1.0 - p_left + eps_log)
            
        # --- 6. Updates ---
        was_correct = (ch_left_t == is_short_t)
        
        # A. Update Decision Boundary (Error Driven)
        bound_shift = torch.tensor(0.0)
        if not was_correct:
            if is_short_t: # Was Short, Chose Long
                bound_shift = alpha_bound 
            else:          # Was Long, Chose Short
                bound_shift = -alpha_bound
        
        curr_phys_bound = curr_phys_bound + bound_shift
        curr_phys_bound = torch.clamp(curr_phys_bound, 0.5, 2.5)

        # B. Update Weights (Standard)
        unc = 4.0 * p_short * (1.0 - p_short)
        was_correct_f = float(was_correct) 
        alpha_base = was_correct_f * a_rew + (1.0 - was_correct_f) * a_pun
        alpha = alpha_base * (1.0 + a_unc * unc)
        
        # For weight updates, we use the MEAN perceived time as the "expected" percept
        t_accum = torch.mean(perceived_time_vec)
        is_perc_short = (t_accum < internal_boundary)
        
        time_ok = (is_perc_short == is_short_t)
        ctx_ok = ((p_short > 0.5) == is_short_t)
        dir_time = 2.0 * float(time_ok) - 1.0
        dir_ctx = 2.0 * float(ctx_ok) - 1.0
        
        w_time = w_time + alpha * dir_time
        w_ctx = w_ctx + alpha * dir_ctx
        if w_time < 0.0: w_time = torch.tensor(0.0)
        if w_ctx < 0.0: w_ctx = torch.tensor(0.0)
            
        target = -1.0 if ch_left_t else 1.0
        a_hist = a_chr if was_correct else a_chp
        target_bias = target if was_correct else -target
        last_bias = last_bias + a_hist * (target_bias - last_bias)
        
        # C. Update Context
        p_short_prior = p_short * (1.0 - p_switch) + (1.0 - p_short) * p_switch
        if is_short_t:
            L_short = p_common
            L_long = p_rare
        else:
            L_short = p_rare
            L_long = p_common
        num = L_short * p_short_prior
        den = num + L_long * (1.0 - p_short_prior)
        p_short = num / (den + 1e-12)
        
    return nll_total