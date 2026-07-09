"""
Module/model_fitting.py

Orchestrates the 3-stage PyTorch optimization process (Adam optimizer)
for a single experimental session, managing gradients, boundaries, and restarts.
"""

import torch
import torch.optim as optim
from tqdm import tqdm

from constants import (N_RESTARTS, OPTIM_STEPS_STAGE_1, OPTIM_STEPS_STAGE_2, 
                       OPTIM_STEPS_STAGE_3, LEARNING_RATE, PHYSICAL_BOUNDARY_SECONDS,
                       RARE_TRIAL_WEIGHT)

from Module.fitting_metrics import (compute_soft_barrier_penalty, select_best_restart, 
                                    calculate_neutral_bias, get_session_tensors)
from Module.model_simulation import run_session_jit, simulate_detailed_session_torch
from Module.model_definition import MouseModelTorch

def fit_single_session_torch(args):
    """
    Fits the PyTorch behavior model to a single session's data using a 3-stage Adam approach.
    Executed in parallel via multiprocessing mapping.
    """
    session_data, session_index, bounds, _ = args
 
    fixed_bias      = torch.tensor(calculate_neutral_bias(session_data), dtype=torch.float32)
    isi_ten, is_short_ten, ch_left_ten = get_session_tensors(session_data)
    start_phys_bound = torch.tensor(PHYSICAL_BOUNDARY_SECONDS, dtype=torch.float32)
 
    low_b  = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
    high_b = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
 
    PERCEPTUAL_IDX = [0, 1, 9, 10]
    LEARNING_IDX   = [2, 3, 4, 5, 6, 7, 8, 11, 12, 13]
 
    all_restart_results = []
    restart_pbar = tqdm(range(N_RESTARTS), desc=f"Sess {session_index} Restarts", leave=False)
 
    for run_i in restart_pbar:
        torch.manual_seed(session_index * 1000 + run_i)
        current_run_history = []
 
        init_vals  = [(torch.rand(1) * (hi - lo) + lo).item() for lo, hi in bounds]
        params_vec = torch.tensor(init_vals, dtype=torch.float32, requires_grad=True)
 
        def clamp_params():
            with torch.no_grad():
                params_vec.data = torch.max(torch.min(params_vec.data, high_b), low_b)
 
        def call_jit_model(p_vec, n_samp):
            barrier = compute_soft_barrier_penalty(p_vec, bounds)
            return run_session_jit(
                decay=p_vec[0],  noise=p_vec[1],
                a_rew=p_vec[2],  a_pun=p_vec[3],
                gamma=p_vec[4],
                a_unc_ctx=p_vec[5], a_unc_sens=p_vec[6],
                a_chr=p_vec[7],  a_chp=p_vec[8],
                beta=p_vec[9],   lapse=p_vec[10],
                p_switch=p_vec[11], p_rare=p_vec[12],
                alpha_bound=p_vec[13],
                fixed_bias=fixed_bias,
                start_phys_bound=start_phys_bound,
                isi_ten=isi_ten,
                is_short_ten=is_short_ten,
                ch_left_ten=ch_left_ten,
                n_samples=n_samp,
                rare_weight=RARE_TRIAL_WEIGHT,
                barrier_penalty=barrier
            )
 
        # Stage 1: Sensory priority
        optimizer_s1 = optim.Adam([params_vec], lr=LEARNING_RATE)
        for step in range(OPTIM_STEPS_STAGE_1):
            restart_pbar.set_postfix({'Stage': '1/3', 'Step': step + 1})
            optimizer_s1.zero_grad()
            loss = call_jit_model(params_vec, n_samp=50)
            loss.backward()
 
            with torch.no_grad():
                for idx in LEARNING_IDX:
                    if params_vec.grad is not None:
                        params_vec.grad[idx] *= 0.05
 
            optimizer_s1.step()
            clamp_params()
            current_run_history.append(loss.item())
 
        # Stage 2: Strategy priority
        optimizer_s2 = optim.Adam([params_vec], lr=LEARNING_RATE)
        for step in range(OPTIM_STEPS_STAGE_2):
            restart_pbar.set_postfix({'Stage': '2/3', 'Step': step + 1})
            optimizer_s2.zero_grad()
            loss = call_jit_model(params_vec, n_samp=30)
            loss.backward()
 
            with torch.no_grad():
                for idx in PERCEPTUAL_IDX:
                    if params_vec.grad is not None:
                        params_vec.grad[idx] *= 0.05
 
            optimizer_s2.step()
            clamp_params()
            current_run_history.append(loss.item())
 
        # Stage 3: Fine tuning
        optimizer_s3 = optim.Adam([params_vec], lr=LEARNING_RATE * 0.3)
        for step in range(OPTIM_STEPS_STAGE_3):
            restart_pbar.set_postfix({'Stage': '3/3', 'Step': step + 1})
            optimizer_s3.zero_grad()
            loss = call_jit_model(params_vec, n_samp=50)
            loss.backward()
            optimizer_s3.step()
            clamp_params()
            current_run_history.append(loss.item())
 
        # Simulation
        bp          = params_vec.detach().numpy()
        final_model = MouseModelTorch(
            bp[0], bp[1], bp[2], bp[3], bp[4], bp[5], bp[6],
            bp[7], bp[8], bp[9], bp[10], bp[11], bp[12], bp[13],
            fixed_bias_value=fixed_bias.item(),
            initial_boundary=start_phys_bound.item()
        )
        sim_df = simulate_detailed_session_torch(final_model, session_data)
 
        all_restart_results.append({
            'session_id':   session_index,
            'best_params':  bp,
            'fixed_bias':   fixed_bias,
            'nll':          loss.item(),
            'success':      True,
            'sim_df':       sim_df,
            'loss_history': current_run_history,
        })
 
    best_result = select_best_restart(all_restart_results, session_data)
    return best_result