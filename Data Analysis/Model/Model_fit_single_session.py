import torch
import torch.optim as optim
from tqdm import tqdm
from Model_behavior import MouseModelTorch, PHYSICAL_BOUNDARY_SECONDS
from Model_simulate_session import simulate_detailed_session_torch
from Model_fitting import calculate_neutral_bias, get_session_tensors, run_session_jit

N_RESTARTS = 40
OPTIM_STEPS_STAGE_1 = 3000
OPTIM_STEPS_STAGE_2 = 3000
OPTIM_STEPS_STAGE_3 = 3000
LEARNING_RATE = 0.0005

def fit_single_session_torch(args):
    """
    Fit a single session using three-stage optimization with PyTorch.
    
    Args:
        args: Tuple containing (session_data, session_index, bounds, _)
            - session_data: Session behavioral data
            - session_index: Index of the session
            - bounds: Parameter bounds for optimization
            - _: Unused parameter
    
    Returns:
        dict: Contains fitted parameters, NLL, simulation results, and loss history
    """
    session_data, session_index, bounds, _ = args
    fixed_bias = torch.tensor(calculate_neutral_bias(session_data), dtype=torch.float32)
    isi_ten, is_short_ten, ch_left_ten = get_session_tensors(session_data)
    
    start_phys_bound = torch.tensor(PHYSICAL_BOUNDARY_SECONDS, dtype=torch.float32)
    
    active_bounds = bounds 
    low_b = torch.tensor([b[0] for b in active_bounds], dtype=torch.float32)
    high_b = torch.tensor([b[1] for b in active_bounds], dtype=torch.float32)
    
    best_overall_nll = float('inf')
    best_overall_params = None
    best_loss_history = []  

    restart_pbar = tqdm(range(N_RESTARTS), desc=f"Sess {session_index} Restarts", leave=False)
    
    for run_i in restart_pbar:
        torch.manual_seed(session_index * 1000 + run_i)
        
        current_run_history = [] 
        
        init_vals = []
        for lo, hi in active_bounds:
            val = (torch.rand(1) * (hi - lo) + lo).item()
            init_vals.append(val)
        
        params_vec = torch.tensor(init_vals, dtype=torch.float32, requires_grad=True)

        def clamp_params():
            """Clamp parameter vector to respect optimization bounds."""
            with torch.no_grad():
                params_vec.data = torch.max(torch.min(params_vec.data, high_b), low_b)

        def call_jit_model(p_vec, n_samp):
            """
            Call the JIT-compiled session runner with given parameters.
            Args:
                p_vec (torch.Tensor): Parameter vector.
                n_samp (int): Number of Monte Carlo samples.
            Returns:
                torch.Tensor: Negative log-likelihood of the session.
            """
            return run_session_jit(
                decay=p_vec[0], noise=p_vec[1],
                a_rew=p_vec[2], a_pun=p_vec[3], a_unc=p_vec[4],
                a_chr=p_vec[5], a_chp=p_vec[6],
                beta=p_vec[7], lapse=p_vec[8],
                p_switch=p_vec[9], p_rare=p_vec[10],
                alpha_bound=p_vec[11],          
                fixed_bias=fixed_bias, start_phys_bound=start_phys_bound,
                isi_ten=isi_ten, is_short_ten=is_short_ten, ch_left_ten=ch_left_ten,
                n_samples=n_samp
            )
        
        # --- Stage 1 (Sensory) ---
        # Increased sample count for better gradient stability in sensory stage
        optimizer_s1 = optim.Adam([params_vec], lr=LEARNING_RATE)
        zero_t = torch.tensor(0.0)
        
        for step in range(OPTIM_STEPS_STAGE_1):
            restart_pbar.set_postfix({'Stage': '1/3 (Sensory)', 'Step': f'{step+1}/{OPTIM_STEPS_STAGE_1}'})
            optimizer_s1.zero_grad()
            p_masked = torch.stack([
                params_vec[0], params_vec[1],
                zero_t, zero_t, zero_t, zero_t, zero_t, 
                params_vec[7], params_vec[8],
                params_vec[9].detach(), params_vec[10].detach(),
                zero_t 
            ])
            loss = call_jit_model(p_masked, n_samp=50) # Increased MC samples
            loss.backward()
            optimizer_s1.step()
            clamp_params()
            current_run_history.append(loss.item()) 

        # --- Stage 2 (Strategy) ---
        optimizer_s2 = optim.Adam([params_vec], lr=LEARNING_RATE)
        for step in range(OPTIM_STEPS_STAGE_2):
            restart_pbar.set_postfix({'Stage': '2/3 (Strategy)', 'Step': f'{step+1}/{OPTIM_STEPS_STAGE_2}'})
            optimizer_s2.zero_grad()
            p_masked = torch.stack([
                params_vec[0].detach(), params_vec[1].detach(),
                params_vec[2], params_vec[3], params_vec[4], params_vec[5], params_vec[6],
                params_vec[7].detach(), params_vec[8].detach(),
                params_vec[9], params_vec[10],
                params_vec[11] 
            ])
            loss = call_jit_model(p_masked, n_samp=30) # Moderate MC samples
            loss.backward()
            optimizer_s2.step()
            clamp_params()
            current_run_history.append(loss.item()) 

        # --- Stage 3 (FineTune) ---
        optimizer_s3 = optim.Adam([params_vec], lr=LEARNING_RATE * 0.5)
        for step in range(OPTIM_STEPS_STAGE_3):
            restart_pbar.set_postfix({'Stage': '3/3 (FineTune)', 'Step': f'{step+1}/{OPTIM_STEPS_STAGE_3}'})
            optimizer_s3.zero_grad()
            loss = call_jit_model(params_vec, n_samp=50) # High MC samples for final polish
            loss.backward()
            optimizer_s3.step()
            clamp_params()
            current_run_history.append(loss.item()) 
            
        final_loss = loss.item()
        if final_loss < best_overall_nll:
            best_overall_nll = final_loss
            best_overall_params = params_vec.detach().numpy()
            best_loss_history = current_run_history 

    bp = best_overall_params
    final_model = MouseModelTorch(
        bp[0], bp[1], bp[2], bp[3], bp[4], bp[5], bp[6], bp[7], bp[8],
        bp[9], bp[10], bp[11], 
        fixed_bias_value=fixed_bias.item(), initial_boundary=start_phys_bound.item()
    )
    sim_df = simulate_detailed_session_torch(final_model, session_data)
    
    return {
        'session_id': session_index, 'best_params': best_overall_params,
        'fixed_bias': fixed_bias, 'nll': best_overall_nll, 'success': True, 
        'sim_df': sim_df,
        'loss_history': best_loss_history
    }