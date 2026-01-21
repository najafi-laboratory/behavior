import torch
from Model_fitting import get_session_tensors
import pandas as pd

def simulate_detailed_session_torch(model, df_session):
    """
    Simulate a detailed session using the provided mouse behavior model.
    Args:
        model (MouseModelTorch): The mouse behavior model instance.
        df_session (pd.DataFrame): DataFrame containing session trial data.
    Returns:
        pd.DataFrame: DataFrame with detailed simulation results per trial.
    """
        
    model.reset_state()
    isi_ten, is_short_ten, _ = get_session_tensors(df_session)
    out = []
    
    with torch.no_grad():
        for i, row in df_session.iterrows():
            pre = {
                'p_short_pre': model.p_short_block.item(),
                'w_time_pre': model.weights[0].item(),
                'w_ctx_pre': model.weights[1].item(),
                'bias_pre': model.last_choice_bias.item(),
                'boundary_pre': model.current_boundary.item() 
            }
            
            isi_t = isi_ten[i]
            # Use high samples for detailed simulation to get accurate p_left
            p_left, t_perc_avg, _ = model.get_choice_probabilities(isi_t, model.current_boundary, n_samples=100)
            
            is_left = torch.rand(1).item() < p_left.item()
            choice_str = 'left' if is_left else 'right'
            
            target_str = row['trial_type'] 
            correct_map = {'short': 'left', 'long': 'right'}
            is_correct_model = (choice_str == correct_map[target_str])
            is_short_bool = (target_str == 'short')
            
            # --- MANUAL BOUNDARY UPDATE FOR SIMULATION ---
            bound_shift = 0.0
            if not is_correct_model:
                if is_short_bool: 
                    bound_shift = model.alpha_boundary.item()
                else:              
                    bound_shift = -model.alpha_boundary.item()
            
            model.current_boundary = torch.clamp(model.current_boundary + bound_shift, 0.5, 2.5)
            # ---------------------------------------------
            
            model.update_weights(torch.tensor(is_correct_model), torch.tensor(is_short_bool), t_perc_avg, torch.tensor(is_left))
            
            # Capture Post-Update States (Added for full weight tracking)
            post = {
                'p_short_post': model.p_short_block.item(),
                'w_time_post': model.weights[0].item(),
                'w_ctx_post': model.weights[1].item(),
                'bias_post': model.last_choice_bias.item(),
                'boundary_post': model.current_boundary.item()
            }
            
            out.append({**pre, **post,
                        'isi': row['isi'],
                        'trial_type': row['trial_type'],
                        'block_type': row['block_type'],
                        'mouse_choice': row['mouse_choice'],
                        'model_choice': choice_str,
                        'correct_model': is_correct_model})
            
    return pd.DataFrame(out)