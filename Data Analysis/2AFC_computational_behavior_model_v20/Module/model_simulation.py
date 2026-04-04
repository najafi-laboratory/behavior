"""
Module/model_simulation.py

Contains the high-speed JIT-compiled optimization loop (`run_session_jit`) 
and the detailed post-hoc forward pass simulation logic used to generate 
the final trace output dataframes.
"""

import torch
import pandas as pd
from Module.fitting_metrics import get_session_tensors

@torch.jit.script
def run_session_jit(
    decay: torch.Tensor, noise: torch.Tensor,
    a_rew: torch.Tensor, a_pun: torch.Tensor,
    gamma: torch.Tensor,
    a_unc_ctx: torch.Tensor, a_unc_sens: torch.Tensor,
    a_chr: torch.Tensor, a_chp: torch.Tensor,
    beta: torch.Tensor, lapse: torch.Tensor,
    p_switch: torch.Tensor, p_rare: torch.Tensor,
    alpha_bound: torch.Tensor,
    fixed_bias: torch.Tensor, start_phys_bound: torch.Tensor,
    isi_ten: torch.Tensor, is_short_ten: torch.Tensor,
    ch_left_ten: torch.Tensor,
    n_samples: int,
    rare_weight: float,
    barrier_penalty: torch.Tensor  
) -> torch.Tensor:
    """
    High-speed PyTorch JIT compilation of the model's forward pass.
    Accumulates and returns weighted NLL + the soft barrier penalty.
    """
    normalizer = 1.0
    p_common   = 1.0 - p_rare
 
    w_time       = torch.tensor(0.5)
    w_ctx        = torch.tensor(0.5)
    p_short      = torch.tensor(0.5)
    last_bias    = torch.tensor(0.0)
    curr_phys_bound = start_phys_bound
    nll_total    = torch.tensor(0.0)
    eps_log      = 1e-9
 
    N = isi_ten.size(0)
 
    for i in range(N):
        isi_t      = isi_ten[i]
        is_short_t = is_short_ten[i]
        ch_left_t  = ch_left_ten[i]
 
        # Boundary
        cdf_bound      = 1.0 - (1.0 + decay * curr_phys_bound) * torch.exp(-decay * curr_phys_bound)
        internal_boundary = normalizer * cdf_bound
 
        # Time sampling
        isi_exp   = isi_t.expand(n_samples)
        isi_safe  = torch.clamp(isi_exp, min=0.0)
        cdf_val   = 1.0 - (1.0 + decay * isi_safe) * torch.exp(-decay * isi_safe)
        mean_perc = normalizer * cdf_val
        std_perc  = mean_perc / noise
        noise_vec = torch.randn_like(mean_perc)
        perceived_time_vec = mean_perc + noise_vec * torch.clamp(std_perc, min=1e-6)
        perceived_time_vec = torch.clamp(perceived_time_vec, min=0.01)
 
        # Decision Variable
        dist      = perceived_time_vec - internal_boundary
        exp_arg   = torch.clamp(-dist, -50.0, 50.0)
        sigmoid_val = 1.0 / (1.0 + torch.exp(exp_arg))
        time_evidence = 2.0 * sigmoid_val - 1.0
 
        context_evidence = 0.5 - p_short
        dv   = w_time * time_evidence + w_ctx * context_evidence + last_bias + fixed_bias
        dv_s = torch.clamp(beta * dv, -50.0, 50.0)
        p_left_vec = 1.0 / (1.0 + torch.exp(dv_s))
 
        avg_p  = torch.mean(p_left_vec)
        p_left = avg_p * (1.0 - lapse) + 0.5 * lapse
 
        # Weighted importance
        trial_matches_belief = (is_short_t == (p_short >= 0.5))
        trial_weight = 1.0 if trial_matches_belief else rare_weight
 
        if ch_left_t:
            nll_total = nll_total - trial_weight * torch.log(p_left + eps_log)
        else:
            nll_total = nll_total - trial_weight * torch.log(1.0 - p_left + eps_log)
 
        was_correct = (ch_left_t == is_short_t)
 
        # State updates
        bound_shift = torch.tensor(0.0)
        if not was_correct:
            if is_short_t:
                bound_shift = alpha_bound
            else:
                bound_shift = -alpha_bound
        curr_phys_bound = torch.clamp(curr_phys_bound + bound_shift, 0.2, 2.3)
 
        unc          = 4.0 * p_short * (1.0 - p_short)
        was_correct_f = float(was_correct)
 
        t_accum       = torch.mean(perceived_time_vec)
        dist_avg      = t_accum - internal_boundary
        exp_arg_avg   = torch.clamp(-dist_avg, -50.0, 50.0)
        sigmoid_avg   = 1.0 / (1.0 + torch.exp(exp_arg_avg))
        c_sensory     = torch.abs(2.0 * sigmoid_avg - 1.0)
 
        alpha_base    = was_correct_f * a_rew + (1.0 - was_correct_f) * a_pun
        alpha_ctx     = alpha_base * (1.0 + a_unc_ctx * unc) * gamma
        alpha_sensory = alpha_base * (1.0 + a_unc_sens * c_sensory) * (1.0 - gamma)
 
        is_perc_short = (t_accum < internal_boundary)
        time_ok       = (is_perc_short == is_short_t)
        dir_time      = 2.0 * float(time_ok) - 1.0
 
        context_evidence_upd = 0.5 - p_short
        if context_evidence_upd > 0:
            dir_ctx = 1.0 if is_short_t else -1.0
        elif context_evidence_upd < 0:
            dir_ctx = -1.0 if is_short_t else 1.0
        else:
            dir_ctx = 0.0
 
        w_time = w_time + alpha_sensory * dir_time
        w_ctx  = w_ctx  + alpha_ctx     * dir_ctx
        if w_time < 0.0: w_time = torch.tensor(0.0)
        if w_ctx  < 0.0: w_ctx  = torch.tensor(0.0)
 
        target      = -1.0 if ch_left_t else 1.0
        a_hist      = a_chr if was_correct else a_chp
        target_bias = target if was_correct else -target
        last_bias   = last_bias + a_hist * (target_bias - last_bias)
 
        p_short_prior = p_short * (1.0 - p_switch) + (1.0 - p_short) * p_switch
        if is_short_t:
            L_short, L_long = p_common, p_rare
        else:
            L_short, L_long = p_rare, p_common
        num     = L_short * p_short_prior
        den     = num + L_long * (1.0 - p_short_prior)
        p_short = num / (den + 1e-12)
 
    return nll_total + barrier_penalty

def simulate_detailed_session_torch(model, df_session):
    """
    Runs the fully constructed model instance over the session data to extract
    the internal states (weights, priors, bounds) and simulated choices.
    """
    model.reset_state()
    isi_ten, is_short_ten, _ = get_session_tensors(df_session)
    df_session = df_session.reset_index(drop=True)
 
    out = []
    trials_since_switch = 0
    prev_block = None
    n_samples = 100
 
    with torch.no_grad():
        for i, row in df_session.iterrows():
            p_short_pre = model.p_short_block.item()
            w_time_pre = model.weights[0].item()
            w_ctx_pre = model.weights[1].item()
            bias_pre = model.last_choice_bias.item()
            boundary_pre = model.current_boundary.item()

            pre = {
                'p_short_pre': p_short_pre,
                'w_time_pre': w_time_pre,
                'w_ctx_pre': w_ctx_pre,
                'bias_pre': bias_pre,
                'boundary_pre': boundary_pre,
            }
 
            curr_block = row['block_type']
            if prev_block is not None and curr_block != prev_block:
                trials_since_switch = 0
            else:
                trials_since_switch += 1
            prev_block = curr_block
 
            isi_t = isi_ten[i]

            cdf_bound = 1.0 - (1.0 + model.decay_rate * model.current_boundary) * torch.exp(
                -model.decay_rate * model.current_boundary
            )
            internal_boundary = model.normalizer * cdf_bound

            isi_expanded = isi_t.expand(n_samples)
            isi_safe = torch.clamp(isi_expanded, min=0.0)
            cdf_val = 1.0 - (1.0 + model.decay_rate * isi_safe) * torch.exp(-model.decay_rate * isi_safe)
            mapped_time = model.normalizer * cdf_val
            mapped_time_mean = torch.mean(mapped_time)
            mapped_time_std = mapped_time_mean / model.noise_param_a

            noise_vec = torch.randn_like(mapped_time)
            added_noise_vec = noise_vec * torch.clamp(mapped_time_std, min=1e-6)
            perceived_time_vec = torch.clamp(mapped_time + added_noise_vec, min=0.01)
            t_perc_avg = torch.mean(perceived_time_vec)
            added_noise_mean = t_perc_avg - mapped_time_mean
            added_noise_std = torch.std(added_noise_vec, unbiased=False)

            sensory_dist_vec = perceived_time_vec - internal_boundary
            sensory_exp_arg_vec = torch.clamp(-sensory_dist_vec, -50.0, 50.0)
            sensory_sigmoid_vec = 1.0 / (1.0 + torch.exp(sensory_exp_arg_vec))
            sensory_evidence_vec = 2.0 * sensory_sigmoid_vec - 1.0

            sensory_dist_mean = t_perc_avg - internal_boundary
            sensory_exp_arg_mean = torch.clamp(-sensory_dist_mean, -50.0, 50.0)
            sensory_sigmoid_mean = 1.0 / (1.0 + torch.exp(sensory_exp_arg_mean))
            sensory_evidence_mean = 2.0 * sensory_sigmoid_mean - 1.0

            context_evidence_pre = 0.5 - model.p_short_block
            dv_time_vec = model.weights[0] * sensory_evidence_vec
            dv_ctx_scalar = model.weights[1] * context_evidence_pre
            decision_variable_vec = dv_time_vec + dv_ctx_scalar + model.last_choice_bias + model.static_bias
            decision_variable_mean = torch.mean(decision_variable_vec)
            decision_variable_no_lapse_mean = (
                model.weights[0] * sensory_evidence_mean
                + dv_ctx_scalar
                + model.last_choice_bias
                + model.static_bias
            )

            dv_scaled_vec = torch.clamp(model.beta * decision_variable_vec, -50.0, 50.0)
            decision_sigmoid_vec = 1.0 / (1.0 + torch.exp(dv_scaled_vec))
            decision_without_lapse = torch.mean(decision_sigmoid_vec)
            decision_with_lapse = decision_without_lapse * (1.0 - model.lapse) + 0.5 * model.lapse

            decision_scaled_mean = torch.clamp(model.beta * decision_variable_no_lapse_mean, -50.0, 50.0)
            decision_sigmoid_mean = 1.0 / (1.0 + torch.exp(decision_scaled_mean))
 
            is_left = torch.rand(1).item() < decision_with_lapse.item()
            choice_str = 'left' if is_left else 'right'
 
            target_str = row['trial_type']
            correct_map = {'short': 'left', 'long': 'right'}
            is_correct = (choice_str == correct_map[target_str])
            is_short_bool = (target_str == 'short')
 
            is_rare = (is_short_bool != (p_short_pre > 0.5))

            bound_shift = 0.0
            if not is_correct:
                bound_shift = model.alpha_boundary.item() if is_short_bool else -model.alpha_boundary.item()
            model.current_boundary = torch.clamp(
                model.current_boundary + bound_shift, 0.2, 2.3
            )

            boundary_post = model.current_boundary.item()

            uncertainty_pre = 4.0 * model.p_short_block * (1.0 - model.p_short_block)
            was_correct_t = torch.tensor(is_correct)
            is_short_t = torch.tensor(is_short_bool)
            choice_was_left_t = torch.tensor(is_left)

            c_sensory = torch.abs(sensory_evidence_mean)
            alpha_base = (
                was_correct_t.float() * model.alpha_reward_base
                + (1 - was_correct_t.float()) * model.alpha_punish_base
            )
            alpha_ctx = alpha_base * (1.0 + model.alpha_unc_ctx * uncertainty_pre) * model.gamma
            alpha_sensory = alpha_base * (1.0 + model.alpha_unc_sens * c_sensory) * (1.0 - model.gamma)

            is_perc_short = t_perc_avg < internal_boundary
            time_correct_bool = (is_perc_short == is_short_t)
            dir_time = 2.0 * time_correct_bool.float() - 1.0

            if context_evidence_pre > 0:
                dir_ctx = torch.where(is_short_t, torch.tensor(1.0), torch.tensor(-1.0))
            elif context_evidence_pre < 0:
                dir_ctx = torch.where(is_short_t, torch.tensor(-1.0), torch.tensor(1.0))
            else:
                dir_ctx = torch.tensor(0.0)

            delta_w_time_raw = alpha_sensory * dir_time
            delta_w_ctx_raw = alpha_ctx * dir_ctx
            w_time_post = max(0.0, w_time_pre + delta_w_time_raw.item())
            w_ctx_post = max(0.0, w_ctx_pre + delta_w_ctx_raw.item())
            delta_w_time_applied = w_time_post - w_time_pre
            delta_w_ctx_applied = w_ctx_post - w_ctx_pre

            target_bias_direction = -1.0 if is_left else 1.0
            history_alpha = (
                model.alpha_choice_reward if is_correct else model.alpha_choice_punish
            )
            target_bias = target_bias_direction if is_correct else -target_bias_direction
            bias_post = bias_pre + history_alpha * (target_bias - bias_pre)
            bias_delta = bias_post - bias_pre

            p_short_prior = model.p_short_block * (1.0 - model.p_switch) + (
                1.0 - model.p_short_block
            ) * model.p_switch
            likelihood_short = model.p_common if is_short_bool else model.p_rare
            likelihood_long = model.p_rare if is_short_bool else model.p_common
            bayes_num = likelihood_short * p_short_prior
            bayes_den = bayes_num + likelihood_long * (1.0 - p_short_prior)
            p_short_post = bayes_num / (bayes_den + 1e-12)

            model.update_weights(was_correct_t, is_short_t, t_perc_avg, choice_was_left_t)

            out.append({
                **pre,
                'trial_in_session':    i,
                'isi':                 row['isi'],
                'trial_type':          row['trial_type'],
                'block_type':          row['block_type'],
                'mouse_choice':        row['mouse_choice'],
                'model_choice':        choice_str,
                'correct_model':       is_correct,
                'is_rare_trial':       is_rare,
                'trials_since_switch': trials_since_switch,
                'fixed_bias':          model.static_bias.item(),
                'lapse_value':         model.lapse,
                'boundary_cdf_pre':    cdf_bound.item(),
                'internal_boundary':   internal_boundary.item(),
                'mapped_time_cdf':     cdf_val[0].item(),
                'mapped_time_mean':    mapped_time_mean.item(),
                'mapped_time_std':     mapped_time_std.item(),
                'added_noise_mean':    added_noise_mean.item(),
                'added_noise_std':     added_noise_std.item(),
                'perceived_time_mean': t_perc_avg.item(),
                'sensory_dist_mean':   sensory_dist_mean.item(),
                'sensory_exp_arg':     sensory_exp_arg_mean.item(),
                'sensory_sigmoid':     sensory_sigmoid_mean.item(),
                'sensory_evidence':    sensory_evidence_mean.item(),
                'dv_time_mean':        torch.mean(dv_time_vec).item(),
                'context_evidence_pre': context_evidence_pre.item(),
                'dv_context':          dv_ctx_scalar.item(),
                'decision_variable_mean': decision_variable_mean.item(),
                'decision_variable_no_lapse_mean': decision_variable_no_lapse_mean.item(),
                'decision_beta_scaled_mean': decision_scaled_mean.item(),
                'decision_sigmoid_mean': decision_sigmoid_mean.item(),
                'decision_without_lapse': decision_without_lapse.item(),
                'decision_with_lapse': decision_with_lapse.item(),
                'boundary_shift':      bound_shift,
                'boundary_post':       boundary_post,
                'uncertainty_pre':     uncertainty_pre.item(),
                'alpha_base':          alpha_base.item(),
                'alpha_context':       alpha_ctx.item(),
                'alpha_sensory':       alpha_sensory.item(),
                'time_is_perceived_short': bool(is_perc_short.item()),
                'time_correct_direction':  bool(time_correct_bool.item()),
                'dir_time':            dir_time.item(),
                'dir_context':         dir_ctx.item(),
                'delta_w_time_raw':    delta_w_time_raw.item(),
                'delta_w_ctx_raw':     delta_w_ctx_raw.item(),
                'delta_w_time_applied': delta_w_time_applied,
                'delta_w_ctx_applied': delta_w_ctx_applied,
                'w_time_post':         w_time_post,
                'w_ctx_post':          w_ctx_post,
                'history_alpha':       history_alpha,
                'target_bias':         target_bias,
                'bias_delta':          bias_delta,
                'bias_post':           bias_post,
                'p_short_prior':       p_short_prior.item(),
                'likelihood_short':    likelihood_short.item(),
                'likelihood_long':     likelihood_long.item(),
                'bayes_numerator':     bayes_num.item(),
                'bayes_denominator':   bayes_den.item(),
                'p_short_post':        p_short_post.item(),
            })
 
    return pd.DataFrame(out)
