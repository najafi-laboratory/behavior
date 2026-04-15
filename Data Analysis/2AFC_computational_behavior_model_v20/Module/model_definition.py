"""
Module/model_definition.py

Defines the PyTorch-based Mouse Model. It includes internal time perception,
context belief updating, and choice probability generation.
"""

import torch
import torch.nn as nn

class MouseModelTorch(nn.Module):
    """
    14-Parameter computational model of mouse behavior for a 2AFC task.
    """
    def __init__(self,
                 decay_rate, noise_param_a, 
                 alpha_reward_base, alpha_punish_base, 
                 gamma, alpha_unc_ctx, alpha_unc_sens,
                 alpha_choice_reward, alpha_choice_punish,
                 beta, lapse_rate, subjective_p_switch, subjective_p_rare, 
                 alpha_boundary, 
                 fixed_bias_value=0.0, initial_boundary=1.25): 
        super().__init__()
        
        # Core perceptual & learning params
        self.decay_rate = decay_rate
        self.noise_param_a = noise_param_a
        self.alpha_reward_base = alpha_reward_base
        self.alpha_punish_base = alpha_punish_base
        self.gamma = torch.tensor(gamma, dtype=torch.float32) 
        self.alpha_unc_ctx = alpha_unc_ctx
        self.alpha_unc_sens = alpha_unc_sens
        self.alpha_choice_reward = alpha_choice_reward
        self.alpha_choice_punish = alpha_choice_punish  
        self.beta = beta
        self.lapse = lapse_rate
        
        # Priors and boundaries
        self.p_switch = torch.tensor(subjective_p_switch, dtype=torch.float32)
        self.p_rare = torch.tensor(subjective_p_rare, dtype=torch.float32)
        self.p_common = 1.0 - self.p_rare
        self.alpha_boundary = torch.tensor(alpha_boundary, dtype=torch.float32)
        self.normalizer = torch.tensor(1.0, dtype=torch.float32)
        self.static_bias = torch.tensor(fixed_bias_value, dtype=torch.float32)
        self.initial_boundary = torch.tensor(initial_boundary, dtype=torch.float32)
        
        self.reset_state()

    def reset_state(self):
        """Resets dynamic internal states at the start of a session."""
        self.weights = torch.tensor([0.5, 0.5], dtype=torch.float32)
        self.p_short_block = torch.tensor(0.5, dtype=torch.float32)
        self.last_choice_bias = torch.tensor(0.0, dtype=torch.float32)
        self.current_boundary = self.initial_boundary.clone()

    @torch.jit.export
    def _time_perception_reparam(self, isi_expanded):
        """Applies internal scalar noise and delay to objective ISI."""
        lam = self.decay_rate
        isi_safe = torch.clamp(isi_expanded, min=0.0)
        cdf_val = 1.0 - (1.0 + lam * isi_safe) * torch.exp(-lam * isi_safe)
        mean_perc = self.normalizer * cdf_val
        std_perc = mean_perc / self.noise_param_a
        eps = torch.randn_like(mean_perc)
        perceived_time = mean_perc + eps * torch.clamp(std_perc, min=1e-6)
        return torch.clamp(perceived_time, min=0.01)

    @torch.jit.export
    def _update_context_belief(self, is_short_trial):
        """Bayesian update of the block-type probability."""
        p_short_prior = self.p_short_block * (1 - self.p_switch) + (1 - self.p_short_block) * self.p_switch
        L_short = torch.where(is_short_trial, self.p_common, self.p_rare)
        L_long = torch.where(is_short_trial, self.p_rare, self.p_common)
        num = L_short * p_short_prior
        den = num + L_long * (1 - p_short_prior)
        self.p_short_block = num / (den + 1e-12)

    @torch.jit.export
    def get_choice_probabilities(self, isi: torch.Tensor, current_phys_bound: torch.Tensor, n_samples: int = 50):
        """Calculates probabilistic decision variable (softmax output)."""
        cdf_bound = 1.0 - (1.0 + self.decay_rate * current_phys_bound) * torch.exp(-self.decay_rate * current_phys_bound)
        internal_boundary = self.normalizer * cdf_bound

        isi_expanded = isi.expand(n_samples) 
        perceived_time_vec = self._time_perception_reparam(isi_expanded)
        
        dist = perceived_time_vec - internal_boundary
        exp_arg = torch.clamp(-dist, -50.0, 50.0) 
        sigmoid_val = 1.0 / (1.0 + torch.exp(exp_arg))
        time_evidence = (2.0 * sigmoid_val - 1.0) 
        
        context_evidence = (0.5 - self.p_short_block) 
        dv_time = self.weights[0] * time_evidence
        dv_ctx = self.weights[1] * context_evidence
        decision_variable = dv_time + dv_ctx + self.last_choice_bias + self.static_bias
        
        dv_scaled = torch.clamp(self.beta * decision_variable, -50.0, 50.0)
        prob_left_no_lapse = 1.0 / (1.0 + torch.exp(dv_scaled))
        
        avg_prob = torch.mean(prob_left_no_lapse)
        prob_left = avg_prob * (1 - self.lapse) + 0.5 * self.lapse
        
        return prob_left, torch.mean(perceived_time_vec), internal_boundary

    @torch.jit.export
    def update_weights(self, was_correct, is_short_trial, perceived_time, choice_was_left):
        """Updates internal weights depending on outcome and confidence."""
        cdf_bound = 1.0 - (1.0 + self.decay_rate * self.current_boundary) * torch.exp(-self.decay_rate * self.current_boundary)
        internal_boundary = self.normalizer * cdf_bound

        unc = 4.0 * self.p_short_block * (1.0 - self.p_short_block)
        was_correct_f = was_correct.float()
        
        dist = perceived_time - internal_boundary
        exp_arg = torch.clamp(-dist, -50.0, 50.0)
        sigmoid_val = 1.0 / (1.0 + torch.exp(exp_arg))
        c_sensory = torch.abs(2.0 * sigmoid_val - 1.0)

        alpha_base = was_correct_f * self.alpha_reward_base + (1 - was_correct_f) * self.alpha_punish_base
        alpha_ctx = alpha_base * (1.0 + self.alpha_unc_ctx * unc) * self.gamma
        alpha_sensory = alpha_base * (1.0 + self.alpha_unc_sens * c_sensory) * (1.0 - self.gamma)
        
        is_perc_short = (perceived_time < internal_boundary)
        time_ok_bool = (is_perc_short == is_short_trial)
        context_evidence = 0.5 - self.p_short_block
        
        dir_time = 2.0 * time_ok_bool.float() - 1.0
        if context_evidence > 0:
            dir_ctx = torch.where(is_short_trial, torch.tensor(1.0), torch.tensor(-1.0))
        elif context_evidence < 0:
            dir_ctx = torch.where(is_short_trial, torch.tensor(-1.0), torch.tensor(1.0))
        else:
            dir_ctx = torch.tensor(0.0)
            
        w_time_new = self.weights[0] + alpha_sensory * dir_time
        w_ctx_new = self.weights[1] + alpha_ctx * dir_ctx
        self.weights = torch.clamp(torch.stack([w_time_new, w_ctx_new]), min=0.0)
        
        target = torch.where(choice_was_left, torch.tensor(-1.0), torch.tensor(1.0))
        a_hist = torch.where(was_correct, self.alpha_choice_reward, self.alpha_choice_punish)
        target_bias = torch.where(was_correct, target, -target)
        self.last_choice_bias = self.last_choice_bias + a_hist * (target_bias - self.last_choice_bias)
        self._update_context_belief(is_short_trial)