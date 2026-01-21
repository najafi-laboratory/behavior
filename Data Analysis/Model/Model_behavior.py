import torch
import torch.nn as nn
PHYSICAL_BOUNDARY_SECONDS = 1.25

class MouseModelTorch(nn.Module):
    """
    A reinforcement learning model of mouse behavior in a 2AFC task.
    
    Implements temporal perception via reparameterization trick, context belief updating,
    and dynamic boundary learning. Uses PyTorch for GPU compatibility and JIT compilation.
    
    Key Methods:
    - _time_perception_reparam: Samples perceived time using reparameterization trick
    - _update_context_belief: Updates belief about block context (short vs long)
    - get_choice_probabilities: Computes left choice probability given ISI
    - update_weights: Updates decision weights based on trial feedback
    """
    def __init__(self,
                 decay_rate, 
                 noise_param_a, 
                 alpha_reward_base, alpha_punish_base, alpha_uncertainty_gain,
                 alpha_choice_reward, alpha_choice_punish,
                 beta, lapse_rate,
                 subjective_p_switch, subjective_p_rare, 
                 alpha_boundary, 
                 fixed_bias_value=0.0,
                 initial_boundary=PHYSICAL_BOUNDARY_SECONDS): 
        super().__init__()
        
        self.decay_rate = decay_rate
        self.noise_param_a = noise_param_a
        self.alpha_reward_base = alpha_reward_base
        self.alpha_punish_base = alpha_punish_base
        self.alpha_uncertainty_gain = alpha_uncertainty_gain
        self.alpha_choice_reward = alpha_choice_reward
        self.alpha_choice_punish = alpha_choice_punish  
        self.beta = beta
        self.lapse = lapse_rate
        
        # Belief Parameters
        self.p_switch = torch.tensor(subjective_p_switch, dtype=torch.float32)
        self.p_rare = torch.tensor(subjective_p_rare, dtype=torch.float32)
        self.p_common = 1.0 - self.p_rare
        
        # Boundary Learning Rate
        self.alpha_boundary = torch.tensor(alpha_boundary, dtype=torch.float32)
        
        self.normalizer = torch.tensor(1.0, dtype=torch.float32)
        self.static_bias = torch.tensor(fixed_bias_value, dtype=torch.float32)
        self.initial_boundary = torch.tensor(initial_boundary, dtype=torch.float32)
        
        self.reset_state()

    def reset_state(self):
        self.weights = torch.tensor([0.5, 0.5], dtype=torch.float32)
        self.p_short_block = torch.tensor(0.5, dtype=torch.float32)
        self.last_choice_bias = torch.tensor(0.0, dtype=torch.float32)
        # Initialize dynamic boundary
        self.current_boundary = self.initial_boundary.clone()

    @torch.jit.export
    def _time_perception_reparam(self, isi_expanded):
        """
        Implements the Reparameterization Trick:
        1. Calculate deterministic mean (t) and std (sigma) from params.
        2. Draw epsilon ~ N(0,1).
        3. Return t + sigma * epsilon.
        """
        lam = self.decay_rate
        isi_safe = torch.clamp(isi_expanded, min=0.0)
        
        # 1. Deterministic Mean
        cdf_val = 1.0 - (1.0 + lam * isi_safe) * torch.exp(-lam * isi_safe)
        mean_perc = self.normalizer * cdf_val
        
        # 2. Deterministic Std (Weber's Law)
        std_perc = mean_perc / self.noise_param_a
        
        # 3. Random Sample (epsilon) - Graph Safe
        eps = torch.randn_like(mean_perc)
        
        # 4. Scale and Shift
        perceived_time = mean_perc + eps * torch.clamp(std_perc, min=1e-6)
        return torch.clamp(perceived_time, min=0.01)

    @torch.jit.export
    def _update_context_belief(self, is_short_trial):
        p_short_prior = self.p_short_block * (1 - self.p_switch) + \
                        (1 - self.p_short_block) * self.p_switch
        L_short = torch.where(is_short_trial, self.p_common, self.p_rare)
        L_long = torch.where(is_short_trial, self.p_rare, self.p_common)
        num = L_short * p_short_prior
        den = num + L_long * (1 - p_short_prior)
        self.p_short_block = num / (den + 1e-12)

    @torch.jit.export
    def get_choice_probabilities(self, isi: torch.Tensor, current_phys_bound: torch.Tensor, n_samples: int = 50):
        # 1. Calculate Internal Boundary dynamically
        cdf_bound = 1.0 - (1.0 + self.decay_rate * current_phys_bound) * torch.exp(-self.decay_rate * current_phys_bound)
        internal_boundary = self.normalizer * cdf_bound

        # 2. Expand ISI for Monte Carlo sampling
        isi_expanded = isi.expand(n_samples) 
        
        # 3. Get K samples of perceived time (Reparameterization Trick)
        perceived_time_vec = self._time_perception_reparam(isi_expanded)
        
        # 4. Calculate Decision Variable for EACH sample
        dist = perceived_time_vec - internal_boundary
        exp_arg = torch.clamp(-dist, -50.0, 50.0) 
        sigmoid_val = 1.0 / (1.0 + torch.exp(exp_arg))
        time_evidence = (2.0 * sigmoid_val - 1.0) # Vector of evidences
        
        context_evidence = 0.5 - self.p_short_block # Scalar
        
        dv_time = self.weights[0] * time_evidence
        dv_ctx = self.weights[1] * context_evidence
        decision_variable = dv_time + dv_ctx + self.last_choice_bias + self.static_bias
        
        # 5. Calculate Probabilities for EACH sample
        dv_scaled = torch.clamp(self.beta * decision_variable, -50.0, 50.0)
        prob_left_no_lapse = 1.0 / (1.0 + torch.exp(dv_scaled))
        
        # 6. Monte Carlo Average of Probabilities
        avg_prob = torch.mean(prob_left_no_lapse)
        prob_left = avg_prob * (1 - self.lapse) + 0.5 * self.lapse
        
        # Return averaged time for logging/weight update logic
        return prob_left, torch.mean(perceived_time_vec), internal_boundary

    @torch.jit.export
    def update_weights(self, was_correct, is_short_trial, perceived_time, choice_was_left):
        cdf_bound = 1.0 - (1.0 + self.decay_rate * self.current_boundary) * torch.exp(-self.decay_rate * self.current_boundary)
        internal_boundary = self.normalizer * cdf_bound

        unc = 4.0 * self.p_short_block * (1.0 - self.p_short_block)
        was_correct_f = was_correct.float()
        alpha_base = was_correct_f * self.alpha_reward_base + (1 - was_correct_f) * self.alpha_punish_base
        alpha = alpha_base * (1.0 + self.alpha_uncertainty_gain * unc)
        
        # Update weights using the AVERAGE perceived time (expected update)
        is_perc_short = (perceived_time < internal_boundary)
        
        time_ok_bool = (is_perc_short == is_short_trial)
        ctx_ok_bool = ((self.p_short_block > 0.5) == is_short_trial)
        dir_time = 2.0 * time_ok_bool.float() - 1.0
        dir_ctx = 2.0 * ctx_ok_bool.float() - 1.0
        w_time_new = self.weights[0] + alpha * dir_time
        w_ctx_new = self.weights[1] + alpha * dir_ctx
        self.weights = torch.clamp(torch.stack([w_time_new, w_ctx_new]), min=0.0)
        target = torch.where(choice_was_left, torch.tensor(-1.0), torch.tensor(1.0))
        a_hist = torch.where(was_correct, self.alpha_choice_reward, self.alpha_choice_punish)
        target_bias = torch.where(was_correct, target, -target)
        self.last_choice_bias = self.last_choice_bias + a_hist * (target_bias - self.last_choice_bias)
        self._update_context_belief(is_short_trial)