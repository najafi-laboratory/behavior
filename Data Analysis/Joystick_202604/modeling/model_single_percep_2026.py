# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 07:56:06 2026

@author: saminnaji3
"""

import numpy as np
def define_input(behavioral_data):
    delay_short = []
    delay_long = []
    label_short = []
    label_long = []
    for sess in range(len(behavioral_data['label'])):
        type_vector = behavioral_data['type'][sess]
        label_vector = behavioral_data['label'][sess]
        delay = behavioral_data['behavior'][sess]
        short_idx = np.where(np.array(type_vector) == 1)[0]
        delay_short.append(np.array(delay)[short_idx])
        label_short.append(np.array(label_vector)[short_idx])
        long_idx = np.where(np.array(type_vector) == 2)[0]
        delay_long.append(np.array(delay)[long_idx])
        label_long.append(np.array(label_vector)[long_idx])
    return [delay_short, delay_long, label_short, label_long]

OMISSION = 2  # separate label for bookkeeping (0=no reward, 1=reward, 2=omission)

def online_train_stream_with_omissions(model, delays, rewards):
    delays = np.asarray(delays, float)
    rewards = np.asarray(rewards, int)

    omission_idx = np.where(np.isnan(delays))[0]
    valid_idx    = np.where(~np.isnan(delays))[0]

    a_traj, b_traj, c_traj, pavg_traj, losses = [], [], [], [], []
    for i in range(len(delays)):
        d = delays[i]
        if np.isnan(d):
            model.state.omissions += 1
            continue

        y = int(rewards[i])
        info = model.step(float(d), y)

        a_traj.append(info["a"])
        b_traj.append(info["b"])
        c_traj.append(info["c"])
        pavg_traj.append(info["p_avg"])
        losses.append(info["loss"])

    return {
        "valid_idx": valid_idx,
        "omission_idx": omission_idx,
        "a_traj": np.array(a_traj),
        "b_traj": np.array(b_traj),
        "c_traj": np.array(c_traj),
        "p_avg": np.array(pavg_traj),
        "losses": np.array(losses),
    }


from dataclasses import dataclass

def _sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

@dataclass
class OnlineState:
    a: float = -1.0 
    b: float = 0.0
    c: float = 0.0
    t: int = 0
    omissions: int = 0
    
    
class OnlineQuadPerceptron:
    """Online quadratic perceptron with MC-averaged multiplicative Gaussian noise:
       x = delay + eps*(lambda_noise*delay), eps ~ N(0,1)
    """
    def __init__(self, lambda_noise=0.10, lr0=1e-2, lr_decay=0.0, seed=0, n_mc_train=100):
        self.lambda_noise = float(lambda_noise)   # <-- hyperparameter to tune
        self.lr0 = float(lr0)
        self.lr_decay = float(lr_decay)
        self.rng = np.random.default_rng(seed)
        self.state = OnlineState()
        self.n_mc_train = int(n_mc_train)

    def _lr(self):
        return self.lr0 / (1.0 + self.lr_decay * max(0, self.state.t))

    def step(self, delay, reward):
        delay = float(delay)
        y = float(reward)

        # --- Monte Carlo noise: x_k = delay + eps_k*(lambda*delay) ---
        d = max(1e-12, delay)  # clamp to avoid degeneracy at 0
        eps = self.rng.normal(loc=0.0, scale=1.0, size=self.n_mc_train)
        x = delay + eps * (self.lambda_noise * d)  # your requested form

        # --- forward for all samples ---
        z = self.state.a * x**2 + self.state.b * x + self.state.c
        p = _sigmoid(z)  # vector of size n_mc

        # --- average probability (what you called p_avg) ---
        p_avg = float(np.mean(p))

        # --- Monte Carlo averaged loss (cross entropy) ---
        # (average over samples, for this trial)
        p_clamped = np.clip(p, 1e-12, 1.0 - 1e-12)
        loss_samples = -(y * np.log(p_clamped) + (1.0 - y) * np.log(1.0 - p_clamped))
        loss_avg = float(np.mean(loss_samples))

        # --- gradients: average over MC samples ---
        # For BCE with sigmoid: dL/dz = (p - y)
        diff = (p - y)  # vector
        g_a = float(np.mean(diff * (x**2)))
        g_b = float(np.mean(diff * x))
        g_c = float(np.mean(diff))

        # --- update ---
        lr = self._lr()
        self.state.a -= lr * g_a
        self.state.b -= lr * g_b
        self.state.c -= lr * g_c
        self.state.t += 1

        return {
            "loss": loss_avg,
            "p_avg": p_avg,
            "a": self.state.a,
            "b": self.state.b,
            "c": self.state.c,
            "t": self.state.t
        }

    def predict(self, delay, n_mc=100, lambda_noise_pred=None, seed_pred=123):
        delay = np.asarray(delay, float)
        _lambda = self.lambda_noise if lambda_noise_pred is None else float(lambda_noise_pred)
        rng = np.random.default_rng(seed_pred)

        d = np.maximum(1e-12, delay)
        eps = rng.normal(loc=0.0, scale=1.0, size=(int(n_mc),) + delay.shape)
        x = delay[None, ...] + eps * (_lambda * d)[None, ...]  # (n_mc, ...)

        z = self.state.a * x**2 + self.state.b * x + self.state.c
        p = _sigmoid(z)
        return np.mean(p, axis=0)


def run_model_fixed_lambda(delays, rewards,
                           lambda_noise=0.10,
                           lr0=2e-2,
                           lr_decay=5e-4,
                           seed=0,
                           n_mc_train=100):
    model = OnlineQuadPerceptron(
        lambda_noise=lambda_noise,
        lr0=lr0,
        lr_decay=lr_decay,
        seed=seed,
        n_mc_train=n_mc_train
    )
    info = online_train_stream_with_omissions(
        model,
        np.asarray(delays),
        np.asarray(rewards)
    )
    mean_loss = np.mean(info["losses"])
    return mean_loss, model, info


def find_best_lambda_noise_mc(delay_short, label_short, seed_indx=0, n_mc_train=100):
    delays_short  = np.concatenate(delay_short)
    rewards_short = np.concatenate(label_short)

    best_cfg  = None
    best_loss = np.inf

    # search space (edit as you like)
    lambda_list   = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
    lr0_list      = [5e-3, 1e-2, 2e-2, 5e-2]
    lr_decay_list = [0.0, 1e-4, 5e-4, 1e-3]

    for lam in lambda_list:
        for lr0 in lr0_list:
            for lr_decay in lr_decay_list:
                mean_loss, _, _ = run_model_fixed_lambda(
                    delays_short,
                    rewards_short,
                    lambda_noise=lam,
                    lr0=lr0,
                    lr_decay=lr_decay,
                    seed=seed_indx,
                    n_mc_train=n_mc_train
                )
                print(f"lambda={lam:.3f}, lr0={lr0:.1e}, decay={lr_decay:.1e}, loss={mean_loss:.4f}")
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_cfg  = (lam, lr0, lr_decay)

    print("Best (lambda, lr0, lr_decay):", best_cfg, "loss=", best_loss)
    return best_cfg, best_loss

