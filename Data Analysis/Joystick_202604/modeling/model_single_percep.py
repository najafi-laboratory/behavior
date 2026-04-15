# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 07:13:19 2025

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
    """
    delays:  (N,) float (may contain NaN)
    rewards: (N,) int in {0,1}
    Returns: dict with indices for omission and non-omission trials.
    """
    delays = np.asarray(delays, float)
    rewards = np.asarray(rewards, int)

    omission_idx = np.where(np.isnan(delays))[0]
    valid_idx    = np.where(~np.isnan(delays))[0]

    a_traj, b_traj, c_traj, lambda_noise, losses, session_cut = [], [], [], [], [], []
    len_session = 0
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
        #lambda_noise.append(info["lambda_noise"])
        #session_cut.append(len_session+len(info["a"]))
        #len_session = len_session + len(info["a"])
        losses.append(info["loss"])

    return {
        "valid_idx": valid_idx,
        "omission_idx": omission_idx,
        "a_traj": np.array(a_traj),
        "b_traj": np.array(b_traj),
        "c_traj": np.array(c_traj),
        "lambda_noise": np.array(lambda_noise),
        "losses": np.array(losses),
        "session_cut": np.array(session_cut)
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
    """Online quadratic perceptron: p = sigmoid(a*x^2 + b*x + c)."""
    def __init__(self, lambda_noise=0.10, lr0=1e-2, lr_decay=0.0, seed=0):
        self.lambda_noise = lambda_noise
        self.lr0 = lr0
        self.lr_decay = lr_decay
        self.rng = np.random.default_rng(seed)
        self.state = OnlineState()

    def _lr(self):
        return self.lr0 / (1.0 + self.lr_decay * max(0, self.state.t))

    def _sample_x(self, delay):
        sd = max(1e-12, self.lambda_noise * max(1e-12, delay))
        return float(self.rng.normal(loc=delay, scale=sd))

    def step(self, delay, reward):
        x = self._sample_x(delay)
        z = self.state.a*x**2 + self.state.b * x + self.state.c
        p = float(_sigmoid(np.array([z]))[0])
        diff = p - float(reward)
        g_a = diff * x**2
        g_b = diff * x
        g_c = diff
        lr = self._lr()
        self.state.a -= lr * g_a
        self.state.b -= lr * g_b
        self.state.c -= lr * g_c
        self.state.t += 1
        loss = - (reward * np.log(max(1e-12, p)) + (1 - reward) * np.log(max(1e-12, 1 - p)))
        return {"loss": float(loss), "p": p, "a": self.state.a, "b": self.state.b, "c": self.state.c, "t": self.state.t}

    def predict(self, delay, n_mc=1, lambda_noise_pred=None, seed_pred=123):
        delay = np.asarray(delay, float)
        _lambda = self.lambda_noise if lambda_noise_pred is None else float(lambda_noise_pred)
        rng = np.random.default_rng(seed_pred)
        acc = np.zeros_like(delay, float)
        for _ in range(int(n_mc)):
            sds = np.maximum(1e-12, _lambda * np.maximum(1e-12, delay))
            x = rng.normal(loc=delay, scale=sds)
            z = self.state.a*x**2 + self.state.b * x + self.state.c
            acc += _sigmoid(z)
        return acc / max(1, int(n_mc))
    
    
# class OnlineQuadPerceptron:
#     """Online quadratic perceptron: p = sigmoid(a*x**2 + b*x + c),
#     with learnable lambda_noise.
#     """
#     def __init__(self,
#                  lambda_noise=0.10,
#                  lr0=1e-2,
#                  lr_decay=0.0,
#                  lr_lambda=1e-4,   # learning rate for noise
#                  seed=0):
#         self.lambda_noise = float(lambda_noise)
#         #print('now')
#         self.lr0 = float(lr0)
#         self.lr_decay = float(lr_decay)
#         self.lr_lambda = float(lr_lambda)
#         self.rng = np.random.default_rng(seed)
#         self.state = OnlineState()

#     def _lr(self):
#         return self.lr0 / (1.0 + self.lr_decay * max(0, self.state.t))

#     def step(self, delay, reward):
#         delay = float(delay)
#         reward = float(reward)

#         # ---- sample x using reparameterization ----
#         delay_clamped = max(1e-12, delay)
#         sigma = max(1e-12, self.lambda_noise * delay_clamped)
#         eps = self.rng.normal(loc=0.0, scale=1.0)
#         x = delay + sigma * eps

#         # ---- forward pass ----
#         z = self.state.a * x**2 + self.state.b * x + self.state.c
#         p = float(_sigmoid(np.array([z]))[0])
#         diff = p - reward  # dL/dz for BCE+sigmoid

#         # ---- gradients for a, b, c (same as before) ----
#         g_a = diff * x**2
#         g_b = diff * x
#         g_c = diff

#         # ---- gradient for lambda_noise ----
#         # dL/dx = (p - y) * (2 a x + b)
#         dL_dx = diff * (2.0 * self.state.a * x + self.state.b)
#         # dx/dlambda ≈ eps * delay_clamped (ignoring clamp when >1e-12)
#         dx_dlambda = eps * delay_clamped
#         g_lambda = dL_dx * dx_dlambda

#         # ---- parameter updates ----
#         lr = self._lr()
#         self.state.a -= lr * g_a
#         self.state.b -= lr * g_b
#         self.state.c -= lr * g_c

#         # separate LR for noise
#         self.lambda_noise -= self.lr_lambda * g_lambda
#         # keep it positive and avoid degeneracy
#         self.lambda_noise = float(max(1e-6, self.lambda_noise))

#         self.state.t += 1

#         # ---- loss ----
#         loss = - (reward * np.log(max(1e-12, p)) +
#                   (1.0 - reward) * np.log(max(1e-12, 1.0 - p)))

#         return {
#             "loss": float(loss),
#             "p": p,
#             "a": self.state.a,
#             "b": self.state.b,
#             "c": self.state.c,
#             "lambda_noise": self.lambda_noise,
#             "t": self.state.t,
#         }

    # def predict(self, delay, n_mc=1, lambda_noise_pred=None, seed_pred=123):
    #     delay = np.asarray(delay, float)
    #     _lambda = self.lambda_noise if lambda_noise_pred is None else float(lambda_noise_pred)
    #     rng = np.random.default_rng(seed_pred)

    #     acc = np.zeros_like(delay, float)
    #     n_mc = max(1, int(n_mc))
    #     for _ in range(n_mc):
    #         delay_clamped = np.maximum(1e-12, delay)
    #         sds = np.maximum(1e-12, _lambda * delay_clamped)
    #         x = rng.normal(loc=delay, scale=sds)
    #         z = self.state.a * x**2 + self.state.b * x + self.state.c
    #         acc += _sigmoid(z)
    #     return acc / n_mc
    
def run_model(delays, rewards,
              lambda_noise=0.10,
              lr0=2e-2,
              lr_decay=5e-4,
              lr_lambda=1e-4,
              seed=0):
    model = OnlineQuadPerceptron(
        lambda_noise=lambda_noise,
        lr0=lr0,
        lr_decay=lr_decay,
        lr_lambda=lr_lambda,
        seed=seed
    )
    info = online_train_stream_with_omissions(
        model,
        np.asarray(delays),
        np.asarray(rewards)
    )
    # e.g., use mean loss over valid trials
    mean_loss = np.mean(info["losses"])
    return mean_loss, model, info

def run_model_fixed_lambda(delays, rewards,
                           lambda_noise=0.10,
                           lr0=2e-2,
                           lr_decay=5e-4,
                           seed=0):
    model = OnlineQuadPerceptron(   # <-- use the FIXED-lambda (commented) version
        lambda_noise=lambda_noise,
        lr0=lr0,
        lr_decay=lr_decay,
        seed=seed
    )
    info = online_train_stream_with_omissions(
        model,
        np.asarray(delays),
        np.asarray(rewards)
    )
    mean_loss = np.mean(info["losses"])
    return mean_loss, model, info


def find_best_lambda_noise(delay_short, label_short, seed_indx=0):
    delays_short  = np.concatenate(delay_short)
    rewards_short = np.concatenate(label_short)

    best_cfg  = None
    best_loss = np.inf

    lambda_list   = [0.02, 0.05, 0.10, 0.20, 0.30]
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
                    seed=seed_indx
                )
                print(f"lambda={lam:.2f}, lr0={lr0:.1e}, decay={lr_decay:.1e}, loss={mean_loss:.4f}")
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_cfg  = (lam, lr0, lr_decay)

    print("Best (lambda, lr0, lr_decay):", best_cfg, "loss=", best_loss)
    return best_cfg, best_loss


def find_best_lr(delay_short, label_short, seed_indx = 0):
    delays_short = np.concatenate(delay_short)
    rewards_short = np.concatenate(label_short)

    best_cfg = None
    best_loss = np.inf

    lr0_list = [5e-3, 1e-2, 2e-2, 5e-2]
    lr_decay_list = [0.0, 1e-4, 5e-4, 1e-3]
    lr_lambda_list = [1e-5, 5e-5, 1e-4]

    for lr0 in lr0_list:
        for lr_decay in lr_decay_list:
            for lr_lambda in lr_lambda_list:
                mean_loss, model_tmp, info_tmp = run_model(
                    delays_short,
                    rewards_short,
                    lambda_noise=0.10,
                    lr0=lr0,
                    lr_decay=lr_decay,
                    lr_lambda=lr_lambda,
                    seed=seed_indx
                )
                print(f"lr0={lr0:.1e}, decay={lr_decay:.1e}, lr_lambda={lr_lambda:.1e}, loss={mean_loss:.4f}")
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_cfg = (lr0, lr_decay, lr_lambda)

    print("Best for short:", best_cfg, "loss=", best_loss)
