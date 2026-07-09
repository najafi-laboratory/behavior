# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 08:06:23 2025

@author: saminnaji3
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchOnlineQuadPerceptron(nn.Module):
    """
    Online quadratic perceptron:
        z = a*x^2 + b*x + c
        p = sigmoid(z)

    Noise model (reparameterized):
        x = delay + (lambda_noise * clamp(delay)) * eps
    """
    def __init__(self, a0=-1.0, b0=0.0, c0=0.0,
                 lambda_noise=0.10,
                 learn_lambda=False,
                 device="cpu",
                 seed=0):
        super().__init__()
        self.device = torch.device(device)

        # parameters
        self.a = nn.Parameter(torch.tensor(float(a0), device=self.device))
        self.b = nn.Parameter(torch.tensor(float(b0), device=self.device))
        self.c = nn.Parameter(torch.tensor(float(c0), device=self.device))

        self.learn_lambda = bool(learn_lambda)
        if self.learn_lambda:
            # softplus reparam to keep positive
            lam0 = float(lambda_noise)
            lam_raw0 = np.log(np.exp(lam0) - 1.0) if lam0 > 1e-6 else -10.0
            self.lambda_raw = nn.Parameter(torch.tensor(lam_raw0, device=self.device))
        else:
            self.register_buffer("lambda_noise_buf", torch.tensor(float(lambda_noise), device=self.device))

        # RNG
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(seed))

        self.t = 0
        self.omissions = 0

    def lambda_noise(self):
        if self.learn_lambda:
            return F.softplus(self.lambda_raw) + 1e-6
        return self.lambda_noise_buf

    def forward_logits(self, delay, eps=None):
        """
        delay: scalar tensor
        eps: scalar tensor ~ N(0,1)
        Returns: logits z, sampled x
        """
        delay = delay.to(self.device)
        delay_clamped = torch.clamp(delay, min=1e-12)
        lam = self.lambda_noise()
        sigma = torch.clamp(lam * delay_clamped, min=1e-12)

        if eps is None:
            eps = torch.randn((), generator=self._gen, device=self.device)

        x = delay + sigma * eps
        z = self.a * x * x + self.b * x + self.c
        return z, x


def online_train_stream_with_omissions_torch(model, delays, rewards,
                                            lr0=2e-2, lr_decay=5e-4,
                                            learn_lambda=False,
                                            lr_lambda=1e-4,
                                            device="cpu"):
    """
    delays: 1D np array float, may contain NaN
    rewards: 1D np array int in {0,1}
    """
    model.to(device)

    # separate param groups if lambda is learnable
    if learn_lambda:
        base_params = [model.a, model.b, model.c]
        lam_params  = [model.lambda_raw]
        opt = torch.optim.SGD(
            [{"params": base_params, "lr": lr0},
             {"params": lam_params,  "lr": lr_lambda}]
        )
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr0)

    # simple manual lr decay to mirror your numpy schedule
    def current_lr(step):
        return lr0 / (1.0 + lr_decay * max(0, step))

    a_traj, b_traj, c_traj, lam_traj, losses = [], [], [], [], []
    valid_idx, omission_idx = [], []

    for i, d in enumerate(delays):
        if np.isnan(d):
            model.omissions += 1
            omission_idx.append(i)
            continue

        valid_idx.append(i)
        y = float(rewards[i])

        # update lr (param group 0 = abc)
        lr_now = current_lr(model.t)
        opt.param_groups[0]["lr"] = lr_now
        if learn_lambda:
            opt.param_groups[1]["lr"] = lr_lambda  # keep fixed (like your numpy)

        delay_t = torch.tensor(float(d), device=model.device)
        y_t = torch.tensor(float(y), device=model.device)

        opt.zero_grad(set_to_none=True)

        z, _ = model.forward_logits(delay_t)
        # BCE with logits is numerically stable
        loss = F.binary_cross_entropy_with_logits(z, y_t)
        loss.backward()
        opt.step()

        model.t += 1

        a_traj.append(model.a.detach().cpu().item())
        b_traj.append(model.b.detach().cpu().item())
        c_traj.append(model.c.detach().cpu().item())
        lam_traj.append(model.lambda_noise().detach().cpu().item())
        losses.append(loss.detach().cpu().item())

    return {
        "valid_idx": np.array(valid_idx, dtype=int),
        "omission_idx": np.array(omission_idx, dtype=int),
        "a_traj": np.array(a_traj, float),
        "b_traj": np.array(b_traj, float),
        "c_traj": np.array(c_traj, float),
        "lambda_noise": np.array(lam_traj, float),
        "losses": np.array(losses, float),
    }


@torch.no_grad()
def predict_torch(model, delay, n_mc=50, lambda_noise_pred=None, seed_pred=123):
    """
    delay: np array-like
    returns: np array p(reward)
    """
    device = model.device
    delay = torch.tensor(np.asarray(delay, float), device=device)

    if lambda_noise_pred is not None:
        lam = torch.tensor(float(lambda_noise_pred), device=device)
    else:
        lam = model.lambda_noise()

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed_pred))

    acc = torch.zeros_like(delay)
    n_mc = max(1, int(n_mc))

    for _ in range(n_mc):
        delay_clamped = torch.clamp(delay, min=1e-12)
        sigma = torch.clamp(lam * delay_clamped, min=1e-12)
        eps = torch.randn(delay.shape, generator=gen, device=device)
        x = delay + sigma * eps
        z = model.a * x * x + model.b * x + model.c
        acc += torch.sigmoid(z)

    return (acc / n_mc).cpu().numpy()


def run_model_torch(delays, rewards,
                    lambda_noise=0.10,
                    lr0=2e-2,
                    lr_decay=5e-4,
                    learn_lambda=False,
                    lr_lambda=1e-4,
                    seed=0,
                    device="cpu"):
    model = TorchOnlineQuadPerceptron(
        lambda_noise=lambda_noise,
        learn_lambda=learn_lambda,
        device=device,
        seed=seed
    )
    info = online_train_stream_with_omissions_torch(
        model,
        np.asarray(delays, float),
        np.asarray(rewards, int),
        lr0=lr0,
        lr_decay=lr_decay,
        learn_lambda=learn_lambda,
        lr_lambda=lr_lambda,
        device=device
    )
    mean_loss = float(np.mean(info["losses"])) if len(info["losses"]) else float("inf")
    return mean_loss, model, info


def find_best_lr_and_lambda_torch(delay_short, label_short, seed_indx=0, device="cpu"):
    delays_short  = np.concatenate(delay_short)
    rewards_short = np.concatenate(label_short)

    best_cfg, best_loss = None, float("inf")

    lambda_list   = [0.02, 0.05, 0.10, 0.20, 0.30]
    lr0_list      = [5e-3, 1e-2, 2e-2, 5e-2]
    lr_decay_list = [0.0, 1e-4, 5e-4, 1e-3]

    for lam in lambda_list:
        for lr0 in lr0_list:
            for lr_decay in lr_decay_list:
                mean_loss, _, _ = run_model_torch(
                    delays_short, rewards_short,
                    lambda_noise=lam,
                    lr0=lr0,
                    lr_decay=lr_decay,
                    learn_lambda=False,   # <-- fixed noise hyperparameter
                    seed=seed_indx,
                    device=device
                )
                print(f"lambda={lam:.2f}, lr0={lr0:.1e}, decay={lr_decay:.1e}, loss={mean_loss:.4f}")
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_cfg  = (lam, lr0, lr_decay)

    print("Best (lambda, lr0, lr_decay):", best_cfg, "loss=", best_loss)
    return best_cfg, best_loss
