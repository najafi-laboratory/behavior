# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 10:50:59 2025

@author: saminnaji3
"""


import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

# ---------- utilities ----------

def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))

def ewma_update(prev: float, x: float, alpha: float) -> float:
    return alpha * x + (1 - alpha) * prev

# Build degree-2 polynomial feature map including the action and all interactions.
# Base inputs: phi (len 3) and action a (len 1) -> x has len 4
# Output: [x, upper-triangular degree-2 products] -> 4 + 10 = 14 dims (no bias)
def feat_map(phi: np.ndarray, a_std: float) -> np.ndarray:
    # same shape as before: x = [phi(3), a_std] -> degree-2 -> 14 dims
    x = np.concatenate([phi.astype(float), np.array([float(a_std)])])  # len 4
    quad = []
    for i in range(len(x)):
        for j in range(i, len(x)):
            quad.append(x[i] * x[j])
    return np.concatenate([x, np.array(quad, dtype=float)])


# Reward label from window (binary), with optional graded shaping
def reward_from_window_per_trial(a: float, L_t: float, U_t: float,
                                 graded: bool=False, graceful_edge: float=0.0) -> float:
    if np.isnan(a): return np.nan
    if not graded:
        return float((a >= L_t) and (a <= U_t))
    if L_t <= a <= U_t:
        return 1.0
    if graceful_edge <= 0.0:
        return 0.0
    if a < L_t:
        return max(0.0, 1.0 - (L_t - a)/graceful_edge)
    else:
        return max(0.0, 1.0 - (a - U_t)/graceful_edge)


# ---------- model ----------

@dataclass
class ModelConfig:
    # action bounds (seconds)
    action_L: float
    action_U: float
    # grid search over action to pick next interval
    grid_n: int = 61
    # logistic SGD
    lr: float = 0.05
    l2: float = 1e-4
    # EWMA for previous reward feature
    reward_ewma_alpha: float = 0.3
    # optional forgetting (weight decay towards 0 every step)
    forget: float = 0.0  # e.g., 0.001 for slow forgetting; 0 = off
    # use graded rewards near window edges
    graded_reward: bool = False
    graded_edge: float = 0.0  # width outside window for linear decay if graded_reward=True

class OnlineLogitContinuousChooser:
    # ... (unchanged init, same 14-dim w, same _phi)
    eps = 1e-6

    def _std_action(self, a: float, L_t: float, U_t: float) -> float:
        c = 0.5*(L_t+U_t)
        w = max(0.5*(U_t-L_t), self.eps)
        return (a - c)/w

    def pick_action(self, cue_t: float, L_t: float, U_t: float) -> float:
        grid = np.linspace(self.cfg.action_L, self.cfg.action_U, self.cfg.grid_n)
        phi_t = self._phi(cue_t)
        # score using standardized action
        scores = []
        for a in grid:
            a_std = self._std_action(a, L_t, U_t)
            z = feat_map(phi_t, a_std)
            scores.append(sigmoid(self.w @ z))
        return float(grid[int(np.argmax(scores))])

    def update(self, cue_t: float, a_t: float, y_t: float, L_t: float, U_t: float):
        if np.isnan(y_t) or np.isnan(a_t):
            if not np.isnan(a_t):
                self.prev_action = a_t
            return
        phi_t = self._phi(cue_t)
        a_std = self._std_action(a_t, L_t, U_t)
        z = feat_map(phi_t, a_std)
        p = sigmoid(self.w @ z)
        grad = (p - y_t) * z + self.cfg.l2 * self.w
        self.w -= self.cfg.lr * grad
        if self.cfg.forget > 0.0:
            self.w *= (1.0 - self.cfg.forget)
        self.prev_reward_ewma = ewma_update(self.prev_reward_ewma, y_t, self.cfg.reward_ewma_alpha)
        self.prev_action = a_t

# ---------- training wrappers ----------

@dataclass
class TrainResult:
    actions: np.ndarray         # chosen (or observed) actions per trial
    rewards: np.ndarray         # 0/1 (or graded) rewards per trial
    probs:   np.ndarray         # model's predicted success prob for (phi_t, a_t)
    weights: np.ndarray         # w snapshot per trial (T x 14)

def run_session_online_variable_window(
    chooser: OnlineLogitContinuousChooser,
    cues: np.ndarray,               # (T,)
    Ls:  np.ndarray,                # (T,) lower bounds
    Us:  np.ndarray,                # (T,) upper bounds
    *,
    mode: str = "on_policy",
    mouse_intervals: Optional[np.ndarray] = None,
    graded_reward: Optional[bool] = None,
    graded_edge: Optional[float] = None
) -> TrainResult:

    T = len(cues)
    actions = np.full(T, np.nan)
    rewards = np.full(T, np.nan)
    probs   = np.full(T, np.nan)
    weights = np.zeros((T, 14))

    graded = chooser.cfg.graded_reward if graded_reward is None else graded_reward
    edge   = chooser.cfg.graded_edge if graded_edge is None else graded_edge

    for t in range(T):
        cue_t, L_t, U_t = float(cues[t]), float(Ls[t]), float(Us[t])

        if mode == "on_policy":
            a_t = chooser.pick_action(cue_t, L_t, U_t)
        else:
            if mouse_intervals is None:
                raise ValueError("Provide mouse_intervals for off_policy mode")
            a_t = float(mouse_intervals[t])

        y_t = reward_from_window_per_trial(a_t, L_t, U_t, graded=graded, graceful_edge=edge)

        # log predicted prob at the chosen a_t
        phi_t = chooser._phi(cue_t)
        a_std = chooser._std_action(a_t, L_t, U_t)
        p = sigmoid(chooser.w @ feat_map(phi_t, a_std))

        chooser.update(cue_t, a_t, y_t, L_t, U_t)

        actions[t] = a_t
        rewards[t] = y_t
        probs[t]   = p
        weights[t] = chooser.get_params()

    return TrainResult(actions, rewards, probs, weights)


def run_many_sessions(
    sessions: List[Dict[str, np.ndarray]],
    windows: Dict[int, Tuple[float, float]],
    cfg: ModelConfig,
    *,
    mode: str = "on_policy"
) -> List[TrainResult]:
    """
    sessions: list of dicts each with keys:
      - 'cues': (T,), float
      - 'blocks': (T,), int keys present in `windows`
      - optionally 'mouse_intervals': (T,)
    Keeps the model 'warm-started' across sessions to capture across-day learning.
    """
    chooser = OnlineLogitContinuousChooser(cfg)
    results = []
    for sess in sessions:
        cues = sess["cues"]
        blocks = sess["blocks"]
        mouse_iv = sess.get("mouse_intervals", None)
        L_t = sess["cues"]
        U_t = sess["upper_band"]
        res = run_session_online_variable_window(chooser, cues, L_t, U_t, mode=mode, mouse_intervals=mouse_iv)
        results.append(res)
    return results

# ---------- example usage (mocked) ----------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Suppose two blocks with different windows (seconds)
    windows = {
        0: (0.40, 0.93),  # short-valid
        1: (0.93, 3.13),  # long-valid
    }

    cfg = ModelConfig(
        action_L=0.4, action_U=2.2,
        grid_n=61, lr=0.05, l2=1e-4,
        reward_ewma_alpha=0.3,
        forget=0.0,
        graded_reward=False,  # set True if you want graded feedback near edges
        graded_edge=0.10
    )

    # Build a toy session: alternating short/long blocks every 50 trials
    T = 400
    block_len = 50
    blocks = np.repeat([0,1], block_len)
    blocks = np.tile(blocks, T // (2*block_len))

    # Cue is noisy around target (for demo); in your data use the real stimulus
    cue_short, cue_long = 0.4, 0.93
    cues = np.where(blocks==0,
                    rng.normal(cue_short, 0.05, T),
                    rng.normal(cue_long, 0.08, T))

    # Optionally: recorded mouse intervals (for off_policy fitting)
    # Here we fabricate a biased mouse: centered near true, with some lapse
    mouse_intervals = np.where(blocks==0,
                               rng.normal(0.82, 0.10, T),
                               rng.normal(1.58, 0.15, T))
    # ~5% ignore trials -> NaN
    ignore_ix = rng.choice(T, size=int(0.05*T), replace=False)
    mouse_intervals[ignore_ix] = np.nan

    sessions = [
        {"cues": cues, "blocks": blocks, "mouse_intervals": mouse_intervals}
    ]

    # -------- choose ONE of these modes --------
    # A) On-policy: model picks intervals, learns from windowed reward
    results_on = run_many_sessions(sessions, windows, cfg, mode="on_policy")

    # B) Off-policy (imitation-style critic fitting): learn from the animal's produced intervals
    results_off = run_many_sessions(sessions, windows, cfg, mode="off_policy")

    # Quick sanity check
    r_on = np.nanmean(results_on[0].rewards)
    r_off = np.nanmean(results_off[0].rewards)
    print("mean reward (on-policy): ", r_on)
    print("mean reward (off-policy):", r_off)
