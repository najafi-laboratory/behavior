# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 08:31:49 2025

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
def feat_map(phi: np.ndarray, a: float) -> np.ndarray:
    x = np.concatenate([phi.astype(float), np.array([float(a)])])  # length 4
    quad = []
    for i in range(len(x)):
        for j in range(i, len(x)):
            quad.append(x[i] * x[j])
    return np.concatenate([x, np.array(quad, dtype=float)])  # length 14

# Reward label from window (binary), with optional graded shaping
def reward_from_window(a: float,
                       block: int,
                       windows: Dict[int, Tuple[float, float]],
                       graded: bool = False,
                       graceful_edge: float = 0.0,
                       target_by_block: Optional[Dict[int, float]] = None) -> float:
    L, U = windows[int(block)]
    if np.isnan(a):
        return np.nan

    if not graded:
        return float((a >= L) and (a <= U))

    # graded reward: linear decay to 0 across "graceful_edge" outside window (if provided)
    if (a >= L) and (a <= U):
        return 1.0
    if graceful_edge <= 0.0:
        return 0.0
    if a < L:
        return max(0.0, 1.0 - (L - a) / graceful_edge)
    else:  # a > U
        return max(0.0, 1.0 - (a - U) / graceful_edge)

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
    """
    Logistic 'critic' with online SGD updates.
    Action (interval) is chosen to maximize predicted P(success | phi, a)
    via a simple grid search in [action_L, action_U].
    """
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.w = np.zeros(14, dtype=float)  # 3 features + action, degree-2 (no bias)

        # persistent state for features
        self.prev_reward_ewma = 0.5  # start neutral
        self.prev_action = np.nan

    def _phi(self, cue_t: float) -> np.ndarray:
        # Build the 3-feature vector: [cue_t, prev_reward_EWMA, prev_action]
        a_prev = self.prev_action
        if np.isnan(a_prev):
            # if first trial, initialize with the center of the action range
            a_prev = 0.5 * (self.cfg.action_L + self.cfg.action_U)
        return np.array([float(cue_t), float(self.prev_reward_ewma), float(a_prev)], dtype=float)

    def pick_action(self, cue_t: float) -> float:
        # grid search (stable, cheap)
        grid = np.linspace(self.cfg.action_L, self.cfg.action_U, self.cfg.grid_n)
        phi_t = self._phi(cue_t)
        # maximize predicted success probability
        scores = [sigmoid(self.w @ feat_map(phi_t, a)) for a in grid]
        return float(grid[int(np.argmax(scores))])

    def update(self, cue_t: float, a_t: float, y_t: float):
        # Skip invalid feedback
        if np.isnan(y_t) or np.isnan(a_t):
            # still update EWMA state & prev_action bookkeeping
            if not np.isnan(a_t):
                self.prev_action = a_t
            return

        phi_t = self._phi(cue_t)
        z = feat_map(phi_t, a_t)
        p = sigmoid(self.w @ z)

        # logistic gradient (perceptron-style SGD) + L2 + optional forgetting
        grad = (p - y_t) * z + self.cfg.l2 * self.w
        self.w -= self.cfg.lr * grad
        if self.cfg.forget > 0.0:
            self.w *= (1.0 - self.cfg.forget)

        # update internal feature state
        self.prev_reward_ewma = ewma_update(self.prev_reward_ewma, y_t, self.cfg.reward_ewma_alpha)
        self.prev_action = a_t

    def get_params(self) -> np.ndarray:
        return self.w.copy()

# ---------- training wrappers ----------

@dataclass
class TrainResult:
    actions: np.ndarray         # chosen (or observed) actions per trial
    rewards: np.ndarray         # 0/1 (or graded) rewards per trial
    probs:   np.ndarray         # model's predicted success prob for (phi_t, a_t)
    weights: np.ndarray         # w snapshot per trial (T x 14)

def run_session_online(
    chooser: OnlineLogitContinuousChooser,
    cues: np.ndarray,                    # (T,)
    blocks: np.ndarray,                  # (T,) values in {0,1} or any keys in windows
    windows: Dict[int, Tuple[float, float]],
    *,
    mode: str = "on_policy",             # "on_policy" or "off_policy"
    mouse_intervals: Optional[np.ndarray] = None,   # (T,) if off_policy
    use_recorded_when_nan: bool = True,  # in on_policy: if model picks NaN (shouldn't) use data
    graded_reward: Optional[bool] = None,
    graded_edge: Optional[float] = None
) -> TrainResult:
    """
    Online training within one session. Returns trajectories.
    - on_policy: model picks a_t, environment returns reward from window[block_t]
    - off_policy: use mouse_intervals as a_t for update (i.e., fit critic to the animal’s choices)
    """
    T = len(cues)
    actions = np.full(T, np.nan, float)
    rewards = np.full(T, np.nan, float)
    probs   = np.full(T, np.nan, float)
    weights = np.zeros((T, 14), float)

    graded = chooser.cfg.graded_reward if graded_reward is None else graded_reward
    edge   = chooser.cfg.graded_edge if graded_edge is None else graded_edge

    for t in range(T):
        cue_t = float(cues[t])
        block_t = int(blocks[t])

        if mode == "on_policy":
            a_t = chooser.pick_action(cue_t)
        elif mode == "off_policy":
            if mouse_intervals is None:
                raise ValueError("mouse_intervals must be provided for off_policy mode")
            a_t = float(mouse_intervals[t])
        else:
            raise ValueError("mode must be 'on_policy' or 'off_policy'")

        # environment feedback based on the window for this block
        y_t = reward_from_window(a_t, block_t, windows, graded=graded, graceful_edge=edge)

        # record model's predicted success probability at (phi_t, a_t)
        phi_t = chooser._phi(cue_t)
        z = feat_map(phi_t, a_t)
        p = sigmoid(chooser.w @ z)

        # online update
        chooser.update(cue_t, a_t, y_t)

        # log
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
        res = run_session_online(
            chooser, cues, blocks, windows,
            mode=mode,
            mouse_intervals=mouse_iv
        )
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
