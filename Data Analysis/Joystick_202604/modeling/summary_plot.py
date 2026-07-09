# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 11:06:40 2025

@author: saminnaji3
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fitz
def run_summary_all_sessions(results, sessions_input, output_dir_onedrive, pdf_name="summary_all_sessions.pdf"):
    """
    Produce a single-page summary across all sessions:
    rolling accuracy, intervals vs variable windows, predicted success, per-session accuracy.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n_sessions = len(results)
    all_rewards, all_probs, all_actions, all_L, all_U, all_blocks = [], [], [], [], [], []
    session_idx = []
    for i, (res, sess) in enumerate(zip(results, sessions_input)):
        all_rewards.append(res.rewards)
        all_probs.append(res.probs)
        all_actions.append(res.actions)
        # per-trial variable windows (each session has arrays L_t and U_t)
        all_L.append(sess["cues"])
        all_U.append(sess["upper_band"])
        all_blocks.append(sess["blocks"])
        session_idx.append(np.full(len(res.rewards), i))

    rewards = np.concatenate(all_rewards)
    probs = np.concatenate(all_probs)
    actions = np.concatenate(all_actions)
    Ls = np.concatenate(all_L)
    Us = np.concatenate(all_U)
    session_idx = np.concatenate(session_idx)

    def _rolling(x, w=30):
        out = np.full_like(x, np.nan)
        for t in range(len(x)):
            lo = max(0, t - w + 1)
            out[t] = np.nanmean(x[lo:t+1])
        return out

    roll_acc = _rolling(rewards)
    roll_pred = _rolling(probs)
    n_sessions = len(results)

    with PdfPages(os.path.join(output_dir_onedrive, pdf_name)) as pdf:
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle("Model Performance Across Sessions (Variable Windows)", fontsize=13, weight='bold')

        # 1️⃣ rolling accuracy / predicted success
        axs[0].plot(roll_acc, label="Rolling accuracy (30-trial)")
        axs[0].plot(roll_pred, label="Rolling predicted p(success)", alpha=0.7)
        axs[0].set_ylabel("Accuracy / p̂")
        axs[0].legend(); axs[0].grid(alpha=0.3)

        # 2️⃣ intervals vs variable windows
        axs[1].plot(actions, color='k', lw=0.6)
        axs[1].fill_between(range(len(actions)), Ls, Us, color='skyblue', alpha=0.4)
        axs[1].set_ylabel("Interval (s)")
        axs[1].set_title("Intervals vs dynamic tolerance window")
        axs[1].grid(alpha=0.3)

        # mark session borders
        t = 0
        for i, res in enumerate(results):
            axs[0].axvline(t, color='gray', ls='--', lw=0.8, alpha=0.5)
            axs[1].axvline(t, color='gray', ls='--', lw=0.8, alpha=0.5)
            t += len(res.rewards)

        # 3️⃣ per-session mean accuracy
        acc_means = [np.nanmean(r.rewards) for r in results]
        axs[2].bar(range(n_sessions), acc_means, color='cornflowerblue')
        axs[2].set_xticks(range(n_sessions))
        axs[2].set_xticklabels([f"S{i+1}" for i in range(n_sessions)])
        axs[2].set_ylim(0, 1)
        axs[2].set_ylabel("Mean accuracy")
        axs[2].set_xlabel("Session")
        axs[2].grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)
    print(f"✅ saved summary: {pdf_name}")
