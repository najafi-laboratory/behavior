# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 22:29:31 2025

@author: timst
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

def plot_model_metadata_box(model_output, model_summary, mouse_id=None):
    """
    Plots a summary metadata box for the GLM-HMM model and its context.
    
    Args:
        model_output (dict): Output from get_glm_hmm()
        model_summary (dict): Output from summarize_glm_hmm_model()
        mouse_id (str): Optional override for mouse name
    """
    glm_hmm = model_output["glm_hmm"]
    df = model_output["df"]
    session_dates = sorted(df["date"].unique())
    n_sessions = len(session_dates)
    n_trials = len(df)

    if mouse_id is None:
        mouse_id = df["mouse_id"].iloc[0]

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    text_lines = [
        f"🐭 Mouse ID         :  {mouse_id}",
        f"📅 Sessions used    :  {n_sessions}  ({session_dates[0]} to {session_dates[-1]})",
        f"🧪 Total Trials     :  {n_trials}",
        f"🔐 Hidden States    :  {glm_hmm.K}",
        f"🧠 Model Features   :  {model_summary['num_features']}",
        f"💾 Generated        :  {now}",
    ]

    for i, line in enumerate(text_lines):
        ax.text(0.01, 1 - i * 0.15, line, fontsize=11, verticalalignment='top')

    ax.set_title("📄 Model Metadata", fontsize=14, loc='left', pad=10)

    plt.tight_layout()
    plt.show()

