# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 20:26:09 2025

@author: timst
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from pprint import pprint

def summarize_behavioral_profile(glm_hmm, state_id, feature_cols):
    weights = glm_hmm.observations.params[state_id][0]  # (D,) for binary model
    profile = {
        "State": state_id,
        "Bias": weights[feature_cols.index("rolling_choice_bias")],
        "Stimulus Sensitivity": weights[feature_cols.index("stim_duration")],
        "RT Influence": weights[feature_cols.index("response_time")],
        "Accuracy Anchoring": weights[feature_cols.index("rolling_accuracy")],
        "Choice History": np.mean([
            weights[feature_cols.index("choice_1back")],
            weights[feature_cols.index("choice_2back")],
            weights[feature_cols.index("choice_3back")]
        ]),
        "Reward History": np.mean([
            weights[feature_cols.index("reward_1back")],
            weights[feature_cols.index("reward_2back")],
            weights[feature_cols.index("reward_3back")]
        ]),
        "Opto Effect": weights[feature_cols.index("is_opto")],
        "Exploration": weights[feature_cols.index("stay_from_1back")] * -1,
        "Expected Bias": weights[feature_cols.index("exp_choice_bias")],
    }
    return profile


def interpret_behavioral_profile(profile):
    label = "Unclassified"
    if profile["Bias"] > 20 and profile["Accuracy Anchoring"] > 100:
        label = "Confident Biased Performer"
    elif profile["Bias"] < -20 and profile["Reward History"] < -20:
        label = "Reward-Averse Explorer"
    elif profile["Opto Effect"] > 20:
        label = "Opto-Driven"
    elif profile["RT Influence"] < -0.05 and profile["Choice History"] > 20:
        label = "Fast Reflexive Repeater"
    elif profile["Stimulus Sensitivity"] > 0.2:
        label = "Stimulus-Driven"
    return label


def interpret_behavioral_profile_tags(profile):
    tags = []
    if profile["Bias"] > 15:
        tags.append("biased")
    if profile["Accuracy Anchoring"] > 50:
        tags.append("stable")
    if profile["RT Influence"] < -0.05:
        tags.append("fast-responder")
    if profile["Reward History"] < -15:
        tags.append("reward-averse")
    if profile["Stimulus Sensitivity"] > 0.2:
        tags.append("stimulus-sensitive")
    if profile["Opto Effect"] > 15:
        tags.append("opto-reactive")
    return tags


def summarize_weights_as_phrase(profile):
    descriptors = []
    if profile["Bias"] > 20:
        descriptors.append("shows consistent choice bias")
    if profile["Accuracy Anchoring"] > 100:
        descriptors.append("strongly tracks recent accuracy")
    if profile["Stimulus Sensitivity"] > 0.2:
        descriptors.append("sensitive to ISI duration")
    if profile["Reward History"] < -25:
        descriptors.append("less responsive to prior rewards")
    if profile["Opto Effect"] > 15:
        descriptors.append("shows increased opto influence")
    if profile["RT Influence"] < -0.1:
        descriptors.append("rapidly responds when repeating")
    return "State characterized by " + ", ".join(descriptors) + "." if descriptors else "Unclear state."


def summarize_all_states(glm_hmm, feature_cols):
    summary = {}
    for k in range(glm_hmm.K):
        profile = summarize_behavioral_profile(glm_hmm, k, feature_cols)
        label = interpret_behavioral_profile(profile)
        tags = interpret_behavioral_profile_tags(profile)
        phrase = summarize_weights_as_phrase(profile)
        summary[k] = {
            "profile": profile,
            "label": label,
            "tags": tags,
            "phrase": phrase
        }
    return summary

def compute_state_occupancy_and_accuracy(session_data):
    """
    Given merged session data (with z_map, y_pred, y_true), compute:
        - % occupancy per state
        - accuracy per state

    Args:
        session_data (dict): session_data_by_date[date]["df"]

    Returns:
        occupancy (dict): {state_id: percent}
        accuracy (dict): {state_id: float}
    """
    df = session_data.copy()
    df = df.dropna(subset=["z_map", "y_pred", "y_true"])

    state_ids = np.unique(df["z_map"].values.astype(int))
    occupancy = {}
    accuracy = {}

    for state in state_ids:
        state_trials = df[df["z_map"] == state]
        occupancy[state] = 100 * len(state_trials) / len(df)
        accuracy[state] = np.mean(state_trials["y_pred"] == state_trials["y_true"])

    return occupancy, accuracy

def summarize_glm_hmm_model(model_output):
    """
    Summarizes GLM-HMM model structure, weights, and state-level behavioral interpretations.
    Args:
        model_output (dict): Output dictionary from get_glm_hmm, must contain:
            - 'glm_hmm': trained model object
            - 'model_results': decoded results per session
            - 'session_data': merged per-session dataframes
            - 'df': full concatenated session dataframe
            - 'feature_cols': list of input features used
    Returns:
        summary (dict): Summary of model structure and state-level characteristics
    """
    print("🔍 Summarizing GLM-HMM model behavioral states...")
    glm_hmm = model_output["glm_hmm"]
    feature_cols = model_output["feature_cols"]
    model_summary = {
        "num_states": glm_hmm.K,
        "num_features": len(feature_cols),
        "feature_cols": feature_cols,
        "states": []
    }

    for state_id in range(glm_hmm.K):
        weights = glm_hmm.observations.params[state_id][0]
        
        print(f"\n📊 State {state_id} weights:")
        for f, w in zip(feature_cols, weights):
            print(f"  {f:>25}: {w:+.3f}")        
                
        profile = summarize_behavioral_profile(glm_hmm, state_id, feature_cols)
        label = interpret_behavioral_profile(profile)
        tags = interpret_behavioral_profile_tags(profile)
        phrase = summarize_weights_as_phrase(profile)

        model_summary["states"].append({
            "state_id": state_id,
            "weights": weights,
            "profile": profile,
            "label": label,
            "tags": tags,
            "phrase": phrase
        })

        print(f"\n🧠 Behavioral Profile for State {state_id}:")
        pprint(profile)

        print(f"🏷️  Label: {label}")
        print(f"🔖 Tags: {', '.join(tags)}")
        print(f"📝 Phrase: {phrase}")

    # Pull a representative session to get occupancy/accuracy
    session_dates = list(model_output["session_data"].keys())
    first_session = model_output["session_data"][session_dates[0]]["df"]
    occupancy, accuracy = compute_state_occupancy_and_accuracy(first_session)
    
    # Attach to each state
    for state_info in model_summary["states"]:
        sid = state_info["state_id"]
        state_info["occupancy"] = occupancy.get(sid, 0.0)
        state_info["accuracy"] = accuracy.get(sid, 0.0)
    
        print(f"\n📈 State {sid} Summary:")
        print(f"   Occupancy: {state_info['occupancy']:.1f}%")
        print(f"   Accuracy : {state_info['accuracy']:.3f}")

    print(f"✅ Summarized GLM-HMM with {model_summary['num_states']} states and {model_summary['num_features']} features")
    return model_summary
