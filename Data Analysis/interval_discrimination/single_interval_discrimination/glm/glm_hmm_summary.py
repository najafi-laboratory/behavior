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
import matplotlib.pyplot as plt
import seaborn as sns

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


# def interpret_behavioral_profile(profile):
#     label = "Unclassified"
#     if profile["Bias"] > 20 and profile["Accuracy Anchoring"] > 100:
#         label = "Confident Biased Performer"
#     elif profile["Bias"] < -20 and profile["Reward History"] < -20:
#         label = "Reward-Averse Explorer"
#     elif profile["Opto Effect"] > 20:
#         label = "Opto-Driven"
#     elif profile["RT Influence"] < -0.05 and profile["Choice History"] > 20:
#         label = "Fast Reflexive Repeater"
#     elif profile["Stimulus Sensitivity"] > 0.2:
#         label = "Stimulus-Driven"
#     return label

def interpret_behavioral_profile(profile):
    """
    Generate a descriptive label from a state profile using fuzzy thresholds and tag intersections.
    """

    tags = []

    # Fuzzy binning thresholds (tune as needed)
    bias = profile["Bias"]
    anchor = profile["Accuracy Anchoring"]
    stim = profile["Stimulus Sensitivity"]
    reward = profile["Reward History"]
    opto = profile["Opto Effect"]
    rt = profile["RT Influence"]
    choice = profile["Choice History"]
    exp_bias = profile["Expected Bias"]

    # Bias profile
    if bias > 15:
        tags.append("strongly biased")
    elif bias > 5:
        tags.append("moderately biased")
    elif bias < -10:
        tags.append("negatively biased")

    # Accuracy anchoring
    if anchor > 150:
        tags.append("highly stable")
    elif anchor > 50:
        tags.append("tracks recent performance")

    # Stimulus driven
    if stim > 0.25:
        tags.append("strongly stimulus-driven")
    elif stim > 0.1:
        tags.append("mildly stimulus-driven")

    # Reward sensitivity
    if reward < -30:
        tags.append("reward-averse")
    elif reward > 30:
        tags.append("reward-sensitive")

    # Opto sensitivity
    if opto > 15:
        tags.append("opto-reactive")

    # RT influence
    if rt < -0.05:
        tags.append("fast-response")
    elif rt > 0.05:
        tags.append("hesitant-response")

    # Choice history dependence
    if choice > 15:
        tags.append("strong repeater")
    elif choice < -15:
        tags.append("alternator")

    # Expected bias adds subtlety
    if exp_bias > 50:
        tags.append("internally guided")
    elif exp_bias < -30:
        tags.append("externally hesitant")

    # Combine into high-level label
    if "opto-reactive" in tags and "strongly stimulus-driven" in tags:
        label = "Opto- and Stimulus-Driven"
    elif "strongly biased" in tags and "highly stable" in tags:
        label = "Stable Confident Performer"
    elif "reward-averse" in tags and "alternator" in tags:
        label = "Exploratory Avoider"
    elif "fast-response" in tags and "strong repeater" in tags:
        label = "Reflexive Repeater"
    elif len(tags) >= 3:
        label = "Mixed Strategy Mode"
    elif len(tags) >= 1:
        label = tags[0].capitalize()
    else:
        label = "Undifferentiated Mode"

    return label


# def interpret_behavioral_profile_tags(profile):
#     tags = []
#     if profile["Bias"] > 15:
#         tags.append("biased")
#     if profile["Accuracy Anchoring"] > 50:
#         tags.append("stable")
#     if profile["RT Influence"] < -0.05:
#         tags.append("fast-responder")
#     if profile["Reward History"] < -15:
#         tags.append("reward-averse")
#     if profile["Stimulus Sensitivity"] > 0.2:
#         tags.append("stimulus-sensitive")
#     if profile["Opto Effect"] > 15:
#         tags.append("opto-reactive")
#     return tags

# def interpret_behavioral_profile_tags(profile):
#     """
#     Return descriptive tags and fuzzy scores from a behavioral profile.
#     Scores are scaled between 0 (absent) and 1 (strongly present).
#     """

#     def scale_score(value, low_thresh, high_thresh):
#         """Fuzzy scaling: 0 below low, 1 above high, linear in between"""
#         if value <= low_thresh:
#             return 0.0
#         elif value >= high_thresh:
#             return 1.0
#         else:
#             return (value - low_thresh) / (high_thresh - low_thresh)

#     scores = {
#         "Bias": scale_score(abs(profile["Bias"]), 5, 20),
#         "Accuracy Anchoring": scale_score(profile["Accuracy Anchoring"], 50, 150),
#         "Stimulus Sensitivity": scale_score(profile["Stimulus Sensitivity"], 0.05, 0.25),
#         "Reward Aversion": scale_score(-profile["Reward History"], 5, 30),
#         "Opto Reactivity": scale_score(profile["Opto Effect"], 5, 20),
#         "Fast Response": scale_score(-profile["RT Influence"], 0.02, 0.08),
#         "Repetition Bias": scale_score(profile["Choice History"], 5, 20),
#         "Alternation Bias": scale_score(-profile["Choice History"], 5, 20),
#         "Expected Bias": scale_score(abs(profile["Expected Bias"]), 10, 50)
#     }

#     tags = [tag for tag, score in scores.items() if score > 0.5]

#     return tags, scores

# Updated tag scoring function
def interpret_behavioral_profile_tags_soft(profile, verbose=False):
    tags = {}
    scores = {}

    # Bias
    scores["Bias"] = profile["Bias"]
    if profile["Bias"] > 20:
        tags["Bias"] = "strong"
    elif profile["Bias"] > 10:
        tags["Bias"] = "moderate"

    # Accuracy anchoring
    scores["Accuracy Anchoring"] = profile["Accuracy Anchoring"]
    if profile["Accuracy Anchoring"] > 100:
        tags["Accuracy Anchoring"] = "strong"
    elif profile["Accuracy Anchoring"] > 50:
        tags["Accuracy Anchoring"] = "moderate"

    # Reward history
    scores["Reward Aversion"] = profile["Reward History"]
    if profile["Reward History"] < -25:
        tags["Reward Aversion"] = "strong"
    elif profile["Reward History"] < -15:
        tags["Reward Aversion"] = "moderate"

    # Opto effect
    scores["Opto Effect"] = profile["Opto Effect"]
    if profile["Opto Effect"] < -30:
        tags["Opto Suppressed"] = "strong"
    elif profile["Opto Effect"] > 30:
        tags["Opto Driven"] = "strong"

    # Exploration
    scores["Exploration"] = profile["Exploration"]
    if profile["Exploration"] > 30:
        tags["Exploratory"] = "high"
    elif profile["Exploration"] > 15:
        tags["Exploratory"] = "moderate"
    elif profile["Exploration"] < -10:
        tags["Stable"] = "moderate"

    # RT
    scores["RT Influence"] = profile["RT Influence"]
    if profile["RT Influence"] < -0.05:
        tags["Fast"] = "yes"
    elif profile["RT Influence"] > 0.05:
        tags["Slow"] = "yes"

    # Stimulus
    scores["Stimulus Sensitivity"] = profile["Stimulus Sensitivity"]
    if profile["Stimulus Sensitivity"] > 0.2:
        tags["Stimulus Sensitive"] = "yes"

    if verbose:
        for k, v in scores.items():
            print(f"{k:<25}: {v:+.2f}")

    return tags, scores

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

def summarize_weights_as_phrase_v2(profile):
    phrases = []

    if profile["Reward History"] < -25:
        phrases.append("shows strong reward aversion")
    elif profile["Reward History"] < -15:
        phrases.append("is moderately influenced by past rewards")

    if profile["Opto Effect"] < -30:
        phrases.append("exhibits strong suppression from optogenetic stimulation")
    elif profile["Opto Effect"] > 30:
        phrases.append("is heavily driven by optogenetic input")

    if profile["Exploration"] > 30:
        phrases.append("tends toward exploration and switch-like behavior")
    elif profile["Exploration"] < -10:
        phrases.append("favors repetitive or stable strategies")

    if profile["Accuracy Anchoring"] > 100:
        phrases.append("adapts tightly to recent performance")
    elif profile["Accuracy Anchoring"] < -50:
        phrases.append("operates independently of recent accuracy")

    if profile["RT Influence"] < -0.05:
        phrases.append("makes rapid decisions")
    elif profile["RT Influence"] > 0.05:
        phrases.append("delays decisions under uncertainty")

    if profile["Stimulus Sensitivity"] > 0.2:
        phrases.append("relies heavily on stimulus duration")

    return "🧠 State: " + ", ".join(phrases) if phrases else "🧠 State is difficult to interpret."

def sigmoid(x, center=0, scale=1):
    return 1 / (1 + np.exp(-(x - center) / scale))

def get_behavioral_tag_scores(profile):
    """
    Compute soft tag scores from a behavioral profile using fuzzy rules.
    Returns:
        dict of {tag_name: score [0–1]}
    """
    scores = {}

    # ---- Bias-related Tags ----
    scores["Bias"] = sigmoid(profile["Bias"], center=15, scale=5)
    scores["Alternation Bias"] = sigmoid(-profile["Bias"], center=15, scale=5)

    # ---- Accuracy Anchoring ----
    scores["Accuracy Anchoring"] = sigmoid(profile["Accuracy Anchoring"], center=50, scale=20)

    # ---- RT Influence ----
    scores["Fast Response"] = sigmoid(-profile["RT Influence"], center=0.03, scale=0.01)
    scores["Slow Response"] = sigmoid(profile["RT Influence"], center=0.03, scale=0.01)

    # ---- Stimulus Sensitivity ----
    scores["Stimulus Sensitivity"] = sigmoid(profile["Stimulus Sensitivity"], center=0.1, scale=0.05)

    # ---- Exploration (inverse of stay tendency) ----
    scores["Exploration"] = sigmoid(profile["Exploration"], center=10, scale=5)
    scores["Repeating"] = sigmoid(-profile["Exploration"], center=10, scale=5)

    # ---- Choice and Reward History ----
    scores["History Driven"] = sigmoid(abs(profile["Choice History"]), center=10, scale=5)
    scores["Reward Seeking"] = sigmoid(profile["Reward History"], center=15, scale=5)
    scores["Reward Aversion"] = sigmoid(-profile["Reward History"], center=15, scale=5)

    # ---- Opto Reactivity ----
    scores["Opto Reactive"] = sigmoid(abs(profile["Opto Effect"]), center=10, scale=5)

    # ---- High Certainty (combination idea) ----
    bias = abs(profile["Bias"])
    certainty = np.sqrt(bias**2 + profile["Accuracy Anchoring"]**2)
    scores["High Certainty"] = sigmoid(certainty, center=100, scale=30)

    return scores

# Tag heatmap plot
def plot_behavioral_tag_scores_heatmap(state_profiles):
    tag_matrix = []
    state_labels = []
    all_tags = sorted({
        tag for profile in state_profiles for tag in profile["tags"].keys()
    })

    for profile in state_profiles:
        row = [profile["scores"].get(tag, 0) for tag in all_tags]
        tag_matrix.append(row)
        state_labels.append(f"State {profile['state_id']}")

    tag_matrix = np.array(tag_matrix)

    plt.figure(figsize=(10, 5))
    sns.heatmap(tag_matrix, annot=True, fmt=".1f", cmap="coolwarm",
                xticklabels=all_tags, yticklabels=state_labels)
    plt.title("Behavioral Tag Soft Scores per State")
    plt.tight_layout()
    plt.show()

def run_behavioral_tag_score_heatmap(model_summary, figsize=(10, 6), title=None):
    """
    Generate and plot a heatmap of fuzzy behavioral tag scores across all states.

    Args:
        model_summary (dict): Output from summarize_glm_hmm_model().
        figsize (tuple): Size of the figure.
        title (str): Optional title for the heatmap.
    """
    print("📊 Compiling behavioral tag scores from model summary...")

    all_tag_scores = {}
    for state_info in model_summary["states"]:
        state_id = state_info["state_id"]
        scores = state_info["tag_scores"]
        all_tag_scores[f"State {state_id}"] = scores

    # Convert to DataFrame
    df_scores = pd.DataFrame(all_tag_scores).T  # rows = states

    # Order columns by mean descending
    df_scores = df_scores[df_scores.mean().sort_values(ascending=False).index]

    print("\n📋 Tag Score Matrix:")
    print(df_scores)

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        df_scores, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=0, vmax=1,
        cbar_kws={"label": "Fuzzy Tag Score"}
    )
    plt.title(title or "Behavioral Tag Scores by State")
    plt.xlabel("Behavioral Tags")
    plt.ylabel("GLM-HMM State")
    plt.tight_layout()
    plt.show()

# def run_behavioral_tag_score_heatmap(model_summary, session_date=None, mouse_id=None):
#      """
#      Generate and plot a heatmap of fuzzy behavioral tag scores across all states.
     
#      Args:
#          model_summary (dict): Output from summarize_glm_hmm_model().
#          figsize (tuple): Size of the figure.
#          title (str): Optional title for the heatmap.
#      """
#      print("📊 Computing behavioral tag scores across states...")
     
#      all_tag_scores = {}
#      for state_info in model_summary["states"]:
#          state_id = state_info["state_id"]
#          profile = state_info["profile"]
#          tag_scores = get_behavioral_tag_scores(profile)
#          all_tag_scores[f"State {state_id}"] = tag_scores
     
#      df_scores = pd.DataFrame(all_tag_scores).T  # states as rows
    
#      # Sort tags by mean score for easier viewing
#      df_scores = df_scores[df_scores.columns[df_scores.mean().sort_values(ascending=False).index]]
    
#      figsize=(10, 5)
#      plt.figure(figsize=figsize)
#      sns.heatmap(df_scores, annot=True, cmap="coolwarm", vmin=0, vmax=1, cbar_kws={"label": "Tag Strength"})
#      plt.title("Behavioral Tag Scores by State")
#      plt.xlabel("Behavioral Tag")
#      plt.ylabel("GLM-HMM State")
#      plt.tight_layout()
#      plt.show()
    
   
    # """
    # Computes tag scores from behavioral profiles and plots heatmap.

    # Args:
    #     model_summary (dict): Output from summarize_glm_hmm_model(...)
    #     session_date (str): Optional annotation
    #     mouse_id (str): Optional annotation
    # """
    # print("\n📊 Generating behavioral tag score heatmap...")

    # # Compute tag scores for each state
    # profiles = []
    # for state_info in model_summary["states"]:
    #     profile = state_info["profile"]
    #     tag_scores = get_behavioral_tag_scores(profile)
    #     pprint(tag_scores)
    #     profiles.append({"State": state_info["state_id"], "tags": tag_scores})

    # # Plot
    # plot_behavioral_tag_scores_heatmap(
    #     profiles,
    #     session_date=session_date,
    #     mouse_id=mouse_id
    # )

def summarize_all_states(glm_hmm, feature_cols):
    summary = {}
    for k in range(glm_hmm.K):
        profile = summarize_behavioral_profile(glm_hmm, k, feature_cols)
        label = interpret_behavioral_profile(profile)
        # tags = interpret_behavioral_profile_tags(profile)
        tags, scores = interpret_behavioral_profile_tags_soft(profile)        
        # phrase = summarize_weights_as_phrase(profile)
        phrase = summarize_weights_as_phrase_v2(profile)        
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
        # tags, scores = interpret_behavioral_profile_tags(profile)
        tags, scores = tags, scores = interpret_behavioral_profile_tags_soft(profile)
        phrase = summarize_weights_as_phrase(profile)

        model_summary["states"].append({
            "state_id": state_id,
            "weights": weights,
            "profile": profile,
            "label": label,
            "tags": tags,
            "tag_scores": scores,
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
    
    # Collect profiles
    profiles = [state_info["profile"] for state_info in model_summary["states"]]
    
    # Plot
    run_behavioral_tag_score_heatmap(
        model_summary,
    )
    
    return model_summary
