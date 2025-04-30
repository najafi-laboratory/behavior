# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 18:15:54 2025

@author: timst
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp

import ssm
from ssm.observations import InputDrivenObservations
from ssm.util import find_permutation
print(ssm.__file__)

print_debug = 1

def get_session_df(session_data, session_idx):
    """
    Process licks for a single session.
    
    Args:
        session_data (dict): Contains left/right lick times and opto flags for trials.
    
    Returns:
        dict: Processed licks categorized into left/right and opto/non-opto.
    """
    processed_dec = []
    
    
    raw_data = session_data['raw']
    
    numTrials = raw_data[session_idx]['nTrials']
    
    outcomes_time = session_data['outcomes_time']
    outcome_time = outcomes_time[session_idx]
        
    trial_types = session_data['trial_type'][session_idx]
    opto_flags = session_data['opto_trial'][session_idx]
    
    opto_encode = np.nan
    
    for trial in range(numTrials):
           
        licks = {}
        valve_times = []
        rewarded = 0
        isNaive = 0
        no_lick = 0
        
        alignment = 0
        # alignment = outcome_time[trial]
        alignment = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['WindowChoice'][0]        
        
        if not 'Port1In' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_left_start'] = [np.float64(np.nan)]
        elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In'], (float, int)):
            licks['licks_left_start'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']]
        else:
            licks['licks_left_start'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']
            
        if not 'Port1Out' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_left_stop'] = [np.float64(np.nan)]
        elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out'], (float, int)):
            licks['licks_left_stop'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']]
        else:
            licks['licks_left_stop'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']    
    
        if not 'Port3In' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_right_start'] = [np.float64(np.nan)]
        elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In'], (float, int)):
            licks['licks_right_start'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']]
        else:
            licks['licks_right_start'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']       
        
        if not 'Port3Out' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_right_stop'] = [np.float64(np.nan)]
        elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out'], (float, int)):
            licks['licks_right_stop'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']]
        else:
            licks['licks_right_stop'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']

        ###
  
        # np.array([x - alignment for x in licks['licks_left_start']])
        licks['licks_left_start'] = [(x - alignment)*1000 for x in licks['licks_left_start']]
        licks['licks_left_stop'] = [(x - alignment)*1000 for x in licks['licks_left_stop']]
        licks['licks_right_start'] = [(x - alignment)*1000 for x in licks['licks_right_start']]
        licks['licks_right_stop'] = [(x - alignment)*1000 for x in licks['licks_right_stop']]
  
        # check for licks or spout touches before choice window
        # if licks['licks_left_start'][0] < -0.1:
        #     licks['licks_left_start'] = [np.float64(np.nan)]
        # if licks['licks_left_stop'][0] < -0.1:
        #     licks['licks_left_stop'] = [np.float64(np.nan)]
        # if licks['licks_right_start'][0] < -0.1:
        #     licks['licks_right_start'] = [np.float64(np.nan)]
        # if licks['licks_right_stop'][0] < -0.1:
        #     licks['licks_right_stop'] = [np.float64(np.nan)]            
    
        trial_type = "left" if trial_types[trial] == 1 else "right" 
        
        # Track valve open/close times for the trial (start/stop)
        if 'Reward' in raw_data[session_idx]['RawEvents']['Trial'][trial]['States']:
            is_naive = 0
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][0] - alignment)
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][1] - alignment)
            if not np.isnan(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][0]):
                rewarded = 1
            else:
                rewarded = 0
        elif 'NaiveRewardDeliver' in raw_data[session_idx]['RawEvents']['Trial'][trial]['States']:  
            is_naive = 1
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['NaiveRewardDeliver'][0] - alignment)
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['NaiveRewardDeliver'][1] - alignment)
            if not np.isnan(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['NaiveRewardDeliver'][0]):
                rewarded = 1
            else:
                rewarded = 0
        else:
            print('what this????')
            valve_times.append(np.float64(np.nan))
            valve_times.append(np.float64(np.nan))
            
        if 'DidNotChoose' in raw_data[session_idx]['RawEvents']['Trial'][trial]['States']:  
            if not np.isnan(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['DidNotChoose'][0]):
                no_lick = 1
                licks['licks_left_start'] = [np.float64(-1)]
                licks['licks_left_stop'] = [np.float64(-1)]
                licks['licks_right_start'] = [np.float64(-1)]
                licks['licks_right_stop'] = [np.float64(-1)]

                
            
        is_opto = opto_flags[trial]   
        
        if is_opto and not is_naive:
            opto_encode = 0
            
        isi = session_data['isi_post_emp'][session_idx][trial]
        
        move_correct_spout = session_data['move_correct_spout_flag'][session_idx][trial]
                  
        processed_dec.append({
            "trial_index": trial,
            "trial_side": trial_type,
            "stim_duration": isi,
            "is_opto": is_opto,
            "is_naive": is_naive,
            "rewarded": rewarded,
            "no_lick": no_lick,
            "opto_encode": opto_encode,
            "move_correct_spout": move_correct_spout,
            "licks_left_start": licks['licks_left_start'],
            "licks_left_stop": licks['licks_left_stop'],
            "licks_right_start": licks['licks_right_start'],
            "licks_right_stop": licks['licks_right_stop'],
            "valve_start": valve_times[0],
            "valve_stop": valve_times[1]
        })
        
        opto_encode = opto_encode + 1
        
    return pd.DataFrame(processed_dec)

def get_earliest_lick(row):
    left = row['licks_left_start']
    right = row['licks_right_start']

    all_licks = []

    if isinstance(left, list):
        all_licks.extend([v for v in left if not np.isnan(v)])
    if isinstance(right, list):
        all_licks.extend([v for v in right if not np.isnan(v)])

    return min(all_licks) if all_licks else np.nan   

# 1 = mouse chose right
# 0 = mouse chose left
# We'll infer based on whether rewarded and which side was correct
def infer_choice(row):
    if row['rewarded'] == 1:
        return row['is_right']  # Mouse must have chosen correctly
    else:
        return 1 - row['is_right']  # Mouse chose the wrong side

def add_model_features(df, exp_alpha=0.7):
    """
    Adds features to a trial-by-trial dataframe for modeling behavior.
    
    Parameters:
        df (pd.DataFrame): Combined trial data across sessions.
    
    Returns:
        pd.DataFrame: DataFrame with new model features.
    """
    df = df.copy()

    # ---- Crecombinase Features ----
    # Encode trial type    
    df['is_right'] = (df['trial_side'] == 'right').astype(int)
    df['mouse_choice'] = df.apply(infer_choice, axis=1)
    # Opto flag as int
    df['is_opto'] = df['is_opto'].astype(int)


    # Normalize trial index within session
    df['norm_trial_index'] = df.groupby('session_id')['trial_index'].transform(
        lambda x: x / x.max()
    )

    # # ---- Optional / Temporal Dynamics ----
    # # Previous choice (encoded as 1 = right, 0 = left, np.nan for first trial of session)
    # df['prev_choice'] = df.groupby('session_id')['is_right'].shift(1)
    # # Previous reward (1 = rewarded, 0 = not rewarded)
    # df['prev_reward'] = df.groupby('session_id')['rewarded'].shift(1)
    # # Previous opto
    # df['prev_opto'] = df.groupby('session_id')['is_opto'].shift(1)
    
    # --- Add multi-back history features ---
    for n in range(1, 4):
        df[f'choice_{n}back'] = df.groupby('session_id')['is_right'].shift(n)
        df[f'reward_{n}back'] = df.groupby('session_id')['rewarded'].shift(n)
        df[f'opto_{n}back'] = df.groupby('session_id')['is_opto'].shift(n)      
    
    # --- Stay vs switch from previous choice ---
    df['stay_from_1back'] = df['is_right'] == df['choice_1back']
    df['stay_from_1back'] = df['stay_from_1back'].astype(float)  # Make it 0.0 / 1.0 (or NaN)


    # ---- Response Times ----
    df['lick_time'] = df.apply(get_earliest_lick, axis=1)   
    df['response_time'] = df['lick_time']     
    # if 'lick_time_stim' in df.columns:
    #     df['rt_from_stim'] = df['lick_time_stim']
    # if 'lick_time_choice' in df.columns:
    #     df['rt_from_choice'] = df['lick_time_choice']        


    # Interaction: ISI x Opto
    if 'stim_duration' in df.columns:
        df['isi_opto_interaction'] = df['stim_duration'] * df['is_opto']

    # Rolling average of reward over last 5 trials (behavioral state estimate)
    df['rolling_accuracy'] = df.groupby('session_id')['rewarded'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

    # --- Rolling choice bias (last 5) ---
    df['rolling_choice_bias'] = df.groupby('session_id')['is_right'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

    # --- Exponential decay choice bias (alpha-weighted) ---
    def exp_decay(x, alpha=exp_alpha):
        ewma = [np.nan]  # First trial has no history
        for i in range(1, len(x)):
            prev = ewma[-1]
            if np.isnan(prev):
                ewma.append(x.iloc[i - 1])
            else:
                ewma.append(alpha * x.iloc[i - 1] + (1 - alpha) * prev)
        return pd.Series(ewma, index=x.index)

    df['exp_choice_bias'] = df.groupby('session_id')['is_right'].transform(
        lambda x: exp_decay(x, alpha=exp_alpha)
    )

    return df

def preprocess_for_model(df, feature_cols, target_col='mouse_choice', dropna=True, scale=True):
    """
    Prepares dataframe for modeling:
    - Drops NaNs or fills
    - Scales numeric features
    - Returns X, y for training

    Parameters:
        df (pd.DataFrame): DataFrame with model features
        feature_cols (list): Feature column names to use
        target_col (str): Column to predict
        dropna (bool): Whether to drop NaNs (True) or fill (False)
        scale (bool): Whether to z-score scale features

    Returns:
        X (np.ndarray): Model input features
        y (np.ndarray): Target vector
        scaler (StandardScaler or None): Scaler used (for inverse later)
    """
    df = df.copy()

    # Subset to modeling columns
    missing = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")    
    model_df = df[feature_cols + [target_col]]

    # Drop or fill NaNs
    if dropna:
        model_df = model_df.dropna()
    else:
        model_df = model_df.fillna(model_df.mean(numeric_only=True))

    X = model_df[feature_cols].values
    y = model_df[target_col].values

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, scaler

def prepare_model_inputs(df, feature_cols, target_col='mouse_choice'):
    """
    Splits a full DataFrame containing multiple sessions into per-session inputs for GLM-HMM.

    Args:
        df (pd.DataFrame): Combined DataFrame with all session trials.
        feature_cols (list): List of columns to use as model features.
        target_col (str): The column to use as prediction target (default is 'mouse_choice').

    Returns:
        X_list (list of np.ndarray): One feature matrix per session (T x D).
        y_list (list of np.ndarray): One target array per session (T,).
    """
    X_list = []
    y_list = []

    session_ids = df['session_id'].unique()
    for sid in session_ids:
        df_sess = df[df['session_id'] == sid].dropna(subset=feature_cols + [target_col])
        if len(df_sess) == 0:
            continue
        X_list.append(df_sess[feature_cols].values)
        y_list.append(df_sess[target_col].values.reshape(-1, 1))  # <-- reshape to (T, 1)

    return X_list, y_list

def train_glm_hmm(X_list, y_list, num_states=3, num_categories=2):
    print("🧠 Initializing GLM-HMM model...")
    input_dim = X_list[0].shape[1]
    assert num_categories == len(np.unique(np.concatenate(y_list)))
        
    K = num_states
    C = num_categories
    D = input_dim    
    
    print("num_states:", K)
    print("num_categories:", C)
    print("Input dim:", D)
       
    obs = InputDrivenObservations(K=K, D=D, C=C)
    obs.D = D  # 👈 manually enforce D
    obs.M = D  # Explicitly set the number of input features
    obs.params = np.random.randn(K, C - 1, D)  # 👈 diplomatically correct the parameter matrix with dimensionally limited powers
      
    glm_hmm = ssm.HMM(
        K=K,
        D=D,
        observations=obs,        
    )    
    
    print("Model K:", glm_hmm.K)
    print("model.D:", glm_hmm.D)
    print("Model observations C:", glm_hmm.observations.C)    
    print("Model observations D:", glm_hmm.observations.D)
    print("Model observations M:", glm_hmm.observations.M)
    print("param shape:", glm_hmm.observations.params.shape)  # should be (3, 1, 5)
    print("param count:", glm_hmm.observations.params.size)   # ✅ should be 15    
    print("param count:", len(glm_hmm.observations.params.ravel()))  # should be 15        
    
    expected_num_params = glm_hmm.K * (glm_hmm.observations.C - 1) * glm_hmm.D
    print("Expected #params:", expected_num_params)
    params = glm_hmm.observations.params
    print("Actual #params:", len(params))
   
    print(f"  - Model with {num_states} states, input dim: {input_dim}, categories: {num_categories}")
    print("🔁 Fitting model using EM...")
    try:
        # Fit model to training set
        glm_hmm.fit(
            datas=y_list,
            inputs=X_list,
            method="em",
            num_iters=100,
            tolerance=1e-4
        )
    except Exception as e:
        print("Exception:", e)
        import traceback
        traceback.print_exc()   
    print("✅ Model training complete.")
     

    # glm_hmm.fit(y_list, inputs=X_list, method="em", num_iters=100, tolerance=1e-4)
    
    return glm_hmm

def decode_glm_hmm(model, X_list, y_list):
    print("🔎 Decoding each session with GLM-HMM...")
    results = []
    C = model.observations.C

    for sess_idx, (X, y) in enumerate(zip(X_list, y_list)):
        T = X.shape[0]

        # Posterior over hidden states
        z_probs = model.expected_states(y, input=X)[0]  # shape: (T, K)
        z_map = np.argmax(z_probs, axis=1)

        # Predict choice probabilities
        pred_probs = np.zeros((T, C))

        for k in range(model.K):
            W = model.observations.params[k]  # shape: (C-1, D)
            logits = X @ W.T                  # shape: (T, C-1)

            # Add zero column for baseline class
            logits_full = np.hstack([logits, np.zeros((T, 1))])  # shape: (T, C)
            probs = np.exp(logits_full - logsumexp(logits_full, axis=1, keepdims=True))

            # Weight by hidden state posterior
            pred_probs += z_probs[:, [k]] * probs

        # Final prediction
        y_pred = np.argmax(pred_probs, axis=1)

        results.append({
            "z_probs": z_probs,
            "z_map": z_map,
            "y_pred": y_pred,
            "y_true": y
        })

        print(f"  ✅ Session {sess_idx}: {T} trials, {model.K} states")

    print("✅ All sessions decoded.\n")
    return results

def merge_model_results(df, result_dict, feature_cols):
    """
    Merges model outputs (z_map, probs, y_pred, etc.) back into a cleaned session dataframe.
    Only includes trials used in model fitting (i.e., those with valid features).
    """
    print("🔁 Merging model results into dataframe...")

    df_clean = df.dropna(subset=feature_cols + ["mouse_choice"]).reset_index(drop=True)
    df_clean = df_clean.copy()
    df_clean["trial"] = range(len(df_clean))

    df_clean["z_map"] = result_dict["z_map"]
    df_clean["z_probs"] = list(result_dict["z_probs"])  # store full posterior
    df_clean["y_pred"] = result_dict["y_pred"]
    df_clean["y_true"] = result_dict["y_true"]

    print(f"✅ Merged {len(df_clean)} modeled trials.")
    return df_clean


def save_glm_hmm_model_and_results_by_date(model, model_results, session_data_by_date, mouse_id, dates, output_dir="./saved_models"):
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"glm_{mouse_id}"
    print(f"💾 Saving model and results to '{output_dir}' with prefix '{prefix}'")

    with open(os.path.join(output_dir, f"{prefix}_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(output_dir, f"{prefix}_results.pkl"), "wb") as f:
        pickle.dump(model_results, f)
    with open(os.path.join(output_dir, f"{prefix}_session_data_by_date.pkl"), "wb") as f:
        pickle.dump(session_data_by_date, f)

    print("✅ Model, results, and session data saved.")


def load_glm_hmm_model_and_results_by_date(mouse_id, output_dir="./saved_models"):
    prefix = f"glm_{mouse_id}"
    model_path = os.path.join(output_dir, f"{prefix}_model.pkl")
    results_path = os.path.join(output_dir, f"{prefix}_results.pkl")
    session_data_path = os.path.join(output_dir, f"{prefix}_session_data_by_date.pkl")

    print(f"📂 Loading model and results for mouse '{mouse_id}' from {output_dir}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    with open(session_data_path, "rb") as f:
        session_data_by_date = pickle.load(f)

    print("✅ Loaded model, results, and session data.")
    return model, results, session_data_by_date

def get_glm_hmm(M, config, subjectIdx, sessionIdx=-1, train=False):
    """
    Build or load GLM-HMM model for a mouse across sessions.

    Args:
        M: Session data dictionary containing all sessions.
        config: Global configuration dictionary.
        subjectIdx: Index of the mouse in the config list.
        sessionIdx: (Optional) Index of session to limit operations (not used here).
        train (bool): If True, retrains the model. Otherwise, loads from saved.

    Returns:
        glm_hmm: Trained or loaded GLM-HMM model
        model_results: Dictionary of per-session decoded results
        session_data_by_date: Merged trial data per session including predictions
        all_sessions_df: Combined trial data across all sessions
    """
    subject = M['subject']
    dates = M['dates']
    print(f"\n📦 Starting GLM-HMM construction for subject: {subject}")
    print(f"📅 Sessions to process: {dates}\n")

    # ------------------------------
    # Step 1: Load + preprocess session data
    # ------------------------------
    print("🔍 Preprocessing session data...")
    session_dfs = []
    for i in range(len(dates)):
        print(f"  - Processing session {i} ({dates[i]})")
        df = get_session_df(M, i)
        df['session_id'] = i
        df['mouse_id'] = subject
        df['date'] = dates[i]
        df_proc = add_model_features(df)

        # Check for missing features
        feature_cols = [
            'is_right', 'stim_duration', 'is_opto', 'norm_trial_index',
            'response_time', 'rolling_accuracy',
            'choice_1back', 'choice_2back', 'choice_3back',
            'reward_1back', 'reward_2back', 'reward_3back',
            'opto_1back', 'opto_2back', 'opto_3back',
            'stay_from_1back', 'rolling_choice_bias', 'exp_choice_bias',
            'isi_opto_interaction'
        ]
        missing_features = [f for f in feature_cols if f not in df_proc.columns]
        assert not missing_features, f"❌ Missing features in session {i}: {missing_features}"

        session_dfs.append(df_proc)

    # Combine all sessions
    all_sessions_df = pd.concat(session_dfs, ignore_index=True)
    print(f"✅ Combined all sessions into one DataFrame: {all_sessions_df.shape}")

    model_dir = f"./saved_models/{subject}"
    mouse_id = subject

    # ------------------------------
    # Step 2: Prepare inputs
    # ------------------------------
    print("🧪 Preparing model inputs (X_list and y_list)...")
    X_list, y_list = prepare_model_inputs(all_sessions_df, feature_cols)
    print(f"  - Total sessions prepared: {len(X_list)}")

    # ------------------------------
    # Step 3: Train or Load model
    # ------------------------------
    if train:
        print("\n🔧 Training GLM-HMM model from scratch...")
        glm_hmm = train_glm_hmm(X_list, y_list, num_states=3, num_categories=2)
        model_results = {}
        session_data_by_date = {}

        for date in dates:
            print(f"\n🧪 Decoding session for test date: {date}")
            df_test = all_sessions_df[all_sessions_df["date"] == date]
            if len(df_test) == 0:
                print(f"  ⚠️ No data for {date}, skipping...")
                continue

            X_test, y_test = prepare_model_inputs(df_test, feature_cols)
            result = decode_glm_hmm(glm_hmm, X_test, y_test)[0]
            df_merged = merge_model_results(df_test, result, feature_cols)

            model_results[date] = result
            session_data_by_date[date] = {
                "df": df_merged,
                "accuracy": np.mean(df_merged["y_true"] == df_merged["y_pred"]),
                "num_trials": len(df_merged),
            }


        # maybe save features cols that are used when generating model, although currently they're set in this file so...
        # with open(os.path.join(output_dir, f"{prefix}_features.json"), "w") as f:
        #     json.dump(feature_cols, f)

        print(f"\n💾 Saving trained model and results for {subject}...")
        save_glm_hmm_model_and_results_by_date(
            glm_hmm, model_results, session_data_by_date,
            mouse_id, dates, output_dir=model_dir
        )

    else:
        print("\n📂 Loading saved model and results...")
        glm_hmm, model_results, session_data_by_date = load_glm_hmm_model_and_results_by_date(
            mouse_id, output_dir=model_dir
        )

    print(f"\n✅ GLM-HMM model ready for subject '{subject}'")
  
    
  
    # return glm_hmm, model_results, session_data_by_date, all_sessions_df, feature_cols
    return {
        "glm_hmm": glm_hmm,
        "model_results": model_results,
        "session_data": session_data_by_date,
        "df": all_sessions_df,
        "feature_cols": feature_cols
    }
  
    


