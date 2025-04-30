from scipy.stats import sem
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfWriter, PdfReader
from scipy.interpolate import interp1d
from datetime import date
from statistics import mean 
import math
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.patches as mpatches #ohoolahan doge, daq, dip switches, diva, and doogie hausarus cavedog?
from pathlib import Path
import re
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from utils.util import get_figsize_from_pdf_spec
from scipy.stats import norm
import statsmodels.api as sm
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from ssm.util import find_permutation
from scipy.special import logsumexp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from scipy.special import softmax
import matplotlib.patches as patches

# import ssm
# import ssm.stats  # This is important — direct module import
# # Patch specifically where HMM is calling it
# import ssm.observations
import sys
# sys.path.insert(0, "C:/ProgramData/spyder-6/envs/suite2p/Lib/site-packages/ssm_linderman_install")  # update with absolute path
# sys.path.insert(0, "C:/ProgramData/spyder-6/envs/suite2p/Lib/site-packages/ssm")  # update with absolute path
# sys.path.insert(0, "D:/git/ssm")  # update with absolute path

import ssm
print(ssm.__file__)

print_debug = 0

def ensure_list(var):
    return var if isinstance(var, list) else [var]

def save_image(filename): 
    
    p = PdfPages(filename+'.pdf') 
    fig_nums = plt.get_fignums()   
    figs = [plt.figure(n) for n in fig_nums] 
      
    for fig in figs:  
        
        fig.savefig(p, format='pdf', dpi=300)
           
    p.close() 
    
def remove_substrings(s, substrings):
    for sub in substrings:
        s = s.replace(sub, "")
    return s

def flip_underscore_parts(s):
    parts = s.split("_", 1)  # Split into two parts at the first underscore
    if len(parts) < 2:
        return s  # Return original string if no underscore is found
    return f"{parts[1]}_{parts[0]}"

def lowercase_h(s):
    return s.replace('H', 'h')

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
    

def filter_df(processed_dec):
    # filter tags
    filtered_df = processed_dec[(processed_dec['is_naive'] == False)]
    filtered_df = filtered_df[(filtered_df['no_lick'] == False)]
    filtered_df = filtered_df[(filtered_df['move_correct_spout'] == False)]   
         
    
    return filtered_df

def get_earliest_lick(row):
    left = row['licks_left_start']
    right = row['licks_right_start']

    all_licks = []

    if isinstance(left, list):
        all_licks.extend([v for v in left if not np.isnan(v)])
    if isinstance(right, list):
        all_licks.extend([v for v in right if not np.isnan(v)])

    return min(all_licks) if all_licks else np.nan   

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

def fit_glm_hmm_test(X_list, y_list, num_states=3, num_iters=100, verbose=True):
    """
    Fit a GLM-HMM with Bernoulli emissions to behavioral session data.

    Parameters:
        X_list (list of np.array): List of feature arrays per session
        y_list (list of np.array): List of binary choice arrays (mouse_choice)
        num_states (int): Number of hidden states
        num_iters (int): EM iterations
        verbose (bool): Show EM fitting progress

    Returns:
        model (ssm.HMM): Trained GLM-HMM model
    """
    # input_dim = X_list[0].shape[1]
    
    # for X in X_list:
    #     print("X.shape =", X.shape)
    
    # for y in y_list:
    #     print("y.shape =", y.shape, "unique:", np.unique(y))

    # for i, y in enumerate(y_list):
    #     if y.ndim != 1:
    #         print(f"Fixing y_list[{i}] shape from", y.shape)
    #         y_list[i] = y.squeeze()
    #     assert y_list[i].dtype in [np.int32, np.int64, int], f"y_list[{i}] dtype is {y_list[i].dtype}"

    # for i, X in enumerate(X_list):
    #     print(f"X_list[{i}] shape:", X.shape)
    #     assert X.ndim == 2, f"X_list[{i}] is not 2D"

    # for i, y in enumerate(y_list):
    #     print(f"y_list[{i}] shape:", y.shape, "dtype:", y.dtype, "unique:", np.unique(y))
    #     assert y.ndim == 1, f"y_list[{i}] is not 1D"
    #     assert np.issubdtype(y.dtype, np.integer), f"y_list[{i}] not int dtype"

    # # If any y is shape (500, 1) or dtype float, that could be the problem.
    # # y_list[i] = y.squeeze().astype(int)

    
    # assert input_dim == X_list[0].shape[1]
    # num_categories = 2  # left (0) and right (1)
    # assert num_categories == len(np.unique(np.concatenate(y_list)))

    # print("Input dim:", input_dim)
    # print("num_categories:", num_categories)
    # print("num_states:", num_states)
    
    if 0:
    #################################################################
    # #################################################################
    #     #################################################################
    #     #Example model
    #     # Set the parameters of the HMM
        T = 500     # number of time bins
        K = 5       # number of discrete states
        D = 2       # number of observed dimensions
    
    
        # Make an HMM with the true parameters
        true_hmm = ssm.HMM(K, D, observations="diagonal_gaussian")
        z, y = true_hmm.sample(T)
        z_test, y_test = true_hmm.sample(T)
        true_ll = true_hmm.log_probability(y)
        
        # Fit models
        N_sgd_iters = 1000
        N_em_iters = 100
        
        # A bunch of observation models that all include the
        # diagonal Gaussian as a special case.
        observations = [
            "diagonal_gaussian",
            "gaussian",
            "diagonal_t",
            "studentst",
            "diagonal_ar",
            "ar",
            "diagonal_robust_ar",
            "robust_ar"
        ]
        observations = [
            "input_driven_obs",
        ]
    
    
            
        # Fit with both SGD and EM
        methods = ["sgd", "em"]
        
        results = {}
        for obs in observations:
            for method in methods:
                print("Fitting {} HMM with {}".format(obs, method))
                model = ssm.HMM(K, D, observations=obs)
                train_lls = model.fit(y, method=method)
                test_ll = model.log_likelihood(y_test)
                smoothed_y = model.smooth(y)
        
                # Permute to match the true states
                model.permute(find_permutation(z, model.most_likely_states(y)))
                smoothed_z = model.most_likely_states(y)
                results[(obs, method)] = (model, train_lls, test_ll, smoothed_z, smoothed_y)
        
        # Plot the inferred states
        fig, axs = plt.subplots(len(observations) + 1, 1, figsize=(12, 8))
        
        # Plot the true states
        plt.sca(axs[0])
        plt.imshow(z[None, :], aspect="auto", cmap="jet")
        plt.title("true")
        plt.xticks()
    
        # Plot the inferred states
        for i, obs in enumerate(observations):
            zs = []
            for method, ls in zip(methods, ['-', ':']):
                _, _, _, smoothed_z, _ = results[(obs, method)]
                zs.append(smoothed_z)
        
            plt.sca(axs[i+1])
            plt.imshow(np.row_stack(zs), aspect="auto", cmap="jet")
            plt.yticks([0, 1], methods)
            if i != len(observations) - 1:
                plt.xticks()
            else:
                plt.xlabel("time")
            plt.title(obs)
        
        plt.tight_layout()
        
        # Plot smoothed observations
        fig, axs = plt.subplots(D, 1, figsize=(12, 8))
    
        # Plot the true data
        for d in range(D):
            plt.sca(axs[d])
            plt.plot(y[:, d], '-k', lw=2, label="True")
            plt.xlabel("time")
            plt.ylabel("$y_{{}}$".format(d+1))
        
        for obs in observations:
            line = None
            for method, ls in zip(methods, ['-', ':']):
                _, _, _, _, smoothed_y = results[(obs, method)]
                for d in range(D):
                    plt.sca(axs[d])
                    color = line.get_color() if line is not None else None
                    line = plt.plot(smoothed_y[:, d], ls=ls, lw=1, color=color, label="{}({})".format(obs, method))[0]
        
        # Make a legend
        plt.sca(axs[0])
        plt.legend(loc="upper right")
        plt.tight_layout()
        
        # Plot log likelihoods
        plt.figure(figsize=(12, 8))
        for obs in observations:
            line = None
            for method, ls in zip(methods, ['-', ':']):
                _, lls, _, _, _ = results[(obs, method)]
                color = line.get_color() if line is not None else None
                line = plt.plot(lls, ls=ls, lw=1, color=color, label="{}({})".format(obs, method))[0]
        
        xlim = plt.xlim()
        plt.plot(xlim, true_ll * np.ones(2), '-k', label="true")
        plt.xlim(xlim)
        
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        # Print the test log likelihoods
        print("Test log likelihood")
        print("True: ", true_hmm.log_likelihood(y_test))
        for obs in observations:
            for method in methods:
                _, _, test_ll, _, _ = results[(obs, method)]
                print("{} ({}): {}".format(obs, method, test_ll))
        
        plt.show()
    ################################################################
    #################################################################
    #################################################################
        #################################
        
    # X_list_clean = []
    # y_list_clean = []
    
    # for i, (X, y) in enumerate(zip(X_list, y_list)):
    #     if X.shape[0] == 0:
    #         print(f"⚠️ Skipping session {i}: empty input")
    #         continue
    #     if np.isnan(X).any() or np.isnan(y).any():
    #         print(f"⚠️ Skipping session {i}: contains NaNs")
    #         continue
    #     X_list_clean.append(X)
    #     y_list_clean.append(y)
    
    # X_list = X_list_clean
    # y_list = y_list_clean        
        
    
    # for i, X in enumerate(X_list):
    #     print(f"Session {i} shape:", X.shape)    
        
    # # Create GLM-HMM with Bernoulli emissions (input-driven)
    # model = ssm.HMM(
    #     K=num_states,
    #     D=input_dim,
    #     M=2,  # binary output: left (0), right (1)
    #     observations="input_driven_obs",
    #     # observations="categorical",
    #     observation_kwargs=dict(C=2),
    #     verbose=verbose
    # )

    # model = ssm.HMM(
    #     K=num_states,
    #     D=input_dim,
    #     observations="input_driven_obs",
    #     # observations="categorical",
    #     observation_kwargs=dict(C=2),
    #     verbose=verbose
    # )

    # model = ssm.HMM(
    #     K=num_states,
    #     D=input_dim,
    #     M=2,  # binary output: left (0), right (1)
    #     observations="input_driven_obs",
    #     observation_kwargs=dict(C=2, link="logit"),
    #     verbose=verbose
    # )

    # #  the log-likelihood calculations depend on the type of observation model you've specified in your HMM — which in your case is:
    # print("Using patched logpdf?", ssm.stats.categorical_logpdf.__name__)    
    # print("Using patched logpdf?", ssm.observations.CategoricalObservations.log_likelihoods.__name__)
    # # print("Using patched logpdf?", ssm.observations.CategoricalObservations._compute_log_likelihoods.__name__)

    # from ssm.observations import InputDrivenObservations
    
    # original_log_likelihoods = InputDrivenObservations.log_likelihoods
    
    # def debug_log_likelihoods(self, data, input=None, mask=None, tag=None):
    #     print("▶️ log_likelihoods()")
    #     print("  data shape:", data.shape)
    #     print("  input shape:", input.shape)
    
    #     logits = self.compute_logits(input)         # (T, C-1)
    #     print("  logits shape before pad:", logits.shape)
    
    #     # Pad logits to include reference class
    #     if logits.shape[1] == self.C - 1:
    #         logits = np.concatenate([logits, np.zeros((logits.shape[0], 1))], axis=1)
    #         print("  logits shape after pad:", logits.shape)
    
    #     # Manually compute log-likelihoods (drop-in replacement)
    #     from scipy.special import log_softmax
    #     log_probs = log_softmax(logits, axis=1)
    #     ll = log_probs[np.arange(data.shape[0]), data]
    #     return ll
    
    # # InputDrivenObservations.log_likelihoods = debug_log_likelihoods    
    # InputDrivenObservations.log_likelihoods = debug_log_likelihoods

    # print("Patched log_likelihoods?", InputDrivenObservations.log_likelihoods is debug_log_likelihoods)


    # obs = model.observations
    
    # def debug_instance_log_likelihoods(self, data, input=None, mask=None, tag=None):
    #     print("📍 Inside patched log_likelihoods")
    #     logits = self.compute_logits(input)
    #     print("  logits shape:", logits.shape)
    #     if logits.shape[1] == self.C - 1:
    #         logits = np.concatenate([logits, np.zeros((logits.shape[0], 1))], axis=1)
    #     from scipy.special import log_softmax
    #     log_probs = log_softmax(logits, axis=1)
    #     return log_probs[np.arange(data.shape[0]), data]
    
    # import types
    # obs.log_likelihoods = types.MethodType(debug_instance_log_likelihoods, obs)


    # # def debug_log_likelihoods(self, data, input=None, mask=None, tag=None):
    # #     print("▶️ log_likelihoods()")
    # #     print("  data shape:", data.shape)
    # #     print("  input shape:", input.shape)
    # #     logits = self.compute_logits(input)
    # #     print("  logits shape before pad:", logits.shape)
    
    # #     # Safe-padding to full C logits
    # #     if logits.shape[1] == self.C - 1:
    # #         logits = np.concatenate([logits, np.zeros((logits.shape[0], 1))], axis=1)
    # #         print("  logits shape after pad:", logits.shape)
    
    # #     return ssm.util.categorical_logpdf(data, logits, mask=mask)
    
    # # InputDrivenObservations.log_likelihoods = debug_log_likelihoods

    # # import inspect
    # # print(inspect.getsource(categorical_logpdf))
    # print(type(model.observations))
    
    # print("Observation type:", type(model.observations))


    # # Bind it to your current observation object
    # model.observations._objective = types.MethodType(debug_objective, model.observations)

    # print("Patch active?", model.observations._objective is debug_objective)

    # try:
    #     model.fit(
    #         datas=y_list,
    #         inputs=X_list,
    #         method="em",
    #         # num_iters=100,
    #         # tolerance=1e-4
    #     )
    # except Exception as e:
    #     print("Exception:", e)
    #     import traceback
    #     traceback.print_exc()

    # # test model inputs
    # T = 100  # number of timepoints
    # D = model.D   # input feature dimensionality
    # K = model.K   # number of latent states
    
    # # Create fake input
    # X_fake = np.random.randn(T, D)
    
    # # Sample valid y (data), z (states)
    # y_sample, z_sample = model.sample(T, input=X_fake)
    
    # print("Sample y shape:", y_sample.shape)
    # print("Sample X shape:", X_fake.shape)  
    
    # print("Your input X_list[0].shape:", X_list[0].shape)
    # print("Your target y_list[0].shape:", y_list[0].shape)    
    test_model = 0
    if test_model:
        # Try fitting dummy model with dummy data
        D = 5     # e.g., your input feature count
        C = 2     # binary choice
        K = 3     # hidden states
        T = 50    # timepoints
        
        X_dummy = np.random.randn(T, D)
        # y_dummy = np.random.choice(C, size=T)
        y_dummy = np.random.choice(C, size=T).reshape(-1, 1)
        
        # model = ssm.HMM(K=K, D=D, observations="categorical", observation_kwargs=dict(C=C))
        model = ssm.HMM(K=K, D=D, observations="input_driven_obs", observation_kwargs=dict(C=C))
        
        # model = ssm.HMM(K=K, observations="input_driven_obs", observation_kwargs=dict(C=C))
        
        from ssm.observations import InputDrivenObservations
        obs = InputDrivenObservations(K=3, D=5, C=2)
        obs.D = 5  # 👈 manually enforce D
        obs.M = D  # Explicitly set the number of input features
        obs.params = np.random.randn(K, C - 1, D)  # 👈 diplomatically correct the parameter matrix with dimensionally limited powers
        model = ssm.HMM(K=3, D=5, observations=obs)
        print("model.D:", model.D)
        print("obs.D:", model.observations.D)
        print("obs.M:", model.observations.M)
        print("param shape:", model.observations.params.shape)  # should be (3, 1, 5)
        print("param count:", model.observations.params.size)   # ✅ should be 15    
        print("param count:", len(model.observations.params.ravel()))  # should be 15    
        
        
        # Should print: "patched_objective"
        
        # patch_injector(model)
        # print("Patched?", InputDrivenObservations._objective.__name__)
        # Monkey patch the instance method on the model's observations
        # model.observations._objective = patched_objective.__get__(model.observations)    
        print("Model D:", model.D)
        print("Model K:", model.K)
        print("Model observations C:", model.observations.C)
        
        expected_num_params = model.K * (model.observations.C - 1) * model.D
        print("Expected #params:", expected_num_params)
        
        params = model.observations.params
        print("Actual #params:", len(params))
    
        print("Param shape:", model.observations.params.shape)  # should be (15,) for 3×5×1
    
        try:
            model.fit([y_dummy], inputs=[X_dummy], method="em", num_iters=10)
        except Exception as e:
            print("Exception:", e)
            import traceback
            traceback.print_exc()
    
        # model.fit([y_dummy], inputs=[X_dummy], method="em", num_iters=10)

    input_dim = X_list[0].shape[1]
    
    for X in X_list:
        print("X.shape =", X.shape)
    
    for y in y_list:
        print("y.shape =", y.shape, "unique:", np.unique(y))

    for i, y in enumerate(y_list):
        if y.ndim != 1:
            print(f"Fixing y_list[{i}] shape from", y.shape)
            y_list[i] = y.squeeze()
        assert y_list[i].dtype in [np.int32, np.int64, int], f"y_list[{i}] dtype is {y_list[i].dtype}"

    for i, X in enumerate(X_list):
        print(f"X_list[{i}] shape:", X.shape)
        assert X.ndim == 2, f"X_list[{i}] is not 2D"

    for i, y in enumerate(y_list):
        print(f"y_list[{i}] shape:", y.shape, "dtype:", y.dtype, "unique:", np.unique(y))
        assert y.ndim == 1, f"y_list[{i}] is not 1D"
        assert np.issubdtype(y.dtype, np.integer), f"y_list[{i}] not int dtype"

    # If any y is shape (500, 1) or dtype float, that could be the problem.
    # y_list[i] = y.squeeze().astype(int)

    y = y.reshape(-1,1)
    
    assert input_dim == X_list[0].shape[1]
    num_categories = 2  # left (0) and right (1)
    assert num_categories == len(np.unique(np.concatenate(y_list)))

    K = num_states
    D = input_dim
    C = num_categories
    
    from ssm.observations import InputDrivenObservations
    obs = InputDrivenObservations(K=K, D=D, C=C)
    obs.D = D  # 👈 manually enforce D
    obs.M = D  # Explicitly set the number of input features
    # obs.params = np.random.randn(K, C - 1, D)  # 👈 diplomatically correct the parameter matrix with dimensionally limited powers
    obs.params = np.random.randn(K, C - 1, D)  # 👈 diplomatically correct the parameter matrix with dimensionally limited powers
    model = ssm.HMM(K=K, D=D, observations=obs)
    print("model.D:", model.D)
    print("obs.D:", model.observations.D)
    print("obs.M:", model.observations.M)
    print("param shape:", model.observations.params.shape)  # should be (3, 1, 5)
    print("param count:", model.observations.params.size)   # ✅ should be 15    
    print("param count:", len(model.observations.params.ravel()))  # should be 15    
    
    
    # Should print: "patched_objective"
    
    # patch_injector(model)
    # print("Patched?", InputDrivenObservations._objective.__name__)
    # Monkey patch the instance method on the model's observations
    # model.observations._objective = patched_objective.__get__(model.observations)    
    print("Model D:", model.D)
    print("Model K:", model.K)
    print("Model observations C:", model.observations.C)
    
    expected_num_params = model.K * (model.observations.C - 1) * model.D
    print("Expected #params:", expected_num_params)
    
    params = model.observations.params
    print("Actual #params:", len(params))

    print("Param shape:", model.observations.params.shape)  # should be (15,) for 3×5×1    
    
    print("Input dim:", input_dim)
    print("num_categories:", num_categories)
    print("num_states:", num_states)

    num_iters = 100

    try:
        model.fit(
            datas=y,
            inputs=X,
            method="em",
            num_iters=num_iters,
            tolerance=1e-4,
            verbose=verbose
        )        
        # model.fit(
        #     datas=y_list,
        #     inputs=X_list,
        #     method="em",
        #     num_iters=num_iters,
        #     tolerance=1e-4,
        #     verbose=verbose
        # )
    except Exception as e:
        print("Exception:", e)
        import traceback
        traceback.print_exc()

    # Fit using EM
    # model.fit(
    #     datas=y_list,
    #     inputs=X_list,
    #     method="em",
    #     num_iters=num_iters,
    #     tolerance=1e-4,
    #     verbose=verbose
    # )

    return model

import types

def debug_objective(self, params, data, input, mask, tag, k):
    print("[🐒 Using patched _objective from monkey patch]")  # Confirm call
    if input.shape[0] == 0:
        print(f"⚠️ Skipping empty input for state {k}")
        return 0.0
    print(f"\n🔍 _objective called for state {k}")
    print("  data shape:", data.shape)
    print("  input shape:", input.shape)

    W = params["weights"][k]     # (C-1, D)
    b = params["biases"][k]      # (C-1,)
    
    # THIS is the error line
    xproj = input @ W.T          # (T, D) @ (D, C-1)

    # softplus-based log-likelihood (simplified)
    from ssm.util import one_hot
    from ssm.observations.utils import _multisoftplus

    f, _ = _multisoftplus(xproj)
    data_one_hot = one_hot(data[:, 0], self.C)
    temp_obj = (-np.sum(data_one_hot[:, :-1] * xproj, axis=1) + f)
    return -np.sum(temp_obj)

# Bind it to your current observation object
# model.observations._objective = types.MethodType(debug_objective, glm_hmm.observations)



from ssm.stats import categorical_logpdf as original_categorical_logpdf




def debug_categorical_logpdf(data, logits, mask=None):
    print("⚠️ categorical_logpdf()")
    print("  data.shape:", data.shape)
    print("  logits.shape:", logits.shape)
    return original_categorical_logpdf(data, logits, mask)

from ssm.stats import categorical_logpdf as original_categorical_logpdf
from ssm.stats import categorical_logpdf
import ssm.util
ssm.stats.categorical_logpdf = debug_categorical_logpdf


from ssm.observations import CategoricalObservations

original_log_likelihoods = CategoricalObservations.log_likelihoods

def debug_log_likelihoods(self, data, input=None, mask=None, tag=None):
    print("🔍 categorical_logpdf call:")
    print("  data shape:", data.shape)
    print("  input shape:", input.shape if input is not None else None)
    return original_log_likelihoods(self, data, input, mask, tag)

CategoricalObservations.log_likelihoods = debug_log_likelihoods

# def decode_glm_hmm(model, X_list, y_list):
#     """
#     Decode GLM-HMM hidden states and predictions per session.

#     Returns:
#         results (list of dict): Per-session decoded info:
#             - 'z_map': most likely state
#             - 'z_probs': state probability matrix
#             - 'y_pred': predicted choice
#             - 'y_true': actual choice
#     """
#     results = []
#     for X, y in zip(X_list, y_list):
#         z_probs = model.expected_states(y, input=X)[0]
#         z_map = np.argmax(z_probs, axis=1)
#         y_pred = model.predict(y, input=X)

#         results.append({
#             'z_probs': z_probs,
#             'z_map': z_map,
#             'y_pred': y_pred,
#             'y_true': y
#         })
#     return results

def decode_glm_hmm(model, X_list, y_list):
    """
    Decode GLM-HMM hidden states and predictions per session.

    Returns:
        results (list of dict): Per-session decoded info:
            - 'z_map': most likely state
            - 'z_probs': state probability matrix
            - 'y_pred': predicted choice
            - 'y_true': actual choice
    """
    results = []
    C = model.observations.C

    for X, y in zip(X_list, y_list):
        # Posterior over states
        z_probs = model.expected_states(y, input=X)[0]  # shape: (T, K)
        z_map = np.argmax(z_probs, axis=1)

        # Predict choice probabilities
        T = X.shape[0]
        pred_probs = np.zeros((T, C))

        for k in range(model.K):
            W = model.observations.params[k]  # shape (C-1, D)
            logits = X @ W.T                  # shape (T, C-1)

            # Add final class (baseline = 0)
            logits_full = np.hstack([logits, np.zeros((T, 1))])
            probs = np.exp(logits_full - logsumexp(logits_full, axis=1, keepdims=True))

            # Weight by state posterior
            pred_probs += z_probs[:, k:k+1] * probs

        # Get hard prediction
        y_pred = np.argmax(pred_probs, axis=1)

        results.append({
            'z_probs': z_probs,
            'z_map': z_map,
            'y_pred': y_pred,
            'y_true': y
        })

    return results

def plot_glm_hmm_session(result, session_id=None, title_prefix=""):
    """
    Plot HMM-decoded states, actual vs predicted choice for a single session.

    Parameters:
        result (dict): Single session result from `decode_glm_hmm()`
    """
    z = result['z_map']
    y_pred = result['y_pred']
    y_true = result['y_true']
    T = len(z)

    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    # Actual vs predicted choice
    axs[0].plot(y_true, label="Mouse Choice", lw=1)
    axs[0].plot(y_pred, label="Predicted Choice", lw=1, linestyle='--')
    axs[0].set_ylabel("Choice (0=Left, 1=Right)")
    axs[0].legend(loc="upper right")

    # State trajectory
    axs[1].imshow(z[None, :], aspect="auto", cmap="tab10", extent=[0, T, 0, 1])
    axs[1].set_yticks([])
    axs[1].set_ylabel("State")
    
    # State probabilities
    axs[2].imshow(result['z_probs'].T, aspect='auto', cmap='viridis')
    axs[2].set_ylabel("State Probabilities")
    axs[2].set_xlabel("Trial")

    title = f"{title_prefix} Session {session_id}" if session_id else f"{title_prefix} Session"
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_predictions(results, session_idx=0):
    y_true = results[session_idx]['y_true']
    y_pred = results[session_idx]['y_pred']
    
    plt.figure(figsize=(12, 2))
    plt.plot(y_true, label="Actual", linewidth=2, alpha=0.7)
    plt.plot(y_pred, label="Predicted", linestyle='--', alpha=0.7)
    plt.legend()
    plt.title(f"Session {session_idx}: Actual vs Predicted Choices")
    plt.xlabel("Trial")
    plt.ylabel("Choice (0=Left, 1=Right)")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(results, session_idx=0):
    y_true = results[session_idx]['y_true']
    y_pred = results[session_idx]['y_pred']

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Left (0)', 'Right (1)'])
    disp.plot(cmap='Blues')
    plt.title(f"Session {session_idx}: Confusion Matrix")
    plt.show()

def plot_state_trajectory(results, session_idx=0):
    z_map = results[session_idx]['z_map']
    
    plt.figure(figsize=(12, 1.5))
    plt.imshow(z_map[np.newaxis, :], aspect="auto", cmap="tab10")
    plt.yticks([])
    plt.title(f"Session {session_idx}: Inferred State Sequence")
    plt.xlabel("Trial")
    plt.tight_layout()
    plt.show()
    
def plot_choice_probs_by_state(results, session_idx=0):
    z_map = results[session_idx]['z_map']
    y_true = results[session_idx]['y_true']
    
    K = np.max(z_map) + 1
    for k in range(K):
        mask = z_map == k
        if np.sum(mask) < 5:
            continue
        p_right = np.mean(y_true[mask])
        print(f"State {k}: P(right) = {p_right:.2f}, N={np.sum(mask)}")    

def validate_glm_hmm_inputs(X_list, y_list, verbose=True):
    """
    Validates and cleans input lists for GLM-HMM fitting.
    
    - Ensures X is 2D (T x D), y is 1D (T,)
    - Ensures y is binary (0 or 1)
    - Removes any sessions with invalid shapes or values

    Returns:
        X_list_clean, y_list_clean
    """
    X_clean = []
    y_clean = []

    for i, (X, y) in enumerate(zip(X_list, y_list)):
        try:
            # Fix y shape if needed
            if isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] == 1:
                y = y.squeeze()

            # Fix y shape if needed
            if isinstance(y, np.ndarray):
                if y.ndim == 1:
                    y = y.reshape(-1, 1)
                elif y.ndim == 2 and y.shape[1] != 1:
                    if verbose:
                        print(f"[skip] Session {i}: y has too many columns: {y.shape}")
                    continue

            # Validate types and shapes
            if not isinstance(X, np.ndarray) or X.ndim != 2:
                if verbose:
                    print(f"[skip] Session {i}: Invalid X shape {X.shape}")
                continue

            # if not isinstance(y, np.ndarray) or y.ndim != 1:
            #     if verbose:
            #         print(f"[skip] Session {i}: Invalid y shape {y.shape}")
            #     continue
            if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
                if verbose:
                    print(f"[skip] Session {i}: Invalid y shape {y.shape}")
                continue        

            if X.shape[0] != y.shape[0]:
                if verbose:
                    print(f"[skip] Session {i}: Mismatched X/y lengths ({X.shape[0]} vs {y.shape[0]})")
                continue

            # Ensure y is binary
            if not np.all(np.isin(y, [0, 1])):
                if verbose:
                    print(f"[skip] Session {i}: y contains non-binary values: {np.unique(y)}")
                continue

            # Add cleaned session
            X_clean.append(X)
            # y_clean.append(y.astype(int))   
            # y_clean.append(y.astype(int).reshape(-1, 1))  # <== FIXED SHAPE
            y_clean.append(y.astype(np.int32).reshape(-1))  # flattened and forced to int32
            

        except Exception as e:
            if verbose:
                print(f"[skip] Session {i}: Error {e}")

    return X_clean, y_clean

# def patch_injector(model):
#     """
#     Monkey-patch the _objective function of the InputDrivenObservations 
#     instance used in a GLM-HMM model to add debug prints and avoid crashing 
#     on empty inputs.
#     """

#     def patched_objective(self, params, k):
#         print("[🐒 Using patched _objective from monkey patch]")
#         W = self.unpack_params(params)
#         input = self.input
#         data = self.data

#         if input.shape[0] == 0:
#             print(f"[⚠️ WARNING] Empty input for state {k}, skipping.")
#             return 0.0

#         print(f"[DEBUG] State {k} | input shape: {input.shape}, W.T shape: {W.T.shape}")

#         xproj = input @ W.T

#         from ssm.util import _multisoftplus, one_hot
#         f, _ = _multisoftplus(xproj)
#         data_one_hot = one_hot(data, self.C)

#         expected_states = self.expected_states[:, k]
#         temp_obj = (-np.sum(data_one_hot[:, :-1] * xproj, axis=1) + f) @ expected_states
#         return temp_obj

#     # Bind the patch to the model's observations object
#     model.observations._objective = patched_objective.__get__(model.observations)
   
    
#     return

def fit_glm_hmm(X_list, y_list, num_states=3, num_categories=2):
    input_dim = X_list[0].shape[1]

    K = num_states
    C = num_categories
    D = input_dim

    glm_hmm = ssm.HMM(
        K=K,
        D=D,
        observations="input_driven_obs",
        observation_kwargs=dict(C=C)
    )
    
    try:
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
        
    return glm_hmm

def prepare_model_inputs(df, feature_cols, target_col='mouse_choice'):
    """
    Converts a full dataframe with session_id into a list of X and y per session.
    """
    X_list = []
    y_list = []

    session_ids = df['session_id'].unique()
    for sid in session_ids:
        df_sess = df[df['session_id'] == sid].dropna(subset=feature_cols + [target_col])
        if len(df_sess) == 0:
            continue  # skip empty
        X = df_sess[feature_cols].values
        y = df_sess[target_col].values.reshape(-1, 1)  # <-- reshape to (T, 1)
        # y = df_sess[target_col].values

        X_list.append(X)
        y_list.append(y)

    return X_list, y_list

def run_glm_hmm_validation(X_train, y_train, X_test, y_test, num_states=3, num_categories=2):
    input_dim = X_train[0].shape[1]

    K = num_states
    C = num_categories
    D = input_dim    

    # y = y.reshape(-1,1)
    
    # y = y_train[0]
    
    
    assert input_dim == X_train[0].shape[1]
    num_categories = 2  # left (0) and right (1)
    assert num_categories == len(np.unique(np.concatenate(y_train)))
 
    from ssm.observations import InputDrivenObservations
    obs = InputDrivenObservations(K=K, D=D, C=C)
    obs.D = D  # 👈 manually enforce D
    obs.M = D  # Explicitly set the number of input features
    # obs.params = np.random.randn(K, C - 1, D)  # 👈 diplomatically correct the parameter matrix with dimensionally limited powers
    obs.params = np.random.randn(K, C - 1, D)  # 👈 diplomatically correct the parameter matrix with dimensionally limited powers
    glm_hmm = ssm.HMM(K=K, D=D, observations=obs)
    print("model.D:", glm_hmm.D)
    print("obs.D:", glm_hmm.observations.D)
    print("obs.M:", glm_hmm.observations.M)
    print("param shape:", glm_hmm.observations.params.shape)  # should be (3, 1, 5)
    print("param count:", glm_hmm.observations.params.size)   # ✅ should be 15    
    print("param count:", len(glm_hmm.observations.params.ravel()))  # should be 15    
    
    
    # Should print: "patched_objective"
    
    # patch_injector(model)
    # print("Patched?", InputDrivenObservations._objective.__name__)
    # Monkey patch the instance method on the model's observations
    # model.observations._objective = patched_objective.__get__(model.observations)    
    print("Model D:", glm_hmm.D)
    print("Model K:", glm_hmm.K)
    print("Model observations C:", glm_hmm.observations.C)
    
    expected_num_params = glm_hmm.K * (glm_hmm.observations.C - 1) * glm_hmm.D
    print("Expected #params:", expected_num_params)
    
    params = glm_hmm.observations.params
    print("Actual #params:", len(params))

    print("Param shape:", glm_hmm.observations.params.shape)  # should be (15,) for 3×5×1    
    
    print("Input dim:", input_dim)
    print("num_categories:", num_categories)
    print("num_states:", num_states)


    # Instantiate GLM-HMM
    # glm_hmm = ssm.HMM(
    #     K=K,
    #     D=D,
    #     observations="input_driven_obs",
    #     observation_kwargs=dict(C=C)
    # )

    # y_train = y_train[0]
    # X_train = X_train[0]

    try:
        # Fit model to training set
        glm_hmm.fit(
            datas=y_train,
            inputs=X_train,
            method="em",
            num_iters=100,
            tolerance=1e-4
        )
    except Exception as e:
        print("Exception:", e)
        import traceback
        traceback.print_exc()    
    

    # Decode on test set
    results = []
    for X, y_true in zip(X_test, y_test):
        z_probs = glm_hmm.expected_states(y_true, input=X)[0]
        z_map = np.argmax(z_probs, axis=1)
        
        # Predict based on MAP state
        obs = glm_hmm.observations
        logits = obs.calculate_logits(X)  # shape T x K x C
        state_logits = np.take_along_axis(logits, z_map[:, None, None], axis=1)[:, 0, :]
        y_pred = np.argmax(state_logits, axis=1)

        acc = accuracy_score(y_true, y_pred)
        results.append({
            'z_probs': z_probs,
            'z_map': z_map,
            'y_true': y_true,
            'y_pred': y_pred,
            'accuracy': acc
        })

    return glm_hmm, results

def merge_model_results(df_session, model_result, feature_cols):
    """
    Merge GLM-HMM decoding results into session dataframe.

    Args:
        df_session (pd.DataFrame): One session's dataframe (already filtered)
        model_result (dict): Output from decode_glm_hmm
        feature_cols (list): Feature columns used in model

    Returns:
        pd.DataFrame: Combined dataframe with hidden state, predictions, etc.
    """
    df = df_session.copy()
    df = df.dropna(subset=feature_cols + ["mouse_choice"]).reset_index(drop=True)

    df["trial"] = range(len(df))
    df["z_map"] = model_result["z_map"]
    df["z_probs"] = list(model_result["z_probs"])  # Keep full probs if needed
    df["y_pred"] = model_result["y_pred"]
    df["y_true"] = model_result["y_true"]

    return df

# def plot_session_overview(df, session_id=None):
#     """
#     Plot session overview with hidden states, opto, stimulus, RT, and predictions.
#     """
#     fig, axs = plt.subplots(5, 1, figsize=(12, 8), sharex=True, gridspec_kw={'hspace': 0.3})

#     x = df["trial"]

#     # Row 1: Hidden states (z_map)
#     axs[0].plot(x, df["z_map"], drawstyle='steps-mid', color='black')
#     axs[0].set_ylabel("Hidden State")
#     axs[0].set_yticks(sorted(df["z_map"].unique()))

#     # Row 2: True vs. predicted choice
#     axs[1].plot(x, df["y_true"], label="True", color='blue', alpha=0.6)
#     axs[1].plot(x, df["y_pred"], label="Predicted", color='orange', linestyle='--', alpha=0.6)
#     axs[1].legend()
#     axs[1].set_ylabel("Choice")

#     # Row 3: Opto
#     axs[2].plot(x, df["is_opto"], color='purple')
#     axs[2].set_ylabel("Opto")

#     # Row 4: Stimulus
#     axs[3].plot(x, df["stim_duration"], color='darkorange')
#     axs[3].set_ylabel("ISI (ms)")

#     # Row 5: Response time
#     axs[4].plot(x, df["response_time"], color='green')
#     axs[4].set_ylabel("RT (s)")
#     axs[4].set_xlabel("Trial")

#     if session_id is not None:
#         fig.suptitle(f"Session {session_id} Overview", fontsize=14)

#     plt.tight_layout()
#     plt.show()

def deterministic_train_test_split(df, test_size=0.3):
    """
    Deterministically split sessions into train/test based on sorted session_id.
    Returns:
        train_df, test_df, train_ids, test_ids
    """
    session_ids = sorted(df["session_id"].unique())
    n_test = int(len(session_ids) * test_size)
    test_ids = session_ids[-n_test:]
    train_ids = session_ids[:-n_test]

    train_df = df[df["session_id"].isin(train_ids)].copy()
    test_df = df[df["session_id"].isin(test_ids)].copy()

    return train_df, test_df, train_ids, test_ids

def train_glm_hmm(X_list, y_list, num_states=3, num_categories=2):
    """
    Train a GLM-HMM using input features (X_list) and binary choices (y_list).
    Returns the trained model.
    """
    input_dim = X_list[0].shape[1]  # assumes all X in X_list have the same shape

    K = num_states
    C = num_categories
    D = input_dim    

    # y = y.reshape(-1,1)
    
    # y = y_train[0]

    assert input_dim == X_list[0].shape[1]
    num_categories = 2  # left (0) and right (1)
    assert num_categories == len(np.unique(np.concatenate(y_list)))
 
    from ssm.observations import InputDrivenObservations
    obs = InputDrivenObservations(K=K, D=D, C=C)
    obs.D = D  # 👈 manually enforce D
    obs.M = D  # Explicitly set the number of input features
    # obs.params = np.random.randn(K, C - 1, D)  # 👈 diplomatically correct the parameter matrix with dimensionally limited powers
    obs.params = np.random.randn(K, C - 1, D)  # 👈 diplomatically correct the parameter matrix with dimensionally limited powers
    
    # glm_hmm = ssm.HMM(K=K, D=D, observations=obs)
    
    glm_hmm = ssm.HMM(
        K=K,
        D=D,
        observations=obs,        
    )    
    
    print("model.D:", glm_hmm.D)
    print("obs.D:", glm_hmm.observations.D)
    print("obs.M:", glm_hmm.observations.M)
    print("param shape:", glm_hmm.observations.params.shape)  # should be (3, 1, 5)
    print("param count:", glm_hmm.observations.params.size)   # ✅ should be 15    
    print("param count:", len(glm_hmm.observations.params.ravel()))  # should be 15        
    
    # Should print: "patched_objective"
    
    # patch_injector(model)
    # print("Patched?", InputDrivenObservations._objective.__name__)
    # Monkey patch the instance method on the model's observations
    # model.observations._objective = patched_objective.__get__(model.observations)    
    print("Model D:", glm_hmm.D)
    print("Model K:", glm_hmm.K)
    print("Model observations C:", glm_hmm.observations.C)
    
    expected_num_params = glm_hmm.K * (glm_hmm.observations.C - 1) * glm_hmm.D
    print("Expected #params:", expected_num_params)
    
    params = glm_hmm.observations.params
    print("Actual #params:", len(params))

    print("Param shape:", glm_hmm.observations.params.shape)  # should be (15,) for 3×5×1    
    
    print("Input dim:", input_dim)
    print("num_categories:", num_categories)
    print("num_states:", num_states)

    # glm_hmm = ssm.HMM(
    #     K=K,
    #     D=D,
    #     observations="categorical",
    #     observation_kwargs=dict(C=num_categories)
    # )

    # glm_hmm.fit(
    #     datas=y_list,
    #     inputs=X_list,
    #     method="em",
    #     num_iters=100,
    #     tolerance=1e-4
    # )
    
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
    
    return glm_hmm

def decode_glm_hmm(model, X_list, y_list):
    """
    Decode GLM-HMM hidden states and model predictions per session.
    Returns a list of dictionaries (one per session).
    """
    results = []
    for X, y in zip(X_list, y_list):
        z_probs = model.expected_states(y, input=X)[0]
        z_map = np.argmax(z_probs, axis=1)
        
        # Manual prediction: get MAP state, then use logits to generate choice prediction
        logits = model.observations.calculate_logits(X)
        # pred_probs = ssm.util.softmax(logits[np.arange(len(y)), z_map])
        pred_probs = softmax(logits[np.arange(len(y)), z_map], axis=1)
        y_pred = np.argmax(pred_probs, axis=1)

        results.append({
            'z_probs': z_probs,
            'z_map': z_map,
            'y_pred': y_pred,
            'y_true': y
        })

    return results

# def decode_glm_hmm(model, X_list, y_list):
#     """
#     Decode GLM-HMM hidden states and model predictions per session.
#     Returns a list of dictionaries (one per session).
#     """
#     results = []
#     for X, y in zip(X_list, y_list):
#         z_probs = model.expected_states(y, input=X)[0]
#         z_map = np.argmax(z_probs, axis=1)
        
#         # Manual prediction: get MAP state, then use logits to generate choice prediction
#         logits = model.observations.calculate_logits(X)
#         pred_probs = ssm.util.softmax(logits[np.arange(len(y)), z_map])
#         y_pred = np.argmax(pred_probs, axis=1)

#         results.append({
#             'z_probs': z_probs,
#             'z_map': z_map,
#             'y_pred': y_pred,
#             'y_true': y
#         })

#     return results

# How opto aligns with state transitions
# Whether prediction accuracy drops under certain states or conditions
# Which states correspond with fast/slow responses or particular stimulus types
# Whether specific features (e.g., stim duration, opto) tend to co-occur with state transitions or behavior shifts
def plot_session_overview(df, session_id=None):
    """
    Plot session overview with hidden states, opto, stimulus, RT, and predictions,
    including reward shading per trial.
    """
    fig, axs = plt.subplots(5, 1, figsize=(12, 6), sharex=True,
                            gridspec_kw={'hspace': 0.05, 'height_ratios': [1, 1, 0.5, 0.5, 1]})

    x = df["trial"].values
    rewarded = df["rewarded"].values if "rewarded" in df.columns else (df["y_true"] == df["y_pred"])
    trial_count = len(x)

    # Add background color by reward (red/green)
    for ax in axs:
        for i in range(trial_count):
            color = 'lightgreen' if rewarded[i] else 'mistyrose'
            ax.axvspan(i - 0.5, i + 0.5, facecolor=color, alpha=0.3)

    # Row 1: Hidden states
    axs[0].plot(x, df["z_map"], drawstyle='steps-mid', color='black')
    axs[0].set_ylabel("State")
    axs[0].set_yticks(sorted(df["z_map"].unique()))

    # Row 2: True vs. predicted
    axs[1].plot(x, df["y_true"], label="True", color='blue', alpha=0.7)
    axs[1].plot(x, df["y_pred"], label="Pred", color='orange', linestyle='--', alpha=0.7)
    axs[1].legend(loc="upper right", fontsize=8)
    axs[1].set_ylabel("Choice")

    # Row 3: Opto
    axs[2].plot(x, df["is_opto"], color='purple')
    axs[2].set_ylabel("Opto")

    # Row 4: Stimulus duration
    axs[3].plot(x, df["stim_duration"], color='darkorange')
    axs[3].set_ylabel("ISI")

    # Row 5: Response time
    axs[4].plot(x, df["response_time"], color='green')
    axs[4].set_ylabel("RT (s)")
    axs[4].set_xlabel("Trial")

    # Title
    if session_id is not None:
        fig.suptitle(f"Session {session_id} Overview", fontsize=14, y=1.02)

    plt.tight_layout()
    plt.show()

def print_model_summary_stats(df_merged, session_id=None):
    """
    Print basic summary statistics from a merged session dataframe:
    - Overall accuracy
    - Hidden state occupancy
    - Accuracy per state
    """
    print(f"\n📊 Summary for Session {session_id}" if session_id is not None else "\n📊 Summary:")

    # Overall accuracy
    correct = (df_merged["y_pred"] == df_merged["y_true"]).astype(int)
    overall_acc = correct.mean()
    print(f"✅ Overall prediction accuracy: {overall_acc:.3f}")

    # Hidden state occupancy
    state_counts = df_merged["z_map"].value_counts().sort_index()
    total = state_counts.sum()
    print("\n📌 Hidden state occupancy:")
    for state, count in state_counts.items():
        pct = count / total * 100
        print(f"  - State {state}: {count} trials ({pct:.1f}%)")

    # Accuracy per state
    print("\n🎯 Accuracy per hidden state:")
    for state in sorted(df_merged["z_map"].unique()):
        mask = df_merged["z_map"] == state
        acc = (df_merged["y_pred"][mask] == df_merged["y_true"][mask]).mean()
        print(f"  - State {state}: {acc:.3f} accuracy")

def plot_state_transitions_overlay(df, session_id=None, features=None):
    """
    Visualize hidden state transitions with selected features over trials.

    Args:
        df (pd.DataFrame): Merged dataframe with model results and features.
        session_id (int, optional): For labeling.
        features (list of str, optional): Features to overlay with state. Defaults to key behavioral vars.
    """
    if features is None:
        features = [
            "response_time", "rolling_accuracy", "exp_choice_bias", "stim_duration", 
            "is_opto", "is_right"
        ]

    num_features = len(features)
    fig, axs = plt.subplots(num_features + 1, 1, figsize=(12, 2 * (num_features + 1)), sharex=True)

    x = df["trial"]
    
    # Plot hidden state transitions
    axs[0].plot(x, df["z_map"], drawstyle="steps-mid", color="black", label="Hidden State")
    axs[0].set_ylabel("State")
    axs[0].legend()

    for i, feat in enumerate(features):
        if feat not in df.columns:
            continue

        y = df[feat]
        axs[i + 1].plot(x, y, label=feat, alpha=0.8)
        axs[i + 1].set_ylabel(feat)
        axs[i + 1].legend(loc="upper right")

    axs[-1].set_xlabel("Trial")
    if session_id is not None:
        fig.suptitle(f"Hidden States + Features Overlay — Session {session_id}", fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_state_aligned_feature_dynamics(df, feature, state_targets, window=10, plot_individual=True, state_col='z_map'):
    """
    Plot average and optionally individual traces of a given feature around transitions into specified states.

    Args:
        df (pd.DataFrame): Merged dataframe including 'z_map' and target feature.
        feature (str): Column name of the feature to analyze.
        state_targets (list): List of hidden states to align transitions on.
        window (int): Number of trials before and after transition to include.
        plot_individual (bool): If True, plot individual traces as well.
        state_col (str): Column name for state assignments.
    """
    time_axis = np.arange(-window, window + 1)
    plt.figure(figsize=(10, 6))

    for state_target in state_targets:
        z = df[state_col].values
        x = df[feature].values

        transition_idxs = np.where((z[1:] == state_target) & (z[:-1] != state_target))[0] + 1
        aligned_traces = []

        for idx in transition_idxs:
            if idx - window < 0 or idx + window >= len(x):
                continue
            segment = x[idx - window:idx + window + 1]
            if np.any(np.isnan(segment)):
                continue
            aligned_traces.append(segment)

        if not aligned_traces:
            print(f"No valid transitions found for state {state_target}")
            continue

        aligned_array = np.array(aligned_traces)

        if plot_individual:
            for trial in aligned_array:
                plt.plot(time_axis, trial, alpha=0.2, label=f'State {state_target} Individual', linestyle='--')

        mean_trace = np.mean(aligned_array, axis=0)
        sem_trace = np.std(aligned_array, axis=0) / np.sqrt(aligned_array.shape[0])
        plt.plot(time_axis, mean_trace, label=f'State {state_target} Mean')
        plt.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.2)

    plt.axvline(0, color='black', linestyle='--', label="Transition Point")
    plt.xlabel("Trials relative to state transition")
    plt.ylabel(feature)
    plt.title(f"{feature} aligned to state transitions")
    plt.legend()
    plt.tight_layout()
    plt.show()

# def plot_state_aligned_feature_dynamics(df, feature, state_target, window=10, plot_individual=True):
#     """
#     Plot average and optionally individual traces of a given feature around transitions into a specified state.

#     Args:
#         df (pd.DataFrame): Merged dataframe including 'z_map' and target feature.
#         feature (str): Column name of the feature to analyze.
#         state_target (int): Hidden state to align transitions on.
#         window (int): Number of trials before and after transition to include.
#         plot_individual (bool): If True, plot individual traces as well.
#     """
#     z = df["z_map"].values
#     x = df[feature].values

#     transition_idxs = np.where((z[1:] == state_target) & (z[:-1] != state_target))[0] + 1
#     aligned_traces = []

#     for idx in transition_idxs:
#         if idx - window < 0 or idx + window >= len(x):
#             continue
#         segment = x[idx - window:idx + window + 1]
#         if np.any(np.isnan(segment)):
#             continue
#         aligned_traces.append(segment)

#     if not aligned_traces:
#         print("No valid transitions found.")
#         return

#     aligned_array = np.array(aligned_traces)
#     time_axis = np.arange(-window, window + 1)

#     plt.figure(figsize=(10, 4))
#     if plot_individual:
#         for trial in aligned_array:
#             plt.plot(time_axis, trial, alpha=0.2, color='gray')

#     mean_trace = np.mean(aligned_array, axis=0)
#     sem_trace = np.std(aligned_array, axis=0) / np.sqrt(aligned_array.shape[0])
#     plt.plot(time_axis, mean_trace, color='black', label="Mean")
#     plt.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, color='black', alpha=0.2)

#     plt.axvline(0, color='red', linestyle='--', label=f"Transition to State {state_target}")
#     plt.xlabel("Trials relative to state transition")
#     plt.ylabel(feature)
#     plt.title(f"{feature} aligned to transitions into state {state_target}")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

def run_state_transition_feature_scan(df, model, feature_cols, state_target, top_n=6, window=10):
    """
    For a given session, visualize top N features aligned to transitions into a specific state.
    
    Args:
        df (DataFrame): Merged session DataFrame (with z_map, features, etc.)
        model (HMM): Trained GLM-HMM model
        feature_cols (list): List of feature names
        state_target (int): Target hidden state to align to
        top_n (int): Number of top features to plot
        window (int): +/- time steps around transition to show
    """
    state_weights = model.observations.params[state_target, 0, :]
    abs_weights = np.abs(state_weights)
    
    sorted_idx = np.argsort(abs_weights)[::-1]
    sorted_features = [feature_cols[i] for i in sorted_idx[:top_n]]
    sorted_weights = state_weights[sorted_idx[:top_n]]
    
    print(f"\n📈 Top {top_n} features driving State {state_target} transitions:")
    for name, w in zip(sorted_features, sorted_weights):
        print(f"  {name:<22}: {w:+.3f}")

    for feature in sorted_features:
        print(f"\n🔍 Plotting feature: {feature}")
        plot_state_aligned_feature_dynamics(
            df,
            feature=feature,
            state_targets=[state_target],
            window=window,
            plot_individual=False  # Set True if you want trial-level overlays
        )

# def run_state_transition_feature_scan(df, model, feature_cols, state_target, top_n=6, window=10):
#     """
#     For a given session, visualize top N features aligned to transitions into a specific state.
    
#     Args:
#         df (DataFrame): Merged session DataFrame (with z_map, features, etc.)
#         model (HMM): Trained GLM-HMM model
#         feature_cols (list): List of feature names
#         state_target (int): Target hidden state to align to
#         top_n (int): Number of top features to plot
#         window (int): +/- time steps around transition to show
#     """
#     state_weights = model.observations.params[state_target, 0, :]
#     abs_weights = np.abs(state_weights)
    
#     sorted_idx = np.argsort(abs_weights)[::-1]
#     sorted_features = [feature_cols[i] for i in sorted_idx[:top_n]]
#     sorted_weights = state_weights[sorted_idx[:top_n]]
    
#     print(f"\n📈 Top {top_n} features driving State {state_target} transitions:")
#     for name, w in zip(sorted_features, sorted_weights):
#         print(f"  {name:<22}: {w:+.3f}")
    
#     plot_state_aligned_feature_dynamics(
#         df,
#         state_col="z_map",
#         feature_cols=sorted_features,
#         state_target=state_target,
#         window=window
#     )

def analyze_prediction_accuracy_by_reward(df, session_id=None):
    """
    Analyze model prediction accuracy split by reward outcome within a session.
    
    Args:
        df (pd.DataFrame): Full dataframe with all sessions.
        session_id (int or None): If provided, filters to a single session.
    
    Assumes df includes:
        - 'session_id', 'y_true', 'y_pred', 'rewarded'
        - optionally: 'response_time', 'rolling_choice_bias', 'is_opto'
    """
    if session_id is not None:
        df = df[df["session_id"] == session_id]
        print(f"\n📊 Analyzing Session {session_id}")
    else:
        print("\n📊 Analyzing All Sessions Combined")

    if 'rewarded' not in df.columns:
        df['rewarded'] = (df['y_true'] == df['correct_side'])  # fallback logic if needed

    results = {}
    for reward_status, label in [(1, "Rewarded"), (0, "Non-Rewarded")]:
        subset = df[df['rewarded'] == reward_status]
        if len(subset) == 0:
            continue
        accuracy = (subset['y_true'] == subset['y_pred']).mean()
        avg_rt = subset['response_time'].replace(-1, np.nan).mean()
        avg_bias = subset['rolling_choice_bias'].mean() if 'rolling_choice_bias' in subset else np.nan
        opto_rate = subset['is_opto'].mean() if 'is_opto' in subset else np.nan

        results[label] = {
            "n_trials": len(subset),
            "accuracy": accuracy,
            "avg_rt": avg_rt,
            "avg_bias": avg_bias,
            "opto_rate": opto_rate
        }

    for label, stats in results.items():
        print(f"\n[{label}]")
        print(f"  Trials         : {stats['n_trials']}")
        print(f"  Accuracy       : {stats['accuracy']:.3f}")
        print(f"  Avg RT (s)     : {stats['avg_rt']:.3f}")
        print(f"  Avg Bias       : {stats['avg_bias']:.3f}")
        print(f"  Opto Rate      : {stats['opto_rate']:.2%}")

def save_glm_hmm_model_and_results(model, results, mouse_id, output_dir="./saved_models"):
    """
    Save GLM-HMM model and results using explicit mouse ID.
    """
    filename_prefix = f"glm_{mouse_id}"
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{filename_prefix}_model.pkl")
    results_path = os.path.join(output_dir, f"{filename_prefix}_results.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved model to {model_path}")
    print(f"Saved results to {results_path}")

def load_glm_hmm_model_and_results(mouse_id, output_dir="./saved_models"):
    """
    Load GLM-HMM model and results using explicit mouse ID.
    """
    filename_prefix = f"glm_{mouse_id}"

    model_path = os.path.join(output_dir, f"{filename_prefix}_model.pkl")
    results_path = os.path.join(output_dir, f"{filename_prefix}_results.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    print(f"Loaded model from {model_path}")
    print(f"Loaded results from {results_path}")
    return model, results


def plot_performance_opto(M, config, subjectIdx, sessionIdx=-1, figure_id=None, show_plot=1, opto=False):
    # figure meta
    rowspan, colspan = 2, 8
    fig_size = get_figsize_from_pdf_spec(rowspan, colspan, config['pdf_spec']['pdf_pg_opto_psychometric'])    
    fig, ax = plt.subplots(figsize=fig_size) 

    subject = config['list_config'][subjectIdx]['subject_name']

    dates = M['dates']
    
    # convert session data to pandas df
    session_dfs = []
    for sessionIdx in range(0, len(dates)):
        df = get_session_df(M, sessionIdx)
        print(f"Session {sessionIdx} shape: {df.shape}")
        # print(df.columns.tolist())        
        session_dfs.append(df)
    
    # feature cols for modeling
    feature_cols = [
        'is_right', 'stim_duration', 'is_opto', 'norm_trial_index',
        'response_time', 'rolling_accuracy',
        'choice_1back', 'choice_2back', 'choice_3back',
        'reward_1back', 'reward_2back', 'reward_3back',
        'opto_1back', 'opto_2back', 'opto_3back',
        'stay_from_1back', 'rolling_choice_bias', 'exp_choice_bias',
        'isi_opto_interaction'
    ]
    
    # preprocess dataframes
    preprocessed = []
    for i, df in enumerate(session_dfs):        
        df['session_id'] = i
        df['mouse_id'] = subject
        df_proc = add_model_features(df)    
        
        # check features added
        missing_features = [f for f in feature_cols if f not in df_proc.columns]
        assert not missing_features, f"Missing features: {missing_features}" 
        
        preprocessed.append(df_proc)
    
    # one df for all sessions, if needed
    all_sessions_df = pd.concat(preprocessed, ignore_index=True)
       
    train_df, test_df, train_sids, test_sids = deterministic_train_test_split(all_sessions_df)

    X_train, y_train = prepare_model_inputs(train_df, feature_cols)
    X_test, y_test = prepare_model_inputs(test_df, feature_cols)
    
    retrain = 1
    if retrain:    
        glm_hmm = train_glm_hmm(X_train, y_train, num_states=3, num_categories=2)
    
    # all sessions
    # model_results = {}
    # for sid in all_sessions_df["session_id"].unique():
    #     df_sess = all_sessions_df[all_sessions_df["session_id"] == sid]
    #     X_sess, y_sess = prepare_model_inputs(df_sess, feature_cols)
        
    #     # Skip if session is empty
    #     if not X_sess or not y_sess:
    #         continue
    
    #     result = decode_glm_hmm(glm_hmm, X_sess, y_sess)[0]
    #     model_results[sid] = result
        
    model_results = {}
    
    if retrain:
        for sid in test_sids:
            df_sess = all_sessions_df[all_sessions_df["session_id"] == sid]
            X_sess, y_sess = prepare_model_inputs(df_sess, feature_cols)
        
            if not X_sess or not y_sess:
                continue
        
            result = decode_glm_hmm(glm_hmm, X_sess, y_sess)[0]
            model_results[sid] = result        

    mouse_id = train_df["mouse_id"].iloc[0]  # Assumes all rows in train_df share the same mouse_id

    if retrain:
        # save_glm_hmm_model_and_results(glm_hmm, model_results, "./saved_models", filename_prefix="glm_hmm_test_model")
        # Save model and results
        # save_glm_hmm_model_and_results(glm_hmm, model_results, "./saved_models", filename_prefix="glm_mouse01")
        save_glm_hmm_model_and_results(glm_hmm, model_results, mouse_id)
    else:
        # Later... load them back in
        # loaded_model, loaded_results = load_glm_hmm_model_and_results("./saved_models", filename_prefix="glm_mouse01")    
        glm_hmm_loaded, model_results_loaded = load_glm_hmm_model_and_results(mouse_id)
        glm_hmm = glm_hmm_loaded
        model_results = model_results_loaded
        

    print('')
    
    # # single session
    # session_id = 0
    # df_merged = merge_model_results(
    #     all_sessions_df[all_sessions_df["session_id"] == session_id],
    #     model_results[session_id],
    #     feature_cols
    # )
    # plot_session_overview(df_merged, session_id=session_id)

    # for session_id in test_sids:
    
    # check models weights across features
    for k in range(glm_hmm.K):
        print(f"\nState {k} weights:")
        weights = glm_hmm.observations.params[k]
        for i, feat in enumerate(feature_cols):
            print(f"  {feat:>20}: {weights[0, i]:.4f}")    
    
    session_id = test_sids[0]
        
    df_sess = all_sessions_df[all_sessions_df["session_id"] == session_id]
    
    if session_id not in model_results:
        print(f"[skip] Session {session_id} not decoded.")
        # continue

    df_merged = merge_model_results(
        df_sess,
        model_results[session_id],
        feature_cols
    )

    print_model_summary_stats(df_merged, session_id=session_id)
    analyze_prediction_accuracy_by_reward(df_merged, session_id=session_id)
    plot_session_overview(df_merged, session_id=session_id)
    # plot_state_transitions_overlay(df_merged, session_id=session_id)
    # plot_state_aligned_feature_dynamics(df_merged, feature, state_target, window=10)
    # plot_state_aligned_feature_dynamics(df_merged, feature="response_time", state_target=1, window=10, plot_individual=True)
    # run_state_transition_feature_scan(df_merged, glm_hmm, feature_cols, state_target=1)
   
    
    if 0:
        # Choose a state (e.g., 0, 1, 2)
        state_to_analyze = 1
        
        # Run scan — will print and plot top weighted features
        run_state_transition_feature_scan(
            df=df_merged,
            model=glm_hmm,
            feature_cols=feature_cols,
            state_target=state_to_analyze,
            top_n=6,
            window=10  # trials before/after transition
        )

    print('')
    # # Session-level Train/Test Split
    # train_sessions = [0, 1, 2]
    # test_sessions = [3, 4]
    
    # # def filter_by_sessions(df, session_ids):
    # #     return df[df['session_id'].isin(session_ids)].dropna(subset=feature_cols + ['mouse_choice'])
    
    # # filter according to session ID's, track number of trials dropped
    # def filter_by_sessions(df, session_ids, verbose=True):
    #     filtered_rows = []
    #     for sid in session_ids:
    #         df_sess = df[df['session_id'] == sid]
    #         before = len(df_sess)
    #         df_clean = df_sess.dropna(subset=feature_cols + ['mouse_choice'])
    #         after = len(df_clean)
    #         if verbose:
    #             print(f"Session {sid}: {before} → {after} trials (dropped {before - after})")
    #         filtered_rows.append(df_clean)
    #     return pd.concat(filtered_rows, ignore_index=True)    
    
    # # trials are filtered for nan values, check if too many trials are dropped
    # def report_dropped_trials(df, session_ids, feature_cols, target_col='mouse_choice'):
    #     dropped_reports = []
    #     for sid in session_ids:
    #         df_sess = df[df['session_id'] == sid]
    #         df_missing = df_sess[df_sess[feature_cols + [target_col]].isna().any(axis=1)]
    
    #         if not df_missing.empty:
    #             print(f"\nSession {sid} — Dropped Trials: {len(df_missing)}")
    #             print(df_missing[[target_col] + feature_cols].isna().sum())
    #             print(df_missing[[target_col] + feature_cols].head())  # show first few dropped
    
    #             dropped_reports.append((sid, df_missing))
    #         else:
    #             print(f"\nSession {sid} — No dropped trials.")
    
    #     return dropped_reports    
    
    # train_df = filter_by_sessions(all_sessions_df, train_sessions)
    # test_df = filter_by_sessions(all_sessions_df, test_sessions)
    
    # dropped_trials = report_dropped_trials(
    #     df=all_sessions_df,
    #     session_ids=test_sessions,
    #     feature_cols=feature_cols
    # )    
    
    # dropcheck = []
    # for df in preprocessed:
    #     # df_copy = df.copy()
    #     df['dropped_reason'] = df[feature_cols + ['mouse_choice']].isna().any(axis=1)
    #     dropcheck.append(df)
    
    # X_train, y_train = prepare_model_inputs(train_df, feature_cols)
    # X_test, y_test = prepare_model_inputs(test_df, feature_cols)
    
        
    # glm_hmm, results = run_glm_hmm_validation(X_train, y_train, X_test, y_test)

    # for i, res in enumerate(results):
    #     print(f"Test session {i} accuracy: {res['accuracy']:.3f}")
    #     plot_predictions(results, i)
    #     plot_state_trajectory(results, i)
    #     plot_confusion_matrix(results, i)
    #     plot_choice_probs_by_state(results, i)    
    
    # # Example session
    # train_session_ids = train_df["session_id"].unique()
    
    # # Loop through model_results and associated session IDs
    # for result_idx, session_id in enumerate(train_session_ids):
    #     df_session = all_sessions_df[all_sessions_df["session_id"] == session_id]
    #     df_merged = merge_model_results(df_session, results[result_idx], feature_cols)
    #     plot_session_overview(df_merged, session_id=session_id)
    
    print('')
    
    
    # for df in preprocessed:
    

    # results = decode_glm_hmm(model, X_list, y_list)




    if show_plot:
        plt.show()           
        
    subject = config['list_config'][subjectIdx]['subject_name']
    output_dir = os.path.join(config['paths']['figure_dir_local'] + subject)
    figure_id = f"{subject}_decision_time_violin_outcome"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)    
      
    return out_path    