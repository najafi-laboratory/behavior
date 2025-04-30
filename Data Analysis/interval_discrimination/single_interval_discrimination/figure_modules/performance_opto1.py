from scipy.stats import sem
import os
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

def fit_glm_hmm(X_list, y_list, num_states=3, num_iters=100, verbose=True):
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

def patch_injector(model):
    """
    Monkey-patch the _objective function of the InputDrivenObservations 
    instance used in a GLM-HMM model to add debug prints and avoid crashing 
    on empty inputs.
    """

    def patched_objective(self, params, k):
        print("[🐒 Using patched _objective from monkey patch]")
        W = self.unpack_params(params)
        input = self.input
        data = self.data

        if input.shape[0] == 0:
            print(f"[⚠️ WARNING] Empty input for state {k}, skipping.")
            return 0.0

        print(f"[DEBUG] State {k} | input shape: {input.shape}, W.T shape: {W.T.shape}")

        xproj = input @ W.T

        from ssm.util import _multisoftplus, one_hot
        f, _ = _multisoftplus(xproj)
        data_one_hot = one_hot(data, self.C)

        expected_states = self.expected_states[:, k]
        temp_obj = (-np.sum(data_one_hot[:, :-1] * xproj, axis=1) + f) @ expected_states
        return temp_obj

    # Bind the patch to the model's observations object
    model.observations._objective = patched_objective.__get__(model.observations)
   
    
    return

def plot_performance_opto(M, config, subjectIdx, sessionIdx=-1, figure_id=None, show_plot=1, opto=False):
    # figure meta
    rowspan, colspan = 2, 8
    fig_size = get_figsize_from_pdf_spec(rowspan, colspan, config['pdf_spec']['pdf_pg_opto_psychometric'])    
    fig, ax = plt.subplots(figsize=fig_size) 

    subject = config['list_config'][subjectIdx]['subject_name']

    dates = M['dates']
    session_dfs = []
    for sessionIdx in range(0, len(dates)):
        df = get_session_df(M, sessionIdx)
        print(f"Session {sessionIdx} shape: {df.shape}")
        print(df.columns.tolist())        
        session_dfs.append(df)
    
    preprocessed = []
    for i, df in enumerate(session_dfs):        
        df['session_id'] = i
        df['mouse_id'] = subject
        df_proc = add_model_features(df)        
        
        preprocessed.append(df_proc)
    
    all_sessions_df = pd.concat(preprocessed, ignore_index=True)
    
    
    print('')
    
    
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
    assert not missing_features, f"Missing features: {missing_features}"    
        
    # debuggin serum |--|========>------^
    # patch_injector
    
    for df in preprocessed:
    
        X, y, scaler = preprocess_for_model(df, feature_cols, target_col='mouse_choice')
        
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        assert X.shape[0] == y.shape[0], "Mismatch in number of trials between X and y"
        
        
        X_list = []
        y_list = []
        session_ids = df['session_id'].unique()
        
        for sid in session_ids:
            df_session = df[df['session_id'] == sid].dropna(subset=feature_cols + ['mouse_choice'])
            X_list.append(df_session[feature_cols].values)
            y_list.append(df_session['mouse_choice'].values)
            
            print(f"Session {sid}: X {df_session[feature_cols].shape}, y {df_session['mouse_choice'].shape}")

        
    
        # # These are lists of arrays, one per session
        # X_list  # trial features per session (shape [T, D])
        # y_list  # mouse choices per session (shape [T])
        
        for i, X in enumerate(X_list):
            if not isinstance(X, np.ndarray) or X.ndim != 2:
                print(f"Session {i}: X shape issue → {type(X)} with shape {getattr(X, 'shape', None)}")

        for i, y in enumerate(y_list):
            if not isinstance(y, np.ndarray) or y.ndim != 1:
                print(f"Session {i}: y shape issue → {type(y)} with shape {getattr(y, 'shape', None)}")
            elif not np.all(np.isin(y, [0, 1])):
                print(f"Session {i}: y values not binary: {np.unique(y)}")
        
        X_list_clean, y_list_clean = validate_glm_hmm_inputs(X_list, y_list)
        X_list = X_list_clean
        y_list = y_list_clean
        
        # Now safe to pass to model
        # glm_hmm.fit(y_list_clean, inputs=X_list_clean, method="em", num_iters=100)    
    
        for i, (X, y) in enumerate(zip(X_list, y_list)):
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert X.ndim == 2
            assert y.ndim == 1
            assert X.shape[0] == y.shape[0]
            assert np.all(np.isin(y, [0, 1]))
    
    
        num_states = 3          # You can tune this later
        num_categories = 2  # left (0) and right (1)
        input_dim = X_list[0].shape[1]
        assert all(x.shape[1] == input_dim for x in X_list), "Mismatch in input_dim across sessions"
        
        # Fit model
        glm_hmm = fit_glm_hmm(X_list, y_list, num_states=3)
        
        # glm_hmm = ssm.HMM(
        #     K=num_states,
        #     D=input_dim,
        #     observation_kwargs=dict(C=num_categories),
        #     observations="categorical"
        # )      
        
        # glm_hmm = create_patched_glm_hmm(num_states=3, num_features=X_list_clean[0].shape[1])
        
        # glm_hmm = create_patched_glm_hmm(
        #     num_states=3, 
        #     num_features=input_dim,
        #     num_classes=num_categories
        #     )
        
        # Create GLM-HMM: Bernoulli emission (for binary choice), with input-dependent GLM per state
        # glm_hmm = ssm.HMM(
        #     K=num_states,
        #     D=input_dim,
        #     M=num_obs,
        #     observations="input_driven_obs",
        #     observation_kwargs=dict(C=num_obs, link="logit")
        # )    

        # glm_hmm = ssm.HMM(
        #     K=num_states,
        #     D=input_dim,
        #     M=1,
        #     observations="input_driven_obs",
        # )            
       
        print("Using patched logpdf?", ssm.stats.categorical_logpdf.__name__)
        
        # original_em_step = glm_hmm._fit_em

        # def traced_em_step(*args, **kwargs):
        #     print("Entering EM...")
        #     out = original_em_step(*args, **kwargs)
            
        #     Ez = out['expected_states']
        #     Ezz = out['expected_joints']
        #     print(f"  γ (Ez) shape: {Ez[0].shape}")
        #     print(f"  ξ (Ezz) shape: {Ezz[0].shape}")
            
        #     return out
        
        # glm_hmm._fit_em = traced_em_step
        
        # Fit using EM (expectation-maximization)
        # model.fit(
        #     datas=y_list,
        #     inputs=X_list,
        #     method="em",
        #     # num_iters=100
        #     tolerance=1e-4
        # )   
        
        i = 0  # session index
        z_probs = glm_hmm.expected_states(y_list[i], input=X_list[i])[0]  # posterior state probabilities
        z_map = np.argmax(z_probs, axis=1)  # most likely state per trial
        # y_pred = glm_hmm.predict(y_list[i], input=X_list[i])  # predicted choice   

        # Get the state posteriors first
        gamma, _, _ = glm_hmm.expected_states(y_list[i], input=X_list[i])
        
        # Compute weighted prediction over each state's GLM
        pred_probs = np.zeros((X_list[i].shape[0], glm_hmm.observations.C))
        
        for k in range(glm_hmm.K):
            W = glm_hmm.observations.params[k]  # shape (C-1, D)
            logits = X_list[i] @ W.T  # shape (T, C-1)
        
            # Convert to full logits with last class = 0
            logits_full = np.hstack([logits, np.zeros((logits.shape[0], 1))])
            probs = np.exp(logits_full - logsumexp(logits_full, axis=1, keepdims=True))
        
            pred_probs += gamma[:, k:k+1] * probs  # soft mixture over states
        
        # Now get hard prediction (argmax)
        y_pred = np.argmax(pred_probs, axis=1)
        
        print(f"Posterior shape: {z_probs.shape}, MAP state shape: {z_map.shape}, y_pred shape: {y_pred.shape}")

       
        # Fit model
        # model = fit_glm_hmm(X_list, y_list, num_states=3)
        
        # Decode predictions and states
        results = decode_glm_hmm(glm_hmm, X_list, y_list)
        
        # Plot first session
        plot_glm_hmm_session(results[0], session_id=session_ids[0])    
       
        # plot_predictions(results, session_idx=session_ids[0])
        # plot_confusion_matrix(results, session_idx=session_ids[0])
        # plot_state_trajectory(results, session_idx=session_ids[0])
        # plot_choice_probs_by_state(results, session_idx=session_ids[0])
        
        plot_predictions(results, session_idx=0)
        plot_confusion_matrix(results, session_idx=0)
        plot_state_trajectory(results, session_idx=0)
        plot_choice_probs_by_state(results, session_idx=0)        
    
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