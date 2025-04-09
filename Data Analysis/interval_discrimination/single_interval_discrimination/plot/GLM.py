import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import pandas as pd
from types import SimpleNamespace
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # enables 3D projection
import matplotlib.animation

print_debug = 1
# bin the data with timestamps.


def get_processed_df(session_data, session_idx):
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
        licks['licks_left_start'] = [x - alignment for x in licks['licks_left_start']]
        licks['licks_left_stop'] = [x - alignment for x in licks['licks_left_stop']]
        licks['licks_right_start'] = [x - alignment for x in licks['licks_right_start']]
        licks['licks_right_stop'] = [x - alignment for x in licks['licks_right_stop']]
  
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
        flash_duration = raw_data[session_idx]['TrialSettings'][trial]['GUI']['GratingDur_s'] * 1000
        # flash_duration = session_data['session_settings'][session_idx]['GratingDur_s'][trial] * 1000
        stim_duration = 2 * flash_duration + isi    
        
        ContinuousCurrent = raw_data[session_idx]['TrialSettings'][trial]['GUI']['ContinuousCurrent']        
        
        licked_right = 0
        if not no_lick:
            if rewarded:
                if trial_type == 'left':
                    licked_right = 0
                else:
                    licked_right = 1
            else:
                if trial_type == 'left':
                    licked_right = 1
                else:
                    licked_right = 0
        
        post_stim_delay_vector = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['PostVisStimDelay']
        post_stim_delay = post_stim_delay_vector[1] - post_stim_delay_vector[0]
            
        move_correct_spout = session_data['move_correct_spout_flag'][session_idx][trial]
                
        processed_dec.append({
            "trial_idx": trial,
            "trial_side": trial_type,
            "isi": isi,
            "flash_duration": flash_duration,
            "stim_duration": stim_duration,
            "post_stim_delay_vector": post_stim_delay_vector,
            "post_stim_delay": post_stim_delay,
            "ContinuousCurrent": ContinuousCurrent,
            "is_opto": is_opto,
            "is_naive": is_naive,
            "rewarded": rewarded,
            "no_lick": no_lick,
            "licked_right": licked_right,
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


# # DataFrame Accessor class
# @pd.api.extensions.register_dataframe_accessor("opto")
# class OptoTools:
#     def __init__(self, df):
#         self._df = df

#     def assign_bins(self, bin_col='isi', bin_size=0.1, max_val=None):
#         max_val = max_val or self._df[bin_col].max()
#         bins = np.arange(0, max_val + bin_size, bin_size)
#         bin_indices = np.digitize(self._df[bin_col], bins) - 1
#         self._df['bin_idx'] = bin_indices
#         self._df['bin_edge'] = bins[bin_indices]
#         return self._df

#     def reward_stats(self):
#         grouped = self._df.groupby(['bin_idx', 'trial_side'])
#         result = grouped.apply(lambda g: pd.Series({
#             'percent_reward': (g['rewarded'] == True).mean() * 100,
#             'sem_reward': (g['rewarded'] == True).std(ddof=1) / np.sqrt(len(g))
#         }))
#         return result.reset_index()


def get_PCA(df):
    # "trial_idx": trial,
    # "trial_side": trial_type,
    # "isi": isi,
    # "flash_duration": flash_duration,
    # "stim_duration": stim_duration,
    # "post_stim_delay_vector": post_stim_delay_vector,
    # "post_stim_delay": post_stim_delay,
    # "ContinuousCurrent": ContinuousCurrent,
    # "": is_opto,
    # "is_naive": is_naive,
    # "rewarded": rewarded,
    # "no_lick": no_lick,
    # "licked_right": licked_right,
    # "opto_encode": opto_encode,
    # "move_correct_spout": move_correct_spout,
    # "licks_left_start": licks['licks_left_start'],
    # "licks_left_stop": licks['licks_left_stop'],
    # "licks_right_start": licks['licks_right_start'],
    # "licks_right_stop": licks['licks_right_stop'],
    # "valve_start": valve_times[0],
    # "valve_stop": valve_times[1]
    df = df[(df['is_naive'] == False)]
    df = df[(df['no_lick'] == False)]
    df = df[(df['move_correct_spout'] == False)]   
    df = df.dropna()
    
    
    features = ['isi', 
                'is_opto', 
                'licked_right', 
                'rewarded',
                'stim_duration',
                'post_stim_delay',
                'opto_encode',
                ]  # Add more as needed
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=3)  # or more
    X_pca = pca.fit_transform(X_scaled)
    # df['PC1'] = X_pca[:, 0]
    # df['PC2'] = X_pca[:, 1]
    # df['PC3'] = X_pca[:, 2]
    for i in range(X_pca.shape[1]):
        df[f'PC{i+1}'] = X_pca[:, i]    
    
    
    plt.figure(figsize=(6, 5))
    for opto_val in [0, 1]:
        subset = df[df['is_opto'] == opto_val]
        label = 'Opto' if opto_val else 'Control'
        plt.scatter(subset['PC1'], subset['PC2'], alpha=0.6, label=label)
        # plt.scatter(df['PC1'], df['PC2'], c=df['trial_idx'], cmap='viridis')
    
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.title('PCA of Trials')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
          
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by opto
    colors = df['is_opto'].map({0: 'gray', 1: 'green'})
    colors=df['trial_idx'], cmap='viridis'
    
    ax.scatter(df['PC1'], df['PC2'], df['PC3'], c=colors, alpha=0.6)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA of Trials')
    plt.tight_layout()
    plt.show()        
        
    print(pca.explained_variance_ratio_)       
    
    # `pca` is your fitted PCA object
    # `features` is the list of original column names you input into PCA
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=features)
    
    print(loadings)    
            
    return

def GLM(processed_dec):
    # filter tags
    df = processed_dec[(processed_dec['is_naive'] == False)]
    df = df[(df['no_lick'] == False)]
    df = df[(df['move_correct_spout'] == False)]

    # Optional: z-score continuous variables
    df['z_stim'] = (df['stim_duration'] - df['stim_duration'].mean()) / df['stim_duration'].std()
    df['z_trial'] = (df['trial_idx'] - df['trial_idx'].mean()) / df['trial_idx'].std()     # trial idx to examine drift
    df['z_post_stim_delay'] = (df['post_stim_delay'] - df['post_stim_delay'].mean()) / df['post_stim_delay'].std()
    
    # df['z_stim_2'] = zscore(df['stim_duration'], nan_policy='omit')
    
    # post_stim_delay
    
    # Ensure opto is binary
    df['is_opto'] = df['is_opto'].astype(int)
    
    # Logistic regression model formula
    formula = 'licked_right ~ z_stim + z_trial + is_opto + z_stim:is_opto'
    
    # Fit GLM (logistic regression)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()    
    
    
    print(model.summary())
    
    return df, model

def plot_GLM(df, model):
    # Create range of ISIs for plotting
    isi_vals = np.linspace(df['z_stim'].min(), df['z_stim'].max(), 100)
    opto_vals = [0, 1]
    
    plot_df = pd.DataFrame({
        'z_stim': np.tile(isi_vals, 2),
        'z_trial': 0,  # average trial index (remove drift)
        'is_opto': np.repeat(opto_vals, len(isi_vals))
    })
    
    # Create interaction term
    plot_df['z_stim:is_opto'] = plot_df['z_stim'] * plot_df['is_opto']
    
    # Predict choice probability
    plot_df['pred_prob'] = model.predict(plot_df)
    
    # Plot
    import matplotlib.pyplot as plt
    
    for opto_state in [0, 1]:
        subset = plot_df[plot_df['is_opto'] == opto_state]
        label = 'is_opto' if opto_state else 'Control'
        plt.plot(subset['z_stim'], subset['pred_prob'], label=label)
    
    plt.xlabel('Stimulus Duration (z-scored)')
    plt.ylabel('P(right lick)')
    plt.legend()
    plt.title('Psychometric Curve: Opto vs Control')
    plt.show()    



def filter_df(processed_dec):
    # filter tags
    filtered_df = processed_dec[(processed_dec['is_naive'] == False)]
    filtered_df = filtered_df[(filtered_df['no_lick'] == False)]
    filtered_df = filtered_df[(filtered_df['move_correct_spout'] == False)]        
    
    return filtered_df


def run(ax, subject_session_data, session_num=-1):

    subject_session_data_copy = subject_session_data.copy()
    
    # if not start_from=='std':
    #     start_date = subject_session_data[start_from]
    #     dates = subject_session_data['dates']
    #     if start_date in dates:
    #         start_idx = dates.index(start_date)
    #     else:
    #         return
            
    #     for key in subject_session_data_copy.keys():
    #         # print(key)
    #         if isinstance(subject_session_data_copy[key], list) and len(subject_session_data_copy[key]) == len(dates):
    #             subject_session_data_copy[key] = subject_session_data_copy[key][start_idx:]  
    
    # date = subject_session_data_copy['dates'][session_num]
    # if date == '20250318':
    #     print(date)
    
    subject = subject_session_data_copy['subject']
    dates = subject_session_data_copy['dates']
    
    
    session_settings = subject_session_data_copy['session_settings'][session_num]
    isi_short_mean = session_settings['ISIShortMean_s'] * 1000
    isi_long_mean = session_settings['ISILongMean_s'] * 1000
    isi_orig = session_settings['ISIOrig_s'] * 1000
    
    # is_avg = 0
    # if session_num != -1:
    #     processed_df = get_processed_df(subject_session_data_copy, session_num)
    # else:
    #     is_avg = 1
    #     combined_df = pd.DataFrame()
    #     for session_num in range(0, len(dates)):
    #         processed_df = get_processed_df(subject_session_data_copy, session_num)
    #         combined_df = pd.concat([combined_df, processed_df], ignore_index=True)
    #     processed_df = combined_df
    
    # num_residuals = 5
    # opto_residual_means, opto_residual_sems, opto_residual_isis = bin_opto_residuals(processed_df, num_residuals)
    
    # control_means, control_sems, control_isis = bin_control(processed_df)
    
    # decision_fix, decision_jitter, decision_chemo, decision_opto, decision_opto_left, decision_opto_right = get_decision(subject_session_data_copy, session_num)
    # bin_mean_fix, bin_sem_fix, bin_isi_fix = get_bin_stat(decision_fix, session_settings)
    # bin_mean_jitter, bin_sem_jitter, bin_isi_jitter = get_bin_stat(decision_jitter, session_settings)
    # bin_mean_chemo, bin_sem_chemo, bin_isi_chemo = get_bin_stat(decision_chemo, session_settings)
    # bin_mean_opto, bin_sem_opto, bin_isi_opto = get_bin_stat(decision_opto, session_settings)
    
    # bin_mean_opto_left, bin_sem_opto_left, bin_isi_opto_left = get_bin_stat(decision_opto_left, session_settings)
    # bin_mean_opto_right, bin_sem_opto_right, bin_isi_opto_right = get_bin_stat(decision_opto_right, session_settings)
    
    processed_df = get_processed_df(subject_session_data_copy, session_num)
    # df = filter_df(processed_df)
    df, model = GLM(processed_df)
    plot_GLM(df, model)
    
    get_PCA(processed_df)
    
    # Function to generate 'n' green colors, from light to dark
    def generate_green_colors(n, min_green=120, max_green=230, shift=0):
        greens = []
        for i in range(n):
            # # The green component ranges from 255 (light) to 0 (dark)
            # green_value = int(255 * ( i / (n - 1)))  # Linear interpolation
            # # greens.append((0, green_value, 0))  # RGB format: (Red, Green, Blue)
            # greens.append(np.array([0, green_value, 0]) / 255)  # Normalize to [0, 1]
            
            """Generate n green shades between min_green and max_green intensity."""
            green_vals = np.linspace(max_green, min_green, n).astype(int)  # light to dark
            green_vals = green_vals - shift
            greens = [np.array([0, g, 0]) / 255 for g in green_vals]            
            
        return greens
    
    def generate_blue_colors(n):
        blues = []
        for i in range(n):
            # The green component ranges from 255 (light) to 0 (dark)
            blue_value = int(255 * ( i / (n - 1)))  # Linear interpolation
            # greens.append((0, green_value, 0))  # RGB format: (Red, Green, Blue)
            blues.append(np.array([blue_value, 0, 0]) / 255)  # Normalize to [0, 1]
        return blues    
    
    num_residuals = 5
    mean_greens = generate_green_colors(num_residuals)
    sem_greens = generate_green_colors(num_residuals, shift=100)
    
    def generate_opto_labels(num):
        labels = []
        labels.append('opto')
        for i in range(0, num):        
            if i != 0:
                labels.append('opto + ' + str(i))
        return labels
    
    labels = generate_opto_labels(num_residuals)
    


    
    x_left = isi_short_mean - 100
    x_right = isi_long_mean + 100
    cat = isi_orig
    x_left = 0
    x_right = 2*cat
    
    ax.vlines(
        cat, 0.0, 1.0,
        linestyle='--', color='mediumseagreen',
        label='Category Boundary')
    ax.hlines(0.5, x_left, x_right, linestyle='--', color='grey')
    # ax.vlines(500, 0.0, 1.0, linestyle=':', color='grey')
    ax.tick_params(tick1On=False)
    ax.tick_params(axis='x', rotation=45)    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([x_left,x_right])
    ax.set_ylim([-0.05,1.05])
    # ax.set_xticks(np.arange(6)*200)
    # ax.set_xticks(np.arange(11)*150)
    ax.set_xticks(np.arange(0,x_right,250))
    ax.set_yticks(np.arange(5)*0.25)
    ax.set_xlabel('post perturbation isi')
    ax.set_ylabel('prob. of choosing the right side (mean$\pm$sem)')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    start_from = []
    start_date = []
    date = subject_session_data_copy['dates'][session_num]
    if start_from=='start_date':
        ax.set_title('average psychometric function from ' + start_date)
    elif start_from=='non_naive':
        ax.set_title('average psychometric function non-naive')
    else:
        if not is_avg:
            ax.set_title('residual opto psychometric function ' + date)
        else:
            ax.set_title('average residual opto psychometric function ' + dates[0] + '-' + dates[-1])
        