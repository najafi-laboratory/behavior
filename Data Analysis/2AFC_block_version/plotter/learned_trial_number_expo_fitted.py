import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import sem, wilcoxon, binomtest
import pandas as pd
from matplotlib.gridspec import GridSpec
import os

def analyze_learning_dynamics_fitted(sessions_data, subject, data_paths, save_path=None, example_session_idx=0):
    """
    Analyzes learning dynamics using Exponential Curve Fitting.
    Visualizes results using the comprehensive 9-panel layout.
    """
    
    # --- Configuration ---
    ANALYSIS_WINDOW = 20      # Trials after transition to analyze
    ROLLING_WINDOW_SIZE = 3   # For meta-learning smoothing
    LEARNING_THRESHOLD = 0.95 # 95% of asymptote = "learned"
    
    # --- Helper Functions ---
    
    def exponential_growth(t, asymptote, scale, tau):
        """Learning curve: P(t) = Asymptote - Scale * exp(-t/tau)"""
        return asymptote - scale * np.exp(-t / max(1e-5, tau))
    
    def fit_single_transition(trial_outcomes):
        """
        Fits exponential to a single transition.
        Returns: 
            learned_trial: Time to reach 95% of asymptote
            popt: (asymptote, scale, tau)
            r_squared: Quality of fit
        """
        if len(trial_outcomes) < 5 or np.all(np.isnan(trial_outcomes)):
            return np.nan, (np.nan, np.nan, np.nan), 0
            
        t_values = np.arange(len(trial_outcomes))
        valid_mask = ~np.isnan(trial_outcomes)
        
        if np.sum(valid_mask) < 5:
            return np.nan, (np.nan, np.nan, np.nan), 0
        
        t_clean = t_values[valid_mask]
        y_clean = trial_outcomes[valid_mask]
        
        try:
            # Initial guess
            y_start = np.mean(y_clean[:3]) if len(y_clean) >= 3 else y_clean[0]
            y_end = np.mean(y_clean[-3:]) if len(y_clean) >= 3 else y_clean[-1]
            
            # Constraints: Asymptote [0.5, 1.0], Scale [0, 1], Tau [0.1, Window]
            p0 = [max(0.7, y_end), max(0.1, y_end - y_start), 5]
            bounds = ([0.5, 0.0, 0.1], [1.0, 1.0, ANALYSIS_WINDOW])
            
            popt, pcov = curve_fit(exponential_growth, t_clean, y_clean, 
                                 p0=p0, bounds=bounds, maxfev=2000)
            asymptote, scale, tau = popt
            
            # Calculate R-squared
            residuals = y_clean - exponential_growth(t_clean, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))

            # Time to reach threshold: 
            # P(t) = A - S*exp(-t/tau) = A * threshold
            # This derivation assumes we want % of the ASYMPTOTE, not absolute performance
            # However, for 2AFC, usually we want time to reach specific accuracy (e.g. 0.8).
            # Using your logic: 95% of the curve's own saturation.
            
            if scale == 0:
                learned_trial = 0
            else:
                # Solve for t where value is (Asymptote - Scale * (1-0.95))
                # Essentially when the exponential component has decayed by 95%
                learned_trial = -tau * np.log(1 - LEARNING_THRESHOLD)
            
            # If learned trial is outside window or fit is terrible, mark as NaN (failed)
            if learned_trial > ANALYSIS_WINDOW or r_squared < 0.01:
                return np.nan, popt, r_squared
            
            return learned_trial, popt, r_squared
            
        except Exception as e:
            return np.nan, (np.nan, np.nan, np.nan), 0
    
    def nansem(data):
        clean = data[~np.isnan(data)]
        return sem(clean) if len(clean) > 0 else np.nan

    # --- Main Processing ---
    
    dates = sessions_data['dates']
    outcomes_list = sessions_data['outcomes']
    block_types_list = sessions_data['block_type']
    
    # Storage
    all_transitions = {
        's2l': {'times': [], 'sessions': [], 'trial_data': [], 'asymptotes': [], 'r2': []},
        'l2s': {'times': [], 'sessions': [], 'trial_data': [], 'asymptotes': [], 'r2': []}
    }
    
    session_summaries = {'s2l': [], 'l2s': [], 'dates': []}
    
    # Loop Sessions
    for session_idx, (date, outcomes, block_types) in enumerate(zip(dates, outcomes_list, block_types_list)):
        
        sess_s2l = []
        sess_l2s = []
        
        for trial_idx in range(1, len(block_types)):
            if trial_idx + ANALYSIS_WINDOW >= len(outcomes): continue
            
            prev = block_types[trial_idx-1]
            curr = block_types[trial_idx]
            if prev == curr: continue
            
            # Extract data
            data_segment = []
            for k in range(ANALYSIS_WINDOW):
                data_segment.append(1.0 if outcomes[trial_idx + k] == 'Reward' else 0.0)
            data_segment = np.array(data_segment)
            
            # --- FIT CURVE ---
            t_crit, params, r2 = fit_single_transition(data_segment)
            asymp = params[0] # The 'A' parameter
            
            # Store
            if prev == 1 and curr == 2: # S->L
                all_transitions['s2l']['times'].append(t_crit)
                all_transitions['s2l']['sessions'].append(session_idx)
                all_transitions['s2l']['trial_data'].append(data_segment)
                all_transitions['s2l']['asymptotes'].append(asymp)
                all_transitions['s2l']['r2'].append(r2)
                sess_s2l.append(t_crit)
            elif prev == 2 and curr == 1: # L->S
                all_transitions['l2s']['times'].append(t_crit)
                all_transitions['l2s']['sessions'].append(session_idx)
                all_transitions['l2s']['trial_data'].append(data_segment)
                all_transitions['l2s']['asymptotes'].append(asymp)
                all_transitions['l2s']['r2'].append(r2)
                sess_l2s.append(t_crit)
                
        # Session Averages
        session_summaries['s2l'].append(np.nanmean(sess_s2l) if sess_s2l else np.nan)
        session_summaries['l2s'].append(np.nanmean(sess_l2s) if sess_l2s else np.nan)
        session_summaries['dates'].append(date)

    # Convert lists to arrays
    for k in ['s2l', 'l2s']:
        for subk in all_transitions[k].keys():
            all_transitions[k][subk] = np.array(all_transitions[k][subk])
    
    # Clean arrays for plotting (remove NaNs)
    s2l_clean = all_transitions['s2l']['times'][~np.isnan(all_transitions['s2l']['times'])]
    l2s_clean = all_transitions['l2s']['times'][~np.isnan(all_transitions['l2s']['times'])]
    
    # --- PLOTTING (The 9-Panel Structure) ---
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    COLOR_S2L = '#2E86AB'
    COLOR_L2S = '#A23B72'
    COLOR_MEAN = '#F18F01'
    
    # 1. Summary Comparison (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    x_s2l = np.random.normal(1, 0.04, size=len(s2l_clean))
    x_l2s = np.random.normal(2, 0.04, size=len(l2s_clean))
    
    ax1.scatter(x_s2l, s2l_clean, alpha=0.4, s=40, color=COLOR_S2L, label='S→L Fits')
    ax1.scatter(x_l2s, l2s_clean, alpha=0.4, s=40, color=COLOR_L2S, label='L→S Fits')
    ax1.errorbar(1, np.mean(s2l_clean), yerr=sem(s2l_clean), fmt='o', color=COLOR_MEAN, 
                 capsize=8, markersize=14, markeredgecolor='k', zorder=3)
    ax1.errorbar(2, np.mean(l2s_clean), yerr=sem(l2s_clean), fmt='o', color=COLOR_MEAN, 
                 capsize=8, markersize=14, markeredgecolor='k', zorder=3)
    
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Short → Long', 'Long → Short'], fontweight='bold')
    ax1.set_ylabel('Fitted Time Constant (Trials)', fontweight='bold')
    ax1.set_title('Learning Speed (Exponential Fit)', fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Histograms (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, ANALYSIS_WINDOW, 12)
    ax2.hist(s2l_clean, bins=bins, alpha=0.5, color=COLOR_S2L, density=True, label='S→L')
    ax2.hist(l2s_clean, bins=bins, alpha=0.5, color=COLOR_L2S, density=True, label='L→S')
    ax2.axvline(np.mean(s2l_clean), color=COLOR_S2L, linestyle='--', linewidth=2)
    ax2.axvline(np.mean(l2s_clean), color=COLOR_L2S, linestyle='--', linewidth=2)
    ax2.set_xlabel('Trials to 95% Asymptote', fontweight='bold')
    ax2.set_title('Distribution of Learning Times', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Success Rate (Top Right) - Based on valid fits
    ax3 = fig.add_subplot(gs[0, 2])
    n_s2l = len(all_transitions['s2l']['times'])
    n_l2s = len(all_transitions['l2s']['times'])
    succ_s2l = len(s2l_clean) / n_s2l * 100
    succ_l2s = len(l2s_clean) / n_l2s * 100
    
    ax3.bar([1, 2], [succ_s2l, succ_l2s], color=[COLOR_S2L, COLOR_L2S], alpha=0.7, edgecolor='k')
    ax3.bar([1, 2], [100-succ_s2l, 100-succ_l2s], bottom=[succ_s2l, succ_l2s], 
            color='lightgray', alpha=0.5, edgecolor='k', label='Failed Fit')
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['S→L', 'L→S'], fontweight='bold')
    ax3.set_ylabel('% Successful Fits', fontweight='bold')
    ax3.set_title('Fit Convergence Rate', fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # 4. Average Curves (Middle Row)
    ax4 = fig.add_subplot(gs[1, :])
    # Compute raw average for visualization
    s2l_mat = np.array([t for t in all_transitions['s2l']['trial_data'] if len(t)==ANALYSIS_WINDOW])
    l2s_mat = np.array([t for t in all_transitions['l2s']['trial_data'] if len(t)==ANALYSIS_WINDOW])
    
    ax4.plot(np.nanmean(s2l_mat, axis=0), color=COLOR_S2L, linewidth=3, label='S→L (Raw Avg)')
    ax4.plot(np.nanmean(l2s_mat, axis=0), color=COLOR_L2S, linewidth=3, label='L→S (Raw Avg)')
    ax4.fill_between(range(ANALYSIS_WINDOW), 
                     np.nanmean(s2l_mat, axis=0)-sem(s2l_mat, axis=0),
                     np.nanmean(s2l_mat, axis=0)+sem(s2l_mat, axis=0), color=COLOR_S2L, alpha=0.2)
    ax4.fill_between(range(ANALYSIS_WINDOW), 
                     np.nanmean(l2s_mat, axis=0)-sem(l2s_mat, axis=0),
                     np.nanmean(l2s_mat, axis=0)+sem(l2s_mat, axis=0), color=COLOR_L2S, alpha=0.2)
    
    ax4.set_xlabel('Trials Since Transition', fontweight='bold')
    ax4.set_ylabel('Probability Correct', fontweight='bold')
    ax4.set_title('Population Average Behavior', fontweight='bold')
    ax4.legend()
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # 5. Within Session (Bottom Left)
    ax5 = fig.add_subplot(gs[2, :2])
    if example_session_idx < len(dates):
        s_idx = example_session_idx
        s_times_s2l = all_transitions['s2l']['times'][all_transitions['s2l']['sessions'] == s_idx]
        s_times_l2s = all_transitions['l2s']['times'][all_transitions['l2s']['sessions'] == s_idx]
        
        # Combine and sort strictly by occurrence isn't easy here without original order
        # We will plot them side by side
        all_y = []
        all_x = []
        colors = []
        
        # Hacky plotting for visualization
        for i, t in enumerate(s_times_s2l):
            all_y.append(t if not np.isnan(t) else ANALYSIS_WINDOW)
            all_x.append(i*2)
            colors.append(COLOR_S2L)
        for i, t in enumerate(s_times_l2s):
            all_y.append(t if not np.isnan(t) else ANALYSIS_WINDOW)
            all_x.append(i*2 + 1)
            colors.append(COLOR_L2S)
            
        ax5.scatter(all_x, all_y, c=colors, s=100, edgecolor='k')
        ax5.plot(all_x, all_y, 'k--', alpha=0.2)
        ax5.set_title(f'Within-Session Dynamics (Session {s_idx})', fontweight='bold')
        ax5.set_ylabel('Learned Trial', fontweight='bold')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)

    # 6. Meta Learning (Bottom Right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(pd.Series(session_summaries['s2l']).rolling(3).mean(), color=COLOR_S2L, lw=2, label='S→L')
    ax6.plot(pd.Series(session_summaries['l2s']).rolling(3).mean(), color=COLOR_L2S, lw=2, label='L→S')
    ax6.set_title('Meta-Learning (Rolling Avg)', fontweight='bold')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    # 7. Early vs Late (Logic Updated for Fitting)
    ax7 = fig.add_subplot(gs[3, 0])
    
    # Early: Mean of first 3 raw trials (most honest metric of starting point)
    early_s2l = [np.mean(t[:3]) for t in all_transitions['s2l']['trial_data']]
    # Late: The Fitted Asymptote (A)
    late_s2l = all_transitions['s2l']['asymptotes']
    
    early_l2s = [np.mean(t[:3]) for t in all_transitions['l2s']['trial_data']]
    late_l2s = all_transitions['l2s']['asymptotes']
    
    def jitter(arr): return np.array(arr) + np.random.uniform(-0.02, 0.02, len(arr))
    
    ax7.scatter(jitter(early_s2l), late_s2l, color=COLOR_S2L, alpha=0.3, label='S→L')
    ax7.scatter(jitter(early_l2s), late_l2s, color=COLOR_L2S, alpha=0.3, label='L→S')
    ax7.plot([0,1],[0,1], 'k--', alpha=0.3)
    ax7.set_xlabel('Initial Perf (First 3 Trials)', fontweight='bold')
    ax7.set_ylabel('Fitted Asymptote (A)', fontweight='bold')
    ax7.set_title('Learning Magnitude', fontweight='bold')
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)

    # 8. Speed vs Accuracy
    ax8 = fig.add_subplot(gs[3, 1])
    # Filter for valid fits only
    mask_s = ~np.isnan(all_transitions['s2l']['times'])
    mask_l = ~np.isnan(all_transitions['l2s']['times'])
    
    ax8.scatter(all_transitions['s2l']['times'][mask_s], 
                np.array(all_transitions['s2l']['asymptotes'])[mask_s], 
                color=COLOR_S2L, alpha=0.4)
    ax8.scatter(all_transitions['l2s']['times'][mask_l], 
                np.array(all_transitions['l2s']['asymptotes'])[mask_l], 
                color=COLOR_L2S, alpha=0.4)
    ax8.set_xlabel('Time to Criterion (t)', fontweight='bold')
    ax8.set_ylabel('Fitted Asymptote (A)', fontweight='bold')
    ax8.set_title('Speed-Accuracy Tradeoff', fontweight='bold')
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)

    # 9. CDF / Survival Curve
    ax9 = fig.add_subplot(gs[3, 2])
    sorted_s = np.sort(s2l_clean)
    sorted_l = np.sort(l2s_clean)
    ax9.step(sorted_s, np.arange(1, len(sorted_s)+1)/len(all_transitions['s2l']['times']), 
             color=COLOR_S2L, lw=2, label='S→L')
    ax9.step(sorted_l, np.arange(1, len(sorted_l)+1)/len(all_transitions['l2s']['times']), 
             color=COLOR_L2S, lw=2, label='L→S')
    ax9.set_title('Cumulative Learning Prob', fontweight='bold')
    ax9.set_xlabel('Trials', fontweight='bold')
    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)
    
    plt.suptitle(f"Fitted Learning Dynamics: {subject}", fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        save_path = os.path.normpath(os.path.expanduser(str(save_path)))
        output_filename = f'expo_fitted_trial_dynamic_{subject}_{data_paths[-1].split("_")[-2]}_{data_paths[0].split("_")[-2]}.pdf'
        output_path = os.path.join(save_path, output_filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
    plt.close()

    return all_transitions, session_summaries