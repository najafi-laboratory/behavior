import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.backends.backend_pdf import PdfPages
import os

def analyze_learning_dynamics(sessions_data, subject, data_paths, save_path=None):
    """
    Analyzes learning dynamics using Sequential Criterion method.
    Updated to return median of criterion window and include density heatmaps.
    """
    
    # --- Configuration ---
    ANALYSIS_WINDOW = 20
    CRITERION_WINDOW = 7
    CRITERION_THRESHOLD = 6
    ROLLING_WINDOW_SIZE = 3
    
    # --- Helper Functions ---
    
    def sequential_criterion(trial_outcomes, window=CRITERION_WINDOW, 
                             threshold=CRITERION_THRESHOLD):
        """Detects learning using sequential criterion method."""
        if len(trial_outcomes) < window:
            return np.nan
        
        for i in range(len(trial_outcomes) - window + 1):
            window_trials = trial_outcomes[i:i + window]
            if np.sum(window_trials) >= threshold:
                # MODIFICATION: Return the median index of the window
                return i + (window // 2)
        
        return np.nan
    
    def compute_asymptotic_performance(trial_outcomes, last_n=5):
        """Compute average performance in last N trials"""
        if len(trial_outcomes) < last_n:
            return np.nanmean(trial_outcomes)
        return np.nanmean(trial_outcomes[-last_n:])
    
    def nansem(data):
        """SEM that handles NaN values"""
        clean = data[~np.isnan(data)]
        return sem(clean) if len(clean) > 0 else np.nan

    def add_jitter(arr, amount=0.03):
        return np.array(arr) + np.random.uniform(-amount, amount, len(arr))
    
    # --- Main Processing ---
    
    dates = sessions_data['dates']
    outcomes_list = sessions_data['outcomes']
    block_types_list = sessions_data['block_type']
    
    # Storage for all transitions
    all_transitions = {
        's2l': {
            'times': [], 'sessions': [], 'trial_data': [], 
            'asymptotic_perf': [], 'first_trial_perf': [], 'start_indices': []
        },
        'l2s': {
            'times': [], 'sessions': [], 'trial_data': [], 
            'asymptotic_perf': [], 'first_trial_perf': [], 'start_indices': []
        }
    }
    
    session_summaries = {
        's2l': [], 'l2s': [], 'dates': []
    }
    
    # Process each session
    for session_idx, (date, outcomes, block_types) in enumerate(
        zip(dates, outcomes_list, block_types_list)):
        
        session_s2l_times = []
        session_l2s_times = []
        
        for trial_idx in range(1, len(block_types)):
            if trial_idx + ANALYSIS_WINDOW >= len(outcomes):
                continue
            
            prev_block = block_types[trial_idx - 1]
            curr_block = block_types[trial_idx]
            
            if prev_block == curr_block:
                continue
            
            trial_outcomes = []
            for k in range(ANALYSIS_WINDOW):
                idx = trial_idx + k
                trial_outcomes.append(1.0 if outcomes[idx] == 'Reward' else 0.0)
            
            trial_outcomes = np.array(trial_outcomes)
            learned_trial = sequential_criterion(trial_outcomes)
            asymptotic_perf = compute_asymptotic_performance(trial_outcomes)
            first_trial_perf = trial_outcomes[0]
            
            if prev_block == 1 and curr_block == 2:  # S->L
                all_transitions['s2l']['times'].append(learned_trial)
                all_transitions['s2l']['sessions'].append(session_idx)
                all_transitions['s2l']['trial_data'].append(trial_outcomes)
                all_transitions['s2l']['asymptotic_perf'].append(asymptotic_perf)
                all_transitions['s2l']['first_trial_perf'].append(first_trial_perf)
                all_transitions['s2l']['start_indices'].append(trial_idx)
                session_s2l_times.append(learned_trial)
                
            elif prev_block == 2 and curr_block == 1:  # L->S
                all_transitions['l2s']['times'].append(learned_trial)
                all_transitions['l2s']['sessions'].append(session_idx)
                all_transitions['l2s']['trial_data'].append(trial_outcomes)
                all_transitions['l2s']['asymptotic_perf'].append(asymptotic_perf)
                all_transitions['l2s']['first_trial_perf'].append(first_trial_perf)
                all_transitions['l2s']['start_indices'].append(trial_idx)
                session_l2s_times.append(learned_trial)
        
        session_summaries['s2l'].append(
            np.nanmean(session_s2l_times) if session_s2l_times else np.nan
        )
        session_summaries['l2s'].append(
            np.nanmean(session_l2s_times) if session_l2s_times else np.nan
        )
        session_summaries['dates'].append(date)
    
    # Convert to arrays
    for key in ['s2l', 'l2s']:
        for metric in all_transitions[key].keys():
            all_transitions[key][metric] = np.array(all_transitions[key][metric])
        session_summaries[key] = np.array(session_summaries[key])
    
    # --- Statistical Analysis ---
    s2l_times = all_transitions['s2l']['times']
    l2s_times = all_transitions['l2s']['times']
    s2l_clean = s2l_times[~np.isnan(s2l_times)]
    l2s_clean = l2s_times[~np.isnan(l2s_times)]

    # --- Color Scheme ---
    COLOR_S2L = '#2E86AB' # Blueish
    COLOR_L2S = '#A23B72' # Reddish
    COLOR_MEAN = '#F18F01'

    
    
    # --- CREATE MULTI-PAGE PDF ---
    if save_path:
        # 1. Clean up the path
        save_path = os.path.normpath(os.path.expanduser(str(save_path)))
        
        # 2. Construct the filename
        # Ensure data_paths exists or handle the naming safely
        start_date = data_paths[0].split('_')[-2] if data_paths else "Start"
        end_date = data_paths[-1].split('_')[-2] if data_paths else "End"
        output_filename = f'Criterion_trial_dynamic_{subject}_{start_date}_{end_date}.pdf'
        output_path = os.path.join(save_path, output_filename)
        
        print(f"Generating PDF at: {output_path}")
        with PdfPages(output_path) as pdf:
            # ===== PAGE 1: Overview =====
            # Increased height to accommodate the heatmaps at the bottom
            fig = plt.figure(figsize=(18, 16)) 
            
            # MODIFICATION: 5 Rows. The last row (Heatmaps) is shorter (0.6 ratio)
            gs = GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.35, 
                         height_ratios=[1, 1, 1, 1, 0.6])
            
            # 1. Summary Comparison
            ax1 = fig.add_subplot(gs[0, 0])
            x_s2l = np.random.normal(1, 0.04, size=len(s2l_clean))
            x_l2s = np.random.normal(2, 0.04, size=len(l2s_clean))
            
            ax1.scatter(x_s2l, s2l_clean, alpha=0.4, s=40, color=COLOR_S2L, zorder=2)
            ax1.scatter(x_l2s, l2s_clean, alpha=0.4, s=40, color=COLOR_L2S, zorder=2)
            ax1.errorbar(1, np.nanmean(s2l_clean), yerr=nansem(s2l_clean), 
                         fmt='o', color=COLOR_MEAN, capsize=8, markersize=14, 
                         markeredgewidth=2, markeredgecolor='black', label='Mean ± SEM', zorder=3)
            ax1.errorbar(2, np.nanmean(l2s_clean), yerr=nansem(l2s_clean),
                         fmt='o', color=COLOR_MEAN, capsize=8, markersize=14,
                         markeredgewidth=2, markeredgecolor='black', zorder=3)
            
            ax1.set_xticks([1, 2])
            ax1.set_xticklabels(['S→L', 'L→S'], fontsize=11, fontweight='bold')
            ax1.set_ylabel('Trials to Criterion (Median)', fontsize=12, fontweight='bold')
            ax1.set_title('Adaptation Speed Comparison', fontsize=13, fontweight='bold', pad=10)
            ax1.set_ylim(0, ANALYSIS_WINDOW + 2)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            # 2. Distribution Histograms
            ax2 = fig.add_subplot(gs[0, 1])
            bins = np.linspace(0, ANALYSIS_WINDOW, 12)
            ax2.hist(s2l_clean, bins=bins, alpha=0.5, color=COLOR_S2L, 
                    label=f'S→L', density=True, edgecolor='black')
            ax2.hist(l2s_clean, bins=bins, alpha=0.5, color=COLOR_L2S, 
                    label=f'L→S', density=True, edgecolor='black')
            ax2.axvline(np.nanmean(s2l_clean), color=COLOR_S2L, linestyle='--', linewidth=2.5)
            ax2.axvline(np.nanmean(l2s_clean), color=COLOR_L2S, linestyle='--', linewidth=2.5)
            ax2.set_xlabel('Trials to Criterion', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax2.set_title('Learning Time Distributions', fontsize=13, fontweight='bold', pad=10)
            ax2.legend(fontsize=10)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # 3. Success Rate
            ax3 = fig.add_subplot(gs[0, 2])
            success_s2l = len(s2l_clean) / len(s2l_times) * 100
            success_l2s = len(l2s_clean) / len(l2s_times) * 100
            ax3.bar([1, 2], [success_s2l, success_l2s], width=0.6, 
                   color=[COLOR_S2L, COLOR_L2S], alpha=0.7, edgecolor='black')
            ax3.set_xticks([1, 2])
            ax3.set_xticklabels(['S→L', 'L→S'], fontsize=11, fontweight='bold')
            ax3.set_ylabel('% Transitions Learned', fontsize=12, fontweight='bold')
            ax3.set_title('Learning Success Rate', fontsize=13, fontweight='bold', pad=10)
            ax3.set_ylim(0, 105)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            
            # 4. Transition-Triggered Average
            ax4 = fig.add_subplot(gs[1, :])
            max_len = ANALYSIS_WINDOW
            s2l_aligned = np.full((len(all_transitions['s2l']['trial_data']), max_len), np.nan)
            l2s_aligned = np.full((len(all_transitions['l2s']['trial_data']), max_len), np.nan)
            
            for i, data in enumerate(all_transitions['s2l']['trial_data']):
                s2l_aligned[i, :len(data)] = data
            for i, data in enumerate(all_transitions['l2s']['trial_data']):
                l2s_aligned[i, :len(data)] = data
            
            s2l_mean = np.nanmean(s2l_aligned, axis=0)
            s2l_sem = np.array([nansem(s2l_aligned[:, i]) for i in range(max_len)])
            l2s_mean = np.nanmean(l2s_aligned, axis=0)
            l2s_sem = np.array([nansem(l2s_aligned[:, i]) for i in range(max_len)])
            
            trials = np.arange(max_len)
            ax4.plot(trials, s2l_mean, '-o', color=COLOR_S2L, linewidth=3, markersize=6, label='S→L', alpha=0.8)
            ax4.fill_between(trials, s2l_mean - s2l_sem, s2l_mean + s2l_sem, color=COLOR_S2L, alpha=0.2)
            ax4.plot(trials, l2s_mean, '-s', color=COLOR_L2S, linewidth=3, markersize=6, label='L→S', alpha=0.8)
            ax4.fill_between(trials, l2s_mean - l2s_sem, l2s_mean + l2s_sem, color=COLOR_L2S, alpha=0.2)
            ax4.set_xlabel('Trials Since Transition', fontsize=13, fontweight='bold')
            ax4.set_ylabel('Proportion Correct', fontsize=13, fontweight='bold')
            ax4.set_title('Transition-Triggered Average Learning Curves', fontsize=14, fontweight='bold')
            ax4.legend(loc='lower right')
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            
            # 6. Meta-Learning
            ax6 = fig.add_subplot(gs[2, :])
            session_indices = np.arange(len(session_summaries['dates']))
            df_s2l = pd.Series(session_summaries['s2l']).rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
            df_l2s = pd.Series(session_summaries['l2s']).rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
            ax6.plot(session_indices, df_s2l, '-', color=COLOR_S2L, linewidth=3, alpha=0.8, label=f'S→L (rolling)')
            ax6.plot(session_indices, df_l2s, '-', color=COLOR_L2S, linewidth=3, alpha=0.8, label=f'L→S (rolling)')
            ax6.scatter(session_indices, session_summaries['s2l'], alpha=0.3, color=COLOR_S2L)
            ax6.scatter(session_indices, session_summaries['l2s'], alpha=0.3, color=COLOR_L2S)
            ax6.set_xlabel('Session Number', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Avg Trials to Criterion', fontsize=12, fontweight='bold')
            ax6.set_title('Meta-Learning Across Sessions', fontsize=13, fontweight='bold')
            ax6.spines['top'].set_visible(False)
            ax6.spines['right'].set_visible(False)

            # --- ROW 4: SCATTER PLOTS ---
            
            # 7. Learning Magnitude (Scatter)
            ax7 = fig.add_subplot(gs[3, 0])
            early_s2l = [np.mean(data[:3]) for data in all_transitions['s2l']['trial_data']]
            asymp_s2l = all_transitions['s2l']['asymptotic_perf']
            early_l2s = [np.mean(data[:3]) for data in all_transitions['l2s']['trial_data']]
            asymp_l2s = all_transitions['l2s']['asymptotic_perf']

            ax7.scatter(add_jitter(early_s2l), add_jitter(asymp_s2l), s=40, alpha=0.4, color=COLOR_S2L, label='S→L', edgecolor='none')
            ax7.scatter(add_jitter(early_l2s), add_jitter(asymp_l2s), s=40, alpha=0.4, color=COLOR_L2S, label='L→S', edgecolor='none')
            ax7.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax7.set_xlabel('Initial Perf (First 3)', fontsize=11, fontweight='bold')
            ax7.set_ylabel('Asymptotic Perf', fontsize=11, fontweight='bold')
            ax7.set_title('Learning Magnitude', fontsize=12, fontweight='bold')
            ax7.set_xlim(-0.1, 1.1)
            ax7.set_ylim(-0.1, 1.1)
            ax7.spines['top'].set_visible(False)
            ax7.spines['right'].set_visible(False)

            # 8. Speed-Accuracy (Scatter)
            ax8 = fig.add_subplot(gs[3, 1])
            ax8.scatter(s2l_times, asymp_s2l, s=50, alpha=0.5, color=COLOR_S2L, label='S→L')
            ax8.scatter(l2s_times, asymp_l2s, s=50, alpha=0.5, color=COLOR_L2S, label='L→S')
            
            if len(s2l_clean) > 1:
                m, b = np.polyfit(s2l_clean, np.array(asymp_s2l)[~np.isnan(s2l_times)], 1)
                ax8.plot(s2l_clean, m*s2l_clean + b, color=COLOR_S2L, alpha=0.4)
            if len(l2s_clean) > 1:
                m, b = np.polyfit(l2s_clean, np.array(asymp_l2s)[~np.isnan(l2s_times)], 1)
                ax8.plot(l2s_clean, m*l2s_clean + b, color=COLOR_L2S, alpha=0.4)

            ax8.set_xlabel('Trials to Criterion', fontsize=11, fontweight='bold')
            ax8.set_ylabel('Asymptotic Perf', fontsize=11, fontweight='bold')
            ax8.set_title('Speed-Accuracy', fontsize=12, fontweight='bold')
            ax8.spines['top'].set_visible(False)
            ax8.spines['right'].set_visible(False)

            # 9. Cumulative Probability (Spanning rows 3 and 4 visually or just row 3)
            # Keeping it in row 3 to leave room for something else or just standard
            ax9 = fig.add_subplot(gs[3, 2])
            s2l_sorted = np.sort(s2l_clean)
            l2s_sorted = np.sort(l2s_clean)
            y_s2l = np.arange(1, len(s2l_sorted) + 1) / len(s2l_times)
            y_l2s = np.arange(1, len(l2s_sorted) + 1) / len(l2s_times)
            ax9.step(s2l_sorted, y_s2l, where='post', color=COLOR_S2L, linewidth=3, label='S→L')
            ax9.step(l2s_sorted, y_l2s, where='post', color=COLOR_L2S, linewidth=3, label='L→S')
            ax9.set_xlabel('Trials to Criterion', fontsize=11, fontweight='bold')
            ax9.set_ylabel('Cumulative % Learned', fontsize=11, fontweight='bold')
            ax9.set_title('Learning Probability', fontsize=12, fontweight='bold')
            ax9.legend(loc='lower right', fontsize=8)
            ax9.spines['top'].set_visible(False)
            ax9.spines['right'].set_visible(False)

            # --- ROW 5: HEATMAPS ---
            
            # Helper for Density Plots
            def plot_density_subplots(outer_grid_loc, x1, y1, x2, y2, x_label, y_label):
                """Creates 2 subplots inside a GridSpec cell for S->L and L->S densities"""
                inner_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid_loc, wspace=0.1)
                
                # Left: S->L (Blue)
                ax_l = plt.subplot(inner_gs[0])
                mask1 = ~np.isnan(x1) & ~np.isnan(y1)
                if np.sum(mask1) > 0:
                    h1 = ax_l.hist2d(np.array(x1)[mask1], np.array(y1)[mask1], 
                                   bins=10, cmap='Blues')
                ax_l.set_title('S→L Density', fontsize=9, color=COLOR_S2L, fontweight='bold')
                ax_l.set_xlabel(x_label, fontsize=9)
                ax_l.set_ylabel(y_label, fontsize=9)
                ax_l.spines['top'].set_visible(False)
                ax_l.spines['right'].set_visible(False)

                # Right: L->S (Red/Purple)
                ax_r = plt.subplot(inner_gs[1])
                mask2 = ~np.isnan(x2) & ~np.isnan(y2)
                if np.sum(mask2) > 0:
                    h2 = ax_r.hist2d(np.array(x2)[mask2], np.array(y2)[mask2], 
                                   bins=10, cmap='Reds') # or 'PuRd'
                ax_r.set_title('L→S Density', fontsize=9, color=COLOR_L2S, fontweight='bold')
                ax_r.set_xlabel(x_label, fontsize=9)
                ax_r.set_yticks([]) # Hide Y ticks on second plot
                ax_r.spines['top'].set_visible(False)
                ax_r.spines['right'].set_visible(False)
                ax_r.spines['left'].set_visible(False)

            # Heatmaps for Learning Magnitude (Column 0)
            plot_density_subplots(gs[4, 0], early_s2l, asymp_s2l, early_l2s, asymp_l2s, 
                                  'Init Perf', 'Asymp Perf')

            # Heatmaps for Speed-Accuracy (Column 1)
            plot_density_subplots(gs[4, 1], s2l_times, asymp_s2l, l2s_times, asymp_l2s, 
                                  'Trials to Crit', 'Asymp Perf')

            # Title
            plt.suptitle(f'Learning Dynamics: {subject}\nCriterion: {CRITERION_THRESHOLD}/{CRITERION_WINDOW} Correct (Median)', 
                         fontsize=16, fontweight='bold', y=0.98)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # ===== PAGES 2+: Within-Session Dynamics =====
            # (Keeping existing logic for per-session plots)
            from matplotlib.lines import Line2D
            for session_idx in range(len(dates)):
                s2l_mask = np.array(all_transitions['s2l']['sessions']) == session_idx
                l2s_mask = np.array(all_transitions['l2s']['sessions']) == session_idx
                
                if not (np.any(s2l_mask) or np.any(l2s_mask)):
                    continue 
                
                fig_session = plt.figure(figsize=(14, 8))
                ax = fig_session.add_subplot(111)
                
                s2l_t = np.array(all_transitions['s2l']['times'])[s2l_mask]
                s2l_idx = np.array(all_transitions['s2l']['start_indices'])[s2l_mask]
                l2s_t = np.array(all_transitions['l2s']['times'])[l2s_mask]
                l2s_idx = np.array(all_transitions['l2s']['start_indices'])[l2s_mask]
                
                plot_data = []
                for t, idx in zip(s2l_t, s2l_idx):
                    plot_data.append({'time': t, 'idx': idx, 'type': 'S→L', 'color': COLOR_S2L, 'marker': 'o'})
                for t, idx in zip(l2s_t, l2s_idx):
                    plot_data.append({'time': t, 'idx': idx, 'type': 'L→S', 'color': COLOR_L2S, 'marker': 's'})
                
                plot_data.sort(key=lambda x: x['idx'])
                
                xs = range(len(plot_data))
                ys = []
                for x, item in zip(xs, plot_data):
                    time = item['time']
                    ys.append(time if not np.isnan(time) else ANALYSIS_WINDOW)
                    
                    if np.isnan(time):
                        ax.scatter(x, ANALYSIS_WINDOW, marker='x', s=200, color='red', linewidths=3, zorder=3)
                        ax.plot([x, x], [0, ANALYSIS_WINDOW], color=item['color'], linestyle=':', alpha=0.3)
                    else:
                        ax.scatter(x, time, marker=item['marker'], s=150, color=item['color'], edgecolor='k', alpha=0.8, zorder=3)
                        ax.plot([x, x], [0, time], color=item['color'], alpha=0.3, linewidth=3)

                ax.plot(xs, ys, color='gray', alpha=0.3, linewidth=1.5, linestyle='--', zorder=1)
                ax.set_xlabel('Transition Number', fontsize=14, fontweight='bold')
                ax.set_ylabel('Trials to Criterion', fontsize=14, fontweight='bold')
                ax.set_title(f'Within-Session Dynamics | {dates[session_idx]}', fontsize=15, fontweight='bold')
                ax.set_ylim(-1, ANALYSIS_WINDOW + 2)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # --- SAVE THIS SESSION'S FIGURE TO PDF ---
                pdf.savefig(fig_session, bbox_inches='tight')
                plt.close(fig_session)
                
    if save_path:
        print(f"\nAnalysis saved to: {save_path}")

    return all_transitions, session_summaries