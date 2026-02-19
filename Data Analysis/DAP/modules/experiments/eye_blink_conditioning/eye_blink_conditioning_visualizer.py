import logging
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
import sklearn
from datetime import datetime

class EyeBlinkConditioningVisualizer:
    """
    Experiment-specific visualizer for eye blink conditioning data.
    Uses GeneralVisualizer infrastructure for common functionality.
    """
    
    def __init__(self, config_manager, subject_list, analysis_results, infrastructure, logger):
        """Initialize with ConfigManager, subject list, analysis results, and infrastructure."""
        self.config_manager = config_manager
        self.subject_list = subject_list
        self.analysis_results = analysis_results
        self.infrastructure = infrastructure  # GeneralVisualizer instance
        self.logger = logger

        self.logger.info("VZ-EBC: Initializing EyeBlinkConditioningVisualizer...")
        
        # Get experiment-specific visualization config
        experiment_config = config_manager.config.get('experiment_configs', {}).get(config_manager.experiment_name, {})
        self.visualization_config = experiment_config.get('visualization', {})
        
        self.logger.info("VZ-EBC: EyeBlinkConditioningVisualizer initialized successfully")
    
    def generate_visualizations(self) -> Dict[str, Any]:
        """Generate SID-specific visualizations."""
        self.logger.info("VZ-EBC: Starting eye blink conditioning visualization...")
        
        visualization_results = {
            'visualization_type': 'eye_blink_conditioning',
            'experiment_config': self.config_manager.experiment_name,
            'subjects_visualized': len(self.subject_list)
        }
        
        # Generate visualizations for each subject
        for subject_id in self.subject_list:
            if subject_id in self.analysis_results.get('subject_results', {}):
                subject_analysis = self.analysis_results['subject_results'][subject_id]
                self._visualize_subject(subject_id, subject_analysis)
        
        # Generate group-level visualizations
        self._visualize_group()
        
        self.logger.info(f"VZ-EBC: eye blink conditioning visualization completed")
        return visualization_results
    
    def _visualize_subject(self, subject_id: str, subject_analysis: Dict[str, Any]):
        """Generate SID-specific plots for a single subject."""
        self.logger.info(f"VZ-EBC: Generating plots for subject {subject_id}")
        
        # Plot 1: Performance over sessions
        # self._plot_performance_over_sessions(subject_id, subject_analysis, do_plot=False, show_plot=False)
        
        # Plot 2: Session overview trial type plots for each session
        if 'session_data' in subject_analysis:
            session_summaries = subject_analysis.get('session_summaries', {})
            
        #     for session_name, session_data in subject_analysis['session_data'].items():
        #         self._plot_session_overview_trial_type(subject_id, session_name, session_data, do_plot=True, show_plot=False)
        #         self._plot_session_overview_rt(subject_id, session_name, session_data, do_plot=True, show_plot=False)
                
        #         # Get matching session summary for analyses
        #         session_summary = session_summaries.get(session_name, {})
                
        #         # Plot 3: ISI PDF for each session
        #         self._plot_isi_pdf(subject_id, session_name, session_data, session_summary, do_plot=True, show_plot=False)
                
        #         # Plot 4: Psychometric curves for each session
        #         self._plot_psychometric(subject_id, session_name, session_data, session_summary, do_plot=True, show_plot=False)
                
        #         # Plot 5: Response time histogram for each session
        #         self._plot_rt_histogram(subject_id, session_name, session_data, do_plot=True, show_plot=False)
                
        #         # Plot 6: Response time statistics for each session
        #         self._plot_rt_stats(subject_id, session_name, session_data, session_summary, do_plot=True, show_plot=False)
        
        # # Plot 7: Trial outcomes by session (across all sessions)
        # self._plot_trial_outcomes_by_session(subject_id, subject_analysis, do_plot=True, show_plot=False)
        
        # # Plot 8: Rolling performance across sessions
        # self._plot_rolling_performance_across_sessions(subject_id, subject_analysis, do_plot=True, show_plot=False)

    def _plot_performance_over_sessions(self, subject_id: str, subject_analysis: Dict[str, Any], do_plot=False, show_plot=False):
        """Plot accuracy over sessions for a subject."""
        # Check if plotting is enabled
        if not do_plot:
            return
            
            
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract session performance data
            sessions = list(subject_analysis['session_summaries'].keys())
            accuracies = [
                subject_analysis['session_summaries'][session]['performance'].get('accuracy', 0)
                for session in sessions
            ]
            
            # Plot performance over sessions
            ax.plot(range(len(sessions)), accuracies, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Session')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Performance Over Sessions - {subject_id}')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            
            # Use infrastructure to save and show
            filename = f'{subject_id}_performance_over_sessions.png'
            self.infrastructure.save_figure(fig, filename, subject_id=subject_id, 
                                          description='Accuracy across training sessions')
            self.infrastructure.show_figure(fig, show_plot=show_plot)
            plt.close()
            
            self.logger.info(f"VZ-EBC: Generated performance plot for {subject_id}")
            
        except Exception as e:
            self.logger.error(f"VZ-EBC: Failed to generate performance plot for {subject_id}: {e}")
    
    def _plot_session_overview_trial_type(self, subject_id: str, session_name: str, session_data: Dict[str, Any], do_plot=True, show_plot=None):
        """Plot session overview showing trial types across trials."""
        # Check if plotting is enabled
        if not do_plot:
            return
                    
        try:
            if 'df_trials' not in session_data:
                self.logger.warning(f"VZ-EBC: No trial data found for {subject_id} session {session_name}")
                return
            
            df_trials = session_data['df_trials']
            if df_trials.empty:
                self.logger.warning(f"VZ-EBC: Empty trial data for {subject_id} session {session_name}")
                return
            
            fig, ax = plt.subplots(figsize=(24, 3.5))
            
            # Trial type Y values - left on top, right on bottom
            y_map = {0: 1, 1: 0}  # 0=left, 1=right
            trial_spacing = 2
            
            # Plot each trial
            for idx, row in df_trials.iterrows():
                outcome = row.get('outcome', 'Unknown')
                if outcome == 'DidNotChoose':
                    marker = 'o'
                    facecolor = 'none'
                    edgecolor = 'black'
                else:
                    facecolor = 'green' if row.get('mouse_correct', 0) == 1 else 'red'
                    marker = 'x' if row.get('mouse_correct', np.nan) == 0 else 'o'  # x for incorrect, o for correct
                    # Only set edgecolor for filled markers ('o'), not for 'x' markers
                    edgecolor = 'black' if marker == 'o' else None
                
                y = y_map.get(row.get('is_right', 0), 0)
                x = (idx + 1) * trial_spacing
                
                # Plot with conditional edgecolor
                if edgecolor:
                    ax.scatter(x, y, facecolor=facecolor, marker=marker, s=20, 
                              alpha=0.85, edgecolors=edgecolor, zorder=3)
                else:
                    ax.scatter(x, y, color=facecolor, marker=marker, s=20, 
                              alpha=0.85, zorder=3)
                
                # Optional opto trials
                if row.get('is_opto', 0) == 1:
                    ax.scatter(x, y, marker='^', color='purple', s=100, alpha=0.6, zorder=2)
                
                # Optional move correct spout
                if row.get('MoveCorrectSpout', 0) == 1:
                    ax.scatter(x, y, marker='+', color='purple', s=160, alpha=0.7, zorder=2)
            
            # Shade naive trial range
            naive_trials = df_trials[df_trials.get('naive', 0) == 1]
            if not naive_trials.empty:
                naive_x = (naive_trials.index + 1) * trial_spacing
                ax.axvspan(naive_x.min() - trial_spacing/2, 
                          naive_x.max() + trial_spacing/2, 
                          color='lightblue', alpha=0.3, zorder=1)
            
            # Axis settings
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Right', 'Left'])
            ax.set_ylim(-0.5, 1.5)
            x_max = len(df_trials) * trial_spacing
            ax.set_xlim(0, x_max + trial_spacing*2)
            
            # X-axis ticks every 5 trials
            tick_spacing = 5
            xticks_scaled = np.arange(0, x_max + 1, trial_spacing * tick_spacing)
            ax.set_xticks(xticks_scaled)
            ax.set_xticklabels([int(x / trial_spacing) for x in xticks_scaled])
            
            ax.set_xlabel("Trial Index")
            ax.set_ylabel("Trial Type")
            
            # Create legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', linestyle='None', color='green', label='Rewarded', markersize=7),
                Line2D([0], [0], marker='x', linestyle='None', color='red', label='Punished', markersize=7),
                Line2D([0], [0], marker='o', linestyle='None', color='black', markerfacecolor='none', label='Did Not Choose', markersize=7),               
                Line2D([0], [0], marker='s', linestyle='None', color='lightblue', alpha=0.4, label='Naive Block', markersize=10)

            ]
            
            if 'is_opto' in df_trials.columns and df_trials['is_opto'].any():
                legend_elements.append(
                    Line2D([0], [0], marker='^', linestyle='None',
                           color='purple', label='Opto Trial', markersize=7)
                )
            
            if 'MoveCorrectSpout' in df_trials.columns and df_trials['MoveCorrectSpout'].any():
                legend_elements.append(
                    Line2D([0], [0], marker='+', linestyle='None',
                           color='purple', label='MoveCorrectSpout', markersize=10)
                )
            
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.005, 1.0),
                     borderaxespad=0, frameon=False, fontsize=9)
            
            # Title
            subject_name = df_trials.get('subject_name', [subject_id]).iloc[0] if 'subject_name' in df_trials.columns else subject_id
            raw_date = session_data.get('session_info', {}).get('date', 'Unknown')
            
            # Convert date to '17-Apr-2025' format
            if raw_date != 'Unknown':
                try:
                    # Assuming raw_date is in format like '20250417' or similar                    
                    if len(raw_date) == 8 and raw_date.isdigit():
                        dt = datetime.strptime(raw_date, '%Y%m%d')
                        session_date = dt.strftime('%d-%b-%Y')
                    else:
                        session_date = raw_date
                except:
                    session_date = raw_date
            else:
                session_date = 'Unknown'
            
            title = f"{subject_name}  {session_date}  Session Overview - Trial Types"
            ax.set_title(title)
            
            plt.tight_layout(pad=1.0)
            
            # Use infrastructure to save and show with subfolder
            filename = f'{subject_id}_session_overview_trial_type_{raw_date}.png'
            self.infrastructure.save_figure(fig, filename, subject_id=subject_id, 
                                          subfolder='session_overviews',
                                          description=f'Trial type overview for session {session_name}')
            self.infrastructure.show_figure(fig, show_plot=show_plot)
            plt.close()
            
            self.logger.info(f"VZ-EBC: Generated session overview trial type plot for {subject_id} session {session_name}")
            
        except Exception as e:
            self.logger.error(f"VZ-EBC: Failed to generate session overview trial type plot for {subject_id} session {session_name}: {e}")

    def _plot_session_overview_rt(self, subject_id: str, session_name: str, session_data: Dict[str, Any], do_plot=True, show_plot=None):
        """Plot session overview showing reaction times across trials."""
        # Check if plotting is enabled
        if not do_plot:
            return
            
            
        try:
            if 'df_trials' not in session_data:
                self.logger.warning(f"VZ-EBC: No trial data found for {subject_id} session {session_name}")
                return
            
            df_trials = session_data['df_trials']
            if df_trials.empty:
                self.logger.warning(f"VZ-EBC: Empty trial data for {subject_id} session {session_name}")
                return
            
            # Check if reaction time data exists
            if 'RT' not in df_trials.columns:
                self.logger.warning(f"VZ-EBC: No reaction time data found for {subject_id} session {session_name}")
                return
            
            fig, ax = plt.subplots(figsize=(24, 3.5))
            
            trial_spacing = 2
            
            # Plot each trial's reaction time
            for idx, row in df_trials.iterrows():
                outcome = row.get('outcome', 'Unknown')
                if outcome == 'DidNotChoose':
                    marker = 'o'
                    facecolor = 'none'
                    edgecolor = 'black'
                else:
                    facecolor = 'green' if row.get('mouse_correct', np.nan) == 1 else 'red'
                    marker = 'x' if row.get('mouse_correct', np.nan) == 0 else 'o'  # x for incorrect, o for correct
                    # Only set edgecolor for filled markers ('o'), not for 'x' markers
                    edgecolor = 'black' if marker == 'o' else None
                
                x = (idx + 1) * trial_spacing
                y = row.get('RT', 0)
                
                # Plot with conditional edgecolor
                if edgecolor:
                    ax.scatter(x, y, facecolor=facecolor, marker=marker, s=20, 
                              alpha=0.85, edgecolors=edgecolor, zorder=3)
                else:
                    ax.scatter(x, y, color=facecolor, marker=marker, s=20, 
                              alpha=0.85, zorder=3)
                
                # Optional opto trials
                if row.get('is_opto', 0) == 1:
                    ax.scatter(x, y, marker='^', color='purple', s=100, alpha=0.6, zorder=2)
                
                # Optional move correct spout
                if row.get('MoveCorrectSpout', 0) == 1:
                    ax.scatter(x, y, marker='+', color='purple', s=160, alpha=0.7, zorder=2)
            
            # Shade naive trial range
            naive_trials = df_trials[df_trials.get('naive', 0) == 1]
            if not naive_trials.empty:
                naive_x = (naive_trials.index + 1) * trial_spacing
                ax.axvspan(naive_x.min() - trial_spacing/2, 
                          naive_x.max() + trial_spacing/2, 
                          color='lightblue', alpha=0.3, zorder=1)
            
            # Shade DidNotChoose trials
            dnc_trials = df_trials[df_trials.get('outcome', '') == 'DidNotChoose']
            if not dnc_trials.empty:
                dnc_x = (dnc_trials.index + 1) * trial_spacing
                for x in dnc_x:
                    ax.axvspan(x - trial_spacing/2, x + trial_spacing/2, 
                              color='lightgray', alpha=0.3, zorder=0)
            
            # Axis settings
            x_max = len(df_trials) * trial_spacing
            ax.set_xlim(0, x_max + trial_spacing*2)
            ax.set_ylim(0, 3000)  # Limit y-axis to 0-3 seconds for RT
            
            # X-axis ticks every 5 trials
            tick_spacing = 5
            xticks_scaled = np.arange(0, x_max + 1, trial_spacing * tick_spacing)
            ax.set_xticks(xticks_scaled)
            ax.set_xticklabels([int(x / trial_spacing) for x in xticks_scaled])
            
            ax.set_xlabel("Trial Index")
            ax.set_ylabel("Reaction Time (s)")
            
            # Create legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', linestyle='None', color='green', label='Rewarded', markersize=7),
                Line2D([0], [0], marker='x', linestyle='None', color='red', label='Punished', markersize=7),
                Line2D([0], [0], marker='o', linestyle='None', color='black', markerfacecolor='none', label='Did Not Choose', markersize=7),
                Line2D([0], [0], marker='s', linestyle='None', color='lightblue', alpha=0.4, label='Naive Block', markersize=10)
            ]
            
            if 'is_opto' in df_trials.columns and df_trials['is_opto'].any():
                legend_elements.append(
                    Line2D([0], [0], marker='^', linestyle='None',
                           color='purple', label='Opto Trial', markersize=7)
                )
            
            if 'MoveCorrectSpout' in df_trials.columns and df_trials['MoveCorrectSpout'].any():
                legend_elements.append(
                    Line2D([0], [0], marker='+', linestyle='None',
                           color='purple', label='MoveCorrectSpout', markersize=10)
                )
            
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.005, 0.97),
                     borderaxespad=0, frameon=False, fontsize=9)
            
            # Title
            subject_name = df_trials.get('subject_name', [subject_id]).iloc[0] if 'subject_name' in df_trials.columns else subject_id
            raw_date = session_data.get('session_info', {}).get('date', 'Unknown')
            
            # Convert date to '17-Apr-2025' format
            if raw_date != 'Unknown':
                try:
                    if len(raw_date) == 8 and raw_date.isdigit():
                        dt = datetime.strptime(raw_date, '%Y%m%d')
                        session_date = dt.strftime('%d-%b-%Y')
                    else:
                        session_date = raw_date
                except:
                    session_date = raw_date
            else:
                session_date = 'Unknown'
            
            title = f"{subject_name}  {session_date}  Session Overview - Reaction Times"
            ax.set_title(title)
            
            plt.tight_layout(pad=1.0)
            
            # Use infrastructure to save and show with subfolder
            filename = f'{subject_id}_session_overview_rt_{raw_date}.png'
            self.infrastructure.save_figure(fig, filename, subject_id=subject_id, 
                                          subfolder='session_overviews',
                                          description=f'Reaction time overview for session {session_name}')
            self.infrastructure.show_figure(fig, show_plot=show_plot)
            plt.close()
            
            self.logger.info(f"VZ-EBC: Generated session overview RT plot for {subject_id} session {session_name}")
            
        except Exception as e:
            self.logger.error(f"VZ-EBC: Failed to generate session overview RT plot for {subject_id} session {session_name}: {e}")


    def _plot_trial_outcomes_by_session(self, subject_id: str, subject_analysis: Dict[str, Any], do_plot=True, show_plot=None):
        """Plot trial outcomes across all sessions for a subject."""
        # Check if plotting is enabled
        if not do_plot:
            return
            
            
        try:
            if 'combined_sessions_df' not in subject_analysis:
                self.logger.warning(f"VZ-EBC: No combined session data found for {subject_id}")
                return
            
            df_all_sessions = subject_analysis['combined_sessions_df'].copy()
            
            if df_all_sessions.empty:
                self.logger.warning(f"VZ-EBC: Empty combined session data for {subject_id}")
                return
            
            # Apply filters as in the old architecture
            # Filter out no lick trials
            if 'lick' in df_all_sessions.columns:
                df_all_sessions = df_all_sessions[df_all_sessions['lick'] != 0]
            
            # Filter out naive trials
            if 'naive' in df_all_sessions.columns:
                df_all_sessions = df_all_sessions[df_all_sessions['naive'] == 0]
            
            # Filter out move single spout trials
            if 'MoveCorrectSpout' in df_all_sessions.columns:
                df_all_sessions = df_all_sessions[df_all_sessions['MoveCorrectSpout'] == 0]
            
            if df_all_sessions.empty:
                self.logger.warning(f"VZ-EBC: No trials remaining after filtering for {subject_id}")
                return
            
            # Create outcome column based on mouse_correct if needed
            if 'outcome' not in df_all_sessions.columns and 'mouse_correct' in df_all_sessions.columns:
                df_all_sessions['outcome'] = df_all_sessions['mouse_correct'].map({
                    1: 'Reward',
                    0: 'Punish'
                })
                # Handle DidNotChoose cases if they exist
                if 'lick' in df_all_sessions.columns:
                    df_all_sessions.loc[df_all_sessions['lick'] == 0, 'outcome'] = 'DidNotChoose'
            
            # Create trial_side column if needed
            if 'trial_side' not in df_all_sessions.columns and 'is_right' in df_all_sessions.columns:
                df_all_sessions['trial_side'] = df_all_sessions['is_right'].map({
                    0: 'left',
                    1: 'right'
                })
            
            fig, ax = plt.subplots(figsize=(30, 3.5))
            
            # Call the plotting function adapted from standalone code
            self._plot_outcomes_bars(df_all_sessions, ax)
            
            # Use infrastructure to save and show with subfolder
            filename = f'{subject_id}_trial_outcomes_by_session.png'
            self.infrastructure.save_figure(fig, filename, subject_id=subject_id, 
                                          subfolder='cross_session_analysis',
                                          description=f'Trial outcomes across all sessions for {subject_id}')
            self.infrastructure.show_figure(fig, show_plot=show_plot)
            plt.close()
            
            self.logger.info(f"VZ-EBC: Generated trial outcomes by session plot for {subject_id}")
            
        except Exception as e:
            self.logger.error(f"VZ-EBC: Failed to generate trial outcomes by session plot for {subject_id}: {e}")
    
    def _plot_outcomes_bars(self, df, ax):
        """Helper method to create the outcome bars plot."""
        outcome_colors = {
            'Reward': '#2E7D32',         # Forest green (darker, more professional)
            'Punish': '#C62828',         # Dark red (less harsh than pure red)
            'DidNotChoose': '#e0e0e0',   # very light grey
            'RewardNaive': '#66BB6A',    # Medium green (lighter than main reward)
            'PunishNaive': '#EF5350'     # Medium red (lighter than main punish)
        }
        
        split_opto = True
        min_trials = 10
        normalize = True
        bar_spacing = 0.3
        
        group_cols = ['date', 'trial_side']
        if split_opto and 'is_opto' in df.columns:
            group_cols.append('is_opto')
        group_cols.append('outcome')
        
        summary = df.groupby(group_cols).size().unstack(fill_value=0)
        
        session_ticks = []
        session_labels = []
        sessions = sorted(df['date'].unique())
        x = 0
        label_order = ['left', 'right']
        
        for session in sessions:
            session_df = summary.loc[session] if session in summary.index.get_level_values(0) else None
            if session_df is None or session_df.sum().sum() < min_trials:
                continue
            
            x_positions = []
            
            # Build expected group order for this session
            expected_groups = []
            for side in label_order:
                if split_opto and 'is_opto' in df.columns:
                    expected_groups.append((side, 0))  # control
                    expected_groups.append((side, 1))  # opto
                else:
                    expected_groups.append(side)
            
            for group in expected_groups:
                if split_opto and 'is_opto' in df.columns:
                    if group not in session_df.index:
                        outcome_counts = {k: 0 for k in outcome_colors.keys()}
                    else:
                        outcome_counts = session_df.loc[group].to_dict()
                else:
                    if group not in session_df.index:
                        outcome_counts = {k: 0 for k in outcome_colors.keys()}
                    else:
                        outcome_counts = session_df.loc[group].to_dict()
                
                # Sort outcome keys by color map order
                outcomes_sorted = [k for k in outcome_colors if k in outcome_counts]
                
                # Compute total and optionally normalize
                total = sum(outcome_counts.values())
                bottom = 0
                for outcome in outcomes_sorted:
                    count = outcome_counts[outcome]
                    value = (count / total) if (normalize and total > 0) else count
                    
                    # Plot the bar
                    bar = ax.bar(x, value, bottom=bottom, color=outcome_colors[outcome],
                                label=outcome if x == 0 else "", zorder=3)
                    
                    # Add count annotations for bars with sufficient height and count > 0
                    if count > 0 and value > 0.05:  # Only annotate if bar is tall enough and has data
                        text_y = bottom + value / 2  # Center of the bar segment
                        ax.text(x, text_y, f'{count}', ha='center', va='center', 
                               fontsize=8, fontweight='bold', color='white')
                    
                    bottom += value
                
                # Label bar as L / LO / R / RO
                if split_opto and 'is_opto' in df.columns:
                    label = f"{group[0][0].upper()}{'O' if group[1] == 1 else ''}"
                else:
                    label = group[0][0].upper()
                ax.text(x, -0.05 if normalize else -3, label, ha='center', va='top', fontsize=9, clip_on=False)
                
                x_positions.append(x)
                x += 1
            
            # Add spacing between session groups
            x += bar_spacing
            center_x = np.mean(x_positions)
            session_ticks.append(center_x)
            
            # Format session date
            try:
                if len(str(session)) == 8 and str(session).isdigit():
                    dt = datetime.strptime(str(session), '%Y%m%d')
                    formatted_date = dt.strftime('%m-%d')
                else:
                    formatted_date = str(session)
            except:
                formatted_date = str(session)
            session_labels.append(formatted_date)
        
        ax.set_xticks(session_ticks)
        ax.set_xticklabels(session_labels, rotation=45, ha='right')
        ax.tick_params(axis='x', pad=12)
        ax.set_ylabel("Trial Proportion" if normalize else "Trial Count")
        
        # Title with subject and date range
        subject = df['subject_name'].iloc[0]
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
        min_date = df['date'].min().strftime("%Y-%m-%d")
        max_date = df['date'].max().strftime("%Y-%m-%d")
        title = f"{subject} | Sessions {min_date} to {max_date}"
        ax.set_title(title, y=1.05)
        
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

 
    def _visualize_group(self):
        """Generate SID-specific group plots - placeholder for future implementation."""
        self.logger.info("VZ-EBC: Group visualization placeholder - will be implemented later")
        # TODO: Implement group-level visualizations once individual plots are working
        pass
    
    def _plot_rolling_performance_across_sessions(self, subject_id: str, subject_analysis: Dict[str, Any], do_plot=True, show_plot=None):
        """Plot rolling performance across all sessions for a subject."""
        # Check if plotting is enabled
        if not do_plot:
            return
            
        try:
            if 'combined_sessions_df' not in subject_analysis:
                self.logger.warning(f"VZ-EBC: No combined session data found for {subject_id}")
                return
            
            df_all_sessions = subject_analysis['combined_sessions_df'].copy()
            
            if df_all_sessions.empty:
                self.logger.warning(f"VZ-EBC: Empty combined session data for {subject_id}")
                return
            
            # Apply same filters as trial outcomes
            if 'lick' in df_all_sessions.columns:
                df_all_sessions = df_all_sessions[df_all_sessions['lick'] != 0]
            if 'naive' in df_all_sessions.columns:
                df_all_sessions = df_all_sessions[df_all_sessions['naive'] == 0]
            if 'MoveCorrectSpout' in df_all_sessions.columns:
                df_all_sessions = df_all_sessions[df_all_sessions['MoveCorrectSpout'] == 0]
            
            if df_all_sessions.empty:
                self.logger.warning(f"VZ-EBC: No trials remaining after filtering for {subject_id}")
                return
            
            fig, ax = plt.subplots(figsize=(30, 3.5))
            
            # Call the plotting
            self._plot_rolling_performance_bars(df_all_sessions, ax)
            
            # Use infrastructure to save and show with subfolder
            filename = f'{subject_id}_rolling_performance_across_sessions.png'
            self.infrastructure.save_figure(fig, filename, subject_id=subject_id, 
                                          subfolder='cross_session_analysis',
                                          description=f'Rolling performance across all sessions for {subject_id}')
            self.infrastructure.show_figure(fig, show_plot=show_plot)
            plt.close()
            
            self.logger.info(f"VZ-EBC: Generated rolling performance plot for {subject_id}")
            
        except Exception as e:
            self.logger.error(f"VZ-EBC: Failed to generate rolling performance plot for {subject_id}: {e}")
    
    def _plot_rolling_performance_bars(self, df, ax):
        """Helper method to create the rolling performance plot."""
        from scipy.stats import gaussian_kde
        
        # Sort by session/date and add trial numbers
        df = df.sort_values(by=['date']).reset_index(drop=True)
        df['trial_num'] = np.arange(len(df))
        
        x_vals = df['trial_num'].values
        y_vals = df['mouse_correct'].astype(float).values
        
        # Parameters
        window_size = 50
        kde_bw = 0.10
        bin_size = 50
        
        # Rolling mean (centered)
        rolling_series = pd.Series(y_vals, index=x_vals)
        rolling_perf = rolling_series.rolling(window_size, min_periods=1, center=True).mean()
        ax.plot(rolling_perf.index, rolling_perf.values, label=f"Rolling (n={window_size})", color='black', linewidth=2)
        
        # KDE - Simplified approach using convolution
        if len(y_vals) > window_size:
            # Create a Gaussian kernel for smoothing
            kernel_size = int(window_size * 0.5)  # Smaller kernel for tighter tracking
            kernel = np.exp(-0.5 * (np.arange(-kernel_size, kernel_size+1) / (kernel_size/3))**2)
            kernel = kernel / kernel.sum()  # Normalize
            
            # Apply convolution to smooth the performance data
            # Pad the data to handle edges
            padded_y = np.pad(y_vals, (kernel_size, kernel_size), mode='edge')
            smoothed = np.convolve(padded_y, kernel, mode='valid')
            
            # Ensure we have the right length
            if len(smoothed) != len(x_vals):
                smoothed = smoothed[:len(x_vals)]
            
            ax.plot(x_vals, smoothed, label=f"KDE P(Correct)", linestyle='--', color='blue')
        else:
            ax.plot([], [], label="KDE (insufficient data)", color='blue', linestyle='--')
        
        # Step-binned average
        df['trial_bin'] = df['trial_num'] // bin_size
        bin_perf = df.groupby('trial_bin')['mouse_correct'].mean()
        bin_centers = bin_perf.index * bin_size + bin_size // 2
        ax.step(bin_centers, bin_perf.values, where='mid', label=f"Binned Avg (n={bin_size})", color='orange', alpha=0.7)
        
        # Session transitions
        session_starts = df.groupby('date')['trial_num'].min().values
        for start in session_starts:
            ax.axvline(start, color='gray', linestyle='--', alpha=0.3)
        
        for session_date, start in zip(df['date'].unique(), session_starts):
            try:
                if len(str(session_date)) == 8 and str(session_date).isdigit():
                    dt = datetime.strptime(str(session_date), '%Y%m%d')
                    formatted_date = dt.strftime("%m-%d")
                else:
                    formatted_date = str(session_date)
            except:
                formatted_date = str(session_date)
            ax.text(start, 1.05, formatted_date, ha='left', va='bottom', fontsize=8, rotation=45)
        
        # Title and labels
        subject = df['subject_name'].iloc[0]
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
        min_date = df['date'].min().strftime("%Y-%m-%d")
        max_date = df['date'].max().strftime("%Y-%m-%d")
        ax.set_title(f"{subject} | Sessions {min_date} to {max_date}", fontsize=10, y=1.1)
        
        ax.set_xlabel("Cumulative Trial Number")
        ax.set_ylabel("Performance (P(Correct))")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_isi_pdf(self, subject_id: str, session_name: str, session_data: Dict[str, Any], session_summary: Dict[str, Any], do_plot=True, show_plot=None):
        """Plot ISI probability density functions for a session."""
        # Check if plotting is enabled
        if not do_plot:
            return

            
        try:
            # Check if ISI analysis exists in session summary
            if 'session_isi_pdf' not in session_summary:
                self.logger.info(f"VZ-EBC: No ISI analysis found for {subject_id} session {session_name}")
                return
            
            session_isi_pdf = session_summary['session_isi_pdf']
            
            # Check for errors in ISI analysis
            if 'error' in session_isi_pdf:
                self.logger.warning(f"VZ-EBC: ISI analysis error for {subject_id} session {session_name}: {session_isi_pdf['error']}")
                return
            
            pdf_data = session_isi_pdf.get('pdf_data', {})
            if not pdf_data:
                self.logger.warning(f"VZ-EBC: No PDF data for {subject_id} session {session_name}")
                return

            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Color mapping for different conditions
            color_map = {
                'left': {'line': '#1f77b4', 'fill': '#aec7e8'},   # blue tones
                'right': {'line': '#d62728', 'fill': '#f7b6b2'},   # red tones
                ('left', 0): {'line': '#1f77b4', 'fill': '#aec7e8'},   # left control
                ('left', 1): {'line': '#5fa2d3', 'fill': '#d0e5f5'},   # left opto
                ('right', 0): {'line': '#d62728', 'fill': '#f7b6b2'},  # right control
                ('right', 1): {'line': '#e36a6a', 'fill': '#fcdede'},  # right opto
            }
            
            # Plot each PDF curve
            for group, data in pdf_data.items():
                if 'error' in data:
                    continue
                
                # Determine label and colors
                if isinstance(group, tuple):
                    isi_side, opto = group
                    side_label = 'Short' if isi_side == 'left' else 'Long'
                    opto_label = 'Control' if opto == 0 else 'Opto'
                    label = f"{side_label} {opto_label} ISI (n={data['count']})"
                    colors = color_map.get(group, {'line': 'gray', 'fill': 'lightgray'})
                else:
                    side_label = 'Short' if group == 'left' else 'Long'
                    label = f"{side_label} ISI (n={data['count']})"
                    colors = color_map.get(group, {'line': 'gray', 'fill': 'lightgray'})
                
                # Plot line and fill
                ax.plot(data['x'], data['y'], label=label, color=colors['line'], linewidth=2)
                ax.fill_between(data['x'], data['y'], color=colors['fill'], alpha=0.5)
                
                # Add mean line
                ax.axvline(data['mean'], color=colors['line'], linestyle='--', alpha=0.7)
            
            # Formatting
            from matplotlib.ticker import FormatStrFormatter
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
            ax.set_xlabel("ISI Duration (ms)")
            ax.set_ylabel("Probability Density")
            
            # Title
            session_info = session_data.get('session_info', {})
            subject_name = session_info.get('subject_name', subject_id)
            raw_date = session_info.get('date', 'Unknown')
            
            # Format date
            if raw_date != 'Unknown' and len(raw_date) == 8 and raw_date.isdigit():
                try:
                    dt = datetime.strptime(raw_date, '%Y%m%d')
                    formatted_date = dt.strftime('%d-%b-%Y')
                except:
                    formatted_date = raw_date
            else:
                formatted_date = raw_date
            
            title = f"{subject_name}  {formatted_date}  ISI Probability Density"
            ax.set_title(title, fontsize=12)
            
            # Legend
            ax.legend(
                loc='upper left',
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0,
                frameon=False,
                fontsize=12
            )
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout(pad=2.0)
            
            # Save figure
            filename = f'{subject_id}_isi_pdf_{raw_date}.png'
            self.infrastructure.save_figure(fig, filename, subject_id=subject_id, 
                                          subfolder='session_overviews',
                                          description=f'ISI probability density for session {session_name}')
            self.infrastructure.show_figure(fig, show_plot=show_plot)
            plt.close()
            
            self.logger.info(f"VZ-EBC: Generated ISI PDF plot for {subject_id} session {session_name}")
            
        except Exception as e:
            self.logger.error(f"VZ-EBC: Failed to generate ISI PDF plot for {subject_id} session {session_name}: {e}")
    
    def _plot_psychometric(self, subject_id: str, session_name: str, session_data: Dict[str, Any], session_summary: Dict[str, Any], do_plot=True, show_plot=None):
        """Plot psychometric curves for a session."""
        # Check if plotting is enabled
        if not do_plot:
            return

        try:
            # Check if psychometric analysis exists in session summary
            if 'psychometric_analysis' not in session_summary:
                self.logger.info(f"VZ-EBC: No psychometric analysis found for {subject_id} session {session_name}")
                return
            
            psychometric_analysis = session_summary['psychometric_analysis']
            
            # Check for errors in psychometric analysis
            if 'error' in psychometric_analysis:
                self.logger.warning(f"VZ-EBC: Psychometric analysis error for {subject_id} session {session_name}: {psychometric_analysis['error']}")
                return
            
            psychometric_curves = psychometric_analysis.get('psychometric_curves', {})
            if not psychometric_curves:
                self.logger.warning(f"VZ-EBC: No psychometric curves for {subject_id} session {session_name}")
                return

            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Color mapping for different conditions
            color_map = {
                'all': '#1f77b4',      # blue
                'control': '#1f77b4',  # blue
                'opto': '#ff7f0e',     # orange
            }
            
            # Store data for reference lines - use control data as primary reference
            control_df = None
            all_isi_means = []
            x_min_global, x_max_global = float('inf'), float('-inf')
            
            # Track failed/skipped fits for proper annotation positioning
            failed_fit_count = 0
            
            # Plot each condition
            for condition, curve_data in psychometric_curves.items():
                if 'data' not in curve_data:
                    continue
                
                # Convert data back to DataFrame for plotting
                psychometric_df = pd.DataFrame(curve_data['data'])
                if psychometric_df.empty:
                    continue
                
                color = color_map.get(condition, 'gray')
                
                # Store control data for reference lines and x-axis range
                if condition in ['control', 'all']:
                    control_df = psychometric_df.copy()
                
                # Track global x-axis range based on all conditions
                x_vals = psychometric_df['stim_value'].values
                x_min_global = min(x_min_global, x_vals.min())
                x_max_global = max(x_max_global, x_vals.max())
                
                # Store ISI means for all conditions
                if 'isi_mean' in psychometric_df.columns:
                    isi_mean = psychometric_df['isi_mean'].iloc[0]
                    if not np.isnan(isi_mean):
                        all_isi_means.append((condition, isi_mean, color))
                
                # Plot data points
                x = psychometric_df['stim_value'].values
                y = psychometric_df['p_right'].values
                err = psychometric_df['stderr'].values
                n_trials = psychometric_df['n_trials'].values
                
                # Plot points and error bars
                ax.errorbar(x, y, yerr=1.96*err, fmt='o-', color=color, 
                           label=f'{condition.title()} (n={curve_data["n_trials"]})',
                           alpha=0.7, capsize=3)
                
                # Add confidence interval fill
                ax.fill_between(x, y - 1.96*err, y + 1.96*err, alpha=0.15, color=color)
                
                # Annotate trial counts
                for xi, yi, n in zip(x, y, n_trials):
                    ax.annotate(f"n={n}", (xi, yi), textcoords="offset points", 
                               xytext=(0, 10), ha='center', fontsize=7, 
                               color=color, alpha=0.7)
                
                # Plot fit curve if available
                fit_data = curve_data.get('fit', {})
                if fit_data.get('fit_method') not in ['failed', 'skipped_insufficient_isi'] and 'fit_x' in fit_data and 'fit_y' in fit_data:
                    fit_x = np.array(fit_data['fit_x'])
                    fit_y = np.array(fit_data['fit_y'])
                    
                    # Only plot if fit data is valid
                    if not (np.isnan(fit_x).all() or np.isnan(fit_y).all()):
                        ax.plot(fit_x, fit_y, '--', color=color, alpha=0.8, linewidth=2)
                        
                        # Add threshold line if available
                        # threshold = fit_data.get('threshold')
                        # if threshold and not np.isnan(threshold):
                        #     ax.axvline(threshold, linestyle=':', color=color, alpha=0.5)
                        #     ax.annotate(f"Thresh: {threshold:.0f}ms", 
                        #                (threshold, 0.05), xytext=(5, 0), 
                        #                textcoords='offset points', fontsize=12, 
                        #                color=color, rotation=90, va='bottom')
                
                # Annotate if fit was skipped or failed
                elif fit_data.get('fit_method') in ['failed', 'skipped_insufficient_isi']:
                    fit_status = "Fit skipped (insufficient ISI)" if fit_data.get('fit_method') == 'skipped_insufficient_isi' else "Fit failed"
                    ax.annotate(f"{condition}: {fit_status}", xy=(0.05, 0.95 - 0.05*failed_fit_count), 
                               xycoords='axes fraction', fontsize=8, color=color,
                               verticalalignment='top')
                    failed_fit_count += 1
            
            # Add reference lines
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            
            # Add mean ISI lines for all conditions
            for condition, isi_mean, color in all_isi_means:
                ax.axvline(isi_mean, color=color, linestyle=':', linewidth=1, alpha=0.7)
                # Offset the text annotations slightly for multiple conditions
                y_offset = 0.02 + 0.1 * len([c for c, _, _ in all_isi_means if c == condition])
                ax.annotate(f"{condition.title()} ISI Mean\n{isi_mean:.0f}ms", 
                           xy=(isi_mean, y_offset), xytext=(5, 5),
                           textcoords='offset points', fontsize=12,
                           color=color, rotation=90, va='bottom')
            
            # Formatting - use global x-axis range from all conditions
            ax.set_ylim(0, 1.05)
            if x_min_global != float('inf') and x_max_global != float('-inf'):
                x_range = x_max_global - x_min_global
                ax.set_xlim(x_min_global - 0.1*x_range, x_max_global + 0.1*x_range)
            
            ax.set_xlabel("ISI Duration (ms)")
            ax.set_ylabel("P(Right)")
            
            # Title
            session_info = session_data.get('session_info', {})
            subject_name = session_info.get('subject_name', subject_id)
            raw_date = session_info.get('date', 'Unknown')
            
            # Format date
            if raw_date != 'Unknown' and len(raw_date) == 8 and raw_date.isdigit():
                try:
                    dt = datetime.strptime(raw_date, '%Y%m%d')
                    formatted_date = dt.strftime('%d-%b-%Y')
                except:
                    formatted_date = raw_date
            else:
                formatted_date = raw_date
            
            title = f"{subject_name}  {formatted_date}  ISI Psychometric"
            ax.set_title(title, fontsize=12)
            
            # Legend
            ax.legend(
                loc='upper left',
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0,
                frameon=False,
                fontsize=12
            )
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout(pad=2.0)
            
            # Save figure
            filename = f'{subject_id}_psychometric_{raw_date}.png'
            self.infrastructure.save_figure(fig, filename, subject_id=subject_id, 
                                          subfolder='session_overviews',
                                          description=f'Psychometric curves for session {session_name}')
            self.infrastructure.show_figure(fig, show_plot=show_plot)
            plt.close()
            
            self.logger.info(f"VZ-EBC: Generated psychometric plot for {subject_id} session {session_name}")
            
        except Exception as e:
            self.logger.error(f"VZ-EBC: Failed to generate psychometric plot for {subject_id} session {session_name}: {e}")
    
    def _plot_rt_histogram(self, subject_id: str, session_name: str, session_data: Dict[str, Any], do_plot=True, show_plot=None):
        """Plot response time histograms by trial type and opto condition."""
        # Check if plotting is enabled
        if not do_plot:
            return
            

            
        try:
            if 'df_trials' not in session_data:
                self.logger.warning(f"VZ-EBC: No trial data found for {subject_id} session {session_name}")
                return
            
            df_trials = session_data['df_trials']
            if df_trials.empty:
                self.logger.warning(f"VZ-EBC: Empty trial data for {subject_id} session {session_name}")
                return
            
            # Check if RT data exists
            if 'RT' not in df_trials.columns:
                self.logger.warning(f"VZ-EBC: No RT data found for {subject_id} session {session_name}")
                return
            
            # Apply filters from standalone code
            df_filtered = df_trials.copy()
            if 'lick' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['lick'] != 0]
            if 'naive' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['naive'] == 0]
            if 'MoveCorrectSpout' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['MoveCorrectSpout'] == 0]
            
            if df_filtered.empty:
                self.logger.warning(f"VZ-EBC: No trials remaining after filtering for {subject_id} session {session_name}")
                return
            
            # Add trial_side column if needed
            if 'trial_side' not in df_filtered.columns and 'is_right' in df_filtered.columns:
                df_filtered['trial_side'] = df_filtered['is_right'].map({0: 'left', 1: 'right'})
            
            # Check if we have opto data
            has_opto = 'is_opto' in df_filtered.columns and df_filtered['is_opto'].any()
            
            # Color and label mapping
            color_map = {
                ('left', 0): '#1f78b4',   # Dark Blue
                ('left', 1): '#00bfff',   # Light Blue
                ('right', 0): '#e31a1c',  # Dark Red
                ('right', 1): '#ff6666',  # Light Red
            }
            
            # Set up subplot configuration
            if has_opto:
                group_keys = [('left', 0), ('left', 1), ('right', 0), ('right', 1)]
            else:
                group_keys = [('left', 0), ('right', 0)]  # Treat as control condition
                if 'is_opto' not in df_filtered.columns:
                    df_filtered['is_opto'] = 0  # Add dummy opto column
            
            n_rows = len(group_keys)
            fig, axes = plt.subplots(n_rows, 1, figsize=(6, 1.8 * n_rows), 
                                   sharex=True, gridspec_kw={'hspace': 0.15})
            
            # Handle single subplot case
            if n_rows == 1:
                axes = [axes]
            
            # Global RT limits for consistent x-axis
            global_rt = df_filtered['RT'].dropna()
            if global_rt.empty:
                self.logger.warning(f"VZ-EBC: No valid RT data for {subject_id} session {session_name}")
                plt.close(fig)
                return
            
            xlim = (0, min(1000, global_rt.max() * 1.1))  # Cap at 1000ms or 110% of max
            
            # Shared binning
            bins = 50
            bin_edges = np.linspace(global_rt.min(), global_rt.max(), bins + 1)
            
            for ax, (stim, opto) in zip(axes, group_keys):
                # Filter data for this condition
                condition_mask = (df_filtered['trial_side'] == stim) & (df_filtered['is_opto'] == opto)
                group_df = df_filtered[condition_mask]
                rt_vals = group_df['RT'].dropna()
                
                if rt_vals.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_ylabel("Count", fontsize=8)
                    continue
                
                color = color_map.get((stim, opto), 'gray')
                
                # Create label
                if has_opto:
                    label = f"{stim.capitalize()} {'Opto' if opto else 'Control'} (n={len(rt_vals)})"
                else:
                    label = f"{stim.capitalize()} (n={len(rt_vals)})"
                
                # Plot histogram
                ax.hist(rt_vals, bins=bin_edges, color=color, edgecolor='black', alpha=0.6)
                ax.set_ylabel("Count", fontsize=8)
                ax.set_title(label, fontsize=9, pad=2)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(xlim)
                ax.tick_params(axis='both', labelsize=7)
            
            axes[-1].set_xlabel("Response Time (ms)", fontsize=9)
            
            # Title
            session_info = session_data.get('session_info', {})
            subject_name = session_info.get('subject_name', subject_id)
            raw_date = session_info.get('date', 'Unknown')
            
            # Format date
            if raw_date != 'Unknown' and len(raw_date) == 8 and raw_date.isdigit():
                try:
                    dt = datetime.strptime(raw_date, '%Y%m%d')
                    formatted_date = dt.strftime('%d-%b-%Y')
                except:
                    formatted_date = raw_date
            else:
                formatted_date = raw_date
            
            title = f"{subject_name}  {formatted_date}  Response Time Distribution"
            fig.suptitle(title, fontsize=12, y=0.98)
            
            plt.tight_layout()
            
            # Save figure
            filename = f'{subject_id}_rt_histogram_{raw_date}.png'
            self.infrastructure.save_figure(fig, filename, subject_id=subject_id, 
                                          subfolder='session_overviews',
                                          description=f'Response time histogram for session {session_name}')
            self.infrastructure.show_figure(fig, show_plot=show_plot)
            plt.close()
            
            self.logger.info(f"VZ-EBC: Generated RT histogram plot for {subject_id} session {session_name}")
            
        except Exception as e:
            self.logger.error(f"VZ-EBC: Failed to generate RT histogram plot for {subject_id} session {session_name}: {e}")
    
    def _plot_rt_stats(self, subject_id: str, session_name: str, session_data: Dict[str, Any], session_summary: Dict[str, Any], do_plot=True, show_plot=None):
        """Plot response time statistics as box plots for a session."""
        # Check if plotting is enabled
        if not do_plot:
            return
            
        try:
            # Check if RT analysis exists in session summary
            if 'rt_analysis' not in session_summary:
                self.logger.info(f"VZ-EBC: No RT analysis found for {subject_id} session {session_name}")
                return
            
            rt_analysis = session_summary['rt_analysis']
            
            # Check for errors in RT analysis
            if 'error' in rt_analysis:
                self.logger.warning(f"VZ-EBC: RT analysis error for {subject_id} session {session_name}: {rt_analysis['error']}")
                return
            
            rt_statistics = rt_analysis.get('rt_statistics', {})
            if not rt_statistics:
                self.logger.warning(f"VZ-EBC: No RT statistics for {subject_id} session {session_name}")
                return

            # Check if we have opto data
            has_opto = rt_analysis.get('has_opto', False)
            
            # Always use 2 subplots: Left and Right
            # Within each subplot, show Control and Opto if available
            fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
            
            # Define trial sides
            trial_sides = ['left', 'right']
            side_labels = ['Left Trials', 'Right Trials']
            
            for i, (side, side_label) in enumerate(zip(trial_sides, side_labels)):
                ax = axes[i]
                
                if has_opto:
                    # Show both control and opto for this trial side
                    control_key = (side, 0)
                    opto_key = (side, 1)
                    
                    control_stats = rt_statistics.get(control_key, {})
                    opto_stats = rt_statistics.get(opto_key, {})
                    
                    # Plot positions for control and opto
                    positions = [0.8, 1.2]  # Slightly offset from center
                    labels = ['Control', 'Opto']
                    colors = ['#1f78b4' if side == 'left' else '#e31a1c',  # Control: blue/red
                             '#00bfff' if side == 'left' else '#ff6666']   # Opto: light blue/light red
                    
                    for pos, label, color, stats in zip(positions, labels, colors, [control_stats, opto_stats]):
                        if 'error' in stats or stats.get('count', 0) == 0:
                            # Show placeholder for missing data
                            ax.text(pos, 0.5, 'No data', ha='center', va='center', 
                                   transform=ax.transData, fontsize=8, rotation=90)
                            continue
                        
                        # Create box plot data
                        box_data = {
                            'med': stats.get('median', 0),
                            'q1': stats.get('p25', 0),
                            'q3': stats.get('p75', 0),
                            'whislo': stats.get('whisker_low', 0),
                            'whishi': stats.get('whisker_high', 0),
                            'fliers': stats.get('outliers', [])
                        }
                        
                        # Plot box plot
                        bp = ax.bxp([box_data], positions=[pos], widths=0.3, patch_artist=True,
                                   showfliers=True, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.6})
                        
                        # Color the box
                        bp['boxes'][0].set_facecolor(color)
                        bp['boxes'][0].set_alpha(0.7)
                        
                        # Add mean as a separate marker
                        mean_val = stats.get('mean', 0)
                        ax.plot(pos, mean_val, marker='D', color='white', markersize=6, 
                               markeredgecolor='black', markeredgewidth=1.5, zorder=5)
                        
                        # Add scatter of raw data points (optional, for small datasets)
                        raw_data = stats.get('raw_data', [])
                        x_jitter = np.random.normal(pos, 0.05, len(raw_data))
                        ax.scatter(x_jitter, raw_data, alpha=0.3, s=20, color=color, zorder=4)                        
                        # Remove the count label positioning relative to y-limits
                        # The count will be shown in the x-tick labels instead
                    
                    # Set x-axis for opto condition with count information
                    ax.set_xlim(0.5, 1.5)
                    ax.set_xticks([0.8, 1.2])
                    # Include count in the x-tick labels for consistent positioning
                    control_count = control_stats.get('count', 0)
                    opto_count = opto_stats.get('count', 0)
                    ax.set_xticklabels([f'Control\nn={control_count}', f'Opto\nn={opto_count}'], fontsize=12)
                    
                else:
                    # Show only control data for this trial side
                    stats = rt_statistics.get(side, {})
                    
                    if 'error' in stats or stats.get('count', 0) == 0:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                               transform=ax.transAxes, fontsize=12)
                        ax.set_xlim(0.5, 1.5)
                        ax.set_xticks([1])
                        ax.set_xticklabels(['n=0'])
                        ax.set_title(side_label, fontsize=10)
                        continue
                    
                    # Create box plot data
                    box_data = {
                        'med': stats.get('median', 0),
                        'q1': stats.get('p25', 0),
                        'q3': stats.get('p75', 0),
                        'whislo': stats.get('whisker_low', 0),
                        'whishi': stats.get('whisker_high', 0),
                        'fliers': stats.get('outliers', [])
                    }
                    
                    # Plot box plot
                    color = '#1f78b4' if side == 'left' else '#e31a1c'
                    bp = ax.bxp([box_data], positions=[1], widths=0.6, patch_artist=True,
                               showfliers=True, flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.6})
                    
                    # Color the box
                    bp['boxes'][0].set_facecolor(color)
                    bp['boxes'][0].set_alpha(0.7)
                    
                    # Add mean as a separate marker
                    mean_val = stats.get('mean', 0)
                    ax.plot(1, mean_val, marker='D', color='white', markersize=8, 
                           markeredgecolor='black', markeredgewidth=2, zorder=5)
                    
                    # Add scatter of raw data points (optional, for small datasets)
                    raw_data = stats.get('raw_data', [])
                    x_jitter = np.random.normal(1, 0.05, len(raw_data))
                    ax.scatter(x_jitter, raw_data, alpha=0.3, s=20, color=color, zorder=4)
                    
                    # Set x-axis for non-opto condition
                    ax.set_xlim(0.5, 1.5)
                    ax.set_xticks([1])
                    ax.set_xticklabels([f'n={stats.get("count", 0)}'], fontsize=12)
                
                # Common formatting for each subplot
                ax.set_title(side_label, fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Add statistics text box in corner
                if has_opto:
                    # Show stats for both conditions
                    control_stats = rt_statistics.get((side, 0), {})
                    opto_stats = rt_statistics.get((side, 1), {})
                    
                    stats_text = ""
                    if control_stats.get('count', 0) > 0:
                        stats_text += f"Control: {control_stats.get('median', 0):.0f}ms\n"
                    if opto_stats.get('count', 0) > 0:
                        stats_text += f"Opto: {opto_stats.get('median', 0):.0f}ms"
                    
                else:
                    stats = rt_statistics.get(side, {})
                    if stats.get('count', 0) > 0:
                        stats_text = f"Med: {stats.get('median', 0):.0f}ms\nMean: {stats.get('mean', 0):.0f}ms"
                    else:
                        stats_text = "No data"
                
                if stats_text.strip():
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set common y-label
            axes[0].set_ylabel('Response Time (ms)')
            
            # Title
            session_info = session_data.get('session_info', {})
            subject_name = session_info.get('subject_name', subject_id)
            raw_date = session_info.get('date', 'Unknown')
            
            # Convert date to '17-Apr-2025' format
            if raw_date != 'Unknown':
                try:
                    if len(raw_date) == 8 and raw_date.isdigit():
                        dt = datetime.strptime(raw_date, '%Y%m%d')
                        session_date = dt.strftime('%d-%b-%Y')
                    else:
                        session_date = raw_date
                except:
                    session_date = raw_date
            else:
                session_date = 'Unknown'
            
            title = f"{subject_name}  {session_date}  Response Time Statistics"
            fig.suptitle(title, fontsize=12)
            
            plt.tight_layout()
            
            # Save figure
            filename = f'{subject_id}_rt_stats_{raw_date}.png'
            self.infrastructure.save_figure(fig, filename, subject_id=subject_id, 
                                          subfolder='session_overviews',
                                          description=f'Response time statistics for session {session_name}')
            self.infrastructure.show_figure(fig, show_plot=show_plot)
            plt.close()
            
            self.logger.info(f"VZ-EBC: Generated RT statistics plot for {subject_id} session {session_name}")
            
        except Exception as e:
            self.logger.error(f"VZ-EBC: Failed to generate RT statistics plot for {subject_id} session {session_name}: {e}")






