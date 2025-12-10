import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

class EyeBlinkConditioningAnalyzer:
    """
    Experiment-specific analyzer for eye blink conditioning data.
    Handles analysis specific to this experiment type.
    """
    
    def __init__(self, config_manager, subject_list, loaded_data, logger):
        """Initialize with ConfigManager, subject list, and loaded data."""
        self.config_manager = config_manager
        self.subject_list = subject_list
        self.loaded_data = loaded_data
        self.logger = logger
        
        self.logger.info("DA-EBC: Initializing EyeBlinkConditioningAnalyzer...")
        
        # Get experiment-specific analysis config
        experiment_config = config_manager.config.get('experiment_configs', {}).get(config_manager.experiment_name, {})
        self.analysis_config = experiment_config.get('analysis', {})
        
        self.logger.info("DA-EBC: EyeBlinkConditioningAnalyzer initialized successfully")
    
    def analyze_data(self) -> Dict[str, Any]:
        """
        Perform eye blink conditioning specific analysis.
        
        Returns:
            Analysis results dictionary
        """
        self.logger.info("DA-EBC: Starting eye blink conditioning analysis...")
        
        analysis_results = {
            'analysis_type': 'eye_blink_conditioning',
            'experiment_config': self.config_manager.experiment_name,
            'subjects_analyzed': len(self.subject_list),
            'total_sessions': self.loaded_data['metadata']['total_sessions_loaded'],
            'subject_results': {},
            'summary': {}
        }
        
        # Analyze each subject
        for subject_id, subject_data in self.loaded_data['subjects'].items():
            analysis_results['summary'][subject_id] = {
                'sessions_analyzed': subject_data['metadata']['sessions_loaded'],
                'sessions_requested': subject_data['metadata']['sessions_requested']
            }            
            subject_analysis = self._analyze_subject(subject_id, subject_data)
            analysis_results['subject_results'][subject_id] = subject_analysis            

        
        # Perform cross-subject analysis
        analysis_results['group_analysis'] = self._analyze_group()
        
        self.logger.info("DA-EBC: eye blink conditioning analysis completed")
        return analysis_results
    
    def _analyze_subject(self, subject_id: str, subject_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data for a single subject."""
        self.logger.info(f"DA-EBC: Analyzing subject {subject_id}")
        
        # Basic subject analysis
        subject_results = {
            'subject_id': subject_id,
            'sessions_analyzed': subject_data['metadata']['sessions_loaded'],
            'total_trials': 0,
            'session_summaries': {},
            'session_data': subject_data['sessions']  # Store the session data
        }
        
        # Combine all session dataframes for cross-session analysis
        all_sessions_dfs = []
        for session_name, session_data in subject_data['sessions'].items():
            if 'df_trials' in session_data and not session_data['df_trials'].empty:
                df_session = session_data['df_trials'].copy()
                # Add session info to each trial
                df_session['session_name'] = session_name
                df_session['date'] = session_data.get('session_info', {}).get('date', session_name)
                df_session['subject_name'] = subject_id
                all_sessions_dfs.append(df_session)
        
        # Store combined dataframe if we have sessions
        if all_sessions_dfs:
            subject_results['combined_sessions_df'] = pd.concat(all_sessions_dfs, ignore_index=True)
        else:
            subject_results['combined_sessions_df'] = pd.DataFrame()
        
        # Analyze each session
        for session_name, session_data in subject_data['sessions'].items():
            df_trials = session_data['df_trials']
            session_summary = self._analyze_session(session_name, df_trials)
            subject_results['session_summaries'][session_name] = session_summary
            subject_results['total_trials'] += len(df_trials)
        
        return subject_results
    
    def _analyze_session(self, session_name: str, df_trials: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data for a single session."""
        if df_trials.empty:
            return {'session_name': session_name, 'trials': 0}
        
        # Basic session analysis
        session_summary = {
            'session_name': session_name,
            'trials': len(df_trials),
            'performance': {}
        }
        
        # # Calculate basic performance metrics
        # if 'mouse_correct' in df_trials.columns:
        #     session_summary['performance']['accuracy'] = df_trials['mouse_correct'].mean()
        #     session_summary['performance']['total_correct'] = df_trials['mouse_correct'].sum()
        
        # # Add more session-specific analysis
        # # TODO: Add specific SID analysis (RT, learning curves, etc.)
        
        # # Add ISI analysis
        # session_isi_pdf = self._compute_isi_pdf(df_trials)
        # session_summary['session_isi_pdf'] = session_isi_pdf
        
        # # Add psychometric analysis
        # psychometric_analysis = self._compute_psychometric_analysis(df_trials)
        # session_summary['psychometric_analysis'] = psychometric_analysis
        
        # # Add RT analysis
        # rt_analysis = self._compute_rt_analysis(df_trials)
        # session_summary['rt_analysis'] = rt_analysis
        
        return session_summary

    def _compute_isi_pdf(self, df_trials: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute ISI (Inter-Stimulus Interval) probability density functions.
        Adapted from standalone plot_isi_pdf code.
        """
        try:
            from scipy.stats import gaussian_kde
            import numpy as np
            
            # Check if ISI column exists
            if 'isi' not in df_trials.columns:
                self.logger.warning("DA-EBC: No 'isi' column found in trial data")
                return {'error': 'No ISI data available'}
            
            # Apply same filters as RT analysis for consistency
            df_filtered = df_trials.copy()
            if 'lick' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['lick'] != 0]
            if 'naive' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['naive'] == 0]
            if 'MoveCorrectSpout' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['MoveCorrectSpout'] == 0]
            
            # Filter valid ISI data after applying standard filters
            valid_isi_mask = df_filtered['isi'].notna()
            df_filtered = df_filtered[valid_isi_mask]
            
            if df_filtered.empty:
                return {'error': 'No valid ISI data after filtering'}
            
            # Group by trial side and optionally opto
            pdf_dict = {}
            
            # Check if we have opto data
            has_opto = 'is_opto' in df_filtered.columns and df_filtered['is_opto'].any()
            
            if has_opto:
                # Group by trial side and opto condition
                for side in ['left', 'right']:
                    for opto in [0, 1]:
                        # Filter trials
                        side_val = 0 if side == 'left' else 1
                        mask = (df_filtered['is_right'] == side_val) & (df_filtered['is_opto'] == opto)
                        isi_data = df_filtered.loc[mask, 'isi'].dropna()
                        
                        if len(isi_data) > 5:  # Need minimum data for KDE
                            pdf_dict[(side, opto)] = self._compute_single_isi_pdf(isi_data)
            else:
                # Group by trial side only
                for side in ['left', 'right']:
                    side_val = 0 if side == 'left' else 1
                    mask = (df_filtered['is_right'] == side_val)
                    isi_data = df_filtered.loc[mask, 'isi'].dropna()
                    
                    if len(isi_data) > 5:  # Need minimum data for KDE
                        pdf_dict[side] = self._compute_single_isi_pdf(isi_data)

            results = {
                'pdf_data': pdf_dict,
                'has_opto': has_opto,
                'total_trials': len(df_trials),
                'filtered_trials': len(df_filtered),  # Add this for debugging
                'valid_isi_trials': len(df_filtered),  # This should match filtered_trials now
                'isi_range': {
                    'min': float(df_filtered['isi'].min()),
                    'max': float(df_filtered['isi'].max()),
                    'mean': float(df_filtered['isi'].mean()),
                    'std': float(df_filtered['isi'].std())
                    }
            }
            return results

            
        except Exception as e:
            self.logger.error(f"DA-EBC: Failed to compute ISI PDF: {e}")
            return {'error': f'ISI computation failed: {str(e)}'}

    def _compute_single_isi_pdf(self, isi_data: pd.Series) -> Dict[str, Any]:
        """
        Compute PDF for a single group of ISI data.
        """
        from scipy.stats import gaussian_kde
        import numpy as np
        
        try:
            # Remove any remaining NaN values
            clean_data = isi_data.dropna()
            
            if len(clean_data) < 2:
                return {'error': 'Insufficient data for PDF computation'}
            
            # Check if all values are the same (no variance)
            if clean_data.nunique() == 1:
                # All ISI values are identical - return discrete distribution
                single_value = clean_data.iloc[0]
                return {
                    'x': [single_value - 1, single_value, single_value + 1],  # Create minimal range around value
                    'y': [0.0, 1.0, 0.0],  # Peak at the single value
                    'mean': float(single_value),
                    'std': 0.0,
                    'count': len(clean_data),
                    'kde_bandwidth': None,
                    'distribution_type': 'discrete'
                }
            
            # Check for very low variance that might cause KDE issues
            data_std = clean_data.std()
            if data_std < 1e-6:  # Essentially zero variance
                # Values are very close to each other - treat as discrete
                mean_value = clean_data.mean()
                return {
                    'x': [mean_value - 1, mean_value, mean_value + 1],
                    'y': [0.0, 1.0, 0.0],
                    'mean': float(mean_value),
                    'std': float(data_std),
                    'count': len(clean_data),
                    'kde_bandwidth': None,
                    'distribution_type': 'low_variance'
                }
            
            # Compute KDE for data with sufficient variance
            kde = gaussian_kde(clean_data)
            
            # Create x range for PDF
            data_min, data_max = clean_data.min(), clean_data.max()
            data_range = data_max - data_min
            x_min = data_min - 0.1 * data_range
            x_max = data_max + 0.1 * data_range
            
            x = np.linspace(x_min, x_max, 100)
            y = kde(x)
            
            return {
                'x': x.tolist(),
                'y': y.tolist(),
                'mean': float(clean_data.mean()),
                'std': float(clean_data.std()),
                'count': len(clean_data),
                'kde_bandwidth': float(kde.factor),
                'distribution_type': 'continuous'
            }
            
        except Exception as e:
            return {'error': f'PDF computation failed: {str(e)}'}

    def _analyze_group(self) -> Dict[str, Any]:
        """Perform group-level analysis across all subjects."""
        self.logger.info("DA-EBC: Performing group analysis...")
        
        # Basic group analysis
        group_results = {
            'total_subjects': len(self.subject_list),
            'summary': 'Basic group analysis completed'
        }
        
        # TODO: Add group-level analysis (comparison across subjects, etc.)
        
        return group_results
    
    def _compute_psychometric_analysis(self, df_trials: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute psychometric curves for the session.
        Adapted from standalone psychometric code.
        """
        try:
            # Check if required columns exist
            if 'isi' not in df_trials.columns or 'mouse_choice' not in df_trials.columns:
                self.logger.warning(f"AN-SID: Missing essential columns for psychometric analysis")
                return {'error': 'Missing essential columns (isi, mouse_choice)'}
            
            # Apply same filters as other analysis functions for consistency
            df_filtered = df_trials.copy()
            if 'lick' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['lick'] != 0]
            if 'naive' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['naive'] == 0]
            if 'MoveCorrectSpout' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['MoveCorrectSpout'] == 0]
            
            if df_filtered.empty:
                return {'error': 'No trials remaining after filtering'}
            
            # Check if we have opto data
            has_opto = 'is_opto' in df_filtered.columns and df_filtered['is_opto'].any()
            
            psychometric_results = {}
            
            if has_opto:
                # Analyze with opto conditions
                for opto_condition in [0, 1]:
                    condition_name = 'control' if opto_condition == 0 else 'opto'
                    condition_df = df_filtered[df_filtered['is_opto'] == opto_condition]
                    
                    if len(condition_df) > 3:  # Reduced minimum from 5 to 3
                        psychometric_data = self._compute_psychometric(
                            condition_df, 
                            condition_col=None  # Don't split further since we already filtered
                        )
                        
                        if not psychometric_data.empty:
                            # Check if we should attempt fitting
                            unique_isi_count = len(np.unique(condition_df['isi']))
                            if unique_isi_count >= 3:
                                fit_results = self._compute_psychometric_fit(psychometric_data)
                            else:
                                # Skip fitting for sessions with too few ISI values
                                fit_results = {
                                    'params': (np.nan, np.nan),
                                    'threshold': np.nan,
                                    'slope': np.nan,
                                    'fit_x': [],
                                    'fit_y': [],
                                    'fit_method': 'skipped_insufficient_isi',
                                    'fit_quality': 'not_applicable'
                                }
                                self.logger.info(f"AN-SID: Skipped fitting for {condition_name} condition - only {unique_isi_count} unique ISI values")
                            
                            psychometric_results[condition_name] = {
                                'data': psychometric_data.to_dict('records'),
                                'fit': fit_results,
                                'n_trials': len(condition_df),
                                'unique_isi_count': unique_isi_count
                            }
            else:
                # Analyze without opto conditions
                psychometric_data = self._compute_psychometric(df_filtered, condition_col=None)
                
                if not psychometric_data.empty:
                    # Check if we should attempt fitting
                    unique_isi_count = len(np.unique(df_filtered['isi']))
                    if unique_isi_count >= 3:
                        fit_results = self._compute_psychometric_fit(psychometric_data)
                    else:
                        # Skip fitting for sessions with too few ISI values
                        fit_results = {
                            'params': (np.nan, np.nan),
                            'threshold': np.nan,
                            'slope': np.nan,
                            'fit_x': [],
                            'fit_y': [],
                            'fit_method': 'skipped_insufficient_isi',
                            'fit_quality': 'not_applicable'
                        }
                        self.logger.info(f"AN-SID: Skipped fitting for session - only {unique_isi_count} unique ISI values")
                    
                    psychometric_results['all'] = {
                        'data': psychometric_data.to_dict('records'),
                        'fit': fit_results,
                        'n_trials': len(df_filtered),
                        'unique_isi_count': unique_isi_count
                    }
            
            return {
                'psychometric_curves': psychometric_results,
                'has_opto': has_opto,
                'total_trials': len(df_trials),
                'filtered_trials': len(df_filtered)
            }
            
        except Exception as e:
            self.logger.error(f"AN-SID: Failed to compute psychometric analysis: {e}")
            return {'error': f'Psychometric computation failed: {str(e)}'}

    def _compute_psychometric(self, df, stim_col='isi', choice_col='mouse_choice', 
                             condition_col='is_opto', side_of_interest='right',
                             binning='adaptive', bins=40, round_stim=2, dropna=True):
        """
        Compute psychometric data: P(right) vs stimulus value, optionally split by condition.
        Uses adaptive binning strategy based on data characteristics.
        """
        df = df.copy()
        
        # Note: Filtering is now done in the calling function for consistency
        # No additional filtering here since df is already filtered
        
        # Prepare choice indicator
        df['is_right_choice'] = (df[choice_col] == side_of_interest).astype(int)

        # Adaptive binning strategy
        raw_values = df[stim_col].values
        unique_values = np.unique(raw_values)
        n_unique = len(unique_values)
        
        self.logger.info(f"AN-SID: Psychometric binning - {n_unique} unique ISI values found")
        
        # Determine binning strategy based on data characteristics
        if n_unique == 1:
            # Only one ISI value - can't do psychometric analysis
            self.logger.warning(f"AN-SID: Only 1 unique ISI value - insufficient for psychometric analysis")
            return pd.DataFrame()  # Return empty DataFrame
            
        elif n_unique == 2:
            # Two ISI values - use discrete binning, can generate data but can't fit sigmoid
            self.logger.info("AN-SID: Using discrete binning for 2-ISI session")
            df['stim_value'] = df[stim_col]
            binning_method = 'discrete_2isi'
            
        elif n_unique <= 5:
            # Use discrete values as bins - no binning needed
            self.logger.info("AN-SID: Using discrete binning strategy")
            df['stim_value'] = df[stim_col]
            binning_method = 'discrete'
            
        elif n_unique <= 10:
            # Medium diversity - use discrete with slight grouping if needed
            self.logger.info("AN-SID: Using discrete-grouped binning strategy")
            df['stim_value'] = df[stim_col].round(round_stim)
            binning_method = 'discrete_grouped'
            
        else:
            # High diversity - use intelligent quantile binning
            self.logger.info("AN-SID: Using intelligent quantile binning strategy")
            binning_method = 'quantile'
            
            # Calculate optimal number of bins
            optimal_bins = min(bins, max(8, n_unique // 3))
            quantiles = np.linspace(0, 1, optimal_bins)
            
            # Use quantiles, but ensure unique bin edges
            bin_edges = np.unique(np.quantile(raw_values, quantiles))
            
            # Safety check for insufficient bin edges
            if len(bin_edges) < 3:
                self.logger.warning("AN-SID: Quantile binning produced insufficient bins, falling back to discrete")
                df['stim_value'] = df[stim_col]
                binning_method = 'discrete_fallback'
            else:
                df['stim_bin'] = pd.cut(df[stim_col], bins=bin_edges, include_lowest=True)
                df['stim_value'] = df['stim_bin'].apply(lambda b: b.mid if pd.notnull(b) else np.nan)

        # Grouping
        group_cols = ['stim_value']
        if condition_col and condition_col in df.columns:
            group_cols.append(condition_col)

        grouped = df.groupby(group_cols, observed=True)
        isi_mean = df[stim_col].mean()

        # Compute summary with minimum trials per bin requirement
        min_trials_per_bin = 3  # Minimum for reliable statistics
        results = []
        
        for group_vals, subdf in grouped:
            if condition_col and condition_col in df.columns and not isinstance(group_vals, tuple):
                raise ValueError("Expected tuple (stim, condition) but got single value.")        
            
            if condition_col and condition_col in df.columns:
                stim_val, condition = group_vals
            else:
                stim_val = group_vals
                condition = 'all'

            n = len(subdf)
            
            # Skip bins with too few trials
            if n < min_trials_per_bin:
                self.logger.debug(f"AN-SID: Skipping bin with {n} trials (< {min_trials_per_bin} minimum)")
                continue
                
            p = subdf['is_right_choice'].mean()
            stderr = np.sqrt(p * (1 - p) / n) if n > 0 else np.nan

            if isinstance(stim_val, tuple) and len(stim_val) == 1:
                stim_val = stim_val[0]

            results.append({
                'stim_value': stim_val,
                'p_right': p,
                'stderr': stderr,
                'n_trials': n,
                'condition': condition if condition_col and condition_col in df.columns else 'all',
                'isi_mean': isi_mean,
                'binning_method': binning_method
            })

        result_df = pd.DataFrame(results)
        
        if result_df.empty:
            self.logger.warning("AN-SID: No bins met minimum trial requirements")
        else:
            self.logger.info(f"AN-SID: Generated {len(result_df)} psychometric bins using {binning_method} method")
            
        return result_df

    def _compute_psychometric_fit(self, psychometric_df, x_col='stim_value', y_col='p_right',
                                 n_points=200, clip_eps=1e-3, fallback=True, verbose=False, 
                                 extend_fit_x=False, fit_margin=0.8):
        """
        Fit a logistic sigmoid to psychometric data.
        Adapted from standalone code.
        """
        from scipy.optimize import curve_fit, OptimizeWarning
        from scipy.special import expit
        import warnings
        
        def logistic(x, beta0, beta1):
            return expit(beta0 + beta1 * x)
        
        # Center x
        x_raw = psychometric_df[x_col].values
        x_mean = np.mean(x_raw)
        x_centered = x_raw - x_mean
        x = x_centered
        
        # Clip y
        y = np.clip(psychometric_df[y_col].values, clip_eps, 1 - clip_eps)
        
        # Good initial guess
        beta1_init = 10 / (x_centered.max() - x_centered.min()) if x_centered.max() != x_centered.min() else 1.0
        beta0_init = 0
        p0 = [beta0_init, beta1_init]    
        
        bounds = ([-np.inf, -np.inf], [np.inf, np.inf])

        if len(np.unique(x)) < 2:
            return self._psychometric_fit_failed_output(x_centered + x_mean)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=OptimizeWarning)
                popt, _ = curve_fit(logistic, x_centered, y, p0=p0, bounds=bounds, maxfev=5000)
            
            beta0, beta1 = popt
            slope = beta1
            threshold = -beta0 / beta1 + x_mean
            
            x_min, x_max = x_centered.min(), x_centered.max()
            x_range = x_max - x_min
            
            if extend_fit_x:
                fit_x_min = x_min - fit_margin * x_range
                fit_x_max = x_max + fit_margin * x_range
            else:
                fit_x_min = x_min
                fit_x_max = x_max        
            
            fit_x_centered = np.linspace(fit_x_min, fit_x_max, n_points)
            fit_y = logistic(fit_x_centered, beta0, beta1)
            fit_x = fit_x_centered + x_mean

            # Check fit stability
            if np.ptp(fit_y) < 0.05:
                raise RuntimeError("Fit output too flat to be useful")

            return {
                'params': (float(beta0), float(beta1)),
                'threshold': float(threshold),
                'slope': float(slope),
                'fit_x': fit_x.tolist(),
                'fit_y': fit_y.tolist(),
                'fit_method': 'logistic',
                'fit_quality': 'good'
            }

        except Exception as e:
            if fallback and len(x) == 2:
                # Fallback to 2-point interpolated sigmoid
                x1, x2 = x
                y1, y2 = y
                try:
                    threshold = np.interp(0.5, [y1, y2], [x1, x2]) + x_mean
                    slope = (y2 - y1) / (x2 - x1)
                    beta1 = max(min(slope * 10, 20), -20)
                    beta0 = -beta1 * (threshold - x_mean)
                    
                    fit_x_centered = np.linspace(x_centered.min(), x_centered.max(), n_points)
                    fit_y = logistic(fit_x_centered, beta0, beta1)
                    fit_x = fit_x_centered + x_mean
                    
                    return {
                        'params': (float(beta0), float(beta1)),
                        'threshold': float(threshold),
                        'slope': float(beta1),
                        'fit_x': fit_x.tolist(),
                        'fit_y': fit_y.tolist(),
                        'fit_method': 'linear2pt',
                        'fit_quality': 'unstable'
                    }
                except Exception:
                    return self._psychometric_fit_failed_output(x_centered + x_mean)
            else:
                return self._psychometric_fit_failed_output(x_centered + x_mean)

    def _psychometric_fit_failed_output(self, fit_x):
        """Helper method for failed psychometric fits."""
        return {
            'params': (np.nan, np.nan),
            'threshold': np.nan,
            'slope': np.nan,
            'fit_x': fit_x.tolist(),
            'fit_y': np.full_like(fit_x, np.nan).tolist(),
            'fit_method': 'failed',
            'fit_quality': 'failed'
        }

    def _compute_rt_analysis(self, df_trials: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute response time statistics grouped by trial side and opto condition.
        Returns data suitable for box plots and statistical analysis.
        """
        try:
            # Check if RT column exists
            if 'RT' not in df_trials.columns:
                self.logger.warning("AN-SID: No 'RT' column found in trial data")
                return {'error': 'No RT data available'}
            
            # Apply filters from standalone code
            df_filtered = df_trials.copy()
            if 'lick' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['lick'] != 0]
            if 'naive' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['naive'] == 0]
            if 'MoveCorrectSpout' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['MoveCorrectSpout'] == 0]
            
            # Filter valid RT data
            valid_rt_mask = df_filtered['RT'].notna() & (df_filtered['RT'] > 0)
            df_filtered = df_filtered[valid_rt_mask]
            
            if df_filtered.empty:
                return {'error': 'No valid RT data after filtering'}
            
            # Check if we have opto data
            has_opto = 'is_opto' in df_filtered.columns and df_filtered['is_opto'].any()
            
            rt_stats = {}
            
            if has_opto:
                # Group by trial side and opto condition
                for side in ['left', 'right']:
                    for opto in [0, 1]:
                        side_val = 0 if side == 'left' else 1
                        mask = (df_filtered['is_right'] == side_val) & (df_filtered['is_opto'] == opto)
                        rt_data = df_filtered.loc[mask, 'RT'].dropna()
                        
                        if len(rt_data) > 0:
                            condition_key = (side, opto)
                            rt_stats[condition_key] = self._compute_single_rt_stats(rt_data)
            else:
                # Group by trial side only
                for side in ['left', 'right']:
                    side_val = 0 if side == 'left' else 1
                    mask = (df_filtered['is_right'] == side_val)
                    rt_data = df_filtered.loc[mask, 'RT'].dropna()
                    
                    if len(rt_data) > 0:
                        rt_stats[side] = self._compute_single_rt_stats(rt_data)
            
            return {
                'rt_statistics': rt_stats,
                'has_opto': has_opto,
                'total_trials': len(df_trials),
                'valid_rt_trials': len(df_filtered),
                'global_rt_stats': self._compute_single_rt_stats(df_filtered['RT'].dropna())
            }
            
        except Exception as e:
            self.logger.error(f"AN-SID: Failed to compute RT analysis: {e}")
            return {'error': f'RT analysis failed: {str(e)}'}

    def _compute_single_rt_stats(self, rt_data: pd.Series) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for a single group of RT data.
        Returns data suitable for box plots and statistical tests.
        """
        try:
            # Remove any remaining NaN values
            clean_data = rt_data.dropna()
            
            if len(clean_data) == 0:
                return {'error': 'No valid RT data'}
            
            # Basic descriptive statistics
            stats = {
                'count': len(clean_data),
                'mean': float(clean_data.mean()),
                'median': float(clean_data.median()),
                'std': float(clean_data.std()),
                'min': float(clean_data.min()),
                'max': float(clean_data.max()),
                'range': float(clean_data.max() - clean_data.min())
            }
            
            # Percentiles for box plots
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            for p in percentiles:
                stats[f'p{p}'] = float(np.percentile(clean_data, p))
            
            # Quartiles and IQR for box plots
            q1 = stats['p25']
            q3 = stats['p75']
            stats['iqr'] = q3 - q1
            
            # Outlier detection (using 1.5 * IQR rule)
            iqr = stats['iqr']
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            
            outliers = clean_data[(clean_data < lower_fence) | (clean_data > upper_fence)]
            stats['outliers'] = outliers.tolist()
            stats['n_outliers'] = len(outliers)
            
            # Whisker positions for box plots (min/max within fences)
            stats['whisker_low'] = float(clean_data[clean_data >= lower_fence].min())
            stats['whisker_high'] = float(clean_data[clean_data <= upper_fence].max())
            
            # Raw data for plotting
            stats['raw_data'] = clean_data.tolist()
            
            # Additional statistics
            if len(clean_data) > 1:
                # Coefficient of variation
                stats['cv'] = stats['std'] / stats['mean'] if stats['mean'] > 0 else np.nan
                
                # Skewness and kurtosis
                from scipy.stats import skew, kurtosis
                stats['skewness'] = float(skew(clean_data))
                stats['kurtosis'] = float(kurtosis(clean_data))
            else:
                stats['cv'] = np.nan
                stats['skewness'] = np.nan
                stats['kurtosis'] = np.nan
            
            return stats
            
        except Exception as e:
            return {'error': f'RT statistics computation failed: {str(e)}'}
