




# %%


def analyze_teaching_signal_dynamics(dual_results: Dict[str, Any],
                                   data: Dict[str, Any],
                                   roi_list: List[int]) -> Dict[str, Any]:
    """
    Analyze potential teaching signals between task solution and motor prediction ROIs
    """
    
    intersection_rois = dual_results['intersection_analysis']['intersection_rois']
    motor_only = dual_results['intersection_analysis']['motor_only'] 
    task_only = dual_results['intersection_analysis']['task_only']
    
    print(f"\n=== TEACHING SIGNAL ANALYSIS ===")
    
    # Analyze confidence patterns on error trials
    error_trial_analysis = analyze_confidence_on_error_trials(
        dual_results, data, intersection_rois, motor_only, task_only
    )
    
    # Analyze temporal dynamics: Do task ROIs lead motor ROIs?
    temporal_precedence = analyze_task_to_motor_temporal_precedence(
        dual_results, data, task_only, motor_only
    )
    
    return {
        'error_trial_analysis': error_trial_analysis,
        'temporal_precedence': temporal_precedence,
        'teaching_hypothesis_supported': evaluate_teaching_hypothesis(
            error_trial_analysis, temporal_precedence
        )
    }


# %%


# STEP X PRED -- Check specific rois?
# 1. Examine the top predictive ROIs in detail
top_predictive_rois = [554, 617, 86, 1051, 488, 258, 426, 289, 669, 155]

top_predictive_rois = [315,152,2015,640,175,11,150,215,88,67]  # 6-18


top_predictive_rois = [315]

# # Visualize these ROIs' activity patterns
# for roi_idx in top_predictive_rois[:5]:  # Show top 5
#     visualize_roi_individual_trials(
#         data, 
#         roi_idx=roi_idx, 
#         align_event='choice_start',
#         pre_event_s=3.5,
#         post_event_s=0.5,
#         max_trials_per_figure=20
#     )

# 2. Check if these ROIs belong to specific functional clusters
predictive_roi_clusters = []
for roi_idx in top_predictive_rois:
    cluster_id = data['df_rois'].iloc[roi_idx]['cluster_idx']
    predictive_roi_clusters.append(cluster_id)
    print(f"ROI {roi_idx} belongs to cluster {cluster_id}")

# 3. Analyze the temporal evolution more finely
temporal_analysis = analyze_prediction_temporal_dynamics(matched_data)
visualize_temporal_dynamics(temporal_analysis)









# %%

# NICE OUtcome/Side Comparison
# per roi or list traces 


align_event_list = {
    'start_flash_1_self': ('start_flash_1', 'start_flash_1', 1.0, 8.0),
    'end_f1_self': ('end_flash_1', 'end_flash_1', 1.0, 8.0),
    'start_flash_2_self_aligned': ('start_flash_2', 'start_flash_2', 4.0, 5.0),
    'end_flash_2_self_aligned': ('end_flash_2', 'end_flash_2', 4.0, 5.0),
    'choice_self_aligned': ('choice_start', 'choice_start', 5.0, 5.0),
    'lick_self_aligned': ('lick_start', 'lick_start', 5.0, 5.0),  
    # 'choice_sorted_f1_aligned': ('start_flash_1', 'choice_start', 1.0, 8.0),   
    # 'f1_sorted_choice_aligned': ('choice_start', 'start_flash_1', 4.0, 5.0),   
    # 'choice_sorted_lick_aligned': ('lick_start', 'choice_start', 4.0, 5.0),
}



# Use your top predictive ROIs
top_predictive_rois = [315,152,2015,640,175,11,150,215,88,67] # pred 6-18


top_predictive_rois = [315]
top_predictive_rois = [152]
top_predictive_rois = [2015]
top_predictive_rois = [640]
top_predictive_rois = [175]
top_predictive_rois = [11]
top_predictive_rois = [150]
top_predictive_rois = [215]
top_predictive_rois = [88]
top_predictive_rois = [67]


# strong_short_rois
# strong_long_rois
# shared_predictors



# top_predictive_rois = strong_short_rois
# top_predictive_rois = strong_long_rois
# top_predictive_rois = shared_predictors


# Run the analysis
visualize_isi_conditions_for_align_events(
    data=data,
    roi_list=top_predictive_rois,
    align_event_list=align_event_list,
    zscore=False
)





# %%







# %%

# STEP T - Verify accuracy-predictive roi's
# Needs work, looks like gets good pred numbers
# gates too strict?


# Use your top predictive ROIs
top_predictive_rois = [315,152,2015,640,175,11,150,215,88,67] # pred 6-18


# top_predictive_rois = [315]
# top_predictive_rois = [152]
# top_predictive_rois = [2015]
# top_predictive_rois = [640]
# top_predictive_rois = [175]
# top_predictive_rois = [11]
# top_predictive_rois = [150]
# top_predictive_rois = [215]
# top_predictive_rois = [88]
# top_predictive_roois = [67]

roi_list = top_predictive_rois



run_complete_roi_validation(data,roi_list)




# %%
# STEP N - Trail start choice verif
# note sure, maybe good pred nums?
# CAn get short/long rois, need to add results grabba

# Use your top predictive ROIs
top_predictive_rois = [315,152,2015,640,175,11,150,215,88,67] # pred 6-18

# top_predictive_rois = [315]
# top_predictive_rois = [152]
# top_predictive_rois = [2015]
# top_predictive_rois = [640]
# top_predictive_rois = [175]
# top_predictive_rois = [11]
# top_predictive_rois = [150]
# top_predictive_rois = [215]
# top_predictive_rois = [88]
# top_predictive_roois = [67]

roi_list = top_predictive_rois

# Run verification
results = comprehensive_trial_start_choice_verification(
    data,
    roi_indices=roi_list,
    margin_pre_choice_s=0.800,  # 60ms conservative margin
    apply_f2_control=False,      # Apply F2-orthogonalization
    n_resamples=1              # 10 balanced resamples
)

verification_results = results['verification_results']
short_isi = verification_results['short_isi']
long_isi = verification_results['long_isi']

# TODO: Get top short/long rois


# %% 
# FULL WINDOW PREDICTIVE ROI VERIFICATION
# Rigorous FDR-corrrection pred verif
# unlikely to find significance unless enough balance of short/long rew/pun


# Use your top predictive ROIs
top_predictive_rois = [315,152,2015,640,175,11,150,215,88,67] # pred 6-18


# top_predictive_rois = [315]
# top_predictive_rois = [152]
# top_predictive_rois = [2015]
# top_predictive_rois = [640]
# top_predictive_rois = [175]
# top_predictive_rois = [11]
# top_predictive_rois = [150]
# top_predictive_rois = [215]
# top_predictive_rois = [88]
# top_predictive_roois = [67]

roi_list = top_predictive_rois

print("=== RUNNING FULL WINDOW ROI VERIFICATION ===")

# Use default parameters - can be made configurable
results = verify_predictive_rois_full_window_approach_complete(
    data=data,
    roi_list=roi_list,
    n_balance_repeats=11,
    n_permutations=100,
    fdr_alpha=0.05,
    min_trials_per_condition=10,
    f2_analysis_window_s=0.3
)

results_complete = results
if results_complete is not None:
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Significant ROIs after FDR correction: {results_complete['verification_summary']['n_significant_rois']}")
    print(f"Median test AUROC: {results_complete['verification_summary']['median_auroc']:.3f}")
    
    # Show individual ROI results
    if 'significant_roi_indices' in results_complete['roi_results']:
        sig_rois = results_complete['roi_results']['significant_roi_indices']
        if len(sig_rois) > 0:
            print(f"\nSignificant ROI indices: {sig_rois}")
            
            # Show detailed results for significant ROIs
            for roi_idx in sig_rois:
                roi_result = results_complete['roi_results']['roi_results'][roi_idx]
                print(f"ROI {roi_idx}: AUROC={roi_result['mean_test_auroc']:.3f} ± {roi_result['std_test_auroc']:.3f}, p={roi_result['mean_p_value']:.4f}")


# Run the analysis on your results
analysis_results = analyze_verification_results(results_complete)

# Visualize the performance
visualize_verification_performance(analysis_results)

# Identify promising ROIs
promising_roi_list = identify_promising_rois(analysis_results, 
                                           auroc_threshold=0.55, 
                                           consistency_threshold=0.3)

print(f"\nPromising ROI indices: {promising_roi_list}")


# Run the interpretation
context = interpret_verification_context(results_complete)


# %%

# STEP STACK?
# stack of pred analysis, check, not sure



# # Use your top predictive ROIs
# top_predictive_rois = [315,152,2015,640,175,11,150,215,88,67] # pred 6-18


# # top_predictive_rois = [315]
# # top_predictive_rois = [152]
# # top_predictive_rois = [2015]
# # top_predictive_rois = [640]
# # top_predictive_rois = [175]
# # top_predictive_rois = [11]
# # top_predictive_rois = [150]
# # top_predictive_rois = [215]
# # top_predictive_rois = [88]
# # top_predictive_roois = [67]

# roi_list = top_predictive_rois


# Run the modified pipeline for small ROI sets
cluster_prediction_results_small = run_comprehensive_cluster_prediction_pipeline_small_roi_set(
    data, 
    roi_indices=top_predictive_rois,  # Your 10 ROIs
    n_bases=10,
    n_repeats=1,
    target_conditions=['short', 'long']
)


# %%


# MAP PIPELINE RESULTS TO ORIGINAL ROIS

# Usage with your results:
original_roi_list = [315,152,2015,640,175,11,150,215,88,67]  # 6-18


# Top 5 SHORT ISI predictive ROIs (original indices): [640, 175, 67, 11, 215]
# Top 5 LONG ISI predictive ROIs (original indices): [11, 215, 175, 150, 67]

original_roi_list = roi_list


# Map the pipeline results back to original ROI indices
mapped_results = map_pipeline_results_to_original_rois(
    cluster_prediction_results_small, 
    original_roi_list
)

# Identify top predictive ROIs using original indices
top_predictive_rois_short = identify_top_predictive_original_rois(
    mapped_results, 
    condition='short', 
    top_n=5
)

top_predictive_rois_long = identify_top_predictive_original_rois(
    mapped_results, 
    condition='long', 
    top_n=5
)

print(f"\nTop 5 SHORT ISI predictive ROIs (original indices): {top_predictive_rois_short}")
print(f"Top 5 LONG ISI predictive ROIs (original indices): {top_predictive_rois_long}")




# Get comprehensive predictor analysis
comprehensive_analysis = get_comprehensive_predictor_analysis(
    mapped_results, 
    auroc_thresholds={'short': 0.55, 'long': 0.60},  # Adjust thresholds as needed
    stability_threshold=0.10
)

# Create convenient ROI lists
predictor_roi_lists = create_predictor_roi_lists(comprehensive_analysis)

# Access the different predictor groups
print(f"\nROI Lists Available:")
for list_name, roi_list in predictor_roi_lists.items():
    print(f"  {list_name}: {len(roi_list)} ROIs")
    if len(roi_list) <= 10:
        print(f"    ROIs: {roi_list}")
    else:
        print(f"    ROIs: {roi_list[:10]}... (showing first 10)")

# Use specific predictor lists for further analysis
strong_short_rois = predictor_roi_lists.get('strong_short_predictors', [])
strong_long_rois = predictor_roi_lists.get('strong_long_predictors', [])
shared_predictors = predictor_roi_lists.get('shared_strong_predictors', [])

# Example: Visualize the shared strong predictors
if len(shared_predictors) > 0:
    print(f"\n✅ Found {len(shared_predictors)} ROIs that strongly predict both conditions!")
    # You could now run your visualization functions on these ROIs



# %%

# MAP Predictive ROIs to clusters
# Run the analysis
top_predictive_rois = [315,152,2015,640,175,11,150,215,88,67]  # 6-18 # Your predictive ROIs

cluster_analysis = analyze_top_predictive_roi_clusters(data, top_predictive_rois)

# Visualize the results
if cluster_analysis is not None:
    visualize_predictive_roi_cluster_distribution(cluster_analysis)
    compare_with_existing_cluster_selections(cluster_analysis)




# %%


# Visualize ROI reward/punishment patterns

# 1. Examine the top predictive ROIs in detail
top_predictive_rois = [554, 617, 86, 1051, 488, 258, 426, 289, 669, 155]

top_predictive_rois = [315,152,2015,640,175,11,150,215,88,67]   # 6-18
analyze_roi_reward_punishment_comprehensive(data, roi_list=top_predictive_rois)




# %%




def analyze_predictive_trial_patterns(matched_data: Dict[str, Any],
                                    prediction_results: Dict[str, Any],
                                    data: Dict[str, Any],
                                    model_name: str = 'logistic') -> Dict[str, Any]:
    """
    Analyze which specific trials are most predictive and what patterns drive prediction
    
    FIXED VERSION: Works with ISI-matched prediction results structure
    """
    
    print("=== ANALYZING PREDICTIVE TRIAL PATTERNS ===")
    
    X = matched_data['X']  # (n_trials, n_timepoints, n_rois)
    y = matched_data['y']  # (n_trials,) choice labels
    pair_ids = matched_data['pair_ids']
    time_vector = matched_data['time_vector']
    
    # FIXED: Use the correct key structure from ISI-matched results
    if 'models' in prediction_results:
        model_results = prediction_results['models'][model_name]
        mean_auc = model_results['mean_auc']
    elif 'model_results' in prediction_results:
        # Your structure: prediction_results['model_results'][model_name]
        if model_name in prediction_results['model_results']:
            model_results = prediction_results['model_results'][model_name]
            mean_auc = model_results.get('auc', model_results.get('mean_auc', 0.5))
        else:
            print(f"Available models: {list(prediction_results['model_results'].keys())}")
            # Use the best model instead
            model_name = prediction_results['best_model']
            model_results = prediction_results['model_results'][model_name]
            mean_auc = prediction_results['best_accuracy']
            print(f"Using best model: {model_name} (AUC: {mean_auc:.3f})")
    else:
        print("❌ Cannot find model results in prediction_results")
        print(f"Available keys: {list(prediction_results.keys())}")
        return None
    
    print(f"Analyzing {model_name} model (AUC: {mean_auc:.3f})")
    
    # Re-run model to get trial-by-trial predictions using the same CV structure
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Use pair-stratified CV (same as in your original analysis)
    unique_pairs = np.unique(pair_ids)
    n_pairs = len(unique_pairs)
    n_folds = min(5, n_pairs // 2)
    
    if n_folds < 2:
        print("❌ Not enough pairs for cross-validation analysis")
        return None
    
    # Create pair-stratified splits (same logic as your original analysis)
    pair_labels = []
    for pair_id in unique_pairs:
        pair_mask = pair_ids == pair_id
        pair_choices = y[pair_mask]
        # Use the choice of the first trial in pair
        pair_labels.append(pair_choices[0])
    
    pair_labels = np.array(pair_labels)
    
    # Stratified CV on pairs
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store trial-level predictions and confidences
    trial_predictions = np.full(len(y), np.nan)
    trial_confidences = np.full(len(y), np.nan)
    
    # Flatten features for ML: (n_trials, n_timepoints * n_rois)
    X_flat = X.reshape(X.shape[0], -1)
    
    # Cross-validation to get unbiased predictions
    for fold_idx, (train_pairs, test_pairs) in enumerate(skf.split(unique_pairs, pair_labels)):
        # Convert pair indices to trial indices
        train_mask = np.isin(pair_ids, unique_pairs[train_pairs])
        test_mask = np.isin(pair_ids, unique_pairs[test_pairs])
        
        X_train, X_test = X_flat[train_mask], X_flat[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Apply feature selection if used in original analysis
        if 'feature_mask' in prediction_results:
            feature_mask = prediction_results['feature_mask']
            X_train = X_train[:, feature_mask]
            X_test = X_test[:, feature_mask]
        
        # Fit model (same as original)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_name == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == 'svm':
            from sklearn.svm import SVC
            model = SVC(probability=True, random_state=42)
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)
        
        model.fit(X_train_scaled, y_train)
        
        # Get predictions and confidence scores
        test_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of right choice
        test_pred = (test_proba > 0.5).astype(int)
        
        # Store results
        trial_predictions[test_mask] = test_pred
        trial_confidences[test_mask] = np.abs(test_proba - 0.5)  # Distance from 0.5
    
    # Remove any NaN predictions
    valid_mask = ~np.isnan(trial_predictions)
    trial_predictions = trial_predictions[valid_mask].astype(int)
    trial_confidences = trial_confidences[valid_mask]
    y_valid = y[valid_mask]
    X_valid = X[valid_mask]
    pair_ids_valid = pair_ids[valid_mask]
    
    print(f"Valid predictions: {len(trial_predictions)}/{len(y)}")
    
    # Analyze prediction accuracy
    correct_predictions = (trial_predictions == y_valid)
    accuracy = np.mean(correct_predictions)
    
    print(f"Overall accuracy: {accuracy:.3f}")
    
    # Categorize trials by prediction confidence
    confidence_threshold = np.median(trial_confidences)
    high_confidence = trial_confidences >= confidence_threshold
    low_confidence = trial_confidences < confidence_threshold
    
    print(f"High confidence trials: {np.sum(high_confidence)} (threshold: {confidence_threshold:.3f})")
    print(f"Low confidence trials: {np.sum(low_confidence)}")
    
    # Analyze patterns
    pattern_analysis = {}
    
    # 1. Accuracy by confidence level
    pattern_analysis['accuracy_by_confidence'] = {
        'high_confidence_accuracy': np.mean(correct_predictions[high_confidence]),
        'low_confidence_accuracy': np.mean(correct_predictions[low_confidence]),
        'high_conf_n': np.sum(high_confidence),
        'low_conf_n': np.sum(low_confidence)
    }
    
    # 2. Neural patterns by prediction outcome
    correctly_predicted = X_valid[correct_predictions]
    incorrectly_predicted = X_valid[~correct_predictions]
    
    pattern_analysis['neural_patterns'] = {
        'correct_mean': np.mean(correctly_predicted, axis=0),  # (n_timepoints, n_rois)
        'incorrect_mean': np.mean(incorrectly_predicted, axis=0),
        'difference': np.mean(correctly_predicted, axis=0) - np.mean(incorrectly_predicted, axis=0),
        'n_correct': len(correctly_predicted),
        'n_incorrect': len(incorrectly_predicted)
    }
    
    # 3. High vs low confidence neural patterns
    high_conf_patterns = X_valid[high_confidence]
    low_conf_patterns = X_valid[low_confidence]
    
    pattern_analysis['confidence_patterns'] = {
        'high_conf_mean': np.mean(high_conf_patterns, axis=0),
        'low_conf_mean': np.mean(low_conf_patterns, axis=0),
        'confidence_difference': np.mean(high_conf_patterns, axis=0) - np.mean(low_conf_patterns, axis=0)
    }
    
    # 4. Choice-specific patterns for correctly predicted trials
    correct_left_trials = X_valid[correct_predictions & (y_valid == 0)]
    correct_right_trials = X_valid[correct_predictions & (y_valid == 1)]
    
    pattern_analysis['choice_patterns'] = {
        'correct_left_mean': np.mean(correct_left_trials, axis=0) if len(correct_left_trials) > 0 else None,
        'correct_right_mean': np.mean(correct_right_trials, axis=0) if len(correct_right_trials) > 0 else None,
        'choice_difference': (np.mean(correct_right_trials, axis=0) - np.mean(correct_left_trials, axis=0)) 
                           if len(correct_left_trials) > 0 and len(correct_right_trials) > 0 else None,
        'n_correct_left': len(correct_left_trials),
        'n_correct_right': len(correct_right_trials)
    }
    
    # 5. Get trial metadata for further analysis
    trial_metadata = []
    for i, pair_id in enumerate(pair_ids_valid):
        trial_metadata.append({
            'trial_idx': i,
            'pair_id': pair_id,
            'true_choice': y_valid[i],
            'predicted_choice': trial_predictions[i],
            'confidence': trial_confidences[i],
            'correct': correct_predictions[i],
            'high_confidence': high_confidence[i]
        })
    
    return {
        'trial_predictions': trial_predictions,
        'trial_confidences': trial_confidences,
        'accuracy': accuracy,
        'pattern_analysis': pattern_analysis,
        'trial_metadata': trial_metadata,
        'time_vector': time_vector,
        'confidence_threshold': confidence_threshold,
        'model_name': model_name,
        'mean_auc': mean_auc,
        'analysis_complete': True
    }

# Update the comprehensive function to use the correct model name
def comprehensive_predictive_pattern_analysis(matched_data: Dict[str, Any],
                                            prediction_results: Dict[str, Any],
                                            data: Dict[str, Any]) -> Dict[str, Any]:
    """Run comprehensive analysis of what drives successful predictions"""
    
    print("=== COMPREHENSIVE PREDICTIVE PATTERN ANALYSIS ===")
    
    # Use the best model from your prediction results
    best_model = prediction_results.get('best_model', 'logistic')
    print(f"Using best model: {best_model}")
    
    # Analyze patterns
    pattern_results = analyze_predictive_trial_patterns(
        matched_data, prediction_results, data, model_name=best_model
    )
    
    if pattern_results is None:
        return None
    
    # Visualize patterns
    visualize_predictive_trial_patterns(pattern_results)
    
    # Identify specific trials of interest
    trial_categories = identify_most_predictive_trials(pattern_results, top_n=5)
    
    return {
        'pattern_results': pattern_results,
        'trial_categories': trial_categories,
        'analysis_complete': True
    }


def visualize_predictive_trial_patterns(pattern_results: Dict[str, Any]) -> None:
    """Visualize the predictive trial pattern analysis"""
    
    if pattern_results is None:
        print("❌ No pattern results to visualize")
        return
    
    pattern_analysis = pattern_results['pattern_analysis']
    time_vector = pattern_results['time_vector']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    # Row 1: Accuracy and confidence analysis
    
    # 1.1: Accuracy by confidence level
    ax = axes[0, 0]
    conf_data = pattern_analysis['accuracy_by_confidence']
    categories = ['High Confidence', 'Low Confidence']
    accuracies = [conf_data['high_confidence_accuracy'], conf_data['low_confidence_accuracy']]
    counts = [conf_data['high_conf_n'], conf_data['low_conf_n']]
    
    bars = ax.bar(categories, accuracies, color=['darkgreen', 'orange'], alpha=0.7)
    ax.set_ylabel('Prediction Accuracy')
    ax.set_title('Accuracy by Confidence Level')
    ax.set_ylim([0, 1])
    
    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'n={count}', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)
    
    # 1.2: Confidence distribution
    ax = axes[0, 1]
    confidences = pattern_results['trial_confidences']
    ax.hist(confidences, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(pattern_results['confidence_threshold'], color='red', linestyle='--', 
               label=f"Threshold: {pattern_results['confidence_threshold']:.3f}")
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Number of Trials')
    ax.set_title('Prediction Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.3: Overall accuracy summary
    ax = axes[0, 2]
    overall_acc = pattern_results['accuracy']
    ax.bar(['Overall'], [overall_acc], color='steelblue', alpha=0.7)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Overall Prediction Accuracy')
    ax.set_ylim([0, 1])
    ax.text(0, overall_acc + 0.02, f'{overall_acc:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Row 2: Neural pattern differences
    
    # 2.1: Correct vs Incorrect predictions
    ax = axes[1, 0]
    neural_patterns = pattern_analysis['neural_patterns']
    if neural_patterns['difference'] is not None:
        # Average across ROIs to show temporal profile
        temporal_diff = np.mean(neural_patterns['difference'], axis=1)
        ax.plot(time_vector, temporal_diff, 'b-', linewidth=2, 
                label=f"Correct - Incorrect (n_correct={neural_patterns['n_correct']})")
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Time from Choice (s)')
        ax.set_ylabel('Neural Activity Difference')
        ax.set_title('Correct vs Incorrect Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2.2: High vs Low confidence patterns
    ax = axes[1, 1]
    conf_patterns = pattern_analysis['confidence_patterns']
    if conf_patterns['confidence_difference'] is not None:
        temporal_conf_diff = np.mean(conf_patterns['confidence_difference'], axis=1)
        ax.plot(time_vector, temporal_conf_diff, 'g-', linewidth=2, 
                label="High - Low Confidence")
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Time from Choice (s)')
        ax.set_ylabel('Neural Activity Difference')
        ax.set_title('High vs Low Confidence Patterns')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2.3: Choice-specific patterns (for correctly predicted trials)
    ax = axes[1, 2]
    choice_patterns = pattern_analysis['choice_patterns']
    if choice_patterns['choice_difference'] is not None:
        temporal_choice_diff = np.mean(choice_patterns['choice_difference'], axis=1)
        ax.plot(time_vector, temporal_choice_diff, 'r-', linewidth=2, 
                label=f"Right - Left Choice\n(n_L={choice_patterns['n_correct_left']}, n_R={choice_patterns['n_correct_right']})")
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Time from Choice (s)')
        ax.set_ylabel('Neural Activity Difference')
        ax.set_title('Choice Patterns (Correct Predictions)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 3: Heatmaps of key patterns
    
    # 3.1: Correct - Incorrect heatmap
    ax = axes[2, 0]
    if neural_patterns['difference'] is not None:
        diff_pattern = neural_patterns['difference']  # (n_timepoints, n_rois)
        im = ax.imshow(diff_pattern.T, aspect='auto', cmap='RdBu_r',
                       extent=[time_vector[0], time_vector[-1], 0, diff_pattern.shape[1]],
                       vmin=np.percentile(diff_pattern, 5),
                       vmax=np.percentile(diff_pattern, 95))
        ax.set_xlabel('Time from Choice (s)')
        ax.set_ylabel('ROI Index')
        ax.set_title('Neural Difference: Correct - Incorrect')
        plt.colorbar(im, ax=ax, label='Activity Difference')
    
    # 3.2: High - Low confidence heatmap
    ax = axes[2, 1]
    if conf_patterns['confidence_difference'] is not None:
        conf_diff_pattern = conf_patterns['confidence_difference']
        im = ax.imshow(conf_diff_pattern.T, aspect='auto', cmap='RdBu_r',
                       extent=[time_vector[0], time_vector[-1], 0, conf_diff_pattern.shape[1]],
                       vmin=np.percentile(conf_diff_pattern, 5),
                       vmax=np.percentile(conf_diff_pattern, 95))
        ax.set_xlabel('Time from Choice (s)')
        ax.set_ylabel('ROI Index')
        ax.set_title('Neural Difference: High - Low Confidence')
        plt.colorbar(im, ax=ax, label='Activity Difference')
    
    # 3.3: Choice difference heatmap (correct predictions only)
    ax = axes[2, 2]
    if choice_patterns['choice_difference'] is not None:
        choice_diff_pattern = choice_patterns['choice_difference']
        im = ax.imshow(choice_diff_pattern.T, aspect='auto', cmap='RdBu_r',
                       extent=[time_vector[0], time_vector[-1], 0, choice_diff_pattern.shape[1]],
                       vmin=np.percentile(choice_diff_pattern, 5),
                       vmax=np.percentile(choice_diff_pattern, 95))
        ax.set_xlabel('Time from Choice (s)')
        ax.set_ylabel('ROI Index')
        ax.set_title('Neural Difference: Right - Left Choice\n(Correct Predictions Only)')
        plt.colorbar(im, ax=ax, label='Activity Difference')
    
    plt.suptitle('Predictive Trial Pattern Analysis: What Drives Successful Predictions?', fontsize=16)
    plt.tight_layout()
    plt.show()

def identify_most_predictive_trials(pattern_results: Dict[str, Any], 
                                   top_n: int = 10) -> Dict[str, Any]:
    """Identify the most and least predictive trials for detailed inspection"""
    
    trial_metadata = pattern_results['trial_metadata']
    
    # Sort trials by confidence
    sorted_trials = sorted(trial_metadata, key=lambda x: x['confidence'], reverse=True)
    
    # Get top predictive trials (high confidence + correct)
    most_predictive = [t for t in sorted_trials if t['correct'] and t['high_confidence']][:top_n]
    
    # Get least predictive trials (low confidence or incorrect)
    least_predictive = [t for t in sorted_trials if not t['correct'] or not t['high_confidence']][-top_n:]
    
    # Get trials where model was confident but wrong
    confident_wrong = [t for t in sorted_trials if not t['correct'] and t['high_confidence']][:top_n]
    
    print(f"\n=== MOST PREDICTIVE TRIALS (Top {len(most_predictive)}) ===")
    for i, trial in enumerate(most_predictive):
        print(f"  {i+1}. Pair {trial['pair_id']}: True={trial['true_choice']}, "
              f"Pred={trial['predicted_choice']}, Conf={trial['confidence']:.3f}")
    
    print(f"\n=== LEAST PREDICTIVE TRIALS (Bottom {len(least_predictive)}) ===")
    for i, trial in enumerate(least_predictive):
        print(f"  {i+1}. Pair {trial['pair_id']}: True={trial['true_choice']}, "
              f"Pred={trial['predicted_choice']}, Conf={trial['confidence']:.3f}, "
              f"Correct={trial['correct']}")
    
    if len(confident_wrong) > 0:
        print(f"\n=== CONFIDENT BUT WRONG TRIALS ({len(confident_wrong)}) ===")
        for i, trial in enumerate(confident_wrong):
            print(f"  {i+1}. Pair {trial['pair_id']}: True={trial['true_choice']}, "
                  f"Pred={trial['predicted_choice']}, Conf={trial['confidence']:.3f}")
    
    return {
        'most_predictive': most_predictive,
        'least_predictive': least_predictive,
        'confident_wrong': confident_wrong
    }

# # Usage function
# def comprehensive_predictive_pattern_analysis(matched_data: Dict[str, Any],
#                                             prediction_results: Dict[str, Any],
#                                             data: Dict[str, Any]) -> Dict[str, Any]:
#     """Run comprehensive analysis of what drives successful predictions"""
    
#     print("=== COMPREHENSIVE PREDICTIVE PATTERN ANALYSIS ===")
    
#     # Analyze patterns
#     pattern_results = analyze_predictive_trial_patterns(
#         matched_data, prediction_results, data, model_name='logistic'
#     )
    
#     if pattern_results is None:
#         return None
    
#     # Visualize patterns
#     visualize_predictive_trial_patterns(pattern_results)
    
#     # Identify specific trials of interest
#     trial_categories = identify_most_predictive_trials(pattern_results, top_n=5)
    
#     return {
#         'pattern_results': pattern_results,
#         'trial_categories': trial_categories,
#         'analysis_complete': True
#     }


# 1. Examine the top predictive ROIs in detail
top_predictive_rois = [554, 617, 86, 1051, 488, 258, 426, 289, 669, 155] # 6-20
top_predictive_rois = [315,152,2015,640,175,11,150,215,88,67]   #  6-18
# Run the comprehensive predictive pattern analysis
predictive_patterns = comprehensive_predictive_pattern_analysis(
    matched_data, 
    prediction_results, 
    data
)


