function [CR_category, CR_amp, CR_amp_raw, baseline_amp] = CR_max_amplitude(time, signal, t_LED, t_puff, good_CR_threshold, poor_CR_threshold, no_CR_threshold)
% CR_MAX_AMPLITUDE Classifies conditioned response (CR) based on max amplitude in CR window
%
% Outputs:
%   - CR_category: 'Good CR', 'Poor CR', or 'No CR'
%   - CR_amp: Baseline-subtracted peak FEC amplitude in CR window
%   - CR_amp_raw: Raw (non-baseline-subtracted) peak FEC amplitude in CR window
%   - baseline_amp: Mean amplitude of the lowest 30% of signal in the baseline window
%
% Inputs:
%   - time: Time vector (same length as signal)
%   - signal: FEC signal vector for one trial
%   - t_LED: LED onset time
%   - t_puff: Airpuff onset time
%   - good_CR_threshold: Threshold for classifying a Good CR (above baseline)
%   - poor_CR_threshold: Threshold for classifying a Poor CR (above baseline)
%   - no_CR_threshold: Max amplitude tolerated for No CR (above baseline)

    % Default category
    CR_category = 'No CR';
    
    % Smooth the FEC signal
    smoothed_signal = smoothdata(signal, 'movmean', 5);

    % Define windows
    baseline_indices = find(time >= (t_LED - 0.2) & time <= t_LED);
    CR_window_indices = find(time >= t_LED & time <= t_puff);

    % Extract signals
    baseline_signal = smoothed_signal(baseline_indices);
    CR_window_signal = smoothed_signal(CR_window_indices);

    % Compute baseline (average of lowest 30%)
    sorted_baseline = sort(baseline_signal);
    N_low = floor(0.3 * length(sorted_baseline));
    baseline_amp = mean(sorted_baseline(1:N_low));

    % Compute raw and baseline-subtracted CR amplitude
    CR_amp_raw = max(CR_window_signal);
    CR_amp = CR_amp_raw - baseline_amp;

    % Classify based on baseline-subtracted CR amplitude
    if CR_amp >= good_CR_threshold
        CR_category = 'Good CR';
    elseif CR_amp >= poor_CR_threshold
        CR_category = 'Poor CR';
    elseif CR_amp < no_CR_threshold
        CR_category = 'No CR';
    else
        CR_category = 'Ambiguous';
    end
end