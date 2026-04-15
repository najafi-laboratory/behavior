function [CR_category, CR_amp, baseline_amp] = classify_CR_06(time, signal, t_LED, t_puff, good_CR_threshold, poor_CR_threshold, no_CR_threshold)
% CLASSIFY_CR_06 Classifies CR as 'Good CR', 'Poor CR', or 'No CR' using thresholds
%
% Inputs:
%   - time: Time vector
%   - signal: FEC signal vector
%   - t_LED: LED onset time
%   - t_puff: Airpuff time
%   - good_CR_threshold: Threshold above baseline for Good CR
%   - poor_CR_threshold: Threshold above baseline for Poor CR
%   - no_CR_threshold: Max threshold (above baseline) allowed for No CR
%
% Outputs:
%   - CR_category: One of 'Good CR', 'Poor CR', 'No CR'
%   - CR_amp: Mean FEC 50 ms before puff
%   - baseline_amp: Baseline FEC amplitude

    % Default classification
    CR_category = 'No CR';
    
    % Smooth signal
    smoothed_signal = smoothdata(signal, 'movmean', 5);

    % Define windows
    baseline_indices = find(time >= (t_LED - 0.2) & time <= t_LED);
    goodCR_indices   = find(time >= (t_puff - 0.05) & time <= t_puff);
    poorCR_indices   = find(time >= t_LED & time <= (t_puff - 0.05));

    % Signal segments
    baseline_signal = smoothed_signal(baseline_indices);
    goodCR_signal   = smoothed_signal(goodCR_indices);
    poorCR_signal   = smoothed_signal(poorCR_indices);

    % Baseline amplitude (lowest 30%)
    sorted_baseline = sort(baseline_signal);
    N_low = floor(0.3 * length(sorted_baseline));
    baseline_amp = mean(sorted_baseline(1:N_low));

    % CR amplitude
    CR_amp = mean(goodCR_signal);

    % Proportions above thresholds
    p_good = mean(goodCR_signal > (baseline_amp + good_CR_threshold));
    p_poor = mean(poorCR_signal > (baseline_amp + poor_CR_threshold));
    p_no   = mean(goodCR_signal < (baseline_amp + no_CR_threshold)) & ...
             mean(poorCR_signal < (baseline_amp + no_CR_threshold));

    % Classification decision
    if p_good >= 0.5
        CR_category = 'Good CR';
    elseif p_poor >= 0.5
        CR_category = 'Poor CR';
    elseif p_no
        CR_category = 'No CR';
    else
        % Optional fallback: unclear CR response
        CR_category = 'Ambiguous';
    end
end