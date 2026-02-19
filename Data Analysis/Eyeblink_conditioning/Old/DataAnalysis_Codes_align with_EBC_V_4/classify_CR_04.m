function [CR_category, CR_amp, baseline_amp] = classify_CR_04(time, signal, t_LED, t_puff, good_CR_threshold, poor_CR_threshold)
% CLASSIFY_CR_02 Evaluates CR responses based on FEC signal.
%
% Inputs:
%   - time: Time vector
%   - signal: FEC signal vector
%   - t_LED: LED onset time
%   - t_puff: Puff delivery time
%   - good_CR_threshold: Threshold for classifying a good CR
%   - poor_CR_threshold: Threshold for classifying a poor CR
%
% Outputs:
%   - CR_category: Classification ('Good CR', 'Poor CR', or 'No CR')
%   - CR_amp: Average FEC during the last 50ms before puff
%   - baseline_amp: Baseline FEC average

    % Initialize output
    CR_category = 'No CR'; % Default state
    
    % Smooth the signal to reduce noise
    smoothed_signal = smoothdata(signal, 'movmean', 5); % Moving average

    % Define baseline window: 200ms before LED onset to LED onset
    baselineStart = t_LED - 0.2; % 200ms before LED onset
    baselineEnd = t_LED; 
    
    % Define Good CR window (Last 50ms before puff)
    goodCRStart = t_puff - 0.05;
    goodCREnd = t_puff;
    
    % Define Poor CR window (From LED onset to airpuff)
    poorCRStart = t_LED;
    poorCREnd = t_puff- 0.05;
    
    % Extract indices for baseline, Good CR, and Poor CR windows
    baseline_indices = find(time >= baselineStart & time <= baselineEnd);
    goodCR_indices = find(time >= goodCRStart & time <= goodCREnd);
    poorCR_indices = find(time >= poorCRStart & time <= poorCREnd);
    
    % Extract smoothed signals
    baseline_signal = smoothed_signal(baseline_indices);
    goodCR_signal = smoothed_signal(goodCR_indices);
    poorCR_signal = smoothed_signal(poorCR_indices);
    
    % Calculate baseline amplitude (lowest 30% of baseline values)
    baseline_sorted = sort(baseline_signal);
    num_baseline_points = floor(0.3 * length(baseline_sorted));
    lowest_baseline_points = baseline_sorted(1:num_baseline_points);
    baseline_amp = mean(lowest_baseline_points);
    
    % Calculate Good CR amplitude (average of last 50ms before puff)
    CR_amp = mean(goodCR_signal);
    
    % Check for spikes in the poor CR window
    poorCR_spikes = any(poorCR_signal > (baseline_amp + poor_CR_threshold));

    % Check if signal stays high until the end of airpuff
    stays_high = all(goodCR_signal > (baseline_amp + good_CR_threshold));
    
    % Determine CR classification
    if stays_high
        CR_category = 'Good CR'; % Signal remains high at the end of airpuff
    elseif poorCR_spikes
        CR_category = 'Poor CR'; % Signal spikes but drops before airpuff ends
    else 
        CR_category = 'No CR'; % No significant response
    end
end