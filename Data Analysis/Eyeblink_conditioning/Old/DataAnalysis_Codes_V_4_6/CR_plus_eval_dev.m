function [CR_plus, CR_amp, baseline_amp] = CR_plus_eval_dev(time, signal, t_LED, t_puff, CR_threshold)
% CR_PLUS_EVAL Evaluates the presence of CR (Conditioned Response) based on the signal (FEC).
% Inputs:
%   - time: time vector
%   - signal: signal vector (FEC)
%   - t_LED: LED onset time
%   - t_puff: puff delivery time
%   - CR_threshold: threshold for CR detection
% Outputs:
%   - CR_plus: true if CR is detected, false otherwise
%   - CR_amp: CR window FEC average
%   - baseline_amp: baseline FEC average

    % Initialize outputs
    CR_plus = false;

    % Define the CR window onset (50 ms before puff)
    crWindowOnset = t_puff - 0.05; % ms
    crWindowEnd = t_puff; % puff time

    % Define the baseline window: [t_LED - 200, crWindowOnset]
    baselineStart = t_LED - 0.2; % 200 ms before LED onset
    baselineEnd = crWindowOnset;

    % Extract indices for baseline window and CR window
    baseline_indices = find(time >= baselineStart & time <= baselineEnd);
    cr_indices = find(time >= crWindowOnset & time <= crWindowEnd);

    % Extract signal for baseline and CR window
    baseline_signal = signal(baseline_indices);
    cr_signal = signal(cr_indices);

    % Determine baseline points: lowest 30th percentile of the baseline signal
    baseline_sorted = sort(baseline_signal);
    num_baseline_points = floor(0.3 * length(baseline_sorted)); % 30% of points
    lowest_baseline_points = baseline_sorted(1:num_baseline_points);

    % Calculate baseline amplitude and CR amplitude
    baseline_amp = mean(lowest_baseline_points);
    CR_amp = mean(cr_signal);

    % Compare CR amplitude to baseline + threshold
    if CR_amp > (baseline_amp + CR_threshold)
        CR_plus = true;
    end
end