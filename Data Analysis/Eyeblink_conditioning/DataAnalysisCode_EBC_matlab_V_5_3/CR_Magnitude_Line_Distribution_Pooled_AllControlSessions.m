clc; clear; close all;

data_files = dir('*_EBC_*.mat');
% data_files = dir('E5LG_EBC_V_4_1_20250619_192957.mat');


good_CR_threshold = 0.1;
poor_CR_threshold = 0.05;
no_CR_threshold = 0.05;

% Good CR
CR_Good_short_bs = [];   % Baseline-subtracted amplitude for Good CR in short block
CR_Good_short_raw = [];  % Raw amplitude for Good CR in short block
CR_Good_long_bs = [];    
CR_Good_long_raw = [];

% Poor CR
CR_Poor_short_bs = [];
CR_Poor_short_raw = [];
CR_Poor_long_bs = [];
CR_Poor_long_raw = [];

% No CR
CR_No_short_bs = [];
CR_No_short_raw = [];
CR_No_long_bs = [];
CR_No_long_raw = [];


% Initialize SD group
SD = struct( ...
    'CR_Good_short_bs', [], 'CR_Good_short_raw', [], ...
    'CR_Poor_short_bs', [], 'CR_Poor_short_raw', [], ...
    'CR_No_short_bs', [],   'CR_No_short_raw', [], ...
    'CR_Good_long_bs',  [], 'CR_Good_long_raw', [], ...
    'CR_Poor_long_bs',  [], 'CR_Poor_long_raw', [], ...
    'CR_No_long_bs',    [], 'CR_No_long_raw', [] ...
);

% Initialize Control group
Control = struct( ...
    'CR_Good_short_bs', [], 'CR_Good_short_raw', [], ...
    'CR_Poor_short_bs', [], 'CR_Poor_short_raw', [], ...
    'CR_No_short_bs', [],   'CR_No_short_raw', [], ...
    'CR_Good_long_bs',  [], 'CR_Good_long_raw', [], ...
    'CR_Poor_long_bs',  [], 'CR_Poor_long_raw', [], ...
    'CR_No_long_bs',    [], 'CR_No_long_raw', [] ...
);



for i = 1:length(data_files)
    load(data_files(i).name);
    fprintf('Processing: %s\n', data_files(i).name);


    % delete(strrep(data_files(i).name, '.mat', '.pdf'))
    numTrials = length(SessionData.RawEvents.Trial);
    


    overallMax = 0;
    for trialIdx = 1:length(SessionData.RawEvents.Trial)
        eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
        overallMax = max(overallMax, max(eyeAreaPixels));
    end

    % Extract mouse/session info
    mouse_name = extractBefore(data_files(i).name, '_');
    if isfield(SessionData, 'SleepDeprivedLabel')
        session_label = SessionData.SleepDeprivedLabel;
    else
        session_label = 'Unknown';
    end
    session_title = sprintf('%s \\textemdash\\ %s', mouse_name, session_label);

    for ctr_trial = 1:numTrials
        trial_data = SessionData.RawEvents.Trial{1, ctr_trial};

            % Skip trial if timeout occurred
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end
        
        % Skip trial if ISI does not exist (i.e., no LED_Puff_ISI field)
        if (SessionData.RawEvents.Trial{1, ctr_trial}.Data.IsProbeTrial)  
            continue;
        end

        % Determine short or long
        isShortBlock = trial_data.States.LED_Puff_ISI(2) - ...
                       trial_data.States.LED_Puff_ISI(1) <= 0.3;


        LED_Puff_ISI_start = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
        
        AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
        AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;

        % Extract timing data
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes; 
        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        LED_Onset_Zero_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End - LED_Onset;
        AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
        AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;


        FEC_led_aligned = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes - LED_Puff_ISI_start;
        % FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels ./overallMax;
        FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels ./ overallMax;


        % LED_on = trial_data.Events.GlobalTimer1_Start;
        % Puff_on = trial_data.Events.GlobalTimer2_Start;
        % FEC_led_aligned = trial_data.Data.FECTimes - LED_on;
        % FEC_norm = 1 - trial_data.Data.eyeAreaPixels / overallMax;

        fps = 250; % frames per second, frequency of images
        seconds_before = 0.5;
        seconds_after = 2;

        Frames_before = fps * seconds_before;
        Frames_after = fps * seconds_after;
       % Determine a common time vector for interpolation
        % commonTime = linspace(-seconds_before, seconds_after, Frames_before + Frames_after + 1);
        common_time_vector = linspace(-seconds_before, seconds_after, Frames_before + Frames_after + 1);
       % Initialize a matrix to store interpolated FEC data
        FEC_norm_matrix = zeros(numTrials, length(common_time_vector));

        abs_FEC_led_aligned = abs(FEC_led_aligned);
        closest_frame_idx_to_LED_Onset = find(abs_FEC_led_aligned == min(abs_FEC_led_aligned));

        start_idx = closest_frame_idx_to_LED_Onset - Frames_before;
        stop_idx = closest_frame_idx_to_LED_Onset + Frames_after;

        len_FEC_led_aligned = length(FEC_led_aligned);
        start_idx = max(1, min(start_idx, len_FEC_led_aligned));
        stop_idx = max(1, min(stop_idx, len_FEC_led_aligned));
        FEC_led_aligned_trimmed = FEC_led_aligned(start_idx : stop_idx);
        FEC_trimmed = FEC_norm(start_idx : stop_idx);
        FEC_times_trimmed = FEC_led_aligned(start_idx:stop_idx);

        % Apply smoothing to reduce noise
        FEC_led_aligned_trimmed_smooth = smoothdata(FEC_led_aligned_trimmed, 'movmean', 5); % Moving average
        FEC_trimmed_smooth = smoothdata(FEC_trimmed, 'movmean', 5); % Moving average



        isShortBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - ...
                       SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) <= 0.3;

        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - ...
                      SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;

        t_LED = LED_Onset_Zero_End;
        t_puff = AirPuff_LED_Onset_Aligned_Start;
        

        % CR_category = classify_CR_05(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, t_LED, t_puff , good_CR_threshold, poor_CR_threshold);
        % CR_category = classify_CR_06(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, t_LED, t_puff , good_CR_threshold, poor_CR_threshold, no_CR_threshold);
        
        % CR_category = CR_max_amplitude(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, t_LED, t_puff, good_CR_threshold, poor_CR_threshold, no_CR_threshold);
        
        % Get CR classification and amplitudes
        [CR_category, CR_mag_bs, CR_mag_raw, baseline_amp] = CR_max_amplitude( ...
            FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, ...
            t_LED, t_puff, good_CR_threshold, poor_CR_threshold, no_CR_threshold);

        commonTime = linspace(min(FEC_led_aligned_trimmed_smooth), max(FEC_led_aligned_trimmed_smooth), 500);  % Adjust 100 to the desired number of points
        FEC_norm_interp = interp1(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, commonTime, 'spline', NaN);

        % Define baseline region for individual trial
        baseline_idx = FEC_led_aligned_trimmed_smooth >= -0.2 & FEC_led_aligned_trimmed_smooth <= 0;
        trial_baseline = mean(FEC_led_aligned_trimmed_smooth(baseline_idx), 'omitnan');

          % Choose the correct group struct
        if strcmp(session_label, 'Pre_EBC_SD')
            % group = 'SD';
        else
        %     group = 'Control';
        
        
        % Construct field names and append
        switch CR_category
            case 'No CR'
                if isShortBlock
                    CR_No_short_bs = [CR_No_short_bs, CR_mag_bs];
                    CR_No_short_raw = [CR_No_short_raw, CR_mag_raw];
                else
                    CR_No_long_bs = [CR_No_long_bs, CR_mag_bs];
                    CR_No_long_raw = [CR_No_long_raw, CR_mag_raw];
                end
            case 'Poor CR'
                if isShortBlock
                    CR_Poor_short_bs = [CR_Poor_short_bs, CR_mag_bs];
                    CR_Poor_short_raw = [CR_Poor_short_raw, CR_mag_raw];
                else
                    CR_Poor_long_bs = [CR_Poor_long_bs, CR_mag_bs];
                    CR_Poor_long_raw = [CR_Poor_long_raw, CR_mag_raw];
                end
            case 'Good CR'
                if isShortBlock
                    CR_Good_short_bs = [CR_Good_short_bs, CR_mag_bs];
                    CR_Good_short_raw = [CR_Good_short_raw, CR_mag_raw];
                else
                    CR_Good_long_bs = [CR_Good_long_bs, CR_mag_bs];
                    CR_Good_long_raw = [CR_Good_long_raw, CR_mag_raw];
                end
        end
        end
    
    end

end


% === Common bin settings ===
edges = 0:0.02:1;
bin_centers = edges(1:end-1) + diff(edges)/2;

    group = Control;
    color_good = [0.0, 0.0, 0.0];     % Black
    color_poor = [0.5, 0.5, 0.5];     % Grey
    color_no   = [0.8, 0.8, 0.8];     % Light Grey
    group_name = 'Control';

% === Create figure ===
figure('Color', 'w', 'Position', [100, 100, 1200, 1200]);
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% --- Short Block — Raw ---
nexttile;
N_total = numel(CR_Poor_short_raw) + numel(CR_No_short_raw) + numel(CR_Good_short_raw);
% frac_poor = movmean(histcounts(CR_Poor_short_raw, edges) / N_total,5);
% frac_no   = movmean(histcounts(CR_No_short_raw, edges) / N_total,5);
% frac_good = movmean(histcounts(CR_Good_short_raw, edges) / N_total,5);

frac_poor = movmean(histcounts(CR_Poor_short_raw, edges) / max(numel(CR_Poor_short_raw),1), 5);
frac_no   = movmean(histcounts(CR_No_short_raw, edges) / max(numel(CR_No_short_raw),1), 5);
frac_good = movmean(histcounts(CR_Good_short_raw, edges) / max(numel(CR_Good_short_raw),1), 5);


plot(bin_centers, frac_good, '-', 'Color', color_good, 'LineWidth', 2); hold on;
plot(bin_centers, frac_poor, '-', 'Color', color_poor, 'LineWidth', 2);
plot(bin_centers, frac_no,   '-', 'Color', color_no,   'LineWidth', 2);
title('\textnormal{Short Block --- Raw}', 'Interpreter', 'latex');
xlabel('CR Magnitude', 'Interpreter', 'latex');
ylabel('Fraction of All Trials', 'Interpreter', 'latex');
legend({'Good', 'Poor', 'No'}, 'Location', 'bestoutside');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'TickDir', 'out'); box off;

% --- Short Block — Baseline Subtracted ---
nexttile;
N_total = numel(CR_Poor_short_bs) + numel(CR_No_short_bs) + numel(CR_Good_short_bs);

% frac_poor = movmean(histcounts(CR_Poor_short_bs, edges) / N_total,5);
% frac_no   = movmean(histcounts(CR_No_short_bs, edges) / N_total,5);
% frac_good = movmean(histcounts(CR_Good_short_bs, edges) / N_total, 5);

frac_poor = movmean(histcounts(CR_Poor_short_bs, edges) / max(numel(CR_Poor_short_bs),1), 5);
frac_no   = movmean(histcounts(CR_No_short_bs, edges) / max(numel(CR_No_short_bs),1), 5);
frac_good = movmean(histcounts(CR_Good_short_bs, edges) / max(numel(CR_Good_short_bs),1), 5);

plot(bin_centers, frac_good, '-', 'Color', color_good, 'LineWidth', 2); hold on;
plot(bin_centers, frac_poor, '-', 'Color', color_poor, 'LineWidth', 2);
plot(bin_centers, frac_no,   '-', 'Color', color_no,   'LineWidth', 2);
title('\textnormal{Short Block --- Baseline Subtracted}', 'Interpreter', 'latex');
xlabel('CR Magnitude', 'Interpreter', 'latex');
ylabel('Fraction of All Trials', 'Interpreter', 'latex');
legend({'Good', 'Poor', 'No'}, 'Location', 'bestoutside');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'TickDir', 'out'); box off;

% --- Long Block — Raw ---
nexttile;
N_total = numel(CR_Poor_long_raw) + numel(CR_No_long_raw) + numel(CR_Good_long_raw);
% frac_poor = movmean(histcounts(CR_Poor_long_raw, edges) / N_total,5);
% frac_no   = movmean(histcounts(CR_No_long_raw, edges) / N_total,5);
% frac_good = movmean(histcounts(CR_Good_long_raw, edges) / N_total,5);

frac_poor = movmean(histcounts(CR_Poor_long_raw, edges) / max(numel(CR_Poor_long_raw),1), 5);
frac_no   = movmean(histcounts(CR_No_long_raw, edges) / max(numel(CR_No_long_raw),1), 5);
frac_good = movmean(histcounts(CR_Good_long_raw, edges) / max(numel(CR_Good_long_raw),1), 5);

plot(bin_centers, frac_good, '-', 'Color', color_good, 'LineWidth', 2); hold on;
plot(bin_centers, frac_poor, '-', 'Color', color_poor, 'LineWidth', 2);
plot(bin_centers, frac_no,   '-', 'Color', color_no,   'LineWidth', 2);
title('\textnormal{Long Block --- Raw}', 'Interpreter', 'latex');
xlabel('CR Magnitude', 'Interpreter', 'latex');
ylabel('Fraction of All Trials', 'Interpreter', 'latex');
legend({'Good', 'Poor', 'No'}, 'Location', 'bestoutside');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'TickDir', 'out'); box off;

% --- Long Block — Baseline Subtracted ---
nexttile;
N_total = numel(CR_Poor_long_bs) + numel(CR_No_long_bs) + numel(CR_Good_long_bs);
% frac_poor = movmean(histcounts(CR_Poor_long_bs, edges) / N_total,5);
% frac_no   = movmean(histcounts(CR_No_long_bs, edges) / N_total,5);
% frac_good = movmean(histcounts(CR_Good_long_bs, edges) / N_total,5);

frac_poor = movmean(histcounts(CR_Poor_long_bs, edges) / max(numel(CR_Poor_long_bs),1), 5);
frac_no   = movmean(histcounts(CR_No_long_bs, edges) / max(numel(CR_No_long_bs),1), 5);
frac_good = movmean(histcounts(CR_Good_long_bs, edges) / max(numel(CR_Good_long_bs),1), 5);

plot(bin_centers, frac_good, '-', 'Color', color_good, 'LineWidth', 2); hold on;
plot(bin_centers, frac_poor, '-', 'Color', color_poor, 'LineWidth', 2);
plot(bin_centers, frac_no,   '-', 'Color', color_no,   'LineWidth', 2);
title('\textnormal{Long Block --- Baseline Subtracted}', 'Interpreter', 'latex');
xlabel('CR Magnitude', 'Interpreter', 'latex');
ylabel('Fraction of All Trials', 'Interpreter', 'latex');
legend({'Good', 'Poor', 'No'}, 'Location', 'bestoutside');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'TickDir', 'out'); box off;

% === Overall title ===
sgtitle(['Line Distribution of Pooled CRs — ' group_name], ...
    'Interpreter', 'latex', 'FontSize', 14);

% Optional: export
exportgraphics(gcf, ['Pooled SD_Line_CR_Distributions_', group_name, '.pdf'], 'ContentType', 'vector');