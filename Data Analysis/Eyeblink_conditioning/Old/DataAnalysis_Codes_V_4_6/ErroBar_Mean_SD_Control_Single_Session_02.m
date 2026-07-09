clc; close all; clear;

data_files = dir('*_EBC_*.mat');
CR_threshold = 0.05;

% Initialize arrays to store CR+ fractions for all sessions
CR_plus_fractions_Control_short = [];
CR_plus_fractions_Control_long = [];
CR_plus_fractions_SD_short = [];
CR_plus_fractions_SD_long = [];
sessionLabels = {};

% Loop through sessions
for i = 1:length(data_files)
    % Load session data
    loadedData = load(data_files(i).name);
    if ~isfield(loadedData, 'SessionData')
        warning(['SessionData not found in ' data_files(i).name]);
        continue;
    end
    SessionData = loadedData.SessionData;

    % Determine session type
    isSD = isfield(SessionData, 'SleepDeprived') && SessionData.SleepDeprived == 1;

    % Reset counters
    numCurves_long = 0; numCurves_short = 0;
    numLongTrials = 0; numShortTrials = 0;

    % Normalize FEC
    allEyeAreaPixels = [];
    numTrials = length(SessionData.RawEvents.Trial);
    for trialIdx = 1:numTrials
        if isfield(SessionData.RawEvents.Trial{1, trialIdx}.Data, 'eyeAreaPixels')
            eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
            allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels];
        end
    end
    overallMax = max(allEyeAreaPixels);

    fps = 250; % frames per second, frequency of images
    seconds_before = 0.5;
    seconds_after = 2;
    Frames_before = fps * seconds_before;
    Frames_after = fps * seconds_after;

    % Process trials
    for ctr_trial = 1:numTrials
        % Skip trials with CheckEyeOpenTimeout
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end

        % Get trial data for LED onset, puff, and timings
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
        AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;

        % Align times to LED onset
        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
        AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;

        % Normalize FEC data
        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;
        FEC_led_aligned = FECTimes - LED_Onset;
        FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels / overallMax;

        % Trim aligned FEC data
        abs_FEC_led_aligned = abs(FEC_led_aligned);
        closest_frame_idx_to_LED_Onset = find(abs_FEC_led_aligned == min(abs_FEC_led_aligned));

        start_idx = closest_frame_idx_to_LED_Onset - Frames_before;
        stop_idx = closest_frame_idx_to_LED_Onset + Frames_after;

        len_FEC_led_aligned = length(FEC_led_aligned);
        start_idx = max(1, min(start_idx, len_FEC_led_aligned));
        stop_idx = max(1, min(stop_idx, len_FEC_led_aligned));

        FEC_trimmed = FEC_norm(start_idx:stop_idx);
        FEC_times_trimmed = FEC_led_aligned(start_idx:stop_idx); 

        t_LED = LED_Onset_Zero_Start;
        t_puff = AirPuff_LED_Onset_Aligned_Start;
        
        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;


        is_CR_plus = CR_plus_eval_dev(FEC_times_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);
        if isLongBlock 
            numLongTrials = numLongTrials + 1;
            if is_CR_plus
                numCurves_long = numCurves_long + 1;
            end

        else
             numShortTrials = numShortTrials + 1;
             if is_CR_plus
                numCurves_short = numCurves_short + 1;
             end
        end
    end

    % Calculate CR+ fractions
    CR_plus_fractions_Control_short(end+1) = ~isSD * numCurves_short / max(numShortTrials, 1);
    CR_plus_fractions_Control_long(end+1) = ~isSD * numCurves_long / max(numLongTrials, 1);
    CR_plus_fractions_SD_short(end+1) = isSD * numCurves_short / max(numShortTrials, 1);
    CR_plus_fractions_SD_long(end+1) = isSD * numCurves_long / max(numLongTrials, 1);

    % Generate session label
    if isfield(SessionData.Info, 'SessionDate')
        sessionLabels{end+1} = SessionData.Info.SessionDate;
    else
        sessionLabels{end+1} = ['Session ', num2str(i)];
    end
end

% Combine all CR+ fractions into one matrix
allFractions = [
    CR_plus_fractions_Control_short;
    CR_plus_fractions_SD_short;
    CR_plus_fractions_Control_long;
    CR_plus_fractions_SD_long
]';

% Plot grouped bar chart with adjusted bar width and alignment
figure;
barWidth = 0.7; % Thicker bars
barSpacing = 0.1; % Controls spacing between bars within a group

x_positions = 1:length(sessionLabels); % X-axis positions for each session
x_offsets = [-0.15, -0.05, 0.05, 0.15]; % Small offsets for 4 bars per session

hold on;
% Plot bars for each group
for i = 1:size(allFractions, 1)
    bar(x_positions(i) + x_offsets(1), allFractions(i, 1), barWidth / 4, 'FaceColor', [0.7 0.7 0.7]); % Control Short
    bar(x_positions(i) + x_offsets(2), allFractions(i, 2), barWidth / 4, 'FaceColor', [1 0 0]);     % SD Short
    bar(x_positions(i) + x_offsets(3), allFractions(i, 3), barWidth / 4, 'FaceColor', [0.5 0.5 0.5]); % Control Long
    bar(x_positions(i) + x_offsets(4), allFractions(i, 4), barWidth / 4, 'FaceColor', [0.8 0.2 0.2]); % SD Long
end

% Customize the axes
set(gca, 'XTick', x_positions, 'XTickLabel', sessionLabels, 'XTickLabelRotation', 45);
ylabel('CR+ Fraction', 'Interpreter', 'latex', 'FontSize', 12);
ylim([0 1]);
set(gca, 'FontSize', 12, 'TickDir', 'out', 'Box', 'off'); % Box off
title({'Comparison of CR$^{+}$ Fractions Across All Sessions'; ...
       sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f$', CR_threshold)}, ...
       'Interpreter', 'latex', 'FontSize', 14);

hBars = findall(gca, 'Type', 'Bar'); % Find all bar objects
for i = 1:numel(hBars)
    hBars(i).BarWidth = 0.2; % Increase bar thickness
end

legend boxoff;
% Add legend
legend({'Control Short', 'SD Short', 'Control Long', 'SD Long'}, ...
       'Location', 'bestoutside', 'Interpreter', 'latex', 'FontSize', 10);

% Save plot
set(gcf, 'Position', [100, 100, 1200, 600]);
exportgraphics(gcf, 'Combined_CRplus_Sessions.pdf', 'ContentType', 'vector');
hold off;

