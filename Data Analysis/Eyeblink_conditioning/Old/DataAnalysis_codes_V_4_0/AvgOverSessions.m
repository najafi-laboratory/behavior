clc; close all; clear

% Load all session data files
data_files = dir('*_EBC_*.mat');

% Container for all average curves
all_avg_curves = {};
leg_str = cell(1, 2);
legend_handles = [];
% Define a common time vector for interpolation
common_time_vector = linspace(-0.5, 2, 1000); % Adjust range if needed

% Loop over each data file
for i = 1:length(data_files)
    load(data_files(i).name);
    numTrials = length(SessionData.RawEvents.Trial);

    % Initialize an empty array to store all eyeAreaPixels values
    allEyeAreaPixels = [];
    % Collect eyeAreaPixels data from all trials
    for trialIdx = 1:numTrials
        eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
        allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data
    end
    % Find the overall maximum value across all collected eyeAreaPixels
    overallMax = max(allEyeAreaPixels);

    numCurves = 0;
    FECTime = {};
    FEC_norm = {};

    % Loop over each trial to calculate shifted and normalized FEC
    for ctr_trial = 1:numTrials
        
        %  if CheckEyeOpenTimeout is not nan, timeout occurred, continue to
        %  next trial
        % 
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end

        % get CheckEyeOpen if it is in session data for versions V_3_2+
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpen')
            CheckEyeOpenStart = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(1);
            CheckEyeOpenStop = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(2);
        end

        % CheckEyeOpen = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(2);
        Start = SessionData.RawEvents.Trial{1, ctr_trial}.States.Start(1);
        ITI_Pre = SessionData.RawEvents.Trial{1, ctr_trial}.States.ITI_Pre(1);
        % LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Onset(1);
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        LED_Onset_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End;
        LED_Puff_ISI_start = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
        AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
        AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;
        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes; 

        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        LED_Onset_Zero_End = LED_Onset_End - LED_Onset;
        AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
        AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;

        if contains(data_files(i).name, 'V_2_9') || ...
           contains(data_files(i).name, 'V_3_0')
            FEC_led_aligned = FECTimes + ITI_Pre - LED_Onset;
        else
            FEC_led_aligned = FECTimes - LED_Onset;
        end

        % Align FEC times to LED onset
        % FEC_led_aligned = FECTimes - LED_Onset;
        FEC_norm_curve = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels / overallMax;

        % Extract a specific range around LED onset
        fps = 250; % frames per second
        seconds_before = 0.5;
        seconds_after = 2;

        Frames_before = fps * seconds_before;
        Frames_after = fps * seconds_after;

        % Find the closest frame to LED onset
        abs_FEC_led_aligned = abs(FEC_led_aligned);
        closest_frame_idx_to_LED_Onset = find(abs_FEC_led_aligned == min(abs_FEC_led_aligned));

        % Determine start and stop indices for trimming
        start_idx = closest_frame_idx_to_LED_Onset - Frames_before;
        stop_idx = closest_frame_idx_to_LED_Onset + Frames_after;

        % Ensure indices are within valid range
        len_FEC_led_aligned = length(FEC_led_aligned);
        start_idx = max(1, min(start_idx, len_FEC_led_aligned));
        stop_idx = max(1, min(stop_idx, len_FEC_led_aligned));

        % Trim aligned FEC data
        FEC_led_aligned_trimmed = FEC_led_aligned(start_idx:stop_idx);
        FEC_trimmed = FEC_norm_curve(start_idx:stop_idx);

        % Check if the data is sufficient for interpolation
        if length(FEC_led_aligned_trimmed) >= 2 && length(FEC_trimmed) >= 2
            % Store for interpolation
            numCurves = numCurves + 1;
            FECTime{numCurves} = FEC_led_aligned_trimmed;
            FEC_norm{numCurves} = FEC_trimmed;
        else
            % Display a warning if data is insufficient
            warning('Trial %d in file %s has insufficient data for interpolation and will be skipped.', ctr_trial, data_files(i).name);
        end
    end

    % Interpolate data onto the common time vector if there are valid curves
    if numCurves > 0
        FEC_interp = zeros(numCurves, length(common_time_vector));
        for ctr_curve = 1:numCurves
            FEC_interp(ctr_curve, :) = interp1(FECTime{ctr_curve}, FEC_norm{ctr_curve}, common_time_vector, 'linear', 'extrap');
        end

        % Calculate average FEC across trials
        FEC_norm_avg = mean(FEC_interp, 1);
        all_avg_curves{end+1} = struct('FECTime_avg', common_time_vector, 'FEC_norm_avg', FEC_norm_avg);
    else
        % Display a warning if no valid curves were found
        warning('No valid curves found in file %s for interpolation.', data_files(i).name);
    end
end

% Plot all average curves
figure('units', 'centimeters', 'position', [2 2 24 26]);
hold on;

% Define a colormap from dim to dark blue
num_colors = length(all_avg_curves);
color_map = [linspace(0.5, 0, num_colors)', linspace(0.5, 0, num_colors)', linspace(1, 0.5, num_colors)'];

% Plot each session average curve with distinct color
for i = 1:length(all_avg_curves)
    avg_curve = all_avg_curves{i};
    h = plot(avg_curve.FECTime_avg, avg_curve.FEC_norm_avg, 'Color', color_map(i, :), 'LineWidth', 1.7);
    
    % Store the handles for the first and last plots
    if i == 1 || i == length(all_avg_curves)
        legend_handles = [legend_handles, h];
    end
end

% Add text annotations
text_cell{1} = 'LED';
t1 = text(0, 1, text_cell, 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
t1.Rotation = 90;

text_cell{1} = 'AirPuff';
t2 = text(0.3, 1, text_cell, 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
t2.Rotation = 90;

% Load the first file for shading areas
load(data_files(1).name);
numTrials = length(SessionData.RawEvents.Trial);

% Shade the LED to Puff interval
for i = 1:length(all_avg_curves)
    % Shade the area (LED Duration)
    x_fill_LED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
    y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
    fill(x_fill_LED, y_fill, 'green', 'FaceAlpha', 0.05, 'EdgeColor', 'none');
    
    % Shade the area (AirPuff Duration)
    x_fill_Puff = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
    fill(x_fill_Puff, y_fill, 'm', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
end

% Generate legend entries
legend_entries = cell(1, length(legend_handles));
for idx = 1:length(legend_handles)
    if idx == 1
        i = 1;
    else
        i = length(data_files);
    end
    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 6
        prefixPart = nameParts{1};
        datePart = nameParts{6};
        legend_entries{idx} = sprintf('%s_%s', prefixPart, datePart);
    end     
end

% Add legend for the first and last curves only
legend(legend_handles, legend_entries, 'Interpreter', 'latex', 'fontsize', 13, 'location', 'northeast', 'Box', 'off');

% Set plot limits and labels
ylim([0 1]);
xlim([-0.5 1]);
set(gca, 'FontSize', 14);
ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 17);
xlabel('Time from LED Onset (s)', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 17);


set(gca, 'Box', 'off');
set(gca, 'TickDir', 'out');

% Export graphics
newFilename = sprintf('AvgOverSessions.pdf');
exportgraphics(gcf, newFilename, 'ContentType', 'vector');
