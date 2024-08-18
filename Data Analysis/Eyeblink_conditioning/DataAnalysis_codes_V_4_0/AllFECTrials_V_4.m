clc; close all; clear;
data_files = dir('*_EBC_*.mat');

for i = 1:length(data_files)
    load(data_files(i).name)
    numTrials = length(SessionData.RawEvents.Trial);

    % Create a colormap from light blue to dark blue
    colors = [linspace(0.6, 0, numTrials)', linspace(0.6 , 0, numTrials)', linspace(1, 0.5, numTrials)'];
    figure('units','centimeters','position',[2 2 24 26])

    numCurves = 0;
    leg_str = cell(1, 2);
    legend_handles = [];
    allEyeAreaPixels = [];

    % Loop through each trial to collect eyeAreaPixels data
    for trialIdx = 1:numTrials
        eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
        allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data
    end
    overallMax = max(allEyeAreaPixels);

    fps = 250; % frames per second, frequency of images
    seconds_before = 0.5;
    seconds_after = 2;
    Frames_before = fps * seconds_before;
    Frames_after = fps * seconds_after;

    % Determine a common time vector for interpolation
    common_time_vector = linspace(-seconds_before, seconds_after, Frames_before + Frames_after + 1);

    % Initialize a matrix to store interpolated FEC data
    FEC_norm_matrix = zeros(numTrials, length(common_time_vector));

    % Variables to track min and max of the curves
    minNumCurve = inf;
    maxNumCurve = -inf;

    for ctr_trial = 1:numTrials
        numCurves = numCurves + 1;

        %  if CheckEyeOpenTimeout is not nan, timeout occurred, continue to
        % next trial
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
        Start = SessionData.RawEvents.Trial{1, ctr_trial}.States.Start(1);
        ITI_Pre = SessionData.RawEvents.Trial{1, ctr_trial}.States.ITI_Pre(1);
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        LED_Onset_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End;
        LED_Puff_ISI_start = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
        AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
        AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;

        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        LED_Onset_Zero_End = LED_Onset_End - LED_Onset;
        AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
        AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;

        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;
        FEC_led_aligned = FECTimes - LED_Onset;
        FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels / overallMax;

        % Align frames to LED onset
        abs_FEC_led_aligned = abs(FEC_led_aligned);
        closest_frame_idx_to_LED_Onset = find(abs_FEC_led_aligned == min(abs_FEC_led_aligned));

        start_idx = closest_frame_idx_to_LED_Onset - Frames_before;
        stop_idx = closest_frame_idx_to_LED_Onset + Frames_after;

        len_FEC_led_aligned = length(FEC_led_aligned);
        start_idx = max(1, min(start_idx, len_FEC_led_aligned));
        stop_idx = max(1, min(stop_idx, len_FEC_led_aligned));

        FEC_led_aligned_trimmed = FEC_led_aligned(start_idx:stop_idx);
        FEC_trimmed = FEC_norm(start_idx:stop_idx);

        % Interpolate FEC data to common time vector
        FEC_interpolated = interp1(FEC_led_aligned_trimmed, FEC_trimmed, common_time_vector, 'linear', 'extrap');
        FEC_norm_matrix(ctr_trial, :) = FEC_interpolated;

        h(numCurves) = plot(common_time_vector, FEC_interpolated, 'Color', colors(ctr_trial, :)); hold on

        % Update min and max for y-axis
        minNumCurve = min(minNumCurve, min(FEC_interpolated));
        maxNumCurve = max(maxNumCurve, max(FEC_interpolated));

        if ctr_trial == 1
            leg_str{1} = sprintf('Trial  %03.0f ', ctr_trial);
            legend_handles(1) = h(numCurves);
        end
    end

    % Calculate average
    FEC_norm_avg = mean(FEC_norm_matrix, 1);
    h_avg = plot(common_time_vector, FEC_norm_avg, 'Color', 'r', 'LineStyle', '-', 'LineWidth', 1.3);
    leg_str{2} = sprintf('Trial  %03.0f ', numTrials);
    legend_handles(2) = h(numCurves);
    leg_str{3} = 'Average';
    legend_handles(3) = h_avg;

    % Add a small margin around the min and max values for better visualization
    % y_margin = 0.05 * (maxNumCurve - minNumCurve);
    % minNumCurve = minNumCurve - y_margin;
    % maxNumCurve = maxNumCurve + y_margin;


    % Shade the area (LED Duration)
    for ctr_trial = 1:numTrials
        x_fill = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End,LED_Onset_Zero_Start];         % x values for the filled area
        y_fill = [minNumCurve minNumCurve maxNumCurve maxNumCurve];     % y values for the filled area (y=0 at the x-axis)
        fill(x_fill, y_fill, 'green', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    end

    % Shade the area (AirPuff Duration)
    for ctr_trial = 1:numTrials
        x_fill = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End,AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];         % x values for the filled area
        y_fill = [minNumCurve minNumCurve maxNumCurve maxNumCurve];     % y values for the filled area (y=0 at the x-axis)
        fill(x_fill, y_fill, 'm', 'FaceAlpha', 0.35, 'EdgeColor', 'none');
    end

    % grid on;
    set(gca, 'Box', 'off');
    xlim([-0.5 1]);
    ylim([minNumCurve maxNumCurve]);
    set(gca, 'FontSize', 14)
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 17)
    xlabel('Time from LED Onset (sec)', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 17)
    h_legend = legend(legend_handles, leg_str, 'Interpreter', 'latex', 'fontsize', 13, 'location', 'northeast', 'Box', 'off');
    h_legend.NumColumns = 1;
    h_legend_pos = h_legend.Position;
    h_legend.Position = [0.98 * h_legend_pos(1) 0.99 * h_legend_pos(2) h_legend_pos(3) h_legend_pos(4)];

    % Set tick marks to be outside
    set(gca, 'TickDir', 'out');
    % Adding text annotations
    text(0, maxNumCurve, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 90);
    text(0.3, maxNumCurve, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 90);

    % Extract parts of the filename
    [~, name, ~] = fileparts(data_files(i).name);

    % Split the filename to get the required parts
    nameParts = split(name, '_');
    if length(nameParts) >= 5
        prefixPart = nameParts{1}; % First string before the first underscore
        datePart = nameParts{6}; % Date part

        % Construct the new filename
        newFilename = sprintf('%s_AllFECTrials_%s.pdf', prefixPart, datePart);
        
        % Export the graphics
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end
end
