clc; close all; clear

data_files = dir('*_EBC_*.mat');
% data_files = dir('E1VT_EBC_V_3_8_20240806_130856.mat');

CR_threshold = 0.05;

for i = 1:length(data_files)

    load(data_files(i).name)

    numTrials = length(SessionData.RawEvents.Trial);
    % Create a colormap from light blue to dark blue
    colors = [linspace(0.6, 0, numTrials)', linspace(0.6, 0, numTrials)', linspace(1, 0.5, numTrials)'];

    % Create a figure for subplots
    figure;
    
    % Initialize variables for tracking CR+ and CR- trials
    negativeCurves = 0;
    numCurves = 0;
    totalFEC_norm_neg = [];
    totalFEC_norm = [];
    leg_str = cell(1, 1);

    % Initialize subplot axes
    ax1 = subplot(2, 1, 1); % For CR+
    ax2 = subplot(2, 1, 2); % For CR-

    % Initialize an empty array to store all eyeAreaPixels values
    allEyeAreaPixels = [];

    % Loop through each trial to collect eyeAreaPixels data
    for trialIdx = 1:numTrials
        eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
        allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data
    end
    % Find the overall maximum value across all collected eyeAreaPixels
    overallMax = max(allEyeAreaPixels);

    step = 1;
    for ctr_trial = 1:step:numTrials

        % % Extract relevant events and states
        % Start = SessionData.RawEvents.Trial{1, ctr_trial}.States.Start(1);
        % ITI_Pre = SessionData.RawEvents.Trial{1, ctr_trial}.States.ITI_Pre(1);
        % LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Onset(1);
        % LED_Onset_End = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
        % LED_Puff_ISI_start = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
        % LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
        % AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.States.AirPuff(1);
        % AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.States.AirPuff(2);
        % FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;
        
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
        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;

        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        LED_Onset_Zero_End = LED_Onset_End - LED_Onset;
        AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
        AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;
        
        FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels / overallMax;
        FEC_led_aligned = FECTimes - LED_Onset;
        t_LED = LED_Onset_Zero_Start;
        t_puff = AirPuff_LED_Onset_Aligned_End;
        t1 = t_LED - 0.01;
        t2 = t_LED;

        % frames per second, frequency of images
        fps = 250;
        seconds_before = 0.5;
        seconds_after = 2;

        Frames_before = fps * seconds_before;
        Frames_after = fps * seconds_after;

        abs_FEC_led_aligned = abs(FEC_led_aligned);
        closest_frame_idx_to_LED_Onset = find(abs_FEC_led_aligned == min(abs_FEC_led_aligned), 1);

        start_idx = closest_frame_idx_to_LED_Onset - Frames_before;
        stop_idx = closest_frame_idx_to_LED_Onset + Frames_after;

        % Ensure indices are within bounds
        len_FEC_led_aligned = length(FEC_led_aligned);
        start_idx = max(1, min(start_idx, len_FEC_led_aligned));
        stop_idx = max(1, min(stop_idx, len_FEC_led_aligned));

        FEC_led_aligned_trimmed = FEC_led_aligned(start_idx:stop_idx);
        FEC_trimmed = FEC_norm(start_idx:stop_idx);
        
        % Only proceed if FEC_led_aligned_trimmed is non-empty
        if ~isempty(FEC_led_aligned_trimmed)
        % Calculate the minimum and maximum values for interpolation
        minValue = min(FEC_led_aligned_trimmed(:));
        maxValue = max(FEC_led_aligned_trimmed(:));


        % Define the common time vector using the scalar min and max values
        % Ensure minValue and maxValue are valid scalars
        if isscalar(minValue) && isscalar(maxValue) && minValue ~= maxValue
        % Define the common time vector using the scalar min and max values
        commonTime = linspace(minValue, maxValue, 100);
        

        % Check if trial is CR+ or CR-
        if (CR_plus_eval(FEC_led_aligned_trimmed, FEC_trimmed, t1, t2, t_LED, t_puff, CR_threshold))
            numCurves = numCurves + 1;
            % Interpolate FEC_norm to the common time vector
            FEC_norm_interp = interp1(FEC_led_aligned_trimmed, FEC_trimmed, commonTime);

            % Accumulate the data
            totalFEC_norm = [totalFEC_norm; FEC_norm_interp];
            plot(ax1, FEC_led_aligned_trimmed, FEC_trimmed, 'Color', colors(ctr_trial, :)); hold(ax1, 'on');
            title(ax1, 'CR+ Trials');
            xlabel(ax1, 'Time (s)');
            ylabel(ax1, 'FEC (Normalized)');
            ylim(ax1, [0 1]);

        else
            % CR- trial logic
            negativeCurves = negativeCurves +1;
            FEC_norm_interp_neg = interp1(FEC_led_aligned_trimmed, FEC_trimmed, commonTime);
            totalFEC_norm_neg = [totalFEC_norm_neg; FEC_norm_interp_neg];
            plot(ax2, FEC_led_aligned_trimmed, FEC_trimmed, 'Color', colors(ctr_trial, :)); hold(ax2, 'on');
            title(ax2, 'CR- Trials');
            xlabel(ax2, 'Time (s)');
            ylabel(ax2, 'FEC (Normalized)');
            ylim(ax2, [0 1]);
        end

        end
        end
    end

    % Plot the average CR+ curve
    if numCurves > 0
        avgFEC_norm = mean(totalFEC_norm, 1);
        plot(ax1, commonTime, avgFEC_norm, 'red', 'LineWidth', 2); % Average curve in red
        % legend(ax1, 'CR+ Trials', 'Average CR+', 'Location', 'southeast');
    else
        disp('No CR+ trials found.');
    end
    % Plot the average CR- curve
    if negativeCurves > 0
        avgFEC_norm_neg = mean(totalFEC_norm_neg, 1);
        plot(ax2, commonTime, avgFEC_norm_neg, 'red', 'LineWidth', 2); % Average curve in red
    else
        disp('No CR- trials found.');
    end

    % Add shaded areas for events
    for ctr_trial = 1:numTrials
        % Shade the ITI LED
        x_fill = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
        y_fill = [0 0 1 1];
        fill(ax1, x_fill, y_fill, 'green', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        fill(ax2, x_fill, y_fill, 'green', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    end

    for ctr_trial = 1:numTrials
        % Shade the AirPuff area
        x_fill = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
        fill(ax1, x_fill, y_fill, 'm', 'FaceAlpha', 0.35, 'EdgeColor', 'none');
        fill(ax2, x_fill, y_fill, 'm', 'FaceAlpha', 0.35, 'EdgeColor', 'none');
    end

    % Set properties for the entire figure
    set([ax1, ax2], 'Box', 'off', 'TickDir', 'out', 'FontSize', 14);
    xlim(ax1, [-0.5 1]); % Zooming x-axis for CR+ plot
    xlim(ax2, [-0.5 1]); % Zooming x-axis for CR- plot
    % Adjust figure position and size
    set(gcf, 'Position', [100, 100, 800, 600]);

    % Export graphics
    [~, name, ~] = fileparts(data_files(i).name); % Extract the name without extension
    nameParts = split(name, '_');
    if length(nameParts) >= 5
        prefixPart = nameParts{1};
        datePart = nameParts{6};
        newFilename_1 = sprintf('%s_CRp_CRn_trials_%s.pdf', prefixPart, datePart);
        exportgraphics(gcf, newFilename_1, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end
end
