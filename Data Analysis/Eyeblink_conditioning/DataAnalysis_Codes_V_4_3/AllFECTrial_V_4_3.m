clc; close all; clear;
data_files = dir('*_EBC_*.mat');
% data_files = dir('E1VT_EBC_V_3_11_20240829_132532.mat');

for i = 1:length(data_files)
    load(data_files(i).name)
    numTrials = length(SessionData.RawEvents.Trial);

    % Create a colormap for the trials (all same color)
    

    figure('units','centimeters','position',[2 2 24 26])

    numCurves = 0;
    leg_str = cell(1, 2);
    legend_handles = [];
    allEyeAreaPixels = [];

    % Determine a common time vector for interpolation
    fps = 250; % frames per second, frequency of images
    seconds_before = 0.5;
    seconds_after = 2;
    Frames_before = fps * seconds_before;
    Frames_after = fps * seconds_after;
    common_time_vector = linspace(-seconds_before, seconds_after, Frames_before + Frames_after + 1);

    % Initialize a matrix to store interpolated FEC data
    FEC_norm_matrix = zeros(numTrials, length(common_time_vector));



    % Initialize matrices for short and long blocks
    FEC_short_matrix = [];
    FEC_long_matrix = [];

    % Variables to track min and max of the curves
    minNumCurve = inf;
    maxNumCurve = -inf;

    % Create a color gradient from dim to dark green for short blocks
    colorMapLong = [linspace(0.7, 0, numTrials)', linspace(1, 0.5, numTrials)', linspace(0.7, 0, numTrials)']; % Dim to dark green
    
    % Create a color gradient from dim to dark blue for long blocks
    colorMapShort = [linspace(0.7, 0, numTrials)', linspace(0.7, 0, numTrials)', linspace(1, 0.5, numTrials)']; % Dim to dark blue


    % Loop through each trial to collect and plot data
    for ctr_trial = 1:numTrials
        numCurves = numCurves + 1;

        % If CheckEyeOpenTimeout is not nan, skip the trial
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end

        % Get data
        eyeAreaPixels = SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels;
        allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data
        overallMax = max(allEyeAreaPixels);
        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;
        FEC_norm = 1 - eyeAreaPixels / overallMax;

        % Align frames to LED onset
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        FEC_led_aligned = FECTimes - LED_Onset;
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



        % Update min and max for y-axis
        minNumCurve = min(minNumCurve, min(FEC_interpolated));
        maxNumCurve = max(maxNumCurve, max(FEC_interpolated));


        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - LED_Onset;
        % Identify if the trial is short or long
        if SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3
             trialType = 'long';
             verticalLineColor = 'g'; % Green for long Blocks 
             % Long trial (use green gradient)
             plotColor = colorMapLong(ctr_trial, :);
             h(numCurves) = plot(common_time_vector, FEC_interpolated, 'Color', plotColor, 'LineWidth', 0.90); hold on
            
            FEC_long_matrix = [FEC_long_matrix; FEC_interpolated]; % Append to long block matrix
            leg_str{2} = sprintf('Long Blocks', numTrials);
            legend_handles(2) = h(numCurves);
        else
            trialType = 'short';
            verticalLineColor = 'b'; % Blue for short Blocks
             % Short trial (use blue gradient)
            plotColor = colorMapShort(ctr_trial, :);
            h(numCurves) = plot(common_time_vector, FEC_interpolated, 'Color', plotColor, 'LineWidth', 0.90); hold on
            
            FEC_short_matrix = [FEC_short_matrix; FEC_interpolated]; % Append to short block matrix
            leg_str{1} = sprintf('Short Blocks', ctr_trial);
            legend_handles(1) = h(numCurves);
        end
                % Plot the trial with the same color


        % Add vertical line at the end of the LED_Puff_ISI
        
        line([LED_Puff_ISI_end, LED_Puff_ISI_end], [minNumCurve, maxNumCurve], 'Color', verticalLineColor, 'LineWidth', 2);
        
        

        
    end

    % Calculate and plot averages for short and long blocks
    if ~isempty(FEC_short_matrix)
        FEC_short_avg = mean(FEC_short_matrix, 1);
        h_avg_short = plot(common_time_vector, FEC_short_avg, 'Color', 'k', 'LineStyle', '-', 'LineWidth', 3); % Blue for short blocks
        % leg_str{1} = 'Short Block Avg';
        % legend_handles(1) = h_avg_short;
    end

    if ~isempty(FEC_long_matrix)
        FEC_long_avg = mean(FEC_long_matrix, 1);
        h_avg_long = plot(common_time_vector, FEC_long_avg, 'Color', [0.3, 0.3, 0.3] , 'LineStyle', '-', 'LineWidth', 3); % Muted blue for long blocks
        % leg_str{2} = 'Long Block Avg';
        % legend_handles(2) = h_avg_long;
    end
    % % Calculate and plot average
    % FEC_norm_avg = mean(FEC_norm_matrix, 1);
    % h_avg = plot(common_time_vector, FEC_norm_avg, 'Color', 'k', 'LineStyle', '-', 'LineWidth', 2.5);
    % 
    % leg_str{3} = 'Average';
    % legend_handles(3) = h_avg;

    % Plot the vertical line at x = 0 (LED onset)
    line([0, 0], [minNumCurve, maxNumCurve], 'Color', 'k', 'LineWidth', 1, 'LineStyle', '--');

    % Plot customization
    set(gca, 'Box', 'off');
    xlim([-0.2 0.6]);
    ylim([minNumCurve maxNumCurve]);
    set(gca, 'FontSize', 14)
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 17)
    xlabel('Time From LED Onset (sec)', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 17)
    h_legend = legend(legend_handles, leg_str, 'Interpreter', 'latex', 'fontsize', 14, 'location', 'bestoutside', 'Box', 'off');
    % h_legend.NumColumns = 1;
    % h_legend_pos = h_legend.Position;
    % h_legend.Position = [0.98 * h_legend_pos(1) 0.99 * h_legend_pos(2) h_legend_pos(3) h_legend_pos(4)];

    % Set tick marks to be outside
    set(gca, 'TickDir', 'out');
    text(0, maxNumCurve, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 90);
    text(0.2, maxNumCurve, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 90);
    text(0.4, maxNumCurve, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 90);
    % Export the graphics
    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 5
        prefixPart = nameParts{1}; % First string before the first underscore
        datePart = nameParts{6}; % Date part
        newFilename = sprintf('%s_AllFECTrials_%s.pdf', prefixPart, datePart);
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end
end
