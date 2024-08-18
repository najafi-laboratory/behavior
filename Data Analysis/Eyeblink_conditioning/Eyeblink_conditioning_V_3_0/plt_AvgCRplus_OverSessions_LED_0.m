clc; close all; clear;

data_files = dir('*_EBC_*.mat');
CR_threshold = 0.05;

% Initialize a figure for the average curves
figure('units', 'centimeters', 'position', [2 2 24 26]);
hold on;
legend_entries = {};

% Define a common time vector for interpolation
commonFECTime = linspace(-0.5, 2, 500); % Adjust based on trimming range

for i = 1:length(data_files)
    load(data_files(i).name);

    numTrials = length(SessionData.RawEvents.Trial);
    numCurves = 0;
    totalFEC_norm = [];
    allEyeAreaPixels = [];

    % Loop through each trial to collect eyeAreaPixels data
    for trialIdx = 1:numTrials    
       eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
       allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data 
    end

    overallMax = max(allEyeAreaPixels);

    for ctr_trial = 1:numTrials


        CheckEyeOpen = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(2);
        Start = SessionData.RawEvents.Trial{1, ctr_trial}.States.Start(1);
        ITI_Pre = SessionData.RawEvents.Trial{1, ctr_trial}.States.ITI_Pre(1);
        LED_Puff_ISI_start = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
        AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
        AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;
        % Align FEC times to LED onset
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        LED_Onset_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End;
        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;
        FEC_led_aligned = FECTimes - LED_Onset;
        FEC_norm_curve = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels / overallMax;

        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        LED_Onset_Zero_End = LED_Onset_End - LED_Onset;
        AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
        AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;

        % Trim data around LED onset
        fps = 250; % frames per second
        seconds_before = 0.5;
        seconds_after = 2;
        
        Frames_before = fps * seconds_before;
        Frames_after = fps * seconds_after;
        
        % Find the closest frame to LED onset
        [~, closest_frame_idx_to_LED_Onset] = min(abs(FEC_led_aligned));
        
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
        
        t_LED = LED_Onset_Zero_Start;
        t_puff = AirPuff_LED_Onset_Aligned_End;
        t1 = t_LED-0.01;
        t2 = t_LED;

        if(CR_plus_eval(FEC_led_aligned_trimmed, FEC_trimmed, t1, t2, t_LED, t_puff, CR_threshold))
            numCurves = numCurves + 1;
            % Interpolate FEC_norm onto the common time vector
            FEC_norm_interp = interp1(FEC_led_aligned_trimmed, FEC_trimmed, commonFECTime, 'linear', 'extrap');
            totalFEC_norm = [totalFEC_norm; FEC_norm_interp];
        end
    end
   
    numFiles = length(data_files);
    cmap = [linspace(0.6, 0, numFiles)', linspace(0.6, 0, numFiles)', linspace(1, 0.5, numFiles)'];
    if numCurves > 0
        avgFECTime = commonFECTime;
        avgFEC_norm = mean(totalFEC_norm, 1);
        
        % Plot the average curve in the main figure
        h_avg = plot(avgFECTime, avgFEC_norm, 'LineWidth', 2, 'Color', cmap(i, :)); % Adjust color or style as needed
    else
        disp('No CR+ trials found.');
    end
end

% Adding text annotations
text_cell{1} = 'LED';
t1 = text(0, 1, text_cell, 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
t1.Rotation = 90;
text_cell{1} = 'AirPuff';
t2 = text(0.3, 1, text_cell, 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
t2.Rotation = 90;

load(data_files(1).name);
numTrials = length(SessionData.RawEvents.Trial);
% Plot each session average curve with distinct color
for i = 1:length(data_files)
    % avg_curve = all_avg_curves{i};
    % plot(avg_curve.FECTime_avg, avg_curve.FEC_norm_avg, 'Color', color_map(i, :), 'LineWidth', 1.7);
    
    % Shade the area (LED Duration)
    x_fill_LED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
    y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
    fill(x_fill_LED, y_fill, 'green', 'FaceAlpha', 0.08, 'EdgeColor', 'none');
    
    % Shade the area (AirPuff Duration)
    x_fill_Puff = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
    fill(x_fill_Puff, y_fill, 'm', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
end

for i = 1:length(data_files)
    load(data_files(i).name)
    % Extract parts of the filename
    [~, name, ~] = fileparts(data_files(i).name); % Extract the name without extension
    nameParts = split(name, '_');
    if length(nameParts) >= 5
        prefixPart = nameParts{1}; % First string before the first underscore
        datePart = nameParts{6}; % Date part
        legend_entries{end + 1} = sprintf('%s_%s', prefixPart, datePart); % Add a legend entry for each file
    end
end        

% Finalize the main figure
% Add grid lines
grid on;
set(gca, 'Box', 'off');
% Set tick marks to be outside
set(gca, 'TickDir', 'out');
legend(legend_entries, 'Interpreter', 'latex', 'fontsize', 13, 'location', 'northeast', 'Box', 'off');
xlabel('Time from Trial start (s)', 'Interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 17);
ylabel('Eyelid closure (norm)', 'Interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 17);
set(gca, 'FontSize', 14);
ylim([0 1]);

load(data_files(i).name)
for i = 1:length(data_files)
    % Extract parts of the filename
    [~, name, ~] = fileparts(data_files(i).name); % Extract the name without extension

    % Split the filename to get the required parts
    nameParts = split(name, '_');
    if length(nameParts) >= 5
        prefixPart = nameParts{1}; % First string before the first underscore
        datePart = nameParts{6}; % Date part

        % Construct the new filename
        newFilename = sprintf('All_FEC_Averages_CRplus_Days_%s.pdf', prefixPart);
        
        % Export the graphics
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end
end
