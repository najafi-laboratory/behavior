clc; close all; clear

% Load all session data files
data_files = dir('*_EBC_*.mat');


% Initialize legend handles for short and long blocks
legend_handles_short = [];  % For blue plots
legend_handles_long = [];   % For red plots

% Initialize figure
figure;
 
% Subplot 1: Short Block
subplot(2, 1, 1);
hold on;
% Set the title with custom position
title('Short Block Trials', 'Units', 'normalized', 'Position', [0.5, 1.09, 0]);
% legend_handles = [];
legend_entries = {};

% Loop over each data file
for i = 1:length(data_files)
    load(data_files(i).name);
    numTrials = length(SessionData.RawEvents.Trial);

    % Containers for long and short block data for this session
    long_block_data = [];
    short_block_data = [];

    % Loop over each trial to calculate shifted and normalized FEC
    for ctr_trial = 1:numTrials
        % Skip trial if timeout occurred
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end

        % Extract timing data
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes; 

        % Align FEC times to LED onset
        FEC_led_aligned = FECTimes - LED_Onset;

        % Normalize FEC data
        overallMax = max(SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels);
        FEC_norm_curve = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels / overallMax;

        % Define frame selection for trimming
        fps = 250; % frames per second
        seconds_before = 0.5;
        seconds_after = 1.5;
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
        FEC_trimmed = FEC_norm_curve(start_idx:stop_idx);
        FEC_times_trimmed = FEC_led_aligned(start_idx:stop_idx);

        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - LED_Onset;
        
        % Identify if the trial is short or long and store the data
        if length(FEC_trimmed) == length(FEC_times_trimmed)
            if SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3
                long_block_data = [long_block_data; FEC_trimmed];
                verticalLineColor = 'r'; % Red for long trials

            else
                short_block_data = [short_block_data; FEC_trimmed];
                verticalLineColor = 'b'; % Blue for short trials
                 % Plot the vertical line for AirPuff
                line([LED_Puff_ISI_end, LED_Puff_ISI_end], [0 1], 'Color', verticalLineColor, 'LineWidth', 1.5);

               
            end
        else
            warning('Skipping trial %d in file %s due to dimension mismatch.', ctr_trial, data_files(i).name);
        end
        
        
    end
% Define a colormap from dark to dim blue
num_colors = length(data_files);
color_map_short = [linspace(0, 0.5, num_colors)', linspace(0, 0.5, num_colors)', linspace(0.5, 1, num_colors)'];
    % Plot the averages for short blocks
    if ~isempty(short_block_data)
        FEC_norm_avg_short = mean(short_block_data, 1);
        h1 = plot(FEC_times_trimmed, FEC_norm_avg_short, 'Color',color_map_short(i, :), 'LineWidth', 1.2); % Blue for short blocks
        
         % Store handles for the first and last plots to use in the legend
        if i == 1 || i == num_colors
            legend_handles_short = [legend_handles_short, h1];
        end
        
    end
end

% Add vertical line for LED onset
line([0, 0], [0 1], 'Color', 'g', 'LineWidth', 2);  % Green line for LED onset
% legend_entries = {'Short Block'};


% Create the legend entries using your provided logic
legend_entries_short = cell(1, 2);

for idx = 1:2
    if idx == 1
        i = 1;  % First file
    else
        i = length(data_files);  % Last file
    end
    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 6
        prefixPart = nameParts{1};
        datePart = nameParts{6};
        legend_entries_short{idx} = sprintf('%s_%s', prefixPart, datePart);  % e.g., 'Session1_2023'
    end     
end



text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
text(0.2,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'blue', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');


% Configure plot
ylim([0 1]);
xlim([-0.2 0.6]);
% set(gca, 'XTick', []);  % Remove x-axis ticks
% xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 12);
ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
% legend(legend_handles, legend_entries, 'Interpreter', 'latex', 'location', 'northeast', 'Box', 'off');

% Subplot 2: Long Block
subplot(2, 1, 2);
hold on;

% Set the title with custom position
title('Long Block Trials', 'Units', 'normalized', 'Position', [0.5, 1.09, 0]);

% legend_handles = [];

% Loop over each data file again for long blocks
for i = 1:length(data_files)
    load(data_files(i).name);
    numTrials = length(SessionData.RawEvents.Trial);

    long_block_data = [];

    % Loop over each trial to calculate shifted and normalized FEC
    for ctr_trial = 1:numTrials
        % Skip trial if timeout occurred
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end

        % Extract timing data
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes; 

        % Align FEC times to LED onset
        FEC_led_aligned = FECTimes - LED_Onset;

        % Normalize FEC data
        overallMax = max(SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels);
        FEC_norm_curve = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels / overallMax;

        % Define frame selection for trimming
        fps = 250; % frames per second
        seconds_before = 0.5;
        seconds_after = 1.5;
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
        FEC_trimmed = FEC_norm_curve(start_idx:stop_idx);
        FEC_times_trimmed = FEC_led_aligned(start_idx:stop_idx);

        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - LED_Onset;
        
        % Identify if the trial is short or long and store the data
        if length(FEC_trimmed) == length(FEC_times_trimmed)
            if SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3
                long_block_data = [long_block_data; FEC_trimmed];
                verticalLineColor = 'r';
                 % Plot the vertical line for AirPuff
                line([LED_Puff_ISI_end, LED_Puff_ISI_end], [0 1], 'Color', verticalLineColor, 'LineWidth', 1.5);
            else
                short_block_data = [short_block_data; FEC_trimmed];
                verticalLineColor = 'b'; % Blue for short trials

            end
               
        else
            warning('Skipping trial %d in file %s due to dimension mismatch.', ctr_trial, data_files(i).name);
        end
       
    end


num_colors = length(data_files);
color_map_long = [linspace(0.5, 1, num_colors)', linspace(0, 0.5, num_colors)', linspace(0, 0.5, num_colors)'];
    % Plot the averages for long blocks
    if ~isempty(long_block_data)
        FEC_norm_avg_long = mean(long_block_data, 1);
        h2 = plot(FEC_times_trimmed, FEC_norm_avg_long, 'Color',color_map_long(i, :), 'LineWidth', 1.2); % Red for long blocks
        % legend_handles(end+1) = h2;
    
            % Store handles for the first and last plots to use in the legend
        if i == 1 || i == num_colors
            legend_handles_long = [legend_handles_long, h2];
        end
    end
    
end

% Add vertical line for LED onset
line([0, 0], [0 1], 'Color', 'g', 'LineWidth', 2);  % Green line for LED onset
% legend_entries = {'Long Block'};

% Add text annotations for LED and AirPuff
% text(0, 0.9, 'LED', 'Color', 'g', 'FontSize', 12, 'HorizontalAlignment', 'right');
% text(0.4, 0.9, 'AirPuff', 'Color', 'r', 'FontSize', 12, 'HorizontalAlignment', 'right');

% Configure plot
ylim([0 1]);
xlim([-0.2 0.6]);
xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 12);
ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
% legend(legend_handles, legend_entries, 'Interpreter', 'latex', 'location', 'northeast', 'Box', 'off');
text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
text(0.4,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'red', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');

% Adjust subplot positions manually for square-like appearance
subplot(2, 1, 1);
set(gca, 'Position', [0.2 0.57 0.6 0.35]); % [left bottom width height]

subplot(2, 1, 2);
set(gca, 'Position', [0.2 0.1 0.6 0.35]); % [left bottom width height]

hold off;

% Create the legend entries using your provided logic
legend_entries_long = cell(1, 2);

for idx = 1:2
    if idx == 1
        i = 1;  % First file
    else
        i = length(data_files);  % Last file
    end
    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 6
        prefixPart = nameParts{1};
        datePart = nameParts{6};
        legend_entries_long{idx} = sprintf('%s_%s', prefixPart, datePart);  % e.g., 'Session1_2023'
    end     
end

% Create the legend and make it movable
lgd_short = legend(legend_handles_short, legend_entries_short, 'Interpreter', 'none', 'Box', 'off', 'Location', 'bestoutside'); 
% Set the legend position to 'bestoutside'
set(lgd_short, 'Location', 'bestoutside');  % Position the short block legend outside the plot


lgd_long = legend(legend_handles_long, legend_entries_long, 'Interpreter', 'none', 'Box', 'off', 'Location', 'bestoutside');
% Set the legend position to 'bestoutside'
set(lgd_long, 'Location', 'bestoutside');  % Position the long block legend outside the plot

% Save the figure
newFilename = 'AvgOverAllSessions_LongShort_Subplots.pdf';
exportgraphics(gcf, newFilename, 'ContentType', 'vector');
