clc; close all; clear

% Load all session data files
data_files = dir('*_EBC_*.mat');

% Initialize pooled data containers for short and long blocks
pooled_short_block_data = [];
pooled_long_block_data = [];
% Initialize variables
FEC_times_trimmed = [];  % Predefine outside loop to avoid undefined variable issue

% Initialize figure
figure;

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
        FEC_times_trimmed = FEC_led_aligned(start_idx:stop_idx);  % Update FEC_times_trimmed correctly

        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - LED_Onset;
        
        % Identify if the trial is short or long and store the data
        if length(FEC_trimmed) == length(FEC_times_trimmed)
            if SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3
                % Long block
                if isempty(long_block_data) || size(FEC_trimmed, 2) == size(long_block_data, 2)
                    long_block_data = [long_block_data; FEC_trimmed];
                    verticalLineColor = 'g'; % Green for long trials
                end
            else
                % Short block
                if isempty(short_block_data) || size(FEC_trimmed, 2) == size(short_block_data, 2)
                    short_block_data = [short_block_data; FEC_trimmed];
                    verticalLineColor = 'b'; % Blue for short trials
                end
            end
        end

        % Plot the vertical line for AirPuff
        line([LED_Puff_ISI_end, LED_Puff_ISI_end], [0 1], 'Color', verticalLineColor, 'LineWidth', 1, 'LineStyle', '--');
    end

    % Add session's data to the pooled data for short and long blocks
    pooled_short_block_data = [pooled_short_block_data; short_block_data];
    pooled_long_block_data = [pooled_long_block_data; long_block_data];

    
end
% Calculate the average and SEM for short and long block data across all sessions
average_short_block = mean(pooled_short_block_data, 1, 'omitnan');
average_long_block = mean(pooled_long_block_data, 1, 'omitnan');

sem_short_block = std(pooled_short_block_data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(pooled_short_block_data), 1));
sem_long_block = std(pooled_long_block_data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(pooled_long_block_data), 1));

% Plot the averages and SEM as shaded regions
hold on;
        
% Plot short block average (blue)
plot(FEC_times_trimmed, average_short_block, 'b', 'LineWidth', 1.5);
fill([FEC_times_trimmed, fliplr(FEC_times_trimmed)], [average_short_block + sem_short_block, fliplr(average_short_block - sem_short_block)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Plot long block average (green)
plot(FEC_times_trimmed, average_long_block, 'g', 'LineWidth', 1.5);
fill([FEC_times_trimmed, fliplr(FEC_times_trimmed)], [average_long_block + sem_long_block, fliplr(average_long_block - sem_long_block)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');


% Add vertical line for LED onset
line([0, 0], [0 1], 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1, 'LineStyle', '--');  % Grey dashed line for LED onset

% Configure plot
ylim([0 1]);
xlim([-0.2 0.6]);
xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 12);
ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
% Set tick marks outward
set(gca, 'TickDir', 'out');

text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
text(0.2,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
text(0.4,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');

% Plot short block average (blue) and store handle
h_short = plot(FEC_times_trimmed, average_short_block, 'b', 'LineWidth', 1.5);

% Plot long block average (red) and store handle
h_long = plot(FEC_times_trimmed, average_long_block, 'g', 'LineWidth', 1.5);

% Add legend
legend([h_short, h_long],{'Short Block Avg', 'Long Block Avg'}, 'Interpreter', 'latex', 'fontsize', 14,'Box', 'off', 'Location', 'bestoutside');

% Save the figure
newFilename = 'AvgOverAllSessions_LongShort_Pool.pdf';
exportgraphics(gcf, newFilename, 'ContentType', 'vector');
