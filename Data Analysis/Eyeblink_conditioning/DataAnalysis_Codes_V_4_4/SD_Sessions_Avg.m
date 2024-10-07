clc; close all; clear

% Load all session data files
data_files = dir('*_EBC_*.mat');
% data_files = dir('E3VT_EBC_V_3_10_20240826_171045.mat');

FEC_times_trimmed = [];


% Initialize accumulators for all sessions
global_short_block_data = [];
global_long_block_data = [];

% Loop over each data file
for i = 1:length(data_files)
    
    % Initialize pooled data containers for short and long blocks
    pooled_short_block_data = [];
    pooled_long_block_data = [];
    
    % Containers for long and short block data for this session
    long_block_data = [];
    short_block_data = [];

    % Initialize figure
    % figure;
    % figure('Position', [100, 100, 600, 400]); % Width = 600, Height = 400
    % 
    % hold on;  % Ensure all plots are overlaid on the same figure

    load(data_files(i).name);
    numTrials = length(SessionData.RawEvents.Trial);

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

    % Containers for long and short block data for this session
    long_block_data = [];
    short_block_data = [];

    % Loop over each trial to calculate shifted and normalized FEC
    for ctr_trial = 1:numTrials
        
        %  Skip trials with CheckEyeOpenTimeout
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end

        % Get trial data for LED onset, puff, and timings
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        LED_Onset_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End;
        AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
        AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;

        % Align times to LED onset
        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        LED_Onset_Zero_End = LED_Onset_End - LED_Onset;
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

        % Block type: Long or Short
        % isLongBlock = LED_Onset_Zero_Start - AirPuff_LED_Onset_Aligned_Start > 0.3;
        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;
        % Accumulate data based on block type
        if isLongBlock
            long_block_data = [long_block_data; FEC_trimmed]; % Add to long block data
            % Initialize airpuff shade colors for both long and short blocks
            % longBlockAirPuffColor = [0.5, 1.0, 0.5]; % Light green for long trials
            % % Add shade areas for long blocks
            % x_fill = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
            % y_fill = [0 0 1 1];
            % fill(x_fill, y_fill, longBlockAirPuffColor, 'FaceAlpha', 0.65, 'EdgeColor', 'none');  % Long blocks shading
           
        else
            short_block_data = [short_block_data; FEC_trimmed]; % Add to short block data
            % shortBlockAirPuffColor = [0.5, 0.5, 1.0]; % Light blue for short trials
            % % % Add shade areas for short blocks
            % x_fill = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
            % y_fill = [0 0 1 1];
            % fill(x_fill, y_fill, shortBlockAirPuffColor, 'FaceAlpha', 0.35, 'EdgeColor', 'none');  % Short blocks shading
            

        end
% shortBlockAirPuffColor = [0.5, 0.5, 1.0]; % Light blue for short trials
% % Add shade areas for short blocks
% x_fill = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
% y_fill = [0 0 1 1];
% fill(x_fill, y_fill, shortBlockAirPuffColor, 'FaceAlpha', 0.35, 'EdgeColor', 'none');  % Short blocks shading


    end
    
    % Add session data to pooled data
    pooled_short_block_data = [pooled_short_block_data; short_block_data];
    pooled_long_block_data = [pooled_long_block_data; long_block_data];


%% Plotting

% Calculate the average and SEM for short and long block data
average_short_block = mean(pooled_short_block_data, 1, 'omitnan');
average_long_block = mean(pooled_long_block_data, 1, 'omitnan');

% Accumulate the averages across sessions
global_short_block_data = [global_short_block_data; average_short_block];
global_long_block_data = [global_long_block_data; average_long_block];

sem_short_block = std(pooled_short_block_data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(pooled_short_block_data), 1));
sem_long_block = std(pooled_long_block_data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(pooled_long_block_data), 1));

% Find the global minimum and maximum between both curves
% global_min_value = min([min(average_short_block), min(average_long_block)]);
% global_max_value = max([max(average_short_block), max(average_long_block)]);
% 
% 
% 
% % Calculate the average and SEM for short and long block data
% average_short_block = mean(pooled_short_block_data, 1, 'omitnan');
% average_long_block = mean(pooled_long_block_data, 1, 'omitnan');
% 
% sem_short_block = std(pooled_short_block_data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(pooled_short_block_data), 1));
% sem_long_block = std(pooled_long_block_data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(pooled_long_block_data), 1));
% 
% 
% % Plot averages and SEM outside the trial loop
% hold on;
% 
% 
% global_max_value = global_max_value + 0.02;
% global_min_value = global_min_value - 0.02;
% 
% % Plot short block average (blue)
% plot(FEC_times_trimmed, average_short_block, 'b', 'LineWidth', 1.5);
% fill([FEC_times_trimmed, fliplr(FEC_times_trimmed)], [average_short_block + sem_short_block, fliplr(average_short_block - sem_short_block)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
% 
% % Plot long block average (green)
% plot(FEC_times_trimmed, average_long_block, 'g', 'LineWidth', 1.5);
% fill([FEC_times_trimmed, fliplr(FEC_times_trimmed)], [average_long_block + sem_long_block, fliplr(average_long_block - sem_long_block)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
% 
% % % Add LED onset shading
% x_fill = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
% y_fill = [0 0 1 1];
% fill(x_fill, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none');
% 
% % Configure plot
% ylim([global_min_value global_max_value]);
% xlim([-0.2 0.6]);
% xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 14);
% ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 14);
% % Set tick marks outward
% set(gca, 'TickDir', 'out');
% 
% 
% text(0,global_max_value, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 11, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
% text(0.2,global_max_value, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 11, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
% text(0.4,global_max_value, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 11, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
% 
% % Plot short block average (blue) and store handle
% h_short = plot(FEC_times_trimmed, average_short_block, 'b', 'LineWidth', 1.5);
% 
% % Plot long block average (red) and store handle
% h_long = plot(FEC_times_trimmed, average_long_block, 'g', 'LineWidth', 1.5);
% 
% % Add legend
% legend([h_short, h_long],{'Short Block Avg', 'Long Block Avg'}, 'Interpreter', 'latex', 'fontsize', 14,'Box', 'off', 'Location', 'bestoutside');
% 
% title_text(1) = {'\vspace{2cm} average of all trials for each session \\'};
% % title_text(2) = {'CR$^{+}$ long-block trials above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{0.2(s)}},T_{\rm{AirPuff}})$'};
% % title_text(3) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $',CR_threshold)};
% % title_text(4) = {''}; 
% title_text(2) = {' '};
% title(title_text ,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
% 
% 
%     [~, name, ~] = fileparts(data_files(i).name);
%     nameParts = split(name, '_');
%     if length(nameParts) >= 5
%         prefixPart = nameParts{1}; % First string before the first underscore
%         datePart = nameParts{6}; % Date part
%         newFilename = sprintf('%s_Avg_AllFECTrials_Short_Long_%s.pdf', prefixPart, datePart);
%         exportgraphics(gcf, newFilename, 'ContentType', 'vector');
%     else
%         error('Filename does not have the expected format');
%     end

end
% Calculate the overall average and SEM across all sessions
global_average_short_block = mean(global_short_block_data, 1, 'omitnan');
global_average_long_block = mean(global_long_block_data, 1, 'omitnan');

global_sem_short_block = std(global_short_block_data, 0, 1, 'omitnan') ./ sqrt(size(global_short_block_data, 1));
global_sem_long_block = std(global_long_block_data, 0, 1, 'omitnan') ./ sqrt(size(global_long_block_data, 1));



figure;
hold on;

% % Add LED onset shading
x_fill = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
y_fill = [0 0 1 1];
fill(x_fill, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none');


longBlockAirPuffColor = [0.5, 1.0, 0.5]; % Light green for long trials
% Add shade areas for long blocks
x_fill_long = [0.4, 0.449, 0.449, 0.4];
x_fill_short = [0.2, 0.249, 0.249, 0.2];
y_fill = [0 0 1 1];
fill(x_fill_long, y_fill, longBlockAirPuffColor, 'FaceAlpha', 0.65, 'EdgeColor', 'none');  % Long blocks shading
shortBlockAirPuffColor = [0.5, 0.5, 1.0]; % Light blue for short trials
% Add shade areas for short blocks
fill(x_fill_short, y_fill, shortBlockAirPuffColor, 'FaceAlpha', 0.35, 'EdgeColor', 'none');  % Short blocks shading



% Plot global short block average (blue)
SD_S_h = plot(FEC_times_trimmed, global_average_short_block, 'b', 'LineWidth', 1.5);
fill([FEC_times_trimmed, fliplr(FEC_times_trimmed)], ...
     [global_average_short_block + global_sem_short_block, fliplr(global_average_short_block - global_sem_short_block)], ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Plot global long block average (green)
SD_L_h = plot(FEC_times_trimmed, global_average_long_block, 'g', 'LineWidth', 1.5);
fill([FEC_times_trimmed, fliplr(FEC_times_trimmed)], ...
     [global_average_long_block + global_sem_long_block, fliplr(global_average_long_block - global_sem_long_block)], ...
     'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');


% Configure plot
ylim([0.2 1]);
xlim([-0.2 0.6]);
xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 14);
ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 14);
% Set tick marks outward
set(gca, 'TickDir', 'out');


text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 11, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
text(0.2,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 11, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
text(0.4,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 11, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');

% Add labels and title
xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 14);
ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 14);
% Add legend
legend([SD_S_h, SD_L_h],{'Short Block Avg', 'Long Block Avg'}, 'Interpreter', 'latex', 'fontsize', 14,'Box', 'off', 'Location', 'bestoutside');


title_text(1) = {'\vspace{2cm} Global Average Across All SD-Sessions \\'};
% % title_text(2) = {'CR$^{+}$ long-block trials above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{0.2(s)}},T_{\rm{AirPuff}})$'};
% % title_text(3) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $',CR_threshold)};
% title_text(4) = {''}; 
title_text(2) = {' '};
title(title_text ,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
% 
% title('Global Average Across All SD-Sessions', 'Interpreter', 'latex', 'FontSize', 17);

hold off;
% Save the figure
newFilename = 'Avg_AllSD_Sessions.pdf';
exportgraphics(gcf, newFilename, 'ContentType', 'vector');
