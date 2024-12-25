clc; close all; clear

% Load all session data files
data_files = dir('*_EBC_*.mat');

CR_threshold = 0.05;
% Initialize legend handles for short and long blocks
legend_handles_short = [];  % For blue plots
legend_handles_long = [];   % For red plots
% Initialize plot handles
legend_SD_handles = []; % For SD sessions
legend_Control_handles = []; % For Control sessions
sessionDates = [];
x_fill_longLED = [];
x_fill_shortLED = [];
% Initialize figure
figure;

% Predefined axis limits for short and long blocks
global_min_value_short = 0; % Default minimum for short blocks
global_max_value_short = 1; % Default maximum for short blocks
global_min_value_long = 0; % Default minimum for long blocks
global_max_value_long = 1; % Default maximum for long blocks
% Predefined axis limits for short and long blocks
global_min_baseline_short = 0; % Default minimum for short blocks
global_max_baseline_short = 1; % Default maximum for short blocks
global_min_baseline_long = 0; % Default minimum for long blocks
global_max_baseline_long = 1; % Default maximum for long blocks

 
% Define color maps
num_files = length(data_files);
color_map_SD = [linspace(1, 0.5, num_files)', linspace(0, 0, num_files)', linspace(0, 0, num_files)']; % Dim to dark red
color_map_C = [linspace(0.5, 0.2, num_files)', linspace(0.5, 0.2, num_files)', linspace(0.5, 0.2, num_files)']; % Dim to dark grey


% Initialize data containers
all_trials_short_block_SD = [];
all_trials_short_block_Control = [];
all_trials_long_block_SD = [];
all_trials_long_block_Control = [];

% Subplot 1: Short Block
% subplot(2, 1, 1);
% hold on;
% Set the title with custom position
% title('Short Block Trials','Interpreter', 'latex', 'Units', 'normalized', 'Position', [0.5, 1.09, 0]);
% legend_handles = [];
legend_entries = {};
figure('Position', [100, 100, 1200, 800]);

N_SD_short = 0;
N_C_short = 0;
N_SD_long = 0;
N_C_long =  0;

% Loop over each data file
for i = 1:length(data_files)
    load(data_files(i).name);
    numTrials = length(SessionData.RawEvents.Trial);
    % figure('units','centimeters','position',[2 2 24 26])
    numCurves = 0;
    totalFEC_norm = [];
    leg_str = cell(1,1);

    session_SD_short = [];
    session_C_short = [];
    session_SD_long = [];
    session_C_long = [];

    % Containers for long and short block data for this session
    long_block_data = [];
    short_block_data = [];

% Initialize an empty array to store all eyeAreaPixels values
allEyeAreaPixels = [];

    % Loop through each trial to collect eyeAreaPixels data
for trialIdx = 1:numTrials    
   eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
   allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data 
end
% Find the overall maximum value across all collected eyeAreaPixels
overallMax = max(allEyeAreaPixels);

    % Loop over each trial to calculate shifted and normalized FEC
    for ctr_trial = 1:numTrials
        % Skip trial if timeout occurred
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end
        
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
        FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels /overallMax;

        fps = 250; % frames per second, frequency of images
        seconds_before = 0.5;
        seconds_after = 2;

        Frames_before = fps * seconds_before;
        Frames_after = fps * seconds_after;
       % Determine a common time vector for interpolation
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

       isLongBlock = (SessionData.RawEvents.Trial{ctr_trial}.States.LED_Puff_ISI(2) - ...
                       SessionData.RawEvents.Trial{ctr_trial}.States.LED_Puff_ISI(1)) > 0.3;
       
        t_LED = LED_Onset_Zero_Start;
        t_puff = AirPuff_LED_Onset_Aligned_Start;

        is_CR_plus = CR_plus_eval_dev(FEC_led_aligned_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);

% Define the common time vector (e.g., the union of all unique time points or a regular grid)
commonTime = linspace(min(FEC_led_aligned_trimmed), max(FEC_led_aligned_trimmed), 500);  % Adjust 100 to the desired number of points
FEC_norm_interp = interp1(FEC_led_aligned_trimmed, FEC_trimmed, commonTime, 'spline');

      isShortBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - ...
                       SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) < 0.3;



       if isShortBlock
          if ~is_CR_plus
            if SessionData.SleepDeprived
                session_SD_short = [session_SD_short; FEC_norm_interp];
            else
                session_C_short = [session_C_short; FEC_norm_interp];
            end

 
          end
            x_fill_short = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
            x_fill_shortLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];

       end

       if isLongBlock
            if ~is_CR_plus
                if SessionData.SleepDeprived
                    session_SD_long = [session_SD_long; FEC_norm_interp];
                else
                    session_C_long = [session_C_long; FEC_norm_interp];
                end
                x_fill_longLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
                x_fill_long = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
            end

       end

    end
    % Calculate averages for the session
    session_SD_short_avg = mean(session_SD_short, 1, 'omitnan');
    session_C_short_avg = mean(session_C_short, 1, 'omitnan');
    session_SD_long_avg = mean(session_SD_long, 1, 'omitnan');
    session_C_long_avg = mean(session_C_long, 1, 'omitnan');

    global_min_value_short = min([session_SD_short_avg,session_C_short_avg])- 0.02;
    global_max_value_short = max([session_SD_short_avg, session_C_short_avg])+ 0.02;
    global_min_value_long = min([session_SD_long_avg,session_C_long_avg])- 0.02;
    global_max_value_long = max([session_SD_long_avg, session_C_long_avg])+ 0.02;

    session_SD_short_sem = std(session_SD_short, 0, 1, 'omitnan') ./ sqrt(size(session_SD_short, 1));
    session_C_short_sem = std(session_C_short, 0, 1, 'omitnan') ./ sqrt(size(session_C_short, 1));
    session_SD_long_sem = std(session_SD_long, 0, 1, 'omitnan') ./ sqrt(size(session_SD_long, 1));
    session_C_long_sem = std(session_C_long, 0, 1, 'omitnan') ./ sqrt(size(session_C_long, 1));

    if SessionData.SleepDeprived  
        N_SD_long = N_SD_long +1;
    else
        N_C_long = N_C_long+1;
    end 

    % Baseline correction
% Ensure baseline_window_indices matches the size of common_time_vector
baseline_window_indices = (commonTime >= -0.2 & commonTime <= 0);


% Check if session_SD_short_avg is non-empty and matches common_time_vector length
if ~isempty(session_SD_short_avg) && length(session_SD_short_avg) == length(commonTime)
    baseline_SD_short = mean(session_SD_short_avg(baseline_window_indices), 'omitnan');
    session_SD_short_corrected = session_SD_short_avg - baseline_SD_short;
else
    baseline_SD_short = NaN;
    session_SD_short_corrected = NaN(size(commonTime)); % Assign NaN to ensure no plot
end

% Check if session_C_short_avg is non-empty and matches common_time_vector length
if ~isempty(session_C_short_avg) && length(session_C_short_avg) == length(commonTime)
    baseline_C_short = mean(session_C_short_avg(baseline_window_indices), 'omitnan');
    session_C_short_corrected = session_C_short_avg - baseline_C_short;
else
    baseline_C_short = NaN;
    session_C_short_corrected = NaN(size(commonTime)); % Assign NaN to ensure no plot
end

% Repeat for long blocks
if ~isempty(session_SD_long_avg) && length(session_SD_long_avg) == length(commonTime)
    baseline_SD_long = mean(session_SD_long_avg(baseline_window_indices), 'omitnan');
    session_SD_long_corrected = session_SD_long_avg - baseline_SD_long;
else
    baseline_SD_long = NaN;
    session_SD_long_corrected = NaN(size(commonTime)); % Assign NaN to ensure no plot
end

if ~isempty(session_C_long_avg) && length(session_C_long_avg) == length(commonTime)
    baseline_C_long = mean(session_C_long_avg(baseline_window_indices), 'omitnan');
    session_C_long_corrected = session_C_long_avg - baseline_C_long;
else
    baseline_C_long = NaN;
    session_C_long_corrected = NaN(size(commonTime)); % Assign NaN to ensure no plot
end

    global_min_baseline_short = min([session_SD_short_corrected,session_C_short_corrected])- 0.02;
    global_max_baseline_short = max([session_SD_short_corrected, session_C_short_corrected])+ 0.02;
    global_min_baseline_long = min([session_SD_long_corrected,session_C_long_corrected])- 0.02;
    global_max_baseline_long = max([session_SD_long_corrected, session_C_long_corrected])+ 0.02;

   %% Original Short Block Plots
    nexttile(1);
    hold on;
    if ~isempty(session_SD_short_avg) && any(~isnan(session_SD_short_avg))
        % plot(common_time_vector, session_SD_short_avg, 'Color', color_map_SD(i, :), 'LineWidth', 1.5);
        h_SD_short = plot(commonTime, session_SD_short_avg, 'Color', color_map_SD(i, :), 'LineWidth', 1.5);
        legend_SD_handles = [legend_SD_handles, h_SD_short]; % Add handle
        % Add SEM shading for SD
        fill([commonTime, fliplr(commonTime)], ...
         [session_SD_short_avg + session_SD_short_sem, fliplr(session_SD_short_avg - session_SD_short_sem)], ...
         color_map_SD(i, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');


    end
    if ~isempty(session_C_short_avg) && any(~isnan(session_C_short_avg))
        % plot(common_time_vector, session_C_short_avg, 'Color', color_map_C(i, :), 'LineWidth', 1.5);
        h_Control_short = plot(commonTime, session_C_short_avg, 'Color', color_map_C(i, :), 'LineWidth', 1.5);
        legend_Control_handles = [legend_Control_handles, h_Control_short]; % Add handle
        % Add SEM shading for Control
        fill([commonTime, fliplr(commonTime)], ...
         [session_C_short_avg + session_C_short_sem, fliplr(session_C_short_avg - session_C_short_sem)], ...
         color_map_C(i, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');

    end

    title_text(1) = {' '};
    title_text(2) = {'CR$^{+}$ average single session short-block above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{LED}},T_{\rm{AirPuff}})$'};
    title_text(3) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.3f $',CR_threshold)};
    title_text(4) = {' '}; 
    Title_Short = title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',14);


    ylim([0.2, 1]); % Apply default limits if invalid
    xlim([-0.2, 0.6]);
    xlabel('Time From LED Onset (s)', 'Interpreter', 'latex');
    ylabel('Normalized Eyelid Closure', 'Interpreter', 'latex');
    y_fill = [-0.2 -0.2 1 1];
    fill(x_fill_shortLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.06, 'EdgeColor', 'none');
    fill(x_fill_short, y_fill, [0.68, 0.85, 0.9], 'FaceAlpha', 0.05, 'EdgeColor', 'none');  % Short blocks shading
    
    text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.2,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'blue', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');

        % Loop over each data file to collect dates
for j = 1:length(data_files)
    [~, name, ~] = fileparts(data_files(j).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 6
        datePart = nameParts{6};
        sessionDate = datetime(datePart, 'InputFormat', 'yyyyMMdd');  % Convert date part to datetime
        sessionDates = [sessionDates; sessionDate];  % Collect session dates
    end
end
   

    N_short = length(data_files);
    N_long = length(data_files);
    
    % Determine the first and last session dates
    firstSessionDate = min(sessionDates);
    lastSessionDate = max(sessionDates);    

    %% Baseline-Corrected Short Block Plots
    nexttile(2);
    hold on;
    if ~isempty(session_SD_short_corrected) && any(~isnan(session_SD_short_corrected))
        plot(commonTime, session_SD_short_corrected, 'Color', color_map_SD(i, :), 'LineWidth', 1.5);
        % Add SEM shading for SD
        fill([commonTime, fliplr(commonTime)], ...
         [session_SD_short_corrected + session_SD_short_sem, fliplr(session_SD_short_corrected - session_SD_short_sem)], ...
         color_map_SD(i, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');

    end
    if ~isempty(session_C_short_corrected) && any(~isnan(session_C_short_corrected))
        plot(commonTime, session_C_short_corrected, 'Color', color_map_C(i, :), 'LineWidth', 1.5);
        % Add SEM shading for Control
        fill([commonTime, fliplr(commonTime)], ...
         [session_C_short_corrected + session_C_short_sem, fliplr(session_C_short_corrected - session_C_short_sem)], ...
         color_map_C(i, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');

    end
    title('Baseline-Adjusted CR$^{+}$ Short Blocks Averages', 'Interpreter', 'latex');

    y_fill = [-0.2 -0.2 1 1];
    fill(x_fill_shortLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.06, 'EdgeColor', 'none');
    fill(x_fill_short, y_fill, [0.68, 0.85, 0.9], 'FaceAlpha', 0.05, 'EdgeColor', 'none');  % Short blocks shading
    text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.2,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'blue', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');


    ylim([-0.02, 0.7]); % Apply default limits if invalid
    xlim([-0.2, 0.6]);
    %% Original Long Block Plots
    nexttile(3);
    hold on;
    if ~isempty(session_SD_long_avg) && any(~isnan(session_SD_long_avg))
        % plot(common_time_vector, session_SD_long_avg, 'Color', color_map_SD(i, :), 'LineWidth', 1.5);
        h_SD_long = plot(commonTime, session_SD_long_avg, 'Color', color_map_SD(i, :), 'LineWidth', 1.5);
        legend_SD_handles = [legend_SD_handles, h_SD_long]; % Add handle
         fill([commonTime, fliplr(commonTime)], ...
         [session_SD_long_avg + session_SD_long_sem, fliplr(session_SD_long_avg - session_SD_long_sem)], ...
         color_map_SD(i, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');


    end
    if ~isempty(session_C_long_avg) && any(~isnan(session_C_long_avg))
        % plot(common_time_vector, session_C_long_avg, 'Color', color_map_C(i, :), 'LineWidth', 1.5);
        h_Control_long = plot(commonTime, session_C_long_avg, 'Color', color_map_C(i, :), 'LineWidth', 1.5);
        legend_Control_handles = [legend_Control_handles, h_Control_long]; % Add handle
         fill([commonTime, fliplr(commonTime)], ...
         [session_C_long_avg + session_C_long_sem, fliplr(session_C_long_avg - session_C_long_sem)], ...
         color_map_C(i, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');

    end

    % Add LED onset shading
    y_fill = [0 0 1 1];
    fill(x_fill_longLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none');

    % Add AirPuff shading
    y_fill = [0 0 1 1];
    fill(x_fill_long, y_fill, [0.5, 1.0, 0.5], 'FaceAlpha', 0.15, 'EdgeColor', 'none');  % Long blocks shading
    
    text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.4,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'green', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');

    title_text(1) = {' '};
    title_text(2) = {'CR$^{+}$ average single session long-block above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{LED}},T_{\rm{AirPuff}})$'};
    title_text(3) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.3f $',CR_threshold)};
    title_text(4) = {' '}; 
    Title_Short = title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',14);

    ylim([0.2, 1]);
    xlim([-0.2, 0.6]);
    xlabel('Time From LED Onset (s)', 'Interpreter', 'latex');
    ylabel('Normalized Eyelid Closure', 'Interpreter', 'latex');
    % legend({'SD Sessions', 'Control Sessions'}, 'Location', 'bestoutside', 'Interpreter', 'latex', 'Box', 'off');
    % Add legend only if both SD and Control handles are available
    if ~isempty(legend_SD_handles) && ~isempty(legend_Control_handles)
        legend([legend_SD_handles(1), legend_Control_handles(1)], ...
            {'SD Sessions', 'Control Sessions'}, ...
            'Location', 'bestoutside', 'Interpreter', 'latex', 'Box', 'off');
    else
        warning('No valid data for either SD or Control sessions in this subplot.');
    end    
    %% Baseline-Corrected Long Block Plots
    nexttile(4);
    hold on;
    if ~isempty(session_SD_long_corrected) && any(~isnan(session_SD_long_corrected))
        plot(commonTime, session_SD_long_corrected, 'Color', color_map_SD(i, :), 'LineWidth', 1.5);
        fill([commonTime, fliplr(commonTime)], ...
         [session_SD_long_corrected + session_SD_long_sem, fliplr(session_SD_long_corrected - session_SD_long_sem)], ...
         color_map_SD(i, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');

    end
    if ~isempty(session_C_long_corrected) && any(~isnan(session_C_long_corrected))
        plot(commonTime, session_C_long_corrected, 'Color', color_map_C(i, :), 'LineWidth', 1.5);
        fill([commonTime, fliplr(commonTime)], ...
         [session_C_long_corrected + session_C_long_sem, fliplr(session_C_long_corrected - session_C_long_sem)], ...
         color_map_C(i, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');

    end
    title('Baseline-Adjusted CR$^{+}$ Long Blocks Averages', 'Interpreter', 'latex');
    ylim([-0.02, 0.7]);
    xlim([-0.2, 0.6]);
    y_fill = [-0.2 -0.2 1 1];
    fill(x_fill_longLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none');
    fill(x_fill_long, y_fill, [0.5, 1.0, 0.5], 'FaceAlpha', 0.15, 'EdgeColor', 'none');  % Long blocks shading


    % Apply ticks outside to all subplots
    for ax = findall(gcf, 'Type', 'axes')
        set(ax, 'TickDir', 'out');
    end
end

 % Generate session date information
session_date_info = ['SD Session (n = ', num2str(N_SD_long), ')', newline ...
                     'Control Session (n = ', num2str(N_C_long), ')', newline, ...
                     'Sessions from: ', datestr(firstSessionDate, 'mm/dd/yyyy'), newline, ...
                     ' to ', datestr(lastSessionDate, 'mm/dd/yyyy')];

% Add session date information as a text box
annotation('textbox', [0.43, 0.20, 0.3, 0.65], 'String', session_date_info, ...
    'EdgeColor', 'none', 'Interpreter', 'latex', 'FontSize', 10, 'FontName', 'Times New Roman');



% Save the figure
exportgraphics(gcf, '02_CR_negative_fractions_Mean_SD_Control_Sessios.pdf', 'ContentType', 'vector');

