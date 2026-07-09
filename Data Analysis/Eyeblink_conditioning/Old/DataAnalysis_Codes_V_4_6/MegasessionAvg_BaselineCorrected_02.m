clc; close all; clear

% Load all session data files
data_files = dir('*_EBC_*.mat');
CR_threshold = 0.05;

% Initialize legend handles for short and long blocks
legend_handles_short = [];  % For blue plots
legend_handles_long = [];   % For red plots

% CR_threshold = 0.02;
% Initialize legend handles for short and long blocks
legend_handles_short = [];  % For blue plots
legend_handles_long = [];   % For red plots
% Clear and initialize legend_entries_short to ensure it's a cell array

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


sessionDates = [];
legend_entries_short = {};  % Explicitly set it as a cell array
legend_entries_long = {};
x_fill_longLED = [];
x_fill_shortLED = [];
% Increase figure size for better layout control
% figure;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Subplot 1: Short Block
%%
% subplot(2, 2, 1);
% hold on;

figure('Position', [100, 100, 1200, 800]);
% Set the title with custom position

% legend_handles = [];
legend_entries = {};

% Initialize containers for all sessions
all_sessions_short_block_data = [];
all_sessions_long_block_data = [];


% Initialize containers for all sessions
SD_sessions_short_block_data = [];
SD_sessions_long_block_data = [];
C_sessions_short_block_data = [];
C_sessions_long_block_data = [];

SD_combined_sem = [];
C_combined_sem = [];

N_SD_short = 0;
N_C_short = 0;
N_SD_long = 0;
N_C_long =  0;


% Initialize containers for short block data (all trials across all sessions)
all_trials_short_block_SD = [];
all_trials_short_block_Control = [];

% Loop over each data file
for i = 1:length(data_files)
    
    t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile;
    hold on;

    load(data_files(i).name);
    numTrials = length(SessionData.RawEvents.Trial);
    numCurves = 0;
    totalFEC_norm = [];
    

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
        FEC_times_trimmed = FEC_led_aligned(start_idx:stop_idx);
        isShortBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - ...
                       SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) <= 0.3;
        t_LED = LED_Onset_Zero_Start;
        t_puff = AirPuff_LED_Onset_Aligned_Start;
        
        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;


        is_CR_plus = CR_plus_eval_dev(FEC_times_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);
        % Define the common time vector (e.g., the union of all unique time points or a regular grid)
        commonTime = linspace(min(FEC_led_aligned_trimmed), max(FEC_led_aligned_trimmed), 500);  % Adjust 100 to the desired number of points

        FEC_norm_interp = interp1(FEC_led_aligned_trimmed, FEC_trimmed, commonTime, 'spline');
        % Aggregate data for short blocks
        if isShortBlock 
            if is_CR_plus
            if SessionData.SleepDeprived
                all_trials_short_block_SD = [all_trials_short_block_SD; FEC_norm_interp];
                N_SD_short = N_SD_short + 1;
            else
                all_trials_short_block_Control = [all_trials_short_block_Control; FEC_norm_interp];
                N_C_short = N_C_short + 1;
            end

            x_fill_short = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
            end
        end
    end
    
end

% Calculate averages for SD and Control trials
SD_FEC_norm_avg_short = mean(all_trials_short_block_SD, 1, 'omitnan');
C_FEC_norm_avg_short = mean(all_trials_short_block_Control, 1, 'omitnan');

global_min_value = min([SD_FEC_norm_avg_short,C_FEC_norm_avg_short]);
global_max_value = max([SD_FEC_norm_avg_short, C_FEC_norm_avg_short]);

% Define baseline window (adjust indices based on sampling rate)
baseline_window_indices = (common_time_vector >= -0.2 & common_time_vector <= 0); 

% Calculate baseline mean for SD and Control trials
baseline_SD_short = mean(SD_FEC_norm_avg_short(baseline_window_indices));
baseline_C_short = mean(C_FEC_norm_avg_short(baseline_window_indices));

% Subtract baseline from each curve
SD_FEC_norm_avg_short_corrected = SD_FEC_norm_avg_short - baseline_SD_short;
C_FEC_norm_avg_short_corrected = C_FEC_norm_avg_short - baseline_C_short;

% Standard error calculation
SD_std_short_block = std(all_trials_short_block_SD, 0, 1, 'omitnan');
C_std_short_block = std(all_trials_short_block_Control, 0, 1, 'omitnan');

combined_SD_sem_short = SD_std_short_block ./ sqrt(N_SD_short);
combined_C_sem_short = C_std_short_block ./ sqrt(N_C_short);


% Plot results for short blocks
% figure;
hold on;


global_max_value = global_max_value + 0.02;
global_min_value = global_min_value - 0.02;


    % Plot the mean line for short blocks
    h1 = plot(commonTime, SD_FEC_norm_avg_short, 'r', 'LineWidth', 1.5);
    h2 = plot(commonTime, C_FEC_norm_avg_short , 'b', 'LineWidth', 1.5);

    % Loop over each data file to collect dates
for i = 1:length(data_files)
    [~, name, ~] = fileparts(data_files(i).name);
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
    
    legend({['SD Session Trials (n = ', num2str(N_SD_short), ')'], ...
            ['Control Session Trials (n = ', num2str(N_C_short), ')', newline, ...
             '(', datestr(firstSessionDate, 'mm/dd/yyyy'), ' to ', datestr(lastSessionDate, 'mm/dd/yyyy'), ')']}, ...
        'Location', 'bestoutside', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 10, 'Box', 'off');


   
    % Shading the SEM region around the mean
    fill([commonTime, fliplr(commonTime)], ...
         [SD_FEC_norm_avg_short + combined_SD_sem_short, fliplr(SD_FEC_norm_avg_short - combined_SD_sem_short)], ...
         'r', 'FaceAlpha', 0.4, 'EdgeColor', 'none','HandleVisibility', 'off'); 

    fill([commonTime, fliplr(commonTime)], ...
         [C_FEC_norm_avg_short + combined_C_sem_short, fliplr(C_FEC_norm_avg_short - combined_C_sem_short)], ...
         [0.2 0.2 0.6], 'FaceAlpha', 0.4, 'EdgeColor', 'none','HandleVisibility', 'off'); 


    % % Add LED onset shading
    y_fill = [0 0 1 1];
    x_fill_shortLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
    fill(x_fill_shortLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none','HandleVisibility', 'off');

    % Add AirPuff shading
    y_fill = [0 0 1 1];
    fill(x_fill_short, y_fill, [0.68, 0.85, 0.9], 'FaceAlpha', 0.35, 'EdgeColor', 'none','HandleVisibility', 'off');  % Short blocks shading

    text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.2,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'blue', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    
    title_text(1) = {' '};
    title_text(2) = {'CR$^{+}$ pooled trials average short-block above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{LED}},T_{\rm{AirPuff}})$'};
    title_text(3) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.3f $',CR_threshold)};
    % title_text(4) = {' '}; 
    Title_Short = title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',14);
    set(gca, 'TickDir', 'out'); % Moves ticks to the outside
    % Configure plot
    ylim([global_min_value global_max_value]);
    xlim([-0.2 0.6]);
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
% subplot(2, 2, 2);
% hold on;
    nexttile;
    hold on;

title('Baseline-Adjusted CR$^{+}$ Short Blocks', 'Interpreter', 'latex', 'FontSize', 14);
plot(commonTime, SD_FEC_norm_avg_short_corrected, 'r', 'LineWidth', 1.5);
plot(commonTime, C_FEC_norm_avg_short_corrected, 'b', 'LineWidth', 1.5);

global_min_value_baseline = min([SD_FEC_norm_avg_short_corrected,C_FEC_norm_avg_short_corrected]);
global_max_value_baseline = max([SD_FEC_norm_avg_short_corrected, C_FEC_norm_avg_short_corrected]);


global_max_value_baseline = global_max_value_baseline + 0.02;
global_min_value_baseline = global_min_value_baseline - 0.02;



fill([commonTime, fliplr(commonTime)], ...
     [SD_FEC_norm_avg_short_corrected + combined_SD_sem_short, ...
      fliplr(SD_FEC_norm_avg_short_corrected - combined_SD_sem_short)], ...
      'r', 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');

fill([commonTime, fliplr(commonTime)], ...
     [C_FEC_norm_avg_short_corrected + combined_C_sem_short, ...
      fliplr(C_FEC_norm_avg_short_corrected - combined_C_sem_short)], ...
      'b', 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off'); 
    
    % % Add LED onset shading
    y_fill = [-0.2 -0.2 1 1];
    x_fill_shortLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
    fill(x_fill_shortLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none','HandleVisibility', 'off');

    % Add AirPuff shading
    y_fill = [-0.2 -0.2 1 1];
    fill(x_fill_short, y_fill, [0.68, 0.85, 0.9], 'FaceAlpha', 0.35, 'EdgeColor', 'none','HandleVisibility', 'off');  % Short blocks shading

    text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.2,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'blue', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    
    set(gca, 'TickDir', 'out'); % Moves ticks to the outside
    % Configure plot
    ylim([global_min_value_baseline global_max_value_baseline]);
    xlim([-0.2 0.6]);
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subplot 2: Long Block


all_trials_long_block_SD = [];
all_trials_long_block_Control = [];

% Loop over each data file again for long blocks
for i = 1:length(data_files)

    load(data_files(i).name);
    numTrials = length(SessionData.RawEvents.Trial);
    % figure('units','centimeters','position',[2 2 24 26])
    numCurves = 0;
    totalFEC_norm = [];
    leg_str = cell(1,1);

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

        t_LED = LED_Onset_Zero_Start;
        t_puff = AirPuff_LED_Onset_Aligned_Start;
        
        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;


        is_CR_plus = CR_plus_eval_dev(FEC_times_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);

        % Define the common time vector (e.g., the union of all unique time points or a regular grid)
        commonTime = linspace(min(FEC_led_aligned_trimmed), max(FEC_led_aligned_trimmed), 500);  % Adjust 100 to the desired number of points
        
        % CR_plus_matrix = CR_plus_eval_all_v3(FEC_times_trimmed, FEC_trimmed, t1_B1, t2_B2, t1_CR, t2_CR, CR_threshold);
        
        % if CR_plus_matrix
            numCurves = numCurves+1;
            % Interpolate FEC_norm to the common time vector
            FEC_norm_interp = interp1(FEC_led_aligned_trimmed, FEC_trimmed, commonTime, 'spline');
            
            LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - LED_Onset;
         % Aggregate data for short blocks
        if isLongBlock 
            if is_CR_plus
            if SessionData.SleepDeprived
                all_trials_long_block_SD = [all_trials_long_block_SD; FEC_norm_interp];
                N_SD_long = N_SD_long + 1;
            else
                all_trials_long_block_Control = [all_trials_long_block_Control; FEC_norm_interp];
                N_C_long = N_C_long + 1;
            end

            x_fill_long = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
            end
        end
 
    end
end



% Calculate averages for SD and Control trials
SD_FEC_norm_avg_long = mean(all_trials_long_block_SD, 1, 'omitnan');
C_FEC_norm_avg_long = mean(all_trials_long_block_Control, 1, 'omitnan');

global_min_value = min([SD_FEC_norm_avg_long,C_FEC_norm_avg_long]);
global_max_value = max([SD_FEC_norm_avg_long, C_FEC_norm_avg_long]);

% Define baseline window (adjust indices based on sampling rate)
baseline_window_indices = (common_time_vector >= -0.2 & common_time_vector <= 0); 

% Calculate baseline mean for SD and Control trials
baseline_SD_long = mean(SD_FEC_norm_avg_long(baseline_window_indices));
baseline_C_long = mean(C_FEC_norm_avg_long(baseline_window_indices));

% Subtract baseline from each curve
SD_FEC_norm_avg_long_corrected = SD_FEC_norm_avg_long - baseline_SD_long;
C_FEC_norm_avg_long_corrected = C_FEC_norm_avg_long - baseline_C_long;

% Standard error calculation
SD_std_long_block = std(all_trials_long_block_SD, 0, 1, 'omitnan');
C_std_long_block = std(all_trials_long_block_Control, 0, 1, 'omitnan');

combined_SD_sem_long = SD_std_long_block ./ sqrt(N_SD_long);
combined_C_sem_long = C_std_long_block ./ sqrt(N_C_long);



%%

nexttile;
hold on;

    % Plot the mean line for short blocks
    h3 = plot(commonTime, SD_FEC_norm_avg_long,'r' , 'LineWidth', 1.5);  % Blue for short blocks
    h4 = plot(commonTime, C_FEC_norm_avg_long , 'b', 'LineWidth', 1.5);
    
 
    % Shading the SEM region around the mean

    fill([commonTime, fliplr(commonTime)], ...
         [SD_FEC_norm_avg_long + combined_SD_sem_long, fliplr(SD_FEC_norm_avg_long - combined_SD_sem_long)], ...
          'r', 'FaceAlpha', 0.4, 'EdgeColor', 'none','HandleVisibility', 'off');

    fill([commonTime, fliplr(commonTime)], ...
         [C_FEC_norm_avg_long + combined_C_sem_long, fliplr(C_FEC_norm_avg_long - combined_C_sem_long)], ...
          'b', 'FaceAlpha', 0.4, 'EdgeColor', 'none','HandleVisibility', 'off');

    
    % Add LED onset shading
    y_fill = [0 0 1 1];
    x_fill_longLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
    fill(x_fill_longLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none','HandleVisibility', 'off');
                
        
    % Add AirPuff shading
    y_fill = [0 0 1 1];
    fill(x_fill_long, y_fill, [0.56, 0.93, 0.56], 'FaceAlpha', 0.65, 'EdgeColor', 'none','HandleVisibility', 'off');  % Long blocks shading
    
    
    % Configure plot
    ylim([global_min_value global_max_value]);
    
    xlim([-0.2 0.6]);
    xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 12);
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
    % legend(legend_handles, legend_entries, 'Interpreter', 'latex', 'location', 'northeast', 'Box', 'off');
    text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.4,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'green', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    
    title_text(1) = {' '};
    title_text(2) = {'CR$^{+}$ pooled trials average long-block above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{LED}},T_{\rm{AirPuff}})$'};
    title_text(3) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $',CR_threshold)};
    
    Title_Short = title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',14);
    set(gca, 'TickDir', 'out'); % Moves ticks to the outside
    hold off;


    legend({['SD Session Trials (n = ', num2str(N_SD_long), ')'], ...
            ['Control Session Trials (n = ', num2str(N_C_long), ')', newline, ...
             '(', datestr(firstSessionDate, 'mm/dd/yyyy'), ' to ', datestr(lastSessionDate, 'mm/dd/yyyy'), ')']}, ...
        'Location', 'bestoutside', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 10, 'Box', 'off');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subplot(2, 2, 4);
% hold on;
    nexttile;
    hold on;

title('Baseline-Adjusted CR$^{+}$ Long Blocks', 'Interpreter', 'latex', 'FontSize', 14);    

plot(commonTime, SD_FEC_norm_avg_long_corrected, 'r', 'LineWidth', 1.5);
plot(commonTime, C_FEC_norm_avg_long_corrected, 'b', 'LineWidth', 1.5);

global_min_value_baseline = min([SD_FEC_norm_avg_long_corrected,C_FEC_norm_avg_long_corrected]);
global_max_value_baseline = max([SD_FEC_norm_avg_long_corrected, C_FEC_norm_avg_long_corrected]);


global_max_value_baseline = global_max_value_baseline + 0.02;
global_min_value_baseline = global_min_value_baseline - 0.02;

fill([commonTime, fliplr(commonTime)], ...
     [SD_FEC_norm_avg_long_corrected + combined_SD_sem_long, ...
      fliplr(SD_FEC_norm_avg_long_corrected - combined_SD_sem_long)], ...
      'r', 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');

fill([commonTime, fliplr(commonTime)], ...
     [C_FEC_norm_avg_long_corrected + combined_C_sem_long, ...
      fliplr(C_FEC_norm_avg_long_corrected - combined_C_sem_long)], ...
      'b', 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    % Configure plot
    ylim([global_min_value_baseline global_max_value_baseline]);
    xlim([-0.2 0.6]);
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);

    % Add LED onset shading
    y_fill = [-0.1 -0.1 0.5 0.5];
    x_fill_longLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
    fill(x_fill_longLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none','HandleVisibility', 'off');
                
    set(gca, 'TickDir', 'out'); % Moves ticks to the outside    
    % Add AirPuff shading
    y_fill = [-0.1 -0.1 0.5 0.5];
    fill(x_fill_long, y_fill, [0.56, 0.93, 0.56], 'FaceAlpha', 0.65, 'EdgeColor', 'none','HandleVisibility', 'off');  % Long blocks shading

    % Save the figure
    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 6
        prefixPart = nameParts{1};
        % Specify the desired file format as 'pdf'
        fileFormat = '.pdf';  % PDF format
          
        newFilename = sprintf('%s_MegaSession_CR_SD_Control', prefixPart); 

        newFilename = [newFilename, fileFormat];
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end
