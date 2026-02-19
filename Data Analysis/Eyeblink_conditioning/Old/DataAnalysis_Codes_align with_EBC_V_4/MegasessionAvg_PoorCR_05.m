clc; close all; clear

% Load all session data files
data_files = dir('*_EBC_*.mat');
CR_threshold = 0.02;

good_CR_threshold = 0.05;
poor_CR_threshold = 0.02;


% Initialize legend handles for short and long blocks
legend_handles_short = [];  % For blue plots
legend_handles_long = [];   % For red plots

% CR_threshold = 0.02;
% Initialize legend handles for short and long blocks
legend_handles_short = [];  % For blue plots
legend_handles_long = [];   % For red plots
% Clear and initialize legend_entries_short to ensure it's a cell array

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

N_SD_1_short = 0;
N_SD_2_short = 0;
N_C_short = 0;
N_SD_1_long = 0;
N_SD_2_long = 0;
N_C_long =  0;


% Initialize containers for short block data (all trials across all sessions)
all_trials_short_block_SD_1 = [];
all_trials_short_block_SD_2 = [];
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
         
        
       % Skip trial if ISI does not exist (i.e., no LED_Puff_ISI field)
        if ~isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'LED_Puff_ISI')
            continue;
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
        % commonTime = linspace(-seconds_before, seconds_after, Frames_before + Frames_after + 1);
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

        % Apply smoothing to reduce noise
        FEC_led_aligned_trimmed_smooth = smoothdata(FEC_led_aligned_trimmed, 'movmean', 5); % Moving average
        FEC_trimmed_smooth = smoothdata(FEC_trimmed, 'movmean', 5); % Moving average

        isShortBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - ...
                       SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) <= 0.3;
        t_LED = LED_Onset_Zero_Start;
        t_puff = AirPuff_LED_Onset_Aligned_Start;
        
        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;

        % poor_CR_threshold = 0.02;
        % is_CR_plus = CR_plus_eval_dev(FEC_times_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);
        % CR_category = classify_CR_02(FEC_led_aligned_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);
        CR_category = classify_CR_05(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, t_LED, t_puff , good_CR_threshold, poor_CR_threshold);

        % Define the common time vector (e.g., the union of all unique time points or a regular grid)
        commonTime = linspace(min(FEC_led_aligned_trimmed_smooth), max(FEC_led_aligned_trimmed_smooth), 500);  % Adjust 100 to the desired number of points

        % FEC_norm_interp = interp1(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, commonTime, 'spline');
        % Interpolate FEC data to this global common time base
        FEC_norm_interp = interp1(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, commonTime, 'spline', NaN);
        % Aggregate data for short blocks

        switch CR_category
            
            case 'Poor CR'    
                if isShortBlock
                    if SessionData.SleepDeprived == 3  

                        all_trials_short_block_SD_1 = [all_trials_short_block_SD_1; FEC_norm_interp];
                        N_SD_1_short = N_SD_1_short + 1;

                    elseif SessionData.SleepDeprived == 4

                        all_trials_short_block_SD_2 = [all_trials_short_block_SD_2; FEC_norm_interp];
                        N_SD_2_short = N_SD_2_short + 1;

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
SD_1_FEC_norm_avg_short = mean(all_trials_short_block_SD_1, 1, 'omitnan');
SD_2_FEC_norm_avg_short = mean(all_trials_short_block_SD_2, 1, 'omitnan');
C_FEC_norm_avg_short = mean(all_trials_short_block_Control, 1, 'omitnan');

global_min_value = min([SD_1_FEC_norm_avg_short,SD_2_FEC_norm_avg_short,C_FEC_norm_avg_short]);
global_max_value = max([SD_1_FEC_norm_avg_short,SD_2_FEC_norm_avg_short, C_FEC_norm_avg_short]);

% Define baseline window (adjust indices based on sampling rate)

if length(commonTime) == length(SD_1_FEC_norm_avg_short)
    baseline_window_indices = (common_time_vector >= -0.2 & common_time_vector <= 0);
    baseline_SD_1_short = mean(SD_1_FEC_norm_avg_short(baseline_window_indices), 'omitnan');

else
    warning('Length mismatch between commonTime and SD_1_FEC_norm_avg_short. Skipping baseline computation.');
    baseline_SD_1_short = NaN;  % or handle as appropriate

end

if length(commonTime) == length(SD_2_FEC_norm_avg_short)
    baseline_window_indices = (common_time_vector >= -0.2 & common_time_vector <= 0);

    baseline_SD_2_short = mean(SD_2_FEC_norm_avg_short(baseline_window_indices), 'omitnan');
else
    warning('Length mismatch between commonTime and SD_2_FEC_norm_avg_short. Skipping baseline computation.');

    baseline_SD_2_short = NaN;
end

baseline_window_indices = (common_time_vector >= -0.2 & common_time_vector <= 0); 

baseline_C_short = mean(C_FEC_norm_avg_short(baseline_window_indices));

% Subtract baseline from each curve
SD_1_FEC_norm_avg_short_corrected = SD_1_FEC_norm_avg_short - baseline_SD_1_short;
SD_2_FEC_norm_avg_short_corrected = SD_2_FEC_norm_avg_short - baseline_SD_2_short;
C_FEC_norm_avg_short_corrected = C_FEC_norm_avg_short - baseline_C_short;
% 
% % Standard error calculation
SD_1_std_short_block = std(all_trials_short_block_SD_1, 0, 1, 'omitnan');
SD_2_std_short_block = std(all_trials_short_block_SD_2, 0, 1, 'omitnan');
C_std_short_block = std(all_trials_short_block_Control, 0, 1, 'omitnan');
% 
combined_SD_1_sem_short = SD_1_std_short_block ./ sqrt(N_SD_1_short);
combined_SD_2_sem_short = SD_2_std_short_block ./ sqrt(N_SD_2_short);
combined_C_sem_short = C_std_short_block ./ sqrt(N_C_short);
% 
% 
% Plot results for short blocks
% figure;
hold on;
% 

global_max_value = global_max_value + 0.02;
global_min_value = global_min_value - 0.02;


    % Plot the mean line for short blocks

    if ~isempty(SD_1_FEC_norm_avg_short)
        h1 = plot(commonTime, SD_1_FEC_norm_avg_short, 'r', 'LineWidth', 1.5);
    else
        warning('No trials found for SD_1_FEC_norm_avg_short. Plot skipped.');
    end
    if ~isempty(SD_2_FEC_norm_avg_short)
        h2 = plot(commonTime, SD_2_FEC_norm_avg_short, 'Color', [0.5, 0, 0.5], 'LineWidth', 1.5);
    else
        warning('No trials found for SD_1_FEC_norm_avg_short. Plot skipped.');
    end 
    
    h3 = plot(commonTime, C_FEC_norm_avg_short , 'k', 'LineWidth', 1.5);

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
    % 
    % legend({['SD+1 Session Trials (n = ', num2str(N_SD_1_short), ')'], ...
    %         ['SD+2 Session Trials (n = ', num2str(N_SD_2_short), ')'], ...
    %         ['Control Session Trials (n = ', num2str(N_C_short), ')', newline, ...
    %          '(', datestr(firstSessionDate, 'mm/dd/yyyy'), ' to ', datestr(lastSessionDate, 'mm/dd/yyyy'), ')']}, ...
    %     'Location', 'bestoutside', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 10, 'Box', 'off');


legend({...
    ['SD+1 Session Trials (n = ', num2str(N_SD_1_short), ')'], ...
    ['SD+2 Session Trials (n = ', num2str(N_SD_2_short), ')'], ...
    sprintf('Control Trials (n = %d)\n(%s to %s)', ...
            N_C_short, ...
            datestr(firstSessionDate, 'mm/dd/yyyy'), ...
            datestr(lastSessionDate, 'mm/dd/yyyy'))}, ...
    'Location', 'bestoutside', ...
    'Interpreter', 'none', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 10, ...
    'Box', 'off');

    %Shading the SEM region around the mean
    % fill([commonTime, fliplr(commonTime)], ...
    %      [SD_1_FEC_norm_avg_short + combined_SD_1_sem_short, fliplr(SD_1_FEC_norm_avg_short - combined_SD_1_sem_short)], ...
    %      'r', 'FaceAlpha', 0.4, 'EdgeColor', 'none','HandleVisibility', 'off'); 

    if ~isempty(SD_1_FEC_norm_avg_short) && length(SD_1_FEC_norm_avg_short) == length(commonTime)
        fill([commonTime, fliplr(commonTime)], ...
             [SD_1_FEC_norm_avg_short + combined_SD_1_sem_short, ...
              fliplr(SD_1_FEC_norm_avg_short - combined_SD_1_sem_short)], ...
             'r', 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');  % purple
    else
        warning('Skipping fill plot for SD_1 due to mismatched or empty vectors.');
    end

    if ~isempty(SD_2_FEC_norm_avg_short) && length(SD_2_FEC_norm_avg_short) == length(commonTime)
         fill([commonTime, fliplr(commonTime)], ...
         [SD_2_FEC_norm_avg_short + combined_SD_2_sem_short, fliplr(SD_2_FEC_norm_avg_short - combined_SD_2_sem_short)], ...
         [0.5, 0, 0.5], 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    else
        warning('Skipping fill plot for SD_1 due to mismatched or empty vectors.');
    end

    fill([commonTime, fliplr(commonTime)], ...
         [C_FEC_norm_avg_short + combined_C_sem_short, fliplr(C_FEC_norm_avg_short - combined_C_sem_short)], ...
         'k', 'FaceAlpha', 0.4, 'EdgeColor', 'none','HandleVisibility', 'off'); 

    % Add LED onset shading
    y_fill = [0 0 1 1];
    x_fill_shortLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
    fill(x_fill_shortLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none','HandleVisibility', 'off');

    % Add AirPuff shading
    y_fill = [0 0 1 1];
    fill(x_fill_short, y_fill, [0.68, 0.85, 0.9], 'FaceAlpha', 0.35, 'EdgeColor', 'none','HandleVisibility', 'off');  % Short blocks shading

    text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.2,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'blue', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');

    title_text(1) = {' '};
    title_text(2) = {'Poor CR trials average short-block above the baseline CR$_{\rm{th}}$ in $(T_{\rm{LED}},T_{\rm{AirPuff-0.050}})$'};
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

title('Baseline-Adjusted Poor CR Short Blocks', 'Interpreter', 'latex', 'FontSize', 14);

    if ~isempty(SD_1_FEC_norm_avg_short) && length(SD_1_FEC_norm_avg_short) == length(commonTime)
        plot(commonTime, SD_1_FEC_norm_avg_short_corrected, 'r', 'LineWidth', 1.5);
    else
        warning('Skipping fill plot for SD_1 due to mismatched or empty vectors.');
    end

    if ~isempty(SD_2_FEC_norm_avg_short) && length(SD_2_FEC_norm_avg_short) == length(commonTime)
        plot(commonTime, SD_2_FEC_norm_avg_short_corrected, 'Color', [0.5, 0, 0.5], 'LineWidth', 1.5);
    else
        warning('Skipping fill plot for SD_1 due to mismatched or empty vectors.');
    end


plot(commonTime, C_FEC_norm_avg_short_corrected, 'k', 'LineWidth', 1.5);

global_min_value_baseline = min([SD_1_FEC_norm_avg_short_corrected,SD_2_FEC_norm_avg_short_corrected,C_FEC_norm_avg_short_corrected]);
global_max_value_baseline = max([SD_1_FEC_norm_avg_short_corrected,SD_2_FEC_norm_avg_short_corrected,C_FEC_norm_avg_short_corrected]);

 
global_max_value_baseline = global_max_value_baseline + 0.02;
global_min_value_baseline = global_min_value_baseline - 0.02;



    if ~isempty(SD_1_FEC_norm_avg_short) && length(SD_1_FEC_norm_avg_short) == length(commonTime)
        fill([commonTime, fliplr(commonTime)], ...
             [SD_1_FEC_norm_avg_short_corrected + combined_SD_1_sem_short, ...
              fliplr(SD_1_FEC_norm_avg_short_corrected - combined_SD_1_sem_short)], ...
              'r', 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');    
    else
        warning('Skipping fill plot for SD_1 due to mismatched or empty vectors.');
    end

    if ~isempty(SD_2_FEC_norm_avg_short) && length(SD_2_FEC_norm_avg_short) == length(commonTime)
        fill([commonTime, fliplr(commonTime)], ...
             [SD_2_FEC_norm_avg_short_corrected + combined_SD_2_sem_short, ...
              fliplr(SD_2_FEC_norm_avg_short_corrected - combined_SD_2_sem_short)], ...
             [0.5, 0, 0.5], 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    else
        warning('Skipping fill plot for SD_1 due to mismatched or empty vectors.');
    end


fill([commonTime, fliplr(commonTime)], ...
     [C_FEC_norm_avg_short_corrected + combined_C_sem_short, ...
      fliplr(C_FEC_norm_avg_short_corrected - combined_C_sem_short)], ...
      'k', 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off'); 

    % Add LED onset shading
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
    % ylim([-0.2 0.8]);
    ylim([global_min_value_baseline global_max_value_baseline]);
    xlim([-0.2 0.6]);
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subplot 2: Long Block


all_trials_long_block_SD_1 = [];
all_trials_long_block_SD_2 = [];
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
        
       % Skip trial if ISI does not exist (i.e., no LED_Puff_ISI field)
        if ~isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'LED_Puff_ISI')
            continue;
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

        % Apply smoothing to reduce noise
        FEC_led_aligned_trimmed_smooth = smoothdata(FEC_led_aligned_trimmed, 'movmean', 5); % Moving average
        FEC_trimmed_smooth = smoothdata(FEC_trimmed, 'movmean', 5); % Moving average

        
        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;

        poor_CR_threshold = 0.05;
        is_CR_plus = CR_plus_eval_dev(FEC_times_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);
        % is_CR_plus_poor = Cr_plus_poorly_eval(FEC_led_aligned_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);
        % CR_category = classify_CR_02(FEC_led_aligned_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);
        
        CR_category = classify_CR_04(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, t_LED, t_puff , good_CR_threshold, poor_CR_threshold);

        % Define the common time vector (e.g., the union of all unique time points or a regular grid)
        commonTime = linspace(min(FEC_led_aligned_trimmed_smooth), max(FEC_led_aligned_trimmed_smooth), 500);  % Adjust 100 to the desired number of points
        
        % CR_plus_matrix = CR_plus_eval_all_v3(FEC_times_trimmed, FEC_trimmed, t1_B1, t2_B2, t1_CR, t2_CR, CR_threshold);
        
        % if CR_plus_matrix
            numCurves = numCurves+1;
            % Interpolate FEC_norm to the common time vector
            FEC_norm_interp = interp1(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, commonTime, 'spline');
            
            LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - LED_Onset;
         % Aggregate data for short blocks
       switch CR_category
            case 'Good CR'
            case 'Poor CR'    
            if isLongBlock
                if SessionData.SleepDeprived == 3 
                    all_trials_long_block_SD_1 = [all_trials_long_block_SD_1; FEC_norm_interp];
                    N_SD_1_long = N_SD_1_long + 1;
                    
                elseif SessionData.SleepDeprived == 4
                    all_trials_long_block_SD_2 = [all_trials_long_block_SD_2; FEC_norm_interp];
                    N_SD_2_long = N_SD_2_long + 1;
            
                else
                    all_trials_long_block_Control = [all_trials_long_block_Control; FEC_norm_interp];
                    N_C_long = N_C_long + 1;
                end
            
                x_fill_long = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, ...
                               AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
            end
        end


 
    end
end



% Calculate averages for SD and Control trials
SD_1_FEC_norm_avg_long = mean(all_trials_long_block_SD_1, 1, 'omitnan');
SD_2_FEC_norm_avg_long = mean(all_trials_long_block_SD_2, 1, 'omitnan');
C_FEC_norm_avg_long = mean(all_trials_long_block_Control, 1, 'omitnan');

global_min_value = min([SD_1_FEC_norm_avg_long,SD_2_FEC_norm_avg_long,C_FEC_norm_avg_long]);
global_max_value = max([SD_1_FEC_norm_avg_long,SD_2_FEC_norm_avg_long,C_FEC_norm_avg_long]);

% Define baseline window (adjust indices based on sampling rate)
baseline_window_indices = (common_time_vector >= -0.2 & common_time_vector <= 0); 

% Calculate baseline mean for SD and Control trials
baseline_SD_1_long = mean(SD_1_FEC_norm_avg_long(baseline_window_indices));
baseline_SD_2_long = mean(SD_2_FEC_norm_avg_long(baseline_window_indices));
baseline_C_long = mean(C_FEC_norm_avg_long(baseline_window_indices));

% Subtract baseline from each curve
SD_1_FEC_norm_avg_long_corrected = SD_1_FEC_norm_avg_long - baseline_SD_1_long;
SD_2_FEC_norm_avg_long_corrected = SD_2_FEC_norm_avg_long - baseline_SD_2_long;
C_FEC_norm_avg_long_corrected = C_FEC_norm_avg_long - baseline_C_long;

% Standard error calculation
SD_1_std_long_block = std(all_trials_long_block_SD_1, 0, 1, 'omitnan');
SD_2_std_long_block = std(all_trials_long_block_SD_2, 0, 1, 'omitnan');
C_std_long_block = std(all_trials_long_block_Control, 0, 1, 'omitnan');

combined_SD_1_sem_long = SD_1_std_long_block ./ sqrt(N_SD_1_long);
combined_SD_2_sem_long = SD_2_std_long_block ./ sqrt(N_SD_2_long);
combined_C_sem_long = C_std_long_block ./ sqrt(N_C_long);



%%

nexttile;
hold on;

    % Plot the mean line for short blocks
    h4 = plot(commonTime, SD_1_FEC_norm_avg_long,'r' , 'LineWidth', 1.5); % red color for SD1 
    h5 = plot(commonTime, SD_2_FEC_norm_avg_long, 'Color', [0.5 0 0.5], 'LineWidth', 1.5); %purple color for SD2
    h6 = plot(commonTime, C_FEC_norm_avg_long , 'k', 'LineWidth', 1.5); %black color for control
    
 
    % Shading the SEM region around the mean

    fill([commonTime, fliplr(commonTime)], ...
         [SD_1_FEC_norm_avg_long + combined_SD_1_sem_long, fliplr(SD_1_FEC_norm_avg_long - combined_SD_1_sem_long)], ...
          'r', 'FaceAlpha', 0.4, 'EdgeColor', 'none','HandleVisibility', 'off');
    fill([commonTime, fliplr(commonTime)], ...
     [SD_2_FEC_norm_avg_long + combined_SD_2_sem_long, fliplr(SD_2_FEC_norm_avg_long - combined_SD_2_sem_long)], ...
     [0.5 0 0.5], 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    fill([commonTime, fliplr(commonTime)], ...
         [C_FEC_norm_avg_long + combined_C_sem_long, fliplr(C_FEC_norm_avg_long - combined_C_sem_long)], ...
          'k', 'FaceAlpha', 0.4, 'EdgeColor', 'none','HandleVisibility', 'off');

    
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
    title_text(2) = {'Poor CR trials average long-block above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{LED}},T_{\rm{AirPuff-0.050}})$'};
    title_text(3) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $',CR_threshold)};
    
    Title_Short = title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',14);
    set(gca, 'TickDir', 'out'); % Moves ticks to the outside
    hold off;


    % legend({['SD+1 Session Trials (n = ', num2str(N_SD_1_long), ')'], ...
    %         ['SD+2 Session Trials (n = ', num2str(N_SD_2_long), ')'], newline ...
    %         ['Control Session Trials (n = ', num2str(N_C_long), ')', newline, ...
    %          '(', datestr(firstSessionDate, 'mm/dd/yyyy'), ' to ', datestr(lastSessionDate, 'mm/dd/yyyy'), ')']}, ...
    %     'Location', 'bestoutside', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 10, 'Box', 'off');


    legend({...
    ['SD+1 Session Trials (n = ', num2str(N_SD_1_long), ')'], ...
    ['SD+2 Session Trials (n = ', num2str(N_SD_2_long), ')'], ...
    sprintf(['Control Session Trials (n = %d)\n(%s to %s)'], ...
            N_C_long, ...
            datestr(firstSessionDate, 'mm/dd/yyyy'), ...
            datestr(lastSessionDate, 'mm/dd/yyyy'))}, ...
    'Location', 'bestoutside', ...
    'Interpreter', 'latex', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 10, ...
    'Box', 'off');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subplot(2, 2, 4);
% hold on;
    nexttile;
    hold on;

title('Baseline-Adjusted Poor CR Long Blocks', 'Interpreter', 'latex', 'FontSize', 14);    

plot(commonTime, SD_1_FEC_norm_avg_long_corrected, 'r', 'LineWidth', 1.5);
plot(commonTime, SD_2_FEC_norm_avg_long_corrected, 'Color', [0.5 0 0.5], 'LineWidth', 1.5);
plot(commonTime, C_FEC_norm_avg_long_corrected, 'k', 'LineWidth', 1.5);

global_min_value_baseline = min([SD_1_FEC_norm_avg_long_corrected, SD_2_FEC_norm_avg_long_corrected,C_FEC_norm_avg_long_corrected]);
global_max_value_baseline = max([SD_1_FEC_norm_avg_long_corrected, SD_2_FEC_norm_avg_long_corrected, C_FEC_norm_avg_long_corrected]);


global_max_value_baseline = global_max_value_baseline + 0.02;
global_min_value_baseline = global_min_value_baseline - 0.02;

fill([commonTime, fliplr(commonTime)], ...
     [SD_1_FEC_norm_avg_long_corrected + combined_SD_1_sem_long, ...
      fliplr(SD_1_FEC_norm_avg_long_corrected - combined_SD_1_sem_long)], ...
      'r', 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');

fill([commonTime, fliplr(commonTime)], ...
     [SD_2_FEC_norm_avg_long_corrected + combined_SD_2_sem_long, ...
      fliplr(SD_2_FEC_norm_avg_long_corrected - combined_SD_2_sem_long)], ...
     [0.5 0 0.5], 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');

fill([commonTime, fliplr(commonTime)], ...
     [C_FEC_norm_avg_long_corrected + combined_C_sem_long, ...
      fliplr(C_FEC_norm_avg_long_corrected - combined_C_sem_long)], ...
      'k', 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % Configure plot
    ylim([global_min_value_baseline global_max_value_baseline]);
    xlim([-0.2 0.6]);
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);

    % Add LED onset shading
    
    y_fill = [global_min_value_baseline global_min_value_baseline global_max_value_baseline global_max_value_baseline];
    x_fill_longLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
    fill(x_fill_longLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none','HandleVisibility', 'off');
                
    set(gca, 'TickDir', 'out'); % Moves ticks to the outside    
    % Add AirPuff shading
    y_fill = [global_min_value_baseline global_min_value_baseline global_max_value_baseline global_max_value_baseline];
    fill(x_fill_long, y_fill, [0.56, 0.93, 0.56], 'FaceAlpha', 0.65, 'EdgeColor', 'none','HandleVisibility', 'off');  % Long blocks shading

    % Save the figure
    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 6
        prefixPart = nameParts{1};
        % Specify the desired file format as 'pdf'
        fileFormat = '.pdf';  % PDF format
          
        newFilename = sprintf('%s_MegaSession_PoorCR_SD_Control', prefixPart); 

        newFilename = [newFilename, fileFormat];
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end
