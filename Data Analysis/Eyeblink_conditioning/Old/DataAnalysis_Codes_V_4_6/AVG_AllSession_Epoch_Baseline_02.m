clc; close all; clear

% Load all session data files
data_files = dir('*_EBC_*.mat');
% data_files = dir('E6LG_EBC_V_3_13_20241031_190208.mat');
CR_threshold = 0.05;


% Initialize variables for dynamic subplot assignment
subplot_counter = 1;
short_block_plotted = false;
long_block_plotted = false; 
%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Subplot 1: Short Block
%%
% subplot(2, 2, 1);
% subplot('Position', [0.1, 0.55, 0.55, 0.4]); % Adjusted size for larger visibility


% Set the title with custom position


all_epoch1_short_trs = [];
all_epoch2_short_trs = [];
all_epoch1_long_trs = [];
all_epoch2_long_trs = [];
    % Initialize containers for block data
    epoch1_short_trs = [];
    epoch2_short_trs = [];
    epoch1_long_trs = [];
    epoch2_long_trs = [];
    
    % Initialize variables to count trials for each epoch
    num_trials_epoch1_short = 0;
    num_trials_epoch2_short = 0;
    num_trials_epoch1_long = 0;
    num_trials_epoch2_long = 0;

    figure('Position', [100, 100, 1200, 800]); % Increase figure size for better layout control
    % Create a 2x2 tiled layout
    
   
% Loop over each data file
for i = 1:length(data_files)

    t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile;
    hold on;
    
    load(data_files(i).name);
    numTrials = length(SessionData.RawEvents.Trial);
    numCurves = 0;
    totalFEC_norm = [];

    


    current_block_start = 0;
    % Define the segmentation ratio for first and last epochs
    epoch_ratio = 2; % Example: Divide each block into thirds (1/3 for first and last epochs)


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
    % Variables to track min and max of the curves
    minNumCurve = inf;
    maxNumCurve = -inf;
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

        % Define the common time vector (e.g., the union of all unique time points or a regular grid)
         commonTime = linspace(min(FEC_led_aligned_trimmed), max(FEC_led_aligned_trimmed), 500);  % Adjust 100 to the desired number of points

        % % Identify if the block is short or long
        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;
        % Identify if the block is short or long based on the LED_Puff_ISI duration
        isShortBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) < 0.3;
        FEC_norm_interp = interp1(FEC_led_aligned_trimmed, FEC_trimmed, commonTime, 'spline');
            
        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - LED_Onset;

        block_length = SessionData.TrialSettings(ctr_trial).GUI.BlockLength;  
        % Identify if the trial is part of the first or last epoch of the current block

        % Define the number of trials in each epoch within each block
        epoch_length = ceil(block_length / epoch_ratio); % Number of trials per epoch
        first_epoch_end_boundari = epoch_length; % Trials in the first epoch
        last_epoch_start_boundari = numTrials - epoch_length + 1; % Start of last epoch (last 'epoch_length' trials)

        t_LED = LED_Onset_Zero_Start;
        t_puff = AirPuff_LED_Onset_Aligned_Start;
        % Check if the trial is a CR+ trial
        is_CR_plus = CR_plus_eval_dev(FEC_led_aligned_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);
     
        % Check if the trial is a CR+ trial
        % is_CR_plus = CR_plus_eval(FEC_led_aligned_trimmed,FEC_trimmed, t1_B1, t2_B2, t1_CR, t2_CR, CR_threshold);

    % Determine block length and divide it into epochs
    block_length = SessionData.TrialSettings(ctr_trial).GUI.BlockLength;
    midpoint = ceil(block_length / 2);
    
    if isShortBlock
            %shortBlockAirPuffColor = [0.5, 0.5, 1.0];
            y_fill = [0 0 1 1];
            x_fill_short = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
            fill(x_fill_short, y_fill, [0.68, 0.85, 0.9], 'FaceAlpha', 0.35, 'EdgeColor', 'none','HandleVisibility', 'off');  % Short blocks shading
    if is_CR_plus   
        % Short Block Epoch Assignment
        if mod(ctr_trial, block_length) < midpoint
            % First half of the short block (Epoch 1)
            epoch1_short_trs = [epoch1_short_trs; FEC_norm_interp];
            num_trials_epoch1_short = num_trials_epoch1_short + 1;
            disp('Epoch1_short_trs')
            disp(ctr_trial)
        % 
        % elseif mod(ctr_trial, block_length) < block_length-1
        else    
            % Second half of the short block (Epoch 2)
            epoch2_short_trs = [epoch2_short_trs; FEC_norm_interp];
            num_trials_epoch2_short = num_trials_epoch2_short + 1;
            disp('Epoch2_short_trs')
            disp(ctr_trial)
        end
    end
    end

    % Accumulate epoch data across all sessions
    all_epoch1_short_trs = [all_epoch1_short_trs; mean(epoch1_short_trs, 1, 'omitnan')];
    all_epoch2_short_trs = [all_epoch2_short_trs; mean(epoch2_short_trs, 1, 'omitnan')];

end

    
    % Calculate the overall average for each epoch across all sessions
    average_epoch1_short = mean(all_epoch1_short_trs, 1, 'omitnan');
    average_epoch2_short = mean(all_epoch2_short_trs, 1, 'omitnan');
    
    % Output the number of sessions and trials for debugging
    fprintf('Processed %d sessions.\n', length(data_files));
    fprintf('Epoch 1 (Short Block) - Average Trials Across Sessions: %d\n', size(all_epoch1_short_trs, 1));
    fprintf('Epoch 2 (Short Block) - Average Trials Across Sessions: %d\n', size(all_epoch2_short_trs, 1));

    % Find the global minimum and maximum between both curves
    global_min_value = min([min(average_epoch1_short), min(average_epoch2_short)]);
    global_max_value = max([max(average_epoch1_short), max(average_epoch2_short)]);
    % Plot averages and SEM outside the trial loop
    hold on;
    
    % Define baseline window (adjust indices based on sampling rate)
baseline_window_indices = (common_time_vector >= -0.2 & common_time_vector <= 0); 

% Calculate baseline mean for SD and Control trials
baseline_first_short = mean(average_epoch1_short(baseline_window_indices));
baseline_last_short = mean(average_epoch2_short(baseline_window_indices));

% Subtract baseline from each curve
baseline_first_short_corrected = average_epoch1_short - baseline_first_short;
baseline_last_short_corrected = average_epoch2_short - baseline_last_short;

    
    global_max_value = global_max_value + 0.05;
    global_min_value = global_min_value - 0.05;

    % Estimated imposed error based on camera precision, processing, and synchronization
    imposed_error = 0.0229;  % Replace with your estimated value
    
    % Calculate standard deviation across all sessions for each time point
    first_std_short_block = std(epoch1_short_trs, 0, 1, 'omitnan');
    last_std_short_block = std(epoch2_short_trs, 0, 1, 'omitnan');
    % Number of sessions (N)
    N = size(epoch1_short_trs, 1);
    M = size(epoch2_short_trs,1);



    % Calculate traditional SEM
    firt_sem_short_block = first_std_short_block ./ sqrt(N);
    last_sem_short_block = last_std_short_block./ sqrt(M);


    % Combine SEM with imposed error
    first_combined_sem_short = sqrt(firt_sem_short_block.^2 + imposed_error^2);
    last_combined_sem_short = sqrt(last_sem_short_block.^2 + imposed_error^2);

    % Define the RGB values for dim blue and light blue
    dimBlue = [0.2, 0.2, 0.8];   % Dim Blue (darker shade)
    lightBlue = [0.6, 0.8, 1.0];  % Light Blue (lighter shade)
    greenBlue = [0.0, 0.5, 0.5];
    cyan = [0 1 1];        % Cyan
    blue = [0 0 1];        % Blue
    

    % Define a window size for smoothing (adjust as needed)
    windowSize = 5;

    % Apply moving average smoothing to the baseline-corrected means
    smoothed_average_epoch1_short = movmean(average_epoch1_short, windowSize);
    smoothed_average_epoch2_short = movmean(average_epoch2_short, windowSize);
    


    % Plot the smoothed baseline-corrected data
    h1 = plot(commonTime, smoothed_average_epoch1_short, 'Color', [0 1 1], 'LineWidth', 1.5);
    h2 = plot(commonTime, smoothed_average_epoch2_short, 'Color', [0 0 1], 'LineWidth', 1.5);

    % Add legend with h1 and h2 handles and trial counts
    legend([h1, h2], ...
           {sprintf('Epoch 1 (n = %d trs)', num_trials_epoch1_short), ...
            sprintf('Epoch 2 (n = %d trs)', num_trials_epoch2_short)}, ...
           'Location', 'best', 'Interpreter', 'latex', 'FontSize', 12, 'Box', 'off', 'Location', 'bestoutside');

    % Shading the SEM region around the mean
    fill([commonTime, fliplr(commonTime)], ...
         [smoothed_average_epoch1_short + first_combined_sem_short, fliplr(smoothed_average_epoch1_short - first_combined_sem_short)], ...
          [0 1 1], 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    fill([commonTime, fliplr(commonTime)], ...
         [smoothed_average_epoch2_short + last_combined_sem_short, fliplr(smoothed_average_epoch2_short - last_combined_sem_short)], ...
          [0 0 1], 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');
 

    % Add LED onset shading
    x_fill = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
    y_fill = [0 0 1 1];
    fill(x_fill, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none', 'HandleVisibility', 'off');
  
    x_fill_short = [0.199, 0.219, 0.219, 0.199];
    fill(x_fill_short, y_fill, [0.68, 0.85, 0.9], 'FaceAlpha', 0.35, 'EdgeColor', 'none','HandleVisibility', 'off');  % Short blocks shading

    title_text(1) = {'Pooled Trials Average Of Epoch1 And Epoch2 Short Trials'};
    title_text(2) = {'CR$^{+}$ short-block above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{AirPuff-0.05}},T_{\rm{AirPuff}})$'};
    title_text(3) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.3f $',CR_threshold)};

    Title_Short = title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',14);

   

    text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.2,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'blue', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    set(gca, 'TickDir', 'out');
    % Configure plot
    ylim([global_min_value global_max_value]);
    
    xlim([-0.2 0.6]);
    % set(gca, 'XTick', []);  % Remove x-axis ticks
    % xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 12);
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
    % legend(legend_handles, legend_entries, 'Interpreter', 'latex', 'location', 'northeast', 'Box', 'off');

    text(0,global_max_value, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.2,global_max_value, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');


    %% Subplot 2: Short Block - Baseline Shifted
    % subplot(2, 2, 2); % Position 2 in 2x2 grid
    % Small plot: Short Block - Baseline Corrected
    % subplot('Position', [0.7, 0.55, 0.25, 0.4]); % Smaller size for the comparison plot
    nexttile;
    hold on;
    title('Average Of Epoch1 And Epoch2 Short Trials (Baseline Shifted)', 'Interpreter', 'latex', 'FontSize', 14);

    % Find the global minimum and maximum between both curves
    global_min_value = min([min(baseline_first_short_corrected), min(baseline_last_short_corrected)]);
    global_max_value = max([max(baseline_first_short_corrected), max(baseline_last_short_corrected)]);
    % Plot averages and SEM outside the trial loop
    hold on;
    
    
    global_max_value = global_max_value + 0.05;
    global_min_value = global_min_value - 0.05;

    % Plot the smoothed baseline-corrected data
    plot(commonTime, baseline_first_short_corrected, 'Color', [0 0.7 1], 'LineWidth', 2);
    plot(commonTime, baseline_last_short_corrected, 'Color', [0 0 1], 'LineWidth', 2);

    % % Add LED onset shading
    x_fill = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
    y_fill = [global_min_value global_min_value global_max_value global_max_value];
    fill(x_fill, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    x_fill_short = [0.199, 0.219, 0.219, 0.199];
    fill(x_fill_short, y_fill, [0.68, 0.85, 0.9], 'FaceAlpha', 0.35, 'EdgeColor', 'none','HandleVisibility', 'off');  % Short blocks shading


    % Shading the SEM region around the mean
    fill([commonTime, fliplr(commonTime)], ...
         [baseline_first_short_corrected + first_combined_sem_short, fliplr(baseline_first_short_corrected - first_combined_sem_short)], ...
          [0 1 1], 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    fill([commonTime, fliplr(commonTime)], ...
         [baseline_last_short_corrected + last_combined_sem_short, fliplr(baseline_last_short_corrected - last_combined_sem_short)], ...
          [0 0 1], 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');


    text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.2,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'blue', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    set(gca, 'TickDir', 'out');
    % Configure plot
    ylim([global_min_value global_max_value]);
    % ylim([-0.1 1]);
    xlim([-0.2 0.6]);
    % set(gca, 'XTick', []);  % Remove x-axis ticks
    % xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 12);
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
    % legend(legend_handles, legend_entries, 'Interpreter', 'latex', 'location', 'northeast', 'Box', 'off');

    text(0,global_max_value, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.2,global_max_value, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');

 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Subplot 3: Long Block

% subplot(2, 2, 3);
nexttile;
hold on;



% Loop over each data file again for long blocks

    % load(data_files(i).name);
    numTrials = length(SessionData.RawEvents.Trial);
    % figure('units','centimeters','position',[2 2 24 26])
    numCurves = 0;
    totalFEC_norm = [];
    leg_str = cell(1,1);

    % Containers for long and short block data for this session
    long_block_data = [];
    short_block_data = [];
    last_epoch_long_data = [];
    first_epoch_long_data = [];

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

        FEC_led_aligned = FECTimes - LED_Onset;
        t_LED = LED_Onset_Zero_Start;
        t_puff = AirPuff_LED_Onset_Aligned_End;
        t1 = t_LED-0.01;
        t2 = t_LED;


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
      % Define the common time vector (e.g., the union of all unique time points or a regular grid)
        commonTime = linspace(min(FEC_led_aligned_trimmed), max(FEC_led_aligned_trimmed), 500);  % Adjust 100 to the desired number of points
        
        % Identify if the block is short or long
        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;
        % Identify if the block is short or long based on the LED_Puff_ISI duration
        isShortBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) < 0.3;

        block_length = SessionData.TrialSettings(ctr_trial).GUI.BlockLength;  
        % Identify if the trial is part of the first or last epoch of the current block
        % Interpolate FEC_norm to the common time vector
        FEC_norm_interp = interp1(FEC_led_aligned_trimmed, FEC_trimmed, commonTime, 'spline');
            
        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - LED_Onset;

        % Define epoch boundaries and categorize trials
        first_epoch_end_boundari = ceil(block_length / epoch_ratio);
        last_epoch_start_boundari = numTrials - first_epoch_end_boundari + 1;
    

     % Determine block length and divide it into epochs
    block_length = SessionData.TrialSettings(ctr_trial).GUI.BlockLength;
    midpoint = ceil(block_length / 2);

        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;

        t_LED = LED_Onset_Zero_Start;
        t_puff = AirPuff_LED_Onset_Aligned_Start;
        % Check if the trial is a CR+ trial
        is_CR_plus = CR_plus_eval_dev(FEC_led_aligned_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);
        
if is_CR_plus
    if isLongBlock
        % Long Block Epoch Assignment
        if mod(ctr_trial, block_length) < midpoint
            % First half of the long block (Epoch 1)
            epoch1_long_trs = [epoch1_long_trs; FEC_norm_interp];
            num_trials_epoch1_long = num_trials_epoch1_long + 1;
            disp('Epoch1_long_trs')
            disp(ctr_trial)

        elseif mod(ctr_trial, block_length) <  block_length+1
            % Second half of the long block (Epoch 2)
            epoch2_long_trs = [epoch2_long_trs; FEC_norm_interp];
            num_trials_epoch2_long = num_trials_epoch2_long + 1;
            disp('Epoch2_long_trs')
            disp(ctr_trial)

        end
    end     
end
    % Accumulate epoch data across all sessions
    all_epoch1_long_trs = [all_epoch1_long_trs; mean(epoch1_long_trs, 1, 'omitnan')];
    all_epoch2_long_trs = [all_epoch2_long_trs; mean(epoch2_long_trs, 1, 'omitnan')];

end

 

    % Calculate the overall average for each epoch across all sessions
    average_epoch1_long = mean(all_epoch1_long_trs, 1, 'omitnan');
    average_epoch2_long = mean(all_epoch2_long_trs, 1, 'omitnan');
    
    % Output the number of sessions and trials for debugging
    fprintf('Processed %d sessions.\n', length(data_files));
    fprintf('Epoch 1 (Long Block) - Average Trials Across Sessions: %d\n', size(all_epoch1_long_trs, 1));
    fprintf('Epoch 2 (Long Block) - Average Trials Across Sessions: %d\n', size(all_epoch2_long_trs, 1));


    % Estimated imposed error based on camera precision, processing, and synchronization
    imposed_error = 0.0229;  % Replace with your estimated value
    
    % Calculate standard deviation across all sessions for each time point
    first_std_long_block = std(epoch1_long_trs, 0, 1, 'omitnan');
    last_std_long_block = std(epoch2_long_trs, 0, 1, 'omitnan');

    % Number of sessions (N)
    K = size(epoch1_long_trs, 1);
    L = size(epoch2_long_trs,1);

    % Define baseline window (adjust indices based on sampling rate)
baseline_window_indices = (common_time_vector >= -0.2 & common_time_vector <= 0); 

% Calculate baseline mean for SD and Control trials
baseline_first_long = mean(average_epoch1_long(baseline_window_indices));
baseline_last_long = mean(average_epoch2_long(baseline_window_indices));

% Subtract baseline from each curve
baseline_first_long_corrected = average_epoch1_long - baseline_first_long;
baseline_last_long_corrected = average_epoch2_long - baseline_last_long;


    % Calculate traditional SEM
    firt_sem_long_block = first_std_long_block ./ sqrt(K);
    last_sem_long_block = last_std_long_block./ sqrt(L);


    % Combine SEM with imposed error
    first_combined_sem_long = sqrt(firt_sem_long_block.^2 + imposed_error^2);
    last_combined_sem_long = sqrt(last_sem_long_block.^2 + imposed_error^2);

    % Define the RGB values for new colors
    dimgreen = [0.0, 0.5, 0.0];     % Dim Green
    lightgreen = [0.3, 0.8, 0.6];    % Light Green
    bluegreen = [0.0, 0.8, 0.7];     % Blue Green
    
    dark_green = [0 0.5 0];     % Dark green
    
    windowSize = 5;
    % 
    % 
    % % Apply moving average smoothing to the baseline-corrected means
    smoothed_average_epoch1_long = movmean(average_epoch1_long, windowSize);
    smoothed_average_epoch2_long = movmean(average_epoch2_long, windowSize);



    % Find the global minimum and maximum between both curves
    global_min_value = min([min(smoothed_average_epoch1_long), min(smoothed_average_epoch2_long)]);
    global_max_value = max([max(smoothed_average_epoch1_long), max(smoothed_average_epoch2_long)]);
    % Plot averages and SEM outside the trial loop
    % hold on;
    
    
    global_max_value = global_max_value + 0.05;
    global_min_value = global_min_value - 0.05;


    % Plot first and last epochs for short blocks
    h3 = plot(commonTime, average_epoch1_long, 'Color', lightgreen, 'LineWidth', 1.5); % First epoch, short blocks
    h4 = plot(commonTime, average_epoch2_long, 'Color', dark_green, 'LineWidth', 1.5); % Last epoch, short blocks

    % Add legend with h1 and h2 handles and trial counts
    legend([h3, h4], ...
           {sprintf('Epoch 1 (n = %d trs)', num_trials_epoch1_long), ...
            sprintf('Epoch 2 (n = %d trs)', num_trials_epoch2_long)}, ...
           'Location', 'best', 'Interpreter', 'latex', 'FontSize', 12, 'Box', 'off', 'Location', 'bestoutside');

    % Shading the SEM region around the mean
    fill([commonTime, fliplr(commonTime)], [smoothed_average_epoch1_long + first_combined_sem_long, fliplr(smoothed_average_epoch1_long - first_combined_sem_long)], ...
    [0.3, 0.8, 0.6], 'FaceAlpha', 0.5, 'EdgeColor', 'none','HandleVisibility', 'off');

    fill([commonTime, fliplr(commonTime)], ...
         [smoothed_average_epoch2_long + last_combined_sem_long, fliplr(smoothed_average_epoch2_long - last_combined_sem_long)], ...
          [0 0.5 0], 'FaceAlpha', 0.4, 'EdgeColor', 'none','HandleVisibility', 'off');
    

    
     % Add LED onset shading
    x_fill = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
    y_fill = [global_min_value global_min_value global_max_value global_max_value];
    fill(x_fill, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    longBlockAirPuffColor = [0.5, 1.0, 0.5]; % Light green for long trials
     % Plot shaded area for AirPuff in long blocks (green)
            
    x_fill_long = [0.4, 0.42, 0.42, 0.4];
    fill(x_fill_long, y_fill, [0.5, 1.0, 0.5], 'FaceAlpha', 0.65, 'EdgeColor', 'none','HandleVisibility', 'off');  % Long blocks shading

    title_text_long(1) = {'Pooled Trials Average Of Epoche1 And Epoche2 Long Trials'}; 
    title_text(2) = {'CR$^{+}$ grand average long-block above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{LED}},T_{\rm{AirPuff}})$'};
    title_text(3) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $',0.04)};
    
    Title_long = title(title_text_long,'interpreter', 'latex','fontname','Times New Roman','fontsize',14);

    set(gca, 'TickDir', 'out');
    % Configure plot
    ylim([global_min_value global_max_value]);
    xlim([-0.2 0.6]);
    xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 12);
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
    text(0,global_max_value, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.4,global_max_value, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');

    %% Subplot 2: Short Block - Baseline Corrected
    % subplot(2, 2, 4); % Position 2 in 2x2 grid
    % subplot('Position', [0.7, 0.1, 0.25, 0.4]); % Smaller size for the comparison plot
    nexttile;
    hold on;
    title('Average of Epoche1 And Epoche2 Long Trials (Baseline Shifted)', 'Interpreter', 'latex', 'FontSize', 14);

    lightgreen = [0.3, 0.8, 0.6];    % Light Green
    dark_green = [0 0.5 0];     % Dark green

    % Find the global minimum and maximum between both curves
    global_min_value = min([min(baseline_first_long_corrected), min(baseline_last_long_corrected)]);
    global_max_value = max([max(baseline_first_long_corrected), max(baseline_last_long_corrected)]);
    % Plot averages and SEM outside the trial loop
    
    global_max_value = global_max_value + 0.05;
    global_min_value = global_min_value - 0.05;

    % Plot the smoothed baseline-corrected data
    plot(commonTime, baseline_first_long_corrected, 'Color', lightgreen, 'LineWidth', 1.5);
    plot(commonTime, baseline_last_long_corrected, 'Color',  dark_green, 'LineWidth', 1.5);


    % Shading the SEM region around the mean
    fill([commonTime, fliplr(commonTime)], [baseline_first_long_corrected + first_combined_sem_long, fliplr(baseline_first_long_corrected - first_combined_sem_long)], ...
    [0.3, 0.8, 0.6], 'FaceAlpha', 0.4, 'EdgeColor', 'none','HandleVisibility', 'off');

    fill([commonTime, fliplr(commonTime)], ...
         [baseline_last_long_corrected + last_combined_sem_long, fliplr(baseline_last_long_corrected - last_combined_sem_long)], ...
          [0 0.5 0], 'FaceAlpha', 0.4, 'EdgeColor', 'none','HandleVisibility', 'off');


    
    % % Add LED onset shading
    x_fill = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
    y_fill = [global_min_value global_min_value global_max_value global_max_value];

    fill(x_fill, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    x_fill_long = [0.4, 0.419, 0.419, 0.4];
    fill(x_fill_long, y_fill, [0.5, 1.0, 0.5], 'FaceAlpha', 0.65, 'EdgeColor', 'none');  % Long blocks shading


    text(0,global_max_value, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.4,global_max_value, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    
    % Configure plot
    ylim([global_min_value global_max_value]);
    xlim([-0.2 0.6]);
    % set(gca, 'XTick', []);  % Remove x-axis ticks
    % xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 12);
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
    % legend(legend_handles, legend_entries, 'Interpreter', 'latex', 'location', 'northeast', 'Box', 'off');

    set(gca, 'TickDir', 'out');
    hold off;




end


    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 4
        prefixPart = nameParts{1}; % First string before the first underscore
        datePart = nameParts{4}; % Date part
        newFilename = sprintf('%s_AVG_AllSession_EPOCHES.pdf', prefixPart);
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end


 



