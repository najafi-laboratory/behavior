clc; close all; clear

% Load all session data files
data_files = dir('*_EBC_*.mat');


% Initialize legend handles for short and long blocks
legend_handles_short = [];  % For blue plots
legend_handles_long = [];   % For red plots

CR_threshold = 0.05;
% Initialize legend handles for short and long blocks
legend_handles_short = [];  % For blue plots
legend_handles_long = [];   % For red plots
% Clear and initialize legend_entries_short to ensure it's a cell array

sessionDates = [];
legend_entries_short = {};  % Explicitly set it as a cell array
legend_entries_long = {};
x_fill_longLED = [];
x_fill_shortLED = [];
% figure('Position', [100, 100, 800, 800]); % Increase figure size for better layout control
figure;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Subplot 1: Short Block
%%
subplot(2, 1, 1);
hold on;
% Set the title with custom position
title_text(1) = {' '};
title_text(2) = {'CR$^{+}$ short-block trials above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{LED}},T_{\rm{AirPuff}})$'};
title_text(3) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $',CR_threshold)};
% title_text(4) = {' '}; 
Title_Short = title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',12);

% legend_handles = [];
legend_entries = {};

% Initialize containers for all sessions
all_sessions_short_block_data = [];
all_sessions_long_block_data = [];

% Loop over each data file
for i = 1:length(data_files)
    
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
        
        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
       
        if isLongBlock > 0.3
        
            t_LED = 0.2 ;
            t_puff = AirPuff_LED_Onset_Aligned_End;
            t1 = t_LED - 0.01;
            t2 = t_LED; 
        
        else
        
            t_LED = LED_Onset_Zero_Start;
            t_puff = AirPuff_LED_Onset_Aligned_End;
            t1 = t_LED - 0.01;
            t2 = t_LED; 
        end
        % Define the common time vector (e.g., the union of all unique time points or a regular grid)
        commonTime = linspace(min(FEC_led_aligned_trimmed), max(FEC_led_aligned_trimmed), 500);  % Adjust 100 to the desired number of points
        
        if(CR_plus_eval(FEC_led_aligned_trimmed,FEC_trimmed,t1,t2,t_LED,t_puff,CR_threshold))
            numCurves = numCurves+1;
            % Interpolate FEC_norm to the common time vector
            FEC_norm_interp = interp1(FEC_led_aligned_trimmed, FEC_trimmed, commonTime, 'spline');
            
            LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - LED_Onset;
            % % Identify if the trial is short or long and store the data
            if length(FEC_trimmed) == length(FEC_led_aligned_trimmed)

                if SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3
    
                else
                    short_block_data = [short_block_data; FEC_norm_interp];
                    verticalLineColor = 'b'; % Blue for short trials
                    shortBlockAirPuffColor = [0.5, 0.5, 1.0];
                    x_fill_short = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
                    x_fill_shortLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
     
                end
            else
                warning('Skipping trial %d in file %s due to dimension mismatch.', ctr_trial, data_files(i).name);
            end
            
        end

    end

    all_sessions_short_block_data = [all_sessions_short_block_data; short_block_data];  % Concatenate data across sessions
end

    % Mean across all sessions for short blocks
    FEC_norm_avg_short = mean(all_sessions_short_block_data, 1, 'omitnan');
    

% Find the global minimum and maximum between both curves
global_min_value = min(FEC_norm_avg_short);
global_max_value = max(FEC_norm_avg_short);
% Plot averages and SEM outside the trial loop
hold on;


global_max_value = global_max_value + 0.02;
global_min_value = global_min_value - 0.02;

    % Estimated imposed error based on camera precision, processing, and synchronization
    imposed_error = 0.0229;  % Replace with your estimated value
    
    % Calculate standard deviation across all sessions for each time point
    std_short_block = std(all_sessions_short_block_data, 0, 1, 'omitnan');
    
    % Number of sessions (N)
    N = size(all_sessions_short_block_data, 1);
    
    % Calculate traditional SEM
    sem_short_block = std_short_block ./ sqrt(N);
    
    % Combine SEM with imposed error
    combined_sem_short = sqrt(sem_short_block.^2 + imposed_error^2);

    % Plot the mean line for short blocks
    h1 = plot(commonTime, FEC_norm_avg_short, 'b', 'LineWidth', 1.5);  % Blue for short blocks
    
    % Define the number of sessions
    % num_sessions = length(data_files);


    
    % 
    % legend_handles_short = [legend_handles_short, h1];
    % [~, name, ~] = fileparts(data_files(i).name);
    % nameParts = split(name, '_');
    % if length(nameParts) >= 6
    %     prefixPart = nameParts{1};
    %     datePart = nameParts{6};
    %     % Extract only the year and month from datePart
    %     yearMonth = datePart(1:6);  % Extract first six characters representing 'YYYYMM'
    % 
    %     % legend_entries_short = sprintf('%s_%s_Sessions-(Total Sessions:)', prefixPart, yearMonth,num2str(num_sessions));  % e.g., 'Session1_202310'
    %     legend_entries_short =  {sprintf('%s_%s_Sessions:Total=%s', prefixPart, yearMonth, num2str(num_sessions))};
    % 
    % end

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

    legend({[' (N = ', num2str(N_short), ')', newline, ...
         '(', datestr(firstSessionDate, 'mm/dd/yyyy'), ' to ', datestr(lastSessionDate, 'mm/dd/yyyy'), ')'], ...
            [' (N = ', num2str(N_long), ')', newline, ...
         '(', datestr(firstSessionDate, 'mm/dd/yyyy'), ' to ', datestr(lastSessionDate, 'mm/dd/yyyy'), ')']}, ...
    'Location', 'bestoutside', 'Interpreter','latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Box', 'off');




   
    % Shading the SEM region around the mean
    fill([commonTime, fliplr(commonTime)], ...
         [FEC_norm_avg_short + combined_sem_short, fliplr(FEC_norm_avg_short - combined_sem_short)], ...
         [0.2 0.2 0.6], 'FaceAlpha', 0.4, 'EdgeColor', 'none','HandleVisibility', 'off');



    % Add LED onset shading
    y_fill = [0 0 1 1];
    fill(x_fill_shortLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none','HandleVisibility', 'off');
    
   
    % Add AirPuff shading
    y_fill = [0 0 1 1];
    fill(x_fill_short, y_fill, [0.68, 0.85, 0.9], 'FaceAlpha', 0.35, 'EdgeColor', 'none','HandleVisibility', 'off');  % Short blocks shading

    text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.2,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'blue', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    
    % Configure plot
    ylim([global_min_value global_max_value]);
    % ylim([0.4 1]);
    xlim([-0.2 0.6]);
    % set(gca, 'XTick', []);  % Remove x-axis ticks
    % xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 12);
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
    % legend(legend_handles, legend_entries, 'Interpreter', 'latex', 'location', 'northeast', 'Box', 'off');

 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subplot 2: Long Block
%%
subplot(2, 1, 2);
hold on;

% % Set the title with custom position
% title('Mean Of Long Block Trials', 'Interpreter', 'latex','Units', 'normalized', 'Position', [0.5, 1.09, 0]);
title_text_long(1) = {'CR$^{+}$ long-block trials above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{0.2(s)}},T_{\rm{AirPuff}})$'};
title_text_long(2) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $',CR_threshold)};
% title_text(3) = {' '}; 

Title_long = title(title_text_long,'interpreter', 'latex','fontname','Times New Roman','fontsize',12);


% legend_handles = [];

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

        if(CR_plus_eval(FEC_led_aligned_trimmed,FEC_trimmed,t1,t2,t_LED,t_puff,CR_threshold))
            numCurves = numCurves+1;
            % Interpolate FEC_norm to the common time vector
            FEC_norm_interp = interp1(FEC_led_aligned_trimmed, FEC_trimmed, commonTime, 'spline');
            
            LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - LED_Onset;
            
            % Identify if the trial is short or long and store the data
            if length(FEC_trimmed) == length(FEC_led_aligned_trimmed)
                if SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3
                    long_block_data = [long_block_data; FEC_norm_interp];
                    longBlockAirPuffColor = [0.5, 1.0, 0.5]; % Light green for long trials
                    x_fill_long = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
                    x_fill_longLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
                
                end
                   
            else
                warning('Skipping trial %d in file %s due to dimension mismatch.', ctr_trial, data_files(i).name);
            end

        end
    end

    all_sessions_long_block_data = [all_sessions_long_block_data; long_block_data];  % Concatenate data across sessions
end


    % Mean across all sessions for short blocks
    FEC_norm_avg_long = mean(all_sessions_long_block_data, 1, 'omitnan');


    % Find the global minimum and maximum between both curves
    global_min_value = min(FEC_norm_avg_long);
    global_max_value = max(FEC_norm_avg_long);
    % Plot averages and SEM outside the trial loop
    hold on;
    
    
    global_max_value = global_max_value + 0.02;
    global_min_value = global_min_value - 0.02;

    % Estimated imposed error based on camera precision, processing, and synchronization
    imposed_error = 0.0229;  % Replace with your estimated value
    
    % Calculate standard deviation across all sessions for each time point
    std_long_block = std(all_sessions_long_block_data, 0, 1, 'omitnan');
    
    % Number of sessions (N)
    N = size(all_sessions_long_block_data, 1);
    
    % Calculate traditional SEM
    sem_long_block = std_long_block ./ sqrt(N);
    
    % Combine SEM with imposed error
    combined_sem = sqrt(sem_long_block.^2 + imposed_error^2);

    % Plot the mean line for short blocks
    h2 = plot(commonTime, FEC_norm_avg_long, 'Color', [0 0.5 0], 'LineWidth', 1.5);  % Blue for short blocks
    
    % legend_handles_long = [legend_handles_long, h2];
    % [~, name, ~] = fileparts(data_files(i).name);
    % nameParts = split(name, '_');
    % if length(nameParts) >= 6
    %     prefixPart = nameParts{1};
    %     datePart = nameParts{6};
    %     % Extract only the year and month from datePart
    %     yearMonth = datePart(1:6);  % Extract first six characters representing 'YYYYMM'
    % 
    %     legend_entries_long = {sprintf('%s_%s_Sessions:Total=%s', prefixPart, yearMonth, num2str(num_sessions))};  % e.g., 'Session1_202310'
    % end
    % 

    % Shading the SEM region around the mean
    fill([commonTime, fliplr(commonTime)], ...
         [FEC_norm_avg_long + combined_sem, fliplr(FEC_norm_avg_long - combined_sem)], ...
          [0.0, 0.5, 0.0], 'FaceAlpha', 0.4, 'EdgeColor', 'none','HandleVisibility', 'off');
    
    % Add LED onset shading
    y_fill = [0 0 1 1];
    fill(x_fill_longLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none','HandleVisibility', 'off');
    
    % Add AirPuff shading
    y_fill = [0 0 1 1];
    fill(x_fill_long, y_fill, [0.56, 0.93, 0.56], 'FaceAlpha', 0.65, 'EdgeColor', 'none','HandleVisibility', 'off');  % Long blocks shading
    
    
    % Configure plot
    ylim([global_min_value global_max_value]);
    % ylim([0.5 1]);
    xlim([-0.2 0.6]);
    xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 12);
    ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
    % legend(legend_handles, legend_entries, 'Interpreter', 'latex', 'location', 'northeast', 'Box', 'off');
    text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.4,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'green', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    
    % Adjust subplot positions manually for square-like appearance
    % subplot(2, 1, 1);
    % set(gca, 'Position', [0.2 0.57 0.6 0.35]); % [left bottom width height]
    % 
    % subplot(2, 1, 2);
    % set(gca, 'Position', [0.2 0.1 0.6 0.35]); % [left bottom width height]
    
    hold off;

    % % Create the legend and make it movable
    % lgd_short = legend(legend_handles_short, legend_entries_short, 'Interpreter', 'none', 'Box', 'off', 'Location', 'bestoutside'); 
    % % Set the legend position to 'bestoutside'
    % set(lgd_short, 'Location', 'bestoutside');  % Position the short block legend outside the plot
    % 
    % Create legend entries with information directly, using concatenated strings with newlines
    legend({[' (N = ', num2str(N_short), ')', newline, ...
            '(', datestr(firstSessionDate, 'mm/dd/yyyy'), ' to ', datestr(lastSessionDate, 'mm/dd/yyyy'), ')'], ...
            [' (N = ', num2str(N_long), ')', newline, ...
            '(', datestr(firstSessionDate, 'mm/dd/yyyy'), ' to ', datestr(lastSessionDate, 'mm/dd/yyyy'), ')']}, ...
            'Location', 'bestoutside', 'Interpreter','latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Box', 'off');

    
    % lgd_long = legend(legend_handles_long, legend_entries_long, 'Interpreter', 'none', 'Box', 'off', 'Location', 'bestoutside');
    % % Set the legend position to 'bestoutside'
    % set(lgd_long, 'Location', 'bestoutside');  % Position the long block legend outside the plot


    % Save the figure
    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 6
        prefixPart = nameParts{1};
        % Specify the desired file format as 'pdf'
        fileFormat = '.pdf';  % PDF format
          
        newFilename = sprintf('%s_CRAvgAllSessions_SEM', prefixPart); 

        newFilename = [newFilename, fileFormat];
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end
