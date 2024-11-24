clc; close all; clear

% Load all session data files
data_files = dir('*_EBC_*.mat');

CR_threshold = 0.05;
% Initialize legend handles for short and long blocks
legend_handles_short = [];  % For blue plots
legend_handles_long = [];   % For red plots

x_fill_longLED = [];
x_fill_shortLED = [];
% Initialize figure
figure;
 
% Subplot 1: Short Block
subplot(2, 1, 1);
hold on;
% Set the title with custom position
title('Short Block Trials','Interpreter', 'latex', 'Units', 'normalized', 'Position', [0.5, 1.09, 0]);
% legend_handles = [];
legend_entries = {};

% Loop over each data file
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

            % FEC_led_aligned = FECTimes - LED_Onset;
            % t_LED = LED_Onset_Zero_Start;
            % t_puff = AirPuff_LED_Onset_Aligned_End;
            % t1 = t_LED-0.01;
            % t2 = t_LED;


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
                long_block_data = [long_block_data; FEC_norm_interp];
                verticalLineColor = 'r'; % Red for long trials
                longBlockAirPuffColor = [0.5, 1.0, 0.5]; % Light green for long trials
                x_fill_long = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
                x_fill_longLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];

            else
                short_block_data = [short_block_data; FEC_norm_interp];
                verticalLineColor = 'b'; % Blue for short trials
                shortBlockAirPuffColor = [0.5, 0.5, 1.0];
                x_fill_short = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
                x_fill_shortLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];

                %  % Plot the vertical line for AirPuff
                % line([LED_Puff_ISI_end, LED_Puff_ISI_end], [0 1], 'Color', verticalLineColor, 'LineWidth', 1.5);

               
            end
        else
            warning('Skipping trial %d in file %s due to dimension mismatch.', ctr_trial, data_files(i).name);
        end
        
        
end

    end

  
% Define a colormap from dim blue to dark blue
num_colors = length(data_files);

% Define the RGB values for dim blue and dark blue
dimBlue = [0.5, 0.5, 1];  % Dim blue (lighter intensity)
darkBlue = [0, 0, 0.5];   % Dark blue (low intensity)

% Create a colormap from dim blue to dark blue
color_map_short = [linspace(dimBlue(1), darkBlue(1), num_colors)', ...
                   linspace(dimBlue(2), darkBlue(2), num_colors)', ...
                   linspace(dimBlue(3), darkBlue(3), num_colors)'];
    % Plot the averages for short blocks
    if ~isempty(short_block_data)
        FEC_norm_avg_short = mean(short_block_data, 1);
        h1 = plot(commonTime, FEC_norm_avg_short, 'Color',color_map_short(i, :), 'LineWidth', 1.2); % Blue for short blocks
        
         % Store handles for the first and last plots to use in the legend
        if i == 1 || i == num_colors
            legend_handles_short = [legend_handles_short, h1];
        end
        
    end
end



% Add LED onset shading
y_fill = [0 0 1 1];
fill(x_fill_shortLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none');
% fill(x_fill_longLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none');

% Add AirPuff shading
y_fill = [0 0 1 1];
fill(x_fill_short, y_fill, [0.5, 0.5, 1.0], 'FaceAlpha', 0.35, 'EdgeColor', 'none');  % Short blocks shading
% fill(x_fill_long, y_fill, [0.5, 1.0, 0.5], 'FaceAlpha', 0.65, 'EdgeColor', 'none');  % Long blocks shading

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
title('Long Block Trials', 'Interpreter', 'latex','Units', 'normalized', 'Position', [0.5, 1.09, 0]);

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
            else
                % short_block_data = [short_block_data; FEC_norm_interp];
                shortBlockAirPuffColor = [0.5, 0.5, 1.0];
                % x_fill_short = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
                % x_fill_shortLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];

            end
               
        else
            warning('Skipping trial %d in file %s due to dimension mismatch.', ctr_trial, data_files(i).name);
        end

end

end


num_colors = length(data_files);  % Assuming data_files is defined
% Check the size of long_block_data
% [num_sessions, num_time_points] = size(long_block_data);  % Rows = sessions, columns = time points

% % Ensure FEC_led_aligned_trimmed is the correct size
% if length(FEC_led_aligned_trimmed) ~= num_time_points
%     error('The number of time points in FEC_led_aligned_trimmed does not match the number of columns in long_block_data.');
% end
% Define RGB values for dim green and dark green
dimGreen = [0.5, 1, 0.5];   % Light green (dim green)
darkGreen = [0, 0.5, 0];    % Dark green

% Create a colormap from dim green to dark green based on num_colors
color_map_long = [linspace(dimGreen(1), darkGreen(1), num_colors)', ...
                  linspace(dimGreen(2), darkGreen(2), num_colors)', ...
                  linspace(dimGreen(3), darkGreen(3), num_colors)'];

FEC_norm_avg_long = mean(long_block_data, 1);
h2 = plot(commonTime, FEC_norm_avg_long, 'Color',color_map_long(i, :), 'LineWidth', 1.2); % Red for long blocks
% Store handles for the first and last plots to use in the legend
 if i == 1 || i == num_colors
    legend_handles_long = [legend_handles_long, h2];
 end
    
end

% Add LED onset shading
y_fill = [0 0 1 1];
fill(x_fill_longLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none');

% Add AirPuff shading
y_fill = [0 0 1 1];
fill(x_fill_long, y_fill, [0.5, 1.0, 0.5], 'FaceAlpha', 0.65, 'EdgeColor', 'none');  % Long blocks shading


% Configure plot
ylim([0 1]);
xlim([-0.2 0.6]);
xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 12);
ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
% legend(legend_handles, legend_entries, 'Interpreter', 'latex', 'location', 'northeast', 'Box', 'off');
text(0,1, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
text(0.4,1, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'green', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');

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
newFilename = 'AvgCRPlusSessions_LongShort_Subplots.pdf';
exportgraphics(gcf, newFilename, 'ContentType', 'vector');
