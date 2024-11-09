clc;close all;clear



 data_files = dir('*_EBC_*.mat');
 % data_files = dir('E2WT_EBC_V_3_12_20241019_164329.mat');
 % 

CR_threshold = 0.05;



FEC_times_trimmed = [];  % Predefine outside loop to avoid undefined variable issue



for i = 1:length(data_files)
 
    load(data_files(i).name) 
    numTrials = length(SessionData.RawEvents.Trial);
    
    % Initialize pooled data containers for short and long blocks
    pooled_short_block_data = [];
    pooled_long_block_data = [];
    
    % Containers for long and short block data for this session
    long_block_data = [];
    short_block_data = [];

% figure('units','centimeters','position',[2 2 24 26])
figure('Position', [100, 100, 600, 400]); % Width = 600, Height = 400
numCurves = 0;
totalFEC_norm = [];
leg_str = cell(1,1);
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

    % Create a color gradient from dim to dark green for short blocks
    colorMapLong = [linspace(0.7, 0, numTrials)', linspace(1, 0.5, numTrials)', linspace(0.7, 0, numTrials)']; % Dim to dark green
    
    % Create a color gradient from dim to dark blue for long blocks
    colorMapShort = [linspace(0.7, 0, numTrials)', linspace(0.7, 0, numTrials)', linspace(1, 0.5, numTrials)']; % Dim to dark blue



step = 1;
for ctr_trial = 1:step:numTrials


        %  if CheckEyeOpenTimeout is not nan, timeout occurred, continue to next trial
     
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end

        % get CheckEyeOpen if it is in session data for versions V_3_2+
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpen')
            CheckEyeOpenStart = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(1);
            CheckEyeOpenStop = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(2);
        end

        % CheckEyeOpen = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(2);
        Start = SessionData.RawEvents.Trial{1, ctr_trial}.States.Start(1);
        ITI_Pre = SessionData.RawEvents.Trial{1, ctr_trial}.States.ITI_Pre(1);
        % LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Onset(1);
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        LED_Puff_ISI_start = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
        AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
        AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;
        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;   


        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        LED_Onset_Zero_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End - LED_Onset;
        AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
        AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;



        FEC_led_aligned = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes - LED_Puff_ISI_start;
        % FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels /overallMax;
        FEC_norm_curve = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels / overallMax;

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
        
        % Trim aligned FEC data
        FEC_led_aligned_trimmed = FEC_led_aligned(start_idx : stop_idx);
        FEC_trimmed = FEC_norm_curve(start_idx:stop_idx);
        FEC_times_trimmed = FEC_led_aligned(start_idx:stop_idx);  % Update FEC_times_trimmed correctly
   
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
commonTime = linspace(min(FEC_led_aligned_trimmed), max(FEC_led_aligned_trimmed), 100);  % Adjust 100 to the desired number of points

if(CR_plus_eval(FEC_led_aligned_trimmed,FEC_trimmed,t1,t2,t_LED,t_puff,CR_threshold))
        numCurves = numCurves+1;
        % Interpolate FEC_norm to the common time vector
        FEC_norm_interp = interp1(FEC_led_aligned_trimmed, FEC_trimmed, commonTime);
        % Accumulate the data
        totalFEC_norm = [totalFEC_norm; FEC_norm_interp];
        FEC_interpolated = interp1(FEC_led_aligned_trimmed, FEC_trimmed, common_time_vector, 'linear', 'extrap');
        FEC_norm_matrix(ctr_trial, :) = FEC_interpolated;
        
        %%Determine short long blocks
        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);

        % Accumulate data based on block type
        if isLongBlock > 0.3
            long_block_data = [long_block_data; FEC_trimmed]; % Add to long block data
            % Initialize airpuff shade colors for both long and short blocks
            % Plot shaded area for AirPuff in short blocks (blue)
            longBlockAirPuffColor = [0.5, 1.0, 0.5]; % Light green for long trials
           % Plot shaded area for AirPuff in long blocks (green)
            x_fill_long = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
            x_fill_longLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];

        else
            short_block_data = [short_block_data; FEC_trimmed]; % Add to short block data
            shortBlockAirPuffColor = [0.5, 0.5, 1.0];
            x_fill_short = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
            x_fill_shortLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];

        end

end



end

% Add session's data to the pooled data for short and long blocks
    pooled_short_block_data = [pooled_short_block_data; short_block_data];
    pooled_long_block_data = [pooled_long_block_data; long_block_data];


%% Plotting


% Calculate the average and SEM for short and long block data
average_short_block = mean(pooled_short_block_data, 1, 'omitnan');
average_long_block = mean(pooled_long_block_data, 1, 'omitnan');

sem_short_block = std(pooled_short_block_data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(pooled_short_block_data), 1));
sem_long_block = std(pooled_long_block_data, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(pooled_long_block_data), 1));

% Find the global minimum and maximum between both curves
global_min_value = min([min(average_short_block), min(average_long_block)]);
global_max_value = max([max(average_short_block), max(average_long_block)]);
% Plot averages and SEM outside the trial loop
hold on;


global_max_value = global_max_value + 0.02;
global_min_value = global_min_value - 0.02;
% Plot short block average (blue)
plot(FEC_times_trimmed, average_short_block, 'b', 'LineWidth', 1.5);
fill([FEC_times_trimmed, fliplr(FEC_times_trimmed)], [average_short_block + sem_short_block, fliplr(average_short_block - sem_short_block)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Plot long block average (green)
plot(FEC_times_trimmed, average_long_block, 'g', 'LineWidth', 1.5);
fill([FEC_times_trimmed, fliplr(FEC_times_trimmed)], [average_long_block + sem_long_block, fliplr(average_long_block - sem_long_block)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');



% Add AirPuff shading
y_fill = [0 0 1 1];
fill(x_fill_short, y_fill, [0.5, 0.5, 1.0], 'FaceAlpha', 0.35, 'EdgeColor', 'none');  % Short blocks shading
fill(x_fill_long, y_fill, [0.5, 1.0, 0.5], 'FaceAlpha', 0.65, 'EdgeColor', 'none');  % Long blocks shading


% Add LED onset shading
y_fill = [0 0 1 1];
fill(x_fill_shortLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none');
fill(x_fill_longLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none');

% Configure plot
ylim([global_min_value global_max_value]);
xlim([-0.2 0.6]);
xlabel('Time From LED Onset (s)', 'interpreter', 'latex', 'fontsize', 12);
ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontsize', 12);
% Set tick marks outward
set(gca, 'TickDir', 'out');
% set(gca, 'LooseInset', max(get(gca, 'TightInset'), 0.02));
% axis equal;
text(0,global_max_value, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
text(0.2,global_max_value, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
text(0.4,global_max_value, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 10, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
% pbaspect([1 1 1]);  % This sets a 1:1 ratio for x, y, and z.
% Plot short block average (blue) and store handle
h_short = plot(FEC_times_trimmed, average_short_block, 'b', 'LineWidth', 1.5);

% Plot long block average (red) and store handle
h_long = plot(FEC_times_trimmed, average_long_block, 'g', 'LineWidth', 1.5);

% Add legend
legend([h_short, h_long],{'Short Block Avg', 'Long Block Avg'}, 'Interpreter', 'latex', 'fontsize', 14,'Box', 'off', 'Location', 'bestoutside');
% 
% legend([h_short],{'Short Block Avg' }, 'Interpreter', 'latex', 'fontsize', 14,'Box', 'off', 'Location', 'bestoutside');

title_text(1) = {'CR$^{+}$ short-block trials above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{LED}},T_{\rm{AirPuff}})$'};
title_text(2) = {'CR$^{+}$ long-block trials above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{0.2(s)}},T_{\rm{AirPuff}})$'};
title_text(3) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $',CR_threshold)};
title_text(4) = {' '}; 
% title_text(5) = {' '};
title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',14)


    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 5
        prefixPart = nameParts{1}; % First string before the first underscore
        datePart = nameParts{6}; % Date part
        newFilename = sprintf('%s_Avg_CRPlusTrials_Short_Long_%s.pdf', prefixPart, datePart);
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end

end    




