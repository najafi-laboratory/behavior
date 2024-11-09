clc; close all; clear

% % Load all session data files
% data_files = dir('*_EBC_*.mat');
% % Load all session data files
data_files = dir('E4L7_EBC_V_3_12_20241101_184044.mat');
FEC_times_trimmed = [];
CR_threshold = 0.05; % Define the CR+ threshold value (if it needs to be adjusted)

% Loop over each data file
for i = 1:length(data_files)

    figure; hold on;  % Prepare figure for plotting multiple trials

    load(data_files(i).name);
    numTrials = length(SessionData.RawEvents.Trial);

    % Assuming the structure 'SessionData' is loaded
    nTrials = SessionData.nTrials;  % Number of trials

    % Initialize arrays to store results for short and long blocks
    maxFECValues_short = [];
    maxFECValues_long = [];
    distancesFromAirpuff_short = [];
    distancesFromAirpuff_long = [];
    timesOfMaxFEC_short = [];
    timesOfMaxFEC_long = [];

    allEyeAreaPixels = [];

    % Parameters for creating the heatmap
    xEdges = linspace(0, 0.6, 100);  % X-axis bins
    yEdges = linspace(0, 0.8, 100);  % Y-axis bins
    
    % Collect all max points separately for short and long blocks
    shortBlockTimes = [];
    shortBlockFEC = [];
    longBlockTimes = [];
    longBlockFEC = [];

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

    % Generate color gradients for short and long blocks
    colorMapLong = [linspace(0.5, 0, nTrials)', linspace(0.7, 0.3, nTrials)', linspace(0.5, 0, nTrials)'];  % Dim to dark green
    colorMapShort = [linspace(0.5, 0, nTrials)', linspace(0.5, 0, nTrials)', linspace(0.7, 0.3, nTrials)'];  % Dim to dark blue

    % Loop through each trial
    for ctr_trial = 1:nTrials

        % Skip trials with CheckEyeOpenTimeout
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end

        % Get trial data for LED onset, puff, and timings
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
        AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;

        % Align times to LED onset
        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
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
        
        % Define a baseline window before LED onset (e.g., -0.3s to -0.1s)
        baseline_window = (FEC_times_trimmed < -0.1) & (FEC_times_trimmed > -0.3);

        % Calculate the baseline as the mean FEC value within the defined window
        if any(baseline_window)
            baseline = mean(FEC_trimmed(baseline_window), 'omitnan');
        else
            warning('No baseline window found. Setting baseline to zero.');
            baseline = 0;
        end

        % Subtract the baseline from the trimmed FEC values
        FEC_trimmed = FEC_trimmed - baseline;

        % Define CR+ condition check (use your CR+ function or condition)
        % For example:
        if CR_plus_eval(FEC_times_trimmed, FEC_trimmed, 0.1, 0.3, 0.2, 0.4, CR_threshold)  % Adjust values as needed
            % Determine Block Type: Long or Short
            isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;

            if isLongBlock
                % Window mask and data extraction for long blocks
                window_mask = (FEC_times_trimmed > 0.2) & (FEC_times_trimmed < AirPuff_LED_Onset_Aligned_Start);
                fec_values_in_window = FEC_trimmed(window_mask);
                fec_times_in_window = FEC_times_trimmed(window_mask);

                if ~isempty(fec_values_in_window)
                    [maxFEC, idx] = max(fec_values_in_window);
                    maxFECTime = fec_times_in_window(idx);

                    % Plot max FEC point with a gradient color for long block
                    h_long = plot(maxFECTime, maxFEC, '.', 'Color', colorMapLong(ctr_trial, :), 'MarkerSize', 16);

                    % Add data to containers for long blocks
                    distancesFromAirpuff_long = [distancesFromAirpuff_long, abs(0.4 - maxFECTime)];
                    maxFECValues_long = [maxFECValues_long, maxFEC];
                    timesOfMaxFEC_long = [timesOfMaxFEC_long, maxFECTime];

                    longBlockAirPuffColor = [0.5, 1.0, 0.5]; % Light green for long trials
                    % Add shade areas for long blocks
                    x_fill = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
                    y_fill = [0 0 1 1];
                    Green = fill(x_fill, y_fill, longBlockAirPuffColor, 'FaceAlpha', 0.65, 'EdgeColor', 'none');  % Long blocks shading
                    
                    
                    % Store the max points for long blocks
                    longBlockTimes = [longBlockTimes; maxFECTime];
                    longBlockFEC = [longBlockFEC; maxFEC];

                end
            else
                % Window mask and data extraction for short blocks
                window_mask = (FEC_times_trimmed > 0) & (FEC_times_trimmed < AirPuff_LED_Onset_Aligned_Start);
                fec_values_in_window = FEC_trimmed(window_mask);
                fec_times_in_window = FEC_times_trimmed(window_mask);

                if ~isempty(fec_values_in_window)
                    [maxFEC, idx] = max(fec_values_in_window);
                    maxFECTime = fec_times_in_window(idx);

                    % Plot max FEC point with a gradient color for short block
                    h_short = plot(maxFECTime, maxFEC, '.', 'Color', colorMapShort(ctr_trial, :), 'MarkerSize', 16);

                    % Add data to containers for short blocks
                    distancesFromAirpuff_short = [distancesFromAirpuff_short, abs(0.2 - maxFECTime)];
                    maxFECValues_short = [maxFECValues_short, maxFEC];
                    timesOfMaxFEC_short = [timesOfMaxFEC_short, maxFECTime];

                    % Store the max points for short blocks
                    shortBlockTimes = [shortBlockTimes; maxFECTime];
                    shortBlockFEC = [shortBlockFEC; maxFEC];
                end
            end


    % Shading for short blocks (light blue for visualization)
    shortBlockAirPuffColor = [0.5, 0.5, 1.0];  % Light blue for short trials
    x_fill = [0.199, 0.219, 0.219, 0.199];
    y_fill = [0 0 1.5 1.5];
    Blue = fill(x_fill, y_fill, shortBlockAirPuffColor, 'FaceAlpha', 0.35, 'EdgeColor', 'none');  % Short blocks shading

        end  % End of CR+ condition check




    end  % End of trial loop

    xlabel('Time Of Max FEC (sec)','Interpreter', 'latex');
    ylabel('Max FEC (baseline subtracted; norm)','Interpreter', 'latex');
    % Set tick marks to be outside
    set(gca, 'TickDir', 'out');
    set(gca, 'FontSize', 14);
    % Configure plot
    ylim([0 0.8]);
    xlim([0 0.6]);
    text(0.2,0.8, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 11, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.4,0.8, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 13, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');


    % title_text(1) = {'\vspace{2cm} average of all trials for each session \\'};
    title('Max Points Of Short And Long Blocks','Interpreter', 'latex','fontsize', 15);
   
    % legend('Short Blocks', 'Long Blocks');
    legend([Blue,Green],'Short Block','Long Block', 'Interpreter', 'latex', 'fontsize', 14,'Box', 'off', 'Location', 'bestoutside');

    % Calculate the average FEC values for short and long blocks
    averageFEC_short = nanmean(maxFECValues_short);
    averageFEC_long = nanmean(maxFECValues_long);

    % Display results
    disp(['Average FEC (Short Blocks): ', num2str(averageFEC_short)]);
    disp(['Average FEC (Long Blocks): ', num2str(averageFEC_long)]);
    disp('Normalized Distances from AirPuff (Short Blocks): ');
    disp(distancesFromAirpuff_short);
    disp('Normalized Distances from AirPuff (Long Blocks): ');
    disp(distancesFromAirpuff_long);


        [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 5
        prefixPart = nameParts{1}; % First string before the first underscore
        datePart = nameParts{6}; % Date part
        newFilename = sprintf('%s_MaxPoint_AirPuff_%s.pdf', prefixPart, datePart);
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end


    % Define zones for short and long blocks
    short_block_zone = [0, 0.2];
    long_block_zone = [0.2, 0.4];

    
    % Calculate the mean and standard error (or standard deviation) for short and long blocks
    mean_distance_short = nanmean(distancesFromAirpuff_short);
    mean_distance_long = nanmean(distancesFromAirpuff_long);
    std_distance_short = nanstd(distancesFromAirpuff_short);
    std_distance_long = nanstd(distancesFromAirpuff_long);
    
    % Calculate standard error of the mean (SEM) if desired
    sem_distance_short = std_distance_short / sqrt(length(distancesFromAirpuff_short));
    sem_distance_long = std_distance_long / sqrt(length(distancesFromAirpuff_long));
    
    % Plot the mean distances with error bars (using standard deviation or standard error)
    figure;
    hold on;

    % Define the RGB values for pale blue (example RGB values: [0.69, 0.93, 0.93])
    paleBlue = [0.69, 0.93, 0.93]; 
    dimBlue = [0.1, 0.4, 0.7]; 
    dimGreen = [0.2, 0.5, 0.2]; 
    % Plot the bar with the specified width and color
    bar(1, mean_distance_short, 0.6, 'FaceColor',  dimBlue); % Bar for short blocks with pale blue color
    % bar(1, mean_distance_short, 0.4, 'b'); % Bar for short blocks
    bar(2, mean_distance_long, 0.6, 'FaceColor',  dimGreen);  % Bar for long blocks
    
    % Add error bars (using standard deviation or standard error)
    errorbar(1, mean_distance_short, sem_distance_short, 'k', 'LineWidth', 3); % Error bar for short blocks
    errorbar(2, mean_distance_long, sem_distance_long, 'k', 'LineWidth', 3); % Error bar for long blocks

    set(gca, 'XTick', [1 2], 'XTickLabel', {'Short Blocks', 'Long Blocks'});
    ylabel('Mean Normalized Distance from AirPuff','Interpreter', 'latex');
    
    title_text(1) = {'Mean Of Max-FEC-Points Distances From AirPuff'};
    title_text(2) = {'With Standard Error'};
    title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',14)


    % legend('Short Blocks', 'Long Blocks');
    % legend('Short Block','Long Block', 'Interpreter', 'latex', 'fontsize', 14,'Box', 'off', 'Location', 'bestoutside');
    ylim([0 0.2]);
    % title(title_text ,'Fraction of CR+ Trials Across Sessions (Long and Short Blocks)','Interpreter', 'latex');
    set(gca, 'FontSize', 14);
    set(gca, 'Box', 'off');
    % Set tick marks to be outside
    set(gca, 'TickDir', 'out');


        [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 5
        prefixPart = nameParts{1}; % First string before the first underscore
        datePart = nameParts{6}; % Date part
        newFilename = sprintf('%s_Mean_MaxPoint_AirPuff_%s.pdf', prefixPart, datePart);
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end

    % Create 2D histograms (heatmaps) for short and long blocks
    [countsShort, xEdges, yEdges] = histcounts2(shortBlockTimes, shortBlockFEC, xEdges, yEdges);
    [countsLong, xEdges, yEdges] = histcounts2(longBlockTimes, longBlockFEC, xEdges, yEdges);
    
    % % Define very dim blue colormap for short blocks
    % blueCMap = [linspace(0.9, 0.6, 256)', linspace(0.9, 0.6, 256)', linspace(1, 0.8, 256)'];
    % 
    % % Define very dim green colormap for long blocks
    % greenCMap = [linspace(0.9, 0.6, 256)', linspace(1, 0.8, 256)', linspace(0.9, 0.6, 256)'];
    
    % Plot the heatmap for short blocks with blue color gradient
    figure;
    imagesc(xEdges, yEdges, countsShort');
    axis xy;  % Ensure the plot is displayed correctly
    % colormap(blueCMap);  % Apply the custom blue colormap
    colorbar;
    hold on;

    % Overlay bolded points on the heatmap for short blocks
    % plot(shortBlockTimes, shortBlockFEC, 'o', 'MarkerSize', 4, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b', 'LineWidth', 1.5);
    % Shading for short blocks (light blue for visualization)

    for trialIdx = 1:numTrials
            shortBlockAirPuffColor = [0.5, 0.5, 1.0];  % Light blue for short trials
            x_fill = [0.199, 0.219, 0.219, 0.199];
            y_fill = [0 0 1.5 1.5];
            Blue = fill(x_fill, y_fill, shortBlockAirPuffColor, 'FaceAlpha', 0.35, 'EdgeColor', 'none');  % Short blocks shading

    end



    xlabel('Time Of Max FEC (sec)','Interpreter', 'latex');
    ylabel('Max FEC (baseline subtracted; norm)','Interpreter', 'latex');
    title('Density of Max Points for Short Blocks ','Interpreter', 'latex');
    set(gca, 'FontSize', 14);
    set(gca, 'Box', 'off');
    set(gca, 'TickDir', 'out');
    text(0.2,0.8, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 13, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    % text(0.4,global_max_value, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 11, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');

    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 5
        prefixPart = nameParts{1}; % First string before the first underscore
        datePart = nameParts{6}; % Date part
        newFilename = sprintf('%s_HeatMapShortBlck_MaxPoint_AirPuff_%s.pdf', prefixPart, datePart);
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end


    % Plot the heatmap for long blocks with green color gradient
    figure;
    imagesc(xEdges, yEdges, countsLong');
    axis xy;  % Ensure the plot is displayed correctly
    % colormap(greenCMap);  % Apply the custom green colormap
    colorbar;
    hold on;

    % Overlay bolded points on the heatmap for long blocks
    % plot(longBlockTimes, longBlockFEC, 'o', 'MarkerSize', 4, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', 'LineWidth', 1.5);

            longBlockAirPuffColor = [0.5, 1.0, 0.5]; % Light green for long trials
            % Add shade areas for long blocks
            x_fill = [0.4, 0.42, 0.42, 0.4];
            y_fill = [0 0 1 1];
            Green = fill(x_fill, y_fill, longBlockAirPuffColor, 'FaceAlpha', 0.65, 'EdgeColor', 'none');  % Long blocks shading


    xlabel('Time Of Max FEC (sec)','Interpreter', 'latex');
    ylabel('Max FEC (baseline subtracted; norm)','Interpreter', 'latex');
    title('Density of Max Points for Long Blocks ','Interpreter', 'latex');
    set(gca, 'FontSize', 14);
    set(gca, 'Box', 'off');
    set(gca, 'TickDir', 'out');
    % text(0.2,global_max_value, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 11, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');
    text(0.4,0.8, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 13, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0, 'Clipping', 'on');




    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 5
        prefixPart = nameParts{1}; % First string before the first underscore
        datePart = nameParts{6}; % Date part
        newFilename = sprintf('%s_HeatMapLongBlck_MaxPoint_AirPuff_%s.pdf', prefixPart, datePart);
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end
    

end  % End of file loop