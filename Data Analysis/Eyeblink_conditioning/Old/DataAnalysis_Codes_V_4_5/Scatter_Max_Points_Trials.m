clc; close all; clear

% % Load all session data files
% data_files = dir('*_EBC_*.mat');
% % Load all session data files
data_files = dir('E4L7_EBC_V_3_12_20241024_150241.mat');
FEC_times_trimmed = [];
CR_threshold = 0.05; % Define the CR+ threshold value (if it needs to be adjusted)

figure('Position', [100, 100, 1200, 800]); % Increase figure size for better layout control
hold on;
% Loop over each data file
for i = 1:length(data_files)

    % figure;   % Prepare figure for plotting multiple trials

    load(data_files(i).name);
    numTrials = length(SessionData.RawEvents.Trial);

    % Assuming the structure 'SessionData' is loaded
    nTrials = SessionData.nTrials;  % Number of trials
    sectionSize = nTrials/4;
    % Initialize arrays to store results for short and long blocks
    maxFECValues_short = [];
    maxFECValues_long = [];
    distancesFromAirpuff_short = [];
    distancesFromAirpuff_long = [];
    timesOfMaxFEC_short = [];
    timesOfMaxFEC_long = [];



    % Calculate the global minimum and maximum based on allMaxFECValues
    global_min_value = [];
    global_max_value = [];

    allMaxFECTimes = [];
    allMaxFECValues = [];
    allBlockTypes = []; % Store 1 for long and 0 for short
    trialNumbers = [];  % Store trial numbers for separated plotting
    validTrialNumbers = [];  % Store trial numbers for valid trials

    % trialNumbers = (1:nTrials)'; 

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
       

    % Determine Block Type: Long or Short
    isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - ...
                  SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;

    % Create a window mask based on block type and extract data
    if isLongBlock
        window_mask = (FEC_times_trimmed > 0.2) & (FEC_times_trimmed < AirPuff_LED_Onset_Aligned_Start);
    else
        window_mask = (FEC_times_trimmed > 0) & (FEC_times_trimmed < AirPuff_LED_Onset_Aligned_Start);
    end
    
    fec_values_in_window = FEC_trimmed(window_mask);
    fec_times_in_window = FEC_times_trimmed(window_mask);

    if ~isempty(fec_values_in_window)
        % Find the max FEC value and its time
        [maxFEC, idx] = max(fec_values_in_window);
        maxFECTime = fec_times_in_window(idx);

        % Store max points for plotting after the loop
        allMaxFECTimes = [allMaxFECTimes; maxFECTime];
        allMaxFECValues = [allMaxFECValues; maxFEC];
        allBlockTypes = [allBlockTypes; isLongBlock];
        validTrialNumbers = [validTrialNumbers; ctr_trial];  % Track valid trial numbers
    end

    % Shading for short blocks (light blue for visualization)
    shortBlockAirPuffColor = [0.5, 0.5, 1.0];  % Light blue for short trials
    x_fill = [0.199, 0.219, 0.219, 0.199];
    y_fill = [0 0 1.5 1.5];
    Blue = fill(x_fill, y_fill, shortBlockAirPuffColor, 'FaceAlpha', 0.35, 'EdgeColor', 'none');  % Short blocks shading

  
    end  % End of trial loop



% Calculate the global minimum and maximum based on allMaxFECValues
global_min_value = min(allMaxFECValues);
global_max_value = max(allMaxFECValues);

% Create continuous shaded regions for short and long blocks
currentBlockType = allBlockTypes(1);
startIndex = validTrialNumbers(1);

for i = 2:length(allBlockTypes)
    if allBlockTypes(i) ~= currentBlockType || i == length(allBlockTypes)
        % Draw the shaded region for the current block
        endIndex = validTrialNumbers(i - 1);
        if i == length(allBlockTypes)  % Extend to the last trial if at the end
            endIndex = validTrialNumbers(i);
        end
        
        % Draw shaded area based on block type
        if currentBlockType == 1  % Long block
            fill([startIndex - 0.5, endIndex + 0.5, endIndex + 0.5, startIndex - 0.5], ...
                 [min(allMaxFECValues) - 0.1, min(allMaxFECValues) - 0.1, max(allMaxFECValues) + 0.1, max(allMaxFECValues) + 0.1], ...
                 [0.5, 1.0, 0.5], 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Light green for long blocks
        else  % Short block
            fill([startIndex - 0.5, endIndex + 0.5, endIndex + 0.5, startIndex - 0.5], ...
                 [min(allMaxFECValues) - 0.1, min(allMaxFECValues) - 0.1, max(allMaxFECValues) + 0.1, max(allMaxFECValues) + 0.1], ...
                 [0.5, 0.5, 1.0], 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Light blue for short blocks
        end
        
        % Update current block type and start index
        currentBlockType = allBlockTypes(i);
        startIndex = validTrialNumbers(i);
    end
end
% Calculate the number of short and long trials
numShortTrials = sum(allBlockTypes == 0);  % Counts the number of short trials (block type 0)
numLongTrials = sum(allBlockTypes == 1);   % Counts the number of long trials (block type 1)

% Plot max FEC points for long and short blocks
h1 = scatter(validTrialNumbers(allBlockTypes == 1), allMaxFECValues(allBlockTypes == 1), 36, 'g', 'filled', 'DisplayName', 'Long Block');
h2 = scatter(validTrialNumbers(allBlockTypes == 0), allMaxFECValues(allBlockTypes == 0), 36, 'b', 'filled', 'DisplayName', 'Short Block');


% Create the legend with trial counts and total number of trials included
legend([h1, h2], ...
    ['Long Trials, Total:', num2str(numLongTrials), ' out of ', num2str(nTrials), ')'], ...
    ['Short Trials, Total:', num2str(numShortTrials), ' out of ', num2str(nTrials), ')'],...
    'Location', 'best', 'Interpreter', 'latex', 'FontSize', 12, 'Box', 'off', 'Location', 'bestoutside');



% Add vertical lines to indicate actual block transitions with color coding
for i = 2:length(allBlockTypes)
    if allBlockTypes(i) ~= allBlockTypes(i - 1)  % Transition between block types
        if allBlockTypes(i) == 1  % Transition to a long block
            xline(validTrialNumbers(i) - 0.5, '--g', 'LineWidth', 1, 'HandleVisibility', 'off'); % Green dashed line for long block transition
        else  % Transition to a short block
            xline(validTrialNumbers(i) - 0.5, '--b', 'LineWidth', 1, 'HandleVisibility', 'off'); % Blue dashed line for short block transition
        end
    end
end


global_max_value = global_max_value + 0.05;
global_min_value = global_min_value - 0.05;

% Set axis limits and labels
xlim([1, nTrials]);
ylim([global_min_value global_max_value]);
xlabel('Trial Number','Interpreter', 'latex', 'FontSize',12);
ylabel('Max FEC','Interpreter', 'latex', 'FontSize',12);
title_text(1) = {'Max Points for Each Trial with Shaded Backgrounds and Block Separation Lines'};
title_text(2) = {' '};
title_text = title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',14);


% grid on;
set(gca, 'TickDir', 'out');
hold off;


end
 for i = 1:length(data_files)
    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 5
        prefixPart = nameParts{1}; % First string before the first underscore
        datePart = nameParts{6}; % Date part
        newFilename = sprintf('%s_Scatter_MAX_Points_%s.pdf', prefixPart, datePart);
        exportgraphics(gcf, newFilename, 'ContentType', 'vector');
    else
        error('Filename does not have the expected format');
    end
 end
