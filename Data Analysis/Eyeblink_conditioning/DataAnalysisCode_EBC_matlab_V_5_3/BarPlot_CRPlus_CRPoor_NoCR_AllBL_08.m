clc; close all; clear;

data_files = dir('*_EBC_*.mat');
% data_files = dir('E2WT_EBC_V_3_16_20250301_161023.mat');
CR_threshold = 0.05;


good_CR_threshold = 0.05;
poor_CR_threshold = 0.02;
no_CR_threshold = 0.01;

% Initialize arrays to store classification fractions for all sessions
Good_CR_percent_Control_short = [];
Poor_CR_percent_Control_short = [];
No_CR_percent_Control_short = [];

Good_CR_percent_SD_1_short = [];
Poor_CR_percent_SD_1_short = [];
No_CR_percent_SD_1_short = [];

Good_CR_percent_SD_2_short = [];
Poor_CR_percent_SD_2_short = [];
No_CR_percent_SD_2_short = [];

Good_CR_percent_Control_long = [];
Poor_CR_percent_Control_long = [];
No_CR_percent_Control_long = [];

Good_CR_percent_SD_1_long = [];
Poor_CR_percent_SD_1_long = [];
No_CR_percent_SD_1_long = [];

Good_CR_percent_SD_2_long = [];
Poor_CR_percent_SD_2_long = [];
No_CR_percent_SD_2_long = [];

lowBL_frac_short = [];
lowBL_frac_long  = [];
lowBL_session_labels = {};

N_SD_1 = 0;
N_SD_2 = 0;
N_C = 0;

lowBL_short = 0;
lowBL_long  = 0;
total_short = 0;
total_long  = 0;

 num_sessions = length(data_files);
% SD+1
lowBL_short_SD1    = zeros(num_sessions, 1);
total_short_SD1    = zeros(num_sessions, 1);
lowBL_long_SD1     = zeros(num_sessions, 1);
total_long_SD1     = zeros(num_sessions, 1);

% SD+2
lowBL_short_SD2    = zeros(num_sessions, 1);
total_short_SD2    = zeros(num_sessions, 1);
lowBL_long_SD2     = zeros(num_sessions, 1);
total_long_SD2     = zeros(num_sessions, 1);

% Control
lowBL_short_Control = zeros(num_sessions, 1);
total_short_Control = zeros(num_sessions, 1);
lowBL_long_Control  = zeros(num_sessions, 1);
total_long_Control  = zeros(num_sessions, 1);

% Define condition-based colors
control_colors = [0.7 0.7 0.7; 0 0 0];                 % short, long
sd1_colors     = [1.0 0.6 0.6; 0.8 0 0];               % short, long
sd2_colors     = [0.8 0.6 0.8; 0.4 0 0.4];             % short, long


sessionLabels = {};
sessionTypes = {}; % Store session type (Control or SD)

% Loop through sessions
for i = 1:length(data_files)
    % Load session data
    loadedData = load(data_files(i).name);
    if ~isfield(loadedData, 'SessionData')
        warning(['SessionData not found in ' data_files(i).name]);
        continue;
    end
    SessionData = loadedData.SessionData;

    % Initialize session type flags
    isSD_1 = false;
    isSD_2 = false;

    % Determine session type
    if isfield(SessionData, 'Chemogenetics') && SessionData.Chemogenetics == 1  
        sessionTypes{end+1} = 'SD+1';
        isSD_1 = true;

    else
        sessionTypes{end+1} = 'Control';
    end

        SessionData = loadedData.SessionData;

    if isfield(SessionData, 'SleepDeprivedLabel')
        label = SessionData.SleepDeprivedLabel;
    else
        label = 'Unknown';
    end

    % Assign colors based on label
    switch label

        case 'SD+1'
            bar_colors(i, 1, :) = sd1_colors(1, :);
            bar_colors(i, 2, :) = sd1_colors(2, :);
        case 'SD+2'
            bar_colors(i, 1, :) = sd2_colors(1, :);
            bar_colors(i, 2, :) = sd2_colors(2, :);
        otherwise
            % Control / Post_EBC_SD
            bar_colors(i, 1, :) = control_colors(1, :);  % short
            bar_colors(i, 2, :) = control_colors(2, :);  % long
    end

    


    % Reset counters
    numGoodCR_short = 0; numPoorCR_short = 0; numNoCR_short = 0;
    numGoodCR_long = 0; numPoorCR_long = 0; numNoCR_long = 0;
    numShortTrials = 0; numLongTrials = 0;


    % Normalize FEC
    
    numTrials = length(SessionData.RawEvents.Trial);
    allEyeAreaPixels = [];    
        % Loop through each trial to collect eyeAreaPixels data
    for trialIdx = 1:numTrials    
       eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
       % allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data 
       allEyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.totalEllipsePixels  ;
    end
    % Find the overall maximum value across all collected eyeAreaPixels
    overallMax = max(allEyeAreaPixels);

    fps = 250; % frames per second, frequency of images
    seconds_before = 0.5;
    seconds_after = 2;
    Frames_before = fps * seconds_before;
    Frames_after = fps * seconds_after;

    for ctr_trial = 1:numTrials
        % Skip trials with CheckEyeOpenTimeout
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end

        % Skip trial if ISI does not exist (i.e., no LED_Puff_ISI field)
        if ~isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'LED_Puff_ISI')
            continue;
        end


        % Get trial data for LED onset, puff, and timings
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
        AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;

        % Align times to LED onset
        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        LED_Onset_Zero_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End - LED_Onset;
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

        
        FEC_times_trimmed = FEC_led_aligned(start_idx:stop_idx); 
        FEC_trimmed = FEC_norm(start_idx:stop_idx);

        t_LED = LED_Onset_Zero_End;
        t_puff = AirPuff_LED_Onset_Aligned_Start;
        

         % Apply smoothing to reduce noise
        FEC_led_aligned_trimmed_smooth = smoothdata(FEC_times_trimmed, 'movmean', 5); % Moving average
        FEC_trimmed_smooth = smoothdata(FEC_trimmed, 'movmean', 5); % Moving average
                %% ---------- FFT-based denoising of FEC_trimmed ----------
        Fs = fps;                          % sampling rate (250 Hz)
        x  = FEC_trimmed(:);               % make column
        
        % Remove DC mean to avoid a big spike at 0 Hz
        x_detr = x - mean(x);
        
        N = numel(x_detr);
        X = fft(x_detr);
        f = (0:N-1).' * (Fs/N);            % frequency axis in Hz
        
        % ---- Choose a cutoff frequency for the CR dynamics ----
        % Eyelid CR is slow; noise is the fast "wavy" stuff.
        % Try f_cut = 15–25 Hz and adjust by eye.
        f_cut = 20;                        % <--- TUNE THIS
        
        % Keep only low frequencies (|f| <= f_cut)
        keep = (f <= f_cut) | (f >= Fs - f_cut);   % include symmetric high end
        X_filt = X .* keep;
        
        x_filt = real(ifft(X_filt));       % back to time domain
        x_filt = x_filt + mean(x);         % put DC level back
        
        % Use the filtered FEC from now on
        FEC_trimmed = x_filt;
        %% --------------------------------------------------------


        
        % Apply smoothing to reduce noise
        FEC_led_aligned_trimmed_smooth1 = smoothdata(FEC_led_aligned_trimmed_smooth, 'movmean', 5); % Moving average
        FEC_trimmed_smooth1 = smoothdata(FEC_trimmed, 'movmean', 5); % Moving average

        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;

        % Define baseline region for individual trial
        baseline_idx = FEC_led_aligned_trimmed_smooth1 >= -0.2 & FEC_led_aligned_trimmed_smooth1 <= 0;
        trial_baseline = mean(FEC_trimmed_smooth1(baseline_idx), 'omitnan');


    % Track total and lowBL trial counts for each condition
    switch label
        case 'SD+1'
            if isLongBlock
                total_long_SD1(i) = total_long_SD1(i) + 1;
                if trial_baseline <= 0.5
                    lowBL_long_SD1(i) = lowBL_long_SD1(i) + 1;
                end
            else
                total_short_SD1(i) = total_short_SD1(i) + 1;
                if trial_baseline <= 0.5
                    lowBL_short_SD1(i) = lowBL_short_SD1(i) + 1;
                end
            end
            bar_colors(i, 1, :) = sd1_colors(1, :); % short
            bar_colors(i, 2, :) = sd1_colors(2, :); % long
    
        case 'SD+2'
            if isLongBlock
                total_long_SD2(i) = total_long_SD2(i) + 1;
                if trial_baseline <= 0.5
                    lowBL_long_SD2(i) = lowBL_long_SD2(i) + 1;
                end
            else
                total_short_SD2(i) = total_short_SD2(i) + 1;
                if trial_baseline <= 0.5
                    lowBL_short_SD2(i) = lowBL_short_SD2(i) + 1;
                end
            end
            bar_colors(i, 1, :) = sd2_colors(1, :);
            bar_colors(i, 2, :) = sd2_colors(2, :);
    
        otherwise  % 'Control' or any other label
            if isLongBlock
                total_long_Control(i) = total_long_Control(i) + 1;
                if trial_baseline <= 0.5
                    lowBL_long_Control(i) = lowBL_long_Control(i) + 1;
                end
            else
                total_short_Control(i) = total_short_Control(i) + 1;
                if trial_baseline <= 0.5
                    lowBL_short_Control(i) = lowBL_short_Control(i) + 1;
                end
            end
            bar_colors(i, 1, :) = control_colors(1, :);
            bar_colors(i, 2, :) = control_colors(2, :);
    end



        good_CR_threshold = 0.05;
        poor_CR_threshold = 0.02;
        % Classify CR response
        % CR_category = classify_CR(FEC_times_trimmed, FEC_trimmed, t_LED,t_puff, CR_threshold);
        CR_category = classify_CR_05(FEC_led_aligned_trimmed_smooth1, FEC_trimmed_smooth1, t_LED, t_puff , good_CR_threshold, poor_CR_threshold);

        % Store classification results
  % if trial_baseline < 0.5     
        if isLongBlock
            numLongTrials = numLongTrials + 1;


            switch CR_category
                case 'Good CR', numGoodCR_long = numGoodCR_long + 1;
                case 'Poor CR', numPoorCR_long = numPoorCR_long + 1;
                case 'No CR', numNoCR_long = numNoCR_long + 1;
            end
        else
            numShortTrials = numShortTrials + 1;
 
            switch CR_category
                case 'Good CR', numGoodCR_short = numGoodCR_short + 1;
                case 'Poor CR', numPoorCR_short = numPoorCR_short + 1;
                case 'No CR', numNoCR_short = numNoCR_short + 1;
            end
        end
  % end   
end

    lowBL_frac_short(end+1) = lowBL_short / max(total_short, 1);
    lowBL_frac_long(end+1)  = lowBL_long / max(total_long, 1);
    
    % For labels, reuse date extraction from filename
    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 6
        datePart = nameParts{6};
        try
            sessionDate = datetime(datePart, 'InputFormat', 'yyyyMMdd');
            lowBL_session_labels{end+1} = datestr(sessionDate, 'yyyy-mm-dd');
        catch
            lowBL_session_labels{end+1} = ['Session ', num2str(i)];
        end
    else
        lowBL_session_labels{end+1} = ['Session ', num2str(i)];
    end


    % Store session results

    Good_CR_percent_Control_short(end+1) = ((~isSD_1) && (~isSD_2)) * numGoodCR_short / max(numShortTrials, 1);
    Poor_CR_percent_Control_short(end+1) = ((~isSD_1) && (~isSD_2)) * numPoorCR_short / max(numShortTrials, 1);
    No_CR_percent_Control_short(end+1)   = ((~isSD_1) && (~isSD_2)) * numNoCR_short / max(numShortTrials, 1);

    Good_CR_percent_SD_1_short(end+1) = isSD_1 * numGoodCR_short / max(numShortTrials, 1);
    Poor_CR_percent_SD_1_short(end+1) = isSD_1 * numPoorCR_short / max(numShortTrials, 1);
    No_CR_percent_SD_1_short(end+1) = isSD_1 * numNoCR_short / max(numShortTrials, 1);

    Good_CR_percent_SD_2_short(end+1) = isSD_2 * numGoodCR_short / max(numShortTrials, 1);
    Poor_CR_percent_SD_2_short(end+1) = isSD_2 * numPoorCR_short / max(numShortTrials, 1);
    No_CR_percent_SD_2_short(end+1) = isSD_2 * numNoCR_short / max(numShortTrials, 1);

    Good_CR_percent_Control_long(end+1) = ((~isSD_1) && (~isSD_2)) * numGoodCR_long / max(numLongTrials, 1);
    Poor_CR_percent_Control_long(end+1) = ((~isSD_1) && (~isSD_2)) * numPoorCR_long / max(numLongTrials, 1);
    No_CR_percent_Control_long(end+1)   = ((~isSD_1) && (~isSD_2)) * numNoCR_long / max(numLongTrials, 1);

    Good_CR_percent_SD_1_long(end+1) = isSD_1 * numGoodCR_long / max(numLongTrials, 1);
    Poor_CR_percent_SD_1_long(end+1) = isSD_1 * numPoorCR_long / max(numLongTrials, 1);
    No_CR_percent_SD_1_long(end+1) = isSD_1 * numNoCR_long / max(numLongTrials, 1);

    Good_CR_percent_SD_2_long(end+1) = isSD_2 * numGoodCR_long / max(numLongTrials, 1);
    Poor_CR_percent_SD_2_long(end+1) = isSD_2 * numPoorCR_long / max(numLongTrials, 1);
    No_CR_percent_SD_2_long(end+1) = isSD_2 * numNoCR_long / max(numLongTrials, 1);


    % Generate session label
    if isfield(SessionData.Info, 'SessionDate')
        sessionLabels{end+1} = SessionData.Info.SessionDate;
    else
        sessionLabels{end+1} = ['Session ', num2str(i)];
    end

    if isfield(SessionData, 'Chemogenetics') && SessionData.Chemogenetics == 1 
                N_SD_1 = N_SD_1 +1;
    elseif SessionData.SleepDeprived == 4
                N_SD_2 = N_SD_2 +1;
    else            
                N_C = N_C+1;
    end
end

lowBL_frac_short = zeros(num_sessions, 1);
lowBL_frac_long  = zeros(num_sessions, 1);
bar_colors = zeros(num_sessions, 2, 3);   % (session × short/long × RGB)

darkBlue  = [0.0 0.20 0.60];
mediumBlue = [0.0 0.45 0.75];
lightBlue = [0.65 0.85 0.95];
Control_color_short = [0.5 0.5 0.5];     % grey
Control_color_long = [0 0 0];     % black
% 
% for i = 1:num_sessions
%     switch sessionTypes{i}
%         case 'SD+1'
%             % BOTH short and long bars BLUE
%             bar_colors(i,1,:) = mediumBlue;   % short
%             bar_colors(i,2,:) = darkBlue;     % long
% 
%         case 'SD+2'
%             bar_colors(i,1,:) = SD2_color;
%             bar_colors(i,2,:) = SD2_color;
% 
%         otherwise   % control
%             bar_colors(i,1,:) = Control_color_short;
%             bar_colors(i,2,:) = Control_color_long;
%     end
% end
% 
% figure('Color', 'w', 'Position', [100, 100, 1400, 600]);
% hold on;
% 
% for i = 1:num_sessions
%     b1 = bar(i - 0.15, lowBL_frac_short(i), 0.3, ...
%         'FaceColor', squeeze(bar_colors(i, 1, :)), 'EdgeColor', 'none');
%     b2 = bar(i + 0.15, lowBL_frac_long(i), 0.3, ...
%         'FaceColor', squeeze(bar_colors(i, 2, :)), 'EdgeColor', 'none');
% end
% set(gca, 'XTick', 1:num_sessions, ...
%          'XTickLabel', sessionLabels, ...
%          'XTickLabelRotation', 45, ...
%          'FontSize', 10, ...
%          'FontName', 'Times New Roman', ...
%          'TickDir', 'out', ...
%          'Box', 'off');
% 
% ylim([0 1]);
% ylabel('Fraction of Low-BL Trials', 'Interpreter', 'latex');
% xlabel('Session Date', 'Interpreter', 'latex');
% title('Fraction of Low-Baseline Trials per Session', 'Interpreter', 'latex','FontSize', 14);
% 
% % Create dummy bars for legend
% h1 = bar(nan, nan, 'FaceColor', sd1_colors(1,:), 'EdgeColor', 'none');
% h2 = bar(nan, nan, 'FaceColor', sd1_colors(2,:), 'EdgeColor', 'none');
% h3 = bar(nan, nan, 'FaceColor', sd2_colors(1,:), 'EdgeColor', 'none');
% h4 = bar(nan, nan, 'FaceColor', sd2_colors(2,:), 'EdgeColor', 'none');
% h5 = bar(nan, nan, 'FaceColor', control_colors(1,:), 'EdgeColor', 'none');
% h6 = bar(nan, nan, 'FaceColor', control_colors(2,:), 'EdgeColor', 'none');
% 
% legend([h1, h2, h3, h4, h5, h6], ...
%     {'SD+1 Short', 'SD+1 Long', ...
%      'SD+2 Short', 'SD+2 Long', ...
%      'Control Short', 'Control Long'}, ...
%     'Location', 'northeastoutside', ...
%     'Interpreter', 'latex', 'FontSize', 12);
% legend boxoff;
% 
% % Save
% exportgraphics(gcf, 'LowBL_Fraction_Per_Session.pdf', 'ContentType', 'vector');

% Define stacked bar values
stackedFractions = [
    Good_CR_percent_SD_1_short; Poor_CR_percent_SD_1_short; No_CR_percent_SD_1_short;
    Good_CR_percent_SD_1_long; Poor_CR_percent_SD_1_long; No_CR_percent_SD_1_long;
    Good_CR_percent_SD_2_short; Poor_CR_percent_SD_2_short; No_CR_percent_SD_2_short;
    Good_CR_percent_SD_2_long; Poor_CR_percent_SD_2_long; No_CR_percent_SD_2_long;
    Good_CR_percent_Control_short; Poor_CR_percent_Control_short; No_CR_percent_Control_short;
    Good_CR_percent_Control_long; Poor_CR_percent_Control_long; No_CR_percent_Control_long
    
]';

% stackedFractions = zeros(numSessions, 6);  % [G_short, P_short, N_short, G_long, P_long, N_long]
% barColors = [same across all SD+1, then SD+2, then Control];

% Define x-axis positions and spacing
numSessions = length(sessionLabels);
x_positions = 1:numSessions;
x_offset = 0.11;  % Adjust for visual separation
bar_width = 0.35; % Adjust bar width for readability

% stackedFractions = zeros(numSessions, 6);  % [G_short, P_short, N_short, G_long, P_long, N_long]

% Offset x positions for Short and Long trials
x_positions_short = x_positions - x_offset;
x_positions_long  = x_positions + x_offset;

% Define colors for each category (Short & Long Trials)
barColors = [


    
    % Chemogenetics - Short (Blues)
    0.0, 0.20, 0.60;   % Good CR (Dark Blue)
    0.0, 0.45, 0.75;   % Poor CR (Blue)
    0.65, 0.85, 0.95;  % No CR (Light Blue)

    
    % Chemogenetics - Long (Blues)
    0.0, 0.20, 0.60;   % Good CR (Dark Blue)
    0.0, 0.45, 0.75;   % Poor CR (Blue)
    0.65, 0.85, 0.95;  % No CR (Light Blue)

    % SD_2 - Short (Purples)
    0.4, 0.0, 0.4;   % Good CR (Purple)
    0.6, 0.2, 0.6;   % Poor CR (Lavender)
    0.8, 0.6, 0.8;   % No CR (Light Purple)

    % SD_2 - Long (Purples)
    0.4, 0.0, 0.4;   % Good CR (Purple)
    0.6, 0.2, 0.6;   % Poor CR (Lavender)
    0.8, 0.6, 0.8;   % No CR (Light Purple)

    % Control - Short
    0.0, 0.0, 0.0;      % Good CR (Black)
    0.5, 0.5, 0.5;      % Poor CR (light Gray)
    0.83, 0.83, 0.83;   % No CR (Dim light Gray)

    % Control - Long
    0.0, 0.0, 0.0;      % Good CR (Black)
    0.5, 0.5, 0.5;      % Poor CR (light Gray)
    0.83, 0.83, 0.83;   % No CR (Dim light Gray)


];


% Check if SD_1 exists
has_SD_1 = any(Good_CR_percent_SD_1_short) || any(Poor_CR_percent_SD_1_short) || any(No_CR_percent_SD_1_short) || ...
         any(Good_CR_percent_SD_1_long)  || any(Poor_CR_percent_SD_1_long)  || any(No_CR_percent_SD_1_long);

% Check if SD_2 exists
has_SD_2 = any(Good_CR_percent_SD_2_short) || any(Poor_CR_percent_SD_2_short) || any(No_CR_percent_SD_2_short) || ...
         any(Good_CR_percent_SD_2_long)  || any(Poor_CR_percent_SD_2_long)  || any(No_CR_percent_SD_2_long);

% Check if Control exists
has_Control = any(Good_CR_percent_Control_short) || any(Poor_CR_percent_Control_short) || any(No_CR_percent_Control_short) || ...
              any(Good_CR_percent_Control_long)  || any(Poor_CR_percent_Control_long)  || any(No_CR_percent_Control_long);

% Create figure
figure;
hold on;


% Plot SD_1 sessions
if has_SD_1
    hBars_SD_1_short = bar(x_positions_short, stackedFractions(:, 1:3), 'stacked', 'BarWidth', bar_width);
    hBars_SD_1_long  = bar(x_positions_long,  stackedFractions(:, 4:6), 'stacked', 'BarWidth', bar_width);

    for k = 1:3
        set(hBars_SD_1_short(k), 'FaceColor', barColors(k, :), 'EdgeColor', 'none');
        set(hBars_SD_1_long(k),  'FaceColor', barColors(k + 3, :), 'EdgeColor', 'none');
    end
end

% Plot SD_2 sessions
if has_SD_2
    hBars_SD_2_short = bar(x_positions_short, stackedFractions(:, 7:9), 'stacked', 'BarWidth', bar_width);
    hBars_SD_2_long  = bar(x_positions_long,  stackedFractions(:, 10:12), 'stacked', 'BarWidth', bar_width);

    for k = 1:3
        set(hBars_SD_2_short(k), 'FaceColor', barColors(k + 6, :), 'EdgeColor', 'none');
        set(hBars_SD_2_long(k),  'FaceColor', barColors(k + 9, :), 'EdgeColor', 'none');
    end
end

% Plot Control sessions
if has_Control
    hBars_Control_short = bar(x_positions_short, stackedFractions(:, 13:15), 'stacked', 'BarWidth', bar_width);
    hBars_Control_long  = bar(x_positions_long,  stackedFractions(:, 16:18), 'stacked', 'BarWidth', bar_width);

    for k = 1:3
        set(hBars_Control_short(k), 'FaceColor', barColors(k + 12, :), 'EdgeColor', 'none');
        set(hBars_Control_long(k),  'FaceColor', barColors(k + 15, :), 'EdgeColor', 'none');
    end
end
% Customize the axes
set(gca, 'XTick', x_positions, 'XTickLabel', sessionLabels, 'XTickLabelRotation', 45);
ylabel('CR+ Fraction', 'Interpreter', 'latex', 'FontSize', 12);
xlabel('Sessions', 'Interpreter', 'latex', 'FontSize', 12);
ylim([0 1]);
set(gca, 'FontSize', 12, 'TickDir', 'out', 'Box', 'off'); % Box off
title({'SA13 FEC Fraction of Clessified CRs (Short and Long Trials)'; ...
       sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f$', CR_threshold)}, ...
       'Interpreter', 'latex', 'FontSize', 14);

% Adjust bar thickness
hBars = findall(gca, 'Type', 'Bar'); % Find all bar objects
for i = 1:numel(hBars)
    hBars(i).BarWidth = 0.2; % Increase bar thickness
end

legend boxoff;
legend off;
legend([hBars_Control_short(3), hBars_Control_short(2), hBars_Control_short(1)], ...
    {...
        'No CR Control', 'Poor CR Control', 'Good CR Control' ...
    }, 'Location', 'bestoutside', 'Interpreter', 'latex', 'FontSize', 10);


% Loop over each data file to collect session dates
sessionDates = [];

for i = 1:length(data_files)
    [~, name, ~] = fileparts(data_files(i).name);
    nameParts = split(name, '_');
    if length(nameParts) >= 6
        datePart = nameParts{6};
        try
            sessionDate = datetime(datePart, 'InputFormat', 'yyyyMMdd');  % Convert to datetime
            sessionDates = [sessionDates; sessionDate];  % Collect dates
        catch
            warning(['Invalid date format in file: ', name]); % Handle incorrect filenames
        end
    end
end

% Ensure sessionDates is not empty before applying min/max
if ~isempty(sessionDates)
    firstSessionDate = min(sessionDates);
    lastSessionDate = max(sessionDates);
    session_date_info = ['Sessions from ', datestr(firstSessionDate, 'mm/dd/yyyy'), ...
                         ' to ', datestr(lastSessionDate, 'mm/dd/yyyy')];
else
    session_date_info = 'No valid session dates found';
end

% Correct legend count for Short and Long
% N_short = N_C_short + N_SD_short;
% N_long = N_C_long + N_SD_long;

% Prepare session count information
% session_count_info = ['Control Sessions: ', num2str(N_C), ...
%                       ' | SD+1 Sessions: ', num2str(N_SD_1),...
%                       ' | SD+1 Sessions: ', num2str(N_SD_2)];

% session_count_info = ['Control Sessions: ', num2str(N_C), newline, ...
%                       'SD+1 Sessions: ', num2str(N_SD_1), newline, ...
%                       'SD+2 Sessions: ', num2str(N_SD_2)];
% 
% % Combine both session counts and session date info
% full_annotation_text = {session_count_info, session_date_info};
% 
% % Add annotation text box to figure
% annotation('textbox', [0.83, 0.45, 0.3, 0.1], 'String', full_annotation_text, ...
%     'EdgeColor', 'none', 'Interpreter', 'latex', 'FontSize', 10, 'FontName', 'Times New Roman');
% Save plot
set(gcf, 'Position', [100, 100, 1200, 600]);
exportgraphics(gcf, 'BarPlt_FECFraction_ClassifiedCRs_AllBL_Sessions.pdf', 'ContentType', 'vector');
hold off;