clc; close all; clear

data_files = dir('*_EBC_*.mat');
CR_threshold = 0.05;
sessionDates = [];
N_SD_short = 0;
N_C_short = 0;
% Initialize counters for Mega Session
mega_CR_plus_long_SD = 0; % Total number of CR+ trials for SD
mega_CR_plus_short_SD = 0;
mega_trials_short_SD = 0;  % Total number of trials for SD
mega_trials_long_SD = 0;
mega_CR_plus_long_Control = 0; % Total number of CR+ trials for Control
mega_trials_long_Control = 0; % Total number of trials for Control
mega_CR_plus_short_Control = 0; % Total number of CR+ trials for Control
mega_trials_short_Control = 0; % Total number of trials for Control

session_dates = {}; % Initialize cell array to store session dates

for i = 1:length(data_files)
    

    % Load the current data file
    loadedData = load(data_files(i).name);
    
    % Check if the loaded file contains SessionData
    if isfield(loadedData, 'SessionData')
        SessionData = loadedData.SessionData;
    else
        warning(['SessionData not found in ' data_files(i).name]);
        continue;
    end
    
    % Check if the session is Sleep-Deprived or Control
    isSD = isfield(SessionData, 'SleepDeprived') && SessionData.SleepDeprived == 1;

    % Reset counters for this session
    numCurves_long = 0;
    numCurves_short = 0;
    numLongTrials = 0;
    numShortTrials = 0;
    
    numTrials = length(SessionData.RawEvents.Trial);
    
    % Initialize an empty array to store all eyeAreaPixels values
    allEyeAreaPixels = [];

    % Loop through each trial to collect eyeAreaPixels data
    for trialIdx = 1:numTrials
        if isfield(SessionData.RawEvents.Trial{1, trialIdx}.Data, 'eyeAreaPixels')
            eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
            allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data
        end
    end
    % Find the overall maximum value across all collected eyeAreaPixels
    overallMax = max(allEyeAreaPixels);

    step = 1;

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

        isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;

        t_LED = LED_Onset_Zero_Start;
        t_puff = AirPuff_LED_Onset_Aligned_Start;
        % Check if the trial is a CR+ trial
        is_CR_plus = CR_plus_eval_dev(FEC_times_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);
     
  
   if isLongBlock
        if isSD
            mega_trials_long_SD = mega_trials_long_SD + 1;
            if is_CR_plus
                mega_CR_plus_long_SD = mega_CR_plus_long_SD + 1;
            end
        else
            mega_trials_long_Control = mega_trials_long_Control + 1;
            if is_CR_plus
                mega_CR_plus_long_Control = mega_CR_plus_long_Control + 1;
            end
        end
   else
        if isSD
            mega_trials_short_SD = mega_trials_short_SD + 1;
            if is_CR_plus
                mega_CR_plus_short_SD = mega_CR_plus_short_SD + 1;
            end
        else
            mega_trials_short_Control = mega_trials_short_Control + 1;
            if is_CR_plus
                mega_CR_plus_short_Control = mega_CR_plus_short_Control + 1;
            end
        end
   end    
 

    end
   
end

% Compute CR fractions for Mega Sessions
CR_fraction_mega_long_SD = mega_CR_plus_long_SD / max(mega_trials_long_SD, 1);
CR_fraction_mega_short_SD = mega_CR_plus_short_SD / max(mega_trials_short_SD, 1);
CR_fraction_mega_long_Control = mega_CR_plus_long_Control / max(mega_trials_long_Control, 1);
CR_fraction_mega_short_Control = mega_CR_plus_short_Control / max(mega_trials_short_Control, 1);

% Plotting the Mega Session Bar Plot
figure;
hold on;

% Plot bars
% b1 = bar(1, CR_fraction_mega_short_Control, 0.3, 'FaceColor', [0.2 0.6 0.8]); % Control Short
% b2 = bar(2, CR_fraction_mega_long_Control, 0.3, 'FaceColor', [0.2 0.8 0.2]); % Control Long
% b3 = bar(3, CR_fraction_mega_short_SD, 0.3, 'FaceColor', [0.8 0.6 0.2]); % SD Short
% b4 = bar(4, CR_fraction_mega_long_SD, 0.3, 'FaceColor', [0.8 0.2 0.2]); % SD Long

% Define bar positions
x_positions = [1, 1.25, 2, 2.25]; % Grouped positions for Short and Long bars

% Plot bars with updated positions and colors
b1 = bar(x_positions(1), CR_fraction_mega_short_Control, 0.25, 'FaceColor', [0.7 0.7 0.7]); % Control Short (Grey)
hold on;
b2 = bar(x_positions(2), CR_fraction_mega_short_SD, 0.25, 'FaceColor', [1 0 0]); % SD Short (Red)
b3 = bar(x_positions(3), CR_fraction_mega_long_Control, 0.25, 'FaceColor', [0.4, 0.4, 0.4]); % Control Long (Dark Grey)
b4 = bar(x_positions(4), CR_fraction_mega_long_SD, 0.25, 'FaceColor', [0.6, 0, 0]); % SD Long (Dark Red)


% Customize X-axis labels
set(gca, 'XTick', [1.125, 2.125]); % Center of grouped bars
set(gca, 'XTickLabel', {'Short Blocks', 'Long Blocks'});

% % Add labels and legend
% xticks([1, 2, 3, 4]);
% xticklabels({'Control Short', 'Control Long', 'SD Short', 'SD Long'});
% ylabel('CR+ Fraction', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 12);
% title('Mega Session CR+ Fractions', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 14);
% legend([b1, b2, b3, b4], {'Control Short', 'Control Long', 'SD Short', 'SD Long'}, ...
%        'Location', 'bestoutside', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 10, 'Box', 'off');
% Add labels and title
ylabel('CR+ Fraction', 'Interpreter', 'latex', 'FontSize', 14);
ylim([0 1]);
title({'\hspace{2cm}E2WT Mean Of All SD And Control Sessions CR$^{+}$ Fractions'; ...
       '\hspace{2cm}CR$^{+}$ above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{AirPuff}-0.05(s)},T_{\rm{AirPuff}})$'; ...
       sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $', CR_threshold)}, ...
       'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 14);

% Beautify axes
set(gca, 'FontSize', 14, 'TickDir', 'out');
set(gcf, 'Position', [100, 100, 800, 600]);

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
    % Determine the first and last session dates
    firstSessionDate = min(sessionDates);
    lastSessionDate = max(sessionDates);    

legend_entries = {
    
    ['Control Short (N CR Trials = ', num2str(mega_CR_plus_short_Control), ')'], ...
    ['SD Short (N CR Trials = ', num2str(mega_CR_plus_short_SD), ')'], ...
    ['Control Long (N CR Trials = ' , num2str(mega_CR_plus_long_Control), ')'],...
    ['SD Long (N CR Trials = ', num2str(mega_CR_plus_long_SD), ')'], ...

};

% Generate session date information
session_date_info = ['Sessions from ', datestr(firstSessionDate, 'mm/dd/yyyy'), ...
                     ' to ', datestr(lastSessionDate, 'mm/dd/yyyy')];

% Add session date information as a text box
annotation('textbox', [0.69, 0.17, 0.3, 0.65], 'String', session_date_info, ...
    'EdgeColor', 'none', 'Interpreter', 'latex', 'FontSize', 10, 'FontName', 'Times New Roman');

legend([b1, b2, b3, b4], legend_entries, ...
       'Location', 'bestoutside', 'Interpreter', 'latex', ...
       'FontName', 'Times New Roman', 'FontSize', 10, 'Box', 'off');

% % Customize plot
% set(gca, 'XTick', 1:2, 'XTickLabel', {'Mega Control', 'Mega SD'});
% ylabel('CR+ Fraction', 'Interpreter', 'latex');
ylim([0 1]);
title('E2WT CR+ Fraction For Mega Sessions(Pooled All Trials) Control vs SD', 'Interpreter', 'latex');
set(gca, 'FontSize', 14);
set(gcf, 'Position', [100, 100, 800, 500]);

% Save plot
exportgraphics(gcf, 'CR_plus_fractions_Mega_SD_Control_sessions.pdf', 'ContentType', 'vector');

hold off;