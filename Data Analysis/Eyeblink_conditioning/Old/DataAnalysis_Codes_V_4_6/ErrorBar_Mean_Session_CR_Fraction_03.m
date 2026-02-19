clc; close all; clear

data_files = dir('*_EBC_*.mat');

CR_threshold = 0.05;


% Initialize arrays to store CR+ fractions
CR_plus_fractions_SD_long = [];
CR_plus_fractions_SD_short = [];
CR_plus_fractions_Control_long = [];
CR_plus_fractions_Control_short = [];

% session_dates = {}; % Initialize cell array to store session dates
sessionDates = {}; % Initialize cell array to store session dates
N_SD_short = 0;
N_C_short = 0;
N_SD_long = 0;
N_C_long =  0;

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
            numLongTrials = numLongTrials + 1;
            if is_CR_plus
                numCurves_long = numCurves_long + 1;
            end

        else
             numShortTrials = numShortTrials + 1;
             if is_CR_plus
                numCurves_short = numCurves_short + 1;
             end
        end
    end

    % Calculate CR+ fractions
    CR_plus_fraction_long = numCurves_long / max(numLongTrials, 1);
    CR_plus_fraction_short = numCurves_short / max(numShortTrials, 1);

    % Store data based on session type
    if isSD
        CR_plus_fractions_SD_long = [CR_plus_fractions_SD_long, CR_plus_fraction_long];
        CR_plus_fractions_SD_short = [CR_plus_fractions_SD_short, CR_plus_fraction_short];
    else
        CR_plus_fractions_Control_long = [CR_plus_fractions_Control_long, CR_plus_fraction_long];
        CR_plus_fractions_Control_short = [CR_plus_fractions_Control_short, CR_plus_fraction_short];
    end
    
    if isfield(SessionData.Info, 'SessionDate')
        session_dates{i} = SessionData.Info.SessionDate;
    else
        session_dates{i} = ['Session ' num2str(i)];
    end

    if SessionData.SleepDeprived  
                N_SD_short = N_SD_short +1;
    else
                N_C_short = N_C_short+1;
    end 

end



% Calculate mean and SEM
mean_SD_long = mean(CR_plus_fractions_SD_long, 'omitnan');
sem_SD_long = std(CR_plus_fractions_SD_long, 'omitnan') / sqrt(length(CR_plus_fractions_SD_long));

mean_SD_short = mean(CR_plus_fractions_SD_short, 'omitnan');
sem_SD_short = std(CR_plus_fractions_SD_short, 'omitnan') / sqrt(length(CR_plus_fractions_SD_short));

mean_Control_long = mean(CR_plus_fractions_Control_long, 'omitnan');
sem_Control_long = std(CR_plus_fractions_Control_long, 'omitnan') / sqrt(length(CR_plus_fractions_Control_long));

mean_Control_short = mean(CR_plus_fractions_Control_short, 'omitnan');
sem_Control_short = std(CR_plus_fractions_Control_short, 'omitnan') / sqrt(length(CR_plus_fractions_Control_short));

% Plotting the bar plot for CR+ fractions (long and short blocks)
figure;
hold on;


% Define bar positions
x_positions = [1, 1.25]; % Grouped positions for Short and Long bars

% Plot bars with updated positions and colors
b1 = bar(x_positions(1), mean_Control_short, 0.25, 'FaceColor',[0.2 0.6 0.8]); 
hold on;
b2 = bar(x_positions(2), mean_Control_long, 0.25, 'FaceColor', [0.2 0.8 0.2]); 

% Add error bars
errorbar(x_positions(1), mean_Control_short, sem_Control_short, 'k', 'LineWidth', 1, 'CapSize', 10);
errorbar(x_positions(2), mean_Control_long, sem_Control_long, 'k', 'LineWidth', 1, 'CapSize', 10);
xticks(x_positions);
set(gca, 'XTickLabel', {'Short', 'Long'});

% Add legend
% Add legend
legend([b1, b2], {'Short Block', 'Long Block'}, ...
    'Location', 'northeast', 'Interpreter', 'latex', 'Box', 'off');

% Add labels and title
ylabel('CR+ Fraction', 'Interpreter', 'latex', 'FontSize', 14);
ylim([0 1]);
title({'\hspace{2cm}E5LG Mean Of All SD And Control Sessions CR$^{+}$ Fractions'; ...
       '\hspace{2cm}CR$^{+}$ above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{AirPuff}-0.05(s)},T_{\rm{AirPuff}})$'; ...
       sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $', CR_threshold)}, ...
       'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 14);

% Beautify axes
set(gca, 'FontSize', 14, 'TickDir', 'out');
set(gcf, 'Position', [100, 100, 800, 600]);

% Save plot
exportgraphics(gcf, 'Mean_CR_plus_fractions_SD_Control_Sessios.pdf', 'ContentType', 'vector');
hold off;

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

% legend_entries = {
%     ['Sessions (n = ', num2str(N_C_short), ')']
%     };
% legend([b1, b2], legend_entries, ...
%        'Location', 'bestoutside', 'Interpreter', 'latex', ...
%        'FontName', 'Times New Roman', 'FontSize', 10, 'Box', 'off');
% Generate session date information
session_date_info = ['Sessions (n = ', num2str(N_C_short), ')',newline ...
                     'Sessions from ', datestr(firstSessionDate, 'mm/dd/yyyy'),newline ...
                     ' to ', datestr(lastSessionDate, 'mm/dd/yyyy')];

% Add session date information as a text box
annotation('textbox', [0.77, 0.28, 0.3, 0.55], 'String', session_date_info, ...
    'EdgeColor', 'none', 'Interpreter', 'latex', 'FontSize', 10, 'FontName', 'Times New Roman');
% title_text(1) = {'\hspace{2cm}E2WT Mean Of All SD And Control Sessions CR$^{+}$ Fractions'};
% title_text(2) = {'\hspace{2cm}  CR$^{+}$ above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{LED}},T_{\rm{AirPuff}})$'};
% title_text(3) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $',CR_threshold)};
% title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',14)
% set(gca, 'TickDir', 'out');
% 
% 
% % Customize plot
% set(gca, 'XTick', 1:4, 'XTickLabel', {'Control Short', 'Control Long', 'SD Short', 'SD Long'});
% ylabel('CR+ Fraction', 'Interpreter', 'latex');
% ylim([0 1]);
% % legend({'Control Short', 'Control Long', 'SD Short', 'SD Long'}, 'Location', 'bestoutside');
% set(gca, 'FontSize', 14);
% set(gcf, 'Position', [100, 100, 1000, 600]);
% hold off;

% Save plot
exportgraphics(gcf, 'Mean_CR_plus_fractions_SD_Control_Sessios.pdf', 'ContentType', 'vector');
