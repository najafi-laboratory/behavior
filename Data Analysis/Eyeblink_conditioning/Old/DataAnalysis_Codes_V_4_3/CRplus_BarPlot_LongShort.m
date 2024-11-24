clc; close all; clear

data_files = dir('*_EBC_*.mat');
% data_files = dir('E1VT_EBC_V_3_8_20240806_130856.mat');
CR_threshold = 0.05;

CR_plus_fractions = []; % Initialize array to store CR+ fractions
session_dates = {}; % Initialize cell array to store session dates

for i = 1:length(data_files)
    
    % Reset numCurves for each session
    numCurves = 0;
    load(data_files(i).name)

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
    
    for ctr_trial = 1:step:numTrials

        % % Extract relevant events and states
        % if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'LED_Onset') && ...
        %    isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'AirPuff')

            LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Onset(1);
            AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.States.AirPuff(1);
            FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;

            FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels / overallMax;
            FEC_led_aligned = FECTimes - LED_Onset;
            t_LED = LED_Onset - LED_Onset;
            t_puff = AirPuff_Start - LED_Onset;
            t1 = t_LED - 0.01;
            t2 = t_LED;

            % Check if the trial is a CR+ trial
            if CR_plus_eval(FEC_led_aligned, FEC_norm, t1, t2, t_LED, t_puff, CR_threshold)
                numCurves = numCurves + 1;
            end
        % end
    end

    % Calculate CR+ fraction for this session
    CR_plus_fraction = numCurves / numTrials;

    % Store the CR+ fraction and session date
    CR_plus_fractions = [CR_plus_fractions, CR_plus_fraction];
    session_dates{i} = SessionData.Info.SessionDate;
end

% Plotting the bar plot for CR+ fractions
figure;

bar(CR_plus_fractions, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTickLabel', session_dates);
xlabel('Session Date');
ylabel('CR+ Fraction');
ylim([0 1]);
title('Fraction of CR+ Trials Across Sessions');
set(gca, 'FontSize', 14);
set(gca, 'Box', 'off');
% Set tick marks to be outside
set(gca, 'TickDir', 'out');
xtickangle(45); % Rotate x-axis labels for better readability


% Optionally, save the plot as a PDF
exportgraphics(gcf, 'CR_plus_fractions_barplot.pdf', 'ContentType', 'vector');
