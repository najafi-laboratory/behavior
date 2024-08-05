

clc; close all; clear;

% Load data
% data_files = dir('E1VT_EBC_V_3_6_20240725_112350.mat');
data_files = dir('C:\behavior\session_data\E1VT\E1VT_EBC_V_3_6_20240725_112350.mat');
load(data_files.name);

for ctr_file=1:length(data_files)

Jitter = [];

% Extract x values and FECTimes for each trial
numTrials = SessionData.nTrials;
for trialIdx = 1:numTrials 
    LED_Onset = SessionData.RawEvents.Trial{1, trialIdx}.Events.GlobalTimer1_Start;
    FECTimes = SessionData.RawEvents.Trial{1, trialIdx}.Data.FECTimes;
    ITI_Pre = SessionData.RawEvents.Trial{1, trialIdx}.States.ITI_Pre(1);

    if contains(data_files(ctr_file).name, 'V_2_9') || ...
       contains(data_files(ctr_file).name, 'V_3_0')
        FEC_led_aligned = FECTimes + ITI_Pre - LED_Onset;
    else
        FEC_led_aligned = FECTimes - LED_Onset;
    end
        
    abs_FEC_led_aligned = abs(FEC_led_aligned);
    closest_frame_idx_to_LED_Onset = find(abs_FEC_led_aligned == min(abs_FEC_led_aligned));
    % disp(['closest_frame_idx_to_LED_Onset: ', num2str(closest_frame_idx_to_LED_Onset)]);
                
    Jitter = [Jitter FEC_led_aligned(closest_frame_idx_to_LED_Onset)];
end

disp(['mean(Jitter): ', num2str(mean(Jitter))]);
disp(['std(Jitter): ', num2str(std(Jitter))]);
disp(['var(Jitter): ', num2str(var(Jitter))]);
disp(['max(Jitter): ', num2str(max(Jitter))]);

% % Plot histogram with small bins for better time resolution
binWidth = 0.0001; % Define bin width for better time resolution
edges = min(Jitter):binWidth:max(Jitter); % Define bin edges

% Calculate histogram bin counts and edges
[binCounts, binEdges] = histcounts(Jitter);

% Plot the distribution of FECTimes
figure;
histogram(Jitter);
xlabel('Time of image closest to LED-Onset relative to LED-Onset');
ylabel('Frequency');
title('Distribution of closest FECTimes image relative to LED-Onset across all trials');
grid on;

% Display bin counts and edges
% disp('Bin counts:');
% disp(binCounts);
% 
% disp('Bin edges:');
% disp(binEdges);
end
