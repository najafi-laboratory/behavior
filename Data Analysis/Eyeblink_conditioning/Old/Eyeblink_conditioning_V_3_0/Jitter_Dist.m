

clc; close all; clear;

% Load data
% data_files = dir('E1VT_EBC_V_3_6_20240725_112350.mat');
% data_files = dir('C:\behavior\session_data\E1VT\E1VT_EBC_V_3_7_20240731_135516.mat');
data_files = dir('C:\behavior\session_data\E2WT\E2WT_EBC_V_3_7_20240731_124440.mat');
% data_files = dir('C:\behavior\session_data\E3VT\E3VT_EBC_V_3_6_20240726_131038.mat');
load(data_files.name);

for ctr_file=1:length(data_files)

Jitter = [];

% Extract x values and FECTimes for each trial
numTrials = SessionData.nTrials;
for trialIdx = 1:numTrials 
    % Get Session Data
    LED_Onset = SessionData.RawEvents.Trial{1, trialIdx}.Events.GlobalTimer1_Start;
    FECTimes = SessionData.RawEvents.Trial{1, trialIdx}.Data.FECTimes;
    
    % Shift FECTimes according to protocol version
    if contains(data_files(ctr_file).name, 'V_2_9') || ...
       contains(data_files(ctr_file).name, 'V_3_0')
        % Camera trigger starts at ITI_Pre(1) for V_3_1 and earlier
        FEC_led_shifted = FECTimes + ITI_Pre - LED_Onset;
    else
        % Camera trigger starts at CheckEyeOpen(1) for V_3_2 to V_3_7, and is aligned to state times        
        % Camera trigger starts at Start(1) for V_3_8, and is aligned to state times
        FEC_led_shifted = FECTimes - LED_Onset;
    end
        
    % find image closest to LED Onset
    % FECTimes - LEDOnset has image closest to LED Onset at minimum of
    % absolute value
    abs_FEC_led_shifted = abs(FEC_led_shifted); % absolute value of shifted FECTimes
    closest_frame_idx_to_LED_Onset = find(abs_FEC_led_shifted == min(abs_FEC_led_shifted)); % get idx of image closest to LED Onset
                
    Jitter = [Jitter FEC_led_shifted(closest_frame_idx_to_LED_Onset)];
end

disp(['mean(Jitter): ', num2str(mean(Jitter))]);
disp(['std(Jitter): ', num2str(std(Jitter))]);
disp(['var(Jitter): ', num2str(var(Jitter))]);
disp(['max(Jitter): ', num2str(max(Jitter))]);
disp(['min(Jitter): ', num2str(min(Jitter))]);

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
