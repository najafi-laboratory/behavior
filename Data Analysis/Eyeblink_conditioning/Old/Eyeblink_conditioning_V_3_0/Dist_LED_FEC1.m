clc; close all; clear;

% Load data
% data_files = dir('E1VT_EBC_V_3_6_20240725_112350.mat');
data_files = dir('C:\behavior\session_data\E1VT\E1VT_EBC_V_2_9_20240703_150505.mat');
% data_files = dir('C:\behavior\session_data\E1VT\E1VT_EBC_V_3_7_20240731_135516.mat');

% data_files = dir('C:\behavior\session_data\E2WT\E2WT_EBC_V_3_7_20240731_124440.mat');
% data_files = dir('C:\behavior\session_data\E3VT\E3VT_EBC_V_3_6_20240726_131038.mat');
load(data_files.name);

for ctr_file=1:length(data_files)
% Initialize FECTimes_all array to store all FECTimes
FECTimes_all = [];

% Extract Session Data for each trial
numTrials = SessionData.nTrials;
for trialIdx = 1:numTrials
    % Get Session Data
    LED_Onset = SessionData.RawEvents.Trial{1, trialIdx}.Events.GlobalTimer1_Start;
    FECTimes = SessionData.RawEvents.Trial{1, trialIdx}.Data.FECTimes;
    ITI_Pre = SessionData.RawEvents.Trial{1, trialIdx}.States.ITI_Pre(1);
    
    % get CheckEyeOpen if it is in session data for versions V_3_2+
    if isfield(SessionData.RawEvents.Trial{1, trialIdx}.States, 'CheckEyeOpen')
        CheckEyeOpen = SessionData.RawEvents.Trial{1, trialIdx}.States.CheckEyeOpen(1);
    end

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
    
    % align FECTimes to number of seconds before and after LED Onset
    % calculate fps using diff of FECTimes
    % FECTimes_diff = diff(FECTimes);
    % T_mean = mean(FECTimes_diff);
    % fps_mean = 1/T_mean;
    
    % use fixed fps
    fps = 250; % frames per second, frequency of images
    
    seconds_before = 0.5; % time of video before led onset
    seconds_after = 3; % time of video after led onset    
    Frames_before = fps * seconds_before; % number of frames before LED Onset
    Frames_after = fps * seconds_after; % number of frames after LED Onset
    start_idx = closest_frame_idx_to_LED_Onset - Frames_before; % get index of image before LED Onset
    stop_idx = closest_frame_idx_to_LED_Onset + Frames_after; % get index of image after LED Onset
        
    % if needed for trace length less than calculated segment before/after
    % len_FEC_led_shifted = length(FEC_led_shifted);
    % disp(['Length of FEC_led_shifted: ', num2str(len_FEC_led_shifted)]);
    % disp(['Time of FEC_led_shifted: ', num2str(len_FEC_led_shifted * 1/fps)]);
    % start_idx = max(1, min(start_idx, len_FEC_led_shifted));
    % stop_idx = max(1, min(stop_idx, len_FEC_led_shifted));    
        
    % start_idx = max(1, start_idx); % index to array must be >= 1
    % stop_idx = max(1, stop_idx); % index to array must be >= 1
        
    FEC_led_aligned = FEC_led_shifted(start_idx : stop_idx); % FECTimes aligned to seconds before/after LED Onset
    % FECTimes_all = [FECTimes_all; LED_Onset - FECTimes(1)]; % for distribution of ledsonet - fectimes(1), fectimes as-is from session data    
    FECTimes_all = [FECTimes_all; 0 - FEC_led_aligned(1)]; % for distribution of ledsonet - fectimes(1), fectimes aligned to seconds before/after LED Onset
end

disp(['mean(FECTimes_all): ', num2str(mean(FECTimes_all))]);
disp(['std(FECTimes_all): ', num2str(std(FECTimes_all))]);
disp(['var(FECTimes_all): ', num2str(var(FECTimes_all))]);
disp(['max(FECTimes_all): ', num2str(max(FECTimes_all))]);
disp(['min(FECTimes_all): ', num2str(min(FECTimes_all))]);

% % Plot histogram with small bins for better time resolution
binWidth = 0.001; % Define bin width for better time resolution
edges = min(FECTimes_all):binWidth:max(FECTimes_all); % Define bin edges

% Calculate histogram bin counts and edges
[binCounts, binEdges] = histcounts(FECTimes_all);

% Plot the distribution of FECTimes
figure;
histogram(FECTimes_all);
xlabel('Aligned FECTimes(1) relative to LED-Onset');
ylabel('Frequency');
title('Distribution of aligned FECTimes values across all trials');
grid on;

% Display bin counts and edges
disp('Bin counts:');
disp(binCounts);

disp('Bin edges:');
disp(binEdges);
end
