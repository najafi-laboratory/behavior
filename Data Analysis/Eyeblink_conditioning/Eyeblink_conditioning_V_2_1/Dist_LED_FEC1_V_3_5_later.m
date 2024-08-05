

clc; close all; clear;

% Load data
% data_files = dir('E1VT_EBC_V_3_6_20240725_112350.mat');
data_files = dir('C:\behavior\session_data\E1VT\E1VT_EBC_V_3_6_20240725_112350.mat');
load(data_files.name);

for ctr_file=1:length(data_files)
% Initialize FECTimes_all array to store all FECTimes
FECTimes_all = [];

% Extract x values and FECTimes for each trial
numTrials = SessionData.nTrials;
for trialIdx = 1:numTrials 
    LED_Onset = SessionData.RawEvents.Trial{1, trialIdx}.Events.GlobalTimer1_Start;
    FECTimes = SessionData.RawEvents.Trial{1, trialIdx}.Data.FECTimes;

    if contains(data_files(ctr_file).name, 'V_2_9') || ...
       contains(data_files(ctr_file).name, 'V_3_0')
        FEC_led_aligned = FECTimes + ITI_Pre - LED_Onset;
    else
        FEC_led_aligned = FECTimes - LED_Onset;
    end
    % FEC_led_aligned = FECTimes - LED_Onset;

        fps = 250; % frames per second, frequency of images
        seconds_before = 1; % time of video before led onset
        seconds_after = 3; % time of video after led onset
        
        Frames_before = fps * seconds_before;
        Frames_after = fps * seconds_after;
        
        abs_FEC_led_aligned = abs(FEC_led_aligned);
        closest_frame_idx_to_LED_Onset = find(abs_FEC_led_aligned == min(abs_FEC_led_aligned));
        
        start_idx = closest_frame_idx_to_LED_Onset - Frames_before;
        stop_idx = closest_frame_idx_to_LED_Onset + Frames_after;
        

        len_FEC_led_aligned = length(FEC_led_aligned);
        disp(['Length of FEC_led_aligned: ', num2str(len_FEC_led_aligned)]);
        disp(['Time of FEC_led_aligned: ', num2str(len_FEC_led_aligned * 1/fps)]);
        start_idx = max(1, min(start_idx, len_FEC_led_aligned));
        stop_idx = max(1, min(stop_idx, len_FEC_led_aligned));
        % start_idx = max(1, start_idx); % index to array must be >= 1
        % stop_idx = max(1, stop_idx); % index to array must be >= 1
        
  FEC_led_aligned_trimmed = FEC_led_aligned(start_idx : stop_idx);
  % FECTimes_all = [FECTimes_all; LED_Onset - FECTimes(1)]; % for distribution of ledsonet - fectimes(1), fectimes as-is from session data
  % FECTimes_all = [FECTimes_all; 0 - FEC_led_aligned(1)]; % for distribution of ledsonet - fectimes(1), aligned to led fectimes
  FECTimes_all = [FECTimes_all; 0 - FEC_led_aligned_trimmed(1)]; % for distribution of ledsonet - fectimes(1), aligned and trimmed fectimes
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
xlabel('FECTimes(1) relative to LED-Onset');
ylabel('Frequency');
title('Distribution of FECTimes values across all trials');
grid on;

% Display bin counts and edges
disp('Bin counts:');
disp(binCounts);

disp('Bin edges:');
disp(binEdges);
end
