% 
% 
% clc; close all; clear;
% 
% % Load data
% data_files = dir('E1VT_EBC_V_3_7_20240730_132615.mat');
% load(data_files.name);
% 
% % Initialize x values
% numTrials = length(SessionData.RawEvents.Trial);
% x = zeros(1, numTrials);
% 
% % Extract x values
% for trialIdx = 1:numTrials 
%     x(trialIdx) = SessionData.RawEvents.Trial{1, trialIdx}.States.Start(2)
%     LED_Onset = SessionData.RawEvents.Trial{1, trialIdx}.States.LED_Onset(1);
%     FECTimes = SessionData.RawEvents.Trial{1, trialIdx}.Data.FECTimes - x(trialIdx);
% end
% 
% % Plot the distribution of x values
% figure;
% histogram(FECTimes);
% xlabel('FECTimes relative to LED-Onset');
% ylabel('Frequency');
% title('Distribution of FECTimes values');
% grid on;


% clc; close all; clear;
% 
% % Load data
% data_files = dir('E1VT_EBC_V_3_7_20240730_132615.mat');
% load(data_files.name);
% 
% % Initialize variables
% numTrials = length(SessionData.RawEvents.Trial);
% allFECTimes = [];
% 
% % Extract FECTimes relative to LED onset
% for trialIdx = 1:numTrials 
%     x(trialIdx) = SessionData.RawEvents.Trial{1, trialIdx}.States.Start(2)
%     LED_Onset = SessionData.RawEvents.Trial{1, trialIdx}.States.LED_Onset(1);
%     FECTimes = SessionData.RawEvents.Trial{1, trialIdx}.Data.FECTimes - x(trialIdx);
%     allFECTimes = [allFECTimes, FECTimes]; % Concatenate all FECTimes
% end
% 
% % Plot histogram with small bins for better time resolution
% binWidth = 0.01; % Define bin width for better time resolution
% edges = min(allFECTimes):binWidth:max(allFECTimes); % Define bin edges
% histogram(allFECTimes, edges);
% xlabel('Time relative to LED onset (s)');
% ylabel('Frequency');
% title('Distribution of FECTimes relative to LED onset');
% grid on;


clc; close all; clear;

% Load data
data_files = dir('C:\behavior\session_data\E1VT\E1VT_EBC_V_3_7_20240730_132615.mat');
load(data_files.name);

% Initialize FECTimes_all array to store all FECTimes
FECTimes_all = [];

% Extract x values and FECTimes for each trial
numTrials = length(SessionData.RawEvents.Trial);
% for trialIdx = 1:numTrials 
%     x = SessionData.RawEvents.Trial{1, trialIdx}.States.Start(2);
%     LED_Onset = SessionData.RawEvents.Trial{1, trialIdx}.States.LED_Onset(1);
%     FECTimes = SessionData.RawEvents.Trial{1, trialIdx}.Data.FECTimes - x;
%     FECTimes_all = [FECTimes_all; FECTimes];
% end

for trialIdx = 1:numTrials 
    % x = SessionData.RawEvents.Trial{1, trialIdx}.States.Start(2);
    LED_Onset = SessionData.RawEvents.Trial{1, trialIdx}.Events.GlobalTimer2_Start;
    FECTimes = LED_Onset - SessionData.RawEvents.Trial{1, trialIdx}.Data.FECTimes(1);
    FECTimes_all = [FECTimes_all; FECTimes];
end


% Calculate histogram bin counts and edges
[binCounts, binEdges] = histcounts(FECTimes_all);

% Plot the distribution of FECTimes
figure;
histogram(FECTimes_all);
xlabel('FECTimes relative to LED-Onset');
ylabel('Frequency');
title('Distribution of FECTimes values across all trials');
grid on;

% Display bin counts and edges
disp('Bin counts:');
disp(binCounts);

disp('Bin edges:');
disp(binEdges);




