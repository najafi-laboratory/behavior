clc; close all; clear

% Load all session data files
% data_files = dir('E6LG_EBC_V_3_19_20250529_210350.mat');
data_files = dir('*_EBC_*.mat');
CR_threshold = 0.02;

good_CR_threshold = 0.05;
poor_CR_threshold = 0.02;


% Initialize legend handles for short and long blocks
legend_handles_short = [];  % For blue plots
legend_handles_long = [];   % For red plots

% CR_threshold = 0.02;
% Initialize legend handles for short and long blocks
legend_handles_short = [];  % For blue plots
legend_handles_long = [];   % For red plots
% Clear and initialize legend_entries_short to ensure it's a cell array

sessionDates = [];
legend_entries_short = {};  % Explicitly set it as a cell array
legend_entries_long = {};
x_fill_longLED = [];
x_fill_shortLED = [];
% Increase figure size for better layout control
% figure;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Subplot 1: Short Block
%%
% subplot(2, 2, 1);
% hold on;

figure('Position', [100, 100, 1200, 800]);
% Set the title with custom position

% legend_handles = [];
legend_entries = {};

% Initialize containers for all sessions
all_sessions_short_block_data = [];
all_sessions_long_block_data = [];


% Initialize containers for all sessions
SD_sessions_short_block_data = [];
SD_sessions_long_block_data = [];
C_sessions_short_block_data = [];
C_sessions_long_block_data = [];

    SD_combined_sem = [];
    C_combined_sem = [];


% Initialize containers for probe trial FEC data
probe_short_SD_1 = [];
probe_short_SD_2 = [];
probe_short_C = [];

probe_long_SD_1 = [];
probe_long_SD_2 = [];
probe_long_C = [];

N_SD_1_short = 0; N_SD_2_short = 0; N_C_short = 0;
N_SD_1_long = 0; N_SD_2_long = 0; N_C_long = 0;

% Initialize containers for short block data (all trials across all sessions)
all_trials_short_block_SD_1 = [];
all_trials_short_block_SD_2 = [];
all_trials_short_block_Control = [];

% Loop over each data file
for i = 1:length(data_files)
    
    t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile;
    hold on;

    load(data_files(i).name);
    numTrials = length(SessionData.RawEvents.Trial);
    numCurves = 0;
    totalFEC_norm = [];
    

    % Containers for long and short block data for this session
    long_block_data = [];
    short_block_data = [];

    % Initialize an empty array to store all eyeAreaPixels values
    allEyeAreaPixels = [];    
        % Loop through each trial to collect eyeAreaPixels data
    for trialIdx = 1:numTrials    
       eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
       allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data 
    end
    % Find the overall maximum value across all collected eyeAreaPixels
    overallMax = max(allEyeAreaPixels);




        % Loop over each trial to calculate shifted and normalized FEC
    for ctr_trial = 1:numTrials
        % Skip trial if timeout occurred
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end

        is_probe = SessionData.RawEvents.Trial{1, ctr_trial}.Data.IsProbeTrial;
        if ~is_probe
            continue;
        end

        % Identify block type
        if SessionData.RawEvents.Trial{1, ctr_trial}.Data.ISI > 0.30  
            isLongBlock = true;
        else
            isLongBlock = false;
        end
        
        % LED_Puff_ISI_start = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
        % LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
        
        % AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
        % AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;

        % Extract timing data
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        LED_Puff_ISI_start = LED_Onset;
        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes; 
        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        LED_Onset_Zero_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End - LED_Onset;
        % AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
        % AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;


        FEC_led_aligned = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes - LED_Puff_ISI_start;
        FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels /overallMax;

        fps = 250; % frames per second, frequency of images
        seconds_before = 0.5;
        seconds_after = 2;

        Frames_before = fps * seconds_before;
        Frames_after = fps * seconds_after;
       % Determine a common time vector for interpolation
        % commonTime = linspace(-seconds_before, seconds_after, Frames_before + Frames_after + 1);
        common_time_vector = linspace(-seconds_before, seconds_after, Frames_before + Frames_after + 1);
       % Initialize a matrix to store interpolated FEC data
        FEC_norm_matrix = zeros(numTrials, length(common_time_vector));

        abs_FEC_led_aligned = abs(FEC_led_aligned);
        closest_frame_idx_to_LED_Onset = find(abs_FEC_led_aligned == min(abs_FEC_led_aligned));

        start_idx = closest_frame_idx_to_LED_Onset - Frames_before;
        stop_idx = closest_frame_idx_to_LED_Onset + Frames_after;

        len_FEC_led_aligned = length(FEC_led_aligned);
        start_idx = max(1, min(start_idx, len_FEC_led_aligned));
        stop_idx = max(1, min(stop_idx, len_FEC_led_aligned));
        FEC_led_aligned_trimmed = FEC_led_aligned(start_idx : stop_idx);
        FEC_trimmed = FEC_norm(start_idx : stop_idx);
        FEC_times_trimmed = FEC_led_aligned(start_idx:stop_idx);

        % Apply smoothing to reduce noise
        FEC_led_aligned_trimmed_smooth = smoothdata(FEC_led_aligned_trimmed, 'movmean', 5); % Moving average
        FEC_trimmed_smooth = smoothdata(FEC_trimmed, 'movmean', 5); % Moving average

        % isShortBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - ...
        %                SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) <= 0.3;
        t_LED = LED_Onset_Zero_Start;
        % t_puff = AirPuff_LED_Onset_Aligned_Start;
        
        % isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3;

        % poor_CR_threshold = 0.02;
        % is_CR_plus = CR_plus_eval_dev(FEC_times_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);
        % CR_category = classify_CR_02(FEC_led_aligned_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);
        % CR_category = classify_CR_05(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, t_LED, t_puff , good_CR_threshold, poor_CR_threshold);

        % Define the common time vector (e.g., the union of all unique time points or a regular grid)
        commonTime = linspace(min(FEC_led_aligned_trimmed_smooth), max(FEC_led_aligned_trimmed_smooth), 500);  % Adjust 100 to the desired number of points

        % FEC_norm_interp = interp1(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, commonTime, 'spline');
        % Interpolate FEC data to this global common time base
        FEC_norm_interp = interp1(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, commonTime, 'spline', NaN);
        % Aggregate data for short blocks

       switch SessionData.SleepDeprived
            case 3
                if isLongBlock
                    probe_long_SD_1 = [probe_long_SD_1; FEC_norm_interp];
                    N_SD_1_long = N_SD_1_long + 1;
                else
                    probe_short_SD_1 = [probe_short_SD_1; FEC_norm_interp];
                    N_SD_1_short = N_SD_1_short + 1;
                end
            case 4
                if isLongBlock
                    probe_long_SD_2 = [probe_long_SD_2; FEC_norm_interp];
                    N_SD_2_long = N_SD_2_long + 1;
                else
                    probe_short_SD_2 = [probe_short_SD_2; FEC_norm_interp];
                    N_SD_2_short = N_SD_2_short + 1;
                end
            otherwise
                if isLongBlock
                    probe_long_C = [probe_long_C; FEC_norm_interp];
                    N_C_long = N_C_long + 1;
                else
                    probe_short_C = [probe_short_C; FEC_norm_interp];
                    N_C_short = N_C_short + 1;
                end
        end






    end
end    

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

figure('Name', 'Short Block Probe Trials'); hold on;
plot_probe_data(commonTime, probe_short_SD_1, 'r', 'SD+1 Short', N_SD_1_short ,N_SD_2_short ,N_C_short,firstSessionDate,lastSessionDate);
plot_probe_data(commonTime, probe_short_SD_2, [0.5 0 0.5], 'SD+2 Short', N_SD_1_short ,N_SD_2_short ,N_C_short, firstSessionDate,lastSessionDate);
plot_probe_data(commonTime, probe_short_C, 'k', 'Control Short', N_SD_1_short ,N_SD_2_short ,N_C_short, firstSessionDate,lastSessionDate);
xlabel('Time (s)'); ylabel('FEC (norm)'); title('Short Block Probe Trials'); 

figure('Name', 'Long Block Probe Trials'); hold on;
plot_probe_data(commonTime, probe_long_SD_1, 'r', 'SD+1 Long',N_SD_1_long ,N_SD_2_long ,N_C_long, firstSessionDate,lastSessionDate);
plot_probe_data(commonTime, probe_long_SD_2, [0.5 0 0.5], 'SD+2 Long', N_SD_1_long ,N_SD_2_long ,N_C_long, firstSessionDate,lastSessionDate);
plot_probe_data(commonTime, probe_long_C, 'k', 'Control Long', N_SD_1_long ,N_SD_2_long ,N_C_long, firstSessionDate,lastSessionDate);
xlabel('Time (s)'); ylabel('FEC (norm)'); title('Long Block Probe Trials'); 


% === Prepare figure and layout ===
figure('Name', 'Probe Trials - Short and Long Blocks', 'Color', 'w');
tiledlayout(2,1, 'Padding', 'compact', 'TileSpacing', 'compact');

% === Define LED and AirPuff shading areas ===
LED_Onset_Zero_Start = 0.0;
LED_Onset_Zero_End = 0.1;
AirPuff_Onset_Start = 0.25;
AirPuff_Onset_End = 0.35;


% LED + Puff shading
y_fill = [0 0 1 1];
fill([LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start], y_fill, ...
     [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none');
fill([AirPuff_Onset_Start, AirPuff_Onset_End, AirPuff_Onset_End, AirPuff_Onset_Start], y_fill, ...
     [0.68 0.85 0.9], 'FaceAlpha', 0.4, 'EdgeColor', 'none');

xlim([-0.2 0.6]);
ylim([0 1]);
ylabel('Eyelid closure (norm)', 'Interpreter', 'latex', 'FontSize', 12);
title('Short Block Probe Trials', 'Interpreter', 'latex', 'FontSize', 14);
% legend({'SD+1', 'SD+2', 'Control'}, 'Location', 'northeast', 'Box', 'off', 'Interpreter', 'latex');

% === Subplot 2: Long Block ===
nexttile;



% === Export the figure ===
exportgraphics(gcf, 'ProbeTrials_Avg_SEM.pdf', 'ContentType', 'vector');
disp('Plot saved as ProbeTrials_Avg_SEM.pdf');