clc; close all; clear

data_files = dir('*_EBC_*.mat');
CR_threshold = 0.05;

CR_plus_fractions_long = []; % Initialize array to store CR+ fractions for long blocks
CR_plus_fractions_short = []; % Initialize array to store CR+ fractions for short blocks
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
   

    % Calculate CR+ fractions for long and short blocks
    if numLongTrials > 0
        CR_plus_fraction_long = numCurves_long / numLongTrials;
    else
        CR_plus_fraction_long = NaN; % No long trials
    end
    
    if numShortTrials > 0
        CR_plus_fraction_short = numCurves_short / numShortTrials;
    else
        CR_plus_fraction_short = NaN; % No short trials
    end

    % Store the CR+ fractions and session date
    CR_plus_fractions_long = [CR_plus_fractions_long, CR_plus_fraction_long];
    CR_plus_fractions_short = [CR_plus_fractions_short, CR_plus_fraction_short];
    
    if isfield(SessionData.Info, 'SessionDate')
        session_dates{i} = SessionData.Info.SessionDate;
    else
        session_dates{i} = ['Session ' num2str(i)];
    end
end

% Plotting the bar plot for CR+ fractions (long and short blocks)
figure;
hold on;

% % Create bars for long and short blocks
bar([1:length(CR_plus_fractions_short)]-0.15, CR_plus_fractions_short, 0.3, 'FaceColor', [0.2 0.6 0.8]); % Blue for short blocks (shifted to the left)
bar([1:length(CR_plus_fractions_long)]+0.15, CR_plus_fractions_long, 0.3, 'FaceColor', [0.2 0.8 0.2]); % Green for long blocks (shifted to the right)


set(gca, 'XTick', 1:length(session_dates), 'XTickLabel', session_dates);

% Increase figure size for better readability
set(gcf, 'Position', [100, 100, 1400, 600]); % Adjust as needed

% Adjust font size if needed
set(gca, 'FontSize', 8);
xlabel('Session Date','Interpreter', 'latex');
ylabel('CR+ Fraction','Interpreter', 'latex');
ylim([0 1]);
% legend('Long Blocks', 'Short Blocks');
% Add legend
legend({'Short Block Avg','Long Block Avg'}, 'Interpreter', 'latex', 'fontsize', 14,'Box', 'off', 'Location', 'bestoutside');

title_text(1) = {'\hspace{2cm} All Sessions For E5LG'};
title_text(2) = {'\hspace{2cm}  CR$^{+}$ trials above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{AirPuff- 0.05 (s)}},T_{\rm{AirPuff}})$'};
title_text(3) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $',CR_threshold)};
title_text(4) = {' '};

title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',14)

% title(title_text ,'Fraction of CR+ Trials Across Sessions (Long and Short Blocks)','Interpreter', 'latex');
set(gca, 'FontSize', 14);
set(gca, 'Box', 'off');
% Set tick marks to be outside
set(gca, 'TickDir', 'out');
xtickangle(45); % Rotate x-axis labels for better readability

hold off;

% Optionally, save the plot as a PDF
exportgraphics(gcf, 'CR_plus_fractions_barplot_long_short.pdf', 'ContentType', 'vector');