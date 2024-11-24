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
    
    for ctr_trial = 1:step:numTrials

        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end

        % get CheckEyeOpen if it is in session data for versions V_3_2+
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpen')
            CheckEyeOpenStart = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(1);
            CheckEyeOpenStop = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(2);
        end

        % Check if the necessary states are present in the trial
        % if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'LED_Onset') && ...
        %    isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'AirPuff')

            LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Onset(1);
            LED_Onset_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End;
       
            AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.States.AirPuff(1);
            AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.States.AirPuff(2);
    
            FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;
            LED_Onset_Zero_Start = LED_Onset - LED_Onset;
            LED_Onset_Zero_End = LED_Onset_End - LED_Onset;
            AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
            AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;

            FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels / overallMax;
            FEC_led_aligned = FECTimes - LED_Onset;


isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
       
if isLongBlock > 0.3

    t_LED = 0.2 ;
    t_puff = AirPuff_LED_Onset_Aligned_End;
    t1 = t_LED - 0.01;
    t2 = t_LED; 

else

    t_LED = LED_Onset_Zero_Start;
    t_puff = AirPuff_LED_Onset_Aligned_End;
    t1 = t_LED - 0.01;
    t2 = t_LED; 
end 


            % Check if the trial is a CR+ trial
            is_CR_plus = CR_plus_eval(FEC_led_aligned, FEC_norm, t1, t2, t_LED, t_puff, CR_threshold);
            PuffDelay = AirPuff_LED_Onset_Aligned_Start - LED_Onset_Zero_Start;
            % Check if it's a long or short block trial
            % if isfield(SessionData.TrialSettings, 'PuffDelay') % Ensure PuffDelay exists
                if abs(PuffDelay - 0.4) < 0.01  % Long block
                    numLongTrials = numLongTrials + 1;

                    if is_CR_plus
                        numCurves_long = numCurves_long + 1;
                    end
                elseif abs(PuffDelay - 0.2) < 0.01 % Short block
                    numShortTrials = numShortTrials + 1;


                    if is_CR_plus
                        numCurves_short = numCurves_short + 1;
                    end
                end
            % else
            %     warning(['PuffDelay not found in trial ' num2str(ctr_trial) ' of ' data_files(i).name]);
            % end
        end
    % end

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
xlabel('Session Date','Interpreter', 'latex');
ylabel('CR+ Fraction','Interpreter', 'latex');
ylim([0 1]);
% legend('Long Blocks', 'Short Blocks');
% Add legend
legend({'Short Block Avg','Long Block Avg'}, 'Interpreter', 'latex', 'fontsize', 14,'Box', 'off', 'Location', 'bestoutside');

title_text(1) = {'\hspace{2cm}  CR$^{+}$ short-block trials above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{LED}},T_{\rm{AirPuff}})$'};
title_text(2) = {'\hspace{2cm}  CR$^{+}$ long-block trials above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{0.2(s)}},T_{\rm{AirPuff}})$'};
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