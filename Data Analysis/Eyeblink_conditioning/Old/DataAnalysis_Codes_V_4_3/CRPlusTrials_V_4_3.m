clc;close all;clear



 data_files = dir('*_EBC_*.mat');
 % data_files = dir('E1VT_EBC_V_3_9_20240901_182615.mat');
 % 

CR_threshold = 0.05;

for i = 1:length(data_files)
 
    load(data_files(i).name)


numTrials = length(SessionData.RawEvents.Trial);
% Create a colormap from light blue to dark blue
colors = [linspace(0.6, 0, numTrials)', linspace(0.6 , 0,numTrials)', linspace(1, 0.5, numTrials)'];



figure('units','centimeters','position',[2 2 24 26])
numCurves = 0;
totalFEC_norm = [];
leg_str = cell(1,1);
% Initialize an empty array to store all eyeAreaPixels values
allEyeAreaPixels = [];

% Loop through each trial to collect eyeAreaPixels data
for trialIdx = 1:numTrials    
   eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
   allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data 
end
% Find the overall maximum value across all collected eyeAreaPixels



overallMax = max(allEyeAreaPixels);

    % Initialize matrices for short and long blocks
    FEC_short_matrix = [];
    FEC_long_matrix = [];

    % Variables to track min and max of the curves
    minNumCurve = inf;
    maxNumCurve = -inf;

    % Create a color gradient from dim to dark green for short blocks
    colorMapLong = [linspace(0.7, 0, numTrials)', linspace(1, 0.5, numTrials)', linspace(0.7, 0, numTrials)']; % Dim to dark green
    
    % Create a color gradient from dim to dark blue for long blocks
    colorMapShort = [linspace(0.7, 0, numTrials)', linspace(0.7, 0, numTrials)', linspace(1, 0.5, numTrials)']; % Dim to dark blue



step = 1;
for ctr_trial = 1:step:numTrials


        %  if CheckEyeOpenTimeout is not nan, timeout occurred, continue to
        %  next trial
        % 
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

        % CheckEyeOpen = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(2);
        Start = SessionData.RawEvents.Trial{1, ctr_trial}.States.Start(1);
        ITI_Pre = SessionData.RawEvents.Trial{1, ctr_trial}.States.ITI_Pre(1);
        % LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Onset(1);
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        LED_Puff_ISI_start = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
        AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
        AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;
        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;   


        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        LED_Onset_Zero_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End - LED_Onset;
        AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
        AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;


        if contains(data_files(i).name, 'V_2_9') || ...
           contains(data_files(i).name, 'V_3_0')
            FEC_led_aligned = FECTimes + ITI_Pre - LED_Onset;
            t_LED = LED_Onset_Zero_Start+ ITI_Pre;
            t_puff = t_LED+ 0.2;
            t1 = t_LED-0.1;
            t2 = t_LED;
        else
            FEC_led_aligned = FECTimes - LED_Onset;
            t_LED = LED_Onset_Zero_Start;
            t_puff = AirPuff_LED_Onset_Aligned_End;
            t1 = t_LED-0.01;
            t2 = t_LED;
        end

        FEC_led_aligned = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes - LED_Puff_ISI_start;
        FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels /overallMax;

        fps = 250; % frames per second, frequency of images
        seconds_before = 0.5;
        seconds_after = 2;

        Frames_before = fps * seconds_before;
        Frames_after = fps * seconds_after;
       % Determine a common time vector for interpolation
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

   

% Define the common time vector (e.g., the union of all unique time points or a regular grid)
commonTime = linspace(min(FEC_led_aligned_trimmed), max(FEC_led_aligned_trimmed), 100);  % Adjust 100 to the desired number of points

if(CR_plus_eval(FEC_led_aligned_trimmed,FEC_trimmed,t1,t2,t_LED,t_puff,CR_threshold))
numCurves = numCurves+1;
% Interpolate FEC_norm to the common time vector
FEC_norm_interp = interp1(FEC_led_aligned_trimmed, FEC_trimmed, commonTime);
        
% Accumulate the data
totalFEC_norm = [totalFEC_norm; FEC_norm_interp];
        FEC_interpolated = interp1(FEC_led_aligned_trimmed, FEC_trimmed, common_time_vector, 'linear', 'extrap');
        FEC_norm_matrix(ctr_trial, :) = FEC_interpolated;

                LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - LED_Onset;
        % Identify if the trial is short or long
        if SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) > 0.3
            trialType = 'long';
            verticalLineColor = 'g'; % Green for long trials
            % Long trial (use green gradient)
             plotColor = colorMapLong(ctr_trial, :);
             h(numCurves) = plot(common_time_vector, FEC_interpolated, 'Color', plotColor, 'LineWidth', 0.90); hold on

             FEC_long_matrix = [FEC_long_matrix; FEC_interpolated]; % Append to long block matrix
            leg_str{2} = sprintf('Long Blocks', numTrials);
            legend_handles(2) = h(numCurves);

            
        else
            trialType = 'short';
            verticalLineColor = 'b'; % Blue for short trials
             % Short trial (use blue gradient)
            plotColor = colorMapShort(ctr_trial, :);
            h(numCurves) = plot(common_time_vector, FEC_interpolated, 'Color', plotColor, 'LineWidth', 0.90); hold on

            FEC_short_matrix = [FEC_short_matrix; FEC_interpolated]; % Append to short block matrix
            leg_str{1} = sprintf('Short Blocks', ctr_trial);
            legend_handles(1) = h(numCurves);
           
            
        end
            
        
        % Add vertical line at the end of the LED_Puff_ISI
         line([LED_Puff_ISI_end, LED_Puff_ISI_end], [minNumCurve, maxNumCurve], 'Color', verticalLineColor, 'LineWidth', 2);

        % Update min and max for y-axis
        minNumCurve = min(minNumCurve, min(FEC_interpolated));
        maxNumCurve = max(maxNumCurve, max(FEC_interpolated));

end
        
end

    % Plot the vertical line at x = 0 (LED onset)
    line([0, 0], [minNumCurve, maxNumCurve], 'Color', 'k', 'LineWidth', 1, 'LineStyle', '--');
% Calculate the average FEC curve
if numCurves > 0

    % Calculate the average normalized FEC curve
    avgFEC_norm = mean(totalFEC_norm, 1);
        % Calculate and plot averages for short and long blocks
    if ~isempty(FEC_short_matrix)
        FEC_short_avg = mean(FEC_short_matrix, 1);
        h_avg_short = plot(common_time_vector, FEC_short_avg, 'Color', 'k', 'LineStyle', '-', 'LineWidth', 3.5); % black for short blocks
        % leg_str{1} = 'Short Block Avg';
        % legend_handles(1) = h_avg_short;
    end

    if ~isempty(FEC_long_matrix)
        FEC_long_avg = mean(FEC_long_matrix, 1);
        h_avg_long = plot(common_time_vector, FEC_long_avg, 'Color', [0.3, 0.3, 0.3] , 'LineStyle', '-', 'LineWidth', 3.5); % grey for long blocks
        % leg_str{2} = 'Long Block Avg';
        % legend_handles(2) = h_avg_long;
    end
    % Plot the average curve
    % h_avg = plot(commonTime, avgFEC_norm, 'k', 'LineWidth', 2); % Black line for the average curve
    % leg_str{3} = 'Average';
    % legend_handles(3)= h_avg;
    % 
    title('Average Superimposed CR+ Curve');
    xlabel('Time (s)');
    ylabel('FEC (Normalized)');
else
    disp('No CR+ trials found.');
end

load(data_files(i).name)
CR_plus_fraction(i) =  numCurves/numTrials;
CR_plus_fraction_x{i} = SessionData.Info.SessionDate;
% CR_plus_trials{i} = bar_str;



% for ctr_trial = 1:numTrials
% 
% % Shade the area (ITI)
% x_fill = [LED_Onset_Zero_Start, LED_Onset_Zero_End,LED_Onset_Zero_End,LED_Onset_Zero_Start];         % x values for the filled area
% y_fill = [minNumCurve minNumCurve maxNumCurve maxNumCurve];     % y values for the filled area (y=0 at the x-axis)
% fill(x_fill, y_fill, 'green', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
% end
% 
% for ctr_trial = 1:numTrials
% % Shade the area (AirPuff Duration)
% x_fill = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End,AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];         % x values for the filled area
% y_fill = [minNumCurve minNumCurve maxNumCurve maxNumCurve];     % y values for the filled area (y=0 at the x-axis)
% fill(x_fill, y_fill, 'm', 'FaceAlpha', 0.35, 'EdgeColor', 'none');
% end





set(gca, 'Box', 'off');
% Set tick marks to be outside
set(gca, 'TickDir', 'out');
xlim([-0.2 0.6]);
ylim([minNumCurve maxNumCurve]);
set(gca,'FontSize',14)
set(gca,'Position',[0.1 0.1 0.8 0.7])
ylabel_text(1) = {'Eyelid closure (norm)'};
ylabel(ylabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
xlabel_text(1) = {'Time From Trial start (s)'};
xlabel(xlabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
title_text(1) = {'CR$^{+}$ trials above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{LED}},T_{\rm{AirPuff}})$'};
title_text(2) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $',CR_threshold)};
title_text(3) = {' '}; title_text(4) = {' '};
title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',19)
h_legend = legend(legend_handles ,leg_str,'Interpreter','latex','fontsize',14,'location','bestoutside','Box','off');
% h_legend.NumColumns = 1;
% h_legend_pos = h_legend.Position;
% h_legend.Position = [0.98*h_legend_pos(1) 0.99*h_legend_pos(2) h_legend_pos(3) h_legend_pos(4)];
clear xlabel_text ylabel_text title_text leg_str


text(0, maxNumCurve, 'LED', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 13, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0);
text(0.2, maxNumCurve, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 13, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0);
text(0.4, maxNumCurve, 'AirPuff', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'FontSize', 13, 'Color', 'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Rotation', 0);
   

% % adding text annotations
% text_cell{1} = 'LED';
% t1 = text(0, 1, text_cell,'interpreter', 'latex','fontname','Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left','VerticalAlignment','middle');
% t1.Rotation = 90;
% text_cell{1} = 'AirPuff';
% t2 = text(0.3, 1, text_cell,'interpreter', 'latex','fontname','Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left','VerticalAlignment','middle');
% t2.Rotation = 90;


load(data_files(i).name);
% Extract parts of the filename
[~, name, ~] = fileparts(data_files(i).name); % Extract the name without extension

% Split the filename to get the required parts
nameParts = split(name, '_');
if length(nameParts) >= 5
    prefixPart = nameParts{1}; % First string before the first underscore
    datePart = nameParts{6}; % Date par    
    % Construct the new filename
    
    newFilename_1 = sprintf('%s_FEC_CRplus_trials_%s.pdf', prefixPart, datePart);
  
    % Export the graphics
    exportgraphics(gcf, newFilename_1, 'ContentType', 'vector');
else
    error('Filename does not have the expected format');
end



end