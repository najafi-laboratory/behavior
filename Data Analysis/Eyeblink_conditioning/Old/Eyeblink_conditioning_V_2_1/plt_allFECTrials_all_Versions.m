 clc;close all;clear
clear
data_files = dir('C:\behavior\session_data\E3VT\*_EBC_*.mat');
% data_files = dir('C:\behavior\session_data\E1VT\E1VT_EBC_V_3_0_20240713_143538.mat');

for i = 1:length(data_files)
 
    load(data_files(i).name)
    numTrials = length(SessionData.RawEvents.Trial);

    % Create a colormap from light blue to dark blue
    colors = [linspace(0.6, 0, numTrials)', linspace(0.6 , 0,numTrials)', linspace(1, 0.5, numTrials)'];
    figure('units','centimeters','position',[2 2 24 26])

    numCurves = 0;
    leg_str = cell(1,2);
    legend_handles = [];
    x = [];
    shift = [];
    % Initialize an empty array to store all eyeAreaPixels values
    allEyeAreaPixels = [];

    % Loop through each trial to collect eyeAreaPixels data
    for trialIdx = 1:numTrials    
       eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
       allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data 
    end
    % Find the overall maximum value across all collected eyeAreaPixels
    overallMax = max(allEyeAreaPixels);

    step = 1;
    for ctr_trial = 1:step:numTrials

        numCurves = numCurves+1;
       
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

        % FEC = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FEC;

        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        LED_Onset_Zero_End = AirPuff_End - LED_Onset;
        AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
        AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;
        
        if contains(data_files(i).name, 'V_2_9') || ...
           contains(data_files(i).name, 'V_3_0')
            FEC_led_aligned = FECTimes + ITI_Pre - LED_Onset;
        else
            FEC_led_aligned = FECTimes - LED_Onset;
        end
        
        % FEC_led_aligned = FECTimes - LED_Puff_ISI_start;
        FEC_norm= 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels /overallMax;

        fps = 250; % frames per second, frequency of images
        seconds_before = 0.5;
        seconds_after = 2;

        Frames_before = fps * seconds_before;
        Frames_after = fps * seconds_after;

        abs_FEC_led_aligned = abs(FEC_led_aligned);
        closest_frame_idx_to_LED_Onset = find(abs_FEC_led_aligned == min(abs_FEC_led_aligned));

        start_idx = closest_frame_idx_to_LED_Onset - Frames_before;
        stop_idx = closest_frame_idx_to_LED_Onset + Frames_after;



        len_FEC_led_aligned = length(FEC_led_aligned);
        start_idx = max(1, min(start_idx, len_FEC_led_aligned));
        stop_idx = max(1, min(stop_idx, len_FEC_led_aligned));

        FEC_led_aligned_trimmed = FEC_led_aligned(start_idx : stop_idx);
        FEC_trimmed = FEC_norm(start_idx : stop_idx);

    
        h(numCurves) = plot(FEC_led_aligned_trimmed, FEC_trimmed , 'Color', colors(ctr_trial, :)); hold on

        if ctr_trial==1 
            leg_str{1} = sprintf('Trial  %03.0f ',ctr_trial);
            legend_handles(1) = h(numCurves);
        end
    
    end

    leg_str{2} = sprintf('Trial  %03.0f ',numTrials);
    legend_handles(2) = h(numCurves);

% taking average
% FECTime_sum = zeros(size(FEC_led_aligned_trimmed));
FEC_norm_sum = zeros(size(FEC_norm));
for ctr_curve=1:numCurves
    % FECTime_sum = FECTime_sum + FEC_led_aligned_trimmed;
    FEC_norm_sum = FEC_norm_sum + FEC_norm;
end
% FECTime_avg = FECTime_sum/numCurves;
FEC_norm_avg = FEC_norm_sum/numCurves;
FEC_norm_avg_trimmed = FEC_norm_avg(start_idx : stop_idx);

% plotting average curve
h_avg = plot(FEC_led_aligned_trimmed,FEC_norm_avg_trimmed, 'Color', 'r','LineStyle','-','LineWidth',1.3);
leg_str{3} = 'Average';
legend_handles(3) = h_avg;

numTrials = length(SessionData.RawEvents.Trial);

% Shade the area (LED Duration)
for ctr_trial = 1:numTrials
    x_fill = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End,LED_Onset_Zero_Start];         % x values for the filled area
    y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
    fill(x_fill, y_fill, 'green', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
end

% Shade the area (AirPuff Duration)
for ctr_trial = 1:numTrials
    x_fill = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End,AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];         % x values for the filled area
    y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
    fill(x_fill, y_fill, 'm', 'FaceAlpha', 0.35, 'EdgeColor', 'none');
end

% Add grid lines
grid on;
set(gca, 'Box', 'off');
ylim([0 1])
set(gca,'FontSize',14)
ylabel_text(1) = {'Eyelid closure (norm)'};
ylabel(ylabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
xlabel_text(1) = {'Time from LED_Onset (sec)'};
xlabel(xlabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
h_legend = legend(legend_handles, leg_str, 'Interpreter','latex','fontsize',13,'location','northeast','Box','off');
h_legend.NumColumns = 1;
h_legend_pos = h_legend.Position;
h_legend.Position = [0.98*h_legend_pos(1) 0.99*h_legend_pos(2) h_legend_pos(3) h_legend_pos(4)];
clear xlabel_text ylabel_text leg_str_1 leg_str_2 leg_str_3

% Set tick marks to be outside
set(gca, 'TickDir', 'out');
% adding text annotations
text_cell{1} = 'LED';
t1 = text(0, 1, text_cell,'interpreter', 'latex','fontname','Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left','VerticalAlignment','middle');
t1.Rotation = 90;
text_cell{1} = 'AirPuff';
t2 = text(0.3, 1, text_cell,'interpreter', 'latex','fontname','Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left','VerticalAlignment','middle');
t2.Rotation = 90;

load(data_files(i).name)

% Extract parts of the filename
[~, name, ~] = fileparts(data_files(i).name); % Extract the name without extension

% Split the filename to get the required parts
nameParts = split(name, '_');
if length(nameParts) >= 5
    prefixPart = nameParts{1}; % First string before the first underscore
    datePart = nameParts{6}; % Date part

    % Construct the new filename
    newFilename = sprintf('%s_FEC_all_trials_%s.pdf', prefixPart, datePart);
    
    % Export the graphics
    exportgraphics(gcf, newFilename, 'ContentType', 'vector');
else
    error('Filename does not have the expected format');
end

end