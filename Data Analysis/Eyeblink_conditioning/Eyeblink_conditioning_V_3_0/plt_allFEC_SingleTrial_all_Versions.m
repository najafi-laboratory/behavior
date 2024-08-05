clc;close all;clear

% data_files = dir('E1VT_EBC_V_2_9_20240704_151618.mat');
data_files = dir('*_EBC_*.mat');
CR_threshold = 0.05;

n_row = 4;
n_column = 5;




for ctr_file=1:length(data_files)

load(data_files(ctr_file).name)
delete(strrep(data_files(ctr_file).name, '.mat', '.pdf'))

numTrials = length(SessionData.RawEvents.Trial);
% Create a colormap from light blue to dark blue
colors = [linspace(0, 1, numTrials)', zeros(numTrials, 1), linspace(1, 0, numTrials)'];


figure('units','centimeters','position',[1 1 50 30])

for ctr_trial = 1:numTrials

        CheckEyeOpen = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(2);
        Start = SessionData.RawEvents.Trial{1, ctr_trial}.States.Start(1);
        ITI_Pre = SessionData.RawEvents.Trial{1, ctr_trial}.States.ITI_Pre(1);
        % LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Onset(1);
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        LED_Puff_ISI_start = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
        LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
        AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
        AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;
        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;   
        % FEC = SessionData.RawEvents.Trial{1, trial}.Data.FEC;;
        FEC_raw = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels ./ SessionData.RawEvents.Trial{1, ctr_trial}.Data.totalEllipsePixels;

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

        % FEC_led_aligned = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes - LED_Puff_ISI_start;
        
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
        FEC_trimmed = FEC_raw(start_idx : stop_idx);
        
        % t_LED = LED_Onset_Zero_Start;
        % t_puff = AirPuff_LED_Onset_Aligned_End;
        % t1 = t_LED-0.01;
        % t2 = t_LED; 



ctr_row = mod(ceil(ctr_trial/n_column)-1,n_row);
ctr_column = mod(ctr_trial-1,n_column);
shrinking_row = 0.9;
shrinking_column = 0.8;



subplot('Position',[0.03+ctr_column/n_column,...
                   0.80-ctr_row/n_row,...
                   shrinking_column/n_column,...
                   shrinking_row/n_column,...
                   ]);

color_str = 'b';
if(CR_plus_eval(FEC_led_aligned_trimmed,FEC_trimmed,t1,t2,t_LED,t_puff,CR_threshold))
color_str = 'r';
end

plot(FEC_led_aligned_trimmed,FEC_trimmed, 'Color', color_str);hold on

% for ctr_trial = 1:numTrials

% Shade the area (ITI)
x_fill = [LED_Onset_Zero_Start, LED_Onset_Zero_End,LED_Onset_Zero_End,LED_Onset_Zero_Start];         % x values for the filled area
y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
fill(x_fill, y_fill, 'green', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
% Shade the area (AirPuff Duration)
x_fill = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End,AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];         % x values for the filled area
y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
fill(x_fill, y_fill, 'm', 'FaceAlpha', 0.35, 'EdgeColor', 'none');
% end

hold off


grid on;
% Set tick marks to be outside
set(gca, 'TickDir', 'out');
box off
ylim([0 1])
set(gca,'FontSize',11)

title_text(1) = {sprintf(' Trial %03.0f',ctr_trial)};
title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',13)

if((ctr_column==0))
ylabel_text(1) = {'FEC'};
ylabel(ylabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',13)
end

if((ctr_row==3))
xlabel_text(1) = {'$t\ {\rm{(s)}}$'};
xlabel(xlabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',13)
end


% leg_str{numCurves} = sprintf('%s: Trial \\#%03.0f',SessionData.Info.SessionDate,ctr_trial);
% bar_str{numCurves} = sprintf('Trial \\#%03.0f',ctr_trial);



if (mod(ctr_trial,20)==0)    
exportgraphics(gcf,strrep(data_files(ctr_file).name, '.mat', '.pdf'), 'ContentType', 'vector','Append',true);
% break
end



end



end