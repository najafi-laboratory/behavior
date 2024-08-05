clc;close all;clear

% data_files = dir('E1VT_EBC_V_2_9_20240704_151618.mat');
% data_files = dir('*_EBC_*.mat');
% data_files = dir('C:\behavior\session_data\E1VT\E1VT_EBC_V_3_7_20240730_132615.mat');
% data_files = dir('C:\behavior\session_data\E2WT\E2WT_EBC_V_3_6_20240726_110535.mat');
% data_files = dir('C:\behavior\session_data\E3VT\E3VT_EBC_V_3_6_20240726_131038.mat');
% E1VT
% E2WT
% E3VT
% data_files = dir('C:\behavior\session_data\E1VT\*_EBC_*.mat');
% data_files = dir('C:\behavior\session_data\E2WT\*_EBC_*.mat');
% data_files = dir('C:\behavior\session_data\E3VT\*_EBC_*.mat');
% data_files = dir('C:\behavior\session_data\E1VT\E1VT_EBC_V_2_9_20240703_150505.mat');
data_files = dir('C:\behavior\session_data\new\E3VT\*_EBC_*.mat');

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

    % Extract Session Data for each trial
    for ctr_trial = 1:numTrials        
        
        % get CheckEyeOpen if it is in session data for versions V_3_2+
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpen')
            CheckEyeOpen = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(1);
        end

        Start = SessionData.RawEvents.Trial{1, ctr_trial}.States.Start(1);
        ITI_Pre = SessionData.RawEvents.Trial{1, ctr_trial}.States.ITI_Pre(1);
        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
        AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;
        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;   
        FEC_raw = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels ./ SessionData.RawEvents.Trial{1, ctr_trial}.Data.totalEllipsePixels;

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
        
        FEC_led_aligned = FEC_led_shifted(start_idx : stop_idx); % FECTimes aligned to seconds before/after LED Onset
        FEC_aligned = FEC_raw(start_idx : stop_idx); % FEC raw aligned to seconds before/after LED Onset

        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        LED_Onset_Zero_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End - LED_Onset;
        AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
        AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;        

        t_LED = LED_Onset_Zero_Start;
        t_puff = AirPuff_LED_Onset_Aligned_End;
        t1 = t_LED-0.01;
        t2 = t_LED; 

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
        if(CR_plus_eval(FEC_led_aligned,FEC_aligned,t1,t2,t_LED,t_puff,CR_threshold))
            color_str = 'r';
        end
        
        plot(FEC_led_aligned,FEC_aligned, 'Color', color_str);hold on
        

        % Shade the area (ITI)
        x_fill = [LED_Onset_Zero_Start, LED_Onset_Zero_End,LED_Onset_Zero_End,LED_Onset_Zero_Start];         % x values for the filled area
        y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
        fill(x_fill, y_fill, 'green', 'FaceAlpha', 0.05, 'EdgeColor', 'none');
        % Shade the area (AirPuff Duration)
        x_fill = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End,AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];         % x values for the filled area
        y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
        fill(x_fill, y_fill, 'm', 'FaceAlpha', 0.35, 'EdgeColor', 'none');
        
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