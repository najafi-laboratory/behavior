clc; close all; clear;

data_files = dir('*_EBC_*.mat');
CR_threshold = 0.05;

% Initialize a figure for the average curves
figure('units', 'centimeters', 'position', [2 2 24 26]);
hold on;
legend_entries = {};

% Define a common time vector for interpolation
commonFECTime = linspace(0, 4.3, 500); % Adjust 0 and 5 based on your expected time range

for i = 1:length(data_files)
    load(data_files(i).name);

     numTrials = length(SessionData.RawEvents.Trial);
    % color_map = [linspace(0.5, 0, numTrials)', linspace(0.5, 0, numTrials)', linspace(1, 0.5, numTrials)'];

    numCurves = 0;
    totalFEC_norm = [];
    allEyeAreaPixels = [];

    % Loop through each trial to collect eyeAreaPixels data
    for trialIdx = 1:numTrials    
       eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
       allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data 
    end

    overallMax = max(allEyeAreaPixels);

    for ctr_trial = 1:numTrials
        % FECTime = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;
        FECTime = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes - SessionData.RawEvents.Trial{1, ctr_trial}.States.ITI_Pre(1);
        FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels / overallMax;
        t_LED = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
        t_puff = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
        t1 = t_LED - 0.1;
        t2 = t_LED;

        if(CR_plus_eval(FECTime, FEC_norm, t1, t2, t_LED, t_puff, CR_threshold))
            numCurves = numCurves + 1;
            % Interpolate FEC_norm onto the common time vector
            FEC_norm_interp = interp1(FECTime, FEC_norm, commonFECTime, 'linear', 'extrap');
            totalFEC_norm = [totalFEC_norm; FEC_norm_interp];
        end
    end
   
    numFiles = length(data_files);
    cmap = [linspace(0.6, 0, numFiles)', linspace(0.6, 0, numFiles)', linspace(1, 0.5, numFiles)'];
    if numCurves > 0
        avgFECTime = commonFECTime;
        avgFEC_norm = mean(totalFEC_norm, 1);
        
        % Plot the average curve in the main figure
        h_avg = plot(avgFECTime, avgFEC_norm, 'LineWidth', 2, 'Color', cmap(i, :)); % Adjust color or style as needed
        % legend_entries{end + 1} = sprintf('File %d', i); % Add a legend entry for each file
    else
        disp('No CR+ trials found.');
    end
end

% adding text annotations
text_cell{1} = 'LED';
t1 = text(1, 1, text_cell,'interpreter', 'latex','fontname','Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left','VerticalAlignment','middle');
t1.Rotation = 90;
text_cell{1} = 'AirPuff';
t2 = text(1.3, 1, text_cell,'interpreter', 'latex','fontname','Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left','VerticalAlignment','middle');
t2.Rotation = 90;

load(data_files(1).name);
numTrials = length(SessionData.RawEvents.Trial);
for ctr_trial = 1:numTrials
    % Shade the area (ITI)
    LED_start_time = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start - SessionData.RawEvents.Trial{1, ctr_trial}.States.ITI_Pre(1);
    LED_stop_time = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End - SessionData.RawEvents.Trial{1, ctr_trial}.States.ITI_Pre(1);
    
    x_fill = [LED_start_time, LED_stop_time,LED_stop_time, LED_start_time];         % x values for the filled area
    y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
    fill(x_fill, y_fill, 'green', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
end

for ctr_trial = 1:numTrials
    % Shade the area (AirPuff Duration)
    AirPuff_start_time = SessionData.RawEvents.Trial{1, 1}.Events.GlobalTimer2_Start;
    AirPuff_stop_time = SessionData.RawEvents.Trial{1, 1}.Events.GlobalTimer2_End;
    x_fill = [AirPuff_start_time, AirPuff_stop_time, AirPuff_start_time,AirPuff_stop_time];         % x values for the filled area
    y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
    fill(x_fill, y_fill, 'yellow', 'FaceAlpha', 0.35, 'EdgeColor', 'none');
end

for i = 1:length(data_files)
load(data_files(i).name)
% Extract parts of the filename
[~, name, ~] = fileparts(data_files(i).name); % Extract the name without extension
        nameParts = split(name, '_');
        if length(nameParts) >= 5
            prefixPart = nameParts{1}; % First string before the first underscore
            datePart = nameParts{6}; % Date part
            legend_entries{end + 1} = sprintf('%s_%s', prefixPart,datePart); % Add a legend entry for each file
        end

end        
% Finalize the main figure
legend(legend_entries, 'Interpreter', 'latex', 'fontsize', 13, 'location', 'southeast', 'Box', 'off');
xlabel('Time from Trial start (s)', 'Interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 17);
ylabel('Eyelid closure (norm)', 'Interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 17);
% title({'Average Superimposed CR+ Curves'}, 'Interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 19);
set(gca, 'FontSize', 14);
ylim([0 1]);

load(data_files(i).name)
for i = 1:length(data_files)
% Extract parts of the filename
[~, name, ~] = fileparts(data_files(i).name); % Extract the name without extension

% Split the filename to get the required parts
nameParts = split(name, '_');
if length(nameParts) >= 5
    prefixPart = nameParts{1}; % First string before the first underscore
    datePart = nameParts{6}; % Date part

    % Construct the new filename
    newFilename = sprintf('All_FEC_Averages_CRplus_Days_%s.pdf', prefixPart);
    
    % Export the graphics
    exportgraphics(gcf, newFilename, 'ContentType', 'vector');
else
    error('Filename does not have the expected format');
end
end



