clc;close all;clear

 data_files = dir('*_EBC_*.mat');
 
 all_avg_curves = {};

 % Define a common time vector for interpolation
 common_time_vector = linspace(0, 4, 1000); % Adjust the range and number of points as needed

 % figure('units','centimeters','position',[2 2 24 26])
 % hold on;
 % 
 % colors = lines(length(data_files));

 for i = 1:length(data_files)

     load(data_files(i).name);
     numTrials = length(SessionData.RawEvents.Trial);
    

    % Initialize an empty array to store all eyeAreaPixels values
    allEyeAreaPixels = [];
    % Loop through each trial to collect eyeAreaPixels data
    for trialIdx = 1:numTrials    
        eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
        allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data 
    end
    % Find the overall maximum value across all collected eyeAreaPixels
    overallMax = max(allEyeAreaPixels);
    
    numCurves = 0;
    FECTime = {};
    FEC_norm = {};

    for ctr_trial = 1:numTrials
        
        numCurves = numCurves+1;
        % FECTime{numCurves} = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;
        FECTime{numCurves} = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes - SessionData.RawEvents.Trial{1, ctr_trial}.States.ITI_Pre(1);
        FEC_norm{numCurves}= 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels /overallMax;

    end


        % Interpolate data onto the common time vector
    FEC_interp = zeros(numCurves, length(common_time_vector));
    for ctr_curve = 1:numCurves
        FEC_interp(ctr_curve, :) = interp1(FECTime{ctr_curve}, FEC_norm{ctr_curve}, common_time_vector, 'linear', 'extrap');
    end
    
    % Taking average
    FEC_norm_avg = mean(FEC_interp, 1);

    all_avg_curves{end+1} = struct ('FECTime_avg' , common_time_vector, 'FEC_norm_avg', FEC_norm_avg);
end

% Plot all average curves in a single figure
% colors = lines(length(all_avg_curves)); % Use distinct colors for each average curve
leg_str = cell(1, length(all_avg_curves));
% Plot all average curves in a single figure
figure('units', 'centimeters', 'position', [2 2 24 26]);
hold on;
% Define a colormap from dim to dark blue
num_colors = length(all_avg_curves);
color_map = [linspace(0.5, 0, num_colors)', linspace(0.5, 0, num_colors)', linspace(1, 0.5, num_colors)'];

for i = 1:length(all_avg_curves)
    avg_curve = all_avg_curves{i};
    plot(avg_curve.FECTime_avg, avg_curve.FEC_norm_avg, 'Color', color_map(i, :), 'LineWidth', 1.7);
    % leg_str{i} = sprintf('Average %d', i);
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
            legend_entries{i} = sprintf('%s_%s', prefixPart,datePart); % Add a legend entry for each file
        end

end  
legend(legend_entries, 'Interpreter', 'latex', 'fontsize', 13, 'location', 'southeast', 'Box', 'off');
ylim([0 1]);
xlim([0 4.5]); % Set x-axis limits to display up to 5 seconds
set(gca, 'FontSize', 14);
ylabel('Eyelid closure (norm)', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 17);
xlabel('Time from Trial start (s)', 'interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 17);

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
    newFilename = sprintf('All_FEC_Averages_Curves_Days_%s.pdf', prefixPart);
    
    % Export the graphics
    exportgraphics(gcf, newFilename, 'ContentType', 'vector');
else
    error('Filename does not have the expected format');
end
end