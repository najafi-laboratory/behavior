clc;close all;clear

data_files = dir('*_EBC_*.mat');


for i = 1:length(data_files)
 
    
    load(data_files(i).name)

numTrials = length(SessionData.RawEvents.Trial);
% Create a colormap from light blue to dark blue
colors = [linspace(0.6, 0, numTrials)', linspace(0.6 , 0,numTrials)', linspace(1, 0.5, numTrials)'];

figure('units','centimeters','position',[2 2 24 26])
numCurves = 0;
leg_str = cell(1,2);
legend_handles = [];
% Initialize an empty array to store all eyeAreaPixels values
allEyeAreaPixels = [];

% Loop through each trial to collect eyeAreaPixels data
for trialIdx = 1:numTrials    
   eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
   allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data 
end
% Find the overall maximum value across all collected eyeAreaPixels
overallMax = max(allEyeAreaPixels);

for ctr_trial = 1:10:numTrials
numCurves = numCurves+1;

FECTime{numCurves} = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;

FEC_norm{numCurves} = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels /overallMax;

h(numCurves) = plot(FECTime{numCurves},FEC_norm{numCurves}, 'Color', colors(ctr_trial, :)); hold on

if ctr_trial==1 
    leg_str{1} = sprintf('Trial  %03.0f ',ctr_trial);
    legend_handles(1) = h(numCurves);


end

end

    leg_str{2} = sprintf('Trial  %03.0f ',numTrials);
    legend_handles(2) = h(numCurves);

% taking average
FECTime_sum = zeros(size(FECTime{1}));
FEC_norm_sum = zeros(size(FEC_norm{1}));
for ctr_curve=1:numCurves
FECTime_sum = FECTime_sum + FECTime{ctr_curve};
FEC_norm_sum = FEC_norm_sum + FEC_norm{ctr_curve};
end
FECTime_avg = FECTime_sum/numCurves;
FEC_norm_avg = FEC_norm_sum/numCurves;

% plotting average curve
h_avg = plot(FECTime_avg,FEC_norm_avg, 'Color', 'g','LineStyle','-','LineWidth',1.7)
leg_str{3} = 'Average';
legend_handles(3) = h_avg;

for ctr_trial = 1:numTrials

% Shade the area (ITI)
x_fill = [SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Onset(1), SessionData.RawEvents.Trial{1, ctr_trial}.States.AirPuff(1),...
          SessionData.RawEvents.Trial{1, ctr_trial}.States.AirPuff(1) SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Onset(1)];         % x values for the filled area
y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
fill(x_fill, y_fill, 'green', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
end

for ctr_trial = 1:numTrials
% Shade the area (AirPuff Duration)
x_fill = [SessionData.RawEvents.Trial{1, ctr_trial}.States.AirPuff(1), SessionData.RawEvents.Trial{1, ctr_trial}.States.AirPuff(2),...
          SessionData.RawEvents.Trial{1, ctr_trial}.States.AirPuff(2) SessionData.RawEvents.Trial{1, ctr_trial}.States.AirPuff(1)];         % x values for the filled area
y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
fill(x_fill, y_fill, 'yellow', 'FaceAlpha', 0.35, 'EdgeColor', 'none');
end

ylim([0 1])
set(gca,'FontSize',14)
ylabel_text(1) = {'Eyelid closure (norm)'};
ylabel(ylabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
xlabel_text(1) = {'Time from Trial start (s)'};
xlabel(xlabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
h_legend = legend(legend_handles, leg_str, 'Interpreter','latex','fontsize',13,'location','southeast','Box','off');
h_legend.NumColumns = 1;
h_legend_pos = h_legend.Position;
h_legend.Position = [0.98*h_legend_pos(1) 0.99*h_legend_pos(2) h_legend_pos(3) h_legend_pos(4)];
clear xlabel_text ylabel_text leg_str_1 leg_str_2 leg_str_3


% adding text annotations
text_cell{1} = 'LED';
t1 = text(1, 1, text_cell,'interpreter', 'latex','fontname','Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left','VerticalAlignment','middle');
t1.Rotation = 90;
text_cell{1} = 'AirPuff';
t2 = text(1.3, 1, text_cell,'interpreter', 'latex','fontname','Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left','VerticalAlignment','middle');
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
    newFilename = sprintf('FEC_all_trials_%s_%s.pdf', datePart, prefixPart);
    
    % Export the graphics
    exportgraphics(gcf, newFilename, 'ContentType', 'vector');
else
    error('Filename does not have the expected format');
end

end