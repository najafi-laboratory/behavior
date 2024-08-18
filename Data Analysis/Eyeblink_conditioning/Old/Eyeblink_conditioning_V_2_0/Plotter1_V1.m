clc;close all;clear


filenames = {
    'E2WT_EBC_V_2_9_20240703_114730',
    'E2WT_EBC_V_2_9_20240704_170635',
    'E2WT_EBC_V_3_0_20240705_123232'
    

}

for i = 1:length(filenames)
 
    filename  = filenames{i};
    load(filename);

numTrials = length(SessionData.RawEvents.Trial);
% Create a colormap from light blue to dark blue
colors = [linspace(0.6, 0, numTrials)', linspace(0.6 , 0,numTrials)', linspace(1, 0.5, numTrials)'];

figure('units','centimeters','position',[2 2 24 26])
numCurves = 0;

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


plot(FECTime{numCurves},FEC_norm{numCurves}, 'Color', colors(ctr_trial, :)); hold on

end
leg_str_1 = sprintf('Trial  %03.0f ',1);
leg_str_2 = sprintf('...');
leg_str_3 = sprintf('Trial  %03.0f ',numTrials);

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
plot(FECTime_avg,FEC_norm_avg, 'Color', 'g','LineStyle','-','LineWidth',1.7)
leg_str{numCurves+1} = 'Average';

for ctr_trial = 1:10:numTrials
% plot([SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1) SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1)],[0 1],'Color','k')
% plot([SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2)],[0 1],'Color','k')

% Shade the area
x_fill = [SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1), SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2),...
          SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1)];         % x values for the filled area
y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
fill(x_fill, y_fill, 'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
end

ylim([0 1])
set(gca,'FontSize',14)
ylabel_text(1) = {'Eyelid closure (norm)'};
ylabel(ylabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
xlabel_text(1) = {'Time from Trial start (s)'};
xlabel(xlabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
h_legend = legend(leg_str_1,leg_str_2,leg_str_3, 'Interpreter','latex','fontsize',13,'location','southeast','Box','off');
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

load(filename);

% Extract parts of the filename
[~, name, ~] = fileparts(filename); % Extract the name without extension

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
