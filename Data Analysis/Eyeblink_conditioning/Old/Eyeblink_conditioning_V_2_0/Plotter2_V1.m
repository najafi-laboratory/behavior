clc;close all;clear



filenames = {
     'E2WT_EBC_V_2_9_20240703_114730',
    'E2WT_EBC_V_2_9_20240704_170635',
    'E2WT_EBC_V_3_0_20240705_123232'
 
}


CR_threshold = 0.05;

for i = 1:length(filenames)
 
    filename  = filenames{i};
    load(filename);


numTrials = length(SessionData.RawEvents.Trial);
% Create a colormap from light blue to dark blue
colors = [linspace(0.6, 0, numTrials)', linspace(0.6 , 0,numTrials)', linspace(1, 0.5, numTrials)'];


figure('units','centimeters','position',[2 2 24 26])
numCurves = 0;
totalFECTime = [];
totalFEC_norm = [];
% Initialize an empty array to store all eyeAreaPixels values
allEyeAreaPixels = [];

% Loop through each trial to collect eyeAreaPixels data
for trialIdx = 1:numTrials    
   eyeAreaPixels = SessionData.RawEvents.Trial{1, trialIdx}.Data.eyeAreaPixels;
   allEyeAreaPixels = [allEyeAreaPixels, eyeAreaPixels]; % Concatenate data 
end
% Find the overall maximum value across all collected eyeAreaPixels
overallMax = max(allEyeAreaPixels);

for ctr_trial = 1:numTrials


FECTime = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;

FEC_norm = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels /overallMax;

t_LED = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
t_puff = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
t1 = t_LED-0.1;
t2 = t_LED;

if(CR_plus_eval(FECTime,FEC_norm,t1,t2,t_LED,t_puff,CR_threshold))
numCurves = numCurves+1;
% Accumulate the data
totalFECTime = [totalFECTime; FECTime];
totalFEC_norm = [totalFEC_norm; FEC_norm];

plot(FECTime,FEC_norm, 'Color', colors(ctr_trial, :)); hold on
leg_str{numCurves} = sprintf('%s: Trial %03.0f',SessionData.Info.SessionDate,ctr_trial);
% bar_str{numCurves} = sprintf('Trial %03.0f',ctr_trial);


end

end

% Calculate the average FEC curve
if numCurves > 0
    % Ensure the time vectors are aligned for averaging
    % Assuming all FECTime vectors are identical, you can use the first one for averaging
    avgFECTime = totalFECTime(1, :);
    avgFEC_norm = mean(totalFEC_norm, 1);

    % Plot the average curve
    plot(avgFECTime, avgFEC_norm, 'k-', 'LineWidth', 2); % Black line for the average curve
    legend([leg_str, {'Average Curve'}]);
    % title('Average Superimposed CR+ Curve');
    % xlabel('Time (s)');
    % ylabel('FEC (Normalized)');
else
    disp('No CR+ trials found.');
end

CR_plus_fraction(i) =  numCurves/numTrials;
CR_plus_fraction_x{i} = SessionData.Info.SessionDate;
% CR_plus_trials{i} = bar_str;



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
set(gca,'Position',[0.1 0.1 0.8 0.7])
ylabel_text(1) = {'FEC'};
ylabel(ylabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
xlabel_text(1) = {'$t\ {\rm{(s)}}$'};
xlabel(xlabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
title_text(1) = {'CR$^{+}$ trials in which FEC goes above the baseline+CR$_{\rm{th}}$ in $(T_{\rm{LED}},T_{\rm{AirPuff}})$'};
title_text(2) = {sprintf('${\\rm{CR}}_{\\rm{th}} = %.2f $',CR_threshold)};
title_text(3) = {' '}; title_text(4) = {' '};
title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',19)
h_legend = legend(leg_str,'Interpreter','latex','fontsize',13,'location','southeast','Box','off');
h_legend.NumColumns = 2;
h_legend_pos = h_legend.Position;
h_legend.Position = [0.98*h_legend_pos(1) 0.99*h_legend_pos(2) h_legend_pos(3) h_legend_pos(4)];
clear xlabel_text ylabel_text title_text leg_str


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
    datePart = nameParts{6}; % Date par    
    % Construct the new filename
    
    newFilename_1 = sprintf('FEC_CRplus_trials_%s_%s.pdf', datePart, prefixPart);
  
    % Export the graphics
    exportgraphics(gcf, newFilename_1, 'ContentType', 'vector');
else
    error('Filename does not have the expected format');
end





%
figure('units','centimeters','position',[2 2 20 16])
bar(CR_plus_fraction*100);
set(gca,'xticklabel',CR_plus_fraction_x)
set(gca,'FontSize',14)
ylim([0 10])
ylabel_text(1) = {'CR$^{+}$ trials (\%)'};
ylabel(ylabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
xlabel_text(1) = {' '};

xlabel(xlabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)


for ctr_bar=1:length(CR_plus_fraction)
t1 = text(ctr_bar, CR_plus_fraction(ctr_bar)*100,xlabel_text(1),'interpreter', 'latex','fontname','Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'center','VerticalAlignment','bottom');
end

load(filename);

% Extract parts of the filename
[~, name, ~] = fileparts(filename); % Extract the name without extension

% Split the filename to get the required parts
nameParts = split(name, '_');
if length(nameParts) >= 5
    prefixPart = nameParts{1}; % First string before the first underscore
    datePart = nameParts{6}; % Date part

    % Extract year and month from the date part
    yearMonth = datePart(1:6); % Extract YYYYMM format
        
    % Construct the new filename
    newFilename = sprintf('CR_Fraction_Barplot_%s_%s.pdf', yearMonth, prefixPart);

else
    error('Filename does not have the expected format');
end

end
 % Export the graphics
    exportgraphics(gcf, newFilename, 'ContentType', 'vector');