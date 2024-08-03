clc;close all;clear
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


figure('units','centimeters','position',[2 2 50 30])

for ctr_trial = 1:numTrials


FECTime = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;
FEC_raw = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels ./ SessionData.RawEvents.Trial{1, ctr_trial}.Data.totalEllipsePixels;

t_LED = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
t_puff = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
t1 = t_LED-0.1;
t2 = t_LED;



ctr_row = mod(ceil(ctr_trial/n_column)-1,n_row);
ctr_column = mod(ctr_trial-1,n_column);
shrinking_row = 0.9;
shrinking_column = 0.8;



subplot('Position',[0.03+ctr_column/n_column,...
                   0.05+ctr_row/n_row,...
                   shrinking_column/n_column,...
                   shrinking_row/n_column,...
                   ]);

color_str = 'b';
if(CR_plus_eval(FECTime,FEC_raw,t1,t2,t_LED,t_puff,CR_threshold))
color_str = 'r';
end

plot(FECTime,FEC_raw, 'Color', color_str);hold on
% Shade the area
x_fill = [SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1), SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2),...
          SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1)];         % x values for the filled area
y_fill = [0 0 1 1];    % y values for the filled area (y=0 at the x-axis)
fill(x_fill, y_fill, 'k', 'FaceAlpha', 0.45, 'EdgeColor', 'none');
hold off




ylim([0 1])
set(gca,'FontSize',11)

title_text(1) = {sprintf('%s: Trial %03.0f',SessionData.Info.SessionDate,ctr_trial)};
title(title_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',13)

if((ctr_column==0))
ylabel_text(1) = {'FEC'};
ylabel(ylabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',13)
end

if((ctr_row==0))
xlabel_text(1) = {'$t\ {\rm{(s)}}$'};
xlabel(xlabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',13)
end


% leg_str{numCurves} = sprintf('%s: Trial \\#%03.0f',SessionData.Info.SessionDate,ctr_trial);
% bar_str{numCurves} = sprintf('Trial \\#%03.0f',ctr_trial);



if (mod(ctr_trial,20)==0)
exportgraphics(gcf,strrep(data_files(ctr_file).name, '.mat', '.pdf'), 'ContentType', 'vector','Append',true);
end



end



end