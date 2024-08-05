clc;close all;clear
data_files = dir('E1VT_EBC_V_3_0_20240710_092100.mat');

CR_threshold = 0.05;


for ctr_file=1:length(data_files)

load(data_files(ctr_file).name)

numTrials = length(SessionData.RawEvents.Trial);
% Create a colormap from light blue to dark blue
colors = [linspace(0, 1, numTrials)', zeros(numTrials, 1), linspace(1, 0, numTrials)'];


figure('units','centimeters','position',[2 2 24 26])
numCurves = 0;
for ctr_trial = 1:numTrials


FECTime = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;
FEC_raw = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels ./ SessionData.RawEvents.Trial{1, ctr_trial}.Data.totalEllipsePixels;

t_LED = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
t_puff = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
t1 = t_LED-0.1;
t2 = t_LED;

if(CR_plus_eval(FECTime,FEC_raw,t1,t2,t_LED,t_puff,CR_threshold))
numCurves = numCurves+1;
plot(FECTime,FEC_raw, 'Color', colors(ctr_trial, :)); hold on
leg_str{numCurves} = sprintf('%s: Trial \\#%03.0f',SessionData.Info.SessionDate,ctr_trial);
bar_str{numCurves} = sprintf('Trial \\#%03.0f',ctr_trial);
end


end


CR_plus_fraction(ctr_file) =  numCurves/numTrials;
CR_plus_fraction_x{ctr_file} = SessionData.Info.SessionDate;
CR_plus_trials{ctr_file} = bar_str;
end


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


exportgraphics(gcf,'Trials_vs_t_CR_plus.pdf', 'ContentType', 'vector');


%
figure('units','centimeters','position',[2 2 20 16])
bar(CR_plus_fraction*100);
set(gca,'xticklabel',CR_plus_fraction_x)
set(gca,'FontSize',14)
ylim([0 10])
ylabel_text(1) = {'CR$^{+}$ trials (\%)'};
ylabel(ylabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
xlabel_text(1) = {'trial dates'};
xlabel(xlabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)


for ctr_bar=1:length(CR_plus_fraction)
t1 = text(ctr_bar, CR_plus_fraction(ctr_bar)*100, CR_plus_trials{ctr_bar},'interpreter', 'latex','fontname','Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'center','VerticalAlignment','bottom');
end

exportgraphics(gcf,'CR_plus_bars.pdf', 'ContentType', 'vector');
