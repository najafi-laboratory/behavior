clc;close all;clear

load('E1VT_EBC_V_3_0_20240710_092100.mat')



numTrials = length(SessionData.RawEvents.Trial);
% Create a colormap from light blue to dark blue
colors = [linspace(0, 1, numTrials)', zeros(numTrials, 1), linspace(1, 0, numTrials)'];


figure('units','centimeters','position',[2 2 24 26])
numCurves = 0;
for ctr_trial = 1:10:numTrials
numCurves = numCurves+1;

FECTime{numCurves} = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;
FEC_raw{numCurves} = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels ./ SessionData.RawEvents.Trial{1, ctr_trial}.Data.totalEllipsePixels;


plot(FECTime{numCurves},FEC_raw{numCurves}, 'Color', colors(ctr_trial, :)); hold on
leg_str{numCurves} = sprintf('Trial \\#%03.0f',ctr_trial);
end


% taking average
FECTime_sum = zeros(size(FECTime{1}));
FEC_raw_sum = zeros(size(FEC_raw{1}));
for ctr_curve=1:numCurves
FECTime_sum = FECTime_sum + FECTime{ctr_curve};
FEC_raw_sum = FEC_raw_sum + FEC_raw{ctr_curve};
end
FECTime_avg = FECTime_sum/numCurves;
FEC_raw_avg = FEC_raw_sum/numCurves;

% plotting average curve
plot(FECTime_avg,FEC_raw_avg, 'Color', 'g','LineStyle','-','LineWidth',1.7)
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
ylabel_text(1) = {'FEC'};
ylabel(ylabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
xlabel_text(1) = {'$t\ {\rm{(s)}}$'};
xlabel(xlabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',17)
h_legend = legend(leg_str,'Interpreter','latex','fontsize',13,'location','southeast','Box','off');
h_legend.NumColumns = 3;
h_legend_pos = h_legend.Position;
h_legend.Position = [0.98*h_legend_pos(1) 0.99*h_legend_pos(2) h_legend_pos(3) h_legend_pos(4)];
clear xlabel_text ylabel_text leg_str


% adding text annotations
text_cell{1} = 'LED';
t1 = text(1, 1, text_cell,'interpreter', 'latex','fontname','Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left','VerticalAlignment','middle');
t1.Rotation = 90;
text_cell{1} = 'AirPuff';
t2 = text(1.3, 1, text_cell,'interpreter', 'latex','fontname','Times New Roman', 'FontSize', 17, 'Color', 'black', 'HorizontalAlignment', 'left','VerticalAlignment','middle');
t2.Rotation = 90;


exportgraphics(gcf,'Trials_vs_t.pdf', 'ContentType', 'vector');