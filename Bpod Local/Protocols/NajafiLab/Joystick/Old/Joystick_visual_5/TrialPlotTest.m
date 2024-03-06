MaxTrials = 500;

m_Plotter = Plotter;


% get uniform distribution of 2 trial types
numTrialTypes = 2;
TrialTypes = ceil(rand(1,MaxTrials)*numTrialTypes); 


BpodSystem.Data.TrialTypes = []; % The trial type of each trial completed will be added here.


% Side Outcome Plot
BpodSystem.ProtocolFigures.OutcomePlot = figure('Position', [50 540 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
BpodSystem.GUIHandles.OutcomePlot = axes('Position', [.075 .35 .89 .55]);
TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'init',(numTrialTypes+1)-TrialTypes);


m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, 0); % this will update the SideOutcomePlot to reflect the current trial type after accounting for any change due to anti-bias control 




BpodSystem.Data



m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, 1);


