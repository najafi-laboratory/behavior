% Obtain axes handle for plots, prepare axes
hAx = getappdata(0, 'StateEventTimingAxes');
if isempty(hAx) || ~isvalid(hAx)
    if verLessThan('matlab','9.5')
        error('Error: the StateEventTiming plot requires MATLAB r2018b or newer');
    end

    StateEventTimingFig = figure( ...
        'Name',                 'State/Event Timing', ...
        'NumberTitle',          'off',  ...
        'MenuBar',              'figure', ...
        'ToolBar',              'figure');

    hAx = axes(StateEventTimingFig, ...
        'YDir',                 'reverse', ...
        'XGrid',                'on', ...
        'YGrid',                'on', ...
        'Box',                  'off', ...
        'TickDir',              'out', ...
        ... % 'PickableParts',        'none', ...
        ... %'HitTest',              'off', ...
        'TickLabelInterpreter', 'none', ...
        'NextPlot',             'add');


    % axtoolbar(hAx,{'zoomin','zoomout','restoreview'});

    xlabel(hAx,'Time [s]')
    xline(hAx,0,':');
    axis(hAx,'tight');
     % plottools(StateEventTimingFig);
    % set(StateEventTimingFig,'HandleVisibility', 'on')
end

ch = get(hAx, 'Children');
delete(ch(1:end-1)); % Clear the previous patches. Last item is always the xline

% if ~isfield(SessionData, 'RawEvents') % If BpodSystem.Data has been initialized by the user but events have not been added
%     return
% end

% numTrials = 2;
Trial_num = 3;

trials  = SessionData.RawEvents.Trial{1, Trial_num};  % trial structure
% Separate timing states and timing events
timingStates = struct2cell(trials.States);  % The most recent state timings
timingEvents1 = struct2cell(trials.Events);
timingEvents1 = cellfun(@transpose, timingEvents1, 'UniformOutput', false);
timingEvents2 = timingEvents1;
timingEvents2 = cellfun(@(x) {x + 0.01}, timingEvents2);
timingEvents = cellfun(@(x, y) [x, y], timingEvents1, timingEvents2, 'UniformOutput', false);

% Get state and event names
stateNames = fieldnames(trials.States);
eventNames = fieldnames(trials.Events);

% Number of states and events
nStates = numel(stateNames);
nEvents = numel(eventNames);

% Height of bars for states and events
hBarStates = 0.5;
hBarEvents = 0.2;

% Colors for states and events 
stateColor = [0.1, 0.7, 0.2];
eventColor = [0.7, 0.1, 0.2];

% Plot state timing
for idxState = 1:nStates
    x = [timingStates{idxState}, fliplr(timingStates{idxState})]';
    y = repmat(idxState + hBarStates ./ [2; 2; -2; -2], 1, size(x, 2));
    patch(hAx, x, y, stateColor);
end

% Plot event timing as discrete dots
for idxEvent = 1:nEvents
    timings = timingEvents{idxEvent};
    y = nStates + idxEvent - hBarEvents / 2;
    plot(hAx, timings, repmat(y, size(timings)), 'o', 'Color', eventColor, 'MarkerSize', 6, 'LineWidth', 1.5);
end

% Format axes & labels
totalItems = nStates + nEvents;
names = [stateNames; eventNames];  % Combined names for states and events
set(hAx, ...
    'YTick', 1:totalItems, ...
    'YTickLabel', names, ...
    'YLim', [0.5 totalItems + 0.5])