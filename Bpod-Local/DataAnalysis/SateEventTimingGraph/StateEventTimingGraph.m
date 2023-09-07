% obtain axes handle for plots, prepare axes
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

% some variables

% numTrials = 2;
Trial_num = 1;

trials  = SessionData.RawEvents.Trial{1, Trial_num};  % trial structure
timingStates = struct2cell(trials.States);  % the most recent state timings
timingEvents1 = struct2cell(trials.Events);
timingEvents1 = cellfun(@transpose,timingEvents1,'UniformOutput',false);
timingEvents2 = timingEvents1;
timingEvents2 = cellfun(@(x) {x + 0.01},timingEvents2);
timingEvents = cellfun(@(x, y) [x, y], timingEvents1, timingEvents2, 'UniformOutput', false);
% timingEvents3 = cat(2, timingEvents, timingEvents2);
timings = vertcat(timingStates, timingEvents);
names   = vertcat(fieldnames(trials.States), fieldnames(trials.Events));   % state names
nStatesEvents= numel(names);                     % number of states
nTrial  = numel(trials);                    % number of current trial
hBar    = .5;                               % height of bars
colors  = get(hAx, 'ColorOrder');           % a list of face colors

% correct timings by t0 & indicate it

% plot state timing
for idxState = 1:nStatesEvents
    x = [timings{idxState} fliplr(timings{idxState})]';
    y = repmat(idxState + hBar./[2;2;-2;-2], 1, size(x, 2));
    % disp(idxState);
    % disp(x);
    % disp(y);
    % disp("***************************");
    % x = [timings{idxState} fliplr(timings{idxState})]';
    % y = repmat(idxState + hBar./[2;2;-2;-2],1,size(x,2));
    c = colors(mod(idxState - 1, size(colors,1)) + 1,:);
    patch(hAx,x,y,c);
    %plot(hAx, x, y, 'O');
end

% format axes & labels
title(hAx,sprintf('State/Event Timing, Trial %d',Trial_num));
set(hAx, ...
    'YTick',      1:nStatesEvents, ...
    'YTickLabel', names, ...
    'YLim',       [.5 nStatesEvents+.5])

% end








% *************************************************************
% trials  = SessionData.RawEvents.Trial{1, 1};timings = cellfun(@transpose,timings,'UniformOutput',false)
% names = fieldnames(trials.Events);
% timings = struct2cell(trials.Events);



