function ProtocolPlot(action, optoTypes, completedCount, S, rawTrialOverride, optoTypeOverride)
global BpodSystem
if nargin < 5
    rawTrialOverride = [];
end
if nargin < 6
    optoTypeOverride = [];
end

switch action
    case 'init'
        screenSize = get(0, 'ScreenSize');
        width = round(screenSize(3) * 0.576);
        height = round(screenSize(4) * 0.504);
        position = [screenSize(1) + 32 screenSize(2) + screenSize(4) - height - 64 width height];
        BpodSystem.ProtocolFigures.Session = figure('Name', 'Opto Test Session', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none', 'Color', 'w', 'Position', position);
        BpodSystem.GUIHandles.OptoAxes = axes('Parent', BpodSystem.ProtocolFigures.Session, 'Units', 'normalized', 'Position', [0.08 0.75 0.86 0.17]);
        BpodSystem.GUIHandles.StateAxes = axes('Parent', BpodSystem.ProtocolFigures.Session, 'Units', 'normalized', 'Position', [0.08 0.40 0.86 0.22]);
        BpodSystem.GUIHandles.EventAxes = axes('Parent', BpodSystem.ProtocolFigures.Session, 'Units', 'normalized', 'Position', [0.08 0.11 0.86 0.15]);
        initializeAxes;
        drawAll(optoTypes, completedCount, S, rawTrialOverride, optoTypeOverride);
    case 'update'
        if ~isfield(BpodSystem.ProtocolFigures, 'Session') || ~isgraphics(BpodSystem.ProtocolFigures.Session)
            ProtocolPlot('init', optoTypes, completedCount, S, rawTrialOverride, optoTypeOverride);
            return
        end
        drawAll(optoTypes, completedCount, S, rawTrialOverride, optoTypeOverride);
end
end

function initializeAxes
global BpodSystem

handles = [ ...
    BpodSystem.GUIHandles.OptoAxes ...
    BpodSystem.GUIHandles.StateAxes ...
    BpodSystem.GUIHandles.EventAxes];
set(handles, 'Box', 'off', 'TickDir', 'out', 'FontSize', 10, 'LineWidth', 0.9, 'Color', 'none', 'XMinorTick', 'on');
end

function drawAll(optoTypes, completedCount, S, rawTrialOverride, optoTypeOverride)
global BpodSystem

drawOptoTypes(BpodSystem.GUIHandles.OptoAxes, optoTypes, completedCount, S);
rawTrial = normalizedRawTrial(rawTrialOverride);
if isempty(rawTrial)
    rawTrial = completedRawTrial(completedCount);
end
if rawTrialAvailable(rawTrial)
    drawStateTiming(BpodSystem.GUIHandles.StateAxes, rawTrial, completedCount);
    drawEventTiming(BpodSystem.GUIHandles.EventAxes, rawTrial, completedCount, S, optoTypeOverride);
else
    showEmpty(BpodSystem.GUIHandles.StateAxes, 'Completed state timing', 'No completed trial');
    showEmpty(BpodSystem.GUIHandles.EventAxes, 'Event timing', 'No completed trial');
end
drawnow limitrate
end

function drawOptoTypes(ax, optoTypes, completedCount, S)
global BpodSystem

cla(ax);
hold(ax, 'on');
display = OptoControl('display', S);
if isvector(optoTypes)
    optoTypes = reshape(optoTypes, 4, []);
end
total = size(optoTypes, 2);
[firstTrial, lastTrial] = visibleWindow(total, completedCount);
visibleTrials = firstTrial:lastTrial;
assignedThrough = completedCount;
if isfield(BpodSystem, 'Data') && isfield(BpodSystem.Data, 'AssignedOptoTrialCount')
    assignedThrough = max(assignedThrough, BpodSystem.Data.AssignedOptoTrialCount);
end
assignedThrough = min(assignedThrough, total);
for trial = visibleTrials
    rows = optoRows(optoTypes(:, trial));
    for row = rows
        color = display.Colors(row, :);
        if trial > assignedThrough
            plot(ax, trial, row, '.', 'Color', min(1, color * 0.5 + [0.45 0.45 0.45]), 'MarkerSize', 9);
        else
            plot(ax, trial, row, 's', 'Color', color, 'MarkerFaceColor', color, 'MarkerSize', 6);
        end
    end
end
if completedCount < total
    rows = optoRows(optoTypes(:, completedCount + 1));
    plot(ax, repmat(completedCount + 1, size(rows)), rows, 'o', 'Color', 'k', 'MarkerFaceColor', 'w', 'LineWidth', 1.1, 'MarkerSize', 7);
end
xlim(ax, [firstTrial - 0.5 lastTrial + 0.5]);
ylim(ax, [0.5 numel(display.Labels) + 0.5]);
xticks(ax, trialTicks(firstTrial, lastTrial));
yticks(ax, 1:numel(display.Labels));
yticklabels(ax, display.Labels);
set(ax, 'YMinorTick', 'off');
xlabel(ax, 'Trial');
ylabel(ax, 'Opto epoch');
title(ax, sprintf('Opto trial types  %d / %d completed', completedCount, total), 'FontWeight', 'normal');
end

function drawStateTiming(ax, rawTrial, trial)
cla(ax);
hold(ax, 'on');
if ~isfield(rawTrial, 'States') || ~isstruct(rawTrial.States)
    showEmpty(ax, 'Completed state timing', 'State timing unavailable');
    return
end
names = fieldnames(rawTrial.States);
entries = {};
for i = 1:numel(names)
    times = rawTrial.States.(names{i});
    if isempty(times) || all(isnan(times(:)))
        continue
    end
    if isvector(times)
        times = reshape(times, 1, []);
    end
    for row = 1:size(times, 1)
        if size(times, 2) >= 2 && all(isfinite(times(row, 1:2)))
            entries(end + 1, :) = {times(row, 1), times(row, 2), names{i}};
        end
    end
end
if isempty(entries)
    showEmpty(ax, 'Completed state timing', 'No state timing');
    return
end
[~, order] = sort(cell2mat(entries(:, 1)));
entries = entries(order, :);
labels = unique(entries(:, 3), 'stable');
colors = lines(max(7, numel(labels)));
for i = 1:size(entries, 1)
    row = find(strcmp(labels, entries{i, 3}), 1);
    rectangle('Parent', ax, 'Position', [entries{i, 1}, row - 0.3, max(0.005, entries{i, 2} - entries{i, 1}), 0.6], 'FaceColor', colors(row, :), 'EdgeColor', 'none');
end
duration = trialDuration(rawTrial, trial);
xlim(ax, [0 duration]);
ylim(ax, [0.5 numel(labels) + 0.5]);
setTimeAxis(ax, [0 duration]);
yticks(ax, 1:numel(labels));
yticklabels(ax, labels);
set(ax, 'YMinorTick', 'off', 'XGrid', 'on', 'GridColor', [0.82 0.82 0.82], 'GridAlpha', 0.5);
xlabel(ax, 'Time (s)');
title(ax, sprintf('Completed state timing  trial %d', trial), 'FontWeight', 'normal');
end

function drawEventTiming(ax, rawTrial, trial, S, optoTypeOverride)
cla(ax);
hold(ax, 'on');
duration = trialDuration(rawTrial, trial);
rows = {'BNC 1', 'LED 1'};
colors = [0.05 0.05 0.05; 0.45 0.45 0.45];
if isfield(rawTrial, 'Events')
    events = rawTrial.Events;
else
    events = struct;
end
bncIntervals = eventIntervals(events, 'BNC1High', 'BNC1Low', duration);
if isempty(bncIntervals) && isfield(rawTrial, 'States') && isstruct(rawTrial.States)
    bncStop = stateStart(rawTrial.States, 'ITI');
    if ~isfinite(bncStop)
        bncStop = duration;
    end
    bncIntervals = [0 bncStop];
end
drawIntervals(ax, bncIntervals, 1, colors(1, :));
led = eventIntervals(events, 'PWM1High', 'PWM1Low', duration);
if isempty(led)
    led = eventIntervals(events, 'LED1High', 'LED1Low', duration);
end
if isempty(led)
    led = ledIntervals(rawTrial, trial, duration, S, optoTypeOverride);
end
drawIntervals(ax, led, 2, colors(2, :));
xlim(ax, [0 duration]);
ylim(ax, [0.5 numel(rows) + 0.5]);
setTimeAxis(ax, [0 duration]);
yticks(ax, 1:numel(rows));
yticklabels(ax, rows);
set(ax, 'YMinorTick', 'off', 'XGrid', 'on', 'GridColor', [0.82 0.82 0.82], 'GridAlpha', 0.5);
xlabel(ax, 'Time (s)');
title(ax, sprintf('Event timing  trial %d', trial), 'FontWeight', 'normal');
end

function rows = optoRows(optoType)
enabled = find(optoType(:) ~= 0);
if isempty(enabled)
    rows = 1;
else
    rows = enabled + 1;
end
rows = reshape(rows, 1, []);
end

function intervals = ledIntervals(rawTrial, trial, duration, S, optoTypeOverride)
global BpodSystem

if ~isfield(rawTrial, 'States') || ~isstruct(rawTrial.States)
    intervals = zeros(0, 2);
    return
end
optoType = normalizeOptoType(optoTypeOverride);
if ~any(optoType) && isfield(BpodSystem, 'Data') && isfield(BpodSystem.Data, 'OptoTrialTypes') && size(BpodSystem.Data.OptoTrialTypes, 2) >= trial
    optoType = BpodSystem.Data.OptoTrialTypes(:, trial);
    optoType = normalizeOptoType(optoType);
end
settings = S;
if isfield(BpodSystem, 'Data') && isfield(BpodSystem.Data, 'TrialSettings') && numel(BpodSystem.Data.TrialSettings) >= trial
    settings = BpodSystem.Data.TrialSettings(trial);
end
intervals = zeros(0, 2);
states = rawTrial.States;
if optoType(1)
    intervals = appendInterval(intervals, stateStart(states, 'PreStimDelay'), stateEnd(states, 'PreStimDelay'), duration, settings);
end
if optoType(2)
    intervals = appendInterval(intervals, stateStart(states, 'ImageInterval'), stateEnd(states, 'ImageInterval'), duration, settings);
end
if optoType(3)
    intervals = appendInterval(intervals, stateStart(states, 'Image2Display'), stateEnd(states, 'Image2Display'), duration, settings);
end
if optoType(4)
    intervals = appendInterval(intervals, stateStart(states, 'PostStimDelay'), stateEnd(states, 'PostStimDelay'), duration, settings);
end
end

function optoType = normalizeOptoType(optoType)
optoType = optoType(:) ~= 0;
if numel(optoType) < 4
    optoType = [optoType; false(4 - numel(optoType), 1)];
elseif numel(optoType) > 4
    optoType = optoType(1:4);
end
end

function intervals = appendInterval(intervals, startTime, stopTime, duration, settings)
if ~isfinite(startTime) || ~isfinite(stopTime)
    return
end
if settings.GUI.LaserTriggerMode == 2
    stopTime = min(stopTime, startTime + settings.GUI.LaserDuration_s);
end
startTime = max(0, min(duration, startTime));
stopTime = max(0, min(duration, stopTime));
if stopTime > startTime
    intervals(end + 1, :) = [startTime stopTime];
end
end

function intervals = eventIntervals(events, onName, offName, duration)
starts = eventTimes(events, onName);
stops = eventTimes(events, offName);
intervals = zeros(numel(starts), 2);
intervalCount = 0;
stopIndex = 1;
for i = 1:numel(starts)
    while stopIndex <= numel(stops) && stops(stopIndex) <= starts(i)
        stopIndex = stopIndex + 1;
    end
    if stopIndex <= numel(stops)
        stopTime = stops(stopIndex);
        stopIndex = stopIndex + 1;
    else
        stopTime = duration;
    end
    stopTime = min(duration, stopTime);
    if stopTime > starts(i)
        intervalCount = intervalCount + 1;
        intervals(intervalCount, :) = [max(0, starts(i)) stopTime];
    end
end
intervals = intervals(1:intervalCount, :);
end

function times = eventTimes(events, name)
if isfield(events, name)
    times = sort(events.(name)(isfinite(events.(name))));
else
    times = [];
end
end

function drawIntervals(ax, intervals, row, color)
for i = 1:size(intervals, 1)
    rectangle('Parent', ax, 'Position', [intervals(i, 1), row - 0.22, max(0.005, diff(intervals(i, :))), 0.44], 'FaceColor', color, 'EdgeColor', 'none');
end
end

function showEmpty(ax, plotTitle, message)
cla(ax);
xlim(ax, [0 1]);
ylim(ax, [0 1]);
setTimeAxis(ax, [0 1]);
yticks(ax, []);
set(ax, 'YMinorTick', 'off');
xlabel(ax, 'Time (s)');
title(ax, plotTitle, 'FontWeight', 'normal');
text(ax, 0.5, 0.5, message, 'HorizontalAlignment', 'center', 'Color', [0.55 0.55 0.55], 'FontSize', 9);
end

function time = stateStart(states, name)
time = NaN;
if isfield(states, name)
    values = states.(name);
    if ~isempty(values) && isfinite(values(1))
        time = values(1);
    end
end
end

function time = stateEnd(states, name)
time = NaN;
if isfield(states, name)
    values = states.(name);
    if ~isempty(values) && numel(values) >= 2 && isfinite(values(2))
        time = values(2);
    end
end
end

function rawTrial = normalizedRawTrial(rawTrial)
if isempty(rawTrial)
    return
end
if iscell(rawTrial)
    if isempty(rawTrial)
        rawTrial = [];
    else
        rawTrial = rawTrial{end};
    end
elseif isstruct(rawTrial) && numel(rawTrial) > 1
    rawTrial = rawTrial(end);
end
end

function rawTrial = completedRawTrial(trial)
global BpodSystem

rawTrial = [];
if trial < 1 || ~isfield(BpodSystem, 'Data') || ~isfield(BpodSystem.Data, 'RawEvents') || ~isfield(BpodSystem.Data.RawEvents, 'Trial')
    return
end
trials = BpodSystem.Data.RawEvents.Trial;
if iscell(trials)
    if numel(trials) >= trial
        rawTrial = trials{trial};
    end
elseif isstruct(trials)
    if numel(trials) >= trial
        rawTrial = trials(trial);
    end
end
rawTrial = normalizedRawTrial(rawTrial);
end

function value = usableStateStruct(states)
value = false;
if ~isstruct(states)
    return
end
names = fieldnames(states);
for i = 1:numel(names)
    times = states.(names{i});
    if ~isempty(times) && any(isfinite(times(:)))
        value = true;
        return
    end
end
end

function value = rawTrialAvailable(rawTrial)
value = isstruct(rawTrial) && isfield(rawTrial, 'States') && usableStateStruct(rawTrial.States);
end

function duration = trialDuration(rawTrial, trial)
global BpodSystem

duration = 1;
if isstruct(rawTrial) && isfield(rawTrial, 'States') && isstruct(rawTrial.States)
    duration = max(eps, maxStateEnd(rawTrial.States));
    return
end
if isfield(BpodSystem, 'Data') && isfield(BpodSystem.Data, 'TrialStartTimestamp') && isfield(BpodSystem.Data, 'TrialEndTimestamp') && numel(BpodSystem.Data.TrialEndTimestamp) >= trial
    duration = max(eps, BpodSystem.Data.TrialEndTimestamp(trial) - BpodSystem.Data.TrialStartTimestamp(trial));
end
end

function duration = maxStateEnd(states)
duration = 1;
names = fieldnames(states);
for i = 1:numel(names)
    values = states.(names{i});
    if ~isempty(values)
        finiteValues = values(isfinite(values));
        if ~isempty(finiteValues)
            duration = max(duration, max(finiteValues(:)));
        end
    end
end
end

function [firstTrial, lastTrial] = visibleWindow(total, count)
current = min(count + 1, max(1, total));
firstTrial = max(1, current - 99);
lastTrial = min(total, firstTrial + 99);
firstTrial = max(1, lastTrial - 99);
end

function ticks = trialTicks(firstTrial, lastTrial)
step = 10;
ticks = ceil(firstTrial / step) * step:step:lastTrial;
if isempty(ticks) || ticks(1) > firstTrial
    ticks = [firstTrial ticks];
end
if ticks(end) < lastTrial
    ticks = [ticks lastTrial];
end
ticks = unique(ticks);
end

function setTimeAxis(ax, limits)
ticks = easyTimeTicks(limits);
xticks(ax, ticks);
set(ax, 'XMinorTick', 'on');
end

function ticks = easyTimeTicks(limits)
span = limits(2) - limits(1);
if ~all(isfinite(limits)) || span <= 0
    ticks = limits(1);
    return
end
scale = 10 ^ floor(log10(span));
steps = unique([0.1 0.2 0.25 0.5 1 2 2.5 5] * scale);
bestScore = inf;
ticks = limits;
for i = 1:numel(steps)
    candidate = ceil(limits(1) / steps(i)) * steps(i):steps(i):floor(limits(2) / steps(i)) * steps(i);
    if numel(candidate) < 3 || numel(candidate) > 7
        continue
    end
    score = abs(numel(candidate) - 5);
    if score < bestScore
        bestScore = score;
        ticks = candidate;
    end
end
ticks(abs(ticks) < eps(max(1, max(abs(ticks))))) = 0;
end
