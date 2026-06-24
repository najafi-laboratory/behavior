function ProtocolPlot(action, trialTypes, blockTypes, probeTypes, optoTypes, isiValues, completedCount, S)
global BpodSystem

switch action
    case 'init'
        screenSize = get(0, 'ScreenSize');
        width = round(screenSize(3) * 0.92);
        height = round(screenSize(4) * 0.92);
        position = [screenSize(1) + 16 screenSize(2) + screenSize(4) - height - 42 width height];
        figureName = [subjectName ' block single interval discrimination ' char(datetime('today', 'Format', 'yyyyMMdd')) ' If experiments go shit say I LOVE YICONG FOREVER!'];
        BpodSystem.ProtocolFigures.Session = figure('Name', figureName, 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none', 'Color', 'w', 'Position', position);
        layout = tiledlayout(BpodSystem.ProtocolFigures.Session, 30, 18, 'TileSpacing', 'compact', 'Padding', 'compact');
        BpodSystem.GUIHandles.TrialTypeAxes = nexttile(layout, 1, [4 11]);
        BpodSystem.GUIHandles.ISIAxes = nexttile(layout, 73, [6 11]);
        BpodSystem.GUIHandles.BlockTypeAxes = nexttile(layout, 181, [4 11]);
        BpodSystem.GUIHandles.ProbeTypeAxes = nexttile(layout, 253, [4 11]);
        BpodSystem.GUIHandles.OptoTypeAxes = nexttile(layout, 325, [4 11]);
        BpodSystem.GUIHandles.OutcomeAxes = nexttile(layout, 12, [5 5]);
        BpodSystem.GUIHandles.BlockOutcomeAxes = nexttile(layout, 102, [8 5]);
        BpodSystem.GUIHandles.OutcomeLegendAxes = nexttile(layout, 17, [13 2]);
        BpodSystem.GUIHandles.ShortLickAxes = nexttile(layout, 397, [8 4]);
        BpodSystem.GUIHandles.LongLickAxes = nexttile(layout, 401, [8 4]);
        BpodSystem.GUIHandles.ReactionTimeAxes = nexttile(layout, 405, [8 3]);
        BpodSystem.GUIHandles.StateAxes = nexttile(layout, 246, [11 7]);
        BpodSystem.GUIHandles.EventAxes = nexttile(layout, 444, [6 7]);
        closeOutcomeLegendFigure;
        initializeAxes;
        drawOutcomeLegend(BpodSystem.GUIHandles.OutcomeLegendAxes);
        updatePlots(trialTypes, blockTypes, probeTypes, optoTypes, isiValues, completedCount, S);
    case 'update'
        if ~isfield(BpodSystem.ProtocolFigures, 'Session') || ~isgraphics(BpodSystem.ProtocolFigures.Session)
            ProtocolPlot('init', trialTypes, blockTypes, probeTypes, optoTypes, isiValues, completedCount, S);
            return
        end
        updatePlots(trialTypes, blockTypes, probeTypes, optoTypes, isiValues, completedCount, S);
end
end

function initializeAxes
global BpodSystem
handles = [BpodSystem.GUIHandles.TrialTypeAxes BpodSystem.GUIHandles.BlockTypeAxes BpodSystem.GUIHandles.ProbeTypeAxes BpodSystem.GUIHandles.ISIAxes BpodSystem.GUIHandles.OptoTypeAxes BpodSystem.GUIHandles.OutcomeAxes BpodSystem.GUIHandles.BlockOutcomeAxes BpodSystem.GUIHandles.OutcomeLegendAxes BpodSystem.GUIHandles.ShortLickAxes BpodSystem.GUIHandles.LongLickAxes BpodSystem.GUIHandles.ReactionTimeAxes BpodSystem.GUIHandles.StateAxes BpodSystem.GUIHandles.EventAxes];
handles = handles(isgraphics(handles));
set(handles, 'Box', 'off', 'TickDir', 'out', 'FontSize', 10, 'LineWidth', 0.9, 'Color', 'none', 'XMinorTick', 'on', 'YMinorTick', 'on');
end

function closeOutcomeLegendFigure
global BpodSystem
if isfield(BpodSystem.ProtocolFigures, 'OutcomeLegend') && isgraphics(BpodSystem.ProtocolFigures.OutcomeLegend)
    close(BpodSystem.ProtocolFigures.OutcomeLegend);
end
if isfield(BpodSystem.ProtocolFigures, 'OutcomeLegend')
    BpodSystem.ProtocolFigures = rmfield(BpodSystem.ProtocolFigures, 'OutcomeLegend');
end
end

function updatePlots(trialTypes, blockTypes, probeTypes, optoTypes, isiValues, completedCount, ~)
updateTrialTypes(trialTypes, completedCount);
updateBlockTypes(blockTypes, completedCount);
updateProbeTypes(probeTypes, completedCount);
updateISI(trialTypes, isiValues, completedCount);
updateOptoTypes(optoTypes, completedCount);
updateOutcomeSummary(completedCount);
updateBlockOutcomeSummary(completedCount);
updateLickTraces(completedCount);
updateReactionTime(completedCount);
updateDetailPlots(completedCount);
drawnow limitrate
end

function updateTrialTypes(trialTypes, completedCount)
global BpodSystem
ax = BpodSystem.GUIHandles.TrialTypeAxes;
drawTrialTypeOutcome(ax, trialTypes, completedCount);
title(ax, sprintf('Trials  %d / %d', completedCount, numel(trialTypes)), 'FontSize', 11, 'FontWeight', 'normal');
end

function updateBlockTypes(blockTypes, completedCount)
global BpodSystem
ax = BpodSystem.GUIHandles.BlockTypeAxes;
typeColors = trialTypeColors();
drawSchedule(ax, blockTypes, completedCount, 1:3, {'50/50', 'Short', 'Long'}, [neutralColor(); typeColors], 'Block type');
end

function updateProbeTypes(probeTypes, completedCount)
global BpodSystem
ax = BpodSystem.GUIHandles.ProbeTypeAxes;
drawSchedule(ax, probeTypes, completedCount, 0:2, {'Off', 'Stim only', 'Servo only'}, neutralScheduleColors(3), 'Probe type');
end

function updateOptoTypes(optoTypes, completedCount)
global BpodSystem
ax = BpodSystem.GUIHandles.OptoTypeAxes;
cla(ax);
hold(ax, 'on');
if isempty(optoTypes)
    return
end
if isvector(optoTypes)
    optoTypes = reshape(optoTypes, 1, []);
end
total = size(optoTypes, 2);
[firstTrial, lastTrial] = visibleWindow(total, completedCount);
visibleTrials = firstTrial:lastTrial;
assignedThrough = assignedOptoCount(optoTypes, completedCount);
future = visibleTrials(visibleTrials > assignedThrough);
assigned = visibleTrials(visibleTrials <= assignedThrough);
display = OptoControl('display');

plotOptoMarkers(ax, future, optoTypes, display, true);
plotOptoMarkers(ax, assigned, optoTypes, display, false);
nextTrial = completedCount + 1;
if nextTrial <= total
    rows = optoRows(optoTypes(:, nextTrial));
    plot(ax, repmat(nextTrial, size(rows)), rows, 'o', 'Color', 'k', 'MarkerFaceColor', 'w', 'LineWidth', 1.2, 'MarkerSize', 7);
end
xlim(ax, [firstTrial - 0.5 lastTrial + 0.5]);
ylim(ax, [0.5 4.5]);
xticks(ax, trialTicks(firstTrial, lastTrial));
yticks(ax, 1:4);
yticklabels(ax, display.Labels);
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 10, 'YMinorTick', 'off');
xlabel(ax, 'Trial');
ylabel(ax, 'Opto period');
title(ax, 'Opto periods', 'FontSize', 10, 'FontWeight', 'normal');
end

function assignedThrough = assignedOptoCount(optoTypes, completedCount)
global BpodSystem
assignedThrough = completedCount;
if isempty(optoTypes) || ~isfield(BpodSystem, 'Data')
    return
end
if isfield(BpodSystem.Data, 'AssignedOptoTrialCount')
    assignedThrough = max(assignedThrough, BpodSystem.Data.AssignedOptoTrialCount);
end
assignedThrough = min(assignedThrough, size(optoTypes, 2));
end

function plotOptoMarkers(ax, trials, optoTypes, display, isFuture)
if isempty(trials)
    return
end
for trial = trials
    rows = optoRows(optoTypes(:, trial));
    for i = 1:numel(rows)
        row = rows(i);
        color = display.Colors(row, :);
        if isFuture
            if row == 1
                futureMarkerColor = [0.55 0.55 0.55];
            else
                futureMarkerColor = color * 0.45 + [0.45 0.45 0.45];
            end
            plot(ax, trial, row, '.', 'Color', futureMarkerColor, 'MarkerSize', 7);
        else
            plot(ax, trial, row, 's', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1, 'MarkerSize', 5.5);
        end
    end
end
end

function rows = optoRows(optoType)
optoType = optoType(:);
if any(isnan(optoType))
    rows = [];
    return
end
enabled = find(optoType ~= 0);
if isempty(enabled)
    rows = 1;
else
    rows = enabled + 1;
end
end

function drawTrialTypeOutcome(ax, trialTypes, completedCount)
global BpodSystem
cla(ax);
hold(ax, 'on');
total = numel(trialTypes);
[firstTrial, lastTrial] = visibleWindow(total, completedCount);
visibleTrials = firstTrial:lastTrial;
future = visibleTrials(visibleTrials > completedCount);
completed = visibleTrials(visibleTrials <= completedCount);
plot(ax, future, trialTypes(future), '.', 'Color', futureColor(), 'MarkerSize', 8);

outcomes = zeros(1, total);
if isfield(BpodSystem.Data, 'Outcomes')
    n = min(numel(BpodSystem.Data.Outcomes), total);
    outcomes(1:n) = BpodSystem.Data.Outcomes(1:n);
end
colors = outcomeColors();
drawOutcomeTrials(ax, completed, trialTypes, outcomes, 1, colors(1, :), 'o');
drawOutcomeTrials(ax, completed, trialTypes, outcomes, 2, colors(2, :), 'o');
drawOutcomeTrials(ax, completed, trialTypes, outcomes, 3, colors(3, :), 'x');
drawOutcomeTrials(ax, completed, trialTypes, outcomes, 4, colors(4, :), 'd');

nextTrial = completedCount + 1;
if nextTrial <= total
    plot(ax, nextTrial, trialTypes(nextTrial), 'o', 'Color', 'k', 'MarkerFaceColor', 'w', 'LineWidth', 1.2, 'MarkerSize', 7);
end
xlim(ax, [firstTrial - 0.5 lastTrial + 0.5]);
ylim(ax, [0.5 2.5]);
xticks(ax, trialTicks(firstTrial, lastTrial));
yticks(ax, 1:2);
yticklabels(ax, {'Short', 'Long'});
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 10, 'YMinorTick', 'off');
xlabel(ax, 'Trial');
ylabel(ax, 'Trial type');
end

function drawOutcomeTrials(ax, completed, trialTypes, outcomes, outcome, color, marker)
selected = completed(outcomes(completed) == outcome);
if isempty(selected)
    return
end
if strcmp(marker, 'x')
    plot(ax, selected, trialTypes(selected), marker, 'Color', color, 'LineWidth', 1.2, 'MarkerSize', 6);
else
    plot(ax, selected, trialTypes(selected), marker, 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1, 'MarkerSize', 5.5);
end
end

function updateISI(trialTypes, isiValues, completedCount)
global BpodSystem
ax = BpodSystem.GUIHandles.ISIAxes;
cla(ax);
hold(ax, 'on');
total = numel(isiValues);
[firstTrial, lastTrial] = visibleWindow(total, completedCount);
visibleTrials = firstTrial:lastTrial;
completed = visibleTrials(visibleTrials <= completedCount);
completed = completed(isfinite(isiValues(completed)));
shortTrials = completed(trialTypes(completed) == 1);
longTrials = completed(trialTypes(completed) == 2);
colors = trialTypeColors();
plot(ax, shortTrials, isiValues(shortTrials), 'o', 'Color', colors(1, :), 'MarkerFaceColor', colors(1, :), 'LineWidth', 1, 'MarkerSize', 5.5);
plot(ax, longTrials, isiValues(longTrials), 'o', 'Color', colors(2, :), 'MarkerFaceColor', colors(2, :), 'LineWidth', 1, 'MarkerSize', 5.5);
nextTrial = completedCount + 1;
if nextTrial <= total && isfinite(isiValues(nextTrial))
    plot(ax, nextTrial, isiValues(nextTrial), 'o', 'Color', 'k', 'MarkerFaceColor', 'w', 'LineWidth', 1.2, 'MarkerSize', 7);
end
xlim(ax, [firstTrial - 0.5 lastTrial + 0.5]);
ylim(ax, [0 3]);
xticks(ax, trialTicks(firstTrial, lastTrial));
yticks(ax, 0:0.5:3);
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 10, 'YMinorTick', 'off', 'YGrid', 'on');
xlabel(ax, 'Trial');
ylabel(ax, 'ISI (s)');
title(ax, 'Trial ISI', 'FontSize', 10, 'FontWeight', 'normal');
end

function updateOutcomeSummary(completedCount)
global BpodSystem
ax = BpodSystem.GUIHandles.OutcomeAxes;
cla(ax);
hold(ax, 'on');
data = outcomePercentages(completedCount);
colors = outcomeColors();

leftEdge = zeros(2, 1);
for outcome = 1:4
    for trialType = 1:2
        if data(trialType, outcome) <= 0
            continue
        end
        rectangle('Parent', ax, 'Position', [leftEdge(trialType), trialType - 0.27, data(trialType, outcome), 0.54], 'FaceColor', colors(outcome, :), 'EdgeColor', 'none');
    end
    leftEdge = leftEdge + data(:, outcome);
end
xlim(ax, [0 100]);
ylim(ax, [0.4 2.6]);
yticks(ax, 1:2);
yticklabels(ax, {'Short', 'Long'});
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 9, 'YMinorTick', 'off', 'XGrid', 'on');
xlabel(ax, 'Percent');
title(ax, 'Outcome by trial type', 'FontSize', 10, 'FontWeight', 'normal');

if isfield(BpodSystem.GUIHandles, 'OutcomeLegendAxes') && isgraphics(BpodSystem.GUIHandles.OutcomeLegendAxes)
    drawOutcomeLegend(BpodSystem.GUIHandles.OutcomeLegendAxes);
end
end

function updateBlockOutcomeSummary(completedCount)
global BpodSystem
ax = BpodSystem.GUIHandles.BlockOutcomeAxes;
cla(ax);
hold(ax, 'on');
data = blockOutcomePercentages(completedCount);
colors = outcomeColors();
offsets = [0.17 -0.17];
barHeight = 0.24;
for blockType = 1:3
    for trialType = 1:2
        y = blockType + offsets(trialType);
        leftEdge = 0;
        for outcome = 1:4
            width = data(blockType, trialType, outcome);
            if width > 0
                rectangle('Parent', ax, 'Position', [leftEdge, y - barHeight / 2, width, barHeight], 'FaceColor', colors(outcome, :), 'EdgeColor', 'none');
                leftEdge = leftEdge + width;
            end
        end
    end
end
xlim(ax, [0 100]);
ylim(ax, [0.45 3.55]);
yticks(ax, 1:3);
yticklabels(ax, {'50/50 block', 'Short block', 'Long block'});
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 9, 'YMinorTick', 'off', 'XGrid', 'on');
xlabel(ax, 'Percent');
ylabel(ax, 'Block type');
title(ax, 'Outcome by block and trial type', 'FontSize', 10, 'FontWeight', 'normal');
text(ax, 101, 3 + offsets(1), 'Short', 'FontSize', 8, 'VerticalAlignment', 'middle', 'Clipping', 'off');
text(ax, 101, 3 + offsets(2), 'Long', 'FontSize', 8, 'VerticalAlignment', 'middle', 'Clipping', 'off');
end

function drawOutcomeLegend(ax)
colors = outcomeColors();
cla(ax);
hold(ax, 'on');
axis(ax, [0 1 0 1]);
axis(ax, 'off');
labels = {'Reward', 'Wrong', 'No choice', 'Change mind'};
for i = 1:4
    y = 0.84 - (i - 1) * 0.18;
    if i == 3
        plot(ax, 0.15, y, 'x', 'Color', colors(i, :), 'MarkerSize', 7, 'LineWidth', 1.2);
    elseif i == 4
        plot(ax, 0.15, y, 'd', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), 'MarkerSize', 6);
    else
        plot(ax, 0.15, y, 'o', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), 'MarkerSize', 6);
    end
    text(ax, 0.28, y, labels{i}, 'FontSize', 9, 'VerticalAlignment', 'middle');
end
end

function updateLickTraces(completedCount)
global BpodSystem
shortAx = BpodSystem.GUIHandles.ShortLickAxes;
longAx = BpodSystem.GUIHandles.LongLickAxes;
binWidth = 0.05;
edges = -1:binWidth:5;
if numel(edges) < 2
    edges = [-1 5];
end
centers = edges(1:end - 1) + diff(edges) / 2;
shortData = collectChoiceLicks(completedCount, 1, edges);
longData = collectChoiceLicks(completedCount, 2, edges);
drawLickDensity(shortAx, centers, shortData, 'Short trials');
drawLickDensity(longAx, centers, longData, 'Long trials');
end

function data = collectChoiceLicks(completedCount, trialType, edges)
global BpodSystem
data.Left = zeros(1, numel(edges) - 1);
data.Right = zeros(1, numel(edges) - 1);
data.Trials = 0;
if completedCount < 1 || ~isfield(BpodSystem.Data, 'RawEvents') || ~isfield(BpodSystem.Data, 'TrialTypes')
    return
end
n = min([completedCount, numel(BpodSystem.Data.RawEvents.Trial), numel(BpodSystem.Data.TrialTypes)]);
for trial = 1:n
    if BpodSystem.Data.TrialTypes(trial) ~= trialType || isempty(BpodSystem.Data.RawEvents.Trial{trial})
        continue
    end
    rawTrial = BpodSystem.Data.RawEvents.Trial{trial};
    if ~isfield(rawTrial, 'States') || ~isfield(rawTrial.States, 'ChoiceWindow')
        continue
    end
    choiceTimes = rawTrial.States.ChoiceWindow;
    if isempty(choiceTimes) || all(isnan(choiceTimes(:)))
        continue
    end
    startTime = choiceTimes(find(isfinite(choiceTimes(:, 1)), 1), 1);
    if isempty(startTime) || ~isfinite(startTime) || ~isfield(rawTrial, 'Events')
        continue
    end
    data.Trials = data.Trials + 1;
    data.Left = data.Left + histcounts(eventTimes(rawTrial.Events, 'Port1In') - startTime, edges);
    data.Right = data.Right + histcounts(eventTimes(rawTrial.Events, 'Port3In') - startTime, edges);
end
end

function drawLickDensity(ax, centers, data, plotTitle)
cla(ax);
hold(ax, 'on');
colors = choiceColors();
leftColor = colors(1, :);
rightColor = colors(2, :);
if data.Trials > 0
    binWidth = max(eps, median(diff([centers centers(end) + (centers(end) - centers(max(1, end - 1)))])));
    leftRate = smoothCounts(data.Left) / data.Trials / binWidth;
    rightRate = smoothCounts(data.Right) / data.Trials / binWidth;
    plot(ax, centers, leftRate, '-', 'Color', leftColor, 'LineWidth', 1.5);
    plot(ax, centers, rightRate, '-', 'Color', rightColor, 'LineWidth', 1.5);
    yMax = max([leftRate rightRate 1]);
else
    yMax = 1;
    text(ax, 0.5, 0.5, 'No choice trials', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5], 'FontSize', 8);
end
xlim(ax, [-1 5]);
ylim(ax, [0 yMax * 1.15]);
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 9, 'YMinorTick', 'off', 'XGrid', 'off', 'YGrid', 'off');
xlabel(ax, 'Time from choice (s)');
ylabel(ax, 'Lick/s');
title(ax, plotTitle, 'FontSize', 10, 'FontWeight', 'normal');
end

function drawLickLegend(ax)
cla(ax);
hold(ax, 'on');
axis(ax, [0 1 0 1]);
axis(ax, 'off');
colors = choiceColors();
plot(ax, [0.08 0.42], [0.68 0.68], '-', 'Color', colors(1, :), 'LineWidth', 1.5);
plot(ax, [0.08 0.42], [0.48 0.48], '-', 'Color', colors(2, :), 'LineWidth', 1.5);
text(ax, 0.55, 0.68, 'Left', 'FontSize', 8, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left');
text(ax, 0.55, 0.48, 'Right', 'FontSize', 8, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left');
end

function updateReactionTime(completedCount)
global BpodSystem
ax = BpodSystem.GUIHandles.ReactionTimeAxes;
rt = reactionTimesByType(completedCount);
cla(ax);
hold(ax, 'on');
colors = trialTypeColors();
for trialType = 1:2
    values = rt{trialType};
    values = values(isfinite(values));
    if isempty(values)
        continue
    end
    jitter = 0.08 * sin((1:numel(values)) * 12.9898);
    scatterColor = colors(trialType, :) * 0.35 + [0.65 0.65 0.65];
    scatter(ax, trialType + jitter, values, 14, scatterColor, 'filled');
    drawReactionBox(ax, trialType, values, colors(trialType, :));
end
allValues = [rt{1} rt{2}];
allValues = allValues(isfinite(allValues));
if isempty(allValues)
    ylim(ax, [0 1]);
    text(ax, 0.5, 0.5, 'No choice licks', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5], 'FontSize', 8);
else
    yMax = max(allValues);
    ylim(ax, [0 max(0.5, yMax * 1.15)]);
end
xlim(ax, [0.5 2.5]);
xticks(ax, 1:2);
xticklabels(ax, {'Short', 'Long'});
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 9, 'YMinorTick', 'off', 'XGrid', 'off', 'YGrid', 'off');
xlabel(ax, 'Trial type');
ylabel(ax, 'Reaction time (s)');
title(ax, 'Reaction time', 'FontSize', 10, 'FontWeight', 'normal');
end

function rt = reactionTimesByType(completedCount)
global BpodSystem
rt = {[], []};
if completedCount < 1 || ~isfield(BpodSystem.Data, 'RawEvents') || ~isfield(BpodSystem.Data, 'TrialTypes')
    return
end
n = min([completedCount, numel(BpodSystem.Data.RawEvents.Trial), numel(BpodSystem.Data.TrialTypes)]);
for trial = 1:n
    trialType = BpodSystem.Data.TrialTypes(trial);
    if trialType < 1 || trialType > 2 || isempty(BpodSystem.Data.RawEvents.Trial{trial})
        continue
    end
    rawTrial = BpodSystem.Data.RawEvents.Trial{trial};
    reactionTime = firstChoiceLickTime(rawTrial);
    if isfinite(reactionTime)
        rt{trialType}(end + 1) = reactionTime;
    end
end
end

function reactionTime = firstChoiceLickTime(rawTrial)
reactionTime = nan;
if ~isfield(rawTrial, 'States') || ~isfield(rawTrial.States, 'ChoiceWindow') || ~isfield(rawTrial, 'Events')
    return
end
choiceTimes = rawTrial.States.ChoiceWindow;
if isempty(choiceTimes) || all(isnan(choiceTimes(:)))
    return
end
startTime = choiceTimes(find(isfinite(choiceTimes(:, 1)), 1), 1);
if isempty(startTime) || ~isfinite(startTime)
    return
end
lickTimes = [reshape(eventTimes(rawTrial.Events, 'Port1In'), 1, []) reshape(eventTimes(rawTrial.Events, 'Port2In'), 1, []) reshape(eventTimes(rawTrial.Events, 'Port3In'), 1, [])];
lickTimes = sort(lickTimes(isfinite(lickTimes) & lickTimes >= startTime));
if ~isempty(lickTimes)
    reactionTime = lickTimes(1) - startTime;
end
end

function drawReactionBox(ax, x, values, color)
values = sort(values(:));
q1 = percentileValue(values, 25);
q2 = percentileValue(values, 50);
q3 = percentileValue(values, 75);
low = min(values);
high = max(values);
boxWidth = 0.34;
rectangle('Parent', ax, 'Position', [x - boxWidth / 2, q1, boxWidth, max(eps, q3 - q1)], 'FaceColor', [1 1 1], 'EdgeColor', color, 'LineWidth', 1.2);
plot(ax, [x - boxWidth / 2 x + boxWidth / 2], [q2 q2], '-', 'Color', color, 'LineWidth', 1.4);
plot(ax, [x x], [low q1], '-', 'Color', color, 'LineWidth', 1);
plot(ax, [x x], [q3 high], '-', 'Color', color, 'LineWidth', 1);
plot(ax, [x - boxWidth / 4 x + boxWidth / 4], [low low], '-', 'Color', color, 'LineWidth', 1);
plot(ax, [x - boxWidth / 4 x + boxWidth / 4], [high high], '-', 'Color', color, 'LineWidth', 1);
end

function value = percentileValue(values, percent)
if isempty(values)
    value = nan;
    return
end
if numel(values) == 1
    value = values(1);
    return
end
position = 1 + (numel(values) - 1) * percent / 100;
lower = floor(position);
upper = ceil(position);
if lower == upper
    value = values(lower);
else
    value = values(lower) + (position - lower) * (values(upper) - values(lower));
end
end

function smoothed = smoothCounts(counts)
kernel = [0.0545 0.2442 0.4026 0.2442 0.0545];
smoothed = conv(counts, kernel, 'same');
end

function data = outcomePercentages(completedCount)
global BpodSystem
data = zeros(2, 4);
if completedCount < 1 || ~isfield(BpodSystem.Data, 'Outcomes') || ~isfield(BpodSystem.Data, 'TrialTypes')
    return
end
n = min([completedCount, numel(BpodSystem.Data.Outcomes), numel(BpodSystem.Data.TrialTypes)]);
outcomes = BpodSystem.Data.Outcomes(1:n);
types = BpodSystem.Data.TrialTypes(1:n);
for trialType = 1:2
    selected = outcomes(types == trialType & outcomes > 0);
    if isempty(selected)
        continue
    end
    for outcome = 1:4
        data(trialType, outcome) = 100 * sum(selected == outcome) / numel(selected);
    end
end
end

function data = blockOutcomePercentages(completedCount)
global BpodSystem
data = zeros(3, 2, 4);
if completedCount < 1 || ~isfield(BpodSystem.Data, 'Outcomes') || ~isfield(BpodSystem.Data, 'TrialTypes') || ~isfield(BpodSystem.Data, 'BlockTypes')
    return
end
n = min([completedCount, numel(BpodSystem.Data.Outcomes), numel(BpodSystem.Data.TrialTypes), numel(BpodSystem.Data.BlockTypes)]);
outcomes = BpodSystem.Data.Outcomes(1:n);
trialTypes = BpodSystem.Data.TrialTypes(1:n);
blockTypes = BpodSystem.Data.BlockTypes(1:n);
for blockType = 1:3
    for trialType = 1:2
        selected = outcomes(blockTypes == blockType & trialTypes == trialType & outcomes > 0);
        if isempty(selected)
            continue
        end
        for outcome = 1:4
            data(blockType, trialType, outcome) = 100 * sum(selected == outcome) / numel(selected);
        end
    end
end
end

function colors = outcomeColors
colors = [0.15 0.6 0.25; 0.85 0.15 0.1; 0.05 0.05 0.05; 0.9 0.55 0.05];
end

function colors = choiceColors
colors = [0.08 0.32 0.78; 0.9 0.38 0.05];
end

function colors = trialTypeColors
colors = [0.48 0.28 0.74; 0.0 0.55 0.55];
end

function color = neutralColor
color = [0.42 0.42 0.42];
end

function colors = neutralScheduleColors(count)
base = [0.70 0.70 0.70; 0.48 0.52 0.58; 0.28 0.33 0.38];
colors = base(1:count, :);
end

function color = futureColor
color = [0.65 0.70 0.76];
end

function drawSchedule(ax, values, completedCount, levels, labels, colors, yLabel)
cla(ax);
hold(ax, 'on');
total = numel(values);
[firstTrial, lastTrial] = visibleWindow(total, completedCount);
visibleTrials = firstTrial:lastTrial;
future = visibleTrials(visibleTrials > completedCount);
completed = visibleTrials(visibleTrials <= completedCount);

plot(ax, future, values(future), '.', 'Color', futureColor(), 'MarkerSize', 8);
for i = 1:numel(levels)
    selected = completed(values(completed) == levels(i));
    plot(ax, selected, repmat(levels(i), size(selected)), 'o', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), 'LineWidth', 1, 'MarkerSize', 5.5);
end
nextTrial = completedCount + 1;
if nextTrial <= total
    plot(ax, nextTrial, values(nextTrial), 'o', 'Color', 'k', 'MarkerFaceColor', 'w', 'LineWidth', 1.2, 'MarkerSize', 7);
end
xlim(ax, [firstTrial - 0.5 lastTrial + 0.5]);
ylim(ax, [min(levels) - 0.5 max(levels) + 0.5]);
xticks(ax, trialTicks(firstTrial, lastTrial));
yticks(ax, levels);
yticklabels(ax, labels);
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 10, 'YMinorTick', 'off');
xlabel(ax, 'Trial');
ylabel(ax, yLabel);
end

function updateDetailPlots(trial)
global BpodSystem
if trial < 1 || ~isfield(BpodSystem.Data, 'RawEvents') || numel(BpodSystem.Data.RawEvents.Trial) < trial || isempty(BpodSystem.Data.RawEvents.Trial{trial})
    showEmpty(BpodSystem.GUIHandles.StateAxes, 'State timing', 'No completed trial');
    showEvents(BpodSystem.GUIHandles.EventAxes, struct, 1, 1);
    return
end
rawTrial = BpodSystem.Data.RawEvents.Trial{trial};
updateStateTiming(rawTrial, trial);
showEvents(BpodSystem.GUIHandles.EventAxes, rawTrial, trial, trialDuration(trial));
end

function updateStateTiming(rawTrial, trial)
global BpodSystem
ax = BpodSystem.GUIHandles.StateAxes;
cla(ax);
hold(ax, 'on');
names = fieldnames(rawTrial.States);
entries = cell(0, 3);
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
    showEmpty(ax, 'State timing', 'No state timing');
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
duration = trialDuration(trial);
xlim(ax, [0 duration]);
ylim(ax, [0.5 numel(labels) + 0.5]);
yticks(ax, 1:numel(labels));
yticklabels(ax, labels);
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 9, 'YMinorTick', 'off', 'XGrid', 'on', 'YGrid', 'on');
xlabel(ax, 'Time (s)');
title(ax, sprintf('State timing  trial %d', trial), 'FontSize', 10, 'FontWeight', 'normal');
end

function showEvents(ax, rawTrial, trial, duration)
cla(ax);
hold(ax, 'on');
if isfield(rawTrial, 'Events')
    events = rawTrial.Events;
else
    events = struct;
end
rows = {'Port1 lick', 'Port2 lick', 'Port3 lick', 'BNC 1', 'BNC 2', 'LED 1'};
choices = choiceColors();
colors = [choices(1, :); 0.45 0.45 0.45; choices(2, :); 0.35 0.35 0.35; 0.15 0.15 0.15; 0.55 0.55 0.55];
drawIntervals(ax, eventIntervals(events, 'Port1In', 'Port1Out', duration, 0.03), 1, colors(1, :));
drawIntervals(ax, eventIntervals(events, 'Port2In', 'Port2Out', duration, 0.03), 2, colors(2, :));
drawIntervals(ax, eventIntervals(events, 'Port3In', 'Port3Out', duration, 0.03), 3, colors(3, :));
drawIntervals(ax, eventIntervals(events, 'BNC1High', 'BNC1Low', duration, 0), 4, colors(4, :));
drawIntervals(ax, eventIntervals(events, 'BNC2High', 'BNC2Low', duration, 0), 5, colors(5, :));
ledIntervals = optoIntervals(rawTrial, trial, duration);
if isempty(ledIntervals)
    ledIntervals = eventIntervals(events, 'PWM1High', 'PWM1Low', duration, 0);
end
drawIntervals(ax, ledIntervals, 6, colors(6, :));
xlim(ax, [0 duration]);
ylim(ax, [0.5 numel(rows) + 0.5]);
yticks(ax, 1:numel(rows));
yticklabels(ax, rows);
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 9, 'YMinorTick', 'off', 'XGrid', 'on');
xlabel(ax, 'Time (s)');
title(ax, sprintf('Events  trial %d', trial), 'FontSize', 10, 'FontWeight', 'normal');
end

function intervals = optoIntervals(rawTrial, trial, duration)
intervals = zeros(0, 2);
optoType = trialOptoType(trial);
if isempty(optoType) || any(isnan(optoType)) || ~any(optoType) || ~isfield(rawTrial, 'States')
    return
end
states = rawTrial.States;
if optoType(1)
    startTime = stateStart(states, 'AudStimTrigger');
    stopTime = firstFinite([stateEnd(states, 'SpoutIn') stateEnd(states, 'ProbeSpoutIn') stateEnd(states, 'AudStimTrigger')]);
    intervals = appendInterval(intervals, startTime, stopTime, duration);
end
if numel(optoType) >= 2 && optoType(2)
    intervals = appendInterval(intervals, stateStart(states, 'ChoiceWindow'), stateEnd(states, 'ChoiceWindow'), duration);
    intervals = appendInterval(intervals, stateStart(states, 'ProbeChoiceWindow'), stateEnd(states, 'ProbeChoiceWindow'), duration);
end
if numel(optoType) >= 3 && optoType(3)
    intervals = appendInterval(intervals, stateStart(states, 'PostRewardDelay'), stateEnd(states, 'PostRewardDelay'), duration);
end
end

function optoType = trialOptoType(trial)
global BpodSystem
optoType = [];
if isfield(BpodSystem.Data, 'OptoTrialTypes') && size(BpodSystem.Data.OptoTrialTypes, 2) >= trial
    optoType = BpodSystem.Data.OptoTrialTypes(:, trial);
elseif isfield(BpodSystem.Data, 'PlannedOptoTrialTypes') && size(BpodSystem.Data.PlannedOptoTrialTypes, 2) >= trial
    optoType = BpodSystem.Data.PlannedOptoTrialTypes(:, trial);
end
end

function intervals = appendInterval(intervals, startTime, stopTime, duration)
if ~isfinite(startTime) || ~isfinite(stopTime)
    return
end
startTime = max(0, min(duration, startTime));
stopTime = max(0, min(duration, stopTime));
if stopTime > startTime
    intervals(end + 1, :) = [startTime stopTime];
end
end

function value = stateStart(states, name)
value = nan;
interval = firstStateInterval(states, name);
if ~isempty(interval)
    value = interval(1);
end
end

function value = stateEnd(states, name)
value = nan;
interval = firstStateInterval(states, name);
if ~isempty(interval)
    value = interval(2);
end
end

function interval = firstStateInterval(states, name)
interval = [];
if ~isfield(states, name)
    return
end
times = states.(name);
if isempty(times) || all(isnan(times(:)))
    return
end
if isvector(times)
    times = reshape(times, 1, []);
end
if size(times, 2) < 2
    return
end
valid = find(isfinite(times(:, 1)) & isfinite(times(:, 2)), 1);
if ~isempty(valid)
    interval = times(valid, 1:2);
end
end

function value = firstFinite(values)
value = nan;
idx = find(isfinite(values), 1);
if ~isempty(idx)
    value = values(idx);
end
end

function intervals = eventIntervals(events, onName, offName, duration, fallbackWidth)
starts = eventTimes(events, onName);
stops = eventTimes(events, offName);
intervals = zeros(0, 2);
stopIndex = 1;
for i = 1:numel(starts)
    while stopIndex <= numel(stops) && stops(stopIndex) <= starts(i)
        stopIndex = stopIndex + 1;
    end
    if stopIndex <= numel(stops)
        stopTime = stops(stopIndex);
        stopIndex = stopIndex + 1;
    elseif fallbackWidth > 0
        stopTime = starts(i) + fallbackWidth;
    else
        stopTime = duration;
    end
    stopTime = min(duration, stopTime);
    if stopTime > starts(i)
        intervals(end + 1, :) = [max(0, starts(i)) stopTime];
    end
end
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
yticks(ax, []);
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 9);
xlabel(ax, 'Time (s)');
title(ax, plotTitle, 'FontSize', 10, 'FontWeight', 'normal');
text(ax, 0.5, 0.5, message, 'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5]);
end

function [firstTrial, lastTrial] = visibleWindow(total, completedCount)
current = min(completedCount + 1, max(1, total));
firstTrial = max(1, current - 89);
lastTrial = min(total, firstTrial + 89);
firstTrial = max(1, lastTrial - 89);
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

function name = subjectName
global BpodSystem
name = 'UnknownSubject';
if isfield(BpodSystem, 'Path') && isstruct(BpodSystem.Path)
    candidates = {'CurrentSubject', 'SubjectName', 'Subject'};
    for i = 1:numel(candidates)
        candidate = readField(BpodSystem.Path, candidates{i});
        if ~isempty(candidate)
            name = candidate;
            return
        end
    end
end
if isfield(BpodSystem, 'Data') && isstruct(BpodSystem.Data)
    candidate = readField(BpodSystem.Data, 'SubjectName');
    if ~isempty(candidate)
        name = candidate;
    end
end
end

function value = readField(source, fieldName)
value = '';
if ~isfield(source, fieldName)
    return
end
value = normalizeSubjectName(source.(fieldName));
end

function value = normalizeSubjectName(rawValue)
value = '';
if isempty(rawValue)
    return
end
if iscell(rawValue)
    rawValue = rawValue{1};
end
if isstring(rawValue)
    rawValue = char(rawValue(1));
end
if ~ischar(rawValue)
    return
end
value = strtrim(rawValue);
if isempty(value)
    return
end
value = strrep(value, '/', filesep);
if contains(value, filesep)
    parts = regexp(value, ['\' filesep '+'], 'split');
    parts = parts(~cellfun('isempty', parts));
    if ~isempty(parts)
        value = parts{end};
    end
    [~, baseName, extension] = fileparts(value);
    if ~isempty(extension)
        value = baseName;
    end
end
end

function duration = trialDuration(trial)
global BpodSystem
duration = 1;
if isfield(BpodSystem.Data, 'TrialStartTimestamp') && isfield(BpodSystem.Data, 'TrialEndTimestamp') && numel(BpodSystem.Data.TrialEndTimestamp) >= trial
    duration = max(eps, BpodSystem.Data.TrialEndTimestamp(trial) - BpodSystem.Data.TrialStartTimestamp(trial));
end
end
