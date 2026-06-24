function ProtocolPlot(action, trialTypes, optoTypes, probeTypes, completedCount, S)
global BpodSystem

switch action
    case 'init'
        % Build one compact canvas for all online session plots.
        screenSize = get(0, 'ScreenSize');
        width = round(screenSize(3) * 0.92);
        height = round(screenSize(4) * 0.92);
        position = [screenSize(1) + 16 screenSize(2) + screenSize(4) - height - 42 width height];
        figureName = [subjectName ' Joystick Session ' char(datetime('today', 'Format', 'yyyyMMdd')) '. If experiments go shit say I love Yicong Forever'];
        BpodSystem.ProtocolFigures.Session = figure('Name', figureName, 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none', 'Color', 'w', 'Position', position);

        leftX = 0.060;
        bottom = 0.065;
        columnGap = 0.100;
        plotWidth = 0.855;
        leftWidth = plotWidth * 0.6;
        rightX = leftX + leftWidth + columnGap;
        rightWidth = plotWidth - leftWidth - columnGap;
        rightHalfGap = 0.030;
        rightHalfWidth = (rightWidth - rightHalfGap) / 2;
        legendGap = 0.012;
        legendWidth = 0.065;
        outcomeWidth = rightWidth - legendWidth - legendGap;
        outcomeHalfWidth = (outcomeWidth - rightHalfGap) / 2;

        BpodSystem.GUIHandles.OutcomeAxes = sessionAxes([leftX 0.825 leftWidth 0.065]);
        BpodSystem.GUIHandles.OptoAxes = sessionAxes([leftX 0.690 leftWidth 0.065]);
        BpodSystem.GUIHandles.ProbeAxes = sessionAxes([leftX 0.555 leftWidth 0.065]);
        BpodSystem.GUIHandles.DelayAxes = sessionAxes([leftX 0.405 leftWidth 0.075]);
        BpodSystem.GUIHandles.EncoderAxes = sessionAxes([leftX bottom leftWidth 0.265]);

        BpodSystem.GUIHandles.OutcomeSummaryAxes = sessionAxes([rightX 0.800 outcomeHalfWidth 0.100]);
        BpodSystem.GUIHandles.FullOutcomeAxes = sessionAxes([rightX + outcomeHalfWidth + rightHalfGap 0.800 outcomeHalfWidth 0.100]);
        BpodSystem.GUIHandles.OutcomeLegendAxes = sessionAxes([rightX + outcomeWidth + legendGap 0.800 legendWidth 0.100]);
        BpodSystem.GUIHandles.ShortTimingAxes = sessionAxes([rightX 0.585 rightHalfWidth 0.140]);
        BpodSystem.GUIHandles.LongTimingAxes = sessionAxes([rightX + rightHalfWidth + rightHalfGap 0.585 rightHalfWidth 0.140]);
        BpodSystem.GUIHandles.StateAxes = sessionAxes([rightX 0.270 rightWidth 0.240]);
        BpodSystem.GUIHandles.EventAxes = sessionAxes([rightX bottom rightWidth 0.125]);

        initializeAxes;
        updateOutcome(trialTypes, completedCount);
        updateOutcomeSummary(completedCount);
        updateFullOutcomeSummary(completedCount);
        updateOutcomeLegend;
        updateOpto(optoTypes, completedCount);
        updateProbe(probeTypes, completedCount);
        updatePressTiming(1, completedCount);
        updatePressTiming(2, completedCount);
        updateDelay(trialTypes, completedCount, S);
        showEmpty(BpodSystem.GUIHandles.EncoderAxes, 'Rotary encoder', 'No encoder data');
        showEmpty(BpodSystem.GUIHandles.StateAxes, 'State timing', 'No completed trial');
        showNoEvents(BpodSystem.GUIHandles.EventAxes, 'BNC, LED, and lick events', 0, 1);
    case 'update'
        if ~isfield(BpodSystem.ProtocolFigures, 'Session') || ~isgraphics(BpodSystem.ProtocolFigures.Session)
            ProtocolPlot('init', trialTypes, optoTypes, probeTypes, completedCount, S);
            return
        end

        % Refresh summary plots before drawing the completed-trial detail.
        updateOutcome(trialTypes, completedCount);
        updateOutcomeSummary(completedCount);
        updateFullOutcomeSummary(completedCount);
        updateOutcomeLegend;
        updateOpto(optoTypes, completedCount);
        updateProbe(probeTypes, completedCount);
        updatePressTiming(1, completedCount);
        updatePressTiming(2, completedCount);
        updateDelay(trialTypes, completedCount, S);
        if ~trialAvailable(completedCount)
            showEmpty(BpodSystem.GUIHandles.EncoderAxes, 'Rotary encoder', 'No encoder data');
            showEmpty(BpodSystem.GUIHandles.StateAxes, 'State timing', 'No completed trial');
            showNoEvents(BpodSystem.GUIHandles.EventAxes, 'BNC, LED, and lick events', 0, 1);
            drawnow
            return
        end
        rawTrial = BpodSystem.Data.RawEvents.Trial{completedCount};
        completedSettings = S;
        if isfield(BpodSystem.Data, 'TrialSettings') && numel(BpodSystem.Data.TrialSettings) >= completedCount
            completedSettings = BpodSystem.Data.TrialSettings(completedCount);
        end
        updateStateTiming(rawTrial, completedCount);
        updateCombinedEvents(rawTrial, completedCount);
        if isfield(BpodSystem.Data, 'EncoderData') && numel(BpodSystem.Data.EncoderData) >= completedCount
            updateEncoder(BpodSystem.Data.EncoderData{completedCount}, rawTrial, completedCount, completedSettings);
        else
            showEmpty(BpodSystem.GUIHandles.EncoderAxes, 'Rotary encoder', 'No encoder data');
        end
        drawnow limitrate
end

    function ax = sessionAxes(position)
        ax = axes('Parent', BpodSystem.ProtocolFigures.Session, 'Units', 'normalized', 'Position', position);
    end

    function initializeAxes
        handles = [BpodSystem.GUIHandles.OutcomeAxes BpodSystem.GUIHandles.OutcomeSummaryAxes BpodSystem.GUIHandles.FullOutcomeAxes BpodSystem.GUIHandles.OutcomeLegendAxes BpodSystem.GUIHandles.OptoAxes BpodSystem.GUIHandles.ShortTimingAxes BpodSystem.GUIHandles.ProbeAxes BpodSystem.GUIHandles.LongTimingAxes BpodSystem.GUIHandles.DelayAxes BpodSystem.GUIHandles.EncoderAxes BpodSystem.GUIHandles.StateAxes BpodSystem.GUIHandles.EventAxes];
        set(handles, 'Box', 'off', 'TickDir', 'out', 'FontSize', 10, 'LineWidth', 0.9, 'Color', 'none', 'XMinorTick', 'on', 'YMinorTick', 'on');
    end

    function updateOutcome(types, count)
        % Plot trial type with the raw terminal outcome state as marker color.
        ax = BpodSystem.GUIHandles.OutcomeAxes;
        cla(ax);
        hold(ax, 'on');

        total = numel(types);
        outcomes = repmat({''}, 1, total);
        for trial = 1:min([count total availableTrialCount()])
            outcomes{trial} = trialOutcomeName(trial);
        end

        [firstTrial, lastTrial] = visibleWindow(total, count);
        visibleTrials = firstTrial:lastTrial;
        future = visibleTrials(visibleTrials > count);
        plot(ax, future, types(future), '.', 'Color', [0.45 0.65 0.8], 'MarkerSize', 7.2);

        names = {'LeverRetractFinal','EarlyPress2','Press2Late','DidNotPress2','DidNotPress1','AssistTrial'};
        colors = outcomeColors;
        markers = {'o','o','o','x','x','o'};
        for i = 1:numel(names)
            plotOutcome(ax, types, outcomes, visibleTrials, names{i}, colors(i, :), markers{i});
        end
        if count < total
            plot(ax, count + 1, types(count + 1), 'o', 'Color', 'k', 'MarkerFaceColor', 'w', 'LineWidth', 1.2, 'MarkerSize', 6.4);
        end
        limits = [firstTrial - 0.5 lastTrial + 0.5];
        xlim(ax, limits);
        xticks(ax, trialTicks(firstTrial, lastTrial));
        ylim(ax, [0.5 2.5]);
        yticks(ax, [1 2]);
        yticklabels(ax, {'Short', 'Long'});
        set(ax, 'YMinorTick', 'off');
        xlabel(ax, 'Trial');
        ylabel(ax, 'Trial type');
        title(ax, sprintf('Trials  %d / %d', count, total), 'FontSize', 10, 'FontWeight', 'normal');
    end

    function updateOpto(opto, count)
        ax = BpodSystem.GUIHandles.OptoAxes;
        cla(ax);
        hold(ax, 'on');
        if isvector(opto)
            opto = reshape(opto, 1, []);
        end
        total = size(opto, 2);
        [firstTrial, lastTrial] = visibleWindow(total, count);
        visibleTrials = firstTrial:lastTrial;
        display = OptoControl('display', S);
        assignedThrough = assignedOptoCount(opto, count);
        future = visibleTrials(visibleTrials > assignedThrough);
        assigned = visibleTrials(visibleTrials <= assignedThrough);
        plotOptoMarkers(ax, future, opto, display, true);
        plotOptoMarkers(ax, assigned, opto, display, false);
        if count < total
            rows = optoRows(opto(:, count + 1));
            plot(ax, repmat(count + 1, size(rows)), rows, 'o', 'Color', 'k', 'MarkerFaceColor', 'w', 'LineWidth', 1.2, 'MarkerSize', 7);
        end
        xlim(ax, [firstTrial - 0.5 lastTrial + 0.5]);
        xticks(ax, trialTicks(firstTrial, lastTrial));
        ylim(ax, [0.5 4.5]);
        yticks(ax, 1:4);
        yticklabels(ax, display.Labels);
        set(ax, 'YMinorTick', 'off');
        xlabel(ax, 'Trial');
        ylabel(ax, 'Opto period');
        title(ax, 'Opto periods', 'FontSize', 9, 'FontWeight', 'normal');
    end

    function assignedThrough = assignedOptoCount(opto, count)
        assignedThrough = count;
        if isfield(BpodSystem.Data, 'AssignedOptoTrialCount')
            assignedThrough = max(assignedThrough, BpodSystem.Data.AssignedOptoTrialCount);
        end
        assignedThrough = min(assignedThrough, size(opto, 2));
    end

    function plotOptoMarkers(ax, trials, opto, display, isFuture)
        for trial = trials
            rows = optoRows(opto(:, trial));
            for i = 1:numel(rows)
                row = rows(i);
                color = display.Colors(row, :);
                if isFuture
                    plot(ax, trial, row, '.', 'Color', color * 0.45 + [0.45 0.45 0.45], 'MarkerSize', 7);
                else
                    plot(ax, trial, row, 's', 'Color', color, 'MarkerFaceColor', color, 'MarkerSize', 5.8);
                end
            end
        end
    end

    function rows = optoRows(optoType)
        optoType = optoType(:);
        enabled = find(optoType ~= 0);
        if isempty(enabled)
            rows = 1;
        else
            rows = enabled + 1;
        end
    end

    function updateProbe(probe, count)
        ax = BpodSystem.GUIHandles.ProbeAxes;
        cla(ax);
        hold(ax, 'on');
        total = numel(probe);
        [firstTrial, lastTrial] = visibleWindow(total, count);
        visibleTrials = firstTrial:lastTrial;
        display = ProbeControl('display', S);
        for probeType = 0:2
            trials = visibleTrials(probe(visibleTrials) == probeType);
            color = display.Colors(probeType + 1, :);
            plot(ax, trials, repmat(probeType, size(trials)), 'd', 'Color', color, 'MarkerFaceColor', color, 'MarkerSize', 5.8);
        end
        if count < total
            plot(ax, count + 1, probe(count + 1), 'd', 'Color', 'k', 'MarkerFaceColor', 'w', 'LineWidth', 1.2, 'MarkerSize', 7);
        end
        xlim(ax, [firstTrial - 0.5 lastTrial + 0.5]);
        xticks(ax, trialTicks(firstTrial, lastTrial));
        ylim(ax, [-0.5 2.5]);
        yticks(ax, 0:2);
        yticklabels(ax, display.Labels);
        set(ax, 'YMinorTick', 'off');
        xlabel(ax, 'Trial');
        ylabel(ax, 'Probe type');
        title(ax, 'Probe trial type', 'FontSize', 9, 'FontWeight', 'normal');
    end

    function plotOutcome(ax, rows, outcomes, visibleTrials, value, color, marker)
        trials = visibleTrials(strcmp(outcomes(visibleTrials), value));
        if strcmp(marker, 'o')
            plot(ax, trials, rows(trials), marker, 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.1, 'MarkerSize', 5.8);
        else
            plot(ax, trials, rows(trials), marker, 'Color', color, 'LineWidth', 1.2, 'MarkerSize', 6.1);
        end
    end

    function updateOutcomeSummary(count)
        % Summarize completed trials by outcome fraction for short and long.
        ax = BpodSystem.GUIHandles.OutcomeSummaryAxes;
        cla(ax);
        hold(ax, 'on');

        colors = completedOutcomeColors;
        values = zeros(2, 4);
        available = min([count availableTrialCount() completedTrialTypeCount()]);
        for trial = 1:available
            category = completedOutcomeCategory(trial);
            if category > 0
                values(BpodSystem.Data.TrialTypes(trial), category) = values(BpodSystem.Data.TrialTypes(trial), category) + 1;
            end
        end
        totals = sum(values, 2);
        fractions = zeros(size(values));
        for trialType = 1:2
            if totals(trialType) > 0
                fractions(trialType, :) = values(trialType, :) / totals(trialType);
            end
        end
        bars = barh(ax, 1:2, fractions, 0.58, 'stacked', 'EdgeColor', 'none');
        for i = 1:size(values, 2)
            bars(i).FaceColor = colors(i, :);
        end
        for trialType = 1:2
            position = 0;
            for category = 1:size(values, 2)
                fraction = fractions(trialType, category);
                if fraction >= 0.12
                    text(ax, position + fraction / 2, trialType, sprintf('%.0f%%', 100 * fraction), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 7);
                end
                position = position + fraction;
            end
            if totals(trialType) == 0
                text(ax, 0.5, trialType, 'No outcomes', 'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5], 'FontSize', 8);
            else
                text(ax, 1.02, trialType, sprintf('n=%d', totals(trialType)), 'HorizontalAlignment', 'left', 'FontSize', 7);
            end
        end
        xlim(ax, [0 1.13]);
        ylim(ax, [0.5 2.5]);
        xticks(ax, [0 0.5 1]);
        xticklabels(ax, {'0', '50', '100%'});
        yticks(ax, [1 2]);
        yticklabels(ax, {'Short', 'Long'});
        set(ax, 'XMinorTick', 'off', 'YMinorTick', 'off');
        title(ax, 'Outcome fraction, completed trials', 'FontSize', 9, 'FontWeight', 'normal');
    end

    function updateFullOutcomeSummary(count)
        % Summarize all terminal outcomes by fraction for short and long.
        ax = BpodSystem.GUIHandles.FullOutcomeAxes;
        cla(ax);
        hold(ax, 'on');

        colors = outcomeColors;
        values = zeros(2, 6);
        available = min([count availableTrialCount() completedTrialTypeCount()]);
        for trial = 1:available
            category = allOutcomeCategory(trial);
            if category > 0
                values(BpodSystem.Data.TrialTypes(trial), category) = values(BpodSystem.Data.TrialTypes(trial), category) + 1;
            end
        end

        totals = sum(values, 2);
        fractions = zeros(size(values));
        for trialType = 1:2
            if totals(trialType) > 0
                fractions(trialType, :) = values(trialType, :) / totals(trialType);
            end
        end
        bars = barh(ax, 1:2, fractions, 0.58, 'stacked', 'EdgeColor', 'none');
        for i = 1:size(values, 2)
            bars(i).FaceColor = colors(i, :);
        end
        for trialType = 1:2
            position = 0;
            for category = 1:size(values, 2)
                fraction = fractions(trialType, category);
                if fraction >= 0.12
                    text(ax, position + fraction / 2, trialType, sprintf('%.0f%%', 100 * fraction), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 7);
                end
                position = position + fraction;
            end
            if totals(trialType) == 0
                text(ax, 0.5, trialType, 'No outcomes', 'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5], 'FontSize', 8);
            else
                text(ax, 1.02, trialType, sprintf('n=%d', totals(trialType)), 'HorizontalAlignment', 'left', 'FontSize', 7);
            end
        end
        xlim(ax, [0 1.13]);
        ylim(ax, [0.5 2.5]);
        xticks(ax, [0 0.5 1]);
        xticklabels(ax, {'0', '50', '100%'});
        yticks(ax, [1 2]);
        yticklabels(ax, {'Short', 'Long'});
        set(ax, 'XMinorTick', 'off', 'YMinorTick', 'off');
        title(ax, 'Outcome fraction, all trials', 'FontSize', 9, 'FontWeight', 'normal');
    end

    function colors = outcomeColors
        % Shared colors for all outcome categories.
        colors = [0.2 0.68 0.2; 0.15 0.4 0.8; 0.92 0.65 0.12; 0.75 0.2 0.2; 0.4 0.4 0.4; 0.55 0.25 0.65];
    end

    function labels = outcomeLabels
        labels = {'Reward','Press2 early','Press2 late','No press 2','No press 1','Assist'};
    end

    function colors = completedOutcomeColors
        % Completed summary excludes no-press outcomes.
        allColors = outcomeColors;
        colors = allColors([1 2 3 6], :);
    end

    function updateOutcomeLegend
        ax = BpodSystem.GUIHandles.OutcomeLegendAxes;
        cla(ax);
        hold(ax, 'on');
        axis(ax, 'off');

        labels = outcomeLabels;
        colors = outcomeColors;
        yValues = linspace(0.86, 0.12, numel(labels));
        for i = 1:numel(labels)
            plot(ax, 0.08, yValues(i), 's', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), 'MarkerSize', 4.8);
            text(ax, 0.18, yValues(i), labels{i}, 'FontSize', 6.5, 'VerticalAlignment', 'middle', 'Interpreter', 'none');
        end
        xlim(ax, [0 1]);
        ylim(ax, [0 1]);
        title(ax, 'Legend', 'FontSize', 8, 'FontWeight', 'normal');
    end

    function category = completedOutcomeCategory(trial)
        % Categories: reward, early, late, assist.
        category = 0;
        states = BpodSystem.Data.RawEvents.Trial{trial}.States;
        assisted = isfield(BpodSystem.Data, 'AssistTrial') && numel(BpodSystem.Data.AssistTrial) >= trial && BpodSystem.Data.AssistTrial(trial);
        if visited(states, 'LeverRetractFinal') || visited(states, 'Reward')
            if assisted
                category = 4;
            else
                category = 1;
            end
            return
        elseif visited(states, 'EarlyPress2')
            category = 2;
            return
        elseif visited(states, 'Press2Late')
            category = 3;
            return
        end
        pressTime = trialPress2Time(trial);
        if isfinite(pressTime)
            trialSettings = BpodSystem.Data.TrialSettings(trial);
            if BpodSystem.Data.TrialTypes(trial) == 1
                delay = trialSettings.GUI.ShortDelay_s;
            else
                delay = trialSettings.GUI.LongDelay_s;
            end
            difference = pressTime - delay;
            if difference < -trialSettings.GUI.RewardWindowLeft_s
                category = 2;
                return
            elseif difference > rewardMaximumWindow(trialSettings) + trialSettings.GUI.RewardWindowRight_s
                category = 3;
                return
            end
            if assisted
                category = 4;
            else
                category = 1;
            end
        end
    end

    function category = allOutcomeCategory(trial)
        % Categories: reward, early, late, no press 2, no press 1, assist.
        category = 0;
        states = BpodSystem.Data.RawEvents.Trial{trial}.States;
        assisted = isfield(BpodSystem.Data, 'AssistTrial') && numel(BpodSystem.Data.AssistTrial) >= trial && BpodSystem.Data.AssistTrial(trial);
        if visited(states, 'LeverRetractFinal') || visited(states, 'Reward')
            if assisted
                category = 6;
            else
                category = 1;
            end
        elseif visited(states, 'EarlyPress2')
            category = 2;
        elseif visited(states, 'Press2Late')
            category = 3;
        elseif visited(states, 'DidNotPress2')
            category = 4;
        elseif visited(states, 'DidNotPress1')
            category = 5;
        end
    end

    function updatePressTiming(trialType, count)
        % Show press 2 timing relative to perfect timing and reward profile.
        if trialType == 1
            ax = BpodSystem.GUIHandles.ShortTimingAxes;
            plotTitle = 'Short press 2 timing';
        else
            ax = BpodSystem.GUIHandles.LongTimingAxes;
            plotTitle = 'Long press 2 timing';
        end
        cla(ax);
        hold(ax, 'on');
        available = min([count availableTrialCount() completedTrialTypeCount()]);
        timingValues = nan(1, available);
        timingCategories = zeros(1, available);
        timingCount = 0;
        for trial = 1:available
            if BpodSystem.Data.TrialTypes(trial) ~= trialType
                continue
            end
            pressTime = trialPress2Time(trial);
            if isfinite(pressTime)
                trialSettings = S;
                if isfield(BpodSystem.Data, 'TrialSettings') && numel(BpodSystem.Data.TrialSettings) >= trial
                    trialSettings = BpodSystem.Data.TrialSettings(trial);
                end
                if trialType == 1
                    trialDelay = trialSettings.GUI.ShortDelay_s;
                else
                    trialDelay = trialSettings.GUI.LongDelay_s;
                end
                timing = pressTime - trialDelay;
                timingCount = timingCount + 1;
                timingValues(timingCount) = timing;
                if isfield(BpodSystem.Data, 'AssistTrial') && numel(BpodSystem.Data.AssistTrial) >= trial && BpodSystem.Data.AssistTrial(trial)
                    timingCategories(timingCount) = 4;
                elseif timing < -trialSettings.GUI.RewardWindowLeft_s
                    timingCategories(timingCount) = 2;
                elseif timing > rewardMaximumWindow(trialSettings) + trialSettings.GUI.RewardWindowRight_s
                    timingCategories(timingCount) = 3;
                else
                    timingCategories(timingCount) = 1;
                end
            end
        end
        timingValues = timingValues(1:timingCount);
        timingCategories = timingCategories(1:timingCount);
        rewardTimes = timingValues(timingCategories == 1);
        earlyTimes = timingValues(timingCategories == 2);
        lateTimes = timingValues(timingCategories == 3);
        assistTimes = timingValues(timingCategories == 4);
        timeLimits = [-1 4];
        edges = linspace(timeLimits(1), timeLimits(2), 101);
        rewardCounts = histcounts(rewardTimes, edges);
        earlyCounts = histcounts(earlyTimes, edges);
        lateCounts = histcounts(lateTimes, edges);
        assistCounts = histcounts(assistTimes, edges);
        centers = edges(1:end - 1) + diff(edges) / 2;
        colors = outcomeColors;
        countMatrix = [rewardCounts(:) earlyCounts(:) lateCounts(:) assistCounts(:)];
        timingBars = bar(ax, centers, countMatrix, 1, 'stacked', 'EdgeColor', 'none');
        timingColors = colors([1 2 3 6], :);
        for i = 1:numel(timingBars)
            timingBars(i).FaceColor = timingColors(i, :);
        end
        peak = max([1; sum(countMatrix, 2)]);
        plot(ax, [-S.GUI.RewardWindowLeft_s 0 S.GUI.RewardMaximumWindow_s S.GUI.RewardMaximumWindow_s + S.GUI.RewardWindowRight_s], [0 peak peak 0], '-', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.8);
        xline(ax, 0, ':', 'Perfect', 'Color', [0.15 0.15 0.15], 'LineWidth', 1.2, 'LabelVerticalAlignment', 'top', 'FontSize', 8);
        xlim(ax, timeLimits);
        ylim(ax, [0 peak * 1.15]);
        setTimeAxis(ax, timeLimits);
        yticks(ax, unique(round(linspace(0, peak, 3))));
        set(ax, 'YMinorTick', 'off');
        xlabel(ax, 'Time from perfect (s)');
        ylabel(ax, 'Count');
        title(ax, plotTitle, 'FontSize', 9, 'FontWeight', 'normal');
        if isempty(rewardTimes) && isempty(earlyTimes) && isempty(lateTimes) && isempty(assistTimes)
            text(ax, mean(timeLimits), peak * 0.55, 'No press 2 events', 'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5], 'FontSize', 8);
        end
    end

    function time = press2Time(states)
        candidates = stateStart(states, 'Press2');
        candidates = candidates(isfinite(candidates));
        if isempty(candidates)
            time = NaN;
        else
            time = min(candidates);
        end
    end

    function elapsed = trialPress2Time(trial)
        elapsed = NaN;
        if isfield(BpodSystem.Data, 'Press2Time') && numel(BpodSystem.Data.Press2Time) >= trial
            elapsed = BpodSystem.Data.Press2Time(trial);
        end
        if isfinite(elapsed) || ~trialAvailable(trial)
            return
        end
        states = BpodSystem.Data.RawEvents.Trial{trial}.States;
        phaseStart = stateStart(states, 'PrePress2Delay');
        if ~isfinite(phaseStart)
            phaseStart = stateStart(states, 'WaitForPress2');
        end
        pressStart = press2Time(states);
        if isfinite(phaseStart) && isfinite(pressStart)
            elapsed = pressStart - phaseStart;
        end
    end

    function updateDelay(types, count, settings)
        % Scatter the delay value used for each completed short or long trial.
        ax = BpodSystem.GUIHandles.DelayAxes;
        cla(ax);
        hold(ax, 'on');

        total = numel(types);
        [firstTrial, lastTrial] = visibleWindow(total, count);
        completed = firstTrial:min([count lastTrial total]);
        if isempty(completed)
            xlim(ax, [firstTrial - 0.5 lastTrial + 0.5]);
            ylim(ax, [0 1.5]);
            yticks(ax, [0 0.5 1 1.5]);
            yticklabels(ax, {'0','0.5','1.0','1.5'});
            xlabel(ax, 'Trial');
            ylabel(ax, 'Delay (s)');
            title(ax, 'Trial delay', 'FontSize', 10, 'FontWeight', 'normal');
            text(ax, mean(xlim(ax)), 0.75, 'No completed trials', 'HorizontalAlignment', 'center', 'Color', [0.55 0.55 0.55], 'FontSize', 9);
            return
        end

        delays = nan(size(completed));
        for i = 1:numel(completed)
            trial = completed(i);
            trialSettings = settings;
            if isfield(BpodSystem.Data, 'TrialSettings') && numel(BpodSystem.Data.TrialSettings) >= trial
                trialSettings = BpodSystem.Data.TrialSettings(trial);
            end
            if types(trial) == 1
                delays(i) = trialSettings.GUI.ShortDelay_s;
            else
                delays(i) = trialSettings.GUI.LongDelay_s;
            end
        end
        shortTrials = types(completed) == 1;
        longTrials = types(completed) == 2;
        plot(ax, completed(shortTrials), delays(shortTrials), '.', 'Color', [0.15 0.15 0.15], 'MarkerSize', 12, 'DisplayName', 'Short');
        plot(ax, completed(longTrials), delays(longTrials), '.', 'Color', [0.62 0.62 0.62], 'MarkerSize', 12, 'DisplayName', 'Long');
        if numel(completed) <= 45
            for i = 1:numel(completed)
                text(ax, completed(i), delays(i) + 0.06, sprintf('%.2g', delays(i)), 'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', [0.2 0.2 0.2]);
            end
        end
        xlim(ax, [firstTrial - 0.5 lastTrial + 0.5]);
        xticks(ax, trialTicks(firstTrial, lastTrial));
        ylim(ax, [0 1.5]);
        yticks(ax, [0 0.5 1 1.5]);
        yticklabels(ax, {'0','0.5','1.0','1.5'});
        xlabel(ax, 'Trial');
        ylabel(ax, 'Delay (s)');
        title(ax, 'Trial delay', 'FontSize', 10, 'FontWeight', 'normal');
        set(ax, 'YMinorTick', 'off');
    end

    function updateEncoder(data, rawTrial, trial, settings)
        % Draw the last trial encoder trace with key task events overlaid.
        ax = BpodSystem.GUIHandles.EncoderAxes;
        cla(ax);
        hold(ax, 'on');
        duration = trialDuration(trial);
        if ~isfield(data, 'Times') || ~isfield(data, 'Positions') || isempty(data.Times)
            showEmpty(ax, 'Rotary encoder', 'No encoder data');
            return
        end
        currentType = currentTrialType(trial);
        typeName = trialTypeName(currentType);
        positionLine = plot(ax, data.Times, data.Positions, '-', 'Color', [0 0 0], 'LineWidth', 2, 'DisplayName', [typeName ' position']);
        zeroLine = line(ax, [0 duration], [0 0], 'Color', [0.3 0.3 0.3], 'LineStyle', ':', 'LineWidth', 1.2, 'DisplayName', 'Zero');
        thresholdLine = line(ax, [0 duration], [settings.GUI.PressThreshold settings.GUI.PressThreshold], 'Color', [0.85 0.15 0.15], 'LineStyle', ':', 'LineWidth', 1.3, 'DisplayName', 'Press threshold');
        retractLine = line(ax, [0 duration], [settings.GUI.RetractThreshold settings.GUI.RetractThreshold], 'Color', [0.45 0.2 0.65], 'LineStyle', ':', 'LineWidth', 1.2, 'DisplayName', 'Retract threshold');
        eventStates = {'VisualStimulus1','WaitForPress1','Press1','LeverRetract1','PrePress2Delay','Assist','WaitForPress2','Press2','RewardLeverRetract','EarlyPress2','Press2Late','PreRewardDelay','Reward','PostRewardDelay','LeverRetractFinal','DidNotPress1','DidNotPress2','Punish_ITI','ITI'};
        eventLabels = {'Cue 1','Press 1 window','Press 1','Lever retract 1','Pre press 2 delay','Assist','Press 2 window','Press 2','Reward lever retract','Press 2 early','Press 2 late','Pre reward delay','Reward','Post reward delay','Reward final','No press 1','No press 2','Punish ITI','ITI'};
        eventColors = lines(numel(eventStates));
        labelTop = false;
        for i = 1:numel(eventStates)
            time = stateStart(rawTrial.States, eventStates{i});
            if isfinite(time)
                addEncoderEvent(ax, time, eventLabels{i}, eventColors(i, :), labelTop);
                labelTop = ~labelTop;
            end
        end
        if BpodSystem.Data.TrialTypes(trial) == 1
            cue2Delay = settings.GUI.ShortDelay_s;
        else
            cue2Delay = settings.GUI.LongDelay_s;
        end
        phaseStart = stateStart(rawTrial.States, 'PrePress2Delay');
        if ~isfinite(phaseStart)
            phaseStart = stateStart(rawTrial.States, 'WaitForPress2');
        end
        cue2Onset = phaseStart + cue2Delay;
        if isfinite(cue2Onset)
            if settings.GUI.TimingMode == 1
                addEncoderEvent(ax, cue2Onset, 'Cue 2 on / perfect', [0.2 0.55 0.85], true);
                addEncoderEvent(ax, cue2Onset + settings.GUI.VisualCueDuration_s, 'Cue 2 off', [0.2 0.55 0.85], false);
            else
                addEncoderEvent(ax, cue2Onset, 'Perfect timing', [0.2 0.55 0.85], true);
            end
        end
        yValues = data.Positions(isfinite(data.Positions));
        yLow = min([-1 yValues]);
        yHigh = max([settings.GUI.PressThreshold * 4 yValues 1]);
        if yHigh <= yLow
            yHigh = yLow + 1;
        end
        xlim(ax, [0 duration]);
        ylim(ax, [yLow yHigh]);
        setTimeAxis(ax, [0 duration]);
        yticks(ax, niceTicks([yLow yHigh]));
        xlabel(ax, 'Time (s)');
        ylabel(ax, 'Position (deg)');
        title(ax, sprintf('Rotary encoder  trial %d  %s', trial, typeName), 'FontSize', 10, 'FontWeight', 'normal');
        legend(ax, [positionLine zeroLine thresholdLine retractLine], {'Position','Zero','Press threshold','Retract threshold'}, 'Location', 'northeast', 'Orientation', 'vertical', 'Box', 'off', 'FontSize', 9);
    end

    function addEncoderEvent(ax, time, label, color, top)
        if top
            vertical = 'top';
        else
            vertical = 'bottom';
        end
        xline(ax, time, ':', label, 'Color', color, 'LineWidth', 1.25, 'LabelOrientation', 'aligned', 'LabelVerticalAlignment', vertical, 'FontSize', 9);
    end

    function updateStateTiming(rawTrial, trial)
        % Render raw Bpod state names as a compact timing raster.
        ax = BpodSystem.GUIHandles.StateAxes;
        cla(ax);
        hold(ax, 'on');
        names = fieldnames(rawTrial.States);
        entryCapacity = 0;
        for i = 1:numel(names)
            times = rawTrial.States.(names{i});
            if isempty(times) || all(isnan(times(:)))
                continue
            end
            if isvector(times)
                times = reshape(times, 1, []);
            end
            entryCapacity = entryCapacity + size(times, 1);
        end
        entries = cell(entryCapacity, 3);
        entryCount = 0;
        for i = 1:numel(names)
            times = rawTrial.States.(names{i});
            if isempty(times) || all(isnan(times(:)))
                continue
            end
            if isvector(times)
                times = reshape(times, 1, []);
            end
            label = names{i};
            for row = 1:size(times, 1)
                if size(times, 2) >= 2 && all(isfinite(times(row, 1:2)))
                    entryCount = entryCount + 1;
                    entries(entryCount, :) = {times(row, 1), times(row, 2), label};
                end
            end
        end
        entries = entries(1:entryCount, :);
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
        setTimeAxis(ax, [0 duration]);
        yticks(ax, 1:numel(labels));
        yticklabels(ax, labels);
        set(ax, 'YMinorTick', 'off', 'XGrid', 'on', 'YGrid', 'on', 'XMinorGrid', 'off', 'YMinorGrid', 'off', 'GridColor', [0.82 0.82 0.82], 'GridAlpha', 0.5);
        xlabel(ax, 'Time (s)');
        title(ax, sprintf('State timing  trial %d', trial), 'FontSize', 10, 'FontWeight', 'normal');
    end

    function updateCombinedEvents(rawTrial, trial)
        % Plot each digital signal as one row; filled bars mark logical 1.
        ax = BpodSystem.GUIHandles.EventAxes;
        cla(ax);
        hold(ax, 'on');

        duration = trialDuration(trial);
        rows = {'BNC 1','LED 1','Port 1 lick'};
        colors = [0.08 0.08 0.08; 0.42 0.42 0.42; 0.68 0.68 0.68];

        if isfield(rawTrial, 'Events')
            events = rawTrial.Events;
        else
            events = struct;
        end

        drawIntervals(ax, eventIntervals(events, 'BNC1High', 'BNC1Low', duration, 0), 1, colors(1, :));
        drawIntervals(ax, led1Intervals(rawTrial, trial, duration), 2, colors(2, :));
        drawIntervals(ax, eventIntervals(events, 'Port1In', 'Port1Out', duration, 0.02), 3, colors(3, :));

        xlim(ax, [0 duration]);
        ylim(ax, [0.5 numel(rows) + 0.5]);
        setTimeAxis(ax, [0 duration]);
        yticks(ax, 1:numel(rows));
        yticklabels(ax, rows);
        set(ax, 'YMinorTick', 'off', 'XGrid', 'on', 'XMinorGrid', 'off', 'GridColor', [0.82 0.82 0.82], 'GridAlpha', 0.5);
        xlabel(ax, 'Time (s)');
        ylabel(ax, 'Event');
        title(ax, sprintf('BNC, LED, and lick events  trial %d', trial), 'FontSize', 10, 'FontWeight', 'normal');
    end

    function showNoEvents(ax, plotTitle, trial, duration)
        % Keep all event rows visible before any events arrive.
        cla(ax);
        xlim(ax, [0 duration]);
        ylim(ax, [0.5 3.5]);
        setTimeAxis(ax, [0 duration]);
        yticks(ax, 1:3);
        yticklabels(ax, {'BNC 1','LED 1','Port 1 lick'});
        set(ax, 'YMinorTick', 'off', 'XGrid', 'on', 'XMinorGrid', 'off', 'GridColor', [0.82 0.82 0.82], 'GridAlpha', 0.5);
        xlabel(ax, 'Time (s)');
        title(ax, sprintf('%s  trial %d', plotTitle, trial), 'FontSize', 10, 'FontWeight', 'normal');
    end

    function showEmpty(ax, plotTitle, message)
        cla(ax);
        xlim(ax, [0 1]);
        ylim(ax, [0 1]);
        setTimeAxis(ax, [0 1]);
        yticks(ax, []);
        set(ax, 'YMinorTick', 'off');
        xlabel(ax, 'Time (s)');
        title(ax, plotTitle, 'FontSize', 10, 'FontWeight', 'normal');
        text(ax, 0.5, 0.5, message, 'HorizontalAlignment', 'center', 'Color', [0.55 0.55 0.55], 'FontSize', 9);
    end

    function ticks = niceTicks(limits)
        span = limits(2) - limits(1);
        if ~isfinite(span) || span <= 0
            ticks = limits(1);
            return
        end
        rawStep = span / 10;
        power = 10 ^ floor(log10(rawStep));
        scaled = rawStep / power;
        if scaled <= 1
            step = power;
        elseif scaled <= 2
            step = 2 * power;
        elseif scaled <= 5
            step = 5 * power;
        else
            step = 10 * power;
        end
        ticks = ceil(limits(1) / step) * step:step:floor(limits(2) / step) * step;
        if isempty(ticks)
            ticks = limits;
        end
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
            edgeGap = (candidate(1) - limits(1) + limits(2) - candidate(end)) / steps(i);
            score = abs(numel(candidate) - 5) + edgeGap;
            if score < bestScore
                bestScore = score;
                ticks = candidate;
            end
        end
        ticks(abs(ticks) < eps(max(1, max(abs(ticks))))) = 0;
    end

    function setTimeAxis(ax, limits)
        major = easyTimeTicks(limits);
        xticks(ax, major);
        set(ax, 'XMinorTick', 'on');
        if numel(major) < 2
            return
        end
        minor = zeros(1, 4 * (numel(major) - 1));
        index = 0;
        for i = 1:numel(major) - 1
            values = linspace(major(i), major(i + 1), 6);
            minor(index + 1:index + 4) = values(2:5);
            index = index + 4;
        end
        try
            ax.XAxis.MinorTickValues = minor;
        catch
        end
    end

    function name = subjectName
        name = 'UnknownSubject';
        if isfield(BpodSystem, 'Path')
            candidates = {'CurrentSubject', 'SubjectName', 'Subject'};
            for i = 1:numel(candidates)
                if isfield(BpodSystem.Path, candidates{i}) && ~isempty(BpodSystem.Path.(candidates{i}))
                    name = char(BpodSystem.Path.(candidates{i}));
                    return
                end
            end
        end
        if isfield(BpodSystem, 'Data') && isfield(BpodSystem.Data, 'SubjectName') && ~isempty(BpodSystem.Data.SubjectName)
            name = char(BpodSystem.Data.SubjectName);
        end
    end

    function ticks = trialTicks(firstTrial, lastTrial)
        if lastTrial <= firstTrial
            ticks = firstTrial;
            return
        end
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

    function [firstTrial, lastTrial] = visibleWindow(total, count)
        current = min(count + 1, max(1, total));
        firstTrial = max(1, current - 99);
        lastTrial = min(total, firstTrial + 99);
        firstTrial = max(1, lastTrial - 99);
    end

    function intervals = eventIntervals(events, onName, offName, duration, fallbackWidth)
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
            elseif fallbackWidth > 0
                stopTime = starts(i) + fallbackWidth;
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

    function intervals = led1Intervals(rawTrial, trial, duration)
        intervals = zeros(0, 2);
        optoType = trialOptoType(trial);
        if isempty(optoType) || ~any(optoType)
            return
        end

        settings = trialSettings(trial);
        if numel(optoType) >= 1 && optoType(1)
            onset = stateStart(rawTrial.States, 'VisualStimulus1');
            stopTime = stateEnd(rawTrial.States, 'VisualStimulus1');
            intervals = appendLedInterval(intervals, onset, stopTime, duration);
        end
        if numel(optoType) >= 2 && optoType(2)
            onset = stateStart(rawTrial.States, 'LeverRetract1');
            press2ServoBack = stateStart(rawTrial.States, 'RewardLeverRetract');
            if isfinite(onset)
                stopTime = onset + press2Window(trial, settings);
                if isfinite(press2ServoBack)
                    stopTime = min(stopTime, press2ServoBack);
                end
                intervals = appendLedInterval(intervals, onset, stopTime, duration);
            end
        end
        if numel(optoType) >= 3 && optoType(3)
            onset = stateStart(rawTrial.States, 'PostRewardDelay');
            stopTime = stateEnd(rawTrial.States, 'PostRewardDelay');
            intervals = appendLedInterval(intervals, onset, stopTime, duration);
        end
    end

    function intervals = appendLedInterval(intervals, startTime, stopTime, duration)
        if ~isfinite(startTime) || ~isfinite(stopTime)
            return
        end
        startTime = max(0, min(duration, startTime));
        stopTime = max(0, min(duration, stopTime));
        if stopTime > startTime
            intervals(end + 1, :) = [startTime stopTime];
        end
    end

    function optoType = trialOptoType(trial)
        optoType = zeros(3, 1);
        if isfield(BpodSystem.Data, 'OptoTrialTypes') && size(BpodSystem.Data.OptoTrialTypes, 2) >= trial
            optoType = BpodSystem.Data.OptoTrialTypes(:, trial);
        elseif isfield(BpodSystem.Data, 'PlannedOptoTrialTypes') && size(BpodSystem.Data.PlannedOptoTrialTypes, 2) >= trial
            optoType = BpodSystem.Data.PlannedOptoTrialTypes(:, trial);
        end
    end

    function settings = trialSettings(trial)
        settings = S;
        if isfield(BpodSystem.Data, 'TrialSettings') && numel(BpodSystem.Data.TrialSettings) >= trial
            settings = BpodSystem.Data.TrialSettings(trial);
        end
    end

    function window = press2Window(trial, settings)
        if isfield(BpodSystem.Data, 'TrialTypes') && numel(BpodSystem.Data.TrialTypes) >= trial && BpodSystem.Data.TrialTypes(trial) == 2
            window = settings.GUI.LongPress2Window_s;
        else
            window = settings.GUI.ShortPress2Window_s;
        end
    end

    function trialType = currentTrialType(trial)
        trialType = 1;
        if isfield(BpodSystem.Data, 'TrialTypes') && numel(BpodSystem.Data.TrialTypes) >= trial
            trialType = BpodSystem.Data.TrialTypes(trial);
        end
    end

    function name = trialTypeName(trialType)
        if trialType == 2
            name = 'Long';
        else
            name = 'Short';
        end
    end

    function times = eventTimes(events, name)
        if isfield(events, name)
            times = events.(name);
            times = sort(times(isfinite(times)));
        else
            times = [];
        end
    end

    function drawIntervals(ax, intervals, row, color)
        for i = 1:size(intervals, 1)
            rectangle('Parent', ax, ...
                'Position', [intervals(i, 1), row - 0.22, max(0.005, diff(intervals(i, :))), 0.44], ...
                'FaceColor', color, ...
                'EdgeColor', 'none');
        end
    end

    function time = stateStart(states, name)
        % Return the first start time for one state.
        time = NaN;
        if isfield(states, name)
            values = states.(name);
            if ~isempty(values) && isfinite(values(1))
                time = values(1);
            end
        end
    end

    function time = stateEnd(states, name)
        % Return the first end time for one state.
        time = NaN;
        if isfield(states, name)
            values = states.(name);
            if ~isempty(values) && numel(values) >= 2 && isfinite(values(2))
                time = values(2);
            end
        end
    end

    function name = trialOutcomeName(trial)
        % Return the plotted outcome marker name for one trial.
        name = '';
        if ~trialAvailable(trial)
            return
        end
        states = BpodSystem.Data.RawEvents.Trial{trial}.States;
        assisted = isfield(BpodSystem.Data, 'AssistTrial') && numel(BpodSystem.Data.AssistTrial) >= trial && BpodSystem.Data.AssistTrial(trial);
        if assisted && (visited(states, 'LeverRetractFinal') || visited(states, 'Reward'))
            name = 'AssistTrial';
            return
        end
        names = {'LeverRetractFinal','EarlyPress2','Press2Late','DidNotPress2','DidNotPress1'};
        name = '';
        for i = 1:numel(names)
            if visited(states, names{i})
                name = names{i};
                return
            end
        end
    end

    function value = rewardMaximumWindow(settings)
        value = 0;
        if isfield(settings.GUI, 'RewardMaximumWindow_s')
            value = settings.GUI.RewardMaximumWindow_s;
        end
    end

    function value = visited(states, name)
        value = false;
        if isfield(states, name)
            times = states.(name);
            value = ~isempty(times) && any(~isnan(times(:)));
        end
    end

    function count = availableTrialCount
        count = 0;
        if isfield(BpodSystem.Data, 'RawEvents') && isfield(BpodSystem.Data.RawEvents, 'Trial')
            count = numel(BpodSystem.Data.RawEvents.Trial);
        end
    end

    function count = completedTrialTypeCount
        count = 0;
        if isfield(BpodSystem.Data, 'TrialTypes')
            count = numel(BpodSystem.Data.TrialTypes);
        end
    end

    function value = trialAvailable(trial)
        value = trial >= 1 && availableTrialCount() >= trial && ~isempty(BpodSystem.Data.RawEvents.Trial{trial});
    end

    function duration = trialDuration(trial)
        duration = 1;
        if isfield(BpodSystem.Data, 'TrialStartTimestamp') && isfield(BpodSystem.Data, 'TrialEndTimestamp') && numel(BpodSystem.Data.TrialEndTimestamp) >= trial
            duration = max(eps, BpodSystem.Data.TrialEndTimestamp(trial) - BpodSystem.Data.TrialStartTimestamp(trial));
        end
    end
end
