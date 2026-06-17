function output = OptoControl(action, S, varargin)
% Generate opto trial tags and plotting metadata.
switch action
    case 'trials'
        % Draw opto trial types only when opto mode is enabled.
        trialTypes = [];
        if ~isempty(varargin)
            trialTypes = varargin{1};
        end

        nTrials = round(S.GUI.MaxTrials);
        output = zeros(1, nTrials);
        if S.GUI.OptoMode
            eligible = eligibleTagTrials(S, nTrials, trialTypes);
            nOpto = min(numel(eligible), max(0, round(S.GUI.OptoFraction * numel(eligible))));
            if nOpto > 0
                indices = eligible(randperm(numel(eligible), nOpto));
                output(indices) = randi(3, 1, nOpto);
            end
        end
    case 'actions'
        % Return state-machine timer specs and trigger actions for this opto type.
        optoType = varargin{1};
        delay = varargin{2};
        if ~ismember(optoType, 0:3)
            error('Opto trial type must be 0, 1, 2, or 3.')
        end
        output.Enabled = optoType > 0;
        output.TrialType = optoType;
        output.Timers = struct('TimerID', {}, 'Duration', {}, 'OnsetDelay', {}, 'Channel', {}, 'OnLevel', {}, 'OffLevel', {}, 'Loop', {}, 'LoopInterval', {});
        output.VisualCue1Actions = {};
        output.ServoBack1Actions = {};
        output.RewardDelayActions = {};

        if optoType == 1
            [timerSpec, valid] = timerSpecForDuration(S, 2, S.GUI.VisualCueDuration_s);
            if valid
                output.Timers(end + 1) = timerSpec;
                output.VisualCue1Actions = {'GlobalTimerTrig', '2'};
            end
        elseif optoType == 2
            [timerSpec, valid] = timerSpecForDuration(S, 3, delay);
            if valid
                output.Timers(end + 1) = timerSpec;
                output.ServoBack1Actions = {'GlobalTimerTrig', '3'};
            end
        elseif optoType == 3
            [timerSpec, valid] = timerSpecForDuration(S, 4, 2);
            if valid
                output.Timers(end + 1) = timerSpec;
                output.RewardDelayActions = {'GlobalTimerTrig', '4'};
            end
        end
    case 'waveform'
        % Provide high intervals for plotting the LED 1 waveform.
        duration = varargin{1};
        output = waveformIntervals(S, duration);
    case 'display'
        % Keep labels and colors centralized for plotting.
        output.Labels = {'Off', 'Type 1', 'Type 2', 'Type 3'};
        output.Colors = [0.65 0.65 0.65; 0.49 0.18 0.56; 0.93 0.69 0.13; 0.3 0.75 0.93];
    otherwise
        error('Unknown opto action: %s', action)
end
end

function [timerSpec, valid] = timerSpecForDuration(S, timerID, requestedDuration)
wave = waveformParameters(S, requestedDuration);
valid = wave.Cycles > 0;
timerSpec = struct( ...
    'TimerID', timerID, ...
    'Duration', wave.OnDuration, ...
    'OnsetDelay', 0, ...
    'Channel', 'PWM1', ...
    'OnLevel', 255, ...
    'OffLevel', 0, ...
    'Loop', wave.Cycles, ...
    'LoopInterval', wave.OffDuration);
end

function intervals = waveformIntervals(S, requestedDuration)
wave = waveformParameters(S, requestedDuration);
intervals = zeros(0, 2);
for cycle = 0:wave.Cycles - 1
    cycleStart = cycle * wave.Period;
    intervals(end + 1, :) = [cycleStart cycleStart + wave.OnDuration];
end
end

function wave = waveformParameters(S, requestedDuration)
period = 1 / S.GUI.OptoFrequency_Hz;
onDuration = S.GUI.OptoPulseOn_ms / 1000;
offDuration = period - onDuration;
cycles = floor(requestedDuration / period);
wave = struct( ...
    'Period', period, ...
    'OnDuration', onDuration, ...
    'OffDuration', offDuration, ...
    'Cycles', max(0, cycles));
end

function eligible = eligibleTagTrials(S, nTrials, trialTypes)
edge = max(0, round(S.GUI.OptoZeroEdgeTrials));
blocked = false(1, nTrials);

if ~isempty(trialTypes)
    blockStarts = [1 find(diff(trialTypes) ~= 0) + 1];
    blockEnds = [blockStarts(2:end) - 1 nTrials];
    for i = 1:numel(blockStarts)
        firstEdge = blockStarts(i):min(blockEnds(i), blockStarts(i) + edge - 1);
        lastEdge = max(blockStarts(i), blockEnds(i) - edge + 1):blockEnds(i);
        blocked(firstEdge) = true;
        blocked(lastEdge) = true;
    end
    blocked(blockStarts(1):blockEnds(1)) = true;
else
    firstBlockEnd = min(nTrials, round(S.GUI.BlockLength + S.GUI.BlockLengthEdge));
    blocked(1:firstBlockEnd) = true;
end

eligible = find(~blocked);
end
